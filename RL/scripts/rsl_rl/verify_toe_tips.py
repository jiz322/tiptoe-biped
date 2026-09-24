"""
Toe tip verification utility.

Two verification methods:
  1. NUMERIC: Compare _get_toe_pos_w() against the FK chain already in rl_env.py
     (which computes the same nail transforms). Run once at standing pose.
  2. VISUAL: Draw debug spheres at the computed toe tip positions every step.
     If the spheres sit on the nail tips in the viewer, the offsets are correct.

Usage in play.py — add after env is created:

    from verify_toe_tips import verify_toe_numeric, ToeDebugMarkers

    # One-time numeric check (prints to console)
    verify_toe_numeric(env.unwrapped)

    # Visual markers (call every step)
    toe_markers = ToeDebugMarkers(env.unwrapped)
    # ... in main loop:
    toe_markers.update()
"""

import torch
import carb
from isaaclab.utils.math import quat_apply, quat_rotate_inverse


def verify_toe_numeric(uw, env_id: int = 0):
    """Compare toe positions from _get_toe_pos_w() vs raw body_pos_w.

    Prints the difference so you can see how far the toe tip is
    from the shin link origin. If offsets are zero, the two would match
    (which would mean the fix isn't working).
    """
    print(f"\n{'=' * 60}")
    print(f"  TOE TIP VERIFICATION (env {env_id})")
    print(f"{'=' * 60}")

    # Method 1: raw link origins (what we used BEFORE)
    old_foot1 = uw.robot.data.body_pos_w[env_id, uw._foot1_idx, :].clone()
    old_foot2 = uw.robot.data.body_pos_w[env_id, uw._foot2_idx, :].clone()

    # Method 2: toe tips with local offsets (what we use NOW)
    toe1_w, toe2_w = uw._get_toe_pos_w()
    new_toe1 = toe1_w[env_id].clone()
    new_toe2 = toe2_w[env_id].clone()

    diff1 = (new_toe1 - old_foot1)
    diff2 = (new_toe2 - old_foot2)

    print(f"\n  Leg 1 (shin_inner → nail tip):")
    print(f"    Link origin (old): [{old_foot1[0]:.5f}, {old_foot1[1]:.5f}, {old_foot1[2]:.5f}]")
    print(f"    Toe tip    (new):  [{new_toe1[0]:.5f}, {new_toe1[1]:.5f}, {new_toe1[2]:.5f}]")
    print(f"    Offset applied:    [{diff1[0]:.5f}, {diff1[1]:.5f}, {diff1[2]:.5f}]")
    print(f"    Distance:          {diff1.norm():.5f} m  ({diff1.norm() * 100:.2f} cm)")

    print(f"\n  Leg 2 (nail___fillet1 → nail tip):")
    print(f"    Link origin (old): [{old_foot2[0]:.5f}, {old_foot2[1]:.5f}, {old_foot2[2]:.5f}]")
    print(f"    Toe tip    (new):  [{new_toe2[0]:.5f}, {new_toe2[1]:.5f}, {new_toe2[2]:.5f}]")
    print(f"    Offset applied:    [{diff2[0]:.5f}, {diff2[1]:.5f}, {diff2[2]:.5f}]")
    print(f"    Distance:          {diff2.norm():.5f} m  ({diff2.norm() * 100:.2f} cm)")

    # Sanity checks
    print(f"\n  SANITY CHECKS:")

    if diff1.norm() < 1e-4:
        print(f"  ❌ Leg 1 offset is ~zero — toe tip transform NOT applied!")
    else:
        print(f"  ✅ Leg 1 offset is {diff1.norm() * 100:.1f} cm — transform is applied")

    if diff2.norm() < 1e-4:
        print(f"  ❌ Leg 2 offset is ~zero — toe tip transform NOT applied!")
    else:
        print(f"  ✅ Leg 2 offset is {diff2.norm() * 100:.1f} cm — transform is applied")

    # Check that toe tips are lower than link origins (they should be near ground)
    if new_toe1[2] < old_foot1[2]:
        print(f"  ✅ Leg 1 toe is BELOW link origin (Δz = {(old_foot1[2] - new_toe1[2]) * 100:.1f} cm)")
    else:
        print(f"  ⚠️  Leg 1 toe is ABOVE link origin — double check offset sign")

    if new_toe2[2] < old_foot2[2]:
        print(f"  ✅ Leg 2 toe is BELOW link origin (Δz = {(old_foot2[2] - new_toe2[2]) * 100:.1f} cm)")
    else:
        print(f"  ⚠️  Leg 2 toe is ABOVE link origin — double check offset sign")

    # Cross-check: FK chain from rl_env.py should give matching z values
    # The FK _compute_fk_height uses the same nail transforms in its T_nail matrices
    root_quat = uw.robot.data.root_quat_w[env_id:env_id+1]
    joint_pos = uw.robot.data.joint_pos[env_id, uw._joint_ids]

    from rl_env import _compute_fk_height
    fk_height = _compute_fk_height(
        base_quat=root_quat,
        hip_roll1=joint_pos[0:1], hip_pitch1=joint_pos[1:2], knee1=joint_pos[2:3],
        hip_roll2=joint_pos[3:4], hip_pitch2=joint_pos[4:5], knee2=joint_pos[5:6],
        device=uw.device,
    )
    print(f"\n  FK chain predicted standing height: {fk_height[0]:.5f} m")
    print(f"  Actual root height:                {uw.robot.data.root_pos_w[env_id, 2]:.5f} m")

    print(f"{'=' * 60}\n")


class ToeDebugMarkers:
    """Draw debug spheres at toe tip positions every step.

    Red sphere = leg 1 toe tip
    Blue sphere = leg 2 toe tip
    Green sphere = feet midpoint (used for COM tracking)

    If the red/blue spheres sit on the nail tips visually,
    the local offsets are correct.
    """

    def __init__(self, uw, radius: float = 0.008, env_id: int = 0):
        self.uw = uw
        self.env_id = env_id
        self.radius = radius
        self._setup_markers()

    def _setup_markers(self):
        """Create debug visualization spheres using Isaac Sim's debug draw."""
        try:
            from omni.isaac.debug_draw import _debug_draw
            self.draw = _debug_draw.acquire_debug_draw_interface()
            self._available = True
            print("[TOE VERIFY] Debug draw markers ENABLED — look for red/blue/green spheres")
        except (ImportError, Exception):
            try:
                import omni.debugdraw
                self.draw = omni.debugdraw.get_debug_draw_interface()
                self._available = True
                print("[TOE VERIFY] Debug draw markers ENABLED (omni.debugdraw)")
            except (ImportError, Exception):
                self._available = False
                print("[TOE VERIFY] Debug draw not available — using console output only")
                print("             Install omni.isaac.debug_draw for visual markers")

    def update(self):
        """Update marker positions. Call once per step."""
        toe1_w, toe2_w = self.uw._get_toe_pos_w()
        t1 = toe1_w[self.env_id].cpu().tolist()
        t2 = toe2_w[self.env_id].cpu().tolist()
        mid = [(t1[i] + t2[i]) * 0.5 for i in range(3)]

        if self._available:
            try:
                self.draw.clear_lines()
            except Exception:
                pass
            s = self.radius
            # Colors as ARGB hex ints
            RED = 0xFFFF0000
            BLUE = 0xFF0000FF
            GREEN = 0xFF00FF00
            # Draw small cross-hairs (3 short lines per point)
            # Red = toe 1
            for axis in range(3):
                p0 = [t1[0], t1[1], t1[2]]
                p1 = [t1[0], t1[1], t1[2]]
                p0[axis] -= s
                p1[axis] += s
                self.draw.draw_line(carb.Float3(*p0), RED, carb.Float3(*p1), RED)
            # Blue = toe 2
            for axis in range(3):
                p0 = [t2[0], t2[1], t2[2]]
                p1 = [t2[0], t2[1], t2[2]]
                p0[axis] -= s
                p1[axis] += s
                self.draw.draw_line(carb.Float3(*p0), BLUE, carb.Float3(*p1), BLUE)
            # Green = midpoint
            for axis in range(3):
                p0 = [mid[0], mid[1], mid[2]]
                p1 = [mid[0], mid[1], mid[2]]
                p0[axis] -= s
                p1[axis] += s
                self.draw.draw_line(carb.Float3(*p0), GREEN, carb.Float3(*p1), GREEN)

    def print_positions(self):
        """Print current positions to console."""
        toe1_w, toe2_w = self.uw._get_toe_pos_w()
        t1 = toe1_w[self.env_id]
        t2 = toe2_w[self.env_id]
        mid = (t1 + t2) * 0.5
        print(f"  Toe1: [{t1[0]:.4f}, {t1[1]:.4f}, {t1[2]:.4f}]  "
              f"Toe2: [{t2[0]:.4f}, {t2[1]:.4f}, {t2[2]:.4f}]  "
              f"Mid: [{mid[0]:.4f}, {mid[1]:.4f}, {mid[2]:.4f}]")