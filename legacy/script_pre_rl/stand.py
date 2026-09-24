"""
bipedal_com_balance.py — Run inside Isaac Sim's Script Editor

COM-based balance controller for a point-foot bipedal robot.

WHY THIS REPLACES THE PITCH-TO-ZERO CONTROLLER:
    The old controller drove torso pitch to zero, assuming upright = balanced.
    That's only true when the body and contact point are rigidly coupled (like
    a balanced wheel). For a multi-link biped, balance means:

        COM projected onto ground is directly above the foot contact centroid.

    A human can lean forward at the hip, bend knees back, and still be perfectly
    balanced — torso pitch is nonzero but COM is over the feet.

CONTROL ARCHITECTURE (three decoupled loops):
    1. BALANCE (highest priority):
       COM_x error (forward/back) → coordinated hip_pitch + knee correction
       COM_y error (lateral)      → hip_roll correction

    2. HEIGHT:
       height error → knee correction

    3. POSTURE (lowest priority, optional):
       torso pitch error → hip_pitch correction (small gain)

    Each loop outputs joint corrections that are summed with the default pose.

JOINT MAPPING (sagittal plane):
    - Hip pitch shifts upper-body mass forward/backward relative to feet
    - Knee bend changes height AND shifts feet relative to COM
    - They CANNOT be rigidly coupled: sometimes you need knee bend without
      torso pitch (lower height) or torso pitch without knee bend (shift COM)

COM COMPUTATION:
    Every physics step, we read world transforms of all links, apply the
    local COM offset from the URDF inertial data, and compute the weighted
    center of mass. This gives us the true COM position in world frame.

Usage:
    1. Paste into Script Editor (Window > Script Editor)
    2. Press Run
    3. Press PLAY in the toolbar
    4. Watch debug output — tune gains as needed
"""

import numpy as np
import math
import asyncio
from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema, Sdf, PhysicsSchemaTools
import omni.usd
import omni.kit.app
import omni.kit.commands
import omni.timeline
import omni.physx

# ============================================================
# USER CONFIG
# ============================================================

USD_PATH = r"<repo>\urdf_onshape_export\bigthigh-urdf\robot\robot.usd"
SPAWN_HEIGHT = 0.42
TORSO_PRIM_PATH = "/World/Robot/torso2hip"

# Default standing pose — the static balance point
DEFAULT_POSE = {
    "hip_roll1":  0.0,
    "hip_pitch1": 0.05,
    "knee1":      0.0,
    "hip_roll2":  0.0,
    "hip_pitch2": -0.05,
    "knee2":      0.0,
}

# Motor parameters — stiffer hips for tight tracking
# (the "foot sliding" we saw earlier was a measurement bug, not real)
JOINT_PARAMS = {
    "hip_roll1":  {"stiffness": 5.0,   "damping": 0.3, "effort": 6.0},
    "hip_roll2":  {"stiffness": 5.0,   "damping": 0.3, "effort": 6.0},
    "hip_pitch1": {"stiffness": 60.0,  "damping": 2.0, "effort": 17.0},
    "hip_pitch2": {"stiffness": 60.0,  "damping": 2.0, "effort": 17.0},
    "knee1":      {"stiffness": 50.0,  "damping": 1.5, "effort": 23.2},
    "knee2":      {"stiffness": 50.0,  "damping": 1.5, "effort": 23.2},
}

# ============================================================
# LINK MASS TABLE (from URDF inertial data)
# Maps link name → (mass, local_com_xyz)
# local_com_xyz is the COM offset in the link's local frame
# ============================================================

LINK_MASS_TABLE = {
    "torso2hip": {
        "mass": 1.698,
        "local_com": (-0.0110578, 2.08706e-05, -0.0316562),
    },
    "chasis_motor2motor": {
        "mass": 1.1141,
        "local_com": (0.0, 0.0, -0.0623004),
    },
    "thigh_outer": {
        "mass": 0.434,
        "local_com": (-0.0223297, -0.118239, 0.0709555),
    },
    "shin_inner": {
        "mass": 0.1905,
        "local_com": (-0.0782005, 0.0113971, -0.0185949),
    },
    # Leg 2 (mirrored names from URDF)
    "lid_bearing_outer___boss_extrude3": {
        "mass": 1.1141,
        "local_com": (0.0, 0.0, -0.0623004),
    },
    "stand_steel_00___fillet1": {
        "mass": 0.434,
        "local_com": (-0.0903635, 0.0795028, 0.0709556),
    },
    "nail___fillet1": {
        "mass": 0.1905,
        "local_com": (0.000800703, 0.0790291, 0.0181051),
    },
}

# Prim paths for each link (under /World/Robot/)
# These must match how the URDF importer names prims in the stage.
# Adjust if your importer uses different naming.
LINK_PRIM_PATHS = {
    "torso2hip":                          "/World/Robot/torso2hip",
    "chasis_motor2motor":                 "/World/Robot/chasis_motor2motor",
    "thigh_outer":                        "/World/Robot/thigh_outer",
    "shin_inner":                         "/World/Robot/shin_inner",
    "lid_bearing_outer___boss_extrude3":  "/World/Robot/lid_bearing_outer___boss_extrude3",
    "stand_steel_00___fillet1":           "/World/Robot/stand_steel_00___fillet1",
    "nail___fillet1":                     "/World/Robot/nail___fillet1",
}

# Foot link names — the shin/foot links whose positions define the support polygon
FOOT_LINKS = ["shin_inner", "nail___fillet1"]

# ============================================================
# BALANCE LOOP GAINS
# ============================================================

# Balance: COM_x error → hip_pitch + knee correction
# NOW USING RAW ERROR (was filtered, seeing 50% of true error)
# NOW USING WINDOWED RATE (was per-step, getting ~2% of true velocity)
# Both signals are much stronger, so gains come way down.
BAL_X_KP = 5.0       # was 12.0 — P term now sees full raw error (2x more)
BAL_X_KD = 3.0       # was 8.0 — D term now sees real velocity (~40x more)
BAL_X_TO_HIP = 1.0
BAL_X_TO_KNEE = 0.0

# Balance: COM_y error → hip_roll correction
BAL_Y_KP = 1.5
BAL_Y_KD = 0.08

# Height: torso height error → knee correction
# DISABLED until balance loop is tuned — was fighting balance
HEIGHT_KP = 0.0
HEIGHT_KD = 0.0
DESIRED_HEIGHT = 0.28  # meters — target torso height above ground

# Posture: torso pitch error → hip_pitch correction (low priority)
# DISABLED until balance loop is tuned — was canceling balance corrections
POSTURE_KP = 0.0
POSTURE_KD = 0.0
DESIRED_PITCH = 0.0   # radians — desired torso pitch (0 = upright)

# ── Direction signs ──────────────────────────────────────────
# These map the sign of each error to the correct joint direction.
# If the robot falls FASTER with control on, flip these.
#
# Convention: positive COM_x error means COM is ahead of feet
#   → need to move COM backward → lean torso back → hip pitch correction < 0
#
# Tune empirically — depends on joint axis conventions in your URDF.
BAL_X_HIP_SIGN  = -1   # CONFIRMED: increasing hp1 pushes COM +x, so positive error → negative correction
BAL_X_KNEE_SIGN = 1    # oppose hip
BAL_Y_SIGN      = -1   # positive COM_y error → which way hip roll moves
HEIGHT_KNEE_SIGN = -1  # positive height error (too low) → which way knee moves
POSTURE_HIP_SIGN = -1  # positive pitch error → which way hip pitch moves

# ── Joint sign mapping ──────────────────────────────────────
# How each correction type maps to individual joints.
# Sign indicates which direction each joint moves for a positive correction.
HIP_PITCH_JOINT_SIGNS = {"hip_pitch1": 1, "hip_pitch2": -1}
KNEE_JOINT_SIGNS      = {"knee1": 1, "knee2": -1}  # might need to be same sign
HIP_ROLL_JOINT_SIGNS  = {"hip_roll1": 1, "hip_roll2": -1}

# ── Clamps ───────────────────────────────────────────────────
MAX_BALANCE_CORRECTION = 0.8   # was 0.5 — higher gains need room
MAX_HEIGHT_CORRECTION  = 0.4   # rad — max knee adjustment from height loop
MAX_POSTURE_CORRECTION = 0.2   # rad — max hip adjustment from posture loop
MAX_ROLL_CORRECTION    = 0.5   # rad

# ── (filters removed — controller now uses raw error + windowed rate) ──

# ── Safety ───────────────────────────────────────────────────
MAX_TILT_DEG    = 45.0
MIN_BASE_HEIGHT = 0.10

# ── Debug ────────────────────────────────────────────────────
PRINT_EVERY_N = 10   # every 10 steps = 0.05 seconds at 200Hz


# ============================================================
# LOW-PASS FILTER
# ============================================================

class LowPassFilter:
    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self.value = 0.0
        self.initialized = False

    def update(self, raw):
        if not self.initialized:
            self.value = raw
            self.initialized = True
        else:
            self.value = self.alpha * raw + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.initialized = False
        self.value = 0.0


# ============================================================
# COM CALCULATOR
# ============================================================

class COMCalculator:
    """
    Computes the whole-body center of mass from link world transforms.

    For each link:
        1. Get world transform matrix from USD
        2. Transform the link's local COM offset to world frame
        3. Accumulate mass-weighted position

    Returns COM in world frame (x, y, z).
    """

    def __init__(self, stage):
        self.stage = stage
        self.link_data = []  # [(prim, mass, local_com_vec), ...]
        self.total_mass = 0.0
        self._resolve_prims()

    def _resolve_prims(self):
        """Find all link prims and cache them with their mass data."""
        self.link_data = []
        self.total_mass = 0.0

        for link_name, info in LINK_MASS_TABLE.items():
            prim_path = LINK_PRIM_PATHS.get(link_name)
            if prim_path is None:
                print(f"  [COM] WARNING: no prim path for link '{link_name}'")
                continue

            prim = self.stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                print(f"  [COM] WARNING: prim not found at '{prim_path}'")
                continue

            mass = info["mass"]
            lc = info["local_com"]
            local_com_vec = Gf.Vec4d(lc[0], lc[1], lc[2], 1.0)  # homogeneous

            self.link_data.append((prim, mass, local_com_vec))
            self.total_mass += mass

        print(f"  [COM] Resolved {len(self.link_data)} links, "
              f"total mass = {self.total_mass:.3f} kg")

    def compute(self):
        """
        Compute whole-body COM in world frame.
        Returns (com_x, com_y, com_z) or None if computation fails.
        """
        if self.total_mass < 1e-6:
            return None

        com_sum = np.zeros(3)

        for prim, mass, local_com_vec in self.link_data:
            xformable = UsdGeom.Xformable(prim)
            try:
                world_mat = xformable.ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                )
            except RuntimeError:
                return None

            # Transform local COM to world frame
            world_com = world_mat.Transform(Gf.Vec3d(
                local_com_vec[0], local_com_vec[1], local_com_vec[2]
            ))

            com_sum[0] += mass * world_com[0]
            com_sum[1] += mass * world_com[1]
            com_sum[2] += mass * world_com[2]

        com = com_sum / self.total_mass
        return com


# ============================================================
# FOOT CONTACT ESTIMATOR
# ============================================================

class FootContactEstimator:
    """
    Estimates the support point — ground contact position of the feet.

    CRITICAL INSIGHT: For a standing robot, the actual ground contact point
    stays FIXED even as joints articulate. The shin link's world-frame origin
    moves when hip/knee flex, but the foot-ground contact doesn't (assuming
    no sliding).

    Strategy: Read foot link positions during the first few physics steps
    (while robot is settling into standing pose), then FREEZE those as the
    fixed contact reference. The live shin link position is NOT the contact
    point once joints start moving.
    """

    def __init__(self, stage):
        self.stage = stage
        self.foot_prims = []
        self.contact_xy = None       # fixed contact centroid (x, y)
        self.contact_captured = False
        self.capture_countdown = 10  # wait N steps for robot to settle
        self._resolve_prims()

    def _resolve_prims(self):
        self.foot_prims = []
        for link_name in FOOT_LINKS:
            prim_path = LINK_PRIM_PATHS.get(link_name)
            if prim_path is None:
                continue
            prim = self.stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                self.foot_prims.append(prim)

        print(f"  [Feet] Resolved {len(self.foot_prims)} foot links")

    def reset(self):
        """Reset for new simulation run."""
        self.contact_xy = None
        self.contact_captured = False
        self.capture_countdown = 10

    def _read_current_positions(self):
        """Read live foot link positions (used only for initial capture)."""
        if not self.foot_prims:
            return None, None

        positions = []
        for prim in self.foot_prims:
            xformable = UsdGeom.Xformable(prim)
            try:
                world_mat = xformable.ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                )
            except RuntimeError:
                return None, None

            pos = world_mat.ExtractTranslation()
            positions.append(np.array([pos[0], pos[1], pos[2]]))

        positions = np.array(positions)
        centroid_xy = np.mean(positions[:, :2], axis=0)
        avg_foot_z  = np.mean(positions[:, 2])

        return centroid_xy, avg_foot_z

    def compute_centroid(self):
        """
        Returns the ground contact centroid (x, y) and foot height.

        For the first N steps, reads live positions to let the robot settle.
        Then captures and freezes the contact point.
        """
        if not self.contact_captured:
            centroid_xy, avg_foot_z = self._read_current_positions()
            if centroid_xy is None:
                return None, None

            self.capture_countdown -= 1
            if self.capture_countdown <= 0:
                self.contact_xy = centroid_xy.copy()
                self.contact_captured = True
                print(f"  [Feet] Contact captured at ({centroid_xy[0]*1000:.1f}, "
                      f"{centroid_xy[1]*1000:.1f})mm — FROZEN as reference")

            return centroid_xy, avg_foot_z

        # After capture: return fixed contact position
        return self.contact_xy, 0.0


# ============================================================
# BALANCE CONTROLLER (COM-BASED)
# ============================================================

class COMBalanceController:
    """
    COM-based balance controller with multi-step rate estimation.

    KEY FIX: USD transforms update at render rate (~60Hz), not physics rate
    (200Hz). So ~2 out of 3 physics steps see identical COM values.
    Per-step differentiation gives alternating zeros and spikes, which the
    filter averages to near-zero — killing the D term.

    Fix: differentiate over a window of N steps (~6-8) to guarantee at least
    one render update is captured. This gives a clean average velocity.
    Also: use RAW error for P term. The position filter was lagging by 50%+
    of the true error.
    """

    RATE_WINDOW = 8  # steps — at 200Hz physics / 60Hz render, this spans ~2.5 render frames

    def __init__(self):
        # Ring buffers for multi-step differentiation
        self.err_x_history = []
        self.err_y_history = []
        self.height_history = []
        self.pitch_history = []

        self.step = 0

    def reset(self):
        self.err_x_history.clear()
        self.err_y_history.clear()
        self.height_history.clear()
        self.pitch_history.clear()
        self.step = 0

    def _windowed_rate(self, history, current, dt):
        """
        Compute rate from ring buffer using oldest available sample.
        Returns rate in units/second.
        """
        history.append(current)
        if len(history) > self.RATE_WINDOW:
            history.pop(0)

        if len(history) < 2:
            return 0.0

        # Use the full window for differentiation
        oldest = history[0]
        newest = history[-1]
        window_dt = (len(history) - 1) * dt

        if window_dt < 1e-9:
            return 0.0

        return (newest - oldest) / window_dt

    def compute(self, com_xyz, foot_centroid_xy, torso_height, torso_pitch, dt):
        """
        Compute joint corrections from COM state.
        Uses RAW error for P term, windowed differentiation for D term.
        """
        # ── Compute errors (RAW, no filter) ───────────────────
        com_error_x = com_xyz[0] - foot_centroid_xy[0]
        com_error_y = com_xyz[1] - foot_centroid_xy[1]
        height_error = DESIRED_HEIGHT - torso_height
        pitch_error = DESIRED_PITCH - torso_pitch

        # ── Rates via windowed differentiation ────────────────
        # Spans ~8 physics steps = ~2.5 render frames
        # Guarantees at least 1 transform update in the window
        com_x_rate  = self._windowed_rate(self.err_x_history, com_error_x, dt)
        com_y_rate  = self._windowed_rate(self.err_y_history, com_error_y, dt)
        height_rate = self._windowed_rate(self.height_history, height_error, dt)
        pitch_rate  = self._windowed_rate(self.pitch_history, pitch_error, dt)

        # ── PD control uses RAW error + windowed rate ─────────
        # No position filter — it was lagging 50%+ behind reality

        # ════════════════════════════════════════════════════════
        # LOOP 1: BALANCE (highest priority)
        # ════════════════════════════════════════════════════════
        bal_x_raw = BAL_X_KP * com_error_x + BAL_X_KD * com_x_rate
        bal_x_raw = np.clip(bal_x_raw, -MAX_BALANCE_CORRECTION, MAX_BALANCE_CORRECTION)

        bal_hip_correction  = BAL_X_HIP_SIGN  * BAL_X_TO_HIP  * bal_x_raw
        bal_knee_correction = BAL_X_KNEE_SIGN * BAL_X_TO_KNEE * bal_x_raw

        bal_y_raw = BAL_Y_KP * com_error_y + BAL_Y_KD * com_y_rate
        bal_y_raw = np.clip(bal_y_raw, -MAX_ROLL_CORRECTION, MAX_ROLL_CORRECTION)
        roll_correction = BAL_Y_SIGN * bal_y_raw

        # ════════════════════════════════════════════════════════
        # LOOP 2: HEIGHT
        # ════════════════════════════════════════════════════════
        height_raw = HEIGHT_KP * height_error + HEIGHT_KD * height_rate
        height_raw = np.clip(height_raw, -MAX_HEIGHT_CORRECTION, MAX_HEIGHT_CORRECTION)
        height_knee_correction = HEIGHT_KNEE_SIGN * height_raw

        # ════════════════════════════════════════════════════════
        # LOOP 3: POSTURE (lowest priority)
        # ════════════════════════════════════════════════════════
        posture_raw = POSTURE_KP * pitch_error + POSTURE_KD * pitch_rate
        posture_raw = np.clip(posture_raw, -MAX_POSTURE_CORRECTION, MAX_POSTURE_CORRECTION)
        posture_hip_correction = POSTURE_HIP_SIGN * posture_raw

        # ════════════════════════════════════════════════════════
        # SUM CORRECTIONS → JOINT TARGETS
        # ════════════════════════════════════════════════════════
        total_hip_pitch = bal_hip_correction + posture_hip_correction
        total_knee = bal_knee_correction + height_knee_correction

        targets = {}
        for name, default in DEFAULT_POSE.items():
            target = default

            if name in HIP_PITCH_JOINT_SIGNS:
                target += HIP_PITCH_JOINT_SIGNS[name] * total_hip_pitch

            if name in KNEE_JOINT_SIGNS:
                target += KNEE_JOINT_SIGNS[name] * total_knee

            if name in HIP_ROLL_JOINT_SIGNS:
                target += HIP_ROLL_JOINT_SIGNS[name] * roll_correction

            targets[name] = target

        self.step += 1

        debug = {
            "com_err_x":    com_error_x * 1000,      # mm (RAW, not filtered)
            "com_err_y":    com_error_y * 1000,      # mm
            "com_rate_x":   com_x_rate,              # m/s (windowed)
            "height_err":   height_error * 1000,      # mm
            "pitch_err":    math.degrees(pitch_error),
            "bal_hip":      bal_hip_correction,
            "bal_knee":     bal_knee_correction,
            "height_knee":  height_knee_correction,
            "posture_hip":  posture_hip_correction,
            "roll_corr":    roll_correction,
            "torso_h":      torso_height,
            "com_z":        com_xyz[2],
        }

        return targets, debug


# ============================================================
# GLOBALS
# ============================================================

_joint_prims     = {}
_controller      = COMBalanceController()
_com_calc        = None
_foot_estimator  = None
_was_playing     = False
_physx_sub       = None
_timeline_sub    = None


# ============================================================
# JOINT HELPER
# ============================================================

def set_joint_target(name, target_rad):
    """Update a single joint's drive target position (rad → deg for PhysX)."""
    if name in _joint_prims:
        prim = _joint_prims[name]
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if drive:
            drive.GetTargetPositionAttr().Set(np.degrees(target_rad))


def get_joint_position(name):
    """Read actual joint position from physics state (deg → rad)."""
    if name not in _joint_prims:
        return None
    prim = _joint_prims[name]
    # Try reading from the joint state
    attr = prim.GetAttribute("state:angular:physics:position")
    if attr and attr.HasValue():
        return math.radians(attr.Get())
    # Fallback: try PhysX joint state
    attr = prim.GetAttribute("physxJoint:jointPosition")
    if attr and attr.HasValue():
        val = attr.Get()
        if val is not None:
            return math.radians(val)
    return None


# ============================================================
# TORSO TILT (for safety check and posture loop)
# ============================================================

def get_torso_state(stage):
    """
    Get torso pitch, height from world transform.
    Returns (pitch_rad, height_m) or (None, None).
    """
    prim = stage.GetPrimAtPath(TORSO_PRIM_PATH)
    if not prim or not prim.IsValid():
        return None, None, None

    xform = UsdGeom.Xformable(prim)
    try:
        world_mat = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    except RuntimeError:
        return None, None, None

    # Body Z-axis in world frame
    body_z_x = world_mat[2][0]
    body_z_y = world_mat[2][1]
    body_z_z = world_mat[2][2]

    # Pitch from forward tilt of body Z
    pitch = math.atan2(body_z_x, body_z_z)
    roll  = math.atan2(body_z_y, body_z_z)

    # Torso height
    height = world_mat[3][2]

    return pitch, roll, height


# ============================================================
# TIMELINE EVENT HANDLER
# ============================================================

def _on_timeline_event(event):
    global _was_playing
    if event.type == int(omni.timeline.TimelineEventType.PLAY):
        _controller.reset()
        if _foot_estimator is not None:
            _foot_estimator.reset()
        _was_playing = True
        print(f"\n[Controller] PLAY — COM-based balance active (200 Hz)")
    elif event.type == int(omni.timeline.TimelineEventType.STOP):
        _was_playing = False
        print(f"[Controller] STOPPED")


# ============================================================
# PHYSICS STEP CALLBACK (200 Hz)
# ============================================================

def physics_step_callback(dt):
    """Called every physics step (200 Hz)."""
    global _com_calc, _foot_estimator

    if not _was_playing:
        return

    if _com_calc is None or _foot_estimator is None:
        return

    stage = omni.usd.get_context().get_stage()

    # ── Get torso state (for safety + posture) ────────────────
    torso_pitch, torso_roll, torso_height = get_torso_state(stage)
    if torso_pitch is None:
        return

    # ── Safety check ──────────────────────────────────────────
    tilt_mag = math.degrees(math.sqrt(torso_pitch**2 + torso_roll**2))
    if tilt_mag > MAX_TILT_DEG or torso_height < MIN_BASE_HEIGHT:
        for name, default in DEFAULT_POSE.items():
            set_joint_target(name, default)
        if _controller.step % PRINT_EVERY_N == 0:
            status = "TILT" if tilt_mag > MAX_TILT_DEG else "FALLEN"
            print(f"  [{status}] tilt={tilt_mag:.1f}°  "
                  f"height={torso_height:.3f}m  — holding default")
        _controller.step += 1
        return

    # ── Compute COM ───────────────────────────────────────────
    com_xyz = _com_calc.compute()
    if com_xyz is None:
        return

    # ── Compute foot contact centroid ─────────────────────────
    foot_centroid_xy, avg_foot_z = _foot_estimator.compute_centroid()
    if foot_centroid_xy is None:
        return

    # ── Run controller ────────────────────────────────────────
    targets, debug = _controller.compute(
        com_xyz, foot_centroid_xy, torso_height, torso_pitch, dt
    )

    for name, target in targets.items():
        set_joint_target(name, target)

    # ── Debug print ───────────────────────────────────────────
    if _controller.step % PRINT_EVERY_N == 0:
        frozen = "F" if _foot_estimator.contact_captured else "L"

        # Read actual joint positions
        hp1_actual = get_joint_position("hip_pitch1")
        hp2_actual = get_joint_position("hip_pitch2")
        kn1_actual = get_joint_position("knee1")
        hr1_actual = get_joint_position("hip_roll1")

        def fmt_pos(val):
            return f"{val:+.3f}" if val is not None else "  N/A"

        print(
            f"  step={_controller.step:5d}  "
            f"err_x={debug['com_err_x']:+6.1f}mm  "
            f"rate={debug['com_rate_x']:+.3f}  "
            f"bal={debug['bal_hip']:+.4f}  "
            f"hp1 tgt={targets['hip_pitch1']:+.3f} act={fmt_pos(hp1_actual)}  "
            f"kn1 act={fmt_pos(kn1_actual)}  "
            f"COM_x={com_xyz[0]*1000:+6.1f}  "
            f"h={debug['torso_h']:.3f}m"
        )


# ============================================================
# SCENE SETUP
# ============================================================

async def setup_scene():
    global _joint_prims, _physx_sub, _timeline_sub, _com_calc, _foot_estimator

    stage = omni.usd.get_context().get_stage()

    # Clean up previous run
    if stage.GetPrimAtPath("/World"):
        omni.kit.commands.execute("DeletePrims", paths=["/World"])
        await omni.kit.app.get_app().next_update_async()

    # Create world
    UsdGeom.Xform.Define(stage, "/World")

    # Physics scene with 200 Hz timestep
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)

    physx_api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx_api.CreateTimeStepsPerSecondAttr(200)

    # Ground plane with friction
    PhysicsSchemaTools.addGroundPlane(
        stage, "/World/GroundPlane", "Z", 10.0,
        Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.5, 0.5, 0.5)
    )
    for prim in stage.Traverse():
        if prim.GetPath().pathString.startswith("/World/GroundPlane"):
            mat = UsdPhysics.MaterialAPI.Apply(prim)
            if mat:
                mat.CreateStaticFrictionAttr(0.9)
                mat.CreateDynamicFrictionAttr(0.8)
                mat.CreateRestitutionAttr(0.1)

    # Load robot
    robot_prim_path = "/World/Robot"
    robot_prim = stage.DefinePrim(robot_prim_path)
    robot_prim.GetReferences().AddReference(USD_PATH)
    await omni.kit.app.get_app().next_update_async()
    await omni.kit.app.get_app().next_update_async()

    # Add friction to robot collision surfaces
    for prim in stage.Traverse():
        if prim.GetPath().pathString.startswith("/World/Robot"):
            if prim.HasAPI(UsdPhysics.CollisionAPI) or "collision" in prim.GetName().lower():
                mat = UsdPhysics.MaterialAPI.Apply(prim)
                if mat:
                    mat.CreateStaticFrictionAttr(0.9)
                    mat.CreateDynamicFrictionAttr(0.8)

    # Spawn height
    xform = UsdGeom.Xformable(robot_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, SPAWN_HEIGHT))

    # ── Discover actual prim hierarchy ────────────────────────
    # Print all link prims so you can verify/fix LINK_PRIM_PATHS
    print("\n" + "=" * 70)
    print("  LINK HIERARCHY DISCOVERY")
    print("=" * 70)
    for prim in stage.Traverse():
        path_str = prim.GetPath().pathString
        if path_str.startswith("/World/Robot") and prim.IsA(UsdGeom.Xform):
            # Check if this looks like a link (has mass/inertia or is named like one)
            name = prim.GetName()
            is_known = name in LINK_MASS_TABLE
            marker = " <<<" if is_known else ""
            print(f"  {path_str}{marker}")

    # ── Configure joints ──────────────────────────────────────
    _joint_prims = {}

    print("\n" + "=" * 70)
    print("  COM-BASED BIPEDAL BALANCE CONTROLLER")
    print("=" * 70)

    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            name = prim.GetName()
            if name in JOINT_PARAMS:
                _joint_prims[name] = prim
                params = JOINT_PARAMS[name]
                target_rad = DEFAULT_POSE.get(name, 0.0)

                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.CreateTypeAttr("force")
                drive.CreateStiffnessAttr(params["stiffness"])
                drive.CreateDampingAttr(params["damping"])
                drive.CreateMaxForceAttr(params["effort"])
                drive.CreateTargetPositionAttr(np.degrees(target_rad))

                print(f"  {name:15s}  default={target_rad:+.3f} rad  "
                      f"Kp={params['stiffness']:<5}  Kd={params['damping']:<5}  "
                      f"max={params['effort']} N·m")

    # ── Initialize COM calculator and foot estimator ──────────
    _com_calc = COMCalculator(stage)
    _foot_estimator = FootContactEstimator(stage)

    # ── Probe joint attributes for position readback ──────────
    if "hip_pitch1" in _joint_prims:
        probe_prim = _joint_prims["hip_pitch1"]
        state_attrs = [a.GetName() for a in probe_prim.GetAttributes()
                       if "state" in a.GetName().lower()
                       or "position" in a.GetName().lower()
                       or "angle" in a.GetName().lower()
                       or "physx" in a.GetName().lower()]
        print(f"\n  [PROBE] hip_pitch1 attrs with state/position/angle/physx:")
        for a in sorted(state_attrs):
            attr = probe_prim.GetAttribute(a)
            val = attr.Get() if attr.HasValue() else "<no value>"
            print(f"    {a} = {val}")

    print(f"\n  Joints configured: {len(_joint_prims)}")
    print(f"  Physics timestep:  200 Hz")
    print(f"  ──────────────────────────────────────────")
    print(f"  BALANCE LOOP (COM_x → hip + knee):")
    print(f"    Kp={BAL_X_KP}  Kd={BAL_X_KD}")
    print(f"    hip_frac={BAL_X_TO_HIP}  knee_frac={BAL_X_TO_KNEE}")
    print(f"    hip_sign={BAL_X_HIP_SIGN:+d}  knee_sign={BAL_X_KNEE_SIGN:+d}")
    print(f"  BALANCE LOOP (COM_y → hip_roll):")
    print(f"    Kp={BAL_Y_KP}  Kd={BAL_Y_KD}  sign={BAL_Y_SIGN:+d}")
    print(f"  HEIGHT LOOP (height → knee):")
    print(f"    Kp={HEIGHT_KP}  Kd={HEIGHT_KD}  target={DESIRED_HEIGHT}m")
    print(f"    sign={HEIGHT_KNEE_SIGN:+d}")
    print(f"  POSTURE LOOP (pitch → hip, low priority):")
    print(f"    Kp={POSTURE_KP}  Kd={POSTURE_KD}  target={math.degrees(DESIRED_PITCH):.1f}°")
    print(f"    sign={POSTURE_HIP_SIGN:+d}")
    print(f"  ──────────────────────────────────────────")
    print(f"  Safety:    max tilt ±{MAX_TILT_DEG}°  min height {MIN_BASE_HEIGHT}m")
    print(f"  ──────────────────────────────────────────")
    print(f"  TUNING GUIDE:")
    print(f"    If robot falls forward/backward:")
    print(f"      → Flip BAL_X_HIP_SIGN or BAL_X_KNEE_SIGN")
    print(f"    If robot falls sideways:")
    print(f"      → Flip BAL_Y_SIGN")
    print(f"    If balance works but height is wrong:")
    print(f"      → Adjust DESIRED_HEIGHT, flip HEIGHT_KNEE_SIGN")
    print(f"    If COM errors are large but corrections are small:")
    print(f"      → Increase BAL_X_KP")
    print(f"    If oscillating:")
    print(f"      → Decrease Kp or increase Kd")
    print(f"    If link paths are wrong:")
    print(f"      → Check LINK HIERARCHY DISCOVERY above")
    print(f"      → Update LINK_PRIM_PATHS dict to match")
    print(f"\n  >>> Press PLAY to start balancing <<<")
    print("=" * 70 + "\n")

    # Register physics step callback
    _physx_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(
        physics_step_callback
    )

    # Register timeline event
    timeline = omni.timeline.get_timeline_interface()
    stream = timeline.get_timeline_event_stream()
    _timeline_sub = stream.create_subscription_to_pop(_on_timeline_event)

    print("[Controller] Physics callback + timeline events registered")


# ============================================================
# LAUNCH
# ============================================================

asyncio.ensure_future(setup_scene())