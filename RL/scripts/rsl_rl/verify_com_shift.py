"""
verify_com_shift.py — prove the coherent whole-body CoM shift DR does what
it claims (S-line, 2026-09-06).

Builds a handful of envs with every other physics DR pinned off and
dr_com_shift_x_max / _y_max on, resets, and checks two things per env:

  1. PhysX read-back: the torso row of root_physx_view.get_coms() equals
     original + shift * M/m_torso (the mechanism), every other body row is
     untouched.
  2. The physics claim: the whole-body CoM, computed from get_masses() and
     the per-body CoM positions transformed into the BASE frame via the
     body poses, moves by exactly the sampled shift (< 1 mm), for every
     env, and the shift differs between envs (i.e. it is per-env).

Also prints the OLD knob's whole-body CoM_x std for comparison (per-link
independent draw at dr_com_offset_max=0.02), which is the bug being fixed.

Usage (Isaac python, headless):
    python scripts/rsl_rl/verify_com_shift.py --task Template-standup_curriculum-v0 --headless --num_envs 8
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Template-standup_curriculum-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--shift_x", type=float, default=0.03)
parser.add_argument("--shift_y", type=float, default=0.01)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import RL.tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402


def whole_body_com_base(uw) -> torch.Tensor:
    """Whole-body CoM in the base (torso) frame, [num_envs, 3], from live PhysX
    masses / CoM offsets and the current body poses."""
    masses = uw.robot.root_physx_view.get_masses().to(uw.device)            # [N, B]
    coms = uw.robot.root_physx_view.get_coms().to(uw.device)[..., :3]        # [N, B, 3] body-local
    body_pos = uw.robot.data.body_pos_w                                        # [N, B, 3]
    body_quat = uw.robot.data.body_quat_w                                      # [N, B, 4]
    com_w = body_pos + quat_apply(body_quat, coms)                             # [N, B, 3]
    total = masses.sum(dim=-1, keepdim=True)
    com_w_total = (masses.unsqueeze(-1) * com_w).sum(dim=1) / total            # [N, 3]
    root_pos = uw.robot.data.root_pos_w
    root_quat = uw.robot.data.root_quat_w
    return quat_apply_inverse(root_quat, com_w_total - root_pos)


def main() -> None:
    cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = 7
    # Pin every physics DR off so the only CoM change is the shift.
    cfg.dr_com_offset_max = 0.0
    cfg.dr_mass_mult_min = cfg.dr_mass_mult_max = 1.0
    cfg.dr_floor_friction_min = cfg.dr_floor_friction_max = 1.0
    cfg.dr_gravity_tilt_max_deg = 0.0
    cfg.curriculum_enable = False
    cfg.pickup_enable = False
    cfg.push_enable = False
    cfg.noise_joint_pos = 0.0
    cfg.init_base_orient_noise_rad = 0.0
    cfg.dr_com_shift_x_max = args_cli.shift_x
    cfg.dr_com_shift_y_max = args_cli.shift_y

    env = gym.make(args_cli.task, cfg=cfg, render_mode=None)
    uw = env.unwrapped
    N = uw.num_envs
    torso = uw._torso_idx
    gain = uw._com_shift_gain
    print(f"\n[verify_com_shift] num_envs={N} torso_idx={torso} M/m_torso gain={gain:.4f}")

    # Baseline: no shift. Reset with the knob off, measure the base-frame CoM.
    uw.cfg.dr_com_shift_x_max = 0.0
    uw.cfg.dr_com_shift_y_max = 0.0
    env.reset()
    uw.sim.step(render=False)          # let PhysX ingest the set_coms write
    uw.scene.update(dt=uw.physics_dt)
    com0 = whole_body_com_base(uw).clone()
    coms0 = uw.robot.root_physx_view.get_coms().clone()

    # With shift.
    uw.cfg.dr_com_shift_x_max = args_cli.shift_x
    uw.cfg.dr_com_shift_y_max = args_cli.shift_y
    env.reset()
    uw.sim.step(render=False)
    uw.scene.update(dt=uw.physics_dt)
    com1 = whole_body_com_base(uw)
    coms1 = uw.robot.root_physx_view.get_coms().clone()
    shift = uw._dr_com_shift.to("cpu")

    ok = True
    # 1. Mechanism: torso row moved by gain*shift, other rows untouched.
    torso_delta = (coms1[:, torso, :3] - coms0[:, torso, :3])
    exp_torso = shift * gain
    err_mech = (torso_delta - exp_torso).abs().max().item()
    other = torch.ones(coms0.shape[1], dtype=torch.bool); other[torso] = False
    err_other = (coms1[:, other, :3] - coms0[:, other, :3]).abs().max().item()
    print(f"  torso row delta vs gain*shift: max err {err_mech*1e3:.3f} mm   other rows max change {err_other*1e3:.3f} mm")
    ok &= err_mech < 1e-4 and err_other < 1e-6

    # 2. Physics: whole-body CoM in base frame moved by exactly `shift`.
    delta_com = (com1 - com0).to("cpu")
    err_phys = (delta_com - shift).abs().max().item()
    print(f"  whole-body CoM(base) delta vs sampled shift: max err {err_phys*1e3:.3f} mm")
    for i in range(min(N, 8)):
        print(f"    env{i}: shift=({shift[i,0]:+.4f},{shift[i,1]:+.4f})  ΔCoM=({delta_com[i,0]:+.4f},{delta_com[i,1]:+.4f},{delta_com[i,2]:+.4f})")
    ok &= err_phys < 1e-3
    ok &= shift[:, 0].std().item() > 0.3 * args_cli.shift_x   # per-env, not a global constant
    print(f"  per-env spread: shift_x std {shift[:,0].std().item():.4f} (max {args_cli.shift_x})")

    # 3. The bug, for the record: old knob's whole-body CoM_x std.
    uw.cfg.dr_com_shift_x_max = 0.0
    uw.cfg.dr_com_shift_y_max = 0.0
    uw.cfg.dr_com_offset_max = 0.02
    env.reset()
    uw.sim.step(render=False)
    uw.scene.update(dt=uw.physics_dt)
    com_old = whole_body_com_base(uw)
    print(f"  OLD per-link knob (dr_com_offset_max=0.02): whole-body CoM_x std over envs = "
          f"{(com_old[:,0]-com0[:,0]).std().item()*1e3:.2f} mm, max |ΔCoM_x| = "
          f"{(com_old[:,0]-com0[:,0]).abs().max().item()*1e3:.2f} mm")

    print(f"\n[verify_com_shift] {'PASS' if ok else 'FAIL'}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
