"""Quasi-static balance manifold of the stand-up: for a grid of symmetric leg
poses (hip_pitch h, knee k; leg 2 mirrored) find the torso pitch θ* that puts
the whole-body CoM exactly above the toe-mid, plus the CoM height above the
toes and the knee heights in that balanced orientation (knee z ≲ 0 → the knee
is on the ground too, i.e. a statically stable kneel).

Uses the sim as the FK: every env is spawned airborne at identity orientation
with its own grid pose (PD target = the pose, one control step so body poses
refresh; free fall over 10 ms is negligible), then CoM / ball-foot / knee positions
are read in the base frame and rotated analytically. Reference point = the
foot BALL centre from the USD sphere collider (the nail-frame point that
_get_toe_pos_w returns sits ~8.6 cm up the shin, so it is only a contact
proxy when the shin is vertical); heights are above the ground (ball bottom). Writes a CSV; plot and
path-design with data/_plot_balance_manifold.py.

Example (GPU 2):
  python scripts/rsl_rl/probe_balance_manifold.py --task Template-standup_curriculum-v0 --headless \
      --hip_n 41 --knee_n 41 --out logs/balance_manifold.csv env.dr_enable=false
"""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Balance manifold probe.")
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--hip_min", type=float, default=0.0)
parser.add_argument("--hip_max", type=float, default=3.10)
parser.add_argument("--knee_min", type=float, default=-3.10)
parser.add_argument("--knee_max", type=float, default=0.0)
parser.add_argument("--hip_n", type=int, default=32)
parser.add_argument("--knee_n", type=int, default=32)
parser.add_argument("--out", type=str, required=True)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import RL.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    n = args_cli.hip_n * args_cli.knee_n
    env_cfg.scene.num_envs = n
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.curriculum_enable = True
    env_cfg.push_enable = False
    env_cfg.pickup_enable = False
    env_cfg.standup_grace_seconds = 1000.0
    env_cfg.kneel_noise_rad = 0.0
    env_cfg.noise_joint_pos = 0.0
    env_cfg.init_base_orient_noise_rad = 0.0
    env_cfg.noise_base_vel_lin = 0.0
    env_cfg.noise_base_vel_ang = 0.0

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    raw = env.unwrapped
    raw.cfg.curriculum_alpha_step = 0.0
    raw._force_fixed_difficulty = True
    raw._curriculum_alpha = 1.0                     # grace active → nothing terminates
    dev = raw.device
    names = list(raw.robot.body_names)
    shin_ids = [names.index(raw.cfg.foot_body_name_1), names.index(raw.cfg.foot_body_name_2)]

    # ── foot collider geometry, MEASURED from the live stage ──
    # Historically hardcoded as (0.0205, bbox centres read 09-08).  Both numbers
    # were slightly wrong: 'part_3' is a 40 mm ball PLUS a boss reaching
    # r = 30.6 mm, so its AABB is a 41.6 mm cube whose centre is dragged off the
    # ball by the flange -- by 1.03 mm on this foot, which is 5 % of the radius
    # and enough to matter for a CoM-margin argument.  Measure instead.
    from RL.tasks.direct.standup_curriculum.pad_contact import measure_ball_geometry
    _geom_paths = tuple(f"/World/envs/env_0/Robot/{nm}"
                        for nm in (raw.cfg.foot_body_name_1, raw.cfg.foot_body_name_2))
    _offs, R = measure_ball_geometry(
        raw.sim.stage,
        (raw.cfg.foot_body_name_1, raw.cfg.foot_body_name_2),
        _geom_paths)
    ball_off_b = {
        raw.cfg.foot_body_name_1: _offs[0].to(dev),
        raw.cfg.foot_body_name_2: _offs[1].to(dev),
    }
    print(f"[GEOM] ball measured from the stage: R={R:.6f} m  offsets="
          f"{ {k: [round(x, 6) for x in v.tolist()] for k, v in ball_off_b.items()} }")
    print(f"[GEOM] ball r={R} offsets={ {k: v.tolist() for k, v in ball_off_b.items()} }")

    def ball_centres():
        out = []
        for nm in (raw.cfg.foot_body_name_1, raw.cfg.foot_body_name_2):
            b = names.index(nm)
            pos = raw.robot.data.body_link_pos_w[:, b]
            qq = raw.robot.data.body_link_quat_w[:, b]
            out.append(pos + quat_apply(qq, ball_off_b[nm].unsqueeze(0).expand(n, -1)))
        return out

    hips = torch.linspace(args_cli.hip_min, args_cli.hip_max, args_cli.hip_n, device=dev)
    knees = torch.linspace(args_cli.knee_min, args_cli.knee_max, args_cli.knee_n, device=dev)
    H, K = torch.meshgrid(hips, knees, indexing="ij")
    h, k = H.reshape(-1), K.reshape(-1)
    q = torch.stack([torch.zeros_like(h), h, k, torch.zeros_like(h), -h, -k], dim=-1)

    with torch.inference_mode():
        env.reset()
        for _ in range(100):
            env.step(torch.zeros(n, raw.cfg.action_space, device=dev))
        b1, b2 = ball_centres()
        org = raw.scene.env_origins
        t1, t2 = raw._get_toe_pos_w()
        print(f"[CHECK] seiza rest: ball centre z {(b1[:,2]-org[:,2]).mean():.4f}/{(b2[:,2]-org[:,2]).mean():.4f} (expect ≈ R={R:.4f}), "
              f"nail-point z {(t1[:,2]-org[:,2]).mean():.4f}/{(t2[:,2]-org[:,2]).mean():.4f}, "
              f"ball ahead of nail point (world) {(b1-t1)[:, :2].norm(dim=-1).mean():.4f} m")
        env.reset()
        root = raw.robot.data.default_root_state[:, :7].clone()
        root[:, :3] = raw.scene.env_origins
        root[:, 2] += 1.0
        root[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
        jp = raw.robot.data.default_joint_pos.clone()
        jp[:, raw._joint_ids] = q
        raw.robot.write_root_pose_to_sim(root)
        raw.robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
        raw.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), None, None)
        raw._env_init_joint_pos[:] = q                 # PD target = pose (grace centre)
        raw._motor_pos_target[:] = q
        raw._act_lag_pf_y[:] = q
        env.step(torch.zeros(n, raw.cfg.action_space, device=dev))

        quat = raw.robot.data.root_quat_w
        masses = raw.robot.root_physx_view.get_masses().to(dev)
        com_w = (raw.robot.data.body_com_pos_w * masses.unsqueeze(-1)).sum(1) / masses.sum(1, keepdim=True)
        toe1, toe2 = ball_centres()          # ball centres (contact = directly below)
        toe_mid = 0.5 * (toe1 + toe2)
        rootp = raw.robot.data.root_pos_w
        knee_w = raw.robot.data.body_link_pos_w[:, shin_ids]              # [N,2,3] shin origins = knees
        jq = raw.robot.data.joint_pos[:, raw._joint_ids]
        drift_e = (jq - q).abs().max(dim=-1).values
        pg = quat_apply_inverse(quat, raw._gravity_vec)
        tilt_e = torch.rad2deg(torch.acos((-pg[:, 2]).clamp(-1, 1)))
        print(f"[INFO] after 1 step: max joint drift {drift_e.max().item():.4f} rad "
              f"({(drift_e > 0.05).sum().item()} cells > 0.05), max base tilt {tilt_e.max().item():.3f} deg")

        def to_b(p):
            return quat_apply_inverse(quat, p - toe_mid)
        v = to_b(com_w)                       # CoM rel toe-mid, base frame
        r = to_b(rootp)
        kn = torch.stack([to_b(knee_w[:, 0]), to_b(knee_w[:, 1])], 1)
        # base pitch about +y that makes v vertical (x' = vx cosθ + vz sinθ = 0)
        theta = torch.atan2(-v[:, 0], v[:, 2])
        c, s = torch.cos(theta), torch.sin(theta)

        def rot(p):  # apply R_y(theta) to base-frame vectors
            x = p[..., 0] * c.reshape(-1, *([1] * (p.dim() - 2))) + p[..., 2] * s.reshape(-1, *([1] * (p.dim() - 2)))
            z = -p[..., 0] * s.reshape(-1, *([1] * (p.dim() - 2))) + p[..., 2] * c.reshape(-1, *([1] * (p.dim() - 2)))
            return x, z
        _, com_h = rot(v)
        root_x, root_h = rot(r)
        knee_x, knee_h = rot(kn)
        # heights above the GROUND (ball bottom), not above the ball centre
        com_h, root_h, knee_h = com_h + R, root_h + R, knee_h + R
    with open(args_cli.out, "w") as f:
        f.write("hip,knee,theta_deg,com_h,root_x,root_h,knee1_x,knee1_h,knee2_x,knee2_h,com_x_b,com_z_b,drift,tilt\n")
        for i in range(n):
            f.write(f"{h[i]:.4f},{k[i]:.4f},{math.degrees(theta[i].item()):.3f},{com_h[i]:.4f},"
                    f"{root_x[i]:.4f},{root_h[i]:.4f},{knee_x[i,0]:.4f},{knee_h[i,0]:.4f},"
                    f"{knee_x[i,1]:.4f},{knee_h[i,1]:.4f},{v[i,0]:.4f},{v[i,2]:.4f},"
                    f"{drift_e[i]:.4f},{tilt_e[i]:.3f}\n")
    print(f"[INFO] wrote {args_cli.out}")
    # sanity anchors
    for hh, kk, lab in ((0.05, 0.0, "stand"), (3.09, -3.06, "seiza")):
        i = ((h - hh).abs() + (k - kk).abs()).argmin().item()
        print(f"  {lab}: grid ({h[i]:.2f},{k[i]:.2f}) θ*={math.degrees(theta[i].item()):.1f}° "
              f"com_h={com_h[i]:.3f} root_h={root_h[i]:.3f} knee_h={knee_h[i,0]:.3f}/{knee_h[i,1]:.3f}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
