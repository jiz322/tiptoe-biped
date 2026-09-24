"""Dump the body geometry of spawn poses (no motion): for each requested
difficulty d, spawn every env at exactly d with an effectively infinite grace,
hold the spawn PD target for --hold steps with zero action, then print per
body: link-origin position relative to the root in the heading frame (x fwd,
y left, z up) and its world height; plus toe positions, whole-body CoM, root
height and torso pitch. Optionally also holds an arbitrary joint pose
(--hold_pose hip,knee, applied as [0,hip,knee,0,-hip,-knee]) from the d spawn
to see where a static pose settles. Diagnostic only — no policy needed.

Example:
  python scripts/rsl_rl/probe_pose_dump.py --task Template-standup_curriculum-v0 --headless \
      --num_envs 16 --alphas 0,0.5,1.0 env.dr_enable=false
"""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Spawn pose geometry dump.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--alphas", type=str, default="0,0.5,1.0")
parser.add_argument("--hold", type=int, default=150, help="steps to hold before the dump")
parser.add_argument("--hold_pose", type=str, default=None,
                    help="'hip,knee[;hip,knee...]' extra static poses to hold from each spawn")
parser.add_argument("--scan", action="store_true",
                    help="compact spawn-consistency scan: one line per d with root z / pitch / speed / "
                         "min body z / knee & ball heights at +1 step, +0.1 s, +0.5 s, +1 s (path mode etc.)")
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
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.curriculum_enable = True
    env_cfg.push_enable = False
    env_cfg.pickup_enable = False
    env_cfg.standup_grace_seconds = 1000.0     # ramp effectively frozen at the spawn pose
    env_cfg.kneel_noise_rad = 0.0
    env_cfg.noise_joint_pos = 0.0
    env_cfg.init_base_orient_noise_rad = 0.0
    env_cfg.noise_base_vel_lin = 0.0
    env_cfg.noise_base_vel_ang = 0.0

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    raw = env.unwrapped
    raw.cfg.curriculum_alpha_step = 0.0
    raw._force_fixed_difficulty = True
    n = raw.num_envs
    names = list(raw.robot.body_names)
    kq = raw._kneel_quat.unsqueeze(0)
    pg_k = quat_apply_inverse(kq, raw._gravity_vec[:1])
    sign = -1.0 if math.asin(max(-1.0, min(1.0, -pg_k[0, 0].item()))) > 0 else 1.0
    zero = torch.zeros(n, raw.cfg.action_space, device=raw.device)

    def dump(label):
        root = raw.robot.data.root_pos_w
        quat = raw.robot.data.root_quat_w
        org = raw.scene.env_origins
        pg = quat_apply_inverse(quat, raw._gravity_vec)
        pitch = sign * torch.rad2deg(torch.asin((-pg[:, 0]).clamp(-1, 1)))
        roll = torch.rad2deg(torch.asin((pg[:, 1]).clamp(-1, 1)))
        fwd = quat_apply(quat, torch.tensor([1.0, 0, 0], device=raw.device).expand(n, -1))
        fwd[:, 2] = 0
        fwd = fwd / fwd.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        left = torch.stack([-fwd[:, 1], fwd[:, 0], torch.zeros_like(fwd[:, 0])], -1)

        def rel(p):  # p [N,3] world -> (fwd, left, up) relative to root, plus world height
            dp = p - root
            return torch.stack([(dp * fwd).sum(-1), (dp * left).sum(-1), dp[:, 2], p[:, 2] - org[:, 2]], -1).mean(0)

        masses = raw.robot.root_physx_view.get_masses().to(raw.device)
        com = (raw.robot.data.body_com_pos_w * masses.unsqueeze(-1)).sum(1) / masses.sum(1, keepdim=True)
        toe1, toe2 = raw._get_toe_pos_w()
        jp = raw.robot.data.joint_pos[:, raw._joint_ids].mean(0)
        print(f"\n=== {label} ===  root h={(root[:, 2] - org[:, 2]).mean():.3f} m  pitch={pitch.mean():.1f}° "
              f"(sd {pitch.std():.1f})  roll={roll.mean():.1f}°  joints={[round(v, 2) for v in jp.tolist()]}")
        print(f"  {'body':34s} {'fwd':>7s} {'left':>7s} {'up':>7s} | {'world z':>8s}   mass")
        for b, nm in enumerate(names):
            r = rel(raw.robot.data.body_link_pos_w[:, b])
            print(f"  {nm:34s} {r[0]:7.3f} {r[1]:7.3f} {r[2]:7.3f} | {r[3]:8.3f}   {masses[0, b]:.3f}")
        for nm, p in (("toe1 (ball centre)", toe1), ("toe2 (ball centre)", toe2), ("whole-body CoM", com)):
            r = rel(p)
            print(f"  {nm:34s} {r[0]:7.3f} {r[1]:7.3f} {r[2]:7.3f} | {r[3]:8.3f}")
        tm = 0.5 * (toe1 + toe2)
        print(f"  CoM ahead of toe-mid: {((com - tm) * fwd).sum(-1).mean():+.3f} m ; "
              f"root ahead of toe-mid: {((root - tm) * fwd).sum(-1).mean():+.3f} m")

    poses = []
    if args_cli.hold_pose:
        for spec in args_cli.hold_pose.split(";"):
            h, k = [float(v) for v in spec.split(",")]
            poses.append((h, k))

    if args_cli.scan:
        BALL = {raw.cfg.foot_body_name_1: torch.tensor([-0.1962, 0.0286, -0.0184], device=raw.device),
                raw.cfg.foot_body_name_2: torch.tensor([0.0015, 0.1992, 0.0184], device=raw.device)}
        bids = [names.index(raw.cfg.foot_body_name_1), names.index(raw.cfg.foot_body_name_2)]

        def snap():
            org = raw.scene.env_origins
            root = raw.robot.data.root_pos_w
            pg = quat_apply_inverse(raw.robot.data.root_quat_w, raw._gravity_vec)
            pitch = torch.rad2deg(torch.asin(pg[:, 0].clamp(-1, 1))).mean().item()
            speed = raw.robot.data.root_lin_vel_w.norm(dim=-1).mean().item()
            bz = raw.robot.data.body_link_pos_w[:, :, 2] - org[:, 2:3]
            knee = bz[:, bids].min(dim=-1).values.mean().item()
            balls = []
            for nm, b in zip((raw.cfg.foot_body_name_1, raw.cfg.foot_body_name_2), bids):
                c = raw.robot.data.body_link_pos_w[:, b] + quat_apply(raw.robot.data.body_link_quat_w[:, b], BALL[nm].expand(n, -1))
                balls.append((c[:, 2] - org[:, 2]).mean().item())
            return (f"z {(root[:, 2] - org[:, 2]).mean():.3f} p {pitch:6.1f} v {speed:5.2f} "
                    f"minbody {bz.min(dim=-1).values.mean():+.3f} knee {knee:.3f} ball {min(balls):+.3f}")
        print(f"\nSCAN mode={raw.cfg.standup_ramp_mode}   columns: +1step | +0.1 s | +0.5 s | +1 s   (ball = ball centre z, rest ≈ 0.020)")
        for d in [float(a) for a in args_cli.alphas.split(",")]:
            raw._curriculum_alpha = d
            with torch.inference_mode():
                env.reset()
                out = []
                for k in range(101):
                    env.step(zero)
                    if k in (0, 9, 49, 99):
                        out.append(snap())
                s0 = 1.0 - d
                print(f"d={d:.3f} s0={s0:.3f} | " + " | ".join(out))
        env.close()
        return

    for d in [float(a) for a in args_cli.alphas.split(",")]:
        raw._curriculum_alpha = d
        with torch.inference_mode():
            env.reset()
            dump(f"d={d:.2f} spawn (t=0)")
            for _ in range(args_cli.hold):
                env.step(zero)
            dump(f"d={d:.2f} held {args_cli.hold} steps at spawn target")
            for h, k in poses:
                env.reset()
                tgt = torch.tensor([0.0, h, k, 0.0, -h, -k], device=raw.device)
                raw._env_init_joint_pos[:] = tgt
                for _ in range(args_cli.hold):
                    env.step(zero)
                dump(f"d={d:.2f} spawn → held target hip={h:+.2f} knee={k:+.2f}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
