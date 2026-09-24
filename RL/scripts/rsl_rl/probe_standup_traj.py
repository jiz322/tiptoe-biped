"""Stand-up trajectory probe: per-step population traces at fixed difficulty.

Companion to probe_standup_ramp.py. For each (d, mode) it spawns every env at
exactly d, runs zero actions ("zero" = the scripted ramp alone) or the
checkpoint ("policy"), and records per control step: root height, signed
torso pitch (deg; sign such that the kneel orientation is negative), root
horizontal offset ahead of the mean toe position along the body's forward
axis (m; + = torso ahead of the feet), toe contact fraction, knee / hip-pitch
joint angles, the ramp's knee action-center, the knee motor target, the mean
raw knee action, and the alive fraction. Saved as one npz per run; plot with
data/_plot_standup_traj.py.

Example (GPU 2):
  python scripts/rsl_rl/probe_standup_traj.py --task Template-standup_curriculum-v0 \
      --headless --num_envs 256 --load_run <run> --checkpoint model_4840.pt \
      --alphas 0.6,0.8,1.0 --modes zero,policy --out logs/standup_traj.npz <recipe overrides...>
"""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Stand-up trajectory probe.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--alphas", type=str, default="0.6,0.8,1.0")
parser.add_argument("--grace_s", type=float, default=None, help="override standup_grace_seconds")
parser.add_argument("--modes", type=str, default="zero,policy")
parser.add_argument("--horizon_after_grace", type=int, default=200)
parser.add_argument("--out", type=str, required=True, help="npz path")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import RL.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.curriculum_enable = True
    env_cfg.push_enable = False
    env_cfg.pickup_enable = False
    if args_cli.grace_s is not None:
        env_cfg.standup_grace_seconds = args_cli.grace_s

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint and os.path.isabs(args_cli.checkpoint):
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    raw = env.unwrapped
    raw.cfg.curriculum_alpha_step = 0.0
    raw._force_fixed_difficulty = True
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] loading {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=raw.device)

    n = raw.num_envs
    fwd_b = torch.tensor([1.0, 0.0, 0.0], device=raw.device).expand(n, -1)
    stand_knee = raw._stand_joint_pos[2].item()
    # sign reference: pitch of the pure kneel orientation
    kq = raw._kneel_quat.unsqueeze(0)
    pg_k = quat_apply_inverse(kq, raw._gravity_vec[:1])
    kneel_pitch_sign = -1.0 if math.asin(max(-1.0, min(1.0, -pg_k[0, 0].item()))) > 0 else 1.0

    def signed_pitch_deg(quat):
        pg = quat_apply_inverse(quat, raw._gravity_vec)
        return kneel_pitch_sign * torch.rad2deg(torch.asin((-pg[:, 0]).clamp(-1, 1)))

    out = {}
    alphas = [float(a) for a in args_cli.alphas.split(",")]
    modes = args_cli.modes.split(",")
    for d in alphas:
        raw._curriculum_alpha = d
        grace = int(d * raw._max_grace_steps)
        horizon = grace + args_cli.horizon_after_grace
        for mode in modes:
            keys = ["z", "pitch", "fwd_off", "contact", "knee", "hip", "knee_center", "knee_target", "knee_act", "alive"]
            rec = {k: np.zeros((horizon, 3), dtype=np.float32) for k in keys}  # mean, p10, p90 over alive envs
            with torch.inference_mode():
                obs, _ = env.reset()
                alive = torch.ones(n, dtype=torch.bool, device=raw.device)
                for t in range(horizon):
                    # state BEFORE this step's action (matches what the policy sees)
                    z = raw.robot.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
                    quat = raw.robot.data.root_quat_w
                    pitch = signed_pitch_deg(quat)
                    toe1, toe2 = raw._get_toe_pos_w()
                    toe_mid = 0.5 * (toe1 + toe2)
                    fwd_w = quat_apply(quat, fwd_b)
                    fwd_w[:, 2] = 0.0
                    fwd_w = fwd_w / fwd_w.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                    off = ((raw.robot.data.root_pos_w[:, :2] - toe_mid[:, :2]) * fwd_w[:, :2]).sum(-1)
                    contact = ((toe1[:, 2] < raw.cfg.no_flight_contact_threshold).float()
                               + (toe2[:, 2] < raw.cfg.no_flight_contact_threshold).float()) * 0.5
                    jp = raw.robot.data.joint_pos[:, raw._joint_ids]
                    knee = 0.5 * (jp[:, 2].abs() + jp[:, 5].abs())
                    hip = 0.5 * (jp[:, 1].abs() + jp[:, 4].abs())
                    gf = (raw._grace_remaining.float() / max(raw._max_grace_steps, 1))
                    center = (raw._env_init_joint_pos[:, 2] * gf + stand_knee * (1 - gf)).abs()
                    target = raw._motor_pos_target[:, 2].abs()
                    if mode == "zero":
                        actions = torch.zeros(n, raw.cfg.action_space, device=raw.device)
                    else:
                        actions = policy(obs)
                    act_knee = 0.5 * (actions[:, 2].abs() + actions[:, 5].abs())
                    vals = dict(z=z, pitch=pitch, fwd_off=off, contact=contact, knee=knee, hip=hip,
                                knee_center=center, knee_target=target, knee_act=act_knee)
                    a = alive
                    for k, v in vals.items():
                        if a.any():
                            va = v[a].float()
                            rec[k][t] = (va.mean().item(), torch.quantile(va, 0.1).item(), torch.quantile(va, 0.9).item())
                        else:
                            rec[k][t] = np.nan
                    rec["alive"][t] = a.float().mean().item()
                    obs, _, dones, _ = env.step(actions)
                    alive = alive & ~(dones > 0)
            tag = f"{mode}_d{d:.2f}"
            for k in keys:
                out[f"{tag}/{k}"] = rec[k]
            out[f"{tag}/grace"] = np.array(grace)
            zz = rec["z"][:, 0]
            print(f"{tag}: grace={grace} z0={zz[0]:.3f} zmax={np.nanmax(zz):.3f}@{int(np.nanargmax(zz))} "
                  f"z@grace={zz[grace-1] if grace > 0 else zz[0]:.3f} alive@grace={rec['alive'][min(grace, horizon-1)][0]:.2f} "
                  f"pitch0={rec['pitch'][0,0]:.0f} pitch@grace={rec['pitch'][grace-1,0]:.0f}")
    np.savez(args_cli.out, **out)
    print(f"[INFO] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
