"""Headless stand-up success rate at fixed curriculum difficulties.

For each requested difficulty d, every env is spawned at exactly d (kneel
interpolation, grace = d × standup_grace_seconds), the deterministic policy
runs until each env has terminated once (or the horizon runs out), and the
episode is a SUCCESS if it survived at least --post_grace_steps control steps
after grace expiry (a robot that never rises terminates too_low the step
grace ends). Pushes are disabled; everything else (DR, delays, exec noise)
follows the hydra overrides you pass — re-pass the training recipe.

Example (GPU 2):
  python scripts/rsl_rl/eval_alpha_success.py --task Template-standup_curriculum-v0 \
      --headless --num_envs 1024 --load_run <run> --checkpoint model_3890.pt \
      --alphas 0.3,0.4,0.5,0.6,0.7,0.8,1.0 env.curriculum_enable=true <recipe overrides...>
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Fixed-difficulty stand-up success eval.")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--alphas", type=str, default="0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
                    help="comma-separated fixed difficulties")
parser.add_argument("--post_grace_steps", type=int, default=100,
                    help="steps survived after grace expiry that count as success")
parser.add_argument("--horizon_after_grace", type=int, default=600,
                    help="stop the rollout this many steps after grace expiry")
parser.add_argument("--out", type=str, default=None, help="optional CSV path")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

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

    alphas = [float(a) for a in args_cli.alphas.split(",")]
    n = raw.num_envs
    max_grace = raw._max_grace_steps
    rows = []
    print(f"\nckpt={os.path.basename(resume_path)} envs={n} success = survive ≥{args_cli.post_grace_steps} steps post-grace")
    print("  d    grace  success  postgrace_mean  postgrace_p10  never_done")
    for d in alphas:
        raw._curriculum_alpha = d
        grace = int(d * max_grace)
        horizon = grace + args_cli.horizon_after_grace
        with torch.inference_mode():
            obs, _ = env.reset()
            first_len = torch.full((n,), -1, dtype=torch.long, device=raw.device)
            for t in range(horizon):
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                newly = (dones > 0) & (first_len < 0)
                first_len[newly] = t + 1
                if (first_len >= 0).all():
                    break
        never = first_len < 0
        ep_len = torch.where(never, torch.full_like(first_len, horizon), first_len)
        post = (ep_len - grace).clamp(min=0).float()
        success = (post >= args_cli.post_grace_steps).float().mean().item()
        p10 = torch.quantile(post, 0.10).item()
        rows.append((d, grace, success, post.mean().item(), p10, never.float().mean().item()))
        print(f"  {d:.2f}  {grace:5d}  {success:7.3f}  {post.mean().item():14.1f}  {p10:13.1f}  {never.float().mean().item():10.3f}")

    if args_cli.out:
        with open(args_cli.out, "w") as f:
            f.write("ckpt,d,grace,success,postgrace_mean,postgrace_p10,never_done\n")
            for r in rows:
                f.write(os.path.basename(resume_path) + "," + ",".join(f"{v}" for v in r) + "\n")
        print(f"[INFO] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
