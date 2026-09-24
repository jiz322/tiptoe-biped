"""Stand-up feasibility probe: scripted ramp vs policy, at fixed difficulty.

The kneel→stand motion in rl_env_p6 is a scripted PD-target ramp: action_center
blends linearly from the spawn pose to the standing pose over the grace window
(difficulty × standup_grace_seconds) and the policy only adds ±action_scale on
top. This probe spawns every env at exactly d and runs three controllers:

  zero     — actions = 0 → the pure ramp (is the scripted stand-up feasible?)
  policy   — the checkpoint, training termination rules
  nofence  — the checkpoint with sat_term_enable=False (how many deaths are
             the saturation fence rather than a fall?)

for each requested grace duration (sets _max_grace_steps at runtime, which
scales both the ramp speed and the termination shield). Everything else comes
from the hydra overrides — pass the training recipe, and for a clean plant
probe zero the DR/noise knobs (see the launch helper in the repo docs).

Per (d, grace, mode) it reports: height at spawn / grace end / +1 s, peak
height, fraction above min_height at grace end, survival ≥1 s and ≥3 s after
grace, knee torque-saturation duty during grace, and the (approximate) cause
of the first termination: sat fence / too_low / tilt / other.

Example (GPU 2):
  python scripts/rsl_rl/probe_standup_ramp.py --task Template-standup_curriculum-v0 \
      --headless --num_envs 256 --load_run <run> --checkpoint model_4840.pt \
      --alphas 0.6,0.8,1.0 --grace_s 4,8,12 --modes zero,policy,nofence <recipe overrides...>
"""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Stand-up ramp feasibility probe.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--alphas", type=str, default="0.6,0.7,0.8,0.9,1.0")
parser.add_argument("--grace_s", type=str, default="4,8,12", help="standup_grace_seconds values to probe")
parser.add_argument("--modes", type=str, default="zero,policy,nofence")
parser.add_argument("--horizon_after_grace", type=int, default=300)
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
from isaaclab.utils.math import quat_apply_inverse

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

    n = raw.num_envs
    dt = raw._control_dt
    knee_ids = raw._knee_ids
    knee_limit = raw._nominal_diag("knee", "effort_limit")
    cos_tilt_lim = math.cos(raw.cfg.max_tilt_rad)
    sat_enable_train = raw.cfg.sat_term_enable
    print(f"[INFO] knee effort limit {knee_limit:.2f} Nm, min_height {raw.cfg.min_height}, "
          f"sat fence {sat_enable_train} (thr {raw.cfg.sat_term_threshold}, tau {raw.cfg.sat_term_tau_s} s), "
          f"action scale {raw._action_scale.tolist()}")

    alphas = [float(a) for a in args_cli.alphas.split(",")]
    graces = [float(g) for g in args_cli.grace_s.split(",")]
    modes = args_cli.modes.split(",")
    rows = []
    hdr = ("mode     d    grace | h0    h_end  h+1s  h_max | up@end  surv1s surv3s | kneeSat | cause: sat  low  tilt other")
    print("\n" + hdr)
    for g in graces:
        raw._max_grace_steps = int(g / dt)
        for d in alphas:
            raw._curriculum_alpha = d
            grace = int(d * raw._max_grace_steps)
            horizon = grace + args_cli.horizon_after_grace
            for mode in modes:
                raw.cfg.sat_term_enable = sat_enable_train and (mode != "nofence")
                with torch.inference_mode():
                    obs, _ = env.reset()
                    z = raw.robot.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
                    h0 = z.mean().item()
                    first_len = torch.full((n,), -1, dtype=torch.long, device=raw.device)
                    cause = torch.zeros(n, dtype=torch.long, device=raw.device)  # 0 none,1 sat,2 low,3 tilt,4 other
                    h_max = z.clone()
                    knee_sat_steps = torch.zeros(n, device=raw.device)
                    h_end = torch.full((n,), float("nan"), device=raw.device)
                    h_p1 = torch.full((n,), float("nan"), device=raw.device)
                    prev_z = z.clone()
                    prev_cos = -quat_apply_inverse(raw.robot.data.root_quat_w, raw._gravity_vec)[:, 2]
                    for t in range(horizon):
                        if mode == "zero":
                            actions = torch.zeros(n, raw.cfg.action_space, device=raw.device)
                        else:
                            actions = policy(obs)
                        obs, _, dones, _ = env.step(actions)
                        alive = first_len < 0
                        newly = (dones > 0) & alive
                        if newly.any():
                            first_len[newly] = t + 1
                            c = torch.full((n,), 4, dtype=torch.long, device=raw.device)
                            c = torch.where(prev_cos < cos_tilt_lim + 0.05, torch.full_like(c, 3), c)
                            c = torch.where(prev_z < raw.cfg.min_height + 0.01, torch.full_like(c, 2), c)
                            c = torch.where(raw._sat_terminated, torch.full_like(c, 1), c)
                            cause[newly] = c[newly]
                        z = raw.robot.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
                        cosv = -quat_apply_inverse(raw.robot.data.root_quat_w, raw._gravity_vec)[:, 2]
                        alive = first_len < 0
                        h_max = torch.where(alive, torch.maximum(h_max, z), h_max)
                        if t < grace:
                            tau = raw.robot.data.applied_torque[:, knee_ids].abs()
                            knee_sat_steps += ((tau >= 0.95 * knee_limit).any(dim=-1) & alive).float()
                        if t + 1 == grace:
                            h_end = torch.where(alive, z, h_end)
                        if t + 1 == grace + 100:
                            h_p1 = torch.where(alive, z, h_p1)
                        prev_z, prev_cos = z.clone(), cosv.clone()
                        if (first_len >= 0).all():
                            break
                never = first_len < 0
                ep_len = torch.where(never, torch.full_like(first_len, horizon), first_len)
                post = (ep_len - grace).clamp(min=0)
                surv1 = (post >= 100).float().mean().item()
                surv3 = (post >= 300).float().mean().item()
                up_end = (~torch.isnan(h_end) & (h_end >= raw.cfg.min_height)).float().mean().item()
                knee_duty = (knee_sat_steps / max(grace, 1)).mean().item()
                cs = [(cause == k).float().mean().item() for k in (1, 2, 3, 4)]
                hend = torch.nanmean(h_end).item() if (~torch.isnan(h_end)).any() else float("nan")
                hp1 = torch.nanmean(h_p1).item() if (~torch.isnan(h_p1)).any() else float("nan")
                print(f"{mode:8s} {d:.2f} {g:5.1f} | {h0:.3f} {hend:6.3f} {hp1:6.3f} {h_max.mean().item():6.3f} | "
                      f"{up_end:6.2f} {surv1:6.2f} {surv3:6.2f} | {knee_duty:7.2f} | {cs[0]:.2f} {cs[1]:.2f} {cs[2]:.2f} {cs[3]:.2f}")
                rows.append((mode, d, g, h0, hend, hp1, h_max.mean().item(), up_end, surv1, surv3, knee_duty, *cs))
    raw.cfg.sat_term_enable = sat_enable_train

    if args_cli.out:
        with open(args_cli.out, "w") as f:
            f.write("mode,d,grace_s,h0,h_end,h_plus1s,h_max,up_at_end,surv1s,surv3s,knee_sat_duty,cause_sat,cause_low,cause_tilt,cause_other\n")
            for r in rows:
                f.write(",".join(f"{v}" for v in r) + "\n")
        print(f"[INFO] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
