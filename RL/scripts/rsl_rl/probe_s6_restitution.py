"""probe_s6_restitution.py — S6 Phase-A verification probe (NO training).

Two checks, answered by running the S6 env for a few iterations and reading
the per-env material buffer and the live physics scene directly:

  1. Does the per-env foot-restitution draw RESAMPLE on every reset?  Read
     material row 45 (the first ball row) for env 0 after two consecutive
     resets.  With 64 buckets the two draws coincide with prob ~1/64; if they
     do, print envs 0-9 instead so the set's spread is visible.
  2. What does the runtime physics scene hold for physxScene:bounceThreshold?

This is the env's OWN DR path — gym.make() builds the RlEnv with the S6
overrides, and the env's reset/step write the buffer through
set_material_properties.  No policy is loaded, no learner is created, nothing
is saved.  Read-back is from get_material_properties() / the USD stage, never
from the sampled tensor.
"""
from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="S6 restitution resample + threshold probe.")
parser.add_argument("--task", type=str, default="Template-standup_curriculum-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=10)
parser.add_argument("--steps_per_iter", type=int, default=24)
parser.add_argument("--iters", type=int, default=3)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── everything else after the app is up ──────────────────────────────────────
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import RL.tasks  # noqa: F401,E402
from RL.tasks.direct.standup_curriculum.rl_env_cfg_p6 import RlEnvCfg  # noqa: E402


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: RlEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    _probe_body(env)
    return env


def row_restitution(env, row: int) -> float:
    """Read material row `row` column 2 (restitution) for env 0 from the
    physics buffer — the value PhysX actually holds."""
    mats = env.unwrapped.robot.root_physx_view.get_material_properties()
    return float(mats[0, row, 2].item())


def runtime_bounce_threshold(env) -> str:
    """physxScene:bounceThreshold as the live stage holds it (probe_foot_drop
    measured the attribute name has no Velocity suffix)."""
    stage = env.unwrapped.sim.stage
    for path in dict.fromkeys(
        [sim_utils.SimulationCfg().physics_prim_path, "/physicsScene", "/World/physicsScene"]
    ):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            attr = prim.GetAttribute("physxScene:bounceThreshold")
            val = attr.Get() if attr else None
            return f"{val!r} (prim {path})"
    return "<physicsScene not found>"


def _probe_body(env) -> None:
    rl = env.unwrapped
    rows = rl._foot_restitution_rows
    print(f"[S6-probe] discovered restitution rows: {rows}", flush=True)
    first_row = rows[0]
    print(f"[S6-probe] reading row {first_row} (first ball row), env 0", flush=True)

    # Reset 1: the env's own reset path ran during gym.make() / first obs.
    e_reset1 = row_restitution(env, first_row)
    print(f"[S6-probe] env0 row{first_row} restitution after RESET 1: {e_reset1:.6f}", flush=True)

    # Advance a few steps so a subsequent reset is a genuine re-draw.
    obs = env.reset()[0]
    for it in range(args_cli.iters):
        act = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        for _ in range(args_cli.steps_per_iter):
            obs, *_ = env.step(act)
    _ = obs

    # Reset 2: force the env's own reset on env 0 and read again.
    rl._reset_idx(torch.tensor([0], device=rl.device))
    e_reset2 = row_restitution(env, first_row)
    print(f"[S6-probe] env0 row{first_row} restitution after RESET 2: {e_reset2:.6f}", flush=True)

    # Reset 3: a second genuine DR draw, so resampling is proven on two draws
    # (not one draw vs the pre-DR carrier, which is trivially different).
    rl._reset_idx(torch.tensor([0], device=rl.device))
    e_reset3 = row_restitution(env, first_row)
    print(f"[S6-probe] env0 row{first_row} restitution after RESET 3: {e_reset3:.6f}", flush=True)

    if abs(e_reset1 - e_reset2) < 1e-9 or abs(e_reset2 - e_reset3) < 1e-9:
        print("[S6-probe] a consecutive pair drew the same value — printing envs 0-9 "
              "to show the set varies:", flush=True)
        for i in range(min(10, args_cli.num_envs)):
            mats = rl.robot.root_physx_view.get_material_properties()
            print(f"    env {i}: row{first_row} restitution = "
                  f"{float(mats[i, first_row, 2].item()):.6f}", flush=True)
        same = True
    else:
        print("[S6-probe] env0 resampled across consecutive resets.", flush=True)
        same = False

    print(f"[S6-probe] runtime physxScene:bounceThreshold = "
          f"{runtime_bounce_threshold(env)}", flush=True)

    print("[S6-probe] DONE — no training ran.", flush=True)


if __name__ == "__main__":
    env = main()
    if env is not None:
        env.close()
    simulation_app.close()
