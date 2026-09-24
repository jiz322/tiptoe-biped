# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""
eval.py — Tier 2 evaluation script (balance test, separate process).

USAGE (from project root, on the 4060 Ti box):

    python eval.py \
        --task <your-task-name> \
        --checkpoint logs/rsl_rl/<exp>/<train_run>/model_<iter>.pt \
        --num_envs 1024 \
        --device cuda:0 \
        --scenarios balance_full_dr balance_extreme_dr

Prints a per-scenario metric summary to stdout, plus a cross-scenario
comparison table when more than one scenario is run.

⚠️  One scenario per process. Isaac Sim tolerates a single simulation
context per process — a second gym.make() after env.close() hangs
(observed 2026-08-23: 2 h spin, 1 CPU core, 0% GPU). Launch one
eval.py process per scenario (see run_eval_p9sel_GPU2.ps1).

Design notes:
  - Re-imports the current RlEnvCfg and mutates it per scenario.
    Do NOT edit rl_env_cfg_v7.py between training and eval, or the
    eval env will diverge from what was trained. Pin via git tag if
    you need to.
  - Per-env stats are frozen at first termination. Survivors get
    capped at the loop's final step. Survival curves remain meaningful
    regardless of when envs auto-reset.
  - Failure-cause breakdown requires a small env-side hook (see
    FAILURE_CAUSE_HOOK comment below) — left as a stub for v1.

Developer notes:
  1. Each scenario builds its own env. With 1024 envs and ~3000 steps
     per scenario, you're looking at maybe 1-2 min per scenario on a
     4060 Ti.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from isaaclab.app import AppLauncher

# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Tier 2 eval for bipedal walking RL.")
parser.add_argument("--task", type=str, required=True,
                    help="Same task name passed to train.py.")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to model_<iter>.pt checkpoint.")
parser.add_argument("--num_envs", type=int, default=1024,
                    help="Eval population. 4060 Ti 16GB handles up to ~4096.")
parser.add_argument("--terminate_frac", type=float, default=0.8,
                    help="Stop scenario once this fraction of envs has died once.")
parser.add_argument("--max_steps", type=int, default=3000,
                    help="Hard cap on steps per scenario (~60s @ 100Hz).")
parser.add_argument("--scenarios", nargs="+",
                    default=["balance_full_dr", "balance_extreme_dr"],
                    help="Which scenarios to run.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--cfg_overrides", nargs="*", default=[],
                    help="key=value pairs applied to the base env cfg before "
                         "scenario overrides (scenario wins on conflict). Use "
                         "to re-pass the training arm's CLI overrides — e.g. "
                         "K-line: servo_kp_hip_pitch=20.0 servo_kd_hip_pitch=1.4 "
                         "servo_kp_knee=30.0 servo_kd_knee=1.5 "
                         "sat_term_threshold=0.5 sat_term_tau_s=0.5. Without "
                         "these a K checkpoint is evaluated on the wrong plant.")
parser.add_argument("--dump_traj", type=str, default=None,
                    help="npz path: per-step channels for the first "
                         "--dump_envs envs (actions, joint pos/vel, applied "
                         "torque, projected gravity) — the sim twin of a "
                         "hardware balance_log recording")
parser.add_argument("--dump_envs", type=int, default=4)
parser.add_argument("--out_json", type=str, default=None,
                    help="If set, write {metadata, results} to this path.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ──────────────────────────────────────────────────────────────────
# Launch Isaac Sim (must precede any isaaclab/torch import that
# touches the runtime).
# ──────────────────────────────────────────────────────────────────
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ──────────────────────────────────────────────────────────────────
# Now safe to import everything else.
# ──────────────────────────────────────────────────────────────────
import torch
import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: F401
import isaaclab_tasks  # noqa: F401
import RL.tasks  # noqa: F401  — same import as train.py to register tasks

# Note: we do NOT hardcode-import the cfg class. The base cfg is fetched
# by task name from the gym registry inside main(), the same mechanism
# train.py uses (via @hydra_task_config → load_cfg_from_registry).
# This keeps eval.py task-agnostic — `--task <name>` is all you pass.


# ──────────────────────────────────────────────────────────────────
# Scenario definitions
# ──────────────────────────────────────────────────────────────────
@dataclass
class EvalScenario:
    name: str
    description: str
    cfg_overrides: dict[str, Any] = field(default_factory=dict)


# Training-range DR values are inherited from the cfg as-is. Only the
# things that differ from training go in cfg_overrides.
COMMON_BALANCE_OVERRIDES = {
    # Disable training-time perturbations
    "push_enable": True,
    "curriculum_enable": False,         # difficulty=0, kneel_start_probability ignored
    # Commands always zero (env reads these via getattr/manual_cmd_override)
    "vx_cmd_zero_probability": 0.4,
    "yaw_rate_cmd_zero_probability": 0.4,
}

# Runbook §0.2 replica gate (P11, K edition): single plant at measured
# central values, every DR band collapsed to a point. Bands are pinned
# rather than toggling dr_enable so every code path (delays, floor
# material, mass rescale) still runs — just deterministically.
# PASS = survives >=6 s in the majority of trials.
_HW_MATCHED_OVERRIDES = {
    **COMMON_BALANCE_OVERRIDES,
    "push_enable": False,
    # Actuator / joint plant → central measured (replant report 08-17)
    "dr_stiffness_mult_min": 1.0,      "dr_stiffness_mult_max": 1.0,
    "dr_damping_mult_min": 1.0,        "dr_damping_mult_max": 1.0,
    "dr_effort_mult_min": 1.0,         "dr_effort_mult_max": 1.0,
    "dr_armature_mult_min": 1.0,       "dr_armature_mult_max": 1.0,
    "dr_joint_friction_mult_min": 1.0, "dr_joint_friction_mult_max": 1.0,
    "dr_dyn_friction_mult_min": 1.0,   "dr_dyn_friction_mult_max": 1.0,
    "dr_visc_friction_mult_min": 1.0,  "dr_visc_friction_mult_max": 1.0,
    "dr_voltage_factor_min": 1.0,      "dr_voltage_factor_max": 1.0,
    # World: mass pinned at the P6 bench-measured +5.9% over URDF
    "dr_mass_mult_min": 1.06,          "dr_mass_mult_max": 1.06,
    "dr_floor_friction_min": 0.8,      "dr_floor_friction_max": 0.8,
    "dr_com_offset_max": 0.0,
    "dr_gravity_tilt_max_deg": 0.0,
    # Delays: measured single-tick loop lag (M1.1: 2.4-2.8 ms)
    "dr_action_delay_min": 1, "dr_action_delay_max": 1,
    "dr_imu_delay_min": 1,    "dr_imu_delay_max": 1,
    "dr_joint_delay_min": 1,  "dr_joint_delay_max": 1,
    # Clean sensors — this is a plant gate, not an obs-robustness probe
    "dr_obs_noise_proj_grav": 0.0, "dr_obs_noise_ang_vel": 0.0,
    "dr_obs_noise_joint_vel": 0.0, "dr_obs_noise_joint_pos": 0.0,
    "dr_joint_pos_bias_max": 0.0,  "dr_proj_grav_bias_max": 0.0,
    "dr_ang_vel_bias_max": 0.0,
}

SCENARIOS: dict[str, EvalScenario] = {
    "balance_full_dr": EvalScenario(
        name="balance_full_dr",
        description="cmd=0, no pushes, training-range DR",
        cfg_overrides={**COMMON_BALANCE_OVERRIDES},
    ),
    "balance_extreme_dr": EvalScenario(
        name="balance_extreme_dr",
        description="cmd=0, no pushes, widened DR (OOD probe)",
        cfg_overrides={
            **COMMON_BALANCE_OVERRIDES,
            # # Physics
            # "dr_mass_mult_min": 0.75,        "dr_mass_mult_max": 1.25,
            # "dr_floor_friction_min": 0.3,    "dr_floor_friction_max": 1.4,
            # "dr_com_offset_max": 0.02,
            # # Actuators
            # "dr_stiffness_mult_min": 0.75,   "dr_stiffness_mult_max": 1.25,
            # "dr_damping_mult_min": 0.5,      "dr_damping_mult_max": 1.5,
            # "dr_effort_mult_min": 0.8,       "dr_effort_mult_max": 1.0,
            # "dr_armature_mult_min": 0.5,     "dr_armature_mult_max": 1.5,
            # "dr_voltage_factor_min": 0.80,   "dr_voltage_factor_max": 1.10,
            # "dr_joint_friction_mult_min": 0.3, "dr_joint_friction_mult_max": 3.0,
            # # Delays
            "dr_action_delay_min": 0,
            "dr_action_delay_max": 0,
            # obs delay is split per-sensor in the cfg (there is no single
            # dr_obs_delay_*): IMU and joint feedback have their own windows.
            "dr_imu_delay_min": 0,
            "dr_imu_delay_max": 0,
            "dr_joint_delay_min": 0,
            "dr_joint_delay_max": 0,
            # # Episodic sensor biases
            # "dr_joint_pos_bias_max": 0.03 ,
            # "dr_proj_grav_bias_max": 0.02 ,
            # "dr_ang_vel_bias_max":   0.03 ,
        },
    ),
    "hw_matched": EvalScenario(
        name="hw_matched",
        description="runbook §0.2 gate: single plant, central measured values, DR off",
        cfg_overrides={**_HW_MATCHED_OVERRIDES},
    ),
    "hw_harsh_arm13": EvalScenario(
        name="hw_harsh_arm13",
        description="runbook §0.2 harsh corner: hw_matched + armature ×1.3 (closest to real hip per 08-20 tether-bias calc)",
        cfg_overrides={
            **_HW_MATCHED_OVERRIDES,
            "dr_armature_mult_min": 1.3, "dr_armature_mult_max": 1.3,
        },
    ),
    # ── NEW 2026-08-27 (P13 R-line sweep) ──────────────────────────
    "hw_push_recovery": EvalScenario(
        name="hw_push_recovery",
        description="P13 gate: hw_matched plant + trained-range pushes "
                    "(10-40 N, 10 steps, p=0.002/step ~= 3 pushes per 15 s)",
        cfg_overrides={
            **_HW_MATCHED_OVERRIDES,
            "push_enable": True,
            "push_probability": 0.002,
            "push_force_min": 10.0,
            "push_force_max": 40.0,
            "push_duration_steps": 10,
        },
    ),
    "r1_train_dr": EvalScenario(
        name="r1_train_dr",
        description="P13 robustness probe: R1's widened training conditions "
                    "(R1 plant-DR box, delays 1-3, spawn-posture DR, "
                    "exec-noise injection, pushes 10-40 N @ p=0.001). Same "
                    "box for every checkpoint (R3's cheap-dim extras NOT "
                    "included) — measures what the R bundle bought vs K.",
        cfg_overrides={
            **COMMON_BALANCE_OVERRIDES,
            # pushes at R1 training level
            "push_probability": 0.001,
            "push_force_min": 10.0,
            "push_force_max": 40.0,
            "push_duration_steps": 10,
            # exec-noise injection (real drivetrain error class incl.
            # belt tooth seating — 08-25 skip was 0.21 rad)
            "exec_noise_target_bias_max": 0.03,
            "exec_noise_target_std": 0.015,
            # spawn-posture DR
            "noise_joint_pos": 0.18,
            "init_base_orient_noise_rad": 0.12,
            # delays (hw measured 3-4 ticks under load)
            "dr_action_delay_min": 1, "dr_action_delay_max": 3,
            "dr_imu_delay_min": 1,    "dr_imu_delay_max": 2,
            "dr_joint_delay_min": 1,  "dr_joint_delay_max": 2,
            # plant DR widened to the R1 box
            "dr_floor_friction_min": 0.4,  "dr_floor_friction_max": 1.1,
            "dr_stiffness_mult_min": 0.80, "dr_stiffness_mult_max": 1.20,
            "dr_effort_mult_min": 0.8,
            "dr_com_offset_max": 0.02,
            "dr_ang_vel_bias_max": 0.05,
            "dr_gravity_tilt_max_deg": 2.0,
        },
    ),
}


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────
def apply_overrides(cfg, overrides: dict[str, Any]) -> None:
    """Mutate a configclass in-place. Verifies each attr exists so a
    typo'd scenario key fails loudly instead of silently being ignored."""
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise AttributeError(f"Cfg has no attribute '{k}' — typo in scenario overrides?")
        setattr(cfg, k, v)


def parse_checkpoint_iter(checkpoint_path: str) -> int:
    fname = os.path.basename(checkpoint_path)
    return int(fname.replace("model_", "").replace(".pt", ""))


def parse_cli_value(s: str):
    """Best-effort typed parse for --cfg_overrides values."""
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


# ──────────────────────────────────────────────────────────────────
# Per-env metric tracker
# ──────────────────────────────────────────────────────────────────
class EpisodeTracker:
    """Tracks per-env stats for the FIRST episode only.

    Once an env terminates, its stats freeze. Survivors at end of loop
    get final-step values. This makes survival curves and quantiles
    meaningful regardless of when envs auto-reset.
    """

    def __init__(self, num_envs: int, device: torch.device):
        self.num_envs = num_envs
        self.device = device
        # -1 = still alive in first episode; >=0 = step at which it died
        self.first_term_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)

        # Step-by-step accumulators (means over first-episode lifetime)
        self.sum_uprightness   = torch.zeros(num_envs, device=device)
        self.sum_base_height   = torch.zeros(num_envs, device=device)
        self.sum_base_vz_abs   = torch.zeros(num_envs, device=device)
        self.sum_action_jerk_abs = torch.zeros(num_envs, device=device)
        self.sum_flight_steps  = torch.zeros(num_envs, device=device)
        self.sum_stance_diff_abs = torch.zeros(num_envs, device=device)
        self.step_count        = torch.zeros(num_envs, device=device)

        # Final-state snapshots (taken at termination or scenario end)
        self.final_pos_drift   = torch.zeros(num_envs, device=device)
        self.final_yaw_drift   = torch.zeros(num_envs, device=device)
        self.final_base_height = torch.zeros(num_envs, device=device)
        self.min_base_height   = torch.full((num_envs,), float("inf"), device=device)
        self.max_base_height   = torch.full((num_envs,), float("-inf"), device=device)

        # DR features — frozen at termination so OOD robustness can be
        # correlated with which DR the dying env happened to draw.
        self.dr_features: dict[str, torch.Tensor] = {}

        # Per-joint saturation of the DETERMINISTIC action. Training's
        # clip_fraction diagnostics are measured with exploration noise
        # applied; deployment runs the mean, so those numbers overstate
        # what hardware sees. This is the sim2real-relevant figure.
        self.sum_clip = torch.zeros(num_envs, 6, device=device)

        # NEW 2026-08-25 (P12 desk 0.1): richer per-episode channels so
        # hardware trials can be pattern-matched against eval JSONs
        # without re-running sweeps (torque levels, |a| sat at the
        # hardware-comparable 0.95, joint-vel energy, pg/z variability).
        self.sum_tau_abs   = torch.zeros(num_envs, 6, device=device)
        self.max_tau_abs   = torch.zeros(num_envs, 6, device=device)
        self.sum_qd_sq     = torch.zeros(num_envs, 6, device=device)
        self.sum_act_abs   = torch.zeros(num_envs, 6, device=device)
        self.sum_sat095    = torch.zeros(num_envs, 6, device=device)
        self.sum_pg_x      = torch.zeros(num_envs, device=device)
        self.sum_pg_x_sq   = torch.zeros(num_envs, device=device)
        self.sum_pg_y      = torch.zeros(num_envs, device=device)
        self.sum_pg_y_sq   = torch.zeros(num_envs, device=device)
        self.sum_z         = torch.zeros(num_envs, device=device)
        self.sum_z_sq      = torch.zeros(num_envs, device=device)

        # Settled-window accumulators (step >= settle_start, env still on
        # first episode): the pickup-free acceptance-gate quantities —
        # startup/pickup transients excluded, matches "last 5 s" analyses.
        self.settle_start  = 0  # set by run_scenario before the loop
        self.settled_count = torch.zeros(num_envs, device=device)
        self.settled_sat095   = torch.zeros(num_envs, 6, device=device)
        self.settled_flight   = torch.zeros(num_envs, device=device)
        self.settled_pg_x     = torch.zeros(num_envs, device=device)
        self.settled_pg_x_sq  = torch.zeros(num_envs, device=device)
        self.settled_pg_y     = torch.zeros(num_envs, device=device)
        self.settled_pg_y_sq  = torch.zeros(num_envs, device=device)
        self.settled_z        = torch.zeros(num_envs, device=device)
        self.settled_z_sq     = torch.zeros(num_envs, device=device)
        self.settled_tau_abs  = torch.zeros(num_envs, 6, device=device)

        # Last action for jerk computation
        self._prev_action      = torch.zeros(num_envs, 6, device=device)
        self._prev_prev_action = torch.zeros(num_envs, 6, device=device)

    def update_step(self, env, step_idx: int, action: torch.Tensor,
                    terminated: torch.Tensor) -> None:
        """Call AFTER env.step() but BEFORE checking the loop break.
        env is the unwrapped DirectRLEnv.

        `terminated` MUST be a bool tensor of true failures (NOT the
        wrapper's `dones`, which is long and includes time_outs). Pass
        `env.unwrapped.reset_terminated.bool()` from the caller — that's
        the env's terminated-only buffer, populated by _get_dones().
        """
        # Defensive: ensure bool. If a long is passed by accident, `&`
        # below would promote to long and break boolean masking silently.
        if terminated.dtype != torch.bool:
            terminated = terminated.bool()
        alive = self.first_term_step == -1

        # ── per-step diagnostics (read from env state) ─────────────
        # Note: after env.step(), state for terminated envs is post-reset.
        # That's why we only accumulate where `alive` is True (was alive
        # entering this step), AND we read dones to detect first death.
        root_pos = env.robot.data.root_pos_w
        root_quat = env.robot.data.root_quat_w
        root_lin_vel = env.robot.data.root_lin_vel_w

        from isaaclab.utils.math import quat_rotate_inverse
        gravity_vec = env._gravity_vec
        proj_grav = quat_rotate_inverse(root_quat, gravity_vec)
        uprightness = -proj_grav[:, 2]
        base_height = root_pos[:, 2]
        base_vz_abs = root_lin_vel[:, 2].abs()

        # Flight = both feet above contact threshold
        toe1_w, toe2_w = env._get_toe_pos_w()
        in_flight = (
            (toe1_w[:, 2] > env.cfg.no_flight_contact_threshold)
            & (toe2_w[:, 2] > env.cfg.no_flight_contact_threshold)
        ).float()

        stance_diff = (env._foot1_contact_ema - env._foot2_contact_ema).abs()

        # ── accumulate, only for envs still on their first episode ──
        # If an env died this step (terminated=True & alive), it counts
        # toward its accumulator one last time, then freezes below.
        was_alive = alive  # snapshot before death-flagging
        if was_alive.any():
            self.sum_uprightness[was_alive]   += uprightness[was_alive]
            self.sum_base_height[was_alive]   += base_height[was_alive]
            self.sum_base_vz_abs[was_alive]   += base_vz_abs[was_alive]
            self.sum_flight_steps[was_alive]  += in_flight[was_alive]
            self.sum_stance_diff_abs[was_alive] += stance_diff[was_alive]
            self.min_base_height[was_alive]   = torch.minimum(
                self.min_base_height[was_alive], base_height[was_alive])
            self.max_base_height[was_alive]   = torch.maximum(
                self.max_base_height[was_alive], base_height[was_alive])

            jerk = (action - self._prev_action) - (self._prev_action - self._prev_prev_action)
            self.sum_action_jerk_abs[was_alive] += jerk[was_alive].abs().sum(dim=-1)

            self.sum_clip[was_alive] += (action[was_alive].abs() > 1.0).float()

            # NEW 2026-08-25: per-joint torque / qd / action / pg / z
            jid = getattr(env, "_joint_ids", slice(None))
            tau_abs = env.robot.data.applied_torque[:, jid].abs()
            qd = env.robot.data.joint_vel[:, jid]
            pg_x, pg_y = proj_grav[:, 0], proj_grav[:, 1]
            self.sum_tau_abs[was_alive] += tau_abs[was_alive]
            self.max_tau_abs[was_alive] = torch.maximum(
                self.max_tau_abs[was_alive], tau_abs[was_alive])
            self.sum_qd_sq[was_alive]   += qd[was_alive] ** 2
            self.sum_act_abs[was_alive] += action[was_alive].abs()
            sat095 = (action.abs() > 0.95).float()
            self.sum_sat095[was_alive]  += sat095[was_alive]
            self.sum_pg_x[was_alive]    += pg_x[was_alive]
            self.sum_pg_x_sq[was_alive] += pg_x[was_alive] ** 2
            self.sum_pg_y[was_alive]    += pg_y[was_alive]
            self.sum_pg_y_sq[was_alive] += pg_y[was_alive] ** 2
            self.sum_z[was_alive]       += base_height[was_alive]
            self.sum_z_sq[was_alive]    += base_height[was_alive] ** 2

            if step_idx >= self.settle_start:
                settled = was_alive
                self.settled_count[settled]   += 1.0
                self.settled_sat095[settled]  += sat095[settled]
                self.settled_flight[settled]  += in_flight[settled]
                self.settled_pg_x[settled]    += pg_x[settled]
                self.settled_pg_x_sq[settled] += pg_x[settled] ** 2
                self.settled_pg_y[settled]    += pg_y[settled]
                self.settled_pg_y_sq[settled] += pg_y[settled] ** 2
                self.settled_z[settled]       += base_height[settled]
                self.settled_z_sq[settled]    += base_height[settled] ** 2
                self.settled_tau_abs[settled] += tau_abs[settled]

            self.step_count[was_alive] += 1.0

        # ── freeze stats for newly dead envs ────────────────────────
        # bool & bool → bool mask. If `terminated` were long, this
        # would silently degenerate to integer indexing below.
        new_term = was_alive & terminated
        if new_term.any():
            self.first_term_step[new_term] = step_idx

        # Rotate action history
        self._prev_prev_action = self._prev_action
        self._prev_action = action

    def snapshot_final_state_pre_step(
        self,
        env,
        pos_drift: torch.Tensor,
        yaw_drift: torch.Tensor,
        base_height: torch.Tensor,
        dones_will_fire: torch.Tensor | None = None,
    ) -> None:
        """Capture the final-state quantities from BEFORE the step that
        will (potentially) terminate the env. Call this immediately before
        env.step(). On the step where an env dies, this is its last
        valid pre-reset state."""
        alive = self.first_term_step == -1
        # For everyone still alive, update the "most recent valid state"
        self.final_pos_drift[alive]   = pos_drift[alive]
        self.final_yaw_drift[alive]   = yaw_drift[alive]
        self.final_base_height[alive] = base_height[alive]


def compute_pos_drift(env) -> torch.Tensor:
    root_xy = env.robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return (root_xy - env._spawn_pos_xy).norm(dim=-1)


def compute_yaw_drift(env) -> torch.Tensor:
    # Inline the 4-line atan2 formula to avoid a second import. Identical
    # to _yaw_from_quat in rl_env.py.
    q = env.robot.data.root_quat_w
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    current_yaw = torch.atan2(siny_cosp, cosy_cosp)
    raw = current_yaw - env._spawn_yaw
    return torch.atan2(torch.sin(raw), torch.cos(raw)).abs()


# ──────────────────────────────────────────────────────────────────
# Scenario runner
# ──────────────────────────────────────────────────────────────────
def run_scenario(
    scenario: EvalScenario,
    env_cfg_base,
    checkpoint_path: str,
    num_envs: int,
    terminate_frac: float,
    max_steps: int,
    device: str,
    seed: int,
) -> dict[str, float]:
    """Build an env for this scenario, load policy, run, return metrics."""

    print(f"\n{'='*60}\n  Scenario: {scenario.name}\n  {scenario.description}\n{'='*60}")

    # 1. Build per-scenario cfg by cloning + applying overrides
    import copy
    cfg = copy.deepcopy(env_cfg_base)
    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    cfg.seed = seed
    apply_overrides(cfg, scenario.cfg_overrides)

    # 2. Build env. We don't go through gym.make so we can skip the
    # video wrapper and any task-level decorators. If your task does
    # something special in gym.make's entrypoint, switch to gym.make.
    env = gym.make(args_cli.task, cfg=cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=None)

    # 3. Configure command override (manual_cmd_override is an env
    # attribute, not a cfg field — see _reset_idx / _sample_commands).
    env.unwrapped.manual_cmd_override = True
    env.unwrapped.manual_vx = 0.0
    env.unwrapped.manual_yaw_rate = 0.0

    # 4. Load policy. We use OnPolicyRunner just for the loader; we
    # don't call learn(). Pass a dummy agent_cfg-compatible dict so the
    # runner can construct the actor-critic with the right architecture.
    # Easier path: rely on the runner's load() which restores arch from
    # the checkpoint's state_dict. Concretely, build the runner with
    # the SAME agent_cfg the training used, then call load(). For now
    # we look up agent_cfg via hydra by task name.
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = device
    agent_cfg.seed = seed

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(checkpoint_path, load_optimizer=False)

    # Get a deterministic inference-mode policy. This is the canonical
    # API and is version-stable:
    #   - rsl_rl 3.x: returns a callable wrapping `self.alg.policy.act_inference`
    #   - rsl_rl 2.x: returns a callable wrapping `self.alg.actor_critic.act_inference`
    # Either way, `policy(obs)` returns the mean of the Gaussian policy
    # — no sampling, no exploration noise.
    policy = runner.get_inference_policy(device=device)

    # 5. Set up tracker
    tracker = EpisodeTracker(num_envs, device=torch.device(device))
    # Settled window = final third of the horizon (last 5 s of a 15 s
    # run) — past startup + any initial-pickup ride; the acceptance-gate
    # frame for z/pg/sat stats.
    tracker.settle_start = (2 * max_steps) // 3

    # 6. Reset and run
    obs, _ = env.reset()
    obs_t = obs["policy"] if isinstance(obs, dict) else obs

    print(f"[eval] running until {int(terminate_frac * num_envs)}/{num_envs} envs have terminated, "
          f"or {max_steps} steps.")

    dump = None
    if args_cli.dump_traj:
        dump = {k: [] for k in ('action', 'joint_pos', 'joint_vel',
                                'applied_torque', 'proj_grav', 'root_z',
                                'terminated')}
        n_dump = min(args_cli.dump_envs, num_envs)

    for step in range(max_steps):
        # Pre-step snapshot — these are the final-state values for any
        # env that's about to die on this step.
        pos_drift   = compute_pos_drift(env.unwrapped)
        yaw_drift   = compute_yaw_drift(env.unwrapped)
        base_height = env.unwrapped.robot.data.root_pos_w[:, 2]
        tracker.snapshot_final_state_pre_step(
            env.unwrapped, pos_drift, yaw_drift, base_height)

        # Policy step (deterministic — get_inference_policy already
        # wraps act_inference, so just call it like a function).
        with torch.inference_mode():
            actions = policy(obs_t)

        obs, _, dones, _ = env.step(actions)
        obs_t = obs["policy"] if isinstance(obs, dict) else obs

        if dump is not None:
            rd = env.unwrapped.robot.data
            # Reindex asset joint order -> policy/sim order (hr1 hp1 kn1
            # hr2 hp2 kn2), same frame as hardware balance_log channels.
            jid = getattr(env.unwrapped, '_joint_ids', slice(None))
            dump['action'].append(actions[:n_dump].cpu().numpy().copy())
            dump['joint_pos'].append(
                rd.joint_pos[:n_dump, jid].cpu().numpy().copy())
            dump['joint_vel'].append(
                rd.joint_vel[:n_dump, jid].cpu().numpy().copy())
            dump['applied_torque'].append(
                rd.applied_torque[:n_dump, jid].cpu().numpy().copy())
            dump['proj_grav'].append(
                rd.projected_gravity_b[:n_dump].cpu().numpy().copy())
            dump['root_z'].append(
                rd.root_pos_w[:n_dump, 2].cpu().numpy().copy())
            dump['terminated'].append(
                env.unwrapped.reset_terminated[:n_dump].cpu().numpy().copy())

        # ⚠️  IMPORTANT: do NOT use `dones` here. The wrapper's `dones` is
        # (terminated | truncated).to(long) — it includes time_outs AND
        # is a long tensor. Mixing it with bool masks silently degenerates
        # to integer indexing. Use the env's terminated-only buffer.
        terminated = env.unwrapped.reset_terminated.bool()
        tracker.update_step(env.unwrapped, step, actions, terminated)

        # Check stop condition every 100 steps (cheap)
        if (step + 1) % 100 == 0:
            n_dead = (tracker.first_term_step >= 0).sum().item()
            if n_dead >= terminate_frac * num_envs:
                print(f"[eval] {n_dead}/{num_envs} terminated by step {step+1}; stopping.")
                break

    final_step = step + 1

    if dump is not None:
        import numpy as _np
        _np.savez(args_cli.dump_traj,
                  control_hz=1.0 / (env.unwrapped.cfg.sim.dt
                                    * env.unwrapped.cfg.decimation),
                  **{k: _np.stack(v) for k, v in dump.items()})
        print(f"[eval] dumped {final_step}-step trajectory of "
              f"{min(args_cli.dump_envs, num_envs)} envs -> {args_cli.dump_traj}")

    # 7. Cap survivors at final_step (they didn't die, treat as having
    # made it the full duration)
    survivor_mask = tracker.first_term_step < 0
    tracker.first_term_step[survivor_mask] = final_step

    # ────────────────────────────────────────────────────────────
    # Aggregate metrics
    # ────────────────────────────────────────────────────────────
    n = num_envs
    ep_lens = tracker.first_term_step.float()  # [N]
    step_counts = tracker.step_count.clamp(min=1.0)

    metrics: dict[str, float] = {}

    # ── headline: survival rate at horizons ─────────────────────
    control_hz = 1.0 / (env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation)
    for t_s in (5, 10, 15, 20, 25, 30):
        threshold_steps = int(t_s * control_hz)
        if threshold_steps > final_step:
            continue
        alive_at_t = (ep_lens >= threshold_steps).float().mean().item()
        metrics[f"survival/at_{t_s}s"] = alive_at_t

    # ── time-to-fall quantiles (over envs that died) ────────────
    died_lens = ep_lens[~survivor_mask]
    if died_lens.numel() > 0:
        for q in (0.10, 0.25, 0.50, 0.75, 0.90):
            metrics[f"time_to_fall/p{int(q*100):02d}_steps"] = (
                torch.quantile(died_lens, q).item()
            )
        metrics["time_to_fall/mean_steps"] = died_lens.mean().item()
    metrics["time_to_fall/survivor_frac"] = survivor_mask.float().mean().item()
    metrics["time_to_fall/num_died"]      = float(died_lens.numel())

    # ── episode-mean diagnostics (computed over first-episode lifetime) ─
    metrics["balance/uprightness_mean"]   = (tracker.sum_uprightness / step_counts).mean().item()
    metrics["balance/base_height_mean"]   = (tracker.sum_base_height / step_counts).mean().item()
    metrics["balance/base_vz_abs_mean"]   = (tracker.sum_base_vz_abs / step_counts).mean().item()
    metrics["balance/action_jerk_abs_mean"] = (tracker.sum_action_jerk_abs / step_counts).mean().item()
    metrics["balance/flight_fraction"]    = (tracker.sum_flight_steps / step_counts).mean().item()
    metrics["balance/stance_diff_abs_mean"] = (tracker.sum_stance_diff_abs / step_counts).mean().item()
    metrics["balance/base_height_min_mean"] = tracker.min_base_height.mean().item()
    metrics["balance/base_height_max_mean"] = tracker.max_base_height.mean().item()

    # Deterministic per-joint clip fractions, in sim action order.
    clip_frac = tracker.sum_clip / step_counts.unsqueeze(-1)
    for j, jname in enumerate(
        ["hr1", "hp1", "kn1", "hr2", "hp2", "kn2"]
    ):
        metrics[f"clip_det/{jname}"] = clip_frac[:, j].mean().item()
    metrics["clip_det/knee_max"] = max(
        metrics["clip_det/kn1"], metrics["clip_det/kn2"])

    # ── NEW 2026-08-25: per-joint torque / qd / action levels ──────
    # (joint-frame Nm; env-mean of per-episode means — comparable to
    # TB's diagnostics/actual_torque_*_mean and to hardware logs)
    _JN = ["hr1", "hp1", "kn1", "hr2", "hp2", "kn2"]
    tau_mean = tracker.sum_tau_abs / step_counts.unsqueeze(-1)
    qd_rms = (tracker.sum_qd_sq / step_counts.unsqueeze(-1)).sqrt()
    act_abs = tracker.sum_act_abs / step_counts.unsqueeze(-1)
    sat095 = tracker.sum_sat095 / step_counts.unsqueeze(-1)
    for j, jname in enumerate(_JN):
        metrics[f"tau_abs_mean/{jname}"] = tau_mean[:, j].mean().item()
        metrics[f"tau_abs_max/{jname}"] = tracker.max_tau_abs[:, j].mean().item()
        metrics[f"qd_rms/{jname}"] = qd_rms[:, j].mean().item()
        metrics[f"act_abs_mean/{jname}"] = act_abs[:, j].mean().item()
        metrics[f"sat095/{jname}"] = sat095[:, j].mean().item()
    metrics["sat095/all"] = sat095.mean().item()

    # pg / z variability over the full first episode
    def _mean_std(s, s_sq, cnt):
        m = s / cnt
        var = (s_sq / cnt - m ** 2).clamp(min=0.0)
        return m, var.sqrt()

    for tag, s, s_sq in (("pg_x", tracker.sum_pg_x, tracker.sum_pg_x_sq),
                         ("pg_y", tracker.sum_pg_y, tracker.sum_pg_y_sq),
                         ("base_z", tracker.sum_z, tracker.sum_z_sq)):
        m, sd = _mean_std(s, s_sq, step_counts)
        metrics[f"var/{tag}_mean"] = m.mean().item()
        metrics[f"var/{tag}_std"] = sd.mean().item()

    # ── settled window (final third; pickup-free-gate frame) ───────
    settled_ok = tracker.settled_count > 10
    if settled_ok.any():
        cnt = tracker.settled_count.clamp(min=1.0)
        s_sat = (tracker.settled_sat095 / cnt.unsqueeze(-1))[settled_ok]
        metrics["settled/sat095_all"] = s_sat.mean().item()
        metrics["settled/sat095_knee_max"] = max(
            s_sat[:, 2].mean().item(), s_sat[:, 5].mean().item())
        metrics["settled/flight_fraction"] = (
            tracker.settled_flight / cnt)[settled_ok].mean().item()
        s_tau = (tracker.settled_tau_abs / cnt.unsqueeze(-1))[settled_ok]
        for j, jname in enumerate(_JN):
            metrics[f"settled/tau_abs_mean_{jname}"] = s_tau[:, j].mean().item()
        for tag, s, s_sq in (
                ("pg_x", tracker.settled_pg_x, tracker.settled_pg_x_sq),
                ("pg_y", tracker.settled_pg_y, tracker.settled_pg_y_sq),
                ("base_z", tracker.settled_z, tracker.settled_z_sq)):
            m, sd = _mean_std(s[settled_ok], s_sq[settled_ok], cnt[settled_ok])
            metrics[f"settled/{tag}_mean"] = m.mean().item()
            metrics[f"settled/{tag}_std"] = sd.mean().item()
        metrics["settled/num_envs"] = float(settled_ok.sum().item())

    # ── final-state drift (taken at termination or scenario end) ─
    metrics["drift/pos_drift_mean"] = tracker.final_pos_drift.mean().item()
    metrics["drift/pos_drift_p90"]  = torch.quantile(tracker.final_pos_drift, 0.90).item()
    metrics["drift/yaw_drift_mean"] = tracker.final_yaw_drift.mean().item()
    metrics["drift/yaw_drift_p90"]  = torch.quantile(tracker.final_yaw_drift, 0.90).item()

    # ── FAILURE_CAUSE_HOOK ──────────────────────────────────────
    # If you add `self._last_term_cause = <int tensor>` to rl_env.py's
    # `_get_dones` (one int per env representing which condition fired,
    # 0=none, 1=too_tilted, 2=too_low, 3=side_split, 4=front_split, 5=knee_split),
    # uncomment and adapt:
    #
    # cause = env.unwrapped._last_term_cause   # but this is post-reset for dying envs
    # Better: hook in _get_dones to APPEND-only into a per-env buffer
    # that holds the last cause at termination time. See env-side
    # patch sketch in design doc.

    # 8. Print summary to stdout
    print(f"\n[eval] {scenario.name} summary:")
    for k in sorted(metrics.keys()):
        print(f"  {k:50s} = {metrics[k]:.4f}")

    # Cleanup
    env.close()
    return metrics


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
def main() -> None:
    device = args_cli.device or "cuda:0"

    # Build the base cfg by looking it up in the gym registry — same as
    # train.py does via @hydra_task_config. Returns a fresh instance of
    # whatever class is wired up as `env_cfg_entry_point` in your task's
    # gym.register(). Each scenario then deep-copies + mutates this.
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    base_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")

    # Re-apply the training arm's CLI overrides (launch-script parity).
    # Applied BEFORE scenario deepcopy, so scenario overrides win on
    # conflict — scenarios pin DR, arm overrides set the plant/gains.
    cli_overrides: dict[str, Any] = {}
    for kv in args_cli.cfg_overrides:
        if "=" not in kv:
            raise ValueError(f"--cfg_overrides entry '{kv}' is not key=value")
        k, v = kv.split("=", 1)
        cli_overrides[k] = parse_cli_value(v)
    if cli_overrides:
        apply_overrides(base_cfg, cli_overrides)
        print(f"[eval] base-cfg overrides: {cli_overrides}")

    results: dict[str, dict[str, float]] = {}
    for scenario_name in args_cli.scenarios:
        if scenario_name not in SCENARIOS:
            print(f"[eval] unknown scenario '{scenario_name}', skipping. "
                  f"Available: {list(SCENARIOS.keys())}")
            continue
        scenario = SCENARIOS[scenario_name]

        results[scenario_name] = run_scenario(
            scenario=scenario,
            env_cfg_base=base_cfg,
            checkpoint_path=args_cli.checkpoint,
            num_envs=args_cli.num_envs,
            terminate_frac=args_cli.terminate_frac,
            max_steps=args_cli.max_steps,
            device=device,
            seed=args_cli.seed,
        )

    # Cross-scenario summary table
    if len(results) > 1:
        print(f"\n{'='*80}")
        print(f"  EVAL SUMMARY  —  ckpt iter {parse_checkpoint_iter(args_cli.checkpoint)}")
        print(f"{'='*80}")
        keys = set()
        for r in results.values():
            keys.update(r.keys())
        for k in sorted(keys):
            row = " | ".join(
                f"{r.get(k, float('nan')):.4f}" for r in results.values()
            )
            print(f"  {k:50s}  {row}")

    if args_cli.out_json:
        import json
        payload = {
            "checkpoint": args_cli.checkpoint,
            "iter": parse_checkpoint_iter(args_cli.checkpoint),
            "cfg_overrides": cli_overrides,
            "num_envs": args_cli.num_envs,
            "max_steps": args_cli.max_steps,
            "terminate_frac": args_cli.terminate_frac,
            "seed": args_cli.seed,
            "results": results,
        }
        out_dir = os.path.dirname(os.path.abspath(args_cli.out_json))
        os.makedirs(out_dir, exist_ok=True)
        with open(args_cli.out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[eval] wrote {args_cli.out_json}")


if __name__ == "__main__":
    main()
    simulation_app.close()