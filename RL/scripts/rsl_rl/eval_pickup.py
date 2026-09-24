# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""
ep_len_impact.py — one-factor-at-a-time (OAT) sensitivity sweep over the
config dimensions that affect episode length (time-to-first-fall) for a
FIXED policy. Generalizes the pickup sweep to every family that matters.

WHAT IS / ISN'T SWEEPABLE AT EVAL TIME
    A checkpoint is fixed, so only things that change the *fixed policy's*
    behavior or its termination condition matter here:
      - SWEEPABLE: disturbances (pickup/push), domain randomization (§12),
        sensor noise/bias, control-loop latency, init-state, action LPF,
        termination thresholds (a GATING effect), commands, standup.
      - NOT SWEEPABLE: reward scales (§9) do nothing at inference — the
        policy never sees rewards; measuring them needs RETRAINING.
        Architecture (obs_history_steps / observation_space / decimation /
        action_space) changes the input dim or needs retraining.
    Those excluded families are intentionally absent below.

WHY TWO PROCESSES
    Isaac Sim's SimulationApp is a singleton and env.close() does not fully
    release the stage/PhysX context, so a 2nd gym.make in one process HANGS
    (GPU drops to 0%, stuck at "Logging to file..."). This runs ONE point
    per process: orchestrator (no Isaac imports) -> spawns -> worker
    (--_worker, launches Sim, runs ONE point, writes JSON, exits).

REGIMES
    Different families need different base conditions; an axis declares one:
      quiet   : push off, pickup off, curriculum off, cmd=0   (the CEILING)
      pickup  : pickup ON,  rest off, cmd=0
      push    : push ON,    rest off, cmd=0
      standup : curriculum ON (kneel->stand), rest off, cmd=0
      walk    : commands SAMPLED (no cmd override), disturbances off
    DR is left enabled in every regime (the trained context); a DR axis
    pins ONE DR axis to a fixed value while the others keep their ranges,
    measuring that axis's marginal effect.

    The orchestrator runs ONE baseline per regime (reused across that
    regime's axes), always runs the `quiet` baseline as a global ceiling,
    and prints a DISTURBANCE-COST table: how much enabling each regime at
    default settings costs vs the quiet ceiling. That answers "what limits
    ep length" at a glance.

USAGE (from project root, e.g. <repo>/RL):

    # triage across all families (~10 axes + per-regime baselines):
    python scripts/rsl_rl/ep_len_impact.py `
        --task=Template-standup_curriculum-v0 `
        --checkpoint="...\\model_700.pt" --num_envs=1024 --headless

    # one family at a time (recommended for the big ones):
    ...  --axes dr            # validate the §12 impact tags
    ...  --axes pickup
    ...  --axes push sensor latency
    ...  --axes all           # everything (long; run overnight)

TIMING
    ~30-60s Sim startup + rollout per worker. At 1024 envs / 3000 steps a
    rollout is ~1-2 min, so ~2-3 min/point. `--axes all` is ~100 runs
    (~3-5 h); prefer per-group runs. 4096 envs just ~4x the rollout for no
    extra signal on most axes — use 1024-2048 unless an axis has very low
    event probability.

Measurement details (same as the pickup version):
  - ep length frozen on env.reset_terminated (true falls), NOT the
    wrapper's `dones` (long, includes time_outs).
  - NO early stop: full clamped max_steps every point so the survivor cap
    is identical and capped-mean ep length is comparable point-to-point.
  - max_steps clamped to one episode (no 2nd-episode contamination).
  - same seed + large num_envs everywhere -> non-swept DR is a held factor.

NOTE on assumed env behavior: the `standup` regime assumes the env's reset
path honors kneel_start_probability when curriculum_enable=True, and
`command` axes assume manual_cmd_override=False lets the env sample
commands per cfg. Verify once against your env if standup/command numbers
look off.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any

# No torch/isaaclab at module scope — orchestrator stays import-light and
# must NOT launch the Sim app. Heavy imports live in _run_worker().

WORKER_FLAG = "--_worker"


# ──────────────────────────────────────────────────────────────────
# Regimes: base disturbance overrides + whether to force cmd=0.
# Point overrides are applied AFTER the regime base, so a point can flip
# any enable (e.g. the pickup_enable=off point).
# ──────────────────────────────────────────────────────────────────
REGIMES: dict[str, dict[str, Any]] = {
    "quiet":   {"base": {"push_enable": False, "pickup_enable": False,
                         "curriculum_enable": False}, "cmd_zero": True},
    "pickup":  {"base": {"push_enable": False, "pickup_enable": True,
                         "curriculum_enable": False}, "cmd_zero": True},
    "push":    {"base": {"push_enable": True, "pickup_enable": False,
                         "curriculum_enable": False}, "cmd_zero": True},
    "standup": {"base": {"push_enable": False, "pickup_enable": False,
                         "curriculum_enable": True}, "cmd_zero": True},
    "walk":    {"base": {"push_enable": False, "pickup_enable": False,
                         "curriculum_enable": False}, "cmd_zero": False},
}


# ──────────────────────────────────────────────────────────────────
# Sweep definitions (pure stdlib)
# ──────────────────────────────────────────────────────────────────
@dataclass
class SweepPoint:
    label: str
    overrides: dict[str, Any] = field(default_factory=dict)  # {} == baseline


@dataclass
class SweepAxis:
    name: str
    group: str
    regime: str
    note: str
    points: list[SweepPoint]


def _baseline() -> SweepPoint:
    return SweepPoint("(baseline)", {})


def mk(label: str, **overrides: Any) -> SweepPoint:
    return SweepPoint(label, dict(overrides))


def pin(min_key: str, max_key: str, v: Any) -> dict[str, Any]:
    """Collapse a DR [min,max] range to a single fixed value v."""
    return {min_key: v, max_key: v}


_AXES: list[SweepAxis] = []


def axis(name, group, regime, note, points):
    _AXES.append(SweepAxis(name, group, regime, note, points))


# ════════════════════════════════════════════════════════════════════
# PICKUP  (regime: pickup)  — defaults mirrored from §8
# ════════════════════════════════════════════════════════════════════
axis("pickup_enable", "pickup", "pickup",
     "Pickup off vs on. 'off' should ~= quiet ceiling; the gap to baseline "
     "is pickup's total cost.",
     [mk("off", pickup_enable=False), _baseline()])

axis("pickup_probability", "pickup", "pickup",
     "Per-step mid-episode trigger rate (spawn pickups held at 0.5). base=5e-4.",
     [mk("0.0", pickup_probability=0.0), _baseline(),
      mk("0.001", pickup_probability=0.001), mk("0.002", pickup_probability=0.002),
      mk("0.004", pickup_probability=0.004)])

axis("pickup_initial_probability", "pickup", "pickup",
     "Fraction of resets straight into LIFT (spawn pickup). base=0.5.",
     [mk("0.0", pickup_initial_probability=0.0), mk("0.25", pickup_initial_probability=0.25),
      _baseline(), mk("0.75", pickup_initial_probability=0.75),
      mk("1.0", pickup_initial_probability=1.0)])

axis("pickup_target_tilt", "pickup", "pickup",
     "Max roll&pitch during HOLD (together). More tilt => harder recovery. base=0.60.",
     [mk("0.00", pickup_target_roll_max=0.0, pickup_target_pitch_max=0.0),
      mk("0.30", pickup_target_roll_max=0.30, pickup_target_pitch_max=0.30), _baseline(),
      mk("0.90", pickup_target_roll_max=0.90, pickup_target_pitch_max=0.90)])

axis("pickup_lift_height", "pickup", "pickup",
     "Lift height (mid-ep + spawn ranges together). Higher => bigger drop.",
     [mk("low", pickup_lift_height_min=0.10, pickup_lift_height_max=0.20,
         pickup_initial_lift_height_min=0.15, pickup_initial_lift_height_max=0.30),
      _baseline(),
      mk("high", pickup_lift_height_min=0.30, pickup_lift_height_max=0.50,
         pickup_initial_lift_height_min=0.40, pickup_initial_lift_height_max=0.70)])

axis("pickup_release_latency", "pickup", "pickup",
     "Feet-down -> full release window. base=(5,40) steps.",
     [mk("fast (1,5)", pickup_release_latency_min_steps=1, pickup_release_latency_max_steps=5),
      _baseline(),
      mk("slow (20,80)", pickup_release_latency_min_steps=20, pickup_release_latency_max_steps=80)])

axis("pickup_orient_authority", "pickup", "pickup",
     "Orient-PD gains + torque cap during pickup (together). base kp25/kd6/cap15.",
     [mk("weak", pickup_orient_kp=8.0, pickup_orient_kd=2.0, pickup_torque_cap=5.0),
      _baseline(),
      mk("strong", pickup_orient_kp=40.0, pickup_orient_kd=10.0, pickup_torque_cap=25.0)])

axis("pickup_z_dynamics", "pickup", "pickup",
     "Vertical-PD natural freq / damping. base omega=7.0, dr=1.2.",
     [mk("soft", pickup_z_omega=4.0, pickup_z_damping_ratio=0.8), _baseline(),
      mk("stiff", pickup_z_omega=12.0, pickup_z_damping_ratio=1.5)])

axis("pickup_hold_steps", "pickup", "pickup", "HOLD duration range. base=(80,300).",
     [mk("short (40,80)", pickup_hold_steps_min=40, pickup_hold_steps_max=80), _baseline(),
      mk("long (300,600)", pickup_hold_steps_min=300, pickup_hold_steps_max=600)])

axis("pickup_pd_gain_factor", "pickup", "pickup",
     "Per-event PD gain randomization width. base=(0.70,1.40).",
     [mk("tight", pickup_pd_gain_factor_min=0.95, pickup_pd_gain_factor_max=1.05), _baseline(),
      mk("wide", pickup_pd_gain_factor_min=0.5, pickup_pd_gain_factor_max=2.0)])

axis("pickup_contact_release_z", "pickup", "pickup",
     "Foot-z triggering early release during LOWER. base=0.010.",
     [mk("0.001", pickup_contact_release_z=0.001), _baseline(),
      mk("0.030", pickup_contact_release_z=0.030), mk("0.060", pickup_contact_release_z=0.060)])

axis("pickup_post_grace_steps", "pickup", "pickup",
     "Landing-settle window (also gates termination). base=50.  [GATING]",
     [mk("0", pickup_post_grace_steps=0), _baseline(),
      mk("100", pickup_post_grace_steps=100), mk("200", pickup_post_grace_steps=200)])

axis("pickup_termination_support_threshold", "pickup", "pickup",
     "Support threshold gating termination during pickup. base=0.3.  [GATING]",
     [mk("0.1", pickup_termination_support_threshold=0.1), _baseline(),
      mk("0.5", pickup_termination_support_threshold=0.5),
      mk("0.7", pickup_termination_support_threshold=0.7)])


# ════════════════════════════════════════════════════════════════════
# PUSH  (regime: push)  — §7
# ════════════════════════════════════════════════════════════════════
axis("push_enable", "push", "push", "Push off vs on. Gap = push's total cost.",
     [mk("off", push_enable=False), _baseline()])

axis("push_probability", "push", "push", "Per-step push trigger rate. base=4e-4.",
     [mk("2e-4", push_probability=0.0002), _baseline(),
      mk("1e-3", push_probability=0.001), mk("2e-3", push_probability=0.002)])

axis("push_force", "push", "push", "Push force range (min,max) together. base=(10,30) N.",
     [mk("light (5,15)", push_force_min=5.0, push_force_max=15.0), _baseline(),
      mk("heavy (20,50)", push_force_min=20.0, push_force_max=50.0),
      mk("very heavy (30,80)", push_force_min=30.0, push_force_max=80.0)])

axis("push_duration_steps", "push", "push", "Push duration. base=100 steps.",
     [mk("50", push_duration_steps=20),
      mk("50", push_duration_steps=30),
      mk("50", push_duration_steps=40)])


# ════════════════════════════════════════════════════════════════════
# DOMAIN RANDOMIZATION  (regime: quiet)  — §12. Pin ONE axis, others ranged.
# This is the measurement that validates / refutes the §12 IMPACT TAGS.
# ════════════════════════════════════════════════════════════════════
axis("dr_enable", "dr", "quiet", "ALL DR off vs on (total DR cost on balance).",
     [mk("DR off", dr_enable=False), _baseline()])

axis("floor_friction", "dr", "quiet",
     "Floor friction mult, pinned. Tag: HIGH. base range=(0.5,1.1). Low=slip.",
     [mk("pin 0.3 (OOD slip)", **pin("dr_floor_friction_min", "dr_floor_friction_max", 0.3)),
      mk("pin 0.5 (train min)", **pin("dr_floor_friction_min", "dr_floor_friction_max", 0.5)),
      _baseline(),
      mk("pin 1.1 (train max)", **pin("dr_floor_friction_min", "dr_floor_friction_max", 1.1))])

axis("mass_mult", "dr", "quiet",
     "Body mass mult, pinned. Tag: MED. base range=(0.88,1.12).",
     [mk("pin 0.88", **pin("dr_mass_mult_min", "dr_mass_mult_max", 0.88)), _baseline(),
      mk("pin 1.12", **pin("dr_mass_mult_min", "dr_mass_mult_max", 1.12)),
      mk("pin 1.25 (OOD heavy)", **pin("dr_mass_mult_min", "dr_mass_mult_max", 1.25))])

axis("com_offset", "dr", "quiet",
     "Per-link CoM offset max (±). Tag: MED. base=0.015 m.",
     [mk("0.0", dr_com_offset_max=0.0), _baseline(),
      mk("0.03", dr_com_offset_max=0.03), mk("0.05 (OOD)", dr_com_offset_max=0.05)])

axis("stiffness_mult", "dr", "quiet",
     "Kp mult, pinned. Tag: MED. base range=(0.85,1.15).",
     [mk("pin 0.7 (OOD)", **pin("dr_stiffness_mult_min", "dr_stiffness_mult_max", 0.7)),
      mk("pin 0.85", **pin("dr_stiffness_mult_min", "dr_stiffness_mult_max", 0.85)), _baseline(),
      mk("pin 1.15", **pin("dr_stiffness_mult_min", "dr_stiffness_mult_max", 1.15))])

axis("joint_friction_mult", "dr", "quiet",
     "Coulomb joint friction mult, pinned. Tag: MED. base range=(0.5,2.0).",
     [mk("pin 0.5", **pin("dr_joint_friction_mult_min", "dr_joint_friction_mult_max", 0.5)),
      _baseline(),
      mk("pin 2.0", **pin("dr_joint_friction_mult_min", "dr_joint_friction_mult_max", 2.0)),
      mk("pin 3.0 (OOD)", **pin("dr_joint_friction_mult_min", "dr_joint_friction_mult_max", 3.0))])

# ── tagged LOW — included to VERIFY the prior (expect small swings) ──
axis("damping_mult", "dr_low", "quiet",
     "Kd mult, pinned. Tag: LOW. base range=(0.7,1.3).",
     [mk("pin 0.7", **pin("dr_damping_mult_min", "dr_damping_mult_max", 0.7)), _baseline(),
      mk("pin 1.3", **pin("dr_damping_mult_min", "dr_damping_mult_max", 1.3))])

axis("armature_mult", "dr_low", "quiet",
     "Rotor inertia mult, pinned. Tag: LOW. base range=(0.7,1.3).",
     [mk("pin 0.7", **pin("dr_armature_mult_min", "dr_armature_mult_max", 0.7)), _baseline(),
      mk("pin 1.3", **pin("dr_armature_mult_min", "dr_armature_mult_max", 1.3))])

axis("dyn_friction_mult", "dr_low", "quiet",
     "Kinetic friction mult, pinned. Tag: LOW. base range=(0.5,2.0).",
     [mk("pin 0.5", **pin("dr_dyn_friction_mult_min", "dr_dyn_friction_mult_max", 0.5)), _baseline(),
      mk("pin 2.0", **pin("dr_dyn_friction_mult_min", "dr_dyn_friction_mult_max", 2.0))])

axis("visc_friction_mult", "dr_low", "quiet",
     "Viscous friction mult, pinned. Tag: LOW. base range=(0.5,2.0).",
     [mk("pin 0.5", **pin("dr_visc_friction_mult_min", "dr_visc_friction_mult_max", 0.5)), _baseline(),
      mk("pin 2.0", **pin("dr_visc_friction_mult_min", "dr_visc_friction_mult_max", 2.0))])

axis("voltage_factor", "dr_low", "quiet",
     "Battery voltage factor (scales v_limit), pinned. Tag: LOW. base range=(0.87,1.07).",
     [mk("pin 0.75 (OOD)", **pin("dr_voltage_factor_min", "dr_voltage_factor_max", 0.75)),
      mk("pin 0.87", **pin("dr_voltage_factor_min", "dr_voltage_factor_max", 0.87)), _baseline(),
      mk("pin 1.07", **pin("dr_voltage_factor_min", "dr_voltage_factor_max", 1.07))])

axis("effort_mult", "dr_low", "quiet",
     "Effort (torque budget) mult, pinned. Tag: LOW. base range=(0.9,1.0).",
     [mk("pin 0.7 (OOD weak)", **pin("dr_effort_mult_min", "dr_effort_mult_max", 0.7)),
      mk("pin 0.9", **pin("dr_effort_mult_min", "dr_effort_mult_max", 0.9)), _baseline(),
      mk("pin 1.0", **pin("dr_effort_mult_min", "dr_effort_mult_max", 1.0))])


# ════════════════════════════════════════════════════════════════════
# LATENCY  (regime: quiet)  — §12b. Tag: HIGH (loop dead time).
# ════════════════════════════════════════════════════════════════════
axis("action_delay", "latency", "quiet", "Action delay (steps), pinned. base range=(0,2).",
     [mk("pin 0", **pin("dr_action_delay_min", "dr_action_delay_max", 0)), _baseline(),
      mk("pin 2", **pin("dr_action_delay_min", "dr_action_delay_max", 2)),
      mk("pin 4 (OOD)", **pin("dr_action_delay_min", "dr_action_delay_max", 4))])

axis("obs_delay", "latency", "quiet", "Obs delay (steps), pinned. base range=(0,2).",
     [mk("pin 0", **pin("dr_obs_delay_min", "dr_obs_delay_max", 0)), _baseline(),
      mk("pin 2", **pin("dr_obs_delay_min", "dr_obs_delay_max", 2)),
      mk("pin 4 (OOD)", **pin("dr_obs_delay_min", "dr_obs_delay_max", 4))])

axis("loop_delay", "latency", "quiet",
     "Combined action+obs delay (round-trip dead time), both pinned to k.",
     [mk("pin 0/0", dr_action_delay_min=0, dr_action_delay_max=0,
         dr_obs_delay_min=0, dr_obs_delay_max=0), _baseline(),
      mk("pin 2/2", dr_action_delay_min=2, dr_action_delay_max=2,
         dr_obs_delay_min=2, dr_obs_delay_max=2),
      mk("pin 4/4 (OOD)", dr_action_delay_min=4, dr_action_delay_max=4,
         dr_obs_delay_min=4, dr_obs_delay_max=4)])


# ════════════════════════════════════════════════════════════════════
# SENSOR noise / bias  (regime: quiet)  — §12c / §12d. Tag: LOW (verify).
# ════════════════════════════════════════════════════════════════════
axis("noise_proj_grav", "sensor", "quiet", "Per-step proj-grav noise (±). base=0.05.",
     [mk("0.0", dr_obs_noise_proj_grav=0.0), _baseline(),
      mk("0.1", dr_obs_noise_proj_grav=0.1), mk("0.2", dr_obs_noise_proj_grav=0.2)])

axis("noise_ang_vel", "sensor", "quiet", "Per-step ang-vel noise (±). base=0.15.",
     [mk("0.0", dr_obs_noise_ang_vel=0.0), _baseline(),
      mk("0.3", dr_obs_noise_ang_vel=0.3), mk("0.6", dr_obs_noise_ang_vel=0.6)])

axis("noise_joint_vel", "sensor", "quiet", "Per-step joint-vel noise (±). base=0.6.",
     [mk("0.0", dr_obs_noise_joint_vel=0.0), _baseline(),
      mk("1.2", dr_obs_noise_joint_vel=1.2)])

axis("noise_joint_pos", "sensor", "quiet", "Per-step joint-pos noise (±). base=0.008.",
     [mk("0.0", dr_obs_noise_joint_pos=0.0), _baseline(),
      mk("0.03", dr_obs_noise_joint_pos=0.03)])

axis("bias_joint_pos", "sensor", "quiet", "Per-episode joint-pos bias max (±). base=0.05.",
     [mk("0.0", dr_joint_pos_bias_max=0.0), _baseline(),
      mk("0.1", dr_joint_pos_bias_max=0.1)])

axis("bias_proj_grav", "sensor", "quiet", "Per-episode proj-grav bias max (±). base=0.02.",
     [mk("0.0", dr_proj_grav_bias_max=0.0), _baseline(),
      mk("0.05", dr_proj_grav_bias_max=0.05)])

axis("bias_ang_vel", "sensor", "quiet", "Per-episode ang-vel bias max (±). base=0.03.",
     [mk("0.0", dr_ang_vel_bias_max=0.0), _baseline(),
      mk("0.08", dr_ang_vel_bias_max=0.08)])


# ════════════════════════════════════════════════════════════════════
# TERMINATION thresholds  (regime: quiet)  — §10.  [GATING, not stability]
# These change WHEN a fall is counted, so a swing = measurement boundary,
# not robustness. Read separately from the disturbance/DR families.
# ════════════════════════════════════════════════════════════════════
axis("max_tilt_rad", "termination", "quiet", "Tilt termination limit. base=1.0 rad.  [GATING]",
     [mk("0.7", max_tilt_rad=0.7), _baseline(), mk("1.3", max_tilt_rad=1.3)])

axis("min_height", "termination", "quiet", "Min-height termination. base=0.30 m.  [GATING]",
     [mk("0.25", min_height=0.25), _baseline(), mk("0.34", min_height=0.34)])

axis("max_hip_roll_rad", "termination", "quiet", "Hip-roll split limit. base=1.8.  [GATING]",
     [mk("1.2", max_hip_roll_rad=1.2), _baseline(), mk("2.4", max_hip_roll_rad=2.4)])

axis("max_hip_pitch_split_rad", "termination", "quiet", "Hip-pitch split limit. base=1.8.  [GATING]",
     [mk("1.2", max_hip_pitch_split_rad=1.2), _baseline(), mk("2.4", max_hip_pitch_split_rad=2.4)])

axis("max_knee_split_rad", "termination", "quiet", "Knee split limit. base=1.8.  [GATING]",
     [mk("1.2", max_knee_split_rad=1.2), _baseline(), mk("2.4", max_knee_split_rad=2.4)])


# ════════════════════════════════════════════════════════════════════
# INIT state  (regime: quiet)  — §6
# ════════════════════════════════════════════════════════════════════
axis("init_base_orient_noise", "init", "quiet",
     "Base roll/pitch noise at reset. base=0.08 rad.",
     [mk("0.0", init_base_orient_noise_rad=0.0), _baseline(),
      mk("0.15", init_base_orient_noise_rad=0.15), mk("0.25", init_base_orient_noise_rad=0.25)])


# ════════════════════════════════════════════════════════════════════
# ACTION processing  (regime: quiet)  — §11 (read live by the policy loop)
# ════════════════════════════════════════════════════════════════════
axis("action_lpf_alpha", "action", "quiet",
     "Action low-pass alpha (higher=less smoothing; 1.0=off). base=0.6.",
     [mk("0.3 (heavy smooth)", action_lpf_alpha=0.3), _baseline(),
      mk("0.9", action_lpf_alpha=0.9), mk("1.0 (no filter)", action_lpf_alpha=1.0)])

axis("action_scale", "action", "quiet",
     "All three action scales scaled together. OOD-RISKY: the policy's "
     "output units were calibrated at training scale; off-scale = different "
     "torque per output. base hip_roll/pitch/knee = 0.30/0.75/2.05.",
     [mk("0.7x", train_action_scale_hip_roll=0.21, train_action_scale_hip_pitch=0.525,
         train_action_scale_knee=1.435, real_action_scale_hip_roll=0.21,
         real_action_scale_hip_pitch=0.525, real_action_scale_knee=1.435),
      _baseline(),
      mk("1.3x", train_action_scale_hip_roll=0.39, train_action_scale_hip_pitch=0.975,
         train_action_scale_knee=2.665, real_action_scale_hip_roll=0.39,
         real_action_scale_hip_pitch=0.975, real_action_scale_knee=2.665)])


# ════════════════════════════════════════════════════════════════════
# STANDUP  (regime: standup)  — §6
# ════════════════════════════════════════════════════════════════════
axis("kneel_start_probability", "standup", "standup",
     "Fraction of resets starting in the kneel pose (must stand up). base=0.5.",
     [mk("0.0", kneel_start_probability=0.0), mk("0.25", kneel_start_probability=0.25),
      _baseline(), mk("0.75", kneel_start_probability=0.75),
      mk("1.0", kneel_start_probability=1.0)])

axis("standup_grace_seconds", "standup", "standup",
     "Grace window before standup failure can terminate. base=4.0 s.",
     [mk("2.0", standup_grace_seconds=2.0), _baseline(), mk("6.0", standup_grace_seconds=6.0)])

axis("stand_target_height", "standup", "standup",
     "Standup target height (ramp top). base=0.42 m.",
     [mk("0.38", stand_target_height=0.38), _baseline(), mk("0.50", stand_target_height=0.50)])


# ════════════════════════════════════════════════════════════════════
# COMMAND / WALK  (regime: walk)  — §5
# ════════════════════════════════════════════════════════════════════
axis("command_vx_range", "command", "walk",
     "Vx command range (±). base=±0.20 m/s. Wider => faster walking demand.",
     [mk("±0.1", vx_cmd_min=-0.1, vx_cmd_max=0.1), _baseline(),
      mk("±0.3", vx_cmd_min=-0.3, vx_cmd_max=0.3),
      mk("±0.5", vx_cmd_min=-0.5, vx_cmd_max=0.5)])

axis("command_yaw_range", "command", "walk",
     "Yaw-rate command range (±). base=±0.5 rad/s.",
     [mk("±0.25", yaw_rate_cmd_min=-0.25, yaw_rate_cmd_max=0.25), _baseline(),
      mk("±1.0", yaw_rate_cmd_min=-1.0, yaw_rate_cmd_max=1.0)])

axis("command_vx_zero_prob", "command", "walk",
     "P(vx command = 0). 1.0 ~= balance; 0.0 = always moving. base=0.3.",
     [mk("1.0 (still)", vx_cmd_zero_probability=1.0), _baseline(),
      mk("0.0 (always move)", vx_cmd_zero_probability=0.0)])

axis("gait_freq", "command", "walk", "Gait clock frequency. base=2.0 Hz.",
     [mk("1.5", gait_freq=1.5), _baseline(), mk("2.5", gait_freq=2.5)])


# ── Index axes by name + group; define curated triage set ───────────
ALL_AXES: dict[str, SweepAxis] = {ax.name: ax for ax in _AXES}
GROUPS: dict[str, list[str]] = {}
for ax in _AXES:
    GROUPS.setdefault(ax.group, []).append(ax.name)

CURATED = [
    "pickup_enable", "push_probability", "dr_enable", "floor_friction",
    "loop_delay", "init_base_orient_noise", "kneel_start_probability",
    "max_tilt_rad", "action_lpf_alpha", "command_vx_range",
]


# ──────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────
def apply_overrides(cfg, overrides: dict[str, Any]) -> None:
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise AttributeError(f"Cfg has no attribute '{k}' — typo in overrides?")
        setattr(cfg, k, v)


def parse_checkpoint_iter(checkpoint_path: str) -> int:
    fname = os.path.basename(checkpoint_path)
    return int(fname.replace("model_", "").replace(".pt", ""))


# ══════════════════════════════════════════════════════════════════
# WORKER MODE — launch Sim once, run ONE point, write JSON, exit.
# ══════════════════════════════════════════════════════════════════
def _run_worker() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser()
    p.add_argument(WORKER_FLAG, action="store_true")
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num_envs", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--cmd_zero", type=int, default=1)
    p.add_argument("--base_overrides", type=str, default="{}")
    p.add_argument("--point_label", type=str, default="(point)")
    p.add_argument("--point_overrides", type=str, default="{}")
    p.add_argument("--result_json", type=str, required=True)
    AppLauncher.add_app_launcher_args(p)  # --headless, --device, ...
    a = p.parse_args()

    app_launcher = AppLauncher(a)
    simulation_app = app_launcher.app

    import torch
    import gymnasium as gym
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isaaclab.utils.math import quat_rotate_inverse
    import isaaclab_tasks  # noqa: F401
    import RL.tasks        # noqa: F401

    device = a.device or "cuda:0"
    base_overrides = json.loads(a.base_overrides)
    point_overrides = json.loads(a.point_overrides)
    cmd_zero = bool(a.cmd_zero)

    base_cfg = load_cfg_from_registry(a.task, "env_cfg_entry_point")
    control_hz = 1.0 / (base_cfg.sim.dt * base_cfg.decimation)
    episode_len_steps = int(round(base_cfg.episode_length_s * control_hz))
    max_steps = min(a.max_steps, episode_len_steps)

    cfg = copy.deepcopy(base_cfg)
    cfg.scene.num_envs = a.num_envs
    cfg.sim.device = device
    cfg.seed = a.seed
    apply_overrides(cfg, base_overrides)    # regime base
    apply_overrides(cfg, point_overrides)   # the swept point

    env = gym.make(a.task, cfg=cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=None)
    if cmd_zero:
        env.unwrapped.manual_cmd_override = True
        env.unwrapped.manual_vx = 0.0
        env.unwrapped.manual_yaw_rate = 0.0

    agent_cfg = load_cfg_from_registry(a.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = device
    agent_cfg.seed = a.seed
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(a.checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy(device=device)

    n = a.num_envs
    dev = torch.device(device)
    first_term = torch.full((n,), -1, dtype=torch.long, device=dev)
    sum_up = torch.zeros(n, device=dev)
    sum_bh = torch.zeros(n, device=dev)
    min_bh = torch.full((n,), float("inf"), device=dev)
    step_count = torch.zeros(n, device=dev)

    obs, _ = env.reset()
    obs_t = obs["policy"] if isinstance(obs, dict) else obs

    print(f"[worker] point='{a.point_label}' cmd_zero={cmd_zero} "
          f"base={base_overrides} pt={point_overrides or '(baseline)'} "
          f"num_envs={n} max_steps={max_steps}", flush=True)
    log_every = max(1, max_steps // 10)
    t0 = time.time()

    for step in range(max_steps):
        with torch.inference_mode():
            actions = policy(obs_t)
        obs, _, _dones, _ = env.step(actions)
        obs_t = obs["policy"] if isinstance(obs, dict) else obs

        terminated = env.unwrapped.reset_terminated.bool()  # NOT _dones
        alive = first_term == -1

        u = env.unwrapped
        proj_grav = quat_rotate_inverse(u.robot.data.root_quat_w, u._gravity_vec)
        upright = -proj_grav[:, 2]
        base_h = u.robot.data.root_pos_w[:, 2]
        if alive.any():
            sum_up[alive] += upright[alive]
            sum_bh[alive] += base_h[alive]
            min_bh[alive] = torch.minimum(min_bh[alive], base_h[alive])
            step_count[alive] += 1.0

        new_term = alive & terminated
        if new_term.any():
            first_term[new_term] = step

        if (step + 1) % log_every == 0:
            print(f"[worker]   step {step+1}/{max_steps}  "
                  f"dead={(first_term >= 0).sum().item()}/{n}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    final_step = max_steps
    surv = first_term < 0
    ep = first_term.clone().float()
    ep[surv] = float(final_step)
    sc = step_count.clamp(min=1.0)

    m: dict[str, float] = {}
    m["ep_len/mean_steps"] = ep.mean().item()
    m["ep_len/median_steps"] = torch.quantile(ep, 0.50).item()
    m["ep_len/p10_steps"] = torch.quantile(ep, 0.10).item()
    m["ep_len/p25_steps"] = torch.quantile(ep, 0.25).item()
    m["ep_len/mean_seconds"] = m["ep_len/mean_steps"] / control_hz
    m["ep_len/survivor_frac"] = surv.float().mean().item()
    for t_s in (5, 10, 15, 20, 25, 30):
        thr = int(t_s * control_hz)
        if thr <= final_step:
            m[f"survival/at_{t_s}s"] = (ep >= thr).float().mean().item()
    died = ep[~surv]
    m["ttf/num_died"] = float(died.numel())
    if died.numel() > 0:
        m["ttf/mean_steps"] = died.mean().item()
        m["ttf/p50_steps"] = torch.quantile(died, 0.50).item()
    m["balance/uprightness_mean"] = (sum_up / sc).mean().item()
    m["balance/base_height_min_mean"] = min_bh.mean().item()
    m["_max_steps_used"] = float(max_steps)
    m["_control_hz"] = float(control_hz)

    env.close()
    with open(a.result_json, "w") as f:
        json.dump(m, f)
    print(f"[worker] wrote {a.result_json}: "
          f"mean_ep={m['ep_len/mean_steps']:.1f}/{max_steps} "
          f"({m['ep_len/mean_seconds']:.2f}s)  [{time.time()-t0:.0f}s]", flush=True)
    simulation_app.close()


# ══════════════════════════════════════════════════════════════════
# ORCHESTRATOR MODE
# ══════════════════════════════════════════════════════════════════
def print_axis_table(ax: SweepAxis, rows, baseline_mean: float) -> None:
    print(f"\n{'='*84}\n  AXIS: {ax.name}   [group={ax.group}, regime={ax.regime}]\n"
          f"  {ax.note}\n{'='*84}")
    hdr = (f"  {'point':24s} {'mean_steps':>11s} {'Δ vs base':>10s} "
           f"{'mean_s':>8s} {'surv_full':>10s} {'surv@10s':>9s} {'upright':>8s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for label, m in rows:
        ms = m["ep_len/mean_steps"]
        print(f"  {label:24s} {ms:11.1f} {ms - baseline_mean:+10.1f} "
              f"{m['ep_len/mean_seconds']:8.2f} "
              f"{m.get('ep_len/survivor_frac', float('nan')):10.3f} "
              f"{m.get('survival/at_10s', float('nan')):9.3f} "
              f"{m.get('balance/uprightness_mean', float('nan')):8.3f}")


def print_cost_table(regime_baselines: dict[str, float]) -> None:
    q = regime_baselines.get("quiet")
    print(f"\n{'#'*84}\n  DISTURBANCE COST — regime baseline (default settings) vs quiet ceiling\n"
          f"  (how much enabling each regime at default costs in mean ep length)\n{'#'*84}")
    print(f"  {'regime':12s} {'baseline_mean':>14s} {'Δ vs quiet':>12s}")
    print("  " + "-" * 40)
    for rg in ["quiet", "pickup", "push", "standup", "walk"]:
        if rg in regime_baselines:
            v = regime_baselines[rg]
            d = (v - q) if q is not None else float("nan")
            print(f"  {rg:12s} {v:14.1f} {d:+12.1f}")


def print_ranking(axis_stats) -> None:
    print(f"\n{'#'*84}\n  SENSITIVITY RANKING — within-axis swing in mean ep length\n"
          f"  (swing = max_mean - min_mean across the axis's points;\n"
          f"   cross-FAMILY level differences are in the DISTURBANCE COST table above)\n{'#'*84}")
    hdr = (f"  {'axis':32s} {'group':12s} {'baseline':>9s} {'min':>8s} "
           f"{'max':>8s} {'swing':>8s}   best")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for s in sorted(axis_stats, key=lambda x: x["swing"], reverse=True):
        print(f"  {s['axis']:32s} {s['group']:12s} {s['baseline']:9.1f} "
              f"{s['min']:8.1f} {s['max']:8.1f} {s['swing']:8.1f}   {s['best_label']}")


def _run_orchestrator() -> None:
    p = argparse.ArgumentParser(description="Episode-length sensitivity sweep (orchestrator).")
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num_envs", type=int, default=2048,
                   help="Per point. 1024-2048 is plenty; 4096 just slows rollout.")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--axes", nargs="+", default=None,
                   help="Axis names, group names "
                        f"({sorted(GROUPS)}), or 'all'. Default=curated triage.")
    p.add_argument("--out_dir", type=str, default="logs/ep_len_impact")
    args, passthrough = p.parse_known_args()  # --headless etc. pass through

    # Resolve axis tokens (axis name | group name | 'all').
    if args.axes is None:
        tokens = list(CURATED)
    elif len(args.axes) == 1 and args.axes[0] == "all":
        tokens = list(ALL_AXES.keys())
    else:
        tokens = args.axes
    axis_names: list[str] = []
    for tok in tokens:
        if tok in GROUPS:
            axis_names += GROUPS[tok]
        elif tok in ALL_AXES:
            axis_names.append(tok)
        else:
            print(f"[sweep] unknown axis/group '{tok}'. "
                  f"axes={list(ALL_AXES.keys())}  groups={list(GROUPS)}")
    # de-dup, preserve order
    seen: set[str] = set()
    axis_names = [a for a in axis_names if not (a in seen or seen.add(a))]
    if not axis_names:
        print("[sweep] nothing to run."); return

    ckpt_iter = parse_checkpoint_iter(args.checkpoint)
    out_dir = os.path.join(args.out_dir, f"iter_{ckpt_iter}")
    os.makedirs(out_dir, exist_ok=True)

    regimes_used = {"quiet"} | {ALL_AXES[a].regime for a in axis_names}
    n_points = sum(1 for a in axis_names for pt in ALL_AXES[a].points if pt.overrides)
    n_runs = len(regimes_used) + n_points
    print(f"[sweep] num_envs={args.num_envs}  axes({len(axis_names)})={axis_names}")
    print(f"[sweep] regimes={sorted(regimes_used)}  -> {n_runs} worker processes "
          f"(one Sim launch each).")

    this_file = os.path.abspath(__file__)
    run_idx = {"i": 0}

    def run_point(label: str, regime: str, overrides: dict[str, Any]) -> dict[str, float]:
        run_idx["i"] += 1
        result_path = os.path.join(out_dir, f"_result_{run_idx['i']:03d}.json")
        reg = REGIMES[regime]
        cmd = [
            sys.executable, this_file, WORKER_FLAG,
            "--task", args.task, "--checkpoint", args.checkpoint,
            "--num_envs", str(args.num_envs), "--seed", str(args.seed),
            "--max_steps", str(args.max_steps), "--device", args.device,
            "--cmd_zero", str(1 if reg["cmd_zero"] else 0),
            "--base_overrides", json.dumps(reg["base"]),
            "--point_label", label, "--point_overrides", json.dumps(overrides),
            "--result_json", result_path,
        ] + passthrough
        print(f"\n[sweep] ({run_idx['i']}/{n_runs}) regime={regime} -> {label}")
        t0 = time.time()
        ret = subprocess.run(cmd)
        if ret.returncode != 0 or not os.path.exists(result_path):
            raise RuntimeError(f"worker failed (rc={ret.returncode}) for '{label}' "
                               f"[regime={regime}]. Re-run with {WORKER_FLAG} to debug.")
        with open(result_path) as f:
            m = json.load(f)
        os.remove(result_path)
        print(f"[sweep]   done {time.time()-t0:.0f}s  mean_ep={m['ep_len/mean_steps']:.1f}")
        return m

    sweep_t0 = time.time()

    # One baseline per regime (reused for that regime's empty-override points).
    regime_baseline: dict[str, dict[str, float]] = {}
    for rg in ["quiet"] + sorted(regimes_used - {"quiet"}):  # quiet first (the ceiling)
        regime_baseline[rg] = run_point(f"(baseline:{rg})", rg, {})
    regime_baseline_mean = {k: v["ep_len/mean_steps"] for k, v in regime_baseline.items()}
    max_steps_used = int(regime_baseline["quiet"]["_max_steps_used"])

    results: dict[str, dict[str, dict[str, float]]] = {}
    axis_stats: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    for name in axis_names:
        ax = ALL_AXES[name]
        rows = []
        results[name] = {}
        for pt in ax.points:
            m = regime_baseline[ax.regime] if not pt.overrides else run_point(pt.label, ax.regime, pt.overrides)
            rows.append((pt.label, m))
            results[name][pt.label] = m
            csv_rows.append({"axis": name, "group": ax.group, "regime": ax.regime,
                             "point": pt.label, "overrides": json.dumps(pt.overrides), **m})
        base_mean = regime_baseline_mean[ax.regime]
        means = {lbl: m["ep_len/mean_steps"] for lbl, m in rows}
        axis_stats.append({
            "axis": name, "group": ax.group, "regime": ax.regime, "baseline": base_mean,
            "min": min(means.values()), "max": max(means.values()),
            "swing": max(means.values()) - min(means.values()),
            "best_label": max(means, key=means.get),
        })
        print_axis_table(ax, rows, base_mean)

    print_cost_table(regime_baseline_mean)
    print_ranking(axis_stats)

    # ── Save artifacts ──────────────────────────────────────────────
    json_path = os.path.join(out_dir, "ep_len_impact.json")
    with open(json_path, "w") as f:
        json.dump({
            "task": args.task, "checkpoint": args.checkpoint, "ckpt_iter": ckpt_iter,
            "num_envs": args.num_envs, "max_steps": max_steps_used, "seed": args.seed,
            "regime_baseline_mean": regime_baseline_mean,
            "axis_stats": axis_stats, "results": results,
        }, f, indent=2)

    csv_path = os.path.join(out_dir, "ep_len_impact.csv")
    fixed = ["axis", "group", "regime", "point", "overrides"]
    all_keys: set[str] = set()
    for r in csv_rows:
        all_keys.update(r.keys())
    fieldnames = fixed + sorted(k for k in all_keys if k not in fixed)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)

    print(f"\n[sweep] total wall time {time.time()-sweep_t0:.0f}s")
    print(f"[sweep] wrote:\n  {json_path}\n  {csv_path}")
    print("\n[sweep] how to read it:")
    print("  - DISTURBANCE COST = which regime (pickup/push/standup/walk) costs the most")
    print("    mean ep length vs the quiet ceiling. That's the headline limiter.")
    print("  - SENSITIVITY RANKING = within a family, which knob moves ep length most.")
    print("  - group 'dr'/'dr_low' validates the §12 HIGH/MED/LOW impact tags by")
    print("    measurement: a big swing on a LOW-tagged axis means the prior is wrong.")
    print("  - [GATING] axes (termination/*, pickup_post_grace, *_support_threshold)")
    print("    change WHEN a fall is counted — read as boundary sensitivity, not stability.")


if __name__ == "__main__":
    if WORKER_FLAG in sys.argv:
        _run_worker()
    else:
        _run_orchestrator()