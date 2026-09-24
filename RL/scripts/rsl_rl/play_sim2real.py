# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""play_sim2real.py — single-env sim2real failure-analysis rollout.

Fork of play_log.py, reshaped to reproduce the DEPLOY scenario of
control/balance_log.py as closely as sim allows:

    - 1 env (default), NO pickup (all three pickup paths forced off).
    - Deploy contract gains by default (balance_log.py PRESET_KP/KD):
        hip_roll 6.0/0.3, hip_pitch 20.0/1.4, knee 30.0/1.5, LPF alpha 0.6.
      Any env.servo_* / env.action_lpf_alpha you pass on the CLI wins.
    - Commands pinned to vx=0, yaw_rate=0 (manual_cmd_override), matching
      balance_log's hard-zeroed cmd slots — the 1000-step resampler is
      thereby disabled too.
    - Gait clock pinned to --gait_phase_init (default pi) on EVERY reset,
      including auto-resets after falls (training randomizes phase per
      episode). VERIFIED 08-28: phase pi makes sim lead with the LEFT leg
      (leg1), matching every real handover; phase 0 leads RIGHT (leg2).
      Hardware starts its clock at 0 yet leads LEFT — unresolved (mirrored
      hw contract vs plant asymmetry; see memory sim2real-gait-phase-flip).
      The pin is a _reset_idx wrap, so the post-reset obs frame already
      carries the pinned sin/cos.
    - Spawn noise scaled by --init_noise_scale (default 0.0 = clean
      handover: exact spawn pose, zero velocity — the robot is PD-held at
      spawn when 'b' is pressed on the real rig).
    - Domain randomization resamples on every reset (training behavior,
      dr_enable default True) — press R repeatedly to sweep plant draws;
      every fall auto-resets into a fresh draw and prints a one-line DR
      summary so failures can be correlated with the draw offline.

Video recording and the B/H npz run-logger are identical to play_log.py
(same LOG_CHANNELS as balance_log.py, so sim and hardware runs feed the
same offline analysis).

Example (mirrors the usual play_log invocation):

    $env:CUDA_VISIBLE_DEVICES="2"; python .\\scripts\\rsl_rl\\play_sim2real.py `
      --task=Template-standup_curriculum-v0 `
      --checkpoint="...\\model_3730.pt" `
      --video --video_length 3000 --seed=0 `
      env.sat_term_threshold=0.5 env.sat_term_tau_s=0.5

Keys (Isaac Sim window focused):
    R = reset (fresh DR draw, gait clock re-pinned)
    B = reset + start log run
    H = stop log run + save npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Sim2real failure-analysis rollout (single env, no pickup).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during rollout.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate (default 1).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# ── Sim2real scenario args ─────────────────────────────────────────────────────
parser.add_argument(
    "--gait_phase_init",
    type=float,
    default=3.14159265,
    help="Gait clock phase (rad) pinned at every reset. VERIFIED 08-28: pi "
    "makes sim step LEFT (leg1) first, matching every real handover; 0.0 "
    "steps RIGHT (leg2) first. NOTE hardware starts its clock at 0 and "
    "still leads LEFT — that open discrepancy (mirrored contract vs plant "
    "asymmetry) is pending the HOLD-mode knee push-test on the rig.",
)
parser.add_argument(
    "--init_noise_scale",
    type=float,
    default=0.0,
    help="Scale on spawn-state noise (joint pos, base lin/ang vel, base orient). "
    "0.0 = clean PD-held handover pose (default); 1.0 = training spawn noise.",
)
parser.add_argument(
    "--mirror_lateral_imu",
    action="store_true",
    default=False,
    help="Replicate the pre-fix HARDWARE IMU bug (docs/imu-sign-check.md "
    "08-28): negate pg_y / gyro_x / gyro_z in every obs frame fed to the "
    "policy — the exact y-mirror the wrong ROLL/GYRO_X/GYRO_Z signs "
    "applied on the robot. Prediction: with --gait_phase_init 0 this flips "
    "sim to LEFT-leg-first, reproducing every real handover to date.",
)
parser.add_argument(
    "--handover",
    type=str,
    choices=["training", "deploy"],
    default="training",
    help="Episode-start obs contract after R/B resets. 'training' (default) "
    "= the env's own reset return: dynamic obs channels zeroed and primed "
    "x5 into history (what every training episode starts with). 'deploy' = "
    "real sensor frame replicated x5 (what balance_log.py hands the policy "
    "at 'b'). p12.2 finding 08-29: these produce entirely different opening "
    "actions from the same physical state.",
)
parser.add_argument(
    "--inject_npz",
    type=str,
    default=None,
    help="p12.3 arm A: path to a HARDWARE npz. After every reset the sim "
    "state is overwritten with the fork state read from this log at "
    "--inject_time (root attitude from raw pg = logged−0.089 trim, body "
    "ang vel, joint pos/vel; root z set so the lower toe sits 4 mm above "
    "ground; zero lin vel), and the policy-side buffers (last_actions x3, "
    "action LPF state, action/obs delay buffers, gait phase, obs history) "
    "are primed from the same log. Overrides --handover priming.",
)
parser.add_argument(
    "--inject_time",
    type=float,
    default=0.18,
    help="Time (s, from first NN tick) in --inject_npz to inject (default "
    "0.18 = the p12.3 post-arrest fork state).",
)
parser.add_argument(
    "--inject_vx",
    type=float,
    default=0.0,
    help="World-frame x (fore-aft) root linear velocity (m/s) at injection. "
    "Hardware does not log root lin vel, so the base injection assumes "
    "quasi-static (0). Negative = backward. Sweep this to find the residual "
    "COM velocity that would make the injected state unrecoverable.",
)
parser.add_argument(
    "--inject_history",
    type=str,
    choices=["log", "replicate"],
    default="log",
    help="Obs-history contract at injection. 'log' (arm A1) = the K "
    "preceding logged frames + logged last_actions — the true "
    "counterfactual. 'replicate' (arm A2) = the injected frame copied "
    "into all K slots (naive). Dynamics state is identical in both.",
)
parser.add_argument(
    "--autolog_steps",
    type=int,
    default=0,
    help="If > 0: headless auto-logging — behave as if B were pressed at "
    "startup, save the npz after this many logged ticks (H), and exit. "
    "No keyboard needed; combine with --headless and --no_obs_summary. "
    "0 (default) = interactive B/H behavior, unchanged.",
)
parser.add_argument(
    "--autolog_runs",
    type=int,
    default=1,
    help="With --autolog_steps: number of reset->log->save cycles before "
    "exiting (each cycle is a fresh env.reset / DR draw, one npz each).",
)
# ── Terminal observation summary args ──────────────────────────────────────────
parser.add_argument(
    "--no_obs_summary",
    action="store_true",
    default=False,
    help="Disable the 2Hz terminal observation summary panel.",
)
parser.add_argument(
    "--watch_env",
    type=int,
    default=0,
    help="Which env index to display in the summary panel (default 0).",
)
parser.add_argument(
    "--summary_hz",
    type=float,
    default=2.0,
    help="Refresh rate of the terminal summary panel (default 2 Hz).",
)
# ── Ball-pad compliance (the PER-ENV path), as a launch-time pin ─────────────
# These write the cfg a moment before the env is built, so they decide whether
# a pad exists at all — which the N/M/COMMA/PERIOD hotkeys cannot, because the
# pad is constructed in RlEnv.__init__.  Passing --pad_enable with no
# --pad_stiffness builds an INERT pad (k=0) that the hotkeys can then tune from
# zero, which is the intended by-feel workflow.
parser.add_argument(
    "--pad_enable",
    action="store_true",
    default=False,
    help="Build the per-env ball-pad model (pad_contact.PadContact). Required "
    "before the N/M/COMMA/PERIOD hotkeys can do anything: the pad is "
    "constructed in RlEnv.__init__, so a hotkey cannot create it.",
)
parser.add_argument(
    "--pad_stiffness",
    type=float,
    default=0.0,
    help="Pin the pad to one floor-normal stiffness (N/m) for every env; "
    "0.0 keeps the band (default). 5e3..5e4 is the 09-12 useful range.",
)
parser.add_argument(
    "--pad_stiffness_min",
    type=float,
    default=None,
    help="Lower rung of the per-env pad DR band, N/m (implies --pad_enable).",
)
parser.add_argument(
    "--pad_stiffness_max",
    type=float,
    default=None,
    help="Upper rung of the per-env pad DR band, N/m (implies --pad_enable).",
)
parser.add_argument(
    "--pad_damping_ratio",
    type=float,
    default=None,
    help="Pad damping as a ratio of its own mode with pad_effective_mass "
    "(default: the cfg's 0.7). Capped at pad_damping_stability_factor*m/dt.",
)
parser.add_argument(
    "--pad_thickness",
    type=float,
    default=None,
    help="Pad relaxed thickness (m). Must exceed the sag it will carry "
    "(m*g/k) or the ball bottoms out on its rigid contact and the knob "
    "goes inert. Default: the cfg's 2 mm.",
)
# ───────────────────────────────────────────────────────────────────────────────

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import carb

try:
    import omni.appwindow
except ModuleNotFoundError:
    # Headless kit has no appwindow module. Only KeyboardController needs it,
    # and --autolog_steps runs use _NullKeyboard instead.
    omni = None  # noqa: F811 — KeyboardController must not be constructed
import math
import os
import time
import traceback

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import quat_apply, quat_rotate_inverse

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import RL.tasks  # noqa: F401


# ═════════════════════════════════════════════════════════════════════════════
# Deploy contract — the plant/loop constants balance_log.py runs under.
# Applied as DEFAULTS: any env.<key>=... passed on the CLI wins (we scan
# hydra_args for an explicit override before touching the cfg field).
# Source: control/balance_log.py PRESET_KP / PRESET_KD (config.REAL_KP/KD,
# joint-space) and MotorBridge(action_lpf_alpha=0.6).
# ═════════════════════════════════════════════════════════════════════════════
BALANCE_CONTRACT = {
    "servo_kp_hip_roll": 6.0,
    "servo_kd_hip_roll": 0.3,
    "servo_kp_hip_pitch": 20.0,
    "servo_kd_hip_pitch": 1.4,
    "servo_kp_knee": 30.0,
    "servo_kd_knee": 1.5,
    "action_lpf_alpha": 0.6,
}


def _hydra_overrides(key: str) -> bool:
    """True if the user explicitly passed env.<key>=... on the CLI."""
    prefix = f"env.{key}="
    return any(a.startswith(prefix) or a.startswith("+" + prefix) for a in hydra_args)


# ═════════════════════════════════════════════════════════════════════════════
# Run logger — mirrors balance.py LOG_CHANNELS exactly so the same offline
# analysis script can ingest sim and hardware runs side-by-side.
# (Identical to play_log.py — see there for the sim-vs-hardware reconciliation
# notes: motor_temp zeros, tau in JOINT frame, tau_cmd == recomputed PD law.)
#
# Trigger is identical to balance.py:
#   B  -> env.reset() + start a fresh log buffer
#   H  -> stop logging, dump to <checkpoint>_simlog_<timestamp>.npz in CWD
#   quit while logging -> buffer is flushed on exit so the run isn't lost
# ═════════════════════════════════════════════════════════════════════════════

LOG_CHANNELS = [
    'time',            # seconds since logging started
    'obs',             # (36,) raw NN-frame observation (newest frame)
    'action',          # (6,)  clipped action that was applied this tick
    'target',          # (6,)  joint targets: default_pos + action * scale
    'sim_joint_pos',   # (6,)  measured joint pos (sim frame, rad)
    'sim_joint_vel',   # (6,)  measured joint vel (sim frame, rad/s)
    'motor_tau_cmd',   # (6,)  applied torque (sim has cmd == meas)
    'motor_tau_meas',  # (6,)  applied torque (joint frame, not motor frame)
    'motor_temp',      # (6,)  zeros in sim
    'proj_grav',       # (3,)  body-frame projected gravity
    'ang_vel',         # (3,)  body-frame angular velocity (rad/s)
    'gait_phase',      # scalar gait phase (rad)
    'mode',            # str   always 'NN_CONTROL' in sim
]

OBS_PER_FRAME = 36

# Must match training config / balance.py constants.
SIM_DEFAULT_JOINT_POS = np.array(
    [0.0, 0.05, 0.0, 0.0, -0.05, 0.0], dtype=np.float32
)
# BUG FIX 08-31: knee scale was 2.05 (stale, pre-P6 convention) but the env
# uses real_action_scale_knee = 2.05/2 = 1.025 (rl_env_cfg_p6, matches
# control/config.py ACTION_SCALES). Every simlog npz written before this fix
# has its knee 'target' channel inflated exactly 2x (positions/torques/obs
# unaffected — the bug was in this logger constant only).
SIM_ACTION_SCALES = np.array(
    [0.3, 0.75, 1.025, 0.3, 0.75, 1.025], dtype=np.float32
)


def new_log_buffer():
    return {ch: [] for ch in LOG_CHANNELS}


# ═════════════════════════════════════════════════════════════════════════════
# Lateral-IMU mirror (--mirror_lateral_imu) — hardware-bug replica
# ═════════════════════════════════════════════════════════════════════════════
# docs/imu-sign-check.md (08-28): config.py's IMU_ROLL_SIGN / IMU_GYRO_X_SIGN /
# IMU_GYRO_Z_SIGN implement a REFLECTION of the 180°-mounted sensor's frame,
# so every hardware run to date fed the policy mirrored {pg_y, ωx, ωz} with
# joints unmirrored. This helper injects the same defect into the sim obs
# (every history frame; slots 1 / 3 / 5 of each 36-dim frame), AFTER the env
# built the obs — matching where the hardware bug lives (raw IMU remap,
# upstream of the NN, downstream of physics). The logged 'obs' channel then
# carries the mirrored values, same semantics as the hardware npz logs.

_LATERAL_IMU_SLOTS = (1, 3, 5)   # pg_y, gyro_x, gyro_z within a 36-dim frame
_mirror_idx_cache = {}


def apply_lateral_mirror(obs, enabled: bool):
    """Return obs with pg_y/ωx/ωz negated in every frame (clone, not in-place —
    the env may reuse its obs buffer). Handles tensor and TensorDict obs."""
    if not enabled:
        return obs
    is_tensor = isinstance(obs, torch.Tensor)
    t = obs if is_tensor else obs['policy']
    dim = t.shape[-1]
    idx = _mirror_idx_cache.get(dim)
    if idx is None:
        idx = torch.tensor(
            [f * OBS_PER_FRAME + s
             for f in range(dim // OBS_PER_FRAME)
             for s in _LATERAL_IMU_SLOTS],
            dtype=torch.long, device=t.device,
        )
        _mirror_idx_cache[dim] = idx
    t = t.clone()
    t[..., idx] *= -1.0
    if is_tensor:
        return t
    obs['policy'] = t
    return obs


def save_log(log, checkpoint_path, control_hz):
    """Dump log buffer to <checkpoint_basename>_simlog_<timestamp>.npz in CWD.

    Returns (filename, num_ticks).
    """
    n = len(log['time'])
    base = os.path.splitext(os.path.basename(checkpoint_path))[0]
    stamp = time.strftime('%Y%m%d_%H%M%S')
    fname = f"{base}_simlog_{stamp}.npz"

    arrays = {ch: np.asarray(vals) for ch, vals in log.items()}
    np.savez(
        fname,
        policy=np.asarray(os.path.basename(checkpoint_path)),
        control_hz=np.asarray(control_hz),
        **arrays,
    )
    return fname, n


# ═════════════════════════════════════════════════════════════════════════════
# Terminal observation-summary panel  (identical to play_log.py)
# ═════════════════════════════════════════════════════════════════════════════

_PANEL_WIDTH = 78
_JOINT_HEADER = f"             hr1     hp1     kn1     hr2     hp2     kn2"


def _safe_item(t):
    """torch scalar -> python float, with NaN-safe fallback."""
    try:
        return float(t.item())
    except Exception:
        return float("nan")


def build_obs_summary_lines(env, env_id: int):
    """Build the formatted panel as a list of lines (no trailing newlines)."""
    e = env.unwrapped
    eid = env_id

    # ── Commands & achieved velocities ────────────────────────────────────────
    vx_cmd = _safe_item(e._vx_cmd[eid])
    yaw_cmd = _safe_item(e._yaw_rate_cmd[eid])

    root_quat = e.robot.data.root_quat_w[eid:eid + 1]
    root_lin_vel = e.robot.data.root_lin_vel_w[eid:eid + 1]
    root_ang_vel = e.robot.data.root_ang_vel_w[eid:eid + 1]
    base_lin_vel = quat_rotate_inverse(root_quat, root_lin_vel)[0]
    base_ang_vel = quat_rotate_inverse(root_quat, root_ang_vel)[0]

    base_vx = _safe_item(base_lin_vel[0])
    base_vy = _safe_item(base_lin_vel[1])
    base_vz = _safe_item(base_lin_vel[2])
    base_yaw_rate = _safe_item(base_ang_vel[2])
    base_h = _safe_item(e.robot.data.root_pos_w[eid, 2])

    # Uprightness via projected gravity in the IMU frame
    gravity_vec = torch.tensor([[0.0, 0.0, -1.0]], device=root_quat.device)
    proj_grav = quat_rotate_inverse(root_quat, gravity_vec)[0]
    uprightness = -_safe_item(proj_grav[2])

    # ── Episode / curriculum state ────────────────────────────────────────────
    grace = int(_safe_item(e._grace_remaining[eid]))
    stabilize = int(_safe_item(e._stabilize_remaining[eid]))
    difficulty = _safe_item(e._env_difficulty[eid])
    curr_alpha = float(getattr(e, "_curriculum_alpha", 0.0))
    gait_phase = _safe_item(e._gait_phase[eid])

    # ── Actions (raw pre-clip vs clipped) ─────────────────────────────────────
    raw_act = e._raw_actions[eid].detach().cpu().tolist()
    clip_act = e.actions[eid].detach().cpu().tolist()

    # ── Torques ───────────────────────────────────────────────────────────────
    joint_ids = e._joint_ids
    torques_eid = e.robot.data.applied_torque[eid, joint_ids].detach().cpu().tolist()
    torques_all_abs = e.robot.data.applied_torque[:, joint_ids].abs()
    torques_max = torques_all_abs.max(dim=0).values.detach().cpu().tolist()

    # Continuous effort limit per joint (Nm) for headroom comparison
    cont_eff = e._continuous_effort.detach().cpu().tolist()

    # ── Header ────────────────────────────────────────────────────────────────
    sep = "═" * _PANEL_WIDTH
    lines = []
    lines.append(sep)
    lines.append(
        f"  SIM2REAL ROLLOUT      "
        f"env {eid}/{e.num_envs}   "
        f"ctrl {1.0 / e._control_dt:.0f} Hz   "
        f"t={e.episode_length_buf[eid].item() * e._control_dt:.2f}s"
    )
    lines.append(sep)

    # ── Commands block ────────────────────────────────────────────────────────
    lines.append(f"  COMMANDS               cmd       actual      error")
    lines.append(
        f"    vx        :       {vx_cmd:+7.3f}    {base_vx:+7.3f}    "
        f"{base_vx - vx_cmd:+7.3f}  m/s"
    )
    lines.append(
        f"    yaw_rate  :       {yaw_cmd:+7.3f}    {base_yaw_rate:+7.3f}    "
        f"{base_yaw_rate - yaw_cmd:+7.3f}  rad/s"
    )

    # ── State block ───────────────────────────────────────────────────────────
    lines.append(f"  STATE")
    lines.append(
        f"    base_h    : {base_h:6.3f} m    "
        f"base_vy: {base_vy:+6.3f}    base_vz: {base_vz:+6.3f}"
    )
    lines.append(
        f"    upright   : {uprightness:6.3f}      "
        f"grace: {grace:4d}      stab:    {stabilize:4d}"
    )
    lines.append(
        f"    difficulty: {difficulty:6.3f}      "
        f"curri α: {curr_alpha:5.3f}    gait φ: {gait_phase:5.2f} rad"
    )

    # ── Actions block ─────────────────────────────────────────────────────────
    lines.append(f"  ACTIONS    {_JOINT_HEADER.lstrip()}")
    lines.append("    raw      :  " + "  ".join(f"{a:+6.3f}" for a in raw_act))
    lines.append("    clipped  :  " + "  ".join(f"{a:+6.3f}" for a in clip_act))
    sat_flags = ["SAT" if abs(a) > 1.0 else " . " for a in raw_act]
    lines.append("    |raw|>1  :  " + "  ".join(f"  {s} " for s in sat_flags))

    # ── Torques block ─────────────────────────────────────────────────────────
    lines.append(f"  TORQUES (Nm){_JOINT_HEADER.lstrip()[3:]}")
    lines.append("    applied  :  " + "  ".join(f"{t:+6.2f}" for t in torques_eid))
    lines.append("    |all|max :  " + "  ".join(f" {t:5.2f}" for t in torques_max))
    lines.append("    cont eff :  " + "  ".join(f" {t:5.2f}" for t in cont_eff))

    # ── DR block ──────────────────────────────────────────────────────────────
    dr_on = bool(getattr(e.cfg, "dr_enable", False))
    if dr_on:
        adelay = int(_safe_item(e._env_action_delay[eid]))
        odelay = int(_safe_item(e._env_obs_delay[eid]))
        tot_steps = adelay + odelay
        tot_ms = tot_steps * e._control_dt * 1000.0

        mass_mult = _safe_item(e._dr_mass_mult[eid])
        fric_mult = _safe_item(e._dr_floor_friction_mult[eid])
        volt = _safe_item(e._dr_voltage_factor[eid])

        stiff = e._dr_stiffness_mult[eid].detach().cpu()
        damp = e._dr_damping_mult[eid].detach().cpu()
        arm = e._dr_armature_mult[eid].detach().cpu()
        eff = e._dr_effort_mult[eid].detach().cpu()
        jfr = e._dr_jfriction_mult[eid].detach().cpu()

        jpos_bias_max = _safe_item(e._dr_joint_pos_bias[eid].abs().max())
        grav_bias_max = _safe_item(e._dr_proj_grav_bias[eid].abs().max())
        com_off_max = _safe_item(e._dr_com_offset[eid].abs().max())

        lines.append(f"  DOMAIN RANDOMIZATION  (env {eid})")
        lines.append(
            f"    delays    : action={adelay:d}  obs={odelay:d}  "
            f"total={tot_steps:d} steps (~{tot_ms:.0f} ms)"
        )
        lines.append(
            f"    mass×     : {mass_mult:5.3f}     "
            f"fric×: {fric_mult:5.3f}     voltage×: {volt:5.3f}"
        )
        lines.append(
            f"    stiff×    : [{stiff.min().item():4.2f}..{stiff.max().item():4.2f}]   "
            f"damp×: [{damp.min().item():4.2f}..{damp.max().item():4.2f}]   "
            f"effort×: [{eff.min().item():4.2f}..{eff.max().item():4.2f}]"
        )
        lines.append(
            f"    armature× : [{arm.min().item():4.2f}..{arm.max().item():4.2f}]   "
            f"jfric×: [{jfr.min().item():4.2f}..{jfr.max().item():4.2f}]"
        )
        lines.append(
            f"    biases    : jpos±{jpos_bias_max:.4f} rad  "
            f"grav±{grav_bias_max:.4f}  com±{com_off_max:.4f} m"
        )
    else:
        lines.append(f"  DOMAIN RANDOMIZATION  : DISABLED")

    lines.append(sep)
    return lines


def get_toe_z(e, eid: int):
    """World-frame toe heights (toe1_z, toe2_z) for env `eid`.

    Replicates the env's own toe computation (rl_env_p6 _compute_gait_state):
    body pos + body-frame toe offset. LEG IDENTITY (verified 08-28):
      foot/leg 1 = URDF hip_roll1 @ y=+0.080 = robot's LEFT  = motors 103/101/102
      foot/leg 2 = URDF hip_roll2 @ y=-0.080 = robot's RIGHT = motors 106/104/105
    (config.py joint map; +x forward, +y left.) NOTE the viewport MIRRORS
    this when the camera faces the robot: its left leg is on your right.
    """
    p1 = e.robot.data.body_pos_w[eid, e._foot1_idx, :]
    q1 = e.robot.data.body_quat_w[eid, e._foot1_idx, :]
    z1 = (p1 + quat_apply(q1.unsqueeze(0), e._toe1_local_offset.unsqueeze(0))[0])[2]
    p2 = e.robot.data.body_pos_w[eid, e._foot2_idx, :]
    q2 = e.robot.data.body_quat_w[eid, e._foot2_idx, :]
    z2 = (p2 + quat_apply(q2.unsqueeze(0), e._toe2_local_offset.unsqueeze(0))[0])[2]
    return float(z1.item()), float(z2.item())


def format_dr_line(e, eid: int) -> str:
    """One-line DR draw summary for the per-episode trace prints."""
    if not bool(getattr(e.cfg, "dr_enable", False)):
        return "DR disabled"
    try:
        adelay = int(_safe_item(e._env_action_delay[eid]))
        odelay = int(_safe_item(e._env_obs_delay[eid]))
        mass = _safe_item(e._dr_mass_mult[eid])
        fric = _safe_item(e._dr_floor_friction_mult[eid])
        volt = _safe_item(e._dr_voltage_factor[eid])
        stiff = e._dr_stiffness_mult[eid]
        arm = e._dr_armature_mult[eid]
        jfr = e._dr_jfriction_mult[eid]
        pg_bias_x = _safe_item(e._dr_proj_grav_bias[eid][0])
        jpos_bias = _safe_item(e._dr_joint_pos_bias[eid].abs().max())
        # Read the PhysX masses back so a mass pin that silently no-ops is visible.
        m_now = e.robot.root_physx_view.get_masses()[eid]
        total_kg = float(m_now.sum().item())
        top_kg = float(m_now[list(getattr(e, "_top_body_ids", []))].sum().item()) if hasattr(e, "_top_body_ids") else float("nan")
        # Torso inertia read-back (trace of the 3x3, vs the original) shows whether
        # dr_mass_recompute_inertia actually rescaled the tensor.
        I_now = e.robot.root_physx_view.get_inertias()[eid, e._torso_idx]
        I_tr = float(I_now[[0, 4, 8]].sum().item())
        I_tr0 = float(e._original_inertias[0, e._torso_idx][[0, 4, 8]].sum().item()) if hasattr(e, "_original_inertias") else float("nan")
        return (
            f"mass×{mass:.3f} (physx total {total_kg:.3f} kg, top3 {top_kg:.3f} kg, torso I×{I_tr / I_tr0:.3f}) "
            f"fric×{fric:.3f} volt×{volt:.3f} "
            f"delay a{adelay}+o{odelay} "
            f"stiff×[{stiff.min().item():.2f}..{stiff.max().item():.2f}] "
            f"arm×[{arm.min().item():.2f}..{arm.max().item():.2f}] "
            f"jfric×[{jfr.min().item():.2f}..{jfr.max().item():.2f}] "
            f"pg_bias_x{pg_bias_x:+.3f} jpos_bias±{jpos_bias:.3f}"
        )
    except Exception as ex:
        return f"<dr read failed: {type(ex).__name__}: {ex}>"


class TerminalSummaryPrinter:
    """Prints the panel at `hz`, overwriting in place via ANSI cursor-up."""

    def __init__(self, hz: float = 2.0):
        self.interval = 1.0 / max(hz, 0.1)
        self._last_print = 0.0
        self._lines_written = 0
        self._first = True

    def maybe_print(self, env, env_id: int):
        now = time.time()
        if now - self._last_print < self.interval:
            return
        self._last_print = now

        # Safety: clamp watched env id to valid range
        env_id = max(0, min(env_id, env.unwrapped.num_envs - 1))

        try:
            lines = build_obs_summary_lines(env, env_id)
        except Exception as ex:
            # Don't kill the play loop over a print bug
            lines = [f"[obs-summary error] {type(ex).__name__}: {ex}"]

        # Move cursor up to overwrite previous panel
        if not self._first:
            sys.stdout.write(f"\033[{self._lines_written}A")
        self._first = False

        # \033[K clears each line to EOL so shorter lines don't leave artifacts
        for line in lines:
            sys.stdout.write(line.ljust(_PANEL_WIDTH)[:_PANEL_WIDTH] + "\033[K\n")
        sys.stdout.flush()
        self._lines_written = len(lines)


# ═════════════════════════════════════════════════════════════════════════════
# Keyboard control
# ═════════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════════
# Elasticity knobs — the same quantities the sweep pins, but changeable here
# one reset at a time, so "the robot looks bouncy" becomes a parameter instead
# of an impression.
#
# Frames are the ones the env defines, and they are the reason these are keyed
# rather than typed: belt stiffness is KNEE-JOINT N m/rad (compare it against
# servo_kp_knee), belt damping is entered as the transmission's damping ratio,
# and foot damping as the ball's contact damping ratio. See
# rl_env_cfg_p6.RlEnvCfg's elasticity block and belt_knee_actuator.damping_for_ratio.
#
# All of them need ONE reset to take effect. Belt parameters are installed on
# the actuator at reset. The foot material is different: it is created with the
# prim at spawn and bound to the ball collision meshes, so a config edit alone
# reaches nothing — the stage is what PhysX reads, and only at stage attach.
#
# Measured 2026-09-12 (data/_test_elasticity_knobs.py's sibling probe,
# RL/scripts/rsl_rl/probe_foot_material.py): the material IS live — two separate
# launches differing only in restitution gave 0.00 m vs 0.42 m of rebound — and
# PhysX DOES re-read an attribute edit across sim.reset(), provided the stage is
# re-attached. So apply_foot_material() below writes the resolved cfg onto the
# live material prims on every reset, which is what makes the foot hotkeys work
# exactly like the belt ones. Editing the config and pressing R without this
# was a silent no-op: the rollout ran the spawn-time material regardless.
# ═════════════════════════════════════════════════════════════════════════════
BELT_STIFFNESS_LADDER = (0.0, 3.0, 8.0, 50.0, 150.0, 550.0, 950.0)
BELT_ZETA_LADDER = (0.0, 0.3, 0.7, 1.0)
FOOT_STIFFNESS_LADDER = (0.0, 200.0, 1000.0, 5000.0, 20000.0)
# UNBOUNDED ON PURPOSE, and it runs past 1.0. A restitution above unity is
# energy-gaining contact: PhysX reads it happily, the ball returns more energy
# than the impact carried, and the plant is now out of distribution in a way no
# policy has seen. That is the point of the top rungs — they are the "make it
# fail" setting, not a physical claim. (foot_contact.py used to reject >1 with a
# ValueError, which turned the third Home press into a crash; that ceiling is
# gone, and stays gone.)
FOOT_RESTITUTION_LADDER = (0.0, 0.03, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0)
FOOT_DAMPING_RATIO_LADDER = (0.0, 0.7, 1.0, 2.0, 10.1)
# Friction is not a ladder: two states carry all the meaning (traction, none),
# and the interesting experiment is the binary one. 1.0 is the value the
# legacy rigid path ships, so toggling back to it restores the trained plant.
FOOT_FRICTION_TRACTION = 1.0
FOOT_FRICTION_SLICK = 0.0

FOOT_MATERIAL_PREFIX = "foot_contact_material_"


def apply_foot_material(e) -> int:
    """Write the current foot cfg onto the live material prims. Returns prims written.

    The prims are authored by the spawner once, at stage build; nothing else in
    the loop touches them. Re-authoring them from cfg is what lets a foot hotkey
    take effect on the next reset, and it also means the printed plant and the
    simulated plant cannot drift apart.

    Writes through the raw attribute names rather than the schema accessors:
    probe_foot_material.py read the same prim back through PhysxMaterialAPI and
    UsdPhysics.MaterialAPI and found those helpers return no attribute for
    these names (they are authored as `physxMaterial:compliantContactStiffness`
    and `physics:restitution`, and the schema read-back returns unauthored),
    while the raw names read and write correctly.
    """
    import isaaclab.sim as sim_utils

    cfg = e.cfg
    stage = e.sim.stage if hasattr(e, "sim") else sim_utils.get_current_stage()
    stiffness = float(cfg.foot_contact_stiffness)
    damping = float(cfg.resolved_foot_contact_damping())
    restitution = float(cfg.foot_restitution)
    static_friction = float(cfg.foot_static_friction)
    dynamic_friction = float(cfg.foot_dynamic_friction)
    written = 0
    for prim in stage.Traverse():
        if not prim.GetName().startswith(FOOT_MATERIAL_PREFIX):
            continue
        for name, value in (
            ("physxMaterial:compliantContactStiffness", stiffness),
            ("physxMaterial:compliantContactDamping", damping),
            ("physics:restitution", restitution),
            ("physics:staticFriction", static_friction),
            ("physics:dynamicFriction", dynamic_friction),
        ):
            attr = prim.GetAttribute(name)
            if not attr:
                raise RuntimeError(
                    f"Foot material prim {prim.GetPath()} has no attribute {name}; "
                    "the material schema is not what this writer expects."
                )
            attr.Set(value)
        written += 1
    return written


def report_foot_bindings(e) -> None:
    """Print what the ball-foot collision prims RESOLVE to in the live env.

    A material prim existing proves the spawner took its material branch. It
    does NOT prove the solver consults it, and in this env those came apart:
    every footMaterial channel measured bit-identical rollouts (stiffness 200 vs
    20000, and friction 0.0 vs 1.0) while the prim read back the right values.
    The prime suspect is the clone/instance path — /World/envs/env_0 is a clone
    of the source root, and a material bound on the source does not always
    follow. So this asks the collision prim itself, per env, and prints both the
    resolved material and the raw relationship the solver uses.
    """
    import isaaclab.sim as sim_utils
    from pxr import UsdPhysics, UsdShade

    stage = e.sim.stage if hasattr(e, "sim") else sim_utils.get_current_stage()
    print("[PLANT] foot binding report (what the solver consults):")
    seen = 0
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        path = str(prim.GetPath())
        if "nail" not in path:
            continue
        seen += 1
        if seen > 8:
            continue
        bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(
            materialPurpose="physics")[0]
        # ComputeBoundMaterial hands back a UsdShade.Material SCHEMA, which is
        # not a Prim: asking one for a path or an attribute without unwrapping
        # it first either returns nonsense or raises "Accessed schema on
        # invalid prim" (it did, on the first run of this report).  Unwrap to
        # the prim, then check validity before touching it.
        bound_prim = bound.GetPrim() if isinstance(bound, UsdShade.Material) else None
        ok = bound_prim is not None and bound_prim.IsValid()
        details = ""
        if ok:
            for name in ("physics:staticFriction", "physics:dynamicFriction",
                         "physics:restitution",
                         "physxMaterial:compliantContactStiffness"):
                attr = bound_prim.GetAttribute(name)
                if attr:
                    details += f"\n        {name} = {attr.Get()}"
        print(f"    {path}")
        print(f"        instance={prim.IsInstance()} "
              f"instanceProxy={prim.IsInstanceProxy()} "
              f"bound={bound_prim.GetPath() if ok else '<none>'}{details}")
    if not seen:
        print("    NO nail collision prim found at all in the live env")


# ═════════════════════════════════════════════════════════════════════════════
# Ball-pad compliance — the PER-ENV path, tunable live (2026-09-12).
#
# This is a different plant from the foot MATERIAL above, and the difference is
# the whole reason it exists. `foot_contact_stiffness` writes a stage-global
# material prim, so one value serves every env and nothing takes effect until
# the stage is re-attached (hence "press R", hence a restart for some edits).
# The pad is a per-substep FORCE the env owns (`pad_contact.PadContact`), so a
# new value is live on the NEXT PHYSICS STEP: no reset, no re-attach.
#
# Which is why these knobs are bound to N/M rather than to the foot ladder, and
# why they print "live" instead of "press R". The pad must exist to be tuned —
# it is built in RlEnv.__init__ from `pad_contact_enable`, before any hotkey can
# run — so launch with `env.pad_contact_enable=true` (run_foot_demo.ps1's pad
# arms, or the flags at the bottom of this block). Pressing N/M without a pad
# prints how to get one instead of silently doing nothing.
#
# Stiffness rungs are the 2026-09-12 calibration band (5e3 … 5e4 is the useful
# range; 200 is a hole in the floor at 27 cm of sag and is kept only so the
# bottom of the old FOOT ladder is reachable for comparison).
# ═════════════════════════════════════════════════════════════════════════════
PAD_STIFFNESS_LADDER = (0.0, 200.0, 1000.0, 5000.0, 10000.0, 20000.0, 50000.0)
PAD_DAMPING_RATIO_LADDER = (0.0, 0.3, 0.7, 1.0, 2.0)


class _PadKnobs:
    """Live ball-pad tuning: a HOLD, applied now and re-applied after a reset.

    Two writes per press, and both are load-bearing (2026-09-12 evening):

    1. `env._pad.set_parameters(...)` — live on the next physics substep, no
       reset and no stage re-attach, which is the whole benefit of the pad
       being a force rather than a material.
    2. `env._dr_override["pad_contact_hold"]` — so the value survives `R`.

    Write 2 needs a SEPARATE key from the sweep driver's `pad_contact`, and the
    distinction is not cosmetic. A sweep driver paints a different value into
    every cell and re-paints per arm, so its override must keep winning and
    short-circuit the band draw entirely. A by-feel pin is one value for the
    whole session: it must be written OVER the band draw, or a single N press
    would silently replace the per-env band with a constant for the rest of the
    run while the terminal still showed a band. `pad_contact_hold` is that
    second, narrower key (rl_env_p6._sample_dr draws the band, then writes the
    held value over the resets that share it).
    """

    def __init__(self, e):
        self.cfg = e.cfg
        self.pad = getattr(e, "_pad", None)
        device = e.device
        n = e.num_envs
        # Full-width rows: the hold is per env per foot, and a knob sets all of
        # them.  Seeded from the live pad so the first key press continues from
        # where the plant actually is rather than from a cfg default that the
        # band draw may have moved.
        seed_k = float(self.cfg.pad_stiffness_max)
        if self.pad is not None:
            seed_k = float(self.pad.stiffness.max().item())
        self.k = torch.full((n, 2), seed_k, device=device)
        self.z = float(self.cfg.pad_damping_ratio)
        self.installed = False
        # The un-collapsed band, kept so the knob can put it back. `padsoft`
        # and `padband` are both bands; a by-feel session must not have to
        # remember which one it started from in order to return to it.
        self.band = (float(self.cfg.pad_stiffness_min), float(self.cfg.pad_stiffness_max))
        # What the knob's own value was seeded from, so the FIRST press does not
        # get blamed for a collapse it did not make.
        self.seed_k = seed_k

    # ── reads ────────────────────────────────────────────────────────────
    def stiffness(self):
        if self.pad is None:
            return None
        return float(self.pad.stiffness.max().item())

    def ratio(self):
        """Effective damping ratio for the live k, read back off the pad.

        Read from the pad rather than echoed from the requested zeta on
        purpose: the damping is capped at the explicit-integrator ceiling, so
        the ratio the plant actually runs can be BELOW the one requested.
        Echoing the request would be the "knob that reports a plausible number"
        failure this campaign keeps repeating.
        """
        if self.pad is None:
            return None
        k = float(self.pad.stiffness.max().item())
        if k <= 0.0:
            return None
        c = float(self.pad.damping.max().item())
        return c / (2.0 * math.sqrt(k * float(self.pad.effective_mass)))

    def _write(self, e, k: float, z: float):
        """Hold (k, zeta) across every env/foot, live now and over resets."""
        self.cfg.pad_contact_enable = k > 0.0
        # Collapse the band to the held value, and keep the original in
        # `self.band` so keying back down to the seed can restore it.
        #
        # Collapsing is not a convenience: the cfg band is what `pad_enabled()`
        # reads, and `_sample_dr` takes its INERT branch (writing zeros for the
        # whole reset) whenever the band is zero. A live hold over an inert band
        # would therefore be silently erased on the next reset AND misreported
        # in the terminal as "INERT — no force applied" while the pad was in
        # fact pushing. Making the hold the band removes both failure modes.
        #
        # What this costs: the cfg no longer describes the arm the process was
        # launched as, so a resumed run re-samples the tuned value instead of
        # the original band. That is the honest trade for a knob that has to
        # survive a reset, and the band comes back the moment the knob is walked
        # down to the value it was seeded from.
        self.cfg.pad_damping_ratio = z
        self.k.fill_(k)
        self.z = z
        if math.isclose(k, self.seed_k, rel_tol=1e-9):
            self.cfg.pad_stiffness_min, self.cfg.pad_stiffness_max = self.band
        else:
            self.cfg.pad_stiffness_min = k
            self.cfg.pad_stiffness_max = k
        ov = getattr(e, "_dr_override", None)
        if ov is None:
            ov = {}
            e._dr_override = ov
        # A copy, not a view of self.k: the knob fill_()s this buffer again on
        # the next press, and the env holds the override across resets.
        ov["pad_contact_hold"] = self.k.clone()
        self.installed = True
        if self.pad is not None:
            _, c_eff = self.pad.resolved_kc(self.k, z)
            self.pad.set_parameters(torch.arange(e.num_envs, device=e.device), self.k, c_eff)

    # ── the knob interface ───────────────────────────────────────────────
    def _no_pad(self, label: str) -> tuple[str, str]:
        return (label, "NO PAD — relaunch with env.pad_contact_enable=true")

    def apply(self, e, key_name, direction):
        """Return (label, value) after stepping, or None if unmapped."""
        if key_name == "PAD_K":
            k = _step_ladder(PAD_STIFFNESS_LADDER, self.stiffness(), direction)
            if self.pad is None:
                return self._no_pad("pad k")
            self._write(e, k, self.z)
            note = ""
            if 0.0 < k < 5000.0:
                # The 09-12 calibration: k > 4900 N/m keeps a 1 kg catch inside
                # 2 mm of travel. Below that the pad's own travel runs out
                # before its spring does, and it stops behaving like a pad — so
                # the rung is still selectable (the bottom of the band is worth
                # seeing) but it says what it is.
                note = f"  <-- soft end: {1.0 * 9.81 / k * 1e3:.1f} mm sag per kg"
            return ("pad k", f"{k:g} N/m  (LIVE, no reset needed){note}")
        if key_name == "PAD_ZETA":
            z = _step_ladder(PAD_DAMPING_RATIO_LADDER, self.ratio(), direction)
            if self.pad is None:
                return self._no_pad("pad zeta")
            self._write(e, float(self.k.max().item()), z)
            eff = self.ratio()
            warn = ""
            if eff is not None and z > 0.0 and eff < 0.9 * z:
                warn = (f"  <-- CAPPED by the integrator at c={self.pad.damping_cap:.0f} N s/m "
                        f"(effective zeta {eff:.2f})")
            return ("pad zeta", f"{z:g}  (LIVE, no reset needed){warn}")
        return None

    def banner(self) -> list[str]:
        """The startup lines describing the pad plant, for the key help."""
        if self.pad is None:
            return ["[INFO]     (no pad built — the foot knobs above are the MATERIAL path)"]
        k = self.stiffness()
        eff = self.ratio()
        if not k:
            state = "INERT (k=0)"
        else:
            state = f"k={k:g} N/m zeta={eff:.2f} (capped at {self.pad.damping_cap:.0f} N s/m)"
        return [
            f"[INFO]     N / M      pad k             {PAD_STIFFNESS_LADDER} N/m  [LIVE]",
            f"[INFO]     , / .      pad zeta          {PAD_DAMPING_RATIO_LADDER}  [LIVE]",
            f"[INFO]     pad now: {state}  m_eff={self.pad.effective_mass:g} kg "
            f"travel={self.pad.pad_thickness*1e3:.1f} mm",
        ]


def report_pad_plant(e) -> None:
    """Print the live pad plant and its two silent-failure modes.

    Both warnings exist because each one produces a knob that reads correctly
    and moves nothing, which is the failure this campaign keeps rediscovering:

    * **Travel.** The pad's relaxed thickness must exceed the sag it carries
      (m*g/k). A pad that bottoms out on the ball's own rigid contact reports
      its THICKNESS as the settled sag and stops responding to stiffness at all.
    * **Damping cap.** An explicit damped contact diverges above 2*m/dt, so the
      damping is clipped at pad_damping_stability_factor*m/dt. A band that
      reaches the cap has quietly stopped meaning what it says. `PadContact`
      already reports the clip once per run; this reports it at startup, before
      any compute is spent.
    """
    pad = getattr(e, "_pad", None)
    if pad is None:
        if getattr(e.cfg, "pad_contact_enable", False):
            print("[PLANT][WARN] pad_contact_enable is set but no pad was built — "
                  "see the geometry error above.")
        return
    print(f"[PLANT] pad live: k={pad.stiffness.max().item():g} N/m (max over env-feet), "
          f"reference radius {pad.reference_radius*1e3:.2f} mm, "
          f"ball R {pad.ball_radius*1e3:.2f} mm, travel {pad.pad_thickness*1e3:.2f} mm, "
          f"c <= {pad.damping_cap:.1f} N s/m, enabled={pad.enabled}")
    k_min = float(e.cfg.pad_stiffness_min)
    if pad.enabled and 0.0 < k_min < 5000.0:
        sag_mm = float(e.cfg.pad_effective_mass) * 9.81 / k_min * 1e3
        print(f"[PLANT][WARN] softest rung k={k_min:g} N/m sags {sag_mm:.1f} mm "
              f"under {e.cfg.pad_effective_mass:g} kg against {pad.pad_thickness*1e3:.1f} mm "
              f"of travel.")
        if sag_mm > pad.pad_thickness * 1e3:
            print("[PLANT][WARN]   that sag EXCEEDS the travel: the ball bottoms out on "
                  "its rigid contact and the knob will read as inert. Raise "
                  "pad_thickness or pad_stiffness_min.")


def _step_ladder(ladder, current, direction):
    """Next value along a ladder, clamped at both ends.

    Snaps to the nearest rung first: a value set from the CLI (or a NaN) is not
    on the ladder, and stepping from an off-ladder value must land somewhere
    predictable rather than skipping a rung.
    """
    if direction == 0 or not ladder:
        return current
    if current is None or not math.isfinite(current):
        idx = 0
    else:
        idx = min(range(len(ladder)), key=lambda i: abs(ladder[i] - current))
    idx = max(0, min(len(ladder) - 1, idx + direction))
    return ladder[idx]


class _ElasticityKnobs:
    """Left/right-arrow plant tuning for the play loop.

    Each knob is (label, ladder, getter, setter). The ladders are short and
    ordered so the user can walk to a plant and back without remembering
    numbers; the interesting range is the one the 09-11 analysis named
    (k_belt/servo_kp_knee <= 1, restitution <= 0.2).
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.notes = []

    def _belt_k(self):
        return self.cfg.belt_series_stiffness_min if self.cfg.belt_series_enable else 0.0

    def _belt_zeta(self):
        z = self.cfg.belt_series_damping_ratio
        return None if math.isnan(z) else z

    def foot_damping_ratio(self):
        return None if math.isnan(self.cfg.foot_contact_damping_ratio) else self.cfg.foot_contact_damping_ratio

    def apply(self, key_name, direction):
        """Return (label, value) after stepping, or None if the key is unmapped."""
        if key_name == "BELT_K":
            k = _step_ladder(BELT_STIFFNESS_LADDER, self._belt_k(), direction)
            self.cfg.belt_series_enable = k > 0.0
            self.cfg.belt_series_stiffness_min = k
            self.cfg.belt_series_stiffness_max = k
            return ("belt k_joint", f"{k:g} N m/rad (enable={k > 0.0})")
        if key_name == "BELT_ZETA":
            z = _step_ladder(BELT_ZETA_LADDER, self._belt_zeta(), direction)
            self.cfg.belt_series_damping_ratio = z if z is not None else float("nan")
            return ("belt zeta", f"{z:g}" if z is not None else "band (off)")
        if key_name == "FOOT_K":
            k = _step_ladder(FOOT_STIFFNESS_LADDER, self.cfg.foot_contact_stiffness, direction)
            self.cfg.foot_contact_stiffness = k
            return ("foot k", f"{k:g} N/m")
        if key_name == "FOOT_ZETA":
            z = _step_ladder(FOOT_DAMPING_RATIO_LADDER, self.foot_damping_ratio(), direction)
            self.cfg.foot_contact_damping_ratio = z if z is not None else float("nan")
            return ("foot zeta", f"{z:g}" if z is not None else "off")
        if key_name == "FOOT_REST":
            r = _step_ladder(FOOT_RESTITUTION_LADDER, self.cfg.foot_restitution, direction)
            self.cfg.foot_restitution = r
            note = "  <-- energy-gaining contact" if r > 1.0 else ""
            return ("foot restitution", f"{r:g}{note}")
        if key_name == "FOOT_FRIC":
            slick = not self.foot_slick()
            self.cfg.foot_static_friction = FOOT_FRICTION_SLICK if slick else FOOT_FRICTION_TRACTION
            self.cfg.foot_dynamic_friction = self.cfg.foot_static_friction
            return ("foot friction", "0.00 SLICK" if slick else "1.00 traction")
        return None

    def foot_slick(self) -> bool:
        """True when the foot carries no traction. Read from cfg, not a flag,
        so a CLI-set value and a hotkey-set value agree."""
        return float(self.cfg.foot_static_friction) == FOOT_FRICTION_SLICK


class KeyboardController:
    """Subscribe to carb keyboard events and expose flags to the play loop.

    Keys handled:
        R   – request a full env reset on the next loop iteration
        B   – reset + start a log run (mirrors balance.py 'b')
        H   – stop the log run and save it (mirrors balance.py 'h')
        Up/Down    – belt series stiffness (knee-joint N m/rad, 0 = rigid)
        Left/Right – belt damping ratio zeta
        PgUp/PgDn  – foot contact stiffness (0 = rigid legacy material)
        Home/End   – foot restitution (steps PAST 1.0: energy-gaining contact)
        Ins/Del    – foot contact damping ratio
        F          – foot friction 1.0 <-> 0.0 (0 = no traction at all)

    Direction is which ARROW, not a sign: Down softens belt stiffness the same
    way PgDn softens the foot, so "softer" is one consistent gesture.
    """

    # carb key -> (knob, direction)
    _ELASTIC_KEYS = {
        "UP": ("BELT_K", +1),
        "DOWN": ("BELT_K", -1),
        "RIGHT": ("BELT_ZETA", +1),
        "LEFT": ("BELT_ZETA", -1),
        "PAGE_UP": ("FOOT_K", +1),
        "PAGE_DOWN": ("FOOT_K", -1),
        "HOME": ("FOOT_REST", +1),
        "END": ("FOOT_REST", -1),
        "INSERT": ("FOOT_ZETA", +1),
        # carb names this DEL, not DELETE (NUMPAD_DEL is the numpad one), and
        # there is no DELETE member to alias — data/_test_elasticity_knobs.py
        # checks every name here against the shipped carb stub, because a name
        # carb does not define is a key that silently does nothing.
        "DEL": ("FOOT_ZETA", -1),
        # A toggle, not a ladder, so the direction is unused. Foot friction is
        # the one foot knob that changes a rollout even when the balls never
        # carry load, which makes it the honest "did the material reach the
        # solver at all" test.
        "F": ("FOOT_FRIC", +1),
        # The PER-ENV pad (pad_contact.PadContact), which is a different plant
        # from the material above and is the only foot knob that takes effect
        # WITHOUT a reset. N/M = stiffness (adjacent keys, stiffer to the right
        # like the arrows), comma/period = damping ratio (the two keys directly
        # left of M, softer to the left). Named rather than keyed numerically
        # for the reason the comment above gives: carb's NAMES are stable across
        # the 4.x/5.x builds this repo runs on, the numeric values are not.
        "N": ("PAD_K", +1),
        "M": ("PAD_K", -1),
        "COMMA": ("PAD_ZETA", +1),
        "PERIOD": ("PAD_ZETA", -1),
    }

    def __init__(self):
        self.reset_requested = False
        self.start_log_requested = False
        self.stop_log_requested = False
        # [(dir, knob_name)] queued for the main loop; applied between ticks so
        # the cfg is never mutated from inside a carb callback.
        self.elastic_requests = []
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event
        )

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.R:
                self.reset_requested = True
            elif event.input == carb.input.KeyboardInput.B:
                self.start_log_requested = True
            elif event.input == carb.input.KeyboardInput.H:
                self.stop_log_requested = True
            else:
                # Keyed by NAME, not by the enum member: carb's KeyboardInput
                # naming is stable across the 4.x/5.x builds this repo runs on,
                # while the numeric values are not guaranteed to be. Verified
                # against the shipped stub (carb/input.pyi) that all ten names
                # below exist — PAGE_UP/PAGE_DOWN, not PGUP/PGDN.
                name = getattr(event.input, "name", None)
                if name in self._ELASTIC_KEYS:
                    self.elastic_requests.append(self._ELASTIC_KEYS[name])
        # Return True so other subscribers (e.g. viewport camera) still get the event
        return True

    def shutdown(self):
        if self._sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
            self._sub = None


class _NullKeyboard:
    """Stand-in for KeyboardController under --autolog_steps (headless):
    same flag attributes, no carb/appwindow subscription."""

    def __init__(self):
        self.reset_requested = False
        self.start_log_requested = False
        self.stop_log_requested = False
        self.elastic_requests = []

    def shutdown(self):
        pass


def inject_fork_state(e, npz_path: str, t_inject: float, history_mode: str,
                      inject_vx: float = 0.0):
    """p12.3 arm A — overwrite the post-reset sim state with the hardware
    fork state and prime every policy-side buffer from the same log.

    Must be called AFTER env.reset() (so DR has drawn a fresh plant and all
    buffers are in their reset state) and returns the stacked obs tensor to
    feed the policy on the first tick.

    Dynamics state (identical for A1/A2):
      - root attitude from the log's RAW projected gravity (logged − 0.089
        trim on x): the quat rotating body-frame gravity onto world -z.
      - root ang vel = logged body ang_vel rotated to world; lin vel = 0
        (not logged on hw; the fork state is quasi-static by construction).
      - joint pos/vel from the log; root z set so the LOWER toe sits 4 mm
        above ground (one throwaway physics step at z=0.6 refreshes FK).
    Policy-side buffers (from the log, both variants):
      - _last_actions / _last_last / _last_last_last = a[i0], a[i0-1], a[i0-2]
      - _filtered_actions = LPF(0.6) run over the logged action stream
      - _action_delay_buffer slot j = that filter state j ticks ago
      - _obs_delay_buffer sensor slots (0:21) from the logged frames
      - _gait_phase = logged gait_phase[i0]
    History (the A1/A2 fork):
      - 'log'       (A1): slot j = logged obs frame i0-j (newest = slot 0)
      - 'replicate' (A2): logged frame i0 copied into all K slots
    """
    TRIM_X = 0.089
    d = np.load(npz_path, allow_pickle=True)
    t = np.asarray(d["time"], dtype=np.float64)
    t = t - t[0]
    i0 = int(np.argmin(np.abs(t - t_inject)))
    K = int(e.cfg.obs_history_steps)
    if i0 < max(K - 1, 2):
        raise ValueError(f"--inject_time {t_inject} leaves only {i0} preceding "
                         f"ticks; need >= {max(K - 1, 2)} for history/last-action priming")
    dev = e.device

    obs_log = np.asarray(d["obs"], dtype=np.float32)
    act_log = np.asarray(d["action"], dtype=np.float32)
    jp_log = np.asarray(d["sim_joint_pos"], dtype=np.float32)
    jv_log = np.asarray(d["sim_joint_vel"], dtype=np.float32)
    pg_log = np.asarray(d["proj_grav"], dtype=np.float64)
    av_log = np.asarray(d["ang_vel"], dtype=np.float64)
    phase_log = np.asarray(d["gait_phase"], dtype=np.float64)

    # ── root attitude: quat q with quat_apply_inverse(q, [0,0,-1]) = raw pg ──
    g_b = pg_log[i0].copy()
    g_b[0] -= TRIM_X                       # remove the hw display trim
    g_b = g_b / np.linalg.norm(g_b)
    g_w = np.array([0.0, 0.0, -1.0])
    axis = np.cross(g_b, g_w)
    s = np.linalg.norm(axis)
    if s < 1e-8:
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
    else:
        ang = float(np.arctan2(s, np.dot(g_b, g_w)))
        axis = axis / s
        quat = torch.tensor(
            [np.cos(ang / 2.0), *(np.sin(ang / 2.0) * axis)],
            dtype=torch.float32, device=dev,
        )
    pitch_deg = float(np.degrees(np.arctan2(g_b[0], -g_b[2])))

    n_env = e.num_envs
    quat_all = quat.unsqueeze(0).repeat(n_env, 1)
    w_body = torch.tensor(av_log[i0], dtype=torch.float32, device=dev)
    w_world = quat_apply(quat_all, w_body.unsqueeze(0).repeat(n_env, 1))

    # full-articulation joint vectors via the env's slot map
    jids = e._joint_ids
    jp_full = e.robot.data.joint_pos.clone()
    jv_full = e.robot.data.joint_vel.clone()
    jp_full[:, jids] = torch.tensor(jp_log[i0], dtype=torch.float32, device=dev)
    jv_full[:, jids] = torch.tensor(jv_log[i0], dtype=torch.float32, device=dev)

    # ── FK pass at z=0.6 to find toe height, then final placement ─────────
    root_pose = torch.zeros((n_env, 7), device=dev)
    root_pose[:, :2] = e.scene.env_origins[:, :2]
    root_pose[:, 2] = 0.6
    root_pose[:, 3:7] = quat_all
    zero_vel = torch.zeros((n_env, 6), device=dev)
    e.robot.write_root_pose_to_sim(root_pose)
    e.robot.write_root_velocity_to_sim(zero_vel)
    e.robot.write_joint_state_to_sim(jp_full, jv_full, None, None)
    e.sim.step(render=False)               # airborne throwaway step -> FK refresh
    e.scene.update(e.physics_dt)
    z1, z2 = get_toe_z(e, 0)
    dz = min(z1, z2) - 0.004
    root_pose[:, 2] = 0.6 - dz
    vel = torch.zeros((n_env, 6), device=dev)
    vel[:, 0] = inject_vx
    vel[:, 3:6] = w_world
    e.robot.write_root_pose_to_sim(root_pose)
    e.robot.write_root_velocity_to_sim(vel)
    e.robot.write_joint_state_to_sim(jp_full, jv_full, None, None)
    e.scene.update(e.physics_dt)

    # spawn anchors -> injected pose so drift metrics/terms are self-relative
    e._spawn_pos_xy[:] = root_pose[:, :2] - e.scene.env_origins[:, :2]
    e._pos_ema[:] = e._spawn_pos_xy
    e._initial_height[:] = root_pose[:, 2]

    # ── policy-side buffers from the log ──────────────────────────────────
    def to_dev(a):
        return torch.tensor(a, dtype=torch.float32, device=dev).unsqueeze(0).repeat(n_env, 1)

    e._last_actions[:] = to_dev(act_log[i0])
    e._last_last_actions[:] = to_dev(act_log[i0 - 1])
    e._last_last_last_actions[:] = to_dev(act_log[i0 - 2])
    e.actions[:] = to_dev(act_log[i0])
    e._raw_actions[:] = to_dev(act_log[i0])

    # LPF state: run the deploy filter over the logged action stream
    alpha = float(e.cfg.action_lpf_alpha)
    filt = np.zeros((i0 + 1, act_log.shape[1]), dtype=np.float32)
    filt[0] = (1.0 - alpha) * act_log[0]
    for k in range(1, i0 + 1):
        filt[k] = alpha * filt[k - 1] + (1.0 - alpha) * act_log[k]
    e._filtered_actions[:] = to_dev(filt[i0])
    n_ad = e._action_delay_buffer.shape[1]
    for j in range(n_ad):
        e._action_delay_buffer[:, j] = to_dev(filt[max(i0 - j, 0)])

    # obs delay buffer: sensor block (0:21) of the logged frames
    n_od = e._obs_delay_buffer.shape[1]
    for j in range(n_od):
        e._obs_delay_buffer[:, j] = to_dev(obs_log[max(i0 - j, 0)][:21])

    e._gait_phase[:] = float(phase_log[i0])

    # ── obs history: the A1/A2 fork ────────────────────────────────────────
    if history_mode == "log":
        for j in range(K):
            e._obs_history_buf[:, j] = to_dev(obs_log[i0 - j])
    else:  # replicate
        for j in range(K):
            e._obs_history_buf[:, j] = to_dev(obs_log[i0])
    e._needs_history_init[:] = False

    stacked = e._obs_history_buf.reshape(n_env, -1).clone()
    print(f"[INJECT] {os.path.basename(npz_path)} @ t={t[i0]:.3f}s (tick {i0})  "
          f"history={history_mode}  pitch={pitch_deg:+.1f}°  "
          f"wy={np.degrees(av_log[i0][1]):+.0f}°/s  vx={inject_vx:+.2f}  "
          f"root_z={float(root_pose[0,2]):.3f}  "
          f"toe_z=({z1 - dz:.3f},{z2 - dz:.3f})")
    return stacked


def install_gait_phase_pin(e, phase_init: float):
    """Wrap the env's _reset_idx so the gait clock is pinned after every
    reset — R key, B key, AND the auto-reset the env performs when an
    episode terminates (fall / timeout).

    Training randomizes _gait_phase per episode (rl_env_p6 _reset_idx);
    hardware always starts the clock at 0 at NN handover. The wrap runs
    inside the step pipeline BEFORE _get_observations, so the first
    post-reset obs frame already carries the pinned sin/cos.
    """
    orig_reset_idx = e._reset_idx

    def _pinned_reset_idx(env_ids):
        orig_reset_idx(env_ids)
        if env_ids is None:
            e._gait_phase[:] = phase_init
        else:
            e._gait_phase[env_ids] = phase_init

    e._reset_idx = _pinned_reset_idx


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Roll out an RSL-RL checkpoint under the deploy scenario."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ══ Sim2real scenario shaping ═════════════════════════════════════════════
    # 1) NO pickup — all three pickup paths off (mid-episode probability,
    #    initial-pickup scheduling, and the pickup_enable master gate).
    #    Hard requirement of this script, applied after hydra on purpose.
    env_cfg.pickup_enable = False
    env_cfg.pickup_probability = 0.0
    env_cfg.pickup_initial_probability = 0.0

    # 2) Stand-branch resets only (difficulty 0, no kneel, no grace) — the
    #    handover on the rig is always from the PD-held stand pose.
    env_cfg.curriculum_enable = False

    # 3) Deploy contract gains/LPF as DEFAULTS — explicit env.* CLI wins.
    contract_applied = []
    for key, val in BALANCE_CONTRACT.items():
        if not _hydra_overrides(key):
            setattr(env_cfg, key, val)
            contract_applied.append(f"{key}={val}")

    # 4) Spawn-state noise scaling (default 0.0 = clean handover pose).
    s = max(0.0, args_cli.init_noise_scale)
    env_cfg.noise_joint_pos *= s
    env_cfg.noise_base_vel_lin *= s
    env_cfg.noise_base_vel_ang *= s
    env_cfg.init_base_orient_noise_rad *= s

    # 5) Ball-pad compliance (the per-env path). Applied here, immediately
    #    before gym.make, for one reason: RlEnv.__init__ builds the pad from
    #    these fields, so this is the LAST moment a launch flag can decide
    #    whether a pad exists. A hotkey cannot create one later — a force
    #    object with no geometry measurement behind it has nothing to push at.
    #    Explicit `env.pad_*` CLI overrides still win, matching rule 3.
    pad_applied = []
    pad_flags = {
        "pad_stiffness_min": args_cli.pad_stiffness_min,
        "pad_stiffness_max": args_cli.pad_stiffness_max,
        "pad_damping_ratio": args_cli.pad_damping_ratio,
        "pad_thickness": args_cli.pad_thickness,
    }
    if args_cli.pad_stiffness > 0.0:
        pad_flags["pad_stiffness_min"] = args_cli.pad_stiffness
        pad_flags["pad_stiffness_max"] = args_cli.pad_stiffness
    if any(v is not None for v in pad_flags.values()):
        args_cli.pad_enable = True
    if args_cli.pad_enable:
        if not _hydra_overrides("env.pad_contact_enable"):
            env_cfg.pad_contact_enable = True
        for key, val in pad_flags.items():
            if val is None or _hydra_overrides(f"env.{key}"):
                continue
            setattr(env_cfg, key, float(val))
            pad_applied.append(f"{key}={val:g}")
    if env_cfg.pad_contact_enable:
        state = "INERT (k=0 — tune with N)" if not env_cfg.pad_enabled() else "active"
        print(f"[PLANT] pad contact: enable=True k=[{env_cfg.pad_stiffness_min:g}, "
              f"{env_cfg.pad_stiffness_max:g}] N/m zeta={env_cfg.pad_damping_ratio:g} "
              f"thickness={env_cfg.pad_thickness*1e3:g} mm  ({state})")
        if pad_applied:
            print(f"[PLANT]   pinned from CLI: {'  '.join(pad_applied)}")

    print("═" * 78)
    print("  SIM2REAL SCENARIO")
    print(f"    envs             : {env_cfg.scene.num_envs}")
    print(f"    pickup           : OFF (forced)")
    print(f"    curriculum/kneel : OFF (stand start, difficulty 0)")
    print(f"    commands         : pinned vx=0, yaw_rate=0 (manual override)")
    print(f"    gait clock init  : {args_cli.gait_phase_init:.3f} rad on every reset")
    print(f"    init noise scale : {s:.2f}")
    print(f"    dr_enable        : {env_cfg.dr_enable}  (fresh draw every reset)")
    print(f"    lateral IMU      : "
          f"{'MIRRORED (pg_y/wx/wz negated — hw-bug replica)' if args_cli.mirror_lateral_imu else 'true-handed (post-fix contract)'}")
    if contract_applied:
        print(f"    contract defaults: {', '.join(contract_applied)}")
    else:
        print(f"    contract defaults: all overridden on CLI")
    print("═" * 78)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # ── Deploy-scenario hooks on the raw env, BEFORE any wrapper can trigger
    #    the first reset: pinned gait clock + zero-command override. ─────────
    e = env.unwrapped
    report_pad_plant(e)
    install_gait_phase_pin(e, args_cli.gait_phase_init)
    # manual_cmd_override is honored by _reset_idx AND the 1000-step
    # mid-episode resampler (_sample_commands), so cmd stays (0, 0) forever —
    # matching balance_log.py's hard-zeroed cmd obs slots.
    e.manual_cmd_override = True
    e.manual_vx = 0.0
    e.manual_yaw_rate = 0.0

    # wrap for video recording — identical to play_log.py
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "sim2real"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during rollout.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    control_hz = 1.0 / dt if dt > 0 else 0.0

    # ── Terminal summary printer ──────────────────────────────────────────────
    summary_printer = None
    if not args_cli.no_obs_summary:
        summary_printer = TerminalSummaryPrinter(hz=args_cli.summary_hz)
        print(
            f"[INFO] Terminal obs-summary ENABLED  "
            f"({args_cli.summary_hz:.1f} Hz, watching env {args_cli.watch_env}). "
            f"Pass --no_obs_summary to disable."
        )
        # Leave one blank line before the panel anchors itself
        print()
    # ── Keyboard control ──────────────────────────────────────────────────────
    # Built for both keyboard modes: under _NullKeyboard no request ever
    # arrives, but --autolog still gets the pad state printed at startup, which
    # is the only place a headless run can report what plant it ran.
    pad_knobs = _PadKnobs(e)
    if args_cli.autolog_steps > 0:
        keyboard = _NullKeyboard()
        print(
            f"[INFO] AUTOLOG: {args_cli.autolog_runs} run(s) x "
            f"{args_cli.autolog_steps} ticks, then exit (keyboard disabled)."
        )
    else:
        keyboard = KeyboardController()
        print("[INFO] Keyboard: focus the Isaac Sim window.")
        print("[INFO]   R = reset (fresh DR draw, gait clock re-pinned)")
        print("[INFO]   B = reset + start log run")
        print("[INFO]   H = stop log run + save to <ckpt>_simlog_<ts>.npz")
        print("[INFO]   Plant knobs — MATERIAL path (need an R reset):")
        print(f"[INFO]     Up/Down    belt k_joint      {BELT_STIFFNESS_LADDER} N m/rad")
        print(f"[INFO]     Left/Right belt zeta         {BELT_ZETA_LADDER}")
        print(f"[INFO]     PgUp/PgDn  foot k            {FOOT_STIFFNESS_LADDER} N/m")
        print(f"[INFO]     Home/End   foot restitution  {FOOT_RESTITUTION_LADDER}")
        print(f"[INFO]     Ins/Del    foot zeta         {FOOT_DAMPING_RATIO_LADDER}")
        print(f"[INFO]     F          foot friction     1.0 <-> 0.0 (0 = no traction)")
        print("[INFO]   (these apply at the next R/B reset: belt params install on "
              "the actuator, foot params are re-authored onto the foot material "
              "prim before the reset re-attaches the stage.)")
        print("[INFO]   Plant knobs — PER-ENV PAD path (LIVE, no reset):")
        for line in pad_knobs.banner():
            print(line)
    # Live plant knobs. Built in both keyboard modes so --autolog runs still
    # report the resolved plant; under _NullKeyboard no request ever arrives.
    elastic = _ElasticityKnobs(e.cfg)
    autolog_runs_left = args_cli.autolog_runs if args_cli.autolog_steps > 0 else 0
    _min_toe = float("inf")   # lowest toe height seen since the last save

    # ── Logging state (mirrors balance.py) ───────────────────────────────────
    log_active = False
    log_buffer = None
    log_start = 0.0
    log_failures = 0

    # ── Per-episode failure trace (watch env) ─────────────────────────────────
    eid_watch = max(0, min(args_cli.watch_env, env.unwrapped.num_envs - 1))
    ep_count = 0
    ticks_since_reset = 0
    # First-step detector: which toe first clears the env's own contact line
    # (no_flight_contact_threshold). Camera-proof ground truth for the
    # left-first / right-first question — the viewport mirrors left/right
    # whenever the camera faces the robot.
    first_step_reported = False
    first_step_thr = float(getattr(e.cfg, "no_flight_contact_threshold", 0.084))

    # Align the stage with cfg once, before the first tick. The spawner wrote
    # the material from the same cfg a moment ago, so this is normally a no-op;
    # it exists so that a cfg the spawner could not have written (e.g. one
    # whose resolved damping differs from `spawn.compliant_contact_damping`)
    # cannot silently disagree with what the rollout runs.
    _n_foot = apply_foot_material(e)
    if _n_foot:
        print(f"[PLANT] foot material prims at start: {_n_foot} "
              f"(k={e.cfg.foot_contact_stiffness:g} N/m "
              f"d={e.cfg.resolved_foot_contact_damping():g} N s/m "
              f"r={e.cfg.foot_restitution:g} "
              f"mu={e.cfg.foot_static_friction:g}/{e.cfg.foot_dynamic_friction:g})")
    # Does the SOLVER see it? The material prim existing says only that the
    # spawner's material branch ran. What matters is what the ball's own
    # collision prims resolve to, and for a cloned/instanced env that is not a
    # given: /World/envs/env_0 is cloned from env_0's source root, and a binding
    # authored on the source does not always follow the clone.
    report_foot_bindings(e)

    # reset environment. The gait-phase pin was installed before the wrappers,
    # so whichever wrapper triggered the initial reset already went through it;
    # re-pin defensively in case the very first reset predated gym.make's return.
    e._gait_phase[:] = args_cli.gait_phase_init
    obs = apply_lateral_mirror(env.get_observations(), args_cli.mirror_lateral_imu)
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # ── Plant knobs (arrows / PgUp / Home / Ins) ──────────────────────────
        # Applied on the main loop rather than inside the carb callback: the
        # callback runs on the input thread, and mutating the cfg (or the
        # spawn config derived from it) there would race the reset that reads
        # it. Nothing takes effect until the next reset either way.
        while keyboard.elastic_requests:
            knob, direction = keyboard.elastic_requests.pop(0)
            # The pad keys take the env as well: unlike every other knob, the
            # pad is applied by writing a live force object and the env's DR
            # override, not by mutating the cfg alone. The "press R to apply"
            # suffix the material knobs print is deliberately absent here —
            # this one is already in the plant by the next physics step, and
            # telling the user otherwise is the exact confusion this path
            # exists to end.
            applied = pad_knobs.apply(e, knob, direction) if knob.startswith("PAD_") \
                else elastic.apply(knob, direction)
            if applied is not None:
                label, value = applied
                suffix = "" if knob.startswith("PAD_") else "  (press R to apply)"
                print(f"[PLANT] {label} -> {value}{suffix}")

        # ── Keyboard-triggered reset (R) ──────────────────────────────────────
        if keyboard.reset_requested:
            keyboard.reset_requested = False
            print("[INFO] R pressed — resetting (fresh DR draw).")
            # Re-author the foot material from cfg BEFORE the reset, so the
            # reset that follows re-attaches the stage against the new values.
            # Measured 2026-09-12: PhysX does pick up an attribute edit across
            # sim.reset(); without this call the foot knobs were config-only.
            _n_foot = apply_foot_material(e)
            if _n_foot:
                print(f"[PLANT] foot material written to {_n_foot} prim(s): "
                      f"k={e.cfg.foot_contact_stiffness:g} N/m "
                      f"d={e.cfg.resolved_foot_contact_damping():g} N s/m "
                      f"r={e.cfg.foot_restitution:g} "
                      f"mu={e.cfg.foot_static_friction:g}/{e.cfg.foot_dynamic_friction:g}")
            with torch.inference_mode():
                # p12.2 finding (08-29): env.reset() returns the TRAINING
                # episode-start obs — dynamic channels ZEROED (the obs-delay
                # buffer is cleared at _reset_idx) and primed into all K
                # history slots ("blind start", act(blind*5) = the reflex
                # opening). The hardware handover instead feeds the real
                # sensor frame replicated x5 (balance_log.py review #2) — a
                # start state training never produces, act(real*5) = the
                # violent torso raise seen in every real trial.
                # --handover selects which contract this rollout reproduces.
                obs, _ = env.reset()
                if args_cli.inject_npz:
                    stacked = inject_fork_state(
                        e, args_cli.inject_npz, args_cli.inject_time,
                        args_cli.inject_history, args_cli.inject_vx,
                    )
                    if isinstance(obs, torch.Tensor):
                        obs = stacked
                    else:
                        obs["policy"] = stacked
                elif args_cli.handover == "deploy":
                    env.get_observations()  # rolls the real frame into slot 0
                    e._obs_history_buf[:, 1:] = e._obs_history_buf[:, 0:1]
                    stacked = e._obs_history_buf.reshape(e.num_envs, -1).clone()
                    if isinstance(obs, torch.Tensor):
                        obs = stacked
                    else:  # TensorDict / dict container from the wrapper
                        obs["policy"] = stacked
                obs = apply_lateral_mirror(obs, args_cli.mirror_lateral_imu)
                all_dones = torch.ones(
                    env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device
                )
                policy_nn.reset(all_dones)
            ticks_since_reset = 0
            first_step_reported = False
            print(f"[EP] new draw: {format_dr_line(e, eid_watch)}")
            if summary_printer is not None:
                summary_printer._first = True
            continue  # skip stepping this tick; resume next iteration with fresh obs

        # ── B: reset + start log run ──────────────────────────────────────────
        # If a log is already armed when 'b' is pressed again, dump it first
        # so we don't silently lose the previous run. Then fall through to
        # env.step + per-tick append (NO continue) so this iteration becomes
        # the first logged tick — mirrors balance.py's 'b' branch and avoids
        # the B-then-H race where H fires on the next iter before any append
        # has happened and saves an empty buffer.
        if keyboard.start_log_requested:
            keyboard.start_log_requested = False
            # Drain any stale H that was queued before B — without this, a
            # same-frame B then H would save the freshly-armed empty buffer.
            keyboard.stop_log_requested = False
            if log_active and log_buffer is not None:
                try:
                    fname, n = save_log(log_buffer, resume_path, control_hz)
                    print(f"[LOG] Saved {n} ticks to {fname} (re-arm before new run)")
                except Exception as ex:
                    print(f"[WARN] log save failed: {ex}")
            print("[INFO] B pressed — resetting envs and starting log.")
            apply_foot_material(e)   # same reason as the R branch: stage, not cfg
            with torch.inference_mode():
                # p12.2 finding (08-29): env.reset() returns the TRAINING
                # episode-start obs — dynamic channels ZEROED (the obs-delay
                # buffer is cleared at _reset_idx) and primed into all K
                # history slots ("blind start", act(blind*5) = the reflex
                # opening). The hardware handover instead feeds the real
                # sensor frame replicated x5 (balance_log.py review #2) — a
                # start state training never produces, act(real*5) = the
                # violent torso raise seen in every real trial.
                # --handover selects which contract this rollout reproduces.
                obs, _ = env.reset()
                if args_cli.inject_npz:
                    stacked = inject_fork_state(
                        e, args_cli.inject_npz, args_cli.inject_time,
                        args_cli.inject_history, args_cli.inject_vx,
                    )
                    if isinstance(obs, torch.Tensor):
                        obs = stacked
                    else:
                        obs["policy"] = stacked
                elif args_cli.handover == "deploy":
                    env.get_observations()  # rolls the real frame into slot 0
                    e._obs_history_buf[:, 1:] = e._obs_history_buf[:, 0:1]
                    stacked = e._obs_history_buf.reshape(e.num_envs, -1).clone()
                    if isinstance(obs, torch.Tensor):
                        obs = stacked
                    else:  # TensorDict / dict container from the wrapper
                        obs["policy"] = stacked
                obs = apply_lateral_mirror(obs, args_cli.mirror_lateral_imu)
                all_dones = torch.ones(
                    env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device
                )
                policy_nn.reset(all_dones)
            ticks_since_reset = 0
            first_step_reported = False
            print(f"[EP] new draw: {format_dr_line(e, eid_watch)}")
            log_buffer = new_log_buffer()
            log_start = time.monotonic()
            log_active = True
            log_failures = 0  # fresh run, fresh chance to warn loudly
            if summary_printer is not None:
                summary_printer._first = True
            # Intentionally no `continue` here — step+append runs below.

        # ── H: stop log run + save ────────────────────────────────────────────
        if keyboard.stop_log_requested:
            keyboard.stop_log_requested = False
            if log_active and log_buffer is not None:
                try:
                    fname, n = save_log(log_buffer, resume_path, control_hz)
                    print(f"[LOG] Saved {n} ticks to {fname}")
                except Exception as ex:
                    print(f"[WARN] log save failed: {ex}")
                log_active = False
                log_buffer = None
            else:
                print("[INFO] H pressed but no active log to save.")
            if summary_printer is not None:
                summary_printer._first = True

        # ── Autolog driver (only when --autolog_steps > 0) ───────────────────
        # Reuses the B/H branches above by setting their request flags, so
        # save/reset semantics stay identical to the interactive path.
        if args_cli.autolog_steps > 0:
            if not log_active:
                # Between runs (and on the very first tick), report how close
                # the toes actually came to the ground. A plant/physics test
                # whose feet never touch cannot show a foot-material effect, and
                # that is invisible in joint angles alone.
                try:
                    z1, z2 = get_toe_z(e, eid_watch)
                    _min_toe = min(_min_toe, z1, z2)
                except Exception:
                    pass
            if (
                log_active
                and log_buffer is not None
                and len(log_buffer["time"]) >= args_cli.autolog_steps
            ):
                keyboard.stop_log_requested = True
                autolog_runs_left -= 1
                print(f"[INFO] lowest toe height this run: {_min_toe:+.4f} m "
                      f"(env's own contact line is {first_step_thr:.3f} m)")
                _min_toe = float("inf")
                continue  # next iteration: H branch saves the buffer
            if not log_active:
                if autolog_runs_left > 0:
                    keyboard.start_log_requested = True
                    continue  # next iteration: B branch resets + arms log
                print("[INFO] Autolog complete — exiting.")
                break

        start_time = time.time()
        # Policy first, BEFORE env.step. We need `actions` to log, but we
        # also need `obs` and env state to still reflect the pre-step values
        # (the same state the NN just consumed) when the log block reads them.
        with torch.inference_mode():
            actions = policy(obs)

        # ── Per-tick logging (mirrors balance.py: state at the START of this
        #    control tick, plus the policy's response to that state) ──────────
        # See play_log.py for the full sim-vs-hardware reconciliation notes;
        # this block is identical.
        if log_active and log_buffer is not None:
            try:
                eid = eid_watch
                joint_ids = e._joint_ids

                # The obs that was just fed to the policy this tick.
                obs_tensor = (
                    obs if isinstance(obs, torch.Tensor) else obs['policy']
                )
                obs_36 = obs_tensor[eid, :OBS_PER_FRAME] \
                    .detach().cpu().numpy().astype(np.float32)

                # Action: the policy's raw output for THIS tick's obs.
                action_clip = actions[eid] \
                    .detach().cpu().numpy().astype(np.float32)
                target = (SIM_DEFAULT_JOINT_POS
                          + action_clip * SIM_ACTION_SCALES).astype(np.float32)

                # Pre-step joint state — same numbers that are baked into the
                # logged obs (after subtracting default pos and dropping units).
                sim_joint_pos = e.robot.data.joint_pos[eid, joint_ids] \
                    .detach().cpu().numpy().astype(np.float32)
                sim_joint_vel = e.robot.data.joint_vel[eid, joint_ids] \
                    .detach().cpu().numpy().astype(np.float32)

                # Joint-frame torques; per-env effective Kp/Kd live on the
                # actuator objects (explicit-PD DCMotor), reached through the
                # env's _actuator_joint_map — see play_log.py for rationale.
                joint_kp_t = torch.zeros(6, device=e.device)
                joint_kd_t = torch.zeros(6, device=e.device)
                for _name, _pairs in e._actuator_joint_map.items():
                    _act = e.robot.actuators[_name]
                    for _my_slot, _local in _pairs:
                        joint_kp_t[_my_slot] = _act.stiffness[eid, _local]
                        joint_kd_t[_my_slot] = _act.damping[eid, _local]
                joint_kp = joint_kp_t.detach().cpu().numpy().astype(np.float32)
                joint_kd = joint_kd_t.detach().cpu().numpy().astype(np.float32)
                tau_cmd_joint = (
                    joint_kp * (target - sim_joint_pos)
                    - joint_kd * sim_joint_vel
                ).astype(np.float32)
                tau_meas_joint = e.robot.data.applied_torque[eid, joint_ids] \
                    .detach().cpu().numpy().astype(np.float32)

                # Pre-step IMU (same as what's embedded in obs).
                root_quat = e.robot.data.root_quat_w[eid:eid + 1]
                root_ang_vel = e.robot.data.root_ang_vel_w[eid:eid + 1]
                gravity_vec = torch.tensor(
                    [[0.0, 0.0, -1.0]], device=root_quat.device
                )
                proj_grav = quat_rotate_inverse(root_quat, gravity_vec)[0] \
                    .detach().cpu().numpy().astype(np.float32)
                ang_vel = quat_rotate_inverse(root_quat, root_ang_vel)[0] \
                    .detach().cpu().numpy().astype(np.float32)

                gait_phase = float(e._gait_phase[eid].item())

                log_buffer['time'].append(len(log_buffer['time']) * dt)
                log_buffer['obs'].append(obs_36)
                log_buffer['action'].append(action_clip)
                log_buffer['target'].append(target)
                log_buffer['sim_joint_pos'].append(sim_joint_pos)
                log_buffer['sim_joint_vel'].append(sim_joint_vel)
                log_buffer['motor_tau_cmd'].append(tau_cmd_joint)
                log_buffer['motor_tau_meas'].append(tau_meas_joint)
                log_buffer['motor_temp'].append(
                    np.zeros(6, dtype=np.float32)
                )
                log_buffer['proj_grav'].append(proj_grav)
                log_buffer['ang_vel'].append(ang_vel)
                log_buffer['gait_phase'].append(gait_phase)
                log_buffer['mode'].append('NN_CONTROL')
            except Exception as ex:
                log_failures += 1
                if log_failures == 1:
                    # One-shot dump on first failure — enough context to
                    # pinpoint the offending line. Subsequent failures stay
                    # silent because the obs-summary panel ANSI-overwrites
                    # repeated prints anyway.
                    obs_shape = (
                        tuple(obs.shape) if hasattr(obs, 'shape')
                        else (f"len={len(obs)}" if hasattr(obs, '__len__') else "<?>")
                    )
                    print(f"\n[WARN] log append failed (silencing further warnings this run):")
                    print(f"        obs:       type={type(obs).__name__}  shape/len={obs_shape}")
                    print(f"        eid={args_cli.watch_env}  num_envs={env.unwrapped.num_envs}")
                    print(f"        error:     {type(ex).__name__}: {ex}")
                    traceback.print_exc()
                    if summary_printer is not None:
                        summary_printer._first = True

        # Step the env AFTER the log block so obs / robot.data.* above all
        # describe the start of this control tick, not the next one.
        with torch.inference_mode():
            obs, _, dones, _ = env.step(actions)
            obs = apply_lateral_mirror(obs, args_cli.mirror_lateral_imu)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)

        # ── Per-episode failure trace ────────────────────────────────────────
        # reset_terminated / reset_time_outs reflect the envs that were
        # auto-reset inside this step. The auto-reset already drew fresh DR
        # and re-pinned the gait clock (via install_gait_phase_pin).
        ticks_since_reset += 1
        try:
            fell = bool(e.reset_terminated[eid_watch].item())
            timed_out = bool(e.reset_time_outs[eid_watch].item())
        except Exception:
            fell, timed_out = False, False
        if fell or timed_out:
            ep_count += 1
            dur = ticks_since_reset * dt
            reason = "FALL" if fell else "timeout"
            print(f"\n[EP {ep_count}] {reason} after {dur:.2f}s")
            print(f"[EP {ep_count}] next draw: {format_dr_line(e, eid_watch)}")
            ticks_since_reset = 0
            first_step_reported = False
            if summary_printer is not None:
                summary_printer._first = True
        elif not first_step_reported:
            # ── First-step detector ──────────────────────────────────────────
            # Ground truth for which INDEX leg lifts first, independent of how
            # the viewport renders left/right. Skipped on the tick an auto-
            # reset happened (state already belongs to the new episode).
            try:
                z1, z2 = get_toe_z(e, eid_watch)
                if z1 > first_step_thr or z2 > first_step_thr:
                    which = (
                        "LEG1 (toe1/shin_inner, URDF y=+0.08 — config.py maps to hardware LEFT 103/101/102)"
                        if z1 > z2 else
                        "LEG2 (toe2/nail___fillet1, URDF y=-0.08 — config.py maps to hardware RIGHT 106/104/105)"
                    )
                    phase_now = _safe_item(e._gait_phase[eid_watch])
                    print(
                        f"\n[FIRST STEP] t={ticks_since_reset * dt:.2f}s  "
                        f"phase={phase_now:.2f} rad  ->  {which}"
                    )
                    first_step_reported = True
                    if summary_printer is not None:
                        summary_printer._first = True
            except Exception:
                first_step_reported = True  # don't retry a broken read every tick

        # ── 2Hz terminal summary refresh ──────────────────────────────────────
        if summary_printer is not None:
            summary_printer.maybe_print(env, args_cli.watch_env)

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # ── Flush log on quit if user didn't press H first ────────────────────────
    if log_active and log_buffer is not None and len(log_buffer['time']) > 0:
        try:
            fname, n = save_log(log_buffer, resume_path, control_hz)
            print(f"\n[LOG] Saved {n} ticks to {fname} (on exit)")
        except Exception as ex:
            print(f"\n[WARN] log save failed: {ex}")

    # close the simulator
    env.close()
    keyboard.shutdown()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
