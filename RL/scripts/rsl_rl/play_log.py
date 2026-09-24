# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# ── NEW: terminal observation summary args ─────────────────────────────────────
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
import omni.appwindow
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
from isaaclab.utils.math import quat_rotate_inverse

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import RL.tasks  # noqa: F401


# ═════════════════════════════════════════════════════════════════════════════
# Run logger — mirrors balance.py LOG_CHANNELS exactly so the same offline
# analysis script can ingest sim and hardware runs side-by-side.
# ═════════════════════════════════════════════════════════════════════════════
#
# Sim-vs-hardware reconciliations:
#   - motor_temp     : zeros (no thermal model in sim).
#   - motor_tau_cmd  : applied_torque in JOINT frame (not motor frame). The
#     hardware log projects through JOINT_SIGN / KNEE_GEAR; we don't have
#     config.py here. Apply the projection externally if your analysis needs
#     it, or hardcode the values in this file to match balance.py.
#   - motor_tau_meas : same as motor_tau_cmd (sim has no separate sensed
#     torque — what the actuator applied IS what was "measured").
#   - mode           : always 'NN_CONTROL' (play.py has no HOLD/OBSERVE).
#
# Trigger is identical to balance.py:
#   B  -> env.reset() + start a fresh log buffer
#   H  -> stop logging, dump to <checkpoint>_simlog_<timestamp>.npz in CWD
#   quit while logging -> buffer is flushed on exit so the run isn't lost

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
# Terminal observation-summary panel
# ═════════════════════════════════════════════════════════════════════════════
#
# Renders a fixed-position dashboard with the things you cannot read off the
# Isaac Sim viewport: the current vx / yaw_rate commands, tracking error,
# pre-clip vs clipped actions (saturation diagnosis), applied joint torques,
# and the per-env DR draw that's currently active. Refreshes at `summary_hz`
# (wall-clock) using ANSI cursor-up to overwrite in place.
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
    # raw_actions is what the network output BEFORE clamp to [-1, 1] — drift to
    # ±10 here is exactly the sim2real failure mode the env config docstring
    # calls out. Comparing the two rows tells you per-joint if you're saturating.
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
        f"  OBSERVATION SUMMARY   "
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
    # Per-joint saturation flag — easy visual scan for which joints rail
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
            # Pad/truncate so width is stable and ANSI overwrite looks clean
            sys.stdout.write(line.ljust(_PANEL_WIDTH)[:_PANEL_WIDTH] + "\033[K\n")
        sys.stdout.flush()
        self._lines_written = len(lines)


# ═════════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════════
# Keyboard control
# ═════════════════════════════════════════════════════════════════════════════
class KeyboardController:
    """Subscribe to carb keyboard events and expose flags to the play loop.

    Keys handled:
        R   – request a full env reset on the next loop iteration
        B   – reset + start a log run (mirrors balance.py 'b')
        H   – stop the log run and save it (mirrors balance.py 'h')
    """

    def __init__(self):
        self.reset_requested = False
        self.start_log_requested = False
        self.stop_log_requested = False
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
        # Return True so other subscribers (e.g. viewport camera) still get the event
        return True

    def shutdown(self):
        if self._sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
            self._sub = None

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
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

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
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
    keyboard = KeyboardController()
    print("[INFO] Keyboard: focus the Isaac Sim window.")
    print("[INFO]   R = reset all envs")
    print("[INFO]   B = reset + start log run")
    print("[INFO]   H = stop log run + save to <ckpt>_simlog_<ts>.npz")

    # ── Logging state (mirrors balance.py) ───────────────────────────────────
    # log_active gates per-tick appends; log_buffer is the dict-of-lists from
    # new_log_buffer(); log_start is the monotonic anchor for the 'time' channel.
    # log_failures counts append exceptions per run — first one dumps a full
    # traceback, the rest are silenced (the obs-summary panel's ANSI overwrite
    # would clobber repeated warnings anyway).
    log_active = False
    log_buffer = None
    log_start = 0.0
    log_failures = 0

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # ── Keyboard-triggered reset (R) ──────────────────────────────────────
        if keyboard.reset_requested:
            keyboard.reset_requested = False
            print("[INFO] R pressed — resetting all envs.")
            with torch.inference_mode():
                obs, _ = env.reset()
                all_dones = torch.ones(
                    env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device
                )
                policy_nn.reset(all_dones)
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
            with torch.inference_mode():
                obs, _ = env.reset()
                all_dones = torch.ones(
                    env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device
                )
                policy_nn.reset(all_dones)
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

        start_time = time.time()
        # Policy first, BEFORE env.step. We need `actions` to log, but we
        # also need `obs` and env state to still reflect the pre-step values
        # (the same state the NN just consumed) when the log block reads them.
        with torch.inference_mode():
            actions = policy(obs)

        # ── Per-tick logging (mirrors balance.py: state at the START of this
        #    control tick, plus the policy's response to that state) ──────────
        # Everything below describes the state at t=i — the same snapshot that
        # fed `policy(obs)` above and produced `actions`. balance.py achieves
        # this by reading sensors at the top of its loop and consuming the
        # same buffers in build_obs() and the append block; we achieve it by
        # logging in between policy() and env.step() so env.robot.data.* are
        # still pre-step.
        #
        # tau_meas here is pre-step applied_torque — the torque applied during
        # the PREVIOUS step, which is what's been active on the joints up to
        # this moment. That matches balance.py's tau_meas (bridge.motor_torque
        # read at the top of the loop = motor's response to the previous
        # send_target). On the very first logged tick after B, this will be
        # the post-reset value (typically zero).
        if log_active and log_buffer is not None:
            try:
                e = env.unwrapped
                eid = max(0, min(args_cli.watch_env, e.num_envs - 1))
                joint_ids = e._joint_ids

                # The obs that was just fed to the policy this tick.
                # Modern IsaacLab + RSL-RL returns a TensorDict; older returns
                # the tensor directly. Handle both.
                obs_tensor = (
                    obs if isinstance(obs, torch.Tensor) else obs['policy']
                )
                obs_36 = obs_tensor[eid, :OBS_PER_FRAME] \
                    .detach().cpu().numpy().astype(np.float32)

                # Action: the policy's raw output for THIS tick's obs.
                # Mirrors balance.py's `last_actions = action.copy()` — that
                # is, the raw policy return, not whatever the env's action
                # manager later writes into e.actions after clipping.
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

                # Joint-frame torques. Hardware log is motor-frame (projected
                # via JOINT_SIGN / KNEE_GEAR); apply that externally if needed.
                #
                # tau_cmd: recomputed in-script PD law, exactly like
                #   balance.py's sim_tau_cmd:  Kp*(target-pos) - Kd*vel
                #   using the env's effective per-env Kp/Kd (already includes
                #   any DR stiffness/damping multipliers).
                # tau_meas: pre-step applied_torque (see block comment above).
                #
                # DCMotor (and every other explicit actuator) runs its PD law
                # in Python and writes only the resulting torque to PhysX, so
                # robot.data.joint_stiffness / joint_damping are flatly zero
                # for these robots — the real per-env effective gains live on
                # the actuator object. The env builds an _actuator_joint_map
                # of (sim-order slot, actuator-local idx) pairs at __init__
                # time for exactly this kind of indexing; we walk it to fill
                # a 6-vector in sim joint order. If you switch back to
                # ImplicitActuatorCfg this still works — actuator.stiffness
                # mirrors robot.data.joint_stiffness in that case.
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
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)

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