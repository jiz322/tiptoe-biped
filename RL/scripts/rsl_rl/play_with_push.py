# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Play a checkpoint with optional force perturbations for balance testing.

Force perturbation controls (keyboard):
    I / K     — push +X / -X  (forward / back)
    J / L     — push -Y / +Y  (left / right)
    R         — random push (random horizontal direction)
    U / O     — decrease / increase push magnitude
    P         — toggle periodic random pushes

Telemetry:
    Joint position targets, actual positions, tracking errors, and PD torques
    are printed to the terminal at a configurable interval (--print_interval).

Launch example:
    python play_with_push.py --task <task> --num_envs 1 --real-time
    python play_with_push.py --task <task> --num_envs 1 --push_force 80 --periodic_push
    python play_with_push.py --task <task> --num_envs 1 --print_interval 10
    python play_with_push.py --task <task> --num_envs 1 --print_interval 0   # every step
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# ── CLI arguments ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Play an RL agent with force perturbations.")
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

# ── Push-specific arguments ───────────────────────────────────
parser.add_argument("--push_force", type=float, default=50.0,
                    help="Initial push force magnitude in Newtons (default: 50 N).")
parser.add_argument("--push_duration", type=int, default=5,
                    help="Number of env steps to sustain each push (default: 5).")
parser.add_argument("--periodic_push", action="store_true", default=False,
                    help="Enable periodic random pushes from the start.")
parser.add_argument("--push_interval", type=int, default=150,
                    help="Steps between periodic pushes (default: 150).")
parser.add_argument("--torso_body", type=str, default="torso2hip",
                    help="Name of the torso/base body to push (default: 'torso2hip').")
parser.add_argument("--print_interval", type=int, default=50,
                    help="Print joint targets & torques every N steps. 0 = every step, -1 = off (default: 50).")
parser.add_argument("--print_env_id", type=int, default=0,
                    help="Which env index to print telemetry for (default: 0).")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
import numpy as np
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

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import RL.tasks  # noqa: F401


# ══════════════════════════════════════════════════════════════
# Force perturbation manager
# ══════════════════════════════════════════════════════════════

class PushManager:
    """Manages keyboard-triggered and periodic force perturbations."""

    def __init__(self, robot, torso_body_name: str, num_envs: int, device: str,
                 initial_force: float = 50.0, push_duration: int = 5,
                 periodic: bool = False, interval: int = 150):
        self.robot = robot
        self.num_envs = num_envs
        self.device = device
        self.force_magnitude = initial_force
        self.push_duration = push_duration
        self.periodic = periodic
        self.interval = interval

        # ── Resolve torso body index ──────────────────────────
        self._torso_ids, self._torso_names = robot.find_bodies([torso_body_name])
        if len(self._torso_ids) == 0:
            # Try common alternatives
            for alt_name in ["base", "base_link", "trunk", "body", "pelvis"]:
                self._torso_ids, self._torso_names = robot.find_bodies([alt_name])
                if len(self._torso_ids) > 0:
                    print(f"[PUSH] Body '{torso_body_name}' not found, using '{alt_name}' instead.")
                    break
            if len(self._torso_ids) == 0:
                # List all available bodies for the user
                all_bodies = robot.body_names
                print(f"[PUSH] WARNING: Could not find torso body. Available bodies: {all_bodies}")
                print(f"[PUSH] Re-run with --torso_body <name> using one of the above.")
                self._torso_ids = [0]
                self._torso_names = ["fallback_body_0"]

        print(f"[PUSH] Target body: {self._torso_names} (ids: {self._torso_ids})")

        # ── Force/torque buffers ──────────────────────────────
        num_bodies = robot.num_bodies
        self._forces = torch.zeros(num_envs, num_bodies, 3, device=device)
        self._torques = torch.zeros(num_envs, num_bodies, 3, device=device)

        # ── Push state ────────────────────────────────────────
        self._remaining_steps = 0
        self._step_counter = 0
        self._pending_push = None  # (fx, fy) or None

        # ── Keyboard setup ────────────────────────────────────
        self._setup_keyboard()

        self._print_controls()

    def _setup_keyboard(self):
        """Register keyboard callbacks via omni.appwindow."""
        import carb.input
        import omni.appwindow

        app_window = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = app_window.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_key)

        # Key mapping: carb key enum → action
        self._key_map = {
            carb.input.KeyboardInput.I: "push_+x",
            carb.input.KeyboardInput.K: "push_-x",
            carb.input.KeyboardInput.J: "push_-y",
            carb.input.KeyboardInput.L: "push_+y",
            carb.input.KeyboardInput.R: "push_random",
            carb.input.KeyboardInput.U: "force_down",
            carb.input.KeyboardInput.O: "force_up",
            carb.input.KeyboardInput.P: "toggle_periodic",
        }

    def _on_key(self, event, *args, **kwargs):
        """Keyboard callback — queues pushes on key press."""
        import carb.input
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True

        action = self._key_map.get(event.input, None)
        if action is None:
            return True

        if action == "push_+x":
            self._pending_push = (self.force_magnitude, 0.0)
            print(f"[PUSH] → +X  ({self.force_magnitude:.0f} N)")
        elif action == "push_-x":
            self._pending_push = (-self.force_magnitude, 0.0)
            print(f"[PUSH] → -X  ({self.force_magnitude:.0f} N)")
        elif action == "push_+y":
            self._pending_push = (0.0, self.force_magnitude)
            print(f"[PUSH] → +Y  ({self.force_magnitude:.0f} N)")
        elif action == "push_-y":
            self._pending_push = (0.0, -self.force_magnitude)
            print(f"[PUSH] → -Y  ({self.force_magnitude:.0f} N)")
        elif action == "push_random":
            angle = np.random.uniform(0, 2 * np.pi)
            fx = self.force_magnitude * np.cos(angle)
            fy = self.force_magnitude * np.sin(angle)
            self._pending_push = (fx, fy)
            print(f"[PUSH] → random ({fx:.1f}, {fy:.1f}) N")
        elif action == "force_up":
            self.force_magnitude = min(self.force_magnitude + 10.0, 500.0)
            print(f"[PUSH] Force magnitude: {self.force_magnitude:.0f} N")
        elif action == "force_down":
            self.force_magnitude = max(self.force_magnitude - 10.0, 5.0)
            print(f"[PUSH] Force magnitude: {self.force_magnitude:.0f} N")
        elif action == "toggle_periodic":
            self.periodic = not self.periodic
            print(f"[PUSH] Periodic pushes: {'ON' if self.periodic else 'OFF'}")

        return True

    def step(self):
        """Call once per env step. Applies/clears forces as needed."""
        self._step_counter += 1

        # ── Check for new push (keyboard or periodic) ─────────
        if self._pending_push is not None:
            self._start_push(*self._pending_push)
            self._pending_push = None
        elif self.periodic and self._step_counter % self.interval == 0:
            angle = np.random.uniform(0, 2 * np.pi)
            fx = self.force_magnitude * np.cos(angle)
            fy = self.force_magnitude * np.sin(angle)
            self._start_push(fx, fy)
            print(f"[PUSH] Periodic push #{self._step_counter // self.interval}: "
                  f"({fx:.1f}, {fy:.1f}) N")

        # ── Sustain or clear ──────────────────────────────────
        if self._remaining_steps > 0:
            self._remaining_steps -= 1
            # Force is already set in the buffer — just apply
            self.robot.set_external_force_and_torque(
                self._forces, self._torques
            )
        else:
            # Clear forces
            self._forces[:] = 0.0
            self._torques[:] = 0.0
            self.robot.set_external_force_and_torque(
                self._forces, self._torques
            )

    def _start_push(self, fx: float, fy: float):
        """Begin a push in world-frame XY."""
        self._forces[:] = 0.0
        self._torques[:] = 0.0
        for bid in self._torso_ids:
            self._forces[:, bid, 0] = fx
            self._forces[:, bid, 1] = fy
        self._remaining_steps = self.push_duration

    def _print_controls(self):
        print(f"\n{'─' * 60}")
        print(f"  FORCE PERTURBATION CONTROLS")
        print(f"{'─' * 60}")
        print(f"  I / K         — push +X / -X  (forward / back)")
        print(f"  J / L         — push -Y / +Y  (left / right)")
        print(f"  R             — random horizontal push")
        print(f"  U / O         — decrease / increase force ({self.force_magnitude:.0f} N)")
        print(f"  P             — toggle periodic pushes ({'ON' if self.periodic else 'OFF'})")
        print(f"  Push duration : {self.push_duration} steps")
        print(f"  Push interval : {self.interval} steps (periodic mode)")
        print(f"{'─' * 60}\n")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent + force perturbations."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

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
    env_cfg.log_dir = log_dir

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

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

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # ── Initialize push manager ───────────────────────────────
    # Access the underlying DirectRLEnv's robot articulation
    raw_env = env.unwrapped
    push_mgr = PushManager(
        robot=raw_env.robot,
        torso_body_name=args_cli.torso_body,
        num_envs=raw_env.num_envs,
        device=raw_env.device,
        initial_force=args_cli.push_force,
        push_duration=args_cli.push_duration,
        periodic=args_cli.periodic_push,
        interval=args_cli.push_interval,
    )

    # ── Play loop ─────────────────────────────────────────────
    obs = env.get_observations()
    timestep = 0

    # Telemetry setup
    joint_ids = raw_env._joint_ids
    joint_names = raw_env._joint_names
    eid = min(args_cli.print_env_id, raw_env.num_envs - 1)
    print_interval = args_cli.print_interval

    # Print header once
    if print_interval >= 0:
        name_strs = [f"{n:>12s}" for n in joint_names]
        header = " | ".join(name_strs)
        print(f"\n[TELEMETRY] Printing env {eid} every {print_interval if print_interval > 0 else 1} step(s)")
        print(f"[TELEMETRY] Joints: {joint_names}\n")

    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            actions = policy(obs)
            # Apply force perturbation BEFORE env step
            push_mgr.step()
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        timestep += 1

        # ── Print joint telemetry ─────────────────────────────
        if print_interval >= 0 and (print_interval == 0 or timestep % print_interval == 0):
            pos_target = raw_env.robot.data.joint_pos_target[eid, joint_ids]
            pos_actual = raw_env.robot.data.joint_pos[eid, joint_ids]
            torques = raw_env.robot.data.applied_torque[eid, joint_ids]

            # Format as compact table
            tgt_str = "  ".join(f"{v:+7.3f}" for v in pos_target.cpu().tolist())
            act_str = "  ".join(f"{v:+7.3f}" for v in pos_actual.cpu().tolist())
            tau_str = "  ".join(f"{v:+7.2f}" for v in torques.cpu().tolist())
            err_str = "  ".join(f"{v:+7.3f}" for v in (pos_target - pos_actual).cpu().tolist())

            print(f"[step {timestep:5d}] ──────────────────────────────────")
            print(f"  target pos : {tgt_str}")
            print(f"  actual pos : {act_str}")
            print(f"  pos error  : {err_str}")
            print(f"  torque     : {tau_str}")

        if args_cli.video:
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()