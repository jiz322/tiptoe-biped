"""Script to play a checkpoint with configurable curriculum difficulty."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
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

# ── NEW: curriculum difficulty argument ───────────────────────
parser.add_argument(
    "--alpha", type=float, default=None,
    help="Test at a fixed curriculum difficulty. 0.0 = standing, 1.0 = full kneel. "
         "All envs will start at exactly this difficulty (not sampled from Uniform). "
         "If not set, uses the config defaults."
)
parser.add_argument(
    "--alpha-random", action="store_true", default=False,
    help="Sample each env's difficulty from Uniform(0, alpha) instead of using alpha exactly."
)

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

import os
import time

import gymnasium as gym
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

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import RL.tasks  # noqa: F401


def set_fixed_difficulty(env, alpha: float, randomize: bool = False):
    """Override curriculum state so all envs use a fixed difficulty."""
    # Freeze curriculum so it never advances during play
    env._curriculum_alpha = alpha
    env.cfg.curriculum_enable = True    # enable so reset uses interpolation
    env.cfg.curriculum_alpha_step = 0.0  # prevent advancement

    # Disable pushes during evaluation for clean visualization
    env.cfg.push_enable = False

    # ── NEW: Trigger the safety switch in the environment ──
    env._force_fixed_difficulty = not randomize

    print(f"\n{'=' * 50}")
    print(f"  PLAY MODE — Difficulty: {alpha:.2f}")
    print(f"  {'Uniform(0, ' + f'{alpha:.2f}' + ')' if randomize else f'Fixed at {alpha:.2f}'}")
    print(f"  0.0 = standing, 1.0 = full kneel")
    print(f"  Pushes: DISABLED")
    print(f"{'=' * 50}\n")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
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

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # ── Apply fixed difficulty if --alpha was specified ────────
    if args_cli.alpha is not None:
        alpha = max(0.0, min(1.0, args_cli.alpha))
        set_fixed_difficulty(env.unwrapped, alpha, randomize=args_cli.alpha_random)

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
    
    # ── NEW: Slow motion settings ───────────────────────────────
    slow_mo_factor = 1.0 
    target_dt = dt * slow_mo_factor
    print(f"\n[INFO] Running in {slow_mo_factor}x Slow Motion...")
    # ────────────────────────────────────────────────────────────
# ── NEW: Interactive Keyboard Controls ──────────────────────
    import carb
    import carb.input
    import omni.appwindow

    class SimControl:
        paused = False
        reset = False

    sim_control = SimControl()

    def keyboard_event_handler(event, *args, **kwargs):
        # Listen specifically for key press events
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            
            # [P] - Pause / Resume
            if event.input == carb.input.KeyboardInput.P:
                sim_control.paused = not sim_control.paused
                print(f"\n[INFO] Playback {'PAUSED' if sim_control.paused else 'RESUMED'}")
                
            # [R] - Reset Environment
            elif event.input == carb.input.KeyboardInput.R:
                sim_control.reset = True
                print("\n[INFO] Environment RESET requested...")
                
            # [,] (Comma) - Decrease Difficulty
            elif event.input == carb.input.KeyboardInput.COMMA:
                new_alpha = max(0.0, env.unwrapped._curriculum_alpha - 0.05)
                env.unwrapped._curriculum_alpha = new_alpha
                env.unwrapped._force_fixed_difficulty = True  # Ensure it stays fixed
                sim_control.reset = True
                print(f"\n[INFO] Difficulty DECREASED ⬇️ | Alpha: {new_alpha:.2f}")

            # [.] (Period) - Increase Difficulty
            elif event.input == carb.input.KeyboardInput.PERIOD:
                new_alpha = min(1.0, env.unwrapped._curriculum_alpha + 0.05)
                env.unwrapped._curriculum_alpha = new_alpha
                env.unwrapped._force_fixed_difficulty = True  # Ensure it stays fixed
                sim_control.reset = True
                print(f"\n[INFO] Difficulty INCREASED ⬆️ | Alpha: {new_alpha:.2f}")

    # Attach the keyboard listener to the active Omniverse window
    appwindow = omni.appwindow.get_default_app_window()
    if appwindow:
        input_interface = carb.input.acquire_input_interface()
        keyboard = appwindow.get_keyboard()
        input_interface.subscribe_to_keyboard_events(keyboard, keyboard_event_handler)
    # ────────────────────────────────────────────────────────────
    obs = env.get_observations()
    timestep = 0
    while simulation_app.is_running():
        # 1. Handle Pause State (Keep this outside inference mode)
        if sim_control.paused:
            simulation_app.update()  # Keeps the GUI responsive
            time.sleep(0.01)         # Prevents CPU maxing out
            continue                 # Skip the physics step

        start_time = time.time()
        
        # 2. Wrap BOTH Reset and Step in Inference Mode
        with torch.inference_mode():
            
            # ── Handle Reset Request ──
            if sim_control.reset:
                env.reset()
                obs = env.get_observations()
                # Force the policy hidden states to reset across all envs
                dones = torch.ones(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)
                policy_nn.reset(dones)
                sim_control.reset = False
                # ── NEW: Print the alpha values ──
                try:
                    # Fetch the difficulty tensor from the unwrapped environment
                    alphas = env.unwrapped._env_difficulty
                    
                    # Print just the first 5 to avoid console spam
                    # We move it to CPU and convert to numpy for clean formatting
                    print(f"\n[INFO] Reset Triggered! Alphas (first 5 envs): {alphas[:5].cpu().numpy().round(3)}")
                except AttributeError:
                    print(f"\n[INFO] Reset Triggered! (Note: _env_difficulty attribute not found in this env version)")
            # ── Normal Step Routine ──
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)
            
        # 3. Video & Timing logic
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        compute_time = time.time() - start_time
        sleep_time = target_dt - compute_time
        if sleep_time > 0:
            time.sleep(sleep_time)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()