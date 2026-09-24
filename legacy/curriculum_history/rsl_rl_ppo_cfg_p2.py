# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""
rsl_rl_ppo_cfg.py — PHASE 2: Standup curriculum.

Load the trained balance policy from phase 1 and enable curriculum
to progressively learn standup from increasingly kneeling poses.

USAGE:
    1. Find your best phase 1 checkpoint (highest ep length, lowest noise)
    2. Update `load_run` and `load_checkpoint` below
    3. Run training with this config + the phase 2 env config
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32          # longer rollouts for standup sequences
    max_iterations = 5000
    save_interval = 100
    experiment_name = "bipedal_phase2_standup"

    # ══════════ RESUME FROM PHASE 1 ══════════════════════════
    resume = True
    load_run = "phase1_checkpoint"            # ← directory you created above
    load_checkpoint = "model_1999.pt"          # ← the phase 1 final checkpoint

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,         # ignored when resuming
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,         # very low — policy already knows balance,
                                    # just needs to extend to standup
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,       # lower — fine-tuning, not learning from scratch
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,           # moderate — allow meaningful updates
        max_grad_norm=1.0,
    )