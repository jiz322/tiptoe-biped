# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""
rsl_rl_ppo_cfg.py — PHASE 1: Balance only.

Train standing balance + push recovery WITHOUT curriculum.
This produces a solid foundation policy to build standup on top of.

After this converges (noise std < 0.5, episode length > 80%),
switch to phase 2 config and resume from the checkpoint.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    experiment_name = "bipedal_phase1_balance"
    # To resume phase 2 from this checkpoint, set:
    #   resume = True
    #   load_run = "bipedal_phase1_balance"
    #   load_checkpoint = "model_XXXX.pt"  (best checkpoint)

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
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
        entropy_coef=0.005,         # moderate — some exploration but not too much
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )