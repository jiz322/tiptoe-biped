# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""
rsl_rl_ppo_cfg.py — PHASE 5: Torque curriculum for sim-to-real.

Resumes from Phase 4c checkpoint. Lower learning rate to avoid
destroying the learned gait while adapting to tighter torque limits.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 128
    max_iterations = 10000         # longer — torque curriculum needs time
    save_interval = 100
    experiment_name = "bipedal_phase5_torque_curriculum"

    # ══════════ RESUME FROM PHASE 4c ══════════════════════════
    resume = True
    load_run = "phase4c_best"      # ← adjust to your phase 4c run name
    load_checkpoint = "model_14995.pt"  # ← adjust to your checkpoint name

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
        entropy_coef=0.0005,
        num_learning_epochs=3,
        num_mini_batches=2,
        learning_rate=1.0e-4,       # slower than phase 4c — preserve gait
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,           # tighter KL — don't jump too far
        max_grad_norm=1.0,
    )
