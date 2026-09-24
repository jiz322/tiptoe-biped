# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""
rsl_rl_ppo_cfg.py — PHASE 3: Push recovery.

Resume from the best phase 2 checkpoint.
Same architecture, moderate training — the policy already balances
and stands up, it just needs to handle full-strength pushes.

This phase should converge relatively quickly (500-1500 iterations)
since the foundation is very strong.
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
    max_iterations = 1000
    save_interval = 100
    experiment_name = "bipedal_phase3_pushrecovery"

    # ══════════ RESUME FROM PHASE 2 ══════════════════════════
    resume = True
    load_run = "phase2_checkpoint"           
    load_checkpoint = "model_8997_reduced.pt"           

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
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )