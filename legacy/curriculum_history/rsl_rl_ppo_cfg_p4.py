# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""
rsl_rl_ppo_cfg.py — PHASE 4: COM-feet offset control walking.

PPO TUNING NOTES (96 steps/env):
    Problem: noise_std drops to 0.35 then rises back to 0.5.
    Root cause: num_mini_batches=1 with 96*4096=393k samples per
    gradient step → updates too large → policy overshoots → destabilizes.

    Fixes:
    1. num_mini_batches: 1 → 4 (each batch ~98k, more stable updates)
    2. num_learning_epochs: 3 → 5 (more passes to extract value from
       the expensive 96-step rollout)
    3. learning_rate: 3e-4 → 1e-4 (slower, stabler with longer rollouts)
    4. value_loss_coef: 1.0 → 0.5 (value_loss was 34k, swamping actor)
    5. entropy_coef: 0.001 → 0.005 (help maintain exploration without
       noise_std rebounding — entropy bonus is smoother than noise)
    6. desired_kl: 0.01 → 0.008 (tighter KL bound prevents big jumps)
    7. gamma: 0.99 → 0.995 (96 steps sees more future — let gamma
       match the longer horizon so the critic values stepping)
    8. lam: 0.95 → 0.97 (less GAE bias, more variance — acceptable
       with 4 mini-batches smoothing things out)
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
    max_iterations = 5000
    save_interval = 100
    experiment_name = "bipedal_phase4_add_yaw"

    # ══════════ RESUME FROM PATCHED PHASE 3 ══════════════════
    resume = True
    load_run = "phase3_patched"
    load_checkpoint = "model_for_p4_patched.pt"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,            # back to original
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0005,             # back to original
        num_learning_epochs=3,          # back to original
        num_mini_batches=2,             # back to original
        learning_rate=3.0e-4,           # back to original
        schedule="adaptive",
        gamma=0.99,                     # back to original
        lam=0.95,                       # back to original
        desired_kl=0.01,                # back to original
        max_grad_norm=1.0,
    )