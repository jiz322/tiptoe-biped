# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""
rsl_rl_ppo_cfg.py — PHASE 6: Standup burst.

Resumes from Phase 5 checkpoint.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 128
    max_iterations = 10000         
    save_interval = 10
    experiment_name = "p9_replant"   # P9 fresh train on the measured plant (P7: rotor armature + coupling); was p8_belt_fresh
    # Actor: policy group only. Critic: policy + privileged group.
    obs_groups = {"policy": ["policy"], "critic": ["policy", "critic"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.15,
        # RESUME 2: 0.005 → 0.01. Noise std fell 0.47 → 0.33 (entropy
        # 3.75 → 1.66) within ~140 iters of the 5× action-penalty resume —
        # exploration collapsing exactly when discovering threshold-crossing
        # liftoffs requires it. init_noise_std is irrelevant on resume (std
        # loads from checkpoint); entropy_coef is the live lever. Walk back
        # toward 0.005 once idle_liftoff_rate is moving and stable
        # (≥0.03) — a permanently fat policy hurts final smoothness.
        # FRESH RUN 2026-08-07: 0.000 was the final-polish value — a fresh
        # policy with zero entropy bonus collapses its std before it finds
        # anything. Start at 0.005; drop toward 0 only in the polish phase.
        # POLISH RESUME 2026-08-10 (forked from C_hipfence @ iter 2630):
        # noise std decay had all but stopped (0.429→0.433 over the last
        # 16 logged iters) while ep_len was still (slowly) climbing —
        # first taper step, not the final 0.000; hip_sat_ema pinned
        # exactly at its 0.65 threshold said the policy was riding the
        # fence rather than still exploring around it.
        entropy_coef=0.003,
        num_learning_epochs=3,
        num_mini_batches=32,
        learning_rate=3.0e-4,       
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        # FRESH RUN 2026-08-07: 0.002 was the polish value (tiny steps to
        # protect a finished policy). Fresh learning needs real update
        # sizes or the adaptive schedule strangles the lr. Tighten back
        # toward 0.002-0.006 for later polish resumes.
        # POLISH RESUME 2026-08-10: tighten alongside entropy_coef, same
        # rationale — half-step, not straight to the polish floor.
        # P9 FRESH 2026-08-18: back to the documented fresh value (the 0.005
        # was a mid-polish leftover; the G2 lineage showed small desired_kl
        # + adaptive schedule pins lr at rsl_rl's 1e-5 floor once annealed).
        desired_kl=0.008,  # 0.008 fresh / 0.002 final polish
        max_grad_norm=1.0,

        # ══════════ NEW: MIRROR AUGMENTATION ══════════════════
        # Each minibatch becomes [original; mirrored], doubling effective
        # batch size. The function below is resolved by rsl_rl via
        # importlib at runtime — adjust the module path on the LHS of ':'
        # to match wherever rl_env.py lives in your package tree.
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func="RL.tasks.direct.standup_curriculum.rl_env_p6:get_symmetric_states",
            use_mirror_loss=True,
            mirror_loss_coeff=1.0,
        ),
    )