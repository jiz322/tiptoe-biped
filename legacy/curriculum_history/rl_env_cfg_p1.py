# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""
rl_env_cfg.py — PHASE 1: Balance only.

Curriculum DISABLED. Normal push forces. Observation space stays at 32
so the network architecture is identical between phases (required for
checkpoint loading). The extra 3 obs dims (difficulty, grace, init_height)
will be constant zeros/spawn_height during phase 1 — the normalizer will
handle this automatically.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


ROBOT_USD_PATH = "assets/tiptoe_v27/robot/robot.usd"  # historical: this phase trained on the earlier bigthigh-urdf export, which is not shipped


@configclass
class RlEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 20.0     # shorter — no standup needed

    # Same obs/act dims as phase 2 so checkpoint is compatible
    action_space = 6
    observation_space = 32
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD_PATH,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.42),
            joint_pos={
                "hip_roll1":  0.0,
                "hip_pitch1": 0.05,
                "knee1":      0.0,
                "hip_roll2":  0.0,
                "hip_pitch2": -0.05,
                "knee2":      0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "hip_roll": ImplicitActuatorCfg(
                joint_names_expr=["hip_roll.*"],
                stiffness=8.0,
                damping=0.5,
                effort_limit=1.8,
            ),
            "hip_pitch": ImplicitActuatorCfg(
                joint_names_expr=["hip_pitch.*"],
                stiffness=80.0,
                damping=3.0,
                effort_limit=17.0,
            ),
            "knee": ImplicitActuatorCfg(
                joint_names_expr=["knee.*"],
                stiffness=60.0,
                damping=2.0,
                effort_limit=23.2,
            ),
        },
    )

    imu_body_name: str = "imu"
    foot_body_name_1: str = "shin_inner"
    foot_body_name_2: str = "nail___fillet1"

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=2.0,
        replicate_physics=True,
    )

    action_scale_hip_roll: float = 0.10
    action_scale_hip_pitch: float = 0.4
    action_scale_knee: float = 0.35

    target_height = 0.35
    spawn_height = 0.42

    gait_freq: float = 1.5
    gait_phase_offset: float = 0.0

    # Full pushes from the start — the ONLY task is balance
    push_enable: bool = True
    push_probability: float = 0.003
    push_force_min: float = 15.0
    push_force_max: float = 60.0
    push_duration_steps: int = 5

    # Kneel pose (defined but not used in phase 1)
    kneel_height: float = 0.22
    kneel_quat_w: float = 0.7071
    kneel_quat_x: float = 0.0
    kneel_quat_y: float = -0.7071
    kneel_quat_z: float = 0.0
    kneel_hip_roll1: float = 0.0
    kneel_hip_pitch1: float = 3.0
    kneel_knee1: float = -3.0
    kneel_hip_roll2: float = 0.0
    kneel_hip_pitch2: float = -3.0
    kneel_knee2: float = 3.0

    # ══════════════ CURRICULUM DISABLED ════════════════════════
    curriculum_enable: bool = False
    curriculum_initial_alpha: float = 0.0
    curriculum_max_alpha: float = 1.0
    curriculum_alpha_step: float = 0.05
    curriculum_advance_threshold: float = 0.5
    curriculum_advance_window: int = 200
    curriculum_advance_cooldown: int = 10

    standup_grace_seconds: float = 5.0

    # ── Reward scales ─────────────────────────────────────────
    rew_scale_alive = 2.0
    rew_scale_upright = 5.0
    rew_scale_gait_step: float = 0.5
    rew_scale_feet_air: float = 0.0
    rew_scale_standup_progress: float = 3.0

    rew_scale_terminated = -10.0
    rew_scale_height = -10.0
    rew_scale_energy = -0.005
    rew_scale_action_rate = -0.1
    rew_scale_joint_vel = -0.001
    rew_scale_lin_vel_xy = -0.5
    rew_scale_ang_vel_z = -0.1
    rew_scale_joint_pos_limits = -0.3
    rew_scale_lin_pos_xy = -0.1

    # Split penalties — penalise approaching the termination thresholds
    rew_scale_hip_roll_split: float = -2.0
    rew_scale_front_split: float = -2.0
    rew_scale_knee_split: float = -2.0

    feet_air_height_threshold = 0.003

    max_tilt_rad = 1.0
    min_height = 0.26
    max_hip_roll_rad = 0.4
    max_hip_pitch_split_rad = 0.5
    max_knee_split_rad = 0.5

    noise_joint_pos = 0.1
    noise_base_vel_lin = 0.3
    noise_base_vel_ang = 0.2