# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""
rl_env_cfg.py — PHASE 5: Torque curriculum for sim-to-real transfer.

CHANGES from phase 4c:
    - TORQUE CURRICULUM: gradually reduces effort_limit, stiffness,
      and damping from training values down to real hardware values.
    - All other rewards/parameters inherited from phase 4c.
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
    episode_length_s = 30.0

    action_space = 6
    observation_space = 36
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

    # Start with TRAINING values (permissive) — will be interpolated
    # toward real values by the torque curriculum.
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
                stiffness=60.0,
                damping=2.0,
                effort_limit=6.0,
            ),
            "hip_pitch": ImplicitActuatorCfg(
                joint_names_expr=["hip_pitch.*"],
                stiffness=40.0,
                damping=2.0,
                effort_limit=17.0,
            ),
            "knee": ImplicitActuatorCfg(
                joint_names_expr=["knee.*"],
                stiffness=40.0,
                damping=1.5,
                effort_limit=23.2,
            ),
        },
    )

    imu_body_name: str = "imu"
    foot_body_name_1: str = "shin_inner"
    foot_body_name_2: str = "nail___fillet1"

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192,
        env_spacing=2.5,
        replicate_physics=True,
    )

    # ── Action scales ──────────────────────────────────────────
    action_scale_hip_roll: float = 0.25
    action_scale_hip_pitch: float = 0.50
    action_scale_knee: float = 0.45

    target_height = 0.35
    spawn_height = 0.42

    gait_freq: float = 1.5
    gait_phase_offset: float = 0.0

    # Pushes ON
    push_enable: bool = True
    push_probability: float = 0.003
    push_force_min: float = 15.0
    push_force_max: float = 60.0
    push_duration_steps: int = 5

    # Kneel pose
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

    # ══════════ STANDUP CURRICULUM ═════════════════════════════
    curriculum_enable: bool = False
    curriculum_initial_alpha: float = 0.0
    curriculum_max_alpha: float = 1.0
    curriculum_alpha_step: float = 0.05
    curriculum_advance_threshold: float = 0.7
    curriculum_advance_window: int = 200
    curriculum_advance_cooldown: int = 5

    standup_grace_seconds: float = 5.0

    # ══════════ TORQUE CURRICULUM (NEW in Phase 5) ═════════════
    # Interpolates actuator params from training → real hardware.
    # torque_alpha=0 → training values, torque_alpha=1 → real values.
    torque_curriculum_enable: bool = True
    torque_alpha_initial: float = 0.0
    torque_alpha_max: float = 1.0
    torque_alpha_step: float = 0.02
    torque_advance_threshold: float = 0.65   # slightly easier than standup
    torque_advance_window: int = 300
    torque_advance_cooldown: int = 10

    # ── Training actuator values (current checkpoint) ──────────
    train_effort_hip_roll: float = 6.0
    train_effort_hip_pitch: float = 17.0
    train_effort_knee: float = 23.2

    train_kp_hip_roll: float = 60.0
    train_kp_hip_pitch: float = 40.0
    train_kp_knee: float = 40.0

    train_kd_hip_roll: float = 2.0
    train_kd_hip_pitch: float = 2.0
    train_kd_knee: float = 1.5

    # ── Real hardware values (target) ──────────────────────────
    real_effort_hip_roll: float = 1.8
    real_effort_hip_pitch: float = 6.0
    real_effort_knee: float = 8.18

    real_kp_hip_roll: float = 6.0
    real_kp_hip_pitch: float = 8.0
    real_kp_knee: float = 16.0

    real_kd_hip_roll: float = 0.3
    real_kd_hip_pitch: float = 0.5
    real_kd_knee: float = 1.0

    # ══════════ VELOCITY-X COMMAND ═════════════════════════════
    vx_cmd_min: float = -0.15
    vx_cmd_max: float = 0.40
    vx_cmd_zero_probability: float = 0.15

    # ══════════ YAW RATE COMMAND ═══════════════════════════════
    yaw_rate_cmd_min: float = -0.5
    yaw_rate_cmd_max: float = 0.5
    yaw_rate_cmd_zero_probability: float = 0.25
    yaw_rate_cmd_anchor_threshold: float = 0.05

    # ══════════ VELOCITY TRACKING ══════════════════════════════
    vel_tracking_sigma: float = 0.25
    rew_scale_vel_tracking: float = 8.0

    # ══════════ YAW RATE TRACKING ══════════════════════════════
    yaw_rate_tracking_sigma: float = 0.30
    rew_scale_yaw_rate: float = 4.0

    # ══════════ LEG ALTERNATION ════════════════════════════════
    leg_symmetry_decay: float = 0.98
    rew_scale_leg_symmetry: float = -30.0

    # ══════════ ZERO-COMMAND ANCHORING ═════════════════════════
    rew_scale_position_anchor: float = -8.0
    rew_scale_heading_anchor: float = -8.0

    # ══════════ BACK-LEG STEP-FORWARD ══════════════════════════
    rew_scale_back_leg_step: float = 80.0

    # ══════════ BODY MASSES ════════════════════════════════════
    com_body_names: list = [
        "torso2hip",
        "chasis_motor2motor",
        "thigh_outer",
        "shin_inner",
        "lid_bearing_outer___boss_extrude3",
        "stand_steel_00___fillet1",
        "nail___fillet1",
    ]
    com_body_masses: list = [
        1.698, 1.1141, 0.434, 0.1905,
        1.1141, 0.434, 0.1905,
    ]

    # ── Reward scales ──────────────────────────────────────────
    rew_scale_alive = 5.0
    rew_scale_upright = 6.0
    rew_scale_standup_progress: float = 10.0

    rew_scale_gait_step: float = 4.0
    rew_scale_feet_air: float = 1.5
    rew_scale_foot_swing: float = 1.0
    rew_scale_foot_progress: float = 100.0

    rew_scale_terminated = -10.0
    rew_scale_height: float = 8.0        # positive — exponential kernel
    rew_scale_energy = -0.005
    rew_scale_action_rate = -1.5
    rew_scale_action_magnitude = -0.1
    rew_scale_joint_vel = -0.002
    rew_scale_joint_pos_limits = -0.1

    rew_scale_hip_roll_split: float = -2.0
    rew_scale_front_split: float = -0.0
    rew_scale_knee_split: float = -0.0

    rew_scale_balance: float = -5.0

    feet_air_height_threshold = 0.005

    max_tilt_rad = 3.0
    min_height = 0.24
    max_hip_roll_rad = 1.8
    max_hip_pitch_split_rad = 1.8
    max_knee_split_rad = 1.8

    noise_joint_pos = 0.1
    noise_base_vel_lin = 0.3
    noise_base_vel_ang = 0.2

    cmd_resample_steps: int = 100
