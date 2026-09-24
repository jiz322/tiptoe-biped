# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""
rl_env_cfg.py — PHASE 4c: Velocity-x + yaw-rate control walking.

CHANGES from phase 4b:
    - COM-x command REPLACED by velocity-x command: the robot is now
      commanded a forward/backward speed (m/s) instead of a static
      COM offset. This directly drives stepping behavior.
    - YAW RATE COMMAND kept from 4b.
    - ROLL SATURATION FIX: action_scale_hip_roll reduced 0.40→0.15,
      hip_roll_split penalty enabled at -2.0.
    - STANDUP CURRICULUM UNLOCKED: starts at alpha=0.0, advances to 1.0.
    - standup_progress reward increased 2.0→10.0.
    - action_magnitude penalty exempted during grace (in env code).
    - Obs remains 36D (vx_cmd replaces com_cmd_dx slot).
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
    # 36D obs layout:
    #   projected_gravity(3) + ang_vel(3) + lin_vel(3)
    #   + joint_pos_norm(6) + joint_vel(6) + last_actions(6)
    #   + gait_sin(1) + gait_cos(1)
    #   + difficulty(1) + grace_norm(1) + init_h(1)
    #   + vx_cmd(1) + yaw_rate_cmd(1)
    #   + yaw_err_sin(1) + yaw_err_cos(1)
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
                stiffness=60.0,     # was 8.0 — stiff for lateral balance
                damping=2.0,
                effort_limit=6.0,
            ),
            "hip_pitch": ImplicitActuatorCfg(
                joint_names_expr=["hip_pitch.*"],
                stiffness=40.0,     # was 80.0 — softer for stride compliance
                damping=2.0,
                effort_limit=17.0,
            ),
            "knee": ImplicitActuatorCfg(
                joint_names_expr=["knee.*"],
                stiffness=40.0,     # was 60.0 — slightly softer
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
    action_scale_hip_roll: float = 0.25    # was 0.15
    action_scale_hip_pitch: float = 0.50   # was 0.30
    action_scale_knee: float = 0.45        # was 0.25

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

    # ══════════ STANDUP CURRICULUM (UNLOCKED) ══════════════════
    curriculum_enable: bool = True
    curriculum_initial_alpha: float = 0.0      # was 1.0 — start easy
    curriculum_max_alpha: float = 1.0
    curriculum_alpha_step: float = 0.05        # was 0.01 — faster ramp
    curriculum_advance_threshold: float = 0.7
    curriculum_advance_window: int = 200
    curriculum_advance_cooldown: int = 5

    standup_grace_seconds: float = 5.0

    # ══════════ VELOCITY-X COMMAND (replaces COM-x) ════════════
    # Forward/backward velocity in m/s, sampled each command interval.
    vx_cmd_min: float = -0.15                 # m/s backward
    vx_cmd_max: float = 0.40                  # m/s forward
    vx_cmd_zero_probability: float = 0.15

    # ══════════ YAW RATE COMMAND (steering) ════════════════════
    yaw_rate_cmd_min: float = -0.5
    yaw_rate_cmd_max: float = 0.5
    yaw_rate_cmd_zero_probability: float = 0.25
    yaw_rate_cmd_anchor_threshold: float = 0.05

    # ══════════ VELOCITY TRACKING ══════════════════════════════
    vel_tracking_sigma: float = 0.25          # m/s tolerance
    rew_scale_vel_tracking: float = 8.0

    # ══════════ YAW RATE TRACKING ══════════════════════════════
    yaw_rate_tracking_sigma: float = 0.30
    rew_scale_yaw_rate: float = 4.0

    # ══════════ LEG ALTERNATION (anti-gallop) ══════════════════
    leg_symmetry_decay: float = 0.98
    rew_scale_leg_symmetry: float = -30.0

    # ══════════ ZERO-COMMAND ANCHORING ═════════════════════════
    rew_scale_position_anchor: float = -8.0
    rew_scale_heading_anchor: float = -8.0

    # ══════════ BACK-LEG STEP-FORWARD REWARD ═══════════════════
    rew_scale_back_leg_step: float = 80.0

    # ══════════ BODY MASSES (from URDF) ════════════════════════
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
        1.698,
        1.1141,
        0.434,
        0.1905,
        1.1141,
        0.434,
        0.1905,
    ]

    # ── Reward scales ──────────────────────────────────────────
    rew_scale_alive = 5.0
    rew_scale_upright = 6.0
    rew_scale_standup_progress: float = 10.0   # was 2.0 — much stronger

    rew_scale_gait_step: float = 4.0
    rew_scale_feet_air: float = 1.5
    rew_scale_foot_swing: float = 1.0
    rew_scale_foot_progress: float = 100.0

    rew_scale_terminated = -10.0
    rew_scale_height = -12.0
    rew_scale_height = 8.0
    rew_scale_energy = -0.005
    rew_scale_action_rate = -1.5
    rew_scale_action_magnitude = -0.1
    rew_scale_joint_vel = -0.002
    rew_scale_joint_pos_limits = -0.1

    # Roll split penalty ENABLED to discourage roll saturation
    rew_scale_hip_roll_split: float = -2.0     # was -0.0
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