# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""
rl_env.py — Bipedal point-foot balance environment with curriculum standup.

CHANGES (v5.1 — action magnitude penalty):
    Added rew_scale_action_magnitude to combat rising noise std.
    This penalty applies ALWAYS (including during grace period),
    giving the policy continuous gradient pressure toward precision.
    
    All other behavior identical to v5.
"""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_rotate_inverse, sample_uniform

from .rl_env_cfg_p3 import RlEnvCfg


# ══════════════════════════════════════════════════════════════
# 3D Forward Kinematics from URDF joint transforms
# ══════════════════════════════════════════════════════════════
_FK_TARGET_FOOT_Z = 0.0716


def _batched_mat4_mul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.bmm(A, B)


def _make_Rz(angles: torch.Tensor) -> torch.Tensor:
    N = angles.shape[0]
    c = torch.cos(angles)
    s = torch.sin(angles)
    m = torch.zeros(N, 4, 4, device=angles.device)
    m[:, 0, 0] = c;  m[:, 0, 1] = -s
    m[:, 1, 0] = s;  m[:, 1, 1] = c
    m[:, 2, 2] = 1.0
    m[:, 3, 3] = 1.0
    return m


def _fixed_joint_transform(xyz, rpy, device) -> torch.Tensor:
    import math as _m
    x, y, z = xyz
    roll, pitch, yaw = rpy
    cr, sr = _m.cos(roll), _m.sin(roll)
    cp, sp = _m.cos(pitch), _m.sin(pitch)
    cy, sy = _m.cos(yaw), _m.sin(yaw)
    R = torch.tensor([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr, x],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr, y],
        [-sp,   cp*sr,            cp*cr,             z],
        [0.0,   0.0,              0.0,               1.0],
    ], device=device).unsqueeze(0)
    return R


def _joint_transform_batched(fixed, angles):
    Rz = _make_Rz(angles)
    return _batched_mat4_mul(fixed.expand(angles.shape[0], -1, -1), Rz)


def _compute_fk_height(base_quat, hip_roll1, hip_pitch1, knee1,
                        hip_roll2, hip_pitch2, knee2, device):
    N = base_quat.shape[0]
    w, x, y, z = base_quat[:, 0], base_quat[:, 1], base_quat[:, 2], base_quat[:, 3]
    T_base = torch.zeros(N, 4, 4, device=device)
    T_base[:, 0, 0] = 1 - 2*(y*y + z*z)
    T_base[:, 0, 1] = 2*(x*y - w*z)
    T_base[:, 0, 2] = 2*(x*z + w*y)
    T_base[:, 1, 0] = 2*(x*y + w*z)
    T_base[:, 1, 1] = 1 - 2*(x*x + z*z)
    T_base[:, 1, 2] = 2*(y*z - w*x)
    T_base[:, 2, 0] = 2*(x*z - w*y)
    T_base[:, 2, 1] = 2*(y*z + w*x)
    T_base[:, 2, 2] = 1 - 2*(x*x + y*y)
    T_base[:, 3, 3] = 1.0

    if not hasattr(_compute_fk_height, "_cached_device") or _compute_fk_height._cached_device != device:
        _compute_fk_height._cached_device = device
        _compute_fk_height._F_hr1 = _fixed_joint_transform(
            (-0.070392, 0.0800218, -0.0315), (1.5708, 0.0, 1.5708), device)
        _compute_fk_height._F_hp1 = _fixed_joint_transform(
            (-0.0415, 0.0, -0.071), (0.0, 1.5708, 0.0), device)
        _compute_fk_height._F_kn1 = _fixed_joint_transform(
            (-0.0592463, -0.172173, 0.0969), (0.0, 0.0, 1.72565), device)
        _compute_fk_height._T_nail1 = torch.tensor([
            [1, 0, 0, -0.138448], [0, 1, 0, 0.0216628],
            [0, 0, 1, -0.01835], [0, 0, 0, 1.0],
        ], device=device).unsqueeze(0)
        _compute_fk_height._F_hr2 = _fixed_joint_transform(
            (-0.070392, -0.0799782, -0.0315), (-3.14159, -1.5708, 0.0), device)
        _compute_fk_height._F_hp2 = _fixed_joint_transform(
            (0.0, -0.0415, -0.071), (-1.5708, 0.908015, 0.0), device)
        _compute_fk_height._F_kn2 = _fixed_joint_transform(
            (-0.154185, 0.107117, 0.0602), (0.0, 0.0, 0.662781), device)
        _compute_fk_height._T_nail2 = torch.tensor([
            [1, 0, 0, -4.95155e-05], [0, 1, 0, 0.140191],
            [0, 0, 1, 0.01835], [0, 0, 0, 1.0],
        ], device=device).unsqueeze(0)

    cache = _compute_fk_height
    T = T_base
    T = _batched_mat4_mul(T, _joint_transform_batched(cache._F_hr1, hip_roll1))
    T = _batched_mat4_mul(T, _joint_transform_batched(cache._F_hp1, hip_pitch1))
    T = _batched_mat4_mul(T, _joint_transform_batched(cache._F_kn1, knee1))
    T = _batched_mat4_mul(T, cache._T_nail1.expand(N, -1, -1))
    foot1_z = T[:, 2, 3]

    T = T_base
    T = _batched_mat4_mul(T, _joint_transform_batched(cache._F_hr2, hip_roll2))
    T = _batched_mat4_mul(T, _joint_transform_batched(cache._F_hp2, hip_pitch2))
    T = _batched_mat4_mul(T, _joint_transform_batched(cache._F_kn2, knee2))
    T = _batched_mat4_mul(T, cache._T_nail2.expand(N, -1, -1))
    foot2_z = T[:, 2, 3]

    min_foot_z = torch.min(foot1_z, foot2_z)
    correct_h = _FK_TARGET_FOOT_Z - min_foot_z
    return correct_h


class RlEnv(DirectRLEnv):
    cfg: RlEnvCfg

    def __init__(self, cfg: RlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ── Resolve joint indices ─────────────────────────────
        _desired_order = [
            "hip_roll1", "hip_pitch1", "knee1",
            "hip_roll2", "hip_pitch2", "knee2",
        ]
        self._joint_ids = []
        self._joint_names = []
        for jname in _desired_order:
            ids, names = self.robot.find_joints([jname])
            if len(ids) == 0:
                raise RuntimeError(
                    f"Joint '{jname}' not found. Available: {self.robot.joint_names}"
                )
            self._joint_ids.append(ids[0])
            self._joint_names.append(names[0])

        self._hip_roll_ids = [self._joint_ids[0], self._joint_ids[3]]
        self._hip_pitch_ids = [self._joint_ids[1], self._joint_ids[4]]
        self._knee_ids = [self._joint_ids[2], self._joint_ids[5]]

        # ── Resolve body indices ──────────────────────────────
        self._imu_body_ids, self._imu_body_names = self.robot.find_bodies(
            [self.cfg.imu_body_name]
        )
        if len(self._imu_body_ids) == 0:
            raise RuntimeError(
                f"IMU body '{self.cfg.imu_body_name}' not found. "
                f"Available bodies: {self.robot.body_names}"
            )
        self._imu_idx = self._imu_body_ids[0]

        self._foot1_ids, self._foot1_names = self.robot.find_bodies(
            [self.cfg.foot_body_name_1]
        )
        self._foot2_ids, self._foot2_names = self.robot.find_bodies(
            [self.cfg.foot_body_name_2]
        )
        if len(self._foot1_ids) == 0 or len(self._foot2_ids) == 0:
            raise RuntimeError(
                f"Foot bodies not found. Looking for '{self.cfg.foot_body_name_1}' "
                f"and '{self.cfg.foot_body_name_2}'. "
                f"Available bodies: {self.robot.body_names}."
            )
        self._foot1_idx = self._foot1_ids[0]
        self._foot2_idx = self._foot2_ids[0]

        self._torso_ids, _ = self.robot.find_bodies(["torso2hip"])
        if len(self._torso_ids) == 0:
            print("[WARNING] torso2hip not found for push forces, using body 0")
            self._torso_ids = [0]
        self._torso_idx = self._torso_ids[0]

        # ── Build per-joint action scale tensor ───────────────
        self._action_scale = torch.tensor(
            [
                self.cfg.action_scale_hip_roll,
                self.cfg.action_scale_hip_pitch,
                self.cfg.action_scale_knee,
                self.cfg.action_scale_hip_roll,
                self.cfg.action_scale_hip_pitch,
                self.cfg.action_scale_knee,
            ],
            device=self.device,
        )

        # ── Core buffers ─────────────────────────────────────
        self._last_actions = torch.zeros(
            self.num_envs, self.cfg.action_space, device=self.device
        )
        self._default_joint_pos = self.robot.data.default_joint_pos.clone()

        self._gravity_vec = torch.tensor(
            [0.0, 0.0, -1.0], device=self.device
        ).repeat(self.num_envs, 1)

        # ── Gait clock phase ──────────────────────────────────
        self._gait_phase = torch.zeros(self.num_envs, device=self.device)
        self._control_dt = self.cfg.sim.dt * self.cfg.decimation

        # ── Spawn XY origins ──────────────────────────────────
        self._spawn_pos_xy = torch.zeros(self.num_envs, 2, device=self.device)

        # ── Training push buffers ─────────────────────────────
        self._push_forces = torch.zeros(
            self.num_envs, self.robot.num_bodies, 3, device=self.device
        )
        self._push_torques = torch.zeros(
            self.num_envs, self.robot.num_bodies, 3, device=self.device
        )
        self._push_remaining = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # ══════════════════════════════════════════════════════
        # Curriculum state
        # ══════════════════════════════════════════════════════
        self._curriculum_alpha = self.cfg.curriculum_initial_alpha

        self._env_difficulty = torch.zeros(self.num_envs, device=self.device)

        self._initial_height = torch.full(
            (self.num_envs,), self.cfg.spawn_height, device=self.device
        )

        self._grace_remaining = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        self._episode_lengths_buffer = torch.zeros(
            self.cfg.curriculum_advance_window, device=self.device
        )
        self._episode_buffer_idx = 0
        self._episode_buffer_count = 0
        self._curriculum_cooldown = 0

        # ── Precompute kneel joint positions tensor ───────────
        self._kneel_joint_pos = torch.tensor(
            [
                self.cfg.kneel_hip_roll1,
                self.cfg.kneel_hip_pitch1,
                self.cfg.kneel_knee1,
                self.cfg.kneel_hip_roll2,
                self.cfg.kneel_hip_pitch2,
                self.cfg.kneel_knee2,
            ],
            device=self.device,
        )

        self._stand_joint_pos = self._default_joint_pos[0, self._joint_ids].clone()

        self._kneel_quat = torch.tensor(
            [
                self.cfg.kneel_quat_w,
                self.cfg.kneel_quat_x,
                self.cfg.kneel_quat_y,
                self.cfg.kneel_quat_z,
            ],
            device=self.device,
        )

        self._stand_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        )

        self._max_grace_steps = int(
            self.cfg.standup_grace_seconds / self._control_dt
        )

        # ── Per-env action center for sliding control ─────────
        self._env_init_joint_pos = torch.zeros(
            self.num_envs, len(self._joint_ids), device=self.device
        )
        self._env_init_joint_pos[:] = self._stand_joint_pos.unsqueeze(0)

        # ── Print banner ──────────────────────────────────────
        print(f"\n{'=' * 60}")
        print(f"  BIPEDAL BALANCE RL ENVIRONMENT")
        print(f"  (v5.1 — rebalanced rewards for noise control)")
        print(f"{'=' * 60}")
        print(f"  Joints: {self._joint_names}")
        print(f"  Joint IDs: {self._joint_ids}")
        print(f"  IMU body: {self._imu_body_names} (id: {self._imu_idx})")
        print(f"  Foot 1: {self._foot1_names} (id: {self._foot1_idx})")
        print(f"  Foot 2: {self._foot2_names} (id: {self._foot2_idx})")
        print(f"  Torso body idx: {self._torso_idx}")
        print(f"  Num envs: {self.num_envs}")
        print(f"  Obs dim: {self.cfg.observation_space}")
        print(f"  Act dim: {self.cfg.action_space}")
        print(f"  Control freq: {1.0 / self._control_dt:.0f} Hz")
        print(f"  Curriculum: {'ON' if self.cfg.curriculum_enable else 'OFF'}")
        print(f"  Curriculum alpha: {self._curriculum_alpha}")
        print(f"  Pushes: {'ON' if self.cfg.push_enable else 'OFF'}")
        print(f"  ── Reward rebalancing ──")
        print(f"  alive:      {self.cfg.rew_scale_alive}")
        print(f"  action_rate:     {self.cfg.rew_scale_action_rate}")
        print(f"  action_magnitude: {self.cfg.rew_scale_action_magnitude}")
        print(f"  energy:     {self.cfg.rew_scale_energy}")
        print(f"  joint_vel:  {self.cfg.rew_scale_joint_vel}")
        print(f"{'=' * 60}\n")

    # ══════════════════════════════════════════════════════════
    # Scene setup
    # ══════════════════════════════════════════════════════════

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ══════════════════════════════════════════════════════════
    # Actions
    # ══════════════════════════════════════════════════════════

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

        self._gait_phase += 2.0 * math.pi * self.cfg.gait_freq * self._control_dt
        self._gait_phase = self._gait_phase % (2.0 * math.pi)

        active_grace = self._grace_remaining > 0
        self._grace_remaining[active_grace] -= 1

        if self.cfg.push_enable:
            self._apply_training_pushes()

    def _apply_action(self) -> None:
        grace_frac = (
            self._grace_remaining.float()
            / max(self._max_grace_steps, 1)
        ).unsqueeze(-1)

        standing_center = self._default_joint_pos[:, self._joint_ids]

        action_center = (
            self._env_init_joint_pos * grace_frac
            + standing_center * (1.0 - grace_frac)
        )

        targets = action_center + (self.actions * self._action_scale)
        self.robot.set_joint_position_target(targets, joint_ids=self._joint_ids)

    # ══════════════════════════════════════════════════════════
    # Training push perturbations
    # ══════════════════════════════════════════════════════════

    def _apply_training_pushes(self):
        eligible = self._grace_remaining <= 0

        push_scale = 0.2 + 0.8 * self._curriculum_alpha

        push_mask = (
            (torch.rand(self.num_envs, device=self.device) < self.cfg.push_probability * push_scale)
            & (self._push_remaining <= 0)
            & eligible
        )
        num_new = push_mask.sum().item()

        if num_new > 0:
            angles = torch.rand(self.num_envs, device=self.device) * 2.0 * math.pi
            scaled_min = self.cfg.push_force_min * push_scale
            scaled_max = self.cfg.push_force_max * push_scale
            magnitudes = (
                torch.rand(self.num_envs, device=self.device)
                * (scaled_max - scaled_min)
                + scaled_min
            )
            fx = magnitudes * torch.cos(angles)
            fy = magnitudes * torch.sin(angles)

            self._push_forces[push_mask] = 0.0
            self._push_forces[push_mask, self._torso_idx, 0] = fx[push_mask]
            self._push_forces[push_mask, self._torso_idx, 1] = fy[push_mask]
            self._push_remaining[push_mask] = self.cfg.push_duration_steps

        active = self._push_remaining > 0
        self._push_remaining[active] -= 1

        just_finished = (~active) & (self._push_forces.abs().sum(dim=(1, 2)) > 0)
        self._push_forces[just_finished] = 0.0

        self.robot.set_external_force_and_torque(
            self._push_forces, self._push_torques
        )

    # ══════════════════════════════════════════════════════════
    # Observations  (32D)
    # ══════════════════════════════════════════════════════════

    def _get_observations(self) -> dict:
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel

        imu_quat = self.robot.data.body_quat_w[:, self._imu_idx, :]
        imu_ang_vel = self.robot.data.body_ang_vel_w[:, self._imu_idx, :]
        imu_lin_vel = self.robot.data.body_lin_vel_w[:, self._imu_idx, :]

        projected_gravity = quat_rotate_inverse(imu_quat, self._gravity_vec)
        ang_vel_imu = quat_rotate_inverse(imu_quat, imu_ang_vel)
        lin_vel_imu = quat_rotate_inverse(imu_quat, imu_lin_vel)

        joint_pos_norm = (
            joint_pos[:, self._joint_ids]
            - self._default_joint_pos[:, self._joint_ids]
        )

        gait_sin = torch.sin(self._gait_phase).unsqueeze(-1)
        gait_cos = torch.cos(self._gait_phase).unsqueeze(-1)

        difficulty = self._env_difficulty.unsqueeze(-1)
        grace_norm = (
            self._grace_remaining.float()
            / max(self._max_grace_steps, 1)
        ).unsqueeze(-1)
        init_h = self._initial_height.unsqueeze(-1)

        obs = torch.cat([
            projected_gravity,                     # (3)
            ang_vel_imu,                           # (3)
            lin_vel_imu,                           # (3)
            joint_pos_norm,                        # (6)
            joint_vel[:, self._joint_ids],          # (6)
            self._last_actions,                    # (6)
            gait_sin,                              # (1)
            gait_cos,                              # (1)
            difficulty,                            # (1)
            grace_norm,                            # (1)
            init_h,                                # (1)
        ], dim=-1)

        return {"policy": obs}

    # ══════════════════════════════════════════════════════════
    # Rewards
    # ══════════════════════════════════════════════════════════

    def _get_rewards(self) -> torch.Tensor:
        root_quat = self.robot.data.root_quat_w
        root_pos = self.robot.data.root_pos_w
        root_lin_vel = self.robot.data.root_lin_vel_w
        root_ang_vel = self.robot.data.root_ang_vel_w
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel

        projected_gravity = quat_rotate_inverse(root_quat, self._gravity_vec)
        joint_pos_norm = (
            joint_pos[:, self._joint_ids]
            - self._default_joint_pos[:, self._joint_ids]
        )

        foot1_z = self.robot.data.body_pos_w[:, self._foot1_idx, 2]
        foot2_z = self.robot.data.body_pos_w[:, self._foot2_idx, 2]

        gait_sin = torch.sin(self._gait_phase)

        in_grace = (self._grace_remaining > 0).float()

        total_reward = compute_rewards(
            # scales
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_upright,
            self.cfg.rew_scale_height,
            self.cfg.rew_scale_feet_air,
            self.cfg.rew_scale_gait_step,
            self.cfg.rew_scale_energy,
            self.cfg.rew_scale_action_rate,
            self.cfg.rew_scale_action_magnitude,    # ← NEW
            self.cfg.rew_scale_joint_vel,
            self.cfg.rew_scale_lin_vel_xy,
            self.cfg.rew_scale_lin_pos_xy,
            self.cfg.rew_scale_ang_vel_z,
            self.cfg.rew_scale_joint_pos_limits,
            self.cfg.rew_scale_standup_progress,
            self.cfg.rew_scale_hip_roll_split,
            self.cfg.rew_scale_front_split,
            self.cfg.rew_scale_knee_split,
            # state
            projected_gravity,
            root_pos[:, 2],
            root_pos[:, :2],
            self._spawn_pos_xy,
            foot1_z,
            foot2_z,
            gait_sin,
            root_lin_vel[:, :2],
            root_ang_vel[:, 2],
            joint_vel[:, self._joint_ids],
            self.robot.data.applied_torque[:, self._joint_ids],
            self.actions,
            self._last_actions,
            self.reset_terminated,
            joint_pos_norm,
            # split joint positions (raw)
            joint_pos[:, self._hip_roll_ids],
            joint_pos[:, self._hip_pitch_ids],
            joint_pos[:, self._knee_ids],
            # standup-specific
            self._initial_height,
            in_grace,
            # targets / thresholds
            self.cfg.target_height,
            self.cfg.feet_air_height_threshold,
            self.cfg.max_hip_roll_rad,
            self.cfg.max_hip_pitch_split_rad,
            self.cfg.max_knee_split_rad,
        )

        self._last_actions[:] = self.actions

        # ── Log curriculum and diagnostic extras ──────────────
        self.extras["log"] = self.extras.get("log", {})
        self.extras["log"]["curriculum/alpha"] = self._curriculum_alpha
        self.extras["log"]["curriculum/mean_difficulty"] = self._env_difficulty.mean().item()
        self.extras["log"]["curriculum/mean_grace_remaining"] = self._grace_remaining.float().mean().item()
        self.extras["log"]["curriculum/buffer_count"] = float(self._episode_buffer_count)
        if self._episode_buffer_count > 0:
            self.extras["log"]["curriculum/buffer_mean_eplen"] = (
                self._episode_lengths_buffer[:self._episode_buffer_count].mean().item()
            )
        self.extras["log"]["diagnostics/base_height"] = root_pos[:, 2].mean().item()
        self.extras["log"]["diagnostics/uprightness"] = (-projected_gravity[:, 2]).mean().item()
        self.extras["log"]["diagnostics/push_scale"] = 0.2 + 0.8 * self._curriculum_alpha
        self.extras["log"]["diagnostics/mean_action_sq"] = (self.actions ** 2).mean().item()

        return total_reward

    # ══════════════════════════════════════════════════════════
    # Termination (with grace period)
    # ══════════════════════════════════════════════════════════

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_quat = self.robot.data.root_quat_w
        root_pos = self.robot.data.root_pos_w
        joint_pos = self.robot.data.joint_pos

        projected_gravity = quat_rotate_inverse(root_quat, self._gravity_vec)
        cos_tilt = -projected_gravity[:, 2]
        too_tilted = cos_tilt < math.cos(self.cfg.max_tilt_rad)

        too_low = root_pos[:, 2] < self.cfg.min_height

        hip_roll_pos = joint_pos[:, self._hip_roll_ids]
        doing_side_splits = torch.any(
            hip_roll_pos.abs() > self.cfg.max_hip_roll_rad, dim=-1
        )

        hip_pitch_pos = joint_pos[:, self._hip_pitch_ids]
        pitch_split = torch.abs(hip_pitch_pos[:, 0] + hip_pitch_pos[:, 1])
        doing_front_splits = pitch_split > self.cfg.max_hip_pitch_split_rad

        knee_pos = joint_pos[:, self._knee_ids]
        knee_split = torch.abs(knee_pos[:, 0] + knee_pos[:, 1])
        doing_knee_splits = knee_split > self.cfg.max_knee_split_rad

        in_grace = self._grace_remaining > 0

        always_terminate = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        post_grace_terminate = (
            too_tilted | too_low | doing_side_splits
            | doing_front_splits | doing_knee_splits
        )

        terminated = always_terminate | (post_grace_terminate & ~in_grace)

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, time_out

    # ══════════════════════════════════════════════════════════
    # Reset (with curriculum interpolation + FK height)
    # ══════════════════════════════════════════════════════════

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        if self.cfg.curriculum_enable:
            self._record_episode_lengths(env_ids)

        super()._reset_idx(env_ids)

        num_reset = len(env_ids)

        # ── Sample per-env difficulty ─────────────────────────
        if self.cfg.curriculum_enable:
            if getattr(self, "_force_fixed_difficulty", False):
                self._env_difficulty[env_ids] = self._curriculum_alpha
            else:
                self._env_difficulty[env_ids] = (
                    torch.rand(num_reset, device=self.device) * self._curriculum_alpha
                )
        else:
            self._env_difficulty[env_ids] = 0.0

        difficulty = self._env_difficulty[env_ids]

        # ── Interpolate joint positions ───────────────────────
        d = difficulty.unsqueeze(-1)
        stand_jpos = self._stand_joint_pos.unsqueeze(0).expand(num_reset, -1)
        kneel_jpos = self._kneel_joint_pos.unsqueeze(0).expand(num_reset, -1)
        interp_joint_pos = stand_jpos * (1.0 - d) + kneel_jpos * d

        self._env_init_joint_pos[env_ids] = interp_joint_pos

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, self._joint_ids] = interp_joint_pos

        noise_scale = 1.0 - 0.8 * difficulty
        joint_pos += sample_uniform(
            -self.cfg.noise_joint_pos,
            self.cfg.noise_joint_pos,
            joint_pos.shape,
            joint_pos.device,
        ) * noise_scale.unsqueeze(-1)

        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        # ── Interpolate root state ────────────────────────────
        default_root = self.robot.data.default_root_state[env_ids].clone()
        default_root[:, :3] += self.scene.env_origins[env_ids]

        stand_q = self._stand_quat.unsqueeze(0).expand(num_reset, -1)
        kneel_q = self._kneel_quat.unsqueeze(0).expand(num_reset, -1)
        interp_quat = _slerp(stand_q, kneel_q, difficulty)
        default_root[:, 3:7] = interp_quat

        interp_height = _compute_fk_height(
            base_quat=interp_quat,
            hip_roll1=interp_joint_pos[:, 0],
            hip_pitch1=interp_joint_pos[:, 1],
            knee1=interp_joint_pos[:, 2],
            hip_roll2=interp_joint_pos[:, 3],
            hip_pitch2=interp_joint_pos[:, 4],
            knee2=interp_joint_pos[:, 5],
            device=self.device,
        )

        default_root[:, 2] = interp_height + self.scene.env_origins[env_ids, 2]

        vel_noise_scale = 1.0 - 0.9 * difficulty
        default_root[:, 7:10] += (
            sample_uniform(
                -self.cfg.noise_base_vel_lin,
                self.cfg.noise_base_vel_lin,
                (num_reset, 3),
                default_root.device,
            )
            * vel_noise_scale.unsqueeze(-1)
        )
        default_root[:, 10:13] += (
            sample_uniform(
                -self.cfg.noise_base_vel_ang,
                self.cfg.noise_base_vel_ang,
                (num_reset, 3),
                default_root.device,
            )
            * vel_noise_scale.unsqueeze(-1)
        )

        # ── Write to sim ──────────────────────────────────────
        self.robot.write_root_pose_to_sim(default_root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # ── Reset buffers ─────────────────────────────────────
        self._last_actions[env_ids] = 0.0
        self._spawn_pos_xy[env_ids] = default_root[:, :2]

        self._gait_phase[env_ids] = (
            torch.rand(num_reset, device=self.device) * 2.0 * math.pi
        )

        self._push_forces[env_ids] = 0.0
        self._push_remaining[env_ids] = 0

        # ── Set grace period and initial height ───────────────
        grace_steps = (difficulty * self._max_grace_steps).long()
        self._grace_remaining[env_ids] = grace_steps
        self._initial_height[env_ids] = interp_height

    # ══════════════════════════════════════════════════════════
    # Curriculum advancement
    # ══════════════════════════════════════════════════════════

    def _record_episode_lengths(self, env_ids):
        ep_lens = self.episode_length_buf[env_ids]

        valid_mask = ep_lens > 0
        valid_lens = ep_lens[valid_mask]

        for length in valid_lens:
            idx = self._episode_buffer_idx % self.cfg.curriculum_advance_window
            self._episode_lengths_buffer[idx] = length.item()
            self._episode_buffer_idx += 1
            self._episode_buffer_count = min(
                self._episode_buffer_count + 1,
                self.cfg.curriculum_advance_window,
            )

        if self._curriculum_cooldown > 0:
            self._curriculum_cooldown -= 1
            return

        if self._episode_buffer_count < self.cfg.curriculum_advance_window:
            return

        mean_len = self._episode_lengths_buffer[:self._episode_buffer_count].mean().item()
        normalized = mean_len / self.max_episode_length

        if normalized >= self.cfg.curriculum_advance_threshold:
            old_alpha = self._curriculum_alpha
            self._curriculum_alpha = min(
                self._curriculum_alpha + self.cfg.curriculum_alpha_step,
                self.cfg.curriculum_max_alpha,
            )
            if self._curriculum_alpha != old_alpha:
                print(
                    f"[CURRICULUM] alpha: {old_alpha:.3f} → {self._curriculum_alpha:.3f}"
                    f"  (mean ep: {mean_len:.0f}/{self.max_episode_length}"
                    f" = {normalized:.2%})"
                )
                self._curriculum_cooldown = self.cfg.curriculum_advance_cooldown
                self._episode_buffer_count = 0
                self._episode_buffer_idx = 0


# ══════════════════════════════════════════════════════════════
# Quaternion SLERP
# ══════════════════════════════════════════════════════════════

def _slerp(q0, q1, t):
    t = t.unsqueeze(-1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs()
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    near_zero = sin_theta.abs() < 1e-6
    s0 = torch.where(near_zero, 1.0 - t, torch.sin((1.0 - t) * theta) / sin_theta)
    s1 = torch.where(near_zero, t, torch.sin(t * theta) / sin_theta)
    result = s0 * q0 + s1 * q1
    result = result / result.norm(dim=-1, keepdim=True)
    return result


# ══════════════════════════════════════════════════════════════
# Reward function (JIT-compiled)
# ══════════════════════════════════════════════════════════════

@torch.jit.script
def compute_rewards(
    # scales
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_upright: float,
    rew_scale_height: float,
    rew_scale_feet_air: float,
    rew_scale_gait_step: float,
    rew_scale_energy: float,
    rew_scale_action_rate: float,
    rew_scale_action_magnitude: float,       # ← NEW
    rew_scale_joint_vel: float,
    rew_scale_lin_vel_xy: float,
    rew_scale_lin_pos_xy: float,
    rew_scale_ang_vel_z: float,
    rew_scale_joint_pos_limits: float,
    rew_scale_standup_progress: float,
    rew_scale_hip_roll_split: float,
    rew_scale_front_split: float,
    rew_scale_knee_split: float,
    # state tensors
    projected_gravity: torch.Tensor,
    base_height: torch.Tensor,
    root_pos_xy: torch.Tensor,
    spawn_pos_xy: torch.Tensor,
    foot1_z: torch.Tensor,
    foot2_z: torch.Tensor,
    gait_sin: torch.Tensor,
    lin_vel_xy: torch.Tensor,
    ang_vel_z: torch.Tensor,
    joint_vel: torch.Tensor,
    applied_torque: torch.Tensor,
    actions: torch.Tensor,
    last_actions: torch.Tensor,
    reset_terminated: torch.Tensor,
    joint_pos_norm: torch.Tensor,
    # split joint positions (raw)
    hip_roll_pos: torch.Tensor,
    hip_pitch_pos: torch.Tensor,
    knee_pos: torch.Tensor,
    # standup-specific
    initial_height: torch.Tensor,
    in_grace: torch.Tensor,
    # targets / thresholds
    target_height: float,
    feet_air_height_threshold: float,
    max_hip_roll_rad: float,
    max_hip_pitch_split_rad: float,
    max_knee_split_rad: float,
) -> torch.Tensor:
    """Compute shaped reward for bipedal balance with standup curriculum."""

    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())

    rew_terminated = rew_scale_terminated * reset_terminated.float()

    uprightness = -projected_gravity[:, 2]
    rew_upright = rew_scale_upright * torch.exp(-3.0 * (1.0 - uprightness))

    post_grace = 1.0 - in_grace

    height_error = base_height - target_height
    rew_height = rew_scale_height * torch.square(height_error) * post_grace

    height_range = target_height - initial_height
    safe_range = torch.clamp(height_range, min=0.01)
    progress = (base_height - initial_height) / safe_range
    rew_standup = rew_scale_standup_progress * torch.clamp(progress, min=0.0, max=1.0)

    foot_height_diff = torch.abs(foot1_z - foot2_z)
    rew_feet_air = rew_scale_feet_air * torch.clamp(
        foot_height_diff / feet_air_height_threshold, max=1.0
    )

    signed_diff = foot1_z - foot2_z
    gait_match = gait_sin * signed_diff / feet_air_height_threshold
    rew_gait_step = rew_scale_gait_step * torch.clamp(gait_match, min=0.0, max=1.0) * post_grace

    rew_energy = rew_scale_energy * torch.sum(torch.square(applied_torque), dim=-1)

    action_diff = actions - last_actions
    rew_action_rate = rew_scale_action_rate * torch.sum(torch.square(action_diff), dim=-1)

    # ══════════════════════════════════════════════════════════
    # NEW: Action magnitude penalty
    # Penalizes ||actions||² — directly pressures noise std down.
    # ALWAYS active (no post_grace gating) so it applies during
    # grace period where most other penalties are suppressed.
    # ══════════════════════════════════════════════════════════
    rew_action_magnitude = rew_scale_action_magnitude * torch.sum(torch.square(actions), dim=-1)

    rew_joint_vel = rew_scale_joint_vel * torch.sum(torch.square(joint_vel), dim=-1) * post_grace

    rew_lin_vel_xy = rew_scale_lin_vel_xy * torch.sum(torch.square(lin_vel_xy), dim=-1) * post_grace

    xy_displacement = root_pos_xy - spawn_pos_xy
    rew_lin_pos_xy = rew_scale_lin_pos_xy * torch.sum(torch.square(xy_displacement), dim=-1) * post_grace

    rew_ang_vel_z = rew_scale_ang_vel_z * torch.square(ang_vel_z) * post_grace

    rew_joint_pos_limits = rew_scale_joint_pos_limits * torch.sum(torch.square(joint_pos_norm), dim=-1) * post_grace

    hip_roll_ratio = hip_roll_pos / max_hip_roll_rad
    rew_hip_roll_split = rew_scale_hip_roll_split * torch.sum(torch.square(hip_roll_ratio), dim=-1) * post_grace

    front_split_val = torch.abs(hip_pitch_pos[:, 0] + hip_pitch_pos[:, 1]) / max_hip_pitch_split_rad
    rew_front_split = rew_scale_front_split * torch.square(front_split_val) * post_grace

    knee_split_val = torch.abs(knee_pos[:, 0] + knee_pos[:, 1]) / max_knee_split_rad
    rew_knee_split = rew_scale_knee_split * torch.square(knee_split_val) * post_grace

    total = (
        rew_alive
        + rew_terminated
        + rew_upright
        + rew_height
        + rew_standup
        + rew_feet_air
        + rew_gait_step
        + rew_energy
        + rew_action_rate
        + rew_action_magnitude       # ← NEW
        + rew_joint_vel
        + rew_lin_vel_xy
        + rew_lin_pos_xy
        + rew_ang_vel_z
        + rew_joint_pos_limits
        + rew_hip_roll_split
        + rew_front_split
        + rew_knee_split
    )

    return total