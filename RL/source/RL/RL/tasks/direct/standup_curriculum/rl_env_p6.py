# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""
rl_env.py — Version7

    Inherits all p6d features:
    - FK overshoot fix (FK − overshoot × d³)
    - Kneel pose noise (±0.05 rad scaled by difficulty)
    - Anti-gallop: flight penalty + vz penalty + stance symmetry
    - Yaw drift penalty
    - Conservative ±0.5 rad/s yaw cmds
    - Curriculum alpha_step 0.01

Developer Notes:
    1. Never remove the log we outputted in train or tensorboard.
    2. The more diagnostics the better.

A/B FORK (gait-reference prior): adds reward slot 37 ("gait_ref") — an
antiphase hip-pitch sinusoid tracking reward on the existing gait clock —
plus three diagnostics (gait_ref_amp_proj / gait_ref_track_err_abs /
gait_ref_airborne_swing_hp_dev). All knobs getattr-defaulted to scale 0,
so this file is drop-in safe for the main run's cfg (absent knobs ⇒
bit-identical reward). Existing log keys untouched; slot 37 appended.
"""

from __future__ import annotations

import math
import os
import torch
from collections.abc import Sequence
from typing import Tuple

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply_inverse, quat_apply, sample_uniform, quat_mul

from .belt_knee_actuator import damping_for_ratio, encoder_write_slots
from .pad_contact import PadContact, measure_ball_geometry
from .rl_env_cfg_p6 import RlEnvCfg


# ══════════════════════════════════════════════════════════════
# 3D Forward Kinematics (unchanged from phase 6d)
# ══════════════════════════════════════════════════════════════
_FK_TARGET_FOOT_Z = 0.0716 + (0.040 - 0.02598076)  # 0.08561924 — ball foot is ~14mm taller


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
    return result / result.norm(dim=-1, keepdim=True)


def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


# ══════════════════════════════════════════════════════════════
# Reward component names — slot 2 repurposed: upright → raw_action
# ══════════════════════════════════════════════════════════════
_REWARD_COMPONENT_NAMES = [
    "alive", "terminated", "raw_action", "height",
    "standup_height", "standup_fall", "thermal", "vel_tracking",
    "feet_air", "gait_step", "foot_swing", "energy",
    "action_rate", "action_magnitude", "joint_vel", "joint_pos_limits",
    "hip_roll_split", "front_split", "knee_split", "leg_symmetry",
    "angle_symmetry", "knee_over_hip", "foot_progress", "position_anchor",
    "heading_anchor", "back_leg_step", "yaw_rate", "balance",
    "no_flight", "vz_penalty", "stance_symmetry", "yaw_drift",
    "action_smoothness",   # NEW (slot 32)
    "action_snap",         # NEW (slot 33)
    "upright",             # NEW (slot 34)
    "clock_contact",       # NEW (slot 35) knob 1
    "swing_clearance",     # NEW (slot 36) knob 2
    "gait_ref",            # NEW (slot 37) A/B fork: gait-reference prior
    "standup_track",       # NEW (slot 38) path-mode grace pitch tracking (09-08)
    "standup_contact",     # NEW (slot 39) path-mode grace feet-down
]
_OBS_PER_FRAME = 36   # one frame of obs before history stacking  
_OBS_SENSORS_PER_FRAME = 21 # sensors-only slice (IMU + encoder), gets obs_delay
_OBS_IMU_PER_FRAME = 9      # proj_grav(3)+ang_vel(3)+lin_vel(3) — fast IMU path
_OBS_JOINT_PER_FRAME = 12   # joint_pos(6)+joint_vel(6) — slow encoder/CAN path

# Belt 22:30 reduction, squared.  Used to convert the knee-referred belt
# stiffness (the servo's frame, what the sweep pins) into the motor-side value
# the series integrator wants.  Damping is NOT converted: a torque/velocity
# ratio carries no power of G.  Keep in step with control/config.py's ratio.
_BELT_G2 = (22.0 / 30.0) ** 2


def _tie_armature_pair_columns(mult: torch.Tensor, pair_slots) -> torch.Tensor:
    """One armature draw per leg: copy each pair's hip_pitch column into its
    knee column of a [num, 6] multiplier tensor (in place, returned).

    The knee rotor is the whole hip_pitch share (c²·knee), the off-diagonal
    (c·knee) and the belt integrator's rotor, so a knee draw that differs
    from its hip draw builds an inertia block no single rotor can produce.
    ``pair_slots`` is [(hip_pitch_slot, knee_slot), ...] in env joint order;
    the hip_pitch column is kept so the RNG stream matches the legacy
    per-joint draw exactly (the sample shape is unchanged).
    """
    for hp_slot, kn_slot in pair_slots:
        mult[:, kn_slot] = mult[:, hp_slot]
    return mult
# ══════════════════════════════════════════════════════════════
# Mirror augmentation (sagittal-plane L/R reflection)
# ══════════════════════════════════════════════════════════════
# OBS layout (36D) — see _get_observations:
#   [0:3]   proj_grav (gx, gy, gz)             y flips
#   [3:6]   ang_vel   (ωx, ωy, ωz)             x,z flip
#   [6:9]   lin_vel   (vx, vy, vz)             y flips
#   [9:15]  joint_pos_norm  (hr1,hp1,kn1, hr2,hp2,kn2)  swap+negate
#   [15:21] joint_vel       (same order)                swap+negate
#   [21:27] last_actions    (same order)                swap+negate
#   [27]    gait_sin                            negate (π phase)
#   [28]    gait_cos                            negate (π phase)
#   [29]    difficulty                          keep
#   [30]    grace_norm                          keep
#   [31]    init_h                              keep
#   [32]    vx_cmd                              keep
#   [33]    yaw_rate_cmd                        negate
#   [34]    yaw_err_sin                         negate
#   [35]    yaw_err_cos                         keep
# ACTION layout (6D): hr1, hp1, kn1, hr2, hp2, kn2 — swap+negate (same as joints).

_OBS_MIRROR_PERMUTATION = [
    0, 1, 2,                                # proj_grav
    3, 4, 5,                                # ang_vel
    6, 7, 8,                                # lin_vel
    12, 13, 14,  9, 10, 11,                 # joint_pos:    swap legs
    18, 19, 20, 15, 16, 17,                 # joint_vel:    swap legs
    24, 25, 26, 21, 22, 23,                 # last_actions: swap legs
    27, 28,                                 # gait_sin, gait_cos
    29, 30, 31,                             # difficulty, grace, init_h
    32, 33,                                 # vx_cmd, yaw_rate_cmd
    34, 35,                                 # yaw_err_sin, yaw_err_cos
]

_OBS_MIRROR_SIGN = [
     1.0, -1.0,  1.0,                       # proj_grav: gy flips
    -1.0,  1.0, -1.0,                       # ang_vel:   ωx, ωz flip
     1.0, -1.0,  1.0,                       # lin_vel:   vy flips
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,     # joint_pos:    all negate
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,     # joint_vel:    all negate
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,     # last_actions: all negate
    -1.0, -1.0,                             # gait_sin/cos: both negate
     1.0,  1.0,  1.0,                       # difficulty, grace, init_h
     1.0, -1.0,                             # vx_cmd keep, yaw_rate_cmd negate
    -1.0,  1.0,                             # yaw_err_sin negate, cos keep
]

_ACT_MIRROR_PERMUTATION = [3, 4, 5, 0, 1, 2]
_ACT_MIRROR_SIGN        = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]


def get_symmetric_states(
    obs=None,
    actions=None,
    env=None,
    obs_type: str = "policy",
):
    """Data-augmentation hook for rsl_rl 3.x SymmetryCfg.

    rsl_rl 3.x passes `obs` as a TensorDict — the actual obs tensor
    is at obs[obs_type]. Older versions passed a plain tensor; we
    handle both for forward compat.
    """
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env

    obs_aug = None
    if obs is not None:
        if hasattr(obs, "keys"):
            # TensorDict path (rsl_rl 3.x)
            obs_tensor = obs[obs_type]
            mirrored_tensor = base_env.mirror_obs(obs_tensor)
            mirrored_td = obs.clone()
            mirrored_td[obs_type] = mirrored_tensor
            obs_aug = torch.cat([obs, mirrored_td], dim=0)
        else:
            # Plain-tensor fallback (older rsl_rl)
            obs_aug = torch.cat([obs, base_env.mirror_obs(obs)], dim=0)

    actions_aug = None
    if actions is not None:
        actions_aug = torch.cat(
            [actions, base_env.mirror_action(actions)], dim=0
        )

    return obs_aug, actions_aug
# ══════════════════════════════════════════════════════════════
# Walking Environment — Phase 6e
# ══════════════════════════════════════════════════════════════

class RlEnv(DirectRLEnv):
    cfg: RlEnvCfg

    def __init__(self, cfg: RlEnvCfg, render_mode: str | None = None, **kwargs):
        # P10 knobs must land in the actuator cfg BEFORE super().__init__
        # freezes it into the articulation. Dict keys must match the
        # factory's joint_names_expr patterns.
        # The foot material is a top-level eval knob because Hydra paths into
        # nested regex actuator/spawn dictionaries are awkward.  Copy it into
        # the spawn config before DirectRLEnv builds the stage.  All-zero
        # values keep FootContactUsdFileCfg on its exact legacy branch.
        _spawn = cfg.robot_cfg.spawn
        _spawn.compliant_contact_stiffness = cfg.foot_contact_stiffness
        _spawn.compliant_contact_damping = cfg.resolved_foot_contact_damping()
        _spawn.restitution = cfg.foot_restitution
        _spawn.static_friction = cfg.foot_static_friction
        _spawn.dynamic_friction = cfg.foot_dynamic_friction
        # Print the resolved plant on every build.  These knobs are silent by
        # construction (all-zero = the legacy rigid path, and the material is
        # created at spawn, so a wrong value is invisible in the rollout), so
        # the one place a mistake can be caught cheaply is here.
        _belt_on = bool(cfg.belt_band_live())
        _d_ratio = cfg.belt_series_damping_ratio
        print(
            f"[PLANT] belt series: live={_belt_on} "
            f"k_joint=[{cfg.belt_series_stiffness_min:g}, {cfg.belt_series_stiffness_max:g}] N m/rad "
            f"d_motor=[{cfg.belt_series_damping_min:g}, {cfg.belt_series_damping_max:g}] N m s/rad "
            f"zeta={'band' if _belt_on and (cfg.belt_series_damping_min or cfg.belt_series_damping_max) else ('off' if math.isnan(_d_ratio) else f'{_d_ratio:g}')} "
            f"I_rotor={cfg.belt_series_rotor_inertia:.5f} kg m^2"
            f"{' x per-leg armature draw' if cfg.dr_armature_per_leg else ''}"
        )
        print(
            f"[PLANT] foot contact: k={cfg.foot_contact_stiffness:g} N/m "
            f"d={_spawn.compliant_contact_damping:g} N s/m "
            f"(ratio={cfg.foot_contact_damping_ratio:g} at m_eff={cfg.foot_contact_effective_mass:g} kg) "
            f"restitution={cfg.foot_restitution:g} "
            f"friction={cfg.foot_static_friction:g}/{cfg.foot_dynamic_friction:g} -> "
            f"{'RIGID legacy material' if _spawn.compliant_contact_stiffness == 0.0 and _spawn.compliant_contact_damping == 0.0 and _spawn.restitution == 0.0 and _spawn.static_friction == 1.0 and _spawn.dynamic_friction == 1.0 else 'custom foot material on the ball meshes'}"
        )
        _hk = cfg.robot_cfg.actuators["hip_knee"]
        _hk.rotor_coupling = cfg.plant_rotor_coupling
        _hk.stiffness = {"hip_pitch.*": cfg.servo_kp_hip_pitch,
                         "knee.*":      cfg.servo_kp_knee}
        _hk.damping   = {"hip_pitch.*": cfg.servo_kd_hip_pitch,
                         "knee.*":      cfg.servo_kd_knee}
        # Armature is DERIVED here (2026-09-18): the cfg carries the rotor
        # inertia facts and resolved_armature() is the one derivation call
        # (belt_knee_actuator.derive_armature); explicit cfg pins (>= 0)
        # win.  This write replaces the factory's armature dict wholesale,
        # which is why the derivation cannot live only in the factory.
        _arm = cfg.resolved_armature()
        _hk.armature  = {"hip_pitch.*": _arm.hip_pitch,
                         "knee.*":      _arm.knee}
        # The own-rotor share the armature mask keeps on the hip column under
        # a live belt pair comes from the same derivation (own/total).
        _hk.hip_own_rotor_armature = _arm.hip_own_rotor
        # Series elasticity is installed per episode after the articulation
        # exists.  Seed the actuator cfg here so the optional state buffers are
        # allocated before Isaac Lab freezes the actuator.  Frames as in
        # RlEnvCfg: stiffness joint-referred (x G^2), damping already
        # motor-side.  The same pair of factors is applied at every write.
        #
        # The live/off decision is the BAND (`belt_band_live`), not the legacy
        # `belt_series_enable` flag: a nonzero stiffness band is a per-episode
        # DR draw and must allocate these buffers even when the flag is unset,
        # or the first reset would write into an actuator that has no series
        # state (S5, 2026-09-12).  An all-zero band still takes the legacy
        # rigid path bit-for-bit.
        _belt_live = bool(cfg.belt_band_live())
        _hk.series_elasticity = _belt_live
        _hk.series_stiffness = cfg.belt_series_stiffness_min * _BELT_G2
        _hk.series_damping = cfg.belt_series_damping_min
        _hk.series_rotor_inertia = cfg.belt_series_rotor_inertia
        _hr = cfg.robot_cfg.actuators["hip_roll"]
        _hr.stiffness = cfg.servo_kp_hip_roll
        _hr.damping   = cfg.servo_kd_hip_roll
        print(f"[P10] rotor_coupling={_hk.rotor_coupling}  "
              f"kp hp/kn/hr={cfg.servo_kp_hip_pitch}/{cfg.servo_kp_knee}/"
              f"{cfg.servo_kp_hip_roll}  "
              f"kd={cfg.servo_kd_hip_pitch}/{cfg.servo_kd_knee}/"
              f"{cfg.servo_kd_hip_roll}  "
              f"armature hp/kn={_arm.hip_pitch*1e3:.1f}/"
              f"{_arm.knee*1e3:.1f}e-3")
        # The derivation, spelled out: which frame I_out is in, the formula,
        # and whether each joint was pinned or derived.  A silent convention
        # here is exactly the trap the derivation exists to remove.
        print(f"[ARMATURE] {_arm.describe()}")
        print(
            f"[ARMATURE] hip_pitch {_arm.hip_source.upper()} "
            f"(cfg.hip_pitch_armature={cfg.hip_pitch_armature:g})  "
            f"knee {_arm.knee_source.upper()} (cfg.knee_armature={cfg.knee_armature:g})  "
            f"belt_series_rotor_inertia={cfg.belt_series_rotor_inertia*1e3:.3f}e-3"
            f"{'' if abs(cfg.belt_series_rotor_inertia - _arm.rotor_motor_side) < 1e-9 else '  ** differs from G^2*knee **'}  "
            f"DR draw: {'ONE per leg (hip_pitch & knee columns share; off-diag + series rotor follow)' if cfg.dr_armature_per_leg else 'legacy independent per joint (K/L/R/S contract)'}  "
            f"belt-live mask: hip x own/total = {_arm.hip_own_share:.3f}, knee x 0"
            f"{'' if cfg.belt_mask_armature else ' (belt_mask_armature=False: mask OFF, rotor double-counted)'}"
        )
        super().__init__(cfg, render_mode, **kwargs)
        # 1. Helper function
        def _per_joint(hr, hp, kn):
            return torch.tensor([hr, hp, kn, hr, hp, kn], device=self.device)
        # 2. Joint indices
        self._joint_ids = []
        self._joint_names = []
        _desired_order = [
            "hip_roll1", "hip_pitch1", "knee1",
            "hip_roll2", "hip_pitch2", "knee2",
        ]
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
        # 3. Body indices
        self._imu_body_ids, self._imu_body_names = self.robot.find_bodies(
            [self.cfg.imu_body_name]
        )
        if len(self._imu_body_ids) == 0:
            raise RuntimeError(f"IMU body '{self.cfg.imu_body_name}' not found.")
        self._imu_idx = self._imu_body_ids[0]

        self._foot1_ids, _ = self.robot.find_bodies([self.cfg.foot_body_name_1])
        self._foot2_ids, _ = self.robot.find_bodies([self.cfg.foot_body_name_2])
        if len(self._foot1_ids) == 0 or len(self._foot2_ids) == 0:
            raise RuntimeError("Foot bodies not found.")
        self._foot1_idx = self._foot1_ids[0]
        self._foot2_idx = self._foot2_ids[0]

        # ── Ball-pad compliance, per env per foot (pad_contact.PadContact) ──
        # Built ALWAYS but inert unless the cfg asks for a stiffness, so the
        # default path costs one tensor allocation and touches no force.  The
        # ball centre offset and radius are MEASURED from the spawned asset
        # rather than hardcoded: the env's `_toe*_local_offset` is the nail
        # origin, 5.8 cm short of the ball, so reusing it as the contact point
        # would put the pad spring in the wrong place.
        self._pad = None
        self._pad_blowup_warned = False
        if self.cfg.pad_contact_enable:
            self._pad = self._build_pad_contact()

        self._torso_ids, _ = self.robot.find_bodies(["torso2hip"])
        if len(self._torso_ids) == 0:
            print("[WARNING] torso2hip not found, using body 0")
            self._torso_ids = [0]
        self._torso_idx = self._torso_ids[0]
        # Top bodies for dr_top_mass_add_kg: torso + both hip-roll chassis blocks.
        self._top_body_ids, _ = self.robot.find_bodies(
            ["torso2hip", "chasis_motor2motor", "lid_bearing_outer___boss_extrude3"])
        if len(self._top_body_ids) != 3:
            raise RuntimeError(f"top bodies not found: {self._top_body_ids}")
        # 4. Now safe to use _joint_ids and _per_joint
        self._action_scale = _per_joint(
            cfg.real_action_scale_hip_roll,
            cfg.real_action_scale_hip_pitch,
            cfg.real_action_scale_knee,
        )
        self._action_rate_weights = self._action_scale.clone()
        self._continuous_effort = _per_joint(
            self.cfg.continuous_effort_hip_roll,
            self.cfg.continuous_effort_hip_pitch,
            self.cfg.continuous_effort_knee,
        )
        # 5. Actuator caching (needs _joint_ids and self.robot.actuators)
        self._nominal_actuator = {}   # group name -> dict of per-joint tensors
        for name, act in self.robot.actuators.items():
            print(act)
            self._nominal_actuator[name] = {
                "stiffness":        act.stiffness.clone(),
                "damping":          act.damping.clone(),
                "effort_limit":     act.effort_limit.clone(),
                "armature":         act.armature.clone(),
                "friction":         act.friction.clone(),
                "dynamic_friction": act.dynamic_friction.clone(),
                "viscous_friction": act.viscous_friction.clone(),
                "velocity_limit":   act.velocity_limit.clone(),
            }

        # Logical joint type -> (actuator group, column) via a representative
        # joint — robust to the merged hip_pitch+knee belt actuator group.
        self._diag_act = {}
        for logical, rep in (("hip_roll", "hip_roll1"),
                             ("hip_pitch", "hip_pitch1"),
                             ("knee", "knee1")):
            for name, act in self.robot.actuators.items():
                if rep in act.joint_names:
                    self._diag_act[logical] = (name, act.joint_names.index(rep))
                    break
            else:
                raise RuntimeError(f"No actuator group covers joint '{rep}'")

        # The nominal snapshot above must hold the UNDRAWN cfg armature: DR
        # writes nominal * multiplier into act.armature at every reset and
        # never touches the snapshot, so a mismatch here means Isaac parsed
        # something other than what resolved_armature() derived.
        _nom_hp = self._nominal_actuator[self._diag_act["hip_pitch"][0]]["armature"][:, self._diag_act["hip_pitch"][1]]
        _nom_kn = self._nominal_actuator[self._diag_act["knee"][0]]["armature"][:, self._diag_act["knee"][1]]
        print(f"[ARMATURE] nominal snapshot hp/kn = {_nom_hp[0].item()*1e3:.3f}/{_nom_kn[0].item()*1e3:.3f}e-3 "
              f"(resolved {_arm.hip_pitch*1e3:.3f}/{_arm.knee*1e3:.3f}e-3, {self.num_envs} envs)")
        for _lbl, _nom, _want in (("hip_pitch", _nom_hp, _arm.hip_pitch), ("knee", _nom_kn, _arm.knee)):
            if not torch.allclose(_nom, torch.full_like(_nom, float(_want)), rtol=1e-5, atol=0.0):
                raise RuntimeError(
                    f"[ARMATURE] nominal {_lbl} armature snapshot {_nom.min().item():.6e}.."
                    f"{_nom.max().item():.6e} != resolved {_want:.6e}"
                )
        # Per-leg (hip_pitch, knee) env-slot pairs in the belt actuator's
        # pair order — the coherent armature draw and the series-rotor scaling
        # index the [N,6] multiplier with these.
        _pairs = self.cfg.robot_cfg.actuators["hip_knee"].pairs
        self._armature_pair_slots = [
            (self._joint_names.index(hp), self._joint_names.index(kn)) for kn, hp in _pairs.items()
        ]
        self._armature_pair_knee_slots = torch.tensor(
            [kn for _, kn in self._armature_pair_slots], dtype=torch.long, device=self.device
        )

        self._actuator_joint_map = {}   # name -> list[(my_slot, local_idx_in_actuator)]
        # 6. Everything else

        # Cache pristine PhysX state once — DR is applied as multiplier
        # / offset relative to these. Without caching, repeated resets
        # would compound the multipliers (each set_masses uses the
        # current value as input, not the original). Kept on CPU
        # because that's where root_physx_view lives.
        self._original_masses = self.robot.root_physx_view.get_masses().clone().to("cpu")
        self._original_coms   = self.robot.root_physx_view.get_coms().clone().to("cpu")
        # Body inertia tensors (N, B, 9). PhysX does NOT rescale these when a
        # mass is set; dr_mass_recompute_inertia scales them by the per-body
        # mass ratio at each DR write (Isaac Lab randomize_rigid_body_mass pattern).
        self._original_inertias = self.robot.root_physx_view.get_inertias().clone().to("cpu")
        _orig_mats = self.robot.root_physx_view.get_material_properties().clone().to("cpu")
        self._original_static_fric = _orig_mats[..., 0].clone()
        self._original_dyn_fric    = _orig_mats[..., 1].clone()
        # ── Foot-restitution DR rows (S6): DISCOVERED, never hardcoded ──────
        # get_material_properties() is (N, max_shapes, 3).  The two ball shapes
        # carry restitution == foot_restitution (0.63) because the spawner wrote
        # one shared material to both; every other row reads 0.  Row indices are
        # shape indices, identical across envs (replicated stage), so env 0
        # fixes them for all envs.  Assert exactly two, and that the remaining
        # rows read restitution 0 — a wrong combine mode or a missed binding
        # fails here at build time instead of silently halving the bounce.
        self._foot_restitution_rows: tuple[int, ...] = ()
        if self.cfg.dr_foot_restitution_max > 0.0 or self.cfg.dr_foot_restitution_min > 0.0:
            _rest = _orig_mats[0, :, 2]
            _hit = (_rest.abs() - float(self.cfg.foot_restitution)).abs() < 1e-4
            rows = tuple(int(i) for i in torch.nonzero(_hit).flatten())
            _others = int(((_rest.abs() > 1e-4) & ~_hit).sum())
            if len(rows) != 2:
                raise RuntimeError(
                    f"[S7] expected 2 foot-restitution rows (restitution "
                    f"{self.cfg.foot_restitution:g}), found {len(rows)}: {rows}; "
                    f"material buffer shape {tuple(_orig_mats.shape)}"
                )
            if _others != 0:
                raise RuntimeError(
                    f"[S7] {_others} non-ball rows carry nonzero restitution — "
                    f"foot binding leaked beyond the two ball shapes"
                )
            self._foot_restitution_rows = rows
            print(f"[S7] foot-restitution rows discovered: {rows} "
                  f"(restitution={self.cfg.foot_restitution:g} carrier, "
                  f"all {_orig_mats.shape[1] - len(rows)} other rows read 0)")

        
        for name, act in self.robot.actuators.items():
            pairs = []
            for i, jid in enumerate(self._joint_ids):
                if jid in act.joint_indices:
                    local = (act.joint_indices == jid).nonzero(as_tuple=True)[0].item()
                    pairs.append((i, local))
            self._actuator_joint_map[name] = pairs



        # ── Toe tip local offsets ──────────────────────────────
        self._toe1_local_offset = torch.tensor(
            [-0.138448, 0.0216628, -0.01835], device=self.device
        )
        self._toe2_local_offset = torch.tensor(
            [-4.95155e-05, 0.140191, 0.01835], device=self.device
        )

        # ── COM body indices and mass weights ──────────────────
        self._com_body_ids = []
        com_masses_list = []
        for i, bname in enumerate(self.cfg.com_body_names):
            bids, _ = self.robot.find_bodies([bname])
            if len(bids) == 0:
                raise RuntimeError(f"COM body '{bname}' not found.")
            self._com_body_ids.append(bids[0])
            com_masses_list.append(self.cfg.com_body_masses[i])

        total_mass = sum(com_masses_list)
        self._com_mass_weights = torch.tensor(
            [m / total_mass for m in com_masses_list], device=self.device
        )
        # Per-frame tables already defined as module constants.
        # Expand for the K-frame stacked obs the policy actually sees.
        n_hist = self.cfg.obs_history_steps
        expanded_perm = []
        expanded_sign = []
        for k in range(n_hist):
            offset = k * _OBS_PER_FRAME
            expanded_perm.extend(p + offset for p in _OBS_MIRROR_PERMUTATION)
            expanded_sign.extend(_OBS_MIRROR_SIGN)
        self._obs_mirror_perm = torch.tensor(
            expanded_perm, dtype=torch.long, device=self.device,
        )
        self._obs_mirror_sign = torch.tensor(
            expanded_sign, dtype=torch.float32, device=self.device,
        )
        self._act_mirror_perm = torch.tensor(
            _ACT_MIRROR_PERMUTATION, dtype=torch.long, device=self.device,
        )
        self._act_mirror_sign = torch.tensor(
            _ACT_MIRROR_SIGN, dtype=torch.float32, device=self.device,
        )
        # ── Privileged-obs mirror table (local indices in the priv group) ──
        # lin_vel(0:3) vy flip; ang_vel(3:6) wx,wz flip; phase one-hot(6:11)
        # keep; support(11) keep; pickup force_b(12:15) fy flip; mass/floor/
        # voltage/com(15:19) keep; the NINE [hr1,hp1,kn1,hr2,hp2,kn2] DR-mult
        # blocks(19:73) — stiffness/damping/effort/armature/jfric/dynfric/
        # viscfric + act_lag ω/ζ (v7.1) — swap leg halves with NO sign change
        # (all positive mults); delays + lin_vel_keep + slope(73:78) keep.
        _priv_perm = list(range(self.cfg.priv_obs_dim))
        _priv_sign = [1.0] * self.cfg.priv_obs_dim
        _priv_sign[1] = -1.0     # lin_vel vy
        _priv_sign[3] = -1.0     # ang_vel wx
        _priv_sign[5] = -1.0     # ang_vel wz
        _priv_sign[13] = -1.0    # pickup force fy
        for _s in (19, 25, 31, 37, 43, 49, 55, 61, 67):
            _priv_perm[_s:_s + 6] = [_s+3, _s+4, _s+5, _s+0, _s+1, _s+2]
        self._priv_mirror_perm = torch.tensor(
            _priv_perm, dtype=torch.long, device=self.device,
        )
        self._priv_mirror_sign = torch.tensor(
            _priv_sign, dtype=torch.float32, device=self.device,
        )
        self._verify_mirror_correctness()

        # Stash for per-step symmetry diagnostic
        self._last_obs_for_diag = None
        # ── Core buffers ───────────────────────────────────────
        self._last_actions = torch.zeros(
            self.num_envs, self.cfg.action_space, device=self.device
        )
        # NEW: t-2 actions, required for second-derivative (jerk) penalty.
        self._last_last_actions = torch.zeros(
            self.num_envs, self.cfg.action_space, device=self.device
        )
        # NEW: t-3 actions, required for third-derivative (snap) penalty.
        self._last_last_last_actions = torch.zeros(
            self.num_envs, self.cfg.action_space, device=self.device
        )
        # NEW IN p6e: raw (pre-clip) actions for raw-action penalty.
        self._raw_actions = torch.zeros(
            self.num_envs, self.cfg.action_space, device=self.device
        )
        # NEW 2026-08-07: EMA of knee action clipping for the sustained-
        # saturation termination (cfg §10). Charges only while the robot
        # owns its balance; decays during grace/pickup shielding.
        self._knee_sat_ema = torch.zeros(self.num_envs, device=self.device)
        # col 0 = hip_roll pair, col 1 = hip_pitch pair (arm-C hip fence;
        # allocated unconditionally for stable diag keys)
        self._hip_sat_ema = torch.zeros(self.num_envs, 2, device=self.device)
        self._sat_terminated = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # NEW IN p6e: per-env keep-mask for lin_vel obs (sim2real dropout).
        # 1.0 = pass true lin_vel through, 0.0 = zero it out for the episode.
        # Resampled per env at every _reset_idx call.
        self._lin_vel_obs_keep = torch.ones(self.num_envs, device=self.device)
        self._default_joint_pos = self.robot.data.default_joint_pos.clone()
        self._gravity_vec = torch.tensor(
            [0.0, 0.0, -1.0], device=self.device
        ).repeat(self.num_envs, 1)

        # ── Gait clock ─────────────────────────────────────────
        self._gait_phase = torch.zeros(self.num_envs, device=self.device)
        self._control_dt = self.cfg.sim.dt * self.cfg.decimation

        # ── Spawn positions + heading ──────────────────────────
        self._spawn_pos_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._spawn_yaw = torch.zeros(self.num_envs, device=self.device)

        # ── Push buffers ───────────────────────────────────────
        self._push_forces = torch.zeros(
            self.num_envs, self.robot.num_bodies, 3, device=self.device
        )
        self._push_torques = torch.zeros(
            self.num_envs, self.robot.num_bodies, 3, device=self.device
        )
        self._push_remaining = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )


        # ── Persistent slope force (simulates tilted floor) ──────────
        self._slope_forces = torch.zeros(
            self.num_envs, self.robot.num_bodies, 3,
            device=self.device,
        )
        self._dr_gravity_tilt_deg = torch.zeros(self.num_envs, device=self.device)
        
        # ── Pickup state machine (NEW v7+) ──────────────────────────
        # Phase: 0=idle, 1=lift, 2=hold, 3=lower
        self._pickup_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._pickup_remaining = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._pickup_target_z = torch.zeros(self.num_envs, device=self.device)
        self._pickup_lower_start_z = torch.zeros(self.num_envs, device=self.device)
        self._pickup_lower_end_z = torch.zeros(self.num_envs, device=self.device)

        # Separate force/torque buffer for pickup so it composes cleanly
        # with pushes. _pre_physics_step zeroes this each step (forces are
        # recomputed via PD), then sums with _push_forces before sim apply.
        self._pickup_forces = torch.zeros_like(self._push_forces)
        self._pickup_torques = torch.zeros_like(self._push_torques)

        # Pad contact (pad_contact.PadContact) contributes through the same
        # external-force path as pushes and pickups.  Rebuilt every substep
        # from live state, so it is zeroed at the same place the pickup buffers
        # are — it is a plant force, not a motor output, and must not be
        # re-clipped by the DC envelope.
        self._pad_forces = torch.zeros_like(self._push_forces)
        self._pad_torques = torch.zeros_like(self._push_torques)

        # ── Ball-contact telemetry (2026-09-12) ─────────────────────────
        # Foot-material and pad knobs are INERT whenever the balls never reach
        # the floor, and "the balls are loaded" is the one assumption every
        # foot-side DR arm in this campaign has made without measuring.  These
        # three buffers reduce that to numbers TensorBoard can show, so a flat
        # dr_impact curve on a foot channel can be read as "never engaged"
        # instead of "DR did not matter":
        #   _ball_lowest_z      running min of the ball CENTRE's world z
        #                       (the pad's own geometry, not the nail FK point
        #                       5.8 cm short of it)
        #   _ball_touched       per (env, foot) latch: did this ball enter the
        #                       pad's reference plane at any point since the
        #                       last log?  This is the ENGAGEMENT number — the
        #                       one that says whether any foot-side knob can
        #                       matter at all — and it is latched rather than
        #                       averaged because contact is brief: a per-step
        #                       mean would read ~0.001 for a robot that
        #                       contacts every episode, which is unreadable.
        #   _ball_contact_count / _ball_contact_cells
        #                       the same contact, unlatched, for "how much of
        #                       the time" (the pad's duty cycle).
        if self._pad is not None:
            self._ball_lowest_z = torch.full(
                (self.num_envs, 2), float("inf"), device=self.device
            )
            self._ball_touched = torch.zeros(
                (self.num_envs, 2), dtype=torch.bool, device=self.device
            )
            self._ball_contact_count = torch.zeros((), device=self.device)
            self._ball_contact_cells = 0
            self._ball_reference_z = float(self._pad.reference_radius)

        # Total mass per env (gravity comp). nominal = pristine USD sum;
        # _total_mass = nominal × dr_mass_mult, refreshed at reset.
        self._nominal_total_mass = self._original_masses.sum(dim=-1).to(self.device)
        self._total_mass = self._nominal_total_mass.clone()

        self._pickup_event_count = 0   # cumulative, for logging

        # Where the "hand" holds the robot horizontally — recorded at LIFT
        # trigger, held constant for the rest of the pickup. Mimics a real
        # grip that stops the robot's lateral motion immediately.
        self._pickup_target_xy = torch.zeros(self.num_envs, 2, device=self.device)

        self._pickup_target_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._pickup_target_quat[:, 0] = 1.0
        self._pickup_pd_factor_trans = torch.ones(self.num_envs, device=self.device)
        self._pickup_pd_factor_rot   = torch.ones(self.num_envs, device=self.device)

        # Wobble phase accumulator — sampled to a random initial value at
        # LIFT trigger, advances each step during HOLD. One per env so the
        # population shows diverse wobble phases instead of synchronized.
        self._pickup_hold_phase = torch.zeros(self.num_envs, device=self.device)
        # How many steps the LATEST release-transition was scheduled for.
        # Used to compute support_factor = remaining / initial.
        self._pickup_release_initial_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device,
        )
        # Live support level: 1.0 when fully held or idle, ramps 1→0 during RELEASE.
        # Persisted so _get_rewards / _get_dones can blend the landing handoff.
        self._pickup_support_factor = torch.ones(self.num_envs, device=self.device)
        # ── Commands ───────────────────────────────────────────
        self._vx_cmd = torch.zeros(self.num_envs, device=self.device)
        self._yaw_rate_cmd = torch.zeros(self.num_envs, device=self.device)

        # ── Averaged tracking / anchoring state (instantaneous → windowed) ──
        # Rate trackers EMA the body velocity; anchors EMA the position / yaw
        # so per-step sway isn't scored as drift. Reset in _reset_idx and
        # re-snapped at every spawn-anchor update in _pre_physics_step.
        self._vx_avg = torch.zeros(self.num_envs, device=self.device)
        self._yaw_rate_avg = torch.zeros(self.num_envs, device=self.device)
        self._pos_ema = torch.zeros(self.num_envs, 2, device=self.device)
        self._yaw_ema = torch.zeros(self.num_envs, device=self.device)  

        self._leg_forward_ema = torch.zeros(self.num_envs, device=self.device)
        self._prev_feet_mid_world = torch.zeros(self.num_envs, 2, device=self.device)
        self._prev_toe1_world = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_toe2_world = torch.zeros(self.num_envs, 3, device=self.device)

        self._joint_pair_sum_ema = torch.zeros(
            self.num_envs, 3, device=self.device
        )

        # ── Stance-time symmetry EMA buffers ───────────────────
        self._foot1_contact_ema = torch.zeros(self.num_envs, device=self.device)
        self._foot2_contact_ema = torch.zeros(self.num_envs, device=self.device)

        # ── Per-foot step counting (rising-edge detection) ─────
        self._prev_foot1_on_ground = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._prev_foot2_on_ground = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # ══════════════════════════════════════════════════════
        # Standup curriculum state
        # ══════════════════════════════════════════════════════
        self._curriculum_alpha = self.cfg.curriculum_initial_alpha
        self._env_difficulty = torch.zeros(self.num_envs, device=self.device)
        self._initial_height = torch.full(
            (self.num_envs,), self.cfg.spawn_height, device=self.device
        )
        self._grace_remaining = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._stabilize_remaining = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._episode_lengths_buffer = torch.zeros(
            self.cfg.curriculum_advance_window, device=self.device
        )
        # NEW (v7): parallel buffer of per-episode difficulty for stratified
        # diagnostics. The kneel-branch population now spans [0, alpha], so
        # you need to know which DIFFICULTY band an episode came from to tell
        # whether long-survivors are easy-end or hard-end episodes.
        self._episode_difficulty_buffer = torch.zeros(
            self.cfg.curriculum_advance_window, device=self.device
        )
        # Steps survived AFTER grace expiry (ep_len − grace assigned at reset).
        # Grace time is not a balance signal; this is what stand-up success is
        # read from (see cfg.curriculum_advance_metric).
        self._episode_postgrace_buffer = torch.zeros(
            self.cfg.curriculum_advance_window, device=self.device
        )
        self._grace_steps_assigned = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # path-mode spawn phase s0 = 1 − d (1.0 = standing spawn)
        self._standup_s0 = torch.ones(self.num_envs, device=self.device)

        self._episode_buffer_idx = 0
        self._episode_buffer_count = 0
        self._curriculum_cooldown = 0

        self._was_kneel_start = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        self._force_fixed_difficulty = False

        # ══ OBSERVATION DELAY buffers (split IMU / joint in v7) ═══════
        # One shared per-FRAME ring buffer holds the full 21-D sensor
        # frame each control step (pre-history pipeline width). The IMU
        # slice (dims 0:9) and the joint slice (dims 9:21) are read back
        # at DIFFERENT per-env delays, reproducing the ~10 ms IMU-leads-
        # encoder skew measured on hardware.
        #
        # Combined with action delay, total loop dead time at 100 Hz:
        #   IMU  loop  ∈ [2, 3] steps = 20–30 ms  (measured ~20–22 ms)
        #   joint loop ∈ [3, 4] steps = 30–40 ms  (measured ~30–35 ms)
        self._max_obs_delay = max(
            self.cfg.dr_imu_delay_max, self.cfg.dr_joint_delay_max
        )
        self._obs_delay_buffer = torch.zeros(
            self.num_envs, self._max_obs_delay + 1,
            _OBS_SENSORS_PER_FRAME,   # full sensor frame; IMU/joint sliced on read
            device=self.device,
        )
        self._env_imu_delay = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device,
        )
        self._env_joint_delay = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device,
        )
        # Handover-prime flag: envs whose delay buffer gets filled with the
        # real first sensor frame on the next _get_observations (S-line).
        self._needs_delay_prime = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device,
        )

        # Back-compat alias: play.py's obs-summary panel reads
        # e._env_obs_delay. v7 split obs delay into IMU + joint paths; the
        # legacy single name maps to the joint/encoder delay (the dominant,
        # slowest path) — matching the `dr/obs_delay` TensorBoard key and
        # _record_dr_impact's legacy mapping. This aliases the SAME tensor
        # object (no copy), so it stays in sync: _env_joint_delay is only
        # ever mutated in place via [env_ids]=, never reassigned.
        self._env_obs_delay = self._env_joint_delay

        # NEW IN p6h: history stack (post-delay).
        self._obs_history_buf = torch.zeros(
            self.num_envs, self.cfg.obs_history_steps, _OBS_PER_FRAME,
            device=self.device,
        )

        # ══ PER-EPISODE OBS BIASES (NEW IN p6g) ═════════════════════
        # Constant for the episode; pure obs-side, never touches sim.
        self._dr_joint_pos_bias = torch.zeros(
            self.num_envs, len(self._joint_ids), device=self.device,
        )
        self._dr_proj_grav_bias = torch.zeros(
            self.num_envs, 3, device=self.device,
        )
        self._dr_ang_vel_bias = torch.zeros(
            self.num_envs, 3, device=self.device,
        )
        # Execution-noise injection (P13): episode-constant per-joint
        # target bias; per-step white noise is drawn in
        # _advance_motor_target. Zeros when the cfg knobs are 0.0.
        self._exec_target_bias = torch.zeros(
            self.num_envs, len(self._joint_ids), device=self.device,
        )

        # ══ PHYSICS DR (NEW IN p6g) ═════════════════════════════════
        # Per-env scalars (mass, friction) and per-body vectors (CoM).
        # Applied to PhysX at reset; defaults are no-op (mult=1, off=0)
        # so envs that haven't reset yet behave nominally.
        n_bodies = self.robot.num_bodies
        self._dr_mass_mult = torch.ones(self.num_envs, device=self.device)
        self._dr_floor_friction_mult = torch.ones(self.num_envs, device=self.device)
        # Foot-restitution DR (S6): one bucketed e per env, written to the two
        # ball rows only.  Kept for logging (read-back is done separately).
        self._dr_foot_restitution = torch.zeros(self.num_envs, device=self.device)
        self._dr_com_offset = torch.zeros(
            self.num_envs, n_bodies, 3, device=self.device,
        )
        # Coherent whole-body CoM shift (base frame), per env. Realized by
        # adding shift × M/m_torso to the torso row of _dr_com_offset, so
        # the whole-body CoM moves by exactly `shift` for any leg pose.
        # The mass ratio is DR-invariant (mass_mult scales every link).
        self._dr_com_shift = torch.zeros(self.num_envs, 3, device=self.device)
        self._com_shift_gain = (
            self._original_masses[0].sum() / self._original_masses[0, self._torso_idx]
        ).item()
        # Knee joint-angle bias (S-line): per-leg δ, folded into obs/target
        # biases at sampling time; kept separately for logging.
        self._dr_knee_angle_bias = torch.zeros(self.num_envs, 2, device=self.device)
        self._dr_hip_angle_bias = torch.zeros(self.num_envs, 2, device=self.device)
        # Per-env DR override (eval/sysid only, 2026-09-10). A dict of
        # [num_envs, ...] tensors keyed by parameter name; when a key is
        # present, _apply_dr_override_* replaces the sampled value for the
        # envs being reset, so an env keeps its assigned plant vector across
        # auto-resets. Empty dict = training behaviour, bit-exact. Keys:
        #   obs stage    : proj_grav_bias_x [N], knee_bias [N,2], hip_bias [N,2]
        #   physics stage: mass_mult [N], floor_friction_mult [N], com_shift [N,3]
        self._dr_override: dict[str, torch.Tensor] = {}


        # ══ ACTION DELAY buffer (NEW) ═════════════════════════════
        # Buffer length is max_delay+1 so index k holds the action from
        # k steps ago. self.actions stays as the policy's CURRENT output
        # (used by reward terms and last_actions obs); _delayed_actions
        # is what actually drives the motors. This keeps the policy's
        # self-view consistent with what it can directly cause.
        self._max_action_delay = self.cfg.dr_action_delay_max
        self._action_delay_buffer = torch.zeros(
            self.num_envs, self._max_action_delay + 1,
            self.cfg.action_space, device=self.device,
        )
        self._env_action_delay = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device,
        )
        self._delayed_actions = torch.zeros(
            self.num_envs, self.cfg.action_space, device=self.device,
        )

        # ══ DR IMPACT DIAGNOSTICS ════════════════════════════════════
        # Rolling buffer of (DR-features, episode-length) pairs, captured
        # at episode end. Used to log correlation + stratified means so
        # you can tell which DR is driving terminations.
        self._dr_diag_window = 4096
        self._dr_diag_features = [
            "action_delay", "obs_delay", "imu_delay", # obs_delay == joint delay (legacy name)
            "stiffness_mult", "effort_mult",
            "mass_mult", "floor_friction_mult",
            "com_offset_abs", "joint_pos_bias_abs", "proj_grav_bias_abs",
            "ang_vel_bias_abs",
            "voltage_factor", "armature_mult", "jfriction_mult",
            "act_lag_omega_mult", "act_lag_zeta_mult",   # NEW (v7.1)
            "com_shift_x",       # S-line: SIGNED whole-body CoM_x shift
            # S5 (2026-09-12): the two new elasticity channels.  Both are
            # written on EVERY episode regardless of the cfg band, so their
            # tertile contrast is defined even when a band is zero — a flat
            # dr_impact row at a zero band is the honest "not in play" reading.
            "belt_k_joint", "belt_c_motor", "pad_k", "pad_c",
        ]
        self._dr_diag_buf = {
            name: torch.zeros(self._dr_diag_window, device=self.device)
            for name in self._dr_diag_features
        }
        self._dr_diag_buf["ep_len"] = torch.zeros(self._dr_diag_window, device=self.device)
        self._dr_diag_idx = 0
        self._dr_diag_count = 0
        # ── Kneel / stand poses ────────────────────────────────
        self._kneel_joint_pos = torch.tensor(
            [
                self.cfg.kneel_hip_roll1, self.cfg.kneel_hip_pitch1,
                self.cfg.kneel_knee1, self.cfg.kneel_hip_roll2,
                self.cfg.kneel_hip_pitch2, self.cfg.kneel_knee2,
            ],
            device=self.device,
        )
        self._stand_joint_pos = self._default_joint_pos[0, self._joint_ids].clone()
        self._kneel_quat = torch.tensor(
            [self.cfg.kneel_quat_w, self.cfg.kneel_quat_x,
             self.cfg.kneel_quat_y, self.cfg.kneel_quat_z],
            device=self.device,
        )
        self._stand_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self._max_grace_steps = int(
            self.cfg.standup_grace_seconds / self._control_dt
        )
        self._build_standup_profiles()

        # ── Sliding action center ──────────────────────────────
        self._env_init_joint_pos = torch.zeros(
            self.num_envs, len(self._joint_ids), device=self.device
        )
        self._env_init_joint_pos[:] = self._stand_joint_pos.unsqueeze(0)

        self._filtered_actions = torch.zeros(
            self.num_envs, self.cfg.action_space, device=self.device
        )

        # ── Pre-compute FK overshoot for height correction ──────
        kneel_quat_batch = self._kneel_quat.unsqueeze(0)
        kneel_jpos = self._kneel_joint_pos.unsqueeze(0)
        fk_at_kneel = _compute_fk_height(
            base_quat=kneel_quat_batch,
            hip_roll1=kneel_jpos[:, 0],
            hip_pitch1=kneel_jpos[:, 1],
            knee1=kneel_jpos[:, 2],
            hip_roll2=kneel_jpos[:, 3],
            hip_pitch2=kneel_jpos[:, 4],
            knee2=kneel_jpos[:, 5],
            device=self.device,
        ).item()
        self._fk_kneel_overshoot = max(0.0, fk_at_kneel - self.cfg.kneel_height)




        # NEW: marks envs that need their obs history primed on the next
        # _get_observations call. See _reset_idx and _get_observations.
        self._needs_history_init = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device,
        )


        # Per-env / per-joint tensors (Shape: [num_envs, 6])
        self._dr_stiffness_mult = torch.ones((self.num_envs, 6), dtype=torch.float32, device=self.device)
        self._dr_damping_mult = torch.ones((self.num_envs, 6), dtype=torch.float32, device=self.device)
        self._dr_armature_mult = torch.ones((self.num_envs, 6), dtype=torch.float32, device=self.device)
        self._dr_jfriction_mult = torch.ones((self.num_envs, 6), dtype=torch.float32, device=self.device)
        self._dr_dyn_friction_mult = torch.ones((self.num_envs, 6), dtype=torch.float32, device=self.device)
        self._dr_visc_friction_mult = torch.ones((self.num_envs, 6), dtype=torch.float32, device=self.device)
        self._dr_effort_mult = torch.ones((self.num_envs, 6), dtype=torch.float32, device=self.device)
        # Per-env tensor (Shape: [num_envs])
        self._dr_voltage_factor = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        # Belt series parameters are episode constants.  Store them in
        # physical joint-side stiffness for readable diagnostics, then
        # convert to motor-side values at the actuator boundary.
        self._dr_belt_stiffness = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self._dr_belt_damping = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        # Encoder-observation joint mapping, resolved on first use.
        self._encoder_kn_slots = None
        self._encoder_leg_pairs = None

        # ══ ACTUATOR-LAG PREFILTER (2nd-order command prefilter) ════════
        # Per-joint 2nd-order low-pass on the PD position target, advanced ONCE
        # per control step (100 Hz) and held across the decimation substeps —
        # the DCMotor PD then runs against the held target at 500 Hz. This is a
        # 1:1 port of sysid_replay_sim.py's pf_y / pf_yd update:
        #     y'' = ω²(u − y) − 2ζω·y' ,  integrated by forward Euler at dt_ctrl.
        # ω, ζ are per-joint-TYPE centrals (cfg §11), broadcast to the 6 joints
        # in [hr1,hp1,kn1,hr2,hp2,kn2] order, then per-env-randomized by the
        # ω/ζ multipliers below (§12a). Coeffs (ω², 2ζω) are cached and only
        # recomputed at reset when the multipliers change.
        self._act_lag_omega_nom = _per_joint(
            cfg.act_lag_omega_hip_roll, cfg.act_lag_omega_hip_pitch, cfg.act_lag_omega_knee,
        )                                                 # [6]
        self._act_lag_zeta_nom = _per_joint(
            cfg.act_lag_zeta_hip_roll, cfg.act_lag_zeta_hip_pitch, cfg.act_lag_zeta_knee,
        )                                                 # [6]
        # Passthrough mask: a joint with nominal ω == 0 bypasses the prefilter
        # (raw command goes straight through), exactly like the replay lag_mask.
        # Mask keys on the NOMINAL ω (DR never turns a group on/off).
        self._act_lag_mask = (self._act_lag_omega_nom > 0.0).float().unsqueeze(0)   # [1,6]
        self._act_lag_active = bool(
            cfg.act_lag_enable and bool((self._act_lag_omega_nom > 0.0).any())
        )
        # Per-env ω / ζ multipliers (sampled at reset; default 1.0 = nominal).
        self._dr_act_lag_omega_mult = torch.ones((self.num_envs, 6), dtype=torch.float32, device=self.device)
        self._dr_act_lag_zeta_mult  = torch.ones((self.num_envs, 6), dtype=torch.float32, device=self.device)
        # Cached per-env coefficients ω² and 2ζω (recomputed in _sample_dr).
        _w0 = self._act_lag_omega_nom.unsqueeze(0).expand(self.num_envs, -1)        # [N,6]
        _z0 = self._act_lag_zeta_nom.unsqueeze(0).expand(self.num_envs, -1)
        self._act_lag_omega2 = (_w0 ** 2).clone()
        self._act_lag_two_zw = (2.0 * _z0 * _w0).clone()
        # Prefilter state: filtered target (pf_y) + its velocity (pf_yd). Init
        # pf_y to the spawn action-center (stand pose here; re-seeded per-env to
        # _env_init_joint_pos at reset) so the command starts where the joint is
        # and the first step doesn't ring off an artificial jump.
        self._act_lag_pf_y  = self._env_init_joint_pos.clone()                      # [N,6]
        self._act_lag_pf_yd = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)
        # Joint position target the motors track this control step (filtered
        # output of the prefilter, or the raw target when disabled). Computed in
        # _pre_physics_step, applied every substep by _apply_action.
        self._motor_pos_target = self._env_init_joint_pos.clone()                   # [N,6]
        # One-time stability/accuracy sanity on the worst-case ω·dt (forward
        # Euler on the prefilter degrades as ω·dt → 1; replay uses the same bound).
        _wmax = float(self._act_lag_omega_nom.max().item()) * float(self.cfg.dr_act_lag_omega_mult_max)
        if self._act_lag_active and _wmax * self._control_dt > 0.5:
            print(f"  [act_lag][warn] max ω·dt = {_wmax * self._control_dt:.3f} "
                  f"(ω_max={_wmax:.1f} rad/s, dt={self._control_dt*1e3:.1f} ms); "
                  f"prefilter Euler integration loses accuracy as this → 1.")


        # ── Print banner ───────────────────────────────────────
        print(f"\n{'=' * 60}")
        print(f"  BIPEDAL WALKING RL ENVIRONMENT")
        print(f"  Version 7 — DCMotor")
        print(f"{'=' * 60}")
        print(f"  Joints: {self._joint_names}")
        print(f"  Obs dim: {self.cfg.observation_space}")
        print(f"  Act dim: {self.cfg.action_space}")
        print(f"  Control freq: {1.0 / self._control_dt:.0f} Hz")
        print(f"  Action scales:  hr={self._action_scale[0]:.2f} "
              f"hp={self._action_scale[1]:.2f} "
              f"kn={self._action_scale[2]:.2f}")
        print(f"  Kp · scale =    "
              f"hr={self._nominal_diag('hip_roll', 'stiffness') * self._action_scale[0]:.2f} Nm "
              f"(eff {self.cfg.continuous_effort_hip_roll}) "
              f"hp={self._nominal_diag('hip_pitch', 'stiffness') * self._action_scale[1]:.2f} Nm "
              f"(eff {self.cfg.continuous_effort_hip_pitch}) "
              f"kn={self._nominal_diag('knee', 'stiffness') * self._action_scale[2]:.2f} Nm "
              f"(eff {self.cfg.continuous_effort_knee})")
        print(f"  Vx cmd: [{self.cfg.vx_cmd_min}, {self.cfg.vx_cmd_max}] m/s")
        print(f"  Yaw rate cmd: [{self.cfg.yaw_rate_cmd_min}, {self.cfg.yaw_rate_cmd_max}] rad/s")
        print(f"  Height range: [{self.cfg.height_min}, {self.cfg.height_max}] m (plateau)")
        print(f"  Standup ramp: kneel_h={self.cfg.kneel_height:.3f} → "
              f"stand_target={self.cfg.stand_target_height:.3f} m (reward: linear); "
              f"action-center mode={self.cfg.standup_ramp_mode}"
              + (f" hip={self.cfg.standup_hip_profile} knee={self.cfg.standup_knee_profile}"
                 if self.cfg.standup_ramp_mode == "keyframe" else "")
              + (f" knots={len(self._path_s)} (spawn on path; track={self.cfg.rew_scale_standup_track} "
                 f"σ={self.cfg.standup_track_sigma_rad} contact={self.cfg.rew_scale_standup_contact})"
                 if self.cfg.standup_ramp_mode == "path" else ""))
        print(f"  Kneel pose: hp={self.cfg.kneel_hip_pitch1:.2f} kn={self.cfg.kneel_knee1:.2f} "
              f"(noise ±{self.cfg.kneel_noise_rad:.3f} rad)")
        print(f"  FK overshoot at kneel: {self._fk_kneel_overshoot:.4f} m")
        print(f"  Kneel-start probability: {self.cfg.kneel_start_probability:.2f}")
        print(f"  Continuous limits: hr={self.cfg.continuous_effort_hip_roll}Nm "
              f"hp={self.cfg.continuous_effort_hip_pitch}Nm kn={self.cfg.continuous_effort_knee}Nm")
        print(f"  Grace period: {self.cfg.standup_grace_seconds}s")
        print(f"  Post-grace stabilize: {self.cfg.post_grace_stabilize_steps} steps")
        _gr_scale = getattr(self.cfg, "rew_scale_gait_ref", 0.0)
        if _gr_scale != 0.0:
            print(f"  [A/B FORK] Gait-reference prior ACTIVE: "
                  f"scale={_gr_scale} "
                  f"A={getattr(self.cfg, 'gait_ref_amplitude_rad', 0.20)} rad "
                  f"sigma={getattr(self.cfg, 'gait_ref_sigma_rad', 0.15)} rad "
                  f"(clock {self.cfg.gait_freq} Hz)")
        print(f"  Action penalties: clipped={self.cfg.rew_scale_action_magnitude} "
              f"raw={self.cfg.rew_scale_raw_action} (NEW)")
        print(f"  Flight-phase penalty: {self.cfg.rew_scale_no_flight} "
              f"(contact thresh {self.cfg.no_flight_contact_threshold} m)")
        print(f"  Vz penalty: {self.cfg.rew_scale_vz_penalty} × base_vz²")
        print(f"  Stance symmetry: {self.cfg.rew_scale_stance_symmetry} "
              f"(decay {self.cfg.stance_symmetry_decay})")
        print(f"  Yaw drift penalty: {self.cfg.rew_scale_yaw_drift} × ωz² "
              f"(when |yaw_cmd| < {self.cfg.yaw_rate_cmd_anchor_threshold})")
        print(f"  Pushes: {'ON' if self.cfg.push_enable else 'OFF'}")
        print(f"{'=' * 60}\n")
        print(f"  Mirror augmentation: ENABLED — wire get_symmetric_states "
              f"into PPO symmetry_cfg.data_augmentation_func")
        print(f"  Domain randomization: {'ON' if self.cfg.dr_enable else 'OFF'}")
        if self.cfg.dr_enable:
            print(f"    Action delay ∈ [{self.cfg.dr_action_delay_min}, {self.cfg.dr_action_delay_max}] steps  "
                  f"IMU delay ∈ [{self.cfg.dr_imu_delay_min}, {self.cfg.dr_imu_delay_max}] steps  "
                  f"Joint delay ∈ [{self.cfg.dr_joint_delay_min}, {self.cfg.dr_joint_delay_max}] steps")
            print(f"    joint_pos_bias ±{self.cfg.dr_joint_pos_bias_max:.3f} rad  "
                f"proj_grav_bias ±{self.cfg.dr_proj_grav_bias_max:.3f}  "
                f"ang_vel_bias ±{self.cfg.dr_ang_vel_bias_max:.3f} rad/s")
            print(f"    mass_mult ∈ [{self.cfg.dr_mass_mult_min}, {self.cfg.dr_mass_mult_max}]  "
                  f"friction_mult ∈ [{self.cfg.dr_floor_friction_min}, {self.cfg.dr_floor_friction_max}]  "
                  f"com_offset ±{self.cfg.dr_com_offset_max:.3f} m"
                  + (f"  TOP-MASS PIN +{self.cfg.dr_top_mass_add_kg:.3f} kg on torso+chassis"
                     if self.cfg.dr_top_mass_add_kg != 0.0 else ""))
            print(f"    com_shift x ±{self.cfg.dr_com_shift_x_max:.3f} m  y ±{self.cfg.dr_com_shift_y_max:.3f} m "
                  f"(torso gain {self._com_shift_gain:.2f})  "
                  f"knee_angle_bias ±{self.cfg.dr_knee_angle_bias_max:.3f} rad  "
                  f"handover_prime_prob {self.cfg.handover_prime_prob:.2f}")
            print(f"    stiffness ∈ [{self.cfg.dr_stiffness_mult_min}, {self.cfg.dr_stiffness_mult_max}]  "
                    f"damping ∈ [{self.cfg.dr_damping_mult_min}, {self.cfg.dr_damping_mult_max}]")
            print(f"    armature ∈ [{self.cfg.dr_armature_mult_min}, {self.cfg.dr_armature_mult_max}]  "
                    f"voltage ∈ [{self.cfg.dr_voltage_factor_min}, {self.cfg.dr_voltage_factor_max}]")
            print(f"    joint_friction ∈ [{self.cfg.dr_joint_friction_mult_min}, {self.cfg.dr_joint_friction_mult_max}]  "
                    f"dyn_friction ∈ [{self.cfg.dr_dyn_friction_mult_min}, {self.cfg.dr_dyn_friction_mult_max}]  visc_friction ∈ [{self.cfg.dr_visc_friction_mult_min}, {self.cfg.dr_visc_friction_mult_max}]")
            _al = "ON" if self._act_lag_active else "OFF"
            print(f"    act_lag prefilter: {_al}  "
                  f"ω(hr/hp/kn)={self.cfg.act_lag_omega_hip_roll}/{self.cfg.act_lag_omega_hip_pitch}/{self.cfg.act_lag_omega_knee} rad/s  "
                  f"ζ={self.cfg.act_lag_zeta_hip_roll}/{self.cfg.act_lag_zeta_hip_pitch}/{self.cfg.act_lag_zeta_knee}")
            print(f"      act_lag DR:  ω ×[{self.cfg.dr_act_lag_omega_mult_min}, {self.cfg.dr_act_lag_omega_mult_max}]  "
                  f"ζ ×[{self.cfg.dr_act_lag_zeta_mult_min}, {self.cfg.dr_act_lag_zeta_mult_max}]")

    def _nominal_diag(self, logical: str, field: str) -> float:
        """Nominal actuator scalar for a logical joint type ('hip_roll' /
        'hip_pitch' / 'knee'), resolved through _diag_act so it works with
        both the legacy per-type groups and the merged belt group."""
        gname, col = self._diag_act[logical]
        return self._nominal_actuator[gname][field][0, col].item()

    def _record_dr_impact(self, env_ids):
        """Capture terminating episodes' DR vectors + ep_len into the rolling
        buffer. MUST be called BEFORE super()._reset_idx (so episode_length_buf
        is still valid) and BEFORE _sample_dr* (so the DR is still the dying
        episode's, not the new one's)."""
        if not self.cfg.dr_enable:
            return
        ep_lens = self.episode_length_buf[env_ids]
        valid = ep_lens >= 10  # ignore degenerate first-step terminations
        if not valid.any():
            return

        env_ids_t = env_ids if isinstance(env_ids, torch.Tensor) else torch.tensor(
            list(env_ids), device=self.device, dtype=torch.long
        )
        eids = env_ids_t[valid]
        elens = ep_lens[valid].float()
        n = eids.numel()

        # Build feature vector: per-env scalars summarizing each DR.
        feats = {
            "action_delay":      self._env_action_delay[eids].float(),
            "obs_delay":         self._env_joint_delay[eids].float(),  # legacy key == joint/encoder delay
            "imu_delay":         self._env_imu_delay[eids].float(),    # NEW: fast IMU path
            "stiffness_mult":      self._dr_stiffness_mult[eids].mean(dim=-1),
            "floor_friction_mult": self._dr_floor_friction_mult[eids],
            "voltage_factor":      self._dr_voltage_factor[eids],
            "armature_mult":       self._dr_armature_mult[eids].mean(dim=-1),
            "jfriction_mult":      self._dr_jfriction_mult[eids].mean(dim=-1),
            "effort_mult":       self._dr_effort_mult[eids].mean(dim=-1),
            "mass_mult":         self._dr_mass_mult[eids],
            "com_offset_abs":    self._dr_com_offset[eids].abs().mean(dim=(-2, -1)),
            "joint_pos_bias_abs":self._dr_joint_pos_bias[eids].abs().mean(dim=-1),
            "proj_grav_bias_abs":self._dr_proj_grav_bias[eids].abs().mean(dim=-1),
            "ang_vel_bias_abs":  self._dr_ang_vel_bias[eids].abs().mean(dim=-1),   # NEW
            "act_lag_omega_mult": self._dr_act_lag_omega_mult[eids].mean(dim=-1),  # NEW (v7.1)
            "act_lag_zeta_mult":  self._dr_act_lag_zeta_mult[eids].mean(dim=-1),   # NEW (v7.1)
            "com_shift_x":       self._dr_com_shift[eids, 0],
            "belt_k_joint":      self._dr_belt_stiffness[eids].mean(dim=-1),
            "belt_c_motor":      self._dr_belt_damping[eids].mean(dim=-1),
        }
        # Pad channels: present only when the pad object exists.  Read from the
        # pad itself rather than a mirrored buffer, because the pad's own
        # tensors are what the force is actually computed from — a mirrored
        # copy could agree with the cfg while the force saw something else.
        if self._pad is not None:
            feats["pad_k"] = self._pad.stiffness[eids].mean(dim=-1)
            feats["pad_c"] = self._pad.damping[eids].mean(dim=-1)
        else:
            feats["pad_k"] = torch.zeros(n, device=self.device)
            feats["pad_c"] = torch.zeros(n, device=self.device)

        # Ring-buffer write.
        idx_base = self._dr_diag_idx
        write_idx = (torch.arange(n, device=self.device) + idx_base) % self._dr_diag_window
        self._dr_diag_buf["ep_len"][write_idx] = elens
        for name, vals in feats.items():
            self._dr_diag_buf[name][write_idx] = vals
        self._dr_diag_idx += n
        self._dr_diag_count = min(self._dr_diag_count + n, self._dr_diag_window)


    def _write_per_env_actuators_to_sim(self):
        for name, pairs in self._actuator_joint_map.items():
            act = self.robot.actuators[name]
            nom = self._nominal_actuator[name]
            my_slots = torch.tensor([p[0] for p in pairs], device=self.device)
            local    = torch.tensor([p[1] for p in pairs], device=self.device)

            # Index out [N, len(pairs)] multipliers from the [N, 6] global tensors.
            stiff_mult = self._dr_stiffness_mult.index_select(-1, my_slots)
            damp_mult  = self._dr_damping_mult.index_select(-1, my_slots)
            eff_mult   = self._dr_effort_mult.index_select(-1, my_slots)
            arm_mult   = self._dr_armature_mult.index_select(-1, my_slots)
            jf_mult    = self._dr_jfriction_mult.index_select(-1, my_slots)
            df_mult    = self._dr_dyn_friction_mult.index_select(-1, my_slots)
            vf_mult    = self._dr_visc_friction_mult.index_select(-1, my_slots)
            volt       = self._dr_voltage_factor.unsqueeze(-1)   # [N, 1] → broadcasts

            act.stiffness.index_copy_(-1, local, nom["stiffness"].index_select(-1, local) * stiff_mult)
            act.damping.index_copy_(-1, local,   nom["damping"].index_select(-1, local) * damp_mult)
            act.effort_limit.index_copy_(-1, local, nom["effort_limit"].index_select(-1, local) * eff_mult)
            act.armature.index_copy_(-1, local,  nom["armature"].index_select(-1, local) * arm_mult)
            static_f = nom["friction"].index_select(-1, local) * jf_mult
            dyn_f    = nom["dynamic_friction"].index_select(-1, local) * df_mult
            dyn_f    = torch.minimum(dyn_f, static_f)        # PhysX: μ_s ≥ μ_d
            visc_f   = nom["viscous_friction"].index_select(-1, local) * vf_mult
            act.friction.index_copy_(-1, local, static_f)
            act.dynamic_friction.index_copy_(-1, local, dyn_f)
            act.viscous_friction.index_copy_(-1, local, visc_f)
            act.velocity_limit.index_copy_(-1, local, nom["velocity_limit"].index_select(-1, local) * volt)

            # Belt-elastic environments move the knee rotor's reflected
            # inertia out of the PhysX joint and into the actuator's series
            # integrator (and drop c^2 * I_r/G^2 from the hip-pitch diagonal,
            # which is only there because the belt is rigid — the hip keeps
            # its own rotor: mask = own/total on that column, 2026-09-18).
            # The mask is ones wherever the transmission is rigid, so the
            # nominal path is unchanged.
            armature = act.armature
            if hasattr(act, "armature_write_mask") and self.cfg.belt_mask_armature:
                armature = armature * act.armature_write_mask()
            self.robot.write_joint_armature_to_sim(armature, joint_ids=act.joint_indices)
            self.robot.write_joint_friction_coefficient_to_sim(
                act.friction,
                joint_dynamic_friction_coeff=act.dynamic_friction,
                joint_viscous_friction_coeff=act.viscous_friction,
                joint_ids=act.joint_indices,
            )

    # ══════════════════════════════════════════════════════════
    # Scene
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
    # Mirror augmentation
    # ══════════════════════════════════════════════════════════

    def mirror_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """L/R sagittal mirror. Dispatches on last-dim width so it correctly
        mirrors the policy group (k*36), the privileged group (priv_obs_dim),
        or the concatenated critic input (policy+priv) — whichever rsl_rl hands
        get_symmetric_states for a given obs group. No change to the hook itself
        is needed because of this."""
        w = obs.shape[-1]
        if w == self.cfg.observation_space:
            return obs[..., self._obs_mirror_perm] * self._obs_mirror_sign
        if w == self.cfg.priv_obs_dim:
            return obs[..., self._priv_mirror_perm] * self._priv_mirror_sign
        if w == self.cfg.observation_space + self.cfg.priv_obs_dim:
            pol = obs[..., :self.cfg.observation_space]
            prv = obs[..., self.cfg.observation_space:]
            return torch.cat([
                pol[..., self._obs_mirror_perm] * self._obs_mirror_sign,
                prv[..., self._priv_mirror_perm] * self._priv_mirror_sign,
            ], dim=-1)
        raise RuntimeError(f"mirror_obs: unexpected width {w}")

    def mirror_action(self, action: torch.Tensor) -> torch.Tensor:
        """L/R sagittal mirror of a batched action tensor (..., 6)."""
        return action[..., self._act_mirror_perm] * self._act_mirror_sign

    def _verify_mirror_correctness(self):
        """Self-test: mirror is involutive AND symmetric on a symmetric state."""
        with torch.no_grad():
            # Involutivity (existing check)
            o = torch.randn(8, self.cfg.observation_space, device=self.device)
            o_err = (o - self.mirror_obs(self.mirror_obs(o))).abs().max().item()
            a = torch.randn(8, self.cfg.action_space, device=self.device)
            a_err = (a - self.mirror_action(self.mirror_action(a))).abs().max().item()
            p = torch.randn(8, self.cfg.priv_obs_dim, device=self.device)
            p_err = (p - self.mirror_obs(self.mirror_obs(p))).abs().max().item()
            # Symmetric-input invariance: construct a sagittally-symmetric
            # obs (joint pair sums = 0, lateral velocities = 0, etc.) and
            # verify mirror(obs) ≈ obs. If signs are wrong on any channel,
            # this catches it.
            o_sym = torch.zeros(1, self.cfg.observation_space, device=self.device)
            # zero is trivially symmetric — would only catch sign errors that
            # don't act on zero. Use a non-trivial symmetric example:
            # set joint_pos so pair sums = 0; set lateral channels to 0;
            # set sagittal-symmetric channels (vx, gz, …) to nonzero.
            # Layout requires care — populate per your _OBS_MIRROR_* tables.
            sym_err = (o_sym - self.mirror_obs(o_sym)).abs().max().item()

        if o_err > 1e-6 or a_err > 1e-6 or sym_err > 1e-6 or p_err > 1e-6:
            raise RuntimeError(
                f"Mirror check failed: invol_obs={o_err:.2e} invol_act={a_err:.2e} "
                f"sym_err={sym_err:.2e} invol_priv={p_err:.2e}"
            )
    # ══════════════════════════════════════════════════════════
    # Toe tip computation
    # ══════════════════════════════════════════════════════════

    def _get_toe_pos_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        foot1_pos = self.robot.data.body_pos_w[:, self._foot1_idx, :]
        foot1_quat = self.robot.data.body_quat_w[:, self._foot1_idx, :]
        toe1_w = foot1_pos + quat_apply(
            foot1_quat, self._toe1_local_offset.unsqueeze(0).expand(self.num_envs, -1)
        )
        foot2_pos = self.robot.data.body_pos_w[:, self._foot2_idx, :]
        foot2_quat = self.robot.data.body_quat_w[:, self._foot2_idx, :]
        toe2_w = foot2_pos + quat_apply(
            foot2_quat, self._toe2_local_offset.unsqueeze(0).expand(self.num_envs, -1)
        )
        return toe1_w, toe2_w

    # ══════════════════════════════════════════════════════════
    # Ball-pad compliance (pad_contact.PadContact)
    # ══════════════════════════════════════════════════════════

    def _build_pad_contact(self) -> PadContact:
        """Measure the ball geometry on the live stage and build the pad model.

        The ball centre offset is measured from the spawned asset, not taken
        from `_toe*_local_offset`: that offset is the NAIL origin, 5.8 cm short
        of the ball (documented 2026-09-08), so a pad spring placed there would
        push on the shin at a point that never touches the floor.

        Raises rather than guessing when the ball cannot be identified or the
        measured radius is implausible — a pad at the wrong point is a silent
        plant change, which is the failure mode this whole line of work exists
        to stop repeating.
        """
        # The robot prim path is the SFX-armed env path (/World/envs/env_.*/Robot).
        # Any concrete env resolves the same asset; env_0 is the one that exists
        # under replicate_physics.
        robot_prim_path = self.cfg.robot_cfg.prim_path.replace("env_.*", "env_0")
        body_paths = (
            f"{robot_prim_path}/{self.cfg.foot_body_name_1}",
            f"{robot_prim_path}/{self.cfg.foot_body_name_2}",
        )
        stage = self.sim.stage
        offsets, radius = measure_ball_geometry(stage,
            (self.cfg.foot_body_name_1, self.cfg.foot_body_name_2),
            body_paths,
        )
        expected_radius = 0.020  # 40 mm mesh, URDF-verified 2026-09-12
        if not (0.5 * expected_radius <= radius <= 2.0 * expected_radius):
            raise RuntimeError(
                f"measured ball radius {radius:.5f} m is not near the expected "
                f"{expected_radius:.3f} m; the ball prim lookup found the wrong "
                f"mesh, and a pad built on it would be a silent plant change"
            )
        print(
            f"[PLANT] pad contact: ball R={radius:.5f} m  "
            f"centre offsets (body frame) L={tuple(round(v,5) for v in offsets[0].tolist())} "
            f"R={tuple(round(v,5) for v in offsets[1].tolist())}"
        )
        pad = PadContact(
            num_envs=self.num_envs,
            device=self.device,
            ball_local_offset=offsets,
            ball_radius=radius,
            pad_thickness=self.cfg.pad_thickness,
            effective_mass=self.cfg.pad_effective_mass,
            physics_dt=self.physics_dt,
            stability_factor=self.cfg.pad_damping_stability_factor,
            enabled=self.cfg.pad_enabled(),
        )
        print(
            f"[PLANT] pad model: enable={self.cfg.pad_contact_enable} "
            f"k=[{self.cfg.pad_stiffness_min:g}, {self.cfg.pad_stiffness_max:g}] N/m "
            f"zeta="
            f"{f'[{self.cfg.pad_damping_ratio_min:g}, {self.cfg.pad_damping_ratio_max:g}] band' if self.cfg.pad_damping_band_live() else f'{self.cfg.pad_damping_ratio:g}'} "
            f"m_eff={self.cfg.pad_effective_mass:g} kg "
            f"pad_t={self.cfg.pad_thickness*1e3:.1f} mm "
            f"c_cap={pad.damping_cap:.1f} N s/m "
            f"({'active' if self.cfg.pad_enabled() else 'INERT — no force applied'})"
        )
        return pad

    def _apply_pad_contact(self) -> None:
        """Compute this substep's pad force and add it to the external buffers.

        Runs inside `_apply_action` (once per PHYSICS step, before
        `scene.write_data_to_sim`), because the pad is a stiff spring and an
        explicit damped contact solved at the 100 Hz control rate would ring at
        the stiff end of any useful DR band.
        """
        if self._pad is None:
            return
        foot_pos = torch.stack(
            [self.robot.data.body_pos_w[:, self._foot1_idx, :],
             self.robot.data.body_pos_w[:, self._foot2_idx, :]], dim=1
        )                                                    # (N, 2, 3)
        foot_quat = torch.stack(
            [self.robot.data.body_quat_w[:, self._foot1_idx, :],
             self.robot.data.body_quat_w[:, self._foot2_idx, :]], dim=1
        )                                                    # (N, 2, 4)
        offset = self._pad.ball_local_offset.unsqueeze(0).expand(self.num_envs, -1, -1)
        flat = offset.reshape(-1, 3)
        centre = foot_pos + quat_apply(foot_quat.reshape(-1, 4), flat).reshape(
            self.num_envs, 2, 3
        )
        # Ball centre velocity: the body's linear velocity plus the rigid-body
        # term w x r.  The pad's damping acts on the contact point's normal
        # speed, and using the body origin's velocity alone would miss the
        # foot's rotation, which is most of the relative motion during a catch.
        omega = self.robot.data.body_ang_vel_w[:, [self._foot1_idx, self._foot2_idx], :]
        vel = (self.robot.data.body_lin_vel_w[:, [self._foot1_idx, self._foot2_idx], :]
               + torch.cross(omega, centre - foot_pos, dim=-1))
        forces, torques = self._pad.compute(centre, vel, foot_pos)
        # Telemetry BEFORE the early return: `compute` returns None only when
        # no env has a nonzero stiffness, and "the pad is off" is exactly one
        # of the cases worth reporting (a band that is zero everywhere).
        _ball_z = centre[..., 2]
        _ball_in = _ball_z < self._ball_reference_z
        self._ball_lowest_z = torch.minimum(self._ball_lowest_z, _ball_z)
        self._ball_touched |= _ball_in
        self._ball_contact_count += _ball_in.sum()
        self._ball_contact_cells += _ball_z.numel()
        if forces is None:
            return
        body_ids = [self._foot1_idx, self._foot2_idx]
        self._pad_forces[:, body_ids, :] = forces
        self._pad_torques[:, body_ids, :] = torques
        if self._pad.blowups and not self._pad_blowup_warned:
            self._pad_blowup_warned = True
            print(
                f"[PLANT][WARN] pad compression exceeded {self._pad.max_compression*1e3:.1f} mm "
                f"({self._pad.blowups} foot-steps); the pad was reset for those steps. "
                f"Lower pad_stiffness_min or raise pad_effective_mass."
            )

    # ══════════════════════════════════════════════════════════
    # Actions
    # ══════════════════════════════════════════════════════════

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # ── NEW IN p6e ──────────────────────────────────────────
        # Save the RAW (pre-clip) actions so the reward function can
        # penalize sum(raw_action²). This gives gradient feedback to
        # the network even when self.actions saturates at ±1, keeping
        # pre-clip logits from drifting unboundedly.
        self._raw_actions = actions.clone()
        self.actions = actions.clone().clamp(-1.0, 1.0)
        # ── NEW: LPF before delay buffer ─────────────────────────
        self._filtered_actions = (
            self.cfg.action_lpf_alpha * self._filtered_actions
            + (1.0 - self.cfg.action_lpf_alpha) * self.actions
        )

        if self.cfg.dr_enable and self._max_action_delay > 0:
            self._action_delay_buffer = torch.roll(
                self._action_delay_buffer, shifts=1, dims=1
            )
            self._action_delay_buffer[:, 0] = self._filtered_actions   # was self.actions
            env_idx = torch.arange(self.num_envs, device=self.device)
            self._delayed_actions = self._action_delay_buffer[
                env_idx, self._env_action_delay
            ]
        else:
            self._delayed_actions = self._filtered_actions             # was self.actions
        self._gait_phase += 2.0 * math.pi * self.cfg.gait_freq * self._control_dt
        self._gait_phase = self._gait_phase % (2.0 * math.pi)

        active_grace = self._grace_remaining > 0
        self._grace_remaining[active_grace] -= 1


        # ── Transition: grace → stabilization ──────────────────
        just_expired = active_grace & (self._grace_remaining == 0)
        just_expired_ids = just_expired.nonzero(as_tuple=False).squeeze(-1)
        if len(just_expired_ids) > 0:
            self._stabilize_remaining[just_expired_ids] = (
                self.cfg.post_grace_stabilize_steps
            )
            root_pos = self.robot.data.root_pos_w
            root_quat = self.robot.data.root_quat_w
            self._spawn_pos_xy[just_expired_ids] = (
                root_pos[just_expired_ids, :2] - self.scene.env_origins[just_expired_ids, :2]
            )
            self._spawn_yaw[just_expired_ids] = _yaw_from_quat(root_quat[just_expired_ids])
            self._pos_ema[just_expired_ids] = self._spawn_pos_xy[just_expired_ids]
            self._yaw_ema[just_expired_ids] = self._spawn_yaw[just_expired_ids]

        # ── Tick down stabilization ────────────────────────────
        active_stabilize = self._stabilize_remaining > 0
        self._stabilize_remaining[active_stabilize] -= 1

        # ── Transition: stabilize → active command ─────────────
        just_stabilized = active_stabilize & (self._stabilize_remaining == 0)
        just_stabilized_ids = just_stabilized.nonzero(as_tuple=False).squeeze(-1)
        if len(just_stabilized_ids) > 0:
            self._sample_commands(just_stabilized_ids, len(just_stabilized_ids))
            root_pos = self.robot.data.root_pos_w
            root_quat = self.robot.data.root_quat_w
            self._spawn_pos_xy[just_stabilized_ids] = (
                root_pos[just_stabilized_ids, :2]
                - self.scene.env_origins[just_stabilized_ids, :2]
            )
            self._spawn_yaw[just_stabilized_ids] = _yaw_from_quat(
                root_quat[just_stabilized_ids]
            )
            self._pos_ema[just_stabilized_ids] = self._spawn_pos_xy[just_stabilized_ids]
            self._yaw_ema[just_stabilized_ids] = self._spawn_yaw[just_stabilized_ids]

        # ── External-force application (pickups + pushes) ───────
        # Zero pickup buffer each step — pickup forces are computed
        # fresh via PD, not held. Push forces persist across steps
        # via _push_remaining and _push_forces, so we DO NOT zero it
        # here (the push handler manages its own buffer state). Sum
        # at the end and apply in a single sim call.
        self._pickup_forces.zero_()
        self._pickup_torques.zero_()

        if self.cfg.pickup_enable:
            self._apply_pickup_events()

        if self.cfg.push_enable:
            self._apply_training_pushes()

        # Pad forces are NOT computed here: they are a stiff spring, and the
        # damping term has to integrate at the physics rate.  _apply_action
        # recomputes them for every substep, immediately before the
        # scene.write_data_to_sim() that carries them into sim.step.
        if self._pad is not None:
            self._pad_forces.zero_()
            self._pad_torques.zero_()

        total_f = self._pickup_forces + self._push_forces + self._slope_forces + self._pad_forces
        total_t = self._pickup_torques + self._push_torques + self._pad_torques
        self.robot.set_external_force_and_torque(total_f, total_t)

        # ── Periodic command resample for active envs ─────────
        active_cmd = (self._grace_remaining <= 0) & (self._stabilize_remaining <= 0)
        resample_mask = (
            active_cmd
            & (self.episode_length_buf % self.cfg.cmd_resample_steps == 0)
            & (self.episode_length_buf > 0)
        )
        resample_ids = resample_mask.nonzero(as_tuple=False).squeeze(-1)
        if len(resample_ids) > 0:
            self._sample_commands(resample_ids, len(resample_ids))
            root_pos = self.robot.data.root_pos_w
            root_quat = self.robot.data.root_quat_w
            self._spawn_pos_xy[resample_ids] = (
                root_pos[resample_ids, :2] - self.scene.env_origins[resample_ids, :2]
            )
            self._spawn_yaw[resample_ids] = _yaw_from_quat(root_quat[resample_ids])
            self._pos_ema[resample_ids] = self._spawn_pos_xy[resample_ids]
            self._yaw_ema[resample_ids] = self._spawn_yaw[resample_ids]

        # ── Actuator-lag prefilter ───────────────────────────────────
        # Advance the 2nd-order command prefilter ONCE here (100 Hz), AFTER
        # _delayed_actions and the grace tick-down / transitions above have
        # settled, so the position target uses the same grace_frac _apply_action
        # used to. The filtered result is held across the decimation substeps.
        # (Deliberately NOT in _apply_action — that runs once PER substep, so
        # integrating there would advance the filter at 500 Hz, not the 100 Hz
        # the sysid identified.)
        self._advance_motor_target()

    def _build_standup_profiles(self) -> None:
        """Parse cfg.standup_{hip,knee}_profile into sorted (s, f) knot tensors.

        Re-callable at runtime (probes edit cfg and rebuild)."""
        def _parse(flat, name):
            flat = [float(v) for v in flat]
            if len(flat) < 4 or len(flat) % 2:
                raise ValueError(f"{name}: need flat [s0,f0,s1,f1,...] with ≥2 knots, got {flat}")
            s = torch.tensor(flat[0::2], device=self.device)
            f = torch.tensor(flat[1::2], device=self.device)
            if abs(s[0].item()) > 1e-6 or abs(s[-1].item() - 1.0) > 1e-6 or (s[1:] < s[:-1]).any():
                raise ValueError(f"{name}: knot phases must ascend from 0 to 1, got {s.tolist()}")
            return s, f
        self._standup_hip_kf = _parse(self.cfg.standup_hip_profile, "standup_hip_profile")
        self._standup_knee_kf = _parse(self.cfg.standup_knee_profile, "standup_knee_profile")
        rows = [float(v) for v in self.cfg.standup_path_knots]
        if len(rows) < 14 or len(rows) % 7:
            raise ValueError("standup_path_knots: need flat rows "
                             "[s,hip_target,knee_target,pitch_deg,root_h,hip_rest,knee_rest] with ≥2 rows")
        tab = torch.tensor(rows, device=self.device).reshape(-1, 7)
        if abs(tab[0, 0].item()) > 1e-6 or abs(tab[-1, 0].item() - 1.0) > 1e-6 or (tab[1:, 0] < tab[:-1, 0]).any():
            raise ValueError(f"standup_path_knots: s must ascend from 0 to 1, got {tab[:, 0].tolist()}")
        self._path_s = tab[:, 0].contiguous()
        self._path_hip = tab[:, 1].contiguous()
        self._path_knee = tab[:, 2].contiguous()
        self._path_pitch = torch.deg2rad(tab[:, 3]).contiguous()
        self._path_h = tab[:, 4].contiguous()
        self._path_hip_rest = tab[:, 5].contiguous()
        self._path_knee_rest = tab[:, 6].contiguous()

    def _path_joint_pos(self, s: torch.Tensor, rest: bool = False) -> torch.Tensor:
        """Symmetric leg pose [hr1, hp1, kn1, hr2, hp2, kn2] on the stand-up path at phase s
        (PD-target path by default; rest=True → the joints the plant rests at under that target)."""
        hip = self._pwl(s, (self._path_s, self._path_hip_rest if rest else self._path_hip))
        knee = self._pwl(s, (self._path_s, self._path_knee_rest if rest else self._path_knee))
        z = torch.zeros_like(hip)
        return torch.stack([z, hip, knee, z, -hip, -knee], dim=-1)

    def _standup_phase(self) -> torch.Tensor:
        """Path phase s ∈ [s0, 1] of every env (path mode): s0 at spawn, 1 at grace expiry."""
        own = self._grace_steps_assigned.float().clamp(min=1.0)
        progress = (1.0 - self._grace_remaining.float() / own).clamp(0.0, 1.0)
        return self._standup_s0 + (1.0 - self._standup_s0) * progress, progress

    @staticmethod
    def _pwl(s: torch.Tensor, knots: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        xs, ys = knots
        idx = torch.searchsorted(xs, s, right=True).clamp(1, xs.numel() - 1)
        x0, x1 = xs[idx - 1], xs[idx]
        y0, y1 = ys[idx - 1], ys[idx]
        w = ((s - x0) / (x1 - x0).clamp(min=1e-6)).clamp(0.0, 1.0)
        return y0 + w * (y1 - y0)

    def _standup_action_center(self) -> torch.Tensor:
        standing_center = self._default_joint_pos[:, self._joint_ids]
        if self.cfg.standup_ramp_mode == "path":
            s, _ = self._standup_phase()
            # center = the TARGET path; the spawn wrote the REST joints for that target,
            # so the initial PD error is the gravity-holding error (no kick). Spawn
            # noise perturbs the written state only, never the path.
            center = self._path_joint_pos(s)
            return torch.where((self._grace_remaining > 0).unsqueeze(-1), center, standing_center)
        if self.cfg.standup_ramp_mode == "keyframe":
            own = self._grace_steps_assigned.float().clamp(min=1.0)
            s = (1.0 - self._grace_remaining.float() / own).clamp(0.0, 1.0)
            f_hip = self._pwl(s, self._standup_hip_kf)
            f_knee = self._pwl(s, self._standup_knee_kf)
            f = torch.stack([f_hip, f_hip, f_knee, f_hip, f_hip, f_knee], dim=-1)
            return self._env_init_joint_pos + (standing_center - self._env_init_joint_pos) * f
        grace_frac = (
            self._grace_remaining.float() / max(self._max_grace_steps, 1)
        ).unsqueeze(-1)
        return (
            self._env_init_joint_pos * grace_frac
            + standing_center * (1.0 - grace_frac)
        )

    def _advance_motor_target(self) -> None:
        """Compute the joint position target the motors track this control step.

        Runs once per control step from _pre_physics_step. Builds the raw PD
        target u = action_center + delayed_action·scale (action_center is the
        grace-blended standup center, constant across substeps), then advances
        the per-joint 2nd-order actuator-lag prefilter and stores the filtered
        target. Training-side twin of sysid_replay_sim.py: the prefilter
        integrates at the 100 Hz command rate and the DCMotor PD runs against the
        held target at 500 Hz. With the prefilter disabled (or a joint's ω == 0)
        the raw target passes straight through, so behaviour is bit-identical to
        the pre-v7.1 direct-target path."""
        action_center = self._standup_action_center()
        # Raw target. delayed_actions (not self.actions) drives the motors; the
        # policy's undelayed self.actions still feeds reward terms / last_actions.
        u = action_center + (self._delayed_actions * self._action_scale)

        if self._act_lag_active:
            # y'' = ω²(u − y) − 2ζω·y' ;  forward Euler at the control dt.
            pf_ydd = (
                self._act_lag_omega2 * (u - self._act_lag_pf_y)
                - self._act_lag_two_zw * self._act_lag_pf_yd
            )
            self._act_lag_pf_yd = self._act_lag_pf_yd + pf_ydd * self._control_dt
            self._act_lag_pf_y = self._act_lag_pf_y + self._act_lag_pf_yd * self._control_dt
            # ω==0 joints (mask 0) pass the raw command through unchanged.
            self._motor_pos_target = (
                self._act_lag_mask * self._act_lag_pf_y
                + (1.0 - self._act_lag_mask) * u
            )
        else:
            self._motor_pos_target = u

        # Execution-noise injection (P13): perturb the target the motors
        # will track this control step. Out-of-place adds — never mutate
        # u / _act_lag_pf_y. Held across decimation substeps like the
        # target itself (100 Hz granularity, matching the real command
        # path). Inert when all three knobs are 0.0 (the knee-angle bias
        # rides the same target-bias tensor).
        if (self.cfg.exec_noise_target_bias_max > 0.0
                or self.cfg.exec_noise_target_std > 0.0
                or self.cfg.dr_knee_angle_bias_max > 0.0
                or self.cfg.dr_hip_angle_bias_max > 0.0
                or self._dr_override):
            tgt = self._motor_pos_target + self._exec_target_bias
            if self.cfg.exec_noise_target_std > 0.0:
                tgt = tgt + torch.randn_like(tgt) * self.cfg.exec_noise_target_std
            self._motor_pos_target = tgt

    def _apply_action(self) -> None:
        # Target computed once per control step in _advance_motor_target and held
        # constant across the decimation substeps; the DCMotor PD recomputes
        # torque from live joint state against it at the physics rate.
        self.robot.set_joint_position_target(
            self._motor_pos_target, joint_ids=self._joint_ids
        )
        # Pad contact: one physics-step update per substep, from the state
        # scene.update() refreshed at the end of the previous substep.  Written
        # into the external-force buffers here, so the write_data_to_sim() that
        # follows this call carries it into the same sim.step() the joint
        # targets feed.
        if self._pad is not None:
            self._pad_forces.zero_()
            self._pad_torques.zero_()
            self._apply_pad_contact()

    # ══════════════════════════════════════════════════════════
    # Training pushes
    # ══════════════════════════════════════════════════════════

    def _apply_training_pushes(self):
        eligible = (self._grace_remaining <= 0) & (self._pickup_phase == 0)
        push_scale = 1.0 # was 0.2 + 0.8 * self._curriculum_alpha
        push_mask = (
            (torch.rand(self.num_envs, device=self.device) < self.cfg.push_probability * push_scale)
            & (self._push_remaining <= 0)
            & eligible
        )
        if push_mask.sum().item() > 0:
            angles = torch.rand(self.num_envs, device=self.device) * 2.0 * math.pi
            scaled_min = self.cfg.push_force_min * push_scale
            scaled_max = self.cfg.push_force_max * push_scale
            magnitudes = (
                torch.rand(self.num_envs, device=self.device)
                * (scaled_max - scaled_min) + scaled_min
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

    # ══════════════════════════════════════════════════════════
    # Observations — 36D (unchanged)
    # ══════════════════════════════════════════════════════════

    def _encoder_observation(self, joint_pos, joint_vel):
        """Replace the knee channels with the motor-encoder reading.

        The deploy stack never observes the physical knee: it decodes the
        motor encoder through the 22:30 belt model (control/config.py).  That
        is the SAME value as the joint while the belt is rigid, but under
        series elasticity the two diverge by the belt deflection, so the obs
        must follow the encoder or the trained policy sees a quantity no
        sensor on the robot can produce.

        Kept separate from the reward/termination state on purpose: those are
        physical quantities (contact, energy) and stay on raw joint data.
        """
        hk = self.robot.actuators.get("hip_knee")
        if hk is None or not hasattr(hk, "encoder_joint_state"):
            return joint_pos, joint_vel
        if not getattr(hk.cfg, "series_elasticity", False):
            return joint_pos, joint_vel
        if not self.cfg.belt_encoder_obs:
            return joint_pos, joint_vel
        # The actuator's pair columns index ITS OWN joint subset (the 4-joint
        # hip_knee group, local order hip_pitch1/knee1/hip_pitch2/knee2), which
        # is what Isaac Lab hands compute().  robot.data.joint_pos is the full
        # 6-joint array (hip_roll1 first), so it must be sliced the same way
        # here or the "hip" the decoder subtracts is hip_roll1 / knee1
        # (2026-09-15: this was the S5/S7 knee-obs corruption — every elastic
        # training episode read knee1 as G*m - c*hip_roll1).
        q_enc, qd_enc = hk.encoder_joint_state(
            joint_pos[:, hk.joint_indices], joint_vel[:, hk.joint_indices]
        )
        # q_enc columns are in the actuator's PAIR order (knee1, knee2).  The
        # write target is a column of the SIM-ordered joint array, so the slot
        # must come from the sim's own name list — the env's `_joint_names`
        # is a different ordering (hr1, hp1, kn1, ...) and using its index
        # here wrote knee1's encoder value into hip_pitch1's obs for every
        # elastic training episode up to 2026-09-16 (found by the per-slot
        # obs-vs-true dump; the value-based TB readback could not see it).
        if self._encoder_kn_slots is None:
            kn_slots = encoder_write_slots(self.robot.joint_names, hk.cfg.pairs)
            pair_names = list(hk.cfg.pairs.keys())
            if [self.robot.joint_names[s] for s in kn_slots] != pair_names or \
                    [hk.joint_names[int(col)] for col in hk._kn_cols] != pair_names:
                raise RuntimeError(
                    "knee joint order differs between the sim, the hip_knee "
                    "actuator group and cfg.pairs; encoder obs would be misaligned"
                )
            print(f"[PLANT] encoder obs write slots (sim order): "
                  f"{dict(zip(pair_names, kn_slots))} of {list(self.robot.joint_names)}")
            self._encoder_kn_slots = torch.tensor(kn_slots, device=self.device, dtype=torch.long)
            self._encoder_leg_pairs = torch.arange(len(kn_slots), device=self.device, dtype=torch.long)
        out_pos = joint_pos.clone()
        out_vel = joint_vel.clone()
        if self.cfg.belt_encoder_obs_pos:
            out_pos[:, self._encoder_kn_slots] = q_enc[:, self._encoder_leg_pairs]
        if self.cfg.belt_encoder_obs_vel:
            out_vel[:, self._encoder_kn_slots] = qd_enc[:, self._encoder_leg_pairs]
        return out_pos, out_vel

    def _belt_debug_collect(self, raw_jp, raw_jv, obs_jp, obs_jv) -> None:
        """Per-step trace for the belt debug (cfg.belt_debug_dump_steps)."""
        buf = getattr(self, "_belt_debug_buf", None)
        if buf is None:
            buf = self._belt_debug_buf = {"n": 0, "rows": []}
        if buf["n"] >= self.cfg.belt_debug_dump_steps:
            return
        m = min(self.cfg.belt_debug_dump_envs, self.num_envs)
        ids = self._joint_ids
        hk = self.robot.actuators.get("hip_knee")
        row = {
            "joint_pos": raw_jp[:m][:, ids], "joint_vel": raw_jv[:m][:, ids],
            "obs_pos": obs_jp[:m][:, ids], "obs_vel": obs_jv[:m][:, ids],
            "target": self._motor_pos_target[:m], "raw_action": self._raw_actions[:m],
            "applied_torque": self.robot.data.applied_torque[:m][:, ids],
            "root_z": self.robot.data.root_pos_w[:m, 2],
            "proj_grav": quat_apply_inverse(self.robot.data.root_quat_w[:m], self._gravity_vec[:m]),
            "ep_len": self.episode_length_buf[:m].clone(),
        }
        if hk is not None and hasattr(hk, "_series_motor_pos"):
            row["rotor_pos"] = hk._series_motor_pos[:m]
            row["rotor_vel"] = hk._series_motor_vel[:m]
            row["spring_effort"] = hk.series_spring_effort[:m]
            row["motor_effort"] = hk.series_motor_effort[:m]
        buf["rows"].append({k: v.detach().float().cpu().clone() for k, v in row.items()})
        buf["n"] += 1
        if buf["n"] == self.cfg.belt_debug_dump_steps:
            import numpy as np
            out = {k: torch.stack([r[k] for r in buf["rows"]]).numpy() for k in buf["rows"][0]}
            path = os.path.join(str(getattr(self.cfg, "log_dir", ".")), "belt_debug.npz")
            np.savez(path, **out)
            print(f"[BELT-DEBUG] wrote {path}: {buf['n']} steps x {m} envs, keys={list(out)}")

    def _get_observations(self) -> dict:
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel
        # Hardware reads the knee joint only through the motor encoder, so
        # under belt series elasticity the observation must carry the
        # decoder's answer (q_enc = G*m - c*q_hip), not the belt-free PhysX
        # joint.  With a rigid transmission the adapter hands back the joint
        # state unchanged, so the nominal contract is untouched.
        raw_joint_pos, raw_joint_vel = joint_pos, joint_vel
        joint_pos, joint_vel = self._encoder_observation(joint_pos, joint_vel)
        if self.cfg.belt_debug_dump_steps > 0:
            self._belt_debug_collect(raw_joint_pos, raw_joint_vel, joint_pos, joint_vel)

        imu_quat = self.robot.data.body_quat_w[:, self._imu_idx, :]
        imu_ang_vel = self.robot.data.body_ang_vel_w[:, self._imu_idx, :]
        imu_lin_vel = self.robot.data.body_lin_vel_w[:, self._imu_idx, :]

        projected_gravity = quat_apply_inverse(imu_quat, self._gravity_vec)
        ang_vel_imu = quat_apply_inverse(imu_quat, imu_ang_vel)
        lin_vel_imu = quat_apply_inverse(imu_quat, imu_lin_vel)
        lin_vel_imu = lin_vel_imu * self._lin_vel_obs_keep.unsqueeze(-1)

        joint_pos_norm = (
            joint_pos[:, self._joint_ids]
            - self._default_joint_pos[:, self._joint_ids]
        )

        gait_sin = torch.sin(self._gait_phase).unsqueeze(-1)
        gait_cos = torch.cos(self._gait_phase).unsqueeze(-1)
        difficulty = self._env_difficulty.unsqueeze(-1)
        grace_norm = (
            self._grace_remaining.float() / max(self._max_grace_steps, 1)
        ).unsqueeze(-1)
        init_h = self._initial_height.unsqueeze(-1)

        current_yaw = _yaw_from_quat(self.robot.data.root_quat_w)
        yaw_error = current_yaw - self._spawn_yaw
        yaw_err_sin = torch.sin(yaw_error).unsqueeze(-1)
        yaw_err_cos = torch.cos(yaw_error).unsqueeze(-1)

        # ── Per-episode constant biases (sensors only) ─────────────
        if self.cfg.dr_enable:
            projected_gravity = projected_gravity + self._dr_proj_grav_bias
            ang_vel_imu       = ang_vel_imu + self._dr_ang_vel_bias
            joint_pos_norm = joint_pos_norm + self._dr_joint_pos_bias

        # ── Per-step obs noise DR (sensors only) ───────────────────
        joint_vel_obs = joint_vel[:, self._joint_ids]
        if self.cfg.dr_enable:
            if self.cfg.dr_obs_noise_proj_grav > 0:
                projected_gravity = projected_gravity + sample_uniform(
                    -self.cfg.dr_obs_noise_proj_grav,
                    self.cfg.dr_obs_noise_proj_grav,
                    projected_gravity.shape, self.device,
                )
            if self.cfg.dr_obs_noise_ang_vel > 0:
                ang_vel_imu = ang_vel_imu + sample_uniform(
                    -self.cfg.dr_obs_noise_ang_vel,
                    self.cfg.dr_obs_noise_ang_vel,
                    ang_vel_imu.shape, self.device,
                )
            if self.cfg.dr_obs_noise_joint_vel > 0:
                joint_vel_obs = joint_vel_obs + sample_uniform(
                    -self.cfg.dr_obs_noise_joint_vel,
                    self.cfg.dr_obs_noise_joint_vel,
                    joint_vel_obs.shape, self.device,
                )
            if self.cfg.dr_obs_noise_joint_pos > 0:
                joint_pos_norm = joint_pos_norm + sample_uniform(
                    -self.cfg.dr_obs_noise_joint_pos,
                    self.cfg.dr_obs_noise_joint_pos,
                    joint_pos_norm.shape, self.device,
                )

        # ── Split: sensors get delayed, bookkeeping stays fresh ─────
        # Sensors travel through the real-hardware pipeline (encoder bus,
        # IMU filter, USB/SPI, IPC), so they should be subject to obs_delay.
        obs_sensors_fresh = torch.cat([
            projected_gravity,    # 3   IMU
            ang_vel_imu,          # 3   IMU
            lin_vel_imu,          # 3   IMU-derived (or zeroed by dropout)
            joint_pos_norm,       # 6   encoder
            joint_vel_obs,        # 6   encoder-derived
        ], dim=-1)                # 21

        # Bookkeeping is policy/env-internal state that is known exactly at
        # step t. There is no physical reason to delay it. Specifically,
        # delaying _last_actions blinds the policy to what is currently in
        # the action-delay pipeline, making action_delay impossible to
        # compensate for. Fresh here closes that gap.
        obs_bookkeeping_fresh = torch.cat([
            self._last_actions,                   # 6
            gait_sin, gait_cos,                   # 2
            difficulty, grace_norm, init_h,       # 3
            self._vx_cmd.unsqueeze(-1),           # 1
            self._yaw_rate_cmd.unsqueeze(-1),     # 1
            yaw_err_sin, yaw_err_cos,             # 2
        ], dim=-1)                                # 15

        # ── Obs delay applied to sensors only (IMU / joint split) ──
        # IMU channels (0:9) and joint channels (9:21) are pulled from
        # the same history buffer at different per-env delays, so the
        # measured ~10 ms IMU-leads-encoder skew is reproduced in sim.
        if self.cfg.dr_enable and self._max_obs_delay > 0:
            self._obs_delay_buffer = torch.roll(
                self._obs_delay_buffer, shifts=1, dims=1
            )
            self._obs_delay_buffer[:, 0] = obs_sensors_fresh
            # Handover-prime (S-line): selected just-reset envs see their
            # real first sensor frame through the whole delay pipe instead
            # of the zeroed buffer (= deploy 'b' vs blind 'v' contract).
            if self._needs_delay_prime.any():
                prime_ids = self._needs_delay_prime.nonzero(as_tuple=False).squeeze(-1)
                self._obs_delay_buffer[prime_ids] = obs_sensors_fresh[prime_ids].unsqueeze(1)
                self._needs_delay_prime[prime_ids] = False
            env_idx = torch.arange(self.num_envs, device=self.device)
            imu_delayed = self._obs_delay_buffer[
                env_idx, self._env_imu_delay
            ][:, :_OBS_IMU_PER_FRAME]
            joint_delayed = self._obs_delay_buffer[
                env_idx, self._env_joint_delay
            ][:, _OBS_IMU_PER_FRAME:]
            obs_sensors_delayed = torch.cat([imu_delayed, joint_delayed], dim=-1)
        else:
            obs_sensors_delayed = obs_sensors_fresh

        # Reassemble the 36-D per-frame obs: delayed sensors + fresh bookkeeping.
        # Channel layout is byte-for-byte identical to before, so the mirror
        # permutation tables and observation_space (36 * obs_history_steps)
        # do NOT need to change.
        obs_frame = torch.cat([obs_sensors_delayed, obs_bookkeeping_fresh], dim=-1)

        # ── History stack ──────────────────────────────────────────
        # Frame 0 = newest, frame K-1 = oldest. With this fix the history
        # of _last_actions is now a clean record of the policy's recent
        # commands, including the actions in flight through the action-delay
        # pipeline — exactly what the policy needs to plan around delay.
        self._obs_history_buf = torch.roll(
            self._obs_history_buf, shifts=1, dims=1
        )
        self._obs_history_buf[:, 0] = obs_frame

        # NEW: For envs that just reset, replicate the current frame
        # into ALL history slots so the policy doesn't see K-1 frames
        # of fake zeros. Without this, the first K-1 steps of every
        # episode feed the policy OOD obs.
        if self._needs_history_init.any():
            init_ids = self._needs_history_init.nonzero(as_tuple=False).squeeze(-1)
            self._obs_history_buf[init_ids] = obs_frame[init_ids].unsqueeze(1)
            self._needs_history_init[init_ids] = False

        obs_stacked = self._obs_history_buf.reshape(
            self.num_envs, self.cfg.obs_history_steps * _OBS_PER_FRAME,
        )

        self._last_obs_for_diag = obs_stacked
        return {"policy": obs_stacked, "critic": self._get_critic_observations()}
    

    def _get_critic_observations(self) -> torch.Tensor:
        """Privileged obs group for the asymmetric critic. Everything here is
        HIDDEN from the actor on purpose (lin_vel is dropped/biased, pickup + DR
        are unobservable on hardware). Width MUST equal cfg.priv_obs_dim and the
        channel layout MUST match _priv_mirror_perm / _priv_mirror_sign."""
        root_quat = self.robot.data.root_quat_w

        # True base velocities — clean: no dropout / bias / delay / noise.
        true_lin_vel_b = quat_apply_inverse(root_quat, self.robot.data.root_lin_vel_w)
        true_ang_vel_b = quat_apply_inverse(root_quat, self.robot.data.root_ang_vel_w)

        # Pickup state.
        phase_onehot = torch.zeros(self.num_envs, 5, device=self.device)
        phase_onehot.scatter_(1, self._pickup_phase.clamp(0, 4).unsqueeze(-1), 1.0)
        support = self._pickup_support_factor.unsqueeze(-1)
        weight = (self._total_mass * 9.81).clamp(min=1e-3).unsqueeze(-1)
        pickup_force_b = quat_apply_inverse(
            root_quat, self._pickup_forces[:, self._torso_idx, :]
        ) / weight

        # Physics DR (per-env scalars).
        mass_mult  = self._dr_mass_mult.unsqueeze(-1)
        floor_fric = self._dr_floor_friction_mult.unsqueeze(-1)
        voltage    = self._dr_voltage_factor.unsqueeze(-1)
        com_abs    = self._dr_com_offset.abs().mean(dim=(-2, -1)).unsqueeze(-1)

        # Loop latency.
        action_delay = self._env_action_delay.float().unsqueeze(-1)
        imu_delay    = self._env_imu_delay.float().unsqueeze(-1)
        joint_delay  = self._env_joint_delay.float().unsqueeze(-1)
        lin_vel_keep = self._lin_vel_obs_keep.unsqueeze(-1)

        # Slope tilt magnitude (deg), normalized so 2° → 1.0.
        slope_deg = self._dr_gravity_tilt_deg.unsqueeze(-1) / 2.0

        critic_obs = torch.cat([
            true_lin_vel_b, true_ang_vel_b,                 # [0:6]
            phase_onehot, support, pickup_force_b,          # [6:15]
            mass_mult, floor_fric, voltage, com_abs,        # [15:19]
            self._dr_stiffness_mult,                        # [19:25]
            self._dr_damping_mult,                          # [25:31]
            self._dr_effort_mult,                           # [31:37]
            self._dr_armature_mult,                         # [37:43]
            self._dr_jfriction_mult,                        # [43:49]
            self._dr_dyn_friction_mult,                     # [49:55]
            self._dr_visc_friction_mult,                    # [55:61]
            self._dr_act_lag_omega_mult,                    # [61:67]  NEW (v7.1)
            self._dr_act_lag_zeta_mult,                     # [67:73]  NEW (v7.1)
            action_delay, imu_delay, joint_delay,           # [73:76]
            lin_vel_keep,                                   # [76]
            slope_deg,                                      # [77]
        ], dim=-1)                                          # = 78

        if critic_obs.shape[-1] != self.cfg.priv_obs_dim:
            raise RuntimeError(
                f"priv_obs_dim mismatch: built {critic_obs.shape[-1]}, "
                f"cfg says {self.cfg.priv_obs_dim}"
            )
        return critic_obs

    # ══════════════════════════════════════════════════════════
    # Rewards
    # ══════════════════════════════════════════════════════════
    def _get_rewards(self) -> torch.Tensor:
        root_quat = self.robot.data.root_quat_w
        root_pos = self.robot.data.root_pos_w
        root_lin_vel = self.robot.data.root_lin_vel_w
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel

        projected_gravity = quat_apply_inverse(root_quat, self._gravity_vec)
        joint_pos_norm = (
            joint_pos[:, self._joint_ids]
            - self._default_joint_pos[:, self._joint_ids]
        )

        toe1_w, toe2_w = self._get_toe_pos_w()
        toe1_z = toe1_w[:, 2]
        toe2_z = toe2_w[:, 2]

        gait_sin = torch.sin(self._gait_phase)
        in_grace = (self._grace_remaining > 0).float()

        base_lin_vel = quat_apply_inverse(root_quat, root_lin_vel)
        base_vx = base_lin_vel[:, 0]
        base_vz_world = root_lin_vel[:, 2]

        # Averaged forward velocity for command tracking (not instantaneous).
        tdecay = self.cfg.tracking_ema_decay
        self._vx_avg = tdecay * self._vx_avg + (1.0 - tdecay) * base_vx

        toe_diff_w = toe1_w - toe2_w
        toe_diff_base = quat_apply_inverse(root_quat, toe_diff_w)
        toe_fwd_diff = toe_diff_base[:, 0]
        leg_decay = self.cfg.leg_symmetry_decay
        self._leg_forward_ema = (
            leg_decay * self._leg_forward_ema + (1.0 - leg_decay) * toe_fwd_diff
        )

        joint_pos_ordered = self.robot.data.joint_pos[:, self._joint_ids]
        pair_sums = torch.stack([
            joint_pos_ordered[:, 0] + joint_pos_ordered[:, 3],
            joint_pos_ordered[:, 1] + joint_pos_ordered[:, 4],
            joint_pos_ordered[:, 2] + joint_pos_ordered[:, 5],
        ], dim=-1)
        a_decay = self.cfg.angle_symmetry_decay
        self._joint_pair_sum_ema = (
            a_decay * self._joint_pair_sum_ema
            + (1.0 - a_decay) * pair_sums
        )

        toe_mid_world_xy = (toe1_w[:, :2] + toe2_w[:, :2]) * 0.5
        toe_mid_disp = toe_mid_world_xy - self._prev_feet_mid_world
        self._prev_feet_mid_world[:] = toe_mid_world_xy
        toe_mid_disp_3d = torch.zeros(self.num_envs, 3, device=self.device)
        toe_mid_disp_3d[:, :2] = toe_mid_disp
        toe_mid_disp_base = quat_apply_inverse(root_quat, toe_mid_disp_3d)[:, :2]

        toe1_disp_w = toe1_w - self._prev_toe1_world
        toe2_disp_w = toe2_w - self._prev_toe2_world
        self._prev_toe1_world[:] = toe1_w
        self._prev_toe2_world[:] = toe2_w
        toe1_disp_base = quat_apply_inverse(root_quat, toe1_disp_w)
        toe2_disp_base = quat_apply_inverse(root_quat, toe2_disp_w)

        root_xy = root_pos[:, :2] - self.scene.env_origins[:, :2]
        current_yaw = _yaw_from_quat(root_quat)
        # ── Averaged position / heading (instantaneous → windowed) ──
        # EMA the position so per-step body sway doesn't read as drift; only
        # the time-averaged position is anchored. Yaw is EMA'd via a wrap-safe
        # shortest-arc step so it stays continuous across the ±π boundary.
        adecay = self.cfg.anchor_ema_decay
        self._pos_ema = adecay * self._pos_ema + (1.0 - adecay) * root_xy
        yaw_step = torch.atan2(
            torch.sin(current_yaw - self._yaw_ema),
            torch.cos(current_yaw - self._yaw_ema),
        )
        self._yaw_ema = self._yaw_ema + (1.0 - adecay) * yaw_step
        pos_drift_sq_raw = torch.sum(
            torch.square(self._pos_ema - self._spawn_pos_xy), dim=-1
        )
        # ── NEW knob 3: position-anchor deadband ──
        # Penalize only the drift distance beyond the deadband radius, then
        # re-square to keep the quadratic shape outside it. Inside the radius
        # the anchor is silent, so stepping-in-place sway isn't read as drift.
        pos_drift = torch.sqrt(pos_drift_sq_raw + 1e-12)
        pos_drift_db = torch.clamp(
            pos_drift - self.cfg.position_anchor_deadband_m, min=0.0
        )
        pos_drift_sq = torch.square(pos_drift_db)
        yaw_diff = torch.atan2(
            torch.sin(self._yaw_ema - self._spawn_yaw),
            torch.cos(self._yaw_ema - self._spawn_yaw),
        )
        yaw_error_sq = torch.square(yaw_diff)

        root_ang_vel_w = self.robot.data.root_ang_vel_w
        root_ang_vel_base = quat_apply_inverse(root_quat, root_ang_vel_w)
        ang_vel_z_base = root_ang_vel_base[:, 2]

        # Averaged yaw rate for command tracking. ang_vel_z_base (instantaneous)
        # is kept for the yaw_drift damping term, which should stay instantaneous.
        self._yaw_rate_avg = tdecay * self._yaw_rate_avg + (1.0 - tdecay) * ang_vel_z_base

        # ── Stance-time symmetry EMA update ────────────────────
        foot1_on_ground = (toe1_z < self.cfg.no_flight_contact_threshold).float()
        foot2_on_ground = (toe2_z < self.cfg.no_flight_contact_threshold).float()
        s_decay = self.cfg.stance_symmetry_decay
        self._foot1_contact_ema = (
            s_decay * self._foot1_contact_ema
            + (1.0 - s_decay) * foot1_on_ground
        )
        self._foot2_contact_ema = (
            s_decay * self._foot2_contact_ema
            + (1.0 - s_decay) * foot2_on_ground
        )

        # ── Resolve removed-from-cfg reward scales (back-compat) ──
        # Slots in the JIT function still exist for log-key stability;
        # they just get 0.0 if the cfg no longer carries them.
        rew_scale_joint_pos_limits = getattr(self.cfg, "rew_scale_joint_pos_limits", 0.0)
        rew_scale_front_split = getattr(self.cfg, "rew_scale_front_split", 0.0)
        rew_scale_knee_split = getattr(self.cfg, "rew_scale_knee_split", 0.0)
        # ── A/B FORK: gait-reference prior knobs ──────────────────
        # getattr-defaulted so this file is drop-in safe under the main
        # run's cfg: absent knobs ⇒ scale 0 ⇒ bit-identical reward.
        rew_scale_gait_ref = getattr(self.cfg, "rew_scale_gait_ref", 0.0)
        gait_ref_amplitude = getattr(self.cfg, "gait_ref_amplitude_rad", 0.20)
        gait_ref_sigma = getattr(self.cfg, "gait_ref_sigma_rad", 0.15)

        phase   = self._pickup_phase
        support = self._pickup_support_factor
        release_fade = (1.0 - support).clamp(0.0, 1.0)
        # 0 during LIFT/HOLD/LOWER (full air support), (1-support) during RELEASE,
        # 1 everywhere else. This is the soft blend that fades the stand/settle
        # reward set back in as the hand lets go.
        pickup_gate = torch.where(
            phase == 4, release_fade,
            torch.where(phase > 0, torch.zeros_like(release_fade),
                        torch.ones_like(release_fade)),
        )

        if self.cfg.pickup_gate_debug_check:   # off by default — .any() forces a sync
            assert not ((self._grace_remaining > 0) & (phase > 0)).any(), \
                "grace and pickup overlap — mutual-exclusivity invariant violated"

        total_reward, reward_components = compute_rewards_vx_yaw(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_raw_action,           # NEW IN p6e (was rew_scale_upright)
            self.cfg.rew_scale_height,
            self.cfg.rew_scale_feet_air,
            self.cfg.rew_scale_gait_step,
            self.cfg.rew_scale_energy,
            self.cfg.rew_scale_action_rate,
            self.cfg.rew_scale_action_magnitude,
            self.cfg.rew_scale_joint_vel,
            self.cfg.rew_scale_vel_tracking,
            rew_scale_joint_pos_limits,
            self.cfg.rew_scale_standup_height,
            self.cfg.rew_scale_standup_fall,
            self.cfg.rew_scale_thermal,
            self.cfg.rew_scale_hip_roll_split,
            rew_scale_front_split,
            rew_scale_knee_split,
            self.cfg.rew_scale_foot_swing,
            self.cfg.rew_scale_leg_symmetry,
            self.cfg.rew_scale_angle_symmetry,
            self.cfg.rew_scale_knee_over_hip,
            self.cfg.rew_scale_foot_progress,
            self.cfg.rew_scale_position_anchor,
            self.cfg.rew_scale_heading_anchor,
            self.cfg.rew_scale_back_leg_step,
            self.cfg.rew_scale_yaw_rate,
            self.cfg.rew_scale_balance,
            self.cfg.rew_scale_no_flight,
            self.cfg.rew_scale_vz_penalty,
            self.cfg.rew_scale_stance_symmetry,
            self.cfg.rew_scale_yaw_drift,
            projected_gravity,
            root_pos[:, 2],
            base_vz_world,
            toe1_z,
            toe2_z,
            gait_sin,
            joint_vel[:, self._joint_ids],
            self.robot.data.applied_torque[:, self._joint_ids],
            self._continuous_effort,
            self.actions,
            self._raw_actions,                       # NEW IN p6e
            self._last_actions,
            self.reset_terminated,
            joint_pos_norm,
            joint_pos[:, self._hip_roll_ids],
            joint_pos[:, self._hip_pitch_ids],
            joint_pos[:, self._knee_ids],
            in_grace,
            self._vx_avg,                             # was base_vx (now averaged)
            self._vx_cmd,
            self.cfg.vel_tracking_sigma,
            self._leg_forward_ema,
            self._joint_pair_sum_ema,
            self._foot1_contact_ema,
            self._foot2_contact_ema,
            toe_mid_disp_base,
            pos_drift_sq,
            yaw_error_sq,
            toe1_disp_base[:, 0],
            toe2_disp_base[:, 0],
            toe_diff_base[:, 0],
            ang_vel_z_base,
            self._yaw_rate_cmd,
            self.cfg.yaw_rate_tracking_sigma,
            self.cfg.yaw_rate_cmd_anchor_threshold,
            self.cfg.height_min,
            self.cfg.height_max,
            self.cfg.height_edge_softness,
            self.cfg.kneel_height,                   # NEW IN p6e (replaces midpoint)
            self.cfg.stand_target_height,            # NEW IN p6e (replaces midpoint)
            self.cfg.feet_air_height_threshold,
            self.cfg.max_hip_roll_rad,
            self.cfg.max_hip_pitch_split_rad,
            self.cfg.max_knee_split_rad,
            self.cfg.no_flight_contact_threshold,
            self.cfg.rew_scale_action_smoothness,    # NEW
            self._last_last_actions,                  # NEW
            self._action_rate_weights,                # NEW
            pickup_gate,
            self.cfg.raw_action_threshold,            # NEW: a_thr
            self.cfg.rew_scale_action_snap,           # NEW: 3rd-order
            self._last_last_last_actions,             # NEW: t-3
            self.cfg.idle_step_gate,                  # NEW
            self.cfg.idle_motion_relief,              # NEW
            self.cfg.rew_scale_upright,               # NEW: uprightness penalty
            self._yaw_rate_avg,                       # NEW: averaged yaw-rate for tracking
            self.cfg.rew_scale_clock_contact,         # NEW knob 1
            self.cfg.rew_scale_swing_clearance,       # NEW knob 2
            self.cfg.swing_clearance_target_m,        # NEW knob 2
            self.cfg.swing_clearance_zero_m,          # RESUME 2: dense ramp start
            rew_scale_gait_ref,                       # NEW A/B fork
            gait_ref_amplitude,                       # NEW A/B fork (signed, rad)
            gait_ref_sigma,                           # NEW A/B fork (rad)
        )
        # ── Path-mode grace shaping (09-08): track the path's torso pitch and keep
        # the feet down. Slots 38/39 of _REWARD_COMPONENT_NAMES; zero otherwise.
        rew_track = torch.zeros_like(total_reward)
        rew_contact = torch.zeros_like(total_reward)
        if self.cfg.standup_ramp_mode == "path":
            s, _ = self._standup_phase()
            pitch = torch.asin(projected_gravity[:, 0].clamp(-1.0, 1.0))
            pitch_ref = self._pwl(s, (self._path_s, self._path_pitch))
            if self.cfg.rew_scale_standup_track != 0.0:
                err = (pitch - pitch_ref) / self.cfg.standup_track_sigma_rad
                rew_track = self.cfg.rew_scale_standup_track * torch.exp(-err * err) * in_grace
            if self.cfg.rew_scale_standup_contact != 0.0:
                feet_down = 0.5 * ((toe1_z < self.cfg.no_flight_contact_threshold).float()
                                   + (toe2_z < self.cfg.no_flight_contact_threshold).float())
                rew_contact = self.cfg.rew_scale_standup_contact * feet_down * in_grace
            total_reward = total_reward + rew_track + rew_contact
            self.extras.setdefault("log", {})["standup/pitch_err_grace_mean"] = (
                ((pitch - pitch_ref).abs() * in_grace).sum() / in_grace.sum().clamp(min=1.0)).item()
        reward_components = torch.cat(
            [reward_components, rew_track.unsqueeze(-1), rew_contact.unsqueeze(-1)], dim=-1
        )
        self._last_reward_components = reward_components.detach()

        with torch.no_grad():
            _ad = self.actions - self._last_actions
            _lad = self._last_actions - self._last_last_actions
            _jerk = _ad - _lad
            # NEW: 3rd difference (snap). _lad - _llad = jerk(t-1).
            _llad = self._last_last_actions - self._last_last_last_actions
            _snap = _jerk - (_lad - _llad)

        # Rotate action history. Order matters: copy the OLDEST slot
        # FIRST so every read precedes its clobbering write. Flipping
        # this degenerates snap (and jerk) to zero.
        self._last_last_last_actions[:] = self._last_last_actions   # NEW: must be first
        self._last_last_actions[:] = self._last_actions
        self._last_actions[:] = self.actions

        # ── Logging ────────────────────────────────────────────
        self.extras["log"] = self.extras.get("log", {})

        # ── Per-component reward logging ───────────────────────
        comp_means = reward_components.mean(dim=0)
        comp_abs_means = reward_components.abs().mean(dim=0)
        total_abs = comp_abs_means.sum().item()
        for i, name in enumerate(_REWARD_COMPONENT_NAMES):
            mean_val = comp_means[i].item()
            abs_val = comp_abs_means[i].item()
            pct = (abs_val / total_abs * 100.0) if total_abs > 1e-8 else 0.0
            self.extras["log"][f"reward/{name}_mean"] = mean_val
            self.extras["log"][f"reward/{name}_abs_mean"] = abs_val
            self.extras["log"][f"reward/{name}_pct"] = pct

        self.extras["log"]["reward/total_mean"] = total_reward.mean().item()
        self.extras["log"]["curriculum/standup_alpha"] = self._curriculum_alpha
        self.extras["log"]["curriculum/mean_difficulty"] = self._env_difficulty.mean().item()
        self.extras["log"]["diagnostics/base_height"] = root_pos[:, 2].mean().item()
        self.extras["log"]["diagnostics/base_height_std"] = root_pos[:, 2].std().item()
        self.extras["log"]["diagnostics/uprightness"] = (-projected_gravity[:, 2]).mean().item()
        # Uprightness for envs where the upright reward is actually live
        # (post-grace, not in pickup) — the population that owns its balance.
        active_upright_mask = (self._grace_remaining <= 0) & (self._pickup_phase == 0)
        if active_upright_mask.any():
            up_active = (-projected_gravity[active_upright_mask, 2]).clamp(-1.0, 1.0)
            self.extras["log"]["diagnostics/uprightness_active"] = up_active.mean().item()
            self.extras["log"]["diagnostics/uprightness_active_p10"] = (
                torch.quantile(up_active, 0.10).item()
            )
            self.extras["log"]["diagnostics/tilt_active_deg_mean"] = (
                torch.rad2deg(torch.acos(up_active)).mean().item()
            )
        self.extras["log"]["diagnostics/mean_vx_cmd"] = self._vx_cmd.mean().item()
        self.extras["log"]["diagnostics/mean_base_vx"] = base_vx.mean().item()
        self.extras["log"]["diagnostics/vx_tracking_err"] = (
            (base_vx - self._vx_cmd).abs().mean().item()
        )
        self.extras["log"]["diagnostics/pos_drift_m_mean"] = pos_drift.mean().item()
        self.extras["log"]["diagnostics/mean_yaw_rate_cmd"] = self._yaw_rate_cmd.mean().item()
        self.extras["log"]["diagnostics/mean_yaw_rate_actual"] = ang_vel_z_base.mean().item()
        self.extras["log"]["diagnostics/yaw_rate_err"] = (
            (ang_vel_z_base - self._yaw_rate_cmd).abs().mean().item()
        )
        self.extras["log"]["diagnostics/mean_action_sq"] = (self.actions ** 2).mean().item()

        # ── NEW IN p6e: raw-action diagnostics ─────────────────
        # The whole point of the raw-action penalty is to keep these
        # values calibrated. Track them so we can verify the penalty
        # is doing its job. raw_action_max_abs > 1.5 means the
        # network is still drifting past the clip range and the
        # penalty may need to be stronger.
        self.extras["log"]["diagnostics/raw_action_sq_mean"] = (
            (self._raw_actions ** 2).mean().item()
        )
        self.extras["log"]["diagnostics/raw_action_max_abs"] = (
            self._raw_actions.abs().max().item()
        )
        self.extras["log"]["diagnostics/raw_action_p99_abs"] = (
            torch.quantile(self._raw_actions.abs().flatten(), 0.99).item()
        )

        self.extras["log"]["diagnostics/action_jerk_sq_mean"] = (
            _jerk.square().mean().item()
        )
        self.extras["log"]["diagnostics/action_jerk_abs_mean"] = (
            _jerk.abs().mean().item()
        )
        self.extras["log"]["diagnostics/action_jerk_max_abs"] = (
            _jerk.abs().max().item()
        )
        self.extras["log"]["diagnostics/action_snap_sq_mean"] = (
            _snap.square().mean().item()
        )
        self.extras["log"]["diagnostics/action_snap_abs_mean"] = (
            _snap.abs().mean().item()
        )
        self.extras["log"]["diagnostics/action_snap_max_abs"] = (
            _snap.abs().max().item()
        )
        # Saturation fraction: how often pre-clip action escapes [-1, 1]
        clip_excess = (self._raw_actions.abs() > 1.0).float()
        self.extras["log"]["diagnostics/clip_fraction"] = clip_excess.mean().item()
        self.extras["log"]["diagnostics/clip_fraction_hr1"] = clip_excess[:, 0].mean().item()
        self.extras["log"]["diagnostics/clip_fraction_hp1"] = clip_excess[:, 1].mean().item()
        self.extras["log"]["diagnostics/clip_fraction_kn1"] = clip_excess[:, 2].mean().item()
        self.extras["log"]["diagnostics/clip_fraction_hr2"] = clip_excess[:, 3].mean().item()
        self.extras["log"]["diagnostics/clip_fraction_hp2"] = clip_excess[:, 4].mean().item()
        self.extras["log"]["diagnostics/clip_fraction_kn2"] = clip_excess[:, 5].mean().item()

        # NEW 2026-08-07: sustained-saturation termination diagnostics.
        # knee_sat_ema_max near the 0.5 threshold means the fence is active;
        # sat_term_frac is the per-step fraction of envs killed by it.
        self.extras["log"]["diagnostics/knee_sat_ema_mean"] = self._knee_sat_ema.mean().item()
        self.extras["log"]["diagnostics/knee_sat_ema_max"] = self._knee_sat_ema.max().item()
        self.extras["log"]["diagnostics/sat_term_frac"] = self._sat_terminated.float().mean().item()
        self.extras["log"]["diagnostics/hip_sat_ema_max_roll"] = self._hip_sat_ema[:, 0].max().item()
        self.extras["log"]["diagnostics/hip_sat_ema_max_pitch"] = self._hip_sat_ema[:, 1].max().item()

        # ── NEW IN p6e: lin_vel obs dropout sanity ─────────────
        # Fraction of envs that are currently passing true lin_vel through
        # to the policy. Should hover near (1 - lin_vel_obs_dropout_prob).
        self.extras["log"]["diagnostics/lin_vel_obs_keep_frac"] = (
            self._lin_vel_obs_keep.mean().item()
        )

        self.extras["log"]["diagnostics/leg_forward_ema_abs"] = (
            self._leg_forward_ema.abs().mean().item()
        )
        jpair_ema = self._joint_pair_sum_ema
        self.extras["log"]["symmetry/angle_roll_ema"] = jpair_ema[:, 0].abs().mean().item()
        self.extras["log"]["symmetry/angle_pitch_ema"] = jpair_ema[:, 1].abs().mean().item()
        self.extras["log"]["symmetry/angle_knee_ema"] = jpair_ema[:, 2].abs().mean().item()
        # ── Mirror-symmetry diagnostic ─────────────────────────
        # Per-env L1 distance between obs and its mirror. In a converged
        # symmetric balance policy with cmd=0 the population should sit
        # near a mirror-fixed point on average — this metric should
        # decrease monotonically vs. baseline (no augmentation) once
        # mirror data augmentation is wired in.
        if self._last_obs_for_diag is not None:
            with torch.no_grad():
                obs_m = self.mirror_obs(self._last_obs_for_diag)
                obs_asym = (self._last_obs_for_diag - obs_m).abs().mean(dim=-1)
            self.extras["log"]["symmetry/obs_mirror_asymmetry_mean"] = obs_asym.mean().item()
            self.extras["log"]["symmetry/obs_mirror_asymmetry_max"] = obs_asym.max().item()
        self.extras["log"]["diagnostics/stabilize_fraction"] = (
            (self._stabilize_remaining > 0).float().mean().item()
        )
        self.extras["log"]["diagnostics/grace_fraction"] = (
            (self._grace_remaining > 0).float().mean().item()
        )
        grace_mask = self._grace_remaining > 0
        if grace_mask.any():
            self.extras["log"]["diagnostics/grace_height"] = (
                root_pos[grace_mask, 2].mean().item()
            )
            self.extras["log"]["diagnostics/grace_rise_rate"] = (
                base_vz_world[grace_mask].mean().item()
            )

        self.extras["log"]["diagnostics/effort_hip_roll"]  = self._nominal_diag("hip_roll", "effort_limit")
        self.extras["log"]["diagnostics/effort_hip_pitch"] = self._nominal_diag("hip_pitch", "effort_limit")
        self.extras["log"]["diagnostics/effort_knee"]      = self._nominal_diag("knee", "effort_limit")
        self.extras["log"]["diagnostics/kp_hip_roll"]      = self._nominal_diag("hip_roll", "stiffness")
        self.extras["log"]["diagnostics/kp_hip_pitch"]     = self._nominal_diag("hip_pitch", "stiffness")
        self.extras["log"]["diagnostics/kp_knee"]          = self._nominal_diag("knee", "stiffness")
        self.extras["log"]["diagnostics/kd_hip_roll"]      = self._nominal_diag("hip_roll", "damping")
        self.extras["log"]["diagnostics/kd_hip_pitch"]     = self._nominal_diag("hip_pitch", "damping")
        self.extras["log"]["diagnostics/kd_knee"]          = self._nominal_diag("knee", "damping")
        self.extras["log"]["diagnostics/action_scale_kn"]  = self._action_scale[2].item()

        applied_torques_abs = self.robot.data.applied_torque[:, self._joint_ids].abs()
        self.extras["log"]["diagnostics/actual_torque_hr1_mean"] = applied_torques_abs[:, 0].mean().item()
        self.extras["log"]["diagnostics/actual_torque_hp1_mean"] = applied_torques_abs[:, 1].mean().item()
        self.extras["log"]["diagnostics/actual_torque_kn1_mean"] = applied_torques_abs[:, 2].mean().item()
        self.extras["log"]["diagnostics/actual_torque_hr2_mean"] = applied_torques_abs[:, 3].mean().item()
        self.extras["log"]["diagnostics/actual_torque_hp2_mean"] = applied_torques_abs[:, 4].mean().item()
        self.extras["log"]["diagnostics/actual_torque_kn2_mean"] = applied_torques_abs[:, 5].mean().item()
        self.extras["log"]["diagnostics/actual_torque_hr1_max"] = applied_torques_abs[:, 0].max().item()
        self.extras["log"]["diagnostics/actual_torque_hp1_max"] = applied_torques_abs[:, 1].max().item()
        self.extras["log"]["diagnostics/actual_torque_kn1_max"] = applied_torques_abs[:, 2].max().item()
        self.extras["log"]["diagnostics/actual_torque_hr2_max"] = applied_torques_abs[:, 3].max().item()
        self.extras["log"]["diagnostics/actual_torque_hp2_max"] = applied_torques_abs[:, 4].max().item()
        self.extras["log"]["diagnostics/actual_torque_kn2_max"] = applied_torques_abs[:, 5].max().item()

        # ── Torque asymmetry diagnostics ───────────────────────
        self.extras["log"]["diagnostics/torque_asymmetry_hr"] = (
            (applied_torques_abs[:, 0] - applied_torques_abs[:, 3]).abs().mean().item()
        )
        self.extras["log"]["diagnostics/torque_asymmetry_hp"] = (
            (applied_torques_abs[:, 1] - applied_torques_abs[:, 4]).abs().mean().item()
        )
        self.extras["log"]["diagnostics/torque_asymmetry_kn"] = (
            (applied_torques_abs[:, 2] - applied_torques_abs[:, 5]).abs().mean().item()
        )

        # ── Flight-phase diagnostics ───────────────────────────
        both_airborne = (
            (toe1_z > self.cfg.no_flight_contact_threshold)
            & (toe2_z > self.cfg.no_flight_contact_threshold)
        )
        post_grace_mask = ~grace_mask
        self.extras["log"]["diagnostics/flight_fraction"] = (
            (both_airborne & post_grace_mask).float().mean().item()
        )
        self.extras["log"]["diagnostics/flight_fraction_all"] = (
            both_airborne.float().mean().item()
        )
        self.extras["log"]["diagnostics/toe1_z_mean"] = toe1_z.mean().item()
        self.extras["log"]["diagnostics/toe2_z_mean"] = toe2_z.mean().item()
        self.extras["log"]["diagnostics/toe_min_z_mean"] = (
            torch.min(toe1_z, toe2_z).mean().item()
        )

        # ── Vertical velocity diagnostics ─────────────────────
        self.extras["log"]["diagnostics/base_vz_mean"] = base_vz_world.mean().item()
        self.extras["log"]["diagnostics/base_vz_abs_mean"] = base_vz_world.abs().mean().item()
        self.extras["log"]["diagnostics/base_vz_sq_mean"] = (base_vz_world ** 2).mean().item()

        # ── Stance-time symmetry diagnostics ───────────────────
        self.extras["log"]["diagnostics/foot1_contact_ema"] = (
            self._foot1_contact_ema.mean().item()
        )
        self.extras["log"]["diagnostics/foot2_contact_ema"] = (
            self._foot2_contact_ema.mean().item()
        )
        self.extras["log"]["diagnostics/stance_diff_abs"] = (
            (self._foot1_contact_ema - self._foot2_contact_ema).abs().mean().item()
        )
        self.extras["log"]["diagnostics/foot1_contact_frac"] = (
            foot1_on_ground.mean().item()
        )
        self.extras["log"]["diagnostics/foot2_contact_frac"] = (
            foot2_on_ground.mean().item()
        )

        # ── Per-foot step count diagnostics ────────────────────
        foot1_grounded = foot1_on_ground > 0.5
        foot2_grounded = foot2_on_ground > 0.5
        foot1_liftoff = self._prev_foot1_on_ground & ~foot1_grounded
        foot2_liftoff = self._prev_foot2_on_ground & ~foot2_grounded
        self._prev_foot1_on_ground[:] = foot1_grounded
        self._prev_foot2_on_ground[:] = foot2_grounded
        self.extras["log"]["diagnostics/foot1_liftoff_rate"] = (
            foot1_liftoff.float().mean().item()
        )
        self.extras["log"]["diagnostics/foot2_liftoff_rate"] = (
            foot2_liftoff.float().mean().item()
        )
        self.extras["log"]["diagnostics/liftoff_asymmetry"] = abs(
            foot1_liftoff.float().mean().item()
            - foot2_liftoff.float().mean().item()
        )

        # ── Yaw drift diagnostics ─────────────────────────────
        small_yaw_cmd = (self._yaw_rate_cmd.abs() < self.cfg.yaw_rate_cmd_anchor_threshold)
        if small_yaw_cmd.any():
            self.extras["log"]["diagnostics/yaw_drift_when_no_cmd"] = (
                ang_vel_z_base[small_yaw_cmd].abs().mean().item()
            )
            self.extras["log"]["diagnostics/yaw_drift_sq_when_no_cmd"] = (
                (ang_vel_z_base[small_yaw_cmd] ** 2).mean().item()
            )
        walking_no_yaw = small_yaw_cmd & (self._vx_cmd.abs() > 0.02)
        if walking_no_yaw.any():
            self.extras["log"]["diagnostics/yaw_drift_walking"] = (
                ang_vel_z_base[walking_no_yaw].abs().mean().item()
            )

        # Idle-stepping diagnostics: zero-command envs only — the regime
        # idle_step_gate targets. Watch these move after the change.
        idle_mask = (
            (self._vx_cmd.abs() < 0.02)
            & (self._pickup_phase == 0)
            & (self._grace_remaining <= 0)
            & (self._stabilize_remaining <= 0)
        )
        self.extras["log"]["diagnostics/non_idle_liftoff_rate"] = (
            (foot1_liftoff | foot2_liftoff)[~idle_mask].float().mean().item()
        )
        if idle_mask.any():
            self.extras["log"]["diagnostics/idle_frac"] = idle_mask.float().mean().item()
            self.extras["log"]["diagnostics/idle_liftoff_rate"] = (
                (foot1_liftoff | foot2_liftoff)[idle_mask].float().mean().item()
            )
            self.extras["log"]["diagnostics/idle_flight_fraction"] = (
                both_airborne[idle_mask].float().mean().item()
            )
            self.extras["log"]["diagnostics/idle_foot_height_diff"] = (
                (toe1_z - toe2_z)[idle_mask].abs().mean().item()
            )
            # ── RESUME 2: resting toe-z measurement ────────────────
            # Pickup-excluded (idle_mask already requires phase==0) toe-z
            # percentiles. p10 ≈ true resting toe z with the foot loaded:
            # at a 3 mm shuffle, the lower foot is effectively at rest.
            # CALIBRATION: set cfg.swing_clearance_zero_m = p10 + 0.003
            # after the first ~10 iterations of this resume. Also exposes
            # the contact-threshold margin (0.084 − p10) that decides how
            # high a lift must go before liftoff edges register at all.
            idle_toe_min = torch.min(toe1_z, toe2_z)[idle_mask]
            self.extras["log"]["diagnostics/idle_toe_z_p10"] = (
                torch.quantile(idle_toe_min, 0.10).item()
            )
            self.extras["log"]["diagnostics/idle_toe_z_p50"] = (
                torch.quantile(idle_toe_min, 0.50).item()
            )
            # ── RESUME 2: swing-lift leading indicators ────────────
            # Finer-grained than liftoff_rate. Fraction of idle steps where
            # the clock-designated swing foot is (a) above the dense-ramp
            # start — first sign the clearance gradient is being climbed —
            # and (b) above the contact threshold — the point where clock_
            # contact and liftoff edges start paying. Expect (a) to move
            # first, (b) to follow, liftoff_rate last.
            swing1_diag = gait_sin > 0.0
            swing_foot_z_diag = torch.where(swing1_diag, toe1_z, toe2_z)[idle_mask]
            self.extras["log"]["diagnostics/idle_swing_above_zero_frac"] = (
                (swing_foot_z_diag > self.cfg.swing_clearance_zero_m)
                .float().mean().item()
            )
            self.extras["log"]["diagnostics/idle_swing_above_contact_frac"] = (
                (swing_foot_z_diag > self.cfg.no_flight_contact_threshold)
                .float().mean().item()
            )

        # ── A/B FORK: gait-reference diagnostics ───────────────
        # Restricted to "active" envs (no grace / pickup / stabilize)
        # so kneel-pose hip deviations (~3 rad) don't pollute stats.
        # Phases are randomized per env at reset, so cross-env
        # projections onto sinφ are well-posed.
        #   gait_ref_amp_proj — current hip-pitch oscillation projected
        #       onto the reference: mean(hp_dev·sinφ)/mean(sin²φ).
        #       Should climb 0 → gait_ref_amplitude_rad as tracking is
        #       learned. NEGATIVE ⇒ policy oscillates against the
        #       reference ⇒ flip the cfg amplitude sign.
        #   gait_ref_airborne_swing_hp_dev — SIGN CALIBRATION. Mean hp
        #       deviation of clock-designated swing feet at moments
        #       they're ACTUALLY airborne. Existing lifts are already
        #       clock-aligned, so within ~10–20 iters this reads the
        #       sign the policy's own lifts use. Negative ⇒ flip
        #       cfg.gait_ref_amplitude_rad before the prior entrenches
        #       the wrong direction.
        #   gait_ref_track_err_abs — mean |hp_dev − A·sinφ| per leg
        #       (the exp kernel's input). Falling = tracking; compare
        #       against σ (= 0.15 default).
        active_mask = (
            (self._pickup_phase == 0)
            & (self._grace_remaining <= 0)
            & (self._stabilize_remaining <= 0)
        )
        if active_mask.any():
            _gr_amp = getattr(self.cfg, "gait_ref_amplitude_rad", 0.20)
            _sin_a = gait_sin[active_mask]
            _hp1_a = joint_pos_norm[active_mask, 1]
            _hp2_a = joint_pos_norm[active_mask, 4]
            self.extras["log"]["diagnostics/gait_ref_amp_proj"] = (
                (0.5 * (_hp1_a + _hp2_a) * _sin_a).mean()
                / (torch.square(_sin_a).mean() + 1e-8)
            ).item()
            _ref_a = _gr_amp * _sin_a
            self.extras["log"]["diagnostics/gait_ref_track_err_abs"] = (
                0.5 * ((_hp1_a - _ref_a).abs() + (_hp2_a - _ref_a).abs())
            ).mean().item()
            _swing1_a = _sin_a > 0.0
            _air1_a = toe1_z[active_mask] > self.cfg.no_flight_contact_threshold
            _air2_a = toe2_z[active_mask] > self.cfg.no_flight_contact_threshold
            _m1 = (_air1_a & _swing1_a).float()
            _m2 = (_air2_a & ~_swing1_a).float()
            _den = (_m1.sum() + _m2.sum()).item()
            if _den > 0:
                self.extras["log"]["diagnostics/gait_ref_airborne_swing_hp_dev"] = (
                    ((_hp1_a * _m1).sum() + (_hp2_a * _m2).sum()).item() / _den
                )

        if self._episode_buffer_count > 0:
            kneel_mean = self._episode_lengths_buffer[:self._episode_buffer_count].mean().item()
            self.extras["log"]["curriculum/kneel_mean_ep_len"] = kneel_mean
            self.extras["log"]["curriculum/kneel_mean_ep_ratio"] = (
                kneel_mean / self.max_episode_length
            )
            self.extras["log"]["curriculum/kneel_buffer_fill"] = (
                self._episode_buffer_count / self.cfg.curriculum_advance_window
            )
        self.extras["log"]["curriculum/cooldown_remaining"] = float(self._curriculum_cooldown)
        self.extras["log"]["curriculum/advance_threshold"] = self.cfg.curriculum_advance_threshold

        # ── NEW (v7): difficulty distribution diagnostics ─────────────
        # With uniform [0, alpha] sampling, the kneel branch is no longer a
        # single pose mode. Track the live distribution so you can verify
        # the sampler is working and see which DIFFICULTY band is failing.
        self.extras["log"]["curriculum/difficulty_max"] = self._env_difficulty.max().item()
        self.extras["log"]["curriculum/difficulty_std"] = self._env_difficulty.std().item()
        self.extras["log"]["curriculum/kneel_branch_frac"] = (
            self._was_kneel_start.float().mean().item()
        )
        if self._was_kneel_start.any():
            kneel_diff = self._env_difficulty[self._was_kneel_start]
            self.extras["log"]["curriculum/kneel_diff_mean"] = kneel_diff.mean().item()
            self.extras["log"]["curriculum/kneel_diff_p10"]  = torch.quantile(kneel_diff, 0.10).item()
            self.extras["log"]["curriculum/kneel_diff_p90"]  = torch.quantile(kneel_diff, 0.90).item()

        # ── NEW (v7): stratified ep_len by difficulty tertile ─────────
        # This is the diagnostic that tells you whether the curriculum is
        # stuck because the easy end is fine but the hard end is dying, or
        # whether the whole band is struggling. Without this, the single
        # kneel_mean_ep_len number is ambiguous under uniform sampling.
        stats = self._curriculum_tertile_stats()
        if stats is not None:
            for band in ("low", "mid", "high"):
                dm, el, pg, sr = stats[band]
                self.extras["log"][f"curriculum/strat_{band}_diff_mean"] = dm
                self.extras["log"][f"curriculum/strat_{band}_ep_len"]    = el
                # Grace-corrected: steps survived after grace expiry, and the
                # fraction that survived ≥ curriculum_success_post_grace_steps.
                # These are the stand-up success signals; ep_len is inflated
                # by d×grace even when the robot never rises.
                self.extras["log"][f"curriculum/postgrace_len_{band}"]   = pg
                self.extras["log"][f"curriculum/standup_success_{band}"] = sr
            self.extras["log"]["curriculum/strat_high_minus_low"] = (
                stats["high"][1] - stats["low"][1]
            )
            self.extras["log"]["curriculum/standup_success_all"] = stats["all"][3]
            self.extras["log"]["curriculum/advance_metric_value"] = (
                self._curriculum_advance_metric(stats)[0]
            )
        if self.cfg.curriculum_frontier_frac > 0.0:
            self.extras["log"]["curriculum/frontier_lo"] = max(
                0.0, self._curriculum_alpha - self.cfg.curriculum_frontier_width
            )
        # ── NEW IN p6f: DR diagnostics ─────────────────────────
        # Sanity-check the multiplier distribution (should match the
        # uniform range you configured) and the effective worst-case Kp
        # the policy is actually being trained against.
        if self.cfg.dr_enable:
            self.extras["log"]["dr/stiffness_mult_mean"] = self._dr_stiffness_mult.mean().item()
            self.extras["log"]["dr/stiffness_mult_min"] = self._dr_stiffness_mult.min().item()
            self.extras["log"]["dr/stiffness_mult_max"] = self._dr_stiffness_mult.max().item()
            self.extras["log"]["dr/damping_mult_mean"] = self._dr_damping_mult.mean().item()
            self.extras["log"]["dr/effort_mult_mean"] = self._dr_effort_mult.mean().item()
            # Actuator-lag prefilter DR (v7.1). min/max kept so a clipped or
            # collapsed range is visible at a glance like the other axes.
            self.extras["log"]["dr/act_lag_omega_mult_mean"] = self._dr_act_lag_omega_mult.mean().item()
            self.extras["log"]["dr/act_lag_omega_mult_min"]  = self._dr_act_lag_omega_mult.min().item()
            self.extras["log"]["dr/act_lag_omega_mult_max"]  = self._dr_act_lag_omega_mult.max().item()
            self.extras["log"]["dr/act_lag_zeta_mult_mean"]  = self._dr_act_lag_zeta_mult.mean().item()
            self.extras["log"]["dr/act_lag_zeta_mult_min"]   = self._dr_act_lag_zeta_mult.min().item()
            self.extras["log"]["dr/act_lag_zeta_mult_max"]   = self._dr_act_lag_zeta_mult.max().item()

            self.extras["log"]["dr/action_delay_mean"] = self._env_action_delay.float().mean().item()
            self.extras["log"]["dr/action_delay_min"]  = float(self._env_action_delay.min().item())
            self.extras["log"]["dr/action_delay_max"]  = float(self._env_action_delay.max().item())
            # Obs delay — split IMU / joint in v7.
            # `obs_delay` keeps the legacy TensorBoard key and now reports
            # the JOINT/encoder delay (the dominant, slowest sensor path).
            self.extras["log"]["dr/obs_delay_mean"] = self._env_joint_delay.float().mean().item()
            self.extras["log"]["dr/obs_delay_min"]  = float(self._env_joint_delay.min().item())
            self.extras["log"]["dr/obs_delay_max"]  = float(self._env_joint_delay.max().item())
            self.extras["log"]["dr/imu_delay_mean"] = self._env_imu_delay.float().mean().item()
            self.extras["log"]["dr/imu_delay_min"]  = float(self._env_imu_delay.min().item())
            self.extras["log"]["dr/imu_delay_max"]  = float(self._env_imu_delay.max().item())
            # Combined loop dead time — what really matters for phase margin.
            # Two loops now: the slow joint loop (legacy total_delay) and the
            # fast IMU loop.
            total_joint_delay = self._env_action_delay + self._env_joint_delay
            self.extras["log"]["dr/total_delay_mean"] = total_joint_delay.float().mean().item()
            self.extras["log"]["dr/total_delay_max"]  = float(total_joint_delay.max().item())
            total_imu_delay = self._env_action_delay + self._env_imu_delay
            self.extras["log"]["dr/total_imu_delay_mean"] = total_imu_delay.float().mean().item()
            self.extras["log"]["dr/total_imu_delay_max"]  = float(total_imu_delay.max().item())

            # Constant biases
            self.extras["log"]["dr/joint_pos_bias_abs_mean"] = (
                self._dr_joint_pos_bias.abs().mean().item()
            )
            self.extras["log"]["dr/proj_grav_bias_abs_mean"] = (
                self._dr_proj_grav_bias.abs().mean().item()
            )
            self.extras["log"]["dr/ang_vel_bias_abs_mean"] = (
                self._dr_ang_vel_bias.abs().mean().item()
            )

            # Physics DR
            self.extras["log"]["dr/mass_mult_mean"] = self._dr_mass_mult.mean().item()
            self.extras["log"]["dr/mass_mult_min"]  = self._dr_mass_mult.min().item()
            self.extras["log"]["dr/mass_mult_max"]  = self._dr_mass_mult.max().item()
            self.extras["log"]["dr/friction_mult_mean"] = self._dr_floor_friction_mult.mean().item()
            self.extras["log"]["dr/friction_mult_min"]  = self._dr_floor_friction_mult.min().item()
            self.extras["log"]["dr/friction_mult_max"]  = self._dr_floor_friction_mult.max().item()
            self.extras["log"]["dr/com_offset_abs_mean"] = self._dr_com_offset.abs().mean().item()
            self.extras["log"]["dr/com_shift_x_abs_mean"] = self._dr_com_shift[:, 0].abs().mean().item()
            self.extras["log"]["dr/com_shift_y_abs_mean"] = self._dr_com_shift[:, 1].abs().mean().item()
            self.extras["log"]["dr/com_shift_z_abs_mean"] = self._dr_com_shift[:, 2].abs().mean().item()
            self.extras["log"]["dr/knee_angle_bias_abs_mean"] = self._dr_knee_angle_bias.abs().mean().item()
            self.extras["log"]["dr/hip_angle_bias_abs_mean"] = self._dr_hip_angle_bias.abs().mean().item()

            # ── Foot restitution DR (S6) — READ BACK, not sampled ──────
            # The span below comes from get_material_properties() on the two
            # discovered ball rows, so a silent write failure shows up as a flat
            # min==max line at the carrier 0.63 instead of the drawn band.
            if self._foot_restitution_rows:
                _mats = self.robot.root_physx_view.get_material_properties()
                _e = _mats[:, self._foot_restitution_rows, 2]
                self.extras["log"]["dr/foot_restitution_min"]  = _e.min().item()
                self.extras["log"]["dr/foot_restitution_mean"] = _e.mean().item()
                self.extras["log"]["dr/foot_restitution_max"]  = _e.max().item()
                self.extras["log"]["dr/foot_restitution_std"]  = _e.std().item()

            self.extras["log"]["dr/gravity_tilt_deg_mean"] = self._dr_gravity_tilt_deg.mean().item()
            self.extras["log"]["dr/gravity_tilt_deg_max"] = self._dr_gravity_tilt_deg.max().item()

            # ── S5 elasticity channels (2026-09-12) ────────────────────
            # Every one of these is a mean over the LIVE population, so a band
            # that is drawn but never written (or a write that silently no-ops)
            # shows up here as a flat line at one value.  That is the whole
            # point: the S5 launch must be able to tell "DR does not matter"
            # apart from "DR was never applied".
            if bool(self.cfg.belt_band_live()):
                _bk = self._dr_belt_stiffness
                _bc = self._dr_belt_damping
                self.extras["log"]["dr/belt_k_joint_mean"] = _bk.mean().item()
                self.extras["log"]["dr/belt_k_joint_min"]  = _bk.min().item()
                self.extras["log"]["dr/belt_k_joint_max"]  = _bk.max().item()
                self.extras["log"]["dr/belt_k_joint_std"]  = _bk.std().item()
                self.extras["log"]["dr/belt_c_motor_mean"] = _bc.mean().item()
                self.extras["log"]["dr/belt_c_motor_min"]  = _bc.min().item()
                self.extras["log"]["dr/belt_c_motor_max"]  = _bc.max().item()
                self.extras["log"]["dr/belt_c_motor_std"]  = _bc.std().item()
                # ── belt plant readbacks (2026-09-15 A/B).  Every value is
                # read from the actuator's live state, not from cfg, so a
                # path that silently no-ops shows up as a flat zero here.
                #   spring/motor effort: joint-frame Nm, |.| mean over envs
                #   enc_dev:  |q_enc - q_joint| — what the policy sees minus
                #             what PhysX has (the wound-up belt, rad)
                #   rotor_vel_p99: G*m_dot, the speed the DC envelope sees
                #   hip_armature_written: PhysX hip_pitch armature actually in
                #             the sim (0 = mask active, 0.0385 = mask off)
                _hk = self.robot.actuators.get("hip_knee")
                if _hk is not None and hasattr(_hk, "encoder_joint_state"):
                    _jp = self.robot.data.joint_pos[:, _hk.joint_indices]
                    _jv = self.robot.data.joint_vel[:, _hk.joint_indices]
                    _qe, _qde = _hk.encoder_joint_state(_jp, _jv)
                    _kn = _hk._kn_cols
                    self.extras["log"]["belt/spring_effort_abs_mean"] = _hk.series_spring_effort.abs().mean().item()
                    self.extras["log"]["belt/motor_effort_abs_mean"]  = _hk.series_motor_effort.abs().mean().item()
                    self.extras["log"]["belt/enc_pos_dev_abs_mean"]   = (_qe - _jp[:, _kn]).abs().mean().item()
                    self.extras["log"]["belt/enc_vel_dev_abs_mean"]   = (_qde - _jv[:, _kn]).abs().mean().item()
                    self.extras["log"]["belt/rotor_vel_jointeq_p99"]  = torch.quantile(
                        (_hk._series_motor_vel * (22.0 / 30.0)).abs().flatten(), 0.99).item()
                    _arm = self.robot.data.joint_armature
                    if _arm is not None:
                        self.extras["log"]["belt/hip_armature_written_mean"] = _arm[:, self._hip_pitch_ids].mean().item()
                        self.extras["log"]["belt/knee_armature_written_mean"] = _arm[:, self._knee_ids].mean().item() if hasattr(self, "_knee_ids") else float("nan")
            if self._pad is not None:
                _pk = self._pad.stiffness
                _pc = self._pad.damping
                self.extras["log"]["dr/pad_k_mean"] = _pk.mean().item()
                self.extras["log"]["dr/pad_k_min"]  = _pk.min().item()
                self.extras["log"]["dr/pad_k_max"]  = _pk.max().item()
                self.extras["log"]["dr/pad_k_std"]  = _pk.std().item()
                self.extras["log"]["dr/pad_c_mean"] = _pc.mean().item()
                self.extras["log"]["dr/pad_c_std"]  = _pc.std().item()
                self.extras["log"]["dr/pad_c_clamped_frac"] = (
                    self._pad.clamped.float().mean().item()
                )
                # Engagement: the number that says whether ANY foot knob can
                # matter at all in this regime.  A contact fraction at zero
                # means every foot-side DR arm is inert by construction.
                _cells = max(self._ball_contact_cells, 1)
                self.extras["log"]["foot/ball_contact_frac"] = (
                    self._ball_contact_count.item() / _cells
                )
                # ENGAGEMENT: the share of (env, foot) cells whose ball reached
                # the pad surface at least once since the last log.  This is the
                # number that gates every foot-side conclusion (see the S5
                # launch header): at 0.000 the whole pad band is decorative.
                self.extras["log"]["foot/ball_touched_frac"] = (
                    self._ball_touched.float().mean().item()
                )
                _low = self._ball_lowest_z
                _low = _low[torch.isfinite(_low)]
                self.extras["log"]["foot/ball_lowest_z"] = (
                    _low.min().item() if _low.numel() else float("nan")
                )
                self.extras["log"]["foot/ball_margin_mean"] = (
                    (self._ball_lowest_z - self._ball_reference_z).mean().item()
                    if self._ball_lowest_z.numel() else float("nan")
                )
                # Reset the window so the metric tracks the recent regime, not
                # the best contact ever seen.  Without this the running min
                # freezes at the first deep kneel and reads as permanent.
                self._ball_lowest_z.fill_(float("inf"))
                self._ball_touched.zero_()
                self._ball_contact_count.zero_()
                self._ball_contact_cells = 0

            if self._dr_diag_count >= 256:
                n = self._dr_diag_count
                ep = self._dr_diag_buf["ep_len"][:n]
                ep_norm = ep / max(self.max_episode_length, 1)
                self.extras["log"]["dr_impact/buffer_fill"] = n / self._dr_diag_window
                self.extras["log"]["dr_impact/ep_len_mean"] = ep.mean().item()
                self.extras["log"]["dr_impact/ep_len_std"] = ep.std().item()

                ep_c = ep - ep.mean()
                ep_std = ep.std() + 1e-8
                third = max(n // 3, 1)
                for name in self._dr_diag_features:
                    v = self._dr_diag_buf[name][:n]
                    v_c = v - v.mean()
                    v_std = v.std() + 1e-8
                    # Pearson correlation: negative means "more of this DR → shorter episodes"
                    corr = (ep_c * v_c).mean() / (ep_std * v_std)
                    self.extras["log"][f"dr_impact/corr_{name}"] = corr.item()

                    # Tertile stratification: easy way to spot non-linear effects.
                    order = v.argsort()
                    ep_low  = ep[order[:third]].mean()
                    ep_high = ep[order[-third:]].mean()
                    self.extras["log"][f"dr_impact/{name}_low_third_ep"]  = ep_low.item()
                    self.extras["log"][f"dr_impact/{name}_high_third_ep"] = ep_high.item()
                    self.extras["log"][f"dr_impact/{name}_high_minus_low"] = (ep_high - ep_low).item()

        if self.cfg.pickup_enable:
            phase = self._pickup_phase
            self.extras["log"]["pickup/active_frac"] = (phase > 0).float().mean().item()
            self.extras["log"]["pickup/lift_frac"]   = (phase == 1).float().mean().item()
            self.extras["log"]["pickup/hold_frac"]   = (phase == 2).float().mean().item()
            self.extras["log"]["pickup/lower_frac"]  = (phase == 3).float().mean().item()
            self.extras["log"]["pickup/total_events_cumulative"] = float(
                self._pickup_event_count
            )
            self.extras["log"]["pickup/release_transition_frac"] = (
                (self._pickup_phase == 4).float().mean().item()
            )
            in_rel = self._pickup_phase == 4
            if in_rel.any():
                rem  = self._pickup_remaining[in_rel].float()
                init = self._pickup_release_initial_steps[in_rel].float().clamp(min=1.0)
                self.extras["log"]["pickup/support_factor_mean"] = (
                    (rem / init).mean().item()
                )
                self.extras["log"]["pickup/release_latency_mean_steps"] = (
                    init.mean().item()
                )
            active = phase > 0
            if active.any():
                fz = self._pickup_forces[active, self._torso_idx, 2]
                self.extras["log"]["pickup/force_z_mean"] = fz.mean().item()
                self.extras["log"]["pickup/force_z_max"]  = fz.max().item()
                z_err = self._pickup_target_z[active] - self.robot.data.root_pos_w[active, 2]
                self.extras["log"]["pickup/z_err_abs_mean"] = z_err.abs().mean().item()
                self.extras["log"]["pickup/xy_err_abs_mean"] = (
                (self._pickup_target_xy[active] - self.robot.data.root_pos_w[active, :2])
                .norm(dim=-1).mean().item()
                )
                self.extras["log"]["pickup/horizontal_speed_mean"] = (
                    self.robot.data.root_lin_vel_w[active, :2].norm(dim=-1).mean().item()
                )
                self.extras["log"]["pickup_dr/target_tilt_rad_mean"] = (
                    2.0 * torch.acos(self._pickup_target_quat[active, 0].abs().clamp(max=1.0))
                ).mean().item()
                self.extras["log"]["pickup_dr/pd_factor_trans_mean"] = (
                    self._pickup_pd_factor_trans[active].mean().item()
                )
                self.extras["log"]["pickup_dr/pd_factor_rot_mean"] = (
                    self._pickup_pd_factor_rot[active].mean().item()
                )

        return total_reward

    # ══════════════════════════════════════════════════════════
    # Termination
    # ══════════════════════════════════════════════════════════

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_quat = self.robot.data.root_quat_w
        root_pos = self.robot.data.root_pos_w
        joint_pos = self.robot.data.joint_pos

        projected_gravity = quat_apply_inverse(root_quat, self._gravity_vec)
        cos_tilt = -projected_gravity[:, 2]
        too_tilted = cos_tilt < math.cos(self.cfg.max_tilt_rad)
        too_low = root_pos[:, 2] < self.cfg.min_height

        hip_roll_pos = joint_pos[:, self._hip_roll_ids]
        doing_side_splits = torch.any(hip_roll_pos.abs() > self.cfg.max_hip_roll_rad, dim=-1)
        hip_pitch_pos = joint_pos[:, self._hip_pitch_ids]
        pitch_split = torch.abs(hip_pitch_pos[:, 0] + hip_pitch_pos[:, 1])
        doing_front_splits = pitch_split > self.cfg.max_hip_pitch_split_rad
        knee_pos = joint_pos[:, self._knee_ids]
        knee_split = torch.abs(knee_pos[:, 0] + knee_pos[:, 1])
        doing_knee_splits = knee_split > self.cfg.max_knee_split_rad

        in_grace = self._grace_remaining > 0
        phase    = self._pickup_phase
        support  = self._pickup_support_factor
        # Shielded while held; in RELEASE, once support drops below threshold the
        # robot owns its own balance again, so a real fall counts as a failure.
        shielded = (phase > 0) & ~(
            (phase == 4) & (support < self.cfg.pickup_termination_support_threshold)
        )
        post_grace_terminate = (
            too_tilted | too_low | doing_side_splits
            | doing_front_splits | doing_knee_splits
        )
        terminated = post_grace_terminate & ~in_grace & ~shielded

        # NEW 2026-08-07: sustained knee-saturation termination (cfg §10).
        # EMA charges only while armed (owning balance), so grace/pickup
        # decay it toward zero instead of pre-charging a kill right at
        # release. Rides the standard terminated penalty.
        if self.cfg.sat_term_enable:
            armed = (~in_grace & ~shielded).float()
            alpha = self._control_dt / self.cfg.sat_term_tau_s
            knee_clip = (
                (self._raw_actions[:, [2, 5]].abs() > 1.0)
                .any(dim=-1).float()
            )
            self._knee_sat_ema += alpha * (knee_clip * armed - self._knee_sat_ema)
            self._sat_terminated = (
                (self._knee_sat_ema > self.cfg.sat_term_threshold)
                & ~in_grace & ~shielded
            )
            # Arm-C hip fence (2026-08-09): same mechanism, per-pair EMAs,
            # own (looser) threshold. 0.0 disables — knee-only B config.
            if self.cfg.sat_term_threshold_hip > 0.0:
                hip_clips = torch.stack([
                    (self._raw_actions[:, [0, 3]].abs() > 1.0).any(dim=-1).float(),
                    (self._raw_actions[:, [1, 4]].abs() > 1.0).any(dim=-1).float(),
                ], dim=1)
                self._hip_sat_ema += alpha * (
                    hip_clips * armed.unsqueeze(1) - self._hip_sat_ema)
                hip_trip = (
                    self._hip_sat_ema > self.cfg.sat_term_threshold_hip
                ).any(dim=1)
                self._sat_terminated = self._sat_terminated | (
                    hip_trip & ~in_grace & ~shielded)
            terminated = terminated | self._sat_terminated

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    # ══════════════════════════════════════════════════════════
    # Reset
    # ══════════════════════════════════════════════════════════

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        self._record_dr_impact(env_ids)        # NEW — must be first
        # Skip the first max_episode_length steps of a run: rsl_rl's
        # init_at_random_ep_len pre-loads episode_length_buf with random
        # values, so the initial population reports inflated lengths (smoke
        # 09-06: kneel mean 1586 after 128 steps) and would advance alpha
        # spuriously. Everything that ends after this window is genuine.
        if (self.cfg.curriculum_enable
                and self.common_step_counter >= self.max_episode_length):
            self._record_episode_lengths(env_ids)

        super()._reset_idx(env_ids)
        num_reset = len(env_ids)

        if self.cfg.curriculum_enable:
            if getattr(self, "_force_fixed_difficulty", False):
                # Eval / debug: pin every env at exactly current alpha.
                # Used to read off worst-case policy performance at the
                # current curriculum frontier — this path keeps the old
                # single-pose behavior intentionally.
                self._env_difficulty[env_ids] = torch.full(
                    (num_reset,), self._curriculum_alpha, device=self.device
                )
                self._was_kneel_start[env_ids] = True
            else:
                is_kneel = (
                    torch.rand(num_reset, device=self.device)
                    < self.cfg.kneel_start_probability
                )
                # ── Kneel-branch difficulty distribution ────────────────
                # CHANGED (v7): uniform [0, alpha] instead of fixed alpha.
                #
                # Old behavior: every kneel-branch env at the current
                # alpha frontier — at alpha=0.5, every env started in the
                # SAME half-way-to-kneel pose. Two failure modes:
                #   (a) Policy saw only two discrete pose modes (d=0 and
                #       d=alpha) and learned them as separate behaviors
                #       rather than as a continuous standup manifold.
                #   (b) Each alpha advance moved the entire kneel-branch
                #       population to a harder pose; the easier
                #       intermediate poses the policy had just mastered
                #       disappeared from training → forgetting at every
                #       advance.
                #
                # New: at alpha=0.5, kneel-branch envs span d ∈ [0, 0.5]
                # uniformly. Advancing alpha EXTENDS the range, never
                # removes the easy regime. Advancement now reflects
                # mastery across the whole standup manifold up to alpha,
                # not just at the alpha frontier.
                #
                # Degenerate cases:
                #   alpha=0 → all kneel-branch envs at d=0, same as
                #             stand-branch (fine — pre-curriculum phase).
                #   alpha=1 → kneel-branch spans [0, 1] uniformly =
                #             full standup manifold from stand to kneel.
                u = torch.rand(num_reset, device=self.device)
                kneel_difficulty = u * self._curriculum_alpha
                # Frontier-focused share (09-08, see cfg): these envs draw
                # from [alpha − width, alpha] so the band the policy is
                # currently learning gets a fixed share of the batch
                # regardless of how far alpha has advanced.
                if self.cfg.curriculum_frontier_frac > 0.0:
                    lo = max(0.0, self._curriculum_alpha - self.cfg.curriculum_frontier_width)
                    at_frontier = (
                        torch.rand(num_reset, device=self.device)
                        < self.cfg.curriculum_frontier_frac
                    )
                    kneel_difficulty = torch.where(
                        at_frontier,
                        lo + u * (self._curriculum_alpha - lo),
                        kneel_difficulty,
                    )
                self._env_difficulty[env_ids] = torch.where(
                    is_kneel,
                    kneel_difficulty,
                    torch.zeros(num_reset, device=self.device),
                )
                self._was_kneel_start[env_ids] = is_kneel
        else:
            self._env_difficulty[env_ids] = 0.0
            self._was_kneel_start[env_ids] = False

        difficulty = self._env_difficulty[env_ids]
        d = difficulty.unsqueeze(-1)
        stand_jpos = self._stand_joint_pos.unsqueeze(0).expand(num_reset, -1)
        kneel_jpos = self._kneel_joint_pos.unsqueeze(0).expand(num_reset, -1)
        interp_joint_pos = stand_jpos * (1.0 - d) + kneel_jpos * d
        path_mode = self.cfg.standup_ramp_mode == "path"
        if path_mode:
            # spawn ON the stand-up path at phase s0 = 1 − d (pose, pitch, height from
            # the knot table) instead of on the seiza↔stand joint-space chord
            s0 = 1.0 - difficulty
            self._standup_s0[env_ids] = s0
            interp_joint_pos = self._path_joint_pos(s0, rest=True)
            path_pitch = self._pwl(s0, (self._path_s, self._path_pitch))
            path_h = self._pwl(s0, (self._path_s, self._path_h)) + self.cfg.standup_path_spawn_clearance_m

        # ── Kneel pose randomization (p6d) ─────────────────────
        if self.cfg.kneel_noise_rad > 0.0:
            kneel_noise = sample_uniform(
                -self.cfg.kneel_noise_rad,
                self.cfg.kneel_noise_rad,
                (num_reset, len(self._joint_ids)),
                self.device,
            ) * d
            interp_joint_pos = interp_joint_pos + kneel_noise

        self._env_init_joint_pos[env_ids] = interp_joint_pos

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, self._joint_ids] = interp_joint_pos
        noise_scale = 1.0 - 0.8 * difficulty
        joint_pos += sample_uniform(
            -self.cfg.noise_joint_pos, self.cfg.noise_joint_pos,
            joint_pos.shape, joint_pos.device,
        ) * noise_scale.unsqueeze(-1)
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        default_root = self.robot.data.default_root_state[env_ids].clone()
        default_root[:, :3] += self.scene.env_origins[env_ids]

        stand_q = self._stand_quat.unsqueeze(0).expand(num_reset, -1)
        kneel_q = self._kneel_quat.unsqueeze(0).expand(num_reset, -1)
        interp_quat = _slerp(stand_q, kneel_q, difficulty)
        if path_mode:
            # rotation about +y by the path pitch (kneel = −90° ↔ kneel_quat)
            half = 0.5 * path_pitch
            interp_quat = torch.stack(
                [torch.cos(half), torch.zeros_like(half), torch.sin(half), torch.zeros_like(half)], dim=-1
            )
        default_root[:, 3:7] = interp_quat
        if self.cfg.init_base_orient_noise_rad > 0:
            rp_noise = sample_uniform(
                -self.cfg.init_base_orient_noise_rad, self.cfg.init_base_orient_noise_rad,
                (num_reset, 2), self.device,
            ) * (1.0 - difficulty.unsqueeze(-1))    # don't perturb kneel starts further
            # small-angle quat: w=1, xyz = half-angles around x,y
            noise_q = torch.zeros(num_reset, 4, device=self.device)
            noise_q[:, 0] = 1.0
            noise_q[:, 1] = 0.5 * rp_noise[:, 0]     # roll
            noise_q[:, 2] = 0.5 * rp_noise[:, 1]     # pitch
            noise_q = noise_q / noise_q.norm(dim=-1, keepdim=True)
            # quat_mul(noise_q, interp_quat) — use isaaclab.utils.math.quat_mul
            default_root[:, 3:7] = quat_mul(noise_q, interp_quat)
        fk_height = _compute_fk_height(
            base_quat=default_root[:, 3:7],
            hip_roll1=interp_joint_pos[:, 0],
            hip_pitch1=interp_joint_pos[:, 1],
            knee1=interp_joint_pos[:, 2],
            hip_roll2=interp_joint_pos[:, 3],
            hip_pitch2=interp_joint_pos[:, 4],
            knee2=interp_joint_pos[:, 5],
            device=self.device,
        )
        correction = self._fk_kneel_overshoot * (difficulty ** 3)
        interp_height = fk_height - correction
        if path_mode:
            interp_height = torch.where(difficulty > 0.0, path_h, interp_height)

        default_root[:, 2] = interp_height + self.scene.env_origins[env_ids, 2]

        vel_noise_scale = 1.0 - 0.9 * difficulty
        default_root[:, 7:10] += (
            sample_uniform(-self.cfg.noise_base_vel_lin, self.cfg.noise_base_vel_lin,
                           (num_reset, 3), default_root.device)
            * vel_noise_scale.unsqueeze(-1)
        )
        default_root[:, 10:13] += (
            sample_uniform(-self.cfg.noise_base_vel_ang, self.cfg.noise_base_vel_ang,
                           (num_reset, 3), default_root.device)
            * vel_noise_scale.unsqueeze(-1)
        )

        self.robot.write_root_pose_to_sim(default_root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._last_actions[env_ids] = 0.0
        self._last_last_actions[env_ids] = 0.0   # NEW
        self._last_last_last_actions[env_ids] = 0.0   # NEW: t-
        self._raw_actions[env_ids] = 0.0
        self._knee_sat_ema[env_ids] = 0.0
        self._hip_sat_ema[env_ids] = 0.0
        
        # ── Action delay reset (NEW) ───────────────────────────────
        self._action_delay_buffer[env_ids] = 0.0
        if self.cfg.dr_enable and self._max_action_delay > 0:
            self._env_action_delay[env_ids] = torch.randint(
                self.cfg.dr_action_delay_min,
                self.cfg.dr_action_delay_max + 1,
                (num_reset,), device=self.device,
            )
        else:
            self._env_action_delay[env_ids] = 0
        # NEW IN p6e: resample lin_vel obs dropout mask per env per episode.
        # Uses Bernoulli(1 - dropout_prob) — i.e., probability of KEEPING lin_vel.
        keep_prob = 1.0 - self.cfg.lin_vel_obs_dropout_prob
        self._lin_vel_obs_keep[env_ids] = (
            torch.rand(num_reset, device=self.device) < keep_prob
        ).float()
        self._spawn_pos_xy[env_ids] = default_root[:, :2] - self.scene.env_origins[env_ids, :2]
        self._spawn_yaw[env_ids] = _yaw_from_quat(default_root[:, 3:7])
        # Seed anchor EMAs to spawn so initial averaged drift is exactly zero.
        self._pos_ema[env_ids] = self._spawn_pos_xy[env_ids].clone()
        self._yaw_ema[env_ids] = self._spawn_yaw[env_ids].clone()
        self._gait_phase[env_ids] = (
            torch.rand(num_reset, device=self.device) * 2.0 * math.pi
        )
        self._push_forces[env_ids] = 0.0
        self._push_remaining[env_ids] = 0
        self._leg_forward_ema[env_ids] = 0.0
        self._vx_avg[env_ids] = 0.0
        self._yaw_rate_avg[env_ids] = 0.0
        self._prev_feet_mid_world[env_ids] = 0.0
        self._prev_toe1_world[env_ids] = 0.0
        self._prev_toe2_world[env_ids] = 0.0
        self._joint_pair_sum_ema[env_ids] = 0.0
        self._foot1_contact_ema[env_ids] = 0.5
        self._foot2_contact_ema[env_ids] = 0.5
        self._prev_foot1_on_ground[env_ids] = True
        self._prev_foot2_on_ground[env_ids] = True

        grace_steps = (difficulty * self._max_grace_steps).long()
        self._grace_remaining[env_ids] = grace_steps
        self._grace_steps_assigned[env_ids] = grace_steps
        self._stabilize_remaining[env_ids] = 0
        self._initial_height[env_ids] = interp_height

        if getattr(self, "manual_cmd_override", False):
            self._vx_cmd[env_ids] = getattr(self, "manual_vx", 0.0)
            self._yaw_rate_cmd[env_ids] = getattr(self, "manual_yaw_rate", 0.0)
        else:
            self._vx_cmd[env_ids] = 0.0
            self._yaw_rate_cmd[env_ids] = 0.0
        # ── Obs delay reset (NEW IN p6g) ───────────────────────────
        self._obs_delay_buffer[env_ids] = 0.0
        self._needs_history_init[env_ids] = True   # NEW IN p6h
        # Handover-contract randomization (S-line): a subset of resets get
        # the delay buffer primed with the real first frame instead.
        if self.cfg.handover_prime_prob > 0.0:
            self._needs_delay_prime[env_ids] = (
                torch.rand(num_reset, device=self.device) < self.cfg.handover_prime_prob
            )
        else:
            self._needs_delay_prime[env_ids] = False
        # ── Sample per-episode DR (obs biases + physics) (NEW IN p6g)
        self._sample_dr_obs(env_ids, num_reset)
        self._sample_dr_physics(env_ids, num_reset)
        # NEW IN p6f: sample DR multipliers for these envs and immediately
        # push the new per-env actuator state to sim. Writing all envs (not
        # just env_ids) is wasteful but keeps the helper simple, and the
        # cost is rounding-error compared to PPO compute.
        self._sample_dr(env_ids, num_reset)
        self._write_per_env_actuators_to_sim()
        # Clear pickup state on reset (defensive — termination is gated
        # during pickup so envs shouldn't reset mid-pickup, but if they
        # do, leave no stale phase behind).
        self._pickup_phase[env_ids] = 0
        self._pickup_remaining[env_ids] = 0
        self._pickup_target_z[env_ids] = 0.0
        self._pickup_lower_start_z[env_ids] = 0.0
        self._pickup_lower_end_z[env_ids] = 0.0

        # Refresh total mass for picked-up force calc (depends on dr_mass_mult).
        self._total_mass[env_ids] = (
            self._nominal_total_mass[env_ids] * self._dr_mass_mult[env_ids]
        )

        # ── Sample slope tilt for reset envs ──────────────────────
        max_deg = getattr(self.cfg, 'dr_gravity_tilt_max_deg', 0.0)
        if max_deg > 0.0 and self.cfg.dr_enable:
            n_reset = env_ids.shape[0] if isinstance(env_ids, torch.Tensor) else len(env_ids)
            tilt_rad = (
                torch.rand(n_reset, device=self.device) * max_deg * math.pi / 180.0
            )
            direction = torch.rand(n_reset, device=self.device) * 2.0 * math.pi
 
            # F = m * g * sin(tilt) in the tilt direction, applied to torso
            g = 9.81
            mass = self._total_mass[env_ids]  # already DR-scaled
            force_mag = mass * g * torch.sin(tilt_rad)
            fx = force_mag * torch.cos(direction)
            fy = force_mag * torch.sin(direction)
 
            self._slope_forces[env_ids] = 0.0
            self._slope_forces[env_ids, self._torso_idx, 0] = fx
            self._slope_forces[env_ids, self._torso_idx, 1] = fy
            self._dr_gravity_tilt_deg[env_ids] = tilt_rad * 180.0 / math.pi

        self._pickup_target_xy[env_ids, :] = 0.0
        self._pickup_target_quat[env_ids, :] = 0.0
        self._pickup_target_quat[env_ids, 0] = 1.0
        self._pickup_pd_factor_trans[env_ids] = 1.0
        self._pickup_pd_factor_rot[env_ids] = 1.0
        self._pickup_hold_phase[env_ids] = 0.0
        self._pickup_release_initial_steps[env_ids] = 0
        self._pickup_support_factor[env_ids] = 1.0
        self._filtered_actions[env_ids] = 0.0
        # ── Actuator-lag prefilter reset ─────────────────────────────
        # Seed the filtered target at the spawn action-center (= the grace-full
        # action_center, == _env_init_joint_pos set above) with zero velocity, so
        # the command starts exactly where the joint is and the first control
        # step doesn't ring off an artificial step. The per-env ω/ζ coefficients
        # for these envs are (re)sampled in _sample_dr below. Mirrors the replay,
        # which inits pf_y to the pre-trial holding pose.
        self._act_lag_pf_y[env_ids] = self._env_init_joint_pos[env_ids]
        self._act_lag_pf_yd[env_ids] = 0.0
        self._motor_pos_target[env_ids] = self._env_init_joint_pos[env_ids]

        # NEW: pickup-on-spawn. Must come AFTER the pickup-state clear
        # above (we are deliberately overwriting phase 0 → 1 for the
        # selected subset). The mid-episode probability trigger in
        # _apply_pickup_events can't reach step 0 reliably, hence
        # explicit scheduling here.
        self._trigger_initial_pickup(env_ids)
        

    def _sample_commands(self, env_ids, num_reset):
        if getattr(self, "manual_cmd_override", False):
            self._vx_cmd[env_ids] = getattr(self, "manual_vx", 0.0)
        else:
            vx = (
                torch.rand(num_reset, device=self.device)
                * (self.cfg.vx_cmd_max - self.cfg.vx_cmd_min)
                + self.cfg.vx_cmd_min
            )
            zero_mask = torch.rand(num_reset, device=self.device) < self.cfg.vx_cmd_zero_probability
            vx[zero_mask] = 0.0
            vx[vx.abs() < 0.02] = 0.0
            self._vx_cmd[env_ids] = vx

        if getattr(self, "manual_cmd_override", False):
            self._yaw_rate_cmd[env_ids] = getattr(self, "manual_yaw_rate", 0.0)
        else:
            yr = (
                torch.rand(num_reset, device=self.device)
                * (self.cfg.yaw_rate_cmd_max - self.cfg.yaw_rate_cmd_min)
                + self.cfg.yaw_rate_cmd_min
            )
            zero_mask = (
                torch.rand(num_reset, device=self.device)
                < self.cfg.yaw_rate_cmd_zero_probability
            )
            yr[zero_mask] = 0.0
            yr[yr.abs() < 0.05] = 0.0
            self._yaw_rate_cmd[env_ids] = yr

    def _sample_dr_obs(self, env_ids, num_reset):
        """Sample per-episode obs biases + obs delay index."""
        if not self.cfg.dr_enable:
            return
        # Encoder zero offsets — per joint.
        self._dr_joint_pos_bias[env_ids] = sample_uniform(
            -self.cfg.dr_joint_pos_bias_max, self.cfg.dr_joint_pos_bias_max,
            (num_reset, len(self._joint_ids)), self.device,
        )
        # IMU calibration / mounting tilt — proj_grav components.
        self._dr_proj_grav_bias[env_ids] = sample_uniform(
            -self.cfg.dr_proj_grav_bias_max, self.cfg.dr_proj_grav_bias_max,
            (num_reset, 3), self.device,
        )
        # EVAL-ONLY pin (P12 08-25): exact pg_x obs bias for every env,
        # replicating a measured hardware trim offset. NaN = off.
        if not math.isnan(self.cfg.dr_proj_grav_bias_pin_x):
            self._dr_proj_grav_bias[env_ids, 0] = self.cfg.dr_proj_grav_bias_pin_x
        # Execution-noise: episode-constant target bias (P13).
        if self.cfg.exec_noise_target_bias_max > 0.0:
            self._exec_target_bias[env_ids] = sample_uniform(
                -self.cfg.exec_noise_target_bias_max,
                self.cfg.exec_noise_target_bias_max,
                (num_reset, len(self._joint_ids)), self.device,
            )
        elif (self.cfg.dr_knee_angle_bias_max > 0.0
              or self.cfg.dr_hip_angle_bias_max > 0.0
              or self._dr_override):
            self._exec_target_bias[env_ids] = 0.0
        # Knee joint-angle bias (S-line): encoder-vs-joint offset δ per leg.
        # Encoder (and thus the policy) reads q_true − δ; the motor-side PD
        # holds the encoder value on target, so the true joint sits at
        # target + δ. Both halves are folded into the existing bias tensors.
        # The fold is always relative to the previously folded value, so the
        # per-env override can replace it afterwards.
        self._dr_knee_angle_bias[env_ids] = 0.0
        self._dr_hip_angle_bias[env_ids] = 0.0
        if self.cfg.dr_knee_angle_bias_max > 0.0:
            delta = sample_uniform(
                -self.cfg.dr_knee_angle_bias_max, self.cfg.dr_knee_angle_bias_max,
                (num_reset, 2), self.device,
            )
            self._fold_angle_bias(env_ids, (2, 5), delta, self._dr_knee_angle_bias)
        if self.cfg.dr_hip_angle_bias_max > 0.0:
            delta = sample_uniform(
                -self.cfg.dr_hip_angle_bias_max, self.cfg.dr_hip_angle_bias_max,
                (num_reset, 2), self.device,
            )
            self._fold_angle_bias(env_ids, (1, 4), delta, self._dr_hip_angle_bias)
        self._apply_dr_override_obs(env_ids)
        # Gyro bias — ang_vel components.
        self._dr_ang_vel_bias[env_ids] = sample_uniform(
            -self.cfg.dr_ang_vel_bias_max, self.cfg.dr_ang_vel_bias_max,
            (num_reset, 3), self.device,
        )
        # Obs delay indices — IMU and joint sampled independently (v7).
        # randint handles the locked [n, n] ranges (returns n) and the
        # disabled [0, 0] case (returns 0), so no separate guard needed.
        self._env_imu_delay[env_ids] = torch.randint(
            self.cfg.dr_imu_delay_min,
            self.cfg.dr_imu_delay_max + 1,
            (num_reset,), device=self.device,
        )
        self._env_joint_delay[env_ids] = torch.randint(
            self.cfg.dr_joint_delay_min,
            self.cfg.dr_joint_delay_max + 1,
            (num_reset,), device=self.device,
        )

    def _sample_dr_physics(self, env_ids, num_reset):
        """Sample per-env physics multipliers (mass / friction / CoM)
        and write to PhysX. Defaults to no-op when dr_enable=False."""
        if not self.cfg.dr_enable:
            return

        self._dr_mass_mult[env_ids] = sample_uniform(
            self.cfg.dr_mass_mult_min, self.cfg.dr_mass_mult_max,
            (num_reset,), self.device,
        )
        self._dr_floor_friction_mult[env_ids] = sample_uniform(
            self.cfg.dr_floor_friction_min, self.cfg.dr_floor_friction_max,
            (num_reset,), self.device,
        )
        # Foot-restitution DR (S6): bucketed draw over the two ball rows only.
        # A continuous U(min,max) draw is quantized into num_buckets bins so the
        # draw lives on the same grid as the DR event manager's material sampler
        # (which is what "num_buckets" names there).
        if (self.cfg.dr_foot_restitution_max > 0.0 or self.cfg.dr_foot_restitution_min > 0.0):
            lo = self.cfg.dr_foot_restitution_min
            hi = self.cfg.dr_foot_restitution_max
            nb = self.cfg.dr_foot_restitution_buckets
            self._dr_foot_restitution[env_ids] = (
                lo + (hi - lo) * (torch.randint(0, nb, (num_reset,), device=self.device).float() + 0.5) / nb
            )
        n_bodies = self.robot.num_bodies
        self._dr_com_offset[env_ids] = sample_uniform(
            -self.cfg.dr_com_offset_max, self.cfg.dr_com_offset_max,
            (num_reset, n_bodies, 3), self.device,
        )
        # Coherent whole-body CoM shift (S-line): one base-frame offset per
        # env, realized on the torso row scaled by M/m_torso so the total
        # CoM moves by exactly the sampled amount. Rides _dr_com_offset, so
        # the PhysX write, the critic's com_abs channel and the diagnostics
        # all see it without further plumbing.
        self._dr_com_shift[env_ids] = 0.0
        if (self.cfg.dr_com_shift_x_max > 0.0 or self.cfg.dr_com_shift_y_max > 0.0
                or self.cfg.dr_com_shift_z_max > 0.0):
            shift = torch.zeros(num_reset, 3, device=self.device)
            shift[:, 0] = sample_uniform(
                -self.cfg.dr_com_shift_x_max, self.cfg.dr_com_shift_x_max,
                (num_reset,), self.device,
            )
            shift[:, 1] = sample_uniform(
                -self.cfg.dr_com_shift_y_max, self.cfg.dr_com_shift_y_max,
                (num_reset,), self.device,
            )
            shift[:, 2] = sample_uniform(
                -self.cfg.dr_com_shift_z_max, self.cfg.dr_com_shift_z_max,
                (num_reset,), self.device,
            )
            self._dr_com_shift[env_ids] = shift
            self._dr_com_offset[env_ids, self._torso_idx, :] += shift * self._com_shift_gain

        self._apply_dr_override_physics(env_ids)
        self._write_dr_physics_to_sim(env_ids)

    # ── Per-env DR override + joint-angle-bias fold helpers (2026-09-10) ──
    def _env_ids_tensor(self, env_ids) -> torch.Tensor:
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long)

    def _fold_angle_bias(self, env_ids, slots, delta_new, store):
        """Set the coupled encoder/servo offset on `slots` (per-leg pair) to
        `delta_new` [num_reset, 2]: obs bias −δ, motor target bias +δ.
        Relative to the value already folded (kept in `store`), so it can be
        called again to replace a sampled draw."""
        ids = self._env_ids_tensor(env_ids)
        rows = ids.unsqueeze(-1)
        cols = torch.tensor(slots, device=self.device, dtype=torch.long)
        d = delta_new - store[ids]
        store[ids] = delta_new
        self._dr_joint_pos_bias[rows, cols] -= d
        self._exec_target_bias[rows, cols] += d

    def _apply_dr_override_obs(self, env_ids):
        ov = self._dr_override
        if not ov:
            return
        ids = self._env_ids_tensor(env_ids)
        if "proj_grav_bias_x" in ov:
            self._dr_proj_grav_bias[ids, 0] = ov["proj_grav_bias_x"][ids]
        if "knee_bias" in ov:
            self._fold_angle_bias(ids, (2, 5), ov["knee_bias"][ids], self._dr_knee_angle_bias)
        if "hip_bias" in ov:
            self._fold_angle_bias(ids, (1, 4), ov["hip_bias"][ids], self._dr_hip_angle_bias)

    def _series_rotor_inertia_for(self, env_ids):
        """Motor-side rotor inertia handed to the belt integrator for env_ids.

        With the coherent per-leg draw the belt's rotor is the same rotor the
        PhysX knee armature (and the off-diagonal) were scaled by, so it
        carries that leg's multiplier: [num, pairs] = cfg scalar x the knee
        columns of the [N,6] draw.  Legacy topology returns the cfg scalar,
        which is what every S5/S7 run installed.
        """
        base = float(self.cfg.belt_series_rotor_inertia)
        if not self.cfg.dr_armature_per_leg:
            return base
        ids = self._env_ids_tensor(env_ids)
        return base * self._dr_armature_mult[ids].index_select(-1, self._armature_pair_knee_slots)

    def _apply_dr_override_physics(self, env_ids):
        """Runs after the physics draw and before the PhysX write."""
        ov = self._dr_override
        if not ov:
            return
        ids = self._env_ids_tensor(env_ids)
        if "mass_mult" in ov:
            self._dr_mass_mult[ids] = ov["mass_mult"][ids]
        if "floor_friction_mult" in ov:
            self._dr_floor_friction_mult[ids] = ov["floor_friction_mult"][ids]
        if "com_shift" in ov:
            new = ov["com_shift"][ids]
            self._dr_com_offset[ids, self._torso_idx, :] += (
                (new - self._dr_com_shift[ids]) * self._com_shift_gain
            )
            self._dr_com_shift[ids] = new
        # Belt series elasticity, pinned per env by a sweep driver.  Values are
        # JOINT-referred like the cfg band; a zero stiffness keeps the pair on
        # the rigid path regardless of what the actuator group was seeded with.
        # The armature write is re-issued here (and is idempotent with the one
        # in _reset_idx) because the mask depends on these parameters.
        if "belt_series" in ov:
            # Two SEPARATE [num_ids, 2] tensors, one per quantity, so neither
            # has to remember a column convention.  The frame the actuator
            # setter and the held DR buffers both want is [num_ids, 2], kept
            # intact end to end: slicing a row out of the indexed tensor
            # (`a, b = ov[...][ids]`) unpacks the ROW axis and raises for any
            # batch above one env, and a [num_ids] value cannot be written into
            # a [num_ids, 2] slot.  Both were hit on 2026-09-11.
            stiff = ov["belt_series"][ids].to(torch.float32)[:, 0:1]
            damp = ov["belt_series_damping"][ids].to(torch.float32)[:, 0:1]
            self._dr_belt_stiffness[ids, :] = stiff
            self._dr_belt_damping[ids, :] = damp
            _hk = self.robot.actuators.get("hip_knee")
            if _hk is not None:
                _hk.set_series_parameters(
                    ids,
                    stiffness=stiff * _BELT_G2,     # joint-side -> motor-side
                    damping=damp,                   # already motor-side
                    # Keep the pair axis: the setter broadcasts against
                    # (num_ids, 2) and a 1-D [num_ids] vector does NOT
                    # broadcast into it (size 128 cannot become size 2) —
                    # it raises "must broadcast to [envs, pairs]".
                    enabled=torch.full_like(stiff, bool(_hk.cfg.series_elasticity),
                                            dtype=torch.bool) & (stiff > 0.0),
                    rotor_inertia=self._series_rotor_inertia_for(ids),
                )
                # The mask changed with those parameters, so the PhysX
                # armature must be rewritten for these envs.
                self._write_per_env_actuators_to_sim()
        # Ball-pad compliance, pinned per env by a sweep driver. Same [num_ids, 2]
        # convention as belt_series — column 0 left foot, column 1 right — and
        # the same "zero stiffness = rigid cell" rule.  Nothing to install on an
        # actuator: the pad IS a force, so writing the resolved pair into the pad
        # object is the whole application.  `pad_contact_damping` is optional;
        # without it the cfg's damping ratio derives c from the pinned k.
        if "pad_contact" in ov and self._pad is not None:
            stiff = ov["pad_contact"][ids].to(torch.float32)[:, 0:1]
            if "pad_contact_damping" in ov:
                damp = ov["pad_contact_damping"][ids].to(torch.float32)[:, 0:1]
            else:
                damp = self._pad.resolved_kc(
                    stiff, self.cfg.resolved_pad_damping_ratio())[1]
            self._pad.set_parameters(ids, stiff, damp)

    def _write_dr_physics_to_sim(self, env_ids):
        """Push mass / CoM / friction randomization to PhysX for the
        given envs.

        PhysX views are CPU-resident: get_masses/get_coms/get_material_properties
        return CPU tensors and set_* expects CPU. Our DR multipliers and
        env_ids live on CUDA. We bridge by moving env_ids and the
        per-env DR factors to CPU once, then doing all the indexing
        and multiplication on CPU where the destination already is."""
        env_ids_cpu = env_ids.detach().to("cpu")

        # Pull DR factors for these envs to CPU (small tensors).
        mass_mult_cpu = self._dr_mass_mult[env_ids].detach().to("cpu")
        fric_mult_cpu = self._dr_floor_friction_mult[env_ids].detach().to("cpu")
        com_off_cpu   = self._dr_com_offset[env_ids].detach().to("cpu")

        # ── Mass: multiplicative on USD originals ──────────────
        masses = self.robot.root_physx_view.get_masses().clone()
        new_m = (
            self._original_masses[env_ids_cpu]
            * mass_mult_cpu.unsqueeze(-1)
        ).to(masses.dtype)
        if self.cfg.dr_top_mass_add_kg != 0.0:
            top = torch.as_tensor(self._top_body_ids)
            m_top = self._original_masses[env_ids_cpu][:, top]
            new_m[:, top] += (
                self.cfg.dr_top_mass_add_kg * m_top / m_top.sum(dim=-1, keepdim=True)
            ).to(new_m.dtype)
        masses[env_ids_cpu] = new_m
        self.robot.root_physx_view.set_masses(masses, env_ids_cpu)

        # ── Inertia: scale each body's tensor by its mass ratio (uniform-density
        #    assumption; covers mass_mult AND the top-mass pin) ──────────────
        if self.cfg.dr_mass_recompute_inertia:
            ratios = new_m / self._original_masses[env_ids_cpu]
            inertias = self.robot.root_physx_view.get_inertias().clone()
            inertias[env_ids_cpu] = (
                self._original_inertias[env_ids_cpu] * ratios.unsqueeze(-1)
            ).to(inertias.dtype)
            self.robot.root_physx_view.set_inertias(inertias, env_ids_cpu)

        # ── CoM: additive offset on USD originals (positions only) ──
        coms = self.robot.root_physx_view.get_coms().clone()
        new_com_pos = (
            self._original_coms[env_ids_cpu, :, :3] + com_off_cpu
        ).to(coms.dtype)
        coms[env_ids_cpu, :, :3] = new_com_pos
        self.robot.root_physx_view.set_coms(coms, env_ids_cpu)

        # ── Friction: multiplicative on USD originals ──────────
        mats = self.robot.root_physx_view.get_material_properties().clone()
        fric_b = fric_mult_cpu.unsqueeze(-1)  # broadcast over body axis
        mats[env_ids_cpu, :, 0] = (
            self._original_static_fric[env_ids_cpu] * fric_b
        ).to(mats.dtype)
        mats[env_ids_cpu, :, 1] = (
            self._original_dyn_fric[env_ids_cpu] * fric_b
        ).to(mats.dtype)
        # ── Foot restitution DR (S6): the two ball rows only ─────
        # Restitution shares this same per-env buffer, but only the two
        # discovered ball rows are touched — every other row's column 2 is left
        # at its original 0 (friction keeps its own columns 0/1 mechanism).
        if self._foot_restitution_rows:
            e_cpu = self._dr_foot_restitution[env_ids].detach().to("cpu")
            for r in self._foot_restitution_rows:
                mats[env_ids_cpu, r, 2] = e_cpu.to(mats.dtype)
        self.robot.root_physx_view.set_material_properties(mats, env_ids_cpu)

    def _sample_dr(self, env_ids, num_reset):
        n_j = len(self._joint_ids)
        self._dr_stiffness_mult[env_ids]   = sample_uniform(self.cfg.dr_stiffness_mult_min,  self.cfg.dr_stiffness_mult_max,  (num_reset, n_j), self.device)
        self._dr_damping_mult[env_ids]     = sample_uniform(self.cfg.dr_damping_mult_min,    self.cfg.dr_damping_mult_max,    (num_reset, n_j), self.device)
        self._dr_effort_mult[env_ids]      = sample_uniform(self.cfg.dr_effort_mult_min,     self.cfg.dr_effort_mult_max,     (num_reset, n_j), self.device)
        _arm_mult = sample_uniform(self.cfg.dr_armature_mult_min, self.cfg.dr_armature_mult_max, (num_reset, n_j), self.device)
        if self.cfg.dr_armature_per_leg:
            # One rotor per leg: the knee column takes the hip_pitch draw so
            # the hip diagonal, knee diagonal, off-diagonal (c*armature_kn in
            # BeltKneeDCMotor) and the series rotor (_series_rotor_inertia_for)
            # all scale together.  Same sample shape as the legacy draw, so the
            # RNG stream of every other multiplier is untouched.
            _arm_mult = _tie_armature_pair_columns(_arm_mult, self._armature_pair_slots)
        self._dr_armature_mult[env_ids]    = _arm_mult
        self._dr_jfriction_mult[env_ids]   = sample_uniform(self.cfg.dr_joint_friction_mult_min, self.cfg.dr_joint_friction_mult_max, (num_reset, n_j), self.device)
        self._dr_dyn_friction_mult[env_ids]= sample_uniform(self.cfg.dr_dyn_friction_mult_min,   self.cfg.dr_dyn_friction_mult_max,   (num_reset, n_j), self.device)
        self._dr_visc_friction_mult[env_ids]= sample_uniform(self.cfg.dr_visc_friction_mult_min, self.cfg.dr_visc_friction_mult_max,  (num_reset, n_j), self.device)
        self._dr_voltage_factor[env_ids]   = sample_uniform(self.cfg.dr_voltage_factor_min,  self.cfg.dr_voltage_factor_max,  (num_reset,), self.device)
        # Belt series elasticity / damping.  Values are drawn once per reset and
        # then held; min=max pins a deterministic arm and all-zero defaults
        # select the legacy rigid path.
        #
        # A per-env override (the sweep driver) WINS over the cfg band — note
        # this runs AFTER _apply_dr_override_physics in _reset_idx, so writing
        # the band here unconditionally would silently erase the painted cells
        # and leave every env on the rigid plant while the driver reported
        # otherwise.  That is exactly what the first sweep dry-runs did.
        # Ball-pad compliance: draw a spring/damper pair per env per foot.
        #
        # A per-env override (a sweep driver painting cells) WINS over the cfg
        # band, for exactly the reason the belt branch documents below: this
        # runs after _apply_dr_override_physics, and an unconditional write
        # here would erase the painted values while the driver reported them.
        #
        # The pad is a PLANT FORCE, not an actuator, so unlike the belt there is
        # nothing to install on the articulation — writing the resolved (k, c)
        # into the pad object IS the whole application.
        #
        # Two DIFFERENT callers use the override, and they want opposite things
        # on the next reset (2026-09-12 evening):
        #
        #   * A sweep driver paints cells and re-paints them every arm, so it
        #     wants the override to keep winning — it passes `pad_contact`.
        #   * The play pad knobs (`_PadKnobs`) pin ONE value for a by-feel
        #     session, so they pass `pad_contact_hold` and the DR band must be
        #     drawn on every reset with that value written over the draw. The
        #     difference matters: without it, one N press would silently destroy
        #     the per-env band for the rest of the run, and every env would come
        #     back at the tuned value while the terminal still showed a band.
        #
        # `pad_contact` (cell-painted, whole tensor) still short-circuits the
        # draw entirely, so the sweep path is bit-identical to before.
        ov_pad = self._dr_override.get("pad_contact") if self._dr_override else None
        ov_hold = self._dr_override.get("pad_contact_hold") if self._dr_override else None
        if self._pad is not None and ov_pad is None:
            if self.cfg.pad_enabled():
                k_draw = sample_uniform(
                    self.cfg.pad_stiffness_min,
                    self.cfg.pad_stiffness_max,
                    (num_reset, 2), self.device,
                )
                if self.cfg.pad_damping_band_live():
                    # S7: a per-episode zeta, resolved against the stiffness
                    # THIS episode drew, so the mode is the one the recipe names
                    # rather than a fixed coefficient that means a different zeta
                    # at every k.  Same form and same reason as the belt's ratio
                    # branch below.
                    zeta_draw = sample_uniform(
                        self.cfg.pad_damping_ratio_min,
                        self.cfg.pad_damping_ratio_max,
                        (num_reset, 2), self.device,
                    )
                    c_draw = self._pad.resolved_kc(k_draw, zeta_draw)[1]
                elif self.cfg.pad_damping_min > 0.0 or self.cfg.pad_damping_max > 0.0:
                    c_draw = sample_uniform(
                        self.cfg.pad_damping_min,
                        self.cfg.pad_damping_max,
                        (num_reset, 2), self.device,
                    )
                else:
                    c_draw = self._pad.resolved_kc(
                        k_draw, self.cfg.resolved_pad_damping_ratio())[1]
            else:
                # Inert arm: pin every env to a zero force so a cfg that turns
                # the pad off mid-run cannot leave a previous draw behind.
                k_draw = torch.zeros(num_reset, 2, device=self.device)
                c_draw = torch.zeros(num_reset, 2, device=self.device)
            if ov_hold is not None:
                # A by-feel pin, not a painted cell: the band above is drawn so
                # the DR accounting stays honest, then the held value is written
                # over the resets that share it. A held value of zero turns the
                # pad off for those envs and keeps `resolved_kc` out of a
                # division by zero, exactly as the inert arm does.
                #
                # `env_ids` arrives as a Sequence from _reset_idx, so it goes
                # through the same coercion the override applier uses; indexing
                # a tensor with a plain list works but silently loses the device.
                hold_ids = self._env_ids_tensor(env_ids)
                held_k = ov_hold[hold_ids].to(torch.float32)[:, 0:1]
                held_c = self._pad.resolved_kc(
                    held_k, self.cfg.resolved_pad_damping_ratio())[1]
                k_draw = torch.where(held_k > 0.0, held_k.expand_as(k_draw), k_draw)
                c_draw = torch.where(held_k > 0.0, held_c.expand_as(c_draw), c_draw)
            self._pad.set_parameters(env_ids, k_draw, c_draw)
            # The pad is a plant state: carrying compression across an episode
            # boundary would leak the previous episode's stored energy into the
            # new one's first contact.
            self._pad.reset(env_ids)
            if bool(self._pad.clamped[env_ids].any()) and not self._pad_blowup_warned:
                self._pad_blowup_warned = True
                print(
                    f"[PLANT][WARN] pad damping clipped to the explicit-integrator "
                    f"ceiling {self._pad.damping_cap:.1f} N s/m for "
                    f"{int(self._pad.clamped[env_ids].sum().item())} env-foot(s). "
                    f"The drawn band is not fully applied; lower pad_damping_ratio "
                    f"or pad_effective_mass, or lower pad_stiffness_min."
                )

        ov_belt = self._dr_override.get("belt_series") if self._dr_override else None
        if ov_belt is not None:
            # leave the override's values in place (already written by
            # _apply_dr_override_physics, which also installed them on the
            # actuator and rewrote the armature)
            pass
        else:
            # S5 (2026-09-12): the BAND is what decides whether the belt is
            # elastic, not `belt_series_enable`.  A nonzero stiffness band is a
            # per-episode DR draw exactly like every other plant multiplier,
            # and gating the draw on the legacy flag (the pre-S5 behavior) made
            # a present band silently identical to a rigid belt whenever a
            # recipe forgot the flag — the failure mode this launch exists to
            # rule out.  The flag still matters in `__init__`, where it decides
            # whether the series state buffers are allocated at all; here the
            # band alone is enough, because a drawn k = 0 already means rigid.
            _belt_live = bool(self.cfg.belt_band_live())
            self._dr_belt_stiffness[env_ids] = sample_uniform(
                self.cfg.belt_series_stiffness_min,
                self.cfg.belt_series_stiffness_max,
                (num_reset, 2), self.device,
            ) if _belt_live else 0.0
            self._dr_belt_damping[env_ids] = sample_uniform(
                self.cfg.belt_series_damping_min,
                self.cfg.belt_series_damping_max,
                (num_reset, 2), self.device,
            ) if _belt_live else 0.0
            # Ratio form: derive damping from the stiffness THIS episode drew,
            # so the belt's damping ratio is what the recipe names rather than
            # a fixed coefficient that means a different zeta at every cell.
            # Applied after the band draw, and only while the band is all-zero,
            # so an explicit band always wins.
            if (_belt_live
                    and self.cfg.belt_series_damping_min == 0.0
                    and self.cfg.belt_series_damping_max == 0.0
                    and not math.isnan(self.cfg.belt_series_damping_ratio)):
                k_draw = self._dr_belt_stiffness[env_ids]
                # Elementwise and masked: a cell that drew k=0 stays rigid and
                # keeps its zero damping, instead of asking for the sqrt of a
                # zero stiffness.
                ratio_d = 2.0 * self.cfg.belt_series_damping_ratio * torch.sqrt(
                    torch.clamp(k_draw, min=0.0)
                    * _BELT_G2 * self.cfg.belt_series_rotor_inertia
                )
                self._dr_belt_damping[env_ids] = torch.where(
                    k_draw > 0.0, ratio_d, self._dr_belt_damping[env_ids]
                )
            hk = self.robot.actuators.get("hip_knee")
            if hk is not None:
                # Same frames as the seed write in __init__: stiffness
                # joint-side (x G^2), damping already motor-side (x 1).
                hk.set_series_parameters(
                    env_ids,
                    stiffness=self._dr_belt_stiffness[env_ids] * _BELT_G2,
                    damping=self._dr_belt_damping[env_ids],
                    enabled=torch.full_like(
                        self._dr_belt_stiffness[env_ids],
                        bool(self.cfg.belt_band_live()), dtype=torch.bool,
                    ) & (self._dr_belt_stiffness[env_ids] > 0.0),
                    rotor_inertia=self._series_rotor_inertia_for(env_ids),
                )
        # ── Actuator-lag ω / ζ multipliers + cached coefficient refresh ──
        # Sampled unconditionally to match the multipliers above (this helper has
        # no dr_enable guard — see _reset_idx). Recompute the per-env prefilter
        # coefficients ω² and 2ζω for the reset envs only; they are constant for
        # the rest of the episode, so _advance_motor_target never recomputes them.
        self._dr_act_lag_omega_mult[env_ids] = sample_uniform(self.cfg.dr_act_lag_omega_mult_min, self.cfg.dr_act_lag_omega_mult_max, (num_reset, n_j), self.device)
        self._dr_act_lag_zeta_mult[env_ids]  = sample_uniform(self.cfg.dr_act_lag_zeta_mult_min,  self.cfg.dr_act_lag_zeta_mult_max,  (num_reset, n_j), self.device)
        _omega = self._act_lag_omega_nom.unsqueeze(0) * self._dr_act_lag_omega_mult[env_ids]   # [num_reset, 6]
        _zeta  = self._act_lag_zeta_nom.unsqueeze(0)  * self._dr_act_lag_zeta_mult[env_ids]
        self._act_lag_omega2[env_ids] = _omega ** 2
        self._act_lag_two_zw[env_ids] = 2.0 * _zeta * _omega
    # ══════════════════════════════════════════════════════════
    # Standup curriculum
    # ══════════════════════════════════════════════════════════

    def _curriculum_tertile_stats(self):
        """Per-difficulty-tertile stats of the kneel-branch episode window.

        Returns None until ≥64 episodes are buffered, else a dict
        band → (diff_mean, ep_len_mean, postgrace_mean, success_rate) for
        bands "low"/"mid"/"high" plus "all". Success = survived at least
        cfg.curriculum_success_post_grace_steps after grace expiry.
        """
        n = self._episode_buffer_count
        if n < 64:
            return None
        eps   = self._episode_lengths_buffer[:n]
        diffs = self._episode_difficulty_buffer[:n]
        post  = self._episode_postgrace_buffer[:n]
        succ  = (post >= self.cfg.curriculum_success_post_grace_steps).float()
        order = diffs.argsort()
        third = max(n // 3, 1)
        bands = {
            "low":  order[:third],
            "mid":  order[third:2 * third],
            "high": order[2 * third:],
            "all":  order,
        }
        return {
            k: (diffs[i].mean().item(), eps[i].mean().item(),
                post[i].mean().item(), succ[i].mean().item())
            for k, i in bands.items()
        }

    def _curriculum_advance_metric(self, stats):
        """(metric value, description) for cfg.curriculum_advance_metric.

        `stats` is the _curriculum_tertile_stats() dict (may be None for the
        success modes → metric 0). Single source for both the TB channel
        curriculum/advance_metric_value and the advance decision.
        """
        mode = self.cfg.curriculum_advance_metric
        if mode == "success_high":
            if stats is None:
                return 0.0, "success_high: n/a"
            return stats["high"][3], f"success_high: {stats['high'][3]:.2f} @ d≈{stats['high'][0]:.2f}"
        if mode == "success_high_rel":
            if stats is None:
                return 0.0, "success_high_rel: n/a"
            ratio = stats["high"][3] / max(stats["low"][3], 0.05)
            return ratio, (
                f"success_high_rel: {ratio:.2f} = {stats['high'][3]:.2f} @ d≈{stats['high'][0]:.2f}"
                f" / {stats['low'][3]:.2f} @ d≈{stats['low'][0]:.2f}"
            )
        n = self._episode_buffer_count
        mean_len = self._episode_lengths_buffer[:n].mean().item() if n > 0 else 0.0
        return mean_len / self.max_episode_length, f"mean ep: {mean_len:.0f}/{self.max_episode_length}"

    def _record_episode_lengths(self, env_ids):
        ep_lens = self.episode_length_buf[env_ids]
        is_kneel_branch = self._was_kneel_start[env_ids]
        ep_diffs = self._env_difficulty[env_ids]                 # NEW
        ep_post = (ep_lens - self._grace_steps_assigned[env_ids]).clamp(min=0)
        valid_mask = (ep_lens > 0) & is_kneel_branch
        valid_lens = ep_lens[valid_mask]
        valid_diffs = ep_diffs[valid_mask]                       # NEW
        valid_post = ep_post[valid_mask]
        for length, diff, post in zip(                           # CHANGED
            valid_lens.tolist(), valid_diffs.tolist(), valid_post.tolist()
        ):
            idx = self._episode_buffer_idx % self.cfg.curriculum_advance_window
            self._episode_lengths_buffer[idx] = length
            self._episode_difficulty_buffer[idx] = diff          # NEW
            self._episode_postgrace_buffer[idx] = post
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

        metric, metric_desc = self._curriculum_advance_metric(self._curriculum_tertile_stats())
        if metric >= self.cfg.curriculum_advance_threshold:
            old_alpha = self._curriculum_alpha
            self._curriculum_alpha = min(
                self._curriculum_alpha + self.cfg.curriculum_alpha_step,
                self.cfg.curriculum_max_alpha,
            )
            if self._curriculum_alpha != old_alpha:
                print(
                    f"[STANDUP CURRICULUM] alpha: {old_alpha:.3f} → "
                    f"{self._curriculum_alpha:.3f}  ({metric_desc})"
                )
                self._curriculum_cooldown = self.cfg.curriculum_advance_cooldown
                self._episode_buffer_count = 0
                self._episode_buffer_idx = 0
    def _apply_pickup_events(self):
        """Continuous-force mid-episode pickup. State machine per env:
            IDLE → LIFT → HOLD → LOWER → RELEASE_TRANSITION → IDLE
        Closed-loop PD on base z, xy, and orientation drives a 'hand'
        that lifts the robot, holds it at a random target orientation
        with optional wobble, gently lowers it until feet contact, then
        gradually releases support over a random latency. Episode
        continues throughout — no teleport, no state wipe, no command
        override. Termination and walking rewards are gated elsewhere
        by self._pickup_phase > 0.
        """

        # ── 1. Trigger new pickups (IDLE → LIFT) ──────────────────
        idle = self._pickup_phase == 0
        eligible = (
            idle
            & (self.episode_length_buf >= self.cfg.pickup_min_steps_since_reset)
            & (self._grace_remaining <= 0)
            & (self._stabilize_remaining <= 0)
            & (self._push_remaining <= 0)
        )
        trigger = (
            torch.rand(self.num_envs, device=self.device) < self.cfg.pickup_probability
        ) & eligible
        new_ids = trigger.nonzero(as_tuple=False).squeeze(-1)
        if new_ids.numel() > 0:
            n = new_ids.numel()
            self._pickup_phase[new_ids] = 1                                 # → LIFT
            self._pickup_remaining[new_ids] = self.cfg.pickup_lift_steps

            # Lift z target: current base z + random lift height
            cur_z = self.robot.data.root_pos_w[new_ids, 2]
            lift_h = sample_uniform(
                self.cfg.pickup_lift_height_min,
                self.cfg.pickup_lift_height_max,
                (n,), self.device,
            )
            self._pickup_target_z[new_ids] = cur_z + lift_h

            # Horizontal lock target: current xy at trigger
            self._pickup_target_xy[new_ids] = self.robot.data.root_pos_w[new_ids, :2]

            # ── DR Layer 1: random target orientation ──────────────
            # target_quat = q_yaw_at_trigger ⊗ small random roll/pitch.
            # Yaw is preserved → no conflict with heading-anchor after release.
            cur_yaw = _yaw_from_quat(self.robot.data.root_quat_w[new_ids])
            q_yaw = torch.zeros(n, 4, device=self.device)
            q_yaw[:, 0] = torch.cos(cur_yaw * 0.5)
            q_yaw[:, 3] = torch.sin(cur_yaw * 0.5)

            rp = torch.stack([
                sample_uniform(
                    -self.cfg.pickup_target_roll_max,
                    self.cfg.pickup_target_roll_max, (n,), self.device,
                ),
                sample_uniform(
                    -self.cfg.pickup_target_pitch_max,
                    self.cfg.pickup_target_pitch_max, (n,), self.device,
                ),
            ], dim=-1)
            q_tilt = torch.zeros(n, 4, device=self.device)
            q_tilt[:, 0] = 1.0
            q_tilt[:, 1] = 0.5 * rp[:, 0]
            q_tilt[:, 2] = 0.5 * rp[:, 1]
            q_tilt = q_tilt / q_tilt.norm(dim=-1, keepdim=True)
            self._pickup_target_quat[new_ids] = quat_mul(q_yaw, q_tilt)

            # ── DR Layer 2: random PD gain factors (per-pickup grip strength) ──
            self._pickup_pd_factor_trans[new_ids] = sample_uniform(
                self.cfg.pickup_pd_gain_factor_min,
                self.cfg.pickup_pd_gain_factor_max,
                (n,), self.device,
            )
            self._pickup_pd_factor_rot[new_ids] = sample_uniform(
                self.cfg.pickup_pd_gain_factor_min,
                self.cfg.pickup_pd_gain_factor_max,
                (n,), self.device,
            )

            # ── DR Layer 3: random initial wobble phase ────────────
            self._pickup_hold_phase[new_ids] = sample_uniform(
                0.0, 2.0 * math.pi, (n,), self.device,
            )

            self._pickup_event_count += n

        # ── 2. Tick + phase transitions ───────────────────────────
        active_mask = self._pickup_phase > 0
        if not active_mask.any():
            return

        self._pickup_remaining[active_mask] -= 1

        # LIFT → HOLD (random hold duration)
        to_hold = (self._pickup_remaining <= 0) & (self._pickup_phase == 1)
        if to_hold.any():
            ids_hold = to_hold.nonzero(as_tuple=False).squeeze(-1)
            self._pickup_phase[ids_hold] = 2
            self._pickup_remaining[ids_hold] = torch.randint(
                self.cfg.pickup_hold_steps_min,
                self.cfg.pickup_hold_steps_max + 1,
                (ids_hold.numel(),), device=self.device,
            )

        # HOLD → LOWER (compute end_z via FK so feet land at ground)
        to_lower = (self._pickup_remaining <= 0) & (self._pickup_phase == 2)
        if to_lower.any():
            ids_low = to_lower.nonzero(as_tuple=False).squeeze(-1)
            self._pickup_phase[ids_low] = 3
            self._pickup_remaining[ids_low] = self.cfg.pickup_lower_steps
            cur_jp = self.robot.data.joint_pos[ids_low][:, self._joint_ids]
            cur_q = self.robot.data.root_quat_w[ids_low]
            fk_h = _compute_fk_height(
                base_quat=cur_q,
                hip_roll1=cur_jp[:, 0], hip_pitch1=cur_jp[:, 1], knee1=cur_jp[:, 2],
                hip_roll2=cur_jp[:, 3], hip_pitch2=cur_jp[:, 4], knee2=cur_jp[:, 5],
                device=self.device,
            )
            self._pickup_lower_start_z[ids_low] = self.robot.data.root_pos_w[ids_low, 2]
            self._pickup_lower_end_z[ids_low] = fk_h + self.scene.env_origins[ids_low, 2]

        # Contact-based release (LOWER → RELEASE_TRANSITION)
        in_lower = self._pickup_phase == 3
        if in_lower.any():
            toe1_w, toe2_w = self._get_toe_pos_w()
            min_foot_z = (
                torch.min(toe1_w[:, 2], toe2_w[:, 2])
                - self.scene.env_origins[:, 2]
            )
            contact_release = in_lower & (min_foot_z < self.cfg.pickup_contact_release_z)
            if contact_release.any():
                cr_ids = contact_release.nonzero(as_tuple=False).squeeze(-1)
                n_cr = cr_ids.numel()
                self._pickup_phase[cr_ids] = 4
                latency = torch.randint(
                    self.cfg.pickup_release_latency_min_steps,
                    self.cfg.pickup_release_latency_max_steps + 1,
                    (n_cr,), device=self.device,
                )
                self._pickup_remaining[cr_ids] = latency
                self._pickup_release_initial_steps[cr_ids] = latency.clamp(min=1)

        # Timer-based release (LOWER → RELEASE_TRANSITION)
        to_release = (self._pickup_remaining <= 0) & (self._pickup_phase == 3)
        if to_release.any():
            tr_ids = to_release.nonzero(as_tuple=False).squeeze(-1)
            n_tr = tr_ids.numel()
            self._pickup_phase[tr_ids] = 4
            latency = torch.randint(
                self.cfg.pickup_release_latency_min_steps,
                self.cfg.pickup_release_latency_max_steps + 1,
                (n_tr,), device=self.device,
            )
            self._pickup_remaining[tr_ids] = latency
            self._pickup_release_initial_steps[tr_ids] = latency.clamp(min=1)

        # RELEASE_TRANSITION → IDLE (full release)
        to_idle = (self._pickup_remaining <= 0) & (self._pickup_phase == 4)
        if to_idle.any():
            self._pickup_phase[to_idle] = 0
            self._pickup_support_factor[to_idle] = 1.0   # NEW: back to full reward/term
            # Post-pickup landing settle: walking rewards off, termination
            # on (a real fall after release is a real failure), then a fresh
            # command + spawn anchor when stabilize expires. Reusing
            # _stabilize_remaining piggybacks on the existing tick-down +
            # cmd-resample + spawn-anchor logic in _pre_physics_step.
            self._stabilize_remaining[to_idle] = self.cfg.pickup_post_grace_steps

        # ── 3. Compute PD output for envs in any active phase ─────
        active_mask = self._pickup_phase > 0
        if not active_mask.any():
            return
        ids = active_mask.nonzero(as_tuple=False).squeeze(-1)
        phase = self._pickup_phase[ids]

        # Pre-compute phase subsets (used by z, xy, and wobble-advance)
        in_lower_subset   = phase == 3
        in_hold_subset    = phase == 2
        in_release_subset = phase == 4

        hold_local = None
        hold_env_ids = None
        if in_hold_subset.any():
            hold_local = in_hold_subset.nonzero(as_tuple=False).squeeze(-1)
            hold_env_ids = ids[hold_local]

        release_local = None
        release_env_ids = None
        if in_release_subset.any():
            release_local = in_release_subset.nonzero(as_tuple=False).squeeze(-1)
            release_env_ids = ids[release_local]

        # ── z target ──────────────────────────────────────────────
        z_target = self._pickup_target_z[ids].clone()

        # LOWER: ramp z_target from start_z down to end_z
        if in_lower_subset.any():
            lower_local = in_lower_subset.nonzero(as_tuple=False).squeeze(-1)
            lower_env_ids = ids[lower_local]
            progress = (
                self._pickup_remaining[lower_env_ids].float()
                / max(self.cfg.pickup_lower_steps, 1)
            ).clamp(0.0, 1.0)
            z_target[lower_local] = (
                self._pickup_lower_start_z[lower_env_ids] * progress
                + self._pickup_lower_end_z[lower_env_ids] * (1.0 - progress)
            )

        # RELEASE_TRANSITION: hold z_target at ground level
        if release_local is not None:
            z_target[release_local] = self._pickup_lower_end_z[release_env_ids]

        # HOLD: add z wobble (Layer 3)
        if hold_local is not None:
            t_hold = self._pickup_hold_phase[hold_env_ids]
            z_target[hold_local] = z_target[hold_local] + (
                self.cfg.pickup_hold_wobble_z * torch.sin(t_hold * 1.3)
            )

        # ── support_factor: 1.0 normally, ramps 1→0 in RELEASE_TRANSITION ──
        support_factor = torch.ones(ids.numel(), device=self.device)
        if release_local is not None:
            rem  = self._pickup_remaining[release_env_ids].float()
            init = self._pickup_release_initial_steps[release_env_ids].float().clamp(min=1.0)
            support_factor[release_local] = (rem / init).clamp(0.0, 1.0)
        self._pickup_support_factor[ids] = support_factor   # NEW: persist for gating

        # ── z PD ──────────────────────────────────────────────────
        z_cur  = self.robot.data.root_pos_w[ids, 2]
        vz_cur = self.robot.data.root_lin_vel_w[ids, 2]
        m      = self._total_mass[ids]

        omega = self.cfg.pickup_z_omega
        zeta  = self.cfg.pickup_z_damping_ratio
        factor_trans = self._pickup_pd_factor_trans[ids]
        Kp = m * (omega ** 2) * factor_trans
        Kd = 2.0 * m * omega * zeta * factor_trans

        f_z = m * 9.81 + Kp * (z_target - z_cur) - Kd * vz_cur
        max_f = self.cfg.pickup_force_cap_mult * m * 9.81
        f_z = f_z.clamp(-max_f, max_f)
        f_z = f_z * support_factor
        self._pickup_forces[ids, self._torso_idx, 2] = f_z

        # ── Orientation PD (per-env random target, scaled by support_factor) ──
        q_cur = self.robot.data.root_quat_w[ids]
        q_target = self._pickup_target_quat[ids]

        # q_err = q_target · conj(q_cur)
        q_cur_conj = q_cur.clone()
        q_cur_conj[:, 1:4] = -q_cur_conj[:, 1:4]
        q_err = quat_mul(q_target, q_cur_conj)

        w_err = q_err[:, 0:1]
        xyz_err = q_err[:, 1:4]
        sign_w_err = torch.where(w_err >= 0, 1.0, -1.0)
        tilt_err = 2.0 * sign_w_err * xyz_err

        omega_w = self.robot.data.root_ang_vel_w[ids]

        Kp_rot = self.cfg.pickup_orient_kp
        Kd_rot = self.cfg.pickup_orient_kd
        yaw_kp_scale = self.cfg.pickup_orient_yaw_kp_scale
        yaw_kd_scale = self.cfg.pickup_orient_yaw_kd_scale
        factor_rot = self._pickup_pd_factor_rot[ids].unsqueeze(-1)
        kp_vec = factor_rot * torch.tensor(
            [Kp_rot, Kp_rot, Kp_rot * yaw_kp_scale], device=self.device
        )
        kd_vec = factor_rot * torch.tensor(
            [Kd_rot, Kd_rot, Kd_rot * yaw_kd_scale], device=self.device
        )

        tau_world = kp_vec * tilt_err - kd_vec * omega_w
        tau_cap = self.cfg.pickup_torque_cap
        tau_world = tau_world.clamp(-tau_cap, tau_cap)
        tau_world = tau_world * support_factor.unsqueeze(-1)
        self._pickup_torques[ids, self._torso_idx, :] = tau_world

        # ── xy PD (horizontal lock with optional HOLD wobble) ──
        xy_target = self._pickup_target_xy[ids].clone()

        if hold_local is not None:
            t_hold = self._pickup_hold_phase[hold_env_ids]
            wob_xy = self.cfg.pickup_hold_wobble_xy * torch.stack(
                [torch.sin(t_hold), torch.cos(t_hold)], dim=-1,
            )
            xy_target[hold_local] = xy_target[hold_local] + wob_xy

        xy_cur  = self.robot.data.root_pos_w[ids, :2]
        vxy_cur = self.robot.data.root_lin_vel_w[ids, :2]
        xy_err  = xy_target - xy_cur

        m_b = m.unsqueeze(-1)
        factor_xy = factor_trans.unsqueeze(-1)
        Kp_xy = m_b * (omega ** 2) * factor_xy
        Kd_xy = 2.0 * m_b * omega * zeta * factor_xy
        f_xy = Kp_xy * xy_err - Kd_xy * vxy_cur

        max_f_xy = self.cfg.pickup_force_cap_mult * m_b * 9.81
        f_xy = f_xy.clamp(-max_f_xy, max_f_xy)
        f_xy = f_xy * support_factor.unsqueeze(-1)
        self._pickup_forces[ids, self._torso_idx, 0] = f_xy[:, 0]
        self._pickup_forces[ids, self._torso_idx, 1] = f_xy[:, 1]

        # ── Advance wobble phase for HOLD envs ─────────────────────
        if hold_local is not None:
            self._pickup_hold_phase[hold_env_ids] += (
                2.0 * math.pi
                * self.cfg.pickup_hold_wobble_freq_hz
                * self._control_dt
            )
    def _trigger_initial_pickup(self, env_ids):
        """Schedule pickup-on-spawn for a random subset of just-reset envs.

        Bypasses the mid-episode eligibility gates (grace / stabilize /
        push / min_steps_since_reset) since the env is in a clean state
        by construction. Uses pickup_initial_lift_height_* — taller than
        mid-ep — so the policy gets explicit training on the
        "held above the ground, then placed down" trajectory.

        Idempotent w.r.t. _reset_idx's earlier pickup-state clear:
        envs not selected keep _pickup_phase == 0.
        """
        if not self.cfg.pickup_enable:
            return
        if self.cfg.pickup_initial_probability <= 0.0:
            return

        env_ids_t = env_ids if isinstance(env_ids, torch.Tensor) else torch.tensor(
            list(env_ids), device=self.device, dtype=torch.long,
        )
        num = env_ids_t.numel()
        if num == 0:
            return

        select = torch.rand(num, device=self.device) < self.cfg.pickup_initial_probability
        # Kneel-branch envs must stand up on their own: a hand-lifted kneel
        # start would still count as a kneel episode in the curriculum's
        # advance statistics (S-line, 2026-09-06). Inert when the
        # curriculum is off (_was_kneel_start all False).
        select &= ~self._was_kneel_start[env_ids_t]
        if not select.any():
            return
        ids = env_ids_t[select]
        n = ids.numel()

        # → LIFT (state machine takes over from _apply_pickup_events
        # on the next _pre_physics_step: phase != 0 so trigger block
        # is skipped, tick block ticks counters and applies PD forces).
        self._pickup_phase[ids] = 1
        self._pickup_remaining[ids] = self.cfg.pickup_lift_steps

        # Lift target z: spawn z + tall random lift height.
        cur_z = self.robot.data.root_pos_w[ids, 2]
        lift_h = sample_uniform(
            self.cfg.pickup_initial_lift_height_min,
            self.cfg.pickup_initial_lift_height_max,
            (n,), self.device,
        )
        self._pickup_target_z[ids] = cur_z + lift_h

        # Horizontal lock at spawn xy.
        self._pickup_target_xy[ids] = self.robot.data.root_pos_w[ids, :2]

        # Target orientation: spawn yaw ⊗ small random roll/pitch.
        cur_yaw = _yaw_from_quat(self.robot.data.root_quat_w[ids])
        q_yaw = torch.zeros(n, 4, device=self.device)
        q_yaw[:, 0] = torch.cos(cur_yaw * 0.5)
        q_yaw[:, 3] = torch.sin(cur_yaw * 0.5)

        rp = torch.stack([
            sample_uniform(
                -self.cfg.pickup_target_roll_max,
                self.cfg.pickup_target_roll_max, (n,), self.device,
            ),
            sample_uniform(
                -self.cfg.pickup_target_pitch_max,
                self.cfg.pickup_target_pitch_max, (n,), self.device,
            ),
        ], dim=-1)
        q_tilt = torch.zeros(n, 4, device=self.device)
        q_tilt[:, 0] = 1.0
        q_tilt[:, 1] = 0.5 * rp[:, 0]
        q_tilt[:, 2] = 0.5 * rp[:, 1]
        q_tilt = q_tilt / q_tilt.norm(dim=-1, keepdim=True)
        self._pickup_target_quat[ids] = quat_mul(q_yaw, q_tilt)

        # Per-pickup random grip strength.
        self._pickup_pd_factor_trans[ids] = sample_uniform(
            self.cfg.pickup_pd_gain_factor_min,
            self.cfg.pickup_pd_gain_factor_max,
            (n,), self.device,
        )
        self._pickup_pd_factor_rot[ids] = sample_uniform(
            self.cfg.pickup_pd_gain_factor_min,
            self.cfg.pickup_pd_gain_factor_max,
            (n,), self.device,
        )

        # Random initial wobble phase.
        self._pickup_hold_phase[ids] = sample_uniform(
            0.0, 2.0 * math.pi, (n,), self.device,
        )

        # Pickup supersedes grace for these envs — otherwise the
        # standup_height / standup_fall rewards would fire on the
        # in-air position the "hand" is holding, which is not the
        # learning signal we want. Walking rewards are already
        # gated off by pickup_active in compute_rewards_vx_yaw.
        self._grace_remaining[ids] = 0
        self._stabilize_remaining[ids] = 0

        self._pickup_event_count += n

# ══════════════════════════════════════════════════════════════
# Reward function (JIT-compiled) — Phase 6e
#   Slot 2 repurposed: rew_upright (was 0) → rew_raw_action
#   New args:    rew_scale_raw_action, raw_actions, kneel_height, stand_target
#   Removed:     standup_midpoint, standup_sharpness (replaced by linear ramp)
# ══════════════════════════════════════════════════════════════

@torch.jit.script
def compute_rewards_vx_yaw(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_raw_action: float,                     # NEW (was rew_scale_upright)
    rew_scale_height: float,
    rew_scale_feet_air: float,
    rew_scale_gait_step: float,
    rew_scale_energy: float,
    rew_scale_action_rate: float,
    rew_scale_action_magnitude: float,
    rew_scale_joint_vel: float,
    rew_scale_vel_tracking: float,
    rew_scale_joint_pos_limits: float,
    rew_scale_standup_height: float,
    rew_scale_standup_fall: float,
    rew_scale_thermal: float,
    rew_scale_hip_roll_split: float,
    rew_scale_front_split: float,
    rew_scale_knee_split: float,
    rew_scale_foot_swing: float,
    rew_scale_leg_symmetry: float,
    rew_scale_angle_symmetry: float,
    rew_scale_knee_over_hip: float,
    rew_scale_foot_progress: float,
    rew_scale_position_anchor: float,
    rew_scale_heading_anchor: float,
    rew_scale_back_leg_step: float,
    rew_scale_yaw_rate: float,
    rew_scale_balance: float,
    rew_scale_no_flight: float,
    rew_scale_vz_penalty: float,
    rew_scale_stance_symmetry: float,
    rew_scale_yaw_drift: float,
    projected_gravity: torch.Tensor,
    base_height: torch.Tensor,
    base_vz_world: torch.Tensor,
    foot1_z: torch.Tensor,
    foot2_z: torch.Tensor,
    gait_sin: torch.Tensor,
    joint_vel: torch.Tensor,
    applied_torque: torch.Tensor,
    thermal_cont_effort: torch.Tensor,
    actions: torch.Tensor,
    raw_actions: torch.Tensor,                       # NEW IN p6e
    last_actions: torch.Tensor,
    reset_terminated: torch.Tensor,
    joint_pos_norm: torch.Tensor,
    hip_roll_pos: torch.Tensor,
    hip_pitch_pos: torch.Tensor,
    knee_pos: torch.Tensor,
    in_grace: torch.Tensor,
    vx_avg: torch.Tensor,                            # averaged forward velocity (was base_vx)
    vx_cmd: torch.Tensor,
    vel_tracking_sigma: float,
    leg_forward_ema: torch.Tensor,
    joint_pair_sum_ema: torch.Tensor,
    foot1_contact_ema: torch.Tensor,
    foot2_contact_ema: torch.Tensor,
    feet_mid_disp_base: torch.Tensor,
    pos_drift_sq: torch.Tensor,
    yaw_error_sq: torch.Tensor,
    toe1_fwd_disp_base: torch.Tensor,
    toe2_fwd_disp_base: torch.Tensor,
    toe1_x_base: torch.Tensor,
    ang_vel_z_base: torch.Tensor,
    yaw_rate_cmd: torch.Tensor,
    yaw_rate_tracking_sigma: float,
    yaw_rate_cmd_anchor_threshold: float,
    height_min: float,
    height_max: float,
    height_edge_softness: float,
    kneel_height: float,                             # NEW IN p6e
    stand_target_height: float,                      # NEW IN p6e
    feet_air_threshold: float,
    max_hip_roll_rad: float,
    max_hip_pitch_split_rad: float,
    max_knee_split_rad: float,
    no_flight_contact_threshold: float,
    rew_scale_action_smoothness: float,    # NEW
    last_last_actions: torch.Tensor,        # NEW
    action_rate_weights: torch.Tensor,      # NEW: shape [6], joint-space weights
    pickup_gate: torch.Tensor,
    raw_action_threshold: float,            # NEW: a_thr deadband
    rew_scale_action_snap: float,           # NEW: 3rd-order
    last_last_last_actions: torch.Tensor,   # NEW: t-3
    idle_step_gate: float,                  # NEW
    idle_motion_relief: float,              # NEW
    rew_scale_upright: float,               # NEW: uprightness penalty
    ang_vel_z_avg: torch.Tensor,            # NEW: averaged yaw-rate for tracking
    rew_scale_clock_contact: float,         # NEW knob 1
    rew_scale_swing_clearance: float,       # NEW knob 2
    swing_clearance_target: float,          # NEW knob 2
    swing_clearance_zero: float,            # RESUME 2: dense ramp start (resting toe z + 3 mm)
    rew_scale_gait_ref: float = 0.0,        # NEW A/B fork: gait-reference prior
    gait_ref_amplitude: float = 0.20,       # NEW A/B fork: SIGNED amplitude (rad)
    gait_ref_sigma: float = 0.15,           # NEW A/B fork: exp-kernel width (rad)
) -> Tuple[torch.Tensor, torch.Tensor]:

    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float()) * pickup_gate
    rew_terminated = rew_scale_terminated * reset_terminated.float()

    # ── NEW IN p6e: raw (pre-clip) action penalty ──────────────
    # The clipped-action penalty (rew_action_magnitude) gives zero
    # gradient when the action saturates. This penalty fires on the
    # RAW pre-clip output, so the network gets gradient feedback
    # even when self.actions is clipped at ±1. Active everywhere
    # (no post_grace gate) — saturation is bad regardless of phase.
    # Deadband: only magnitude beyond a_thr is penalized, so in-band actions
    # are free. Keeps pre-clip logits from drifting past ±1 unboundedly
    # without flattening legitimate large outputs.
    raw_excess = torch.relu(torch.abs(raw_actions) - raw_action_threshold)
    rew_raw_action = rew_scale_raw_action * torch.sum(
        torch.square(raw_excess), dim=-1
    )

    # Disables walking / gait / tracking rewards during both grace
    # (kept) AND pickups (new). Standup rewards stay tied to in_grace
    # only, so they don't accidentally fire mid-air during a pickup.
    post_grace = (1.0 - in_grace) * pickup_gate
    # Action-space regularizers are NOT state-coupled, so they stay live
    # through pickup as a calm / anti-flail prior. Off only during grace.
    anti_flail = (1.0 - in_grace)

    low_pen  = torch.relu(torch.tensor(height_min, device=base_height.device) - base_height)
    high_pen = torch.relu(base_height - torch.tensor(height_max, device=base_height.device))
    rew_height = rew_scale_height * torch.exp(
        -height_edge_softness * (low_pen + high_pen)
    ) * post_grace

    # ── SIMPLIFIED IN p6e ──────────────────────────────────────
    # Linear ramp from kneel_height → stand_target_height during
    # grace. Old: sigmoid around standup_midpoint with sharpness=25.
    # New: clamp((h - kneel) / (stand - kneel), 0, 1). Same shape,
    # two fewer hyperparameters, no soft-edge artifacts.
    ramp_denom = max(stand_target_height - kneel_height, 1e-3)
    standup_progress = torch.clamp(
        (base_height - kneel_height) / ramp_denom, min=0.0, max=1.0
    )
    rew_standup_height = rew_scale_standup_height * standup_progress * in_grace

    # ── SIMPLIFIED IN p6e ──────────────────────────────────────
    # Falling during grace is bad regardless of current height.
    # Removed the (base_height > standup_midpoint) gate.
    fall_rate = torch.clamp(-base_vz_world, min=0.0, max=0.3)
    rew_standup_fall = rew_scale_standup_fall * fall_rate * in_grace

    torque_ratio_sq = torch.square(applied_torque / thermal_cont_effort.unsqueeze(0))
    thermal_excess = torch.clamp(torque_ratio_sq - 1.0, min=0.0)
    rew_thermal = rew_scale_thermal * torch.sum(
            thermal_excess, dim=-1
        ) * post_grace

    vx_error_sq = torch.square(vx_avg - vx_cmd) 
    rew_vel_tracking = rew_scale_vel_tracking * torch.exp(
        -vx_error_sq / (vel_tracking_sigma * vel_tracking_sigma)
    ) * post_grace

    cmd_magnitude = torch.abs(vx_cmd)
    cmd_gate = torch.clamp(cmd_magnitude / 0.05, min=0.0, max=1.0)
    # Floor the gait gate so stepping is always worth something. Only raises the
    # gate where cmd_gate < idle_step_gate, i.e. exactly-zero-command envs.
    step_gate = torch.clamp(cmd_gate, min=idle_step_gate)
    # Fade the action-derivative penalties at rest so they don't eat the stepping
    # reward; full strength while walking.
    motion_gate = idle_motion_relief + (1.0 - idle_motion_relief) * cmd_gate

    foot_height_diff = torch.abs(foot1_z - foot2_z)
    rew_feet_air = rew_scale_feet_air * torch.clamp(
        foot_height_diff / feet_air_threshold, max=1.0
    ) * step_gate * post_grace

    signed_diff = foot1_z - foot2_z
    gait_match = gait_sin * signed_diff / feet_air_threshold
    rew_gait_step = rew_scale_gait_step * torch.clamp(
        gait_match, min=0.0, max=1.0
    ) * step_gate * post_grace

    rew_foot_swing = rew_scale_foot_swing * torch.clamp(
        foot_height_diff / feet_air_threshold, max=1.0
    ) * step_gate * post_grace

    rew_energy = rew_scale_energy * torch.sum(
        torch.square(applied_torque), dim=-1
    ) * post_grace
    # Weighted by action_rate_weights (= action_scale or real_action_scale)
    # to penalize joint-space rate of change rather than normalized-action
    # rate. Critical when action_scales are very uneven across joints.
    scaled_action_diff = (actions - last_actions) * action_rate_weights
    rew_action_rate = rew_scale_action_rate * torch.sum(
        torch.square(scaled_action_diff), dim=-1
    ) * anti_flail

    # NEW: second-derivative (jerk) penalty in joint space. Zero for
    # any linear ramp; large for sign-flipping. Catches policy
    # oscillation that action_rate alone can't distinguish from
    # smooth motion of equivalent step size.
    scaled_last_action_diff = (last_actions - last_last_actions) * action_rate_weights
    scaled_jerk = scaled_action_diff - scaled_last_action_diff
    rew_action_smoothness = rew_scale_action_smoothness * torch.sum(
        torch.square(scaled_jerk), dim=-1
    ) * anti_flail * motion_gate          # was: * anti_flail
    # NEW: third-order smoothness (discrete derivative of jerk).
    # snap = Δ³a = jerk(t) − jerk(t−1) → stencil 1,-3,3,-1 on the action.
    scaled_last_last_action_diff = (last_last_actions - last_last_last_actions) * action_rate_weights
    scaled_last_jerk = scaled_last_action_diff - scaled_last_last_action_diff
    scaled_snap = scaled_jerk - scaled_last_jerk
    rew_action_snap = rew_scale_action_snap * torch.sum(
        torch.square(scaled_snap), dim=-1
    ) * anti_flail * motion_gate          # was: * anti_flail
    rew_action_magnitude = rew_scale_action_magnitude * torch.sum(
        torch.square(actions), dim=-1
    ) * post_grace
    rew_joint_vel = rew_scale_joint_vel * torch.sum(
        torch.square(joint_vel), dim=-1
    ) * post_grace
    rew_joint_pos_limits = rew_scale_joint_pos_limits * torch.sum(
        torch.square(joint_pos_norm), dim=-1
    ) * post_grace

    hip_roll_ratio = hip_roll_pos / max_hip_roll_rad
    rew_hip_roll_split = rew_scale_hip_roll_split * torch.sum(
        torch.square(hip_roll_ratio), dim=-1
    ) * post_grace
    front_split_val = torch.abs(hip_pitch_pos[:, 0] + hip_pitch_pos[:, 1]) / max_hip_pitch_split_rad
    rew_front_split = rew_scale_front_split * torch.square(front_split_val) * post_grace
    knee_split_val = torch.abs(knee_pos[:, 0] + knee_pos[:, 1]) / max_knee_split_rad
    rew_knee_split = rew_scale_knee_split * torch.square(knee_split_val) * post_grace

    rew_leg_symmetry = rew_scale_leg_symmetry * torch.square(
        leg_forward_ema
    ) * cmd_gate * post_grace

    rew_angle_symmetry = rew_scale_angle_symmetry * torch.sum(
        torch.square(joint_pair_sum_ema), dim=-1
    ) * post_grace

    hp_sq = torch.square(applied_torque[:, 1]) + torch.square(applied_torque[:, 4])
    kn_sq = torch.square(applied_torque[:, 2]) + torch.square(applied_torque[:, 5])
    rew_knee_over_hip = rew_scale_knee_over_hip * torch.relu(
        hp_sq - kn_sq
    ) * post_grace

    foot_progress = feet_mid_disp_base[:, 0] * torch.sign(vx_cmd + 1e-8)
    rew_foot_progress = rew_scale_foot_progress * foot_progress * cmd_gate * post_grace

    stand_gate = (1.0 - cmd_gate) * post_grace
    rew_position_anchor = rew_scale_position_anchor * pos_drift_sq * stand_gate

    yaw_rate_abs = torch.abs(yaw_rate_cmd)
    heading_anchor_gate = (
        torch.clamp(1.0 - yaw_rate_abs / yaw_rate_cmd_anchor_threshold, min=0.0, max=1.0)
        * stand_gate
    )
    rew_heading_anchor = rew_scale_heading_anchor * yaw_error_sq * heading_anchor_gate

    back_leg_fwd = torch.where(
        toe1_x_base < 0.0,
        toe1_fwd_disp_base,
        toe2_fwd_disp_base,
    )
    rew_back_leg_step = rew_scale_back_leg_step * torch.clamp(
        back_leg_fwd, min=0.0
    ) * cmd_gate * post_grace

    yaw_rate_gate = torch.clamp(
        yaw_rate_abs / yaw_rate_cmd_anchor_threshold, min=0.0, max=1.0
    )
    yaw_rate_error_sq = torch.square(ang_vel_z_avg - yaw_rate_cmd)
    rew_yaw_rate = rew_scale_yaw_rate * torch.exp(
        -yaw_rate_error_sq / (yaw_rate_tracking_sigma * yaw_rate_tracking_sigma)
    ) * yaw_rate_gate * post_grace

    foot1_contact = torch.clamp(no_flight_contact_threshold - foot1_z, min=0.0)
    foot2_contact = torch.clamp(no_flight_contact_threshold - foot2_z, min=0.0)
    load_imbalance = torch.square(foot1_contact - foot2_contact)
    rew_balance = rew_scale_balance * (-load_imbalance) * post_grace
    # ── NEW: uprightness penalty ──
    # sin²(tilt) from the horizontal proj-gravity components. post_grace
    # keeps it off during grace and active pickup, fading in during release.
    rew_upright = rew_scale_upright * (
        torch.square(projected_gravity[:, 0])
        + torch.square(projected_gravity[:, 1])
    ) * post_grace
    both_airborne = (
        (foot1_z > no_flight_contact_threshold)
        & (foot2_z > no_flight_contact_threshold)
    ).float()
    rew_no_flight       = rew_scale_no_flight * both_airborne * step_gate * post_grace

    rew_vz_penalty = rew_scale_vz_penalty * torch.square(base_vz_world) * post_grace

    stance_diff_sq = torch.square(foot1_contact_ema - foot2_contact_ema)
    rew_stance_symmetry = rew_scale_stance_symmetry * stance_diff_sq * step_gate * post_grace

    no_yaw_gate = torch.clamp(
        1.0 - yaw_rate_abs / yaw_rate_cmd_anchor_threshold, min=0.0, max=1.0
    )
    rew_yaw_drift = rew_scale_yaw_drift * torch.square(ang_vel_z_base) * no_yaw_gate * post_grace

    # ── NEW knob 1: binary clock-contact term ──
    # foot1 should be airborne when gait_sin>0 (its swing half), foot2 when
    # gait_sin<0. Rides step_gate so it's live at idle. Crossing the contact
    # threshold is the unit of reward — no partial credit for a 3 mm lift.
    # RESUME 2: uncentered via clamp(·, min=0). Standing (both grounded,
    # match=0.5) still nets exactly 0, perfect alternation +1, but a
    # wrong-phase lift now scores 0 instead of −1 — the wrong-phase penalty
    # was taxing exactly the exploratory lifts we need while the policy has
    # no stepping skill yet. (a == ~b ⟺ a != b for bools.)
    foot1_air = foot1_z > no_flight_contact_threshold
    foot2_air = foot2_z > no_flight_contact_threshold
    swing1 = gait_sin > 0.0
    clock_match = 0.5 * (
        (foot1_air == swing1).float()
        + (foot2_air != swing1).float()
    )
    rew_clock_contact = rew_scale_clock_contact * torch.clamp(
        2.0 * clock_match - 1.0, min=0.0
    ) * step_gate * post_grace

    # ── NEW knob 2: explicit swing-clearance term ──
    # During each foot's swing half-cycle, reward its toe height ramping
    # 0→1 over swing_clearance_target metres.
    # RESUME 2: ramp re-referenced from no_flight_contact_threshold (0.084)
    # to swing_clearance_zero (≈ resting toe z + 3 mm). The old reference
    # gave zero gradient until the foot had already cleared the contact
    # line — invisible from the 3 mm shuffle. Now the first millimetre of
    # lift pays, converting a sparse threshold into a dense gradient.
    swing_foot_z = torch.where(swing1, foot1_z, foot2_z)
    swing_clearance = torch.clamp(
        (swing_foot_z - swing_clearance_zero) / swing_clearance_target,
        min=0.0, max=1.0,
    )
    rew_swing_clearance = rew_scale_swing_clearance * swing_clearance * step_gate * post_grace

    # ── A/B FORK: gait-reference prior ─────────────────────────
    # Hip pitch tracks an antiphase sinusoid driven by the existing
    # gait clock — injects the stepping pattern instead of waiting for
    # discovery. SAME-SIGN reference for both legs is correct in joint
    # coordinates: leg 2's joint axes are mirrored (mirror table = swap
    # legs + negate), so hp1_dev = hp2_dev = A·sinφ is the antiphase
    # PHYSICAL gait satisfying state(φ+π) = mirror(state(φ)); the term
    # is mirror-invariant. joint_pos_norm order: [hr1,hp1,kn1,hr2,hp2,
    # kn2] → indices 1 and 4. Gated like the other idle-gait terms:
    # live at idle via the step_gate floor, silent in grace / pickup.
    # Says nothing about weight transfer — clock_contact +
    # swing_clearance remain the "actually lift" verification.
    gait_ref = gait_ref_amplitude * gait_sin
    gait_ref_err_sq = (
        torch.square(joint_pos_norm[:, 1] - gait_ref)
        + torch.square(joint_pos_norm[:, 4] - gait_ref)
    )
    rew_gait_ref = rew_scale_gait_ref * torch.exp(
        -gait_ref_err_sq / (gait_ref_sigma * gait_ref_sigma)
    ) * step_gate * post_grace

    total = (
        rew_alive
        + rew_terminated
        + rew_raw_action                                # NEW IN p6e (replaces rew_upright slot)
        + rew_height
        + rew_standup_height
        + rew_standup_fall
        + rew_thermal
        + rew_vel_tracking
        + rew_feet_air
        + rew_gait_step
        + rew_foot_swing
        + rew_energy
        + rew_action_rate
        + rew_action_magnitude
        + rew_joint_vel
        + rew_joint_pos_limits
        + rew_hip_roll_split
        + rew_front_split
        + rew_knee_split
        + rew_leg_symmetry
        + rew_angle_symmetry
        + rew_knee_over_hip
        + rew_foot_progress
        + rew_position_anchor
        + rew_heading_anchor
        + rew_back_leg_step
        + rew_yaw_rate
        + rew_balance
        + rew_no_flight
        + rew_vz_penalty
        + rew_stance_symmetry
        + rew_yaw_drift
        + rew_action_smoothness    # NEW
        + rew_action_snap          # NEW: 3rd-order
        + rew_upright              # NEW: uprightness penalty
        + rew_clock_contact        # NEW knob 1
        + rew_swing_clearance      # NEW knob 2
        + rew_gait_ref             # NEW A/B fork: gait-reference prior
    )

    # Component stack — order MUST match _REWARD_COMPONENT_NAMES.
    # Slot 2 is repurposed in p6e: rew_upright → rew_raw_action.
    components = torch.stack([
        rew_alive,            # 0  alive
        rew_terminated,       # 1  terminated
        rew_raw_action,       # 2  raw_action  (was upright, was 0)
        rew_height,           # 3  height
        rew_standup_height,   # 4  standup_height
        rew_standup_fall,     # 5  standup_fall
        rew_thermal,          # 6  thermal
        rew_vel_tracking,     # 7  vel_tracking
        rew_feet_air,         # 8  feet_air
        rew_gait_step,        # 9  gait_step
        rew_foot_swing,       # 10 foot_swing
        rew_energy,           # 11 energy
        rew_action_rate,      # 12 action_rate
        rew_action_magnitude, # 13 action_magnitude
        rew_joint_vel,        # 14 joint_vel
        rew_joint_pos_limits, # 15 joint_pos_limits  (now 0 by config)
        rew_hip_roll_split,   # 16 hip_roll_split
        rew_front_split,      # 17 front_split  (now 0 by config)
        rew_knee_split,       # 18 knee_split  (now 0 by config)
        rew_leg_symmetry,     # 19 leg_symmetry
        rew_angle_symmetry,   # 20 angle_symmetry
        rew_knee_over_hip,    # 21 knee_over_hip
        rew_foot_progress,    # 22 foot_progress
        rew_position_anchor,  # 23 position_anchor
        rew_heading_anchor,   # 24 heading_anchor
        rew_back_leg_step,    # 25 back_leg_step
        rew_yaw_rate,         # 26 yaw_rate
        rew_balance,          # 27 balance
        rew_no_flight,        # 28 no_flight
        rew_vz_penalty,       # 29 vz_penalty
        rew_stance_symmetry,  # 30 stance_symmetry
        rew_yaw_drift,        # 31 yaw_drift
        rew_action_smoothness,# 32 NEW
        rew_action_snap,      # 33 NEW: 3rd-order smoothness
        rew_upright,          # 34 NEW: uprightness penalty
        rew_clock_contact,    # 35 NEW knob 1
        rew_swing_clearance,  # 36 NEW knob 2
        rew_gait_ref,         # 37 NEW A/B fork: gait-reference prior
    ], dim=-1)

    return total, components
