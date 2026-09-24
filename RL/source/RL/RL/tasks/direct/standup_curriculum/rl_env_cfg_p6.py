# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""
rl_env_cfg_v7.py — Version 7: switch to DCMotorCfg (explicit motor model).

════════ A/B FORK (gait-reference prior) — SCOUT RUN, SPARE GPU ════════
    Forked from the SAME checkpoint the main run resumed from. The ONLY
    functional delta vs. the main run is the "A/B FORK" block in §9
    (rew_scale_gait_ref / gait_ref_amplitude_rad / gait_ref_sigma_rad)
    plus the new reward slot + diagnostics in rl_env_p6.py. Setting
    rew_scale_gait_ref = 0.0 recovers the main run's reward function
    bit-for-bit (the new slot logs 0 and contributes no gradient), so
    the change set stays attributable.
    The queued relative-to-stance clearance fix is deliberately NOT
    folded in — one experimental variable per fork.
════════════════════════════════════════════════════════════════════════

Major actuator rewrite. Resumes-from-scratch — incompatible with any
p6 checkpoint because the policy now controls a fundamentally different
joint dynamics (real-motor PD in Python + velocity-torque envelope +
armature + Coulomb / viscous joint friction), not PhysX implicit PD
with effort clipping.

WHY THE SWITCH:

    On hardware, the implicit-actuator policy showed two failure modes
    that no amount of p6e/f/g tuning could close:

      1. Bang-bang on hip_roll even with action_scale widened. The
         policy assumed flat torque availability up to effort_limit
         that the real motor cannot deliver near its speed limit.

      2. Phase-margin collapse under any real-world disturbance —
         the policy compensated for sim's idealized PD response,
         then lost authority on hardware where back-EMF dropped
         available torque at speed.

    DCMotor models both directly:
        τ_max(v) = saturation_effort · (1 − |v|/v_limit),
        clipped to ±effort_limit.

    The policy now sees the same speed-torque dropoff sim and real,
    and learns to plan trajectories that don't exceed it.


CHANGES VS. p6h:

    1. ACTUATORS — DCMotorCfg with measured per-joint params:
            stiffness    = 6.0 / 8.0 / 4.0       (real motor Kp)
            damping      = 0.3 / 0.5 / 0.3       (real motor Kd)
            saturation   = 6.0 / 17.0 / 23.2 Nm  (stall / peak)
            effort_limit = 1.8 / 6.0 / 8.2 Nm    (continuous)
            velocity_limit = 28 / 27.5 / 20 rad/s (no-load @ 30V)
            armature     = 7e-4 / 4.2e-3 / 7.8e-3 kg·m²
            friction (static / dynamic / viscous) per measured table
       Knee values reflect the 30:22 belt reduction.

    2. ACTION SCALES — recalibrated to Kp · scale = effort_limit:
            hip_roll  : 1.8 / 6.0 = 0.30
            hip_pitch : 6.0 / 8.0 = 0.75
            knee      : 8.2 / 4.0 = 2.05
       Action [-1, 1] now maps to ±continuous torque at zero velocity.
       Burst is reachable through DCMotor's saturation curve when the
       PD setpoint is far from current pose AND velocity is low.

    3. THERMAL PENALTY DISABLED (rew_scale_thermal = 0.0):
       DCMotor clips applied_torque at effort_limit (= continuous),
       so (applied_torque / continuous)² ≤ 1 always, so the penalty
       is identically zero with zero gradient. The motor model itself
       now enforces what the penalty used to. The reward-component
       slot is kept (compute_rewards still receives _continuous_effort)
       so log keys remain stable; the slot just reads 0.

    4. NO MORE Kp / SCALE CURRICULUM:
       The p6 train_kp → real_kp interpolation existed because PhysX
       implicit PD at Kp=6 was numerically fragile and we needed to
       train at Kp=60 first. DCMotor's explicit Python PD is stable
       at low Kp, so we start at real values. Deleted: train_kp_*,
       real_kp_*, train_kd_*, real_kd_*, train_action_scale_*,
       params_curriculum_*, action_rate_alpha_*. Standup-pose
       curriculum (kneel_start_probability ramp) is preserved.

    5. EXPANDED PER-EPISODE DR for the new actuator model:
            stiffness mult       0.85 – 1.15   (Kp tolerance)
            damping mult         0.70 – 1.30   (independent of Kp now)
            effort mult          0.90 – 1.00   (capped at nominal)
            armature mult        0.70 – 1.30   (rotor inertia tolerance)
            joint friction mult  0.50 – 2.00   (Coulomb, breakaway)
            dyn friction mult    0.50 – 2.00   (kinetic)
            visc friction mult   0.50 – 2.00   (damping vs ω)
            voltage factor       0.87 – 1.07   (per-env, scales v_limit)
       voltage_factor is ONE scalar per robot (one battery), applied
       to all three joints' velocity_limit. Friction ranges are wide
       because the measured numbers are placeholders pending bench data.

    6. INIT-STATE RANDOMIZATION:
       Added base roll / pitch perturbation at reset
       (init_base_orient_noise_rad = 0.08, scaled by 1−difficulty so
       kneel-branch resets don't get extra tilt on top of the kneel
       quat). Joint pos / vel / base lin & ang vel noise unchanged.

    7. DAMPING DECOUPLED FROM STIFFNESS:
       p6f coupled the two draws to avoid catatonic low-Kp + high-Kd
       corners. With DCMotor's friction terms now in the mix, the
       parameter space is large enough that independent draws don't
       reliably hit those corners, and the coupling was suppressing
       genuine variance the real motor exhibits.

INHERITED FROM p6 SERIES:
    - 5-frame obs history stack, per-frame mirror tables
    - Action LPF (α=0.6, ~8 Hz at 100 Hz)
    - Action smoothness (jerk) penalty, weighted in joint space
    - Raw (pre-clip) action penalty for anti-saturation gradient
    - lin_vel obs dropout (set to 1.0 — always zero, matching hardware)
    - Obs delay + action delay DR, joint-pos / proj-grav per-episode
      biases, floor-friction / mass / CoM physics DR
    - Anti-gallop trio (no-flight, vz penalty, stance symmetry EMA)
    - Yaw drift penalty, position / heading anchoring
    - Linear standup-height ramp (no sigmoid)
    - Mirror data augmentation hook for rsl_rl symmetry_cfg
    - DR impact diagnostics (per-DR correlation with episode length)

PHYSICS / CONTROL RATE:
    Physics:  2000 Hz (dt = 1/2000) — explicit DCMotor PD is stable
                                       well below this; the extra rate keeps
                                       foot contact resolution clean.
    Control:  100 Hz (decimation 20) — matches real robot command rate.
    (Was 500 Hz / decimation 5 through 09-15.  Changing either of these means
    changing `hold_decimation` in the hip_knee actuator cfg AND `series_dt` /
    `series_substeps` in BeltKneeDCMotorCfg — the 09-16 move to 2000 Hz missed
    the belt integrator and cost a day of false belt-stiffness conclusions.)

DEVELOPER NOTES:
    1. Never remove logs already emitted to TensorBoard.
    2. The more diagnostics the better.
    3. First-iteration sanity: verify
         self.robot.actuators["hip_roll"].stiffness.std() > 0
       after one reset. If zero, the per-env DR write path isn't
       reaching the actuator object.
    4. Watch reward/raw_action_p99_abs and diagnostics/clip_fraction —
       with smaller action scales the calibration of rew_scale_raw_action
       may need a re-tune.

────────────────────────────────────────────────────────────────────
FILE ORGANIZATION (v7 cleanup — no values changed, only regrouped):
    §1  Core rates & spaces        §7   Push disturbance
    §2  Simulation                 §8   Pickup ("virtual hand")
    §3  Robot / actuators / scene  §9   Reward scales
    §4  Observation pipeline       §10  Termination limits & obs noise
    §5  Commands                   §11  Action processing
    §6  Standup / kneel curriculum §12  DOMAIN RANDOMIZATION (impact-tagged)
────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import DCMotorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .belt_knee_actuator import (
    BELT_COUPLING, BELT_G, BELT_MOTOR_INERTIA, ROTOR_INERTIA_OUTPUT,
    BeltKneeDCMotorCfg, damping_for_ratio, derive_armature,
)
from .foot_contact import FootContactUsdFileCfg


# Robot asset. Repo-relative by default (assets/tiptoe_v27/robot/robot.usd);
# override with the TIPTOE_ROBOT_USD environment variable.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 7)))
ROBOT_USD_PATH = os.environ.get(
    "TIPTOE_ROBOT_USD", os.path.join(_REPO_ROOT, "assets", "tiptoe_v27", "robot", "robot.usd"))


# ────────────────────────────────────────────────────────────────────
# Actuator factories — the belt variant is the training default; the
# legacy variant reproduces the pre-belt p6/v7 model exactly, kept for
# sim2real A/B in data/sysid_replay_policy_log.py (--actuator legacy).
#
# P7 (2026-08-18): friction + armature centrals replaced with the P5/P7
# bench values (replant report, review + armature sections). All values
# JOINT-space; knee motor-space numbers were reflected through G=22/30
# (I,b ×(30/22)^2 = 1.860; C ×30/22 = 1.364). Provenance:
#   friction  C_s   hip 0.146 / knee 0.130*1.364=0.177 / roll 0.187
#   dynamic   C_k   hip 0.144 / knee ≈0.170            / roll 0.185
#   viscous   b     hip 0.043 (CI 0.008-0.062) / knee 0.016*1.86=0.030
#                   / roll ≈0 → keep 0.003 floor
#   armature        rotor-side inertia from the bench-mode decomposition
#                   (URDF links account for only 17e-3 of the measured
#                   31e-3 hip mode / 2.5e-3 of the knee's 38.5e-3):
#                   P7 fit I_r_hp=13.9e-3, I_r_kn(out)=19.4e-3 →
#                   hip 13.9e-3 + c²·36.0e-3 = 33.2e-3 (was 8.4e-3)
#                   knee I_r_kn/G² = 36.0e-3            (was 7.81e-3)
#                   roll P4-total 31-34e-3 − links 25.2e-3 ≈ 7.0e-3
#                                                        (was 7.0e-4)
#   2026-09-18: hip_pitch/knee are DERIVED, not entered.  Both factories
#   and the env build path call belt_knee_actuator.derive_armature() on
#   ROTOR_INERTIA_OUTPUT = 4.2e-3 (VENDOR spec, RS01 rotor reflected to
#   its output shaft) → knee I_out/G² = 7.81e-3, hip own 4.2e-3 + c²·knee
#   = 8.40e-3, off-diagonal c·knee = 5.72e-3.  That is the ORIGINAL P6
#   plant (header: 4.2e-3 / 7.8e-3) and NOT the P7 33.2/36.0e-3 above nor
#   the L-line 38.5e-3: those rested on bench APPARENT-inertia fits
#   (13.9e-3 / 19.4e-3 / 19.3e-3 "rotors") that no 380 g motor can carry;
#   the bench signal is being re-attributed (belt compliance / torque
#   scale / delay / friction — data/coastdown_fit.py --pinned-inertia).
#   Every K/L/R/S/S7 checkpoint was trained on the OLD plant and is still
#   reachable by pinning: make_*_actuators(hip_pitch_armature=33.2e-3,
#   knee_armature=36.0e-3) / env.hip_pitch_armature=0.0385
#   env.knee_armature=0.036 (L/R/S/S7) or 0.0332 + 0.036 (P7/K).
#   NOTE the commanded-τ caveat still holds: every bench number shares the
#   unknown per-motor torque-scale, and armature here no longer absorbs
#   it — the replica gate (eval.py hw_matched) must be re-run on this plant.
# ────────────────────────────────────────────────────────────────────

def _factory_armature(hip_pitch_armature, knee_armature,
                      rotor_inertia_output, hip_pitch_own_rotor_armature):
    """Factory-side pin semantics: ``None`` derives, a number pins exactly
    (what the pre-change ``hip_pitch_armature=33.2e-3`` argument did)."""
    return derive_armature(
        rotor_inertia_output, hip_pitch_own_rotor_armature,
        hip_pitch_pin=-1.0 if hip_pitch_armature is None else float(hip_pitch_armature),
        knee_pin=-1.0 if knee_armature is None else float(knee_armature),
    )

def _hip_roll_actuator():
    return DCMotorCfg(
        joint_names_expr=["hip_roll.*"],
        saturation_effort=6.0,        # peak/stall — EL05 峰值
        effort_limit=6.0,             # continuous  — EL05 额定
        velocity_limit=28.0,          # no-load at 30V
        effort_limit_sim=8.0,
        velocity_limit_sim=55.0,
        stiffness=6.0, damping=0.3,
        armature=7.0e-3,              # P7: P4 fit 31-34e-3 minus URDF links
                                      # 25.2e-3 (tether-confounded — wide DR)
        # friction: P5 bench (brk_103_hang, ±0.004 repeatable).
        friction=0.187, dynamic_friction=0.185, viscous_friction=0.003,
    )


def make_legacy_actuators(
    hip_pitch_armature: float | None = None,
    knee_armature: float | None = None,
    rotor_inertia_output: float = ROTOR_INERTIA_OUTPUT,
    hip_pitch_own_rotor_armature: float = float("nan"),
):
    """Pre-belt p6/v7 groups: plain DCMotor, rigid direct-drive knee.
    P7: same measured centrals as the belt factory (runbook: 'centrals in
    BOTH factories' — the Phase 8 replica A/B runs through this one).
    Armature: derived from ``rotor_inertia_output`` (see the header block);
    ``hip_pitch_armature`` / ``knee_armature`` given as numbers pin exactly."""
    _arm = _factory_armature(hip_pitch_armature, knee_armature,
                             rotor_inertia_output, hip_pitch_own_rotor_armature)
    return {
        "hip_roll": _hip_roll_actuator(),
        "hip_pitch": DCMotorCfg(
            joint_names_expr=["hip_pitch.*"],
            saturation_effort=17.0,       # peak/stall — RS01 峰值
            effort_limit=17.0,             # continuous  — RS01 额定
            velocity_limit=27.5,          # no-load at 30V
            effort_limit_sim=22.0,
            velocity_limit_sim=45.0,
            stiffness=8.0, damping=0.5,
            armature=_arm.hip_pitch,
            # friction: P5 bench (brk_101_hang + bal2/bal3 pooled fit).
            friction=0.146, dynamic_friction=0.144, viscous_friction=0.043,
        ),
        "knee": DCMotorCfg(
            joint_names_expr=["knee.*"],
            saturation_effort=23.2,       # 17 × 30/22 (belt)
            effort_limit=23.2,             # 6 × 30/22  (belt)
            velocity_limit=20.0,          # 27.5 × 22/30
            effort_limit_sim=30.0,
            velocity_limit_sim=35.0,
            stiffness=8.0, damping=0.5,
            armature=_arm.knee,
            # friction: P5 bench (brk_102_belt_on), joint-space ×1.364.
            # Belt-on totals — belt-off split pending refines apportionment.
            friction=0.177, dynamic_friction=0.170, viscous_friction=0.030,
        ),
    }


def make_belt_actuators(
    knee_decouple_ff: bool = True,
    hip_reaction: bool = False,
    stale_hip_hold: bool = True,
    hip_pitch_armature: float | None = None,
    rotor_coupling: bool = True,
    knee_armature: float | None = None,
    rotor_inertia_output: float = ROTOR_INERTIA_OUTPUT,
    hip_pitch_own_rotor_armature: float = float("nan"),
):
    """Belt-drivetrain "firmware replica": hip_pitch + knee merged into one
    BeltKneeDCMotor group so the knee columns can read hip state (motor-space
    PD, stale-hip target hold, motor-velocity torque envelope).

    Armature (2026-09-18): DERIVED by belt_knee_actuator.derive_armature()
    from ``rotor_inertia_output`` (RS01 inertia at its output shaft, motor
    frame): knee = I_out/G², hip_pitch = own rotor + c²·knee — the knee rotor
    reflected through the belt anchor.  Defaults give 35.9e-3 / 38.6e-3, the
    P12 L-line plant.  ``hip_pitch_armature`` / ``knee_armature`` given as a
    number pin that joint exactly (the P7 33.2e-3/36.0e-3 plant is
    ``hip_pitch_armature=33.2e-3, knee_armature=36.0e-3``).  The old 8.4e-3
    assumed a 4.2e-3 vendor rotor that three bench signatures contradict.

    rotor_coupling=True adds the off-diagonal of the same inertia block,
    c·knee ≈ 26.3e-3, as a plant force in BeltKneeDCMotor — on the fall
    logs that term is a launch-window hip torque as large as the hip's own
    commands (rms ratio 0.96, peaking at the 0.16 s fall-initiation time).
    With it, the sim reproduces ALL THREE bench modes (knee-only 38.5e-3,
    hip-with-knee-motor-held 31e-3, hip-with-knee-joint-fixed) exactly.

    knee_decouple_ff=True models the hardware KNEE_DECOUPLE_FF v_des patch
    live since 2026-08-06 (config.py) — train against the plant as deployed.
    """
    _arm = _factory_armature(hip_pitch_armature, knee_armature,
                             rotor_inertia_output, hip_pitch_own_rotor_armature)
    return {
        "hip_roll": _hip_roll_actuator(),
        "hip_knee": BeltKneeDCMotorCfg(
            joint_names_expr=["hip_pitch.*", "knee.*"],
            saturation_effort={"hip_pitch.*": 17.0, "knee.*": 23.2},  # RS01 峰值 (knee ×30/22 belt)
            effort_limit     ={"hip_pitch.*": 17.0, "knee.*": 23.2},
            velocity_limit   ={"hip_pitch.*": 27.5, "knee.*": 20.0},  # no-load at 30V (knee ×22/30)
            effort_limit_sim ={"hip_pitch.*": 22.0, "knee.*": 30.0},
            velocity_limit_sim={"hip_pitch.*": 45.0, "knee.*": 35.0},
            stiffness=8.0, damping=0.5,
            armature={"hip_pitch.*": _arm.hip_pitch, "knee.*": _arm.knee},
            # friction: P5 bench, joint-space (see factory header block).
            friction={"hip_pitch.*": 0.146, "knee.*": 0.177},
            dynamic_friction={"hip_pitch.*": 0.144, "knee.*": 0.170},
            viscous_friction={"hip_pitch.*": 0.043, "knee.*": 0.030},
            coupling=BELT_COUPLING,       # config.py KNEE_HIP_COUPLING, one source
            hip_own_rotor_armature=_arm.hip_own_rotor,   # mask keeps own/total under a live belt
            knee_decouple_ff=knee_decouple_ff,
            hip_reaction=hip_reaction,
            stale_hip_hold=stale_hip_hold,
            rotor_coupling=rotor_coupling,
            rotor_coupling_dt=1.0 / 2000.0,  # =  sim dt (compute per physics step)
            hold_decimation=20,            # keep  in sync with RlEnvCfg.decimation
        ),
    }

_BELT_RATIO = BELT_G                       # single source: belt_knee_actuator
_BELT_MOTOR_INERTIA = BELT_MOTOR_INERTIA   # single source: belt_knee_actuator



@configclass
class RlEnvCfg(DirectRLEnvCfg):

    # ════════════════════════════════════════════════════════════════
    # §1  CORE RATES & SPACES
    # ════════════════════════════════════════════════════════════════
    decimation = 20                  #  physics 2000 Hz → control 100 Hz
    episode_length_s = 30.0
    action_space = 6

    # ── Observation history stack ────────────────────────────────────
    # Stack K recent obs frames as policy input. With delay DR active
    # (total 1–7 step round-trip), a feedforward MLP fed only the
    # current frame has no way to recover velocity from stale position.
    # Stacking gives the network N-1 finite differences worth of history
    # per step — enough for an implicit Kalman estimate. Stacking happens
    # AFTER obs_delay, so each frame is itself delayed. Frame 0 = newest.
    # ⚠️ observation_space MUST equal 36 * obs_history_steps. Cannot
    #    resume a p6g checkpoint with this on (input dim changes).
    obs_history_steps: int = 5
    observation_space = 36 * obs_history_steps   # keep in sync with 36
    # ── Asymmetric critic (privileged) observation group ──────────────
    # Actor sees `observation_space` (policy group). Critic ALSO sees this
    # privileged group (concatenated via obs_groups in the PPO cfg): the env
    # state the actor is deliberately blind to — true base velocity, pickup
    # phase/support/force, and the full per-env DR vector. Removes the value-
    # function shocks those cause without ever leaking into the deployed policy.
    # MUST equal the width built in _get_critic_observations.
    # v7.1: +12 for the two new actuator-lag DR blocks (omega_mult[6] +
    #       zeta_mult[6]); was 66. Incompatible with a v7.0 checkpoint
    #       (critic input width changes) — resumes-from-scratch like the
    #       rest of the DCMotor rewrite.
    priv_obs_dim: int = 78
    state_space = priv_obs_dim

    # ════════════════════════════════════════════════════════════════
    # §2  SIMULATION
    # ════════════════════════════════════════════════════════════════
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 2000.0, #
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(
            # 600k from S7 on: S5 asked for 320 560 … 501 974 against the old
            # 320 000 and flooded the log with "Patch buffer overflow" (5354
            # lines, S4a: zero).  See docs/s5-elastic-dr-20260914.md §4.
            gpu_max_rigid_patch_count=600_000,   # S5 asked up to 501 974
            bounce_threshold_velocity=0.05,      # S6: balls bounce on low-speed contact
        ),
    )

    # ════════════════════════════════════════════════════════════════
    # §3  ROBOT / ACTUATORS / SCENE
    # ════════════════════════════════════════════════════════════════
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=FootContactUsdFileCfg(
            usd_path=ROBOT_USD_PATH,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.434),
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
        # Belt "firmware replica" actuators (see make_belt_actuators above and
        # belt_knee_actuator.py). Legacy pre-belt groups: make_legacy_actuators().
        actuators=make_belt_actuators(),
    )

    imu_body_name: str = "imu"
    foot_body_name_1: str = "shin_inner"
    foot_body_name_2: str = "nail___fillet1"

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=32768,
        env_spacing=2.5,
        replicate_physics=True,
    )

    # ── Body masses (for CoM / gravity-comp bookkeeping) ─────────────
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
        1.698, 1.1141, 0.434, 0.2005,
        1.1141, 0.434, 0.2005,
    ]

    # ════════════════════════════════════════════════════════════════
    # §4  OBSERVATION PIPELINE (sim2real shaping; NOT randomization)
    # ════════════════════════════════════════════════════════════════
    # lin_vel observation dropout — the real robot has no body-frame
    # velocity estimator (lin_vel_imu reads ~0 always on hardware). In
    # sim it carries true body vel: a 3-slot obs gap. Per-env Bernoulli
    # mask zeros the lin_vel slots at reset, held for the episode.
    #   0.0 → always true lin_vel (sim-only)   0.5 → half/half
    #   1.0 → always zero (matches hardware; current training value)
    lin_vel_obs_dropout_prob: float = 1.0

    # ════════════════════════════════════════════════════════════════
    # §5  COMMANDS
    # ════════════════════════════════════════════════════════════════
    # Velocity-X command
    vx_cmd_min: float = -0.10 #
    vx_cmd_max: float = 0.10
    vx_cmd_zero_probability: float = 0.6

    # Yaw-rate command (conservative ±0.5, p6d)
    yaw_rate_cmd_min: float = -0.1
    yaw_rate_cmd_max: float = 0.1
    yaw_rate_cmd_zero_probability: float = 0.5
    yaw_rate_cmd_anchor_threshold: float = 0.05

    cmd_resample_steps: int = 1000

    # Gait clock
    gait_freq: float = 2.5
    gait_phase_offset: float = 0.0

    # ════════════════════════════════════════════════════════════════
    # §6  STANDUP / KNEEL CURRICULUM
    # ════════════════════════════════════════════════════════════════
    # Stance height — discovered by policy (not commanded)
    height_min: float = 0.32
    height_max: float = 0.55
    height_edge_softness: float = 50.0
    spawn_height = 0.434

    # Kneel pose (p6d: measured)
    kneel_height: float = 0.234
    kneel_quat_w: float = 0.7071
    kneel_quat_x: float = 0.0
    kneel_quat_y: float = -0.7071
    kneel_quat_z: float = 0.0
    kneel_hip_roll1: float = 0.0
    kneel_hip_pitch1: float = 3.09
    kneel_knee1: float = -3.06
    kneel_hip_roll2: float = 0.0
    kneel_hip_pitch2: float = -3.09
    kneel_knee2: float = 3.06
    kneel_noise_rad: float = 0.05

    # Standup curriculum — p6e mid-run change: skip the alpha ramp. The
    # math-interpolated kneel→stand init plus the linear standup-height
    # reward give a smooth gradient already. Start alpha=1.0 from iter 0;
    # advance logic is a no-op (curriculum_max_alpha=1.0) but the
    # kneel-branch sampling path still needs curriculum_enable=True.
    curriculum_enable: bool = False
    curriculum_initial_alpha: float = 0.3
    curriculum_max_alpha: float = 1.0
    curriculum_alpha_step: float = 0.05
    curriculum_advance_threshold: float = 0.5
    curriculum_advance_window: int = 500
    curriculum_advance_cooldown: int = 1500
    # Advance metric (09-07). "ep_len": legacy — mean kneel-branch episode
    # length / max_episode_length. Grace (difficulty × standup_grace_seconds)
    # counts toward ep_len, so a deep-kneel env that NEVER stands still
    # contributes d×400 steps and the criterion fires on the cooldown clock
    # (S1b/S2b 09-06: alpha 0.4→1.0 while d≥0.75 stand-up success was zero).
    # "success_high": fraction of the HARDEST difficulty tertile of the window
    # that survived ≥ curriculum_success_post_grace_steps after grace expiry
    # (a robot that never rises terminates too_low the step grace ends).
    # "success_high_rel" (09-08): success_high / max(success_low, 0.05).
    # The low tertile (d≈0.1 = near-standing start) already loses 20–32 %
    # of episodes to spawn/push/DR falls within 1.4 s; those losses cap
    # every band, so an absolute bar demands a different fraction of the
    # attainable ceiling per arm (S2c 0.6/0.68 = 88 %). The ratio ≈
    # P(stands up | no early fall).
    curriculum_advance_metric: str = "ep_len"
    curriculum_success_post_grace_steps: int = 100
    # Frontier-focused difficulty sampling (09-08). With d ~ U[0, alpha] on a
    # kneel_start_probability share of resets, only ~6 % of episodes land
    # in the frontier band [alpha−0.1, alpha]; the rest re-train mastered
    # poses. This fraction of kneel-branch envs draws d from
    # U[max(0, alpha − frontier_width), alpha] instead; the remainder keeps
    # the full U[0, alpha] so the easy regime is never removed. 0.0 = legacy.
    curriculum_frontier_frac: float = 0.0
    curriculum_frontier_width: float = 0.3

    standup_grace_seconds: float = 4.0
    post_grace_stabilize_steps: int = 50
    kneel_start_probability: float = 0.50
    # Stand-up ramp shape (09-08). The grace-phase PD action center travels
    # from the spawn pose to the standing pose; the policy only adds
    # ±action_scale on top, so this path IS the stand-up motion.
    #   "linear"   — legacy P6 ramp: every joint blends linearly, normalized by
    #                the MAX grace (so at t=0 the center sits at blend d², not
    #                d — an instant PD kick for d<1). Bit-exact with all runs
    #                before 09-08.
    #   "keyframe" — per-joint piecewise-linear timing on the env's OWN grace:
    #                phase s = 1 − grace_remaining/grace_assigned ∈ [0,1],
    #                q_j(s) = init_j + (stand_j − init_j)·f_j(s). Profiles are
    #                flat [s0,f0, s1,f1, ...] lists with s ascending from 0 to
    #                1 and f from 0 to 1; hip_roll follows the hip profile.
    #                [0,0,1,1] on both = linear timing (with the d² kick fixed).
    #   "path"     — (09-08) absolute joint-space keyframe path seiza → stand with
    #                the SPAWN placed ON the path: difficulty d spawns at phase
    #                s0 = 1 − d in the resting state of standup_path_knots and
    #                the action center travels s0 → 1 over the assigned grace
    #                (constant path speed). Rows are
    #                  [s, hip_target, knee_target, pitch_deg, root_h_m, hip_rest, knee_rest]
    #                (leg 2 mirrored, hip_roll 0). *_target = the PD action-center
    #                path; *_rest / pitch / root_h = the state the plant actually
    #                RESTS in while holding that target (joints sag ~0.15 rad under
    #                load, knee pinned at its −3.06 limit while the shin is flat).
    #                Spawning writes the rest joints and the center starts at the
    #                target, so the initial PD error is exactly the gravity-holding
    #                error — no kick. Measured 09-08 with probe_standup_keyframes
    #                (holdA candidates, nominal plant, zero action, 4 s settle;
    #                `script/show_standup_path.py -Scan` is the independent
    #                acceptance check: every row must read "ok"). Phase A (s ≤ 0.45)
    #                rests on knees+balls and is passively stable; phase B (s > 0.45)
    #                is the straight squat line to standing at the CoM-over-ball
    #                pitch θ* (RL/logs/balance_manifold_ball.csv), rest = target.
    standup_ramp_mode: str = "linear"
    standup_hip_profile: list = [0.0, 0.0, 1.0, 1.0]
    standup_knee_profile: list = [0.0, 0.0, 1.0, 1.0]
    standup_path_knots: list = [
        0.00, 3.090, -3.060, -90.0, 0.235, 3.090, -3.060,
        0.05, 2.869, -2.998, -78.3, 0.239, 2.878, -3.054,
        0.10, 2.648, -2.936, -63.8, 0.235, 2.623, -3.053,
        0.15, 2.427, -2.873, -52.7, 0.233, 2.391, -3.014,
        0.20, 2.206, -2.811, -42.8, 0.230, 2.160, -2.955,
        0.25, 1.984, -2.749, -33.4, 0.225, 1.933, -2.893,
        0.30, 1.763, -2.687, -24.2, 0.218, 1.707, -2.827,
        0.35, 1.542, -2.624, -14.9, 0.207, 1.482, -2.766,
        0.40, 1.321, -2.562,  -6.2, 0.197, 1.261, -2.695,
        0.45, 1.100, -2.500,   2.7, 0.184, 1.039, -2.630,
        0.5875, 0.838, -1.875,  9.7, 0.305, 0.838, -1.875,
        0.725, 0.575, -1.250,   8.4, 0.375, 0.575, -1.250,
        0.8625, 0.312, -0.625,  2.3, 0.424, 0.312, -0.625,
        1.00, 0.050,  0.000,   0.0, 0.439, 0.050,  0.000,
    ]
    standup_path_spawn_clearance_m: float = 0.003
    # Grace-phase balance shaping, path mode only (0 = off). track: exp(−(Δpitch/σ)²)
    # against the path's pitch(s); contact: fraction of feet on the ground.
    rew_scale_standup_track: float = 0.0
    standup_track_sigma_rad: float = 0.3
    rew_scale_standup_contact: float = 0.0

    # Standup reward shape (p6e: linear ramp from kneel_height →
    # stand_target_height; no sigmoid / midpoint / sharpness anymore).
    stand_target_height: float = 0.434

    init_base_orient_noise_rad: float = 0.08

    # ════════════════════════════════════════════════════════════════
    # §7  PUSH DISTURBANCE
    # ════════════════════════════════════════════════════════════════
    push_enable: bool = False
    push_probability: float = 0.0006    # was 0.0002
    push_force_min: float = 10.0        # was 10
    push_force_max: float = 30.0        # was 30
    push_duration_steps: int = 10 # was 100, big impact on ep length

    # ════════════════════════════════════════════════════════════════
    # §8  PICKUP — continuous-force "virtual hand"
    # ════════════════════════════════════════════════════════════════
    # State machine: LIFT → HOLD → LOWER → release. Each phase tunes a
    # closed-loop PD that drives base z to a target. No teleport, no
    # state wipes; obs / action / LPF / delay all flow continuously.
    pickup_enable: bool = True
    pickup_probability: float = 0.0005          # ~one event per 2000 steps
    pickup_min_steps_since_reset: int = 400

    # Lift target height (added to current base z when LIFT begins)
    pickup_lift_height_min: float = 0.20
    pickup_lift_height_max: float = 0.35

    # Phase durations, in control steps (100 Hz)
    pickup_lift_steps: int = 60                  # 0.6s — slow lift
    pickup_hold_steps_min: int = 80              # 0.8s
    pickup_hold_steps_max: int = 300             # 3.0s
    pickup_lower_steps: int = 80                 # 0.8s — slow descent

    # PD on base z. Critically-damped 2nd-order parametrization.
    pickup_z_omega: float = 7.0                  # natural freq, rad/s
    pickup_z_damping_ratio: float = 1.2          # slightly over-damped for safety

    # Force cap as multiple of robot weight (3× = plenty; stops the PD
    # from explosively yanking the robot if z error spikes).
    pickup_force_cap_mult: float = 3.0

    # Contact-based early release: if any foot dips below this z during
    # LOWER, release immediately (soft landing regardless of PD lag).
    pickup_contact_release_z: float = 0.024

    # Pickup orientation PD — targets q=(1,0,0,0) (upright). Small-angle
    # linearization is fine; torque cap saves us at large tilts. Yaw is
    # loosened so the policy keeps some rotational authority.
    pickup_orient_kp: float = 25.0          # was 6.0
    pickup_orient_kd: float = 6.0           # was 1.5
    pickup_torque_cap: float = 15.0         # was 4.0
    pickup_orient_yaw_kp_scale: float = 0.4 # softer yaw (was 0.5)
    pickup_orient_yaw_kd_scale: float = 1.2 # stronger yaw damping (was 1.0)

    pickup_target_roll_max:  float = 0.60    # ± rad (~33°)
    pickup_target_pitch_max: float = 0.60    # ± rad
    # Yaw target is preserved (locked to current yaw at trigger). Don't
    # randomize yaw — it would compose with the heading-anchor reward
    # after release and create spurious yaw recovery work.

    pickup_pd_gain_factor_min: float = 0.70
    pickup_pd_gain_factor_max: float = 1.40

    pickup_hold_wobble_xy: float = 0.015    # m, peak amplitude
    pickup_hold_wobble_z:  float = 0.010
    pickup_hold_wobble_freq_hz: float = 0.4 # ~one cycle every 2.5s

    # Release latency: time between feet-on-ground and full hand release.
    # During this window all PD outputs scale linearly full → zero,
    # simulating gradual weight transfer from hand to legs.
    pickup_release_latency_min_steps: int = 5     # 50ms minimum
    pickup_release_latency_max_steps: int = 40    # 400ms maximum

    # Pickup-on-spawn — fraction of resets scheduled straight into LIFT.
    # 0.0 disables; ~0.5 gives half the starts on the ground and half
    # being placed down. Spawn pickups lift higher than mid-episode ones
    # (a real operator lifting off the floor goes higher than an
    # in-flight bump; mid-ep range is ~0.05–0.20 m).
    pickup_initial_probability: float = 0.5
    pickup_initial_lift_height_min: float = 0.25
    pickup_initial_lift_height_max: float = 0.50

    pickup_post_grace_steps: int = 50   # ~0.5 s landing settle at 100 Hz

    pickup_gate_debug_check: bool = False

    pickup_termination_support_threshold: float = 0.3
    

    # ════════════════════════════════════════════════════════════════
    # §9  REWARD SCALES
    # ════════════════════════════════════════════════════════════════
    # ── Liveness ──
    rew_scale_alive: float = 6.0
    rew_scale_terminated: float = -10.0

    # ── Standup ──
    rew_scale_standup_height: float = 10.0
    rew_scale_standup_fall:   float = -15.0

    # ── Command tracking ──
    vel_tracking_sigma: float = 0.15
    rew_scale_vel_tracking: float = 2.0/3
    yaw_rate_tracking_sigma: float = 0.30
    rew_scale_yaw_rate: float = 2.0
    # Averaged command-tracking / anchoring windows (instantaneous → windowed).
    # EMA decay α; time constant ≈ 1/(1-α) control steps at 100 Hz.
    #   0.99 ≈ 100 steps ≈ 1.0 s (~2.5 strides at gait_freq 2.5 Hz).
    # Tracking (vel + yaw-rate) and anchoring (position + heading) split so
    # they can be tuned independently.
    tracking_ema_decay: float = 0.99
    anchor_ema_decay: float = 0.99

    # ── Gait ──
    rew_scale_gait_step: float = 4.0
    rew_scale_feet_air: float = 1.5
    rew_scale_foot_swing: float = 1.0
    rew_scale_foot_progress: float = 100.0
    rew_scale_back_leg_step: float = 80.0
    feet_air_height_threshold: float = 0.005 *2 *5
    # NEW knob 1: binary clock-contact reward. Pays for matching the gait
    # clock's stance/swing schedule per foot (sign of gait_sin). The liftoff
    # itself is the unit of reward — no partial credit for a 3 mm lift.
    # RESUME 2 CHANGE — uncentered (clamped at 0): standing-still = 0,
    # perfect alternation = +1, wrong-phase = 0 (was −1). The −1 wrong-phase
    # penalty taxed exploration: with 1–2 steps of action delay and zero
    # stepping skill, a first tentative lift is ~coin-flip on phase, so the
    # centered term had ~zero expected value for the explorer while adding
    # risk. no_flight / stance_symmetry already police the pathologies.
    # Re-center later if lazy half-credit behavior appears.
    rew_scale_clock_contact: float = 2.0
    # NEW knob 2: explicit swing-clearance reward. During each foot's swing
    # half-cycle, rewards toe height above swing_clearance_zero_m, ramping
    # 0→1 over swing_clearance_target_m. Directly opposes the shuffle.
    #
    # RESUME 2 CHANGE — densified. Previously the ramp started at
    # no_flight_contact_threshold (0.084): zero gradient until the foot had
    # already cleared the contact line, i.e. invisible from the 3 mm shuffle
    # the policy actually sits at. Now the ramp starts at
    # swing_clearance_zero_m, just above the RESTING toe z, so the gradient
    # is felt from the very first millimetre of lift.
    #
    # ⚠️ swing_clearance_zero_m = 0.075 is a PLACEHOLDER (resting toe z has
    # never been measured; 0.075 sits just under the 0.084 contact line).
    # After ~10 iterations of the resume, read diagnostics/idle_toe_z_p10
    # (new this round) and set:
    #     swing_clearance_zero_m = idle_toe_z_p10 + 0.003
    # If p10 reads ≥ 0.075 the placeholder is too low (rewarding rest) —
    # fix immediately. If it reads ≪ 0.075, lower it to restore the dense
    # gradient. Scale raised 1.0 → 2.0: at step_gate 0.5 the dense pull is
    # 2.0·0.5/0.04 = 25 reward-units per metre of lift, doubling the
    # feet_air+foot_swing gradient (≈25/m) that was losing the argument.
    rew_scale_swing_clearance: float = 2.0
    swing_clearance_target_m: float = 0.04
    # CALIBRATED 2026-08-07 (iter 236 of p8_belt_fresh): idle_toe_z_p10 read
    # 0.0615 → 0.0560 → 0.0519 over iters 0/73/236 (drifting down as the
    # policy crouches: base_height 0.72 → 0.66). Per the rule above,
    # p10 + 0.003 = 0.055. The 0.075 placeholder left the first ~23 mm of
    # lift with ZERO gradient — invisible from where the feet actually rest.
    # Re-read if base_height shifts much further.
    # RE-CALIBRATED 2026-08-10 (iter 2630 of C_hipfence): idle_toe_z_p10
    # has drifted back UP to ~0.072 (stance height reversed course from
    # the 236-iter reading). 0.055 now sits ~17mm BELOW resting toe z —
    # idle stance was again collecting reward past the ramp start with
    # zero lift, the exact bug this mechanism exists to prevent. p10 +
    # 0.003 = 0.075.
    swing_clearance_zero_m: float = 0.075

    # ══ P10 A/B knobs (2026-08-19): plant/servo arms, CLI-set ═══════
    # Consumed in rl_env_p6.py BEFORE super().__init__ builds the robot
    # (the actuator cfg is frozen into the articulation at that point, so
    # a plain cfg field is the only hydra-reachable path — the nested
    # actuator dicts carry regex keys with dots that break hydra paths).
    # Defaults reproduce the P9 plant bit-for-bit.
    #
    # plant_rotor_coupling=False drops ONLY the off-diagonal belt-inertia
    # echo torque in BeltKneeDCMotor. The diagonals (armature 33.2/36.0e-3)
    # stay — they are correct whether or not the off-diagonal is modeled.
    #
    # servo_kp/kd are JOINT-space. Design basis (P5/P7 measured J):
    #   hip_pitch J≈50.3e-3 → kp 20, kd 1.4 = 3.2 Hz, ζ0.70
    #   knee      J≈38.5e-3 → kp 30, kd 1.5 = 4.4 Hz, ζ0.70
    # (restores the servo envelope the P8 policies converged under, which
    # the old 4.2e-3 armature had only pretended to provide). Action
    # scales are deliberately NOT coupled to kp — position envelope and
    # action_rate weights stay fixed; Kp·scale (±1-action authority) is
    # printed at startup. Deploy mapping when a gains arm flies: motor-side
    # knee kp/kd = joint × (22/30)² → 30/1.5 joint = 16.1/0.81 motor.
    plant_rotor_coupling: bool = True
    servo_kp_hip_pitch: float = 8.0
    servo_kd_hip_pitch: float = 0.5
    servo_kp_knee: float = 8.0
    servo_kd_knee: float = 0.5
    servo_kp_hip_roll: float = 6.0
    servo_kd_hip_roll: float = 0.3

    # ══ Armature: given facts, derived values (2026-09-18) ═══════════
    # Replaces the P12 recenter knob pair (hip 33.2e-3 / knee 36.0e-3 as
    # two unrelated literals, L-line override hip 0.0385).  The cfg now
    # carries the GIVEN quantity and the env derives the PhysX block in
    # ONE place (belt_knee_actuator.derive_armature, see its docstring):
    #
    #   knee      = rotor_inertia_output / G²            = 7.81e-3
    #   hip_pitch = hip_pitch_own_rotor + c² · knee      = 8.40e-3
    #
    # rotor_inertia_output — kg m², the knee RS01's rotor inertia REFLECTED
    #   TO ITS OUTPUT SHAFT through the motor's internal 7.75:1 reduction
    #   (motor frame, before the 22:30 belt; = the series integrator's
    #   rotor).  VENDOR SPEC 4.2e-3 (RS01: 17 Nm peak, 7.75:1, ~380 g).
    #   The P7/P12 bench figures (13.9e-3 / 19.4e-3 / 19.3e-3) are
    #   APPARENT inertias (bench total minus URDF links) kept only as
    #   historical fits — a 19.3e-3 rotor would need ~75 % of the motor
    #   mass at the housing radius.  The bench signal itself is not
    #   disputed; its attribution to the rotor is (re-fit with the rotor
    #   pinned: data/coastdown_fit.py --pinned-inertia).
    # hip_pitch_own_rotor_armature — the hip RS01's own output-shaft
    #   inertia (direct drive, so already joint side).  NaN = same motor
    #   model, same value as rotor_inertia_output.  P7 had fitted 13.9e-3
    #   here (→ hip 33.2e-3).  Pin 0.0139 to rebuild P7.
    # hip_pitch_armature / knee_armature — a value >= 0 PINS that joint
    #   exactly, as before (the pre-change recipes: every L/R/S/S7 launch
    #   passes env.hip_pitch_armature=0.0385, + knee_armature=0.036 to be
    #   exact; K/P9 = 0.0332 + knee 0.036).  Negative (default) = derived.
    #   A derived hip uses the RESOLVED knee.
    # Same consume path as the P10 knobs above (the actuator dict is
    # frozen into the articulation before super().__init__ and its regex
    # keys aren't hydra-reachable); resolved values print as [ARMATURE].
    #
    # NOT the plant of ANY checkpoint: hip 33.2e-3 (P9/K) / 38.5e-3
    # (L/R/S/S7) → 8.4e-3, knee 36.0e-3 → 7.8e-3, series rotor 19.36e-3 →
    # 4.2e-3 (−78 %).  A resume onto the derived defaults is a plant
    # change; pin (above) to stay on the trained plant, or start a new
    # line (launch_train_S8_vendorJ.ps1).  The P10 servo gains (kp 30 /
    # kd 1.5, designed for knee J≈38.5e-3) read as ~8.6 Hz, ζ≈1.3 here.
    rotor_inertia_output: float = ROTOR_INERTIA_OUTPUT
    hip_pitch_own_rotor_armature: float = float("nan")
    hip_pitch_armature: float = -1.0
    knee_armature: float = -1.0
    # DR topology (2026-09-18): ONE armature multiplier per leg for the
    # co-axial hip_pitch/knee pair.  The knee rotor IS the hip share and the
    # off-diagonal, so scaling the three with independent draws (what every
    # K/L/R/S run did) produced inertia blocks no single rotor can make.
    # True: the [N,6] multiplier's hip_pitch and knee columns of a leg are
    # the same draw, and the off-diagonal (c·armature_kn) and the series
    # rotor handed to the belt integrator follow it by construction; hip_roll
    # keeps its own draw.  This CHANGES THE DR DISTRIBUTION versus every
    # K/L/R/S run — it is an A/B arm, not a refactor.  False = the legacy
    # independent per-joint draws (same RNG consumption, bit-identical).
    dr_armature_per_leg: bool = True

    # ══ Elasticity/contact evaluation knobs (2026-09-11) ══
    # Intentionally zero/disabled by default.  The two parameters live in
    # DIFFERENT frames because they are compared against different things:
    #   stiffness — KNEE-JOINT N m/rad, the servo's frame: leave it at
    #               servo_kp_knee for a belt exactly as stiff as the position
    #               loop, then go down.  The env converts with G^2 (torque
    #               scales once and angle once): k_motor = k_joint*G^2.
    #   damping   — MOTOR-side N m s/rad, the spring's OWN coordinate frame,
    #               because the useful scale is the belt's damping ratio,
    #               zeta = d_m / (2*sqrt(k_m*I_rotor_motor)), and no power of
    #               G appears in a torque/velocity ratio.  Joint-referring it
    #               would have implied a G^2 that the parallel spring/damper
    #               pair does not have.
    # Setting min=max pins a deterministic arm.  No observation dimension
    # changes, but under elasticity the knee obs channels read the motor
    # encoder rather than the physical joint (see RlEnv._encoder_observation).
    #
    # ── DR FORM (2026-09-12, S5 line) ──────────────────────────────────
    # A nonzero band is now a per-episode DR draw, not only a pinned sweep
    # cell: every env draws its own (k_joint, d_motor) pair at reset, exactly
    # the way every other plant multiplier in §12 is drawn.  The belt is then
    # LIVE whether or not `belt_series_enable` is set — the flag's remaining
    # job is to keep the pair on the rigid path when BOTH bounds are zero, so
    # the pre-S5 recipes stay bit-identical.  Setting enable=False while a
    # nonzero band is present no longer disables the pair; it is the band that
    # decides.  (This matters: the old gate silently turned a band into the
    # rigid plant, which is precisely the "assume it changed, it never did"
    # failure the S5 launch is meant to avoid.)
    #
    # The band is the knob that must not lie, and the belt is the case where a
    # zero bound is meaningful rather than a dead value: k_joint = 0 is a rigid
    # belt, so `[0, 0]` is "off" and any nonzero endpoint is a real spring.
    belt_series_enable: bool = False
    belt_series_stiffness_min: float = 0.0       # knee-joint N m/rad
    belt_series_stiffness_max: float = 0.0
    belt_series_damping_min: float = 0.0         # MOTOR-side N m s/rad
    belt_series_damping_max: float = 0.0
    belt_series_rotor_inertia: float = _BELT_MOTOR_INERTIA
    # Damping entered AS A RATIO of the transmission mode, the readable form:
    # d_motor = 2*zeta*sqrt(k_joint*G^2*I_rotor), with zeta measured against the
    # belt's own rotor, not the robot.  Set >= 0 and the damping is DERIVED per
    # episode from whichever stiffness that episode drew (the damping band then
    # stays at zero and is ignored); NaN (default) leaves
    # belt_series_damping_min/max in charge, and a nonzero band always wins.
    # This is the form that stays meaningful across a stiffness sweep, where a
    # fixed N m s/rad number would silently over-damp the soft cells and ring
    # the stiff ones.
    belt_series_damping_ratio: float = float("nan")
    # ── Belt A/B isolation knobs (2026-09-15).  Both default to the behaviour
    # every S5/S7 run had, so nothing moves unless a launch names them.  They
    # exist because the train-vs-play gap at equal k was not explained by the
    # encoder indexing bug: the one code path that is ON in training (band on
    # the CLI => series_elasticity at build) and OFF in play's hotkey mode
    # (band set after build) is the motor-side knee observation, and the one
    # plant change common to both is the armature mask.  Flip one at a time.
    #   belt_encoder_obs=False  -> knee obs channels read the PHYSICAL joint
    #                              even under an elastic belt (play-hotkey obs)
    #   belt_mask_armature=False-> knee+hip_pitch PhysX armature stays at its
    #                              cfg value (the rotor is then counted twice,
    #                              once in PhysX and once in the integrator —
    #                              a diagnostic, not a plant)
    #   2026-09-18: with the mask ON, the hip_pitch column is now scaled to
    #   its own-rotor share (own/total, 0.50 derived / 0.42 under P7 pins)
    #   instead of ZERO.  Every S5/S6/S7 belt-live env trained with the hip
    #   armature zeroed; this is a plant change on those lines (A/B arm).
    belt_encoder_obs: bool = True
    belt_mask_armature: bool = True
    # Per-channel split of the encoder obs (ladder arms E/F, 2026-09-16): the
    # A/B result put the kill on the encoder obs while the readbacks showed the
    # encoder-vs-joint deviation to be 0.01 rad in position (inside the trained
    # 0.008 per-step + 0.07 bias bands) and ~1.2 rad/s in velocity (several
    # times the trained 0.6 band, and correlated).  These pick which channel
    # actually reads the encoder; both True = the full encoder contract.
    belt_encoder_obs_pos: bool = True
    belt_encoder_obs_vel: bool = True
    # Per-step trace of what enters the policy vs what PhysX has, for the first
    # N control steps of the run on the first M envs, saved once as
    # <log_dir>/belt_debug.npz (2026-09-16; 0 = off).  Exists because arms E/F
    # of the ladder died on an encoder deviation the readbacks put at 0.01 rad.
    belt_debug_dump_steps: int = 0
    belt_debug_dump_envs: int = 8

    # Contact material is attached only to the two ball-foot collision
    # meshes.  With all values zero the original UsdFileCfg path is retained.
    # Damping has the same ratio form as the belt, against the ball's own
    # contact mode with the mass that actually participates in a single-support
    # catch (~1 kg), NOT the 5.57 kg whole body — using the total over-damps by
    # sqrt(5.57) and would conflate "soft" with "heavily damped".
    foot_contact_stiffness: float = 0.0          # N/m, force-based PhysX mode
    foot_contact_damping: float = 0.0            # N s/m, explicit
    foot_contact_damping_ratio: float = float("nan")
    foot_contact_effective_mass: float = 1.0     # kg, for the ratio form
    foot_restitution: float = 0.0                # normal coefficient; >1 is energy-gaining
    # Ball traction against the floor. The floor material is 1.0 with
    # friction_combine_mode="multiply", so the ball's value is the ball's
    # effective friction: set it to 0 and no foot setting can hold the robot.
    # This is the failure knob that works from ANY pose, including the S4a
    # crouch where the balls never carry load (k and restitution are both
    # force-per-contact-area knobs, so they stay invisible there).
    # NOTE: 1.0 is the pass-through value, so changing it is what creates the
    # material; leaving both at 1.0 with everything else zero keeps the legacy
    # rigid path bit-exact.
    foot_static_friction: float = 1.0
    foot_dynamic_friction: float = 1.0

    # ── Foot restitution as a per-env DR band (S6, 2026-09-14) ────────────
    # The material prim is stage-global (one foot_contact material shared by
    # every env), BUT get_material_properties()/set_material_properties() on
    # root_physx_view address a per-env, per-SHAPE buffer (N, max_shapes, 3):
    # static_friction, dynamic_friction, restitution.  That is the home for a
    # per-env restitution draw that the material spawner cannot express.
    # The two ball shapes are DISCOVERED at runtime from the buffer (rows whose
    # restitution reads foot_restitution), never hardcoded.  These band knobs
    # are inert while both are 0, so every other run keeps its bit-exact path.
    dr_foot_restitution_min: float = 0.0
    dr_foot_restitution_max: float = 0.0
    dr_foot_restitution_buckets: int = 64
    # The material knobs above are stage-global: one material prim is shared by
    # every env, and the articulation view exposes no channel for compliant
    # gains, so a per-env pad stiffness has no home in the physics material.
    # These knobs model the pad as an external spring/damper applied to each
    # shin by the step (pad_contact.PadContact), which is what makes pad DR
    # possible at all.  They compose with the material knobs rather than
    # replacing them: the material's rigid contact remains the backstop, and
    # the pad adds compliance on top of it.
    #
    # `pad_stiffness_max = 0` (the default) is INERT — no force, no state, the
    # legacy plant bit-for-bit — so adding these fields cannot move a training
    # line that never sets them.
    pad_contact_enable: bool = False
    # Knee-referred?  No: the pad is at the floor, so its stiffness is a true
    # floor-normal N/m, the same frame as foot_contact_stiffness.
    pad_stiffness_min: float = 0.0
    pad_stiffness_max: float = 0.0
    # Damping as a ratio of the pad's OWN mode against the mass that actually
    # participates in a single-support catch (~1 kg), not the whole body —
    # the same convention as foot_contact_damping_ratio, and for the same
    # reason (using the total mass over-damps by sqrt(m_robot/m_eff)).
    pad_damping_ratio: float = 0.7
    # Per-episode damping-RATIO band (S7, 2026-09-14).  The scalar above is the
    # pinned form; this is the DR form, and it is entered as a RATIO rather than
    # an explicit N s/m band because the S5 explicit band (0–0.3 N s/m) was a
    # knob whose value could not affect the plant: zeta ~1e-3 at every point, and
    # the relaxation time tau = m/c was clamped at its 4*dt = 8 ms floor
    # (docs/s5-elastic-dr-20260914.md §4).  A zeta band fixes both at once: zeta
    # is the number that sets how much energy the pad returns, and `tau` follows
    # from it.  `pad_damping_ratio_max > min` makes the band live and the scalar
    # is then ignored by the DR draw (a pinned/held value still wins, as before).
    pad_damping_ratio_min: float = -1.0
    pad_damping_ratio_max: float = -1.0
    pad_effective_mass: float = 1.0
    # The pad's own relaxed thickness, which sets where its contact surface sits
    # (ball radius minus this).  It is the one number in the model that cannot be
    # measured off the asset, and it is a HARD CONSTRAINT, not a preference:
    # a pad whose travel is shorter than the sag it will carry (`pad_stiffness_min`
    # and the per-foot load) bottoms out on the ball's rigid contact and stops
    # behaving like a pad at all — the settled sag then reads as the thickness
    # instead of the stiffness.  Keep it several times the equilibrium sag.
    pad_thickness: float = 0.002
    # An explicit damping band, in N s/m.  Nonzero wins over the ratio, so a
    # pinned arm keeps meaning exactly what it says.
    pad_damping_min: float = 0.0
    pad_damping_max: float = 0.0
    # Explicit-damping stability ceiling is 2*m_eff/dt; this keeps a margin
    # below it.  Damping above the cap is clipped and reported, never silently
    # applied.
    pad_damping_stability_factor: float = 0.5

    def resolved_armature(self):
        """The hip_pitch/knee armature block from this cfg's facts and pins —
        the ONE call the env build path makes (it overwrites the factory's
        armature dict wholesale, so the factory cannot be the source)."""
        return derive_armature(
            self.rotor_inertia_output, self.hip_pitch_own_rotor_armature,
            hip_pitch_pin=self.hip_pitch_armature, knee_pin=self.knee_armature,
        )

    def belt_band_live(self) -> bool:
        """True when the belt band asks for a series spring on any episode.

        The band decides, not `belt_series_enable`: a nonzero stiffness bound
        is a real elastic belt, and the flag exists only to keep the all-zero
        default on the rigid legacy path.  A band whose MAX is zero draws
        k = 0 on every episode, which is the rigid belt, so it is not live.
        """
        return self.belt_series_stiffness_max > 0.0

    def pad_enabled(self) -> bool:
        """True when this cfg asks for any pad force at all."""
        return bool(self.pad_contact_enable) and (
            self.pad_stiffness_max > 0.0 or self.pad_stiffness_min > 0.0
        )

    def pad_damping_band_live(self) -> bool:
        """True when the pad's damping is a per-episode RATIO band.

        Negative sentinels are the "unset" state, so the band is live only when
        both bounds are >= 0 and the max exceeds the min.  A band whose two ends
        are equal is a pinned value wearing a band's name, and it is rejected
        here rather than silently drawing a constant (the S5 launch's whole
        failure mode was a channel that read as set and was not).
        """
        if self.pad_damping_ratio_min < 0.0 or self.pad_damping_ratio_max < 0.0:
            return False
        return self.pad_damping_ratio_max > self.pad_damping_ratio_min

    def resolved_pad_damping_ratio(self) -> float:
        """The single ratio the ratio form resolves against.

        Returns NaN while the band is live: every drawn ratio is then resolved
        per env instead, and any caller that reaches for one scalar has a bug
        rather than a default.
        """
        return float("nan") if self.pad_damping_band_live() else float(self.pad_damping_ratio)

    def resolved_foot_contact_damping(self, stiffness: float | None = None) -> float:
        """Foot contact damping for a stiffness, honouring whichever form is set.

        Explicit ``foot_contact_damping`` wins over the ratio when it is
        nonzero, so an existing recipe cannot be silently re-pointed by adding
        a ratio later.  The ratio form needs a positive stiffness: rigid
        contact with restitution has no spring to damp.
        """
        k = self.foot_contact_stiffness if stiffness is None else float(stiffness)
        if self.foot_contact_damping > 0.0:
            return float(self.foot_contact_damping)
        if math.isnan(self.foot_contact_damping_ratio):
            return 0.0
        if k <= 0.0:
            raise ValueError(
                "foot_contact_damping_ratio needs a positive foot_contact_stiffness; "
                "for rigid contact with bounce set foot_restitution and leave damping at 0"
            )
        return 2.0 * float(self.foot_contact_damping_ratio) * math.sqrt(
            k * self.foot_contact_effective_mass
        )

    def resolved_belt_damping(self, stiffness_joint: float) -> float:
        """Motor-side belt damping for a drawn knee-referred stiffness.

        Band wins when it is nonzero — that is the explicit form, and a recipe
        that pins belt_series_damping_min=max must keep meaning exactly what it
        says.  The ratio form applies only while the band is still all zero.
        """
        if self.belt_series_damping_min > 0.0 or self.belt_series_damping_max > 0.0:
            return 0.0  # caller drew from the band instead
        if math.isnan(self.belt_series_damping_ratio):
            return 0.0
        return damping_for_ratio(
            stiffness_joint, self.belt_series_damping_ratio, self.belt_series_rotor_inertia
        )

    # ══ A/B FORK: gait-reference prior (the "reserve weapon") ══════
    # Injects HOW to step instead of waiting for discovery — the
    # Cassie-lineage trick for exactly this failure mode (policy
    # understands the clock, won't commit amplitude). Reward:
    #
    #   exp( -[(hp1_dev - A·sinφ)² + (hp2_dev - A·sinφ)²] / σ² )
    #       × step_gate × post_grace
    #
    # hp_dev = hip_pitch - default, φ = the EXISTING gait clock
    # (gait_freq 2.5 Hz, already in the obs — the policy has everything
    # it needs to track it). Live at idle via the step_gate floor (0.5),
    # full strength while walking, silent during grace / pickup.
    #
    # SAME-SIGN reference for both legs is intentional: the obs mirror
    # table swaps legs AND negates all joints (defaults +0.05 / −0.05
    # are one symmetric stance), so leg 2's axes are mirrored and an
    # antiphase PHYSICAL gait satisfying state(φ+π) = mirror(state(φ))
    # has hp1_dev = hp2_dev = A·sinφ in JOINT coordinates. The term is
    # therefore mirror-invariant and cannot fight the symmetry loss /
    # data augmentation. (Sanity: pair-sum hp1+hp2 = 2A·sinφ averages
    # to ~0 over the angle_symmetry EMA (τ≈200 steps ≈ 5 strides), so
    # that −40 penalty doesn't fight it either.)
    #
    # This says nothing about weight transfer — clock_contact and
    # swing_clearance stay on as the "actually lift" verification.
    # SCAFFOLDING, NOT THE BUILDING: once idle_liftoff_rate ≥ 0.03
    # sustained, anneal the scale on a later resume (3.0 → 1.0 → 0.3 →
    # 0) and let the task terms own the behavior. Watch whether the
    # imposed 2.5 Hz, ±A swing fights the LPF + 5× smoothness penalties
    # (it shouldn't: 2.5 Hz ≪ the ~8 Hz LPF corner).
    # POLISH RESUME 2026-08-10 (forked from C_hipfence @ iter 2630):
    # idle_liftoff_rate has read 0.037-0.048, sustained, since well before
    # iter 2400 — trigger condition met a while ago. First step of the
    # anneal per the note above.
    rew_scale_gait_ref: float = 1.0  # was 3.0; anneal 3.0 → 1.0 → 0.3 → 0
    # SIGNED amplitude (rad). The sign decides which way hip pitch
    # deviates during each clock half — i.e. whether the reference
    # agrees with clock_contact that leg 1 swings when sinφ > 0. Not
    # derivable from the cfg alone; resolved empirically: after ~10–20
    # iters read diagnostics/gait_ref_airborne_swing_hp_dev (mean hp
    # deviation of clock-designated swing feet at moments they're
    # ACTUALLY airborne — and we know existing lifts are already clock-
    # aligned, clock_contact ≈ above_contact_frac). If it reads
    # NEGATIVE, flip this sign. Same symptom from the other side:
    # reward/gait_ref rising while clock_contact falls.
    gait_ref_amplitude_rad: float = -0.12
    # Kernel width (rad). Known leak: standing at default pose still
    # collects exp(−2(A/σ)²sin²φ) — phase-averaged ≈ 0.33 of max at
    # A=0.20, σ=0.15 (≈ 0.5 reward/step at idle gate vs 1.5 for perfect
    # tracking). Acceptable — the ~1.0/step differential IS the prior;
    # shrink σ to sharpen once tracking appears.
    gait_ref_sigma_rad: float = 0.15

    # ── Anti-gallop trio (p6d) ──
    rew_scale_no_flight: float = -5.0
    no_flight_contact_threshold: float = 0.084    # was 0.075
    rew_scale_vz_penalty: float = -2.0
    stance_symmetry_decay: float = 0.98
    rew_scale_stance_symmetry: float = -80.0

    # ── Symmetry ──
    leg_symmetry_decay: float = 0.98
    rew_scale_leg_symmetry: float = -80.0
    angle_symmetry_decay: float = 0.995
    # p6e mid-run change: halved from -80.0. At -80 this term ate ~14.7%
    # of total reward magnitude, dominating the gradient and pushing the
    # policy toward fragile low-amplitude shuffle gaits.
    rew_scale_angle_symmetry: float = -40.0

    # ── Stance / balance ──
    rew_scale_height: float = 1.0
    rew_scale_balance: float = 5.0 # should be positive
    rew_scale_knee_over_hip: float = -0.02
    rew_scale_hip_roll_split: float = -2.0
    # ── Uprightness ──
    # Penalty on horizontal projected gravity (= sin²(tilt)). Pulls the
    # base toward vertical; gradient is strongest in the 20–70° range
    # (d/dθ sin²θ = sin2θ), i.e. exactly where you are now, and →0 near
    # vertical so it doesn't over-tax the small working tilt walking needs.
    # Gated by post_grace: off during grace (kneel→stand is tilted by
    # design) and during pickup (the hand owns orientation). Point-foot
    # bipeds balance dynamically — don't crank this so hard it kills the
    # idle stepping you just added or shortens episodes.
    rew_scale_upright: float = -2.0
    # ── Anchoring (zero-command) ──
    rew_scale_position_anchor: float = -8.0
    rew_scale_heading_anchor: float = -8.0
    # NEW knob 3: position-anchor deadband (metres). Drift inside this radius is
    # free; only drift beyond it is penalized (re-squared outside the deadband).
    # Lets stepping-in-place body sway happen without being taxed as drift.
    position_anchor_deadband_m: float = 0.04
    rew_scale_yaw_drift: float = -1.0       # anti-yaw-cheat (p6d)

    # ── Thermal / energy ──
    # Thermal kept at 0.0: DCMotor clips at continuous effort, so the
    # (applied/continuous)² penalty is identically zero. Slot retained
    # for stable log keys.
    rew_scale_thermal: float = -0.0
    rew_scale_energy:  float = -0.001

    # ── Action regularization ──
    rew_scale_action_rate: float = -0.3/5/2
    # Second-derivative (jerk) penalty on CLIPPED actions; catches
    # oscillations action_rate cannot distinguish from smooth motion.
    # FRESH RUN 2026-08-07: reverted to the pre-boost value. The 5× boost
    # (-0.08 → -0.4) was polish for an already-standing policy; a fresh
    # explorer's inevitable early jitter would pay it every step. Re-boost
    # on the polish resume once standing is solid.
    rew_scale_action_smoothness: float = -0.08  # was -0.2*4/10*4*2/8*5 = -0.4 (5× on resume)
    # NEW: third-order action smoothness — discrete derivative of the jerk
    # term, i.e. the 3rd finite difference of the action (1,-3,3,-1 stencil).
    # Zero for constant-jerk trajectories; fires on changing curvature.
    # Starting value ≈ smoothness × (6/20): the summed squared-stencil ratio
    # between Δ² (1,2,1 → 6) and Δ³ (1,3,3,1 → 20), so for comparable action
    # noise its contribution roughly matches the jerk term. Tune against
    # diagnostics/action_snap_sq_mean.
    # FRESH RUN 2026-08-07: reverted to pre-boost (same rationale as
    # smoothness above). Re-boost to -0.1 on the polish resume.
    rew_scale_action_snap: float = -0.02  # was -0.1 (5× on resume)
    rew_scale_action_magnitude: float = -0.02      # on CLIPPED action²
    # On RAW (pre-clip) action² (p6e). Gives gradient feedback even when
    # the action saturates, keeping pre-clip logits calibrated. Without
    # it, raw outputs drift to ±10–±15 with no corrective signal.
    # p6e mid-run change: -0.5 → -1.0 → currently -0.3. Stronger penalty
    # pulls in tail outliers (±9.6) that became bigger torque kicks once
    # action_scale widened.
    rew_scale_raw_action: float = -0.3*10 /20 #*20  # 5× on resume (-0.6 → -3.0)
    raw_action_threshold: float = 0.95 

    # ── Joint regularization ──
    rew_scale_joint_vel: float = -0.000

    # ════════════════════════════════════════════════════════════════
    # §10  TERMINATION LIMITS & BASE-STATE OBS NOISE
    # ════════════════════════════════════════════════════════════════
    max_tilt_rad: float = 1.8
    min_height: float = 0.30
    max_hip_roll_rad: float = 1.8
    max_hip_pitch_split_rad: float = 1.8
    max_knee_split_rad: float = 1.8

    # ── Sustained knee-saturation termination (2026-08-07) ──
    # Hardware forensics: the 3 regressing hang logs sit at ~32% action clip
    # fraction, where no PD law explains the measured torque any more (belt
    # slip/compliance regime; torque-fit R² 0.23–0.45 vs 0.75–0.89 on calm
    # logs). Firmware register reads rule out current limiting (limit_torque
    # = 17/6 Nm = the assumed caps), so the unmodelable regime is mechanical
    # and knee-specific — the plant model, and any policy trained on it, is
    # untrustworthy exactly there. Terminate on SUSTAINED knee clipping so
    # the policy learns to stay in the modelable regime; brief transient
    # clips (catch steps, standup burst) pass through the EMA unharmed.
    # Rides the standard terminated penalty; shielded during pickup/grace
    # like every other termination, and the EMA decays while shielded so
    # pickup phases can't pre-charge it.
    sat_term_enable: bool = True
    # EMA time constant: 100% sustained clipping trips the 0.5 threshold in
    # ~0.35 s from zero; isolated 1–3-step clips never accumulate past ~0.1.
    # Tune against diagnostics/knee_sat_ema_max and _term_frac.
    # A/B RESULT 2026-08-09 (fresh p8 run, forked at iter 236, compared to
    # ~iter 1550): STRICT fence (threshold 0.35, tau 0.3) vs loose (0.5/0.5).
    # Strict arm won on EVERYTHING: knee clip 0.08 vs 0.19, ep_len 957 vs
    # 729 through the growth explosion, smoother actions (jerk 1.35 vs
    # 1.82), lower noise std at same stage. Both arms ride their own fence
    # (ema_max pins just under threshold), so the threshold directly sets
    # the knee-saturation operating point — and the task pays nothing for
    # the stricter one. Folded the strict values into the defaults.
    sat_term_tau_s: float = 0.3      # was 0.5
    sat_term_threshold: float = 0.35  # was 0.5
    # A/B arm C knob (2026-08-09): extend the fence to the hips. Per-pair
    # EMAs (hip_roll 0/3, hip_pitch 1/4) against THIS threshold; 0.0 =
    # disabled (knee-only fence, the proven B config). Rationale: hr clip
    # drifted to ~0.55 / hp ~0.45 during the growth phase. Unlike the knee
    # there is NO plant-validity evidence against hip saturation (hip
    # torque fits held on the bad logs; caps register-verified), so this
    # fence targets CONTROL ROBUSTNESS (bang-bang leaves no authority
    # margin for delays/biases), not model validity — hence looser
    # threshold and experiment-first. 0.5 would mass-kill at resume with
    # hr duty already ~0.55; start 0.65 and anneal down on later resumes
    # if ep_len holds.
    # ARM-C RESULT 2026-08-10 (killed B same day): with the fence at 0.65,
    # C held ep_len growth (880→~980+ through iter 2630) at less than
    # half B's terminal hip clip (hr 0.30 vs 0.57, hp 0.21 vs 0.50) while
    # B's noise std started CLIMBING (0.72→0.73) at a flat ep_len — a
    # regression with no fence pushing back. B killed; C is now the
    # surviving lineage. Folding the fence on by default (like the knee
    # threshold after the first A/B) and taking the first anneal step
    # (0.65→0.55) since ep_len held through the initial engagement.
    sat_term_threshold_hip: float = 0.55  # was 0.0 (disabled) / 0.65 (arm C start)

    # Base-state obs noise (legacy; distinct from the §12 DR noise block)
    noise_joint_pos: float = 0.1
    noise_base_vel_lin: float = 0.3
    noise_base_vel_ang: float = 0.2

    # ════════════════════════════════════════════════════════════════
    # §11  ACTION PROCESSING
    # ════════════════════════════════════════════════════════════════
    # Action low-pass filter
    action_lpf_alpha: float = 0.6      # cutoff ≈ 8 Hz at 100 Hz

    # Continuous-effort reference (defines "above-continuous" — physical)
    continuous_effort_hip_roll:  float = 1.8
    continuous_effort_hip_pitch: float = 6.0
    continuous_effort_knee:      float = 8.18

    # Action scales — train == real in v7 (no curriculum). Calibrated so
    # Kp · scale = effort_limit, i.e. action ±1 → ±continuous torque at
    # zero velocity. train_/real_ pairs kept for interface compatibility.
    train_action_scale_hip_roll:  float = 0.3     # 1.8 / 6.0
    train_action_scale_hip_pitch: float = 0.75    # 6.0 / 8.0
    train_action_scale_knee:      float = 2.05/2    # 8.2 / 4.0
    real_action_scale_hip_roll:   float = 0.3
    real_action_scale_hip_pitch:  float = 0.75
    real_action_scale_knee:       float = 2.05/2

    # ── Actuator-lag prefilter (2nd-order command prefilter) ─────────
    # sysid finding: the actuator closed-loop position response is 2nd-order,
    # NOT a pure transport delay (the step rises gradually from t=0). It is
    # modelled as a per-joint 2nd-order low-pass on the PD position target,
    # advanced at the 100 Hz command rate, exactly as in sysid_replay_sim.py:
    #     y'' + 2·ζ·ω·y' + ω²·y = ω²·u           (u = raw target, y = filtered)
    # ω sets the bulk lag / rise time (~14–16 rad/s ⇒ ~0.12 s rise); ζ sets the
    # overshoot (LOWER ζ = more ring). Values are the identified per-joint
    # centrals (hip_roll EL05 rings most, knee RS01+belt rings least — matches
    # the "knee overshoots a hair only at +0.6" overlay). Set act_lag_enable
    # False (or a joint's ω to 0) to bypass the prefilter on that path.
    #
    # NOTE: this is a COMMAND prefilter, distinct from the §12b action_delay
    # (a transport dead-time). Both are real and stack: dead-time first
    # (discrete steps), then the 2nd-order lag on the delayed command.
    # P7 (2026-08-18): DISABLED — P4 killed the prefilter with three
    # independent falsifiers (ring moves with kp = servo mode, not a
    # command filter; M1.1 onsets 2.4-2.8 ms; free prefilter fits
    # transparent). The ω/ζ below are kept only as dead reference for the
    # old-bus fits. The dr_act_lag mults are pinned at 1.0 so the 12
    # priv-obs dims [61:73] stay allocated (constant) — no renumbering.
    act_lag_enable: bool = False   # P4 verdict: phantom on all 3 joints
    act_lag_omega_hip_roll:  float = 21.0   # rad/s (≈ 2.09 Hz)
    act_lag_omega_hip_pitch: float = 37.9  # rad/s (≈ 2.24 Hz)
    act_lag_omega_knee:      float = 27.6   # rad/s (≈ 2.54 Hz)
    act_lag_zeta_hip_roll:   float = 0.414    # most ring (hip_roll EL05)
    act_lag_zeta_hip_pitch:  float = 0.803
    act_lag_zeta_knee:       float = 0.580    # least ring (knee RS01 + belt)

    # ════════════════════════════════════════════════════════════════
    # §12  DOMAIN RANDOMIZATION
    # ════════════════════════════════════════════════════════════════
    #
    # IMPACT TAGS = estimated correlation magnitude of each DR axis with
    # episode length, i.e. how much it gates "does the robot stay up."
    # These are PRIORS from control/physics reasoning, NOT measured —
    # confirm against the logged diagnostics/<dr>_eplen_corr series and
    # re-tag. Tighten ranges on HIGH axes if they block learning; widen
    # LOW axes freely for cheap extra robustness.
    #
    #   HIGH  — dominant difficulty driver; sim2real risk concentrates here
    #   MED   — meaningful but policy usually absorbs it
    #   LOW   — policy is ~indifferent across the range (free robustness)
    #
    #   ── HIGH ─────────────────────────────────────────────────────
    #     action_delay, imu/joint delay   (loop dead time → phase margin)
    #     floor_friction               (foot traction → slip/fall)
    #   ── MED ──────────────────────────────────────────────────────
    #     com_offset                   (balance point shift)
    #     mass_mult                    (torque-to-weight budget)
    #     joint_friction (Coulomb)     (breakaway/stiction near v≈0)
    #     stiffness_mult (Kp)          (PD response)
    #   ── LOW ──────────────────────────────────────────────────────
    #     damping, armature, dyn/visc friction, voltage_factor,
    #     effort_mult (narrow 0.9–1.0), all per-step obs noise,
    #     all per-episode obs biases
    #
    # NOTE: §10 noise_* and §4 lin_vel_obs_dropout_prob are sim2real
    # shaping, not part of this randomization block.
    # ─────────────────────────────────────────────────────────────────
    dr_enable: bool = True

    # ── 12a. Actuator / drivetrain physics (per-joint, per-episode) ──
    dr_stiffness_mult_min: float = 0.85    # impact: MED  (Kp tolerance)
    dr_stiffness_mult_max: float = 1.15
    dr_damping_mult_min: float = 0.7       # impact: LOW  (decoupled from Kp in v7)
    dr_damping_mult_max: float = 1.3
    dr_effort_mult_min: float = 0.9        # impact: LOW  (narrow, capped at nominal)
    dr_effort_mult_max: float = 1.0
    # P7: armature now DOMINATES the leg inertia (rotor ≈ 33/36e-3 vs links
    # 17/2.5e-3 about hip/knee) — retagged LOW→HIGH. The 0.7–1.3 band also
    # covers the 40% hip-vs-knee rotor estimate disagreement (same motor
    # model, commanded-τ scale degeneracy — see actuator factory header).
    # Topology: see dr_armature_per_leg (one draw per co-axial pair, 09-18).
    dr_armature_mult_min: float = 0.7      # impact: HIGH (was LOW pre-P7)
    dr_armature_mult_max: float = 1.3
    dr_joint_friction_mult_min: float = 0.5    # impact: MED  (Coulomb / breakaway)
    dr_joint_friction_mult_max: float = 2.0
    dr_dyn_friction_mult_min: float = 0.5      # impact: LOW  (kinetic)
    dr_dyn_friction_mult_max: float = 2.0
    dr_visc_friction_mult_min: float = 0.5     # impact: LOW  (damping vs ω)
    dr_visc_friction_mult_max: float = 2.0
    # voltage_factor: ONE scalar per robot (one battery) scaling all
    # three joints' velocity_limit. impact: LOW (only bites near v_limit,
    # which a balancing robot rarely reaches).
    dr_voltage_factor_min: float = 0.87
    dr_voltage_factor_max: float = 1.07
    # Actuator-lag prefilter DR (per-joint, per-episode). Multiplies the
    # identified ω / ζ centrals in §11. The DR is CENTERED on the sysid values
    # (mult midpoint = 1.0); bands are modest.
    #   ω  ── ±18%: the rise-time / bulk-lag bandwidth. impact: MED (it's
    #         effectively a command-bandwidth knob; phase-margin-adjacent but
    #         the bulk lag was uniform across joints in sysid, so well pinned).
    #   ζ  ── ±30% (wider): the overshoot, and the most LOAD-sensitive of the
    #         two — a given joint's ring changes with the inertia hanging off
    #         it (stance vs swing, pickup), so the band is deliberately wider
    #         than ω. impact: MED. Stays underdamped (<1) across the whole band
    #         for all three joints (max ζ = 0.68·1.30 = 0.88), so the prefilter
    #         keeps some ring everywhere — it never collapses to an overdamped
    #         pure-lag that would smooth the ring off (the failure the per-joint
    #         ζ split was introduced to avoid).
    # P7: pinned at 1.0 (act_lag disabled — see §11). Sampling still runs,
    # returns exactly 1.0, priv-obs slots [61:73] become constants.
    dr_act_lag_omega_mult_min: float = 1.0
    dr_act_lag_omega_mult_max: float = 1.0
    dr_act_lag_zeta_mult_min:  float = 1.0
    dr_act_lag_zeta_mult_max:  float = 1.0

    # ── 12b. Control-loop latency (per-env, per-episode) ── impact: HIGH
    # Combined round-trip [action_delay + obs_delay] = effective loop
    # dead time, the prime destabilizer for a balancing controller. Every
    # delay below is quantized to the 10 ms control step.
    #
    # Obs latency is SPLIT by sensor (v7): the IMU and the joint encoders
    # are on different buses and were measured ~10 ms apart, so each gets
    # its own per-env delay (_env_imu_delay / _env_joint_delay). A single
    # shared obs delay (the old design) can't represent the IMU-leads-
    # encoder skew the real robot actually has.
    #
    # BENCH MEASUREMENT (P1, CAN2SPI, 2026-08-17 — replant report):
    #     action delay  2.4-2.8 ms  → ⌈x/10⌉ = 1 step floor (never instant)
    #     IMU obs       ~2 ms lead  → 1 step
    #     joint obs     IMU+1.96 ms → 1 step (old bus: 15-20 ms → 2)
    # The 15 ms-era numbers shrank ~10× on CAN2SPI but round into the same
    # 1-step buckets; only the joint path moves (2 → 1): its old value
    # modeled a 10 ms IMU-vs-encoder skew that M1.2 measured at 1.96 ms.
    dr_action_delay_min: int = 1     # was 0 — never instant (M1.1: 2.4-2.8 ms)
    dr_action_delay_max: int = 2
    # IMU path (proj_grav, ang_vel, lin_vel). Locked at 1 (never 0) so the
    # balance loop can't learn a zero-latency reflex it won't have on
    # hardware (that was the 2 Hz wobble).
    dr_imu_delay_min: int = 1
    dr_imu_delay_max: int = 1
    # Joint/encoder path (joint_pos, joint_vel) over CAN2SPI. P7: 2 → 1.
    # M1.2 measured the encoder trailing the IMU by 1.96 ms = 0.2 ticks —
    # holding it a full tick behind overstated the skew ~5×. (Caveat: M1.2
    # pinned the RELATIVE skew only; 1 is the runbook round-up floor.)
    dr_joint_delay_min: int = 1
    dr_joint_delay_max: int = 1
    # Handover-contract randomization (S-line, 2026-09-06). At reset the
    # obs-delay buffer is zeroed, so the first delayed sensor frame(s) are
    # zeros and the ×5 history is primed with that = the "blind" contract
    # (deploy 'v'). Hardware can also hand over with the real first frame
    # primed ×5 (deploy 'b'). With probability handover_prime_prob an env's
    # delay buffer is instead filled with its real first sensor frame, so
    # the policy is trained for either opening. 0.0 = blind only (legacy).
    handover_prime_prob: float = 0.0

    # ── 12c. Per-step sensor noise (additive uniform, ± value) ── impact: LOW
    # Real measured stds (hanging static): proj_grav.xy ~0.0007, ang_vel
    # ~0.007 rad/s. DR set ~3-7× the measured floor for headroom.
    dr_obs_noise_proj_grav: float = 0.05
    dr_obs_noise_ang_vel:   float = 0.15
    # joint_vel: couldn't measure on hardware (stationary → zero floor).
    # Blind prior ±0.3 rad/s typical for 100 Hz finite-diff encoder.
    dr_obs_noise_joint_vel: float = 0.6
    dr_obs_noise_joint_pos: float = 0.008   # ± rad, per-step

    # ── 12d. Per-episode sensor bias (constant offset, per-env) ── impact: LOW
    # NOT physics — only the obs the policy reads is biased. Models
    # imperfect zeroing/calibration; policy must be robust to an unknown
    # but constant offset per sensor.
    dr_joint_pos_bias_max: float = 0.07    # ± rad, encoder zero offset
    # P9 (2026-08-18): 0.03 → 0.10. The supine IMU cross-check is physically
    # impossible (the Pi + wiring live on the torso's back — it cannot rest
    # there), so the crawl-derived mounting offset (−0.032 ± 0.03) cannot be
    # independently confirmed and IMU_PG_OFFSET stays at the deployed −0.089.
    # Under that trim the policy's effective pg_x bias on hardware is
    # ~0.07–0.11 — the flying G2i variant (…_imuoffset010) trained with 0.10
    # for exactly this reason. Revert to 0.03 only after a trim recalibration
    # is actually validated on hardware.
    dr_proj_grav_bias_max: float = 0.10    # ± unitless, IMU mounting/cal
    # EVAL-ONLY probe (P12 08-25): pin the pg_x obs bias to an exact value
    # for every env (overrides the uniform draw's x component; y/z still
    # sampled from bias_max). NaN = disabled (default — training never
    # sets this). Used to replicate the measured hardware obs offset
    # (+0.10 at physical upright) in a hw_matched eval.
    dr_proj_grav_bias_pin_x: float = float("nan")
    dr_ang_vel_bias_max:   float = 0.03    # ± rad/s, gyro bias

    # ── 12d-bis. Execution-noise injection (P13, 2026-08-25) ──
    # Random error on the FINAL joint position target the motors track —
    # invisible to the policy except through dynamics. Trains catch-steps
    # under imperfect actuation (torque-space-perturbation idea, arXiv
    # 2504.06585, realized in target space so it flows through the belt
    # actuator's own torque limits/model). Two components:
    #   bias: per-env, per-joint, episode-constant (drivetrain offsets,
    #         belt tooth seating, zero drift — the 08-25 tooth skip is
    #         0.21 rad of exactly this class)
    #   std:  per-control-step white (torque ripple, contact chatter)
    # kp maps rad → Nm (hip 20: 0.05 rad ≈ 1 Nm; knee 30: ≈ 1.5 Nm).
    # Defaults 0.0 = bit-exact inert for every existing line.
    exec_noise_target_bias_max: float = 0.0   # ± rad, episode-constant
    exec_noise_target_std: float = 0.0        # rad, per-step white

    # ── 12d-ter. Knee joint-angle bias (S-line, 2026-09-06) ──
    # A persistent offset between the knee ENCODER angle and the TRUE joint
    # angle: belt tooth skip (1 tooth = 0.21 rad joint-side, observed
    # 08-25 and 09-03), belt stretch, crawl-zero error. Unlike the two
    # independent knobs above (obs bias, target bias), a real encoder
    # offset couples them: the policy reads q_true − δ AND the motor-side
    # PD holds the encoder value on target, so q_true = target + δ.
    # Sampled per leg per episode, folded into _dr_joint_pos_bias (−δ) and
    # _exec_target_bias (+δ) for the knee slots. 0.0 = bit-exact inert.
    dr_knee_angle_bias_max: float = 0.0       # ± rad, per knee
    # Same coupled encoder/servo offset for the hip-pitch joints (slots 1, 4),
    # added 2026-09-10 for the plant sysid (scripts/rsl_rl/sysid_optuna.py).
    # 0.0 = bit-exact inert.
    dr_hip_angle_bias_max: float = 0.0        # ± rad, per hip pitch

    # ── 12e. World / body physics (per-env, per-episode) ──
    # Floor friction multiplier (friction_combine_mode="multiply", so
    # effective μ scales uniformly). 0.5–1.1 covers carpet → laminate →
    # rubber mat. impact: HIGH (traction gates everything for legs).
    dr_floor_friction_min: float = 0.5
    dr_floor_friction_max: float = 1.1
    # Body mass multiplier — single per-env scalar on ALL link masses.
    # Captures cable/battery routing + USD vs as-built tolerance; affects
    # torque budget and gravity comp. impact: MED.
    # P6 weigh-in (2026-08-18): hardware 5.5 kg vs URDF 5.195 = ×1.059 →
    # recentered on the measured ratio, same ±0.12 width. The missing
    # ~0.30 kg's LOCATION is unknown — keep dr_com_offset honest.
    dr_mass_mult_min: float = 0.94
    dr_mass_mult_max: float = 1.18
    # EVAL-ONLY pin (2026-09-10): extra mass [kg] ADDED to the top bodies only
    # (torso2hip + both hip-roll chassis blocks, split in proportion to their
    # nominal masses), on top of dr_mass_mult. Models the user's hypothesis
    # that the weigh-in surplus (wires, fasteners) sits on the upper body,
    # which the uniform mass_mult cannot express. CoM positions and link
    # inertia tensors are left unchanged (mass only). 0 = off.
    dr_top_mass_add_kg: float = 0.0
    # Scale each body's inertia tensor by its mass ratio at every DR write
    # (2026-09-10). PhysX set_masses does NOT touch inertia; every run before
    # this date trained with URDF-nominal link tensors under every mass draw
    # (the m·r² terms did scale — only the bodies' own tensors were fixed).
    # Uniform-density assumption, same as Isaac Lab randomize_rigid_body_mass.
    dr_mass_recompute_inertia: bool = True
    # Per-link CoM offset — independent ± per body, per axis. Shifts each
    # link's effective rotational dynamics / balance point. impact: MED.
    # NOTE (2026-09-06): because every link/axis is drawn independently,
    # the WHOLE-BODY CoM barely moves (std ≈ 3–4 mm at 0.02) — this knob
    # never trained a persistent balance-point bias. Use dr_com_shift_*.
    dr_com_offset_max: float = 0.015   # ± m
    # Coherent whole-body CoM shift (S-line, 2026-09-06) — ONE per-env
    # offset of the total CoM in the base (torso/IMU) frame, x = forward,
    # y = lateral. Realized on the torso link only, scaled by M/m_torso
    # (≈3.06) so the whole-body CoM moves by exactly the sampled amount
    # regardless of leg pose (link frames are arbitrary Onshape frames, so
    # a base-frame shift cannot be applied exactly to the leg links). This
    # is the plant-side counterpart of the deploy IMU trim knob
    # (docs/p12-4-session-20260903.md §3). 0.0 = bit-exact inert.
    dr_com_shift_x_max: float = 0.0    # ± m, whole-body CoM_x
    dr_com_shift_y_max: float = 0.0    # ± m, whole-body CoM_y
    dr_com_shift_z_max: float = 0.0    # ± m, whole-body CoM_z (up), 2026-09-10
    # ── 12f. Ground slope simulation (per-env, per-episode) ── impact: HIGH-expected
    # Equivalent horizontal force from standing on a tilted floor.
    # Real indoor floors have 0.5–2° of slope; this is invisible in sim
    # but shifts the balance point persistently (unlike pushes). Sampled
    # as uniform [0, max_deg] with random direction per env per episode.
    dr_gravity_tilt_max_deg: float = 1.0
    #new

    # Idle stepping incentive. The gait rewards (gait_step/feet_air/foot_swing)
    # are gated to 0 by cmd_gate at |vx_cmd|<0.05, so at rest there is no reason
    # to step. A point-foot biped has no static sagittal balance, so this floor
    # keeps a stepping cadence rewarded at zero command. no_flight/stance_symmetry
    # ride the same floor so the idle gait is alternating single-support, not a hop.
    # 0.0 = old behavior; raise toward 0.5 if the robot stays static at cmd=0.
    idle_step_gate: float = 0.5
    # Relief for the action-derivative penalties at zero command. snap/smoothness
    # tax exactly the fast-reversing action a catch-step needs, so at rest they
    # cancel the stepping reward. Fade to this fraction at cmd=0; full while walking.
    idle_motion_relief: float = 0.08
