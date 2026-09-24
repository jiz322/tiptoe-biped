"""belt_knee_actuator.py — "firmware replica" DCMotor for the belt-driven knee.

The real knee motor sits on the hip chassis and drives the knee joint through
a 22:30 belt, so the knee's MOTOR coordinate is coupled to hip pitch:

    m = (q_kn + c * q_hp) / G          (sim-space; hardware signs absorbed)

with G = 22/30 and c = config.py KNEE_HIP_COUPLING (0.733). The stock sim
model (plain DCMotor on the knee joint with belt-scaled scalars) misses three
firmware-side consequences that motor_bridge.py + the RobStride MIT-mode PD
create on hardware:

  1. Belt drag — the firmware Kd acts on MOTOR velocity, which contains
     c*qd_hp/G. In joint space: an extra -kd*c*qd_hp on the knee whenever
     the hip moves. Confirmed by the 2026-08-06 teacher-forcing replay
     (--coupling-terms knee: TF pos RMSE 0.0599 -> 0.0581).
  2. Stale-hip target — motor_bridge._sim_to_motor bakes the hip position
     MEASURED AT SEND TIME into the knee motor target; firmware holds it for
     the whole 10 ms tick while the hip keeps moving. Phantom knee PD error
     kp*c*(q_hp_latch - q_hp) within the tick.
  3. Motor-velocity torque envelope — the DC torque-speed curve saturates on
     motor speed (q_kn_dot + c*q_hp_dot)/G, not knee joint speed. Evaluating
     the existing joint-frame envelope (sat 23.2 = 17/G, vlim 20 = 27.5*G) at
     the belt-effective velocity q_kn_dot + c*q_hp_dot is mathematically the
     motor-frame envelope.

The deployed KNEE_DECOUPLE_FF patch (config.py, live since 2026-08-06) sends
knee v_des = c*qd_hp*sign/G, cancelling drag term 1 with the TICK-START hip
velocity. `knee_decouple_ff=True` replicates that: the damping reference
becomes c*qd_hp_latch, so only the within-tick velocity change leaks through
— exactly like hardware.

The ideal J^T belt reaction on the hip (+c*tau_knee) is behind `hip_reaction`
and OFF by default: the 2026-08-06 replay showed the real hip does NOT feel
it (injecting it made the hip match worse by +0.02 rad one-step RMSE).

ROTOR INERTIAL COUPLING (P7, 2026-08-18). The knee rotor spins at
th_m = (q_kn + c*q_hp)/G, so its kinetic energy contributes the full
joint-space inertia block

    dM = (I_r/G^2) * [[c^2, c], [c, 1]]      (order [hip, knee])

PhysX `armature` is diagonal-only: the diagonals (c^2*I_r/G^2 into hip,
I_r/G^2 into knee) live in the cfg armature values, and the OFF-DIAGONAL
was silently dropped — pre-P7 that was 5.7e-3 and looked ignorable, but the
P5/P7 bench decomposition puts I_r/G^2 at ~36e-3, making the off-diagonal
c*I_r/G^2 ~ 26e-3 (2026-09-18: the rotor is the VENDOR 4.2e-3 again, so
m_x is back at ~5.7e-3; the 36e-3 was an apparent-inertia fit — see ARMATURE
DERIVATION below). On the 08-13 fall logs that term, at the 36e-3 figure, was
a hip torque the same size as the hip's own commands (median rms ratio 0.96, peaks ~5 Nm at
t=0.16-0.20 s — the fall-initiation window). `rotor_coupling=True` applies
it as an explicit plant-side generalized force with a one-physics-step
finite-difference acceleration estimate:

    tau_hp += -m_x * qdd_kn,   tau_kn += -m_x * qdd_hp,
    m_x = c * armature_knee    (per-env DR draw => coupling scales with it)

Applied AFTER _clip_effort: it is an inertial reaction between plant bodies,
not motor output — it must not consume the DC envelope. Stability: the
implied 2x2 mass matrix stays PD for all DR draws (det > 0 at worst-case
0.7/1.3 corners), and the one-step lag is far inside the stable region
(m_x^2 / (M_hp*M_kn) ~ 0.36) at the current 2000 Hz physics (sim.dt=1/2000,
decimation=20, hold_decimation=20); it was already so at the 500 Hz rate these
constants were first tuned under.

Cadence invariant: `compute()` must be called exactly `hold_decimation` times
per control tick (true for DirectRLEnv's decimation loop and for
data/sysid_replay_policy_log.py's substep loop; teacher-forcing
write_joint_state_to_sim/update calls do NOT trigger compute). The stale-hip
latch keys off a global substep counter under that assumption. `hold_decimation`
must ALSO equal the env's `decimation` — on a rate change, both move together.

Hip-pitch columns in the group get stock DCMotor behavior. All per-env DR
buffers (stiffness/damping/effort_limit/armature/frictions/velocity_limit)
are the inherited ActuatorBase tensors, so _write_per_env_actuators_to_sim
works unchanged — EXCEPT that the env must multiply its armature write by
`armature_write_mask()`: for environments running the optional series path
below, the reflected rotor inertia moves out of the PhysX joint and into the
series integrator, and the hip-pitch diagonal loses the c²·I_r/G² that was
only there while the belt was rigid — but KEEPS the hip motor's own rotor
(mask = own/total on the hip column, derived from the cfg; 2026-09-18.  Until
then the hip column was zeroed, i.e. every S5/S7 belt-live env ran with NO
hip_pitch armature at all).

SERIES ELASTICITY (2026-09-11, default OFF). A motor-side spring/damper in the
belt transmission, for the S4 plant sweep. Parameters are set per episode with
set_series_parameters(); stiffness arrives motor-side, and the env converts the
knee-referred CLI value with G² (damping needs no conversion — it is a torque
over velocity). While a pair is active:

  * the servo acts on the virtual motor coordinate m, not on the knee, so the
    belt can deflect under load;
  * the state fed to the joint is the spring reaction, and the hip sees only
    the transmission's internal residual, so the belt's stored energy is a
    plant force rather than motor output and is not re-clipped by the DC
    envelope;
  * the reflected rotor armature is removed from PhysX (see the mask above);
  * `encoder_joint_state()` reproduces what the deploy stack would actually
    read — the motor encoder decoded through the belt model — which is what
    the environment's knee observation channels use in this mode.

ARMATURE DERIVATION (2026-09-18).  The hip_pitch and knee PhysX armatures are
no longer two hand-entered literals: `derive_armature()` below is the ONE place
that turns the given facts into the rotor-inertia block, and the env build
path, both cfg factories and `BELT_MOTOR_INERTIA` all consume it.

  Convention (printed at every build, never implicit): `rotor_inertia_output`
  is the RS01's rotor inertia REFLECTED TO ITS OUTPUT SHAFT (through the
  motor's internal 7.75:1 reduction), i.e. in the MOTOR frame of the series
  integrator, before the 22:30 belt.  Value: the VENDOR spec, 4.2e-3 kg m^2
  (RS01: 17 Nm peak, 7.75:1, ~380 g).  The P7/P12 bench figures (13.9e-3 hip /
  19.4e-3 knee / 19.3e-3 recenter) are APPARENT inertias — bench total minus
  URDF links — and are kept only as historical fits: 19.3e-3 would need ~75 %
  of the motor mass spinning at the housing radius, so the bench SIGNAL stands
  but its attribution to the rotor does not (re-fit with the rotor pinned:
  data/coastdown_fit.py --pinned-inertia).

      knee_armature      = I_out / G^2               (joint side, 4.2e-3 -> 7.81e-3)
      hip share          = c^2 * knee_armature       (the knee rotor riding the hip)
      hip_pitch_armature = own_rotor + hip share     (4.2e-3 + 4.2e-3 = 8.40e-3)
      off-diagonal m_x   = c * knee_armature         (rotor_coupling: 5.72e-3)
      BELT_MOTOR_INERTIA = G^2 * knee_armature = I_out (series integrator rotor)

  so the whole block is dM = knee_armature * [[c^2, c], [c, 1]], order
  [hip_pitch, knee], plus the hip motor's own rotor on the hip diagonal.  The
  hip motor is direct drive, so its own output-shaft inertia is already joint
  side; by default it is the same RS01 and therefore the same I_out.

  These defaults are the ORIGINAL P6 plant (rl_env_cfg_p6 header: armature
  7e-4 / 4.2e-3 / 7.8e-3) and DIFFER from every K/L/R/S/S7 plant: hip
  33.2e-3 (P7 pins) / 38.5e-3 (L-line pin) -> 8.4e-3, knee 36.0e-3 -> 7.8e-3,
  series rotor 19.36e-3 -> 4.2e-3.  No checkpoint was trained on it.  An
  explicit cfg pin (hip_pitch_armature / knee_armature >= 0) still wins, so
  every pre-change recipe stays reachable bit-for-bit (hip 0.0385 + knee
  0.036 = the L/R/S/S7 plant; hip 0.0332 + knee 0.036 = P7/K).

  c (0.733, control/config.py KNEE_HIP_COUPLING) and G (22/30) are both
  pulley-geometry ratios that differ in the fourth digit.  They are consumed
  here from ONE module constant each (BELT_COUPLING, BELT_G); unifying them
  is a separate call and neither value is changed by the derivation.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import math

import torch

# 22:30 belt reduction.  Joint-referred quantities scale by G^2 (torque once,
# angle once); a torque-over-velocity ratio carries no power of G.  Keep in
# step with control/config.py KNEE_GEAR_RATIO and rl_env_p6._BELT_G2.
BELT_G = 22.0 / 30.0
BELT_G2 = BELT_G ** 2
# Belt coupling coefficient c (control/config.py KNEE_HIP_COUPLING): the knee
# motor coordinate is m = (q_kn + c*q_hp)/G.  Numerically close to G but NOT
# the same constant — see the ARMATURE DERIVATION note in the module docstring.
BELT_COUPLING = 0.733
# RS01 rotor inertia reflected to the motor OUTPUT SHAFT (motor frame, before
# the belt), kg m^2, one value for both RS01s.  VENDOR SPEC (RS01: 17 Nm peak,
# 7.75:1 internal reduction, ~380 g).  The bench-fit figures that stood here
# before 2026-09-18 — P7 hip 13.9e-3 (CI 12.5-15.7) / knee 19.4e-3 (CI
# 18.3-20.5), P12 recenter 19.3e-3 ("rotors ~19e-3 at output, both") — are
# retained only as historical APPARENT-inertia fits (bench total minus URDF
# links); a 19.3e-3 rotor would need ~75 % of the motor mass spinning at the
# housing radius.  This vendor value is the fact the armatures derive from.
ROTOR_INERTIA_OUTPUT = 4.2e-3


@dataclasses.dataclass(frozen=True)
class ArmatureDerivation:
    """Resolved hip_pitch/knee armature block and the facts it came from.

    All inertias in kg m^2.  ``knee`` and ``hip_pitch`` are the joint-side
    PhysX armature values; ``rotor_motor_side`` is the same knee rotor in the
    series integrator's motor frame (= G^2 * knee).  ``*_source`` is
    "derived" or "pinned".
    """
    rotor_inertia_output: float
    hip_own_rotor: float
    coupling: float
    gear_ratio: float
    knee: float
    hip_pitch: float
    knee_source: str
    hip_source: str

    @property
    def hip_share(self) -> float:
        """c^2 * knee: the knee rotor's contribution to the hip diagonal."""
        return self.coupling ** 2 * self.knee

    @property
    def off_diagonal(self) -> float:
        """c * knee: the [hip, knee] mass-matrix entry rotor_coupling applies."""
        return self.coupling * self.knee

    @property
    def rotor_motor_side(self) -> float:
        """G^2 * knee: the knee rotor in the motor frame (series integrator)."""
        return self.gear_ratio ** 2 * self.knee

    @property
    def hip_own_share(self) -> float:
        """own_rotor / hip_pitch, in [0, 1]: the fraction of the hip diagonal
        that stays in PhysX when the belt goes elastic (the rest rides the
        belt into the series integrator).  0 when the hip armature is 0."""
        if self.hip_pitch <= 0.0:
            return 0.0
        return min(1.0, max(0.0, self.hip_own_rotor / self.hip_pitch))

    def describe(self) -> str:
        """One-line derivation, for the startup print."""
        return (
            f"I_out={self.rotor_inertia_output * 1e3:.3f}e-3 (RS01 output shaft, motor frame, "
            f"pre-belt)  G={self.gear_ratio:.5f}  c={self.coupling:.4f}  "
            f"own_rotor(hip)={self.hip_own_rotor * 1e3:.3f}e-3 | "
            f"knee {'= I_out/G^2 ' if self.knee_source == 'derived' else ''}= {self.knee * 1e3:.3f}e-3 [{self.knee_source}]  "
            f"hip_pitch {'= own + c^2*knee = ' + f'{self.hip_own_rotor * 1e3:.3f}e-3 + {self.hip_share * 1e3:.3f}e-3 ' if self.hip_source == 'derived' else ''}"
            f"= {self.hip_pitch * 1e3:.3f}e-3 [{self.hip_source}]"
            f"{'' if self.hip_source == 'derived' else f' (own {self.hip_own_rotor * 1e3:.3f}e-3 + c^2*knee {self.hip_share * 1e3:.3f}e-3 = {(self.hip_own_rotor + self.hip_share) * 1e3:.3f}e-3 if derived)'}  "
            f"off-diag c*knee = {self.off_diagonal * 1e3:.3f}e-3  "
            f"series rotor G^2*knee = {self.rotor_motor_side * 1e3:.3f}e-3"
        )


def derive_armature(
    rotor_inertia_output: float = ROTOR_INERTIA_OUTPUT,
    hip_own_rotor: float = float("nan"),
    hip_pitch_pin: float = -1.0,
    knee_pin: float = -1.0,
    coupling: float = BELT_COUPLING,
    gear_ratio: float = BELT_G,
) -> ArmatureDerivation:
    """The single source of the hip_pitch/knee armature block.

    ``rotor_inertia_output``: knee RS01 inertia at its output shaft (motor
    frame, pre-belt, kg m^2).  ``hip_own_rotor``: the hip RS01's own
    output-shaft inertia — direct drive, so already joint side; NaN (the
    default) means "same motor model, same value as rotor_inertia_output".
    ``hip_pitch_pin`` / ``knee_pin`` >= 0 pin that joint EXACTLY (the
    pre-change cfg semantics; a negative value derives).  A derived hip uses
    the RESOLVED knee (pinned or derived) so the block stays coherent.
    """
    for name, v in (("rotor_inertia_output", rotor_inertia_output),
                    ("coupling", coupling), ("gear_ratio", gear_ratio)):
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError(f"{name} must be finite and > 0, got {v!r}")
    own = rotor_inertia_output if math.isnan(hip_own_rotor) else float(hip_own_rotor)
    if not math.isfinite(own) or own < 0.0:
        raise ValueError(f"hip_own_rotor must be finite and >= 0 (or NaN), got {hip_own_rotor!r}")
    if knee_pin >= 0.0:
        knee, knee_src = float(knee_pin), "pinned"
    else:
        knee, knee_src = rotor_inertia_output / gear_ratio ** 2, "derived"
    if hip_pitch_pin >= 0.0:
        hip, hip_src = float(hip_pitch_pin), "pinned"
    else:
        hip, hip_src = own + coupling ** 2 * knee, "derived"
    return ArmatureDerivation(
        rotor_inertia_output=float(rotor_inertia_output), hip_own_rotor=own,
        coupling=float(coupling), gear_ratio=float(gear_ratio),
        knee=knee, hip_pitch=hip, knee_source=knee_src, hip_source=hip_src,
    )


# Motor-side rotor inertia of the series integrator = G^2 * knee armature,
# which by construction is ROTOR_INERTIA_OUTPUT itself (the knee armature is
# I_out/G^2).  Written as the product so the identity is what the code says.
# Consumers restating this literal (sweep scripts, data/_belt_cosim_stability)
# carry ROTOR_INERTIA_OUTPUT, and data/_test_armature_derivation.py checks it.
BELT_MOTOR_INERTIA = BELT_G2 * derive_armature(ROTOR_INERTIA_OUTPUT).knee


def damping_for_ratio(knee_stiffness_joint: float, damping_ratio: float,
                      rotor_inertia_motor: float = BELT_MOTOR_INERTIA) -> float:
    """Motor-side belt damping (N m s/rad) for a target transmission damping ratio.

    The belt's own mode is wn = sqrt(k_motor / I_rotor_motor) with
    k_motor = k_joint * G^2, so

        d_motor = 2*zeta*sqrt(k_joint * G^2 * I_rotor_motor)

    and zeta = 1 is the transmission critically damped against its own rotor
    inertia, NOT against the robot mass.  Entering damping this way is what
    makes a sweep readable: a raw N m s/rad number says nothing about whether
    the belt rings or is over-damped, while zeta does, and it moves with the
    stiffness cell the way the physics does.
    """
    if knee_stiffness_joint <= 0.0:
        raise ValueError(
            f"a damping ratio needs a positive knee stiffness, got {knee_stiffness_joint!r}"
        )
    if damping_ratio < 0.0 or not math.isfinite(damping_ratio):
        raise ValueError(f"damping ratio must be finite and >= 0, got {damping_ratio!r}")
    if rotor_inertia_motor <= 0.0:
        raise ValueError(f"rotor inertia must be > 0, got {rotor_inertia_motor!r}")
    return 2.0 * damping_ratio * math.sqrt(
        knee_stiffness_joint * BELT_G2 * rotor_inertia_motor
    )

def encoder_write_slots(sim_joint_names, pairs):
    """Sim-array column for each knee in ``pairs`` (dict knee -> hip), in pair order.

    The env writes the decoded encoder value into ``robot.data.joint_pos``,
    which is in the ARTICULATION's joint order — not the env's own
    hip_roll/hip_pitch/knee listing and not the actuator group's.  Resolving
    the slot from anything but the sim's name list is the bug that put knee1's
    encoder reading into hip_pitch1's observation (found 2026-09-16: PhysX
    orders this robot hr1, hr2, hp1, hp2, kn1, kn2; the env lists hr1, hp1,
    kn1, ..., so "index 2" is knee1 in one list and hip_pitch1 in the other).
    """
    names = list(sim_joint_names)
    slots = []
    for kn_name in pairs.keys():
        if kn_name not in names:
            raise ValueError(f"knee joint {kn_name!r} not in sim joint names {names}")
        slots.append(names.index(kn_name))
    return slots


from isaaclab.actuators import DCMotor, DCMotorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class BeltKneeDCMotor(DCMotor):
    """DCMotor whose knee columns replicate the belt-drivetrain firmware PD."""

    cfg: BeltKneeDCMotorCfg

    def __init__(self, cfg: BeltKneeDCMotorCfg, *args, **kwargs):
        # Skip DCMotor.__init__ (it treats saturation_effort as a scalar and
        # would choke on the per-joint dict) — run the ActuatorBase init, then
        # replicate DCMotor's setup with per-joint parsing.
        super(DCMotor, self).__init__(cfg, *args, **kwargs)
        if self.cfg.saturation_effort is None:
            raise ValueError("The saturation_effort must be provided for the DC motor actuator model.")
        if self.cfg.velocity_limit is None:
            raise ValueError("The velocity limit must be provided for the DC motor actuator model.")
        self._saturation_effort = self._parse_joint_parameter(self.cfg.saturation_effort, None)
        self._vel_at_effort_lim = self.velocity_limit * (1 + self.effort_limit / self._saturation_effort)
        self._joint_vel = torch.zeros_like(self.computed_effort)
        self._zeros_effort = torch.zeros_like(self.computed_effort)

        # Resolve knee/hip column pairs BY NAME — never trust group ordering.
        kn_cols, hp_cols = [], []
        for kn_name, hp_name in self.cfg.pairs.items():
            if kn_name not in self.joint_names or hp_name not in self.joint_names:
                raise RuntimeError(
                    f"BeltKneeDCMotor pair ({kn_name}, {hp_name}) not covered by this "
                    f"actuator group (joints: {self.joint_names}). The group's "
                    "joint_names_expr must span both knees and both hip pitches."
                )
            kn_cols.append(self.joint_names.index(kn_name))
            hp_cols.append(self.joint_names.index(hp_name))
        self._kn_cols = torch.tensor(kn_cols, dtype=torch.long, device=self._device)
        self._hp_cols = torch.tensor(hp_cols, dtype=torch.long, device=self._device)

        # Own-rotor share of the hip_pitch armature, own/total per pair, for
        # armature_write_mask() under a live series pair.  Taken from the
        # NOMINAL (cfg, undrawn) armature — self.armature is exactly that at
        # construction — because the per-env DR draw scales own and total by
        # the same multiplier (one draw per leg), so the ratio is invariant.
        # NaN cfg own-rotor = the residual hip - c^2*knee of the nominal
        # dict, which IS the own rotor by construction of derive_armature()
        # (P7 pins: 33.2 - c^2*36.0 = 13.86e-3 -> 0.42; derived: 0.50).
        hip_nom = self.armature[:, self._hp_cols]
        knee_nom = self.armature[:, self._kn_cols]
        own_cfg = float(self.cfg.hip_own_rotor_armature)
        if math.isnan(own_cfg):
            own_t = (hip_nom - self.cfg.coupling ** 2 * knee_nom).clamp_min(0.0)
        else:
            if own_cfg < 0.0:
                raise ValueError(f"hip_own_rotor_armature must be >= 0 or NaN, got {own_cfg!r}")
            own_t = torch.full_like(hip_nom, own_cfg)
        share = torch.where(hip_nom > 0.0, own_t / hip_nom.clamp_min(1e-12), torch.zeros_like(hip_nom))
        self._hip_own_share = share.clamp(0.0, 1.0)

        # Stale-hip latch state (tick-start hip pos/vel, per knee).
        n_pairs = len(kn_cols)
        self._hp_latch_pos = torch.zeros(self._num_envs, n_pairs, device=self._device)
        self._hp_latch_vel = torch.zeros(self._num_envs, n_pairs, device=self._device)
        self._latch_valid = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        self._substep = 0

        # Rotor-coupling FD-acceleration state (previous physics-step velocity).
        self._prev_vel = torch.zeros_like(self.computed_effort)
        self._prev_vel_valid = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)

        # Optional motor-side series-elastic transmission state.  The rigid
        # path above remains the exact default; these buffers are touched only
        # for environments explicitly enabled through set_series_parameters().
        n_pairs = len(kn_cols)
        self._series_enabled = torch.full(
            (self._num_envs, n_pairs), bool(self.cfg.series_elasticity),
            dtype=torch.bool, device=self._device,
        )
        self._series_stiffness = torch.full(
            (self._num_envs, n_pairs), float(self.cfg.series_stiffness), device=self._device
        )
        self._series_damping = torch.full(
            (self._num_envs, n_pairs), float(self.cfg.series_damping), device=self._device
        )
        self._series_rotor_inertia = torch.full(
            (self._num_envs, n_pairs), float(self.cfg.series_rotor_inertia), device=self._device
        )
        if not math.isfinite(cfg.series_dt) or cfg.series_dt <= 0.0:
            raise ValueError("series_dt must be finite and > 0")
        if not isinstance(cfg.series_substeps, int) or cfg.series_substeps < 1:
            raise ValueError("series_substeps must be a positive integer")
        # motor encoder coordinate m and speed mdot, in motor radians/rad/s.
        self._series_motor_pos = torch.zeros(self._num_envs, n_pairs, device=self._device)
        self._series_motor_vel = torch.zeros(self._num_envs, n_pairs, device=self._device)
        self._series_state_valid = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        # Exposed for rollout diagnostics; both are in knee-equivalent N m.
        self.series_motor_effort = torch.zeros_like(self._series_motor_pos)
        self.series_spring_effort = torch.zeros_like(self._series_motor_pos)
        # Latest motor-frame command after the DC envelope, in MOTOR units
        # (G * joint-equivalent N m).  Kept as a buffer so the hip-reaction
        # branch below never has to guess whether a local is in scope.
        self._series_tau_motor_cmd = torch.zeros_like(self._series_motor_pos)
        self._series_any_enabled = False
        # Seed from the cfg, but only ENABLE where a stiffness was actually
        # seeded: the env may turn series_elasticity on while leaving the
        # stiffness band at 0 and pin a real value per episode later (the
        # sweep does exactly that).  Enabling with zero stiffness would mean a
        # belt with no spring, which the parameter guard rejects by design.
        self.set_series_parameters(
            slice(None), cfg.series_stiffness, cfg.series_damping,
            bool(cfg.series_elasticity) and float(cfg.series_stiffness) > 0.0,
            cfg.series_rotor_inertia,
        )

    def set_series_parameters(
        self,
        env_ids: Sequence[int] | slice | None,
        stiffness: torch.Tensor | float,
        damping: torch.Tensor | float,
        enabled: torch.Tensor | bool = True,
        rotor_inertia: torch.Tensor | float | None = None,
    ):
        """Set per-episode belt spring parameters.

        ``stiffness`` and ``damping`` are motor-side values (N m/rad and
        N m s/rad); ``rotor_inertia`` is motor-side kg m². Call only at an
        episode reset: parameters are held and the motor state is invalidated.
        ``enabled=False`` selects the untouched rigid implementation. Inputs
        broadcast from scalar, [pairs], [envs], [envs, 1], or [envs, pairs].
        A bare ``[envs]`` is accepted and treated as per-env-both-pairs,
        because that is the natural shape for a mask and it is NOT
        broadcastable to [envs, pairs] (size `envs` cannot become `pairs`).

        A pair is inactive unless ``enabled`` AND ``stiffness > 0``.  Setting
        ``enabled=True`` with zero stiffness raises rather than silently
        falling back to rigid — the env code zeroes stiffness when the flag
        is off, so the two always agree at a reset.
        """
        if env_ids is None:
            env_ids = slice(None)
        ids = torch.arange(self._num_envs, device=self._device)[env_ids].reshape(-1)
        shape = (ids.numel(), len(self._kn_cols))

        def _as_pair(value, dtype=torch.float32):
            t = torch.as_tensor(value, dtype=dtype, device=self._device)
            # [envs] -> [envs, 1] so it broadcasts across the pair axis.
            if t.dim() == 1 and t.numel() == shape[0] and shape[1] != 1:
                t = t.unsqueeze(-1)
            try:
                return torch.broadcast_to(t, shape)
            except RuntimeError as err:
                raise ValueError(
                    f"series parameters must broadcast to [envs, pairs]={shape}; "
                    f"got shape {tuple(t.shape)}"
                ) from err

        k, d = _as_pair(stiffness), _as_pair(damping)
        inertia = self._series_rotor_inertia[ids] if rotor_inertia is None else _as_pair(rotor_inertia)
        en = _as_pair(enabled, torch.bool)
        if not all(torch.isfinite(t).all() for t in (k, d, inertia)):
            raise ValueError("series stiffness, damping, and inertia must be finite")
        if (k < 0.0).any() or (en & (k <= 0.0)).any() or (d < 0.0).any():
            raise ValueError("enabled series stiffness must be > 0; stiffness/damping must be >= 0")
        # An enabled pair must carry its own inertia: the env writes ZERO
        # PhysX armature on those joints (armature_write_mask), so there is
        # nothing left to fall back on — a zero here would make the belt
        # massless and the belt deflection instantaneous.
        if (inertia < 0.0).any() or (en & (inertia <= 0.0)).any():
            raise ValueError(
                "enabled series rotor inertia must be > 0; inertia must be >= 0. "
                "The elastic path zeroes the reflected PhysX armature, so it "
                "cannot borrow the joint armature as a fallback."
            )
        self._series_stiffness[ids] = k
        self._series_damping[ids] = d
        self._series_rotor_inertia[ids] = inertia
        self._series_enabled[ids] = en
        self._series_any_enabled = bool(self._series_enabled.any())
        self._series_state_valid[ids] = False
        self.series_motor_effort[ids] = 0.0
        self.series_spring_effort[ids] = 0.0

    def armature_write_mask(self) -> torch.Tensor:
        """Per-joint mask of ``self.armature`` columns that must be written to
        PhysX in the current mode.

        Active transmission pairs mask their knee column to zero and their
        coupled hip-pitch column to the hip's OWN-rotor share:

        * knee — the reflected rotor inertia (``I_r/G²``, the whole point of
          the knee armature value) now rides inside the series integrator, so
          leaving it in the PhysX joint would count the rotor twice;
        * hip_pitch — the knee rotor reaches the hip only through the belt
          anchor, so a compliant belt means the hip no longer carries
          ``c²·I_r/G²`` — but the hip motor's own rotor is direct drive and
          stays.  The column is scaled by ``own/total`` (``_hip_own_share``,
          derived from the cfg: 0.50 under the derived armature, 13.9/33.2 =
          0.42 under the P7 pins), never by a literal.  Before 2026-09-18 this
          column was zeroed, which dropped the whole hip armature.  The
          off-diagonal ``c·I_r/G²`` is supplied by BeltKneeDCMotor's
          rotor-coupling term, which is likewise disabled for these
          environments.

        Inactive pairs (the rigid path, and every environment when
        ``series_elasticity`` is False) return ones, i.e. the untouched
        configured armature.
        """
        mask = torch.ones_like(self.armature)
        if not self._series_any_enabled:
            return mask
        active = self._series_enabled & (self._series_stiffness > 0.0)
        # Index with the per-joint columns and a value of the SAME shape as
        # the selection, so this works whatever device/broadcast the DR
        # tensors landed on (indexing a tensor with a mismatched-device index
        # tensor does not raise a device error — it raises a confusing
        # IndexError).
        zeros = torch.zeros_like(active, dtype=mask.dtype)
        own_share = self._hip_own_share.to(dtype=mask.dtype)
        mask[:, self._kn_cols] = torch.where(active, zeros, mask[:, self._kn_cols])
        mask[:, self._hp_cols] = torch.where(active, own_share, mask[:, self._hp_cols])
        return mask

    def encoder_joint_state(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor):
        """Return joint-side knee encoder state implied by the virtual motor.

        This is the hardware-equivalent knee reading: the deploy stack gets
        ONLY this motor-side quantity (``control/config.py`` decodes the
        encoder through the 22:30 belt model), so an observation built from
        the physical joint would be untrainable-by-construction once the belt
        can stretch.  The env feeds these two tensors into the obs slots in
        place of ``robot.data.joint_pos/vel``.

        The belt rest length is initialized unloaded (``_series_state_valid``
        is cleared at every reset and re-seeded from live state), so at
        handover q_enc equals the physical knee and the opening frame is
        unchanged from the rigid contract.  Offset only accumulates once the
        belt is actually loaded.
        """
        c = self.cfg.coupling
        G = BELT_G
        hp = self._hp_cols
        # The pair columns index this GROUP's joints (what compute() receives),
        # not the articulation's full joint array.  A caller that passes the
        # full array gets the wrong hip subtracted with no shape error, so the
        # width is checked here rather than trusted (bitten 2026-09-15).
        n_group = len(self._joint_names)
        if joint_pos.shape[1] != n_group or joint_vel.shape[1] != n_group:
            raise ValueError(
                f"encoder_joint_state expects the actuator-local joint arrays "
                f"[envs, {n_group}] (slice with .joint_indices); got "
                f"{tuple(joint_pos.shape)} / {tuple(joint_vel.shape)}"
            )
        q_enc = G * self._series_motor_pos - c * joint_pos[:, hp]
        qd_enc = G * self._series_motor_vel - c * joint_vel[:, hp]
        active = self._series_enabled & (self._series_stiffness > 0.0)
        active = active & self._series_state_valid.unsqueeze(-1)
        # Gather the rigid fallback explicitly rather than indexing inside
        # torch.where: gathering with a [envs, pairs] index keeps the pair axis
        # second, whereas indexing inside torch.where broadcasts against
        # whichever axis happens to be smaller — which returned a [pairs, pairs]
        # tensor whenever the env count differed from the pair count.
        fallback_pos = torch.gather(joint_pos, 1, self._kn_cols.expand(joint_pos.shape[0], -1))
        fallback_vel = torch.gather(joint_vel, 1, self._kn_cols.expand(joint_vel.shape[0], -1))
        q_enc = torch.where(active, q_enc, fallback_pos)
        qd_enc = torch.where(active, qd_enc, fallback_vel)
        return q_enc, qd_enc

    def reset(self, env_ids: Sequence[int]):
        # Per-env resets land on control-tick boundaries, so the global
        # substep counter stays phase-aligned; only the latch is invalidated
        # (it re-latches from live state at the next compute).
        if env_ids is None or env_ids == slice(None):
            self._latch_valid[:] = False
            self._prev_vel_valid[:] = False
            self._series_state_valid[:] = False
            self._substep = 0
        else:
            self._latch_valid[env_ids] = False
            self._prev_vel_valid[env_ids] = False
            self._series_state_valid[env_ids] = False

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        c = self.cfg.coupling
        kn, hp = self._kn_cols, self._hp_cols

        # ── stale-hip latch (motor_bridge send-time read_state) ─────────
        hp_pos = joint_pos[:, hp]
        hp_vel = joint_vel[:, hp]
        if self.cfg.stale_hip_hold:
            if self._substep % self.cfg.hold_decimation == 0:
                self._hp_latch_pos[:] = hp_pos
                self._hp_latch_vel[:] = hp_vel
                self._latch_valid[:] = True
            elif not self._latch_valid.all():
                inv = ~self._latch_valid
                self._hp_latch_pos[inv] = hp_pos[inv]
                self._hp_latch_vel[inv] = hp_vel[inv]
                self._latch_valid[inv] = True
            self._substep += 1
            hp_pos_ref, hp_vel_ref = self._hp_latch_pos, self._hp_latch_vel
        else:
            hp_pos_ref, hp_vel_ref = hp_pos, hp_vel

        # ── stock PD for every column ───────────────────────────────────
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel
        effort = self.stiffness * error_pos + self.damping * error_vel + control_action.joint_efforts

        # ── knee columns: firmware identity in joint space ──────────────
        # tau_kn = kp*(tgt - q_kn) + kp*c*(q_hp_latch - q_hp)
        #        + kd*(v_ref - q_kn_dot - c*q_hp_dot) + tau_ff
        # where v_ref = c*q_hp_latch_dot when the deployed v_des FF is live,
        # else 0 (pre-patch firmware: Kd fights the full belt drag).
        kp_kn = self.stiffness[:, kn]
        kd_kn = self.damping[:, kn]
        v_ref = control_action.joint_velocities[:, kn]
        if self.cfg.knee_decouple_ff:
            v_ref = v_ref + c * hp_vel_ref
        effort_kn = (
            kp_kn * (control_action.joint_positions[:, kn] - joint_pos[:, kn])
            + kp_kn * c * (hp_pos_ref - hp_pos)
            + kd_kn * (v_ref - joint_vel[:, kn] - c * hp_vel)
            + control_action.joint_efforts[:, kn]
        )
        effort[:, kn] = effort_kn
        self.computed_effort = effort

        # ── DC envelope: knee columns saturate on belt-effective velocity ─
        self._joint_vel[:] = joint_vel
        self._joint_vel[:, kn] = joint_vel[:, kn] + c * hp_vel
        self.applied_effort = self._clip_effort(self.computed_effort)

        # ── optional motor-side series elasticity -----------------------
        # y = (q_knee + c q_hip) / G is the output coordinate in motor
        # radians.  The spring stores 1/2*k*(m-y)^2 and the damper dissipates
        # d*(m_dot-y_dot)^2.  A backward-Euler update handles both without an
        # explicit-Euler stability bound; it adds numerical damping (substeps
        # reduce it) and avoids an additional PhysX joint.
        #
        # The env zeroes this actuator group's reflected rotor armature on
        # elastic environments (armature_write_mask), so the rotor inertia
        # lives here and only here.  Parameters are motor-side throughout.
        series_active = self._series_enabled & (self._series_stiffness > 0.0)
        if series_active.any():
            G = BELT_G
            y = (joint_pos[:, kn] + c * hp_pos) / G
            ydot = (joint_vel[:, kn] + c * hp_vel) / G
            active = series_active
            init = series_active & ~self._series_state_valid.unsqueeze(-1)
            self._series_motor_pos = torch.where(init, y, self._series_motor_pos)
            self._series_motor_vel = torch.where(init, ydot, self._series_motor_vel)
            self._series_state_valid |= active.any(dim=-1)

            # Servo acts on the virtual motor encoder, not on the physical
            # knee.  This is equivalent to the rigid law when m=(qk+c*qh)/G
            # but lets the belt deflect under load.
            m_target = (control_action.joint_positions[:, kn] + c * hp_pos_ref) / G
            md_target = control_action.joint_velocities[:, kn] / G
            if self.cfg.knee_decouple_ff:
                md_target = md_target + c * hp_vel_ref / G
            kp_m = (G * G) * self.stiffness[:, kn]
            kd_m = (G * G) * self.damping[:, kn]
            tau_m_raw = kp_m * (m_target - self._series_motor_pos)
            tau_m_raw = tau_m_raw + kd_m * (md_target - self._series_motor_vel)
            tau_m_raw = tau_m_raw + G * control_action.joint_efforts[:, kn]
            motor_vel_joint = G * self._series_motor_vel
            self._joint_vel[:, kn] = torch.where(active, motor_vel_joint, self._joint_vel[:, kn])
            # Clip in the existing joint-equivalent units so the rigid limits
            # remain unchanged; the envelope sees actual motor speed above.
            clip_in = self.computed_effort.clone()
            clip_in[:, kn] = tau_m_raw / G
            tau_motor_joint = self._clip_effort(clip_in)[:, kn]
            tau_motor_cmd = G * tau_motor_joint
            inertia = self._series_rotor_inertia
            # Belt: set_series_parameters already rejects a non-positive
            # inertia on an ENABLED pair, so the [:, kn] armature fallback
            # below can only fire for a pair that is inactive (and is then
            # overwritten by the torch.where(active, ...) writes at the end).
            # Kept as a NaN guard, not as a physical fallback.
            inertia = inertia.clamp_min(1.0e-7)
            # Backward-Euler spring/damper update.  Solving the linear pair
            # implicitly is unconditionally stable for k,d >= 0 and removes
            # the explicit-Euler energy injection at contact transients.
            dt = float(self.cfg.series_dt) / max(1, int(self.cfg.series_substeps))
            new_pos, new_vel = self._series_motor_pos, self._series_motor_vel
            n_sub = max(1, int(self.cfg.series_substeps))
            # The transmission anchor rides with the measured output at
            # constant velocity within the tick (output acceleration over one
            # 2 ms control step is negligible).  The anchor velocity is
            # advanced alongside the position: the damper term is treated
            # implicitly in (m_dot - y_dot), which is what makes the update
            # unconditional-stable for d >= 0 rather than only for k.
            y_sub, ydot_sub = y, ydot
            for _ in range(n_sub):
                y_sub = y_sub + dt * ydot_sub
                k = self._series_stiffness
                d = self._series_damping
                # Semi-implicit ("backward-Euler spring") update: the spring
                # force is taken at the NEW position, the damper term is made
                # implicit so its coefficient lands in the denominator.
                #   v_new = [I*v_old + dt*(tau + k*(y - x) + d*y_dot)] / (I + dt*d)
                #   x_new = x + dt*v_new
                # The (I + dt*d) denominator keeps damping unconditionally
                # stable for d >= 0.  No dt^2*k term in the denominator: that
                # makes the step *fully* implicit, which is dissipative in
                # proportion to k*dt^2/I and so silently damps the STIFF end
                # of the sweep harder than the soft end (measured: 33 % of
                # amplitude per step at k=800, ~0 at k=2 — i.e. the numerical
                # error would have masqueraded as the plant parameter under
                # test).  Semi-implicit instead conserves the oscillator to
                # within a few percent at the same substep count (measured
                # +4.6 % peak at k=150 motor-side, and decreasing as damping
                # is added), with the same unconditional damping stability.
                rhs = inertia * new_vel + dt * (tau_motor_cmd + k * (y_sub - new_pos) + d * ydot_sub)
                den = inertia + dt * d
                new_vel = rhs / den.clamp_min(1.0e-9)
                new_pos = new_pos + dt * new_vel
            # Reaction from the same anchor the final substep solved against,
            # in motor units, then reduced to the joint frame.
            new_tau = self._series_stiffness * (new_pos - y_sub) + self._series_damping * (new_vel - ydot_sub)
            joint_tau = new_tau / G
            self._series_tau_motor_cmd = torch.where(active, tau_motor_cmd, self._series_tau_motor_cmd)
            self.series_motor_effort = torch.where(active, tau_motor_joint, self.series_motor_effort)
            self.series_spring_effort = torch.where(active, joint_tau, self.series_spring_effort)
            self.applied_effort[:, kn] = torch.where(active, joint_tau, self.applied_effort[:, kn])
            # Generalized hip reaction of the internal transmission.  The
            # existing hip_reaction flag may add the ideal motor reaction as
            # an explicit A/B option below.
            hip_rxn = c * (joint_tau - tau_motor_joint)
            hp_eff = self.applied_effort[:, hp]
            self.applied_effort[:, hp] = hp_eff + torch.where(active, hip_rxn, torch.zeros_like(hip_rxn))
            self._series_motor_pos = torch.where(active, new_pos, self._series_motor_pos)
            self._series_motor_vel = torch.where(active, new_vel, self._series_motor_vel)

        # ── optional ideal J^T belt reaction on the hip (default OFF) ───
        if self.cfg.hip_reaction:
            effort = self.applied_effort.clone()
            # For series-enabled envs the residual c*(spring - motor) was
            # already applied above; adding c*motor here recovers the ideal
            # c*spring reaction without double-counting it.  The motor-frame
            # command comes from the buffer, so there is no scope guessing.
            if series_active.any():
                tau_motor_joint = self._series_tau_motor_cmd / BELT_G
                ideal = c * self.applied_effort[:, kn]
                ideal = torch.where(series_active, c * tau_motor_joint, ideal)
                effort[:, hp] = effort[:, hp] + ideal
            else:
                effort[:, hp] = effort[:, hp] + c * self.applied_effort[:, kn]
            clipped_reaction = self._clip_effort(effort)
            # Spring output is a plant force: energy released by the belt is
            # not an additional motor command and must not be re-clipped by
            # the DC envelope.  Preserve it in the active transmission path.
            # (The hip reaction itself still passes through the envelope in
            # this arm; the reaction has no measured hardware magnitude.)
            if series_active.any():
                clipped_reaction[:, kn] = torch.where(series_active, effort[:, kn], clipped_reaction[:, kn])
                clipped_reaction[:, hp] = torch.where(series_active, effort[:, hp], clipped_reaction[:, hp])
            self.applied_effort = clipped_reaction

        # ── rotor inertial coupling (post-clip: plant force, not motor) ──
        # tau_hp -= m_x*qdd_kn, tau_kn -= m_x*qdd_hp with m_x = c*armature_kn
        # (knee armature IS the reflected rotor I_r/G^2, so the coupling
        # scales with the per-env armature DR draw automatically).
        # SAFETY (P9 post-explosion hardening): contact impacts produce FD
        # qdd spikes far beyond rigid-body scale; the original +-3000 rad/s^2
        # clamp allowed ~100 Nm injected torque (3x effort_limit_sim) — a
        # plausible trigger for the iter-564 PhysX blow-up. Now: qdd clamped
        # to +-1500, NaN-guarded, and the coupling TORQUE itself saturates at
        # rotor_coupling_tau_max (default 15 Nm — fall-log peaks were ~5-7.6).
        if self.cfg.rotor_coupling:
            # Elastic envs get this off-diagonal from the series state itself
            # (the rotor is inside the integrator), so the explicit term is
            # zeroed there to avoid double-counting — not because it stopped
            # being a real plant force.  The armature mask removes the
            # matching diagonal.
            series_env = series_active.any(dim=-1)
            qdd = (joint_vel - self._prev_vel) / self.cfg.rotor_coupling_dt
            qdd = torch.nan_to_num(qdd, nan=0.0, posinf=0.0, neginf=0.0)
            qdd = torch.clamp(qdd, -1500.0, 1500.0)
            qdd = qdd * self._prev_vel_valid.float().unsqueeze(-1)
            m_x = c * self.armature[:, kn]
            tmax = self.cfg.rotor_coupling_tau_max
            tau_hp_x = torch.clamp(-m_x * qdd[:, kn], -tmax, tmax)
            tau_kn_x = torch.clamp(-m_x * qdd[:, hp], -tmax, tmax)
            tau_hp_x = torch.where(series_env.unsqueeze(-1), torch.zeros_like(tau_hp_x), tau_hp_x)
            tau_kn_x = torch.where(series_env.unsqueeze(-1), torch.zeros_like(tau_kn_x), tau_kn_x)
            coupled = self.applied_effort.clone()
            coupled[:, hp] = coupled[:, hp] + tau_hp_x
            coupled[:, kn] = coupled[:, kn] + tau_kn_x
            self.applied_effort = coupled
        self._prev_vel[:] = joint_vel
        self._prev_vel_valid[:] = True

        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class BeltKneeDCMotorCfg(DCMotorCfg):
    """Configuration for :class:`BeltKneeDCMotor`."""

    class_type: type = BeltKneeDCMotor

    saturation_effort: float | dict[str, float] = None
    """Peak motor torque (N·m). Widened to accept per-joint dicts."""

    coupling: float = BELT_COUPLING
    """Belt coupling coefficient c (config.py KNEE_HIP_COUPLING); the module
    constant is the one source, the derivation reads the same one."""

    pairs: dict[str, str] = {"knee1": "hip_pitch1", "knee2": "hip_pitch2"}
    """knee joint name -> its coupled hip pitch joint name."""

    hip_own_rotor_armature: float = float("nan")
    """The hip motor's OWN output-shaft rotor inertia (direct drive, so joint
    side; kg m^2) — the part of the hip_pitch armature that stays in PhysX
    under a live series pair (armature_write_mask = own/total on the hip
    column).  NaN = the residual hip - c^2*knee of the nominal armature dict,
    which derive_armature() builds as exactly own + c^2*knee.  The factories
    and the env build path pass the derivation's own_rotor explicitly."""

    knee_decouple_ff: bool = True
    """Replicate the deployed KNEE_DECOUPLE_FF v_des (drag cancelled with the
    tick-start hip velocity). False = pre-patch firmware (full belt drag)."""

    hip_reaction: bool = False
    """Ideal J^T belt reaction +c*tau_knee on the hip. Empirically absent on
    hardware (2026-08-06 replay) — keep OFF unless re-tested."""

    stale_hip_hold: bool = True
    """Latch the hip pos/vel at control-tick boundaries (firmware holds the
    knee motor target computed from the tick-start hip)."""

    hold_decimation: int = 5
    """compute() calls per control tick (env decimation)."""

    rotor_coupling: bool = True
    """Apply the knee rotor's off-diagonal inertial coupling
    (tau_hp -= c*armature_kn*qdd_kn and vice versa) as a post-clip plant
    force. See module docstring — on hardware this is a launch-window hip
    torque the size of the hip's own commands. False = pre-P7 behavior."""

    rotor_coupling_dt: float = 1.0 / 500.0
    """Physics-step dt used by the finite-difference acceleration estimate.
    Must equal the sim physics dt (compute() runs once per physics step)."""

    rotor_coupling_tau_max: float = 15.0
    """Saturation on the injected coupling torque (N·m). Physical peaks on
    the fall logs are ~5-7.6 Nm; contact-impact FD spikes must not inject
    beyond this (the unclamped term reached ~100 Nm and is the prime suspect
    for the P9 iter-564 sim explosion)."""

    # Optional series-elastic belt transmission.  Disabled by default so the
    # legacy rigid actuator path is bit-for-bit unchanged.  Stiffness and
    # damping are motor-side (before the 22:30 reduction); per-episode values
    # can be installed with BeltKneeDCMotor.set_series_parameters().
    series_elasticity: bool = False
    series_stiffness: float = 0.0
    series_damping: float = 0.0
    series_rotor_inertia: float = 0.0

    # Timestep density of the belt integrator (2026-09-16).  Internal step is
    # series_dt/series_substeps = 0.125 ms, i.e. 4x FINER than the 0.5 ms
    # physics step (sim.dt = 1/2000, decimation = 20).
    #
    # This is a plant-model correctness parameter, not a tuning knob, and it is
    # the difference between a usable and an unusable belt DR band.  The values
    # here used to be 1/500 x2 — a 1 ms internal step — which was correct while
    # the env ran 500 Hz physics, but the 09-16 move to 2000 Hz changed
    # `decimation`, `sim.dt` and `rotor_coupling_dt` and left THIS behind,
    # because nothing in rl_env_p6.py or rl_env_cfg_p6.py writes series_dt.
    # The belt then advanced 1 ms of virtual time per 0.5 ms physics step,
    # relaxing against an anchor that had moved half as far as the integrator
    # assumed.  Two measured consequences (data/_s7_belt_config_score.py, the
    # one-step map of the production update):
    #
    #   * 2x virtual stiffness inflation plus a damper phase error;
    #   * spectral radius > 1 — energy injected every step — for every stiff
    #     cell: 0.964 at k=150, 2.225 at k=550, 3.583 at k=950 joint N m/rad.
    #     A stiff belt therefore behaved WORSE than a soft one, which is what
    #     the 09-16 stiffness ladder saw ("950 = totally loose control").
    #     At the setting below every cell reads 0.990-0.993 across 100-3000.
    #
    # KEEP IN SYNC with the env's physics dt if it ever moves again.  A cheap
    # guard: the [PLANT] belt line does not print series_dt, so if a run's belt
    # feels wrong the first thing to read is this comment, not the recipe.
    #
    # COST: 8 tiny [envs, 2] tensor ops per physics step instead of 4 — well
    # under 1% of step time even at 4096 envs.  There is no accuracy trade:
    # series_dt moving with the physics rate is what makes 0.3 N m s/rad mean
    # the same thing at both rates.
    series_dt: float = 1.0 / 2000.0 #
    series_substeps: int = 4 #
