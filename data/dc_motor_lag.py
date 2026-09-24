"""
dc_motor_lag.py — DCMotor with a first-order torque (current-loop) lag.

WHY A TORQUE LAG AND NOT A TARGET LAG
    A low-pass on the position *target* smooths the command, so the joint
    follows a monotonic ramp and can NEVER overshoot. Real hardware overshoots
    a step and shows a resonant peak in a chirp — that is an UNDER-damped loop,
    not a smoothed command. Lagging the *torque* instead injects phase lag
    INSIDE the control loop: it reproduces the bulk rise-time AND the overshoot,
    the chirp resonance, and the steeper high-frequency roll-off. It also
    correctly slows disturbance rejection (matters for the held / float cases
    and for a balance policy), which a target low-pass does not.

MODEL
    tau  : actuator time constant [s]  (closed-loop / current-loop bandwidth)
    The lag runs at the physics dt because compute() is called every substep.
    Pass lag_alpha = exp(-dt_phys / tau).  At dt_phys=0.002, tau=0.07 -> 0.9718.

USE
    Same class works in sysid_replay_sim.py and the training env, so the policy
    trains against the actuator it will actually meet on hardware. Swap the
    actuator cfgs from DCMotorCfg to DCMotorWithLagCfg (see the helper at the
    bottom) and set lag_alpha.
"""

from __future__ import annotations

import torch
from isaaclab.actuators import DCMotor, DCMotorCfg
from isaaclab.utils import configclass


class DCMotorWithLag(DCMotor):
    """DCMotor whose output torque is passed through a first-order low-pass."""

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._lag_alpha = float(getattr(cfg, "lag_alpha", 0.0))
        self._prev_effort = None  # lazily init from first computed torque

    def reset(self, env_ids=None):
        super().reset(env_ids)
        # Re-init the lag state. Single-env replay does a full reset per trial,
        # so clearing entirely is correct. For partial training resets we zero
        # only the reset rows (one-substep ramp from 0; negligible).
        if self._prev_effort is not None and env_ids is not None:
            self._prev_effort[env_ids] = 0.0
        else:
            self._prev_effort = None

    def compute(self, *args, **kwargs):
        # Run the normal DCMotor PD + torque-speed clip first, then lag the
        # resulting torque. Using control_action.joint_efforts keeps this
        # robust across IsaacLab versions.
        control_action = super().compute(*args, **kwargs)
        if self._lag_alpha > 0.0 and control_action.joint_efforts is not None:
            if self._prev_effort is None:
                self._prev_effort = control_action.joint_efforts.clone()
            a = self._lag_alpha
            eff = a * self._prev_effort + (1.0 - a) * control_action.joint_efforts
            self._prev_effort = eff
            control_action.joint_efforts = eff
            self.applied_effort = eff  # keep diagnostics/logging consistent
        return control_action


@configclass
class DCMotorWithLagCfg(DCMotorCfg):
    class_type: type = DCMotorWithLag
    lag_alpha: float = 0.9718  # exp(-dt_phys / tau); 0 disables the lag


def wrap_actuators_with_lag(robot_cfg, tau_s: float, dt_phys: float):
    """Replace every DCMotorCfg actuator on robot_cfg with a lagged version.

    Call BEFORE the Articulation is constructed. tau_s<=0 is a no-op.
    Copies all existing fields so effort/velocity limits, gains, armature and
    friction are preserved untouched.
    """
    if tau_s is None or tau_s <= 0.0:
        return robot_cfg
    import math
    alpha = math.exp(-dt_phys / tau_s)
    for name, acfg in list(robot_cfg.actuators.items()):
        lag = DCMotorWithLagCfg()
        for f in acfg.__dataclass_fields__:
            setattr(lag, f, getattr(acfg, f))
        lag.lag_alpha = alpha
        robot_cfg.actuators[name] = lag
    print(f"  [lag] torque lag tau={tau_s:.3f}s -> alpha={alpha:.4f} "
          f"(dt_phys={dt_phys})  applied to: {list(robot_cfg.actuators)}")
    return robot_cfg
