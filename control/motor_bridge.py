"""
motor_bridge.py — sim ↔ real joint coordinate translator + driver wrapper.

Handles, in this order:
  1. Joint reordering (sim idx → real motor ID).
  2. Per-joint sign flip + zero offset.
  3. Knee gear ratio (22:30): joint = motor * (22/30).
  4. Hip-knee belt coupling: joint = motor*ratio - hip_pitch * coupling.
  5. PD-gain transformation: motor-space Kp = sim-space Kp * gear² (knees).

All sim-facing methods speak in the simulation's joint convention.
The driver's MIT-mode protocol is hidden inside.

PROFILER HOOKS (added for profile_balance.py — backwards compatible):
    - send_target() accepts return_timing=False. When True, it returns
      the driver's timing dict augmented with 'transform_ms' (time spent
      in the sim→motor transform itself, separate from CAN wall time).
    - The most recent motor-space target is cached on
      self._last_motor_target so the profiler can log what was actually
      sent over the wire vs what the policy/PD asked for in sim space.
"""

import time
import numpy as np

from robstride_socketcan_driver import make_driver
import config as C

TWO_PI = 2.0 * np.pi

# Indices in the 6-vector
HR1, HP1, KN1, HR2, HP2, KN2 = 0, 1, 2, 3, 4, 5
KNEE_INDICES = (KN1, KN2)
HIP_PITCH_OF_KNEE = {KN1: HP1, KN2: HP2}


class MotorBridge:
    """6-DOF biped motor interface in sim joint coordinates."""

    def __init__(self, driver=None, action_lpf_alpha: float = 0.6):
        if driver is not None:
            self.driver = driver
        else:
            # Transport (CAN2USB serial vs CAN2SPI SocketCAN) is selected
            # by config.CAN_TRANSPORT — see robstride_socketcan_driver.
            self.driver = make_driver(C.MOTOR_IDS)

        self._sign     = np.array(C.JOINT_SIGN,   dtype=np.float32)
        self._offset   = np.array(C.JOINT_OFFSET, dtype=np.float32)
        self._gear     = C.KNEE_GEAR_RATIO
        self._coupling = C.KNEE_HIP_COUPLING

        # Latest cached state (sim coordinates)
        self.sim_joint_pos = np.zeros(6, dtype=np.float32)
        self.sim_joint_vel = np.zeros(6, dtype=np.float32)

        # Reusable buffers
        self._motor_pos = np.zeros(6, dtype=np.float32)
        self._motor_vel = np.zeros(6, dtype=np.float32)

        # Last motor-space command actually sent (for the profiler).
        self._last_motor_target = np.zeros(6, dtype=np.float32)

        # ── Action LPF state ─────────────────────────────────────
        # First-order LPF on the SIM-space position target:
        #   y[n] = α y[n-1] + (1-α) x[n]
        # α=0    → passthrough (no filter)
        # α=0.7  → cutoff ≈ 5.7 Hz at 100 Hz
        # α=0.85 → cutoff ≈ 2.6 Hz at 100 Hz (very smooth, sluggish)
        # Filter is initialized from the measured sim pose on the first
        # send_target() after enable() / reset_filter() to avoid a startup
        # transient (would otherwise jump from zero to the first command).
        self._lpf_alpha = float(action_lpf_alpha)
        self._filtered_target = np.zeros(6, dtype=np.float32)
        self._filter_initialized = False
        self.sim_joint_torque      = np.zeros(6, dtype=np.float32)
        self.sim_joint_temperature = np.zeros(6, dtype=np.float32)
        self.motor_torque         = np.zeros(6, dtype=np.float32)
        self.motor_temperature    = np.zeros(6, dtype=np.float32)

        # CAN drop tracking (review #7): count of reply frames parsed
        # on the most recent send_target/send_idle call.
        self.last_replies_parsed = 0


    def set_lpf_alpha(self, alpha: float):
        """Set LPF alpha. 0 = passthrough, 1 = frozen.
        Does NOT reset filter state — alpha changes take effect on next sample."""
        if not (0.0 <= alpha < 1.0):
            raise ValueError(f"alpha must be in [0, 1), got {alpha}")
        self._lpf_alpha = float(alpha)

    def reset_filter(self):
        """Force the LPF to re-initialize from measured pose on next send_target.
        Call after long idle periods or any time you want to drop transients."""
        self._filter_initialized = False
    # ── lifecycle ────────────────────────────────────────────
    def enable(self):
        self.driver.enable_all()
        self._filter_initialized = False   # re-init from measured pose

    def disable(self):
        self.driver.disable_all()

    def close(self):
        self.driver.close()

    def set_zero_at_current_pose(self):
        self.driver.set_zero_all()

    # ── coordinate transforms ────────────────────────────────
    def _motor_to_sim(self, motor_pos, motor_vel):
        sim_pos = motor_pos * self._sign + self._offset
        sim_vel = motor_vel * self._sign

        for kn in KNEE_INDICES:
            hp = HIP_PITCH_OF_KNEE[kn]
            sim_pos[kn] = (motor_pos[kn] * self._sign[kn] * self._gear
                        - sim_pos[hp] * self._coupling
                        + self._offset[kn])
            sim_vel[kn] = (motor_vel[kn] * self._sign[kn] * self._gear
                        - sim_vel[hp] * self._coupling)
        return sim_pos, sim_vel

    def _sim_to_motor(self, sim_target, hip_pitch_actual):
        motor_target = (sim_target - self._offset) * self._sign

        for kn in KNEE_INDICES:
            hp = HIP_PITCH_OF_KNEE[kn]
            sim_kn_relative = (sim_target[kn] - self._offset[kn]
                               + hip_pitch_actual[hp] * self._coupling)
            motor_target[kn] = sim_kn_relative / self._gear * self._sign[kn]
        return motor_target

    def _sim_to_motor_gains(self, sim_kp, sim_kd):
        kp_m = sim_kp.copy()
        kd_m = sim_kd.copy()
        for kn in KNEE_INDICES:
            kp_m[kn] = sim_kp[kn] * (self._gear ** 2)
            kd_m[kn] = sim_kd[kn] * (self._gear ** 2)
        return kp_m, kd_m
    
    def _motor_to_sim_torque(self, motor_torque):
        sim_torque = motor_torque * self._sign
        for kn in KNEE_INDICES:
            sim_torque[kn] = motor_torque[kn] * self._sign[kn] / self._gear
        return sim_torque

    # ── public API ───────────────────────────────────────────
    def read_state(self):
        for i, mid in enumerate(C.MOTOR_IDS):
            p, v, tau, temp = self.driver.get_state(mid)   # was: p, v, _
            self._motor_pos[i]         = p
            self._motor_vel[i]         = v
            self.motor_torque[i]      = tau
            self.motor_temperature[i] = temp
        self.sim_joint_pos, self.sim_joint_vel = self._motor_to_sim(
            self._motor_pos, self._motor_vel
        )
        self.sim_joint_torque      = self._motor_to_sim_torque(self.motor_torque)
        self.sim_joint_temperature = self.motor_temperature.copy()  # no transform
        return self.sim_joint_pos, self.sim_joint_vel

    def send_target(self, sim_target, sim_kp, sim_kd,
                    effort_limit=None,
                    sequential=False, return_timing=False):
        """Send a position target in sim joint space.

        sim_target/sim_kp/sim_kd are 6-vectors (np.float32).
        effort_limit: optional 6-vector of max joint-space torque (Nm).
            Defaults to C.REAL_EFFORT.  Clamps position error so
            kp * |error| <= limit, preventing the motor from exceeding
            the policy's trained effort range.
        LPF applied to sim_target before the motor coordinate transform.
        Knee decoupling uses the most recently read hip pitch.
        sequential=True sends one motor at a time (slower but reliable).
        return_timing=True returns the driver's timing dict augmented
            with 'transform_ms' (time spent in LPF + _sim_to_motor + gain map).
        """
        t0 = time.monotonic() if return_timing else 0.0

        # ── Action LPF on sim-space target ─────────────────────────
        sim_target = np.asarray(sim_target, dtype=np.float32)
        if self._lpf_alpha > 0.0:
            if not self._filter_initialized:
                # Avoid startup step from 0 → first command. self.sim_joint_pos
                # was populated by the most recent read_state(); if read_state
                # hasn't been called yet, it's zeros and we get one ramp-up
                # cycle, which is harmless.
                self._filtered_target[:] = self.sim_joint_pos
                self._filter_initialized = True
            a = self._lpf_alpha
            self._filtered_target = a * self._filtered_target + (1.0 - a) * sim_target
            effective_target = self._filtered_target
        else:
            effective_target = sim_target

        # ── Effort soft-clip (review #5) ───────────────────────────
        # Clamp position error so kp * |error| stays within the effort
        # limit.  This is the only torque limiting in the stack — motor
        # firmware current limits are the last backstop.
        if effort_limit is None:
            effort_limit = np.array(C.REAL_EFFORT, dtype=np.float32)
        else:
            effort_limit = np.asarray(effort_limit, dtype=np.float32)
        sim_kp_arr = np.asarray(sim_kp, dtype=np.float32)
        for i in range(6):
            if sim_kp_arr[i] > 0.0:
                max_err = effort_limit[i] / sim_kp_arr[i]
                err = effective_target[i] - self.sim_joint_pos[i]
                if abs(err) > max_err:
                    effective_target[i] = self.sim_joint_pos[i] + np.sign(err) * max_err

        # ── Existing transform path (unchanged below, just uses effective_target) ──
        motor_target = self._sim_to_motor(effective_target, self.sim_joint_pos)
        kp_m, kd_m = self._sim_to_motor_gains(sim_kp, sim_kd)
        self._last_motor_target = motor_target.copy()

        # ── Belt-coupling decoupling feedforwards (see config.py) ──────
        # Makes the real drivetrain torque behave like sim's rigid
        # direct-drive knee, so the trained policy meets the plant it
        # expects. Motor-frame conventions per _motor_to_sim/_sim_to_motor.
        v_des = np.zeros(6, dtype=np.float32)
        t_ff  = np.zeros(6, dtype=np.float32)
        if getattr(C, 'KNEE_DECOUPLE_FF', False):
            kd_arr = np.asarray(sim_kd, dtype=np.float32)
            for kn in KNEE_INDICES:
                hp = HIP_PITCH_OF_KNEE[kn]
                qd_hp = self.sim_joint_vel[hp]
                # Knee joint still + hip moving ⇒ belt forces the knee
                # motor to spin at c*qd_hp/G. Track it instead of letting
                # kd_m fight it (which cancelled ~30% of knee torque RMS).
                v_des[kn] = self._coupling * qd_hp * self._sign[kn] / self._gear
                # Ideal-transmission belt reaction on the hip (J^T torque
                # mapping). OFF by default — the 20260806 teacher-forcing
                # replay contradicted this term; see config.HIP_REACTION_FF.
                if getattr(C, 'HIP_REACTION_FF', False):
                    tau_kn_joint = (sim_kp[kn]
                                    * (effective_target[kn] - self.sim_joint_pos[kn])
                                    - kd_arr[kn] * self.sim_joint_vel[kn])
                    # Safety: never exceed the hip's own effort budget
                    # (hip is direct drive: motor Nm == joint Nm).
                    t_ff[hp] = float(np.clip(
                        -self._coupling * tau_kn_joint * self._sign[hp],
                        -effort_limit[hp], effort_limit[hp]))
        t1 = time.monotonic() if return_timing else 0.0

        if sequential:
            t_send_start = time.monotonic()
            replies_ok = 0
            for i, mid in enumerate(C.MOTOR_IDS):
                before = self.driver.motor_states[mid]['pos']
                self.driver.send_command(
                    mid,
                    p_des=float(motor_target[i]), v_des=float(v_des[i]),
                    kp=float(kp_m[i]), kd=float(kd_m[i]), t_ff=float(t_ff[i]),
                )
                replies_ok += 1   # send_command doesn't expose success; assume ok
            t_send_end = time.monotonic()
            self.last_replies_parsed = replies_ok

            if return_timing:
                return {
                    'encode_ms':       0.0,
                    'write_ms':        0.0,
                    'wait_ms':         0.0,
                    'parse_ms':        0.0,
                    'total_ms':        (t_send_end - t_send_start) * 1000.0,
                    'transform_ms':    (t1 - t0) * 1000.0,
                    'bridge_total_ms': (t_send_end - t0) * 1000.0,
                    'bytes_received':  6 * 17,        # placeholder — assume full
                    'expected_bytes':  6 * 17,
                    'replies_parsed':  replies_ok,    # placeholder — assume full
                }
            return None

        commands = [
            (C.MOTOR_IDS[i],
             float(motor_target[i]), float(v_des[i]),
             float(kp_m[i]), float(kd_m[i]), float(t_ff[i]))
            for i in range(len(C.MOTOR_IDS))
        ]
        # Always request timing internally so we can track replies_parsed
        timing = self.driver.send_batch(commands, return_timing=True)
        if timing is not None:
            self.last_replies_parsed = timing.get('replies_parsed', 0)
        if return_timing and timing is not None:
            timing['transform_ms']     = (t1 - t0) * 1000.0
            timing['bridge_total_ms']  = (time.monotonic() - t0) * 1000.0
            return timing
        return timing if return_timing else None

    def send_idle(self):
        """Zero command — motors limp. Also triggers encoder replies."""
        commands = [(mid, 0, 0, 0, 0, 0) for mid in C.MOTOR_IDS]
        self.driver.send_batch(commands)