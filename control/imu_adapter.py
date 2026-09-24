"""
imu_adapter.py — IMU → sim-style observation pieces.

The simulation's observation includes:
  - projected_gravity (3) — gravity expressed in IMU body frame
  - ang_vel_imu       (3) — body-frame angular velocity [rad/s]
  - lin_vel_imu       (3) — body-frame linear velocity [m/s]   ← problematic
  - root yaw          — used for the yaw-error sin/cos pair

The Yesense IMU gives euler (deg) and gyro (deg/s). We convert via quaternion
so the math matches isaaclab.utils.math.quat_rotate_inverse exactly.

About lin_vel_imu: the IMU does NOT give absolute linear velocity. The sim
gave the policy a perfect velocity oracle. On hardware, options are:
    (a) Set to zero. Simple. Policy may bias actions but usually survives.
    (b) Integrate accel - gravity (drifts within seconds).
    (c) Leg odometry from foot kinematics + contact (best, but complex).
We use (a) as default. See get_lin_vel_body() to swap in (b)/(c).
"""

import math
import numpy as np

import config as C
from imu_driver_6_400hz import IMUDriver6Axis


DEG2RAD = math.pi / 180.0


def _euler_zyx_to_quat(roll, pitch, yaw):
    """ZYX intrinsic rotation (yaw → pitch → roll) → quaternion (w, x, y, z)."""
    cy, sy = math.cos(yaw * 0.5),   math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5),  math.sin(roll * 0.5)

    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    return np.array([w, x, y, z], dtype=np.float32)


def _quat_rotate_inverse(q, v):
    """Rotate vector v by the inverse of unit quaternion q (w,x,y,z).

    Mirrors isaaclab.utils.math.quat_rotate_inverse.
    """
    w, x, y, z = q
    # q_conj * v * q (treating v as pure quat)
    # Standard formula:
    vx, vy, vz = v
    t2 = 2.0 * (y * vz - z * vy)
    t3 = 2.0 * (z * vx - x * vz)
    t4 = 2.0 * (x * vy - y * vx)
    out_x = vx - w * t2 + (y * t4 - z * t3)
    out_y = vy - w * t3 + (z * t2 - x * t4)
    out_z = vz - w * t4 + (x * t3 - y * t2)
    return np.array([out_x, out_y, out_z], dtype=np.float32)


_GRAVITY_WORLD = np.array([0.0, 0.0, -1.0], dtype=np.float32)

# Attitude trim, applied to the x/y components of projected gravity (see
# config.py IMU_PG_OFFSET for the measurement + trim procedure). z is left
# untouched: for the small angles involved the correction is ~quadratic in
# z and the policy reads tilt from x/y.
_PG_OFFSET = np.array(
    [C.IMU_PG_OFFSET[0], C.IMU_PG_OFFSET[1], 0.0], dtype=np.float32
) if hasattr(C, "IMU_PG_OFFSET") else np.zeros(3, dtype=np.float32)


class IMUAdapter:
    """Wraps IMUDriver6Axis and exposes sim-space observation pieces."""

    def __init__(self):
        self.driver = IMUDriver6Axis(C.IMU_PORT, C.IMU_BAUD)
        self._last_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._last_ang_vel = np.zeros(3, dtype=np.float32)
        self._last_yaw = 0.0
        self.pg_offset = _PG_OFFSET.copy()

    def set_pg_offset(self, x, y=None):
        """Override the config.IMU_PG_OFFSET trim for this process (per-trial
        trim bracketing without editing config.py). y=None keeps config's y."""
        self.pg_offset = np.array(
            [x, C.IMU_PG_OFFSET[1] if y is None else y, 0.0], dtype=np.float32
        )

    def open(self):
        if not self.driver.open():
            raise RuntimeError("IMU failed to open")

    def close(self):
        self.driver.close()

    @property
    def fps(self):
        return self.driver.fps

    # ── core read ────────────────────────────────────────────
    def update(self):
        """Pull latest IMU data and refresh cached sim-space pieces."""
        d = self.driver.get_data()
        # Yesense order: (pitch, roll, yaw) in deg
        pitch_deg, roll_deg, yaw_deg = d['euler']
        gx, gy, gz = d['gyro']  # deg/s

        # Apply user-tuned axis remap
        roll  = (C.IMU_ROLL_SIGN  * roll_deg ) * DEG2RAD
        pitch = (C.IMU_PITCH_SIGN * pitch_deg) * DEG2RAD
        yaw   = (C.IMU_YAW_SIGN   * yaw_deg  ) * DEG2RAD

        ang_vel_x = (C.IMU_GYRO_X_SIGN * gx) * DEG2RAD
        ang_vel_y = (C.IMU_GYRO_Y_SIGN * gy) * DEG2RAD
        ang_vel_z = (C.IMU_GYRO_Z_SIGN * gz) * DEG2RAD

        self._last_quat = _euler_zyx_to_quat(roll, pitch, yaw)
        self._last_ang_vel = np.array(
            [ang_vel_x, ang_vel_y, ang_vel_z], dtype=np.float32
        )
        self._last_yaw = yaw

    # ── derived quantities (sim space) ───────────────────────
    def projected_gravity(self):
        """Gravity in IMU body frame, matching sim convention.

        Trimmed by self.pg_offset (config.IMU_PG_OFFSET unless overridden
        via set_pg_offset).
        """
        pg = _quat_rotate_inverse(self._last_quat, _GRAVITY_WORLD)
        return pg - self.pg_offset

    def projected_gravity_raw(self):
        """Untrimmed projected gravity — for logging/calibration only."""
        return _quat_rotate_inverse(self._last_quat, _GRAVITY_WORLD)

    def ang_vel_body(self):
        """Body-frame angular velocity in rad/s (already in body frame)."""
        # Gyro readings are already body-frame, no rotation needed.
        return self._last_ang_vel

    def lin_vel_body(self):
        """Body-frame linear velocity in m/s.

        Default: zero (best honest guess without leg odometry).
        Replace with a state estimator if you build one.
        """
        return np.zeros(3, dtype=np.float32)

    def yaw(self):
        """Current yaw angle [rad] in world frame (drifts since VRU mode)."""
        return self._last_yaw

    def quat(self):
        return self._last_quat

    def upright_cos(self):
        """cos(tilt). 1.0 = perfectly upright, 0 = horizontal."""
        return float(-self.projected_gravity()[2])