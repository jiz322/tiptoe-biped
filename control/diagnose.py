#!/usr/bin/env python3
"""
diagnose.py — measure sim2real transfer gaps to drive DR decisions.

This is a READ-ONLY probe in static and timing modes; only stepresp
actually drives a motor (gently, one joint, ~5 deg, low Kp).

Setup
-----
Hang the robot from its torso, exactly as for balance.py. All modes
assume the robot is motionless and free to swing. Do NOT run on the
floor — stepresp will twitch one leg.

Modes
-----
  static    Robot still, 30s. Logs sensor noise floors and compares
            motor-reported joint_vel to a finite-difference estimate.
            Outputs recommended per-step DR magnitudes for IMU and
            joint_vel observation noise.

  timing    Robot still, 10s. Times each phase of the control loop
            (IMU read, motor read, motor send) and reports the
            synchronous round-trip lower bound. Outputs a recommended
            action_delay range in 100 Hz control steps.

  stepresp  Hang robot, ~10s. Commands a small step on hip_pitch1 at
            low Kp and measures time-to-first-motion. This catches
            latency that timing/ misses (motor controller + bus pipeline
            + sample staleness). The total of timing + stepresp is the
            real action_delay you should model.

  all       Runs all three in sequence.

Usage
-----
  sudo python3 diagnose.py static
  sudo python3 diagnose.py timing
  sudo python3 diagnose.py stepresp [joint_idx=1]
  sudo python3 diagnose.py all
"""

import sys
import time
import numpy as np

from motor_bridge import MotorBridge
from imu_adapter import IMUAdapter

CONTROL_HZ = 100
PERIOD = 1.0 / CONTROL_HZ

SIM_NAMES = ['L_roll', 'L_hip', 'L_knee', 'R_roll', 'R_hip', 'R_knee']

# Pulled from the matching sim cfg slots so the printed comparison is
# meaningful. If you change cfg, update these.
SIM_NOISE_PROJ_GRAV_PER_STEP = 0.0      # sim has no per-step IMU noise
SIM_NOISE_ANG_VEL_PER_STEP   = 0.0
SIM_NOISE_JOINT_VEL_PER_STEP = 0.0


# ────────────────────────────────────────────────────────────
# Mode 1: static noise floor
# ────────────────────────────────────────────────────────────

def static_test(bridge, imu, duration=30.0):
    print(f"\n{'=' * 64}")
    print(f"  STATIC NOISE TEST  ({duration:.0f}s)")
    print(f"{'=' * 64}")
    print("  Robot must be hanging motionless. Recording...\n")

    n = int(duration * CONTROL_HZ)

    proj_grav = np.zeros((n, 3), dtype=np.float32)
    ang_vel   = np.zeros((n, 3), dtype=np.float32)
    joint_pos = np.zeros((n, 6), dtype=np.float32)
    joint_vel = np.zeros((n, 6), dtype=np.float32)
    timestamps = np.zeros(n, dtype=np.float64)

    bridge.send_idle()
    time.sleep(0.1)

    t0 = time.monotonic()
    for i in range(n):
        t_loop = time.monotonic()
        imu.update()
        bridge.read_state()

        proj_grav[i]  = imu.projected_gravity()
        ang_vel[i]    = imu.ang_vel_body()
        joint_pos[i]  = bridge.sim_joint_pos
        joint_vel[i]  = bridge.sim_joint_vel
        timestamps[i] = t_loop - t0

        if i % CONTROL_HZ == 0 and i > 0:
            print(f"  [{i // CONTROL_HZ:2d}/{int(duration):2d}s]", end='\r')

        elapsed = time.monotonic() - t_loop
        if elapsed < PERIOD:
            time.sleep(PERIOD - elapsed)

    # Stats
    pg_std,  pg_mean  = proj_grav.std(axis=0), proj_grav.mean(axis=0)
    av_std,  av_mean  = ang_vel.std(axis=0),   ang_vel.mean(axis=0)
    jp_std            = joint_pos.std(axis=0)
    jv_std,  jv_mean  = joint_vel.std(axis=0), joint_vel.mean(axis=0)

    dt = np.diff(timestamps).mean()
    jv_fd = np.diff(joint_pos, axis=0) / dt
    jv_fd_std = jv_fd.std(axis=0)

    print("\n\n  RESULTS\n  " + "-" * 60)

    print("\n  IMU:")
    print(f"    {'channel':<22s}  {'mean':>10s}  {'std':>10s}")
    for i, axis in enumerate(['proj_grav.x', 'proj_grav.y', 'proj_grav.z']):
        print(f"    {axis:<22s}  {pg_mean[i]:>+10.5f}  {pg_std[i]:>10.5f}")
    for i, axis in enumerate(['ang_vel.x (rad/s)', 'ang_vel.y (rad/s)', 'ang_vel.z (rad/s)']):
        print(f"    {axis:<22s}  {av_mean[i]:>+10.5f}  {av_std[i]:>10.5f}")

    print("\n  Joints (sim space):")
    print(f"    {'joint':<8s}  {'pos_std':>10s}  {'vel_std (motor)':>17s}  "
          f"{'vel_std (FD)':>14s}  {'FD/motor':>10s}")
    for i in range(6):
        ratio = jv_fd_std[i] / max(jv_std[i], 1e-6)
        print(f"    {SIM_NAMES[i]:<8s}  {jp_std[i]:>10.6f}  "
              f"{jv_std[i]:>17.5f}  {jv_fd_std[i]:>14.5f}  {ratio:>10.2f}x")

    print("\n  CONTROL LOOP DT (sanity check):")
    dt_arr = np.diff(timestamps)
    print(f"    target: {PERIOD * 1000:.2f}ms   "
          f"actual mean: {dt_arr.mean() * 1000:.2f}ms   "
          f"std: {dt_arr.std() * 1000:.2f}ms   "
          f"max: {dt_arr.max() * 1000:.2f}ms")

    # Recommendations
    print("\n  RECOMMENDED DR (per-step uniform, ± the value):")
    pg_rec = max(pg_std[:2].max() * 2, 0.005)
    av_rec = max(av_std.max()       * 2, 0.02)
    jv_rec = max(max(jv_std.max(), jv_fd_std.max()) * 2, 0.10)
    print(f"    projected_gravity (xy):  ± {pg_rec:.4f}     "
          f"(measured std max {pg_std[:2].max():.5f})")
    print(f"    ang_vel:                 ± {av_rec:.4f} rad/s "
          f"(measured std max {av_std.max():.5f})")
    print(f"    joint_vel:               ± {jv_rec:.4f} rad/s "
          f"(motor max {jv_std.max():.5f}, FD max {jv_fd_std.max():.5f})")

    print("\n  INTERPRETATION:")
    fd_motor_ratio = jv_fd_std.max() / max(jv_std.max(), 1e-6)
    if fd_motor_ratio > 5.0:
        print("    ! Motor's velocity filter is heavy (FD is >>5x motor-reported).")
        print("      Sim sees ground-truth velocity; real sees a smoothed signal.")
        print("      Two ways to close this gap:")
        print("      (a) Add joint_vel DR sized to motor_std (small) — models")
        print("          residual noise after filtering. Phase lag remains.")
        print("      (b) Filter joint_vel obs in sim to match the motor's filter.")
        print("          Better but harder; requires knowing the filter spec.")
        print("    Default to (a) for now; revisit if behavior still doesn't transfer.")
    elif fd_motor_ratio > 2.0:
        print("    Motor reports a moderately filtered velocity. Use the larger")
        print("    of motor_std and FD_std for joint_vel DR magnitude.")
    else:
        print("    Motor velocity is close to FD — minimal filtering. Use")
        print("    motor_std directly for joint_vel DR.")

    if pg_mean[2] > -0.95:
        print(f"    ! IMU projected_gravity.z mean = {pg_mean[2]:+.4f} (expected ~-1.0)")
        print("      IMU may be tilted or axis signs wrong. Check before trusting other numbers.")

    return locals()


# ────────────────────────────────────────────────────────────
# Mode 2: synchronous loop timing
# ────────────────────────────────────────────────────────────

def timing_test(bridge, imu, duration=10.0):
    print(f"\n{'=' * 64}")
    print(f"  TIMING TEST  ({duration:.0f}s)")
    print(f"{'=' * 64}")
    print("  Recording per-phase loop timing...\n")

    n = int(duration * CONTROL_HZ)

    t_imu   = np.zeros(n)
    t_read  = np.zeros(n)
    t_send  = np.zeros(n)
    t_total = np.zeros(n)

    bridge.send_idle()
    time.sleep(0.1)
    bridge.read_state()
    target = bridge.sim_joint_pos.copy()
    kp_z = np.zeros(6, dtype=np.float32)
    kd_z = np.zeros(6, dtype=np.float32)

    for i in range(n):
        t0 = time.monotonic_ns()
        imu.update()
        t1 = time.monotonic_ns()
        bridge.read_state()
        t2 = time.monotonic_ns()
        bridge.send_target(target, kp_z, kd_z)  # zero gains → no torque
        t3 = time.monotonic_ns()

        t_imu[i]   = (t1 - t0) / 1e6
        t_read[i]  = (t2 - t1) / 1e6
        t_send[i]  = (t3 - t2) / 1e6
        t_total[i] = (t3 - t0) / 1e6

        elapsed = (time.monotonic_ns() - t0) / 1e9
        if elapsed < PERIOD:
            time.sleep(PERIOD - elapsed)

    # Drop first second (warm-up)
    skip = CONTROL_HZ
    print("\n  RESULTS  (ms)\n  " + "-" * 60)
    print(f"  {'phase':<22s}  {'mean':>7s}  {'p50':>7s}  "
          f"{'p95':>7s}  {'p99':>7s}  {'max':>7s}")
    for name, arr in [
        ('imu.update()',         t_imu),
        ('bridge.read_state()',  t_read),
        ('bridge.send_target()', t_send),
        ('TOTAL synchronous',    t_total),
    ]:
        a = arr[skip:]
        print(f"  {name:<22s}  {a.mean():>7.2f}  "
              f"{np.percentile(a, 50):>7.2f}  "
              f"{np.percentile(a, 95):>7.2f}  "
              f"{np.percentile(a, 99):>7.2f}  "
              f"{a.max():>7.2f}")

    p50_total = np.percentile(t_total[skip:], 50)
    p99_total = np.percentile(t_total[skip:], 99)
    p50_steps = p50_total / (PERIOD * 1000)
    p99_steps = p99_total / (PERIOD * 1000)

    print("\n  INTERPRETATION:")
    print(f"    Synchronous loop overhead p50 = {p50_total:.1f}ms ({p50_steps:.1f} steps)")
    print(f"    Synchronous loop overhead p99 = {p99_total:.1f}ms ({p99_steps:.1f} steps)")
    print()
    print("    This is a LOWER bound on sensor-to-actuator latency. The real")
    print("    delay also includes (not measured here):")
    print("      • IMU sample staleness (up to 2.5ms at 400 Hz)")
    print("      • Motor encoder sample staleness (depends on CAN frame rate)")
    print("      • Motor controller torque-loop latency (typ. 1–5ms)")
    print()
    print("    Run stepresp for a measurement that captures all of these.")
    print()
    print(f"    Provisional action_delay DR (refine after stepresp):")
    print(f"      min = {max(int(round(p50_steps)), 1)} steps   "
          f"max = {max(int(np.ceil(p99_steps)) + 1, 3)} steps")

    return locals()


# ────────────────────────────────────────────────────────────
# Mode 3: step response on a single joint
# ────────────────────────────────────────────────────────────

def step_response_test(bridge, imu, joint_idx=1,
                       step_size=0.10, kp=2.0, kd=0.05,
                       n_pre=50, n_post=200):
    """Drive one joint with a small step, measure response time.

    joint_idx: 0..5  (default 1 = hip_pitch1)
    step_size: rad   (default 0.10 ≈ 5.7°)
    kp, kd:    sim-space gains while testing (~25% of nominal)
    """
    print(f"\n{'=' * 64}")
    print(f"  STEP RESPONSE TEST  ({SIM_NAMES[joint_idx]})")
    print(f"{'=' * 64}")
    print(f"  step:    {step_size:.3f} rad  (~{np.degrees(step_size):.1f}°)")
    print(f"  Kp/Kd:   {kp:.2f} / {kd:.3f}  (sim-space)")
    print(f"  pre/post: {n_pre} / {n_post} samples ({n_pre*PERIOD*1000:.0f}ms / "
          f"{n_post*PERIOD*1000:.0f}ms)")
    print()
    print("  Hang robot. The selected joint will twitch ~5° in 3 seconds.")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    # Settle reading
    bridge.read_state()
    initial_pos = float(bridge.sim_joint_pos[joint_idx])

    target = bridge.sim_joint_pos.copy().astype(np.float32)
    kp_vec = np.zeros(6, dtype=np.float32)
    kd_vec = np.zeros(6, dtype=np.float32)
    kp_vec[joint_idx] = kp
    kd_vec[joint_idx] = kd

    n = n_pre + n_post
    pos_log = np.zeros(n)
    cmd_log = np.zeros(n)
    t_log   = np.zeros(n)

    t0 = time.monotonic()
    step_t = None
    step_idx = None

    for i in range(n):
        t_loop = time.monotonic()
        bridge.read_state()

        if i < n_pre:
            target[joint_idx] = initial_pos
        else:
            if step_t is None:
                # First post-step iteration: timestamp the moment we
                # set the new target (closest we can get to "command sent").
                step_t   = t_loop - t0
                step_idx = i
            target[joint_idx] = initial_pos + step_size

        bridge.send_target(target, kp_vec, kd_vec)

        pos_log[i] = bridge.sim_joint_pos[joint_idx]
        cmd_log[i] = target[joint_idx]
        t_log[i]   = t_loop - t0

        elapsed = time.monotonic() - t_loop
        if elapsed < PERIOD:
            time.sleep(PERIOD - elapsed)

    bridge.send_idle()

    # Pre-step noise
    pre = pos_log[:n_pre]
    pre_std = pre.std()
    pre_mean = pre.mean()
    threshold = max(pre_std * 5.0, 0.003)

    # Time-to-first-motion: sustained motion, not a one-sample blip.
    # Require 3 consecutive samples > threshold to count.
    post_pos = pos_log[n_pre:]
    post_t   = t_log[n_pre:]
    delta    = np.abs(post_pos - pre_mean)

    above = delta > threshold
    first_idx = None
    for k in range(len(above) - 2):
        if above[k] and above[k+1] and above[k+2]:
            first_idx = k
            break

    print("\n  RESULTS\n  " + "-" * 60)
    print(f"\n  Pre-step pos: mean = {pre_mean:+.5f}  std = {pre_std:.5f}")
    print(f"  Motion threshold = {threshold:.5f} (5×std, min 0.003)")

    if first_idx is not None:
        t_at_motion = post_t[first_idx]
        delay_s = t_at_motion - step_t
        delay_ms = delay_s * 1000.0
        delay_steps = delay_s / PERIOD
        print(f"  Time-to-first-motion: {delay_ms:.1f}ms  ({delay_steps:.1f} sim steps)")
    else:
        delay_ms = float('nan')
        delay_steps = float('nan')
        print("  ! Joint did not move above threshold. Try larger step or more Kp.")

    # Print a compact response trace
    print("\n  Response trace (relative to step):")
    print(f"  {'t (ms)':>8s}  {'pos_delta':>10s}  {'cmd_delta':>10s}  visualization")
    for i in range(0, min(n_post, 100), 4):
        t_rel = (post_t[i] - step_t) * 1000
        pd = post_pos[i] - pre_mean
        cd = cmd_log[n_pre + i] - pre_mean
        marker = '*' if (first_idx is not None and i == first_idx) else ' '
        bar_len = int(min(abs(pd) / max(abs(step_size), 1e-6) * 30, 30))
        bar = '#' * bar_len
        print(f"  {t_rel:>8.1f}  {pd:>+10.5f}  {cd:>+10.5f}  {marker} {bar}")

    print("\n  INTERPRETATION:")
    if not np.isnan(delay_steps):
        print(f"    Total sensor-to-actuator round-trip ≈ {delay_ms:.0f}ms ≈ "
              f"{delay_steps:.1f} control steps.")
        print()
        print("    Recommended action_delay DR:")
        lo = max(int(round(delay_steps)) - 1, 1)
        hi = max(int(np.ceil(delay_steps)) + 1, lo + 2)
        print(f"      Random per-env per-episode in [{lo}, {hi}] steps.")
        print(f"      Implementation: action buffer of size {hi}, sample delay")
        print(f"      from uniform[{lo}, {hi}] at reset, hold for the episode.")

    return locals()


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    mode = sys.argv[1].lower()

    print("Initializing motor bridge and IMU...")
    bridge = MotorBridge()
    imu = IMUAdapter()
    imu.open()

    # Settle IMU
    for _ in range(20):
        imu.update()
        time.sleep(0.005)

    bridge.enable()
    time.sleep(0.3)
    bridge.send_idle()
    bridge.read_state()
    print("Ready.\n")

    try:
        if mode == 'static':
            static_test(bridge, imu)
        elif mode == 'timing':
            timing_test(bridge, imu)
        elif mode == 'stepresp':
            joint_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 1
            step_response_test(bridge, imu, joint_idx=joint_idx)
        elif mode == 'all':
            static_test(bridge, imu)
            timing_test(bridge, imu)
            step_response_test(bridge, imu, joint_idx=1)
        else:
            print(f"Unknown mode: {mode}\n")
            print(__doc__)
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n[interrupted]")
    finally:
        try:
            for _ in range(3):
                bridge.send_idle()
                time.sleep(0.02)
            bridge.disable()
        except Exception as e:
            print(f"[WARN] disable failed: {e}")
        try:
            bridge.close()
        except Exception:
            pass
        try:
            imu.close()
        except Exception:
            pass
        print("\nMotors disabled. Done.")


if __name__ == '__main__':
    main()
