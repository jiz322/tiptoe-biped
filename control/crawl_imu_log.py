"""crawl_imu_log.py — Phase 6 (replant runbook): hardware IMU attitude log
at a self-supporting reference pose. NO motor traffic — IMU serial only.

Purpose: the +0.075 static balance-point bias conflates IMU mounting,
joint zeroing, and CoM error. A mounting rotation is invisible to the IMU
alone; it only shows against an IMU-independent reference. The crawl pose
is self-supporting (floor + geometry set the torso attitude), so

    imu_mounting_offset ≈ pg_hw_RAW(crawl) − pg_sim(crawl)

with pg_sim(crawl) from data/_crawl_imu_reference.py. A second pose with
very different pitch (supine on-back) separates a fixed mounting rotation
(same delta both poses) from pose-dependent geometry error.

Usage (robot resting on floor in the pose, motors unpowered, hands off,
tether slack):
    python crawl_imu_log.py --tag crawl --seconds 15
    python crawl_imu_log.py --tag on_back --seconds 15

Prints mean/std of RAW projected gravity (trim NOT applied), the trimmed
value the policy would see, gyro means (bias sanity), and saves an npz.
"""
import argparse
import time

import numpy as np

import config as C
from imu_adapter import IMUAdapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="crawl")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--hz", type=float, default=100.0)
    args = ap.parse_args()

    imu = IMUAdapter()
    imu.open()
    print(f"[crawl_imu_log] warming up IMU stream 2 s ...")
    t0 = time.monotonic()
    while time.monotonic() - t0 < 2.0:
        imu.update()
        time.sleep(0.01)

    n_target = int(args.seconds * args.hz)
    pg_raw = np.zeros((n_target, 3), dtype=np.float64)
    gyro = np.zeros((n_target, 3), dtype=np.float64)
    quat = np.zeros((n_target, 4), dtype=np.float64)
    ts = np.zeros(n_target, dtype=np.float64)

    print(f"[crawl_imu_log] logging {args.seconds:.0f} s at {args.hz:.0f} Hz "
          f"(tag={args.tag}) — do not touch the robot")
    period = 1.0 / args.hz
    t_next = time.monotonic()
    for i in range(n_target):
        imu.update()
        pg_raw[i] = imu.projected_gravity_raw()
        gyro[i] = imu.ang_vel_body()
        quat[i] = imu.quat()
        ts[i] = time.monotonic()
        t_next += period
        dt = t_next - time.monotonic()
        if dt > 0:
            time.sleep(dt)
    imu.close()

    mean = pg_raw.mean(0)
    sd = pg_raw.std(0)
    gmean = gyro.mean(0)
    # movement guard: a hanging / touched robot shows up as pg drift
    half = n_target // 2
    drift = pg_raw[half:].mean(0) - pg_raw[:half].mean(0)
    pitch_deg = np.degrees(np.arcsin(np.clip(mean[0], -1, 1)))

    trim = np.array([C.IMU_PG_OFFSET[0], C.IMU_PG_OFFSET[1], 0.0])
    print()
    print(f"  RAW  proj_grav  mean  x {mean[0]:+.4f}  y {mean[1]:+.4f}  z {mean[2]:+.4f}")
    print(f"                  sd    x {sd[0]:.4f}  y {sd[1]:.4f}  z {sd[2]:.4f}")
    print(f"  first/second-half drift x {drift[0]:+.4f}  y {drift[1]:+.4f}"
          f"   (>0.005 = robot moved / swinging — redo)")
    print(f"  gyro mean [rad/s]  x {gmean[0]:+.4f}  y {gmean[1]:+.4f}  z {gmean[2]:+.4f}")
    print(f"  body pitch from RAW pg_x: {pitch_deg:+.2f} deg")
    print(f"  trimmed (policy-visible) x {mean[0]-trim[0]:+.4f}  y {mean[1]-trim[1]:+.4f}")
    print(f"  IMU fps: {imu.fps if hasattr(imu, 'fps') else '?'}")

    out = f"crawl_imu_{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    np.savez(out, pg_raw=pg_raw, gyro=gyro, quat=quat, ts=ts,
             tag=args.tag, pg_offset_cfg=np.asarray(C.IMU_PG_OFFSET, float))
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
