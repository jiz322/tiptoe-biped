#!/usr/bin/env python3
"""
ballistic_torque_probe.py — Replant M3.1 without a load rig: measure the
RS01 torque ceiling from acceleration against a known inertia instead of
a stall. NO human contact, no lever, no load cell.

Protocol per rep: PD-hold at home → fire tau_ff for a short burst → abort
at travel/speed guard → actively damp. Command levels sweep 4→17 Nm, both
directions. Levels <= --cal-max Nm are the calibration group: below any
candidate clamp, delivered == commanded there, so they pin the effective
inertia I_eff = tau/alpha in place (no need to know added-mass radius).
At high levels, delivered = I_eff * alpha. If the 23 A current clamp is
real, delivered flattens at ~12.9 Nm while commanded keeps rising; if
the ceiling is the 17 Nm command clip, the line stays straight.

STRONGLY RECOMMENDED: strap a rigid ~1 kg mass TIGHTLY to the shank
first — it slows the burst (~40 ms windows, end speeds ~10-15 rad/s
instead of ~35) and lengthens the measurement. Any rigid mass works; the
calibration group absorbs its inertia automatically. Clear swing arc,
e-stop in reach; motor temp is logged each rep and the sweep aborts
above --temp-max.

Usage:
    python3 ballistic_torque_probe.py --motor 101 --out torq_101.npz
"""

import argparse
import time

import numpy as np

import config as C
from robstride_socketcan_driver import make_driver

LEVELS = [4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 17.0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--motor', type=int, default=101)
    ap.add_argument('--reps', type=int, default=4, help='per level (both dirs)')
    ap.add_argument('--dir', type=int, default=0, choices=(-1, 0, 1),
                    help='+1/-1 = fire ALL bursts in that direction only '
                         '(use the direction AWAY from the torso when a '
                         'heavy mass is strapped on); 0 = alternate')
    ap.add_argument('--travel', type=float, default=0.50, help='abort, rad')
    ap.add_argument('--vmax', type=float, default=25.0, help='abort, rad/s')
    ap.add_argument('--window', type=float, default=0.10, help='max burst, s')
    ap.add_argument('--cal-max', type=float, default=6.0,
                    help='levels <= this calibrate I_eff')
    ap.add_argument('--temp-max', type=float, default=60.0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    drv = make_driver(motor_ids=[args.motor])
    store, results = {}, []
    n_rep = 0
    try:
        drv.enable_all()
        for _ in range(5):
            drv.send_command(args.motor, 0.0, 0.0, 0.0, 0.0, 0.0)
            time.sleep(0.01)
        home = drv.get_state(args.motor)[0]
        print(f"[torq] motor {args.motor}: levels {LEVELS} Nm x {args.reps}, "
              f"home {home:+.3f} rad")

        for level in LEVELS:
            for rep in range(args.reps):
                sign = args.dir if args.dir else (1 if rep % 2 == 0 else -1)
                t_end = time.monotonic() + 1.2            # hold + cool spacing
                while time.monotonic() < t_end:
                    drv.send_command(args.motor, home, 0.0, 8.0, 0.8, 0.0)
                    time.sleep(0.005)
                temp = drv.get_state(args.motor)[3]
                if temp > args.temp_max:
                    print(f"[torq] ABORT: motor temp {temp:.1f} C")
                    raise KeyboardInterrupt
                ts, ps, vs = [], [], []
                t0 = time.monotonic()
                while True:
                    t = time.monotonic() - t0
                    if t > args.window:
                        break
                    drv.send_command(args.motor, 0.0, 0.0, 0.0, 0.0,
                                     sign * level)
                    pos, vel, _, _ = drv.get_state(args.motor)
                    ts.append(t)
                    ps.append(pos)
                    vs.append(vel)
                    if abs(pos - home) > args.travel or abs(vel) > args.vmax:
                        break
                t_end = time.monotonic() + 0.8            # PD pull-back brake
                while time.monotonic() < t_end:
                    drv.send_command(args.motor, home, 0.0, 6.0, 1.5, 0.0)
                    time.sleep(0.005)
                key = f"n{n_rep}"
                store[key + "_t"] = np.asarray(ts)
                store[key + "_v"] = np.asarray(vs)
                store[key + "_p"] = np.asarray(ps)
                store[key + "_meta"] = np.array([level, sign, temp])
                n_rep += 1
                # slope of v(t): post current-rise, pre guard
                t_a, v_a = np.asarray(ts), sign * np.asarray(vs)
                m = (t_a > 0.0025) & (v_a < 0.9 * args.vmax)
                if m.sum() >= 6:
                    A = np.stack([t_a[m], np.ones(m.sum())], axis=1)
                    (alpha, _), *_ = np.linalg.lstsq(A, v_a[m], rcond=None)
                    results.append((level, sign, alpha))
                    print(f"  {level:4.0f} Nm dir {sign:+d}: "
                          f"alpha {alpha:7.1f} rad/s^2  n={len(t_a)}  "
                          f"T={temp:.0f}C")
                else:
                    print(f"  {level:4.0f} Nm dir {sign:+d}: too short "
                          f"(n={len(t_a)}) — reduce speed with more mass?")
    except KeyboardInterrupt:
        print("\n[torq] stopped")
    finally:
        try:
            t_end = time.monotonic() + 1.0
            while time.monotonic() < t_end:
                drv.send_command(args.motor, home, 0.0, 8.0, 1.0, 0.0)
                time.sleep(0.005)
            drv.disable_all()
        except Exception:
            pass

    if results:
        r = np.array(results)
        cal = r[r[:, 0] <= args.cal_max]
        if len(cal):
            I_eff = np.median(cal[:, 0] / cal[:, 2])
            print(f"\n[torq] I_eff from <= {args.cal_max} Nm group: "
                  f"{I_eff*1000:.2f}e-3 kg m^2")
            print(f"  {'cmd':>6s} {'delivered':>10s} {'ratio':>6s}")
            for level in sorted(set(r[:, 0])):
                a = np.median(r[r[:, 0] == level][:, 2])
                dl = I_eff * a
                print(f"  {level:6.1f} {dl:10.2f} {dl/level:6.2f}")
            print("  (flat 'delivered' above ~12.9 = current clamp real; "
                  "ratio ~1 to 17 = command clip only)")
    if args.out and store:
        np.savez(args.out, motor=args.motor, home=home, **store)
        print(f"[torq] saved {args.out}")


if __name__ == '__main__':
    main()
