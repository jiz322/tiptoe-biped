#!/usr/bin/env python3
"""
pos_step_test.py — Replant runbook M2.3: is the EL05 position-reference
filter (loc_ref_filt_gain=0.04, config-table only) live in MIT mode?

MIT mode, kp nominal, kd=0, tau_ff=0: command a position step and record
the rise at ~1.9 kHz. With no reference filter the response is the bare
2nd-order  I q'' = kp (q_des - q) - friction  → rings at sqrt(kp/I) with
only friction damping. A live first-order reference filter slews q_des
→ slower first-order-ish rise, less ring. Depending on the rate the
0.04 gain runs at, the filter is anywhere from ~2 ms (13 kHz) to ~50 ms
(500 Hz) — a step cleanly separates those from "not live at all".

Run on a roll motor (103/106, EL05) and on a hip pitch (101, RS01) as
the no-filter control group. Reps alternate step direction around home;
travel-guarded; PD hold + damp between reps; Ctrl-C safe.

Usage:
    python3 pos_step_test.py --motor 103 --kp 6  --out step_103.npz
    python3 pos_step_test.py --motor 101 --kp 8  --out step_101.npz
"""

import argparse
import time

import numpy as np

import config as C
from robstride_socketcan_driver import make_driver


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--motor', type=int, default=103)
    ap.add_argument('--kp', type=float, default=6.0, help='motor-space kp')
    ap.add_argument('--kd', type=float, default=0.0,
                    help='motor-space kd DURING the step (P11 1.1 runs at '
                         'deploy kd; original M2.3 battery used 0)')
    ap.add_argument('--step', type=float, default=0.15, help='step size, rad')
    ap.add_argument('--window', type=float, default=0.5, help='record len, s')
    ap.add_argument('--reps', type=int, default=16)
    ap.add_argument('--travel', type=float, default=0.45)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    drv = make_driver(motor_ids=[args.motor])
    store = {}
    home = None
    try:
        drv.enable_all()
        for _ in range(5):
            drv.send_command(args.motor, 0.0, 0.0, 0.0, 0.0, 0.0)
            time.sleep(0.01)
        home = drv.get_state(args.motor)[0]
        print(f"[step] motor {args.motor}: kp={args.kp}, kd={args.kd}, "
              f"±{args.step} rad x {args.reps}, home {home:+.3f}")

        for rep in range(args.reps):
            sign = 1 if rep % 2 == 0 else -1
            target = home + sign * args.step
            t_end = time.monotonic() + 0.8                # settle at home
            while time.monotonic() < t_end:
                drv.send_command(args.motor, home, 0.0, args.kp, 0.8, 0.0)
                time.sleep(0.005)
            ts, ps = [], []
            t0 = time.monotonic()
            while True:                                   # the step
                t = time.monotonic() - t0
                if t > args.window:
                    break
                drv.send_command(args.motor, target, 0.0, args.kp, args.kd, 0.0)
                pos, _, _, _ = drv.get_state(args.motor)
                ts.append(t)
                ps.append(pos)
                if abs(pos - home) > args.travel:
                    break
            key = f"r{rep}"
            store[key + "_t"] = np.asarray(ts)
            store[key + "_p"] = np.asarray(ps)
            store[key + "_sign"] = sign
            p = np.asarray(ps)
            if len(p) > 50:
                rise = p[-1] - p[0]
                overshoot = (sign * (p - home)).max() / args.step - 1.0
                print(f"  rep {rep+1:2d} dir {sign:+d}: n={len(p)}  "
                      f"overshoot {100*overshoot:+.0f}%")
    except KeyboardInterrupt:
        print("\n[step] interrupted")
    finally:
        try:
            if home is not None:
                t_end = time.monotonic() + 0.8
                while time.monotonic() < t_end:
                    drv.send_command(args.motor, home, 0.0, args.kp, 0.8, 0.0)
                    time.sleep(0.005)
        except Exception:
            pass
        try:
            drv.disable_all()
        except Exception:
            pass

    if args.out and store:
        np.savez(args.out, motor=args.motor, kp=args.kp, kd=args.kd,
                 step=args.step, home=home, **store)
        print(f"[step] saved {args.out}")


if __name__ == '__main__':
    main()
