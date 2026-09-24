#!/usr/bin/env python3
"""
kd_brake_test.py — Replant M2.1 (decisive variant): kd-path filter via
brake-decay signature.

The tau-sine FRF approach drowned in stiction harmonics and base
compliance (kdchirp_101_2 post-mortem). This test is a direct
discriminator: spin the joint to ~1.5 rad/s with a short tau pulse, then
switch to PURE kd damping (kp=0, v_des=0, tau=0) and record the decay.

    no filter:   I v' = -kd v            -> clean exponential, T = I/kd
    filter lag:  I v' = -kd H(s) v       -> 2nd order; for one-pole H,
                 poles of  s(1+s*tau_f) + kd/I = 0  — at kd=2, a tau_f of
                 even a few ms makes the brake RING (v overshoots through
                 zero). The presence/absence of that ring at high kd is
                 hard to fake with friction or compliance.

Reps alternate direction; two kd values (default 0.5 and 2.0) — the high
one maximizes the ring signature. Travel per rep ~0.1 rad. PD re-center
between reps; Ctrl-C damps + disables.

Usage:
    python3 kd_brake_test.py --motor 101 --out kdbrake_101.npz
"""

import argparse
import time

import numpy as np

import config as C
from robstride_socketcan_driver import make_driver


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--motor', type=int, default=101)
    ap.add_argument('--kds', default='0.5,2.0')
    ap.add_argument('--tau', type=float, default=0.8, help='pulse torque, Nm')
    ap.add_argument('--pulse', type=float, default=0.020, help='pulse len, s')
    ap.add_argument('--decay', type=float, default=0.30, help='record len, s')
    ap.add_argument('--reps', type=int, default=20, help='per kd value')
    ap.add_argument('--travel', type=float, default=0.35)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    kds = [float(s) for s in args.kds.split(',')]
    drv = make_driver(motor_ids=[args.motor])
    store = {}
    try:
        drv.enable_all()
        for _ in range(5):
            drv.send_command(args.motor, 0.0, 0.0, 0.0, 0.0, 0.0)
            time.sleep(0.01)
        home = drv.get_state(args.motor)[0]
        print(f"[brake] motor {args.motor}, home {home:+.3f}, "
              f"kds {kds}, pulse {args.tau} Nm x {args.pulse*1000:.0f} ms")

        for kd in kds:
            for rep in range(args.reps):
                sign = 1 if rep % 2 == 0 else -1
                t_end = time.monotonic() + 0.8            # re-center
                while time.monotonic() < t_end:
                    drv.send_command(args.motor, home, 0.0, 8.0, 0.8, 0.0)
                    time.sleep(0.005)
                ts, ps, vs = [], [], []
                t0 = time.monotonic()
                while True:                               # pulse, then brake
                    t = time.monotonic() - t0
                    if t > args.pulse + args.decay:
                        break
                    if t < args.pulse:
                        drv.send_command(args.motor, 0.0, 0.0, 0.0, 0.0,
                                         sign * args.tau)
                    else:
                        drv.send_command(args.motor, 0.0, 0.0, 0.0, kd, 0.0)
                    pos, vel, _, _ = drv.get_state(args.motor)
                    ts.append(t)
                    ps.append(pos)
                    vs.append(vel)
                    if abs(pos - home) > args.travel:
                        break
                key = f"kd{kd:g}_r{rep}"
                store[key + "_t"] = np.asarray(ts)
                store[key + "_p"] = np.asarray(ps)
                store[key + "_v"] = np.asarray(vs)
                vpk = float(np.abs(np.asarray(vs)).max()) if vs else 0.0
                if rep % 5 == 0:
                    print(f"  kd={kd:g} rep {rep+1}: v_peak {vpk:4.2f} rad/s, "
                          f"n={len(ts)}")
    except KeyboardInterrupt:
        print("\n[brake] interrupted")
    finally:
        try:
            t_end = time.monotonic() + 0.8
            while time.monotonic() < t_end:
                drv.send_command(args.motor, home, 0.0, 8.0, 0.8, 0.0)
                time.sleep(0.005)
            drv.disable_all()
        except Exception:
            pass

    if args.out and store:
        np.savez(args.out, kds=np.array(kds), home=home,
                 pulse=args.pulse, tau=args.tau, **store)
        print(f"[brake] saved {args.out} ({len(store)//3} reps)")


if __name__ == '__main__':
    main()
