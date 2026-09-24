#!/usr/bin/env python3
"""
kd_filter_chirp.py — Replant runbook M2.1: identify the velocity filter
in the MIT-mode kd path (spd_filt_gain EMA), hands-free.

Why not the hand-backdrive from the runbook: commanded iq is not
observable in MIT mode (private reads are ignored there, torque_fdb is
filtered junk), and M1.2 already showed the TELEMETRY velocity is only
~0.7 ms behind truth — the open question is whether the KD PATH sees the
same light filter or a heavier one.

Method: the motor excites itself with a small tau_ff sinusoid at each
frequency in the ladder, robot hanging, joint near its free-hang
equilibrium. Two passes: kd=0, then kd=--kd. Linearized dynamics:

    I q'' = T - [b + kd*H(s)] q' (+ gravity ~ pendulum stiffness)

Per dwell, lock-in extraction of the complex fundamental Q(f)/T(f); the
kd=0 pass pins I and b; the kd-on pass then yields kd*H(f) — gain and
phase of H across the ladder locate the EMA corner (and its update
rate). Also logged: vel_fb per sample, so the same run re-measures the
feedback-channel filter (phase of vel_fb vs d(pos)/dt per dwell) —
M1.2's 0.7 ms cross-check for free.

Safety: travel guard aborts a dwell beyond --travel rad (retries at half
amplitude once), PD re-centering between dwells, Ctrl-C damps and
disables. Amplitude grows with frequency to fight the 1/(I w^2) rolloff
but is capped at --tau-max.

Usage:
    python3 kd_filter_chirp.py --motor 101 --kd 0.5 --out kdchirp_101.npz
"""

import argparse
import math
import time

import numpy as np

import config as C
from robstride_socketcan_driver import make_driver

FREQS = [1.0, 2.0, 4.0, 8.0, 12.0, 18.0, 25.0]


def damp_to(drv, mid, target, secs, kp=8.0, kd=0.8):
    t_end = time.monotonic() + secs
    while time.monotonic() < t_end:
        drv.send_command(mid, target, 0.0, kp, kd, 0.0)
        time.sleep(0.005)


def run_dwell(drv, mid, freq, amp, kd, dwell_s, home, travel):
    """One sinusoidal tau dwell. Returns (t, pos, vel, tau) or None if the
    travel guard tripped."""
    ts, ps, vs, cs = [], [], [], []
    t0 = time.monotonic()
    while True:
        t = time.monotonic() - t0
        if t > dwell_s:
            break
        tau = amp * math.sin(2 * math.pi * freq * t)
        drv.send_command(mid, 0.0, 0.0, 0.0, kd, tau)
        pos, vel, _, _ = drv.get_state(mid)
        ts.append(t)
        ps.append(pos)
        vs.append(vel)
        cs.append(tau)
        if abs(pos - home) > travel:
            return None
    return (np.asarray(ts), np.asarray(ps), np.asarray(vs), np.asarray(cs))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--motor', type=int, default=101)
    ap.add_argument('--kd', type=float, default=0.5,
                    help='motor-space kd for the kd-on pass')
    ap.add_argument('--tau0', type=float, default=0.35,
                    help='tau amplitude at 1 Hz, Nm')
    ap.add_argument('--tau-max', type=float, default=2.0)
    ap.add_argument('--dwell', type=float, default=8.0, help='s per frequency')
    ap.add_argument('--travel', type=float, default=0.45, help='abort, rad')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if args.motor not in C.MOTOR_IDS:
        raise SystemExit(f"motor {args.motor} not in config.MOTOR_IDS")

    drv = make_driver(motor_ids=[args.motor])
    store = {}
    try:
        drv.enable_all()
        for _ in range(5):
            drv.send_command(args.motor, 0.0, 0.0, 0.0, 0.0, 0.0)
            time.sleep(0.01)
        home = drv.get_state(args.motor)[0]
        print(f"[kdchirp] motor {args.motor}, home {home:+.3f} rad, "
              f"passes kd=0 and kd={args.kd}, ladder {FREQS} Hz")

        for kd in (0.0, args.kd):
            for freq in FREQS:
                amp = min(args.tau0 * max(1.0, freq / 4.0), args.tau_max)
                for attempt in range(2):
                    damp_to(drv, args.motor, home, 1.0)
                    r = run_dwell(drv, args.motor, freq, amp, kd,
                                  args.dwell, home, args.travel)
                    if r is not None:
                        break
                    amp *= 0.5
                    print(f"  [guard] kd={kd} f={freq}: travel tripped, "
                          f"retrying at {amp:.2f} Nm")
                if r is None:
                    print(f"  [skip] kd={kd} f={freq}")
                    continue
                t, p, v, c = r
                key = f"kd{0 if kd == 0 else 1}_f{freq:g}"
                store[key + "_t"] = t
                store[key + "_p"] = p
                store[key + "_v"] = v
                store[key + "_c"] = c
                q_amp = (p.max() - p.min()) / 2
                print(f"  kd={kd:<4g} f={freq:>4g} Hz  tau {amp:4.2f} Nm  "
                      f"q_amp {q_amp*1000:6.1f} mrad  n={len(t)}")
    except KeyboardInterrupt:
        print("\n[kdchirp] interrupted")
    finally:
        try:
            damp_to(drv, args.motor, home, 0.8)
            drv.disable_all()
        except Exception:
            pass

    if args.out and store:
        np.savez(args.out, freqs=np.array(FREQS), kd_on=args.kd, home=home,
                 **store)
        print(f"[kdchirp] saved {args.out} ({len(store)//4} dwells)")


if __name__ == '__main__':
    main()
