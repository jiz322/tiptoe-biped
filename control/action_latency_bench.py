#!/usr/bin/env python3
"""
action_latency_bench.py — Replant runbook M1.1: action -> torque-onset
latency on CAN2SPI, host clock only.

MIT mode. Each rep: PD-hold the joint at its start pose right up to the
step instant (so the baseline is true rest — a limp settle lets gravity
swing the leg and buries the onset), host-timestamp the tau_ff step
(kp=kd=0, pure tau), then poll state as fast as the bus allows (~1.9 kHz;
every send_command reply is a sample). The tau hold is aborted the moment
the joint moves --travel rad (default 0.06) and the joint is actively
damped, so the limb only twitches. Onset is found per-rep by detrending
the first 6 ms and fitting the constant-acceleration model —
sqrt(residual) is linear in t, its zero-intercept is t0. (A plain
threshold crossing overestimates latency by the time-to-threshold; both
numbers are reported.)

Directions alternate +/-. The joint must be free to move ~0.3 rad both
ways from its start pose. Ctrl-C makes the motor limp and disables it.

Feeds -> cfg §12b dr_action_delay_min/max, per joint type (run once each
on a hip pitch 101/104, a knee 102/105, a roll 103/106).

Usage:
    python3 action_latency_bench.py --motor 101 --out lat_101.npz
    python3 action_latency_bench.py --motor 103 --tau 0.8 --out lat_103.npz
"""

import argparse
import math
import time

import numpy as np

import config as C
from robstride_socketcan_driver import make_driver


def measure_rep(drv, mid, tau, window_s, home, kp_hold, travel):
    """PD-hold at home right up to the step instant (rest baseline), then
    swap to pure tau and poll. Aborts the tau hold once |travel| is
    exceeded so the limb only twitches. Returns (t[], pos[]) with t=0 at
    the step command."""
    # settle under PD hold
    t_end = time.monotonic() + 0.6
    while time.monotonic() < t_end:
        drv.send_command(mid, home, 0.0, kp_hold, 0.5, 0.0)
        time.sleep(0.005)
    # held baseline, polled fast (~15 ms)
    tb, pb = [], []
    t_ref = time.monotonic()
    while time.monotonic() - t_ref < 0.015:
        drv.send_command(mid, home, 0.0, kp_hold, 0.5, 0.0)
        tb.append(time.monotonic() - t_ref)
        pb.append(drv.get_state(mid)[0])
    p_base = float(np.median(pb))
    # the step: pure tau, kp=kd=0
    ts, ps = [], []
    t_send = time.monotonic()
    drv.send_command(mid, 0.0, 0.0, 0.0, 0.0, tau)
    while True:
        now = time.monotonic()
        if now - t_send > window_s:
            break
        pos = drv.get_state(mid)[0]
        ts.append(now - t_send)
        ps.append(pos)
        if abs(pos - p_base) > travel:          # guard: cut tau, done
            break
        drv.send_command(mid, 0.0, 0.0, 0.0, 0.0, tau)
    # actively damp whatever momentum is left
    t_end = time.monotonic() + 0.25
    while time.monotonic() < t_end:
        drv.send_command(mid, 0.0, 0.0, 0.0, 1.0, 0.0)
        time.sleep(0.005)
    return np.asarray(ts), np.asarray(ps)


def onset_from_rep(t, p, direction, lsb=25.14 / 65536):
    """Detrended constant-acceleration onset (2026-08-17 rev: the naive
    threshold+wide-band fit failed on gravity drift — detrend on the
    first 6 ms, then fit sqrt(residual) over a NARROW fixed band).
    Returns (t0_fit, t_thresh) in seconds, or Nones."""
    if len(t) < 30:
        return None, None
    m = t < 0.006
    if m.sum() < 6:
        return None, None
    A = np.stack([t[m], np.ones(m.sum())], axis=1)
    (v0, c0), *_ = np.linalg.lstsq(A, p[m], rcond=None)
    r = (p - (v0 * t + c0)) * direction        # + = with tau
    noise = np.abs(r[m]).max()
    lo = max(2 * noise, 3 * lsb)
    hi = lo + 25 * lsb
    over = np.nonzero(r > lo)[0]
    t_thresh = t[over[0]] if len(over) else None
    sel = (r > lo) & (r < hi)
    if sel.sum() < 5:
        return None, t_thresh
    x, y = t[sel], np.sqrt(r[sel])
    A2 = np.stack([x, np.ones_like(x)], axis=1)
    (c, icpt), *_ = np.linalg.lstsq(A2, y, rcond=None)
    if c <= 0:
        return None, t_thresh
    t0 = -icpt / c
    if not (0.0 < t0 < 0.05):
        return None, t_thresh
    return t0, t_thresh


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--motor', type=int, default=101)
    ap.add_argument('--tau', type=float, default=1.5, help='step torque, Nm')
    ap.add_argument('--reps', type=int, default=50, help='total (both dirs)')
    ap.add_argument('--window', type=float, default=0.06,
                    help='poll window after the step, s')
    ap.add_argument('--travel', type=float, default=0.06,
                    help='abort tau once the joint moves this far, rad')
    ap.add_argument('--kp-hold', type=float, default=8.0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if args.motor not in C.MOTOR_IDS:
        raise SystemExit(f"motor {args.motor} not in config.MOTOR_IDS")
    if abs(args.tau) > 3.0:
        raise SystemExit("--tau capped at 3 Nm for the bench")

    drv = make_driver(motor_ids=[args.motor])
    results = []            # (direction, t0_fit_ms, t_thresh_ms)
    raw_t, raw_p, raw_dir = [], [], []
    try:
        drv.enable_all()
        for _ in range(5):
            drv.send_command(args.motor, 0.0, 0.0, 0.0, 0.0, 0.0)
            time.sleep(0.01)
        home = drv.get_state(args.motor)[0]
        print(f"[lat] motor {args.motor}: tau ±{args.tau} Nm x {args.reps} "
              f"reps, window {args.window*1000:.0f} ms, home {home:+.3f} rad")

        for rep in range(args.reps):
            sign = 1 if rep % 2 == 0 else -1
            t, p = measure_rep(drv, args.motor, sign * args.tau, args.window,
                               home, args.kp_hold, args.travel)
            t0, tth = onset_from_rep(t, p, sign)
            raw_t.append(t); raw_p.append(p); raw_dir.append(sign)
            if t0 is not None:
                results.append((sign, t0 * 1000, tth * 1000))
                print(f"  rep {rep+1:3d} dir {sign:+d}: onset {t0*1000:6.2f} ms "
                      f"(thresh {tth*1000:6.2f} ms)  n={len(t)}")
            else:
                print(f"  rep {rep+1:3d} dir {sign:+d}: NO ONSET "
                      f"(n={len(t)}, thresh at "
                      f"{'-' if tth is None else f'{tth*1000:.1f}ms'}) — "
                      f"friction not broken? raise --tau")
    except KeyboardInterrupt:
        print("\n[lat] interrupted")
    finally:
        try:
            t_end = time.monotonic() + 0.5
            while time.monotonic() < t_end:
                drv.send_command(args.motor, 0.0, 0.0, 0.0, 0.5, 0.0)
                time.sleep(0.005)
            drv.disable_all()
        except Exception:
            pass

    if results:
        arr = np.array(results)
        print(f"\n[lat] motor {args.motor}, {len(arr)}/{args.reps} reps "
              f"with onset:")
        for lbl, col in (('fit-onset', 1), ('threshold', 2)):
            for dsel, dl in ((arr[:, 0] > 0, '+'), (arr[:, 0] < 0, '-'),
                             (np.ones(len(arr), bool), 'both')):
                x = arr[dsel, col]
                if len(x):
                    print(f"    {lbl:>9s} dir {dl:>4s}: mean {x.mean():5.2f}  "
                          f"p95 {np.percentile(x, 95):5.2f}  "
                          f"std {x.std():4.2f} ms  (n={len(x)})")
        both = arr[:, 1]
        step_ms = 1000.0 / getattr(C, 'CONTROL_HZ', 100)
        spread = np.percentile(both, 95) - np.percentile(both, 5)
        print(f"    p95-p05 spread {spread:.2f} ms vs control step "
              f"{step_ms:.0f} ms -> "
              + ("OK" if spread < step_ms else "INVESTIGATE JITTER (runbook)"))
        print(f"    feeds dr_action_delay: min "
              f"{math.floor(both.min() / step_ms)}, max "
              f"{math.ceil(np.percentile(both, 95) / step_ms)} steps")
    else:
        print("[lat] no usable reps")

    if args.out:
        np.savez(args.out,
                 results=np.array(results) if results else np.zeros((0, 3)),
                 dirs=np.array(raw_dir),
                 **{f"t_{i}": a for i, a in enumerate(raw_t)},
                 **{f"p_{i}": a for i, a in enumerate(raw_p)})
        print(f"[lat] saved {args.out}")


if __name__ == '__main__':
    main()
