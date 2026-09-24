#!/usr/bin/env python3
"""
vel_decode_check.py — Replant runbook Phase 0: velocity-decode spin test.

Drives ONE joint through a slow position sine in MIT mode and logs the
motor's own (pos, vel) feedback at ~100 Hz. Position is the integral of
velocity, so for a self-consistent decode  vel == d(pos)/dt  regardless
of the physics. Fits  vel_fb ~= k * dpos/dt;  PASS iff |k - 1| < 2%.

The June sysid data failed exactly this with k ~= 1.136 (= 50/44: RS01
velocity decoded with the EL05 range). Fixed since in MOTOR_MIT_OVERRIDES
(robstride_biped_driver.py); this is the on-hardware regression check the
runbook gates everything on — if it fails, STOP, every measurement today
inherits the error.

Robot on stand / hanging so the joint can move freely. Gentle gains, sine
amplitude ramps in over 2 s, motor is made limp and disabled on exit or
Ctrl-C.

Usage:
    python3 vel_decode_check.py --motor 101              # L hip_pitch (RS01)
    python3 vel_decode_check.py --motor 103 --amp 0.25   # L roll (EL05)
"""

import argparse
import math
import time

import numpy as np

import config as C
from robstride_socketcan_driver import make_driver


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--motor', type=int, default=101)
    ap.add_argument('--amp', type=float, default=0.4, help='sine amplitude, rad')
    ap.add_argument('--freq', type=float, default=0.3, help='sine frequency, Hz')
    ap.add_argument('--secs', type=float, default=20.0)
    ap.add_argument('--kp', type=float, default=8.0)
    ap.add_argument('--kd', type=float, default=0.5)
    ap.add_argument('--hz', type=float, default=100.0)
    ap.add_argument('--ramp', type=float, default=2.0,
                    help='amplitude ramp-in seconds (excluded from the fit)')
    ap.add_argument('--out', default=None, help='save t/pos/vel to this .npz')
    args = ap.parse_args()

    if args.motor not in C.MOTOR_IDS:
        raise SystemExit(f"motor {args.motor} not in config.MOTOR_IDS")

    drv = make_driver(motor_ids=[args.motor])
    dt_cmd = 1.0 / args.hz
    t_log, p_log, v_log = [], [], []
    try:
        drv.enable_all()
        # settle: read the current position, center the sine there
        for _ in range(5):
            drv.send_command(args.motor, 0.0, 0.0, 0.0, 0.0, 0.0)
            time.sleep(0.01)
        p0 = drv.get_state(args.motor)[0]
        print(f"[vel_check] motor {args.motor}: center {p0:+.3f} rad, "
              f"±{args.amp} rad @ {args.freq} Hz for {args.secs:.0f}s")

        t_start = time.monotonic()
        next_tick = t_start
        while True:
            t = time.monotonic() - t_start
            if t > args.secs:
                break
            a = args.amp * min(1.0, t / args.ramp)
            p_des = p0 + a * math.sin(2 * math.pi * args.freq * t)
            drv.send_command(args.motor, p_des, 0.0, args.kp, args.kd, 0.0)
            pos, vel, _, temp = drv.get_state(args.motor)
            t_log.append(time.monotonic() - t_start)
            p_log.append(pos)
            v_log.append(vel)
            next_tick += dt_cmd
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\n[vel_check] interrupted")
    finally:
        try:
            drv.send_command(args.motor, 0.0, 0.0, 0.0, 0.5, 0.0)  # damp only
            time.sleep(0.2)
            drv.disable_all()
        except Exception:
            pass

    t = np.asarray(t_log)
    p = np.asarray(p_log)
    v = np.asarray(v_log)
    keep = t > args.ramp
    if keep.sum() < 100:
        raise SystemExit("not enough samples after ramp — run longer")
    t, p, v = t[keep], p[keep], v[keep]

    # central difference on the real (jittery) timestamps
    dpdt = np.gradient(p, t)
    # least-squares  v ~= k * dpdt + b   (diagnostic only: x-noise from
    # timestamp jitter dilutes this slope ~2% low — 2026-08-17 bench)
    A = np.stack([dpdt, np.ones_like(dpdt)], axis=1)
    (k, b), *_ = np.linalg.lstsq(A, v, rcond=None)
    resid = v - (k * dpdt + b)
    r2 = 1.0 - resid.var() / v.var()

    # PRIMARY verdict: lock-in at the drive frequency (2026-08-24 rewrite).
    # v(t) = k * dp/dt holds PER FOURIER COMPONENT for any waveform, so
    #     k = |V(f0)| / (w0 * |P(f0)|)
    # is an exact decode-scale estimate under stiction flats, gravity
    # asymmetry, vel DC bias, slow drift and quantization — the projection
    # onto sin/cos at f0 over whole cycles rejects all of them.
    #
    # The previous "integral fit"  k = (iv@iv)/(pu@iv)  was structurally
    # biased HIGH: with iv = k*pu + n, it reads k + ||n||^2-ish, and the
    # vel channel's ~-0.01 rad/s DC offset integrates into a ramp that
    # mean-removal does not remove. On-rig 2026-08-24 it read +5..24%
    # (worst at small amplitude) while the naive slope pinned ~1.00 —
    # every failure was k>1, the estimator's bias direction.
    w0 = 2.0 * math.pi * args.freq
    n_cyc = int((t[-1] - t[0]) * args.freq)
    if n_cyc < 3:
        raise SystemExit("need >=3 whole drive cycles after ramp — run longer")
    m = t <= (t[0] + n_cyc / args.freq)          # whole cycles only
    tm, pm, vm = t[m], p[m], v[m]
    ph = w0 * (tm - tm[0])

    _trapz = getattr(np, 'trapezoid', None) or np.trapz  # numpy 2.x rename

    def _lockin_amp(x):
        x = x - x.mean()
        c = _trapz(x * np.cos(ph), tm)
        s = _trapz(x * np.sin(ph), tm)
        return math.hypot(c, s)

    p_amp = _lockin_amp(pm)
    k_lock = _lockin_amp(vm) / (w0 * p_amp) if p_amp > 0 else float('nan')
    err_pct = abs(k_lock - 1.0) * 100.0

    # Secondary diagnostic: the cumtrapz comparison, done right — detrend
    # both series (kills the DC-bias ramp) and regress with the clean
    # variable (pos) in the quadratic:  k = <pu,iv>/<pu,pu>.
    pu = p - p.mean()
    iv = np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) / 2 * np.diff(t))])
    tc = t - t.mean()
    for arr in (pu, iv):
        arr -= arr.mean()
        arr -= (arr @ tc) / (tc @ tc) * tc
    k_int = float(pu @ iv) / float(pu @ pu) if pu @ pu else float('nan')

    print(f"\n[vel_check] n={len(t)}  peak |vel| {np.abs(v).max():.2f} rad/s  "
          f"({n_cyc} whole cycles)")
    print(f"[vel_check] naive slope: vel = {k:.4f} * dpos/dt {b:+.4f}  "
          f"R^2={r2:.4f}  (diagnostic — reads low under jitter)")
    print(f"[vel_check] integral fit (detrended, diagnostic): k = {k_int:.4f}")
    print(f"[vel_check] lock-in @ {args.freq} Hz: k = {k_lock:.4f}   "
          f"|k-1| = {err_pct:.2f}%  ->  "
          + ("PASS (<2%)" if err_pct < 2.0 else
             "FAIL — STOP: decode error, see docstring"))
    if args.out:
        np.savez(args.out, t=t, pos=p, vel=v, k=k, b=b, r2=r2,
                 k_int=k_int, k_lock=k_lock)
        print(f"[vel_check] saved {args.out}")


if __name__ == '__main__':
    main()
