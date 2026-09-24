#!/usr/bin/env python3
"""
kd_noise_hold.py — P11 runbook 1.1 kd-noise check: at-rest PD hold at
deploy gains, 30 s log, torque/velocity noise std.

The concern: kd multiplies velocity-decode noise into torque noise.
Baseline is the same hold at kd 0.5; deploy-kd std should scale
~linearly with kd (x2.8 hip at 1.4, x1.6 knee at 0.81), not blow up.

tau_fdb caveat (08-14 forensics): firmware filter ~25-35 ms + 0.15 Nm
deadband — absolute stds are telemetry-shaped; the RATIO between two
runs at identical kp is the meaningful number. Velocity std is logged
too as the unfiltered kd input.

No motion: captures the pose at enable and holds it.

Usage:
    sudo python3 kd_noise_hold.py --motor 101 --kp 20 --kd 0.5 --out ../data/kdn_101_kd05.npz
    sudo python3 kd_noise_hold.py --motor 101 --kp 20 --kd 1.4 --out ../data/kdn_101_kd14.npz
"""

import argparse
import time

import numpy as np

import config as C  # noqa: F401  (driver factory reads transport config)
from robstride_socketcan_driver import make_driver


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--motor', type=int, required=True)
    ap.add_argument('--kp', type=float, required=True, help='motor-space kp')
    ap.add_argument('--kd', type=float, required=True, help='motor-space kd')
    ap.add_argument('--dur', type=float, default=30.0, help='log length, s')
    ap.add_argument('--hz', type=float, default=100.0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    drv = make_driver(motor_ids=[args.motor])
    ts, ps, vs, taus, temps = [], [], [], [], []
    home = None
    try:
        drv.enable_all()
        for _ in range(5):
            drv.send_command(args.motor, 0.0, 0.0, 0.0, 0.0, 0.0)
            time.sleep(0.01)
        home = drv.get_state(args.motor)[0]
        print(f"[kdn] motor {args.motor}: hold kp={args.kp} kd={args.kd} "
              f"at home {home:+.3f} rad for {args.dur:.0f} s")
        period = 1.0 / args.hz
        t0 = time.monotonic()
        next_t = t0
        while True:
            t = time.monotonic() - t0
            if t > args.dur:
                break
            drv.send_command(args.motor, home, 0.0, args.kp, args.kd, 0.0)
            p, v, tau, temp = drv.get_state(args.motor)
            ts.append(t); ps.append(p); vs.append(v)
            taus.append(tau); temps.append(temp)
            next_t += period
            dt = next_t - time.monotonic()
            if dt > 0:
                time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[kdn] interrupted")
    finally:
        try:
            drv.disable_all()
        except Exception:
            pass

    if not ts:
        return
    tau = np.asarray(taus)
    v = np.asarray(vs)
    p = np.asarray(ps)
    # drop the first second (enable/settle transient)
    m = np.asarray(ts) > 1.0
    print(f"[kdn] n={m.sum()}  tau std {tau[m].std()*1e3:7.2f} mNm  "
          f"vel std {v[m].std()*1e3:7.2f} mrad/s  "
          f"pos std {p[m].std()*1e3:6.3f} mrad  "
          f"temp {temps[-1]:.1f} C")
    if args.out:
        np.savez(args.out, motor=args.motor, kp=args.kp, kd=args.kd,
                 t=np.asarray(ts), p=p, v=v, tau=tau,
                 temp=np.asarray(temps), home=home)
        print(f"[kdn] saved {args.out}")


if __name__ == '__main__':
    main()
