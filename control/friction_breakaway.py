#!/usr/bin/env python3
"""
friction_breakaway.py — Replant P5 breakaway (static friction), commanded-
torque edition. Replaces the June friction_id.py static phase, which used
torque_fdb as truth (banned: ~25-35 ms filter + 0.15 Nm deadband).

Protocol per trial: PD-hold at home -> settle -> open-loop tau_ff ramp
from 0 at a fixed rate (kp=0, tiny kd for post-breakaway damping only;
kd*v = 0 at v = 0 so the threshold is unbiased) -> onset detected from
encoder displacement past lash AND sustained velocity in the commanded
direction -> record the COMMANDED tau at onset -> brake, re-hold, rest.
Both directions, N trials each.

Gravity cancels by symmetry (same home pose every trial):
    C_s        = (tau_break_pos - tau_break_neg) / 2
    tau_g_bias = -(tau_break_pos + tau_break_neg) / 2
Run suspended (tether), leg near its hanging equilibrium; tau_g_bias
should print near 0 as a sanity check.

Full traces are stored in the npz so onset can be re-thresholded offline:
    python3 friction_breakaway.py --analyze brk_101_xxx.npz [--disp X --vsus Y]

All quantities MOTOR-SPACE raw. Space conversion happens in analysis.

Usage:
    python3 friction_breakaway.py --motor 101 --tag hang --out brk_101.npz
    python3 friction_breakaway.py --motor 102 --hold 101 --tag belt_on --out brk_102_on.npz
"""

import argparse
import time

import numpy as np

import config as C
from robstride_socketcan_driver import make_driver

# per-motor defaults: (tau_max ramp ceiling Nm motor-side, disp thresh rad,
#                      hard travel rad)
PROFILES = {
    101: (1.5, 0.04, 0.6), 104: (1.5, 0.04, 0.6),   # hip_pitch RS01
    102: (1.5, 0.10, 0.6), 105: (1.5, 0.10, 0.6),   # knee RS01 (belt lash)
    103: (1.0, 0.04, 0.3), 106: (1.0, 0.04, 0.3),   # hip_roll EL05
}

LOOP_DT = 0.0025          # ~400 Hz command/log rate
HOLD_KP, HOLD_KD = 8.0, 0.8
OTHER_KP, OTHER_KD = 15.0, 1.0
RAMP_KD = 0.2             # damps motion AFTER breakaway only
BRAKE_KP, BRAKE_KD = 6.0, 1.5


def detect_onset(t, p, v, tau, disp, vsus, vwin=0.10):
    """Offline onset detection on one trial trace. Returns index or None.
    Gates: |p - p0| > disp AND rolling-mean velocity over vwin seconds in
    the ramp direction > vsus. Direction taken from the tau ramp sign."""
    if len(t) < 10:
        return None
    sign = 1.0 if tau[-1] >= 0 else -1.0
    p0 = p[0]
    n = max(3, int(vwin / max(np.median(np.diff(t)), 1e-4)))
    vm = np.convolve(v * sign, np.ones(n) / n, mode='same')
    ok = (np.abs(p - p0) > disp) & (vm > vsus)
    idx = np.flatnonzero(ok)
    return int(idx[0]) if len(idx) else None


def analyze(path, disp=None, vsus=None):
    d = np.load(path)
    motor = int(d['motor'])
    disp = disp if disp is not None else PROFILES[motor][1]
    vsus = vsus if vsus is not None else 0.15
    breaks = {+1: [], -1: []}
    n = int(d['n_trials'])
    for k in range(n):
        t, p, v = d[f'n{k}_t'], d[f'n{k}_p'], d[f'n{k}_v']
        tau = d[f'n{k}_tau']
        i = detect_onset(t, p, v, tau, disp, vsus)
        s = +1 if tau[-1] >= 0 else -1
        if i is not None:
            breaks[s].append(tau[i])
            print(f"  trial {k:2d} dir {s:+d}: tau_break {tau[i]:+.3f} Nm "
                  f"at t={t[i]:.2f}s")
        else:
            print(f"  trial {k:2d} dir {s:+d}: NO onset (raise --taumax "
                  f"or lower thresholds)")
    report(motor, breaks)


def report(motor, breaks):
    print(f"\n[brk] motor {motor} summary (MOTOR-SPACE, commanded tau):")
    med = {}
    for s in (+1, -1):
        if breaks[s]:
            a = np.asarray(breaks[s])
            med[s] = float(np.median(a))
            print(f"  dir {s:+d}: median {med[s]:+.3f} Nm  "
                  f"(mean {a.mean():+.3f} +- {a.std():.3f}, n={len(a)})")
    if +1 in med and -1 in med:
        cs = 0.5 * (med[+1] - med[-1])
        tg = -0.5 * (med[+1] + med[-1])
        print(f"  C_s (static friction) = {cs:.3f} Nm motor-side")
        print(f"  tau_g bias at home    = {tg:+.3f} Nm "
              f"(expect ~0 at hanging equilibrium)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--analyze', default=None, help='npz to re-threshold')
    ap.add_argument('--motor', type=int, default=101)
    ap.add_argument('--hold', default='',
                    help='comma-sep motor ids to PD-hold (e.g. knee while '
                         'swinging hip); others stay disabled')
    ap.add_argument('--trials', type=int, default=10, help='per direction')
    ap.add_argument('--taumax', type=float, default=None)
    ap.add_argument('--ramp-time', type=float, default=6.0)
    ap.add_argument('--disp', type=float, default=None)
    ap.add_argument('--vsus', type=float, default=0.15)
    ap.add_argument('--rest', type=float, default=1.5)
    ap.add_argument('--temp-max', type=float, default=60.0)
    ap.add_argument('--tag', default='')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if args.analyze:
        analyze(args.analyze, args.disp, args.vsus)
        return

    prof = PROFILES[args.motor]
    taumax = args.taumax if args.taumax is not None else prof[0]
    disp = args.disp if args.disp is not None else prof[1]
    travel_hard = prof[2]
    holds = [int(x) for x in args.hold.split(',') if x]

    drv = make_driver(motor_ids=[args.motor] + holds)
    store, breaks = {}, {+1: [], -1: []}
    n_trial = 0
    home = hold_pos = None

    def send(p, v, kp, kd, tff):
        cmds = [(args.motor, p, v, kp, kd, tff)]
        for m in holds:
            cmds.append((m, hold_pos[m], 0.0, OTHER_KP, OTHER_KD, 0.0))
        drv.send_batch(cmds)

    def phase_hold(dur, kp=HOLD_KP, kd=HOLD_KD):
        t_end = time.monotonic() + dur
        while time.monotonic() < t_end:
            send(home, 0.0, kp, kd, 0.0)
            time.sleep(LOOP_DT)

    try:
        drv.enable_all()
        for _ in range(5):
            send_all = [(m, 0.0, 0.0, 0.0, 0.0, 0.0)
                        for m in [args.motor] + holds]
            drv.send_batch(send_all)
            time.sleep(0.01)
        home = drv.get_state(args.motor)[0]
        hold_pos = {m: drv.get_state(m)[0] for m in holds}
        print(f"[brk] motor {args.motor} home {home:+.3f} rad, "
              f"holding {holds or 'none'}, ramp 0->{taumax} Nm "
              f"over {args.ramp_time} s x {args.trials}/dir")

        order = [+1, -1] * args.trials
        for sign in order:
            phase_hold(0.8)
            temp = drv.get_state(args.motor)[3]
            if temp > args.temp_max:
                print(f"[brk] ABORT: temp {temp:.1f} C")
                break
            ts, ps, vs, taus, tfs = [], [], [], [], []
            t0 = time.monotonic()
            onset_tau = None
            vwin_n = max(3, int(0.10 / LOOP_DT))
            while True:
                t = time.monotonic() - t0
                if t > args.ramp_time:
                    break
                tau = sign * taumax * (t / args.ramp_time)
                send(0.0, 0.0, 0.0, RAMP_KD, tau)
                pos, vel, tfb, _ = drv.get_state(args.motor)
                ts.append(t); ps.append(pos); vs.append(vel)
                taus.append(tau); tfs.append(tfb)
                if abs(pos - home) > travel_hard:
                    break
                if len(ts) > vwin_n and abs(pos - ps[0]) > disp:
                    vm = np.mean(np.asarray(vs[-vwin_n:])) * sign
                    if vm > args.vsus:
                        onset_tau = tau
                        break
                time.sleep(LOOP_DT)
            # brake back to home
            t_end = time.monotonic() + 1.0
            while time.monotonic() < t_end:
                send(home, 0.0, BRAKE_KP, BRAKE_KD, 0.0)
                time.sleep(LOOP_DT)

            key = f"n{n_trial}"
            store[key + '_t'] = np.asarray(ts)
            store[key + '_p'] = np.asarray(ps)
            store[key + '_v'] = np.asarray(vs)
            store[key + '_tau'] = np.asarray(taus)
            store[key + '_tfb'] = np.asarray(tfs)   # telemetry, NOT truth
            n_trial += 1
            if onset_tau is not None:
                breaks[sign].append(onset_tau)
                print(f"  dir {sign:+d}: tau_break {onset_tau:+.3f} Nm  "
                      f"T={temp:.0f}C")
            else:
                print(f"  dir {sign:+d}: NO breakaway to {taumax} Nm "
                      f"(peak disp {max(abs(np.asarray(ps)-home)):.3f})")
            time.sleep(args.rest)
    except KeyboardInterrupt:
        print("\n[brk] stopped")
    finally:
        try:
            t_end = time.monotonic() + 1.0
            while time.monotonic() < t_end:
                send(home if home is not None else 0.0, 0.0,
                     HOLD_KP, HOLD_KD, 0.0)
                time.sleep(LOOP_DT)
            drv.disable_all()
        except Exception:
            pass

    report(args.motor, breaks)
    if args.out and store:
        np.savez(args.out, motor=args.motor, home=home,
                 n_trials=n_trial, tag=args.tag, **store)
        print(f"[brk] saved {args.out}")


if __name__ == '__main__':
    main()
