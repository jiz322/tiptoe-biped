#!/usr/bin/env python3
"""
coast_down_bench.py — Replant P5 friction/inertia capture. Modes:

ballistic (DEFAULT): reproduces the M3.1-proven tether-safe envelope
EXACTLY — short tau_ff bursts (<=0.12 s), STRICTLY ALTERNATING
directions, tight travel abort, immediate PD pull-back brake, settle
between reps. The brake phase is logged too: the firmware PD law is
known (kp*(home-p) - kd*v, clipped at limit_torque), so the fit
reconstructs commanded torque through the deceleration/reversal — where
much of the C and b information lives. Never applies sustained torque.

pingpong: constant tau flipping sign inside a window. ONLY for a rigid
external fixture — on the tether the base swings and the window bounds
lie (2026-08-17: hard thigh/torso collisions).

coast: burst + limp ringdown. Only for lightly damped joints with a
clear arc; hip_pitch is overdamped (b~0.22 vs mgl~0.5, zeta>=1) and
cannot ring.

TETHER LESSON (learned the hard way, twice): the hanging torso is a
free-floating base — sustained torque swings it, relative-angle bounds
stop meaning anything, and the leg finds the torso at speed. The only
validated tether envelope is M3.1's: alternate short bursts, small
travel, brake home immediately. The stall guard (all modes) ends any
drive that stops moving within ~0.5 s — never press into contact.

All modes fit with data/coastdown_fit.py (global regression over all
samples with known/reconstructable commanded tau) — the third,
independent inertia method (vs ballistic initial-slope and kp-sweep).

Telemetry ground rules respected: commanded tau only (delivery 1:1
verified by M3.1 to 17 Nm); torque_fdb stored but labeled telemetry.
All raw MOTOR-SPACE; conversions happen in the fit.

Usage:
    python3 coast_down_bench.py --motor 101 --hold 102 --tag hang --out bal_101.npz
    python3 coast_down_bench.py --motor 102 --hold 101 --tag belt_on --out bal_102_on.npz
"""

import argparse
import time

import numpy as np

import config as C
from robstride_socketcan_driver import make_driver

# per-motor defaults:
#   levels: burst tau_ff magnitudes (Nm motor-side)
#   vtarget: end-of-burst speed (rad/s motor-side)
#   vmax: abort speed; travel_hard: abort |p - home| (rad)
#   coast_time: limp logging window (s)
PROFILES = {
    101: dict(levels=[2.0, 3.0, 4.0], vtarget=10.0, vmax=20.0,
              travel_hard=1.2, coast_time=6.0),
    104: dict(levels=[2.0, 3.0, 4.0], vtarget=10.0, vmax=20.0,
              travel_hard=1.2, coast_time=6.0),
    102: dict(levels=[2.0, 3.0, 4.0], vtarget=10.0, vmax=20.0,
              travel_hard=1.2, coast_time=6.0),
    105: dict(levels=[2.0, 3.0, 4.0], vtarget=10.0, vmax=20.0,
              travel_hard=1.2, coast_time=6.0),
    103: dict(levels=[0.8, 1.2, 1.6], vtarget=3.0, vmax=6.0,
              travel_hard=0.30, coast_time=4.0),
    106: dict(levels=[0.8, 1.2, 1.6], vtarget=3.0, vmax=6.0,
              travel_hard=0.30, coast_time=4.0),
}

# ping-pong mode defaults (2026-08-17 session 3: hip_pitch is OVERDAMPED —
# b~0.22 Nm s/rad vs mgl~0.5 Nm/rad, zeta>=1 — a free ringdown parks after
# half a swing and the pendulum coast is uninformative; and the wide coast
# window reached the torso. Ping-pong keeps a small window and known tau
# at every sample: steady legs give C/b/gravity, flip transients give I.)
#   window: flip bounds +-rad rel home; hard abort at 1.5x window
PROFILES_PP = {
    101: dict(levels=[0.5, 0.8, 1.2, 1.6], window=0.50, vmax=12.0),
    104: dict(levels=[0.5, 0.8, 1.2, 1.6], window=0.50, vmax=12.0),
    102: dict(levels=[0.4, 0.7, 1.0, 1.4], window=0.50, vmax=12.0),
    105: dict(levels=[0.4, 0.7, 1.0, 1.4], window=0.50, vmax=12.0),
    103: dict(levels=[0.25, 0.40, 0.60], window=0.18, vmax=5.0),
    106: dict(levels=[0.25, 0.40, 0.60], window=0.18, vmax=5.0),
}
PP_TRAVERSALS = 6          # direction flips per level
PP_TIMEOUT_S = 15.0        # per level
PP_STALL_V = 0.15          # rad/s: below this while driving = stalled
PP_STALL_T = 0.35          # s of stall before reacting
PP_FLIP_GRACE = 0.5        # s after a flip before stall detection arms

LOOP_DT = 0.0025          # ~400 Hz command/log rate
OTHER_KP, OTHER_KD = 15.0, 1.0
HOLD_KP, HOLD_KD = 8.0, 0.8
BRAKE_KP, BRAKE_KD = 6.0, 1.5
BURST_TMAX = 0.6          # s, burst give-up time
# ballistic mode defaults — the M3.1-proven tether envelope (M3.1:
# travel 0.50 / window 0.10 s / alternate dirs). Low levels are the
# information carriers for the fit (slow bursts = many samples, wide
# v coverage, breaks the tau~sign(v) collinearity); high levels pin the
# ceiling. Bursts end at vtarget (soft) or travel or burst time.
PROFILES_BAL = {
    101: dict(levels=[1.0, 1.5, 2.0, 3.0, 4.0], travel=0.45, vmax=12.0,
              vtarget=10.0, burst=0.30),
    104: dict(levels=[1.0, 1.5, 2.0, 3.0, 4.0], travel=0.45, vmax=12.0,
              vtarget=10.0, burst=0.30),
    102: dict(levels=[1.0, 1.5, 2.0, 3.0, 4.0], travel=0.30, vmax=10.0,
              vtarget=8.0, burst=0.30),
    105: dict(levels=[1.0, 1.5, 2.0, 3.0, 4.0], travel=0.30, vmax=10.0,
              vtarget=8.0, burst=0.30),
    103: dict(levels=[0.4, 0.6, 0.9, 1.2], travel=0.18, vmax=5.0,
              vtarget=4.0, burst=0.30),
    106: dict(levels=[0.4, 0.6, 0.9, 1.2], travel=0.18, vmax=5.0,
              vtarget=4.0, burst=0.30),
}
BAL_BRAKE_S = 1.0          # logged PD brake after each burst
BAL_STALL_V = 0.10         # rad/s: burst not moving by BAL_STALL_T
BAL_STALL_T = 0.06         # s -> below breakaway, end burst

PH_BURST, PH_COAST, PH_BRAKE = 1, 2, 3


class _Done(Exception):
    """Internal: pingpong branch finished, skip the coast loop."""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--motor', type=int, default=101)
    ap.add_argument('--mode', choices=('ballistic', 'coast', 'pingpong'),
                    default='ballistic',
                    help='ballistic (default): M3.1-style short alternating '
                         'bursts + logged PD brake — the only tether-safe '
                         'envelope. pingpong: rigid fixture only. coast: '
                         'lightly damped joints with a clear arc only.')
    ap.add_argument('--window', type=float, default=None,
                    help='pingpong flip bound, +-rad rel home')
    ap.add_argument('--traversals', type=int, default=PP_TRAVERSALS)
    ap.add_argument('--hold', default='',
                    help='comma-sep motor ids to PD-hold stiff (hold the '
                         'knee while swinging hip_pitch so the shank is '
                         'rigid; a flopping link breaks rigid-plant fits)')
    ap.add_argument('--hold-kp', type=float, default=OTHER_KP,
                    help='PD kp for the held motors (default 15). P7 note: '
                         'the knee bench at kp15 lets the held hip yield '
                         '~0.73*tau/15 rad through the belt; rerunning at '
                         '40 discriminates hip-yield coupling from true '
                         'rotor inertia in the apparent knee I.')
    ap.add_argument('--hold-kd', type=float, default=OTHER_KD)
    ap.add_argument('--reps', type=int, default=4,
                    help='per level, STRICTLY alternating direction')
    ap.add_argument('--levels', default=None,
                    help='comma-sep Nm, overrides profile')
    ap.add_argument('--vtarget', type=float, default=None)
    ap.add_argument('--vmax', type=float, default=None)
    ap.add_argument('--travel', type=float, default=None,
                    help='hard |p-home| abort (rad)')
    ap.add_argument('--coast-time', type=float, default=None)
    ap.add_argument('--rest', type=float, default=1.5)
    ap.add_argument('--temp-max', type=float, default=60.0)
    ap.add_argument('--tag', default='')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if args.mode == 'ballistic':
        prof = PROFILES_BAL[args.motor]
        travel = args.travel or prof['travel']
        vmax = args.vmax or prof['vmax']
        vtarget = args.vtarget or prof['vtarget']
        burst_t = prof['burst']
        window = coast_time = None
    elif args.mode == 'pingpong':
        prof = PROFILES_PP[args.motor]
        window = args.window or prof['window']
        vmax = args.vmax or prof['vmax']
        travel = args.travel or 1.5 * window
        vtarget = coast_time = None
    else:
        prof = PROFILES[args.motor]
        window = None
        vtarget = args.vtarget or prof['vtarget']
        vmax = args.vmax or prof['vmax']
        travel = args.travel or prof['travel_hard']
        coast_time = args.coast_time or prof['coast_time']
    levels = ([float(x) for x in args.levels.split(',')]
              if args.levels else prof['levels'])
    holds = [int(x) for x in args.hold.split(',') if x]

    drv = make_driver(motor_ids=[args.motor] + holds)
    store = {}
    n_rep = 0
    home = hold_pos = None

    def send(p, v, kp, kd, tff):
        cmds = [(args.motor, p, v, kp, kd, tff)]
        for m in holds:
            cmds.append((m, hold_pos[m], 0.0, args.hold_kp, args.hold_kd, 0.0))
        drv.send_batch(cmds)

    def phase_hold(dur, kp=HOLD_KP, kd=HOLD_KD):
        t_end = time.monotonic() + dur
        while time.monotonic() < t_end:
            send(home, 0.0, kp, kd, 0.0)
            time.sleep(LOOP_DT)

    try:
        drv.enable_all()
        for _ in range(5):
            drv.send_batch([(m, 0.0, 0.0, 0.0, 0.0, 0.0)
                            for m in [args.motor] + holds])
            time.sleep(0.01)
        home = drv.get_state(args.motor)[0]
        hold_pos = {m: drv.get_state(m)[0] for m in holds}
        print(f"[coast] motor {args.motor} home {home:+.3f} rad, "
              f"holding {holds or 'none'}, mode {args.mode}")

        if args.mode == 'ballistic':
            print(f"[coast] BALLISTIC: levels {levels} Nm x {args.reps} "
                  f"(strict alternate; reps 3-4 start OFFSET and burst "
                  f"through home — breaks the v~p degeneracy that "
                  f"aliases viscous b into the gravity slope), travel "
                  f"+-{travel}, burst <={burst_t}s, brake {BAL_BRAKE_S}s "
                  f"logged")
            for level in levels:
                for rep in range(args.reps):
                    sign = 1 if rep % 2 == 0 else -1   # ALTERNATE — the
                    # M3.1-validated pattern; same-direction repeats build
                    # base swing on the tether. Not configurable.
                    start_off = 0.0 if rep < 2 else -sign * 0.55 * travel
                    phase_hold(0.8)
                    p_now = drv.get_state(args.motor)[0]
                    if abs(p_now - home) > 0.10:
                        print(f"  [warn] settled {p_now-home:+.3f} rad "
                              f"off home before rep — base/stiction "
                              f"drift; continuing from home")
                    temp = drv.get_state(args.motor)[3]
                    if temp > args.temp_max:
                        print(f"[coast] ABORT: temp {temp:.1f} C")
                        raise KeyboardInterrupt
                    if start_off:
                        # PD-ramp to the offset start, then settle
                        p0 = drv.get_state(args.motor)[0]
                        tgt = home + start_off
                        t0m = time.monotonic()
                        while time.monotonic() - t0m < 0.6:
                            a = min(1.0, (time.monotonic() - t0m) / 0.6)
                            send(p0 + (tgt - p0) * a, 0.0,
                                 HOLD_KP, HOLD_KD, 0.0)
                            time.sleep(LOOP_DT)
                        t0m = time.monotonic()
                        while time.monotonic() - t0m < 0.4:
                            send(tgt, 0.0, HOLD_KP, HOLD_KD, 0.0)
                            time.sleep(LOOP_DT)
                    ts, ps, vs, taus, tfs, phs = [], [], [], [], [], []
                    aborted = ''
                    t0 = time.monotonic()
                    # ── burst (short, known tau) ──
                    while True:
                        t = time.monotonic() - t0
                        if t > burst_t:
                            break
                        tau = sign * level
                        send(0.0, 0.0, 0.0, 0.0, tau)
                        pos, vel, tfb, _ = drv.get_state(args.motor)
                        ts.append(t); ps.append(pos); vs.append(vel)
                        taus.append(tau); tfs.append(tfb)
                        phs.append(PH_BURST)
                        if abs(pos - home) > travel:
                            break
                        if abs(vel) >= vtarget:
                            break
                        if abs(vel) > vmax:
                            aborted = 'vmax'
                            break
                        if t > BAL_STALL_T and abs(vel) < BAL_STALL_V:
                            aborted = 'below breakaway'
                            break
                        time.sleep(LOOP_DT)
                    # ── brake, LOGGED (fit reconstructs PD torque) ──
                    t1 = time.monotonic()
                    while time.monotonic() - t1 < BAL_BRAKE_S:
                        t = time.monotonic() - t0
                        send(home, 0.0, BRAKE_KP, BRAKE_KD, 0.0)
                        pos, vel, tfb, _ = drv.get_state(args.motor)
                        ts.append(t); ps.append(pos); vs.append(vel)
                        taus.append(0.0); tfs.append(tfb)
                        phs.append(PH_BRAKE)
                        time.sleep(LOOP_DT)
                    key = f"n{n_rep}"
                    store[key + '_t'] = np.asarray(ts)
                    store[key + '_p'] = np.asarray(ps)
                    store[key + '_v'] = np.asarray(vs)
                    store[key + '_tau'] = np.asarray(taus)
                    store[key + '_tfb'] = np.asarray(tfs)
                    store[key + '_ph'] = np.asarray(phs)
                    store[key + '_meta'] = np.array([level, sign, temp,
                                                    start_off])
                    n_rep += 1
                    pha = np.asarray(phs)
                    nb = int((pha == PH_BURST).sum())
                    vpk = (float(np.max(np.abs(np.asarray(vs))))
                           if vs else 0.0)
                    trv = (float(np.max(np.abs(np.asarray(ps) - home)))
                           if ps else 0.0)
                    print(f"  {level:4.1f} Nm dir {sign:+d} "
                          f"from {start_off:+.2f}: burst n={nb} "
                          f"brake n={len(ts)-nb}, vpk {vpk:5.1f}, travel "
                          f"{trv:.2f} {aborted}  T={temp:.0f}C")
                    time.sleep(args.rest)
            raise _Done

        if args.mode == 'pingpong':
            print(f"[coast] levels {levels} Nm, window +-{window:.2f} rad, "
                  f"{args.traversals} traversals/level, hard +-{travel:.2f}")
            for level in levels:
                phase_hold(1.0)
                temp = drv.get_state(args.motor)[3]
                if temp > args.temp_max:
                    print(f"[coast] ABORT: temp {temp:.1f} C")
                    raise KeyboardInterrupt
                ts, ps, vs, taus, tfs, phs = [], [], [], [], [], []
                aborted = ''
                tau_now, flips, stalls = level, 0, 0
                t0 = time.monotonic()
                t_flip = t0
                stall_since = None
                while flips < args.traversals:
                    now = time.monotonic()
                    t = now - t0
                    if t > PP_TIMEOUT_S:
                        aborted = 'timeout'
                        break
                    send(0.0, 0.0, 0.0, 0.0, tau_now)
                    pos, vel, tfb, _ = drv.get_state(args.motor)
                    ts.append(t); ps.append(pos); vs.append(vel)
                    taus.append(tau_now); tfs.append(tfb)
                    phs.append(PH_BURST)
                    rel = pos - home
                    if tau_now > 0 and rel > window:
                        tau_now, flips = -level, flips + 1
                        t_flip, stall_since = now, None
                    elif tau_now < 0 and rel < -window:
                        tau_now, flips = level, flips + 1
                        t_flip, stall_since = now, None
                    # stall guard: driving but not moving = pressing on an
                    # obstacle (or below breakaway). NEVER keep pressing:
                    # flip once, end the level on the second stall.
                    if now - t_flip > PP_FLIP_GRACE:
                        if abs(vel) < PP_STALL_V:
                            if stall_since is None:
                                stall_since = now
                            elif now - stall_since > PP_STALL_T:
                                stalls += 1
                                if stalls >= 2:
                                    aborted = (f'stalled x{stalls} '
                                               '(obstacle or below '
                                               'breakaway) — level ended')
                                    break
                                tau_now = -tau_now
                                t_flip, stall_since = now, None
                        else:
                            stall_since = None
                    if abs(rel) > travel or abs(vel) > vmax:
                        aborted = 'guard'
                        break
                    time.sleep(LOOP_DT)
                t_end = time.monotonic() + 1.0
                while time.monotonic() < t_end:
                    send(home, 0.0, BRAKE_KP, BRAKE_KD, 0.0)
                    time.sleep(LOOP_DT)
                key = f"n{n_rep}"
                store[key + '_t'] = np.asarray(ts)
                store[key + '_p'] = np.asarray(ps)
                store[key + '_v'] = np.asarray(vs)
                store[key + '_tau'] = np.asarray(taus)
                store[key + '_tfb'] = np.asarray(tfs)
                store[key + '_ph'] = np.asarray(phs)
                store[key + '_meta'] = np.array([level, 0, temp])
                n_rep += 1
                vpk = float(np.max(np.abs(np.asarray(vs)))) if vs else 0.0
                print(f"  {level:4.2f} Nm: n={len(ts)}, flips {flips}, "
                      f"stalls {stalls}, vpk {vpk:5.1f} rad/s {aborted}  "
                      f"T={temp:.0f}C")
                time.sleep(args.rest)
            raise _Done

        print(f"[coast] levels {levels} Nm x {args.reps}, vtarget "
              f"{vtarget}, coast {coast_time}s, travel +-{travel}")

        for level in levels:
            for rep in range(args.reps):
                sign = 1 if rep % 2 == 0 else -1
                phase_hold(1.0)
                temp = drv.get_state(args.motor)[3]
                if temp > args.temp_max:
                    print(f"[coast] ABORT: temp {temp:.1f} C")
                    raise KeyboardInterrupt
                ts, ps, vs, taus, tfs, phs = [], [], [], [], [], []
                aborted = ''
                t0 = time.monotonic()
                # ── burst ──
                while True:
                    t = time.monotonic() - t0
                    if t > BURST_TMAX:
                        break
                    tau = sign * level
                    send(0.0, 0.0, 0.0, 0.0, tau)
                    pos, vel, tfb, _ = drv.get_state(args.motor)
                    ts.append(t); ps.append(pos); vs.append(vel)
                    taus.append(tau); tfs.append(tfb); phs.append(PH_BURST)
                    if abs(vel) >= vtarget:
                        break
                    if abs(pos - home) > 0.6 * travel:
                        break
                    time.sleep(LOOP_DT)
                # ── coast (fully limp, keep polling state) ──
                t1 = time.monotonic()
                while True:
                    t = time.monotonic() - t0
                    if time.monotonic() - t1 > coast_time:
                        break
                    send(0.0, 0.0, 0.0, 0.0, 0.0)
                    pos, vel, tfb, _ = drv.get_state(args.motor)
                    ts.append(t); ps.append(pos); vs.append(vel)
                    taus.append(0.0); tfs.append(tfb); phs.append(PH_COAST)
                    if abs(pos - home) > travel or abs(vel) > vmax:
                        aborted = 'guard'
                        break
                    time.sleep(LOOP_DT)
                # ── brake ──
                t_end = time.monotonic() + 1.0
                while time.monotonic() < t_end:
                    send(home, 0.0, BRAKE_KP, BRAKE_KD, 0.0)
                    time.sleep(LOOP_DT)

                key = f"n{n_rep}"
                store[key + '_t'] = np.asarray(ts)
                store[key + '_p'] = np.asarray(ps)
                store[key + '_v'] = np.asarray(vs)
                store[key + '_tau'] = np.asarray(taus)
                store[key + '_tfb'] = np.asarray(tfs)  # telemetry, NOT truth
                store[key + '_ph'] = np.asarray(phs)
                store[key + '_meta'] = np.array([level, sign, temp])
                n_rep += 1
                nb = int(np.sum(np.asarray(phs) == PH_BURST))
                vpk = float(np.max(np.abs(np.asarray(vs)))) if vs else 0.0
                print(f"  {level:4.1f} Nm dir {sign:+d}: burst n={nb}, "
                      f"coast n={len(ts)-nb}, vpk {vpk:5.1f} rad/s "
                      f"{aborted}  T={temp:.0f}C")
                time.sleep(args.rest)
    except _Done:
        pass
    except KeyboardInterrupt:
        print("\n[coast] stopped")
    finally:
        try:
            t_end = time.monotonic() + 1.2
            while time.monotonic() < t_end:
                send(home if home is not None else 0.0, 0.0,
                     HOLD_KP, HOLD_KD, 0.0)
                time.sleep(LOOP_DT)
            drv.disable_all()
        except Exception:
            pass

    if args.out and store:
        np.savez(args.out, motor=args.motor, home=home, n_reps=n_rep,
                 tag=args.tag, holds=np.asarray(holds), mode=args.mode,
                 hold_kp=args.hold_kp, hold_kd=args.hold_kd,
                 brake_kp=BRAKE_KP, brake_kd=BRAKE_KD, **store)
        print(f"[coast] saved {args.out}")


if __name__ == '__main__':
    main()
