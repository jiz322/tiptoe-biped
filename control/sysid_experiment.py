"""
sysid_experiment.py — Automated system-identification for sim2real gap closure.

Hangs the robot, excites one joint at a time with known inputs (position
steps, frequency chirps, triangle ramps), and records the full 6-joint
response.  Output is a single .npz consumed by sysid_fit.py.

Assumptions:
    - Robot is already zeroed and hanging.
    - Motors have been enabled at least once (driver is reachable).

Usage:
    sudo python3 sysid_experiment.py          # run all trials
    sudo python3 sysid_experiment.py --leg 2  # test leg 2 instead

Controls during run:
    SPACE   start the trial sequence
    s       skip the current trial
    q       abort and save whatever was recorded so far
"""

import math
import os
import sys
import time
import select
import termios
import tty
import argparse
import json
import numpy as np

from motor_bridge import MotorBridge
import config as C

# ════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════
CONTROL_HZ = 100
DT = 1.0 / CONTROL_HZ
DEFAULT_POS = np.array(C.DEFAULT_JOINT_POS, dtype=np.float32)

# Joint type for each of the 6 indices
JTYPE = ['hip_roll', 'hip_pitch', 'knee',
         'hip_roll', 'hip_pitch', 'knee']

# Excite gains — DCMotor Kp/Kd the policy trains against
EXCITE = {
    'hip_roll':  (6.0,  0.3),
    'hip_pitch': (8.0,  0.5),
    'knee':      (8.0,  0.5),
}

# Hold gains — 2.5× excite, keeps passive joints rigid
HOLD = {
    'hip_roll':  (15.0, 0.8),
    'hip_pitch': (20.0, 1.2),
    'knee':      (20.0, 1.2),
}

# Generous effort limits (allow full transients, firmware current-limits
# are the last backstop)
EFFORT_LIM = np.array([6.0, 17.0, 23.2, 6.0, 17.0, 23.2], dtype=np.float32)

# Position limits (rad from default). Hip-roll is mechanically tighter;
# hip-pitch and knee have ~2× the cfg range per user report.
POS_LIM = np.array([0.6, 1.5, 1.5, 0.6, 1.5, 1.5], dtype=np.float32)

TEMP_WARN  = 50.0   # °C — pause and cool
TEMP_ABORT = 65.0   # °C — stop the experiment

# ════════════════════════════════════════════════════════════════
# Signal generators
# ════════════════════════════════════════════════════════════════

def sig_step(t, amp, **_):
    return amp


def sig_chirp(t, amp, f0, f1, dur, **_):
    # Exponential (log) chirp: equal time per octave, so most of the sweep sits
    # near the ~2 Hz actuator corner where omega/zeta live (a linear chirp spends
    # ~85% of its time above it, where the response is attenuated = low SNR on
    # omega/zeta). Requires f0 > 0.
    K = (f1 / f0) ** (t / dur)
    phase = 2.0 * math.pi * f0 * dur / math.log(f1 / f0) * (K - 1.0)
    return amp * math.sin(phase)


def sig_triangle(t, vel, lim, **_):
    # Triangle wave bouncing between ±lim at ±vel
    period = 4.0 * lim / abs(vel)
    p = (t % period) / period          # normalised phase 0-1
    if p < 0.25:
        return 4.0 * lim * p
    elif p < 0.75:
        return lim - 4.0 * lim * (p - 0.25)
    else:
        return -lim + 4.0 * lim * (p - 0.75)


_GENERATORS = {'step': sig_step, 'chirp': sig_chirp, 'triangle': sig_triangle}


def get_signal(trial, t):
    return _GENERATORS[trial['signal']](t, **trial)

def cv_ramps(vels, pos_lim, plateau=0.6, periods=1.5, dur_max=12.0):
    """(vel, lim, dur) triples for a constant-velocity friction sweep. `lim`
    grows with speed so every ramp holds steady velocity for ~`plateau` s before
    reversing, while staying inside +/- pos_lim — so the + and - passes cover the
    same positions and gravity cancels in the friction fit."""
    out = []
    for v in vels:
        lim    = min(0.85 * pos_lim, max(0.15, v * plateau))
        period = 4.0 * lim / v
        dur    = min(periods * period, dur_max)
        out.append((v, round(lim, 3), round(dur, 2)))
    return out
# ════════════════════════════════════════════════════════════════
# Trial list builder
# ════════════════════════════════════════════════════════════════

def build_trials(leg=1):
    """Return list of trial dicts for the chosen leg (1 or 2)."""
    off = 0 if leg == 1 else 3      # index offset
    hr, hp, kn = 0 + off, 1 + off, 2 + off
    hp_coupled = hp                  # hip-pitch coupled to this knee

    trials = []

    # ── Hip roll ──────────────────────────────────────────────
    for a in [0.2, -0.2, 0.4, -0.4]:
        trials.append(dict(name=f'hr_step_{a:+.1f}', idx=hr,
                           signal='step', amp=a, dur=2.5))
    trials.append(dict(name='hr_chirp', idx=hr,
                       signal='chirp', amp=0.2, f0=0.5, f1=12.0, dur=15.0))
    for v, lim, dur in cv_ramps([0.1, 0.25, 0.5, 1.0], POS_LIM[hr]):
            trials.append(dict(name=f'hr_cv_{v:g}'.replace('.', 'p'), idx=hr,
                            signal='triangle', vel=v, lim=lim, dur=dur))

    # ── Hip pitch ─────────────────────────────────────────────
    for a in [0.3, -0.3, 0.6, -0.6]:
        trials.append(dict(name=f'hp_step_{a:+.1f}', idx=hp,
                           signal='step', amp=a, dur=2.5))
    trials.append(dict(name='hp_chirp', idx=hp,
                       signal='chirp', amp=0.2, f0=0.5, f1=12.0, dur=15.0))
    for v, lim, dur in cv_ramps([0.1, 0.25, 0.5, 1.0, 2.0, 3.0], POS_LIM[hp]):
            trials.append(dict(name=f'hp_cv_{v:g}'.replace('.', 'p'), idx=hp,
                            signal='triangle', vel=v, lim=lim, dur=dur))

    # ── Knee, hip-pitch held stiff ────────────────────────────
    for a in [0.3, -0.3, 0.6, -0.6]:
        trials.append(dict(name=f'kn_stiff_step_{a:+.1f}', idx=kn,
                           signal='step', amp=a, dur=2.5))
    trials.append(dict(name='kn_stiff_chirp', idx=kn,
                       signal='chirp', amp=0.2, f0=0.5, f1=12.0, dur=15.0))
    for v, lim, dur in cv_ramps([0.1, 0.25, 0.5, 1.0, 2.0, 3.0], POS_LIM[kn]):
            trials.append(dict(name=f'kn_stiff_cv_{v:g}'.replace('.', 'p'), idx=kn,
                            signal='triangle', vel=v, lim=lim, dur=dur))

    # ── Knee, hip-pitch floating ──────────────────────────────
    for a in [0.3, -0.3]:
        trials.append(dict(name=f'kn_float_step_{a:+.1f}', idx=kn,
                           signal='step', amp=a, dur=2.5,
                           hip_float=hp_coupled))
    trials.append(dict(name='kn_float_chirp', idx=kn,
                       signal='chirp', amp=0.2, f0=0.5, f1=12.0, dur=15.0,
                       hip_float=hp_coupled))
    trials.append(dict(name='kn_float_tri_1', idx=kn,
                       signal='triangle', vel=0.5, lim=0.8, dur=4.0,
                       hip_float=hp_coupled))

    return trials


# ════════════════════════════════════════════════════════════════
# Keyboard helpers (same pattern as balance_log.py)
# ════════════════════════════════════════════════════════════════

def setup_raw_keys():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    return fd, old

def restore_keys(fd, old):
    termios.tcsetattr(fd, termios.TCSADRAIN, old)

def get_key():
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


# ════════════════════════════════════════════════════════════════
# Control-loop helpers
# ════════════════════════════════════════════════════════════════

def build_gains(trial):
    """Return (kp, kd) 6-vectors: excite the target joint, hold the rest."""
    kp = np.zeros(6, dtype=np.float32)
    kd = np.zeros(6, dtype=np.float32)
    idx = trial['idx']
    for i in range(6):
        jt = JTYPE[i]
        if i == idx:
            kp[i], kd[i] = EXCITE[jt]
        else:
            kp[i], kd[i] = HOLD[jt]
    # Knee-float: let the coupled hip-pitch go limp
    hf = trial.get('hip_float')
    if hf is not None:
        kp[hf] = 0.0
        kd[hf] = 0.3
    return kp, kd


def hold_at(bridge, target, kp, kd, duration):
    """Hold position for duration seconds (blocking)."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < duration:
        tt = time.monotonic()
        bridge.read_state()
        bridge.send_target(target, kp, kd, effort_limit=EFFORT_LIM)
        rem = DT - (time.monotonic() - tt)
        if rem > 0:
            time.sleep(rem)


def smooth_return(bridge, kp, kd, duration=1.0):
    """Cosine-blend from current position back to DEFAULT_POS."""
    start = bridge.sim_joint_pos.copy()
    t0 = time.monotonic()
    while time.monotonic() - t0 < duration:
        tt = time.monotonic()
        a = 0.5 * (1.0 - math.cos(math.pi * min((tt - t0) / duration, 1.0)))
        target = start * (1.0 - a) + DEFAULT_POS * a
        bridge.read_state()
        bridge.send_target(target, kp, kd, effort_limit=EFFORT_LIM)
        rem = DT - (time.monotonic() - tt)
        if rem > 0:
            time.sleep(rem)


def record_trial(bridge, trial, kp, kd):
    """Execute the trial signal and record data.

    Returns dict of arrays, or None on user abort ('q').
    """
    idx = trial['idx']
    duration = trial['dur']
    buf = {k: [] for k in ('time', 'p_des', 'pos', 'vel', 'torque', 'temp')}

    t0 = time.monotonic()
    while True:
        tt = time.monotonic()
        t = tt - t0
        if t >= duration:
            break

        k = get_key()
        if k in ('q', '\x03'):
            return None
        if k == 's':
            break

        bridge.read_state()

        p_des = DEFAULT_POS.copy()
        p_des[idx] = np.clip(
            DEFAULT_POS[idx] + get_signal(trial, t),
            DEFAULT_POS[idx] - POS_LIM[idx],
            DEFAULT_POS[idx] + POS_LIM[idx],
        )
        bridge.send_target(p_des, kp, kd, effort_limit=EFFORT_LIM)

        buf['time'].append(t)
        buf['p_des'].append(p_des.copy())
        buf['pos'].append(bridge.sim_joint_pos.copy())
        buf['vel'].append(bridge.sim_joint_vel.copy())
        buf['torque'].append(bridge.sim_joint_torque.copy())
        buf['temp'].append(bridge.motor_temperature.copy())

        rem = DT - (time.monotonic() - tt)
        if rem > 0:
            time.sleep(rem)

    return {k: np.array(v) for k, v in buf.items()}


# ════════════════════════════════════════════════════════════════
# Save
# ════════════════════════════════════════════════════════════════

def save_all(results, leg):
    stamp = time.strftime('%Y%m%d_%H%M%S')
    fname = f'sysid_leg{leg}_{stamp}.npz'

    arrays = {}
    names = []
    configs = []
    for name, rec in results.items():
        names.append(name)
        configs.append({k: v.item() if isinstance(v, np.generic) else v
                        for k, v in rec['trial'].items()
                        if not isinstance(v, np.ndarray)})
        for ch in ('time', 'p_des', 'pos', 'vel', 'torque', 'temp', 'kp', 'kd'):
            arrays[f'{name}__{ch}'] = rec[ch]

    arrays['trial_names'] = np.array(names)
    arrays['trial_configs'] = np.array(json.dumps(configs))
    arrays['control_hz'] = np.array(CONTROL_HZ)
    arrays['default_pos'] = DEFAULT_POS
    arrays['knee_hip_coupling'] = np.array(C.KNEE_HIP_COUPLING)
    arrays['knee_gear_ratio'] = np.array(C.KNEE_GEAR_RATIO)

    np.savez(fname, **arrays)
    return fname


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--leg', type=int, default=1, choices=[1, 2])
    args = ap.parse_args()

    trials = build_trials(leg=args.leg)
    total_dur = sum(t['dur'] + 3.5 for t in trials)  # +settle+return
    print(f"=== sysid_experiment.py — leg {args.leg}, "
          f"{len(trials)} trials, ~{total_dur:.0f}s ===\n")

    bridge = MotorBridge(action_lpf_alpha=0.0)    # LPF off for clean data
    bridge.enable()
    time.sleep(0.3)
    for _ in range(3):
        bridge.send_idle()
        time.sleep(0.02)
    bridge.read_state()

    print("Initial joint positions (sim frame):")
    for i in range(6):
        print(f"  [{i}] {bridge.sim_joint_pos[i]:+7.3f}  "
              f"(default {DEFAULT_POS[i]:+7.3f})")

    fd, old_term = setup_raw_keys()
    print("\nPress SPACE to start, 'q' to quit.")

    try:
        # Wait for start key
        while True:
            k = get_key()
            if k == ' ':
                break
            if k in ('q', '\x03'):
                raise KeyboardInterrupt
            time.sleep(0.05)

        results = {}
        for ti, trial in enumerate(trials):
            name = trial['name']
            idx  = trial['idx']
            print(f"\n[{ti+1}/{len(trials)}] {name}  "
                  f"(joint {idx} {JTYPE[idx]}, {trial['signal']}, "
                  f"{trial['dur']:.1f}s)")

            kp, kd = build_gains(trial)

            # Settle at default
            hold_at(bridge, DEFAULT_POS, kp, kd, duration=1.5)

            # Record
            rec = record_trial(bridge, trial, kp, kd)
            if rec is None:                # user pressed 'q'
                print("  Aborted.")
                break

            rec['kp'] = kp
            rec['kd'] = kd
            rec['trial'] = trial
            results[name] = rec
            print(f"  {len(rec['time'])} ticks recorded.")

            # Return to default
            smooth_return(bridge, kp, kd, duration=1.0)

            # Temperature check
            bridge.read_state()
            tmax = bridge.motor_temperature.max()
            if tmax > TEMP_ABORT:
                print(f"  ABORT — motor temperature {tmax:.1f}°C!")
                break
            if tmax > TEMP_WARN:
                print(f"  Cooling (max temp {tmax:.1f}°C) ...")
                hold_at(bridge, DEFAULT_POS, kp, kd, duration=10.0)

        # Save
        if results:
            fname = save_all(results, args.leg)
            print(f"\nSaved {len(results)} trials → {fname}")
        else:
            print("\nNo trials recorded.")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        restore_keys(fd, old_term)
        for _ in range(3):
            bridge.send_idle()
            time.sleep(0.02)
        bridge.disable()
        bridge.close()
        print("Motors disabled. Done.")


if __name__ == '__main__':
    main()
