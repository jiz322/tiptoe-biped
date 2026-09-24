"""
sysid_loaded.py — loaded, IMU-aware system-ID for the point-foot biped.
INTERACTIVE / HAND-ARMED version.

Companion to sysid_experiment.py. Puts the robot in a TWO-LEG KNEEL on the
ground and excites a joint *pair* symmetrically so the supporting joints carry
real body load AND the torso tilts (IMU sees real, command-correlated
projected_gravity / angular_velocity). Logs full joint state + IMU every tick.

Two blocks (hip-roll loaded omitted — the roll motor can't carry it):
    hip_pitch   knee held stiff (deep kneel), hip-pitch rocked fore/aft
    knee        hip-pitch held stiff,         knee flexed about its loaded point

═══ SAFETY MODEL (same as joint_tuner_mirror.py / balance.py) ═══
  * Motors start ENABLED but LIMP (Kp=Kd=0). The held target is pinned to the
    MEASURED pose, so ramping gains up holds where the robot already is — it
    never moves to get into posture, and there is no jump.
  * YOU place the robot in the kneel by hand and ramp the gains up, watching.
    The script only ever commands the small excitation signal during a trial.
  * SPACE / r = PANIC -> all gains 0 -> limp, any time.
  * A trial drops ONLY the excited pair's gain down to policy gains, via a
    smooth 1 s ramp (gentle load-sag, never slammed), sweeps around the joint's
    current position CLIPPED to the safe window, then ramps the gain back up.

Workflow
--------
  1. Place the robot roughly in the kneel on a mat.
  2. sudo python3 sysid_loaded.py [--block hip_pitch|knee]
  3. The position target starts at the predetermined block posture (knee deep,
     hip-pitch held, etc). Gains start at 0, so nothing moves yet. With the robot
     hand-placed near that pose, ramp gains up and it settles onto the designed
     operating point. Watch the 'err' column — large error × rising Kp = a yank,
     so reposition by hand or press 'k' (slow approach) before stiffening.
  4. Press 't' to run the next trial (sweeps around the designed center, clipped).
     's' skips. On a block change, lower gains and press 'k' to ease into the new
     posture.
  5. 'q' quits and saves.

Keys
----
  1/2/3  select pair (hip_roll/hip_pitch/knee)
  +/-    nudge active pair Δ (mirrored)        x  cycle Δ step
  ]/[    Kp ±0.5    {/}  Kp ±5 (coarse)        '/;  Kd ±0.1
  v      refit Δ to measured pose (hold where it is)
  k      slowly approach the next trial's suggested posture (at current gains)
  0      active pair Δ -> 0 (toward default — careful)
  t      run next trial      s  skip next trial
  r/SPACE PANIC: all gains 0 (limp)
  e/d    enable / disable motors
  q      quit + save
"""

import math
import sys
import time
import argparse
import json
import numpy as np

from motor_bridge import MotorBridge
from imu_adapter import IMUAdapter
import config as C

from sysid_experiment import (
    sig_chirp, sig_triangle,
    setup_raw_keys, restore_keys, get_key,
)

# ════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════
CONTROL_HZ = 100
DT = 1.0 / CONTROL_HZ
DEFAULT_POS = np.array(C.DEFAULT_JOINT_POS, dtype=np.float32)   # spawn [0,.05,0,0,-.05,0]

PAIR = {'hip_roll': (0, 3), 'hip_pitch': (1, 4), 'knee': (2, 5)}
GROUP_ORDER = ['hip_roll', 'hip_pitch', 'knee']
SIM_GROUPS = ['hip_roll', 'hip_pitch', 'knee', 'hip_roll', 'hip_pitch', 'knee']
SIM_NAMES  = ['L_roll', 'L_hip', 'L_knee', 'R_roll', 'R_hip', 'R_knee']
RIGHT_SIGN = {'hip_roll': -1.0, 'hip_pitch': -1.0, 'knee': -1.0}
GROUP_KEYS = {'1': 'hip_roll', '2': 'hip_pitch', '3': 'knee'}

# Suggested HOLD gains to hand-ramp the scaffolding joints toward (you tune live).
HOLD_SUGGEST = {'hip_roll': (20.0, 0.8), 'hip_pitch': (40.0, 1.5), 'knee': (50.0, 1.5)}
# Policy gains the excited pair drops to during a trial (the regime being ID'd).
EXCITE = {'hip_roll': (6.0, 0.3), 'hip_pitch': (8.0, 0.5), 'knee': (8.0, 0.5)}

EFFORT_LIM = np.array([6.0, 17.0, 23.2, 6.0, 17.0, 23.2], dtype=np.float32)
KP_STEP, KD_STEP, KP_MAX, KD_MAX = 0.5, 0.1, 50.0, 50.0
DELTA_STEPS = [0.01, 0.02, 0.05]
DELTA_MAX = {'hip_roll': 1.0, 'hip_pitch': 3.0, 'knee': 3.5}

TEMP_WARN, TEMP_ABORT, COOL_HOLD_S = 55.0, 65.0, 12.0
GAIN_RAMP_S = 1.0          # smooth gain transition into/out of a trial
CONTACT_PAIR = 'knee'

# Terminal colors
DIM, BOLD, YEL, GRN, CYA, RED, RST = ("\033[2m", "\033[1m", "\033[33m",
                                      "\033[32m", "\033[36m", "\033[31m", "\033[0m")


# ════════════════════════════════════════════════════════════════
# Trial definitions — which pair, signal, safe clip (abs Δ), duration.
# `center_suggest` / `hold_suggest` are GUIDES for hand-positioning; the trial
# actually sweeps around the joint's CURRENT measured Δ, clipped to `clip`.
# ════════════════════════════════════════════════════════════════
def build_trials(blocks):
    trials = []
    if 'hip_pitch' in blocks:
        clip = (0.3, 1.8)
        hold = {'hip_roll': 0.0, 'knee': -2.6}
        for a in (0.3, -0.3, 0.5, -0.5):
            trials.append(dict(block='hip_pitch', name=f'hp_step_{a:+.1f}',
                               target_pair='hip_pitch', signal='step', amp=a,
                               dur=3.0, clip=clip, center_suggest=1.0, hold_suggest=hold))
        trials.append(dict(block='hip_pitch', name='hp_chirp', target_pair='hip_pitch',
                           signal='chirp', amp=0.2, f0=0.5, f1=8.0, dur=8.0,
                           clip=clip, center_suggest=1.0, hold_suggest=hold))
        trials.append(dict(block='hip_pitch', name='hp_cv_0p25', target_pair='hip_pitch',
                           signal='triangle', vel=0.25, lim=0.5, dur=6.0,
                           clip=clip, center_suggest=1.0, hold_suggest=hold))
    if 'knee' in blocks:
        clip = (-2.25, -1.90)        # straighter than -1.90 cmd -> tip risk
        hold = {'hip_roll': 0.0, 'hip_pitch': 1.0}
        for a in (-0.15, -0.10, 0.08, 0.10):
            trials.append(dict(block='knee', name=f'kn_step_{a:+.2f}',
                               target_pair='knee', signal='step', amp=a,
                               dur=3.0, clip=clip, center_suggest=-2.05, hold_suggest=hold))
        trials.append(dict(block='knee', name='kn_chirp', target_pair='knee',
                           signal='chirp', amp=0.10, f0=0.5, f1=8.0, dur=6.0,
                           clip=clip, center_suggest=-2.05, hold_suggest=hold))
        trials.append(dict(block='knee', name='kn_cv_0p25', target_pair='knee',
                           signal='triangle', vel=0.25, lim=0.12, dur=4.0,
                           clip=clip, center_suggest=-2.05, hold_suggest=hold))
    return trials


def signal_value(trial, t):
    s = trial['signal']
    if s == 'step':     return trial['amp']
    if s == 'chirp':    return sig_chirp(t, trial['amp'], trial['f0'], trial['f1'], trial['dur'])
    if s == 'triangle': return sig_triangle(t, trial['vel'], trial['lim'])
    raise ValueError(s)


# ════════════════════════════════════════════════════════════════
# Pose / gains helpers
# ════════════════════════════════════════════════════════════════
def fit_delta_from_pose(measured):
    """Symmetric best-fit of per-pair Δ from a measured 6-vector (so a gain ramp
    holds where the robot already is). Mirrors joint_tuner_mirror.py."""
    d = {}
    for g in GROUP_ORDER:
        L, R = PAIR[g]
        dl = measured[L] - DEFAULT_POS[L]
        dr = (measured[R] - DEFAULT_POS[R]) * RIGHT_SIGN[g]
        d[g] = float(np.clip(0.5 * (dl + dr), -DELTA_MAX[g], DELTA_MAX[g]))
    return d


def build_target(d):
    t = DEFAULT_POS.copy()
    for g in GROUP_ORDER:
        L, R = PAIR[g]
        t[L] = DEFAULT_POS[L] + d[g]
        t[R] = DEFAULT_POS[R] + RIGHT_SIGN[g] * d[g]
    return t


def expand_gains(kp, kd):
    return (np.array([kp[g] for g in SIM_GROUPS], np.float32),
            np.array([kd[g] for g in SIM_GROUPS], np.float32))


def trial_gain_vecs(kp, kd, target_pair):
    """6-vec gains: held pairs keep current hand gains, excited pair -> EXCITE."""
    kpv, kdv = expand_gains(kp, kd)
    L, R = PAIR[target_pair]
    kpv = kpv.copy(); kdv = kdv.copy()
    kpv[L] = kpv[R] = EXCITE[target_pair][0]
    kdv[L] = kdv[R] = EXCITE[target_pair][1]
    return kpv, kdv


def block_posture(block):
    """Predetermined per-pair Δ for a block: held pairs at hold_suggest, excited
    pair at center_suggest (read off that block's first trial). This is the pose
    you tune the gains up against."""
    tr = next(t for t in build_trials([block]) if t['block'] == block)
    d = {g: 0.0 for g in GROUP_ORDER}
    for g, v in tr['hold_suggest'].items():
        d[g] = v
    d[tr['target_pair']] = tr['center_suggest']
    return d


# ════════════════════════════════════════════════════════════════
# Motion primitives (all hold position; gains may interpolate)
# ════════════════════════════════════════════════════════════════
def _tick(bridge, imu, target, kpv, kdv):
    imu.update()
    bridge.read_state()
    bridge.send_target(target.astype(np.float32), kpv, kdv, effort_limit=EFFORT_LIM)


def approach_posture(bridge, imu, d_goal, kp, kd, duration=4.0):
    """Slow cosine position ramp from the current measured pose to
    build_target(d_goal), holding at the CURRENT gains. The safe way to settle
    into a known posture or cross between blocks — bounded error, no gain-vs-
    error yank. Returns False on abort ('q')."""
    kpv, kdv = expand_gains(kp, kd)
    imu.update(); bridge.read_state()
    start = bridge.sim_joint_pos.copy()
    goal = build_target(d_goal)
    t0 = time.monotonic()
    while True:
        tt = time.monotonic(); a = (tt - t0) / duration
        if a >= 1.0: break
        b = 0.5 * (1.0 - math.cos(math.pi * a))
        tgt = start * (1.0 - b) + goal * b
        if get_key() in ('q', '\x03'):
            return False
        _tick(bridge, imu, tgt, kpv, kdv)
        rem = DT - (time.monotonic() - tt)
        if rem > 0: time.sleep(rem)
    return True


def hold_pose(bridge, imu, target, kpv, kdv, duration, abort_keys=('q', '\x03')):
    t0 = time.monotonic()
    while time.monotonic() - t0 < duration:
        tt = time.monotonic()
        if get_key() in abort_keys:
            return False
        _tick(bridge, imu, target, kpv, kdv)
        rem = DT - (time.monotonic() - tt)
        if rem > 0: time.sleep(rem)
    return True


def ramp_gains(bridge, imu, target, kpv0, kdv0, kpv1, kdv1, duration):
    """Linearly interpolate gain vectors while holding `target` fixed."""
    t0 = time.monotonic()
    while True:
        tt = time.monotonic()
        a = (tt - t0) / duration
        if a >= 1.0: break
        kpv = kpv0 * (1 - a) + kpv1 * a
        kdv = kdv0 * (1 - a) + kdv1 * a
        if get_key() in ('q', '\x03'):
            return False
        _tick(bridge, imu, target, kpv.astype(np.float32), kdv.astype(np.float32))
        rem = DT - (time.monotonic() - tt)
        if rem > 0: time.sleep(rem)
    return True


# ════════════════════════════════════════════════════════════════
# Trial execution (called when armed and the user presses 't')
# ════════════════════════════════════════════════════════════════
def run_trial(bridge, imu, trial, d, kp, kd):
    """Sweep the excited pair around its current Δ (clipped), at policy gains.
    `d`,`kp`,`kd` are the live hand-set state. Returns (rec|None, info)."""
    tp = trial['target_pair']
    L, R = PAIR[tp]
    clo, chi = trial['clip']

    center = float(np.clip(d[tp], clo, chi))
    if abs(center - d[tp]) > 1e-6:
        info = f"Δ {d[tp]:+.3f} outside safe window {trial['clip']} -> clipped to {center:+.3f}"
    else:
        info = f"sweeping around Δ {center:+.3f}, window {trial['clip']}"

    d_run = dict(d); d_run[tp] = center
    base_target = build_target(d_run)

    kpv_hold, kdv_hold = expand_gains(kp, kd)
    kpv_ex, kdv_ex = trial_gain_vecs(kp, kd, tp)

    # 1) smooth gain drop on the excited pair -> let it sag to loaded equilibrium
    if not ramp_gains(bridge, imu, base_target, kpv_hold, kdv_hold,
                      kpv_ex, kdv_ex, GAIN_RAMP_S):
        return None, "aborted during gain ramp-down"
    if not hold_pose(bridge, imu, base_target, kpv_ex, kdv_ex, 1.0):
        return None, "aborted during settle"

    # 2) record the excitation
    chans = ('time', 'p_des', 'pos', 'vel', 'torque', 'temp',
             'proj_grav', 'ang_vel', 'quat')
    buf = {k: [] for k in chans}
    t0 = time.monotonic()
    aborted = False
    while True:
        tt = time.monotonic(); t = tt - t0
        if t >= trial['dur']: break
        k = get_key()
        if k in ('q', '\x03'): aborted = True; break
        if k == 's': break

        imu.update(); bridge.read_state()
        delta = float(np.clip(center + signal_value(trial, t), clo, chi))
        d_run[tp] = delta
        p_des = build_target(d_run)
        bridge.send_target(p_des.astype(np.float32), kpv_ex, kdv_ex, effort_limit=EFFORT_LIM)

        buf['time'].append(t)
        buf['p_des'].append(p_des.copy())
        buf['pos'].append(bridge.sim_joint_pos.copy())
        buf['vel'].append(bridge.sim_joint_vel.copy())
        buf['torque'].append(bridge.sim_joint_torque.copy())
        buf['temp'].append(bridge.motor_temperature.copy())
        buf['proj_grav'].append(imu.projected_gravity().copy())
        buf['ang_vel'].append(imu.ang_vel_body().copy())
        buf['quat'].append(imu.quat().copy())
        rem = DT - (time.monotonic() - tt)
        if rem > 0: time.sleep(rem)

    # 3) smooth gain ramp back up to the hand-set hold, at center
    d_run[tp] = center
    base_target = build_target(d_run)
    ramp_gains(bridge, imu, base_target, kpv_ex, kdv_ex, kpv_hold, kdv_hold, GAIN_RAMP_S)

    if aborted:
        return None, "aborted during recording"

    rec = {k: np.array(v) for k, v in buf.items()}
    rec['kp'] = kpv_ex
    rec['kd'] = kdv_ex
    cfg = {k: v for k, v in trial.items() if not isinstance(v, np.ndarray)}
    cfg['center_used'] = center
    cfg['hold_used'] = {g: d[g] for g in GROUP_ORDER}
    rec['trial'] = cfg
    return rec, info


# ════════════════════════════════════════════════════════════════
# Display
# ════════════════════════════════════════════════════════════════
def render(kp, kd, d, active, dstep, enabled, bridge, imu, trials, idx, n_done, msg):
    L = []
    state = f"{GRN}{BOLD}ENABLED{RST}" if enabled else f"{YEL}{BOLD}DISABLED{RST}"
    armed = any(kp[g] > 0 for g in GROUP_ORDER)
    tag = "" if armed else f"  {YEL}● all gains 0 → limp{RST}"
    L.append(f"  Motors: {state}{tag}    {DIM}recorded {n_done}{RST}")
    L.append(f"  {DIM}1/2/3 pair  +/- Δ±{DELTA_STEPS[dstep]:g}  ]/[ Kp  {{/}} Kp±5  "
             f"'/; Kd  v refit  k approach  t run  s skip  r/SPACE limp  q quit{RST}")
    if idx < len(trials):
        tr = trials[idx]
        L.append(f"  {CYA}next [{idx+1}/{len(trials)}] {tr['name']}  ({tr['block']}, "
                 f"{tr['signal']}, {tr['dur']:.1f}s){RST}  "
                 f"{DIM}suggest: excite {tr['target_pair']} Δ≈{tr['center_suggest']:+.2f}, "
                 f"holds {tr['hold_suggest']}{RST}")
    else:
        L.append(f"  {GRN}all trials done — press q to save{RST}")
    if msg:
        L.append(f"  {YEL}{msg}{RST}")
    L.append("")
    L.append(f"  {BOLD}{'pair':<10s}  Kp      Kd     Δ      {'rec Kp/Kd':>12s}{RST}")
    for g in GROUP_ORDER:
        mk = f"{YEL}>{RST}" if g == active else " "
        gn = f"{BOLD}{g:<10s}{RST}" if g == active else f"{g:<10s}"
        rk, rd = HOLD_SUGGEST[g]
        L.append(f"  {mk} {gn}  {kp[g]:5.1f}  {kd[g]:5.2f}  {d[g]:+6.3f}   "
                 f"{DIM}{rk:.0f}/{rd:.1f}{RST}")
    L.append("")
    target = build_target(d)
    big_err = False
    L.append(f"  {BOLD}{'joint':<8s}  {'pos':>7s}  {'err':>7s}  {'vel':>7s}  "
             f"{'τ':>6s}  {'temp':>6s}{RST}")
    for i in range(6):
        err = target[i] - bridge.sim_joint_pos[i]
        if abs(err) > 0.15:
            big_err = True
            es = f"{RED}{err:+7.3f}{RST}"
        else:
            es = f"{err:+7.3f}"
        t = bridge.motor_temperature[i]
        ts = (f"{RED}{t:6.1f}{RST}" if t >= TEMP_WARN else
              f"{YEL}{t:6.1f}{RST}" if t >= 50 else f"{t:6.1f}")
        L.append(f"  {SIM_NAMES[i]:<8s}  {bridge.sim_joint_pos[i]:+7.3f}  {es}  "
                 f"{bridge.sim_joint_vel[i]:+7.3f}  {bridge.motor_torque[i]:+6.2f}  {ts}")
    if big_err and armed:
        L.append(f"  {RED}● large pos error vs target — reposition by hand or "
                 f"'k' before raising Kp (gain×error = yank){RST}")
    pg = imu.projected_gravity()
    L.append(f"  {BOLD}IMU{RST} upright={imu.upright_cos():+.3f}  "
             f"proj_grav=({pg[0]:+.2f},{pg[1]:+.2f},{pg[2]:+.2f})")
    return L


# ════════════════════════════════════════════════════════════════
# Save
# ════════════════════════════════════════════════════════════════
def save_all(results):
    fname = f"sysid_loaded_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    arrays, names, configs = {}, [], []
    for name, rec in results.items():
        names.append(name)
        configs.append(rec['trial'])
        for ch in ('time', 'p_des', 'pos', 'vel', 'torque', 'temp',
                   'proj_grav', 'ang_vel', 'quat', 'kp', 'kd'):
            arrays[f'{name}__{ch}'] = rec[ch]
    arrays['trial_names']   = np.array(names)
    arrays['trial_configs'] = np.array(json.dumps(configs))
    arrays['control_hz']    = np.array(CONTROL_HZ)
    arrays['default_pos']   = DEFAULT_POS
    arrays['contact_pair']  = np.array(CONTACT_PAIR)
    arrays['pair_index']    = np.array(json.dumps(PAIR))
    arrays['right_sign']    = np.array(json.dumps(RIGHT_SIGN))
    arrays['knee_hip_coupling'] = np.array(C.KNEE_HIP_COUPLING)
    arrays['knee_gear_ratio']   = np.array(C.KNEE_GEAR_RATIO)
    np.savez(fname, **arrays)
    return fname


# ════════════════════════════════════════════════════════════════
# Main — interactive loop
# ════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--block', choices=['hip_pitch', 'knee'], action='append')
    ap.add_argument('--dry', action='store_true')
    args = ap.parse_args()
    blocks = args.block if args.block else ['hip_pitch', 'knee']
    trials = build_trials(blocks)

    print(f"=== sysid_loaded.py (hand-armed) — blocks {blocks}, {len(trials)} trials ===")
    for t in trials:
        print(f"  {t['block']:<9s} {t['name']:<16s} {t['signal']:<9s} {t['dur']:.1f}s")
    if args.dry:
        return
    print("\n*** Place the robot in an approximate two-leg kneel on a mat. ***")
    print("Motors will start LIMP. Ramp gains by hand before doing anything.\n")

    bridge = MotorBridge(action_lpf_alpha=0.0)
    imu = IMUAdapter(); imu.open()
    for _ in range(10):
        imu.update(); time.sleep(0.005)
    bridge.enable(); time.sleep(0.3)
    for _ in range(3):
        bridge.send_idle(); time.sleep(0.02)
    imu.update(); bridge.read_state()

    # Start LIMP. Target the predetermined block posture so that as you ramp
    # gains up (with the robot hand-placed near the kneel) it settles into the
    # designed operating point — and trials then sweep around the designed
    # centers. 'v' re-pins to the measured pose if you ever want that instead.
    kp = {g: 0.0 for g in GROUP_ORDER}
    kd = {g: 0.0 for g in GROUP_ORDER}
    d = block_posture(trials[0]['block']) if trials else fit_delta_from_pose(bridge.sim_joint_pos)
    active = 'hip_roll'
    dstep = 1
    enabled = True
    idx = 0
    results = {}
    msg = "limp — place robot near kneel, ramp gains up (gains 0 = no motion yet), then t"

    fd, old = setup_raw_keys()
    period = DT
    last_print, first, nlines = 0.0, True, 0
    try:
        while True:
            t_loop = time.monotonic()
            redraw = False
            k = get_key()
            if k:
                redraw = True
                g = active
                if k in GROUP_KEYS:
                    active = GROUP_KEYS[k]
                elif k in ('+', '='):
                    d[g] = min(DELTA_MAX[g], d[g] + DELTA_STEPS[dstep])
                elif k in ('-', '_'):
                    d[g] = max(-DELTA_MAX[g], d[g] - DELTA_STEPS[dstep])
                elif k == ']':
                    kp[g] = min(KP_MAX, kp[g] + KP_STEP)
                elif k == '[':
                    kp[g] = max(0.0, kp[g] - KP_STEP)
                elif k == '}':
                    kp[g] = min(KP_MAX, kp[g] + 5.0)      # coarse
                elif k == '{':
                    kp[g] = max(0.0, kp[g] - 5.0)         # coarse
                elif k == "'":
                    kd[g] = min(KD_MAX, kd[g] + KD_STEP)
                elif k == ';':
                    kd[g] = max(0.0, kd[g] - KD_STEP)
                elif k == 'k':
                    if idx < len(trials):
                        goal = block_posture(trials[idx]['block'])
                        sys.stdout.write("\n\n  approaching posture (q aborts) ...\n")
                        sys.stdout.flush()
                        if approach_posture(bridge, imu, goal, kp, kd):
                            d = goal; msg = "reached suggested posture"
                        else:
                            msg = "approach aborted"
                        first = True
                elif k == 'v':
                    d = fit_delta_from_pose(bridge.sim_joint_pos)
                    msg = "Δ refit to measured pose"
                elif k == '0':
                    d[g] = 0.0
                elif k == 'x':
                    dstep = (dstep + 1) % len(DELTA_STEPS)
                elif k in ('r', ' '):
                    kp = {gg: 0.0 for gg in GROUP_ORDER}
                    kd = {gg: 0.0 for gg in GROUP_ORDER}
                    msg = "PANIC — gains zeroed (limp)"
                elif k == 'e':
                    if not enabled:
                        bridge.enable(); enabled = True
                        bridge.read_state(); d = fit_delta_from_pose(bridge.sim_joint_pos)
                elif k == 'd':
                    if enabled:
                        bridge.disable(); enabled = False
                elif k == 's':
                    if idx < len(trials):
                        msg = f"skipped {trials[idx]['name']}"; idx += 1
                elif k == 't':
                    if idx >= len(trials):
                        msg = "no more trials"
                    elif kp[trials[idx]['target_pair']] <= 0:
                        msg = f"ramp {trials[idx]['target_pair']} gains up first"
                    else:
                        tr = trials[idx]
                        # temp gate
                        bridge.read_state()
                        if float(bridge.motor_temperature.max()) > TEMP_ABORT:
                            msg = "TOO HOT — cool down before running"
                        else:
                            sys.stdout.write(f"\n\n  running {tr['name']} ...\n")
                            sys.stdout.flush()
                            rec, info = run_trial(bridge, imu, tr, d, kp, kd)
                            if rec is None:
                                msg = f"{tr['name']}: {info}"
                            else:
                                results[tr['name']] = rec
                                msg = f"{tr['name']} recorded ({len(rec['time'])} ticks) — {info}"
                                idx += 1
                                if idx < len(trials) and trials[idx]['block'] != tr['block']:
                                    msg += "  | NEXT BLOCK: lower gains, press k to approach new posture"
                            first = True   # re-anchor display after the trial
                            bridge.read_state()
                            if float(bridge.motor_temperature.max()) > TEMP_WARN:
                                msg += f"  | max temp {bridge.motor_temperature.max():.0f}°C — consider cooling"
                elif k in ('q', '\x03'):
                    break

            imu.update(); bridge.read_state()
            kpv, kdv = expand_gains(kp, kd)
            if enabled:
                bridge.send_target(build_target(d).astype(np.float32), kpv, kdv,
                                   effort_limit=EFFORT_LIM)
            else:
                bridge.send_idle()

            now = time.monotonic()
            if redraw or (now - last_print >= 0.1):
                lines = render(kp, kd, d, active, dstep, enabled, bridge, imu,
                               trials, idx, len(results), msg)
                text = "\n".join(lines)
                if first:
                    sys.stdout.write(text); first = False
                else:
                    sys.stdout.write(f"\033[{nlines-1}F\033[J"); sys.stdout.write(text)
                nlines = len(lines); sys.stdout.flush(); last_print = now

            el = time.monotonic() - t_loop
            if el < period: time.sleep(period - el)

    except KeyboardInterrupt:
        pass
    finally:
        try: restore_keys(fd, old)
        except Exception: pass
        try:
            for _ in range(3):
                bridge.send_idle(); time.sleep(0.02)
        except Exception: pass
        try: bridge.disable()
        except Exception: pass
        try: bridge.close()
        except Exception: pass
        try: imu.close()
        except Exception: pass
        if results:
            print(f"\n\nSaved {len(results)} trials → {save_all(results)}")
        else:
            print("\n\nNo trials recorded.")
        print("Motors disabled. Done.")


if __name__ == '__main__':
    main()