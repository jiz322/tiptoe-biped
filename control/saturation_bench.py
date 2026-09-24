"""
saturation_bench.py — Phase C bench test for the high-saturation torque gap.

Companion to data/saturation_forensics.py (Phase A, offline forensics on hang
logs). Phase A ruled out current-limit/decode-rail (0% at-rail in the corpus)
and weakened CAN staleness, and instead found a knee-specific, reproducible,
direction-locked low-pass signature — pointing at belt compliance/slip/backlash.
This script confirms and sizes that on hardware, using the SAME motor model
with and without the belt:

    hip_pitch1 (motor 101, RS01, direct drive)
    knee1      (motor 102, RS01, through the 22:30 belt)

Anything that shows up on both is firmware/motor-generic; anything knee-only
is belt mechanics.

═══ TEST MODES ═══

FREE-HANGING (default — robot hanging, feet clear, nothing else needed):
  [lash] Direction-reversal lash test. From a settled hold, open-loop t_ff
      starts at the measured hold torque (no step, no kick) and ramps slowly
      toward +-amp. The instant sustained joint velocity appears, the ramp
      STOPS and the leg is gently caught (low-kp position hold). Measured:
      raw MOTOR-frame displacement before motion onset (wind-up + lash) and
      the breakaway t_ff. hip_pitch1's onset displacement is ordinary
      stiction/backlash; knee1's excess over it is belt-specific lash.

LOCKED (--locked — REQUIRES a rigid mechanical fixture blocking the joint;
      NEVER hold the leg by hand: commanded torques reach the 17 Nm cap):
  [sat] PD saturation steps. Joint blocked -> position error is sustained ->
      commanded torque = clip(kp*err) held for the full step. Escalating
      torque levels, operator keypress between levels. Motor-encoder creep
      while the joint is physically locked = belt stretch/slip (for the
      knee, sim_p is DERIVED from the motor encoder, so belt slip appears
      as phantom joint motion — logged as raw_disp).
      [Note 2026-08-07: firmware registers were read directly (limit_torque
      = 17/6 Nm = the assumed caps, no derating), so the current-limit
      question is already answered; locked mode is optional belt-only
      follow-up and the user has deprioritized it.]
  [tff] Open-loop t_ff ramp from the hold torque up past the decode rail.
      tau_meas vs t_ff linearity and where it caps.

  Deep saturation is IMPOSSIBLE free-hanging: 17 Nm on a hanging shank is
  >300 rad/s^2 — the 5 rad/s abort trips within ~4 control ticks, before
  anything is measured. Hence the fixture requirement.

═══ POSITION SAFETY IS RELATIVE TO THE ARMED POSE ═══
friction_id.py's absolute MOTOR_PROFILES windows predate the current
zero-pose/coupling conventions — today's hang pose alone sits outside them
(hip_pitch ~0.5 rad), which insta-aborted the first live run. The bench
therefore pins its own window around wherever the robot actually hangs at
arm time: excursion beyond +-BENCH_POS_HARD_REL from the armed pose aborts,
and "recenter" means "ramp back to the armed pose". The lash test itself
moves the joint only a few hundredths of a radian before the catch.

═══ SAFETY MODEL (arming pattern from balance_log.py) ═══
  * Motors start ENABLED but LIMP (all gains 0). Nothing moves until you
    press 'g', which smoothly ramps hold gains up over 2 s with every target
    pinned to the measured pose — no jump by construction.
  * SPACE or 'p' at ANY time = panic: Kd-only brake until velocities decay,
    then limp. 'q' = quit through the same path. Keys are polled every tick.
  * EVERY exit path (panic, safety abort, Ctrl-C, crash, normal end) goes
    through safe_stop(): brake -> wait for |vel| to decay -> limp -> disable.
    Motors are never disabled while moving.
  * 200 Hz hard aborts: armed-pose-relative position window, vel 5.0 rad/s,
    temp 70 C, held-joint drift 0.15 rad (vel/temp/drift via friction_id's
    _check_safety). In locked mode an additional tight motion gate.
  * --scale (default 0.5) multiplies every torque amplitude. First live run
    at the default; pass --scale 1.0 only after a clean shakedown.

Usage:
    python3 saturation_bench.py --dry-run              # full plan, no hardware
    sudo python3 saturation_bench.py                   # free-hanging: lash only
    sudo python3 saturation_bench.py --locked --motor knee1 --scale 1.0
"""

import argparse
import csv
import sys
import time
from datetime import datetime

import numpy as np

import config as C
from robstride_socketcan_driver import make_driver
from friction_id import (
    compute_sim, torque_joint_to_motor, torque_motor_to_joint,
    kd_joint_to_motor, raw_target_for_joint, _is_knee,
    _read_all_raw, _send_test, _send_all, _check_safety, _get_state_joint,
    _capture_hold_positions, _sleep_until,
    MOTOR_TO_JOINT_IDX,
    LOOP_DT, CONTROL_HZ, VEL_HARD_LIMIT_JOINT, TEMP_LIMIT_C,
    RECENTER_KP_MOTOR, RECENTER_KD_MOTOR, STATIC_RAMP_KD_MOTOR,
    HOLD_DRIFT_LIMIT_RAD, OTHER_HOLD_KP_MOTOR, OTHER_HOLD_KD_MOTOR,
    BRAKE_KD_MOTOR,
)

# Kp and Kd are both "torque per unit joint-side error/rate" — the belt
# transform (gear^2) applies identically to either, so reuse the same fn.
gain_joint_to_motor = kd_joint_to_motor

BENCH_MOTORS = {'hip_pitch1': 101, 'knee1': 102}

# ── arming / stop ───────────────────────────────────────────────────────
ARM_RAMP_S = 2.0            # limp -> hold-gain ramp duration
STOP_BRAKE_KD = BRAKE_KD_MOTOR   # Kd-only brake used by safe_stop / panic
STOP_VEL_RAW = 0.3          # rad/s (motor frame): "stopped" threshold
STOP_BRAKE_TIMEOUT_S = 2.5
SETTLE_S = 0.5              # hold-and-measure-tau0 window before open-loop

# ── armed-pose-relative position safety (see docstring) ─────────────────
BENCH_POS_HARD_REL = 0.35   # rad joint-side excursion from armed pose -> abort
RETURN_RAMP_S = 2.0         # ramp back to the armed pose between trials

# ── [lash] free-hanging direction-reversal ──────────────────────────────
LASH_AMP_JOINT = {'hip_pitch': 3.0, 'knee': 3.0}   # Nm, x --scale
LASH_RAMP_RATE = 0.75       # Nm/s joint-side — slow enough to catch onset cleanly
LASH_N_TRIALS = 4
LASH_VEL_ONSET_JOINT = 0.03  # rad/s sustained mean -> "moving"
LASH_VEL_WINDOW_S = 0.08
LASH_CATCH_S = 0.5          # gentle low-kp catch after onset

# ── [sat] locked PD saturation steps ────────────────────────────────────
SAT_KP_JOINT = {'hip_pitch': 85.0, 'knee': 60.0}
SAT_KD_JOINT = {'hip_pitch': 1.5, 'knee': 1.5}
SAT_MULTS = [0.7, 1.0, 1.5, 2.0]   # x burst cap, x --scale; keypress between levels
SAT_HOLD_S = 1.5
SAT_STEP_RAMP_S = 0.25      # target onset ramp — no instantaneous torque step
LOCKED_VEL_GATE = 1.5       # rad/s joint-side: fixture-failure brake (tighter than 5.0)
LOCKED_SLIP_ABORT = 0.30    # rad apparent joint motion while locked -> abort (slip IS
                            # the measurement below this; above it the fixture moved)

# ── [tff] locked open-loop ramp ─────────────────────────────────────────
TFF_RAMP_RAIL_MULT = 1.3    # ramp to this x the decode rail (17 Nm motor-side), x --scale
TFF_RAMP_DURATION = 6.0
RS01_TFF_MAX = 17.0         # robstride_biped_driver.py MOTOR_MIT_OVERRIDES _RS01


class BenchPanic(Exception):
    """Operator pressed the panic/quit key."""


# ══════════════════════════════════════════════════════════════════════
# Keyboard — non-blocking single chars, pattern from balance_log.py.
# termios is Linux-only; guarded so --dry-run still runs anywhere.
# ══════════════════════════════════════════════════════════════════════
try:
    import select
    import termios
    import tty
    _HAVE_TERMIOS = True
except ImportError:
    _HAVE_TERMIOS = False

_key_state = None


def setup_raw_keys():
    global _key_state
    if not _HAVE_TERMIOS:
        return
    fd = sys.stdin.fileno()
    _key_state = (fd, termios.tcgetattr(fd))
    tty.setcbreak(fd)


def restore_keys():
    if _key_state is not None:
        termios.tcsetattr(_key_state[0], termios.TCSADRAIN, _key_state[1])


def get_key_nonblocking():
    if not _HAVE_TERMIOS:
        return None
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def poll_panic():
    """Call every control tick. Raises BenchPanic on SPACE/'p'/'q'."""
    k = get_key_nonblocking()
    if k in (' ', 'p'):
        raise BenchPanic("operator panic key")
    if k == 'q':
        raise BenchPanic("operator quit")
    return k


def wait_for_key(prompt, keys=('g',), tick=None):
    """Block until one of `keys` is pressed; SPACE/'p'/'q' still panic.
    `tick` (if given) is called every LOOP_DT so motors keep receiving
    commands and safety checks keep running during the wait — MIT firmware
    executes the last received frame forever, so an unsupervised wait would
    leave gains active with no temp/drift monitoring."""
    print(prompt, flush=True)
    while True:
        k = poll_panic()
        if k in keys:
            return k
        if tick is not None:
            tick()
        time.sleep(LOOP_DT if tick is not None else 0.02)


# ══════════════════════════════════════════════════════════════════════
# Arming, stopping, pose-relative safety — no sudden motion on any path
# ══════════════════════════════════════════════════════════════════════
def _check_safety_rel(driver, mid, j_idx, sim_p0, hold_positions=None):
    """friction_id's vel/temp/held-drift checks, plus a position window
    RELATIVE to the armed pose (friction_id's absolute windows are stale —
    see docstring)."""
    sim_p, sim_v, temp, sim_all = _check_safety(
        driver, mid, j_idx, float('inf'), hold_positions)
    if abs(sim_p - sim_p0) > BENCH_POS_HARD_REL:
        raise RuntimeError(
            f"HARD position (rel): sim_p={sim_p:+.3f} is "
            f"{sim_p - sim_p0:+.3f} rad from armed pose "
            f"(limit +-{BENCH_POS_HARD_REL})")
    return sim_p, sim_v, temp, sim_all


def hold_tick(driver, mid, j_idx, sim_p0, hold_positions):
    """Factory: a per-tick callback that holds the tested motor at its
    current position (gentle recenter gains) and runs safety checks.
    Used to keep waits-for-keypress supervised."""
    raw_now = driver.get_state(mid)[0]

    def tick():
        _send_test(driver, mid, raw_now, 0.0,
                   RECENTER_KP_MOTOR, RECENTER_KD_MOTOR, 0.0,
                   hold_positions=hold_positions)
        _check_safety_rel(driver, mid, j_idx, sim_p0, hold_positions)
    return tick


def safe_stop(driver, reason=""):
    """Brake -> wait for velocities to decay -> limp -> disable.
    Safe to call from any state, including mid-swing. Never raises."""
    print(f"\n[safe_stop] {reason}" if reason else "\n[safe_stop]")
    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < STOP_BRAKE_TIMEOUT_S:
            ls = time.monotonic()
            _send_all(driver, lambda m: (0.0, 0.0, 0.0, STOP_BRAKE_KD, 0.0))
            vmax = max(abs(driver.get_state(m)[1]) for m in C.MOTOR_IDS)
            if vmax < STOP_VEL_RAW and time.monotonic() - t0 > 0.4:
                break
            _sleep_until(ls + LOOP_DT)
        for _ in range(10):     # explicit limp before disable
            _send_all(driver, lambda m: (0.0, 0.0, 0.0, 0.0, 0.0))
            time.sleep(LOOP_DT)
    except Exception as e:
        print(f"[safe_stop] brake phase error ({e}) — disabling anyway")
    try:
        driver.disable_all()
    except Exception:
        pass


def arm_interactive(driver, mid, j_idx):
    """balance_log.py arming model: enabled-but-limp, targets pinned to the
    measured pose, operator key starts a smooth gain ramp.
    Returns (hold_positions, raw_p0, sim_p0): the pinned pose in motor and
    joint frames — the anchor for all position safety and recentering."""
    print("  [arm] motors enabled, LIMP (all gains 0).")
    end = time.monotonic() + 0.5
    while time.monotonic() < end:       # limp idle; populate motor_states
        ls = time.monotonic()
        _send_all(driver, lambda m: (0.0, 0.0, 0.0, 0.0, 0.0))
        poll_panic()
        _sleep_until(ls + LOOP_DT)

    pinned = _capture_hold_positions(driver)
    sim_all = compute_sim(_read_all_raw(driver))
    sim_p0 = float(sim_all[j_idx])
    raw_p0 = pinned[mid]
    print("  [arm] measured pose (sim frame): "
          + "  ".join(f"{C.SIM_JOINT_NAMES[i]}={sim_all[i]:+.3f}" for i in range(6)))
    print(f"  [arm] position window: {C.SIM_JOINT_NAMES[j_idx]} = "
          f"{sim_p0:+.3f} +- {BENCH_POS_HARD_REL} rad (armed-pose-relative)")
    wait_for_key("  [arm] press 'g' to ramp hold gains up over "
                 f"{ARM_RAMP_S:.0f}s  (SPACE/'p' panic, 'q' quit)", keys=('g',),
                 tick=lambda: _send_all(driver,
                                        lambda m: (0.0, 0.0, 0.0, 0.0, 0.0)))

    t0 = time.monotonic()
    while True:
        ls = time.monotonic()
        a = min(1.0, (ls - t0) / ARM_RAMP_S)
        _send_all(driver, lambda m: (pinned[m], 0.0,
                                     a * OTHER_HOLD_KP_MOTOR,
                                     a * OTHER_HOLD_KD_MOTOR, 0.0))
        poll_panic()
        _check_safety_rel(driver, mid, j_idx, sim_p0)
        if a >= 1.0:
            break
        _sleep_until(ls + LOOP_DT)
    print("  [arm] holding at pinned pose.")
    return pinned, raw_p0, sim_p0


def return_to_pose(driver, mid, j_idx, raw_p0, sim_p0, hold_positions,
                   ramp_duration=RETURN_RAMP_S):
    """Ramp the tested motor back to the armed pose (replaces friction_id's
    _recenter, which targets a stale absolute window center)."""
    raw_start = driver.get_state(mid)[0]
    t0 = time.monotonic()
    while True:
        ls = time.monotonic()
        a = min(1.0, (ls - t0) / ramp_duration)
        p_cmd = raw_start + (raw_p0 - raw_start) * a
        _send_test(driver, mid, p_cmd, 0.0,
                   RECENTER_KP_MOTOR, RECENTER_KD_MOTOR, 0.0,
                   hold_positions=hold_positions)
        poll_panic()
        _check_safety_rel(driver, mid, j_idx, sim_p0, hold_positions)
        if a >= 1.0:
            break
        _sleep_until(ls + LOOP_DT)
    end = time.monotonic() + 0.15       # brief settle
    while time.monotonic() < end:
        ls = time.monotonic()
        _send_test(driver, mid, raw_p0, 0.0,
                   RECENTER_KP_MOTOR, RECENTER_KD_MOTOR, 0.0,
                   hold_positions=hold_positions)
        poll_panic()
        _check_safety_rel(driver, mid, j_idx, sim_p0, hold_positions)
        _sleep_until(ls + LOOP_DT)


def settle_tau0(driver, mid, j_idx, sim_p0, hold_positions):
    """Hold the tested motor at its current position and return the median
    holding torque (joint-side) — the kick-free start value for t_ff ramps."""
    raw_hold = driver.get_state(mid)[0]
    taus = []
    end = time.monotonic() + SETTLE_S
    while time.monotonic() < end:
        ls = time.monotonic()
        _send_test(driver, mid, raw_hold, 0.0,
                   RECENTER_KP_MOTOR, RECENTER_KD_MOTOR, 0.0,
                   hold_positions=hold_positions)
        poll_panic()
        _check_safety_rel(driver, mid, j_idx, sim_p0, hold_positions)
        taus.append(_get_state_joint(driver, mid, j_idx)[2])
        _sleep_until(ls + LOOP_DT)
    return float(np.median(taus)), raw_hold


def catch_and_hold(driver, mid, j_idx, sim_p0, hold_positions):
    """Gentle catch after a lash onset: low-kp position hold at wherever the
    joint is right now (kp=4 motor-side — a nudge, not a snap)."""
    raw_now = driver.get_state(mid)[0]
    end = time.monotonic() + LASH_CATCH_S
    while time.monotonic() < end:
        ls = time.monotonic()
        _send_test(driver, mid, raw_now, 0.0,
                   RECENTER_KP_MOTOR, 2.0, 0.0,
                   hold_positions=hold_positions)
        poll_panic()
        _check_safety_rel(driver, mid, j_idx, sim_p0, hold_positions)
        _sleep_until(ls + LOOP_DT)


# ══════════════════════════════════════════════════════════════════════
# [lash] free-hanging direction-reversal — the default test
# ══════════════════════════════════════════════════════════════════════
def measure_lash(driver, mid, j_idx, jtype, bench0, log_writer,
                 hold_positions, scale):
    raw_p0, sim_p0 = bench0['raw_p0'], bench0['sim_p0']
    amp_j = LASH_AMP_JOINT[jtype] * scale
    window_n = max(2, int(LASH_VEL_WINDOW_S * CONTROL_HZ))

    print(f"\n== [lash] motor {mid}/{C.SIM_JOINT_NAMES[j_idx]} — t_ff ramps from "
          f"hold torque toward +-{amp_j:.1f}Nm at {LASH_RAMP_RATE:.2f}Nm/s, "
          f"stop at onset, {LASH_N_TRIALS} trials/direction ==")

    results = {+1: [], -1: []}
    for trial in range(LASH_N_TRIALS):
        for direction in (+1, -1):
            return_to_pose(driver, mid, j_idx, raw_p0, sim_p0, hold_positions)
            tau0, _ = settle_tau0(driver, mid, j_idx, sim_p0, hold_positions)
            raw_start = driver.get_state(mid)[0]

            vel_window, onset = [], None
            t0 = time.monotonic()
            max_dur = amp_j / LASH_RAMP_RATE + 1.0
            while time.monotonic() - t0 < max_dur:
                ls = time.monotonic()
                # kick-free: starts exactly at tau0, ramps outward
                t_ff_j = tau0 + direction * min(amp_j, LASH_RAMP_RATE * (ls - t0))
                _send_test(driver, mid, 0.0, 0.0, 0.0, STATIC_RAMP_KD_MOTOR,
                           torque_joint_to_motor(t_ff_j, j_idx),
                           hold_positions=hold_positions)
                poll_panic()
                sim_p, sim_v, temp, _ = _check_safety_rel(
                    driver, mid, j_idx, sim_p0, hold_positions)
                raw_disp = driver.get_state(mid)[0] - raw_start

                vel_window.append(sim_v)
                if len(vel_window) > window_n:
                    vel_window.pop(0)
                log_writer.writerow(['lash', direction, trial, f"{ls:.6f}",
                                     f"{t_ff_j:.4f}", f"{sim_p:.4f}",
                                     f"{sim_v:.4f}", '', f"{raw_disp:.5f}",
                                     f"{temp:.2f}"])
                if (len(vel_window) == window_n and
                        abs(sum(vel_window) / window_n) > LASH_VEL_ONSET_JOINT):
                    onset = dict(raw_disp=abs(raw_disp),
                                 tff_break=abs(t_ff_j - tau0))
                    break               # STOP the ramp the moment it moves
                _sleep_until(ls + LOOP_DT)

            catch_and_hold(driver, mid, j_idx, sim_p0, hold_positions)

            if onset:
                results[direction].append(onset)
                print(f"   trial {trial+1} dir={direction:+d}: onset at "
                      f"|d_raw|={onset['raw_disp']:.4f} rad, "
                      f"breakaway {onset['tff_break']:.2f} Nm")
            else:
                print(f"   trial {trial+1} dir={direction:+d}: no onset within "
                      f"{amp_j:.1f} Nm (raise --scale)")

    return_to_pose(driver, mid, j_idx, raw_p0, sim_p0, hold_positions)
    print("\n   lash summary (motor-frame rad before onset / breakaway Nm):")
    for d in (+1, -1):
        if results[d]:
            disp = np.array([r['raw_disp'] for r in results[d]])
            brk = np.array([r['tff_break'] for r in results[d]])
            print(f"     dir {d:+d}: disp {disp.mean():.4f} +- {disp.std():.4f}  "
                  f"breakaway {brk.mean():.2f} +- {brk.std():.2f}  (n={len(disp)})")
        else:
            print(f"     dir {d:+d}: no data")
    return results


# ══════════════════════════════════════════════════════════════════════
# [sat] locked PD saturation steps — fixture required
# ══════════════════════════════════════════════════════════════════════
def measure_saturation_steps(driver, mid, j_idx, jtype, bench0, log_writer,
                             hold_positions, scale):
    sim_p0 = bench0['sim_p0']
    kp_j, kd_j = SAT_KP_JOINT[jtype], SAT_KD_JOINT[jtype]
    kp_m = gain_joint_to_motor(kp_j, j_idx)
    kd_m = gain_joint_to_motor(kd_j, j_idx)
    cap_j = C.BURST_EFFORT[j_idx]

    print(f"\n== [sat] motor {mid}/{C.SIM_JOINT_NAMES[j_idx]} LOCKED — kp={kp_j:.0f} "
          f"joint-side, escalating {[f'{m*scale:.2f}x' for m in SAT_MULTS]} of "
          f"cap {cap_j:.1f}Nm, keypress between levels ==")

    results = []
    for mult in SAT_MULTS:
        eff_mult = mult * scale
        err_j = eff_mult * cap_j / kp_j
        wait_for_key(f"   next: {eff_mult:.2f}x cap (|kp*err|={eff_mult*cap_j:.1f}Nm, "
                     f"err={err_j:.3f}rad). Check the fixture, press 'g' "
                     f"(SPACE panic).", keys=('g',),
                     tick=hold_tick(driver, mid, j_idx, sim_p0, hold_positions))
        for sign in (+1, -1):
            # everything relative to the fixture pose — no recenter in locked mode
            _, raw_hold = settle_tau0(driver, mid, j_idx, sim_p0, hold_positions)
            sim_all = compute_sim(_read_all_raw(driver))
            sim_start = float(sim_all[j_idx])
            target_j = sim_start + sign * err_j
            raw_target = raw_target_for_joint(target_j, j_idx, sim_all)

            tau_trace, disp_trace = [], []
            t0 = time.monotonic()
            while time.monotonic() - t0 < SAT_STEP_RAMP_S + SAT_HOLD_S:
                ls = time.monotonic()
                a = min(1.0, (ls - t0) / SAT_STEP_RAMP_S)   # ramped onset, no torque step
                p_cmd = raw_hold + (raw_target - raw_hold) * a
                _send_test(driver, mid, p_cmd, 0.0, kp_m, kd_m, 0.0,
                           hold_positions=hold_positions)
                poll_panic()
                sim_p, sim_v, temp, _ = _check_safety_rel(
                    driver, mid, j_idx, sim_p0, hold_positions)
                # fixture-failure gates (tighter than the generic 5.0 rad/s)
                if abs(sim_v) > LOCKED_VEL_GATE:
                    raise RuntimeError(f"LOCKED motion gate: |v|={sim_v:.2f} — "
                                       "fixture failed?")
                if abs(sim_p - sim_start) > LOCKED_SLIP_ABORT:
                    raise RuntimeError(f"LOCKED slip abort: joint moved "
                                       f"{sim_p - sim_start:+.3f} rad")
                tau_j = _get_state_joint(driver, mid, j_idx)[2]
                tau_trace.append(tau_j)
                disp_trace.append(sim_p - sim_start)
                log_writer.writerow(['sat_step', sign, f"{eff_mult:.2f}",
                                     f"{ls:.6f}", f"{target_j:.4f}",
                                     f"{sim_p:.4f}", f"{sim_v:.4f}",
                                     f"{tau_j:.4f}", f"{sim_p - sim_start:.5f}",
                                     f"{temp:.2f}"])
                _sleep_until(ls + LOOP_DT)

            tau = np.array(tau_trace)
            n = len(tau)
            late = tau[int(0.6 * n):]
            creep = disp_trace[-1] - disp_trace[min(n - 1, int(0.2 * n))]
            print(f"   {eff_mult:.2f}x dir={sign:+d}: tau_late={late.mean():+.2f}"
                  f"+-{late.std():.3f}  peak|tau|={np.abs(tau).max():.2f}  "
                  f"creep={creep:+.4f} rad")
            results.append(dict(mult=eff_mult, sign=sign,
                                tau_late=float(late.mean()),
                                tau_late_std=float(late.std()),
                                tau_peak=float(np.abs(tau).max()),
                                creep=float(creep)))
    return results


# ══════════════════════════════════════════════════════════════════════
# [tff] locked open-loop ramp — fixture required
# ══════════════════════════════════════════════════════════════════════
def measure_tff_ramp(driver, mid, j_idx, jtype, bench0, log_writer,
                     hold_positions, scale):
    sim_p0 = bench0['sim_p0']
    rail_j = abs(torque_motor_to_joint(RS01_TFF_MAX, j_idx))
    ramp_cap_j = TFF_RAMP_RAIL_MULT * rail_j * scale

    print(f"\n== [tff] motor {mid}/{C.SIM_JOINT_NAMES[j_idx]} LOCKED — open-loop "
          f"ramp to +-{ramp_cap_j:.1f}Nm joint over {TFF_RAMP_DURATION:.1f}s "
          f"(decode rail +-{rail_j:.2f}Nm) ==")

    results = []
    for direction in (+1, -1):
        wait_for_key(f"   next: dir {direction:+d}. Check the fixture, press 'g' "
                     f"(SPACE panic).", keys=('g',),
                     tick=hold_tick(driver, mid, j_idx, sim_p0, hold_positions))
        tau0, _ = settle_tau0(driver, mid, j_idx, sim_p0, hold_positions)
        sim_start = float(compute_sim(_read_all_raw(driver))[j_idx])

        t_ff_trace, tau_trace = [], []
        t0 = time.monotonic()
        while time.monotonic() - t0 < TFF_RAMP_DURATION:
            ls = time.monotonic()
            t_ff_j = tau0 + direction * ramp_cap_j * ((ls - t0) / TFF_RAMP_DURATION)
            _send_test(driver, mid, 0.0, 0.0, 0.0, STATIC_RAMP_KD_MOTOR,
                       torque_joint_to_motor(t_ff_j, j_idx),
                       hold_positions=hold_positions)
            poll_panic()
            sim_p, sim_v, temp, _ = _check_safety_rel(
                driver, mid, j_idx, sim_p0, hold_positions)
            if abs(sim_v) > LOCKED_VEL_GATE:
                raise RuntimeError(f"LOCKED motion gate: |v|={sim_v:.2f} — "
                                   "fixture failed?")
            if abs(sim_p - sim_start) > LOCKED_SLIP_ABORT:
                raise RuntimeError(f"LOCKED slip abort: joint moved "
                                   f"{sim_p - sim_start:+.3f} rad")
            tau_j = _get_state_joint(driver, mid, j_idx)[2]
            t_ff_trace.append(t_ff_j)
            tau_trace.append(tau_j)
            log_writer.writerow(['tff_ramp', direction, 0, f"{ls:.6f}",
                                 f"{t_ff_j:.4f}", f"{sim_p:.4f}", f"{sim_v:.4f}",
                                 f"{tau_j:.4f}", f"{sim_p - sim_start:.5f}",
                                 f"{temp:.2f}"])
            _sleep_until(ls + LOOP_DT)

        t_ff = np.array(t_ff_trace)
        tau = np.array(tau_trace)
        X = np.column_stack([t_ff, np.ones_like(t_ff)])
        coeffs, *_ = np.linalg.lstsq(X, tau, rcond=None)
        pred = X @ coeffs
        ss_t = float(np.sum((tau - tau.mean()) ** 2)) or 1e-9
        r2 = 1.0 - float(np.sum((tau - pred) ** 2)) / ss_t
        peak = float(np.abs(tau).max())
        at_rail = abs(peak - rail_j) < 0.05
        print(f"   dir={direction:+d}: gain={coeffs[0]:.3f} R2={r2:.3f} "
              f"peak|tau|={peak:.2f}"
              + ("  (AT DECODE RAIL — ambiguous)" if at_rail else ""))
        results.append(dict(direction=direction, gain=float(coeffs[0]),
                            r2=float(r2), tau_peak=peak, at_rail=at_rail))
    return results


# ══════════════════════════════════════════════════════════════════════
# Plan printing / comparison
# ══════════════════════════════════════════════════════════════════════
def build_plan(motor_names):
    plan = []
    for name in motor_names:
        mid = BENCH_MOTORS[name]
        j_idx = MOTOR_TO_JOINT_IDX[mid]
        jtype = 'knee' if _is_knee(j_idx) else 'hip_pitch'
        plan.append(dict(name=name, mid=mid, j_idx=j_idx, jtype=jtype,
                         cap_j=C.BURST_EFFORT[j_idx]))
    return plan


def print_plan(plan, phases, scale, locked):
    print(f"mode: {'LOCKED (fixture required)' if locked else 'free-hanging'}   "
          f"scale: {scale:.2f}   phases: {phases}")
    for e in plan:
        print(f"\n{e['name']}  (motor {e['mid']}, joint "
              f"{C.SIM_JOINT_NAMES[e['j_idx']]}, burst cap {e['cap_j']:.1f}Nm)")
        print(f"  pos window: armed pose +-{BENCH_POS_HARD_REL} rad "
              f"(measured at arm time — absolute profiles are stale)")
        if 'lash' in phases:
            amp = LASH_AMP_JOINT[e['jtype']] * scale
            print(f"  [lash] ramp from hold torque to +-{amp:.1f}Nm at "
                  f"{LASH_RAMP_RATE:.2f}Nm/s, stop at onset "
                  f"({LASH_VEL_ONSET_JOINT:.2f}rad/s over "
                  f"{LASH_VEL_WINDOW_S*1000:.0f}ms), {LASH_N_TRIALS} trials/dir")
        if 'sat' in phases:
            kp_j = SAT_KP_JOINT[e['jtype']]
            for m in SAT_MULTS:
                em = m * scale
                print(f"  [sat] {em:.2f}x cap: |kp*err|={em*e['cap_j']:.1f}Nm, "
                      f"err={em*e['cap_j']/kp_j:.3f}rad, hold {SAT_HOLD_S:.1f}s "
                      f"(onset ramp {SAT_STEP_RAMP_S:.2f}s)")
        if 'tff' in phases:
            rail_j = abs(torque_motor_to_joint(RS01_TFF_MAX, e['j_idx']))
            print(f"  [tff] ramp hold->{TFF_RAMP_RAIL_MULT*rail_j*scale:.1f}Nm "
                  f"over {TFF_RAMP_DURATION:.1f}s (rail +-{rail_j:.2f}Nm)")
    print(f"\nsafety: arm=limp+{ARM_RAMP_S:.0f}s ramp on 'g'   SPACE/'p' panic   "
          f"pos +-{BENCH_POS_HARD_REL}rad rel   vel_hard={VEL_HARD_LIMIT_JOINT:.1f}rad/s   "
          f"temp={TEMP_LIMIT_C:.0f}C   drift={HOLD_DRIFT_LIMIT_RAD:.2f}rad"
          + (f"   locked gates: vel {LOCKED_VEL_GATE:.1f}rad/s, "
             f"slip {LOCKED_SLIP_ABORT:.2f}rad" if locked else ""))


def print_comparison(all_results):
    print(f"\n{'='*68}\n  COMPARISON — hip_pitch1 (direct) vs knee1 (belt)\n{'='*68}")
    both = {}
    for name in ('hip_pitch1', 'knee1'):
        lash = all_results.get(name, {}).get('lash')
        if not lash:
            continue
        disp = np.array([r['raw_disp'] for d in (+1, -1) for r in lash[d]])
        if len(disp):
            both[name] = disp
            print(f"  {name}: lash width {disp.mean():.4f} +- {disp.std():.4f} rad "
                  f"(motor-frame, n={len(disp)})")
    if len(both) == 2:
        hp, kn = both['hip_pitch1'], both['knee1']
        delta = kn.mean() - hp.mean()
        sig = kn.mean() > hp.mean() + 2 * hp.std()
        print(f"  delta (knee - hip_pitch): {delta:+.4f} rad  "
              f"{'-> BELT-SPECIFIC LASH CONFIRMED' if sig else '-> inconclusive at 2-sigma'}")


# ══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--motor', choices=['hip_pitch1', 'knee1', 'both'],
                    default='both')
    ap.add_argument('--locked', action='store_true',
                    help='joint is blocked by a rigid fixture: unlocks [sat] and '
                         '[tff]. NEVER hold the leg by hand.')
    ap.add_argument('--scale', type=float, default=0.5,
                    help='multiplier on all torque amplitudes (default 0.5; '
                         'use 1.0 only after a clean shakedown run)')
    ap.add_argument('--skip', choices=['sat', 'tff', 'lash'], action='append',
                    default=[], help='skip a test phase (repeatable)')
    ap.add_argument('--dry-run', action='store_true',
                    help='print the trial plan and safety limits; touch no hardware')
    args = ap.parse_args()

    motor_names = ['hip_pitch1', 'knee1'] if args.motor == 'both' else [args.motor]
    if args.locked:
        # lash needs a FREE joint (a blocked joint never produces a velocity
        # onset, and its recenter would fight the fixture) — locked runs
        # sat/tff only. Run lash free-hanging first, then strap and re-run.
        phases = [p for p in ('sat', 'tff') if p not in args.skip]
        if 'lash' not in args.skip:
            print("[i] [lash] is skipped in --locked mode (needs a free joint). "
                  "Run it free-hanging first.")
    else:
        phases = [p for p in ('lash',) if p not in args.skip]
        dropped = [p for p in ('sat', 'tff') if p not in args.skip]
        if dropped:
            print(f"[i] {dropped} need --locked (deep saturation is impossible "
                  f"free-hanging: 17Nm on a hanging shank trips the vel abort "
                  f"in <50ms). Running free-hanging phases only.")
    plan = build_plan(motor_names)

    print("=== saturation_bench.py — Phase C (hip_pitch1 vs knee1) ===")
    print_plan(plan, phases, args.scale, args.locked)

    if args.dry_run:
        print("\n--dry-run: no hardware touched.")
        return
    if not phases:
        print("nothing to run.")
        return

    if args.locked:
        print("\n*** LOCKED MODE: the joint under test must be blocked by a "
              "RIGID FIXTURE (strap/clamp to the stand). NEVER a hand or foot. "
              "Commanded torque reaches the 17 Nm motor cap. ***")
        if input("Type LOCKED to confirm the fixture is on: ").strip() != "LOCKED":
            print("not confirmed — exiting.")
            return
    print("\n*** Robot hanging, feet clear. Motors stay LIMP until you arm. ***")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"saturation_bench_{stamp}.csv"
    driver = make_driver(list(C.MOTOR_IDS))
    f = open(csv_path, 'w', newline='')
    w = csv.writer(f)
    w.writerow(['phase', 'direction', 'trial', 't', 'cmd_joint', 'sim_pos',
                'sim_vel', 'tau_joint', 'raw_disp', 'temp'])

    all_results = {}
    setup_raw_keys()
    try:
        driver.enable_all()
        for e in plan:
            mid, j_idx, jtype = e['mid'], e['j_idx'], e['jtype']
            print(f"\n{'='*68}\n  BENCH: {e['name']} (motor {mid})\n{'='*68}")
            hold_positions, raw_p0, sim_p0 = arm_interactive(driver, mid, j_idx)
            bench0 = dict(raw_p0=raw_p0, sim_p0=sim_p0)

            res = {}
            runners = {'lash': measure_lash, 'sat': measure_saturation_steps,
                       'tff': measure_tff_ramp}
            for ph in phases:
                wait_for_key(f"  press 'g' to start [{ph}] on {e['name']} "
                             f"(SPACE panic, 'q' quit)", keys=('g',),
                             tick=hold_tick(driver, mid, j_idx, sim_p0,
                                            hold_positions))
                res[ph] = runners[ph](driver, mid, j_idx, jtype, bench0, w,
                                      hold_positions, args.scale)
            all_results[e['name']] = res
    except BenchPanic as e:
        print(f"\n[!] PANIC: {e}")
    except KeyboardInterrupt:
        print("\n[!] Interrupted")
    except RuntimeError as e:
        print(f"\n[!] Safety abort: {e}")
    except Exception as e:
        print(f"\n[!] Error: {e}")
    finally:
        safe_stop(driver)       # brake -> limp -> disable, never limp mid-swing
        driver.close()
        f.close()
        restore_keys()

    print(f"\nLog written to {csv_path}")
    if 'hip_pitch1' in all_results and 'knee1' in all_results:
        print_comparison(all_results)


if __name__ == '__main__':
    main()
