"""
friction_id.py — Static, Coulomb, viscous friction identification per
                 motor on the ASSEMBLED bipedal robot.   [VERSION 3]

═══════════════════════════════════════════════════════════════════════
KEY CHANGES vs v2
───────────────────────────────────────────────────────────────────────
  v3.1 STATIC: breakaway threshold raised past backlash range, small Kd
       added during torque ramp to damp oscillation, and sustained
       average-velocity criterion added so backlash settling / shake
       no longer counts as breakaway.  Both gates (displacement past
       lash AND sustained velocity in commanded direction) must trip.
  v3.2 SWEEP:  max knee/hip velocity backed off from 6 → 4 rad/s
       (joint-side). Above this the motor couldn't decelerate before
       hitting the hard limit in the available window.
  v3.3 PARTIAL RESULTS: if velocity sweep raises, partial data is now
       still passed to fit_friction (previously the unbinding of the
       tuple assignment caused all data to be lost).

(v2 changes retained: hold non-tested motors stiff, raw-motor breakaway
 detection, quasi-static gravity profile, central-band sampling,
 hold-drift safety check.)

Operates in JOINT-SIDE coordinates for the friction parameter output
(ready for DCMotor cfg). Recentering uses inverse compute_sim so it
works regardless of motor-zero / joint-zero offset.
"""

import csv
import time
from datetime import datetime

import numpy as np

import config as C
from robstride_socketcan_driver import make_driver


# ══════════════════════════════════════════════════════════════════════
# EDIT — which motor to test
# ══════════════════════════════════════════════════════════════════════
MOTOR_UNDER_TEST = 102

# ══════════════════════════════════════════════════════════════════════
# EDIT — per-motor profile (JOINT-SIDE)
# ──────────────────────────────────────────────────────────────────────
# Sweep velocities roughly DOUBLED on hip/knee vs v1 to boost viscous SNR.
# Roll windows are too narrow to fit a long high-v leg, kept conservative.
# ══════════════════════════════════════════════════════════════════════
MOTOR_PROFILES = {
    # Roll
    103: {'pos_min': -0.20, 'pos_max': +0.20, 'pos_hard': 0.25,
          'sweep': [0.3, 0.6, 1.0, 1.5, 2.0, 2.5], 'static_max': 1.0},
    106: {'pos_min': -0.20, 'pos_max': +0.20, 'pos_hard': 0.25,
          'sweep': [0.3, 0.6, 1.0, 1.5, 2.0, 2.5], 'static_max': 1.0},
    # Hip
    101: {'pos_min': -0.40, 'pos_max': +0.40, 'pos_hard': 0.50,
          'sweep': [0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0],
          'static_max': 1.5},
    104: {'pos_min': -0.40, 'pos_max': +0.40, 'pos_hard': 0.50,
          'sweep': [0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0],
          'static_max': 1.5},
    # Knee — capped at 4 rad/s; above this, 1.9 rad window is too narrow
    # to decelerate before hitting hard limit.
    102: {'pos_min': -1.20, 'pos_max': +0.70, 'pos_hard': 1.80,
          'sweep': [0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0],
          'static_max': 1.5},
    105: {'pos_min': -1.20, 'pos_max': +0.70, 'pos_hard': 1.80,
          'sweep': [0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0],
          'static_max': 1.5},
}

# ══════════════════════════════════════════════════════════════════════
# EDIT — hard limits and gains (joint-side unless noted)
# ══════════════════════════════════════════════════════════════════════
VEL_HARD_LIMIT_JOINT  = 5.0     # was 4.0 — slight bump for new max sweep 4
TEMP_LIMIT_C          = 70.0
HOLD_KD_JOINT         = 0.5

# Non-tested-motor position-hold gains (motor-side). MUST be stiff enough
# that reaction torque from the motor under test doesn't move them.
OTHER_HOLD_KP_MOTOR   = 15.0
OTHER_HOLD_KD_MOTOR   = 1.0
HOLD_DRIFT_LIMIT_RAD  = 0.15    # abort if any held motor drifts this far

# Recenter ramp (motor-side gains — homing of the tested motor)
RECENTER_KP_MOTOR     = 4.0
RECENTER_KD_MOTOR     = 2.0
RECENTER_RAMP_S       = 2.5

# Safe-startup (brake) parameters (motor-side)
BRAKE_KD_MOTOR        = 3.0
BRAKE_DURATION_S      = 0.5
POST_BRAKE_HOLD_S     = 0.3
POST_BRAKE_KP_MOTOR   = 3.0
POST_BRAKE_KD_MOTOR   = 2.0

# ══════════════════════════════════════════════════════════════════════
# EDIT — timing
# ══════════════════════════════════════════════════════════════════════
CONTROL_HZ                 = 200
RAMP_DURATION              = 4.0
VEL_BREAK_THRESHOLD_JOINT  = 0.08
SETTLE_TIME                = 0.3
AVERAGE_TIME_PER_DIR       = 0.8
INTER_REST                 = 0.4
N_STATIC_TRIALS            = 5
MAX_PINGPONG_CYCLES        = 8
STATIC_HOLD_KP_MOTOR  = 4.0
STATIC_HOLD_KD_MOTOR  = 2.0
STATIC_HOLD_SETTLE_S  = 0.6
STATIC_GRAV_AVG_S     = 0.25

# v3: torque-ramp damping. Small Kd during the open-loop torque ramp
# kills oscillation in backlash without biasing the breakaway torque
# (Kd contributes zero at v=0, so the threshold for sliding is
# unchanged; it only damps motion AFTER breakaway, giving a clean
# terminal velocity to detect).
STATIC_RAMP_KD_MOTOR  = 0.3

# v3: breakaway gates. BOTH must trip simultaneously:
#   (a) raw motor displacement > BREAKAWAY_DISP_RAD (joint-side, converted)
#       — past linkage / gearbox backlash, not just into it.
#   (b) running-average velocity over BREAKAWAY_VEL_WINDOW_S in the
#       commanded direction exceeds BREAKAWAY_VEL_AVG_JOINT — rules out
#       symmetric oscillation / shake whose mean is ~0.
BREAKAWAY_DISP_RAD       = 0.5  # was 0.015 — past typical backlash
BREAKAWAY_VEL_WINDOW_S   = 0.10  # rolling-mean window
BREAKAWAY_VEL_AVG_JOINT  = 0.10  # rad/s sustained mean in commanded dir

# Gravity profile sweep parameters
GRAVITY_SWEEP_DURATION_S = 12.0  # very slow → quasi-static
GRAVITY_SAMPLE_INTERVAL  = 0.05  # ≈20 Hz logging
# ══════════════════════════════════════════════════════════════════════

LOOP_DT = 1.0 / CONTROL_HZ
MOTOR_TO_JOINT_IDX = {mid: i for i, mid in enumerate(C.MOTOR_IDS)}


# ──────────────────────────────────────────────────────────────────────
# Coordinate transforms — unchanged from v1
# ──────────────────────────────────────────────────────────────────────
def _is_knee(joint_idx):
    return joint_idx in (2, 5)


def _hip_idx_for_knee(joint_idx):
    return 1 if joint_idx == 2 else 4


def compute_sim(raw):
    sign = C.JOINT_SIGN
    off = C.JOINT_OFFSET
    gear = C.KNEE_GEAR_RATIO
    coupling = C.KNEE_HIP_COUPLING
    sim = [raw[i] * sign[i] + off[i] for i in range(6)]
    sim[2] = raw[2] * sign[2] * gear - sim[1] * coupling + off[2]
    sim[5] = raw[5] * sign[5] * gear - sim[4] * coupling + off[5]
    return sim


def vel_joint_to_motor(v_j, j_idx):
    s = C.JOINT_SIGN[j_idx]
    return v_j * s / C.KNEE_GEAR_RATIO if _is_knee(j_idx) else v_j * s


def vel_motor_to_joint(v_m, j_idx):
    s = C.JOINT_SIGN[j_idx]
    return v_m * s * C.KNEE_GEAR_RATIO if _is_knee(j_idx) else v_m * s


def torque_joint_to_motor(t_j, j_idx):
    s = C.JOINT_SIGN[j_idx]
    return t_j * s * C.KNEE_GEAR_RATIO if _is_knee(j_idx) else t_j * s


def torque_motor_to_joint(t_m, j_idx):
    s = C.JOINT_SIGN[j_idx]
    return t_m * s / C.KNEE_GEAR_RATIO if _is_knee(j_idx) else t_m * s


def kd_joint_to_motor(kd_j, j_idx):
    return kd_j * C.KNEE_GEAR_RATIO ** 2 if _is_knee(j_idx) else kd_j


def raw_target_for_joint(target_joint, j_idx, sim_all):
    """Inverse of compute_sim: raw motor position that gives target_joint."""
    sign = C.JOINT_SIGN[j_idx]
    off = C.JOINT_OFFSET[j_idx]
    if _is_knee(j_idx):
        gear = C.KNEE_GEAR_RATIO
        coupling = C.KNEE_HIP_COUPLING
        sim_hip = sim_all[_hip_idx_for_knee(j_idx)]
        return (target_joint - off + sim_hip * coupling) / (sign * gear)
    return (target_joint - off) * sign


def _breakaway_disp_motor(j_idx):
    """Convert joint-side breakaway threshold to motor-side."""
    if _is_knee(j_idx):
        return BREAKAWAY_DISP_RAD / C.KNEE_GEAR_RATIO
    return BREAKAWAY_DISP_RAD


# ──────────────────────────────────────────────────────────────────────
# Driver helpers — MODIFIED to support holding non-tested motors
# ──────────────────────────────────────────────────────────────────────
def _read_all_raw(driver):
    return [driver.get_state(m)[0] for m in C.MOTOR_IDS]


def _capture_hold_positions(driver):
    return {m: driver.get_state(m)[0] for m in C.MOTOR_IDS}


def _send_test(driver, mid, p, v, kp, kd, tff,
               hold_positions=None,
               hold_kp=OTHER_HOLD_KP_MOTOR,
               hold_kd=OTHER_HOLD_KD_MOTOR):
    """Command motor `mid` with (p,v,kp,kd,tff). Other motors are either
    held at hold_positions[m] (when given) or commanded limp (legacy)."""
    cmds = []
    for m in C.MOTOR_IDS:
        if m == mid:
            cmds.append((m, p, v, kp, kd, tff))
        elif hold_positions is not None:
            cmds.append((m, hold_positions[m], 0.0, hold_kp, hold_kd, 0.0))
        else:
            cmds.append((m, 0.0, 0.0, 0.0, 0.0, 0.0))
    driver.send_batch(cmds)


def _send_all(driver, builder):
    cmds = [(m, *builder(m)) for m in C.MOTOR_IDS]
    driver.send_batch(cmds)


def _check_safety(driver, mid, j_idx, pos_hard,
                  hold_positions=None, skip_vel=False):
    raw_all = _read_all_raw(driver)
    sim_all = compute_sim(raw_all)
    sim_p = sim_all[j_idx]
    _, raw_v, _, temp = driver.get_state(mid)
    sim_v = vel_motor_to_joint(raw_v, j_idx)

    if abs(sim_p) > pos_hard:
        raise RuntimeError(f"HARD position: sim_p={sim_p:+.3f} > {pos_hard}")
    if not skip_vel and abs(sim_v) > VEL_HARD_LIMIT_JOINT:
        raise RuntimeError(f"HARD velocity: sim_v={sim_v:+.3f}")
    if temp > TEMP_LIMIT_C:
        raise RuntimeError(f"Over-temp: {temp:.1f} °C")

    # NEW: held-motor drift check
    if hold_positions is not None:
        for i, m in enumerate(C.MOTOR_IDS):
            if m == mid:
                continue
            drift = abs(raw_all[i] - hold_positions[m])
            if drift > HOLD_DRIFT_LIMIT_RAD:
                raise RuntimeError(
                    f"HOLD DRIFT: motor {m} drifted {drift:.3f} rad "
                    f"(>{HOLD_DRIFT_LIMIT_RAD}). Stiffen OTHER_HOLD_K*.")
    return sim_p, sim_v, temp, sim_all


def _get_state_joint(driver, mid, j_idx):
    raw_all = _read_all_raw(driver)
    sim_all = compute_sim(raw_all)
    sim_p = sim_all[j_idx]
    _, raw_v, raw_t, temp = driver.get_state(mid)
    return (sim_p, vel_motor_to_joint(raw_v, j_idx),
            torque_motor_to_joint(raw_t, j_idx), temp)


def _sleep_until(t):
    r = t - time.monotonic()
    if r > 0:
        time.sleep(r)


# ──────────────────────────────────────────────────────────────────────
# Safe startup — returns captured hold_positions
# ──────────────────────────────────────────────────────────────────────
def _safe_startup(driver, mid, j_idx, pos_hard):
    print("  [startup] braking...")
    end = time.monotonic() + BRAKE_DURATION_S
    while time.monotonic() < end:
        ls = time.monotonic()
        _send_all(driver, lambda m: (0.0, 0.0, 0.0, BRAKE_KD_MOTOR, 0.0))
        _check_safety(driver, mid, j_idx, pos_hard, skip_vel=True)
        _sleep_until(ls + LOOP_DT)

    print("  [startup] holding at current position...")
    current = _capture_hold_positions(driver)
    end = time.monotonic() + POST_BRAKE_HOLD_S
    while time.monotonic() < end:
        ls = time.monotonic()
        _send_all(driver, lambda m: (current[m], 0.0,
                                     POST_BRAKE_KP_MOTOR,
                                     POST_BRAKE_KD_MOTOR, 0.0))
        _check_safety(driver, mid, j_idx, pos_hard)
        _sleep_until(ls + LOOP_DT)

    sim_all = compute_sim(_read_all_raw(driver))
    print(f"  [startup] OK. sim_pos[{j_idx}]={sim_all[j_idx]:+.3f} rad")
    return current


# ──────────────────────────────────────────────────────────────────────
# Ramped position move — accepts hold_positions
# ──────────────────────────────────────────────────────────────────────
def _drive_to_raw(driver, mid, j_idx, raw_target, pos_hard,
                  ramp_duration, hold_positions, tol=0.04):
    raw_start = driver.get_state(mid)[0]
    t0 = time.monotonic()
    while True:
        ls = time.monotonic()
        elapsed = ls - t0
        alpha = min(1.0, elapsed / ramp_duration)
        p_des = raw_start + (raw_target - raw_start) * alpha
        _send_test(driver, mid, p_des, 0.0,
                   RECENTER_KP_MOTOR, RECENTER_KD_MOTOR, 0.0,
                   hold_positions=hold_positions)
        _check_safety(driver, mid, j_idx, pos_hard, hold_positions)
        p_raw = driver.get_state(mid)[0]
        if alpha >= 1.0:
            if abs(p_raw - raw_target) < tol:
                break
            if elapsed > ramp_duration + 1.5:
                break
        _sleep_until(ls + LOOP_DT)
    # Brief settle
    end = time.monotonic() + 0.15
    while time.monotonic() < end:
        ls = time.monotonic()
        _send_test(driver, mid, raw_target, 0.0,
                   RECENTER_KP_MOTOR, RECENTER_KD_MOTOR, 0.0,
                   hold_positions=hold_positions)
        _check_safety(driver, mid, j_idx, pos_hard, hold_positions)
        _sleep_until(ls + LOOP_DT)


def _recenter(driver, mid, j_idx, profile, hold_positions,
              ramp_duration=None):
    if ramp_duration is None:
        ramp_duration = RECENTER_RAMP_S
    target_joint = 0.5 * (profile['pos_min'] + profile['pos_max'])
    sim_all = compute_sim(_read_all_raw(driver))
    raw_target = raw_target_for_joint(target_joint, j_idx, sim_all)
    _drive_to_raw(driver, mid, j_idx, raw_target,
                  profile['pos_hard'], ramp_duration, hold_positions)


def _hold_limp(driver, mid, j_idx, duration, pos_hard, hold_positions):
    """Motor under test goes limp; OTHER motors continue to be held."""
    end = time.monotonic() + duration
    while time.monotonic() < end:
        ls = time.monotonic()
        _send_test(driver, mid, 0, 0, 0, 0, 0,
                   hold_positions=hold_positions)
        _check_safety(driver, mid, j_idx, pos_hard, hold_positions)
        _sleep_until(ls + LOOP_DT)


# ──────────────────────────────────────────────────────────────────────
# NEW: Gravity profile — slow bidirectional sweep
# ──────────────────────────────────────────────────────────────────────
def measure_gravity_profile(driver, mid, j_idx, profile, log_writer,
                            hold_positions):
    pos_min, pos_max = profile['pos_min'], profile['pos_max']
    pos_hard = profile['pos_hard']

    print(f"\n── Gravity profile (motor {mid} / "
          f"{C.SIM_JOINT_NAMES[j_idx]}) ──")
    print(f"   Slow sweep across [{pos_min:+.2f}, {pos_max:+.2f}] "
          f"in {GRAVITY_SWEEP_DURATION_S:.0f} s each way")

    samples = []  # (sim_p, sim_v, tau_j)
    sim_all = compute_sim(_read_all_raw(driver))
    raw_a = raw_target_for_joint(pos_min, j_idx, sim_all)
    raw_b = raw_target_for_joint(pos_max, j_idx, sim_all)

    # Drive to pos_min start
    _drive_to_raw(driver, mid, j_idx, raw_a, pos_hard,
                  ramp_duration=2.0, hold_positions=hold_positions)

    for direction_label, raw_from, raw_to in [
        ('gravity_fwd', raw_a, raw_b),
        ('gravity_bwd', raw_b, raw_a),
    ]:
        t0 = time.monotonic()
        last_log = 0.0
        while True:
            ls = time.monotonic()
            elapsed = ls - t0
            alpha = min(1.0, elapsed / GRAVITY_SWEEP_DURATION_S)
            p_des = raw_from + (raw_to - raw_from) * alpha
            _send_test(driver, mid, p_des, 0.0,
                       RECENTER_KP_MOTOR, RECENTER_KD_MOTOR, 0.0,
                       hold_positions=hold_positions)
            sim_p, sim_v, _, _ = _check_safety(
                driver, mid, j_idx, pos_hard, hold_positions)
            _, _, tau_j, temp = _get_state_joint(driver, mid, j_idx)

            if (elapsed - last_log) >= GRAVITY_SAMPLE_INTERVAL:
                samples.append((sim_p, sim_v, tau_j))
                log_writer.writerow([
                    direction_label, 0, 0,
                    f"{ls:.6f}", f"{p_des:.4f}",
                    f"{sim_p:.4f}", f"{sim_v:.4f}", f"{tau_j:.4f}",
                    f"{temp:.2f}",
                ])
                last_log = elapsed

            if alpha >= 1.0:
                break
            _sleep_until(ls + LOOP_DT)

    if len(samples) < 20:
        print(f"   too few samples ({len(samples)}); gravity = 0")
        return (0.0, 0.0, 0.0)

    p = np.array([s[0] for s in samples])
    v = np.array([s[1] for s in samples])
    t = np.array([s[2] for s in samples])

    # Fit τ = a·sin(p) + c·cos(p) + e·sign(v) + d
    # The sign(v) term absorbs the small Coulomb component at low v,
    # leaving sin/cos clean for gravity.
    sv = np.sign(v)
    sv[sv == 0] = 1.0  # avoid degenerate rows
    X = np.column_stack([np.sin(p), np.cos(p), sv, np.ones_like(p)])
    coeffs, *_ = np.linalg.lstsq(X, t, rcond=None)
    a_g, c_g, e_g, d_g = coeffs

    pred = X @ coeffs
    ss_r = float(np.sum((t - pred) ** 2))
    ss_t = float(np.sum((t - t.mean()) ** 2)) or 1e-12
    r2 = 1.0 - ss_r / ss_t

    print(f"   τ_g(θ) ≈ {a_g:+.3f}·sin(θ) + {c_g:+.3f}·cos(θ) "
          f"+ {d_g:+.3f}   R²={r2:.3f}   n={len(samples)}")
    print(f"   Coulomb-at-low-v nuisance e = {e_g:+.3f} Nm "
          f"(not used downstream)")
    return (float(a_g), float(c_g), float(d_g))


# ──────────────────────────────────────────────────────────────────────
# STATIC FRICTION — raw-motor breakaway + hold others
# ──────────────────────────────────────────────────────────────────────
def measure_static(driver, mid, j_idx, profile, log_writer, hold_positions):
    static_max = profile['static_max']
    pos_hard   = profile['pos_hard']
    breakaway_motor = _breakaway_disp_motor(j_idx)

    print(f"\n── Static friction (motor {mid} / "
          f"{C.SIM_JOINT_NAMES[j_idx]}) ──")
    print(f"   Ramp τ_grav → τ_grav ± {static_max:.2f} Nm "
          f"over {RAMP_DURATION:.1f} s (joint-side)")
    print(f"   Breakaway: |Δraw| > {breakaway_motor:.4f} rad (motor-side)")

    results = {+1: [], -1: []}

    for direction in (+1, -1):
        for trial in range(N_STATIC_TRIALS):
            _recenter(driver, mid, j_idx, profile, hold_positions,
                      ramp_duration=1.2)

            # PHASE A: position-hold settle
            raw_hold = driver.get_state(mid)[0]
            end = time.monotonic() + STATIC_HOLD_SETTLE_S
            while time.monotonic() < end:
                ls = time.monotonic()
                _send_test(driver, mid, raw_hold, 0.0,
                           STATIC_HOLD_KP_MOTOR, STATIC_HOLD_KD_MOTOR, 0.0,
                           hold_positions=hold_positions)
                _check_safety(driver, mid, j_idx, pos_hard, hold_positions)
                _sleep_until(ls + LOOP_DT)

            # PHASE B: measure τ_grav at this θ under position hold
            tau_g_samples = []
            end = time.monotonic() + STATIC_GRAV_AVG_S
            while time.monotonic() < end:
                ls = time.monotonic()
                _send_test(driver, mid, raw_hold, 0.0,
                           STATIC_HOLD_KP_MOTOR, STATIC_HOLD_KD_MOTOR, 0.0,
                           hold_positions=hold_positions)
                _, _, tau_j_meas, _ = _get_state_joint(driver, mid, j_idx)
                tau_g_samples.append(tau_j_meas)
                _check_safety(driver, mid, j_idx, pos_hard, hold_positions)
                _sleep_until(ls + LOOP_DT)
            tau_grav_j = float(np.median(tau_g_samples))

            # PHASE C: open-loop torque ramp (with small Kd damping in v3).
            # Breakaway requires BOTH:
            #   (a) raw motor displacement past lash
            #   (b) sustained mean velocity in the commanded direction
            # (Either alone is insufficient: (a) alone catches backlash
            # settling, (b) alone could catch acceleration during slip
            # within lash. Together they catch real breakaway only.)
            raw_p_start = driver.get_state(mid)[0]
            vel_window  = []
            window_n    = int(BREAKAWAY_VEL_WINDOW_S * CONTROL_HZ)
            peak_disp   = 0.0   # diagnostic

            t0 = time.monotonic()
            breakaway = None
            while True:
                ls = time.monotonic()
                elapsed = ls - t0
                if elapsed >= RAMP_DURATION:
                    break

                tau_ramp_j  = direction * static_max * (elapsed / RAMP_DURATION)
                tau_total_j = tau_grav_j + tau_ramp_j
                tau_m_cmd   = torque_joint_to_motor(tau_total_j, j_idx)

                # NOTE the STATIC_RAMP_KD_MOTOR — small damping kills
                # oscillation without biasing static-friction threshold
                # (since Kd*v = 0 at v = 0).
                _send_test(driver, mid, 0.0, 0.0, 0.0,
                           STATIC_RAMP_KD_MOTOR, tau_m_cmd,
                           hold_positions=hold_positions)
                sim_p, sim_v, _, _ = _check_safety(
                    driver, mid, j_idx, pos_hard, hold_positions)
                _, _, tau_j_meas, temp = _get_state_joint(driver, mid, j_idx)
                raw_p_now = driver.get_state(mid)[0]
                raw_disp = abs(raw_p_now - raw_p_start)
                peak_disp = max(peak_disp, raw_disp)

                # Update rolling velocity window
                vel_window.append(sim_v)
                if len(vel_window) > window_n:
                    vel_window.pop(0)

                log_writer.writerow([
                    'static', direction, trial,
                    f"{ls:.6f}", f"{tau_ramp_j:.4f}",
                    f"{sim_p:.4f}", f"{sim_v:.4f}", f"{tau_j_meas:.4f}",
                    f"{temp:.2f}",
                ])

                # Combined breakaway check
                if (raw_disp > breakaway_motor
                        and len(vel_window) == window_n):
                    avg_v = sum(vel_window) / window_n
                    if avg_v * direction > BREAKAWAY_VEL_AVG_JOINT:
                        breakaway = (tau_ramp_j, tau_grav_j, tau_j_meas)
                        break

                # Hard-stop backup: if velocity blows up entirely
                if abs(sim_v) > 0.5 * VEL_HARD_LIMIT_JOINT:
                    breakaway = (tau_ramp_j, tau_grav_j, tau_j_meas)
                    break

                _sleep_until(ls + LOOP_DT)

            # Re-settle under position hold
            end = time.monotonic() + 0.3
            while time.monotonic() < end:
                ls = time.monotonic()
                _send_test(driver, mid, raw_hold, 0.0,
                           STATIC_HOLD_KP_MOTOR, STATIC_HOLD_KD_MOTOR, 0.0,
                           hold_positions=hold_positions)
                _check_safety(driver, mid, j_idx, pos_hard, hold_positions)
                _sleep_until(ls + LOOP_DT)

            if breakaway is None:
                print(f"   dir={direction:+d} trial {trial+1}: "
                      f"NO BREAKAWAY  τ_grav={tau_grav_j:+.3f}  "
                      f"peak |Δraw|={peak_disp:.4f} rad "
                      f"(thresh {breakaway_motor:.4f}) — raise static_max")
            else:
                tr, tg, tm = breakaway
                print(f"   dir={direction:+d} trial {trial+1}: "
                      f"τ_s={tr:+.3f} Nm  "
                      f"τ_grav={tg:+.3f}  τ_meas={tm:+.3f}")
                results[direction].append(tr)

    return results


# ──────────────────────────────────────────────────────────────────────
# VELOCITY SWEEP — central-window sampling + hold others
# ──────────────────────────────────────────────────────────────────────
def _sample_leg(driver, mid, j_idx, v_j_target, edge_sim_p,
                direction_sign, pos_hard, log_writer, hold_positions,
                sample_lo, sample_hi):
    """Run motor at velocity v_j_target until joint reaches edge_sim_p.
    Only RECORD samples while sim_pos is inside [sample_lo, sample_hi]."""
    v_m_cmd = vel_joint_to_motor(v_j_target, j_idx)
    kd_m    = kd_joint_to_motor(HOLD_KD_JOINT, j_idx)
    max_leg = SETTLE_TIME + AVERAGE_TIME_PER_DIR + 2.0

    ps, vs, ts = [], [], []
    t0 = time.monotonic()
    while True:
        ls = time.monotonic()
        if ls - t0 > max_leg:
            break

        _send_test(driver, mid, 0.0, v_m_cmd, 0.0, kd_m, 0.0,
                   hold_positions=hold_positions)
        sim_p, sim_v, _, _ = _check_safety(
            driver, mid, j_idx, pos_hard, hold_positions)
        _, _, tau_j, temp = _get_state_joint(driver, mid, j_idx)

        if direction_sign > 0 and sim_p >= edge_sim_p:
            break
        if direction_sign < 0 and sim_p <= edge_sim_p:
            break

        # Record only after settle AND only inside central position band.
        in_band = sample_lo <= sim_p <= sample_hi
        if (ls - t0) >= SETTLE_TIME and in_band:
            ps.append(sim_p); vs.append(sim_v); ts.append(tau_j)
            log_writer.writerow([
                'sweep', int(direction_sign), 0,
                f"{ls:.6f}", f"{v_j_target:.4f}",
                f"{sim_p:.4f}", f"{sim_v:.4f}", f"{tau_j:.4f}",
                f"{temp:.2f}",
            ])
        _sleep_until(ls + LOOP_DT)
    return ps, vs, ts


def measure_velocity_sweep(driver, mid, j_idx, profile, log_writer,
                           hold_positions):
    pos_min, pos_max = profile['pos_min'], profile['pos_max']
    pos_hard  = profile['pos_hard']
    safe_high = 0.90 * pos_max
    safe_low  = 0.90 * pos_min

    # Central sampling band — half the window, centered.
    center  = 0.5 * (pos_min + pos_max)
    half_w  = 0.25 * (pos_max - pos_min)
    band_lo = center - half_w
    band_hi = center + half_w

    # Central band is half the window → expect ~half the samples per leg
    target_samples = int(AVERAGE_TIME_PER_DIR * CONTROL_HZ * 0.5)

    print(f"\n── Velocity sweep (motor {mid} / "
          f"{C.SIM_JOINT_NAMES[j_idx]}) ──")
    print(f"   Joint window [{pos_min:+.2f}, {pos_max:+.2f}] rad")
    print(f"   Sample band  [{band_lo:+.2f}, {band_hi:+.2f}] rad")

    all_p, all_v, all_t = [], [], []
    sweep_error = None

    for v_mag in profile['sweep']:
        try:
            _recenter(driver, mid, j_idx, profile, hold_positions,
                      ramp_duration=1.5)

            ps_p, vs_p, ts_p = [], [], []
            ps_n, vs_n, ts_n = [], [], []
            for _ in range(MAX_PINGPONG_CYCLES):
                a, b, c = _sample_leg(driver, mid, j_idx,
                                      +v_mag, safe_high, +1, pos_hard,
                                      log_writer, hold_positions,
                                      band_lo, band_hi)
                ps_p.extend(a); vs_p.extend(b); ts_p.extend(c)
                a, b, c = _sample_leg(driver, mid, j_idx,
                                      -v_mag, safe_low, -1, pos_hard,
                                      log_writer, hold_positions,
                                      band_lo, band_hi)
                ps_n.extend(a); vs_n.extend(b); ts_n.extend(c)
                if (len(vs_p) >= target_samples
                        and len(vs_n) >= target_samples):
                    break

            for ps, vs, ts, s in [(ps_p, vs_p, ts_p, +1),
                                  (ps_n, vs_n, ts_n, -1)]:
                if len(vs) < 10:
                    print(f"   |v|={v_mag:.2f} sign={s:+d}: "
                          f"NOT ENOUGH SAMPLES ({len(vs)})")
                    continue
                vm, vsd = float(np.mean(vs)), float(np.std(vs))
                tm, tsd = float(np.mean(ts)), float(np.std(ts))
                print(f"   |v|={v_mag:.2f} sign={s:+d}: "
                      f"v={vm:+5.2f}±{vsd:.3f}  τ={tm:+6.3f}±{tsd:.3f}  "
                      f"n={len(vs)}")
                all_p.extend(ps); all_v.extend(vs); all_t.extend(ts)

            _hold_limp(driver, mid, j_idx, INTER_REST, pos_hard,
                       hold_positions)
        except Exception as e:
            print(f"   |v|={v_mag:.2f}: ABORTED ({e}) — "
                  f"keeping partial data")
            sweep_error = e
            break

    if sweep_error is not None:
        # Re-raise so main's outer try/except records it, but the
        # partial all_p/v/t have already been accumulated in the
        # caller's scope (we return them below before re-raising).
        # Actually no — easier to NOT re-raise and just print.
        pass

    return all_p, all_v, all_t


# ──────────────────────────────────────────────────────────────────────
# Friction fit — subtracts pre-fitted gravity first
# ──────────────────────────────────────────────────────────────────────
def fit_friction(all_p, all_v, all_t, gravity):
    p = np.asarray(all_p, dtype=float)
    v = np.asarray(all_v, dtype=float)
    t = np.asarray(all_t, dtype=float)
    if len(v) == 0:
        return None

    a_g, c_g, d_g = gravity
    t_corr = t - (a_g * np.sin(p) + c_g * np.cos(p) + d_g)

    mask = np.abs(v) > 0.05
    p, v, t_corr = p[mask], v[mask], t_corr[mask]
    if len(v) < 20:
        return None

    sign_pos = (v > 0).astype(float)
    sign_neg = (v < 0).astype(float)
    v_pos = np.where(v > 0, v, 0.0)
    v_neg = np.where(v < 0, v, 0.0)

    # 4-parameter friction model (gravity already removed)
    X = np.column_stack([sign_pos, sign_neg, v_pos, v_neg])
    coeffs, *_ = np.linalg.lstsq(X, t_corr, rcond=None)
    tc_p, tc_n, b_p, b_n = coeffs

    pred = X @ coeffs
    ss_r = float(np.sum((t_corr - pred) ** 2))
    ss_t = float(np.sum((t_corr - t_corr.mean()) ** 2)) or 1e-12
    r2_total = 1.0 - ss_r / ss_t

    def per_dir(m, slope, intercept):
        if m.sum() < 5:
            return None
        vv, tt = v[m], t_corr[m]
        pp = intercept + slope * vv
        sr = float(np.sum((tt - pp) ** 2))
        st = float(np.sum((tt - tt.mean()) ** 2)) or 1e-12
        return 1.0 - sr / st

    return {
        'tc_pos': float(tc_p), 'b_pos': float(b_p),
        'tc_neg': float(tc_n), 'b_neg': float(b_n),
        'a_grav': float(a_g),  'c_grav': float(c_g), 'd_grav': float(d_g),
        'r2_total': float(r2_total),
        'r2_pos': per_dir(v > 0, b_p, tc_p),
        'r2_neg': per_dir(v < 0, b_n, tc_n),
        'n_samples': int(len(v)),
    }


# ──────────────────────────────────────────────────────────────────────
def main():
    mid = MOTOR_UNDER_TEST
    if mid not in MOTOR_PROFILES:
        raise SystemExit(f"No profile for motor {mid}")
    if mid not in MOTOR_TO_JOINT_IDX:
        raise SystemExit(f"Motor {mid} not in C.MOTOR_IDS")

    j_idx = MOTOR_TO_JOINT_IDX[mid]
    profile = MOTOR_PROFILES[mid]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"friction_{mid}_{stamp}.csv"

    print(f"=== Friction ID v2 — motor {mid} "
          f"({C.SIM_JOINT_NAMES[j_idx]}) ===")
    print(f"  sign={C.JOINT_SIGN[j_idx]:+d}  "
          f"offset={C.JOINT_OFFSET[j_idx]:+.3f}", end='')
    if _is_knee(j_idx):
        print(f"  gear={C.KNEE_GEAR_RATIO}  coupling={C.KNEE_HIP_COUPLING}")
    else:
        print()
    print(f"  joint window: [{profile['pos_min']:+.2f}, "
          f"{profile['pos_max']:+.2f}] rad  hard ±{profile['pos_hard']:.2f}")
    print(f"  sweep (joint, rad/s): {profile['sweep']}")
    print(f"  log: {csv_path}")

    driver = make_driver(list(C.MOTOR_IDS))

    f = open(csv_path, 'w', newline='')
    w = csv.writer(f)
    w.writerow(['phase', 'direction', 'trial', 't',
                'cmd_joint', 'sim_pos', 'sim_vel', 'tau_joint', 'temp'])

    static_results = {+1: [], -1: []}
    all_p, all_v, all_t = [], [], []
    gravity = (0.0, 0.0, 0.0)
    hold_positions = None

    try:
        driver.enable_all()
        hold_positions = _safe_startup(driver, mid, j_idx,
                                       profile['pos_hard'])
        _recenter(driver, mid, j_idx, profile, hold_positions)

        # ── NEW: gravity profile before friction tests ──
        gravity = measure_gravity_profile(driver, mid, j_idx, profile,
                                          w, hold_positions)

        static_results = measure_static(driver, mid, j_idx, profile,
                                        w, hold_positions)
        _recenter(driver, mid, j_idx, profile, hold_positions)
        all_p, all_v, all_t = measure_velocity_sweep(
            driver, mid, j_idx, profile, w, hold_positions)

    except KeyboardInterrupt:
        print("\n[!] Interrupted")
    except Exception as e:
        print(f"\n[!] Error: {e}")
    finally:
        try:
            if hold_positions is not None:
                _hold_limp(driver, mid, j_idx, 0.3,
                           profile['pos_hard'], hold_positions)
        except Exception:
            pass
        driver.close()
        f.close()

    fit = fit_friction(all_p, all_v, all_t, gravity)

    print("\n" + "═" * 68)
    print(f"  RESULTS — motor {mid} ({C.SIM_JOINT_NAMES[j_idx]})")
    print("═" * 68)
    print("  Gravity model (from quasi-static sweep, JOINT-SIDE):")
    print(f"    τ_g(θ) ≈ {gravity[0]:+.3f}·sin(θ) + "
          f"{gravity[1]:+.3f}·cos(θ) + {gravity[2]:+.3f}  Nm")

    # Consistency check at recenter pose
    center_pos = 0.5 * (profile['pos_min'] + profile['pos_max'])
    pred_grav = (gravity[0] * np.sin(center_pos) +
                 gravity[1] * np.cos(center_pos) + gravity[2])
    print(f"    Predicted τ_g at center θ={center_pos:+.2f}: "
          f"{pred_grav:+.3f} Nm  (compare to static τ_grav prints above)")

    print("\n  JOINT-SIDE STATIC (for DCMotor cfg):")
    if static_results[+1]:
        a = np.array(static_results[+1])
        print(f"    τ_s (+) = {a.mean():+6.3f} ± {a.std():.3f} Nm  "
              f"(n={len(a)})")
    if static_results[-1]:
        a = np.array(static_results[-1])
        print(f"    τ_s (-) = {a.mean():+6.3f} ± {a.std():.3f} Nm  "
              f"(n={len(a)})")

    if fit is not None:
        r2p = (f"{fit['r2_pos']:.3f}"
               if fit['r2_pos'] is not None else "n/a")
        r2n = (f"{fit['r2_neg']:.3f}"
               if fit['r2_neg'] is not None else "n/a")
        print(f"\n  Friction fit (n={fit['n_samples']}, "
              f"R²_total={fit['r2_total']:.3f}, gravity pre-subtracted):")
        print(f"    τ_c (+) = {fit['tc_pos']:+6.3f} Nm   "
              f"b (+) = {fit['b_pos']:+6.4f} Nm·s/rad   R²(+)={r2p}")
        print(f"    τ_c (-) = {fit['tc_neg']:+6.3f} Nm   "
              f"b (-) = {fit['b_neg']:+6.4f} Nm·s/rad   R²(-)={r2n}")
    else:
        print("  Sweep fit: NOT ENOUGH SAMPLES")

    s = C.JOINT_SIGN[j_idx]
    g = C.KNEE_GEAR_RATIO if _is_knee(j_idx) else 1.0
    print("\n  Motor-side (cross-check):")
    if static_results[+1]:
        a = np.array(static_results[+1])
        print(f"    τ_s motor (+) ≈ {a.mean()*s*g:+6.3f} Nm")
    if static_results[-1]:
        a = np.array(static_results[-1])
        print(f"    τ_s motor (-) ≈ {a.mean()*s*g:+6.3f} Nm")
    if fit is not None:
        print(f"    τ_c motor (+) ≈ {fit['tc_pos']*s*g:+6.3f}  "
              f"b motor (+) ≈ {fit['b_pos']*g*g:+6.4f}")
        print(f"    τ_c motor (-) ≈ {fit['tc_neg']*s*g:+6.3f}  "
              f"b motor (-) ≈ {fit['b_neg']*g*g:+6.4f}")
    print("═" * 68)
    print(f"  CSV: {csv_path}")


if __name__ == "__main__":
    main()