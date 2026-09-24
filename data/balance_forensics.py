"""
balance_forensics.py — offline forensics on a balance_log.py NPZ. No Isaac, no GPU.

Answers, from the log alone, the questions the obs-parity check structurally
cannot (obs_parity_live.py never steps the sim and never looks at target /
torque / timing):

  1. TIMING     — did the 100 Hz loop actually run at 100 Hz? The gait clock
                  advances by a fixed 2*pi*f/hz per tick, so any loop overrun
                  silently desyncs gait phase from wall-clock. Invisible to
                  obs parity (both sides use the same nominal constant).
  2. BASE       — the robot hung from a harness (base NOT fixed). How much did
                  the base actually move? Bounds the error of a fixed-base
                  replay and decides whether the driven-base mode matters.
  3. CAN        — dropped reply frames => stale joint feedback that tick.
  4. TORQUE     — did the run operate inside the DCMotor envelope the policy
                  was trained against? Fraction above continuous rating?
  5. LPF+CLIP   — reconstruct the hardware target filter (incl. the effort
                  soft-clip that writes back into the filter state,
                  motor_bridge.py:191-210) and validate against the logged
                  motor_tau_cmd. This reconstruction is what the sim replay
                  (sysid_replay_policy_log.py) builds on.
  6. VELOCITY   — is sim_joint_vel consistent with d(pos)/dt (k ~ 1)?
                  (Same check as vel_consistency.py, applied to this log.)
  7. TAU RESID  — PD-predicted torque vs motor-reported torque, per joint:
                  free actuator-model information before any sim run.
  8. DELAY      — model-free command->motion lag per joint via
                  cross-correlation, to compare against the trained action
                  delay range (1-2 steps @ 10 ms).

Usage:
    python balance_forensics.py --npz ../control/model_4150_log_20260803_140243.npz

Outputs: a PASS/WARN/INFO report on stdout, <npz>_forensics.pdf (plots) and
<npz>_forensics.json (machine-readable summary) next to the input file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Single source of truth for hardware constants (signs, gear, scales, limits).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "control"))
import config as C  # noqa: E402

JOINT_NAMES = ["hr1", "hp1", "kn1", "hr2", "hp2", "kn2"]
GROUPS      = ["hip_roll", "hip_pitch", "knee", "hip_roll", "hip_pitch", "knee"]
KNEE_IDX    = (2, 5)

SIGN  = np.array(C.JOINT_SIGN, dtype=np.float64)
GEAR  = float(C.KNEE_GEAR_RATIO)                       # 22/30
DEFAULT_POS   = np.array(C.DEFAULT_JOINT_POS, dtype=np.float64)
ACTION_SCALES = np.array(C.ACTION_SCALES, dtype=np.float64)
CONT_EFFORT   = np.array(C.REAL_EFFORT, dtype=np.float64)    # 1.8 / 6.0 / 8.18 (joint frame)
BURST_EFFORT  = np.array(C.BURST_EFFORT, dtype=np.float64)   # 6.0 / 17.0 / 23.2 (joint frame)
GAIT_FREQ     = float(C.GAIT_FREQ_HZ)

# DCMotor speed-torque envelope, JOINT frame — mirrors rl_env_cfg_p6.py §3
# (saturation_effort / velocity_limit per group). Informational only; the
# training cfg is the authority.
SATURATION_EFFORT = np.array([6.0, 17.0, 23.2, 6.0, 17.0, 23.2])
VELOCITY_LIMIT    = np.array([28.0, 27.5, 20.0, 28.0, 27.5, 20.0])

# Gains confirmed by the operator for the 20260803 run (= training gains).
# Future logs carry per-tick kp/kd channels; use those when present.
FALLBACK_KP = np.array([6.0, 8.0, 8.0, 6.0, 8.0, 8.0])
FALLBACK_KD = np.array([0.3, 0.5, 0.5, 0.3, 0.5, 0.5])
FALLBACK_LPF_ALPHA = 0.6   # MotorBridge constructor default


def _status(level: str, msg: str) -> None:
    tag = {"PASS": "[PASS]", "WARN": "[WARN]", "INFO": "[info]", "FAIL": "[FAIL]"}[level]
    print(f"  {tag} {msg}")


# ────────────────────────────────────────────────────────────────────
# Hardware target-filter reconstruction (motor_bridge.send_target twin)
# ────────────────────────────────────────────────────────────────────

def recover_gains_and_alpha(target, pos, vel, tau_cmd_joint, warmup):
    """Recover the (kp, kd, alpha) actually in effect, per joint, from the log.

    balance_log.py computes motor_tau_cmd from the LPF'd target:
        tau[t] = kp*(f[t] - pos[t]) - kd*vel[t],   f[t] = a*f[t-1] + (1-a)*u[t]
    Eliminating f gives, for a fixed alpha, a relation LINEAR in (kp, kd):
        tau[t] - a*tau[t-1] = kp*[(1-a)*u[t] - pos[t] + a*pos[t-1]]
                            + kd*[a*vel[t-1] - vel[t]]
    Solve least-squares per candidate alpha (the discrete set balance_log.py
    cycles through) and keep the alpha with the smallest residual. Holds
    exactly (float32 rounding only) whenever the effort soft-clip did not
    engage; a large residual therefore also flags clip activity.

    Returns (kp[6], kd[6], alpha[6], rmse[6]).
    """
    alphas = [0.0, 0.5, 0.6, 0.65, 0.7]
    kp_out = np.zeros(6); kd_out = np.zeros(6)
    a_out = np.zeros(6); rmse_out = np.zeros(6)
    for i in range(6):
        best = None
        for a in alphas:
            y = (tau_cmd_joint[1:, i] - a * tau_cmd_joint[:-1, i])[warmup:]
            A = ((1 - a) * target[1:, i] - pos[1:, i] + a * pos[:-1, i])[warmup:]
            B = (a * vel[:-1, i] - vel[1:, i])[warmup:]
            M = np.stack([A, B], axis=1)
            coef, *_ = np.linalg.lstsq(M, y, rcond=None)
            r = float(np.sqrt(np.mean((M @ coef - y) ** 2)))
            if best is None or r < best[0]:
                best = (r, coef[0], coef[1], a)
        rmse_out[i], kp_out[i], kd_out[i], a_out[i] = best
    return kp_out, kd_out, a_out, rmse_out


def reconstruct_filtered_target(target, pos, kp, alpha, effort_limit,
                                init_state=None):
    """Replay MotorBridge's LPF + effort soft-clip over the logged run.

    target[t] is the pre-LPF sim-space target logged at tick t; pos[t] the
    joint position measured at the top of tick t (what the clip saw).

    Mirrors motor_bridge.py send_target():
      filt = a*filt + (1-a)*target          (init from measured pose)
      err  = filt - pos; clamp so kp*|err| <= effort_limit,
      and the clamp WRITES BACK into the filter state (effective_target is a
      reference to _filtered_target).

    init_state: filter state BEFORE tick 0 (the HOLD-phase filter carries
    over — 'b' does not reset it). If None, uses pos[0].

    Returns (filtered[t] post-clip, clip_engaged[t] bool per joint).
    """
    n = target.shape[0]
    filt = np.zeros_like(target)
    clipped = np.zeros(target.shape, dtype=bool)
    state = (init_state if init_state is not None else pos[0]).astype(np.float64).copy()
    for t in range(n):
        a = alpha[t] if hasattr(alpha, "__len__") else alpha
        state = a * state + (1.0 - a) * target[t]
        for i in range(6):
            if kp[t, i] > 0.0:
                max_err = effort_limit[i] / kp[t, i]
                err = state[i] - pos[t, i]
                if abs(err) > max_err:
                    state[i] = pos[t, i] + np.sign(err) * max_err
                    clipped[t, i] = True
        filt[t] = state
    return filt, clipped


def joint_to_motor_torque(tau_joint):
    """Joint-frame torque -> motor frame (balance_log.py:679-681)."""
    tau_m = tau_joint * SIGN
    tau_m[..., KNEE_IDX[0]] *= GEAR
    tau_m[..., KNEE_IDX[1]] *= GEAR
    return tau_m


def motor_to_joint_torque(tau_motor):
    """Motor-frame torque -> joint frame (motor_bridge._motor_to_sim_torque)."""
    tau_j = tau_motor * SIGN
    tau_j[..., KNEE_IDX[0]] /= GEAR
    tau_j[..., KNEE_IDX[1]] /= GEAR
    return tau_j


def xcorr_delay(cmd_vel, meas_vel, max_lag):
    """argmax_k corr(cmd_vel[t-k], meas_vel[t]) — effective lag in ticks."""
    best_k, best_r = 0, -np.inf
    rs = []
    c = cmd_vel - cmd_vel.mean()
    m = meas_vel - meas_vel.mean()
    for k in range(max_lag + 1):
        a = c[: len(c) - k] if k else c
        b = m[k:]
        denom = np.std(a) * np.std(b)
        r = float(np.mean(a * b) / denom) if denom > 1e-12 else 0.0
        rs.append(r)
        if r > best_r:
            best_r, best_k = r, k
    return best_k, best_r, rs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--npz", required=True, help="balance_log.py output NPZ")
    ap.add_argument("--alpha", type=float, default=None,
                    help="LPF alpha override (default: log channel or 0.6)")
    ap.add_argument("--warmup", type=int, default=20,
                    help="ticks to skip in RMSE-style stats (init transients)")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    n = len(d["time"])
    t_arr   = d["time"].astype(np.float64)
    action  = d["action"].astype(np.float64)
    target  = d["target"].astype(np.float64)
    pos     = d["sim_joint_pos"].astype(np.float64)
    vel     = d["sim_joint_vel"].astype(np.float64)
    tau_cmd_log  = d["motor_tau_cmd"].astype(np.float64)    # motor frame
    tau_meas     = d["motor_tau_meas"].astype(np.float64)   # motor frame
    proj_grav    = d["proj_grav"].astype(np.float64)
    ang_vel      = d["ang_vel"].astype(np.float64)
    can_replies  = d["can_replies"].astype(np.int64)
    control_hz   = float(d["control_hz"]) if "control_hz" in d else 100.0
    period       = 1.0 / control_hz
    w = min(args.warmup, n // 4)

    # Per-tick run-config channels exist in logs from the patched logger;
    # for older logs, recover the gains from the log's own tau_cmd recursion
    # (exact when the effort clip never engaged) and cross-check against the
    # operator-recalled fallback.
    tau_cmd_joint_log = motor_to_joint_torque(d["motor_tau_cmd"].astype(np.float64))
    if "kp" in d.files:
        kp_arr = d["kp"].astype(np.float64)
        kd_arr = d["kd"].astype(np.float64)
        gains_src = "log"
    else:
        kp_rec, kd_rec, a_rec, g_rmse = recover_gains_and_alpha(
            target, pos, vel, tau_cmd_joint_log, warmup=w)
        if g_rmse.max() < 0.02:
            kp_arr = np.tile(kp_rec, (n, 1))
            kd_arr = np.tile(kd_rec, (n, 1))
            gains_src = "recovered from tau_cmd recursion"
        else:
            kp_arr = np.tile(FALLBACK_KP, (n, 1))
            kd_arr = np.tile(FALLBACK_KD, (n, 1))
            gains_src = "fallback (operator-recalled training gains)"
    if args.alpha is not None:
        alpha = float(args.alpha)
        alpha_src = "cli"
    elif "lpf_alpha" in d.files:
        alpha = d["lpf_alpha"].astype(np.float64)
        alpha_src = "log"
    elif gains_src.startswith("recovered"):
        alpha = float(a_rec[0]) if np.allclose(a_rec, a_rec[0]) else FALLBACK_LPF_ALPHA
        alpha_src = "recovered from tau_cmd recursion"
    else:
        alpha = FALLBACK_LPF_ALPHA
        alpha_src = "fallback (MotorBridge default)"

    summary: dict = {"npz": os.path.basename(args.npz), "n_ticks": n,
                     "policy": str(d["policy"]) if "policy" in d.files else "?",
                     "gains_source": gains_src, "alpha_source": alpha_src}

    print(f"\n=== balance_forensics: {os.path.basename(args.npz)} ===")
    print(f"  ticks={n}  ({t_arr[-1]:.2f} s)   policy={summary['policy']}   "
          f"control_hz={control_hz:.0f}")
    print(f"  gains: {gains_src}   lpf_alpha: {alpha_src}")
    if gains_src.startswith("recovered"):
        print("  recovered per-joint (kp, kd, alpha, fit rmse):")
        for i in range(6):
            print(f"    {JOINT_NAMES[i]}: kp={kp_rec[i]:6.3f}  kd={kd_rec[i]:6.3f}  "
                  f"alpha={a_rec[i]:.2f}  rmse={g_rmse[i]:.2e} Nm")
        summary["recovered_gains"] = {"kp": kp_rec.tolist(), "kd": kd_rec.tolist(),
                                      "alpha": a_rec.tolist()}
        if not (np.allclose(kp_rec, FALLBACK_KP, atol=0.15)
                and np.allclose(kd_rec, FALLBACK_KD, atol=0.05)):
            _status("WARN", "recovered gains differ from the training gains "
                            f"(training kp={FALLBACK_KP.tolist()}, "
                            f"kd={FALLBACK_KD.tolist()}) — the policy ran "
                            "against a DIFFERENT PD than it was trained with")

    # ── 1. Timing ────────────────────────────────────────────────────
    print("\n-- 1. Loop timing --")
    dt = np.diff(t_arr)
    mean_hz = 1.0 / dt.mean()
    n_over = int((dt > 1.1 * period).sum())
    worst = float(dt.max())
    # Gait clock advances 2*pi*f/hz per TICK; wall clock says ticks took
    # sum(dt). Cumulative phase desync vs wall time:
    gait_desync = 2.0 * math.pi * GAIT_FREQ * (t_arr[-1] - (n - 1) * period)
    lvl = "PASS" if (abs(mean_hz - control_hz) < 1.0 and n_over < 0.01 * n) else "WARN"
    _status(lvl, f"mean rate {mean_hz:.2f} Hz (nominal {control_hz:.0f});  "
                 f"{n_over}/{n-1} ticks >{1.1*period*1e3:.0f} ms;  worst {worst*1e3:.1f} ms")
    lvl = "PASS" if abs(gait_desync) < 0.2 else "WARN"
    _status(lvl, f"cumulative gait-phase desync {gait_desync:+.3f} rad "
                 f"({math.degrees(gait_desync):+.1f} deg over the run)")
    summary["timing"] = {"mean_hz": mean_hz, "overrun_ticks": n_over,
                         "worst_dt_ms": worst * 1e3,
                         "gait_desync_rad": gait_desync}

    # ── 2. Base motion (robot hung from a harness — base NOT fixed) ──
    print("\n-- 2. Base motion (harness) --")
    g_mean = proj_grav.mean(axis=0)
    g_mean /= np.linalg.norm(g_mean)
    gn = proj_grav / np.linalg.norm(proj_grav, axis=1, keepdims=True)
    tilt_dev = np.degrees(np.arccos(np.clip(gn @ g_mean, -1.0, 1.0)))
    tilt_from_vert = np.degrees(np.arccos(np.clip(-gn[:, 2], -1.0, 1.0)))
    av_rms = np.sqrt((ang_vel ** 2).mean(axis=0))
    av_max = np.abs(ang_vel).max(axis=0)
    frac = math.sin(math.radians(float(tilt_dev.max())))
    lvl = "PASS" if tilt_dev.max() < 5.0 else ("WARN" if tilt_dev.max() < 15.0 else "FAIL")
    _status(lvl, f"tilt wander vs run mean: max {tilt_dev.max():.2f} deg, "
                 f"mean {tilt_dev.mean():.2f} deg  "
                 f"(mean tilt from vertical {tilt_from_vert.mean():.2f} deg)")
    _status("INFO", f"body ang_vel rms ({av_rms[0]:.3f},{av_rms[1]:.3f},{av_rms[2]:.3f}) "
                    f"max ({av_max[0]:.2f},{av_max[1]:.2f},{av_max[2]:.2f}) rad/s")
    _status("INFO", f"=> gravity-torque modulation on joints bounded by "
                    f"~sin(max wander) = {100*frac:.1f}% — "
                    f"{'fixed-base replay is a fair approximation' if frac < 0.09 else 'prefer the driven-base replay mode'}")
    summary["base_motion"] = {"tilt_dev_max_deg": float(tilt_dev.max()),
                              "tilt_dev_mean_deg": float(tilt_dev.mean()),
                              "ang_vel_rms": av_rms.tolist(),
                              "grav_torque_modulation_frac": frac}

    # ── 3. CAN health ────────────────────────────────────────────────
    print("\n-- 3. CAN health --")
    n_drop = int((can_replies < 6).sum())
    lvl = "PASS" if n_drop == 0 else ("WARN" if n_drop < 0.02 * n else "FAIL")
    _status(lvl, f"{n_drop}/{n} ticks with <6 reply frames "
                 f"(min {int(can_replies.min())})")
    summary["can"] = {"dropped_ticks": n_drop, "min_replies": int(can_replies.min())}

    # ── 4. Policy action stats ───────────────────────────────────────
    print("\n-- 4. Action saturation --")
    sat_frac = (np.abs(action) >= 0.999).mean(axis=0)
    for i in range(6):
        lvl = "PASS" if sat_frac[i] < 0.05 else "WARN"
        _status(lvl, f"{JOINT_NAMES[i]}: |action|=1 on {100*sat_frac[i]:5.1f}% of ticks")
    summary["action_saturation_frac"] = sat_frac.tolist()

    # target identity: target == DEFAULT + action*scales (constants check)
    tgt_err = np.abs(target - (DEFAULT_POS + action * ACTION_SCALES)).max()
    lvl = "PASS" if tgt_err < 1e-5 else "FAIL"
    _status(lvl, f"target == default + action*scale  (max err {tgt_err:.2e})")

    # ── 5. LPF + effort-clip reconstruction, tau_cmd validation ─────
    print("\n-- 5. Hardware target-filter reconstruction --")
    alpha_vec = alpha if hasattr(alpha, "__len__") else np.full(n, float(alpha))
    # The filter state carries over from HOLD ('b' does not reset it), so
    # tick 0's state is unknown a priori — but tau_cmd[0] encodes it exactly:
    #   filt[0] = tau_j[0]/kp + kd*vel[0]/kp + pos[0]
    # Invert one LPF update to get the pre-run state (a=0 needs no init).
    filt0 = tau_cmd_joint_log[0] / kp_arr[0] + kd_arr[0] * vel[0] / kp_arr[0] + pos[0]
    a0 = alpha_vec[0]
    init_state = (filt0 - (1.0 - a0) * target[0]) / a0 if a0 > 0 else None
    filt, clip_mask = reconstruct_filtered_target(
        target, pos, kp_arr, alpha_vec, BURST_EFFORT, init_state=init_state)
    tau_cmd_recon_joint = kp_arr * (filt - pos) - kd_arr * vel
    tau_cmd_recon = joint_to_motor_torque(tau_cmd_recon_joint)
    recon_err = np.abs(tau_cmd_recon - tau_cmd_log)
    rerr = recon_err[w:].max()
    lvl = "PASS" if rerr < 0.05 else ("WARN" if rerr < 0.5 else "FAIL")
    _status(lvl, f"reconstructed motor_tau_cmd vs logged: max |err| {rerr:.4f} Nm "
                 f"(mean {recon_err[w:].mean():.4f}) after {w}-tick warmup")
    if lvl != "PASS":
        _status("INFO", "   large error => wrong alpha/kp/kd assumption; "
                        "try --alpha, or trust per-tick channels once the "
                        "patched logger ships them")
    clip_frac = clip_mask.mean(axis=0)
    any_clip = clip_frac.max() > 0
    for i in range(6):
        if clip_frac[i] > 0:
            _status("WARN", f"{JOINT_NAMES[i]}: effort soft-clip engaged on "
                            f"{100*clip_frac[i]:.1f}% of ticks (burst limit "
                            f"{BURST_EFFORT[i]:.1f} Nm) — nonlinearity the sim "
                            f"action chain does not have")
    if not any_clip:
        _status("PASS", "effort soft-clip never engaged")
    summary["lpf_recon"] = {"tau_cmd_max_err": float(rerr),
                            "clip_frac": clip_frac.tolist()}

    # ── 6. Torque regime (joint frame) ───────────────────────────────
    print("\n-- 6. Torque regime vs DCMotor envelope (joint frame) --")
    tau_meas_joint = motor_to_joint_torque(tau_meas)
    above_cont  = (np.abs(tau_meas_joint) > CONT_EFFORT).mean(axis=0)
    above_burst = (np.abs(tau_meas_joint) > BURST_EFFORT).mean(axis=0)
    # DCMotor 4-quadrant clip: driving torque limited by sat*(1 -/+ v/vlim),
    # clipped to [0, effort] / [-effort, 0] per side; braking side full.
    up  = np.clip(SATURATION_EFFORT * (1.0 - vel / VELOCITY_LIMIT), 0.0, BURST_EFFORT)
    lo  = np.clip(SATURATION_EFFORT * (-1.0 - vel / VELOCITY_LIMIT), -BURST_EFFORT, 0.0)
    outside_env = ((tau_meas_joint > up + 1e-9) | (tau_meas_joint < lo - 1e-9)).mean(axis=0)
    for i in range(6):
        lvl = "INFO" if above_cont[i] < 0.30 else "WARN"
        _status(lvl, f"{JOINT_NAMES[i]}: |tau|>continuous({CONT_EFFORT[i]:.2f}) "
                     f"{100*above_cont[i]:5.1f}%   >burst({BURST_EFFORT[i]:.1f}) "
                     f"{100*above_burst[i]:4.1f}%   outside DCMotor envelope "
                     f"{100*outside_env[i]:4.1f}%")
    summary["torque"] = {"above_continuous_frac": above_cont.tolist(),
                         "above_burst_frac": above_burst.tolist(),
                         "outside_envelope_frac": outside_env.tolist()}

    # ── 7. Velocity channel consistency ──────────────────────────────
    print("\n-- 7. Velocity consistency (vel ~ k * dpos/dt) --")
    dpos_dt = np.gradient(pos, t_arr, axis=0)
    ks, r2s = [], []
    for i in range(6):
        x, y = dpos_dt[w:, i], vel[w:, i]
        k = float(np.dot(x, y) / max(np.dot(x, x), 1e-12))
        resid = y - k * x
        r2 = 1.0 - float(np.var(resid) / max(np.var(y), 1e-12))
        ks.append(k); r2s.append(r2)
        lvl = "PASS" if 0.8 < k < 1.25 else "WARN"
        _status(lvl, f"{JOINT_NAMES[i]}: k={k:5.2f}  R2={r2:5.2f}")
    summary["vel_consistency"] = {"k": ks, "r2": r2s}

    # ── 8. PD-predicted vs motor-reported torque ─────────────────────
    print("\n-- 8. tau_cmd vs tau_meas residual (motor frame) --")
    resid = tau_cmd_log - tau_meas
    for i in range(6):
        r = resid[w:, i]
        _status("INFO", f"{JOINT_NAMES[i]}: rmse {np.sqrt((r**2).mean()):6.3f} Nm   "
                        f"bias {r.mean():+6.3f}   |max| {np.abs(r).max():6.3f}")
    summary["tau_residual_rmse"] = [float(np.sqrt((resid[w:, i]**2).mean())) for i in range(6)]

    # ── 9. Model-free command->motion delay ──────────────────────────
    print("\n-- 9. Effective command->motion lag (xcorr, ticks of 10 ms) --")
    print("      (total lag incl. LPF+motor response; trained action delay 1-2)")
    filt_vel = np.gradient(filt, t_arr, axis=0)
    delays = []
    for i in range(6):
        k, r, _ = xcorr_delay(filt_vel[w:, i], vel[w:, i], max_lag=15)
        delays.append(k)
        moved = np.abs(filt_vel[w:, i]).mean() > 0.02
        lvl = "INFO" if not moved else ("PASS" if k <= 4 else "WARN")
        _status(lvl, f"{JOINT_NAMES[i]}: lag {k:2d} ticks  (r={r:.2f})"
                     + ("" if moved else "  [joint barely moved — unreliable]"))
    summary["xcorr_delay_ticks"] = delays

    # ── plots ────────────────────────────────────────────────────────
    pdf_path = os.path.splitext(args.npz)[0] + "_forensics.pdf"
    with PdfPages(pdf_path) as pdf:
        # per-joint tracking
        fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
        for i in range(6):
            ax = axes[i % 3][i // 3]
            ax.plot(t_arr, target[:, i], "k:", lw=0.7, alpha=0.6, label="target (pre-LPF)")
            ax.plot(t_arr, filt[:, i], "g-", lw=0.8, alpha=0.8, label="filtered (recon)")
            ax.plot(t_arr, pos[:, i], "b-", lw=0.9, label="measured pos")
            ax.set_title(JOINT_NAMES[i], fontsize=9)
            if i == 0:
                ax.legend(fontsize=7)
        fig.suptitle("Joint tracking: commanded vs measured (rad)")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
        for i in range(6):
            ax = axes[i % 3][i // 3]
            ax.plot(t_arr, vel[:, i], "b-", lw=0.8, label="logged vel")
            ax.plot(t_arr, dpos_dt[:, i], "r--", lw=0.7, alpha=0.7,
                    label=f"dpos/dt (k={ks[i]:.2f})")
            ax.set_title(JOINT_NAMES[i], fontsize=9)
            if i == 0:
                ax.legend(fontsize=7)
        fig.suptitle("Velocity channel consistency (rad/s)")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
        for i in range(6):
            ax = axes[i % 3][i // 3]
            ax.plot(t_arr, tau_cmd_log[:, i], "b-", lw=0.8, label="tau_cmd (logged)")
            ax.plot(t_arr, tau_cmd_recon[:, i], "g--", lw=0.7, label="tau_cmd (recon)")
            ax.plot(t_arr, tau_meas[:, i], "r-", lw=0.7, alpha=0.7, label="tau_meas")
            ax.set_title(JOINT_NAMES[i], fontsize=9)
            if i == 0:
                ax.legend(fontsize=7)
        fig.suptitle("Torque: PD command vs motor-reported (motor frame, Nm)")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        axes[0].hist(dt * 1e3, bins=60)
        axes[0].axvline(period * 1e3, color="k", ls=":")
        axes[0].set_title(f"tick dt (ms), mean {dt.mean()*1e3:.2f}")
        axes[1].plot(t_arr[1:], dt * 1e3, lw=0.6)
        axes[1].axhline(period * 1e3, color="k", ls=":")
        axes[1].set_title("dt over time (ms)")
        axes[2].plot(t_arr, can_replies, lw=0.6)
        axes[2].set_ylim(-0.5, 6.5); axes[2].set_title("CAN replies/tick")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        for j, lbl in enumerate("xyz"):
            axes[0].plot(t_arr, proj_grav[:, j], lw=0.7, label=lbl)
            axes[1].plot(t_arr, ang_vel[:, j], lw=0.7, label=lbl)
        axes[0].set_title("proj_grav (body)"); axes[0].legend(fontsize=7)
        axes[1].set_title("ang_vel (body, rad/s)"); axes[1].legend(fontsize=7)
        axes[2].plot(t_arr, tilt_dev, lw=0.7)
        axes[2].set_title("tilt wander vs run mean (deg)")
        fig.suptitle("Base motion in the harness")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(3, 2, figsize=(13, 10))
        vgrid = np.linspace(-1.05, 1.05, 200)
        for i in range(6):
            ax = axes[i % 3][i // 3]
            ax.scatter(vel[:, i], tau_meas_joint[:, i], s=3, alpha=0.4)
            vv = vgrid * VELOCITY_LIMIT[i]
            ax.plot(vv, np.clip(SATURATION_EFFORT[i] * (1 - vv / VELOCITY_LIMIT[i]),
                                0, BURST_EFFORT[i]), "r-", lw=0.8)
            ax.plot(vv, np.clip(SATURATION_EFFORT[i] * (-1 - vv / VELOCITY_LIMIT[i]),
                                -BURST_EFFORT[i], 0), "r-", lw=0.8)
            ax.axhline(CONT_EFFORT[i], color="orange", ls=":", lw=0.8)
            ax.axhline(-CONT_EFFORT[i], color="orange", ls=":", lw=0.8)
            ax.set_title(f"{JOINT_NAMES[i]}  (red=DCMotor env, orange=cont)", fontsize=8)
            ax.set_xlabel("joint vel (rad/s)", fontsize=7)
            ax.set_ylabel("tau (Nm, joint)", fontsize=7)
        fig.suptitle("Speed-torque operating points vs trained envelope")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(13, 4))
        for i in range(6):
            ax.plot(t_arr, action[:, i], lw=0.7, label=JOINT_NAMES[i])
        ax.axhline(1.0, color="k", ls=":", lw=0.6); ax.axhline(-1.0, color="k", ls=":", lw=0.6)
        ax.legend(fontsize=7, ncol=6); ax.set_title("Policy actions (post-clamp)")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    json_path = os.path.splitext(args.npz)[0] + "_forensics.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  plots  -> {pdf_path}")
    print(f"  summary-> {json_path}\n")


if __name__ == "__main__":
    main()
