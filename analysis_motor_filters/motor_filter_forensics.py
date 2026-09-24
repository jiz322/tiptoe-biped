#!/usr/bin/env python3
"""
motor_filter_forensics.py — Do the firmware telemetry filters (spd_filt_gain,
cur_filt_gain / torque_fdb chain) corrupt the sysid + delivery-ratio analyses?

Key physical facts this exploits:
  * Each CAN reply frame carries (pos, vel, torque, temp) sampled at the SAME
    instant by the firmware. So comparing channels WITHIN a frame isolates
    firmware-side filtering from transport/staleness entirely.
  * pos is raw (encoder); vel is spd_filt_gain-filtered; torque_fdb is
    filtered iq * Kt. Central-difference of pos is a ZERO-PHASE velocity
    reference; its 2nd difference is a zero-phase acceleration reference.
  * In a PD step trial the commanded torque jumps discontinuously. True plant
    torque then shows up in accel immediately; a filtered TELEMETRY channel
    ramps. Comparing accel onset vs torque_fdb onset separates plant from
    sensor.

Outputs: analysis_motor_filters/results.txt, results.json, *.png
"""
import json
import math
import os
import sys
import glob

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "analysis_motor_filters")
os.makedirs(OUT, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DT = 0.01
GEAR = 22.0 / 30.0
COUP = 0.733
SIGN = np.array([+1, +1, -1, +1, +1, -1], dtype=np.float64)
KNEE_IDX = (2, 5)
HP_OF_KN = {2: 1, 5: 4}
JNAMES = ["L_roll", "L_hip", "L_knee", "R_roll", "R_hip", "R_knee"]
GROUPS = ["hip_roll", "hip_pitch", "knee", "hip_roll", "hip_pitch", "knee"]
FW_TAU_RANGE = np.array([6.0, 17.0, 17.0, 6.0, 17.0, 17.0])  # motor-frame encode range EL05/RS01

REPORT = []


def log(msg=""):
    print(msg)
    REPORT.append(msg)


# ────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────

def central_diff(x, t):
    """Zero-phase first derivative on a (possibly slightly nonuniform) grid."""
    d = np.zeros_like(x)
    d[1:-1] = (x[2:] - x[:-2]) / (t[2:] - t[:-2])
    d[0] = d[1]
    d[-1] = d[-2]
    return d


def iir(x, g):
    """First-order IIR y[k] = y[k-1] + g*(x[k]-y[k-1]), y[0]=x[0]."""
    y = np.empty_like(x)
    acc = x[0]
    om = 1.0 - g
    for i in range(len(x)):
        acc = om * acc + g * x[i]
        y[i] = acc
    return y


def g_to_tau_ms(g, dt=DT):
    if g >= 1.0:
        return 0.0
    if g <= 0.0:
        return float("inf")
    return -dt / math.log(1.0 - g) * 1000.0


TAU_GRID_MS = np.concatenate([[0.01], np.geomspace(0.5, 120.0, 60)])
G_GRID = 1.0 - np.exp(-DT * 1000.0 / TAU_GRID_MS)  # per-100Hz-sample gains


def fit_scale_filter_delay(x, y, deltas=(0, 1, 2), mask=None):
    """Fit y[k] ~= s * IIR_g(x)[k-d].  Returns dict(best d, tau_ms, s, R2)."""
    best = None
    n = len(x)
    if mask is None:
        mask = np.ones(n, dtype=bool)
    for g in G_GRID:
        xf_full = iir(x, g)
        for d in deltas:
            xf = xf_full[: n - d] if d > 0 else xf_full
            yy = y[d:]
            mm = mask[d:]
            xs = xf[mm]
            ys = yy[mm]
            if len(xs) < 50:
                continue
            denom = float(np.dot(xs, xs))
            if denom < 1e-9:
                continue
            s = float(np.dot(xs, ys) / denom)
            resid = ys - s * xs
            sse = float(np.dot(resid, resid))
            sst = float(np.dot(ys - ys.mean(), ys - ys.mean())) + 1e-12
            r2 = 1.0 - sse / sst
            if best is None or sse < best["sse"]:
                best = dict(delta=int(d), g=float(g), tau_ms=g_to_tau_ms(g),
                            scale=s, sse=sse, r2=r2)
    return best


def xcorr_lag_ms(x, y, t, max_lag=6):
    """Lag of y relative to x (positive = y later), sub-sample via parabola."""
    x = x - x.mean()
    y = y - y.mean()
    lags = np.arange(-max_lag, max_lag + 1)
    c = np.array([np.dot(x[max(0, -l):len(x) - max(0, l)],
                         y[max(0, l):len(y) - max(0, -l)]) /
                  (len(x) - abs(l)) for l in lags])
    i = int(np.argmax(c))
    lag = float(lags[i])
    if 0 < i < len(c) - 1:
        d = (c[i - 1] - c[i + 1]) / (2 * (c[i - 1] - 2 * c[i] + c[i + 1]) + 1e-12)
        lag += float(d)
    return lag * np.median(np.diff(t)) * 1000.0


def welch_tf(x, y, fs=100.0, nseg=256):
    """Cross-spectral transfer estimate H = Sxy/Sxx with coherence."""
    n = len(x)
    if n < nseg * 2:
        nseg = max(64, 2 ** int(math.log2(n // 3))) if n >= 192 else None
        if nseg is None:
            return None
    step = nseg // 2
    win = np.hanning(nseg)
    Sxx = None
    Syy = None
    Sxy = None
    cnt = 0
    for i0 in range(0, n - nseg + 1, step):
        xs = (x[i0:i0 + nseg] - x[i0:i0 + nseg].mean()) * win
        ys = (y[i0:i0 + nseg] - y[i0:i0 + nseg].mean()) * win
        X = np.fft.rfft(xs)
        Y = np.fft.rfft(ys)
        if Sxx is None:
            Sxx = np.abs(X) ** 2
            Syy = np.abs(Y) ** 2
            Sxy = np.conj(X) * Y
        else:
            Sxx += np.abs(X) ** 2
            Syy += np.abs(Y) ** 2
            Sxy += np.conj(X) * Y
        cnt += 1
    if cnt < 3:
        return None
    f = np.fft.rfftfreq(nseg, 1.0 / fs)
    H = Sxy / (Sxx + 1e-15)
    coh = np.abs(Sxy) ** 2 / (Sxx * Syy + 1e-15)
    return f, H, coh


def motor_frame(pos_sim, vel_sim):
    """Invert motor_bridge._motor_to_sim (offsets dropped — cancel in errors)."""
    pos_m = np.array(pos_sim, dtype=np.float64) * SIGN
    vel_m = np.array(vel_sim, dtype=np.float64) * SIGN
    for kn, hp in HP_OF_KN.items():
        pos_m[:, kn] = (pos_sim[:, kn] + COUP * pos_sim[:, hp]) * SIGN[kn] / GEAR
        vel_m[:, kn] = (vel_sim[:, kn] + COUP * vel_sim[:, hp]) * SIGN[kn] / GEAR
    return pos_m, vel_m


def sim_target_to_motor(tgt_sim, hip_actual_sim):
    """_sim_to_motor with offsets dropped (consistent with motor_frame)."""
    m = np.array(tgt_sim, dtype=np.float64) * SIGN
    for kn, hp in HP_OF_KN.items():
        m[:, kn] = (tgt_sim[:, kn] + hip_actual_sim[:, hp] * COUP) / GEAR * SIGN[kn]
    return m


# ════════════════════════════════════════════════════════════════════
# PART 1 — sysid npz: per-motor channel forensics
# ════════════════════════════════════════════════════════════════════

def analyze_sysid(path, results):
    d = np.load(path, allow_pickle=True)
    names = list(d["trial_names"])
    configs = json.loads(str(d["trial_configs"]))
    tag = os.path.basename(path)
    log(f"\n=== SYSID {tag}: {len(names)} trials ===")

    vel_fits = {j: [] for j in range(6)}
    tq_fits_fv = {j: [] for j in range(6)}   # kd uses firmware (reported) vel
    tq_fits_rv = {j: [] for j in range(6)}   # kd uses raw central-diff vel
    vel_lag_ms = {j: [] for j in range(6)}
    step_rows = []

    for nm, cfg in zip(names, configs):
        t = d[f"{nm}__time"].astype(np.float64)
        if len(t) < 30:
            continue
        p_des = d[f"{nm}__p_des"].astype(np.float64)
        pos = d[f"{nm}__pos"].astype(np.float64)
        vel = d[f"{nm}__vel"].astype(np.float64)
        tq = d[f"{nm}__torque"].astype(np.float64)     # sim/joint frame
        kp = d[f"{nm}__kp"].astype(np.float64)          # sim frame, 6-vec
        kd = d[f"{nm}__kd"].astype(np.float64)
        idx = cfg["idx"]

        pos_m, vel_m = motor_frame(pos, vel)
        # torque back to motor frame
        tq_m = tq * SIGN[None, :]
        for kn in KNEE_IDX:
            tq_m[:, kn] = tq[:, kn] * SIGN[kn] * GEAR

        # excited joint only (others barely move)
        j = idx
        dpos_m = central_diff(pos_m[:, j], t)
        moving = np.abs(dpos_m) > 0.05
        if moving.sum() > 60:
            fit = fit_scale_filter_delay(dpos_m, vel_m[:, j], deltas=(0,),
                                         mask=moving)
            if fit:
                vel_fits[j].append(fit)
            vel_lag_ms[j].append(xcorr_lag_ms(dpos_m[moving.argmax():],
                                              vel_m[moving.argmax():, j], t))

        # firmware-expected motor torque. Target executed during the sample at
        # tick k is p_des[k-1] (state is read before this tick's send).
        p_des_m = sim_target_to_motor(p_des, pos)
        if kp.ndim == 1:
            kp = np.broadcast_to(kp, (len(t), 6)).copy()
            kd = np.broadcast_to(kd, (len(t), 6)).copy()
        kp_m = kp.copy()
        kd_m = kd.copy()
        kp_m[:, list(KNEE_IDX)] *= GEAR ** 2
        kd_m[:, list(KNEE_IDX)] *= GEAR ** 2
        tgt = np.concatenate([[p_des_m[0, j]], p_des_m[:-1, j]])
        err = tgt - pos_m[:, j]
        tau_exp_fv = kp_m[:, j] * err - kd_m[:, j] * vel_m[:, j]
        tau_exp_rv = kp_m[:, j] * err - kd_m[:, j] * dpos_m
        active = np.abs(tau_exp_fv) > 0.15
        if active.sum() > 60:
            f1 = fit_scale_filter_delay(tau_exp_fv, tq_m[:, j], mask=active)
            f2 = fit_scale_filter_delay(tau_exp_rv, tq_m[:, j], mask=active)
            if f1:
                tq_fits_fv[j].append(f1)
            if f2:
                tq_fits_rv[j].append(f2)

        # STEP trials: onset shape of accel vs reported torque vs expected
        if cfg.get("signal") == "step":
            amp = cfg["amp"]
            acc = central_diff(dpos_m, t)
            # step happens at t=0 (first tick). Normalize each channel by its
            # own steady peak within 0.4 s.
            n40 = min(len(t), 41)
            sl = slice(0, n40)
            def rise_1090(sig):
                s = np.abs(sig[sl])
                pk = np.max(s)
                if pk < 1e-6:
                    return np.nan, np.nan
                i10 = np.argmax(s >= 0.1 * pk)
                i90 = np.argmax(s >= 0.9 * pk)
                return t[i10] * 1000.0, t[i90] * 1000.0
            a10, a90 = rise_1090(acc)
            q10, q90 = rise_1090(tq_m[:, j])
            e10, e90 = rise_1090(tau_exp_fv)
            step_rows.append(dict(trial=nm, joint=JNAMES[j], amp=amp,
                                  accel_t10=a10, accel_t90=a90,
                                  tqmeas_t10=q10, tqmeas_t90=q90,
                                  tqexp_t10=e10, tqexp_t90=e90))
            # save an overlay figure for the biggest steps
            if abs(amp) >= 0.55 or (j % 3 == 0 and abs(amp) >= 0.35):
                fig, ax = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
                ax[0].plot(t[sl] * 1000, p_des[sl, j], "k--", label="p_des (sim)")
                ax[0].plot(t[sl] * 1000, pos[sl, j], "b-", label="pos (raw)")
                ax[0].set_ylabel("rad")
                ax[0].legend(fontsize=7)
                ax[0].set_title(f"{tag} {nm} — {JNAMES[j]}")
                ax[1].plot(t[sl] * 1000, tau_exp_fv[sl], "k--", label="PD-expected τ")
                ax[1].plot(t[sl] * 1000, tq_m[sl, j], "r-", label="reported τ (torque_fdb)")
                ax[1].set_ylabel("Nm (motor)")
                ax[1].legend(fontsize=7)
                axa = ax[2]
                axa.plot(t[sl] * 1000, acc[sl], "g-", label="accel (d²pos/dt², raw)")
                axa.set_ylabel("rad/s²")
                axa.set_xlabel("ms since step")
                axa.legend(fontsize=7)
                fig.tight_layout()
                fig.savefig(os.path.join(OUT, f"step_{tag[:-4]}_{nm}.png"), dpi=110)
                plt.close(fig)

        # CHIRP: transfer functions
        if cfg.get("signal") == "chirp":
            tf1 = welch_tf(dpos_m, vel_m[:, j])
            tf2 = welch_tf(tau_exp_fv, tq_m[:, j])
            fig, ax = plt.subplots(2, 2, figsize=(10, 6))
            for (tf, col, name) in ((tf1, 0, "vel_reported / d(pos)/dt"),
                                    (tf2, 1, "τ_reported / τ_PD-expected")):
                if tf is None:
                    continue
                f, H, coh = tf
                good = (f > 0.3) & (f < 30) & (coh > 0.5)
                ax[0, col].semilogx(f[good], 20 * np.log10(np.abs(H[good]) + 1e-9), "b.-")
                ax[0, col].set_title(f"{JNAMES[j]}  |{name}|", fontsize=9)
                ax[0, col].set_ylabel("dB")
                ax[0, col].grid(True, which="both", alpha=0.3)
                ax[1, col].semilogx(f[good], np.degrees(np.angle(H[good])), "r.-")
                ax[1, col].set_ylabel("deg")
                ax[1, col].set_xlabel("Hz")
                ax[1, col].grid(True, which="both", alpha=0.3)
            fig.suptitle(f"{tag} {nm}", fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, f"chirp_tf_{tag[:-4]}_{nm}.png"), dpi=110)
            plt.close(fig)

    # aggregate
    def agg(fits):
        out = {}
        for j, fl in fits.items():
            if not fl:
                continue
            out[JNAMES[j]] = dict(
                tau_ms=float(np.median([f["tau_ms"] for f in fl])),
                tau_ms_iqr=[float(np.percentile([f["tau_ms"] for f in fl], 25)),
                            float(np.percentile([f["tau_ms"] for f in fl], 75))],
                scale=float(np.median([f["scale"] for f in fl])),
                scale_iqr=[float(np.percentile([f["scale"] for f in fl], 25)),
                           float(np.percentile([f["scale"] for f in fl], 75))],
                delta=int(np.median([f["delta"] for f in fl])),
                r2=float(np.median([f["r2"] for f in fl])),
                n=len(fl))
        return out

    res = dict(
        vel_filter=agg(vel_fits),
        vel_xcorr_lag_ms={JNAMES[j]: float(np.median(v)) for j, v in vel_lag_ms.items() if v},
        torque_filter_fwvel=agg(tq_fits_fv),
        torque_filter_rawvel=agg(tq_fits_rv),
        steps=step_rows,
    )
    results[f"sysid::{tag}"] = res

    log("\n  velocity channel vs d(pos)/dt  (per excited motor):")
    for k, v in res["vel_filter"].items():
        log(f"    {k:7s} tau={v['tau_ms']:5.1f} ms (IQR {v['tau_ms_iqr'][0]:.1f}-{v['tau_ms_iqr'][1]:.1f})"
            f"  scale={v['scale']:.3f}  R2={v['r2']:.3f}  n={v['n']}"
            f"   xcorr_lag={res['vel_xcorr_lag_ms'].get(k, float('nan')):.1f} ms")
    log("\n  reported torque vs PD-expected (kd on firmware vel):")
    for k, v in res["torque_filter_fwvel"].items():
        log(f"    {k:7s} tau={v['tau_ms']:5.1f} ms (IQR {v['tau_ms_iqr'][0]:.1f}-{v['tau_ms_iqr'][1]:.1f})"
            f"  scale={v['scale']:.3f} (IQR {v['scale_iqr'][0]:.2f}-{v['scale_iqr'][1]:.2f})"
            f"  delta={v['delta']}t  R2={v['r2']:.3f}  n={v['n']}")
    log("\n  reported torque vs PD-expected (kd on RAW dpos vel):")
    for k, v in res["torque_filter_rawvel"].items():
        log(f"    {k:7s} tau={v['tau_ms']:5.1f} ms  scale={v['scale']:.3f}"
            f"  delta={v['delta']}t  R2={v['r2']:.3f}  n={v['n']}")
    log("\n  STEP onset timing (ms to 10%/90% of own peak, first 400 ms):")
    log(f"    {'trial':<22s}{'joint':<8s}{'accel t10/t90':>16s}{'τ_meas t10/t90':>17s}{'τ_exp t10/t90':>17s}")
    for r in step_rows:
        log(f"    {r['trial']:<22s}{r['joint']:<8s}"
            f"{r['accel_t10']:7.0f}/{r['accel_t90']:<8.0f}"
            f"{r['tqmeas_t10']:8.0f}/{r['tqmeas_t90']:<8.0f}"
            f"{r['tqexp_t10']:8.0f}/{r['tqexp_t90']:<8.0f}")


# ════════════════════════════════════════════════════════════════════
# PART 2 — balance logs: delivery ratio with/without filter correction
# ════════════════════════════════════════════════════════════════════

def analyze_balance_logs(paths, label, results):
    log(f"\n=== BALANCE LOGS [{label}] — {len(paths)} files ===")
    per_joint_naive = {j: [] for j in range(6)}
    per_joint_filt = {j: [] for j in range(6)}
    per_joint_tau = {j: [] for j in range(6)}
    per_joint_vel_tau = {j: [] for j in range(6)}
    speed_bins = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    speed_ratio = {j: [[] for _ in range(len(speed_bins) - 1)] for j in range(6)}
    motoring_ratio = {j: {"motoring": [], "braking": []} for j in range(6)}
    tau_max_notes = set()
    nn_ticks_total = 0

    for p in paths:
        try:
            d = np.load(p, allow_pickle=True)
        except Exception as e:
            log(f"  [skip] {os.path.basename(p)}: {e}")
            continue
        need = ["time", "sim_joint_pos", "sim_joint_vel", "motor_tau_meas",
                "kp", "kd", "filtered_target"]
        if any(k not in d for k in need):
            log(f"  [skip] {os.path.basename(p)}: missing channels")
            continue
        t = d["time"].astype(np.float64)
        if len(t) < 40:
            continue
        pos = d["sim_joint_pos"].astype(np.float64)
        vel = d["sim_joint_vel"].astype(np.float64)
        tmeas = d["motor_tau_meas"].astype(np.float64)   # motor frame
        kp = d["kp"].astype(np.float64)
        kd = d["kd"].astype(np.float64)
        ftgt = d["filtered_target"].astype(np.float64)
        tau_max = d["tau_max"].astype(np.float64) if "tau_max" in d else None
        if tau_max is not None:
            # decode-range correction: firmware encodes ±FW_TAU_RANGE
            ratio = FW_TAU_RANGE[None, :] / np.maximum(tau_max, 1e-6)
            if np.any(np.abs(ratio - 1.0) > 1e-3):
                tau_max_notes.add(os.path.basename(p))
                tmeas = tmeas * ratio

        pos_m, vel_m = motor_frame(pos, vel)
        tgt_m = sim_target_to_motor(ftgt, pos)
        kp_m = kp.copy()
        kd_m = kd.copy()
        kp_m[:, list(KNEE_IDX)] *= GEAR ** 2
        kd_m[:, list(KNEE_IDX)] *= GEAR ** 2
        # v_des feedforward (KNEE_DECOUPLE_FF live on hardware since 08-06)
        vdes_m = np.zeros_like(vel_m)
        for kn, hp in HP_OF_KN.items():
            vdes_m[:, kn] = COUP * vel[:, hp] * SIGN[kn] / GEAR

        n = len(t)
        nn_ticks_total += n
        for j in range(6):
            tgt = np.concatenate([[tgt_m[0, j]], tgt_m[:-1, j]])
            vd = np.concatenate([[vdes_m[0, j]], vdes_m[:-1, j]])
            tau_exp = kp_m[:, j] * (tgt - pos_m[:, j]) + kd_m[:, j] * (vd - vel_m[:, j])
            active = (np.abs(tau_exp) > 0.3) & (kp[:, j] > 0.1)
            if active.sum() < 40:
                continue
            # naive scale (no filter)
            xs = tau_exp[active]
            ys = tmeas[active, j]
            s_naive = float(np.dot(xs, ys) / (np.dot(xs, xs) + 1e-9))
            per_joint_naive[j].append(s_naive)
            # filter-aware
            fit = fit_scale_filter_delay(tau_exp, tmeas[:, j], mask=active)
            if fit:
                per_joint_filt[j].append(fit["scale"])
                per_joint_tau[j].append(fit["tau_ms"])
            # velocity filter from this log too
            dpos_m = central_diff(pos_m[:, j], t)
            mov = np.abs(dpos_m) > 0.15
            if mov.sum() > 60:
                vfit = fit_scale_filter_delay(dpos_m, vel_m[:, j], deltas=(0,), mask=mov)
                if vfit:
                    per_joint_vel_tau[j].append(vfit["tau_ms"])
            # speed dependence of instantaneous delivery (filter-corrected via
            # best-fit reconstruction): ratio of measured to filtered-expected
            if fit and fit["tau_ms"] > 0.05:
                texp_f = iir(tau_exp, fit["g"])
            else:
                texp_f = tau_exp
            dd = fit["delta"] if fit else 0
            te = texp_f[: n - dd] if dd else texp_f
            tm = tmeas[dd:, j]
            vm = vel_m[dd:, j] if dd else vel_m[:, j]
            am = active[dd:] & (np.abs(te[: len(tm)]) > 1.0)
            r = tm[am] / te[: len(tm)][am]
            v_at = np.abs(vm[: len(tm)][am])
            for b in range(len(speed_bins) - 1):
                mb = (v_at >= speed_bins[b]) & (v_at < speed_bins[b + 1])
                if mb.sum() > 5:
                    speed_ratio[j][b].extend(r[mb].tolist())
            # motoring vs braking
            mot = (te[: len(tm)][am] * vm[: len(tm)][am]) > 0
            if mot.sum() > 5:
                motoring_ratio[j]["motoring"].extend(r[mot].tolist())
            if (~mot).sum() > 5:
                motoring_ratio[j]["braking"].extend(r[~mot].tolist())

    log(f"  total ticks analyzed: {nn_ticks_total}")
    if tau_max_notes:
        log(f"  [note] decode-range correction applied in {len(tau_max_notes)} logs "
            f"(tau_max != firmware range)")
    summary = {}
    log(f"\n  {'joint':<8s}{'naive scale':>12s}{'filt scale':>12s}{'τ_filter':>10s}"
        f"{'τ_vel':>8s}{'n_logs':>7s}")
    for j in range(6):
        if not per_joint_naive[j]:
            continue
        nv = np.median(per_joint_naive[j])
        fv = np.median(per_joint_filt[j]) if per_joint_filt[j] else float("nan")
        tv = np.median(per_joint_tau[j]) if per_joint_tau[j] else float("nan")
        vt = np.median(per_joint_vel_tau[j]) if per_joint_vel_tau[j] else float("nan")
        log(f"  {JNAMES[j]:<8s}{nv:12.3f}{fv:12.3f}{tv:9.1f}m{vt:7.1f}m"
            f"{len(per_joint_naive[j]):7d}")
        summary[JNAMES[j]] = dict(naive_scale=float(nv), filt_scale=float(fv),
                                  tau_filter_ms=float(tv), tau_vel_ms=float(vt),
                                  n=len(per_joint_naive[j]))
    log("\n  delivery ratio vs |motor speed| (median per bin, filter-corrected):")
    hdr = "  " + " " * 8 + "".join(f"{speed_bins[b]:.1f}-{speed_bins[b+1]:.1f}".rjust(10)
                                   for b in range(len(speed_bins) - 1))
    log(hdr)
    spd = {}
    for j in range(6):
        cells = []
        row = []
        for b in range(len(speed_bins) - 1):
            v = speed_ratio[j][b]
            if len(v) > 10:
                cells.append(f"{np.median(v):10.2f}")
                row.append(float(np.median(v)))
            else:
                cells.append(f"{'--':>10s}")
                row.append(None)
        if any(c.strip() != "--" for c in cells):
            log(f"  {JNAMES[j]:<8s}" + "".join(cells))
            spd[JNAMES[j]] = row
    log("\n  motoring vs braking delivery (median):")
    mb = {}
    for j in range(6):
        m = motoring_ratio[j]["motoring"]
        b = motoring_ratio[j]["braking"]
        if len(m) > 20 and len(b) > 20:
            log(f"  {JNAMES[j]:<8s} motoring={np.median(m):.2f} (n={len(m)})"
                f"   braking={np.median(b):.2f} (n={len(b)})")
            mb[JNAMES[j]] = dict(motoring=float(np.median(m)),
                                 braking=float(np.median(b)))
    results[f"balance::{label}"] = dict(summary=summary, speed_bins=speed_bins.tolist(),
                                        speed_ratio=spd, motoring_braking=mb)


# ════════════════════════════════════════════════════════════════════

def main():
    results = {}
    sysids = sorted(glob.glob(os.path.join(DATA, "sysid_leg1_2026*.npz")))
    sysids = [p for p in sysids if "replay" not in p and "synthetic" not in p]
    for p in sysids:
        analyze_sysid(p, results)

    g2i = sorted(glob.glob(os.path.join(DATA, "policy_G2i4760_log_*.npz")))
    f34 = sorted(glob.glob(os.path.join(DATA, "policy_F3490_log_*.npz")))
    analyze_balance_logs(g2i, "G2i4760 (08-13)", results)
    analyze_balance_logs(f34, "F3490 (08-11..13)", results)

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=1, default=str)
    with open(os.path.join(OUT, "results.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT))
    print(f"\nSaved -> {OUT}\\results.txt / results.json / *.png")


if __name__ == "__main__":
    main()
