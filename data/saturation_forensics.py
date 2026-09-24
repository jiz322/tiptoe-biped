"""saturation_forensics.py — why does the firmware-PD torque fit collapse on
high-saturation logs? (R² 0.23-0.45 on the 32%-saturation runs vs 0.75-0.89
on calm runs, best lag stretching 2 -> 3-4 ticks; see
reports/belt_coupling_sim2real/report.md.)

Conditioned (masked) firmware-replica fits on existing hardware logs, no
hardware needed. Hypotheses each mask tests:

  mask "clean"/"dirty"  — CAN reply staleness: a tick with can_replies < 6 or
                          a near-overrun tick carries silently-stale state for
                          some motor (send_batch discards late replies; no
                          per-motor freshness flag). If "clean" restores R² on
                          the bad logs, the collapse is a MEASUREMENT artifact.
  mask "sat"/"unsat"    — does the misfit live inside the saturated samples
                          (plant nonlinearity: firmware current limit) or is
                          it spread over the whole log (systemic)?
  mask "vel+"/"vel-"    — direction-conditioned fit; knee-only asymmetry
                          would point at belt backlash/slip.
  mask "cmdQ4"/"cmdLo"  — top-quartile |tau_cmd|: current limiting shows as a
                          gain droop / one-sided residual only in Q4.

Prediction = the firmware-exact motor-frame PD from firmware_pd_replica.py
(stale target at t-k, belt-decoupled knee state, kp*G^2 gains), with the
mask/lag interaction handled properly: a sample at tick t only enters a
conditioned fit if BOTH t and its stale-target source tick t-k satisfy the
mask. Fits use OLS with an intercept (the legacy two_reg_fit had no intercept
but a mean-centred ss_tot, which biases segment comparisons); for a single
regressor with intercept R² == corr², so we report (k*, gain, intercept,
corr, R², n). The through-origin gain g0 is also printed for direct
comparison against firmware_pd_replica.py's gFW (sanity gate).

--knee-ff on adds the deployed KNEE_DECOUPLE_FF v_des term
(+kd_m * c * qd_hp[t-k] * sign / G) to the knee prediction. All 2026-08-06
logs predate the patch -> default off. Required for any post-patch log
(balance_log's motor_tau_cmd channel does not include it).

Run from repo root:
    python data/saturation_forensics.py                # default log set
    python data/saturation_forensics.py --logs control/model_4150_log_20260806_115256.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "control"))
import config as C  # noqa: E402

G = C.KNEE_GEAR_RATIO
COUP = C.KNEE_HIP_COUPLING
SIGN = np.array(C.JOINT_SIGN, dtype=np.float64)
OFF = np.array(C.JOINT_OFFSET, dtype=np.float64)
NAMES = ["hr1", "hp1", "kn1", "hr2", "hp2", "kn2"]
HIP_OF_KNEE = {2: 1, 5: 4}

# The 7-log coupling corpus + the other three 22-channel hardware logs.
DEFAULT_LOGS = [
    "model_4150_log_20260806_115256.npz",
    "model_4150_log_20260806_115327.npz",
    "model_4150_log_20260806_115703.npz",
    "model_4150_log_20260806_115715.npz",
    "2026-06-20_22-02-53_jrk020_grf13_log_20260806_120142.npz",
    "2026-06-20_22-02-53_jrk020_grf13_log_20260806_120222.npz",
    "2026-06-20_22-02-53_jrk020_grf13_log_20260806_120301.npz",
    "model_4150_log_20260806_084306.npz",
    "2026-06-15_07-07-49_jrk032_grf14_log_20260806_084106.npz",
    "2026-06-20_22-02-53_jrk020_grf13_log_20260806_083901.npz",
]
# Logs the belt replay flagged as regressing under the coupling model.
BAD_LOGS = {"model_4150_log_20260806_115256.npz",
            "model_4150_log_20260806_115327.npz",
            "2026-06-20_22-02-53_jrk020_grf13_log_20260806_120301.npz"}

# dataviz reference palette (light mode)
BLUE, ORANGE, RED, AQUA = "#2a78d6", "#eb6834", "#e34948", "#1baf7a"
INK, INK2, MUTED, GRID, BASELINE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "text.color": INK,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 8.5, "legend.frameon": False, "axes.axisbelow": True,
})


def build_prediction(d, j, knee_ff):
    """Full-length motor-frame prediction pieces for joint j.

    Returns (th_des, th, thd, kpm, kdm, vdes) where the lag-k prediction is
        tau_fw[t] = kpm[t]*(th_des[t-k] - th[t]) + kdm[t]*(vdes[t-k] - thd[t])
    vdes is zero unless knee_ff and j is a knee (the deployed v_des tracks the
    belt-forced motor spin computed from the tick-start hip velocity, i.e. the
    same stale tick as the target).
    """
    pos = d["sim_joint_pos"].astype(np.float64)
    vel = d["sim_joint_vel"].astype(np.float64)
    filt = d["filtered_target"].astype(np.float64)
    kp = d["kp"].astype(np.float64)
    kd = d["kd"].astype(np.float64)
    n = pos.shape[0]
    vdes = np.zeros(n)
    if j in HIP_OF_KNEE:
        hp = HIP_OF_KNEE[j]
        th_des = (filt[:, j] - OFF[j] + pos[:, hp] * COUP) / G * SIGN[j]
        th = (pos[:, j] - OFF[j] + pos[:, hp] * COUP) / G * SIGN[j]
        thd = (vel[:, j] + COUP * vel[:, hp]) * SIGN[j] / G
        kpm = kp[:, j] * G * G
        kdm = kd[:, j] * G * G
        if knee_ff:
            vdes = COUP * vel[:, hp] * SIGN[j] / G   # motor_bridge.py:231
    else:
        th_des = (filt[:, j] - OFF[j]) * SIGN[j]
        th = (pos[:, j] - OFF[j]) * SIGN[j]
        thd = vel[:, j] * SIGN[j]
        kpm = kp[:, j]
        kdm = kd[:, j]
    return th_des, th, thd, kpm, kdm, vdes


def tau_fw_at_lag(pieces, k):
    th_des, th, thd, kpm, kdm, vdes = pieces
    n = len(th)
    tau = np.full(n, np.nan)
    if k == 0:
        tau = kpm * (th_des - th) + kdm * (vdes - thd)
    else:
        tau[k:] = (kpm[k:] * (th_des[:-k] - th[k:])
                   + kdm[k:] * (vdes[:-k] - thd[k:]))
    return tau


def fit(pred, meas):
    """OLS meas ~ a*pred + b. Returns (g0, a, b, corr, r2, n).
    g0 = through-origin gain (firmware_pd_replica.py continuity)."""
    n = len(pred)
    if n < 30 or np.std(pred) < 1e-9 or np.std(meas) < 1e-9:
        return (np.nan,) * 5 + (n,)
    g0 = float(np.dot(pred, meas) / np.dot(pred, pred))
    X = np.stack([pred, np.ones(n)], axis=1)
    (a, b), *_ = np.linalg.lstsq(X, meas, rcond=None)
    c = float(np.corrcoef(pred, meas)[0, 1])
    return g0, float(a), float(b), c, c * c, n


def arx_fit(pieces, meas, mask, warmup, max_lag):
    """First-order-filter test:  tau_meas[t] ~ alpha*tau_meas[t-1]
    + beta*tau_fw[t](lag k) + b.  If the reported torque is a low-passed
    version of the executed PD torque, this recovers R2 with a consistent
    alpha across logs and DC gain beta/(1-alpha) ~ 1; the plain fit's
    amplitude-dependent 'gain' and stretched lag are then artifacts of
    fitting a filter with a pure gain."""
    n = len(meas)
    best = None
    for k in range(max_lag + 1):
        tau = tau_fw_at_lag(pieces, k)
        valid = np.zeros(n, dtype=bool)
        valid[max(warmup, k, 1):] = True
        valid &= mask & ~np.isnan(tau)
        if k > 0:
            src_ok = np.zeros(n, dtype=bool)
            src_ok[k:] = mask[:-k]
            valid &= src_ok
        prev_ok = np.zeros(n, dtype=bool)
        prev_ok[1:] = mask[:-1]
        valid &= prev_ok
        m = valid.sum()
        if m < 40:
            continue
        prev = np.empty(n)
        prev[1:] = meas[:-1]
        prev[0] = meas[0]
        X = np.stack([prev[valid], tau[valid], np.ones(m)], axis=1)
        y = meas[valid]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - float(np.sum(resid ** 2)) / max(ss_tot, 1e-12)
        if best is None or r2 > best["r2"]:
            alpha = float(coef[0])
            dc = float(coef[1] / (1.0 - alpha)) if alpha < 1.0 else np.nan
            tau_s = (-0.01 / np.log(alpha)) if 0.0 < alpha < 1.0 else np.nan
            best = {"k": k, "alpha": alpha, "beta": float(coef[1]),
                    "dc_gain": dc, "tau_s": tau_s, "b": float(coef[2]),
                    "r2": r2, "n": int(m)}
    return best or {"k": np.nan, "alpha": np.nan, "beta": np.nan,
                    "dc_gain": np.nan, "tau_s": np.nan, "b": np.nan,
                    "r2": np.nan, "n": 0}


def etfe(pred, meas, dt=0.01, nseg=256):
    """Welch empirical transfer function tau_fw -> tau_meas.
    Returns (freqs, |H|, phase_deg, coherence). Hand-rolled (numpy only)."""
    n = min(len(pred), len(meas))
    if n < nseg * 2:
        nseg = max(64, 1 << int(np.log2(n // 4))) if n >= 256 else 64
    step = nseg // 2
    win = np.hanning(nseg)
    pxx = pyy = pxy = None
    count = 0
    for s in range(0, n - nseg + 1, step):
        x = (pred[s:s + nseg] - pred[s:s + nseg].mean()) * win
        y = (meas[s:s + nseg] - meas[s:s + nseg].mean()) * win
        X = np.fft.rfft(x)
        Y = np.fft.rfft(y)
        if pxx is None:
            pxx = np.abs(X) ** 2
            pyy = np.abs(Y) ** 2
            pxy = np.conj(X) * Y
        else:
            pxx += np.abs(X) ** 2
            pyy += np.abs(Y) ** 2
            pxy += np.conj(X) * Y
        count += 1
    if count == 0:
        return None
    freqs = np.fft.rfftfreq(nseg, dt)
    H = pxy / np.maximum(pxx, 1e-12)
    coh = np.abs(pxy) ** 2 / np.maximum(pxx * pyy, 1e-12)
    return freqs, np.abs(H), np.degrees(np.angle(H)), coh


def friction_fit(pieces, meas, mask, warmup, max_lag):
    """Friction-augmented fit:
        tau_meas ~ a*tau_fw + f_c*tanh(thd_m/0.2) + f_v*thd_m + b
    with thd_m the MOTOR-shaft velocity (belt-effective on knees). If the base
    fit's misfit is unmodeled motor/belt Coulomb+viscous friction, this
    recovers R² and yields a stable gain `a` across calm and aggressive logs.
    Lag applies to the stale target only; friction rides the live velocity."""
    thd = pieces[2]
    n = len(meas)
    sgn = np.tanh(thd / 0.2)
    best = None
    for k in range(max_lag + 1):
        tau = tau_fw_at_lag(pieces, k)
        valid = np.zeros(n, dtype=bool)
        valid[max(warmup, k):] = True
        valid &= mask & ~np.isnan(tau)
        if k > 0:
            src_ok = np.zeros(n, dtype=bool)
            src_ok[k:] = mask[:-k]
            valid &= src_ok
        m = valid.sum()
        if m < 40:
            continue
        X = np.stack([tau[valid], sgn[valid], thd[valid], np.ones(m)], axis=1)
        y = meas[valid]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - float(np.sum(resid ** 2)) / max(ss_tot, 1e-12)
        if best is None or r2 > best["r2"]:
            best = {"k": k, "a": float(coef[0]), "f_c": float(coef[1]),
                    "f_v": float(coef[2]), "b": float(coef[3]),
                    "r2": r2, "n": int(m)}
    return best or {"k": np.nan, "a": np.nan, "f_c": np.nan, "f_v": np.nan,
                    "b": np.nan, "r2": np.nan, "n": 0}


def masked_fit(pieces, meas, mask, warmup, max_lag):
    """Best-lag fit under a mask, requiring both t and the stale-target
    source t-k to satisfy the mask. Returns dict with best row + lag profile."""
    n = len(meas)
    profile = []
    best = None
    for k in range(max_lag + 1):
        tau = tau_fw_at_lag(pieces, k)
        valid = np.zeros(n, dtype=bool)
        valid[max(warmup, k):] = True
        valid &= mask
        if k > 0:
            src_ok = np.zeros(n, dtype=bool)
            src_ok[k:] = mask[:-k]
            valid &= src_ok
        valid &= ~np.isnan(tau)
        row = fit(tau[valid], meas[valid])
        profile.append((k,) + row)
        if not np.isnan(row[3]) and (best is None or row[3] > best[4]):
            best = (k,) + row
    if best is None:
        best = (np.nan,) + (np.nan,) * 5 + (0,)
    k, g0, a, b, c, r2, nn = best
    return {"k": k, "g0": g0, "a": a, "b": b, "c": c, "r2": r2, "n": int(nn),
            "profile": profile}


def build_masks(d, j):
    """Per-joint condition masks over ticks."""
    n = len(d["time"])
    action = d["action"].astype(np.float64)
    vel = d["sim_joint_vel"].astype(np.float64)
    tau_cmd = d["motor_tau_cmd"].astype(np.float64)
    ones = np.ones(n, dtype=bool)
    masks = {"all": ones}
    sat = np.abs(action[:, j]) >= 0.999
    masks["sat"] = sat
    masks["unsat"] = ~sat
    if "can_replies" in d.files and "tick_busy_ms" in d.files:
        clean = (d["can_replies"].astype(int) == 6) \
            & (d["tick_busy_ms"].astype(np.float64) < 9.5)
        masks["clean"] = clean
        masks["dirty"] = ~clean
    masks["vel+"] = vel[:, j] > 0.2
    masks["vel-"] = vel[:, j] < -0.2
    q3 = np.percentile(np.abs(tau_cmd[:, j]), 75)
    masks["cmdQ4"] = np.abs(tau_cmd[:, j]) >= q3
    masks["cmdLo"] = np.abs(tau_cmd[:, j]) < q3
    return masks


MASK_ORDER = ["all", "sat", "unsat", "clean", "dirty",
              "vel+", "vel-", "cmdQ4", "cmdLo"]


def plateau_stats(d, j):
    """Current-limit / decode-rail signature: how close does |tau_meas| get to
    the decode ceiling (tau_max), and how often does it sit there?"""
    tm = np.abs(d["motor_tau_meas"].astype(np.float64)[:, j])
    ceil = float(d["tau_max"][0][j]) if "tau_max" in d.files else 17.0
    return {"tau_ceiling": ceil,
            "p99_abs_tau": float(np.percentile(tm, 99)),
            "max_abs_tau": float(tm.max()),
            "frac_ge_95pct_ceiling": float((tm >= 0.95 * ceil).mean()),
            "frac_at_rail": float((tm >= 0.999 * ceil).mean())}


def make_pdf(stem, d, results, knee_ff, warmup):
    tau_meas = d["motor_tau_meas"].astype(np.float64)
    action = d["action"].astype(np.float64)
    with PdfPages(stem + ".pdf") as pdf:
        # page 1: tau_pred(best-k, all) vs tau_meas scatter, coloured by saturation
        fig, axes = plt.subplots(3, 2, figsize=(10, 10))
        for j in range(6):
            ax = axes[j % 3][j // 3]
            r = results[j]["all"]
            pieces = build_prediction(d, j, knee_ff)
            tau = tau_fw_at_lag(pieces, int(r["k"]) if not np.isnan(r["k"]) else 0)
            ok = ~np.isnan(tau)
            ok[:warmup] = False
            sat = np.abs(action[:, j]) >= 0.999
            ax.scatter(tau[ok & ~sat], tau_meas[ok & ~sat, j], s=4, alpha=0.35,
                       color=BLUE, label="unsaturated")
            ax.scatter(tau[ok & sat], tau_meas[ok & sat, j], s=5, alpha=0.5,
                       color=RED, label="|action|=1")
            lim = np.nanmax(np.abs(tau[ok])) if ok.any() else 1.0
            xs = np.array([-lim, lim])
            ax.plot(xs, r["a"] * xs + r["b"], color=INK2, lw=1.0, ls=(0, (4, 3)))
            ax.set_title(f"{NAMES[j]}  k*={r['k']:.0f} a={r['a']:.2f} "
                         f"R²={r['r2']:.2f}", fontsize=8.5)
            if j == 0:
                ax.legend(fontsize=7, markerscale=2)
        for ax in axes[-1]:
            ax.set_xlabel("tau_fw prediction (Nm, motor frame)")
        for row in axes:
            row[0].set_ylabel("tau_meas (Nm)")
        fig.suptitle(f"{os.path.basename(stem)} — firmware-PD fit at best lag "
                     "(dashed = intercept fit)", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig); plt.close(fig)

        # page 2: lag profiles (corr vs k) for key masks
        fig, axes = plt.subplots(3, 2, figsize=(10, 10))
        for j in range(6):
            ax = axes[j % 3][j // 3]
            for mname, col in [("all", INK2), ("clean", BLUE),
                               ("dirty", MUTED), ("sat", RED), ("unsat", AQUA)]:
                if mname not in results[j]:
                    continue
                prof = results[j][mname]["profile"]
                ks = [p[0] for p in prof]
                cs = [p[4] for p in prof]
                ax.plot(ks, cs, "-o", ms=3.5, lw=1.2, color=col, label=mname)
            ax.set_title(NAMES[j], fontsize=9)
            ax.set_ylim(0, 1)
            if j == 0:
                ax.legend(fontsize=7, ncols=2)
        for ax in axes[-1]:
            ax.set_xlabel("stale-target lag k (ticks)")
        for row in axes:
            row[0].set_ylabel("corr(tau_fw, tau_meas)")
        fig.suptitle(f"{os.path.basename(stem)} — lag profile by condition",
                     fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig); plt.close(fig)

        # page 3: residual vs |tau_cmd| binned (current-limit droop check)
        tau_cmd = d["motor_tau_cmd"].astype(np.float64)
        fig, axes = plt.subplots(3, 2, figsize=(10, 10))
        for j in range(6):
            ax = axes[j % 3][j // 3]
            r = results[j]["all"]
            pieces = build_prediction(d, j, knee_ff)
            tau = tau_fw_at_lag(pieces, int(r["k"]) if not np.isnan(r["k"]) else 0)
            ok = ~np.isnan(tau)
            ok[:warmup] = False
            resid = tau_meas[:, j] - (r["a"] * tau + r["b"])
            x = np.abs(tau_cmd[:, j])
            edges = np.quantile(x[ok], np.linspace(0, 1, 9))
            centers, means, p90s = [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = ok & (x >= lo) & (x <= hi)
                if m.sum() < 10:
                    continue
                centers.append(0.5 * (lo + hi))
                means.append(resid[m].mean())
                p90s.append(np.percentile(np.abs(resid[m]), 90))
            ax.plot(centers, means, "-o", ms=4, color=BLUE, label="mean resid")
            ax.plot(centers, p90s, "-o", ms=4, color=ORANGE, label="p90 |resid|")
            ax.axhline(0, color=BASELINE, lw=1.0)
            ax.set_title(NAMES[j], fontsize=9)
            if j == 0:
                ax.legend(fontsize=7)
        for ax in axes[-1]:
            ax.set_xlabel("|tau_cmd| (Nm) — octile bins")
        for row in axes:
            row[0].set_ylabel("residual (Nm)")
        fig.suptitle(f"{os.path.basename(stem)} — fit residual vs command "
                     "magnitude", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig); plt.close(fig)


def analyze_log(path, knee_ff, warmup, max_lag, make_plots=True):
    d = np.load(path, allow_pickle=True)
    need = {"filtered_target", "kp", "kd"}
    if not need.issubset(set(d.files)):
        print(f"[skip] {os.path.basename(path)} — pre-patch schema "
              f"(missing {sorted(need - set(d.files))})")
        return None
    n = len(d["time"])
    w = min(warmup, n // 4)
    tau_meas = d["motor_tau_meas"].astype(np.float64)
    results = {}
    for j in range(6):
        pieces = build_prediction(d, j, knee_ff)
        masks = build_masks(d, j)
        results[j] = {mname: masked_fit(pieces, tau_meas[:, j], m, w, max_lag)
                      for mname, m in masks.items()}
        results[j]["plateau"] = plateau_stats(d, j)
        results[j]["fric"] = friction_fit(pieces, tau_meas[:, j],
                                          masks["all"], w, max_lag)
        results[j]["arx"] = arx_fit(pieces, tau_meas[:, j],
                                    masks["all"], w, max_lag)
        results[j]["etfe"] = etfe(tau_fw_at_lag(pieces, 0)[w:],
                                  tau_meas[w:, j])

    base = os.path.basename(path)
    sat_frac = float((np.abs(d["action"]) >= 0.999).mean())
    dirty_frac = (float((d["can_replies"].astype(int) < 6).mean())
                  if "can_replies" in d.files else np.nan)
    print(f"\n=== {base}  (n={n}, |action|sat={sat_frac:.1%}, "
          f"drop-ticks={dirty_frac:.1%}) ===")
    print(f"{'j':<4}{'mask':<7}{'n':>6}{'k*':>4}{'gain':>7}{'icept':>7}"
          f"{'corr':>7}{'R2':>6}")
    for j in range(6):
        for mname in MASK_ORDER:
            if mname not in results[j]:
                continue
            r = results[j][mname]
            if r["n"] < 30:
                print(f"{NAMES[j]:<4}{mname:<7}{r['n']:>6}    (too few samples)")
                continue
            print(f"{NAMES[j]:<4}{mname:<7}{r['n']:>6}{r['k']:>4.0f}"
                  f"{r['a']:>7.2f}{r['b']:>7.2f}{r['c']:>7.2f}{r['r2']:>6.2f}")
        p = results[j]["plateau"]
        print(f"{NAMES[j]:<4}{'rail':<7}  p99|tau|={p['p99_abs_tau']:5.2f}"
              f"/{p['tau_ceiling']:.0f}Nm  ge95%={p['frac_ge_95pct_ceiling']:.1%}"
              f"  at-rail={p['frac_at_rail']:.2%}")
        fr = results[j]["fric"]
        print(f"{NAMES[j]:<4}{'fric':<7}{fr['n']:>6}{fr['k']:>4.0f}"
              f"{fr['a']:>7.2f}  fc={fr['f_c']:+.2f} fv={fr['f_v']:+.3f}"
              f"  R2={fr['r2']:.2f}")
        ax = results[j]["arx"]
        print(f"{NAMES[j]:<4}{'arx':<7}{ax['n']:>6}{ax['k']:>4.0f}"
              f"  alpha={ax['alpha']:.3f} tau={ax['tau_s']*1e3 if ax['tau_s']==ax['tau_s'] else float('nan'):.0f}ms"
              f" dc={ax['dc_gain']:.2f}  R2={ax['r2']:.2f}")
        tf = results[j]["etfe"]
        if tf is not None:
            fr, mag, ph, coh = tf
            cells = []
            for f0 in (1.0, 2.5, 5.0, 10.0):
                i = int(np.argmin(np.abs(fr - f0)))
                cells.append(f"{f0:g}Hz |H|={mag[i]:.2f} coh={coh[i]:.2f}")
            print(f"{NAMES[j]:<4}{'etfe':<7}  " + "  ".join(cells))

    stem = os.path.splitext(path)[0] + "_satforensics"
    out = {"n": n, "sat_frac": sat_frac, "dirty_frac": dirty_frac,
           "knee_ff": knee_ff}
    for j in range(6):
        out[NAMES[j]] = {m: ({k: v for k, v in r.items() if k != "profile"}
                             if isinstance(r, dict) and "profile" in r else r)
                         for m, r in results[j].items() if m != "etfe"}
    with open(stem + ".json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    if make_plots:
        make_pdf(stem, d, results, knee_ff, w)
        print(f"  -> {stem}.json / .pdf")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=None, help="comma list of npz paths")
    ap.add_argument("--logs-dir", default=os.path.join(_ROOT, "control"))
    ap.add_argument("--knee-ff", choices=["on", "off"], default="off",
                    help="model the deployed v_des FF (post-2026-08-06 logs)")
    ap.add_argument("--max-lag", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    if args.logs:
        paths = [p if os.path.isabs(p) or os.path.exists(p)
                 else os.path.join(args.logs_dir, p)
                 for p in args.logs.split(",")]
    else:
        paths = [os.path.join(args.logs_dir, f) for f in DEFAULT_LOGS]

    knee_ff = args.knee_ff == "on"
    all_results = {}
    for p in paths:
        if not os.path.isfile(p):
            print(f"[skip] missing {p}")
            continue
        r = analyze_log(p, knee_ff, args.warmup, args.max_lag,
                        make_plots=not args.no_plots)
        if r is not None:
            all_results[os.path.basename(p)] = r

    # ── aggregate: the verdict table ─────────────────────────────────
    print("\n" + "=" * 86)
    print("AGGREGATE — mean R² per condition (knees kn1/kn2 pooled, "
          "hips hp1/hp2 pooled)")
    print(f"{'group':<18}" + "".join(f"{m:>8}" for m in
          ["all", "sat", "unsat", "clean", "dirty", "cmdQ4", "fric", "arx"]))
    for label, sel in [("BAD logs", lambda b: b in BAD_LOGS),
                       ("good logs", lambda b: b not in BAD_LOGS)]:
        logs = [b for b in all_results if sel(b)]
        for gname, jids in [("knees", (2, 5)), ("hip_pitch", (1, 4))]:
            vals = []
            for m in ["all", "sat", "unsat", "clean", "dirty", "cmdQ4",
                      "fric", "arx"]:
                xs = [all_results[b][j][m]["r2"] for b in logs for j in jids
                      if m in all_results[b][j]
                      and all_results[b][j][m]["n"] >= 30
                      and not np.isnan(all_results[b][j][m]["r2"])]
                vals.append(np.mean(xs) if xs else np.nan)
            print(f"{label + ' ' + gname:<18}" + "".join(
                f"{v:>8.2f}" if not np.isnan(v) else f"{'—':>8}" for v in vals))
    print("=" * 86)
    print("Read: if BAD-log 'clean' recovers toward good-log 'all' -> reply "
          "staleness artifact.\n      if 'sat'/'cmdQ4' alone collapse while "
          "'unsat' holds -> plant nonlinearity (current limit).\n      "
          "knee-only vel+/vel- asymmetry (per-log tables) -> belt "
          "backlash/slip.")

    # ── cross-log transfer function: does |H(f)| collapse to one curve? ──
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), sharex=True)
    for col, (j, jname) in enumerate([(2, "kn1"), (1, "hp1")]):
        for base, res in all_results.items():
            tf = res[j].get("etfe")
            if tf is None:
                continue
            fr, mag, ph, coh = tf
            col_c = RED if base in BAD_LOGS else BLUE
            lw = 1.5 if base in BAD_LOGS else 1.1
            keep = fr <= 15.0
            axes[0][col].plot(fr[keep], mag[keep], color=col_c, lw=lw,
                              alpha=0.85)
            axes[1][col].plot(fr[keep], coh[keep], color=col_c, lw=lw,
                              alpha=0.85)
        axes[0][col].set_title(f"{jname} — |H(f)| tau_fw -> tau_meas",
                               fontsize=10)
        axes[0][col].axhline(1.0, color=BASELINE, lw=1.0, ls=(0, (4, 3)))
        axes[0][col].set_ylim(0, 1.6)
        axes[1][col].set_ylim(0, 1.05)
        axes[1][col].set_ylabel("coherence")
        axes[1][col].set_xlabel("frequency (Hz)")
    axes[0][0].set_ylabel("|H|")
    axes[0][0].plot([], [], color=RED, label="regressing logs")
    axes[0][0].plot([], [], color=BLUE, label="other logs")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Empirical transfer function per log — a fixed measurement "
                 "filter would collapse all curves", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = os.path.join(args.logs_dir, "_satforensics_tf.png")
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"\ntransfer-function overlay -> {out_png}")


if __name__ == "__main__":
    main()
