#!/usr/bin/env python3
"""
part2_dc_and_quasistatic.py — filter-free discriminators.

W1 (telemetry lies): knee torque_fdb underscaled; plant delivers ~full.
W2 (plant weak): knee truly delivers 0.3-0.6 of PD command.

Discriminators (all filter-immune):
  1. STEP PLATEAU (DC): after settling, expected holding torque =
     kp_m * steady error. Filter gain at DC = 1, so reported/expected at the
     plateau is the pure telemetry scale.
  2. STEP ONSET: first ticks from rest, expected = kp_m * amp_m exactly.
  3. CV PLATEAUS: quasi-static at each sweep speed -> scale+offset vs speed.
  4. VEL INTEGRAL: integral of reported vel vs delta pos -> pure vel channel
     scale (filter DC gain = 1).
  5. CHIRP BAND-SPLIT: torque scale in 0.5-1.5 Hz vs 4-8 Hz bands.
Plus: dump per-run kp/kd/tau_max used in balance logs.
"""
import json
import math
import os
import glob

import numpy as np
from scipy.signal import butter, filtfilt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "analysis_motor_filters")

DT = 0.01
GEAR = 22.0 / 30.0
COUP = 0.733
SIGN = np.array([+1, +1, -1, +1, +1, -1], dtype=np.float64)
KNEE_IDX = (2, 5)
HP_OF_KN = {2: 1, 5: 4}
JNAMES = ["L_roll", "L_hip", "L_knee", "R_roll", "R_hip", "R_knee"]
FW_TAU_RANGE = np.array([6.0, 17.0, 17.0, 6.0, 17.0, 17.0])

REPORT = []


def log(msg=""):
    print(msg)
    REPORT.append(msg)


def central_diff(x, t):
    d = np.zeros_like(x)
    d[1:-1] = (x[2:] - x[:-2]) / (t[2:] - t[:-2])
    d[0] = d[1]
    d[-1] = d[-2]
    return d


def motor_frame(pos_sim, vel_sim):
    pos_m = np.array(pos_sim, dtype=np.float64) * SIGN
    vel_m = np.array(vel_sim, dtype=np.float64) * SIGN
    for kn, hp in HP_OF_KN.items():
        pos_m[:, kn] = (pos_sim[:, kn] + COUP * pos_sim[:, hp]) * SIGN[kn] / GEAR
        vel_m[:, kn] = (vel_sim[:, kn] + COUP * vel_sim[:, hp]) * SIGN[kn] / GEAR
    return pos_m, vel_m


def sim_target_to_motor(tgt_sim, hip_actual_sim):
    m = np.array(tgt_sim, dtype=np.float64) * SIGN
    for kn, hp in HP_OF_KN.items():
        m[:, kn] = (tgt_sim[:, kn] + hip_actual_sim[:, hp] * COUP) / GEAR * SIGN[kn]
    return m


def load_sysid(path):
    d = np.load(path, allow_pickle=True)
    names = list(d["trial_names"])
    configs = json.loads(str(d["trial_configs"]))
    trials = []
    for nm, cfg in zip(names, configs):
        t = d[f"{nm}__time"].astype(np.float64)
        if len(t) < 30:
            continue
        kp = d[f"{nm}__kp"].astype(np.float64)
        kd = d[f"{nm}__kd"].astype(np.float64)
        if kp.ndim == 1:
            kp = np.broadcast_to(kp, (len(t), 6)).copy()
            kd = np.broadcast_to(kd, (len(t), 6)).copy()
        trials.append(dict(
            name=nm, cfg=cfg, t=t,
            p_des=d[f"{nm}__p_des"].astype(np.float64),
            pos=d[f"{nm}__pos"].astype(np.float64),
            vel=d[f"{nm}__vel"].astype(np.float64),
            torque=d[f"{nm}__torque"].astype(np.float64),
            kp=kp, kd=kd))
    return trials


def motorize(tr):
    pos_m, vel_m = motor_frame(tr["pos"], tr["vel"])
    tq_m = tr["torque"] * SIGN[None, :]
    for kn in KNEE_IDX:
        tq_m[:, kn] = tr["torque"][:, kn] * SIGN[kn] * GEAR
    p_des_m = sim_target_to_motor(tr["p_des"], tr["pos"])
    kp_m = tr["kp"].copy()
    kd_m = tr["kd"].copy()
    kp_m[:, list(KNEE_IDX)] *= GEAR ** 2
    kd_m[:, list(KNEE_IDX)] *= GEAR ** 2
    return pos_m, vel_m, tq_m, p_des_m, kp_m, kd_m


def tau_expected(tr, j, pos_m, vel_m, p_des_m, kp_m, kd_m):
    tgt = np.concatenate([[p_des_m[0, j]], p_des_m[:-1, j]])
    return kp_m[:, j] * (tgt - pos_m[:, j]) - kd_m[:, j] * vel_m[:, j]


def main():
    results = {}

    # ════════ sysid-based discriminators ════════
    for path in sorted(glob.glob(os.path.join(DATA, "sysid_leg1_2026*.npz"))):
        if "replay" in path or "synthetic" in path:
            continue
        tag = os.path.basename(path)
        trials = load_sysid(path)
        log(f"\n=== {tag} ===")

        # 1) STEP PLATEAU (DC telemetry scale) + 2) STEP ONSET
        log("\n  [1] STEP PLATEAU (last 0.5 s) — DC telemetry scale, filter-free")
        log(f"      {'trial':<22s}{'err_inf':>9s}{'τ_exp':>8s}{'τ_meas':>8s}{'ratio':>7s}")
        plate = {}
        for tr in trials:
            if tr["cfg"].get("signal") != "step":
                continue
            j = tr["cfg"]["idx"]
            pos_m, vel_m, tq_m, p_des_m, kp_m, kd_m = motorize(tr)
            texp = tau_expected(tr, j, pos_m, vel_m, p_des_m, kp_m, kd_m)
            n = len(tr["t"])
            sl = slice(n - 50, n)
            if np.abs(vel_m[sl, j]).max() > 0.35:
                continue  # not settled
            e = float(np.median((p_des_m[sl, j] - pos_m[sl, j])))
            te = float(np.median(texp[sl]))
            tm = float(np.median(tq_m[sl, j]))
            r = tm / te if abs(te) > 0.15 else float("nan")
            log(f"      {tr['name']:<22s}{e:9.3f}{te:8.2f}{tm:8.2f}{r:7.2f}")
            plate.setdefault(JNAMES[j], []).append(r)

        log("\n  [2] STEP ONSET (ticks 1-3 from rest) — expected = kp_m·err, filter shows as ramp")
        log(f"      {'trial':<22s}{'τ_exp@1':>9s}{'meas@1':>8s}{'meas@2':>8s}{'meas@3':>8s}{'meas@5':>8s}")
        for tr in trials:
            if tr["cfg"].get("signal") != "step":
                continue
            j = tr["cfg"]["idx"]
            pos_m, vel_m, tq_m, p_des_m, kp_m, kd_m = motorize(tr)
            texp = tau_expected(tr, j, pos_m, vel_m, p_des_m, kp_m, kd_m)
            def rat(k):
                return tq_m[k, j] / texp[k] if abs(texp[k]) > 0.2 else float("nan")
            log(f"      {tr['name']:<22s}{texp[1]:9.2f}{rat(1):8.2f}{rat(2):8.2f}"
                f"{rat(3):8.2f}{rat(5):8.2f}")

        # 3) CV plateaus: quasi-static scale vs speed
        log("\n  [3] CV PLATEAUS — quasi-static τ_meas vs τ_exp (scale a, offset b)")
        log(f"      {'trial':<18s}{'v_tgt':>6s}{'|v|med':>8s}{'a':>7s}{'b':>7s}{'n':>5s}")
        for tr in trials:
            if tr["cfg"].get("signal") != "triangle":
                continue
            j = tr["cfg"]["idx"]
            vt = tr["cfg"].get("vel", float("nan"))
            pos_m, vel_m, tq_m, p_des_m, kp_m, kd_m = motorize(tr)
            texp = tau_expected(tr, j, pos_m, vel_m, p_des_m, kp_m, kd_m)
            dv = central_diff(vel_m[:, j], tr["t"])
            m = (np.abs(dv) < 8.0) & (np.abs(vel_m[:, j]) > 0.6 * abs(vt) / GEAR * 0.5)
            m &= np.abs(texp) > 0.1
            if m.sum() < 40:
                continue
            X = np.vstack([texp[m], np.ones(m.sum())]).T
            (a, b), *_ = np.linalg.lstsq(X, tq_m[m, j], rcond=None)
            log(f"      {tr['name']:<18s}{vt:6.2f}{np.median(np.abs(vel_m[m, j])):8.2f}"
                f"{a:7.2f}{b:7.2f}{m.sum():5d}")

        # 4) VEL INTEGRAL — channel scale at DC
        log("\n  [4] VEL INTEGRAL — sum(vel·dt) / Δpos per excited motor (DC gain of vel channel)")
        vint = {}
        for tr in trials:
            j = tr["cfg"]["idx"]
            pos_m, vel_m, _, _, _, _ = motorize(tr)
            t = tr["t"]
            # windows of monotonic displacement > 0.15 rad
            dp = pos_m[:, j] - pos_m[0, j]
            total = 0.0
            integ = 0.0
            k0 = 0
            for k in range(1, len(t)):
                integ_step = vel_m[k, j] * (t[k] - t[k - 1])
                integ += integ_step
            span = pos_m[-1, j] - pos_m[0, j]
            # use only trials with decent net or gross displacement
            gross = np.abs(np.diff(pos_m[:, j])).sum()
            if gross < 0.5:
                continue
            # gross-path comparison: integral of |vel| vs path length
            path_vel = float(np.sum(np.abs(vel_m[1:, j]) * np.diff(t)))
            ratio = path_vel / gross
            vint.setdefault(JNAMES[j], []).append(ratio)
        for k, v in vint.items():
            log(f"      {k:8s} ∫|vel|dt / Σ|Δpos| = {np.median(v):.3f}  "
                f"(IQR {np.percentile(v,25):.3f}-{np.percentile(v,75):.3f}, n={len(v)})")
        results[f"velint::{tag}"] = {k: float(np.median(v)) for k, v in vint.items()}

        # 5) CHIRP BAND-SPLIT
        log("\n  [5] CHIRP BAND-SPLIT — torque scale (LS) in low vs high band")
        for tr in trials:
            if tr["cfg"].get("signal") != "chirp":
                continue
            j = tr["cfg"]["idx"]
            pos_m, vel_m, tq_m, p_des_m, kp_m, kd_m = motorize(tr)
            texp = tau_expected(tr, j, pos_m, vel_m, p_des_m, kp_m, kd_m)
            fs = 100.0
            for (f1, f2, lbl) in ((0.5, 1.5, "0.5-1.5Hz"), (4.0, 8.0, "4-8Hz")):
                b, a = butter(3, [f1 / (fs / 2), f2 / (fs / 2)], btype="band")
                xf = filtfilt(b, a, texp)
                yf = filtfilt(b, a, tq_m[:, j])
                s = float(np.dot(xf, yf) / (np.dot(xf, xf) + 1e-9))
                log(f"      {tr['name']:<18s}{lbl:>10s}  scale={s:5.2f}")

    # ════════ balance logs: run configs + near-static ratio ════════
    for pat, label in ((os.path.join(DATA, "policy_G2i4760_log_*.npz"), "G2i"),
                       (os.path.join(DATA, "policy_F3490_log_*.npz"), "F")):
        paths = sorted(glob.glob(pat))
        log(f"\n=== BALANCE RUN CONFIGS [{label}] ===")
        log(f"  {'file':<42s}{'ticks':>6s}{'kp (hr/hp/kn)':>16s}{'kd':>14s}"
            f"{'tau_max':>13s}{'lpf':>5s}{'drop%':>7s}")
        static_ratio = {j: [] for j in range(6)}
        for p in paths:
            d = np.load(p, allow_pickle=True)
            if "kp" not in d or len(d["time"]) < 30:
                continue
            kp = d["kp"]
            kd = d["kd"]
            tm = d["tau_max"]
            lpf = d["lpf_alpha"]
            cr = d["can_replies"] if "can_replies" in d else None
            drop = 100.0 * np.mean(cr < 6) if cr is not None else float("nan")
            log(f"  {os.path.basename(p):<42s}{len(d['time']):>6d}"
                f"  {kp[0,0]:4.1f}/{kp[0,1]:4.1f}/{kp[0,2]:4.1f}"
                f"  {kd[0,0]:3.1f}/{kd[0,1]:3.1f}/{kd[0,2]:3.1f}"
                f"  {tm[0,0]:3.0f}/{tm[0,1]:3.0f}/{tm[0,2]:3.0f}"
                f"{lpf[0]:5.1f}{drop:7.1f}")
            # near-static early ticks: |vel| small on all joints
            pos = d["sim_joint_pos"].astype(np.float64)
            vel = d["sim_joint_vel"].astype(np.float64)
            tq = d["motor_tau_meas"].astype(np.float64)
            ft = d["filtered_target"].astype(np.float64)
            t = d["time"].astype(np.float64)
            pos_m, vel_m = motor_frame(pos, vel)
            tgt_m = sim_target_to_motor(ft, pos)
            kpm = d["kp"].astype(np.float64).copy()
            kdm = d["kd"].astype(np.float64).copy()
            kpm[:, list(KNEE_IDX)] *= GEAR ** 2
            kdm[:, list(KNEE_IDX)] *= GEAR ** 2
            for j in range(6):
                tgt = np.concatenate([[tgt_m[0, j]], tgt_m[:-1, j]])
                texp = kpm[:, j] * (tgt - pos_m[:, j]) - kdm[:, j] * vel_m[:, j]
                quiet = (np.abs(vel_m[:, j]) < 0.5) & (np.abs(texp) > 0.4)
                # exclude the first 3 ticks (handover transient)
                quiet[:3] = False
                if quiet.sum() >= 8:
                    r = tq[quiet, j] / texp[quiet]
                    static_ratio[j].extend(r.tolist())
        log(f"\n  NEAR-STATIC ratio τ_meas/τ_exp (|motor vel|<0.5 rad/s, |τ_exp|>0.4):")
        for j in range(6):
            if static_ratio[j]:
                v = np.array(static_ratio[j])
                log(f"    {JNAMES[j]:<8s} median={np.median(v):5.2f}  "
                    f"IQR {np.percentile(v,25):5.2f}-{np.percentile(v,75):5.2f}  n={len(v)}")
        results[f"static::{label}"] = {JNAMES[j]: float(np.median(static_ratio[j]))
                                       for j in range(6) if static_ratio[j]}

    with open(os.path.join(OUT, "results_part2.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT))
    with open(os.path.join(OUT, "results_part2.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nSaved -> {OUT}\\results_part2.txt")


if __name__ == "__main__":
    main()
