#!/usr/bin/env python3
"""part3 — knee quasi-static delivery, balance-log vel scales, and
filter-transparent knee delivery in the operating regime."""
import json
import os
import glob

import numpy as np

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


def main():
    # ── [A] knee (and all) triangle/cv quasi-static, robust extraction ──
    for path in sorted(glob.glob(os.path.join(DATA, "sysid_leg1_2026*.npz"))):
        if "replay" in path or "synthetic" in path:
            continue
        tag = os.path.basename(path)
        d = np.load(path, allow_pickle=True)
        names = list(d["trial_names"])
        configs = json.loads(str(d["trial_configs"]))
        log(f"\n=== {tag} — triangle quasi-static (robust) ===")
        log(f"  {'trial':<20s}{'joint':<8s}{'|v|med':>8s}{'a':>7s}{'b':>7s}{'n':>5s}")
        for nm, cfg in zip(names, configs):
            if cfg.get("signal") != "triangle":
                continue
            t = d[f"{nm}__time"].astype(np.float64)
            if len(t) < 60:
                continue
            j = cfg["idx"]
            pos = d[f"{nm}__pos"].astype(np.float64)
            vel = d[f"{nm}__vel"].astype(np.float64)
            tq = d[f"{nm}__torque"].astype(np.float64)
            p_des = d[f"{nm}__p_des"].astype(np.float64)
            kp = d[f"{nm}__kp"].astype(np.float64)
            kd = d[f"{nm}__kd"].astype(np.float64)
            if kp.ndim == 1:
                kp = np.broadcast_to(kp, (len(t), 6)).copy()
                kd = np.broadcast_to(kd, (len(t), 6)).copy()
            pos_m, vel_m = motor_frame(pos, vel)
            tq_m = tq * SIGN[None, :]
            for kn in KNEE_IDX:
                tq_m[:, kn] = tq[:, kn] * SIGN[kn] * GEAR
            p_des_m = sim_target_to_motor(p_des, pos)
            kp_m = kp.copy()
            kd_m = kd.copy()
            kp_m[:, list(KNEE_IDX)] *= GEAR ** 2
            kd_m[:, list(KNEE_IDX)] *= GEAR ** 2
            tgt = np.concatenate([[p_des_m[0, j]], p_des_m[:-1, j]])
            texp = kp_m[:, j] * (tgt - pos_m[:, j]) - kd_m[:, j] * vel_m[:, j]
            dv = central_diff(vel_m[:, j], t)
            # quasi-static: low accel, actually moving, expected torque slowly
            # varying (filter-transparent): |d(texp)/dt| small
            dte = central_diff(texp, t)
            m = (np.abs(dv) < 6.0) & (np.abs(vel_m[:, j]) > 0.25)
            m &= (np.abs(dte) < 8.0) & (np.abs(texp) > 0.15)
            if m.sum() < 30:
                continue
            X = np.vstack([texp[m], np.ones(m.sum())]).T
            (a, b), *_ = np.linalg.lstsq(X, tq_m[m, j], rcond=None)
            log(f"  {nm:<20s}{JNAMES[j]:<8s}{np.median(np.abs(vel_m[m, j])):8.2f}"
                f"{a:7.2f}{b:7.2f}{m.sum():5d}")

    # ── [B]+[C] balance logs: vel scale per motor; filter-transparent knee ──
    for pat, label in ((os.path.join(DATA, "policy_G2i4760_log_*.npz"), "G2i"),
                       (os.path.join(DATA, "policy_F3490_log_*.npz"), "F")):
        paths = sorted(glob.glob(pat))
        vel_scale = {j: [] for j in range(6)}
        vel_int = {j: [] for j in range(6)}
        trans_ratio = {j: [] for j in range(6)}
        for p in paths:
            d = np.load(p, allow_pickle=True)
            if "kp" not in d or len(d["time"]) < 40:
                continue
            t = d["time"].astype(np.float64)
            pos = d["sim_joint_pos"].astype(np.float64)
            vel = d["sim_joint_vel"].astype(np.float64)
            tq = d["motor_tau_meas"].astype(np.float64)
            ft = d["filtered_target"].astype(np.float64)
            kp = d["kp"].astype(np.float64)
            kd = d["kd"].astype(np.float64)
            pos_m, vel_m = motor_frame(pos, vel)
            tgt_m = sim_target_to_motor(ft, pos)
            kp_m = kp.copy()
            kd_m = kd.copy()
            kp_m[:, list(KNEE_IDX)] *= GEAR ** 2
            kd_m[:, list(KNEE_IDX)] *= GEAR ** 2
            vdes_m = np.zeros_like(vel_m)
            for kn, hp in HP_OF_KN.items():
                vdes_m[:, kn] = COUP * vel[:, hp] * SIGN[kn] / GEAR
            for j in range(6):
                dpos = central_diff(pos_m[:, j], t)
                mov = np.abs(dpos) > 0.3
                if mov.sum() > 40:
                    # LS scale (no filter — scale robust to mild filtering
                    # since spectra similar): use regression
                    s = float(np.dot(dpos[mov], vel_m[mov, j]) /
                              (np.dot(dpos[mov], dpos[mov]) + 1e-9))
                    vel_scale[j].append(s)
                    gross = np.abs(np.diff(pos_m[:, j])).sum()
                    if gross > 0.4:
                        pathv = float(np.sum(np.abs(vel_m[1:, j]) * np.diff(t)))
                        vel_int[j].append(pathv / gross)
                tgt = np.concatenate([[tgt_m[0, j]], tgt_m[:-1, j]])
                vd = np.concatenate([[vdes_m[0, j]], vdes_m[:-1, j]])
                texp = kp_m[:, j] * (tgt - pos_m[:, j]) + kd_m[:, j] * (vd - vel_m[:, j])
                dte = central_diff(texp, t)
                # filter-transparent: texp slowly varying over >=3 ticks
                m = (np.abs(dte) < 10.0) & (np.abs(texp) > 0.6) & (kp[:, j] > 0.1)
                m[:4] = False
                if m.sum() >= 6:
                    trans_ratio[j].extend((tq[m, j] / texp[m]).tolist())
        log(f"\n=== BALANCE [{label}] velocity channel scale (reported vs d(pos)/dt) ===")
        for j in range(6):
            if vel_scale[j]:
                vi = (f"{np.median(vel_int[j]):.3f}" if vel_int[j] else "--")
                log(f"  {JNAMES[j]:<8s} LS scale={np.median(vel_scale[j]):.3f} "
                    f"(IQR {np.percentile(vel_scale[j],25):.3f}-{np.percentile(vel_scale[j],75):.3f}, "
                    f"n={len(vel_scale[j])})   integral={vi}")
        log(f"\n=== BALANCE [{label}] FILTER-TRANSPARENT delivery "
            f"(|dτ_exp/dt|<10 Nm/s, |τ_exp|>0.6) ===")
        for j in range(6):
            if trans_ratio[j]:
                v = np.array(trans_ratio[j])
                log(f"  {JNAMES[j]:<8s} median={np.median(v):5.2f}  "
                    f"IQR {np.percentile(v,25):5.2f}-{np.percentile(v,75):5.2f}  n={len(v)}")

    with open(os.path.join(OUT, "results_part3.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT))
    print(f"\nSaved -> {OUT}\\results_part3.txt")


if __name__ == "__main__":
    main()
