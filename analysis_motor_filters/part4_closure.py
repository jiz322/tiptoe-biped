#!/usr/bin/env python3
"""part4 — closure tests.
[A] Fix the filter at tau=25ms, delta=1 and fit ONLY the scale s per sysid
    trial: if the knee's s rises from 0.31 to ~1, the knee "deficit" was the
    common telemetry filter acting on a fast, light joint.
[B] Knee triangle directional probe: friction shows up as a +/- offset split.
    If reported torque contains the friction the PD fights, offsets match the
    expectation; if reported ~ net (post-friction), reported offset ~ 0.
[C] Sweep tau in {15,20,25,30,40} to see sensitivity of s.
"""
import json
import math
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
VEL_FIX_JUNE = np.array([1.0, 44.0 / 50.0, 44.0 / 50.0, 1.0, 44.0 / 50.0, 44.0 / 50.0])
# June logs: RS01 vel decoded over +/-50 instead of +/-44 -> multiply by 44/50
# to recover true vel. EL05 (+/-50) was correct.

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


def iir_delay(x, tau_ms, delta):
    g = 1.0 - math.exp(-DT * 1000.0 / tau_ms) if tau_ms > 0.05 else 1.0
    y = np.empty_like(x)
    acc = x[0]
    for i in range(len(x)):
        acc += g * (x[i] - acc)
        y[i] = acc
    if delta > 0:
        y = np.concatenate([np.full(delta, y[0]), y[:-delta]])
    return y


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
    for path in sorted(glob.glob(os.path.join(DATA, "sysid_leg1_2026*.npz"))):
        if "replay" in path or "synthetic" in path:
            continue
        tag = os.path.basename(path)
        d = np.load(path, allow_pickle=True)
        names = list(d["trial_names"])
        configs = json.loads(str(d["trial_configs"]))
        log(f"\n=== {tag} ===")
        log("\n[A/C] fixed-filter scale fits (delta=1):  s per tau")
        log(f"  {'trial':<20s}{'joint':<8s}" + "".join(f"tau={t:d}ms".rjust(10) for t in (15, 20, 25, 30, 40)) + "     R2@25")
        for nm, cfg in zip(names, configs):
            t = d[f"{nm}__time"].astype(np.float64)
            if len(t) < 60:
                continue
            j = cfg["idx"]
            pos = d[f"{nm}__pos"].astype(np.float64)
            vel = d[f"{nm}__vel"].astype(np.float64) * VEL_FIX_JUNE[None, :]
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
            act = np.abs(texp) > 0.15
            if act.sum() < 60:
                continue
            row = f"  {nm:<20s}{JNAMES[j]:<8s}"
            r2_25 = float("nan")
            for tau in (15, 20, 25, 30, 40):
                xf = iir_delay(texp, tau, 1)
                xs = xf[act]
                ys = tq_m[act, j]
                s = float(np.dot(xs, ys) / (np.dot(xs, xs) + 1e-9))
                row += f"{s:10.2f}"
                if tau == 25:
                    resid = ys - s * xs
                    sst = float(np.dot(ys - ys.mean(), ys - ys.mean())) + 1e-12
                    r2_25 = 1.0 - float(np.dot(resid, resid)) / sst
            log(row + f"    {r2_25:5.2f}")

        # [B] knee triangle directional offsets
        log("\n[B] triangle directional split — friction accounting")
        log(f"  {'trial':<20s}{'joint':<8s}{'exp v>0':>9s}{'meas v>0':>10s}{'exp v<0':>9s}{'meas v<0':>10s}")
        for nm, cfg in zip(names, configs):
            if cfg.get("signal") != "triangle":
                continue
            t = d[f"{nm}__time"].astype(np.float64)
            j = cfg["idx"]
            pos = d[f"{nm}__pos"].astype(np.float64)
            vel = d[f"{nm}__vel"].astype(np.float64) * VEL_FIX_JUNE[None, :]
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
            quasi = np.abs(dv) < 6.0
            pos_v = quasi & (vel_m[:, j] > 0.2)
            neg_v = quasi & (vel_m[:, j] < -0.2)
            if pos_v.sum() < 20 or neg_v.sum() < 20:
                continue
            log(f"  {nm:<20s}{JNAMES[j]:<8s}"
                f"{np.median(texp[pos_v]):9.2f}{np.median(tq_m[pos_v, j]):10.2f}"
                f"{np.median(texp[neg_v]):9.2f}{np.median(tq_m[neg_v, j]):10.2f}")

    with open(os.path.join(OUT, "results_part4.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT))
    print(f"\nSaved -> {OUT}\\results_part4.txt")


if __name__ == "__main__":
    main()
