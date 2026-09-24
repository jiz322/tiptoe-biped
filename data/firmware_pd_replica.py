"""firmware_pd_replica.py — decisive test: is the torque "attenuation" real,
or an artifact of comparing tau_meas against the wrong prediction?

tau_cmd (logged) = kp*(filtered_target[t] - pos[t]) - kd*vel[t]   at SEND time.
But the motor firmware, when its state/torque is sampled at tick t, is still
executing the target sent at tick t-1 (or earlier), evaluated at the CURRENT
position. For the knees the stale target also bakes in a STALE hip position
inside the belt-decoupling term — the user's "coupling cancellation":
when the hip moves within a tick, the belt drags the knee motor, its PD
fights the drag, and none of that appears in tau_cmd.

Firmware-exact replica, per joint, lag k (which stale target the firmware
is holding when sampled):
  non-knee:  tau_fw[t] = kp*(filt[t-k] - pos[t])*sign... (motor frame:
             angles map th = (q-off)*sign, so error transforms by sign;
             torque in motor frame = sign * joint-frame PD)
  knee:      th_des[t-k] = (filt[t-k][kn] - off + pos[t-k][hp]*c)/G*sign
             th[t]       = (pos[t][kn]   - off + pos[t][hp] *c)/G*sign
             thd[t]      = (vel[t][kn] + c*vel[t][hp])*sign/G
             tau_fw[t]   = kp*G^2*(th_des[t-k] - th[t]) - kd*G^2*thd[t]

Compare per joint:  gain/corr of  tau_meas ~ g*tau_cmd   (old)
                              vs  tau_meas ~ g*tau_fw    (new, best k)
If gain(new) ~ 1 with high corr, there is NO real torque attenuation —
the gap was the prediction, not the plant.

Run from control/:  python ../data/firmware_pd_replica.py
"""
import sys
import numpy as np

sys.path.insert(0, ".")
import config as C

G = C.KNEE_GEAR_RATIO
COUP = C.KNEE_HIP_COUPLING
SIGN = np.array(C.JOINT_SIGN, dtype=np.float64)
OFF = np.array(C.JOINT_OFFSET, dtype=np.float64)
NAMES = ["hr1", "hp1", "kn1", "hr2", "hp2", "kn2"]
HIP_OF_KNEE = {2: 1, 5: 4}

LOGS = [
    "model_4150_log_20260806_115256.npz",
    "model_4150_log_20260806_115327.npz",
    "model_4150_log_20260806_115703.npz",
    "model_4150_log_20260806_115715.npz",
    "2026-06-20_22-02-53_jrk020_grf13_log_20260806_120142.npz",
    "2026-06-20_22-02-53_jrk020_grf13_log_20260806_120222.npz",
    "2026-06-20_22-02-53_jrk020_grf13_log_20260806_120301.npz",
]

# Any CLI args replace the default corpus (paths, relative to cwd).
if len(sys.argv) > 1:
    LOGS = sys.argv[1:]


def gain_corr(pred, meas):
    if np.std(pred) < 1e-9 or np.std(meas) < 1e-9:
        return np.nan, np.nan
    g = float(np.dot(pred, meas) / np.dot(pred, pred))
    c = float(np.corrcoef(pred, meas)[0, 1])
    return g, c


pool = {j: {"old_g": [], "old_c": [], "new_g": [], "new_c": [], "k": []}
        for j in range(6)}

print(f"{'log':<40}{'j':<5}{'gOld':>6} {'cOld':>6}   {'k':>2} {'gFW':>6} {'cFW':>6}")
for npz in LOGS:
    d = np.load(npz, allow_pickle=True)
    n = len(d["time"])
    w = min(20, n // 4)
    pos = d["sim_joint_pos"].astype(np.float64)
    vel = d["sim_joint_vel"].astype(np.float64)
    filt = d["filtered_target"].astype(np.float64)
    kp = d["kp"].astype(np.float64)
    kd = d["kd"].astype(np.float64)
    tau_cmd = d["motor_tau_cmd"].astype(np.float64)
    tau_meas = d["motor_tau_meas"].astype(np.float64)

    for j in range(6):
        meas = tau_meas[w:, j]
        old_g, old_c = gain_corr(tau_cmd[w:, j], meas)

        best = None
        for k in range(0, 5):
            if j in HIP_OF_KNEE:
                hp = HIP_OF_KNEE[j]
                th_des = ((filt[:, j] - OFF[j] + pos[:, hp] * COUP)
                          / G * SIGN[j])
                th = ((pos[:, j] - OFF[j] + pos[:, hp] * COUP)
                      / G * SIGN[j])
                thd = (vel[:, j] + COUP * vel[:, hp]) * SIGN[j] / G
                kpm = kp[:, j] * G * G
                kdm = kd[:, j] * G * G
            else:
                th_des = (filt[:, j] - OFF[j]) * SIGN[j]
                th = (pos[:, j] - OFF[j]) * SIGN[j]
                thd = vel[:, j] * SIGN[j]
                kpm = kp[:, j]
                kdm = kd[:, j]
            if k == 0:
                tau_fw = kpm * (th_des - th) - kdm * thd
            else:
                tau_fw = np.empty(n)
                tau_fw[:k] = np.nan
                tau_fw[k:] = (kpm[k:] * (th_des[:-k] - th[k:])
                              - kdm[k:] * thd[k:])
            p = tau_fw[w:]
            ok = ~np.isnan(p)
            g, c = gain_corr(p[ok], meas[ok])
            if best is None or (not np.isnan(c) and c > best[2]):
                best = (k, g, c)

        print(f"{npz[:38]:<40}{NAMES[j]:<5}{old_g:>6.2f} {old_c:>6.2f}   "
              f"{best[0]:>2} {best[1]:>6.2f} {best[2]:>6.2f}")
        pool[j]["old_g"].append(old_g)
        pool[j]["old_c"].append(old_c)
        pool[j]["new_g"].append(best[1])
        pool[j]["new_c"].append(best[2])
        pool[j]["k"].append(best[0])

print("\nPooled means:")
print(f"{'joint':<6}{'gOld':>7}{'cOld':>7}{'gFW':>7}{'cFW':>7}{'k*':>5}")
for j in range(6):
    p = pool[j]
    print(f"{NAMES[j]:<6}{np.mean(p['old_g']):>7.2f}{np.mean(p['old_c']):>7.2f}"
          f"{np.mean(p['new_g']):>7.2f}{np.mean(p['new_c']):>7.2f}"
          f"{np.median(p['k']):>5.0f}")
