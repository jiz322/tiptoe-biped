"""coupling_torque_analysis.py — test the hypothesis that the knee torque
"attenuation" (gain 0.15-0.27 in tau_meas vs tau_cmd) is caused by the
hip-knee belt coupling, i.e. by comparing against the WRONG prediction.

Real kinematics (motor_bridge.py, sim-oriented motor angle th = motor*sign):
    q_kn = G*th_kn - c*q_hp + off          G = 22/30,  c = 0.733
so the knee MOTOR state is
    th_kn  = (q_kn - off + c*q_hp) * sign / G
    thd_kn = (qd_kn + c*qd_hp)     * sign / G
and the firmware PD (MIT mode, kp_m = kp*G^2, kd_m = kd*G^2, v_des=0) gives
    tau_motor = kp_m*(th_des - th_kn) - kd_m*thd_kn
              = motor_tau_cmd  -  sign*G*kd*c*qd_hp        (exact identity)
                                  ^^^^^^^^^^^^^^^^^^^ cross-damping term:
                                  knee motor Kd fights the decoupling spin
                                  forced by hip pitch motion. Absent in sim
                                  and absent from the logged tau_cmd.

Regression per knee, per log, best lag k in 0..6:
  A) tau_meas ~ gA * tau_cmd                (previous analysis, joint PD law)
  B) tau_meas ~ gB * tau_full               (full motor-frame PD prediction)
  C) tau_meas ~ a*tau_cmd + b*(G*c*qd_hp)   (free 2-regressor decomposition;
                                             firmware-exact => a=1, b=-sign)

Also reports, per log:
  - RMS of the cross term vs RMS of tau_cmd (how big is the cancellation)
  - corr(cross term, tau_cmd)  (correlated => biases gA downward)
  - hip reaction: the knee motor torque reacts on the hip JOINT with
    Q_hp = c * tau_kn_joint (virtual work) — RMS vs hip tau_cmd RMS.

Run from control/:  python ../data/coupling_torque_analysis.py
"""
import sys
import numpy as np

sys.path.insert(0, ".")
import config as C

G = C.KNEE_GEAR_RATIO
COUP = C.KNEE_HIP_COUPLING
SIGN = np.array(C.JOINT_SIGN, dtype=np.float64)
KNEES = [(2, 1, "kn1"), (5, 4, "kn2")]   # (knee idx, its hip pitch idx, name)

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
# No-arg invocation is unchanged — make_figs.py screen-scrapes it.
import sys as _sys
if len(_sys.argv) > 1:
    LOGS = _sys.argv[1:]


def best_lag_fit(pred, meas, max_lag=6):
    """meas[t] ~ g * pred[t-k]; return (k, g, corr, rmse) at best |corr|."""
    best = None
    for k in range(max_lag + 1):
        p = pred[:len(pred) - k] if k else pred
        m = meas[k:]
        if np.std(p) < 1e-9 or np.std(m) < 1e-9:
            continue
        c = float(np.corrcoef(p, m)[0, 1])
        if best is None or abs(c) > abs(best[2]):
            g = float(np.dot(p, m) / np.dot(p, p))
            r = float(np.sqrt(np.mean((m - g * p) ** 2)))
            best = (k, g, c, r)
    return best


def two_reg_fit(x1, x2, meas, k):
    """meas[t] ~ a*x1[t-k] + b*x2[t-k], OLS."""
    n = len(meas) - k
    X = np.stack([x1[:n], x2[:n]], axis=1)
    y = meas[k:]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    r = float(np.sqrt(np.mean(resid ** 2)))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / max(ss_tot, 1e-12)
    return coef[0], coef[1], r, r2


print(f"{'log':<44}{'j':<5}{'k':>2} {'gA(cmd)':>8} {'rA':>6}  "
      f"{'gB(full)':>8} {'rB':>6}   {'a(cmd)':>7} {'b(cross)':>8} {'R2c':>6}  "
      f"{'rmsX/rmsC':>9} {'corrXC':>7}")
pool = {name: {"A": [], "B": [], "ratio": [], "hipfrac": []} for _, _, name in KNEES}
for npz in LOGS:
    d = np.load(npz, allow_pickle=True)
    n = len(d["time"])
    w = min(20, n // 4)
    pos = d["sim_joint_pos"].astype(np.float64)
    vel = d["sim_joint_vel"].astype(np.float64)
    kd = d["kd"].astype(np.float64)
    tau_cmd = d["motor_tau_cmd"].astype(np.float64)
    tau_meas = d["motor_tau_meas"].astype(np.float64)

    for kn, hp, name in KNEES:
        # exact cross-damping term (motor frame): -sign*G*kd*c*qd_hp
        cross = -SIGN[kn] * G * kd[:, kn] * COUP * vel[:, hp]
        tau_full = tau_cmd[:, kn] + cross

        cmd = tau_cmd[w:, kn]
        full = tau_full[w:]
        meas = tau_meas[w:, kn]
        xr = cross[w:]

        A = best_lag_fit(cmd, meas)
        B = best_lag_fit(full, meas)
        a, b, rc, r2c = two_reg_fit(cmd, G * COUP * vel[w:, hp], meas, A[0])
        rms_ratio = float(np.sqrt(np.mean(xr ** 2)) /
                          max(np.sqrt(np.mean(cmd ** 2)), 1e-9))
        cxc = float(np.corrcoef(xr, cmd)[0, 1])

        # hip reaction: knee joint torque reacts on hip joint with c*tau_kn_joint
        tau_kn_joint = tau_meas[w:, kn] * SIGN[kn] / G
        hip_react_rms = float(np.sqrt(np.mean((COUP * tau_kn_joint) ** 2)))
        hip_cmd_rms = float(np.sqrt(np.mean((tau_cmd[w:, hp]) ** 2)))

        print(f"{npz[:42]:<44}{name:<5}{A[0]:>2} {A[1]:>8.3f} {A[2]:>6.2f}  "
              f"{B[1]:>8.3f} {B[2]:>6.2f}   {a:>7.3f} {b:>8.3f} {r2c:>6.2f}  "
              f"{rms_ratio:>9.2f} {cxc:>7.2f}")
        pool[name]["A"].append(A[1])
        pool[name]["B"].append(B[1])
        pool[name]["ratio"].append(rms_ratio)
        pool[name]["hipfrac"].append(hip_react_rms / max(hip_cmd_rms, 1e-9))

print("\nPooled (mean over logs):")
for _, _, name in KNEES:
    p = pool[name]
    print(f"  {name}: gain(cmd-only)={np.mean(p['A']):.3f}  "
          f"gain(full motor PD)={np.mean(p['B']):.3f}  "
          f"cross/cmd RMS={np.mean(p['ratio']):.2f}  "
          f"hip-reaction/hip-cmd RMS={np.mean(p['hipfrac']):.2f}")
