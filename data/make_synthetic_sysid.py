"""
make_synthetic_sysid.py — fabricate a .npz in the sysid_experiment.py schema
so sysid_diagnostics.py can be smoke-tested without hardware.

Each excited joint is simulated as a closed loop (host PD + transport delay +
2nd-order actuator prefilter) around a rigid plant with KNOWN inertia, viscous,
Coulomb, and gravity. Because the ground truth is known, running the diagnostics
on this file is also a correctness check: the triangle regression should recover
roughly the J / b / tau_c / gravity that were planted here.

    python make_synthetic_sysid.py            # writes sysid_leg1_synthetic.npz
"""
from __future__ import annotations
import json
import math
import numpy as np

CONTROL_HZ = 100
DT = 1.0 / CONTROL_HZ
DEFAULT_POS = np.array([0.0, -0.3, 0.6, 0.0, -0.3, 0.6], dtype=np.float32)

JTYPE = ["hip_roll", "hip_pitch", "knee",
         "hip_roll", "hip_pitch", "knee"]
EXCITE = {"hip_roll": (6.0, 0.3), "hip_pitch": (8.0, 0.5), "knee": (8.0, 0.5)}
HOLD = {"hip_roll": (15.0, 0.8), "hip_pitch": (20.0, 1.2), "knee": (20.0, 1.2)}
EFFORT_LIM = np.array([6.0, 17.0, 23.2, 6.0, 17.0, 23.2], dtype=np.float32)

# GROUND TRUTH planted per joint type (the diagnostics should recover these)
TRUTH = {
    "hip_roll":  dict(J=0.020, b=0.030, tau_c=0.10, g_amp=0.4),
    "hip_pitch": dict(J=0.060, b=0.040, tau_c=0.12, g_amp=2.0),
    "knee":      dict(J=0.045, b=0.035, tau_c=0.06, g_amp=1.5),
}
# actuator extras the prefilter is meant to model
ACT_OMEGA = {"hip_roll": 13.0, "hip_pitch": 14.0, "knee": 16.0}
ACT_ZETA = {"hip_roll": 0.30, "hip_pitch": 0.55, "knee": 0.68}
DELAY_STEPS = 2  # transport delay in control ticks


def sig_step(t, amp, **_):
    return amp


def sig_chirp(t, amp, f0, f1, dur, **_):
    phase = 2 * math.pi * (f0 * t + (f1 - f0) * t * t / (2 * dur))
    return amp * math.sin(phase)


def sig_triangle(t, vel, lim, **_):
    period = 4 * lim / abs(vel)
    p = (t % period) / period
    if p < 0.25:
        return 4 * lim * p
    elif p < 0.75:
        return lim - 4 * lim * (p - 0.25)
    return -lim + 4 * lim * (p - 0.75)


GEN = {"step": sig_step, "chirp": sig_chirp, "triangle": sig_triangle}


def build_trials(leg=1):
    off = 0 if leg == 1 else 3
    hr, hp, kn = off, 1 + off, 2 + off
    T = []
    for a in [0.2, -0.2, 0.4, -0.4]:
        T.append(dict(name=f"hr_step_{a:+.1f}", idx=hr, signal="step", amp=a, dur=2.5))
    T.append(dict(name="hr_chirp", idx=hr, signal="chirp", amp=0.2, f0=0.5, f1=12.0, dur=15.0))
    for v in [0.5, 2.0]:
        T.append(dict(name=f"hr_tri_{v:.0f}", idx=hr, signal="triangle", vel=v, lim=0.45, dur=4.0))
    for a in [0.3, -0.3, 0.6, -0.6]:
        T.append(dict(name=f"hp_step_{a:+.1f}", idx=hp, signal="step", amp=a, dur=2.5))
    T.append(dict(name="hp_chirp", idx=hp, signal="chirp", amp=0.2, f0=0.5, f1=12.0, dur=15.0))
    for v in [0.5, 2.0]:
        T.append(dict(name=f"hp_tri_{v:.0f}", idx=hp, signal="triangle", vel=v, lim=0.8, dur=4.0))
    for a in [0.3, -0.3, 0.6, -0.6]:
        T.append(dict(name=f"kn_stiff_step_{a:+.1f}", idx=kn, signal="step", amp=a, dur=2.5))
    T.append(dict(name="kn_stiff_chirp", idx=kn, signal="chirp", amp=0.2, f0=0.5, f1=12.0, dur=15.0))
    for v in [0.5, 2.0]:
        T.append(dict(name=f"kn_stiff_tri_{v:.0f}", idx=kn, signal="triangle", vel=v, lim=0.8, dur=4.0))
    return T


def simulate(trial, rng):
    """Closed-loop sim of the excited joint; others sit at default + noise."""
    idx = int(trial["idx"])
    jt = JTYPE[idx]
    tr = TRUTH[jt]
    kp, kd = EXCITE[jt]
    J, b, tau_c, g_amp = tr["J"], tr["b"], tr["tau_c"], tr["g_amp"]
    w, z = ACT_OMEGA[jt], ACT_ZETA[jt]

    dur = trial["dur"]
    n = int(dur * CONTROL_HZ)
    # finer integration substeps for the plant
    sub = 5
    h = DT / sub

    th = float(DEFAULT_POS[idx])
    thd = 0.0
    pf_y, pf_yd = th, 0.0          # 2nd-order prefilter state
    cmd_hist = [th] * (DELAY_STEPS + 1)

    t_arr = np.zeros(n)
    p_des_full = np.tile(DEFAULT_POS.astype(float), (n, 1))
    pos_full = np.tile(DEFAULT_POS.astype(float), (n, 1))
    vel_full = np.zeros((n, 6))
    tau_full = np.zeros((n, 6))

    for k in range(n):
        t = k * DT
        raw = float(np.clip(DEFAULT_POS[idx] + GEN[trial["signal"]](t, **trial),
                            DEFAULT_POS[idx] - 1.5, DEFAULT_POS[idx] + 1.5))
        cmd_hist.append(raw)
        delayed = cmd_hist[-1 - DELAY_STEPS]
        # 2nd-order prefilter on the (delayed) command
        ydd = w * w * (delayed - pf_y) - 2 * z * w * pf_yd
        pf_yd += ydd * DT
        pf_y += pf_yd * DT
        target = pf_y

        tau_cmd = 0.0
        for _ in range(sub):
            tau_cmd = kp * (target - th) + kd * (0.0 - thd)
            tau_cmd = float(np.clip(tau_cmd, -EFFORT_LIM[idx], EFFORT_LIM[idx]))
            grav = g_amp * math.sin(th)
            fric = b * thd + tau_c * math.tanh(thd / 0.02)  # smoothed Coulomb
            thdd = (tau_cmd - grav - fric) / J
            thd += thdd * h
            th += thd * h

        t_arr[k] = t
        p_des_full[k, idx] = raw
        pos_full[k, idx] = th + rng.normal(0, 2e-4)
        vel_full[k, idx] = thd + rng.normal(0, 1e-3)
        # recorded torque = applied motor torque (what hardware reports)
        tau_full[k, idx] = tau_cmd + rng.normal(0, 5e-3)

    return dict(time=t_arr, p_des=p_des_full, pos=pos_full,
                vel=vel_full, torque=tau_full,
                temp=np.full((n, 6), 30.0),
                kp=np.full(6, kp, dtype=np.float32),
                kd=np.full(6, kd, dtype=np.float32))


def main():
    rng = np.random.default_rng(0)
    trials = build_trials(leg=1)
    arrays, names, configs = {}, [], []
    for tr in trials:
        rec = simulate(tr, rng)
        nm = tr["name"]
        names.append(nm)
        configs.append({k: v for k, v in tr.items()})
        for ch in ("time", "p_des", "pos", "vel", "torque", "temp", "kp", "kd"):
            arrays[f"{nm}__{ch}"] = rec[ch]
    arrays["trial_names"] = np.array(names)
    arrays["trial_configs"] = np.array(json.dumps(configs))
    arrays["control_hz"] = np.array(CONTROL_HZ)
    arrays["default_pos"] = DEFAULT_POS
    arrays["knee_hip_coupling"] = np.array(0.0)
    arrays["knee_gear_ratio"] = np.array(30.0 / 22.0)
    fn = "sysid_leg1_synthetic.npz"
    np.savez(fn, **arrays)
    print(f"wrote {fn}")
    print("planted ground truth (the triangle regression should recover ~these):")
    for jt, tr in TRUTH.items():
        print(f"  {jt:10s} J={tr['J']:.3f}  b={tr['b']:.3f}  "
              f"tau_c={tr['tau_c']:.3f}  g_amp={tr['g_amp']:.3f}")


if __name__ == "__main__":
    main()
