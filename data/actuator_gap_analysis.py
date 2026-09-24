"""
actuator_gap_analysis.py — documents/reproduces the "how did we get 16-42%
torque delivery, 2-3 tick lag" numbers quoted in the sim2real actuator-model
report (built during the Aug 5-6 investigation, ad hoc in a Bash shell at the
time; this is the script version of that analysis).

Pure numpy, no Isaac. Needs:
  - the balance_log.py NPZ (real hardware)
  - the sysid_replay_policy_log.py free-run output NPZ (sim replay, any base
    mode/delay — only used for the command-chain-replica check, which needs
    the live cfg's omega/zeta/scales, not the sim trajectories themselves)

Two independent analyses:

  A. TORQUE ATTENUATION (tau_meas ~= g * tau_cmd, best-fit lag)
     Question: does the motor deliver the commanded PD torque? tau_cmd here
     is balance_log.py's own logged motor_tau_cmd (computed hardware-side
     from the LPF'd target and the recovered gains — NOT the sim's target).
     For each joint and each candidate lag k in [0..max_lag], regress
     tau_meas[t] ~ g * tau_cmd[t-k] (through the origin) and keep the (k, g)
     with the highest correlation r. A g << 1 at the best-correlated lag
     means the motor is NOT delivering the commanded torque even after
     accounting for transport delay — i.e. an attenuation the current
     actuator model (DCMotor speed-torque envelope) does not capture (the
     envelope predicts saturation only well outside this run's torque
     regime, see balance_forensics.py check 6).

  B. COMMAND-CHAIN REPLICA (computed TARGET mismatch, sim vs hardware)
     Question: independent of tracking (PD response), do the two systems
     even compute the same desired position from the same action log?
     Replicates the sim's chain purely in numpy (clamp -> LPF(alpha) ->
     delay -> default+scale, matching rl_env_p6.py; the 2nd-order prefilter
     is the ODE stage and is included here as a discrete-time simulation of
     the same equation used in sysid_replay_policy_log.py) and compares
     against the hardware's own filtered target (recovered by
     balance_forensics.reconstruct_filtered_target, LPF + effort-clip only,
     no prefilter — hardware has no prefilter stage). This RMSE is a
     command-construction mismatch, separate from and additional to any
     PD-tracking mismatch measured elsewhere.

Usage:
    python data/actuator_gap_analysis.py \\
        --npz control/model_4150_log_20260803_140243.npz \\
        --max-lag 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "control"))
import config as C  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from balance_forensics import (  # noqa: E402
    recover_gains_and_alpha,
    reconstruct_filtered_target,
    motor_to_joint_torque,
    JOINT_NAMES,
    BURST_EFFORT,
    FALLBACK_LPF_ALPHA,
)

CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "RL", "source",
                         "RL", "RL", "tasks", "direct", "standup_curriculum", "rl_env_cfg_p6.py")


def _read_cfg_constants(path):
    """Extract the handful of scalar defaults this script needs directly
    from the cfg SOURCE TEXT, instead of importing RlEnvCfg. Importing the
    real class pulls in Isaac Sim (`pxr`, only available inside a running
    AppLauncher, see isaaclab-conda-env memory) — this script is meant to
    run with plain numpy, so it reads the same numbers the way a human
    reviewing a diff would, rather than hardcoding a second copy."""
    import re
    src = open(path, "r", encoding="utf-8").read()

    def grab(name):
        m = re.search(rf"^\s*{name}\s*:\s*\w+\s*=\s*([0-9.eE+-]+)", src, re.MULTILINE)
        if not m:
            raise RuntimeError(f"could not find cfg constant {name!r} in {path}")
        return float(m.group(1))

    return {
        "action_lpf_alpha": grab("action_lpf_alpha"),
        "act_lag_omega_hip_roll": grab("act_lag_omega_hip_roll"),
        "act_lag_omega_hip_pitch": grab("act_lag_omega_hip_pitch"),
        "act_lag_omega_knee": grab("act_lag_omega_knee"),
        "act_lag_zeta_hip_roll": grab("act_lag_zeta_hip_roll"),
        "act_lag_zeta_hip_pitch": grab("act_lag_zeta_hip_pitch"),
        "act_lag_zeta_knee": grab("act_lag_zeta_knee"),
    }

DEFAULT_POS = np.array(C.DEFAULT_JOINT_POS, dtype=np.float64)
ACTION_SCALES = np.array(C.ACTION_SCALES, dtype=np.float64)
GROUPS = ["hip_roll", "hip_pitch", "knee", "hip_roll", "hip_pitch", "knee"]


def fit_gain_with_lag(tau_cmd, tau_meas, max_lag):
    """Per-joint: for each lag k, fit tau_meas[t] = g*tau_cmd[t-k] (origin-
    through regression) and correlation r; return the lag with the highest r.

    Returns dict joint -> list of (lag, gain, r) for all lags, plus best.
    """
    n, nj = tau_cmd.shape
    out = []
    for j in range(nj):
        rows = []
        for k in range(max_lag + 1):
            a = tau_cmd[: n - k, j]
            b = tau_meas[k:, j]
            denom = a.std() * b.std()
            r = float(np.mean((a - a.mean()) * (b - b.mean())) / denom) if denom > 1e-12 else 0.0
            g = float(np.dot(a, b) / max(np.dot(a, a), 1e-12))
            rows.append((k, g, r))
        best = max(rows, key=lambda row: row[2])
        out.append({"lag_scan": rows, "best_lag": best[0], "best_gain": best[1], "best_r": best[2]})
    return out


def command_chain_replica(action, alpha, delay, omega, zeta, dt):
    """Pure-numpy replica of rl_env_p6.py's action chain: clamp -> LPF(alpha)
    on the normalized action -> delay ring buffer -> target = default + a*scale
    -> 2nd-order lag prefilter (forward Euler), per joint group.

    action: (n, 6) post-clamp actions as logged (already in [-1, 1]).
    Returns pf_y: (n, 6) prefiltered target trajectory, in sim joint coords.
    """
    n, nj = action.shape
    a = np.clip(action, -1.0, 1.0)

    # Action LPF (state init at a[0], matching zero-initial-condition warmup
    # used elsewhere in this investigation — the LPF is on the NORMALIZED
    # action here, per rl_env_p6.py, not on the scaled target as in hardware).
    lpf = np.zeros_like(a)
    state = a[0].copy()
    for t in range(n):
        state = alpha * state + (1.0 - alpha) * a[t]
        lpf[t] = state

    # Transport delay: shift by `delay` ticks, holding a[0] for t < delay.
    delayed = np.zeros_like(lpf)
    if delay > 0:
        delayed[delay:] = lpf[:-delay]
        delayed[:delay] = lpf[0]
    else:
        delayed = lpf

    target = DEFAULT_POS + delayed * ACTION_SCALES

    # 2nd-order lag prefilter: y'' = omega^2*(u - y) - 2*zeta*omega*y'
    pf_y = np.zeros_like(target)
    pf_yd = np.zeros(nj)
    y = target[0].copy()
    for t in range(n):
        u = target[t]
        ydd = omega ** 2 * (u - y) - 2.0 * zeta * omega * pf_yd
        pf_yd = pf_yd + ydd * dt
        y = y + pf_yd * dt
        pf_y[t] = y
    return pf_y


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--npz", required=True, help="balance_log.py output NPZ")
    ap.add_argument("--max-lag", type=int, default=6, help="max lag (ticks) to scan")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--delays", type=str, default="0,1,2",
                     help="comma list of action-delay ticks to try for part B")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    n = len(d["time"])
    t_arr = d["time"].astype(np.float64)
    action = d["action"].astype(np.float64)
    target = d["target"].astype(np.float64)
    pos = d["sim_joint_pos"].astype(np.float64)
    vel = d["sim_joint_vel"].astype(np.float64)
    tau_cmd_motor = d["motor_tau_cmd"].astype(np.float64)
    tau_meas_motor = d["motor_tau_meas"].astype(np.float64)
    w = min(args.warmup, n // 4)
    dt = float(np.median(np.diff(t_arr)))

    tau_cmd_joint = motor_to_joint_torque(tau_cmd_motor)
    tau_meas_joint = motor_to_joint_torque(tau_meas_motor)

    if "kp" in d.files:
        kp_arr, kd_arr = d["kp"].astype(np.float64), d["kd"].astype(np.float64)
    else:
        kp_rec, kd_rec, a_rec, _ = recover_gains_and_alpha(target, pos, vel, tau_cmd_joint, warmup=w)
        kp_arr, kd_arr = np.tile(kp_rec, (n, 1)), np.tile(kd_rec, (n, 1))
    alpha_hw = float(a_rec[0]) if "kp" not in d.files and np.allclose(a_rec, a_rec[0]) else FALLBACK_LPF_ALPHA

    print(f"\n=== actuator_gap_analysis: {os.path.basename(args.npz)} ===")
    print(f"  ticks={n}  dt~{dt*1e3:.2f}ms  warmup skipped={w}")

    # ── A. Torque attenuation: tau_meas ~= g * tau_cmd, best lag ─────────
    print("\n-- A. Torque attenuation (tau_meas ~ g * tau_cmd(t-lag), motor frame) --")
    fits = fit_gain_with_lag(tau_cmd_motor[w:], tau_meas_motor[w:], args.max_lag)
    for i, f in enumerate(fits):
        print(f"  {JOINT_NAMES[i]}: best lag={f['best_lag']}  gain={f['best_gain']:.3f}  "
              f"r={f['best_r']:.3f}   (lag=0: gain={f['lag_scan'][0][1]:.3f} r={f['lag_scan'][0][2]:.3f})")
    gains = [f["best_gain"] for f in fits]
    print(f"  => gain range across joints at best lag: {min(gains):.3f}-{max(gains):.3f} "
          f"({100*min(gains):.0f}%-{100*max(gains):.0f}% of PD-law-implied torque delivered)")

    # ── B. Command-chain replica: computed TARGET mismatch ──────────────
    print("\n-- B. Command-chain replica: sim-chain target vs hardware filtered target --")
    cfg = _read_cfg_constants(CFG_PATH)
    omega = np.array([{"hip_roll": cfg["act_lag_omega_hip_roll"],
                        "hip_pitch": cfg["act_lag_omega_hip_pitch"],
                        "knee": cfg["act_lag_omega_knee"]}[g] for g in GROUPS])
    zeta = np.array([{"hip_roll": cfg["act_lag_zeta_hip_roll"],
                       "hip_pitch": cfg["act_lag_zeta_hip_pitch"],
                       "knee": cfg["act_lag_zeta_knee"]}[g] for g in GROUPS])
    alpha_sim = float(cfg["action_lpf_alpha"])

    # Hardware's own filtered target (LPF + effort-clip, no prefilter stage
    # — the real MotorBridge has none): same reconstruction balance_forensics
    # uses, seeded the same way (invert tick-0's tau_cmd for pre-run state).
    filt0 = tau_cmd_joint[0] / kp_arr[0] + kd_arr[0] * vel[0] / kp_arr[0] + pos[0]
    init_state = (filt0 - (1.0 - alpha_hw) * target[0]) / alpha_hw if alpha_hw > 0 else None
    hw_filtered, _ = reconstruct_filtered_target(
        target, pos, kp_arr, np.full(n, alpha_hw), BURST_EFFORT, init_state=init_state)

    for delay_str in args.delays.split(","):
        delay = int(delay_str)
        sim_target = command_chain_replica(action, alpha_sim, delay, omega, zeta, dt)
        err = sim_target[w:] - hw_filtered[w:]
        rmse = np.sqrt((err ** 2).mean(axis=0))
        print(f"  delay={delay}: per-joint target RMSE (rad) = "
              + ", ".join(f"{JOINT_NAMES[i]}={rmse[i]:.4f}" for i in range(6))
              + f"   mean={rmse.mean():.4f}")

    print()


if __name__ == "__main__":
    main()
