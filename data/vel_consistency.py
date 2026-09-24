#!/usr/bin/env python3
"""
vel_consistency.py — Is the SIM or the REAL velocity channel wrong?

Runs on a *_replay.npz produced by sysid_replay_sim.py. No IsaacLab needed,
just numpy + matplotlib, so it runs on your plotting box.

WHY THIS WORKS
--------------
Position is the time-integral of velocity, so for ANY self-consistent
(pos, vel) recording:   d/dt pos  ==  vel   (and  pos = pos0 + integral(vel)).
This is a property of the data, not of the physics.

You tuned the prefilter so sim_pos ~= real_pos. If the two POSITION traces
overlap but the two VELOCITY traces differ by a ~constant factor, then at
most ONE velocity channel can be consistent with its own position. The
inconsistent one is the bug. (PhysX joint_pos and joint_vel are the same
state vector integrated together, so the sim side is essentially guaranteed
consistent — which is what makes the real side the suspect.)

For each trial, on the excited joint, it does two independent checks:
  (1) integrate vel -> compare to recorded pos        <- your proposed test
  (2) differentiate pos -> compare to recorded vel, and linear-fit the slope
      k in  recorded_vel ~= k * d(pos)/dt + b.  k~=1 is consistent; k far
      from 1 both LOCALIZES and SIZES the error (e.g. k~=6 on the real side
      => real velocity is ~6x too big -> motor-side/gear or wrong-dt finite
      difference in the logger).
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def trapz_cumulative(y, t):
    """Cumulative trapezoidal integral of 1-D y over 1-D t, starting at 0."""
    out = np.zeros_like(y, dtype=float)
    dt = np.diff(t)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dt)
    return out


def central_diff(y, t):
    """Central finite difference dy/dt on 1-D arrays (one-sided at the ends)."""
    v = np.empty_like(y, dtype=float)
    v[1:-1] = (y[2:] - y[:-2]) / (t[2:] - t[:-2])
    v[0]    = (y[1]  - y[0])   / (t[1]  - t[0])
    v[-1]   = (y[-1] - y[-2])  / (t[-1] - t[-2])
    return v


def scale_fit(v_recorded, v_from_pos, frac=0.1):
    """Slope k of  v_recorded ~= k * v_from_pos + b, fit on the high-motion
    samples only (|v_from_pos| > frac * max) so the noise floor at the
    reversals doesn't dominate. k~=1 => recorded velocity agrees with the
    position trace; k far from 1 quantifies the scale/unit error."""
    thr = frac * np.nanmax(np.abs(v_from_pos))
    m = np.abs(v_from_pos) > max(thr, 1e-9)
    if m.sum() < 5:
        m = np.ones_like(v_from_pos, dtype=bool)
    k, b = np.polyfit(v_from_pos[m], v_recorded[m], 1)
    return float(k), float(b)


def load_replay(path):
    d = np.load(path, allow_pickle=True)
    names = sorted({k.rsplit("__", 1)[0] for k in d.keys() if "__" in k})
    trials = {}
    for nm in names:
        if f"{nm}__sim_vel" not in d:
            continue
        trials[nm] = {
            "t":        np.asarray(d[f"{nm}__t"], dtype=float),
            "sim_pos":  np.asarray(d[f"{nm}__sim_pos"], dtype=float),
            "sim_vel":  np.asarray(d[f"{nm}__sim_vel"], dtype=float),
            "real_pos": np.asarray(d[f"{nm}__real_pos"], dtype=float),
            "real_vel": np.asarray(d[f"{nm}__real_vel"], dtype=float),
        }
    return trials


def excited_joint(real_pos):
    """Joint with the largest peak-to-peak motion = the one being excited."""
    return int(np.argmax(real_pos.max(0) - real_pos.min(0)))


def verdict_for(k_real, k_sim, tol=0.30):
    """tol in log-space: |ln k| < ln(1+tol) counts as consistent."""
    thr = np.log(1.0 + tol)
    lr, ls = abs(np.log(abs(k_real) + 1e-9)), abs(np.log(abs(k_sim) + 1e-9))
    if lr < thr and ls < thr:
        return "both consistent"
    if lr >= ls:
        return f"REAL vel off ~{k_real:.1f}x"
    return f"SIM vel off ~{k_sim:.1f}x"


def analyze(path):
    trials = load_replay(path)
    if not trials:
        print(f"No usable trials found in {path}")
        return

    print(f"{'trial':<30s} {'jt':>3s} {'k_real':>8s} {'k_sim':>8s}   verdict")
    print("-" * 74)
    rows = []
    for nm, tr in trials.items():
        t = tr["t"]
        j = excited_joint(tr["real_pos"])
        rp, rv = tr["real_pos"][:, j], tr["real_vel"][:, j]
        sp, sv = tr["sim_pos"][:, j],  tr["sim_vel"][:, j]
        k_real, _ = scale_fit(rv, central_diff(rp, t))
        k_sim,  _ = scale_fit(sv, central_diff(sp, t))
        v = verdict_for(k_real, k_sim)
        print(f"{nm:<30s} {j:>3d} {k_real:8.2f} {k_sim:8.2f}   {v}")
        rows.append((nm, j, t, rp, rv, sp, sv, k_real, k_sim))

    ks_real = np.array([r[7] for r in rows])
    ks_sim  = np.array([r[8] for r in rows])
    print("-" * 74)
    print(f"median k_real = {np.median(ks_real):.2f}    "
          f"median k_sim = {np.median(ks_sim):.2f}")
    print("k≈1 is the consistent (correct) channel; the other is the bug.\n")

    for (nm, j, t, rp, rv, sp, sv, k_real, k_sim) in rows:
        fig, ax = plt.subplots(2, 2, figsize=(11, 7))
        fig.suptitle(f"{nm}   joint {j}", fontsize=12)
        for row, (pos, vel, k, who) in enumerate(
                [(rp, rv, k_real, "real"), (sp, sv, k_sim, "sim")]):
            ip = pos[0] + trapz_cumulative(vel, t)        # your test
            ax[row, 0].plot(t, pos, "k-",  lw=1.0, label=f"{who}_pos (recorded)")
            ax[row, 0].plot(t, ip,  "r--", lw=1.0, label=f"integral({who}_vel)")
            ax[row, 0].set_title(f"{who}:  integral(vel) vs pos", fontsize=9)
            ax[row, 0].set_ylabel("rad"); ax[row, 0].legend(fontsize=7)
            vp = central_diff(pos, t)                     # the quantitative kicker
            ax[row, 1].plot(t, vp,  "k-",  lw=1.0, label=f"d/dt {who}_pos")
            ax[row, 1].plot(t, vel, "r--", lw=1.0, label=f"{who}_vel (recorded)")
            ax[row, 1].set_title(
                f"{who}:  vel vs d/dt pos   (vel ≈ {k:.2f}·d/dt pos)", fontsize=9)
            ax[row, 1].set_ylabel("rad/s"); ax[row, 1].legend(fontsize=7)
        for a in ax.ravel():
            a.set_xlabel("t [s]"); a.tick_params(labelsize=7)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = f"velcheck_{nm}.pdf"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  -> {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python vel_consistency.py <something>_replay.npz")
        sys.exit(1)
    analyze(sys.argv[1])
