"""
sysid_diagnostics.py — Pull physics out of the sysid data you ALREADY have.

No new hardware runs required. Operates directly on the .npz written by
sysid_experiment.py. Optionally overlays the sim from *_replay.npz.

The point of this tool is to attack the two risks in the replay fit:
    (1) the 2nd-order command prefilter may be standing in for the WRONG
        physics (a transport delay, a friction/inertia error, or belt
        compliance all look near-identical in one step);
    (2) the whole identification was done hanging, so the mass model is
        unvalidated even in the air.

What it extracts, and from which existing trial:

  A. SATURATION REPORT (every trial)
        Are any trials clipping the effort limit or flat-topping torque?
        If so, the linear/lag premise is already broken for those trials and
        no amount of omega/zeta tuning is meaningful there.

  B. QUASI-RIGID REGRESSION on the triangle  (the headline)
        Fit  tau = J*thddot + b*thd + tau_c*sign(thd) + g0 + g1*sin(th) + g2*cos(th)
        by linear least squares over the slow + fast triangle pooled. This
        returns FOUR things INDEPENDENT of the step-shape fit:
            * gravity curve g(th)   -> validates the URDF mass/COM (the
                                       prerequisite for the loaded case)
            * tau_c (Coulomb)       -> compare to FRICTION_TUNE static/dynamic
            * b (viscous)           -> compare to viscous_friction
            * J (effective inertia) -> compare to the armature you tune
        If the friction/inertia the triangle reports disagree with the values
        you fit to match the STEP, that disagreement IS the "wrong cause"
        smell made quantitative.

  C. AMPLITUDE-NORMALIZED STEPS
        Overlay each step divided by its commanded amplitude. A linear lag
        gives curves that lie on top of each other. If small steps lag more
        (stiction) or large steps flatten (saturation), the lag is nonlinear
        and the single 2nd-order block is hiding it.

  D. CHIRP FREQUENCY RESPONSE (FRF + coherence)
        Empirical transfer function p_des -> pos. The Bode SHAPE says which
        model you need: pure delay = flat magnitude + linear phase; true
        2nd order = a defined roll-off through -90 deg; compliance = a
        resonant peak. Coherence < ~0.8 flags friction/saturation (non-LTI)
        directly. With a replay file it overlays the SIM FRF so you can see
        at WHICH frequency sim and real diverge.
        CAVEAT: your chirp is only 0.2 rad, so at low frequency the joint
        barely leaves the stick regime and coherence will be poor there.
        Treat low-coherence bands as "LTI doesn't hold," not as signal.

Usage:
    python sysid_diagnostics.py sysid_leg1_*.npz
    python sysid_diagnostics.py sysid_leg1_*.npz --replay sysid_leg1_*_replay.npz
    python sysid_diagnostics.py sysid_leg1_*.npz --effort 6 17 23.2 6 17 23.2
"""
from __future__ import annotations
import argparse
import json
import sys

import numpy as np

try:
    from scipy.signal import savgol_filter, coherence, csd, welch
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - we install scipy, this is a guard
    _HAVE_SCIPY = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


JTYPE = ["hip_roll", "hip_pitch", "knee",
         "hip_roll", "hip_pitch", "knee"]

# Mirror sysid_experiment.py defaults so the tool is self-contained.
DEFAULT_EFFORT_LIM = np.array([6.0, 17.0, 23.2, 6.0, 17.0, 23.2], dtype=float)


# ══════════════════════════════════════════════════════════════════
# Loading
# ══════════════════════════════════════════════════════════════════
def load_experiment(path):
    """Read the raw hardware .npz into a dict of trials."""
    d = np.load(path, allow_pickle=True)
    names = list(d["trial_names"])
    configs = json.loads(str(d["trial_configs"]))
    dpos = np.asarray(d["default_pos"], dtype=float)
    fs = float(d["control_hz"])

    trials = {}
    for nm, cfg in zip(names, configs):
        nm = str(nm)
        rec = dict(
            t=np.asarray(d[f"{nm}__time"], dtype=float),
            p_des=np.asarray(d[f"{nm}__p_des"], dtype=float),
            pos=np.asarray(d[f"{nm}__pos"], dtype=float),
            vel=np.asarray(d[f"{nm}__vel"], dtype=float),
            cfg=cfg,
        )
        # torque is written by sysid_experiment.py but is optional here.
        tk = f"{nm}__torque"
        rec["torque"] = np.asarray(d[tk], dtype=float) if tk in d.files else None
        trials[nm] = rec
    return trials, dpos, fs


def load_replay(path):
    """Read sim pos/vel from *_replay.npz (written by sysid_replay_sim.py)."""
    if path is None:
        return None
    d = np.load(path, allow_pickle=True)
    sim = {}
    for k in d.files:
        if k.endswith("__sim_pos") or k.endswith("__sim_vel"):
            sim[k] = np.asarray(d[k], dtype=float)
    return sim


# ══════════════════════════════════════════════════════════════════
# A. Saturation report
# ══════════════════════════════════════════════════════════════════
def saturation_report(trials, effort_lim, frac=0.90):
    """Flag trials where torque approaches the effort limit or flat-tops."""
    print("\n" + "=" * 72)
    print("A. SATURATION / EFFORT REPORT")
    print("   (if a trial clips, its 'lag' may be saturation, not bandwidth)")
    print("=" * 72)
    print(f"{'trial':<26s}{'joint':>6s}{'|tau|max':>10s}{'lim':>8s}"
          f"{'%lim':>7s}{'|vel|max':>10s}  flag")
    print("-" * 72)
    any_flag = False
    for nm, tr in trials.items():
        idx = int(tr["cfg"]["idx"])
        vmax = float(np.max(np.abs(tr["vel"][:, idx])))
        if tr["torque"] is None:
            print(f"{nm:<26s}{idx:>6d}{'   n/a':>10s}{'-':>8s}{'-':>7s}"
                  f"{vmax:>10.2f}  (no torque logged)")
            continue
        tau = tr["torque"][:, idx]
        tmax = float(np.max(np.abs(tau)))
        lim = float(effort_lim[idx])
        pct = 100.0 * tmax / lim if lim > 0 else float("nan")
        # flat-top detection: many samples pinned near the peak magnitude
        near_peak = np.mean(np.abs(np.abs(tau) - tmax) < 0.02 * max(tmax, 1e-6))
        flag = ""
        if pct >= 100.0 * frac:
            flag = "EFFORT"
        if near_peak > 0.05 and tmax > 0.3 * lim:
            flag = (flag + "+FLAT").lstrip("+")
        if flag:
            any_flag = True
        print(f"{nm:<26s}{idx:>6d}{tmax:>10.2f}{lim:>8.1f}{pct:>7.0f}"
              f"{vmax:>10.2f}  {flag}")
    if not any_flag:
        print("\n  No trial approaches the effort limit -> linear premise OK on torque.")
    else:
        print("\n  Flagged trials saturate; exclude them from any LTI/lag fit.")


# ══════════════════════════════════════════════════════════════════
# B. Quasi-rigid regression on the triangle
# ══════════════════════════════════════════════════════════════════
def _accel(vel, t):
    """d(vel)/dt from the MEASURED velocity (single differentiation)."""
    dt = float(np.median(np.diff(t))) if len(t) > 2 else 0.01
    v = vel.copy()
    if _HAVE_SCIPY and len(v) >= 11:
        win = min(31, len(v) // 2 * 2 - 1)
        win = max(5, win)
        v = savgol_filter(v, win, 3)
    return np.gradient(v, dt)


def regress_triangle(trials, dpos):
    """Per excited joint, fit the quasi-rigid EOM over all triangle trials.

        tau = J*acc + b*vel + tau_c*sign(vel) + g0 + g1*sin(th) + g2*cos(th)

    Returns {idx: params dict} and makes one figure per joint type.
    """
    print("\n" + "=" * 72)
    print("B. QUASI-RIGID REGRESSION on the triangle  (independent of step fit)")
    print("=" * 72)

    # group triangle trials by excited joint index
    by_joint = {}
    for nm, tr in trials.items():
        if tr["cfg"].get("signal") != "triangle":
            continue
        if tr["torque"] is None:
            continue
        by_joint.setdefault(int(tr["cfg"]["idx"]), []).append(nm)

    if not by_joint:
        print("  No triangle trials with torque found - skipping.")
        return {}

    out = {}
    figs = {}
    for idx, names in sorted(by_joint.items()):
        th_all, thd_all, acc_all, tau_all, dir_all = [], [], [], [], []
        for nm in names:
            tr = trials[nm]
            t = tr["t"]
            th = tr["pos"][:, idx]
            thd = tr["vel"][:, idx]
            tau = tr["torque"][:, idx]
            acc = _accel(thd, t)
            th_all.append(th); thd_all.append(thd)
            acc_all.append(acc); tau_all.append(tau)
            dir_all.append(np.sign(thd))
        th = np.concatenate(th_all); thd = np.concatenate(thd_all)
        acc = np.concatenate(acc_all); tau = np.concatenate(tau_all)
        sgn = np.sign(thd)

        # Drop the dead reversal samples where sign(vel) is ill-defined.
        vmax = np.max(np.abs(thd)) + 1e-9
        keep = np.abs(thd) > 0.03 * vmax
        # Design matrix: [J, b, tau_c, g0, g1, g2]
        A = np.column_stack([
            acc[keep], thd[keep], sgn[keep],
            np.ones(keep.sum()), np.sin(th[keep]), np.cos(th[keep]),
        ])
        y = tau[keep]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        J, b, tau_c, g0, g1, g2 = coef
        resid = y - A @ coef
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        g_amp = float(np.hypot(g1, g2))           # effective m*g*l of the link
        g_phase = float(np.arctan2(-g2, g1))      # offset angle (rough)

        out[idx] = dict(J=float(J), b=float(b), tau_c=float(abs(tau_c)),
                        g0=float(g0), g1=float(g1), g2=float(g2),
                        g_amp=g_amp, g_phase=g_phase, r2=r2, trials=names)

        jt = JTYPE[idx]
        print(f"\n  joint {idx} ({jt})  from {names}")
        print(f"    inertia   J     = {J: .4e}  kg m^2  (compare to armature)")
        print(f"    viscous   b     = {b: .4e}  Nm/(rad/s)  (compare viscous_friction)")
        print(f"    Coulomb   tau_c = {abs(tau_c): .4f}  Nm  (compare static/dynamic_friction)")
        print(f"    gravity   amp   = {g_amp: .4f}  Nm  (= effective m*g*l; compare URDF)")
        print(f"    fit       R^2   = {r2: .3f}")
        if r2 < 0.8:
            print("    [warn] low R^2: quasi-rigid model is missing something "
                  "(compliance? backlash? saturation?) - that itself is a finding.")

        # figure: torque-vs-angle hysteresis + fitted gravity centerline
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        up = thd > 0
        ax[0].scatter(th[up], tau[up], s=3, alpha=0.3, label="up-sweep")
        ax[0].scatter(th[~up], tau[~up], s=3, alpha=0.3, label="down-sweep")
        ths = np.linspace(th.min(), th.max(), 200)
        ax[0].plot(ths, g0 + g1 * np.sin(ths) + g2 * np.cos(ths),
                   "k-", lw=2, label="fitted gravity g(θ)")
        ax[0].set_xlabel("joint angle θ [rad]")
        ax[0].set_ylabel("motor torque τ [Nm]")
        ax[0].set_title(f"{jt}: hysteresis loop\n"
                        f"half-gap ≈ Coulomb+viscous; centerline = gravity")
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)

        ax[1].scatter(thd, tau - (g0 + g1 * np.sin(th) + g2 * np.cos(th)),
                      s=3, alpha=0.3)
        vv = np.linspace(thd.min(), thd.max(), 200)
        ax[1].plot(vv, b * vv + tau_c * np.sign(vv), "k-", lw=2,
                   label="b·θ̇ + τ_c·sign(θ̇)")
        ax[1].set_xlabel("joint velocity θ̇ [rad/s]")
        ax[1].set_ylabel("τ − gravity [Nm]")
        ax[1].set_title(f"{jt}: friction vs velocity\n"
                        f"intercept = Coulomb, slope = viscous")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        figs[jt] = fig

    for jt, fig in figs.items():
        fn = f"sysid_diag_triangle_{jt}.pdf"
        fig.savefig(fn, dpi=150)
        plt.close(fig)
        print(f"  -> {fn}")
    return out


# ══════════════════════════════════════════════════════════════════
# C. Amplitude-normalized step overlay
# ══════════════════════════════════════════════════════════════════
def amplitude_normalized_steps(trials, dpos):
    print("\n" + "=" * 72)
    print("C. AMPLITUDE-NORMALIZED STEPS  (linear lag -> curves coincide)")
    print("=" * 72)

    by_jt = {}
    for nm, tr in trials.items():
        c = tr["cfg"]
        if c.get("signal") != "step":
            continue
        amp = float(c.get("amp", 0.0))
        if abs(amp) < 1e-6:
            continue
        by_jt.setdefault(JTYPE[int(c["idx"])], []).append(nm)

    if not by_jt:
        print("  No step trials found - skipping.")
        return

    for jt, names in by_jt.items():
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        reached = []  # (amp, fraction-of-command reached at steady state)
        for nm in names:
            tr = trials[nm]
            idx = int(tr["cfg"]["idx"])
            amp = float(tr["cfg"]["amp"])
            t = tr["t"]
            dev = (tr["pos"][:, idx] - dpos[idx]) / amp
            ax[0].plot(t, dev, lw=1.0, label=f"{nm} (a={amp:+.2f})")
            # steady state = mean of last 25% of the trial
            n = len(dev)
            ss = float(np.mean(dev[int(0.75 * n):]))
            reached.append((amp, ss))
        ax[0].axhline(1.0, color="k", ls=":", lw=0.6)
        ax[0].set_xlabel("time [s]")
        ax[0].set_ylabel("(pos − default) / amplitude")
        ax[0].set_title(f"{jt}: normalized step responses\n"
                        f"overlap ⇒ linear; spread ⇒ nonlinear (stiction/sat)")
        ax[0].legend(fontsize=7)
        ax[0].grid(alpha=0.3)

        reached.sort()
        a = np.array([r[0] for r in reached])
        f = np.array([r[1] for r in reached])
        ax[1].plot(np.abs(a), f, "o-")
        ax[1].axhline(1.0, color="k", ls=":", lw=0.6)
        ax[1].set_xlabel("|command amplitude| [rad]")
        ax[1].set_ylabel("fraction of command reached")
        ax[1].set_title(f"{jt}: dead-band signature\n"
                        f"smaller steps reaching less ⇒ static friction band")
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fn = f"sysid_diag_steps_{jt}.pdf"
        fig.savefig(fn, dpi=150)
        plt.close(fig)
        # quick numeric read-out
        spread = float(np.nanmax(f) - np.nanmin(f)) if len(f) else float("nan")
        print(f"  {jt:10s} reached-fraction spread across amplitudes = {spread:.3f}"
              f"   (large ⇒ nonlinear)   -> {fn}")


# ══════════════════════════════════════════════════════════════════
# D. Chirp frequency response
# ══════════════════════════════════════════════════════════════════
def chirp_frf(trials, fs, replay=None, fmin=0.3, fmax=15.0):
    print("\n" + "=" * 72)
    print("D. CHIRP FREQUENCY RESPONSE  (Bode shape says WHICH model)")
    print("=" * 72)
    if not _HAVE_SCIPY:
        print("  scipy unavailable - skipping FRF.")
        return

    chirps = [(nm, tr) for nm, tr in trials.items()
              if tr["cfg"].get("signal") == "chirp"]
    if not chirps:
        print("  No chirp trials found - skipping.")
        return

    def frf(u, y):
        n = len(u)
        nperseg = int(min(512, max(64, n // 4)))
        f, Puu = welch(u, fs=fs, nperseg=nperseg)
        _, Puy = csd(u, y, fs=fs, nperseg=nperseg)
        H = Puy / (Puu + 1e-18)
        _, C = coherence(u, y, fs=fs, nperseg=nperseg)
        return f, H, C

    for nm, tr in chirps:
        idx = int(tr["cfg"]["idx"])
        jt = JTYPE[idx]
        u = tr["p_des"][:, idx] - np.mean(tr["p_des"][:, idx])
        y = tr["pos"][:, idx] - np.mean(tr["pos"][:, idx])
        f, H, C = frf(u, y)
        band = (f >= fmin) & (f <= fmax)

        fig, ax = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
        ax[0].semilogx(f[band], 20 * np.log10(np.abs(H[band]) + 1e-9),
                       "b-", label="real")
        ax[1].semilogx(f[band], np.degrees(np.angle(H[band])), "b-")
        ax[2].semilogx(f[band], C[band], "b-")

        # optional sim overlay (same recorded command, sim position)
        if replay is not None and f"{nm}__sim_pos" in replay:
            ys = replay[f"{nm}__sim_pos"][:, idx]
            ys = ys[: len(u)] - np.mean(ys[: len(u)])
            fs_, Hs, Cs = frf(u, ys)
            bs = (fs_ >= fmin) & (fs_ <= fmax)
            ax[0].semilogx(fs_[bs], 20 * np.log10(np.abs(Hs[bs]) + 1e-9),
                           "r--", label="sim")
            ax[1].semilogx(fs_[bs], np.degrees(np.angle(Hs[bs])), "r--")

        # mark the low-coherence region as unreliable
        lowc = band & (C < 0.8)
        if np.any(lowc):
            ax[2].fill_between(f[band], 0, 1,
                               where=(C[band] < 0.8), color="orange",
                               alpha=0.15, step="mid")

        ax[0].set_ylabel("|H|  [dB]")
        ax[0].set_title(f"{nm}: p_des → pos  (delay = linear phase; "
                        f"2nd-order = −90° roll-off; peak = compliance)")
        ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")
        ax[1].set_ylabel("phase [deg]"); ax[1].grid(alpha=0.3, which="both")
        ax[2].set_ylabel("coherence"); ax[2].set_ylim(0, 1.05)
        ax[2].axhline(0.8, color="k", ls=":", lw=0.6)
        ax[2].set_xlabel("frequency [Hz]"); ax[2].grid(alpha=0.3, which="both")
        fig.tight_layout()
        fn = f"sysid_diag_frf_{nm}.pdf"
        fig.savefig(fn, dpi=150)
        plt.close(fig)

        good = band & (C >= 0.8)
        cov = 100.0 * np.sum(good) / max(np.sum(band), 1)
        print(f"  {nm:<22s} coherence≥0.8 over {cov:4.0f}% of {fmin}-{fmax}Hz"
              f"   -> {fn}")
        if cov < 50:
            print("     [warn] poor coherence: chirp amplitude too small / "
                  "stiction dominates. FRF unreliable; raise chirp amp on the "
                  "next hardware run for a clean Bode.")


# ══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="raw hardware .npz from sysid_experiment.py")
    ap.add_argument("--replay", default=None,
                    help="optional *_replay.npz for sim FRF overlay")
    ap.add_argument("--effort", type=float, nargs=6, default=None,
                    help="effort limits per joint (defaults to experiment values)")
    args = ap.parse_args()

    trials, dpos, fs = load_experiment(args.npz)
    replay = load_replay(args.replay)
    effort = np.array(args.effort) if args.effort else DEFAULT_EFFORT_LIM

    print(f"Loaded {len(trials)} trials from {args.npz}  (fs={fs:.0f} Hz)")
    if replay is not None:
        print(f"Loaded sim overlay from {args.replay}")

    saturation_report(trials, effort)
    params = regress_triangle(trials, dpos)
    amplitude_normalized_steps(trials, dpos)
    chirp_frf(trials, fs, replay=replay)

    # one-line cross-check the user actually cares about
    if params:
        print("\n" + "=" * 72)
        print("CROSS-CHECK: triangle-derived vs your replay-tuned numbers")
        print("  Put these J / tau_c / b next to ARMATURE_MULT and FRICTION_TUNE.")
        print("  If they disagree, the step fit (and the 2nd-order prefilter")
        print("  compensating for it) is absorbing physics that belongs here.")
        print("=" * 72)
    print("\nDone. PDFs written to the current directory.")


if __name__ == "__main__":
    main()
