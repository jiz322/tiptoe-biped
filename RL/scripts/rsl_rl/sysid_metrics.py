"""sysid_metrics.py — the statistics the plant sysid matches between hardware
balance trials and sim rollouts (2026-09-10).

Pure numpy, no Isaac imports: data/_s4_sysid_targets.py uses it to turn the
hardware npz files into `sysid_targets.json`, and sysid_optuna.py uses the
SAME functions on the sim rollouts, so both sides are scored identically.

CLEAN WINDOW (user, 09-10 eve): on the rig the safety tether can be in contact
from the first deep dip onward, so only the opening of each recording is
trustworthy — 0.6 s by default, per-trial overrides where the user knows better
(143406: 0.8 s). Everything the OBJECTIVE uses lives inside that window:
    pitch_grid : TRUE torso pitch (deg, from raw projected gravity, negative =
                 leaning back) on GRID, NaN at/after the trial's clean window
                 (or after a fall / the end of the data)
    t_exc      : first time the tilt exceeds EXC_TILT_DEG (25 deg) inside the
                 window — the "failure onset" event that precedes any tether
                 action; censored at the window end otherwise
Cell statistics = mean pitch of the trials still clean and standing at each
grid time + fraction that crossed 25 deg by each time in EXC_T. Distance =
chi-square-like with SIGMA_PITCH_DEG / SIGMA_E tolerances.

Also computed and REPORTED (never in the objective): t_fall, the 45 deg fall
time over the whole recording, and F = fraction fallen by SURV_T. For hardware
these are tether-contaminated; for sim they are the long-horizon prediction.
"""
from __future__ import annotations

import warnings

import numpy as np

DEFAULT_WINDOW_S = 0.6
GRID = np.round(np.arange(0.1, 2.0 + 1e-9, 0.1), 3)      # s, pitch-trace grid (window masks the tail)
EXC_TILT_DEG = 25.0
EXC_T = (0.4, 0.6)                                          # s, "crossed 25 deg by" checkpoints (inside window)
SIGMA_PITCH_DEG = 4.0        # pitch trace tolerance (trial-to-trial spread ~4-5 deg)
SIGMA_E = 0.25               # exceedance-fraction tolerance (n = 3..6 trials per cell)
MIN_N = 2                    # trials needed for a grid point / checkpoint to count
PITCH_PENALTY_Z = 3.0        # z assigned where hw is valid but sim has < MIN_N valid envs

FALL_TILT_DEG = 45.0         # reported only
END_TILT_DEG = 28.0          # hw: recording ends beyond this, still going down -> fall at t_end
SURV_T = (1.0, 1.5, 2.5, 5.0)


def pitch_tilt_deg(pg):
    """pg [..., 3] projected gravity in the body frame -> (pitch, tilt) in deg."""
    pg = np.asarray(pg, dtype=np.float64)
    pitch = np.degrees(np.arcsin(np.clip(pg[..., 0], -1.0, 1.0)))
    tilt = np.degrees(np.arccos(np.clip(-pg[..., 2], -1.0, 1.0)))
    return pitch, tilt


def trial_features(t, pg, t_term=np.inf, hw_end_rule=False, t_clean=DEFAULT_WINDOW_S):
    """One trial -> dict(t_fall, t_end, t_exc, t_clean, pitch_grid).

    t       : [T] seconds from handover (first policy tick = 0)
    pg      : [T, 3] TRUE / raw projected gravity
    t_term  : sim env termination time (s), inf for hardware
    hw_end_rule : "log stopped inside a fall" rule (hardware only)
    t_clean : end of the trustworthy window (s)
    """
    t = np.asarray(t, dtype=np.float64)
    pitch, tilt = pitch_tilt_deg(pg)
    t_end = float(t[-1])
    over = tilt > FALL_TILT_DEG
    t_fall = float(t[np.argmax(over)]) if over.any() else np.inf
    t_fall = min(t_fall, float(t_term))
    if hw_end_rule and not np.isfinite(t_fall):
        tail = tilt[t >= t_end - 0.3]
        if len(tail) and tail.mean() > END_TILT_DEG and tilt[-1] >= tail[0] - 2.0:
            t_fall = t_end
    t_clean = float(min(t_clean, t_end + 1e-9, float(t_term)))
    exc = (tilt > EXC_TILT_DEG) & (t <= t_clean + 1e-6)     # window end inclusive
    t_exc = float(t[np.argmax(exc)]) if exc.any() else np.inf
    idx = np.searchsorted(t, GRID - 1e-6)
    ok = (GRID < t_fall) & (GRID <= t_clean + 1e-6) & (idx < len(t))
    grid = np.full(len(GRID), np.nan)
    grid[ok] = pitch[idx[ok]]
    return {"t_fall": t_fall, "t_end": t_end, "t_exc": t_exc, "t_clean": t_clean, "pitch_grid": grid}


def _fraction_by(t_event, t_info, checkpoints):
    """fraction of trials with event <= T among trials informative at T
    (event happened, or observed clean up to T)."""
    F, n = [], []
    for T in checkpoints:
        hit = t_event <= T
        info = hit | (t_info >= T - 0.011)     # one control tick of slack (sim episode ends at (T-1)*dt)
        F.append(float(hit.sum() / info.sum()) if info.sum() else np.nan)
        n.append(int(info.sum()))
    return np.asarray(F), np.asarray(n)


def cell_stats(features):
    """Aggregate a list of trial_features dicts."""
    t_fall = np.asarray([f["t_fall"] for f in features], dtype=np.float64)
    t_end = np.asarray([f["t_end"] for f in features], dtype=np.float64)
    t_exc = np.asarray([f["t_exc"] for f in features], dtype=np.float64)
    t_clean = np.asarray([f["t_clean"] for f in features], dtype=np.float64)
    pitch_grid = np.stack([f["pitch_grid"] for f in features]).astype(np.float64)
    n_grid = np.isfinite(pitch_grid).sum(axis=0)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_grid = np.where(n_grid > 0, np.nanmean(pitch_grid, axis=0), np.nan)
    E, E_n = _fraction_by(t_exc, t_clean, EXC_T)
    F, F_n = _fraction_by(t_fall, t_end, SURV_T)
    return {"n": int(len(features)), "pitch_mean": mean_grid, "pitch_n": n_grid,
            "E": E, "E_n": E_n, "F": F, "F_n": F_n,
            "t_fall": t_fall, "t_end": t_end, "t_exc": t_exc, "t_clean": t_clean}


def cell_distance(sim, hw):
    """Chi-square-like distance between two cell_stats dicts (sim vs hardware),
    using ONLY in-window quantities (pitch grid + 25-deg exceedance)."""
    hw_mean = np.asarray(hw["pitch_mean"], dtype=np.float64)
    hw_n = np.asarray(hw["pitch_n"])
    sim_mean = np.asarray(sim["pitch_mean"], dtype=np.float64)
    sim_n = np.asarray(sim["pitch_n"])
    hw_valid = (hw_n >= MIN_N) & np.isfinite(hw_mean)
    z = np.zeros(len(GRID))
    both = hw_valid & (sim_n >= MIN_N) & np.isfinite(sim_mean)
    z[both] = (sim_mean[both] - hw_mean[both]) / SIGMA_PITCH_DEG
    z[hw_valid & ~both] = PITCH_PENALTY_Z
    pitch_term = float(np.mean(z[hw_valid] ** 2)) if hw_valid.any() else 0.0
    pitch_rms = float(np.sqrt(np.mean((sim_mean[both] - hw_mean[both]) ** 2))) if both.any() else np.nan

    E_hw = np.asarray(hw["E"], dtype=np.float64)
    E_hw_n = np.asarray(hw["E_n"])
    E_sim = np.asarray(sim["E"], dtype=np.float64)
    ok = (E_hw_n >= MIN_N) & np.isfinite(E_hw) & np.isfinite(E_sim)
    exc_term = float(np.mean(((E_sim[ok] - E_hw[ok]) / SIGMA_E) ** 2)) if ok.any() else 0.0
    return pitch_term + exc_term, {
        "pitch_term": pitch_term,
        "exc_term": exc_term,
        "pitch_rms_deg": pitch_rms,
        "n_grid_used": int(hw_valid.sum()),
        "n_grid_both": int(both.sum()),
    }


def _lst(a, nan_to=None):
    out = []
    for v in np.asarray(a, dtype=np.float64):
        out.append(nan_to if not np.isfinite(v) else round(float(v), 4))
    return out


def stats_to_json(s):
    return {"n": s["n"], "pitch_mean": _lst(s["pitch_mean"]), "pitch_n": [int(v) for v in s["pitch_n"]],
            "E": _lst(s["E"]), "E_n": [int(v) for v in s["E_n"]],
            "F": _lst(s["F"]), "F_n": [int(v) for v in s["F_n"]],
            "t_fall": _lst(s["t_fall"]), "t_end": _lst(s["t_end"]), "t_exc": _lst(s["t_exc"]),
            "t_clean": _lst(s["t_clean"])}


def _arr(lst, none_to=np.nan):
    return np.asarray([none_to if v is None else v for v in lst], dtype=np.float64)


def stats_from_json(d):
    inf = float("inf")
    return {"n": d["n"], "pitch_mean": _arr(d["pitch_mean"]), "pitch_n": np.asarray(d["pitch_n"]),
            "E": _arr(d["E"]), "E_n": np.asarray(d["E_n"]), "F": _arr(d["F"]), "F_n": np.asarray(d["F_n"]),
            "t_fall": _arr(d["t_fall"], inf), "t_end": _arr(d["t_end"]), "t_exc": _arr(d["t_exc"], inf),
            "t_clean": _arr(d["t_clean"])}


def format_cell_table(name, sim, hw):
    """Side-by-side text for one cell: in-window pitch + 25-deg exceedance (objective),
    then the 45-deg fall fractions (reported only; hardware values tether-suspect)."""
    picks = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    idx = [int(np.argmin(np.abs(GRID - p))) for p in picks]
    hdr = "  ".join(f"p@{GRID[i]:.1f}s" for i in idx) + "  |  " + "  ".join(f"E@{T:g}s" for T in EXC_T) \
        + "  ||  " + "  ".join(f"F@{T:g}s" for T in SURV_T)
    lines = [f"[{name}]  n_hw={hw['n']}  n_sim={sim['n']}   (E = crossed 25 deg by; F = 45-deg fall by, report only)",
             f"          {hdr}"]
    for lab, s in (("hw ", hw), ("sim", sim)):
        pm = "  ".join(f"{s['pitch_mean'][i]:6.1f}" if np.isfinite(s["pitch_mean"][i]) else "   nan" for i in idx)
        em = "  ".join(f"{v:5.2f}" if np.isfinite(v) else "  nan" for v in s["E"])
        fm = "  ".join(f"{v:5.2f}" if np.isfinite(v) else "  nan" for v in s["F"])
        lines.append(f"   {lab}   {pm}  |  {em}  ||  {fm}")
    return "\n".join(lines)
