"""make_figs.py — figures for the belt-coupling sim2real report.

Reads the replay-sweep JSONs/NPZs and ablation logs in control/, regenerates
every figure into figs/. Run from the repo root with any python that has
numpy+matplotlib (no Isaac needed):

    python reports/belt_coupling_sim2real/make_figs.py

Palette: dataviz reference palette (light mode), used unchanged — categorical
slots 1-3 (#2a78d6 blue, #eb6834 orange, #1baf7a aqua; all-pairs validated per
palette.md), diverging blue #2a78d6 / red #e34948, ink/grid tokens as documented.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CTRL = os.path.join(ROOT, "control")
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

# ── palette (dataviz reference, light mode) ─────────────────────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"        # categorical slot 1 / diverging cool pole
BLUE_LIGHT = "#86b6ef"  # sequential step 250 (2-shade dumbbell pair)
ORANGE = "#eb6834"      # categorical slot 2
AQUA = "#1baf7a"        # categorical slot 3 (sub-3:1 on light -> table backs it)
RED = "#e34948"         # diverging warm pole

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11.5, "axes.titlelocation": "left",
    "axes.titleweight": "bold", "axes.titlecolor": INK,
    "font.size": 9.5, "legend.frameon": False,
    "savefig.dpi": 170, "savefig.bbox": "tight",
})

JL = ["hr1", "hp1", "kn1", "hr2", "hp2", "kn2"]
LOGS = [  # (stem, short tag) — coupling corpus, chronological
    ("model_4150_log_20260806_115256", "4150 11:52"),
    ("model_4150_log_20260806_115327", "4150 11:53"),
    ("model_4150_log_20260806_115703", "4150 11:57a"),
    ("model_4150_log_20260806_115715", "4150 11:57b"),
    ("2026-06-20_22-02-53_jrk020_grf13_log_20260806_120142", "jrk020 12:01"),
    ("2026-06-20_22-02-53_jrk020_grf13_log_20260806_120222", "jrk020 12:02"),
    ("2026-06-20_22-02-53_jrk020_grf13_log_20260806_120301", "jrk020 12:03"),
]
CELL = "fixed_d3"   # every sweep run picked fixed_d3 as best; compare like-for-like


def load_cell(stem, variant):
    with open(os.path.join(CTRL, f"{stem}_simreplay_{variant}_tf.json"),
              encoding="utf-8") as fh:
        j = json.load(fh)
    c = j[CELL]
    return {k: np.array(c[k]) for k in ("rmse_pos", "rmse_vel", "rmse_tau")}


DATA = {(stem, v): load_cell(stem, v)
        for stem, _ in LOGS for v in ("legacy", "belt_ffoff")}

_POSLINE = re.compile(
    r"pos RMSE\s+hr1=([\d.]+)\s+hp1=([\d.]+)\s+kn1=([\d.]+)"
    r"\s+hr2=([\d.]+)\s+hp2=([\d.]+)\s+kn2=([\d.]+)\s+mean=([\d.]+)")


def parse_poslines(logfile):
    out = []
    with open(os.path.join(CTRL, logfile), encoding="utf-8", errors="replace") as fh:
        for m in _POSLINE.finditer(fh.read()):
            out.append(np.array([float(g) for g in m.groups()]))
    return out


def r2_from_coupling_analysis():
    """Per-log knee torque-model fit R2 (mean kn1/kn2) from the hardware-side
    two-regressor fit in data/coupling_torque_analysis.py (run + parse)."""
    p = subprocess.run([sys.executable, os.path.join(ROOT, "data", "coupling_torque_analysis.py")],
                      cwd=CTRL, capture_output=True, text=True, check=True)
    rows = [ln.split() for ln in p.stdout.splitlines()
            if re.search(r"\skn[12]\s", ln)]
    assert len(rows) == 2 * len(LOGS), f"expected {2*len(LOGS)} rows, got {len(rows)}"
    # rows come in LOGS order (kn1, kn2 per log); R2c is 3rd column from the end
    return {LOGS[i][0]: (float(rows[2 * i][-3]) + float(rows[2 * i + 1][-3])) / 2
            for i in range(len(LOGS))}


def style_ax(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)


def header(ax, title, sub=None):
    """Title + optional subtitle above the axes, spaced in offset points so
    panel height doesn't cause collisions."""
    y0 = 8
    if sub:
        ax.annotate(sub, xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, y0), textcoords="offset points",
                    va="bottom", fontsize=8.5, color=INK2)
        y0 += 13 * (sub.count("\n") + 1) + 4
    ax.annotate(title, xy=(0, 1), xycoords="axes fraction",
                xytext=(0, y0), textcoords="offset points",
                va="bottom", fontsize=11.5, fontweight="bold", color=INK)


# ════════════════════════════════════════════════════════════════════
# Fig 1 — equivalence gate (grouped bars, 3 series x 6 joints)
# ════════════════════════════════════════════════════════════════════
eq_legacy_rows = parse_poslines("_eq_legacy.log")        # [uncoupled, +injection]
eq_belt_rows = parse_poslines("_eq_belt_arm42.log")      # [belt arm4.2e-3]
gate = {"legacy (uncoupled)": eq_legacy_rows[0][:6],
        "legacy + knee injection": eq_legacy_rows[1][:6],
        "belt actuator": eq_belt_rows[0][:6]}

fig, ax = plt.subplots(figsize=(7.4, 3.1))
x = np.arange(6)
w = 0.26
for i, ((name, vals), col) in enumerate(zip(gate.items(), [BLUE, ORANGE, AQUA])):
    ax.bar(x + (i - 1) * w, vals, w * 0.92, color=col, label=name,
           edgecolor=SURFACE, linewidth=1.2)
ax.set_xticks(x, JL)
ax.set_ylabel("one-step pos RMSE (rad)")
header(ax, "Equivalence gate — new actuator reproduces the validated injection exactly",
       "log 115703, teacher forcing, fixed base, delay 3 · orange and aqua bars are identical to 1e-4")
ax.grid(axis="y"), ax.grid(axis="x", visible=False)
ax.set_axisbelow(True)
ax.legend(loc="upper left", fontsize=8.5, bbox_to_anchor=(0.0, 0.98))
fig.savefig(os.path.join(FIGS, "fig1_equivalence_gate.png"))
plt.close(fig)

# ════════════════════════════════════════════════════════════════════
# Fig 2 — per-log mean pos RMSE, legacy -> belt (dumbbell)
# ════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7.4, 3.4))
y = np.arange(len(LOGS))[::-1]
for yi, (stem, tag) in zip(y, LOGS):
    a = DATA[(stem, "legacy")]["rmse_pos"].mean()
    b = DATA[(stem, "belt_ffoff")]["rmse_pos"].mean()
    ax.plot([a, b], [yi, yi], color=BASELINE, lw=1.6, zorder=1)
    ax.plot(a, yi, "o", ms=8, color=BLUE_LIGHT, zorder=2)
    ax.plot(b, yi, "o", ms=8, color=BLUE, zorder=3)
    d = b - a
    ax.annotate(f"{d:+.4f}", (max(a, b) + 0.004, yi), va="center",
                fontsize=8.5, color=INK2)
ax.set_yticks(y, [t for _, t in LOGS])
ax.set_xlabel("mean one-step pos RMSE over 6 joints (rad)")
ax.set_xlim(0, 0.145)
header(ax, "Belt model helps 4 logs, hurts 3 — the single-log conclusion does not generalize",
       "teacher forcing, fixed base, delay 3 · annotation = belt − legacy")
style_ax(ax)
ax.plot([], [], "o", ms=8, color=BLUE_LIGHT, label="legacy actuator")
ax.plot([], [], "o", ms=8, color=BLUE, label="belt actuator")
ax.legend(loc="upper left", fontsize=8.5)
fig.savefig(os.path.join(FIGS, "fig2_perlog_dumbbell.png"))
plt.close(fig)

# ════════════════════════════════════════════════════════════════════
# Fig 3 — per-joint corpus-mean delta (belt − legacy), 3 metrics
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
metrics = [("rmse_pos", "Δ pos RMSE (rad)"), ("rmse_vel", "Δ vel RMSE (rad/s)"),
           ("rmse_tau", "Δ torque RMSE (Nm)")]
for ax, (mk, xlabel) in zip(axes, metrics):
    d = np.mean([DATA[(s, "belt_ffoff")][mk] - DATA[(s, "legacy")][mk]
                 for s, _ in LOGS], axis=0)
    cols = [BLUE if v < 0 else RED for v in d]
    ax.barh(np.arange(6)[::-1], d, 0.55, color=cols)
    ax.set_yticks(np.arange(6)[::-1], JL)
    ax.axvline(0, color=BASELINE, lw=1.2)
    ax.set_xlabel(xlabel)
    style_ax(ax)
header(axes[0], "Per-joint change, mean over 7 logs (belt − legacy)",
       "blue = belt model closer to real, red = further · teacher forcing, fixed d3")
fig.tight_layout(w_pad=2.0, rect=[0, 0, 1, 0.88])
fig.savefig(os.path.join(FIGS, "fig3_perjoint_delta.png"))
plt.close(fig)

# ════════════════════════════════════════════════════════════════════
# Fig 4 — discriminator: torque-model fit R² vs knee delta
# ════════════════════════════════════════════════════════════════════
R2 = r2_from_coupling_analysis()
fig, ax = plt.subplots(figsize=(6.6, 3.6))
for stem, tag in LOGS:
    dk = np.mean([DATA[(stem, "belt_ffoff")]["rmse_pos"][i]
                  - DATA[(stem, "legacy")]["rmse_pos"][i] for i in (2, 5)])
    col = BLUE if dk < 0 else RED
    ax.plot(R2[stem], dk, "o", ms=9, color=col, zorder=3)
    dx, ha = (0.012, "left") if R2[stem] < 0.7 else (-0.012, "right")
    ax.annotate(tag, (R2[stem] + dx, dk), va="center", ha=ha,
                fontsize=8.5, color=INK2)
ax.axhline(0, color=BASELINE, lw=1.2, ls=(0, (4, 3)))
ax.set_xlabel("actuator torque-model fit R² on hardware log (mean kn1/kn2)")
ax.set_ylabel("Δ knee pos RMSE, belt − legacy (rad)")
header(ax, "Coupling model helps exactly where the base motor model holds",
       "fit: τ_meas ~ a·τ_cmd + b·belt-cross (coupling_torque_analysis.py) · red = belt regresses\n"
       "regressions only occur where NO PD law explains measured torque (R² < 0.5) · jrk020 12:02 is near-static")
ax.set_axisbelow(True)
fig.savefig(os.path.join(FIGS, "fig4_r2_discriminator.png"))
plt.close(fig)

# ════════════════════════════════════════════════════════════════════
# Fig 5 — term ablation on the worst regressing log (115256)
# ════════════════════════════════════════════════════════════════════
leg_115256 = DATA[("model_4150_log_20260806_115256", "legacy")]["rmse_pos"].mean()
both_115256 = DATA[("model_4150_log_20260806_115256", "belt_ffoff")]["rmse_pos"].mean()
stale_only = parse_poslines("_abl_staleonly.log")[0][6]
drag_only = parse_poslines("_abl_dragonly.log")[0][6]
rows = [("legacy (no coupling)", leg_115256),
        ("+ stale-hip hold only", stale_only),
        ("+ belt drag only", drag_only),
        ("+ both (belt, ff off)", both_115256)]
fig, ax = plt.subplots(figsize=(6.8, 2.7))
yy = np.arange(len(rows))[::-1]
ax.barh(yy, [v for _, v in rows], 0.55, color=BLUE)
for yi, (_, v) in zip(yy, rows):
    ax.annotate(f"{v:.4f}", (v + 0.0012, yi), va="center", fontsize=8.5, color=INK2)
ax.axvline(leg_115256, color=MUTED, lw=1.1, ls=(0, (4, 3)))
ax.set_yticks(yy, [n for n, _ in rows])
ax.set_xlabel("mean one-step pos RMSE (rad)")
ax.set_xlim(0.09, 0.128)
header(ax, "On a regressing log both terms hurt, roughly additively",
       "log 4150 11:52 (32% action saturation, torque-fit R² 0.36) · dashed = legacy baseline\n"
       "note: x-axis starts at 0.09 to resolve the deltas")
style_ax(ax)
fig.savefig(os.path.join(FIGS, "fig5_ablation_115256.png"))
plt.close(fig)

# ════════════════════════════════════════════════════════════════════
# Fig 6 — knee one-step residual traces, good log vs bad log
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(8.6, 4.6), sharex=False)
for ax, (stem, tag, note) in zip(axes, [
        ("model_4150_log_20260806_115703", "4150 11:57a", "improving log (torque-fit R² 0.80)"),
        ("model_4150_log_20260806_115256", "4150 11:52", "regressing log (torque-fit R² 0.36)")]):
    real = np.load(os.path.join(CTRL, stem + ".npz"), allow_pickle=True)["sim_joint_pos"]
    for variant, col, name in [("legacy", BLUE, "legacy"), ("belt_ffoff", ORANGE, "belt")]:
        sp = np.load(os.path.join(CTRL, f"{stem}_simreplay_{variant}_tf.npz"),
                     allow_pickle=True)[f"{CELL}__sim_pos"]
        res = sp[:-1, 2] - real[1:, 2]       # kn1 one-step prediction error
        t = np.arange(len(res)) * 0.01
        ax.plot(t, res, color=col, lw=1.1, label=name, alpha=0.95)
    ax.axhline(0, color=BASELINE, lw=1.0)
    ax.set_ylabel("kn1 residual (rad)")
    ax.set_title(f"{tag} — {note}", fontsize=10)
    ax.grid(axis="y"), ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
axes[0].legend(loc="upper right", fontsize=8.5, ncols=2)
axes[1].set_xlabel("time (s)")
fig.suptitle("Knee one-step prediction residual (sim − real), teacher forcing",
             x=0.125, ha="left", fontsize=11.5, fontweight="bold", color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(FIGS, "fig6_residual_traces.png"))
plt.close(fig)

print("figures written to", FIGS)
