"""Aggregate RL/logs/eval_p9sel/<name>__<scenario>.json into a markdown table.

Part of the 2026-08-23 P9 checkpoint-selection sweep (run_eval_p9sel_GPU2.ps1;
one JSON per (checkpoint, scenario) because Isaac Sim allows one env per
process). Usage:  python RL/scripts/rsl_rl/aggregate_eval_p9sel.py [outdir]
"""
import glob
import json
import os
import sys

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "..", "..", "logs", "eval_p9sel")

# (scenario, metric, column label, format)
COLS = [
    ("hw_matched",      "time_to_fall/survivor_frac",   "hw surv15s", "{:.3f}"),
    ("hw_matched",      "balance/action_jerk_abs_mean", "hw jerk6",   "{:.2f}"),
    ("hw_matched",      "clip_det/knee_max",            "hw kneeclip", "{:.2f}"),
    ("hw_matched",      "balance/flight_fraction",      "hw flight",  "{:.2f}"),
    ("hw_matched",      "drift/pos_drift_mean",         "hw drift",   "{:.2f}"),
    ("hw_harsh_arm13",  "time_to_fall/survivor_frac",   "arm13 surv", "{:.3f}"),
    ("hw_harsh_arm13",  "survival/at_5s",               "arm13 5s",   "{:.3f}"),
    ("balance_full_dr", "time_to_fall/survivor_frac",   "dr surv",    "{:.3f}"),
    ("balance_full_dr", "survival/at_5s",               "dr 5s",      "{:.3f}"),
    ("balance_full_dr", "balance/action_jerk_abs_mean", "dr jerk6",   "{:.2f}"),
    # 2026-08-27 P13 scenarios (logs/eval_p13r; older dirs show "—")
    ("hw_push_recovery", "time_to_fall/survivor_frac",  "push surv",  "{:.3f}"),
    ("r1_train_dr",     "time_to_fall/survivor_frac",   "r1dr surv",  "{:.3f}"),
    ("r1_train_dr",     "survival/at_5s",               "r1dr 5s",    "{:.3f}"),
]

# name -> scenario -> metrics
merged: dict[str, dict[str, dict[str, float]]] = {}
for path in sorted(glob.glob(os.path.join(OUT_DIR, "*__*.json"))):
    base = os.path.splitext(os.path.basename(path))[0]
    name, _, scen = base.partition("__")
    with open(path) as f:
        d = json.load(f)
    merged.setdefault(name, {}).update(d.get("results", {}))

rows = []
for name in sorted(merged):
    row = [name]
    for scen, metric, _, fmt in COLS:
        v = merged[name].get(scen, {}).get(metric)
        row.append(fmt.format(v) if v is not None else "—")
    rows.append(row)

hdr = ["checkpoint"] + [c[2] for c in COLS]
w = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(len(hdr))]
print("| " + " | ".join(h.ljust(w[i]) for i, h in enumerate(hdr)) + " |")
print("|" + "|".join("-" * (w[i] + 2) for i in range(len(hdr))) + "|")
for r in rows:
    print("| " + " | ".join(str(v).ljust(w[i]) for i, v in enumerate(r)) + " |")
print()
print("jerk6 = deterministic-policy |jerk| summed over 6 joints; /6 for the")
print("per-joint figure comparable to TB's diagnostics/action_jerk_abs_mean")
print("(which is measured on NOISY sampled actions and so sits ~1.95*sigma higher).")
