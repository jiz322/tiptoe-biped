"""replay_sweep.py — batch driver for the sim2real actuator A/B.

Runs data/sysid_replay_policy_log.py over the 7-log coupling corpus
(coupling_torque_analysis.py's list) x actuator variants, then aggregates the
per-run JSONs into one before/after table (per-joint pos / vel / tau RMSE).

Run inside the Isaac Lab conda env (env_isaaclab), from the repo root:

    python data/replay_sweep.py                     # legacy + belt, TF, all logs
    python data/replay_sweep.py --variants legacy,belt,legacy_arm
    python data/replay_sweep.py --logs model_4150_log_20260806_115703.npz
    python data/replay_sweep.py --aggregate-only    # just re-read the JSONs

Each replay is a separate process (Isaac Lab app can't rebuild an
articulation's actuators in-process). Output of run i goes to
control/_sweep_<log>_<variant>.log; results land next to the NPZ as
<stem>_simreplay_<variant>[_tf].json (the replay's own naming).

KNEE-FF PROTOCOL: all 7 corpus logs predate the KNEE_DECOUPLE_FF hardware
patch (they are the 2026-08-06 runs that revealed the belt drag), so the belt
variant replays them with --knee-ff off. Add post-patch logs to LOG_KNEE_FF
with "on".
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
REPLAY = os.path.join(_HERE, "sysid_replay_policy_log.py")

JOINT_LABELS = ["hr1", "hp1", "kn1", "hr2", "hp2", "kn2"]

# The coupling corpus (coupling_torque_analysis.py LOGS), relative to control/.
CORPUS = [
    "model_4150_log_20260806_115256.npz",
    "model_4150_log_20260806_115327.npz",
    "model_4150_log_20260806_115703.npz",
    "model_4150_log_20260806_115715.npz",
    "2026-06-20_22-02-53_jrk020_grf13_log_20260806_120142.npz",
    "2026-06-20_22-02-53_jrk020_grf13_log_20260806_120222.npz",
    "2026-06-20_22-02-53_jrk020_grf13_log_20260806_120301.npz",
]

# Which firmware the log was recorded under: "off" = pre-KNEE_DECOUPLE_FF
# (all 2026-08-06 corpus logs), "on" = post-patch. The belt variant must
# model the firmware that actually ran.
LOG_KNEE_FF = {name: "off" for name in CORPUS}

# variant name -> extra replay args. The belt variants get the per-log
# --knee-ff appended automatically (unless the variant pins it).
VARIANTS = {
    "legacy":     ["--actuator", "legacy"],
    "belt":       ["--actuator", "belt"],
    # belt with the legacy hip armature — isolates the drivetrain terms from
    # the reflected-inertia change. 7-log corpus verdict (2026-08-07): pos
    # RMSE within noise of 'belt' (hp identical, kn slightly worse); vel RMSE
    # clearly prefers 8.4e-3. Kept for future ablations.
    "belt_arm42": ["--actuator", "belt", "--hip-armature", "0.0042"],
    # armature-only attribution: legacy model + reflected hip armature
    "legacy_arm": ["--actuator", "legacy", "--hip-armature", "0.0084"],
    # belt without the stale-hip hold (ablation)
    "belt_nostale": ["--actuator", "belt", "--no-stale-hip"],
}


def variant_stem_suffix(vargs: list[str]) -> str:
    """Mirror of the replay's output-stem naming (keep the two in sync)."""
    get = lambda flag: vargs[vargs.index(flag) + 1] if flag in vargs else None
    suffix = "_" + (get("--actuator") or "belt")
    if (get("--actuator") or "belt") == "belt":
        if get("--knee-ff") == "off":
            suffix += "_ffoff"
        if "--no-stale-hip" in vargs:
            suffix += "_nostale"
        if "--hip-reaction" in vargs:
            suffix += "_hiprx"
    arm = get("--hip-armature")
    if arm is not None:
        suffix += f"_arm{float(arm):g}"
    return suffix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=None,
                    help="comma list of NPZ names (default: the 7-log corpus)")
    ap.add_argument("--logs-dir", default=os.path.join(_ROOT, "control"))
    ap.add_argument("--variants", default="legacy,belt")
    ap.add_argument("--delays", default="1,2,3")
    ap.add_argument("--base", default="both")
    ap.add_argument("--no-tf", action="store_true",
                    help="free-run instead of teacher forcing")
    ap.add_argument("--skip-existing", action="store_true",
                    help="reuse an existing result JSON instead of re-running")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="don't run anything; aggregate whatever JSONs exist")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logs = (args.logs.split(",") if args.logs else CORPUS)
    variants = args.variants.split(",")
    for v in variants:
        if v not in VARIANTS:
            sys.exit(f"unknown variant '{v}' (have: {list(VARIANTS)})")
    tf = not args.no_tf

    if not (args.aggregate_only or args.dry_run) and "ISAAC_PATH" not in os.environ:
        sys.exit("run from the activated Isaac Lab env (conda activate "
                 "env_isaaclab) — its activation sets ISAAC_PATH/PYTHONPATH "
                 "that the replay's isaaclab import needs")

    # ── run ───────────────────────────────────────────────────────────
    jobs = []   # (log, variant, json_path)
    for log in logs:
        npz = os.path.join(args.logs_dir, log)
        if not os.path.isfile(npz):
            sys.exit(f"missing log: {npz}")
        for vname in variants:
            vargs = list(VARIANTS[vname])
            if "--actuator" in vargs and vargs[vargs.index("--actuator") + 1] == "belt" \
                    and "--knee-ff" not in vargs:
                vargs += ["--knee-ff", LOG_KNEE_FF.get(log, "off")]
            stem = os.path.splitext(npz)[0] + "_simreplay" + variant_stem_suffix(vargs)
            if tf:
                stem += "_tf"
            jpath = stem + ".json"
            jobs.append((log, vname, jpath))
            if args.aggregate_only or (args.skip_existing and os.path.isfile(jpath)):
                continue
            cmd = [sys.executable, REPLAY, "--npz", npz,
                   "--base", args.base, "--delays", args.delays,
                   "--headless"] + vargs
            if tf:
                cmd.append("--teacher-forcing")
            runlog = os.path.join(args.logs_dir,
                                  f"_sweep_{os.path.splitext(log)[0]}_{vname}.log")
            print(f"[sweep] {log} / {vname}\n        -> {' '.join(cmd)}")
            if args.dry_run:
                continue
            with open(runlog, "w", encoding="utf-8") as fh:
                rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                     cwd=_ROOT)
            if rc != 0:
                print(f"[sweep]   FAILED (rc={rc}) — see {runlog}")
            elif not os.path.isfile(jpath):
                print(f"[sweep]   no JSON produced — see {runlog}")

    if args.dry_run:
        return

    # ── aggregate ─────────────────────────────────────────────────────
    # For each (log, variant): the config cell with min mean pos RMSE.
    import numpy as np
    table = {}   # (log, variant) -> dict(rmse_pos[6], rmse_vel[6], rmse_tau[6], cell)
    for log, vname, jpath in jobs:
        if not os.path.isfile(jpath):
            print(f"[agg] missing {jpath} — skipped")
            continue
        with open(jpath, encoding="utf-8") as fh:
            j = json.load(fh)
        cells = {k: v for k, v in j.items()
                 if isinstance(v, dict) and "rmse_pos" in v}
        cell = min(cells, key=lambda k: float(np.mean(cells[k]["rmse_pos"])))
        table[(log, vname)] = {
            "cell": cell,
            "rmse_pos": np.array(cells[cell]["rmse_pos"]),
            "rmse_vel": np.array(cells[cell]["rmse_vel"]),
            "rmse_tau": np.array(cells[cell].get("rmse_tau", [np.nan] * 6)),
        }

    have_logs = sorted({lg for lg, _ in table})
    print(f"\n{'=' * 100}\nAGGREGATE ({'teacher-forcing' if tf else 'free-run'}, "
          f"best cell per log; {len(have_logs)} logs)\n{'=' * 100}")
    for metric, unit in [("rmse_pos", "rad"), ("rmse_vel", "rad/s"),
                         ("rmse_tau", "Nm")]:
        print(f"\n── {metric} [{unit}] — mean over logs, per joint ──")
        header = f"{'variant':<14}" + "".join(f"{jl:>9}" for jl in JOINT_LABELS) + f"{'mean':>9}"
        print(header)
        base_mean = None
        for vname in variants:
            rows = [table[(lg, vname)][metric] for lg in have_logs
                    if (lg, vname) in table]
            if not rows:
                continue
            m = np.mean(rows, axis=0)
            print(f"{vname:<14}" + "".join(f"{x:>9.4f}" for x in m)
                  + f"{np.mean(m):>9.4f}")
            if vname == "legacy":
                base_mean = m
        if base_mean is not None:
            for vname in variants:
                if vname == "legacy":
                    continue
                rows = [table[(lg, vname)][metric] - table[(lg, "legacy")][metric]
                        for lg in have_logs
                        if (lg, vname) in table and (lg, "legacy") in table]
                if not rows:
                    continue
                dm = np.mean(rows, axis=0)
                dmed = np.median([np.mean(r) for r in rows])
                print(f"{'Δ ' + vname:<14}" + "".join(f"{x:>+9.4f}" for x in dm)
                      + f"{np.mean(dm):>+9.4f}  (median-over-logs mean Δ {dmed:+.4f})")

    # per-log detail for the delta distribution (noise-floor check)
    if "belt" in variants and "legacy" in variants:
        print(f"\n── per-log mean pos RMSE: legacy -> belt ──")
        for lg in have_logs:
            if (lg, "legacy") in table and (lg, "belt") in table:
                a = float(np.mean(table[(lg, "legacy")]["rmse_pos"]))
                b = float(np.mean(table[(lg, "belt")]["rmse_pos"]))
                print(f"  {lg:<58} {a:.4f} -> {b:.4f}  (Δ {b - a:+.4f})"
                      f"   cells {table[(lg, 'legacy')]['cell']} / {table[(lg, 'belt')]['cell']}")

    out = os.path.join(args.logs_dir, "_sweep_aggregate.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({f"{lg}|{vn}": {k: (v.tolist() if hasattr(v, "tolist") else v)
                                  for k, v in row.items()}
                   for (lg, vn), row in table.items()}, fh, indent=2)
    print(f"\naggregate -> {out}")


if __name__ == "__main__":
    main()
