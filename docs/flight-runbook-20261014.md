# Flight runbook — 2026-10-14 (S10b / S11 / S12, heading hold)

Prep record and evidence: `data/flight_20261014/` (`_tb_pick.json`, `_stall_*.json`, dry logs).
Previous flight: `docs/s11-flight-analysis-20261012.md` (its §12 is the to-do list closed here).

## 1. What changed since 10-11 (all on the Pi, md5-checked)

| item | change | verified |
|---|---|---|
| re-arm stall (B3) | **root cause found**: the logger's local alias `lb` kept the previous run's log alive past `flush_log`; the next run's tick 0 rebound it and freed ~35 × N arrays **after** `send_target`. Fixed (`lb = None` after each logged tick). Also GC frozen+disabled for the session (collect only on the way to HOLD) and `v`/`n` now arm, with the run starting on the next tick | Pi `--dry`, B2→B3 sequence (38 s run, sit, re-arm): re-arm first tick **39.9 ms → 1.3 ms**. GC-only fix left it at 39.6 ms, so **GC hypothesis contradicted**. Fresh first tick 9.8 → 2.0 ms. Max tick in the run 2.65 ms |
| heading | `--heading hold` is now the **default**. Guard: if \|target − yaw\| > 60° while commands are active, the reference follows yaw (= the old `off`, event `heading_trip`, display shows TRIPPED). `a`/`d` nudges are clamped to 45° from the current yaw. `z` re-engages hold | `s11_contract.test_episode` (reversed-sign trip case added) PASS; dry runs PASS |
| why the guard is enough | with a reversed yaw sign, a 15° nudge grows at ≤ 0.4 rad/s (the trained turn cap) and trips at 60° in ~2 s. The trip sets the error to 0, which is the env's own zero-Δψ re-anchor, so it is in distribution | — |
| thermal | display line THERMAL hip_roll: yellow ≥ 40 °C (cool before the next trial), red ≥ 60 °C (sit now); run seconds + max tick (red > 15 ms) in the banner | dry |
| log save | now prints its duration: 66 ms for a 38 s log (motor CAN watchdog 200 ms) | Pi dry |
| new log channels | `heading_follow`; meta `heading_trip_rad`, `heading_nudge_max_rad`, `ring_unused` | — |

## 2. Policies (all through `balance_s11.py`, same contract guard)

| file | source | note |
|---|---|---|
| `s11_11880.ts.pt` | S11_hist 11880 | flown 10-11, reference |
| `s12_13010.ts.pt` | S12_steplen 13010 (run still training) | step-length error 0.048 → 0.026 m since fork; **vx_err_moving worsened 0.09 → 0.12** |
| `s10b_12179.ts.pt` | S10b_heading final 12179, zero-column surgery → hist layout, exported `--ring-unused` | action bit-identical to S10b (surgery Δ 0, ring reversed/random Δ 0). `sup` heads are untrained, so the ESTIMATE line is display-only noise. S10b trained **without** pg-bias DR (0.0 vs S11's 0.05) |

TB (last 50 iters): stand-up success_high 0.815 / 0.823 / 0.812, sit 0.98 / 0.99 / 0.98, idle drift 0.26 / 0.27 / 0.29 m
(S10b / S11 / S12). This matches the expectation that they don't differ much in sim.

## 3. Bench checks (before trial 1)

1. Crawl-pose tooth check (`test_position.py`), as usual.
2. **R_knee CAN connector**: reseat/inspect (lost replies in B3 after the fall).
3. **Supine/prone pg_x** (closes the sign-convention question, 10-12 doc §10): run the script, stay limp in HOLD,
   lay the robot flat on its **back** and read the IMU line's `raw x`, then on its **front**. Prediction:
   back ≈ −1, front ≈ +1. Report both numbers.
4. YAW CHECK line: limp, twist the robot about vertical by ~30° within 0.5 s. Read the verdict and note it.
   It's informational only, because the trip guard covers a reversed sign.
5. hip_roll temperature < 40 °C (THERMAL line).

## 4. Command (one fresh process per trial; the policy swap needs one anyway)

```
cd ~/Documents/big_thigh/control
sudo ~/miniforge3/bin/python3 balance_s11.py --policy s11_11880.ts.pt --alpha 0.6 --pg-offset -0.015
```
Defaults now: `--grace-seconds 2.5`, `--heading hold`, `--heading-trip-deg 60`. Swap `--policy` for
`s12_13010.ts.pt` / `s10b_12179.ts.pt`. All three guards should print PASS.

**Order (ABCCBA, balances heat and battery): s11 → s12 → s10b → s10b → s12 → s11.** Cool to < 40 °C between trials.

## 5. Per-trial script (~45 s of stepping, under the 60 s thermal cap)

| t after `cmd_active` | key | purpose |
|---|---|---|
| 0–10 s | — | idle: heading hold + drift |
| 10 s | `a` | +15°, wait 5 s |
| 15 s | `d` `d` | −15° net, wait 5 s |
| 20 s | `a` | back to 0°, wait 5 s |
| 25 s | `w` `w` | vx 0.10 for 8 s |
| 33 s | `x` `x` | vx 0, 5 s |
| 38 s | `s` | sit |

If the display shows **TRIPPED**, say "trip" aloud, press `z` once, and skip the remaining heading keys in that trial.
Tether slack; say "pull" aloud if it goes taut. LED in frame; floor tape every 0.25 m if possible.

## 6. Pre-registered checks for the analysis

| id | supported | contradicted |
|---|---|---|
| F1 re-arm fix | every trial `time[0]` ≤ 10 ms and no tick > 15 ms in grace | any trial violates it |
| H1 heading hold | idle window \|net heading\| ≤ 15° with no trip; after each 15° nudge, \|err\| < 5° within 5 s | a trip, or the error grows past 30° after a nudge (= reversed sign or no turning authority). Inconclusive if "pull" was called |
| H2 policies equivalent on hardware (user's prior) | no policy differs from another by 2 trials in stand-ups or uncommanded falls (n = 2 each) | a 2-vs-0 split in either |
| H3 supine sign | back ≈ −1 | back ≈ +1 → the 09-30 / 10-03 fore/aft labels were right and the 10-12 relabel is wrong |

## 7. After the flight

Copy `~/Documents/big_thigh/control/*_s11_log_20261014_*.npz` into `data/flight_20261014/` and add per-trial notes
to `data/flight_20261014/README.txt`, as on 10-11. On the Pi, `balance_s11.py` and `s11_contract.py` differ from
commit c53e7aa (they were copied over by scp). Commit from Windows, then `git checkout -- control/` on the Pi before
the next pull.
