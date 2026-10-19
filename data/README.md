# Hardware recordings (sim2real dataset)

Every flight and bench session of the campaign was recorded by
`control/balance_log.py` (and, from 10-03, its stand-up and S11 variants) at 100 Hz
into one `.npz` per trial. The recordings are **not in git**. The first nine sessions,
2026-08-24 to 09-10 (69 trials), are published as one zip per session folder on the
[GitHub Release page](https://github.com/jiz322/tiptoe-biped/releases/tag/v0.1.0)
(`tiptoe-dataset-<folder>.zip`), licensed CC BY 4.0. Unzip them into this folder to
make the paths in the docs and analysis scripts resolve.

The six sessions from 09-30 on (27 flight trials) are documented here but not in a
release yet: each folder holds the `README.txt` contract written on the day, the
analysis scripts that produced its doc, and the figures the doc shows. Their npz are
with the owner.

The analysis scripts in this folder (`*.py`) need only numpy, scipy and
matplotlib. They read npz from the session folders by relative path.

## Sessions

Dates are 2026. "trim" is `IMU_PG_OFFSET[0]`, the pitch trim the policy saw
(see the decode rule below). Handover: `b` = sighted (the real IMU frame
primed ×5), `v` = blind (history zeroed, the training contract). IMU signs:
the lateral sign set was mirrored until the 08-28 fix (retrospective #12).

| Folder | Date | Policy | Trials | Trim | Handover | IMU signs | Notes |
|---|---|---|---|---|---|---|---|
| `p11_flight_20260824` | 08-24 | K4b_3600 | 12 | −0.089 | b | old (mirrored) | First flight of the K line. Every trial fell at the first catch-step. |
| `p12_flight_20260825` | 08-25 | K4b_3600 ×7, L1_4330 ×5 | 12 | −0.089 | b | old | The day the right-knee tooth skip was first seen on video. |
| `p12_2_flight_20260827` | 08-27 | K4b_3600 | 10 | −0.089 | b | old | Per-trial outcomes not summarised. README.txt in folder. |
| `p12_2_flight_20260828_6pm` | 08-28 | R1r_3150 | 8 | −0.089 | b | **unverified** (fix committed that afternoon) | Baseline for the handover A/B. 7/7 backward falls. |
| `p12_3_flight_20260831_v` | 08-31 | R1r_3150 | 6 | −0.089 | v | fixed | Blind handover A/B: time to 25° tilt 0.34 → 0.67 s. Predictions in `docs/PREDICTIONS_p12_3.md`. |
| `p12_4_flight_20260903_v_notrim` | 09-03 | R1r_3150 | 6 | 0 | v | fixed | Trim bracket, middle arm. 6/6 backward. |
| `p12_4_flight_20260903_v_postrim` | 09-03 | R1r_3150 | 1 | +0.089 | v | fixed | **The 6.06 s stand.** Still upright at log stop. |
| `r1r_flight_20260906` | 09-06 | R1r_3150 | 3 | +0.089 | v | fixed | Same configuration as the 09-03 stand. 1.67 s fall; **5.23 s and 8.79 s still upright at log stop** (the 8.79 s trial drifted into an obstacle). Companion trials to the S4 test. |
| `s4_flight_20260910` | 09-10 | S4a_6520, S4b_6520, R1r_3150 | 11 | +0.089 / 0 (per file) | v | fixed | S-line checkpoints vs R1r; three tooth skips logged. README.txt in folder; `sysid_targets.json` alongside. |
| `t0_s8b_flight_20260930` | 09-30 | S8b_8880 ×2, T0_1020 ×2 | 4 | −0.015 | v | fixed | **First sustained stepping: S8b 8.42 s and 8.72 s** at the physically measured trim; T0 (heavy-armature pins) 1.71 s and 2.64 s. Both S8b runs end pitched forward, one after recovering from −0.63 rad. |
| `s8b_standup_flight_20261003` | 10-03 | S8b_8880 | 6 | −0.015 | v | fixed | First stand-up from d = 0.6: **0/6**. Tooth skips in 5 of 6 (crawl check); the HOLD pose revealed the knee's reaction on the hip. |
| `s11_flight_20261011` | 10-11 | S11_11880 (TorchScript) | 5 | −0.015 | v | fixed | **3/5 stand-ups; A1 steps 18.9 s, B2 43.6 s, both ended by a commanded sit.** The two failures are the same-process re-arm trials (A2, B3). No tooth skips. |
| `flight_20261014` | 10-14 | s11_11880 ×8, s10b_12179, s12_13010 | 10 | −0.015 | v | fixed | **0/10**, all backward. A sustained yaw after 0.5 s separates every trial from 10-11; the 10-15 inspection found loose screws and play in a hip_roll housing. |
| `flight_20261016` | 10-16 | s11_11880 | 2 (+2 not copied) | −0.015 | v | fixed | Both fall right with R_knee belt skips. Disassembly: 3 of 7 steel cords snapped; a new belt broke the same way in two more trials. |

The last-listed trial in the 08-27, 08-28 and 08-31 sessions started with the
robot held ("pickup"); all others start on the ground. The per-folder
`README.txt` files are the contracts written on the day; the ones from 09-30 on
also carry the user's floor notes and the open questions at the time.

## Trim decode rule

The policy sees `pg_displayed = pg_raw − IMU_PG_OFFSET`. Recordings before
09-10 log `proj_grav` **as displayed**, so:

| Folder | raw `pg_x` = |
|---|---|
| everything up to `p12_3_flight_20260831_v` | `logged − 0.089` |
| `p12_4_flight_20260903_v_notrim` | `logged` |
| `p12_4_flight_20260903_v_postrim`, `r1r_flight_20260906` | `logged + 0.089` |
| `s4_flight_20260910` and everything after it | `logged + imu_pg_offset` (stored in the npz), and `proj_grav_raw` is logged directly |

From 09-30 on every session flies at trim −0.015, the measured bias. The +0.089 was
the September plant-gap compensator and is retired (retrospective II §0).

## npz schema

From `p12_4_flight_20260903_v_postrim/r1r_3150_log_20260903_134700.npz`
(603 frames, 6.06 s). Joint order is the sim order: `hip_roll1, hip_pitch1,
knee1, hip_roll2, hip_pitch2, knee2` (1 = left). Joint angles are joint-side
radians derived from the motor encoders through the 22:30 belt decode: belt
slip is invisible here (retrospective, checklist item 9).

| key | shape | meaning |
|---|---|---|
| `policy`, `control_hz`, `handover` | scalars | checkpoint name, 100, `blind` or `sighted` |
| `time` | (T,) | seconds since handover |
| `obs` | (T, 36) | one observation frame as fed to the policy, before normalisation (the policy stacks 5) |
| `action` | (T, 6) | raw policy output |
| `target`, `filtered_target` | (T, 6) | joint targets before and after the action low-pass filter (α 0.6) |
| `sim_joint_pos`, `sim_joint_vel` | (T, 6) | joint state in sim frames |
| `motor_tau_cmd`, `motor_tau_meas`, `motor_temp` | (T, 6) | motor-side torque command, measured torque (firmware-filtered), temperature |
| `proj_grav`, `ang_vel` | (T, 3) | IMU-derived projected gravity (as displayed to the policy) and body angular rate |
| `gait_phase` | (T,) | clock input |
| `mode` | (T,) | `HOLD`, `OBSERVE`, `NN_CONTROL` |
| `can_replies`, `tick_busy_ms` | (T,) | motors that replied this tick (6 = all), loop busy time |
| `kp`, `kd`, `tau_max`, `lpf_alpha` | (T, 6) / (T,) | gains and limits in effect |

Later sessions add keys; the folder's `README.txt` lists them. The 09-30 and 10-03
sessions add the as-run deploy scripts themselves, the 10-03 stand-up adds
`grace_remaining`, `path_s`, `hold_target`, `action_center`, `filtered_action`, and the
10-11 S11 session adds the history contract (`hist_frame`, `ring_first_tick`,
`contract_json`, `pre_*`), all with their meta keys.

Load one with:

```python
import numpy as np
d = np.load("data/p12_4_flight_20260903_v_postrim/r1r_3150_log_20260903_134700.npz", allow_pickle=True)
t, pg = d["time"], d["proj_grav"]
pitch_deg = np.degrees(np.arctan2(pg[:, 0] + 0.089, -pg[:, 2]))   # raw pitch, + = forward
```

`balance_forensics.py` and `imu_offset_from_log.py` in this folder are the
two general-purpose readers; the rest are the fits behind specific docs
(coast-down, kp sweep, chirp, firmware PD replica, sysid replay). The `_`-prefixed
scripts in the session folders are the per-flight analyses behind the docs of the
same date.

## Bench recordings

The bench sessions (coast-down, kp sweep, chirp, saturation, motor-filter
forensics) are summarised in `docs/replant-report-20260817.html`,
`reports/belt_coupling_sim2real/` and `analysis_motor_filters/`. Their raw
files are not part of the v0.1 release; ask in Discussions if you need one.

`s11_pi_latency_20261010/` is the one bench session included here: the timed
forward pass that decided the S11 history actor fits the 100 Hz loop (p50 0.70 ms
per tick, no missed deadline), with its raw output.
