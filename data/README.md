# Hardware recordings (sim2real dataset)

Every flight and bench session of the campaign was recorded by
`control/balance_log.py` at 100 Hz into one `.npz` per trial. The recordings
are **not in git**. They are published as one zip per session folder on the
GitHub Release page (`tiptoe-dataset-<folder>.zip`), licensed CC BY 4.0. Unzip
them into this folder to make the paths in the docs and analysis scripts
resolve.

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
| `r1r_flight_20260906` | 09-06 | R1r_3150 | 3 | +0.089 | v | fixed | Companion trials to the S4 test. |
| `s4_flight_20260910` | 09-10 | S4a_6520, S4b_6520, R1r_3150 | 11 | +0.089 / 0 (per file) | v | fixed | S-line checkpoints vs R1r; three tooth skips logged. README.txt in folder; `sysid_targets.json` alongside. |

The last-listed trial in the 08-27, 08-28 and 08-31 sessions started with the
robot held ("pickup"); all others start on the ground. The per-folder
`README.txt` files (where present) are the contracts written on the day.

## Trim decode rule

The policy sees `pg_displayed = pg_raw − IMU_PG_OFFSET`. Recordings before
09-10 log `proj_grav` **as displayed**, so:

| Folder | raw `pg_x` = |
|---|---|
| everything up to `p12_3_flight_20260831_v` | `logged − 0.089` |
| `p12_4_flight_20260903_v_notrim` | `logged` |
| `p12_4_flight_20260903_v_postrim`, `r1r_flight_20260906` | `logged + 0.089` |
| `s4_flight_20260910` | `logged + imu_pg_offset` (stored in the npz), and `proj_grav_raw` is logged directly |

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

Load one with:

```python
import numpy as np
d = np.load("data/p12_4_flight_20260903_v_postrim/r1r_3150_log_20260903_134700.npz", allow_pickle=True)
t, pg = d["time"], d["proj_grav"]
pitch_deg = np.degrees(np.arctan2(pg[:, 0] + 0.089, -pg[:, 2]))   # raw pitch, + = forward
```

`balance_forensics.py` and `imu_offset_from_log.py` in this folder are the
two general-purpose readers; the rest are the fits behind specific docs
(coast-down, kp sweep, chirp, firmware PD replica, sysid replay).

## Bench recordings

The bench sessions (coast-down, kp sweep, chirp, saturation, motor-filter
forensics) are summarised in `docs/replant-report-20260817.html`,
`reports/belt_coupling_sim2real/` and `analysis_motor_filters/`. Their raw
files are not part of the v0.1 release; ask in Discussions if you need one.
