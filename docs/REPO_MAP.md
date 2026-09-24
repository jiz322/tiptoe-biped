# Repo map

Where things are, what is live and what is history, and the gotchas that
recur. Derived from the private repo's working notes on 2026-09-24.

## Directories

| Path | Status | What |
|---|---|---|
| `README.md` | live | Start here. |
| `docs/` | live | Campaign reports and runbooks, dated. `RETROSPECTIVE-2026-09.md` first. Index in the README. |
| `hardware/` | live | BOM with paid prices, assembly notes, pointers to the CAD on MakerWorld and the Release page. |
| `assets/tiptoe_v27/` | live | The robot asset training loads: `robot.urdf` (Onshape export, mass-corrected 2026-09-18, meshes relative), `meshes/*.stl`, `robot/robot.usd` (+ `configuration/`), `config.json` (the onshape-to-robot export config with joint limits and efforts). |
| `RL/` | live | Isaac Lab task, training and eval scripts, launchers, rsl_rl patch. See `RL/README.md`. |
| `control/` | live | Deploy stack on the Raspberry Pi 5, the released checkpoint `r1r_3150.pt`, its TorchScript export `exported/policy.pt`, motor firmware dumps. See `control/README.md`. |
| `data/` | live | Analysis scripts + per-session READMEs. The recordings themselves are Release assets (`data/README.md`). |
| `script/` | live | Isaac Sim pose viewers `show_{crawl,kneel,standup_path}.py` with their `run_show_*.ps1` launchers. |
| `analysis_motor_filters/` | concluded | 2026-08-14 motor telemetry forensics that established the `torque_fdb` filter and dead-band artefacts (`results*.txt`). |
| `reports/belt_coupling_sim2real/` | concluded | 2026-08-07 evaluation of the belt-knee actuator model by teacher-forcing replay of 7 hardware logs, with figures. |
| `legacy/` | history | Every launch script, curriculum phases p1–p5, CAN-over-USB scripts, pre-RL controllers. Read `legacy/README.md`. |

Not shipped: the other 22 policy checkpoints of the campaign, ~120 one-off
analysis probes (`data/_*.py`), training logs and TensorBoard runs, CAD
sources (see `hardware/README.md`), vendor motor documentation (get it from
RobStride).

## The robot in numbers

- Joints per leg: hip_roll (EL05, 6 Nm), hip_pitch (RS01, 17 Nm), knee (RS01
  through a 22:30 belt: 23.2 Nm joint-side). Sim joint order:
  `hip_roll1, hip_pitch1, knee1, hip_roll2, hip_pitch2, knee2` (1 = left).
- Motor ids 101/104 hip pitch, 102/105 knee, 103/106 hip roll.
- Belt coupling: `q_knee = (22/30)·θ_motor − 0.733·q_hip_pitch`. Motor-side
  kp/kd = joint-side × (22/30)².
- Control 100 Hz on 500 Hz physics (2000 Hz from the 09-16 S7 run on;
  decimation follows). Observation 36 × 5-frame history = 180; 6 actions
  (PD position targets).
- Deploy gains `REAL_KP [6, 20, 30]`, `REAL_KD [0.3, 1.4, 1.5]` joint-side;
  action low-pass α 0.60; `KNEE_DECOUPLE_FF = True`; `IMU_PG_OFFSET = [+0.089, 0]`.
- Mass 5.57 kg in flight configuration (URDF 5.195 kg); spawn height 0.4401 m.

## Gotchas that recur

- **Resume re-passes every override.** `train.py --resume` reads cfg defaults
  for anything not on the command line. Launch scripts are complete recipes.
- **The rsl_rl patch is not optional.** `RL/patches/`.
- **The knee feed-forward flag must match on both sides** (`KNEE_DECOUPLE_FF`
  in `control/config.py` and the training cfg). It silently reverted once.
- **USD DriveAPI angular stiffness is per degree.** Isaac Sim writes
  kp_rad·π/180; the old viewers' `JOINT_STIFFNESS = 40` is ~2300 Nm/rad, not
  training's 8.
- **The toe point is not the contact.** `_get_toe_pos_w` returns a point 8.6 cm
  up the shin in standing; the 2 cm ball's contact is 5.8 cm beyond it. Never
  use the toe point for a CoM-margin argument.
- **Trim decode.** Displayed `pg_x = raw − IMU_PG_OFFSET[0]`; the offset
  changed twice on 09-03. `data/README.md` has the per-session rule. From the
  09-10 session on, `proj_grav_raw` and `imu_pg_offset` are in the npz.
- **Channel trust.** `sim_joint_pos` in hardware npz is motor-encoder-derived
  through the belt decode: belt slip is invisible to it. IMU lateral signs were
  mirrored until 08-28. `torque_fdb` is firmware-filtered. Every channel needs
  one independent physical check before it anchors an argument.
- **Known data bug.** Sim npz written by `play_log.py` / `play_sim2real.py`
  before 08-31 have the knee `target` channel ×2.
- **Pad contact model is retired** (2026-09-17): its reference plane sat below
  the floor, so it never applied a force. `pad_contact.py` and
  `docs/pad-contact-dr-20260912.md` are kept as the record. Foot compliance,
  if wanted, goes through the PhysX material in `foot_contact.py`.
- **`docs/p12-4-bench-runbook.md`** still carries the retracted 11/14 Hz tap
  numbers and the 6× armature claim; `docs/p12-4-session-20260903.md` and
  `docs/armature-derivation-note-20260918.md` are the corrections.
- **Old comments in `control/config.py`** above `IMU_PG_OFFSET` describe the
  −0.089 derivation; the value is +0.089. Left as-is because the file was
  deployed at a pinned commit; the retrospective §4 lists these.

## Sim2real deployment checklist

Every new policy deployment, and every sim reproduction of a hardware trial,
verifies these by diffing code and data, not by assumption:

1. Reset / handover observation contract on both sides (training starts blind:
   obs history zeroed; deploy `v` matches that, `b` does not).
2. IMU axis signs and Euler/quaternion convention (`docs/imu-sign-check.md`).
3. Units at every adapter boundary (deg vs rad, deg/s vs rad/s).
4. Joint ordering and left/right mirror signs.
5. Timestamp alignment between video, npz and sim: anchor on physical events.
6. Latency measured with the same metric on both sides (closed-loop PD lag
   ~40–50 ms is not transport delay; sim shows the same number).
7. Torque limits and frames (motor vs joint, knee ×22/30). Policies routinely
   command 4–6× deliverable knee torque in sim and survive the clip; that is
   normal, not a fault.
8. Trim state: log and check the raw projected gravity alongside the displayed one.
9. One independent physical verification per logged channel.

## Conventions

"Trial" = a real hardware recording; "rollout" = a sim episode. Describe
motion as CoM acceleration plus joint-angle changes rather than "steps".
Frame-by-frame video observations are ground truth; data channels are
interpreted against them. Every analysis states a pre-registered hypothesis
and returns supported / contradicted / inconclusive with the discriminating
signal named.
