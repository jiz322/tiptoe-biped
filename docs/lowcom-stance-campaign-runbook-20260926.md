# Low-CoM stance campaign ("T-line") — runbook, 2026-09-26

Status: **SUPERSEDED 2026-09-28.** Phase B (T0 + T1) was **killed by the user
on 2026-09-28** at iterations 1040 (T0) and 1190 (T1), after the T1 low-CoM
stance was judged by eye as "gait looked right, but no good" — feet rising too
high and a large roll-side shrug. The decision record D1–D4 and the §10
execution record stand as written; what changed is the verdict on the *recipe*,
not on the stance choice.

**Diagnosis and the replacement plan are in
`docs/tline-marchfix-runbook-20260928.md`** — read that first for T-line state.
Short version: the gait-reference prior is inert (`rew_scale_gait_ref=0.0` in
both arms), the marching is bought by three one-sided foot-height ramps that
have live gradient at T1's own step height and pay nothing for going higher,
the roll shrug is under-priced against `stance_symmetry`, and
`max_hip_roll_rad` is a *termination* bound (not a shaping knob) so it must not
be lowered.

## 0. What changes and why

The S-line policies stand at full leg length: `diagnostics/base_height` reads
0.436 m at idle in S4a, S8 and S8b (spawn height 0.434), knees at the default
0.0. Every zero in the training contract is that straight-leg pose:

| zero-centred on the default pose | where |
|---|---|
| joint-position obs (`joint_pos − default_joint_pos`) | `rl_env_p6.py:1974` |
| action targets (`default + action × scale`) | `balance.py:547`, env equivalent |
| gait reference `A·sinφ` on hip pitch | `rl_env_p6.py:4889` |
| stand pose / curriculum end knot (s = 1.00 row) | `rl_env_cfg_p6.py:611` |
| deploy `DEFAULT_JOINT_POS` / `SPAWN_JOINT_POS` | `control/config.py:138`, `control/balance.py:73` |

The campaign moves that zero to a bent-knee, hip-flexed, torso-vertical
stance. Expected mechanism (not "static stability" — a point-foot biped has
none at any height): shorter capture steps (∝ √h), smaller CoM error per
angular error (h·δ, incl. the IMU-trim bias), and knee leg-length authority
(a straight knee is a singularity for shortening the leg). Cost: continuous
knee-belt load and hip-pitch standing moment. Predictions in §7.

Terminology for the stance, to avoid the torso/thigh-angle ambiguity: leg 1
hip_pitch +a (thigh forward of torso), knee −b (flexed; knee flexion is
NEGATIVE in this asset — see the knot table), leg 2 mirrored (−a, +b).

## 1. Decisions (user)

- **D1 stance depth.** Pick from the balance manifold (§3), on the
  torso-vertical contour. Reference points from the path knot table
  (hip, knee, hip height): (0.312, −0.625, 0.424) / (0.575, −1.25, 0.375) /
  (0.838, −1.875, 0.305). "Knee slightly bent" (b ≈ 0.6–0.9) lowers the hip
  only 1–3 cm; a visible CoM drop (≥5 cm) needs b ≈ 1.2. Recommended first
  pick: b ≈ 0.9 with a from the contour; the second arm, if any, is b ≈ 1.25.
- **D2 physics rate.** 500 Hz, belt series DR OFF (recommended — §2).
- **D3 plant inertia.** S7 armature pins (hip 0.0385 / knee 0.036 / belt rotor
  0.01936, `dr_armature_per_leg=false`, mult 0.80–1.20) as the base. These
  are the values with hardware-ring support (hw hip ring 3.06–3.34 Hz vs the
  vendor plant's 4.88 Hz prediction, memory vendor-rotor-inertia-20260918).
  The vendor plant (S8) is an optional third arm, not the base.
- **D4 experimental design.** Two fresh arms on the two idle 4090s:
  **T0** = old straight nominal, **T1** = new stance. Same code, same rate,
  same recipe; the ONLY diff is the nominal pose block (§4). Without T0 the
  fresh-start-vs-resume change confounds every verdict.

## 2. Physics rate: 500 Hz, belt DR off

Measured cost from checkpoint mtimes (32768 envs, 128 steps/env, same PPO):

| run | physics | s/iter | iters/h |
|---|---|---|---|
| S4a_measDR (dt 0.002, decimation 5) | 500 Hz | 98 | 36.8 |
| S6_restDR | 500 Hz | 117 | 30.7 |
| S8_vendorJ (dt 0.0005, decimation 20, belt DR) | 2000 Hz | 320 | 11.3 |
| S8b_air03 (belt DR) | 2000 Hz | 277 | 13.0 |

A fresh 6000-iteration campaign is ~7 days per arm at 500 Hz and ~20–22
days at 2000 Hz. S8/S8b each managed ~1000 iterations in 3.5 days
(09-18 → 09-22) and ended at `standup_success_high` 0.62 / 0.59 vs S4a's
0.87 (different plant and resume point, so not a clean A/B, but nothing in
sim favoured the belt-DR arms and none was flown).

Why the belt band can be dropped for this campaign:
- The belt series stiffness has NO hardware measurement (the knee push-test
  was never run; memory p12-4-decision-tree). The band 100–1000 N m/rad is
  a guess.
- What a bent-knee stance adds is a STATIC belt deflection under the standing
  knee torque: at 2 N m and 300 N m/rad joint-side that is 0.4°, at
  100 N m/rad 1.1° — a constant encoder-vs-joint offset. That is exactly
  `dr_knee_angle_bias_max` (exists, rate-independent, free). Use it.
- Only the belt's dynamic ringing needs 2000 Hz, and its relevance is the
  unmeasured part. Keep the code; re-enable as a SHORT finishing resume
  (~500 iters at 2000 Hz) only if the push-test lands a k inside the band.

The rate move is not bit-exact and must carry all five together (memory
physics-rate-500hz-belt-limit): `sim.dt = 1/500`, `decimation = 5`,
`rotor_coupling_dt = 1/500` (`rl_env_cfg_p6.py:342`), and in
`belt_knee_actuator.py` `series_dt = 1/500`, `series_substeps = 2` (inert
with the belt off, but set them so a later belt arm cannot inherit a 2000 Hz
step silently). Reference for the 500 Hz plant: S4a's `params/env.yaml`
(dt 0.002, decimation 5). Overrides that go with it:
`env.belt_series_enable=false`, all four `belt_series_*` = 0,
`env.dr_knee_angle_bias_max=0.02` (≈1.1°, covers the static stretch above).

## 3. Pick the stance (desk, GPU 2, under 1 h)

1. Balance manifold over the standing region (Isaac Lab env, prepare and the
   user runs):
   `probe_balance_manifold.py --hip_min 0.2 --hip_max 1.0 --hip_n 9 --knee_min -1.4 --knee_max -0.4 --knee_n 11 --out RL/logs/balance_manifold_stand.csv`
   It returns, per (h, k), the torso pitch θ* that puts the CoM over the
   toe-mid plus CoM height. Take the θ* ≈ 0 contour; that is the set of
   torso-vertical balanced stances. Choose b per D1, read a.
2. Geometry of the chosen pose:
   `probe_pose_dump.py --hold_pose <a,-b,...> --hold 400`.
   Record: root height (→ `spawn_height`, `stand_target_height`), toe z at
   rest, knee z, CoM x relative to toe-mid (must be ≈ 0).
3. Contact-line re-derivation (the 09-12 trap: a knob whose channel never
   fires is not a knob). The toe point `_get_toe_pos_w` is a NAIL origin
   8.6 cm up the shin in the straight stance; its rest height moves when
   the shin tilts. From step 2's toe z at rest:
   `swing_clearance_zero_m = toe_z_rest + 0.003`,
   `no_flight_contact_threshold = toe_z_rest + 0.008` (today 0.084 vs rest
   0.073–0.078), and re-check `feet_air_height_threshold` (S8b's 0.03 arm).
   Re-read `diagnostics/idle_toe_z_p10` at +200 iters and re-apply the
   p10 + 0.003 rule (memory swing-clearance-zero-rule).
4. Viewer check: `run_show_standup_path.ps1 -Smoke` after §4.3, so the new
   end knot rests without penetration (memory standup-path-k0k1-penetration).

## 4. Cfg changes (fresh run; not resume-compatible)

4.1 Nominal pose block (`rl_env_cfg_p6.py:427`): `init_state.pos` z =
    root height from §3.2; `joint_pos` hip_pitch1 = +a, knee1 = −b,
    hip_pitch2 = −a, knee2 = +b. This moves the obs and action zeros at once.
4.2 Height terms: `spawn_height`, `stand_target_height` = new root height;
    `height_min` / `height_max` re-centred (today 0.32 / 0.55 around 0.434;
    keep the same ±0.11 width, so 0.29 / 0.51 for a 0.40 stance).
4.3 Stand-up path: the s = 1.00 knot row becomes the new stance (7 columns:
    s, hip, knee, pitch°, height, rest hip, rest knee). If the stance sits
    at or below an existing row (0.725 / 0.8625), truncate the table there
    and rescale s so the last row is 1.00; keep every row's rest-state
    columns valid (knot-table rule, `docs/standup-path-mode-20260908.md`).
4.4 Gait reference: `rew_scale_gait_ref=0` (its centre is the old zero and
    the anneal plan already ended at 0). T0 also runs it at 0.
4.5 Action range sanity: knee scale 1.025 around −b reaches 0 (full
    extension) for any b ≤ 1.0; hip scale 0.75 around a is fine for a ≤ 0.7.
    For b = 1.25 the knee cannot fully extend — acceptable (it never
    should), note it in the header.
4.6 Deploy mirror of 4.1: `control/config.py` `DEFAULT_JOINT_POS`,
    `control/balance.py` `SPAWN_JOINT_POS`; the HOLD pose before handover is
    this stance, so the belt carries the standing load during HOLD. Zeroing
    (`ZERO_POSE_JOINT_POS`, crawl set_zero) is unchanged.
4.7 Curriculum from scratch: `curriculum_max_alpha=1.0`, advance metric
    `success_high_rel` threshold 0.7, frontier 0.5 / 0.3, kneel start 0.6 —
    the S8 block verbatim.

## 5. Launch recipe

Copy `launch_train_S8_vendorJ.ps1` → `launch_train_T0_straight.ps1` and
`launch_train_T1_lowcom.ps1`. Edits vs S8: drop `--resume` / `--load_run` /
`--checkpoint`; add the D3 pins; the §2 rate/belt block; §3.3 thresholds;
`rew_scale_gait_ref=0`. T1 carries no extra override (the pose lives in the
cfg block, 4.1), so the two arms differ by the cfg block only and the launch
files are diff-identical except run name, GPU line, log names.

Check before launch: the `[PLANT]` startup line prints a rigid belt and the
`[ARMATURE]` line prints `[pinned]` values; `RL/patches/rsl_rl_lr_ceiling_v2.patch`
is applied (desired_kl 0.02 needs the ceiling); `params/env.yaml` shows
dt 0.002 / decimation 5.

Phase A: `--max_iterations 5`, 128 envs, both arms, read the startup lines
and one `diagnostics/base_height`. Phase B: full run, 32768 envs.

## 6. Kill and gate rules

- Kill: `value_function` > 42000 for 50 consecutive iters; PhysX error
  flood; `Train/mean_episode_length` < 400 at +600 (fresh start, so later
  than the S8 +300).
- Gate at +2000: episode_length ≥ 1400 with every DR span live, AND
  `diagnostics/base_height` within ±0.02 of the arm's nominal. T1 drifting
  back up toward 0.43 = the height band is too soft; tighten
  `height_edge_softness` before spending more iterations.
- Compare at equal iterations (+4000 and +6000), deterministic eval only
  (TB clip/jerk are σ-dominated — memory action-jerk-tb-metric).

## 7. Pre-registered predictions (T1 vs T0, same iteration)

- P1 idle `base_height`: T0 ≈ 0.436; T1 = nominal ± 0.02.
- P2 `actual_torque_kn*_mean` higher in T1 (static load), but
  `knee_sat_ema_max` and deterministic knee clip LOWER in T1 (fewer reflex
  peaks). If both are higher, the stance is buying nothing.
- P3 push eval (p13 scenario set, `eval_alpha_success.py` + the R-line push
  eval): T1 survival ≥ T0; catch-step foot displacement smaller in T1.
- P4 idle torso pitch (a `probe_pose_dump`-style dump of a rollout, or
  proj_grav_x in a play log): T1 within ±5° of vertical; T0 shows the
  known forward lean.
- P5 stand-up `standup_success_high` at alpha 1.0: T1 ≥ T0 (the path ends
  earlier).

Verdicts: supported / contradicted / inconclusive, per prediction.

## 8. Hardware (after the sim gate; the user runs all motion)

1. Contract diff, both sides, before any power: default pose (4.6), obs
   history init ('v' blind handover), action scales, trim.
   `policy_contract_check.py` must pass on the new export.
2. HOLD in the new stance for 60 s at the bench: knee motor torque steady,
   no belt tooth skip (crawl-pose check before and after), Kinovea knee
   marker vs encoder = static belt stretch (first hardware number for the
   belt; feeds `dr_knee_angle_bias_max` and the belt-band question).
3. Trim: the +0.089 compensator was fitted at the straight stance and the
   09-20 work says the true IMU bias is −0.013; the lever arm changed, so
   re-bracket (0, +0.045, +0.089) on the first T1 flights, n ≥ 3 each.
4. Flight protocol otherwise per `docs/p11-hardware-runbook.md` §2 gates.

## 9. Open items this runbook does not settle

- Knee thermal: `rew_scale_thermal` is 0; a bent-knee hold is a stall load.
  Decide from the §8.2 bench temperature, not in sim.
- The belt push-test (never run) is the only thing that can re-open 2000 Hz.

## 10. Execution record — 2026-09-26 (appended after running §2–§5)

### Decisions, as executed
D1 stance = `hip_pitch ±0.50 / knee ∓0.90` (leg 2 mirrored) — the runbook's
first pick, b ≈ 0.9. Chosen off the FRESH manifold probe (§3.1 below), not the
09-08 CSV: the two agree on where the θ* ≈ 0 contour sits, but the fresh one is
this asset at this build. D2 500 Hz + belt DR off, D3 S7 armature pins, D4 two
fresh arms — all as written.

Deploy mirror done: `control/config.py` `STANCE_PRESET` / `DEFAULT_JOINT_POS` /
`SPAWN_HEIGHT` and `control/balance.py` `SPAWN_JOINT_POS` / `SPAWN_HEIGHT` now
read one preset table (§4.6). `STANCE_PRESET="straight"` keeps every pre-09-26
policy bit-exact.

### §3.1 manifold (RUN, 4060 Ti, ~1 min)
`RL/logs/balance_manifold_stand.csv`, hip 0.2–1.0 × knee −1.4..−0.4 (9×11).
Torso-vertical contour, hip interpolated per knee row:

| knee | hip @ theta*=0 | root_h | com_h | knee_h |
|---|---|---|---|---|
| −1.40 | 0.777 | 0.388 | 0.314 | 0.189 |
| −1.20 | 0.661 | 0.405 | 0.329 | 0.199 |
| −1.00 | 0.552 | 0.417 | 0.339 | 0.207 |
| −0.90 | 0.498 | 0.422 | 0.344 | 0.210 |
| −0.70 | 0.377 | 0.430 | 0.350 | 0.216 |
| −0.40 | 0.210 | 0.432 | 0.353 | 0.221 |

The exact grid cell (0.5, −0.9) reads theta* = −0.1°, com_x_b 0.6 mm, root_h
0.422. `root_h` here is the height above the ball bottom in the analytically
rotated frame; the number the env actually spawns at is §3.2's.

### §3.2 geometry of the chosen pose (RUN, 4060 Ti)
`RL/logs/stance_probe/pose_*.log`, d = 0, two holds:
* spawn (t=0): root **0.430 m**, joints exactly [0, 0.5, −0.9, 0, −0.5, 0.9],
  CoM 0.018 m ahead of toe-mid, nail-toe world z 0.215 (spawn pins the nail at
  `_FK_TARGET_FOOT_Z = 0.0856` above the root, in ANY pose).
* held 400 steps at the stance target: root **0.344 m**, pitch **+17.6°**,
  knee sags to −1.25. The straight stance does the SAME thing (0.405 m /
  +9.4°, knee 0.02) — this is the known idle forward lean (memory
  trained-forward-lean-idle), amplified by the bent knee, NOT a broken pose.
  It is a zero-action PD hold at frozen grace, which is not what the reward or
  a trained policy produces; do not read the 0.344 as a plant finding.

### §3.3 contact lines (the 09-12 trap, applied)
The env's "toe" is the nail-frame point (`_toe1_local_offset`), 8.6 cm up the
shin; `_FK_TARGET_FOOT_Z` pins it at 0.0856 at SPAWN regardless of pose, and the
reward uses its world z. Loaded rest in the new stance = **0.068 m** (straight:
0.079). Applied, by the same rule the straight values encode:
`swing_clearance_zero_m = 0.068 + 0.003 = 0.071`,
`no_flight_contact_threshold = 0.068 + 0.008 = 0.076`. **PROVISIONAL** — re-read
`diagnostics/idle_toe_z_p10` at +200 iters and re-apply the p10 + 0.003 rule
(memory swing-clearance-zero-rule). `feet_air_height_threshold` left at 0.05
(S8b's 0.03 arm is a separate question §9 does not settle).

### §3.4 viewer check (RUN, 4060 Ti)
`run_show_standup_path.ps1 -Smoke -Preset lowcom_b090` (new `-Preset` flag; the
viewer now mirrors `apply_stance_preset`). Result vs the STRAIGHT table run the
same day: **the numbers are identical where the poses are identical.** K0–K9
(seiza→squat) read the same Δz / Δpitch / |v| / q-err to the last digit, and
K10–K12 fall in BOTH tables. Two consequences: (a) the new table's leading rows
are the old poses, so no new penetration was introduced; (b) `K10/K11/K12 FELL`
is **pre-existing legacy behaviour, not a T1 regression** — the viewer's
[kp 200 / kd 10 Nm/rad] HOLD is far softer than training's joint-side servo, and
those rows are the dynamic squat-to-stand segment the policy is meant to rescue
(memory standup-ramp-topple-finding). No action taken.

### §4/§5 build, as executed
* New `RlEnvCfg.stance_preset` + `STANCE_PRESETS` + `apply_stance_preset()`
  (rl_env_cfg_p6.py), called FIRST in `RlEnv.__init__` so it reaches
  train/play/eval alike; prints one `[STANCE]` line per build. "straight" is a
  no-op. This is the T0/T1 diff: **one override line.**
* §2 rate: `sim.dt 1/500`, `decimation 5`, `rotor_coupling_dt 1/500`,
  `hold_decimation 5`, `series_dt 1/500` ×2 — all five moved together, plus
  `belt_series_enable=false` and the four bands at 0. Verified in
  `params/env.yaml` of the Phase A runs, not assumed.
* §4.4 `rew_scale_gait_ref=0` on BOTH arms (T0 included, as written).
* Launches: `RL/launch_train_T0_straight.ps1` (GPU 0) and
  `RL/launch_train_T1_lowcom.ps1` (GPU 1), both with `-PhaseA`. Substantive diff
  between them = GPU line + run name + the preset line, nothing else.
* The straight table's own values were NOT changed (T0 stays bit-exact).

### §6 Phase A gate — PASSED, both arms (5 iters, 128 envs)
From TB, not stdout (`data/_tb_read.py`):

| metric | T0 straight | T1 lowcom |
|---|---|---|
| base_height | 0.4012 | 0.3915 |
| base_height_std | 0.0377 | 0.0365 |
| knee1 torque mean / max | 10.01 / 27.57 | 9.43 / 28.27 |
| knee2 torque mean / max | 9.57 / 25.17 | 9.47 / 27.77 |
| knee_sat_ema_max | 0.468 | 0.442 |
| mean_episode_length | 48.95 | 55.37 |

Startup lines on the final code: `[STANCE] preset=lowcom_b090 root_z=0.430 ...
height=[0.305, 0.525] swing_zero=0.071 contact_thr=0.076 path_rows=13`;
`[PLANT] belt series: live=False ... I_rotor=0.01936`;
`[ARMATURE] knee = 36.000e-3 [pinned] hip_pitch = 38.500e-3 [pinned]`.

**P1 read at Phase A (DIRECTIONAL ONLY — 5 iters of a fresh random policy; §7's
verdicts need equal-iteration comparisons much later):** T1 idle height 0.3915
vs T0 0.4012, i.e. T1 sits ~1 cm low, not the ~6 cm the nominal would suggest,
and neither arm is within ±0.02 of its nominal yet. Both are early in the
curriculum (alpha 0, mostly kneel/seiza resets), which is why. P2's direction is
already as predicted (T1 `knee_sat_ema_max` lower, 0.442 vs 0.468) but the
torque means are within noise. **No verdict is claimed.**

### §7 predictions — status
P1–P5 UNRESOLVED. First hard reading at +2000 via the §6 gate; verdicts at
+4000 / +6000 on deterministic eval only.

### Deviations from the written procedure (all small, all listed)
1. §3 ran on the 4060 Ti rather than "desk" — the probe is pure geometry and the
   two 4090s were the point.
2. The knot table's `root_h` and the preset `root_z` come from the MEASURED FK
   spawn height (0.430), not the manifold's rotated-frame number (0.422) nor the
   runbook's illustrative 0.40. Height band [0.305, 0.525] = ±0.11 re-centred on
   0.415, which is what §4.2's "keep the same ±0.11 width" yields once the
   stance's own settle height is allowed for.
3. T1's `standup_path_knots`: rows above the old (0.575, −1.25) point truncated,
   s rescaled so the last row is 1.00 at the new stance; the knees-down/balls
   boundary moved from s 0.45 to 0.561 (d 0.55 → 0.44), path speed unchanged at
   ~0.22 s per rad of knee. Every surviving row keeps its rest columns valid
   (knot-table rule).
4. §4.5 handled by note rather than override: knee scale 1.025 around b = 0.90
   still reaches full extension, so no clip at the stance.

### Open, per §9 (unchanged by this run)
Knee thermal stall load — decide from the §8.2 bench temperature, not sim. The
belt push-test is still the only thing that can re-open 2000 Hz.

### Hardware (§8) — NOT STARTED
Desk contract diff, HOLD bench, trim re-bracket all await the sim gate; the user
runs every motion command.
