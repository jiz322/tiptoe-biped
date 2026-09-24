# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# S-line arm B, RELAUNCH #5 — "S4b_measDR_pg05" — GPU 1 (RTX 4090 D, by UUID), 2026-09-09.
#
# ARM B = ARM A (launch_train_S4a_measDR.ps1, header below is A's, kept verbatim
# for the shared rationale) + ONE change: env.dr_proj_grav_bias_max 0.13 → 0.05.
#   * 0.05 = ±2.9° of unknowable pitch. Measured IMU mounting offset −0.032 ±0.03
#     (crawl-derived, replant P6) sits inside it; the old 0.10/0.13 bands were sized
#     to contain the +0.089 DEPLOY TRIM (obs bias ≈ +0.12 at physical upright),
#     i.e. to absorb a plant-gap compensator in the observation channel. The
#     S-line's coherent dr_com_shift now models that gap on the PLANT side, so the
#     obs band can be honest again.
#   * dr_impact says |pg_bias| is the 3rd largest ep-len tax (−171) and the only
#     one that directly corrupts the stand-up TRACK reward's input (true pitch
#     rewarded, biased pitch observed → irreducible tracking error).
#   * DEPLOY CONTRACT if B flies: IMU_PG_OFFSET must be set so RAW pg_x at physical
#     upright is within ±0.05 (true mounting, NOT the +0.089 compensator). Check
#     with the supine/crawl pose before the first handover. A B-policy flown under
#     the +0.089 trim is out-of-distribution by design.
# PREDICTIONS specific to B (vs A, same wall clock):
#   PB1 standup/pitch_err_grace_mean falls below 0.25 rad within 300 iters (A stays ≥0.30).
#   PB2 B reaches advance_metric 0.7 (alpha 0.6) ≥ 150 iters before A.
#   PB3 quasi-static response map on B shows pitch dependence at wy=0 (A flat)
#       — the dead-zone-was-trained test in the shared header.
#   B ties A on PB1/PB2 → pg-bias band is not the limiter; keep A (deploy-compatible).
#   B beats A → the trim was compensating a training artefact; B is the flight pick
#       AND the trim must be recalibrated (contract above).
#
#
# WHY (S3a/S3b post-mortem, TB 2026-09-08_20-52-15_S3{a,b}_path, both killed 09-09 eve):
#   * S3a ran 950 iters (5230→6180, ~24 h) at alpha 0.5 and never advanced:
#     curriculum/standup_success_high 0.01→0.26 (peak @6090), success_low
#     0.52→0.68, advance_metric (high/low) peaked 0.39 vs the 0.7 bar, slope
#     ≈ +2.5 pts/100 iters and flattening. standup/pitch_err_grace_mean did NOT
#     shrink (0.30→0.33 rad). At 6100→6120 a value-loss break (6.5k→16k),
#     entropy 2.49→2.98, sigma 0.387→0.418, ep len 838→650 — a regression
#     wobble (raw_action_max stayed ≈4.4, not the K4 runaway). Best ckpt =
#     model_6090.
#   * S3b (arm B extras: com_shift 0.03, knee_angle_bias 0.15, handover_prime
#     0.5, lean 0.20) was strictly worse on every channel: success_high 0.035,
#     success_low 0.49, population ep len 376 vs 699. The B bundle costs ~0.15
#     success even in the EASY band → arm B is dropped; the S3 A/B is closed.
#   * THE CEILING: the stand branch (d=0, 60 % of envs) survives only ~700–850
#     of 3000 steps under the R1 "robust" DR bundle, and the easy kneel band
#     (d≈0.1, near-standing spawn) succeeds only 0.64–0.68. The stand-up skill
#     is being learned on top of a plant the policy cannot even stand on
#     reliably. dr_impact tertile contrasts (ep-len, high third − low third,
#     S3a last 100 iters; DR draws are independent of d so these are
#     randomized causal contrasts on a mean ep len of 704):
#         action_delay        −327   (delay 3 vs 1 step = 30 vs 10 ms)
#         floor_friction      −196   (HIGH friction is worse — grippy-mat regime)
#         proj_grav_bias_abs  −171   (|bias| 0.09–0.13 vs 0–0.04 ≈ 5–7.5° vs ≤2.5°)
#         stiffness_mult      +159   (soft 0.80 hurts; stiff 1.20 helps)
#         mass_mult           −116   (1.12–1.20 vs 0.95–1.03)
#         imu_delay           −114   (2 vs 1 step)
#         com_offset_abs       −97   (per-link ±0.02)
#         armature_mult        −62,  joint_delay −57, joint_pos_bias −49
#         com_shift_x          +88   (FORWARD whole-body CoM helps — the plant's
#                                     nominal balance point sits behind where the
#                                     policy wants it; consistent with the +0.089
#                                     deploy trim being a compensator)
#     Delay alone costs almost half the mean episode. Every one of the top
#     terms is set 3–10× wider than the bench measurement:
#         measured action delay 2.4–2.8 ms (M1.1) → 1 step; trained 1–3 (10–30 ms)
#         measured IMU ~2 ms, encoder IMU+1.96 ms → 1 step each; trained 1–2 each
#         → trained total dead time mean 33 ms / max 50 ms vs ~5 ms on the bus
#         kp-sweep bench gate PASS 08-22 → stiffness ±20 % is not a real unknown
#         rotor inertia CONFIRMED 08-22 (hip armature recentered 0.0385) → ±30 % is not
#         mass weighed 5.5 kg = ×1.059 (memory says ×1.072) → 0.95–1.20 is ±12 %
#           around a number known to ±2 %
#         per-link CoM ±0.02 is redundant since the coherent dr_com_shift (S-line)
#         full 17 Nm delivered (08-17 ballistic) → effort 0.8 floor is not real
#     These bands came from the R-line "robustness via diversity" bundle for a
#     walking/push-recovery objective where survival under abuse was the goal.
#     For a curriculum gated on a stand-up SUCCESS RATIO they are a tax on
#     every gradient step. Standard practice: acquire the skill on a plant
#     close to the measured one, THEN widen (stage 2, a later launch script).
#
# OUT-OF-THE-BOX HYPOTHESIS (pre-registered, not acted on here): the 08-31
# "fixed-point dead zone" (quasi-static pitch response annihilated, only RATE
# wakes the policy) is exactly what a policy SHOULD learn under ±0.10–0.13
# pitch-obs bias + 33–50 ms dead time — a static pitch reading is
# uninformative to ±7°, and high static gain + delay oscillates, so the
# optimum is a low-static-gain rate-driven controller. i.e. the R-line DR
# bundle TRAINED the dead zone. Test (after this run): a quasi-static
# (pitch, wy) response map (data/_p12_3_response_map.py construction) on
# S4b (pg-bias ±0.05) vs S4a (±0.13) — S4b should show pitch dependence at
# wy=0; S4a should stay flat. Supported → the deploy trim (+0.089) was
# compensating a training artefact, and arm B is the flight candidate.
#
# CHANGES vs launch_train_S3a_path.ps1 (everything else identical, incl. PPO,
# curriculum recipe, path mode, pushes, exec noise, spawn DR, com_shift):
#   resume 2026-09-08_20-52-15_S3a_path/model_6090.pt (pre-break peak), no
#     reset_optimizer; alpha restarts 0.5 (alpha is not in the ckpt).
#   DR → measured-plant bands:
#     env.dr_action_delay_max   3 → 2      (cfg default; still 4–7× the bench value)
#     env.dr_imu_delay_max      2 → 1      (cfg default, measured)
#     env.dr_joint_delay_max    2 → 1      (cfg default, measured)
#     env.dr_stiffness_mult     0.80–1.20 → 0.90–1.10
#     env.dr_armature_mult      0.70–1.30 → 0.80–1.20  (was cfg default, never overridden)
#     env.dr_mass_mult          0.95–1.20 → 1.00–1.14  (centre 1.07, ±7 %)
#     env.dr_com_offset_max     0.02 → 0.01           (coherent shift keeps ±0.02/0.01)
#     env.dr_effort_mult_min    0.8 → 0.9  (cfg default)
#     env.dr_floor_friction_min 0.4 → 0.6  (max stays 1.1: high friction IS the
#                                           hardware regime and the harder case)
#   Speed: env.kneel_start_probability 0.4 → 0.6 (stand branch is already
#     trained; 1.5× the stand-up samples per iteration; TB kneel_branch_frac
#     was 0.35 → expect ≈0.5).
#   NOT changed: dr_proj_grav_bias_max stays 0.13 in THIS arm (deploy-trim
#     compatible: hardware obs bias ≈ +0.12 at upright under the +0.089 trim).
#     Arm B (launch_train_S4b_measDR_pg05.ps1) cuts it to 0.05 — the single
#     difference between the arms.
#
# PREDICTIONS (fail = DR was not the blocker):
#   P1 population mean_episode_length ≥ 1100 within 200 iters (S3a: 700–840).
#   P2 curriculum/standup_success_low ≥ 0.80 within 300 iters (S3a: 0.68).
#   P3 alpha 0.5 → 0.6 (advance_metric ≥ 0.7) within 500 iters (S3a: 0.39 @950).
#   P4 alpha 0.6 → 0.7 (the liftoff band, d 0.55–0.7 kneeling) is the honest
#      test of the path design, not of DR: if success_high ≈ 0 and
#      standup/pitch_err_grace_mean not shrinking for >600 iters at alpha 0.6,
#      the levers are a scripted rock at K1 or a liftoff reward — not DR, not PPO.
#   If P1–P3 all fail in BOTH arms, the phase-B crouch rise itself (path shape,
#   grace reward) is the blocker and DR narrowing is done — do not narrow further.
# WATCH: curriculum/standup_success_{low,high}, advance_metric_value, standup_alpha,
#   standup/pitch_err_grace_mean, Train/mean_episode_length, Loss/value_function.
# KILL: value_function > 3× its first-100-iter mean for >100 iters with ep len
#   falling, or raw_action_max_abs > 10 (K4 runaway), or reward collapse >300 iters.
# STAGE 2 (after alpha 1.0): re-widen delays/stiffness/mass toward the R1 bands
#   for robustness, resume, re-check the stand-up success does not regress.
$ErrorActionPreference = 'Stop'

$envRoot = '<CONDA_ENV>'
$env:CUDA_VISIBLE_DEVICES = 'GPU-<uuid>'   # RTX 4090 D (index 1 on 09-09)
$env:CARB_APP_PATH = '<ISAACLAB>\_isaac_sim\kit'
$env:EXP_PATH      = '<ISAACLAB>\_isaac_sim\apps'
$env:ISAAC_PATH    = '<ISAACLAB>\_isaac_sim'
$env:PYTHONPATH    = ';<ISAACLAB>\_isaac_sim\site'
$env:RESOURCE_NAME = 'IsaacSim'
$env:CONDA_PREFIX  = $envRoot
$env:PATH = "$envRoot;$envRoot\Library\mingw-w64\bin;$envRoot\Library\usr\bin;$envRoot\Library\bin;$envRoot\Scripts;$envRoot\bin;" + $env:PATH

$argList = @(
  '-u', 'scripts/rsl_rl/train.py',
  '--task', 'Template-standup_curriculum-v0',
  '--headless',
  '--run_name', 'S4b_measDR_pg05',
  '--resume',
  '--load_run', '2026-09-08_20-52-15_S3a_path',
  '--checkpoint', 'model_6090.pt',   # S3a pre-break peak (success_high 0.26, ep len 838)
  # PPO params — desired_kl paired with the 1e-3 lr ceiling
  'agent.algorithm.entropy_coef=0.005',
  'agent.algorithm.desired_kl=0.02',
  'agent.algorithm.clip_param=0.12',
  # base K recipe
  'env.rew_scale_gait_ref=3.0',
  'env.sat_term_threshold=0.5',
  'env.sat_term_tau_s=0.5',
  'env.swing_clearance_zero_m=0.078',
  'env.servo_kp_hip_pitch=20.0',
  'env.servo_kd_hip_pitch=1.4',
  'env.servo_kp_knee=30.0',
  'env.servo_kd_knee=1.5',
  # bench-corrected plant
  'env.hip_pitch_armature=0.0385',
  # L1 calm trio
  'env.swing_clearance_target_m=0.015',
  'env.rew_scale_no_flight=-15.0',
  'env.rew_scale_vz_penalty=-8.0',
  # smoothness/snap (R1_resume3110 level)
  'env.rew_scale_action_smoothness=-0.24',
  'env.rew_scale_action_snap=-0.06',
  # exec-noise injection (R1 level) — kept: models belt seating / tooth skip, hardware-observed
  'env.exec_noise_target_bias_max=0.03',
  'env.exec_noise_target_std=0.015',
  # spawn-posture DR (R1 level) — kept: scaled down by (1−0.8d)/(1−d) on kneel spawns
  'env.noise_joint_pos=0.18',
  'env.init_base_orient_noise_rad=0.12',
  # pushes ON (R1 level) — kept: gated off during grace; the hardware-relevant skill
  'env.push_enable=true',
  'env.push_probability=0.001',
  'env.push_force_min=10.0',
  'env.push_force_max=40.0',
  'env.push_duration_steps=10',
  # ── delays → MEASURED (S4, 09-09): bench 2.4–2.8 ms action, ~2 ms IMU, +1.96 ms encoder
  'env.dr_action_delay_min=1',
  'env.dr_action_delay_max=2',      # was 3
  'env.dr_imu_delay_min=1',
  'env.dr_imu_delay_max=1',         # was 2
  'env.dr_joint_delay_min=1',
  'env.dr_joint_delay_max=1',       # was 2
  # ── plant DR → MEASURED (S4, 09-09)
  'env.dr_floor_friction_min=0.6',  # was 0.4
  'env.dr_floor_friction_max=1.1',
  'env.dr_stiffness_mult_min=0.90', # was 0.80 (kp-sweep gate PASS 08-22)
  'env.dr_stiffness_mult_max=1.10', # was 1.20
  'env.dr_armature_mult_min=0.80',  # was 0.70 cfg default (rotor inertia confirmed 08-22)
  'env.dr_armature_mult_max=1.20',  # was 1.30
  'env.dr_effort_mult_min=0.9',     # was 0.8 (full 17 Nm delivered, 08-17)
  'env.dr_com_offset_max=0.01',     # was 0.02 (per-link; coherent shift below covers the CoM)
  'env.dr_ang_vel_bias_max=0.05',   # unchanged (dr_impact ≈ 0)
  'env.dr_gravity_tilt_max_deg=2.0',
  # ── S-line deltas ─────────────────────────────────────────
  # coherent whole-body CoM shift (base frame) — kept (static residual ≲1 cm measured)
  'env.dr_com_shift_x_max=0.02',
  'env.dr_com_shift_y_max=0.01',
  # obs pitch-bias band — ARM B: honest mounting band (see header: deploy contract!)
  'env.dr_proj_grav_bias_max=0.05',   # ARM B: was 0.13 (the one A/B difference)
  # mass recentred on the weigh-in (×1.059–1.072), ±7 %
  'env.dr_mass_mult_min=1.00',      # was 0.95
  'env.dr_mass_mult_max=1.14',      # was 1.20
  # kneel→stand curriculum ON — ceiling-relative criterion + frontier sampling (09-08)
  'env.curriculum_enable=true',
  'env.curriculum_initial_alpha=0.5',
  'env.curriculum_max_alpha=1.0',
  'env.curriculum_alpha_step=0.1',
  'env.curriculum_advance_metric=success_high_rel',
  'env.curriculum_success_post_grace_steps=100',
  'env.curriculum_advance_threshold=0.7',
  'env.curriculum_frontier_frac=0.5',
  'env.curriculum_frontier_width=0.3',
  'env.curriculum_advance_window=2000',
  'env.curriculum_advance_cooldown=6000',
  'env.kneel_start_probability=0.6',  # was 0.4 (speed: 1.5× stand-up samples)
  # pickup OFF (mid-episode and on-spawn)
  'env.pickup_enable=false',
  # ── stand-up PATH mode (09-08) ─────────────────────────────
  'env.standup_ramp_mode=path',
  'env.standup_grace_seconds=5.0',
  'env.standup_path_spawn_clearance_m=0.006',
  'env.rew_scale_standup_track=4.0',
  'env.rew_scale_standup_contact=2.0'
)
$p = Start-Process -FilePath "$envRoot\python.exe" `
  -WorkingDirectory '<REPO>\RL' `
  -ArgumentList $argList `
  -RedirectStandardOutput '<REPO>\RL\logs\train_S4b_measDR_pg05.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_S4b_measDR_pg05.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
