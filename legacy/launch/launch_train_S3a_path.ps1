# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# S-line arm A, RELAUNCH #4 — "S3a_path" — GPU 0 (RTX 4090, by UUID), 2026-09-08.
#
# WHY: the 09-08 stand-up investigation (memory standup-ramp-topple-finding,
# probes RL/scripts/rsl_rl/probe_standup_{ramp,traj,keyframes,pose_dump,
# balance_manifold}.py) showed the P6 linear joint-space ramp CANNOT stand
# the robot up from d=1.0 (seiza: knees + foot balls on the ground, torso
# vertical, CoM at the front edge of the knee→ball base): with zero action it
# rocks back onto its knees, then the continuing ramp throws it onto its
# hips ("seal" pose) at EVERY d ≥ 0.7. S1d (this arm) was stopped at
# model_5110 (alpha 0.7) — it could not have passed d≈0.7 under that ramp.
#
# NEW (rl_env_p6 / rl_env_cfg_p6 "path" mode, 09-08):
#   * standup_ramp_mode=path — absolute keyframe path seiza → K1 crouch
#     (hip 1.10, knee −2.63, torso ≈ +8°, s=0.45) → straight squat line to
#     standing (s=1). Phase A (s ≤ 0.45) was validated with ZERO policy action
#     on the nominal plant: passively stable on knees+balls, torso −90 → +8°,
#     CoM within ±2 cm of the ball contact, no rocking. Phase B (s > 0.45)
#     is the CoM-over-ball squat contour (θ* +15 → −4.5°, balance_manifold_
#     ball.csv); the kneel→balls LIFTOFF at K1 is an inherently dynamic ~3 cm
#     CoM rock that no open-loop ramp can do (manifold: no pose balances on
#     the balls with the knees down) — that is the policy's job here.
#   * spawn ON the path: difficulty d spawns at phase s0 = 1 − d with the
#     measured resting pitch/height (A) or the balanced-on-balls state (B).
#     d ≤ 0.55 = crouches on the balls (torso at standing pitch, knees
#     −0.0…−2.6); d > 0.55 = kneeling states incl. the liftoff.
#   * grace shaping: rew_scale_standup_track=4 (exp(−(Δpitch/0.3)²) vs the
#     path pitch), rew_scale_standup_contact=2 (feet down); standup_height
#     kept (rewards B's rise, ≈0 in A). standup_grace_seconds 4 → 5 (constant
#     path speed: d=1 takes 5 s, B ≈ 2.75 s).
#   * curriculum restarts at alpha 0.5 (phase-B band) with the S1d frontier /
#     success_high_rel recipe; alpha ≥ 0.6 brings in the liftoff.
#
# CHANGES vs launch_train_S1d_plantfix.ps1: resume S1d model_5110 (no
# reset_optimizer); + env.standup_ramp_mode=path, env.rew_scale_standup_track
# =4.0, env.rew_scale_standup_contact=2.0, env.standup_grace_seconds=5.0,
# env.standup_path_spawn_clearance_m=0.006; env.curriculum_initial_alpha
# 0.7 → 0.5. Everything else identical.
#
# PREDICTIONS: alpha 0.5 band (balls-only crouches) mastered within ~300
# iters (S1c policy already survives 29 % of d=0.2 spawns untrained); the
# liftoff band (alpha 0.6–0.7, d ∈ [0.55, 0.7]) is the real test: if
# curriculum/standup_success_high stays ≈ 0 with standup/pitch_err_grace_mean
# not shrinking for >800 iters at alpha 0.6, the dynamic liftoff is not being
# discovered → next levers are a scripted rock (hip lead burst at K1) or a
# liftoff-specific reward, NOT PPO knobs. WATCH: standup/pitch_err_grace_mean,
# reward/standup_track_mean, curriculum/standup_success_{high,low}, alpha.
# KILL: as S1c (reward collapse >300 iters, raw_action_max_abs runaway).
#
# RELAUNCH #2 (09-08 eve, from 2026-09-08_17-27-35_S3a_path/model_5230.pt, ~80 iters at alpha 0.5):
# the first S3 knot table paired ramp TARGETS with the pitch/height of a MOVING
# population → phase-A spawns (d ∈ [0.85,1.0)) were inconsistent with the
# ground (found by script/show_standup_path.py -Scan: K1 ejected at 3 m/s; env
# scan showed a 0.5–0.8 m/s depenetration kick and 3–5° / ≤3 cm settling
# everywhere in A). Table is now 7 columns [s, hip_T, knee_T, pitch, root_h,
# hip_rest, knee_rest] from 4 s zero-action holds of each target; spawn writes
# the rest joints, center follows the targets (no offset blend). Env scan: every
# A spawn rests within 0.3° / 2 mm at ≤0.1 m/s; viewer -Scan 0:0.5:0.025 all ok.
# Phase-B rows unchanged, so the alpha-0.5 progress carries over.
$ErrorActionPreference = 'Stop'

$envRoot = '<CONDA_ENV>'
$env:CUDA_VISIBLE_DEVICES = 'GPU-<uuid>'   # RTX 4090 (index 0 on 09-06)
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
  '--run_name', 'S3a_path',
  '--resume',
  '--load_run', '2026-09-08_17-27-35_S3a_path',
  '--checkpoint', 'model_5230.pt',   # RELAUNCH #2 09-08 eve (was 2026-09-08_08-01-02_S1d_plantfix/model_5110.pt)
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
  # exec-noise injection (R1 level)
  'env.exec_noise_target_bias_max=0.03',
  'env.exec_noise_target_std=0.015',
  # spawn-posture DR (R1 level)
  'env.noise_joint_pos=0.18',
  'env.init_base_orient_noise_rad=0.12',
  # pushes ON (R1 level)
  'env.push_enable=true',
  'env.push_probability=0.001',
  'env.push_force_min=10.0',
  'env.push_force_max=40.0',
  'env.push_duration_steps=10',
  # delays (R1 level)
  'env.dr_action_delay_min=1',
  'env.dr_action_delay_max=3',
  'env.dr_imu_delay_min=1',
  'env.dr_imu_delay_max=2',
  'env.dr_joint_delay_min=1',
  'env.dr_joint_delay_max=2',
  # plant DR widened (R1 level)
  'env.dr_floor_friction_min=0.4',
  'env.dr_floor_friction_max=1.1',
  'env.dr_stiffness_mult_min=0.80',
  'env.dr_stiffness_mult_max=1.20',
  'env.dr_effort_mult_min=0.8',
  'env.dr_com_offset_max=0.02',
  'env.dr_ang_vel_bias_max=0.05',
  'env.dr_gravity_tilt_max_deg=2.0',
  # ── S-line deltas ─────────────────────────────────────────
  # THE FIX: coherent whole-body CoM shift (base frame)
  'env.dr_com_shift_x_max=0.02',
  'env.dr_com_shift_y_max=0.01',
  # obs pitch-bias band must contain the deploy trim regime
  'env.dr_proj_grav_bias_max=0.13',
  # mass recenter on measured weight
  'env.dr_mass_mult_min=0.95',
  'env.dr_mass_mult_max=1.20',
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
  'env.kneel_start_probability=0.4',
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
  -RedirectStandardOutput '<REPO>\RL\logs\train_S3a_path.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_S3a_path.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
