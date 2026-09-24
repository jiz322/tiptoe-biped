# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# S-line arm B — "S2_directive" — GPU 1 (RTX 4090 D, selected by UUID), 2026-09-06.
#
# CONTEXT: see launch_train_S1_plantfix.ps1 (arm A) — same fork point
# (R1_resume3110/model_3150_currnorm.pt), same curriculum settings, same
# CoM-fix / pg-band / mass-recenter deltas. This arm is the "attack the
# hypothesis" bet: it adds every remaining item of the next-training directive
# (memory `next-training-directive`, 08-31 → 09-03) on top of A.
#
# EXTRA DELTAS vs arm A:
#   env.dr_com_shift_x_max 0.02 → 0.03   (whole-body CoM_x ±3 cm; torso row
#                                          moves ±9.2 cm — deliberately strong)
#   env.dr_knee_angle_bias_max=0.15      NEW: persistent knee encoder-vs-joint
#       offset δ per leg (belt tooth skip = 0.21 rad, stretch, crawl-zero).
#       COUPLED: policy reads q_true−δ AND the servo holds q_true = target+δ
#       (the independent obs-bias / exec-bias knobs cannot express this).
#       Leading candidate for what the +0.12 IMU trim compensates (09-03).
#   env.handover_prime_prob=0.5          NEW: half the resets see the real
#       first sensor frame through the delay pipe (deploy 'b' contract), the
#       other half stay blind (deploy 'v'). Policy robust to either handover.
#   env.init_base_orient_noise_rad 0.12 → 0.20   static-lean spawn to ±11.5°
#       at zero rate (the p12.3 fixed-point dead zone: policy only wakes on
#       rate; kneel-branch poses add far more static-lean coverage on top).
#
# RISK: this is the R2 pattern (many new dims at once). R2 never trained.
# If this arm's reward stays flat/collapses while A learns, A is the fallback
# and the extra dims get re-introduced one at a time from A's checkpoint.
#
# PREDICTIONS / WATCH / KILL: as arm A, plus dr/knee_angle_bias_abs_mean ≈ 0.075,
# dr/com_shift_x_abs_mean ≈ 0.015. Compare A vs B on curriculum/standup_alpha
# progress and on the post-training pickup-free hw_matched + hw_push_recovery
# eval (run_eval_p13r_GPU2.ps1 pattern; re-pass armature AND the new knobs).
$ErrorActionPreference = 'Stop'

$envRoot = '<CONDA_ENV>'
$env:CUDA_VISIBLE_DEVICES = 'GPU-<uuid>'   # RTX 4090 D (index 1 on 09-06)
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
  '--run_name', 'S2_directive',
  '--resume',
  '--load_run', '2026-08-27_10-30-37_R1_resume3110',
  '--checkpoint', 'model_3150_currnorm.pt',
  '--reset_optimizer',
  # PPO params (R1_resume3110 level)
  'agent.algorithm.entropy_coef=0.005',
  'agent.algorithm.desired_kl=0.008',
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
  # spawn-posture DR — joint noise R1 level, static lean WIDENED (arm B)
  'env.noise_joint_pos=0.18',
  'env.init_base_orient_noise_rad=0.20',
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
  # ── S-line shared deltas ──────────────────────────────────
  'env.dr_proj_grav_bias_max=0.13',
  'env.dr_mass_mult_min=0.95',
  'env.dr_mass_mult_max=1.20',
  'env.curriculum_enable=true',
  'env.curriculum_initial_alpha=0.3',
  'env.curriculum_max_alpha=1.0',
  'env.curriculum_alpha_step=0.1',
  'env.curriculum_advance_threshold=0.20',
  'env.curriculum_advance_window=2000',
  'env.curriculum_advance_cooldown=1500',
  'env.kneel_start_probability=0.4',
  'env.pickup_initial_probability=0.4',
  # ── arm B extras ──────────────────────────────────────────
  'env.dr_com_shift_x_max=0.03',
  'env.dr_com_shift_y_max=0.01',
  'env.dr_knee_angle_bias_max=0.15',
  'env.handover_prime_prob=0.5'
)
$p = Start-Process -FilePath "$envRoot\python.exe" `
  -WorkingDirectory '<REPO>\RL' `
  -ArgumentList $argList `
  -RedirectStandardOutput '<REPO>\RL\logs\train_S2_directive.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_S2_directive.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
