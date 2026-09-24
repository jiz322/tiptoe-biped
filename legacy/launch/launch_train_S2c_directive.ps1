# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# S-line arm B, RELAUNCH #2 — "S2c_directive" — GPU 1 (RTX 4090 D, by UUID), 2026-09-07.
#
# CONTEXT: see launch_train_S1c_plantfix.ps1 for the curriculum-criterion fix
# (grace inflated ep_len → alpha ran 0.3→1.0 on the cooldown clock) and
# launch_train_S2_directive.ps1 for what arm B adds over arm A (CoM shift
# 0.03, knee-angle bias DR 0.15 ≈ 0.7 belt tooth, handover-prime 0.5,
# static lean 0.20). S2b was stable throughout (iters 3360→~3980); its
# grace-corrected survival profile matched S1b's: d≈0.33 +240, d≈0.42 +180,
# d≈0.50 +110, d≈0.58 +55, d≈0.67 +18, d≥0.75 ≈ 0.
#
# CHANGES vs launch_train_S2b_directive.ps1 (same set as S1c):
#   --load_run <S2b run> --checkpoint model_<latest>.pt, no reset_optimizer
#   env.curriculum_advance_metric=success_high
#   env.curriculum_success_post_grace_steps=100
#   env.curriculum_advance_threshold 0.12 → 0.6   (success RATE of the
#       hardest tertile)
#   env.curriculum_initial_alpha 0.3 → 0.5
#   arm-B extras unchanged.
#
# WATCH / KILL: as S1c. Arm comparison: curriculum/standup_success_high at
# equal alpha, and how many iters each arm needs per alpha advance.
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
  '--run_name', 'S2c_directive',
  '--resume',
  '--load_run', '2026-09-06_14-14-08_S2b_directive',
  '--checkpoint', 'model_4000.pt',
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
  # kneel→stand curriculum ON — grace-corrected success criterion (09-07)
  'env.curriculum_enable=true',
  'env.curriculum_initial_alpha=0.5',
  'env.curriculum_max_alpha=1.0',
  'env.curriculum_alpha_step=0.1',
  'env.curriculum_advance_metric=success_high',
  'env.curriculum_success_post_grace_steps=100',
  'env.curriculum_advance_threshold=0.6',
  'env.curriculum_advance_window=2000',
  'env.curriculum_advance_cooldown=6000',
  'env.kneel_start_probability=0.4',
  # pickup OFF (mid-episode and on-spawn)
  'env.pickup_enable=false',
  # ── arm B extras ──────────────────────────────────────────
  'env.dr_com_shift_x_max=0.03',
  'env.dr_com_shift_y_max=0.01',
  'env.dr_knee_angle_bias_max=0.15',
  'env.handover_prime_prob=0.5'
)
$p = Start-Process -FilePath "$envRoot\python.exe" `
  -WorkingDirectory '<REPO>\RL' `
  -ArgumentList $argList `
  -RedirectStandardOutput '<REPO>\RL\logs\train_S2c_directive.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_S2c_directive.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
