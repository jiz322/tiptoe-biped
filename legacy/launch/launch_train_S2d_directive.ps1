# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# S-line arm B, RELAUNCH #3 — "S2d_directive" — GPU 1 (RTX 4090 D, by UUID), 2026-09-08.
#
# WHY: S2c (launch_train_S2c_directive.ps1, iters 4000→4910) was healthy but
# stuck at alpha 0.6 for 870 iters: its d≈0.50 band rose 0.34→0.45 (+1.3 pts
# per 100 iters) against an absolute bar of 0.6 while its LOW band (d≈0.10)
# only reaches 0.68 — arm B's DR bundle (lean 0.20 rad, knee bias, CoM 3 cm,
# handover prime) loses 32 % of episodes within 1.4 s of spawn regardless of
# kneel depth, so 0.6 demanded 88 % of the attainable ceiling (arm A: 75 %).
# The user needs alpha 1.0 (prone-fold ground-rest pose). Same two levers as
# S1d: ceiling-relative criterion + frontier-focused sampling; see
# launch_train_S1d_plantfix.ps1 for the measurements.
#
# CHANGES vs launch_train_S2c_directive.ps1:
#   --load_run <S2c run> --checkpoint model_4920.pt (no reset_optimizer)
#   env.curriculum_initial_alpha 0.5 → 0.6   (S2c's current alpha)
#   env.curriculum_advance_metric success_high → success_high_rel, threshold
#       0.6 → 0.7 (≈ 0.48 absolute at arm B's ceiling — roughly where the
#       d≈0.5 band already is, so expect an early 0.6→0.7 step)
#   env.curriculum_frontier_frac=0.5, env.curriculum_frontier_width=0.3
#   everything else identical to S2c (arm-B extras unchanged).
#
# WATCH: standup_success_low — if it keeps falling as alpha rises, the DR
# bundle plus deep kneels is too much at once and arm B should be judged
# on balance evals only. KILL: as S2c.
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
  '--run_name', 'S2d_directive',
  '--resume',
  '--load_run', '2026-09-07_07-39-00_S2c_directive',
  '--checkpoint', 'model_4920.pt',
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
  # kneel→stand curriculum ON — ceiling-relative criterion + frontier sampling (09-08)
  'env.curriculum_enable=true',
  'env.curriculum_initial_alpha=0.6',
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
  # ── arm B extras ──────────────────────────────────────────
  'env.dr_com_shift_x_max=0.03',
  'env.dr_com_shift_y_max=0.01',
  'env.dr_knee_angle_bias_max=0.15',
  'env.handover_prime_prob=0.5'
)
$p = Start-Process -FilePath "$envRoot\python.exe" `
  -WorkingDirectory '<REPO>\RL' `
  -ArgumentList $argList `
  -RedirectStandardOutput '<REPO>\RL\logs\train_S2d_directive.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_S2d_directive.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
