# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# P13 R1 "robust bundle" RESUME from model_3110 on GPU 0 (2026-08-27).
#
# CONTEXT: R1_robust hit peak reward at iter 3110 (~16.8k), then declined.
# User diagnosed: healthy entropy, but needs tighter control — lower LR,
# tighter clip, and boost smoothness/snap penalties to suppress jitter.
#
# STRATEGY: fork from model_3110 with --reset_optimizer to escape the
# current optimizer momentum, then:
#   - Lower desired_kl: 0.02 → 0.008 (tighten LR schedule; with 1e-3
#     ceiling patch active, 0.008 should NOT pin at floor under lighter
#     optimizer momentum from the reset)
#   - Tighter clip: 0.15 → 0.12 (reduce policy update step size)
#   - Boost smoothness: -0.08 → -0.24 (3× increase, K5_polish level)
#   - Boost snap: -0.02 → -0.06 (3× increase, K5_polish level)
#
# All other hyperparams and DR settings UNCHANGED from the parent run.
# WATCH: reward should stabilize or climb slowly from ~16.8k; sigma should
# drift down; action jerk/snap metrics should drop. KILL RULE: sustained
# reward collapse below 14k.
$ErrorActionPreference = 'Stop'

$envRoot = '<CONDA_ENV>'
$env:CUDA_VISIBLE_DEVICES = '0'
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
  '--run_name', 'R1_resume3110',
  '--resume',
  '--load_run', '2026-08-25_14-54-09_R1_robust',
  '--checkpoint', 'model_3110.pt',
  '--reset_optimizer',
  # TIGHTENED PPO params (the delta vs parent)
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
  # BOOSTED smoothness/snap (the delta vs parent)
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
  'env.dr_gravity_tilt_max_deg=2.0'
)
$p = Start-Process -FilePath "$envRoot\python.exe" `
  -WorkingDirectory '<REPO>\RL' `
  -ArgumentList $argList `
  -RedirectStandardOutput '<REPO>\RL\logs\train_R1_resume3110.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_R1_resume3110.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
