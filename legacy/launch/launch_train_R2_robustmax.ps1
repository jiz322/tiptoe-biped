# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# P13 R2 "robust bundle, MAX amplitude" on GPU 1 (2026-08-25 evening).
#
# Same design as R1_robust (see launch_train_R1_robust.ps1 header for the
# full rationale, resume point, lr-cap removal, and kill rule). The A/B
# axis is AMPLITUDE ONLY: every randomization/injection knob pushed to the
# aggressive end. Known risk (deliberate): over-wide DR can fail to train
# or produce a conservative policy — that is exactly what R1 hedges.
# Deltas vs R1:
#   exec-noise: bias ±0.05 rad (hip ~1 Nm / knee ~1.5 Nm), white 0.03
#   spawn: joints ±0.28 rad, base roll/pitch ±0.18 (~10 deg)
#   pushes: p=0.0015, 15-55 N
#   delays: action 1-4, joint 1-3
#   floor mu 0.3-1.2, stiffness ×0.70-1.30, damping ×0.6-1.5,
#   effort ×0.7-1.0, com 0.03, mass ×0.90-1.22, encoder bias 0.10,
#   ang-vel bias 0.06, gravity tilt 3 deg, obs noise ×1.5
#     (pg 0.075, w 0.22, qd 0.9, q 0.012)
$ErrorActionPreference = 'Stop'

$envRoot = '<CONDA_ENV>'
$env:CUDA_VISIBLE_DEVICES = '1'
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
  '--run_name', 'R2_robustmax',
  '--resume',
  '--load_run', '2026-08-20_06-45-41_K_r400_zero069',
  '--checkpoint', 'model_2040.pt',
  '--reset_optimizer',
  'agent.algorithm.entropy_coef=0.005',
  # RESTART v2 (08-25 eve): desired_kl 0.02 + 1e-3 lr ceiling — see
  # R1 script note; R2's first launch hit a 95k vloss spike + declining
  # reward within 100 iters under the uncapped scheduler.
  'agent.algorithm.desired_kl=0.02',
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
  # NEW: execution-noise injection (max)
  'env.exec_noise_target_bias_max=0.05',
  'env.exec_noise_target_std=0.03',
  # spawn-posture DR (max)
  'env.noise_joint_pos=0.28',
  'env.init_base_orient_noise_rad=0.18',
  # pushes ON (max)
  'env.push_enable=true',
  'env.push_probability=0.0015',
  'env.push_force_min=15.0',
  'env.push_force_max=55.0',
  'env.push_duration_steps=10',
  # delays (max)
  'env.dr_action_delay_min=1',
  'env.dr_action_delay_max=4',
  'env.dr_imu_delay_min=1',
  'env.dr_imu_delay_max=2',
  'env.dr_joint_delay_min=1',
  'env.dr_joint_delay_max=3',
  # plant DR (max)
  'env.dr_floor_friction_min=0.3',
  'env.dr_floor_friction_max=1.2',
  'env.dr_stiffness_mult_min=0.70',
  'env.dr_stiffness_mult_max=1.30',
  'env.dr_damping_mult_min=0.6',
  'env.dr_damping_mult_max=1.5',
  'env.dr_effort_mult_min=0.7',
  'env.dr_com_offset_max=0.03',
  'env.dr_mass_mult_min=0.90',
  'env.dr_mass_mult_max=1.22',
  'env.dr_joint_pos_bias_max=0.10',
  'env.dr_ang_vel_bias_max=0.06',
  'env.dr_gravity_tilt_max_deg=3.0',
  # obs noise x1.5
  'env.dr_obs_noise_proj_grav=0.075',
  'env.dr_obs_noise_ang_vel=0.22',
  'env.dr_obs_noise_joint_vel=0.9',
  'env.dr_obs_noise_joint_pos=0.012'
)
$p = Start-Process -FilePath "$envRoot\python.exe" `
  -WorkingDirectory '<REPO>\RL' `
  -ArgumentList $argList `
  -RedirectStandardOutput '<REPO>\RL\logs\train_R2_robustmax.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_R2_robustmax.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
