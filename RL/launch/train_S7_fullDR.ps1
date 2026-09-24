# TipToe: train the S7 "full DR" arm (canonical training template, Windows / PowerShell).
#
# This is the latest training recipe of the campaign (arm S7, 2026-09-16), with the
# machine-specific parts replaced by environment variables. The historical script,
# with its full header explaining every band, is legacy/launch/launch_train_S7_fullDR.ps1.
#
# Differences from the historical arm:
#   * It trains FROM SCRATCH. S7 resumed from the S4a checkpoint (model_7780, not shipped);
#     uncomment the RESUME block to resume from your own run.
#   * GPU is selected by index ($Gpu), not by UUID.
#
# Requirements (docs/ENVIRONMENT.md): Isaac Sim 5.1 binary install, Isaac Lab 0.54, this
# repo installed with `pip install -e RL/source/RL`, and the rsl_rl patch applied
# (RL/patches/README.md). Without the patch the LR schedule differs and the recipe does
# not reproduce.
#
# Usage:  powershell -File RL\launch\train_S7_fullDR.ps1 [-Gpu 0] [-RunName S7_fullDR] [-MaxIter 0] [-Headless]
#   -MaxIter 5   runs a 5-iteration smoke test (Phase A in the historical header): read back
#                every [PLANT] line and DR span, then launch the full run.
param(
    [string]$Gpu = '0',
    [string]$RunName = 'S7_fullDR',
    [int]$MaxIter = 0,
    [switch]$Headless
)
$ErrorActionPreference = 'Stop'

# Set these for your machine (or export them before calling):
#   ISAACSIM_PATH = Isaac Sim binary install (the folder that contains kit\ and apps\)
#   CONDA_PREFIX  = the conda env with Isaac Lab + this repo installed (set by `conda activate`)
$isaacSim = if ($env:ISAACSIM_PATH) { $env:ISAACSIM_PATH } else { throw 'set ISAACSIM_PATH (see docs/ENVIRONMENT.md)' }
$envRoot  = if ($env:CONDA_PREFIX)  { $env:CONDA_PREFIX }  else { throw 'activate the Isaac Lab conda env (CONDA_PREFIX unset)' }
$env:CUDA_VISIBLE_DEVICES = $Gpu
$env:CARB_APP_PATH = "$isaacSim\kit"
$env:EXP_PATH      = "$isaacSim\apps"
$env:ISAAC_PATH    = "$isaacSim"
$env:PYTHONPATH    = ";$isaacSim\site"
$env:RESOURCE_NAME = 'IsaacSim'
$env:PATH = "$envRoot;$envRoot\Library\mingw-w64\bin;$envRoot\Library\usr\bin;$envRoot\Library\bin;$envRoot\Scripts;$envRoot\bin;" + $env:PATH
$env:PYTHONUNBUFFERED = '1'

$RL = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $RL
New-Item -ItemType Directory -Force -Path "$RL\logs" | Out-Null

$argList = @(
  '-u', 'scripts/rsl_rl/train.py',
  '--task', 'Template-standup_curriculum-v0',
  '--run_name', $RunName,
  # ── RESUME block (uncomment to resume; a resume must re-pass EVERY override below,
  #    otherwise the run silently reverts to cfg defaults)
  # '--resume', '--load_run', '<timestamp>_<run_name>', '--checkpoint', 'model_<n>.pt',
  # PPO params: desired_kl paired with the 1e-3 lr ceiling (RL/patches)
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
  # smoothness / snap
  'env.rew_scale_action_smoothness=-0.24',
  'env.rew_scale_action_snap=-0.06',
  # exec-noise injection on motor targets
  'env.exec_noise_target_bias_max=0.03',
  'env.exec_noise_target_std=0.015',
  # spawn-posture DR
  'env.noise_joint_pos=0.18',
  'env.init_base_orient_noise_rad=0.12',
  # pushes ON
  'env.push_enable=true',
  'env.push_probability=0.001',
  'env.push_force_min=10.0',
  'env.push_force_max=40.0',
  'env.push_duration_steps=10',
  # delays (S4 measured)
  'env.dr_action_delay_min=1',
  'env.dr_action_delay_max=2',
  'env.dr_imu_delay_min=1',
  'env.dr_imu_delay_max=1',
  'env.dr_joint_delay_min=1',
  'env.dr_joint_delay_max=1',
  # plant DR (S4 measured)
  'env.dr_floor_friction_min=0.6',
  'env.dr_floor_friction_max=1.1',
  'env.dr_stiffness_mult_min=0.90',
  'env.dr_stiffness_mult_max=1.10',
  'env.dr_armature_mult_min=0.80',
  'env.dr_armature_mult_max=1.20',
  'env.dr_effort_mult_min=0.9',
  'env.dr_com_offset_max=0.01',
  'env.dr_ang_vel_bias_max=0.05',
  'env.dr_gravity_tilt_max_deg=2.0',
  # S-line deltas: coherent whole-body CoM shift, mass
  'env.dr_com_shift_x_max=0.02',
  'env.dr_com_shift_y_max=0.01',
  'env.dr_proj_grav_bias_max=0.00',
  'env.dr_mass_mult_min=1.00',
  'env.dr_mass_mult_max=1.14',
  # S7: belt transmission as a per-episode DR channel (knee-joint N m/rad; motor-side N m s/rad)
  'env.belt_series_stiffness_min=100.0',
  'env.belt_series_stiffness_max=1000.0',
  'env.belt_series_damping_min=0.0',
  'env.belt_series_damping_max=0.3',
  # S7: foot-ball restitution as a per-episode DR channel (0.63 is the row marker, not a value)
  'env.foot_restitution=0.63',
  'env.dr_foot_restitution_min=0.2',
  'env.dr_foot_restitution_max=0.8',
  'env.dr_foot_restitution_buckets=64',
  # kneel -> stand curriculum
  'env.curriculum_enable=true',
  'env.curriculum_initial_alpha=0.0',
  'env.curriculum_max_alpha=1.0',
  'env.curriculum_alpha_step=0.1',
  'env.curriculum_advance_metric=success_high_rel',
  'env.curriculum_success_post_grace_steps=100',
  'env.curriculum_advance_threshold=0.7',
  'env.curriculum_frontier_frac=0.5',
  'env.curriculum_frontier_width=0.3',
  'env.curriculum_advance_window=2000',
  'env.curriculum_advance_cooldown=6000',
  'env.kneel_start_probability=0.6',
  'env.pickup_enable=false',
  # stand-up PATH mode (seiza -> crouch -> squat; docs/standup-path-mode-20260908.md)
  'env.standup_ramp_mode=path',
  'env.standup_grace_seconds=5.0',
  'env.standup_path_spawn_clearance_m=0.006',
  'env.rew_scale_standup_track=4.0',
  'env.rew_scale_standup_contact=2.0'
)
if ($Headless) { $argList += '--headless' }
if ($MaxIter -gt 0) { $argList += @('--max_iterations', "$MaxIter") }

$p = Start-Process -FilePath "$envRoot\python.exe" `
  -WorkingDirectory $RL `
  -ArgumentList $argList `
  -RedirectStandardOutput "$RL\logs\train_$RunName.out.log" `
  -RedirectStandardError  "$RL\logs\train_$RunName.err.log" `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id); logs in RL\logs\train_$RunName.{out,err}.log; tensorboard --logdir RL\logs\rsl_rl"
