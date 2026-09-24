# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# P9 K5b on GPU 1: seed-45 twin of K5_polish — same branch point
# (K4b model_3600, the P11 primary deploy pick), same final-polish
# recipe (entropy 0.003 + desired_kl 0.002), ONLY the seed differs.
# See launch_train_K5_polish.ps1 header for full context/rationale.
# Purpose: two independently polished candidates for the P11 hardware
# A/B (K4b/K4b2 playbook, one generation later); also guards against a
# polish-induced degradation being mistaken for the recipe when it's
# the seed. Kill rule: reward collapse (not vloss spikes).
# NOTE: resume re-reads cfg defaults — ALL overrides re-passed here.
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
  '--run_name', 'K5b_polish_seed45',
  '--seed', '45',
  '--resume',
  '--load_run', '2026-08-23_08-56-43_K4b_resume3420',
  '--checkpoint', 'model_3600.pt',
  'agent.algorithm.entropy_coef=0.003',
  'agent.algorithm.desired_kl=0.002',
  'env.rew_scale_gait_ref=3.0',
  'env.sat_term_threshold=0.5',
  'env.sat_term_tau_s=0.5',
  'env.swing_clearance_zero_m=0.078',
  'env.rew_scale_swing_clearance=4.0',
  'env.rew_scale_action_smoothness=-0.24',
  'env.rew_scale_action_snap=-0.06',
  'env.servo_kp_hip_pitch=20.0',
  'env.servo_kd_hip_pitch=1.4',
  'env.servo_kp_knee=30.0',
  'env.servo_kd_knee=1.5'
)
$p = Start-Process -FilePath "$envRoot\python.exe" `
  -WorkingDirectory '<REPO>\RL' `
  -ArgumentList $argList `
  -RedirectStandardOutput '<REPO>\RL\logs\train_K5b_polish_seed45.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_K5b_polish_seed45.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
