# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# P9 K3s arm B on GPU 1: same base + zero fix + jerk penalties as K3
# (GPU 0), PLUS swing_clearance scale 2.0 -> 4.0 — the A/B variable.
# Design intent (owner, 2026-08-21): the target idle behavior is MARCHING
# IN PLACE with real clearance — quiet standing was never trained and is
# not the goal. The failure mode is the DEGENERATE march: 3 mm shuffle
# that farms the gait terms without ever truly lifting (idle_swing_above_
# contact_frac 0.089 vs above_zero_frac 0.997 on K_r400/K2 = feet live
# in the 6 mm between ramp start and the 0.084 contact line).
# Arm A tests whether re-referencing the ramp (zero 0.078) alone converts
# shuffle -> lift. This arm doubles the ramp's SCALE on top: at 4.0 a
# full 40 mm lift pays 4.0/step vs alive 6.0 — lifting becomes a
# first-order term instead of a 2.4% garnish. Risk this tests: overshoot
# (feet_air/no_flight guard the double-support side) and jerk fighting
# the doubled smoothness penalties.
# A/B contrast vs K3 (GPU 0) is rew_scale_swing_clearance ONLY.
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
  '--run_name', 'K3s_zero078_clr4',
  '--seed', '42',
  '--resume',
  '--load_run', '2026-08-20_06-45-41_K_r400_zero069',
  '--checkpoint', 'model_2040.pt',
  'agent.algorithm.entropy_coef=0.005',
  'env.rew_scale_gait_ref=3.0',
  'env.sat_term_threshold=0.5',
  'env.sat_term_tau_s=0.5',
  'env.swing_clearance_zero_m=0.078',
  'env.rew_scale_swing_clearance=4.0',
  'env.rew_scale_action_smoothness=-0.16',
  'env.rew_scale_action_snap=-0.04',
  'env.servo_kp_hip_pitch=20.0',
  'env.servo_kd_hip_pitch=1.4',
  'env.servo_kp_knee=30.0',
  'env.servo_kd_knee=1.5'
)
$p = Start-Process -FilePath "$envRoot\python.exe" `
  -WorkingDirectory '<REPO>\RL' `
  -ArgumentList $argList `
  -RedirectStandardOutput '<REPO>\RL\logs\train_K3s_zero078_clr4.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_K3s_zero078_clr4.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
