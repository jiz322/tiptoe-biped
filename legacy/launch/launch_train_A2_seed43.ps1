# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# Launch P9 second seed on GPU 1: fresh train, seed 43, deployment recipe
# (dr_proj_grav_bias_max=0.10 cfg default). Replaces the B_pgbias003 control
# arm, which trained for the cancelled IMU-trim recalibration (08-18 decision:
# trim stays -0.089, so only the 0.10 recipe can deploy). Second seed hedges
# the stochastic breakout window (P8 lineage broke out at iter ~1000-1300).
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
  '--run_name', 'A2_seed43_pgbias010',
  '--seed', '43'
)
$p = Start-Process -FilePath "$envRoot\python.exe" `
  -WorkingDirectory '<REPO>\RL' `
  -ArgumentList $argList `
  -RedirectStandardOutput '<REPO>\RL\logs\train_A2_seed43.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_A2_seed43.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
