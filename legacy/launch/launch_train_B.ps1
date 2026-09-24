# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# Launch P9 run B on GPU 1 (detached): control arm with dr_proj_grav_bias_max=0.03
# vs run A (live on GPU 0) which trains the uncommitted 0.10 widening.
# Mirrors conda env_isaaclab activation (etc/conda/activate.d/env_vars.ps1) so
# python.exe resolves Isaac Sim's bundled site-packages without conda.
$ErrorActionPreference = 'Stop'

$envRoot = '<CONDA_ENV>'
$env:CUDA_VISIBLE_DEVICES = '1'          # hard isolation: GPU 0 is full with run A
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
  '--run_name', 'B_pgbias003',
  'env.dr_proj_grav_bias_max=0.03'
)
$p = Start-Process -FilePath "$envRoot\python.exe" `
  -WorkingDirectory '<REPO>\RL' `
  -ArgumentList $argList `
  -RedirectStandardOutput '<REPO>\RL\logs\train_B_pgbias003.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_B_pgbias003.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
