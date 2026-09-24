# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# Resume P9 run A on GPU 0 from the last pre-explosion checkpoint.
# Run A's sim exploded at iter 564 (value loss ~5e11, envs frozen); model_560 is
# the last clean save. --reset_optimizer clears the Adam/LR state that was
# pinned at the 1e-2 adaptive ceiling during the blow-up; weights load intact.
# Recipe unchanged: dr_proj_grav_bias_max=0.10 (cfg default), seed 42.
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
  '--run_name', 'A_pgbias010_r560',
  '--resume',
  '--load_run', '2026-08-18_16-06-53',
  '--checkpoint', 'model_560.pt',
  '--reset_optimizer'
)
$p = Start-Process -FilePath "$envRoot\python.exe" `
  -WorkingDirectory '<REPO>\RL' `
  -ArgumentList $argList `
  -RedirectStandardOutput '<REPO>\RL\logs\train_A_r560.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_A_r560.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
