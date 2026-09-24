# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# P9 aggressive A/B, arm X on GPU 0: exploration unlock ONLY.
# Resumes the plateau checkpoint (A's model_560, reward ~500, active_frac
# ~0.71) with entropy_coef 0.003 -> 0.005 (the documented fresh value the
# successful P8 lineage actually trained with; 0.003 was a polish leftover).
# Everything else unchanged. Readout: does pickup/active_frac break below
# ~0.65 within ~200 iters where the 0.003 runs stayed flat at 0.71?
# NOTE: the entropy override is CLI-only — any future resume of this run
# must re-pass agent.algorithm.entropy_coef=0.005 or it reverts to 0.003.
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
  '--run_name', 'X_ent005_r560',
  '--resume',
  '--load_run', '2026-08-18_16-06-53',
  '--checkpoint', 'model_560.pt',
  '--reset_optimizer',
  'agent.algorithm.entropy_coef=0.005'
)
$p = Start-Process -FilePath "$envRoot\python.exe" `
  -WorkingDirectory '<REPO>\RL' `
  -ArgumentList $argList `
  -RedirectStandardOutput '<REPO>\RL\logs\train_X_ent005.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_X_ent005.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
