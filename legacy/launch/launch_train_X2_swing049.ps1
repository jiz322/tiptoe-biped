# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# P9 arm X2 on GPU 0: X + the swing-clearance zero fix (the discriminator).
# Evidence (08-19): every run that ever went under active_frac 0.71 satisfied
# the cfg rule swing_clearance_zero_m = idle_toe_z_p10 + 0.003; every run
# that violated it (P8 first-fresh, all P9 runs) parked at >=0.71. P9's
# stance is p10 ~0.046 vs zero 0.075 (+29mm) -> the anti-shuffle swing
# reward has zero gradient from resting stance. X ran 207 iters of
# entropy-0.005 alone and stayed flat 0.71 -> entropy necessary-not-
# sufficient. X2 resumes X's own model_760 (keeps the re-inflated std)
# and sets zero = 0.046 + 0.003 = 0.049.
# NOTE: CLI-only overrides — any resume must re-pass BOTH.
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
  '--run_name', 'X2_ent005_swing049',
  '--resume',
  '--load_run', '2026-08-19_08-48-06_X_ent005_r560',
  '--checkpoint', 'model_760.pt',
  'agent.algorithm.entropy_coef=0.005',
  'env.swing_clearance_zero_m=0.049'
)
$p = Start-Process -FilePath "$envRoot\python.exe" `
  -WorkingDirectory '<REPO>\RL' `
  -ArgumentList $argList `
  -RedirectStandardOutput '<REPO>\RL\logs\train_X2_swing049.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_X2_swing049.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
