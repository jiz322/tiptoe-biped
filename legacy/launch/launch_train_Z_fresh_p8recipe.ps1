# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# P9 arm Z on GPU 1: FROM-SCRATCH, full P8 training recipe on the P9 plant.
# Rationale: no fresh P9-plant run has ever used the recipe P8 actually
# converged with. The killed 08-18 A/B ran entropy 0.003 (polish leftover),
# gait_ref 1.0, tight sat fence — and stalled at active_frac 0.71. P8-fresh
# ran 0.005 / 3.0 / 0.5+0.5s and declined steadily from iter 0. Z replicates
# the P8 recipe exactly, changing only the plant (the thing P9 exists to
# test), on the post-explosion clamped actuator.
# Complements arm X (GPU0): X asks "can the plateau policy escape";
# Z asks "does the proven recipe converge on the new plant at all".
# NOTE: overrides are CLI-only — a resume must re-pass all four.
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
  '--run_name', 'Z_fresh_p8recipe',
  'agent.algorithm.entropy_coef=0.005',
  'env.rew_scale_gait_ref=3.0',
  'env.sat_term_threshold=0.5',
  'env.sat_term_tau_s=0.5'
)
$p = Start-Process -FilePath "$envRoot\python.exe" `
  -WorkingDirectory '<REPO>\RL' `
  -ArgumentList $argList `
  -RedirectStandardOutput '<REPO>\RL\logs\train_Z_fresh_p8recipe.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_Z_fresh_p8recipe.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
