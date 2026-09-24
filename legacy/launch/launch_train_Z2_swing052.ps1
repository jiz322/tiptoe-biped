# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# P9 arm Z2 on GPU 1: Z (fresh P8-early recipe) + the swing-clearance zero
# fix, resuming Z's own model_160 to keep its 2h of progress. Z measured
# idle_toe_z_p10 ~0.0486 at iter 160 -> zero = 0.0486 + 0.003 ~= 0.052
# (same rule A_baseline applied at the equivalent phase: 0.052 + 0.003 =
# 0.055 — the arms the 08-07 lineage used to first break under 0.71).
# Keeps Z's other overrides (all CLI-only — a resume must re-pass ALL four).
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
  '--run_name', 'Z2_p8recipe_swing052',
  '--resume',
  '--load_run', '2026-08-19_09-49-15_Z_fresh_p8recipe',
  '--checkpoint', 'model_160.pt',
  'agent.algorithm.entropy_coef=0.005',
  'env.rew_scale_gait_ref=3.0',
  'env.sat_term_threshold=0.5',
  'env.sat_term_tau_s=0.5',
  'env.swing_clearance_zero_m=0.052'
)
$p = Start-Process -FilePath "$envRoot\python.exe" `
  -WorkingDirectory '<REPO>\RL' `
  -ArgumentList $argList `
  -RedirectStandardOutput '<REPO>\RL\logs\train_Z2_swing052.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_Z2_swing052.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
