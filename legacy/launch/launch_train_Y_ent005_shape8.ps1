# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# P9 aggressive A/B, arm Y on GPU 1: exploration + P8 shaping restoration.
# Same plateau resume as arm X (A's model_560) plus the three shaping values
# the successful P8 fresh recipe used and P9 tightened/weakened:
#   entropy_coef        0.003 -> 0.005  (same as X)
#   rew_scale_gait_ref  1.0   -> 3.0    (P8's stronger gait guidance)
#   sat_term_threshold  0.35  -> 0.5    (knee fence the P9 runs ride at 0.33-0.37)
#   sat_term_tau_s      0.3   -> 0.5    (P8's slower saturation EMA)
# threshold_hip stays 0.55 (rides but rarely fires; sat_term_frac ~1e-4).
# Attribution vs X: Y-only breakout => shaping/fence was binding;
# both break => entropy sufficed; neither => plant itself, go after the
# pickup curriculum next, not PPO.
# NOTE: all overrides are CLI-only — a future resume must re-pass them.
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
  '--run_name', 'Y_ent005_shape8_r560',
  '--resume',
  '--load_run', '2026-08-18_16-06-53',
  '--checkpoint', 'model_560.pt',
  '--reset_optimizer',
  'agent.algorithm.entropy_coef=0.005',
  'env.rew_scale_gait_ref=3.0',
  'env.sat_term_threshold=0.5',
  'env.sat_term_tau_s=0.5'
)
$p = Start-Process -FilePath "$envRoot\python.exe" `
  -WorkingDirectory '<REPO>\RL' `
  -ArgumentList $argList `
  -RedirectStandardOutput '<REPO>\RL\logs\train_Y_ent005_shape8.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_Y_ent005_shape8.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
