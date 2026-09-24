# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# S6 Phase-A verification probe — resample check + runtime bounceThreshold.
# See scripts/rsl_rl/probe_s6_restitution.py.  NO training: gym.make() only,
# no learner, nothing saved.  The S6 overrides are passed so the probe reads
# the env's OWN DR path (carrier + band), matching the smoke config.
$ErrorActionPreference = 'Stop'

$envRoot = '<CONDA_ENV>'
$env:CUDA_VISIBLE_DEVICES = 'GPU-<uuid>'   # RTX 4090 (index 0)
$env:CARB_APP_PATH = '<ISAACLAB>\_isaac_sim\kit'
$env:EXP_PATH      = '<ISAACLAB>\_isaac_sim\apps'
$env:ISAAC_PATH    = '<ISAACLAB>\_isaac_sim'
$env:PYTHONPATH    = ';<ISAACLAB>\_isaac_sim\site'
$env:RESOURCE_NAME = 'IsaacSim'
$env:CONDA_PREFIX  = $envRoot
$env:PATH = "$envRoot;$envRoot\Library\mingw-w64\bin;$envRoot\Library\usr\bin;$envRoot\Library\bin;$envRoot\Scripts;$envRoot\bin;" + $env:PATH

$RL = '<REPO>\RL'
Set-Location $RL

$argList = @(
  '-u', "$RL\scripts\rsl_rl\probe_s6_restitution.py",
  '--headless',
  '--task=Template-standup_curriculum-v0',
  '--num_envs', '10',
  '--steps_per_iter', '24',
  '--iters', '3',
  # S6 overrides (match launch_train_S6_restDR.ps1 Phase A)
  'env.foot_restitution=0.63',
  'env.dr_foot_restitution_min=0.4',
  'env.dr_foot_restitution_max=0.8',
  'env.dr_foot_restitution_buckets=64',
  'env.pickup_enable=false',
  'env.curriculum_enable=true'
)

$ErrorActionPreference = 'Continue'   # Isaac writes to stderr; PS 5.1 would otherwise abort
& "$envRoot\python.exe" @argList 2>&1
$code = $LASTEXITCODE
Write-Output "[s6-probe] exit=$code"
exit $code
