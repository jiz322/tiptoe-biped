# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# Debug probe for the ball-foot contact material: does the foot parameter reach
# the stage at all? See scripts/rsl_rl/probe_foot_material.py for what it prints.
#
#   powershell -File run_probe_foot_material.ps1                          # k=100.1 d=10.1z r=1.0 (the run that "looked the same")
#   powershell -File run_probe_foot_material.ps1 -Stiffness 0 -Rest 1.0   # rigid + restitution
#   powershell -File run_probe_foot_material.ps1 -OnlyStage               # no physics probe
param(
  [double]$Stiffness = 100.1,
  [double]$DampingRatio = 10.1,
  [double]$Restitution = 1.0,
  [switch]$OnlyStage,
  [switch]$ResetAB,
  [switch]$WriterCheck,
  [switch]$ContactSweep,
  [switch]$SceneStyle
)
$ErrorActionPreference = 'Stop'

$envRoot = '<CONDA_ENV>'
$env:CUDA_VISIBLE_DEVICES = '0'                 # RTX 4090, free as of writing
$env:CARB_APP_PATH = '<ISAACLAB>\_isaac_sim\kit'
$env:EXP_PATH      = '<ISAACLAB>\_isaac_sim\apps'
$env:ISAAC_PATH    = '<ISAACLAB>\_isaac_sim'
$env:PYTHONPATH    = ';<ISAACLAB>\_isaac_sim\site'
$env:RESOURCE_NAME = 'IsaacSim'
$env:CONDA_PREFIX  = $envRoot
$env:PATH = "$envRoot;$envRoot\Library\mingw-w64\bin;$envRoot\Library\usr\bin;$envRoot\Library\bin;$envRoot\Scripts;$envRoot\bin;" + $env:PATH
$env:PYTHONUNBUFFERED = '1'

$RL = '<REPO>\RL'
Set-Location $RL
$outDir = Join-Path $RL 'logs\foot_material_probe'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$log = Join-Path $outDir "probe_$stamp.log"

$args = @('-u', "$RL\scripts\rsl_rl\probe_foot_material.py", '--headless',
          '--stiffness', "$Stiffness", '--damping_ratio', "$DampingRatio",
          '--restitution', "$Restitution", '--trace_every', '0')
if ($OnlyStage) { $args += '--no_drop' }
if ($ResetAB)   { $args += '--reset_ab' }
if ($WriterCheck) { $args += '--writer_check' }
if ($ContactSweep) { $args += '--contact_sweep' }
if ($SceneStyle) { $args += '--scene_style' }

$ErrorActionPreference = 'Continue'   # Isaac writes to stderr; PS 5.1 would otherwise abort
& "$envRoot\python.exe" @args 2>&1 | Tee-Object -FilePath $log
$code = $LASTEXITCODE
Write-Output "[probe] exit=$code  log=$log"
exit $code
