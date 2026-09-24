# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# Phase 0 wiring gate for the restitution campaign: reproduce the felt-pad drop
# test (200 mm release -> 75-85 mm rebound at e = 0.63) inside Isaac Lab.
#
# Runs probe_foot_drop.py, which spawns the training ground plane and the real
# robot asset with the training's own foot-material spawner, reads every
# contact/physics knob back off the live stage, and drops the foot.
#
#   powershell -File RL/run_probe_foot_drop.ps1                        # e = 0.63, expect 70-90 mm
#   powershell -File RL/run_probe_foot_drop.ps1 -Rest 0.0              # expect ~0 mm
#   powershell -File RL/run_probe_foot_drop.ps1 -Rest 0.3              # expect ~18 mm
#   powershell -File RL/run_probe_foot_drop.ps1 -GroundCombine multiply # shows the zeroing
#   powershell -File RL/run_probe_foot_drop.ps1 -Arm sphere            # isolated ball
#   powershell -File RL/run_probe_foot_drop.ps1 -BounceThreshold 0.5   # training default, to compare
param(
  [double]$Rest = 0.63,
  [double]$DropHeight = 0.200,
  [double]$BounceThreshold = 0.05,
  [string]$GroundCombine = 'max',
  [string]$FootCombine = 'max',
  [string]$Arm = 'locked',
  [int]$Steps = 500,
  [int]$TraceEvery = 20,
  [string]$Tag = '',
  # RTX 4090 (GPU1). GPU2 carries two leftover sweep processes from the 09-11
  # elasticity sweep (seen 09-14); do not disturb them.
  [string]$GPU = 'GPU-<uuid>',
  [switch]$Headless
)
$ErrorActionPreference = 'Stop'

$envRoot = '<CONDA_ENV>'
$env:CUDA_VISIBLE_DEVICES = $GPU
$env:CARB_APP_PATH = '<ISAACLAB>\_isaac_sim\kit'
$env:EXP_PATH      = '<ISAACLAB>\_isaac_sim\apps'
$env:ISAAC_PATH    = '<ISAACLAB>\_isaac_sim'
$env:PYTHONPATH    = ';<ISAACLAB>\_isaac_sim\site'
$env:RESOURCE_NAME = 'IsaacSim'
$env:CONDA_PREFIX  = $envRoot
$env:PATH = "$envRoot;$envRoot\Library\mingw-w64\bin;$envRoot\Library\usr\bin;$envRoot\Library\bin;$envRoot\Scripts;$envRoot\bin;" + $env:PATH

$argList = @(
  '-u', 'scripts/rsl_rl/probe_foot_drop.py',
  '--task', 'Template-standup_curriculum-v0',
  '--num_envs', '1',
  '--restitution', "$Rest",
  '--drop_height', "$DropHeight",
  '--bounce_threshold', "$BounceThreshold",
  '--ground_combine', $GroundCombine,
  '--foot_combine', $FootCombine,
  '--arm', $Arm,
  '--steps', "$Steps",
  '--trace_every', "$TraceEvery"
)
if ($Tag -ne '') { $argList += @('--tag', $Tag) }
# The probe prints its read-back block to stdout; the viewer is a distraction
# unless the user asks for it.
if (-not $Headless) { $argList += @('--headless') }

$errFile = '<REPO>\RL\logs\probe_foot_drop.err.log'
# The script path in $argList is relative to the RL workspace, same as the
# train/play launch scripts (they pass -WorkingDirectory to Start-Process).
Set-Location '<REPO>\RL'
# Isaac writes to stderr constantly; PS 5.1 would otherwise abort the run.
$ErrorActionPreference = 'Continue'
if ($env:FOOT_DROP_TEE -eq '1') {
  & "$envRoot\python.exe" @argList 2>&1 | Tee-Object -FilePath "$errFile.tee.log"
} else {
  & "$envRoot\python.exe" @argList
}
"exit=$LASTEXITCODE"
