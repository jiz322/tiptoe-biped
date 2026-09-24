# TipToe: play the released checkpoint (R1r_3150) in the Isaac Lab viewer (Windows / PowerShell).
#
# Loads control/r1r_3150.pt, the policy that stood on 2026-09-03, into the training env
# with the R1 plant recipe it was trained under (servo gains, armature, fences). Pushes and
# the kneel curriculum are off; the robot spawns standing and balances.
#
# Usage:  powershell -File RL\launch\play.ps1 [-Gpu 0] [-NumEnvs 16] [-Checkpoint <path>] [-Headless] [-Video]
#   -Video   record one mp4 (needs --enable_cameras; written under RL\logs\rsl_rl\...\videos)
param(
    [string]$Gpu = '0',
    [int]$NumEnvs = 16,
    [string]$Checkpoint = '',
    [switch]$Headless,
    [switch]$Video
)
$ErrorActionPreference = 'Stop'
$isaacSim = if ($env:ISAACSIM_PATH) { $env:ISAACSIM_PATH } else { throw 'set ISAACSIM_PATH (see docs/ENVIRONMENT.md)' }
$envRoot  = if ($env:CONDA_PREFIX)  { $env:CONDA_PREFIX }  else { throw 'activate the Isaac Lab conda env (CONDA_PREFIX unset)' }
$env:CUDA_VISIBLE_DEVICES = $Gpu
$env:CARB_APP_PATH = "$isaacSim\kit"
$env:EXP_PATH      = "$isaacSim\apps"
$env:ISAAC_PATH    = "$isaacSim"
$env:PYTHONPATH    = ";$isaacSim\site"
$env:RESOURCE_NAME = 'IsaacSim'
$env:PATH = "$envRoot;$envRoot\Library\mingw-w64\bin;$envRoot\Library\usr\bin;$envRoot\Library\bin;$envRoot\Scripts;$envRoot\bin;" + $env:PATH
$env:PYTHONUNBUFFERED = '1'

$RL = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $RL
if ($Checkpoint -eq '') { $Checkpoint = Resolve-Path (Join-Path $RL '..\control\r1r_3150.pt') }
if (-not (Test-Path -LiteralPath $Checkpoint)) { throw "checkpoint missing: $Checkpoint" }

$argList = @(
  '-u', 'scripts/rsl_rl/play.py',
  '--task', 'Template-standup_curriculum-v0',
  '--num_envs', "$NumEnvs",
  '--checkpoint', "$Checkpoint",
  # R1 plant recipe (legacy/launch/launch_train_R1_resume3110.ps1), inference-relevant part
  'env.sat_term_threshold=0.5', 'env.sat_term_tau_s=0.5',
  'env.servo_kp_hip_pitch=20.0', 'env.servo_kd_hip_pitch=1.4',
  'env.servo_kp_knee=30.0', 'env.servo_kd_knee=1.5',
  'env.hip_pitch_armature=0.0385',
  'env.swing_clearance_zero_m=0.078',
  'env.push_enable=false',
  'env.curriculum_enable=false',
  'env.pickup_enable=false'
)
if ($Headless) { $argList += '--headless' }
if ($Video)    { $argList += @('--video', '--video_length', '600', '--enable_cameras') }

$ErrorActionPreference = 'Continue'
& "$envRoot\python.exe" @argList
if ($LASTEXITCODE -ne 0) { Write-Output "[FAIL] exit=$LASTEXITCODE" }
