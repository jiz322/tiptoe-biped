# Launcher for script\show_standup_path.py (stand-up path knot tester, GUI).
#
# Plain `python show_standup_path.py` fails with "No module named 'pxr'": pxr/omni
# only exist inside the Kit runtime, which needs the Isaac Sim binary-install
# env vars below (same recipe as RL\run_*.ps1) before SimulationApp starts.
#
# Usage:  powershell -File script\run_show_standup_path.ps1 [-Gpu 4060|4090] [-Smoke] [-Cfg <path>]
#   -Gpu    which card renders the viewer (default 4060 Ti: the 4090s are usually
#           full with training runs; physics is CPU either way)
#   -Smoke  headless self-test: spawn at every knot + one d=1.0 ramp, print reports, exit
#   -Cfg    alternative rl_env_cfg_p6.py to read standup_path_knots from
#   -Scan   headless spawn-consistency sweep over s, 'start:stop:step' (e.g. 0.0:0.2:0.02)
param(
    [string]$Gpu = '0',
    [switch]$Smoke,
    [string]$Cfg = '',
    [string]$Scan = ''
)
$ErrorActionPreference = 'Stop'

# Set these for your machine (or export them before calling):
#   ISAACSIM_PATH = Isaac Sim binary install (the folder that contains kit\ and apps\)
#   CONDA_PREFIX  = the conda env with Isaac Lab + this repo installed (set by `conda activate`)
$isaacSim = if ($env:ISAACSIM_PATH) { $env:ISAACSIM_PATH } else { throw 'set ISAACSIM_PATH (see docs/ENVIRONMENT.md)' }
$envRoot  = if ($env:CONDA_PREFIX)  { $env:CONDA_PREFIX }  else { throw 'activate the Isaac Lab conda env (CONDA_PREFIX unset)' }
$gpuUuid = @{
    '4090' = 'GPU-<uuid>'   # RTX 4090 (training GPU0)
    '4060' = 'GPU-<uuid>'   # RTX 4060 Ti
}[$Gpu]
if (-not $gpuUuid) { $gpuUuid = $Gpu }   # or pass a CUDA index / a GPU UUID from nvidia-smi -L
$env:CUDA_VISIBLE_DEVICES = $gpuUuid
$env:CARB_APP_PATH = "$isaacSim\kit"
$env:EXP_PATH      = "$isaacSim\apps"
$env:ISAAC_PATH    = "$isaacSim"
$env:PYTHONPATH    = ";$isaacSim\site"
$env:RESOURCE_NAME = 'IsaacSim'
$env:CONDA_PREFIX  = $envRoot
$env:PATH = "$envRoot;$envRoot\Library\mingw-w64\bin;$envRoot\Library\usr\bin;$envRoot\Library\bin;$envRoot\Scripts;$envRoot\bin;" + $env:PATH

$pyArgs = @()
if ($Smoke) { $pyArgs += '--smoke' }
if ($Cfg)   { $pyArgs += @('--cfg', $Cfg) }
if ($Scan)  { $pyArgs += @('--scan', $Scan) }

# Relax around the native call: PS 5.1 + Stop + stderr = NativeCommandError trap
$ErrorActionPreference = 'Continue'
& "$envRoot\python.exe" -u "$PSScriptRoot\show_standup_path.py" @pyArgs
if ($LASTEXITCODE -ne 0) { Write-Output "[FAIL] exit=$LASTEXITCODE" }
