# Launcher for script\show_kneel.py (kneeling pose viewer, GUI).
#
# Plain `python show_kneel.py` fails with "No module named 'pxr'": pxr/omni
# only exist inside the Kit runtime, which needs the Isaac Sim binary-install
# env vars below (same recipe as RL\run_*.ps1) before SimulationApp starts.
#
# Usage:  powershell -File script\run_show_kneel.ps1
$ErrorActionPreference = 'Stop'

# Set these for your machine (or export them before calling):
#   ISAACSIM_PATH = Isaac Sim binary install (the folder that contains kit\ and apps\)
#   CONDA_PREFIX  = the conda env with Isaac Lab + this repo installed (set by `conda activate`)
$isaacSim = if ($env:ISAACSIM_PATH) { $env:ISAACSIM_PATH } else { throw 'set ISAACSIM_PATH (see docs/ENVIRONMENT.md)' }
$envRoot  = if ($env:CONDA_PREFIX)  { $env:CONDA_PREFIX }  else { throw 'activate the Isaac Lab conda env (CONDA_PREFIX unset)' }
# Select the 4090 by UUID (enumeration shifted once after the 08-27 GPU1 bus drop)
$env:CUDA_VISIBLE_DEVICES = 'GPU-<uuid>'
$env:CARB_APP_PATH = "$isaacSim\kit"
$env:EXP_PATH      = "$isaacSim\apps"
$env:ISAAC_PATH    = "$isaacSim"
$env:PYTHONPATH    = ";$isaacSim\site"
$env:RESOURCE_NAME = 'IsaacSim'
$env:CONDA_PREFIX  = $envRoot
$env:PATH = "$envRoot;$envRoot\Library\mingw-w64\bin;$envRoot\Library\usr\bin;$envRoot\Library\bin;$envRoot\Scripts;$envRoot\bin;" + $env:PATH

# Relax around the native call: PS 5.1 + Stop + stderr = NativeCommandError trap
$ErrorActionPreference = 'Continue'
& "$envRoot\python.exe" -u "$PSScriptRoot\show_kneel.py"
if ($LASTEXITCODE -ne 0) { Write-Output "[FAIL] exit=$LASTEXITCODE" }
