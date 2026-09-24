# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# Gate for the per-env ball-pad contact model (pad_contact.PadContact).
# See scripts/rsl_rl/probe_pad_contact.py for what each mode prints.
#
# Run the three modes in this order; each one is cheap and each answers a
# question the next one depends on.
#
#   powershell -File run_probe_pad_contact.ps1 -Mode geometry_only
#       Is the ball centre offset measured correctly? This is the only part of
#       the model that reads the asset, and a wrong contact point makes every
#       later number meaningless. Compares the measured offset against the env's
#       own toe offset (documented 5.8 cm short of the ball).
#
#   powershell -File run_probe_pad_contact.ps1 -Mode sag_ab
#       Does a pinned stiffness move the plant? PD-holds the stand pose with the
#       pad APPLIED to the sim, at k = 0 / 5000 / 20000 / 50000. A live pad
#       settles lower as k drops and shows a smaller peak force; identical rows
#       mean the balls never loaded in this pose.
#
#   powershell -File run_probe_pad_contact.ps1 -Mode env_sweep -PadEnable
#       Does the ENV's own DR path reach the model? Pins the band through
#       RlEnv._pad and reads it back, so a broken draw or a broken per-substep
#       write shows up as a column that does not track.
#
# -Mode drop_ab is the same as sag_ab but with a 0.55 m release, for the impact
# transient rather than the settle.
param(
  [ValidateSet('geometry_only', 'sag_ab', 'drop_ab', 'env_sweep')]
  [string]$Mode = 'geometry_only',
  [double[]]$Stiffness = @(0.0, 5000.0, 20000.0, 50000.0),
  [double]$DampingRatio = 0.7,
  [double]$EffectiveMass = 1.0,
  [double]$PadThickness = 0.002,
  [int]$Steps = 400,
  [int]$NumEnvs = 1,
  [switch]$PadEnable
)
$ErrorActionPreference = 'Stop'

$envRoot = '<CONDA_ENV>'
$env:CUDA_VISIBLE_DEVICES = '0'                 # RTX 4090
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
$outDir = Join-Path $RL 'logs\pad_contact_probe'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$log = Join-Path $outDir "pad_$stamp.log"

$argList = @('-u', "$RL\scripts\rsl_rl\probe_pad_contact.py", '--headless',
             '--task=Template-standup_curriculum-v0',
             '--mode', $Mode,
             '--num_envs', "$NumEnvs",
             '--damping_ratio', "$DampingRatio",
             '--effective_mass', "$EffectiveMass",
             '--pad_thickness', "$PadThickness",
             '--steps', "$Steps",
             '--stiffness')
$argList += ($Stiffness | ForEach-Object { "$_" })

# env_sweep reads the env's OWN pad, which only exists when the cfg asks for one.
# The band is pinned to the first arm so the per-episode draw has something to
# land on; the probe then overrides per arm by hand.
if ($Mode -eq 'env_sweep' -or $PadEnable) {
  $kMin = $Stiffness | Measure-Object -Minimum | Select-Object -ExpandProperty Minimum
  $kMax = $Stiffness | Measure-Object -Maximum | Select-Object -ExpandProperty Maximum
  $argList += @('env.pad_contact_enable=true',
                "env.pad_stiffness_min=$kMin",
                "env.pad_stiffness_max=$kMax",
                "env.pad_damping_ratio=$DampingRatio",
                "env.pad_effective_mass=$EffectiveMass",
                "env.pad_thickness=$PadThickness")
}

Write-Output "[pad-probe] mode=$Mode stiffness=$($Stiffness -join ',')"
$ErrorActionPreference = 'Continue'   # Isaac writes to stderr; PS 5.1 would otherwise abort
& "$envRoot\python.exe" @argList 2>&1 | Tee-Object -FilePath $log
$code = $LASTEXITCODE
Write-Output "[pad-probe] exit=$code  log=$log"
exit $code
