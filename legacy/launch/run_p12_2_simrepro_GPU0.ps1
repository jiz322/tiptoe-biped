# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# P12.2 sim2real fall investigation — headless reproduction rollouts (2026-08-29).
#
# Runs play_sim2real.py with --autolog_steps (new flag, 08-29): B/H cycle is
# driven automatically, one npz per run, saved to the RL dir as
# model_3150_simlog_<ts>.npz. Each run = fresh reset (clean handover pose,
# gait phase pinned 0) logged for 300 ticks (3 s); episode auto-resets on a
# fall are captured inside the log (split post-hoc on gait-phase snaps).
#
# ARM is the injection under test. Baseline arm = the user's 08-28/29 repro
# condition verbatim (pin 0.11 + swing 0.069 + friction 0.6, NO
# hip_pitch_armature override — note R1r trained with 0.0385, cfg default
# 0.0332; that confound is a separate arm).
#
# Usage:  powershell -File run_p12_2_simrepro_GPU0.ps1 [-Arm baseline]
param([string]$Arm = 'baseline')
$ErrorActionPreference = 'Stop'

$envRoot = '<CONDA_ENV>'
# Select the 4090 by UUID (enumeration shifted once after the 08-27 GPU1 bus drop)
$env:CUDA_VISIBLE_DEVICES = 'GPU-<uuid>'
$env:CARB_APP_PATH = '<ISAACLAB>\_isaac_sim\kit'
$env:EXP_PATH      = '<ISAACLAB>\_isaac_sim\apps'
$env:ISAAC_PATH    = '<ISAACLAB>\_isaac_sim'
$env:PYTHONPATH    = ';<ISAACLAB>\_isaac_sim\site'
$env:RESOURCE_NAME = 'IsaacSim'
$env:CONDA_PREFIX  = $envRoot
$env:PATH = "$envRoot;$envRoot\Library\mingw-w64\bin;$envRoot\Library\usr\bin;$envRoot\Library\bin;$envRoot\Scripts;$envRoot\bin;" + $env:PATH

$RL = '<REPO>\RL'
Set-Location $RL
$outDir = Join-Path $RL "logs\p12_2_simrepro\$Arm"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$CKPT = '<REPO>\RL\logs\rsl_rl\p9_replant\2026-08-27_10-30-37_R1_resume3110\model_3150.pt'

# Overrides shared by all arms (the user's repro condition; friction and the
# pg_x pin split out as per-arm parameters)
$BASE_OVR = @(
  'env.swing_clearance_zero_m=0.069',
  'env.sat_term_threshold=0.5', 'env.sat_term_tau_s=0.5',
  'env.rew_scale_gait_ref=3.0'
)
$FRIC = 0.6
$PIN = 0.11
$HANDOVER = 'training'
switch ($Arm) {
  'baseline'  { $OVR = $BASE_OVR }   # ran 08-29 pre-fix = training contract (blind*5 start)
  'sighted'   { $OVR = $BASE_OVR }   # 08-29 intermediate fix = [real, blind*4] hybrid; superseded
  'deploy'    { $OVR = $BASE_OVR; $HANDOVER = 'deploy' }  # real*5 = hardware handover contract
  'armature'  { $OVR = $BASE_OVR + @('env.hip_pitch_armature=0.0385') }
  'fric04'    { $OVR = $BASE_OVR; $HANDOVER = 'deploy'; $FRIC = 0.4 }
  'fric03'    { $OVR = $BASE_OVR; $HANDOVER = 'deploy'; $FRIC = 0.3 }
  'fric03_blind' { $OVR = $BASE_OVR; $HANDOVER = 'training'; $FRIC = 0.3 }
  'fric02'    { $OVR = $BASE_OVR; $HANDOVER = 'deploy'; $FRIC = 0.2 }
  # Trim experiment (08-29 item 2): pin 0.013 = trim removed (raw hardware
  # bias at handover); compare against the 0.11-pin arms above (~= current
  # trim's +0.102 displayed).
  'pin0013_deploy_f06' { $OVR = $BASE_OVR; $HANDOVER = 'deploy';   $FRIC = 0.6; $PIN = 0.013 }
  'pin0013_deploy_f03' { $OVR = $BASE_OVR; $HANDOVER = 'deploy';   $FRIC = 0.3; $PIN = 0.013 }
  'pin0013_blind_f06'  { $OVR = $BASE_OVR; $HANDOVER = 'training'; $FRIC = 0.6; $PIN = 0.013 }
  'pin0013_blind_f03'  { $OVR = $BASE_OVR; $HANDOVER = 'training'; $FRIC = 0.3; $PIN = 0.013 }
  default     { throw "unknown arm '$Arm'" }
}
$OVR = $OVR + @("env.dr_floor_friction_min=$FRIC", "env.dr_floor_friction_max=$FRIC",
                "env.dr_proj_grav_bias_pin_x=$PIN")

$log = Join-Path $outDir 'run.out.log'
Write-Output "[run ] p12_2 simrepro arm=$Arm  $(Get-Date -Format HH:mm:ss)  log=$log"
# PS 5.1 + ErrorActionPreference Stop + redirected native stderr =
# NativeCommandError on the first stderr line (kills the run and eats the
# traceback body). Relax it around the long-running native call only.
$ErrorActionPreference = 'Continue'
& "$envRoot\python.exe" -u scripts\rsl_rl\play_sim2real.py `
  --task=Template-standup_curriculum-v0 `
  --checkpoint="$CKPT" `
  --gait_phase_init 0 `
  --seed=0 `
  --headless --no_obs_summary `
  --handover $HANDOVER `
  --autolog_steps 300 --autolog_runs 10 `
  @OVR *> $log
if ($LASTEXITCODE -ne 0) { Write-Output "[FAIL] exit=$LASTEXITCODE  see $log" }
else {
  # collect the npz files this run produced into the arm dir
  Get-ChildItem "$RL\model_3150_simlog_*.npz" -ErrorAction SilentlyContinue |
    Move-Item -Destination $outDir
  Write-Output "[done] $(Get-Date -Format HH:mm:ss)  npz in $outDir"
}
