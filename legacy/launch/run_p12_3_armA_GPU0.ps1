# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# P12.3 arm A — dead-zone state-injection rollouts (2026-08-31).
#
# Injects the hardware fork state (T2 @ 0.18 s: pitch ≈ −14°, wy ≈ +37°/s
# rebound, joints/vels from the log) into sim after every reset, via
# play_sim2real.py --inject_npz (new flag, 08-31). Two history contracts:
#   injA1 = --inject_history log        (K preceding logged frames + logged
#                                        last_actions — true counterfactual)
#   injA2 = --inject_history replicate  (injected frame ×5 — naive)
#
# PRE-REGISTERED PREDICTIONS (on record before first run):
#   A1: ≥7/10 rollouts fall backward (tilt crosses 25° and never re-enters,
#       topple within 0.5–2.5 s of injection) — the quasi-static dead zone is
#       real in sim too once the state is entered quasi-statically.
#   A2: same, ≥7/10 — the dead zone is carried by the current frame +
#       last_actions (fixed-point analysis 08-31), not by history diversity.
#   A1 falls & A2 recovers → dead zone is history-dependent (document).
#   Both recover ≥8/10   → dead-zone story WRONG; plant gap is bigger than
#                          state entry — say so plainly.
#   Caveat pre-stated: at inject_time 0.18 the five logged history frames
#   span the landing transient (0.14–0.18), so A1/A2 differ mainly in wy
#   history; limited discrimination is expected and acceptable.
#
# DELIBERATE DEVIATION from the p12_2 grid BASE_OVR: sat_term disabled
# (threshold 2.0, unreachable) so every termination is a physical fall or
# timeout — the dead-zone fixed point rails hips and could otherwise fire
# sat_term and be misclassified as a fall. Friction pinned 0.6 (grid showed
# μ-insensitivity 0.2–1.0), pg pin 0.11 (trim-on, matches the logged frames).
#
# Usage:  powershell -File run_p12_3_armA_GPU0.ps1 -Arm injA1
param([string]$Arm = 'injA1')
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
$outDir = Join-Path $RL "logs\p12_3_armA\$Arm"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$CKPT = '<REPO>\RL\logs\rsl_rl\p9_replant\2026-08-27_10-30-37_R1_resume3110\model_3150.pt'
$NPZ  = '<REPO>\data\p12_3_flight_20260831_v\r1r_3150_log_20260831_150252.npz'

$OVR = @(
  'env.swing_clearance_zero_m=0.069',
  'env.sat_term_threshold=2.0', 'env.sat_term_tau_s=0.5',
  'env.rew_scale_gait_ref=3.0',
  'env.dr_floor_friction_min=0.6', 'env.dr_floor_friction_max=0.6',
  'env.dr_proj_grav_bias_pin_x=0.11'
)
# vx sweep arms (08-31, after A1/A2 recovered 19/20): hardware does not log
# root lin vel; the base injection assumed quasi-static (vx=0). These arms
# find the residual backward COM velocity at which the injected state becomes
# unrecoverable — the video foot/torso-displacement measurement then decides
# whether hardware actually carried that much.
# PRE-REGISTERED: if sim only falls at |vx| >= 0.3 m/s (implausibly large for
# a quasi-static arrest; = 30 cm/s torso drift, video would show it plainly),
# the incomplete-state-entry alternative is dead and the gap is plant-side.
$VX = 0.0
$RUNS = 10
switch ($Arm) {
  'injA1' { $HIST = 'log' }
  'injA2' { $HIST = 'replicate' }
  'injA1_vx01' { $HIST = 'log'; $VX = -0.1; $RUNS = 5 }
  'injA1_vx02' { $HIST = 'log'; $VX = -0.2; $RUNS = 5 }
  'injA1_vx03' { $HIST = 'log'; $VX = -0.3; $RUNS = 5 }
  default { throw "unknown arm '$Arm'" }
}

$log = Join-Path $outDir 'run.out.log'
Write-Output "[run ] p12_3 armA arm=$Arm hist=$HIST  $(Get-Date -Format HH:mm:ss)  log=$log"
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
  --inject_npz "$NPZ" --inject_time 0.18 --inject_history $HIST --inject_vx $VX `
  --autolog_steps 300 --autolog_runs $RUNS `
  @OVR *> $log
if ($LASTEXITCODE -ne 0) { Write-Output "[FAIL] exit=$LASTEXITCODE  see $log" }
else {
  # collect the npz files this run produced into the arm dir
  Get-ChildItem "$RL\model_3150_simlog_*.npz" -ErrorAction SilentlyContinue |
    Move-Item -Destination $outDir
  Write-Output "[done] $(Get-Date -Format HH:mm:ss)  npz in $outDir"
}
