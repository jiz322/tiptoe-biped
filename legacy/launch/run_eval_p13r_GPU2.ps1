# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# P13 R-line eval sweep on GPU 2 (2026-08-27; GPUs 0/1 are running the
# R1_resume3110 / R3_resume3680 trainings — this must NOT touch them).
#
# Candidates: R1_robust 3110 (pre-decline peak), R3_r1plus 3680 (peak),
# and the 08-27 tightened-PPO resumes R1r 3150 / R3r 3760. Baseline:
# K4b_3600 (the policy that flew 08-25) re-run under the exact same
# protocol so every comparison is apples-to-apples.
#
# Protocol changes vs run_eval_p9sel_GPU2.ps1 (the 08-23 sweep):
#   1. PICKUP-FREE everywhere (pickup_initial_probability=0,
#      pickup_probability=0) — p11-first-flight-verdict: pickup-on
#      contaminates flight/z/pg and even survival (K4b_3600 hw_matched
#      1.000 pickup-on -> 0.9971 pickup-free). All numbers from this
#      sweep are pickup-free truth; do NOT mix with the 08-23 table.
#   2. R rows re-pass hip_pitch_armature=0.0385 (their trained,
#      bench-corrected value; the L-line eval trap) on top of K_OVR.
#      K4b stays on its trained cfg-default 0.0332.
#   3. Two new scenarios (eval.py 08-27):
#        hw_push_recovery — hw_matched plant + trained-range pushes
#          (10-40 N, 10 steps, p=0.002/step ≈ 3 pushes / 15 s). The
#          pre-registered P13 gate scenario from the R1 launch header.
#        r1_train_dr — R1's full widened training conditions (plant-DR
#          box, delays 1-3, spawn-posture DR, exec-noise injection,
#          pushes). Same box for every row incl. K4b: measures what the
#          robustness bundle bought. NOTE: exec noise + wide spawns hit
#          K4b OOD — that asymmetry is the point of the probe.
#
# Gate bar (R1 launch header): pickup-free hw_matched survival >= K4b
# baseline (0.9971) + push recovery >= K4b.
#
# ONE SCENARIO PER PROCESS (Isaac Sim single-context limit). Resumable:
# existing .json = skip; delete a .json to re-run. ~25 pairs, expect
# ~60-70 min. Aggregate:
#   python scripts/rsl_rl/aggregate_eval_p9sel.py logs/eval_p13r
$ErrorActionPreference = 'Stop'

$envRoot = '<CONDA_ENV>'
# 08-27 16:00: GPU1 (4090 D) fell off the bus mid-sweep ("GPU is lost",
# needs reboot) which shifted CUDA enumeration and broke CUDA_VISIBLE_DEVICES=2
# — select the 4060 Ti by UUID instead (stable across enumeration changes).
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
$outDir = Join-Path $RL 'logs\eval_p13r'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

# Shared dynamics/termination overrides (reward scales irrelevant to eval)
$K_OVR = @(
  'servo_kp_hip_pitch=20.0', 'servo_kd_hip_pitch=1.4',
  'servo_kp_knee=30.0',      'servo_kd_knee=1.5',
  'sat_term_threshold=0.5',  'sat_term_tau_s=0.5'
)
# R line trained on the bench-corrected hip armature
$R_OVR = $K_OVR + @('hip_pitch_armature=0.0385')
# Pickup-free — applied to every job (scenarios never touch pickup keys)
$PF = @('pickup_initial_probability=0.0', 'pickup_probability=0.0')

# NOTE: collection name must differ from the loop variable — PowerShell
# variables are case-insensitive, so `foreach ($scen in $SCEN)` silently
# overwrites the array with the last element after the first job (burned
# the 08-27 first pass).
$SCENARIO_LIST = @('hw_matched', 'hw_harsh_arm13', 'hw_push_recovery',
                   'r1_train_dr', 'balance_full_dr')

$jobs = @(
  @{ Name = 'R1_3110';  Ovr = $R_OVR
     Ckpt = 'logs/rsl_rl/p9_replant/2026-08-25_14-54-09_R1_robust/model_3110.pt' }
  @{ Name = 'R3_3680';  Ovr = $R_OVR
     Ckpt = 'logs/rsl_rl/p9_replant/2026-08-26_09-09-50_R3_r1plus/model_3680.pt' }
  @{ Name = 'R1r_3150'; Ovr = $R_OVR
     Ckpt = 'logs/rsl_rl/p9_replant/2026-08-27_10-30-37_R1_resume3110/model_3150.pt' }
  @{ Name = 'R3r_3760'; Ovr = $R_OVR
     Ckpt = 'logs/rsl_rl/p9_replant/2026-08-27_10-30-37_R3_resume3680/model_3760.pt' }
  @{ Name = 'K4b_3600'; Ovr = $K_OVR
     Ckpt = 'logs/rsl_rl/p9_replant/2026-08-23_08-56-43_K4b_resume3420/model_3600.pt' }
)

foreach ($j in $jobs) {
  foreach ($scen in $SCENARIO_LIST) {
    $tag  = "$($j.Name)__$scen"
    $log  = Join-Path $outDir "$tag.out.log"
    $json = Join-Path $outDir "$tag.json"
    if (Test-Path $json) {
      Write-Output "[skip] $tag - json exists"
      continue
    }
    Write-Output "[run ] $tag  $(Get-Date -Format HH:mm:ss)"
    $argList = @(
      '-u', 'scripts/rsl_rl/eval.py',
      '--task', 'Template-standup_curriculum-v0',
      '--headless',
      '--checkpoint', $j.Ckpt,
      '--num_envs', '1024',
      '--max_steps', '1500',
      '--seed', '42',
      '--scenarios', $scen,
      '--out_json', $json,
      '--cfg_overrides'
    ) + $j.Ovr + $PF
    & "$envRoot\python.exe" @argList *> $log
    if ($LASTEXITCODE -ne 0) {
      Write-Output "[FAIL] $tag  exit=$LASTEXITCODE  see $log"
    } else {
      Write-Output "[done] $tag  $(Get-Date -Format HH:mm:ss)"
    }
  }
}
Write-Output "[all ] sweep finished $(Get-Date -Format HH:mm:ss)"
