# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# P9 checkpoint-selection eval sweep on GPU 2 (4060 Ti — GPUs 0/1 are
# running K4b / K4b2 training; this must NOT touch them).
#
# Purpose (2026-08-23): mean-policy comparison for the P11 deploy pick.
# Training-curve diagnostics/action_jerk_abs_mean is noise-dominated —
# checkpoint noise stds: P8 line sigma ~0.11-0.15 (=> noise-floor jerk
# 1.95*sigma ~ 0.21-0.29), K line sigma ~0.37-0.43 (=> ~0.72-0.84).
# The observed 0.3-vs-0.7 gap is the exploration noise, not the policy;
# eval.py runs the DETERMINISTIC mean policy, which is what deploys.
#
# ONE SCENARIO PER PROCESS: Isaac Sim tolerates a single simulation
# context per process — eval.py with >1 scenario hangs at the second
# env build (observed 08-23: 2 h spin at 'Scenario: hw_harsh_arm13',
# 1 core busy, 0% GPU). So each (checkpoint, scenario) pair gets its
# own python invocation and its own <name>__<scenario>.json.
#
# Scenarios (see eval.py):
#   hw_matched      — runbook §0.2 replica gate: single plant, central
#                     measured values (mass +5.9%), DR pinned. PASS =
#                     survives >=6 s majority of trials.
#   hw_harsh_arm13  — §0.2 harsh corner: armature ×1.3 (closest to the
#                     real hip per the 08-20 tether-bias calc).
#   balance_full_dr — training-range DR robustness (deploy candidates
#                     + G2i hardware reference only — startup ~2.5 min
#                     per invocation, so historical rows skip it).
#
# Checkpoints: K-line candidates get their launch-script overrides
# re-passed (servo 20/1.4, 30/1.5 + sat fence) — without them eval
# builds the WRONG plant (cfg default kp 8). P8 references (F3490,
# G2i4760 = the policies that actually flew) run at cfg defaults, which
# match their trained gains (stiffness 8/0.5 pre-P10); their plant is
# still today's replant values — that's the deploy-relevant frame, and
# the same frame in which G2i died on hardware at 0.25 s.
#
# Results: RL/logs/eval_p9sel/<name>__<scenario>.json + .out.log. A
# pair whose .json already exists is skipped, so the sweep is resumable
# — delete a .json to re-run that pair. Aggregate with
# scripts/rsl_rl/aggregate_eval_p9sel.py.
$ErrorActionPreference = 'Stop'

$envRoot = '<CONDA_ENV>'
$env:CUDA_VISIBLE_DEVICES = '2'
$env:CARB_APP_PATH = '<ISAACLAB>\_isaac_sim\kit'
$env:EXP_PATH      = '<ISAACLAB>\_isaac_sim\apps'
$env:ISAAC_PATH    = '<ISAACLAB>\_isaac_sim'
$env:PYTHONPATH    = ';<ISAACLAB>\_isaac_sim\site'
$env:RESOURCE_NAME = 'IsaacSim'
$env:CONDA_PREFIX  = $envRoot
$env:PATH = "$envRoot;$envRoot\Library\mingw-w64\bin;$envRoot\Library\usr\bin;$envRoot\Library\bin;$envRoot\Scripts;$envRoot\bin;" + $env:PATH

$RL = '<REPO>\RL'
Set-Location $RL
$outDir = Join-Path $RL 'logs\eval_p9sel'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

# K-line launch-script overrides (dynamics/termination only; reward
# scales are irrelevant to eval metrics)
$K_OVR = @(
  'servo_kp_hip_pitch=20.0', 'servo_kd_hip_pitch=1.4',
  'servo_kp_knee=30.0',      'servo_kd_knee=1.5',
  'sat_term_threshold=0.5',  'sat_term_tau_s=0.5'
)
# P8 recipe (Z-script provenance): sat fence 0.5/0.5, gains = cfg defaults
$P8_OVR = @('sat_term_threshold=0.5', 'sat_term_tau_s=0.5')

$HW   = @('hw_matched', 'hw_harsh_arm13')
$FULL = @('hw_matched', 'hw_harsh_arm13', 'balance_full_dr')

# Deploy candidates first (K4_3420 = prime, K4b_3600 = established pick,
# K5_4000/K5b_4100 = 08-24 polish anneals sigma 0.36→0.26), then backups
# (K4b_3500 / K4b2_3460 = cleanest post-patch, K3s_3360 = last clean
# fallback-line iter), then context rows. K3s_3380 kept deliberately: TB
# shows vloss 128k / rawmax 120 there — its eval quantifies what deploying
# a just-past-collapse checkpoint would look like.
$jobs = @(
  @{ Name = 'K5_4000';   Ovr = $K_OVR;  Scen = $FULL
     Ckpt = 'logs/rsl_rl/p9_replant/2026-08-23_17-47-23_K5_polish/model_4000.pt' }
  @{ Name = 'K5b_4100';  Ovr = $K_OVR;  Scen = $FULL
     Ckpt = 'logs/rsl_rl/p9_replant/2026-08-23_17-47-24_K5b_polish_seed45/model_4100.pt' }
  @{ Name = 'K4b_3600';  Ovr = $K_OVR;  Scen = $FULL
     Ckpt = 'logs/rsl_rl/p9_replant/2026-08-23_08-56-43_K4b_resume3420/model_3600.pt' }
  @{ Name = 'K4_3420';   Ovr = $K_OVR;  Scen = $FULL
     Ckpt = 'logs/rsl_rl/p9_replant/2026-08-22_17-26-54_K4_smooth024/model_3420.pt' }
  @{ Name = 'K4b_3500';  Ovr = $K_OVR;  Scen = $FULL
     Ckpt = 'logs/rsl_rl/p9_replant/2026-08-23_08-56-43_K4b_resume3420/model_3500.pt' }
  @{ Name = 'K4b2_3460'; Ovr = $K_OVR;  Scen = $FULL
     Ckpt = 'logs/rsl_rl/p9_replant/2026-08-23_11-15-49_K4b2_seed45/model_3460.pt' }
  @{ Name = 'K3s_3360';  Ovr = $K_OVR;  Scen = $FULL
     Ckpt = 'logs/rsl_rl/p9_replant/2026-08-21_21-44-12_K3s_zero078_clr4/model_3360.pt' }
  @{ Name = 'K4b_3560';  Ovr = $K_OVR;  Scen = $HW
     Ckpt = 'logs/rsl_rl/p9_replant/2026-08-23_08-56-43_K4b_resume3420/model_3560.pt' }
  @{ Name = 'K4b2_3520'; Ovr = $K_OVR;  Scen = $HW
     Ckpt = 'logs/rsl_rl/p9_replant/2026-08-23_11-15-49_K4b2_seed45/model_3520.pt' }
  @{ Name = 'K3s_2820';  Ovr = $K_OVR;  Scen = $HW
     Ckpt = 'logs/rsl_rl/p9_replant/2026-08-21_21-44-12_K3s_zero078_clr4/model_2820.pt' }
  @{ Name = 'K3s_3380';  Ovr = $K_OVR;  Scen = $HW
     Ckpt = 'logs/rsl_rl/p9_replant/2026-08-21_21-44-12_K3s_zero078_clr4/model_3380.pt' }
  @{ Name = 'G2i_4760';  Ovr = $P8_OVR; Scen = $FULL
     Ckpt = 'logs/rsl_rl/p8_belt_fresh/2026-08-12_08-07-50_G2_delay14_imuoffset010/model_4760.pt' }
  @{ Name = 'F_3490';    Ovr = $P8_OVR; Scen = $HW
     Ckpt = 'logs/rsl_rl/p8_belt_fresh/2026-08-10_21-00-21_F_smooth/model_3490.pt' }
)

foreach ($j in $jobs) {
  foreach ($scen in $j.Scen) {
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
    ) + $j.Ovr
    & "$envRoot\python.exe" @argList *> $log
    if ($LASTEXITCODE -ne 0) {
      Write-Output "[FAIL] $tag  exit=$LASTEXITCODE  see $log"
    } else {
      Write-Output "[done] $tag  $(Get-Date -Format HH:mm:ss)"
    }
  }
}
Write-Output "[all ] sweep finished $(Get-Date -Format HH:mm:ss)"
