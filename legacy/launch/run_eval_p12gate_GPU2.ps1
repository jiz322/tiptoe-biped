# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# P12 desk 0.2 — pickup-free acceptance gate on GPU 2 (GPUs 0/1 are
# running L1/L2 training; do NOT touch them).
#
# Runs eval.py hw_matched with the pickup curriculum OFF
# (pickup_initial_probability=0 pickup_probability=0) — see
# p11-first-flight-verdict correction: with pickup on, flight/z/pg
# numbers are contaminated by operator-lift rides.
#
# Jobs: L1 + L2 latest checkpoints (L overrides = K_OVR +
# hip_pitch_armature=0.0385 — the eval trap: without it the L policies
# are evaluated on the K plant), plus K4b_3600 rerun pickup-free as the
# apples-to-apples baseline (its published numbers were pickup-on).
#
# eval.py was extended 08-25 with tau_abs_mean/*, qd_rms/*, sat095/*,
# var/pg*_std, settled/* — the gate reads settled/* directly from the
# JSON now (no npz post-processing needed; traj npz still dumped for
# spectra work).
#
# PASS bar (l1-calmidle-arm): survival 1.000 AND settled/base_z_std
# < 0.003 AND settled/sat095_all < 0.04 AND settled/pg_x_std <= 0.04
# AND settled/flight_fraction < 0.05 AND TB idle_liftoff >= 0.03.
# ONE SCENARIO PER PROCESS. Resumable: existing .json = skip.
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
$outDir = Join-Path $RL 'logs\eval_p12gate'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$K_OVR = @(
  'servo_kp_hip_pitch=20.0', 'servo_kd_hip_pitch=1.4',
  'servo_kp_knee=30.0',      'servo_kd_knee=1.5',
  'sat_term_threshold=0.5',  'sat_term_tau_s=0.5'
)
$L_OVR = $K_OVR + @('hip_pitch_armature=0.0385')
$PF = @('pickup_initial_probability=0.0', 'pickup_probability=0.0')

$jobs = @(
  @{ Name = 'L1_4330'; Ovr = $L_OVR + $PF
     Ckpt = 'logs/rsl_rl/p9_replant/2026-08-24_11-57-00_L1_calmidle/model_4330.pt' }
  @{ Name = 'L2_3770'; Ovr = $L_OVR + $PF
     # L2 slow-burn collapse from ~3850 (reward 16.4k@3773 -> 4.3k@4381,
     # vloss spikes 2e5, rawmax 150 UNDER the lr cap) — eval the
     # pre-degradation peak, never the latest ckpt.
     Ckpt = 'logs/rsl_rl/p9_replant/2026-08-24_12-20-45_L2_gaitref10/model_3770.pt' }
  @{ Name = 'K4b_3600_pf'; Ovr = $K_OVR + $PF
     Ckpt = 'logs/rsl_rl/p9_replant/2026-08-23_08-56-43_K4b_resume3420/model_3600.pt' }
)

foreach ($j in $jobs) {
  $tag  = "$($j.Name)__hw_matched_pf"
  $log  = Join-Path $outDir "$tag.out.log"
  $json = Join-Path $outDir "$tag.json"
  $traj = Join-Path $outDir "$tag`_traj.npz"
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
    '--scenarios', 'hw_matched',
    '--dump_traj', $traj,
    '--dump_envs', '4',
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
Write-Output "[all ] gate sweep finished $(Get-Date -Format HH:mm:ss)"
