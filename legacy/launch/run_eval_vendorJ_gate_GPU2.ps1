# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# Vendor-rotor replica gate (2026-09-18) — GPU 2 (GPUs 0/1 carry S8 / S7).
#
# Runs the runbook §0.2 gate (eval.py hw_matched, DR off, pickup off) on ONE
# checkpoint twice:
#   row A  "trainedplant" — S4a model_7780 on the plant it trained on
#          (hip 0.0385 / knee 0.036 pins, legacy per-joint armature draw)
#   row B  "vendorJ"      — the same checkpoint on the DERIVED vendor plant
#          (no armature pins: knee 7.81e-3 / hip 8.40e-3 / off-diag 5.72e-3)
# Both rows re-pass the S4a servo gains and the path-mode stand-up recipe the
# checkpoint was trained with, with the kneel curriculum OFF so the gate is a
# standing-survival number, not a stand-up number.
#
# WHAT IT ANSWERS: whether the POLICY tolerates the plant shift (row B survival
# vs row A).  It does NOT say which plant is real — that arbiter is the
# hardware kp-sweep: the vendor hip plant rings at 4.88 Hz under the kp-20 step
# battery vs 3.06-3.34 Hz measured (runbook 1.1), already contradicted on paper.
#
# PRE-REGISTERED READING: row A survival ~0.99+ (its published hw_matched
# numbers); row B >= 0.95 -> policy plant-insensitive at this size of change;
# row B < 0.80 -> the policy was leaning on the heavy rotor (armature was doing
# servo work), which is itself a sim2real finding.  Neither outcome is a
# result until the json exists and is pasted back.
#
# ONE SCENARIO PER PROCESS.  Resumable: existing .json = skip.
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
$outDir = Join-Path $RL 'logs\eval_vendorJ_gate'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$CKPT = 'logs/rsl_rl/p9_replant/2026-09-09_22-13-40_S4a_measDR/model_7780.pt'

# S4a recipe items that shape the plant/servo (from launch_train_S4a_measDR.ps1
# via S7); hw_matched already pins DR to central values and zeroes delays.
$BASE_OVR = @(
  'servo_kp_hip_pitch=20.0', 'servo_kd_hip_pitch=1.4',
  'servo_kp_knee=30.0',      'servo_kd_knee=1.5',
  'sat_term_threshold=0.5',  'sat_term_tau_s=0.5',
  'standup_ramp_mode=path',
  'curriculum_enable=false', 'kneel_start_probability=0.0',
  'pickup_enable=false', 'pickup_initial_probability=0.0', 'pickup_probability=0.0'
)
# row A: the plant S4a trained on (see docs/armature-derivation-note-20260918.md)
$TRAINED_PINS = @(
  'hip_pitch_armature=0.0385', 'knee_armature=0.036', 'dr_armature_per_leg=false'
)

$jobs = @(
  @{ Name = 'S4a_7780__hw_matched_trainedplant'; Ovr = $BASE_OVR + $TRAINED_PINS }
  @{ Name = 'S4a_7780__hw_matched_vendorJ';      Ovr = $BASE_OVR }
)

foreach ($j in $jobs) {
  $tag  = $j.Name
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
    '--checkpoint', $CKPT,
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
    # read back the plant the env actually built — the [ARMATURE] line must
    # say [pinned] 38.500e-3 for row A and [derived] 8.396e-3 for row B
    Select-String -Path $log -Pattern '\[ARMATURE\]' | Select-Object -First 2 | ForEach-Object { Write-Output "        $($_.Line)" }
  }
}
Write-Output "[all ] vendorJ gate finished $(Get-Date -Format HH:mm:ss)"
