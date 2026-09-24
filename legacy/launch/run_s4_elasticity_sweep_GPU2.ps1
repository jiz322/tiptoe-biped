# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# Fixed-policy virtual sweep of the two plant mechanisms left open by the
# 09-11 S4 stepping-gate analysis: belt series elasticity and ball-foot contact
# compliance/restitution. See scripts/rsl_rl/sweep_plant_elasticity.py for the
# question, the grid, and the PRE-REGISTERED readings P1-P4.
#
# WHY THIS SHAPE. Foot contact properties are created with the prim and cloned
# per environment, so they cannot vary between envs inside one process the way
# the per-env belt draw can. The sweep therefore loops the CONTACT arms in
# separate Isaac launches (identical seeds, identical plant recipe), and inside
# each launch varies the BELT arm per env.
#
# CONTACT ARMS (the doc's axis 2 + 3): rigid baseline, three softening levels at
# a fixed damping ratio, and restitution as an independent bounce axis with the
# contact left rigid, so "bouncy" and "slippery/soft" cannot be confused.
#
# BELT ARMS (per launch). k is joint-referred N m/rad (compare against
# servo_kp_knee=30, so 0.1x / 0.27x / 1x / 5x the position loop); d is given as
# a damping ratio of the belt's OWN mode, so zeta=1 is the transmission's own
# critical damping and the damping axis means the same thing at every k:
#   rigid       k=0     — the baseline; must be in every launch
#   3:1         0.1x servo,  zeta 1
#   8:0.7z      0.27x servo, zeta 0.7  (underdamped: the one that rings)
#   8:1         0.27x servo, zeta 1
#   8:2z        0.27x servo, zeta 2    (overdamped)
#   30:1        1.0x servo,  zeta 1
#   150:1       5x servo,    zeta 1
# A belt at k = servo_kp is the interesting boundary: the transmission is as
# stiff as the position loop, so the servo's own target error and the belt
# deflection are the same size.
#
# REFERENCE. The hardware opening (09-11 analysis) is: pitch-rate divergence
# after ~0.2-0.3 s, min pitch rate ~-340 deg/s (S4a trim 0) and -443 (trim
# +0.089) vs ~-175 in the twin, roll small. R1r hardware ~-148 deg/s.
#
# Usage:  powershell -File run_s4_elasticity_sweep_GPU2.ps1                  # full sweep
#         powershell -File run_s4_elasticity_sweep_GPU2.ps1 -Smoke            # ~3 min pipeline check
#         powershell -File run_s4_elasticity_sweep_GPU2.ps1 -ContactOnly 3    # one contact arm
#         powershell -File run_s4_elasticity_sweep_GPU2.ps1 -Belt "0:0,8:1"   # custom belt cells
param(
  [string]$Name = '',
  [int]$Seeds = 8,
  [double]$EpisodeS = 2.0,
  [double]$Trim = 0.0,
  [string]$Belt = '0:0,3:1z,8:0.7z,8:1z,8:2z,30:1z,150:1z',
  # [string[]], NOT [string]: a [string] type constraint re-casts every later
  # assignment back to a scalar, so `+=` STRING-CONCATENATES and the two
  # policies arrive welded into one argument ('...model_3150.ptS4a=...').
  # Confirmed 2026-09-11 by probe: with [string] the array reports
  # count=1 / type=String; untyped or [string[]] it reports count=2.
  [string[]]$Policies = @(),
  [int[]]$ContactOnly = @(),
  [switch]$DryRun,
  [switch]$Smoke
)
$ErrorActionPreference = 'Stop'

$envRoot = '<CONDA_ENV>'
$env:CUDA_VISIBLE_DEVICES = 'GPU-<uuid>'   # RTX 4060 Ti (GPU2)
$env:CARB_APP_PATH = '<ISAACLAB>\_isaac_sim\kit'
$env:EXP_PATH      = '<ISAACLAB>\_isaac_sim\apps'
$env:ISAAC_PATH    = '<ISAACLAB>\_isaac_sim'
$env:PYTHONPATH    = ';<ISAACLAB>\_isaac_sim\site'
$env:RESOURCE_NAME = 'IsaacSim'
$env:CONDA_PREFIX  = $envRoot
$env:PATH = "$envRoot;$envRoot\Library\mingw-w64\bin;$envRoot\Library\usr\bin;$envRoot\Library\bin;$envRoot\Scripts;$envRoot\bin;" + $env:PATH
$env:PYTHONUNBUFFERED = '1'

$RL = '<REPO>\RL'
if ($Name -eq '') { $Name = (Get-Date -Format 'yyyyMMdd_HHmm') + $(if ($Smoke) { '_smoke' } else { '' }) }
$outDir = Join-Path $RL "logs\s4_elasticity\$Name"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
Set-Location $RL

# The two policies the analysis needs side by side: the deployed stand-in
# (must stay recoverable) and the stepping policy under test.
#
# Each entry is appended on its own line, NOT written as
# @('a' + $x, 'b' + $y). PowerShell evaluates that array literal as a single
# concatenation expression, so it produces ONE element with the two policies
# joined by a space — which is the 2026-09-11 failure ("checkpoint not found:
# <R1r path> S4a=<path>") and passed the real path to a nonexistent file.
if ($Policies.Count -eq 0) {
  $Policies = @()
  $Policies += 'R1r=' + "$RL\logs\rsl_rl\p9_replant\2026-08-27_10-30-37_R1_resume3110\model_3150.pt"
  $Policies += 'S4a=' + "$RL\logs\rsl_rl\p9_replant\2026-09-09_22-13-40_S4a_measDR\model_6520.pt"
} elseif ($Policies.Count -eq 1 -and $Policies[0] -like '*;*') {
  # A caller passing one ';'-joined string still works.
  $Policies = @($Policies[0] -split ';' | Where-Object { $_ -ne '' })
}
if ($Policies.Count -eq 0) { throw "no policies resolved" }
# Guard against a silent weld: every entry must carry a name and a real file.
foreach ($p in $Policies) {
  if ($p -notmatch '^[^=]+=.+') { throw "malformed policy entry '$p' (expected NAME=PATH)" }
  $pPath = $p.Substring($p.IndexOf('=') + 1)
  if (-not (Test-Path -LiteralPath $pPath)) { throw "checkpoint missing for '$($p.Split('=')[0])': $pPath" }
}

# S4 measured-plant recipe (same block as run_s4_sysid_GPU2.ps1), minus the
# knobs this sweep sets itself. belt_series_enable=true is REQUIRED for any
# nonzero belt cell: it seeds the actuator buffers before the articulation is
# built (env code raises if a cell asks for stiffness without it).
$PLANT = @(
  'env.rew_scale_gait_ref=3.0',
  'env.sat_term_threshold=2.0', 'env.sat_term_tau_s=0.5',
  'env.swing_clearance_zero_m=0.078',
  'env.servo_kp_hip_pitch=20.0', 'env.servo_kd_hip_pitch=1.4',
  'env.servo_kp_knee=30.0', 'env.servo_kd_knee=1.5',
  'env.hip_pitch_armature=0.0385',
  'env.exec_noise_target_bias_max=0.03', 'env.exec_noise_target_std=0.015',
  'env.dr_action_delay_min=1', 'env.dr_action_delay_max=2',
  'env.dr_imu_delay_min=1', 'env.dr_imu_delay_max=1',
  'env.dr_joint_delay_min=1', 'env.dr_joint_delay_max=1',
  'env.dr_stiffness_mult_min=0.90', 'env.dr_stiffness_mult_max=1.10',
  'env.dr_armature_mult_min=0.80', 'env.dr_armature_mult_max=1.20',
  'env.dr_effort_mult_min=0.9',
  'env.dr_com_offset_max=0.01',
  'env.dr_com_shift_x_max=0.0', 'env.dr_com_shift_y_max=0.01', 'env.dr_com_shift_z_max=0.0',
  'env.dr_proj_grav_bias_max=0.13',
  'env.curriculum_enable=false', 'env.kneel_start_probability=0.0',
  'env.pickup_enable=false',
  'env.standup_ramp_mode=path',
  'env.belt_series_enable=true',
  'env.belt_series_rotor_inertia=0.01936'
)

# Contact arms: stiffness N/m, damping N s/m, restitution.
# damping = 2*zeta*sqrt(k * m_eff) with zeta = 0.7 and m_eff = 1.0 kg — the
# mass that actually participates in the ball-foot contact during a single-
# support catch, NOT the 5.57 kg total (using the total over-damps by ~2.4x and
# would have conflated "soft" with "heavily damped").
# The foot_contact.cfg guard rejects c > 10x critical for the full robot mass,
# which is 1493 N s/m at k=1000 — these values sit well inside it.
$contactArms = @(
  @{ label = 'rigid';      s = 0.0;    d = 0.0;   r = 0.0 },
  @{ label = 'soft2e4';    s = 20000;  d = 198;   r = 0.0 },
  @{ label = 'soft5e3';    s = 5000;   d = 99;    r = 0.0 },
  @{ label = 'soft1e3';    s = 1000;   d = 44;    r = 0.0 },
  @{ label = 'soft2e2';    s = 200;    d = 20;    r = 0.0 },
  @{ label = 'rigid_rest003'; s = 0.0; d = 0.0;   r = 0.03 },
  @{ label = 'rigid_rest01';  s = 0.0; d = 0.0;   r = 0.1 },
  @{ label = 'rigid_rest02';  s = 0.0; d = 0.0;   r = 0.2 }
)

Write-Output "[run ] s4_elasticity name=$Name out=$outDir"
Write-Output "[run ] belt cells: $Belt"
Write-Output "[run ] policies ($($Policies.Count)): $($Policies -join '  |  ')"
Write-Output "[run ] Isaac Sim startup takes ~1-2 min per contact arm before output appears."

$n = 0
foreach ($arm in $contactArms) {
  $n++
  if ($ContactOnly.Count -gt 0 -and ($ContactOnly -notcontains $n)) { continue }
  Write-Output ""
  Write-Output "[arm ] $n/$($contactArms.Count) contact=$($arm.label) stiffness=$($arm.s) damping=$($arm.d) restitution=$($arm.r)"
  $armDir = Join-Path $outDir $arm.label
  New-Item -ItemType Directory -Force -Path $armDir | Out-Null

  $argList = @('-u', "$RL\scripts\rsl_rl\sweep_plant_elasticity.py",
    '--task', 'Template-standup_curriculum-v0', '--headless',
    '--out', $armDir,
    '--belt', $Belt,
    '--seeds', "$Seeds",
    # Explicit scene size: 2 policy blocks x (1 + belt cells) x seeds.  Left to
    # the task cfg this is 32768 (the training size), which builds a
    # training-scale stage for a fixed-policy sweep — observed 2026-09-11 as a
    # dry run that spent minutes composing 32768 envs.
    '--num-envs', "$(2 * ((1 + ($Belt -split ',').Count) * $Seeds))",
    '--episode-s', "$EpisodeS",
    '--trim', "$Trim",
    '--contact-stiffness', "$($arm.s)",
    '--contact-damping', "$($arm.d)",
    '--contact-restitution', "$($arm.r)")
  # One --policy flag per checkpoint, one array element each.  Array splatting
  # (@argList) passes each element as a single argument, so this is safe even
  # if a checkpoint path contains a space — unlike building one 'a,b'/'a;b'
  # string, which is how the first real run lost its second policy.
  foreach ($p in $Policies) { $argList += "--policy=$p" }
  if ($Smoke)  { $argList += '--smoke' }
  if ($DryRun) { $argList += '--dry-run' }
  $argList += $PLANT

  $ErrorActionPreference = 'Continue'   # Isaac writes to stderr; PS 5.1 would otherwise abort
  & "$envRoot\python.exe" @argList
  $code = $LASTEXITCODE
  $ErrorActionPreference = 'Stop'
  Write-Output "[arm ] contact=$($arm.label) exit=$code  $(Get-Date -Format HH:mm:ss)"
  if ($code -ne 0) {
    Write-Output "[arm ] FAILED - stopping the sweep so the failure is not buried under later arms."
    Write-Output "[arm ] (a rigid-arm spawn failure means BALL_FOOT_COLLISION_PATHS is wrong;"
    Write-Output "[arm ]  the error lists the collision prims that DO exist under the robot root.)"
    exit $code
  }
}
Write-Output ""
Write-Output "[done] s4_elasticity name=$Name  $(Get-Date -Format HH:mm:ss)"
Write-Output "[done] per-arm report: $outDir\<arm>\report.txt ; traces: <arm>\rollout.npz ; per-trial: <arm>\cells.csv"
