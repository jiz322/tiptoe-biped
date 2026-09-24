# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# S6 foot-restitution ABSORPTION A/B sweep — eval only, NO training.
# GPU1 (RTX 4090), copied from run_s4_elasticity_sweep_GPU2.ps1 and edited for
# one question: does the S6_restDR policy (foot-ball restitution trained as a
# per-env DR band over 0.4-0.8) stay flat across e, where S4a degrades?
#
# CELLS. Foot contact properties are created with the prim and cloned per
# environment, so they cannot vary between envs inside one process.  The four
# restitution cells are therefore four Isaac launches (identical seeds,
# identical plant recipe), each holding a RIGID foot (stiffness 0, damping 0)
# with a pinned restitution e in {0.0, 0.3, 0.63, 0.8} and combine mode "max".
# No belt sweep, no pad: the only thing that differs across cells is e.
#
# POLICIES. S4a (baseline, trained with e=0 fixed) and S6 (candidate).  Both
# run side by side in every launch on the same seeds.  NOTE model_7900 is only
# +120 iters into S6 (resume was at 7780) — an early checkpoint, flagged in the
# report if the absorption bar is missed.
#
# VERIFICATION. sweep_rest_absorption.py reads back, once per cell, that the
# ball rows hold the cell's restitution (combine mode "max" on the material)
# and that physxScene:bounceThreshold is 0.05 (the 0.5 default would flatten
# the curve and fake an absorption).
#
# Usage:  powershell -File run_s6_eval_sweep_GPU1.ps1            # full sweep
#         powershell -File run_s6_eval_sweep_GPU1.ps1 -Smoke     # ~3 min check
#         powershell -File run_s6_eval_sweep_GPU1.ps1 -DryRun    # verify only
param(
  [string]$Name = '',
  [int]$Seeds = 8,
  [double]$EpisodeS = 2.0,
  [double]$Trim = 0.0,
  [string[]]$Policies = @(),
  [int[]]$ContactOnly = @(),
  [switch]$DryRun,
  [switch]$Smoke
)
$ErrorActionPreference = 'Stop'

$envRoot = '<CONDA_ENV>'
$env:CUDA_VISIBLE_DEVICES = 'GPU-<uuid>'   # RTX 4090 (GPU1)
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
$outDir = Join-Path $RL "logs\s6_absorption\$Name"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
Set-Location $RL

# The two policies side by side.  Each entry appended on its own line, NOT
# written as @('a' + $x, 'b' + $y): PowerShell evaluates that literal as one
# concatenation expression and welds the two policies into one argument (the
# 2026-09-11 failure).
if ($Policies.Count -eq 0) {
  $Policies = @()
  $Policies += 'S4a=' + "$RL\logs\rsl_rl\p9_replant\2026-09-09_22-13-40_S4a_measDR\model_7780.pt"
  $Policies += 'S6='  + "$RL\logs\rsl_rl\p9_replant\2026-09-15_10-57-26_S6_restDR\model_7900.pt"
} elseif ($Policies.Count -eq 1 -and $Policies[0] -like '*;*') {
  $Policies = @($Policies[0] -split ';' | Where-Object { $_ -ne '' })
}
if ($Policies.Count -eq 0) { throw "no policies resolved" }
foreach ($p in $Policies) {
  if ($p -notmatch '^[^=]+=.+') { throw "malformed policy entry '$p' (expected NAME=PATH)" }
  $pPath = $p.Substring($p.IndexOf('=') + 1)
  if (-not (Test-Path -LiteralPath $pPath)) { throw "checkpoint missing for '$($p.Split('=')[0])': $pPath" }
}

# S6 plant recipe for a fixed-policy standing-start rollout: belt OFF, pad OFF,
# curriculum OFF.  DR channels (stiffness/armature/effort/com/ang-vel/gravity
# tilt/delay) stay at the S6-trained values so each cell is judged inside the
# trained envelope; only foot restitution is pinned per cell.  sat_term at the
# S6-trained 0.5 (not the sweep's 2.0) so a fall is a fall.
$PLANT = @(
  'env.rew_scale_gait_ref=3.0',
  'env.sat_term_threshold=0.5', 'env.sat_term_tau_s=0.5',
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
  'env.dr_ang_vel_bias_max=0.05',
  'env.dr_gravity_tilt_max_deg=2.0',
  'env.dr_floor_friction_min=0.6', 'env.dr_floor_friction_max=1.1',
  'env.curriculum_enable=false', 'env.kneel_start_probability=0.0',
  'env.pickup_enable=false',
  'env.standup_ramp_mode=path',
  # belt OFF: pinned rigid (the sweep passes belt cells as "0:0", which needs
  # no series path; belt_series_enable stays false so the rigid legacy plant
  # is bit-exact).
  'env.belt_series_enable=false',
  # pad OFF (the S5 pad arm was killed 09-14; do not re-enter it here).
  'env.pad_contact_enable=false'
)

# The four restitution cells, all on a RIGID foot.
$contactArms = @(
  @{ label = 'e0';    s = 0.0; d = 0.0; r = 0.0 },
  @{ label = 'e03';   s = 0.0; d = 0.0; r = 0.3 },
  @{ label = 'e063';  s = 0.0; d = 0.0; r = 0.63 },
  @{ label = 'e08';   s = 0.0; d = 0.0; r = 0.8 }
)

Write-Output "[run ] s6_absorption name=$Name out=$outDir"
Write-Output "[run ] policies ($($Policies.Count)): $($Policies -join '  |  ')"
Write-Output "[run ] Isaac Sim startup takes ~1-2 min per contact arm before output appears."

$perCell = @()
$n = 0
foreach ($arm in $contactArms) {
  $n++
  if ($ContactOnly.Count -gt 0 -and ($ContactOnly -notcontains $n)) { continue }
  Write-Output ""
  Write-Output "[arm ] $n/$($contactArms.Count) label=$($arm.label) restitution=$($arm.r)"
  $armDir = Join-Path $outDir $arm.label
  New-Item -ItemType Directory -Force -Path $armDir | Out-Null

  $argList = @('-u', "$RL\scripts\rsl_rl\sweep_rest_absorption.py",
    '--task', 'Template-standup_curriculum-v0', '--headless',
    '--out', $armDir,
    '--belt', '0:0',
    '--seeds', "$Seeds",
    # Explicit scene size: 2 policy blocks x (1 belt cell) x seeds.  Left to the
    # task cfg this is 32768 (the training size), which builds a training-scale
    # stage for a fixed-policy sweep.
    '--num-envs', "$(2 * $Seeds)",
    '--episode-s', "$EpisodeS",
    '--trim', "$Trim",
    '--contact-stiffness', "$($arm.s)",
    '--contact-damping', "$($arm.d)",
    '--contact-restitution', "$($arm.r)")
  foreach ($p in $Policies) { $argList += "--policy=$p" }
  if ($Smoke)  { $argList += '--smoke' }
  if ($DryRun) { $argList += '--dry-run' }
  $argList += $PLANT

  # The sweep writes summary.json as its LAST action (after report.txt, cells.csv
  # and rollout.npz), so its appearance means the arm's work is done.
  $sumPath = Join-Path $armDir 'summary.json'
  if (Test-Path -LiteralPath $sumPath) { Remove-Item -LiteralPath $sumPath -Force }

  # Isaac's SimulationApp.close() HANGS on teardown on this machine (measured
  # 09-15: the kit log stops at "Using CUDA device ordinal 0" and the process
  # never exits, so `& python @args` blocks forever and the launcher never
  # reaches the next cell).  So the arm is launched detached and the launcher
  # waits for the OUTPUT rather than for the exit code, then reaps the process
  # tree.  Exit code is therefore NOT the success signal here: summary.json is.
  $outFile = Join-Path $armDir 'stdout.log'
  $errFile = Join-Path $armDir 'stderr.log'
  $p = Start-Process -FilePath "$envRoot\python.exe" -ArgumentList $argList `
    -RedirectStandardOutput $outFile -RedirectStandardError $errFile `
    -WindowStyle Hidden -PassThru

  $armTimeoutS = [Math]::Max(600, [int](180 + 120 * $Seeds))
  $t0 = Get-Date
  while ($true) {
    if (Test-Path -LiteralPath $sumPath) { break }
    if ($p.HasExited) { break }
    if (((Get-Date) - $t0).TotalSeconds -gt $armTimeoutS) {
      Write-Output "[arm ] TIMEOUT after ${armTimeoutS}s — reaping the arm"
      break
    }
    Start-Sleep -Seconds 5
  }
  $got = Test-Path -LiteralPath $sumPath
  if (-not $p.HasExited) {
    # Kill the whole tree: python.exe spawns the kit children, and an orphaned
    # kit holding GPU1 is what silently starved the earlier runs.
    & taskkill.exe /PID $p.Id /T /F | Out-Null
    Start-Sleep -Seconds 3
  }
  Write-Output "[arm ] label=$($arm.label) done=$got exited=$($p.HasExited)  $(Get-Date -Format HH:mm:ss)"
  if (-not $got) {
    Write-Output "[arm ] FAILED - no summary.json for this cell; stopping the sweep so the"
    Write-Output "[arm ] failure is not buried under later arms.  stderr tail:"
    if (Test-Path -LiteralPath $errFile) { Get-Content -LiteralPath $errFile -Tail 20 | Write-Output }
    if (Test-Path -LiteralPath $outFile) { Get-Content -LiteralPath $outFile -Tail 30 | Write-Output }
    exit 1
  }
  if ($DryRun) { continue }
  $perCell += (Get-Content -LiteralPath $sumPath -Raw | ConvertFrom-Json)
}
Write-Output ""
Write-Output "[done] s6_absorption name=$Name  $(Get-Date -Format HH:mm:ss)"

# ── absorption table across the four cells ────────────────────────────────────
if (-not $DryRun -and $perCell.Count -gt 0) {
  $pols = @($Policies | ForEach-Object { $_.Split('=')[0] })
  $table = New-Object System.Collections.Generic.List[object]
  foreach ($c in ($perCell | Sort-Object { [double]$_.e })) {
    foreach ($pn in $pols) {
      $r = $c.policies.$pn
      $table.Add([pscustomobject]@{
        policy = $pn
        e      = [double]$c.e
        ep_len = [double]$r.mean_ep_len
        fall   = [double]$r.fall_rate
        vx     = [double]$r.mean_vx
      })
    }
  }

  function CellVal($pn, $e) {
    $hit = $table | Where-Object { $_.policy -eq $pn -and $_.e -eq $e }
    if ($hit) { return $hit[0].ep_len } else { return $null }
  }
  $s4a0   = CellVal 'S4a' 0.0
  $s4a08  = CellVal 'S4a' 0.8
  $s6_0   = CellVal 'S6'  0.0
  $s6_08  = CellVal 'S6'  0.8
  $s4afall = $null; $s6fall = $null; $ratio = $null
  if ($null -ne $s4a0 -and $null -ne $s4a08) { $s4afall = $s4a0 - $s4a08 }
  if ($null -ne $s6_0 -and $null -ne $s6_08) { $s6fall  = $s6_0  - $s6_08 }
  if ($null -ne $s4afall -and $s4afall -ne 0.0 -and $null -ne $s6fall) {
    $ratio = $s6fall / $s4afall
  }

  Write-Output ""
  Write-Output "=== ABSORPTION TABLE (policy x e) ==="
  Write-Output ("{0,-6} {1,4} {2,10} {3,10} {4,10}" -f 'policy','e','ep_len_s','fall_rate','mean_vx')
  foreach ($row in ($table | Sort-Object policy, e)) {
    Write-Output ("{0,-6} {1,4} {2,10:F2} {3,10:F2} {4,10:F3}" -f $row.policy, $row.e, $row.ep_len, $row.fall, $row.vx)
  }
  Write-Output ""
  Write-Output "S4a fall e=0 -> e=0.8 (ep_len) : $([math]::Round($s4afall, 3)) s"
  Write-Output "S6  fall e=0 -> e=0.8 (ep_len) : $([math]::Round($s6fall, 3)) s"
  Write-Output "absorption ratio (S6/S4a)     : $([math]::Round($ratio, 3))"

  if ($null -ne $ratio) {
    if ($ratio -lt 0.5) {
      $verdict = 'ABSORBED'
    } elseif ($ratio -le 0.9) {
      $verdict = 'PARTIAL'
    } else {
      $verdict = 'NOT YET'
    }
    Write-Output "verdict                        : $verdict"
    if ($verdict -eq 'NOT YET') {
      Write-Output "NOTE: model_7900 is only +120 iters into S6 (early). Re-run with the"
      Write-Output "      latest S6 checkpoint available at the time before concluding."
    }
  } else {
    Write-Output "verdict                        : UNCOMPUTABLE (missing a cell; see arm warnings above)"
  }
  $table | ConvertTo-Json | Out-File -FilePath (Join-Path $outDir 'absorption_table.json') -Encoding utf8
}
Write-Output "[done] per-cell: $outDir\<arm>\report.txt, summary.json ; cross-cell: absorption_table.json"
