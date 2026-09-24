# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# Play the S6_restDR checkpoint on ITS OWN plant, with an optional stand-up
# view. Twin of run_play_S8_vendorJ.ps1; run_foot_demo.ps1 is left untouched.
#
#   powershell -File run_play_S6_restDR.ps1                     # stand start, S6 DR draws
#   powershell -File run_play_S6_restDR.ps1 -Plant pin          # restitution pinned at 0.63
#   powershell -File run_play_S6_restDR.ps1 -Alpha 1.0          # stand-up from the seiza rest pose
#   powershell -File run_play_S6_restDR.ps1 -Alpha 0.6 -NumEnvs 4
#   powershell -File run_play_S6_restDR.ps1 -Alpha 0.8 -AlphaRandom    # each env d ~ U[0, 0.8]
#   powershell -File run_play_S6_restDR.ps1 -Checkpoint 7890    # pin a checkpoint number
#   powershell -File run_play_S6_restDR.ps1 -AutologSteps 500   # headless simlog npz (stand mode only)
#   powershell -File run_play_S6_restDR.ps1 -VideoSteps 600     # record an mp4, then exit
#   powershell -File run_play_S6_restDR.ps1 -Alpha 1.0 -VideoSteps 600
#
# ── Two play modes ──────────────────────────────────────────────────────────
#   -Alpha < 0 (default)  play_sim2real.py — the deploy-contract viewer: stand
#                         start at difficulty 0, blind handover, gait phase 0,
#                         curriculum forced OFF by the script itself. Same
#                         script and flags as run_foot_demo.ps1.
#   -Alpha 0..1           play_alpha.py — every env spawns at exactly this
#                         curriculum difficulty on the stand-up path (0 = stand,
#                         1 = seiza rest pose). Keys: `,` / `.` step alpha by
#                         0.05 and reset, R reset, P pause. -AlphaRandom samples
#                         d ~ U[0, alpha] per env with the training kneel /
#                         frontier split instead of pinning every env.
#                         play_alpha.py DISABLES pushes on its own and ignores
#                         -AutologSteps (that logger lives in play_sim2real).
#                         It also re-exports policy.pt/.onnx into the run's
#                         `exported/` folder every launch — harmless, but that
#                         folder is not a deploy artifact.
#
# ── -VideoSteps N (both modes) ──────────────────────────────────────────────
#   Passes `--video --video_length N` to the play script: cameras are enabled,
#   ONE clip is recorded from step 0 for N control steps (50 Hz -> N/50 s) and
#   the script exits when the clip is done. Output mp4 lands inside the RUN
#   folder: `videos\sim2real\` in stand mode, `videos\play\` in -Alpha mode
#   (gym RecordVideo naming, rl-video-step-0.mp4). Works with or without
#   -AutologSteps; the GUI window still opens unless -AutologSteps made the
#   run headless. Recording is slower than real time — expect ~2-4x.
#
# ── What is in $RECIPE and what is not ──────────────────────────────────────
# Copied from launch_train_S6_restDR.ps1 (2026-09-15), keeping ONLY the
# overrides a play rollout can feel: plant, obs/action contract (delay buffers,
# obs bias bands, exec noise), termination fence, spawn contract. Dropped on
# purpose, because they touch nothing but the reward or the trainer:
#   rew_scale_gait_ref, rew_scale_no_flight, rew_scale_vz_penalty,
#   rew_scale_action_smoothness, rew_scale_action_snap,
#   swing_clearance_zero_m, swing_clearance_target_m,
#   rew_scale_standup_track, rew_scale_standup_contact,
#   curriculum_enable/initial_alpha/max_alpha/alpha_step/advance_* (play_alpha
#   pins alpha and zeroes the step; play_sim2real forces curriculum off).
# If launch_train_S6_restDR.ps1 is ever edited, re-diff this list against it.
#
# ══ PLANT PINS — REQUIRED, S6 predates the 09-18 vendor-inertia change ══════
# S6 trained on the P7-era plant: hip_pitch armature pinned 0.0385 by its
# launch, knee 0.036 and belt rotor 0.01936 as the then-defaults, and one
# armature draw for both legs. The code now derives hip/knee from the vendor
# rotor inertia (4.2e-3) and draws armature per leg, so S6 on today's defaults
# would run a hip ~4.6x lighter than it trained on. The four pins below restore
# its plant bit-exactly (memory: vendor-rotor-inertia-20260918; the same block
# is quoted in launch_train_S8_vendorJ.ps1's header). Do not drop them, and do
# not copy them into the S8 script — S8 trained unpinned.
#
# -Plant arms (S6's one live channel is foot restitution; NO belt series):
#   train   restitution ~ U[0.4,0.8] per env per episode on the 0.63 carrier —
#           the S6 training band (narrower than S7/S8's 0.2-0.8). With
#           -NumEnvs > 1 the envs differ; read the draw off the [PLANT] line.
#   pin     restitution pinned at the 0.63 carrier (band ends 0 -> per-env
#           write skipped). N envs run one identical plant; the A/B arm.
#   rigid   restitution 0 — the legacy pre-S5 plant.
# Foot MATERIAL stays rigid in all three; layer -FootStiffness / -FootZeta /
# -FootRest on top exactly as in run_foot_demo.ps1.
param(
  [double]$Alpha = -1.0,
  [switch]$AlphaRandom,
  [ValidateSet('train', 'pin', 'rigid')]
  [string]$Plant = 'train',
  [int]$NumEnvs = 1,
  [int]$AutologSteps = 0,
  # > 0 = record one mp4 of this many control steps, then exit (see header).
  [int]$VideoSteps = 0,
  # '' = newest model_*.pt in $RUN; a bare number = model_<n>.pt in $RUN; a
  # path = that file.
  [string]$Checkpoint = '',
  [string]$Gpu = '2',
  [double]$FootStiffness = -1.0,
  [double]$FootZeta = -1.0,
  [double]$FootRest = -1.0
)
$ErrorActionPreference = 'Stop'

$envRoot = '<CONDA_ENV>'
$env:CUDA_VISIBLE_DEVICES = $Gpu
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

$RUN = "$RL\logs\rsl_rl\p9_replant\2026-09-15_10-57-26_S6_restDR"
if ($Checkpoint -eq '') {
  $ck = Get-ChildItem -LiteralPath $RUN -Filter 'model_*.pt' |
        Sort-Object { [int]($_.BaseName -replace '\D', '') } | Select-Object -Last 1
  if (-not $ck) { throw "no model_*.pt in $RUN" }
  $CKPT = $ck.FullName
} elseif ($Checkpoint -match '^\d+$') {
  $CKPT = "$RUN\model_$Checkpoint.pt"
} else {
  $CKPT = $Checkpoint
}
if (-not (Test-Path -LiteralPath $CKPT)) { throw "checkpoint missing: $CKPT" }

# launch_train_S6_restDR.ps1 minus the reward/trainer-only lines (see header),
# plus the four plant pins the 09-18 code change made necessary.
$RECIPE = @(
  # termination fence
  'env.sat_term_threshold=0.5', 'env.sat_term_tau_s=0.5',
  # deploy servo gains
  'env.servo_kp_hip_pitch=20.0', 'env.servo_kd_hip_pitch=1.4',
  'env.servo_kp_knee=30.0', 'env.servo_kd_knee=1.5',
  # PLANT PINS (pre-vendor-inertia plant, see header)
  'env.hip_pitch_armature=0.0385',
  'env.knee_armature=0.036',
  'env.belt_series_rotor_inertia=0.019360',
  'env.dr_armature_per_leg=false',
  # exec-noise injection
  'env.exec_noise_target_bias_max=0.03', 'env.exec_noise_target_std=0.015',
  # spawn-posture DR (play_sim2real scales these by --init_noise_scale, default 0)
  'env.noise_joint_pos=0.18', 'env.init_base_orient_noise_rad=0.12',
  # pushes (play_alpha turns these off itself)
  'env.push_enable=true', 'env.push_probability=0.001',
  'env.push_force_min=10.0', 'env.push_force_max=40.0', 'env.push_duration_steps=10',
  # obs/action delay buffers
  'env.dr_action_delay_min=1', 'env.dr_action_delay_max=2',
  'env.dr_imu_delay_min=1', 'env.dr_imu_delay_max=1',
  'env.dr_joint_delay_min=1', 'env.dr_joint_delay_max=1',
  # plant DR
  'env.dr_floor_friction_min=0.6', 'env.dr_floor_friction_max=1.1',
  'env.dr_stiffness_mult_min=0.90', 'env.dr_stiffness_mult_max=1.10',
  'env.dr_armature_mult_min=0.80', 'env.dr_armature_mult_max=1.20',
  'env.dr_effort_mult_min=0.9',
  'env.dr_com_offset_max=0.01',
  'env.dr_com_shift_x_max=0.02', 'env.dr_com_shift_y_max=0.01',
  'env.dr_mass_mult_min=1.00', 'env.dr_mass_mult_max=1.14',
  'env.dr_gravity_tilt_max_deg=2.0',
  # obs bias bands (S6 kept the 0.13 pg band; S8 zeroed it)
  'env.dr_proj_grav_bias_max=0.13',
  'env.dr_ang_vel_bias_max=0.05',
  # spawn / stand-up contract
  'env.pickup_enable=false',
  'env.standup_ramp_mode=path',
  'env.standup_grace_seconds=5.0',
  'env.standup_path_spawn_clearance_m=0.006',
  # reset sampling under -AlphaRandom (inert when alpha is pinned)
  'env.kneel_start_probability=0.6',
  'env.curriculum_frontier_frac=0.5', 'env.curriculum_frontier_width=0.3'
)

switch ($Plant) {
  'train' {
    $FOOT = @('env.foot_restitution=0.63',
              'env.dr_foot_restitution_min=0.4',
              'env.dr_foot_restitution_max=0.8',
              'env.dr_foot_restitution_buckets=64')
  }
  'pin' {
    $FOOT = @('env.foot_restitution=0.63')
  }
  'rigid' {
    $FOOT = @('env.foot_contact_stiffness=0.0',
              'env.foot_restitution=0.0')
  }
}

# Live foot-material overrides on top of the arm (last occurrence wins in Hydra).
$KEEP = -1.0
function Get-FootArg([string]$key) {
  $hit = $FOOT | Where-Object { $_ -like "$key=*" } | Select-Object -Last 1
  if ($hit) { return [double]($hit -replace '^[^=]*=', '') }
  return $KEEP
}
if ($FootStiffness -ne $KEEP) { $FOOT = @($FOOT | Where-Object { $_ -notlike 'env.foot_contact_stiffness=*' })
                                $FOOT += "env.foot_contact_stiffness=$FootStiffness" }
if ($FootRest -ne $KEEP)      { $FOOT = @($FOOT | Where-Object { $_ -notlike 'env.foot_restitution=*' })
                                $FOOT += "env.foot_restitution=$FootRest" }
if ($FootZeta -ne $KEEP) {
  $effK = Get-FootArg 'env.foot_contact_stiffness'
  if ($FootStiffness -ne $KEEP) { $effK = $FootStiffness }
  if ($effK -le 0.0) {
    throw "-FootZeta $FootZeta needs a foot stiffness to damp. Pass -FootStiffness too (e.g. 5000)."
  }
  $FOOT = @($FOOT | Where-Object { $_ -notlike 'env.foot_contact_damping*' })
  $FOOT += "env.foot_contact_damping_ratio=$FootZeta"
}

if ($Alpha -ge 0.0) {
  if ($AutologSteps -gt 0) { throw "-AutologSteps is a play_sim2real logger; not available with -Alpha." }
  $argList = @('-u', "$RL\scripts\rsl_rl\play_alpha.py",
    '--task=Template-standup_curriculum-v0',
    "--checkpoint=$CKPT",
    '--num_envs', "$NumEnvs",
    '--alpha', "$Alpha",
    '--seed=0')
  if ($AlphaRandom) { $argList += '--alpha-random' }
  $mode = "standup alpha=$Alpha random=$($AlphaRandom.IsPresent)"
} else {
  $argList = @('-u', "$RL\scripts\rsl_rl\play_sim2real.py",
    '--task=Template-standup_curriculum-v0',
    "--checkpoint=$CKPT",
    '--num_envs', "$NumEnvs",
    '--gait_phase_init', '0',
    '--seed=0')
  if ($AutologSteps -gt 0) {
    $argList += @('--headless', '--autolog_steps', "$AutologSteps", '--autolog_runs', '1', '--no_obs_summary')
  }
  $mode = 'stand (deploy handover)'
}
if ($VideoSteps -gt 0) {
  $argList += @('--video', '--video_length', "$VideoSteps")
  $vdir = if ($Alpha -ge 0.0) { 'play' } else { 'sim2real' }
  Write-Output "[S6] video: $VideoSteps steps -> $(Split-Path -Parent $CKPT)\videos\$vdir\"
}
$argList += $RECIPE
$argList += $FOOT

Write-Output "[S6] ckpt=$CKPT"
Write-Output "[S6] mode=$mode  plant=$Plant  foot: $($FOOT -join '  ')"
$ErrorActionPreference = 'Continue'   # Isaac writes to stderr
& "$envRoot\python.exe" @argList
$code = $LASTEXITCODE
Write-Output "[S6] exit=$code"
exit $code
