# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# S4 hardware-twin rollouts (2026-09-10) — sim reference for the 09-10 balance
# trials (data/s4_flight_20260910). Deploy contract via play_sim2real.py
# (blind 'training' handover, deploy gains, cmd=0, 1 env, no pickup), with the
# obs pitch bias PINNED to what the hardware policy saw:
#     pin = -IMU_PG_OFFSET  (pg_logged = raw - offset  ->  bias -0.089 / 0.0)
# Balance branch only (curriculum off, kneel spawn 0) = the hardware test.
# sat_term disabled (hardware has no saturation termination).
#
# PRE-REGISTERED (user hypothesis 09-10, "0.0 looks like sim; 0.089 compensates
# a mass-model error for the shuffle gait but not the stepping gait"):
#   H1  S4a sim survives >=8/8 at BOTH pins (sim absorbs the bias) while hw at
#       pin 0 fell 3/3 by 0.8 s -> the hw fall is a plant gap, not the trim.
#   H2  S4a sim gait amplitude (knee band-RMS at gait clock) >> R1r sim
#       (stepping vs shuffle), matching the hw a_knee contrast.
#   H3  S4a sim settled raw pitch at pin -0.089 is ~5 deg forward of pin 0
#       (the policy holds perceived pitch) — hw showed the same shift only in
#       the first 0.2 s, then dove to -20..-30 deg regardless.
#
# Usage: powershell -File run_s4_hwtwin_GPU2.ps1 -Arm S4a_m089
param([string]$Arm = 'S4a_m089')
$ErrorActionPreference = 'Stop'

$envRoot = '<CONDA_ENV>'
$env:CUDA_VISIBLE_DEVICES = 'GPU-<uuid>'   # RTX 4060 Ti (idle)
$env:CARB_APP_PATH = '<ISAACLAB>\_isaac_sim\kit'
$env:EXP_PATH      = '<ISAACLAB>\_isaac_sim\apps'
$env:ISAAC_PATH    = '<ISAACLAB>\_isaac_sim'
$env:PYTHONPATH    = ';<ISAACLAB>\_isaac_sim\site'
$env:RESOURCE_NAME = 'IsaacSim'
$env:CONDA_PREFIX  = $envRoot
$env:PATH = "$envRoot;$envRoot\Library\mingw-w64\bin;$envRoot\Library\usr\bin;$envRoot\Library\bin;$envRoot\Scripts;$envRoot\bin;" + $env:PATH

$RL = '<REPO>\RL'
$outDir = Join-Path $RL "logs\s4_hwtwin\$Arm"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
Set-Location $outDir

$S4A = "$RL\logs\rsl_rl\p9_replant\2026-09-09_22-13-40_S4a_measDR\model_6520.pt"
$S4B = "$RL\logs\rsl_rl\p9_replant\2026-09-09_22-13-44_S4b_measDR_pg05\model_6520.pt"
$R1R = "$RL\logs\rsl_rl\p9_replant\2026-08-27_10-30-37_R1_resume3110\model_3150.pt"

# S4 recipe (launch_train_S4a_measDR.ps1) minus curriculum/pickup, plus eval pins
$S4_OVR = @(
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
  'env.dr_floor_friction_min=0.6', 'env.dr_floor_friction_max=1.1',
  'env.dr_stiffness_mult_min=0.90', 'env.dr_stiffness_mult_max=1.10',
  'env.dr_armature_mult_min=0.80', 'env.dr_armature_mult_max=1.20',
  'env.dr_effort_mult_min=0.9',
  'env.dr_com_offset_max=0.01',
  'env.dr_com_shift_x_max=0.02', 'env.dr_com_shift_y_max=0.01',
  'env.dr_mass_mult_min=1.00', 'env.dr_mass_mult_max=1.14',
  'env.curriculum_enable=false', 'env.kneel_start_probability=0.0',
  'env.pickup_enable=false',
  'env.standup_ramp_mode=path'
)
$R1_OVR = @(
  'env.swing_clearance_zero_m=0.069',
  'env.sat_term_threshold=2.0', 'env.sat_term_tau_s=0.5',
  'env.rew_scale_gait_ref=3.0',
  'env.pickup_enable=false'
)

switch ($Arm) {
  'S4a_m089' { $CKPT = $S4A; $OVR = $S4_OVR + @('env.dr_proj_grav_bias_max=0.13', 'env.dr_proj_grav_bias_pin_x=-0.089') }
  'S4a_0'    { $CKPT = $S4A; $OVR = $S4_OVR + @('env.dr_proj_grav_bias_max=0.13', 'env.dr_proj_grav_bias_pin_x=0.0') }
  'S4b_m089' { $CKPT = $S4B; $OVR = $S4_OVR + @('env.dr_proj_grav_bias_max=0.05', 'env.dr_proj_grav_bias_pin_x=-0.089') }
  'S4b_0'    { $CKPT = $S4B; $OVR = $S4_OVR + @('env.dr_proj_grav_bias_max=0.05', 'env.dr_proj_grav_bias_pin_x=0.0') }
  'R1r_m089' { $CKPT = $R1R; $OVR = $R1_OVR + @('env.dr_proj_grav_bias_pin_x=-0.089') }
  'R1r_0'    { $CKPT = $R1R; $OVR = $R1_OVR + @('env.dr_proj_grav_bias_pin_x=0.0') }
  # ── TOP-MASS arms (user hypothesis 09-10: the weigh-in surplus sits on the upper
  #    body; uniform mass_mult cannot express it). env.dr_top_mass_add_kg adds mass
  #    to torso2hip + both hip-roll chassis blocks only (mass, not CoM/inertia
  #    tensors); mass_mult pinned 1.0 so total = URDF + added. Trim 0 (pin 0), the
  #    contract of the "looks like sim" hardware runs.
  #    PRE-REGISTERED: user hypothesis TRUE -> +0.3 kg reproduces hardware:
  #      single-support bursts median <= -180 deg/s with a tail beyond -400, and
  #      >= 4/8 runs fall within 3 s.  My expectation: bursts get slightly SMALLER
  #      (heavier top rotates slower for the same torque) and survival stays 8/8 at
  #      +0.3; +1.0 kg is the exaggerated control.
  'S4a_0_top03' { $CKPT = $S4A; $OVR = $S4_OVR + @('env.dr_proj_grav_bias_max=0.13', 'env.dr_proj_grav_bias_pin_x=0.0',
                  'env.dr_mass_mult_min=1.0', 'env.dr_mass_mult_max=1.0', 'env.dr_top_mass_add_kg=0.3') }
  'S4a_0_top06' { $CKPT = $S4A; $OVR = $S4_OVR + @('env.dr_proj_grav_bias_max=0.13', 'env.dr_proj_grav_bias_pin_x=0.0',
                  'env.dr_mass_mult_min=1.0', 'env.dr_mass_mult_max=1.0', 'env.dr_top_mass_add_kg=0.6') }
  'S4a_0_top10' { $CKPT = $S4A; $OVR = $S4_OVR + @('env.dr_proj_grav_bias_max=0.13', 'env.dr_proj_grav_bias_pin_x=0.0',
                  'env.dr_mass_mult_min=1.0', 'env.dr_mass_mult_max=1.0', 'env.dr_top_mass_add_kg=1.0') }
  default { throw "unknown arm '$Arm'" }
}

$log = Join-Path $outDir 'run.out.log'
Write-Output "[run ] s4_hwtwin arm=$Arm  $(Get-Date -Format HH:mm:ss)  log=$log"
$argList = @('-u', "$RL\scripts\rsl_rl\play_sim2real.py",
  '--task', 'Template-standup_curriculum-v0', '--headless', '--no_obs_summary',
  '--checkpoint', $CKPT, '--handover', 'training',
  '--autolog_steps', '500', '--autolog_runs', '8', '--seed', '0') + $OVR
$p = Start-Process -FilePath "$envRoot\python.exe" -WorkingDirectory $outDir `
  -ArgumentList $argList -RedirectStandardOutput $log `
  -RedirectStandardError (Join-Path $outDir 'run.err.log') -WindowStyle Hidden -PassThru -Wait
Write-Output "[done] arm=$Arm exit=$($p.ExitCode)  $(Get-Date -Format HH:mm:ss)"
