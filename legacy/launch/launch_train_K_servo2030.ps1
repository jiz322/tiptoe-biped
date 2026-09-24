# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# P10 arm K on GPU 1: FROM-SCRATCH, full P8 recipe, correct swing zero,
# designed servo gains — kp/kd from the MEASURED plant, coupling stays ON.
# Rationale: P7's armature correction silently halved the servo envelope at
# fixed kp 8/kd 0.5 — knee went 4.4 Hz ζ0.87 (believed) → 2.3 Hz ζ0.45
# (real), hip_pitch 3.1 Hz ζ0.61 → 2.0 Hz ζ0.40. Every recipe that ever
# converged did so on the FAST believed plant; hardware always was the slow
# one. K restores the converged-envelope bandwidth on the honest inertia:
#   hip_pitch J≈50.3e-3: kp 20, kd 1.4 → 3.2 Hz, ζ0.70
#   knee      J≈38.5e-3: kp 30, kd 1.5 → 4.4 Hz, ζ0.70
#   hip_roll unchanged 6/0.3 (roll J tether-confounded; falls pitch-initiate)
# Action scales unchanged (position envelope + action_rate weights fixed);
# Kp·scale becomes 15.0 / 30.75 Nm — i.e. the effort_limit convention against
# the CORRECT P7 limits (17 / 23.2-30). ±1 knee action can now transiently
# saturate: watch diagnostics/sat_term_frac early (tripwire ~1e-2 @ iter 50).
# Deploy mapping if K flies: motor-side knee kp/kd = joint × (22/30)² =
# 16.1 / 0.81; hip_pitch direct 20 / 1.4 — update REAL_KP contract then.
# Single diff vs Z. NOTE: overrides are CLI-only — a resume must re-pass all nine.
$ErrorActionPreference = 'Stop'

$envRoot = '<CONDA_ENV>'
$env:CUDA_VISIBLE_DEVICES = '1'
$env:CARB_APP_PATH = '<ISAACLAB>\_isaac_sim\kit'
$env:EXP_PATH      = '<ISAACLAB>\_isaac_sim\apps'
$env:ISAAC_PATH    = '<ISAACLAB>\_isaac_sim'
$env:PYTHONPATH    = ';<ISAACLAB>\_isaac_sim\site'
$env:RESOURCE_NAME = 'IsaacSim'
$env:CONDA_PREFIX  = $envRoot
$env:PATH = "$envRoot;$envRoot\Library\mingw-w64\bin;$envRoot\Library\usr\bin;$envRoot\Library\bin;$envRoot\Scripts;$envRoot\bin;" + $env:PATH

$argList = @(
  '-u', 'scripts/rsl_rl/train.py',
  '--task', 'Template-standup_curriculum-v0',
  '--headless',
  '--run_name', 'K_servo2030_swing052',
  'agent.algorithm.entropy_coef=0.005',
  'env.rew_scale_gait_ref=3.0',
  'env.sat_term_threshold=0.5',
  'env.sat_term_tau_s=0.5',
  'env.swing_clearance_zero_m=0.052',
  'env.servo_kp_hip_pitch=20.0',
  'env.servo_kd_hip_pitch=1.4',
  'env.servo_kp_knee=30.0',
  'env.servo_kd_knee=1.5'
)
$p = Start-Process -FilePath "$envRoot\python.exe" `
  -WorkingDirectory '<REPO>\RL' `
  -ArgumentList $argList `
  -RedirectStandardOutput '<REPO>\RL\logs\train_K_servo2030.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_K_servo2030.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
