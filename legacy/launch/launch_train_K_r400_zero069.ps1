# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# P10 K-line continuation on GPU 1: resume K model_400 with the swing zero
# recalibrated for the stiff-servo stance.
# The 08-19 K run (kp 20/30, kd 1.4/1.5) escaped the 0.71 shelf and reached
# active_frac 0.545 by iter 384 — faster than any P8 reference. Its stance
# sits higher than the Z-lineage (less gravity droop at kp 20/30):
# idle_toe_z_p10 settled at 0.066, so zero=0.052 is 17 mm BELOW resting
# toes — 99% of idle swing samples collect the clearance ramp with zero
# lift (the documented 08-10 bug; that recalibration moved 0.055→0.075 and
# the lineage then converged). Rule: zero = p10 + 0.003 = 0.069.
# model_400 = post-crash-recovery best (reward 1.11e3, active_frac 0.546);
# the iter-384 crash is in-family LR turbulence (A_baseline logged 19 such
# >30% reward drops while converging) — schedule deliberately untouched.
# NOTE: resume re-reads cfg defaults — ALL overrides re-passed here.
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
  '--run_name', 'K_r400_zero069',
  '--resume',
  '--load_run', '2026-08-19_20-43-14_K_servo2030_swing052',
  '--checkpoint', 'model_400.pt',
  'agent.algorithm.entropy_coef=0.005',
  'env.rew_scale_gait_ref=3.0',
  'env.sat_term_threshold=0.5',
  'env.sat_term_tau_s=0.5',
  'env.swing_clearance_zero_m=0.069',
  'env.servo_kp_hip_pitch=20.0',
  'env.servo_kd_hip_pitch=1.4',
  'env.servo_kp_knee=30.0',
  'env.servo_kd_knee=1.5'
)
$p = Start-Process -FilePath "$envRoot\python.exe" `
  -WorkingDirectory '<REPO>\RL' `
  -ArgumentList $argList `
  -RedirectStandardOutput '<REPO>\RL\logs\train_K_r400_zero069.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_K_r400_zero069.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
