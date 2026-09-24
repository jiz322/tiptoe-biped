# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# P9 K3 arm A on GPU 0: resume K_r400 model_2040 with the swing zero
# RE-recalibrated to the rule value for the ACTUAL settled stance.
# Diagnosis (2026-08-21, from K_r400/K2 tensorboard): idle_toe_z_p10
# converged to 0.075 in BOTH runs — not the 0.066 the 0.069 zero was
# calibrated from (that number came from K@384, mid-descent stance).
# zero=0.069 therefore sits 6 mm BELOW the resting toe: idle_swing_
# above_zero_frac = 0.997, i.e. the dense clearance ramp pays ~0.15-0.22
# of max passively while standing — the documented 08-10 bug reproduced,
# third time this lineage. Result: weak lift gradient, 3 mm shuffle
# (idle_swing_above_contact_frac 0.089, idle_liftoff_rate 0.020).
# Rule: zero = p10 + 0.003 = 0.078. Contact threshold 0.084 is now only
# 6 mm above ramp start; full ramp at 0.118.
# Also (shared with arm B): action_smoothness/snap doubled (-0.16/-0.04)
# for the jerk goal — K_r400 parked at jerk_abs_mean 0.95-0.97 and the
# K kp30 knee sits near the 4.2-4.4 Hz belt mode; jerk is excitation.
# Design intent (owner): idle target is marching in place with REAL
# clearance, not quiet standing — the zero fix is the anti-degenerate-
# march lever, not an anti-stepping one.
# A/B contrast vs K3s (GPU 1) is rew_scale_swing_clearance ONLY (2 vs 4).
# NOTE: resume re-reads cfg defaults — ALL overrides re-passed here.
$ErrorActionPreference = 'Stop'

$envRoot = '<CONDA_ENV>'
$env:CUDA_VISIBLE_DEVICES = '0'
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
  '--run_name', 'K3_zero078',
  '--seed', '42',
  '--resume',
  '--load_run', '2026-08-20_06-45-41_K_r400_zero069',
  '--checkpoint', 'model_2040.pt',
  'agent.algorithm.entropy_coef=0.005',
  'env.rew_scale_gait_ref=3.0',
  'env.sat_term_threshold=0.5',
  'env.sat_term_tau_s=0.5',
  'env.swing_clearance_zero_m=0.078',
  'env.rew_scale_action_smoothness=-0.16',
  'env.rew_scale_action_snap=-0.04',
  'env.servo_kp_hip_pitch=20.0',
  'env.servo_kd_hip_pitch=1.4',
  'env.servo_kp_knee=30.0',
  'env.servo_kd_knee=1.5'
)
$p = Start-Process -FilePath "$envRoot\python.exe" `
  -WorkingDirectory '<REPO>\RL' `
  -ArgumentList $argList `
  -RedirectStandardOutput '<REPO>\RL\logs\train_K3_zero078.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_K3_zero078.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
