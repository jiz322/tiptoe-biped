# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# P9 K3s2 on GPU 0: SEED REPLICATE of K3s (GPU 1) — identical config
# (zero 0.078, clr scale 4.0, doubled jerk penalties), seed 43 vs 42.
# Context (2026-08-22): K3 (clr 2.0, GPU 0) diverged at ~iter 2450 —
# raw_action penalty ran away (-1164/step, value loss 3e8), unrecoverable
# after 500+ iters; killed. K3s meanwhile is the healthy arm: reward
# 10-11k, active_frac touching the 0.20-0.27 deploy band, idle_swing_
# above_contact_frac 0.089 -> 0.29 (real lifts, not shuffle) and clearer
# stepping in the 2550 rollout. This arm answers two questions at once:
#   1. Is the clr4 recipe seed-robust (K/K2 pattern) before it becomes
#      the P11 deploy line?
#   2. Was the K3 divergence stochastic or clr2-specific? A seed-43
#      divergence here would implicate the shared recipe (doubled jerk
#      penalties), not the clearance scale.
# A/B contrast vs K3s is --seed ONLY.
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
  '--run_name', 'K3s2_seed43_clr4',
  '--seed', '43',
  '--resume',
  '--load_run', '2026-08-20_06-45-41_K_r400_zero069',
  '--checkpoint', 'model_2040.pt',
  'agent.algorithm.entropy_coef=0.005',
  'env.rew_scale_gait_ref=3.0',
  'env.sat_term_threshold=0.5',
  'env.sat_term_tau_s=0.5',
  'env.swing_clearance_zero_m=0.078',
  'env.rew_scale_swing_clearance=4.0',
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
  -RedirectStandardOutput '<REPO>\RL\logs\train_K3s2_seed43_clr4.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_K3s2_seed43_clr4.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
