# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# P9 K4b2 on GPU 1: SEED-TWIN of K4b_resume3420 — same branch point
# (K4's model_3420), same overrides, same patched lr ceiling, seed 45.
# Context (2026-08-23): K4b (GPU 0, seed 44) validated the lr-ceiling
# patch short-term — 70 iters, reward 14-17k (above K4's pre-collapse
# 13-14k), vloss ~5e3, lr capped exactly 2e-4 with zero excursions.
# K3s (killed 08-23 ~11:15, PID 49000) never got the patch and bled out
# 13.8k -> 6.2k with vloss spikes still growing => in-process runs can't
# be saved, only relaunches pick up the patch.
# Purpose of this arm (the K3s2 playbook, now under the patched scheduler):
#   1. Seed-robustness: every pre-patch arm on this recipe eventually
#      diverged (K3, K3s2, K4 dead; K3s degraded) and K3s was declared
#      "the lucky seed". If seed 45 ALSO holds under the 2e-4 ceiling,
#      the luck era is over and the lr-ceiling diagnosis is confirmed
#      as THE fix — the recipe becomes reproducible for P11.
#   2. Deploy redundancy: an independent second policy from the same
#      lithe branch point for hardware A/B; replaces K3s as fallback
#      (K3s's fallback role is obsolete — K4 line is the smoothness
#      branch built FOR the ~1s-survival hardware goal).
# Divergence watch unchanged: kill if value_function loss >1e5 sustained;
# pickup/active_frac should hold the 0.20-0.27 deploy band.
# Overrides IDENTICAL to launch_train_K4b_resume3420.ps1 — only seed
# and GPU differ. NOTE: resume re-reads cfg defaults — ALL re-passed.
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
  '--run_name', 'K4b2_seed45',
  '--seed', '45',
  '--resume',
  '--load_run', '2026-08-22_17-26-54_K4_smooth024',
  '--checkpoint', 'model_3420.pt',
  'agent.algorithm.entropy_coef=0.005',
  'env.rew_scale_gait_ref=3.0',
  'env.sat_term_threshold=0.5',
  'env.sat_term_tau_s=0.5',
  'env.swing_clearance_zero_m=0.078',
  'env.rew_scale_swing_clearance=4.0',
  'env.rew_scale_action_smoothness=-0.24',
  'env.rew_scale_action_snap=-0.06',
  'env.servo_kp_hip_pitch=20.0',
  'env.servo_kd_hip_pitch=1.4',
  'env.servo_kp_knee=30.0',
  'env.servo_kd_knee=1.5'
)
$p = Start-Process -FilePath "$envRoot\python.exe" `
  -WorkingDirectory '<REPO>\RL' `
  -ArgumentList $argList `
  -RedirectStandardOutput '<REPO>\RL\logs\train_K4b2_seed45.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_K4b2_seed45.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
