# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# P9 K4 on GPU 0: SMOOTHNESS-TUNE branch of K3s — resume from K3s's own
# model_2820 (not the shared model_2040), smoothness/snap penalties x1.5:
#   rew_scale_action_smoothness -0.16 -> -0.24
#   rew_scale_action_snap       -0.04 -> -0.06
# Context (2026-08-22): K3s2 (seed-43 replicate of K3s, GPU 0) diverged
# at ~iter 2140-2170 — value_function loss spiked 5e3 -> 1e6..1.6e7 and
# never recovered, reward collapsed 9.5k -> ~335, limped at 3-4k for
# ~100 iters. Per the K3s2 header's own decision rule: seed-43 divergence
# implicates the SHARED K3 recipe (doubled jerk penalties + resume from
# model_2040), not the clearance scale. Tally on that recipe: K3 dead,
# K3s2 dead, K3s alive => K3s is the lucky seed, recipe is seed-fragile.
# Consequences encoded here:
#   - Branch from K3s model_2820 (a policy already INSIDE the stable
#     basin, reward 11-12k, val loss 5-10k) instead of re-rolling the
#     fragile 2040 resume; K3s continues untouched on GPU 1 as the
#     fallback P11 deploy line if this arm diverges too.
#   - Goal is hardware ~1s survival (vs 0.15s): softer foot strikes /
#     less snap on the recovery step, NOT new reward terms for the
#     occasional stance-foot slide (friction already DR'd 0.5-1.1).
#   - x1.5 not x2: the divergences correlate with the last penalty
#     doubling; go gentle. Watch value_function loss (kill if >1e5
#     sustained) and pickup/active_frac staying in the 0.20-0.27 band.
# A/B contrast vs K3s is smoothness+snap ONLY (plus seed 44, branch pt).
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
  '--run_name', 'K4_smooth024',
  '--seed', '44',
  '--resume',
  '--load_run', '2026-08-21_21-44-12_K3s_zero078_clr4',
  '--checkpoint', 'model_2820.pt',
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
  -RedirectStandardOutput '<REPO>\RL\logs\train_K4_smooth024.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_K4_smooth024.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"