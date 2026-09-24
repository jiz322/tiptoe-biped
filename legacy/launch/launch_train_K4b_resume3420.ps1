# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# P9 K4b on GPU 0: REVIVAL of K4_smooth024 from its own model_3420.
# Context (2026-08-23): K4 diverged at iter 3429->3430 (07:32) — value_function
# loss 5e3 -> 1.08e7 in one iteration, raw_action_max 21 -> 354, reward
# 13.6k -> -150, unrecoverable. K3s hit the same failure mode at ~3390 the
# same morning. Forensics (RL/logs/_probe_drop*.py): NOT a machine event
# (fps flat through both drops); trigger is rsl_rl's adaptive-KL schedule —
# 32 minibatches x 3 epochs = 96 lr updates/iter lets lr ride the 1e-5 floor
# to the 1e-2 ceiling INSIDE one iteration (lr>1e-3 on ~21-25% of iters in
# K3s2/K3s/K4; every fatal collapse preceded by a ceiling ride). Recipe tally:
# K3 dead, K3s2 dead ~2150, K3s degraded ~3390, K4 dead 3430 — the scheduler,
# not the seed, is the fragility.
# Fix: rsl_rl ppo.py locally patched 2026-08-23 (ceiling 1e-2 -> 2e-4, marked
# "LOCAL PATCH (big_thigh 2026-08-23)"). Adaptive breathing kept (1e-5..2e-4);
# picked up by newly launched processes only. If instability persists, next
# lever is agent.algorithm.desired_kl=0.002 ("final polish" per cfg comment)
# to pin lr at the floor outright. Do NOT use schedule=fixed on a resume —
# the optimizer reloads the checkpoint's saved lr and fixed never overwrites it.
# model_3420 is clean (reward 12.9k, vloss 5.25e3, raw_action_max 8.5) and is
# the checkpoint judged most lithe in the viewer — also exported as a P11
# deploy candidate independent of this run.
# Everything below except run_name/seed-source/resume-target is IDENTICAL to
# launch_train_K4_smooth024.ps1 — single-variable change is the lr ceiling.
# NOTE: resume re-reads cfg defaults — ALL overrides re-passed here.
# Kill the old K4 process before launching (same GPU).
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
  '--run_name', 'K4b_resume3420',
  '--seed', '44',
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
  -RedirectStandardOutput '<REPO>\RL\logs\train_K4b_resume3420.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_K4b_resume3420.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
