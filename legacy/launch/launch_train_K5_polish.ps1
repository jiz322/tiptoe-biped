# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# P9 K5 on GPU 0: FINAL-POLISH arm from K4b model_3600.
# Context (2026-08-23 evening): K4b/K4b2 both plateaued under the patched
# scheduler — K4b reward flat ~16.0k since iter 3560, K4b2 flat ~15.4k
# since 3500, noise_std stuck 0.36-0.38 → current PPO params at
# equilibrium, further iters were sideways. Both killed (user directive:
# reallocate GPUs to best-value work). K4b2's purpose (seed-45
# reproducibility of the K4 recipe) is COMPLETE: held healthy 270+ iters
# past branch, through the 3390-3430 hazard zone that killed every
# unpatched run.
# Branch point model_3600 = the P11 primary deploy pick (user-judged best
# gait; eval 08-23: hw_matched 15s survival 1.000/1024 envs — zero falls,
# armature x1.3 corner 0.991, det. per-joint jerk 0.21; see
# docs/p9-checkpoint-selection-20260823.md).
# THIS ARM = the documented "final polish" endgame (the P8 G2-lineage
# recipe that produced the sigma 0.11 flown policies):
#   entropy_coef 0.005 -> 0.003   (polish value per agent cfg comment)
#   desired_kl        -> 0.002    ("0.008 fresh / 0.002 final polish";
#                                  with adaptive schedule pins lr at the
#                                  1e-5 floor once annealed — slow, safe
#                                  drift; ceiling 2e-4 patch still active,
#                                  verify `grep "LOCAL PATCH" ppo.py`
#                                  before launching)
# Goal: anneal sigma down from 0.37, tighten the gait — a possibly-better
# P11 candidate. Deploy does NOT wait on this arm; K4b_3600 flies as-is
# if polish doesn't beat it on the eval gates + viewer.
# Readout: sigma trending down + reward holding >= 15k = polish working;
# eval any candidate ckpt via RL/run_eval_p9sel_GPU2.ps1 (add manifest
# entry). Kill rule: reward collapse (not vloss spikes — patched-scheduler
# transients recover, see 3520 episode).
# K5b_polish_seed45 (GPU 1) = seed-only twin from the SAME model_3600.
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
  '--run_name', 'K5_polish',
  '--seed', '44',
  '--resume',
  '--load_run', '2026-08-23_08-56-43_K4b_resume3420',
  '--checkpoint', 'model_3600.pt',
  'agent.algorithm.entropy_coef=0.003',
  'agent.algorithm.desired_kl=0.002',
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
  -RedirectStandardOutput '<REPO>\RL\logs\train_K5_polish.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_K5_polish.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
