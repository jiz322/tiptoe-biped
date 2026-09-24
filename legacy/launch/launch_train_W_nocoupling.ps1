# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# P10 arm W on GPU 0: FROM-SCRATCH, full P8 recipe, correct swing zero,
# rotor_coupling=False — the coupling-echo ablation.
# Rationale: X2/Z2 (both zero-rule-correct) sat flat at active_frac 0.71-0.72
# for 224 iters where every P8 escaper was at 0.66-0.70 and sloping by
# matched iters → the zero rule is necessary-not-sufficient on the P9 plant,
# recipe and entropy are excluded (X2 vs Z2 span them; Z2 std 0.59 = the
# band A_baseline escaped at). The one plant term that acts impulsively at
# touch-down/release — exactly the transition that won't learn — is the
# off-diagonal rotor-coupling echo (one-step-lagged FD q̈, clamp 15 Nm ≈ hip
# full authority, prime suspect in the iter-564 explosion). W deletes ONLY
# that term; diagonals 33.2/36.0e-3 stay. Single diff vs Z.
# W is DIAGNOSTIC-first: if it converges, the fix is a better-conditioned
# coupling implementation (filtered q̈ / tighter clamp / same-step), not
# shipping a policy trained without real physics.
# NOTE: overrides are CLI-only — a resume must re-pass all six.
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
  '--run_name', 'W_nocoupling_swing052',
  'agent.algorithm.entropy_coef=0.005',
  'env.rew_scale_gait_ref=3.0',
  'env.sat_term_threshold=0.5',
  'env.sat_term_tau_s=0.5',
  'env.swing_clearance_zero_m=0.052',
  'env.plant_rotor_coupling=false'
)
$p = Start-Process -FilePath "$envRoot\python.exe" `
  -WorkingDirectory '<REPO>\RL' `
  -ArgumentList $argList `
  -RedirectStandardOutput '<REPO>\RL\logs\train_W_nocoupling.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_W_nocoupling.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
