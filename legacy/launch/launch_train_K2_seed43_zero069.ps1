# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# P10 K-line insurance twin on GPU 0: FRESH, seed 43, K servo gains, swing
# zero 0.069 baked from iter 0.
# Two purposes: (1) redundancy for the K line (the resume arm carries a
# mid-run reward-landscape shift from the zero change; this twin has no such
# shift), (2) clean test of the zero rule on the stiff-servo stance — K's
# settled p10 was 0.066 → rule value 0.069. Early-stance p10 starts ~0.064
# so 0.069 is ~2 mm above rule at iter 0, converging onto it as stance
# settles — inside the tolerance every converged P8 run had.
# W (no-coupling, GPU0) was killed: flat 0.71-0.72 for 320 iters → coupling
# echo exonerated as the blocker; coupling stays ON here (real physics).
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
  '--run_name', 'K2_seed43_zero069',
  '--seed', '43',
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
  -RedirectStandardOutput '<REPO>\RL\logs\train_K2_seed43_zero069.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_K2_seed43_zero069.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
