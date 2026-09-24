# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# Plant sysid vs the hardware balance trials (2026-09-10) — CMA-ES / Optuna in
# ONE vectorized Isaac process on GPU2. See scripts/rsl_rl/sysid_optuna.py for
# the question, search space, cells, objective and PRE-REGISTERED readings.
#
# Search : com_x, com_z (coherent whole-body CoM shift, m), knee_bias, hip_bias
#          (coupled encoder/servo zero error, rad, L convention, R mirrored)
# Pinned : total mass 5.57 kg (weigh-in), floor friction mult 1.0, pg_x obs
#          bias = -trim per cell; every other band = S4 measured-plant recipe
#          (launch_train_S4a_measDR.ps1) minus curriculum/pickup. The recipe's
#          random coherent CoM_x draw is OFF here (it is the search variable).
# Fit    : R1r @ trim -0.089 / 0 / +0.089      Held-out prediction: S4a @ 0 / +0.089
# Clean window (user 09-10): only the first 0.6 s of each hardware trial (0.8 s
# for s4a 143406) enters the objective — the tether may act after the first dip.
# Search episodes are 1 s; the final top-k re-evaluation runs 5 s for the
# long-horizon prediction in the report.
#
# Runs in the FOREGROUND: the tqdm bar (with time remaining) is on the terminal;
# every print also goes to <out>\run.log. Results: <out>\report.txt, best.json,
# trials.csv, generations.csv, study.db (optuna), topk\*.npz (simlog rollouts).
#
# Usage:  powershell -File run_s4_sysid_GPU2.ps1                 # full run (~30 gens, ~10 min)
#         powershell -File run_s4_sysid_GPU2.ps1 -Sensitivity    # one-at-a-time perturbation table (~3 min)
#         powershell -File run_s4_sysid_GPU2.ps1 -Smoke          # 2-min pipeline check
#         powershell -File run_s4_sysid_GPU2.ps1 -Name myrun -Gens 40 -Pop 24 -Seeds 8
#         powershell -File run_s4_sysid_GPU2.ps1 -Name myrun -Resume   # continue a run
param(
  [string]$Name = '',
  [int]$Gens = 30,
  [int]$Pop = 24,
  [int]$Seeds = 8,
  [double]$EpisodeS = 1.0,
  [double]$FinalEpisodeS = 5.0,
  [double]$TotalMassKg = 5.57,
  [double]$FrictionMult = 1.0,
  [string]$Sampler = 'cma',
  [switch]$Smoke,
  [switch]$Sensitivity,
  [switch]$Resume
)
$ErrorActionPreference = 'Stop'

$envRoot = '<CONDA_ENV>'
$env:CUDA_VISIBLE_DEVICES = 'GPU-<uuid>'   # RTX 4060 Ti (GPU2)
$env:CARB_APP_PATH = '<ISAACLAB>\_isaac_sim\kit'
$env:EXP_PATH      = '<ISAACLAB>\_isaac_sim\apps'
$env:ISAAC_PATH    = '<ISAACLAB>\_isaac_sim'
$env:PYTHONPATH    = ';<ISAACLAB>\_isaac_sim\site'
$env:RESOURCE_NAME = 'IsaacSim'
$env:CONDA_PREFIX  = $envRoot
$env:PATH = "$envRoot;$envRoot\Library\mingw-w64\bin;$envRoot\Library\usr\bin;$envRoot\Library\bin;$envRoot\Scripts;$envRoot\bin;" + $env:PATH
$env:PYTHONUNBUFFERED = '1'

$RL = '<REPO>\RL'
if ($Name -eq '') { $Name = (Get-Date -Format 'yyyyMMdd_HHmm') + $(if ($Smoke) { '_smoke' } elseif ($Sensitivity) { '_sens' } else { '' }) }
$outDir = Join-Path $RL "logs\s4_sysid\$Name"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
Set-Location $RL

$R1R = "$RL\logs\rsl_rl\p9_replant\2026-08-27_10-30-37_R1_resume3110\model_3150.pt"
$S4A = "$RL\logs\rsl_rl\p9_replant\2026-09-09_22-13-40_S4a_measDR\model_6520.pt"

# S4 measured-plant recipe (run_s4_hwtwin_GPU2.ps1 $S4_OVR) minus the knobs the
# driver pins per env (mass, friction, pg_x bias, coherent CoM shift).
$PLANT = @(
  'env.rew_scale_gait_ref=3.0',
  'env.sat_term_threshold=2.0', 'env.sat_term_tau_s=0.5',
  'env.swing_clearance_zero_m=0.078',
  'env.servo_kp_hip_pitch=20.0', 'env.servo_kd_hip_pitch=1.4',
  'env.servo_kp_knee=30.0', 'env.servo_kd_knee=1.5',
  'env.hip_pitch_armature=0.0385',
  'env.exec_noise_target_bias_max=0.03', 'env.exec_noise_target_std=0.015',
  'env.dr_action_delay_min=1', 'env.dr_action_delay_max=2',
  'env.dr_imu_delay_min=1', 'env.dr_imu_delay_max=1',
  'env.dr_joint_delay_min=1', 'env.dr_joint_delay_max=1',
  'env.dr_stiffness_mult_min=0.90', 'env.dr_stiffness_mult_max=1.10',
  'env.dr_armature_mult_min=0.80', 'env.dr_armature_mult_max=1.20',
  'env.dr_effort_mult_min=0.9',
  'env.dr_com_offset_max=0.01',
  'env.dr_com_shift_x_max=0.0', 'env.dr_com_shift_y_max=0.01', 'env.dr_com_shift_z_max=0.0',
  'env.dr_proj_grav_bias_max=0.13',
  'env.curriculum_enable=false', 'env.kneel_start_probability=0.0',
  'env.pickup_enable=false',
  'env.standup_ramp_mode=path'
)

$argList = @('-u', "$RL\scripts\rsl_rl\sysid_optuna.py",
  '--task', 'Template-standup_curriculum-v0', '--headless',
  '--ckpt-r1r', $R1R, '--ckpt-s4a', $S4A, '--out', $outDir,
  '--gens', "$Gens", '--episode-s', "$EpisodeS", '--final-episode-s', "$FinalEpisodeS",
  '--total-mass-kg', "$TotalMassKg", '--friction-mult', "$FrictionMult",
  '--sampler', $Sampler, '--seed', '0')
# -Smoke keeps the tiny 4x2 layout unless -Pop/-Seeds are given explicitly
if (-not $Smoke -or $PSBoundParameters.ContainsKey('Pop'))   { $argList += @('--pop', "$Pop") }
if (-not $Smoke -or $PSBoundParameters.ContainsKey('Seeds')) { $argList += @('--seeds', "$Seeds") }
if ($Smoke)       { $argList += '--smoke' }
if ($Sensitivity) { $argList += '--sensitivity' }
if ($Resume) { $argList += '--resume-run' }
$argList += $PLANT

Write-Output "[run ] s4_sysid name=$Name  $(Get-Date -Format HH:mm:ss)  out=$outDir"
Write-Output "[run ] Isaac Sim startup takes ~1-2 min before the progress bar appears."
$ErrorActionPreference = 'Continue'   # Isaac/tqdm write to stderr; PS 5.1 would otherwise abort on it
& "$envRoot\python.exe" @argList
Write-Output "[done] s4_sysid name=$Name exit=$LASTEXITCODE  $(Get-Date -Format HH:mm:ss)  report=$outDir\report.txt"
