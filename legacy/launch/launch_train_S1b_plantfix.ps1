# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# S-line arm A, RELAUNCH — "S1b_plantfix" — GPU 0 (RTX 4090, by UUID), 2026-09-06 pm.
#
# WHY A RELAUNCH: S1_plantfix (launch_train_S1_plantfix.ps1, iters 3150→3361)
# ran into the K4-style optimizer runaway from iter ~3290: raw_action_max_abs
# 40→180 sustained, value loss 7e3 → 1e5–6.5e5 spikes, adaptive lr pinned at
# the 1e-5 floor, sigma 0.51→0.58 monotonic, reward 2200→1500, and the EASIEST
# kneel band (d≈0.07) collapsing 500→275 steps together with the stand branch
# → whole-policy degradation, not curriculum difficulty (the hard band d≈0.33
# survived as long as the easy band after the 0.3→0.4 advance at iter 3174).
# Pre-registered kill rule met; killed by user at 3361. Last clean ckpt = 3270.
#
# CHANGES vs launch_train_S1_plantfix.ps1 (user-approved 09-06):
#   --load_run <S1 run> --checkpoint model_3270.pt, NO --reset_optimizer
#       (Adam state is in the ckpt; the reset + 100 iters at the 1e-3 lr
#       ceiling preceded the first KL blowout).
#   agent.algorithm.desired_kl 0.008 → 0.02
#       rsl_rl ppo.py local patch v2 (1e-3 ceiling) says to pair with ~0.02;
#       at 0.008 the lr ping-ponged 1e-5 ↔ 1e-3 within single iterations.
#   env.pickup_enable=false   (user call) — no mid-episode lifts, no hand-
#       placed spawns. Reset mix is now 40 % kneel-branch / 60 % plain stand.
#   env.curriculum_advance_threshold 0.20 → 0.12
#       Under S-line DR the kneel branch ALREADY survives as long as the
#       stand branch (S1: 0.17 vs population 0.18; S2: 0.115 vs 0.13) — the
#       absolute 0.20 measured general DR survival, not standup mastery, and
#       could never fire for arm B.
#   env.curriculum_advance_cooldown 1500 → 6000
#       Cooldown decrements once per env STEP (per _reset_idx call), so 1500
#       was ≈12 iterations, not the 62 the old header claimed. 6000 ≈ 47 iters
#       between advances → 0.4→1.0 in ≥ ~280 iters.
#   env.curriculum_initial_alpha 0.3 → 0.4   (alpha is not stored in the
#       ckpt; 3270 had already earned 0.4 and the hard band was healthy).
#
# WATCH: Loss/learning_rate should now settle between 1e-4 and 1e-3 instead
# of alternating floor/ceiling; Loss/value_function ~7e3 without >5e4 spikes;
# diagnostics/raw_action_max_abs < 40; Policy/mean_noise_std flat ~0.51.
# curriculum/standup_alpha stepping every ~50 iters while
# curriculum/kneel_mean_ep_ratio ≥ 0.12. pickup/active_frac is no longer logged.
# KILL RULE: unchanged — raw_action_max_abs > 40 sustained, vloss spikes > 5e4
# recurring, or Train/mean_reward downtrend > 300 iters. If it recurs at the
# same alpha, the frozen-normalizer/deep-kneel hypothesis (header of the
# 09-06 analysis) gets tested next, not another PPO knob.
$ErrorActionPreference = 'Stop'

$envRoot = '<CONDA_ENV>'
$env:CUDA_VISIBLE_DEVICES = 'GPU-<uuid>'   # RTX 4090 (index 0 on 09-06)
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
  '--run_name', 'S1b_plantfix',
  '--resume',
  '--load_run', '2026-09-06_08-37-24_S1_plantfix',
  '--checkpoint', 'model_3270.pt',
  # PPO params — desired_kl raised to pair with the 1e-3 lr ceiling
  'agent.algorithm.entropy_coef=0.005',
  'agent.algorithm.desired_kl=0.02',
  'agent.algorithm.clip_param=0.12',
  # base K recipe
  'env.rew_scale_gait_ref=3.0',
  'env.sat_term_threshold=0.5',
  'env.sat_term_tau_s=0.5',
  'env.swing_clearance_zero_m=0.078',
  'env.servo_kp_hip_pitch=20.0',
  'env.servo_kd_hip_pitch=1.4',
  'env.servo_kp_knee=30.0',
  'env.servo_kd_knee=1.5',
  # bench-corrected plant
  'env.hip_pitch_armature=0.0385',
  # L1 calm trio
  'env.swing_clearance_target_m=0.015',
  'env.rew_scale_no_flight=-15.0',
  'env.rew_scale_vz_penalty=-8.0',
  # smoothness/snap (R1_resume3110 level)
  'env.rew_scale_action_smoothness=-0.24',
  'env.rew_scale_action_snap=-0.06',
  # exec-noise injection (R1 level)
  'env.exec_noise_target_bias_max=0.03',
  'env.exec_noise_target_std=0.015',
  # spawn-posture DR (R1 level)
  'env.noise_joint_pos=0.18',
  'env.init_base_orient_noise_rad=0.12',
  # pushes ON (R1 level)
  'env.push_enable=true',
  'env.push_probability=0.001',
  'env.push_force_min=10.0',
  'env.push_force_max=40.0',
  'env.push_duration_steps=10',
  # delays (R1 level)
  'env.dr_action_delay_min=1',
  'env.dr_action_delay_max=3',
  'env.dr_imu_delay_min=1',
  'env.dr_imu_delay_max=2',
  'env.dr_joint_delay_min=1',
  'env.dr_joint_delay_max=2',
  # plant DR widened (R1 level)
  'env.dr_floor_friction_min=0.4',
  'env.dr_floor_friction_max=1.1',
  'env.dr_stiffness_mult_min=0.80',
  'env.dr_stiffness_mult_max=1.20',
  'env.dr_effort_mult_min=0.8',
  'env.dr_com_offset_max=0.02',
  'env.dr_ang_vel_bias_max=0.05',
  'env.dr_gravity_tilt_max_deg=2.0',
  # ── S-line deltas ─────────────────────────────────────────
  # THE FIX: coherent whole-body CoM shift (base frame)
  'env.dr_com_shift_x_max=0.02',
  'env.dr_com_shift_y_max=0.01',
  # obs pitch-bias band must contain the deploy trim regime
  'env.dr_proj_grav_bias_max=0.13',
  # mass recenter on measured weight
  'env.dr_mass_mult_min=0.95',
  'env.dr_mass_mult_max=1.20',
  # kneel→stand curriculum ON (relaunch settings)
  'env.curriculum_enable=true',
  'env.curriculum_initial_alpha=0.4',
  'env.curriculum_max_alpha=1.0',
  'env.curriculum_alpha_step=0.1',
  'env.curriculum_advance_threshold=0.12',
  'env.curriculum_advance_window=2000',
  'env.curriculum_advance_cooldown=6000',
  'env.kneel_start_probability=0.4',
  # pickup OFF (mid-episode and on-spawn)
  'env.pickup_enable=false'
)
$p = Start-Process -FilePath "$envRoot\python.exe" `
  -WorkingDirectory '<REPO>\RL' `
  -ArgumentList $argList `
  -RedirectStandardOutput '<REPO>\RL\logs\train_S1b_plantfix.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_S1b_plantfix.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
