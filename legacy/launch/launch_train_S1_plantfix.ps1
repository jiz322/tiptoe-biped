# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# S-line arm A — "S1_plantfix" — GPU 0 (RTX 4090, selected by UUID), 2026-09-06.
#
# CONTEXT: 09-03 the R1r_3150 policy stood 6 s on hardware only with the IMU
# pitch trim pushed +0.12 past the true mounting offset — the trim is a
# plant-gap compensator. Two training-distribution holes were found:
#   (1) dr_com_offset_max was drawn PER LINK, PER AXIS, independently, so the
#       whole-body CoM never shifted coherently (std ~5 mm at 0.02; verified
#       by scripts/rsl_rl/verify_com_shift.py). The policy never trained
#       against a persistent balance-point bias. FIXED: dr_com_shift_x/y_max
#       = one per-env whole-body CoM shift, realized on the torso × M/m_torso.
#   (2) The kneel→stand curriculum was OFF in every K/L/R run. This arm turns
#       it ON (S-line A/B share this).
#
# A/B DESIGN (user call, dev speed first — no fine ablations):
#   A = THIS ARM: minimal plant-fix bundle on top of the exact R1r_3150 recipe.
#   B = launch_train_S2_directive.ps1: A + knee joint-angle bias DR + handover
#       prime randomization + wider static-lean spawn + CoM shift 0.03.
#   Both fork R1_resume3110/model_3150 — via model_3150_currnorm.pt, the same
#   weights with the obs-normalizer's `difficulty`/`grace_norm` channels
#   (std=0 in the original: never varied) set to a [0,1] prior. Without that
#   patch a kneel start feeds ×50–100 inputs to actor and critic.
#
# CURRICULUM PARAMS (from R1r_3150 TB): max_episode_length = 3000 steps; the
# stand branch averaged 848 steps (0.28) under this DR. The default advance
# threshold 0.5 can therefore never fire. 0.20 (= 600 steps) means the kneel
# branch survives ~70 % as long as the stand branch = mastery of d∈[0,alpha].
# alpha_step 0.1 → 7 advances; cooldown 1500 steps ≈ 62 iters each →
# alpha 1.0 reachable in ~450 iters. Window 2000 episodes (500 was ~25 steps
# of resets at 32k envs — too noisy).
# Pickup-on-spawn no longer selects kneel-branch envs (it zeroed their grace
# and hand-lifted them, yet they counted as kneel successes — env fix).
# Reset mix: kneel 0.4; pickup 0.4 of the stand branch → 40 % kneel /
# 24 % hand-placed / 36 % plain ground stand (the deploy scenario).
#
# DELTAS vs launch_train_R1_resume3110.ps1 (everything else re-passed verbatim):
#   --checkpoint model_3150_currnorm.pt (+ --reset_optimizer)
#   env.dr_com_shift_x_max=0.02  env.dr_com_shift_y_max=0.01   (THE FIX)
#   env.dr_proj_grav_bias_max 0.10 → 0.13   (deploy trim regime −0.12 vs true)
#   env.dr_mass_mult 0.94–1.18 → 0.95–1.20  (recenter on measured 5.57/5.195)
#   env.curriculum_enable=true + params above; kneel 0.4; pickup_initial 0.4
#
# PREDICTIONS: reward is NOT comparable to R1r's 2764 (standup terms change
# the scale; kneel episodes are short at first). Expect an initial drop, then
# curriculum/kneel_mean_ep_ratio climbing toward 0.20 and
# curriculum/standup_alpha stepping 0.3→1.0 within ~500–800 iters.
# WATCH: diagnostics/uprightness (stand-branch health) should stay > 0.85;
# dr/com_shift_x_abs_mean ≈ 0.010; curriculum/kneel_branch_frac ≈ 0.40;
# pickup/active_frac below R1r's 0.30.
# KILL RULE: sustained Train/mean_reward downtrend > 300 iters after the
# initial reset drop, or diagnostics/raw_action_max_abs > 40 sustained
# (K4-style runaway), or alpha stuck at 0.3 for > 1000 iters (then lower
# the threshold on a resume — re-pass EVERY override).
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
  '--run_name', 'S1_plantfix',
  '--resume',
  '--load_run', '2026-08-27_10-30-37_R1_resume3110',
  '--checkpoint', 'model_3150_currnorm.pt',
  '--reset_optimizer',
  # PPO params (R1_resume3110 level)
  'agent.algorithm.entropy_coef=0.005',
  'agent.algorithm.desired_kl=0.008',
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
  # kneel→stand curriculum ON
  'env.curriculum_enable=true',
  'env.curriculum_initial_alpha=0.3',
  'env.curriculum_max_alpha=1.0',
  'env.curriculum_alpha_step=0.1',
  'env.curriculum_advance_threshold=0.20',
  'env.curriculum_advance_window=2000',
  'env.curriculum_advance_cooldown=1500',
  'env.kneel_start_probability=0.4',
  'env.pickup_initial_probability=0.4'
)
$p = Start-Process -FilePath "$envRoot\python.exe" `
  -WorkingDirectory '<REPO>\RL' `
  -ArgumentList $argList `
  -RedirectStandardOutput '<REPO>\RL\logs\train_S1_plantfix.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_S1_plantfix.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
