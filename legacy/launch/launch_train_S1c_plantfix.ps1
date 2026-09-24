# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# S-line arm A, RELAUNCH #2 — "S1c_plantfix" — GPU 0 (RTX 4090, by UUID), 2026-09-07.
#
# WHY: S1b (launch_train_S1b_plantfix.ps1, iters 3270→~3890) was STABLE
# (raw max 4–5, vloss 6–7k, sigma 0.52→0.46, lr at the 1e-3 ceiling) but its
# curriculum ran alpha 0.4→1.0 between iters 3294 and 3528 — one step every
# 47 iters, i.e. purely on the cooldown clock. The advance criterion
# `kneel_mean_ep_len/3000 ≥ 0.12` was ALWAYS satisfied because the kneel
# branch's grace (d × 4 s) counts toward ep_len: a deep-kneel env that never
# stands still contributes ~330 steps. Grace-corrected survival
# (ep_len − 400·d) per difficulty tertile, TB 09-07:
#     d≈0.33: +456   d≈0.42: +230   d≈0.50: +70   d≈0.58: +40
#     d≈0.67: +7     d≥0.75: 0 ± 1 (sinks to kneel height during grace,
#                                    dies too_low the step grace expires)
# → at alpha 1.0 a third of kneel resets are guaranteed fails with no
# learning signal, and the arm was no longer testing the curriculum.
#
# CHANGES vs launch_train_S1b_plantfix.ps1:
#   --load_run <S1b run> --checkpoint model_<latest>.pt (no reset_optimizer;
#       350 extra stable iters on the easy/mid bands are kept)
#   env.curriculum_advance_metric=success_high   (NEW knob, rl_env_p6
#       09-07): advance when the HARDEST difficulty tertile of the window
#       has stand-up success ≥ threshold. Success = survived ≥
#       curriculum_success_post_grace_steps (100 = 1 s) after grace expiry.
#   env.curriculum_advance_threshold 0.12 → 0.6   (now a success RATE)
#   env.curriculum_initial_alpha 0.4 → 0.5   (frontier band d 0.33–0.5 is
#       partially mastered — that is where the learning signal is; the
#       fixed criterion holds alpha there until it actually is mastered)
#   everything else identical to S1b (desired_kl 0.02, pickup OFF,
#       cooldown 6000 ≈ 47 iters, window 2000 kneel episodes).
#
# NEW TB channels: curriculum/standup_success_{low,mid,high,all},
# curriculum/postgrace_len_{low,mid,high}, curriculum/advance_metric_value.
# WATCH: standup_success_high climbing toward 0.6 at alpha 0.5 → advance;
# standup_success_low should stay ≥ 0.8. KILL: as S1b (raw_action_max_abs
# > 40 sustained, vloss spikes > 5e4 recurring, reward downtrend > 300 iters).
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
  '--run_name', 'S1c_plantfix',
  '--resume',
  '--load_run', '2026-09-06_14-14-08_S1b_plantfix',
  '--checkpoint', 'model_3910.pt',
  # PPO params — desired_kl paired with the 1e-3 lr ceiling
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
  # kneel→stand curriculum ON — grace-corrected success criterion (09-07)
  'env.curriculum_enable=true',
  'env.curriculum_initial_alpha=0.5',
  'env.curriculum_max_alpha=1.0',
  'env.curriculum_alpha_step=0.1',
  'env.curriculum_advance_metric=success_high',
  'env.curriculum_success_post_grace_steps=100',
  'env.curriculum_advance_threshold=0.6',
  'env.curriculum_advance_window=2000',
  'env.curriculum_advance_cooldown=6000',
  'env.kneel_start_probability=0.4',
  # pickup OFF (mid-episode and on-spawn)
  'env.pickup_enable=false'
)
$p = Start-Process -FilePath "$envRoot\python.exe" `
  -WorkingDirectory '<REPO>\RL' `
  -ArgumentList $argList `
  -RedirectStandardOutput '<REPO>\RL\logs\train_S1c_plantfix.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_S1c_plantfix.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
