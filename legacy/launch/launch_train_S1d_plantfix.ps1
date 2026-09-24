# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# S-line arm A, RELAUNCH #3 — "S1d_plantfix" — GPU 0 (RTX 4090, by UUID), 2026-09-08.
#
# WHY: S1c (launch_train_S1c_plantfix.ps1, iters 3910→4830) was healthy and
# the grace-corrected criterion worked (0.5→0.6 @3934, 0.6→0.7 @4685), but
# the curriculum is too slow for the target: the user needs alpha 1.0 (d=1.0
# = the prone-fold ground-rest pose the robot is placed in). Measured:
#   * frontier learning rate ≈ +2 pts stand-up success / 100 iters at fixed d
#     (d≈0.50: 0.38→0.54 over 751 iters ≈ 19 h per alpha step, getting
#     harder with depth; baseline d=0.8 success 0.01).
#   * DILUTION: d ~ U[0, alpha] on 40 % of resets → only ~6 % of episodes in
#     the frontier band [alpha−0.1, alpha]; the rest re-train mastered poses.
#   * CEILING: the low tertile (d≈0.10, near-standing start) succeeds only
#     0.80 — spawn/push/DR falls within 1.4 s, unrelated to kneel depth —
#     so the absolute 0.6 bar is 75 % of the attainable ceiling here and
#     88 % for arm B (low 0.68), which is why S2c never advanced past 0.6.
#
# CHANGES vs launch_train_S1c_plantfix.ps1:
#   --load_run <S1c run> --checkpoint model_4840.pt (no reset_optimizer)
#   env.curriculum_initial_alpha 0.5 → 0.7   (S1c's current alpha; not in ckpt)
#   env.curriculum_frontier_frac=0.5, env.curriculum_frontier_width=0.3
#       (NEW, rl_env_p6 09-08): half of kneel-branch resets draw d from
#       U[alpha−0.3, alpha]; the other half keep U[0, alpha] (easy regime
#       never removed). Frontier share of all episodes ≈ 6 % → ~20 %; the
#       window's high tertile now sits ≈ alpha−0.07 instead of alpha−0.12.
#   env.curriculum_advance_metric success_high → success_high_rel
#       (NEW): success_high / max(success_low, 0.05) ≈ P(stands up | no early
#       fall); env.curriculum_advance_threshold 0.6 → 0.7 (≈ 0.56 absolute at
#       the current ceiling, slightly looser than S1c's 0.6 — cheap because
#       the frontier band keeps the just-advanced-past d dense).
#   everything else identical to S1c.
#
# PREDICTIONS: 0.7→0.8 within ~400 iters (was heading for ~1500); if the
# d≥0.8 cliff (baseline 0.01) is physical, standup_success_high will sit
# near 0 at alpha 0.9–1.0 with postgrace_len_high ≈ 0 for >1000 iters — that
# is the "torque/geometry wall" verdict, not a training bug; check with
# eval_alpha_success.py --alphas 0.8,0.9,1.0 before touching PPO.
# NEW TB channel: curriculum/frontier_lo. WATCH/KILL: as S1c.
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
  '--run_name', 'S1d_plantfix',
  '--resume',
  '--load_run', '2026-09-07_07-38-26_S1c_plantfix',
  '--checkpoint', 'model_4840.pt',
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
  # kneel→stand curriculum ON — ceiling-relative criterion + frontier sampling (09-08)
  'env.curriculum_enable=true',
  'env.curriculum_initial_alpha=0.7',
  'env.curriculum_max_alpha=1.0',
  'env.curriculum_alpha_step=0.1',
  'env.curriculum_advance_metric=success_high_rel',
  'env.curriculum_success_post_grace_steps=100',
  'env.curriculum_advance_threshold=0.7',
  'env.curriculum_frontier_frac=0.5',
  'env.curriculum_frontier_width=0.3',
  'env.curriculum_advance_window=2000',
  'env.curriculum_advance_cooldown=6000',
  'env.kneel_start_probability=0.4',
  # pickup OFF (mid-episode and on-spawn)
  'env.pickup_enable=false'
)
$p = Start-Process -FilePath "$envRoot\python.exe" `
  -WorkingDirectory '<REPO>\RL' `
  -ArgumentList $argList `
  -RedirectStandardOutput '<REPO>\RL\logs\train_S1d_plantfix.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_S1d_plantfix.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
