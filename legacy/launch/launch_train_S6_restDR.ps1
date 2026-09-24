# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# S-line arm A, RELAUNCH #7 — "S6_restDR" — GPU 0 (RTX 4090, by UUID), 2026-09-14.
#
# WHAT THIS IS
#   Resume the S4a_measDR line from its latest checkpoint (model_7780.pt) with
#   ONE change: foot-ball RESTITUTION becomes a per-env DR channel.  Every other
#   override is carried over verbatim from launch_train_S4a_measDR.ps1 — a resume
#   reverts anything not re-passed, so the recipe is complete, not incremental.
#   Stock PhysX only: NO pad, NO belt, no new physics code.  The S5 pad/belt band
#   (killed 09-14, value divergence) is NOT re-entered here.
#
# HOW THE DR IS REALIZED (S6 wiring, see rl_env_p6.py / rl_env_cfg_p6.py)
#   The foot material spawner (FootContactUsdFileCfg) is kept ACTIVE with
#   restitution_combine_mode="max" and foot_restitution=0.63.  That nonzero
#   carrier is what keeps the binding: if foot_restitution were 0 the spawner
#   takes the legacy branch and the balls fall back to combine mode "average",
#   silently halving the bounce.  0.63 is the ROW MARKER and the combine-mode
#   carrier, NOT the training value.
#   root_physx_view.get_material_properties() is a per-env, per-SHAPE buffer
#   (N, max_shapes, 3) = static_friction, dynamic_friction, restitution.  The env
#   DISCOVERS the two ball rows at build time (rows whose restitution reads the
#   carrier 0.63; asserts exactly two and every other row reads 0), then each
#   reset draws e ~ U(0.4, 0.8) bucketed into 64 bins and writes
#   [1, 1, e] to ONLY those two rows per env.  Friction keeps its own columns.
#   The dr/foot_restitution_{min,mean,max,std} TB channels are READ BACK from the
#   buffer each iteration, not copied from the sampled tensor.
#   bounce_threshold_velocity is set to 0.05 in the env cfg (default 0.5): balls
#   now bounce on low-speed contact instead of damping it out.
#
# PHASE A (smoke): run with --max_iterations 5, confirm the [S6] row-discovery
#   line, the read-back span covers 0.4–0.8, and value_function stays stable,
#   then STOP for approval.  PHASE B (after approval): drop --max_iterations and
#   resume the full run.
#
# PRE-REGISTERED KILLS (Phase B): value_function > 10x the S4a endpoint (~4200)
#   for 50 consecutive iters, OR episode_length < 400 at iter +300, OR a PhysX
#   error flood.
# SUCCESS GATE at +1000 iters: episode_length >= 1400 AND dr/foot_restitution
#   spans 0.4–0.8.  Then export the policy and stop.
$ErrorActionPreference = 'Stop'

$envRoot = '<CONDA_ENV>'
$env:CUDA_VISIBLE_DEVICES = 'GPU-<uuid>'   # RTX 4090 (index 0)
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
  # '--max_iterations', '5',     # PHASE A smoke — REMOVE for the Phase B full run
  '--run_name', 'S6_restDR',
  '--resume',
  '--load_run', '2026-09-09_22-13-40_S4a_measDR',
  '--checkpoint', 'model_7780.pt',
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
  # ── delays (S4 measured, unchanged)
  'env.dr_action_delay_min=1',
  'env.dr_action_delay_max=2',
  'env.dr_imu_delay_min=1',
  'env.dr_imu_delay_max=1',
  'env.dr_joint_delay_min=1',
  'env.dr_joint_delay_max=1',
  # ── plant DR (S4 measured, unchanged)
  'env.dr_floor_friction_min=0.6',
  'env.dr_floor_friction_max=1.1',
  'env.dr_stiffness_mult_min=0.90',
  'env.dr_stiffness_mult_max=1.10',
  'env.dr_armature_mult_min=0.80',
  'env.dr_armature_mult_max=1.20',
  'env.dr_effort_mult_min=0.9',
  'env.dr_com_offset_max=0.01',
  'env.dr_ang_vel_bias_max=0.05',
  'env.dr_gravity_tilt_max_deg=2.0',
  # ── S-line deltas (unchanged)
  'env.dr_com_shift_x_max=0.02',
  'env.dr_com_shift_y_max=0.01',
  'env.dr_proj_grav_bias_max=0.13',
  'env.dr_mass_mult_min=1.00',
  'env.dr_mass_mult_max=1.14',
  # ══ S6 NEW: foot-ball restitution as a per-env DR channel ══════════════
  # carrier (must stay nonzero — see header) + combine-mode marker
  'env.foot_restitution=0.63',
  # band drawn per reset, 64 buckets, written to the two discovered ball rows
  'env.dr_foot_restitution_min=0.4',
  'env.dr_foot_restitution_max=0.8',
  'env.dr_foot_restitution_buckets=64',
  # ── kneel→stand curriculum (unchanged from S4a)
  'env.curriculum_enable=true',
  'env.curriculum_initial_alpha=0.5',
  'env.curriculum_max_alpha=1.0',
  'env.curriculum_alpha_step=0.1',
  'env.curriculum_advance_metric=success_high_rel',
  'env.curriculum_success_post_grace_steps=100',
  'env.curriculum_advance_threshold=0.7',
  'env.curriculum_frontier_frac=0.5',
  'env.curriculum_frontier_width=0.3',
  'env.curriculum_advance_window=2000',
  'env.curriculum_advance_cooldown=6000',
  'env.kneel_start_probability=0.6',
  'env.pickup_enable=false',
  # ── stand-up PATH mode (unchanged from S4a)
  'env.standup_ramp_mode=path',
  'env.standup_grace_seconds=5.0',
  'env.standup_path_spawn_clearance_m=0.006',
  'env.rew_scale_standup_track=4.0',
  'env.rew_scale_standup_contact=2.0'
)
$p = Start-Process -FilePath "$envRoot\python.exe" `
  -WorkingDirectory '<REPO>\RL' `
  -ArgumentList $argList `
  -RedirectStandardOutput '<REPO>\RL\logs\train_S6_restDR.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_S6_restDR.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
