# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# S-line arm A, RELAUNCH #6 — "S5_elasticDR" — GPU 0 (RTX 4090, by UUID), 2026-09-12.
#
# WHAT THIS IS
#   Resume the S4a_measDR line from its latest checkpoint (model_7780.pt) with
#   ONE change: the belt transmission and the ball-foot pad become per-episode
#   DR channels instead of pinned constants.  Every other override is carried
#   over verbatim from launch_train_S4a_measDR.ps1 — a resume reverts anything
#   not re-passed, so the recipe must be complete, not incremental.
#
# WHICH CHANNELS ARE LIVE (and which are not — read this before reading TB)
#   belt_series_stiffness  100 – 1000   knee-joint N m/rad      -> LIVE
#   belt_series_damping      0 – 0.3    motor-side N m s/rad    -> LIVE
#   pad_stiffness        10000 – 200000 floor-normal N/m        -> LIVE IF THE BALLS LOAD
#   pad_damping              0 – 0.3    N s/m explicit          -> LIVE IF THE BALLS LOAD
#   foot_restitution         0.0        -> NOT 0.632.  This line does NOT pass a
#       foot_restitution override, so the cfg default (0.0) is what the run gets.
#       An earlier revision of this header claimed "pinned at the measured 0.632";
#       that was FALSE and is corrected here (2026-09-14).  Restitution cannot be
#       randomized per env, but it can at least be SET — see below for why it
#       still is not set in this script.
#
#   THE FOOT MATERIAL CANNOT BE RANDOMIZED PER ENV.  `foot_restitution` and
#   `foot_contact_stiffness` are written onto ONE material prim shared by all
#   32768 envs (foot_contact.py), and PhysX parses the material when the
#   physics scene attaches — a per-episode write reaches nobody without a full
#   stage re-attach.  The pad force (pad_contact.PadContact) is the per-env
#   substitute for stiffness/damping, and it is why the pad knobs above are
#   the ones that exist.  Restitution has no force-side equivalent that is not
#   a lie: a restitution impulse implemented as a velocity-dependent force
#   double-counts the pad's own damping.  Restitution is therefore left for a
#   per-process sweep (`sweep_plant_elasticity.py --contact-restitution`, which
#   already exists) rather than a per-env band.
#
#   NOTE (2026-09-14): this run was KILLED at iteration 7787 — value_function
#   diverged to 1.5e16 and raw_action_max_abs hit 1247.  Cause: the pad band
#   above is 3-20x too stiff for the load the foot carries (half-body-weight per
#   foot is 27.3 N; k=200000 at the 8 mm pad reaches 1600 N = 29x body weight).
#   See docs/s5-elastic-dr-20260914.md §4.  Do NOT re-launch this band — re-enter
#   the pad as pad_stiffness_min=5000 pad_stiffness_max=30000 pad_thickness=0.012.
#
# PRE-REGISTERED READINGS (decide these BEFORE looking at any curve)
#   E0 ENGAGEMENT — read `foot/ball_contact_frac` in TB first.
#        == 0.000 for the whole run  -> every foot-side channel is INERT. The
#           pad band is then decorative and the run is a belt-only experiment.
#           Report it as such rather than reading a flat pad_impact row as
#           "compliance does not matter".
#        > 0.02                      -> the pad band is in play; the pad_impact
#           rows are meaningful.
#       Same logic for the belt: `dr/belt_k_joint_std` must be > 0. A std of
#       exactly 0 means the draw collapsed and the channel is a constant.
#   E1 belt   dr_impact/belt_k_joint_high_minus_low is the SIGNED contrast,
#             in episode-length steps, between the stiffest and softest third.
#        |contrast| < 25 steps on a mean episode of ~800 -> belt elasticity
#        does NOT drive outcomes at this band; say so plainly, and note that
#        the band may simply be too narrow to matter rather than "the belt is
#        irrelevant".  Widen (100 -> 30) in a follow-up before concluding.
#        contrast < -50 steps -> a STIFF belt is better. That is the direction
#        the rigid-plant assumption has always had, and it means the deploy
#        plant is stiffer than the band's soft half.
#   E2 pad    same construction on pad_k, but ONLY read it if E0 passed.
#        If E0 was 0, the honest sentence is "the pad channel was never
#        excited; no conclusion about pad compliance is available from this
#        run" — not "pad DR has no effect".
#   E3 skill  the run must not LOSE what S4a had. success_low >= 0.75 and
#        alpha >= 0.6 within 400 iters of the S4a endpoint. A drop here means
#        the elasticity band is a tax on the curriculum; narrow the soft end.
#   E4 sanity dr/belt_c_motor_min must read 0.0 (some envs draw zero damping
#        by construction, since the band starts at 0) and dr/pad_c_clamped_frac
#        must stay < 0.01. A high clamped fraction means the damping band is
#        above the explicit-integrator ceiling and is silently being cut.
# KILL: value_function > 3x its first-100-iter mean for >100 iters with ep len
#   falling; raw_action_max_abs > 10; or reward collapse > 300 iters.  A soft
#   belt at 100 N m/rad is 0.3x the servo's own knee stiffness (30.0) — if the
#   run kills on torque saturation rather than value loss, the soft end is the
#   suspect and the band should start at 300.
#
# NOT DONE HERE, DELIBERATELY
#   * The band endpoints are a FIRST CUT with no bench measurement behind them.
#     The belt has never been characterized for torsional stiffness; the only
#     hardware fact is a 1-tooth (12 deg joint-side) skip seen under catch-steps
#     (08-25).  The intended follow-up is a bench k measurement, then re-centre
#     the band on it.  Until then this run answers "is the belt STIFFNESS a
#     live variable for the policy", not "what is the belt's k".
#   * No restitution randomization: see above.
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
  '--run_name', 'S5_elasticDR',
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
  # ══ S5 NEW: belt transmission as a per-episode DR channel ══════════════
  # Frames (cfg §"Elasticity/contact evaluation knobs"): stiffness is KNEE-JOINT
  # N m/rad and the env converts with G^2 = (22/30)^2 at the actuator boundary;
  # damping is MOTOR-side N m s/rad and is passed straight through.  The ratio
  # form (belt_series_damping_ratio) is deliberately NOT used here — the user
  # asked for a damping BAND, and a nonzero explicit band always wins over the
  # ratio in the draw, so leaving the ratio at its NaN default keeps the intent
  # unambiguous.
  # zeta check at these bounds: wn = sqrt(k_joint*G2/I_rotor) with I_rotor =
  # 19.36e-3 kg m^2 -> wn = 17.6 rad/s at k=100 (zeta = d/(2*wn*I) = 0.44 at
  # d=0.3) and wn = 55.7 rad/s at k=1000 (zeta = 0.14).  So the band spans
  # "well damped, slow" to "lightly damped, fast" — a real spread, and the whole
  # range is on the compliant side of the servo's own 30.0 N m/rad knee loop.
  'env.belt_series_stiffness_min=100.0',
  'env.belt_series_stiffness_max=1000.0',
  'env.belt_series_damping_min=0.0',
  'env.belt_series_damping_max=0.3',
  # ══ S5 NEW: ball-foot pad as a per-episode DR channel ══════════════════
  # `pad_contact_enable=true` is REQUIRED — it is what builds the PadContact
  # object and allocates the per-env state; the band alone is not enough.
  # Damping is entered EXPLICITLY (pad_damping_min/max) rather than as a ratio,
  # matching the belt form.  At m_eff = 1.0 kg and dt = 1/500 s the explicit
  # damping ceiling is 0.5*1/0.002 = 250 N s/m, so a 0.3 N s/m band is 800x
  # below the cap and can never be clipped — drafter's note: that ALSO means
  # this band is nearly undamped (zeta = 0.3/(2*sqrt(200000*1)) = 3e-4 at the
  # stiff end).  It is what was asked for, and it will read as "no impact"
  # unless the pad is engaged at all; E0 decides that, not this number.
  'env.pad_contact_enable=true',
  'env.pad_stiffness_min=10000.0',
  'env.pad_stiffness_max=200000.0',
  'env.pad_damping_min=0.0',
  'env.pad_damping_max=0.3',
  # pad_thickness is a HARD travel constraint, not a preference: the settled sag
  # m*g/k must stay well under it or the pad bottoms out on the ball's rigid
  # contact and the stiffness stops meaning anything (pad_contact.py §3).  At
  # k=10000 and the ~54 N a two-foot support carries, the sag is 5.4 mm, so a
  # 2 mm pad would bottom out over the whole soft third of the band.  8 mm keeps
  # travel above the soft-end sag; dr/pad_c_clamped_frac and the [PLANT][WARN]
  # blowup line are the checks.
  'env.pad_thickness=0.008',
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
  -RedirectStandardOutput '<REPO>\RL\logs\train_S5_elasticDR.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_S5_elasticDR.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
