# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# S-line arm — "S8b_air03" — GPU 1 (RTX 4090 D by UUID), 2026-09-19.
#
# WHAT THIS IS
#   launch_train_S8_vendorJ.ps1 (vendor-rotor plant, no armature pin, belt DR,
#   per-env foot restitution, S4a measured-DR base, kneel->stand path
#   curriculum, resume from S4a model_7780) with EXACTLY ONE change:
#
#       env.feet_air_height_threshold  0.05 (cfg default 0.005*2*5)  ->  0.03
#
#   The knob is the |foot1_z - foot2_z| at which three gait rewards saturate
#   (rl_env_p6 compute_rewards: rew_feet_air, rew_gait_step, rew_foot_swing all
#   clamp foot_height_diff / feet_air_threshold at 1.0).  At 0.03 a 3 cm foot
#   split earns full credit where S8 needed 5 cm; the same lift is worth up to
#   1.67x more reward until saturation.  Nothing else — not the plant, not the
#   servo, not the DR spans, not the curriculum.  Diff this file against
#   launch_train_S8_vendorJ.ps1 to confirm: one added override, run name,
#   GPU line, log names.
#
# WHY (2026-09-19): the swing-clearance trio already asks for a 1.5 cm lift
#   (swing_clearance_target_m=0.015, zero 0.078); the gait terms' 5 cm
#   saturation is the P6-era walking value and sits well above what a
#   calm-idle policy is meant to produce.  This arm asks whether shrinking the
#   saturation point buys stepping credit at the lift height we actually want.
#
# WHAT TO EXPECT (pre-registered, 2026-09-19)
#   * [ARMATURE] / [PLANT] startup lines IDENTICAL to S8_vendorJ (knee 7.81e-3,
#     hip_pitch 8.40e-3, I_rotor=0.0042).  Any difference = wrong checkout.
#   * TB rew/feet_air, rew/gait_step, rew/foot_swing rise vs S8 at equal iters
#     (mechanical: same lifts, lower divisor).  This is NOT the signal.
#   * The signal is idle_toe_z / active_frac against S8: if lift height falls
#     toward ~3 cm while episode_length stays >= S8's, the smaller saturation
#     is doing what it should; if lifts stay >= 5 cm, the knob is inert here.
#   * Same resume, same reward scales elsewhere: expect the first ~100 iters
#     to look like S8's (the policy's opening lifts are already past 3 cm).
#
# KILLS / GATE: as S8 (value_function > 42000 for 50 consecutive iters,
#   episode_length < 400 at +300, PhysX error flood; gate at +1000:
#   episode_length >= 1400 with every DR span live).
#
# Everything below the header is S8_vendorJ's arg list verbatim plus the one
# override; read RL/launch_train_S8_vendorJ.ps1 (and S7's) for the rationale
# of the rest.
# ══════════════════════════════════════════════════════════════════════════════
$ErrorActionPreference = 'Stop'

$envRoot = '<CONDA_ENV>'
# $env:CUDA_VISIBLE_DEVICES = 'GPU-<uuid>'  # RTX 4090 (index 0); S8_vendorJ holds it
$env:CUDA_VISIBLE_DEVICES = 'GPU-<uuid>'   # RTX 4090 D (index 1)
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
  # '--max_iterations', '5',       # PHASE A smoke — REMOVE for the Phase B full run
  '--run_name', 'S8b_air03',
  '--resume',
  '--load_run', '2026-09-18_23-09-51_S8_vendorJ',
  '--checkpoint', 'model_7920.pt',
  # PPO params — desired_kl paired with the 1e-3 lr ceiling
  'agent.algorithm.entropy_coef=0.005',
  'agent.algorithm.desired_kl=0.02',
  'agent.algorithm.clip_param=0.12',
  # base K recipe
  'env.rew_scale_gait_ref=3.0',
  'env.sat_term_threshold=0.5',
  'env.sat_term_tau_s=0.5',
  'env.swing_clearance_zero_m=0.078',
  # ══ S8b: THE ONLY CHANGE vs S8_vendorJ — gait-reward saturation height ══
  # |foot1_z - foot2_z| at which rew_feet_air / rew_gait_step / rew_foot_swing
  # clamp to full credit.  cfg default 0.05 (= 0.005*2*5); S8 ran the default.
  'env.feet_air_height_threshold=0.03',
  'env.servo_kp_hip_pitch=20.0',
  'env.servo_kd_hip_pitch=1.4',
  'env.servo_kp_knee=30.0',
  'env.servo_kd_knee=1.5',
  # PLANT: NO armature pin.  hip_pitch/knee derive from the vendor rotor
  # inertia (belt_knee_actuator.ROTOR_INERTIA_OUTPUT = 4.2e-3): knee 7.81e-3,
  # hip_pitch 8.40e-3, series rotor 4.2e-3.  S7 pinned hip 0.0385 here.
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
  'env.dr_armature_mult_min=0.80',   # PLACEHOLDER (S7 band) — pending the
  'env.dr_armature_mult_max=1.20',   # Step 2 pinned-inertia verdict, see header
  'env.dr_effort_mult_min=0.9',
  'env.dr_com_offset_max=0.01',
  'env.dr_ang_vel_bias_max=0.05',
  'env.dr_gravity_tilt_max_deg=2.0',
  # ── S-line deltas (unchanged)
  'env.dr_com_shift_x_max=0.02',
  'env.dr_com_shift_y_max=0.01',
  'env.dr_proj_grav_bias_max=0.00',
  'env.dr_mass_mult_min=1.00',
  'env.dr_mass_mult_max=1.14',
  # ══ S7: belt transmission as a per-episode DR channel ══════════════════
  # Frames: stiffness is KNEE-JOINT N m/rad (the env converts with G^2 at the
  # actuator boundary); damping is MOTOR-side N m s/rad, passed straight through.
  # The explicit damping band is kept rather than the ratio form so the S5 belt
  # result stays comparable — see NOTE ON BELT DAMPING in the header.
  #
  # The TIMESTEP that makes this band usable lives in belt_knee_actuator.py
  # (series_dt=1/2000, series_substeps=4), NOT here — it is a plant-model fix,
  # not a per-run knob, and it must not drift between recipes.  If a run ever
  # needs a different density, change it there and say so in the commit.
  'env.belt_series_stiffness_min=100.0',
  'env.belt_series_stiffness_max=1000.0',
  'env.belt_series_damping_min=0.0',
  'env.belt_series_damping_max=0.3',
  # ══ DEAD: ball-foot pad as a per-episode DR channel — RETIRED 2026-09-17 ══
  # Do NOT re-enable these.  The pad never applied a force: PadContact's
  # reference plane sits 12 mm below the rigid floor, so the ball always finds
  # the floor first and k / zeta / thickness / m_eff are inert by construction
  # (memory: pad-reference-plane-inverted-20260916; the note in
  # run_foot_demo.ps1's `trainplantpin` arm is the by-hand proof).  S7's result
  # is a legacy-foot result, and a S7 checkpoint must be resumed WITHOUT these.
  #
  # 'env.pad_contact_enable=true',
  # 'env.pad_stiffness_min=5000.0',
  # 'env.pad_stiffness_max=30000.0',
  # 'env.pad_damping_ratio_min=0.1',
  # 'env.pad_damping_ratio_max=0.5',
  # 'env.pad_effective_mass=1.0',
  # 'env.pad_thickness=0.012',
  #
  # ══ PINNED FOOT MATERIAL — the live home for foot contact dynamics ═══════
  # Commented out, so this launch is bit-identical to S7's rigid feet (all-zero
  # = the legacy contact on the exact legacy branch).  This is the per-RUN
  # constant, the only form PhysX offers: one material prim is shared by every
  # env, so per-env DR of these gains does not exist.  To test contact
  # compliance, uncomment and run a NEW run_name — do not resume into S7.
  #
  #   stiffness  N/m    force-based PhysX compliant contact; 0 = rigid.
  #                     ENGAGEMENT IS NOT GUARANTEED and must be read, not
  #                     assumed: the floor material is rigid with
  #                     restitution_combine_mode="multiply", and PhysX's rule
  #                     for combining one compliant shape with one rigid shape
  #                     is not documented on this machine.  A/B against the k=0
  #                     run and compare the SETTLED height (ball centre bottoms
  #                     at 18.7 mm standing, against a 20 mm radius) plus knee
  #                     torque.  Bit-identical => the contact is rigid anyway,
  #                     and the fix is a compliant FLOOR material, not a value.
  #                     Anchor if it does engage: 27.3 N per foot gives 5.5 mm
  #                     sag at 5000, 0.9 mm at 30000.
  #   zeta       ratio  d = 2*zeta*sqrt(k*m_eff); needs k > 0 or the cfg raises.
  #                     0.3–0.7 is the physically sensible band.  This is the
  #                     knob that decides how much energy the contact gives back.
  #   restitution       already live as a per-env DR band below (0.2-0.8); the
  #                     material carrier 0.63 is the ROW MARKER that band writes
  #                     into — leave it at 0.63 or the band has no rows to find.
  #
  # 'env.foot_contact_stiffness=50000.0',
  # 'env.foot_contact_damping_ratio=0.3',
  # 'env.foot_contact_effective_mass=1.5',
  # ══ S7: foot-ball restitution as a per-episode DR channel (S6 wiring) ══
  'env.foot_restitution=0.63',
  'env.dr_foot_restitution_min=0.2',
  'env.dr_foot_restitution_max=0.8',
  'env.dr_foot_restitution_buckets=64',
  # ── kneel→stand curriculum (unchanged from S4a)
  'env.curriculum_enable=true',
  'env.curriculum_initial_alpha=0.0',
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
# $p = Start-Process -FilePath "$envRoot\python.exe" `
#   -WorkingDirectory '<REPO>\RL' `
#   -ArgumentList $argList `
#   -RedirectStandardOutput '<REPO>\RL\logs\train_S8b_air03.out.log' `
#   -RedirectStandardError  '<REPO>\RL\logs\train_S8b_air03.err.log' `
#   -WindowStyle Hidden -PassThru
# "Launched PID $($p.Id)"
& "$envRoot\python.exe" @argList