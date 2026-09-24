# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# S-line arm A, RELAUNCH #9 — "S7_fullDR" — GPU 1 (RTX 4090 D by UUID), 2026-09-16.
#
# ══════════════════════════════════════════════════════════════════════════════
# 2026-09-17 — PAD CHANNEL RETIRED.  Read this before anything below it.
#
# Every pad claim further down this header is HISTORICAL: the pad band, its
# scale argument, the pad force-blowup kill, and the "ball_touched_frac says the
# balls DO load" line.  The pad never applied a force.  PadContact's reference
# plane sits 12 mm BELOW the rigid floor, so the ball's contact is always with
# the floor and the pad's k / zeta / thickness / m_eff are inert by construction
# (memory: pad-reference-plane-inverted-20260916; confirmed by hand in
# run_foot_demo.ps1's `trainplantpin` comment — k, m_eff and thickness all
# changed with no effect, restitution very much did not).
#
# The pad overrides are therefore REMOVED from $argList, not left inert.  The pad
# is not to be re-proposed, re-band-scaled, or "fixed" by flipping the plane
# sign: a sign flip alone gives a frictionless hovering foot.
#
# What this run actually was, and still is: belt-series elasticity DR + per-env
# foot restitution, on RIGID feet.  Both of those channels are live and
# exercised, so S7's results stand as a legacy-foot result.
#
# The foot CONTACT dynamics the pad was meant to provide have a live path, the
# PhysX compliant material, and it is a per-run constant (the material prim is
# stage-global, so a per-env DR draw is not expressible).  The knobs are in the
# PINNED FOOT MATERIAL block in $argList: foot_contact_stiffness (N/m),
# foot_contact_damping_ratio (zeta against foot_contact_effective_mass),
# foot_restitution, foot_static/dynamic_friction.  All-zero = the rigid legacy
# contact, bit-exact.
# ══════════════════════════════════════════════════════════════════════════════
#
# WHAT CHANGED SINCE RELAUNCH #8 (2026-09-14) — read this block or the run will
# look like a repeat of a run that never worked.
#
# The belt DR was structurally broken and is now fixed at the SOURCE, not here:
#
#   belt_knee_actuator.py  BeltKneeDCMotorCfg
#       series_dt      1.0/500  ->  1.0/2000
#       series_substeps     2   ->  4
#
# The belt series integrator advanced a HARDCODED 1/500 (2 substeps of 1 ms)
# per physics step while the env ran decimation=20 at sim dt=1/2000.  Nothing
# in rl_env_p6.py or rl_env_cfg_p6.py ever wrote series_dt — the 09-16 rate
# change to 2000 Hz moved `decimation`, `sim.dt` and `rotor_coupling_dt` but
# left the belt integrator behind.  Consequences, both observed:
#
#   * the belt advanced 1 ms of virtual time per 0.5 ms physics step, relaxing
#     against an anchor that had moved half as far as the integrator assumed —
#     a 2x virtual stiffness inflation plus a phase error in the damper;
#   * the substep density was 1/500/2 = 1 ms, so the coupled belt/joint mode
#     grew instead of settling as stiffness rose.  Measured spectral radius of
#     the one-step map (data/_s7_belt_config_score.py), swing-shin inertia,
#     zeta 0.3:  k=150 -> 0.964, k=550 -> 2.225, k=950 -> 3.583 (all > 1 =
#     energy injected every step).  At the new setting every cell is 0.990-0.993.
#
# THE ONE-LINE DIAGNOSIS: physics 2000 Hz, belt 1/500.  The belt loop was the
# only part of the plant still running at the pre-09-16 rate.
#
# The 09-16 outcome with the fix in place: the robot WALKS with belt stiffness
# pinned at 950 joint-side and damping 0.3 — the cell that was "totally loose
# control" before.  Stiffness is now usable across the whole band.
# ══════════════════════════════════════════════════════════════════════════════
#
# WHAT THIS IS
#   The corrected re-do of S5 (killed at 7787 on a mis-scaled pad stiffness band)
#   PLUS the S6 restitution channel, PLUS the belt-timestep fix above, in one run.
#   Plant DR as it now stands: belt series elasticity + per-env foot restitution,
#   drawn per episode on top of the S4a measured-DR recipe (which is re-passed
#   verbatim — a resume reverts anything not re-passed, so the recipe is complete,
#   not incremental).  The pad term is gone; see the retirement banner above.
#
#   NO NEW PHYSICS CODE.  Two modules carry it, both already in the tree:
#     * the S5 belt wiring in rl_env_p6.py / belt_knee_actuator.py
#                         (09-16: series_dt/substeps corrected, see above)
#     * the S6 restitution row-discovery wiring in rl_env_p6.py
#   The change made for relaunch #8 is in the cfg/env: pad damping is enterable as
#   a per-episode zeta BAND (pad_damping_ratio_min/max) rather than an explicit
#   N s/m band.  That change is now MOOT (the pad is retired) and is left
#   described here only so the #8 header does not read as a live recipe.
#
# CORRECTED BANDS — the two that remain live
#   belt_k_joint        100 – 1000    knee-joint N m/rad    (UNCHANGED, but now
#                                     actually exercised — see the fix above.
#                                     100-1000 is the range that behaved sanely
#                                     in the 09-16 manual ladder: 150 ok,
#                                     550 ok, 950 ok after the fix.  The cells
#                                     that failed (550/950) failed on the timing
#                                     bug, not on the band being too wide.)
#   belt_c_motor        0 – 0.3       motor-side N m s/rad  (explicit band; see
#                                     the NOTE ON BELT DAMPING at the end)
#   foot_restitution    0.2 – 0.8     coefficient              (S6 wiring, 64 buckets)
#   gpu_max_rigid_patch_count  600_000  (was 320_000; S5 asked up to 501 974)
#
#   RETIRED, kept only so the history is readable: pad_k 5000–30000 N/m and
#   pad_damping_ratio 0.1–0.5.  The scale argument that justified them (27.3 N
#   per foot, 0.9–5.5 mm settled sag, 12 mm travel = 2.2x the soft-end sag) was
#   correct as arithmetic and never reached the plant, because the pad's
#   reference plane is below the floor.  Do not re-derive a band from it.
#
# RESTITUTION (S6 wiring, reused here)
#   The foot spawner stays ACTIVE with restitution_combine_mode="max" and
#   foot_restitution=0.63: that nonzero carrier is what keeps the binding (at 0
#   the spawner takes the legacy branch, the balls fall back to combine mode
#   "average" and silently halve the bounce).  0.63 is the ROW MARKER and the
#   combine-mode carrier, NOT the training value.  The env discovers the two ball
#   rows at build time (rows whose restitution reads 0.63), asserts exactly two,
#   then writes the drawn e into those two rows per env per reset.  The
#   dr/foot_restitution_* TB channels are READ BACK from the buffer.
#   bounce_threshold_velocity = 0.05 (cfg default 0.5) so the balls bounce on
#   low-speed contact instead of damping it out.
#
# STARTUP CONTRACT — check these before believing anything (all printed by the env)
#   [PLANT] belt series: live=True k_joint=[100, 1000] d_motor=[0, 0.3] I_rotor=0.01936
#   [S7] restitution row discovery: exactly two rows at 0.63
#   [PLANT] encoder obs write slots (sim order): {'knee1': 4, 'knee2': 5}
#   A belt line reading series_dt=1/500 anywhere is the old bug coming back.
#   There is NO pad line any more — a `[PLANT] pad model` line, or any
#   `pad_contact*` override in the command, means a stale script was launched.
#   The foot material line reads RIGID legacy material unless the pinned block
#   below is uncommented; if you uncomment it, read the k / zeta / e it resolved.
#
# PHASE A: add '--max_iterations', '5' and read back every DR span, the patch
#   count, the bounce threshold, value_function and the [PLANT] lines, then STOP.
# PHASE B (after approval): drop --max_iterations.
#
# PRE-REGISTERED KILLS (Phase B): value_function > 42000 (10x the S4a endpoint
#   ~4200) for 50 consecutive iters, OR episode_length < 400 at iter +300, OR a
#   PhysX error flood.
# SUCCESS GATE at +1000: episode_length >= 1400 AND every DR span live
#   (belt_k_joint, belt_c_motor, foot_restitution all with std > 0).
#   Then export the policy and stop.
#
# OPEN RISK, STATED UP FRONT
#   * The belt band's stability rests entirely on the series_dt/substeps fix in
#     belt_knee_actuator.py.  If a run ever needs a different integrator density,
#     change it there and say so in the commit — never from a launch script.
#   * The 09-16 "it walks at k=950" result is a 4-env PLAY rollout, not a
#     training signal.  It proves the plant is stable; it does NOT prove the
#     band is learnable.  The first 100 iters of this run are the real test.
#   * The belt fix changes the plant for every elastic env, so S5/S6 checkpoints
#     are not comparable to checkpoints from this run.
#
# NOTE ON BELT DAMPING (pre-registered, do NOT tune mid-run)
#   The explicit band 0–0.3 is kept so the S5 belt result stays comparable.  It
#   does mean zeta FALLS as stiffness rises (0.147 at k=100 -> 0.048 at k=1000),
#   so the top of the stiffness band is the least damped cell.  That is now
#   stable, but if the belt band turns out to underpin half the failures, the
#   alternative is belt_series_damping_ratio (constant zeta across the band) —
#   one CLI swap, deliberately NOT taken here because it would change two things
#   at once on the first run after a structural fix.
# ══════════════════════════════════════════════════════════════════════════════
$ErrorActionPreference = 'Stop'

$envRoot = '<CONDA_ENV>'
$env:CUDA_VISIBLE_DEVICES = 'GPU-<uuid>'   # RTX 4090 D (index 1)
# $env:CUDA_VISIBLE_DEVICES = 'GPU-<uuid>'  # 4060 Ti
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
  # '--headless',
  # '--max_iterations', '5',       # PHASE A smoke — REMOVE for the Phase B full run
  '--run_name', 'S7_fullDR',
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
  # 'env.foot_contact_stiffness=0.0',
  # 'env.foot_contact_damping_ratio=0.3',
  # 'env.foot_contact_effective_mass=1.0',
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
$p = Start-Process -FilePath "$envRoot\python.exe" `
  -WorkingDirectory '<REPO>\RL' `
  -ArgumentList $argList `
  -RedirectStandardOutput '<REPO>\RL\logs\train_S7_fullDR.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_S7_fullDR.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
