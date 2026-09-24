# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# S-line arm A, RELAUNCH #8 — "S7_fullDR" — GPU 1 (RTX 4090 by UUID), 2026-09-14.
#
# WHAT THIS IS
#   The corrected re-do of S5 (killed at 7787 on a mis-scaled pad stiffness band)
#   PLUS the S6 restitution channel, in one run.  Full-DR plant: belt series
#   elasticity, ball-pad compliance, and per-env foot restitution, all drawn per
#   episode on top of the S4a measured-DR recipe (which is re-passed verbatim —
#   a resume reverts anything not re-passed, so the recipe is complete, not
#   incremental).
#
#   NO NEW PHYSICS CODE.  Two modules carry it, both already in the tree:
#     * pad_contact.py   — per-env pad spring/damper as a step-applied FORCE
#                          (the material is stage-global; see the module docstring)
#     * the S5 belt wiring in rl_env_p6.py / belt_knee_actuator.py
#     * the S6 restitution row-discovery wiring in rl_env_p6.py
#   The ONE change made for this run is in the cfg/env: pad damping is now
#   enterable as a per-episode zeta BAND (pad_damping_ratio_min/max) rather than
#   an explicit N s/m band — see "CORRECTED BANDS" below for why that is the
#   fix and not a preference.
#
# CORRECTED BANDS — the S5 numbers are NOT reused
#   belt_k_joint        100 – 1000    knee-joint N m/rad    (unchanged; S5: engaged cleanly, zeta 0.44 → 0.14)
#   belt_c_motor        0 – 0.3       motor-side N m s/rad  (unchanged; explicit band wins over the ratio)
#   pad_k               5000 – 30000  floor-normal N/m      (S5: 10000–200000 = 3–20x too stiff for the load)
#   pad_damping_ratio   0.1 – 0.5     zeta of the pad's own mode, m_eff = 1 kg
#                       (S5: explicit 0–0.3 N s/m gave zeta ~1e-3 at every point,
#                        and tau = m/c was clamped at its 4*dt = 8 ms floor, so the
#                        state advanced at 25% of the gap per step regardless of the
#                        stiffness drawn — a knob that could not affect the plant)
#   foot_restitution    0.4 – 0.8     coefficient              (S6 wiring, 64 buckets)
#   gpu_max_rigid_patch_count  600_000  (was 320_000; S5 asked up to 501 974 and flooded the log)
#
# WHY THE CORRECTED PAD BAND IS THE RIGHT SCALE (docs/s5-elastic-dr-20260914.md §4-5)
#   Half-body-weight per foot is 27.3 N.  At k_max = 30000 the settled sag is
#   27.3/30000 = 0.9 mm; at k_min = 5000 it is 5.5 mm.  pad_thickness = 0.012 m is
#   2.2x the soft-end sag, so the pad never bottoms out on the ball's rigid
#   backstop over the whole band — which is the condition under which the stiffness
#   numbers mean what they say.  (S5's 8 mm pad bottomed out over the soft third
#   of its band, and its stiff end reached 1600 N = 29x body weight at 8 mm.)
#   zeta band check at m_eff = 1 kg: c = 2*zeta*sqrt(k*m) is 4.5–15.8 N s/m at
#   k=5000 and 10.9–38.7 N s/m at k=30000, against a c_cap of 250 N s/m — a 6x
#   margin at the worst corner, so dr/pad_c_clamped_frac must read 0.000 and a
#   nonzero value is a bug, not a band that is too wide.
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
# PHASE A (this script as written, --max_iterations 5): read back EVERY channel
#   span, the patch count, the bounce threshold and value_function, confirm the
#   [S7] restitution row-discovery line and the [PLANT] lines, then STOP.
#   PHASE B (after approval): uncomment the full run and drop --max_iterations.
#
# PRE-REGISTERED KILLS (Phase B): value_function > 42000 (10x the S4a endpoint
#   ~4200) for 50 consecutive iters, OR episode_length < 400 at iter +300, OR a
#   PhysX error flood.
# SUCCESS GATE at +1000: episode_length >= 1400 AND every DR span live
#   (belt_k_joint, belt_c_motor, pad_k, pad_c, foot_restitution all with std > 0).
#   Then export the policy and stop.
#
# OPEN RISK, STATED UP FRONT (do not let it be discovered later)
#   The pad band has never been run at a live k at all — S5 diverged in 7 iters
#   with the mis-scaled version and never exercised the corrected one.  The first
#   thing that can go wrong here is a pad force blowup, not a value divergence.
#   Watch foot/ball_touched_frac FIRST (the S5 finding: ~0.09–0.13, so the balls
#   DO load and the pad channel is NOT decorative), then the [PLANT][WARN] blowup
#   lines.  Any force blowup at 30000 N/m => STOP, report, do not tune.
$ErrorActionPreference = 'Stop'

$envRoot = '<CONDA_ENV>'
$env:CUDA_VISIBLE_DEVICES = 'GPU-<uuid>'   # RTX 4090 D (index 1)
# $env:CUDA_VISIBLE_DEVICES = 'GPU-<uuid>'
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
  # actuator boundary); damping is MOTOR-side N m s/rad, passed straight
  # through.  The explicit damping band is used rather than the ratio form so
  # the S5 belt result stays comparable — the belt band was well-posed (zeta
  # 0.44 at k=100 to 0.14 at k=1000, both on the compliant side of the servo's
  # own 30.0 N m/rad knee loop) and is deliberately NOT part of the correction.
  'env.belt_series_stiffness_min=50.00',
  'env.belt_series_stiffness_max=50.001',
  'env.belt_series_damping_min=0.0',
  'env.belt_series_damping_max=0.0',
  # ══ S7: ball-foot pad as a per-episode DR channel (CORRECTED BAND) ═════
  # `pad_contact_enable=true` is REQUIRED — it builds the PadContact object and
  # allocates the per-env state; the band alone is not enough.
  'env.pad_contact_enable=true',
  'env.pad_stiffness_min=5000.0',
  'env.pad_stiffness_max=30000.0',
  # # Damping entered as a per-episode zeta BAND (the S7 code change).  Explicit
  # # pad_damping_min/max are left at 0 so the ratio branch is the one that runs.
  'env.pad_damping_ratio_min=0.1',
  'env.pad_damping_ratio_max=0.5',
  'env.pad_effective_mass=1.0',
  # pad_thickness is a HARD travel constraint, not a preference: the settled sag
  # m*g/k must stay well under it or the pad bottoms out on the ball's rigid
  # backstop and the stiffness stops meaning anything.  12 mm is 2.2x the
  # soft-end sag (5.5 mm at k=5000, 27.3 N).
  'env.pad_thickness=0.012',
  # ══ S7: foot-ball restitution as a per-episode DR channel (S6 wiring) ══
  'env.foot_restitution=0.63',
  'env.dr_foot_restitution_min=0.4',
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
