# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# Belt A/B ladder — 2026-09-15.  Four SHORT headless runs (4 iters each, ~2 min
# apiece at 128 envs), identical to launch_train_S7_fullDR_test.ps1 except the
# belt block, which is pinned at k = 150 N m/rad, d = 0 and then has ONE of the
# two belt-only code paths switched off per arm:
#
#   arm A  enc=on  mask=on   the failing configuration (ep len flat ~60)
#   arm B  enc=OFF mask=on   knee obs read the physical joint (play-hotkey obs)
#   arm C  enc=on  mask=OFF  knee+hip armature stay in PhysX (rotor double-counted)
#   arm D  enc=OFF mask=OFF  both
#   arm R  rigid             belt band [0,0] — the control (ep len 60->100->150)
#
# Pre-registered reading (Train/mean_episode_length at iter 3, rigid ~150):
#   B recovers, C does not  -> the motor-side knee OBSERVATION is the killer
#   C recovers, B does not  -> the ARMATURE MASK (hip 0.0385 -> 0) is the killer
#   only D recovers          -> both contribute
#   none recovers            -> neither belt-only path; the integrator/spring
#                              itself is the difference, look at belt/* readbacks
# ROUND 1 RESULT (09-16 morning): B and D recover (ep len 159 / 172 at iter 4,
#   rigid 213), A and C stay flat at ~62 -> the encoder OBS is the killer, the
#   armature mask is not.  Readbacks: enc-vs-joint dev 0.010 rad / 1.15 rad/s.
# ROUND 2 (arms E, F): which channel.  E = position from encoder only,
#   F = velocity from encoder only.
#   E dies, F lives  -> the 0.01 rad POSITION deviation kills -> not credible as
#                       physics (inside the trained bias band) -> bug hunt in
#                       the position path
#   F dies, E lives  -> the correlated ~1.2 rad/s VELOCITY deviation kills ->
#                       the policy's knee-velocity feedback cannot tolerate the
#                       motor-side velocity; that is a real sim2real finding
#                       (hardware reads the motor too) and a DR/obs-noise
#                       decision, not a bug
#   both die         -> either channel alone suffices; treat as F
#   neither dies     -> the two channels kill only together; rerun A to confirm
#     powershell -File RL\run_belt_ab_ladder.ps1 -Arms E,F
# Readbacks (new TB channels, every arm): belt/spring_effort_abs_mean,
# belt/motor_effort_abs_mean, belt/enc_pos_dev_abs_mean, belt/enc_vel_dev_abs_mean,
# belt/rotor_vel_jointeq_p99, belt/hip_armature_written_mean,
# belt/knee_armature_written_mean.  Summarize with:
#   python data\_belt_ab_read.py
#
# Usage:  powershell -File RL\run_belt_ab_ladder.ps1            (all five, sequential)
#         powershell -File RL\run_belt_ab_ladder.ps1 -Arms B,C  (subset)
param(
    [string[]]$Arms = [string[]]@('A', 'B', 'C', 'D', 'R'),
    [int]$Iters = 4,
    [double]$K = 150.0
)
$ErrorActionPreference = 'Stop'

$envRoot = '<CONDA_ENV>'
$env:CUDA_VISIBLE_DEVICES = 'GPU-<uuid>'
$env:CARB_APP_PATH = '<ISAACLAB>\_isaac_sim\kit'
$env:EXP_PATH      = '<ISAACLAB>\_isaac_sim\apps'
$env:ISAAC_PATH    = '<ISAACLAB>\_isaac_sim'
$env:PYTHONPATH    = ';<ISAACLAB>\_isaac_sim\site'
$env:RESOURCE_NAME = 'IsaacSim'
$env:CONDA_PREFIX  = $envRoot
$env:PATH = "$envRoot;$envRoot\Library\mingw-w64\bin;$envRoot\Library\usr\bin;$envRoot\Library\bin;$envRoot\Scripts;$envRoot\bin;" + $env:PATH

$BASE = @(
  '-u', 'scripts/rsl_rl/train.py',
  '--task', 'Template-standup_curriculum-v0',
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
  # ══ S7: ball-foot pad as a per-episode DR channel (CORRECTED BAND) ═════
  # `pad_contact_enable=true` is REQUIRED — it builds the PadContact object and
  # allocates the per-env state; the band alone is not enough.
  # 'env.pad_contact_enable=true',
  # 'env.pad_stiffness_min=5000.0',
  # 'env.pad_stiffness_max=30000.0',
  # # Damping entered as a per-episode zeta BAND (the S7 code change).  Explicit
  # # pad_damping_min/max are left at 0 so the ratio branch is the one that runs.
  # 'env.pad_damping_ratio_min=0.1',
  # 'env.pad_damping_ratio_max=0.5',
  # 'env.pad_effective_mass=1.0',
  # pad_thickness is a HARD travel constraint, not a preference: the settled sag
  # m*g/k must stay well under it or the pad bottoms out on the ball's rigid
  # backstop and the stiffness stops meaning anything.  12 mm is 2.2x the
  # soft-end sag (5.5 mm at k=5000, 27.3 N).
  # 'env.pad_thickness=0.012',
  # # ══ S7: foot-ball restitution as a per-episode DR channel (S6 wiring) ══
  # 'env.foot_restitution=0.63',
  # 'env.dr_foot_restitution_min=0.4',
  # 'env.dr_foot_restitution_max=0.8',
  # 'env.dr_foot_restitution_buckets=64',
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

New-Item -ItemType Directory -Force -Path '<REPO>\RL\logs' | Out-Null

$kStr = '{0:F4}' -f $K          # "150.0000"
$kMaxStr = '{0:F4}' -f ($K + 0.001)  # "150.0010"

$ARM_MAP = @{
    'A' = @("env.belt_series_stiffness_min=$kStr", "env.belt_series_stiffness_max=$kMaxStr", 'env.belt_encoder_obs=true',  'env.belt_mask_armature=true')
    'B' = @("env.belt_series_stiffness_min=$kStr", "env.belt_series_stiffness_max=$kMaxStr", 'env.belt_encoder_obs=false', 'env.belt_mask_armature=true')
    'C' = @("env.belt_series_stiffness_min=$kStr", "env.belt_series_stiffness_max=$kMaxStr", 'env.belt_encoder_obs=true',  'env.belt_mask_armature=false')
    'D' = @("env.belt_series_stiffness_min=$kStr", "env.belt_series_stiffness_max=$kMaxStr", 'env.belt_encoder_obs=false', 'env.belt_mask_armature=false')
    'R' = @('env.belt_series_stiffness_min=0.0', 'env.belt_series_stiffness_max=0.0')
  # Round 2 (2026-09-16): A/B/C/D said the encoder OBS is the killer and the
  # armature mask is not.  Split the obs by channel, mask left ON as in A:
'E' = @("env.belt_series_stiffness_min=$kStr", "env.belt_series_stiffness_max=$kMaxStr", 'env.belt_encoder_obs=true', 'env.belt_encoder_obs_pos=true',  'env.belt_encoder_obs_vel=false')
'F' = @("env.belt_series_stiffness_min=$kStr", "env.belt_series_stiffness_max=$kMaxStr", 'env.belt_encoder_obs=true', 'env.belt_encoder_obs_pos=false', 'env.belt_encoder_obs_vel=true')
  # Round 3 (09-16): E AND F both died on a 0.01 rad / 1.2 rad/s deviation.
  # G takes the encoder code path but writes NEITHER channel: if G dies the
  # kill is not the channel values at all.  Every arm now also dumps a
  # per-step trace (belt_debug.npz in the run dir); read with
  #   python data\_belt_dump_read.py E B
'G' = @("env.belt_series_stiffness_min=$kStr", "env.belt_series_stiffness_max=$kMaxStr", 'env.belt_encoder_obs=true', 'env.belt_encoder_obs_pos=false', 'env.belt_encoder_obs_vel=false')
  # ROUND 3 RESULT (09-16): G == B bit-for-bit (encoder path inert unless it
  # writes); E and F both die.  The per-step dump showed the write landing on
  # hip_pitch1 (sim index 2), not knee1: the slot came from the env's joint
  # list, the target array is sim-ordered (hr1,hr2,hp1,hp2,kn1,kn2).  FIXED
  # via encoder_write_slots().  Round 4 = re-run A (and E) after the fix:
  #   prediction: A ~= B/G (ep len ~160 at iter 4), dump shows kn1/kn2 the
  #   ONLY modified slots, ~0.01 rad; hip slots exactly 0.
}
foreach ($arm in $Arms) {
    if ($ARM_MAP.Keys -notcontains $arm) { throw "unknown arm $arm" }
    $name = "S7_beltAB_$arm"
    $argList = $BASE + @('--headless', '--max_iterations', "$Iters", '--run_name', $name,
                         'env.belt_series_damping_min=0.0', 'env.belt_series_damping_max=0.0',
                         'env.belt_debug_dump_steps=300', 'env.belt_debug_dump_envs=8') + $ARM_MAP[$arm]
    Write-Output "[ladder] arm $arm -> $name  ($($ARM_MAP[$arm] -join ' '))"
    $p = Start-Process -FilePath "$envRoot\python.exe" `
        -WorkingDirectory '<REPO>\RL' `
        -ArgumentList $argList `
        -RedirectStandardOutput "<REPO>\RL\logs\train_$name.out.log" `
        -RedirectStandardError  "<REPO>\RL\logs\train_$name.err.log" `
        -WindowStyle Hidden -PassThru -Wait
    Write-Output "[ladder] arm $arm exit=$($p.ExitCode)"
}
Write-Output "[ladder] done — run:  python data\_belt_ab_read.py"
