# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# P13 R1 "robust bundle" on GPU 0 (2026-08-25 evening).
#
# GOAL: real robot stands >0.5 s. Strategy per user: AGGRESSIVE STACKING —
# every sim2real/robustness measure that does no harm and may help, applied
# together (no one-by-one ablations). A/B vs R2_robustmax (GPU 1) = same
# bundle at HIGHER amplitude; if R2's DR is too wide to train (the known
# conservative-policy failure mode of over-randomization), R1 survives.
#
# RESUME: K_r400_zero069 model_2040 (user call: pre-crisp-gait, fatter σ,
# more plastic than K4b_3600; the checkpoint the whole K3/K4 line branched
# from). --reset_optimizer: the DR landscape changes massively.
#
# LR CAP REMOVED (user call, 2026-08-25): site-packages ppo.py restored to
# stock 1e-2 ceiling. History: the 2e-4 cap fixed K3/K3s2/K4 late-maturity
# divergences, but L2 collapsed UNDER it; at this resume point (σ~0.5+)
# ceiling rides were "in-family turbulence" (P10 verdict). KILL RULE:
# sustained reward collapse only — vloss spikes and 30-50-iter dips are
# in-family; if a K4-style unrecoverable raw_action runaway appears,
# kill and re-apply the cap (see memory rsl-rl-lr-ceiling-patch).
#
# THE BUNDLE (all CLI, resume re-passes EVERYTHING — the G-run lesson):
#   base K recipe: entropy 0.005, gait_ref 3.0, fence 0.5/0.5, zero 0.078
#     (stance-rule value for the late-K stance; watch idle_toe_z_p10,
#     re-derive if it settles elsewhere), servo 20/1.4 + 30/1.5.
#   bench-corrected plant: hip_pitch_armature 0.0385 (rotor ~19e-3 x2).
#   L1 calm trio (sim-validated, no observed harm): clearance target
#     0.015, no_flight -15, vz -8.
#   NEW exec-noise injection (target-space, arXiv 2504.06585 pattern):
#     bias ±0.03 rad episode-constant + 0.015 rad/step white
#     (hip 20kp: ~0.6 Nm bias / knee 30kp: ~0.9 Nm — real drivetrain
#     error class incl. belt tooth seating; the 08-25 skip = 0.21 rad).
#   spawn-posture DR (existing knobs, widened): joint noise 0.10->0.18,
#     base roll/pitch 0.08->0.12 rad — owns the entry transition from
#     imperfect postures (E1a hardware entries).
#   pushes ON (training had them OFF; eval always had them on):
#     p=0.001/step, 10-40 N, 10 steps.
#   delays: action 1-3 (hw measured 3-4 under load), imu 1-2, joint 1-2.
#   plant DR widened: floor mu 0.4-1.1, stiffness ×0.80-1.20,
#     effort ×0.8-1.0, com 0.02 m, ang-vel bias 0.05, gravity tilt 2 deg.
#
# WATCH: reward collapses at t0 then recovers (new penalties + DR);
# idle_toe_z_p10 (zero rule); pickup/active_frac should stay <=0.25;
# diagnostics/clip_fraction* will read higher (noise) — judge on eval.
# GATE before any flight: pickup-free hw_matched (run_eval_p12gate
# pattern) + push_recovery once built; survival >= K4b baseline 0.9971.
$ErrorActionPreference = 'Stop'

$envRoot = '<CONDA_ENV>'
$env:CUDA_VISIBLE_DEVICES = '0'
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
  '--run_name', 'R1_robust',
  '--resume',
  '--load_run', '2026-08-20_06-45-41_K_r400_zero069',
  '--checkpoint', 'model_2040.pt',
  '--reset_optimizer',
  'agent.algorithm.entropy_coef=0.005',
  # RESTART v2 (08-25 eve): desired_kl 0.008->0.02 — under this DR the
  # 0.008 band pinned lr at the 1e-5 floor 60% of updates with rail-to-
  # rail slams to 6.6e-3 (TB, first 107 iters); paired with the new
  # intermediate 1e-3 lr ceiling in the ppo.py patch (full removal gave
  # R2 a 95k vloss spike within 100 iters).
  'agent.algorithm.desired_kl=0.02',
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
  # NEW: execution-noise injection
  'env.exec_noise_target_bias_max=0.03',
  'env.exec_noise_target_std=0.015',
  # spawn-posture DR
  'env.noise_joint_pos=0.18',
  'env.init_base_orient_noise_rad=0.12',
  # pushes ON
  'env.push_enable=true',
  'env.push_probability=0.001',
  'env.push_force_min=10.0',
  'env.push_force_max=40.0',
  'env.push_duration_steps=10',
  # delays
  'env.dr_action_delay_min=1',
  'env.dr_action_delay_max=3',
  'env.dr_imu_delay_min=1',
  'env.dr_imu_delay_max=2',
  'env.dr_joint_delay_min=1',
  'env.dr_joint_delay_max=2',
  # plant DR widened
  'env.dr_floor_friction_min=0.4',
  'env.dr_floor_friction_max=1.1',
  'env.dr_stiffness_mult_min=0.80',
  'env.dr_stiffness_mult_max=1.20',
  'env.dr_effort_mult_min=0.8',
  'env.dr_com_offset_max=0.02',
  'env.dr_ang_vel_bias_max=0.05',
  'env.dr_gravity_tilt_max_deg=2.0'
)
$p = Start-Process -FilePath "$envRoot\python.exe" `
  -WorkingDirectory '<REPO>\RL' `
  -ArgumentList $argList `
  -RedirectStandardOutput '<REPO>\RL\logs\train_R1_robust.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_R1_robust.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
