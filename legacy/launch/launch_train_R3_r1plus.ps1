# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# P13 R3 "R1-plus: cheap-dim widening" on GPU 1 (2026-08-26, replaces killed R2).
#
# CONTEXT: R2_robustmax never trained — 770 iters flat at reward ~300-440
# (vs R1 1259->2714), then genuine divergence from ~iter 2640 (sigma ran
# 0.61->0.78, reward halved, vloss spikes to 375k window-mean). Killed
# 08-26. TB post-mortem (see tb_r1r2_analysis.py in RL/logs/):
#   - R2's dr_impact splits are FLAT (all high-minus-low thirds < 10 steps
#     on ep_len ~266): no tracked plant-DR dim is the binding constraint.
#     The collapse is global — prime suspects are the UNTRACKED deltas:
#     exec-noise doubled (0.05/0.03), spawn widened (0.28/0.18 rad),
#     obs noise x1.5. R3 keeps ALL of those at R1 levels.
#   - R1's dr_impact (the informative run — its splits discriminate) ranks
#     the EXPENSIVE dims: action_delay (-221 steps high-vs-low third),
#     floor_friction high (-116), proj_grav_bias (-106), imu_delay (-77),
#     mass (-60), obs_delay (-49), joint_pos_bias (-36), stiffness low
#     end (-71). R3 does NOT widen these beyond R1.
#   - CHEAP dims (|impact| < ~11 steps in R1, ~0 corr in both runs):
#     effort_mult, com_offset, ang_vel_bias, jfriction, voltage, armature,
#     act_lag. R3 takes R2's amplitude on these + intermediate damping /
#     mass / encoder-bias: robustness coverage at ~zero training cost.
#
# RESUME: forks R1_robust model_2720 (TB-verified healthy: reward 2714 @
# iter 2731, sigma 0.54 stable) — "best value on the R2 card" = don't
# re-pay R1's 700 iters; strictly-wider DR from R1's current best.
# --reset_optimizer as before. R1 itself keeps running on GPU 0 as the
# un-widened control.
#
# Deltas vs R1 (everything else IDENTICAL — resume re-passes every
# override, the G-run lesson):
#   effort mult 0.8->0.7 floor, com 0.02->0.03 m, ang-vel bias
#   0.05->0.06, gravity tilt 2->3 deg, damping x0.6-1.5 (default was
#   0.7-1.3), mass x0.92-1.20 (default 0.94-1.18), encoder bias
#   0.07->0.08 rad.
#
# KILL RULE unchanged (sustained reward collapse only). WATCH: same as R1
# + reward should start near R1's ~2000-2700 band and dip only modestly at
# t0; if it craters to <1000 and stays, the "cheap" extrapolation beyond
# R1's measured band was wrong — kill and re-fork with defaults on
# damping/mass/encoder-bias.
$ErrorActionPreference = 'Stop'

$envRoot = '<CONDA_ENV>'
$env:CUDA_VISIBLE_DEVICES = '1'
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
  '--run_name', 'R3_r1plus',
  '--resume',
  '--load_run', '2026-08-25_14-54-09_R1_robust',
  '--checkpoint', 'model_2720.pt',
  '--reset_optimizer',
  'agent.algorithm.entropy_coef=0.005',
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
  # exec-noise injection (R1 level — untracked prime suspect in R2 collapse)
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
  # delays (R1 level — action_delay is the #1 measured ep-len cost)
  'env.dr_action_delay_min=1',
  'env.dr_action_delay_max=3',
  'env.dr_imu_delay_min=1',
  'env.dr_imu_delay_max=2',
  'env.dr_joint_delay_min=1',
  'env.dr_joint_delay_max=2',
  # plant DR: expensive dims at R1 level
  'env.dr_floor_friction_min=0.4',
  'env.dr_floor_friction_max=1.1',
  'env.dr_stiffness_mult_min=0.80',
  'env.dr_stiffness_mult_max=1.20',
  # plant DR: cheap dims widened to R2 level (the R3 delta)
  'env.dr_effort_mult_min=0.7',
  'env.dr_com_offset_max=0.03',
  'env.dr_ang_vel_bias_max=0.06',
  'env.dr_gravity_tilt_max_deg=3.0',
  'env.dr_damping_mult_min=0.6',
  'env.dr_damping_mult_max=1.5',
  # plant DR: intermediate widenings (default -> partway to R2)
  'env.dr_mass_mult_min=0.92',
  'env.dr_mass_mult_max=1.20',
  'env.dr_joint_pos_bias_max=0.08'
)
$p = Start-Process -FilePath "$envRoot\python.exe" `
  -WorkingDirectory '<REPO>\RL' `
  -ArgumentList $argList `
  -RedirectStandardOutput '<REPO>\RL\logs\train_R3_r1plus.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_R3_r1plus.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
