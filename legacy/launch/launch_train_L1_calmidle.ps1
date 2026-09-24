# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# P12 L1 on GPU 0: CALM-IDLE arm, resumed from K4b model_3600 (the P11
# primary deploy pick — hw gate 1.000/1024, the policy that flew 08-24).
# Context (2026-08-24 flight verdict, docs/p11-hardware-runbook.md): falls
# keep G2i's pitch-forward clock but the character is an overdriven march
# (2x sim action saturation, 5.6-17 Hz action content vs sim's clean
# 2.5 Hz). Contract/decode/servo all eliminated; the lever left is the gait.
#
# CORRECTION vs runbook item 1 as written ("zero 0.078 -> ~0.03-0.04"):
# K4b TB reads settled idle_toe_z_p10 = 0.0735 (flat over last 20 iters;
# = resting toe z in the reward frame; pooled p50 0.0769 sits just above).
# A 0.03-0.04 zero would be 3-4 cm BELOW rest -> the swing term saturates
# for a foot flat on the ground = the exact shuffle exploit the zero-rule
# exists to prevent (3 recalibration recurrences, see memory). Zero STAYS
# 0.078 (= rest + 4.5 mm; anti-shuffle rule genuinely intact). The calm
# levers are instead:
#   swing_clearance_target_m 0.04 -> 0.015  clearance credit saturates at
#                                    rest+~18 mm (was rest+45 mm): removes
#                                    the high-lift race; near-ground liftoff
#                                    gradient gets STEEPER (anti-shuffle
#                                    sharpened, calm on top)
#   rew_scale_no_flight  -5 -> -15   both-feet-airborne tick fine 3x. An
#                                    alternating single-support march never
#                                    pays this by construction.
#   rew_scale_vz_penalty -2 -> -8    4x tax on base bounce (K4b TB
#                                    base_vz_sq_mean 0.0715; settled eval
#                                    bounce +/-1 cm @ 5 Hz).
# Everything else = K4b contract bit-for-bit: gait_ref 3.0 @ 2.5 Hz clock,
# sat fence 0.5/0.5, smoothness -0.24 / snap -0.06 (do NOT raise to chase
# TB jerk — that metric is sigma-noise), servo 20/1.4 + 30/1.5.
#
# FOLDED (runbook item 5): hip armature recenter 33.2e-3 -> 38.5e-3 via the
# NEW cfg knob env.hip_pitch_armature (bench 1.2: rotors ~19e-3 both ->
# 19.3 own + 19.3 knee-on-hip-body). Knee 36.0e-3 stands. DR 0.7-1.3 now
# brackets the real hip mid-band instead of at its edge. EVAL TRAP: the
# run_eval_p9sel K_OVR list does NOT carry this knob — L1 eval/gate jobs
# must add hip_pitch_armature=0.0385 or they rebuild the K plant.
#
# Why K4b_3600 (not K5/K5b/K4_3420/fresh): the acceptance gate is defined
# relative to K4b; sigma ~0.37 still fat = plasticity for a reward reshape
# (K5/K5b annealed sigma to ~0.26 under kl 0.002 with lr floor-pinned —
# wrong base for re-learning, and 08-24 eval showed polish did not change
# the envelope); K4_3420 is the same line 180 iters greener; a fresh train
# throws away the standup/pickup curriculum for no attribution gain.
#
# Watch (TB): idle_liftoff_rate >= 0.03 sustained (K4b idles at 0.050;
# sustained below = shuffle capture -> kill, walk target back toward 0.02);
# idle_toe_z_p10 holding ~0.073 (pinned there with liftoff_rate collapsing
# = same alarm); base_vz_sq_mean falling from 0.0715; flight_fraction_all
# falling from 0.19. Reward (~16.0k at branch) DROPS at t0 from the new
# penalties, then recovers — kill on collapse only, not on vloss spikes
# (patched-scheduler transients recover; ceiling patch verified present
# 08-24 — after any env reinstall re-check `grep "LOCAL PATCH" ppo.py`).
# Expect a few hundred iters (K4->K4b re-equilibrated in ~180).
#
# ACCEPTANCE GATE — PICKUP-FREE (the verdict's "flight 0.22 / z 0.41-0.72 /
# pg_x +/-0.44" envelope was pickup-ride contamination: eval leaves the
# pickup curriculum ON. Settled K4b truth from the 08-24 traj dump: z std
# 0.0055 @ 5 Hz, pg_x std 0.034-0.037, |a|>0.95 on 8-13% of ticks):
#   eval.py --scenarios hw_matched, 1024 envs, dump_traj, --cfg_overrides
#   = K_OVR + hip_pitch_armature=0.0385 + pickup_initial_probability=0.0
#   + pickup_probability=0.0
# PASS = 15 s survival 1.000 (>= K4b) AND settled z std < 0.003 AND
# sat frac < 0.04 AND pg_x std <= 0.04 AND flight_fraction < 0.05 (now
# meaningful) AND TB idle_liftoff_rate >= 0.03 (march still real).
#
# GPU 0 free as of 08-24 late (K5/K5b already stopped — verified no live
# training processes).
# NOTE: resume re-reads cfg defaults — ALL overrides re-passed here
# (entropy 0.005 explicit: the agent-cfg default is the 0.003 polish value).
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
  '--run_name', 'L1_calmidle',
  '--seed', '44',
  '--resume',
  '--load_run', '2026-08-23_08-56-43_K4b_resume3420',
  '--checkpoint', 'model_3600.pt',
  'agent.algorithm.entropy_coef=0.005',
  'agent.algorithm.desired_kl=0.008',
  'env.rew_scale_gait_ref=3.0',
  'env.sat_term_threshold=0.5',
  'env.sat_term_tau_s=0.5',
  'env.swing_clearance_zero_m=0.078',
  'env.rew_scale_swing_clearance=4.0',
  'env.swing_clearance_target_m=0.015',
  'env.rew_scale_no_flight=-15.0',
  'env.rew_scale_vz_penalty=-8.0',
  'env.rew_scale_action_smoothness=-0.24',
  'env.rew_scale_action_snap=-0.06',
  'env.servo_kp_hip_pitch=20.0',
  'env.servo_kd_hip_pitch=1.4',
  'env.servo_kp_knee=30.0',
  'env.servo_kd_knee=1.5',
  'env.hip_pitch_armature=0.0385'
)
$p = Start-Process -FilePath "$envRoot\python.exe" `
  -WorkingDirectory '<REPO>\RL' `
  -ArgumentList $argList `
  -RedirectStandardOutput '<REPO>\RL\logs\train_L1_calmidle.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_L1_calmidle.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
