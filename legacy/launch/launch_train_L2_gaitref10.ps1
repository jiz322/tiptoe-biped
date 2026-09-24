# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# P12 L2 on GPU 1: MECHANISM A/B against L1_calmidle — single delta:
# rew_scale_gait_ref 3.0 -> 1.0. Same branch point (K4b model_3600), same
# seed 44 ON PURPOSE (this arm tests a knob, not randomness — seed twins
# retired: K4b2/K5b already demonstrated recipe reproducibility post-patch,
# and two resumes of the SAME parent differ far less by seed than fresh
# trains do). See launch_train_L1_calmidle.ps1 for the shared calm-idle
# context, the zero-stays-0.078 correction, and the pickup-free gate.
#
# Rationale: L1's penalties (no_flight -15, vz -8, target 0.015) tax the
# SYMPTOMS of the hot march; the gait-ref prior (+/-0.12 rad hip swing @
# 2.5 Hz, scale 3.0) is its amplitude ENGINE. The cfg's own plan says the
# prior is "SCAFFOLDING, NOT THE BUILDING: once idle_liftoff_rate >= 0.03
# sustained, anneal the scale (3.0 -> 1.0 -> 0.3 -> 0)" — K4b idles at
# 0.050, the trigger has been met since before the branch. The K line
# re-raised 1.0 -> 3.0 to force liftoff commitment on the replanted plant;
# that job is done. Cross-lineage hint (confounded by plant/gains era, but
# directional): the P8/G2 line trained AND flew at gait_ref 1.0 and idled
# far calmer (G2i tipped passively where K4b hot-marches).
# Ladder, one variable per rung:  K4b -> L1 (calm penalties) -> L2
# (penalties + prior anneal). L1-K4b isolates the penalties; L2-L1
# isolates the anneal. Both run the same pickup-free acceptance gate
# (L1 header); fly whichever passes with the widest margin.
# Risk: the prior was the amplitude scaffolding — if the march decays
# toward shuffle, idle_liftoff_rate < 0.03 sustained = kill (same rule as
# L1; clock_contact + the steepened swing_clearance ramp now own liftoff).
# Watch reward/gait_ref shrink toward its idle leak floor — expected, not
# an alarm; judge on the liftoff/flight/vz diagnostics.
# GPU 1 free as of 08-24 late (K5b stopped).
# NOTE: resume re-reads cfg defaults — ALL overrides re-passed here
# (gait_ref=1.0 happens to equal the cfg default; passed explicitly so the
# arm's contract survives future default drift).
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
  '--run_name', 'L2_gaitref10',
  '--seed', '44',
  '--resume',
  '--load_run', '2026-08-23_08-56-43_K4b_resume3420',
  '--checkpoint', 'model_3600.pt',
  'agent.algorithm.entropy_coef=0.005',
  'agent.algorithm.desired_kl=0.008',
  'env.rew_scale_gait_ref=1.0',
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
  -RedirectStandardOutput '<REPO>\RL\logs\train_L2_gaitref10.out.log' `
  -RedirectStandardError  '<REPO>\RL\logs\train_L2_gaitref10.err.log' `
  -WindowStyle Hidden -PassThru
"Launched PID $($p.Id)"
