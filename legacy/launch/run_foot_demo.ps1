# ARCHIVED launch script (tiptoe-biped legacy/launch). Kept as the record of the exact recipe
# for this training arm. Machine paths were replaced with <REPO>, <ISAACLAB>, <CONDA_ENV>, GPU-<uuid>.
# Not runnable as-is: use RL/launch/ for the maintained template and paste this arm's overrides.

# Visual A/B for the ball-foot contact material: the same S4a recipe, run twice,
# differing ONLY in the foot parameters. Launch the two in separate shells and
# watch the same seed side by side.
#
#   powershell -File run_foot_demo.ps1 -Arm rigid       # k=0   d=0     r=0    (legacy plant)
#   powershell -File run_foot_demo.ps1 -Arm soft        # k=200  zeta 1  r=0   (squishy foot)
#   powershell -File run_foot_demo.ps1 -Arm stiff       # k=20000 zeta 1 r=0  (soft's A/B pair)
#   powershell -File run_foot_demo.ps1 -Arm bouncy      # k=1000 zeta 1  r=0.9 (springy + bouncy)
#   powershell -File run_foot_demo.ps1 -Arm stance      # k=5000 zeta 0.3 r=0.63 (loaded-foot test)
#   powershell -File run_foot_demo.ps1 -Arm stancek0    # k=0    zeta 0    r=0.63 (its control)
#   powershell -File run_foot_demo.ps1 -Arm user        # exactly what you typed, rest fixed
#   powershell -File run_foot_demo.ps1 -Arm slick       # friction 0 — no traction (FAILS)
#   powershell -File run_foot_demo.ps1 -Arm r2          # restitution 2  (energy-gaining)
#   powershell -File run_foot_demo.ps1 -Arm r20         # restitution 20 (absurd, OOD)
#
# To test a number of your own, pick any arm and override on top of it:
#
#   powershell -File run_foot_demo.ps1 -Arm stance -FootStiffness 15000 -FootZeta 0.7
#   powershell -File run_foot_demo.ps1 -Arm stance -FootStiffness 0      # this arm's control
#   powershell -File run_foot_demo.ps1 -Arm stance -FootRest 1.0
#
# Stiffness is built for you and the settled height is logged every step, so one
# run answers "did it engage" without needing a second shell to A/B against.
#
# `-Arm user` is the recipe you were running with the two corrections that this
# arm itself made unnecessary (restitution 100.0 -> 1.0 was the ValueError you
# hit; both are now legal). It is the arm to compare against `rigid`.
#
# IF YOU WANT TO SEE THE ROBOT FAIL, USE `slick`. Stiffness and restitution are
# both force-per-contact-area properties: they only matter while the balls are
# actually loaded, and the S4a policy's stance keeps them lightly loaded, which
# is why turning those knobs looks like a no-op. Foot friction is different — it
# is the entire shear capacity of the contact — so 0 fails the robot from any
# pose. `r2`/`r20` are the "extreme and out of distribution" arms.
#
# ══ THE PAD ARMS ARE GONE (2026-09-17) ═══════════════════════════════════════
# The four pad arms (`padnull`, `padlive`, `padband`, `padsoft`) and the pad
# lines inside the training-parity arms have been DELETED, and no arm in this
# file builds a pad any more. Reason, stated once so it is never re-litigated:
# PadContact's reference plane sits 12 mm BELOW the rigid floor, so the ball's
# contact is with the floor and the pad's force is identically zero. Everything
# you felt while tuning it was the material's restitution, which is real.
#
#   `trainplantpin` (your own note, kept verbatim at that arm):
#     pad k        1.0 … 15000 … 50000   no difference
#     pad m_eff    0.001 … 100           no difference
#     pad thickness 0.012 … 990012 mm    no difference
#     foot_restitution 0.63 -> 0.0       really makes it not bouncy, so it works
#
# That is the signature of a force that is never applied, not of a knob that
# needs a better value. Do not re-propose the pad, do not "fix" it by flipping
# the plane sign (a sign flip alone gives a frictionless hovering foot), and do
# not tune `pad_*` here — those overrides no longer exist in this file.
#
# The live home for foot CONTACT dynamics is the material path the arms below
# use. It is a per-RUN constant, not a per-env DR: one material prim is shared
# by every env and PhysX never re-reads a mid-run edit, which is why the `R`
# hotkey exists. To vary it across a training distribution you need a separate
# arm per value, or a run-level band — see the S7 launch script's PINNED FOOT
# MATERIAL block.
#
# ── The training-parity arms (2026-09-16, pad lines removed 09-17) ──────────
# The material arms above leave the belt at its default and the restitution band
# off, so a policy trained under the S-line plant is being evaluated on a plant
# it never saw. `trainplant` is the fix: every LIVE S7 channel exactly as
# `launch_train_S7_fullDR.ps1` passes it.
#
#   powershell -File run_foot_demo.ps1 -Arm trainplant      # S7 plant, DR draws
#   powershell -File run_foot_demo.ps1 -Arm trainplantpin   # S7 plant, pinned
#
# Two arms because the training values are DR BANDS, not numbers, and the two
# are for different questions:
#
#   `trainplant`  draws per env per episode from the S7 bands, exactly as
#                 training does — belt k ~ U[100, 1000] N m/rad, belt damping
#                 ~ U[0, 0.3] motor-side, foot restitution ~ U[0.2, 0.8]. Use it
#                 to ask "is the deployed policy in-distribution". With
#                 -NumEnvs > 1 the envs differ, so the numbers you write down are
#                 meaningless unless you read the per-env draw off the startup
#                 [PLANT] line.
#
#   `trainplantpin`  the same plant with every band collapsed to a single value,
#                 so N envs run the identical plant and two runs are comparable.
#                 Belt k 550 (band midpoint), belt damping 0.15, restitution
#                 pinned at the S7 carrier 0.63 (band ends at 0 -> the per-env
#                 write is skipped and the material keeps the carrier its spawn
#                 wrote — which is also what the restitution ROW DISCOVERY needs;
#                 a band that writes into no rows is an unbound knob).
#                 This is the arm for A/B against `rigid`, not the arm for "what
#                 training saw" — the pinned values are inside the training bands
#                 but are not the training distribution.
#
# Both arms leave the foot MATERIAL rigid, so their contact is the legacy one.
# Override `-FootStiffness` on top of either to ask whether the S7 belt /
# restitution bands interact with a compliant foot.
param(
  [ValidateSet('rigid', 'soft', 'stiff', 'bouncy', 'user', 'slick', 'r2', 'r20',
               'stance', 'stancek0',
               'trainplant', 'trainplantpin')]
  [string]$Arm = 'trainplant',
  [int]$NumEnvs = 1,
  [int]$AutologSteps = 0,
  # Live foot-material overrides, applied ON TOP of the chosen arm. -1 = leave
  # the arm's own value. These are the only foot contact knobs this file has.
  [double]$FootStiffness = -1.0,
  [double]$FootZeta = -1.0,
  [double]$FootRest = -1.0
)
$ErrorActionPreference = 'Stop'

$envRoot = '<CONDA_ENV>'
$env:CUDA_VISIBLE_DEVICES = '2'                 # RTX 4090D
$env:CARB_APP_PATH = '<ISAACLAB>\_isaac_sim\kit'
$env:EXP_PATH      = '<ISAACLAB>\_isaac_sim\apps'
$env:ISAAC_PATH    = '<ISAACLAB>\_isaac_sim'
$env:PYTHONPATH    = ';<ISAACLAB>\_isaac_sim\site'
$env:RESOURCE_NAME = 'IsaacSim'
$env:CONDA_PREFIX  = $envRoot
$env:PATH = "$envRoot;$envRoot\Library\mingw-w64\bin;$envRoot\Library\usr\bin;$envRoot\Library\bin;$envRoot\Scripts;$envRoot\bin;" + $env:PATH
$env:PYTHONUNBUFFERED = '1'

$RL = '<REPO>\RL'
Set-Location $RL

$CKPT = "$RL\logs\rsl_rl\p9_replant\2026-09-09_22-13-40_S4a_measDR\model_7780.pt"
# $CKPT = "<REPO>\control\r1r_3150.pt"
# $CKPT = "$RL\logs\rsl_rl\p9_replant\2026-09-15_10-57-26_S6_restDR\model_7890.pt"
# $CKPT = "$RL\logs\rsl_rl\p9_replant\2026-09-18_23-09-51_S8_vendorJ\model_7910.pt"

if (-not (Test-Path -LiteralPath $CKPT)) { throw "checkpoint missing: $CKPT" }

# Your recipe, byte for byte, minus the two foot lines this arm sets itself.
$RECIPE = @(
  'env.rew_scale_gait_ref=3.0',
  'env.sat_term_threshold=0.5', 'env.sat_term_tau_s=0.5',
  'env.swing_clearance_zero_m=0.078',
  'env.servo_kp_hip_pitch=20.0', 'env.servo_kd_hip_pitch=1.4',
  'env.servo_kp_knee=30.0', 'env.servo_kd_knee=1.5',
  'env.hip_pitch_armature=0.0385',
  'env.swing_clearance_target_m=0.015',
  'env.rew_scale_no_flight=-15.0', 'env.rew_scale_vz_penalty=-8.0',
  'env.rew_scale_action_smoothness=-0.24', 'env.rew_scale_action_snap=-0.06',
  'env.exec_noise_target_bias_max=0.03', 'env.exec_noise_target_std=0.015',
  'env.noise_joint_pos=0.18', 'env.init_base_orient_noise_rad=0.12',
  'env.push_enable=true', 'env.push_probability=0.001',
  'env.push_force_min=10.0', 'env.push_force_max=40.0', 'env.push_duration_steps=10',
  'env.dr_action_delay_min=1', 'env.dr_action_delay_max=2',
  'env.dr_imu_delay_min=1', 'env.dr_imu_delay_max=1',
  'env.dr_joint_delay_min=1', 'env.dr_joint_delay_max=1',
  'env.dr_floor_friction_min=0.6', 'env.dr_floor_friction_max=1.1',
  'env.dr_stiffness_mult_min=0.90', 'env.dr_stiffness_mult_max=1.10',
  'env.dr_armature_mult_min=0.80', 'env.dr_armature_mult_max=1.20',
  'env.dr_effort_mult_min=0.9',
  'env.dr_com_offset_max=0.03',
  'env.dr_com_shift_x_max=0.02', 'env.dr_com_shift_y_max=0.01',
  'env.dr_proj_grav_bias_max=0.13',
  'env.dr_mass_mult_min=1.00', 'env.dr_mass_mult_max=1.14',
  'env.dr_ang_vel_bias_max=0.05', 'env.dr_gravity_tilt_max_deg=2.0',
  'env.pickup_enable=false',
  'env.standup_ramp_mode=path'
)

# Foot arms. damping = 2*zeta*sqrt(k*m_eff) at zeta=1, m_eff=1 kg, i.e. the
# ball's own critically damped contact mode — the ratio form the cfg exposes.
switch ($Arm) {
  # The rigid arm must leave the ratio at its NaN default: passing an explicit
  # 0.0 makes it a real damping request, and resolved_foot_contact_damping()
  # refuses a ratio with no stiffness to damp.
  'rigid'  { $FOOT = @('env.foot_contact_stiffness=0.0',
                       'env.foot_restitution=0.0') }
  'soft'   { $FOOT = @('env.foot_contact_stiffness=200.0',
                       'env.foot_contact_damping_ratio=1.0',
                       'env.foot_restitution=0.0') }
  # `stiff` exists ONLY to pair with `soft`: identical restitution and identical
  # damping ratio, stiffness differing by 100x. If the two rollouts come back
  # bit-identical, the contact is carrying no load in this pose and the material
  # (however live it is elsewhere) cannot show up here.
  'stiff'  { $FOOT = @('env.foot_contact_stiffness=20000.0',
                       'env.foot_contact_damping_ratio=1.0',
                       'env.foot_restitution=0.0') }
  # `bouncy` pins K=1000, D=2*zeta*sqrt(K*m_eff) at zeta=1, m_eff=1 kg.
  'bouncy' { $FOOT = @('env.foot_contact_stiffness=1000.0',
                       'env.foot_contact_damping_ratio=1.0',
                       'env.foot_restitution=0.9') }
  'user'   { $FOOT = @('env.foot_contact_stiffness=50000.0',
                       'env.foot_contact_damping_ratio=0.3',
                       'env.foot_contact_effective_mass=0.05',
                       'env.foot_restitution=0.63') }
  # ── The loaded-foot pair (2026-09-17). The arms above all inherited the S4a
  # crouch, where the balls hang 1.6 mm clear of the floor (lowest toe height
  # +0.0856 m against a 0.084 m contact line) and a foot contact knob has nothing
  # to act on. These two run the S7 plant, where the balls DO carry the robot
  # (ball centre bottoms at 18.7 mm against a 20 mm radius), so a compliance
  # number has something to bite on.
  #
  # `stance` and `stancek0` differ ONLY in stiffness+damping; restitution is the
  # S7 material carrier in both, which is also what keeps the restitution row
  # discovery bound. Compare the two runs' SETTLED ball-centre height and knee
  # torque. Bit-identical => the compliant shape is not engaging against the
  # rigid floor, and the fix is a compliant floor material, not a different
  # number here.
  #
  # The [PLANT] foot-contact line at startup confirms the material was CREATED
  # and bound to both ball meshes; it does not confirm the solver uses it, and
  # nothing in this repo logs ball height per step. The discriminator is a
  # recorded run either side, so use -AutologSteps on both arms of the pair.
  #
  # Anchor: 27.3 N per foot gives 5.5 mm of sag at k=5000, 0.9 mm at k=30000.
  # zeta here is the material's d = 2*zeta*sqrt(k*m_eff) against the ball's own
  # contact mode, NOT the pad's parameterization. At k=5000, m_eff=1 kg:
  # zeta 0.3 -> d = 42.4 N s/m, tau = 2m/d = 47 ms. Raise zeta before raising k
  # when the goal is a contact that stops ringing.
  'stance'   { $FOOT = @('env.foot_contact_stiffness=5000.0',
                         'env.foot_contact_damping_ratio=0.3',
                         'env.foot_contact_effective_mass=1.0',
                         'env.foot_restitution=0.63') }
  'stancek0' { $FOOT = @('env.foot_contact_stiffness=0.0',
                         'env.foot_restitution=0.63') }
  # The three "make it fail" arms. Unlike stiffness/restitution, NONE of these
  # depend on the balls carrying load, which is why they are the ones that work
  # on a policy whose stance keeps its feet lightly loaded.
  #
  # `slick` removes the ball's traction entirely. The floor material is 1.0 with
  # friction_combine_mode="multiply", so the ball's 0.0 is the ball's friction:
  # the feet cannot push sideways or hold a shear load at all. This is the
  # single most reliable failure arm — a biped with frictionless feet has no way
  # to arrest a lateral CoM drift, whatever it does with its joints.
  'slick'  { $FOOT = @('env.foot_contact_stiffness=0.0',
                       'env.foot_static_friction=0.0',
                       'env.foot_dynamic_friction=0.0') }
  # Energy-gaining contact. PhysX reads restitution > 1 without complaint and
  # the ball returns more energy than the impact carried, so every touchdown
  # pumps energy in. 20 is deliberately absurd: this is an out-of-distribution
  # plant, not a physical claim, and the top of the Home ladder.
  'r2'     { $FOOT = @('env.foot_contact_stiffness=0.0',
                       'env.foot_restitution=2.0') }
  'r20'    { $FOOT = @('env.foot_contact_stiffness=0.0',
                       'env.foot_restitution=20.0') }
  # ── The per-env pad arms are DELETED (2026-09-17) with the whole pad channel:
  # the pad's reference plane sits below the floor, so no arm in this file builds
  # one and no `pad_*` override is passed. See the header block. The four arms
  # were `padnull`, `padlive`, `padband`, `padsoft`.
  # ── The training-parity arms. Copied knob-for-knob out of
  # `launch_train_S7_fullDR.ps1` (the S7 block in $argList); change them there
  # first if S7 is ever edited, and re-copy — these two lists must not drift
  # apart. Only the LIVE channels are copied: belt series + foot restitution.
  # The pad lines S7 used to pass are gone from both files together.
  #
  # These set the material to the S7 carrier value 0.63 rather than 0.0, which is
  # deliberate and load-bearing: the per-env restitution DR addresses the material
  # rows whose restitution READS BACK as 0.63 (`_foot_restitution_rows` is
  # discovered by matching that value), so a run that left it at 0.0 would build
  # no material at all and the band would be an unbound knob.
  'trainplant' {
    $FOOT = @('env.belt_series_stiffness_min=100.0',
              'env.belt_series_stiffness_max=1000.0',
              'env.belt_series_damping_min=0.0',
              'env.belt_series_damping_max=0.3',
              'env.foot_restitution=0.63',
              'env.dr_foot_restitution_min=0.2',
              'env.dr_foot_restitution_max=0.8',
              'env.dr_foot_restitution_buckets=64')
  }
  # Same plant, bands collapsed. Note what "pinned" means for each knob and why
  # these particular values, because collapsing a band is NOT how you pin it:
  #
  #   belt k      100..1000 -> 550, the band midpoint.
  #   belt damp   0..0.3    -> 0.15, the band midpoint of the EXPLICIT motor-side
  #               band (N m s/rad). 0.15 is not an arbitrary half: the ratio only
  #               depends on k, so half the damping at half the stiffness IS
  #               zeta = 0.15 at k=550, which is the near-critical value the S7
  #               header's zeta-vs-k table implies (zeta 0.147 at k=100, 0.048 at
  #               k=1000). Do not "correct" it to 0.149 by re-deriving through
  #               damping_for_ratio — the band is explicit, not a ratio band.
  #   restitution 0.63 + both band ends 0 -> the per-env write is skipped
  #               entirely (`dr_foot_restitution_max > 0` gates it) and the
  #               material keeps the carrier its spawn wrote.
  #
  # So this arm's plant is: belt k=550, damping 0.15 motor-side, restitution
  # 0.63, RIGID feet. All three sit inside the S7 bands; the belt pair is the S7
  # band midpoint and restitution is pinned at the carrier.
  #
  # Foot material stays rigid here, so `-FootStiffness` can be layered on top:
  #
  #   powershell -File run_foot_demo.ps1 -Arm trainplantpin -FootStiffness 5000
  'trainplantpin' {
    $FOOT = @('env.belt_series_stiffness_min=550.0',
              'env.belt_series_stiffness_max=550.0',
              'env.belt_series_damping_min=0.15',
              'env.belt_series_damping_max=0.15',
              'env.foot_restitution=0.63')
  }
}

# ── Live foot-material overrides, applied over whatever the arm set. Appending is
# what makes them win: Hydra takes the LAST occurrence of a key, and the arm's
# value is earlier in the list.
#
# The "leave it alone" sentinel is a variable with a name of its own, NOT the -1
# literal: inside a function PowerShell resolves a bare $FootStiffness to the
# SCRIPT-scope parameter, which is always -1, so a function that tested the
# literal would silently skip the decision it exists to make.
$KEEP = -1.0
function Get-FootArg([string]$key) {
  $hit = $FOOT | Where-Object { $_ -like "$key=*" } | Select-Object -Last 1
  if ($hit) { return [double]($hit -replace '^[^=]*=', '') }
  return $KEEP
}
if ($FootStiffness -ne $KEEP) { $FOOT = @($FOOT | Where-Object { $_ -notlike 'env.foot_contact_stiffness=*' })
                                $FOOT += "env.foot_contact_stiffness=$FootStiffness" }
if ($FootRest -ne $KEEP)      { $FOOT = @($FOOT | Where-Object { $_ -notlike 'env.foot_restitution=*' })
                                $FOOT += "env.foot_restitution=$FootRest" }
if ($FootZeta -ne $KEEP) {
  # resolved_foot_contact_damping() raises on a ratio above zero with no
  # stiffness to damp, so catch it here with the arm named rather than as a
  # traceback out of the cfg.
  $effK = Get-FootArg 'env.foot_contact_stiffness'
  if ($FootStiffness -ne $KEEP) { $effK = $FootStiffness }
  if ($effK -le 0.0) {
    throw "-FootZeta $FootZeta needs a foot stiffness to damp. Pass -FootStiffness too (e.g. 5000), or use an arm that sets one (soft, stiff, stance)."
  }
  $FOOT = @($FOOT | Where-Object { $_ -notlike 'env.foot_contact_damping*' })
  $FOOT += "env.foot_contact_damping_ratio=$FootZeta"
}
# Last, and deliberately after the zeta branch: turning stiffness OFF while
# leaving an arm's damping ratio behind is the one combination the cfg refuses
# (a ratio with nothing to damp). `-Arm stance -FootStiffness 0` is the control
# for `stance`, so it has to work. `damping_ratio=0.0` is the legal spelling of
# "no damping" next to `stiffness=0.0`, and it is what the `rigid` arm uses.
elseif ((Get-FootArg 'env.foot_contact_stiffness') -le 0.0) {
  $dz = Get-FootArg 'env.foot_contact_damping_ratio'
  if ($dz -gt 0.0) {
    Write-Output "[demo] note: stiffness is 0, so the damping ratio $dz has nothing to act on; writing 0.0 (same plant — a rigid contact has no spring to damp)."
    $FOOT = @($FOOT | Where-Object { $_ -notlike 'env.foot_contact_damping*' })
    $FOOT += 'env.foot_contact_damping_ratio=0.0'
  }
}

$argList = @('-u', "$RL\scripts\rsl_rl\play_sim2real.py",
  '--task=Template-standup_curriculum-v0',
  "--checkpoint=$CKPT",
  '--num_envs', "$NumEnvs",
  '--gait_phase_init', '0',
  '--seed=0')
if ($AutologSteps -gt 0) {
  $argList += @('--headless', '--autolog_steps', "$AutologSteps", '--autolog_runs', '1', '--no_obs_summary')
}
$argList += $RECIPE
$argList += $FOOT

Write-Output "[demo] arm=$Arm  foot: $($FOOT -join '  ')"
$ErrorActionPreference = 'Continue'   # Isaac writes to stderr
& "$envRoot\python.exe" @argList
$code = $LASTEXITCODE
Write-Output "[demo] arm=$Arm exit=$code"
exit $code
