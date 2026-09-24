# CONCISE SUMMARY of rl_env_cfg_p6.py — for AI context-sharing only.
# Not runnable. Numeric values kept where they carry design intent (reward
# scales, DR ranges); implementation boilerplate (exact sim/scene ctor calls,
# body name lists) is abstracted to one-liners. See rl_env_cfg_p6.py for the
# authoritative, exact config.

"""
rl_env_cfg_v7 — bipedal ball-foot standup/walk RL, DCMotor actuator rewrite.

WHY DCMotor (replacing PhysX implicit PD):
  Hardware showed (1) bang-bang on hip_roll — policy assumed torque
  available up to effort_limit at any speed, real motor can't deliver
  it near v_limit; (2) phase-margin collapse under disturbance — policy
  over-fit sim's idealized PD response, lost authority when real back-EMF
  cut available torque at speed.
  DCMotor models: tau_max(v) = saturation_effort * (1 - |v|/v_limit),
  clipped to effort_limit. Same speed-torque dropoff in sim and real.

OTHER v7 CHANGES vs p6h:
  - Action scales recalibrated so action in [-1,1] * scale == effort_limit
    at zero velocity (hip_roll 0.30, hip_pitch 0.75, knee 2.05/2).
  - Thermal penalty zeroed out: DCMotor already clips at continuous
    effort, so the old (torque/continuous)^2 penalty is identically 0.
  - Deleted the p6 Kp/Kd/action-scale curriculum (train->real interpolation):
    DCMotor's explicit PD is stable at real Kp from the start.
  - DR expanded to actuator physics: stiffness/damping/effort/armature/
    friction multipliers, per-env voltage_factor (scales velocity_limit).
  - Added base roll/pitch init noise at reset, scaled by (1-difficulty).
  - Damping DR decoupled from stiffness DR (was coupled in p6f to avoid
    catatonic corners; DCMotor's added friction terms make that
    unnecessary and the coupling suppressed real motor variance).

A/B FORK (gait-reference prior, separate scout run, same checkpoint):
  Only delta vs main run: rew_scale_gait_ref != 0 (see reward section).
  Setting it to 0 recovers the main run bit-for-bit.

RATES: physics 500 Hz, control 100 Hz (decimation 5).

FILE MAP: §1 rates/spaces, §2 sim, §3 robot/actuators, §4 obs pipeline,
§5 commands, §6 standup/kneel curriculum, §7 push (disabled), §8 pickup
("virtual hand"), §9 reward scales, §10 termination/obs-noise,
§11 action processing, §12 domain randomization (impact-tagged).
"""

# ── §1 Rates & spaces ──
# decimation=5, episode_length_s=30, action_space=6
# obs_history_steps=5 frames stacked (post-delay) -> observation_space = 36*5 = 180
#   why: with 1-7 step control-loop delay, a feedforward net needs history
#   to reconstruct velocity from stale position (implicit finite-diff).
# Asymmetric actor-critic: critic also gets priv_obs_dim=78 privileged obs
#   (true base velocity, pickup state, full per-env DR vector) that the
#   actor never sees — removes value-function shocks from things the
#   deployed policy can't observe.

# ── §2 Simulation ── dt=1/500, physics_material friction=1.0/1.0 multiply-combine

# ── §3 Robot / actuators ──
# Three DCMotorCfg actuators, measured per-joint:
#   hip_roll : saturation 6.0 Nm, continuous 6.0 Nm (=6.0? see file for exact
#              effort_limit), v_limit 28 rad/s, stiffness 6.0, damping 0.3,
#              armature 7e-4, friction (static/dynamic/viscous) 0.12/0.10/0.003
#   hip_pitch: saturation 17.0 Nm, v_limit 27.5, stiffness 8.0, damping 0.5,
#              armature 4.2e-3, friction 0.12/0.10/0.003
#   knee     : saturation 23.2 Nm (17 * 30/22 belt ratio), v_limit 20.0,
#              stiffness 8.0, damping 0.5, armature 7.81e-3,
#              friction 0.06/0.05/0.003
# 32768 parallel envs, env_spacing 2.5. Body CoM masses tracked for 7 named
# links (torso/hip/thigh/shin/etc, ~0.2-1.7 kg each).

# ── §4 Observation pipeline (sim2real shaping, not randomization) ──
# lin_vel_obs_dropout_prob = 1.0: real robot has no body-frame velocity
#   estimator, so lin_vel obs slots are always zeroed (matches hardware).

# ── §5 Commands ──
# vx command in [-0.10, 0.10] m/s, 60% chance of zero.
# yaw_rate command in [-0.1, 0.1] rad/s, 50% chance of zero.
# resampled every 1000 steps. Gait clock freq = 2.5 Hz.

# ── §6 Standup / kneel curriculum ──
# height range [0.32, 0.55] discovered by policy, not commanded.
# Kneel pose is a measured fixed joint config (deep knee bend, ~180cm-ish
# pitch angles) with quat matching a face-down kneel orientation.
# curriculum_enable=False in current run: p6e found the interpolated
# kneel->stand init + linear standup-height reward already gives a smooth
# gradient, so alpha starts at 1.0 (no ramp needed), though kneel-branch
# sampling still requires curriculum_enable's gating flag.
# standup_grace_seconds=4.0, kneel_start_probability=0.5.
# init_base_orient_noise_rad=0.08, scaled by (1-difficulty).

# ── §7 Push disturbance — DISABLED (push_enable=False) ──

# ── §8 Pickup ("virtual hand") ──
# State machine: IDLE -> LIFT -> HOLD -> LOWER -> RELEASE_TRANSITION -> IDLE.
# Simulates a human picking the robot up mid-episode (or at spawn).
# Closed-loop critically-damped PD (omega=7 rad/s, damping_ratio=1.2) drives
# base z to a lift target (0.20-0.35m added height); separate PD drives
# orientation to upright (kp=25, kd=6, torque-capped) with yaw loosened so
# the policy keeps rotational authority. Small hold-phase wobble (1.5cm xy,
# 1cm z at 0.4Hz) simulates an unsteady human grip.
# Release: contact-based early release if a foot dips near ground during
# LOWER; otherwise a latency window (5-40 steps) linearly fades PD
# authority to zero, simulating weight transfer from hand to legs.
# pickup_probability ~0.0005/step (~1 event per 2000 steps mid-episode);
# pickup_initial_probability=0.5 (half of resets start already being placed
# down, lifted higher than mid-episode pickups).

# ── §9 Reward scales (the actual behavior-shaping knobs) ──
# Liveness:        alive +6.0, terminated -10.0
# Standup:         height-ramp +10.0, fall -15.0 (linear ramp kneel->stand
#                  height, no sigmoid — p6e simplification)
# Tracking:        vel +2/3 (sigma 0.15), yaw_rate +2.0 (sigma 0.30),
#                  both EMA-smoothed (decay 0.99, ~1s time constant) so
#                  reward tracks a windowed average, not instantaneous vel.
# Gait:            gait_step +4.0, feet_air +1.5, foot_swing +1.0,
#                  foot_progress +100, back_leg_step +80,
#                  clock_contact +2.0 (binary: rewards matching the gait
#                    clock's stance/swing phase per foot; clamped at 0 so
#                    standing-still=0, wrong-phase=0 -- NOT -1, since a
#                    -1 wrong-phase penalty was found to tax early
#                    exploration when a first tentative step is a coin flip),
#                  swing_clearance +2.0 (dense ramp rewarding toe height
#                    above a near-ground threshold during the swing half of
#                    the gait cycle, directly opposing a "shuffle" gait that
#                    never truly lifts the foot).
# A/B fork gait-reference prior (rew_scale_gait_ref, default 3.0/5, 0 in main
#   run): exp(-[(hp1_dev - A*sin(phi))^2 + (hp2_dev - A*sin(phi))^2] / sigma^2),
#   phi = existing gait clock. Injects HOW to step (an explicit antiphase
#   hip-pitch sinusoid target) instead of waiting for the policy to
#   discover stepping on its own -- a known trick from Cassie-lineage work
#   for exactly this "understands the clock but won't commit amplitude"
#   failure mode. Same-signed reference for both legs is intentional: the
#   L/R mirror table negates+swaps legs, so an antiphase physical gait
#   satisfies hp1_dev=hp2_dev=A*sin(phi) in joint coordinates, making the
#   term mirror-invariant. Meant as scaffolding to anneal away once real
#   stepping emerges, not a permanent reward term.
# Anti-gallop trio: no_flight -5.0 (penalize both feet airborne at once),
#   vz_penalty -2.0, stance_symmetry -80.0 (EMA-decayed 0.98) -- together
#   prevent a bounding/galloping gait on a biped that should walk.
# Symmetry: leg_symmetry -80.0, angle_symmetry -40.0 (halved from -80 in
#   p6e after it was found to dominate ~14.7% of total reward magnitude and
#   push toward a fragile low-amplitude shuffle).
# Stance/balance: height +1.0, balance +5.0, knee_over_hip -0.02,
#   hip_roll_split -2.0, upright -2.0 (penalizes horizontal projected
#   gravity = sin^2(tilt); gated off during grace/pickup).
# Anchoring (zero-command): position_anchor -8.0, heading_anchor -8.0, with
#   a 4cm deadband (drift inside is free, re-squared penalty outside) so
#   stepping-in-place body sway isn't taxed as drift. yaw_drift -1.0.
# Thermal: 0.0 (redundant given DCMotor's own clipping, kept for log-key
#   stability). energy -0.001.
# Action regularization: action_rate (small negative), action_smoothness
#   (2nd-derivative/jerk penalty, "5x on resume"), action_snap (3rd-
#   derivative penalty), action_magnitude on clipped action^2 (-0.02),
#   raw_action on PRE-CLIP action^2 (-0.3*10/20) -- gives gradient even
#   when actions saturate, preventing pre-clip logits from drifting to
#   extreme unclipped values.
# idle_step_gate=0.5: floor value for gait rewards even at zero command,
#   because a point-foot biped has no static sagittal balance and needs a
#   baseline stepping cadence to stay upright at rest.
# idle_motion_relief=0.08: fraction of full action-derivative penalty kept
#   at zero command (relief for the fast catch-steps stepping-in-place needs).

# ── §10 Termination limits & legacy obs noise ──
# max_tilt_rad=1.8, min_height=0.30, joint-split limits 1.8 rad each.
# noise_joint_pos=0.1, noise_base_vel_lin=0.3, noise_base_vel_ang=0.2
#   (this is a separate legacy noise block from the §12 DR noise -- kept
#   for continuity, don't confuse the two).

# ── §11 Action processing ──
# Action LPF alpha=0.6 (~8 Hz cutoff at 100 Hz).
# continuous_effort refs per joint (1.8/6.0/8.18 Nm) define "above
#   continuous" for diagnostics.
# Action scales: train==real in v7 (no curriculum), Kp*scale=effort_limit
#   so action +-1 maps to +-continuous torque at zero velocity.
# Actuator-lag prefilter (sysid finding: real actuator position response is
#   2nd-order, not pure transport delay): y'' + 2*zeta*omega*y' + omega^2*y
#   = omega^2*u, advanced at 100Hz on the PD target. Per-joint omega
#   (21/37.9/27.6 rad/s) and zeta (0.414/0.803/0.580) are sysid-identified
#   centrals. This is distinct from and stacks with the §12b action_delay
#   (a pure transport dead-time): dead-time first, then this 2nd-order lag.

# ── §12 Domain randomization (impact-tagged HIGH/MED/LOW = estimated
#         correlation with episode length, i.e. how much each axis gates
#         "does the robot stay up." Priors from physics reasoning, meant
#         to be confirmed/re-tagged against logged per-DR eplen correlation.) ──
# HIGH: action_delay & imu/joint obs delay (loop dead time -> phase margin),
#       floor_friction (traction -> slip/fall).
# MED:  com_offset, mass_mult, joint_friction (Coulomb/breakaway),
#       stiffness_mult (Kp), actuator-lag omega/zeta mult.
# LOW:  damping/armature/dyn/visc friction mult, voltage_factor,
#       effort_mult (narrow 0.9-1.0), all per-step obs noise, all
#       per-episode obs biases.
#
# 12a actuator physics mults (per-joint, per-episode): stiffness
#   0.85-1.15, damping 0.7-1.3, effort 0.9-1.0 (capped at nominal),
#   armature 0.7-1.3, joint/dyn/visc friction 0.5-2.0 each,
#   voltage_factor 0.87-1.07 (single scalar per robot/battery, scales all
#   3 joints' velocity_limit). Actuator-lag DR: omega mult 0.82-1.18,
#   zeta mult 0.70-1.30 (wider band -- overshoot is load-sensitive;
#   deliberately stays underdamped across the whole band).
# 12b control-loop latency (quantized to 10ms steps, bench-measured):
#   action_delay 1-2 steps, IMU obs delay locked at 1 step (never 0 --
#   avoids the policy learning a zero-latency reflex it won't have on
#   hardware), joint/encoder obs delay locked at 2 steps (measured ~10ms
#   behind IMU, over CAN).
# 12c per-step sensor noise (additive uniform): proj_grav 0.05, ang_vel
#   0.15 rad/s, joint_vel 0.6 rad/s (blind prior, couldn't measure
#   stationary), joint_pos 0.008 rad.
# 12d per-episode sensor bias (constant per-env offset, models
#   miscalibration): joint_pos +-0.07 rad, proj_grav +-0.03, ang_vel
#   +-0.03 rad/s.
# 12e world/body physics: floor_friction mult 0.5-1.1 (carpet->laminate->
#   rubber), mass mult 0.88-1.12 (single scalar, all links), per-link CoM
#   offset +-1.5cm.
# 12f ground slope: up to 1 deg tilt, random direction per env/episode --
#   simulates real floor imperfection as a persistent horizontal force
#   (unlike a push, this doesn't decay).
