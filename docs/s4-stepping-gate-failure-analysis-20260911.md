# S4 stepping-gate failure investigation

**Date:** 2026-09-11  
**Data:** `data/r1r_flight_20260906/`, `data/s4_flight_20260910/`  
**Hardware logger:** `control/balance_log.py`

## Question

Why did every S4 stepping-gate trial fail, while the R1r shuffle reference survived longer?

The analysis was restricted to the first short interval after NN handover because the tether can contaminate later motion. The default comparison window is 0.6 s; one S4a trial was documented as clean to 0.8 s.

## Data and logging contract

There are three R1r recordings from 2026-09-06 and eleven S4/R1r recordings from 2026-09-10. The September 10 folder README records the policy, trim, and known knee-tooth events.

Every supplied trial uses `handover='blind'` (`balance_log.py` key `v`). The first policy frame has the delayed sensor block zeroed and is replicated across the five-frame history. Hardware starts `gait_phase` at 0.0 rad and increments it by about 0.157 rad per 100 Hz tick. The first logged sample is about 2.8 ms after the LED/log anchor.

The NPZ has no tether force, tether state, foot force, contact flag, or tether-onset timestamp. Tether contamination cannot be identified from the NPZ alone and must be anchored to video or another independent physical observation.

There are two logger alignment details that matter when comparing with simulation:

- `obs` contains individual 36-D frames; the policy consumes a rolling 5-frame/180-D history.
- Hardware logs state/torque feedback at the start of the control tick, while the command is sent later in that tick. `gait_phase` is logged after it is incremented, so it is one tick ahead of the phase contained in the logged observation frame.

September 6 R1r files predate the raw-gravity metadata. Their raw gravity was reconstructed as `proj_grav + [0.089, 0, 0]`, following the documented trim history.

The nominal deploy gains and action scales agree between the supplied hardware logs and the S4 twin recipe: gains `[6/0.3, 20/1.4, 30/1.5]` per leg, action scales `[0.30, 0.75, 1.025]`, and target LPF alpha `0.6`. The twin nevertheless includes randomized one-to-two-tick action delay and one-tick IMU/joint observation delay; hardware has no deliberate equivalent sensor-delay buffer.

The original `play_sim2real.py` twin logger also records the policy's raw action before the environment clips it to `[-1, 1]`, while hardware `Policy.forward` clips before logging. About 9–10% of S4 twin action elements exceed that range. Raw twin `target` and hypothetical PD `motor_tau_cmd` therefore cannot be compared directly with the processed hardware target and command channels. The conclusions below use the hardware actions and gravity/rate channels directly and treat raw sim torque/target comparisons as diagnostic only.

## Hypothesis audit

### Policy/checkpoint or observation-history bug — contradicted

I replayed each September 10 NPZ using the corresponding local policy checkpoint, the logged 36-D frames, the exact blind five-frame history construction, and the checkpoint normalizer.

- Maximum action reconstruction error across the first 0.6 s of every trial: below `8e-7`.
- Logged normalizer mean and standard deviation matched the checkpoint exactly.

This rules out a wrong checkpoint, missing normalizer, incorrect action clipping at inference, or a normal five-frame history reconstruction error as the common S4 failure mechanism. The deployed policy is producing the actions recorded in the NPZ.

Replay output: `outputs/s4_failure_20260911/policy_replay.json`.

### Gait-phase handover mismatch — real contract bug, but not sufficient

The original hardware twins in `RL/logs/s4_hwtwin/` started at `gait_phase=pi`, while `balance_log.py` starts hardware at 0.0 rad. This is a half-cycle shift. The first policy input therefore differs in the gait sin/cos channels (`[0, +1]` on hardware versus `[0, -1]` in the original twin), and the cyclic leg commands differ later in the opening.

I ran a preregistered S4a counterfactual with the hardware phase (`0`) and the old twin phase (`pi`), using the same blind/training handover and S4 hardware-twin recipe. Each condition used eight one-second rollouts and preserved the 2026-09-10 inertia contract.

| Condition | Rollouts exceeding 25° total tilt by 0.6 s | Mean maximum tilt |
|---|---:|---:|
| S4a, +0.089 trim, phase 0 | 3/8 | 25.25° |
| S4a, +0.089 trim, phase pi | 5/8 | 25.38° |
| S4a, trim 0, phase 0 | 7/8 | 27.36° |
| S4a, trim 0, phase pi | 7/8 | 27.31° |

No rollout auto-fell within one second. Phase changes did not create a distinct failure signature; trim had the larger effect. Therefore phase must be corrected for future sim-to-real comparisons, but phase alone does not explain the hardware stepping-gate failures.

Results: `outputs/s4_failure_20260911/phase_score.txt`.

### Missing upper-body mass/inertia — contradicted as a sufficient explanation

The S4 twin sweep added 0.3, 0.6, and 1.0 kg to the upper-body mass, with and without the corresponding inertia scaling. All eight rollouts in each arm remained stable for the five-second test. Opening pitch and pitch-rate bursts changed only marginally.

This rejects the specific hypothesis that a uniformly missing upper-body mass/inertia term is sufficient. It does not rule out a different CoM distribution, concentrated inertia, or a contact model error.

### Knee-tooth slip — not the common cause

The README reports tooth skips in selected trials, but all three S4a trials at `+0.089` were recorded as having no tooth skip. They still failed early. Tooth slip remains a serious state-estimation and reliability issue for individual trials, but it cannot explain the common S4 failure.

### IMU trim — changes the margin, does not explain the failure

R1r at `+0.089` remains the most stable group. S4a fails at both `+0.089` and `0.0`. In the provisional first 0.6 s window:

| Group | Mean raw pitch at 0.4 s | Trials crossing 25° total tilt by 0.6 s |
|---|---:|---:|
| R1r, +0.089 | −9.7° | 0/4 |
| S4a, +0.089 | −16.7° | 3/3 |
| S4a, 0.0 | −21.8° | 2/3 |
| S4b, +0.089 | −18.1° | 4/4 |

The trim affects the opening margin, but neither trim produces a reliable S4 stepping recovery.

### Static CoM/zero offsets — fit shuffle, fail stepping prediction

The system-identification run `RL/logs/s4_sysid/20260910_2119/report.txt` fitted whole-body CoM shifts and joint zero biases to R1r trials, then evaluated S4a as held-out data. The best candidate had approximately:

- `com_x = −18 mm`
- `com_z = −55 mm`
- `knee_bias = +0.008 rad`
- `hip_bias = +0.050 rad`

It fit the R1r opening reasonably (`J_fit ≈ 3.1`) but predicted S4a poorly (`J_pred ≈ 21.9`). For held-out S4a, the fitted sim crossed 25° by 0.6 s in only about 4% of rollouts, while hardware crossed it in 67% to 100% of trials depending on trim. The static-offset family therefore explains part of the shuffle response but not the stepping-gate failure.

## Robust hardware/simulation divergence

The original S4 twins survive in simulation, while the hardware develops a much larger sagittal pitch-rate burst during the first catch-step. After correcting for the known comparison confounds (raw sim action/target channels, gait phase, and different command processing), the safe statement is:

- The first hardware policy command is consistent with the deployed policy and is broadly consistent with the twin after leg mirroring.
- Hardware pitch-rate divergence develops after roughly 0.2–0.3 s.
- Representative minimum pitch rates are about `−340°/s` for S4a trim 0 and `−443°/s` for S4a at `+0.089`, versus about `−177°/s` and `−172°/s` in the corresponding twins.
- R1r hardware is less severe, about `−148°/s` in the comparable opening analysis.
- Roll remains comparatively small, so the common opening failure is sagittal.

The measured encoder trajectories do not show a simple universal “the joints did not move” explanation. S4 hardware has substantial knee and hip motion during the opening, and all no-tooth-skip S4a trials fail. The NPZ does not independently verify the actual joint-side knee angle because `sim_joint_pos` is decoded from motor-side encoder data through the belt model.

## Current causal conclusion

The S4 policy is being executed correctly, but its stepping maneuver is outside the real robot's recoverable support/contact envelope. The real plant develops backward angular momentum during the first dynamic catch step that the simulated plant arrests. The evidence does not identify one unique plant parameter yet.

The investigation is now deliberately limited to two plant mechanisms:

1. belt stretch stores and releases energy during the catch step;
2. point-foot/sole contact is more compliant or more restitutive than the rigid simulated contact.

Backlash, tooth slip, CoM offsets, and generic actuator/transport changes are out of scope for this pass. They remain possible in the broader sim-to-real problem, but mixing them into this experiment would make the result uninterpretable.

For some S4a trim-0 trials, the video notes describe the swing foot remaining airborne, followed by the support foot sliding forward and the torso rotating backward. Other trials fail differently, so this sequence should not be generalized to every recording. It does, however, support investigating the first single-support catch as the key event rather than treating the later tether-ended fall as the primary failure.

## Revised plan: virtual elasticity sweep

The robot cannot accept a joint-side encoder, and reliable contact instrumentation is difficult. The next discriminator is therefore a simulation-only plant sweep. Expose the plausible elasticity and bounce parameters in the RL environment, run fixed-policy virtual trials, and compare the resulting opening signature with the short hardware window and the user's video impression.

The current environment does **not** yet contain these two energy-storage mechanisms:

- `BeltKneeDCMotor` models the 22:30 ratio, belt coupling, stale-hip latch, knee feed-forward, motor-speed torque envelope, and rotor inertial coupling. It does not have a series spring/damper state between motor and knee joint, so belt stretch cannot store and return energy.
- The default ground/contact material is rigid (`compliant_contact_stiffness=0`, `compliant_contact_damping=0`) with restitution `0.0`. The existing floor friction range is useful, but it is not a bounce model.

Add the following as independently switchable, per-episode DR parameters. Keep the old values as the exact nominal condition:

1. **Belt series elasticity:** knee-joint spring stiffness, with zero compliance as the current baseline. Sweep stiffness as a dimensionless multiple of the knee servo stiffness — on the deploy recipe (`servo_kp_knee = 30`) that is the boundary where belt deflection and servo target error are the same size — for example `k_belt/k_servo = {∞, 5, 1, 0.27, 0.1}`. Sweep damping as a ratio of the transmission's own mode, `zeta = d_motor / (2*sqrt(k_motor * I_rotor_motor))`, rather than guessing a final viscous coefficient. Frames matter and are fixed in the implementation: **stiffness is entered knee-referred** and converted with `(22/30)²`; **damping is entered motor-side** and converted by nothing, because a torque-over-velocity ratio carries no power of the reduction. The spring sits in the motor-to-joint transmission path, so it changes the delivered knee angle/torque; the encoder observation contract is enforced, not assumed — under elasticity the knee observation channels are fed the motor-encoder decoding `G*m - c*q_hip` rather than the belt-free PhysX joint.
2. **Foot/sole normal compliance:** contact stiffness and damping on the actual foot collision shapes. Start with rigid contact and progressively softer contacts. Use a geometric stiffness grid and damping ratios; do not claim any one value is measured yet.
3. **Normal restitution:** `0`, `0.03`, `0.1`, `0.2` as an independent bounce axis. Keep tangential friction separate so a bouncy-but-not-slippery case can be distinguished from a sliding case.
The existing environment exposes many other DR axes, but they should be held fixed for this experiment. The new belt-spring and foot-contact parameters must be explicit rather than hidden inside an existing friction or actuator multiplier.

Do not randomize every axis immediately and then inspect an undifferentiated success rate. First run one-axis and two-axis fixed-policy grids with identical seeds. Rank candidates by the same early signals that are trustworthy in hardware: pitch at 0.2/0.4/0.6 s, minimum pitch rate, total tilt crossing 25°, and whether the simulated swing/support sequence qualitatively matches video. Then combine only the leading elasticity/contact ranges into a broader DR distribution for retraining.

The first virtual checks should answer these questions:

- Does belt compliance produce the larger backward-rate burst after 0.2 s without requiring a large static pitch offset?
- Does foot compliance/restitution create a short bounce that leaves the swing foot airborne or unloads the support foot?
- Does either mechanism reproduce both S4 failure classes, or only the trials where video shows support-foot sliding?
- Can R1r remain recoverable under the same plant draw while S4 fails? A candidate that makes both policies fail is not a stepping-specific explanation.

Use event ordering to separate the two elasticity families without contact sensors. If the rate divergence begins before the first foot impact, belt/actuator elasticity is the leading candidate. If the trajectories agree until impact and then show a velocity reversal, extra normal contact compliance or restitution is the leading candidate. If only the swing-foot landing changes while the torso rate is unchanged, the contact change is probably secondary. This is a virtual discriminator, not a claim that the NPZ identifies contact timing by itself.

Use the hardware phase (`gait_phase_init=0`) for all new twins. Compare processed hardware targets and motor-frame torque channels with sim only after applying the same action clipping, LPF, delay, and effort-limit semantics. Raw sim `action`/`target` and hardware processed command channels are not directly comparable. Do not use the later 45° fall time as a plant discriminator unless video confirms the tether remained inactive through that interval.

## Files and scripts used

- `CLAUDE.md` — current sim-to-real contract and channel-trust rules
- `control/balance_log.py` — hardware handover and NPZ logging contract
- `data/s4_flight_20260910/README.txt` — trial provenance and tooth-slip notes
- `data/_s4_flight_summary.py` — per-trial metrics
- `data/_s4_opening_compare.py`, `data/_s4_opening_dynamics.py` — opening comparison
- `data/_s4_sim_vs_hw.py` — twin versus hardware summary
- `data/_s4_knee_transient.py` — knee/hip transient inspection (used with torque-channel caveats)
- `data/_s4_topmass_verdict.py` — upper-body mass counterfactual
- `RL/logs/s4_sysid/20260910_2119/report.txt` — static-offset fit and held-out S4 prediction
- `outputs/s4_failure_20260911/phase_score.txt` — matched phase counterfactual
- `outputs/s4_failure_20260911/policy_replay.json` — exact hardware policy replay check

### Sweep implementation (built 2026-09-11, not yet run)

- `RL/run_s4_elasticity_sweep_GPU2.ps1` — launcher. Loops the CONTACT arms as
  separate Isaac launches (the foot material is a stage property created with
  the prim, so it cannot vary per env inside one process) and varies the BELT
  arm per env within each launch.
- `RL/scripts/rsl_rl/sweep_plant_elasticity.py` — driver. Pre-registers P1–P4
  in its module docstring, verifies every override by read-back before
  recording, writes `report.txt` / `cells.csv` / `rollout.npz` per contact arm.
- `RL/source/RL/RL/tasks/direct/standup_curriculum/belt_knee_actuator.py` —
  the series path. Two corrections to what the first draft implied: under
  elasticity the environment removes the reflected rotor armature from the
  PhysX joint (knee and the coupled hip-pitch diagonal) because the rotor
  inertia now lives inside the integrator, and the update is semi-implicit,
  not fully implicit (a fully implicit step is dissipative in proportion to
  `k·dt²/I`, so it would damp the stiff end of the sweep harder than the soft
  end and masquerade as the parameter under test).
- `RL/source/RL/RL/tasks/direct/standup_curriculum/foot_contact.py` — contact
  material spawner with a force-based (not acceleration-based) compliant
  spring, binding verified by read-back.
- `data/_test_belt_series.py`, `data/_test_elasticity_sweep_args.py` — the
  checks, including an independent RK4 reference for the integrator and a
  cross-file check that the driver's conversion constants equal the env's.
- `data/_test_elasticity_knobs.py` — numeric checks on the damping-ratio form
  and the play-loop knob ladders.

### The same knobs, live (added 2026-09-11 later)

The sweep pins a plant for a whole Isaac session, which is the wrong loop for
"does this feel like the robot". `play_sim2real.py` therefore carries the same
quantities on the arrow / PgUp / Home / Ins keys, and `RlEnvCfg` exposes them
as plain `env.<key>=` overrides, so a plant can be walked up to by feel and
then written down as the sweep's next arm. The keys are Up/Down belt stiffness,
Left/Right belt zeta, PgUp/PgDn foot stiffness, Home/End restitution, Ins/Del
foot zeta, and each change needs an `R` reset to take effect:

| quantity | cfg key | frame |
|---|---|---|
| belt stiffness | `belt_series_stiffness_min/max` | knee-joint N m/rad (compare to `servo_kp_knee`) |
| belt damping | `belt_series_damping_ratio` | ratio of the transmission mode, `2*zeta*sqrt(k*G^2*I_rotor)` |
| foot contact stiffness | `foot_contact_stiffness` | N/m, force-based PhysX mode |
| foot contact damping | `foot_contact_damping_ratio` | ratio of the ball's contact mode at `foot_contact_effective_mass` |
| normal restitution | `foot_restitution` | coefficient [0,1], independent of friction |

Damping is entered as a ratio because a raw N m s/rad number means a different
thing at every stiffness: the conversion lives in one place,
`belt_knee_actuator.damping_for_ratio`, and every caller (cfg, env DR draw,
sweep driver) goes through it. An explicit damping band still wins when set,
so no existing recipe is re-pointed by adding a ratio later. Belt changes apply
at the next reset; a foot-material change only fully applies in a fresh process,
because the material is created with the prim. Both facts are stated by the key
handler and by the `[PLANT]` lines that every env build now prints.

## Bottom line

The evidence rejects deployment inference, gait phase alone, tooth slip as a common cause, trim alone, and a simple missing upper-body mass as explanations. It places the common failure at the first dynamic stepping/contact interaction: the hardware develops sagittal backward momentum that the simulation does not. The next investigation is a virtual sweep of belt series elasticity and point-foot contact bounce, followed by DR training if those simulations reproduce the hardware feeling.
