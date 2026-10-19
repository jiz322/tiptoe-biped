# T0 / S8 hardware flight analysis — 2026-09-30

*Naming: the winning policy is referred to as **S8** throughout, per the user. Its
file, folder and npz keys all say **`s8b_8880`** (S8b is a resume of S8_vendorJ 7920,
itself a resume of S4a 7780) — S8 and S8b below are the same policy, not two arms.*

**Date:** 2026-09-30
**Data:** `data/t0_s8b_flight_20260930/` (4 trials, two policies)
**As-run deploy scripts:** the frozen `balance.py` / `balance_log.py` / `config.py` /
`policy.py` copies in that folder (mtimes 09-30 14:36)
**Sim side:** `RL/logs/rsl_rl/p9_replant/2026-09-19_08-14-51_S8b_air03/params/env.yaml`,
`.../2026-09-26_22-11-33_T0_straight/params/env.yaml`

## Question

Four flights, one afternoon, two raw training checkpoints. T0 (`model_1020`) fails
almost immediately in both trials. S8 (`s8b_8880`, `model_8880`) is the checkpoint
that steps on real hardware — in the user's judgement from the session video it is the
best-looking checkpoint, and currently the best one the project has — surviving past 8 s
in both trials, but it does not settle. Four claims to test, plus three conditions
that had to be checked before any of them could be read (T0's kill state, whether
restitution was live in training, and whether the floor or rope could explain the
survival) — the verdict table answers all seven:

1. T0 fails immediately, S8 steps — is that a policy property or a plant property?
2. Survival is real but "it can live longer if ideal" — what is actually consuming
   the 8 s, and can the analysis see the noisy floor and the safety rope?
3. Restitution "plays some role, bouncing/ringing observed in practice" — is the
   observed oscillation a contact-restitution effect, and was restitution even live
   in training?
4. The vendor armature is correct — does the data support that?

## Data and logging contract

All four trials are `mode=NN_CONTROL` for every logged tick: there is no HOLD or
OBSERVE segment in any npz, so the whole file is post-handover policy response and
no launch transient is hidden in it.

| item | value in all four trials |
|---|---|
| obs layout | 36 × 5 = 180 |
| control rate | 100 Hz (`dt` 9.5–11.4 ms, mean 10.06) |
| handover | `blind` (`v`) — first frame sensor block obs[0:27] zeroed, primed ×5 |
| `imu_pg_offset` | `[-0.015, 0.0]` (the 09-20 tile measurement, not the +0.089 compensator) |
| kp / kd | `[6, 20, 30, 6, 20, 30]` / `[0.3, 1.4, 1.5, 0.3, 1.4, 1.5]`, constant all run |
| effort soft-clip | `[6.0, 17.0, 23.2] × 2` (`BURST_EFFORT`) |
| `tau_max` (decode range) | `[6, 17, 17] × 2` motor frame |
| lpf_alpha | 0.60 (corner ≈ 4.4 Hz at 100 Hz) |
| stance / spawn | `straight`, `DEFAULT_JOINT_POS = [0, 0.05, 0, 0, -0.05, 0]`, init_h 0.4401 |
| action scales | `[0.30, 0.75, 1.025] × 2` |
| gait clock | `GAIT_FREQ_HZ = 2.5`, `GAIT_PHASE_OFFSET = 0`, logged one tick ahead of the frame |

Handover quality was good in all four: first-frame `pg_raw_x` within ±0.02 rad of
zero, i.e. the robot was physically upright at the moment the policy took over. So
none of these four failures is a bad handover.

Health of the log itself:

| trial | dur | replies | tick_busy p50/p95 | temp max |
|---|---|---|---|---|
| t0 144158 | 1.71 s | 6/6 all ticks | 2.68 / 2.98 ms | 35.3 °C |
| t0 144211 | 2.64 s | 6/6 all ticks | 2.71 / 3.01 ms | 35.3 °C |
| s8b 145106 | 8.42 s | 6/6 except 2 ticks | 2.72 / 2.99 ms | 37.3 °C |
| s8b 145128 | 8.72 s | 6/6 except 1 tick | 2.71 / 2.98 ms | 37.3 °C |

Transport and loop timing are clean. Three dropped-reply ticks in 2×8 s runs
(0.1–0.2 %, single ticks, no runs) are not a fall mechanism — but note that a
dropped reply means that tick's obs carried a stale/zero joint block, and the two
S8 drops sit mid-run at 0.97 s / 2.00 s / 3.63 s, not at the end.

**A video of the 09-30 session exists and the user has reviewed it; it was not
delivered to this analysis.** The user's read of it is the campaign's ground-truth
channel and is recorded in the Campaign state section below. Everything else in this
report is channel evidence and is labelled as such. Where the two could conflict, the
video reading wins; no such conflict was raised.

## The two plants are different

This matters before any policy comparison: T0 and S8 trained on different physics,
and the difference is exactly the armature the user wants to defend.

| | T0_straight (500 Hz) | S8b_air03 (2000 Hz) |
|---|---|---|
| `sim.dt` / `decimation` | 0.002 / 5 | 0.0005 / 20 |
| `hip_pitch_armature` | **0.0385 (pinned)** | −1.0 → derived **8.40e-3** |
| `knee_armature` | **0.036 (pinned)** | −1.0 → derived **7.81e-3** |
| `rotor_inertia_output` | 0.0042 | 0.0042 |
| `series_rotor_inertia` | 0.01936 | 0.0042 |
| belt series elasticity | **off** (band 0) | **on**, k 53.8, ζ 0, 4 substeps |
| `dr_armature_per_leg` | false | true |
| `dr_armature_mult` | 0.8–1.2 | 0.8–1.2 (PLACEHOLDER, S7 band) |
| foot restitution | 0.63, `combine: max`, DR 0.2–0.8 | same |
| `dr_knee_angle_bias_max` | 0.02 | **0.0** |
| `standup_ramp_mode` | path | path |
| curriculum metric | `success_high_rel` | `success_high_rel` |

So T0 is the heavy-inertia, rigid-belt plant (hip 4.6× S8's derived term); S8 is
the vendor-rotor, series-elastic plant. A T0-vs-S8 survival difference is therefore
**confounded with the plant shift, and with T0's kill state** — see below — and
cannot be read as a pure policy verdict.

## Trial outcomes

| file | policy | dur | end pitch | pitch range | any-action clip | rocking amp ratio (2nd ½ / 1st ½) |
|---|---|---|---|---|---|---|
| t0 144158 | t0_1020 | 1.71 s | +39.8° | −10.7 … +48.5° | 0.78 | 1.15 |
| t0 144211 | t0_1020 | 2.64 s | −44.5° | −50.9 … +2.9° | 0.58 | 1.37 |
| s8b 145106 | s8b_8880 | 8.42 s | −11.3° | −39.2 … +6.4° | 0.50 | 1.08 |
| s8b 145128 | s8b_8880 | 8.72 s | −16.3° | −32.7 … +0.2° | 0.47 | 1.14 |

Pitch here is the RAW projected-gravity x channel (`arcsin`, prior to any trim), so
it is the physical tilt of the torso, not what the policy saw.

### T0 — two opposite collapses, not one bias

The two T0 trials end at +39.8° and −44.5°. A one-signed trim, CoM or friction bias
cannot produce that, so these are two separate destabilizations of a policy already
at its authority limit. Supporting facts:

- **The hip pitch is commanding hard in both.** Mean |action| on hip_pitch is 0.78
  (trial 1) and 0.46–0.60 (trial 2) against 0.50 for every other joint; hip_pitch
  clip fraction is 0.556 in trial 1, the single highest in the session.
- **The whole-body command is at the rail.** 78 % of ticks have at least one action
  element pinned at ±1 in trial 1.
- **But the plant is not saturated.** `|tau_cmd|` p95 on the knee is 0.90 of the
  23.2 Nm joint budget and never exceeds it (max 27.5 Nm motor-frame, which is
  20.2 Nm joint — under 23.2). The hip_pitch commands max 17.2 against a 17.0 limit,
  i.e. touching, not exceeded. This is the documented normal case: the trained
  policies routinely demand more knee torque than sim delivers and survive the clip.

T0 is also a **policy the user killed on 09-28 at iteration 1040**, on the diagnosis
that its straight-knee stance is a near-singularity and that four foot-height reward
terms were one-sided ramps with live gradient at its own pose, producing a march the
recipe intended to pay 2 mm but that paid 29 mm. Its
`idle_swing_above_contact_frac` of 0.019 vs T1's 0.784 is the discriminator that
killed the line. Trial 1's failure is fast (last |pg_x| < 0.15 at 1.02 s) and
single-signed within itself, with wz reaching +2.79 rad/s in the final window — a
yaw-coupled pitch capsize. Verdict: **T0 fails immediately because it is a killed
line flown anyway, on the heavier plant, with its hip pitch already commanding at
the rail.** The data is consistent with that and adds no new mechanism.

### S8 — 8 s of a live limit cycle, not 8 s of balance

Both S8 trials survive past 8 s and neither settles:

- The `pg_x` rocking envelope does not decay. Splitting each run in half, the RMS of
  the de-trended pitch is 8 % and 14 % *larger* in the second half. This is the
  signature of a self-sustained oscillation (limit cycle), not a transient ringdown
  or a converging recovery.
- The pitch mean sits persistently negative (−16° to −30° raw) for the entire run.
  The robot is pitched forward the whole time and rocking around a forward-leaning
  operating point. This is the same qualitative state the 08-25 note called a
  "trained forward-lean idle" and retracted as a fall *mechanism* — it survives here
  as a description of the operating point, and it is where the run eventually drifts
  to failure.
- `wy` and `wx` peak-to-peak stay at 7–11 rad/s every single second, including the
  ones where `pg_x` looks calm (e.g. s8b 145106, second 3: `pgx_pp` 0.229 but
  `wy_pp` 7.87, `wx_pp` 7.75). The body is rotating hard in both lateral axes at all
  times; the pitch channel alone understates the violence.
- Both trials end with pitch drifting more negative (−0.196 → −0.280 at the last
  tick, having reached −0.63 mid-run in trial 1). Trial 1 recovered from that −0.63
  excursion once, which is the "it can step" evidence, and then fell later.

## The oscillation: 2.5 Hz plus its second harmonic

Spectral content of the body-rate channels is concentrated in two bands, and the
second is numerically twice the first:

| trial | channel | 2.5 Hz band | 5 Hz band | f₂/f₁ |
|---|---|---|---|---|
| s8b 145106 | `pg_x` | 2.49 Hz, a=0.055 | 4.98 Hz, a=0.029 | 2.000 |
| s8b 145106 | `wy` | 2.49 Hz, a=0.862 | 4.98 Hz, a=0.951 | 2.000 |
| s8b 145128 | `pg_x` | 2.63 Hz, a=0.044 | 4.92 Hz, a=0.023 | 1.870 |
| s8b 145128 | `wy` | 2.63 Hz, a=0.770 | 4.92 Hz, a=0.724 | 1.870 |
| t0 144158 | `wy` | 2.90 Hz, a=1.487 | 4.07 Hz, a=1.567 | 1.750 |

Two things follow, and they point the same way:

1. **`f₁` sits on the policy's own 2.5 Hz gait clock**, and the pitch autocorrelation
   peak lands at 0.392 s / 0.402 s = 2.55 / 2.48 Hz. When the pitch is projected
   onto the commanded gait clock and removed, only 8 % (trial 1) and 2 % (trial 2)
   of the band amplitude disappears — so the clock is not *driving* the rocking
   directly, but the body mode and the clock are the same frequency. The policy is
   marching at 2.5 Hz and the body is rocking at 2.5 Hz.
2. **The 5 Hz content is the second harmonic** (exactly 2.000 in the clean trial).
   A saturated, near-bang-bang command (see below) against a compliant transmission
   and a point contact generates harmonics of the drive frequency; a linear resonance
   would not be pinned to an exact multiple of the clock.

`wx` is the largest single axis (2.5 Hz amplitude 1.35 rad/s, larger than `wy`'s
0.86), so the rocking is substantially *lateral* — a roll component locked to the
gait, not only a fore-aft pitch pumping. That is the same signature the P13 note
recorded ("the one-signed roll is downstream, not a lateral CoM bias").

### The command is at the rail, so the plant is driven with a square-ish wave

| trial | any-action clip | hip_pitch clip | knee clip | hip_roll clip | effort-clip bind, knee |
|---|---|---|---|---|---|
| s8b 145106 | 0.50 | 0.16 | 0.16 | 0.08 | 2.9 % (filtered) |
| s8b 145128 | 0.47 | 0.14 | 0.16 | 0.06 | 1.7 % |
| t0 144158 | 0.78 | 0.44 | 0.28 | 0.12 | 13.5 % |
| t0 144211 | 0.58 | 0.15 | 0.22 | 0.16 | 7.6 % |

Half of all S8 ticks have some joint pinned at the action rail, so the effective
command is a saturated square-ish wave. The separate *effort* soft-clip
(`max_err = effort_limit/kp` = 0.773 rad knee, 0.850 rad hip_pitch) binds only 1.7–2.9 %
of ticks on S8 — so the knee is not a bang-bang relay, but the policy's own action
saturation is doing the limiting, not the torque budget. The lpf_alpha of 0.60 puts
the command filter corner at ≈4.4 Hz, which is *inside* both oscillation bands: the
command filter is not attenuating the ring, it is passing it and adding phase lag at
it. That is a plausible part of why the loop does not suppress the 2.5/5 Hz motion.

### Knee and hip tracking error is large, and it is not the hip coupling

| trial | hip_pitch L offset | knee L offset | knee demeaned RMS (vs target) | knee peak |
|---|---|---|---|---|
| s8b 145106 | −0.108 rad | +0.019 rad | 0.530 rad | 1.664 rad |
| s8b 145128 | −0.096 rad | +0.044 rad | 0.498 rad | 1.704 rad |
| t0 144158 | −0.086 rad | +0.001 rad | 0.664 rad | 1.767 rad |
| t0 144211 | −0.040 rad | −0.045 rad | 0.663 rad | 1.769 rad |

The knee error is not explained by the hip coupling term: `c·hip_pitch` has std
0.09–0.16 rad against knee error std 0.50–0.66, and the regression `err ≈ −1.65·(c·hp)`
leaves 0.508 of 0.530 rad RMS unexplained. So this is **not a decode artifact** of
the `−0.733·hip_pitch` transform. It is measured motor-side mis-tracking against the
commanded joint target.

**Caveat, per checklist item 9:** `sim_joint_pos` is motor-encoder-derived through
the 22:30 belt decode, so belt stretch/slip under load is invisible in this number.
A 0.5 rad RMS knee error therefore proves the *command* is not being realised at the
motor, and says nothing about whether the joint followed the motor.

## Restitution: was it live, and is it the ring?

**It was live, and the user is right that it is in the loop.** The foot material is
spawned with `restitution: 0.63`, `restitution_combine_mode: "max"` on the two ball
collision prims, against a `GroundPlaneCfg` whose own material is 1.0 friction /
0.0 restitution. `foot_contact.py` binds it explicitly per foot prim, and PhysX
`max` combine means the foot value wins over the ground's 0.0. In training the DR
band `dr_foot_restitution 0.2–0.8` (64 buckets, per-env via the shape buffer) was
active, so the policy saw restitution as a randomized plant property spanning the
plausible real-ball range.

Two honest qualifications:

- **The S8 yaml records `restitution_combine_mode: max` while today's
  `rl_env_cfg_p6.py` source writes `multiply` at the sim level with floor 0.0.** The
  foot prim's mode is the one that governs the ball-floor pair, and it is `max` in
  the trained run, so 0.63 was live. But the two files disagree textually and the
  discrepancy should be resolved before the next run rather than left to reading.
- **"Restitution plays a role" is supported as *participation*, not as *the cause of
  the 5 Hz ring*.** The ring's frequency is an exact multiple of the command clock,
  which is the signature of a driven nonlinearity, not of a passive contact
  resonance. A pure restitution bounce between point feet would not lock to 2.000×
  the policy's own gait frequency. Bouncing is consistent with the *presence* of
  5 Hz content and with the non-decaying envelope; it is not yet shown to be the
  dominant term.

The single decisive test, not run here: a `dr_foot_restitution`-collapsed sim twin
(`play_sim2real.py`) against the same handover state. If the sim ring at 5 Hz
disappears when restitution is pinned to 0, restitution is the driver; if the ring
persists, it is the servo/transmission loop and restitution is a passenger.

## The vendor armature claim

The user's directive on 09-18 was to treat the vendor figure as authoritative, and
this session's evidence is consistent with that, **but the two flights cannot test
it**:

- T0 flew the **heavy pinned** plant (hip 0.0385 / knee 0.036); S8 flew the
  **vendor-derived** plant (8.40e-3 / 7.81e-3). So the T0-vs-S8 survival contrast
  (1.7–2.6 s vs 8.4–8.7 s) does point in the direction of the vendor armature being
  the better-matched plant — **but it is confounded** with T0 being a killed line,
  with T0's 500 Hz rigid-belt plant, and with `dr_knee_angle_bias_max` being 0.02 for
  T0 and 0.0 for S8. It is a 2-trial-per-arm, one-afternoon comparison, and the
  video (user-reviewed, not delivered here) does not isolate armature from the other
  three differences either. It is suggestive, not a verdict.
- The prior evidence is genuinely split, and the split is on record: the 09-18
  kp-sweep argued the pinned hip plant should ring at 4.88 Hz while hardware measured
  3.06–3.34 Hz, which contradicted the vendor plant; but this session's S8
  (vendor) runs show a body-rate band at **4.92–4.98 Hz**, which is much closer to
  the vendor-plant prediction than to the 3.06–3.34 Hz figure. That is a new and
  directly relevant observation, and it is worth recording rather than resolving:
  the 09-30 hardware ring sits where the vendor plant predicts and not where the
  pinned plant does.
- The 09-18 pinned-inertia re-fit came back with all four hypotheses contradicted
  and an unaccounted excess of ~15e-3 at both RS01 output shafts, load-independent.
  Nothing in this session's data addresses that.

**Recommendation on the armature: hold the vendor number as the plant of record, as
the user already directed, and do not re-open it on this data.** The 4.9–5.0 Hz
observation is a *supporting* datum for the vendor plant, not a new measurement of
rotor inertia. The open item is unchanged: the ~15e-3 load-independent excess at the
output shafts is still unattributed.

## What the data cannot see

Stated explicitly so no reader over-reads the above:

- **No video in this analysis.** A 09-30 video exists and the user reviewed it, but
  it was not delivered here, so the "bouncing/ringing" and "it is stepping" readings
  are the user's eye against channel spectra, and the two have not been co-registered
  frame by frame. The user's readings are treated as ground truth regardless (see
  Campaign state).
- **No contact force, contact flag, or toe height channel.** The npz carries no foot
  state at all, so "the balls are loading" cannot be checked here. This is the same
  gap that made the 09-12 foot-DR request a wrong call, and the same line to read
  first (`lowest toe height +0.0856 m` vs contact line `0.084`) before any foot arm.
- **No tether/rope force.** Roughly 8 s of survival with a rope attached cannot be
  separated from free balance, and the user's own note says the rope is a noise
  source. The rope is a *candidate* for both the sustained rocking (a compliant
  constraint to ground) and the eventual drift.
- **`sim_joint_pos` is motor-side.** Belt slip/stretch invisible (checklist item 9).
- **The floor is not characterised.** Nothing here measures the 09-30 ground's
  friction or restitution. Note the P11 finding that the tile was slick enough that
  a yoga mat skated on it, and that floor friction was never measured.
- **N = 2 per policy.** Every per-arm statement above is a two-sample statement.

## Campaign state at 2026-09-30

**Two policies are live for hardware: `r1r_3150` and S8 (`s8b_8880`). S8 is the
better-looking of the two and currently the best one: it steps.** This is the user's
judgement, supported by their review of the 09-30 video (not delivered to this
analysis) and by the two 8.4 s / 8.7 s runs analysed here. The claim is about which
checkpoint looks and performs best on the floor today; it is not a claim that S8 is
the most converged or most robust checkpoint in the training record.

- `control/r1r_3150.pt` — the P13 robust-bundle deploy pick, `model_3150.pt` from
  `p9_replant/2026-08-27_10-30-37_R1_resume3110`; the policy behind the 09-03
  sustained-balance result (6.06 s at trim +0.089).
- `control/s8b_8880.pt` — S8b (`model_8880`), a resume of S8_vendorJ 7920, itself a
  resume of S4a 7780; the vendor-rotor plant with belt series elasticity on.
- Everything else in `control/*.pt` is a historical snapshot, not a candidate. In
  particular **T0 (`t0_1020`) is not live** — it was killed on 09-28 at iteration
  1040 and its 09-30 flights (this report) fail immediately, consistent with that.

User's on-floor observations from the 09-30 session, recorded as ground truth:

- T0 fails immediately (both trials).
- S8 steps, and is the best-looking checkpoint; currently the best one.
- Bouncing / ringing was observed in practice on the floor; restitution is believed
  to play a role in it.
- The real environment is noisy — objects on the ground, and the safety rope — and
  none of that is modelled.

Consequence for this report: the "S8 steps" verdict is not only a channel reading,
it is corroborated by eye. The limit-cycle characterisation above stands unchanged —
stepping and a non-decaying 2.5/5 Hz rocking envelope are both true at once, and the
gap between "steps" and "settles" is the thing worth closing.

## Verdicts

| question | verdict | discriminating signal |
|---|---|---|
| T0 fails immediately | **supported**, but as a killed line | two opposite-signed collapses (+39.8° / −44.5°), hip_pitch clip 0.556, flown on its 09-28 kill date |
| S8 steps on real hardware | **supported, and user-confirmed on video** | 8.42 s / 8.72 s, trial 1 recovered once from −0.63 raw pitch; user reports visible stepping and rates S8 the best-looking checkpoint, currently the best one |
| survival would improve "if ideal" | **supported as a limit cycle** | rocking RMS grows 8–14 % between run halves; no decay |
| restitution is in the loop | **supported** | foot prim 0.63 with `combine: max` vs floor 0.0; `dr_foot_restitution` 0.2–0.8 trained |
| restitution causes the ring | **inconclusive** | ring is at exactly 2.000× the command clock, which favours a driven nonlinearity over a passive contact resonance; needs the collapsed-restitution sim twin |
| vendor armature is correct | **not tested by these flights**; consistent | S8 (vendor) shows 4.92–4.98 Hz where the vendor plant predicts 4.88 Hz and the pinned plant predicted 3.06–3.34 Hz |
| the floor/rope explain the 8 s | **inconclusive** | neither is logged; the user's video review did not attribute the end to either |

## Recommended next steps

1. **Collapse-restitution sim twin** of the exact 09-30 handover state
   (`play_sim2real.py` at `--pg-offset -0.015`, blind handover) with
   `dr_foot_restitution` pinned to 0 and to 0.63, and compare the 2.5/5 Hz band
   amplitudes against the 0.055 / 0.029 measured here. This is the one test that
   decides claim 3, and it reuses existing infrastructure.
2. **Resolve the `restitution_combine_mode` discrepancy** between the S8 yaml
   (`max`) and the current cfg source (`multiply`) — a one-line diff, before it
   contaminates a future run's interpretation.
3. **Fix the `lpf_alpha` / ring interaction.** The command filter corner (≈4.4 Hz at
   α=0.60) sits inside the 2.5/5 Hz bands. An A/B at α = 0.85 (corner ≈2.6 Hz) is a
   cheap desk-side change to the existing contract, but it is a contract change and
   must match the trained plant.
4. **Fit a notch or lower kd at the ring frequency** only after (1) says whether the
   loop or the contact owns the mode. The P11 runbook already lists "oscillation /
   chatter / instability" as an immediate kill with a deploy-gain-mapping audit.
5. **Extend the floor notes beyond the one-liners now recorded.** The folder README
   carries the user's verdicts (T0 fails; S8 steps; bouncing observed; rope and floor
   noise not modelled) as of 09-30, but the *physical* detail is still missing and is
   what a future analysis will need: the floor surface and any mat, where the rope
   attached and whether it ever went taut, and which trial bounced visibly and when.
   Without that, the next session is in the same position as this one.
6. **Do not re-open the armature.** Record the 4.9–5.0 Hz observation as supporting
   the vendor plant and move on to the unattributed 15e-3 output-shaft excess when
   there is bench time.

## Artifacts

No new analysis scripts were written for this report; every number above is
reproducible from the four npz in `data/t0_s8b_flight_20260930/` with the as-run
`balance_log.py` schema, using the RAW `proj_grav_raw` channel for all tilt figures
and the logged `action` / `filtered_target` / `sim_joint_pos` / `motor_tau_cmd`
channels for the tracking and torque figures.
