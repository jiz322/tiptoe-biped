# big_thigh — Project Retrospective II, 2026-09-05 → 2026-10-19

**Status of this document:** follow-up to `docs/RETROSPECTIVE-2026-09.md` (which ends
at the first sustained free-standing balance, 09-03). This document covers how the
robot went from that single 6 s stand to **stepping, standing up from a crouch, and
sitting back down on command** (10-11), and what then broke. Written 2026-10-19 from
the campaign memory files (09-05 → 10-16), `docs/` campaign reports, per-folder
`data/*/README.txt`, the launch-script headers, `control/config.py`, and git (16
commits after `efdd12d`; the last is `c53e7aa`, 10-11 — most of the October work is
**uncommitted**, see §4). Facts that rest on the user's statement rather than on the
record are tagged **(user-attested)** and listed in §6.

Audience: same as part I — a locomotion engineer who has never seen this repo, and who
has read part I. Conventions are unchanged ("trial" = hardware, "rollout" = sim; dates
are 2026). One convention is now pinned that part I left implicit: **negative raw
`pg_x` = torso pitched backward** (anchored on video, 10-12).

**Public release note (2026-10-19).** Part I's note applies: this was written inside the
private development repo (`big_thigh`, the internal codename of TipToe), the cited commit
SHAs belong to that history, and paths are as they were there; the public layout is in
`docs/REPO_MAP.md`. Three differences from part I's note. (1) Most of the work described
here is uncommitted in the private repo, so this is its first committed record. (2) The
checkpoints, launchers and deploy scripts the campaigns name (`s8b_8880`, `s11_11880`,
`balance_s11.py`, `RL/launch_train_S1*`) are not shipped: the public repo releases one
checkpoint, `control/r1r_3150.pt`, and documents the rest. (3) `CLAUDE.md` and the
campaign memory files quoted in the text are the private working notes; `docs/REPO_MAP.md`
is their published equivalent.

---

## Summary

- **Sim2real is closed for the current task (user-attested, 10-19).** Per the user, two
  plant corrections closed it: **(1) the vendor rotor inertia is right** — the sim had
  been training a robot that believed its legs were ~4.6× heavier in rotor inertia than
  they are; **(2) the knee motor's belt reaction on the hip** (`plant_hip_reaction`)
  was flipped from `false` to `true`.
- **What the robot does (10-11, `s11_11880`, 5 trials):** 3/5 stand-ups from a d = 0.6
  crouch (3/3 in a fresh process); two of them stepped in place for 18.9 s and 43.6 s
  and **ended with a commanded sit-down**; ~66 s of stepping with one uncommanded fall,
  against 17 s and 2 falls for S8b on 09-30.
- **Two conclusions of part I are reversed.** Part I put "rotor inertia 4.2e-3 → 19e-3"
  on the list of things Claude Code got right, and it listed the user's 08-19 challenge
  to it as wrong. It was the other way round. Part I also said that the ideal J^T hip
  reaction was "tested and refuted". It is real, and a static-pose test showed it.
- **The +0.089 IMU trim is retired.** Every flight since 09-30 has used the physically
  measured trim −0.015 (09-20). Part I's open question ("what does +0.089 compensate?")
  is no longer a live deploy problem.
- **What blocks the robot now is hardware robustness, not the sim.** On 10-14 it went
  0/10 (loose screws and play in a hip_roll housing). On 10-16 the right knee belt
  failed (3 of 7 steel cords snapped), and a **new** belt broke the same way within 2
  trials. The knee drive is being redesigned (33:45 pulleys, 20 mm belt, no idler; CAD
  `big_thigh_v3.0_wider_belt_larger_pulley/`).

---

## 0. Corrections to part I

| Part I said | Status now | Evidence |
|---|---|---|
| §1 08-18, §2 #6, §3: "RS01 rotor ≈ 19e-3 at output, vendor 4.2e-3 is 3–4.6× low"; P7 armature hip 33.2e-3 / knee 36.0e-3; listed as a Claude Code win | **Reversed 09-18 (user directive).** Plant now uses the vendor 4.2e-3 → knee 7.81e-3, hip_pitch 8.40e-3, off-diagonal 5.73e-3 | Plausibility: a 19.3e-3 rotor needs ~75 % of the motor's mass at the housing radius. The bench signal (~15e-3 of τ/α-like excess at both RS01 shafts) is **real and still unexplained**: all four pinned-rotor refits (friction, torque scale, delay, belt) were contradicted on every trial (`docs/armature-derivation-note-20260918.md`). First policy trained on the vendor plant (S8) is the first that steps on hardware (§1 Phase G) |
| §3 "User hypotheses the record shows wrong: … the vendor 4.2e-3 might be right (08-19)" | **The user was right** | as above |
| §2 correction 3: "the ideal J^T hip-reaction term was tested and refuted (injecting it made the hip worse)" — the 08-06 teacher-forcing replay | **Reversed 10-03.** With the knee motor on the hip chassis, knee torque reacts on the hip as +c·τ_knee (c = 0.733) | The static HOLD crouch: q_hip = hold + (c·τ_k − 1.28)/20 predicts **all 6 flights × 2 legs within 0.014 rad with no fitted parameter**, including a flight with a different hold target (predicted 1.572, measured 1.572). A static pose has no velocity or filter lag, unlike the 08-06 one-step replay. In the training plant since S9 (10-04) |
| §4/§5: "trim +0.089 is the working value; leading candidate for what it compensates is knee-angle bias from belt slip" | **Retired.** Flights fly at the measured trim −0.015 | 09-20 tile measurement: raw `pg_x` −0.013 (IMU misalignment hypothesis contradicted). S8b flew 8.4 / 8.7 s at −0.015 on 09-30. The knee-bias candidate is inert in sim (pinned-knob sweep K1, 10-01). Whether the heavy-rotor plant is what +0.089 compensated on R1r was never tested directly |
| §0/§4: IMU "Yesense 6-axis" | **WHEELTEC H30** that speaks the Yesense frame format (BOM, 09-24) | `hardware/BOM.md` |
| §0/§4: 500 Hz physics | **2000 Hz** (`sim.dt` 1/2000, decimation 20) for every run with belt series elasticity | 09-17: the belt DR band is numerically unstable at 500 Hz above ~666 N m/rad |
| `docs/t0-s8b-…-20260930.md`, `docs/s8b-standup-…-20261003.md`: "negative pg_x = forward" | **Reversed label**: negative = backward | User video of 10-11 A2/B3 ("fell backward") ends at raw `pg_x` −0.93/−0.99. Those two docs' fore/aft words are probably backwards |

---

## 1. Timeline (from evidence)

### Phase E — S-line: DR holes, stand-up curriculum, path mode (09-05 → 09-10)

- **09-06** S1/S2 (`launch_train_S1_plantfix` / `S2_directive`), fork R1r_3150. These
  runs carried the part-I training directive. **Coherent whole-body CoM-x shift DR**
  (`dr_com_shift_x_max`) replaced the per-link knob, which only ever moved the
  whole-body CoM by 3–4 mm. Arm B added knee-angle-bias DR, handover priming and lean
  noise. A kneel→stand curriculum started. Two restarts followed: an optimizer runaway,
  and an advance criterion broken by grace inflation, in which kneel-branch envs count
  their grace steps toward episode length. The fix was a `success_high` metric over
  post-grace survival, plus `eval_alpha_success.py`.
- **09-08** Stand-up redesign (`377fca7`). Measurements showed the linear joint-space
  ramp topples at **every** d, including 0.5, with zero action. The d = 1.0 rest pose
  is a seiza, and the foot contact is a 2 cm ball 5.8 cm beyond the "toe" point the env
  had used. → **path mode**: a 7-column knot table measured from zero-action holds. A
  spawn-penetration bug in the knot table was found by the viewer and fixed the same
  night. Arm B's bundle cost ~0.15 success even in the easy band and was dropped.
- **09-09 → 09-10** S4a/S4b, with DR narrowed to measured bands. The only A/B was the
  pg-bias band. **09-10 flight** (`data/s4_flight_20260910/`): S4 followed its sim twin
  within 3° for 0.5 s, then the second step ratcheted it backward, and every trial fell
  within 1.5–2.5 s. R1r showed a plant balance point ~3 cm behind sim. The user's
  "unmodelled torso mass" hypothesis was tested in sim (+0.3 / 0.6 / 1.0 kg, with and
  without inertia recompute). It did **not** reproduce the hardware failure. A
  CMA-ES static-offset sysid also failed (CoM_z at bound).

### Phase F — the elasticity/contact detour (09-11 → 09-17)

- **09-11** The belt series-elasticity model and the foot-contact sweep were built.
  Five launcher/config bugs were caught at startup. One more would have silently
  reverted every env to the rigid plant.
- **09-14** S5_elasticDR **diverged** (value loss 1.5e16). The pad stiffness band was
  29× body weight. Foot restitution had never been wired.
- **09-15/16** The knee-encoder obs bug: knee-1's motor-side encoder was written into the
  **hip_pitch-1** observation slot, because the env joint order differs from the PhysX
  BFS order. It fired in every elastic training run and never in play. Found by
  dumping obs-vs-true per slot, after a value readback had looked fine.
- **09-16** The pad contact was found to be inert by construction: its reference plane
  sat below the rigid floor. **User abandoned the pad.** Restitution stayed as a
  PhysX material DR (0.2–0.8).
- **09-17** A 500 Hz physics trial was reverted the same day. The belt DR band needs
  2000 Hz, and all five rate constants have to move together.

### Phase G — the plant reversal and first stepping (09-18 → 09-30)

- **09-18** **Vendor rotor inertia restored (user directive).** No existing checkpoint
  had trained on this plant. S8_vendorJ (resumed from S4a 7780, belt DR, foot
  restitution DR) was launched. On 09-19, S8b_air03 = S8 + one gait-reward knob
  (`feet_air_height_threshold` 0.05 → 0.03).
- **09-20** IMU re-measured on a tile: raw −0.013 → trim −0.015. The IMU-misalignment
  and remassed-CoM explanations of +0.089 were both contradicted.
- **09-24** Open-source campaign started (public name **tiptoe-biped**; BOM ≈ ¥5360;
  a tracked Onshape API key found that must be rotated). This repository is that
  release.
- **09-26 → 09-28** T-line: low-CoM stance presets on the **old heavy-armature pins**
  at 500 Hz. The user killed T0/T1 on 09-28 (feet too high, roll shrug); the
  marching was paid for by one-sided reward ramps. U1/U2 were prepared and are moot now.
- **09-30 flight** (`data/t0_s8b_flight_20260930/`, trim −0.015 for all four):
  **S8b 8.42 s and 8.72 s**, stepping, one recovery from −0.63 raw pitch. T0
  (heavy-armature pins) fell at 1.71 s and 2.64 s. These are the first sustained
  stepping on hardware, and the first at the physically measured trim. The contrast
  is **confounded** (T0 also differs in physics rate, belt model and training state).
  "Steps" is not "settles": S8b holds a 2.5 Hz limit cycle.
- **10-01** Pinned-knob sweep of S8b (69 points × 32 seeds): 18 of 22 plant knobs are
  inert. The act-lag prefilter breaks it (a contract change), pg-bias ≥ 0.20 topples
  it, and stiffness < 0.85 saturates it.

### Phase H — stand-up, the hip reaction, sit (10-01 → 10-11)

- **10-02** After watching the sim stand-up videos, the **user dropped the seiza start**.
  The α 0.6/0.7 crouches are already stable, so any stable pose → dynamic balance is
  enough. The deploy stand-up stack was built: `balance_standup.py`, the
  `standup_path.py` numpy twin, and the desk check.
- **10-03 stand-up flight** (S8b, d = 0.6, `data/s8b_standup_flight_20261003/`):
  **0/6.** One foot slip; belt tooth skips in F2–F6 (knee command 22–27 Nm against the
  17 Nm clamp during the lift-off rock). The useful finding was the HOLD pose:
  hip 9° deeper than sim with the knee matching. That is the **belt reaction on the
  hip**, and `plant_hip_reaction` was built the same day.
- **10-04** S9a/S9b: **`plant_hip_reaction = true`** (hip PhysX effort cap raised
  22 → 44 Nm so the reaction is not clipped), α capped at 0.6, a stand-up speed A/B
  (2.5 s fixed vs a 1.67–5 s band), and a **sit branch that ends in the d = 0.6
  crouch** (user decision; obs ch29 = −0.6 while sitting).
- **10-07** S9 play review: 39 % of spawns started inside the floor. The rest rows were
  re-measured under the hip reaction (settles 0.13 rad deeper). → S10a/S10b
  (command-tracking ranges, rate vs heading yaw, F1 spawn lift, F2 rest preset).
- **10-10** S11_hist: S10b 11420 + zero-column surgery. Adds 0.5 s / 2.5 s conv history
  encoders trained only by auxiliary losses (v_est, contact, bias draws, an 80 ms
  prediction head). The Pi forward pass is 0.60 ms.
- **10-11** S11 deploy: `balance_s11.py`, `s11_contract.py`, TorchScript with an embedded
  contract and fixture (`c53e7aa`). The user added an **idler pulley** on the 22T knee
  pulley. **Flight** (`data/s11_flight_20261011/`): **3/5 stand-ups; A1 and B2 stepped
  18.9 s / 43.6 s and sat down on `s`.** Zero tooth skips. Offline replay matched to
  1.8e-6. Both failures came after a same-process re-arm (a 24 / 62 ms stall after the
  tick-0 send).
- **10-12** S12_steplen launched (touchdown step-length reward ×100, user's
  non-conservative call).

### Phase I — hardware robustness (10-14 → 10-19)

- **10-14** Re-arm stall root cause: a logger alias kept the previous run's buffer alive
  and freed it on the next tick 0 (39.9 → 1.3 ms; the GC hypothesis was contradicted).
  **Flight: 0/10**, every fall backward, with a sustained yaw after 0.5 s as the
  discriminator. Floor traction was ruled out by the user. **10-15 inspection: loose
  screws in many places and play at an R hip_roll housing seam.** The play sits after
  the encoders, which is why every npz fingerprint matched 10-11. R_roll was replaced
  with a new EL05 (re-addressed to 106, host 0x06A).
- **10-16 flight:** 2 trials, both fell right with R knee belt skips. A software
  overload was **contradicted**: the R load was inside the no-skip envelope and below
  the L load, which held. Disassembly found 3 of 7 steel cords snapped. A **new** R
  belt broke at the same place (22T next to the idler) in 2 more trials. Margin on both
  legs is ≈ 0: ~970 N pull at the 17 Nm clamp against ~780 N admissible for a 5M-10 belt.
  **User decision:** no load test; redesign to 33:45 (same 0.7333 ratio, so no
  constant changes), 20 mm HTD 5M 565, no idler.

---

## 2. What closed the gap, and what did not

| # | Change | Symptom it answered | Verification in the record | Class |
|---|---|---|---|---|
| 16 | **Vendor rotor inertia** (`ROTOR_INERTIA_OUTPUT` 19.3e-3 → 4.2e-3; S8, 09-18) | Policies trained on the heavy plant (K/R/S4, and T0 on pins) fell in 0.2–2.6 s; R1r needed a +0.12 trim past the physical value | 09-30: S8b steps 8.4/8.7 s at the physical trim; T0 (heavy pins) 1.7/2.6 s the same session. **No same-policy A/B** (a policy cannot be flown on two plants); the contrast is confounded. User-attested as one of the two closing changes | unmodeled dynamics (over-modeled) |
| 17 | **`plant_hip_reaction` false → true** (+c·τ_knee on the hip, post-clip; hip cap 44 Nm; S9, 10-04) | HOLD hip 9° deeper than sim; stand-up crouch off the trained manifold | Static: 12/12 legs ≤ 0.014 rad, no fitted parameter. Flight: S8b (off) 0/6 stand-ups on 10-03 → S11 (on) 3/5 on 10-11, but **six things changed in between** (hip reaction, sit branch, speed band, command tracking, history encoder, idler pulley), and the 10-12 analysis pre-registered the attribution as inconclusive. The sim dependence test (`run_play_S11 -NoHipReaction`) was prepared and not run. User-attested as the other closing change | unmodeled dynamics |
| 18 | Stand-up path mode + d = 0.6 crouch start | Linear ramp toppled at every d; seiza start unreachable | Zero-action validation of phase A; 10-11 stand-ups | training distribution |
| 19 | Sit branch (reverse path to the crouch, ch29 = −0.6) | No way to end a trial except falling | 10-11: A1 and B2 sat on command (B2 "less elegant" at grace 1.7 s) | training distribution |
| 20 | Deploy contract carried in the artifact (TorchScript + `contract.json` + fixture; startup refuses on mismatch) | Part I's contract bugs (#5, #11, #13) were each found after flights | 10-11 replay gate 1.4e-6 action / 1.8e-6 `sup` on all trials | initialization / contract |
| 21 | Re-arm stall fix (`lb = None`) | A2/B3 failed after a same-process re-arm | Pi dry replica 39.9 → 1.3 ms; 10-14 tick 0 at 3.9–5.1 ms in all trials | software |
| 22 | Knee idler pulley (10-11) → redesign (pending) | Tooth skips on every 10-03 stand-up | 10-11: 0 skips in 5 trials. 10-16: belt destroyed in 2 trials on R. **Not a fix**: it bought one session | hardware |

**What did not close.** The 2.5 Hz limit cycle persists: S11 cut the 2.5 Hz pitch-rate
amplitude 36–55 %, but energy moved to the 5 Hz harmonic (+45–65 %). The rocking
envelope is flat, and the action sits at the ±1 rail about half the time. hip_roll heats
at +26–37 °C per minute of stepping (~1–1.5 min budget). The knee belt has zero margin
at the motor clamp, and the policy hits the clamp on every stand-up.

### How each could have been found earlier (only where concrete)

- **#16** — the mass-at-radius plausibility check on the 19e-3 number (a rotor that
  heavy needs ~75 % of the motor's mass at the housing radius) was available on 08-18.
  So was the user's challenge on 08-19. A 4× disagreement with a datasheet needs a
  mechanism before it enters the plant.
- **#17** — the HOLD pose had been logged at the start of every trial since 08-31.
  Comparing it with the sim rest pose at the same target is a one-line check. The
  08-06 dynamic replay was the wrong instrument for a static torque term.
- **10-14 regression** — a post-flight mechanical checklist (screws, housing play, belt
  cords) before any data analysis. The play sat after the encoders, so the npz could
  not have shown it.

---

## 3. Who did what (this period)

Values as in part I: **user** / **Claude Code** / **joint** / **unknown**.

| Item | Noticed | Localized | Implemented | Verified |
|---|---|---|---|---|
| #16 vendor rotor inertia | **user** (08-19 challenge, 09-18 directive) | joint — user's physical ruling; Claude Code's plausibility arithmetic and the four contradicted refits | Claude Code | user (flight 09-30) |
| #17 hip reaction | Claude Code (10-03 HOLD-pose comparison) | Claude Code (zero-parameter c·τ_k prediction) | Claude Code (knob); **user** adopted it for all later training | joint (static 12/12); closure user-attested |
| #18 crouch start instead of seiza | **user** (10-02, sim video review) | — | Claude Code | user (flight) |
| #19 sit to the crouch | **user** (10-04) | — | Claude Code | user (flight) |
| #20 contract-in-artifact | Claude Code | Claude Code | Claude Code | joint |
| #21 re-arm stall | **user** (both re-arm trials failed: "bug in the re-arm path?") | Claude Code (logger alias; GC contradicted) | Claude Code | joint (Pi dry run) |
| #22 idler, belt failure, redesign | **user** | **user** (disassembly: snapped cords) | user | user |
| 10-14 loose screws / housing play | **user** (inspection) | user | user | pre-registered re-test superseded by the 10-16 belt failure |
| Encoder obs slot bug | Claude Code (its own 09-11 code) | Claude Code | Claude Code | Claude Code |
| Pad contact | Claude Code built it and found it inert | Claude Code | — | **user** abandoned it |
| Pitch-sign label | **user** (10-12 video) | Claude Code (convention) | — | — |

**Claude Code claims the record shows wrong (this period):** rotor inertia 19e-3
(carried over from part I until 09-18); the pad, which was built inverted, then got a
false-positive "pad is live" verdict, then a "balls load" reading taken from a diverged
run; the curriculum threshold 0.12 ("made it worse"); "belt tightening didn't help"
(10-14); floor traction as a 10-14 candidate; GC as the stall cause; "negative
`pg_x` = forward" in two docs; the 08-06 J^T refutation (from part I).

**User hypotheses the record shows wrong:** unmodelled torso mass as the 3 cm offset
(09-10, not reproduced in sim); motor friction change and bus voltage for the 10-14
regression (contradicted). **Right where Claude Code was wrong:** the vendor rotor
inertia; "the re-arm path has a bug"; "belt tightening helped"; "the R leg is a
hardware failure".

### Lessons (additions to part I)

1. **A bench fit is not a parameter until it has a mechanism.** The 19e-3 rotor fit four
   signatures and was still wrong. The excess it measured remains unexplained. Keep
   such a signal as an open residual. Do not push it into the plant.
2. **Static tests beat dynamic replays for plant terms.** The J^T hip reaction was
   refuted by a one-step dynamic replay and confirmed by a static hold with no fitted
   parameter.
3. **When the sim is right, the residual moves into the hardware.** After 10-11, every
   failure (loose screws, housing play, belt cords) was mechanical. None was visible in
   any motor-side channel. Inspect the hardware first after an unexplained regression.
4. **User scope cuts paid off:** pad abandoned (09-16), seiza dropped (10-02), the H-C0
   probe skipped (10-10), the load test deferred (10-16). Each removed a branch that
   Claude Code would have kept investigating.

---

## 4. Current state (2026-10-19)

**Hardware.** As part I, except: IMU = WHEELTEC H30 (Yesense protocol); R_roll = new
EL05 since 10-15; knee drive 22:30 HTD 5M-10 with the 10-11 idler. **The R knee belt is
broken, and the robot is not flyable until the redesign lands** (33:45, 20 mm, no idler,
same ratio; CAD in `CAD/big_thigh_v3.0_wider_belt_larger_pulley/`, in progress). The
user suspects the PLA thigh plate has deformed (amount unknown).

**Policy that stood and sat.** `S11_hist model_11880` → `control/s11_11880.ts.pt`
(TorchScript, embedded contract + fixture), lineage S8 → S8b → S9b → S10b → S11.
Plant: vendor rotor, `plant_hip_reaction=true` (hip cap 44 Nm), belt series DR
100–1000 N m/rad, foot restitution 0.2–0.8, mass ×1.00–1.14, 2000 Hz physics.
Training: S12_steplen (from S11 12330). Deploy: `balance_s11.py --alpha 0.6
--grace-seconds 2.5 --pg-offset -0.015`, blind `v` handover, `--heading off` for
stand-ups (heading hold feeds a large error during grace). `HIP_REACTION_FF = False`
on deploy, correctly: the plant now models the reaction, and the deploy must not cancel it.

**Known inconsistencies (documented, not fixed).**
- `CLAUDE.md` "Current state" still names `s8b_8880` as the live policy and says the
  stand-up deploy is unbuilt. Both are stale.
- The `control/config.py` `HIP_REACTION_FF` comment still cites the 08-06 replay as
  evidence that "the real hip does not feel the J^T reaction". The conclusion (keep
  False) still holds, but the reason is now wrong.
- `balance_standup.py` keeps the same logger alias that caused the re-arm stall
  (unfixed; not flown since 10-03).
- **Uncommitted:** the S11/S12 launchers, `hist_policy.py`, the export/surgery scripts,
  `balance_s11.py` / `s11_contract.py` edits (also scp'd to the Pi, so Pi ≠ `c53e7aa`),
  the env/cfg edits since 10-11, and the flight-data READMEs.

---

## 5. Next steps (short)

**P0 — knee drive redesign** (user's; decided 10-16). Before the first flight on it:
the crawl-pose knee check between trials stays mandatory. The 140 Hz pluck floor
still applies, but the old 0.048 kg/m tension conversion does not.

**P1 — re-establish the 10-11 result on the rebuilt robot.** S11 11880, same procedure,
n ≥ 3 fresh-process stand-ups with crawl checks. This also serves as the 10-15
pre-registered test that the belt failure pre-empted.

**Parked** (pointer = memory file):
- Limit cycle / rail time — `live-hardware-policies-20260930`, `s11-flight-20261011`.
- hip_roll thermal budget — `s11-flight-20261011`.
- Hip-reaction sim dependence test (`run_play_S11 -NoHipReaction`) — would turn #17's
  attribution from user-attested into measured.
- Open-source release — `opensource-campaign-20260924` (key rotation first).

---

## 6. Sources, and what could not be verified

**Read:** all memory files dated 09-05 → 10-16; `docs/s11-flight-analysis-20261012.md`
(§1–§6); the heads of `docs/flight-analysis-20261014.md` and
`docs/s8b-standup-flight-analysis-20261003.md`; `data/s11_flight_20261011/README.txt`;
`data/flight_20261016/README.txt`; the `RL/launch_train_S8b_air03.ps1` header; grep of
`docs/s9-*`, `docs/t0-s8b-*` and the S9b/S11 launchers for the plant overrides;
`control/config.py` (the knee/hip FF block, grace); `git log efdd12d..HEAD`.

**Not read:** the S9–S12 campaign docs in full, `docs/s4-*`, `docs/s5-*`,
`docs/armature-derivation-note-20260918.md` (its result is taken from memory), the
pinned-knob sweep output, any npz, tensorboard.

**User-attested (10-19), not established by the record:**
- **"Sim2real is closed."** The record shows that sim-trained behaviour transfers:
  stand-up, stepping and sit on 10-11 at the physical trim, and the static HOLD pose
  predicted within 0.014 rad. It contains no sim-vs-hardware trajectory comparison of
  S11 that scores the closure.
- **That the two closing changes are the vendor armature and the hip reaction.** Neither
  has a same-policy A/B. #16's flight evidence is confounded (T0 vs S8b). #17's flight
  attribution was pre-registered and came out inconclusive (six concurrent changes).
- **"The robot thought it was heavier."** This document reads it as the inflated rotor
  inertia (19.3e-3 vs 4.2e-3; hip_pitch armature 38.5e-3 vs 8.40e-3). The training
  mass band (×1.00–1.14) is centred on the measured 5.57 kg and is not the
  over-weight term.

**Could not verify:** the outcome of the 10-16 F3/F4 trials beyond the crawl readouts
(npz not copied); the PLA thigh-plate deformation; whether the snapped cords sit at one
edge or spread across the width.
