# P12 fall-isolation runbook — flight 2026-08-25

## DAY VERDICT — **CORRECTED 08-25 pm: the lean claim is RETRACTED**

User challenge ("why is a lean worse than level — both need precise
stepping") prompted the check the verdict skipped: full-kinematics CoM
(URDF masses + env-FK + traj joint_pos + pitch from pg) sits **−1.1 cm
± 0.5–1.2 from the stance toe** through the settled march. The 16° torso
pitch is a **pitched CROUCH (knees bent ±0.45 rad) with the CoM over the
feet** — a balanced posture, not perpetual falling. pg measures torso
orientation, not CoM-over-support.

What survives: sim spawn = the E1a entry and sim survives it 99.7%
(including the squat+pitch transition), hardware dies in the same window
→ the mechanism reverts to **execution of the trained dynamic maneuvers
on the real plant** (2× saturation, 5.6–17 Hz action content, belt
tooth-skip transients). **E3 (0.9× kp) regains top priority for the next
session.** Residue of the posture finding: the crouch loads the knees at
5.7 Nm settled / clip 14–23% — thinnest margin on the belt-driven joint
(a margin argument, not a fall mechanism). P13 direction (user's):
robustness via diversity — execution/torque-noise injection in training,
initial-posture DR, strengthened pushes; gate with a recovery-basin eval
(impulse sweep → recovery fraction). Original text below kept for the
reasoning trail.

## ~~DAY VERDICT (08-25 midday): mechanism isolated — the TRAINED FORWARD-LEAN IDLE~~

All trials ≈0 s free-standing regardless of policy (K4b/L1), entry
(chopstick/hand-lower), or surface (taped mat ≈ 08-24 tile). Desk
forensics then found the mechanism (full writeup: memory
`trained-forward-lean-idle`):

- **Every policy ever flown idles at a 15–23° dynamic forward lean**
  (F 22.2°, G2i 22.8°, K ~19°, L1 16.0° — `uprightness_mean` = cos(tilt),
  in every eval JSON since 08-23). Sim spawns upright and drives to
  pg_x ≈ −0.27 in 0.5 s, overshooting to −0.44.
- **Static balance is 2° from vertical** (env-FK toes x −0.100 + URDF CoM
  x −0.089 → contact→CoM +2.2°, balance pg_x ≈ +0.04). The lean is a
  LEARNED posture ~0.3 rad beyond static balance — perpetual falling,
  caught by steps at 2.5 Hz. Hardware dies within the first catch-step
  = the historical 0.14–0.26 s pitch-forward clock, policy/entry/surface
  independent.
- Sim discriminators (hw_matched pickup-free, 1024 envs): obs pg_x bias
  pinned +0.10 (measured hw trim residual) → survival 0.9971→0.9922
  only; CoM offset DR ±1/2/4 cm → unchanged 0.9971. **Sim absorbs every
  replicated hardware imperfection because its catch-steps always land**
  — the lean creates the exposure; real catch-step execution (contact
  texture, belt tooth-skip, 2× saturation) fails it.
- Bins: A (entry) eliminated; B (surface μ) weakened (mat=tile falls);
  D (L-line reward calm) eliminated (L1 reduced bounce, kept the lean —
  settled pg_x −0.264 vs K4b −0.277); C (loop gain, E3) untested and now
  second-order. NEW bin: the lean itself.
- **P13 lever: upright-idle retrain** — idle incentive targeting the
  static-balance posture (pg_x mean ≈ +0.04; steepen the upright reward
  — 19° currently costs ~5% of one cos term), keep the liftoff shuffle
  guard; gate adds settled/pg_x_mean ∈ +0.04±0.05 and uprightness ≥0.995.
  Plus: fix the +0.10 trim residual (free margin), and reinterpret every
  historical |pg_x|>0.25 "fall time" (threshold < trained lean 0.27).
- Also this day: right-knee belt skipped one tooth (12.0° = 360/30T)
  under catch-steps on the grippy mat — load-gated slip regime confirmed
  real; crawl-pose check between trials is now standing protocol.

Goal: **identify why the robot falls.** Not "make it stand" — make the fall
*assign blame*. After 08-24, contract / servo / obs / decode / drivetrain-lash
are all verified clean (see `p11-hardware-runbook.md` VERDICT + corrections),
so the residual sim2real gap lives in a short list. Tomorrow's job is to A/B
the list on hardware with one variable per cell.

Candidate bins (everything still standing after 08-24):

- **A. Entry / launch transient** — every 08-24 grounded trial had free-stand
  0.00–0.17 s measured from load onset; the put-down itself may be seeding the
  fall (T5 fell *backward* at the exact put-down instant; training lowers over
  0.8 s and releases on contact + 50–400 ms latency).
- **B. Floor contact (μ + compliance)** — satin tile slick enough that the
  08-24 mat skated on it; TPU-ball-on-tile μ unmeasured and (user call) not
  accurately measurable — the mat-vs-tile A/B and touchdown slow-mo replace
  the number. Training DR: floor μ 0.5–1.1, gate pins 0.8.
- **C. Policy-loop amplification (loop gain)** — hw runs 2× sim's action
  saturation with 5.6–17 Hz action content (sim: clean 2.5 Hz); hand-hold
  thrash at 4.8–6.9 Hz vs tether 1.4 Hz is the same loop closed through a
  stiff support. kp is the direct gain knob; stiffness DR trained ×0.85–1.15.
- **D. Gait heat (policy side)** — the L1/L2 arms attack this in training;
  they enter tomorrow only if they pass the pickup-free gate (0.2).

Per-trial discipline (non-negotiable, 08-24 lessons):

- `balance_log` every trial; **video every trial** (slow-mo 120/240 fps if
  the phone allows — aim at the FEET for touchdown slip).
- Metric = **grounded-time** (`_quicklook.py` now prints it: knee-load onset
  → first |pg_x|>0.25). Never quote log length. ≥3 trials per cell before
  reading anything.
- Hand-assisted trials carry **no survival signal** (balance reward is gated
  off while supported in training) — they are not a cell.

## 0. Desk (tonight / before rig)

- [x] **0.1 L-arm TB check — DONE 08-25.** L1 @4333 healthy: vz_sq 0.051
  (−33% vs parent-settled), idle_flight 0.0009, liftoff 0.047, toe_z
  0.0733, reward 15.7k stable. **L2 in slow-burn collapse from ~3850**
  (reward 16.4k@3773 → 4.3k@4381, vloss spikes 2e5, rawmax bursts 150,
  σ 0.52) — FIRST collapse under the patched 2e-4 lr cap; only changed
  variable = gait_ref 3→1 ⇒ the gait_ref prior at weight 3 appears
  load-bearing for optimization stability at this maturity. Kill/keep =
  user's call; eval used pre-collapse model_3770.
- [x] **0.2 Pickup-free gate — DONE 08-25 (`run_eval_p12gate_GPU2.ps1`,
  results `RL/logs/eval_p12gate/`). ALL THREE FAIL the pre-registered
  bar; L1 dominates K4b on every gate metric but misses two absolutes:**

  | metric (settled = last ⅓) | bar | K4b_3600 | L1_4330 | L2_3770 |
  |---|---|---|---|---|
  | survival 15 s | 1.000 | 0.9971 F | **1.0000 P** | 0.9980 F |
  | base_z std | <0.003 | 0.0057 F | 0.0037 F | 0.0042 F |
  | sat095 all | <0.04 | 0.0802 F | 0.0565 F | 0.0784 F |
  | pg_x std | ≤0.04 | 0.0361 P | 0.0395 P | 0.0429 F |
  | flight frac | <0.05 | 0.0000 P | 0.0000 P | 0.0000 P |

  Notes: (a) K4b's published survival 1.000 was pickup-ON; pickup-free
  it's 0.9971 — 3 ground-start envs die. (b) L1's calm was bought with
  ROLL work: settled hip-roll τ +15–23% (1.16/1.21 vs 0.94/1.05 Nm) and
  pg_y std 0.0525 vs 0.0431 — the lateral channel got HOTTER (matches
  the user's TB hr2 observation). (c) L2_3770 ≈ K4b everywhere → the
  gait_ref anneal bought nothing in eval AND destabilized training —
  arm answered, twice over. (d) settled pg_x MEAN ≈ −0.26..−0.28 for
  ALL policies — the frame/trim question (16°?) is policy-independent;
  resolve the convention before comparing sim pg thresholds to hw.
  ~~Decision per pre-registration: no PASS → K4b_3600 flies every cell~~
  **DECISION REVISED 08-25 (user call, documented override of the
  pre-registration):** the no-pass rule existed to keep a *worse* policy
  off the hardware; L1 dominates K4b on every gate metric, and the
  z-std bar it missed was derived from the contaminated 4-env npz
  (honest 1024-env K4b baseline 0.0057 — the 0.003 absolute was
  unmeetable even by the incumbent). **E4 is REVIVED: fly BOTH,
  K4b_3600 vs L1_4330, interleaved.** L1 also keeps training.
  Recalibrate future gate bars to "beat the K4b pickup-free baseline",
  not absolutes.
- [x] **0.3 Export — DONE 08-25.** K4b: `control/policy_K4b3600.pt`
  already on the pi (flew 08-24, parity 2e-6 in flight). L1:
  `control/policy_L1_4330.pt` exported (method B, normalizer found and
  baked, trace diff 0.0; parity vs checkpoint on 481 real hw obs ticks
  = 0.0 bit-exact). **On the pi, md5-verified (8b512090…).**
  Deploy contract is UNCHANGED vs K4b (same
  gains/scales/obs; hip_pitch_armature is sim-plant only) — no new
  bench gate needed. First L1 trial: run the `_launch_forensics.py`
  action-parity check on its log before trusting the rest of the cell.
- [ ] **0.2 Pickup-free gate eval, GPU2** (one scenario per process!).
  L1 latest, L2 latest if TB sane, **plus K4b_3600 pickup-free baseline**
  (its published settled numbers came from a pickup-contaminated npz's
  settled window — rerun clean for apples-to-apples):

  (Executed via `RL/run_eval_p12gate_GPU2.ps1` — supersedes the inline
  command draft; eval.py extended 08-25 with tau/qd/sat095/var/settled
  metric families, gate read directly from the JSONs.)

- [x] **0.4 Mat prep — DONE 08-25** (user: better mat, edges taped).
- [x] **0.5 CAN_TIMEOUT — CLOSED 08-25 on user confirmation** (armed;
  register write not independently verified — the 08-15 dumps predate it
  and no dump/commit records the write. Host code is hardened either way
  (3800bfd), so today's protocol is identical armed or not. If a future
  session needs certainty: hand-feel test = HOLD on tether → `kill -9`
  the balance process → joints limp ≈0.2 s later = armed / stay stiff =
  off; recover with `test_motors.py --no-zero` + Ctrl-C, whose cleanup
  disables all.)
- [ ] **0.6 Chopsticks dry-run** (power off): practice the release — two
  light contact points high on the torso, slide off *sideways* in ≤0.5 s,
  no push. The point is a minimal-wrench, instantly-removable support that
  imitates the trained release (contact-triggered + 50–400 ms latency).

## 1. Bring-up guard (unchanged ladder)

`ip -details link show can0` error-active, counters 0 → `sudo python3
test_motors.py --no-zero` → reply-ids 6/6 distinct → temps <45 °C.
Python for anything importing torch: `~/miniforge3/bin/python3`.

## 2. Day-calibration probes (~5 min, before any trial)

- [ ] **2.1 PD-hold stance probe** (`balance_log` 'l'), ~30 s **on mat** and
  ~30 s **on tile**: policy-free texture baseline per surface (08-24 tile
  baseline: qd at encoder noise floor 0.01–0.10 rms, flat spectra).
- [ ] **2.2 One hanging trial** (policy on, tether): drivetrain+policy
  baseline. 08-24 reference: calm 1.4 Hz pedal, qd rms 3.3, hip sat ~0.25.

## 3. Experiment ladder (≥3 trials/cell, video every trial)

- [ ] **E1 — entry isolation (bin A), on anchored mat, nominal gains.**
  Cell E1a: chopsticks entry (0.6 protocol). Cell E1b: trained-entry
  imitation — 'b' during pickup, lower over ~1 s, release at contact, no
  push. Reference to beat: 08-24 hand entries, best free-stand 0.17 s (T9).

  **E1a RESULT + INCIDENT 08-25 (~10:16):** 3 trials (101548/101612/
  101637), grounded-times **0.07 / 0.00 / 0.01 s** — chopstick entry did
  NOT extend survival (bin A weakened), fall directions now MIXED
  (bwd/fwd/bwd vs 08-24's uniform fwd). **RIGHT-KNEE BELT SKIPPED ONE
  TOOTH** (user observed sliding tooth after trial 3; crawl-pose check =
  ~12° offset = exactly 360°/30T). All motor-side telemetry is BLIND to
  a skip (encoder is upstream of the belt; tick-0 reads 0.000 in every
  log) — physical pose checks are the only tripwire. Trials are
  confounded (unknown which ran with the offset); E1b not run. Note the
  mechanism candidate: the GRIPPY mat removes foot-slip as the transient
  relief valve → catch-step impact loads go through the belt instead —
  fixing friction may have moved the overload to the drivetrain.
  **New standing protocol rule: crawl-pose knee-symmetry check between
  every trial** (test_position, expect L≈−R within ~3°); re-zero + belt
  inspection before any further trials.
- [ ] **E2 — surface A/B (bin B), chopsticks entry, nominal gains.**
  3 trials on bare tile with the identical E1a protocol. This is the
  friction/contact readout (replaces the impossible μ measurement). If you
  choose to stay mat-only for the day, the surface factor goes unread —
  then bin B evidence must come from touchdown slow-mo alone (foot skate
  visible at catch-step?).
- [ ] **E3 — loop-gain A/B (bin C), best surface from E1/E2.**
  Deploy at **0.9× kp, kd unchanged**: `REAL_KP` hip_roll 5.4/0.3,
  hip_pitch 18/1.4, knee **motor-side 14.5**/0.81 (= joint 27 × (22/30)²;
  already-converted convention — do not re-apply G²). Inside the trained
  stiffness DR (×0.85–1.15). Mark the config edit as a DELIBERATE contract
  deviation in the run notes; **revert after the cell**. Read: grounded-time
  + action sat + dominant-freq vs the matched nominal-gain cell.
- [ ] **E4 — policy A/B (bin D): REVIVED 08-25 (user override of the
  no-pass rule — see 0.2).** K4b_3600 vs **L1_4330**, identical
  best-condition protocol, **interleaved trials (K, L, K, L…)** so
  floor/battery/operator drift doesn't alias into the comparison.
  Sim priors for the readout: L1 is calmer vertically (z std 0.0037 vs
  0.0057, sat 0.057 vs 0.080, knee sat 0.124 vs 0.176) but hotter
  laterally (hip-roll τ +20%, pg_y std 0.053 vs 0.043) — so watch the
  ROLL axis on L1 trials specifically. Readings: L1 survives longer →
  gait heat was binding (L-line validated); L1 calmer in logs (sat,
  5–17 Hz content) but same fall clock → texture side binds; L1 falls
  in a NEW way (roll-flavored) → the lateral trade is real, feed back
  into L-line rewards.
- [ ] **E5 (optional, if a cell reaches ≥5 s)**: repeat that cell to confirm,
  then gentle disturbance tests (finger pushes at torso).

## 4. Pre-registered readings

| Result | Reading | Next move |
|---|---|---|
| Chopstick/trained entries ≥3× the 08-24 hand entries (same surface) | Launch transient was a dominant term | Entry protocol becomes standard; consider put-down curriculum in training only if it still caps duration |
| Mat ≫ tile (both chopstick-entry) | Contact binds (μ and/or compliance, not separable on hardware) | Anchored mat = standard rig; next arm widens `dr_floor_friction_min` below 0.5 + compliant-contact DR |
| 0.9× kp calms (sat ↓, 5–17 Hz content ↓, grounded-time ↑) | Deterministic amplification binds | P13: kp-anneal or kp-DR recenter arm + colored qd-noise injection (spectra from 2.1/2.2 probes) |
| L-winner visibly calmer on hw but same fall clock | Texture side binds (pre-registered 08-24) | Floor + noise-injection arms take priority over further reward shaping |
| Nothing moves the 0.14–0.26 s clock in ANY cell | The instability is in the policy×plant contact instant itself | Touchdown slow-mo forensics; contact-compliance modeling; torque-space perturbation injection in training (arXiv 2504.06585-style); re-check pg frame convention (settled sim pg_x mean −0.27 question) before trusting the 0.25 threshold |
| Any cell reaches ≥5 s | That variable was the binding constraint | Confirm by repetition, then disturbance tests; celebrate proportionally |

## 5. Safety & bookkeeping

- User runs all motion/power commands; tether just-slack at all times.
- One variable per cell — never change surface + gains + policy together.
- Name videos to match npz timestamps; scp the day's logs into
  `data/p12_flight_20260825/` and run `_quicklook.py` before interpreting
  anything out loud.
