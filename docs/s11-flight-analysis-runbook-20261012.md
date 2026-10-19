# S11 flight analysis — runbook (written 2026-10-12, to be executed in a later session)

**Data:** `data/s11_flight_20261011/` (5 npz, as-run `balance_s11.py` / `s11_contract.py` / `config.py`,
user notes in `README.txt`). Policy `control/s11_11880.ts.pt` = S11_hist model_11880, blind handover,
trim −0.015, alpha 0.6 (d=0.6 crouch stand-up). Campaign doc: `docs/s11-history-estimator-campaign-20261010.md`.

**Status:** nothing analysed yet. §2 lists the inventory checks run while this runbook was written; they are
facts, not verdicts. Every block in §4 states its hypotheses and verdict rules **before** any analysis runs.

| trial | file stamp | grace | dur | process | user outcome |
|---|---|---|---|---|---|
| A1 | 124840 | 2.5 s | 18.9 s | fresh | stood, stepped, sat on `s` |
| A2 | 125112 | 2.5 s | 2.5 s | re-arm after A1 (`r` → `k` → `v`) | stand-up failed; tether pulled |
| B1 | 131450 | 1.7 s | 10.0 s | fresh | stood (tether helped?), lost stepping "strangely" at 6.37 s |
| B2 | 132958 | 1.7 s | 43.6 s | fresh | stood (tether helped?), vx/yaw tried, sat |
| B3 | 133104 | 1.7 s | 1.7 s | re-arm after B2 | stand-up failed |

---

## 0. The user's questions and where each is answered

| # | user observation / question (10-12) | block |
|---|---|---|
| 1 | Stepping is much more stable than S8b; suspect the hip-reaction plant fix | §4.3 |
| 2 | 1.7 s stand-up looks worse than 2.5 s; stand-up may need optimizing | §4.4 |
| 3 | B1: at 6.37 s it stopped stepping and then fell. Why? | §4.2 |
| 4 | B2 commands "somewhat work", but there is no standard to judge them; maybe a higher control layer is needed | §4.8 |
| 5 | Should the S11 latent (`sup`) be used in the analysis? | §4.6 |
| 6 | Stand-up moves the robot backward; feels inevitable | §4.5 |
| 7 | The successful stand-ups looked fine | §4.4 (quantified, not just eyeballed) |
| README Q1 | Both failed stand-ups are the two same-process re-arms: a reset bug? | §4.1 |
| added | Does S11 drift less at idle on hardware (the hypothesis S11 was built for)? | §4.7 |
| added | hip_roll heating + idler-pulley friction | §4.9 |

---

## 1. Inputs and tools

**Per-tick npz channels** (all 5 files): `obs` (36), `action`, `filtered_action`, `action_center`, `target`,
`sim_joint_pos/vel`, `motor_tau_cmd/meas`, `motor_temp`, `proj_grav(_raw)`, `ang_vel`, `gait_phase`,
`grace_remaining`, `path_s`, `stabilize`, `sit_active`, `events`, **`sup` (19, physical units, labels in
`sup_labels`)**, **`hist_frame` (24, the raw ring frame)**, `ring_z` (max |z| of the frame against the hist
normalizer), `yaw`, `yaw_ref`, `vx_set`, `tick_busy_ms`, `can_replies`.
**Per-run:** `pre_*` (2 s HOLD pre-roll: gyro, raw pg, joint pos, yaw), `ring_first_tick`, `contract_json`,
`obs_norm_*`, `hist_norm_*`, spawn/path tables.

**Comparison sets:**
- `data/t0_s8b_flight_20260930/` — S8b standing-start flights (8.4 / 8.7 s); metrics in
  `docs/t0-s8b-flight-analysis-20260930.md` (2.5 Hz + 5 Hz spectrum, clip fraction, pitch range, rocking ratio).
  **Reuse those metric definitions exactly** so the numbers are comparable.
- `data/s8b_standup_flight_20261003/` — S8b d=0.6 stand-ups (0/6), scripts `_summary.py`, `_onset.py`,
  `_skipsig.py`; doc `docs/s8b-standup-flight-analysis-20261003.md` (HOLD-pose hip offset, lift-off rock,
  knee leash).

**Tools that exist:** `RL/run_play_S11.ps1` (`-Checkpoint 11880`, `-GraceS`, `-SettleS`, `-PgBiasX`,
`-Sit auto`, `-VideoSteps`), `RL/scripts/rsl_rl/eval_alpha_success.py` (fixed-d success; task id must be
`Template-standup_curriculum-hist-v0` and every S11 override must be re-passed), the TorchScript policy (runs
on any Windows CPU Python that has torch; otherwise use the env_isaaclab python).

**Tools that do not exist** (cost in §5): offline replay / predictor-skill script (campaign doc §8.4),
play-side `sup` logging (§8.5), a sim twin with npz state injection for the hist actor (`play_sim2real.py`
refuses `--inject_npz` with this actor), a root-displacement logger in the play scripts.

**Scripts go in** `data/s11_flight_20261011/_*.py`. Each one writes a summary file (json/txt) and figures under
`data/s11_flight_20261011/figs/`; the report reads the summaries, never raw stdout.

---

## 2. Facts found while writing this runbook (10-12, inventory only)

1. **t = 0 is identical in all 5 trials.** `ring_first_tick` is all zeros in every file. That is correct for a
   blind handover: all 24 ring channels come from obs[0:27], which is zeroed. The obs tail at tick 0 is the
   same everywhere (`[d 0.6, grace_norm 0.6, init_h 0.203, 0, 0, 0, 1]`), and so is the tick-0 action
   (max |Δa| = 0 between any pair). Actions diverge from tick 1 as sensors differ (≈0.5 by tick 5).
2. **Only the re-arm trials have a first-tick stall.** The first logged tick lands 23.9 ms (A2) and
   62.0 ms (B3) after the handover, against 3.9–6.1 ms in the three fresh runs (`tick_busy_ms[0]` 25.2 / 63.4
   vs 5.0–7.9). No other tick in any trial exceeds 9.1 ms. The stall size tracks the previous run's log
   length (A1 1883 ticks → 24 ms; B2 4335 ticks → 62 ms), which suggests Python GC of the previous log
   buffer (untested).
3. **Starting conditions are nearly identical.** Pre-roll joint pose is within 0.02 rad across trials. Raw
   pg_x at HOLD is −0.263…−0.287, except B3 at −0.246 (≈1° more upright). Motors are warmer at the start of
   the re-arm trials: knee 44.6 °C (A2) vs 40.6 (A1), and 45.3 (B3) vs 43.3 (B1/B2).
4. **The two failed stand-ups end in the same orientation.** Final raw pg_x is −0.93 (A2) and −0.99 (B3):
   same sign, torso near horizontal. B1 ends at +0.46, the other way. The direction name (fore/aft) has to be
   taken from video plus the IMU convention.
5. **The action rail is not lower than S8b's.** The fraction of ticks with any action at ±1 is A1 0.48,
   B1 0.65, B2 0.53 (S8b 09-30: 0.47–0.50), and A2 0.98, B3 0.96 in the failed stand-ups. Whatever improved,
   it was not a calmer command.
6. **B2 turned a lot, but the policy never received a yaw command.**
   - **What the policy saw:** all 5 trials ran `--heading off` (npz `heading_mode`; the README command
     lines pass no `--heading`). The logged policy input is ch33 = 0 and sin(yaw_err) = 0 on every tick of
     every trial. In off mode the as-run code refuses `a`/`d` (it only prints a refusal message), and they
     are not joint-group keys.
   - **What the robot did:** B2 rotated for real. Integrated gyro_z agrees with IMU yaw within ~10 %. Yaw
     reached −148° by 9.5 s (vx = 0), came back to −42° by 15.5 s, went to +65° at 29.5 s (vx 0.10), and
     ended near +18°.
   - **A1/B1 also yaw fast, but in both directions.** 1 s yaw-rate p95 is 29.5 (A1) / 59.2 (B1) /
     39.2 (B2) deg/s, max 56 / 98 / 72 deg/s, against the trained command cap of 0.4 rad/s = 23 deg/s. A1 and
     B1 net out within ±11° / ±26°; B2 alone has sustained one-way stretches.
   - **The user saw** B2 turning in response to `a`/`d` (10-12). The log and the user's account conflict;
     §4.8 resolves this with video.
   - **Other B2 commands:** vx was 0 until 18.4 s, then 0.05, then 0.10 from 23.9 s until the sit at 41.0 s.
7. **hip_roll heats fast.** In B2, hip_roll went 37 → 54/56 °C in 43.6 s. The knee and hip_pitch temperatures
   barely move (43.3 → 43.9). Check that before trusting either: does the firmware refresh those registers?
8. **`sup` is in physical units.** A1 standing 4–15 s: contact_L/R mean 0.65, v_x 0.023 ± 0.13 m/s,
   vs_x −0.028, pg_b_x −0.014, gyro_b ≈ 0. The pre-roll static gyro mean is 0.000 on all three axes.
9. **The HOLD hip sits deeper than the policy's centre.** Pre-roll hip_pitch is 1.418–1.434 against
   `spawn_target_jpos` 1.321 and the `hip_reaction_1007` rest pose 1.393. See §4.3c.

---

## 3. Ground truth needed from the user before analysis (video)

Frame-by-frame video is ground truth (CLAUDE.md). The trial LED switches ON at handover (npz t = 0), so convert
every video time to npz time with that edge. Needed, per trial:

| item | trials | used by |
|---|---|---|
| LED-on frame (t = 0 anchor) | all | everything |
| Is "6.37 s" video time or npz time? | B1 | §4.2 |
| Frames where the tether went taut, and roughly which way it pulled | A2, B1, B2, (A1, B3: none?) | §4.1, §4.4 |
| Foot touchdown / lift-off frames for ~6 strides (one steady window, plus the window before 6.37 s) | B1, B2 | §4.2, §4.6 |
| Floor displacement, fore/aft + lateral (Kinovea, a floor tape mark): stand-up start → grace end → each 5 s | A1, B1, B2 | §4.5, §4.7, §4.8 |
| Fall direction and first visible event (foot slip / knee buckle / torso pitch) | A2, B1, B3 | §4.1, §4.2 |
| Heading change over the stand-still windows (deg, from overhead or floor lines) | A1, B2 | §4.7, §4.8 |
| Fill the README TODO lines | all | — |

Without the displacement and contact items, §4.5–4.8 drop to "inconclusive: no independent check" for every
claim that rests on `sup`.

---

## 4. Analysis blocks (in execution order)

### 4.0 Gate: log integrity and bit-exact replay — do first

**Claim:** each tick's logged `action` and `sup` can be reproduced on Windows from the logged `obs` stack and
`hist_frame` (rebuilt into a 250×24 ring with the same prime-on-first-frame rule) through `s11_11880.ts.pt`.
- **Pass:** max |Δaction| ≤ 1e-5 and max |Δsup| ≤ 1e-5 on all 5 trials. Pi vs x86 float differences are
  allowed up to 1e-4; report the actual number.
- **Fail:** stop. Either the logging does not match what the policy saw, or the ring rebuild differs from
  `S.HistRing`. Every later block depends on this.
- Also check: `can_replies` 5 on any tick (B3 has one), Δt between ticks (9.5–11.5 ms expected), and the
  sit-complete tick in A1/B2.
- Byproduct: a replay harness that §4.6b extends.

### 4.1 Why the two re-arm trials failed to stand (README Q1)

Pre-registered hypotheses:

| id | hypothesis | supported if | contradicted if |
|---|---|---|---|
| R1 | deploy state not reset in `balance_s11.py` / `s11_contract.py` | an obs/ring/tail/filter value at t = 0 differs between fresh and re-armed trials | t = 0 identical (§2.1 already shows this for obs, ring, tail and action) → **expected contradicted**; confirm by diffing the first 3 ticks of every logged state channel, and by reading `to_hold` / `start_run` against the training `_reset_idx` |
| R1b | state outside the script persists: `motor_bridge` filters, IMU driver, CAN | a channel the script does not own is offset at t = 0 in re-arm trials (e.g. `sim_joint_vel` non-zero while static, a knee coupling offset, a `pg_raw` jump) | those channels look the same as in the fresh starts |
| R2 | the first-tick stall (24 / 62 ms, §2.2) sends the lift-off off-script | the stall overlaps the start of motion and the re-arm trajectories diverge from the fresh ones in the first ~0.2 s beyond the fresh-to-fresh spread | the divergence starts later than ~0.2 s, or stays inside the A1/B1/B2 spread until the fall phase |
| R3 | thermal: warmer motors (≈+4 °C knee) change friction or torque | knee tracking error / torque per unit error during lift-off is outside the fresh spread in A2/B3 | inside the spread |
| R4 | handling: tether pull (A2 reported) or reduced help (B3) | the video shows a taut tether at the onset in A2 (and/or the B1/B2 "help" coincides with their critical window) | no tether event in the window |
| R5 | chance | — | — |

R5 note: with p(success) ≈ 0.6–0.8 per stand-up, two failures in five trials is unremarkable. If the failures
were exchangeable, the probability that they land exactly on the two re-arms is 1/C(5,2) = 0.10. This is a
weak coincidence and must not be read as proof of a bug.

**Method:** overlay the 5 stand-ups on `path_s` time (ticks since handover): pg_x/pitch, gyro_y, knee/hip
tracking error (`sim_joint_pos − target`), knee motor torque vs the 17 Nm clamp, action rail flags,
`grace_remaining`. Mark the stall, the tether frames, and the divergence time (first tick where a re-armed
trace leaves the fresh envelope by > 2× the fresh spread).
**Verdict:** name the earliest discriminating signal. A1/B1/B2 are the only controls; if the help in B1/B2 is
confirmed, they are contaminated controls, and the report says so.
**Cheap fix regardless of verdict:** `gc.collect()` inside `to_hold()` (or `gc.freeze()` after start-up), so
the next flight removes R2. ~5 lines; the user decides.

### 4.2 B1: lost stepping at 6.37 s, then fell

First pin down the clock (§3). Then cut the window [t−2, t+1] around the event and test which signal **leads**:

| id | hypothesis | discriminating signal (must lead the stop) |
|---|---|---|
| B-a | foot slip / contact loss | `sup` contact_L/R alternation breaks from `gait_phase` before the stop; the video shows the foot sliding |
| B-b | saturation | hip_roll at its 6 Nm clip or knee motor at 17 Nm for ≥ 3 consecutive ticks; action rail run-length rises |
| B-c | velocity run-away | `sup` v_x / vs_x ramp outside the training command band (−0.15…0.30) before the stop; pitch drifts monotonically |
| B-d | lateral | roll / gyro_x amplitude growth; one-sided hip_roll |
| B-e | out-of-distribution input | `ring_z` \|z\| > 4 sustained before the stop (compare A1/B2 steady windows for the baseline) |
| B-f | external | tether or contact with the user in the video |

The 2.5 Hz gait clock never stops: "stopped stepping" means the feet stopped following the clock. Quantify it
as the phase lag between `gait_phase` and the contact estimate, plus the video touchdowns.
**Verdict:** the first signal to leave its A1/B2 steady-state band. If two lead together, say "co-leading".
If none leads, say "inconclusive".

### 4.3 Stepping stability vs S8b (user #1)

**(a) Measure the improvement.** Use the same metrics as the 09-30 doc on matched windows:
- S11: steady stepping at vx = 0 after `cmd_active` (A1 ≈ 2–15.9 s, B2 ≈ 1.5–18.4 s, B1 ≈ 1.5–6 s);
- S8b: the full 09-30 trials.

Metrics: survival; pitch range; 2.5 Hz / 5 Hz band amplitude of gyro_y and pg_x; f₂/f₁; rocking amplitude
ratio (2nd half / 1st half); action rail fraction (§2.5 already shows it is not lower); knee effort-clip bind %;
stride regularity (cycle-to-cycle CV of the gyro_y period).
- **Supported "more stable"** if the 2.5 Hz amplitude ≥ 30 % lower and the rocking ratio ≤ 1.0 (not growing).
- **Contradicted** if both are within ±15 % of S8b, i.e. the gain is survival only.

**(b) Attribution: what flight data can and cannot say.** S8b → S11 changed at least: hip_reaction plant (S9),
α pinned 0.6 + grace band + sit branch (S9), commands + tracking reward + F1 lift + F2 rest preset (S10),
history encoder + pg-bias DR 0.05 (S11), **and the hardware itself (idler pulley: no tooth skip; S8b
flights had skips)**. Flight data cannot separate these. Two possible tests:
- **sim, cheap:** play 11880 with `env.plant_hip_reaction=false` and everything else re-passed. If stepping
  degrades, the policy relies on the reaction model. This shows dependence, not that training with it caused
  the stability.
- **hardware, expensive:** fly an S10b checkpoint. That needs a non-hist deploy path (S10 §8, unbuilt) and
  isolates only the history encoder anyway.

The report states the attribution as **inconclusive from flight data** unless one of these runs.

**(c) Plant-match evidence for the hip reaction (cheap, supports #1 indirectly).** The 10-03 finding: hardware
HOLD hip was 9° deeper than the sim without the reaction, and the reaction predicted it (≤ 0.014 rad). Compare
the S11 pre-roll HOLD hip (1.418–1.434) to the sim PD-hold settle of the same target under hip_reaction on
(`run_play_S11.ps1 -SettleS 3`, read the settled joint pos).
- **Supported** if |Δhip| ≤ 0.014 rad.
- **Contradicted** if > 0.03 rad.

### 4.4 Stand-up speed: 2.5 s vs 1.7 s (user #2, #7)

Five trials with confounded help cannot rank two speeds. Hardware supplies **mechanism metrics**; sim supplies
**rates**.

**Hardware metrics, per stand-up** (A1 vs B1/B2; A2/B3 as failure references): peak and duration of knee motor
torque at the 17 Nm clamp during lift-off; leash usage `|filtered_action × scale|` / scale on knee and hip; path
tracking error; pitch excursion over grace + stabilize; time from grace expiry to the first 3 regular alternations;
backward travel (§4.5). Hypothesis **G1**: the 1.7 s stand-up loads the knee clamp more (higher peak, longer
clamp time) than the 2.5 s one.
- **Supported** if B1 and B2 both exceed A1 on clamp duration by > 50 %.
- **Contradicted** if they are within ±20 %.

**Sim rate curve (prepare, the user runs):** `eval_alpha_success.py` on the hist task, model_11880, d = 0.6,
grace fixed at 1.667 / 2.0 / 2.5 / 3.0 / 4.0 / 5.0 (`standup_grace_seconds_min = max = 0`, the S11 overrides
re-passed), 1024 envs. This needs a check that the script handles the grace band and the hist actor; edit it
if not (est. 30 min). Hypothesis **G2**: success at 1.667 (the band edge) is ≥ 0.05 below the 2.5–3.0 plateau.
**Decision this feeds:** the next flight's `--grace-seconds`. "Optimizing the stand-up trajectory" (knot
table, retraining) is a separate campaign. Propose it only if G1 is supported **and** sim success is flat
across speeds, which would mean the hardware alone is the bottleneck.

### 4.5 Backward travel during the stand-up (user #6)

| id | hypothesis | supported if |
|---|---|---|
| BW1 | it is the trained strategy (sim does it too) | sim root x displacement over grace + stabilize has the same sign and is within ±50 % of the hardware value |
| BW2 | plant gap (lift-off rock / belt) | hardware travel > 2× sim, or opposite sign |
| — | neither | sim travel ≈ 0 and hardware travel is small (< 3 cm): not a real effect |

- **Hardware:** Kinovea floor displacement (§3); cross-check with ∫`sup` v_x, trusted only after §4.6a.
- **Sim:** play 11880 at d = 0.6, grace 2.5 and 1.7, log root position. This needs a small logger in the play
  script (est. 30–45 min) or a `-Smoke`-style print.

If BW1 is supported, the travel is "inevitable" only under the current reward. A grace-phase position term
would be the knob; that is a training decision, not part of this analysis.

### 4.6 Using the S11 latent on real data (user #5)

**Yes, use it, but only after validation.** `sup` is a sim-trained estimator and an untested channel on
hardware (deploy checklist item 9). Validate each group against an independent reference first; then use only
the groups that pass.

**(a) Validation** (pre-registered bars):

| `sup` group | independent reference | pass bar |
|---|---|---|
| gyro_b (3) | pre-roll static gyro mean (`pre_gyro`, 2 s, ≈ 0.000) | \|gyro_b\| ≤ 0.02 rad/s steady (sim R² context: TB `r2_gyro_bias`) |
| pg_b (3) | known residual: true raw bias −0.013, trim −0.015 → residual ≈ +0.002 on x | \|pg_b_x − 0.002\| ≤ 0.02. If it reads consistently ≈ −0.014 (§2.8), the estimator is seeing something; that links to the 09-20 open joint-zero branch, but **only** if jb (below) agrees |
| jb (6) | crawl-pose readout + `setzero_crawl`; L/R mirror symmetry | report only (expected weakly observable, campaign P2) |
| contact_L/R | video touchdown / lift-off frames (§3) | ≥ 80 % of video stance frames have p > 0.5, and the swing frames p < 0.5 |
| v_x, vs_x | video displacement per window (§3) | sign right in every window, and \|mean error\| ≤ 0.05 m/s over 5 s windows |

**(b) Offline predictor skill (campaign §8.4; the diagnostic the self-supervised head exists for).** Extend the
§4.0 harness: load the predictor weights from `model_11880.pt` (training-only, absent from the TorchScript),
run encoders + predictor over each trial's logged ring + actions, and compute
skill = 1 − MSE / MSE_persistence per channel group (pg, gyro, jpos, jvel), per bin (20–80 ms), and per gait
phase bin.
- Reference: TB `pred_skill_*` at 11880.
- **Reading:** the channels and phases where real skill falls well below sim skill are where the hardware
  dynamics differ from sim, located in time and joint. This is the most direct sim2real-gap map the project
  has had.
- Hypothesis **P-hw:** real skill ≥ 0.5× sim skill on pg and gyro at bin 3 during steady stepping.
  **Contradicted** if < 0.25×; then name the worst channel and phase.

Cost ≈ half a day, CPU only (§5).

**(c) Uses once validated:** v_est at the B1 event (§4.2 B-c), stand-up travel (§4.5), idle drift (§4.7), vx
tracking (§4.8), contact rhythm as the "stepping quality" metric (§4.3).

### 4.7 Idle drift on hardware: S11's own hypothesis (added)

S11 was built to cut idle drift (campaign P5, sim `cmd/pos_drift_idle_m`, S10b 0.271). The flights contain vx = 0
standing windows: A1 ≈ 14 s, B2 ≈ 17 s, B1 ≈ 4.5 s.
- **Measure:** video displacement and heading change per 10 s at vx = 0 (heading off, so heading drift is
  uncorrected by design).
- **Pre-registered:** "S11 holds position at idle on hardware" is **supported** if drift ≤ 0.15 m per 10 s
  (≈ the sim number scaled to 10 s; replace with the TB value for 11880 before reading the data). It is
  **contradicted** if > 0.4 m per 10 s.
- S8b has no matched baseline (it never stood long enough), so this establishes the hardware number;
  it does not support a comparison.

### 4.8 Command tracking: a standard to judge by (user #4)

**Fact first (§2.6):** the policy's yaw input was zero all flight. Only vx reached the policy: 0.05 for 5.5 s,
then 0.10 for 17 s. B2 nevertheless turned up to ~150° away and back, and the user saw it respond to `a`/`d`.
Resolve this before judging yaw tracking.

| id | hypothesis for B2's sustained turns | supported if | contradicted if |
|---|---|---|---|
| Y1 | `a`/`d` did reach the policy by a path the log misses | the as-run code or the Pi shell history shows `--heading hold` for B2, or the turn onsets line up with key presses within ~0.3 s every time | the B2 command line had no `--heading` (README) and onsets do not follow key presses |
| Y2 | external torque: tether / handling (the user reported helping in B2) | the tether is taut or the user is touching the robot at the onset of each sustained turn | the rope stays slack through the −148° excursion |
| Y3 | the policy wanders on its own; B2 is longer, so it accumulates more | similar one-way stretches appear in A1/B1 windows of equal length; the yaw rate stays bounded in the same band (p95 30–60 deg/s) | B2's sustained rates sit clearly outside the A1/B1 distribution over matched window lengths |
| Y4 | the turns come from foot pivot / slip | the video shows the stance foot spinning on the floor (ball foot = a point contact with little yaw friction) during the turn | the feet are planted while the body yaws |

**Discriminators:**
- **Video:** turn onsets vs key presses (ask the user for the key-press times or the screen recording of the
  display; refusals show as a red line); the tether state; foot pivot.
- **Data:** mean gyro_z per 2 s window in B2 vs A1/B1 windows of the same length; whether B2's turns correlate
  with hip_roll asymmetry (L − R target), which would mean the policy is generating them.

**Note:** fast two-way yaw in every trial, beyond the trained 23 deg/s cap, is itself a finding. B1's 98 deg/s
peak is at 7.4 s, right after the 6.37 s event: add it to §4.2 (B-d/B-a).

**Analysis of B2:** video speed in the 0 / 0.05 / 0.10 windows vs command.
- **Sim reference:** `cmd/vx_err_moving` 0.083 m/s; vx_cmd_min_abs = 0.05, so 0.05 sits at the edge of the
  trained band.
- **Pre-registered:** vx "works" if the mean speed is ordered 0 < 0.05 < 0.10 with the correct sign and
  |error at 0.10| ≤ 0.08 m/s. If the speed difference between 0 and 0.10 is within the idle drift (§4.7), it
  is **inconclusive**.
- Also report the uncommanded yaw rate in the same windows (does vx induce turning?).

**Proposed standard for the next flight** (written into the next flight plan, not built here):
- vx ladder 0 → 0.1 → 0.2 → 0 → −0.1, 8 s each, floor tape every 0.25 m, overhead or side camera.
- Yaw: run the display YAW CHECK first (twist the limp robot; the sign must read SAME), then `--heading hold`,
  steps of ±15° (`a`/`d`). Metric: heading error at 3 s after each step, and its overshoot.
- Pass bars mirror the sim metrics: vx error ≤ 1.5× sim `vx_err_moving`; heading error ≤ 1.5× sim
  `heading_err_abs_mean`.

**"Another control layer?" — difficulty prediction:** a joystick/keyboard → (vx, heading target) mapping with
rate limits and a heading integrator is standard practice (every legged-robot RL stack ships one) and costs
< 1 h inside `balance_s11.py`. It needs no retraining. A *learned* high-level layer is not needed unless the
low level fails the bars above. Decide after the next flight's ladder.

### 4.9 hip_roll thermal budget and idler friction (added)

- **hip_roll heating** (§2.7: +17–19 °C in 44 s): time fraction at the 6 Nm clip; mean |τ| in stance vs swing;
  the heating rate per trial vs flight length. Extrapolate the flight time until a chosen limit (ask the user
  for the motor's derating temperature from the vendor GUI). First establish whether the knee/hip_pitch
  temperature channels update at all.
- **Idler friction:** compare knee command → tracking at the matched HOLD pose and during lift-off against the
  10-03 S8b stand-ups (same d, same gains, old belt). Hypothesis **F1**: the idler adds knee friction, i.e. a
  larger steady knee tracking error at HOLD for the same commanded torque.
  - **Supported** if the steady error rises > 30 %.
  - **Contradicted** if within ±15 %.
  - Pre-roll has no torque channel: use the first 0.2 s after handover (still at HOLD under the path centre),
    or the S8b-style HOLD segments if present.

---

## 5. Build list with cost (the user decides what to build)

| item | needed by | common / invented | cheaper workaround | est. cost |
|---|---|---|---|---|
| bit-exact replay harness (ring rebuild + TS forward) | 4.0, 4.6b | common (log replay) | none, it is the gate | 1 h |
| predictor-skill replay (weights from .pt) | 4.6b | invented for this campaign (the head exists) | skip; validate `sup` only | 3–4 h |
| overlay/metric scripts (stand-up envelope, S8b metrics) | 4.1–4.4, 4.9 | common; 09-30 / 10-03 scripts reusable | — | 2–3 h |
| `eval_alpha_success` grace sweep on hist task | 4.4 | common | eyeball `run_play_S11 -GraceS` (no rate) | 30 min edit + user GPU run |
| play-side root displacement logger | 4.5 | common | print root pos at grace end | 30–45 min |
| sim settle for the HOLD-pose check | 4.3c | existing (`-SettleS`) | — | user run only |
| hip_reaction-off play | 4.3b | existing knob | — | user run only |
| gc fix in `to_hold` | next flight | common | fresh process per trial | 5 lines |
| sim twin with npz injection for the hist actor | not required now | invented | the blocks above | 1 day+, defer |

Long runs (sim eval, play) are prepared as commands for the user (CLAUDE.md rule). Analysis scripts run locally
with `python`.

---

## 6. Execution order and deliverable

1. §3 video items from the user (can run in parallel with step 2).
2. §4.0 gate.
3. §4.1 re-arm (it decides whether the next flight needs the gc fix or a fresh process per trial).
4. §4.2 B1 event.
5. §4.6a `sup` validation (unlocks the `sup`-based parts of 4.5/4.7/4.8).
6. §4.3, §4.4 hardware metrics; then prepare the sim runs (4.3b/c, 4.4, 4.5) for the user.
7. §4.5, §4.7, §4.8, §4.9.
8. §4.6b predictor skill (if built).

**Deliverable:** `docs/s11-flight-analysis-YYYYMMDD.md`, single pass, one verdict table (supported /
contradicted / inconclusive + the discriminating signal for every id above, including "neither"), figures in
`data/s11_flight_20261011/figs/`, plus a next-flight procedure section: grace pick, gc fix / fresh process,
tether protocol (hands-off or logged pulls), the command ladder, YAW CHECK → heading hold.
