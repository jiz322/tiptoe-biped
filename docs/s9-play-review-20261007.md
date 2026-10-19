# S9 play review: spawn penetration, reset contract, settle-before-handover, sit end state (2026-10-07)

**Status:** analysis plus one sim probe (§4). Nothing in `rl_env_p6.py`, `play_alpha.py` or `control/`
was changed. Main probe run complete: `RL/analysis/s9_spawn_reset_probe/20261007_1416/summary.md`.
Settle extension (H10, contradicted): `…/20261007_1550_D/summary.md`, see §4.2.
**Runs:** S9a_fast25 and S9b_band were killed by the user (marginal return) at model_10190 / model_10180.
Campaign doc: `docs/s9-standup-sit-campaign-20261004.md`.
**Input:** the user's three viewer sessions on S9b 10180 (`run_play_S9.ps1 -Arm b -Sit key`, with
`-GraceS` 4.9 / 2.5 / 1.667, R and X pressed, videos in `…/S9b_band/videos/play/{4.9,2.5,1.667}.mp4`).
The user's frame-by-frame video observations are ground truth. Code and probe are read against them.

## 0. Summary

| # | finding | evidence | affects |
|---|---|---|---|
| F1 | Path-mode spawn height ignores the spawn noise. At d = 0.6, 39 % of spawns put a foot ball > 5 mm under the floor and 9 % > 15 mm. 43 % get a root v_z kick > 0.3 m/s in the first 30 ms | H1, H2 supported | training + play |
| F2 | The d ≥ 0.5 knot-table rest rows predate `plant_hip_reaction`. The noise-free d = 0.6 spawn then settles 0.13 rad deeper at the hip and 6.7° further back in pitch | H9 supported (drift 0.141 rad with the reaction, 0.037 without) | training + play |
| F3 | play_alpha's R key gives a sighted newest frame (training gives none). The contract is different, but **harmless**: success is unchanged within ±2 pts | H5 contradicted | play only |
| F4 | `-SettleS` releases with a sighted history, not the `v` blind contract | code. Sighted is 1–8 pts below blind in H7 | play only |
| — | The S9 policies fail ~20 % of d = 0.6 stand-ups from their **own** training distribution at every speed (success 76–82 %). That base rate is what the user saw after pressing R | probe B | — |
| — | A 1 s PD settle on the training spawns halves d = 0.6 success (80 → 41–45 %), but that is mostly falls **during the hold**: 29–31 % of noisy spawns topple under a held target before the policy ever acts. From a settled, noise-free crouch the policy succeeds 70–77 % (blind). That is 3–6 pts better than the noise-free start without a settle, and ~8 pts below its in-distribution 78–82 %. The policy does not fail from rest. It is mildly tuned to its noisy training opening | H10 contradicted, H7, H4 contradicted | training design |
| — | Sighted release after a settle is 3–13 pts worse than blind. So F4 matters: `-SettleS` in play_alpha underestimates the rig's `v` handover | H7 / D rows | play only |
| — | Phase B start poses (d ≤ 0.45) are not passively stable: a wait there is a collapse. d ≥ 0.5 settles in 92–100 % of noise-free spawns | H3 partial | settle design |
| — | Faster grace does not remove knee clipping during the stand-up (35–47 % of ticks at every speed, in the range of the hardware's 20–43 %). It cuts clipping in the [0.8, 1.6] s window from 59–65 % (4.9 s) to 13–18 % (2.5 s) | leash check contradicted | 10-03 H2 |
| ok | In-step reset IMU read-back (Δ = 0 exactly), belt state re-seed, X-key timing: clean | H8 contradicted | — |

## 1. What was seen, and the code item each observation points to

| run | observation (user) | reading |
|---|---|---|
| all | at α=0.6 the policy seems to exploit an unsettled start. Proposal: wait until stable before NN control, randomized wait | partly: the start *is* unsettled (F1, F2), and the policy is best on that start (78–82 %). But from a settled crouch it still makes 70–77 % (H10). §3 |
| 2.5 | one foot spawned half under the ground. The robot jumps and cannot stand up | F1. The deep tail (< −15 mm, 9 % of spawns) fails 44–46 % vs 20 % for a clean spawn |
| 1.667 | successful stand-ups are smooth: up in 0.6 s, gait transition 0.6–0.9 s, stepping from 0.9 s | grace here is 0.6 × 1.667 = 1.0 s. The policy runs ahead of the path |
| 1.667 | after R, many failures, one a slip. Some failures "not trying the correct thing". Suspected an R-triggered obs bug | the R contract is real (F3) but does not change success (H5). The failures are the ~20 % base rate plus the F1 tail. Friction is randomized (F6) and is a weak factor |
| 1.667 | the sit visibly lowers the body. Can it end stable, maybe via another layer? | §5: yes, under a PD hold. Not under the NN |
| 4.9 | the early phase is no slower than at 1.667. Nearly up at 0.4 s, shuffled until 2.7 s, then stepped | F7: structural. Knee clipping in [0.8, 1.6] s is 59–65 % at this speed |

## 2. Code findings

### F1. Path-mode spawn height ignores the spawn noise (training and play)

`_reset_idx` ([rl_env_p6.py:3427-3498](../RL/source/RL/RL/tasks/direct/standup_curriculum/rl_env_p6.py#L3427-L3498)):
for d > 0 the root height is `path_h(s0) + standup_path_spawn_clearance_m` (0.006). That is the rest height of
the noise-free knot pose. Three perturbations are applied afterwards and never enter the height:

| term (S9 `params/env.yaml`) | scale with d | at d = 0.6 |
|---|---|---|
| `noise_joint_pos` 0.18, every joint independently, L≠R | × (1 − 0.8 d) | ±0.094 rad |
| `kneel_noise_rad` 0.05 (added to the written pose) | × d | ±0.030 rad |
| `init_base_orient_noise_rad` 0.12, roll and pitch | × (1 − d) | ±0.048 rad |
| `noise_base_vel_lin` 0.3 (includes v_z) | × (1 − 0.9 d) | ±0.14 m/s |

PhysX resolves the overlap at up to `max_depenetration_velocity` = 10 m/s
([rl_env_cfg_p6.py:545](../RL/source/RL/RL/tasks/direct/standup_curriculum/rl_env_cfg_p6.py#L545)).
That is the "jump" in the 2.5 video. This is the same family as 09-08's "knots s ∈ (0, 0.15] spawn inside the
floor", which was fixed for the nominal pose only.

Measured (Block A, 2048 spawns, balls only, since the knee collision mesh is not measured):

| d | ball < 0 | < −5 mm | < −15 mm | p01 | kick v_z > 0.3 / > 1 m/s |
|---|---|---|---|---|---|
| 0.6, training noise | 57 % | 39 % | 9 % | −22.5 mm | 43 % / 6.5 % |
| 0.6, joint noise only | 58 % | 39 % | 3.7 % | −16.8 mm | 34 % / 2.2 % |
| 0.6, orientation only | 15 % | 0 % | 0 % | −2.5 mm | 10 % / 0 % |
| 0.6, kneel noise only | 14 % | 0 % | 0 % | −1.9 mm | 0 % / 0 % |
| 0.6, no noise | 0 % | 0 % | 0 % | +5.1 mm | 0 % / 0 % |
| 0.5 / 0.45 / 0.3 / 0.2, training noise | 64 / 54 / 40 / 49 % | 48 / 36 / 23 / 31 % | 15 / 9 / 3 / 6 % | | 39 / 33 / 29 / 36 % > 0.3 |

Joint noise is the cause. The deep tail is what costs success (§4.1: pen < −15 mm fails 44–46 % vs 20 % for
pen ≥ 0, and the top kick decile fails 46–50 %). Moderate penetration (0 to −15 mm) does not raise the
failure rate.

Fix options, cheapest first:
1. Lift each env by its own penetration after the noise is applied: FK of the noised pose over the contact
   set (balls via `measure_ball_geometry`, plus a knee point still to be located), then
   `h = max(path_h, h_needed) + clearance`. About 40 lines, plus a one-off knee-point measurement.
2. Shrink the kneel-start joint noise. One line, but see H4: the policy is currently *better* with the noise
   than without it, so removing diversity is not free.
3. A settled-state bank (§3), which removes the problem by construction.

### F2. The d ≥ 0.5 rest rows predate the hip-reaction plant (training and play)

The knot table's `hip_rest` / `knee_rest` / pitch / root_h columns were measured 09-08 by a zero-action
settle on the plant without `plant_hip_reaction`. S9 trains with the reaction on. Block A, noise off, 2 s
PD hold at the path target, medians over 2048 spawns:

| d (s0) | spawn hip / knee rad | settled hip / knee rad | pitch settled / path | root z settled / path_h |
|---|---|---|---|---|
| 0.60 (0.40), reaction **on** | 1.261 / −2.695 | **1.393 / −2.710** | **−12.9° / −6.2°** | 0.211 / 0.197 m |
| 0.60 (0.40), reaction off | 1.261 / −2.695 | 1.235 / −2.709 | −3.9° / −6.2° | 0.189 / 0.197 m |
| 0.55 (0.45), on | 1.039 / −2.630 | 1.172 / −2.624 | −5.2° / +2.7° | 0.204 / 0.184 m |
| 0.50 (0.50), on | 0.966 / −2.356 | 1.037 / −2.357 | −12.7° / +5.2° | 0.250 / 0.228 m |
| 0.40 (0.60), on / off | 0.814 / −1.818 | 0.824 / −1.805 (on), 0.818 / −1.806 (off) | joints hold; the body collapses (H3) | 0.005 / 0.311 m |

With the reaction off, the table holds to 0.03 rad. With it on, the phase-A hip settles 0.13 rad (7.6°)
deeper and the body pitches further back. This is the sim reproduction of 10-03's "HOLD hip 9° deeper than
sim" (memory hold-pose-hip-offset-belt-reaction-20261003), now inside training. Every S9 kneel episode
opened with this settling motion, and the hardware never shows it: the rig's crouch has already settled
before `v`. The rows above are the corrected rest values for s = 0.40 / 0.45 / 0.50. A full re-knot uses
`probe_standup_keyframes.py` on the hip-reaction plant.

### F3. play_alpha's R key: a different obs history, no effect on success (play only)

Each `_get_observations()` call rolls `_obs_delay_buffer` and `_obs_history_buf`
([rl_env_p6.py:2154-2200](../RL/source/RL/RL/tasks/direct/standup_curriculum/rl_env_p6.py#L2154-L2200)).
Training's auto-reset makes one call: the delayed sensor frame is zero (delay buffer zeroed, delay 1), and the
history-init copies it into all 5 frames, so the start is blind. The R handler in
[play_alpha.py](../RL/scripts/rsl_rl/play_alpha.py) does `env.reset()` (one call) and then
`env.get_observations()` (a second call). The first obs after R is therefore frame 0 sighted, frames 1–4
blind (probe: sensor frames non-zero 100/0/0/0/0 vs 0/0/0/0/0). Viewer startup makes three calls (wrapper
`__init__` reset, `OnPolicyRunner.__init__`, play_alpha), so frames 0–1 are sighted for the first episode of
every session.

H5: success with train / `env.reset()` / R contract = 80.8 / 82.2 / 80.9 % (S9a, grace 1.667). Every
grace and both arms are within ±2.4 pts. The deviation is real and harmless for this policy. The fix (use
`obs, _ = env.reset()` only, and re-blind once before the loop) is cosmetic now. Worth doing so the viewer
cannot mislead the next time a contract matters (08-29).

### F4. play_alpha `-SettleS` releases sighted, not with the rig's blind `v` handover (play only)

`SettleHold` steps the env with zero actions, so the history fills with real settle frames. `end()` clears
only the saturation EMAs (probe: release frames 100/100/100/100/100). Campaign doc §4 calls this rollout
"the hardware handover (`v`, blind)", which it is not. Sighted is 1–8 pts below blind (H7 rows). Fix: in
`end()`, zero `_obs_delay_buffer` and the three `_last*_actions`, set `_needs_history_init`, and recompute
obs once.

### F5. Checked and clean

- **In-step reset reads a fresh IMU state.** The write calls invalidate the body buffers and
  `body_link_pose_w` runs `update_articulations_kinematic()` on read. Probe C: Δquat = Δω = Δv = 0 exactly,
  after 80 steps of a moving stand-up (pre-reset |ω| median 1.5 rad/s).
- **Belt series state.** `BeltKneeActuator.reset` clears `_series_state_valid`, the first compute re-seeds
  from the live joint, and the encoder obs falls back to the physical knee while invalid
  ([belt_knee_actuator.py:548-605](../RL/source/RL/RL/tasks/direct/standup_curriculum/belt_knee_actuator.py#L548-L605)).
- **The X sit key** runs `_start_sit` between steps, the same position as the env's own countdown.

### F6. Floor friction is randomized, per reset (answers the user's question)

`run_play_S9.ps1 -Plant train` (the default) passes μ multiplier U[0.6, 1.1], redrawn at every reset
including R. Materials are 1.0 with `multiply` combine, so the effective μ ∈ [0.6, 1.1]. H6 is
inconclusive: the low μ tercile fails 23.3 % vs 17.1–17.5 % for the high tercile (ratio 1.33–1.36). So μ is
a weak factor, and the slip episode was a low draw, an F1 kick, or both. The strongest DR correlate (not
pre-registered) is the CoM shift: x < −1 cm fails 38 %, x > +1 cm fails 11–12 %.

### F7. Early rise, then shuffle until grace expiry (structural)

While ch30 > 0 the policy is on the stand-up path. Commands start only after grace plus 0.5 s of
stabilization. At grace 4.9 the path from d = 0.6 lasts 2.94 s, so a policy that is up early shuffles until
expiry. The knee clips in 59–65 % of live ticks in [0.8, 1.6] s at 4.9, 13–18 % at 2.5 and 23–25 % at 1.667
(that window already runs past the 1.0 s grace end). Over the whole grace phase the knee clips 35–47 % at
every speed. So in sim, as on hardware (20–43 %), clipping is the stand-up's style, and a faster path
removes the long shuffle, not the clipping (10-03 H2 holds only for the shuffle part).

## 3. The settle-before-handover proposal (difficulty prediction, per the workflow rule)

**What the probe says about the premise.** The training start *is* unsettled (F1 kicks, the F2 settling
motion), and the policy does best on exactly that start: 78–82 % at d = 0.6. On the starts the rig presents,
it does somewhat worse but does not collapse:

| d = 0.6 start (blind) | S9a | S9b |
|---|---|---|
| training spawn (noise, unsettled) | 78–82 % | 79–81 % |
| noise-free, unsettled (F2 transient only) | 67–72 % | 66–70 % |
| **noise-free, settled 1 s** (≈ the rig's HOLD → `v`) | **70–77 %** | **70–74 %** |
| noisy, settled 1 s, upright at release | 59–66 % | 61–64 % |
| noisy, settled 1 s, all envs (29–31 % fell during the hold) | 41–46 % | 43–44 % |

So the user's hypothesis holds in the weak form: the policy is tuned to its training opening and loses about
8 pts on a calm start. The strong form ("fails from rest") is contradicted (H10). The halving in H7 is mostly
the hold itself: a noised crouch held by PD topples 29–31 % of the time, including 19–23 % of spawns with no
penetration at all. This bounds the settle design: a hold in training needs a start that does not topple
under the held target, i.e. F1 fixed and the joint noise reduced, or the hold just produces falls.

**What a settle buys that a re-knot does not.** Fixing F1 and F2 makes the *nominal* spawn settled. The
settled pose still depends on each episode's DR draw (mass ×1.00–1.14, CoM shift ±2 cm, belt k 100–1000,
the hip reaction). Only a settle, or a bank of settled states, gives the policy the rig's opening under
every draw.

**Is it common?** Spawning from physically valid, settled states is common: reference-state
initialization in imitation work, and stand-up / get-up work that makes start postures by letting the robot
settle in sim (the HoST humanoid stand-up work does this, as far as I recall). A randomized wait *inside*
the training episode is less common. Most pipelines settle offline.

**Where a wait is possible (H3).** Noise-free spawns settle in 100 % at d = 0.55–0.6 and 92 % at d = 0.5.
At d = 0.45, only 8 % settle within 2 s. At d ≤ 0.4, 30–50 % settle and the median root ends near the floor.
Below d = 0.5, a wait is a collapse, so the wait must be 0 there (user's caveat confirmed).

| option | what | cost | risks |
|---|---|---|---|
| 0. Fix F1 + re-knot F2 | lift by the measured penetration; rest rows for s ≤ 0.5 from §2 F2 / `probe_standup_keyframes.py` on the hip-reaction plant | ~0.5 day | the opening still carries a small DR-dependent transient. Does not by itself teach the settled start |
| 1. Randomized in-episode hold (user's proposal) | per-env `_hold_remaining` ~ U[0, T] at reset, d ≥ 0.5 only. Target = path(s0) whatever the action; grace / gait / clock frozen; fences off; reward 0. At expiry re-blind (= `v`) | ~1 day | hold steps still enter PPO. Their actions do nothing, so their advantages are value noise: variance proportional to the hold share. A clean mask needs a second unversioned rsl_rl `ppo.py` patch. 2048 envs × 1 s of hold per kneel episode is a few % of samples |
| 2. Settled-state bank | Block A-style holds offline per d with DR draws; store the settled states with their draws; reset by sampling the bank and re-applying the draw | ~1–1.5 days (Block A is most of the builder) | lower start diversity. Note that H4 says diversity currently helps. DR coupling plumbing |

Recommendation (H10 is contradicted): do option 0 regardless, since F1's deep tail and F2 are artifacts. A
training-side settle is then a modest gain, not a rescue. The measured gap between the training spawn and a
settled calm start is ~8 pts. Option 0 should already move the training start toward the settled one,
because the nominal spawn becomes the settled pose. Re-run this probe after option 0. If the "noise-free,
settled" row still trails the training-spawn row by more than ~5 pts, option 1 (hold at d ≥ 0.5 only, blind
release, with the joint noise reduced so the hold does not topple) is the cheaper, DR-faithful way to close
it. The user decides.

## 4. Probe

Script `RL/scripts/rsl_rl/probe_s9_spawn_reset.py`, launcher `RL/launch_probe_s9_spawn.ps1` (recipe =
`run_play_S9.ps1`'s, `-Sit key -Plant train`, grace band off and set per block, pushes off), aggregator
`RL/scripts/rsl_rl/aggregate_s9_probe.py` (applies the rules mechanically). 2048 envs per block, seed 0,
S9a 10190 on GPU 0 and S9b 10180 on GPU 1. Block A has no policy and the same seed, so its two arm
copies are the *same* draws (identical numbers), not replicates.

### 4.1 Main run (pre-registered 2026-10-07, launched 14:16, complete)

Blocks: A = 2 s zero-action PD hold, d ∈ {0.1 … 0.6} × noise {train, off} + noise attribution + hip-reaction
off. C = IMU read-back at a training-style reset. B = policy at d = 0.6 (and d = 0.3), grace {1.667, 2.5,
4.9} × {noise × reset contract, settle 1 s blind / sighted}. Success = no termination before grace + 3 s.

| id | hypothesis | rule | result (S9a / S9b) | verdict |
|---|---|---|---|---|
| H1 | spawn penetration at d = 0.6 | ball < −5 mm ≥ 10 % (train) and ≤ 1 % (off) | 38.7 % vs 0 % | **supported** |
| H2 | depenetration kick | root v_z > 0.3 m/s in 30 ms ≥ 5 % and off ≤ 1 % | 43.4 % vs 0 % | **supported** |
| H3 | noise-free pose settles at d ≥ 0.55, topples at d ≤ 0.4 | settled ≥ 90 %; toppled ≥ 50 % | settled 100 % (0.55, 0.6); toppled 27–56 %, settled only 30–50 % with the root on the floor | **partial**: the upper half holds; the lower half collapses rather than topples |
| H4 | spawn noise costs success at d = 0.6 | off − train ≥ 5 pts and fail ratio pen<−5mm/pen≥0 ≥ 2 | off − train = −10 … −15 pts; ratio 1.0. At d = 0.3 the reverse: off 98–100 % vs train 71–75 % | **contradicted** at d = 0.6 |
| H5 | the R contract lowers success | train − double ≥ 5 pts | within ±2.4 pts at every grace, both arms | **contradicted** |
| H6 | low μ drives failures | low/high tercile fail ratio ≥ 1.5 | 1.36 / 1.33 | **inconclusive** |
| H7 | (descriptive) 1 s settle | — | success none / blind / sighted: S9a 78–82 / 43–46 / 34–38 %, S9b 79–81 / 41–45 / 40–41 % | reported |
| H8 | stale IMU at the in-step reset | Δ > 1e-3 | Δ = 0 exactly | **contradicted** |
| H9 | noise-free d = 0.6 spawn off rest because of hip_reaction | drift > 0.05 rad on, < 0.05 off | 0.141 vs 0.037 rad | **supported** |
| leash | knee clip in [0.8, 1.6] s < 5 % at grace 2.5 / 4.9 | — | 12.9 % / 59.1 % (S9a), 18.0 % / 65.1 % (S9b) | **contradicted** |

Stand-up success at d = 0.6 (training noise and contract): S9a 80.8 / 82.3 / 77.6 %, S9b 81.0 / 80.4 /
78.5 % at grace 1.667 / 2.5 / 4.9. "Fail at grace" (never up, or down by grace expiry) is 15–21 %.
"Fail after grace" is 0.4–4 %. Failure-rate bins (pooled over graces, per arm):

| bin | S9a fail | S9b fail |
|---|---|---|
| ball pen ≥ 0 / 0…−5 / −5…−15 / < −15 mm | 20.6 / 15.9 / 13.2 / **46.2 %** | 20.5 / 19.0 / 12.9 / **43.9 %** |
| kick v_z ≤ median / top decile | 15.6 / **50.2 %** | 17.1 / **45.7 %** |
| μ low / high tercile | 23.3 / 17.1 % | 23.3 / 17.5 % |
| belt k low / high tercile | 21.6 / 18.8 % | 20.1 / 20.2 % |
| CoM shift x < −1 cm / > +1 cm | **38.2** / 10.7 % | **38.7** / 12.5 % |

### 4.2 Settle extension H10 (pre-registered 2026-10-07 after reading 4.1, before running)

H7 mixes two things: envs that topple during the noisy hold, and envs that fail from a genuinely settled
crouch. Block D runs d = 0.6, 1 s settle, noise **off**, blind and sighted release, at the three graces. It
also re-runs the noisy blind settle with a topple-during-settle flag (|pitch − path| > 0.6 rad or
|roll| > 0.5 rad at release).

**H10:** the policy loses ≥ 10 pts from a settled noise-free crouch: success(noise off, no settle) −
success(noise off, settle 1 s blind) ≥ 10 pts at every grace → supported (the policy depends on the opening
transient). ≤ 5 pts at every grace → contradicted (the H7 drop is the in-hold topples). Otherwise
inconclusive.

Result (`RL/analysis/s9_spawn_reset_probe/20261007_1550_D/summary.md`, merged with the main run):
**contradicted** for both arms. Noise off, no settle → noise off, settle 1 s blind: S9a 68.1 → 72.8,
72.1 → 77.3, 66.5 → 69.7 %; S9b 65.8 → 71.6, 69.6 → 74.4, 67.4 → 69.9 % (grace 1.667 / 2.5 / 4.9). The settle
*helps* by 3–6 pts. Noise-free spawns never topple in the hold (upright at release = success-eligible
population, 99.9 %+).

The noisy blind settle, re-run with the flag: 28.5–30.6 % topple during the 1 s hold. Of those with a clean
spawn (ball ≥ 0), 19–23 % still topple. Success among the upright: 59–66 %. Sighted vs blind release from the
noise-free settle: S9a 59–65 % vs 70–77 %, S9b 64–70 % vs 70–74 %. The blind contract is worth 3–13 pts (F4).

Neither run is a hardware statement: the knee contact is unmeasured, and everything is the sim plant.

## 5. Can the sit end in a stable state?

Yes, under a PD hold, not under the NN. The sit target s = 0.40 is the d = 0.6 crouch, which under a frozen
PD target settled in 100 % of 2048 noise-free spawns (Block A). It settles at the F2 pose (hip 1.39 rad,
pitch −13°), not at the knot-table row. Training ends every sit as a time-out 1.5 s into the hold, so the
net never learned to hold indefinitely. A long hold feeds it constant obs (ch29 = −0.6, ch30 = 0.6) for
longer than it ever trained on.

So the stable end belongs in a deploy-side layer. Once the descent is done and the robot is quiet (|ω| below
a threshold, pitch near the settled crouch pitch, joint speeds low, for N ms), hand over from the NN to the
scripted PD hold at path(0.40). That is the same `HOLD` mode `control/balance_standup.py` runs before the
stand-up. The pattern is common: a state machine with a scripted rest pose at each end of the learned
transition. Cost: small (one gate and one state, on top of the campaign doc's §7 sit plumbing). Take the
gate tolerances from the F2 settled values. Doing it in training (a longer `sit_hold_seconds` plus a
stillness reward) is redundant, because the pose is passively stable.

## 6. Decisions for the user

1. F1 fix (lift by the measured penetration) and F2 re-knot of the s ≤ 0.5 rest rows on the hip-reaction
   plant: build or not (§3 option 0).
2. Settle-before-handover (option 1, in-episode hold, d ≥ 0.5 only, with reduced joint noise; or option 2,
   bank): H10 says it is a ~8-pt refinement, not a rescue. Decide after re-probing on top of option 0.
3. play_alpha fixes: F4 (blind `-SettleS` release) matters, worth 3–13 pts in the sim twin of a hardware
   handover. F3 (R key) is cosmetic for this policy (H5).
4. Sit end state: the deploy-side NN → PD-hold gate (§5), as part of the campaign doc's §7 sit work.
