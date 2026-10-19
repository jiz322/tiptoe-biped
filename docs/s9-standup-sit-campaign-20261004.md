# S9 campaign: faster stand-up from the crouch + sit-down, on the corrected plant (2026-10-04)

**Status:** training launched (two arms, both 4090s). Nothing in `control/` has changed yet.
**Base policy:** `s8b_8880` (`RL/logs/rsl_rl/p9_replant/2026-09-19_08-14-51_S8b_air03/model_8880.pt`
= `control/s8b_8880.pt`, the live hardware policy).
**Supersedes:** the arm sequence of `docs/sit-campaign-plan-20261001.md` (Sit-0 replay → Sit-1 →
Sit-2). The 10-01 constraints still hold and are used here: the grace counter is the termination
shield (§1.1), obs ch 30 is the phase channel (§1.2), and the separate-net and channel-append
routes stay rejected (§3). What changed is the order and the sit target, both driven by the
10-03 flight (`docs/s8b-standup-flight-analysis-20261003.md`).

## 1. What the 10-03 flight changed, and what this campaign does about it

| 10-03 finding | user's call (10-04) | in this campaign |
|---|---|---|
| §1 HOLD hip 9° deeper than sim = belt J^T reaction c·τ_knee on the hip, unmodeled | sim2real plant fix, apply to all following trainings | `env.plant_hip_reaction=true` in both arms; hip PhysX cap raised so the reaction is not clipped (§3.1) |
| §2 knee command 22–27 Nm vs the 17 Nm motor clamp at the lift-off rock | treat the clamp as a feature, not a bug | no torque-demand penalty, no clamp change. The sim already clips the knee at 23.2 Nm joint-side |
| §3 the "shuffle" is the knee riding its action leash from 0.8 s: the path is too slow | make α=0.6 the new α=1, then make the stand-up faster | curriculum pinned at α=0.6; path speed doubled (S9a) or randomized (S9b) |
| §5 belt tooth skip, every flight | acknowledged; training does not wait for the fix | not modeled. Section 6 lists what this costs |
| (question) do stand-up / sit need more quality reward? | add grace-phase smoothness only | `env.grace_smooth_frac=0.5`. Sit shaping is required, not optional (§3.4) |

Decisions taken this session (asked and answered):

1. **α 0.6 → "α 1" = cap the curriculum at 0.6.** No re-knot. The hardest start is the d=0.6
   crouch (knot s=0.40: hip 1.321 / knee −2.562, root 0.197 m, knees and foot balls down), which
   is the pose the 10-03 flight launched from. Obs ch29 (difficulty) and ch30 (grace_norm) keep
   the meaning the net trained on, and the deploy knot table and d=0.6 HOLD stay exactly as flown.
   A re-knot would have changed both channels' meaning and still not changed the speed (finding 3).
2. **Speed: A/B on both GPUs.** S9a fixed, S9b per-episode band (§2).
3. **The sit ends in the d=0.6 crouch** (s=0.40), not in seiza. The deploy loop becomes
   crouch → stand → crouch. Hardware has never flown the seiza ↔ crouch segment, and the
   stand-up no longer starts from seiza.
4. **Quality reward: grace smoothness only.** The foot-slip penalty (evidence: one trial, video
   only) and a leash-saturation penalty were declined. The leash penalty would also have
   confounded the speed A/B, because the slow path causes the saturation.

## 2. Arms

Both arms resume `S8b_air03/model_8880.pt` with S8b's full override list re-passed verbatim, plus
the 2000 Hz sextet. The current cfg defaults are the T-line 500 Hz set, and the belt DR band
100–1000 is unstable at 500 Hz (memory: physics-rate-500hz-belt-limit). The two launchers
differ only in the speed lines (`diff RL/launch_train_S9a_fast25.ps1 RL/launch_train_S9b_band.ps1`).

| | S9a_fast25 | S9b_band |
|---|---|---|
| launcher | `RL/launch_train_S9a_fast25.ps1` | `RL/launch_train_S9b_band.ps1` |
| GPU | 0 (RTX 4090) | 1 (RTX 4090 D) |
| `standup_grace_seconds` | 2.5 fixed | 2.5 nominal, per-episode U[1.667, 5.0] |
| crouch → stand (d=0.6) | 1.5 s (was 3.0 s) | 1.0 – 3.0 s |
| sit descent stand → crouch | 1.5 s | 1.0 – 3.0 s (same per-episode speed) |
| everything else | identical | identical |

Shared S9 deltas vs S8b:

| key | value | why |
|---|---|---|
| `env.plant_hip_reaction` | true | 10-03 §1 |
| rate sextet | `decimation 20`, `sim.dt 0.0005`, `render_interval 20`, `hold_decimation 20`, `rotor_coupling_dt 0.0005`, `series_dt 0.0005`, `series_substeps 4` | S8b's as-run rate. This is not a change, but it must be pinned now |
| `env.curriculum_initial_alpha` / `_max_alpha` | 0.6 / 0.6 | decision 1. Kneel starts span d ∈ [0, 0.6], half of them in the frontier band [0.3, 0.6] |
| `env.standup_grace_seconds` | 5.0 → 2.5 | decision 2 |
| `env.grace_smooth_frac` | 0.5 | decision 4 |
| `env.sit_enable` / `sit_probability` / `sit_target_s` / `sit_hold_seconds` | true / 0.3 / 0.4 / 1.5 | decision 3 |
| `env.rew_scale_sit_height` / `_sit_descent` | 4.0 / −15.0 | sit shaping (§3.4) |
| `--max_iterations` | 3000 (ends at 11880) | bounded. The kill and gate rules below decide earlier |

**Why 2.5 s for S9a.** At 2.5 s the path is at s = 0.72 by t = 0.8 s, a knee target of −73°.
Flights 4–6 measured the knee at −61…−72° at 0.8 s. So the S9a path moves at the speed the
hardware already chose by itself, and the leash should stop binding. The S9b band brackets this
speed from 1.5× faster to 2× slower, so a deploy-side speed choice is interpolation, not
extrapolation.

## 3. What was built (all defaults inert: S8b and earlier recipes stay bit-exact)

Files: `RL/source/RL/RL/tasks/direct/standup_curriculum/rl_env_cfg_p6.py` (S9 block after
`rew_scale_standup_contact`) and `rl_env_p6.py`.

### 3.1 Hip reaction: PhysX cap

`plant_hip_reaction` already existed (built 10-03, post-clip plant force). The open caveat was that
PhysX still capped the hip's total joint effort at `effort_limit_sim` 22 Nm. At the lift-off rock
that is 17 Nm of motor plus up to 0.733 × 23.2 = 17 Nm of reaction. With the knob on, the env now
raises the hip's `effort_limit_sim` by c × knee `effort_limit_sim` (22 + 0.733·30 = **44 Nm**). The
DC envelope that clips the hip *motor* (17 Nm) is unchanged. The `[P10]` startup line prints the
value. Deploy needs nothing: the real plant has the reaction, and `control/config.py
HIP_REACTION_FF` must stay False.

### 3.2 Speed band

`standup_grace_seconds_min/max` (0 = off). The env samples grace seconds per episode, and the
assigned grace is d × that. Obs ch30 is normalized by the episode's own maximum, so
`grace_norm = 1 − s` holds at every speed. With the band off, the normalizer equals the old
global value, so the result is bit-exact. No other code path reads the global maximum in path mode.

### 3.3 Grace smoothness

`grace_smooth_frac`. `action_smoothness` (jerk) and `action_snap` were multiplied by
`anti_flail = 1 − in_grace`, so every stand-up through S8b ran with no jerk regularizer. They now
carry this fraction of their nominal scale during grace (stand-up and sit), without the idle
`motion_gate` (0.08), because grace is a motion. At 0.5 the grace weight is 6× the idle
post-grace weight. Jerk penalties are small on a smooth 2–3 Hz sinusoid and large on the ±1
chatter that leash riding produces (20–43 % clipped ticks on hardware). The chatter is the target.

### 3.4 Sit branch

`sit_*` keys (path mode only, `ValueError` otherwise).

- **Trigger.** With `sit_probability`, an episode gets a sit countdown of U[1, 4] s that runs only
  while the robot stands (post-grace). The trigger covers both "spawned standing" and "stood up
  from the crouch". The latter is the deploy sequence.
- **Motion.** The trigger re-arms `_grace_remaining`, which is the termination shield. The action
  center runs the **same knot table backward**, s: 1 → 0.40, at the episode's own path speed.
  It then holds s = 0.40 for `sit_hold_seconds`, and the episode ends as a **time-out**
  (bootstrapped, success). It never ends as a termination. path(s=1) equals the standing default
  pose (hip 0.05, knee 0), so the trigger causes no jump.
- **Obs.** While sitting, `grace_norm = 1 − s` (the stand-up's own (grace_norm, pose) pairs,
  10-01 §1.2) and **`difficulty = −0.6`** as the direction signal. A stand-up never sees a
  negative difficulty, so at equal (grace_norm, pose) the net can tell rising from sitting.
  ch29 is mirror-invariant, so the symmetry tables are untouched. **Deploy contract item:** the
  hardware sit must feed −0.6 on ch29.
- **Fences.** The absolute fences (min_height 0.30, tilt 1.8) would kill the crouch (root 0.197),
  so a sitting env is shielded from them for the rest of its episode, as 10-01 §1.1 required. It
  gets path-relative fences instead: |pitch − path_pitch(s)| > 0.8 rad, |roll| > 0.6 rad, or any
  split fence → terminated (−10). The saturation fence stays disarmed, as in every grace phase.
- **Shaping.** Pitch tracking and feet-down (the stand-up's grace pair) work unchanged on the
  reversed s. The stand-up height-progress term (+10 × progress toward standing) and the fall
  term (−15 × downward speed) would pay the robot for not sitting, so on sitting envs they are
  removed. In their place:
  `sit_height` = 4 · exp(−((h − path_h(s)) / 0.05)²), and
  `sit_descent` = −15 · clamp(−v_z − v_path − 0.05, 0, 0.3), where v_path is the path's own
  descent speed at that s (0 during the hold). This is the soft-landing term: it costs nothing to
  follow the path and costs only for dropping faster than it.
- **Curriculum record.** A sit episode enters the stand-up success buffer with its length at the
  sit trigger, so `success_high_rel` stays a stand-up metric.
- **Logs.** `sit/active_frac`, `sit/pitch_err_mean`, `sit/height_err_mean`,
  `sit/success_rate` (completed / (completed + fenced), refreshed every 500 sit endings),
  `sit/completed_total`, `sit/failed_total` (cumulative, see §3.5), `reward/sit_height_*`, `reward/sit_descent_*`.

### 3.5 Smoke check (64 envs, 8 iterations, sit p=1, 0.05–0.2 s delay, 0.3 s hold; S9b launcher `-Smoke`)

This checks code paths only, not performance. The smoke resumes model_8880, so it is the
untrained S8b policy meeting the sit for the first time.

- Startup: `[P10] … hip_reaction=True (hip effort_limit_sim 44.0)`, `[S9] grace_seconds=U[1.6667, 5.0] per episode …`,
  α pinned at 0.600, and the checkpoint loads cleanly. No traceback, empty stderr.
- Sits run both ways. At the end of iteration 8 the totals were **102 completed** (time-out) and
  **124 fenced** (≈ 45 % zero-shot). The smoke settings are harsher than the run's: descents as
  short as 1.0 s and a 0.3 s hold.
- While sitting: height error 5–9 cm from path_h(s), pitch error 0.18–0.27 rad.
- Logging caveat found here: rsl_rl appends the env's `extras["log"]` dict **by reference**, and
  the env mutates the same dict every step. So every per-iteration value in TB is effectively the
  last step's snapshot, not a rollout mean. This is true of every metric this repo has ever
  logged. Event counts are therefore logged as cumulative totals (`sit/completed_total`,
  `sit/failed_total`), and `sit/success_rate` is refreshed per 500 sit endings. Read rates from
  the totals' slopes, not from single-iteration values.

## 4. Pre-registered predictions (written before any full-run data)

Baselines are S8b's own TB means over iterations 8780–8880 (α = 1.0, seiza included): episode
length 2032, `standup/pitch_err_grace_mean` 0.150 rad, `Loss/value_function` 3566.

| id | prediction | read at | discriminating signal |
|---|---|---|---|
| P1 | startup prints `hip_reaction=True (hip effort_limit_sim 44.0)` and the arm's `[S9]` line | launch | log grep. Anything else = wrong checkout |
| P2 | episode length dips in the first ~200 iters (new plant force, faster path, sit reward swap, and sit episodes are short by construction) and recovers to ≥ 1400 by +1000 | +300, +1000 | `Train/mean_episode_length` |
| P3 | sit is learnable from the stand-up policy: `sit/success_rate` ≥ 0.5 by +1000, ≥ 0.8 is the bar for a hardware sit | +600, +1000 | `sit/success_rate` |
| P4 | the faster path removes the leash riding without adding tracking error: `standup/pitch_err_grace_mean` ≤ 0.150 at +1000 | +1000 | TB, both arms |
| P5 | the stand-up stays learned at α 0.6: `curriculum/success_high_rel` ≥ 0.7 | +1000 | TB |
| P6 | S9b pays for generality: at +1000 S9b trails S9a on P2–P5 by a small margin (< 10 % relative). A larger gap means rate-agnosticism is expensive and S9a is the deploy pick | +1000 | the same four signals, side by side |

The decisive test is the sim twin, not TB: a fixed-d=0.6 rollout with `-SettleS` under the
hardware handover (`v`, blind), checking whether the knee action still clips 20–43 % of ticks in
[0.8, 1.6] s as on hardware. If clipping drops below ~5 %, H2 of 10-03 (the slow path causes the
shuffle) is confirmed in sim. That needs the play-side npz logger 10-03 §7 asked for. It is still
not built and is the first item in §7.

## 5. Kills and gates

- `Loss/value_function` > 42000 for 50 consecutive iterations → kill (S8 convention).
- `Train/mean_episode_length` < 400 at +300 → kill.
- PhysX error flood in `train_<RUN>.err.log` → kill.
- `sit/success_rate` == 0, or `sit/completed_total` flat over 100 iterations, at +600 → the branch is not being
  learned. Kill rather than extend.
- Gate at +1000: P2 and P3 met, with every DR band live.

## 6. Known gaps (accepted on purpose)

- **Tooth skip is not modeled.** The sim belt is elastic (DR 100–1000 N m/rad) but it never
  slips. On hardware a skip moves the knee zero by 12° per tooth, invisibly to the policy
  (10-03 §5: the one-sided-skip flights carried a 2–3× roll bias through the crouch). A policy
  from this campaign meets that unmodeled on every flight that skips. The crawl-pose check
  between trials stays mandatory.
- **The hip can now carry 34 Nm in sim at the rock.** That is the measured physics, but nothing
  on the rig has measured the dynamic case. 10-03's bench check (hold the crouch, step only the
  knee target, expect Δq_hip ≈ 0.037 rad per knee Nm) is still the cheap confirmation. If the
  dynamic reaction turns out smaller (the 08-06 replay read it as absent), S9 trained against a
  stronger forward pitch than the rig has.
- **Faster path, possibly higher lift-off peaks.** The rock already hits the knee clamp. A 2×
  faster center asks for it sooner, not harder, because the clamp caps the knee command. So more
  skipping is possible, not certain.
- **The sit trains from standing only.** A sit from a disturbed or stepping state (the hardware
  limit cycle, 2.5 Hz) is reached only through the 1–4 s delay window. The 10-01 watchdog (auto-sit
  on a pre-fall signature) would ask for exactly that, and it is out of scope here.

## 7. Deploy work before any S9 flight (not started)

1. `control/config.py STANDUP_GRACE_SECONDS` 5.0 → 2.5 (S9a), or the speed picked from an S9b
   sim sweep. `grace_norm` and the action center both follow from it.
2. Sit on the deploy side: `standup_path.py` reverse traversal (s 1 → 0.4, center = path(s)),
   obs ch29 = **−0.6**, ch30 = 1 − s, then hold at s = 0.4. Add a key in `balance_standup.py`
   to trigger it from NN control.
3. `HIP_REACTION_FF` stays False (desk-check item).
4. Play-side: an npz logger in `play_alpha.py` (10-03 §7) plus a sit mode, so §4's decisive test
   and a sim twin of every flight exist before the flight.
5. `standup_path_check.py` desk check against the S9 run's `params/env.yaml` (grace, speed,
   sit keys), the same way 10-02 checked S8b.

## 8. Launch record

Launched 2026-10-04 15:36 (local), detached, both from model_8880 → `--max_iterations 3000` (to 11880).

| | S9a_fast25 | S9b_band |
|---|---|---|
| run dir | `RL/logs/rsl_rl/p9_replant/2026-10-04_15-37-07_S9a_fast25/` | `RL/logs/rsl_rl/p9_replant/2026-10-04_15-37-07_S9b_band/` |
| logs | `RL/logs/train_S9a_fast25.{out,err}.log` | `RL/logs/train_S9b_band.{out,err}.log` |
| PID at launch | 45796 | 54072 |
| P1 | **met**: `hip_reaction=True (hip effort_limit_sim 44.0)`, `[S9] grace_seconds=2.5 fixed … SIT p=0.3` | **met**: same `[P10]`, `[S9] grace_seconds=U[1.6667, 5.0] per episode … SIT p=0.3` |
| iteration time | 181–188 s | 188–190 s |
| iter 8881 | ep len 164, value loss 1.6e5, sit 0 completed / 743 fenced | ep len 195, value loss 1.6e6, sit 0 completed / 571 fenced |

Reading the first two iterations:

- **Iteration time.** S8b ran at 9635 fps, about 435 s per iteration (TB `Perf/total_fps`), so
  ~188 s is faster, not slower. Expect +300 after about 16 h and +1000 after about 2.2 days.
- **Zero completions at iteration 2 are structural.** The earliest possible completion needs
  ≥ 1 s of standing delay + 1.5 s of descent + 1.5 s of hold, about 400 steps, and only 256
  steps had run. The first `sit/success_rate` refresh (0.0) therefore contains fences only.
  Read the sit from `sit/completed_total` / `sit/failed_total` slopes after ~+20 iterations.
- **The value-loss spike** (1.6e5 / 1.6e6) is the reward-swap shock P2 expected. The kill rule
  needs 50 consecutive iterations above 42000.

Leftovers: `RL/logs/rsl_rl/p9_replant/2026-10-04_15-1*_smoke_S9b_band/` are the smoke runs
(8 iterations, 64 envs). They can be deleted.
