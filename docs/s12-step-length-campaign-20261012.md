# S12 campaign: touchdown step-length reward for the vx command (2026-10-12)

**Status:** training launched 2026-10-12 22:36 on GPU 1 (RTX 4090 D), run
`RL/logs/rsl_rl/p9_replant/2026-10-12_22-36-09_S12_steplen/`, launcher `RL/launch_train_S12_steplen.ps1`.
**Control:** S11_hist, still running on GPU 0 and passing through the same checkpoint (model_12330) on its way to
12920. Nothing in `control/` has changed.
**Decision source:** user 10-12. The vx tracking reward scores a 1 s EMA of body velocity, so a gait that surges and
slows inside the window scores the same as an even one. Proposal: score each footstep instead, by how far the landing
foot is placed ahead of the planted foot compared with a commanded distance. The user declined the pre-check
(stride-to-stride speed spread on an existing rollout) and asked for a non-conservative weight "to see the difference".

## 0. Summary

| item | value |
|---|---|
| change vs S11 | one new reward term, slot 42 `step_length`. Every other override is S11's, verbatim |
| what it scores | at each touchdown of foot i: lead s = (ball_i − ball_j) · heading, against s* = vx_cmd / (2 · gait_freq) = vx_cmd / 5 |
| payout per touchdown | 100 · exp(−(s − s*)² / 0.02²) · min(Δt_own · 2.5, 1), moving commands only, after grace |
| weight | 100 per touchdown ≈ 500/s at the clock's 5 touchdowns/s: **1.7× the vel_tracking budget** (3.0/tick = 300/s), near `alive` (600/s). Deliberately not conservative |
| targets | vx 0.05 → 1 cm, 0.15 → 3 cm, 0.30 → 6 cm, −0.15 → −3 cm; σ 2 cm |
| start | S11_hist model_12330 (hist task, deployable through `control/balance_s11.py` after export) |
| length | 1000 iterations → 13330; 168 s/iteration (155.9 s collection + 12.3 s learning) → about 47 h, ending around 10-14 21:30 |
| verification | 3 smokes: ball radius 20.00 mm, ball geometry level to 3.6 mm at the default pose, detector at the clock cadence (5.05 Hz), every `step/*` key logged each step (§3) |
| baseline finding | S11 lands each foot **6–7 cm (mean abs) off the target, std 7–10 cm**, step to step, even at vx = 0. Step placement is very irregular, which supports the premise of the change (smoke, 64 envs, §3) |

## 1. What was built

`RL/source/RL/RL/tasks/direct/standup_curriculum/`:

- `rl_env_cfg_p6.py`: seven new knobs, all inert at their defaults (`rew_scale_step_length = 0.0` is bit-exact
  with S11): `rew_scale_step_length`, `step_length_sigma_m`, `step_length_idle_frac`, `step_td_arm_m`,
  `step_td_land_m`, `step_td_min_air_s`, `step_length_track_enable`.
- `rl_env_p6.py`:
  - `_init_step_length()` measures the ball centre offset and radius on the live stage with the same
    `measure_ball_geometry` fit the pad and the S10 spawn lift use, and refuses to run if the radius is not about
    20 mm.
  - `_step_length_reward()` runs every step, including grace, so the first touchdown after grace is a real one.
    It advances a per-foot touchdown latch, pays at events and writes the `step/*` diagnostics.
  - The result is appended as reward slot 42 after the sit slots. `compute_rewards_vx_yaw`'s signature is
    unchanged.
  - Reset clears the latch.
- `RL/launch_train_S12_steplen.ps1`: S11's launcher, GPU 1, fork point, 1000 iterations, plus the S12 block.

### 1.1 The term

```
ball_k      = shin_k pose ⊕ measured ball offset        (k = L, R)
bottom_k    = ball_k.z − R − floor
latch       : armed when bottom > 6 mm; touchdown when an armed foot comes back below 2 mm
              after ≥ 50 ms armed (shorter spells, i.e. bounces, disarm without an event)
lead_i      = (ball_i − ball_j)_xy · (cos ψ, sin ψ)      ψ = base yaw at the touchdown tick
s*          = vx_cmd / (2 · gait_freq)
r_i         = 100 · exp(−(lead_i − s*)² / σ²) · min(Δt_own,i · gait_freq, 1)
r           = Σ_i r_i · td_i · clamp(|vx_cmd| / 0.05, 0, 1) · (1 − in_grace) · pickup_gate
```

Why each piece is there:

- **Lead over the planted foot, not the foot's own travel.** In alternating gait the planted foot has not moved
  since its own touchdown, so its current position *is* "the previous right foot". A constant fore-aft split
  shows up as alternating long and short steps, and the per-step kernel penalises it: the term carries an
  anti-limp component for free. A stride-based version (each foot against its own last contact) would double
  the signal but cannot see limping.
- **Ball, not nail.** `_get_toe_pos_w` is the nail origin, 59 mm up the shin. A 10° change of shank angle at
  touchdown moves it about 1 cm horizontally relative to the ball, which is the entire target at vx 0.05.
- **Rate factor.** A positive per-event reward otherwise pays for more events. With `min(Δt_own · f, 1)`, stepping
  faster than the 2.5 Hz clock earns proportionally less per event, so the payout rate peaks at the clock cadence.
  Stepping slower simply produces fewer events.
- **Moving commands only** (`step_length_idle_frac 0`). Idle envs (vx_cmd = 0, half of all segments) get nothing,
  so the A/B reads on vx and does not also add a "step in place" incentive to the idle-drift problem.
- **Kept unchanged:** vel_tracking (1 s EMA, σ 0.10, ×3.0), foot_progress (×100, linear "further is better") and
  back_leg_step (×80). In the control they are worth 1.84, 0.041 and 0.049 per step respectively, so the two
  untargeted foot terms are about 1/40 of the new term and do not compete with it.

### 1.2 Diagnostics (`step/*` in TensorBoard)

rsl_rl logs a last-step snapshot of `extras["log"]`, and touchdowns are sparse per tick, so the term keeps
decaying sums (factor 0.995, about a 2 s window) and writes every key every step:

| key | meaning |
|---|---|
| `step/td_rate_hz` | touchdowns per second per post-grace env, both feet (clock cadence = 5.0) |
| `step/events_moving_per_s` | moving-command touchdowns per second, averaged over all envs |
| `step/len_err_abs`, `step/len_err_mean`, `step/len_err_std` | lead − s* over moving-command touchdowns |
| `step/len_gain` | Σ lead·s* / Σ s*², slope through the origin (like `cmd/vx_gain`) |
| `step/kernel_mean`, `step/rate_factor_mean` | mean kernel and rate factor at those events |
| `step/len_abs_idle` | mean \|lead\| at touchdowns under vx_cmd = 0 (unpaid, watched for drift) |

`step_length_track_enable=true` turns the detector and these keys on with the reward at 0. Use it to evaluate
S11 checkpoints on the same scale. The running S11 control process predates this code and logs none of them.

## 2. Design decisions

| decision | chosen | alternative not taken |
|---|---|---|
| keep or replace vel_tracking | **keep**. Footstep rewards in the literature (Duan et al. 2022, Cassie footstep-constrained walking; Walk-These-Ways' Raibert-heuristic term) supplement body-velocity tracking; feet alone can hit their marks while the body lags | replacing it |
| reference foot | **planted (other) foot** → anti-limp for free | same foot's last contact (2× signal, limp-blind) |
| event vs dense | **event**, as legged_gym's `feet_air_time`, with the rate cap against tapping | Walk-These-Ways-style dense Raibert target (needs a phase-dependent foot target) |
| cheaper alternative | not taken (user's call): replace the 1 s EMA with a one-gait-period (0.4 s) boxcar of body vx, ~10 lines, no detector | |
| weight | 100 per touchdown (user: not conservative) | 60 = parity with vel_tracking |
| σ | 2 cm. The control's vx error (0.080 m/s) equals 1.6 cm per step at 5 steps/s, so the kernel starts mid-slope | |
| fork point | S11_hist 12330: the line that flew on 10-11, deployable through `balance_s11` | S10b 12179 (plain task, no deploy script built) |

## 3. Verification

| check | result |
|---|---|
| `py_compile` of both files | pass |
| Isaac smoke 1 (64 envs, 4 iterations, from model_12330) | ball radius measured **20.00 mm**; resume, rollout and update ran; `step/td_rate_hz` 6.98 on a 64-env single-tick snapshot. Found the snapshot problem: `step/len_*` keys missing when the last tick had no moving-command touchdown → switched to decaying sums |
| Isaac smoke 2 (same) | all `step/*` keys present every iteration; `reward/step_length_mean` 0.17 on a nonzero snapshot. **Finding:** lead error 6–7 cm mean absolute, std 7–10 cm, and \|lead\| 6–7 cm even at vx = 0 (unpaid), far above the 1–6 cm targets |
| geometry check (smoke 3) | one-time print at the default pose, env 0, base frame: ball L x −83.7 / R x −87.3 mm, each 59 mm straight below its nail → **lead 3.6 mm** (negligible vs σ 20 mm). Per-foot signed lead means over 4 iterations: L +0.4 / +2.6 / +3.4 / +4.9 cm, R +0.1 / +2.8 / +3.0 / +4.6 cm, i.e. same sign, no L/R offset. The 6–7 cm scatter is the S11 gait itself, not a ball-offset bug (the risk `measure_ball_geometry`'s docstring names for the right foot) |
| smoke 3 last iteration (64 envs) | `td_rate_hz` 5.05 (clock 5.0), `rate_factor_mean` 0.97, `len_gain` 1.08, `len_err_abs` 0.062, `len_err_std` 0.078, `kernel_mean` 0.20 |
| defaults bit-exact | `rew_scale_step_length = 0` and `step_length_track_enable = false` skip the whole block (`_step_on` false); slot 42 is a zero column |

## 4. Launch facts

| | |
|---|---|
| run | `RL/logs/rsl_rl/p9_replant/2026-10-12_22-36-09_S12_steplen/`, PID 50028, GPU 1 |
| start | `2026-10-10_12-33-17_S11_hist/model_12330.pt` |
| length | `--max_iterations 1000` → 13330 |
| overrides | S11's full list verbatim plus `rew_scale_step_length=100.0`, `step_length_sigma_m=0.02`, `step_length_idle_frac=0.0`, `step_td_arm_m=0.006`, `step_td_land_m=0.002`, `step_td_min_air_s=0.05`. **A resume or eval must re-pass all of them** |
| control | S11_hist on GPU 0, same checkpoint. Its run ends at 12920, so matched-iteration TB data cover S12 up to +590 |
| S12 iteration 12330 (first, 4096 envs) | `step/td_rate_hz` 5.21, `len_err_abs` 0.068, `len_err_std` 0.088, `len_err_mean` +0.003, `kernel_mean` 0.165, `rate_factor_mean` 0.96, `len_abs_idle` 0.070, lead L/R +2.0/+2.4 cm; `reward/step_length_mean` 0.050. Episode length 124 and `vx_err_moving` 0.149 are the post-resume reset transient (every env starts in grace), not a reading |
| pace vs control | S11 is currently at ~270 s/iteration on GPU 0 (171 s at its own launch), S12 at 168 s, so S12 reaches the end of matched data (+590) about 17 h before S11 does |
| S11 reference, iterations 12281–12330 (50-iteration means) | `cmd/vx_err_moving` **0.080** (σ 0.007); `cmd/vx_gain` 0.818; `cmd/vx_cmd_abs_moving` 0.153; `cmd/pos_drift_idle_m` 0.274; `cmd/heading_err_abs_mean` 0.150; `curriculum/standup_success_high` 0.796; `sit/success_rate` 0.980; episode length 1746; `Loss/value_function` 3989; `reward/vel_tracking_mean` 1.84 |

## 5. Pre-registered predictions (10-12, before any S12 data beyond 12330)

All "at +N" reads are 50-iteration means ending at 12330+N; S11 is read over the same iterations.

| id | claim | supported if | contradicted if |
|---|---|---|---|
| P1 (engagement) | the detector counts real steps | first 10 iterations: `step/td_rate_hz` in [4, 6.5] and `step/rate_factor_mean` ≥ 0.8 | `td_rate_hz` < 3 or > 8 (latch thresholds wrong → K1) |
| P2 (learned) | the policy places feet to the target | at +200: `step/len_err_abs` ≤ 0.7× its first-10-iteration value | > 0.9× at +300 |
| P3 (user's hypothesis, evenness) | per-step placement gets more regular | at +300: `step/len_err_std` ≤ 0.7× its first-10-iteration value | > 0.9× at +300 |
| P4 (speed follows) | step accuracy at fixed cadence lowers speed error | at +400: `cmd/vx_err_moving` ≤ 0.9× S11's, and `cmd/vx_gain` closer to 1 than S11's | `vx_err_moving` > S11's at +400 |
| P5 (no regression) | the heavy weight does not cost the rest | at +400: `standup_success_high` ≥ S11 − 0.03, `sit/success_rate` ≥ S11 − 0.02, episode length ≥ 0.95× S11, `heading_err_abs_mean` ≤ S11 + 10 %, `pos_drift_idle_m` ≤ S11 + 10 % | any bar missed |

How to read the outcome grid:

- P2 + P3 + P4 supported: footstep scoring makes the speed both more even and more accurate. Next: build the
  matched eval (§7, item 1) and confirm on deterministic rollouts before any flight.
- P2 + P3 supported, P4 contradicted: the feet obey but the body does not follow. The step term and the EMA term
  disagree, and the 1 s EMA (or the fixed clock) is the binding constraint.
- P2 contradicted: the term is not learnable at this σ/weight (or the detector is noisy). Check P1 first.
- P5 missed with P2 supported: 100 per touchdown is too heavy. Rerun at 60 from the same fork.
- Neither P2 outcome (0.7–0.9×): inconclusive; carry on to +500.

## 6. Kills

| id | condition | action |
|---|---|---|
| K1 | P1 contradicted | stop; fix the latch thresholds against the measured ball-bottom trace |
| K2 | `step/rate_factor_mean` < 0.7 for 50 iterations (tapping exploit despite the cap) or `step/td_rate_hz` > 7 | stop, inspect gait |
| K3 | `Loss/value_function` > 20 000 (5× S11's 3 989) for 20 consecutive iterations, or episode length < 0.85× S11's for 100 iterations | stop, inspect |
| K4 | `standup_success_high` < 0.70 for 100 iterations | stop, inspect |

## 7. Not built

1. **Matched step evaluation.** A deterministic rollout of S11 vs S12 checkpoints at fixed vx (0.05 / 0.15 / 0.30)
   with `env.step_length_track_enable=true`, reporting per-env stride-to-stride lead spread and stride-averaged vx
   spread. The TB `step/len_err_std` mixes envs and commands, so it is not the clean evenness number.
2. **Deploy:** none needed for the reward. An S12 checkpoint exports through `RL/scripts/export_hist_policy.py`
   like S11 and flies on `control/balance_s11.py` (same obs and ring contract).
