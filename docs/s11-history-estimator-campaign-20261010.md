# S11 campaign: long-history conv encoders + supervised and self-supervised latent (2026-10-10)

**Status:** training launched 2026-10-10 12:33 on GPU 0 (RTX 4090), run
`RL/logs/rsl_rl/p9_replant/2026-10-10_12-33-17_S11_hist/`, launcher `RL/launch_train_S11_hist.ps1`.
S10a_rate was stopped at model_11420 to free the GPU. **Control:** S10b_heading, still running on GPU 1 from
the same model_11420. Nothing in `control/` deploy code has changed (§8).
**Decision source:** the 10-09/10 design discussion on history length, estimators and network shape (unpublished),
on top of `docs/future-campaign-options-20261009.md` §9. User decision 10-10: skip the offline probe H-C0, use a window
longer than 0.5 s if Pi latency allows, add the 1D CNN, add self-supervised prediction, resume from the latest
S10b_heading.

## 0. Summary

| item | result |
|---|---|
| architecture | S10's 5×36 fast stack unchanged, plus two 1D-conv encoders: **mid** 0.5 s at 100 Hz → 16-d, **slow** 2.5 s at 20 Hz → 20-d. Actor input 180 → 216, same 512-256-128 MLP |
| supervised latent | 19 sim-truth targets: body velocity, foot contact, smoothed heading-frame velocity, gyro / gravity / joint-zero bias draws |
| self-supervised head | predicts the next 80 ms of the 18 sensor channels from (z, a_t); training-only, saved in every checkpoint |
| resume | zero-column surgery on S10b model_11420: **max \|Δaction\| = 0.0** at iteration 0, Adam state carried over |
| Pi latency (Pi 5, measured) | forward p50 0.60 ms vs 0.17 ms for the S10 actor. A 100 Hz loop including the ring update ran p99 0.80 ms, max 1.70 ms, **0 of 2000 ticks over 10 ms**. Estimated tick ≈ 3.2 ms of 10 → **go** |
| training cost | 171 s/iteration (158.8 s collection + 12.3 s learning) vs S10b's 173 s. 1500 iterations ≈ 3 days |
| verification | CPU tests pass; the Isaac smoke and every live update check the rebuilt history ring against the env's ring: **error 0.0** |
| open | deploy side not built (§8). Matched control data end at S11 +760 (S10b's own run ends at 12180) |

## 1. What was built

```
fast  5 × 36 @100 Hz   (obs "policy", unchanged)  ─── normalizer ──────────────────┐
mid  50 × 24 @100 Hz   (newest 50 of the ring)    ─ hist norm → enc_mid  → z_mid  16├→ 216 → 512 → 256 → 128 → 6
slow 50 × 24 @ 20 Hz   (ring, 5-frame means)      ─ hist norm → enc_slow → z_slow 20┘
                         ring = obs "hist": 250 frames × 24 ch, oldest → newest
training only:  z_mid sup (5), z_slow sup (14)  ← obs "aux" (19 sim-truth targets)
                predictor(z 36, a_t 6) → Δ sensors over 4 bins × 20 ms (80 ms)
```

| file | role |
|---|---|
| `RL/source/RL/RL/tasks/direct/standup_curriculum/hist_policy.py` | `HistEncoder`, `ActorCriticHist`, `HistRolloutStorage`, `PPOHist`, `HistPolicyExport` (TorchScript deploy module), layouts and mirror tables. Importing it registers the classes with rsl_rl's runner (`eval(class_name)`) |
| `rl_env_p6.py` `_hist_observations` | ring update (same replicate-first-frame rule as the 5-frame stack) and aux targets. Gated by `hist_enable`; **False = bit-exact S10** |
| `rl_env_cfg_p6.py` | `hist_enable`, `hist_ring_frames=250`, `hist_aux_vel_ema_tau_s=0.4`; `RlEnvHistCfg` |
| `agents/rsl_rl_ppo_cfg_p6.py` | `RslRlHistActorCriticCfg`, `RslRlHistPpoAlgorithmCfg`, `PPORunnerHistCfg` |
| `__init__.py` | task **`Template-standup_curriculum-hist-v0`**. Play and eval scripts must use this id to get the ring |
| `RL/scripts/extend_actor_hist_checkpoint.py` | surgery + equivalence check + optional TorchScript export (runs without Isaac) |
| `RL/scripts/test_hist_policy.py` | CPU tests: mirror involutions, exact ring rebuild across resets, future-frame validity, update end-to-end, export parity, grad-accumulation equivalence |
| `RL/scripts/bench_hist_update.py` | update time and peak CUDA memory at training scale, PPO vs PPOHist |
| `control/hist_latency_bench.py` | Pi latency bench (TorchScript S10 vs S11, ring bookkeeping, 100 Hz loop) |
| `RL/launch_train_S11_hist.ps1` | the arm. A diff against the S10b launcher shows only the S11 lines |

**Encoder** (`HistEncoder`, about 21 k parameters each): 1×1 conv 24→32, Conv1d(k8, s4), Conv1d(k5),
Conv1d(k5). The time axis shrinks 50 → 11 → 7 → 3. The 3 positions are flattened (keeping a coarse early /
middle / recent order) into two linear heads: supervised and free. Free dims pass through tanh. Contact logits
are squashed to (−1, 1) for the actor. Parameters: actor 276 k, encoders 42 k, predictor 31 k, total 647 k
(including the unchanged critic).

**Ring frame (24 ch):** delayed pg 3, gyro 3, joint pos 6, joint vel 6, plus the fresh last action 6. The dead
lin_vel (dropout 1.0) and the bookkeeping already in the fast path are dropped.

**Aux targets (obs group `aux`, layout in `hist_policy.py`):**

| dims | target | head | loss | scale |
|---|---|---|---|---|
| 0:3 | base lin vel, body frame | z_mid sup | MSE | 0.3 m/s |
| 3:5 | foot contact L/R (toe z < `no_flight_contact_threshold`, the reward's own rule) | z_mid sup | BCE | — |
| 5:7 | heading-frame vx, vy, causal EMA τ 0.4 s (one stride) | z_slow sup | MSE | 0.15 m/s |
| 7:10 | gyro bias draw | z_slow sup | MSE | 0.05 rad/s |
| 10:13 | projected-gravity bias draw | z_slow sup | MSE | 0.05 |
| 13:19 | joint-pos (encoder zero) bias draw | z_slow sup | MSE | 0.07 rad |

**Prediction target:** for bins i = 0..3, the mean of the frames at t+2i+1 and t+2i+2, minus the frame at t,
over the 18 sensor channels, divided by the hist normalizer std. Bins that cross an episode reset or the
rollout end are masked. The metric is **skill = 1 − MSE / MSE_persistence**, with persistence predicting Δ = 0.

## 2. Design decisions, including where the build departs from the discussion

1. **Window: 0.5 s at 100 Hz plus 2.5 s at 20 Hz, from one 250-frame ring.** This is the multi-rate design from
   the last turn of the discussion. Both views come from the same ring by slicing and a reshape-mean, so deploy
   holds one 250×24 array and no pooling-phase state.
2. **The encoders learn only from the aux losses; the actor reads z detached.** The discussion let PPO shape
   the free dims. In this build:
   - the free dims are shaped by the prediction loss instead;
   - the encoders and predictor have their own Adam group at a fixed lr of 5e-4, independent of the
     KL-adaptive actor lr (S10b sat at 3.9e-5 to 1e-3);
   - `v_est` stays a pure estimate, which is what the hardware video check (H-C1) and any later real-data
     recalibration need.

   PPO learns to *use* z through the actor's new first-layer columns.
3. **The predictor is conditioned on a_t only, not a_t..a_{t+7}.** In closed loop a_{t+k} = π(obs_{t+k}),
   and obs_{t+k} contains the very frames being predicted. Feeding future actions lets the predictor invert
   the policy instead of using z. With a_t only, the head forecasts the closed-loop future. The policy is part
   of the plant in sim and on hardware alike, so a sim-vs-real skill gap still measures the plant.
4. **Ring priming = replicate the first frame**, the same rule the 5-frame stack already uses. Under the blind
   `v` contract that first frame has zero sensor channels. The 2.5 s ring is therefore partly blind for 2.5 s
   after every reset, exactly as training sees it. Deploy must reproduce this (§8).
5. **Rollout storage does not store the ring.** 32768 envs × 128 steps × 2400 floats of views would be about
   40 GB. `HistRolloutStorage` keeps one sequence S [N, R+T−1, 24]: the ring at step 0, then each step's newest
   frame (1.2 GB). It rebuilds any sample's ring by indexing. An episode that starts inside the rollout is
   re-primed by clamping indices at its first frame. Every update checks the rebuild against the env's true
   ring for 32 envs and raises on any mismatch (`hist_verify_envs=32`). Observed error: 0.0.
6. **Memory: two aborted launches, then fixed with no recipe change.** The first launch spent 485 s in the
   update and the second 191 s, against S10's 13 s. The 24 GB card was
   already near full: the S11 process held about 20 GB during collection, before the update's transients. On
   Windows the overflow spills into shared system memory rather than raising OOM. Three fixes, all mathematically neutral:
   - latents computed chunk by chunk, keeping only z;
   - the mirror-loss actions taken from `act()`'s distribution mean, the same actor on the same input, instead
     of a second graph;
   - 2-way gradient accumulation per minibatch (`grad_accum=2`): the same 32 minibatches, KL pooled before
     the lr adaptation, one clipped step.

   `test_hist_policy.py` shows accumulation equals the plain update to 9.9e-8 per parameter.
   `bench_hist_update.py` at full scale gives an update peak of 9.13 GiB, against 11.00 GiB for S10's PPO.
   The aborted run directories are renamed `*_ABORTED_memspill{,2}`; each holds one iteration and should not
   be used.
7. **Slow-path supervised dims widened from 8 to 14.** The discussion sized z_slow at 8. Supervising the gyro,
   gravity and joint-zero bias draws makes the slow path's job measurable: an R² per bias says whether 2.5 s of
   history recovers it.
8. **`dr_proj_grav_bias_max` 0.00 → 0.05.** The discussion called for this: without it the slow path has no
   trim to learn, and trim is what S8b is sensitive to. 0.05 is a quarter of the level where S8b's pinned
   sweep toppled (0.20). **This confounds the drift comparison with S10b slightly**, against S11: its task is
   a little harder.
9. **`cmd_range_ramp_steps` 51200 → 0.** `_cmd_range_frac` reads `common_step_counter`, which restarts at 0 on
   resume. The S10 ramp would have shrunk the command ranges to 0.33× for 400 iterations, while S10b trains at
   full range.
10. **H-C0 skipped** (user decision). Its information question is answered online instead: `Loss/r2_*` is
    measured on fresh data in each update's first epoch, before that batch's step.

## 3. Verification

| check | result |
|---|---|
| surgery: S10b 11420 vs extended model, 4096 obs drawn from normalizer stats + random rings | max \|Δaction\| **0.0**. Traced TorchScript S10 vs S11 also 0.0 |
| Adam state | group 0 = the parent's 17 parameter tensors (std, actor 8, critic 8) in their original order (actor.0.weight moments padded: exp_avg 0, exp_avg_sq = per-row mean of the old columns); group 1 = new modules, empty state |
| `test_hist_policy.py` (CPU) | ALL PASS: involutions; ring rebuild exact over 22 in-rollout resets; future-frame validity; views identical; update trains the aux group and grows the latent columns; export parity; grad_accum 2 ≡ 1 |
| Isaac smoke (64 envs, 4 iterations, every env verified) | exit 0; rebuild error 0.0 every update; contact accuracy 0.96, prediction skill 0.3–0.5 after 4 iterations |
| live run, first iteration | rebuild error 0.0; 24.5 k steps/s (S10b 24.2 k); learning 12.3 s (S10b 12.8 s) |

## 4. Pi 5 latency (measured 10-10, `data/s11_pi_latency_20261010/bench.json`)

Pi 5, PREEMPT_RT kernel, torch 2.10.0, TorchScript traced from the surgery checkpoint, 5000 calls per cell.

| | 1 thread p50 / p99 | 4 threads p50 / p99 / max |
|---|---|---|
| S10 actor (180 → 6) | 0.19 / 0.22 ms | 0.17 / 0.22 / 0.30 ms |
| S11 actor (180 + ring 6000 → 6, + sup 19) | 0.61 / 0.70 ms | 0.60 / 0.67 / 1.35 ms |
| ring update, `np.roll` 250×24 | 0.033 / 0.044 ms (one 6 ms outlier in 5000) | |
| ring update, double-write buffer (no roll) | 0.012 / 0.013 ms (max 0.046) | |
| 100 Hz loop for 20 s (double-write ring + S11 forward, 4 threads) | p50 0.70 / p99 0.80 / max 1.70 ms; **0** ticks over 10 ms, **0** over 3 ms | |

Against the 09-30 flight tick (busy p50 2.71 ms, p95 2.99 ms), S11 adds about 0.45 ms: estimated p50 ≈ 3.2 ms
of the 10 ms budget. **Verdict: go.** Deploy should use the double-write ring; `np.roll` showed one 6 ms stall.

## 5. Launch facts

| | |
|---|---|
| run | `RL/logs/rsl_rl/p9_replant/2026-10-10_12-33-17_S11_hist/`, PID 59832, GPU 0 |
| start | surgery checkpoint `2026-10-10_11-56-52_S11_hist_surgery/model_11420.pt` (= S10b 11420) |
| length | `--max_iterations 1500` → 12920; about 3 days at 171 s/iteration |
| overrides | S10b's full list verbatim except: task id, run name, load_run/checkpoint, `dr_proj_grav_bias_max=0.05`, `cmd_range_ramp_steps=0`, the S11 agent lines (`aux_learning_rate 5e-4`, `aux_coef_sup 1.0`, `aux_coef_pred 0.5`, `aux_samples 16384`, `grad_accum 2`, `hist_verify_envs 32`). **A resume or eval must re-pass all of them** |
| control | S10b_heading, same start (11420), GPU 1. Its run ends at 12180, so matched-iteration TB data cover S11 up to +760 |
| S10b reference, iterations 11371–11420 (50-iteration means) | `cmd/pos_drift_idle_m` 0.271 (σ 0.010); `cmd/vx_err_moving` 0.083; `cmd/heading_err_abs_mean` 0.135; `curriculum/standup_success_high` 0.797; `sit/success_rate` 0.981; episode length 1751 |

New TB tags (all under `Loss/`):
- `aux_sup`, `aux_pred`;
- `r2_v_body_xy`, `r2_v_body_z`, `r2_v_smooth_xy`, `r2_gyro_bias`, `r2_pg_bias`, `r2_jpos_bias`;
- `contact_acc`;
- `pred_skill_{pg,gyro,jpos,jvel}`, `pred_skill_bin{0..3}`;
- `z_col_norm` (Frobenius norm of the actor's 36 latent columns) and `obs_col_norm` (the 180 obs columns:
  82.47 at start);
- `hist_verify_err`.

## 6. Pre-registered predictions (10-10, before any S11 data beyond iteration 11420)

All "at +N" reads are 50-iteration means ending at 11420+N. S10b is read over the same iterations.

| id | claim | supported if | contradicted if |
|---|---|---|---|
| P1 (built-in H-C0) | the 2.5 s history carries stride-scale velocity | at +300: `r2_v_smooth_xy` ≥ 0.5 **and** `r2_v_body_xy` ≥ 0.5 | `r2_v_smooth_xy` < 0.2 at +300 (→ K1) |
| P2 | the slow path recovers sensor biases | at +500: `r2_gyro_bias` ≥ 0.5 and `r2_pg_bias` ≥ 0.3 | either < 0.1 at +500. `r2_jpos_bias` is reported, no threshold (expected < 0.3: weakly observable) |
| P3 | the latent carries dynamics | at +300: `pred_skill_bin3` ≥ 0.5 | < 0.3 at +300 |
| P4 | PPO uses z | at +500: `z_col_norm` ≥ 3.7, i.e. the per-column RMS of the latent columns ≥ 0.1× that of the obs columns (82.47/√180 = 6.15) | `z_col_norm` < 1.0 at +500 |
| P5 (H-C1, behaviour) | better estimation lowers idle drift | at +500: S11 `cmd/pos_drift_idle_m` ≤ 0.6× S10b's, **and** no regression: `standup_success_high` ≥ S10b − 0.03, `sit/success_rate` ≥ S10b − 0.02, episode length ≥ 0.95× S10b, `vx_err_moving` and `heading_err_abs_mean` ≤ S10b + 5 % | drift > 0.85× S10b's at +500 |

**How to read the outcome grid:**

- P1 and P5 supported: estimation was the drift problem.
- P1 and P4 supported, P5 contradicted: the policy has the estimate and uses it, yet drift stays, so the cause
  is reward or plant (future-campaign-options §7.2, H-B).
- P1 supported, P4 contradicted: PPO has not picked z up. Run longer, or retry the surgery with tiny random
  (std 1e-3) instead of zero columns.
- P1 contradicted: the information is not in 2.5 s of proprioception; stop (K1).
- Neither P1 outcome (0.2 ≤ R² < 0.5): inconclusive; carry on to +500 and judge on P5.

## 7. Kills

| id | condition | action |
|---|---|---|
| K1 | `r2_v_smooth_xy` < 0.2 at +300 | stop; drift goes to reward or plant (H-B) |
| K2 | `Loss/value_function` > 18 500 (5× S10b's 3 700) for 20 consecutive iterations, or episode length < 0.85× S10b's at matched iterations for 100 iterations | stop, inspect |
| K3 | `standup_success_high` < 0.70 for 100 iterations | stop, inspect |
| K4 | any `[PPOHist] ring reconstruction` error (the run raises by itself) | fix before resuming |
| K5 | P5 contradicted at +750 with P1 and P4 supported | stop: reward problem, not observation |

## 8. Not built yet (deploy side and tools)

**Update 2026-10-11:** items 1–3 are built (commit c53e7aa): `control/balance_s11.py` (flight script, also covers
the S9 §7 sit and the S10 §8 heading/vx items), `control/s11_contract.py` (ring, bookkeeping, guard, self-tests),
`RL/scripts/export_hist_policy.py` (TorchScript + embedded contract.json + parity fixture). The static gyro
capture is a 2 s HOLD pre-roll saved into every run's npz. Items 4–6 remain open.

Required before an S11 checkpoint can fly. None of it is built, and the S10 deploy items (S10 doc §8) are also
still open:

1. **Deploy ring:** a 250×24 float32 double-write buffer in `balance_log.py` / `controller.py`, fed the same 24
   channels: delayed pg, gyro, joint pos, joint vel, plus the last action, in the training units and signs. Prime it by
   replicating the first frame at handover, which under `v` is the blind frame. Export with `HistPolicyExport`
   (`extend_actor_hist_checkpoint.py --export-dir` already traces it); its signature is
   `(obs [1,180], ring [1,6000]) → (action, sup[19])`.
2. **Contract check:** `policy_contract_check.py` must check ring width 6000, the oldest→newest order, priming
   and the per-frame mirror table. A reversed time axis would not crash; it would silently ruin the policy.
3. **Logging:** log `sup[19]` (`v_est` = sup[0:3], the contact probabilities, the bias estimates) and the raw
   ring frame each tick, plus a sync tap per flight and a 1–2 s static gyro capture before stand-up (the
   discussion's prep for the video check and for any later real-data recalibration).
4. **Offline replay / skill script:** run the exported encoders and predictor over a flight's logged ring and
   actions, then compare prediction skill per channel and gait phase, sim vs real. This is the diagnostic
   the self-supervised head exists for.
5. **Play scripts:** they work unchanged if launched with task `Template-standup_curriculum-hist-v0`. The
   policy fills `policy.last_sup_phys` on every call, so a simlog can record `v_est`, but no play script logs
   it yet.
6. **Control extension:** S10b ends at 12180 (about 01:00 on 10-11). To keep a matched control past S11 +760,
   resume S10b from 12180 with `cmd_range_ramp_steps=0` on GPU 1. Not launched; it's the user's call.
