# S10 campaign: command tracking (vx, yaw rate vs heading) + S9 spawn fixes (2026-10-07)

**Status:** training launched 2026-10-07 21:54, two arms, both 4090s. Nothing in `control/` has changed.
**Base policy:** S9b_band `model_10180.pt` (`RL/logs/rsl_rl/p9_replant/2026-10-04_15-37-07_S9b_band/`), the last
S9b checkpoint. S9 was killed at marginal return; consecutive checkpoints differ little.
**Inputs:** `docs/s9-play-review-20261007.md` (spawn / reset findings F1–F4) and the user's 10-07 viewer
observation: *vx commands do nothing; yaw commands work, but sluggishly.*

## 0. Summary

| item | finding | in S10 |
|---|---|---|
| vx does nothing | Not a dropout problem. The trained vx range is ±0.10 m/s. Standing still under a 0.10 m/s command loses 0.24 reward/step (4 % of `alive`). The command sits below the gait's own drift (S9b: mean \|vx − cmd\| 0.15 m/s). | range [−0.15, +0.30], σ 0.15 → 0.10, scale 0.67 → 3.0, ramp |
| yaw sluggish | Trained range ±0.10 rad/s (5.7°/s), non-zero in only 25 % of segments, σ 0.30. Standing still under the largest command keeps 89 % of the reward. Sluggishness was what the policy was trained to do. | range ±0.40, σ 0.15, scale 3.0, ramp |
| lin_vel dropout 1.0? | Correct. The critic sees the true body velocity, so the vx gradient exists without it. 50 % would make "0" ambiguous on hardware (§2.2). | unchanged (1.0) |
| steering by angle? | Yes, at the operator interface: the legged_gym `heading_command` pattern. Obs layout is unchanged (ch33 stays rad/s; ch34/35 already carry a heading error). No weight surgery. | **the A/B**: S10a rate vs S10b heading |
| F1 spawn penetration | built: per-env lift from the FK of the noised spawn | both arms |
| F2 stale rest rows | built: s = 0.40 / 0.45 rest columns re-measured on the hip-reaction plant | both arms |
| F3 / F4 play contract | built in `play_alpha.py` (R key = one obs call; `-SettleS` releases blind) | play only |
| settle-before-handover | deferred, as the review recommended: re-probe on top of F1+F2 first (§7.2) | — |
| sit end state | deploy-side NN → PD-HOLD gate (review §5), not built | — |

## 1. Why vx and yaw did not work (S9b, measured and computed)

### 1.1 The reward landscape

Tracking terms (`compute_rewards_vx_yaw`): `scale · exp(−(avg − cmd)² / σ²)`, where avg is the 1 s EMA of the
body velocity. The vx term is live on every post-grace step. The yaw term is gated on |cmd| ≥ 0.05.

| | S9 range | sampled non-zero | σ | scale | reward lost by standing still at max command |
|---|---|---|---|---|---|
| vx | ±0.10 m/s, \|vx\| < 0.02 → 0 | 40 % of segments | 0.15 | 0.667 | 0.24 / step (4 % of alive 6.0) |
| yaw rate | ±0.10 rad/s, \|ω\| < 0.05 → 0 | 25 % (half in [0.05, 0.10]) | 0.30 | 2.0 | 0.21 / step |
| vx, S10 | [−0.15, +0.30], \|vx\| < 0.05 → 0 | 50 % | 0.10 | 3.0 | 1.90 at 0.10, 3.0 at 0.30 |
| yaw, S10 | ±0.40 | 50 % | 0.15 | 3.0 | 1.08 at 0.10, 3.0 at 0.40 |

What competes with it (S9b TB, mean of the last 100 iterations): `alive` +6.0, `gait_ref` +1.17,
`position_anchor` −1.00, `gait_step` +0.99, `clock_contact` +0.72, `stance_symmetry` −0.61. The robot's own
noise floor: instantaneous |ω_z| at zero yaw command 0.49 rad/s (the point-foot swing reaction); mean
|vx − cmd| 0.15 m/s; mean anchor drift 0.33 m. The whole command range lay inside the noise of the limit
cycle, and following it was worth less than any of the gait terms. The policy learned to ignore it.

Two smaller defects, also fixed:

- For 0.02 ≤ |vx| < 0.05 the command gate was partial: the position anchor still pulled the robot back
  while the tracking reward asked it to move. S10 zeroes anything below 0.05 (`vx_cmd_min_abs`).
- `back_leg_step` paid the rear leg for stepping **forward** under any non-zero command, including a
  backward one. S10 expresses it along the command direction (`back_leg_step_signed`).

### 1.2 lin_vel dropout: keep 1.0

- **The gradient does not come from the actor's input.** The actor learns to respond to `vx_cmd` (ch32)
  through advantages, and those come from the reward and the critic. The critic gets the true body
  velocity (`_get_critic_observations` [0:3], plus the whole policy obs). With dropout 1.0, the value of
  tracking is learnable and the actor still gets a vx gradient. S9 had no usable gradient because the
  reward was flat (§1.1), not because the input was missing.
- **50 % would make zero ambiguous.** `lin_vel_keep` is in the critic obs only, so the actor cannot tell
  "dropped" from "measured 0 m/s". The hardware always feeds 0. A policy that learned to use the channel
  half the time reads that 0 as "I am standing still" while it walks, and that half of its behaviour pushes
  toward the command forever. The dropped half does cover the hardware case, but the shared weights pay
  for the sighted half.
- **What others do.** Either blind proprioception with a privileged critic (what we do), or an explicit
  velocity estimator trained alongside the policy and fed to it (Ji et al. 2022, "concurrent
  estimator"). Partial dropout is not a standard substitute for either.
- **The real limit to watch.** The actor history is 5 frames = 50 ms. That is short for estimating velocity
  implicitly, so the policy tracks vx mostly open-loop, through step length. If S10 tracks the *mean*
  command but drifts within a segment, the next lever is a longer history or an estimator head. Both
  change the obs width (net surgery). Not in S10.

### 1.3 What we can ask of this robot

The robot is a 6-DOF point-foot biped, ~0.40 m leg, hips 0.16 m apart, with no hip-yaw joint and a fixed
2.5 Hz gait clock (5 steps/s; `gait_freq` is not a command).

- **vx.** Speed comes from step length only: v = 5 · ℓ. 0.30 m/s is a 6 cm step (15 % of leg length),
  Froude number v²/(gL) = 0.023. Human walking is ~0.25. Kinematically easy. The binding limits are
  balance on point feet with 40–50 ms of loop delay, and the knee belt. Backward is capped at 0.15
  because the policy idles in a forward-pitched crouch. The deploy RC already maps −0.15 … +0.40
  (`control/config.py RC_VX_*`), so the hardware was asking for 4× the trained range.
- **yaw.** With no yaw joint and a point contact, yaw comes from (a) double support with differential hip
  pitch (the pelvis rotates about the two contact points) and (b) the reaction to the swing leg. That
  second mechanism is also the 0.49 rad/s yaw noise. 0.40 rad/s at 5 steps/s is 0.08 rad per step, a
  1.3 cm fore-aft foot differential across 0.16 m. Feasible. A 50° turn then takes ~2.2 s plus settling.
- **Not drone-like.** The robot can't step-change velocity or heading: every change is paid in steps,
  ~0.2 s each, and must keep the CoM over a point contact. The realistic bar: the 1 s-averaged velocity
  within ~0.05 m/s of the command, the heading within ~5° once settled, no rise in falls. Anything
  faster or tighter would be a new campaign (gait-frequency command, longer history).

### 1.4 Rate vs heading steering

**Recommendation: steer by heading at the operator interface. That is the A/B arm.** It is the standard
pattern: legged_gym `heading_command` and Isaac Lab `UniformVelocityCommand(heading_command=True)`. A
heading target is sampled, and every step the yaw-rate command is computed as
`clip(k · wrap(target − yaw))` (legged_gym uses k = 0.5, ±1 rad/s). The policy input stays a rate
command, so the network does not change.

This env already feeds the policy a heading error: ch34/35 = sin/cos(yaw − `_spawn_yaw`), where
`_spawn_yaw` is the reference the heading anchor holds. In heading mode that reference *is* the target:

| | S10a `yaw_cmd_mode=rate` (legacy) | S10b `yaw_cmd_mode=heading` |
|---|---|---|
| segment start (every 10 s) | sample ω ∈ ±0.40·f, hold it | sample Δψ: 50 % zero, else U[0.15, 1.57] rad with a random sign; target = yaw + Δψ |
| ch33 (yaw_rate_cmd, rad/s) | the sampled ω | clamp(1.0 · wrap(target − yaw), ±0.40·f), every step |
| ch34/35 | yaw since the segment start (an integrator; meaningless while turning) | sin/cos(yaw − target): the error itself |
| heading anchor | on when ω = 0 (and vx = 0) | on within 0.05 rad (2.9°) of the target (and vx = 0) |
| walking straight | yaw-drift penalty only | any drift > 2.9° becomes a rate command again: closed loop |
| deploy | as today: RC stick → rate | stick moves the target; ch33/34/35 computed from IMU yaw (§8) |

The channel count, units and mirror signs are unchanged, so S10b resumes from the same weights with no
surgery. The *meaning* of ch34/35 while turning changes (error instead of integral). The network has seen
values of the same size: 0.1 rad/s × 10 s = 1 rad in S9's rate segments.

**Difficulty prediction** (the workflow rule):

- *Common?* Yes. Heading commands, command curricula and asymmetric critics are all standard. Nothing here
  is invented.
- *Cheaper workaround?* For heading, the deploy side could convert a stick angle into a rate profile
  without retraining. That leaves the loop open: IMU yaw drift and the 0.49 rad/s swing reaction go
  uncorrected, and that is the sluggish behaviour the user saw.
- *Cost:* ~0.5 day of env code (done) plus a deploy change for the heading arm (§8, ~0.5 day, not built).

## 2. S9 review fixes (item 1)

| # | fix | where | default |
|---|---|---|---|
| F1 | `spawn_lift_enable`: after all spawn noise is written, the env reads the FK body poses (the write invalidates them; `body_link_pose_w` recomputes) and lifts the root so the lowest collision point sits 1 mm above the floor. Support points: every collision mesh under every body, reduced to extreme points along 192 directions (116–191 per body; `imu` has none). The lift never lowers a spawn. The init_h obs (ch31) keeps the table value. | `rl_env_p6.py` `_measure_collision_support`, `_spawn_penetration_lift` | off |
| F2 | `standup_path_rest_preset=hip_reaction_1007`: s = 0.40 rest hip/knee/pitch 1.261/−2.695/−6.2° → **1.393/−2.710/−12.9°**, s = 0.45: 1.039/−2.630/+2.7° → **1.172/−2.624/−5.2°** (S9 probe Block A medians). Target columns, root_h, ch29/30/31 and the deploy path are untouched. The F1 lift takes up the settled pose's extra height (0.211 vs 0.197 m at s = 0.40). | `_PATH_REST_PRESETS` | "" |
| F3 | R key: `obs, _ = env.reset()`, one obs call as in training; viewer start re-blinded the same way | `play_alpha.py` | — |
| F4 | `SettleHold.end()` zeroes the delay buffer and action memory and re-inits the history, so the release is blind like the rig's `v` | `play_alpha.py` | — |

The F1 self-check prints at startup. At the default pose of env 0, the lowest support point is +4.07 mm on
`nail___fillet1`. The probe's sphere-fit ball bottom on the same foot is +3.92 mm. The two agree to
0.15 mm, so the mesh frames and scale are right.

Not built, on purpose:

- **Settle-before-handover (option 1/2):** the review recommended re-probing after F1 + F2 first. F2 makes
  the noise-free spawn the settled crouch, which is most of what the hold was for. The joint noise (0.18 ×
  (1 − 0.8 d)) is unchanged: the review's H4 says the policy is currently tuned to it, and S10 changes one
  thing at a time on the stand-up side.
- **Sit end-state gate:** deploy side (review §5), with the rest of the sit deploy work.

## 3. What was built (all defaults bit-exact for older recipes)

`rl_env_cfg_p6.py` §5 (commands) and the S10 spawn block after `sit_*`:

| key | default | S10 |
|---|---|---|
| `vx_cmd_min_abs` / `yaw_rate_cmd_min_abs` | 0.02 / 0.05 (the old hardcoded values) | 0.05 / 0.05 |
| `cmd_range_ramp_steps` / `cmd_range_start_frac` | 0 / 1.0 (off) | 51200 (= 400 iterations) / 0.33 |
| `yaw_cmd_mode` | "rate" | arm line |
| `heading_cmd_kp`, `heading_cmd_delta_min/max_rad` | 1.0, 0.15 / 1.5708 | same |
| `back_leg_step_signed` | False (× 1.0, bit-exact) | True |
| `spawn_lift_enable` / `spawn_lift_clearance_m` | False / 0.001 | True / 0.001 |
| `standup_path_rest_preset` | "" | hip_reaction_1007 |

The ramp scales the sampled range by f = 0.33 → 1 linearly in `common_step_counter`, which starts at 0 at
process start, so a resume restarts the ramp. At f = 0.33: vx ∈ [−0.05, 0.10], and the 0.05 dead band
zeroes every backward command until f ≈ 0.4. The yaw cap is 0.13 rad/s.

New TB keys (population snapshots at one step, the S9 §3.5 caveat): `cmd/vx_gain` and `cmd/yaw_gain` (the
slope of the 1 s-averaged response on the command, over active envs with |cmd| ≥ 0.05: 1 = follows,
0 = ignores), `cmd/vx_err_moving`, `cmd/yaw_err_turning`, `cmd/heading_err_abs_mean`,
`cmd/pos_drift_idle_m` (anchor drift on zero-command envs: the "settle" signal, since
`diagnostics/pos_drift_m_mean` now mixes in walking envs), `cmd/range_frac`, `spawn/lift_mm_{mean,max}`,
`spawn/lifted_gt1mm_frac` (last reset batch).

**Smoke** (64 envs, 8 iterations, resample 200 steps, ramp 512 steps, both arms, `-Smoke`): exit 0, empty
stderr, resume at 10180 clean, `[S10]` lines as configured, ramp reaches 1.0 at iteration 4, lifts 0–26 mm
(the review's p01 penetration was −22.5 mm), heading and rate commands both live, `back_leg_step` live.
This is a code-path check only.

## 4. Arms

Both arms resume S9b 10180 and re-pass S9b's full override list verbatim: 2000 Hz sextet, hip reaction,
α pinned 0.6, grace band U[1.667, 5], grace smoothness, sit branch, every DR band. The launchers differ only
in the yaw-mode lines (`diff RL/launch_train_S10a_rate.ps1 RL/launch_train_S10b_heading.ps1`).

| | S10a_rate | S10b_heading |
|---|---|---|
| launcher | `RL/launch_train_S10a_rate.ps1` | `RL/launch_train_S10b_heading.ps1` |
| GPU | 0 (RTX 4090) | 1 (RTX 4090 D) |
| yaw | `yaw_cmd_mode=rate` | `yaw_cmd_mode=heading`, kp 1.0, \|Δψ\| U[0.15, 1.57] |
| shared | F1 + F2, vx/yaw ranges + tracking, ramp, signed back-leg | same |
| `--max_iterations` | 2000 (to 12180) | 2000 |

## 5. Pre-registered predictions (written before any full-run data)

S9b reference (last 100 iterations): `curriculum/standup_success_high` 0.70, advance metric
(success_high_rel) 0.96, `sit/success_rate` 0.97, `Train/mean_episode_length` 1375, `Loss/value_function`
5071. The command-following baseline is **each arm's own first 5 iterations** (the policy is still ≈ S9b
10180, at f ≈ 0.33).

| id | prediction | read at | signal |
|---|---|---|---|
| P1 | startup prints both `[S10]` lines as launched, the preset line, and a lift self-check within 1 mm of a ball-fit bottom | launch | log grep |
| P2 | S9b baseline does not follow: `cmd/vx_gain` < 0.3 over the first 5 iterations. If it is higher, the user's "vx does nothing" is not a sim property and the play-side command path needs a look before anything else is read | +5 | TB |
| P3 | vx is learned: `cmd/vx_gain` ≥ 0.5 at +400 (ramp end) and ≥ 0.7 at +1000, with `cmd/vx_err_moving` ≤ 0.08 m/s at +1000, both arms | +400, +1000 | TB, 20-iter mean |
| P4 | yaw is learned. S10a: `cmd/yaw_gain` ≥ 0.6 at +1000. S10b: `cmd/heading_err_abs_mean` falls by ≥ 30 % from its first-5 baseline by +1000 | +1000 | TB |
| P5 | no regression: `standup_success_high` ≥ 0.65, `sit/success_rate` ≥ 0.90, `mean_episode_length` ≥ 1100 at +1000 | +1000 | TB |
| P6 | standing gets calmer (vx tracking at zero command is now 3.0 / σ 0.10): `cmd/pos_drift_idle_m` at +1000 ≤ 0.7 × its first-5 baseline | +1000 | TB |
| P7 | F1 + F2 work (re-probe, §7.2): ball < −5 mm ≤ 1 % (was 39 %), root v_z kick > 0.3 m/s ≤ 5 % (was 43 %), noise-free d = 0.6 settle drift < 0.05 rad (was 0.141) | after +1000 | probe Block A |
| A/B | **steering**: decided by the turn test (§7.1), not by TB. The two arms' TB yaw signals measure different things | after +1000 | §7.1 |

## 6. Kills and gates

- `Loss/value_function` > 42000 for 50 consecutive iterations → kill (S8/S9 convention). The reward
  rescale shocks the critic (the smoke spiked to 1.9e6 for one iteration), so read the window, not single
  points.
- `Train/mean_episode_length` < 700 at +300 → kill.
- `curriculum/standup_success_high` < 0.5 at +400 → kill. The spawn change broke the stand-up.
- `spawn/lift_mm_max` > 50 at any read → stop and inspect the support geometry.
- `sit/success_rate` < 0.85 at +400 → flag, not kill.
- PhysX error flood in `train_<RUN>.err.log` → kill.
- Gate at +1000: P3, P4 and P5 met on an arm → that arm goes to the turn test and the re-probe.

## 7. Evaluation after training (prepared, not run)

### 7.1 Turn test (decides the A/B; to build after +1000, ~0.5 day on the probe infrastructure)

Fixed task, both arms, 2048 envs, standing start (d = 0), training plant DR, pushes off. After 3 s of zero
command, command a +50° turn (and −50°, mirrored):

- **S10b**: heading target = yaw₀ + 50°.
- **S10a**: what an operator stick does: ω = 0.30 rad/s for 2.91 s, then 0.

Metrics: |heading error| at t = 8 s, time to within 5°, falls during the 8 s, and |vx| drift during the
turn. **Rule:** S10b wins if its median final error ≤ 0.5 × S10a's and its fall rate is no more than 2 pts
higher. S10a wins if S10b falls ≥ 2 pts more with no error advantage. Otherwise *neither*: pick on deploy
cost (S10a needs no controller change). A second row with vx = 0.15 m/s during the turn covers turning
while walking.

### 7.2 Re-probe F1 + F2 (the review's §3 condition for building the settle hold)

```
pwsh -Command "& '<repo>\RL\launch_probe_s9_spawn.ps1' -Blocks 'A,C,B' -Suffix _S10 `
  -RunA 'p9_replant\<S10a run dir>' -RunB 'p9_replant\<S10b run dir>' -TagA S10a -TagB S10b `
  -CkptA <n> -CkptB <n> -Extra 'env.spawn_lift_enable=true','env.standup_path_rest_preset=hip_reaction_1007'"
```

Then Block D the same way (`-Blocks D -Merge <main dir>`). The decision rule is carried over from the
review: if "noise-free, settled 1 s, blind" still trails the training-spawn row by > 5 pts, build option 1
(in-episode hold, d ≥ 0.5 only, blind release, reduced joint noise).

## 8. Deploy work before any S10 flight (not started)

1. `control/config.py`: `RC_VX_MIN/MAX` → −0.15 / +0.30 (now +0.40, above anything trained);
   `RC_YAW_RATE_MAX` 0.5 → 0.40.
2. **S10b only:** `controller.py` heading mode. The stick moves `target` (e.g. ±0.40 rad/s of target
   motion at full stick). ch33 = clamp(1.0 · wrap(target − yaw), ±0.40), ch34/35 = sin/cos(yaw − target).
   The target persists, so it is *not* re-anchored every `YAW_REF_RESAMPLE_STEPS`. That matches the
   training's manual-override path, which keeps the target across resamples. READY → WALK sets
   target = current yaw.
3. **S10a:** no controller change. The 1000-step yaw re-anchor stays (it matches training).
4. Everything from S9 §7 still applies: grace seconds from the speed pick, the sit path with ch29 = −0.6,
   the play-side npz logger, the `standup_path_check.py` desk check. The desk check reads
   `standup_path_knots` from `params/env.yaml`. That table is unchanged: the F2 preset is a separate key.
   The deploy uses the target columns for the pose gate; the rest columns are print-only.
5. `HIP_REACTION_FF` stays False.

## 9. Known gaps

- **No velocity estimate on the actor, 50 ms history** (§1.2): open-loop vx tracking within a segment.
- **Fixed gait clock.** Speed is step length only. A 0.30 m/s command asks for 6 cm steps at 5 steps/s.
  If P3 fails at the top of the range but holds below 0.2 m/s, the clock, not the reward, is the limit.
- **IMU yaw on hardware is gyro-integrated** and drifts. Heading mode is relative: a 50° turn needs yaw
  stable over seconds, not minutes.
- **Tooth skip** (S9 §6) is still unmodeled. Walking adds knee load cycles, so expect more skips per
  flight, not fewer.
- **The F1 lift trusts the collision mesh as authored.** PhysX's convex hull of a mesh can differ from the
  vertex hull by a fraction of a millimetre. The 1 mm clearance covers that.

## 10. Launch record

Launched 2026-10-07 21:54 (local), detached, both from S9b `model_10180.pt`, `--max_iterations 2000` (to 12180).

| | S10a_rate | S10b_heading |
|---|---|---|
| run dir | `RL/logs/rsl_rl/p9_replant/2026-10-07_21-54-57_S10a_rate/` | `RL/logs/rsl_rl/p9_replant/2026-10-07_21-54-58_S10b_heading/` |
| logs | `RL/logs/train_S10a_rate.{out,err}.log` | `RL/logs/train_S10b_heading.{out,err}.log` |
| PID at launch | 45252 | 48336 |
| P1 | **met**: `[S10] yaw_cmd_mode=rate … ramp 0.33→1 over 51200 steps …`, preset line, self-check +4.07 / +3.92 mm; iteration 10180 running, stderr empty | **met**: `[S10] yaw_cmd_mode=heading (kp 1.0, \|Δψ\| U[0.15, 1.5708]) …`, same preset / self-check; iteration 10180 running, stderr empty |

Expected pace ~188 s/iteration (S9): +400 (ramp end) after ~21 h, +1000 after ~2.2 days.

Leftovers: `RL/logs/rsl_rl/p9_replant/2026-10-07_21-49-37_smoke_S10{a_rate,b_heading}/` are the smoke runs.
They can be deleted.
