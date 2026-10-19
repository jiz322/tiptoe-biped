# Future campaign options after S10 (2026-10-09)

**Status:** planning document. No code changed. S10a/S10b were running at +900 (iteration ~11080) when the
numbers below were read.
**Question (user, 10-09):** besides sim2real and hardware work, which investigation is worth most? The four
candidates on the table were (1) a 30 s episode that may be too short, (2) more reward weight on vx/yaw,
(3) whether `diagnostics/actual_torque_hp1_mean` ≈ 2.7 Nm is contaminated, and (4) terrain (slopes,
bumps). The bigger options were jumping and VLA.
**Inputs:** S10 TB (both arms, first 5 vs last 20 iterations), `docs/s10-command-tracking-campaign-20261007.md`,
`belt_knee_actuator.py`, `rl_env_p6.py`, the Isaac Lab terrain package (`<IsaacLab>\source\isaaclab\isaaclab\terrains`),
and the 09-30 S8b flight npz (`data/t0_s8b_flight_20260930/`).

## 0. Summary

| candidate | verdict | discriminating evidence | recommendation |
|---|---|---|---|
| 1. longer episodes | **not binding** by construction | no time input to the policy; timeouts are bootstrapped; γ = 0.99 at 100 Hz is a ~1 s horizon; hardware falls happen at 2–9 s | don't change. Add a survival-vs-time log (§2) and change the length only if late-episode hazard rises |
| 2. heavier vx/yaw weights | **not the limit in sim** | `cmd/vx_gain` 0.82, S10a `cmd/yaw_gain` 0.96 at +900 | don't raise. The open problem is idle drift (it rose: 0.16 → 0.28 m). A weight doesn't fix it; whether observation does is tested in §9 |
| 3. `actual_torque_hp` | **not a bug, wrong quantity for the question** | it is the *total* hip-joint torque (hip motor + belt reaction 0.733·τ_knee + rotor coupling, added after the clip). Its max of 25–30 Nm is impossible for a 17 Nm motor | add motor-side channels (§4, ~0.5 day) and compare with the flight npz through `play_sim2real` on S8b |
| 4. terrain | **standard and buildable, not the binding problem** | hardware fails on a flat floor. The existing "slope" DR is a 1.9 N torso push, not a slope | park real slopes. Small-amplitude roughness is reasonable inside the next fresh-training campaign (§5) |
| jumping | **blocked by the knee belt** | the belt tooth-skips on every stand-up flight. Sim walking already peaks at 35–41 Nm joint-side spring torque | sim-only exploration at most until the knee transmission is upgraded |
| VLA | **premature** | longest hardware survival is 8.7 s. The Pi can't run a VLA | park until hardware walks ≥ 60 s. Then it is a high-level command source over S10b heading mode |
| **A. belt tooth-skip** (new) | **highest value** | skips on every stand-up flight; unmodeled; sim spring peaks exceed the skip level | §7.1: bench-measure the skip torque, log exceedance in sim, then decide on a penalty, a model, or hardware |
| **B. quiet-stand feasibility** (new) | **decides what "settle" means** | 2.5 Hz limit cycle on hardware. A point foot has no ankle torque | §7.2: a sim probe, before more reward work on settling |
| **C. longer history / estimator** (new) | **plausible, not yet evidenced** | 50 ms is shorter than one action→response round trip (60–80 ms) and 1/8 of the 2.5 Hz cycle; P6 drift is weak evidence (confounded); the 0.50 rad/s yaw figure is **not** evidence (§9.2) | §9: an offline information probe first (H-C0, ~0.5 day). If supported: a dual-history estimator **resumed** from S10 by zero-init surgery, not fresh training. Pi latency is not a constraint below ~1 M params |

**Recommended order:** finish the S10 reads and turn test (already planned) → the diagnostics bundle (§8,
~1 day; it answers 1 and 3 and feeds A; H-C0 from §9.6 can run alongside it) → A, co-designed with the
hardware work → B → C (if H-C0 supports it) plus roughness as the next campaign. Parked: longer episodes,
weight bumps, slopes, jumping, VLA.

## 1. Where things stand

**Hardware.** S8b survives 8.4 s and 8.7 s with a 2.5 Hz limit cycle (09-30). Stand-up went 0/6 (10-03), and
the knee belt tooth-skipped on every flight. No S10 checkpoint has flown; the deploy work in S10 §8 is not
built. Every sim gain since S8b is still unproven on hardware.

**S10 at +900** (last-20 means; TB values are one-step population snapshots):

| prediction (S10 §5) | read | status at +900 |
|---|---|---|
| P2: baseline `vx_gain` < 0.3 | 0.86 (a) / 0.92 (b), first 5 iterations | **contradicted.** Caveat: the gain is fit through the origin (`rl_env_p6.py:2805`), so a constant forward drift reads as following. A slope-with-intercept fit separates the two (§8) |
| P3: `vx_gain` ≥ 0.7, `vx_err_moving` ≤ 0.08 at +1000 | 0.82 / 0.08 (a), 0.82 / 0.079 (b) | on track |
| P4 S10a: `yaw_gain` ≥ 0.6 | 0.96 | met early |
| P4 S10b: heading error −30 % | 0.134 → 0.124 rad (−8 %) | not met yet |
| P5: no regression | stand-up high 0.80, sit 0.98, episode length 1756 / 1817 | met |
| P6: idle drift ≤ 0.7 × baseline | 0.16 → 0.28 m (a), 0.18 → 0.27 m (b) | **contradicted:** standing got *less* calm |

Throughput: 23.4 k env-steps/s, 165 s collection + 14 s learning per iteration. Collection dominates
because of the 2000 Hz physics the belt band needs. A 2000-iteration campaign is ~4 days per GPU, which
makes "one change at a time" expensive and favours bundling changes that already require fresh training.

## 2. Episode length (30 s)

**Facts.** `episode_length_s = 30` gives 3000 control steps at 100 Hz (decimation 20 × 0.5 ms). Commands
resample every 1000 steps, so an episode spans 3 segments. γ = 0.99 and λ = 0.95: the discount horizon is
~100 steps (1 s). Rollouts are 128 steps. Timeouts are bootstrapped: the Isaac Lab wrapper sets
`extras["time_outs"] = truncated` and rsl_rl `ppo.py:177` adds γ·V back. The policy obs has no
episode-time channel; `grace_norm` saturates after the grace period. Sit episodes (30 %) end as a
time-out about 6–12 s in.

**What that implies.** Within the 1 s horizon, a state is valued the same whether the episode is 30 s or
300 s long. The timeout is invisible to the learner. Lengthening the episode changes only the **state
distribution**:

- more samples from late states (accumulated anchor drift, yaw re-anchors, many command switches);
- fewer samples from the stand-up and grace phase (roughly 15–20 % of samples: grace 1.7–5 s over an ~18 s mean episode, an estimate). That share is what keeps
  `standup_success_high` up, so a longer episode risks a stand-up regression.

Rough fall rate: `reward/terminated_mean` −0.0019 at scale −10 gives ~1.9e-4 terminations per env-step,
about 40 % of non-sit episodes ending before 30 s (~11 points of it stand-up failures, from
`standup_success_all` 0.89). This is consistent with the 1771-step mean length. It is a snapshot-based
estimate, not a measurement.

**Hypothesis H1.** The policy degrades in states that only occur after 30 s. *Test:* a 120 s deterministic
eval (play, training DR, pushes on, commands resampling) with the survival hazard and idle drift binned
per 10 s. *Supported* if the hazard or the drift in 30–120 s is ≥ 1.5× its 10–30 s value: then lengthen
to 60 s and keep the stand-up share with `kneel_start_probability`. *Contradicted* if flat: leave 30 s.
Cost: ~2 h on the play infrastructure. Also log the termination cause per episode (timeout / sit /
fall, by phase) so §1's fall estimate becomes a measurement.

## 3. Reward weight on vx / yaw

Both tracking terms are already 3.0 with σ 0.10 (vx) and 0.15 (yaw). Together they earn 2.35 / step,
12.7 % of the total |reward| (`reward/*_pct`), up from 1.63 at the first 5 iterations. In sim the command is followed: vx gain
0.82 with a 0.08 m/s moving error, S10a yaw gain 0.96. Raising the weights buys little that the gain does not
already show. It also trades against the gait terms that keep the robot up.

"Works somewhat" has three possible locations, and each has a different fix:

| where | evidence | lever |
|---|---|---|
| sim tracking accuracy within a segment | `vx_err_moving` 0.08 m/s at a 0.15 m/s mean command; the actor sees 50 ms of history | observation (§9), not weight |
| standing still at zero command | idle drift 0.16 → 0.28 m (P6 contradicted); `yaw_drift_when_no_cmd` 0.50 rad/s flat | §7.2 first: whether a still stand exists |
| hardware | untested: no S10 deploy yet; RC ranges in `control/config.py` exceed training (S10 §8.1) | build S10 §8 and fly |

If a weight change is ever needed, the cheap one is σ, not scale. A tighter σ sharpens the gradient near
the command without changing what standing still costs.

## 4. `diagnostics/actual_torque_hp1_mean` ≈ 2.7 Nm

**What it is.** `self.robot.data.applied_torque` is the actuator's `applied_effort` buffer. In
`BeltKneeDCMotor.compute` (`belt_knee_actuator.py:750–811`), the hip-pitch column receives three additions
**after** the hip's DC clip:

1. `c·(spring − motor)`, the series-belt residual (c = 0.733, `BELT_COUPLING`);
2. with `plant_hip_reaction=true` (all S9/S10 runs), `c·τ_motor_cmd`: together with 1, the full J^T
   reaction of the knee motor on the hip;
3. rotor inertial coupling (zeroed on series-active envs).

So `actual_torque_hp` is the **total generalized torque at the hip joint**, not what the hip motor delivers.
Physically, the hip motor's load is that total minus 0.733·τ_knee(motor, joint-equivalent).

**Discriminating evidence.**

- `actual_torque_hp*_max` is 25–30 Nm. The hip motor clips at 17 Nm (`effort_limit`); only the post-clip
  terms can push it higher. The same channel cannot be the motor output.
- `actual_torque_kn*_max` is 35–41 Nm against a 23.2 Nm joint-side limit. The knee column is the belt
  **spring** torque, whose transients are not clipped.
- The values are one-step population means over all envs: stance and swing legs, sitting, standing up,
  freshly reset. A stance-leg figure would be roughly twice a mean over both legs.

**Is 2.7 plausible as a joint torque?** Yes. For a stance leg, hip-joint torque ≈ ground force × horizontal
foot-to-hip offset, and the hip sits nearly above the foot (27–55 N × 5–10 cm ≈ 1.4–5 Nm). Your intuition
that the hip *motor* works harder is probably right. The 10-03 HOLD finding says the reaction pushes the
hip deeper, in the same direction as the torso load. If so, the motor carries roughly 2.7 + 0.733·|τ_knee|.

**Hardware anchor (different policy, S8b, mid 60 % of each 09-30 trial):** hip-pitch `motor_tau_cmd`
mean |τ| 4.7–4.95 Nm and `motor_tau_meas` 2.7–3.15 Nm. Knee `motor_tau_cmd` 5.4–6.7 Nm (motor side) and
`motor_tau_meas` 1.5–1.9. `tau_meas` is filter- and deadband-loaded (memory:
torque-delivery-ratio-measurement-trap), and `tau_cmd` includes the D term on noisy velocity, so neither
is a clean comparison. They do not support "2.7 is wrong". They do say the hardware motors are commanded
2–4× the sim's joint-side means, which deserves the like-for-like comparison below.

**Effect on training:** small. `rew_scale_thermal` is 0. `energy` (Σ applied_torque²) includes the reaction,
but it is 0.26 % of the reward.

**H3.** On S8b, sim hip **motor** torque (the reaction excluded) matches the hardware's `motor_tau_cmd` hip
mean within 25 %. *Test:* log `hip_motor` = the clipped DC output for hip-pitch before additions 1–3, and
`knee_motor_cmd_preclip` = `tau_m_raw` (the PD command before the clip, the closest analogue of the
hardware's `motor_tau_cmd`). Then run `play_sim2real.py --handover training` on S8b 8880 with
`plant_hip_reaction=true` and compare the same 60 % windows to the two flight npz files. *Supported* →
the sim torque budget is trustworthy. *Contradicted* (hardware ≥ 1.5× sim) → a plant-load gap, which
probably interacts with tooth skip. Cost ~0.5 day.

## 5. Terrain

**How Isaac Lab does it.** Replace the flat plane (`_setup_scene`, `rl_env_p6.py:1450`
`spawn_ground_plane`) with a `TerrainImporterCfg(terrain_type="generator",
terrain_generator=TerrainGeneratorCfg(...))`. Each sub-terrain is a tile. Rows are difficulty levels;
the curriculum moves an env up or down a row by how far it walked. Relevant generators:
`HfPyramidSlopedTerrainCfg` (slope), `HfRandomUniformTerrainCfg` (bumpy, e.g. 2–10 mm noise),
`HfWaveTerrainCfg`, `MeshRandomGridTerrainCfg` (boxes), stairs. The reference mix is `ROUGH_TERRAINS_CFG` in
`terrains/config/rough.py`, used by every locomotion velocity task (H1, G1, Cassie, Digit configs).

**What it costs here (difficulty prediction).** Common: yes, entirely standard. Integration in this env is
the real cost:

- **Spawn:** `_reset_idx` already adds `env_origins` z (`rl_env_p6.py:3653, 3693`) and the F1 lift reads
  `env_origins[:, 2]` as the floor (1606, 1629). On bumps the floor under each foot differs, so the lift
  needs a height lookup.
- **World-z assumptions:** ≥ 9 sites read absolute z. Examples: the `min_height` termination (3441), the
  height reward (2706), swing-clearance toe z (2444, 4673), base-height diagnostics. On a slope these
  become wrong. They need terrain-relative heights (a ray-cast height scanner or the stance-foot height).
- **Engagement (the CLAUDE.md rule):** every Isaac Lab sub-terrain has a flat platform at its centre, and
  `position_anchor` holds the robot near its spawn. A robot that idles 30 % of its time at zero command
  would never meet the terrain. Use uniform roughness everywhere (platform 0) and measure engagement:
  the fraction of foot contacts landing ≥ 3 mm off nominal.
- **Scale:** the foot is a 2 cm ball. 10 mm bumps are half its diameter and hit contact timing hard.

Estimate: 1.5–2.5 days for roughness with terrain-relative heights. Slopes add the curriculum and IMU work.

**The existing "slope" DR is not a slope.** `dr_gravity_tilt_max_deg` (`rl_env_p6.py:3826–3844`) applies
a constant horizontal force m·g·sin(tilt) to the torso only: at 2°, 1.9 N. The IMU sees no tilt, the floor
normal and friction cone are unchanged, and the force acts at the torso rather than at each link's CoM. A
real slope changes the projected gravity relative to the floor and the foot height difference. Widening this
band is a push DR, not slope training.

**Value.** Hardware fails on a flat floor, so terrain does not attack the binding problem. Its plausible
value is as **contact-timing DR**: rough-terrain training is known to produce more robust flat-ground
gaits (Rudin et al. 2022). That is a reason to include small roughness (≤ 5 mm) in the next campaign that
already needs fresh training (§7.3). Running a dedicated terrain campaign now is not.

## 6. Jumping and VLA

**Jumping.** Torque on paper is not obviously the limit. A 3–5 cm hop needs ~1 m/s of CoM velocity from
~10 cm of leg extension: about 40 N per leg, a few Nm at the knee in a mid crouch (back-of-envelope).
**Landing is the limit.** Impacts of 3–5× body weight load the belt, and the belt already skips: on
every stand-up flight, at commands of 22–27 Nm against a 17 Nm motor clamp. Sim *walking* already shows
35–41 Nm spring peaks. A jumping policy trained on the current plant would mostly teach us that the belt
skips. Sim-only exploration costs 2–4 days (new reward set, flight-phase terms; Li et al. 2023 did this for
Cassie). The deployment chance is near zero until the knee transmission changes. It belongs after the
hardware improvement, as its validation task.

**VLA.** A vision-language-action model here would output velocity or heading commands, not joint
targets: a 100 Hz blind balance problem on point feet is not what VLAs are for. That needs a camera and
off-board compute (the Pi can't run one), and a robot that walks reliably. Longest hardware survival today
is 8.7 s. S10b's heading mode is the natural interface ("turn 45° left, walk 1 m"). Park it until hardware
walks ≥ 60 s. Then the cheap version is an off-board VLM sending heading and vx targets over the existing RC
path: demo value high, research value low.

## 7. Investigations ranked above the original list

### 7.1 A. Belt tooth-skip (highest value)

Skip is unmodeled (S10 §9), happens on every stand-up flight, and walking adds knee load cycles. In sim the
belt spring torque, which is what loads the belt teeth, peaks at 35–41 Nm joint-side
(≈ 26–30 Nm motor-equivalent). `knee_sat_ema_mean` is 0.30. The policy lives
near and above the level at which the real belt skips.

| step | cost | who |
|---|---|---|
| a. bench-measure the skip torque: shank held, ramp the knee motor torque until a skip (and at 2–3 belt tensions) | ~30 min | user, on the rig |
| b. sim diagnostic: fraction of env-steps with \|spring\| above the measured threshold, by phase (stand-up, walk, sit, impact) | ~2 h | desk |
| c. if b > ~1 % in walking: a soft penalty on spring torque above threshold (standard, like any torque-limit penalty) | ~0.5 day + training | desk |
| d. alternatively a skip model (motor offset jumps one tooth = 12° when the spring exceeds the threshold) as DR, so the policy learns to recover | ~1–2 days; uncommon (we would be inventing it) | desk |
| e. hardware: tensioner, wider or higher-pitch belt, or a direct knee | — | hardware campaign |

**H-A.** Training-time spring exceedance predicts hardware skips. *Test:* b on S8b and S9b at the
stand-up lift-off rock against the 10-03 flights' skip times. If the sim never exceeds the threshold where
the hardware skipped, the skip comes from something the plant model lacks (belt stretch, tension loss),
and c/d would be blind to it.

### 7.2 B. Can this robot stand still? (decides the "settle" target)

With two point feet, the support is the line between the balls. Sagittal balance in double support has
no ankle torque and only the hip (torso-momentum) strategy, under 40–50 ms of loop delay. As far as I know,
commercial point-foot bipeds mark time continuously rather than stand still; verify before relying on it.
If a still stand has no practical basin on this plant, then "steps ≠ settles" is not a policy defect, and
reward work aimed at stillness will keep contradicting itself (P6 did).

**H-B.** A still double-support stand on the S8b plant with training DR survives 10 s in ≥ 50 % of
rollouts. *Test:* a stand-only short training (zero command, clock frozen in double stance, a no-step
reward) from S10. Then a deterministic eval. *Supported* → settling is learnable: add a stand mode to the
next campaign. *Contradicted* → redefine settle as low-amplitude marching and target the limit-cycle
amplitude and the idle drift, not zero steps. Cost ~1 day of desk work plus ~1 day of GPU.

### 7.3 C. Longer history or a velocity estimator

Full treatment in §9: what 50 ms can and can't see, Pi inference cost, network shape, a resume path,
and the pre-registered tests.

**Bundle:** if H-C0 (§9.6) supports C, the C campaign is the right one to also absorb small roughness
(§5), the skip penalty if A says so (§7.1c), and a stand mode if B is supported. One ~4-day campaign
instead of four. Per §9.5 it can be a resume from S10, not a fresh start.

## 8. Diagnostics bundle (do first, ~1 day, no training)

| item | answers | where |
|---|---|---|
| `hip_motor_torque_{mean,p95}` (pre-reaction), `knee_motor_cmd_preclip_{mean,p95}` | Q3 / H3 | `belt_knee_actuator.py`: stash before the additions; log in `rl_env_p6.py` near 2931 |
| `belt/spring_over_skip_frac` by phase (threshold from 7.1a; placeholder 17 Nm motor-eq.) | H-A | same |
| termination cause per episode + survival hazard per 10 s bin | Q1 / H1 | `_get_dones` + a play-side 120 s eval |
| `cmd/vx_gain` with intercept (slope and offset reported separately) | the P2 caveat | `rl_env_p6.py:2805` |

All four are logging only: defaults bit-exact, no effect on the running S10 arms until a restart picks
them up.

## 9. History length, estimators and network shape (C in detail)

**Question (user, 10-09):** is 50 ms of history enough, and would more help the robot? If the network
grows, does Pi inference latency become a worry, and would a bottleneck-shaped MLP help?

**Short answer.** 50 ms is enough for the fast balance reflex and too short for the slow quantities:
stride-mean velocity, the robot's own action→response, sensor offsets, plant parameters. Whether those slow
quantities are what limits S10 is **not yet shown**. Run the offline information probe (H-C0, §9.6) before
building anything. Latency is not a real constraint: the whole 09-30 tick, NN included, used 2.7 ms of 10, and
any shape below ~1 M parameters stays sub-millisecond. A bottleneck helps, but for **learning**, not for
speed. The shape worth building is a dual history with a small supervised latent (§9.4 D). It can be
**resumed** from S10 by zero-init surgery (§9.5).

### 9.1 What the actor sees now

| item | value | source |
|---|---|---|
| frame | 36 channels: 21 delayed sensor (pg 3, gyro 3, lin_vel 3, joint pos 6, joint vel 6) + 15 fresh bookkeeping (last action 6, clock 2, difficulty/grace/init_h 3, vx/yaw cmd 2, yaw-err sin/cos 2) | `rl_env_p6.py` `_get_observations` (~2285) |
| history | 5 frames = 50 ms, stacked after the obs delay | `rl_env_cfg_p6.py:497` |
| lin_vel | **zero on both sides**: `lin_vel_obs_dropout_prob = 1.0`, deploy writes zeros (`balance_log.py:261`). 3 of 36 channels are dead | cfg `:604` |
| actor | 180 → 512 → 256 → 128 → 6, ELU: **257,670 parameters**, ~0.26 M MAC, 1.03 MB fp32 | `rsl_rl_ppo_cfg_p6.py:30` |
| loop dead time | action 1–2 steps + IMU 1 + joint 1 → round trip 20–30 ms; the PD tracking lag adds ~40–50 ms more before a joint visibly responds | cfg §12b; CLAUDE.md checklist 6 |

So the only velocity information is gyro rate plus leg kinematics (joint velocity through a knee encoder that
reads the motor side, with belt stretch invisible), over a window shorter than one action→response cycle.

### 9.2 What 50 ms can and cannot see

| quantity | timescale | observable in 50 ms? |
|---|---|---|
| tilt, pitch/roll rate, joint state (the balance reflex) | instantaneous | **yes**. More history adds nothing here |
| the robot's own action→response (delay + PD lag) | 60–80 ms | **no**: the window is shorter than one round trip. The policy can't see how its last command landed |
| stride-mean velocity (what integrates into drift) | one cycle = 400 ms at 2.5 Hz | **no**: 50 ms is 1/8 of a cycle, so the window mean is dominated by sway phase. Idle drift of 0.28 m over a ≤ 10 s segment is a mean of ≥ 0.03 m/s. On hardware, gyro pitch/roll swing at 1.9 rad/s std (09-30, mid 60 %), far above that |
| constant offsets: gyro bias (S10 DR ±0.05 rad/s), joint zero (±0.07 rad), IMU trim | seconds | **no**. Trim is exactly what S8b is sensitive to (pinned sweep K2: topples at pg-bias 0.20; the 09-03 trim bracket flipped hardware outcomes). Note S10 trains with `dr_proj_grav_bias_max=0.00`, so no history could learn trim without that DR on |
| plant parameters (stiffness, friction, belt) | seconds | **no**. RMA's whole motivation |
| position offset from the anchor | integral over the whole episode | **never**, from any proprioceptive history. A better velocity estimate lowers the drift *rate*; it cannot undo drift already accumulated |

**Hardware noise (09-30 S8b flights, second-difference estimate of the high-frequency content; it mixes
sensor noise with real impact content, so treat it as an upper bound):** gyro x/y 0.42–0.46 rad/s, z 0.14
rad/s; pg 0.003; joint velocity 0.47–0.88 rad/s. The training noise DR is gyro ±0.15 uniform (std 0.087) and pg
±0.05 (std 0.029). So during walking the hardware gyro is **noisier** than training at high frequency, and the
pg channel is ~10× cleaner. Averaging over a longer window has real value on hardware, not only in sim.
Lowering the noise DR is not a substitute for gyro, though it might be for pg.

**Correction to the earlier §7.3 evidence.** `diagnostics/yaw_drift_when_no_cmd` (`rl_env_p6.py:3017`) is
the mean of |ω_z| at each instant. The gyro observes that directly, so the 0.50 rad/s figure is yaw sway amplitude, not an
estimation failure, and it is not evidence for C. P6 (idle drift 0.16 → 0.28 m) is weak evidence. S10
changed the vx reward, the spawn (F1/F2) and the command range together, so the drift rise has other
candidate causes. **The case for C rests on the timescale argument above, not on S10 data.** That is why
H-C0 comes first.

### 9.3 Inference cost on the Pi 5

**Measured (09-30, both S8b trials):** `tick_busy_ms` p50 2.71 / p95 2.99 / max 3.96 ms for the whole tick:
IMU read, obs build, NN, CAN send/receive, logging. The CAN batch alone was 1.83 ms p50 at the 08-16 bench,
so everything else, NN included, is ≤ ~0.9 ms today. Budget: 10 ms. The measured action delay of 2.4–2.8 ms
sits in the 1-step training bucket, and stays there as long as the command leaves within the tick. **Headroom
is ~6 ms.**

**Why size barely matters here (estimate, not measured):** at batch 1 and ≤ 1 M parameters, a forward pass
on a Cortex-A76 is dominated by per-op dispatch (each torch op costs tens of µs from Python), not FLOPs.
Adding layers costs more than widening them. The one regime change is cache: the Pi 5 has 512 KB L2
per core and 2 MB shared L3. A net whose fp32 weights exceed ~2 MB streams them from DRAM on every call,
roughly +0.5–1 ms with more jitter. Even that fits the budget.

| shape | params | fp32 weights | MAC | Pi 5 forward (estimate) |
|---|---|---|---|---|
| today: 5 × 36 flat, 512-256-128 | 0.26 M | 1.0 MB | 0.26 M | ≤ 0.9 ms (measured bound, whole non-CAN tick) |
| A15: 15 × 36 flat (150 ms) | 0.44 M | 1.8 MB | 0.44 M | ~ today |
| A50: 50 × 36 flat (500 ms) | 1.09 M | 4.4 MB (> L3) | 1.09 M | today + ~0.5–1 ms |
| B: 50 × 36 → **64** → 512-256-128 | 0.31 M | 1.25 MB | 0.31 M | ~ today |
| D: 5 × 36 flat + 50 × 24 conv encoder → 16-d latent | 0.29 M | 1.15 MB | 0.45 M | today + ~0.1–0.2 ms (more ops) |
| E: GRU 128 on the 36-ch frame + 512-256-128 | 0.29 M | 1.2 MB | 0.29 M | ~ today |

**To verify before committing:** `control/nn_latency_bench.py` already measures per-call p50/p99 and a
deadline sweep on the Pi, but it loads a real checkpoint. A `--dummy-shape` option (random weights for a
given architecture) is ~20 lines, then ~5 min on the Pi. It is pure compute, no motors.

### 9.4 Bottleneck shape: yes, but for learning, not speed

In a flat stack the parameters sit in the first layer: input width × 512. At 50 frames that layer is
922 k of A50's 1.09 M (85 %). It must learn an unshared temporal filter for each of 1800 inputs from the
noisy PPO gradient alone. That makes it sample-inefficient and prone to fitting DR noise. A bottleneck
fixes this: compress the long history into a few numbers before the main MLP.

| option | what it is | common? | for this repo |
|---|---|---|---|
| A. longer flat stack | 15–50 frames into the same MLP | yes for short stacks (Humanoid-Gym uses 15 frames at 100 Hz) | A15 is the cheapest test of "does more history help at all". A50 learns slowly |
| B. narrow first layer | 1800 → 64 → MLP | uncommon as a named method | Cheap, and close to a learned linear filter bank, which suits velocity and bias estimation. No structure; still trained by the RL gradient only |
| C. dual history, conv encoder | the existing 5-frame stack goes straight in; 0.5 s of 24 sensor+action channels → 1D temporal conv (weights shared over time) → 16-d latent → concatenated | yes: Cassie (Li et al. 2024, short + 2 s long history via 1D CNN); RMA adaptation module (1D CNN over 50 steps → 8-d) | Fast path unchanged, slow path compressed |
| **D. C + supervised latent** | as C, with a training-only head that regresses the latent onto privileged truth the critic already gets (base lin vel 3, foot contact 2; optionally the DR biases), with the remaining latent dims free | yes: concurrent estimator (Ji et al. 2022); DreamWaQ / HIM variants | **Recommended.** A supervised loss learns faster than RL. And the estimate is a **loggable hardware channel** (`v_est`) that can be checked against video (checklist 9). That turns "is drift an estimation problem" into a measurement |
| E. recurrent (GRU/LSTM) | hidden state carries unlimited memory | yes (rsl_rl 3.1.2 `ActorCriticRecurrent`; Unitree's G1 example uses an LSTM, as I recall) | **Avoid here.** The hidden state is invisible handover state (checklist 1, the contract the pinned sweep's K22 lost 32/32 on). `play_sim2real` npz injection cannot rebuild it from a short window |

Literature attributions above are from memory; check them before citing externally.

Choices for D: 50 frames (0.5 s, > one stride and > the response round trip). Long-history channels are pg, gyro,
joint pos/vel, last action (24); drop the dead lin_vel and the bookkeeping. Encoder: per-frame 24→32,
Conv1d(k8, s4) → Conv1d(k5) → Conv1d(k5) → 96 → 16, about 21 k parameters. The policy MLP input goes 180 → 196.

**Contract cost (deploy side).** A 50-frame ring buffer in `controller.py`/`balance_log.py`, primed with
zeros exactly as `_reset_idx` primes the sim buffer: the blind `v` contract now spans 0.5 s, not 50 ms.
The encoder is exported in the same TorchScript/ONNX file. `policy_contract_check.py` gets a check on the
long-buffer width and priming. The per-frame mirror tables apply to the long buffer frame by frame. Log `v_est`.

### 9.5 Resume instead of fresh training

The earlier §7.3 said an obs-width change forces fresh training. For D it does not have to. Append the
16 latent inputs to the actor's first layer as **zero-initialized columns**, and extend the normalizer
with mean 0 / var 1. That is the same surgery `RL/scripts/extend_critic_checkpoint.py` already does for the critic. At
iteration 0 the extended S10 checkpoint outputs bit-identical actions. The estimator trains from step 0
on its supervised loss; the actor's gradient on the new columns is (∂L/∂h)·z, nonzero as soon as z is, so
the policy starts using the latent when it carries information. Resume with every S10 override re-passed.

Risk: zero-init function-preserving extension is standard ML practice, but uncommon in the locomotion
literature (people retrain or distill), so this part is partly invented. Fallback: rsl_rl 3.1.2
`StudentTeacher` distillation with S10 as the teacher.

**Build cost (difficulty prediction).** Env: a long-history obs group (rsl_rl 3.x `obs_groups`) + the
privileged targets, 0.5 day. A custom ActorCritic module + the auxiliary loss, done as a PPO subclass in the
repo rather than another unversioned `ppo.py` patch, plus export: 1–1.5 days. Checkpoint surgery script:
0.25 day. Deploy ring buffer + contract check + `v_est` logging: 0.5 day. Tests: 0.5 day. **Total ~3 desk
days** plus the campaign GPU time. A15 (option A, no new code beyond `obs_history_steps=15` and a fresh run or
a 180→540 zero-column surgery) is the ~0.5-day cheap alternative if H-C0 shows 150 ms already captures
most of the gain.

### 9.6 Pre-registered tests

**H-C0 (information, before building; ~0.5 day, no training).** The slow quantities are recoverable
from 0.5 s of the actor's own observations and not from 50 ms. *Test:* roll out S10 (latest checkpoint,
all S10 overrides, training DR) with `play_log.py`, logging the 36-channel delayed obs plus truth: base lin
vel, its 0.4 s (one stride) mean, and the per-episode gyro and joint-bias DR draws. Offline
(`python`), fit ridge regression and a small MLP on windows of 5 / 15 / 30 / 50 frames, with episode-held-out splits.
*Supported* if the stride-mean vx/vy R² at 50 frames is ≥ 0.5 and exceeds the 5-frame R² by ≥ 0.2.
*Contradicted* if the 5-frame R² is within 0.05 of the 50-frame value. In that case history won't fix
drift: drop C, and drift goes to reward or plant (H-B). Report the same table for the biases. The 15-frame
row decides A15 vs D. This tests whether the information exists, which is necessary but not sufficient.

**H-C1 (after training, D vs the S10 control at matched iterations).** `cmd/pos_drift_idle_m` ≤ 0.6× the
control's, with no regression on the P5 metrics (stand-up high, sit, episode length). `v_est` RMS error vs
the critic's truth ≤ 0.03 m/s on idle envs. On hardware, `v_est` integrated over a trial agrees in sign and
within 50 % with the torso travel read from video (Kinovea). *Contradicted* if drift is unchanged while
`v_est` is accurate: the policy has the estimate and does not use it, which is a reward problem, not an
observation problem.
