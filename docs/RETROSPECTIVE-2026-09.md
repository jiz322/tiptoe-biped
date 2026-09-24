# big_thigh — Project Retrospective, 2026-03-28 → 2026-09-03

**Status of this document:** authoritative record of how a 6-DOF point-foot biped
went from an empty repo (2026-03-28) to its first sustained free-standing balance
(2026-09-03, 6.06 s, still upright at log stop). Written 2026-09-05 from the git
history (372 commits), the campaign memory files, `CLAUDE.md`, `docs/`, and
`data/` folder contents. Where the record is silent or contradictory the text says
**unverified**. Nothing here was taken from recollection alone: the few facts that
rest on the user's memory rather than on git, memory files, or data are tagged
**(user-attested)** in place and listed in §6.

Audience: a locomotion engineer who has never seen this repo. Machine and access
details were removed for publication; the attribution section is published as
written.

**Public release note (2026-09-24).** This document was written inside the private
development repo (`big_thigh`, the internal codename of TipToe). Commit SHAs cited
below refer to that repo's history, which was not carried into the public repo
(the public tree is one squashed snapshot). Paths are as they were in the private
repo; the public layout is in `docs/REPO_MAP.md` (in short:
`urdf_onshape_export/ballfeet-remass-urdf/` is now `assets/tiptoe_v27/`, the
`RL/launch_train_*.ps1` scripts are under `legacy/launch/`, and only one
checkpoint, `control/r1r_3150.pt`, is shipped).

Conventions: "trial" = a real hardware recording; "rollout" = a sim episode. Dates
are 2026. Commit hashes are short SHAs from `git log --all`. `pg_x` is the x
component of projected gravity in the policy frame (positive = forward pitch).

---

## Summary

- **Stood:** 2026-09-03, **6.06 s**, upright at log stop; R1r_3150, blind handover,
  trim `+0.089`; `data/p12_4_flight_20260903_v_postrim/`. **n = 1.** Prior best ≈0.17 s.
- **Same-policy hardware evidence:** trim bracket #14 (−0.089/0/+0.089 → fell/fell/
  stood); blind handover #13 (0.34 → 0.67 s, p = 0.003, cross-session, possible
  sign-fix confound); first trim #9 (F3490, ordered); FF flag #11 (null).
- **Inferred necessity** (sim gates or no earlier successful flight): normalizer #5
  (offline parity only), belt model #6 (old policies 512/512 falls on corrected
  plant), fences #7, CAN/SPI #10, IMU signs #12, R bundle #15, #1–#4.
- **Open question:** trim is 0.12 past the measured mounting (−0.032); what +0.089
  compensates is unidentified. Leading candidate: knee joint-angle bias from belt
  slip — the right knee skipped a tooth during the standing trial.

---

## 0. The robot and the problem

- **Hardware.** Two legs × (hip_roll, hip_pitch, knee), point/ball feet, no ankle.
  Hip roll: Robstride EL05 (6 Nm peak, motor ids 103/106). Hip pitch and knee:
  Robstride RS01 (17 Nm peak, ids 101/102/104/105). The **knee motor sits on the hip
  chassis** and drives the knee through a 22:30 belt, so hip pitch motion changes the
  knee joint angle with the knee motor still (`q_knee = G·θ_motor − c·q_hip`,
  G = 22/30, c = 0.733). Compute: Raspberry Pi 5 (user-attested; the repo says
  only "Pi") running the 100 Hz loop in Python. Mass 5.5 kg (08-18 scale) /
  5.57 kg in flight configuration (09-01); ≈50 cm tall (user-attested; base spawn
  height in sim 0.44 m). IMU: Yesense 6-axis at 400 Hz.
- **Training.** PPO (rsl_rl) in Isaac Lab, task `Template-standup_curriculum-v0`
  (`RL/source/RL/RL/tasks/direct/standup_curriculum/rl_env_p6.py` +
  `rl_env_cfg_p6.py`). 36-dim observation × 5-frame history = 180 dims; 6 actions
  (PD position targets); 100 Hz control on 500 Hz physics; 32 768 envs on RTX 4090s.
- **Deploy.** `control/` on the Pi: `balance_log.py` (bench loop with keyboard-gated
  HOLD → OBSERVE → NN handover), `motor_bridge.py` (sim ↔ motor frames, belt
  decoupling), `policy.py` (rsl_rl checkpoint loader incl. obs normalizer),
  `imu_adapter.py`, `robstride_socketcan_driver.py` (CAN over SPI, since 08-15).
- **Why it is hard.** Point feet give no static sagittal balance; the policy must
  step reactively. Every millimetre of CoM error, every degree of IMU bias, and
  every unmodelled Nm in the knee drivetrain shows up as a fall within 0.2–0.7 s.

---

## 1. Timeline (from evidence)

Grouped by phase. Each entry: what changed → why → what it produced.

### Phase A — sim only: reward shaping and curriculum (03-28 → 04-17)

- **03-28** `2444848` initial commit; `12fec55` "stand v1" — classical PD/LQR
  stand controllers in `script/` (legacy, Isaac Sim Script Editor era). `711c81f`
  LQR added 03-29.
- **03-29/30** `4d69191`, `9c755e7` first RL task; `9a92245` "RL V1 SUMMERY" →
  root `README.md`, the **v1 post-mortem**: the policy "balanced" by saturating both
  hip-roll actions at ±0.5 rad (≈4 Nm constant per joint, 2× the motor's continuous
  rating), which pressed the feet into the floor. Diagnosed causes: energy penalty
  ~100× too small, joint-limit penalty measured actual not commanded position,
  action-rate penalty is zero for a constant action, no action-magnitude penalty,
  `effort_limit` set to the burst rating, and observations taken from the
  articulation root rather than an IMU frame.
- **04-03** `0c2ffca` stand-up curriculum added (task folder `standup_curriculum`);
  **04-04** `ecdd897` "Curriculum Learning Successful". Phase files `rl_env_cfg_p1`
  (balance only) → `p2` (standup curriculum) → `p3` (reward rebalance: new
  action-magnitude penalty, action-rate ×5, alive bonus halved, energy ×3,
  joint-vel ×5 — the fix for the v1 exploit).
- **04-05/06** `9f12bda` phase 4 command control; `0f228f4` CoM-command;
  `3a1b358` **"This is Galloping"** → the anti-gallop trio (no-flight penalty,
  vertical-velocity penalty, stance-symmetry EMA) that persists in every later cfg;
  `b535823` reward reference moved from shin CoM to toe; `abb5859` yaw control
  (04-10). Phase 4c replaced the CoM command by a velocity command (drives stepping).
- **04-16** `416b75e` phase 5 "walking within torque limit" — a torque curriculum
  that anneals `effort_limit`/kp/kd from training values to real-motor values;
  `5814c73` phase 6; `defecaa` "v2, accept gemini critics" (an external AI reviewer
  was consulted; details unverified).

### Phase B — hardware bring-up and first sim2real loop (04-18 → 06-29)

- **04-18** `1e804bc` **control stack v1**: `motor_bridge.py` already contains the
  22:30 gear, the hip–knee kinematic coupling for **position and velocity**, and
  the kp/kd × G² gain mapping (`_sim_to_motor_gains`); `9d304e3` 400 Hz IMU driver
  and the six IMU sign constants **inherited from an earlier self-balancer robot**
  (never physically verified until 08-28). 04-19 motor tests, zeroing.
  04-21 → 04-26: `main.py` RC-armed loop, `tune_kp.py`, "wrap bug fixed"
  (`b3327c1`). First policies on the robot; outcomes not recorded.
- **05-01 → 05-04** `d29a322` obs inspector; `82c3795`/`ae64952` kp/kd DR with
  **coupled draws** (independent kp/kd corners produced "catatonic" joints);
  `5a863b7` broader physics DR; `61136fe`/`40cad10` **crawl-pose zeroing**
  (`CRAWL_JOINT_POS`, the robot is zeroed lying in a crawl pose — still the zero
  reference today). ~12 driver commits on 05-04 (`58d87e4` … `d7fb232`,
  `e53ebd4` `diag_replies.py`, `8622901` `profile_balance.py`) hand-tuned the
  CAN2USB frame pacing (1.0 ms inter-frame sleep + 2.8 ms reply deadline) to reach
  100 Hz with all six motors replying — the tuning history was recovered from git
  on 08-12.
- **05-10** `ee4b990`/`16cbcef` action low-pass filter added on both sides;
  **05-14** `54ee6da` obs delay removed for last-action channel; **05-15**
  `4716bf0` "Hardware test". **05-19/20** `friction_id.py` bench friction ID
  (later superseded — it used `torque_fdb` as truth, see 08-14).
- **05-26** `85b6ab0` **Train env v7: `ImplicitActuatorCfg` → `DCMotorCfg`**
  (explicit motor model with speed–torque envelope, armature, Coulomb/viscous
  friction), physics 200 → 500 Hz (decimation 2 → 5), action scale =
  effort_limit/kp per joint, kp curriculum removed, DR expanded to armature,
  friction, battery voltage. The header states the motivation from hardware:
  hip-roll bang-bang despite wider action scale, and phase-margin collapse where
  back-EMF removed torque at speed. Incompatible with earlier checkpoints.
  **05-27/28** `528cd9d`/`eea92d2` **pickup curriculum** (a virtual hand holds and
  releases the robot; 50 % of episodes start held).
- **06-05** latency test scripts (IMU, joint, action). **06-07** `f1fbc97`
  `urdf_onshape_export/ballfeet-urdf/` added; **06-12** `ecac7e1` training
  switched to the **ball-foot asset** (`ROBOT_USD_PATH`; FK toe target 0.0716 →
  0.0856 m, "ball foot is ~14 mm taller"), plus an **asymmetric critic**
  (privileged obs group), a clock-contact reward and a swing-clearance reward.
  The reason for the sphere foot is not recorded in git; the user recalls it as a
  design judgment — the original point-foot shape was considered absurd and a
  sphere assumed better, with no test (user-attested). Effect **unverified**. The
  real feet are TPU balls per later notes.
- **06-16** `3bda606` "The first version that stepping not shuffling".
  **06-19 → 06-22** hardware logs (`data/*2026061[9]*`, `*20260621*`,
  `*20260622*`), `sysid_experiment.py`, `sysid_replay_sim.py` (teacher-forcing
  replay of hardware logs in Isaac). **06-26** `2d63347` driver fix: RS01 velocity
  decoded over ±50 instead of ±44 rad/s — **every June sysid log has hip/knee
  velocity ×1.136 inflated**; nobody knew until the 08-14 forensics. 06-28/29
  `joint_tuner*.py`, `sysid_loaded.py`, "new ckpt".
- **06-29 → 08-06: no commits.** No work was done on bipedal; the user stopped to find better tools, and resumed when a new model release in Claude
  Code (the user's term: "fable5") restored confidence (user-attested). The
  campaign memory (which starts 08-06) says all June and 08-03 flights ran the
  un-normalized policy (see 08-06).

### Phase C — the August campaign: contract bugs, plant identification, retrain (08-06 → 08-23)

- **08-06** `7c419e6` "critic bug fix" (misnamed) = **obs-normalizer fix in
  `control/policy.py`**. The loader searched checkpoints for `running_mean`; rsl_rl
  3.x stores `actor_obs_normalizer._mean/_std`, so the search missed and the actor
  ran on **raw observations** in every hardware run before 11:52 that day. Actions
  differed from the trained function by 0.4–0.6 (in ±1 units). Same day: torque
  forensics on hang logs found the **knee belt-drag term** (firmware Kd fights
  belt-forced motor spin, ~30 % of knee command RMS) → `KNEE_DECOUPLE_FF` knee
  velocity feed-forward in `motor_bridge.py` (committed 08-11 in `8ba262e`, set
  False for the F lineage).
- **08-07** `BeltKneeDCMotor` sim actuator (belt drag, stale-hip latch,
  motor-velocity torque envelope; hip armature 8.4e-3) — equivalent to the validated
  injection to 1e-4; 7-log replay corpus mixed (`reports/belt_coupling_sim2real/`).
  Firmware register reads (via a protocol switch + power cycle): limits exactly
  17/6 Nm, 23/11 A — **no hidden derating**. Free-hanging lash bench: no free lash
  (k_belt > ~20 Nm/rad); "compliance/slip only under high tension" fenced as
  unreachable on the bench. **Sustained knee-saturation termination** (`sat_term`)
  added to the env as a constrained-RL fence.
- **08-08 → 08-11** `p8_belt_fresh` retrain; strict fence won; hip fence (08-10,
  motivated by user pushback that hips railing at 0.55 duty is a robustness
  problem); D_polish; **F_smooth** won (08-11). Hardware 08-11: hand-held hang log
  — knee saturation 1.1–1.5 % vs 32 % in the bad logs, **deploy pipeline verified
  exact** (logged obs → policy → clamp reproduces logged actions to 3e-6). Ten
  ground trials: **all fell in 1.47–1.99 s**, but the policy visibly stepped ("best
  ever"). Sim sweep: action delay 3–4 ticks was the dominant tested factor →
  `G_delay14`. Then the **user's IMU-offset hypothesis**: launch `pg_x` =
  −0.089 ± 0.017 across 9 trials, 3× outside the trained ±0.03 bias band; a forced
  obs bias of that size kills 88 % of sim robots → `IMU_PG_OFFSET = [−0.089, 0]`
  trim added in `imu_adapter.py`, `dr_proj_grav_bias_max` 0.03 → 0.10
  (`G_delay14_imuoffset010`). Also found: resuming a run silently reverts
  hyperparameters to cfg defaults (`resume-reverts-polish-hyperparams`).
- **08-12** control-loop audit: 9.3 ms/tick came from the hand-tuned sleeps;
  `can_timing_probe.py` on hardware: the **CAN2USB adapter buffers ~3 replies and
  has a deterministic 2.93 ms turnaround** — the 1.0 ms spacing was provably
  optimal, software was out of runway, and an SPI SocketCAN HAT (MCP2518FD) was
  the recommended path. `desired_kl=0.003` found to pin the adaptive LR at its
  1e-5 floor.
- **08-13** hardware: 14 G2i4760 trials all tip at **0.25 ± 0.02 s**; F3490 trim
  sweep; the user's **knife-edge test** (kp 30 hold, repeated releases) put the
  static balance point at displayed `pg_x = +0.075` under the −0.089 trim and fixed
  the sign convention (larger `pg_x` = forward). Sim with hardware-matched launch
  survives ~6 s → **plant-response gap, not launch/IMU/DR**. `b631c7a`
  `KNEE_DECOUPLE_FF` True (the correct pairing for belt-trained policies): 12
  trials, departure 0.164 vs 0.161 s — correct but not the cause. Vendor confirmed
  the CAN2USB adapter is their host-PC debug tool.
- **08-14** motor-telemetry forensics (`analysis_motor_filters/`): `torque_fdb` is
  filtered ~25–35 ms with a ~0.15 Nm deadband; the June velocity decode bug; the
  fitted act-lag prefilter was ≥2/3 phantom. Consequence: **every torque-based
  "delivery ratio" and "belt slip" conclusion to date was artifact** — including
  the one that nearly led to retraining a 0.6 knee-delivery scale. Line-by-line
  deployment-contract audit: clean, two latent bugs.
- **08-15/16** `1c8658f` **CAN over SPI** (`robstride_socketcan_driver.py`,
  `setup_can0.sh`, `CAN2SPI_BRINGUP.md`): RTT p50 2.99 → 0.57 ms, tick busy p95
  9.3 → 3.9 ms. All six motors replied on the same arbitration id 0x0FD →
  collisions → 0.5 ms pacing. `26fe2fe` per-motor host ids written to motor NVRAM
  (`set_host_ids.py`) → **zero pacing, busy p95 2.45 ms, 120 s soak 100.00 %
  replies**. `SPAWN_HEIGHT` 0.434 → 0.4401 (the only OOD obs channel, −1.07 σ).
- **08-17/18 replant bench** (`500032b` snapshot tag `pre-replant-20260817`;
  report `docs/replant-report-20260817.html`). Phases 0–6 on the tethered robot:
  loop timing ~10× faster than sim assumed; kd-path filter transparent (0.7 ms);
  **full 1:1 torque delivery 4 → 17 Nm** (current-clamp hypothesis dead);
  act-lag prefilter phantom on all joints (`act_lag_enable=False`); friction and
  inertia: hip_pitch I = 31e-3 (CI 29.6–32.8), knee 21e-3, Coulomb 0.14–0.19 Nm,
  viscous small; weigh-in 5.5 kg (+5.9 % vs URDF 5.195); crawl-pose IMU
  decomposition: **true mounting offset ≈ −0.032**, so the −0.089 trim
  over-corrects by ~0.057. Review found 7 claims to correct and that the
  **viscous-friction DR band sat 7× below the measured value**.
- **08-18** `2d0441f` **P7 replant**: RS01 rotor inertia ≈ 14–19e-3 at output
  (vendor 4.2e-3 was 3–4.6× low); cfg armature hip 33.2e-3 / knee 36.0e-3 / roll
  7.0e-3; **off-diagonal rotor coupling** c·I_r/G² = 26.4e-3 injected as a plant
  force (PhysX armature is diagonal-only) — on the 08-13 fall logs that term is a
  hip torque as large as the hip's own command, peaking in the fall-initiation
  window; measured frictions; mass DR recentred. **P8 gate: G2i, which survived
  6 s on the old plant, falls 512/512 on the replanted plant** — so the
  hardware-matched sim finally fell where hardware fell, and P9 had to be a fresh
  run (`b26d9ce`, `7c6d933`, pg-bias DR 0.10 because a supine IMU check is
  physically impossible: the Pi is on the torso's back).
- **08-19 → 08-23** P9 plateaued at `pickup/active_frac` 0.71; the user spotted
  the discriminator across every run: **`swing_clearance_zero_m` must equal the
  settled idle toe height + 3 mm** (3 recurrences). `415257e` P10 CLI plant/servo
  knobs; **K line = gains designed from measured inertia** (kp 20/30, kd 1.4/1.5
  vs the old 8/8) — hardware always was the slow servo (2.3 Hz), the old sim
  flattered itself (4.4 Hz). 08-20 tether-bias bound; 08-22 braced ballistic +
  hip-clamped knee confirm both rotors ≈ 19e-3; kp-sweep flight gate PASS.
  08-21 `docs/knee-hip-coupling-review-20260821.md` (found the `KNEE_DECOUPLE_FF`
  revert in `500032b` and re-armed it). 08-23 checkpoint selection
  (`docs/p9-checkpoint-selection-20260823.md`): K4b_3600 pick; **P8 policies
  collapse to ≤9 % survival on the replanted plant where K scores 99 %**. rsl_rl
  adaptive-LR ceiling patched 1e-2 → 2e-4 in site-packages (unversioned) after
  ceiling rides killed K3/K3s2/K4.

### Phase D — flights, contract bugs #2 and #3, robustness, and the trim (08-24 → 09-03)

- **08-24 P11 flight** (`data/p11_flight_20260824/`, K4b_3600, 12 npz): **deploy
  contract verified clean end-to-end on hardware for the first time** (gains,
  α, effort frames, policy bit-exact 2e-6, CAN 6/6). Grounded trials fell
  pitch-forward at 0.14–0.26 s with active catch-stepping; best honest free stand
  T9 ≈ 0.17 s (the "2.35 s" T5 was in-hand). `CAN_TIMEOUT` 200 ms armed on all
  motors (`3800bfd`). Vel-decode gate fixed (lock-in estimator, `70185df`).
- **08-25 P12 flight** (`data/p12_flight_20260825/`, 12 npz, chopstick entries,
  taped mat): all ≈ 0 s free-standing. **Right-knee belt skipped one tooth
  (12° = 0.21 rad)** — observed physically by the user, invisible to every npz
  channel (encoder is motor-side). Crawl-pose knee-symmetry check between every
  trial adopted. A "trained forward-lean idle" fall mechanism was proposed and
  **retracted the same day** after user pushback (full-kinematics CoM sits −1.1 cm
  over the stance toe: it is a balanced pitched crouch). P13 launched
  (`0a5b1ae`, `8badffa`): **exec-noise injection** on motor targets, pushes on in
  training, action delay DR 1–3, wider spawn DR, μ 0.4–1.1; LR ceiling patch v2
  (1e-3) + `desired_kl` 0.02 after the cap-free restart began collapsing in 100
  iters.
- **08-26/27** R2 (max DR) never trained and diverged; R3 (cheap-dim widening)
  dominated. Eval sweep: **R1r_3150 = deploy pick** — central plant a tie with
  K4b (0.99+), push recovery 0.86 vs 0.73, R1-conditions box 0.59 vs 0.29, knee
  clip 26 % → 4 %. `19eae76` `control/r1r_3150.pt`. Hardware `data/p12_2_flight_20260827/` (K4b_3600, 10 trials;
  per-trial outcomes not summarized — unverified).
- **08-28** sim steps RIGHT first at gait phase 0, hardware always LEFT →
  `e066509`/`221e397` **IMU sign check**: the inherited sign set implemented a
  **left–right mirror** of the 180°-about-z-mounted sensor — `IMU_ROLL_SIGN`,
  `IMU_GYRO_X_SIGN`, `IMU_GYRO_Z_SIGN` flipped (pitch channels were always
  right). Every earlier recording fed the policy mirrored `{pg_y, ω_x, ω_z}`.
  User physically verified the mount; confirmation re-run skipped. 18:00 session
  (`data/p12_2_flight_20260828_6pm/`, R1r_3150, 8 npz): 7/7 ground trials fell
  **backward** with a violent tick-0 torso raise (R_hip saturated). Whether this
  session ran the corrected or the old sign constants is not recorded and was left
  unresolved by user decision (user-attested).
- **08-29** `54d97d7` **handover-contract root cause**: `rl_env_p6._reset_idx`
  zeroes the obs-delay buffer, so training episodes start **blind** (obs[0:27] = 0,
  replicated ×5); `balance_log.py` 'b' fed the real frame ×5 — a state training
  never produces; the policy answers the two with entirely different openings.
  'v' blind handover added; sim `--handover {training,deploy}`; offline replay
  parity 1e-6; predictions pre-registered (`docs/PREDICTIONS_p12_3.md`).
- **08-31 P12.3 flight** (`data/p12_3_flight_20260831_v/`, 6 npz): contract PASS
  (tick-0 action = predicted blind reflex within 0.005), the violent opening gone,
  survival to 25° tilt **0.34 → 0.67 s** (p = 0.003) — but **5/5 still fell
  backward** into the same −13…−16° attractor. Offline: the policy's quasi-static
  pitch response is annihilated at equilibrium by its own action/joint feedback
  (**fixed-point dead zone**; only pitch rate wakes it). Arm A: sim initialized at
  the exact hardware fork state **recovers 19/20 in place** under the same weak
  commands → the dead zone is real but not sufficient; the gap is plant-side.
  Simlog knee `target` channel found inflated ×2 in all pre-08-31 sim npz. The
  trim A/B was **deferred** (user order, after a counterfactual replay predicted
  no material change from trim).
- **09-01** drag test: floor μ_k 0.86 bare / 1.08 taped — **inside** the DR band,
  contact hypothesis dead; belt tap test k_b ≥ 181 Nm/rad — kp-OOD hypothesis
  dead; crawl reference sim-validated to ≤1.3°. `docs/p12-4-bench-runbook.md`
  (740 lines) authored.
- **09-03** (`docs/p12-4-session-20260903.md`): the user **abandoned the bench
  runbook** (too complex, not doable in practice) and **retracted the knee
  armature anomaly** (tap frequency had been double-counted). Trim bracket in one
  day: `423ace7` `IMU_PG_OFFSET=[0,0]` → 6/6 backward, PONR25 1.13 s, mostly
  tether-ended; `efdd12d` `IMU_PG_OFFSET=[+0.089, 0]` →
  **`r1r_3150_log_20260903_134700.npz`: 6.06 s, never past 25° for good, still
  standing at log stop**, raw pitch −14.5° ± 6.7°, knees ±0.31 rad symmetric
  crouch. Right-knee belt **slipped mid-trial (~3.2–3.8 s)** and the robot kept
  balancing (hip-roll R−L splay +0.18 rad afterwards). Same day: training hole
  found — `dr_com_offset_max=0.015` is sampled **per link per axis**, so the
  whole-body CoM_x essentially never shifts (std 3–4 mm).

---

## 2. Fixes without which the robot would not stand

In discovery order. "Verified" is what the record actually shows, not what was
intended. Four fixes have a hardware before/after on the same policy: #14 (trim
bracket on R1r_3150: fell / fell / stood, n = 1 at the winner), #13 (blind
handover on R1r_3150: 0.34 → 0.67 s to 25° tilt, p = 0.003, cross-session and
possibly confounded by the same-day sign fix), #9 (first trim: fall time ordered
by trim value on F3490, 08-13), and #11 (knee feed-forward flag on G2i4760: null
result). #5 (normalizer) is verified only as offline function parity — the
post-fix policy computes the trained function — with no hardware outcome
comparison. For the rest (#1–#4, #6–#8, #10, #12, #15) necessity is inferred from
sim gates or from the absence of any successful flight before them.

| # | Fix | Symptom | Root cause | Fix (commit) | Verification in record | Class |
|---|---|---|---|---|---|---|
| 1 | v1 reward/limit rework | Sim policy "balanced" with hip roll railed at ±0.5 rad, ~4 Nm constant (2× continuous) | Energy penalty ~100× too weak; joint-limit penalty on actual not commanded pos; no action-magnitude term; `effort_limit` = burst rating | `9a92245` (post-mortem, 03-30); p3 cfg reward rebalance (`0c2ffca`…`ecdd897`, 04-03/04) | Curriculum "successful" 04-04 (sim only); hip clip stayed a recurring issue until the fences (#7) | training distribution |
| 2 | Anti-gallop + stepping rewards | "This is Galloping" (04-06); later shuffling instead of stepping | Gait rewards paid for flight; swing-clearance zero point misplaced vs resting toe height | `3a1b358` (04-06) no-flight/vz/symmetry; `ecac7e1` (06-12) clock-contact + swing-clearance; `3bda606` (06-16) "stepping not shuffling"; zero rule 08-19 (`swing_clearance_zero_m = idle_toe_z_p10 + 0.003`) | 08-19: every run violating the zero rule parked at `active_frac` 0.71; every compliant run broke out (3 recurrences) | training distribution |
| 3 | Actuator model ideal → DC motor | Hardware: hip-roll bang-bang despite wider action scale; loss of authority at speed | PhysX implicit PD had flat torque to `effort_limit`; real motor loses torque with back-EMF | `85b6ab0` Train env v7 (05-26): `DCMotorCfg`, 500 Hz physics, action scale = effort/kp, DR for armature/friction/voltage | Header rationale only; **no A/B in record**, and the user recalls none (user-attested) — necessity unverified | unmodeled dynamics |
| 4 | Foot geometry → ball feet | not recorded | Design judgment: original point-foot shape considered absurd, sphere assumed better; no test (user-attested) | `f1fbc97` asset (06-07), `ecac7e1` training switch (06-12); +14 mm toe height | **Effect unverified** — nothing in git or data; the real feet are TPU balls | unmodeled dynamics / hardware |
| 5 | Obs normalizer in deploy | Actions on hardware diverged 0.4–0.6 from the trained function; knee channels anti-correlated for some checkpoints | `policy.py` searched for `running_mean`; rsl_rl 3.x stores `actor_obs_normalizer._mean/_std`; loader fell back to no normalizer | `7c419e6` (08-06, commit titled "critic bug fix") | `normalizer_gap_check.py`; 08-11 offline parity 3e-6; every log before 08-06 11:52 is un-normalized | initialization / contract |
| 6 | Knee belt drivetrain model (user's #3) | Knee tracking R² collapse under saturation; hang-log knee gain 0.28; falls at 0.16–0.25 s on every G2i trial | Sim knee was rigid direct-drive. Missing: firmware Kd fighting belt-forced motor spin; rotor inertia (vendor 4.2e-3 vs measured ≈19e-3 at output); off-diagonal inertial coupling c·I_r/G² = 26.4e-3; stale-hip latch | `KNEE_DECOUPLE_FF` v_des FF (code 08-06, `8ba262e` 08-11); `BeltKneeDCMotor` (08-07, in `500032b`); **`2d0441f` P7** (08-18) armature + off-diagonal + frictions; K gains (`415257e`, 08-19) | Replica = injection to 1e-4; bench 08-22 (+2.9e-3 braced jump, hip-clamped knee no drop); **P8 gate: G2i 6 s → 512/512 falls on the corrected plant; P8 policies ≤9 % vs K 99 %** | unmodeled dynamics |
| 7 | Saturation fences (knee 08-07, hip 08-10) | 32 % knee action clip on hardware; belt regime unmodellable there | Constrained-RL termination on sustained clip (EMA) | `sat_term_*` in cfg §10 (08-07, in `500032b`); hip fence 08-10 | Hardware knee saturation 32 % → 1.1–1.5 % (08-11 hang log), 3–4 % standing | training distribution |
| 8 | Delay DR widened | Sim sweep: delay 3–4 ticks doubled mortality; hardware lag under load 3–4 ticks | `dr_action_delay` 1–2 → 1–4 (G_delay14, 08-11); R1 1–3 | via CLI overrides (unversioned launch args) | Later killed as *fall cause* (08-18 zero-delay gate identical; 08-29 sim lag = hw lag by identical metric) — kept as robustness | training distribution |
| 9 | IMU pitch trim v1 (−0.089) | Launch `pg_x` −0.089 ± 0.017 with joints at default, 3× outside DR; forced obs bias kills 88 % in sim | Assumed IMU mounting; later measured mounting is only −0.032 (08-18) | `IMU_PG_OFFSET` in `imu_adapter.py` (08-11); `dr_proj_grav_bias_max` 0.03 → 0.10 | 08-13 pre-launch `pg_x` z-score 3 σ → 0.54 σ; tip-time ordered by trim; **did not stop the falls** | convention / sign (as framed then) |
| 10 | CAN2USB → CAN over SPI (user's #1) | Tick busy 9.3 ms of 10; obs 6 ms stale; 100 Hz was the ceiling | Adapter = vendor debug tool: 2.93 ms deterministic turnaround, ~3-deep reply FIFO (probe 08-12; vendor confirmed 08-13) | `1c8658f` (08-15) SocketCAN driver, MCP2518FD HAT; `26fe2fe` (08-16) per-motor host ids → zero pacing | Bench: RTT 0.57 ms, busy p95 2.45 ms, soak 100 %; **no balance A/B** — latency was later excluded as the fall cause (see #8) | hardware / wiring |
| 11 | `KNEE_DECOUPLE_FF` flag parity | Hardware False while training True → 0.37·q̇_hip Nm unmodelled knee drag (21–23 % of knee PD) | Inverted config comment ("double-compensates"); `500032b` re-reverted it | `b631c7a` (08-13) True; re-armed 08-21 (review §5); runbook gate §0.3 | 12 trials 08-13: departure 0.164 vs 0.161 s — correct, **not the cause** | initialization / contract |
| 12 | IMU lateral signs (user's #2) | Sim steps right-first, hardware left-first; one-signed roll in falls; 5–7 Hz hand-hold thrash | Six sign constants inherited from a prior robot; sensor mounted 180° about z; old set = left–right **mirror** of `{pg_y, ω_x, ω_z}` | `221e397` (08-28): `IMU_ROLL_SIGN` −1, `IMU_GYRO_X_SIGN` −1, `IMU_GYRO_Z_SIGN` +1 | Acc-ground-truth arbitration (corr −0.77/−0.85, CCW yaw certified); user verified mount physically; **confirmation re-run skipped**; no same-policy A/B | convention / sign |
| 13 | Handover contract blind ×5 (user's #5) | Violent tick-0 torso raise, R_hip saturated, 7/7 backward falls (08-28) | Training `_reset_idx` zeroes obs[0:27] and primes history ×5; deploy 'b' primed the real frame ×5 | `54d97d7` (08-29): `balance_log.py` 'v'; `play_sim2real.py --handover` | 08-31: tick-0 action = prediction within 0.005; PONR25 0.34 → **0.67 s** (p = 0.003, R1r_3150 both sides, baseline = 08-28 18:00 session); falls persisted. 0.34 → 0.67 s possibly confounded by the same-day sign fix if the 18:00 session ran the old constants. | initialization / contract |
| 14 | Trim as plant-gap compensator (+0.089) (user's #4) | With blind handover: 5/5 backward at −0.089; 6/6 backward at 0 | True mounting ≈ −0.032 → working knob is +0.12 past truth: the policy must be told it leans ~7° further back than it does. Whole-body CoM_x never covered by DR (per-link sampling). Leading candidate for what is compensated: **knee joint-angle bias from belt slip/stretch/zero error** (12° × 0.2 m shank ≈ 4 cm foot displacement) — same mechanism as the 08-25/09-03 tooth skips, **candidate not fact** | `423ace7` (0), `efdd12d` (+0.089), 09-03 | **6.06 s standing, n = 1**; static CoM ≲ 1 cm by knife-edge; Kinovea knee-marker check not yet done | training distribution (compensated on hardware) |
| 15 | P13 robustness bundle → R1r_3150 | Every K flight fell at the first catch-step | Policy brittle to execution error; pushes were eval-only | `0a5b1ae`/`8badffa` (08-25): exec-noise on targets, pushes in training, delay 1–3, spawn DR; `19eae76` export | Eval: push recovery 0.86 vs 0.73; r1-box 0.59 vs 0.29; knee clip 26 % → 4 %; the policy that stood on 09-03 | training distribution |

### Corrections to the user's list of five

1. **CAN adapter.** Right that it was the vendor's tuning/debug tool and that the
   swap was needed to run 100 Hz with fresh observations. The record does **not**
   support latency as a fall cause: the 08-18 zero-delay gate was statistically
   identical, and on 08-29 the closed-loop lag measured identically in sim and
   hardware (3–7 vs 4–6 ticks). Necessity is infrastructural, not causal.
2. **IMU signs.** Correct, including the detail that joints were verified and the
   IMU was not: `docs/imu-sign-check.md` records that the constants came from the
   earlier balancer and the `tools/test_imu.py` the config comment referenced never
   existed. The pitch channels (the fall axis) were always right.
3. **Knee belt.** Partly wrong. `1e804bc` (04-18) already had **position and
   velocity** kinematic decoupling and the kp/kd × G² mapping. What was missing was
   (a) the firmware-Kd belt-drag term (fixed by the v_des feed-forward and modelled
   in `BeltKneeDCMotor`), (b) rotor/reflected inertia — vendor 4.2e-3 vs measured
   ≈19e-3 at output, hip armature 8.4e-3 → 33.2e-3 (→38.7e-3 recommended), and
   (c) the off-diagonal inertial term 26.4e-3 that PhysX cannot express. The
   ideal J^T hip-reaction term was tested and **refuted** (injecting it made the
   hip worse).
4. **CoM_x / trim.** Correct as stated. Two additions: the trim's *first* value
   (08-11, −0.089) was framed as IMU calibration and was wrong by 0.057 against the
   08-18 crawl decomposition; the working value has the **opposite sign**, so the
   total swing is 0.178 (≈10°). "Item 4 and the tooth-skip problem are probably the
   same issue" is the record's leading candidate, not a finding: the static CoM is
   bounded ≲1 cm, screws cannot do it, and the Kinovea marker check that would
   discriminate has not been run.
5. **Handover.** Correct on the mechanism; one precision: reset zeroes **obs[0:27]**
   (projected gravity, angular velocity, linear velocity, joint pos/vel, last
   actions) and replicates that frame ×5; the bookkeeping tail (gait clock,
   commands, difficulty, grace, `init_h`) stays real. 36 × 5 = 180 dims is right.

### How each could have been found earlier (only where concrete)

- **#5 normalizer** — an offline parity check (logged obs → policy → compare to
  logged actions) is a 20-line script; it was first run 08-11. Every deployment
  since then has it (`CLAUDE.md` checklist).
- **#12 IMU signs** — a two-minute hand-tilt test with motors unpowered
  (`control/imu_sign_check.py`). Distribution audits cannot see it: a sign flip on
  a zero-mean channel is distribution-symmetric.
- **#13 handover** — a code-level diff of the two reset paths. Invisible to npz
  analysis and to video (`CLAUDE.md` item 1).
- **#10 CAN adapter** — ask the vendor what the adapter is for. They said "debug
  tool" the day they were asked (08-13).
- **#11 flag parity** — a gate that diffs the training run's `params/env.yaml`
  against `config.py` before each flight (now runbook §0.3).
- **#6/#14 belt slip** — a crawl-pose knee-symmetry check between trials (L ≈ −R
  within 3°). Nobody checked pose symmetry between trials before 08-25; the
  encoder cannot see a skip.

---

## 3. Who did what

Values: **user** / **Claude Code** / **joint** / **unknown**. "Unknown" is used for
March–July, before the campaign memory begins (08-06); commit authorship is the
user's account throughout and does not resolve it. For that period the user
attests that code and analysis were mostly produced by Claude Opus 4.6 / 4.8 via
the web app, with manual copy-paste and user-curated context — entered below as
"Claude (web), user-attested" in the Implemented column only.

| # | Fix | Noticed symptom | Localized root cause | Implemented | Verified |
|---|---|---|---|---|---|
| 1 | v1 reward rework | unknown | unknown | Claude (web), user-attested | unknown |
| 2 | Anti-gallop / stepping; zero rule | unknown (04–06); **user** spotted the zero-rule discriminator across runs (08-19) | joint (08-19) | Claude (web), user-attested (04–06); Claude Code (08-19 CLI) | joint (TB) |
| 3 | DC motor model | user (hardware failure modes per header) | unknown | Claude (web), user-attested | none recorded |
| 4 | Ball feet | unknown | unknown | user (CAD/hardware); sim switch Claude (web), user-attested | none recorded |
| 5 | Obs normalizer | unknown (trigger not recorded) | Claude Code (code vs rsl_rl source) | Claude Code | joint (script + user-run logs) |
| 6 | Belt drivetrain model | **user** ("knee motor is at the hip chassis in real, at the knee in sim"; physical tooth skip 08-25) | joint — Claude Code: drag-term forensics, armature decomposition, off-diagonal math; user: framing, and the 08-19 vendor-number challenge that forced the arbitration | Claude Code | joint (Claude Code replay/fits; user ran the bench) |
| 7 | Saturation fences | Claude Code (32 % clip forensics); **user** pushback produced the hip fence | Claude Code | Claude Code | joint |
| 8 | Delay DR | Claude Code (sim sweep) | Claude Code | Claude Code | joint; later shown not causal |
| 9 | Trim v1 −0.089 | **user** (challenged the delay-only story with "IMU offset?") | joint | Claude Code (`imu_adapter`); user (knife-edge 08-13) | user (flights) |
| 10 | CAN over SPI | user (100 Hz ceiling, hand-tuned pacing in May) | joint — Claude Code probe pinned the FIFO/turnaround and recommended the SPI HAT; user obtained vendor confirmation | joint — Claude Code driver/bench/host-id tool; user hardware install + motor NVRAM writes | joint (bench, user-run) |
| 11 | Flag parity | Claude Code (08-13 audit; 08-21 review found the revert) | Claude Code (inverted comment) | Claude Code | joint (12 trials) |
| 12 | IMU lateral signs | joint — user: hardware steps left-first; Claude Code: yaw/gyro-z inconsistency by inspection | joint — Claude Code acc-truth arbitration; **user** physically verified the 180° mount | Claude Code (config) | user (mount); re-run skipped |
| 13 | Handover contract | unknown who first flagged the violent opening | Claude Code (reset-path diff) | Claude Code | joint (pre-registered prediction; user flew) |
| 14 | Trim +0.089 | **user** (abandoned the runbook, ran the bracket) | **user** (empirical); Claude Code found the per-link CoM DR hole afterwards | user | user (n = 1) |
| 15 | P13 bundle | **user** ("stack everything that does no harm") | joint | Claude Code | joint (eval + flight) |

### Assessment against the user's account

- *"You did nearly all coding, physics derivation, and math."* Supported from
  08-06 onward (bridge feed-forward, `BeltKneeDCMotor`, armature decomposition,
  tether-bias bound, DR-sampling analysis, ~80 analysis scripts in `data/_*.py`).
  For March–July the record is silent about who wrote what; the user attests it
  was mostly Claude Opus 4.6 / 4.8 through the web app with manual copy-paste
  (user-attested), and `defecaa` shows at least one other AI reviewer was in the
  loop.
- *"You cannot perceive hardware. IMU sign, CAN adapter, belt slip — all came from
  me observing the robot."* Belt slip: fully supported — Claude Code had closed
  belt slip as a telemetry artifact on 08-14; the user saw the tooth skip on 08-25.
  IMU sign: partly — the mirror was found by analysing a recorded hand maneuver
  (acc arbitration) with a tool written for the purpose; the user's physical
  verification of the 180° mount closed it, and the trigger (hardware stepping
  left-first) was a user observation. CAN adapter: partly — the user identified the
  100 Hz ceiling in May by tuning; the mechanism (3-deep FIFO, 2.93 ms turnaround)
  and the SPI recommendation are Claude Code's 08-12 probe, and the vendor
  confirmation came through the user.
- *"You did not raise the knee decoupling gap until I named it."* Supported for the
  inertial coupling: P7's off-diagonal term is recorded as the answer to the user's
  "knee motor at the hip chassis" framing (08-18). Not resolvable for the 08-06
  belt-drag term (its trigger is not recorded). The kinematic decoupling itself
  predates the record (04-18).
- *"Your default method — proliferate experiments to eliminate hypotheses —
  repeatedly cost days."* Supported for: the 740-line `p12-4-bench-runbook.md`
  (abandoned 09-03), the tap-test armature anomaly (double-counted, retracted), the
  08-14 "belt slip is artifact" closure, and the 08-31 counterfactual replay that
  predicted "no material change" from the trim and contributed to deferring the
  one experiment that worked. Not supported as a blanket statement: the 08-17/18
  bench (two days) produced the P7 replant, and the record's strongest sim
  evidence (P8 policies ≤9 % on the corrected plant, K/R 99 %) rests on it.
- *"You never suggested the video comparison."* The record has Claude Code writing
  "record video on every future trial" (08-24) and proposing the Kinovea knee-marker
  check (09-03), but the frame-by-frame video-vs-sim comparison as the primary
  method was the user's practice; `CLAUDE.md`'s "video observations by the user
  are ground truth" rule was written after the fact (08-29/31).
- *"I supplied the instincts and the questions; you supplied execution."* Largely
  supported, with the exceptions listed next.

**Where Claude Code was decisive (record):** the normalizer bug (#5, pure code
diff); the handover contract (#13, pure code diff); the CAN adapter mechanism
(#10); rotor inertia 4.2e-3 → 19e-3 and the off-diagonal term (#6); the
telemetry-filter forensics that stopped a retrain at 0.6 knee delivery (08-14);
the viscous-DR-band and per-link-CoM-DR holes; the rsl_rl LR-ceiling diagnosis
after K3/K3s2/K4 collapsed; the swing-zero mechanism once the user spotted the
pattern.

**Claude Code hypotheses the record shows wrong:** reply-triggered CAN pacing
(08-12, refuted by own probe); "falls match eval distribution" on n = 2 (08-11);
delay as the dominant factor (08-11, user's IMU offset was larger); the 08-13
"forward" fall label (knife-edge fixed the sign); supine IMU calibration (Pi is on
the back); belt slip = artifact (08-14 vs 08-25 tooth skip); "T5 2.35 s best"
(in-hand); forward-lean idle as fall mechanism (08-25, retracted in hours); "trim
A/B: no material change expected" (08-31 vs 09-03); knee armature 6× anomaly
(09-01, double-counted).

**User hypotheses the record shows wrong:** the latency explanation for the P8
timing residual (08-18, zero-delay gate identical); the challenge that the vendor's
4.2e-3 rotor figure might be right (08-19; bench confirmed ≈19e-3 on 08-22);
contact/mat as the plant gap (drag test 09-01: inside the DR band); and, in part,
the framing of the trim as IMU calibration — the knob works, but the measured
mounting is −0.032 and what +0.089 compensates is still unidentified.

### Lessons on working with an AI coding agent on a physical robot

1. **Give it one physical fact per session and demand it be reconciled first.** The
   agent's analyses were internally consistent and wrong three times (belt slip,
   forward lean, trim replay); each was overturned within hours by one physical
   observation. Observation is cheap for the human and impossible for the agent.
2. **Have it diff code paths, not fit data, when two systems disagree.** Both
   contract bugs that mattered (#5, #13) were found by reading two code paths side
   by side. Neither was visible in any npz, video, or z-score audit.
3. **Cap the size of any experiment plan it writes.** The 740-line runbook was
   abandoned; the one-day trim bracket worked. Ask for the single cheapest
   discriminating test, pre-registered, and refuse the tree.
4. **Every channel needs one independent physical check before it anchors an
   argument.** Motor-side encoders were blind to belt slip; `torque_fdb` was
   filtered; IMU lateral signs were mirrored; the obs normalizer was missing. The
   agent will reason correctly from a lying channel indefinitely.
5. **Make it pre-register predictions and score them.** The three flights with
   pre-registered predictions (08-24, 08-31, 09-03) each produced a clean verdict
   in one session; the earlier flights produced days of post-hoc analysis.

---

## 4. Current state (2026-09-05)

**Hardware.** 6-DOF biped, EL05 × 2 (hip roll) + RS01 × 4 (hip pitch, knee), knee
via 22:30 belt from the hip chassis, TPU ball feet, Raspberry Pi 5 (user-attested),
5.57 kg flight weight (URDF 5.195 kg), ≈50 cm (user-attested). CAN over SPI (MCP2518FD HAT, SocketCAN), per-motor
host ids in motor NVRAM, zero inter-frame pacing, `CAN_TIMEOUT` 200 ms armed. IMU
Yesense 400 Hz, mounted 180° about z, signs corrected 08-28.

**Policy.** `R1r_3150` — `control/r1r_3150.pt` (`19eae76`, 08-27), raw rsl_rl
checkpoint loaded via `policy.py` checkpoint mode with the actor obs normalizer
applied; lineage K_r400 (kp 20/30 designed gains) → R1_robust (08-25 bundle) →
R1_resume3110 → model_3150. Eval: hw-matched survival 0.996, push recovery 0.86.

**Deploy contract (HEAD `efdd12d`).** `REAL_KP [6, 20, 30]`, `REAL_KD [0.3, 1.4,
1.5]` joint-space (knee mapped ×(22/30)² to motor side), action LPF α 0.60,
`KNEE_DECOUPLE_FF = True`, `HIP_REACTION_FF = False`, `SPAWN_HEIGHT 0.4401`,
`CRAWL_JOINT_POS [0, 1.59, −0.2, 0, −1.59, 0.2]` as zero reference,
`IMU_PG_OFFSET = [+0.089, 0.0]`, handover **'v' (blind, training contract)**,
`CONTROL_HZ 100`, gait phase init 0 on hardware (sim `play_sim2real.py` defaults to
π to reproduce the pre-fix left-first stepping — likely obsolete after 08-28,
untested).

**Record.** `data/p12_4_flight_20260903_v_postrim/r1r_3150_log_20260903_134700.npz`:
6.06 s, still standing at log stop, raw pitch −14.5° ± 6.7°, one 34° excursion,
crouch knees ±0.31 rad. **n = 1.**

**Known limitations.**
- **Right-knee belt slip** — one-tooth skips observed 08-25 (post catch-steps on a
  grippy mat) and mid-trial 09-03. Motor-side encoder cannot see it; the 09-01 tap
  test says the belt is elastically stiff (k_b ≥ 181 Nm/rad) with permanent creep
  (tensioner advanced ~5 months); measured catch peaks (6.5–12.6 Nm joint) exceed
  the detension kink at current tension (4.2–6.2 Nm).
- **Backward-lean idle** — the policy holds ≈ −14.5° real torso pitch (trained
  settled pg_x ≈ −0.27, a pitched crouch). Combined with tethers this ended most
  trim-0 trials; the fixed-point dead zone (only pitch *rate* wakes the policy)
  stands as a policy property.
- **Trim is uncalibrated compensation** — +0.089 is 0.12 past the measured
  mounting; the working window's forward edge is unknown (+0.13 not tried).
- **Single trial** at the working configuration.
- **Handover** relies on a training artifact (blind first frame); not robust to a
  sighted handover.

**Known inconsistencies (documented, not fixed).**
- `control/config.py` lines ~225–258: the comment block above `IMU_PG_OFFSET`
  still says "2026-08-24 … STAYS −0.089 for the P11/K flight" and describes the
  −0.089 derivation; the value is `[0.089, 0.0]`. Left untouched on 09-03 because
  the file is deployed on the Pi at a pinned commit (`docs/p12-4-session-20260903.md`
  §5).
- `control/config.py` line ~262: "must match sim: dt=1/200, decimation=2" — the sim
  has been 1/500 × 5 since `85b6ab0` (05-26). Both give 100 Hz; comment only.
- `docs/p12-4-bench-runbook.md` still carries the retracted 11/14 Hz tap numbers
  and the 6× armature claim; memory `knee-armature-tap-contradiction` is the
  correction.
- `control/README.md` documents the Phase-6 *walking* deploy; the architecture
  section is current, the procedure is not.
- Sim npz written by `play_log.py`/`play_sim2real.py` before 08-31 have the knee
  `target` channel ×2 (`CLAUDE.md` KNOWN DATA BUG).
- rsl_rl `ppo.py` LR-ceiling patch (v2, 1e-3) is not upstream. It ships as
  `RL/patches/rsl_rl_lr_ceiling_v2.patch` and must be re-applied after any env
  reinstall.
- **Uncommitted on the Windows workstation as of 09-05:** the 09-03 rewrite of
  `CLAUDE.md` checklist item 8 (trim-history decode rule),
  `docs/p12-4-session-20260903.md`, `data/p12_4_flight_20260903_v_postrim/`
  (gitignored data, expected), and this document. The Pi and any fresh clone do
  not have them until they are committed and pushed.

**Pi vs HEAD (verified 09-05 over ssh).** Both at `efdd12d` (09-03), branch
`main`, no divergence from `origin/main`. Pi working tree clean except one
untracked file, `control/baseline_can2spi_20260816_155207.json.bak_pre_cantimeout`
(a bench-baseline backup, harmless). Pi `config.py` reads `IMU_PG_OFFSET =
[0.089, 0.0]`, `KNEE_DECOUPLE_FF = True`, `REAL_KP [6, 20, 30]`,
`CAN2SPI_PACING_S = 0.0`, IMU roll/gyro-x/gyro-z signs −1/−1/+1 — identical to
HEAD, so the stale comment block above `IMU_PG_OFFSET` is present on the Pi too.
`balance_log.py` loads `POLICY_PATH = "r1r_3150.pt"` (its trailing comment still
says "P11 primary (K4b_3600)" — another stale comment). Policy files on the Pi:
`r1r_3150.pt` (6.68 MB raw checkpoint, 08-28), `policy_L1_4330.pt`,
`policy_K4b3600.pt`, `policy_K4_3420.pt`, `policy_G2i4760.pt`, `policy_G2d4760.pt`,
`policy_F3490.pt`, `model_4595_2000hz_std20_jk31.pt` (05-25).

---

## 5. Next steps (short)

**P0 — right-knee belt slip.** The only active work item. Frame: reliability
*and* state estimation — the motor-side encoder cannot observe the joint after a
skip, so the policy's knee observation and the deploy zero silently drift by
0.21 rad per tooth. Options, not a plan:
- crawl-pose zero check (and re-zero) before every trial — current mitigation;
- aluminium pulleys, same tooth count;
- wider belt and/or an idler to increase wrap/tension margin;
- threadlocker on the tension screw (creep site unidentified: magnet/weld check
  pending, `knee-belt-coupling-findings`);
- knee-side magnetic encoder (gives joint-side truth; also the discriminator for
  the trim question).

**Parked** (one line each; pointer = memory file):
- Trim repeats, n ≥ 5 at +0.089 — `p12-4-notrim-flight`.
- Bracket step +0.13 to find the forward edge; deploy at window centre —
  `p12-4-notrim-flight`.
- Backward room: remove the tether constraint that ended most trim-0 trials —
  `p12-4-notrim-flight`.
- Retrain directive: coherent whole-body CoM_x DR ±2–3 cm; per-leg knee-angle-bias
  DR ±0.1–0.2 rad; static-lean reset DR at zero rate; handover-contract
  randomization; `dr_proj_grav_bias_max ≥ 0.12`; mass recentre ×1.072 —
  `next-training-directive`.
- Belt tension pluck check (140 Hz floor) each session — `knee-belt-coupling-findings`.

The user's time now goes to release and outreach. This document, `CLAUDE.md`, and
`docs/p12-4-session-20260903.md` are the entry points for anyone (including a
future agent session) resuming the work without the user present.

---

## 6. Sources read, and what could not be verified

**Read in full:** `git log --all` (372 commits) with `--stat`/diffs of `7c419e6`,
`85b6ab0`, `ecac7e1`, `f1fbc97`, `ae64952`, `1c8658f`, `26fe2fe`, `221e397`,
`e066509`, `54d97d7`, `26921c1`, `423ace7`, `efdd12d`, `19eae76`, `2d0441f`,
`415257e`, `500032b`, `651cfd3`, `2d63347`, `61136fe`, `40cad10`; `CLAUDE.md`;
all 42 campaign memory files (the agent's private session notes);
`docs/p12-4-session-20260903.md`, `docs/imu-sign-check.md`,
`docs/knee-hip-coupling-review-20260821.md`, `docs/PREDICTIONS_p12_3.md` (head),
`docs/p9-checkpoint-selection-20260823.md` (head); `README.md` (v1 post-mortem);
`RL/README.md`; `control/README.md` (head); `reports/belt_coupling_sim2real/report.md`
(head); `control/config.py` (constants and the IMU/trim block);
`control/motor_bridge.py` (frame conversions); `control/balance_log.py` (handover
keys); `RL/.../rl_env_cfg_p6.py` header and `rl_env_cfg_p1..p5.py` headers;
`data/` folder and file listing with per-folder `README.txt` where present.

**Not read:** `docs/replant-report-20260817.html`, `docs/replant-runbook.html`,
`docs/p11-hardware-runbook.md`, `docs/p12-fall-isolation-runbook.md`,
`docs/p12-4-bench-runbook.md` (740 lines; its retracted content is covered by
memory), the npz data themselves, `rl_env_p6.py` body, the tensorboard runs.

**Could not verify (no record, no user recollection offered):**
- Whether the ball feet (06-07/12) changed hardware outcomes.
- Whether the `DCMotorCfg` switch (05-26) was necessary.
- Whether the 08-28 18:00 session ran the corrected IMU signs (fix committed the
  same afternoon; not stated in the folder README). Bears on #13's 0.34 → 0.67 s.
- Per-trial outcomes of the 08-27 K4b_3600 session.

**User-attested (recollection, not record):** these statements come from the
user's memory on 2026-09-05 and are tagged "(user-attested)" where the body uses
them. They are not contradicted by anything in the record.
- **06-29 → 08-06 gap:** no work. Tool improvements are made; work resumed
  when a new model release in Claude Code (user's term: "fable5") restored
  confidence. (§1 Phase B.)
- **Pre-08-06 code and analysis:** mostly Claude Opus 4.6 / 4.8 via the web app,
  manual copy-paste, user-curated context. Entered in §3 as "Claude (web),
  user-attested" for the Implemented column of fixes 1–4; Noticed/Localized stay
  unknown.
- **Ball feet 06-07/12:** a design judgment — the original point-foot shape was
  considered absurd and a sphere assumed better. No test was run. Effect remains
  unverified (§1, §2 #4).
- **`DCMotorCfg` 05-26:** the user recalls no A/B; necessity remains unverified
  (§2 #3).
- **08-28 18:00 IMU-sign status:** left unresolved by user decision; the
  confound on #13 is noted in its verification cell.
- **"50 cm" tall and "Raspberry Pi 5":** not in the repo (§0, §4).

**Resolved during writing (09-05):** the Pi is at `efdd12d` = HEAD with a clean
tree (§4); `control/export_policy.py` now uses the exact rsl_rl normalizer keys and
formula (`actor_obs_normalizer._mean/_std`, `(x − mean)/(std + 1e-2)`), so the
08-06 export-path bug note in memory `obs-normalizer-deployment-bug` is stale.
