# Sit-down campaign — plan (2026-10-01)

**Session**: plan only, no implementation. **Campaign line**: SIT (S9). **Base policy**:
`s8b_8880` (`control/s8b_8880.pt`, byte-identical to
`RL/logs/rsl_rl/p9_replant/2026-09-19_08-14-51_S8b_air03/model_8880.pt`; best windowed
mean episode length over all its checkpoints 2156.5; the checkpoint that steps on
hardware per `docs/t0-s8b-flight-analysis-20260930.md`).

**Decisions taken by the user this session** (asked and answered, not assumed):

1. "Sit" = the **seiza rest pose** — the d=1.0 end of the existing stand-up path
   (knees and foot balls on the ground, torso vertical). Not a deep squat, not a
   controlled collapse.
2. Architecture: **try no new network first** — reverse-replay the existing path
   under the unmodified s8b policy before any checkpoint surgery or training.
3. Safety: the auto-sit trigger lives in a **deploy-side watchdog** (`control/`),
   not in the policy.

Everything below is the campaign built on those three decisions.

---

## 1. The two facts that decide the whole campaign

### 1.1 The 5 s counter is not just a clock — it is the termination shield

The user's instinct to reuse the stand-up counter is right, for a stronger reason than
timing.

`rl_env_p6._get_dones`: `terminated = post_grace_terminate & ~in_grace & ~shielded`, with

| fence | value | seiza value | verdict |
|---|---|---|---|
| `min_height` | 0.30 m | root z **0.235 + 0.006** | **inside the fence** |
| `max_tilt_rad` | 1.8 rad | path pitch reaches **−90°** | **inside the fence** |
| `max_hip_roll_rad` / split fences | 1.8 rad | 0 | outside (safe) |

The entire lower half of the sit trajectory, and its final pose, is past the kill line.
The stand-up lives with this because `in_grace` suppresses every fence; that same
suppression is what makes a sit expressible at all. Two hard requirements follow:

1. The sit must run under the same grace counter (`standup_grace_seconds x d`; at d=1
   that is **5.0 s = 500 steps @ 100 Hz** — the same counter the user proposed).
2. The **seated hold after the counter expires needs an explicit exception**, otherwise
   the robot is terminated the instant it arrives, exactly like "a d=1.0 episode that
   never rises" (cfg comment at `rl_env_p6.py:671`). A sit branch cannot be a
   sign-flipped copy of the stand-up branch for this reason alone.

### 1.2 The phase channel the user wanted to append already exists

The instinct was to append a channel to the trained checkpoint "to save time". It is
not needed: obs ch 30 (`grace_norm`) is already a trained, in-distribution phase signal
in [0, 1].

* During a d=1.0 stand-up, `grace_norm = 1 − progress` and the path phase is
  `s = progress`, so `grace_norm = 1 − s` for the whole episode.
* A sit is that same traversal driven the other way: rho 0 → 1 with
  `action_center = path(1 − rho)` and obs ch 30 fed **rho**.

The policy never emits the trajectory. The env's `action_center` does, and the network
only adds `± action_scale` on top — that is the stand-up design. A reversed traversal
therefore presents the network with the same set of *(grace_norm, pose)* pairs it
trained on, in reverse order. A memoryless actor with a 5-frame history cannot tell
order.

This is the cheapest possible read of the user's own idea, and it costs zero checkpoint
surgery.

### 1.3 What that argument does not prove

Time reversal is not symmetry: gravity, contact and friction are not reversible. The
seiza entry is a documented **dynamic rock** onto the balls (no static pose balances on
the balls with knees down — `RL/logs/balance_manifold_ball.csv`), and phase A (s ≤ 0.45)
is described as passively stable while phase B (s > 0.45) is the straight-squat line at
theta* ≈ +1.7°. Descending means entering the rock instead of leaving it. Whether the
trained tracking behaviour inverts is empirical, which is why arm Sit-0 exists and why
it comes first.

---

## 2. Campaign arms

Each arm is gated on the previous one. No arm starts before its predecessor's verdict
is written down.

### Sit-0 — reversed-path replay, no training, no env change (~1 day desk, 0 GPU-hours)

**Question**: does the unmodified s8b policy track the path downward, with the scripted
action center carrying the motion?

**Build**: a thin play-side harness (new `RL/scripts/rsl_rl/play_sit.py`, or a `--sit`
flag on `play_alpha.py`) reusing `play_sim2real`'s recipe loader, doing only this: spawn
standing (difficulty 0 pose, `init_h` 0.434/0.439 as trained), then drive
`action_center = path(1 − rho)` with rho linear over 5.0 s (500 steps), feeding
`grace_norm = rho` into obs ch 30 and `difficulty = 1.0`, `init_h = 0.241` — the exact
d=1.0 train-time triple, the only triple at which `path(1−rho)` was ever seen.

Also run the **`action = 0` variant** (scripted center alone; "zero" mode as in
`probe_standup_traj.py`). It splits the failure cleanly:

| zero-action result | policy result | reading |
|---|---|---|
| reaches seiza | reaches seiza | **campaign over — ship the deploy path, no training** |
| reaches seiza | falls | trajectory is fine, the policy fights it → Sit-2 |
| falls | falls | trajectory shape is the problem (the 09-08 linear-ramp topple class) → Sit-1 |

**Pre-registered H-S0a**: at rho = 1, both toes below `no_flight_contact_threshold`
(0.084), root z ≤ 0.30, torso pitch within ±10° of the path reference at every s, and
**0 falls over 20 seeded rollouts**. Verdict is supported / contradicted /
inconclusive, with the discriminator named as the first s at which the tracking error
exceeds 15°.

**Cost**: zero training hours, reuses `probe_standup_traj.py` / `play_alpha.py`
infrastructure, one GPU. Both 4090s were at 0 % utilisation at planning time.

### Sit-1 — trajectory diagnosis (only if the scripted center alone fails)

Cheap probes, in this order:

1. `script/run_show_standup_path.ps1 -Smoke` on the reversed table: is every row
   reachable at all, or do rows spawn inside the floor (the
   `standup-path-k0k1-penetration` failure mode)?
2. `probe_standup_ramp.py` in both directions, to read the topple point per direction.
3. If a specific s-interval topples: re-time the knots over that interval. Constant path
   speed is what the s values encode, and **a new table means Sit-2 plus a retrain**,
   because the trained policy saw exactly one speed profile.

**Pre-registered**: if the reversed table topples in the same s-band as the forward
table topples at the same speed, the wall is the pose family, not the direction, and the
seiza target itself is in question. That verdict returns to the user.

### Sit-2 — sit branch in training (only if Sit-0/Sit-1 leave a policy problem)

**One net**, resumed from `s8b_8880`, with a sit branch alongside the stand-up branch.

* `sit_enable`, `sit_branch_probability` (start 0.3, A/B against 0.6),
  `sit_path_knots` **defaulting to the existing 14-row table read from the stand-up
  cfg** — no duplicated literals.
* `_sit_phase()` mirroring `_standup_phase()`, reusing `_grace_remaining` **as the sit
  counter**, so the fence shield of section 1.1 comes for free and obs ch 30 keeps its
  trained meaning. Direction is a per-env flag, not a second counter.
* Sit-branch reset: spawn **standing** at the trained standing pose, `difficulty = 1.0`,
  `init_h = 0.241`.
* Rewards: **reuse the existing grace-gated trio with the reference reversed** —
  `rew_scale_standup_track` (pitch vs `path_pitch(1−rho)`), `rew_scale_standup_contact`
  (feet down), `rew_scale_standup_height` / `_fall` against `path_h(1−rho)`. This is the
  largest saving in the arm: the shaping exists and is already tuned.
* New: a **seated-rest exception** in `_get_dones` — after rho = 1, suppress `too_low`
  (and only `too_low`) while the pose is inside the seated band; keep every other fence.
  Without it, success is indistinguishable from death.
* Curriculum: mirror the stand-up's `success_high_rel` logic with a sit metric
  (`sit_success_high` = fraction of sit-branch episodes reaching rho = 1 with both knees
  grounded and surviving `curriculum_success_post_grace_steps` after the counter).

### Sit-3 — hardware (watchdog armed, mat down, low height)

Runs only after Sit-2's gate. Needs the deploy-side work that does not exist today
(section 4 and section 5).

---

## 3. Why not the two alternatives

**Separate sit network plus handoff** — rejected. The repo's most expensive interface bug
was exactly a handoff contract (08-29: training starts blind with a zeroed obs-delay
buffer, deploy fed the real frame x5, and the policy answers those two states with
entirely different opening actions). A second net means owning that contract again, plus
normalizer-state and delay-buffer handoff, to buy nothing the reversed path plus one
branch does not already buy.

**Appending a channel to the s8b checkpoint** — deferred, and not for the reason it
looks. The patch itself is 30 lines; the problem is layout. The channel must be
appended **to every frame of the 5-frame stack**, and the existing surgery tools do not
do that:

* `RL/scripts/patch_checkpoint.py` appends to the **end of the flat vector** (180 to
  181). That means "one extra trailing channel", not "one extra channel per frame". The
  net loads, runs, and is silently wrong.
* Correct layout is a new dim at index `36*k + 36` for k = 0..4, i.e. 185 dims, with the
  critic first layer going 258 to 263.
* `_OBS_MIRROR_PERMUTATION` / `_OBS_MIRROR_SIGN` (`rl_env_p6.py:224-260`) are built for
  36-blocks and would need a rebuild. A wrong mirror table is invisible in training and
  fatal in deploy.
* The 36 hard-codes are spread over `rl_env_cfg_p6.py:497`, `balance_log.py`
  (`OBS_PER_FRAME`), `control/config.py:315`, `policy_contract_check.py` and
  `obs_parity_live.py`. Five sites, one contract.

It stays on the table as the fallback if Sit-2 cannot converge with the existing
channel — but it is the *expensive* option, not the cheap one, and the user's stated
reason for wanting it, saving time, is served better by section 1.2.

---

## 4. Safety watchdog

### 4.1 Honest limit first

A sit-down takes the full counter (5 s). A fall develops in about 0.16 s, and the fence
is crossed at 25° tilt. **The watchdog cannot catch a fall that is already happening.**
What it can do: trip on the pre-fall signature before the point of no return, park the
robot under control instead of letting it hit the ground hot, and stop spending motor
thermal budget on a losing fight. Anything sold as more than that would be wrong.

### 4.2 Where it lives

`control/` — a new `watchdog.py`, a pure function of measured channels plus small state,
called once per tick from the flown script.

| tripwire | threshold | why this one |
|---|---|---|
| raw pitch | abs(raw pg_x) > 12° sustained 0.1 s | **use the raw channel, not the trimmed one** — `IMU_PG_OFFSET` changed twice and logged `proj_grav` is trim-dependent (checklist item 8). 12° sits above the trained band edge (0.10 rad = 5.7°) and well below the 25° fence |
| pitch rate | abs(omega_y) > 60 deg/s | the 09-31 finding: no fixed point exists at abs(omega_y) ≥ 25 deg/s, and rate is the one channel that reliably wakes the policy. 60 deg/s is 2.4x that, so it fires only after the policy has had its chance |
| knee saturation EMA | fraction of the last 0.5 s with abs(raw action_knee) = 1 above 0.5 | **this is the trained termination fence itself** (`sat_term_threshold` 0.5, `sat_term_tau_s` 0.5). Reusing it means the hardware trip fires on the same signal the policy treats as death |
| bus health | 3+ dropped replies in 100 ticks, or motor temp ≥ 40 °C | both already logged; both are run-killers |
| operator | any key → SIT | trivial and the most reliable tripwire; keep it |

**Commit, do not oscillate**: once SIT triggers it runs to completion (rho 0 → 1) or
aborts to a ground rest. A mid-sit reversal from a rotated state is worse than either
endpoint — the same reason `balance.py` gates NN control behind OBSERVE.

### 4.3 Its acceptance test needs no robot

Replay the four 09-30 npz in `data/t0_s8b_flight_20260930/` through the watchdog
offline, and assert it fires **before** the documented failure points (the T0 trials'
last abs(pg_x) < 0.15 at 1.02 s; S8's pitch drift past −0.196 at the last tick) and
**does not fire** in the first 3 s of the two S8 trials. That is a desk test on data
already on disk, and it is the gate for arming it on hardware.

---

## 5. Naming, files, recipe

**Runs**: `SIT0_replay` (play side, no training), `SIT1_*` (probes), `SIT2_sit0p3` /
`SIT2_sit0p6` (the A/B on `sit_branch_probability`). Launch scripts
`RL/launch_train_SIT2_*.ps1`, copied from `launch_train_S8b_air03.ps1` with its header
convention.

**Files to touch when implementation starts** (not this session):

| file | change |
|---|---|
| `RL/.../standup_curriculum/rl_env_cfg_p6.py` | `sit_enable`, `sit_branch_probability`, `sit_path_knots` (read from the stand-up table), sit curriculum keys |
| `RL/.../standup_curriculum/rl_env_p6.py` | `_sit_phase()`, action-center branch, sit-branch reset, seated-rest exception in `_get_dones` |
| `RL/scripts/rsl_rl/play_sit.py` | Sit-0 harness (reversed center, `grace_norm = rho`) |
| `control/config.py` | the 14x7 knot table (deploy has **none** today) plus `SIT_GRACE_SECONDS = 5.0` |
| `control/controller.py` | a `SIT` state beside `STANDUP`; today `_apply_action` uses the **linear** grace blend, not path mode |
| `control/watchdog.py` | new, section 4 |
| `control/balance_log.py` | call the watchdog; SIT branch in the action center |

**Recipe discipline — the two silent-revert traps that will bite this campaign:**

1. **Rate keys.** S8b's as-run yaml is 2000 Hz / `decimation` 20 / `render_interval` 20 /
   `rotor_coupling_dt` 0.0005 / `series_dt` 0.0005 / `series_substeps` 4. The **current
   cfg defaults are the T-line 500 Hz set** (`decimation = 5`, `dt = 1/500`,
   `rotor_coupling_dt = 1/500`, `hold_decimation = 5`), changed 09-26 *after* S8b
   trained. The S8b launcher passes none of them because it did not have to. The SIT2
   launcher **must pin all six explicitly**, or the plant silently becomes a different
   one. (Same hazard as D2 in
   `RL/analysis/pinned_knob_sweep/20261001_0851/DECISIONS.md`.)
2. **Resume overrides.** A resume must re-pass every override or it reverts to cfg
   defaults (the G-run failure). The SIT2 script must carry S8b's full list: servo gains
   20/1.4 and 30/1.5, `feet_air_height_threshold` 0.03, the belt band 100–1000 with
   damping 0–0.3, `foot_restitution` 0.63 plus DR 0.2–0.8, `dr_armature_mult` 0.8–1.2,
   `dr_proj_grav_bias_max` 0.00, `handover_prime_prob` 0.0, `standup_grace_seconds` 5.0,
   `standup_path_spawn_clearance_m` 0.006, `rew_scale_standup_track` 4.0 /
   `_contact` 2.0, `pickup_enable=false`, and the curriculum block (`success_high_rel`,
   0.7, frontier 0.5/0.3, kneel 0.6).

**One live mismatch to fix while here**: `control/config.py:289` has
`STANDUP_GRACE_SECONDS = 4.0` (so `MAX_GRACE_STEPS` = 400) while S8b trained at **5.0**
(500 steps). It is currently inert because the flown script freezes grace at 0, but a
sit deploy will use that counter, so it must be corrected to 5.0 before Sit-3, or the
deploy descends 20 % faster than the trained path speed.

---

## 6. Pre-registered predictions, gates and kills

**Predictions (written before any run):**

* **P1** — Sit-0's `action = 0` variant reaches seiza. The path is a scripted PD target
  family the env already spawns on and holds, and the 09-08 finding was that the
  *linear* ramp topples, which is why path mode exists.
* **P2** — Sit-0's policy variant tracks phase A (s ≤ 0.45, the passively stable
  kneel/ball region) and, if it fails, fails **at the entry rock** — the small dynamic
  rock onto the balls, reversed — not at the top.
* **P3** — the risk axis at the seated rest is roll: hip_roll is the weakest actuator
  (6 Nm, kp 6 / kd 0.3) and the seiza support is a narrow base. Expect a lateral topple,
  not a pitch one.
* **P4** — the watchdog as specified in section 4 trips on all four 09-30 npz and does
  not trip in the first 3 s of either S8 trial.

**Kills (S8's convention, applied to SIT2):**

* `value_function > 42000` for 50 consecutive iters → kill.
* `episode_length < 400` at +300 iters → kill.
* Gate at +1000: `episode_length ≥ 1400` with every DR band live.
* Sit-specific gate: `sit_success_high` must rise above 0.2 within 600 iters of the
  branch going live. If it sits at 0 with the curriculum not advancing, the branch is
  not being learned and the arm is killed rather than extended.

**Metric of record** for the campaign: **both toes below 0.084 m and root z ≤ 0.30 held
for 3 s with the fences armed**. Not "looks seated on video" — the user's video read is
ground truth for *what happened*, but the campaign's number is the sustained hold.

---

## 7. Cheap-workaround check (the 09-12 rule)

Stated before any code, because the rule exists to stop exactly this kind of build:

1. **Is this a common thing?** Sit-down / crouch-to-rest is common in humanoid RL, but
   nearly always as whole-body mocap-tracked motion, or on a robot with many more DoF.
   For a 6-DOF point-foot biped with no hip yaw the seated pose is close to a kinematic
   singularity, and `RL/logs/balance_manifold_ball.csv` shows the stable pocket at the
   seiza corner is **narrow**: the neighbouring knee cells at the same hip read drift
   0.44–0.53 m and tilt 6–10°, where the corner cells read ≈0.
2. **What do others do?** They script the trajectory and let the policy stabilise around
   it — which is precisely the stand-up path-mode design already in this repo.
3. **Is there a cheaper workaround?** Yes, and it is arm Sit-0: the path table, the
   counter, the shield and the phase channel all already exist. Sit-0 is a play-side
   harness, not a training change.
4. **Rough cost.** Sit-0 about one desk day, no GPU, no training. Sit-1 about one day.
   Sit-2 about one day of env code plus 1–2 days on one 4090. Sit-3 follows the existing
   hardware-run discipline.

The campaign is ordered so that the cheapest arm is also the most likely to end it.

---

## 8. What would falsify this plan

* **P1 contradicted** (the scripted reversed center alone cannot reach the seiza rest):
  the trajectory family is wrong, not the policy, and the seiza target returns to the
  user for a geometry decision.
* **P2 contradicted with s ≤ 0.45 clean**: the failure is in the straight-squat phase,
  the region the T-line spent four arms on. That re-opens the stance question, not the
  sit question.
* **The 20-knob sweep's sensitive set matters here**: K22 (contract), K2 (pg-bias), K3
  (stiffness) and K13 (graded) are the four of 22 that are not inert
  (`RL/analysis/pinned_knob_sweep/20261001_0851/ranking.md`). A sit that works at
  baseline and fails at a K2 or K3 edge is a robustness finding, not a sit failure, and
  the sit arm should not be extended to chase it.

---

## 9. Next session, first actions

1. Write `RL/scripts/rsl_rl/play_sit.py` (Sit-0) — the only new file the first arm needs.
2. Run it in the `zero` and `policy` variants at 20 seeds, plus the `-Smoke` check of the
   reversed table.
3. Write the verdict against H-S0a and stop. If supported, the campaign is over and the
   deliverable is a deploy path, not a policy.
