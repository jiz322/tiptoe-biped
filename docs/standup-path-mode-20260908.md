# Stand-up redesign: from the P6 linear ramp to the seiza → crouch → squat path (2026-09-08)

Status: implemented and training (S3a_path / S3b_path, GPUs 0/1). This note records
what was found, what changed in the env, how it was validated, and what is still open.
Freshness ranking for the stand-up topic is now: this note > memory `standup-path-mode`
> `standup-ramp-topple-finding` > older curriculum notes.

## 1. Why

The user requires stand-up from difficulty d = 1.0 — the ground-rest pose the robot is
placed in (cfg `kneel_*`: hip ±3.09, knee ∓3.06, torso pitch −90°). The S-line
curriculum runs (S1c/S2c → S1d/S2d) advanced at ≈ +2 pts stand-up success per 100
iterations and never cleared d ≈ 0.7. The question asked was whether the action
saturation termination or the stand-up parameters were responsible.

## 2. Findings (all on the nominal sim plant unless stated)

### 2.1 Saturation termination — not the cause
`probe_standup_ramp.py` runs the S1c checkpoint at fixed d with the fence on and off.
Survival is identical (d=0.5: 0.70 vs 0.69; d=0.6: 0.31 vs 0.33; d≥0.7: 0.00 both) and
96–100 % of first deaths at d ≥ 0.7 are `too_low` at the step grace expires. Separately,
the fence ends ≈ 2/3 of *all* episodes in the stand branch (`sat_term_frac` 0.0009/step
vs ≈ 0.0013 total; knee/hip EMA maxima pinned at the 0.5/0.55 thresholds) — a
balance-branch matter, unrelated to the curriculum wall.

### 2.2 The stand-up was a scripted linear joint-space ramp, and it topples by itself
During grace the PD action center blended linearly from the spawn pose to the standing
pose over the grace window (`_advance_motor_target`, P6 era); the policy only added
±0.3/0.75/1.025 rad on a ≈3 rad travel. With **zero policy action** this ramp fails at
every d ≥ 0.5 (100 % `too_low`). Torque is not binding (knee saturation duty ≈ 0), and
slower ramps do not help (8 s / 12 s grace: zero-action still 0, the trained policy gets
worse). Side bug: `grace_frac` was normalized by the *maximum* grace, so a d<1 spawn's
center started at blend d² instead of d — an instant PD kick (≈0.65 rad knee error at
d = 0.5).

### 2.3 Geometry (this changed the design)
Measured with `probe_pose_dump.py` (body positions are stale right after `reset()` —
only post-step values are valid):

* **d = 1.0 is a seiza.** Knee joints and foot balls on the ground, shins flat, knees
  14 cm *behind* the balls, torso vertical, root 0.235 m, CoM at the front edge of the
  knee→ball support. Standing is a near-horizontal torso (root 0.14 m ahead of the hips),
  hips 4 cm behind the balls, CoM 1 cm ahead.
* **`_get_toe_pos_w` is not the contact.** It returns the nail-frame origin, 8.6 cm up
  the shin in standing. The actual contact is a ball of r ≈ 0.0205 m (USD collision
  mesh `part_3`), centre 5.8 cm further along the shin (shin-frame: leg 1
  (−0.196, 0.0286, −0.018), leg 2 (0.0015, 0.199, 0.018)). With the shin flat the ball
  is ~6 cm ahead of the nail point, so every earlier "CoM ahead of toe" number was
  biased by up to 6 cm. All probes now report `com_ball`.
* **Quasi-static balance manifold** (`probe_balance_manifold.py` → 
  `RL/logs/balance_manifold_ball.csv`, plot `data/_plot_balance_manifold.py`): for each
  symmetric (hip, knee) the torso pitch θ* that puts the CoM over the balls, plus CoM /
  root / knee heights there. Two facts: (a) along the old linear path θ* runs smoothly
  −67° → −4°, so the ramp was quasi-statically "ideal" but demanded a 22°/s torso rotation
  the heavy torso (3.9 of 5.2 kg incl. hip motors) cannot follow — it lags, the CoM falls
  behind the balls, the robot rocks back onto its knees, the legs slide forward and it
  sits on its hips ("seal"); (b) **no pose in the valid grid balances on the balls with
  the knees down** (knee height at θ* ≥ 8 cm everywhere). Every kneeling rest therefore
  has the CoM 2–4 cm behind the balls; pitching the torso further forward only folds the
  body over the knees; knee extension removes the rear support without moving the CoM.
  ⇒ the kneel → balls liftoff is a small dynamic rock (~3 cm of CoM) that no open-loop
  ramp can do. It is the policy's job.

### 2.4 Shape search (`probe_standup_keyframes.py`, rounds r1–r6 in `RL/logs/standup_kf_*`)
Eight timing variants of the two-joint ramp (hips-first, knees-first, overlaps, pauses)
all fail at d ≥ 0.7 in both zero and policy mode. The two-phase path below is the only
shape whose first phase survives with zero action: seiza → K1 crouch (hip 1.10,
knee −2.50 target, torso ≈ +3…+9°) keeps the knees down and the CoM within ±2 cm of the
balls the whole way, and **holding K1 is statically stable** (kcHold: 2.2 s, zero action,
zero pitch rate). From K1 a straight joint-space line to standing keeps θ* within
+15° → −4.5° (a plain squat contour).

## 3. What changed in the env (`rl_env_cfg_p6.py` §6, `rl_env_p6.py`)

Defaults are bit-exact legacy; every new knob is opt-in.

* `standup_ramp_mode = "linear" | "keyframe" | "path"`.
  * `keyframe`: per-joint piecewise-linear timing profiles (`standup_hip_profile`,
    `standup_knee_profile`, flat `[s,f,…]`) on the env's *own* grace — also fixes the d²
    kick. Used by the probes.
  * `path` (the S3 line): absolute keyframe table `standup_path_knots`, rows
    `[s, hip_target, knee_target, pitch_deg, root_h_m, hip_rest, knee_rest]`
    (leg 2 mirrored, hip_roll 0). Difficulty d spawns **on the path** at phase
    s₀ = 1 − d: joint state = the *rest* columns, root quaternion = rotation about +y by
    `pitch_deg` (kneel = −90° ↔ `kneel_quat`), height = `root_h` +
    `standup_path_spawn_clearance_m`. The action center follows the *target* columns
    from s₀ to 1 over the assigned grace (constant path speed); spawn noise perturbs the
    written state only. Phase A (s ≤ 0.45) = seiza → K1 on knees + balls; phase B
    (s > 0.45) = the squat line, rest = target, spawns are balanced-on-balls crouches.
* Grace-phase shaping (path mode only): `rew_scale_standup_track` ×
  exp(−(Δpitch/`standup_track_sigma_rad`)²) against the path pitch, and
  `rew_scale_standup_contact` × fraction of feet down — reward slots 38/39
  (`standup_track`, `standup_contact`). The root-height term is kept; on its own it
  *penalizes* the correct motion because the root drops 0.24 → 0.18 m during phase A.
  New TB channel `standup/pitch_err_grace_mean`.
* Curriculum knobs from earlier the same day (S1d/S2d line): `curriculum_frontier_frac`
  / `curriculum_frontier_width` (half the kneel resets sample d near alpha — plain
  U[0, alpha] put only ~6 % of episodes at the frontier) and
  `curriculum_advance_metric = "success_high_rel"` (high/low tertile success ratio; the
  low band caps at 0.80/0.68 from spawn/push falls, so an absolute bar penalizes arm B).

### 3.1 The knot-table bug (found by `script/show_standup_path.py -Scan`, fixed same day)
A resting state under PD is a *pair*: the target the servos hold and the joints the plant
sags to (~0.15 rad; the knee pins at its −3.06 limit while the shin is flat). The first
table paired ramp targets with the pitch/height of a *moving* population (torso rotating
~60°/s), the second used the sagged joints as targets (double sag). Symptoms: K1
(s = 0.10) ejected at 3 m/s in the viewer; in the env a 0.5–0.8 m/s depenetration kick
for s₀ ∈ [0.04, 0.12] and 3–5° / ≤ 3 cm settling for every phase-A spawn. Fix: each
phase-A target held 4 s with zero action until Δ ≈ 0 (`holdA*` candidates), both target
and rest joints stored. Acceptance now, both tools: env scan
(`probe_pose_dump.py --scan`, training gains) rests within 0.3° / 2 mm at ≤ 0.1 m/s on
every row; viewer `-Scan 0.0:0.5:0.025` reads "ok" on every row. Rule for future edits:
measure (target, rest joints, pitch, height) with a hold, and run both scans before
training on a table.

## 4. Tools added

| file | purpose |
|---|---|
| `RL/scripts/rsl_rl/probe_standup_ramp.py` | fixed-d survival table: zero / policy / no-fence, grace sweep, first-death cause, knee saturation duty |
| `RL/scripts/rsl_rl/probe_standup_traj.py` + `data/_plot_standup_traj.py` | per-step population traces of the legacy ramp |
| `RL/scripts/rsl_rl/probe_standup_keyframes.py` + `data/_plot_standup_keyframes.py` | shape search: candidates `"legacy;path;name:hip_profile:knee_profile"` (one `;`-separated argument), ball-referenced CoM, knee height, ground-contact map |
| `RL/scripts/rsl_rl/probe_pose_dump.py` | body / ball / CoM geometry of spawn or held poses; `--scan` = spawn-consistency scan over d |
| `RL/scripts/rsl_rl/probe_balance_manifold.py` + `data/_plot_balance_manifold.py` | quasi-static θ*(hip, knee) manifold with the sim as FK (airborne one-step trick) |
| `script/show_standup_path.py`, `script/run_show_standup_path.ps1` | (other session) Isaac Sim viewer of the path: knot buttons, phase/difficulty sliders, gain presets, `-Smoke`, `-Scan`; reads the cfg table by AST, now 5- or 7-column |
| `RL/scripts/rsl_rl/eval_alpha_success.py`, `RL/scripts/patch_curriculum_normalizer.py`, `RL/scripts/rsl_rl/verify_com_shift.py` | S-line curriculum / CoM-DR support (09-06/07) |

Probe outputs live in `RL/logs/` (untracked): `standup_kf_r{1..6}_*`, `standup_path_r{5,6}_*`,
`standup_holdA.*`, `balance_manifold{,_ball}.{csv,png}`, `probe_*.{out,err}.log`.

## 5. Runs

* S1d (`launch_train_S1d_plantfix.ps1`) stopped at `model_5110` (alpha 0.7), S2d at
  `model_5180` (alpha 0.6) — user-authorized; under the linear ramp they could not have
  passed d ≈ 0.7.
* `launch_train_S3a_path.ps1` (GPU 0, arm A recipe, resumed S1d 5110) and
  `launch_train_S3b_path.ps1` (GPU 1, arm B recipe, resumed S2d 5180): path mode, grace
  5 s (constant path speed; d = 1 takes 5 s, phase B ≈ 2.75 s), track 4 / contact 2,
  curriculum restarted at alpha 0.5 (= balls-only crouch band; alpha ≥ 0.6 brings in the
  liftoff). Relaunched late 09-08 from S3a `model_5230` / S3b `model_5300` after the
  knot-table fix (phase-B rows were unchanged, so the alpha-0.5 progress carries over).
* Watch: `curriculum/standup_success_{high,low}`, `standup/pitch_err_grace_mean`,
  `reward/standup_track_mean`, alpha. Pre-registered verdict: if `standup_success_high`
  stays ≈ 0 and the pitch error does not shrink for > 800 iterations at alpha 0.6, the
  dynamic liftoff is not being discovered → next levers are a scripted rock burst at K1
  or a liftoff-specific reward, not PPO knobs.
* Any resume/eval of S3 checkpoints must re-pass the path-mode overrides (the table
  itself is code, the mode/rewards/grace are CLI).

## 6. Open items

1. **Real-robot pose check** (user): confirm the robot's ground-rest pose is the seiza as
   the sim has it (knees *and* feet touching, shins flat, torso vertical) and, if
   possible, that it rests in the K1 crouch (hip ≈ 1.1, knee ≈ −2.6 rad, torso ≈ +3…+9°,
   knees down). If the real rest has the knees up or the torso folded onto the thighs,
   phase A must be re-measured.
2. The liftoff (kneel → balls) is the crux and appears only at d > 0.55; whether PPO
   finds the rock from the kneel is the open experiment.
3. The saturation fence ending ≈ 2/3 of stand-branch episodes deserves its own look.
4. Foot geometry: the FK constant `_FK_TARGET_FOOT_Z` (0.0856) and the no-flight contact
   threshold (0.084) are consistent with the nail point, but any CoM-margin or contact
   argument must use the ball.
