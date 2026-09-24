# P9 checkpoint selection — 2026-08-23

Deterministic-policy eval sweep for the P11 deploy pick, run on GPU 2 while
K4b / K4b2 trained on GPUs 0/1. Method, artifacts, and how to re-run are at
the bottom. Companion to [p11-hardware-runbook.md](p11-hardware-runbook.md)
(fills §0.1, executes §0.2 in eval form).

## TL;DR

> **08-27 update:** superseded by **§7 (P13 R-line sweep)** — current
> deploy pick is **R1r_3150**, with the robustness-bundle verdict.

- **Primary deploy: K4 `model_3420`** (`p9_replant/2026-08-22_17-26-54_K4_smooth024`),
  the checkpoint already exported as `policy_*` per the K4b launch header.
- **Backup: K4b2 `model_3460`** (seed-45 twin — near-identical scores, so the
  recipe, not the seed, is what works). **Fallback line: K3s `model_3360`**
  (last clean iter — NOT 3380+, see forensics below).
- **The training-curve jerk scare was a measurement artifact.** TB's
  `diagnostics/action_jerk_abs_mean` is computed on noisy sampled actions;
  the K line's exploration σ (~0.37–0.43) fully explains 0.7+, and the
  deterministic policies are as smooth as P8's (see §2). **No extra
  smoothness/snap reward shaping is warranted** — and the K3/K3s2 divergence
  history argues actively against raising those penalties.
- **P8 policies collapse on the replant plant** (≤9 % survival where K scores
  99+ %) — the strongest direct evidence yet that the P7 replant closed the
  right gap.

## 1. Candidates and their training-side health (TB forensics)

| ckpt | vloss | raw_max | reward | σ (noise std) | note |
|---|---|---|---|---|---|
| K4 3420 | 5.2k | 8.5 | 12.9k | 0.394 | clean; "most lithe" in viewer; exported |
| K4b 3500 | 4.6k | 3.7 | 15.0k | ~0.37 | cleanest post-patch K4b |
| K4b 3560 | 16k | 4.4 | 14.8k | 0.368 | post-transient, vloss still settling |
| K4b2 3460 | 7.6k | 6.5 | 13.8k | ~0.37 | cleanest K4b2 |
| K4b2 3520 | 12k | 44.6 | 15.9k | 0.377 | raw_max spike at snapshot |
| K3s 3360 | 5.6k | 20.8 | 12.8k | 0.423 | last clean K3s iter |
| K3s 3380 | **128k** | **119.8** | 10.2k | 0.423 | inside the collapse — do not deploy |
| K3s 2820 | 6.6k | 24.8 | ~11.5k | 0.428 | K4 branch point (context) |
| G2i 4760 | — | — | — | 0.107 | last hardware flight (P8) |
| F 3490 | — | — | — | 0.148 | P8-recipe reference, flew earlier |

Two corrections this forensics produced: **K3s's collapse starts at 3380**,
not ~3390 (vloss 128k → 297k by 3400; the actor still evals fine at 3380 —
critic damage leads actor damage — but there is no reason to deploy from
inside a collapse). And **K4b hit a vloss transient at 3520 (640k) with the
lr-ceiling patch active, then recovered within ~30 iters with no reward
collapse** — first direct evidence the patch converts the fatal K3/K4
divergence mode into a recoverable blip (reward stayed 14–15.8k throughout).

## 2. The action-jerk question, resolved

TB's `diagnostics/action_jerk_abs_mean` is the 2nd difference of **sampled**
actions. White noise of std σ contributes ≈ `1.95·σ` to it (Δ²a of iid noise
has std `σ·√6`, mean-abs `= σ·√6·√(2/π)`). Reading σ straight out of the
checkpoints:

| line | σ (mean over joints) | predicted noise floor | observed TB jerk |
|---|---|---|---|
| P8 G2i-4760 | 0.107 | 0.21 | ~0.2–0.3 |
| P8 F-3490 | 0.148 | 0.29 | "< 0.3" |
| K4b-3560 | 0.368 | 0.72 | 0.73 |
| K3s-3380 | 0.423 | 0.82 | 0.81 |

Prediction matches observation to ~2 decimal places on both lines: **the
training curve measures exploration noise, not policy smoothness.** The K
line simply runs ~3.5× P8's σ (same entropy_coef 0.005 — the heavier
replant plant sustains a wider optimum σ, and σ was still annealing at kill
time: 0.394 → 0.366 over K4b's run).

The deploy-relevant number is the **deterministic** policy's jerk, measured
below: K checkpoints land at 0.19–0.23 per joint on the hw-matched plant —
*equal to or below* the P8 references (0.18) and below the 0.3 target the
P8-era curve was tuned to. Watching the TB jerk curve is still useful
**within** the K line (same σ ⇒ comparable), but only relative to K's own
~0.72 noise floor, and never across lines with different σ.

**Consequence:** do not add smoothness/snap reward weight to chase the TB
number. The mean policy is already smooth, and the K-line history (K3 ×2
penalties → 2/3 seeds diverged; K4's ×1.5 was chosen as the safe step) says
raising these penalties is the most divergence-correlated knob in the recipe.

## 3. Eval sweep results (deterministic policy, 1024 envs, seed 42, 15 s cap)

Scenarios: `hw_matched` = runbook §0.2 replica gate (single plant, central
measured values, mass ×1.06, DR pinned, clean sensors); `hw_harsh_arm13` =
same + armature ×1.3 (the corner the 08-20 tether-bias calc says is closest
to the real hip); `balance_full_dr` = training-range DR. K rows evaluated
with their launch overrides (servo 20/1.4 + 30/1.5, sat fence); P8 rows at
cfg-default gains (= their trained 8/0.5) on the same replant plant.

| checkpoint | hw surv15s | hw jerk6 | hw kneeclip | hw flight | hw drift | arm13 surv | arm13 5s | dr surv | dr 5s | dr jerk6 |
|------------|------------|----------|-------------|-----------|----------|------------|----------|---------|-------|----------|
| F_3490     | 0.087      | 1.10     | 0.05        | 0.38      | 0.45     | 0.102      | 0.296    | —       | —     | —        |
| G2i_4760   | 0.085      | 1.08     | 0.06        | 0.38      | 0.46     | 0.077      | 0.274    | 0.151   | 0.151 | 1.31     |
| K3s_2820   | 0.998      | 1.30     | 0.21        | 0.22      | 0.15     | 0.990      | 0.995    | —       | —     | —        |
| K3s_3360   | 0.996      | 1.16     | 0.22        | 0.22      | 0.08     | 0.993      | 0.994    | 0.890   | 0.960 | 1.70     |
| K3s_3380   | 0.998      | 1.20     | 0.21        | 0.22      | 0.10     | 0.998      | 0.998    | —       | —     | —        |
| K4_3420    | 0.998      | 1.14     | 0.26        | 0.22      | 0.11     | 0.997      | 0.997    | 0.906   | 0.959 | 1.55     |
| K4b2_3460  | 0.996      | 1.21     | 0.24        | 0.21      | 0.06     | 0.996      | 0.998    | 0.903   | 0.958 | 1.70     |
| K4b2_3520  | 0.997      | 1.21     | 0.24        | 0.22      | 0.06     | 0.996      | 0.996    | —       | —     | —        |
| K4b_3500   | 0.997      | 1.25     | 0.24        | 0.22      | 0.19     | 0.997      | 0.997    | 0.890   | 0.958 | 1.76     |
| K4b_3560   | 0.999      | 1.35     | 0.24        | 0.21      | 0.16     | 0.993      | 0.993    | 0.997*  | —     | —        |
| K5_4000    | 0.998      | 1.16     | 0.25        | 0.22      | 0.05     | 0.998      | 0.998    | 0.906   | 0.963 | 1.68     |
| K5b_4100   | 0.999      | 1.15     | 0.23        | 0.22      | 0.08     | 0.997      | 0.997    | 0.896   | 0.973 | 1.68     |

`surv` = fraction of 1024 envs alive at the 15 s cap; `jerk6` = mean |Δ²a|
summed over 6 joints (÷6 ≈ per-joint, comparable to the TB metric's units);
`kneeclip` = fraction of steps the deterministic knee action exceeds |1|;
`drift` = mean base drift (m) at episode end. (*not run for that ckpt.)

### Reading

- **Runbook §0.2 gate: every K candidate passes overwhelmingly** — ≥ 99.6 %
  survive 15 s hw-matched (gate asks ≥ 6 s majority), ≥ 99.0 % at the
  armature ×1.3 harsh corner. The few falls are early-launch (K4_3420's two
  deaths at ~0.67 s), not slow degradation.
- **P8 on the replant plant reproduces the hardware failure**: 91 % of
  G2i-4760 trials fall (p10 0.64 s, median 2.3 s), and it saturates hip_roll
  16–21 % of steps fighting the heavier plant (K: ~3 %). The 08-13 result
  "hw-matched sim survives 6 s" used the *pre-replant* plant; on the
  measured plant the same policy dies. Plant gap confirmed from the other
  direction.
- **Smoothness parity**: K jerk6 1.14–1.35 vs P8 1.08–1.10 — per-joint
  0.19–0.23 vs 0.18, with K carrying 2.5–3.75× stiffer servo gains. No
  regression.
- **K5/K5b polish (0.36→0.26 σ) yields NO measurable gate improvement**:
  hw-matched survival identical to K4b_3600 (99.8–99.9%, all statistically
  co-best), jerk slightly better (1.15–1.16 vs 1.27), drift slightly better
  (0.05–0.08 vs 0.13 m), knee clip unchanged (23–25%). The 28% noise
  reduction translates to slightly tighter action statistics but does not
  change pass/fail outcomes on any gate.
- **The K candidates are statistically interchangeable on these gates** —
  selection therefore falls to training-side health and secondary metrics.

### Watch items (not blockers)

1. **Deterministic knee clip 21–26 %** on all K checkpoints (P8: ~5 %). The
   knee rides its action-scale limit a quarter of the time. Sim and hardware
   share the mapping, so behavior will match sim — but it means the policy
   wants more knee authority than `real_action_scale_knee = 1.025` grants.
   If P12 retrains, consider widening knee scale; for P11, just note flight
   telemetry should log clip fraction.
2. **K4b/K4b2 are still training.** If a later checkpoint is preferred at
   flight time, re-run this sweep on it first (one command, ~8 min per
   checkpoint) and re-check §1-style vloss/raw_max cleanliness at that iter.
3. `active_frac` at K4b 3500+ is 0.193–0.199, a hair under the runbook's
   0.20–0.27 band (K4_3420: 0.202, in-band).

## 4. Selection

**Update (08-23 evening):** K4b `model_3600` was evaluated after this doc's
first cut (TB at 3600: vloss 4.8k, rawmax 3.8, af 0.193 — one of the run's
cleanest snapshots) and **promoted to primary on the user's gait judgment**:

| | hw surv15s | hw jerk6 | arm13 surv | dr surv |
|---|---|---|---|---|
| K4b_3600 | **1.000** (0 falls / 1024) | 1.27 | 0.991 | 0.882 |
| K4_3420 | 0.998 | 1.14 | **0.997** | **0.906** |

Statistically co-best: 3600 is perfect on the central (hw-matched) plant;
3420 is a few falls-per-thousand stronger at the armature ×1.3 corner and
under full DR. Both pass every gate with wide margin.

**Update (08-24):** K5/K5b polish checkpoints evaluated. The 0.36→0.26 σ
reduction achieved via the final-polish recipe (entropy 0.003 + desired_kl
0.002, 400-500 iters) produced slightly better secondary metrics (drift
0.05–0.08 vs 0.13 m, jerk 1.15–1.16 vs 1.27) but **no change in survival**
on any scenario. All K4b/K5/K5b checkpoints pass the hw-matched gate at
99.8–100%. The question "is 0.26 σ meaningfully better than 0.36 σ" is
answered: not on these gates. The 0.26 vs hypothetical 0.15 question (your
original query) is now moot — **go to hardware**. Real telemetry from P11
bench phase 2 will be more informative than further simulation polish.

| rank | checkpoint | why |
|---|---|---|
| **1** | **K4b_3600** | User-judged best gait; perfect hw-matched survival; cleanest-snapshot TB health; patched-scheduler lineage. Needs export via `control/export_policy.py`. |
| **2** | K4_3420 | Strongest at the harsh corner + full DR; training-clean; already exported. |
| **3** | K5_4000 or K5b_4100 | Polish-recipe checkpoints; slightly better drift/jerk but survival identical to K4b_3600. If gait aesthetics matter, pick one and export. Otherwise redundant. |
| **4** | K4b2_3460 | Seed-45 twin, best drift (0.06 m) — flying it would additionally validate recipe reproducibility on hardware. |
| **5** | K3s_3360 | Fallback line; last clean K3s iter. Only if the K4 lineage is disqualified on the bench. |

Do **not** deploy K3s_3380/3400+ (critic collapse in progress) or K4b_3560
(vloss still settling post-transient).

**Training wind-down (08-23 evening):** K4b and K4b2 were killed after both
plateaued (K4b flat ~16.0k since 3560, K4b2 ~15.4k since 3500, σ stuck
0.36–0.38 — PPO equilibrium reached; final checkpoints banked at 3680/3690).
K4b2's reproducibility purpose was complete (270+ healthy iters past branch,
through the 3390–3430 hazard zone). GPUs 0/1 relaunched with **K5_polish /
K5b_polish_seed45**: final-polish recipe (entropy 0.003 + desired_kl 0.002,
the P8 G2-lineage endgame) branched from K4b model_3600, seed-only twins.

**Final verdict (08-24):** The K5/K5b polish experiment confirmed what the
memory warned: **diminishing returns below σ ≈ 0.3 for this plant**. The
0.36→0.26 anneal (400-570 iters, entropy/kl at the adaptive-LR floor)
tightened secondary metrics but changed no gate outcomes. K4b_3600 flies
as-is. Further polish is not warranted — the replant work closed the plant
gap; 0.26 vs 0.15 is gait preference, not sim2real. **Hardware telemetry
from P11 bench phase 2 is the next information bottleneck**, not more
simulation training.

**K5/K5b polish results (08-24 morning):** Both runs annealed successfully
overnight. K5 ran 400 iters (3600→4000), K5b ran 570 iters (3600→4170).
Noise std trajectories:

| checkpoint | σ (mean) | Δ from parent |
|---|---|---|
| K4b_3600 (parent) | 0.3617 | — |
| K5_4000 (+400) | 0.2568 | −29% |
| K5b_4100 (+500) | 0.2610 | −28% |

K5 annealed smoothly (0.36 → 0.30 → 0.27 → 0.26 → 0.257); K5b plateaued at
0.26 after iter 4100. Both achieved substantial tightening (0.36 → 0.26 range).
Training stopped 08-24 morning — further annealing at current hyperparams would
be glacially slow (~0.005/100 iters), and dropping entropy/kl tighter than
0.003/0.002 would pin adaptive LR deeper at its 1e-5 floor with no evidence of
benefit.

**Eval sweep completed 08-24 08:12** with K5_4000 and K5b_4100 (full scenarios:
hw_matched, hw_harsh_arm13, balance_full_dr). Results below.

### Polish vs parent comparison

Direct comparison of K4b_3600 (parent, σ=0.36) against K5/K5b polish checkpoints
(σ=0.26):

| checkpoint | σ | hw surv | hw drift | hw jerk6 | arm13 surv | dr surv | dr 5s |
|---|---|---|---|---|---|---|---|
| K4b_3600 | 0.362 | **1.000** | 0.134 | 1.27 | 0.991 | 0.882 | 0.919 |
| K5_4000 | 0.257 | 0.998 | **0.049** | **1.16** | **0.998** | **0.906** | **0.963** |
| K5b_4100 | 0.261 | 0.999 | 0.079 | **1.15** | 0.997 | 0.896 | **0.973** |

The polish improved drift (0.13 → 0.05–0.08 m), jerk (1.27 → 1.15–1.16), and
full-DR survival slightly (+1.4–2.4 %), but hw-matched survival remains
99.8–100% for all three — statistically tied, all passing every gate.

## 5. Method / artifacts / how to re-run

- [scripts/rsl_rl/eval.py](../RL/scripts/rsl_rl/eval.py) gained
  `--cfg_overrides` (re-pass the training arm's launch overrides — without
  them a K checkpoint evals on the wrong plant, cfg-default kp 8),
  `--out_json`, and the `hw_matched` / `hw_harsh_arm13` scenarios.
  **One scenario per process** — Isaac Sim hangs on a second env build in
  the same process (cost us 2 h; now documented in the eval.py header).
- Driver: [RL/run_eval_p9sel_GPU2.ps1](../RL/run_eval_p9sel_GPU2.ps1)
  (`CUDA_VISIBLE_DEVICES=2`; skips any (ckpt, scenario) whose JSON exists —
  delete a JSON to re-run; add a manifest entry + re-run to eval a new
  checkpoint). Full 28-pair sweep took ~61 min.
- Results: `RL/logs/eval_p9sel/<ckpt>__<scenario>.json` (+ per-run logs).
  Table: `python RL/scripts/rsl_rl/aggregate_eval_p9sel.py`.
- Noise-floor math and σ extraction: §2 above; σ read from each
  checkpoint's `model_state_dict['std']`.

## 6. Next steps (runbook desk phase)

1. §0.1 recorded: K4_3420 primary / K4b2_3460 backup (runbook updated).
2. §0.2 executed in eval form — PASS for all candidates (this doc). The
   original `_launch_match_sim` phase-0/default-pose launch variant can
   still be run on K4_3420 for extra margin if desired.
3. Export K4b2_3460 via `control/export_policy.py` as the backup artifact
   (K4_3420 already exported); then §0.3 contract updates + obs-parity
   against whichever artifact flies, §0.4 (no retrim), §0.5 CAN_TIMEOUT.

## 7. P13 R-line sweep — 2026-08-27

Eval of the P13 robustness arms (R1_robust, R3_r1plus, and their 08-27
tightened-PPO resumes) against the flown K4b_3600 baseline, for the
quick hardware test of 08-27. Run on GPU 2 via
[RL/run_eval_p13r_GPU2.ps1](../RL/run_eval_p13r_GPU2.ps1), results in
`RL/logs/eval_p13r/`.

### Protocol changes vs the §3 sweep — do not mix the tables

1. **Pickup-free everywhere** (`pickup_initial_probability=0`,
   `pickup_probability=0`) — the p11 flight post-mortem showed pickup-on
   contaminates flight/z/pg and even survival. All §7 numbers are
   pickup-free truth; K4b_3600 was re-run under the identical protocol,
   which is why its row here differs from §3/§4 (hw jerk6 1.53 vs 1.27,
   kneeclip 0.15 vs 0.24 — idle without pickup rides is quieter).
2. **R rows re-pass `hip_pitch_armature=0.0385`** (their trained,
   bench-corrected value — the L-line eval trap). K4b stays on its
   trained cfg-default 0.0332. `hw_harsh_arm13` is ×1.3 over each row's
   own central value.
3. **Two new scenarios** (added to eval.py 08-27):
   - `hw_push_recovery` — hw_matched plant + trained-range pushes
     (10–40 N, 10 steps, p=0.002/step ≈ 3 pushes per 15 s). The
     pre-registered P13 gate from the R1 launch header.
   - `r1_train_dr` — R1's full widened training conditions (R1 plant-DR
     box, action delay 1–3, spawn-posture DR 0.18/0.12 rad, exec-noise
     injection 0.03/0.015 rad, pushes @ p=0.001). Same box for every
     row including K4b: exec noise and wide spawns are OOD for K4b —
     that asymmetry is the point; it measures what the bundle bought.

### Candidates and training-side health

σ read from each checkpoint (`print_sigma_p13r.py`); TB window-means ±30
iters (`RL/logs/tb_p13r_health.py`). The R line runs σ ≈ 0.51–0.52 —
1.4× K4b's 0.36, sustained by the much heavier DR; TB jerk/clip numbers
are correspondingly noise-inflated and were ignored per §2.

| ckpt | σ (mean) | TB reward @iter | vloss | raw_max | note |
|---|---|---|---|---|---|
| R1_3110 | 0.517 | 2605 | 32k | 21 | R1 peak; run declines after (tail: reward 1425, raw_max 61) |
| R3_3680 | 0.524 | 1792 | 11k | 14 | R3 peak; same decline pattern |
| R1r_3150 | 0.510 | 2537 | 9.2k | 19 | tightened resume (kl 0.008, clip 0.12, smooth ×3), cleanest snapshot |
| R3r_3760 | 0.509 | 1864 | 33k | 12 | tightened resume of R3 |
| K4b_3600 | 0.362 | — | 4.8k | 3.8 | flown baseline (§4) |

(TB reward is not comparable across R/K lines — different DR difficulty
and penalty scales.)

### Results (deterministic policy, 1024 envs, seed 42, 15 s cap, pickup-free)

| checkpoint | hw surv15s | hw jerk6 | hw kneeclip | hw drift | arm13 surv | dr surv | dr 5s | push surv | r1dr surv | r1dr 5s |
|------------|------------|----------|-------------|----------|------------|---------|-------|-----------|-----------|---------|
| K4b_3600   | 0.997      | 1.53     | 0.15        | 0.16     | 0.987      | 0.883   | 0.952 | 0.726     | 0.293     | 0.527   |
| R1_3110    | 0.996      | 1.90     | 0.04        | 0.10     | 0.988      | 0.895   | 0.938 | 0.862     | 0.600     | 0.769   |
| R1r_3150   | 0.996      | 1.84     | 0.04        | 0.16     | 0.987      | 0.897   | 0.938 | 0.859     | 0.590     | 0.764   |
| R3_3680    | 0.990      | 1.83     | 0.04        | 0.09     | 0.990      | 0.844   | 0.939 | 0.809     | 0.514     | 0.739   |
| R3r_3760   | 0.994      | 1.80     | 0.05        | 0.13     | 0.989      | 0.875   | 0.953 | 0.828     | 0.572     | 0.774   |

### Reading

- **The robustness bundle worked.** On the central plant everything is
  a statistical tie (hw 0.990–0.997, arm13 0.987–0.990), but under
  disturbances the R line separates hard: push recovery 0.86 vs
  K4b's 0.73, and under R1's full training conditions 0.59–0.60 vs
  **0.29** — 2× survival where exec noise, imperfect entry postures,
  and wide plant DR stack. This is exactly the failure class P12 blamed
  for the hardware falls (execution-on-real-plant, bin C/belt).
- **Gate bar** (R1 launch header: pickup-free hw_matched ≥ K4b's
  0.9971): R1_3110/R1r_3150 land at 0.9961 — 4 vs 3 falls per 1024,
  within noise; treated as PASS. Push-recovery ≥ K4b: PASS with margin.
- **The knee no longer rides its limit**: deterministic knee clip
  dropped from 15–26 % (K line) to 4–5 % — the §3 watch-item resolved
  by recipe rather than by widening `real_action_scale_knee`.
- **Cost of robustness: jerk.** R rows run hw jerk6 1.8–1.9 vs K4b's
  1.53 (per-joint 0.30–0.32 vs 0.26). The tightened resumes (smoothness
  ×3) recover part of it (R1: 1.90 → 1.84; under DR 2.85 → 2.69) at
  zero survival cost — the resume direction is confirmed but had only
  ~40–80 iters at eval time.
- **R3's cheap-dim widening bought nothing measurable** — equal or
  slightly worse on every scenario than R1 despite the extra DR
  coverage. Consistent with its dims being measured-cheap in training
  (flat dr_impact splits); their eval-time absence changes nothing.

### Selection (08-27 hardware test)

| rank | checkpoint | why |
|---|---|---|
| **1** | **R1r_3150** (`2026-08-27_10-30-37_R1_resume3110/model_3150.pt`) | Ties R1_3110 on every survival gate, smoother (jerk 1.84/2.69), cleanest TB snapshot (vloss 9.2k, raw_max 19). Export via `control/export_policy.py`; knee v_des FF = True (K contract). |
| 2 | R1_3110 | The un-resumed peak; marginally higher push survival (0.862). |
| 3 | K4b_3600 | Conservative fallback, already exported — but it is the policy that already fell on hardware, and it is 2× worse under the disturbance box. |

Do **not** fly the R3 line (dominated), R1 tail checkpoints past ~3110
(reward decline, raw_max 61 at tail), or R3r past 3760 unverified.

**Incident note:** GPU 1 (4090 D) dropped off the bus at ~15:59 during
the sweep ("GPU is lost", reboot required) — the R3_resume3680 training
died with it (last checkpoint model_3900 banked; process hung). GPU 0's
R1_resume3110 training was unaffected. The sweep completed on the
4060 Ti by selecting it via GPU UUID (see driver header). Separately,
the driver's first pass silently ran only 9/25 pairs due to PowerShell's
case-insensitive variables (`$scen` loop variable overwrote the `$SCEN`
array) — fixed in the driver; worth knowing for future sweep scripts.
