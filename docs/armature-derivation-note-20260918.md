# Armature derivation — vendor rotor inertia (2026-09-18, revised the same day)

**Status.** `belt_knee_actuator.ROTOR_INERTIA_OUTPUT` is the **vendor** RS01
rotor inertia reflected to the output shaft, **4.2e-3 kg m²** (17 Nm peak,
7.75:1 internal reduction, ~380 g). The bench-fit figures that stood here
earlier today and since P7 — hip 13.9e-3 / knee 19.4e-3 / P12 recenter
19.3e-3 — are kept in comments only as historical *apparent*-inertia fits
(bench total minus URDF links). A 19.3e-3 rotor would need ~75 % of the motor
mass spinning at the housing radius. The bench **signal** is not disputed; its
attribution to the rotor is, and Step 2 below re-attributes it.

Mechanics are unchanged from the morning's two commits (7ee51cc derivation +
per-leg DR draw, 538325c mask own-share): `derive_armature()` is the one
source, pins ≥ 0 win, `[ARMATURE]` prints at build.

| quantity              | formula            | value now (vendor) | value this morning (19.3e-3) |
|-----------------------|--------------------|--------------------|------------------------------|
| knee                  | I_out / G²         | **7.810e-3**       | 35.888e-3                    |
| hip_pitch             | own + c² · knee    | **8.396e-3**       | 38.582e-3                    |
| off-diagonal          | c · knee           | **5.725e-3**       | 26.306e-3                    |
| series rotor (motor)  | G² · knee = I_out  | **4.200e-3**       | 19.300e-3                    |
| hip own share (mask)  | own / hip_pitch    | 0.500              | 0.500                        |

These are the **original P6 plant** (`rl_env_cfg_p6.py` header: armature
7e-4 / 4.2e-3 / 7.8e-3). Tests: `data/_test_armature_derivation.py` 63/63,
`data/_test_elasticity_knobs.py` pass (`python`).

## What moves without a pin — every line, none bit-exact

| recipe                                   | hip                      | knee            | belt rotor (motor) | off-diag        |
|------------------------------------------|--------------------------|-----------------|--------------------|-----------------|
| P9/K default (no hip override)           | 33.2e-3 → 8.40e-3 (−75 %) | 36.0 → 7.81e-3 | 19.36 → 4.20e-3    | 26.4 → 5.72e-3  |
| L/R/S/S7 (`env.hip_pitch_armature=0.0385`)| 38.5e-3 (pinned, unchanged) | 36.0 → 7.81e-3 | 19.36 → 4.20e-3 | 26.4 → 5.72e-3 |

**No K/L/R/S/S7 checkpoint was trained on this plant.** A resume onto the
derived defaults is a plant shift, not a refactor.

## Pins that put a recipe back on its trained plant

```
env.hip_pitch_armature=0.0385      # L/R/S/S5/S6/S7 (K/P9 rows: 0.0332)
env.knee_armature=0.036
env.belt_series_rotor_inertia=0.019360   # only if the belt band is live (S5/S7)
env.dr_armature_per_leg=false
```

- **S7_fullDR** (training now on GPU 1): do not resume it on this code at all —
  the mask commit changed the hip column under a live belt from 0 to own/total
  and no flag reproduces the zeroed hip. Finish S7 on the checkout it started on.
- Rigid-belt recipes (S1*–S4*, R*, L*, K_servo2030, W_nocoupling,
  launcher_probe2): hip pin + `knee_armature=0.036` + `dr_armature_per_leg=false`.
- Eval / sim2real / sweep launchers (`run_eval_p12gate_GPU2.ps1`,
  `run_eval_p13r_GPU2.ps1`, `run_p12_2_simrepro_GPU0.ps1`,
  `run_s4_elasticity_sweep_GPU2.ps1`, `run_s4_hwtwin_GPU2.ps1`,
  `run_s4_sysid_GPU2.ps1`, `run_s6_eval_sweep_GPU1.ps1`, `run_foot_demo.ps1`,
  `run_belt_ab_ladder.ps1`): they pass the hip pin only (K rows none). Add
  `knee_armature=0.036` (+ `hip_pitch_armature=0.0332` for K rows) and
  `dr_armature_per_leg=false` whenever the eval must see the trained plant.
- Factory callers (`data/sysid_replay_policy_log.py`, `data/fast_2.py`):
  bare `make_*_actuators()` now returns the vendor plant; pass
  `hip_pitch_armature=`, `knee_armature=` to pin.

**New line:** `RL/launch_train_S8_vendorJ.ps1` = S7's recipe with the hip pin
removed (GPU 0; plant-shift resume of S4a `model_7780`; header pre-registers the
`[ARMATURE]`/`[PLANT]` lines to expect; `dr_armature_mult 0.80–1.20` is a
placeholder until Step 2 returns).

## Hard-coded literal audit (Step 1 grep)

Pattern `19.3e-3|0.0193|19.36e-3|0.01936|0.0385|0.0332|0.036|36.0e-3|33.2e-3|13.9e-3|19.4e-3`
over `RL/scripts/rsl_rl/*.py RL/*.ps1 data/*.py control/*.py`:

| hit | verdict | action |
|-----|---------|--------|
| `sweep_plant_elasticity.py:151`, `sweep_rest_absorption.py:123`, `data/_belt_cosim_stability.py:25` (`ROTOR_INERTIA_OUTPUT = 19.3e-3`) | derived consumer | **changed to 4.2e-3** (test-checked against the actuator) |
| `data/_test_armature_derivation.py`, `data/_test_elasticity_knobs.py` (assert 19.3e-3) | test literal | **updated** to 4.2e-3 / 8.40 / 7.81e-3 |
| every L/R/S launch `env.hip_pitch_armature=0.0385` (L1, L2, R1, R1_resume3110, R2, R3, R3_resume3680, S1–S1d, S2–S2d, S3a/b, S4a/b, S5, S6, S7, S7_test), `run_belt_ab_ladder`, `run_eval_p12gate/p13r`, `run_foot_demo`, `run_p12_2_simrepro`, `run_s4_*`, `run_s6_eval_sweep`, `launcher_probe2` | pinned (trained plant) | **left as-is on purpose** — they reproduce the plant those checkpoints trained on; a resume on this code additionally needs `knee_armature=0.036` |
| `launcher_probe2.ps1:114`, `run_s4_elasticity_sweep_GPU2.ps1:123` (`belt_series_rotor_inertia=0.01936`) | pinned | left (trained-plant reproduction) |
| `launch_train_S7_fullDR.ps1:113` (`[PLANT] … I_rotor=0.01936` expected-print line) | historical | leave; S8's header states `0.0042` |
| `K_servo2030`, `W_nocoupling`, `L1_calmidle`, `S4a/S4b`, `S5` header comments (38.5e-3, 33.2/36.0e-3, 19.36e-3) | historical | leave |
| `data/_belt_rotor_noise_sim.py`, `_p7_offdiag_phantom.py`, `_s7_belt_{at_500hz,boundary,config_score,growth}.py` (`I_R = G2 * 36.0e-3`) | historical one-offs, concluded | leave |
| `data/_test_belt_series.py` (19.36e-3 / 0.036 as explicit fixtures) | test inputs, explicit | leave |
| `data/_tether_bias_bound.py:155` (13.9e-3) | historical | leave |
| `control/*.py` | **no hit** — only `coast_down_bench.py:145` help text mentions rotor inertia | deploy side clean |

## Step 2 — re-fit the bench with the rotor pinned (analysis, not a plant change)

Tool: `data/coastdown_fit.py --pinned-inertia` (added to the tool that already
reads the ballistic `bal_*.npz` format — `sysid_fit.py`'s loader reads the
PD-sysid `trial_names` format and cannot open these files, so the mode lives
here and borrows sysid_fit's DE + L-BFGS-B recipe for the one nonlinear fit).
Rotor pinned at 4.2e-3; link inertia pinned from the URDF desk calc
(`data/_p7_armature_from_urdf.py`: hip bench pose 17.09e-3, shin+foot 2.50e-3
about the knee, joint side; knee reflected ×G²). One metric for every row:
R² of the SG motor-side acceleration on the burst samples that `fit_burst`
uses, so the unpinned PRIMARY fit is the reference row.

| hypothesis | free | discriminator (pre-registered) |
|------------|------|--------------------------------|
| H-friction | C, b, gravity | R² ≥ 0.98 and C, b ≥ 0. Expected CONTRADICTED (friction changes decay, not τ/α). |
| H-tscale   | one τ scale s, C, b, gravity | R² ≥ 0.98 **and** fitted gravity slope within 0.8–1.2 of the URDF 0.913 Nm/rad (a fit returns g/s, independent of any inertia assumption) **and** one s for hip and knee (spread ≤ 15 %). Reconciles the 08-17 "17 Nm delivered" probe: that probe was self-calibrated on α ∝ τ linearity and cannot see a constant scale; the gravity slope can. |
| H-delay    | delay T (0–40 ms grid), C, b, gravity; brake-phase closed loop | R² ≥ 0.98 at a best T ≤ 5 ms (measured CAN2SPI dead time). Ballistic bursts are open loop, so only the brake phase carries delay information. |
| H-belt (knee only) | k_j, d_m, C, b, gravity; two-inertia sim, encoder on the rotor | R² ≥ 0.98 and k_j in the S7 band 100–1000 N m/rad. Physics bound stated up front: with no friction a series compliance moves the motor-side apparent inertia only between I_r and I_r + G²·I_link (5.5e-3); friction at its bound is what lifts the naive slope above that. |

Run (≈5–10 min; the two knee files each run a 40-generation DE):

```
cd <repo>
python data/coastdown_fit.py --pinned-inertia data/bal_101_braced.npz data/bal_101_braced_early.npz data/bal_103_hang.npz data/bal_103_hang_hold.npz data/bal_102_hipclamped.npz data/bal_102_belt_on.npz > outputs/pinned_fit_20260918.txt 2>&1
findstr /V /R "^ *rep " outputs\pinned_fit_20260918.txt
```

Paste the table (everything from the `====` line on). The hip rows also carry a
**kp-sweep check**: the pinned hip plant rings at sqrt(20/21.3e-3)/2π =
**4.88 Hz** under the step_dep kp-20 battery; hardware measured 3.06–3.34 Hz
(runbook 1.1). That row is decided on paper, before any run.

Smoke results (tool check only, `--de-iters 2` on the knee — NOT the verdict):

| trial | hypothesis | R² | fitted | verdict |
|-------|------------|----|--------|---------|
| bal_101_braced | unpinned (ref) | 0.952 | I_app 34.9e-3 = 1.64× pinned; g1 −1.086 | — |
| bal_101_braced | H-friction | 0.895 | C 0.13, b 0.21 | contradicted |
| bal_101_braced | H-tscale | 0.952 | s 0.61; \|g1\|/URDF = 0.73 (gravity says s ≈ 1.19) | contradicted |
| bal_101_braced | H-delay | 0.474 | best T 5 ms (R² 0.18 at T = 0) | contradicted |
| bal_102_hipclamped | unpinned (ref) | 0.986 | I_app 22.5e-3 = 4.06× pinned | — |
| bal_102_hipclamped | H-friction | 0.625 | C 0.25, b 0.18 | contradicted |
| bal_102_hipclamped | H-tscale | 0.986 | s 0.246 (hip needs 0.61 → one scalar fails) | per-file supported, cross-trial contradicted |
| bal_102_hipclamped | H-delay | 0.961 | best T 3 ms (R² 0.89 at T = 0) | contradicted (below bar) |
| bal_102_hipclamped | H-belt (2 gens) | 0.915 | k_j 2000, ζ 0.26, b at bound | not a result at this budget |

If the full set returns all four contradicted, say so; the candidates that are
NOT pre-registered and would need the user's go are an under-massed URDF
shin/foot (`urdf_onshape_export/ballfeet-remass-urdf/` exists, untracked) and
gearbox compliance on the direct-drive hip.

## Step 2 RESULT (user ran the full set, 2026-09-18 late; raw: outputs/pinned_fit_20260918.txt)

All four pre-registered hypotheses CONTRADICTED on every trial. `bal_103_*` are
motor 103 = hip_roll (EL05), not hip-pitch hang trials, and were skipped (the
hip hanging trials are P7's bal2/bal3, not in this set).

| trial | unpinned I_app (x pinned) | H-friction | H-tscale | H-delay | H-belt |
|---|---|---|---|---|---|
| bal_101_braced | 34.9e-3 (1.64x), R2 0.952 | R2 0.895 X | s 0.61, gravity ratio 0.73 X | R2 0.47 X | n/a |
| bal_101_braced_early | 41.0e-3 (1.93x), R2 0.898 | R2 0.842 X | s 0.52, gravity ratio 0.73 X | R2 0.45 X | n/a |
| bal_102_hipclamped | 22.5e-3 (4.06x), R2 0.986 | R2 0.625 X | s 0.246 (per-file ok) | T 3 ms, R2 0.961 X | k 1954, R2 0.915 X |
| bal_102_belt_on | 20.7e-3 (3.73x), R2 0.986 | R2 0.634 X | s 0.268 (per-file ok) | T 3 ms, R2 0.961 X | k 20, R2 0.927 X |
| cross-trial | | | s spread 89 % -> X | | |

Discriminating signals: (1) the excess is tau/alpha-like and of similar size
at both RS01 output shafts (hip 14-20e-3, knee 17e-3 motor side) and
load-independent (hip carries 17e-3 of link, knee 1.3e-3) -> it reads like
motor-side inertia, which is how the bench got 19e-3; (2) the hip gravity
anchor puts the torque scale at 0.84-1.2 (unpinned |g1|/URDF 1.19 and 1.40;
P7 hanging 1.08), not the 0.5-0.6 the pinned inertia needs, and the knee
would need 0.25 — no single scalar; (3) the belt model is unidentified on
burst data (k lands on opposite bounds with the same R2, friction at its
bound does the work), consistent with the 08-22 kp-sweep seeing no second
knee mode; (4) delay improves the brake-phase fit (0.89 -> 0.96 at 3 ms) but
cannot touch the open-loop burst where the excess is measured.
Not pre-registered, awaiting the user's go: per-motor commanded-torque scale
(firmware Kt/limit_cur registers, GUI read), URDF link under-mass
(`ballfeet-remass-urdf/`, desk re-run of `_p7_armature_from_urdf.py`; can
reach the hip's 1.6-1.9x, not the knee's 4x).

## Step 3 — consequences before any training

**Belt co-sim stability** (`data/_belt_cosim_stability.py`, now with
`--rate --nsub --rotor`; defaults = the script's original 500 Hz/2-substep model):

| physics | rotor | ceiling (rho > 1 first at) | S7 band 100–1000 |
|---------|-------|----------------------------|------------------|
| 500 Hz / 2 sub | 19.3e-3 | ~750 (ζ 0.3), ~1000 (d 0) | top third unstable (memory physics-rate-500hz-belt-limit) |
| 500 Hz / 2 sub | 4.2e-3  | ~750 (ζ 0.3), ~1000 (d 0) | top third unstable — same |
| **2000 Hz / 4 sub (live plant)** | 19.3e-3 | > 2000 (5000 unstable at ζ 0.3, I_j 2.5e-3) | all-stable |
| **2000 Hz / 4 sub (live plant)** | **4.2e-3** | **none through 5000** (rho ≤ 0.988 every cell, every I_j) | **all-stable; no cap needed** |

The lighter rotor makes the co-simulation more stable, not less: the ceiling
moved up. Keep 2000 Hz; the 500 Hz cap of ~600 still applies if the rate ever drops.

**DR band (`dr_armature_mult`).** Not decidable until the Step 2 table is in.
Rule, pre-registered: if H-belt wins, the knee excess is already in the belt DR
and the armature band stays narrow (0.8–1.2 around 4.2e-3). If H-tscale wins,
the excess belongs in `dr_effort_mult` / stiffness, not armature — band stays
narrow. If all four are contradicted, 0.8–1.2 around 4.2e-3 (3.4–5.0e-3) does
not cover a bench that reads 19–21e-3 at the rotor, and the honest options are
(a) run S8 as the "vendor plant" arm with the narrow band and let the hardware
kp-sweep prediction (4.88 vs 3.1–3.3 Hz) be the arbiter, or (b) a wide
log-uniform band 1.0–5.0 spanning vendor-to-bench with the coherent per-leg
draw. Recommendation: (a) first — (b) is a knob whose input channel we cannot
yet name. The S8 script carries 0.8–1.2 marked PLACEHOLDER.

**Checkpoint compatibility.** Stated plainly: S7 must not resume onto this
plant without pins. The pin set that restores S7's plant is
`env.hip_pitch_armature=0.0385 env.knee_armature=0.036
env.belt_series_rotor_inertia=0.019360 env.dr_armature_per_leg=false` (and S7
itself should not resume on this code at all, see above); the P7/K pair is
`env.hip_pitch_armature=0.0332 env.knee_armature=0.036`.

**Deploy side.** Verified by grep over `control/*.py`: nothing consumes rotor
inertia or armature (`config.py`, `motor_bridge.py`, `policy.py` clean; the only
hit is a help string in `coast_down_bench.py`). No deploy change.

**Servo gains.** The P10 design kp 30 / kd 1.5 (knee J≈38.5e-3 → 4.4 Hz, ζ 0.7)
reads as ~8.6 Hz, ζ≈1.3 with knee J≈10.3e-3. Not retuned (single-change rule);
flagged in the S8 header.

**Replica gate.** The runbook §0.2 gate is `eval.py --scenarios hw_matched`
(`RL/run_eval_p12gate_GPU2.ps1` is the template). Prepared:
`RL/run_eval_vendorJ_gate_GPU2.ps1` runs S4a `model_7780` twice on hw_matched —
row A on its trained plant (pins) and row B on the vendor plant — and reports
survival for both. That measures whether the *policy* tolerates the plant
shift; it does not say which plant is real. The small-signal claim in runbook
1.1 ("sim and hw agree at small signal") rested on J≈42e-3 giving 3.5 Hz at
kp 20; on the vendor plant the same arithmetic gives 4.88 Hz, so that
agreement is **contradicted on paper** for the vendor plant. No gate result is
claimed until you paste one back.

## Mechanics (unchanged from the morning note)

1. `derive_armature(rotor_inertia_output, hip_own_rotor=NaN, hip_pitch_pin=-1,
   knee_pin=-1, coupling=0.733, gear_ratio=22/30)`: knee = I_out/G², hip = own +
   c²·knee (own NaN = same RS01 = I_out), off-diagonal c·knee, series rotor
   G²·knee. Pins ≥ 0 win exactly; a derived hip uses the RESOLVED knee.
2. `dr_armature_per_leg=True` (default): one multiplier per leg for the
   co-axial pair; off-diagonal and belt rotor follow. `False` = legacy
   independent draws, same RNG stream.
3. Mask under a live belt pair: hip column keeps own/total (0.50 derived, 0.42
   under P7 pins); `belt_mask_armature=False` disables the mask entirely.

Scratch files from this session's patching, safe to delete:
`data/_patch_vendorJ_step1.py`, `data/_patch_vendorJ_step2.py`,
`data/_pinned_block.py`.
