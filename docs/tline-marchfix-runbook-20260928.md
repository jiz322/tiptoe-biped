# U-line march/shrug fix — runbook, 2026-09-28

Status: **EDITS APPLIED, NOT LAUNCHED.** Supersedes its own earlier T2/T3 draft of
today: the A/B is now **U1/U2, one variable (stance depth)**, with the §2 march
fix carried by *both* arms. Terrain is dropped and no DR band is widened (§4).
T0 (`T0_straight`) and T1 (`T1_lowcom`) were killed by the user on 2026-09-28 at
iterations 1040 and 1190.

Source of the diagnosis: the two T runs' tensorboard event files
(`RL/logs/rsl_rl/p9_replant/2026-09-26_22-11-33_T0_straight` and
`.../2026-09-26_22-12-16_T1_lowcom`), read with `data/_tb_read.py`; their
launched `params/env.yaml`; the full-recipe diff; frames from the two rollout
videos (`videos/sim2real/`); and the balance manifold
`RL/logs/balance_manifold_stand.csv`.

---

## 0. What this document changed from its earlier draft

| | earlier draft | now |
|---|---|---|
| arms | T2 `marchfix` vs T3 `marchfix_tilt` | **U1 vs U2** |
| A/B variable | fix-shape vs fix-shape + hostile ground | **stance depth** |
| terrain | §3 argued for/against gravel; tilt/friction widening on T3 | **dropped** (§4) |
| DR widening | `dr_gravity_tilt_max_deg` 2.0→5.0, friction 0.6/1.1→0.3/1.3 on T3 | **none, both arms unchanged** |
| the fix | carried by both T2 and T3 | carried by **both U1 and U2** |

The reason for the change: with the fix on both arms, **U1 vs the killed T1
isolates the fix alone**, and **U1 vs U2 isolates the stance alone**. The earlier
design could do neither, because every arm moved two things against T1.

---

## 1. Why — the T1 evidence, condensed

The load-bearing facts (full argument in the superseded draft):

1. **The gait-reference prior is inert.** `rew_scale_gait_ref=0.0` in both T
   arms' `env.yaml`, so `gait_ref_amplitude_rad=-0.12` pays nothing. "The
   gait-ref is too strong" cannot be the cause.
2. **T1 is not shuffling.** `idle_swing_above_contact_frac` 0.784 vs T0's 0.0186;
   `idle_foot_height_diff` 0.02857 m vs 0.00205; `idle_liftoff_rate` 9×. 78 % of
   idle swing-frames clear the contact line.
3. **Four foot-height terms are one-sided ramps with live gradient at T1's own
   step height and no cost above it.** At `idle_foot_height_diff` 0.0286 against
   `feet_air_height_threshold` 0.05, `rew_feet_air`/`rew_foot_swing` sit at 0.57
   of full marks; `rew_swing_clearance` at `(0.0809 − 0.071)/0.015 = 0.66`.
   **Not saturated** — the marginal gradient at 28.6 mm is still positive.
4. **The low stance amplified it.** Straight-knee T0 is near a singularity for
   shortening the leg; bent-knee T1 has real leg-length authority, so the same
   reward buys a 29 mm march instead of a 2 mm shuffle. The recipe did not
   change; the plant's response to it did.
5. **The roll shrug is underpriced by the term that names it.** Per-leg
   `|hip_roll|` 0.231 rad (13.2°) in T1 vs 0.120 (6.9°) in T0, but
   `rew_hip_roll_split` (−2.0 scale against `max_hip_roll_rad` 1.8) is the
   **weakest of the three objectors** (0.85 % of total) while `stance_symmetry`
   (−80) is the loudest at 9.05 %.
   ⚠️ **`max_hip_roll_rad` is a TERMINATION bound** (`rl_env_p6.py:3131`,
   `doing_side_splits`), not a shaping knob. Lowering it 1.8 → 0.5 would count a
   13° shrug as a fall. Never lower it for shaping.

---

## 2. The fix — one coherent change, carried by BOTH arms

### 2.1 Toe lines, re-derived (inside the preset, never as an override)

The p10 rule was never applied to `lowcom_b090`; T1 launched with the preset's
provisional values. Measured `diagnostics/idle_toe_z_p10` = **0.0624** against a
provisional `swing_clearance_zero_m` = 0.071, so the dense ramp was starting
8.6 mm above the resting foot. Apply, **inside each preset**:

| preset | `swing_clearance_zero_m` | `no_flight_contact_threshold` |
|---|---|---|
| `lowcom_b090` (U1) | 0.071 → **0.065** (p10 0.0624 + 0.003) | 0.076 → **0.070** (p10 + 0.008) |
| `lowcom_b060` (U2, new) | **0.079** — MEASURED, rest 0.076 + 0.003 | **0.084** — MEASURED, rest + 0.008 |

**Flagged side effect:** `no_flight_contact_threshold` is also the airborne
classifier. Lowering it reclassifies more steps as airborne, moving
`foot1/2_contact_frac` (T1: 0.616), `no_flight`, `clock_contact` and
`stance_symmetry` together. T1 already lost 37 % of its foot contact relative to
T0; this makes the line tighter. Re-read `foot*_contact_frac` at +200 rather than
assuming the preset absorbed it.

### 2.2 The height band — the actual structural fix (code edit)

Replace `rew_swing_clearance`'s one-sided ramp with a **trapezoid**: rise 0→1
over `[zero, zero + target]`, flat-topped at the peak, then fall 1→0 over
`[zero + target, zero + target + shoulder]`. One new cfg field
`swing_clearance_shoulder_m: float = 0.0` (default 0.0 = **bit-exact** the legacy
one-sided ramp, so no existing run changes meaning).

```python
swing_clearance = torch.clamp(
    (swing_foot_z - swing_clearance_zero) / swing_clearance_target, min=0.0, max=1.0)
if swing_clearance_shoulder > 0.0:          # 0.0 -> legacy ramp, bit-exact
    swing_clearance = swing_clearance * torch.clamp(
        (swing_clearance_zero + swing_clearance_target + swing_clearance_shoulder
         - swing_foot_z) / swing_clearance_shoulder, min=0.0, max=1.0)
```

Values for both arms: `swing_clearance_target_m = 0.015` (unchanged),
`swing_clearance_shoulder_m = 0.030`. Because `zero` is preset-owned, the band
self-calibrates to each stance:

| | `zero` | peak at `zero + 0.015` | charge reaches zero at `zero + 0.045` |
|---|---|---|---|
| U1 (`b090`) | 0.065 | **0.080** | 0.110 |
| U2 (`b060`) | 0.079 | **0.094** | 0.124 |

**Why the target stays short at 0.015 rather than widening to 0.040.** The
superseded draft's §2.1 said "widen the target to 0.040 to keep gradient alive".
That advice is **conditional on the overrides-only variant** — with no shoulder, a
0.015 target saturates at the robot's own mean toe height and pays full marks for
the average step. With the band, the margin gradient is *negative* instead of
zero, so that trap is gone and the target must stay short or the peak sits above
the marching. **§2.1's two options are mutually exclusive; this document takes
the band, so the target stays short. Do not apply both.**

⚠️ **The peak height (`zero + target`) is the most sensitive number in this
runbook.** At U1's numbers T1's marched mean toe (0.0809) sits essentially *on*
the peak, so the downward gradient at the bug height is small. Re-read
`idle_toe_z_p10` at +200 and, if the march survives, the first lever is a
**shorter** `swing_clearance_target_m` (peak below the bug), not a larger scale.

### 2.3 Roll shrug — raise the term that names it

`rew_scale_hip_roll_split` −2.0 → **−6.0**, alone. Do **not** simultaneously
rebalance `stance_symmetry`: the two ride the same `step_gate = 0.5` idle floor
and both measure a bilateral weight shift, so raising one while cutting the other
leaves them fighting over the same motion. `max_hip_roll_rad` stays 1.8.

### 2.4 Deliberately NOT changed

- **`feet_air_height_threshold` stays 0.05.** Raising it (the superseded draft's
  overrides-only lever) would slow the march without bounding it, and moving two
  height levers at once makes the verdict unreadable. It is the **fallback
  lever** if the band underperforms (§2.2's flag).
- **`rew_scale_gait_ref` stays 0.0.** Enabling it would add a second stepping
  pressure to a robot already being paid to overstep.
- **`swing_clearance_zero_m` in the launch scripts.** T1 passes
  `env.swing_clearance_zero_m=0.078`, which `apply_stance_preset` then
  **overwrites** with the preset value. That override is dead weight and a trap
  for the next reader; it is **removed** from the U launch files (§6).

---

## 3. The two arms

Both fresh (no `--resume`), both 32768 envs, 500 Hz / decimation 5, both carrying
every line of §2, both on the S7 armature pins. The **only** difference is the
stance preset.

| | **U1 `U1_marchfix`** | **U2 `U2_marchfix_b060`** |
|---|---|---|
| GPU | 0 (RTX 4090) | 1 (RTX 4090 D) |
| stance preset | `lowcom_b090` (T1's pose, `hip +0.50 / knee −0.90`) | **`lowcom_b060`** (new, `hip +0.30 / knee −0.60`) |
| preset edits vs T1 | toe lines 0.065 / 0.070 (§2.1) | new block, §3.2 |
| band (§2.2) | target 0.015, shoulder 0.030 | same |
| `rew_scale_hip_roll_split` | −6.0 | −6.0 |
| everything else | identical to T1's recipe | identical to U1 |

**What each contrast answers:**

- **U1 vs the killed T1** — does the §2 fix kill the march? (T1 is the baseline
  that produced 28.6 mm and 0.784.)
- **U1 vs U2** — does stance depth alone change it, with the fix held constant?
  This is the fork the earlier design could not settle.

### 3.1 The depth axis (why `hip +0.30 / knee −0.60`)

`root_h` is a **bad** depth metric: the preset's `root_z` is the FK spawn height,
which pins the nail point at `_FK_TARGET_FOOT_Z = 0.08561924` in *any* pose, so
it barely moves with depth. An analytic FK chain (validated to ~0.3 mm against
both measured presets — straight 0.4390 vs cfg 0.434 / deploy 0.4401; T1 0.4303
vs preset 0.430) gives, on the θ* ≈ 0 contour:

| knee | hip* | spawn_h | com_h | d(spawn_h)/d(knee) | vs T1 | CoM gain |
|---|---|---|---|---|---|---|
| −0.40 | 0.210 | 0.4417 | 0.352 | 0.0252 | 0.47× | −8.7 mm |
| −0.60 | **0.330** (cell 0.30) | **0.4388** | **0.350** | **0.0361** | **0.68×** | **−6.5 mm** |
| −0.75 | 0.410 | 0.4359 | — | 0.0455 | 0.81× | — |
| −0.90 | 0.498 | 0.4302 | 0.344 | 0.0536 | 1.00× | 0.0 mm |
| −1.20 | 0.661 | 0.4131 | 0.330 | 0.0708 | 1.33× | 13.3 mm |

Consequences to state plainly:

- **The whole T-line depth excursion buys ~1.4 cm of CoM.** Straight → T1 is
  0.352 → 0.344 m. A shallower stance therefore costs almost nothing in balance
  margin while giving back **one third of T1's knee-height authority** (0.68×),
  which is the axis that produced the march in the first place (§1.4). That is
  why depth, not height, is the U2 variable.
- **The straight-knee singularity is real and steep:** `d(spawn_h)/d(knee)` is
  0.0265 at knee −0.20 but 0.109 at −0.90. Going shallower moves *back* toward
  the singular end, so U2 is a bet that the march fix is the binding constraint
  and the stance merely amplifies it. §7's predictions are written to fail
  visibly if that bet is wrong.

### 3.2 The `lowcom_b060` preset block (exact)

`STANCE_PRESETS` in `rl_env_cfg_p6.py`, alongside `lowcom_b090`:

```python
    "lowcom_b060": {
        "root_z": 0.439,          # analytic FK spawn height 0.4388 (validated chain)
        "joint_pos": {
            "hip_roll1": 0.0, "hip_pitch1": 0.30, "knee1": -0.60,
            "hip_roll2": 0.0, "hip_pitch2": -0.30, "knee2": 0.60,
        },
        "height_min": 0.314,      # b090's offsets (-0.125 / +0.095) re-centred
        "height_max": 0.534,
        "swing_clearance_zero_m": 0.075,      # PROVISIONAL - see section 5
        "no_flight_contact_threshold": 0.080, # PROVISIONAL - see section 5
        "standup_path_knots": [ ... table below ... ],
    },
```

- **Torso pitch.** The exact grid cell (0.30, −0.60) reads θ* = **+1.74°** (mild
  forward lean). The contour value at this knee is hip **+0.330** (θ* ≈ 0). U2
  takes the measured cell (0.30, −0.60) for consistency with the T1 method (T1
  also took a grid cell). Switch to +0.330 if a strictly vertical torso is
  wanted; the difference is 1.7° and 0.03 rad.
- **Height band.** Offsets copied from `lowcom_b090` ([−0.125, +0.095]), which
  the T1 comment records as deliberately reaching below the zero-action PD rest.
  U2's rest will sit *higher* than T1's measured 0.344 m (less knee sag), so the
  lower edge has more room, not less.
- **Path knot table.** Straight rows up to s = 0.725, scaled by the same
  **1.2469** the T1 table used, then the stance row appended at s = 1.00. The
  straight row at 0.8625 (`0.312, −0.625`) maps to s = 1.075 > 1 and is dropped
  for the same reason it was dropped in T1 — **it is the U2 stance's own pose**,
  so nothing is lost. 13 rows, and **rows 0–11 are identical to T1's** (both are
  scaled straight rows), so §5's viewer smoke check inherits T1's already-verified
  K0–K9 rows.

```
0.0000, 3.090, -3.060, -90.0, 0.235, 3.090, -3.060,
0.0623, 2.869, -2.998, -78.3, 0.239, 2.878, -3.054,
0.1247, 2.648, -2.936, -63.8, 0.235, 2.623, -3.053,
0.1870, 2.427, -2.873, -52.7, 0.233, 2.391, -3.014,
0.2494, 2.206, -2.811, -42.8, 0.230, 2.160, -2.955,
0.3117, 1.984, -2.749, -33.4, 0.225, 1.933, -2.893,
0.3741, 1.763, -2.687, -24.2, 0.218, 1.707, -2.827,
0.4364, 1.542, -2.624, -14.9, 0.207, 1.482, -2.766,
0.4988, 1.321, -2.562,  -6.2, 0.197, 1.261, -2.695,
0.5611, 1.100, -2.500,   2.7, 0.184, 1.039, -2.630,
0.7325, 0.838, -1.875,   9.7, 0.305, 0.838, -1.875,
0.9040, 0.575, -1.250,   8.4, 0.375, 0.575, -1.250,
1.0000, 0.300, -0.600,   0.0, 0.439, 0.300, -0.600,
```

⚠️ The final segment (−1.25 → −0.60, 0.65 rad) is roughly twice T1's
(−1.25 → −0.90, 0.35 rad) and slightly coarser than the straight table's own
0.725 → 0.8625 step (0.625 rad). In family, but it is the one row that differs in
character; §5's `-Scan` acceptance check is what catches it if the pose does not
rest.

### 3.3 Deploy mirror

Add to `control/config.py`'s `_STANCE_PRESETS` (the table `STANCE_PRESET` selects
from):

```python
    "lowcom_b060": ([0.0,  0.30, -0.60, 0.0, -0.30,  0.60], 0.439),
```

`STANCE_PRESET` itself stays `"straight"` until a U checkpoint is chosen for
hardware. Do **not** flip it before §8.

---

## 4. Terrain and DR — dropped, not deferred-with-a-build

Per the user, **no terrain and no DR change** in this A/B. Both arms keep T1's
bands exactly: `dr_floor_friction_min/max` 0.6 / 1.1, `dr_gravity_tilt_max_deg`
2.0, `dr_stiffness_mult` 0.90–1.10, `dr_mass_mult` 1.00–1.14, delays unchanged.

The terrain analysis from the superseded draft is retained only as a standing
verdict, because it is the reason not to re-open it casually:

- Heightfield rough-terrain DR is standard in legged RL and Isaac Lab ships the
  machinery, but **this repo has none**: the floor is one stage-global
  `GroundPlaneCfg` plane, there is no `TerrainImporter` in the direct-RL path, and
  `scene.replicate_physics=True` at 32768 envs assumes homogeneous geometry. A
  per-env heightfield forces `replicate_physics=False`, far slower here.
- **The hidden cost is calibration, not generation:** `_FK_TARGET_FOOT_Z`,
  `_compute_fk_height`, the path spawn interpolation, all four toe lines,
  `height_min/max` and the 13-row knot table all assume a plane at z = 0.
- **The modelling point:** a heightfield models *shape*. Loose gravel's dominant
  effect is that contact *yields and slides* — compliance and tangential flow.
  Gravel-shaped geometry alone would not reproduce what gravel does to a ball
  foot.
- **Do not re-propose the pad/foot-compliance channel.** Abandoned by the user
  09-16; its reference plane sits below the rigid floor, so it is inert by
  construction (memory `pad-reference-plane-inverted-20260916`).

If ground hostility is wanted later, the cheapest live channel is
`dr_gravity_tilt_max_deg` (2.0 → 5.0) plus a wider `dr_floor_friction` — both
one-line overrides, neither a build.

---

## 5. Probes to run before launch

The preset's two toe lines and the held-pose settle are **measurements**, not
derivations. `RL/run_probe_stance_T1.ps1` has been retargeted from the busy
4060 Ti to **GPU 0 (RTX 4090)** — its `CUDA_VISIBLE_DEVICES` line and header now
say so.

1. **Pose dump** — the two provisional numbers and the knot table's rest columns:

   ```
   powershell -File RL\run_probe_stance_T1.ps1 -Mode pose -Extra 'env.stance_preset=lowcom_b060'
   ```

   Read from `RL/logs/stance_probe/pose_<stamp>.log`:
   - loaded nail-toe rest z → re-apply `zero = rest + 0.003`,
     `contact = rest + 0.008` **inside the preset**;
   - root height and torso pitch of the held pose → confirm `height_min/max`
     contains it;
   - knee sag under the hold → confirm the s = 1.00 row rests.

2. **Viewer smoke** — the knot-table acceptance check:

   ```
   powershell -File script\run_show_standup_path.ps1 -Smoke -Preset lowcom_b060 -Gpu 4090
   ```

   Rows K0–K9 must read identical to T1's run (same numbers). K10–K12 are
   expected to `FALL` for the same pre-existing legacy reason recorded in the
   09-26 runbook §3.4 (soft viewer gains, not a regression).

3. Re-read `diagnostics/idle_toe_z_p10` at **+200 iters** on both arms and
   re-apply the p10 + 0.003 rule — the calibration is a running process, not a
   one-time fix (`rl_env_cfg_p6.py:858` documents p10 drifting 0.052–0.072).

---

## 6. Launch recipe

Copy `RL/launch_train_T1_lowcom.ps1` twice → `launch_train_U1_marchfix.ps1`
(GPU 0) and `launch_train_U2_marchfix_b060.ps1` (GPU 1). Edits vs T1:

- `--run_name` and the GPU UUID line (per §3's table).
- **Remove** `'env.swing_clearance_zero_m=0.078'` — dead, the preset owns it
  (§2.4). Leaving it is the worst of both worlds: it reads as authoritative and
  does nothing.
- `'env.swing_clearance_target_m=0.015'` — unchanged, keep explicitly.
- **Add** `'env.swing_clearance_shoulder_m=0.030'`.
- **Add** `'env.rew_scale_hip_roll_split=-6.0'` (T1 ran the −2.0 cfg default, so
  this is a §2.3 change).
- U2 additionally: `'env.stance_preset=lowcom_b060'`.
- U1 keeps `'env.stance_preset=lowcom_b090'`.

Everything else byte-identical to T1's list. **Re-pass every override on any
resume** (CLAUDE.md; this is what burned the G runs).

**Phase A first:** `-PhaseA` (5 iters, 128 envs), then read the startup lines and
one `diagnostics/base_height` before Phase B.

Expected startup contract lines:

```
[STANCE] preset=lowcom_b090 root_z=0.430 hip_pitch1=+0.500 knee1=-0.900
    spawn_h=0.430 stand_target_h=0.430 height=[0.305, 0.525]
    swing_zero=0.065 contact_thr=0.070 path_rows=13 path_end=(+0.500, -0.900, h 0.430)
[STANCE] preset=lowcom_b060 root_z=0.439 hip_pitch1=+0.300 knee1=-0.600
    spawn_h=0.439 stand_target_h=0.439 height=[0.314, 0.534]
    swing_zero=0.075 contact_thr=0.080 path_rows=13 path_end=(+0.300, -0.600, h 0.439)
[PLANT] belt series: live=False k_joint=[0, 0] ... I_rotor=0.01936
[ARMATURE] knee = 36.000e-3 [pinned]  hip_pitch = 38.500e-3 [pinned]
```

Any of those reading the *straight* values means the preset did not apply — it
lands before the articulation is built, so a silently-ignored preset trains the
wrong pose. A `[derived]` tag or 7.8/8.4e-3 on the `[ARMATURE]` line means the S7
pins did not land.

---

## 7. Pre-registered predictions

Deterministic eval only at equal iterations (TB clip/jerk are σ-dominated —
memory `action-jerk-tb-metric-noise-dominated`).

**U1 vs the killed T1 (does the fix work?):**

- **P1** `idle_foot_height_diff` falls from T1's 28.6 mm toward **≤ 10 mm**.
- **P2** `idle_swing_above_contact_frac` falls from 0.784 toward **≤ 0.30**.
- **P3** `reward/hip_roll_split` below 0.03 (from 0.066) and
  `torque_asymmetry_hr` below 1.1 (from 1.32).
- **P4** `standup_success_high` **not below T1's 0.966** — the fix must not cost
  stand-up.
- **P5** `reward/swing_clearance` **lower** in U1 than T1 (T1's 0.8123 was bought
  by over-lifting; the band must charge it back).

**U2 vs U1 (does depth alone matter?):**

- **P6** U2's `idle_foot_height_diff` **≥ U1's** — going shallower moves toward
  the straight-knee singularity, so *if the march is fixed* U2 should be no
  better. U2 marching **less** than U1 would mean the stance, not the reward, was
  the binding constraint and §1.3 is wrong.
- **P7** U2's `standup_success_high` **≤ U1's** — the path to a shallow squat is
  longer and its final segment coarser (§3.2).
- **P8** U2's `actual_torque_kn*_mean` at idle **lower** than U1's (less static
  knee load), and `pos_drift_m_mean` **lower** (less leg-length authority to
  wander with).

**Verdict rule.** If `idle_foot_height_diff` stays above 20 mm in **both** U1 and
U2, the marching is **stance-driven rather than reward-driven** and the next lever
is the stance preset itself, not the reward. If it falls in U1 but not U2, the fix
works and the stance is merely an amplifier — and U2 was the wrong arm.

---

## 8. Kill / gate

- **Kill:** `Loss/value_function` > 42000 for 50 consecutive iterations; PhysX
  error flood; `Train/mean_episode_length` < 400 at +600. Both arms are fresh
  starts, so +600 not +300.
  **Note:** T1's `value_function` read 8836 against T0's 3316 at similar
  iterations — the low-CoM arms carry a noisier critic, so watch from +500.
- **Gate at +2000:** `episode_length ≥ 1400`; every DR span live;
  `diagnostics/base_height` within ±0.02 of the arm's own nominal (0.430 for U1,
  0.439 for U2 — **not** the other arm's). U2 drifting back up toward 0.44 = the
  height band is too soft; tighten `height_edge_softness` (50.0) before spending
  more iterations.
- **Compare** at +4000 / +6000, deterministic eval only.
- **Duration (decided 2026-09-28): same as T.**  Both arms run to the same
  schedule the T arms were on when killed — the user's call, so the U runs stay
  comparable to T0/T1 rather than to a new budget.  Practical consequence: ~98
  s/iter at 32768 envs, so +4000 is ~4.5 days per arm and +6000 ~6.8 days.  The
  T arms were killed at ~1100 on a by-eye judgement, so **the +2000 gate below is
  the first real decision point**, not a formality.

---

## 9. Out of scope / open

- **APPLIED 2026-09-28.** §2.2's band (`swing_clearance_shoulder_m`, default 0.0 =
  bit-exact legacy ramp), §2.1's `lowcom_b090` toe lines (0.065 / 0.070), §3.2's
  `lowcom_b060` preset, §3.3's deploy mirror, and both launch scripts are **in the
  tree and syntax-checked**. Verified: preset table parses with all three presets,
  `lowcom_b060`'s 13 rows have s ascending 0 -> 1 and its s = 1.00 row matches its
  `joint_pos`; the reward call site passes 96 positional args with the shoulder at
  index 92; the trapezoid was checked numerically (legacy 1.000 / zero gradient at
  T1's 0.0809 mean toe, band 0.970 with a negative gradient, identical below the
  peak). The two launch scripts diff ONLY by run name, GPU line and preset.
- **MEASURED 2026-09-28** (§5.1 probe, `RL/logs/stance_probe/pose_20260928_134418.log`,
  GPU 0): spawn root **0.439 m** (analytic FK predicted 0.4388 — the chain is
  confirmed), loaded nail-toe rest **0.076 m** → `swing_clearance_zero_m` **0.079** /
  `no_flight_contact_threshold` **0.084**, and the endpoint's balanced pitch **+1.7°**
  (the manifold cell (0.30, −0.60) reads θ* = +1.74°), now in the knot table's
  s = 1.00 row.  **Nothing in the preset awaits a measurement any more.**  The
  `*_rest` columns stay equal to `*_target` on the final row, the convention the
  straight and b090 tables use: the probe's zero-action hold settles to root
  0.404 m / knee −0.69 / pitch 7.7°, and that sag is the known frozen-grace
  artifact, **not** a row value (b090's hold sagged to 0.344 m / 17.6° the same way).
- **§5.2 viewer smoke RUN 2026-09-28 on `lowcom_b060`** (`-Smoke -Preset lowcom_b060
  -Gpu 4090`).  Verdict: **knot table accepted.**  Rows K0-K9 are **bit-identical** to
  the T1 run's, which is the strongest available check — it proves the shared leading
  table (the straight rows scaled by 1.2469) survived the new endpoint unchanged, so no
  new penetration was introduced.  The +0.1 s spawn-consistency column is clean
  everywhere (worst |Δz| 2.0 mm at K0 for train gains, and the s = 1.00 row spawns at
  root 0.439 m / pitch +1.2° against its measured 1.7° reference — within the viewer's
  own soft-gain slack).  K10-K12 and the ramp `FELL` in **both** tables under both gain
  sets, which is the pre-existing legacy behaviour recorded in the 09-26 runbook §3.4:
  the viewer's [kp 200 / kd 10] hold is far softer than training's joint-side servo, and
  those rows are the dynamic squat-to-stand segment the policy is meant to rescue
  (memory `standup-ramp-topple-finding`).  **No action taken, and no T1-vs-U2 regression.**
- **Hardware is untouched.** Per CLAUDE.md the user runs every motion and power
  command. `docs/lowcom-stance-campaign-runbook-20260926.md` §8 still stands
  (contract diff, HOLD bench, trim re-bracket) — and the trim re-bracket is
  **more** pressing for U2 than for T1: the +0.089 compensator was fitted at the
  straight stance, and U2's lever arm is *closer* to straight's than T1's was.
- **`next-training-directive` items EXCLUDED by decision, 2026-09-28.**
  `handover_prime_prob=0.0` and `dr_hip_angle_bias_max=0.0` do **not** ride this
  A/B. They are a separate axis, and carrying them would move U1/U2 off T1 by
  more than the march fix, making the fix verdict unreadable. The T arms' fresh
  start already discarded the S-line's coverage of items 1/2/6, so those items
  are now uncovered until a future arm takes them deliberately — recorded here
  so the gap is explicit rather than implied.
- **Knee thermal:** `rew_scale_thermal` is 0 and a stepped gait is a stall load.
  Decide from the bench temperature, not sim.
- **The belt push-test** (never run) remains the only thing that can re-open
  2000 Hz.
