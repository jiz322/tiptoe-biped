# S5 elastic-DR campaign — belt + ball-pad domain randomization

**Run:** `RL/logs/rsl_rl/p9_replant/2026-09-14_15-55-27_S5_elasticDR`
**Arm:** `RL/launch_train_S5_elasticDR.ps1` (GPU 0, RTX 4090 by UUID)
**Resumed from:** `2026-09-09_22-13-40_S4a_measDR/model_7780.pt`
**Status: KILLED at iteration 7787**, 21 minutes in, by its own pre-registered kill
condition (value function diverging). Cause identified and quantified below.

---

## 1. What was asked

Add four domain-randomization channels on top of the S4a measured-DR recipe, and
report back on anything that cannot be added:

| Channel | Requested band | Units | Outcome |
|---|---|---|---|
| Belt stiffness | 100 – 1000 | knee-joint N m/rad | **live, engaged correctly** |
| Belt damping | 0 – 0.3 | motor-side N m s/rad | **live, engaged correctly** |
| Ball-foot stiffness | 10000 – 200000 | floor-normal N/m | **live, but mis-scaled — diverged** |
| Ball-foot damping | 0 – 0.3 | N s/m | **live, but effectively inert** |
| Ball-foot restitution | ~0.632 (ideally 0.4 – 0.7) | coefficient | **cannot be randomized per env** |

Plus: "I need TB record dr impact. If no impact that means dr does not work."

---

## 2. The restitution question

**What value is restitution at now? Zero. It was never wired to 0.632.**

`foot_restitution` defaults to `0.0` in [rl_env_cfg_p6.py:838](../RL/source/RL/RL/tasks/direct/standup_curriculum/rl_env_cfg_p6.py#L838).
Neither `launch_train_S4a_measDR.ps1` nor `launch_train_S5_elasticDR.ps1` passes
an override for it, so every run on this line — S3, S4a, S4b, and S5 — has trained
with a perfectly rigid, perfectly inelastic ball-foot contact. The `[PLANT]` startup
line confirms it:

```
[PLANT] foot contact: k=0 N/m d=0 N s/m (ratio=nan at m_eff=1 kg) restitution=0 friction=1/1 -> RIGID legacy material
```

The `0.632` that appears in the S5 launch header is **a comment, not an override.**
That is a documentation error on my part and it is exactly the failure mode you
warned about — a value that reads as set but is in fact constant at its default.
The header said "pinned at the measured 0.632"; the code sends nothing. I have
corrected the header to say the truth (see §7).

### Is restitution effectively covered by pad kp/kd?

**No. They are different physics, and the pad as configured cannot substitute for it.**

A restitution coefficient is an instantaneous velocity-reversal law: the outgoing
normal speed is `-e * incoming`. It carries no state. The pad is a stateful
spring-damper: it stores energy in `k*x` and returns it over a contact that lasts
tens of milliseconds, minus whatever `c` removes.

In principle a spring-damper *does* produce an effective restitution. For an
impulsive contact the map is

```
e_eff = exp( -pi * zeta / sqrt(1 - zeta^2) )
```

which is near 1 for light damping and drops toward 0 as zeta approaches 1. So if
the pad were the only compliance in the loop, its damping band would *be* a
restitution band — an indirect one, with the value set by zeta rather than by the
coefficient, and with the collision no longer instantaneous (contact time goes from
sub-millisecond to 20–45 ms, which changes the whole trajectory, not just the bounce).

But the pad is in **series with an 80 mm travel limit and a rigid backstop**, and
the numbers in §4 show the requested band does not put the pad anywhere near the
regime where `e_eff` is a usable restitution knob. Two further reasons it cannot
stand in:

- **The pad acts only where the ball is the contact.** The S4a stance keeps the
  balls lightly loaded at best; the pad's contribution to a hip or knee touchdown
  is zero.
- **The relaxation integrator dominates the response.** As computed in §4, over
  the whole requested damping band the pad's state advances at the `tau = m/c`
  floor, not at a stiffness-dependent rate. Whatever `e_eff` the pad has is set by
  the integrator's floor, not by the band — so the band cannot express a restitution.

An impulse implemented as a velocity-dependent force was rejected during design:
it double-counts the pad's own damping and would fire on the pad's relaxation
transient rather than on the true contact, which is a different plant wearing the
name of restitution.

**Consequence for this campaign: the foot has no bounce at all.** If the 0.632 is
a real measurement of the ball-foot pad against your floor, then the current plant
is missing real physics entirely — the ball hit is fully absorbed in sim and
partially reflected on hardware. That is a plant gap, not a DR gap, and it is
arguably a bigger deal than any of the four channels requested.

---

## 3. What the TB instrumentation shows (the good news)

The dr-impact instrumentation works, and it answered the engagement question
decisively within seven log points. Every channel below was drawn on every reset
and recorded:

```
dr/belt_k_joint    min=100      mean=543.6   max=1000    std=259.5    ← full band, engaged
dr/belt_c_motor    min=1.7e-06  mean=0.1499  max=0.300   std=0.0864   ← full band, engaged
dr/pad_k           min=10005.8  mean=104851  max=199999  std=54831    ← full band, engaged
dr/pad_c           mean=0.150   std=0.0866                            ← full band, engaged
dr/pad_c_clamped_frac = 0                                             ← E4 sanity PASSES
```

**E0 (engagement) — the instrumented question, answered:**

```
foot/ball_touched_frac  0.132 0.092 0.119 0.112 0.132 0.119 0.124 0.112 0.126
foot/ball_contact_frac  0.130 0.091 0.118 0.111 0.131 0.118 0.123 0.111 0.125
```

This is the finding that costs the least and matters the most. **The balls do
engage — about 12% of foot-steps-and-envs, latched per episode.** That directly
contradicts the S4a reachability warning (CLAUDE.md: "the S4a rollout never loads
the balls anyway, `lowest toe height +0.0856 m` vs contact line `0.084`"), and it
means the pad band is **not** decorative the way the foot-material channels were.
The 09-12 lesson does not repeat here.

Caveat on the same numbers: `foot/ball_lowest_z` reads −29 to −30 m, i.e. the foot
spheres end up far below the floor plane. That is a symptom of the divergence in
§4, not a physics result, and it makes `foot/ball_margin_mean` unusable in this
run. The contact/touched fractions are the trustworthy readings; the margin
statistic needs a clean re-run.

**E1/E2 (impact contrasts) — did the channels drive outcomes?**

```
dr_impact/belt_k_joint_high_minus_low  +0.6 -30.8 -4.1 -7.2 -11.8 -11.6 -8.1 -12.0 -13.0  (steps)
dr_impact/belt_c_motor_high_minus_low  -11.4 +23.7 -4.2 +2.2 +0.1 -0.4 +4.1 +2.5 +0.5
dr_impact/pad_k_high_minus_low         +34.5 +0.8 +2.7 -3.1 +1.4 -2.4 +4.3 +2.2 -3.4
dr_impact/pad_c_high_minus_low         -20.1 -17.4 +3.7 +0.7 +0.4 -3.7 -4.1 -0.7 +1.1
```

Read against the pre-registered bar (|contrast| < 25 steps on a ~100-step episode
means no effect *at this band*), the honest verdict is **inconclusive**: the run
lasted seven iterations, the episode length is still crashing from the resume
transient (79 → 121 → 110 steps against a settled ~1700–1900 for S4a), and the
belt contrast is drifting monotonically (−7 → −13) rather than settling. There is
a weak, consistent *stiff-belt-is-better* leaning in `belt_k_joint` (−13 steps,
correlation −0.068), which is the direction the rigid-plant assumption has always
had — but a 7-point trend on a diverging run is not evidence, and I will not
present it as such.

**Verdict: the dr-impact channel is now proven to work and is readable. It needs a
run that survives past iteration 7800 to say anything about the belt or the pad.**

---

## 4. Why the run died, and what it says about the pad band

### The kill

```
Loss/value_function  4.9e7  5.5e10  inf  3.9e13  1.7e14  1.1e15  4.4e15  1.3e16  1.5e16
diagnostics/raw_action_max_abs   13.7  1247  3.9  4.0  4.1  4.0  4.0  3.9  4.2
```

The value function multiplied by ~300 every iteration, passed through `inf` at
7782, and was at 1.5e16 when I killed it. `raw_action_max_abs = 1247` means the
policy network emitted an action three orders of magnitude outside its clip range,
and the learning rate pinned at its 1e-3 ceiling in the same window. This is the
pre-registered kill condition met exactly (`value_function > 3x` its mean for
>100 iters with episode length falling).

For contrast, S4a at its endpoint held `value_function ≈ 4200` and `episode_length
≈ 1700` with `alpha = 1.0`. S5 was at `episode_length ≈ 110` and `alpha = 0.5`.
Nothing about the policy survived the first update.

### The mechanism

The pad's force is `F = k * x`, and its blowup guard only fires at
`max_compression = 4 * ball_radius = 80 mm`. The requested stiffness band, at
plausible penetrations, puts absurd forces into that gap:

| k (N/m) | x = 2 mm | x = 8 mm | x = 80 mm (guard limit) |
|---|---|---|---|
| 10 000 | 20 N (0.4× weight) | 80 N (1.5×) | 800 N (14.6×) |
| 200 000 | 400 N (7.3× weight) | **1600 N (29×)** | **16 000 N (293× weight)** |

16 000 N on a 5.57 kg body is 2870 m/s². Even the *nominal* 8 mm pad thickness at
the stiff end is 287 m/s², sustained. The pad was pushing the robot through the
floor — which is why `foot/ball_lowest_z` settled at −30 m. A single 2 ms step of
that force is a 0.6 m/s velocity jump on a shin of ~0.2 kg.

**This is a specification error, not an implementation error.** The `[PLANT]`
lines prove the mechanism did exactly what the band asked. The band was wrong: it
was chosen as an order-of-magnitude ladder around a measured material stiffness
without ever being checked against the load the foot actually carries.

### The damping band is a second, quieter specification error

`pad_damping_max = 0.3` N s/m. The pad's relaxation time is `tau = m/c`:

| k (N/m) | wn (rad/s) | contact half-period | zeta at c = 0.3 | tau = m/c |
|---|---|---|---|---|
| 5 000 | 70.7 | 44.4 ms | 0.0021 | 3.33 s |
| 10 000 | 100.0 | 31.4 ms | 0.0015 | 3.33 s |
| 30 000 | 173.2 | 18.1 ms | 0.0009 | 3.33 s |
| 200 000 | 447.2 | 7.0 ms | 0.0003 | 3.33 s |

At every point in the band the pad is essentially undamped (zeta ~ 1e-3), and
because `tau` is clamped at `4*dt = 8 ms`, the state advances at a **fixed 25% of
the gap per step regardless of the stiffness drawn**. Two consequences:

- The pad can never soften a capture. Reducing `k` does not make the contact
  gentler in time, because the 8 ms relaxation floor dominates.
- The band spans 3 orders of magnitude of zeta but the physics barely moves,
  which is why `pad_c_high_minus_low` reads ±4 steps.

The belt band, by contrast, is well-posed: zeta 0.44 at the soft end to 0.14 at
the stiff end, spanning well-damped to lightly-damped, both on the compliant side
of the servo's own 30 N m/rad knee loop. It engaged cleanly.

### Secondary finding: PhysX patch buffer overflow

S5 produced **5354** `Patch buffer overflow` errors, asking for
`at least 320 560 … 501 974` against the configured
`gpu_max_rigid_patch_count = 320 000`. S4a produced **zero**. Two candidate causes,
not distinguished by this run:

- the pad's extra per-step contact work adds patches, or
- the divergence itself generated piles of degenerate contacts.

FPS dropped 32 405 vs S4a's 44 215 (−27%), which is consistent with either. This
needs raising before the next run regardless; it is a one-line cfg change and a
plausible contributor to the divergence.

---

## 5. Recommendations

**Do not re-run this band.** Three changes are needed, in priority order.

**1. Re-derive the pad stiffness band from the foot's actual load, not from a
ladder.** Half-body-weight per foot is 27.3 N. A band that sags 0.9 mm at the
stiff end and 5.5 mm at the soft end, on a 12 mm pad, is:

```
pad_stiffness_min = 5000      # 5.5 mm sag at half-weight
pad_stiffness_max = 30000     # 0.9 mm sag
pad_thickness     = 0.012     # 2.2x the soft-end sag
```

The dynamic cross-check agrees: arresting a 1 kg effective foot mass from 0.3 m/s
over 5 ms needs ~30 N, which is `k = 30 000` at 1 mm. The requested 10000–200000
band is 3–20× too stiff at its lower end and 7× too stiff at its top.

**2. Fix the damping band or delete it.** A 0–0.3 N s/m band is 800× below the
integrator's 250 N s/m cap and three orders of magnitude below critical at every
point — it is a knob whose value cannot affect the plant. Either widen it to a
physically meaningful range (`0 – 50` N s/m gives zeta 0.14–0.5 at k = 10000) or
drop it and keep the ratio form (`pad_damping_ratio`), which is the parameter that
actually maps to a measurable quantity.

**3. Raise `gpu_max_rigid_patch_count`** from 320 000 to at least 600 000.

**4. On restitution:** it cannot be a per-env channel and should be dropped from
the per-env DR list. If the 0.632 is real, the honest experiment is a **per-process
sweep** — five short runs at `foot_restitution ∈ {0, 0.2, 0.4, 0.632, 0.8}` — with
the drift between runs read as the impact. That cannot isolate restitution in one
TB curve, which is the cost of the material being stage-global. It is worth doing
anyway, because a foot with no bounce at all is a plant gap independent of DR.

---

## 6. Difficulty prediction for the remaining work (per CLAUDE.md)

The directive asks this **before** building. Applying it to the three fixes above:

**Pad band re-derivation.** *Common or invented:* entirely common — sizing a
contact spring from the load it carries is the standard first step, and skipping
it is what went wrong here. *What others do:* derive `k` from a target sag under
the body weight, then check the contact's natural frequency against the timestep;
this is textbook. *Cheaper workaround:* none needed — it is arithmetic, already
done in §5, no code change. *Rough cost:* zero build, one line edited in the
launch script.

**Damping band.** *Common or invented:* common in form, but the ratio parameter
(`zeta`) is the idiomatic entry point and the explicit band is the unusual one.
*What others do:* express contact damping as a ratio against the participating
mass, which this module already supports. *Cheaper workaround:* use the existing
`pad_damping_ratio` and skip the band entirely. *Rough cost:* zero build.

**Restitution sweep.** *Common or invented:* common practice — a parameter sweep
across processes is the standard answer when a material is stage-global. *What
others do:* exactly this. *Cheaper workaround:* the S4 elasticity sweep harness
(`sweep_plant_elasticity.py`, `run_s4_elasticity_sweep_GPU2.ps1`) already does
policy × plant-cell sweeps with a `--contact-restitution` flag. *Rough cost:*
one launch-script variant reusing the existing sweep, ~1 day of GPU time.

**`foot_contact_effective_mass` — the knob you nominated.** It only scales the
ratio-form damping (`c = 2*zeta*sqrt(k*m_eff)`). With an explicit damping band in
use it is inert, so it does not need a DR. The knob that *does* deserve one, on
the evidence of §4, is `pad_thickness` — it sets the resting height and is a hard
travel constraint, so a per-episode draw on it is a draw on how much pad exists at
all. That is a real physical spread (a pad wears, compresses, or is replaced) and
it is currently pinned. I would add it once the band above survives a run.

---

## 7. Corrections applied

- `RL/launch_train_S5_elasticDR.ps1` header said `foot_restitution 0.632 (pinned)`.
  **This was false** — no override is passed and the cfg default is 0.0. Corrected
  to state the value is 0.0 and that the 0.632 is not applied.

---

## 8. Reusable tooling produced

- `data/_tb_read.py` — a dependency-free TFRecord/Event reader. The `env_isaaclab`
  interpreter cannot import numpy from a direct invocation and the separate analysis Python has
  no tensorboard, so no existing tool could read these event files. Usage:

  ```
  python data/_tb_read.py <events-file> [substring]
  ```

  Prints every scalar tag with its sample count, last step, and last value.
