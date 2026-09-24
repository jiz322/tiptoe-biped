# Ball-pad compliance as a per-environment DR channel

**Written 2026-09-12.** Implementation note and handover. Nothing here has been
run in Isaac Sim yet: the model passes 53 CPU checks, the geometry measurement
has never executed, and the first thing to do is the probe described in §7.

The pad is modelled as a **per-environment external force**, not as a contact
material. That choice is forced, and §2 is the argument for it.

---

## 1. The problem being solved

Pad (ball-foot) compliance had to become a domain-randomization channel for the
next training campaign. The material path already existed — `foot_contact.py`
writes `compliantContactStiffness` / `compliantContactDamping` / restitution /
friction onto a material prim bound to the two ball collision meshes — and it
cannot carry DR. Two independent reasons, both verified by reading this build:

**The material is stage-global.** `physxMaterial:compliantContact*` are
attributes of a material prim, and one prim is shared by every environment in
the stage. `ArticulationView` exposes exactly three per-shape tensor channels —
static friction, dynamic friction, restitution — and nothing for compliant
gains. There is no place to put a per-env stiffness. This is not a missing call;
it is the shape of the API.

**A mid-run material edit does not reach the solver.** PhysX parses the stage
when the physics scene is attached, which happens inside
`SimulationContext.reset(soft=False)` — at env construction, guarded by
`get_physics_sim_view() is None`. `env.reset()` only calls `_reset_idx`
(`direct_rl_env.py:292`) and never re-attaches. So a material attribute written
after the scene exists is read by nobody. The play hotkeys
(`PgUp`/`Home`/`Ins`/`F`) write the prim correctly and change nothing, which is
what the user observed: **a new value requires a new process**.

The earlier probe evidence for "PhysX does re-read an attribute edit across
`sim.reset()`" does not survive inspection. Its decisive number came from a drop
test whose own earlier section reports the limp robot face-planting with the
balls never loaded (`lowest base height 0.0826 m`), and its friction A/B measured
how far a toppling robot travelled (`-2.39 m` vs `-1.53 m`), not traction.
Neither is a measurement of the foot material.

---

## 2. What replaced it

`PadContact` — a second spring/damper in parallel with PhysX's rigid contact,
applied to each shin body by the environment step. This is the same shape as
`belt_knee_actuator`'s series path, and for the same reason: the belt is per-env
DR precisely because it is a plant force the env owns, not a USD property.

```
F = -(k * x + c * approach_limit)        along the floor normal
x  = pad deflection, driven by the ball's overlap into the pad surface
c  = 2 * zeta * sqrt(k * m_eff)          m_eff = 1.0 kg (one-support catch mass)
```

### Files

| file | role |
|---|---|
| `RL/source/RL/RL/tasks/direct/standup_curriculum/pad_contact.py` | the model, the damping-ratio helpers, `measure_ball_geometry` |
| `rl_env_cfg_p6.py` (§ pad block) | `pad_*` knobs, all inert by default |
| `rl_env_p6.py` | `_build_pad_contact`, `_apply_pad_contact`, DR draw in `_sample_dr`, `pad_contact` override key |
| `RL/scripts/rsl_rl/probe_pad_contact.py` + `RL/run_probe_pad_contact.ps1` | the gate (§7) |
| `data/_test_pad_contact.py` | 44 numeric checks on the model + the geometry fit |
| `data/_test_pad_probe_logic.py` | 16 checks on the probe's own maths and force plumbing |
| `data/_check_ball_fit.py` | 17 checks: the real geometry routine vs the real `robot.usd` (needs `usd-core`) |

### Wiring

- `_build_pad_contact()` runs in `RlEnv.__init__` when `pad_contact_enable` is
  set, measures the ball geometry off the live stage, and constructs the pad.
- `_apply_pad_contact()` runs **once per physics substep**, from `_apply_action`,
  immediately before the `scene.write_data_to_sim()` that carries it into
  `sim.step`. The pre-physics hook zeroes the buffer only; computing there would
  integrate the damping at 100 Hz and ring at the stiff end of the band.
- The force goes through the same external-force path as pushes and pickups, so
  it is a plant force and is not re-clipped by the DC envelope.
- `_sample_dr` draws `(num_envs, 2)` — column 0 left foot, column 1 right, the
  same convention as `_dr_belt_stiffness` — and resets the pad state per
  episode. A per-env override under the `pad_contact` key wins over the band,
  matching the belt branch's precedence and for the same documented reason.

### Inert by default

`pad_contact_enable = False` and `pad_stiffness_max = 0.0`. With those,
`compute()` returns `None` before touching state, the buffers stay zero, and the
legacy plant is unchanged — so adding this cannot move a training line that
never sets the knobs.

---

## 3. The model, and the five rules that were measured rather than reasoned

Every rule below exists because the version without it produced a wrong number
that looked plausible. The measurements are in `data/_test_pad_contact.py`.

**The reference surface is `ball_radius - pad_thickness`.** With the pad's plane
at the bare ball radius, the pad can only push while the ball is *inside the
floor*, so its equilibrium force is zero and the settled sag read 6x too small.
The ball must rest on the pad, not on the floor it is already touching.

**Damping acts on the approach speed only.** A pad resists being compressed and
releases freely. Damping the absolute speed turned the pad into glue that hauled
the foot back down as the spring relaxed: a measured **500 N pull** during
separation.

**Damping is not added twice.** The state update already determines how fast the
deflection relaxes, so adding `c * v` to the force as well doubles the damping
and halves the settled sag — measured as **245 µm of the analytic 490 µm** at
k = 20000, m = 1 kg, zeta = 1, exactly the factor of two a doubled term predicts.

**The sign.** The force is `+z` (the floor pushes up). The first version
negated it at the wrong layer and drove the robot into the floor; the fix is one
`torch.where`.

**Damping above `2*m_eff/dt` diverges.** It is capped at
`pad_damping_stability_factor * m_eff / dt` (default 0.5, i.e. a quarter of the
hard limit) and the clip is **reported**, never silently applied. A DR band that
reaches the cap is a band that has quietly stopped meaning what it says.

### The travel constraint

The pad's relaxed thickness must exceed the sag it will carry (`m*g/k`), or the
ball bottoms out on its own rigid contact and the settled sag reads as the
thickness instead of the stiffness. This is a real way to build an inert knob
and it is why `pad_thickness` is a cfg field. At the built-in `pad_thickness`
of 2 mm the hard floor is `k > 4900 N/m` for a 1 kg catch.

### Calibration

The pad's sag under a per-foot load is the number you can measure on the bench
in a minute, and everything follows from it:

| load | k = 5,000 N/m | k = 20,000 N/m | k = 50,000 N/m |
|---|---|---|---|
| 1.0 kg | 1.96 mm | 0.49 mm | 0.20 mm |
| 2.8 kg (half body) | 5.49 mm | 1.37 mm | 0.55 mm |

The existing `FOOT_STIFFNESS_LADDER` (200 … 20000) spans 27 cm of sag at the
bottom rung — that end is not a foot, it is a hole in the floor. Expect the
useful band to be roughly 5e3 … 5e4.

---

## 4. Geometry facts worth not re-deriving

- The **ball is `part_3`**, and it is a **40 mm sphere on a boss, not a clean
  sphere** (measured from the authored STL, 2026-09-13). A least-squares sphere
  of R = 20.003 mm carries 2201 of 2721 distinct vertices; the other 520 reach
  out to r = 30.6 mm as a collar around the mesh's local ±z. Each foot carries
  its OWN copy as a separate collision: on the left it hangs off the
  `shin_inner` link, on the right off `nail___fillet1`.
- **Consequence, and it bit us:** the boss makes the AABB a 41.6 mm cube whose
  extents differ by only 3.9 %, so a cube-ness test passes it happily — and
  both the bbox centre and the bbox radius are then wrong. On the left foot the
  bbox centre is **1.03 mm** off the true ball centre (5 % of the radius) and
  the bbox radius reads **20.81 mm** instead of 20.003. On the right foot the
  same part, differently oriented, is off by only 0.03 mm — i.e. the error is
  orientation-dependent and cannot be corrected by a constant. Hence
  `fit_sphere`: fit the mesh, never measure the box.
- Ball centre, **left foot**, in the `shin_inner` frame:
  `(-0.197057, 0.029197, -0.018350)`.
  **Right foot**, in the `nail___fillet1` frame: `(0.001547, 0.199261, 0.018350)`.
  Both fitted, both matching the URDF-composed values to **0.5 µm**.
- The env's `_toe*_local_offset` is the **NAIL origin**, and both feet measure
  **59.09 mm** from it to the ball centre — which is the "5.8 cm" documented on
  2026-09-08. It must never be used as the contact point.
- **`nail___fillet1` is not a child of `shin_inner`.** It is a separate link
  under the `knee2` joint (parent `stand_steel_00___fillet1`, origin
  `(-0.154185, 0.107117, 0.0602)`, rpy `(0, 0, 0.662781)`). Composing the
  shin→nail chain by hand is the easy way to get the right foot's contact point
  wrong by ~9 cm; `measure_ball_geometry` avoids the question entirely by
  walking each body's own collision prims.
- **The collision prims live inside an instanced subtree.** Every link's
  `collisions` and `visuals` scope in `robot.usd` is `IsInstance() == True`, and
  **`Usd.PrimRange` does not descend into an instanced subtree**. A plain walk
  finds *zero* collision prims on a perfectly healthy stage. Both
  `measure_ball_geometry` and `_list_collision_prims` now pass
  `Usd.TraverseInstanceProxies()`; without it the probe raised "No spherical
  ball collision prim found" and printed an empty list, which reads like a
  missing asset and is not. The meshes also arrive as **triangle soup** (5438
  triangles → 16314 points, every corner repeated), so the fit dedupes first —
  otherwise local triangle density weights the fit.
- `measure_ball_geometry` reads all of this at runtime rather than hardcoding
  it, so an asset change moves the contact point with it. It raises rather than
  guessing when the ball cannot be identified, when the fit is not sphere-like
  (fewer than half the distinct vertices within 0.4 mm), or when the radius is
  implausible.

---

## 5. What is unverified

**The geometry measurement no longer runs blind.** It failed its first launch
(2026-09-13, `geometry_only`) because it walked instanced subtrees without
`TraverseInstanceProxies`. That is fixed, and the fix is now verified against
the real `robot.usd` by `data/_check_ball_fit.py` — 17 checks, centres matching
the URDF to 0.5 µm, run on the analysis interpreter with `usd-core` and no
Isaac Sim. So the *stage reading* is no longer the open question. What is still
unverified in the live env is only that Isaac Lab has not de-instanced or
otherwise rewritten these subtrees during spawn — `foot_contact.py` DOES call
`make_uninstanceable` on the foot collision subtree for the material path, so a
live env could present either shape. The traversal handles both, and
`geometry_only` is still the gate that proves it.

**Everything else in-sim.** The probe has been checked for its maths and its
force plumbing (16 checks) but has not launched past step A. Expect that its
stand-pose PD gains, its spawn height, and its `env.cfg.decimation` reference
need one round of correction against the live env.

**No interaction with the material path has been measured.** The pad and the
`foot_contact.py` material compose (rigid contact stays as the backstop), but
`foot_restitution`, `foot_contact_stiffness`, and the pad have never run
together. If both are set, the sum is emergent and not a designed plant.

---

## 6. Design rules this work followed

These are the lessons that keep reappearing in this repo; the pad work was
structured around them and they are worth keeping.

1. **Never trust a channel because it reports a plausible number.** The pad's
   sign error produced a smooth, finite, entirely wrong rollout. The static
   equilibrium test (hold the deflection at `m*g/k`, read the force, compare to
   `m*g`) is what caught it, and it is why `_test_pad_contact.py` tests
   equilibrium directly instead of only testing a closed-loop drop, which
   measures the harness.
2. **A knob that moves nothing must say so.** Every inert case in the model
   returns `None` and leaves state untouched, and every clamp prints. The
   material path's failure mode was exactly this: a parameter that was authored,
   bound, readable, and connected to nothing.
3. **Prefer the per-episode force over the stage property.** If a quantity has
   to vary per environment, it cannot live in USD. The belt series path already
   proved this shape works.
4. **Refuse an impossible value rather than capping it silently.** The
   validator bounds only what PhysX cannot express; the damping cap is applied
   and reported, not hidden.

---

## 7. Carry on from here

**Step 1 — the gate, in this order.** Run from `<repo>\RL`.
The launcher sets the Isaac environment itself, so no preamble is needed — but
follow the `PATH` de-duplication rule in
`docs/s4-checkpoint-testing-20260911.md` §0 if the shell has sourced other
launch scripts, because a duplicated `PATH` shows up as a confusing import
failure rather than a clean error.

```powershell
# 1. does the geometry measurement work?
powershell -File run_probe_pad_contact.ps1 -Mode geometry_only

# 2. does a pinned stiffness move the plant? (force APPLIED to the sim)
powershell -File run_probe_pad_contact.ps1 -Mode sag_ab

# 3. does the env's own DR path reach the model?
powershell -File run_probe_pad_contact.ps1 -Mode env_sweep -PadEnable
```

Logs land in `RL/logs/pad_contact_probe/pad_<stamp>.log`. The model's own checks
need no Isaac Sim and run on the analysis interpreter:

```powershell
python data/_test_pad_contact.py        # 44 checks
python data/_test_pad_probe_logic.py    # 16 checks
python data/_test_pad_knobs.py          # 54 checks (§9)
```

The geometry measurement itself is checkable with no Isaac Sim at all, because
it only needs USD — `data/_check_ball_fit.py` runs the real routine against the
real `robot.usd` and confirms the fitted centres against the URDF to 0.5 µm
(17 checks). It needs `usd-core` on the analysis interpreter:

```powershell
python -m pip install usd-core          # once
python data/_check_ball_fit.py          # 17 checks
```

Once `geometry_only` passes, the by-feel loop is a normal play launch (§9) —
no probe needed, and no reset between values:

```powershell
powershell -File run_foot_demo.ps1 -Arm padlive     # then press N / M / , / .
```

`geometry_only` prints both feet's offsets, the radius, and the distance from
the nail origin (expect ≈ 5.8 cm). A very different distance means the ball
lookup found the wrong mesh; the raise names the collision prims present.

`sag_ab` must show the settled base height dropping and the peak pad force
falling as k decreases. **Identical rows mean the pad is inert in that pose** —
check the toe height against the env's own contact line
(`no_flight_contact_threshold`, 0.084) before concluding anything, because the
S4a stance keeps the balls lightly loaded and that is the known way every foot
experiment in this campaign returned a null result.

**Step 2 — calibrate.** Measure the real pad's sag under a known load. That
gives `k`, and `pad_thickness` must be several times the resulting sag.

**Step 3 — the ladder.** Only then pick a DR band, from Step 2's measurement
plus a physically plausible spread, and check the softest rung against §3's
travel constraint.

**Step 4 — training.** Turn on `pad_contact_enable` with the band. Note that the
pad's per-env draw is **not** in tensorboard: the env logs rewards, curriculum
and diagnostics, but no plant-channel histograms, and the belt draw is not
logged either. So the DR draw is verified by `env_sweep` (Step 1.3) or not at
all — do not expect TB to show it. If a per-episode band histogram is wanted,
it has to be added to `_get_rewards`' extras alongside the existing
`diagnostics/*` keys.

Two ways a band can look like a distribution and behave like a constant: a
band collapsed to `min == max`, and a band whose soft rungs all bottom out on
the travel constraint in §3. Both are worth an explicit check before a long run.

**Open questions carried forward.**

- Does pad compliance even matter on this robot at the S4a stance, given how
  lightly the balls load? Steps 1 and 2 answer this before any training compute
  is spent, which is the point of gating.
- Should `pad_effective_mass` stay 1.0? It is the single-support catch mass and
  it sets the damping, not the stiffness. A measured value would be better than
  the convention inherited from `foot_contact_effective_mass`.
- ~~The play hotkeys still cannot retune the pad live.~~ **BUILT 2026-09-12
  (evening) — see §9.** The pad is a force rather than a material, so a knob
  reaches it on the next physics substep with no reset and no re-attach. The
  design's own benefit, collected.

---

## 9. Live pad tuning, and the pad arms (2026-09-12 evening)

The §7 prediction held: because the pad is a force, the by-feel loop cost one
small class and no new mechanism. Nothing here changes the model in §2–§3; it
changes who can write `set_parameters()` and when.

### Keys (play_sim2real, live — no reset)

| key | knob | ladder |
|---|---|---|
| `N` / `M` | pad stiffness | `0, 200, 1e3, 5e3, 1e4, 2e4, 5e4` N/m |
| `,` / `.` | pad damping ratio | `0, 0.3, 0.7, 1.0, 2.0` |

These are `PAD_*` knobs and are deliberately **not** on the material ladder's
keys. The material keys (`PgUp`/`Home`/`Ins`/`F`) address the stage-global
`foot_contact_material_*` prims and still need a reset; the pad keys address the
per-env force and are in the plant by the next physics step. Two plants, two key
families, two different messages in the terminal — the split is the point, not
an accident of key availability.

### How a press is applied

`_PadKnobs._write()` does **two** things, and both are load-bearing:

1. `env._pad.set_parameters(...)` — the live force object. Live on the next
   physics substep.
2. `env._dr_override["pad_contact_hold"] = k` — a full `(num_envs, 2)` row, so
   the value survives `R`.

Write 2 needed a **new** key rather than the existing `pad_contact`, and the
distinction is not cosmetic. A sweep driver paints a different value into every
cell and re-paints per arm, so its override must keep winning and skip the band
draw entirely. A by-feel pin is one value for the whole session, and if it used
the same key a single `N` press would replace the per-env band with a constant
for the rest of the run while the terminal still showed a band. So
`pad_contact_hold` is a narrower contract: `_sample_dr` draws the band as usual,
then writes the held value over the resets that share it (a held value of zero
leaves the draw alone, which is what keeps the inert arm inert without a special
case). The sweep driver's path is untouched — it passes neither key.

The cfg band is also collapsed onto the held value on a press. That is not a
convenience: `_sample_dr` takes its **inert** branch — writing zeros for the
whole reset — whenever `pad_stiffness_max` is zero, so a live hold over a
zero band would be erased on the next reset and misreported as "INERT — no force
applied" while the pad was in fact pushing. Collapsing removes both. The cost is
that a resumed process re-samples the tuned value instead of the original band;
the knob keeps the original in `self.band` and restores it when keyed back down
to the value it was seeded from.

### Launch flags

The pad is built in `RlEnv.__init__`, before any hotkey can run, so a hotkey can
never *create* one. `play_sim2real` therefore gained `--pad_enable`,
`--pad_stiffness`, `--pad_stiffness_min/max`, `--pad_damping_ratio` and
`--pad_thickness`, applied to the cfg immediately before `gym.make` (an explicit
`env.pad_*` hydra override still wins, matching the script's rule 3 for the
deploy contract). `--pad_stiffness_min/max` imply `--pad_enable`.

`run_foot_demo.ps1` gained four arms that use the per-env path and leave every
material knob at its default:

- **`padnull`** — the pad built at `k=0`. The honest control: the object exists
  and the geometry was measured, but a zero-stiffness pad applies nothing. If
  this differs from `rigid`, the pad's mere presence moves the sim, and that has
  to be understood before any pad result is believed.
- **`padlive`** — built inert, tuned by hand with the keys above.
- **`padband`** — a real per-env draw, 5e3…5e4 N/m.
- **`padsoft`** — 500 N/m, below the travel constraint on purpose, so the
  startup warning fires and can be trusted the next time it stays silent.

### Two silent-failure modes, now reported at startup

`report_pad_plant()` runs right after the env is built and prints the live
plant, plus the two ways this knob would look broken while reading correctly:

- **Sag vs travel.** A pad whose equilibrium sag (`m*g/k`) exceeds
  `pad_thickness` bottoms out on the ball's rigid contact and then reports its
  *thickness* as the settled sag. At the 2 mm default the hard floor is
  `k > 4900 N/m` for a 1 kg catch; `padsoft` is built to trip this.
- **Damping cap.** Damping is clipped at `pad_damping_stability_factor*m/dt`
  (250 N s/m here). At `k=5e4` the uncapped `zeta=2` request would be 894, so
  the sim runs `zeta_eff ≈ 0.56`. `_PadKnobs` reads the effective ratio **off
  the pad** and never echoes the request, and the key press prints the cap when
  it bites.

Both are the "knob that reports a plausible number" failure with a fresh face,
so both are checked in `data/_test_pad_knobs.py` (54 checks; the warning must
fire on `padsoft`'s rung and stay silent in the useful band). A third of the
same family is checked there too: the knob must report the **effective** damping
ratio read back off the pad, never the ratio it asked for — at `k=5e4` the
uncapped `zeta=2` request becomes `zeta_eff ≈ 0.56`, and a knob that echoed its
own request would be describing a plant that is not running.

### Still unverified

Unchanged from §5: **the geometry measurement and every in-sim behaviour**. The
pad knobs are wired and unit-tested, but nothing in this section has run in
Isaac Sim either — a live knob is only as live as the pad it writes to, and no
pad has yet been built. `run_probe_pad_contact.ps1 -Mode geometry_only` is still
the first thing to run, and it now gates the hotkeys too.

---

## 8. Bottom line

Pad compliance DR is now possible, per environment, through a force the env
owns, and that force is tunable live (§9) from the play loop and from the
command line. It is wired, gated, inert by default, and unit-tested where a unit
test was possible. The unverified part is narrow and named: the geometry
measurement and every in-sim behaviour. Run `geometry_only` first — it is cheap,
and it is the one thing standing between this and a calibration run. It now
gates the live knobs as well as the probe: a hotkey writing a pad that was never
built prints a message and changes nothing, which is the correct behaviour and
also the reason `geometry_only` must pass before any by-feel session.
