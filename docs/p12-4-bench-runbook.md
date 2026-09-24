# p12.4 bench runbook — plant-gap isolation (drag / knee push / supine IMU)

Written 2026-08-31, after the p12.3 arm-A verdict (see CLAUDE.md sim2real section
and memory `p12-3-blind-flight-verdict`): sim recovers in place from the exact
hardware post-arrest fork state 19/20 under the same weak commands where hardware
fell 12/12 → the gap is plant-side re-righting. Candidates: **H-belt** (knee belt
compliance/slip at low load), **H-contact** (mat/ball-foot), **H-COM** (mass model).
This session measures all three. User runs all motion/power; protocols accepted
2026-08-31.

## 0. Mechanical reference (user description, recorded 09-01)

Condensed here because several facts below are load-bearing for §2 and §4.

- 6-DOF point-foot biped, **5.57 kg**, ~50 cm standing height. In the nominal
  balanced posture all six actuators sit near one horizontal plane at the top.
- **Torso / hip roll**: torso = single plate carrying the high-mass battery,
  bolted to the REAR (non-output) faces of both hip-roll motors; hip-roll
  output flanges bolt to the L/R chassis plates, with a large V-groove bearing
  at each motor–chassis interface sharing radial load. **Torso COM is anterior
  to the hip-roll axis** (built-in forward-lean moment; torso at chassis height
  in zero pose). → H-COM audit lead: verify the USD torso/battery COM
  reproduces this forward offset (§5).
- **Chassis / hip / knee layout**: per-leg PETG chassis plate; hip-pitch and
  knee motors mounted back-to-back, output flanges facing laterally. Hip-pitch
  flange → medial side of PLA upper leg. Knee motor drives a **printed PETG
  pulley** (bearing-supported) → remote knee drive along the upper leg; the
  resulting hip–knee kinematic coupling is decoupled in software
  (motor_bridge).
- **Knee transmission**: PU steel-corded timing belt, 5M pitch, 10 mm wide,
  500 mm nominal circumference — **measures ~504–508 mm when tensioned**
  (0.8–1.6% elongation at installation tension; feeds the §2 compliance
  prior). Driven pulley at the lower-leg proximal end, 22:30 ratio (joint
  pulley r ≈ 23.9 mm, 1 tooth = 12° joint-side). Tension is set by translating
  the knee ground shaft (center distance) via set screw. Knee axis = ground
  shaft + bearings; lower leg PLA.
- **Foot**: TPU spherical point foot (drag-tested §1: not squashy, free-rolls).
- **Tooth skip is easy to observe by eye** — the slip failure mode is directly
  visible, so §2's slip-vs-compliance disambiguation is cheap; what is NOT
  visible without instrumentation is elastic windup, which is what §2 exists
  to measure.

## 1. Mat drag test (SAME mat + tape state as the p12_3 trials)

Robot in stiff HOLD at zero pose throughout (kp turned up — a limp robot folds
when dragged; user arms/powers). Partner steadies with fingertips at the torso,
**horizontal forces only — any vertical hand force corrupts N**. Weigh the robot
in flight config first (W feeds every ratio below).

Protocol note (revised 09-01): 1c is measured as TWO separate quasi-static
items, because whether a ball foot rolls or slides is set by the HEIGHT of the
applied force line (low line ≈ through the contact → can only slide; high line
→ pivots/rolls) — so height is a controlled variable, not an outcome to
classify — and because any "allow drift then catch" phase contaminates the
reading with falling momentum. The robot is never released in any item.

- **1a. Linear μ** (both feet on mat — no one-foot balance needed here): drag
  the whole robot horizontally with the luggage scale, pull line LOW at foot
  level (a high pull adds a topple moment the partner then fights). Slow ramp;
  record breakaway F_s and steady-slide F_k; μ_s = F_s/W, μ_k = F_k/W.
  3 repeats **per direction** (fwd AND back — taped mat may be anisotropic; the
  failing case is a forward-loaded foot). Classify sliding: smooth vs
  stick-slip (note jerk amplitude).
  **Return:** W; per direction: 3× F_s, 3× F_k, smooth/stick-slip.

  **RESULT 09-01 (variant: one foot dragged, other planted → N₁ ≈ W/2):**
  no identifiable static breakaway — steady even-speed slide ("milling"/plowing
  through the squashy TPU-on-mat contact), verdict SMOOTH, μ_s ≈ μ_k within
  scale resolution. F_k = 2.4–3.0 kgf (23.5–29.4 N); the 3.0 was on properly
  TAPED regions — tape *increases* drag (p12_3 trials ran taped). Slippery
  tile reference: 0.8 kgf. Multiple repeats, multiple directions.
  **W = 5.57 kg (54.6 N) → N₁ ≈ 27.3 N → μ_k ≈ 0.86 (bare mat) – 1.08 (taped),
  ±~20% load-share uncertainty; tile μ_k ≈ 0.29 (sanity anchor, method OK).**
  Interpretation: available drag ≫ few-N post-arrest horizontal forces →
  foot-slide support loss DISFAVORED; H-contact weight shifts to 1c-ii
  (divot/roll-grip behavior of the indented ball).
- **1b. Torsional breakaway — DEAD-WEIGHT variant (revised 09-01, on-robot
  attempts failed)**: torsional grip is a pure ball+mat+load property, so take
  the robot out of the test. Unscrew one ball foot (robot off) or use a spare;
  mount it foot-down on the sacrificial 3D-printed shin; stack known mass on it
  (~2.7 kg ≈ half-weight; add ~5.5 kg full-weight level if easy) on the TAPED
  mat area. Lever across the shin, scale perpendicular at measured arm r, slow
  ramp; T_s = F×r. Expect T_s only ~0.2–0.5 Nm → use a LONG arm (≥15–20 cm).
  3 repeats per load. Report T_s and a_eff = T_s/(μ_s·N).
  **Why not the 1a both-feet way:** twisting one foot of the rigid robot forces
  the planted foot to translate → the reading is that foot's linear friction ×
  stance width (~4–5 Nm), burying the true ~0.2–0.4 Nm torsional term.
  Fallback if the foot can't come off: other foot fully airborne, partner
  horizontal-only at torso (N₁ ≈ W), accepting hand-yaw contamination.
  **Return:** stacked mass per level, r, 3× breakaway F per level.

  **RESULT 09-01 (dead-weight rig: uninstalled shin + pliers, 1.080 kg,
  N = 10.6 N):** two lever arms agree to 1% — 85 mm × ~50 g → 0.0417 Nm;
  35 mm × 120 g → 0.0412 Nm ⇒ **T_s ≈ 0.041 Nm @ 1.08 kg, a_eff ≈ 4 mm**
  (μ from 1a). Tile: 0 g (below scale resolution, consistent with near-point
  contact). Extrapolated to half/full robot weight: ~0.1–0.2/~0.3–0.5 Nm.
  **Verdict: torsional grip DISFAVORED as the gap** — sim's sphere-on-plane
  contact has ZERO torsional friction, so the real plant has MORE yaw grip
  than sim, the wrong direction to explain hw-falls-where-sim-recovers.
  H-contact now rides entirely on 1c-ii (divot roll-grip, where extra grip
  WOULD hurt re-righting). Optional: second load level ~2.5–3 kg to pin the
  T_s-vs-N scaling (expect ~100–150 g at 85 mm).
- **1c-i. Forward slide breakaway, single loaded foot** (the post-arrest stance
  foot pushed forward by knee extension): easiest on the 1b dead-weight rig —
  same foot+shin+stacked-mass setup, pull forward horizontally at the shin
  just above the ball, slow ramp to breakaway. (On-robot alternative: one-foot
  lean, partner pins the torso — hands on hips, horizontal only — so the rigid
  robot's foot can only slide.) Ink dot on the front of the ball + ruler on
  mat, film close (dot fixed on ball while the ball translates confirms
  slide). 3 repeats × 2 load levels. Millimeters matter — the fall needed only
  ~cm of effective support loss.
  **Return:** per load level: N (stacked or N₁), 3× breakaway F, close video.
- **1c-ii. Roll grip, single loaded foot**: no scale, no release. On the
  dead-weight rig: secure the weights to the shin (tape/strap) and slowly tilt
  the shin forward and back ~5–10°, hands on it the whole time — quasi-static
  throughout. (On-robot alternative: partner tilts the robot about the loaded
  foot.) Close film of the ink dot + ruler + the mat DIVOT (1a showed the ball
  indents the squashy mat — watch how the divot couples to the ball).
  Classify: FREE ROLL (dot migrates smoothly, contact point advances ≈ R·Δθ) /
  STICK-SLIP (dot fixed, then jumps) / GRIP (dot fixed, tape adheres / ball
  sits in its divot, contact acts as a pin). This answers the H-contact
  discriminator directly — does the loaded ball behave as a free wheel or an
  anchored pivot — with no force measurement and no free-fall phase.
  **Return:** close video, classification, ball radius R (from CAD is fine).

  **RESULT 09-01 (hands-on with the uninstalled shin+ball, ~1 kg load; 1c-i
  force not formally measured):** TPU ball not that squashy (the 1a squash was
  mat-side); **FREE ROLL, rolling resistance below luggage-scale resolution**.
  **§1 VERDICT — H-contact DEAD.** Sim contact (rl_env_cfg_p6 L338: μ=1.0
  nominal, DR 0.5–1.1, PhysX = no torsional/rolling friction → free-rolling
  sphere): measured linear μ 0.86–1.08 is INSIDE the DR band; torsional grip
  real 0.041 Nm > sim 0 (stabilizing direction); rolling free = free. No
  contact property differs from the recovering arm-A sim in the destabilizing
  direction. NOTE — pre-registered tree branch "forward roll under load →
  H-contact" was MIS-SPECIFIED (it assumed the sim foot holds; the sim ball
  free-rolls too, checked in cfg), overridden on that stated basis.
  Residual caveats (recorded, not load-bearing): roll test at ~1 kg not half
  weight (deeper divot could only ADD grip = wrong direction); mat vertical
  compliance unmodeled (rigid plane in sim), no mechanism to the slow topple.
  Gap candidates remaining: H-belt (§2, elevated) and H-COM (§4 video).

## 2. Knee push-test (HOLD mode, low-load sweep FIRST) — both knees

**Why this test matters now (09-01, after §1):** with H-contact dead this is
the ONLY direct discriminator left for H-belt — the T2 video (§4) only speaks
to H-COM. The knee encoder is motor-side; belt windup between motor flange and
joint is invisible to every logged channel (channel-trust rule #9), so an
independent joint-side angle is the only way to see it. And it is genuinely
undecided: the measured belt stretch (§0, 4–8 mm tensioned on a 500 mm loop)
implies a windup prior at 1.5 Nm of roughly

  θ ≈ τ·L_span·δ / (2·T₀·L_belt·r_j²) ≈ **0.01–0.08 rad** for T₀ = 50–200 N

(L_span ≈ 185 mm per span, r_j ≈ 23.9 mm) — a range that STRADDLES both
pre-registered thresholds below, because installation tension T₀ is unknown.
The printed PETG pulley teeth add series compliance on top; the test measures
the aggregate motor-flange→joint compliance, which is exactly the quantity a
sim arm would need. Two secondary predictions worth checking in the data:
(i) if the deflection-vs-torque curve KINKS upward partway through the sweep,
the slack span has detensioned (happens at τ ≈ 2·T₀·r_j, e.g. 2 Nm ↔ T₀≈42 N)
— itself a low-tension diagnostic and a precursor to tooth jump; (ii) since
tooth skip is easily visible (§0), any persistent offset should also have been
SEEN as a skip — a persistent offset with no observed skip would implicate
tensioner/set-screw slip instead.

**Minimal version if time-limited:** the 0.5/1/1.5/2 Nm steps are the only
decision-relevant ones (the tree's discriminator lives entirely at ≤2 Nm);
the 4/8 Nm steps are characterization (compliance curve + tooth-jump margin)
and may be dropped or done on one knee only.

**Setup — no fixturing needed (revised 09-01).** Two facts make pins/clamps
unnecessary: (a) the readout is the RELATIVE thigh–shin angle from markers on
both segments, so rigid-body motion of the whole robot cancels — the thigh only
needs to REST somewhere, not be clamped; (b) gravity replaces the hand-applied
force, so nothing pushes on the robot at all.

Protocol: robot lies on its back on the table, thigh flat on the tabletop,
shin overhanging the edge roughly horizontal. Knee in HOLD at deploy gains
(30/1.5 joint-side) near the post-arrest angle (~0.2 rad flexion). If the
torso/thigh lifts under load, lay a book across it — that is the entire
fixture. Flip the robot (prone vs supine) to load the other knee direction;
**prioritize the direction that resists knee EXTENSION** (the failure regime:
motor extending against external load).

Loading: hang known masses from the foot (string loop behind the TPU ball).
τ = m·g·d where d = HORIZONTAL distance from knee axis to the string — measure
d per frame in Kinovea (ruler in shot), don't assume it. At d ≈ 0.20 m the
steps 0.5/1.0/1.5/2.0 Nm ≈ 0.26/0.51/0.77/1.02 kg — water bottles work.
The shin's own weight adds a constant pre-torque; irrelevant, since every
reading is a DELTA from the unloaded overhanging pose (re-zero each cycle).
Fallback if edge-overhang geometry doesn't work: luggage scale pulled
perpendicular to the shin at measured r (τ = F×r), original protocol.

**Torque steps: 0.5, 1.0, 1.5, 2.0 Nm first — the post-arrest window ran at
knee torques <1.5 Nm — then 4, 8 Nm (optional, see minimal version).**
Markers: 2 high-contrast on thigh + 2 on shin, spaced as far apart as the
segments allow. Camera propped/tripod on the knee-axis line (looking at the
robot's side, table height). Zero-load frame 3× first — angle-noise floor
must be < ~1°. Hold ~3 s per step; record (i) encoder knee angle (GUI readout
fine, CSV export broken), (ii) Kinovea thigh–shin angle. 3 load-hold-release
cycles per step; after release check video angle returns to zero (persistent
offset = TOOTH SLIP — cross-check by eye, skips are visible per §0;
proportional deflection that returns = COMPLIANCE). Right knee had the 08-25
tooth skip — test both. Crawl-pose check before AND after the session
(standing rule; reference + tolerances in §2b).

Readout: belt deflection = Δangle(video) − Δangle(encoder) per torque.

**PRE-REGISTERED thresholds (on record 08-31):**
- deflection ≥ ~0.1 rad (≈6°) at ≤2 Nm → **H-belt confirmed** (~1–2 cm lost COM
  travel, the scale of the balance margin);
- ≤ ~0.03 rad at ≤2 Nm → **H-belt dead for this fall regardless of high-load
  behavior** — compliance/slip appearing only above ~2 Nm cannot explain a
  failure occurring below 1.5 Nm; promote H-contact/H-COM;
- between → inconclusive; compute implied COM deficit and rank.

### 2b. Crawl-pose reference for the §2 tooth-skip checks (sim-validated 2026-09-01)

The zeroing/check reference `config.py CRAWL_JOINT_POS = [0, 1.59, −0.2, 0, −1.59, 0.2]`
was independently re-derived by letting the **ballfeet** asset settle limp on a
near-frictionless floor (`script/show_crawl.py`, new crawl-mode button: joints
limp + μ≈0.02 + ramped ~5 N/3 N push on thighs/foot tips; standalone launch
`script/run_show_crawl.ps1`). Settled pose vs reference: hips 1.6009/−1.5777,
knees −0.2020/+0.1778 — worst error 1.3°, and the L/R mean nails the reference
(hip 1.5893 vs 1.59, knee 0.1899 vs 0.20; the residual is a whole-body tilt).

Consequences for the bench check:
- The **knee ≈ ±0.2 rad flexion in crawl is real geometry**, not a mesh or
  placement artifact: both legs use entirely different meshes and the old
  bigthigh-urdf export (source of the May reference) reproduces to ~1°. Do NOT
  "correct" the knees toward 0 when placing the robot.
- Residuals ≤ ~1.5° at placement are normal and irrelevant: check tolerance is
  0.15 rad and one belt tooth = 12° joint-side — unambiguous either way.
- Roll motors stay off for the check; ≤0.5° roll residual contributes ~0.005 rad
  offset, noise vs placement repeatability.

## 3. Supine IMU check (~10 min, no motors)

Stream displayed pg_x ~30 s per pose, take means:
- supine ≈ **−0.911**; prone ≈ **+1.089** (displayed |pg_x| > 1 = the −0.089 trim
  made directly visible);
- **prone + supine sum = +0.178 ± 0.01** (mounting tilt cancels exactly — verifies
  the trim is applied as a constant);
- mounting tilt = (upright-against-level-reference mean) − 0.089.

## 4. Consolidated data-return checklist

From §1 (the **Return** lines, gathered):
- W = robot weight in flight config: **5.57 kg (done 09-01)**.
- 1a: **done 09-01** — μ_k ≈ 0.86 bare / 1.08 taped, smooth, no static peak;
  tile anchor 0.29.
- 1b: stacked mass per level, lever arm r, 3× breakaway F per level.
- 1c-i: two load levels, each with N, 3× breakaway F, close video (dot+ruler).
- 1c-ii: close video of the slow tilt, FREE ROLL / STICK-SLIP / GRIP call, R.

From §2: per knee, per torque step (0.5/1/1.5/2 Nm required; 4/8 Nm optional,
see §2 minimal version): encoder angle + Kinovea thigh–shin angle,
ramp-hold-release ×3, return-to-zero yes/no; note any visible tooth skip and
any kink in deflection vs torque; crawl-pose check before and after.

From §3: pg_x means supine / prone / upright-against-reference (~30 s each).

Plus:
- T2 video annotations: [0.08–0.17] right-foot contact + per-foot travel (cm);
  [0.2–0.6] feet planted? / torso backward translation (cm) / knees visibly
  extending?

## 5. Next session opens by applying the p12-4 decision tree AS-IS

(memory `p12-4-decision-tree`; mirrored in CLAUDE.md)
- Push-test joint-side lag/slip at ≤1.5–2 Nm → H-belt confirmed → sim arm with
  joint-side compliance model; update next-training directive.
- ~~Drag test → H-contact~~ **RESOLVED 09-01: H-contact DEAD** (see §1
  results: μ inside DR band, torsional real>sim=0, free roll = sim's
  free-rolling sphere; the "forward roll → H-contact" branch was
  mis-specified — sim rolls too). No wave-2 contact arms.
- Video [0.2–0.6]: knees extend joint-side + feet planted + torso still drifts
  back → H-COM → mass-model audit (first check: §0 says the torso/battery COM
  is ANTERIOR to the hip-roll axis — verify the USD reproduces that forward
  offset before auditing anything subtler).
- Multiple positive → rank by measured magnitude; combined sim injection arm.
- All clean → all three candidates dead; say so plainly; re-open hypothesis
  formation from scratch.

Deferred/held: trim A/B flight (indefinitely; method-validation only), wave-2
contact arms (until drag numbers exist).

## 6. H-kp-OOD — knee series-elastic stiffness vs the training DR box (desk analysis 09-01)

Hypothesis: training modeled the knee drivetrain as kinematically RIGID; if the
real effective joint stiffness k_eff = (1/kp + 1/k_belt)^-1 is below the kp DR
lower bound the R1r policy trained under, the deployed knee loop is
out-of-distribution. This is the quantitative low-load-compliance branch of
H-belt (§2) — same physical object, sharper thresholds, plus a kd axis. It does
NOT replace §2's pre-registered thresholds; it adds its own boundaries on the
same measurement (§6.5).

**DESK VERDICT: NOT DECIDABLE FROM CURRENT NUMBERS — the two available belt-EA
estimates disagree by 10–40×, and k_eff lands OOD on one branch and comfortably
in-distribution on the other. The §2 push-test decides it for free (one added
readout + the 4 Nm step upgraded to decision-relevant). All formulas and
boundaries below are pre-registered before any push-test data exists.**

### 6.1 Gating checks (all passed — the series formula applies)

1. **Encoder location — MOTOR-side, confirmed in code.** The deploy stack sends
   kp/kd to the RobStride firmware, whose MIT-mode PD closes on the motor's own
   encoder (`motor_bridge.py:253-257` sequential / `:277-284` batch: `send_command(mid,
   p_des, v_des, kp, kd, t_ff)` — gains go INTO the motor). There is no
   joint-side encoder anywhere in `control/config.py`. The knee "joint position"
   in every log is the rigid decode `motor×G − hip×c` (`motor_bridge.py:116-123`).
   ⇒ static joint stiffness IS the series combination, and belt windup is
   invisible to the policy's obs AND to all npz channels (channel-trust rule #9).
2. **Gain coordinates — same side, N² exactly once.** Deploy: joint 30/1.5
   (`config.py:181-182`) → motor 16.13/0.807 via ×(22/30)² once
   (`motor_bridge.py:135-141`); reflected back through the belt this is joint
   stiffness 30 for a rigid belt. Sim: `servo_kp_knee`/`servo_kd_knee` are applied
   directly as JOINT-space actuator stiffness/damping (`rl_env_p6.py:286-289`),
   PD error and 23.2 Nm clip both joint-frame (`belt_knee_actuator.py:168,181-185`
   + module docstring item 3). Contract clean; any stiffness gap is plant-side.
3. **Premise — sim belt is kinematically rigid, confirmed.** `belt_knee_actuator.py`
   models firmware-PD coupling terms (belt drag, stale-hip latch, motor-velocity
   torque envelope, rotor inertial coupling) but has NO elastic state: efforts are
   computed from true joint positions, transmission enters only through the
   coupling coefficient. No series-spring term exists (whole file read 09-01).
4. **Coupling / off-diagonal stiffness — reduces to the knee diagonal.** Real
   static knee row: tau_kn = −k_eff·(q_kn + c·q_hp − G·θ_m_des); the bridge bakes
   the measured hip into θ_m_des each tick (`motor_bridge.py:131`), so the
   −k_eff·c·q_hp cross-term is cancelled at 100 Hz and only the within-tick
   phantom error survives — which sim replicates exactly (stale-hip latch,
   `belt_knee_actuator.py:181-183`), just with kp instead of k_eff. Discrepancy
   ≈ (kp−k_eff)·c·(within-tick hip motion) ≈ 0.05–0.09 Nm at 1 rad/s hip — noise
   at the 1.5 Nm scale. Hip-row reaction c·tau_kn is empirically absent on
   hardware (08-06 replay, `config.py:126-133`; sim `hip_reaction=False` matches).
   ⇒ no significant second OOD axis; H-kp-OOD is a knee-diagonal question.
   (Hip-pitch itself is direct-drive — flange to thigh, no belt — kp 20 rigid.)

### 6.2 First-principles numbers

Geometry (derived from 22T/30T, 5 mm pitch, 100-tooth loop): r_m = 110/2π =
17.51 mm, r_j = 150/2π = 23.87 mm ✓ §0's 23.9; center distance C from
500 = 2C + π(r_j+r_m) + (r_j−r_m)²/C → C ≈ 184.9 mm; free span
L_s = sqrt(C² − Δr²) ≈ **184.8 mm** ✓ §0's ≈185.

**T0 from the pluck (measured f = 140 Hz, right knee, fastened):**
T0 = μ·(2·L_s·f)² = μ·(51.74 m/s)² = μ·2677 m²/s².
Datasheet linear density for 5M-10 steel-cord PU: 48 g/m
([NK-Technics](https://nk-technics.com/htd-rpp-std-pitch/pu-timing-belt-htd5m));
maker spread ~33–48 g/m. ⇒ **T0 = 88–129 N** (128 N at the found datasheet
value) — the earlier "≈100 N" is verified as order-correct. [derived]
Slack-span detension kink: τ_kink = 2·T0·r_j = **4.2–6.2 Nm** — between the
optional 4 and 8 Nm §2 steps; a kink observed there would pin T0 independently.

**EA — the fork that decides everything:**
- (i) *Elongation-derived* [from §0's measured 504–508 mm tensioned]:
  EA = T0·L_belt/δ = (88–129)·0.5/(0.004–0.008) = **5.5–16 kN**.
- (ii) *Steel-cord datasheet*: specific stiffness for 5M/10 mm steel belts
  ≈ 1.6×10⁵ N ([Optibelt PU manual](https://www.optibelt.com/fileadmin/extranet/power_transmission/technische_Unterlagen/Optibelt-TM-Polyurethane-Timing-Belts.pdf))
  to 2.3×10⁵ N ([SDP/SI T-10](https://www.sdp-si.com/D265/PDF/D265T010.pdf)):
  **EA = 1.6–2.3×10⁵ N**.
These disagree by **10–40×**. A real steel-cord belt at T0 ≈ 110 N stretches
δ = T0·L/EA ≈ **0.3 mm**, not 4–8 mm. So EITHER the 504–508 measurement is not
cord strain — candidates: printed-PETG pulley OD oversize (+1% diameter = +4 mm
path with zero strain), outside-surface vs pitch-line measurement (back of belt
sits ~3 mm above pitch line), one-time tensioner/set-screw seating travel (§0:
tension set by translating the knee ground shaft — any seating/mount compliance
is in series and would ALSO show up in the push-test, correctly, but not in cord
EA) — OR the compliance is real but lives in the PETG pulley teeth /
tooth-engagement / shaft mount rather than the cords. Note the prior "k_belt
40–80 back-calculated from a 1.5 Nm windup of 0.018–0.036 rad" is NOT an
independent estimate — it is the same elongation number fed through the same
formula (verified: §2's windup prior θ = τ·L_s·δ/(2·T0·L_belt·r_j²) is
algebraically τ/k_b with k_b = 2·EA·r_j²/L_s and EA = T0·L_belt/δ). One suspect
measurement carries the entire soft branch.

**Torsional stiffness at the knee joint (both spans taut, motor locked):**
k_b = 2·(EA/L_s)·r_j² = EA × 6.17×10⁻³ /m:
- soft branch (i): **k_b = 34–99 Nm/rad** (reproduces the prior 40–80);
- datasheet branch (ii): **k_b ≈ 990–1420 Nm/rad**.

**Effective quantities the policy meets (kp = 30, kd = 1.5 joint-side):**
- k_eff = (1/30 + 1/k_b)⁻¹: soft **16–23 Nm/rad** (mult 0.53–0.77); datasheet
  **29.1–29.4** (mult 0.97).
- kd_eff = kd·(k_b/(k_b+kp))² (low-frequency term of the series impedance
  Z(s) = k_b(kp+kd·s)/((kp+k_b)+kd·s)): soft **0.42–0.88 Nm·s/rad** (mult
  0.28–0.59); datasheet **1.41–1.44** (mult 0.94–0.96).
- Free-shin series mode f = (1/2π)·sqrt(k_b/I_shin), I_shin ≈ 0.01 kg·m²
  [assumed, user estimate]: soft **9–16 Hz** (reproduces the 10–15 Hz prior);
  datasheet **50–60 Hz**. The prior ζ ≈ 1.7 "overdamped" does NOT survive
  re-derivation — the motor-side damper sits on the far side of the spring and
  damps this mode only weakly; expect an underdamped ring unless belt hysteresis
  dominates. Tap test (§6.5 E3) measures it. In STANCE the load-side inertia is
  the whole robot, not the shin — the fall regime is quasi-static, so stiffness
  (not the mode) is the relevant quantity there.

### 6.3 The trained box (R1r_3150 actual values) and the verdict per branch

R1r_3150 = `model_3150.pt` from `RL/logs/rsl_rl/p9_replant/2026-08-27_10-30-37_R1_resume3110`
(`RL/run_p12_3_armA_GPU0.ps1:49`). Its saved `params/env.yaml` (resume re-passed
all overrides — verified identical to the 08-25 parent):
- `servo_kp_knee: 30.0`, `servo_kd_knee: 1.5` (env.yaml:369-370) — cfg defaults
  are 8.0/0.5 (`rl_env_cfg_p6.py:674-675`), so the override check was mandatory;
- `dr_stiffness_mult: 0.8–1.2` (env.yaml:435-436) — **the R line widened the cfg
  default 0.85–1.15** (`rl_env_cfg_p6.py:970-971`); box = **kp 24–36 Nm/rad**;
- `dr_damping_mult: 0.7–1.3` (env.yaml:437-438); box = **kd 1.05–1.95**;
- `dr_effort_mult: 0.8–1.0` (env.yaml:439-440) — irrelevant at ≤2 Nm vs 23.2;
- obs: `dr_obs_noise_joint_pos: 0.008`, `dr_joint_pos_bias_max: 0.07`
  (env.yaml:464-465); `knee_decouple_ff: true` (env.yaml:243) = deploy ✓.
DR mults are applied per-env, per-episode onto the joint-space nominals
(`rl_env_p6.py:1025`, sampled at `:2844-2845`).

| quantity | real, soft branch | real, datasheet branch | trained box | soft-branch margin |
|---|---|---|---|---|
| knee kp (Nm/rad) | 16–23 | 29.1–29.4 | 24–36 | **1–8 below floor (4–33%)** |
| knee kd (Nm·s/rad) | 0.42–0.88 | 1.41–1.44 | 1.05–1.95 | **0.17–0.63 below floor (16–60%)** |
| off-diag (Nm/rad) | ~0 net (see 6.1.4) | ~0 | 0 (per-joint DR) | not an axis |

Notes: (a) **kd goes OOD before kp** as the belt softens — kd_eff crosses its
floor at k_b < 154 Nm/rad vs kp at k_b < 120 — so kd is the more sensitive axis;
the kd claim is however confounded by the separately-measured excess physical
viscous friction (memory `viscous-dr-band-below-measured`), which adds real
damping the DR also missed, in the OPPOSITE direction. (b) The obs side is
partially covered by training: quasi-static windup at ≤2 Nm (soft branch
0.02–0.06 rad) is inside the ±0.07 rad trained encoder-bias band in magnitude —
the genuinely untrained axis is the LOOP STIFFNESS, not the obs offset.
(c) Arm A pinned only friction and pg-bias (`run_p12_3_armA_GPU0.ps1:56-57`),
so its 19/20 recoveries sampled stiffness draws spanning the whole 24–36 box —
sim recovers at the box floor; the soft-branch real k_eff sits below every draw
sim recovered under. The datasheet branch sits mid-box and kills the hypothesis.

### 6.4 Explanatory coverage vs the p12.3 observations, and rank

Against the fall record (12/12 backward quasi-static topples 08-31, arm-A fork
19/20 sim recovery; `data/p12_3_flight_20260831_v/`, memory
`p12-3-blind-flight-verdict`):
- **Post-arrest re-righting failure where sim recovers** — explains
  QUALITATIVELY: under backward gravity load the knee yields by τ/k_b beyond the
  servo error, and the motor-side encoder reports "on target" while the joint
  lags, so neither policy nor logs see it. MAGNITUDE IS MARGINAL: at the
  <1.5 Nm window torques the soft branch gives 15–44 mrad windup ≈ **1.5–9 mm
  COM-travel deficit** (§2's 0.1 rad ≈ 1–2 cm scaling). Direction-consistent
  (yield follows the load = backward), but whether a few mm breaks a 19/20
  recovery needs the injection arm (E6), not desk arithmetic.
- **Fixed-point dead zone / weak commands** — does not explain (policy-side
  property, exists in sim too).
- **Early forward-step arrest at −13° @0.17 s** — does not explain; out of
  scope anyway (arm A injects the post-arrest state, so the live gap is
  entirely post-fork).
- **08-25 right-knee tooth skip** (memory `knee-belt-coupling-findings`) —
  UNIQUELY explained among live hypotheses: finite T0 = 88–129 N predicts
  slack-span detension at 4.2–6.2 Nm; catch-step knee demands routinely exceed
  that (clip at 23.2 is normal operation, checklist #7). Same finite-tension
  root, different regime — the skip itself says nothing about ≤2 Nm compliance.
- **§1 contact results** — orthogonal, unaffected.

**Rank: co-lead with H-COM; strict sharpening of H-belt (subsumes it).**
H-kp-OOD adds to H-belt: quantitative OOD thresholds, the kd axis, the
motor-side-encoder obs argument, and the tooth-skip link. It does NOT overtake
H-COM on desk numbers: its in-scope magnitude is a few mm of COM travel
(marginal), its soft branch rests on ONE suspect measurement (the 4–8 mm
elongation, contradicted 10–40× by steel-cord datasheets), and H-COM more
naturally owns the systematic backward directionality plus the early arrest.
H-COM's premise (USD vs real anterior torso COM, §0) is equally unverified.
Both are decided this week: H-kp-OOD by §2+E1–E4, H-COM by §4 video + USD audit.
"Confirmed OOD" ≠ "confirmed as the fall cause" — causality needs E6.

### 6.5 Falsification plan (cheapest first; §2 protocol and thresholds UNCHANGED)

- **E1 (free — added readout on §2 data).** Per torque step, windup
  D = Δangle(video) − Δangle(encoder); fit slope s = D/τ over the 0.5–2 Nm
  steps (+ 4 Nm, below). Then k_b = 1/s and k_eff = (1/30 + s)⁻¹.
  **Pre-registered H-kp-OOD boundaries:**
  - s > 8.3 mrad/Nm (k_b < 120) → **kp-OOD confirmed** (k_eff < 24);
  - s > 6.5 mrad/Nm (k_b < 154) → **kd-OOD confirmed** (kd_eff < 1.05);
  - s < 6.5 mrad/Nm → **H-kp-OOD dead entirely** (both axes in-box).
  Predictions: soft branch s = 10–29 mrad/Nm (visible: 1.2–3.4° at 2 Nm);
  datasheet branch s ≈ 0.7–1.0 mrad/Nm (INVISIBLE — under the ~1° noise floor
  even at 4 Nm). Note the soft branch predicts §2's own outcome lands
  "inconclusive-to-dead" by §2's bars (0.015–0.044 rad at 1.5 Nm, mostly under
  the 0.03 bound, far under the 0.1 confirm bound) while still being OOD by
  these bars — **a §2 "H-belt dead" verdict does NOT kill H-kp-OOD**; only
  s < 6.5 mrad/Nm does.
  **Consequence: the 4 Nm step (one knee suffices) is upgraded from optional to
  decision-relevant for H-kp-OOD** — at ≤2 Nm the kp boundary (0.96° at 2 Nm)
  sits at the noise floor; at 4 Nm it is 1.9°, cleanly resolvable. Check
  linearity across steps: an upward kink near 4–6 Nm = slack-span detension,
  pins T0 (§6.2) and is itself a low-tension diagnostic.
- **E2 (+10 min).** Coupling separation: during each push, also record the HIP
  encoder (GUI readout); correct the expected knee kinematics by c·Δq_hp before
  computing D. If apparent knee windup vanishes after correction → it was
  hip-side motion/compliance through the coupling, not belt. Optional variant:
  repeat one 1.5 Nm step with the hip motor LIMP (thigh held by the table/book
  only) vs hip in HOLD — separates hip-servo compliance from belt compliance.
- **E3 (5 min).** Tap test: knee in HOLD at the §2 pose, flick the free shin,
  240 fps slow-mo. Ring frequency: soft branch predicts **9–16 Hz**, datasheet
  branch **50–60 Hz** — a 4× discriminator robust to the I_shin guess. Combined
  with E1's static k_b it also extracts I_shin, and the decay gives the real
  damping ratio → c_b for the sim model (the ζ≈1.7 prior is unverified).
- **E4 (5 min).** Pluck BOTH belts, left and right: T0 = μ·(2·L_s·f)² per leg
  (μ = 0.048 kg/m datasheet). The 08-25 skip was the right knee — a lower right
  f would show tension loss (skip aftermath or set-screw drift) and per-leg
  asymmetry the diagonal DR also never modeled.
- **E5 (5 min, desk+calipers).** Attack the suspect measurement directly:
  caliper the printed pulley ODs vs nominal (PETG oversize check) and re-derive
  the expected tensioned path; measure elongation properly via tooth count over
  a marked chord. If the 4–8 mm dissolves into pulley oversize/measurement
  method, the soft branch loses its only support and E1 should come back flat.
- **E6 (only if E1 confirms; ~1 GPU-day).** Causality: series-elastic knee in
  sim at the MEASURED k_b, c_b (+ detension kink at 2·T0·r_j), rerun the arm-A
  injection protocol (`play_sim2real.py --inject_npz ...`, same fork states).
  Pre-register before running: H-kp-OOD is the gap only if recovery collapses
  from 19/20 toward the hardware 0/12.

### 6.6 Fix options if confirmed (cross-ref memory `next-training-directive`)

- **(a) Widen the DR box** (knee `dr_stiffness_mult_min` down to cover the
  measured k_eff/30, e.g. 0.55; `dr_damping_mult_min` to ~0.3): one-line
  override, but a knowingly imperfect proxy — in sim a low-kp draw shows its
  tracking error in the obs (sim obs = true joint), while on hardware the
  motor-side encoder HIDES the windup (obs reads on-target). Low-kp DR teaches
  "react to visible sag"; the real plant gives no such signal. Also a kp mult
  scales torque at all loads, whereas windup is a load-proportional position
  offset with hysteresis. Stopgap robustness only.
- **(b) Series-elastic knee in sim** (preferred if E1 confirms): extend
  `BeltKneeDCMotor` with belt spring k_b, damper c_b, optional detension
  kink/backlash at 2·T0·r_j — all measured by E1/E3/E4 — AND decode the knee
  obs from the motor-side coordinate through the rigid decode, matching the
  hardware channel semantics exactly. Fixes plant AND obs contract, and
  reproduces the tooth-skip precursor regime for free. Cost: one fast internal
  state at the 500 Hz substep. Add to the next-training-directive list either
  way, grounded by the E1/E3/E4 numbers.

### 6.7 ADJUDICATION 09-01 (bench: pulley OD, tensioner history, tap test) — **H-kp-OOD DEAD**

Bench inputs (user, 09-01, trusted): pulley OD nominal within 0.3%; tensioner
advance over ~5 months ≡ 1–2% PERMANENT elongation, reproduced across new
pulleys and a replacement thigh (creep in the belt itself); tap test at kd=0:
11 Hz @ kp=30 both legs, 14 Hz @ kp=50, ratio = √kp to 1.5%; single-inertia
series fit: belt-only frequency ~53 Hz point / ≥27 Hz lower bound (2-point fit,
zero DOF), servo ≈96% of total compliance at kp=30.

**Killed twice, independently, at the §6.5 pre-registered boundaries:**
1. *Elastic-stiffness bound.* Tap fit: J = kp/(2πf)² = 30/(2π·11)² = 6.28e-3
   and 50/(2π·14)² = 6.46e-3 kg·m² (consistent ⇒ servo-spring-dominated).
   Belt-only ≥27 Hz ⇒ **k_b ≥ 181 Nm/rad** (point 53 Hz ⇒ k_b ≈ 660). §6.5
   boundaries: kp-OOD needs k_b < 120, kd-OOD needs k_b < 154. Even the
   conservative bound clears both: k_eff ≥ 25.7 (mult ≥ 0.86 vs floor 0.80,
   +7%; point 28.7, mult 0.96) and kd_eff = 1.5·(k_b/(k_b+30))² ≥ 1.10 (floor
   1.05, +5%; point 1.37). The kd margin at the bound is thin (5%) but the
   point estimate clears by 30%.
2. *Soft-branch foundation dissolved.* The §6.2 soft branch's ONLY support was
   reading the 4–8 mm loop elongation as ELASTIC strain (EA = T0·L/δ ≈ 6–16 kN).
   The tensioner history shows that elongation is PERMANENT CREEP (reproduced
   across pulleys/thigh), not elastic stretch; the phantom-path alternative
   (pulley oversize) is also dead (OD within 0.3%). Elastic EA from the tap:
   k_b/6.17e-3 = **≥29 kN, point ≈107 kN** — datasheet-class, as branch (ii)
   predicted. (Note: steel cords should not creep 1–2% — the creep SITE is now
   its own question, §7 + forensics.)

Closure requirements: **no third kp point and no locked-rotor push needed** for
this hypothesis — §2 as planned suffices as the static cross-check (tap is
dynamic small-amplitude; §2 catches static stiction/backlash the tap can't;
predicted §2 outcome given k_b ≥ 181: deflection ≤ 11 mrad (0.6°) at 2 Nm,
point ≈3 mrad — i.e. **§2 should land in its own "≤0.03 rad → H-belt dead"
branch; any measured deflection ≥ 1° at 2 Nm would contradict the tap and
reopen**). **Supersedes §6.5's upgrade of the 4 Nm step: it reverts to
optional** — its H-kp-OOD decision role is void; if run anyway, it now serves
§7 (kink hunt at τ ≈ 2·T0·r_j = 4.2–6.2 Nm at today's tension = an independent
T0 measurement). §2 reverts to its original three-hypothesis scope, thresholds
untouched; its load-hold-release hysteresis cycles additionally serve the §6.8
backlash question at zero protocol cost. E6 (series-elastic sim arm) is
cancelled as an H-kp-OOD test; fix option (a) moot; option (b) survives only as
general model fidelity, low priority. Live fall-gap competitors after this
adjudication: **H-COM (§4/§5) and the §6.8 anomaly below.**

### 6.8 Inertia reconciliation — the tap CONTRADICTS the trained knee armature (open anomaly)

The same tap data that killed H-kp-OOD opens a bigger question. Sim-side knee
inertia at tap conditions (free shin, hip static):
- Link (shank+foot+pulley, one rigid body) about the knee axis, from the
  canonical URDF: `shin_inner` m = 0.2005 kg, COM 85.3 mm off-axis, Izz_com =
  1.039e-3 → **I_link = Izz + m·d² = 2.50e-3 kg·m²**
  (`urdf_onshape_export/ballfeet-urdf/robot.urdf:446-450`; right leg
  `nail___fillet1` :881-885 gives 2.50e-3 identically). Matches the cfg's own
  note "links 17/2.5e-3 about hip/knee" (`rl_env_cfg_p6.py:976-978`).
- Reflected rotor: `knee_armature: 0.036` (R1r env.yaml:374), DR mult 0.7–1.3
  (env.yaml:441-442) → **trained total J = 27.7–49.3e-3, nominal 38.5e-3**.

Measured: J = kp/(2πf)² = **6.28–6.46e-3 kg·m²** — clean √kp scaling across
two kp values, both legs. That is **6.1× below the trained nominal and ~4.4×
below the DR-box floor** (armature-mult equivalent ≈ 0.10–0.17 vs floor 0.7),
and 2.5× ABOVE the bare link value. Predicted tap under the trained plant:
√(30/0.0385)/2π = **4.4 Hz** (kp=50: 5.7 Hz). Observed **11/14 Hz**.

This is not attributable to belt stiffness: a 2-DOF mode analysis (rotor
36e-3 grounded by the servo, shin 2.5e-3 through belt k_b) has NO k_b that
reproduces 11→14 Hz with √kp scaling — mode 1 caps at √(kp/(J_m+I_s))/2π =
4.4 Hz and mode 2 is kp-insensitive (checked at k_b = 11, 24, 661). Nor is it
a gain-frame error: if the tap kp had accidentally been motor-side (30 motor =
55.8 joint), J = 11.7e-3 — still 3.3× below trained nominal. The data require
the participating inertia to actually BE ~6.3e-3: numerically ≈ link 2.5e-3 +
~3.8e-3, and **3.8e-3 joint-side ≈ the PRE-P7 armature value 4.2e-3** that P7
retired (`rl_env_cfg_p6.py:667`). The P7/P8 "rotor ≈19e-3 motor-side →
36e-3 reflected" decomposition carried a documented commanded-τ scale
degeneracy (`rl_env_cfg_p6.py:976-979`, memory
`torque-delivery-ratio-measurement-trap`); the tap is torque-scale-free. The
opposing evidence (braced-ballistic jump + hip-clamped no-drop, 08-22) was
physical but high-acceleration/high-load — IF both are right, drivetrain
engagement is amplitude/load-gated (micro-backlash in gearbox or belt teeth),
a third regime-gated interface finding after the tooth skip and the telemetry
deadband.

Alternative sinks for the 2.5e-3→6.3e-3 link excess: real shank+foot heavier
than URDF (≈ +60 g at the foot would close it — H-COM-adjacent, feeds the §5
mass audit).

**Decisive 10-min test — MASS-ADDED TAP (torque- and kp-error-free absolute J):**
tape m = 100 g at measured lever l = 200 mm (ΔJ = m·l² = 4.0e-3), repeat the
kp=30 tap. J = ΔJ·f₂²/(f₁²−f₂²). Predictions: real-J ≈ 6.3e-3 → f drops
11 → ~8.4 Hz (−24%); hidden 36e-3 rotor participating → −5% barely. Optional
follow-ups if confirmed: large-amplitude tap (≥0.1 rad — engagement gating:
f falling toward 4.4 Hz = rotor re-engages), same protocol on the hip (its
33.2→38.5e-3 armature came from the same decomposition; servo design numbers
`rl_env_cfg_p6.py:662-664` inherit any correction). NOT yet registered as a
fall hypothesis — if the mass-added tap confirms, register (working name
H-armature-lowamp: policy trained expecting a 6× more sluggish low-load knee)
and add to the next-training directive.

## 7. H-tension-decay — belt tension as a maintenance-history sawtooth (registered 09-01)

Statement: T0 is not a constant — belt creep (measured: tensioner advance ≡
1–2% permanent elongation over ~5 months) decays tension between re-tension
events, so T0(t) is a sawtooth set by maintenance history. Today's pluck
(140 Hz ⇒ T0 = 88–129 N, §6.2) says nothing about T0 on any earlier flight
day. Detension threshold τ_kink = 2·T0·r_j scales linearly with T0, so every
tension level maps to a torque above which the slack span goes slack —
the tooth-skip precursor and a stiffness-halving nonlinearity.

### 7a. Tension timeline — mostly UNKNOWN, needs user's notes

Searched 09-01: git log (no tension/maintenance commits), docs/ grep (only §0's
set-screw description and the 08-21 coupling review's generic "belt remains a
compliance/failure element" note), data/ READMEs. **The repo contains ZERO
re-tension dates.** Known anchors only:
- belt in service ≈ 5 months (≈ 2026-04, from the creep statement);
- 09-01: pluck 140 Hz ⇒ T0 = 88–129 N (μ-band), 128 N at datasheet μ = 0.048.
- **FILL IN (user):** last re-tension date ___; was 08-25 before/after it ___;
  any earlier re-tension events ___. Until filled, §7c's maintenance
  correlation is untestable retroactively; §8's log fixes this forward.

### 7b. Detension threshold vs T0, and 08-25 skip plausibility

τ_kink = 2·T0·r_j = 0.0477·T0 [Nm, T0 in N]. Against MEASURED 08-25 knee
torques (all 8 flight trials in `data/p12_flight_20260825/`, `motor_tau_meas`
knees ×30/22; telemetry is 25–35 ms filtered + 0.15 Nm deadband → these are
LOWER bounds on true peaks, memory `motor-telemetry-not-plant`): per-trial
peaks **6.5–12.6 Nm joint-side** (campaign max 12.56 Nm, RIGHT knee — the leg
that skipped; p99.5 levels 6–12 Nm).

| T0 vs today | T0 (N) | τ_kink (Nm) | 08-25 peaks vs kink |
|---|---|---|---|
| 100% (today) | 88–129 | 4.2–6.2 | exceeded in EVERY flight trial |
| 75% | 66–97 | 3.2–4.6 | exceeded even at p99.5 sustained levels |
| 50% | 44–65 | 2.1–3.1 | crossed by moderate transients |
| 25% | 22–32 | 1.1–1.5 | the ≤1.5 Nm quasi-static balance window itself crosses |

**Verdict on 08-25:** the skip needs NO tension decay — today's tension already
puts every catch-step above the detension threshold (kink 4.2–6.2 vs peaks
7–12.6 Nm). H-tension-decay is therefore a RISK MODULATOR (skip frequency and
severity grow as T0 decays; at ≤50% the nonlinearity starts touching the
balance regime, and at 25% even quasi-static standing detensions), not a
necessary condition for the observed skip. Corollary: since kink < achievable
peaks at ANY maintainable tension (§8 shows 3× margin at 12.6 Nm needs
T0 ≈ 792 N ≥ the belt's ~780 N admissible), **hard-catch skips are a standing
design exposure** — mitigated procedurally (crawl-pose check between trials,
mandated since 08-25) or by redesign (wider belt / larger pulleys / idler
wrap), not by tension alone. Note the point-estimate elastic k_b ≈ 660 halves
to ≈330 post-kink — still above the §6.5 OOD boundaries (at the conservative
bound 181→90 it would dip below 120, another reason to keep peaks under kink).

### 7c. Rank for the 08-25 failures specifically

- Tooth skip on 08-25: **OBSERVED** — right knee, exactly 1 tooth = 12°
  joint-side, found by physical crawl-pose check (memory
  `knee-belt-coupling-findings`; npz step-detector hunt in
  `data/p12_flight_20260825/_e1a_skip_forensics.py`).
- **Post-skip trials: the belt family dominates.** A skip is a sustained
  0.209 rad knee obs-vs-joint offset — 3× the trained ±0.07 rad encoder-bias
  band (env.yaml:465), invisible to the motor-side encoder — a contract
  violation that outranks H-COM (a few-mm COM question) and H-contact for
  those trials outright.
- Pre-skip trials: unattributed by this hypothesis. H-COM stays open (§4/§5);
  H-contact for 08-25's grippy/taped mat: measured taped μ = 1.08 sits just
  inside the DR band edge 1.1 (§1) — disfavored, thin margin.
- H-tension-decay's UNIQUE prediction — failure severity correlates with
  time-since-maintenance (intermittent, right-knee-first, maintenance-reset) —
  is retroactively untestable: every flight campaign (08-13, 08-24, 08-25,
  08-28/29, 08-31) ended in falls, so the all-fall record carries no
  correlation signal, and skip surveillance (crawl checks) only exists since
  08-25. Testable PROSPECTIVELY only, via the §8 log.

### 7d. Hardware forensics → replace-vs-maintain decision matrix (user runs both)

| magnet on cords | weld/splice | reading | decision |
|---|---|---|---|
| magnetic (steel) | clean | creep is NOT cord stretch → PU/tooth shear or cord-PU adhesion micro-slip; elastic k_b high (tap ✓) | **MAINTAIN**: re-tension per §8, log dates; replace only if §8 shows creep rate NOT decelerating (adhesion slip progressing) |
| magnetic (steel) | necked / gapped / deformed teeth at one spot | creep localized at the splice = progressive failure point (skip nucleation, eventual snap) | **REPLACE BEFORE NEXT FLIGHT**; order true-endless (mold-endless) steel-cord, not welded |
| non-magnetic | any | belt is not steel-cord (mislabeled/wrong SKU): native creep, lower EA, larger stretch under peak loads | **REPLACE** with verified steel-cord; re-baseline pluck + tap after install |
| magnetic | clean, but §8 log later shows undiminished decay after a re-tension | cord-PU adhesion slip (cheap belt) | replace with brand-name belt (BRECO/Optibelt class) |

## 8. Belt maintenance appendix — 30-s pluck health check (adopted 09-01)

**Check (before every hardware session, ~30 s):** pluck the accessible free
span of EACH knee belt, read the fundamental with a phone tuner/spectrogram
app; log (date, f_L, f_R) in the session notes, plus a line for every
re-tension event (date, achieved f). Conversion at datasheet μ = 0.048 kg/m,
L_s = 0.1848 m: **T0 = (f/12.35)² N**; **τ_kink = 0.0477·T0 Nm**. A fixed-Hz
floor scales with the μ guess, but same-belt TRENDS are μ-free
(T ratio = f² ratio); on belt replacement, weigh it (μ = mass/0.5 m) and
rescale.

**Floors and what they buy** (f = 12.35·√T0):

| floor | T0 | τ_kink | margin |
|---|---|---|---|
| **140 Hz (ADOPTED)** | 128 N | 6.1 Nm | 3× over the ≤2 Nm balance window; ~1× at 6 Nm catches |
| 201 Hz | 264 N | 12.6 Nm | 1× at the measured 08-25 max catch (12.6 Nm, filtered → true peaks higher) |
| 348 Hz | 792 N | 37.8 Nm | 3× at max catch — **INFEASIBLE**: exceeds the ~780 N/10 mm admissible belt tension (and PETG pulley / bearing / PLA shaft loads long before) |

The requested "3× detension margin at max catch-step torque" floor does not
exist for this belt — no maintainable tension provides it (row 3). The adopted
floor 140 Hz = today's measured state: it preserves 3× margin over the
quasi-static fall window and keeps the kink above 6 Nm; catch-step detension
remains accepted risk covered by the crawl-pose check.

**Alarm rules:** (i) f < 140 Hz on either leg → re-tension before flying;
(ii) L/R asymmetry (f_hi/f_lo)² > 1.2 (>20% tension asymmetry) → re-tension
the low side (per-leg asymmetry is an axis no diagonal DR ever covered);
(iii) session-over-session drop > 10% in f (≈20% in T0) → creep active,
inspect weld (§7d) before continuing the campaign.
