# IMU offset re-measured; the +0.089 trim is not an IMU bias (2026-09-20)

Session type: desk measurement, no flight. Robot torso rested upright on a floor
tile, motors off, `balance_log.py` IMU readout only. The readings are independent
of the loaded policy (s4a/s4b appear only because the script needs a policy path).

Trim convention (`balance_log.py`, checklist item 8 in CLAUDE.md):
`displayed pg_x = raw pg_x − pg_offset`. The policy sees `displayed`.

## 1. Measurements

| condition | pg-offset | displayed pg | raw pg_x | notes |
|---|---|---|---|---|
| no battery | 0.000 | (−0.005, +0.013, −1.000) | −0.005 | tile A |
| no battery | 0.000 | (−0.007, +0.013, −1.000) | −0.007 | tile B |
| no battery | +0.089 | (−0.094, +0.013, −1.000) | −0.005 | tile A |
| no battery | +0.089 | (−0.096, +0.013, −1.000) | −0.007 | tile B |
| battery in torso | 0.000 | (−0.013, +0.020, −1.000) | −0.013 | |
| battery in torso | 0.000 | (−0.012, +0.019, −1.000) | −0.012 | same spot, yawed 180° |
| battery in torso | +0.089 | (−0.101, +0.019, −1.000) | −0.012 | |
| battery in torso | +0.089 | (−0.103, +0.020, −1.000) | −0.014 | same spot, yawed 180° |
| battery in torso | −0.015 | (+0.001, +0.020, −1.000) | −0.014 | trim that zeroes x |

Angular velocity ≤ 0.002 rad/s in every row (robot at rest). Repeatability across
tiles and across a 180° yaw spin: ±0.001 in x, ±0.001 in y.

## 2. What the numbers establish

- **Raw IMU pitch bias with the deploy battery installed: −0.013 (0.7°).** Without
  the battery −0.006. The battery moves it by −0.007 (0.4°): either torso flex under
  the battery mass or seating of the IMU mount; small either way, and the
  battery-in value is the deploy-relevant one. The user saw no torso deformation.
- **The trim that zeroes the policy's pitch channel at the tile-level torso is
  `--pg-offset −0.015`**, not +0.089. This agrees with the 08-18 decomposition
  (mounting ≈ −0.032) in sign and order, and is tighter.
- **Roll bias: raw pg_y = +0.020 (1.1°)**, battery in. `IMU_PG_OFFSET[1]` is 0.
  Not a fall-backward candidate (falls are pitch-initiated, memory
  `fall-mechanism-pitch-initiates`); can be trimmed to +0.020 later, low priority.
- **The excess the plant demands: +0.089 − (−0.015) = +0.104 → the policy holds
  the torso pitched ~6.0° forward of tile-level** (asin 0.104) when it balances.
  This is the number every remaining hypothesis has to produce. Converted to a
  CoM-over-contact displacement at CoM height h above the ball contact:

  | h above contact | forward CoM shift needed |
  |---|---|
  | 0.30 m | 3.1 cm |
  | 0.35 m | 3.6 cm |

  This matches the independent 09-10 flight estimate ("plant balance point ~3 cm
  behind sim", memory `s4-flight-20260910-verdict`).

Assumption stated: the torso face resting on the tile is parallel to the URDF
torso frame. If it is not, the "IMU bias" measured here is really
IMU-vs-torso-frame bias plus torso-face-vs-frame angle, and the excess below
would partly be a torso-frame error. A digital angle gauge on the torso top
board vs a CAD face angle closes this in one reading (§5).

## 3. Hypothesis ledger for the +0.104 excess

| hypothesis | verdict today | discriminating evidence |
|---|---|---|
| IMU pitch misalignment (trim was "correct") | **contradicted** | tile test: raw bias −0.013, not −0.104 |
| Whole-body CoM behind the URDF CoM by ~3 cm | **contradicted** (model-level) | remassed Onshape CoM moved (−86.943, 0.021, −71.473) mm → (−86.114, 0.021, −78.566) mm: +0.8 mm along x (user: forward), −7.1 mm in z. Needs ~−30 mm; got +0.8 mm. |
| Joint zero error (hip_pitch and/or knee) putting the real foot ~3 cm forward of the modelled foot | **open, sized in §4** | none yet; 08-13 knife-edge is against it unless the zero has changed (§4.3) |
| Contact-point geometry (ball position on the real shin ≠ URDF part_3) | open, unsized | same signature as a knee zero error; a Kinovea photo separates them |

CoM caveats, so the branch is closed for the right reasons:
- The remass CoM is still a CAD number from part masses, not a physical CoM. A
  two-scale test (one scale under each end of a laid-flat robot, or a single hang
  point) costs nothing and gives the sagittal CoM to ~2 mm.
- The training asset `urdf_onshape_export/ballfeet-urdf/robot/robot.usd` still
  carries the OLD masses. Old vs remass link masses (kg): torso 1.698 → 1.8095,
  thigh 0.434 → 0.481, shin 0.2005 → 0.3104 (+55 %), total 5.20 → 5.62.
  `ballfeet-remass-urdf/` is untracked and nothing loads it. The static balance
  point barely moves (that is the CoM verdict above), but the shin mass change
  is a swing-leg inertia change and belongs in the next training asset.

## 4. The joint-zero hypothesis

### 4.1 Why it is credible mechanically

`control/config.py`: `ZERO_POSE_NAME = 'crawl'`, `JOINT_OFFSET` is derived from
`CRAWL_JOINT_POS = [0, 1.59, −0.2, 0, −1.59, 0.2]`, and `set_zero_all()` is called
with the robot hand-posed in the crawl. **The hardware joint zero is therefore
exactly as accurate as that hand placement.** The crawl reference was
sim-validated to 1.3° (runbook §2b), but that validation is a limp robot on a
near-frictionless floor settling under gravity; on the real mat (μ ≈ 1) the legs
do not slide to the settled pose, they stay where they were put. A few degrees
of systematic error per joint, the same on both legs, is the expected outcome of
this procedure, not a surprise. Every re-zero since 08-13 (each bench session,
and after the 08-25 tooth skip) re-rolled this error.

### 4.2 Size and direction required

Lever arms from `ballfeet-urdf/robot.urdf`: hip_pitch→knee 0.206 m, knee→ball
centre 0.199 m (shin-frame ball centre (−0.196, 0.0286, −0.018), path-mode report
§1). Required: real ball contact **forward** of the modelled contact by 3.1–3.6 cm
(or, equivalently, modelled contact behind the real one).

| error placed entirely in | zero error needed | equivalent |
|---|---|---|
| knee (both legs) | 0.16–0.18 rad = **9–10°** | 0.75–0.85 belt tooth (1 tooth = 12° joint-side) |
| hip_pitch (both legs) | 0.08–0.09 rad = **4.5–5°** | |
| one knee only | ~0.33 rad = 19° | ~1.6 teeth |

Direction, in pose terms (leg-1 signs; leg 2 mirrored): the real knee must be
**more extended** than the encoder claims, i.e. at zeroing the real crawl knee
flexion was smaller than 0.2 rad (shin placed nearly straight with the thigh:
the exact "correcting the knees toward 0" the 09-01 runbook note warns against),
and/or the real hip more flexed than 1.59 rad at zeroing (torso nose lifted
relative to the thighs while prone). Sign check to do by hand before trusting
this paragraph: flex the leg-1 knee toward negative and confirm the ball moves
toward the robot's back.

Note that the 09-10 sysid (`sysid_optuna.py`, searched knee/hip bias and CoM) did
not find such an offset, but it fitted dynamic fall trajectories dominated by the
policy dead zone; it is not a static test and does not rule this out.

### 4.3 The datum that has to be reconciled: the 08-13 knife-edge

08-13 (config.py comment block above `IMU_PG_OFFSET`): kp = 30 PD hold, repeated
hand releases, balance point at displayed +0.075 under trim −0.089 → raw
pg_x = −0.014, i.e. the torso tile-level at the static balance point, within
~1° of sim's frozen-pose balance point (0). A joint-zero error of the size in
§4.2 would have put that balance point at raw ≈ +0.09. Either the zero was
different on 08-13 (it was; the robot has been re-zeroed several times since),
or the +0.104 excess is not a static plant property at all and only appears
under the policy in closed loop. **That is the first thing to test, and it
costs nothing.**

### 4.4 Pre-registered test A: knife-edge repeat with the current zero (desk, ~20 min)

Hold the S4a idle posture (or the crawl-derived standing pose) with a stiff PD,
hand-release repeatedly, bracket the raw pg_x that separates forward from
backward falls. Log raw pg_x, hip_pitch and knee encoders at each release.

- Reads raw ≈ +0.09 to +0.10 → the excess is static plant geometry (joint zero
  and/or contact point). Go to test B/C to decompose.
- Reads raw ≈ −0.01 (tile-level, as on 08-13) → joint zero is NOT the cause;
  the +0.089 works through the policy's closed loop (its fixed point is flat in
  pitch and the bias moves it, memory `p12-3-blind-flight-verdict`). Then the
  fix is policy-side (bias band pinned around the deploy trim, reset-state DR
  with static lean; `next-training-directive`), and no angle gauge is needed.
- In between (+0.03 to +0.06) → both contribute; report the split.

### 4.5 Pre-registered test B: hard-stop / rest-pose encoder check (desk, free)

The sim's d = 1.0 rest state is fully determined by geometry: seiza with the
knee pinned on its −3.06 rad limit and the shin flat, hip 3.090 (row s = 0 of
`standup_path_knots`, rest columns). Put the real robot in that pose, motors
off, read the decoded joint angles through the normal bridge, and difference
against (hip 3.090, knee −3.060) per leg. A hard stop is a fixed geometric
reference; hand placement plays no role.

- |Δknee| ≥ 0.15 rad on both legs with the sign from §4.2 → H-joint-zero
  supported; re-zero at this pose instead of the crawl (`ZERO_POSE_NAME` is the
  one knob) and repeat test A.
- |Δ| ≤ 0.05 rad on both joints → the crawl zero is fine; contact-point geometry
  or closed-loop explanation remains.
- Requires that the physical fold actually reaches the same stop the sim
  reaches (thigh–shin mesh contact). If the real fold stops on a cable or cover
  first, note the gap and use test C.

### 4.6 Pre-registered test C: photo + Kinovea, plumb-line reference (desk, free)

Camera on a tripod ≥ 1.5 m away, perpendicular to the sagittal plane, robot held
in a fixed pose with encoders logged, a string plumb bob in frame. Mark hip-roll
axis, hip-pitch axis, knee axis, ball centre; compute thigh and shin angles from
the plumb line and compare to the forward kinematics of the logged encoders.
This measures the axis-to-axis lines that the URDF actually defines, so it
separates a knee zero error from a mis-placed ball on the shin. Expected
accuracy ~0.5°, well inside the 4.5–10° being looked for.

## 5. Buying a measurement device

**Cross-line laser level: not recommended.** It only projects a plumb/level line
in space; the link-to-line alignment is still judged by eye against link edges
that are not guaranteed parallel to the URDF axis-to-axis lines. Best case ~1°,
awkward on a robot that has to be held in pose, and its only real contribution,
a vertical reference in a photo, is what a string plumb bob gives for free.

**If anything is bought: a digital angle gauge / inclinometer cube** (magnetic
base, 0.1° resolution, ±0.2° accuracy, roughly $20–30). Set it on a flat face of
the torso, thigh and shin in a held pose; the differences are joint angles, and
Onshape gives each face's angle to its link's axis line. It also directly
checks the §2 assumption (torso resting face vs torso frame). This is the
standard cheap tool for this job; the standard non-cheap ones are a
calibration jig at a hard stop (what most legged-robot vendors ship) or
mocap-based kinematic calibration, neither needed here.

Order of operations: tests A and B today with nothing bought. Test A alone
decides whether joint zero is even the right branch. Test B, if A points
plant-side, yields a re-zero procedure that removes the hand-placement error
for good. Buy the gauge only if A and B disagree or B cannot reach the stop.

## 6. Open items carried forward

- Set `IMU_PG_OFFSET = [−0.015, +0.020]` as the physical trim in config, keeping
  the +0.089 flight value documented as a plant-gap compensator, not a trim.
- Physical CoM check by two-scale weighing (closes §3 at the hardware level).
- Rebuild the training asset from the remass URDF when the next line is
  launched (shin +55 % mass).
