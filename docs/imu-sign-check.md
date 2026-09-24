# IMU sign check — procedure & rationale (2026-08-28)

## RESULT of run 1 (2026-08-28, npz `data/imu_sign_check_20260828_163528.npz`)

Two phases were performed wrong (tip-left was a second forward tip, acc tilt
directions cos = +0.96; roll-rock was a yaw twist, raw euler 44.8° yaw vs
7.9° roll), but raw-accelerometer arbitration
(`data/_imu_sign_check_arbitrate.py`, `_imu_sign_check_arbitrate2.py`)
produced verdicts for **all six signs** anyway — the acc is untouched by the
remap, and the forward-tip phase pins the mounting: **sensor x points
BACKWARD, z up (180° mount about z)**. Method self-validated on the two
known-good channels (pitch +0.98, gyro-y +0.90) and independently certified
the yaw maneuver as genuinely CCW (raw right-handed gyro-z +0.20 rad/s).

| Constant          | Current | Verdict | Evidence |
|-------------------|---------|---------|----------|
| `IMU_PITCH_SIGN`  | −1      | OK      | corr(pg_x, a_sx) = +0.98, 1692 smp |
| `IMU_ROLL_SIGN`   | +1      | **→ −1**| corr(pg_y, a_sy) = −0.77, 1239 smp |
| `IMU_YAW_SIGN`    | +1      | OK      | certified CCW → yaw +40.5° |
| `IMU_GYRO_X_SIGN` | +1      | **→ −1**| corr(w_x, −d(a_sy)/dt) = −0.85, 548 smp |
| `IMU_GYRO_Y_SIGN` | −1      | OK      | corr(w_y, +d(a_sx)/dt) = +0.90, 914 smp |
| `IMU_GYRO_Z_SIGN` | −1      | **→ +1**| certified CCW → w_z −0.20 (want +) |

**The flipped set {pg_y, w_x, w_z} is exactly a left-right mirror** — the
current config implements a reflection of the true frame, not a rotation
(the old balancer's signs missed the three lateral corrections of the 180°
mount). Consequences:

- Every hardware flight/recording to date fed the policy **mirrored lateral
  IMU channels** (pg_y, ωx, ωz), with joint obs/actions unmirrored.
- This flip-set is identical to the sim2real y-mirror transform in
  `rl_env_p6.py` — prime suspect for the open sim-steps-RIGHT /
  hw-steps-LEFT gait-phase flip, and plausibly for the one-signed
  downstream roll in falls and the 5–7 Hz hand-hold thrash (wrong-handed
  lateral feedback loop, excited when grip adds stiffness).
- `IMU_PG_OFFSET` is x-only → the −0.089 trim **survives** the roll fix.
- Pitch channels (the primary fall axis) were correct all along — the
  G2i-era pitch plant gap is a separate, still-real issue.

**Status: flips APPLIED 2026-08-28** (`IMU_ROLL_SIGN` −1, `IMU_GYRO_X_SIGN`
−1, `IMU_GYRO_Z_SIGN` +1). The user physically verified the 180°-about-z
mount, which together with the acc-truth analysis pins all three flips; the
confirmation re-run was skipped. At the next bench session: re-run this
check (expect all-OK / all-green acc-truth column), then
`obs_parity_check.py`. Expect changed lateral behavior on hardware — re-run
the P11 desk gates before any flight, and re-test the first-step side (the
gait-phase flip should disappear if the mirror was its cause).

## Why

The six sign constants in `control/config.py` (`IMU_PITCH_SIGN`, `IMU_ROLL_SIGN`,
`IMU_YAW_SIGN`, `IMU_GYRO_{X,Y,Z}_SIGN`) were inherited from the pre-biped
balancer and have **never been physically verified** on this robot. The
`tools/test_imu.py` the config comment referenced never existed.

Why nothing has caught a sign error so far:

- The 08-14 obs-contract audit checked channels were *in-distribution* — a
  sign flip on a zero-mean channel (any gyro axis, yaw error at handover) is
  distribution-symmetric and invisible to it.
- `spawn_yaw` is latched at NN handover, so `yaw_err` always *starts* at 0,
  and every flight so far ran `yaw_rate_cmd = 0`. The whole yaw pathway is
  effectively untested on hardware.
- Roll/pitch pg components are strongly exercised (balance works at all), so
  those are *probably* right — but "probably" is what this check retires.

**Known suspect found by inspection:** `IMU_YAW_SIGN = +1` while
`IMU_GYRO_Z_SIGN = -1`. In a consistent right-handed remap the yaw angle and
yaw rate flip together; at most one of these is correct — unless the Yesense
reports compass-style (clockwise-positive) heading alongside a right-handed
gyro, which is exactly what the check distinguishes. Consequence if `w_z` is
the wrong one: the policy's trained yaw damping (`yaw_drift` reward term)
becomes positive feedback on hardware — a spin-up tendency, plausibly
masked so far by short grounded times.

## Ground truth (sim frame)

Isaac frame: **x forward, y left, z up**, right-handed. Verified against
`quat_apply_inverse(imu_quat, (0,0,-1))` in `rl_env_p6.py` and the
sim2real mirror table (flips ωx/ωz/pg_y — a y-mirror in exactly this frame).

| Physical maneuver (robot frame)  | Expected reading            |
|----------------------------------|-----------------------------|
| Tip torso forward                | `pg_x > 0`                  |
| Tip torso left                   | `pg_y > 0`                  |
| Roll-right rate                  | `ang_vel_x > 0`             |
| Pitch-forward rate               | `ang_vel_y > 0`             |
| Yaw left (CCW seen from above)   | `ang_vel_z > 0`, yaw ↑      |
| Level                            | `pg ≈ (0, 0, -1)`           |

Useful decoupling: `pg` is yaw-independent (gravity is z-aligned), so the
roll/pitch tests can't be contaminated by a yaw sign error and vice versa.

## Procedure

Bench, motors **unpowered**, no motor traffic at all (IMU serial only). Robot
handled by hand. On the Pi:

```
cd ~/Documents/big_thigh/control
python3 imu_sign_check.py --save
```

Guided phases (skippable individually, ~2 min total):

1. **Baseline** — level & still. Sanity: `pg ≈ (0,0,-1)`, gyro ≈ 0. Also
   captures gyro bias used by later phases.
2. **Tip forward, hold** → pins `IMU_PITCH_SIGN` (`pg_x > 0`).
3. **Tip left, hold** → pins `IMU_ROLL_SIGN` (`pg_y > 0`).
4. **Rock forward/back ~1 Hz** → pins `IMU_GYRO_Y_SIGN` via
   `corr(d(pg_x)/dt, w_y)` (expectation conditioned on phase-2 outcome, so
   it stays valid even if pitch sign was flipped).
5. **Rock left/right ~1 Hz** → pins `IMU_GYRO_X_SIGN` via
   `corr(d(pg_y)/dt, w_x)` (expected negative when both are correct).
6. **Rotate ~90° CCW (from above), hold** → pins `IMU_YAW_SIGN` (Δyaw > 0)
   and `IMU_GYRO_Z_SIGN` (`w_z > 0` during motion) independently.
7. **Bonus gimbal probe** — slow pitch to ±90° (the standup range): watch
   for euler-singularity glitches in `pg` (|pg| off 1.0, step jumps). The
   Yesense euler→quat path must stay smooth through the kneel/crawl pitch
   range or standup obs are corrupted near vertical.

Each phase prints numbers live; the end summary prints a per-constant verdict
(`OK` / `FLIP` → suggested new value / `SWAP?` = axis swap, not a sign fix /
`WEAK` = redo with bigger motion). `--live` gives a freeform display for
eyeballing. Uses `projected_gravity_raw()` — **trim not applied**.

## If a sign flips

- Edit `config.py`, re-run the check to confirm, then re-run
  `obs_parity_check.py`.
- A pitch or roll flip **invalidates `IMU_PG_OFFSET`** (trim was tuned under
  the old signs — the -0.089 story restarts) and invalidates the crawl/supine
  mounting decomposition from 08-18.
- A `w_z` or yaw flip does *not* touch the trim, but re-run the P11 §0.x desk
  gates before flying yaw commands.
- Any flip means all prior hardware npz recordings carry the old sign —
  annotate before comparing old traces to new.
