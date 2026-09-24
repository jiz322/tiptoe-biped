"""imu_sign_check.py — physical verification of every IMU sign in config.py.

v2 (2026-08-28): adds an ACCELEROMETER GROUND-TRUTH layer. The raw acc is
untouched by the sign remap under test, so once the baseline and the
forward-tip phase establish the mounting (sensor axes vs robot axes), every
later phase is judged against acc-derived truth as well as against the
instructed maneuver. A sloppy maneuver then degrades the maneuver-based
verdict but NOT the acc-truth verdict. First run's lesson (see
data/_imu_sign_check_arbitrate*.py): phases 3 and 5 were done as
forward-tip / yaw-twist by mistake, yet the acc layer still produced
verdicts for all six signs.

The deployed path (IMUAdapter, raw pg — trim NOT applied) is tested against
the sim convention:

    frame: x forward, y LEFT, z up, right-handed
    tip forward        -> pg_x > 0
    tip left           -> pg_y > 0
    roll-right rate    -> ang_vel_x > 0
    pitch-forward rate -> ang_vel_y > 0
    yaw-left (CCW from above) -> ang_vel_z > 0 AND yaw angle increasing

NO motor traffic — IMU serial only. Robot unpowered, held/handled by hand.
All directions are in the ROBOT's frame (robot's forward, robot's left =
the side of its LEFT leg — if you face the robot head-on, that is YOUR
right).

Usage (on the Pi):
    python3 imu_sign_check.py            # guided check, prints verdict table
    python3 imu_sign_check.py --live     # freeform live display, ctrl-C to exit
    python3 imu_sign_check.py --save     # also dump per-phase npz

Assumptions of the acc-truth layer (checked at runtime): sensor z points UP
(baseline acc_z > 0), sensor axes right-handed (MEMS standard), phase 2's
forward tip done correctly (it and the baseline are the only phases that
MUST be performed right).
"""
import argparse
import math
import sys
import time

import numpy as np

import config as C
from imu_adapter import IMUAdapter


HZ = 100.0
GYRO_ACTIVE_THRESH = 0.15   # rad/s — samples counted as "in motion"
TILT_MIN = 0.25             # min |pg| swing for a static tilt (~15 deg)
CORR_MIN = 0.5              # min |correlation| for a rocking verdict
YAW_MIN = 0.5               # min |yaw delta| for the yaw test (~30 deg)

GYRO_SIGNS = np.array([C.IMU_GYRO_X_SIGN, C.IMU_GYRO_Y_SIGN, C.IMU_GYRO_Z_SIGN],
                      dtype=np.float64)


def record(imu, seconds, countdown=0):
    """Record sim-space pg_raw / ang_vel / yaw plus raw driver data."""
    if countdown:
        for k in range(countdown, 0, -1):
            print(f"    starting in {k} ...", end="\r")
            time.sleep(1.0)
        print(" " * 30, end="\r")
    n = int(seconds * HZ)
    pg = np.zeros((n, 3))
    w = np.zeros((n, 3))
    yaw = np.zeros(n)
    eul = np.zeros((n, 3))   # raw Yesense (pitch, roll, yaw) deg
    acc = np.zeros((n, 3))   # raw accel, IMU frame, m/s^2
    t_next = time.monotonic()
    for i in range(n):
        imu.update()
        pg[i] = imu.projected_gravity_raw()
        w[i] = imu.ang_vel_body()
        yaw[i] = imu.yaw()
        d = imu.driver.get_data()
        eul[i] = d['euler']
        acc[i] = d['acc']
        t_next += 1.0 / HZ
        dt = t_next - time.monotonic()
        if dt > 0:
            time.sleep(dt)
    return dict(pg=pg, w=w, yaw=np.unwrap(yaw), eul=eul, acc=acc)


def prompt(msg):
    ans = input(f"\n>>> {msg}\n    [enter]=go  s=skip  q=quit : ").strip().lower()
    if ans == 'q':
        sys.exit(0)
    return ans != 's'


def motion_corr(a, b, gate, thresh):
    """Correlation between a and b, gated to samples where |gate|>thresh."""
    mask = np.abs(gate) > thresh
    if mask.sum() < 50:
        return None, int(mask.sum())
    a, b = a[mask], b[mask]
    denom = a.std() * b.std()
    if denom < 1e-9:
        return None, int(mask.sum())
    return float(np.mean((a - a.mean()) * (b - b.mean())) / denom), int(mask.sum())


def dominant_axis(w):
    return int(np.argmax(w.std(axis=0)))


# ── accelerometer ground-truth layer ──────────────────────────
class AccTruth:
    """Maps raw sensor data to robot-frame truth once the mount is known.

    Mount model: sensor z parallel to robot z (up), sensor xy rotated by an
    arbitrary yaw vs the robot. `f`/`l` = robot forward/left expressed in
    sensor xy. Truth relations (a_u = unit proper acc = up-vector, g_raw =
    raw right-handed gyro):
        pg_true = (-a_u.f, -a_u.l, -a_u_z)
        w_true  = (g_xy.f,  g_xy.l,  g_z)
    """
    def __init__(self):
        self.a0 = None      # baseline acc unit (sensor frame)
        self.f = None       # robot fwd in sensor xy
        self.l = None       # robot left in sensor xy
        self.ok = False

    def set_baseline(self, rec):
        a = rec['acc'].mean(axis=0)
        self.a0 = a / np.linalg.norm(a)
        if self.a0[2] < 0.8:
            print("    !! baseline acc_z not ~+1g: sensor z is not 'up' — "
                  "acc-truth layer disabled (mount model doesn't apply).")
            self.a0 = None

    def set_forward(self, rec):
        if self.a0 is None:
            return
        a = rec['acc'].mean(axis=0)
        au = a / np.linalg.norm(a)
        d = au[:2] - self.a0[:2]
        if np.linalg.norm(d) < 0.15:
            print("    !! forward tip too small to fix the mount yaw — "
                  "acc-truth layer disabled.")
            return
        # tipping robot-forward drives proper-acc AWAY from robot fwd:
        # a_robot_x < 0  =>  fwd(sensor) = -tilt direction
        self.f = -d / np.linalg.norm(d)
        self.l = np.array([-self.f[1], self.f[0]])   # +90 deg CCW (z up shared)
        self.ok = True
        ang = math.degrees(math.atan2(self.f[1], self.f[0]))
        print(f"    [acc-truth armed] robot-fwd in sensor xy = "
              f"({self.f[0]:+.2f},{self.f[1]:+.2f})  (mount yaw ~ {ang:+.0f} deg)")

    def truth(self, rec):
        """Return (pg_true, w_true) in robot frame, or None."""
        if not self.ok:
            return None
        au = rec['acc'] / np.linalg.norm(rec['acc'], axis=1, keepdims=True)
        pg_t = np.column_stack([
            -(au[:, :2] @ self.f), -(au[:, :2] @ self.l), -au[:, 2]])
        g_raw = rec['w'] * GYRO_SIGNS          # undo remap -> raw rad/s
        w_t = np.column_stack([
            g_raw[:, :2] @ self.f, g_raw[:, :2] @ self.l, g_raw[:, 2]])
        return pg_t, w_t

    def tilt_dir_robot(self, rec):
        """Direction the robot is tilted toward, robot frame, degrees
        (0 = forward, +90 = left). For maneuver feedback."""
        t = self.truth(rec)
        if t is None:
            return None
        pg_t = t[0].mean(axis=0)
        if np.hypot(pg_t[0], pg_t[1]) < 0.10:
            return None
        return math.degrees(math.atan2(pg_t[1], pg_t[0]))


def acc_cross_check(at, rec, acc_verdicts):
    """Judge displayed pg/gyro against acc-derived robot-frame truth.
    Updates acc_verdicts, keeping the highest-|evidence| verdict per key."""
    t = at.truth(rec)
    if t is None:
        return
    pg_t, w_t = t
    pg, w = rec['pg'], rec['w']
    checks = [
        ("IMU_PITCH_SIGN",  pg[:, 0], pg_t[:, 0], pg_t[:, 0], 0.08),
        ("IMU_ROLL_SIGN",   pg[:, 1], pg_t[:, 1], pg_t[:, 1], 0.05),
        ("IMU_GYRO_X_SIGN", w[:, 0],  w_t[:, 0],  w_t[:, 0],  GYRO_ACTIVE_THRESH),
        ("IMU_GYRO_Y_SIGN", w[:, 1],  w_t[:, 1],  w_t[:, 1],  GYRO_ACTIVE_THRESH),
        ("IMU_GYRO_Z_SIGN", w[:, 2],  w_t[:, 2],  w_t[:, 2],  GYRO_ACTIVE_THRESH),
    ]
    for key, disp, truth, gate, thresh in checks:
        corr, n = motion_corr(disp, truth, gate, thresh)
        if corr is None or abs(corr) < CORR_MIN:
            continue
        strength = abs(corr) * min(n, 500)
        status = "OK" if corr > 0 else "FLIP"
        prev = acc_verdicts.get(key)
        if prev is None or strength > prev[2]:
            acc_verdicts[key] = (status, f"acc-truth corr {corr:+.2f} ({n} smp)",
                                 strength)
        print(f"      acc-truth {key:16s}: corr {corr:+.2f} ({n:4d} smp) -> {status}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    imu = IMUAdapter()
    imu.open()
    print("[imu_sign_check] warming up IMU stream 2 s ...")
    t0 = time.monotonic()
    while time.monotonic() - t0 < 2.0:
        imu.update()
        time.sleep(0.01)

    if args.live:
        print("live mode — sim-space values (pg RAW, no trim). ctrl-C to exit.")
        try:
            while True:
                imu.update()
                pg = imu.projected_gravity_raw()
                w = imu.ang_vel_body()
                d = imu.driver.get_data()
                print(f"pg=({pg[0]:+.3f},{pg[1]:+.3f},{pg[2]:+.3f})  "
                      f"w=({w[0]:+.2f},{w[1]:+.2f},{w[2]:+.2f})rad/s  "
                      f"yaw={math.degrees(imu.yaw()):+7.1f}deg  "
                      f"raw_eul(p,r,y)=({d['euler'][0]:+6.1f},{d['euler'][1]:+6.1f},"
                      f"{d['euler'][2]:+6.1f})  imu={imu.fps:5.1f}Hz", end="\r")
                time.sleep(0.05)
        except KeyboardInterrupt:
            print()
            return

    keys = ("IMU_PITCH_SIGN", "IMU_ROLL_SIGN", "IMU_YAW_SIGN",
            "IMU_GYRO_X_SIGN", "IMU_GYRO_Y_SIGN", "IMU_GYRO_Z_SIGN")
    verdicts = {k: ("SKIP", "not run") for k in keys}   # maneuver-based
    acc_verdicts = {}                                    # {key: (status, detail, strength)}
    at = AccTruth()
    recs = {}
    s_pitch = +1.0
    s_roll = +1.0
    gyro_bias = np.zeros(3)

    # ── 1. flat baseline ──────────────────────────────────────
    if prompt("Phase 1/6 BASELINE: robot torso LEVEL and still "
              "(e.g. flat on the bench). Recording 3 s."):
        r = record(imu, 3.0, countdown=2)
        recs['baseline'] = r
        at.set_baseline(r)
        pg_m, w_m = r['pg'].mean(axis=0), r['w'].mean(axis=0)
        gyro_bias = w_m.copy()
        print(f"    pg mean = ({pg_m[0]:+.3f},{pg_m[1]:+.3f},{pg_m[2]:+.3f})  "
              f"(want ~(0,0,-1))")
        print(f"    gyro mean = ({w_m[0]:+.3f},{w_m[1]:+.3f},{w_m[2]:+.3f}) rad/s "
              f"(want ~0)")
        if abs(pg_m[2] + 1.0) > 0.15:
            print("    !! pg_z far from -1 while level — mounting/axis problem, "
                  "fix before trusting anything below.")

    # ── 2. static tip forward → pg_x > 0 ─────────────────────
    if prompt("Phase 2/6 PITCH SIGN: tip the torso FORWARD (robot's front "
              "toward the floor) ~20-30 deg and HOLD still. This phase "
              "calibrates the acc-truth layer — do it carefully. "
              "Recording 3 s."):
        r = record(imu, 3.0, countdown=2)
        recs['tip_forward'] = r
        at.set_forward(r)
        pg_m = r['pg'].mean(axis=0)
        print(f"    pg mean = ({pg_m[0]:+.3f},{pg_m[1]:+.3f},{pg_m[2]:+.3f})")
        if abs(pg_m[1]) > abs(pg_m[0]):
            verdicts["IMU_PITCH_SIGN"] = ("SWAP?", f"tilt landed in pg_y ({pg_m[1]:+.3f}) "
                                          "not pg_x — euler axes swapped, not a sign fix")
        elif pg_m[0] > TILT_MIN:
            verdicts["IMU_PITCH_SIGN"] = ("OK", f"forward tip -> pg_x {pg_m[0]:+.3f}")
        elif pg_m[0] < -TILT_MIN:
            verdicts["IMU_PITCH_SIGN"] = ("FLIP", f"forward tip -> pg_x {pg_m[0]:+.3f} (want +)")
            s_pitch = -1.0
        else:
            verdicts["IMU_PITCH_SIGN"] = ("WEAK", f"pg_x {pg_m[0]:+.3f} too small — tilt more")
        acc_cross_check(at, r, acc_verdicts)

    # ── 3. static tip left → pg_y > 0 ─────────────────────────
    if prompt("Phase 3/6 ROLL SIGN: tilt the torso sideways toward the "
              "robot's LEFT (the side of its LEFT leg; facing the robot "
              "head-on that is YOUR right) ~20-30 deg and HOLD. Keep it "
              "facing the same direction — tilt like a metronome, do NOT "
              "twist it and do NOT tip it forward again. Recording 3 s."):
        r = record(imu, 3.0, countdown=2)
        recs['tip_left'] = r
        pg_m = r['pg'].mean(axis=0)
        print(f"    pg mean = ({pg_m[0]:+.3f},{pg_m[1]:+.3f},{pg_m[2]:+.3f})")
        tdir = at.tilt_dir_robot(r)
        if tdir is not None:
            print(f"    acc-truth says the robot is tilted toward {tdir:+.0f} deg "
                  f"(0=forward, +90=left) — want ~+90")
            if abs(tdir - 90.0) > 45.0:
                print("    !! that was not a LEFT tilt — maneuver-based verdict "
                      "unreliable; acc-truth verdict below still valid.")
        if abs(pg_m[0]) > abs(pg_m[1]):
            verdicts["IMU_ROLL_SIGN"] = ("SWAP?", f"tilt landed in pg_x ({pg_m[0]:+.3f}) "
                                         "not pg_y — wrong maneuver or axis swap")
        elif pg_m[1] > TILT_MIN:
            verdicts["IMU_ROLL_SIGN"] = ("OK", f"left tip -> pg_y {pg_m[1]:+.3f}")
        elif pg_m[1] < -TILT_MIN:
            verdicts["IMU_ROLL_SIGN"] = ("FLIP", f"left tip -> pg_y {pg_m[1]:+.3f} (want +)")
            s_roll = -1.0
        else:
            verdicts["IMU_ROLL_SIGN"] = ("WEAK", f"pg_y {pg_m[1]:+.3f} too small — tilt more")
        acc_cross_check(at, r, acc_verdicts)

    # ── 4. pitch rocking → gyro y ──────────────────────────────
    if prompt("Phase 4/6 GYRO-Y SIGN: rock the torso FORWARD and BACK "
              "continuously (~1 Hz, ~20 deg swings). Recording 5 s."):
        r = record(imu, 5.0, countdown=2)
        recs['pitch_rock'] = r
        w = r['w'] - gyro_bias
        dom = dominant_axis(w)
        dpg = np.gradient(r['pg'][:, 0]) * HZ
        corr, nact = motion_corr(dpg, w[:, 1], w[:, 1], GYRO_ACTIVE_THRESH)
        print(f"    dominant gyro axis = {'xyz'[dom]}  "
              f"corr(d(pg_x)/dt, w_y) = {corr if corr is None else round(corr, 2)} "
              f"({nact} active samples)")
        # pure pitch: d(pg_x)/dt = +cos*wy -> expected corr sign = s_pitch
        if dom != 1:
            verdicts["IMU_GYRO_Y_SIGN"] = ("SWAP?", f"pitch motion dominant on gyro "
                                           f"{'xyz'[dom]} — wrong maneuver or axis swap")
        elif corr is None or abs(corr) < CORR_MIN:
            verdicts["IMU_GYRO_Y_SIGN"] = ("WEAK", "motion too small/slow for a verdict")
        elif math.copysign(1.0, corr) == s_pitch:
            verdicts["IMU_GYRO_Y_SIGN"] = ("OK", f"corr {corr:+.2f}")
        else:
            verdicts["IMU_GYRO_Y_SIGN"] = ("FLIP", f"corr {corr:+.2f}, expected sign {s_pitch:+.0f}")
        acc_cross_check(at, r, acc_verdicts)

    # ── 5. roll rocking → gyro x ───────────────────────────────
    if prompt("Phase 5/6 GYRO-X SIGN: rock the torso sideways LEFT and RIGHT "
              "continuously (~1 Hz, ~20 deg swings) like a metronome. The "
              "robot keeps facing the same direction the whole time — do NOT "
              "twist it about the vertical axis. Recording 5 s."):
        r = record(imu, 5.0, countdown=2)
        recs['roll_rock'] = r
        w = r['w'] - gyro_bias
        dom = dominant_axis(w)
        dpg = np.gradient(r['pg'][:, 1]) * HZ
        corr, nact = motion_corr(dpg, w[:, 0], w[:, 0], GYRO_ACTIVE_THRESH)
        print(f"    dominant gyro axis = {'xyz'[dom]}  "
              f"corr(d(pg_y)/dt, w_x) = {corr if corr is None else round(corr, 2)} "
              f"({nact} active samples)")
        # pure roll: d(pg_y)/dt = -cos*wx -> expected corr sign = -s_roll
        if dom != 0:
            verdicts["IMU_GYRO_X_SIGN"] = ("SWAP?", f"roll motion dominant on gyro "
                                           f"{'xyz'[dom]} — if z: you twisted (yaw) "
                                           "instead of tilting; redo")
        elif corr is None or abs(corr) < CORR_MIN:
            verdicts["IMU_GYRO_X_SIGN"] = ("WEAK", "motion too small/slow for a verdict")
        elif math.copysign(1.0, corr) == -s_roll:
            verdicts["IMU_GYRO_X_SIGN"] = ("OK", f"corr {corr:+.2f}")
        else:
            verdicts["IMU_GYRO_X_SIGN"] = ("FLIP", f"corr {corr:+.2f}, expected sign {-s_roll:+.0f}")
        acc_cross_check(at, r, acc_verdicts)

    # ── 6. yaw CCW → yaw angle up AND gyro z > 0 ───────────────
    if prompt("Phase 6/6 YAW + GYRO-Z SIGNS: keep the torso level; rotate the "
              "whole robot about the VERTICAL axis ~90 deg to its LEFT "
              "(counter-clockwise seen from ABOVE), smoothly within the "
              "window, then hold. Recording 6 s."):
        r = record(imu, 6.0, countdown=2)
        recs['yaw_ccw'] = r
        w = r['w'] - gyro_bias
        dyaw = float(r['yaw'][-1] - r['yaw'][0])
        mask = np.abs(w[:, 2]) > GYRO_ACTIVE_THRESH
        wz_mean = float(w[mask, 2].mean()) if mask.sum() > 20 else 0.0
        dom = dominant_axis(w)
        print(f"    yaw delta = {math.degrees(dyaw):+.1f} deg  "
              f"gyro_z during motion = {wz_mean:+.2f} rad/s  "
              f"dominant gyro axis = {'xyz'[dom]}")
        # acc-truth can certify the rotation direction (raw right-handed
        # z-up gyro: raw g_z > 0 = CCW) even if the user turned the wrong way
        if at.ok:
            g_raw_z = (r['w'][:, 2]) * C.IMU_GYRO_Z_SIGN
            gmask = np.abs(g_raw_z) > GYRO_ACTIVE_THRESH
            if gmask.sum() > 20:
                ccw = float(g_raw_z[gmask].mean()) > 0
                print(f"    acc-truth: rotation was {'CCW' if ccw else 'CW'} "
                      f"(raw g_z {g_raw_z[gmask].mean():+.2f})")
                if not ccw:
                    print("    !! you rotated CW — maneuver-based verdicts below "
                          "are inverted; trust the acc-truth rows.")
        if abs(dyaw) < YAW_MIN:
            verdicts["IMU_YAW_SIGN"] = ("WEAK", f"yaw delta only {math.degrees(dyaw):+.1f} deg")
        elif dyaw > 0:
            verdicts["IMU_YAW_SIGN"] = ("OK", f"CCW -> yaw {math.degrees(dyaw):+.1f} deg")
        else:
            verdicts["IMU_YAW_SIGN"] = ("FLIP", f"CCW -> yaw {math.degrees(dyaw):+.1f} deg (want +)")
        if dom != 2:
            verdicts["IMU_GYRO_Z_SIGN"] = ("SWAP?", f"yaw motion dominant on gyro "
                                           f"{'xyz'[dom]} — gyro axes swapped")
        elif abs(wz_mean) < GYRO_ACTIVE_THRESH:
            verdicts["IMU_GYRO_Z_SIGN"] = ("WEAK", "rotation too slow for a verdict")
        elif wz_mean > 0:
            verdicts["IMU_GYRO_Z_SIGN"] = ("OK", f"CCW -> w_z {wz_mean:+.2f}")
        else:
            verdicts["IMU_GYRO_Z_SIGN"] = ("FLIP", f"CCW -> w_z {wz_mean:+.2f} (want +)")
        acc_cross_check(at, r, acc_verdicts)
        # yaw euler vs certified direction, through acc-truth
        if at.ok and abs(dyaw) > YAW_MIN:
            g_raw_z = (r['w'][:, 2]) * C.IMU_GYRO_Z_SIGN
            gmask = np.abs(g_raw_z) > GYRO_ACTIVE_THRESH
            if gmask.sum() > 20:
                ok = dyaw * float(g_raw_z[gmask].mean()) > 0
                acc_verdicts["IMU_YAW_SIGN"] = (
                    "OK" if ok else "FLIP",
                    f"yaw delta vs certified rotation dir", abs(dyaw) * 100)

    # ── bonus: euler-singularity probe over the standup pitch range ──
    if prompt("BONUS (optional) GIMBAL PROBE: slowly pitch the torso "
              "nose-DOWN to ~90 deg and back, then nose-UP to ~90 deg and "
              "back (~4 s each way). Watch for pg glitches near vertical. "
              "Recording 16 s."):
        r = record(imu, 16.0, countdown=2)
        recs['gimbal'] = r
        norm = np.linalg.norm(r['pg'], axis=1)
        jump = np.abs(np.diff(r['pg'], axis=0)).max()
        print(f"    |pg| range = [{norm.min():.3f}, {norm.max():.3f}] (want ~1.0)  "
              f"max one-step pg jump = {jump:.3f}")
        if norm.min() < 0.9 or norm.max() > 1.1 or jump > 0.15:
            print("    !! pg misbehaves near vertical pitch — euler gimbal "
                  "region matters for standup; inspect the npz trace.")

    # ── summary ────────────────────────────────────────────────
    current = {k: getattr(C, k) for k in keys}
    print("\n" + "=" * 76)
    print("VERDICTS   maneuver-based | acc-truth   (fix = flip where FLIP)")
    print("=" * 76)
    any_flip = False
    conflict = False
    for k in keys:
        status, detail = verdicts[k]
        astat, adetail = (acc_verdicts[k][0], acc_verdicts[k][1]) \
            if k in acc_verdicts else ("—", "")
        final = astat if astat in ("OK", "FLIP") else status
        if status in ("OK", "FLIP") and astat in ("OK", "FLIP") and status != astat:
            final = "CONFLICT"
            conflict = True
        cur = current[k]
        line = f"{k:18s} = {cur:+d}   {status:6s}| {astat:6s} {final:9s} {detail}"
        if adetail:
            line += f" | {adetail}"
        if final == "FLIP":
            line += f"   -> set {-cur:+d}"
            any_flip = True
        print(line)
    print("-" * 76)
    if conflict:
        print("CONFLICT rows: maneuver and acc-truth disagree — most likely the")
        print("maneuver was done in the wrong direction; redo that phase.")
    if any_flip:
        print("After editing config.py, RE-RUN this script to confirm, then")
        print("re-run obs_parity_check.py. IMU_PG_OFFSET is x-only, so a ROLL")
        print("flip does not invalidate the trim; a PITCH flip does.")
    elif not conflict and all(
            (verdicts[k][0] in ("OK", "SKIP")) and
            (k not in acc_verdicts or acc_verdicts[k][0] == "OK") for k in keys):
        print("All tested signs consistent with the sim convention.")

    if args.save and recs:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = f"imu_sign_check_{stamp}.npz"
        np.savez(out, **{f"{ph}_{k}": v for ph, r in recs.items()
                         for k, v in r.items()})
        print(f"saved {out}")

    imu.close()


if __name__ == "__main__":
    main()
