"""
analyze_fall_direction.py — is the fall consistent with the +0.075 static
balance-point offset (knife-edge, 08-13)?  Check per run:
  - pg_x at release (t=0), at 0.1s, sign at fall onset
  - is the fall always the same sign?
  - hip-pitch response direction: does hip pitch target move against the tilt?
  - where does pg_x sit while "quiet" (policy's believed equilibrium)?
"""
import glob, os
import numpy as np

DATA = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(os.path.join(DATA, "policy_G2i4760_log_20260813_*.npz")))

rows = []
for f in files:
    d = np.load(f, allow_pickle=True)
    t = d['time']; n = len(t)
    if n < 10:
        continue
    pg = d['proj_grav']
    obs = d['obs']
    act = d['action']
    tgt = d['target']
    tilted = np.abs(pg[:, 0]) > 0.3
    fall_i = int(np.argmax(tilted)) if tilted.any() else n - 1

    i01 = min(10, n - 1)   # 0.1 s
    i16 = min(16, n - 1)   # 0.16 s — pitch initiation per prior forensics
    pgx0 = pg[0, 0]
    pgx01 = pg[i01, 0]
    pgx_fall_sign = np.sign(pg[fall_i, 0])
    # mean hip-pitch action (L=idx1 in act? action is 6-dim sim order) in
    # first 0.15 s vs at fall
    hp_act_early = act[:i16, [1, 4]].mean(axis=0)
    hp_act_fall = act[max(0, fall_i - 3):fall_i + 1, [1, 4]].mean(axis=0)
    rows.append((os.path.basename(f)[-10:-4], n, t[fall_i], pgx0, pgx01,
                 pgx_fall_sign, hp_act_early, hp_act_fall,
                 pg[:i16, 0].mean()))

print(f"{'run':>7s} {'fall_s':>6s} {'pgx@0':>7s} {'pgx@.1':>7s} {'sign@fall':>9s} "
      f"{'pgx mean<0.16':>13s}  {'hp_act early L/R':>18s}  {'hp_act @fall L/R':>18s}")
signs = []
for r in rows:
    name, n, ft, p0, p01, sgn, ae, af, pmean = r
    signs.append(sgn)
    print(f"{name:>7s} {ft:>6.2f} {p0:>+7.3f} {p01:>+7.3f} {sgn:>+9.0f} "
          f"{pmean:>+13.3f}  {ae[0]:>+8.3f}/{ae[1]:>+8.3f}  {af[0]:>+8.3f}/{af[1]:>+8.3f}")

signs = np.array(signs)
print()
print(f"fall sign: {np.sum(signs > 0)} positive, {np.sum(signs < 0)} negative of {len(signs)}")
p0s = np.array([r[3] for r in rows]); pms = np.array([r[8] for r in rows])
print(f"pg_x at release: mean {p0s.mean():+.4f} +- {p0s.std():.4f}")
print(f"pg_x mean over first 0.16s: mean {pms.mean():+.4f} +- {pms.std():.4f}")
print()
print("Knife-edge balance point (displayed, 08-13): pg_x = +0.075")
print("=> gap between where policy sits and true balance point:",
      f"{0.075 - pms.mean():+.4f}")
