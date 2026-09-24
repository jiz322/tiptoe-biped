"""
analyze_equilibrium_fit.py — estimate the TRUE balance point from fall dynamics.

Model: near upright, point-foot sagittal dynamics ~ inverted pendulum:
    pg_x_ddot = (g/l) * (pg_x - pg_star)
Regress pg_ddot on pg_x over the pre-fall window of each run:
    slope  -> g/l  (sanity: l = g/slope should be ~0.3-0.5 m)
    pg_star = -intercept/slope  (estimated equilibrium in DISPLAYED pg units)
Knife-edge measured +0.075. Policy believes 0.
"""
import glob, os
import numpy as np

DATA = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(os.path.join(DATA, "policy_G2i4760_log_20260813_*.npz")))

stars, ls, r2s = [], [], []
for f in files:
    d = np.load(f, allow_pickle=True)
    t = d['time']; n = len(t)
    if n < 20:
        continue
    pg = d['proj_grav'][:, 0]
    tilted = np.abs(d['proj_grav'][:, 0]) > 0.3
    fall_i = int(np.argmax(tilted)) if tilted.any() else n - 1
    # use release -> fall onset; smooth pg lightly before double-diff
    w_end = max(fall_i, 15)
    x = pg[:w_end].astype(np.float64)
    tt = t[:w_end].astype(np.float64)
    # uniform-ish 10 ms ticks; smooth with 5-tap box then central 2nd diff
    k = np.ones(5) / 5
    xs = np.convolve(x, k, mode='same')
    dt = np.median(np.diff(tt))
    xdd = np.gradient(np.gradient(xs, dt), dt)
    # drop edges (filter + gradient artifacts)
    sl = slice(4, len(x) - 4)
    X, Y = xs[sl], xdd[sl]
    if len(X) < 8:
        continue
    A = np.vstack([X, np.ones_like(X)]).T
    (a, b), res, *_ = np.linalg.lstsq(A, Y, rcond=None)
    if a <= 0:
        stars.append(np.nan); ls.append(np.nan); r2s.append(np.nan)
        continue
    pg_star = -b / a
    l_eff = 9.81 / a
    ss_res = np.sum((Y - A @ [a, b]) ** 2)
    ss_tot = np.sum((Y - Y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    stars.append(pg_star); ls.append(l_eff); r2s.append(r2)
    print(f"{os.path.basename(f)[-10:-4]}  pg* = {pg_star:+.4f}   l_eff = {l_eff:5.2f} m   R2 = {r2:.2f}")

stars = np.array(stars); ls = np.array(ls); r2s = np.array(r2s)
ok = ~np.isnan(stars)
# weight by fit quality, keep physically sane pendulum lengths
sane = ok & (ls > 0.1) & (ls < 2.0) & (r2s > 0.3)
print()
print(f"valid fits: {ok.sum()}/{len(stars)}, physically sane (0.1<l<2, R2>0.3): {sane.sum()}")
if sane.sum():
    print(f"pg* over sane fits: mean {stars[sane].mean():+.4f}  median {np.median(stars[sane]):+.4f} "
          f" +- {stars[sane].std():.4f}")
    print(f"l_eff: median {np.median(ls[sane]):.2f} m")
print()
print("reference: knife-edge balance point +0.075; policy equilibrium ~0")
