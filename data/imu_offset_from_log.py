"""Estimate the IMU mounting offset from a hardware balance log. numpy only.

THE OBSERVABILITY ARGUMENT (why this is built the way it is):
a rotated IMU mount rotates the gyro by the same amount as the gravity
vector, so the attitude dynamics stay self-consistent -- the offset cancels
out of every IMU-only estimator (gyro/gravity consistency, integration
residuals, all of it). It is observable ONLY against an IMU-independent
reference. On this robot that reference is the joint encoders: a
self-supporting point-foot biped must hold its CoM over the contact patch,
so the leg configuration pins the torso attitude.

This also side-steps the circularity that defeats "compare mean pg to sim":
the policy servos the OBSERVED attitude to what it believes is upright, so a
biased IMU makes the robot physically lean by about -offset. That behavioural
response contaminates any IMU-vs-IMU comparison, but it shows up honestly in
the joint angles, which are measured on a separate channel.

    offset = mean( pg_measured - f(joint_pos) )

with f learned from sim (exact kinematics + masses) by _imu_map_collect.py
over a range of forced obs biases, so f covers the leaning regime hardware
actually visits rather than only the upright one.

Modes:
  --fit  <samples.npz>          fit f, leave-one-bias-out CV, write map json
  --self-test <samples.npz>     inject a known offset, verify recovery
  --apply <hw_log.npz>          estimate the offset from a hardware log
"""
import argparse, json, os
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--fit", metavar="SAMPLES_NPZ")
p.add_argument("--self-test", metavar="SAMPLES_NPZ")
p.add_argument("--apply", metavar="HW_LOG_NPZ")
p.add_argument("--map", default="data/imu_map.json")
p.add_argument("--ridge", type=float, default=1e-4)
p.add_argument("--vel-max", type=float, default=0.6,
               help="quasi-static gate on |joint_vel| (rad/s)")
p.add_argument("--angvel-max", type=float, default=0.5,
               help="quasi-static gate on |ang_vel| (rad/s)")
p.add_argument("--upright-min", type=float, default=0.85,
               help="gate on -pg_z: only self-supporting ticks")
p.add_argument("--inject", type=float, default=-0.09,
               help="--self-test: true offset to inject on pg_x")
args = p.parse_args()

JN = ["hip_roll1", "hip_pitch1", "knee1", "hip_roll2", "hip_pitch2", "knee2"]


def design(q):
    """Feature map for joint_pos -> attitude. Quadratic + per-leg
    pitch*knee products: the torso lean is set by where the CoM sits, which
    is trigonometric in the leg chain -- a quadratic captures the local
    curvature over the range the policy visits without inviting the
    wild extrapolation a higher order would."""
    q = np.atleast_2d(q)
    return np.column_stack([
        np.ones(len(q)), q, q ** 2,
        q[:, 1] * q[:, 2], q[:, 4] * q[:, 5],
        q[:, 1] * q[:, 4], q[:, 0] * q[:, 3],
    ])


def ridge_fit(X, Y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    sd[0] = 1.0
    mu[0] = 0.0
    Z = (X - mu) / sd
    A = Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1])
    W = np.linalg.solve(A, Z.T @ Y)
    return {"W": W, "mu": mu, "sd": sd}


def predict(m, q):
    Z = (design(q) - m["mu"]) / m["sd"]
    return Z @ m["W"]


def gate(jvel, angvel, pgz):
    return ((np.abs(jvel).max(1) < args.vel_max)
            & (np.abs(angvel).max(1) < args.angvel_max)
            & (-pgz > args.upright_min))


def block_se(r, block=50):
    """Standard error of mean(r) under serial correlation: batch-means over
    contiguous blocks. Per-tick residuals at 100 Hz are heavily correlated,
    so the naive sd/sqrt(n) would overstate precision by ~an order."""
    nb = max(2, len(r) // block)
    bm = np.array([b.mean() for b in np.array_split(r, nb)])
    return bm.std(ddof=1) / np.sqrt(nb)


def load_samples(path):
    d = np.load(path)
    return (d["jpos"], d["jvel"], d["pg_true"], d["angvel"], d["bias"])


def do_fit(path, verbose=True):
    jp, jv, pg, av, bias = load_samples(path)
    X, Y = design(jp), pg[:, :2]
    groups = np.unique(bias)
    if verbose:
        print(f"\n{'='*66}\n  fit joint_pos -> torso attitude   n={len(jp)}"
              f"  biases={list(groups)}\n{'='*66}")
        print("  leave-one-bias-out CV (tests extrapolation into unseen lean)")
        print(f"  {'held-out':>9} {'n':>7} {'resid_sd_x':>11} "
              f"{'mean_err_x':>11} {'mean_err_y':>11}")
    worst = 0.0
    for g in groups:
        te = bias == g
        m = ridge_fit(X[~te], Y[~te], args.ridge)
        r = Y[te] - predict(m, jp[te])
        worst = max(worst, abs(r[:, 0].mean()))
        if verbose:
            print(f"  {g:>+9.3f} {te.sum():>7d} {r[:,0].std():>11.4f} "
                  f"{r[:,0].mean():>+11.4f} {r[:,1].mean():>+11.4f}")
    m = ridge_fit(X, Y, args.ridge)
    r = Y - predict(m, jp)
    if verbose:
        print(f"\n  full-fit in-sample resid sd: x {r[:,0].std():.4f}"
              f"  y {r[:,1].std():.4f}")
        print(f"  worst held-out mean error (x): {worst:.4f}")
        verdict = ("RESOLVABLE" if worst < 0.03 else
                   "MARGINAL" if worst < 0.06 else "NOT RESOLVABLE")
        print(f"  -> a -0.09 offset is {verdict} by this estimator "
              f"(need worst-case mean error << 0.09)")
    return m, worst
def save_map(m, path):
    json.dump({k: v.tolist() for k, v in m.items()}, open(path, "w"))
    print(f"  map -> {path}")


def load_map(path):
    d = json.load(open(path))
    return {k: np.asarray(v) for k, v in d.items()}


def do_self_test(path):
    """Hold out one bias group, build a synthetic 'hardware' log from it with
    a known injected offset, and check the estimator recovers it."""
    jp, jv, pg, av, bias = load_samples(path)
    groups = np.unique(bias)
    held = groups[np.argmin(np.abs(groups - args.inject))]
    te = bias == held
    X, Y = design(jp), pg[:, :2]
    m = ridge_fit(X[~te], Y[~te], args.ridge)
    print(f"\n{'='*66}\n  SELF-TEST  held-out bias {held:+.3f}, "
          f"injected offset {args.inject:+.4f}\n{'='*66}")
    pg_meas = pg[te].copy()
    pg_meas[:, 0] += args.inject
    pred = predict(m, jp[te])
    r = pg_meas[:, :2] - pred
    est = r.mean(0)
    se = np.array([block_se(r[:, 0]), block_se(r[:, 1])])
    err = est[0] - args.inject
    print(f"  recovered x {est[0]:+.4f} +- {1.96*se[0]:.4f} (95%)   "
          f"truth {args.inject:+.4f}   error {err:+.4f}")
    print(f"  recovered y {est[1]:+.4f} +- {1.96*se[1]:.4f} (95%)   "
          f"truth  0.0000")
    ok = abs(err) < 0.03
    print(f"  -> {'PASS' if ok else 'FAIL'} (need |error| < 0.03 to call a "
          f"-0.09 hardware reading real)")
    return ok


def do_apply(path, m):
    d = np.load(path, allow_pickle=True)
    jp = d["sim_joint_pos"]
    jv = d["sim_joint_vel"]
    pg = d["proj_grav"]
    av = d["ang_vel"]
    mode = np.asarray(d["mode"]).astype(str)
    t = d["time"]
    nn = np.array(["NN" in s.upper() or "CONTROL" in s.upper() for s in mode])
    g = gate(jv, av, pg[:, 2]) & nn
    print(f"\n{'='*66}\n  apply  {os.path.basename(path)}\n{'='*66}")
    print(f"  ticks {len(jp)}   NN-control {nn.sum()}   "
          f"quasi-static & self-supporting & NN: {g.sum()}")
    if g.sum() < 40:
        print("  !! too few usable ticks. The robot must be BALANCING on its "
              "own feet (not hand-held, not fallen) for the CoM-over-contact "
              "physics to hold. Need >=40 ticks; a single 1 s balance at "
              "100 Hz gives 100.")
        return
    r = pg[g][:, :2] - predict(m, jp[g])
    est, se = r.mean(0), np.array([block_se(r[:, 0]), block_se(r[:, 1])])
    print(f"  window {t[g].min():.2f}..{t[g].max():.2f} s")
    print(f"\n  imu_offset_x = {est[0]:+.4f}  +- {1.96*se[0]:.4f} (95% CI)")
    print(f"  imu_offset_y = {est[1]:+.4f}  +- {1.96*se[1]:.4f} (95% CI)")
    print(f"  measured pg mean  x {pg[g][:,0].mean():+.4f}"
          f"  y {pg[g][:,1].mean():+.4f}")
    print(f"  kinematic pred    x {predict(m, jp[g])[:,0].mean():+.4f}"
          f"  y {predict(m, jp[g])[:,1].mean():+.4f}")
    sig = abs(est[0]) > 1.96 * se[0]
    print(f"\n  x offset is {'SIGNIFICANT' if sig else 'NOT significant'} "
          f"vs its own CI")
    print("  interpretation:")
    print("    |offset| < 0.03  -> mount is fine; the launch lean is real "
          "posture, and the fix is the widened DR band, not a constant.")
    print("    |offset| > 0.05  -> real mounting/deformation error; subtract "
          "it in imu_adapter.projected_gravity() (no retrain needed).")
    print("  cross-check: re-run on a second log. A mounting rotation is "
          "constant across sessions; posture is not.")


if args.fit:
    m, _ = do_fit(args.fit)
    save_map(m, args.map)
if args.self_test:
    do_self_test(args.self_test)
if args.apply:
    do_apply(args.apply, load_map(args.map))
if not (args.fit or args.self_test or args.apply):
    p.error("pick one of --fit / --self-test / --apply")

