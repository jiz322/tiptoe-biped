#!/usr/bin/env python3
"""
kp_sweep_fit.py — Replant M4.1 closure: fit each joint's kp-triplet of
pos_step_test captures jointly, comparing three loop models.

Empirical driver: measured ring frequency grows SLOWER than sqrt(kp)
and zeta falls with kp, on all three joints. A rigid inertia behind a
proportional loop cannot do either. Candidate mechanisms:

    rigid    tau = kp (u - q)                     (baseline, must lose)
    fbfilt   tau = kp (u - q_f),  q_f' = w_f (q - q_f)
             one-pole low-pass in the firmware position feedback
    fbdelay  tau = kp (u - q(t - t_l))
             pure extra loop delay (beyond the 2.36 ms M1.1 transport)

All three share the plant  I q'' = tau - b q' - C tanh(q'/eps)
- mgl sin(q - q_v), command dead-time fixed at the measured M1.1 value.
Plant params are shared across the triplet; only kp differs per file —
that coupling is what makes the models distinguishable.

--selftest simulates a known fbdelay truth at three kp and verifies the
fit recovers it and prefers the right model before real data is trusted.

Usage:
    python kp_sweep_fit.py --selftest
    python kp_sweep_fit.py --joint 101
    python kp_sweep_fit.py --joint 103 --files data/step_103_kp3.npz ...
"""

import argparse
import glob

import numpy as np
from scipy.optimize import least_squares

T_D_M11 = 0.00236       # M1.1 transport dead-time, s
FS_SIM = 4000.0
EPS_V = 0.01
DECIM = 8

MODELS = ('rigid', 'fbfilt', 'fbdelay')


def simulate(theta, reps, model, fs=FS_SIM):
    I, mgl, q_v, b, C = theta[:5]
    extra = theta[5] if len(theta) > 5 else None
    n = len(reps)
    q0 = np.array([r['q0'] for r in reps])
    kp = np.array([r['kp'] for r in reps])
    tgt = np.array([r['target'] for r in reps])
    q = q0.copy()
    v = np.zeros(n)
    qf = q0.copy()
    dt = 1.0 / fs
    t_max = max(r['t'][-1] for r in reps)
    nstep = int(t_max * fs) + 2
    out = np.empty((nstep, n))
    for k in range(nstep):
        out[k] = q
        u = np.where(k * dt >= T_D_M11, tgt, q0)
        if model == 'fbfilt':
            qf = qf + extra * (q - qf) * dt
            q_meas = qf
        elif model == 'fbdelay':
            # fractional delay via linear interp so t_l stays differentiable
            idx = k - extra * fs
            if idx <= 0.0:
                q_meas = q0
            else:
                i0 = int(idx)
                frac = idx - i0
                q_meas = out[i0] * (1.0 - frac) + out[min(i0 + 1, k)] * frac
        else:
            q_meas = q
        tau = kp * (u - q_meas)
        acc = (tau - b * v - C * np.tanh(v / EPS_V)
               - mgl * np.sin(q - q_v)) / I
        v = v + acc * dt
        q = q + v * dt
    grid = np.arange(nstep) * dt
    return [np.interp(r['t'], grid, out[:, i]) for i, r in enumerate(reps)]


def residuals(theta, reps, model):
    sim = simulate(theta, reps, model)
    return np.concatenate([(s - r['p']) / r['step'] for r, s in zip(reps, sim)])


def fit(reps, model):
    q0_med = float(np.median([r['q0'] for r in reps]))
    x0 = [20e-3, 0.5, q0_med, 0.05, 0.10]
    lo = [1e-3, 0.0, q0_med - 3.2, 0.0, 0.0]
    hi = [500e-3, 30.0, q0_med + 3.2, 2.0, 2.0]
    if model == 'fbfilt':
        x0 += [60.0]; lo += [2.0]; hi += [2000.0]
    elif model == 'fbdelay':
        x0 += [0.020]; lo += [0.0]; hi += [0.080]
    return least_squares(residuals, x0, bounds=(lo, hi), args=(reps, model),
                         x_scale='jac', max_nfev=80, xtol=1e-10, ftol=1e-10)


def load(files):
    reps = []
    for path in files:
        d = np.load(path)
        kp, home, step = float(d['kp']), float(d['home']), float(d['step'])
        n = sum(1 for k in d.keys() if k.endswith('_sign'))
        for i in range(n):
            t = d[f'r{i}_t'][::DECIM].astype(float)
            p = d[f'r{i}_p'][::DECIM].astype(float)
            if len(t) < 60:
                continue
            sign = int(d[f'r{i}_sign'])
            reps.append(dict(t=t - t[0], p=p, q0=float(np.median(p[:5])),
                             target=home + sign * step, step=step, kp=kp))
    return reps


def report(reps, label):
    kps = sorted({r['kp'] for r in reps})
    print(f"\n=== {label}: {len(reps)} reps, kp={kps}, "
          f"t_d fixed {T_D_M11*1000:.2f} ms ===")
    best = None
    for model in MODELS:
        r = fit(reps, model)
        rms = float(np.sqrt(np.mean(r.fun ** 2)))
        I, mgl, q_v, b, C = r.x[:5]
        line = (f"  {model:8s}: rms {rms*100:5.2f}%   I={I*1e3:6.2f}e-3  "
                f"mgl={mgl:5.2f}  b={b:.4f}  C={C:.4f}")
        if model == 'fbfilt':
            line += f"  w_f={r.x[5]:6.1f} rad/s ({1000/r.x[5]:.1f} ms)"
        elif model == 'fbdelay':
            line += f"  t_l={r.x[5]*1000:5.1f} ms"
        print(line)
        if best is None or rms < best[1]:
            best = (model, rms, r.x)
    print(f"  -> best: {best[0]}")
    return best


def selftest():
    rng = np.random.default_rng(1)
    truth = [18e-3, 0.6, -1.4, 0.05, 0.10, 0.030]   # fbdelay, t_l=30 ms
    reps = []
    for kp in (4.0, 8.0, 16.0):
        for i in range(8):
            sign = 1 if i % 2 == 0 else -1
            t = np.arange(0, 0.6, DECIM / 1900.0)
            reps.append(dict(t=t, p=np.zeros_like(t), q0=-1.4,
                             target=-1.4 + sign * 0.15, step=0.15, kp=kp))
    sim = simulate(truth, reps, 'fbdelay')
    for r, s in zip(reps, sim):
        r['p'] = s + rng.normal(0, 2e-4, len(s))
    print("[selftest] truth: fbdelay t_l=30 ms, I=18e-3")
    model, rms, x = report(reps, "selftest")
    ok = model == 'fbdelay' and abs(x[5] - 0.030) < 0.004 \
        and abs(x[0] - 18e-3) / 18e-3 < 0.10
    print("[selftest]", "PASS" if ok else "FAIL — do not trust real fits")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--joint', type=int)
    ap.add_argument('--files', nargs='*')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        selftest()
    if args.joint or args.files:
        files = args.files or sorted(glob.glob(f"data/step_{args.joint}_kp*.npz"))
        report(load(files), f"motor {args.joint} ({', '.join(files)})")


if __name__ == '__main__':
    main()
