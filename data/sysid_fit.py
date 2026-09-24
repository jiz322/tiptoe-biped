#!/usr/bin/env python3
"""
sysid_fit.py — Fit DCMotor parameters from recorded sysid data.

Uses a manual semi-implicit Euler integrator at 500 Hz (no solve_ivp
overhead) with pre-interpolated command grids. Runs DE global search
+ L-BFGS-B polish.

Usage:
    python3 sysid_fit.py sysid_leg1_*.npz
    python3 sysid_fit.py data.npz --vel-weight 0.05
"""

import math
import sys
import json
import time as _time
import argparse
import numpy as np
from scipy.optimize import differential_evolution, minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════
# Motor specs — sim-space values from DCMotorCfg
# I0/coul0/visc0 are initial-guess priors for reference.
# ══════════════════════════════════════════════════════════════
SPECS = {
    'hip_roll': dict(
        sat=6.0, eff_lim=6.0, v_lim=28.0, kp=6.0, kd=0.3,
        I0=1.4e-3, coul0=0.30, visc0=0.002, grav0=0.3),
    'hip_pitch': dict(
        sat=17.0, eff_lim=17.0, v_lim=27.5, kp=8.0, kd=0.5,
        I0=8.4e-3, coul0=0.40, visc0=0.003, grav0=1.0),
    'knee': dict(
        sat=23.2, eff_lim=23.2, v_lim=20.0, kp=8.0, kd=0.5,
        I0=1.6e-2, coul0=0.50, visc0=0.003, grav0=0.5),
}

PARAM_NAMES = ['inertia', 'coulomb', 'viscous', 'grav_amp',
               'kp_scale', 'kd_scale']
PARAM_UNITS = ['kg*m2', 'Nm', 'Nm*s/rad', 'Nm', 'x', 'x']
BOUNDS = [
    (1e-4,  0.5),    # I
    (0.0,   3.0),    # coulomb
    (0.0,   0.5),    # viscous
    (0.0,   5.0),    # grav_amp
    (0.5,   2.0),    # kp_scale
    (0.5,   2.0),    # kd_scale
]

SIM_DT = 0.002        # 500 Hz physics
DS     = 5             # downsample to 100 Hz for cost


# ══════════════════════════════════════════════════════════════
# Fast integrator (semi-implicit Euler, all-scalar, no scipy)
# ══════════════════════════════════════════════════════════════

def _simulate(p_grid, cv_grid, params, spec, theta_eq, coupling,
              th0, om0):
    """Semi-implicit Euler at 500 Hz.  Returns (pos, vel) at 100 Hz."""
    I, coul, visc, grav, kps, kds = params
    kp   = spec['kp'] * kps
    kd   = spec['kd'] * kds
    sat  = spec['sat']
    eff  = spec['eff_lim']
    vl   = spec['v_lim']
    invI = 1.0 / max(I, 1e-6)
    dt   = SIM_DT

    n     = len(p_grid)
    n_out = (n + DS - 1) // DS
    pos   = np.empty(n_out)
    vel   = np.empty(n_out)
    th, om = float(th0), float(om0)
    j = 0

    for i in range(n):
        if i % DS == 0 and j < n_out:
            pos[j] = th
            vel[j] = om
            j += 1

        # Divergence guard — bad param combos blow up; bail early
        if not (-100.0 < th < 100.0 and -1e4 < om < 1e4):
            for k in range(j, n_out):
                pos[k] = float('nan')
                vel[k] = float('nan')
            return pos[:n_out], vel[:n_out]

        # PD torque
        tau = kp * (p_grid[i] - th) - kd * (om + cv_grid[i] * coupling)
        # velocity-torque envelope
        aom = om if om >= 0.0 else -om
        tm  = sat * (1.0 - aom / vl) if aom < vl else 0.0
        if tau > tm:   tau = tm
        elif tau < -tm: tau = -tm
        if tau > eff:   tau = eff
        elif tau < -eff: tau = -eff
        # friction + gravity
        tau -= coul * math.tanh(om * 20.0) + visc * om
        tau -= grav * math.sin(th - theta_eq)
        # integrate (semi-implicit: velocity first, then position)
        om += dt * tau * invI
        th += dt * om

    return pos[:j], vel[:j]


# ══════════════════════════════════════════════════════════════
# Data loading + pre-interpolation
# ══════════════════════════════════════════════════════════════

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    names   = list(d['trial_names'])
    configs = json.loads(str(d['trial_configs']))
    dpos    = d['default_pos']
    coup    = float(d.get('knee_hip_coupling', 1.0))
    trials  = {}
    for nm, cfg in zip(names, configs):
        trials[nm] = dict(
            t=d[f'{nm}__time'], p_des=d[f'{nm}__p_des'],
            pos=d[f'{nm}__pos'], vel=d[f'{nm}__vel'],
            torque=d[f'{nm}__torque'],
            kp=d[f'{nm}__kp'], kd=d[f'{nm}__kd'], cfg=cfg)
    return trials, dpos, coup


def _extract(trials, j, prefix, hp_j=None):
    out = []
    for nm, tr in trials.items():
        if not nm.startswith(prefix):
            continue
        t = tr['t']
        if len(t) < 10:
            continue
        e = dict(name=nm, t=t,
                 p_des=tr['p_des'][:, j].astype(np.float64),
                 pos=tr['pos'][:, j].astype(np.float64),
                 vel=tr['vel'][:, j].astype(np.float64),
                 pos0=float(tr['pos'][0, j]),
                 vel0=float(tr['vel'][0, j]))
        if hp_j is not None:
            e['cv'] = tr['vel'][:, hp_j].astype(np.float64)
        out.append(e)
    return out


def precompute(prep):
    """Pre-interpolate every trial to a fixed 500 Hz grid.

    Stores p_grid, cv_grid (500 Hz) and pos_100, vel_100 (100 Hz)
    directly in each trial dict, so the hot loop does zero interpolation.
    """
    dt_out = SIM_DT * DS                       # 0.01 s
    for tr in prep:
        t     = tr['t']
        t_end = float(t[-1])
        n500  = int(t_end / SIM_DT) + 1
        t500  = np.arange(n500) * SIM_DT

        tr['p_grid'] = np.interp(t500, t, tr['p_des'])
        cv = tr.get('cv')
        tr['cv_grid'] = (np.interp(t500, t, cv)
                         if cv is not None else np.zeros(n500))

        n100  = (n500 + DS - 1) // DS
        t100  = np.arange(n100) * dt_out
        tr['pos_100'] = np.interp(t100, t, tr['pos'])
        tr['vel_100'] = np.interp(t100, t, tr['vel'])


# ══════════════════════════════════════════════════════════════
# Cost function
# ══════════════════════════════════════════════════════════════

def _cost(params, prep, spec, theta_eq, coupling, vw):
    err = 0.0
    for tr in prep:
        sp, sv = _simulate(tr['p_grid'], tr['cv_grid'],
                           params, spec, theta_eq, coupling,
                           tr['pos0'], tr['vel0'])
        n = min(len(sp), len(tr['pos_100']))
        if n < 5:
            return 1e6
        dp = sp[:n] - tr['pos_100'][:n]
        dv = sv[:n] - tr['vel_100'][:n]
        err += np.dot(dp, dp) / n + vw * np.dot(dv, dv) / n
    return err / len(prep)


# ══════════════════════════════════════════════════════════════
# Fitting
# ══════════════════════════════════════════════════════════════

def fit_joint(prep, spec, theta_eq, coupling=0.0, vw=0.01, seed=42):
    if not prep:
        print("  (no trials)")
        return None

    precompute(prep)

    def obj(p):
        return _cost(p, prep, spec, theta_eq, coupling, vw)

    # Quick sanity: eval cost at DCMotorCfg prior
    p0 = [spec['I0'], spec['coul0'], spec['visc0'],
          spec['grav0'], 1.0, 1.0]
    c0 = obj(p0)
    print(f"  Prior cost = {c0:.6e}")

    t0 = _time.monotonic()
    n_eval = [0]
    def cb(xk, convergence=0):
        n_eval[0] += 1
        if n_eval[0] % 20 == 0:
            elapsed = _time.monotonic() - t0
            print(f"    DE iter {n_eval[0]}, {elapsed:.0f}s elapsed")

    print(f"  DE global search ({len(prep)} trials) ...")
    de = differential_evolution(obj, BOUNDS, seed=seed,
                                maxiter=150, popsize=15,
                                tol=1e-8, callback=cb, workers=1)
    print(f"    DE cost = {de.fun:.6e}  ({_time.monotonic()-t0:.0f}s)")

    print("  L-BFGS-B polish ...")
    ref = minimize(obj, de.x, method='L-BFGS-B', bounds=BOUNDS,
                   options={'maxiter': 500})
    elapsed = _time.monotonic() - t0
    print(f"    Final cost = {ref.fun:.6e}  ({elapsed:.0f}s total)")
    return ref.x


# ══════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════

def plot_overlay(prep, params, spec, theta_eq, coupling, label, path):
    n = len(prep)
    cols = min(n, 4)
    rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5*cols, 3*rows),
                             squeeze=False)
    fig.suptitle(f'{label}: model (red) vs hardware (blue)', fontsize=12)

    for i, tr in enumerate(prep):
        ax = axes[i // cols][i % cols]
        sp, _ = _simulate(tr['p_grid'], tr['cv_grid'],
                          params, spec, theta_eq, coupling,
                          tr['pos0'], tr['vel0'])
        n_pts = min(len(sp), len(tr['pos_100']))
        t_plot = np.arange(n_pts) * SIM_DT * DS

        ax.plot(t_plot, tr['pos_100'][:n_pts], 'b-',  lw=0.9, label='real')
        ax.plot(t_plot, sp[:n_pts],            'r--', lw=0.9, label='sim')
        # p_des at 100 Hz for reference
        p100 = tr['p_grid'][::DS][:n_pts]
        ax.plot(t_plot, p100, 'k:', lw=0.4, alpha=0.35, label='cmd')

        rmse = np.sqrt(np.mean((sp[:n_pts] - tr['pos_100'][:n_pts])**2))
        ax.set_title(f"{tr['name']}  RMSE={rmse:.4f}", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7, loc='upper right')

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> {path}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz')
    ap.add_argument('--vel-weight', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    trials, dpos, coupling = load_npz(args.npz)
    all_idx = [tr['cfg']['idx'] for tr in trials.values()]
    off = 0 if min(all_idx) < 3 else 3
    hr, hp, kn = off, off+1, off+2
    print(f"Leg {'1' if off==0 else '2'}  "
          f"(hr={hr} hp={hp} kn={kn}, coupling={coupling:.3f})\n")

    results = {}

    # ── Hip roll ──
    print("== Hip Roll ==")
    s = SPECS['hip_roll']
    p = _extract(trials, hr, 'hr_')
    par = fit_joint(p, s, dpos[hr], vw=args.vel_weight, seed=args.seed)
    if par is not None:
        results['hip_roll'] = dict(zip(PARAM_NAMES, par.tolist()))
        plot_overlay(p, par, s, dpos[hr], 0.0,
                     'hip_roll', 'sysid_fit_hip_roll.pdf')

    # ── Hip pitch ──
    print("\n== Hip Pitch ==")
    s = SPECS['hip_pitch']
    p = _extract(trials, hp, 'hp_')
    par = fit_joint(p, s, dpos[hp], vw=args.vel_weight, seed=args.seed)
    if par is not None:
        results['hip_pitch'] = dict(zip(PARAM_NAMES, par.tolist()))
        plot_overlay(p, par, s, dpos[hp], 0.0,
                     'hip_pitch', 'sysid_fit_hip_pitch.pdf')

    # ── Knee (stiff) ──
    print("\n== Knee (hip stiff) ==")
    s = SPECS['knee']
    p = _extract(trials, kn, 'kn_stiff_', hp_j=hp)
    par = fit_joint(p, s, dpos[kn], coupling=coupling,
                    vw=args.vel_weight, seed=args.seed)
    if par is not None:
        results['knee'] = dict(zip(PARAM_NAMES, par.tolist()))
        plot_overlay(p, par, s, dpos[kn], coupling,
                     'knee_stiff', 'sysid_fit_knee_stiff.pdf')
        # cross-validate on float data
        pf = _extract(trials, kn, 'kn_float_', hp_j=hp)
        if pf:
            print("\n== Knee (hip float) -- validation ==")
            precompute(pf)
            vc = _cost(par, pf, s, dpos[kn], coupling, args.vel_weight)
            print(f"  Validation cost = {vc:.6e}")
            plot_overlay(pf, par, s, dpos[kn], coupling,
                         'knee_float (val)', 'sysid_fit_knee_float.pdf')

    # ── Summary table ──
    print("\n" + "=" * 72)
    hdr = f"{'':>12s}"
    for jt in ('hip_roll', 'hip_pitch', 'knee'):
        hdr += f"  {jt:>12s}"
    print(hdr)
    print("-" * 72)
    for pn, pu in zip(PARAM_NAMES, PARAM_UNITS):
        row = f"{pn:>12s}"
        for jt in ('hip_roll', 'hip_pitch', 'knee'):
            r = results.get(jt, {})
            row += f"  {r[pn]:12.5f}" if pn in r else f"  {'--':>12s}"
        row += f"   {pu}"
        print(row)
    print("-" * 72)
    print("DCMotorCfg priors:")
    for jt in ('hip_roll', 'hip_pitch', 'knee'):
        s = SPECS[jt]
        print(f"  {jt}: I={s['I0']:.4f}  coul={s['coul0']:.3f}  "
              f"visc={s['visc0']:.4f}  grav~{s['grav0']:.2f}")
    print("=" * 72)

    out = args.npz.replace('.npz', '_fit.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == '__main__':
    main()