#!/usr/bin/env python3
"""
step_lag_fit.py — Replant M4.1: refit the hip_pitch act_lag command
prefilter from pos_step_test.py data, through the verified CAN2SPI
stack, with a synthetic self-test.

Model (single hanging joint, kp-only step, kd=0, tau_ff=0):

    command path:  u(t) = step(t - t_d)          t_d = M1.1 dead time
                   prefilter (act_lag, 2nd order, OPTIONAL):
                       y'' + 2 z w y' + w^2 y = w^2 u
    plant:         I q'' = kp (y - q) - b q' - C tanh(q'/eps)
                             - mgl sin(q - q_v)

Free: I, mgl, q_v, b, C, and (prefilter mode) w_p, z_p.  t_d is FIXED at
the M1.1 measured 2.36 ms — the runbook's fit order is "subtract the
measured delay and filters first, then fit whatever 2nd-order lag
remains".  kd=0 in this test, so the M2.1 velocity filter is off-path.

Decision rule (runbook Phase 4): if the residual command lag is under one
control step (10 ms at 100 Hz), set act_lag_enable=False.  Reported as
the fitted prefilter 10-90% rise time, plus a no-prefilter A/B: if
dropping the prefilter does not raise the residual, it is not needed.

Bonus, and the reason this matters beyond act_lag: the ring frequency of
a kp-driven step pins I directly (w_ring = sqrt((kp + mgl cos)/I)), in
the motor-driven regime the policy actually runs in.  That is an
independent read on the open 43e-3 vs 8.5e-3 inertia gap (M3.1 vs the
M2.1 pendulum), on the same bare limb.

Usage:
    python step_lag_fit.py --selftest
    python step_lag_fit.py --npz data/step_101.npz data/step_101_2.npz
"""

import argparse

import numpy as np
from scipy.optimize import least_squares

T_D_M11 = 0.00236       # M1.1 measured hip_pitch action->motion, s
FS_SIM = 10000.0        # plant integration rate, Hz
EPS_V = 0.01            # tanh stiction width, rad/s
DECIM = 4               # fit every Nth sample (data is ~1.9 kHz)


def simulate(theta, reps, kp, use_pre, t_d=T_D_M11, fs=FS_SIM):
    """Vectorised over reps: integrate the plant on a common grid.

    reps: list of dicts with q0 (IC), target, t (sample times).
    Returns list of position arrays sampled at each rep's own t.
    """
    I, mgl, q_v, b, C = theta[:5]
    if use_pre:
        w_p, z_p = theta[5], theta[6]
    n = len(reps)
    q = np.array([r['q0'] for r in reps], dtype=float)
    tgt = np.array([r['target'] for r in reps], dtype=float)
    v = np.zeros(n)
    y = q.copy() if use_pre else None      # prefilter output starts at rest
    yd = np.zeros(n) if use_pre else None
    t_max = max(r['t'][-1] for r in reps)
    nstep = int(t_max * fs) + 2
    dt = 1.0 / fs
    out = np.empty((nstep, n))
    for k in range(nstep):
        out[k] = q
        t = k * dt
        u = np.where(t >= t_d, tgt, [r['q0'] for r in reps])
        if use_pre:
            ydd = w_p * w_p * (u - y) - 2.0 * z_p * w_p * yd
            yd = yd + ydd * dt
            y = y + yd * dt
            cmd = y
        else:
            cmd = u
        acc = (kp * (cmd - q) - b * v - C * np.tanh(v / EPS_V)
               - mgl * np.sin(q - q_v)) / I
        v = v + acc * dt
        q = q + v * dt
    grid = np.arange(nstep) * dt
    return [np.interp(r['t'], grid, out[:, i]) for i, r in enumerate(reps)]


def residuals(theta, reps, kp, use_pre):
    sim = simulate(theta, reps, kp, use_pre)
    out = []
    for r, s in zip(reps, sim):
        out.append((s - r['p']) / r['step'])
    return np.concatenate(out)


def fit(reps, kp, use_pre):
    #      I      mgl    q_v    b      C   [ w_p   z_p ]
    # q_v is the gravity-vertical in MOTOR coordinates: the homes sit near
    # -1.4 / -2.0 rad, so the bound has to span the real encoder frame, not
    # a small offset around zero.
    q0_med = float(np.median([r['q0'] for r in reps]))
    x0 = [30e-3, 4.7, q0_med, 0.02, 0.10]
    lo = [1e-3, 0.0, q0_med - 3.2, 0.0, 0.0]
    hi = [500e-3, 30.0, q0_med + 3.2, 2.0, 2.0]
    if use_pre:
        x0 += [40.0, 0.8]
        lo += [3.0, 0.05]
        hi += [800.0, 3.0]
    r = least_squares(residuals, x0, bounds=(lo, hi), args=(reps, kp, use_pre),
                      x_scale='jac', xtol=1e-12, ftol=1e-12)
    return r


def rise_10_90(w_p, z_p, fs=20000.0):
    """10-90% rise time of the 2nd-order prefilter step response, s."""
    n = int(2.0 * fs)
    y = yd = 0.0
    dt = 1.0 / fs
    t10 = None
    for k in range(n):
        ydd = w_p * w_p * (1.0 - y) - 2.0 * z_p * w_p * yd
        yd += ydd * dt
        y += yd * dt
        if t10 is None and y >= 0.1:
            t10 = k * dt
        if y >= 0.9:
            return (k * dt) - (t10 or 0.0)
    return float('nan')


def load_reps(paths):
    reps, kps = [], []
    for path in paths:
        d = np.load(path)
        kp = float(d['kp'])
        home = float(d['home'])
        step = float(d['step'])
        kps.append(kp)
        n_rep = sum(1 for k in d.keys() if k.endswith('_sign'))
        for i in range(n_rep):
            t = d[f'r{i}_t'][::DECIM].astype(float)
            p = d[f'r{i}_p'][::DECIM].astype(float)
            if len(t) < 100:
                continue
            sign = int(d[f'r{i}_sign'])
            q0 = float(np.median(p[:5]))
            reps.append(dict(t=t - t[0], p=p, q0=q0,
                             target=home + sign * step, step=step,
                             sign=sign, src=path))
    if len(set(kps)) != 1:
        raise SystemExit(f"mixed kp across files: {kps}")
    return reps, kps[0]


def report(reps, kp, label):
    print(f"\n=== {label} ===")
    print(f"  {len(reps)} reps, kp={kp}, t_d fixed at {T_D_M11*1000:.2f} ms "
          f"(M1.1)")
    out = {}
    for use_pre in (False, True):
        r = fit(reps, kp, use_pre)
        rms = float(np.sqrt(np.mean(r.fun ** 2)))
        I, mgl, q_v, b, C = r.x[:5]
        tag = 'with prefilter ' if use_pre else 'no prefilter  '
        line = (f"  {tag}: rms {rms*100:5.2f}% of step   "
                f"I={I*1e3:6.2f}e-3  mgl={mgl:5.2f}  q_v={q_v:+.3f}  "
                f"b={b:.4f}  C={C:.4f}")
        if use_pre:
            w_p, z_p = r.x[5], r.x[6]
            tr = rise_10_90(w_p, z_p)
            line += f"\n                   w_p={w_p:6.1f} rad/s  z_p={z_p:.3f}  " \
                    f"rise(10-90)={tr*1000:.2f} ms"
            out['w_p'], out['z_p'], out['rise'] = w_p, z_p, tr
        print(line)
        out['rms_pre' if use_pre else 'rms_nopre'] = rms
        out['I_pre' if use_pre else 'I_nopre'] = I
        out['mgl'] = mgl
    # ring frequency implied by the no-prefilter fit -> independent I check
    I, mgl, q_v = out['I_nopre'], out['mgl'], 0.0
    w_ring = np.sqrt(max(kp + mgl, 1e-9) / I)
    print(f"\n  implied ring: {w_ring:.1f} rad/s ({w_ring/(2*np.pi):.2f} Hz) "
          f"from kp+mgl={kp+mgl:.2f} over I={I*1e3:.2f}e-3")
    gain = (out['rms_nopre'] - out['rms_pre']) / out['rms_nopre']
    print(f"  prefilter buys {gain*100:+.1f}% of the residual")
    step_s = 0.01
    verdict = (out.get('rise', 0.0) < step_s) or gain < 0.10
    print(f"\n  M4.1 verdict: act_lag_enable = {'False' if verdict else 'True'}"
          f"   (rule: rise < {step_s*1000:.0f} ms control step, or prefilter "
          f"buys <10%)")
    return out


# ── synthetic self-test ──────────────────────────────────────────────

def selftest():
    """Two truths: a slow prefilter (must be recovered) and none (must
    be reported as fast/unneeded). Guards sign conventions and the
    identifiability of I against mgl before any real fit is trusted."""
    kp = 8.0
    base = dict(I=30e-3, mgl=1.2, q_v=0.05, b=0.02, C=0.10)
    ok = True
    for name, truth_pre in (("slow prefilter w=25 z=0.6", (25.0, 0.6)),
                            ("no prefilter", None)):
        theta = [base['I'], base['mgl'], base['q_v'], base['b'], base['C']]
        use = truth_pre is not None
        if use:
            theta += list(truth_pre)
        reps = []
        for i in range(8):
            sign = 1 if i % 2 == 0 else -1
            t = np.arange(0, 0.5, 1.0 / 1900.0)
            reps.append(dict(t=t, p=np.zeros_like(t), q0=-1.98,
                             target=-1.98 + sign * 0.15, step=0.15,
                             sign=sign, src='synth'))
        sim = simulate(theta, reps, kp, use)
        for r, s in zip(reps, sim):
            r['p'] = s + np.random.default_rng(0).normal(0, 2e-4, len(s))
        print(f"\n[selftest] truth: {name}, I={base['I']*1e3:.1f}e-3")
        got = report(reps, kp, f"selftest — {name}")
        I_err = abs(got['I_nopre'] - base['I']) / base['I']
        if use:
            w_err = abs(got['w_p'] - truth_pre[0]) / truth_pre[0]
            good = w_err < 0.20 and got['rms_pre'] < 0.01
            print(f"[selftest] w_p error {w_err*100:.1f}% -> "
                  f"{'PASS' if good else 'FAIL'}")
        else:
            good = got['rise'] < 0.01 or (got['rms_nopre'] - got['rms_pre']) \
                / got['rms_nopre'] < 0.10
            print(f"[selftest] I error {I_err*100:.1f}% (no-pre fit) -> "
                  f"{'PASS' if good else 'FAIL'}")
        ok = ok and good
    print("\n[selftest]", "PASS" if ok else "FAIL — do not trust the real fit")
    return ok


def modelfree(reps, kp, label):
    """Optimizer-free reading of the two facts a kp-step gives up cheaply.

    1. Steady-state droop.  A step of size S against gravity stiffness kg
       settles at S*kp/(kp+kg), so settle/S -> kg = kp*(S/settle - 1).
       This bounds the gravity stiffness with no dynamics assumed.
    2. Ring.  t_peak = pi/w_d and overshoot -> zeta, hence w_n and then
       I = (kp + kg)/w_n^2.  Independent of the prefilter question.
    """
    print(f"\n=== model-free — {label} ===")
    wd, zt, settle = [], [], []
    for r in reps:
        y = r['sign'] * (r['p'] - r['q0'])
        yf = float(np.median(y[-100:]))
        k_pk = int(np.argmax(y))
        if k_pk < 5 or yf <= 0:
            continue
        os_frac = y[k_pk] / yf - 1.0
        if os_frac <= 1e-3:
            continue
        z = -np.log(os_frac) / np.sqrt(np.pi ** 2 + np.log(os_frac) ** 2)
        wd.append(np.pi / r['t'][k_pk])
        zt.append(z)
        settle.append(yf / r['step'])
    wd, zt, settle = np.array(wd), np.array(zt), np.array(settle)
    wn = wd / np.sqrt(1.0 - zt ** 2)
    kg = kp * (1.0 / settle - 1.0)
    I = (kp + kg) / wn ** 2

    def s(name, v, unit='', sc=1.0):
        print(f"  {name:<16} {np.median(v)*sc:8.3f}{unit}   "
              f"[{v.min()*sc:.3f}, {v.max()*sc:.3f}]  n={len(v)}")
    s('settle/step', settle)
    s('overshoot', np.exp(-np.pi * zt / np.sqrt(1 - zt ** 2)))
    s('w_d', wd, ' rad/s')
    s('zeta', zt)
    s('w_n', wn, ' rad/s')
    s('k_grav', kg, ' Nm/rad')
    s('I', I, 'e-3 kg.m2', 1e3)
    print(f"\n  pinned ratio (kp+k_grav)/I = {np.median((kp+kg)/I):.0f} s^-2")
    print(f"  under M2.1's k_grav=4.73:  I = {(kp+4.73)/np.median(wn)**2*1e3:.2f}e-3")
    print(f"  under measured k_grav:     I = {np.median(I)*1e3:.2f}e-3")
    return dict(I=float(np.median(I)), kg=float(np.median(kg)),
                wn=float(np.median(wn)), zeta=float(np.median(zt)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--npz', nargs='*')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--modelfree', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        selftest()
    if args.npz:
        reps, kp = load_reps(args.npz)
        if args.modelfree:
            modelfree(reps, kp, ' + '.join(args.npz))
        else:
            report(reps, kp, ' + '.join(args.npz))


if __name__ == '__main__':
    main()
