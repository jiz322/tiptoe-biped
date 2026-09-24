#!/usr/bin/env python3
"""
kd_chirp_fit.py — Replant M2.1 analysis: fit the kd-path velocity filter
from kd_filter_chirp.py dwell data, with a synthetic self-test.

Model (single hanging joint, small motion about equilibrium):

    I q'' + [b + b_coul(A_v)] q' + kg q  =  tau_cmd(t - t_d) - kd*H(s) q'

  b_coul = 4C/(pi*A_v)   Coulomb describing function, A_v = w*A_q
  t_d    = actuation delay (command send -> torque, ~2.4 ms per M1.1)
  H(s)   = kd-path velocity filter: pure delay e^{-s*tau_f} or one-pole
           1/(1 + s*tau_f) — both fitted, residuals compared.

Phasor convention: P(x) = j*(2/T)*integral(x e^{-jwt}) over integer
cycles, so x = A sin(wt+phi) -> P = A e^{j phi} and FRF phases subtract
in the usual sense.

Complex residual per dwell:
    r = [ P(tau) - e^{+jw t_d} (kg - I w^2 + jw B) P(q) ] / |P(tau)|
with B = b + 4C/(pi w A_q) + kd*H(jw) (kd pass only).

--selftest simulates the exact bench protocol (20 kHz plant, EMA-filtered
kd velocity, delayed command, 1.9 kHz sampling, same freq/amp ladder)
with known parameters and verifies recovery before the real fit is
trusted.

Usage:
    python kd_chirp_fit.py --selftest
    python kd_chirp_fit.py --npz data/kdchirp_101_2.npz --kd 0.5
"""

import argparse
import math

import numpy as np
from scipy.optimize import least_squares

FREQS_DEFAULT = [1.0, 2.0, 4.0, 8.0, 12.0, 18.0, 25.0]


def phasor(t, x, f, t_skip=2.0):
    m = t > t_skip
    t, x = t[m], x[m] - x[m].mean()
    n_cyc = int((t[-1] - t[0]) * f)
    if n_cyc < 2:
        return None
    m2 = t < t[0] + n_cyc / f
    t, x = t[m2], x[m2]
    w = 2 * np.pi * f
    z = 2 * np.trapz(x * np.exp(-1j * w * t), t) / (t[-1] - t[0])
    return 1j * z


def load_dwells_npz(path):
    d = np.load(path)
    dwells = []
    for kd_i in (0, 1):
        for f in FREQS_DEFAULT:
            key = f"kd{kd_i}_f{f:g}"
            if key + "_t" not in d:
                continue
            t, p, c = d[key + "_t"], d[key + "_p"], d[key + "_c"]
            Pq, Pt = phasor(t, p, f), phasor(t, c, f)
            if Pq is None:
                continue
            dwells.append(dict(f=f, kd_on=kd_i, Pq=Pq, Pt=Pt, Aq=abs(Pq)))
    return dwells


def residuals(theta, dwells, kd, hmodel):
    I, kg, b, C, t_d, tau_f = theta
    out = []
    for dw in dwells:
        w = 2 * np.pi * dw['f']
        B = b + 4 * C / (np.pi * w * dw['Aq'])
        if dw['kd_on']:
            H = (np.exp(-1j * w * tau_f) if hmodel == 'delay'
                 else 1.0 / (1.0 + 1j * w * tau_f))
            B = B + kd * H
        pred = np.exp(1j * w * t_d) * (kg - I * w * w + 1j * w * B) * dw['Pq']
        r = (dw['Pt'] - pred) / abs(dw['Pt'])
        out.extend([r.real, r.imag])
    return np.asarray(out)


def fit(dwells, kd):
    x0 = np.array([8e-3, 4.5, 0.02, 0.05, 0.002, 0.001])
    lo = np.array([1e-4, 0.0, 0.0, 0.0, 0.0, 0.0])
    hi = np.array([1.0, 50.0, 1.0, 1.0, 0.02, 0.05])
    results = {}
    for hmodel in ('delay', 'onepole'):
        r = least_squares(residuals, x0, bounds=(lo, hi),
                          args=(dwells, kd, hmodel), xtol=1e-12, ftol=1e-12)
        results[hmodel] = r
    return results


def report(results, label):
    print(f"\n=== {label} ===")
    for hmodel, r in results.items():
        I, kg, b, C, t_d, tau_f = r.x
        cost = np.sqrt(np.mean(r.fun ** 2))
        print(f"  H={hmodel:7s}: rms resid {cost*100:5.2f}%  "
              f"I={I*1000:.2f}e-3  kg={kg:.2f}  b={b:.4f}  C={C:.4f}  "
              f"t_d={t_d*1000:.2f} ms  tau_f={tau_f*1000:.2f} ms")
    return results


# ── synthetic self-test ──────────────────────────────────────────────

def simulate_dwell(f, amp, kd, secs, I, kg, b, C, t_d, tau_f,
                   fs_plant=20000.0, fs_samp=1900.0):
    dt = 1.0 / fs_plant
    n = int(secs * fs_plant)
    alpha = dt / (tau_f + dt)                    # EMA giving lag tau_f
    q = v = vf = 0.0
    delay_n = int(round(t_d * fs_plant))
    ts, qs, cs = [], [], []
    keep = int(fs_plant / fs_samp)
    w = 2 * np.pi * f
    for i in range(n):
        t = i * dt
        tau_cmd = amp * math.sin(w * t)
        tau_app = amp * math.sin(w * (t - delay_n * dt)) if i >= delay_n else 0.0
        vf += alpha * (v - vf)
        acc = (tau_app - kd * vf - b * v - C * math.tanh(v / 0.01)
               - kg * q) / I
        v += acc * dt
        q += v * dt
        if i % keep == 0:
            ts.append(t)
            qs.append(q)
            cs.append(tau_cmd)
    return np.asarray(ts), np.asarray(qs), np.asarray(cs)


def selftest():
    TRUE = dict(I=8.5e-3, kg=4.7, b=0.015, C=0.08, t_d=0.0024, tau_f=0.005)
    kd = 0.5
    print("[selftest] simulating ladder with", TRUE)
    dwells = []
    for kd_i, kdv in ((0, 0.0), (1, kd)):
        for f in [2.0, 4.0, 8.0, 12.0, 18.0, 25.0]:
            amp = min(0.35 * max(1.0, f / 4.0), 2.0)
            t, q, c = simulate_dwell(f, amp, kdv, 8.0, **TRUE)
            Pq, Pt = phasor(t, q, f), phasor(t, c, f)
            dwells.append(dict(f=f, kd_on=kd_i, Pq=Pq, Pt=Pt, Aq=abs(Pq)))
    results = report(fit(dwells, kd), "selftest (truth: tau_f=5.00 ms, "
                                      "t_d=2.40 ms, H=EMA/onepole)")
    r = results['onepole']
    ok = (abs(r.x[0] - TRUE['I']) / TRUE['I'] < 0.05
          and abs(r.x[5] - TRUE['tau_f']) / TRUE['tau_f'] < 0.15
          and abs(r.x[4] - TRUE['t_d']) < 0.001)
    print("[selftest]", "PASS" if ok else "FAIL — do not trust the real fit")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--npz')
    ap.add_argument('--kd', type=float, default=0.5)
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        selftest()
    if args.npz:
        dwells = load_dwells_npz(args.npz)
        print(f"[fit] {len(dwells)} dwells from {args.npz}")
        report(fit(dwells, args.kd), args.npz)


if __name__ == '__main__':
    main()
