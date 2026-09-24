#!/usr/bin/env python3
"""
coastdown_fit.py — Replant P5 global fit for coast_down_bench.py captures.

Model (motor-space, per rep):
    I*dv/dt = tau_cmd - C*sign(v) - b*v - A*sin(th_j + phi) - d
linearized as
    dv/dt = bt*tau + bs*sign(v) + bv*v + b1*sin(th) + b2*cos(th) + b0
so I = 1/bt, C = -bs*I, b = -bv*I, A = I*hypot(b1,b2), d = -b0*I.
The burst samples (known commanded tau, delivery 1:1 per M3.1) are what
identify I; the coast ringdown pins C, b, gravity. A coast-only fit
(ratios C/I, b/I) is printed as a cross-check.

dv/dt comes from Savitzky-Golay differentiation of the (uniformly
resampled) velocity; samples near |v|=0 and near the burst->coast
transition are masked. Velocity source is telemetry (decode verified in
P0); the position-derivative consistency ratio is printed as a guard.

Self-test (run FIRST, before real data — methodology rule):
    python3 coastdown_fit.py --selftest
plants known I/C/b/gravity in a simulated protocol (timestamp jitter +
sensor noise included) and must recover them.

Usage:
    python3 coastdown_fit.py coast_101_hang.npz [more.npz]
    python3 coastdown_fit.py --pinned-inertia bal_101_braced.npz bal_102_hipclamped.npz
        (2026-09-18: rotor PINNED at the vendor 4.2e-3 + URDF link, the four
        pre-registered re-attribution hypotheses, one table row per file x
        hypothesis — see the Pinned-inertia section below)
"""

import argparse
import math

import numpy as np

# joint projection for the gravity regressor argument (see config.py):
# th_joint = k * p_raw + const (const absorbed by the sin/cos pair).
# knees: JOINT_SIGN -1, gear 22/30; hips/rolls: direct, sign +1.
K_JOINT = {101: 1.0, 104: 1.0, 103: 1.0, 106: 1.0,
           102: -22.0 / 30.0, 105: -22.0 / 30.0}

# firmware limit_torque per motor (RS01 17 Nm, EL05 6 Nm) — clips the
# reconstructed PD brake torque
TAU_CLIP = {101: 17.0, 104: 17.0, 102: 17.0, 105: 17.0,
            103: 6.0, 106: 6.0}

PH_BURST, PH_COAST, PH_BRAKE = 1, 2, 3


def savgol_coeffs(window, order, deriv):
    half = window // 2
    x = np.arange(-half, half + 1, dtype=float)
    V = np.vander(x, order + 1, increasing=True)
    # pinv row `deriv` gives the kernel; factorial for derivative scale
    kern = np.linalg.pinv(V)[deriv]
    return kern * float(math.factorial(deriv))


def sg_filter(y, window, order, deriv, dt):
    k = savgol_coeffs(window, order, deriv)[::-1]
    out = np.convolve(y, k, mode='same') / dt ** deriv
    return out


def prep_rep(t, p, v, tau, ph, window=21, order=3, brake=None,
             onset_s=0.04):
    """Uniform resample + SG smooth/derivative. Returns sample dict with
    edge and torque-discontinuity regions masked out.

    brake=(kp, kd, home, clip): reconstruct commanded torque during
    PH_BRAKE samples from the known firmware PD law on the SMOOTHED
    states, clipped at limit_torque. Without it, brake samples are
    dropped. The fast decel pole at brake onset (kd/I, ~10-15 ms) is
    faster than the SG bandwidth — the transition mask covers it.

    onset_s: mask this much time at the start of each contiguous
    PH_BURST run. Real bursts open with a lash-engagement transient
    (rotor accelerates alone through gear lash at ~tau/I_rotor, then
    dumps momentum into the leg — observed on hip_pitch 101,
    2026-08-17) that the rigid model cannot represent."""
    if len(t) < window + 4:
        return None
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return None
    tu = np.arange(t[0], t[-1], dt)
    pu = np.interp(tu, t, p)
    vu = np.interp(tu, t, v)
    tau_u = np.interp(tu, t, tau)
    ph_u = np.round(np.interp(tu, t, ph.astype(float)))
    vs = sg_filter(vu, window, order, 0, dt)
    a = sg_filter(vu, window, order, 1, dt)
    vp = sg_filter(pu, window, order, 1, dt)   # position-derivative vel
    half = window // 2
    good = np.ones(len(tu), bool)
    good[:half] = good[-half:] = False
    # mask where the SG window spans a command discontinuity: phase
    # changes (burst->coast/brake) and logged-tau jumps (pingpong flips).
    # Computed on LOGGED tau, before brake reconstruction — reconstructed
    # brake torque varies smoothly and must not be masked.
    trans = set(np.flatnonzero(np.abs(np.diff(tau_u)) > 0.05))
    trans |= set(np.flatnonzero(np.abs(np.diff(ph_u)) > 1e-6))
    for i in trans:
        good[max(0, i - half):i + half + 1] = False
    # onset mask: start of each contiguous burst run
    is_b = ph_u == PH_BURST
    starts = np.flatnonzero(is_b & ~np.roll(is_b, 1))
    if len(is_b) and is_b[0]:
        starts = np.union1d(starts, [0])
    n_on = int(np.ceil(onset_s / dt))
    for i in starts:
        good[i:i + n_on] = False
    m3 = ph_u == PH_BRAKE
    if brake is not None and m3.any():
        kp, kd, home, clip = brake
        ps_s = sg_filter(pu, window, order, 0, dt)
        tau_u = tau_u.copy()
        tau_u[m3] = np.clip(kp * (home - ps_s[m3]) - kd * vs[m3],
                            -clip, clip)
    else:
        good &= ~m3
    return dict(t=tu, p=pu, v=vs, a=a, vp=vp, tau=tau_u,
                ph=ph_u, good=good, sg=(window, order, dt))


def fit(samples, k_joint, vmin, with_tau=True):
    """Pooled linear regression. samples = list of prep_rep dicts."""
    cols, ys = [], []
    for s in samples:
        m = s['good'] & (np.abs(s['v']) > vmin)
        if not with_tau:
            m &= (s['ph'] == PH_COAST) & (s['tau'] == 0.0)
        if m.sum() == 0:
            continue
        th = k_joint * s['p'][m]
        base = [np.sign(s['v'][m]), s['v'][m],
                np.sin(th), np.cos(th), np.ones(m.sum())]
        if with_tau:
            base.insert(0, s['tau'][m])
        cols.append(np.column_stack(base))
        ys.append(s['a'][m])
    if not cols:
        return None
    X = np.vstack(cols)
    y = np.concatenate(ys)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_r = float(np.sum((y - pred) ** 2))
    ss_t = float(np.sum((y - y.mean()) ** 2)) or 1e-12
    r2 = 1.0 - ss_r / ss_t
    rms = float(np.sqrt(ss_r / len(y)))
    return beta, r2, rms, len(y)


def fit_burst(samples, vmin):
    """PRIMARY fit for ballistic captures: burst samples only (tau
    exactly known, no PD-model assumption, no stick-slip regime),
    onset-masked. Gravity linearized over the small arc:
        dv/dt = [tau + g0 + g1*(p-home) - C*sign(v) - b*v] / I
    C and I separate through the level contrast + direction flips."""
    cols, ys = [], []
    for s in samples:
        m = s['good'] & (np.abs(s['v']) > vmin) & (s['ph'] == PH_BURST)
        if not m.sum():
            continue
        dp = s['p'][m] - s['home']
        cols.append(np.column_stack([s['tau'][m], np.ones(m.sum()), dp,
                                     np.sign(s['v'][m]), s['v'][m]]))
        ys.append(s['a'][m])
    if not cols:
        return None
    X = np.vstack(cols)
    y = np.concatenate(ys)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_r = float(np.sum((y - pred) ** 2))
    ss_t = float(np.sum((y - y.mean()) ** 2)) or 1e-12
    I = 1.0 / beta[0]
    return dict(I=I, g0=beta[1] * I, g1=beta[2] * I, C=-beta[3] * I,
                b=-beta[4] * I, r2=1.0 - ss_r / ss_t,
                rms=float(np.sqrt(ss_r / len(y))), n=len(y))


def naive_table(samples):
    """Model-free anchor: central-half slope of each burst (the M3.1
    method). Robust to the onset transient by construction."""
    print("[fit] naive per-burst I = tau / (central-half dv/dt):")
    vals = []
    for i, s in enumerate(samples):
        m = s['ph'] == PH_BURST
        if m.sum() < 8:
            continue
        tb, vb = s['t'][m], s['v'][m]
        lvl = float(np.median(np.abs(s['tau'][m])))
        n = m.sum()
        sl = slice(n // 4, 3 * n // 4)
        A = np.column_stack([tb[sl], np.ones(len(tb[sl]))])
        (al, _), *_ = np.linalg.lstsq(A, vb[sl], rcond=None)
        if abs(al) > 1e-3:
            vals.append(lvl / abs(al) * 1e3)
            print(f"  rep {i:2d}: {lvl:4.1f} Nm  alpha {al:+7.1f}  "
                  f"I {lvl/abs(al)*1e3:5.1f} e-3")
    if vals:
        print(f"  median I_naive = {float(np.median(vals)):.1f} e-3 "
              f"(upper-bound flavored: ignores friction/gravity)")


def report_ballistic(samples, vmin):
    naive_table(samples)
    fb = fit_burst(samples, vmin)
    if fb:
        print(f"\n[fit] BURST-ONLY PRIMARY (onset-masked, n={fb['n']}, "
              f"R2={fb['r2']:.4f}, rms {fb['rms']:.2f} rad/s^2):")
        print(f"  I  = {fb['I']*1000:8.2f} e-3 kg m^2")
        print(f"  C  = {fb['C']:8.4f} Nm   (Coulomb, kinetic)")
        print(f"  b  = {fb['b']:8.4f} Nm s/rad   (viscous)")
        print(f"  gravity: g0 {fb['g0']:+.3f} Nm at home, slope "
              f"{fb['g1']:+.3f} Nm/rad")
    return fb


def report(samples, k_joint, vmin):
    # velocity-decode guard: telemetry vel vs d(pos)/dt
    vv = np.concatenate([s['v'][s['good']] for s in samples])
    vp = np.concatenate([s['vp'][s['good']] for s in samples])
    mm = np.abs(vp) > max(vmin, 0.5)
    ratio = float(np.median(vv[mm] / vp[mm])) if mm.sum() > 50 else np.nan
    print(f"[fit] vel telemetry / d(pos)/dt median ratio: {ratio:.4f} "
          f"(expect ~1.00; June bug read 1.136)")

    out = {}
    g = fit(samples, k_joint, vmin, with_tau=True)
    if g:
        beta, r2, rms, n = g
        bt, bs, bv, b1, b2, b0 = beta
        I = 1.0 / bt
        C, b = -bs * I, -bv * I
        A, d = I * float(np.hypot(b1, b2)), -b0 * I
        print(f"\n[fit] GLOBAL (burst+coast, n={n}, R2={r2:.4f}, "
              f"rms {rms:.2f} rad/s^2):")
        print(f"  I  = {I*1000:8.2f} e-3 kg m^2   (1/beta_tau)")
        print(f"  C  = {C:8.4f} Nm   (Coulomb, kinetic)")
        print(f"  b  = {b:8.4f} Nm s/rad   (viscous)")
        print(f"  A  = {A:8.3f} Nm  gravity amplitude, d = {d:+.4f} Nm")
        out.update(I=I, C=C, b=b, A=A, r2=r2)
    c = fit(samples, k_joint, vmin, with_tau=False)
    if c:
        beta, r2, rms, n = c
        bs, bv, b1, b2, b0 = beta
        print(f"\n[fit] COAST-ONLY cross-check (n={n}, R2={r2:.4f}):")
        print(f"  C/I = {-bs:8.2f}   b/I = {-bv:8.2f}   (1/s^2, 1/s)")
        if 'I' in out:
            print(f"  -> with global I: C = {-bs*out['I']:.4f} Nm, "
                  f"b = {-bv*out['I']:.4f} Nm s/rad")
    # per-rep residuals against the global model
    if g:
        print("\n[fit] per-rep residual rms (rad/s^2) — outliers suggest "
              "tether/base motion:")
        beta = g[0]
        for i, s in enumerate(samples):
            m = s['good'] & (np.abs(s['v']) > vmin)
            if m.sum() < 20:
                continue
            th = k_joint * s['p'][m]
            X = np.column_stack([s['tau'][m], np.sign(s['v'][m]), s['v'][m],
                                 np.sin(th), np.cos(th), np.ones(m.sum())])
            res = s['a'][m] - X @ beta
            print(f"  rep {i:2d}: {float(np.sqrt(np.mean(res**2))):7.2f} "
                  f"(n={m.sum()})")
    return out


def load_reps(paths, sg_window=None):
    samples, motor = [], None
    for path in paths:
        d = np.load(path, allow_pickle=True)
        motor = int(d['motor'])
        # ballistic bursts are ~25 samples: a 21-sample SG window smears
        # them (selftest: 8% I bias). 9 recovers <=0.5% there; longer
        # smooth records (coast/pingpong) keep 21.
        mode = str(d['mode']) if 'mode' in d.files else 'coast'
        w = sg_window or (9 if mode == 'ballistic' else 21)
        brake = None
        if 'brake_kp' in d.files:
            brake = (float(d['brake_kp']), float(d['brake_kd']),
                     float(d['home']), TAU_CLIP[motor])
        for k in range(int(d['n_reps'])):
            s = prep_rep(d[f'n{k}_t'], d[f'n{k}_p'], d[f'n{k}_v'],
                         d[f'n{k}_tau'], d[f'n{k}_ph'], window=w,
                         brake=brake)
            if s is not None:
                s['home'] = float(d['home'])
                s['is_bal'] = mode == 'ballistic'
                s['brake'] = brake
                samples.append(s)
    return samples, motor


# ────────────────────────────────────────────────────────────
# Pinned-inertia re-fit (2026-09-18) — `--pinned-inertia`
# ────────────────────────────────────────────────────────────
# The plant's rotor inertia is now the VENDOR value
# (belt_knee_actuator.ROTOR_INERTIA_OUTPUT = 4.2e-3 at the RS01 output
# shaft), so the bench's 4-5x apparent-inertia excess has to be
# re-attributed rather than carried as armature.  Four pre-registered
# hypotheses, each fitted with the rigid inertia PINNED at rotor + URDF link
# and all scored with ONE metric — R^2 of the SG motor-side acceleration on
# the same samples/mask fit_burst uses — so the unpinned PRIMARY fit is the
# reference row of the table:
#   friction : I pinned, tau scale 1, C/b/gravity free (linear LS).
#              Expected CONTRADICTED: friction moves decay, not tau/alpha.
#   tscale   : I pinned, ONE scalar tau scale s free (linear LS).  The
#              fitted gravity slope must still match the URDF slope
#              (independent of any inertia assumption: a fit returns g/s),
#              and s must agree between hip and knee files.
#   delay    : I pinned, brake-phase PD torque rebuilt from states delayed
#              by T (grid), C/b/gravity free.  Plausible only if the best T
#              is <= the measured 5 ms CAN2SPI dead time.
#   belt     : knee files only — two-inertia simulation (rotor | belt k, d |
#              link, encoder on the rotor), DE + polish; k_joint must land in
#              the S7 band 100-1000 N m/rad.  Physics bound stated up front:
#              with no friction a series compliance can only move the
#              motor-side apparent inertia BETWEEN I_rotor and
#              I_rotor + G^2*I_link; anything above that is friction (C, b)
#              masquerading as inertia in the naive slope.
ROTOR_INERTIA_OUTPUT = 4.2e-3      # kg m^2 at the motor output shaft (vendor)
G_BELT = 22.0 / 30.0
# URDF link inertia about the bench axis, JOINT side
# (data/_p7_armature_from_urdf.py: hip bench pose knee raw 1.60 -> 17.09e-3
# about the hip; shin+foot 2.50e-3 about the knee).
LINK_INERTIA_JOINT = {101: 17.09e-3, 104: 17.09e-3, 102: 2.50e-3, 105: 2.50e-3}
# joint -> motor-side inertia factor (the knee sits behind the 22:30 belt)
JOINT_TO_MOTOR_I = {101: 1.0, 104: 1.0, 102: G_BELT ** 2, 105: G_BELT ** 2}
# URDF gravity slope at the hip bench home (same script; bench-fit 0.985):
# the independent torque-scale anchor — a fitted slope reads g_URDF / s.
URDF_GRAV_SLOPE = {101: 0.913, 104: 0.913}
R2_BAR = 0.98
BELT_BAND = (100.0, 1000.0)        # S7 belt_series_stiffness band, joint side
DELAY_MAX_S = 0.005                # measured CAN2SPI dead time
KP_SWEEP_KP = 20.0                 # step_dep_* deploy-gain batteries (hip)
KP_SWEEP_F_HW = (3.06, 3.34)       # measured hip fundamentals, runbook 1.1


def pinned_inertia(motor):
    f = JOINT_TO_MOTOR_I[motor]
    return ROTOR_INERTIA_OUTPUT + f * LINK_INERTIA_JOINT[motor], f


def _phase_mask(s, vmin, phase):
    return s['good'] & (np.abs(s['v']) > vmin) & (s['ph'] == phase)


def _r2(y, pred):
    ss_r = float(np.sum((y - pred) ** 2))
    ss_t = float(np.sum((y - y.mean()) ** 2)) or 1e-12
    return 1.0 - ss_r / ss_t, float(np.sqrt(ss_r / len(y)))


def fit_pinned_linear(samples, vmin, I_pin, free_scale):
    """H-friction (free_scale=False) / H-tscale (True): burst-only LS with
    I_pin*a as the target.  The known tau term is part of the prediction,
    so R^2 is comparable across hypotheses (and to the belt/delay rows)."""
    xs, ys, taus = [], [], []
    for s in samples:
        m = _phase_mask(s, vmin, PH_BURST)
        if not m.sum():
            continue
        dp = s['p'][m] - s['home']
        xs.append(np.column_stack([np.ones(m.sum()), dp,
                                   np.sign(s['v'][m]), s['v'][m]]))
        ys.append(I_pin * s['a'][m])
        taus.append(s['tau'][m])
    if not xs:
        return None
    X, y, tau = np.vstack(xs), np.concatenate(ys), np.concatenate(taus)
    if free_scale:
        Xf = np.column_stack([tau, X])
        beta, *_ = np.linalg.lstsq(Xf, y, rcond=None)
        sc, (g0, g1, bs, bv) = float(beta[0]), beta[1:]
        pred = Xf @ beta
    else:
        beta, *_ = np.linalg.lstsq(X, y - tau, rcond=None)
        sc, (g0, g1, bs, bv) = 1.0, beta
        pred = tau + X @ beta
    r2, rms = _r2(y, pred)
    return dict(s=sc, g0=float(g0), g1=float(g1), C=float(-bs), b=float(-bv),
                r2=r2, rms=rms / I_pin, n=len(y))


def fit_pinned_delay(samples, vmin, I_pin, delays):
    """H-delay on the brake phase (the only closed-loop segment in a
    ballistic file): PD torque rebuilt from states delayed by T,
    I pinned, s = 1, C/b/gravity free.  Best-T row plus R^2 at T = 0."""
    best, r2_0 = None, np.nan
    for T in delays:
        xs, ys, taus = [], [], []
        for s in samples:
            br = s.get('brake')
            if br is None:
                continue
            kp, kd, home, clip = br
            m = _phase_mask(s, vmin, PH_BRAKE)
            if m.sum() < 10:
                continue
            t = s['t']
            p_d = np.interp(t - T, t, s['p'])
            v_d = np.interp(t - T, t, s['v'])
            tau_d = np.clip(kp * (home - p_d) - kd * v_d, -clip, clip)
            dp = s['p'][m] - home
            xs.append(np.column_stack([np.ones(m.sum()), dp,
                                       np.sign(s['v'][m]), s['v'][m]]))
            ys.append(I_pin * s['a'][m])
            taus.append(tau_d[m])
        if not xs:
            return None
        X, y, tau = np.vstack(xs), np.concatenate(ys), np.concatenate(taus)
        beta, *_ = np.linalg.lstsq(X, y - tau, rcond=None)
        r2, rms = _r2(y, tau + X @ beta)
        if T == delays[0]:
            r2_0 = r2
        row = dict(T=T, g0=float(beta[0]), g1=float(beta[1]), C=float(-beta[2]),
                   b=float(-beta[3]), r2=r2, rms=rms / I_pin, n=len(y))
        if best is None or r2 > best['r2']:
            best = row
    best['r2_T0'] = r2_0
    return best


def _belt_reps(samples, vmin):
    """Burst segments as (t_rel, tau_on_grid, m0, md0) plus the eval mask."""
    reps = []
    for s in samples:
        isb = s['ph'] == PH_BURST
        if isb.sum() < 8:
            continue
        i0, i1 = int(np.argmax(isb)), int(len(isb) - np.argmax(isb[::-1]))
        t = s['t'][i0:i1] - s['t'][i0]
        reps.append(dict(t=t, tau=s['tau'][i0:i1], m0=float(s['p'][i0]),
                         md0=float(s['v'][i0]), home=s['home'], sl=(i0, i1),
                         mask=_phase_mask(s, vmin, PH_BURST)[i0:i1],
                         a=s['a'][i0:i1], sg=s['sg']))
    return reps


def _simulate_two_mass(reps, params, I_r, I_y, h=2.5e-4):
    """Vectorised over reps.  Motor-referred coordinates: rotor m (encoder,
    inertia I_r) | belt k_m = k_j*G^2, d_m | link y (I_y = G^2*I_link).
    Semi-implicit Euler at h; stable while sqrt(k_m*(1/I_r+1/I_y))*h < 2
    (k_j <= 3000 at h = 0.25 ms).  Returns motor velocity per rep grid."""
    k_j, d_m, C, b, g0, g1 = params
    k_m = k_j * G_BELT ** 2
    n = len(reps)
    K = int(max(r['t'][-1] for r in reps) / h) + 2
    tauF = np.empty((n, K))
    for i, r in enumerate(reps):
        idx = np.minimum((np.arange(K) * h / (r['t'][1] - r['t'][0])).astype(int),
                         len(r['tau']) - 1)
        tauF[i] = r['tau'][idx]
    m = np.array([r['m0'] for r in reps]); md = np.array([r['md0'] for r in reps])
    y, yd = m.copy(), md.copy()
    home = np.array([r['home'] for r in reps])
    hist = np.empty((n, K))
    for k in range(K):
        f_b = k_m * (m - y) + d_m * (md - yd)
        a_m = (tauF[:, k] - C * np.tanh(md / 0.05) - b * md - f_b) / I_r
        a_y = (f_b + g0 + g1 * (y - home)) / I_y
        md = md + h * a_m
        yd = yd + h * a_y
        m = m + h * md
        y = y + h * yd
        hist[:, k] = md
    tf = np.arange(K) * h
    return [np.interp(r['t'], tf, hist[i]) for i, r in enumerate(reps)]


def _belt_cost(params, reps, I_r, I_y, return_parts=False):
    vs = _simulate_two_mass(reps, params, I_r, I_y)
    ys, preds, naive = [], [], []
    for r, v in zip(reps, vs):
        if not np.all(np.isfinite(v)):
            return 1e9 if not return_parts else (np.nan, np.nan, np.nan)
        w, o, dt = r['sg']
        a_sim = sg_filter(v, w, o, 1, dt)
        ys.append(r['a'][r['mask']])
        preds.append(a_sim[r['mask']])
        n = len(v)
        sl = slice(n // 4, 3 * n // 4)
        A = np.column_stack([r['t'][sl], np.ones(len(r['t'][sl]))])
        (al, _), *_ = np.linalg.lstsq(A, v[sl], rcond=None)
        lvl = float(np.median(np.abs(r['tau'])))
        if abs(al) > 1e-3:
            naive.append(lvl / abs(al))
    y, pred = np.concatenate(ys), np.concatenate(preds)
    if return_parts:
        r2, rms = _r2(y, pred)
        return r2, rms, (float(np.median(naive)) if naive else np.nan)
    return float(np.mean((y - pred) ** 2))


def fit_pinned_belt(samples, vmin, I_r, I_y, de_iters=40, seed=42):
    """H-belt: two-inertia sim fit, rotor pinned, belt k_j (joint N m/rad),
    d_m (motor N m s/rad), C, b, g0, g1 free.  DE global + L-BFGS-B polish
    (the sysid_fit.py recipe).  Returns the fitted row with the R^2 on the
    same burst-acceleration samples as the linear rows and the apparent
    inertia the fitted model would show to naive_table()."""
    from scipy.optimize import differential_evolution, minimize
    reps = _belt_reps(samples, vmin)
    if not reps:
        return None
    bounds = [(20.0, 3000.0), (0.0, 2.0), (0.0, 1.5), (0.0, 0.1),
              (-3.0, 3.0), (-3.0, 3.0)]
    obj = lambda p: _belt_cost(p, reps, I_r, I_y)
    de = differential_evolution(obj, bounds, seed=seed, maxiter=de_iters,
                                popsize=8, tol=1e-7, polish=False, workers=1)
    ref = minimize(obj, de.x, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 200})
    p = ref.x if ref.fun <= de.fun else de.x
    r2, rms, I_naive_sim = _belt_cost(p, reps, I_r, I_y, return_parts=True)
    k_j, d_m, C, b, g0, g1 = p
    zeta = d_m / (2.0 * np.sqrt(k_j * G_BELT ** 2 * I_r)) if k_j > 0 else np.nan
    return dict(k_j=float(k_j), d_m=float(d_m), zeta=float(zeta), C=float(C),
                b=float(b), g0=float(g0), g1=float(g1), r2=r2, rms=rms,
                I_naive_sim=I_naive_sim, n_reps=len(reps), de_cost=float(de.fun))


def run_pinned(args):
    hyps = ('friction', 'tscale', 'delay', 'belt') if args.pinned_hyp == 'all' \
        else (args.pinned_hyp,)
    delays = np.arange(0.0, 0.0405, 0.001)
    rows, scales = [], {}
    print(f"[pinned] rotor I_out = {ROTOR_INERTIA_OUTPUT*1e3:.2f}e-3 (vendor, motor "
          f"output shaft); R^2 bar {R2_BAR}; belt band {BELT_BAND}; delay bar "
          f"{DELAY_MAX_S*1e3:.0f} ms")
    for path in args.npz:
        samples, motor = load_reps([path], args.sg_window)
        trial = path.replace('\\', '/').split('/')[-1].replace('.npz', '')
        if motor not in LINK_INERTIA_JOINT:
            print(f"\n[pinned] {trial}: motor {motor} SKIPPED — no URDF link "
                  f"inertia pinned for it (LINK_INERTIA_JOINT covers hip_pitch "
                  f"101/104 and knee 102/105 only); add an entry to fit it")
            continue
        I_pin, f = pinned_inertia(motor)
        I_r = ROTOR_INERTIA_OUTPUT
        I_y = f * LINK_INERTIA_JOINT[motor]
        print(f"\n[pinned] {trial}: motor {motor}, {len(samples)} reps, "
              f"I_pin = {I_r*1e3:.2f}e-3 rotor + {I_y*1e3:.2f}e-3 link "
              f"= {I_pin*1e3:.2f}e-3 (motor side)")
        naive_med = None
        fb = fit_burst(samples, args.vmin)
        if fb:
            ratio = fb['I'] / I_pin
            rows.append((trial, 'unpinned (ref)', fb['r2'],
                         f"I_app {fb['I']*1e3:.1f}e-3 = {ratio:.2f}x pinned; "
                         f"C {fb['C']:.3f} b {fb['b']:.3f} g1 {fb['g1']:+.3f}",
                         '—', 'reference: the excess the hypotheses must absorb'))
        if motor in URDF_GRAV_SLOPE:
            f_pred = np.sqrt(KP_SWEEP_KP / I_pin) / (2 * np.pi)
            rows.append((trial, 'kp-sweep check', np.nan,
                         f"pinned plant f1 = {f_pred:.2f} Hz at kp {KP_SWEEP_KP:g}",
                         'supported' if KP_SWEEP_F_HW[0] <= f_pred <= KP_SWEEP_F_HW[1]
                         else 'contradicted',
                         f"hw hip fundamentals {KP_SWEEP_F_HW[0]}-{KP_SWEEP_F_HW[1]} Hz "
                         f"(runbook 1.1); closed-loop f = sqrt(s*kp/I)/2pi"))
        if 'friction' in hyps:
            r = fit_pinned_linear(samples, args.vmin, I_pin, free_scale=False)
            if r:
                ok = r['r2'] >= R2_BAR and r['C'] >= 0 and r['b'] >= 0
                rows.append((trial, 'H-friction', r['r2'],
                             f"C {r['C']:.3f} b {r['b']:.4f} g1 {r['g1']:+.3f}",
                             'supported' if ok else 'contradicted',
                             f"R^2 vs bar; C,b must be >= 0 (rms {r['rms']:.1f} rad/s^2)"))
        if 'tscale' in hyps:
            r = fit_pinned_linear(samples, args.vmin, I_pin, free_scale=True)
            if r:
                scales[trial] = r['s']
                fitted, sig = f"s {r['s']:.3f} C {r['C']:.3f} b {r['b']:.4f} g1 {r['g1']:+.3f}", ''
                ok = r['r2'] >= R2_BAR
                if motor in URDF_GRAV_SLOPE:
                    g_ratio = abs(r['g1']) / URDF_GRAV_SLOPE[motor]
                    ok = ok and 0.8 <= g_ratio <= 1.2
                    sig = (f"fitted |g1| / URDF slope {URDF_GRAV_SLOPE[motor]} = {g_ratio:.2f} "
                           f"(must be 0.8-1.2: gravity is the s anchor); ")
                rows.append((trial, 'H-tscale', r['r2'], fitted,
                             'supported' if ok else 'contradicted',
                             sig + "one s for hip AND knee (see summary)"))
        if 'delay' in hyps:
            r = fit_pinned_delay(samples, args.vmin, I_pin, delays)
            if r:
                ok = r['r2'] >= R2_BAR and r['T'] <= DELAY_MAX_S + 1e-9
                rows.append((trial, 'H-delay', r['r2'],
                             f"T {r['T']*1e3:.0f} ms (R^2 at T=0: {r['r2_T0']:.3f}) "
                             f"C {r['C']:.3f} b {r['b']:.4f}",
                             'supported' if ok else 'contradicted',
                             f"brake-phase closed loop; best T must be <= {DELAY_MAX_S*1e3:.0f} ms "
                             f"AND R^2 >= bar"))
        if 'belt' in hyps:
            if motor in (102, 105):
                r = fit_pinned_belt(samples, args.vmin, I_r, I_y, de_iters=args.de_iters)
                if r:
                    ok = r['r2'] >= R2_BAR and BELT_BAND[0] <= r['k_j'] <= BELT_BAND[1]
                    rows.append((trial, 'H-belt', r['r2'],
                                 f"k_j {r['k_j']:.0f} N m/rad d_m {r['d_m']:.3f} (zeta {r['zeta']:.2f}) "
                                 f"C {r['C']:.3f} b {r['b']:.4f}; model's naive I "
                                 f"{r['I_naive_sim']*1e3:.1f}e-3 vs bound {(I_r+I_y)*1e3:.1f}e-3",
                                 'supported' if ok else 'contradicted',
                                 f"k_j must sit in {BELT_BAND} AND R^2 >= bar; with C=b=0 a "
                                 f"series spring cannot lift naive I above I_r + G^2 I_link "
                                 f"(friction at its bound is what lifts it)"))
            else:
                rows.append((trial, 'H-belt', np.nan, 'direct drive, no belt',
                             'n/a', 'hip motor has no belt; gearbox compliance not pre-registered'))
    print("\n" + "=" * 118)
    print(f"{'trial':<24s} {'hypothesis':<16s} {'R^2':>7s}  {'fitted':<58s} {'verdict':<13s}")
    print("-" * 118)
    for trial, hyp, r2, fitted, verdict, sig in rows:
        r2s = f"{r2:7.4f}" if np.isfinite(r2) else f"{'—':>7s}"
        print(f"{trial:<24s} {hyp:<16s} {r2s}  {fitted:<58s} {verdict:<13s}")
        print(f"{'':<24s} {'':<16s} {'':>7s}  signal: {sig}")
    if len(scales) >= 2:
        vals = list(scales.values())
        spread = (max(vals) - min(vals)) / max(abs(np.mean(vals)), 1e-9)
        print("-" * 118)
        print(f"H-tscale cross-trial: s = " + ", ".join(f"{k} {v:.3f}" for k, v in scales.items())
              + f"  -> spread {spread*100:.0f}% of mean; ONE scalar needs <= 15%: "
              + ('supported' if spread <= 0.15 else 'contradicted'))
    print("=" * 118)


# ────────────────────────────────────────────────────────────
# Synthetic self-test
# ────────────────────────────────────────────────────────────
def simulate(I, C, b, mgl, th_eq, level, vtarget, coast_time,
             dt=0.0025, jitter=0.0003, vnoise=0.02, pnoise=0.0005,
             seed=0):
    rng = np.random.default_rng(seed)
    h = 1e-4

    def acc(thm, vm, tau):
        return (tau - C * np.tanh(vm / 0.01) - b * vm
                - mgl * np.sin(thm - th_eq)) / I

    th, v = 0.0, 0.0
    t, phase = 0.0, PH_BURST
    ts, ps, vs, taus, phs = [], [], [], [], []
    t_next = 0.0
    t_coast0 = None
    while True:
        tau = level if phase == PH_BURST else 0.0
        # RK4 microstep
        k1v = acc(th, v, tau); k1p = v
        k2v = acc(th + 0.5*h*k1p, v + 0.5*h*k1v, tau); k2p = v + 0.5*h*k1v
        k3v = acc(th + 0.5*h*k2p, v + 0.5*h*k2v, tau); k3p = v + 0.5*h*k2v
        k4v = acc(th + h*k3p, v + h*k3v, tau); k4p = v + h*k3v
        th += h * (k1p + 2*k2p + 2*k3p + k4p) / 6
        v += h * (k1v + 2*k2v + 2*k3v + k4v) / 6
        t += h
        if phase == PH_BURST and (abs(v) >= vtarget or t > 2.0):
            phase = PH_COAST
            t_coast0 = t
        if phase == PH_COAST and t - t_coast0 > coast_time:
            break
        if t >= t_next:
            ts.append(t + rng.normal(0, jitter))
            ps.append(th + rng.normal(0, pnoise))
            vs.append(v + rng.normal(0, vnoise))
            taus.append(tau)
            phs.append(phase)
            t_next += dt
    idx = np.argsort(ts)
    return (np.asarray(ts)[idx], np.asarray(ps)[idx], np.asarray(vs)[idx],
            np.asarray(taus)[idx], np.asarray(phs)[idx])


def simulate_pingpong(I, C, b, mgl, th_eq, level, window,
                      traversals=6, dt=0.0033, jitter=0.0003,
                      vnoise=0.02, pnoise=0.0005, seed=0):
    """Constant-tau drive flipping sign at +-window — the pingpong
    protocol of coast_down_bench.py. All samples ph=PH_BURST."""
    rng = np.random.default_rng(seed)
    h = 1e-4

    def acc(thm, vm, tau):
        return (tau - C * np.tanh(vm / 0.01) - b * vm
                - mgl * np.sin(thm - th_eq)) / I

    th, v, t = 0.0, 0.0, 0.0
    tau, flips = level, 0
    ts, ps, vs, taus, phs = [], [], [], [], []
    t_next = 0.0
    while flips < traversals and t < 60.0:
        k1v = acc(th, v, tau); k1p = v
        k2v = acc(th + 0.5*h*k1p, v + 0.5*h*k1v, tau); k2p = v + 0.5*h*k1v
        k3v = acc(th + 0.5*h*k2p, v + 0.5*h*k2v, tau); k3p = v + 0.5*h*k2v
        k4v = acc(th + h*k3p, v + h*k3v, tau); k4p = v + h*k3v
        th += h * (k1p + 2*k2p + 2*k3p + k4p) / 6
        v += h * (k1v + 2*k2v + 2*k3v + k4v) / 6
        t += h
        if t >= t_next:
            ts.append(t + rng.normal(0, jitter))
            ps.append(th + rng.normal(0, pnoise))
            vs.append(v + rng.normal(0, vnoise))
            taus.append(tau)
            phs.append(PH_BURST)
            t_next += dt
        if tau > 0 and th > window:
            tau, flips = -level, flips + 1
        elif tau < 0 and th < -window:
            tau, flips = level, flips + 1
    idx = np.argsort(ts)
    return (np.asarray(ts)[idx], np.asarray(ps)[idx], np.asarray(vs)[idx],
            np.asarray(taus)[idx], np.asarray(phs)[idx])


def simulate_ballistic(I, C, b, mgl, th_eq, level, sign, travel=0.45,
                       burst_t=0.30, vmax=12.0, vtarget=10.0,
                       kp=6.0, kd=1.5,
                       clip=17.0, brake_s=1.0, dt=0.0033,
                       jitter=0.0003, vnoise=0.02, pnoise=0.0005,
                       lash=0.015, I_rotor=8.5e-3, start_off=0.0,
                       seed=0):
    """M3.1-envelope protocol of coast_down_bench --mode ballistic:
    short tau burst -> logged PD brake to home. Logged tau is ff only
    (0 during brake), matching the bench; the fit must reconstruct.

    lash/I_rotor: burst-onset engagement transient observed on real
    hip_pitch (2026-08-17): the rotor first accelerates ALONE through
    gear lash at tau/I_rotor, then engages and dumps momentum into the
    leg (v *= I_rotor/I). The fit's onset mask must make recovery
    robust to it."""
    rng = np.random.default_rng(seed)
    h = 1e-4

    def acc(thm, vm, tau):
        return (tau - C * np.tanh(vm / 0.01) - b * vm
                - mgl * np.sin(thm - th_eq)) / I

    th, v, t = 0.0, 0.0, 0.0
    ts, ps, vs, taus, phs = [], [], [], [], []
    t_next = 0.0
    phase = PH_BURST
    t_brake0 = None
    th = start_off
    engaged = lash <= 0
    lash_travel = 0.0
    while True:
        if phase == PH_BURST:
            tau = sign * level
        else:
            tau = float(np.clip(kp * (0.0 - th) - kd * v, -clip, clip))
        if not engaged:
            v += h * tau / I_rotor
            th += h * v
            lash_travel += abs(v) * h
            if lash_travel >= lash:
                engaged = True
                v *= I_rotor / I          # momentum onto combined body
        else:
            k1v = acc(th, v, tau); k1p = v
            k2v = acc(th + 0.5*h*k1p, v + 0.5*h*k1v, tau)
            k2p = v + 0.5*h*k1v
            k3v = acc(th + 0.5*h*k2p, v + 0.5*h*k2v, tau)
            k3p = v + 0.5*h*k2v
            k4v = acc(th + h*k3p, v + h*k3v, tau); k4p = v + h*k3v
            th += h * (k1p + 2*k2p + 2*k3p + k4p) / 6
            v += h * (k1v + 2*k2v + 2*k3v + k4v) / 6
        t += h
        if t >= t_next:
            ts.append(t + rng.normal(0, jitter))
            ps.append(th + rng.normal(0, pnoise))
            vs.append(v + rng.normal(0, vnoise))
            taus.append(sign * level if phase == PH_BURST else 0.0)
            phs.append(phase)
            t_next += dt
        if phase == PH_BURST and (t > burst_t or abs(th) > travel
                                  or abs(v) >= vtarget or abs(v) > vmax):
            phase = PH_BRAKE
            t_brake0 = t
        if phase == PH_BRAKE and t - t_brake0 > brake_s:
            break
    idx = np.argsort(ts)
    return (np.asarray(ts)[idx], np.asarray(ps)[idx], np.asarray(vs)[idx],
            np.asarray(taus)[idx], np.asarray(phs)[idx])


def selftest():
    cases = [
        dict(name='hip-like (heavy b)', I=22e-3, C=0.05, b=0.20,
             mgl=1.5, vt=10.0),
        dict(name='knee-like (light b)', I=15e-3, C=0.13, b=0.045,
             mgl=0.8, vt=10.0),
        dict(name='ballistic-like I', I=43e-3, C=0.05, b=0.10,
             mgl=2.0, vt=10.0),
    ]
    # pingpong protocol with the MEASURED hip_pitch character
    # (overdamped: zeta >= 1; the case the coast protocol cannot excite)
    pp_cases = [
        dict(name='pingpong overdamped hip (measured params)',
             I=20e-3, C=0.146, b=0.217, mgl=0.5,
             levels=[0.5, 0.8, 1.2, 1.6], window=0.5, vmin=0.3),
        dict(name='pingpong roll-like narrow window',
             I=31e-3, C=0.11, b=0.06, mgl=0.8,
             levels=[0.25, 0.40, 0.60], window=0.18, vmin=0.15),
    ]
    ok_all = True
    for c in cases:
        samples = []
        for i, (lvl, sgn) in enumerate([(2, 1), (3, -1), (4, 1),
                                        (2, -1), (3, 1), (4, -1)]):
            t, p, v, tau, ph = simulate(c['I'], c['C'], c['b'], c['mgl'],
                                        0.0, sgn * lvl, c['vt'], 6.0,
                                        seed=i)
            s = prep_rep(t, p, v, tau, ph)
            if s:
                samples.append(s)
        print(f"\n=== selftest: {c['name']} "
              f"(planted I={c['I']*1e3:.1f}e-3, C={c['C']}, b={c['b']}, "
              f"mgl={c['mgl']}) ===")
        out = report(samples, 1.0, vmin=0.3)
        eI = abs(out['I'] - c['I']) / c['I']
        eC = abs(out['C'] - c['C']) / c['C']
        eb = abs(out['b'] - c['b']) / c['b']
        eA = abs(out['A'] - c['mgl']) / c['mgl']
        ok = eI < 0.05 and eC < 0.10 and eb < 0.10 and eA < 0.05
        ok_all &= ok
        print(f"  recovery errors: I {eI*100:.1f}%  C {eC*100:.1f}%  "
              f"b {eb*100:.1f}%  mgl {eA*100:.1f}%  -> "
              f"{'PASS' if ok else 'FAIL'}")
    for c in pp_cases:
        samples = []
        for i, lvl in enumerate(c['levels']):
            t, p, v, tau, ph = simulate_pingpong(
                c['I'], c['C'], c['b'], c['mgl'], 0.0, lvl,
                c['window'], seed=100 + i)
            s = prep_rep(t, p, v, tau, ph)
            if s:
                samples.append(s)
        print(f"\n=== selftest: {c['name']} "
              f"(planted I={c['I']*1e3:.1f}e-3, C={c['C']}, b={c['b']}, "
              f"mgl={c['mgl']}) ===")
        out = report(samples, 1.0, vmin=c['vmin'])
        eI = abs(out['I'] - c['I']) / c['I']
        eC = abs(out['C'] - c['C']) / c['C']
        eb = abs(out['b'] - c['b']) / c['b']
        eA = abs(out['A'] - c['mgl']) / c['mgl']
        ok = eI < 0.05 and eC < 0.10 and eb < 0.10 and eA < 0.10
        ok_all &= ok
        print(f"  recovery errors: I {eI*100:.1f}%  C {eC*100:.1f}%  "
              f"b {eb*100:.1f}%  mgl {eA*100:.1f}%  -> "
              f"{'PASS' if ok else 'FAIL'}")
    # ballistic protocol (M3.1 envelope) WITH the real-hardware lash
    # engagement transient AND the real pose's gravity slope (~3 Nm/rad)
    # planted. Reps 3-4 per level start offset and burst through home —
    # without them b and the gravity slope are degenerate (v~p locked in
    # one-way bursts; the 2026-08-17 real fit returned b=-0.15).
    bal_cases = [
        dict(name='ballistic hip: I 40e-3, small b', I=40e-3, C=0.13,
             b=0.05, mgl=3.1, btol=0.03),
        dict(name='ballistic hip: I 40e-3, heavy b', I=40e-3, C=0.13,
             b=0.217, mgl=3.1, btol=None),
        dict(name='ballistic small-I 20e-3, heavy b', I=20e-3, C=0.146,
             b=0.217, mgl=3.1, btol=None),
    ]
    for c in bal_cases:
        samples = []
        i = 0
        T = 0.45
        for lvl in (1.0, 1.5, 2.0, 3.0, 4.0):
            for sign, off in ((1, 0.0), (-1, 0.0),
                              (1, -0.55 * T), (-1, 0.55 * T)):
                t, p, v, tau, ph = simulate_ballistic(
                    c['I'], c['C'], c['b'], c['mgl'], 0.0, lvl, sign,
                    start_off=off, seed=200 + i)
                i += 1
                s = prep_rep(t, p, v, tau, ph, window=9,
                             brake=(6.0, 1.5, 0.0, 17.0))
                if s:
                    s['home'] = 0.0
                    s['is_bal'] = True
                    samples.append(s)
        print(f"\n=== selftest: {c['name']} "
              f"(planted I={c['I']*1e3:.1f}e-3, C={c['C']}, b={c['b']}, "
              f"mgl={c['mgl']}) ===")
        fb = report_ballistic(samples, vmin=0.3)
        eI = abs(fb['I'] - c['I']) / c['I']
        eC = abs(fb['C'] - c['C']) / c['C']
        if c['btol'] is not None:
            ok_b = abs(fb['b'] - c['b']) < c['btol']
            b_txt = f"b abs err {abs(fb['b']-c['b']):.3f} (tol {c['btol']})"
        else:
            ok_b = abs(fb['b'] - c['b']) / c['b'] < 0.15
            b_txt = f"b {abs(fb['b']-c['b'])/c['b']*100:.1f}%"
        eg = abs(-fb['g1'] - c['mgl']) / c['mgl']
        ok = eI < 0.05 and eC < 0.15 and ok_b and eg < 0.15
        ok_all &= ok
        print(f"  recovery errors (burst-only): I {eI*100:.1f}%  "
              f"C {eC*100:.1f}%  {b_txt}  mgl {eg*100:.1f}%  -> "
              f"{'PASS' if ok else 'FAIL'}")
    verdict = ('ALL PASS' if ok_all
               else 'FAILURES — do not trust real-data fits until fixed')
    print(f"\n[selftest] {verdict}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('npz', nargs='*')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--vmin', type=float, default=0.3,
                    help='|v| mask for the sign(v) regressor')
    ap.add_argument('--sg-window', type=int, default=None,
                    help='SG window override (odd); default 9 for '
                         'ballistic files, 21 otherwise')
    ap.add_argument('--pinned-inertia', action='store_true',
                    help='re-fit with the rotor pinned at the vendor 4.2e-3 '
                         '(+ URDF link) under the four pre-registered '
                         'hypotheses; one table row per file x hypothesis')
    ap.add_argument('--pinned-hyp', default='all',
                    choices=('all', 'friction', 'tscale', 'delay', 'belt'))
    ap.add_argument('--de-iters', type=int, default=40,
                    help='H-belt DE generations (smoke test: 3)')
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.npz:
        raise SystemExit('give npz files or --selftest')
    if args.pinned_inertia:
        run_pinned(args)
        return
    samples, motor = load_reps(args.npz, args.sg_window)
    print(f"[fit] motor {motor}: {len(samples)} reps loaded")
    if any(s.get('is_bal') for s in samples):
        report_ballistic(samples, args.vmin)
        print("\n[fit] ---- global fit incl. reconstructed brake "
              "(CROSS-CHECK ONLY: PD-model assumption + stick-slip "
              "return regime) ----")
    report(samples, K_JOINT[motor], args.vmin)
    print("\n[fit] NOTE: I here is the full swing inertia about the motor "
          "axis in THIS pose/config — compare against ballistic and "
          "kp-sweep numbers taken in the same condition.")


if __name__ == '__main__':
    main()
