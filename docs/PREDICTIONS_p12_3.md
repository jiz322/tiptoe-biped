# PREDICTIONS_p12_3 — pre-registered hardware A/B: handover contract (sighted vs blind)

Registered 2026-08-29, before any p12.3 hardware trial. Policy: r1r_3150
(raw rsl_rl checkpoint, loaded via policy.py checkpoint mode; offline replay
parity vs the 08-28 hardware logs = 1e-6, so the prediction pipeline is the
deployed pipeline).

Protocol (user-specified): same session, same mat, alternating arms b/v/b/v,
>=5 trials per arm, logging on for all (npz `handover` field tags the arm),
spotter on for the first blind trials, crawl-pose knee check between trials.

## Background (one paragraph)

Training episodes start BLIND: the env zeroes the obs-delay buffer at reset,
so the first policy frame has obs[0:27] = 0 (pg, ang_vel, lin_vel, jpos, jvel,
last_actions) with only the bookkeeping tail real, primed x5 into history.
The historical deploy handover ('b') instead feeds the real sensor frame x5 —
a state training never produces — and the policy answers it with a violent
torso-raise opening that pitched the base backward ~17 deg within 130 ms in
all 7 trials of 08-28, ending in 7/7 backward falls. 'v' (new, 08-29) hands
over the training contract.

## Arm 'b' (sighted — the 08-28 behavior), predicted signature

- tick-0 action ~ [-0.4+/-0.15, +0.87+/-0.06, +0.67+/-0.06, -0.05+/-0.10,
  <= -0.90 (R_hip at/near saturation), +0.15+/-0.06]  (envelope of the 7
  trials of 08-28).
- Violent symmetric torso raise from tick 0; knee-bend hop with free flight
  ~[0.07-0.13]; landing impact ~0.13.
- Raw pg_x (logged - 0.089): ~ -0.05 at 0.05 s, -0.21+/-0.02 at 0.10 s,
  -0.29+/-0.03 at landing.
- Outcome prediction: backward fall <= 1.2 s (7/7 precedent).

## Arm 'v' (blind — training contract), predicted signature

tick-0 action computed offline on the hw-exact blind frame (init_h 0.4401):

| joint  | a0     | target rad (deg) | first LPF step (deg) |
|--------|--------|------------------|----------------------|
| L_roll | +0.795 | +0.239 (+13.7)   | +5.5                 |
| L_hip  | -0.319 | -0.189 (-10.8)   | -2.6                 |
| L_knee | -0.434 | -0.445 (-25.5)   | -10.2                |
| R_roll | -0.421 | -0.126 (-7.2)    | -2.9                 |
| R_hip  | -0.176 | -0.182 (-10.4)   | -5.9                 |
| R_knee | +0.855 | +0.876 (+50.2)   | +20.1                |

- Physical: NO torso whip in the first 150 ms; a one-tick reflex — both knees
  bend (right ~2x deeper), hips extend slightly, left hip-roll swings out —
  then sighted corrections take over from tick 1 (LPF passes ~40% of the
  step in the first 10 ms).
- Sim blind arms: raw pg_x ~ 0 at 0.05-0.10 s, only -0.03..-0.12 at 0.13 s
  (transfer caveat: sim numbers, quoted as expectation bands not guarantees).
- Outcome prediction: stands, or at minimum clearly outlasts the sighted arm.

## Verification order on each trial (before any conclusion)

1. Check the npz `handover` field matches the intended arm.
2. Check logged tick-0 action against the arm's predicted vector. If the 'v'
   tick-0 action does NOT match the table above, the deploy blind frame is
   wrong (contract bug in balance_log.py) — fix before interpreting the
   trial. This check is the arm's identity test.
3. Only then read the outcome.

## Branch logic

- **'v' survives, 'b' falls** (>=4/5 each direction): handover contract
  confirmed as the deployable root cause. Adopt 'v' for p12.3 flights;
  proceed to the trim A/B (pin experiment predicted removal helps: all four
  trim-removed sim cells 0/10, including the only previously-falling cell);
  schedule training-side handover randomization for the next run.
- **'v' falls the same way** (with tick-0 identity check PASSED): the
  handover contract is real but not sufficient on the real plant. Escalate
  plant-side: torsional/patch contact (sim feet currently pivot with ZERO
  torsional friction — torsional_patch_radius unset everywhere; feed the toe
  drag test's a_eff into a sim arm), right-heavy com offset (~0.2 Nm hip-hold
  delta), 30-run rate grid.
- **Mixed / marginal**: n=5 per arm cannot resolve rates. The discriminator
  is the OPENING SIGNATURE (tick-0 action match + raw-pg_x-through-150 ms
  band), not the fall count. Extend trials only if signatures are clean.

## Standing caveat

The sim fall-rate cells behind this experiment are n=1-fall statistics
(deploy-handover x mu<=0.3: 1 fall in 10-11 episodes; every other cell 0/10).
Sim under-predicts the real fall rate (real: 7/7). Rates in sim are
directional evidence only; the hardware discriminator is the opening
signature. What sim DOES predict sharply — and what this A/B tests — is
which opening the policy executes as a function of the handover contract.
