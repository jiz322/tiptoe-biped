# P11 Hardware-Cycle Runbook — K-line checkpoint deployment

**Written 2026-08-20, while K_r400 (iter ~994, active_frac 0.378) and K2
(iter ~578, 0.44) are still descending. Execute when a K-line checkpoint
converges (pickup/active_frac stable in 0.20–0.27 band).**

**What changed since the last flight:** the P7 replant (armature ×4.6 +
belt off-diagonal + measured frictions) and the K servo gains
(joint-space kp 20/30, kd 1.4/1.5 on hip_pitch/knee; roll unchanged
6/0.3). The K gains are a **deployment-contract change** — motor-side
gains and the REAL_KP mapping move. Every prior flight flew policies
trained on a plant that believed itself 2× faster than the hardware was;
this is the first generation where sim, bench, and (pending checks
below) firmware agree.

---

## 0. Desk phase — before touching the robot (no hardware needed)

- [x] **0.1 Pick the checkpoint.** Converged K-line model (active_frac
  ≤0.27 stable ≥100 iters, no crash within trailing 50 iters). Record
  run dir + model number here: **K4b_3600 primary**
  (`p9_replant/2026-08-23_08-56-43_K4b_resume3420/model_3600.pt`,
  user-judged best gait; hw-matched 15 s survival 1.000/1024; NOT yet
  exported — run `control/export_policy.py` on it), **K4_3420 backup**
  (strongest at armature ×1.3 corner + full DR; already exported),
  K4b2_3460 second backup, K3s_3360 fallback line (NOT 3380+ — critic
  collapse starts at 3380).
  Full selection evidence: [p9-checkpoint-selection-20260823.md](p9-checkpoint-selection-20260823.md).
- [x] **0.2 Sim replica gate** (the P8 gate, K edition) — **PASS 2026-08-23**,
  executed via `eval.py --scenarios hw_matched hw_harsh_arm13` (DR pinned to
  central measured values / armature ×1.3, K gains re-passed): K4_3420
  survives 15 s in 99.8 % of 1024 trials hw-matched, 99.7 % at the harsh
  corner; all K candidates ≥99.0 %. Same plant kills G2i_4760 in 91 % of
  trials (median 2.3 s) — replicates the hardware failure. The original
  `_launch_match_sim` phase-0-launch variant remains optional for margin.
- [x] **0.3 Deploy-contract update in policy.py / config.py: DONE 2026-08-24.**
  Every constant re-derived from the K4b run's frozen `env.yaml` (commit
  `58c5d17`): REAL_KP/KD 6/20/30 · 0.3/1.4/1.5, KNEE_DECOUPLE_FF=True
  (comment de-inverted), G² applied once (verified), scales confirmed.
  Beyond the list below, the same sweep caught and fixed: REAL_EFFORT
  [1.8/6/8.18]→[6/17/23.2] (sim effort_limit; old row = 0.30 rad error
  clamp at kp 20), STANDUP_GRACE 5→4 s, YAW_REF_RESAMPLE 100→1000,
  KNEEL_HEIGHT 0.2188→0.234. Confirmed action LPF α=0.6 is IN the K sim
  plant (`action_lpf_alpha`, rl_env_p6 pre-delay-buffer) — bridge stays
  0.6, do not zero. balance_log.py: POLICY_PATH→policy_K4b3600.pt
  (export verified bit-exact vs checkpoint, 3-way), 'k' hotkey slews
  gains exactly onto this contract. Normalizer baked in JIT; live
  obs-parity re-check scheduled on the first flight log of the day.
  - REAL_KP/REAL_KD: hip_pitch **20 / 1.4** (direct-drive, joint=motor),
    knee **motor-side 16.1 / 0.81** (= joint 30/1.5 × (22/30)²),
    hip_roll unchanged 6 / 0.3.
  - Verify `motor_bridge._sim_to_motor_gains` applies G² exactly once —
    the knee number above is ALREADY motor-side; do not double-convert.
    (The stale `REAL_KP=16` constant from the 08-14 audit: this cycle
    retires it — delete, don't update.)
  - Action scales unchanged (0.3 / 0.75 / 1.025) — K deliberately kept
    them; confirm policy.py agrees.
  - Normalizer keys: re-run the obs-parity check
    (`scripts/rsl_rl/obs_parity_live.py`) against the NEW checkpoint —
    the pre-08-06 raw-obs bug must stay dead.
  - **KNEE_DECOUPLE_FF = True** (config.py). Rule: the hardware flag at
    flight time must equal the sim `knee_decouple_ff` the flying policy
    trained under — K-line trained True; hardware was reverted to False
    in the 08-17 pre-replant snapshot (`500032b`). Flying False re-arms
    the quantified G2i mismatch (~21–23% unmodeled knee drag, equiv.
    damping mult 0.70 at the DR floor; ×3 absolute at K kd 1.5). Also
    rewrite the stale "double-compensates" comment above the flag — it
    is inverted and likely caused the 08-17 revert. See
    [knee-hip-coupling-review-20260821.md](knee-hip-coupling-review-20260821.md) §5.
- [x] **0.4 IMU trim decision: DONE 2026-08-24** — verified −0.089 in
  config.py and recorded the decision + rationale in the comment block.
  IMU_PG_OFFSET stays −0.089 (P9/P10
  trained with dr_proj_grav_bias_max=0.10 which covers the ~0.057
  over-correction). Do NOT retrim before flight — the policy was
  trained against this exact bias family.
- [x] **0.5 DONE 2026-08-24** — owner set 4000 on all 6 motors via vendor
  GUI, EL05 50 µs scaling confirmed in the GUI. Fallout handled: (a) all
  deploy resume paths audited — they re-enable (arm→enable_all; stalls
  disarm first); (b) startup gaps >200 ms in balance_log/test_motors
  hardened with idle-frame feeds (commit `3800bfd`); (c) the 08-16
  reference baseline JSON's fw section patched to 4000 with a provenance
  note (vendor CSV export still broken), original kept as
  `*.bak_pre_cantimeout`. Original item:
  **Arm the firmware comms watchdog (CAN_TIMEOUT), all 6 motors.**
  08-15/08-17 dumps: `CAN_TIMEOUT = 0` everywhere — watchdog disabled, so
  a host death (SIGKILL, Pi freeze, CAN2SPI wedge) leaves motors servoing
  the last MIT command at full kp forever. Every existing safety layer
  (controller.py watchdog/tilt/temp/CAN-drop) dies with the process; this
  register is the only one that doesn't. Vendor doc: non-zero → motor
  enters **reset mode** (limp — safe on the gantry) after that long
  without a CAN command; units 50 µs, `20000 = 1 s`.
  - Set **4000 (= 200 ms)** — mirrors WATCHDOG_TIMEOUT_S; 20 missed
    ticks at the 100 Hz per-motor rate, way beyond nominal jitter
    (tick busy p95 2.45 ms). Vendor GUI, per motor (CSV export broken).
  - Confirm in the GUI that the EL05 firmware (app 1.0.5.0.1) uses the
    same 50 µs scaling — the doc quote is from the RS01 manual.
  - Sysid-neutral: never fires in nominal operation, touches no
    control-loop/filter/limit register.
  - Fallout: (a) driver must re-**enable** a motor that timed out into
    reset before resuming MIT streaming — "stopped replying then came
    back" ≠ resume; (b) bench scripts that enable motors then pause
    >200 ms will now drop motors into reset — expected, not a fault;
    (c) refresh the `can_baseline_bench.py` FW_WHITELIST reference dump
    after the change, or the next baseline flags a fw regression.

## 1. Bench phase — one session, tether rig, user drives motion

Order matters: 1.1 can veto the flight; 1.2–1.3 are opportunistic.

- [x] **1.1 kp-sweep at deploy gains (the gate). PASS (2026-08-22).**
  Batteries `data/step_dep_10{1,2,4,5}.npz` (step 0.12 / travel 0.22 —
  torso-clearance-limited; ζ and f amplitude-independent), quick-look
  `data/_p11_step_quicklook.py`. Hip fundamentals 3.06–3.34 Hz (in
  band); knee shows NO second mechanical mode — spectra second peak is
  exactly 2×f₁ (settle harmonic) on all four motors, no buzz/squeal.
  ZERO overshoot everywhere — over-damped, and the desk check says
  that's CORRECT for the measured plant at this amplitude (kp·0.12 =
  2.4 Nm vs C 0.146: viscous-only ζ≈0.9, Coulomb quenches the rest;
  ss-err −2..−6% ≈ C/kp stiction band; the 3.0–3.2/ζ0.6–0.75 gate
  numbers were friction-free predictions). Training plant carries the
  same measured C/b/armature → sim and hw agree at small signal.
  kd-noise: hip 56.8→75.5 mNm std (×1.33 vs ×2.8 linear bound), knee
  47.1→50.6 (×1.08 vs ×1.6). No chatter, temps flat. **Flight is GO
  from the servo side.**
- [x] **1.2 Braced-torso ballistic. DONE (2026-08-22) — jump present at
  the predicted low edge.** `data/bal_101_braced.npz`, pillar-press
  brace (imperfect — contact broke on harder kicks; late reps' residual
  rms climbs 13→41 rad/s², so naive per-burst medians are the
  instrument, not the global fit: R² 0.45, unmodeled intermittent
  constraint). Naive median I: hanging 39.0/40.0e-3 (bal2/bal3) →
  braced 42.4e-3 = **+2.9e-3 (~2.5σ)** vs predicted +3–6e-3. Leaky
  bracing only biases LOW (contact-broken moments look like hanging),
  and the largest level fit highest (44.8e-3) — opposite of base-recoil
  contamination. **Verdict: tether-bias confirmed at the floor; RS01
  rotors ≈19e-3 both; P11 retrain recenters hip armature ~38–39e-3
  (inside DR, training-quality item, not a flight blocker).**
- [x] **1.3 Knee ballistic `--hold 101 --hold-kp 40`. DONE (2026-08-22)
  — knee inertia REAL.** `data/bal_102_hipclamped.npz` vs 08-17
  `bal_102_belt_on.npz`: burst-only primary 20.7→22.5e-3 (no drop
  toward 15; slight rise), naive per-burst rep-for-rep identical,
  R² 0.986 both. Hip-yield coupling ruled out; knee armature 19–21e-3
  motor-side stands, inside DR.

## 2. Flight phase — launch protocol

- [x] **2.1 Bring-up guard: PASS 2026-08-24.** Reply-ids 6/6 distinct
  (0x065–0x06A). Decode spot-check via `vel_decode_check` lock-in
  (estimator rewritten this day — the old integral fit was biased high
  by vel-DC drift, false-failing healthy motors; see commit `70185df`):
  101 k=0.995/0.988 (0.49/1.25%), 103 k=0.988 (1.18%), naive/detrended/
  lock-in estimators mutually consistent. Gate npz: `vel10{1,3}_gate.npz`
  on the pi.
- [x] **2.2 DONE 2026-08-24** — 11 tick-0 logs captured (9 trials + PD-hold
  probe + mat trial), `data/p11_flight_20260824/` with analysis scripts.
  Addendum for next session: **camera on every trial** (the one video
  taken was the most interpretable artifact of the day), and balance_log
  now has 'l' = log-anywhere for bench probes.
- [x] **2.3 DONE 2026-08-24** — 9 trials: T1 hanging+held (calm 1.4 Hz
  pedal), T2–T4+T9 clean ground starts (all fell pitch-forward,
  |pg_x|>0.25 at 0.14–0.26 s), T5 'b'-during-pickup entry (**best: 2.35 s**
  with bidirectional recoveries), T6–T8 grounded hand-assisted (never
  self-stable; −0.8 pg excursions caught by hand). Protocol note: T5's
  policy-live-descent entry is closest to training — use it next time.

  **CORRECTED 08-24 eve (`_reflect_20260824.py`, knee-load segmentation):**
  T5's 2.35 s is log length, not free survival. Knee tau_meas ≈ 0 (hand
  carrying the weight) until **1.70 s**; the pg_x +0.28 "recovery" at
  0.28 s happened in-hand; free-standing ≈ **0.05 s**, and the fall was
  BACKWARD at the put-down (opposite sign to T2–T4's forward falls → the
  manual plant likely seeded it). Best honest free stand of the day =
  **T9 ≈ 0.17 s** (load onset → |pg_x|>0.25). Protocol addendum: the
  b-during-pickup entry is still the right one, but imitate the trained
  put-down — lower over ~1 s, release at ground contact, no push (training
  lowers in 0.8 s and releases on contact at 2.4 cm + 50–400 ms latency).
  Quote grounded-time (quicklook now prints it), never log length.
- [x] **2.4 DONE 2026-08-24** — no tripwires fired: temps ≤39 °C, no
  squeal/smell; PD-hold stance probe at K gains clean (see verdict).

## 3. Reading the result (pre-registered, to keep us honest)

| Outcome (majority of ≥5 trials) | Reading | Next move |
|---|---|---|
| Stands ≥5 s, catch-steps | Plant story confirmed end-to-end | Push duration; start disturbance tests; celebrate |
| Stands 1–5 s, falls recoverable-looking | Servo fix real; residual plant gap (belt lash/compliance is the top unmodeled candidate) | Feed fall logs to replica gate; fit the residual |
| Falls ~0.25 s pitch-first, same signature as G2i | K did NOT change the failure → the fall driver is still unmodeled AND upstream of gains | The 0.25 s determinism becomes the key clue: look at launch transient + contact, the two things no bench test excites |
| Falls, but NEW signature (different axis/time/character) | Old failure mode fixed; new one exposed | New fall logs → replica → iterate; this is progress |
| Oscillation/chatter/instability | Deploy-gain mapping error or belt resonance missed at bench | Kill immediately; audit 0.3 + 1.1 before any retry |

The third row is the pessimism case. Note it requires the fall to
survive: corrected plant + restored servo + clean contract + verified
obs — the list of things it could still be is now short (launch
transient, contact/foot geometry, belt small-signal lash). Even that
outcome, with logs from 2.2, feeds the replica gate a discriminating
dataset the campaign has never had.

## VERDICT (2026-08-24): mixed rows 3+4 — and the residual is the gait, not the plant

Observed: initial departure keeps G2i's clock (pitch-forward,
|pg_x|>0.25 at 0.14–0.26 s) = row 3; but the character is new — active
catch-stepping, near-bang-bang corrections, one 2.35 s fight with
recoveries (corrected: 1.7 s of it in-hand, ~0.05 s free — see 2.3;
the row-4 "new character" call stands on T2–T4/T9's catch-stepping and
the mat-run video, not on T5's duration), where G2i tipped passively
= row 4.

What the day's probes eliminated, in order:
- **Contract**: in-flight logs show exact K gains/α/effort; policy on pi
  bit-exact vs checkpoint on logged obs (2e-6); obs statics in-dist.
- **Decode**: lock-in gate PASS both families (k=0.988–0.995; the old
  integral-fit FAILs were instrument bias, fixed in `70185df`).
- **Servo/drivetrain**: PD-hold stance probe at K gains, grounded, with
  fore-aft load reversals — joint velocities at encoder noise floor,
  spectra flat, no lash bursts, no ringing. **Belt small-signal lash
  de-prioritized.** Static stance torques measured: hip ±1.1–1.7 Nm,
  knees ≈0.
- **Sim twin** (`eval.py --dump_traj`, hw_matched, 64 envs): K4b survives
  15 s at 100% **while bounding** — flight_fraction 0.22, base z
  0.41–0.72 m, pg_x rocking ±0.44, knee torque ±12 Nm motor-frame at
  2.5 Hz. Hardware runs the same gross envelope with 2× action
  saturation and 5.6–17 Hz action content (sim: clean 2.5 Hz).

Reading: **the trained idle is a hot bounding march with thin stability
margin; real contact (satin tile + TPU point feet — slick enough that a
yoga mat skated on it; slip un-measured) plus per-step compounding tips
it over, and the deterministic policy amplifies the fast motion it
creates.** All 13 evaluated K checkpoints incl. K5/K5b polish share the
envelope — checkpoint swaps are not a lever.

### P12 queue (ranked)

1. **Calm-idle retrain**: swing_clearance_zero 0.078 → ~0.03–0.04 (stays
   above the shuffle floor — anti-shuffle rule intact), penalize flight
   ticks + |base_vz| at idle. Sim acceptance gate before any flight:
   survival ≥ K4b AND flight_fraction <0.05 AND pg_x std <0.1
   (measure via `--dump_traj`). Keep 2.5 Hz and the K servo contract
   unchanged — one variable at a time.

   **CORRECTED at arm-authoring (08-24 late, `launch_train_L1_calmidle.ps1`):**
   the 0.03–0.04 zero is wrong — K4b TB reads settled idle_toe_z_p10 =
   0.0735, i.e. resting toe z in the reward frame is ~0.073, so that zero
   would sit 3–4 cm *below* rest and pay swing reward to a grounded foot
   (the exploit the zero-rule exists to prevent). Zero stays **0.078**;
   the calm levers are `swing_clearance_target_m` 0.04→**0.015** (credit
   saturates at rest+18 mm — kills the lift race, steepens the liftoff
   gradient), `no_flight` −5→**−15**, `vz_penalty` −2→**−8**. Gate
   protocol fix: the "flight 0.22 / base z 0.41–0.72 / pg_x ±0.44"
   envelope above was **pickup-ride contamination** (eval leaves the
   pickup curriculum on; settled K4b actually reads z std 0.0055 @ 5 Hz,
   pg_x std 0.035, |a|>0.95 on 8–13 % of ticks — the sim idle is far
   calmer than hardware's, sharpening the contact-texture reading). Gate
   runs pickup-free (`pickup_initial_probability=0.0
   pickup_probability=0.0` + K_OVR + `hip_pitch_armature=0.0385`);
   thresholds in the L1 script header.
2. **Floor friction measure** (5 min desk item): drag a limp foot across
   the tile with a spring scale, μ ≈ F/W. Optionally taped-mat A/B.
3. **Launch protocol**: hand over during hang, lower with policy live
   (T5-style), tether slack as catch.
4. Belt lash bench-ID: only if a calm-gait policy still shows the
   hardware-only 5–17 Hz action content.
5. Armature recenter (hip ~38–39e-3, from bench 1.2) — fold into the
   P12 retrain (was already queued as training-quality item).
   **FOLDED 08-24**: new cfg knob `env.hip_pitch_armature` (default
   33.2e-3 = old plant bit-for-bit); L1 passes 0.0385. Eval jobs for L
   checkpoints must re-pass it (K_OVR doesn't carry it).

## Deferred / explicitly out of scope this cycle

- Supine IMU check — physically impossible (Pi on the torso back).
- Straightedge balance — needs CAD fixtures.
- hip_roll gain redesign — roll inertia unresolved on this rig; falls
  are pitch-initiated; revisit only on lateral-signature evidence.
- Armature recenter retrain (hip ~38–39e-3) — pending 1.2's verdict;
  P11 training item, not a flight blocker (inside current DR).
