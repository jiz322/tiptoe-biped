# Knee–Hip Belt Coupling — Design Review, Mini Cheetah Comparison, Sim2Real Risk Ranking

**Date:** 2026-08-21 · **Scope:** consolidated record of the coupling discussion: how this
codebase handles the belt-coupled knee, how MIT Mini Cheetah avoided the problem, a
quantified cost of our arrangement, which channels are live sim2real risks, and the full
`KNEE_DECOUPLE_FF` contract history (including the 08-17 revert and its verdict for the
replant bench).

---

## 1. The mechanism, and the four-layer treatment here

The knee motor sits on the **hip chassis** (does not rotate with the thigh) and drives the
knee through a 22:30 belt. Hip pitch motion therefore changes the knee joint angle with the
knee motor still:

```
q_knee = motor·G − c·q_hip        G = 22/30,  c = KNEE_HIP_COUPLING = 0.733
```

(c = G exactly — the planetary relation of a pulley whose housing stays fixed;
[config.py:87-101](../control/config.py#L87-L101)). Four layers handle it:

| Layer | What | Where |
|---|---|---|
| 1. Kinematic transform | Exact bidirectional map: reads subtract `c·q_hip`, targets add `c·hip_actual`, gains map by G² | [motor_bridge.py:111-140](../control/motor_bridge.py#L111-L140) |
| 2. Torque feedforward | `KNEE_DECOUPLE_FF`: knee `v_des = c·qd_hip·sign/G` so the firmware Kd tracks belt-forced motor spin instead of fighting it (~30% of knee cmd RMS); `HIP_REACTION_FF` implemented but **refuted** and off | [motor_bridge.py:219-245](../control/motor_bridge.py#L219-L245), [config.py:103-126](../control/config.py#L103-L126) |
| 3. Firmware-replica training plant | `BeltKneeDCMotor`: belt-drag Kd term, stale-hip latch (send-time hip baked into the knee target, held 10 ms), motor-velocity torque envelope, off-diagonal rotor inertial coupling `dM = (I_r/G²)·[[c²,c],[c,1]]` injected as a clamped plant force (PhysX armature is diagonal-only) | [belt_knee_actuator.py](../RL/source/RL/RL/tasks/direct/standup_curriculum/belt_knee_actuator.py), [rl_env_cfg_p6.py:244-292](../RL/source/RL/RL/tasks/direct/standup_curriculum/rl_env_cfg_p6.py#L244-L292) |
| 4. Per-term empirical validation | Teacher-forcing replay vs 7 hardware logs: replica reproduces the validated injection experiment to 1e-4; each term kept/rejected on replay evidence (drag ✓, stale-hip ✓, J^T hip reaction ✗ — injecting it made the hip match *worse* by +0.02 rad) | [reports/belt_coupling_sim2real/report.md](../reports/belt_coupling_sim2real/report.md) |

## 2. MIT Mini Cheetah: same remote-knee idea, opposite mounting choice

Mini Cheetah's knee actuator is also at the hip (coaxial), driving the knee through a belt
(extra ~1.55:1 over the 6:1 module → 9.33:1 total). The decisive difference: **the knee
module mounts on the hip actuator's output and co-rotates with the thigh**, so the knee
motor measures/drives the knee angle *relative to the thigh* directly. Hip motion never
enters the knee coordinate.

Their software proves it: in
[Cheetah-Software rt_spi](https://mit-biomimetics.github.io/d1/d62/rt__spi_8cpp.html) every
joint maps independently — `q_knee = (spine_q − offset)·knee_side_sign`, with
`side_sign = ±0.6429 = 1/1.555` (belt ratio folded into a constant). No hip term anywhere.
[MiniCheetah.h](https://github.com/mit-biomimetics/Cheetah-Software/blob/master/common/include/Dynamics/MiniCheetah.h)
holds only scalar gear ratios (6/6/9.33), and their `FloatingBaseModel` treats rotors as
geared bodies, so reflected + gyroscopic rotor terms come out natively.

They didn't solve the software problem better — **they chose a mechanical arrangement in
which the software problem doesn't exist.** Costs they accepted: knee actuator body spins
with the leg (negligible — it's coaxial with the hip axis), and the belt remains a
compliance/failure element (tensioners; clones ship spare belts — the one coupling risk both
robots share).

## 3. Quantified cost of the chassis-mount arrangement

Constants: c = 0.733, knee kd (joint) 1.0 current / 1.5 K-contract, kp 16 / 30 K-contract,
knee budget 8.18 Nm cont / 23.2 peak, vel limit 20 rad/s, knee reflected rotor 36.0e-3 kg·m².

| Channel | big_thigh | Mini Cheetah |
|---|---|---|
| Kinematic cross term | 0.733 rad/rad — hip-state error leaks 73% into knee estimate | 0 |
| PD belt drag | kd·c·qd_hip ≈ 0.73–1.10 Nm per rad/s of hip speed; measured ~30% of knee cmd RMS | 0 |
| Stale-hip phantom PD | kp·c·qd_hip·10 ms → 0.12–0.22 Nm per rad/s, 100 Hz sawtooth; ~0.5–0.9 Nm at fall-window hip speeds | 0 |
| Torque-envelope theft | belt adds c·qd_hip to knee motor speed → ~0.85 Nm headroom lost per rad/s of hip speed; ~3.4 Nm at 4 rad/s = 42% of continuous budget | 0 |
| Rotor inertial cross term | m_x = c·36e-3 = 26.4e-3 kg·m²; off/diag = 73%. On the 08-13 fall logs: median rms **0.96× the hip's own command RMS**, peaks 5–7.6 Nm at t = 0.16–0.20 s (fall initiation) | off/diag = 1/9.33 ≈ 11%, captured natively |
| J^T belt reaction on hip | ideal ≈ 35% of hip cmd RMS; hardware measurably doesn't feel the rigid version → a 0–35% model-uncertainty *band* | 0 by construction |
| Hip-axis armature penalty | reflected knee rotor c²·36e-3 = 19.3e-3 on the hip — 1.4× the hip's own 13.9e-3; coupling more than doubles hip armature (→33.2e-3) | ~0 |

Engineering-cost proxy: ~280-line custom actuator class, 7-log replay campaign, two config
flags with a pairing hazard (see §5), one live firmware patch, one training explosion
(iter-564, unclamped coupling injection) — vs Mini Cheetah's one constant per joint.

## 4. Sim2real risk ranking (which channels are live risks)

1. **KNEE_DECOUPLE_FF contract mismatch — ARMED for P11, high severity, one-line fix.**
   K/P9/P10 lineage trained with sim `knee_decouple_ff=True`
   ([rl_env_cfg_p6.py:245](../RL/source/RL/RL/tasks/direct/standup_curriculum/rl_env_cfg_p6.py#L245));
   hardware is currently `False` ([config.py:120](../control/config.py#L120)). Sim-True
   *assumes* the hardware FF is on — flying False re-creates the quantified G2i mismatch
   (§5): unmodeled knee drag ≈ 21–23% of knee PD torque, equivalent damping multiplier ~0.70
   sitting at the DR edge, and at K gains (kd 1.5) the absolute drag triples to
   ~1.10·qd_hip Nm. **Rule: hardware flag at flight time must equal the sim flag the flying
   policy trained under → True for any K-line flight.** Now in the P11 runbook §0.3.
2. **Rotor cross term parameters — high impact, medium probability.** Modeled since P7, but
   the bench hip-vs-knee rotor estimates disagree 40% while the armature DR band is ±30%;
   all bench numbers share the unknown commanded-τ scale; the FD-acceleration
   implementation is un-physical near contact. A 30–40% m_x error leaves a residual hip
   disturbance ~0.3–0.4× hip authority in the fall-initiation window.
3. **Hip-reaction refutation is regime-limited.** "Hip doesn't feel c·τ_knee" comes from
   *hang* logs. Under stance load the belt tension regime differs; if partially transmitted,
   up to 35% of hip cmd RMS of untrained, load-dependent disturbance appears exactly during
   weight-bearing. The open knee belt-off / loaded-bench item is the closing test.
4. **Belt two-mass resonance — unmodeled by construction, gated not solved.** Every layer
   assumes an inextensible belt. P11 kp-sweep bench gate exists precisely for this (knee
   ring ≈ 4.2–4.4 Hz; squeal or second ring = stop). K's kp 30 pushes toward the mode.
5. **Envelope theft — low.** Modeled on belt-effective velocity; 08-17 ballistic probe
   confirmed full 17 Nm delivery, current clamp dead; voltage DR covers sag.
6. **Stale-hip phantom — low.** Exact 5-substep latch replica; residual is timing jitter on
   a ≤0.9 Nm sawtooth (loop p95 2.45 ms since 08-16).
7. **Kinematic transform — lowest.** c = G is forced by tooth counts, not fitted. Real leak
   path is upstream hip *velocity decode* error entering at 0.733× (driver vel-decode check
   from the 08-14 audit still open).

Pattern: the physics channels are the well-closed ones; the danger has migrated into the
**contract surface** — sim flags, config flags, and firmware must agree tick-for-tick.

## 5. KNEE_DECOUPLE_FF — full contract history and the replant verdict

| When | Hardware flag | Sim flag (training) | Contract |
|---|---|---|---|
| 08-06 → 08-11 | **True** (patch live) | legacy rigid-knee sim (no belt) | consistent — FF makes plant ≈ rigid |
| 08-11 08:04 (`8ba262e`, "F test") | **False** | — | set for the F lineage |
| 08-13 flights (G2i4760) | **False** | **True** (`p8_belt_fresh/...G2.../env.yaml:237`) | **MISMATCH** — real, quantified: 0.3665·qd_hip Nm unmodeled knee drag, 21–23% of knee PD torque, equiv. damping mult 0.70/0.68 vs DR floor 0.7 |
| 08-13 21:45 (`b631c7a`) | **True** (the fix) | True | consistent. 12 trials: pitch departure 0.164±8 ms vs 0.161±15 ms — correct change, **not** the fall cause |
| 08-17 10:55 (`500032b`, pre-replant snapshot) | **False** (reverted) | True (frozen in the same commit) | mismatch re-armed — but no policy flew after |
| P11 (pending) | must be **True** | K-line trained True | — |

The [config.py:115-119](../control/config.py#L115-L119) comment ("running both
double-compensates … OFF for the p8_belt_fresh lineage") has the logic inverted — the
replica models the plant *with* the FF live, so sim-True **requires** hardware-True. That
comment plausibly caused the 08-17 revert and should be rewritten when the flag is flipped
(deploy-code edit — owner's call).

### Did False affect the 08-17 replant bench? **No — verdict: clean.**

The flag lives only in `motor_bridge.send_target()` (knee `v_des`), and its effect is
proportional to **hip velocity while the knee PD holds through the bridge**. Phase by phase:

- **P5 friction & inertia bench (the load-bearing phase — frictions, armature, rotor
  coupling all come from here):** `coast_down_bench.py` talks to the driver **directly in
  motor space** — the bridge code containing the flag never executes. Coast segments are
  torque-burst/passive (no PD on the swinging joint), and the fit reconstructs the brake PD
  from the known firmware law using *measured* motor velocity, so belt-forced spin is
  accounted, not hidden. The flag is physically incapable of touching these numbers.
- **P0 preflight, P2 filters, P6 mass/balance/init_h:** static, telemetry-only, or desk
  analysis — no motion through the bridge.
- **P1 loop timing:** PD holds/steps for timing measurement; hip stationary → `v_des ≡ 0`
  identically whether the flag is True or False.
- **P4 step-response sysid** (does use the bridge): single-joint steps with the other joint
  held. Knee steps: hip held; only the small belt yield (~0.73·τ/kp_hold rad, the known P7
  hold-kp effect) gives a brief sub-Nm drag transient vs multi-Nm test torques. Hip steps:
  drag lands on the *knee* motor and the hip doesn't feel it (refuted reaction path). P4's
  hip numbers were superseded by P5 anyway (act_lag phantom, b artifact).
- **P7/P8:** pure sim.
- The only flag-sensitive activity would have been a policy run through the bridge with the
  hip moving — none occurred between 08-17 and now (last flights 08-13, which flew under
  the mismatch and are analyzed above).

**Conclusion:** the 08-17 revert had zero effect on every replant number; it only matters
for the next flight, and is now a runbook §0.3 gate.

## 6. Sources

- [Mini Cheetah ICRA 2019](https://dl.acm.org/doi/abs/10.1109/ICRA.2019.8793865)
- [Cheetah-Software rt_spi docs](https://mit-biomimetics.github.io/d1/d62/rt__spi_8cpp.html) ·
  [MiniCheetah.h](https://github.com/mit-biomimetics/Cheetah-Software/blob/master/common/include/Dynamics/MiniCheetah.h)
- [Ben Katz build log](https://build-its-inprogress.blogspot.com/2019/03/hello-there-mini-cheetah.html) ·
  [clone teardown](https://build-its-inprogress.blogspot.com/2022/11/mini-cheetah-clone-teardown.html)
- In-repo: [belt_coupling_sim2real report](../reports/belt_coupling_sim2real/report.md) ·
  [replant-report-20260817.html](replant-report-20260817.html) ·
  [p11-hardware-runbook.md](p11-hardware-runbook.md)
