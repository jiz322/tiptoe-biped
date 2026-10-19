# Flight prep — T0_straight + S8b_air03 on hardware, 2026-09-29

Desk work done with the Pi off. Everything the Pi needs is staged in `control/`;
the user runs all motion (CLAUDE.md rule).

## 1. What flies

| arm | staged file | source checkpoint | why this iteration |
|---|---|---|---|
| T0 | `control/t0_1020.pt` | `p9_replant/2026-09-26_22-11-33_T0_straight/model_1020.pt` | windowed (±5 it) mean_episode_length peak 1524 (final 1040 = 1445); run killed at 1040 |
| S8b | `control/s8b_8880.pt` | `p9_replant/2026-09-19_08-14-51_S8b_air03/model_8880.pt` | windowed peak 2156 (final 8930 = 2080); plateau, difference is within noise |

Raw rsl_rl checkpoints, byte-identical copies (`cmp` clean). `control/policy.py`
loads them in checkpoint mode (actor MLP + EmpiricalNormalization rebuilt, iter
stamped into the npz as `policy_iter`) — same path as the S4 flight. No JIT export
needed; the `exported/policy.pt` files inside the run folders are play_alpha
by-products, not deploy artifacts.

TB table: `data/_flight_pick_20260929.txt` (tool `data/_pick_flight_ckpt_20260929.py`,
runs with any local python that has numpy and tensorboard installed).

## 2. Contract diff (checklist items 1, 3, 4, 7) — both runs vs `control/config.py`

Read from `params/env.yaml` of each run, not from the launch files.

| item | T0_straight | S8b_air03 | deploy (`config.py` / `balance_log.py`) | verdict |
|---|---|---|---|---|
| obs layout | 180 = 36 × 5 | 180 = 36 × 5 | OBS_DIM 180 | match (actor.0 width checked by the loader) |
| stance preset | straight | straight | `STANCE_PRESET = "straight"` (uncommitted edit, bit-exact legacy pose) | match |
| init_h (ch 31) | spawn 0.434, band 0.385 ± 0.085 | 0.417 ± 0.069 | SPAWN_HEIGHT 0.4401 | z = +0.58 / +0.29, in band |
| servo kp/kd hip_pitch, knee | 20/1.4, 30/1.5 | same | REAL_KP/KD 20/1.4, 30/1.5 (hip_roll 6/0.3) | match |
| action scales | 0.3 / 0.75 / 1.025 | same | balance_log ACTION_SCALES | match (unchanged since K) |
| action LPF | 0.6 | 0.6 | shown at startup, verify 0.60 | match |
| knee decouple FF | true | true | `KNEE_DECOUPLE_FF = True` | match (§0.3 gate) |
| handover_prime_prob | 0.0 | 0.0 | → **'v' blind handover only**, never 'b' | match |
| delays | action 1–2, imu 1, joint 1 | same | measured S4 values | match |
| dr_proj_grav_bias_max | **0.00** | **0.00** | IMU_PG_OFFSET x = +0.089 in config | **open knob — see §3** |
| dr_knee_angle_bias_max | 0.02 | 0.0 | n/a (plant DR) | no deploy consequence |
| physics rate / plant | 500 Hz, S7 armature pins, belt DR off | 2000 Hz, vendor rotor 4.2e-3, belt DR 100–1000 | n/a | no deploy consequence (nothing deploy-side consumes rotor inertia) |

`policy_contract_check.py` (desk): **PASS** on both files, every balance-test
constant |z| < 1.4. Trained pg_x band (newest frame, whole-training stats):
T0 +0.022 ± 0.186, S8b −0.196 ± 0.196 (S4a reference −0.180 ± 0.210).

## 3. The one open contract knob: IMU pitch trim

Both policies trained with **zero** pg-bias DR, so the trim is not covered by
any band — it is a pure plant-gap lever. Facts on file:

- physical IMU bias at tile-level torso = raw −0.013 → contract-faithful trim
  `--pg-offset -0.015` (docs/imu-offset-and-joint-zero-20260920.md);
- `+0.089` = the compensator that produced the first sustained balance (R1r,
  09-03) and which the T-line runbook §8.3 says to re-bracket;
- S4a (band 0.13) at trim 0.0 "looked more like sim" but still leaned back.

Pre-registered bracket, per policy, n ≥ 3 trials per point, raw pg logged
per tick so nothing has to be recovered later:

    -0.015  (true trim, contract-faithful)   →   +0.045   →   +0.089

Read after each block: raw torso pitch at the last quiet 0.5 s before
departure, fall direction, grounded survival to |tilt| > 25°
(`data/_s4_flight_summary.py` computes all three; add today's folder to its
SETS list). Verdict rule: if survival is monotone in trim for BOTH policies the
gap is a static pitch offset again (joint-zero branch stays open); if the two
policies prefer different trims, the lever is policy-side, not plant-side.

`config.py` y trim stays 0.0 (true +0.020, low priority, unchanged since 09-20).

## 4. Sync to the Pi (Pi off at prep time — run when it is up)

On this machine (the S4 precedent e10762c committed the raw `.pt` files; `*.pt`
is not gitignored):

    git add control/t0_1020.pt control/s8b_8880.pt control/config.py control/balance.py \
            data/t0_s8b_flight_20260929/README.txt docs/flight-prep-20260929.md \
            data/_pick_flight_ckpt_20260929.py data/_flight_pick_20260929.txt
    git commit -m "Stage T0_1020 + S8b_8880 for the 09-29 flight; stance-preset deploy mirror"
    git push

`control/config.py` / `control/balance.py` carry the 09-26 stance-preset
mirror (uncommitted until now). With `STANCE_PRESET = "straight"` they are
bit-exact with the committed constants, so pulling them on the Pi changes
nothing for today's policies — but the Pi and desk must run the same file for
the contract diff above to mean anything.

On the Pi:

    ssh <pi>
    cd ~/Documents/big_thigh && git pull
    ip -details link show can0                      # ERROR-ACTIVE, counters 0
    sudo python3 control/test_motors.py --no-zero   # 6/6 replies, stored zeros kept
    cd control && sudo ~/miniforge3/bin/python3 policy_contract_check.py t0_1020.pt s8b_8880.pt

(`policy_contract_check.py` imports torch → miniforge python, not /usr/bin.)

## 5. Trial protocol (per docs/p11-hardware-runbook.md §2 + 09-10 lessons)

1. Crawl-pose check (`sudo python3 test_position.py`): knee L ≈ −R within 3°.
   Repeat **between every trial** — 1 tooth = 12°, three skips on 09-10.
2. Camera on every trial; the GPIO-18 LED marks t = 0 of the log.
3. Launch, per trial (trim from the §3 bracket):

       sudo ~/miniforge3/bin/python3 balance_log.py --policy t0_1020.pt  --pg-offset -0.015
       sudo ~/miniforge3/bin/python3 balance_log.py --policy s8b_8880.pt --pg-offset -0.015

   Startup: read the `[Policy] ... iter 1020 / 8880` line, the contract-guard
   PASS, LPF alpha 0.60, and the trim echo. Ramp gains to REAL_KP, HOLD, then
   **'v'** (blind handover). 'h' back to HOLD saves the npz.
4. Ground start, lowered over ~1 s, release at contact, no push
   (the trained put-down, runbook 2.3 addendum).
5. Tripwires: motor temps ≤ 40 °C, no squeal, any chatter → 'h' immediately
   and stop the arm.
6. After the session: `scp pi:~/Documents/big_thigh/control/*_log_2026*.npz
   data/t0_s8b_flight_20260929/`, fill the README table (policy, trim, crawl
   check, notes per file), then run `data/_s4_flight_summary.py` with the
   folder added.

## 6. What each policy is expected to show (pre-registered, from sim)

- T0_1020 is a 1040-iteration fresh policy at 500 Hz with `base_height` 0.40
  and `standup_success_high` 0.86–0.90 in TB; its idle in sim is the known
  forward-lean crouch. Expect a LESS settled idle than S8b, not a cleaner one.
- S8b_8880: plateaued (episode length ~2100/3000), idle_toe_z_p10 0.0735,
  knee_sat_ema_max ~0.50 — the same class as S4a on hardware (shuffle, lean
  back). The feet_air 0.03 change should show as smaller lifts on video; if
  lifts look ≥ 5 cm the knob was inert (its own header prediction).
- Neither arm addresses the 09-31 fixed-point dead zone or the +0.104 pitch
  excess; a fall backward at trim −0.015 on both is the null result and is
  consistent with everything on file.
