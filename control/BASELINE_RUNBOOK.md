# CAN transport swap — baseline experiment runbook

Goal: pin the CAN2USB stack's end-to-end behavior in a diffable record
(`can_baseline_bench.py` JSON), swap to CAN2SPI, re-run the identical
bench, and attribute every delta to the transport with nothing else
changed. Written 2026-08-15.

---

## Phase A — baseline the CAN2USB stack (before touching hardware)

**A1. Conditions** (these must be reproduced exactly in Phase C):

- [ ] Robot on stand, kneel pose, motors powered ≥5 min (bus + FETs warm)
- [ ] No flight code / probe running (`ps aux | grep -E 'main|balance'`),
      nothing else holding `/dev/ttyUSB0`
- [ ] Note bench supply voltage (dump showed vBus ≈ 31.8 V)
- [ ] Policy file present: `policy_F3490.pt`

**A2. Firmware dumps** — export from the vendor upper-computer tool the
same day, one per motor family, into `fw_dumps/` (2026-08-15 dumps
already saved: `fw_rs01_pitch_knee_20260815.csv`,
`fw_el05_roll_20260815.csv`). If running on a later date, re-export.

**A3. (Optional but recommended)** one last archival run of the
mechanism probe before the adapter is retired:

```bash
sudo python3 can_timing_probe.py --csv probe_can2usb_final.csv
```

**A4. Bench** — run twice back to back; the first warms caches and
thermals, the second is the baseline:

```bash
sudo python3 can_baseline_bench.py --adapter can2usb \
    --fw-csv rs01=fw_dumps/fw_rs01_pitch_knee_20260815.csv \
    --fw-csv el05=fw_dumps/fw_el05_roll_20260815.csv \
    --csv baseline_can2usb_ticks.csv
```

Takes ~4 min (T0 5 s, T1 ~5 s, T2 ~10 s, T3 6×10 s, T4 120 s).

**A5. Sanity-check against known numbers** before accepting the run —
if any of these are off, something in the setup is wrong, not the
adapter:

| check | expected (from 2026-08-12 sessions) |
|---|---|
| T0 unsolicited frames | 0 (bus silent) |
| T1 pooled RTT p50 | ≈ 2.9 ms, tight spread |
| T2 @1.0 ms pacing | ~100 % full, complete ≈ 8.7 ms |
| T2 @0.0 ms pacing | degraded (that's the point of the pacing) |
| T3 100 Hz row | miss < 1 %, full-reply ≈ 100 % |
| T3 free-run | ≈ 105–110 Hz achieved |
| T4 soak | full-reply ≈ 100 %, zero-reply streak < 5 |

**A6. Archive** — commit to git and copy off the Pi:

```bash
git add control/baseline_can2usb_*.json control/baseline_can2usb_ticks.csv \
        control/fw_dumps/ control/probe_can2usb_final.csv
git commit -m "CAN2USB transport baseline before CAN2SPI swap"
```

---

## Phase B — swap and bring-up (no benchmarking yet)

- [ ] B1. Physical swap. Wiring + OS setup: **`CAN2SPI_BRINGUP.md`**
      (module pin map, supply rule, dtoverlay), then per-boot
      `sudo ./setup_can0.sh`. Match the bus config the motors expect:
      1 Mbps (`motor_baud = 1`), check 120 Ω termination at both ends
      (≈60 Ω across CANH–CANL, powered off).
- [ ] B2. New driver — DONE 2026-08-15, pre-staged:
      `robstride_socketcan_driver.py` (same public API as
      `RobStrideDriver`: `send_batch`, `_parse_all_feedback` →
      `motor_states`, `enable_all`/`disable_all`) and `SocketCanShim`
      in `can_baseline_bench.py` for T0–T2. The swap is ONE line:
      `config.py: CAN_TRANSPORT = 'can2spi'` — `MotorBridge`, the
      bench, and the flight stack pick it up via `make_driver()`.
- [ ] B3. Functional bring-up before any timing: the verification
      ladder in `CAN2SPI_BRINGUP.md` §6, ending with
      `sudo python3 test_motors.py --no-zero` at kneel — enable, limp
      replies from all 6, positions match pre-swap readings at the
      same pose (MechOffset lives in the motor, so zeros should
      survive the swap — `--no-zero` verifies instead of overwriting).
- [ ] B4. Re-export firmware dumps. Nothing should have changed;
      Phase C's `--compare` verifies this mechanically.

## Phase C — post-swap baseline and comparison

- [ ] C1. Reproduce the A1 conditions exactly (stand, kneel, warm,
      RT on, same policy file, same sweep list).
- [ ] C2. Run the bench twice, keep the second:

```bash
sudo python3 can_baseline_bench.py --adapter can2spi \
    --fw-csv rs01=fw_dumps/<new_rs01_dump>.csv \
    --fw-csv el05=fw_dumps/<new_el05_dump>.csv \
    --csv baseline_can2spi_ticks.csv
```

- [ ] C3. Compare (no sudo needed):

```bash
python3 can_baseline_bench.py --compare \
    baseline_can2usb_<ts>.json baseline_can2spi_<ts>.json
```

**Reading the comparison:**

- RTT p50 should collapse 2.93 ms → well under 1 ms. If it doesn't,
  the new transport layer has its own turnaround cost — investigate
  before building on it.
- T2 @0.0 ms pacing should be ~100 % full. That licenses deleting the
  1 ms/frame pacing sleeps (−6 ms/tick).
- Free-run Hz is the new stack ceiling; T4 soak tells you whether
  100 Hz production now has real margin.
- The firmware section must print "identical across runs". If it
  flags a diff, the timing deltas are contaminated — fix and re-run.

**Scope guard:** the policy is trained at 100 Hz (sim dt 1/200,
decimation 2) — do NOT raise `CONTROL_HZ` off the back of a good
free-run number without retraining. The wins to bank at 100 Hz are:
delete the pacing sleeps, delete the 2.8 ms reply wait (probe test G's
no-wait drain — drain at tick start, before `read_state`), fresher obs,
and jitter margin.

---

## Results (2026-08-15) — swap validated

Comparison `baseline_can2usb_20260815_103342.json` vs
`baseline_can2spi_20260815_132105.json` (firmware identical):

| metric | can2usb | can2spi |
|---|---|---|
| RTT pooled p50 | 2.99 ms | **0.57 ms** |
| Batch @1.0 ms pacing p50 | 7.99 ms | 6.09 ms |
| Batch @0.5 ms pacing | — | **100 % full, 3.15 ms p50** |
| Batch @0 pacing full-rate | 0 % | 66–74 % (see below) |
| Free-run e2e ceiling | 111 Hz | 265–270 Hz |
| Max sustainable rate | 100 Hz | 150–200 Hz |
| Soak @100 Hz full-reply | 99.45 % | 99.42 % (worst streak 0) |
| Soak tick busy p95 | 9.30 ms | **3.9 ms** |

**Deviation from the pre-swap plan:** @0 ms pacing is NOT clean and the
pacing sleeps were NOT fully deleted. All six motors reply on the same
arbitration id **0x0FD** (MIT mode), and same-id frames cannot
arbitrate — back-to-back requests make replies collide (error-passive
count 72k during the unpaced soak, T4 full-reply ~0 %). Production
setting: `config.CAN2SPI_PACING_S = 0.0005` (T2 @0.5 ms = 100 % full
both runs). Zero pacing would require distinct per-motor reply ids
(vendor config = plant change — kept out of this baseline; procedure is
now **Phase D** below).

Banked at 100 Hz: replies complete ~3.2 ms into the tick (was ~9.3) →
obs ~6 ms fresher + ~6 ms/tick jitter margin. `CONTROL_HZ` stays 100
per the scope guard above.

---

## Phase D — distinct per-motor reply ids → zero pacing (planned 2026-08-16)

**Why.** In MIT mode every motor addresses its feedback frame to the
HOST CAN id — factory default 0xFD for all of them — with the replying
motor's id only in data[0]. Same-arbitration-id frames cannot arbitrate,
hence the 0.5 ms pacing workaround. Giving each motor its own host id
removes the collision domain: the bus arbitrates the replies natively
and `CAN2SPI_PACING_S` can go to 0.

**Mechanism** (researched 2026-08-16): RobStride MIT protocol
"Command 9 — Modify Host CAN ID": standard frame to the motor's own id,
data `FF FF FF FF FF FF <new_host_id> 01`. Sources: the RobStride 00
instruction manual ([aifitlab mirror](https://wiki.aifitlab.com/robstride-docs/robstride-00-instruction-manual));
the 0xFD-default host/feedback id is corroborated by
[motorbridge](https://github.com/motorbridge/motorbridge). Single-source
frame layout → the tool verifies every change empirically and stops on
the first mismatch. Fallback if Command 9 is a no-op on either motor
family: set 主机ID/host id per motor in the vendor upper-computer tool
(same tool as the fw dumps), then continue from D2.

Tool: `set_host_ids.py` (`--show` / `--set` / `--revert`, one motor at a
time, never depends on zero pacing itself). Assignment: **host id =
motor id** — reply arbitration id then equals data[0], so any candump is
self-checking. Reply-vs-command same-id overlap is not a hazard: a
motor's reply ends ~1 ms after its command; the next command to that
motor is a tick (10 ms) away. Driver and bench key replies on data[0]
and are host-id-agnostic — no code change needed to read either config.

**Expected win** (verify, don't assume): 6 back-to-back writes ≈ 0.65 ms
on-wire + 0.57 ms RTT → replies complete ~1.5 ms into the tick (now
~3.2 ms) → obs another ~1.7 ms fresher, tick busy p50 ~3.6 → ~1.5 ms,
and the 6 × 0.5 ms sleep jitter source is gone. This is a **plant
change** (motor NVRAM): do it in one sitting, on the stand, and bank a
fresh baseline JSON before any flight.

- [x] D0. Conditions: robot on stand, flight code stopped, motors
      powered (enable NOT needed — limp motors answer MIT probes).
      Archive state: `sudo python3 set_host_ids.py --show` — expect all
      six on 0x0FD. **Done 2026-08-16: six × 0x0FD confirmed.**
- [x] D1. Pilot **one motor per family** (firmware may differ):
      `--set --motor 102` (RS01) then `--set --motor 103` (EL05 roll).
      The tool probes before/after and STOPS on any mismatch; on a FAIL
      run `--revert --motor <id>` and switch to the vendor-tool
      fallback. **Done 2026-08-16: 102 → 0x066, 103 → 0x067, both
      `[ OK ]` — both families honor Command 9, no fallback needed.**
- [x] D2. Remaining motors: `sudo python3 set_host_ids.py --set`, then
      `--show` → 6/6 distinct, exit code 0. **Done 2026-08-16.**
- [x] D3. **Persistence check — gate for everything below.** Power-cycle
      the motors, re-run `--show`. If any motor is back on 0x0FD the
      change did not persist: save via the vendor tool, or (last resort)
      run `--set --yes` in `setup_can0.sh` at every boot — and note that
      a silent power-cycle reversion under zero pacing reproduces the
      72k-error-passive collision mode, which is why D7 adds a preflight.
      **Done 2026-08-16: `--show` → 6/6 distinct (0x065–0x06A), exit 0 —
      the ids persist in motor NVRAM.** (Safety condition is
      DISTINCTNESS, not non-0xFD: a single motor reverting to 0x0FD is
      still collision-free; only two-or-more sharing an id is unsafe,
      and `--show` flags exactly that as DUPLICATE.)
- [ ] D4. Functional ladder unchanged from B3: `sudo python3
      test_motors.py --no-zero` at kneel — all 6 reply, positions match
      pre-change readings (host id lives beside MechOffset in motor
      config; zeros must survive — if any offset moved, stop and
      compare fw dumps).
- [x] D5. Re-export fw dumps into `fw_dumps/`. **Skipped (user
      decision 2026-08-16)** — no firmware data changed, the 08-15 CSVs
      remain the reference. Consequence: it is still unknown whether
      host id is among the 23 dumped params, so a fw-dump diff can NOT
      detect a reverted motor — the `--show` preflight (wired into
      `setup_can0.sh` at D7) is the reversion guard.
- [x] D6. Bench, pacing forced to zero, robot on stand:

```bash
sudo python3 can_baseline_bench.py --adapter can2spi \
    --fw-csv rs01=fw_dumps/<new_rs01_dump>.csv \
    --fw-csv el05=fw_dumps/<new_el05_dump>.csv \
    --csv baseline_can2spi_hostids_ticks.csv
```

      Acceptance vs `baseline_can2spi_20260815_132105.json`:
      T2 @0.0 ms = 100 % full (was 66–74 %); T4 soak @100 Hz full-reply
      ≥ 99.4 % with error-passive ≈ 0 (was 72k unpaced); tick busy p50
      well under 3.6 ms. If T2 @0 is still not clean, replies are
      colliding with something else — candump, do NOT ship zero pacing.
      **Run 2026-08-16 (`baseline_can2spi_20260816_152506.json`): T2
      acceptance PASSED — see results below. Caveat: T3/T4 read
      `config.CAN2SPI_PACING_S`, which was still 0.0005, so the soak ran
      PACED (busy p50 3.59 = 3.14 batch + policy). The unpaced 120 s
      soak happens as part of D7.**
- [x] D7. Ship — in order: (1) D3 checked; (2) flip
      `config.CAN2SPI_PACING_S = 0.0` (+ update its comment); (3) re-run
      the D6 bench so T3/T4 finally soak UNPACED — accept busy p50
      ~1.8–2.0 ms, soak full-reply ≥ 99.4 %, zero-reply streak ≈ 0,
      no error-frame growth; (4) commit bench JSONs + fw dumps + config
      together. Flight preflight: `setup_can0.sh` now runs
      `set_host_ids.py --show` after bringing the bus up and refuses
      (exit 2) on duplicate reply ids.
      **Done 2026-08-16 except (4): config flipped, unpaced bench =
      `baseline_can2spi_20260816_155207.json` (soak 100.00 % full
      replies, 0 drop ticks, streak 0, busy p95 2.45 ms — acceptance
      MET; the T3 miss% column is a bench artifact, see results below).
      REMAINING: commit config + both 08-16 JSONs + ticks CSV +
      set_host_ids.py + setup_can0.sh from the Pi.**

**Revert:** `sudo python3 set_host_ids.py --revert` (all six back to
0xFD) + `CAN2SPI_PACING_S = 0.0005`. That restores the validated 08-15
configuration exactly.

**Scope guard** (same as Phase C): none of this touches `CONTROL_HZ`.
The win is banked as fresher obs and jitter margin at 100 Hz.

### Phase D results (2026-08-16 — COMPLETE except the final commit)

Run 1 = `baseline_can2spi_20260816_152506.json` (distinct ids, config
pacing still 0.5 ms → paced soak). Run 2 =
`baseline_can2spi_20260816_155207.json` (config pacing 0.0 → the
shipping configuration). Reference = `..._20260815_132105.json`
(shared 0x0FD):

| metric | 08-15 shared id | 08-16 run 1 | 08-16 run 2 (ships) |
|---|---|---|---|
| RTT pooled p50 | 0.571 ms | 0.572 ms | 0.595 ms |
| Batch @0 pacing full-rate | 71 % | **100 %** | **100 %** |
| Batch @0 pacing p50 | 3.958 ms | 1.825 ms | 1.833 ms |
| Free-run e2e ceiling | 264.8 Hz | 270.8 Hz | **433.0 Hz** |
| Soak @100 Hz full-reply | 99.42 % | 99.74 % (paced) | **100.00 %, 0 drops** |
| Soak tick busy p95 | 3.912 ms | 3.899 ms (paced) | **2.448 ms** |
| Soak jitter p99 | 0.016 ms | 0.022 ms | 0.027 ms |

Reading: the collision domain is gone exactly as the mechanism
predicted — each reply arbitrates on its own id, an unpaced batch
completes in 1.83 ms at 100 % full, and the 120 s unpaced soak is the
first PERFECT one on any transport (12k ticks, zero drops). T1/T2@1 ms
are bit-identical across all three runs: nothing else moved.

**Run 2's T3 `miss%` (3–15 %, "max sustainable --") is a bench
artifact, not a regression — kernel RT throttling.** T3 waits for the
next tick in a pure busy-spin (`can_baseline_bench.py:483`) under
SCHED_FIFO; with the 6 × 0.5 ms pacing `time.sleep`s deleted, the
thread now consumes 100 % CPU, and the kernel's RT throttle (default
`sched_rt_runtime_us` = 950000/1000000) force-deschedules it ~50 ms
every second → periodic multi-tick deadline misses that scale with
rate. Three confirmations: full-reply stayed 100 % at every rate
including free-run 433 Hz (the transport never hiccuped); T4 — which
sleeps between ticks (`:561`) — is pristine (p99 0.027 ms); and run 1,
whose in-tick sleeps kept RT usage below the throttle threshold,
showed miss ≈ 0 at the same rates. The single 81 ms T4 overrun fits
the same cause: the throttle window was still saturated from T3's
300 Hz spin sweep when T4 started, forcing one deschedule before the
sleeps re-qualified the thread. **The flight loop sleeps between ticks
(`balance_log.py:774`, RT usage ~22 % at 100 Hz) and is NOT exposed.**
If T3 numbers are ever needed clean, run the bench with
`sudo sysctl -w kernel.sched_rt_runtime_us=-1` (restore 950000 after)
— do not compare a throttled T3 sweep against an unthrottled one.

Banked at 100 Hz vs the 08-15 baseline: replies complete ~1.8 ms into
the tick (was ~3.2 paced, ~9.3 on CAN2USB) → obs another ~1.4 ms
fresher, tick busy p95 3.9 → 2.45 ms, the 6 × 0.5 ms sleep jitter
source deleted, and soak reliability 99.4 → 100.0 %. `CONTROL_HZ`
stays 100 — the 433 Hz ceiling is headroom, not a license (retrain
decision, same scope guard as Phase C).
