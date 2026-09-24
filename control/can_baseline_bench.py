"""
can_baseline_bench.py — transport-swap baseline: end-to-end numbers for the
CURRENT CAN adapter, in a fixed JSON schema, so the exact same script can be
re-run after the CAN2USB → CAN2SPI swap and diffed with --compare.

Why this exists (2026-08-15)
----------------------------
can_timing_probe.py already answered *why* the CAN2USB adapter needs 1 ms
pacing (L = 2.93 ms serialized transactions, no pipelining; only the
no-wait-drain restructure validated). What it does NOT give is a
before/after-comparable record of what the robot actually gets end to end:
max closed-loop rate with the real policy in the loop, per-phase tick
budget, reply integrity at rate, and sustained-run jitter. This script
measures those, tagged with an --adapter label, and archives them.

Test battery
------------
  T0  Idle-bus silence: motors enabled, NO commands, listen for unsolicited
      frames. Pins the request-reply-only bus contract: echoFreHz=500 is
      configured in the motor firmware, and if the echo/push mechanism
      (echoPara1-4) were ever enabled — by the vendor tool, or by a config
      migration on the new adapter — unsolicited frames would collide with
      request-reply and silently change every timing number below.
  T1  Single-transaction RTT per motor (transport shim; generous timeout).
      Headline transport number. CAN2USB measured 2.93 ms deterministic;
      SPI/socketcan should land ~0.2-0.5 ms.
  T2  Batch-of-6 completion time vs inter-frame pacing {1.0, 0.5, 0.0 ms}
      with a 15 ms observation window: full-reply rate, completion
      p50/p95, per-motor reply rate. On the new transport, 0.0 pacing
      should be 100% — that alone removes 6 ms/tick.
  T3  End-to-end closed-loop rate sweep with the REAL policy running:
      imu.update → read_state → production obs build (controller's own
      _build_obs_frame/_stack_history) → policy.forward → send_target
      with kp = kd = 0 everywhere (motors enabled but limp — same safety
      pattern as can_timing_probe.py). Target 0 in the sweep list means
      FREE-RUN: no scheduler sleep, the loop goes as fast as it can —
      that achieved Hz IS the max control frequency of this stack.
      Per rate: achieved Hz, deadline-miss %, full-reply %, per-phase
      p50/p95 (imu/read/obs/nn/transform/write/wait/parse).
  T4  Sustained soak at production CONTROL_HZ (default 120 s): period
      jitter p50/p95/p99/max, overruns, reply losses + worst consecutive
      drop streak, IMU fps under CAN load, motor temps start/end.

What ports to the new transport unchanged
-----------------------------------------
T3/T4 talk only to the MotorBridge/driver public API — they re-run as-is
with a new driver underneath. T0/T1/T2 poke the serial layer directly
through the small TransportShim class below; that is the ONLY part to
reimplement for socketcan/SPI (send one encoded frame; count reply frames;
timestamps at the same layer: command handed to transport → reply parsed
in Python).

Firmware snapshot (--fw-csv)
----------------------------
Motor-side config shapes everything downstream of the transport (reply
pos/vel/torque are firmware-FILTERED: spd_filt_gain differs RS01=0.1 vs
EL05=0.5; torque_fdb carries a ~25-35 ms filter — see the 08-14 telemetry
forensics) but CANNOT be read in-session: MIT mode ignores private-
protocol reads (see robstride_param_read.py's protocol round-trip). So the
bench embeds parameter dumps exported from the vendor upper-computer tool:

    --fw-csv rs01_pitch=fw_rs01.csv --fw-csv el05_roll=fw_el05.csv

A curated whitelist (limits, filters, echo config, CAN config, versions)
is stored in the JSON so a later firmware change shows up in --compare's
firmware section. Export fresh dumps the same day as the baseline run.

Comparability checklist (keep these identical across runs)
----------------------------------------------------------
  - robot on stand, motors powered, same pose (kneel), bus warm
  - same policy file (default policy_F3490.pt), same --sweep-hz list
  - RT scheduling on (default; --no-rt to disable) — record if not
  - run twice back to back; trust the second (cache/thermal warm-up)
  - archive the JSON under docs/ or logs/, never overwrite the old one

Usage
-----
    sudo python3 can_baseline_bench.py --adapter can2usb
    sudo python3 can_baseline_bench.py --adapter can2usb --no-imu
    sudo python3 can_baseline_bench.py --adapter can2spi --out b_spi.json
    python3 can_baseline_bench.py --compare baseline_can2usb.json b_spi.json

Safety
------
Same pattern as can_timing_probe.py / diag_replies.py: motors are enabled
(required for MIT-mode replies) but every command carries Kp=Kd=0 and
t_ff=0 (HIP_REACTION_FF is off; the knee v_des feedforward is torqueless
at Kd=0), so the robot stays limp throughout. disable_all() on every exit
path. Robot on stand.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import subprocess
import time
from collections import Counter

import numpy as np

import config as C

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
GREEN, YELLOW, RED = "\033[32m", "\033[33m", "\033[31m"

SCHEMA_VERSION = 1
DEFAULT_POLICY = 'policy_F3490.pt'
SIM_NAMES = ['L_roll', 'L_hip', 'L_knee', 'R_roll', 'R_hip', 'R_knee']
ZERO6 = np.zeros(6, dtype=np.float32)


def rate_color(rate):
    return GREEN if rate >= 99 else YELLOW if rate >= 90 else RED


def pct(a, q):
    return float(np.percentile(np.asarray(a, dtype=np.float64), q))


def dist_ms(samples):
    """Fixed-schema latency distribution dict (ms)."""
    if not len(samples):
        return None
    a = np.asarray(samples, dtype=np.float64)
    return {'p50': round(pct(a, 50), 3), 'p95': round(pct(a, 95), 3),
            'p99': round(pct(a, 99), 3), 'max': round(float(a.max()), 3),
            'n': int(a.size)}


# ════════════════════════════════════════════════════════════
# Transport shim — the ONLY class to reimplement for CAN2SPI
# ════════════════════════════════════════════════════════════
class TransportShim:
    """Raw frame send / reply count at the transport layer.

    CAN2USB implementation: AT-framed serial through the driver's port.
    For socketcan/SPI, SocketCanShim below reimplements the same
    interface on top of the new driver; timestamp semantics (send =
    frame handed to the transport API, reply = frame available to
    Python) are identical. make_shim() picks the right one.
    """
    HEADER = b'\x41\x54'
    FRAME_LEN = 17

    def __init__(self, driver):
        self.driver = driver
        self.frames = {mid: driver._encode_mit_frame(mid, 0, 0, 0, 0, 0)
                       for mid in C.MOTOR_IDS}

    def flush(self):
        self.driver.ser.reset_input_buffer()

    def send_frame(self, mid):
        self.driver.ser.write(self.frames[mid])

    def pending_bytes(self):
        return self.driver.ser.in_waiting

    def read_raw(self):
        return self.driver.ser.read_all()

    def wait_replies(self, n_frames, timeout_s):
        """Spin until n_frames replies are in (or timeout). Returns
        (elapsed_s, reply_motor_ids)."""
        want = n_frames * self.FRAME_LEN
        t0 = time.monotonic()
        deadline = t0 + timeout_s
        while self.driver.ser.in_waiting < want:
            if time.monotonic() > deadline:
                break
        elapsed = time.monotonic() - t0
        raw = self.driver.ser.read_all() if self.driver.ser.in_waiting else b''
        return elapsed, self._parse_ids(raw)

    def _parse_ids(self, raw):
        ids, offset = [], 0
        while True:
            idx = raw.find(self.HEADER, offset)
            if idx == -1 or idx + self.FRAME_LEN > len(raw):
                break
            ids.append(raw[idx + 7])
            offset = idx + self.FRAME_LEN
        return ids


class SocketCanShim:
    """TransportShim counterpart for the SocketCAN (CAN2SPI) driver.

    Same interface and timestamp semantics; frames are 16-byte
    struct can_frame (id 4B, dlc 1B, pad 3B, data 8B), so the replying
    motor id — data[0] — sits at offset 8 of each frame.
    """
    FRAME_LEN = 16

    def __init__(self, driver):
        self.driver = driver
        self.frames = {mid: driver._encode_mit_frame(mid, 0, 0, 0, 0, 0)
                       for mid in C.MOTOR_IDS}

    def flush(self):
        self.driver.reset_input_buffer()

    def send_frame(self, mid):
        self.driver._send_raw(self.frames[mid])

    def pending_bytes(self):
        return self.driver._rx_bytes_pending()

    def read_raw(self):
        return self.driver._read_all()

    def wait_replies(self, n_frames, timeout_s):
        want = n_frames * self.FRAME_LEN
        t0 = time.monotonic()
        deadline = t0 + timeout_s
        while self.pending_bytes() < want:
            if time.monotonic() > deadline:
                break
        elapsed = time.monotonic() - t0
        return elapsed, self._parse_ids(self.read_raw())

    def _parse_ids(self, raw):
        return [raw[off + 8] for off
                in range(0, len(raw) - self.FRAME_LEN + 1, self.FRAME_LEN)]


def make_shim(driver):
    """Serial driver exposes .ser; the SocketCAN driver does not."""
    return TransportShim(driver) if hasattr(driver, 'ser') \
        else SocketCanShim(driver)


# ════════════════════════════════════════════════════════════
# Null IMU — lets the bench run CAN-only (--no-imu)
# ════════════════════════════════════════════════════════════
class NullIMU:
    fps = 0.0
    def update(self): pass
    def projected_gravity(self): return np.array([0., 0., -1.], np.float32)
    def ang_vel_body(self): return np.zeros(3, np.float32)
    def lin_vel_body(self): return np.zeros(3, np.float32)
    def yaw(self): return 0.0
    def upright_cos(self): return 1.0
    def close(self): pass


# ════════════════════════════════════════════════════════════
# Firmware snapshot — curated whitelist from vendor-tool CSV dumps
# ════════════════════════════════════════════════════════════
# Everything here either (a) must be matched by the new transport's config
# (motor_baud, CAN_TIMEOUT), (b) gates the bus contract (echo*), or
# (c) filters the data the policy sees (*/filt_gain, loop gains, limits) —
# a silent firmware change in any of these would masquerade as a
# transport-swap regression.
FW_WHITELIST = (
    'AppCodeVersion', 'AppBuildDate',
    'echoFreHz', 'echoPara1', 'echoPara2', 'echoPara3', 'echoPara4',
    'motor_baud', 'CAN_MASTER', 'CAN_TIMEOUT', 'protocol_1',
    'limit_torque', 'limit_spd', 'limit_cur',
    'cur_filt_gain', 'spd_filt_gain', 'loc_ref_filt_gai',
    'cur_kp', 'cur_ki', 'spd_kp', 'spd_ki', 'loc_kp',
    'Kt_Nm/Amp',
)


def parse_fw_csv(path):
    """Vendor upper-computer dump (Function Code,Name,...,Current Value)
    → {name: current_value} for whitelisted params. First occurrence wins
    (some dumps repeat read-only names)."""
    out = {}
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        for row in csv.reader(f):
            if len(row) < 7:
                continue
            name, value = row[1].strip(), row[6].strip()
            if name in FW_WHITELIST and name not in out:
                out[name] = value
    missing = [n for n in FW_WHITELIST if n not in out]
    if missing:
        out['_missing'] = missing      # e.g. loc_ref_filt_gai on RS01
    return out


# ════════════════════════════════════════════════════════════
# T0 — idle-bus silence (request-reply contract check)
# ════════════════════════════════════════════════════════════
def test_idle_bus(shim, driver, seconds=5.0):
    print(f"\n{BOLD}=== T0: idle-bus listen, motors enabled, no commands "
          f"({seconds:.0f} s) ==={RESET}\n")
    events = []
    with driver.lock:
        shim.flush()
        t0 = time.monotonic()
        last = 0
        while time.monotonic() - t0 < seconds:
            n = shim.pending_bytes()
            if n > last:
                events.append(((time.monotonic() - t0), n - last))
                last = n
            time.sleep(0.001)
        raw = shim.read_raw() if last else b''
    ids = shim._parse_ids(raw)
    frames = len(ids)
    by_motor = dict(Counter(ids))
    if frames == 0 and last == 0:
        print(f"  {GREEN}Bus silent — strict request-reply confirmed "
              f"(echo push inactive).{RESET}")
    else:
        print(f"  {RED}{last} unsolicited bytes / {frames} frames!{RESET} "
              f"per-motor: {by_motor}")
        print(f"  {DIM}Echo push (echoFreHz/echoPara) or periodic status is "
              f"active — every timing\n  number below includes collision "
              f"effects. Fix the config before baselining.{RESET}")
    return {'seconds': seconds, 'unsolicited_bytes': last,
            'unsolicited_frames': frames,
            'by_motor': {str(k): v for k, v in by_motor.items()}}


# ════════════════════════════════════════════════════════════
# T1 — single-transaction RTT
# ════════════════════════════════════════════════════════════
def test_rtt(shim, driver, n_reps=100, timeout_ms=8.0):
    print(f"\n{BOLD}=== T1: single-transaction RTT "
          f"({n_reps} reps/motor) ==={RESET}\n")
    per_motor = {}
    pooled = []
    print(f"  {'motor':>6s}  {'name':<8s}  {'ok':>8s}  {'p50':>7s}  {'p95':>7s}")
    for i, mid in enumerate(C.MOTOR_IDS):
        rtts = []
        for _ in range(n_reps):
            with driver.lock:
                shim.flush()
                t0 = time.monotonic()
                shim.send_frame(mid)
                _, ids = shim.wait_replies(1, timeout_ms / 1000.0)
                if mid in ids:
                    rtts.append((time.monotonic() - t0) * 1000.0)
            time.sleep(0.005)
        d = dist_ms(rtts)
        per_motor[str(mid)] = {'name': SIM_NAMES[i], 'ok': len(rtts),
                               'reps': n_reps, 'rtt_ms': d}
        pooled.extend(rtts)
        if d:
            print(f"  {mid:>6d}  {SIM_NAMES[i]:<8s}  {len(rtts):>4d}/{n_reps}"
                  f"  {d['p50']:7.2f}  {d['p95']:7.2f}")
        else:
            print(f"  {mid:>6d}  {SIM_NAMES[i]:<8s}  {RED}0/{n_reps}{RESET}")
    pooled_d = dist_ms(pooled)
    if pooled_d:
        print(f"\n  Pooled RTT p50: {pooled_d['p50']:.2f} ms   "
              f"{DIM}(CAN2USB 08-12: 2.93 ms; physics floor ~0.7 ms){RESET}")
    return {'per_motor': per_motor, 'pooled_ms': pooled_d}


# ════════════════════════════════════════════════════════════
# T2 — batch completion vs pacing
# ════════════════════════════════════════════════════════════
def test_batch(shim, driver, pacings_ms=(1.0, 0.5, 0.0), n_ticks=100,
               window_ms=15.0):
    print(f"\n{BOLD}=== T2: batch-of-6 completion vs pacing "
          f"({n_ticks} ticks each) ==={RESET}\n")
    n_motors = len(C.MOTOR_IDS)
    results = []
    print(f"  {'pacing':>7s}  {'full':>8s}  {'complete p50/p95':>17s}  "
          f"{'worst-motor %':>13s}")
    for pacing in pacings_ms:
        counts = Counter()
        full = 0
        complete_ms = []
        for _ in range(n_ticks):
            with driver.lock:
                shim.flush()
                t0 = time.monotonic()
                for mid in C.MOTOR_IDS:
                    shim.send_frame(mid)
                    if pacing > 0:
                        time.sleep(pacing / 1000.0)
                _, ids = shim.wait_replies(
                    n_motors, (pacing * n_motors + window_ms) / 1000.0)
            t_done = (time.monotonic() - t0) * 1000.0
            for mid in ids:
                counts[mid] += 1
            if len(set(ids) & set(C.MOTOR_IDS)) >= n_motors:
                full += 1
                complete_ms.append(t_done)
            time.sleep(0.02)
        rates = {str(mid): round(100.0 * counts.get(mid, 0) / n_ticks, 1)
                 for mid in C.MOTOR_IDS}
        worst = min(rates.values())
        d = dist_ms(complete_ms)
        comp = (f"{d['p50']:5.2f}/{d['p95']:5.2f}ms" if d else "    never    ")
        print(f"  {pacing:5.1f}ms  {rate_color(100.0 * full / n_ticks)}"
              f"{full:>4d}/{n_ticks}{RESET}  {comp:>17s}  "
              f"{rate_color(worst)}{worst:12.0f}%{RESET}")
        results.append({'pacing_ms': pacing,
                        'full_rate_pct': round(100.0 * full / n_ticks, 1),
                        'complete_ms': d, 'per_motor_pct': rates})
    return results


# ════════════════════════════════════════════════════════════
# T3 / T4 shared — one production-shaped e2e tick
# ════════════════════════════════════════════════════════════
def e2e_tick(imu, motors, ctrl, phase_acc):
    """Runs one full production-shaped control tick with zero gains.
    Appends per-phase ms to phase_acc; returns replies_parsed."""
    t0 = time.monotonic()
    imu.update()
    t1 = time.monotonic()
    motors.read_state()
    t2 = time.monotonic()
    ctrl._gait_phase = (ctrl._gait_phase
                        + 2.0 * np.pi * C.GAIT_FREQ_HZ * C.DT) % (2.0 * np.pi)
    frame = ctrl._build_obs_frame()
    obs = ctrl._stack_history(frame)
    t3 = time.monotonic()
    action = ctrl.policy.forward(obs)
    t4 = time.monotonic()
    ctrl._last_action = action
    # Production _apply_action math, but gains forced to zero (limp).
    targets = ctrl._default_jpos + action * ctrl._action_scale
    actual = motors.sim_joint_pos
    for i in range(6):
        lo = actual[i] - ctrl._MAX_CMD_ERROR_RAD
        hi = actual[i] + ctrl._MAX_CMD_ERROR_RAD
        targets[i] = max(lo, min(hi, targets[i]))
    timing = motors.send_target(targets, ZERO6, ZERO6, return_timing=True)
    t5 = time.monotonic()

    phase_acc['imu'].append((t1 - t0) * 1000.0)
    phase_acc['read'].append((t2 - t1) * 1000.0)
    phase_acc['obs'].append((t3 - t2) * 1000.0)
    phase_acc['nn'].append((t4 - t3) * 1000.0)
    phase_acc['transform'].append(timing.get('transform_ms', 0.0))
    phase_acc['write'].append(timing.get('write_ms', 0.0))
    phase_acc['wait'].append(timing.get('wait_ms', 0.0))
    phase_acc['parse'].append(timing.get('parse_ms', 0.0))
    phase_acc['tick_busy'].append((t5 - t0) * 1000.0)
    return timing.get('replies_parsed', 0)


def new_phase_acc():
    return {k: [] for k in ('imu', 'read', 'obs', 'nn', 'transform',
                            'write', 'wait', 'parse', 'tick_busy')}


# ════════════════════════════════════════════════════════════
# T3 — closed-loop rate sweep (0 = free-run)
# ════════════════════════════════════════════════════════════
def test_e2e_sweep(imu, motors, ctrl, hz_list, seconds, csv_rows):
    print(f"\n{BOLD}=== T3: end-to-end closed-loop sweep, real policy, "
          f"zero gains ({seconds:.0f} s per rate) ==={RESET}\n")
    print(f"  {'target':>8s}  {'achieved':>9s}  {'miss%':>6s}  "
          f"{'full-reply%':>11s}  {'busy p50/p95':>14s}")
    results = []
    for hz in hz_list:
        ctrl._needs_history_init = True
        acc = new_phase_acc()
        full = 0
        misses = 0
        n = 0
        free_run = (hz == 0)
        period = 0.0 if free_run else 1.0 / hz
        t_start = time.monotonic()
        t_end = t_start + seconds
        next_t = t_start
        while time.monotonic() < t_end:
            replies = e2e_tick(imu, motors, ctrl, acc)
            n += 1
            if replies >= len(C.MOTOR_IDS):
                full += 1
            csv_rows.append(('T3', hz, n, acc['tick_busy'][-1], replies))
            if not free_run:
                next_t += period
                now = time.monotonic()
                if now > next_t:
                    misses += 1
                else:
                    while time.monotonic() < next_t:
                        pass
        wall = time.monotonic() - t_start
        achieved = n / wall
        full_pct = 100.0 * full / max(n, 1)
        miss_pct = 100.0 * misses / max(n, 1)
        busy = acc['tick_busy']
        label = 'free-run' if free_run else f'{hz:.0f} Hz'
        print(f"  {label:>8s}  {achieved:8.1f}H  "
              f"{(GREEN if miss_pct < 1 else RED)}{miss_pct:5.1f}{RESET}  "
              f"{rate_color(full_pct)}{full_pct:10.1f}{RESET}  "
              f"{pct(busy, 50):6.2f}/{pct(busy, 95):5.2f}ms")
        results.append({
            'target_hz': hz, 'free_run': free_run,
            'achieved_hz': round(achieved, 1), 'ticks': n,
            'miss_pct': round(miss_pct, 2),
            'full_reply_pct': round(full_pct, 2),
            'phases_ms': {k: dist_ms(v) for k, v in acc.items()},
        })
    sustainable = [r for r in results if not r['free_run']
                   and r['miss_pct'] < 1.0 and r['full_reply_pct'] >= 99.0
                   and r['achieved_hz'] >= 0.99 * r['target_hz']]
    max_hz = max((r['target_hz'] for r in sustainable), default=0)
    free = next((r for r in results if r['free_run']), None)
    print(f"\n  Max sustainable rate (achieved>=99% target, miss<1%, "
          f"full-reply>=99%): {BOLD}{max_hz or '--'} Hz{RESET}")
    if free:
        print(f"  Free-run ceiling: {BOLD}{free['achieved_hz']:.1f} Hz{RESET} "
              f"at {free['full_reply_pct']:.0f}% full replies "
              f"{DIM}(reply losses at free-run are expected on CAN2USB: the "
              f"2.8 ms wait gets starved){RESET}")
    return {'sweep': results, 'max_sustainable_hz': max_hz,
            'free_run_hz': free['achieved_hz'] if free else None,
            'free_run_full_reply_pct': free['full_reply_pct'] if free else None}


# ════════════════════════════════════════════════════════════
# T4 — production-rate soak
# ════════════════════════════════════════════════════════════
def test_soak(imu, motors, ctrl, driver, seconds, csv_rows):
    print(f"\n{BOLD}=== T4: soak at {C.CONTROL_HZ} Hz for {seconds:.0f} s "
          f"==={RESET}\n")
    ctrl._needs_history_init = True
    acc = new_phase_acc()
    periods = []
    full = 0
    drop_ticks = 0
    consec = 0
    worst_consec = 0
    overruns = 0
    imu_fps = []
    n = 0
    temps_start = {str(m): driver.motor_states[m]['temperature']
                   for m in C.MOTOR_IDS}
    t_start = time.monotonic()
    next_t = t_start
    last_top = None
    while time.monotonic() - t_start < seconds:
        top = time.monotonic()
        if last_top is not None:
            periods.append((top - last_top) * 1000.0)
        last_top = top
        replies = e2e_tick(imu, motors, ctrl, acc)
        n += 1
        if replies >= len(C.MOTOR_IDS):
            full += 1
            consec = 0
        else:
            drop_ticks += 1
            if replies == 0:
                consec += 1
                worst_consec = max(worst_consec, consec)
            else:
                consec = 0
        if n % C.CONTROL_HZ == 0:
            imu_fps.append(float(imu.fps))
        csv_rows.append(('T4', C.CONTROL_HZ, n, acc['tick_busy'][-1], replies))
        next_t += C.DT
        sleep = next_t - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
        elif sleep < -0.05:
            overruns += 1
            next_t = time.monotonic()
    temps_end = {str(m): driver.motor_states[m]['temperature']
                 for m in C.MOTOR_IDS}
    jit = np.abs(np.asarray(periods) - C.DT * 1000.0)
    full_pct = 100.0 * full / max(n, 1)
    print(f"  ticks: {n}   full replies: {rate_color(full_pct)}"
          f"{full_pct:.2f}%{RESET}   drop ticks: {drop_ticks}   "
          f"worst 0-reply streak: {worst_consec} "
          f"{DIM}(HOLD trips at {C.CAN_CONSEC_DROP_LIMIT}){RESET}")
    print(f"  period jitter |dt-10ms|: p50 {pct(jit, 50):.3f}  "
          f"p95 {pct(jit, 95):.3f}  p99 {pct(jit, 99):.3f}  "
          f"max {jit.max():.3f} ms   overruns(>50ms): {overruns}")
    if imu_fps:
        print(f"  IMU under CAN load: {np.median(imu_fps):.0f} fps median")
    return {'seconds': seconds, 'ticks': n,
            'full_reply_pct': round(full_pct, 2),
            'drop_ticks': drop_ticks, 'worst_zero_reply_streak': worst_consec,
            'overruns_gt50ms': overruns,
            'period_jitter_ms': dist_ms(jit),
            'phases_ms': {k: dist_ms(v) for k, v in acc.items()},
            'imu_fps_median': round(float(np.median(imu_fps)), 1)
                              if imu_fps else None,
            'motor_temps_c': {'start': temps_start, 'end': temps_end}}


# ════════════════════════════════════════════════════════════
# Metadata + compare
# ════════════════════════════════════════════════════════════
def gather_meta(args, policy_path):
    def _git():
        try:
            return subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'],
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None

    def _sha(path):
        try:
            h = hashlib.sha256()
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(1 << 20), b''):
                    h.update(chunk)
            return h.hexdigest()[:16]
        except Exception:
            return None

    return {
        'schema': SCHEMA_VERSION,
        'adapter': args.adapter,
        'date': datetime.datetime.now().isoformat(timespec='seconds'),
        'git': _git(),
        'host': platform.node(),
        'kernel': platform.release(),
        'port': C.MOTOR_PORT, 'baud': C.MOTOR_BAUD,
        'transport': getattr(C, 'CAN_TRANSPORT', 'can2usb'),
        'can_channel': getattr(C, 'CAN_CHANNEL', None),
        'can_bitrate': getattr(C, 'CAN_BITRATE', None),
        'motor_ids': C.MOTOR_IDS,
        'policy': {'file': os.path.basename(policy_path),
                   'sha256_16': _sha(policy_path)},
        'imu': not args.no_imu,
        'rt': not args.no_rt,
        'config': {'control_hz': C.CONTROL_HZ,
                   'knee_decouple_ff': getattr(C, 'KNEE_DECOUPLE_FF', False)},
    }


def compare(path_a, path_b):
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)

    def g(d, *keys):
        for k in keys:
            d = d.get(k) if isinstance(d, dict) else None
            if d is None:
                return None
        return d

    def batch_metric(d, pacing, *keys):
        for row in d.get('batch') or []:
            if row.get('pacing_ms') == pacing:
                return g(row, *keys)
        return None

    rows = [
        ('Idle-bus unsolicited frames',
         g(a, 'idle_bus', 'unsolicited_frames'),
         g(b, 'idle_bus', 'unsolicited_frames')),
        ('RTT pooled p50 (ms)',
         g(a, 'rtt', 'pooled_ms', 'p50'), g(b, 'rtt', 'pooled_ms', 'p50')),
        ('Batch @1.0ms pacing p50 (ms)',
         batch_metric(a, 1.0, 'complete_ms', 'p50'),
         batch_metric(b, 1.0, 'complete_ms', 'p50')),
        ('Batch @0 pacing full-rate (%)',
         batch_metric(a, 0.0, 'full_rate_pct'),
         batch_metric(b, 0.0, 'full_rate_pct')),
        ('Batch @0 pacing p50 (ms)',
         batch_metric(a, 0.0, 'complete_ms', 'p50'),
         batch_metric(b, 0.0, 'complete_ms', 'p50')),
        ('Free-run e2e ceiling (Hz)',
         g(a, 'e2e', 'free_run_hz'), g(b, 'e2e', 'free_run_hz')),
        ('Max sustainable rate (Hz)',
         g(a, 'e2e', 'max_sustainable_hz'), g(b, 'e2e', 'max_sustainable_hz')),
        ('Soak full-reply (%)',
         g(a, 'soak', 'full_reply_pct'), g(b, 'soak', 'full_reply_pct')),
        ('Soak jitter p99 (ms)',
         g(a, 'soak', 'period_jitter_ms', 'p99'),
         g(b, 'soak', 'period_jitter_ms', 'p99')),
        ('Soak tick busy p95 (ms)',
         g(a, 'soak', 'phases_ms', 'tick_busy', 'p95'),
         g(b, 'soak', 'phases_ms', 'tick_busy', 'p95')),
    ]
    la = f"{g(a, 'meta', 'adapter') or a.get('adapter')} " \
         f"({(g(a, 'meta', 'date') or a.get('date', '') or '')[:10]})"
    lb = f"{g(b, 'meta', 'adapter') or b.get('adapter')} " \
         f"({(g(b, 'meta', 'date') or b.get('date', '') or '')[:10]})"
    print(f"\n{BOLD}{'metric':<34s}  {la:>20s}  {lb:>20s}{RESET}")
    for name, va, vb in rows:
        fa = '--' if va is None else f'{va:g}'
        fb = '--' if vb is None else f'{vb:g}'
        print(f"{name:<34s}  {fa:>20s}  {fb:>20s}")

    # Firmware diff: any whitelisted param that changed between runs is a
    # plant change masquerading as a transport result — flag it loudly.
    fw_a, fw_b = a.get('firmware') or {}, b.get('firmware') or {}
    diffs = []
    for label in sorted(set(fw_a) | set(fw_b)):
        pa, pb = fw_a.get(label, {}), fw_b.get(label, {})
        for k in sorted(set(pa) | set(pb)):
            if k.startswith('_'):
                continue
            if pa.get(k) != pb.get(k):
                diffs.append((label, k, pa.get(k, '--'), pb.get(k, '--')))
    if diffs:
        print(f"\n{RED}{BOLD}FIRMWARE CHANGED between runs — timing deltas "
              f"are not attributable to the transport alone:{RESET}")
        for label, k, va, vb in diffs:
            print(f"  {label}.{k}: {va} -> {vb}")
    elif fw_a and fw_b:
        print(f"\n{GREEN}Firmware params identical across runs.{RESET}")
    print()


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n')[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--adapter', default=getattr(C, 'CAN_TRANSPORT', 'can2usb'),
                    help='label stored in the JSON (default: config.'
                         'CAN_TRANSPORT — can2usb | can2spi | ...)')
    ap.add_argument('--policy', default=DEFAULT_POLICY)
    ap.add_argument('--out', default=None,
                    help='JSON output (default baseline_<adapter>_<date>.json)')
    ap.add_argument('--csv', default=None, help='optional per-tick CSV')
    ap.add_argument('--sweep-hz', default='0,50,100,150,200,300',
                    help='T3 rates; 0 = free-run')
    ap.add_argument('--sweep-seconds', type=float, default=10.0)
    ap.add_argument('--soak-seconds', type=float, default=120.0)
    ap.add_argument('--skip', default='',
                    help='comma list of tests to skip, e.g. "0,1,2"')
    ap.add_argument('--fw-csv', action='append', default=[],
                    metavar='LABEL=PATH',
                    help='embed a vendor-tool firmware dump, e.g. '
                         'rs01_pitch=fw_rs01.csv (repeatable)')
    ap.add_argument('--no-imu', action='store_true',
                    help='CAN-only run (imu phase = 0, obs gravity/gyro zeroed)')
    ap.add_argument('--no-rt', action='store_true')
    ap.add_argument('--compare', nargs=2, metavar=('OLD.json', 'NEW.json'))
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    skip = set(s.strip() for s in args.skip.split(',') if s.strip())

    if not args.no_rt:
        try:
            os.sched_setaffinity(0, {2})
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(50))
            print('[rt] SCHED_FIFO 50, CPU 2')
        except Exception as e:
            print(f'[rt] unavailable ({e}) — recording rt=False')
            args.no_rt = True

    from motor_bridge import MotorBridge
    from controller import BipedController
    from policy import Policy
    import torch
    torch.set_num_threads(1)

    meta = gather_meta(args, args.policy)
    firmware = {}
    for spec in args.fw_csv:
        label, _, path = spec.partition('=')
        if not path:
            raise SystemExit(f"--fw-csv wants LABEL=PATH, got '{spec}'")
        firmware[label] = parse_fw_csv(path)
        print(f"[fw] {label}: {len(firmware[label])} params from {path}")
    out_path = args.out or (
        f"baseline_{args.adapter}_"
        f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    transport = getattr(C, 'CAN_TRANSPORT', 'can2usb')
    print(f"{BOLD}=== can_baseline_bench — adapter '{args.adapter}' "
          f"==={RESET}")
    if args.adapter != transport:
        print(f"{RED}WARNING: --adapter '{args.adapter}' but "
              f"config.CAN_TRANSPORT='{transport}' — the JSON label will "
              f"not match the transport actually driven.{RESET}")
    if transport == 'can2spi':
        print(f"SocketCAN {getattr(C, 'CAN_CHANNEL', 'can0')} @ "
              f"{getattr(C, 'CAN_BITRATE', 1000000)} bps   "
              f"motors {C.MOTOR_IDS}   policy {meta['policy']['file']}")
    else:
        print(f"Port {C.MOTOR_PORT} @ {C.MOTOR_BAUD}   motors {C.MOTOR_IDS}   "
              f"policy {meta['policy']['file']}")
    print(f"{YELLOW}Motors will be enabled but limp (Kp=Kd=0). "
          f"Robot on stand.{RESET}")

    if args.no_imu:
        imu = NullIMU()
    else:
        from imu_adapter import IMUAdapter
        imu = IMUAdapter()
        imu.open()
        time.sleep(0.5)

    motors = MotorBridge()
    driver = motors.driver
    policy = Policy(args.policy, device='cpu')
    ctrl = BipedController(motors, imu, policy)   # obs-build code only;
    shim = make_shim(driver)                      # its state machine idles

    driver.enable_all()
    time.sleep(0.3)
    for _ in range(5):                            # warm-up, prime encoder cache
        motors.send_idle()
        time.sleep(0.03)
    motors.read_state()

    report = {'meta': meta, 'firmware': firmware, 'idle_bus': None,
              'rtt': None, 'batch': None, 'e2e': None, 'soak': None}
    csv_rows = []
    try:
        if '0' not in skip:
            report['idle_bus'] = test_idle_bus(shim, driver)
        if '1' not in skip:
            report['rtt'] = test_rtt(shim, driver)
        if '2' not in skip:
            report['batch'] = test_batch(shim, driver)
        if '3' not in skip:
            hz_list = [float(h) for h in args.sweep_hz.split(',') if h.strip()]
            report['e2e'] = test_e2e_sweep(imu, motors, ctrl, hz_list,
                                           args.sweep_seconds, csv_rows)
        if '4' not in skip:
            report['soak'] = test_soak(imu, motors, ctrl, driver,
                                       args.soak_seconds, csv_rows)
    except KeyboardInterrupt:
        print(f'\n{YELLOW}Interrupted — writing partial report.{RESET}')
    finally:
        try:
            driver.disable_all()
        except Exception as e:
            print(f'[WARN] disable: {e}')
        try:
            motors.close()
        except Exception:
            pass
        try:
            imu.close()
        except Exception:
            pass

    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\n{BOLD}Report written: {out_path}{RESET}')
    if args.csv and csv_rows:
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['test', 'target_hz', 'tick', 'busy_ms', 'replies'])
            w.writerows(csv_rows)
        print(f'Per-tick CSV: {args.csv}')
    print(f"After the swap: rerun with --adapter can2spi, then\n"
          f"    python3 can_baseline_bench.py --compare {out_path} <new.json>")


if __name__ == '__main__':
    main()
