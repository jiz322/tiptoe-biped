"""
diag_replies.py — pinpoint why CAN replies are dropping.

Standalone diagnostic. Talks directly to RobStrideDriver — does not load
motor_bridge, IMU, or policy. Runs four focused tests in ~15 seconds and
prints a guide for interpreting the results.

Tests
-----
1. Per-motor reply rate (100 ticks @ 100 Hz, 4 ms timeout).
   Tells you which motors drop, and whether it's the same motors every
   tick (= adapter buffer overrun) or random (= bus collision / motor
   timing).

2. Wait-timeout sweep (50 ticks each at 4 / 8 / 16 / 50 ms).
   If replies climb monotonically with longer waits, the motors are
   slow-but-reliable and the 4 ms timeout is too short. If they don't
   climb at all, replies aren't getting back — wiring / adapter / bus.

3. Reply arrival timing histogram (100 ticks, 10 ms timeout).
   Shows when in the wait window bytes actually arrive. A peak bunched
   at the right edge means replies are being clipped.

4. Raw byte dump (first 3 ticks).
   Lets you eyeball framing, count per-motor presence, and verify the
   parser is sane.

Usage
-----
    sudo python3 diag_replies.py

Safety
------
All commands sent are zero-everything (Kp=Kd=Tff=0) — motors stay limp
even though enable_all is required for them to reply in MIT mode.
"""

import sys
import time
from collections import Counter

import numpy as np

from robstride_biped_driver import RobStrideDriver
import config as C


# ════════════════════════════════════════════════════════════
# Display
# ════════════════════════════════════════════════════════════
DIM    = "\033[2m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
RESET  = "\033[0m"

SIM_NAMES = ['L_roll', 'L_hip', 'L_knee', 'R_roll', 'R_hip', 'R_knee']
MOTOR_TO_NAME = dict(zip(C.MOTOR_IDS, SIM_NAMES))

HEADER    = b'\x41\x54'
FRAME_LEN = 17


# ════════════════════════════════════════════════════════════
# Core: one timed burst with full metrics
# ════════════════════════════════════════════════════════════
def burst_with_metrics(driver, wait_ms=4.0, do_flush=True, capture_raw=False):
    """Send a zero-command burst to all 6 motors, time the reply.

    Returns:
        bytes_in        bytes received
        replied_ids     list of motor IDs in reply order (deduped: see below)
        first_byte_us   us from end-of-write to first new byte (or None)
        last_byte_us    us from end-of-write to last new byte (or None)
        wait_ms         actual wait duration (ms)
        timeout_hit     True if we exited via deadline rather than full reply
        raw             raw bytes (only if capture_raw=True)
    """
    frames = bytearray()
    for mid in C.MOTOR_IDS:
        frames.extend(driver._encode_mit_frame(mid, 0.0, 0.0, 0.0, 0.0, 0.0))
    expected_bytes = len(C.MOTOR_IDS) * FRAME_LEN

    with driver.lock:
        if do_flush:
            driver.ser.reset_input_buffer()
        driver.ser.write(frames)
        t_write_done = time.monotonic()

        deadline        = t_write_done + wait_ms / 1000.0
        first_byte_time = None
        last_byte_time  = None
        last_count      = 0
        timeout_hit     = True

        while True:
            now   = time.monotonic()
            count = driver.ser.in_waiting
            if count > last_count:
                if first_byte_time is None:
                    first_byte_time = now
                last_byte_time = now
                last_count     = count
            if count >= expected_bytes:
                timeout_hit = False
                break
            if now > deadline:
                break
            time.sleep(0.0001)

        t_end    = time.monotonic()
        bytes_in = driver.ser.in_waiting
        raw      = driver.ser.read_all() if bytes_in > 0 else b''

    # Parse motor IDs out of the raw stream (byte 7 of each AT-prefixed frame)
    replied_ids = []
    offset = 0
    while True:
        idx = raw.find(HEADER, offset)
        if idx == -1 or idx + FRAME_LEN > len(raw):
            break
        if idx + 7 < len(raw):
            replied_ids.append(raw[idx + 7])
        offset = idx + FRAME_LEN

    return {
        'bytes_in':       bytes_in,
        'expected_bytes': expected_bytes,
        'replied_ids':    replied_ids,
        'first_byte_us':  ((first_byte_time - t_write_done) * 1e6
                           if first_byte_time else None),
        'last_byte_us':   ((last_byte_time  - t_write_done) * 1e6
                           if last_byte_time  else None),
        'wait_ms':        (t_end - t_write_done) * 1000.0,
        'timeout_hit':    timeout_hit,
        'raw':            raw if capture_raw else None,
    }


# ════════════════════════════════════════════════════════════
# Test 1: per-motor reply rate
# ════════════════════════════════════════════════════════════
def test_per_motor_rate(driver, n_ticks=100, hz=100.0):
    print(f"\n{BOLD}=== Test 1: per-motor reply rate "
          f"({n_ticks} ticks @ {hz:.0f} Hz, 4 ms timeout) ==={RESET}\n")

    period      = 1.0 / hz
    counts      = Counter()
    full_ticks  = 0
    total_bytes = 0

    for _ in range(n_ticks):
        t0 = time.monotonic()
        r  = burst_with_metrics(driver, wait_ms=4.0, do_flush=True)
        for mid in r['replied_ids']:
            counts[mid] += 1
        if len(set(r['replied_ids']) & set(C.MOTOR_IDS)) >= len(C.MOTOR_IDS):
            full_ticks += 1
        total_bytes += r['bytes_in']
        elapsed = time.monotonic() - t0
        if elapsed < period:
            time.sleep(period - elapsed)

    expected_total = n_ticks * len(C.MOTOR_IDS) * FRAME_LEN
    print(f"  Full ticks (all 6 motors replied): "
          f"{full_ticks}/{n_ticks} ({100*full_ticks/n_ticks:.1f}%)")
    print(f"  Total bytes:  {total_bytes}/{expected_total} "
          f"({100*total_bytes/expected_total:.1f}%)\n")

    print(f"  {'motor':>6s}  {'name':<8s}  {'replies':>11s}  "
          f"{'rate':>7s}  bar")
    for mid in C.MOTOR_IDS:
        n    = counts.get(mid, 0)
        rate = 100.0 * n / n_ticks
        col  = GREEN if rate >= 95 else YELLOW if rate >= 50 else RED
        bar  = '█' * int(rate / 5)
        print(f"  {mid:>6d}  {MOTOR_TO_NAME[mid]:<8s}  "
              f"{col}{n:>4d}/{n_ticks:<4d}{RESET}     "
              f"{col}{rate:5.1f}%{RESET}  {col}{bar}{RESET}")

    unexpected = set(counts.keys()) - set(C.MOTOR_IDS)
    if unexpected:
        print(f"\n  {RED}WARN: unexpected reply IDs: "
              f"{sorted(unexpected)}{RESET}")


# ════════════════════════════════════════════════════════════
# Test 2: wait-timeout sweep
# ════════════════════════════════════════════════════════════
def test_timeout_sweep(driver, n_ticks=50):
    print(f"\n{BOLD}=== Test 2: wait-timeout sweep "
          f"({n_ticks} ticks per setting) ==={RESET}\n")

    timeouts_ms = [4.0, 8.0, 16.0, 50.0]
    expected = len(C.MOTOR_IDS) * FRAME_LEN

    print(f"  {'timeout':>9s}  {'mean bytes':>12s}  "
          f"{'mean replies':>13s}  {'full ticks':>11s}  {'mean wait':>10s}")
    print(f"  {'-'*9}  {'-'*12}  {'-'*13}  {'-'*11}  {'-'*10}")

    for tmo in timeouts_ms:
        period       = max(0.020, (tmo + 5.0) / 1000.0)
        bytes_list   = []
        replies_list = []
        wait_list    = []
        full_ticks   = 0

        for _ in range(n_ticks):
            t0 = time.monotonic()
            r  = burst_with_metrics(driver, wait_ms=tmo, do_flush=True)
            bytes_list.append(r['bytes_in'])
            replies_list.append(len(r['replied_ids']))
            wait_list.append(r['wait_ms'])
            if len(set(r['replied_ids']) & set(C.MOTOR_IDS)) >= len(C.MOTOR_IDS):
                full_ticks += 1
            elapsed = time.monotonic() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

        mean_bytes   = float(np.mean(bytes_list))
        mean_replies = float(np.mean(replies_list))
        mean_wait    = float(np.mean(wait_list))
        col = (GREEN  if full_ticks == n_ticks else
               YELLOW if full_ticks > n_ticks // 2 else RED)
        print(f"  {col}{tmo:>6.1f}ms{RESET}    "
              f"{mean_bytes:>5.1f}/{expected}    "
              f"{mean_replies:>6.2f}/6        "
              f"{full_ticks:>4d}/{n_ticks:<4d}     "
              f"{mean_wait:>6.2f}ms")


# ════════════════════════════════════════════════════════════
# Test 3: arrival timing histogram
# ════════════════════════════════════════════════════════════
def test_arrival_timing(driver, n_ticks=100, hz=50.0, wait_ms=10.0):
    print(f"\n{BOLD}=== Test 3: reply arrival timing "
          f"({n_ticks} ticks @ {hz:.0f} Hz, {wait_ms:.0f} ms timeout) ==={RESET}\n")

    period       = max(1.0 / hz, (wait_ms + 5.0) / 1000.0)
    first_us     = []
    last_us      = []
    timeout_hits = 0

    for _ in range(n_ticks):
        t0 = time.monotonic()
        r  = burst_with_metrics(driver, wait_ms=wait_ms, do_flush=True)
        if r['first_byte_us'] is not None:
            first_us.append(r['first_byte_us'])
        if r['last_byte_us'] is not None:
            last_us.append(r['last_byte_us'])
        if r['timeout_hit']:
            timeout_hits += 1
        elapsed = time.monotonic() - t0
        if elapsed < period:
            time.sleep(period - elapsed)

    if first_us:
        a = np.array(first_us) / 1000.0   # ms
        print(f"  First-byte arrival (ms after write):")
        print(f"    mean {a.mean():6.2f}  p50 {np.percentile(a,50):6.2f}  "
              f"p95 {np.percentile(a,95):6.2f}  max {a.max():6.2f}")
    if last_us:
        a = np.array(last_us) / 1000.0
        print(f"\n  Last-byte arrival (ms after write):")
        print(f"    mean {a.mean():6.2f}  p50 {np.percentile(a,50):6.2f}  "
              f"p95 {np.percentile(a,95):6.2f}  max {a.max():6.2f}")
    print(f"\n  Timeout hits: {timeout_hits}/{n_ticks}  "
          f"{DIM}(exited at deadline rather than after full reply){RESET}")

    if last_us:
        print(f"\n  Last-byte arrival histogram (1 ms bins):")
        a    = np.array(last_us) / 1000.0
        bins = np.arange(0, wait_ms + 1.01, 1.0)
        hist, _ = np.histogram(a, bins=bins)
        peak    = max(hist.max(), 1)
        for i, n in enumerate(hist):
            bar = '█' * int(40 * n / peak)
            edge = (i == len(hist) - 1)
            col  = RED if edge and n > 0 else ''
            rst  = RESET if col else ''
            print(f"    {col}{i:2d}-{i+1:2d}ms  {n:>4d}  {bar}{rst}")


# ════════════════════════════════════════════════════════════
# Test 4: raw dump
# ════════════════════════════════════════════════════════════
def test_raw_dump(driver, n_ticks=3, wait_ms=10.0):
    print(f"\n{BOLD}=== Test 4: raw byte dump "
          f"({n_ticks} ticks, {wait_ms:.0f} ms timeout) ==={RESET}\n")

    expected = set(C.MOTOR_IDS)
    expected_bytes = len(C.MOTOR_IDS) * FRAME_LEN

    for i in range(n_ticks):
        r = burst_with_metrics(driver, wait_ms=wait_ms,
                               do_flush=True, capture_raw=True)
        replied     = list(r['replied_ids'])
        replied_set = set(replied)
        missing     = sorted(expected - replied_set)
        unexpected  = sorted(replied_set - expected)

        ok  = (r['bytes_in'] == expected_bytes and not missing)
        col = GREEN if ok else YELLOW
        timeout_str = ', timeout' if r['timeout_hit'] else ''
        print(f"  Tick {i+1}: {col}{r['bytes_in']}/{expected_bytes} bytes{RESET}, "
              f"{len(replied)} reply frames "
              f"({r['wait_ms']:.2f}ms wait{timeout_str})")

        order_str = " ".join(
            f"{mid}({MOTOR_TO_NAME.get(mid,'??')})" for mid in replied
        )
        print(f"    replied:    {order_str}")

        if missing:
            miss_str = " ".join(
                f"{mid}({MOTOR_TO_NAME[mid]})" for mid in missing
            )
            print(f"    {RED}missing:    {miss_str}{RESET}")
        if unexpected:
            print(f"    {RED}unexpected: {unexpected}{RESET}")

        if r['raw'] is not None:
            hex_str = ' '.join(f'{b:02x}' for b in r['raw'])
            line_w  = 48 * 3   # 48 hex bytes per line
            for j in range(0, len(hex_str), line_w):
                print(f"    {DIM}{hex_str[j:j+line_w]}{RESET}")
        print()
        time.sleep(0.05)


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════
def main():
    print(f"{BOLD}=== diag_replies.py — CAN reply diagnostic ==={RESET}\n")
    print(f"Port: {C.MOTOR_PORT}    Motors: {C.MOTOR_IDS}\n")

    print("Initializing driver ...")
    driver = RobStrideDriver(C.MOTOR_PORT, C.MOTOR_IDS)

    print("Enabling motors (gains 0 — limp anyway) ...")
    driver.enable_all()
    time.sleep(0.3)

    # Warm-up: a few zero bursts to flush any boot-time noise
    for _ in range(5):
        burst_with_metrics(driver, wait_ms=10.0, do_flush=True)
        time.sleep(0.02)

    print(f"\nMotor ID → joint mapping:")
    for mid in C.MOTOR_IDS:
        print(f"  {mid}  →  {MOTOR_TO_NAME[mid]}")

    try:
        test_per_motor_rate(driver, n_ticks=100, hz=100.0)
        test_timeout_sweep(driver, n_ticks=50)
        test_arrival_timing(driver, n_ticks=100, hz=50.0, wait_ms=10.0)
        test_raw_dump(driver, n_ticks=3, wait_ms=10.0)

    except KeyboardInterrupt:
        print(f"\n{YELLOW}Interrupted.{RESET}")
    finally:
        try:
            for _ in range(3):
                burst_with_metrics(driver, wait_ms=2.0, do_flush=True)
                time.sleep(0.02)
            driver.disable_all()
        except Exception as e:
            print(f"[WARN] cleanup: {e}")
        try:
            driver.close()
        except Exception:
            pass

    print(f"\n{BOLD}=== Interpretation guide ==={RESET}\n")
    print("Test 1 — per-motor rate")
    print("    Same motors fail every tick (e.g. last 3 in MOTOR_IDS order)")
    print("    → adapter or host serial buffer overrun on the burst.")
    print("    Random failure spread across motors")
    print("    → bus collision / motor-side reply timing.")
    print("    All motors at similar low rate → adapter / baud / wiring.")
    print()
    print("Test 2 — timeout sweep")
    print("    Bytes climb monotonically with timeout")
    print("    → motors are slow but reliable; the 4ms cap is too tight.")
    print("    Plateau below 6 even at 50ms")
    print("    → replies aren't arriving at all, hardware or wiring problem.")
    print()
    print("Test 3 — arrival timing histogram")
    print("    Last-byte peak in the rightmost bin")
    print("    → real replies are being clipped by the timeout.")
    print("    Last-byte peak well left of timeout but bytes_in still low")
    print("    → replies vanish on the wire, not just arrive late.")
    print()
    print("Test 4 — raw dump")
    print("    Same motor IDs missing each tick → ordering / overrun.")
    print("    AT framing intact and IDs correct → parser is fine.")
    print(f"\n{BOLD}=== Done ==={RESET}\n")


if __name__ == '__main__':
    main()