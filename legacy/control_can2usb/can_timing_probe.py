"""
can_timing_probe.py — identify the physical mechanism behind the 1 ms
inter-frame pacing requirement, and measure whether latency can be cut.

Background (git archaeology, 2026-05-04 tuning session):
    send_batch() pacing was hand-tuned: sleeps 0.3/0.5/0.7/0.9/1.0/1.7 ms
    and deadlines 1-9 ms were tried across ~12 commits, converging on
    1.0 ms sleep + 2.8 ms deadline = 9.3 ms tick busy = the 100 Hz cap.
    Shorter sleeps caused per-motor reply dropouts ("huge freq difference
    between individual motors"). WHY was never established. This probe
    discriminates the candidate mechanisms:

    H1  Adapter concurrency limit — the AT-protocol MCU cannot forward a
        CAN reply to serial while it is receiving/transmitting the next
        command; replies of LATER motors in the burst drop first, and
        reply-triggered pacing (test C) sustains 100%.
    H2  USB full-speed frame quantization — reply bytes reach pyserial
        only on ~1 ms USB frame boundaries; arrival events (test A)
        cluster at integer milliseconds and no software pacing beats a
        ~2-3 ms floor. Fix is a different transport, not different code.
    H3  Replies are LATE, not lost — they arrive after the 2.8 ms
        deadline and are destroyed by the next tick's
        reset_input_buffer() (test D counts stragglers). Fix is
        asynchronous reply draining, no pacing needed at all.
    H4  Bus/wiring marginality — drops are random across motors and
        persist even at 1.0 ms pacing (test B). Fix is termination/
        wiring, not software.

Tests
-----
    A. Arrival micro-timing: single write() of all 6 frames (no pacing),
       12 ms observation window, every in_waiting increment timestamped.
    B. Inter-frame sleep sweep 0.0-1.2 ms with a generous 8 ms window —
       measures per-motor reply rate AND true completion time separately
       (the historical tuning conflated them via the fixed deadline).
    C. Reply-triggered pacing: next command is sent as soon as the
       previous motor's reply bytes are in — the candidate replacement
       for fixed sleeps. Predicts the achievable tick time with 100%
       replies if H1 holds (~4-5 ms vs today's 9.3 ms).
    D. Straggler count: production constants (1.0 ms / 2.8 ms), then
       keep listening 6 ms past the deadline.

Usage
-----
    sudo python3 can_timing_probe.py            # all tests
    sudo python3 can_timing_probe.py --skip A,B,C,D   # round-2 tests only
    sudo python3 can_timing_probe.py --csv out.csv    # + raw event dump

Round-2 tests (added after the 2026-08-12 run settled H1):
    E. Single-motor RTT — is the 2.7 ms batch tail a fixed per-transaction
       adapter latency (RTT ~2.7 even alone) or queueing (RTT <1)?
    F. Windowed pipelining at 0.5 ms spacing — directly probes the
       in-flight transaction cap W (model says 3) and measures the true
       floor batch time this adapter allows.
    G. No-wait send + next-tick drain at 100 Hz — validates deleting the
       2.8 ms reply spin (replies parsed next tick, same freshness as
       today's read_state) before touching the production driver.

Safety
------
Same pattern as diag_replies.py: all frames are zero-everything
(Kp=Kd=Tff=0), motors limp throughout. enable_all() is required for
MIT-mode replies; disable_all() on every exit path. Robot on stand.
"""

import argparse
import csv
import time
from collections import Counter

import numpy as np

from robstride_biped_driver import RobStrideDriver
import config as C

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
GREEN, YELLOW, RED = "\033[32m", "\033[33m", "\033[31m"

SIM_NAMES = ['L_roll', 'L_hip', 'L_knee', 'R_roll', 'R_hip', 'R_knee']
MOTOR_TO_NAME = dict(zip(C.MOTOR_IDS, SIM_NAMES))
HEADER = b'\x41\x54'
FRAME_LEN = 17
N_MOTORS = len(C.MOTOR_IDS)
EXPECTED = N_MOTORS * FRAME_LEN


def zero_frames(driver):
    """One encoded zero-command frame per motor, in MOTOR_IDS order."""
    return [driver._encode_mit_frame(mid, 0.0, 0.0, 0.0, 0.0, 0.0)
            for mid in C.MOTOR_IDS]


def parse_ids(raw):
    """Motor IDs present in a raw reply stream, in arrival order."""
    ids, offset = [], 0
    while True:
        idx = raw.find(HEADER, offset)
        if idx == -1 or idx + FRAME_LEN > len(raw):
            break
        ids.append(raw[idx + 7])
        offset = idx + FRAME_LEN
    return ids


def rate_color(rate):
    return GREEN if rate >= 99 else YELLOW if rate >= 90 else RED


def watch_arrivals(driver, t_ref, window_s, stop_at_bytes=None):
    """Tight-poll in_waiting until window elapses (or byte target met).

    Returns (events, bytes_final): events = [(t_ms_since_ref, total_bytes)]
    recorded at every in_waiting increase.
    """
    deadline = t_ref + window_s
    events, last = [], 0
    while True:
        now = time.monotonic()
        n = driver.ser.in_waiting
        if n > last:
            events.append(((now - t_ref) * 1000.0, n))
            last = n
        if stop_at_bytes is not None and n >= stop_at_bytes:
            break
        if now > deadline:
            break
    return events, last


# ════════════════════════════════════════════════════════════
# Test A — single-burst arrival micro-timing
# ════════════════════════════════════════════════════════════
def test_arrival_quantization(driver, n_ticks=100, window_ms=12.0):
    print(f"\n{BOLD}=== Test A: single-write burst, arrival micro-timing "
          f"({n_ticks} ticks, {window_ms:.0f} ms window) ==={RESET}\n")

    frames = b''.join(zero_frames(driver))
    counts = Counter()
    all_event_ms = []
    complete_ms = []
    per_tick_rows = []

    for tick in range(n_ticks):
        with driver.lock:
            driver.ser.reset_input_buffer()
            t0 = time.monotonic()
            driver.ser.write(frames)
            events, nbytes = watch_arrivals(driver, t0, window_ms / 1000.0,
                                            stop_at_bytes=EXPECTED)
            raw = driver.ser.read_all() if nbytes else b''
        ids = parse_ids(raw)
        for mid in ids:
            counts[mid] += 1
        all_event_ms.extend(t for t, _ in events)
        if nbytes >= EXPECTED and events:
            complete_ms.append(events[-1][0])
        per_tick_rows.append(('A', tick, 0.0, nbytes, len(ids),
                              events[-1][0] if events else -1.0))
        time.sleep(0.02)

    print(f"  {'motor':>6s}  {'name':<8s}  {'reply rate':>10s}")
    for mid in C.MOTOR_IDS:
        r = 100.0 * counts.get(mid, 0) / n_ticks
        print(f"  {mid:>6d}  {MOTOR_TO_NAME[mid]:<8s}  "
              f"{rate_color(r)}{r:9.1f}%{RESET}")

    if complete_ms:
        a = np.array(complete_ms)
        print(f"\n  Full-reply completion (ms after write): "
              f"p50 {np.percentile(a, 50):.2f}  p95 {np.percentile(a, 95):.2f}"
              f"  max {a.max():.2f}   ({len(complete_ms)}/{n_ticks} full)")
    else:
        print(f"\n  {RED}No tick ever completed all {EXPECTED} bytes.{RESET}")

    if all_event_ms:
        a = np.array(all_event_ms)
        frac = a - np.round(a)          # distance to nearest integer ms
        print(f"\n  Arrival-event alignment to 1 ms grid "
              f"(|t - round(t)|, ms): p50 {np.percentile(np.abs(frac), 50):.3f}"
              f"  p90 {np.percentile(np.abs(frac), 90):.3f}")
        print(f"  {DIM}p50 < 0.15 with multi-modal arrivals => H2 USB "
              f"1 ms quantization; flat => not quantized{RESET}")
        bins = np.arange(0.0, window_ms + 0.25, 0.25)
        hist, _ = np.histogram(a, bins=bins)
        peak = max(hist.max(), 1)
        print(f"\n  Arrival-event histogram (0.25 ms bins):")
        for i, n in enumerate(hist):
            if n == 0:
                continue
            print(f"    {bins[i]:5.2f}-{bins[i + 1]:5.2f} ms  {n:>5d}  "
                  f"{'█' * int(40 * n / peak)}")
    return per_tick_rows


# ════════════════════════════════════════════════════════════
# Test B — inter-frame sleep sweep, generous window
# ════════════════════════════════════════════════════════════
def test_sleep_sweep(driver, n_ticks=100, window_ms=8.0):
    print(f"\n{BOLD}=== Test B: inter-frame sleep sweep "
          f"({n_ticks} ticks each, {window_ms:.0f} ms observation window) "
          f"==={RESET}\n")

    sleeps_ms = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    frames = zero_frames(driver)
    rows = []

    print(f"  {'sleep':>7s}  {'full':>6s}  {'complete p50/p95':>17s}  "
          f"{'per-motor reply % (' + '/'.join(str(m) for m in C.MOTOR_IDS) + ')':<40s}")
    for sl in sleeps_ms:
        counts = Counter()
        full = 0
        complete_ms = []
        for tick in range(n_ticks):
            with driver.lock:
                driver.ser.reset_input_buffer()
                t0 = time.monotonic()
                for f in frames:
                    driver.ser.write(f)
                    if sl > 0:
                        time.sleep(sl / 1000.0)
                events, nbytes = watch_arrivals(
                    driver, t0, (sl * N_MOTORS + window_ms) / 1000.0,
                    stop_at_bytes=EXPECTED)
                raw = driver.ser.read_all() if nbytes else b''
            ids = parse_ids(raw)
            for mid in ids:
                counts[mid] += 1
            if nbytes >= EXPECTED:
                full += 1
                complete_ms.append(events[-1][0])
            rows.append(('B', tick, sl, nbytes, len(ids),
                         events[-1][0] if events else -1.0))
            time.sleep(0.02)

        rates = [100.0 * counts.get(mid, 0) / n_ticks for mid in C.MOTOR_IDS]
        worst = min(rates)
        if complete_ms:
            a = np.array(complete_ms)
            comp = f"{np.percentile(a, 50):5.2f}/{np.percentile(a, 95):5.2f}ms"
        else:
            comp = "    never    "
        rate_str = "/".join(f"{r:.0f}" for r in rates)
        print(f"  {sl:5.1f}ms  {rate_color(100.0 * full / n_ticks)}"
              f"{full:>4d}/{n_ticks}{RESET}  {comp:>17s}  "
              f"{rate_color(worst)}{rate_str}{RESET}")

    print(f"\n  {DIM}Later-position motors dropping first at short sleeps "
          f"=> H1 adapter concurrency.\n  Random motors dropping even at "
          f"1.0+ ms => H4 wiring/termination.{RESET}")
    return rows


# ════════════════════════════════════════════════════════════
# Test C — reply-triggered pacing (candidate replacement)
# ════════════════════════════════════════════════════════════
def test_reply_triggered(driver, n_ticks=200, per_motor_timeout_ms=2.0):
    print(f"\n{BOLD}=== Test C: reply-triggered pacing "
          f"({n_ticks} ticks, {per_motor_timeout_ms:.1f} ms per-motor "
          f"timeout) ==={RESET}\n")

    frames = zero_frames(driver)
    counts = Counter()
    full = 0
    total_ms = []
    rtt_ms = [[] for _ in range(N_MOTORS)]
    rows = []

    for tick in range(n_ticks):
        with driver.lock:
            driver.ser.reset_input_buffer()
            t0 = time.monotonic()
            timeouts = 0
            for i, f in enumerate(frames):
                t_w = time.monotonic()
                driver.ser.write(f)
                want = (i + 1) * FRAME_LEN
                _, nbytes = watch_arrivals(
                    driver, t_w, per_motor_timeout_ms / 1000.0,
                    stop_at_bytes=want)
                if nbytes >= want:
                    rtt_ms[i].append((time.monotonic() - t_w) * 1000.0)
                else:
                    timeouts += 1
            t_total = (time.monotonic() - t0) * 1000.0
            raw = driver.ser.read_all() if driver.ser.in_waiting else b''
        ids = parse_ids(raw)
        for mid in ids:
            counts[mid] += 1
        if len(set(ids) & set(C.MOTOR_IDS)) >= N_MOTORS:
            full += 1
        total_ms.append(t_total)
        rows.append(('C', tick, -1.0, len(raw), len(ids), t_total))
        time.sleep(0.02)

    a = np.array(total_ms)
    print(f"  Full ticks: {rate_color(100.0 * full / n_ticks)}"
          f"{full}/{n_ticks}{RESET}")
    print(f"  Batch total (ms): p50 {np.percentile(a, 50):.2f}  "
          f"p95 {np.percentile(a, 95):.2f}  max {a.max():.2f}   "
          f"{DIM}(production send_batch: ~9.3 ms){RESET}")
    print(f"\n  {'motor':>6s}  {'name':<8s}  {'reply rate':>10s}  "
          f"{'RTT p50':>8s}  {'RTT p95':>8s}")
    for i, mid in enumerate(C.MOTOR_IDS):
        r = 100.0 * counts.get(mid, 0) / n_ticks
        if rtt_ms[i]:
            b = np.array(rtt_ms[i])
            p50, p95 = np.percentile(b, 50), np.percentile(b, 95)
            print(f"  {mid:>6d}  {MOTOR_TO_NAME[mid]:<8s}  "
                  f"{rate_color(r)}{r:9.1f}%{RESET}  {p50:7.2f}  {p95:7.2f}")
        else:
            print(f"  {mid:>6d}  {MOTOR_TO_NAME[mid]:<8s}  "
                  f"{rate_color(r)}{r:9.1f}%{RESET}  {'--':>7s}  {'--':>7s}")
    full_rate = 100.0 * full / n_ticks
    p50_total = float(np.percentile(a, 50))
    if full_rate >= 99.0 and p50_total < 6.0:
        print(f"\n  {DIM}VERDICT: reply-triggered pacing works on this "
              f"adapter — H1 confirmed, p50 total\n  {p50_total:.2f} ms is "
              f"the new achievable tick cost.{RESET}")
    else:
        print(f"\n  {DIM}VERDICT: reply-triggered pacing FAILED here "
              f"({full}/{n_ticks} full, p50 {p50_total:.2f} ms)\n  — do "
              f"NOT adopt it; the adapter serializes transactions "
              f"internally, so waiting on\n  each reply just adds L per "
              f"motor. (2026-08-12 session 2: 0/200 @ 12.03 ms.){RESET}")
    return rows


# ════════════════════════════════════════════════════════════
# Test D — stragglers past the production deadline
# ════════════════════════════════════════════════════════════
def test_stragglers(driver, n_ticks=100, extra_ms=6.0):
    print(f"\n{BOLD}=== Test D: production constants + {extra_ms:.0f} ms "
          f"extra listen ({n_ticks} ticks) ==={RESET}\n")

    frames = zero_frames(driver)
    in_time, late, never = 0, 0, 0
    late_bytes_hist = []

    for _ in range(n_ticks):
        with driver.lock:
            driver.ser.reset_input_buffer()
            for f in frames:
                driver.ser.write(f)
                time.sleep(0.0010)               # production pacing
            t_wd = time.monotonic()
            deadline = t_wd + 0.0028             # production deadline
            while driver.ser.in_waiting < EXPECTED:
                if time.monotonic() > deadline:
                    break
            n_at_deadline = driver.ser.in_waiting
            _, n_final = watch_arrivals(driver, time.monotonic(),
                                        extra_ms / 1000.0,
                                        stop_at_bytes=EXPECTED)
            driver.ser.read_all()
        if n_at_deadline >= EXPECTED:
            in_time += 1
        elif n_final >= EXPECTED:
            late += 1
            late_bytes_hist.append(EXPECTED - n_at_deadline)
        else:
            never += 1
        time.sleep(0.02)

    print(f"  complete by 2.8 ms deadline:    {in_time}/{n_ticks}")
    print(f"  complete only in extra window:  "
          f"{YELLOW if late else GREEN}{late}/{n_ticks}{RESET}"
          + (f"   (median missing at deadline: "
             f"{int(np.median(late_bytes_hist))} bytes)" if late else ""))
    print(f"  never complete:                 "
          f"{RED if never else GREEN}{never}/{n_ticks}{RESET}")
    print(f"\n  {DIM}late >> never => H3: replies survive but the next "
          f"tick's reset_input_buffer\n  destroys them — async draining "
          f"recovers them with zero pacing risk.{RESET}")


# ════════════════════════════════════════════════════════════
# Test E — single-motor RTT (isolates per-transaction latency L)
# ════════════════════════════════════════════════════════════
def test_single_rtt(driver, n_reps=50, timeout_ms=6.0):
    print(f"\n{BOLD}=== Test E: single-motor round-trip latency "
          f"({n_reps} reps/motor, {timeout_ms:.0f} ms timeout) ==={RESET}\n")
    print(f"  {DIM}Model check: batch completion fit 6*sleep + 2.7 ms, "
          f"implying per-transaction\n  latency L ~= 2.7 ms even on a "
          f"quiet bus. If single RTT here is ~2.7 ms, L is\n  adapter/USB "
          f"transaction cost (fixed). If it is <1 ms, the 2.7 ms is "
          f"queueing and\n  smarter scheduling can compress the batch."
          f"{RESET}\n")

    print(f"  {'motor':>6s}  {'name':<8s}  {'ok':>7s}  {'RTT p50':>8s}  "
          f"{'RTT p10':>8s}  {'RTT p95':>8s}")
    all_p50 = []
    for mid in C.MOTOR_IDS:
        frame = driver._encode_mit_frame(mid, 0.0, 0.0, 0.0, 0.0, 0.0)
        rtts = []
        for _ in range(n_reps):
            with driver.lock:
                driver.ser.reset_input_buffer()
                t0 = time.monotonic()
                driver.ser.write(frame)
                _, nbytes = watch_arrivals(driver, t0, timeout_ms / 1000.0,
                                           stop_at_bytes=FRAME_LEN)
                if nbytes >= FRAME_LEN:
                    rtts.append((time.monotonic() - t0) * 1000.0)
                driver.ser.read_all()
            time.sleep(0.01)
        if rtts:
            a = np.array(rtts)
            p50 = np.percentile(a, 50)
            all_p50.append(p50)
            print(f"  {mid:>6d}  {MOTOR_TO_NAME[mid]:<8s}  "
                  f"{len(rtts):>4d}/{n_reps}  {p50:7.2f}  "
                  f"{np.percentile(a, 10):7.2f}  {np.percentile(a, 95):7.2f}")
        else:
            print(f"  {mid:>6d}  {MOTOR_TO_NAME[mid]:<8s}  "
                  f"{RED}0/{n_reps}{RESET}")
    if all_p50:
        print(f"\n  Pooled RTT p50: {np.median(all_p50):.2f} ms   "
              f"{DIM}(physics floor ~0.7 ms; excess = adapter internal)"
              f"{RESET}")


# ════════════════════════════════════════════════════════════
# Test F — windowed pipelining (probes the in-flight limit W)
# ════════════════════════════════════════════════════════════
def test_windowed(driver, n_ticks=100, spacing_ms=0.5, timeout_ms=15.0):
    print(f"\n{BOLD}=== Test F: windowed pipelining, spacing "
          f"{spacing_ms:.1f} ms ({n_ticks} ticks/window) ==={RESET}\n")
    print(f"  {DIM}Keeps at most W commands unanswered. If the adapter "
          f"truly caps in-flight\n  transactions at 3, W=2/3 stay ~100% "
          f"and W=4+ starts dropping. Completion time\n  at the best "
          f"reliable W is the floor achievable with THIS adapter.\n"
          f"  ⚠ NOT reproducible across sessions (2026-08-12: W=4 read "
          f"100/100 in one\n  session, 96/100 in another; W=6 17/100 vs "
          f"39/100). W=4 sits on the adapter's\n  concurrency cliff — "
          f"require 100% in MULTIPLE sessions before trusting any "
          f"W>2.{RESET}\n")

    frames = zero_frames(driver)
    print(f"  {'W':>3s}  {'full':>8s}  {'complete p50/p95':>17s}  "
          f"{'worst-motor %':>13s}")
    for W in (2, 3, 4, 6):
        counts = Counter()
        full = 0
        complete_ms = []
        for _ in range(n_ticks):
            with driver.lock:
                driver.ser.reset_input_buffer()
                t0 = time.monotonic()
                sent = 0
                last_write = -1e9
                deadline = t0 + timeout_ms / 1000.0
                while True:
                    now = time.monotonic()
                    got = driver.ser.in_waiting // FRAME_LEN
                    if (sent < N_MOTORS and sent - got < W
                            and (now - last_write) >= spacing_ms / 1000.0):
                        driver.ser.write(frames[sent])
                        sent += 1
                        last_write = time.monotonic()
                    if sent == N_MOTORS and \
                            driver.ser.in_waiting >= EXPECTED:
                        break
                    if now > deadline:
                        break
                t_done = (time.monotonic() - t0) * 1000.0
                nbytes = driver.ser.in_waiting
                raw = driver.ser.read_all() if nbytes else b''
            ids = parse_ids(raw)
            for mid in ids:
                counts[mid] += 1
            if nbytes >= EXPECTED:
                full += 1
                complete_ms.append(t_done)
            time.sleep(0.02)
        worst = min(100.0 * counts.get(mid, 0) / n_ticks
                    for mid in C.MOTOR_IDS)
        if complete_ms:
            a = np.array(complete_ms)
            comp = f"{np.percentile(a, 50):5.2f}/{np.percentile(a, 95):5.2f}ms"
        else:
            comp = "    never    "
        print(f"  {W:>3d}  {rate_color(100.0 * full / n_ticks)}"
              f"{full:>4d}/{n_ticks}{RESET}  {comp:>17s}  "
              f"{rate_color(worst)}{worst:12.0f}%{RESET}")


# ════════════════════════════════════════════════════════════
# Test G — no-wait send + next-tick drain at 100 Hz (async model)
# ════════════════════════════════════════════════════════════
def test_nowait_drain(driver, n_ticks=300, period_ms=10.0, sleep_ms=1.0):
    print(f"\n{BOLD}=== Test G: production pacing, NO reply wait, drain at "
          f"next tick ({n_ticks} ticks @ {1000.0 / period_ms:.0f} Hz) "
          f"==={RESET}\n")
    print(f"  {DIM}Candidate loop restructure: send_batch returns right "
          f"after the last write;\n  replies (all in by last-write "
          f"+2.7 ms) are parsed at the NEXT tick before\n  sending — same "
          f"freshness as today's read_state, minus the 2.8 ms spin."
          f"{RESET}\n")

    frames = zero_frames(driver)
    counts = Counter()
    full = 0
    busy_ms = []
    with driver.lock:
        driver.ser.reset_input_buffer()
    for tick in range(n_ticks):
        t0 = time.monotonic()
        with driver.lock:
            raw = driver.ser.read_all() if driver.ser.in_waiting else b''
            for f in frames:
                driver.ser.write(f)
                time.sleep(sleep_ms / 1000.0)
        busy_ms.append((time.monotonic() - t0) * 1000.0)
        if tick > 0:                       # tick 0 has nothing to drain
            ids = parse_ids(raw)
            for mid in ids:
                counts[mid] += 1
            if len(set(ids) & set(C.MOTOR_IDS)) >= N_MOTORS:
                full += 1
        el = time.monotonic() - t0
        if el < period_ms / 1000.0:
            time.sleep(period_ms / 1000.0 - el)
    with driver.lock:                      # collect the final tick's replies
        time.sleep(0.005)
        driver.ser.read_all()

    n_scored = n_ticks - 1
    a = np.array(busy_ms)
    print(f"  Full drains: {rate_color(100.0 * full / n_scored)}"
          f"{full}/{n_scored}{RESET}")
    for mid in C.MOTOR_IDS:
        r = 100.0 * counts.get(mid, 0) / n_scored
        print(f"    {mid} {MOTOR_TO_NAME[mid]:<8s} "
              f"{rate_color(r)}{r:6.1f}%{RESET}")
    print(f"  tick busy (send only): p50 {np.percentile(a, 50):.2f}  "
          f"p95 {np.percentile(a, 95):.2f}  max {a.max():.2f} ms   "
          f"{DIM}(production send+wait: ~9.3){RESET}")
    print(f"\n  {DIM}100% full drains here => the 2.8 ms wait can be "
          f"deleted with zero data loss;\n  frees ~2.8 ms/tick of margin "
          f"(jitter headroom), same obs freshness.{RESET}")


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, default=None,
                    help='dump per-tick rows (test, tick, sleep_ms, bytes, '
                         'replies, complete_ms) to CSV')
    ap.add_argument('--skip', type=str, default='',
                    help='comma list of tests to skip, e.g. "A,D"')
    args = ap.parse_args()
    skip = set(s.strip().upper() for s in args.skip.split(',') if s.strip())

    print(f"{BOLD}=== can_timing_probe.py — CAN pacing mechanism probe "
          f"==={RESET}")
    print(f"Port {C.MOTOR_PORT} @ {C.MOTOR_BAUD}    motors {C.MOTOR_IDS}")
    print(f"{YELLOW}Motors will be enabled but limp (zero gains). "
          f"Robot on stand.{RESET}\n")

    driver = RobStrideDriver(C.MOTOR_PORT, C.MOTOR_IDS, baud=C.MOTOR_BAUD)
    driver.enable_all()
    time.sleep(0.3)
    frames = b''.join(zero_frames(driver))
    for _ in range(5):                            # warm-up / flush boot noise
        with driver.lock:
            driver.ser.reset_input_buffer()
            driver.ser.write(frames)
        time.sleep(0.03)
        with driver.lock:
            driver.ser.read_all()

    rows = []
    try:
        if 'A' not in skip:
            rows += test_arrival_quantization(driver)
        if 'B' not in skip:
            rows += test_sleep_sweep(driver)
        if 'C' not in skip:
            rows += test_reply_triggered(driver)
        if 'D' not in skip:
            test_stragglers(driver)
        if 'E' not in skip:
            test_single_rtt(driver)
        if 'F' not in skip:
            test_windowed(driver)
        if 'G' not in skip:
            test_nowait_drain(driver)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Interrupted.{RESET}")
    finally:
        try:
            driver.disable_all()
        except Exception as e:
            print(f"[WARN] disable: {e}")
        try:
            driver.close()
        except Exception:
            pass

    if args.csv and rows:
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['test', 'tick', 'sleep_ms', 'bytes_in',
                        'replies', 'complete_ms'])
            w.writerows(rows)
        print(f"\nCSV written: {args.csv}")

    print(f"\n{BOLD}=== Decision guide ==={RESET}")
    print("  These are IF-THEN branches — check each condition against the "
          "numbers above\n  before acting on its arrow. "
          "(2026-08-12 result: C failed 0/200, F did not\n  reproduce "
          "across sessions; only G's no-wait drain validated on this "
          "adapter.)")
    print("  C at 100% and p50 total < 6 ms  -> adopt reply-triggered "
          "pacing in send_batch\n     (config-gated), keep 100 Hz, pocket "
          "the latency. C below 100% -> the\n     adapter serializes "
          "internally; waiting per reply only adds latency.")
    print("  A shows 1 ms-grid arrivals      -> USB transport floor; "
          "software cannot beat it.\n     Path to lower latency/higher Hz "
          "= SocketCAN hardware (CAN HAT / gs_usb\n     adapter), or a "
          "second adapter with 3 motors per bus.")
    print("  D shows late >> never           -> add async reply drain; "
          "drop the wait\n     entirely; replies from tick k consumed at "
          "tick k+1 (already the obs model).")
    print("  F: only trust a window W at 100% across MULTIPLE sessions; "
          "a 96-99% W is\n     on the concurrency cliff and will silently "
          "drop frames in production.")
    print("  B random drops even at 1.2 ms   -> check bus termination/"
          "wiring before anything.")


if __name__ == '__main__':
    main()
