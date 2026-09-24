"""
profile_balance.py — bridge / CAN / decoupling latency profiler.

Standalone instrumentation for diagnosing where the knee-decoupling
latency lives. Runs the same loop balance.py runs in HOLD mode (NN
never executes), but every tick is timed and logged to CSV.

What gets measured every tick
-----------------------------
  - loop_dt:     period between successive loop tops (loop rate jitter).
  - send_target wall time, broken into:
      transform_ms : sim→motor coord transform + gain map.
      encode_ms    : MIT frame packing for all 6 motors.
      write_ms     : serial write of all frames.
      wait_ms      : poll for replies (capped at 4ms by driver).
      parse_ms    : reply parse.
  - bytes_received vs expected (102 bytes = 6 full replies).
  - replies_parsed (how many of the 6 motors actually reported back).
  - sim_pos[6], sim_vel[6], motor_pos[6], motor_vel[6].
  - target_sim[6] (what we asked for) and motor_target[6] (what was
    actually sent over the wire after the bridge transform).
  - kp[6], kd[6] in sim joint space.

Safety
------
ALL gains start at 0 — motors are limp on entry. Use the same key
bindings as balance.py (1/2/3 + - ] [) to ramp Kp/Kd on the active
group when you are ready to do a hand-spin / step test.

Usage
-----
    sudo python3 profile_balance.py [options]

Options:
    --seconds N   Auto-quit after N seconds (default: 0 = manual quit).
    --hz HZ       Loop rate (default: 100, matches balance.py).
    --out PATH    CSV output path (default: profile.csv).
    --no-imu      Skip IMU initialization (timing-only profiling).

Keys (during run):
    1 / 2 / 3   group select (hip_roll / hip_pitch / knee)
    + / -       Kp ± KP_STEP for active group
    ] / [       Kd ± KD_STEP for active group
    r           reset all gains to 0
    s           print labeled snapshot of current state to scrollback
    q           quit, write CSV, print final report
"""

import argparse
import csv
import sys
import time
import select
import termios
import tty
from collections import deque

import numpy as np

from motor_bridge import MotorBridge
import config as C

# Optional — only imported if --no-imu is not set.
IMUAdapter = None


# ════════════════════════════════════════════════════════════
# Constants — mirror balance.py for HOLD parity
# ════════════════════════════════════════════════════════════
SPAWN_JOINT_POS = np.array(
    [0.0, 0.05, 0.0,  0.0, -0.05, 0.0], dtype=np.float32
)

KP_STEP = 0.5
KD_STEP = 0.05
KP_MAX  = 50.0
KD_MAX  = 5.0

PRINT_HZ        = 10
WINDOW_TICKS    = 200    # rolling stats window for live display

SIM_NAMES   = ['L_roll', 'L_hip', 'L_knee', 'R_roll', 'R_hip', 'R_knee']
SIM_GROUPS  = ['hip_roll', 'hip_pitch', 'knee',
               'hip_roll', 'hip_pitch', 'knee']
GROUP_ORDER = ['hip_roll', 'hip_pitch', 'knee']
GROUP_KEYS  = {'1': 'hip_roll', '2': 'hip_pitch', '3': 'knee'}

# ANSI
ESC_CLR_TO_END = "\033[J"
def ESC_UP(n): return f"\033[{n}F"
DIM, BOLD     = "\033[2m", "\033[1m"
YELLOW, GREEN = "\033[33m", "\033[32m"
CYAN, RED     = "\033[36m", "\033[31m"
RESET         = "\033[0m"


# ════════════════════════════════════════════════════════════
# Terminal helpers
# ════════════════════════════════════════════════════════════
def setup_raw_keys():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    return fd, old


def restore_keys(fd, old):
    termios.tcsetattr(fd, termios.TCSADRAIN, old)


def get_key_nonblocking():
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def expand_gains(kp_dict, kd_dict):
    kp = np.array([kp_dict[g] for g in SIM_GROUPS], dtype=np.float32)
    kd = np.array([kd_dict[g] for g in SIM_GROUPS], dtype=np.float32)
    return kp, kd


# ════════════════════════════════════════════════════════════
# Stats helpers
# ════════════════════════════════════════════════════════════
def stats(arr):
    """min/mean/p50/p95/p99/max from an iterable of numbers."""
    if len(arr) == 0:
        return None
    a = np.asarray(arr, dtype=np.float64)
    return {
        'mean': float(a.mean()),
        'p50':  float(np.percentile(a, 50)),
        'p95':  float(np.percentile(a, 95)),
        'p99':  float(np.percentile(a, 99)),
        'max':  float(a.max()),
        'min':  float(a.min()),
    }


def fmt_stats_line(label, s, unit='ms'):
    if s is None:
        return f"  {label:<12s} --"
    return (f"  {label:<12s} mean {s['mean']:5.2f}  "
            f"p50 {s['p50']:5.2f}  p95 {s['p95']:5.2f}  "
            f"p99 {s['p99']:5.2f}  max {s['max']:6.2f} {unit}")


# ════════════════════════════════════════════════════════════
# Live display
# ════════════════════════════════════════════════════════════
def build_display(*, kp, kd, active_group, target,
                  sim_pos, sim_vel, motor_pos,
                  proj_grav, ang_vel, imu_active,
                  loop_dt_window, timing_window, replies_window,
                  tick, t_elapsed, n_logged):
    lines = []

    lines.append(
        f"  {YELLOW}{BOLD}PROFILER (HOLD only){RESET}  "
        f"{DIM}target=spawn — Kp/Kd start at 0, motors limp{RESET}"
    )
    lines.append(
        f"  {DIM}1/2/3 group   +/- Kp ±{KP_STEP}   "
        f"]/[ Kd ±{KD_STEP}   r reset   s snapshot   q quit{RESET}"
    )
    lines.append("")

    # Gains
    lines.append(f"  {BOLD}    {'group':<10s}     Kp        Kd{RESET}")
    for g in GROUP_ORDER:
        is_active = (g == active_group)
        marker = f"{YELLOW}>{RESET}" if is_active else " "
        gname  = f"{BOLD}{g:<10s}{RESET}" if is_active else f"{g:<10s}"
        lines.append(
            f"  {marker}   {gname}   {kp[g]:7.2f}   {kd[g]:6.3f}"
        )
    lines.append("")

    # Joints
    lines.append(
        f"  {BOLD}{'joint':<8s}  {'target':>8s}  {'sim_pos':>8s}  "
        f"{'err':>8s}  {'sim_vel':>8s}  {'motor_pos':>10s}{RESET}"
        f"  {DIM}(sim rad){RESET}"
    )
    for i in range(6):
        err = target[i] - sim_pos[i]
        lines.append(
            f"  {SIM_NAMES[i]:<8s}  {target[i]:+8.3f}  "
            f"{sim_pos[i]:+8.3f}  {err:+8.3f}  "
            f"{sim_vel[i]:+8.3f}  {motor_pos[i]:+10.3f}"
        )
    lines.append("")

    # IMU
    if imu_active:
        lines.append(
            f"  {BOLD}IMU{RESET}  "
            f"proj_grav: ({proj_grav[0]:+.3f},{proj_grav[1]:+.3f},"
            f"{proj_grav[2]:+.3f})  "
            f"ang_vel: ({ang_vel[0]:+.3f},{ang_vel[1]:+.3f},"
            f"{ang_vel[2]:+.3f}) rad/s"
        )
    else:
        lines.append(f"  {BOLD}IMU{RESET}  {DIM}skipped (--no-imu){RESET}")
    lines.append("")

    # Timing block (always 6 lines so line_count stays stable)
    lines.append(f"  {BOLD}Timing — last {len(loop_dt_window)} ticks{RESET}")
    lines.append(fmt_stats_line('loop_dt:', stats(list(loop_dt_window))))

    if len(timing_window) > 0:
        totals     = [t['total_ms']     for t in timing_window]
        transforms = [t['transform_ms'] for t in timing_window]
        encodes    = [t['encode_ms']    for t in timing_window]
        writes     = [t['write_ms']     for t in timing_window]
        waits      = [t['wait_ms']      for t in timing_window]
        parses     = [t['parse_ms']     for t in timing_window]

        lines.append(fmt_stats_line('send_total:', stats(totals)))
        lines.append(
            f"  {DIM}phase mean: transform {np.mean(transforms):5.2f}  "
            f"encode {np.mean(encodes):5.2f}  "
            f"write {np.mean(writes):5.2f}  "
            f"wait {np.mean(waits):5.2f}  "
            f"parse {np.mean(parses):5.2f} ms{RESET}"
        )

        replies = list(replies_window)
        n_full    = sum(1 for r in replies if r >= 6)
        n_partial = sum(1 for r in replies if 0 < r < 6)
        n_zero    = sum(1 for r in replies if r == 0)
        col = (GREEN if n_full == len(replies) else
               YELLOW if n_zero == 0 else RED)
        lines.append(
            f"  replies:     {col}{n_full}/{len(replies)} full{RESET}  "
            f"{n_partial} partial  {n_zero} empty"
        )
    else:
        lines.append("  send_total:  --")
        lines.append("  phase mean:  --")
        lines.append("  replies:     --")

    lines.append("")
    lines.append(
        f"  Tick: {tick}   Time: {t_elapsed:5.1f}s   Logged: {n_logged} rows"
    )

    return lines


# ════════════════════════════════════════════════════════════
# CSV / report
# ════════════════════════════════════════════════════════════
def csv_header():
    cols = ['t_s', 'loop_dt_ms',
            'batch_total_ms', 'transform_ms',
            'encode_ms', 'write_ms', 'wait_ms', 'parse_ms',
            'bytes_in', 'bytes_expected', 'replies_ok']
    for prefix in ['sim_pos', 'sim_vel', 'motor_pos', 'motor_vel',
                   'target_sim', 'motor_target', 'kp', 'kd']:
        cols.extend([f'{prefix}_{n}' for n in SIM_NAMES])
    return cols


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(csv_header())
        for r in rows:
            w.writerow([f'{x:.6f}' if isinstance(x, float) else x for x in r])


def format_final_report(rows, csv_path, hz_target):
    if not rows:
        return "No data collected."

    arr = np.array(rows, dtype=np.float64)
    duration = float(arr[-1, 0])
    n = len(rows)

    loop_dt    = arr[:, 1]
    total_ms   = arr[:, 2]
    transform  = arr[:, 3]
    encode     = arr[:, 4]
    write      = arr[:, 5]
    wait       = arr[:, 6]
    parse      = arr[:, 7]
    bytes_in   = arr[:, 8]
    bytes_exp  = arr[:, 9]
    replies    = arr[:, 10]

    def line(label, a):
        return (f"  {label:<14s} mean {a.mean():6.3f}   "
                f"p50 {np.percentile(a,50):6.3f}   "
                f"p95 {np.percentile(a,95):6.3f}   "
                f"p99 {np.percentile(a,99):6.3f}   "
                f"max {a.max():7.3f}")

    out = []
    out.append("\n" + "=" * 72)
    out.append("FINAL REPORT")
    out.append("=" * 72)
    out.append(f"Ticks:             {n}")
    out.append(f"Duration:          {duration:.2f}s")
    out.append(f"Mean rate:         {n / duration:.1f} Hz   "
               f"(target {hz_target:.0f} Hz)")
    out.append("")
    out.append("Loop period (ms):")
    out.append(line('loop_dt', loop_dt))
    out.append("")
    out.append("send_target wall time (ms):")
    out.append(line('total', total_ms))
    out.append(line('  transform', transform))
    out.append(line('  encode',    encode))
    out.append(line('  write',     write))
    out.append(line('  wait',      wait))
    out.append(line('  parse',     parse))
    out.append("")

    out.append("Reply success:")
    n_full = int((replies == 6).sum())
    n_partial = int(((replies > 0) & (replies < 6)).sum())
    n_zero = int((replies == 0).sum())
    out.append(f"  full (6/6 replies):   "
               f"{n_full:>5d} / {n}  ({100*n_full/n:5.1f}%)")
    out.append(f"  partial (1-5):        "
               f"{n_partial:>5d} / {n}  ({100*n_partial/n:5.1f}%)")
    out.append(f"  empty (0):            "
               f"{n_zero:>5d} / {n}  ({100*n_zero/n:5.1f}%)")
    out.append(f"  bytes_in mean / max:  "
               f"{bytes_in.mean():.1f} / {bytes_in.max():.0f}  "
               f"(expected {bytes_exp[0]:.0f})")
    out.append("")

    # Quick interpretive hints — facts only, no editorializing.
    out.append("Quick interpretation hints:")
    pct_at_target = float(((np.abs(loop_dt - 1000.0/hz_target)) <
                           1000.0/hz_target * 0.1).mean()) * 100.0
    out.append(f"  - {pct_at_target:.1f}% of ticks within 10% of target period.")
    if wait.max() > 3.5:
        out.append("  - Some 'wait' phases hit the 4ms reply timeout — "
                   "replies were dropped or late.")
    if total_ms.mean() + 1.0 > 1000.0/hz_target:
        out.append(f"  - Mean send_total ({total_ms.mean():.2f}ms) is close to "
                   f"the loop budget ({1000.0/hz_target:.2f}ms).")
    if n_zero > 0 or n_partial > 0:
        out.append(f"  - {n_zero + n_partial} ticks had missing replies — "
                   "cache will hold stale state on those ticks.")
    out.append("")
    out.append(f"CSV written: {csv_path}")
    out.append(f"Columns ({len(csv_header())}): see header row.")
    return "\n".join(out)


# ════════════════════════════════════════════════════════════
# Snapshot — printed below the live display, then live display resumes
# ════════════════════════════════════════════════════════════
def format_snapshot(*, kp, kd, sim_pos, motor_pos, motor_target,
                    timing_window, replies_window):
    lines = []
    lines.append(f"{BOLD}=== Snapshot @ {time.strftime('%H:%M:%S')} ==={RESET}")
    lines.append(f"  Gains  hip_roll Kp={kp['hip_roll']:.2f} Kd={kd['hip_roll']:.3f}   "
                 f"hip_pitch Kp={kp['hip_pitch']:.2f} Kd={kd['hip_pitch']:.3f}   "
                 f"knee Kp={kp['knee']:.2f} Kd={kd['knee']:.3f}")
    lines.append(f"  {BOLD}{'joint':<8s} {'sim_pos':>9s} {'motor_pos':>11s} "
                 f"{'motor_target':>13s}{RESET}")
    for i in range(6):
        lines.append(
            f"  {SIM_NAMES[i]:<8s} {sim_pos[i]:+9.4f} "
            f"{motor_pos[i]:+11.4f} {motor_target[i]:+13.4f}"
        )
    if len(timing_window) > 0:
        totals = [t['total_ms'] for t in timing_window]
        s = stats(totals)
        replies = list(replies_window)
        n_full = sum(1 for r in replies if r >= 6)
        lines.append(
            f"  Last {len(timing_window)}: send_total mean {s['mean']:.2f} "
            f"p99 {s['p99']:.2f} max {s['max']:.2f} ms   "
            f"replies {n_full}/{len(replies)} full"
        )
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Bridge / CAN / decoupling latency profiler."
    )
    parser.add_argument('--seconds', type=float, default=0.0,
                        help='Auto-quit after N seconds (0 = manual)')
    parser.add_argument('--hz', type=float, default=100.0,
                        help='Loop rate in Hz (default 100)')
    parser.add_argument('--out', type=str, default='profile.csv',
                        help='CSV output path')
    parser.add_argument('--no-imu', action='store_true',
                        help='Skip IMU init (timing-only profiling)')
    args = parser.parse_args()

    print(f"{BOLD}=== profile_balance.py — bridge timing profiler ==={RESET}\n")
    print(f"Target loop rate: {args.hz:.1f} Hz "
          f"(period {1000.0/args.hz:.2f} ms)")
    print(f"CSV output:       {args.out}")
    if args.seconds > 0:
        print(f"Auto-quit after:  {args.seconds:.1f} s")
    print()

    print("Initializing motors ...")
    bridge = MotorBridge()

    imu = None
    imu_active = False
    if not args.no_imu:
        try:
            global IMUAdapter
            from imu_adapter import IMUAdapter as _IMU
            IMUAdapter = _IMU
            imu = IMUAdapter()
            imu.open()
            for _ in range(10):
                imu.update()
                time.sleep(0.005)
            imu_active = True
            print("Initializing IMU ... ok")
        except Exception as e:
            print(f"{YELLOW}[WARN] IMU init failed: {e}{RESET}")
            print(f"{YELLOW}[WARN] Continuing without IMU.{RESET}")
            imu = None

    bridge.enable()
    time.sleep(0.3)
    for _ in range(3):
        bridge.send_idle()
        time.sleep(0.02)
    bridge.read_state()

    print()
    print(f"{BOLD}Initial sim joint positions{RESET} "
          f"{DIM}(zeroed at {C.ZERO_POSE_NAME}){RESET}")
    print(f"  {'joint':<8s}  {'measured':>9s}  {'spawn':>9s}")
    for i in range(6):
        print(f"  {SIM_NAMES[i]:<8s}  {bridge.sim_joint_pos[i]:+9.3f}  "
              f"{SPAWN_JOINT_POS[i]:+9.3f}")
    print()
    print(f"{DIM}Motors are limp (Kp=Kd=0). Ramp Kp on the active group "
          f"with + once you're ready.{RESET}")
    print(f"{DIM}Profiler will log every tick to {args.out}.{RESET}")
    time.sleep(1.0)

    # ── State ────────────────────────────────────────────────
    kp = {g: 0.0 for g in GROUP_ORDER}
    kd = {g: 0.0 for g in GROUP_ORDER}
    active_group = 'knee'   # default to knee — that's what we're testing

    # Logging
    log_rows = []
    timing_window  = deque(maxlen=WINDOW_TICKS)
    loop_dt_window = deque(maxlen=WINDOW_TICKS)
    replies_window = deque(maxlen=WINDOW_TICKS)

    fd, old_termios = setup_raw_keys()
    period = 1.0 / args.hz
    last_print  = 0.0
    first_print = True
    line_count  = 0

    t_start       = time.monotonic()
    last_loop_top = t_start
    tick          = 0

    # Zeros for IMU placeholders if IMU not running
    zeros3 = np.zeros(3, dtype=np.float32)

    try:
        while True:
            t_loop_top = time.monotonic()
            loop_dt_s  = t_loop_top - last_loop_top
            last_loop_top = t_loop_top

            redraw_now = False

            # ── handle one keypress ─────────────────────────
            k = get_key_nonblocking()
            if k:
                redraw_now = True
                if k in GROUP_KEYS:
                    active_group = GROUP_KEYS[k]
                elif k in ('+', '='):
                    kp[active_group] = min(KP_MAX, kp[active_group] + KP_STEP)
                elif k in ('-', '_'):
                    kp[active_group] = max(0.0,    kp[active_group] - KP_STEP)
                elif k == ']':
                    kd[active_group] = min(KD_MAX, kd[active_group] + KD_STEP)
                elif k == '[':
                    kd[active_group] = max(0.0,    kd[active_group] - KD_STEP)
                elif k == 'r':
                    kp = {g: 0.0 for g in GROUP_ORDER}
                    kd = {g: 0.0 for g in GROUP_ORDER}
                elif k == 's':
                    sys.stdout.write("\n\n")
                    sys.stdout.write(format_snapshot(
                        kp=kp, kd=kd,
                        sim_pos=bridge.sim_joint_pos,
                        motor_pos=bridge._motor_pos,
                        motor_target=bridge._last_motor_target,
                        timing_window=timing_window,
                        replies_window=replies_window,
                    ))
                    sys.stdout.write("\n\n")
                    sys.stdout.flush()
                    first_print = True
                elif k in ('q', '\x03'):
                    break

            # ── refresh sensors ─────────────────────────────
            if imu is not None:
                imu.update()
            bridge.read_state()

            kp_vec, kd_vec = expand_gains(kp, kd)

            # ── target = spawn (HOLD only) ─────────────────
            target = SPAWN_JOINT_POS.copy()

            # ── send + time ────────────────────────────────
            timing = bridge.send_target(
                target.astype(np.float32),
                kp_vec, kd_vec,
                return_timing=True,
                # sequential=True
            )

            # ── log row ────────────────────────────────────
            if timing is not None:
                row = [
                    t_loop_top - t_start,
                    loop_dt_s * 1000.0,
                    timing['total_ms'],
                    timing['transform_ms'],
                    timing['encode_ms'],
                    timing['write_ms'],
                    timing['wait_ms'],
                    timing['parse_ms'],
                    int(timing['bytes_received']),
                    int(timing['expected_bytes']),
                    int(timing['replies_parsed']),
                ]
                row.extend(bridge.sim_joint_pos.tolist())
                row.extend(bridge.sim_joint_vel.tolist())
                row.extend(bridge._motor_pos.tolist())
                row.extend(bridge._motor_vel.tolist())
                row.extend(target.tolist())
                row.extend(bridge._last_motor_target.tolist())
                row.extend(kp_vec.tolist())
                row.extend(kd_vec.tolist())
                log_rows.append(row)

                timing_window.append(timing)
                replies_window.append(int(timing['replies_parsed']))

            loop_dt_window.append(loop_dt_s * 1000.0)

            # ── refresh display ─────────────────────────────
            now = time.monotonic()
            if redraw_now or (now - last_print >= 1.0 / PRINT_HZ):
                proj_grav = imu.projected_gravity() if imu is not None else zeros3
                ang_vel   = imu.ang_vel_body()      if imu is not None else zeros3

                lines = build_display(
                    kp=kp, kd=kd,
                    active_group   = active_group,
                    target         = target,
                    sim_pos        = bridge.sim_joint_pos,
                    sim_vel        = bridge.sim_joint_vel,
                    motor_pos      = bridge._motor_pos,
                    proj_grav      = proj_grav,
                    ang_vel        = ang_vel,
                    imu_active     = imu_active,
                    loop_dt_window = loop_dt_window,
                    timing_window  = timing_window,
                    replies_window = replies_window,
                    tick           = tick,
                    t_elapsed      = t_loop_top - t_start,
                    n_logged       = len(log_rows),
                )
                text = "\n".join(lines)

                if first_print:
                    sys.stdout.write(text)
                    first_print = False
                else:
                    sys.stdout.write(ESC_UP(line_count - 1))
                    sys.stdout.write(ESC_CLR_TO_END)
                    sys.stdout.write(text)
                line_count = len(lines)
                sys.stdout.flush()
                last_print = now

            tick += 1

            # ── auto-quit ───────────────────────────────────
            if args.seconds > 0 and (t_loop_top - t_start) >= args.seconds:
                break

            # ── pace control loop ───────────────────────────
            elapsed = time.monotonic() - t_loop_top
            if elapsed < period:
                time.sleep(period - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            restore_keys(fd, old_termios)
        except Exception:
            pass
        try:
            for _ in range(3):
                bridge.send_idle()
                time.sleep(0.02)
        except Exception as e:
            print(f"\n[WARN] send_idle failed: {e}")
        try:
            bridge.disable()
        except Exception as e:
            print(f"\n[WARN] motor disable failed: {e}")
        try:
            bridge.close()
        except Exception:
            pass
        if imu is not None:
            try:
                imu.close()
            except Exception:
                pass

        # Write CSV + report
        try:
            write_csv(args.out, log_rows)
        except Exception as e:
            print(f"\n[WARN] CSV write failed: {e}")
        print(format_final_report(log_rows, args.out, args.hz))
        print("\n  Final gains:")
        for g in GROUP_ORDER:
            print(f"    {g:<10s}  Kp = {kp[g]:7.2f}    Kd = {kd[g]:6.3f}")
        print("  Motors disabled. Done.\n")


if __name__ == '__main__':
    main()