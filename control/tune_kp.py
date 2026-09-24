"""
tune_kp_kd.py — tune Kp and Kd for all 6 motors holding the kneel pose.

Procedure:
    1. PLACE THE ROBOT AT KNEEL POSE BEFORE RUNNING.
       The script captures current motor positions at startup and uses
       them as the hold target. Display is in sim coordinates (after
       sign/offset/gear/coupling), so values should look like
       KNEEL_JOINT_POS while the robot sits at kneel.
    2. sudo python3 tune_kp_kd.py
    3. Controls:
         1 / 2 / 3 : select group (hip_roll / hip_pitch / knee)
         + / -     : adjust active group's Kp by ±KP_STEP
         ] / [     : adjust active group's Kd by ±KD_STEP
         c         : re-capture current pose as the new hold target
         r         : reset all gains to start values
         q  Ctrl-C : quit (motors disable, final values printed)

The PD controller still operates in raw motor coordinates internally —
we just convert to sim for display and command the captured raw target
to each motor.
"""

import math
import sys
import time
import select
import termios
import tty
import config as C
from robstride_socketcan_driver import make_driver

# ════════════════════════════════════════════════════════════
# Tuning parameters — edit these
# ════════════════════════════════════════════════════════════

KP_START = {
    'hip_roll':  0.0,
    'hip_pitch': 0.0,
    'knee':      0.0,
}
KD_START = {
    'hip_roll':  0.0,
    'hip_pitch': 0.0,
    'knee':      0.0,
}

KP_STEP = 0.1      # ±KP_STEP per +/- press
KD_STEP = 0.05     # ±KD_STEP per ]/[ press

KP_MAX = 200.0
KD_MAX = 5.0

CONTROL_HZ = 100
PRINT_HZ   = 20

# ════════════════════════════════════════════════════════════

# Sim joint ordering — parallel to C.MOTOR_IDS (positional, NOT by ID).
# C.MOTOR_IDS[i] is the motor that physically drives SIM_NAMES[i].
SIM_NAMES  = ['L_roll', 'L_hip', 'L_knee', 'R_roll', 'R_hip', 'R_knee']
SIM_GROUPS = ['hip_roll', 'hip_pitch', 'knee',
              'hip_roll', 'hip_pitch', 'knee']

GROUP_ORDER = ['hip_roll', 'hip_pitch', 'knee']
GROUP_KEYS  = {'1': 'hip_roll', '2': 'hip_pitch', '3': 'knee'}

# ── ANSI helpers ──────────────────────────────────────────────
ESC_CLR_TO_END = "\033[J"
def ESC_UP(n): return f"\033[{n}F"
DIM    = "\033[2m"
BOLD   = "\033[1m"
YELLOW = "\033[33m"
RESET  = "\033[0m"


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


def limp_ping(driver):
    """Send zero command to all motors and read replies — populates state."""
    for mid in driver.motor_ids:
        driver.send_command(mid, 0, 0, 0, 0, 0)
        time.sleep(0.001)


def read_raw(driver):
    """Limp-ping all motors and return raw positions in MOTOR_IDS order."""
    limp_ping(driver)
    return [driver.get_state(mid)[0] for mid in driver.motor_ids]


def compute_sim(raw):
    """Apply sign / offset / knee gear / hip coupling — same as test_motors."""
    sign = C.JOINT_SIGN
    off = C.JOINT_OFFSET
    gear = C.KNEE_GEAR_RATIO
    coupling = C.KNEE_HIP_COUPLING
    sim = [raw[i] * sign[i] + off[i] for i in range(6)]
    sim[2] = raw[2] * sign[2] * gear - sim[1] * coupling + off[2]
    sim[5] = raw[5] * sign[5] * gear - sim[4] * coupling + off[5]
    return sim


def main():
    driver = make_driver(C.MOTOR_IDS)
    driver.enable_all()
    time.sleep(0.3)

    # Populate state cache and capture hold target
    for _ in range(3):
        read_raw(driver)
        time.sleep(0.03)

    held_raw = read_raw(driver)
    held_sim = compute_sim(held_raw)

    print(f"\n  {BOLD}Captured hold target{RESET} "
          f"(place robot at kneel before launch):")
    print(f"  {'joint':<8s}  {'raw':>8s}  {'sim':>8s}")
    for i in range(6):
        print(f"  {SIM_NAMES[i]:<8s}  "
              f"{held_raw[i]:+8.3f}  {held_sim[i]:+8.3f}")
    print()
    print(f"  {DIM}Verify the sim column matches your expected kneel pose. "
          f"Press 'c' later to re-capture.{RESET}")
    time.sleep(1.0)

    kp = dict(KP_START)
    kd = dict(KD_START)
    active_group = 'hip_roll'

    fd, old_termios = setup_raw_keys()
    first_print = True
    last_print = 0.0
    period = 1.0 / CONTROL_HZ

    try:
        print(f"\n  {BOLD}KP/KD TUNING{RESET}  target = captured pose above")
        print(f"  {DIM}1/2/3 select group   "
              f"+/- Kp ±{KP_STEP}   ]/[ Kd ±{KD_STEP}   "
              f"c recapture   r reset   q quit{RESET}\n")

        while True:
            t_loop = time.monotonic()
            redraw_now = False

            # ── handle keypress ──────────────────────────────
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
                elif k == 'c':
                    held_raw = [driver.get_state(mid)[0] for mid in C.MOTOR_IDS]
                    held_sim = compute_sim(held_raw)
                elif k == 'r':
                    kp = dict(KP_START)
                    kd = dict(KD_START)
                elif k in ('q', '\x03'):
                    break

            # ── send PD commands in raw motor coords ─────────
            cmds = []
            for i, mid in enumerate(C.MOTOR_IDS):
                group = SIM_GROUPS[i]
                cmds.append(
                    (mid, held_raw[i], 0.0, kp[group], kd[group], 0.0)
                )
            driver.send_batch(cmds)

            # ── refresh display ──────────────────────────────
            now = time.monotonic()
            if redraw_now or (now - last_print >= 1.0 / PRINT_HZ):
                actual_raw = [driver.get_state(mid)[0] for mid in C.MOTOR_IDS]
                actual_sim = compute_sim(actual_raw)

                lines = []

                # Group gains
                lines.append(
                    f"  {BOLD}    {'group':<10s}     Kp       Kd{RESET}"
                )
                for g in GROUP_ORDER:
                    is_active = (g == active_group)
                    if is_active:
                        marker = f"{YELLOW}>{RESET}"
                        gname  = f"{BOLD}{g:<10s}{RESET}"
                    else:
                        marker = " "
                        gname  = f"{g:<10s}"
                    lines.append(
                        f"  {marker}   {gname}   {kp[g]:7.2f}   {kd[g]:6.3f}"
                    )

                # Joint table — pos/target/err in sim, vel/tff in raw
                lines.append("")
                lines.append(
                    f"  {BOLD}{'joint':<8s}    {'target':>8s}  "
                    f"{'pos':>8s}  {'err':>8s}  {'vel':>8s}  {'tff':>8s}"
                    f"{RESET}  {DIM}(sim rad){RESET}"
                )
                for i, mid in enumerate(C.MOTOR_IDS):
                    err = held_sim[i] - actual_sim[i]
                    _, vel, tff = driver.get_state(mid)
                    lines.append(
                        f"  {SIM_NAMES[i]:<8s}    "
                        f"{held_sim[i]:+8.3f}  {actual_sim[i]:+8.3f}  "
                        f"{err:+8.3f}  {vel:+8.3f}  {tff:+8.3f}"
                    )

                text = "\n".join(lines)

                if first_print:
                    sys.stdout.write(text)
                    first_print = False
                else:
                    sys.stdout.write(ESC_UP(len(lines) - 1))
                    sys.stdout.write(ESC_CLR_TO_END)
                    sys.stdout.write(text)
                sys.stdout.flush()
                last_print = now

            # ── pace the control loop ────────────────────────
            elapsed = time.monotonic() - t_loop
            if elapsed < period:
                time.sleep(period - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        restore_keys(fd, old_termios)
        for _ in range(2):
            for mid in driver.motor_ids:
                driver.send_command(mid, 0, 0, 0, 0, 0)
                time.sleep(0.001)
            time.sleep(0.02)
        driver.disable_all()
        driver.close()

        print("\n\n  Final tuned gains:")
        for g in GROUP_ORDER:
            print(f"    {g:<10s}  Kp = {kp[g]:7.2f}    Kd = {kd[g]:6.3f}")
        print("  done.")


if __name__ == '__main__':
    main()