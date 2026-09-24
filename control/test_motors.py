"""
test_motors.py — zero at kneel and read positions, with a labeled live readout.

Usage:
    1. Place the robot in the KNEEL pose.
    2. sudo python3 test_motors.py
    3. The script auto-zeros all motors, then shows a live readout.
    4. Manually move the robot toward the standing pose — sim values
       should read close to DEFAULT_JOINT_POS (~0).
    5. Ctrl-C to quit.

    --no-zero   skip the auto-zero (transport-swap bring-up, runbook
                Phase B3: MechOffset lives in the motor, so the point is
                to VERIFY the surviving zeros — not overwrite them).
"""

import math
import sys
import time
import config as C
from robstride_socketcan_driver import make_driver

TWO_PI = 2 * math.pi
WRAP = (0, 1, 3, 4)

# Short, position-aware labels for the 6 sim joints.
# Order matches C.SIM_JOINT_NAMES / C.MOTOR_IDS.
SHORT_LABELS = ['L_roll', 'L_hip', 'L_knee', 'R_roll', 'R_hip', 'R_knee']

# Highlight a sim cell when it's within this many rad of the target pose.
TOL_RAD = 0.15

# ── ANSI escape helpers ────────────────────────────────────────
ESC_CLR_TO_END = "\033[J"           # clear from cursor to end of screen
def ESC_UP(n):  return f"\033[{n}F" # cursor up n lines, to column 0
GREEN = "\033[32m"
DIM   = "\033[2m"
RESET = "\033[0m"


def wrap(v):
    while v > math.pi:  v -= TWO_PI
    while v < -math.pi: v += TWO_PI
    return v


def limp(driver):
    for mid in driver.motor_ids:
        driver.send_command(mid, 0, 0, 0, 0, 0)
        time.sleep(0.001)


def read_raw(driver):
    limp(driver)
    out = []
    for i, mid in enumerate(driver.motor_ids):
        p = driver.get_state(mid)[0]
        out.append(p)
    return out


def compute_sim(raw):
    """Apply sign / offset / knee gear / hip coupling — matches motor_bridge."""
    sign = C.JOINT_SIGN
    off = C.JOINT_OFFSET
    gear = C.KNEE_GEAR_RATIO
    coupling = C.KNEE_HIP_COUPLING
    sim = [raw[i] * sign[i] + off[i] for i in range(6)]
    sim[2] = raw[2] * sign[2] * gear - sim[1] * coupling + off[2]
    sim[5] = raw[5] * sign[5] * gear - sim[4] * coupling + off[5]
    return sim


def _cell(v, target=None):
    """One numeric cell, colored green if close to target."""
    s = f"{v:+7.2f}"
    if target is not None and abs(v - target) < TOL_RAD:
        return f"{GREEN}{s}{RESET}"
    return s


def format_row(label, values, targets=None, dim=False):
    """Labeled row, split into LEFT | RIGHT halves of 3 joints each."""
    cells = [
        _cell(v, targets[i] if targets is not None else None)
        for i, v in enumerate(values)
    ]
    left  = "  ".join(cells[0:3])
    right = "  ".join(cells[3:6])
    line = f"  {label:<11s}{left}   |   {right}"
    if dim:
        line = f"{DIM}{line}{RESET}"
    return line


def render_header(motor_ids):
    name_cells = [f"{n:>6s}" for n in SHORT_LABELS]
    id_cells   = [f"{f'({mid})':>6s}" for mid in motor_ids]
    pad = " " * 13   # matches "  {label:<11s}" indent
    line1 = pad + "  ".join(name_cells[0:3]) + "   |   " + "  ".join(name_cells[3:6])
    line2 = pad + "  ".join(id_cells[0:3])   + "   |   " + "  ".join(id_cells[3:6])
    return line1, line2


def main():
    ids = C.MOTOR_IDS
    no_zero = '--no-zero' in sys.argv

    driver = make_driver(ids)
    driver.enable_all()
    # CAN_TIMEOUT=200 ms firmware watchdog (P11 §0.5): no silent gaps
    # >200 ms after enable — feed limp frames through the settle instead.
    for _ in range(4):
        limp(driver)
        time.sleep(0.05)

    try:
        # ── Auto-zero at kneel ───────────────────────────────
        if no_zero:
            print(f"\n  --no-zero: keeping the motors' stored zeros "
                  f"(swap bring-up verification).")
        else:
            print(f"\n  Assuming robot is in KNEEL pose — zeroing all motors...")
            driver.set_zero_all()
            time.sleep(0.1)
            print(f"  Zeroed. Software offsets will shift to sim coordinates.")
        print(f"  Move robot to standing → sim values should read ~0.\n")

        # ── Live readout ─────────────────────────────────────
        print(f"  Live readout (Ctrl-C to quit)")
        print(f"  Green cells = sim value within {TOL_RAD:.2f} rad of closest target pose.\n")

        header1, header2 = render_header(ids)
        first = True

        while True:
            raw = read_raw(driver)
            sim = compute_sim(raw)
            sim_deg = [math.degrees(v) for v in sim]
            # Pick whichever reference pose this sim is closest to
            d_def = sum(abs(sim[i] - C.DEFAULT_JOINT_POS[i]) for i in range(6))
            d_knl = sum(abs(sim[i] - C.KNEEL_JOINT_POS[i])   for i in range(6))
            if d_def <= d_knl:
                closest_name, closest_target = "default", C.DEFAULT_JOINT_POS
            else:
                closest_name, closest_target = "kneel",   C.KNEEL_JOINT_POS

            block = [
                header1,
                header2,
                format_row("raw (rad):", raw),
                format_row("sim (rad):", sim, targets=closest_target),
                format_row("sim (deg):", sim_deg),
                format_row("→ default:", C.DEFAULT_JOINT_POS, dim=True),
                format_row("→ kneel:",   C.KNEEL_JOINT_POS,   dim=True),
                f"  closest pose: {closest_name}   "
                f"(Σ|err|  default={d_def:5.2f}  kneel={d_knl:5.2f})",
            ]
            text = "\n".join(block)

            if first:
                sys.stdout.write(text)
                first = False
            else:
                sys.stdout.write(ESC_UP(len(block) - 1))
                sys.stdout.write(ESC_CLR_TO_END)
                sys.stdout.write(text)
            sys.stdout.flush()
            # 0.15 s + read overhead keeps the per-motor command period
            # under the 200 ms firmware CAN_TIMEOUT window (P11 §0.5).
            time.sleep(0.15)

    except KeyboardInterrupt:
        print()
    finally:
        limp(driver)
        time.sleep(0.02)
        limp(driver)
        driver.disable_all()
        driver.close()
        print("  done.")


if __name__ == '__main__':
    main()