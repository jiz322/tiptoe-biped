"""
test_position.py — live readout of sim positions in rad and deg.

No zeroing. Just enables the motors, holds them limp, and prints
sim (rad) and sim (deg) in the labeled LEFT | RIGHT format.

Usage:
    sudo python3 test_position.py
    Ctrl-C to quit.
"""

import math
import sys
import time
import config as C
from robstride_socketcan_driver import make_driver
print(C.JOINT_OFFSET)
# Order matches C.SIM_JOINT_NAMES / C.MOTOR_IDS.
SHORT_LABELS = ['L_roll', 'L_hip', 'L_knee', 'R_roll', 'R_hip', 'R_knee']

# ── ANSI escape helpers ────────────────────────────────────────
ESC_CLR_TO_END = "\033[J"
def ESC_UP(n): return f"\033[{n}F"


def limp(driver):
    for mid in driver.motor_ids:
        driver.send_command(mid, 0, 0, 0, 0, 0)
        time.sleep(0.001)


def read_raw(driver):
    limp(driver)
    out = []
    for mid in driver.motor_ids:
        p, _, _, _ = driver.get_state(mid)
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


def format_row(label, values):
    """Labeled row, split into LEFT | RIGHT halves of 3 joints each."""
    cells = [f"{v:+7.2f}" for v in values]
    left = "  ".join(cells[0:3])
    right = "  ".join(cells[3:6])
    return f"  {label:<11s}{left}   |   {right}"


def render_header(motor_ids):
    name_cells = [f"{n:>6s}" for n in SHORT_LABELS]
    id_cells = [f"{f'({mid})':>6s}" for mid in motor_ids]
    pad = " " * 13   # matches "  {label:<11s}" indent
    line1 = pad + "  ".join(name_cells[0:3]) + "   |   " + "  ".join(name_cells[3:6])
    line2 = pad + "  ".join(id_cells[0:3])   + "   |   " + "  ".join(id_cells[3:6])
    return line1, line2


def main():
    ids = C.MOTOR_IDS

    driver = make_driver(ids)
    driver.enable_all()
    time.sleep(0.3)

    try:
        print(f"\n  Live readout (Ctrl-C to quit)\n")

        header1, header2 = render_header(ids)
        first = True

        while True:
            raw = read_raw(driver)
            sim = compute_sim(raw)
            sim_deg = [math.degrees(v) for v in sim]

            block = [
                header1,
                header2,
                format_row("sim (rad):", sim),
                format_row("sim (deg):", sim_deg),
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
            time.sleep(0.2)

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