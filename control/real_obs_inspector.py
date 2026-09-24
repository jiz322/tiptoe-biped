"""
real_obs_inspector.py — locked-safe sim2real observation dumper.

Companion script to sim_obs_inspector.py. There is intentionally:
    - no NN forward pass
    - no path to NN_CONTROL
    - no key to ramp Kp
    - no ability to drive motors away from their measured pose

Motors stay at Kp = Kd = 0 (limp). The only thing this script does is read
the IMU + bridge state, build the same 36-D observation that the trained
policy would see, and print/display it. You hang the robot, push joints by
hand, tilt the IMU by hand, spin the body — and watch what changes.

Pair with sim_obs_inspector.py: same physical state should produce
matching obs values. Sign mismatches in joint_vel / proj_grav / ang_vel
all show up cleanly here before any motor is energised.

Usage:
    sudo python3 real_obs_inspector.py
        h        — manually re-zero last_actions/gait_phase (HOLD reset)
        space    — pause / resume continuous printing
        d        — print full obs block right now
        q        — quit (motors disable cleanly)

Quaternion / joint conventions match balance.py:
    - quat order               : (w, x, y, z)
    - SPAWN_JOINT_POS          : (0, +0.05, 0,  0, -0.05, 0)
    - DEFAULT_JOINT_POS        : same
    - SPAWN_HEIGHT (obs[31])   : 0.42
    - ACTION_SCALES            : (0.30, 0.75, 0.51) per leg
    - GAIT_FREQ_HZ             : 1.5
"""

import math
import sys
import time
import select
import termios
import tty
import numpy as np

from motor_bridge import MotorBridge
from imu_adapter import IMUAdapter

# ════════════════════════════════════════════════════════════
# Constants — must match balance.py / rl_env_cfg
# ════════════════════════════════════════════════════════════

SPAWN_JOINT_POS = np.array(
    [0.0, 0.05, 0.0, 0.0, -0.05, 0.0], dtype=np.float32
)
DEFAULT_JOINT_POS = SPAWN_JOINT_POS.copy()
ACTION_SCALES = np.array(
    [0.30, 0.75, 0.51, 0.30, 0.75, 0.51], dtype=np.float32
)
GAIT_FREQ_HZ = 1.5
SPAWN_HEIGHT = 0.42

CONTROL_HZ = 100
PRINT_HZ   = 10

SIM_NAMES = ['L_roll', 'L_hip', 'L_knee', 'R_roll', 'R_hip', 'R_knee']

DIM    = "\033[2m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
RED    = "\033[31m"
RESET  = "\033[0m"
ESC_CLR_TO_END = "\033[J"
def ESC_UP(n): return f"\033[{n}F"


# ════════════════════════════════════════════════════════════
# Observation builder — same layout as balance.py
# ════════════════════════════════════════════════════════════

def build_obs(proj_grav, ang_vel, lin_vel,
              joint_pos, joint_vel, last_actions, gait_phase):
    joint_pos_norm = joint_pos - DEFAULT_JOINT_POS
    obs = np.concatenate([
        proj_grav.astype(np.float32),                           # 0:3
        ang_vel.astype(np.float32),                             # 3:6
        lin_vel.astype(np.float32),                             # 6:9
        joint_pos_norm.astype(np.float32),                      # 9:15
        joint_vel.astype(np.float32),                           # 15:21
        last_actions.astype(np.float32),                        # 21:27
        np.array([math.sin(gait_phase),
                  math.cos(gait_phase)], dtype=np.float32),     # 27:29
        np.array([0.0, 0.0, SPAWN_HEIGHT,
                  0.0, 0.0, 0.0, 1.0], dtype=np.float32),       # 29:36
    ]).astype(np.float32)
    assert obs.shape == (36,)
    return obs


# ════════════════════════════════════════════════════════════
# Keyboard
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


# ════════════════════════════════════════════════════════════
# Display
# ════════════════════════════════════════════════════════════

def fmt3(v):
    return f"({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f})"

def fmt6(v):
    return ", ".join(f"{x:+.2f}" for x in v)


def build_display(obs, paused, gait_phase):
    pg, av, lv = obs[0:3], obs[3:6], obs[6:9]
    jpn, jv, la = obs[9:15], obs[15:21], obs[21:27]
    sc, tail = obs[27:29], obs[29:36]

    lines = []
    state = (f"{YELLOW}{BOLD}LIVE{RESET}" if not paused
             else f"{DIM}{BOLD}PAUSED{RESET}")
    lines.append(
        f"  {RED}{BOLD}REAL OBS INSPECTOR{RESET}  "
        f"{state}   "
        f"{DIM}Kp=Kd=0 (motors limp). No NN. No motor drive.{RESET}"
    )
    lines.append(
        f"  {DIM}h=reset gait/last_act   space=pause   d=dump now   q=quit{RESET}"
    )
    lines.append("")

    lines.append(f"  {BOLD}IMU block (0:9){RESET}")
    lines.append(f"    proj_grav     {fmt3(pg)}   "
                 f"{DIM}upright→(0,0,-1){RESET}")
    lines.append(f"    ang_vel_body  {fmt3(av)} rad/s")
    lines.append(f"    lin_vel_body  {fmt3(lv)} m/s   "
                 f"{DIM}sim drops this on hw — should ~0{RESET}")
    lines.append("")

    lines.append(f"  {BOLD}joint block (9:21){RESET}")
    for i in range(6):
        lines.append(
            f"    {SIM_NAMES[i]:<8s}  pos_norm={jpn[i]:+8.4f}  "
            f"vel={jv[i]:+8.4f}  last_act={la[i]:+8.4f}"
        )
    lines.append("")

    lines.append(f"  {BOLD}tail (27:36){RESET}")
    lines.append(f"    gait sin/cos  ({sc[0]:+.3f}, {sc[1]:+.3f})  "
                 f"{DIM}phase={gait_phase:+.2f}{RESET}")
    lines.append(
        f"    diff={tail[0]:+.2f}  grace={tail[1]:+.2f}  h={tail[2]:+.2f}  "
        f"vx_cmd={tail[3]:+.2f}  yr_cmd={tail[4]:+.2f}  "
        f"yes={tail[5]:+.2f}  yec={tail[6]:+.2f}"
    )
    lines.append("")
    lines.append(
        f"  {DIM}flat 36-D: "
        + " ".join(f"{v:+.2f}" for v in obs)
        + f"{RESET}"
    )
    return lines


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main():
    print(f"{BOLD}=== real_obs_inspector.py ==={RESET}")
    print(f"{DIM}Always HOLD. Kp = Kd = 0. No NN. Hang robot, push by hand, "
          f"watch obs.{RESET}\n")

    bridge = MotorBridge()
    imu    = IMUAdapter()
    imu.open()

    for _ in range(10):
        imu.update()
        time.sleep(0.005)

    bridge.enable()
    time.sleep(0.3)
    for _ in range(3):
        bridge.send_idle()
        time.sleep(0.02)
    bridge.read_state()

    print(f"{BOLD}Initial sim joint positions{RESET}")
    print(f"  {'joint':<8s}  {'measured':>9s}  {'spawn':>9s}")
    for i in range(6):
        print(f"  {SIM_NAMES[i]:<8s}  "
              f"{bridge.sim_joint_pos[i]:+9.3f}  "
              f"{SPAWN_JOINT_POS[i]:+9.3f}")
    print()
    print(f"{DIM}Hold the robot upright for the first dump — that's your "
          f"static reference.{RESET}")
    time.sleep(1.0)

    kp_zero = np.zeros(6, dtype=np.float32)
    kd_zero = np.zeros(6, dtype=np.float32)
    last_actions = np.zeros(6, dtype=np.float32)
    gait_phase   = 0.0

    fd, old_termios = setup_raw_keys()
    period = 1.0 / CONTROL_HZ
    last_print = 0.0
    paused = False
    first_print = True
    line_count = 0

    try:
        while True:
            t_loop = time.monotonic()
            redraw_now = False
            dump_now   = False

            # ── keys ──
            k = get_key_nonblocking()
            if k:
                redraw_now = True
                if k == 'h':
                    last_actions = np.zeros(6, dtype=np.float32)
                    gait_phase = 0.0
                elif k == ' ':
                    paused = not paused
                elif k == 'd':
                    dump_now = True
                elif k in ('q', '\x03'):
                    break

            # ── sensors ──
            imu.update()
            bridge.read_state()

            obs = build_obs(
                proj_grav    = imu.projected_gravity(),
                ang_vel      = imu.ang_vel_body(),
                lin_vel      = imu.lin_vel_body(),
                joint_pos    = bridge.sim_joint_pos,
                joint_vel    = bridge.sim_joint_vel,
                last_actions = last_actions,
                gait_phase   = gait_phase,
            )

            gait_phase = (
                gait_phase + 2.0 * math.pi * GAIT_FREQ_HZ / CONTROL_HZ
            ) % (2.0 * math.pi)

            # ── motors stay limp, always. ──
            # send_target with Kp=Kd=0 = soft idle, no torque, no movement.
            bridge.send_target(
                SPAWN_JOINT_POS.astype(np.float32),
                kp_zero, kd_zero,
            )

            # ── display ──
            now = time.monotonic()
            should_redraw = redraw_now or (now - last_print >= 1.0 / PRINT_HZ)
            if should_redraw and not paused:
                lines = build_display(obs, paused, gait_phase)
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

            # ── on-demand full dump (single block, doesn't move cursor) ──
            if dump_now:
                print()
                print(f"{CYAN}{BOLD}[dump @ {time.strftime('%H:%M:%S')}]{RESET}")
                print(f"  proj_grav     {fmt3(obs[0:3])}")
                print(f"  ang_vel_body  {fmt3(obs[3:6])}")
                print(f"  lin_vel_body  {fmt3(obs[6:9])}")
                print(f"  jpos_norm    [{fmt6(obs[9:15])}]")
                print(f"  jvel         [{fmt6(obs[15:21])}]")
                print(f"  last_action  [{fmt6(obs[21:27])}]")
                print(f"  sincos       ({obs[27]:+.3f}, {obs[28]:+.3f})")
                print(f"  tail7        [{fmt6(obs[29:36])}]")
                first_print = True   # next live redraw starts fresh

            elapsed = time.monotonic() - t_loop
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
        try:
            imu.close()
        except Exception:
            pass

        print("\n\n  Motors disabled. Done.")


if __name__ == '__main__':
    main()
