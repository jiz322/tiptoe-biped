"""
joint_tuner.py — keyboard tuner for per-joint position + per-group Kp/Kd.

A stripped-down sibling of balance_log.py: no NN policy, no IMU, no logging.
Just a safe, interactive way to (a) ramp PD gains from zero and (b) nudge each
joint's position target with the keyboard while watching live feedback.

Safety model (same spirit as balance_log.py)
--------------------------------------------
  * Motors start ENABLED but LIMP — Kp = Kd = 0 for every group, so no torque
    is produced until you deliberately ramp gains up.
  * Position targets are initialized to the MEASURED pose, so the instant you
    ramp Kp the robot holds exactly where it already is (no startup jump).
    You then nudge from there.
  * send_target()'s effort soft-clip (C.REAL_EFFORT) stays on: it clamps the
    position error so Kp * |err| <= effort limit, so a far target can't yank
    a joint at full torque even with high Kp.
  * Space (or 'r') is a panic key — instantly zeroes all gains -> limp.

Assumes the motors are already zeroed correctly (run your set-zero step first).

Procedure
---------
0. Hang / support the robot. Confirm motors are zeroed.
1. sudo python3 joint_tuner.py
2. Starts LIMP (Kp=Kd=0). Targets = measured pose.
3. Select a joint (1-6) and ramp its group's Kp/Kd until it holds firmly:
       1..6     select active joint
       ] / [    Kp +/- for the active joint's group
       ' / ;    Kd +/- for the active joint's group
4. Nudge the joint position target:
       + / -    target +/- (active joint)   (= and _ also work)
       0        snap active joint target -> default pose
       z        snap ALL targets -> default pose
       v        snap ALL targets -> current measured pose (capture & hold)
       x        cycle position step size
5. Other keys:
       f        cycle action-LPF alpha (smoothing)
       r / SPACE reset ALL gains to 0  (panic -> limp)
       e / d    enable / disable motors
       q / Ctrl-C  quit (motors limp + disable cleanly)
"""

import sys
import time
import select
import termios
import tty
import numpy as np

from motor_bridge import MotorBridge
import config as C


# ════════════════════════════════════════════════════════════
# Tuning parameters
# ════════════════════════════════════════════════════════════
CONTROL_HZ = 100
PRINT_HZ   = 20

KP_STEP = 0.5
KD_STEP = 0.1
KP_MAX  = 50.0
KD_MAX  = 50.0

# Position step sizes cycled with 'x' (radians).
POS_STEPS   = [0.005, 0.01, 0.02, 0.05]
POS_STEP_0  = 1                       # index into POS_STEPS at startup (0.01)

# Action-LPF alphas cycled with 'f' (matches balance_log.py).
LPF_ALPHAS  = [0.0, 0.5, 0.6, 0.65, 0.7]
LPF_ALPHA_0 = 0.6

# Per-GROUP soft travel limit, applied as default_pos ± delta (radians).
# These are conservative guards against a fat-fingered nudge — widen them to
# match your robot's actual joint limits if you need a larger range.
POS_DELTA = {
    'hip_roll':  4.0,
    'hip_pitch': 5.0,
    'knee':      5.5,
}

# ════════════════════════════════════════════════════════════
# Joint layout (mirrors balance_log.py / config order)
# ════════════════════════════════════════════════════════════
SIM_NAMES  = ['L_roll', 'L_hip', 'L_knee', 'R_roll', 'R_hip', 'R_knee']
SIM_GROUPS = ['hip_roll', 'hip_pitch', 'knee',
              'hip_roll', 'hip_pitch', 'knee']
GROUP_ORDER = ['hip_roll', 'hip_pitch', 'knee']

DEFAULT_JOINT_POS = np.array(C.DEFAULT_JOINT_POS, dtype=np.float32)

# Precompute per-joint soft position limits from default ± group delta.
POS_LO = np.array(
    [DEFAULT_JOINT_POS[i] - POS_DELTA[SIM_GROUPS[i]] for i in range(6)],
    dtype=np.float32)
POS_HI = np.array(
    [DEFAULT_JOINT_POS[i] + POS_DELTA[SIM_GROUPS[i]] for i in range(6)],
    dtype=np.float32)


# ════════════════════════════════════════════════════════════
# Terminal helpers
# ════════════════════════════════════════════════════════════
ESC_CLR_TO_END = "\033[J"
def ESC_UP(n): return f"\033[{n}F"
DIM    = "\033[2m"
BOLD   = "\033[1m"
YELLOW = "\033[33m"
GREEN  = "\033[32m"
CYAN   = "\033[36m"
RED    = "\033[31m"
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


def expand_gains(kp_dict, kd_dict):
    """Per-group gain dicts -> 6-vectors in joint order."""
    kp = np.array([kp_dict[g] for g in SIM_GROUPS], dtype=np.float32)
    kd = np.array([kd_dict[g] for g in SIM_GROUPS], dtype=np.float32)
    return kp, kd


# ════════════════════════════════════════════════════════════
# Display
# ════════════════════════════════════════════════════════════
def build_display(*, kp, kd, active_joint, pos_step, lpf_alpha, enabled,
                  target, sim_pos, sim_vel, tau_meas, temp_c):
    lines = []
    active_group = SIM_GROUPS[active_joint]

    state = (f"{GREEN}{BOLD}ENABLED{RESET}" if enabled
             else f"{YELLOW}{BOLD}DISABLED (limp){RESET}")
    any_gain = any(kp[g] > 0 for g in GROUP_ORDER)
    limp_tag = "" if any_gain else f"   {YELLOW}● all gains 0 → limp{RESET}"
    lines.append(f"  Motors: {state}{limp_tag}")
    lines.append(
        f"  {DIM}1-6 joint   +/- pos ±{POS_STEPS[pos_step]:<5g} "
        f"]/[ Kp ±{KP_STEP}   '/; Kd ±{KD_STEP}{RESET}"
    )
    lines.append(
        f"  {DIM}0 joint→default  z all→default  v all→measured  "
        f"x pos-step  f LPF α={lpf_alpha:.2f}{RESET}"
    )
    lines.append(
        f"  {DIM}r/SPACE limp   e enable   d disable   q quit{RESET}"
    )
    lines.append("")

    # Per-group gains
    lines.append(f"  {BOLD}    {'group':<10s}     Kp        Kd{RESET}")
    for g in GROUP_ORDER:
        is_active = (g == active_group)
        marker = f"{YELLOW}>{RESET}" if is_active else " "
        gname  = f"{BOLD}{g:<10s}{RESET}" if is_active else f"{g:<10s}"
        lines.append(f"  {marker}   {gname}   {kp[g]:7.2f}   {kd[g]:6.3f}")

    lines.append("")

    # Per-joint table
    lines.append(
        f"  {BOLD}  {'joint':<8s}  {'target':>7s}  {'pos':>7s}  "
        f"{'err':>7s}  {'vel':>7s}  {'τ_meas':>7s}  {'temp':>6s}{RESET}  "
        f"{DIM}(sim rad | motor Nm | °C){RESET}"
    )
    for i in range(6):
        is_active = (i == active_joint)
        marker = f"{YELLOW}>{RESET}" if is_active else " "
        nm = f"{BOLD}{SIM_NAMES[i]:<8s}{RESET}" if is_active else f"{SIM_NAMES[i]:<8s}"
        err = target[i] - sim_pos[i]

        t = temp_c[i]
        if   t >= 80.0: temp_str = f"{RED}{t:6.1f}{RESET}"
        elif t >= 60.0: temp_str = f"{YELLOW}{t:6.1f}{RESET}"
        else:           temp_str = f"{t:6.1f}"

        lines.append(
            f"  {marker} {nm}  "
            f"{target[i]:+7.3f}  {sim_pos[i]:+7.3f}  "
            f"{err:+7.3f}  {sim_vel[i]:+7.3f}  "
            f"{tau_meas[i]:+7.2f}  {temp_str}"
        )

    return lines


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════
def main():
    print(f"{BOLD}=== joint_tuner.py — keyboard position + Kp/Kd tuner ==={RESET}\n")
    print("Initializing motors ...")

    bridge = MotorBridge()
    bridge.set_lpf_alpha(LPF_ALPHA_0)

    bridge.enable()
    enabled = True
    time.sleep(0.3)

    for _ in range(3):
        bridge.send_idle()
        time.sleep(0.02)
    bridge.read_state()

    # Targets start at the MEASURED pose so ramping Kp holds in place.
    target = bridge.sim_joint_pos.astype(np.float32).copy()

    zero_name = getattr(C, 'ZERO_POSE_NAME', 'set-zero pose')
    print(f"\n{BOLD}Initial sim joint positions{RESET} "
          f"{DIM}(motors were zeroed at {zero_name}){RESET}")
    print(f"  {'joint':<8s}  {'measured':>9s}  {'default':>9s}")
    for i in range(6):
        print(f"  {SIM_NAMES[i]:<8s}  {bridge.sim_joint_pos[i]:+9.3f}  "
              f"{DEFAULT_JOINT_POS[i]:+9.3f}")
    print(f"\n{DIM}Motors are LIMP (Kp=Kd=0). Ramp gains with ]/[ and '/; "
          f"before nudging position.{RESET}")
    time.sleep(1.0)

    # ── State ────────────────────────────────────────────────
    kp = {g: 0.0 for g in GROUP_ORDER}
    kd = {g: 0.0 for g in GROUP_ORDER}
    active_joint = 0
    pos_step_idx = POS_STEP_0
    lpf_idx      = LPF_ALPHAS.index(LPF_ALPHA_0) if LPF_ALPHA_0 in LPF_ALPHAS else 2

    fd, old_termios = setup_raw_keys()
    period      = 1.0 / CONTROL_HZ
    last_print  = 0.0
    first_print = True
    line_count  = 0

    try:
        while True:
            t_loop = time.monotonic()
            redraw_now = False

            # ── handle one keypress ─────────────────────────
            k = get_key_nonblocking()
            if k:
                redraw_now = True
                g = SIM_GROUPS[active_joint]
                step = POS_STEPS[pos_step_idx]

                if k in '123456':
                    active_joint = int(k) - 1

                elif k in ('+', '='):
                    target[active_joint] = min(
                        POS_HI[active_joint], target[active_joint] + step)
                elif k in ('-', '_'):
                    target[active_joint] = max(
                        POS_LO[active_joint], target[active_joint] - step)

                elif k == ']':
                    kp[g] = min(KP_MAX, kp[g] + KP_STEP)
                elif k == '[':
                    kp[g] = max(0.0,    kp[g] - KP_STEP)
                elif k == "'":
                    kd[g] = min(KD_MAX, kd[g] + KD_STEP)
                elif k == ';':
                    kd[g] = max(0.0,    kd[g] - KD_STEP)

                elif k == '0':
                    target[active_joint] = DEFAULT_JOINT_POS[active_joint]
                elif k == 'z':
                    target = DEFAULT_JOINT_POS.copy()
                elif k == 'v':
                    target = bridge.sim_joint_pos.astype(np.float32).copy()

                elif k == 'x':
                    pos_step_idx = (pos_step_idx + 1) % len(POS_STEPS)

                elif k == 'f':
                    lpf_idx = (lpf_idx + 1) % len(LPF_ALPHAS)
                    bridge.set_lpf_alpha(LPF_ALPHAS[lpf_idx])
                    bridge.reset_filter()

                elif k in ('r', ' '):
                    kp = {gg: 0.0 for gg in GROUP_ORDER}
                    kd = {gg: 0.0 for gg in GROUP_ORDER}

                elif k == 'e':
                    if not enabled:
                        bridge.enable()
                        enabled = True
                        bridge.read_state()
                        # Re-capture pose so we don't jump on re-enable.
                        target = bridge.sim_joint_pos.astype(np.float32).copy()
                elif k == 'd':
                    if enabled:
                        bridge.disable()
                        enabled = False

                elif k in ('q', '\x03'):
                    break

            # ── refresh feedback ────────────────────────────
            bridge.read_state()
            kp_vec, kd_vec = expand_gains(kp, kd)

            # ── drive motors (or idle if disabled) ──────────
            if enabled:
                bridge.send_target(target.astype(np.float32), kp_vec, kd_vec)
                # effort_limit defaults to C.REAL_EFFORT inside send_target.
            else:
                bridge.send_idle()

            # ── refresh display ─────────────────────────────
            now = time.monotonic()
            if redraw_now or (now - last_print >= 1.0 / PRINT_HZ):
                lines = build_display(
                    kp=kp, kd=kd,
                    active_joint=active_joint,
                    pos_step=pos_step_idx,
                    lpf_alpha=bridge._lpf_alpha,
                    enabled=enabled,
                    target=target,
                    sim_pos=bridge.sim_joint_pos,
                    sim_vel=bridge.sim_joint_vel,
                    tau_meas=bridge.motor_torque,
                    temp_c=bridge.motor_temperature,
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

            # ── pace control loop ───────────────────────────
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

        print("\n\n  Final gains:")
        for g in GROUP_ORDER:
            print(f"    {g:<10s}  Kp = {kp[g]:7.2f}    Kd = {kd[g]:6.3f}")
        print("  Final targets (sim rad):")
        for i in range(6):
            print(f"    {SIM_NAMES[i]:<8s}  {target[i]:+7.3f}")
        print("  Motors disabled. Done.")


if __name__ == '__main__':
    main()
