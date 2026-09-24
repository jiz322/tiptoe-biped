"""
joint_tuner_mirror.py — keyboard tuner that moves LEFT + RIGHT joints together
as a mirrored pair (right side gets the opposite sign), plus per-group Kp/Kd.

Same safety model as joint_tuner.py:
  * Motors start ENABLED but LIMP (Kp = Kd = 0).
  * send_target()'s effort soft-clip (C.REAL_EFFORT) stays on.
  * Space / 'r' is a panic key -> all gains 0 -> limp.

Mirrored position model
-----------------------
You tune ONE signed value Δ per pair (roll / hip / knee). It is applied as:

    left_target  = default_left  + Δ
    right_target = default_right + RIGHT_SIGN[pair] * Δ      (RIGHT_SIGN = -1)

so Δ = 0 is the default (symmetric) pose and a nudge drives the two sides
oppositely. RIGHT_SIGN is per-pair: if one joint type actually needs the SAME
sign on both sides, set its entry to +1.

Targets are always exactly mirror-symmetric by construction. On startup (and on
'v') Δ is fitted to the measured pose with a symmetric best-fit, so ramping Kp
won't jerk the robot.

Assumes the motors are already zeroed correctly.

Keys
----
    1 / 2 / 3   select pair        (hip_roll / hip_pitch / knee)
    + / -       Δ +/- for the active pair        (mirrored; = and _ also work)
    ] / [       Kp +/- for the active pair's group
    ' / ;       Kd +/- for the active pair's group
    0           active pair Δ -> 0   (both sides -> default)
    z           ALL Δ -> 0           (whole robot -> default pose)
    v           re-fit Δ from current measured pose (symmetric)
    x           cycle position step size
    f           cycle action-LPF alpha
    r / SPACE   reset ALL gains to 0  (panic -> limp)
    e / d       enable / disable motors
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

# Action-LPF alphas cycled with 'f'.
LPF_ALPHAS  = [0.0, 0.5, 0.6, 0.65, 0.7]
LPF_ALPHA_0 = 0.6

# Max |Δ| per pair (radians). Soft guard against a runaway keypress — widen to
# match your robot. 1.0 rad ≈ 57°, 2.0 ≈ 115°, 3.14 ≈ 180°.
DELTA_MAX = {
    'hip_roll':  2.0,
    'hip_pitch': 3.0,
    'knee':      3.5,
}


# ════════════════════════════════════════════════════════════
# Joint / pair layout (mirrors config order)
#   idx: 0=L_roll 1=L_hip 2=L_knee 3=R_roll 4=R_hip 5=R_knee
# ════════════════════════════════════════════════════════════
SIM_NAMES  = ['L_roll', 'L_hip', 'L_knee', 'R_roll', 'R_hip', 'R_knee']
SIM_GROUPS = ['hip_roll', 'hip_pitch', 'knee',
              'hip_roll', 'hip_pitch', 'knee']
GROUP_ORDER = ['hip_roll', 'hip_pitch', 'knee']

PAIR_LEFT  = {'hip_roll': 0, 'hip_pitch': 1, 'knee': 2}
PAIR_RIGHT = {'hip_roll': 3, 'hip_pitch': 4, 'knee': 5}

# right_target = default_right + RIGHT_SIGN[pair] * Δ
# All -1 → left and right move oppositely. Flip an entry to +1 if a given
# joint type should move the SAME direction on both sides.
RIGHT_SIGN = {'hip_roll': -1.0, 'hip_pitch': -1.0, 'knee': -1.0}

GROUP_KEYS = {'1': 'hip_roll', '2': 'hip_pitch', '3': 'knee'}

DEFAULT_JOINT_POS = np.array(C.DEFAULT_JOINT_POS, dtype=np.float32)


def build_target(d):
    """Build the 6-vector sim target from per-pair signed deltas `d`."""
    t = DEFAULT_JOINT_POS.copy()
    for g in GROUP_ORDER:
        L, R = PAIR_LEFT[g], PAIR_RIGHT[g]
        t[L] = DEFAULT_JOINT_POS[L] + d[g]
        t[R] = DEFAULT_JOINT_POS[R] + RIGHT_SIGN[g] * d[g]
    return t


def fit_delta_from_pose(measured):
    """Symmetric best-fit of Δ per pair from a measured 6-vector pose.

    left  ≈ def_L + Δ            -> Δ ≈ meas_L - def_L
    right ≈ def_R + sign·Δ       -> Δ ≈ (meas_R - def_R)·sign   (sign = ±1)
    Average the two for the closest symmetric target, then clamp.
    """
    d = {}
    for g in GROUP_ORDER:
        L, R = PAIR_LEFT[g], PAIR_RIGHT[g]
        dl = measured[L] - DEFAULT_JOINT_POS[L]
        dr = (measured[R] - DEFAULT_JOINT_POS[R]) * RIGHT_SIGN[g]
        val = 0.5 * (dl + dr)
        lim = DELTA_MAX[g]
        d[g] = float(max(-lim, min(lim, val)))
    return d


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
    kp = np.array([kp_dict[g] for g in SIM_GROUPS], dtype=np.float32)
    kd = np.array([kd_dict[g] for g in SIM_GROUPS], dtype=np.float32)
    return kp, kd


# ════════════════════════════════════════════════════════════
# Display
# ════════════════════════════════════════════════════════════
def build_display(*, kp, kd, d, active_group, pos_step, lpf_alpha, enabled,
                  target, sim_pos, sim_vel, tau_meas, temp_c):
    lines = []

    state = (f"{GREEN}{BOLD}ENABLED{RESET}" if enabled
             else f"{YELLOW}{BOLD}DISABLED (limp){RESET}")
    any_gain = any(kp[g] > 0 for g in GROUP_ORDER)
    limp_tag = "" if any_gain else f"   {YELLOW}● all gains 0 → limp{RESET}"
    lines.append(f"  Motors: {state}{limp_tag}")
    lines.append(
        f"  {DIM}1-3 pair   +/- Δ ±{POS_STEPS[pos_step]:<5g} "
        f"(mirrored)   ]/[ Kp ±{KP_STEP}   '/; Kd ±{KD_STEP}{RESET}"
    )
    lines.append(
        f"  {DIM}0 pair→default  z all→default  v refit measured  "
        f"x pos-step  f LPF α={lpf_alpha:.2f}{RESET}"
    )
    lines.append(
        f"  {DIM}r/SPACE limp   e enable   d disable   q quit{RESET}"
    )
    lines.append("")

    # Per-pair gains + delta + resulting L/R targets
    lines.append(
        f"  {BOLD}    {'pair':<10s}     Kp        Kd      Δ      "
        f"{'L_tgt':>7s}  {'R_tgt':>7s}{RESET}"
    )
    for g in GROUP_ORDER:
        is_active = (g == active_group)
        marker = f"{YELLOW}>{RESET}" if is_active else " "
        gname  = f"{BOLD}{g:<10s}{RESET}" if is_active else f"{g:<10s}"
        L, R = PAIR_LEFT[g], PAIR_RIGHT[g]
        sgn = "+1" if RIGHT_SIGN[g] > 0 else "-1"
        lines.append(
            f"  {marker}   {gname}   {kp[g]:7.2f}   {kd[g]:6.3f}   "
            f"{d[g]:+6.3f}   {target[L]:+7.3f}  {target[R]:+7.3f}  "
            f"{DIM}R={sgn}·Δ{RESET}"
        )

    lines.append("")

    # Per-joint table (both joints of the active pair highlighted)
    lines.append(
        f"  {BOLD}  {'joint':<8s}  {'target':>7s}  {'pos':>7s}  "
        f"{'err':>7s}  {'vel':>7s}  {'τ_meas':>7s}  {'temp':>6s}{RESET}  "
        f"{DIM}(sim rad | motor Nm | °C){RESET}"
    )
    for i in range(6):
        is_active = (SIM_GROUPS[i] == active_group)
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
    print(f"{BOLD}=== joint_tuner_mirror.py — mirrored L/R position + Kp/Kd ==={RESET}\n")
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

    # Fit Δ to measured pose so ramping Kp holds near where we are.
    d = fit_delta_from_pose(bridge.sim_joint_pos)
    target = build_target(d)

    zero_name = getattr(C, 'ZERO_POSE_NAME', 'set-zero pose')
    print(f"\n{BOLD}Initial sim joint positions{RESET} "
          f"{DIM}(motors were zeroed at {zero_name}){RESET}")
    print(f"  {'joint':<8s}  {'measured':>9s}  {'default':>9s}")
    for i in range(6):
        print(f"  {SIM_NAMES[i]:<8s}  {bridge.sim_joint_pos[i]:+9.3f}  "
              f"{DEFAULT_JOINT_POS[i]:+9.3f}")
    print(f"\n{DIM}Motors are LIMP (Kp=Kd=0). Ramp gains with ]/[ and '/; "
          f"before nudging Δ.{RESET}")
    print(f"{DIM}Right side mirrors left: "
          f"R = default_R + ({'/'.join(str(int(RIGHT_SIGN[g])) for g in GROUP_ORDER)})·Δ "
          f"per pair.{RESET}")
    time.sleep(1.0)

    # ── State ────────────────────────────────────────────────
    kp = {g: 0.0 for g in GROUP_ORDER}
    kd = {g: 0.0 for g in GROUP_ORDER}
    active_group = 'hip_roll'
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
                g = active_group
                step = POS_STEPS[pos_step_idx]
                lim = DELTA_MAX[g]

                if k in GROUP_KEYS:
                    active_group = GROUP_KEYS[k]

                elif k in ('+', '='):
                    d[g] = min(lim, d[g] + step)
                elif k in ('-', '_'):
                    d[g] = max(-lim, d[g] - step)

                elif k == ']':
                    kp[g] = min(KP_MAX, kp[g] + KP_STEP)
                elif k == '[':
                    kp[g] = max(0.0,    kp[g] - KP_STEP)
                elif k == "'":
                    kd[g] = min(KD_MAX, kd[g] + KD_STEP)
                elif k == ';':
                    kd[g] = max(0.0,    kd[g] - KD_STEP)

                elif k == '0':
                    d[g] = 0.0
                elif k == 'z':
                    d = {gg: 0.0 for gg in GROUP_ORDER}
                elif k == 'v':
                    d = fit_delta_from_pose(bridge.sim_joint_pos)

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
                        d = fit_delta_from_pose(bridge.sim_joint_pos)
                elif k == 'd':
                    if enabled:
                        bridge.disable()
                        enabled = False

                elif k in ('q', '\x03'):
                    break

            # ── refresh feedback + rebuild target ───────────
            bridge.read_state()
            target = build_target(d)
            kp_vec, kd_vec = expand_gains(kp, kd)

            # ── drive motors (or idle if disabled) ──────────
            if enabled:
                bridge.send_target(target.astype(np.float32), kp_vec, kd_vec)
            else:
                bridge.send_idle()

            # ── refresh display ─────────────────────────────
            now = time.monotonic()
            if redraw_now or (now - last_print >= 1.0 / PRINT_HZ):
                lines = build_display(
                    kp=kp, kd=kd, d=d,
                    active_group=active_group,
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

        print("\n\n  Final gains / deltas:")
        for g in GROUP_ORDER:
            print(f"    {g:<10s}  Kp = {kp[g]:7.2f}  Kd = {kd[g]:6.3f}  "
                  f"Δ = {d[g]:+6.3f}")
        print("  Final targets (sim rad):")
        tgt = build_target(d)
        for i in range(6):
            print(f"    {SIM_NAMES[i]:<8s}  {tgt[i]:+7.3f}")
        print("  Motors disabled. Done.")


if __name__ == '__main__':
    main()
