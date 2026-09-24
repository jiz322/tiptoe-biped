"""
balance.py — phase-1 balance script for the bipedal robot.

THREE-MODE VERSION with explicit handover gate:

    HOLD          NN idle, motors held at spawn pose by PD.
    OBSERVE       NN forward pass runs every tick (you can see its
                  outputs), but motors are STILL held at spawn.
    NN CONTROL    NN drives the motors. Only reachable FROM OBSERVE
                  (you must look before you leap).

Procedure
---------
0. Hang the robot. Make sure all motors are properly zeroed.
1. sudo python3 balance.py
2. Robot starts in HOLD with all gains at 0 (motors limp).
3. Ramp Kp/Kd per group with the keyboard until the robot is firmly
   held in the spawn pose:
       1 / 2 / 3   select group (hip_roll / hip_pitch / knee)
       + / -       Kp ± KP_STEP for active group
       ] / [       Kd ± KD_STEP for active group
       r           reset all gains to 0
4. Press 'g' to enter OBSERVE. NN forward pass starts; motors still
   held at spawn. Confirm the action values look reasonable.
5. Press 'n' to enter NN CONTROL. The policy now drives the motors.
   (Only works from OBSERVE — pressing 'n' from HOLD is ignored.)
6. Press 'h' at any time to slam back to HOLD.
7. q or Ctrl-C  quit. Motors disable cleanly before exit.
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
from policy import Policy
import config as C

# ════════════════════════════════════════════════════════════
# Tuning parameters
# ════════════════════════════════════════════════════════════
POLICY_PATH = "jerk30_std018.pt"
KP_STEP = 0.05
KD_STEP = 0.05
KP_MAX  = 50.0
KD_MAX  = 50.0
TAU_STEP = 1.0
TAU_MAX  = 17.0
TAU_MIN  = 1.0     # never let it collapse to 0 (would zero-span the decode)

# Per-group motor-frame torque encoding range. Must match motor model.
#   EL05 → 6.0,  RS01 → 17.0
DEFAULT_TAU_MAX = {
    'hip_roll':   6.0,   # EL05
    'hip_pitch': 17.0,   # RS01
    'knee':      17.0,   # RS01
}
CONTROL_HZ = 100   # matches sim decimation × dt = 5/500 = 100 Hz
PRINT_HZ   = 10

# ════════════════════════════════════════════════════════════
# Sim-side constants — must match rl_env_cfg_p6e.py
# ════════════════════════════════════════════════════════════
JOINT_SIGN_ARR = np.array(C.JOINT_SIGN, dtype=np.float32)
KNEE_GEAR      = C.KNEE_GEAR_RATIO
KNEE_IDX       = (2, 5)   # KN1, KN2 in the 6-vector

SPAWN_JOINT_POS = np.array(
    [0.0, 0.05, 0.0,  0.0, -0.05, 0.0], dtype=np.float32
)
DEFAULT_JOINT_POS = SPAWN_JOINT_POS.copy()

ACTION_SCALES = np.array(
    [0.3, 0.75, 2.05, 0.3, 0.75, 2.05], dtype=np.float32
)
OBS_PER_FRAME     = 36
OBS_HISTORY_STEPS = 5     # must match cfg.obs_history_steps in training
GAIT_FREQ_HZ = 1.5
SPAWN_HEIGHT = 0.42

# ════════════════════════════════════════════════════════════
# Modes
# ════════════════════════════════════════════════════════════

MODE_HOLD       = 'HOLD'
MODE_OBSERVE    = 'OBSERVE'
MODE_NN_CONTROL = 'NN_CONTROL'

# ════════════════════════════════════════════════════════════
# Display
# ════════════════════════════════════════════════════════════

SIM_NAMES   = ['L_roll', 'L_hip', 'L_knee', 'R_roll', 'R_hip', 'R_knee']
SIM_GROUPS  = ['hip_roll', 'hip_pitch', 'knee',
               'hip_roll', 'hip_pitch', 'knee']
GROUP_ORDER = ['hip_roll', 'hip_pitch', 'knee']
GROUP_KEYS  = {'1': 'hip_roll', '2': 'hip_pitch', '3': 'knee'}

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

def apply_tau_max(bridge, tau_max):
    """Push per-group decode range into the driver's _ranges table.
    Re-applies on every change so live keyboard tuning takes effect
    on the very next reply parse. Symmetric: tff_min = -tff_max.
    """
    for i, mid in enumerate(C.MOTOR_IDS):
        g = SIM_GROUPS[i]
        bridge.driver._ranges[mid]['tff_max'] =  tau_max[g]
        bridge.driver._ranges[mid]['tff_min'] = -tau_max[g]
# ════════════════════════════════════════════════════════════
# Observation builder — must match rl_env._get_observations
# ════════════════════════════════════════════════════════════

def build_obs(proj_grav, ang_vel, lin_vel,
              joint_pos, joint_vel, last_actions,
              gait_phase):
    """Pack the 36D observation in sim layout, with all command/curriculum
    slots hard-zeroed for the balance-only case (alpha=0, no grace,
    cmd=0, yaw_err≈0).
    """
    joint_pos_norm = joint_pos - DEFAULT_JOINT_POS

    obs = np.concatenate([
        proj_grav.astype(np.float32),                  # 3
        ang_vel.astype(np.float32),                    # 3
        np.zeros(3, dtype=np.float32),                 # 3  lin_vel: dropout=1.0 in training
        joint_pos_norm.astype(np.float32),             # 6
        joint_vel.astype(np.float32),                  # 6
        last_actions.astype(np.float32),               # 6
        np.array([math.sin(gait_phase),
                  math.cos(gait_phase)],
                 dtype=np.float32),                    # 2
        np.array([0.0,            # difficulty (alpha=0)
                  0.0,            # grace_norm
                  SPAWN_HEIGHT,   # init_h
                  0.0,            # vx_cmd
                  0.0,            # yaw_rate_cmd
                  0.0,            # yaw_err_sin = sin(0)
                  1.0],           # yaw_err_cos = cos(0)
                 dtype=np.float32),                    # 7
    ]).astype(np.float32)
    assert obs.shape == (36,), f"obs shape mismatch: {obs.shape}"
    return obs

# ════════════════════════════════════════════════════════════
# Obs snapshot — labeled 36D dump for sanity-checking NN inputs
# ════════════════════════════════════════════════════════════
#
# Layout matches build_obs() above, which mirrors rl_env._get_observations:
#   [ 0: 3] projected_gravity
#   [ 3: 6] ang_vel_body
#   [ 6: 9] lin_vel_body          (zeroed on hardware)
#   [ 9:15] joint_pos_norm        (sim order)
#   [15:21] joint_vel             (sim order)
#   [21:27] last_actions          (sim order)
#   [27:29] gait sin / cos
#   [29:36] difficulty, grace_norm, init_h, vx_cmd, yaw_rate_cmd,
#           yaw_err_sin, yaw_err_cos

def format_obs_dump(obs):
    lines = []
    lines.append(
        f"{BOLD}=== NN observation snapshot (36D) ==={RESET}  "
        f"{DIM}{time.strftime('%H:%M:%S')}{RESET}"
    )

    # 3-vectors, one per line
    lines.append(
        f"  [ 0: 3] {BOLD}projected_gravity{RESET}  "
        f"({obs[0]:+.4f}, {obs[1]:+.4f}, {obs[2]:+.4f})"
    )
    lines.append(
        f"  [ 3: 6] {BOLD}ang_vel_body     {RESET}  "
        f"({obs[3]:+.4f}, {obs[4]:+.4f}, {obs[5]:+.4f})  {DIM}rad/s{RESET}"
    )
    lines.append(
        f"  [ 6: 9] {BOLD}lin_vel_body     {RESET}  "
        f"({obs[6]:+.4f}, {obs[7]:+.4f}, {obs[8]:+.4f})  "
        f"{DIM}m/s (zeroed on hw){RESET}"
    )

    # Per-joint table — pos_norm, vel, last_action side-by-side
    lines.append("")
    lines.append(
        f"  {BOLD}{'joint':<8s}"
        f"  [ 9:15] {'pos_norm':>10s}"
        f"   [15:21] {'vel':>10s}"
        f"   [21:27] {'last_action':>10s}{RESET}"
    )
    for i in range(6):
        lines.append(
            f"  {SIM_NAMES[i]:<8s}"
            f"          {obs[ 9 + i]:+9.4f}"
            f"           {obs[15 + i]:+9.4f}"
            f"            {obs[21 + i]:+9.4f}"
        )

    # Gait
    lines.append("")
    lines.append(
        f"  [27:29] {BOLD}gait            {RESET}  "
        f"sin = {obs[27]:+.4f}   cos = {obs[28]:+.4f}"
    )

    # Cmd / curriculum tail
    cmd_labels = [
        'difficulty', 'grace_norm', 'init_h',
        'vx_cmd', 'yaw_rate_cmd', 'yaw_err_sin', 'yaw_err_cos',
    ]
    lines.append("")
    lines.append(f"  {BOLD}[29:36] cmd / curriculum{RESET}")
    for j, lbl in enumerate(cmd_labels):
        lines.append(f"    [{29 + j:2d}] {lbl:<14s}  {obs[29 + j]:+9.4f}")

    return "\n".join(lines)

# ════════════════════════════════════════════════════════════
# Display — fixed layout so ESC_UP works cleanly
# ════════════════════════════════════════════════════════════

def build_display(*, mode, kp, kd, tau_max, active_group, target,
                    sim_pos, sim_vel, tau_cmd, tau_meas, temp_c,
                    proj_grav, ang_vel,
                    last_action, gait_phase, lpf_alpha):
    lines = []

    # Mode banner — explicit about who is driving the motors.
    if mode == MODE_NN_CONTROL:
        banner = (f"{RED}{BOLD}NN CONTROL{RESET}  "
                  f"{DIM}(motors driven by policy — press 'h' to release){RESET}")
    elif mode == MODE_OBSERVE:
        banner = (f"{CYAN}{BOLD}OBSERVE{RESET}  "
                  f"{DIM}(NN running, motors held at spawn){RESET}")
    else:
        banner = (f"{YELLOW}{BOLD}HOLD{RESET}  "
                  f"{DIM}(motors held at spawn, NN idle){RESET}")
    lines.append(f"  Mode: {banner}")
    lines.append(
        f"  {DIM}1/2/3 group   "
        f"+/- Kp ±{KP_STEP}   ]/[ Kd ±{KD_STEP}   "
        f",/. τ_max ±{TAU_STEP}   "
        f"r reset gains{RESET}"
    )
    lines.append("")
    lines.append(f"  {DIM}LPF α = {lpf_alpha:.2f}{RESET}")
    lines.append(
        f"  {DIM}g observe   n NN control   h hold   "
        f"f LPF alpha   p print obs   q quit{RESET}"
    )
    lines.append("")
    
    lines.append(
        f"  {BOLD}    {'group':<10s}     Kp        Kd      τ_max{RESET}  "
        f"{DIM}(motor Nm){RESET}"
    )
    for g in GROUP_ORDER:
        is_active = (g == active_group)
        marker = f"{YELLOW}>{RESET}" if is_active else " "
        gname  = f"{BOLD}{g:<10s}{RESET}" if is_active else f"{g:<10s}"
        lines.append(
            f"  {marker}   {gname}   {kp[g]:7.2f}   {kd[g]:6.3f}   {tau_max[g]:6.1f}"
        )

    lines.append("")
    lines.append(
        f"  {BOLD}{'joint':<8s}  {'target':>7s}  {'pos':>7s}  "
        f"{'err':>7s}  {'vel':>7s}  {'τ_cmd':>7s}  {'τ_meas':>7s}  "
        f"{'temp':>6s}{RESET}  "
        f"{DIM}(sim rad | motor Nm | °C){RESET}"
    )
    for i in range(6):
        err = target[i] - sim_pos[i]
        t = temp_c[i]
        if   t >= 80.0: temp_str = f"{RED}{t:6.1f}{RESET}"
        elif t >= 60.0: temp_str = f"{YELLOW}{t:6.1f}{RESET}"
        else:           temp_str = f"{t:6.1f}"

        # Highlight tau_meas in red when it's within 5% of the encoding ceiling
        # — that's the saturation tell.
        ceiling = tau_max[SIM_GROUPS[i]]
        if abs(tau_meas[i]) >= 0.95 * ceiling:
            meas_str = f"{RED}{tau_meas[i]:+7.2f}{RESET}"
        else:
            meas_str = f"{tau_meas[i]:+7.2f}"

        lines.append(
            f"  {SIM_NAMES[i]:<8s}  "
            f"{target[i]:+7.3f}  {sim_pos[i]:+7.3f}  "
            f"{err:+7.3f}  {sim_vel[i]:+7.3f}  "
            f"{tau_cmd[i]:+7.2f}  {meas_str}  "
            f"{temp_str}"
        )

    lines.append("")
    lines.append(
        f"  {BOLD}IMU{RESET}  "
        f"proj_grav: ({proj_grav[0]:+.3f},{proj_grav[1]:+.3f},{proj_grav[2]:+.3f})  "
        f"ang_vel: ({ang_vel[0]:+.3f},{ang_vel[1]:+.3f},{ang_vel[2]:+.3f}) rad/s"
    )

    lines.append("")
    if mode == MODE_NN_CONTROL:
        nn_label = f"{RED}{BOLD}NN  raw action      -> applied offset (rad){RESET}"
        applied_tag = f"{RED}[APPLIED to motors]{RESET}"
    elif mode == MODE_OBSERVE:
        nn_label = f"{CYAN}{BOLD}NN  raw action      would-be offset (rad){RESET}"
        applied_tag = f"{DIM}[NOT applied]{RESET}"
    else:
        nn_label = None
        applied_tag = None

    if mode in (MODE_OBSERVE, MODE_NN_CONTROL):
        lines.append(
            f"  {nn_label}   "
            f"{DIM}gait_phase={gait_phase:+.2f}  cmd=(0,0){RESET}  {applied_tag}"
        )
        for i in range(6):
            offset = last_action[i] * ACTION_SCALES[i]
            lines.append(
                f"  {SIM_NAMES[i]:<8s}  {last_action[i]:+8.4f}    "
                f"->  {offset:+8.4f}"
            )
    else:
        lines.append(
            f"  {BOLD}NN{RESET}  "
            f"{DIM}idle — press 'g' to start observe-only NN forward pass{RESET}"
        )
        for _ in range(6):
            lines.append(f"  {DIM}    --{RESET}")
    
    return lines


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main():
    print(f"{BOLD}=== balance.py — hold / observe / NN control ==={RESET}\n")
    print("Initializing motors, IMU, policy ...")

    bridge = MotorBridge()
    imu    = IMUAdapter()
    imu.open()
    policy = Policy(POLICY_PATH)

    for _ in range(10):
        imu.update()
        time.sleep(0.005)

    bridge.enable()
    time.sleep(0.3)

    for _ in range(3):
        bridge.send_idle()
        time.sleep(0.02)
    bridge.read_state()

    print()
    print(f"{BOLD}Initial sim joint positions{RESET} "
      f"{DIM}(should be near spawn; motors were zeroed at "
      f"{C.ZERO_POSE_NAME}){RESET}")
    print(f"  {'joint':<8s}  {'measured':>9s}  {'spawn':>9s}")
    for i in range(6):
        print(f"  {SIM_NAMES[i]:<8s}  {bridge.sim_joint_pos[i]:+9.3f}  "
              f"{SPAWN_JOINT_POS[i]:+9.3f}")
    print()
    print(f"{DIM}Motors are still limp (Kp=Kd=0). "
          f"Use +/- to ramp Kp once you're ready.{RESET}")
    print(f"{DIM}Workflow: HOLD --(g)--> OBSERVE --(n)--> NN CONTROL.  "
          f"'h' returns to HOLD from any mode.{RESET}")
    time.sleep(1.0)

    # ── State ────────────────────────────────────────────────
    kp = {g: 0.0 for g in GROUP_ORDER}
    kd = {g: 0.0 for g in GROUP_ORDER}
    tau_max = dict(DEFAULT_TAU_MAX)
    apply_tau_max(bridge, tau_max)

    active_group = 'hip_roll'

    mode         = MODE_HOLD
    last_actions = np.zeros(6, dtype=np.float32)
    obs_history = np.zeros(
        (OBS_HISTORY_STEPS, OBS_PER_FRAME), dtype=np.float32
    )
    gait_phase   = 0.0

    fd, old_termios = setup_raw_keys()
    period       = 1.0 / CONTROL_HZ
    last_print   = 0.0
    first_print  = True
    line_count   = 0

    try:
        while True:
            t_loop = time.monotonic()
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
                elif k == 'g':
                    if mode == MODE_HOLD:
                        mode         = MODE_OBSERVE
                        gait_phase   = 0.0
                        last_actions = np.zeros(6, dtype=np.float32)
                        obs_history[:] = 0.0          # NEW — match env _reset_idx
                elif k == 'n':
                    # OBSERVE -> NN CONTROL.  Gated: only valid from
                    # OBSERVE so you have to see the NN output first.
                    # Pressed from HOLD: ignored (no warning, just a no-op
                    # so a stray keypress doesn't do anything surprising).
                    if mode == MODE_OBSERVE:
                        mode = MODE_NN_CONTROL
                elif k == 'h':
                        mode         = MODE_HOLD
                        last_actions = np.zeros(6, dtype=np.float32)
                        gait_phase   = 0.0
                        obs_history[:] = 0.0              # NEW
                elif k == 'p':
                    # Snapshot the current 36D NN observation. Builds a
                    # fresh obs from live sensors so it works in HOLD too,
                    # before you've handed over to the policy. Writes
                    # below the live display and re-anchors the redraw.
                    snap_obs = build_obs(
                        proj_grav    = imu.projected_gravity(),
                        ang_vel      = imu.ang_vel_body(),
                        lin_vel      = imu.lin_vel_body(),
                        joint_pos    = bridge.sim_joint_pos,
                        joint_vel    = bridge.sim_joint_vel,
                        last_actions = last_actions,
                        gait_phase   = gait_phase,
                    )
                    sys.stdout.write("\n\n")
                    sys.stdout.write(format_obs_dump(snap_obs))
                    sys.stdout.write("\n\n")
                    sys.stdout.flush()
                    # Force the next redraw to lay down fresh (no ESC_UP)
                    # so the dump stays in the scrollback above it.
                    first_print = True

                elif k == 'f':
                    alphas = [0.0, 0.5, 0.6, 0.65, 0.7]
                    cur = bridge._lpf_alpha
                    idx = min(range(len(alphas)), key=lambda i: abs(alphas[i] - cur))
                    next_alpha = alphas[(idx + 1) % len(alphas)]
                    bridge.set_lpf_alpha(next_alpha)
                    bridge.reset_filter()
                    # redraw_now=True is already set above; live display will reflect it
                elif k == '.':
                    tau_max[active_group] = min(TAU_MAX, tau_max[active_group] + TAU_STEP)
                    apply_tau_max(bridge, tau_max)
                elif k == ',':
                    tau_max[active_group] = max(TAU_MIN, tau_max[active_group] - TAU_STEP)
                    apply_tau_max(bridge, tau_max)
                elif k in ('q', '\x03'):
                    break

            # ── refresh sensors ─────────────────────────────
            imu.update()
            bridge.read_state()

            kp_vec, kd_vec = expand_gains(kp, kd)

            # ── run NN forward pass when active ─────────────
            # In OBSERVE we run it for display only; in NN_CONTROL
            # the same output also drives the target.  In HOLD we
            # don't bother running it.
            nn_running = (mode in (MODE_OBSERVE, MODE_NN_CONTROL))
            if nn_running:
                obs_frame = build_obs(
                    proj_grav    = imu.projected_gravity(),
                    ang_vel      = imu.ang_vel_body(),
                    lin_vel      = imu.lin_vel_body(),
                    joint_pos    = bridge.sim_joint_pos,
                    joint_vel    = bridge.sim_joint_vel,
                    last_actions = last_actions,
                    gait_phase   = gait_phase,
                )
                    # Match env _get_observations: roll dim, write newest to slot 0,
                # flatten to 180. Frame 0 = newest, K-1 = oldest.
                obs_history = np.roll(obs_history, shift=1, axis=0)
                obs_history[0] = obs_frame
                obs_stacked = obs_history.reshape(-1)        # (180,)
                action       = policy.forward(obs_stacked)
                last_actions = action.copy()

                gait_phase = (
                    gait_phase
                    + 2.0 * math.pi * GAIT_FREQ_HZ / CONTROL_HZ
                ) % (2.0 * math.pi)

            # ── pick target — the only place mode steers motors ──
            if mode == MODE_NN_CONTROL:
                target = DEFAULT_JOINT_POS + last_actions * ACTION_SCALES
            else:
                # HOLD and OBSERVE both pin motors to spawn.
                target = SPAWN_JOINT_POS.copy()

            bridge.send_target(target.astype(np.float32), kp_vec, kd_vec)

            # Sim-frame PD output (what the policy/PD asked for, in joint space).
            sim_tau_cmd = (kp_vec * (target - bridge.sim_joint_pos)
                        - kd_vec * bridge.sim_joint_vel)
            # Inverse of _motor_to_sim_torque: project back into motor frame so
            # tau_cmd, tau_meas, tau_max are all in the same frame.
            motor_tau_cmd = (sim_tau_cmd * JOINT_SIGN_ARR).astype(np.float32)
            motor_tau_cmd[KNEE_IDX[0]] *= KNEE_GEAR
            motor_tau_cmd[KNEE_IDX[1]] *= KNEE_GEAR
            # Commanded torque in sim frame — what the PD law asked for.
            # v_des = 0 in send_target, so the damping term is -Kd * vel.
            tau_cmd = (kp_vec * (target - bridge.sim_joint_pos)
                    - kd_vec *           bridge.sim_joint_vel).astype(np.float32)
            # ── refresh display ─────────────────────────────
            now = time.monotonic()
            if redraw_now or (now - last_print >= 1.0 / PRINT_HZ):
                lines = build_display(
                    mode         = mode,
                    kp = kp, kd = kd, tau_max = tau_max,            # NEW
                    active_group = active_group,
                    target       = target,
                    sim_pos      = bridge.sim_joint_pos,
                    sim_vel      = bridge.sim_joint_vel,
                    tau_cmd      = motor_tau_cmd,                   # NEW (motor frame)
                    tau_meas     = bridge.motor_torque,             # NEW (motor frame)
                    temp_c       = bridge.motor_temperature,        # NEW
                    proj_grav    = imu.projected_gravity(),
                    ang_vel      = imu.ang_vel_body(),
                    last_action  = last_actions,
                    gait_phase   = gait_phase,
                    lpf_alpha    = bridge._lpf_alpha,
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
        try:
            imu.close()
        except Exception:
            pass

        print("\n\n  Final gains:")
        for g in GROUP_ORDER:
            print(f"    {g:<10s}  Kp = {kp[g]:7.2f}    Kd = {kd[g]:6.3f}")
        print("  Motors disabled. Done.")


if __name__ == '__main__':
    main()