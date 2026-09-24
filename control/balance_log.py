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
6. Press 'b' for a one-shot HOLD -> NN CONTROL + start logging.
   Logging runs until 'h' is pressed (or quit), then the buffer is
   saved to <checkpoint_basename>_log_<timestamp>.npz in CWD.
7. Press 'h' at any time to slam back to HOLD (saves log if active).
8. q or Ctrl-C  quit. Motors disable cleanly before exit.

Camera sync: the LED on GPIO 18 turns ON at 'b'/'v'/'g' (for b/v the edge
is placed right at the log's t=0) and OFF at 'h'/quit, so video can be
aligned to the npz. If the GPIO is unavailable it warns and continues.

S4 balance test (2026-09-10, checkpoints s4a_6520.pt / s4b_6520.pt)
--------------------------------------------------------------------
The S-line obs layout is UNCHANGED (36 x 5 = 180; verified on the actor
weights) — the curriculum added no channels, it only made channels 29/30/31
(difficulty, grace_norm, init_h) vary across training episodes. The balance
test feeds the stand-branch values (0, 0, SPAWN_HEIGHT), which is what 40 %
of S4 episodes saw; policy_contract_check.py (desk) and the startup guard
below (Pi) z-score them against the loaded checkpoint's normalizer and
refuse to fly on an OOD constant. Handover: S4 trained with
handover_prime_prob=0 -> blind only -> use 'v' (not 'b').

The one contract knob that differs BETWEEN the arms is the IMU pitch trim
(config.IMU_PG_OFFSET, +0.089 since 09-03): S4a trained under a +-0.13
pg-bias band and is compatible with it; S4b trained under +-0.05 and is
OOD under +0.089 by design (its launch header: raw pg_x at physical upright
must be within +-0.05 -> trim ~0). Set the trim per trial with
    sudo ~/miniforge3/bin/python3 balance_log.py --policy s4a_6520.pt --pg-offset 0.089
    sudo ~/miniforge3/bin/python3 balance_log.py --policy s4b_6520.pt --pg-offset 0.0
instead of editing config.py; the value in effect is shown in the display
and saved in the npz (imu_pg_offset), and proj_grav_raw is logged per tick
so the trim never has to be recovered from a README again (checklist #8).
"""

import math
import os
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
import obs_contract as OC

# ════════════════════════════════════════════════════════════
# Tuning parameters
# ════════════════════════════════════════════════════════════
POLICY_PATH = "r1r_3150.pt"   # R1r_3150: the policy that stood on 2026-09-03 (the only shipped checkpoint)
# P12 E4 policy A/B (runbook 08-25): switch per trial WITHOUT editing this
# file — `sudo ... balance_log.py --policy policy_L1_4330.pt`. Log filenames
# derive from the policy path, so trials self-label (policy_L1_4330_log_*).
if '--policy' in sys.argv:
    POLICY_PATH = sys.argv[sys.argv.index('--policy') + 1]
# IMU pitch trim for THIS run (x component of config.IMU_PG_OFFSET; y kept).
# None = config value. S4 A/B: arm A flies at the config +0.089, arm B at ~0.
PG_OFFSET_X = None
if '--pg-offset' in sys.argv:
    PG_OFFSET_X = float(sys.argv[sys.argv.index('--pg-offset') + 1])
# Startup obs-contract guard: refuse to run if a balance-test constant is OOD
# for the loaded checkpoint's normalizer. --force-contract downgrades to a warning.
FORCE_CONTRACT = '--force-contract' in sys.argv
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

# ── P11 K-line deploy-gain preset (runbook §0.3) ─────────────────
# 'k' slews the live gains onto EXACTLY these joint-space values
# (config.REAL_KP/REAL_KD — the gains the flying policy trained under).
# Use it AFTER the robot is already firmly held by a manual coarse ramp:
# the slew converges on the contract, it is not the gentle initial pull.
PRESET_KP = {'hip_roll': 6.0, 'hip_pitch': 20.0, 'knee': 30.0}
PRESET_KD = {'hip_roll': 0.3, 'hip_pitch': 1.4,  'knee': 1.5}
KP_SLEW_PER_TICK = 0.10   # 10 kp/s at 100 Hz — full preset in ~3 s
KD_SLEW_PER_TICK = 0.01   # 1 kd/s

# ════════════════════════════════════════════════════════════
# Sim-side constants — imported from config.py (single source of truth)
# ════════════════════════════════════════════════════════════
JOINT_SIGN_ARR = np.array(C.JOINT_SIGN, dtype=np.float32)
KNEE_GEAR      = C.KNEE_GEAR_RATIO
KNEE_IDX       = (2, 5)   # KN1, KN2 in the 6-vector

SPAWN_JOINT_POS   = np.array(C.DEFAULT_JOINT_POS, dtype=np.float32)
DEFAULT_JOINT_POS = SPAWN_JOINT_POS.copy()

ACTION_SCALES     = np.array(C.ACTION_SCALES, dtype=np.float32)
OBS_PER_FRAME     = C.OBS_FRAME_DIM
OBS_HISTORY_STEPS = C.OBS_HISTORY_STEPS
GAIT_FREQ_HZ     = C.GAIT_FREQ_HZ
SPAWN_HEIGHT      = C.SPAWN_HEIGHT

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


class TrialLED:
    """Camera-sync indicator LED on a GPIO pin.

    ON marks trial start ('b'/'v'/'g'), OFF marks return to HOLD ('h') or
    exit — gives the video a visible window aligned with the log. on()/off()
    are a single register write via gpiozero (microseconds); the slow part
    (pin-factory init) happens once here in __init__, so keypress latency is
    negligible vs the 10 ms tick and ~33 ms video frame.

    Degrades to a no-op with a warning if GPIO is unavailable (gpiozero
    missing, pin busy, not a Pi). Note: an unwired LED is undetectable from
    software — this only catches GPIO-level failures.
    """

    def __init__(self, pin=18):
        self._led = None
        try:
            from gpiozero import LED
            self._led = LED(pin)
            self._led.off()
        except Exception as e:
            print(f"{YELLOW}[WARN] trial LED unavailable "
                  f"(GPIO {pin}): {e} — continuing without it{RESET}")

    def on(self):
        if self._led is not None:
            try:
                self._led.on()
            except Exception as e:
                print(f"{YELLOW}[WARN] trial LED on() failed: {e}{RESET}")

    def off(self):
        if self._led is not None:
            try:
                self._led.off()
            except Exception as e:
                print(f"{YELLOW}[WARN] trial LED off() failed: {e}{RESET}")

    def close(self):
        if self._led is not None:
            try:
                self._led.off()
                self._led.close()
            except Exception:
                pass
            self._led = None


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
                  0.0,            # vx_cmd  (MUST be 0 for balance test)
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
# Run logger — records every NN tick from 'b' until 'h'
# ════════════════════════════════════════════════════════════
#
# Schema: dict of lists, one list per channel, one entry per control tick.
# On save, each list is stacked into a numpy array and dumped into a single
# .npz next to balance.py. Filename is derived from POLICY_PATH so multiple
# checkpoints can coexist without clobbering each other.

LOG_CHANNELS = [
    'time',            # seconds since logging started
    'obs',             # (36,) raw NN-frame observation
    'action',          # (6,)  raw policy output (pre-scale)
    'target',          # (6,)  joint targets sent to PD (sim frame, rad)
    'sim_joint_pos',   # (6,)  measured joint pos (sim frame, rad)
    'sim_joint_vel',   # (6,)  measured joint vel (sim frame, rad/s)
    'motor_tau_cmd',   # (6,)  PD-commanded torque, projected to motor frame
    'motor_tau_meas',  # (6,)  motor-reported torque (motor frame, Nm)
    'motor_temp',      # (6,)  motor temperatures (°C)
    'proj_grav',       # (3,)  body-frame projected gravity (trimmed = what the NN saw)
    'proj_grav_raw',   # (3,)  untrimmed (proj_grav + imu_pg_offset); checklist #8
    'ang_vel',         # (3,)  body-frame angular velocity (rad/s)
    'gait_phase',      # scalar gait phase (rad)
    'mode',            # str   mode at this tick
    'can_replies',     # int   CAN reply frames parsed this tick (review #7)
    # Run configuration per tick — makes the log self-contained for offline
    # replay (gains are hand-ramped and can change mid-run via keys; without
    # these the 20260803 run's gains had to be recovered from the tau_cmd
    # recursion after the fact).
    'kp',              # (6,)  sim-space Kp in effect (expanded per joint)
    'kd',              # (6,)  sim-space Kd in effect
    'tau_max',         # (6,)  motor-frame torque decode range per joint
    'lpf_alpha',       # scalar bridge LPF alpha
    'filtered_target', # (6,)  post-LPF post-effort-clip target (what motors got)
    'tick_busy_ms',    # scalar compute+CAN time this tick, before the sleep
]


def new_log_buffer():
    return {ch: [] for ch in LOG_CHANNELS}


def save_log(log, policy_path, handover='sighted', meta=None):
    """Save log buffer to <checkpoint_basename>_log_<timestamp>.npz in CWD.

    meta: extra run-level arrays (imu_pg_offset, policy_iter, obs normalizer
    stats) stored alongside; keys are additive to the historical schema.
    Returns (filename, num_ticks).
    """
    # 'tick_busy_ms' is appended later in the loop than the other channels,
    # so a Ctrl-C mid-tick can leave it one entry short. Truncate everything
    # to the shortest channel so the arrays stay tick-aligned.
    n = min(len(vals) for vals in log.values())
    base  = os.path.splitext(os.path.basename(policy_path))[0]
    stamp = time.strftime('%Y%m%d_%H%M%S')
    fname = f"{base}_log_{stamp}.npz"

    arrays = {}
    for ch, vals in log.items():
        # np.array handles ragged-free lists; 'mode' becomes a string array.
        arrays[ch] = np.asarray(vals[:n])

    np.savez(
        fname,
        policy=np.asarray(os.path.basename(policy_path)),
        control_hz=np.asarray(CONTROL_HZ),
        handover=np.asarray(handover),   # 'sighted' ('b') vs 'blind' ('v')
        **(meta or {}),
        **arrays,
    )
    return fname, n


def run_meta(policy, imu):
    """Run-level provenance stored in every npz (see save_log)."""
    meta = {
        'imu_pg_offset': np.asarray(imu.pg_offset[:2], dtype=np.float32),
        'policy_iter': np.asarray(-1 if policy.iter is None else policy.iter),
    }
    if policy.norm_mean is not None:
        meta['obs_norm_mean'] = np.asarray(policy.norm_mean, dtype=np.float32)
        meta['obs_norm_std'] = np.asarray(policy.norm_std, dtype=np.float32)
    return meta


def contract_guard(policy, imu):
    """Startup obs-contract check (checklist #1): z-score the balance-test
    constants against the loaded checkpoint's normalizer. Exits on OOD unless
    --force-contract. Also prints the trained pg band so the live trimmed
    IMU reading at upright can be compared against it by eye."""
    print()
    if policy.norm_mean is None:
        print(f"{YELLOW}[CONTRACT] no normalizer stats in {POLICY_PATH} — "
              f"cannot score the obs tail (JIT without buffers?){RESET}")
        return
    rows = OC.tail_report(policy.norm_mean, policy.norm_std)
    print(OC.format_report(rows, os.path.basename(POLICY_PATH)))
    (mx, sx), (my, sy) = OC.pg_band(policy.norm_mean, policy.norm_std)
    print(f"  trained pg band: pg_x {mx:+.3f} +- {sx:.3f}  pg_y {my:+.3f} +- {sy:.3f}"
          f"   |  trim in effect: IMU_PG_OFFSET = ({imu.pg_offset[0]:+.3f}, "
          f"{imu.pg_offset[1]:+.3f})")
    bad = OC.ood_rows(rows)
    if bad:
        names = ', '.join(f"{lbl}[{i}] z={z:+.1f}" for i, lbl, *_r, z in bad)
        if FORCE_CONTRACT:
            print(f"{YELLOW}[CONTRACT] OOD constants ({names}) — continuing "
                  f"because --force-contract{RESET}")
        else:
            print(f"{RED}[CONTRACT] OOD balance-test constant(s): {names}. "
                  f"This checkpoint was not trained on the frame build_obs "
                  f"emits — refusing to run (override: --force-contract).{RESET}")
            raise SystemExit(2)
    else:
        print(f"{GREEN}[CONTRACT] PASS — balance-test constants inside the "
              f"trained band{RESET}")


# ════════════════════════════════════════════════════════════
# Display — fixed layout so ESC_UP works cleanly
# ════════════════════════════════════════════════════════════

def build_display(*, mode, kp, kd, tau_max, active_group, target,
                    sim_pos, sim_vel, tau_cmd, tau_meas, temp_c,
                    proj_grav, ang_vel,
                    last_action, gait_phase, lpf_alpha, pg_offset,
                    log_active, log_len):
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

    if log_active:
        banner += f"   {GREEN}{BOLD}● LOG {log_len} ticks{RESET}"
    else:
        banner += f"   {DIM}○ log idle{RESET}"

    lines.append(f"  Mode: {banner}")
    lines.append(
        f"  {DIM}1/2/3 group   "
        f"+/- Kp ±{KP_STEP}   ]/[ Kd ±{KD_STEP}   "
        f",/. τ_max ±{TAU_STEP}   "
        f"r reset gains   k → K-contract gains{RESET}"
    )
    lines.append("")
    lines.append(f"  {DIM}LPF α = {lpf_alpha:.2f}   "
                 f"IMU trim pg_offset = ({pg_offset[0]:+.3f}, {pg_offset[1]:+.3f})"
                 f"{RESET}")
    lines.append(
        f"  {DIM}b run+log   g observe   n NN control   h hold/save   "
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
        f"{DIM}raw x {proj_grav[0] + pg_offset[0]:+.3f}{RESET}  "
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
            f"{DIM}idle — press 'g' to start observe-only NN forward pass, "
            f"or 'b' to run+log{RESET}"
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

    # P11 (runbook §0.3): action LPF STAYS 0.6 — it is part of the trained
    # plant, not an unmodeled lag. K env.yaml action_lpf_alpha=0.6, applied
    # in rl_env_p6.py _pre_physics_step (LPF before the delay buffer), and
    # the §0.2 replica gate ran with it. Bridge default == 0.6; the display
    # shows alpha — verify 0.60 before handover. Do NOT zero it: sim filters
    # actions, bridge filters the target; with a fixed action-center
    # (balance, grace=0) the two are algebraically identical.
    bridge = MotorBridge(action_lpf_alpha=0.6)
    imu    = IMUAdapter()
    if PG_OFFSET_X is not None:
        imu.set_pg_offset(PG_OFFSET_X)
    imu.open()
    policy = Policy(POLICY_PATH)
    # Obs-contract guard before any motor is enabled (S4: curriculum channels).
    contract_guard(policy, imu)
    log_meta = run_meta(policy, imu)
    # Camera-sync LED (GPIO 18): built once here so the b/v/g edge is fast.
    trial_led = TrialLED(pin=18)

    for _ in range(10):
        imu.update()
        time.sleep(0.005)

    bridge.enable()
    # CAN_TIMEOUT=200 ms firmware watchdog (P11 runbook §0.5): any silent
    # gap >200 ms after enable drops the motors into reset (limp), and only
    # a re-enable — not resumed streaming — recovers them. Keep the bus fed
    # with idle frames instead of sleeping silently.
    for _ in range(6):
        bridge.send_idle()
        time.sleep(0.05)
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
    print(f"{DIM}Shortcut:  'b' from HOLD jumps straight to NN CONTROL and "
          f"starts logging obs; 'h' stops + saves.{RESET}")
    # Feed idle frames through the 1 s settle (see CAN_TIMEOUT note above),
    # then re-enable as belt-and-braces in case anything upstream stalled
    # past 200 ms (enable on an already-enabled motor is a no-op).
    for _ in range(20):
        bridge.send_idle()
        time.sleep(0.05)
    bridge.enable()

    # ── State ────────────────────────────────────────────────
    kp = {g: 0.0 for g in GROUP_ORDER}
    kd = {g: 0.0 for g in GROUP_ORDER}
    tau_max = dict(DEFAULT_TAU_MAX)
    apply_tau_max(bridge, tau_max)

    active_group = 'hip_roll'
    preset_slew  = False   # 'k' arms; any manual gain key cancels

    mode         = MODE_HOLD
    last_actions = np.zeros(6, dtype=np.float32)
    obs_history = np.zeros(
        (OBS_HISTORY_STEPS, OBS_PER_FRAME), dtype=np.float32
    )
    needs_history_init = True   # replicate first real frame into all K slots
    gait_phase   = 0.0
    # Handover contract (p12.2 finding 08-29): 'b' = sighted (real first
    # frame x5 — the historical deploy behavior, which training never
    # produces); 'v' = blind (training contract: sensor block obs[0:27]
    # zeroed on the first frame, primed x5 — what every training episode
    # starts with, because rl_env_p6._reset_idx zeroes _obs_delay_buffer).
    handover_mode     = 'sighted'
    blind_first_frame = False   # one-shot: zero obs[0:27] on the next NN frame

    # Logging state. log_active gates per-tick capture; log_buffer is the
    # dict-of-lists schema from new_log_buffer(); log_start is the monotonic
    # reference for the 'time' channel. obs_frame_for_log holds the last
    # 36D obs we built this tick (only valid when nn_running).
    log_active   = False
    log_buffer   = None
    log_start    = 0.0

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
                if k in ('+', '=', '-', '_', ']', '[', 'r'):
                    preset_slew = False   # manual gain edit wins over 'k'
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
                elif k == 'k':
                    # Slew gains to the P11 K deploy contract (see PRESET_KP).
                    preset_slew = True
                elif k == 'l':
                    # Toggle logging in ANY mode (bench probes: PD-hold under
                    # stance load etc.). 'b' remains the NN+log one-shot.
                    if log_active and log_buffer is not None:
                        try:
                            fname, n = save_log(log_buffer, POLICY_PATH, handover_mode, log_meta)
                            sys.stdout.write(f"\n\n{GREEN}[LOG] Saved {n} ticks "
                                             f"to {fname}{RESET}\n\n")
                        except Exception as e:
                            sys.stdout.write(f"\n\n{RED}[LOG] save failed: "
                                             f"{e}{RESET}\n\n")
                        sys.stdout.flush()
                        first_print = True
                        log_active = False
                        log_buffer = None
                    else:
                        log_buffer = new_log_buffer()
                        log_start = time.monotonic()
                        log_active = True
                elif k == 'g':
                    if mode == MODE_HOLD:
                        trial_led.on()
                        mode         = MODE_OBSERVE
                        gait_phase   = 0.0
                        last_actions = np.zeros(6, dtype=np.float32)
                        needs_history_init = True   # will replicate first real frame
                elif k == 'n':
                    # OBSERVE -> NN CONTROL.  Gated: only valid from
                    # OBSERVE so you have to see the NN output first.
                    # Zero last_actions and re-prime history so we don't
                    # carry OBSERVE's open-loop state (review #3).
                    if mode == MODE_OBSERVE:
                        mode = MODE_NN_CONTROL
                        last_actions = np.zeros(6, dtype=np.float32)
                        needs_history_init = True
                elif k == 'b':
                    # One-shot "go": HOLD -> NN CONTROL + start logging.
                    # Bypasses the OBSERVE gate by design — use 'g' first
                    # if you want to eyeball the policy before handover.
                    # Only valid from HOLD so you can't accidentally
                    # re-arm logging mid-run.
                    if mode == MODE_HOLD:
                        mode         = MODE_NN_CONTROL
                        gait_phase   = 0.0
                        last_actions = np.zeros(6, dtype=np.float32)
                        needs_history_init = True   # will replicate first real frame
                        handover_mode     = 'sighted'
                        blind_first_frame = False
                        log_buffer   = new_log_buffer()
                        # LED edge adjacent to log_start: video ON edge ≈ log t=0.
                        trial_led.on()
                        log_start    = time.monotonic()
                        log_active   = True
                elif k == 'v':
                    # BLIND handover (p12.2 A/B arm, 08-29): same one-shot
                    # HOLD -> NN CONTROL + logging as 'b', but the first NN
                    # frame has its sensor block zeroed and is primed x5 —
                    # the exact training episode-start contract. Tick 0 acts
                    # on zeros (the trained blind reflex); tick 1 sees real
                    # sensors. Expect a different, gentler opening than 'b'.
                    if mode == MODE_HOLD:
                        mode         = MODE_NN_CONTROL
                        gait_phase   = 0.0
                        last_actions = np.zeros(6, dtype=np.float32)
                        needs_history_init = True
                        handover_mode     = 'blind'
                        blind_first_frame = True
                        log_buffer   = new_log_buffer()
                        # LED edge adjacent to log_start: video ON edge ≈ log t=0.
                        trial_led.on()
                        log_start    = time.monotonic()
                        log_active   = True
                elif k == 'h':
                        # LED off first — 'h' is the panic key; mark trial
                        # end on video before the save stall.
                        trial_led.off()
                        # If a log run is active, save before tearing down
                        # so we don't lose data on a panic-hold.
                        if log_active and log_buffer is not None:
                            try:
                                fname, n = save_log(log_buffer, POLICY_PATH, handover_mode, log_meta)
                                sys.stdout.write(
                                    f"\n\n{GREEN}[LOG] Saved {n} ticks to "
                                    f"{fname}{RESET}\n\n"
                                )
                                sys.stdout.flush()
                                # Re-anchor: next redraw lays down fresh
                                # below the save message instead of
                                # ESC_UP-ing over it.
                                first_print = True
                            except Exception as e:
                                sys.stdout.write(
                                    f"\n\n{RED}[LOG] save failed: {e}{RESET}\n\n"
                                )
                                sys.stdout.flush()
                                first_print = True
                            log_active = False
                            log_buffer = None

                        mode         = MODE_HOLD
                        last_actions = np.zeros(6, dtype=np.float32)
                        gait_phase   = 0.0
                        needs_history_init = True
                        blind_first_frame = False   # safety: never carry into next run
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

            # 'k' preset: slew each group's gains toward the K contract,
            # snapping exactly onto the target on arrival.
            if preset_slew:
                slew_done = True
                for g in GROUP_ORDER:
                    for cur, tgt, step in ((kp, PRESET_KP, KP_SLEW_PER_TICK),
                                           (kd, PRESET_KD, KD_SLEW_PER_TICK)):
                        diff = tgt[g] - cur[g]
                        if abs(diff) <= step:
                            cur[g] = tgt[g]
                        else:
                            cur[g] += math.copysign(step, diff)
                            slew_done = False
                if slew_done:
                    preset_slew = False

            kp_vec, kd_vec = expand_gains(kp, kd)

            # ── run NN forward pass when active ─────────────
            # In OBSERVE we run it for display only; in NN_CONTROL
            # the same output also drives the target.  In HOLD we
            # don't bother running it.
            nn_running = (mode in (MODE_OBSERVE, MODE_NN_CONTROL))
            obs_frame = None
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
                # Blind handover ('v'): zero the delayed-sensor block of the
                # FIRST frame only (pg, ang_vel, lin_vel, jpos, jvel,
                # last_actions = obs[0:27]); the bookkeeping tail (gait
                # sin/cos at phase 0 = (0,1), difficulty/grace/init_h, cmds)
                # stays real — byte-identical to what rl_env_p6 emits at a
                # training reset. History init below then primes it x5.
                if blind_first_frame:
                    obs_frame[:27] = 0.0
                    blind_first_frame = False
                # History init: replicate first real frame into all K slots
                # so the policy never sees zero-history OOD input (review #2).
                if needs_history_init:
                    obs_history[:] = obs_frame[np.newaxis, :]
                    needs_history_init = False
                else:
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

            bridge.send_target(target.astype(np.float32), kp_vec, kd_vec,
                   effort_limit=np.array(C.BURST_EFFORT, dtype=np.float32))

            # Compute tau_cmd from the LPF'd target the bridge actually sent,
            # not the unfiltered target (review #8: logs overstate transients).
            filtered = bridge._filtered_target
            sim_tau_cmd = (kp_vec * (filtered - bridge.sim_joint_pos)
                        - kd_vec * bridge.sim_joint_vel)
            # Project to motor frame for display/logging.
            motor_tau_cmd = (sim_tau_cmd * JOINT_SIGN_ARR).astype(np.float32)
            motor_tau_cmd[KNEE_IDX[0]] *= KNEE_GEAR
            motor_tau_cmd[KNEE_IDX[1]] *= KNEE_GEAR

            # ── per-tick logging ─────────────────────────────
            # Only appends while a log run is armed (started by 'b',
            # stopped by 'h' or quit). 'b' is HOLD-gated, so log_active
            # implies nn_running, which means obs_frame is populated.
            if log_active and log_buffer is not None:
                log_buffer['time'].append(time.monotonic() - log_start)
                log_buffer['obs'].append(
                    obs_frame.copy() if obs_frame is not None
                    else np.zeros(OBS_PER_FRAME, dtype=np.float32)
                )
                log_buffer['action'].append(last_actions.copy())
                log_buffer['target'].append(target.astype(np.float32).copy())
                log_buffer['sim_joint_pos'].append(bridge.sim_joint_pos.copy())
                log_buffer['sim_joint_vel'].append(bridge.sim_joint_vel.copy())
                log_buffer['motor_tau_cmd'].append(motor_tau_cmd.copy())
                log_buffer['motor_tau_meas'].append(bridge.motor_torque.copy())
                log_buffer['motor_temp'].append(bridge.motor_temperature.copy())
                log_buffer['proj_grav'].append(imu.projected_gravity().copy())
                log_buffer['proj_grav_raw'].append(imu.projected_gravity_raw().copy())
                log_buffer['ang_vel'].append(imu.ang_vel_body().copy())
                log_buffer['gait_phase'].append(float(gait_phase))
                log_buffer['mode'].append(mode)
                log_buffer['can_replies'].append(bridge.last_replies_parsed)
                # Run configuration (self-contained offline replay; gains can
                # change mid-run via keys, so per-tick not per-run).
                log_buffer['kp'].append(kp_vec.copy())
                log_buffer['kd'].append(kd_vec.copy())
                log_buffer['tau_max'].append(
                    np.array([tau_max[g] for g in SIM_GROUPS], dtype=np.float32)
                )
                log_buffer['lpf_alpha'].append(float(bridge._lpf_alpha))
                log_buffer['filtered_target'].append(
                    bridge._filtered_target.copy()
                )
                # 'tick_busy_ms' is appended at the pacing point below so it
                # includes the display redraw.

            # ── refresh display ─────────────────────────────
            now = time.monotonic()
            if redraw_now or (now - last_print >= 1.0 / PRINT_HZ):
                lines = build_display(
                    mode         = mode,
                    kp = kp, kd = kd, tau_max = tau_max,
                    active_group = active_group,
                    target       = target,
                    sim_pos      = bridge.sim_joint_pos,
                    sim_vel      = bridge.sim_joint_vel,
                    tau_cmd      = motor_tau_cmd,                   # motor frame
                    tau_meas     = bridge.motor_torque,             # motor frame
                    temp_c       = bridge.motor_temperature,
                    proj_grav    = imu.projected_gravity(),
                    ang_vel      = imu.ang_vel_body(),
                    last_action  = last_actions,
                    gait_phase   = gait_phase,
                    lpf_alpha    = bridge._lpf_alpha,
                    pg_offset    = imu.pg_offset,
                    log_active   = log_active,
                    log_len      = (len(log_buffer['time'])
                                    if log_buffer is not None else 0),
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
            if log_active and log_buffer is not None:
                # busy time this tick (compute + CAN + redraw), before the
                # sleep — overrun forensics for the fixed-step gait clock.
                log_buffer['tick_busy_ms'].append(elapsed * 1000.0)
            if elapsed < period:
                time.sleep(period - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        trial_led.close()
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

        # Final flush of the log buffer if the user quit (q / Ctrl-C)
        # without pressing 'h' first. Don't let the run data evaporate.
        if log_active and log_buffer is not None and len(log_buffer['time']) > 0:
            try:
                fname, n = save_log(log_buffer, POLICY_PATH, handover_mode, log_meta)
                print(f"\n[LOG] Saved {n} ticks to {fname} (on exit)")
            except Exception as e:
                print(f"\n[WARN] log save failed: {e}")

        print("\n\n  Final gains:")
        for g in GROUP_ORDER:
            print(f"    {g:<10s}  Kp = {kp[g]:7.2f}    Kd = {kd[g]:6.3f}")
        print("  Motors disabled. Done.")


if __name__ == '__main__':
    main()