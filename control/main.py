"""
main.py — entry point.

Hardware initialization → RC monitoring loop → 100Hz controller tick.

Motor zeroing is performed OUT-OF-BAND by test_motors.py:
    1. Place robot in KNEEL pose
    2. sudo python3 test_motors.py   (this writes zeros to the motors)
    3. sudo python3 main.py          (this assumes zeros are already set)

This script does NOT zero motors and does NOT auto-level. If sim values
look wrong on startup, re-run test_motors.py at the kneel pose.

RC mapping:
    CH8 (Aux4) high  → arm
    CH8 (Aux4) low   → disarm
    CH7 (Aux3) high  → trigger standup (rising edge only)
    CH2 (E)          → vx command
    CH4 (R)          → yaw rate command

Run:
    sudo python3 main.py                          # use defaults
    sudo python3 main.py --policy /path/to.pt
    sudo python3 main.py --duration 60            # auto-stop after 60s
    sudo python3 main.py --no-rt                  # skip realtime scheduling
"""

import argparse
import datetime
import os
import signal
import time
import traceback

import numpy as np

import config as C
from motor_bridge import MotorBridge
from imu_adapter import IMUAdapter
from policy import Policy
from controller import BipedController, DISARMED, READY, STANDUP, WALK
from biped_rc_mapper import BipedRCMapper
from crsf_reader import CRSFReader


def _try_realtime():
    try:
        os.sched_setaffinity(0, {2})
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(50))
        print("[Main] Realtime scheduling enabled (SCHED_FIFO core 2)")
    except PermissionError:
        print("[Main] No RT permissions, running normal priority")
    except Exception as e:
        print(f"[Main] RT scheduling skipped: {e}")


def _to_np(x):
    """Convert torch tensor / array / None into a flat numpy array (or None)."""
    if x is None:
        return None
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x).flatten()


def _save_standup_log(timestamps, obs_list, act_list, partial=False):
    """Dump captured obs/action samples to ./standup_log_<ts>.txt (CSV)."""
    if not timestamps:
        return None
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_partial" if partial else ""
    fname = f"standup_log_{ts}{suffix}.txt"
    try:
        ts_arr  = np.asarray(timestamps, dtype=np.float64).reshape(-1, 1)
        obs_arr = np.asarray(obs_list,   dtype=np.float64)
        act_arr = np.asarray(act_list,   dtype=np.float64)
        rows = np.hstack([ts_arr, obs_arr, act_arr])

        # Build header dynamically so we don't lie about column count if
        # obs/action shapes ever change. Use known names where available,
        # generic obs_i / act_i otherwise.
        n_obs = obs_arr.shape[1] if obs_arr.ndim == 2 else 0
        n_act = act_arr.shape[1] if act_arr.ndim == 2 else 0
        obs_names = list(_OBS_COL_NAMES[:n_obs]) + \
                    [f'obs_{i}' for i in range(len(_OBS_COL_NAMES), n_obs)]
        act_names = list(_ACT_COL_NAMES[:n_act]) + \
                    [f'act_{i}' for i in range(len(_ACT_COL_NAMES), n_act)]
        header = ','.join(['t'] + obs_names + act_names)

        np.savetxt(fname, rows, delimiter=',', fmt='%.6f',
                   header=header, comments='')
        return os.path.abspath(fname)
    except Exception as e:
        print(f"\n[Main] Failed to save standup log: {e}")
        return None


# Column names for the 36-D obs and 6-D action vectors — must match the
# layout in the status block in main(). Used as the .txt header.
_OBS_COL_NAMES = (
    'proj_grav_x', 'proj_grav_y', 'proj_grav_z',
    'ang_vel_x',   'ang_vel_y',   'ang_vel_z',
    'lin_vel_x',   'lin_vel_y',   'lin_vel_z',
    'hip_roll1_pos', 'hip_pitch1_pos', 'knee1_pos',
    'hip_roll2_pos', 'hip_pitch2_pos', 'knee2_pos',
    'hip_roll1_vel', 'hip_pitch1_vel', 'knee1_vel',
    'hip_roll2_vel', 'hip_pitch2_vel', 'knee2_vel',
    'last_act_hr1', 'last_act_hp1', 'last_act_kn1',
    'last_act_hr2', 'last_act_hp2', 'last_act_kn2',
    'gait_sin', 'gait_cos', 'difficulty',
    'grace_norm', 'init_h', 'vx_cmd',
    'yaw_rate_cmd', 'yaw_err_sin', 'yaw_err_cos',
)
_ACT_COL_NAMES = (
    'act_hip_roll1', 'act_hip_pitch1', 'act_knee1',
    'act_hip_roll2', 'act_hip_pitch2', 'act_knee2',
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy',   type=str, default=C.POLICY_PATH)
    ap.add_argument('--duration', type=float, default=3600.0)
    ap.add_argument('--no-rt',    action='store_true')
    ap.add_argument('--device',   type=str, default='cpu')
    args = ap.parse_args()

    if not args.no_rt:
        _try_realtime()

    # ── ELRS first so we can refuse to start without RC link ──────────
    crsf = CRSFReader(C.ELRS_PORT, C.ELRS_BAUD)
    crsf.start()
    rc_map = BipedRCMapper()
    print(f"[Main] ELRS reader started on {C.ELRS_PORT}, waiting for link...")
    t0 = time.monotonic()
    while time.monotonic() - t0 < 10.0:
        _, linked = crsf.get_channels()
        if linked:
            print(f"[Main] RC linked ({time.monotonic() - t0:.1f}s)")
            break
        time.sleep(0.1)
    else:
        print("[Main] WARNING: no RC link after 10s — continuing anyway")

    # ── IMU ──────────────────────────────────────────────────────────
    imu = IMUAdapter()
    imu.open()
    time.sleep(0.5)

    # ── Motors ───────────────────────────────────────────────────────
    # Zeros are assumed to already be set via test_motors.py at kneel pose.
    # No auto-level, no auto-zero here.
    motors = MotorBridge()

    # ── Policy ───────────────────────────────────────────────────────
    policy = Policy(args.policy, device=args.device)

    # ── Controller ───────────────────────────────────────────────────
    ctrl = BipedController(motors, imu, policy)

    # ── Shutdown handling ────────────────────────────────────────────
    shutdown = False
    def _sig(_s, _f):
        nonlocal shutdown
        shutdown = True
    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)

    print(f"\n{'=' * 65}")
    print(f"  BIPED SIM2REAL — Phase 6 policy")
    print(f"{'=' * 65}")
    print(f"  Control rate : {C.CONTROL_HZ} Hz   ({C.DT * 1000:.1f} ms)")
    print(f"  Motor IDs    : {C.MOTOR_IDS}")
    print(f"  Knee gear    : {C.KNEE_GEAR_RATIO:.4f}  "
          f"(coupling = {C.KNEE_HIP_COUPLING})")
    print(f"  Standup      : grace = {C.STANDUP_GRACE_SECONDS}s "
          f"({C.MAX_GRACE_STEPS} steps)")
    print(f"  RC bindings  : CH2→vx, CH4→yaw, CH7→standup, CH8→arm")
    print(f"  Safety       : tilt limit ±{C.MAX_TILT_RAD:.2f} rad")
    print(f"  Zeroing      : assumed pre-set via test_motors.py at kneel")
    print(f"{'=' * 65}\n")
    print(f"  State machine:")
    print(f"    CH8 high  → ARM (soft hold, robot stays where it is)")
    print(f"    CH7 high  → STANDUP (from kneel, 5s grace + burst torque)")
    print(f"    CH2/CH4   → WALK (if already standing in READY)")
    print(f"    CH8 low   → DISARM (motors off)\n")

    # ── State edges ──────────────────────────────────────────────────
    prev_armed   = False
    prev_standup = False

    last_print = time.monotonic()
    loop_times = []
    overrun_count = 0
    status_lines_printed = 0   # for in-place ANSI overwrite of the status block

    # ── Standup obs/action logging ───────────────────────────────────
    # On rising edge of CH7 standup, capture obs+action at 10 Hz for 5 s
    # and dump to ./standup_log_<timestamp>.txt (CSV) in the cwd.
    LOG_DURATION = 5.0
    LOG_PERIOD   = 0.1   # 10 Hz
    log_active     = False
    log_t_start    = 0.0
    log_t_next     = 0.0
    log_timestamps = []
    log_obs        = []
    log_actions    = []

    try:
        t_start = time.monotonic()
        next_tick = time.monotonic()

        while not shutdown:
            now = time.monotonic()
            if now - t_start > args.duration:
                print("\n[Main] Duration reached.")
                break

            # ── Sample RC ────────────────────────────────────────────
            channels, linked = crsf.get_channels()
            cmds = rc_map.map(channels)
            armed_req     = cmds['armed']   and linked
            standup_req   = cmds['standup'] and linked

            # Lost-link safety: force disarm
            if not linked:
                armed_req = False

            # Edge transitions
            if armed_req and not prev_armed:
                ctrl.arm()
            elif (not armed_req) and prev_armed:
                ctrl.disarm()
            prev_armed = armed_req

            if standup_req and not prev_standup and ctrl.state in (READY, WALK):
                ctrl.trigger_standup()
                if not log_active:
                    log_active     = True
                    log_t_start    = time.monotonic()
                    log_t_next     = log_t_start
                    log_timestamps = []
                    log_obs        = []
                    log_actions    = []
                    print("\n[Main] Standup logging armed "
                          "(5s @ 10Hz → ./standup_log_*.txt)")
            prev_standup = standup_req

            # Forward locomotion commands (only honored in WALK)
            ctrl.set_commands(cmds['vx_cmd'], cmds['yaw_rate_cmd'])

            # ── Tick ─────────────────────────────────────────────────
            tick_start = time.monotonic()
            ctrl.step()
            tick_ms = (time.monotonic() - tick_start) * 1000
            loop_times.append(tick_ms)
            if len(loop_times) > 200:
                loop_times.pop(0)

            # ── Standup log sampling (10 Hz for 5 s after standup edge) ──
            if log_active:
                t_log = time.monotonic()
                if t_log >= log_t_next:
                    obs_s = _to_np(getattr(ctrl, '_last_obs',    None))
                    act_s = _to_np(getattr(ctrl, '_last_action', None))
                    if obs_s is not None and act_s is not None:
                        log_timestamps.append(t_log - log_t_start)
                        log_obs.append(obs_s)
                        log_actions.append(act_s)
                    log_t_next += LOG_PERIOD
                if t_log - log_t_start >= LOG_DURATION:
                    path = _save_standup_log(log_timestamps, log_obs, log_actions)
                    if path:
                        print(f"\n[Main] Standup log saved → {path} "
                              f"({len(log_timestamps)} samples)")
                    log_active = False

            # ── Status print (~5 Hz) ─────────────────────────────────
            if now - last_print > 0.2:
                last_print = now
                if not hasattr(ctrl, '_dumped'):
                    print("\n[DEBUG] ctrl attrs:",
                          [a for a in dir(ctrl) if not a.startswith('__')])
                    ctrl._dumped = True
                tilt_deg = (1.0 - ctrl.imu.upright_cos()) * 90.0  # rough
                grace_s  = ctrl._grace_remaining * C.DT
                rc_flag  = 'RC' if linked else '--'
                avg_ms   = sum(loop_times) / len(loop_times) if loop_times else 0
                max_ms   = max(loop_times) if loop_times else 0

                # Pull obs / action from controller (tolerate torch tensors / None)
                obs = _to_np(getattr(ctrl, '_last_obs',    None))
                act = _to_np(getattr(ctrl, '_last_action', None))

                lines = [
                    f"  [{ctrl.state:8s}] {rc_flag}  "
                    f"tilt~{tilt_deg:4.1f}°  grace={grace_s:4.2f}s  "
                    f"tick={avg_ms:4.1f}/{max_ms:4.1f}ms  "
                    f"imu={imu.fps:5.1f}Hz  overrun={overrun_count}"
                ]

                if obs is not None and obs.size >= 36:
                    lines.append("  OBS (36D):")
                    lines.append(f"{'proj_grav_x':>18}={obs[ 0]:+.4f}{'proj_grav_y':>16}={obs[ 1]:+.4f}{'proj_grav_z':>16}={obs[ 2]:+.4f}")
                    lines.append(f"{'ang_vel_x':>18}={obs[ 3]:+.4f}{'ang_vel_y':>16}={obs[ 4]:+.4f}{'ang_vel_z':>16}={obs[ 5]:+.4f}")
                    lines.append(f"{'lin_vel_x':>18}={obs[ 6]:+.4f}{'lin_vel_y':>16}={obs[ 7]:+.4f}{'lin_vel_z':>16}={obs[ 8]:+.4f}")
                    lines.append(f"{'hip_roll1_pos':>18}={obs[ 9]:+.4f}{'hip_pitch1_pos':>16}={obs[10]:+.4f}{'knee1_pos':>16}={obs[11]:+.4f}")
                    lines.append(f"{'hip_roll2_pos':>18}={obs[12]:+.4f}{'hip_pitch2_pos':>16}={obs[13]:+.4f}{'knee2_pos':>16}={obs[14]:+.4f}")
                    lines.append(f"{'hip_roll1_vel':>18}={obs[15]:+.4f}{'hip_pitch1_vel':>16}={obs[16]:+.4f}{'knee1_vel':>16}={obs[17]:+.4f}")
                    lines.append(f"{'hip_roll2_vel':>18}={obs[18]:+.4f}{'hip_pitch2_vel':>16}={obs[19]:+.4f}{'knee2_vel':>16}={obs[20]:+.4f}")
                    lines.append(f"{'last_act_hr1':>18}={obs[21]:+.4f}{'last_act_hp1':>16}={obs[22]:+.4f}{'last_act_kn1':>16}={obs[23]:+.4f}")
                    lines.append(f"{'last_act_hr2':>18}={obs[24]:+.4f}{'last_act_hp2':>16}={obs[25]:+.4f}{'last_act_kn2':>16}={obs[26]:+.4f}")
                    lines.append(f"{'gait_sin':>18}={obs[27]:+.4f}{'gait_cos':>16}={obs[28]:+.4f}{'difficulty':>16}={obs[29]:+.4f}")
                    lines.append(f"{'grace_norm':>18}={obs[30]:+.4f}{'init_h':>16}={obs[31]:+.4f}{'vx_cmd':>16}={obs[32]:+.4f}")
                    lines.append(f"{'yaw_rate_cmd':>18}={obs[33]:+.4f}{'yaw_err_sin':>16}={obs[34]:+.4f}{'yaw_err_cos':>16}={obs[35]:+.4f}")

                if act is not None and act.size >= 6:
                    lines.append("  ACTIONS (6D):")
                    lines.append(f"{'hip_roll1':>14} = {act[0]:+.4f}")
                    lines.append(f"{'hip_pitch1':>14} = {act[1]:+.4f}")
                    lines.append(f"{'knee1':>14} = {act[2]:+.4f}")
                    lines.append(f"{'hip_roll2':>14} = {act[3]:+.4f}")
                    lines.append(f"{'hip_pitch2':>14} = {act[4]:+.4f}")
                    lines.append(f"{'knee2':>14} = {act[5]:+.4f}")

                # In-place overwrite: move cursor up to top of previous block,
                # then clear from cursor to end of screen.
                if status_lines_printed > 0:
                    print(f"\033[{status_lines_printed}F\033[J", end='')
                print("\n".join(lines), flush=True)
                status_lines_printed = len(lines)
            # ── Sleep to maintain CONTROL_HZ ─────────────────────────
            next_tick += C.DT
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            elif sleep < -0.05:
                # Loop overran by >50ms: resync, don't death-spiral
                overrun_count += 1
                next_tick = time.monotonic()

    except Exception:
        print("\n[Main] Fatal error:")
        traceback.print_exc()
    finally:
        print("\n[Main] Shutting down...")
        if log_active and log_timestamps:
            path = _save_standup_log(log_timestamps, log_obs, log_actions,
                                     partial=True)
            if path:
                print(f"[Main] Partial standup log saved → {path} "
                      f"({len(log_timestamps)} samples)")
        try:
            ctrl.disarm()
        except Exception:
            pass
        try:
            motors.close()
        except Exception:
            pass
        try:
            imu.close()
        except Exception:
            pass
        try:
            crsf.stop()
        except Exception:
            pass
        print("[Main] Done.")


if __name__ == '__main__':
    main()