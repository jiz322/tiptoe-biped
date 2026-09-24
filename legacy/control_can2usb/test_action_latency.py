#!/usr/bin/env python3
"""
test_action_latency.py — command-to-motion (action) latency.

Robot hung in the air. Motors stream a limp command, so the joints droop away
from their zero pose. On each trigger the script lights an LED (BCM GPIO18) and,
in the same breath, releases a "go to zero position" MIT command (your KP/KD
below) to the joints. Film at 400 Hz: the LED rising edge is t0 (command
released); count frames to the first frame any joint visibly starts to move.

    delta_frames * 2.5 ms = action latency
      = command encode + serial/CAN tx + motor control loop + current ramp +
        torque + mechanical onset

Two deliberate choices:
  - The LED is fired immediately BEFORE the command bytes are written, so t0 can't
    land after the command actually leaves the host — that keeps the number from
    reading optimistically short.
  - It does NOT call driver.send_command() (which blocks ~1 reply round-trip
    waiting for feedback). It writes the pre-encoded frames straight to the port
    and ignores replies — this test reads motion from the video, not from feedback.

ZERO POSITION:
  MIT frames are written in RAW motor space, but the robot's zero pose lives in
  SIM space, and the two are NOT the same. As in test_position.compute_sim(), raw
  maps to sim through per-joint sign, offset, the knee gear ratio, and the knee<-hip
  coupling, so commanding raw 0.0 does NOT put the robot at its zero pose. This
  script inverts that transform to find the raw target each joint must be driven to
  so that its sim position reads TARGET_SIM_POS (0.0 = zero pose). The raw targets
  are solved once at setup — before the LED/write hot path — so the pre-encode and
  real-time guarantees are unchanged.

Set KP / KD / TARGET_SIM_POS at the top. TARGET_SIM_POS is in SIM radians.

SAFETY: a hung robot snapping to zero under high KP can move hard. Start with a
LOW KP, keep the robot clear of obstacles and your hands, and raise it gradually.
A higher KP gives a sharper motion onset (easier to time) at the cost of force.

Usage:
    sudo python3 test_action_latency.py
    Press Enter to fire one action; Ctrl-C to quit.
"""

import os
import sys
import gc
import time
import select

sys.path.append('..')
import config as C
from robstride_biped_driver import RobStrideDriver
import lgpio


# ===== action command (TUNE — start low) =====
KP = 10.0             # position gain; higher = snappier motion = sharper onset, more force
KD = 0.5              # damping
TARGET_SIM_POS = 0.0  # target in SIM rad (0.0 = robot zero pose); converted to raw per joint
TARGET_MOTORS = None #[102]  # None = all motors; or e.g. [103] to move ONE joint (tightest timing)

HOLD_SECONDS = 2.0    # keep driving to zero so the move completes and holds
STREAM_HZ    = 100    # command refresh during the hold (keeps motors alive)

LED_PIN   = 18        # BCM
GPIO_CHIP = 0         # Pi 5: try 4 if the LED never lights
RT_PRIORITY = 80
RT_CPU      = 3


def compute_raw_targets(sim_target):
    """Inverse of test_position.compute_sim(): desired SIM position (rad) -> the
    RAW motor target each joint must be commanded to.

    Forward (raw -> sim) for the six joints, matching motor_bridge:
        non-knee:  sim = raw*sign + off
        knee:      sim = raw*sign*gear - sim_hip*coupling + off

    Solving for raw:
        non-knee:  raw = (sim - off) / sign
        knee:      raw = (sim - off + sim_hip*coupling) / (sign*gear)

    sim_target is applied uniformly to all six joints, so 0.0 yields the zero pose
    (and the knee coupling term drops out because the hip sim is then 0 too).
    Returns {motor_id: raw_target_rad}, aligned with C.MOTOR_IDS.
    """
    sign = C.JOINT_SIGN
    off = C.JOINT_OFFSET
    gear = C.KNEE_GEAR_RATIO
    coupling = C.KNEE_HIP_COUPLING

    sim = [sim_target] * 6                                   # desired sim for every joint
    raw = [(sim[i] - off[i]) / sign[i] for i in range(6)]    # hips / rolls
    raw[2] = (sim[2] - off[2] + sim[1] * coupling) / (sign[2] * gear)  # left knee
    raw[5] = (sim[5] - off[5] + sim[4] * coupling) / (sign[5] * gear)  # right knee
    return {mid: raw[i] for i, mid in enumerate(C.MOTOR_IDS)}


def set_realtime(priority, cpu):
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(priority))
    except PermissionError:
        print("[warn] no RT priority (run with sudo for SCHED_FIFO)")
    except (OSError, AttributeError, ValueError) as e:
        print(f"[warn] sched_setscheduler failed: {e}")
    try:
        os.sched_setaffinity(0, {cpu})
    except OSError as e:
        print(f"[warn] could not pin to CPU {cpu}: {e}")


def main():
    ids = list(C.MOTOR_IDS) if TARGET_MOTORS is None else list(TARGET_MOTORS)

    # Solve raw targets for the zero pose (sim -> raw) ONCE, up front.
    raw_targets = compute_raw_targets(TARGET_SIM_POS)
    unknown = [m for m in ids if m not in raw_targets]
    if unknown:
        raise SystemExit(f"[err] motor id(s) {unknown} not in C.MOTOR_IDS={list(C.MOTOR_IDS)}")

    driver = RobStrideDriver(C.MOTOR_PORT, C.MOTOR_IDS)
    driver.enable_all()
    time.sleep(0.3)

    try:
        h = lgpio.gpiochip_open(GPIO_CHIP)
    except lgpio.error as e:
        print(f"[err] gpiochip {GPIO_CHIP} open failed: {e}; try GPIO_CHIP=4")
        driver.close()
        raise
    lgpio.gpio_claim_output(h, LED_PIN, 0)

    # Pre-encode once so firing does no float math between the LED and the write.
    # Position is the per-joint RAW target that lands sim at TARGET_SIM_POS.
    action = [driver._encode_mit_frame(m, raw_targets[m], 0.0, KP, KD, 0.0) for m in ids]
    limp   = [driver._encode_mit_frame(m, 0.0, 0.0, 0.0, 0.0, 0.0) for m in ids]

    def burst(frames):
        for fr in frames:
            driver.ser.write(fr)

    set_realtime(RT_PRIORITY, RT_CPU)
    gc.disable()
    period = 1.0 / STREAM_HZ

    print(f"\n  Action latency test — moving {ids} to sim {TARGET_SIM_POS} rad (KP={KP}, KD={KD})")
    for m in ids:
        print(f"    motor {m}: raw target = {raw_targets[m]:+.4f} rad")
    print("  Robot should hang freely; joints will droop, then snap to zero on Enter.")
    print("  Press Enter to FIRE one action; Ctrl-C to quit.\n")

    trial = 0
    try:
        while True:
            # keep-alive: stream limp (joints droop) until Enter is pressed
            driver.ser.reset_input_buffer()           # drop unread replies so the buffer can't pile up
            while True:
                dr, _, _ = select.select([sys.stdin], [], [], 0)
                if dr:
                    sys.stdin.readline()
                    break
                burst(limp)
                time.sleep(period)

            trial += 1
            # ===== t0: LED edge FIRST, then release the command burst (no reply wait) =====
            lgpio.gpio_write(h, LED_PIN, 1)
            burst(action)
            t_fire = time.monotonic()
            print(f"  [{trial}] FIRED — watch the video: LED-on -> first joint motion")

            # hold: keep streaming the zero target so the motion completes and holds
            while time.monotonic() - t_fire < HOLD_SECONDS:
                burst(action)
                time.sleep(period)
            lgpio.gpio_write(h, LED_PIN, 0)

    except KeyboardInterrupt:
        print()
    finally:
        for _ in range(3):
            burst(limp)
            time.sleep(0.01)
        try:
            lgpio.gpio_write(h, LED_PIN, 0)
            lgpio.gpiochip_close(h)
        except Exception:
            pass
        driver.close()      # disables motors, then closes the port
        print("  done.")


if __name__ == '__main__':
    main()