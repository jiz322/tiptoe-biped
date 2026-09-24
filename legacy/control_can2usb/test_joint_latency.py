#!/usr/bin/env python3
"""
test_joint_latency.py — absolute observation latency of a robot joint angle.

Same idea as the IMU latency test: make one physical event that the 400 Hz camera
and the joint-position stream both witness, fire an LED (BCM GPIO18) the instant
the software detects it, and count frames in the video.

Position has no sharp spike of its own (it's the integral of motion), so you key
off a velocity feature derived from position deltas:

  stop  (default): let the limp joint swing into a hard stop. Speed is high during
                   the swing and drops sharply at the stop; fire when it falls
                   below a fraction of the swing peak.  Video t0: the frame where
                   the limb's per-frame displacement drops sharply.
  onset:           give the limp joint a sharp flick from rest. Fire when speed
                   first rises above a small threshold.  Video t0: the first frame
                   of visible motion.

Fire the LED on the SAME feature you call t0 in the video (stop->stop, onset->onset)
so both sides are anchored to the same instant.

THE READ PATH MATTERS HERE.
  This polls ONE joint as fast as the bus allows (send limp command -> get_state).
  The measured latency includes the motor feedback + CAN + USB adapter + driver
  parse, PLUS up to one poll period of staleness. The poll period is printed; if
  it's large, that staleness is a real part of your obs latency. If the joint
  feedback goes through a USB-serial CAN adapter, its latency_timer often dominates
  (the classic FTDI 16 ms default) — this script tries to set it to 1 ms; verify it
  took. Polling all six joints (as test_position.py does, with 1 ms sleeps) would
  serialize the round trips and inflate latency, so the test isolates one joint.

Usage:
    sudo python3 test_joint_latency.py --calibrate              # see swing speed + poll rate
    sudo python3 test_joint_latency.py --motor-id 1             # run (stop mode)
    sudo python3 test_joint_latency.py --mode onset --motor-id 1
Ctrl-C to finish; writes a CSV.
"""

import os
import sys
import gc
import math
import time
import argparse
import threading
from collections import deque

sys.path.append('..')
import config as C
from robstride_biped_driver import RobStrideDriver
import lgpio


SHORT_LABELS = ['L_roll', 'L_hip', 'L_knee', 'R_roll', 'R_hip', 'R_knee']

# ------------------------------ Config -------------------------------
LED_PIN   = 18      # BCM
GPIO_CHIP = 0       # Pi 5: try 4 if the LED never lights

RT_PRIORITY = 80
RT_CPU      = 3     # isolate via isolcpus=3

# Detection, in rad and rad/s. Tune with --calibrate.
MOVE_MIN_SPEED = 0.1    # rad/s; a swing must exceed this to count as a real move
STOP_FRACTION  = 0.07    # stop fires when speed drops below STOP_FRACTION * swing peak
ONSET_SPEED    = 0.5    # rad/s; onset fires when speed first exceeds this
SPEED_WINDOW   = 4      # samples used to estimate speed (smooths encoder/timing noise)

LED_ON_SECONDS   = 0.15
COOLDOWN_SECONDS = 0.5
QUIET_SPEED      = 0.12   # rad/s
QUIET_SECONDS    = 0.3

# state machine
WAIT_QUIET, ARMED, WAIT_MOVING, MOVING, COOLDOWN = 0, 1, 2, 3, 4


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


def set_usb_latency_timer(port, value=1):
    """Best-effort: drop a USB-serial adapter's latency_timer (FTDI default is 16 ms).
    No-op for native CAN (socketcan) or non-FTDI adapters."""
    try:
        dev = os.path.basename(str(port))                 # e.g. ttyUSB0
        path = f"/sys/bus/usb-serial/devices/{dev}/latency_timer"
        with open(path, "w") as f:
            f.write(str(value))
        with open(path) as f:
            got = f.read().strip()
        print(f"[adapter] {dev} latency_timer = {got} ms")
    except (OSError, ValueError) as e:
        print(f"[adapter] latency_timer not set ({e}); "
              f"if using a USB-CAN adapter, set it manually and re-check")


class JointLatencyTester:
    def __init__(self, motor_id, mode="stop", calibrate=False):
        self.mid = motor_id
        self.mode = mode
        self.calibrate = calibrate
        self.driver = None
        self.h = None
        self.running = False
        self.poller = None

        self.state = WAIT_QUIET
        self.led_on = False
        self.t_led_ns = 0
        self.t_quiet_ns = time.monotonic_ns()
        self.prev_speed = 0.0
        self.peak_speed = 0.0
        self.cur_trial = None

        self.hist = deque(maxlen=SPEED_WINDOW + 1)   # (t_ns, pos_rad)
        self.trials = []
        self.trial_lock = threading.Lock()

        # diagnostics
        self.cal_peak = 0.0
        self.sample_count = 0
        self.loop_dt_sum_ns = 0
        self.loop_dt_max_ns = 0
        self.pos_min = math.inf
        self.pos_max = -math.inf

    # ------------------------------ open ------------------------------
    def open(self):
        set_usb_latency_timer(C.MOTOR_PORT)
        self.driver = RobStrideDriver(C.MOTOR_PORT, C.MOTOR_IDS)
        self.driver.enable_all()
        time.sleep(0.3)

        if self.mid not in C.MOTOR_IDS:
            print(f"[err] motor id {self.mid} not in {C.MOTOR_IDS}")
            return False
        label = SHORT_LABELS[C.MOTOR_IDS.index(self.mid)]
        print(f"[motor] testing joint {label} (id {self.mid}), mode={self.mode}")

        self._poll_once()        # prime the speed window
        self._poll_once()

        try:
            self.h = lgpio.gpiochip_open(GPIO_CHIP)
        except lgpio.error as e:
            print(f"[err] gpiochip {GPIO_CHIP} open failed: {e}; try GPIO_CHIP=4")
            raise
        lgpio.gpio_claim_output(self.h, LED_PIN, 0)
        print(f"[gpio] LED on BCM{LED_PIN} ready, LOW")

        self.running = True
        self.poller = threading.Thread(target=self._poll_loop, daemon=True)
        self.poller.start()
        return True

    # ------------------------- poll + detect --------------------------
    def _poll_once(self):
        # keep the joint limp (kp=kd=tau=0) and elicit a feedback frame
        self.driver.send_command(self.mid, 0, 0, 0, 0, 0)
        p, *_ = self.driver.get_state(self.mid)
        t = time.monotonic_ns()
        self.hist.append((t, p))
        return t, p

    def _speed_from_hist(self):
        if len(self.hist) < 2:
            return 0.0
        t1, p1 = self.hist[0]
        t2, p2 = self.hist[-1]
        dt = (t2 - t1) * 1e-9
        if dt <= 0:
            return 0.0
        return abs(p2 - p1) / dt

    def _poll_loop(self):
        set_realtime(RT_PRIORITY, RT_CPU)
        gc.disable()
        prev_t = self.hist[-1][0] if self.hist else time.monotonic_ns()
        while self.running:
            self.driver.send_command(self.mid, 0, 0, 0, 0, 0)
            p, *_ = self.driver.get_state(self.mid)
            t = time.monotonic_ns()             # timestamp the position sample
            self.hist.append((t, p))
            self.sample_count += 1

            dt = t - prev_t
            prev_t = t
            if dt > 0:
                self.loop_dt_sum_ns += dt
                if dt > self.loop_dt_max_ns:
                    self.loop_dt_max_ns = dt

            speed = self._speed_from_hist()

            if self.calibrate:
                if speed > self.cal_peak:
                    self.cal_peak = speed
                if p < self.pos_min:
                    self.pos_min = p
                if p > self.pos_max:
                    self.pos_max = p
                self.prev_speed = speed
                continue

            self._detect(speed, p, t)
            self._service_timers()
            self.prev_speed = speed

    def _detect(self, speed, p, t):
        s = self.state
        if s == WAIT_QUIET:
            if speed > QUIET_SPEED:
                self.t_quiet_ns = t                 # still moving; reset settle timer
            elif t - self.t_quiet_ns >= QUIET_SECONDS * 1e9:
                if self.mode == "onset":
                    self.state = ARMED
                    print("[armed] flick the joint from rest")
                else:
                    self.state = WAIT_MOVING
                    print("[armed] swing the joint into the stop")
        elif s == ARMED:                            # onset mode
            if speed > ONSET_SPEED and self.prev_speed <= ONSET_SPEED:
                self._fire(p, t, speed)
        elif s == WAIT_MOVING:                       # stop mode
            if speed > MOVE_MIN_SPEED:
                self.peak_speed = speed
                self.state = MOVING
        elif s == MOVING:                            # stop mode
            if speed > self.peak_speed:
                self.peak_speed = speed
            if speed < STOP_FRACTION * self.peak_speed:
                self._fire(p, t, self.peak_speed)
        # COOLDOWN is exited in _service_timers

    def _fire(self, p, t_event, peak_speed):
        # ===== CRITICAL: write the LED edge FIRST, timestamp immediately =====
        lgpio.gpio_write(self.h, LED_PIN, 1)
        t_led = time.monotonic_ns()
        self.led_on = True
        self.t_led_ns = t_led
        self.state = COOLDOWN
        trial = {
            'n': len(self.trials) + 1,
            't_event_ns': t_event,
            't_led_ns': t_led,
            'pos_rad': p,
            'peak_speed': peak_speed,
            'recv_to_led_us': (t_led - t_event) / 1000.0,
        }
        with self.trial_lock:
            self.trials.append(trial)
        self.cur_trial = trial
        print(f"[TRIAL {trial['n']:>3}] LED fired | pos={p:+.3f} rad  "
              f"peak={peak_speed:5.2f} rad/s | software recv->led="
              f"{trial['recv_to_led_us']:6.1f} us")

    def _service_timers(self):
        if not self.led_on and self.state != COOLDOWN:
            return
        now = time.monotonic_ns()
        if self.led_on and now - self.t_led_ns >= LED_ON_SECONDS * 1e9:
            lgpio.gpio_write(self.h, LED_PIN, 0)
            self.led_on = False
        if self.state == COOLDOWN and now - self.t_led_ns >= COOLDOWN_SECONDS * 1e9:
            self.state = WAIT_QUIET
            self.t_quiet_ns = now

    # ----------------------------- close ------------------------------
    def close(self):
        self.running = False
        if self.poller is not None:
            self.poller.join(timeout=2)
        if self.h is not None:
            try:
                lgpio.gpio_write(self.h, LED_PIN, 0)
                lgpio.gpiochip_close(self.h)
            except Exception:
                pass
        if self.driver is not None:
            try:
                for mid in self.driver.motor_ids:
                    self.driver.send_command(mid, 0, 0, 0, 0, 0)
                self.driver.disable_all()
                self.driver.close()
            except Exception:
                pass
        print("[closed]")


def main():
    ap = argparse.ArgumentParser(description="Robot joint angle observation latency test")
    ap.add_argument('--motor-id', type=int, default=C.MOTOR_IDS[0],
                    help="which joint to test (default: first in C.MOTOR_IDS)")
    ap.add_argument('--mode', choices=['stop', 'onset'], default='stop',
                    help="event type: swing into a stop, or flick from rest")
    ap.add_argument('--calibrate', action='store_true',
                    help="print peak swing speed, poll period, and position range")
    ap.add_argument('--csv', default='joint_latency_trials.csv')
    args = ap.parse_args()

    t = JointLatencyTester(args.motor_id, mode=args.mode, calibrate=args.calibrate)
    if not t.open():
        sys.exit(1)
    time.sleep(0.3)

    if args.calibrate:
        print("\n=== CALIBRATION ===  swing the limp joint a few times (into the stop).")
        print("Set MOVE_MIN_SPEED below your swing peak; watch the poll period. Ctrl-C to stop.\n")
        try:
            while True:
                time.sleep(0.25)
                n = max(1, t.sample_count)
                mean_dt = t.loop_dt_sum_ns / n / 1000.0
                max_dt = t.loop_dt_max_ns / 1000.0
                lo = t.pos_min if math.isfinite(t.pos_min) else 0.0
                hi = t.pos_max if math.isfinite(t.pos_max) else 0.0
                print(f"  peak speed: {t.cal_peak:6.2f} rad/s | "
                      f"poll dt mean/max: {mean_dt:6.1f}/{max_dt:7.1f} us | "
                      f"pos range: [{lo:+.2f}, {hi:+.2f}] rad   ", end='\r')
                t.cal_peak = 0.0
        except KeyboardInterrupt:
            print("\ndone.")
        finally:
            t.close()
        return

    print(f"\n=== JOINT LATENCY TEST (mode={args.mode}) ===")
    if args.mode == 'stop':
        print("Swing the limp joint into a hard stop. Each stop -> one LED flash.")
    else:
        print("Flick the limp joint sharply from rest. Each flick -> one LED flash.")
    print("Film at 400 Hz, then count frames from the event to the LED. Ctrl-C to finish.\n")

    written = 0
    f = open(args.csv, 'w')
    f.write("trial,t_event_ns,t_led_ns,pos_rad,peak_speed,recv_to_led_us\n")
    f.flush()
    try:
        while True:
            time.sleep(0.1)
            with t.trial_lock:
                new = t.trials[written:]
            for tr in new:
                f.write(f"{tr['n']},{tr['t_event_ns']},{tr['t_led_ns']},"
                        f"{tr['pos_rad']:.5f},{tr['peak_speed']:.4f},"
                        f"{tr['recv_to_led_us']:.2f}\n")
            if new:
                f.flush()
                written += len(new)
    except KeyboardInterrupt:
        pass
    finally:
        f.flush()
        f.close()
        n = max(1, t.sample_count)
        print(f"\n\n{written} trials -> {args.csv}")
        print(f"poll dt mean/max: {t.loop_dt_sum_ns / n / 1000.0:.1f}/"
              f"{t.loop_dt_max_ns / 1000.0:.1f} us  (one poll period quantizes the result)")
        t.close()


if __name__ == '__main__':
    main()
