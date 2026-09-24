#!/usr/bin/env python3
"""
IMU Absolute Observation Latency Test  (tap / impact method)
============================================================
Measures end-to-end latency:  physical motion  ->  sample readable in software.

Setup
-----
- IMU and an LED (BCM GPIO18) rigidly co-mounted on ONE platform, both inside
  the 400 Hz camera's field of view. (LED in series with a ~330 ohm resistor to GND.)
- Give the platform a sharp impact. A pendulum/rod swinging into a hard stop is
  the most repeatable; a firm tap with a hard object also works. The impact is a
  large acceleration spike that BOTH the IMU and the camera witness at the same
  instant.
- This script reads the IMU at full rate and, the instant the acceleration spike
  crosses a threshold, drives GPIO18 HIGH.
- Film at 400 Hz.  In post:  count frames between the moment of contact (motion
  stops / reverses) and the moment the LED lights:

      delta_frames * 2.5 ms  =  full loop latency
        = IMU internal (sense + fuse) + UART + kernel + this script + GPIO write

  Subtract the separately-measured GPIO write latency (loopback test) to isolate
  the IMU pipeline. The CSV this writes is for DIAGNOSTICS: it confirms the
  software added negligible latency (recv->led), so the camera-measured delay can
  be attributed to the IMU.

Note: keep the IMU output set identical to production (here ACC+GYRO+EULER), so
the per-frame UART transmission time matches what you'll actually run.

Run
---
    sudo python3 imu_latency_test.py --calibrate     # tap a few times, find threshold
    sudo python3 imu_latency_test.py --threshold 35  # run the test
(sudo is needed for SCHED_FIFO real-time priority.)
"""

import os
import sys
import gc
import time
import struct
import array
import fcntl
import argparse
import threading

import serial
import lgpio


# ------------------------------ Config -------------------------------
IMU_PORT  = "/dev/ttyAMA0"
BAUD_RATE = 460800

LED_PIN   = 18      # BCM numbering (physical pin 12)
GPIO_CHIP = 0       # Pi 5: usually 0 on current OS. If the LED never lights,
                    # try GPIO_CHIP = 4 (some kernels map the header there).

RT_PRIORITY = 80    # SCHED_FIFO priority for the reader thread
RT_CPU      = 3     # pin reader to this core (isolate it via isolcpus=3 in cmdline)

GRAVITY = 9.80665   # m/s^2, baseline |acc| at rest (independent of orientation)

# Detection (tune with --calibrate). Deviation of |acc| from gravity, in m/s^2.
IMPACT_THRESHOLD = 30.0   # ~3 g of deviation; only a hard impact should exceed it
LED_ON_SECONDS   = 0.15   # how long LED stays lit (>= 1 frame; longer eases review)
COOLDOWN_SECONDS = 0.5    # ignore everything right after a trigger (bounce guard)
QUIET_DEV        = 3.0    # re-arm only once |acc| dev stays below this...
QUIET_SECONDS    = 0.3    # ...for this long (lets the platform settle)

# --------------------------- YIS protocol ----------------------------
HEADER   = bytes([0x59, 0x53])
ID_ACC   = 0x10
ID_GYRO  = 0x20
ID_EULER = 0x40
SCALE    = 0.000001

# detection state machine
ARMED, COOLDOWN, WAIT_QUIET = 0, 1, 2


def set_realtime(priority, cpu):
    """Apply SCHED_FIFO + CPU affinity to the *calling thread*."""
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


def set_low_latency_uart(fd):
    """Set ASYNC_LOW_LATENCY via TIOCSSERIAL (trims the tty flip-buffer delay).
    ttyAMA0 is a native PL011 UART (no FTDI 16 ms latency_timer), so the gain is
    small, but it's harmless. Non-fatal if unsupported."""
    TIOCGSERIAL = 0x541E
    TIOCSSERIAL = 0x541F
    ASYNC_LOW_LATENCY = 0x2000
    try:
        buf = array.array('i', [0] * 32)
        fcntl.ioctl(fd, TIOCGSERIAL, buf)
        buf[4] |= ASYNC_LOW_LATENCY          # flags is field index 4 of serial_struct
        fcntl.ioctl(fd, TIOCSSERIAL, buf)
    except OSError as e:
        print(f"[warn] low-latency UART not set: {e}")


class LatencyTester:
    def __init__(self, calibrate=False, threshold=IMPACT_THRESHOLD):
        self.calibrate = calibrate
        self.threshold = threshold
        self.ser = None
        self.h = None
        self.running = False
        self.reader = None

        self.state = WAIT_QUIET          # start by waiting for the platform to be still
        self.led_on = False
        self.t_led_ns = 0
        self.t_trigger_quiet = time.monotonic_ns()
        self.prev_dev = 0.0
        self.cur_trial = None

        self.trials = []                 # reader appends; main thread writes CSV
        self.trial_lock = threading.Lock()
        self.cal_peak = 0.0              # rolling peak for --calibrate
        self.frame_count = 0

    # ----- protocol: build + send config commands (same as the driver) -----
    @staticmethod
    def _checksum(data):
        ck1 = ck2 = 0
        for b in data:
            ck1 = (ck1 + b) & 0xFF
            ck2 = (ck2 + ck1) & 0xFF
        return ck1, ck2

    def _send(self, data_class, operator, data_bytes):
        n = len(data_bytes)
        byte1 = ((n & 0x1F) << 3) | (operator & 0x07)
        byte2 = (n >> 5) & 0xFF
        payload = bytes([data_class, byte1, byte2]) + bytes(data_bytes)
        ck1, ck2 = self._checksum(payload)
        self.ser.write(HEADER + payload + bytes([ck1, ck2]))
        time.sleep(0.2)

    def _configure_imu(self):
        # VRU mode (6-axis, no mag), gyro-bias init, ACC+GYRO+EULER, 400 Hz.
        # Output set kept identical to production so latency matches reality.
        self._send(0x4D, 0x01, [0x02, 0x02])             # algo mode = VRU
        time.sleep(0.3)
        self._send(0x4D, 0x01, [0x50, 0x01])             # gyro bias self-collect (stay still)
        time.sleep(1.0)
        mask = (1 << 7) | (1 << 6) | (1 << 4)            # ACC | GYRO | EULER = 0x00D0
        self._send(0x04, 0x01, [mask & 0xFF, (mask >> 8) & 0xFF])
        time.sleep(0.3)
        self._send(0x03, 0x01, [0x0A])                   # 400 Hz
        time.sleep(0.3)

    # ------------------------------ open ------------------------------
    def open(self):
        self.ser = serial.Serial(IMU_PORT, BAUD_RATE, timeout=0.05)
        set_low_latency_uart(self.ser.fileno())
        time.sleep(0.5)
        print(f"[imu] {IMU_PORT} @ {BAUD_RATE} open; configuring (stay still ~1 s)...")
        self._configure_imu()
        self.ser.reset_input_buffer()

        try:
            self.h = lgpio.gpiochip_open(GPIO_CHIP)
        except lgpio.error as e:
            print(f"[err] gpiochip {GPIO_CHIP} open failed: {e}\n"
                  f"      On some Pi 5 kernels the header is gpiochip4 - set GPIO_CHIP=4.")
            raise
        lgpio.gpio_claim_output(self.h, LED_PIN, 0)      # start LOW
        print(f"[gpio] LED on BCM{LED_PIN} (chip {GPIO_CHIP}) ready, LOW")

        self.running = True
        self.reader = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader.start()
        return True

    # ----------------------- read + detect loop -----------------------
    def _reader_loop(self):
        set_realtime(RT_PRIORITY, RT_CPU)
        gc.disable()                     # no GC pauses in the hot path
        buf = b''
        while self.running:
            n = self.ser.in_waiting
            chunk = self.ser.read(n if n else 1)   # blocks up to timeout if idle
            if chunk:
                buf += chunk
                while len(buf) >= 7:
                    i = buf.find(HEADER)
                    if i < 0:
                        buf = buf[-1:]
                        break
                    if i > 0:
                        buf = buf[i:]
                        continue
                    if len(buf) < 5:
                        break
                    dlen = buf[4]
                    flen = 5 + dlen + 2
                    if len(buf) < flen:
                        break
                    frame = buf[:flen]
                    buf = buf[flen:]
                    ck1, ck2 = self._checksum(frame[2:5 + dlen])
                    if ck1 != frame[-2] or ck2 != frame[-1]:
                        continue
                    t_recv = time.monotonic_ns()      # <-- arrival timestamp
                    self.frame_count += 1
                    self._handle_payload(frame[5:5 + dlen], t_recv)
            # non-blocking housekeeping: LED off + re-arm timing
            self._service_timers()

    def _handle_payload(self, payload, t_recv):
        # pull ACC out of the TLV
        off = 0
        ax = ay = az = None
        while off + 2 <= len(payload):
            did = payload[off]
            dl = payload[off + 1]
            off += 2
            if off + dl > len(payload):
                break
            if did == ID_ACC and dl >= 12:
                ax, ay, az = struct.unpack('<iii', payload[off:off + 12])
                ax *= SCALE; ay *= SCALE; az *= SCALE
            off += dl
        if ax is None:
            return

        mag = (ax * ax + ay * ay + az * az) ** 0.5
        dev = abs(mag - GRAVITY)

        if self.calibrate:
            if dev > self.cal_peak:
                self.cal_peak = dev
            self.prev_dev = dev
            return

        if self.state == ARMED:
            # rising-edge crossing -> single fire at the onset of the spike
            if dev > self.threshold and self.prev_dev <= self.threshold:
                # ===== CRITICAL: write the LED edge FIRST, timestamp immediately =====
                lgpio.gpio_write(self.h, LED_PIN, 1)
                t_led = time.monotonic_ns()
                self.led_on = True
                self.t_led_ns = t_led
                self.state = COOLDOWN
                self._record(t_recv, t_led, mag, dev)
        elif self.state == COOLDOWN:
            if self.cur_trial is not None and dev > self.cur_trial['peak_dev']:
                self.cur_trial['peak_dev'] = dev     # track the impact peak for the log
        elif self.state == WAIT_QUIET:
            now = time.monotonic_ns()
            if dev > QUIET_DEV:
                self.t_trigger_quiet = now           # reset quiet timer (still moving)
            elif now - self.t_trigger_quiet >= QUIET_SECONDS * 1e9:
                self.state = ARMED
                print("[armed] ready for the next tap")

        self.prev_dev = dev

    def _service_timers(self):
        if not self.led_on and self.state != COOLDOWN:
            return
        now = time.monotonic_ns()
        if self.led_on and now - self.t_led_ns >= LED_ON_SECONDS * 1e9:
            lgpio.gpio_write(self.h, LED_PIN, 0)
            self.led_on = False
        if self.state == COOLDOWN and now - self.t_led_ns >= COOLDOWN_SECONDS * 1e9:
            self.state = WAIT_QUIET
            self.t_trigger_quiet = now

    def _record(self, t_recv, t_led, mag, dev):
        trial = {
            'n': len(self.trials) + 1,
            't_recv_ns': t_recv,
            't_led_ns': t_led,
            'acc_mag': mag,
            'onset_dev': dev,
            'peak_dev': dev,
            'recv_to_led_us': (t_led - t_recv) / 1000.0,
        }
        with self.trial_lock:
            self.trials.append(trial)
        self.cur_trial = trial
        print(f"[TRIAL {trial['n']:>3}] LED fired  |  |acc|={mag:6.2f}  "
              f"dev={dev:6.2f}  software recv->led={trial['recv_to_led_us']:6.1f} us")

    # ----------------------------- close ------------------------------
    def close(self):
        self.running = False
        if self.reader is not None:
            self.reader.join(timeout=2)
        if self.h is not None:
            try:
                lgpio.gpio_write(self.h, LED_PIN, 0)
                lgpio.gpiochip_close(self.h)
            except Exception:
                pass
        if self.ser and self.ser.is_open:
            try:
                self._send(0x03, 0x01, [0x08])    # back to 100 Hz
            except Exception:
                pass
            self.ser.close()
        print("[closed]")


def main():
    ap = argparse.ArgumentParser(description="IMU absolute observation latency test")
    ap.add_argument('--calibrate', action='store_true',
                    help="print peak |acc| deviation per tap; never fires the LED")
    ap.add_argument('--threshold', type=float, default=IMPACT_THRESHOLD,
                    help="impact detection threshold, m/s^2 deviation from gravity")
    ap.add_argument('--csv', default='latency_trials.csv', help="output CSV path")
    args = ap.parse_args()

    t = LatencyTester(calibrate=args.calibrate, threshold=args.threshold)
    if not t.open():
        sys.exit(1)
    time.sleep(0.5)

    if args.calibrate:
        print("\n=== CALIBRATION ===  tap the platform a few times.")
        print("Set --threshold to ~50-60% of a clean impact peak. Ctrl+C to stop.\n")
        try:
            while True:
                time.sleep(0.25)
                print(f"  peak dev since last: {t.cal_peak:7.2f} m/s^2   "
                      f"(frames: {t.frame_count})      ", end='\r')
                t.cal_peak = 0.0
        except KeyboardInterrupt:
            print("\ndone.")
        finally:
            t.close()
        return

    print("\n=== LATENCY TEST ===")
    print(f"threshold = {args.threshold} m/s^2 dev. Tap, wait ~1 s, repeat.")
    print("Each impact -> one LED flash. Film at 400 Hz. Ctrl+C to finish.\n")

    written = 0
    f = open(args.csv, 'w')
    f.write("trial,t_recv_ns,t_led_ns,acc_mag,onset_dev,peak_dev,recv_to_led_us\n")
    f.flush()
    try:
        while True:
            time.sleep(0.1)
            with t.trial_lock:
                new = t.trials[written:]
            for tr in new:
                f.write(f"{tr['n']},{tr['t_recv_ns']},{tr['t_led_ns']},"
                        f"{tr['acc_mag']:.4f},{tr['onset_dev']:.4f},"
                        f"{tr['peak_dev']:.4f},{tr['recv_to_led_us']:.2f}\n")
            if new:
                f.flush()
                written += len(new)
    except KeyboardInterrupt:
        pass
    finally:
        f.flush()
        f.close()
        print(f"\n\n{written} trials written to {args.csv}")
        t.close()


if __name__ == "__main__":
    main()
