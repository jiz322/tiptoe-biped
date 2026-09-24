"""
robstride_biped_driver.py — 6-motor MIT mode driver for the bipedal robot.

Key changes from the 2-motor balancer driver:
    1. Per-motor MIT encoding ranges — different motor models (e.g. RS01
       vs RS03) have different position, velocity, torque, Kp, Kd ranges.
    2. Motor IDs 101-106 (extended CAN IDs, not 1-2).
    3. Bulk read_all / write_all for timing-critical loops.

PROFILER HOOKS (added for profile_balance.py — backwards compatible):
    - send_batch() accepts an optional return_timing flag. When True it
      returns a dict with phase wall-times in ms and reply byte/frame
      counts. Default is None (legacy behaviour).
    - _parse_all_feedback() now returns the count of successfully parsed
      reply frames (legacy callers ignore the return value).

IMPORTANT: You MUST set the correct encoding ranges for each motor model.
Check your RobStride motor datasheet page ~50 for "MIT Control Mode"
encoding table. Typical values:


"""

import serial
import struct
import time
from threading import Lock


# ════════════════════════════════════════════════════════════
# MIT protocol encoding ranges — per motor MODEL
# ════════════════════════════════════════════════════════════
# RS01 and EL05 differ in BOTH velocity and torque range; position,
# Kp, Kd and all bit widths are identical, so those stay in the default
# and only vel + tff are overridden per motor.
#   RS01:  vel ±44 rad/s, tff ±17 N·m   (RS01 manual §6 MIT + §4.4 #defines)
#   EL05:  vel ±50 rad/s, tff ±6  N·m   (EL05 MIT "指令3")
# NOTE: every configured motor is specified explicitly below — do NOT
# lean on the default for vel/tff. Silently inheriting vel from the
# default is exactly what made RS01 read ±50 before.

DEFAULT_MIT_RANGES = {
    'pos_min': -12.57,  'pos_max': 12.57,   'pos_bits': 16,
    'vel_min': -44.0,   'vel_max': 44.0,    'vel_bits': 12,   # fallback = RS01
    'kp_min':  0.0,     'kp_max':  500.0,   'kp_bits':  12,
    'kd_min':  0.0,     'kd_max':  5.0,     'kd_bits':  12,
    'tff_min': -17.0,   'tff_max': 17.0,    'tff_bits': 12,   # fallback = RS01
}

_RS01 = {'vel_min': -44.0, 'vel_max': 44.0, 'tff_min': -17.0, 'tff_max': 17.0}
_EL05 = {'vel_min': -50.0, 'vel_max': 50.0, 'tff_min':  -6.0, 'tff_max':  6.0}

MOTOR_MIT_OVERRIDES = {
    # config.py: 101=L_hip_pitch, 102=L_knee, 103=L_roll,
    #            104=R_hip_pitch, 105=R_knee, 106=R_roll
    101: dict(_RS01),   # L_hip_pitch RS01
    102: dict(_RS01),   # L_knee      RS01
    103: dict(_EL05),   # L_roll      EL05
    104: dict(_RS01),   # R_hip_pitch RS01
    105: dict(_RS01),   # R_knee      RS01
    106: dict(_EL05),   # R_roll      EL05
}

# ════════════════════════════════════════════════════════════
# Software safety limits (applied before encoding)
# ════════════════════════════════════════════════════════════

MAX_POS    = 12.57
MAX_VEL    = 30.0
MAX_TORQUE = 17.0
MAX_KP     = 200.0
MAX_KD     = 5.0


class RobStrideDriver:
    def __init__(self, port, motor_ids, baud=921600):
        self.port = port
        self.motor_ids = list(motor_ids)
        self.ser = None
        self.lock = Lock()

        self.motor_states = {
            mid: {'pos': 0.0, 'vel': 0.0, 'torque': 0.0, 'temperature': 0.0}
            for mid in self.motor_ids
        }

        self._ranges = {}
        for mid in self.motor_ids:
            r = dict(DEFAULT_MIT_RANGES)
            if mid in MOTOR_MIT_OVERRIDES:
                r.update(MOTOR_MIT_OVERRIDES[mid])
            self._ranges[mid] = r

        try:
            self.ser = serial.Serial(port, baud, timeout=0.002)
            print(f"[Driver] Serial {port} @ {baud} opened, "
                  f"motors: {self.motor_ids}")
        except Exception as e:
            print(f"[Driver] Serial open failed: {e}")
            raise

    def close(self):
        if self.ser and self.ser.is_open:
            self.disable_all()
            self.ser.close()
            print("[Driver] Serial closed")

    # ── frame packing ────────────────────────────────────────
    def _pack_std_can_frame(self, can_id, data_bytes):
        header = b'\x41\x54'
        packed_id_val = (can_id << 21)
        id_bytes = struct.pack('>I', packed_id_val)
        dlc = b'\x08'
        footer = b'\x0d\x0a'
        return header + id_bytes + dlc + data_bytes + footer

    @staticmethod
    def _float_to_uint(x, x_min, x_max, bits):
        x = max(x_min, min(x_max, x))
        span = x_max - x_min
        return int((x - x_min) * ((1 << bits) - 1) / span)

    @staticmethod
    def _uint_to_float(x_int, x_min, x_max, bits):
        span = x_max - x_min
        return (x_int * span / ((1 << bits) - 1)) + x_min

    # ── enable / disable / zero ──────────────────────────────
    def enable_all(self):
        payload = b'\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFC'
        for mid in self.motor_ids:
            frame = self._pack_std_can_frame(mid, payload)
            with self.lock:
                self.ser.write(frame)
                time.sleep(0.005)
        print(f"[Driver] Motors {self.motor_ids} enabled")

    def disable_all(self):
        payload = b'\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFD'
        for mid in self.motor_ids:
            frame = self._pack_std_can_frame(mid, payload)
            with self.lock:
                self.ser.write(frame)
                time.sleep(0.002)
        print(f"[Driver] Motors {self.motor_ids} disabled")

    def set_zero_all(self):
        self.set_zero(self.motor_ids)

    def set_zero(self, motor_ids):
        payload = b'\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFE'
        for mid in motor_ids:
            frame = self._pack_std_can_frame(mid, payload)
            with self.lock:
                self.ser.write(frame)
                time.sleep(0.1)
        print(f"[Driver] Motors {list(motor_ids)} zeroed")

    # ── MIT command ──────────────────────────────────────────
    def _encode_mit_frame(self, motor_id, p_des, v_des, kp, kd, t_ff):
        p_des = max(-MAX_POS, min(MAX_POS, p_des))
        v_des = max(-MAX_VEL, min(MAX_VEL, v_des))
        kp    = max(0,        min(MAX_KP,  kp))
        kd    = max(0,        min(MAX_KD,  kd))
        t_ff  = max(-MAX_TORQUE, min(MAX_TORQUE, t_ff))

        r = self._ranges[motor_id]
        p_int  = self._float_to_uint(p_des, r['pos_min'], r['pos_max'], r['pos_bits'])
        v_int  = self._float_to_uint(v_des, r['vel_min'], r['vel_max'], r['vel_bits'])
        kp_int = self._float_to_uint(kp,    r['kp_min'],  r['kp_max'],  r['kp_bits'])
        kd_int = self._float_to_uint(kd,    r['kd_min'],  r['kd_max'],  r['kd_bits'])
        t_int  = self._float_to_uint(t_ff,  r['tff_min'], r['tff_max'], r['tff_bits'])

        buf = bytearray(8)
        buf[0] = p_int >> 8
        buf[1] = p_int & 0xFF
        buf[2] = v_int >> 4
        buf[3] = ((v_int & 0xF) << 4) | (kp_int >> 8)
        buf[4] = kp_int & 0xFF
        buf[5] = kd_int >> 4
        buf[6] = ((kd_int & 0xF) << 4) | (t_int >> 8)
        buf[7] = t_int & 0xFF

        return self._pack_std_can_frame(motor_id, buf)

    def send_command(self, motor_id, p_des, v_des, kp, kd, t_ff):
        frame = self._encode_mit_frame(motor_id, p_des, v_des, kp, kd, t_ff)
        with self.lock:
            self.ser.reset_input_buffer()
            self.ser.write(frame)
            deadline = time.monotonic() + 0.003
            while self.ser.in_waiting < 17:        # one full reply frame
                if time.monotonic() > deadline:
                    break
                # No sleep — just yield. CPU spin for ~1 ms is fine.
            if self.ser.in_waiting > 0:
                raw_resp = self.ser.read_all()
                self._parse_all_feedback(raw_resp)

    def send_batch(self, commands, return_timing=False):
        """Batch MIT command to all motors, then collect replies.

        MEASURED MODEL OF THE ADAPTER (can_timing_probe.py, 2026-08-12,
        two sessions). Only L and the sleep-sweep fit reproduced across
        sessions; treat everything pipelining-related as session-dependent.

          - Per-transaction round-trip latency L = 2.93 ms, DETERMINISTIC
            (p10-p95 spread 0.02 ms, identical for all 6 motors, RS01 and
            EL05 alike). A lone command on a quiet bus still takes 2.93 ms
            => fixed firmware/USB turnaround inside the adapter, not
            queueing. Physics floor is ~0.7 ms; the adapter adds ~2.2 ms.
            Zero variance = one serialized transaction at a time; the
            adapter CANNOT be pipelined around this floor.
          - Test B fit: completion = 6*sleep + L, exact at 1.0 and 1.2 ms
            sleeps. Writes are serialized by the sleeps; the last reply
            lands L after the last write. 0.8 ms sleeps gave 92/100 full
            ticks, 1.0 ms gives 100/100 — hence the 1.0 ms below.
          - Test F (windowed pipelining) DID NOT REPRODUCE: session 1
            gave W=4 -> 100/100 @ 5.92 ms; session 2 gave W=4 -> 96/100
            (worst motor 96%) and W=6 -> 39/100 vs 17/100. W=4 sits on
            the adapter's concurrency cliff and its reliability varies
            between sessions. DO NOT ship windowed pipelining on this
            adapter; a 4% silent-drop floor is not a production mode.
          - Test C (reply-triggered pacing) is NOT a fix either: 0/200
            full ticks, 12.03 ms total, one motor at 0% — worse than
            this code. (The probe's printed "H1 confirmed" text under
            Test C is a pre-written decision-guide branch, not a result.)
          - The ONLY validated upgrade for THIS adapter — Test G: keep
            the 1.0 ms pacing, DELETE the 2.8 ms reply wait, drain
            replies at the next tick. 299/299 full reply sets @ 100 Hz,
            tick busy 9.3 -> 6.44 ms. Integration caveat: the drain must
            run at the START of the control tick (before read_state
            feeds the obs), NOT inside send_batch, or joint states
            silently become 2 ticks old.
          - Vendor (RobStride support, 2026-08-13): this CAN2USB module is
            their host-PC DEBUG tool; their 0.5 ms/motor rate figure was
            measured with their own controller on the CAN bus directly.
            Deployment-grade latency needs direct CAN (SocketCAN HAT /
            MCP2518FD SPI); the 2.93 ms turnaround is the hard floor here.

        The legacy constants below (1.0 ms sleep + 2.8 ms deadline
        = ~9.3 ms/tick) are the optimum for THIS code shape and remain
        the flight-tested default.
        """
        t0 = time.monotonic()

        expected_bytes = len(commands) * 17
        
        with self.lock:
            self.ser.reset_input_buffer()
            for (mid, p, v, kp, kd, tff) in commands:
                frame = self._encode_mit_frame(mid, p, v, kp, kd, tff)
                self.ser.write(frame)
                time.sleep(0.0010)   # 1 ms inter-frame pacing for USB-serial adapter
            t_write_done = time.monotonic()
            
            deadline = time.monotonic() + 0.0028
            while self.ser.in_waiting < expected_bytes:
                if time.monotonic() > deadline:
                    break
            bytes_received = self.ser.in_waiting
            t_wait_done = time.monotonic()
            
            replies_parsed = 0
            if bytes_received > 0:
                replies_parsed = self._parse_all_feedback(self.ser.read_all())
        t_end = time.monotonic()
        
        if return_timing:
            return {
                'encode_ms': 0.0,
                'write_ms': (t_write_done - t0) * 1000.0,
                'wait_ms':  (t_wait_done - t_write_done) * 1000.0,
                'parse_ms': (t_end - t_wait_done) * 1000.0,
                'total_ms': (t_end - t0) * 1000.0,
                'bytes_received': bytes_received,
                'expected_bytes': expected_bytes,
                'replies_parsed': replies_parsed,
            }
        return None

    def _parse_all_feedback(self, raw_bytes):
        """Parse ALL reply frames in a buffer (may contain multiple motors).

        Returns the count of successfully parsed reply frames.
        """
        HEADER = b'\x41\x54'
        FRAME_LEN = 17   # AT(2) + ID(4) + DLC(1) + DATA(8) + END(2)
        offset = 0
        count = 0
        while offset < len(raw_bytes) - FRAME_LEN + 1:
            idx = raw_bytes.find(HEADER, offset)
            if idx == -1 or idx + FRAME_LEN > len(raw_bytes):
                break

            data_start = idx + 7
            data = raw_bytes[data_start:data_start + 8]
            if len(data) < 8:                       # was < 6; need bytes 6-7 for temp
                offset = idx + 1
                continue

            reply_id = data[0]
            if reply_id in self._ranges:
                r = self._ranges[reply_id]
                try:
                    p_int = (data[1] << 8) | data[2]
                    v_int = (data[3] << 4) | (data[4] >> 4)
                    t_int = ((data[4] & 0xF) << 8) | data[5]
                    temp_raw = (data[6] << 8) | data[7]      # NEW — °C × 10

                    self.motor_states[reply_id]['pos'] = self._uint_to_float(
                        p_int, r['pos_min'], r['pos_max'], r['pos_bits'])
                    self.motor_states[reply_id]['vel'] = self._uint_to_float(
                        v_int, r['vel_min'], r['vel_max'], r['vel_bits'])
                    self.motor_states[reply_id]['torque'] = self._uint_to_float(
                        t_int, r['tff_min'], r['tff_max'], r['tff_bits'])
                    self.motor_states[reply_id]['temperature'] = temp_raw / 10.0  # NEW
                    count += 1
                except Exception:
                    pass

            offset = idx + FRAME_LEN
        return count

    def _parse_feedback(self, motor_id, raw_bytes):
        """Legacy single-motor parser (kept for compat)."""
        self._parse_all_feedback(raw_bytes)

    def get_state(self, motor_id):
        s = self.motor_states[motor_id]
        return s['pos'], s['vel'], s['torque'], s['temperature']