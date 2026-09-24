"""
robstride_socketcan_driver.py — 6-motor MIT-mode driver over SocketCAN
(MCP2518FD SPI adapter). Drop-in replacement for RobStrideDriver.

Public API parity with robstride_biped_driver.RobStrideDriver:
    motor_ids, motor_states, lock, get_state(), enable_all(),
    disable_all(), set_zero()/set_zero_all(), send_command(),
    send_batch(commands, return_timing=...), _parse_all_feedback(),
    _encode_mit_frame(), close().
MotorBridge, can_baseline_bench T3/T4, and the flight stack run
unchanged on top of this class. Selection is config-driven: flip
config.CAN_TRANSPORT to 'can2spi' and everything that constructs its
driver through make_driver() below picks it up.

Differences from the CAN2USB serial driver (all deliberate):
  - Frames are 16-byte `struct can_frame` on a raw AF_CAN socket — no
    AT framing, no 921600-baud serial ceiling, no adapter firmware
    turnaround (the 2.93 ms L documented in send_batch there).
  - No inter-frame pacing by default (PACING_S = 0.0). The 1 ms sleeps
    exist because the CAN2USB adapter serializes transactions; SocketCAN
    queues to the kernel and the bus arbitrates. can_baseline_bench T2
    verifies 0-pacing reply integrity before this touches flight code.
  - The reply wait keeps the same 2.8 ms DEADLINE as the serial driver
    but exits as soon as all replies are in — on a healthy bus that is
    the actual RTT, not 2.8 ms. Both knobs are constructor args so the
    bench compares like against like.

Timestamp semantics (bench contract — see TransportShim docstring):
  send  = frame handed to the transport API (socket.send() returns)
  reply = frame readable in Python (recv() returns it)

Bring-up: CAN2SPI_BRINGUP.md (wiring + dtoverlay) and setup_can0.sh
(per-boot `ip link` bring-up). Constructing this class fails with a
clear error if the interface is missing or down.
"""

import errno
import socket
import struct
import time
from threading import Lock

import config as C
from robstride_biped_driver import (
    DEFAULT_MIT_RANGES, MOTOR_MIT_OVERRIDES,
    MAX_POS, MAX_VEL, MAX_TORQUE, MAX_KP, MAX_KD,
    RobStrideDriver,
)

CAN_FRAME_FMT  = "=IB3x8s"          # id(4) dlc(1) pad(3) data(8)
CAN_FRAME_SIZE = struct.calcsize(CAN_FRAME_FMT)     # 16
CAN_ERR_FLAG   = 0x20000000
CAN_RTR_FLAG   = 0x40000000

_ENABLE  = b'\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFC'
_DISABLE = b'\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFD'
_ZERO    = b'\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFE'


def make_driver(motor_ids=None):
    """Config-driven driver factory — the ONE swap point for the transport.

    config.CAN_TRANSPORT:
        'can2usb' → RobStrideDriver on MOTOR_PORT (flight-tested serial)
        'can2spi' → RobStrideSocketCanDriver on CAN_CHANNEL (MCP2518FD)
    """
    ids = list(C.MOTOR_IDS if motor_ids is None else motor_ids)
    transport = getattr(C, 'CAN_TRANSPORT', 'can2usb')
    if transport == 'can2spi':
        return RobStrideSocketCanDriver(
            getattr(C, 'CAN_CHANNEL', 'can0'), ids,
            pacing_s=getattr(C, 'CAN2SPI_PACING_S', 0.0))
    if transport != 'can2usb':
        raise ValueError(f"config.CAN_TRANSPORT={transport!r} — "
                         f"expected 'can2usb' or 'can2spi'")
    return RobStrideDriver(C.MOTOR_PORT, ids, baud=C.MOTOR_BAUD)


class RobStrideSocketCanDriver:
    REPLY_WAIT_S = 0.0028   # same deadline as the serial send_batch; exits early
    PACING_S     = 0.0      # no inter-frame pacing on SocketCAN (T2 verifies)

    def __init__(self, channel='can0', motor_ids=(),
                 pacing_s=None, reply_wait_s=None):
        self.channel = channel
        self.motor_ids = list(motor_ids)
        self.lock = Lock()
        self.tx_enobufs = 0             # kernel TX queue full events (drops)
        self._rx_buf = bytearray()      # frames drained from the kernel,
                                        # not yet consumed by a parser
        if pacing_s is not None:
            self.PACING_S = float(pacing_s)
        if reply_wait_s is not None:
            self.REPLY_WAIT_S = float(reply_wait_s)

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
            self.sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW,
                                      socket.CAN_RAW)
            self.sock.bind((channel,))
            self.sock.setblocking(False)
        except (AttributeError, OSError) as e:
            raise RuntimeError(
                f"[Driver] SocketCAN open failed on '{channel}': {e}\n"
                f"          Interface up? -> sudo ./setup_can0.sh "
                f"(see CAN2SPI_BRINGUP.md)") from e
        # bind() succeeds on an interface that exists but is DOWN (e.g.
        # after a reboot without setup_can0.sh) — send() would then die
        # with ENETDOWN mid-flight. Check IFF_UP now instead.
        try:
            with open(f"/sys/class/net/{channel}/flags") as f:
                if_up = int(f.read().strip(), 16) & 0x1     # IFF_UP
        except OSError:
            if_up = True    # can't read sysfs — let send() report it
        if not if_up:
            self.sock.close()
            raise RuntimeError(
                f"[Driver] '{channel}' exists but is DOWN (per-boot "
                f"bring-up not run?) -> sudo ./setup_can0.sh")
        print(f"[Driver] SocketCAN {channel} opened, motors: {self.motor_ids}")

    def close(self):
        if getattr(self, 'sock', None) is not None:
            try:
                self.disable_all()
            finally:
                self.sock.close()
                self.sock = None
                print("[Driver] SocketCAN closed")

    # ── raw socket helpers ───────────────────────────────────
    # can_raw sockets don't support the FIONREAD ioctl (ENOTTY on modern
    # kernels), so "bytes pending" is implemented by draining the kernel
    # queue into a driver-side buffer and counting that. Timestamp
    # semantics are unchanged: a reply counts as arrived once recv() has
    # returned it to Python.
    def _drain_to_buf(self):
        while True:
            try:
                self._rx_buf += self.sock.recv(CAN_FRAME_SIZE)
            except BlockingIOError:
                break
            except OSError as e:
                if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    break
                raise

    def _rx_bytes_pending(self):
        """Bytes of received-but-unconsumed frames (multiple of 16).
        The SocketCAN analogue of serial in_waiting."""
        self._drain_to_buf()
        return len(self._rx_buf)

    def _read_all(self):
        self._drain_to_buf()
        out = bytes(self._rx_buf)
        self._rx_buf.clear()
        return out

    def reset_input_buffer(self):
        self._drain_to_buf()
        self._rx_buf.clear()

    def _send_raw(self, frame):
        """send() with a bounded ENOBUFS retry — the kernel TX queue can
        momentarily fill; a dropped command shows up as a missing reply."""
        deadline = time.monotonic() + 0.002
        while True:
            try:
                self.sock.send(frame)
                return
            except OSError as e:
                if e.errno != errno.ENOBUFS:
                    raise
                self.tx_enobufs += 1
                if time.monotonic() > deadline:
                    return

    # ── frame packing (bit layout MUST match robstride_biped_driver) ──
    _float_to_uint = staticmethod(RobStrideDriver._float_to_uint)
    _uint_to_float = staticmethod(RobStrideDriver._uint_to_float)

    def _pack_can_frame(self, can_id, data_bytes):
        return struct.pack(CAN_FRAME_FMT, can_id, len(data_bytes),
                           bytes(data_bytes))

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
        return self._pack_can_frame(motor_id, buf)

    # ── enable / disable / zero ──────────────────────────────
    def enable_all(self):
        for mid in self.motor_ids:
            with self.lock:
                self._send_raw(self._pack_can_frame(mid, _ENABLE))
                time.sleep(0.005)
        print(f"[Driver] Motors {self.motor_ids} enabled")

    def disable_all(self):
        for mid in self.motor_ids:
            with self.lock:
                self._send_raw(self._pack_can_frame(mid, _DISABLE))
                time.sleep(0.002)
        print(f"[Driver] Motors {self.motor_ids} disabled")

    def set_zero_all(self):
        self.set_zero(self.motor_ids)

    def set_zero(self, motor_ids):
        for mid in motor_ids:
            with self.lock:
                self._send_raw(self._pack_can_frame(mid, _ZERO))
                time.sleep(0.1)
        print(f"[Driver] Motors {list(motor_ids)} zeroed")

    # ── MIT command ──────────────────────────────────────────
    def send_command(self, motor_id, p_des, v_des, kp, kd, t_ff):
        frame = self._encode_mit_frame(motor_id, p_des, v_des, kp, kd, t_ff)
        with self.lock:
            self.reset_input_buffer()
            self._send_raw(frame)
            deadline = time.monotonic() + 0.003
            while self._rx_bytes_pending() < CAN_FRAME_SIZE:
                if time.monotonic() > deadline:
                    break
            raw = self._read_all()
            if raw:
                self._parse_all_feedback(raw)

    def send_batch(self, commands, return_timing=False):
        """Batch MIT command to all motors, then collect replies.

        Same contract as RobStrideDriver.send_batch (timing-dict keys
        identical), with two transport-level differences:
          - writes are back-to-back (PACING_S = 0.0 default) — the bus,
            not the adapter, serializes; and
          - byte counts are struct-can_frame bytes (16/frame vs 17 on
            the AT serial framing).
        """
        t0 = time.monotonic()
        expected_bytes = len(commands) * CAN_FRAME_SIZE

        with self.lock:
            self.reset_input_buffer()
            for (mid, p, v, kp, kd, tff) in commands:
                self._send_raw(self._encode_mit_frame(mid, p, v, kp, kd, tff))
                if self.PACING_S > 0:
                    time.sleep(self.PACING_S)
            t_write_done = time.monotonic()

            deadline = t_write_done + self.REPLY_WAIT_S
            while self._rx_bytes_pending() < expected_bytes:
                if time.monotonic() > deadline:
                    break
            bytes_received = self._rx_bytes_pending()
            t_wait_done = time.monotonic()

            replies_parsed = 0
            if bytes_received > 0:
                replies_parsed = self._parse_all_feedback(self._read_all())
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
        """Parse ALL reply frames in a raw can_frame buffer.

        Reply data layout is identical to the serial path: data[0] is the
        replying motor id, then pos16/vel12/torque12/temp16. Returns the
        count of successfully parsed reply frames.
        """
        count = 0
        for off in range(0, len(raw_bytes) - CAN_FRAME_SIZE + 1,
                         CAN_FRAME_SIZE):
            can_id, dlc, data = struct.unpack_from(CAN_FRAME_FMT,
                                                   raw_bytes, off)
            if can_id & (CAN_ERR_FLAG | CAN_RTR_FLAG) or dlc < 8:
                continue
            reply_id = data[0]
            if reply_id not in self._ranges:
                continue
            r = self._ranges[reply_id]
            p_int = (data[1] << 8) | data[2]
            v_int = (data[3] << 4) | (data[4] >> 4)
            t_int = ((data[4] & 0xF) << 8) | data[5]
            temp_raw = (data[6] << 8) | data[7]          # °C × 10

            st = self.motor_states[reply_id]
            st['pos'] = self._uint_to_float(
                p_int, r['pos_min'], r['pos_max'], r['pos_bits'])
            st['vel'] = self._uint_to_float(
                v_int, r['vel_min'], r['vel_max'], r['vel_bits'])
            st['torque'] = self._uint_to_float(
                t_int, r['tff_min'], r['tff_max'], r['tff_bits'])
            st['temperature'] = temp_raw / 10.0
            count += 1
        return count

    def get_state(self, motor_id):
        s = self.motor_states[motor_id]
        return s['pos'], s['vel'], s['torque'], s['temperature']
