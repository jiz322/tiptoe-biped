"""
CRSF Reader — Threaded ELRS receiver for /dev/ttyAMA2
"""

import serial
import threading
import time

CRSF_SYNC = 0xC8
CRSF_RC_CHANNELS = 0x16


def crc8_dvb_s2(data):
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0xD5) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def parse_rc_channels(payload):
    """Unpack 16x 11-bit channels from 22 bytes."""
    bits = int.from_bytes(payload, 'little')
    return [(bits >> (11 * i)) & 0x7FF for i in range(16)]


class CRSFReader:
    """Threaded CRSF reader for ELRS receiver."""

    def __init__(self, port='/dev/ttyAMA2', baud=420000):
        self.port = port
        self.baud = baud
        self.channels = [992] * 16
        self.connected = False
        self.last_update = 0.0
        self.frame_count = 0
        self._lock = threading.Lock()
        self._running = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def get_channels(self):
        with self._lock:
            return list(self.channels), self.connected

    def _run(self):
        ser = serial.Serial(self.port, self.baud, timeout=0.01)
        buf = bytearray()

        while self._running:
            data = ser.read(64)
            if not data:
                if time.monotonic() - self.last_update > 0.5:
                    self.connected = False
                continue

            buf.extend(data)

            while len(buf) >= 4:
                try:
                    idx = buf.index(CRSF_SYNC)
                except ValueError:
                    buf.clear()
                    break
                if idx > 0:
                    buf = buf[idx:]

                if len(buf) < 2:
                    break
                frame_len = buf[1]
                total = frame_len + 2

                if frame_len < 2 or frame_len > 62:
                    buf.pop(0)
                    continue
                if len(buf) < total:
                    break

                frame = buf[:total]
                buf = buf[total:]

                crc_data = frame[2:-1]
                if crc8_dvb_s2(crc_data) != frame[-1]:
                    continue

                self.frame_count += 1

                if frame[2] == CRSF_RC_CHANNELS and len(crc_data) >= 23:
                    channels = parse_rc_channels(crc_data[1:23])
                    with self._lock:
                        self.channels = channels
                        self.connected = True
                        self.last_update = time.monotonic()

        ser.close()