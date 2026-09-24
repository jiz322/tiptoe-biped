#!/usr/bin/env python3
"""
ELRS / CRSF Receiver Test
==========================
Reads CRSF frames from the ELRS receiver and prints channel values live.
Move your sticks and switches to verify everything works.

Usage:
    python test_elrs.py                         # default /dev/ttyAMA2
    python test_elrs.py --port /dev/ttyAMA2     # explicit port
    python test_elrs.py --raw                   # show raw hex frames too

CRSF channel range: 172 (min) — 992 (center) — 1811 (max)
"""

import serial
import time
import argparse
import sys

CRSF_SYNC = 0xC8
CRSF_RC_CHANNELS = 0x16

# Other useful frame types for debugging
FRAME_NAMES = {
    0x02: "GPS",
    0x07: "VARIO",
    0x08: "BATTERY",
    0x09: "BARO_ALT",
    0x0A: "HEARTBEAT",
    0x14: "LINK_STATS",
    0x16: "RC_CHANNELS",
    0x1C: "SUBSET_RC",
    0x28: "DEVICE_PING",
    0x29: "DEVICE_INFO",
}


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


def channel_to_percent(val):
    """172-1811 → -100% to +100%, 992 = 0%."""
    return (val - 992) / (1811 - 992) * 100.0


def channel_bar(val, width=20):
    """Simple ASCII bar for a channel value."""
    pct = (val - 172) / (1811 - 172)  # 0.0 to 1.0
    pct = max(0.0, min(1.0, pct))
    filled = int(pct * width)
    return '█' * filled + '░' * (width - filled)


def main():
    parser = argparse.ArgumentParser(description="ELRS / CRSF Receiver Test")
    parser.add_argument("--port", type=str, default="/dev/ttyAMA2")
    parser.add_argument("--baud", type=int, default=420000)
    parser.add_argument("--raw", action="store_true", help="Print raw hex frames")
    parser.add_argument("--all-frames", action="store_true", help="Show non-RC frame types too")
    args = parser.parse_args()

    print(f"{'=' * 65}")
    print(f"  ELRS / CRSF Receiver Test")
    print(f"{'=' * 65}")
    print(f"  Port: {args.port}  Baud: {args.baud}")
    print(f"  Expecting CRSF sync byte 0xC8")
    print(f"  Channel range: 172 (min) — 992 (center) — 1811 (max)")
    print(f"  Ctrl-C to quit")
    print(f"{'=' * 65}\n")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except Exception as e:
        print(f"[ERROR] Cannot open {args.port}: {e}")
        sys.exit(1)

    print(f"[OK] Port {args.port} opened. Waiting for CRSF data...\n")

    buf = bytearray()
    frame_count = 0
    rc_count = 0
    other_types = {}
    t_start = time.monotonic()
    last_print = 0

    try:
        while True:
            data = ser.read(128)
            if not data:
                elapsed = time.monotonic() - t_start
                print(f"\r  [{elapsed:.1f}s] No data received... "
                      f"(check wiring, binding, UART overlay)", end='', flush=True)
                continue

            buf.extend(data)

            while len(buf) >= 4:
                # Find sync
                try:
                    idx = buf.index(CRSF_SYNC)
                except ValueError:
                    if args.raw:
                        print(f"  [JUNK] {buf.hex()}")
                    buf.clear()
                    break

                if idx > 0:
                    if args.raw:
                        print(f"  [SKIP] {buf[:idx].hex()}")
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

                # CRC
                crc_data = frame[2:-1]
                expected_crc = crc8_dvb_s2(crc_data)
                if expected_crc != frame[-1]:
                    if args.raw:
                        print(f"  [CRC FAIL] {frame.hex()} "
                              f"(calc={expected_crc:#04x} got={frame[-1]:#04x})")
                    continue

                frame_count += 1
                frame_type = frame[2]
                frame_name = FRAME_NAMES.get(frame_type, f"0x{frame_type:02X}")

                if args.raw:
                    print(f"  [FRAME #{frame_count}] type={frame_name} "
                          f"len={frame_len} hex={frame.hex()}")

                if frame_type == CRSF_RC_CHANNELS and len(crc_data) >= 23:
                    rc_count += 1
                    channels = parse_rc_channels(crc_data[1:23])

                    now = time.monotonic()
                    if now - last_print >= 0.1:  # 10Hz display
                        last_print = now
                        elapsed = now - t_start

                        # Clear and redraw
                        sys.stdout.write('\033[2J\033[H')  # clear screen
                        print(f"{'=' * 65}")
                        print(f"  ELRS CRSF — LIVE  |  t={elapsed:.1f}s  "
                              f"frames={frame_count}  rc={rc_count}")
                        print(f"{'=' * 65}")
                        print()

                        # Show first 8 channels (most useful)
                        for i in range(8):
                            val = channels[i]
                            pct = channel_to_percent(val)
                            bar = channel_bar(val)
                            label = ["Roll/A", "Pitch/E", "Throt/T",
                                     "Yaw/R", "Aux1/5", "Aux2/6",
                                     "Aux3/7", "Aux4/8"][i]
                            print(f"  CH{i+1:2d} ({label:7s}): "
                                  f"{val:4d}  {pct:+6.1f}%  {bar}")

                        # Show remaining channels compact
                        if any(channels[i] != 992 for i in range(8, 16)):
                            print()
                            row = "  CH9-16: " + "  ".join(
                                f"{channels[i]:4d}" for i in range(8, 16))
                            print(row)

                        print(f"\n  {'─' * 55}")
                        print(f"  Link rate: ~{rc_count / max(elapsed, 0.1):.0f} Hz")

                        if other_types:
                            types_str = ", ".join(
                                f"{v}×{FRAME_NAMES.get(k, f'0x{k:02X}')}"
                                for k, v in sorted(other_types.items()))
                            print(f"  Other frames: {types_str}")

                        print(f"\n  Move sticks to verify channels respond!")
                        print(f"  Ctrl-C to quit")

                elif args.all_frames:
                    other_types[frame_type] = other_types.get(frame_type, 0) + 1

                else:
                    other_types[frame_type] = other_types.get(frame_type, 0) + 1

    except KeyboardInterrupt:
        print(f"\n\n[Done] Received {frame_count} frames ({rc_count} RC) "
              f"in {time.monotonic() - t_start:.1f}s")
    finally:
        ser.close()


if __name__ == "__main__":
    main()