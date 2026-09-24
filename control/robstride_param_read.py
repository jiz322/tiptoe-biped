"""
robstride_param_read.py — read RobStride firmware parameters directly over
the private protocol (29-bit extended CAN frames). READ-ONLY, ZERO-MOTION.

Replaces the abandoned --locked saturation bench: instead of pressing 17 Nm
against a fixture to infer the firmware current limit empirically, just ask
the firmware. Registers of interest (RobStride manual sec. 4.x, confirmed
against the sirwart/robstride SDK):

    0x7018 limit_cur    (float, A)   <- THE current limit
    0x700B limit_torque (float, Nm)  <- torque limit
    0x7017 limit_spd    (float)      <- velocity limit
    0x701A iqf          (float, A)   <- live filtered q-axis current
    0x701C VBUS         (float, V)
    0x3022 faultSta     (u32)        <- fault status register
    ... plus run_mode, mechPos/mechVel, loop gains.

SAFETY: by default this script sends ONLY communication-type 0x11 (single
parameter READ) frames — never enables motors, never writes parameters.
The one exception is the explicit, opt-in `--set-protocol` command (typed
confirmation required), which implements the protocol round-trip below.

═══ WHY READS FAIL IN MIT MODE, AND THE SUPPORTED WAY AROUND ═══
Our motors run in MIT compatibility mode (11-bit frames, FF..FC enable magic
— see robstride_biped_driver.py) and the firmware ignores private-protocol
extended frames there. But RobStride's official SampleProgram (Robstride01.cpp,
RobStride_Motor_MIT_MotorModeSet) shows the protocol switch has an MIT-NATIVE
form: a STANDARD 11-bit frame `FF FF FF FF FF FF <proto> FD` — same magic
family as the enable/disable frames we already use, so the MIT firmware
definitely hears it. proto: 00=private, 01=CANopen, 02=MIT. The change takes
effect ONLY AFTER A POWER CYCLE and persists.

Round-trip to read the registers (bench context: robot hanging or on stand,
motors disabled, flight code NOT running):
  1. `--set-protocol private --motors 101`   (standard-frame magic, heard in MIT mode)
  2. power-cycle the motor bus
  3. plain read (`--motors 101 --all`) now works; `--probe` type-0 ping too
  4. `--set-protocol mit --motors 101`       (extended type-25 frame, heard in private mode)
  5. power-cycle again; re-zero motors per the normal session procedure
Do one motor at a time (limits blast radius). Known residual risk: step 3/4
need the adapter to pass extended frames — if reads are STILL silent in
private mode, the adapter is the blocker and that motor waits for a rescue
via the RobStride upper-computer tool (annoying, not damaging: CAN ID and
calibration are untouched).

Adapter framing (same 17-byte 'AT' frames as the driver):
    'AT' + ID32(4, big-endian) + DLC(1) + DATA(8) + '\\r\\n'
    standard 11-bit: ID32 = id << 21          (driver's _pack_std_can_frame)
    extended 29-bit: ID32 = (id << 3) | 0x04  (bit 2 = IDE flag)

Transport (2026-08-17): follows config.CAN_TRANSPORT. On 'can2spi' a
SocketCAN shim speaks the same 17-byte 'AT' framing to the rest of this
script and raw can_frames on the wire, so every mode (--probe, --watch,
--set-protocol, plain reads) works unchanged on either adapter. The
old "USB-CAN adapter drops extended frames" failure mode does not
exist on SocketCAN — the MCP2518FD passes 29-bit frames natively.
Host-id note: since BASELINE_RUNBOOK Phase D each motor's MIT host id
equals its own motor id; reads fall back through 0xFD -> 0x00 -> the
motor's own id automatically.

Usage:
    python3 robstride_param_read.py --dry-run          # print frames, no port
    python3 robstride_param_read.py                    # read all 6 motors
    python3 robstride_param_read.py --motors 101,102 --all
    python3 robstride_param_read.py --watch iqf --motors 102 --secs 10
"""

import argparse
import struct
import sys
import time

import config as C

FRAME_LEN = 17
READ_TYPE = 0x11        # single parameter read
WRITE_TYPE = 0x12       # single parameter write (RAM — reverts on power cycle)
FAULT_TYPE = 0x15       # fault feedback frame (type 21), decode if seen
FEEDBACK_TYPE = 0x02    # private-protocol feedback (fault bits in arb ID)

# (index, ctype) — ctype: 'f' float32, 'u8', 'u32'; all little-endian in bytes 4-7
PARAMS = {
    'run_mode':      (0x7005, 'u8'),
    'iq_ref':        (0x7006, 'f'),
    'spd_ref':       (0x700A, 'f'),
    'limit_torque':  (0x700B, 'f'),
    'cur_kp':        (0x7010, 'f'),
    'cur_ki':        (0x7011, 'f'),
    'cur_filt_gain': (0x7014, 'f'),
    'limit_spd':     (0x7017, 'f'),
    'limit_cur':     (0x7018, 'f'),
    'mechPos':       (0x7019, 'f'),
    'iqf':           (0x701A, 'f'),
    'mechVel':       (0x701B, 'f'),
    'VBUS':          (0x701C, 'f'),
    'loc_kp':        (0x701E, 'f'),
    'spd_kp':        (0x701F, 'f'),
    'spd_ki':        (0x7020, 'f'),
    # 0x7021 VERIFIED on hardware 2026-08-17: reads 0.1 on all four RS01,
    # 0.5 on both EL05 — matches the fw_dumps CSVs. loc_ref_filt_gain has
    # NO 0x7xxx runtime mirror (full scan 0x7000-0x7037 on an EL05 found
    # no 0.04; 0x7022 reads 20.0 RS01 / 10.0 EL05 = something else, and
    # config-table 0x2xxx addresses don't answer type-0x11 reads). The
    # 0.04 is reachable only through the vendor upper-computer tool.
    'spd_filt_gain': (0x7021, 'f'),
    'faultSta':      (0x3022, 'u32'),
}
DEFAULT_SET = ['run_mode', 'limit_cur', 'limit_torque', 'limit_spd',
               'spd_filt_gain', 'cur_filt_gain',
               'VBUS', 'iqf', 'mechPos', 'mechVel', 'faultSta']
UNITS = {'limit_cur': 'A', 'iqf': 'A', 'limit_torque': 'Nm', 'VBUS': 'V',
         'mechPos': 'rad', 'mechVel': 'rad/s', 'limit_spd': 'rad/s'}

# faultSta bits named in the manual (sec 3.3.7); others printed raw
FAULT_BITS = {16: 'A-phase current sampling overcurrent',
              14: 'motor stall/overload protection',
              7: 'encoder uncalibrated'}


def pack_ext_frame(ext29, data8):
    return b'AT' + struct.pack('>I', (ext29 << 3) | 0x04) + b'\x08' + data8 + b'\r\n'


def pack_std_frame(std_id, data8):
    # same convention as robstride_biped_driver._pack_std_can_frame
    return b'AT' + struct.pack('>I', std_id << 21) + b'\x08' + data8 + b'\r\n'


PROTO = {'private': 0x00, 'canopen': 0x01, 'mit': 0x02}


def mit_protocol_switch_frame(motor_id, proto):
    """Protocol switch sent FROM MIT mode — standard 11-bit magic frame per
    RobStride SampleProgram RobStride_Motor_MIT_MotorModeSet. NOTE: differs
    from the DISABLE magic (FF*7,FD) only in byte 6."""
    return pack_std_frame(motor_id, b'\xFF' * 6 + bytes([proto, 0xFD]))


def private_protocol_switch_frame(host_id, motor_id, proto):
    """Protocol switch sent FROM private mode — extended type-25 (0x19) frame
    with the 01..06 magic interlock; byte 7 is don't-care (sample uses 0x08)."""
    ext29 = (0x19 << 24) | ((host_id & 0xFFFF) << 8) | (motor_id & 0xFF)
    return pack_ext_frame(ext29, bytes([1, 2, 3, 4, 5, 6, proto, 0x08]))


def device_id_ping(host_id, motor_id):
    """Type-0 'get device ID' — answered by any motor in private mode
    (reply: type 0, motor id in bits 8-23, data = 64-bit MCU UID)."""
    ext29 = ((host_id & 0xFFFF) << 8) | (motor_id & 0xFF)
    return pack_ext_frame(ext29, b'\x00' * 8)


def read_request(host_id, motor_id, index):
    ext29 = (READ_TYPE << 24) | ((host_id & 0xFFFF) << 8) | (motor_id & 0xFF)
    data = struct.pack('<H', index) + b'\x00' * 6
    return pack_ext_frame(ext29, data)


# M2.2 firmware A/B (replant runbook): writable filter gains only. Bounds
# from the fw_dumps CSVs. Type-0x12 writes hit RAM — a power cycle reverts
# them. NOTE (2026-08-17): MIT-mode firmware ignores extended frames even
# on CAN2SPI, and the protocol switch back to MIT itself requires a power
# cycle — so a RAM write CANNOT survive into MIT-mode testing. Persistent
# changes (the actual M2.2 A/B) go through the vendor upper-computer tool;
# this write path remains useful for live private-mode experiments only.
WRITABLE = {
    'spd_filt_gain':     (0.0, 1.0),
    'cur_filt_gain':     (0.0, 1.0),
}


def write_request(host_id, motor_id, index, value):
    ext29 = (WRITE_TYPE << 24) | ((host_id & 0xFFFF) << 8) | (motor_id & 0xFF)
    data = struct.pack('<H', index) + b'\x00\x00' + struct.pack('<f', value)
    return pack_ext_frame(ext29, data)


def parse_frames(buf):
    """Walk the serial buffer for 17-byte 'AT' frames; return (id32, data8)."""
    out, i = [], 0
    while i + FRAME_LEN <= len(buf):
        if (buf[i:i+2] == b'AT' and buf[i+6] == 0x08
                and buf[i+15:i+17] == b'\r\n'):
            id32 = struct.unpack('>I', buf[i+2:i+6])[0]
            out.append((id32, bytes(buf[i+7:i+15])))
            i += FRAME_LEN
        else:
            i += 1
    return out


def decode_value(raw4, ctype):
    if ctype == 'f':
        return struct.unpack('<f', raw4)[0]
    if ctype == 'u8':
        return raw4[0]
    return struct.unpack('<I', raw4)[0]


CAN_EFF_FLAG   = 0x80000000
CAN_ERR_FLAG   = 0x20000000
CAN_RTR_FLAG   = 0x40000000
CAN_FRAME_FMT  = "=IB3x8s"          # id(4) dlc(1) pad(3) data(8)
CAN_FRAME_SIZE = struct.calcsize(CAN_FRAME_FMT)


class SocketCanAT:
    """pyserial-shaped SocketCAN transport speaking this script's 17-byte
    'AT' framing: write() unpacks AT frames onto the raw socket, read()
    re-encodes received can_frames as AT bytes so parse_frames() and every
    caller work unchanged on either adapter."""

    def __init__(self, channel='can0'):
        import socket as _s
        self.sock = _s.socket(_s.AF_CAN, _s.SOCK_RAW, _s.CAN_RAW)
        try:
            self.sock.bind((channel,))
        except OSError as e:
            raise SystemExit(
                f"cannot bind {channel!r} ({e}) — interface up? "
                f"per-boot: sudo ./setup_can0.sh")
        self.sock.setblocking(False)
        self.channel = channel

    def write(self, at_bytes):
        for id32, data in parse_frames(at_bytes):
            if id32 & 0x04:
                can_id = (id32 >> 3) | CAN_EFF_FLAG
            else:
                can_id = id32 >> 21
            self.sock.send(struct.pack(CAN_FRAME_FMT, can_id, 8, data))

    def read(self, n=256):
        out = b''
        while True:
            try:
                raw = self.sock.recv(CAN_FRAME_SIZE)
            except BlockingIOError:
                break
            if len(raw) < CAN_FRAME_SIZE:
                break
            can_id, dlc, data = struct.unpack(CAN_FRAME_FMT, raw)
            if can_id & (CAN_ERR_FLAG | CAN_RTR_FLAG):
                continue
            if can_id & CAN_EFF_FLAG:
                id32 = ((can_id & 0x1FFFFFFF) << 3) | 0x04
            else:
                id32 = (can_id & 0x7FF) << 21
            out += b'AT' + struct.pack('>I', id32) + b'\x08' + data + b'\r\n'
        if not out:
            time.sleep(0.002)       # mirror pyserial's timeout=0.002 pacing
        return out

    def reset_input_buffer(self):
        while True:
            try:
                self.sock.recv(CAN_FRAME_SIZE)
            except BlockingIOError:
                return

    def close(self):
        self.sock.close()


def open_transport():
    if getattr(C, 'CAN_TRANSPORT', 'can2usb') == 'can2spi':
        t = SocketCanAT(getattr(C, 'CAN_CHANNEL', 'can0'))
        print(f"[param_read] SocketCAN {t.channel} (CAN2SPI)")
        return t
    import serial
    ser = serial.Serial(C.MOTOR_PORT, C.MOTOR_BAUD, timeout=0.002)
    print(f"[param_read] {C.MOTOR_PORT} @ {C.MOTOR_BAUD}")
    return ser


def decode_fault_sta(v):
    if v == 0:
        return "0 (no fault)"
    named = [f"bit{b}={n}" for b, n in FAULT_BITS.items() if v & (1 << b)]
    other = v & ~sum(1 << b for b in FAULT_BITS)
    if other:
        named.append(f"raw 0x{v:08X}")
    return f"0x{v:08X}: " + ("; ".join(named) if named else "unknown bits")


def scan_replies(frames, motor_id, index, ctype):
    """Find a type-0x11 reply for (motor, index); also surface any fault
    frames that happen to arrive."""
    value = None
    for id32, data in frames:
        if not (id32 & 0x04):
            continue                      # standard frame (MIT traffic) — skip
        ext = id32 >> 3
        typ = (ext >> 24) & 0x1F
        if typ == READ_TYPE:
            rid = (ext >> 8) & 0xFF       # reply: motor id in bits 8-15
            ridx = struct.unpack('<H', data[0:2])[0]
            if rid == motor_id and ridx == index:
                value = decode_value(data[4:8], ctype)
        elif typ == FAULT_TYPE:
            fv, wv = struct.unpack('<II', data)
            print(f"    [!] fault frame from motor {(ext >> 8) & 0xFF}: "
                  f"fault=0x{fv:08X} warn=0x{wv:08X}")
        elif typ == FEEDBACK_TYPE:
            fbits = (ext >> 16) & 0x3F
            if fbits:
                print(f"    [!] feedback fault bits from motor "
                      f"{(ext >> 8) & 0xFF}: 0x{fbits:02X}")
    return value


def read_param(ser, host_id, motor_id, name, tries=3, wait_s=0.03):
    index, ctype = PARAMS[name]
    req = read_request(host_id, motor_id, index)
    for _ in range(tries):
        ser.reset_input_buffer()
        ser.write(req)
        deadline = time.monotonic() + wait_s
        buf = b''
        while time.monotonic() < deadline:
            buf += ser.read(256)
            v = scan_replies(parse_frames(buf), motor_id, index, ctype)
            if v is not None:
                return v
            time.sleep(0.002)
    return None


def fmt(name, v):
    if v is None:
        return "no reply"
    if name == 'faultSta':
        return decode_fault_sta(v)
    if name == 'run_mode':
        return f"{v} ({'operation/MIT' if v == 0 else ['?', 'position', 'speed', 'current'][v] if v <= 3 else '?'})"
    u = UNITS.get(name, '')
    return f"{v:+.3f} {u}".strip() if isinstance(v, float) else str(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--motors', default=None,
                    help='comma-separated motor IDs (default: all from config)')
    ap.add_argument('--params', default=None,
                    help=f'comma-separated names (default: {",".join(DEFAULT_SET)})')
    ap.add_argument('--all', action='store_true', help='read every known param')
    ap.add_argument('--host-id', type=lambda s: int(s, 0), default=0xFD,
                    help='host CAN ID in the request (default 0xFD; 0x00 is '
                         'auto-tried on silence)')
    ap.add_argument('--watch', default=None, choices=sorted(PARAMS),
                    help='stream one param continuously (e.g. iqf)')
    ap.add_argument('--secs', type=float, default=5.0, help='watch duration')
    ap.add_argument('--csv', default=None, help='watch mode: also log to CSV')
    ap.add_argument('--probe', action='store_true',
                    help='send a type-0 device-ID ping per motor and dump raw '
                         'reply bytes (diagnoses extended-frame plumbing; '
                         'only answered in private mode)')
    ap.add_argument('--set-protocol', default=None, choices=['private', 'mit'],
                    help='switch motor wire protocol (typed confirmation; '
                         'takes effect after POWER CYCLE, persists). '
                         '"private" is sent as the MIT-mode standard magic '
                         'frame; "mit" as the private-mode type-25 frame. '
                         'Requires explicit --motors.')
    ap.add_argument('--write', default=None, metavar='NAME=VALUE',
                    help='write one param (RAM only — power cycle reverts; '
                         'M2.2 spd_filt_gain A/B). Allowed: '
                         + ', '.join(sorted(WRITABLE))
                         + '. Typed confirmation; requires explicit --motors; '
                         'motor must be in private mode (same round-trip as '
                         'reads); read-back verifies.')
    ap.add_argument('--dry-run', action='store_true',
                    help='print the exact request frames in hex; no serial port')
    args = ap.parse_args()

    motors = ([int(m) for m in args.motors.split(',')] if args.motors
              else list(C.MOTOR_IDS))
    names = (sorted(PARAMS) if args.all
             else args.params.split(',') if args.params else DEFAULT_SET)
    for n in names:
        if n not in PARAMS:
            sys.exit(f"unknown param '{n}' (known: {', '.join(sorted(PARAMS))})")

    if args.set_protocol and not args.motors:
        sys.exit("--set-protocol requires explicit --motors (no default-all; "
                 "do one motor at a time)")

    wname = wval = None
    if args.write:
        if '=' not in args.write:
            sys.exit("--write expects NAME=VALUE (e.g. spd_filt_gain=0.3)")
        wname, s = args.write.split('=', 1)
        if wname not in WRITABLE:
            sys.exit(f"'{wname}' is not writable here "
                     f"(allowed: {', '.join(sorted(WRITABLE))})")
        try:
            wval = float(s)
        except ValueError:
            sys.exit(f"--write value {s!r} is not a number")
        lo, hi = WRITABLE[wname]
        if not lo <= wval <= hi:
            sys.exit(f"{wname}={wval} outside firmware bounds [{lo}, {hi}]")
        if not args.motors or len(motors) != 1:
            sys.exit("--write requires exactly one motor via --motors "
                     "(one at a time limits blast radius)")

    if args.dry_run:
        print(f"host_id=0x{args.host_id:02X}  motors={motors}")
        if args.set_protocol:
            proto = PROTO[args.set_protocol]
            for mid in motors:
                if args.set_protocol == 'private':
                    fr = mit_protocol_switch_frame(mid, proto)
                    print(f"  motor {mid}  ->private (MIT-mode std frame): {fr.hex(' ')}")
                else:
                    fr = private_protocol_switch_frame(args.host_id, mid, proto)
                    print(f"  motor {mid}  ->mit (private-mode type-25): {fr.hex(' ')}")
        if args.probe:
            for mid in motors:
                print(f"  motor {mid}  type-0 ping: "
                      f"{device_id_ping(args.host_id, mid).hex(' ')}")
        if args.write:
            idx, _ = PARAMS[wname]
            fr = write_request(args.host_id, motors[0], idx, wval)
            print(f"  motor {motors[0]}  write {wname} (0x{idx:04X}) <- "
                  f"{wval}: {fr.hex(' ')}")
        if not args.set_protocol and not args.probe and not args.write:
            for mid in motors:
                for n in names:
                    idx, ct = PARAMS[n]
                    print(f"  motor {mid}  {n:<14s} (0x{idx:04X},{ct}): "
                          f"{read_request(args.host_id, mid, idx).hex(' ')}")
        print("--dry-run: no serial port touched.")
        return

    ser = open_transport()
    try:
        if args.set_protocol:
            proto = PROTO[args.set_protocol]
            print(f"\n*** PROTOCOL SWITCH -> {args.set_protocol.upper()} for "
                  f"motors {motors} ***\n"
                  f"  - takes effect after POWER CYCLE, and PERSISTS\n"
                  f"  - motors will NOT respond to the "
                  f"{'MIT flight stack' if args.set_protocol == 'private' else 'private protocol'} "
                  f"after the power cycle\n"
                  f"  - do NOT run balance/flight code until switched back and re-zeroed\n"
                  f"  - frames to send:")
            frames = []
            for mid in motors:
                fr = (mit_protocol_switch_frame(mid, proto)
                      if args.set_protocol == 'private'
                      else private_protocol_switch_frame(args.host_id, mid, proto))
                frames.append(fr)
                print(f"      motor {mid}: {fr.hex(' ')}")
            if input("Type SWITCH to send: ").strip() != "SWITCH":
                print("not confirmed — nothing sent.")
                return
            for fr in frames:
                ser.write(fr)
                time.sleep(0.02)
                ser.write(fr)          # once more for luck; idempotent
                time.sleep(0.02)
            print("\nsent. NOW POWER-CYCLE THE MOTOR BUS.")
            if args.set_protocol == 'private':
                print("then verify with:  --probe --motors "
                      + ",".join(str(m) for m in motors)
                      + "\nthen read with:    --motors "
                      + ",".join(str(m) for m in motors) + " --all")
            else:
                print("then verify MIT is back (driver enable + limp command "
                      "answers), and RE-ZERO per the normal session procedure.")
            return

        if args.write:
            mid = motors[0]
            host = args.host_id
            before = read_param(ser, host, mid, wname)
            if before is None:
                for alt in (0x00, mid):
                    if alt == host:
                        continue
                    before = read_param(ser, alt, mid, wname)
                    if before is not None:
                        host = alt
                        break
            if before is None:
                sys.exit(f"motor {mid}: no reply reading {wname} — the motor "
                         "must be in PRIVATE mode (protocol round-trip in the "
                         "docstring) and the 0x7021/0x7022 index verified "
                         "against the fw_dumps CSV first")
            print(f"\n*** RAM WRITE motor {mid}: {wname} {before:+.4f} -> "
                  f"{wval:+.4f}  (volatile — power cycle reverts) ***")
            if input("Type WRITE to send: ").strip() != "WRITE":
                print("not confirmed — nothing sent.")
                return
            idx, _ = PARAMS[wname]
            ser.write(write_request(host, mid, idx, wval))
            time.sleep(0.05)
            after = read_param(ser, host, mid, wname)
            ok = after is not None and abs(after - wval) < 1e-4
            print("read-back: " + ("no reply" if after is None
                                   else f"{wname} = {after:+.4f}")
                  + ("  [OK]" if ok else "  [MISMATCH — write not applied?]"))
            return

        if args.probe:
            for mid in motors:
                ser.reset_input_buffer()
                ser.write(device_id_ping(args.host_id, mid))
                deadline = time.monotonic() + 0.1
                buf = b''
                while time.monotonic() < deadline:
                    buf += ser.read(256)
                print(f"motor {mid}: raw reply "
                      f"{'(none)' if not buf else buf.hex(' ')}")
                for id32, data in parse_frames(buf):
                    if id32 & 0x04:
                        ext = id32 >> 3
                        typ = (ext >> 24) & 0x1F
                        print(f"    ext frame type=0x{typ:02X} "
                              f"id-field=0x{ext & 0xFFFFFF:06X} data={data.hex(' ')}"
                              + ("  <- device-ID reply (private mode alive)"
                                 if typ == 0 else ""))
            return

        if args.watch:
            mid = motors[0]
            idx, ct = PARAMS[args.watch]
            csv_f = open(args.csv, 'w') if args.csv else None
            if csv_f:
                csv_f.write("t,value\n")
            print(f"watching {args.watch} on motor {mid} for {args.secs:.0f}s "
                  f"(Ctrl-C to stop)")
            t0 = time.monotonic()
            n_ok = n_miss = 0
            try:
                while time.monotonic() - t0 < args.secs:
                    v = read_param(ser, args.host_id, mid, args.watch,
                                   tries=1, wait_s=0.02)
                    t = time.monotonic() - t0
                    if v is None:
                        n_miss += 1
                    else:
                        n_ok += 1
                        print(f"\r  t={t:6.2f}s  {args.watch}={fmt(args.watch, v):<24s}"
                              f"(ok {n_ok}, miss {n_miss})", end='', flush=True)
                        if csv_f:
                            csv_f.write(f"{t:.4f},{v}\n")
            except KeyboardInterrupt:
                pass
            print(f"\ndone: {n_ok} reads, {n_miss} misses")
            if csv_f:
                csv_f.close()
            return

        any_reply = False
        for mid in motors:
            print(f"\nmotor {mid}:")
            host = args.host_id
            for i, n in enumerate(names):
                v = read_param(ser, host, mid, n)
                if v is None and i == 0:
                    # some fw expect host 0; since Phase D the MIT host id
                    # is the motor's own id — try both before giving up
                    for alt in (0x00, mid):
                        if alt == host:
                            continue
                        v2 = read_param(ser, alt, mid, n)
                        if v2 is not None:
                            host = alt
                            v = v2
                            print(f"    (motor answers host_id 0x{alt:02X}, "
                                  f"switching)")
                            break
                any_reply = any_reply or (v is not None)
                print(f"    {n:<14s} {fmt(n, v)}")
        if not any_reply:
            print("\n[!] No motor answered any private-protocol read.\n"
                  "    In MIT mode that is expected — use the protocol "
                  "round-trip (see docstring):\n"
                  "      --set-protocol private --motors <id>  ->  power-cycle "
                  "->  read  ->  --set-protocol mit  ->  power-cycle + re-zero\n"
                  "    If reads are silent even in PRIVATE mode, the USB-CAN "
                  "adapter is dropping extended\n"
                  "    frames — rescue that motor with the RobStride "
                  "upper-computer tool.")
    finally:
        ser.close()


if __name__ == '__main__':
    main()
