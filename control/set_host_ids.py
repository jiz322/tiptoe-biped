"""set_host_ids.py — give each motor a DISTINCT MIT reply arbitration id.

Why: in MIT mode every RobStride-protocol motor addresses its feedback
frame to the HOST CAN id (default 0xFD = 253), with the replying motor's
own id in data[0]. All six motors therefore reply on the same
arbitration id, same-id frames cannot arbitrate, and back-to-back
requests collide (runbook Results 2026-08-15: T4 @0 pacing ~0% full,
error-passive 72k). CAN2SPI_PACING_S = 0.5 ms works around it; distinct
per-motor host ids remove the collision domain entirely and license
zero pacing. Full procedure: BASELINE_RUNBOOK.md Phase D.

Mechanism (RobStride MIT protocol, "Command 9 — Modify Host CAN ID"):
    standard frame, arbitration id = motor id,
    data = FF FF FF FF FF FF <new_host_id> 01
Source: RobStride 00 instruction manual (wiki.aifitlab.com mirror),
single-source as of 2026-08-16 — which is why every --set here is
verified empirically (probe reply id before/after) and --revert exists.
Both driver and bench key replies on data[0], never on the arbitration
id, so they work unchanged under any host-id assignment.

Default assignment: host_id = motor_id (reply id == data[0] — self
checking, no new id space). Reply-vs-command same-id collision is not a
concern: a motor's reply ends ~1 ms after its command; the next command
to that motor is a full tick (10 ms) away.

Usage (on the Pi, robot ON STAND, flight code stopped, motors powered;
motors do NOT need to be enabled — limp motors answer MIT frames):
    sudo python3 set_host_ids.py --show              # current reply ids
    sudo python3 set_host_ids.py --set --motor 102   # pilot, one motor
    sudo python3 set_host_ids.py --set               # all six
    sudo python3 set_host_ids.py --revert            # back to 0xFD
After --set: power-cycle motors, re-run --show — persistence across
power cycles is UNVERIFIED (Phase D3). If ids revert on power-up, the
setting must be saved via the vendor upper-computer tool instead.
"""

import argparse
import struct
import sys
import time

import config as C
from robstride_socketcan_driver import (
    CAN_ERR_FLAG, CAN_FRAME_FMT, CAN_FRAME_SIZE, CAN_RTR_FLAG,
    RobStrideSocketCanDriver,
)

FACTORY_HOST_ID = 0xFD
CMD9_TAIL = 0x01          # data[7] of the Modify-Host-CAN-ID frame
PROBE_WINDOW_S = 0.05
SETTLE_S = 0.05


def probe_reply_ids(drv, mid, window_s=PROBE_WINDOW_S):
    """Send one torque-neutral MIT frame (all fields 0, kp=kd=0) and
    return the arbitration ids of every reply whose data[0] == mid."""
    drv.reset_input_buffer()
    drv._send_raw(drv._encode_mit_frame(mid, 0.0, 0.0, 0.0, 0.0, 0.0))
    deadline = time.monotonic() + window_s
    ids = []
    while time.monotonic() < deadline:
        raw = drv._read_all()
        for off in range(0, len(raw) - CAN_FRAME_SIZE + 1, CAN_FRAME_SIZE):
            can_id, dlc, data = struct.unpack_from(CAN_FRAME_FMT, raw, off)
            if can_id & (CAN_ERR_FLAG | CAN_RTR_FLAG) or dlc < 8:
                continue
            if data[0] == mid:
                ids.append(can_id & 0x1FFFFFFF)
        if ids:
            break
        time.sleep(0.001)
    return ids


def send_cmd9(drv, mid, host_id):
    data = b'\xFF' * 6 + bytes([host_id, CMD9_TAIL])
    drv._send_raw(drv._pack_can_frame(mid, data))
    time.sleep(SETTLE_S)


def show(drv, motor_ids):
    print(f"{'motor':>6} {'reply id':>9}  note")
    ok = True
    seen = {}
    for mid in motor_ids:
        ids = probe_reply_ids(drv, mid)
        if not ids:
            print(f"{mid:>6} {'—':>9}  NO REPLY (powered? MIT mode? bus up?)")
            ok = False
            continue
        rid = ids[0]
        note = ''
        if len(set(ids)) > 1:
            note = f"MULTIPLE ids {sorted(set(ids))} — investigate"
            ok = False
        elif rid == FACTORY_HOST_ID:
            note = 'factory default (shared → collision-prone)'
        if rid in seen:
            note += f"  DUPLICATE of motor {seen[rid]}"
            ok = False
        seen.setdefault(rid, mid)
        print(f"{mid:>6}   0x{rid:03X}  {note}")
        time.sleep(0.01)
    distinct = len(seen) == len(motor_ids) and ok
    print(f"\n[{'OK' if distinct else 'NOT OK'}] "
          f"{len(seen)}/{len(motor_ids)} motors on distinct reply ids"
          + ("" if distinct else " — zero pacing is NOT safe"))
    return distinct


def apply_plan(drv, plan, yes):
    print("Host-id plan (MIT Command 9, one motor at a time):")
    for mid, host in plan.items():
        print(f"  motor {mid}: reply id -> 0x{host:03X}")
    if not yes:
        ans = input("Proceed? [y/N] ").strip().lower()
        if ans != 'y':
            print("Aborted, nothing sent.")
            return False

    for mid, host in plan.items():
        before = probe_reply_ids(drv, mid)
        if not before:
            print(f"[FAIL] motor {mid}: no reply to probe — skipping this "
                  f"motor and STOPPING (fix before configuring further).")
            return False
        send_cmd9(drv, mid, host)
        after = probe_reply_ids(drv, mid)
        b = f"0x{before[0]:03X}"
        a = f"0x{after[0]:03X}" if after else "—"
        if after and after[0] == host:
            print(f"[ OK ] motor {mid}: {b} -> {a}")
        else:
            print(f"[FAIL] motor {mid}: {b} -> {a} (wanted 0x{host:03X}). "
                  f"Command 9 may be unsupported on this family — STOPPING. "
                  f"Run --revert, then use the vendor upper-computer tool "
                  f"path (runbook Phase D fallback).")
            return False
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Show/set/revert per-motor MIT reply (host) CAN ids.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--show', action='store_true',
                      help='probe and print each motor\'s reply id')
    mode.add_argument('--set', action='store_true',
                      help='assign distinct host ids (default: = motor id)')
    mode.add_argument('--revert', action='store_true',
                      help=f'set host id back to 0x{FACTORY_HOST_ID:02X} '
                           f'on the selected motors')
    ap.add_argument('--motor', type=int, action='append', default=None,
                    help='limit to this motor id (repeatable); '
                         'default: all of config.MOTOR_IDS')
    ap.add_argument('--host-offset', type=int, default=0,
                    help='host id = motor id + OFFSET (default 0)')
    ap.add_argument('--channel', default=getattr(C, 'CAN_CHANNEL', 'can0'))
    ap.add_argument('--yes', action='store_true',
                    help='skip the interactive confirmation')
    args = ap.parse_args()

    motor_ids = args.motor if args.motor else list(C.MOTOR_IDS)
    unknown = [m for m in motor_ids if m not in C.MOTOR_IDS]
    if unknown:
        sys.exit(f"unknown motor id(s) {unknown} — not in config.MOTOR_IDS")

    if args.set:
        plan = {mid: mid + args.host_offset for mid in motor_ids}
    else:
        plan = {mid: FACTORY_HOST_ID for mid in motor_ids}
    if args.set:
        bad = [h for h in plan.values() if not (1 <= h <= 0xFF)]
        if bad:
            sys.exit(f"host id(s) {bad} out of range 1..255 — fix --host-offset")
        if len(set(plan.values())) != len(plan):
            sys.exit("host ids not unique — fix --host-offset")
        clash = [(m, h) for m, h in plan.items()
                 if h in C.MOTOR_IDS and h != m]
        if clash:
            sys.exit(f"host id(s) {clash} collide with ANOTHER motor's "
                     f"command id — pick a different --host-offset")

    # Driver gives us encode/pack/_send_raw + the rx buffering; pacing here
    # is irrelevant (this tool is strictly one-motor-at-a-time).
    drv = RobStrideSocketCanDriver(args.channel, motor_ids, pacing_s=0.001)
    try:
        if args.show:
            ok = show(drv, motor_ids)
            sys.exit(0 if ok else 1)
        ok = apply_plan(drv, plan, args.yes)
        print()
        show(drv, motor_ids)
        if ok and args.set:
            print("\nNext: power-cycle the motors and re-run --show to test "
                  "persistence (runbook Phase D3) BEFORE touching "
                  "CAN2SPI_PACING_S.")
        sys.exit(0 if ok else 1)
    finally:
        # close() sends disable_all — motors are limp on the stand, benign.
        drv.close()


if __name__ == '__main__':
    main()
