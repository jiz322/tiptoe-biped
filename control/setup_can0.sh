#!/usr/bin/env bash
# setup_can0.sh — per-boot bring-up of the MCP2518FD SocketCAN interface
# for the RobStride motor bus. One-time OS config (dtoverlay) lives in
# CAN2SPI_BRINGUP.md. Idempotent; run with sudo.
set -euo pipefail

CH="${1:-can0}"
BITRATE=1000000      # must match motor firmware motor_baud=1 (1 Mbps)

if ! ip link show "$CH" &>/dev/null; then
    echo "ERROR: $CH does not exist."
    echo "  dtoverlay not loaded / SPI wiring / oscillator mismatch —"
    echo "  see CAN2SPI_BRINGUP.md, then check: dmesg | grep -i mcp251"
    exit 1
fi

ip link set "$CH" down 2>/dev/null || true
# restart-ms 100: auto-recover from bus-off instead of freezing the loop
ip link set "$CH" up type can bitrate "$BITRATE" restart-ms 100
ip link set "$CH" txqueuelen 64

ip -details link show "$CH" | head -n 4
echo "OK: $CH up @ ${BITRATE} bps"

# ── Host-id preflight (runbook Phase D7) ─────────────────────────────
# CAN2SPI_PACING_S=0 relies on every motor replying on its OWN
# arbitration id. Two motors sharing an id (e.g. a replacement motor
# still at factory 0xFD next to another) re-creates reply collisions,
# so duplicates are a hard stop. Motors unpowered is only a warning —
# re-check before flight.
DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "$DIR/set_host_ids.py" ]]; then
    out="$(cd "$DIR" && python3 set_host_ids.py --show 2>&1)" || true
    if grep -q "NO REPLY" <<<"$out"; then
        echo "WARN: motor(s) not answering (bus power off?) — before"
        echo "      flight, re-run: sudo python3 set_host_ids.py --show"
    elif grep -q "NOT OK" <<<"$out"; then
        echo "$out"
        echo "ERROR: shared/duplicate reply ids on the bus — zero pacing"
        echo "       (CAN2SPI_PACING_S=0) is UNSAFE. Fix: set_host_ids.py"
        echo "       --set, or revert config to CAN2SPI_PACING_S=0.0005."
        exit 2
    else
        echo "OK: distinct per-motor reply ids confirmed"
    fi
fi
