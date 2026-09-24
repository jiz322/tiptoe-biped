# CAN2SPI bring-up — MCP2518FD module wiring + Raspberry Pi setup

Companion to `BASELINE_RUNBOOK.md` Phase B. Covers: physically wiring
the MCP2518FD SPI CAN module to the Pi, the one-time OS config, the
per-boot bring-up (`setup_can0.sh`), and the verification ladder before
`can_baseline_bench.py --adapter can2spi` is worth running.

The module is an SPI **CAN FD** controller + transceiver, but we run it
as classic CAN 2.0 at 1 Mbps — that is what the motors speak
(`motor_baud=1` in the firmware dumps). No FD frames anywhere.

---

## 1. Identify the pins on the module

Generic MCP2518FD boards vary in silkscreen but carry the same signals.
On this board: a 12-pin header, a 2-pin header near the screw terminal,
and a 2-pin header near the 12-pin row.

**The 2-pin header near the screw terminal** is almost certainly the
**120 Ω termination jumper** (silk `120R` / `R` / `TERM` / `J1`) — it
switches an onboard 120 Ω resistor across CANH–CANL. See §3.

**The 2-pin header near the 12-pin row** — read the silk:
- `INT0` / `INT1` (or `GPIO0`/`GPIO1`): extra MCP2518FD interrupt pins →
  leave unconnected (the kernel driver only needs the main `INT`).
- `5V` / `VIN` (with the 12-pin row carrying `3V3`): alternate supply
  input → see the supply rule below.
- `120R`: then THIS is the termination jumper and the pair near the
  screw terminal is a CANH/CANL duplicate.

**Signal-by-signal wiring** (silkscreen name → Pi 40-pin header):

| module silk (any of)   | signal          | Pi pin | Pi name        |
|------------------------|-----------------|--------|----------------|
| `VCC` / `3V3` / `VDD`  | supply (see rule)| 1     | 3V3            |
| `5V` / `VIN` (if present)| transceiver 5 V| 2      | 5V             |
| `GND` / `VSS`          | ground          | 6      | GND            |
| `SCK` / `SCLK` / `CLK` | SPI clock       | 23     | GPIO11 (SCLK)  |
| `SI` / `SDI` / `MOSI`  | Pi → module data| 19     | GPIO10 (MOSI)  |
| `SO` / `SDO` / `MISO`  | module → Pi data| 21     | GPIO9  (MISO)  |
| `CS` / `nCS` / `SS`    | chip select     | 24     | GPIO8  (CE0)   |
| `INT` / `nINT`         | interrupt       | 22     | GPIO25         |
| `INT0`, `INT1`         | extra IRQs      | —      | leave open¹    |
| `CLKO` / `SOF`         | clock out       | —      | leave open     |
| `STBY` / `S` / `SIL`   | transceiver standby | any GND pin | tie LOW² |

¹ Optionally `INT1`/`GPIO1` (RX interrupt) → Pi pin 18 (GPIO24) with
  `rx_interrupt=24` in the overlay; skip for first bring-up.
² If left floating on some boards the transceiver stays in standby and
  you get TX but no bus traffic. Many boards pull it down already —
  tying it to GND is always safe.

**Supply rule** (this is the one way to damage the Pi — read first):

1. Board has **both** a 5 V and a 3V3 pin → wire both (5 V feeds the
   transceiver, 3V3 feeds the controller/logic). SPI stays at 3.3 V. ✓
2. Board has **one** VCC and an onboard regulator (small 3-pin SOT-23
   marked `662K`, or an `AMS1117-3.3`) → VCC to **5 V**; the regulator
   makes the 3.3 V logic rail internally.
3. Board has **one** VCC and **no** regulator → VCC to **3.3 V**.
   Check the transceiver marking: `SN65HVD230`, `TJA1042T/3`,
   `MCP2542` are fine at 3.3 V. A 5-V-only transceiver
   (`TJA1051T`, `MCP2562FD` with VIO strapped to VDD, `ATA6563`) may be
   marginal at 3.3 V — but *never* feed VCC 5 V on a no-regulator
   board: the MCP2518FD would then drive **5 V into the Pi's MISO** and
   can kill the Pi. When in doubt, start at 3.3 V; wrong-but-safe beats
   fried.

**Crystal**: read the metal can's marking now — `40.000` → 40 MHz
(overlay default), `20.000` → 20 MHz (`oscillator=20000000`). A
mismatch is the classic "interface comes up, zero frames" failure.

Keep the SPI leads short (< 10 cm) and away from the motor phase leads;
20 MHz SPI on flying dupont wires is the next-most-classic failure.

## 2. Screw terminal → motor bus

- `CANH` → bus CANH, `CANL` → bus CANL (twisted pair).
- If the terminal has a `GND` position, connect bus/battery ground —
  the Pi and the motors must share a ground reference; if the Pi is
  powered from the same battery rail this already exists.

## 3. Termination (runbook B1)

The bus needs 120 Ω at **both physical ends** and nowhere else. The
retired CAN2USB adapter terminated its end; the module now takes over
that end → **short the termination jumper**.

Verify with the bus **powered off**: multimeter across CANH–CANL at the
screw terminal, module connected and jumpered → **≈ 60 Ω** (two 120 Ω
in parallel). ≈ 120 Ω → only one termination present; fix before
benchmarking (missing termination shows up as exactly the kind of
error-frame flakiness that would poison the Phase C comparison).

## 4. One-time OS config (Raspberry Pi)

Append to `/boot/firmware/config.txt` (older Pi OS: `/boot/config.txt`):

```ini
dtparam=spi=on
# 40 MHz crystal (overlay default; set oscillator=20000000 for 20.000)
dtoverlay=mcp251xfd,spi0-0,oscillator=40000000,interrupt=25
```

`spi0-0` = SPI0/CE0 (the wiring above); `interrupt=25` = INT on GPIO25.
The mainline `mcp251xfd` kernel driver caps the SPI clock safely from
the oscillator value — no `speed=` needed.

Install userspace tools and reboot:

```bash
sudo apt install can-utils
sudo reboot
```

After reboot, this must show the controller detected:

```bash
dmesg | grep -i mcp251
#  mcp251xfd spi0.0 can0: MCP2518FD rev0.0 ... successfully initialized.
```

If instead you see `Failed to detect MCP251xfd` → SPI wiring/CS problem;
if it probes but later spews `CRC read error` → SPI leads too
long/noisy.

## 5. Per-boot bring-up

```bash
cd ~/big_thigh/control
sudo ./setup_can0.sh          # ip link up, 1 Mbps, restart-ms 100
```

(Once everything works, make it permanent with a systemd unit or
`/etc/network/interfaces.d/can0` — until then the script keeps it
explicit.)

## 6. Verification ladder (before any benchmarking)

Each step must pass before the next is meaningful. Motors on, robot on
stand.

```bash
# 1. Interface state is ERROR-ACTIVE, bitrate 1000000
ip -details link show can0

# 2. Silent listen — bus should be quiet (motors are request-reply)
candump -td can0        # expect nothing; Ctrl-C

# 3. Poke one motor by hand: enable motor 101 (0x65), expect ONE reply
cansend can0 065#FFFFFFFFFFFFFFFC
candump -td can0 -n 1
cansend can0 065#FFFFFFFFFFFFFFFD    # disable it again

# 4. Full stack, surviving zeros (runbook B3 — note --no-zero!)
#    First flip config.py: CAN_TRANSPORT = 'can2spi'
sudo python3 test_motors.py --no-zero
#    → enable + limp replies from all 6, positions at kneel match the
#      pre-swap readings (MechOffset lives in the motor).

# 5. Bus health after a few minutes of test_motors traffic:
ip -details -statistics link show can0    # error counters ≈ 0, no bus-off
```

Then continue with runbook Phase B4/C (fresh firmware dumps, bench,
`--compare`).

## 7. Code map (what changed for the swap)

- `config.py` — `CAN_TRANSPORT` (`'can2usb'` ↔ `'can2spi'`),
  `CAN_CHANNEL`, `CAN_BITRATE`. **The flip is this one line.**
- `robstride_socketcan_driver.py` — `RobStrideSocketCanDriver`
  (raw AF_CAN socket, stdlib only, no python-can dependency) with the
  exact `RobStrideDriver` public API, plus `make_driver()` factory.
- `motor_bridge.py` / `test_motors.py` — construct via `make_driver()`;
  flight stack (`main.py` → MotorBridge) is transport-agnostic already.
  Other bench/diag scripts still hardcode the serial driver
  (`can_timing_probe.py`, `diag_replies.py`, `setzero_crawl.py`, …);
  switch any of them the same one-line way when needed.
- `can_baseline_bench.py` — `SocketCanShim` + `make_shim()` dispatch;
  T0–T2 now run on either transport; JSON meta records
  transport/channel/bitrate.

Driver behavior notes (vs the serial driver):
- `send_batch` writes back-to-back, **no pacing** (`PACING_S = 0.0`) —
  the kernel/bus serializes, not the adapter. T2 @0.0 ms must be ~100 %
  full before this is trusted (runbook Phase C reading).
- The 2.8 ms reply-wait **deadline** is kept for like-for-like
  comparison, but it exits as soon as all 6 replies land — expect
  `wait_ms` ≈ real RTT (< 1 ms), not 2.8 ms.
- Do NOT delete the serial driver's pacing sleeps or raise CONTROL_HZ
  off these numbers — runbook scope guard applies unchanged.
