# Bill of materials — TipToe v2.7

Prices are what the owner actually paid, checked 2026-09-24, Chinese domestic
supply (Taobao / JD), in CNY. Western equivalents cost roughly 1.3–1.6× on
AliExpress. Lines marked **TBD** were not in the checked list yet; they are
small, but a build needs them.

Motor ids and roles are from `control/config.py`; part names are from the
v2.7 CAD file list.

## Priced lines

| # | Item | Qty | Unit ¥ | Line ¥ | Role / notes |
|---|---|---|---|---|---|
| 1 | RobStride RS01 | 4 | 599 | 2396 | Hip pitch ×2, knee ×2 (ids 101/102/104/105). 17 Nm peak at the motor; the knee joint sees 23.2 Nm through the 22:30 belt. |
| 2 | RobStride EL05 | 2 | 299 | 598 | Hip roll ×2 (ids 103/106). 6 Nm peak. |
| 3 | Raspberry Pi 5 | 1 | 724 | 724 | Runs the 100 Hz control loop in Python. RAM size: **TBD** (record from the order). |
| 4 | microSD card | 1 | 289 | 289 | Capacity / class: **TBD**. |
| 5 | WHEELTEC H30 IMU | 1 | 254 | 254 | UART at 460800 baud, 400 Hz. Read by `control/imu_driver_6_400hz.py` (the adapter comments call the protocol "Yesense"; same frame format). |
| 6 | MCP2518FD SPI CAN module | 1 | 40 | 40 | CAN FD chip run as classic CAN 2.0 at 1 Mbps over SocketCAN. Wiring, crystal check (20 vs 40 MHz) and OS setup in `control/CAN2SPI_BRINGUP.md`. |
| 7 | 4S 6500 mAh semi-solid LiPo | 2 | 381 | 762 | Bus voltage for all six motors. Two packs: one flies, one charges (**confirm**: if only one is on the robot at a time, the build cost is ¥381 lower). |
| 8 | Matek PDB-HEX | 1 | 98 | 98 | Power distribution; its BEC feeds the Pi and IMU 5 V rail (**confirm** which output). |
| 9 | RA5008 crossed-roller bearing (hip roll) | 2 | 140 | 280 | One per hip roll joint; carries the leg load so the EL05 hip roll motor is not loaded radially. |
| 10 | HTD 5M timing belt | 2 | 30 | 60 | One per knee, motor in the hip body → knee. Length / width: read from the v2.7 assembly. The 10-16 redesign moves to a 20 mm belt with 33:45 pulleys; that is not this BOM (see `ASSEMBLY.md` §3). |
| 11 | 3D-print filament | — | — | <100 | Whole robot prints from `all.3mf` (v2.7). Material, grams and print time: **TBD**. |
| 12 | Screws, nuts, heat-set inserts | — | — | <100 | Sizes: **TBD** (M3/M4 per CAD hole callouts). |
| | **Subtotal (priced)** | | | **≈ 5700** | Upper bound; lines 11–12 rounded up to 100 each. ¥5501 before them. |

## Unpriced lines — TBD

| # | Item | Qty | Notes |
|---|---|---|---|
| 13 | ELRS receiver | 1 | CRSF on `/dev/ttyAMA2` at 420000 baud (`control/crsf_reader.py`). Arms the loop via `main.py`. Transmitter is not counted as robot cost. |
| 14 | Pulleys 22T and 30T, HTD 5M | 2 + 2 | `pulley_5m_22*.STL` and `pulley_5m_30.STL` exist in CAD, so they may be printed rather than bought. **Confirm** which was used on the flying robot (the observed tooth skip is a load-gated belt/pulley effect). The 10-16 redesign replaces this pair with 33:45. |
| 15 | Other bearings | ? | `bearing_big`, `lid_bearing_inner`, `lid_bearing_outer` in CAD. Sizes: **TBD**. |
| 16 | Steel parts | 2 sets | `knee_steel`, `stand_steel_00_*` (several revisions; note which is current), `nail` (foot pin). Laser-cut or turned. STEP/drawing per part goes on the release. |
| 17 | Ball feet | 2 | 20 mm ball on the nail. Material: **TBD**. |
| 18 | Wiring | — | XT30/XT60 leads, CAN bus twisted pair with 120 Ω terminations, motor power leads, Pi USB-C. |

## Total

Priced lines come to about ¥5700 (¥5320 if only one battery pack is
counted). The unpriced lines are small, so the build lands near the ¥5700
figure the project quotes. Update this file when the TBD lines are filled and
remove this paragraph.

## Also to record before release

- Print time, filament mass, which parts need supports (from `all.gcode.3mf`).
- A photo of each purchased part next to its listing, so a builder can match
  Taobao variants (the MCP2518FD boards in particular vary in silkscreen).
