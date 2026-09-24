# Hardware

TipToe is a 6-DOF point-foot biped: per leg hip roll (RobStride EL05), hip
pitch (RobStride RS01) and knee (RS01), with the knee motor sitting in the hip
body and driving the knee through a 22:30 HTD 5M belt. Compute is a Raspberry
Pi 5 running the 100 Hz loop in Python over SocketCAN. Flight mass 5.57 kg,
about 50 cm tall. Version on the flying robot: **v2.7**.

| File | What |
|---|---|
| [BOM.md](BOM.md) | Bill of materials with the prices actually paid (CNY, 2026-09). Priced lines total about ¥5360. |
| [ASSEMBLY.md](ASSEMBLY.md) | Build notes: print, belt, feet, wiring, IMU, zeroing. |
| `control/CAN2SPI_BRINGUP.md` | MCP2518FD wiring, crystal check, Raspberry Pi OS overlay and `can0` bring-up. |
| `assets/tiptoe_v27/` | URDF with meshes (Onshape export, mass-corrected 2026-09-18) and the Isaac Sim USD built from it. |

## Where the CAD is

CAD is not in git (440 MB across revisions).

- **MakerWorld** (print files): `all.3mf` with print profiles for the whole
  robot, plus per-part STL. Link: *to be added at launch*.
- **GitHub Release v0.1.0**: STEP export of the v2.7 assembly and per-part
  STEP, for people who do not use SolidWorks. Link: *to be added at launch*.
- Native SolidWorks parts (103 SLDPRT, 4 SLDASM) are available on request.

## Licence

Hardware files are CC BY 4.0 (`LICENSE-CC-BY-4.0`). Build it, sell it,
change it; keep the attribution.
