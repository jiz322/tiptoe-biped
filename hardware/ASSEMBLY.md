# Assembly notes — v2.7

Written for: someone building a second TipToe from the MakerWorld files and
the BOM. This is a first draft from the design files and the campaign docs;
lines marked **TODO(owner)** need a measurement or a photo from the original
build before v0.1.0 ships.

## 1. Printed parts

Everything structural prints from `all.3mf` (v2.7). Parts, from the CAD file
list: `torso_top_board`, `torso2hip`, `chasis_side` ×2, `chasis_motor2motor`
×2, `chasis_rotate_00` ×2, `el05_flang` ×2, `thigh_inner` / `thigh_outer` ×2
each, `v2_thighlid_passive_inner` / `_inner2` / `_outer` ×2 each,
`shin_inner` / `shin_outer` ×2 each, `lid_bearing_inner` / `lid_bearing_outer`
×2 each, `pulley_5m_22` and `pulley_5m_30` ×2 each (printed pulleys; see §3),
`battery` holder.

- Material, grams, print time, which parts need supports: **TODO(owner)** from
  `all.gcode.3mf`.
- The feet are 20 mm balls printed in TPU on the steel `nail` pin (§4).

## 2. Machined parts

Three steel parts per leg, laser-cut or turned: `knee_steel` (the knee axle
plate), `stand_steel_00` (the shin's load path; CAD has five revisions, the
one on the flying robot is **TODO(owner)**) and `nail` (the foot pin). STEP
files for these will be on the Release page.

## 3. The belt knee

The knee motor (RS01, ids 102/105) is mounted in the hip body next to the hip
pitch motor. A 22-tooth pulley on the knee motor drives a 30-tooth pulley on
the knee axle through an HTD 5M belt inside the thigh, so:

- knee joint torque = motor torque × 30/22 (17 Nm motor → 23.2 Nm joint);
- knee joint angle = (22/30)·θ_motor − 0.733·q_hip_pitch: moving the hip with
  the knee motor still moves the knee. The deploy stack decouples this
  (`control/motor_bridge.py`); the sim models the drivetrain
  (`RL/.../belt_knee_actuator.py`).

Known weakness: the drive has **no margin**. At the 17 Nm motor clamp the belt sees
~970 N against the ~780 N a 10 mm belt is rated for. It skips a tooth under
catch-step and stand-up loads (6.5–12.6 Nm joint-side measured in September), and on
2026-10-16 a belt lost 3 of its 7 steel cords at the 22T pulley; a fresh belt broke
the same way within two trials. Every skip shifts the knee zero by 0.21 rad and the
motor-side encoder cannot see it. The idler pulley added on 10-11 (pressing the belt
onto the 22T) bought one session and is **not** a fix. The owner's redesign — 33:45
pulleys (the same 0.7333 ratio), a 20 mm HTD 5M belt, no idler — was decided on
10-16 and is not published yet. Build that, or aluminium pulleys, or a knee-side
encoder, before flying this drive hard; the 22:30 version in these files is the one
that failed.

Tension check used in the campaign: pluck the belt and listen for the ~140 Hz
floor; re-zero the knees at the crawl pose before every session
(`control/setzero_crawl.py`), and read the crawl-pose knee angles between every
trial — 1 tooth = 12°.

## 4. Feet

A 20 mm TPU ball on the `nail` pin at the end of each shin. Point feet mean
no static sagittal balance: the policy has to step. Ball centre sits 18.7 mm
above the floor when standing; the contact point is 5.8 cm beyond the shin's
nominal toe point (this matters for any centre-of-mass argument, see
`docs/standup-path-mode-20260908.md`).

## 5. Electronics and wiring

- **Power**: two 4S 6500 mAh packs (one flies, one charges) into a Matek
  PDB-HEX; its BEC feeds the Pi and the IMU 5 V rail. Fuse and XT connector
  sizes: **TODO(owner)**.
- **CAN**: all six motors on one 1 Mbps classic CAN bus with 120 Ω at both
  ends. MCP2518FD board on the Pi's SPI, `mcp251xfd` overlay, `can0`. Wiring,
  crystal check (20 vs 40 MHz boards) and per-boot `setup_can0.sh` are in
  `control/CAN2SPI_BRINGUP.md`. Each motor must be given its own MIT host id
  (`control/set_host_ids.py`), otherwise replies collide.
- **IMU**: WHEELTEC H30 on the torso top board, UART 460800 baud, 400 Hz,
  read by `control/imu_driver_6_400hz.py`. It is mounted rotated 180° about z;
  the sign constants in `control/config.py` encode that. Verify with
  `control/imu_sign_check.py` and `docs/imu-sign-check.md` after mounting
  yours. The measured pitch mounting offset is about −0.03. The +0.089 trim in the
  released config is **not** a mounting correction — it was the plant-gap compensator
  the September policy needed and is retired; the bias was re-measured on 09-20
  (−0.013) and every flight since 09-30 used trim −0.015 (retrospective II §0).
- **Radio**: ELRS receiver, CRSF on `/dev/ttyAMA2` at 420000 baud; arms the
  RC loop in `control/main.py`. The bench script `balance_log.py` does not
  need it.
- **Pi**: Raspberry Pi 5, mounted on the back of the torso (which is why a
  supine IMU calibration is not possible). Python with torch from a miniforge
  environment; the system Python has no torch.

## 6. Motor setup and zeroing

1. Switch each RobStride motor to MIT mode with the vendor tool, ids 101–106
   as in `control/config.py` (101/104 hip pitch, 102/105 knee, 103/106 hip
   roll). Firmware register dumps of the campaign's motors are in
   `control/fw_dumps/` for comparison.
2. Write distinct host ids: `sudo python3 control/set_host_ids.py`.
3. Bring-up guard: `ip -details link show can0` (error-active, counters 0),
   then `sudo python3 test_motors.py --no-zero` shows a live readout.
4. Zero at the crawl pose (`CRAWL_JOINT_POS` in config): `setzero_crawl.py`.
   The knee zero is what belt slip corrupts; check it at the crawl pose before
   every session.

## 7. Rest pose

The robot rests and starts in seiza: knees and foot balls on the ground, torso
vertical. That pose is the d = 1.0 end of the stand-up curriculum and the
"ground start" of every flight recording.

## Photos

**TODO(owner)**: assembled robot in seiza; exploded or print-plate view; the
belt path with the thigh lid off; CAN board wiring; IMU orientation on the
top board. Put them in `hardware/img/` and link them from the sections above.
