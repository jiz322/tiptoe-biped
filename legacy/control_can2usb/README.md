# CAN-over-USB era deploy scripts (before 2026-08-15)

The robot originally talked to the RobStride motors through the vendor's
USB-to-CAN serial adapter (`/dev/ttyUSB0`). Retrospective fix #10 replaced it
with an MCP2518FD SPI board on SocketCAN. These scripts construct the serial
driver directly and fail with "no such port" on the current robot:

- `diag_replies.py`: why CAN replies were dropping (the 3-deep reply FIFO).
- `can_timing_probe.py`: the probe that measured the 2.93 ms adapter turnaround
  and led to the SPI swap.
- `imu_latency_test.py`, `test_action_latency.py`, `test_joint_latency.py`:
  tap-test latency measurements from 2026-06.

Kept because the probes are the evidence behind fix #10. Live equivalents are
`control/can_baseline_bench.py` and `control/CAN2SPI_BRINGUP.md`.
