P12.4 flight recordings, 2026-09-03 ~12:54-12:56, policy R1r_3150 (6 trials).
Source: pi:~/Documents/big_thigh/control/r1r_3150_log_20260903_*.npz (balance_log),
copied 2026-09-03 13:08. Pi + local repo at commit 423ace7
("try flight. IMU_PG_OFFSET = [-0.0, 0.0]").

WHAT CHANGED vs the 08-31 session (data/p12_3_flight_20260831_v):
- IMU pitch trim REMOVED: config.py IMU_PG_OFFSET = [0, 0] (was [-0.089, 0]).
  The policy now sees the RAW proj_grav. Consequence for analysis: the
  "raw pg_x = logged - 0.089" rule from the sim2real checklist (item 8) does
  NOT apply to these files - logged == raw. Verified: pg_x at the first NN
  frame is +0.004 / +0.008 / +0.001 / +0.012 / +0.023 / -0.009 (upright), vs
  ~+0.10 in the 08-28/08-31 npz under the old trim.

Handover mode: ALL trials started with the 'v' key (blind / training-contract
handover; obs-history zeroed like a training episode). npz field
handover='blind' in all six files.

Trial start condition: ALL six trials start ON THE GROUND (no pickup).

Per-file (control_hz=100, whole file is mode NN_CONTROL):
  r1r_3150_log_20260903_125450.npz  328 frames  3.29 s
  r1r_3150_log_20260903_125502.npz  203 frames  2.03 s
  r1r_3150_log_20260903_125517.npz  427 frames  4.29 s
  r1r_3150_log_20260903_125548.npz  180 frames  1.80 s
  r1r_3150_log_20260903_125600.npz  161 frames  1.61 s
  r1r_3150_log_20260903_125616.npz  237 frames  2.37 s
Durations are recording length (handover to log stop), NOT scored survival
time - score with data/_p12_3_alive_time.py or equivalent.

Reminder (checklist item 9): sim_joint_pos is motor-encoder-derived through
the 22:30 belt decode; knee joint angle is not independently verified here.
