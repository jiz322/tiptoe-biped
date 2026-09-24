P12.4 flight recording, 2026-09-03 13:47, policy R1r_3150 (1 trial).
Source: pi:~/Documents/big_thigh/control/r1r_3150_log_20260903_134700.npz
(balance_log), copied 2026-09-03. Pi + local repo at commit efdd12d
("flight IMU_PG_OFFSET = [0.089, 0.0]").

WHAT CHANGED vs the two earlier sessions today:
- IMU pitch trim set to +0.089 (config.py IMU_PG_OFFSET = [0.089, 0.0]),
  i.e. the SIGN-REVERSED bracket of the historical -0.089 trim.
  Session sequence on 2026-09-03:
    data/p12_3_flight_20260831_v        trim -0.089 (08-31 baseline)
    data/p12_4_flight_20260903_v_notrim trim  0      (12:54-12:56, 6 trials)
    THIS FOLDER                         trim +0.089 (13:47, 1 trial)
- Raw-value rule for THIS file: pg_corrected = pg_measured - IMU_PG_OFFSET, so
  raw pg_x = logged + 0.089. First NN frame logged pg_x = -0.075 -> raw
  ~ +0.014 (physically upright at handover, as in previous sessions).

User verdict at collection time: "surprisingly good, very successful"
(the qualitative outcome; not yet scored).

Handover: 'v' key (blind / training-contract handover); npz handover='blind'.
Trial start condition: on the ground (no pickup), as with the 12:5x session.

File (control_hz=100, whole file mode NN_CONTROL):
  r1r_3150_log_20260903_134700.npz  603 frames  6.06 s
Recording length = handover to log stop, not scored survival time. For
comparison, the longest recording in the 12:5x no-trim session was 4.29 s and
the 08-31 (-0.089) session's scored survival was ~0.67 s.

Reminder (checklist item 9): sim_joint_pos is motor-encoder-derived through
the 22:30 belt decode; knee joint angle is not independently verified here.
