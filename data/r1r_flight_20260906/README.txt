R1r flight recordings, 2026-09-06, policy R1r_3150 (3 trials).
Companion trials to the S4 checkpoint test of 2026-09-10 (see
data/s4_flight_20260910/README.txt and docs/s4-stepping-gate-failure-analysis-20260911.md).

Contract: config as of 2026-09-06 = the 09-03 standing configuration:
- IMU pitch trim +0.089 (raw pg_x = logged + 0.089).
- Handover 'v' (blind, training contract).
- IMU signs corrected (2026-08-28 set).
Trial start condition: on the ground.

Outcomes (from the npz, tilt = angle of body z from vertical; owner's notes):
  r1r_3150_log_20260906_155236.npz  1.67 s  fell (tilt > 25 deg at 1.30 s, never recovered)
  r1r_3150_log_20260906_155253.npz  5.23 s  upright at log stop (final tilt 12 deg)
  r1r_3150_log_20260906_155309.npz  8.79 s  upright at log stop (final tilt 20 deg); the
                                            robot shuffled into an obstacle, which is why
                                            the trial ended. Longest hardware balance to date.
Both surviving trials crossed 25 deg transiently (36 and 39 deg peaks) and
caught themselves by stepping, as the 09-03 trial did (34 deg excursion).
