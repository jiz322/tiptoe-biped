S4 balance flight, 2026-09-10 14:11-15:00. Policies s4a_6520 / s4b_6520 (raw
S4 training ckpts, iter 6520) and r1r_3150 (reference). balance_log.py 'v'
(blind) handover, ground start, --pg-offset per file (npz key imu_pg_offset;
proj_grav_raw logged directly, raw = logged + offset).

file                               policy  trim   user notes
s4b_6520_log_20260910_141108.npz   S4b   +0.089  R knee slipped 2 teeth (crawl check R +0.62 vs +0.20)
r1r_3150_log_20260910_141904.npz   R1r   +0.089  L knee slipped 1 tooth (L -0.42 vs -0.20)
s4a_6520_log_20260910_1433xx/1434  S4a   +0.089  3 trials, no tooth skip
s4b_6520_log_20260910_1435xx/1436  S4b   +0.089  3 trials, LAST one skipped 1 tooth (R +0.41)
s4a_6520_log_20260910_1500xx       S4a   +0.000  3 trials, no tooth skip; user: "looks more like sim"

User video verdict: R1r still the most stable (shuffle lasts longest); even at
+0.089 the robot tends to lean backward.  All three tooth skips read MORE
flexion at the crawl pose afterwards (encoder ahead of joint toward flexion).

Sim twins: RL/logs/s4_hwtwin/<arm>/ (run_s4_hwtwin_GPU2.ps1, pin = -trim).
Analysis: data/_s4_flight_summary.py, _s4_opening_compare.py, _s4_sim_vs_hw.py,
_s4_knee_transient.py, _s4_hw_drift_dynamics.py.  Companion 09-06 R1r trials
(trim +0.089, config at that date) in data/r1r_flight_20260906/.
