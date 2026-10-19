S11_hist stand-up / balance flights, 2026-10-11.  5 trials, data collected only --
NO analysis yet (to be done in a later session). User notes recorded below; the
"TODO" lines are for the user to complete.

Source: pi:~/Documents/big_thigh/control/  (scp'd 10-11 13:48, md5 verified vs Pi)
Also in this folder: balance_s11.py / s11_contract.py / config.py exactly as they sat
on the Pi (Pi tree == commit c53e7aa, `git diff` empty) -- the as-run deploy code.
(_tb_pick.py / _tb_pick.json predate the flights: checkpoint pick from deploy prep.)

Policy: control/s11_11880.ts.pt = S11_hist model_11880, exported by
RL/scripts/export_hist_policy.py (embedded contract.json + fixture; npz key
policy_iter = 11880, contract_json stored per file).

Commands (both groups: handover = blind, imu_pg_offset = [-0.015, 0], alpha 0.6):
  Group A:  sudo ~/miniforge3/bin/python3 balance_s11.py --policy s11_11880.ts.pt \
              --alpha 0.6 --grace-seconds 2.5 --pg-offset -0.015
  Group B:  same with --grace-seconds 1.7

file                                     grp trial grace  n     dur     process
---------------------------------------------------------------------------------------
s11_11880_s11_log_20261011_124840.npz    A   A1    2.5    1883  18.92s  fresh start
s11_11880_s11_log_20261011_125112.npz    A   A2    2.5     244   2.44s  SAME process as A1 (r -> k)
  -- tooth-skip check + setzero_crawl between groups (see below) --
s11_11880_s11_log_20261011_131450.npz    B   B1    1.7     993   9.97s  fresh start
s11_11880_s11_log_20261011_132958.npz    B   B2    1.7    4335  43.57s  fresh start (rerun)
s11_11880_s11_log_20261011_133104.npz    B   B3    1.7     167   1.66s  SAME process as B2 (r -> k)

(n / dur read from the `time` key; process column is from the user's account, not
yet confirmed from the npz.)

User notes
----------
A1 (grace 2.5): SUCCESS. Stood up; then pressed `s` and it sat down.
A2 (grace 2.5): FAILED to stand up. Cannot rule out user influence: user held the
    tether rope and it somewhat pulled the robot during stand-up. Cannot separate
    cause from outcome.
    10-12 video: same as B3 -- charged backward as usual, torso pitched backward too
    far, fell backward.

Between groups -- tooth-skip check: NO tooth skipped.
    test_position.py (crawl pose) readout:
                 L_roll   L_hip   L_knee  |  R_roll   R_hip   R_knee
      sim (rad):  +0.00   +1.59   -0.19   |  -0.00   -1.59   +0.19
      sim (deg):  +0.25  +91.11  -10.99   |  -0.05  -91.11  +10.93
    (expected [0.0, 1.59, 0.96547, 0.0, -1.59, -0.96547] printed at start)
    User ran setzero_crawl anyway before group B.

B1 (grace 1.7): stand-up SUCCEEDED, but user feels they helped it with the tether.
    While stepping it failed "strangely".
    10-12 video: stopped stepping at ~6.37 s NPZ time (counted from launch; LED not in
    frame, roughly accurate); looked like the robot lost power for a span.
B2 (grace 1.7): stand-up SUCCEEDED, again user feels they helped with the tether.
    Maybe 1.7 s is too fast / maybe the trajectory needs optimizing. cmd_x and yaw
    tested: work somewhat, no detailed analysis yet. Sit-down also works, but less
    elegant than with grace 2.5.
    10-12: heading change over the stand-still window "should be small" (derive from data).
B3 (grace 1.7): same process as B2; FAILED to stand up.
    10-12 video: same as A2 -- charged backward, torso pitched backward too far, fell backward.

10-12 user, all trials: tether-taut frames and floor displacement are too rough to
judge from video -- low importance, not provided. No other video items.
Analysis: docs/s11-flight-analysis-20261012.md (scripts _*.py here, figures figs/).

End of group B: NO tooth skipping.

User thinking / open questions
------------------------------
1. Only A1->A2 and B2->B3 reused the process (r, then k, then flight); every other
   trial was a fresh balance_s11.py run. BOTH reused-process trials (A2, B3) failed
   to stand up. Does this point at a control-code bug in the re-arm path (state not
   reset between runs in one process)?  -> to check (candidate npz keys: pre_*,
   ring_first_tick, hist_frame, ring_z, path_s0, grace_remaining, events).
2. Belt fix works: no tooth skip in any of today's 5 trials. Mechanical change: an
   idler pulley added on the top side of the small 22T driver pulley. It squeezes
   the timing belt against the driver so the 22T pulley cannot slip, and also
   tensions the belt. Side effects observed: belt drifts to the INNER side of the
   22T pulley groove and rubs the flange (same to a lesser extent on the 30T
   pulley). Possible extra friction -- effect on flight unknown; certainly a
   longevity concern for the PU-steel belt and the plastic pulleys.

