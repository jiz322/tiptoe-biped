S8b stand-up flight, 2026-10-03.  Six trials, one policy (control/s8b_8880.pt =
p9_replant/2026-09-19_08-14-51_S8b_air03/model_8880.pt), one difficulty (d = 0.6,
s0 = 0.4, 300-step grace), blind 'v' handover, trim -0.015 on all six.
Analysis: docs/s8b-standup-flight-analysis-20261003.md (figures in docs/figs/standup-flight-20261003/).

Source: pi:~/Documents/big_thigh/control/  (scp'd 10-03 13:21)
Also here: balance_standup.py / standup_path.py / config.py / policy.py as they sat on
the Pi during the runs. All four are byte-identical to the repo at commit 04c533a
("Flight standup alpha0.6"); the Pi clone was on that commit with no local edits.

Pi command: sudo ~/miniforge3/bin/python3 balance_standup.py --policy s8b_8880.pt --alpha 0.6 --pg-offset -0.015
Log schema: balance_log channels + grace_remaining, path_s, action_center, filtered_action,
hold_target, pose_gate_ok; meta keys difficulty, path_s0, grace_steps_assigned,
max_grace_steps, init_h, spawn_*_jpos, standup_path_knots, action_lpf_alpha, obs_norm_*.
Every tick is mode NN_CONTROL: log t = 0 is the 'v' edge.

file                                        n    dur    onset  dir   end pg_x   user notes
----------------------------------------------------------------------------------------------
s8b_8880_standup_log_20261003_103129.npz   242  2.43s  1.48s  fwd   -0.96     foot slipped (video). crawl after: no tooth skip.
s8b_8880_standup_log_20261003_114235.npz   157  1.57s  ~0.9s  bwd   +0.10     knee shim changed + shin anti-slip. skip L 2 teeth, R 1.
s8b_8880_standup_log_20261003_115546.npz   256  2.57s  1.77s  fwd   -0.99     belt loose. skip L ~4, R ~3.
s8b_8880_standup_log_20261003_120707.npz   351  3.52s  2.48s  fwd   -0.95     belt tightened; stood to a crouch. skip L 1.
s8b_8880_standup_log_20261003_122513.npz   330  3.31s  1.55s  bwd   +0.43     belt tightest; sat down backward. crawl L -0.3 / R +10.9.
s8b_8880_standup_log_20261003_123247.npz   359  3.60s  1.86s  fwd   -0.96     same as flight 5. crawl L +1.6 / R +10.2.

onset = first t where |asin(pg_x raw) - path pitch| > 25 deg and stays. dir = fall
direction (negative pg_x = forward). Logs 4-6 run past the 3.01 s grace expiry only
because 'h' was pressed late — the robot had already fallen (user).

Crawl-pose reference after setzero_crawl.py: L knee -11.3 deg, R +11.4 (L ~= -R).
1 tooth = 12 deg. OPEN: was setzero_crawl re-run between flights 4, 5, 6? If not,
flights 5 and 6 skipped nothing new.

Channel trust: sim_joint_pos is motor-side derived — belt skip/stretch is invisible in it.
motor_tau_meas (torque_fdb) is filtered ~25-35 ms and reads low on transients. No video
on the Pi; the user's frame-by-frame calls above are the ground truth.

Scripts: _summary.py (table), _detail.py (per-window stats), _onset.py (clip fractions,
onset windows, z-scores), _skipsig.py (roll bias, torque-limit windows, lift-off),
_figs.py (report figures). Run with python.
