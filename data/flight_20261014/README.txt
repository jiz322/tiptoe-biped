Flight 2026-10-14: S11 / S10b / S12 stand-up from d=0.6, heading hold default. 6 trials, 0 stand-ups.
Analysis: docs/flight-analysis-20261014.md.  Runbook: docs/flight-runbook-20261014.md.

hw/  scp'd from pi:~/Documents/big_thigh/control/ on 10-14 (first three 14:55, rest 15:21).
Command: sudo ~/miniforge3/bin/python3 balance_s11.py --policy <p> --alpha 0.6 --pg-offset -0.015
(defaults: grace 2.5, heading hold; blind handover).  Deploy code as run = Pi tree, which differs from c53e7aa in
balance_s11.py / s11_contract.py (scp'd 10-14, see runbook section 7).

id  file                                    policy      n    dur    note (from npz)
T1  s11_11880_s11_log_20261014_144055.npz   s11_11880   179  1.79s  fell backward 1.32 s
T2  s11_11880_s11_log_20261014_144204.npz   s11_11880   146  1.46s  L_knee silent from 0.95 s; fell 1.30 s
T3  s11_11880_s11_log_20261014_145153.npz   s11_11880   165  1.65s  heading_mode off; fell 1.06 s
T4  s11_11880_s11_log_20261014_150836.npz   s11_11880   389  3.90s  fell 1.20 s (cmd_active fired on the floor)
T5  s10b_12179_s11_log_20261014_151126.npz  s10b_12179  192  1.92s  roll -20..-30 deg; fell 1.29 s ("most violent")
T6  s12_13010_s11_log_20261014_151500.npz   s12_13010   136  1.36s  roll -25..-30 deg; fell 0.74 s
T7  s11_11880_s11_log_20261014_153118.npz   s11_11880   225  2.25s  fell 1.24 s (scp'd 10-14 later)
T8  s11_11880_s11_log_20261014_153200.npz   s11_11880   332  3.33s  pitch -38 deg at 1.5 s, then +30 forward (help?)
--  s11_11880_s11_log_20261014_142311_DRY.npz  dry run on the stand (pg_x -0.013 throughout; not a supine check)

User notes
----------
10-14: no flight stood up, all failed very quickly. S10b was the worst (violent).
User's thinking: (1) belt-pulley friction got worse, the belt is close to degrading; (2) the voltage was higher
than last time. Real reason unknown.
10-14 (after first analysis): user NEGATES "stand-up was always marginal": 10-11 fresh trials stood every time
(stall-bug trials excepted); today it does not stand EVEN WITH HELP, and tightening the belt did not help either.
asrun/ = balance_s11.py / s11_contract.py / config.py pulled from the Pi (as flown).

TODO (user):
  - video, 0.5-1.0 s after handover: do the feet pivot/slide? (the sustained yaw, doc section 2)
  - which trials had help / tightened belt (T7, T8?)
  - foot balls: worn / dirty / cleaned?
  - per trial from video: fall direction, tether slack/taut, did the feet slip/slide backward?
  - floor surface: same as 10-11?
  - battery voltage at the start of 10-11 and 10-14
  - bench checks done before flight? (supine/prone pg_x numbers, YAW CHECK verdict, R_knee connector)
  - crawl-pose tooth check after the flight
  - T3: why --heading off?
