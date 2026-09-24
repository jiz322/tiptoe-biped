P11 flight recordings, 2026-08-24, policy K4b_3600 (12 trials).
First flight of the K training line. Every trial fell at the first catch-step.

Contract at the time (see data/README.md for the decode rules):
- IMU pitch trim -0.089 (raw pg_x = logged - 0.089).
- Handover 'b' (sighted: the real IMU frame primed x5). The blind 'v'
  handover did not exist yet (added 2026-08-29).
- IMU lateral sign constants were the OLD, mirrored set (fixed 2026-08-28):
  pg_y, omega_x, omega_z carry the wrong sign in these files.

Trial start condition: on the ground unless noted otherwise in the retrospective.
Session context: docs/RETROSPECTIVE-2026-09.md Phase D, docs/p11-hardware-runbook.md.
