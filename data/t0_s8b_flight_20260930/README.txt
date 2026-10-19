T0 / S8b balance flight, 2026-09-30.  (analysis summary below is read off the npz;
the "user notes" column is for you to fill in.)

Source: pi:~/Documents/big_thigh/control/  (scp'd 09-30)
Also in this folder: balance.py / balance_log.py / config.py / policy.py exactly
as they sat on the Pi during the runs (mtimes 09-30 14:36) — the as-run deploy
scripts, so any obs/action contract question can be answered against the code
that actually executed.

Policies (same picks as 09-29, raw training checkpoints, loaded bit-exact by
control/policy.py; npz key policy_iter stamps the iteration):
  control/t0_1020.pt   = p9_replant/2026-09-26_22-11-33_T0_straight/model_1020.pt
  control/s8b_8880.pt  = p9_replant/2026-09-19_08-14-51_S8b_air03/model_8880.pt
                        (S8 vendor-rotor plant + belt DR 100-1000, feet_air_thr
                         0.03, resume of S8_vendorJ 7920 <- S4a 7780)

Contract at run time: obs 36x5=180, control_hz 100, handover = blind ('v') on
ALL FOUR trials, imu_pg_offset = [-0.015, 0.0], lpf_alpha 0.6, kp/kd logged
per-run. Trained with dr_proj_grav_bias_max = 0.00, so the trim is the open
contract knob — all four here used the 09-20 tile measurement (-0.015), NOT the
+0.089 plant-gap compensator.

file                                   policy     iters  n     dur    pg_raw first -> last   notes
-------------------------------------------------------------------------------------------------------
t0_1020_log_20260930_144158.npz        t0_1020    1020   171   1.71s  -0.011 -> +0.640
t0_1020_log_20260930_144211.npz        t0_1020    1020   263   2.64s  -0.008 -> -0.700
s8b_8880_log_20260930_145106.npz       s8b_8880   8880   838   8.42s  +0.001 -> -0.196
s8b_8880_log_20260930_145128.npz       s8b_8880   8880   868   8.72s  -0.019 -> -0.280

pg_raw = proj_grav_raw column 0 (RAW, before trim); logged proj_grav =
raw - imu_pg_offset[0]. Both trial groups opened near raw 0 (within +-0.02), so
the handover frame was well conditioned in all four.

Read-off observations (mechanical, not conclusions):
 - T0 (1.71 s / 2.64 s): short, and the two trials diverge in OPPOSITE pitch
   directions (+0.64 vs -0.70 rad). A one-signed lateral/trim bias cannot
   produce that; treat these as two separate destabilizations, not one bias.
 - S8b (8.42 s / 8.72 s): both runs survive past 8 s and both end with pg_x
   drifting NEGATIVE (-0.20, -0.28) — pitched forward, same sign both trials.
   The first S8b trial reached raw min -0.631 mid-run, i.e. it recovered once
   and fell later.

Terms: "trial" = real hardware recording. `sim_joint_pos` is MOTOR-side-derived
(encoder through the 22:30 belt decode) — belt stretch/slip is invisible in it,
so a "tracked" knee target does not prove the joint tracked. No video was
recorded to the Pi for this session (checked: no mp4/avi/mkv anywhere under ~
newer than 09-29 12:00), so the 09-30 trials have no frame-by-frame ground truth.

Tooth-skip rule: crawl-pose knee L ~= -R within ~3 deg between EVERY trial
(test_position.py); 1 tooth = 12 deg.  User notes column: fill in below.

file                                   user notes (what happened on the floor)
-----------------------------------------------------------------------------
t0_1020_log_20260930_144158.npz
t0_1020_log_20260930_144211.npz
s8b_8880_log_20260930_145106.npz
s8b_8880_log_20260930_145128.npz

--- user notes, added 2026-09-30 (from the user; video reviewed but not delivered) ---
t0_1020_log_20260930_144158.npz : T0 fails immediately.
t0_1020_log_20260930_144211.npz : T0 fails immediately.
s8b_8880_log_20260930_145106.npz: S8 stepping; the best-looking checkpoint of the day.
s8b_8880_log_20260930_145128.npz: S8 stepping; the best-looking checkpoint of the day.

Campaign state at 09-30: only two policies are live for hardware — r1r_3150 and S8
(s8b_8880). S8 is currently the best of the two: it steps. Bouncing/ringing was
observed in practice on the floor, and restitution is believed to play a role.
Real environment is noisy (objects on the ground, safety rope) — not modelled.
Vendor armature (rotor_inertia_output 4.2e-3) is taken as correct.
