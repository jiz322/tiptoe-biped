"""
config.py — Sim2real configuration for the bipedal robot.

Every constant that ties simulation to hardware lives here. If a value
appears in both rl_env_cfg.py (sim) and on the real robot, it MUST match.
"""

import math

# ════════════════════════════════════════════════════════════
# Hardware ports
# ════════════════════════════════════════════════════════════
MOTOR_PORT = '/dev/ttyUSB0'   # CAN-USB adapter (RobStride)
IMU_PORT   = '/dev/ttyAMA0'   # Yesense YIS IMU
ELRS_PORT  = '/dev/ttyAMA2'   # ELRS receiver (CRSF)

MOTOR_BAUD = 921600
IMU_BAUD   = 460800
ELRS_BAUD  = 420000

# ── CAN transport selection (2026-08-15 CAN2USB → CAN2SPI swap) ──
# 'can2usb': RobStride serial AT adapter on MOTOR_PORT (flight-tested;
#            L = 2.93 ms/transaction, needs 1 ms pacing).
# 'can2spi': MCP2518FD SPI adapter via SocketCAN (wiring + OS bring-up
#            in CAN2SPI_BRINGUP.md; per-boot: sudo ./setup_can0.sh).
# Flip at runbook Phase B — everything that builds its driver through
# robstride_socketcan_driver.make_driver() picks it up, and
# can_baseline_bench selects the matching TransportShim automatically.
CAN_TRANSPORT = 'can2spi'
CAN_CHANNEL   = 'can0'
CAN_BITRATE   = 1_000_000     # motor firmware motor_baud=1 → 1 Mbps classic CAN
# Inter-frame pacing for the CAN2SPI batch path — 0.0 since 2026-08-16
# (BASELINE_RUNBOOK.md Phase D): each motor carries its OWN MIT host id
# (= its motor id, set via set_host_ids.py; factory was 0xFD on all six,
# which made every reply share arbitration id 0x0FD and collide on
# back-to-back requests — that's what the old 0.5 ms pacing worked
# around). With distinct ids the bus arbitrates replies natively:
# unpaced batch 100 % full @ 1.83 ms, 120 s soak 100.00 % full replies
# (baseline_can2spi_20260816_155207.json). Guard: two motors sharing a
# reply id again (e.g. a replacement at factory 0xFD next to another)
# re-creates the collisions — setup_can0.sh runs set_host_ids.py --show
# at bring-up and refuses on duplicates. Full revert = set_host_ids.py
# --revert + 0.0005 here (the validated 08-15 shared-id config).
CAN2SPI_PACING_S = 0.0000

# ════════════════════════════════════════════════════════════
# Joint mapping  (sim index ↔ real motor ID)
# ════════════════════════════════════════════════════════════
# Sim joint order (FROZEN — must match rl_env.py _desired_order):
#   0: hip_roll1   1: hip_pitch1   2: knee1
#   3: hip_roll2   4: hip_pitch2   5: knee2
#
# User-stated mapping:
#   "left hip(=pitch), left knee, left roll, right hip, right knee, right roll
#    maps to id 101..106"
#
# So leg-1 = left, leg-2 = right:
#   sim 0 (hip_roll1)  ←→ motor 103  (left roll)
#   sim 1 (hip_pitch1) ←→ motor 101  (left hip pitch)
#   sim 2 (knee1)      ←→ motor 102  (left knee)
#   sim 3 (hip_roll2)  ←→ motor 106  (right roll)
#   sim 4 (hip_pitch2) ←→ motor 104  (right hip pitch)
#   sim 5 (knee2)      ←→ motor 105  (right knee)
SIM_JOINT_NAMES = [
    'hip_roll1', 'hip_pitch1', 'knee1',
    'hip_roll2', 'hip_pitch2', 'knee2',
]
MOTOR_IDS = [103, 101, 102, 106, 104, 105]

# Per-joint sign flip applied as: sim_angle = motor_angle * sign + offset
# Verify each one with tools/test_motors.py before running the policy.
JOINT_SIGN = [
    +1,  # hip_roll1
    +1,  # hip_pitch1
    -1,  # knee1
    +1,  # hip_roll2
    +1,  # hip_pitch2
    -1,  # knee2
]

# Zero-pose offsets [rad] — computed by _make_joint_offset() below.
# Motors are zeroed at the selected ZERO_POSE (test_motors.py does this
# automatically). The offsets shift motor-zero to sim-zero so that sim
# coords read correctly across the full range of motion.

# ════════════════════════════════════════════════════════════
# Knee gear ratio + hip coupling
# ════════════════════════════════════════════════════════════
# Knee motor mounted on hip body, drives knee through 22:30 reduction.
#   knee_joint_speed  = motor_speed * (22/30)
#   knee_joint_torque = motor_torque * (30/22)
#
# Sim says knee_joint_torque_max = 23.2 Nm. Hardware motor max = 17 Nm.
#   17 * (30/22) = 23.18 Nm ✓
#
# COUPLING: because the knee belt is anchored to the hip body, rotating
# the hip pitch shifts the knee joint angle even when the motor is still.
# Common belt-driven design: knee_joint_angle = motor*ratio - hip_pitch * coupling.
# Set 1.0 to start, tune empirically (test_motors.py shows hip motion → knee shift).
KNEE_GEAR_RATIO   = 22.0 / 30.0
KNEE_HIP_COUPLING =  0.733

# ── Belt-coupling torque compensation (2026-08-06 forensics) ──────
# The position transforms above make knee POSITION tracking correct, but
# the firmware PD still fights the belt: torque forensics on the 08-06
# logs showed the knee motor's Kd cancels ~30% (RMS) of its own commanded
# torque whenever hip pitch moves (it damps the belt-forced motor spin),
# and the knee motor's torque reacts on the hip joint at c*tau_knee
# (~35% of the hip's own command RMS). Sim has neither effect — the
# trained policy expects a rigid direct-drive knee.
# KNEE_DECOUPLE_FF=True sends, per tick:
#   knee motor v_des = c * qd_hip * sign / G   (Kd tracks the belt drag
#                                               instead of fighting it)
#
# CONTRACT RULE (runbook §0.3, knee-hip-coupling-review-20260821 §5):
# this flag must EQUAL the sim `knee_decouple_ff` the flying policy
# trained under. The belt sim (BeltKneeDCMotor) models the drivetrain
# WITH this feedforward live (env.yaml knee_decouple_ff: true = "train
# against the plant as deployed") — so sim-True REQUIRES hardware-True.
# A previous comment here claimed the opposite ("running both
# double-compensates") — that inversion drove the 08-17 revert and
# re-armed the quantified G2i mismatch: 0.3665·qd_hip Nm unmodeled knee
# drag, 21–23% of knee PD torque, equivalent damping mult ~0.70 (the DR
# floor) — ×3 absolute at K kd 1.5. K-line trained True → True for P11.
KNEE_DECOUPLE_FF = True
# HIP_REACTION_FF: also send hip t_ff = -c * tau_knee_expected to cancel the
# ideal-transmission belt reaction on the hip. DISABLED by default: the
# 2026-08-06 teacher-forcing replay showed injecting +c*tau_knee into the SIM
# hip makes sim match the real hip WORSE (+0.02 rad one-step RMSE), i.e. the
# real hip does not feel the ideal J^T reaction — so "cancelling" it on real
# hardware would introduce a disturbance, not remove one. Re-test on the
# harness before ever enabling.
HIP_REACTION_FF = False

# ════════════════════════════════════════════════════════════
# Reference poses (must match rl_env_cfg.py)
# ════════════════════════════════════════════════════════════
DEFAULT_JOINT_POS = [
    0.0,    # hip_roll1
    0.05,   # hip_pitch1
    0.0,    # knee1
    0.0,    # hip_roll2
    -0.05,  # hip_pitch2
    0.0,    # knee2
] # do we still need this???

KNEEL_JOINT_POS = [
    0.0,   3.09, -3.06,
    0.0,  -3.09,  3.06,
]
CRAWL_JOINT_POS   = [0.0, 1.59, -0.2,  0.0, -1.59, 0.2]   # NEW
KNEEL_QUAT = (0.7071, 0.0, -0.7071, 0.0)   # (w, x, y, z)

TARGET_HEIGHT = 0.35        # do we still need this???
SPAWN_HEIGHT  = 0.4401       # must match training init_h (was 0.2188)
KNEEL_HEIGHT  = 0.234        # K env.yaml kneel_height (was 0.2188 pre-K)

# ════════════════════════════════════════════════════════════
# Action scales (must match rl_env_cfg.py)
# ════════════════════════════════════════════════════════════
ACTION_SCALES = [
    0.30,   # hip_roll1   (training-matched)
    0.75,   # hip_pitch1  (training-matched)
    1.025,  # knee1       (training = 2.05/2, belt reduction)
    0.30,   # hip_roll2
    0.75,   # hip_pitch2
    1.025,  # knee2
]

# ════════════════════════════════════════════════════════════
# PD gains in SIM joint space
# ════════════════════════════════════════════════════════════
# These are the gains the policy expects at the joint level.
# motor_bridge.py converts to motor-space gains (multiply by gear² for
# knees, exactly ONCE): knee joint 30/1.5 → motor-side 16.1/0.81.
# P11 K-line servo contract (frozen in the K runs' env.yaml servo_kp/kd):
# hip_roll 6/0.3 unchanged, hip_pitch 20/1.4, knee 30/1.5.
# Frames trap (runbook §0.3): the retired pre-K row had knee JOINT-side
# kp 16.0; the K contract's knee MOTOR-side kp is 16.1. Same digits,
# different frames — this file is always JOINT-side.
REAL_KP  = [6.0, 20.0, 30.0,  6.0, 20.0, 30.0]
REAL_KD  = [0.3,  1.4,  1.5,  0.3,  1.4,  1.5]

BURST_KP = [60.0, 40.0, 40.0, 60.0, 40.0, 40.0]
BURST_KD = [2.0,  2.0,  1.5,  2.0,  2.0,  1.5]

# Joint-space effort limits (enforced as soft-clip in motor_bridge.py).
# REAL_EFFORT must equal the sim actuator effort_limit the flying policy
# trained under (K env.yaml: 6.0 / 17.0 / 23.2). The old continuous-
# rating row [1.8, 6.0, 8.18] combined with K kp 20/30 would clamp
# position error at 6/20 = 0.30 rad on hip_pitch — a ~2.8× torque
# haircut vs the trained plant, binding exactly during catch responses.
# Motor firmware current limit (17 Nm motor-side) stays the hard backstop.
REAL_EFFORT  = [6.0, 17.0, 23.2,  6.0, 17.0, 23.2]
BURST_EFFORT = [6.0, 17.0, 23.2,  6.0, 17.0, 23.2]

# ════════════════════════════════════════════════════════════





# ════════════════════════════════════════════════════════════
# IMU axis remap
# ════════════════════════════════════════════════════════════
# The Yesense IMU returns euler as (pitch, roll, yaw) in deg, gyro in deg/s.
# 2026-08-28: first physical sign check (imu_sign_check.py + acc-ground-truth
# arbitration, data/_imu_sign_check_arbitrate*.py, docs/imu-sign-check.md)
# found the sensor mounted 180° about z (sensor x BACKWARD, z up — user
# physically verified the mount) and the old sign set implementing a
# left-right MIRROR of the true frame: {pg_y, w_x, w_z} came out flipped.
# ROLL/GYRO_X/GYRO_Z corrected same day (old values +1/+1/-1). Every
# hardware recording/flight BEFORE 08-28 carries mirrored lateral channels
# (pg_y, w_x, w_z) — annotate when comparing old npz traces. IMU_PG_OFFSET
# is x-only, so the -0.089 trim survives this fix. Re-run imu_sign_check.py
# (expect all-OK) + obs_parity_check.py at next bench session.
IMU_PITCH_SIGN  = -1
IMU_ROLL_SIGN   = -1
IMU_YAW_SIGN    = +1

IMU_GYRO_X_SIGN = -1
IMU_GYRO_Y_SIGN = -1
IMU_GYRO_Z_SIGN = +1

# ── IMU mounting/attitude trim (2026-08-11) ──────────────────────
# Constant subtracted from projected_gravity before it reaches the policy:
#     pg_corrected = pg_measured - IMU_PG_OFFSET
# The 10-trial ground session measured pg_x = -0.089 +- 0.017 at launch
# with joints at DEFAULT — 3x outside the trained dr_proj_grav_bias_max
# band, and a forced obs bias of that size kills 88% of sim robots.
# A mounting rotation is unobservable from IMU data alone (the gyro is
# rotated identically), and on point feet the joint encoders only see
# ~20% of torso lean (contact rotation is a free variable — measured via
# data/imu_offset_from_log.py leave-one-bias-out CV), so this is a TRIM
# parameter, set empirically:
#   start at the launch-stat estimate below, then adjust by fall
#   direction: still falls pitch-FORWARD -> make x more negative
#   (stronger correction); falls BACKWARD -> overcorrected, back off.
# Trim to within ~+-0.05; the widened dr_proj_grav_bias_max=0.1 policy
# (G_delay14_imuoffset010) tolerates the residual.
#
# 2026-08-13 knife-edge measurement (kp=30 PD hold, repeated hand
# releases): the static balance point reads DISPLAYED pg_x = +0.075
# under this trim (below: always backward, above: always forward).
# Sim's frozen-pose balance point reads 0, so SOME static bias of
# +0.075 exists — but it is NOT attributable to the IMU alone: joint
# zeroing (set_zero_crawl), CoM-vs-URDF error, and point-foot contact
# geometry all fold into the same number. Folding it into this trim
# (x = -0.089+0.075 = -0.014) would re-center the policy on the true
# balance point but hide the root cause; deliberately NOT applied
# until the contributors are separated (IMU: pg at a jig-held
# reference pose; zeros: re-zero repeatability; CoM: bench balance).
#
# 2026-08-24 (runbook §0.4 decision): STAYS -0.089 for the P11/K flight.
# The 08-18 decomposition found mounting ≈ -0.032, i.e. this trim
# over-corrects by ~0.057 — but P9/P10 trained with
# dr_proj_grav_bias_max = 0.10, which covers exactly that bias family.
# Do NOT retrim before flight; the policy was trained against this bias.
IMU_PG_OFFSET = [0.089, 0.0]   # [x, y]; [0,0] disables

# ════════════════════════════════════════════════════════════
# Control loop (must match sim: dt=1/200, decimation=2 → 100 Hz)
# ════════════════════════════════════════════════════════════
CONTROL_HZ = 100
DT         = 1.0 / CONTROL_HZ

# Gait clock (must match sim)
GAIT_FREQ_HZ = 2.5

# Spawn-yaw resample period (must match cmd_resample_steps in cfg)
YAW_REF_RESAMPLE_STEPS = 1000   # K env.yaml cmd_resample_steps (was 100 pre-K)

# ════════════════════════════════════════════════════════════
# Standup
# ════════════════════════════════════════════════════════════
STANDUP_GRACE_SECONDS = 4.0   # K env.yaml standup_grace_seconds (was 5.0 pre-K)
MAX_GRACE_STEPS = int(STANDUP_GRACE_SECONDS / DT)   # 500 @ 100Hz

# ════════════════════════════════════════════════════════════
# Safety
# ════════════════════════════════════════════════════════════
MAX_TILT_RAD       = 1.20    # ~70° emergency stop
MAX_HIP_ROLL_RAD   = 1.80
WATCHDOG_TIMEOUT_S = 0.20    # if main loop stalls > 200ms, kill motors
MAX_MOTOR_TEMP_C   = 80.0    # auto-HOLD if any motor exceeds this
CAN_CONSEC_DROP_LIMIT = 5    # consecutive all-miss ticks → HOLD

# ════════════════════════════════════════════════════════════
# RC mapping ranges
# ════════════════════════════════════════════════════════════
RC_VX_MIN = -0.15
RC_VX_MAX = +0.40
RC_YAW_RATE_MAX = 0.5
RC_DEADBAND = 30
RC_ARM_THRESHOLD = 1500
RC_STANDUP_THRESHOLD = 1500

# ════════════════════════════════════════════════════════════
# Policy
# ════════════════════════════════════════════════════════════
POLICY_PATH = 'r1r_3150.pt'   # R1r_3150: the policy that stood on 2026-09-03 (the only shipped checkpoint)
OBS_FRAME_DIM     = 36
OBS_HISTORY_STEPS = 5
OBS_DIM           = OBS_FRAME_DIM * OBS_HISTORY_STEPS   # 180
ACT_DIM           = 6



# ── Zero pose selection ──────────────────────────────────
# The pose the robot is in when set_zero_all() is called.
# Switching this is the only knob you need to flip if you change
# zeroing procedure — JOINT_OFFSET is derived from it.
ZERO_POSE_NAME    = 'crawl'
ZERO_POSE_JOINT_POS = CRAWL_JOINT_POS

def _make_joint_offset(zero_pose):
    """Offsets such that sim == zero_pose when raw == 0.
    Inverts the forward map used by motor_bridge / test_motors:
        sim[i]    = raw[i] * sign[i] + off[i]                       (non-knee)
        sim[knee] = raw[knee] * sign[knee] * gear
                    - sim[hip] * coupling + off[knee]
    """
    off = list(zero_pose)
    off[2] = zero_pose[2] + zero_pose[1] * KNEE_HIP_COUPLING
    off[5] = zero_pose[5] + zero_pose[4] * KNEE_HIP_COUPLING
    return off

JOINT_OFFSET = _make_joint_offset(ZERO_POSE_JOINT_POS)