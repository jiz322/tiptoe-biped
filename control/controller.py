"""
controller.py — high-level state machine and policy execution loop.

States:
    DISARMED  → motors off, idle
    READY     → armed, softly holds captured pose, slowly blends to
                standing. Policy NOT running. Waiting for standup trigger.
    STANDUP   → grace_remaining > 0, action_center slides kneel→stand,
                burst PD gains active
    WALK      → normal locomotion, real PD gains, policy in charge

The per-frame observation layout MIRRORS rl_env._get_observations EXACTLY
(36 floats).  K=5 frames are stacked into a flat (180,) vector for the
policy, matching training's obs_history_steps.

    [ projected_gravity (3),
      ang_vel_imu       (3),
      lin_vel_imu       (3),
      joint_pos_norm    (6),   # current - default
      joint_vel         (6),
      last_actions      (6),
      gait_sin          (1),
      gait_cos          (1),
      difficulty        (1),
      grace_norm        (1),
      init_h            (1),
      vx_cmd            (1),
      yaw_rate_cmd      (1),
      yaw_err_sin       (1),
      yaw_err_cos       (1) ]
"""

import math
import time
import numpy as np

import config as C
from motor_bridge import MotorBridge
from imu_adapter import IMUAdapter
from policy import Policy


# State enum
DISARMED = 'DISARMED'
READY    = 'READY'
STANDUP  = 'STANDUP'
WALK     = 'WALK'


class BipedController:
    def __init__(self, motors: MotorBridge, imu: IMUAdapter, policy: Policy):
        self.motors = motors
        self.imu = imu
        self.policy = policy

        # Pose constants (sim space)
        self._default_jpos = np.array(C.DEFAULT_JOINT_POS, dtype=np.float32)
        self._kneel_jpos   = np.array(C.KNEEL_JOINT_POS,   dtype=np.float32)
        self._action_scale = np.array(C.ACTION_SCALES,     dtype=np.float32)
        self._real_kp  = np.array(C.REAL_KP,  dtype=np.float32)
        self._real_kd  = np.array(C.REAL_KD,  dtype=np.float32)
        self._burst_kp = np.array(C.BURST_KP, dtype=np.float32)
        self._burst_kd = np.array(C.BURST_KD, dtype=np.float32)
        self._real_effort  = np.array(C.REAL_EFFORT,  dtype=np.float32)
        self._burst_effort = np.array(C.BURST_EFFORT, dtype=np.float32)

        # Per-step state
        self.state = DISARMED
        self._step = 0

        # Policy I/O
        self._last_action = np.zeros(C.ACT_DIM, dtype=np.float32)
        self._gait_phase  = 0.0

        # ── Observation history (review #2) ──────────────────────
        # K frames of 36 floats.  Slot 0 = newest, K-1 = oldest.
        # On reset / state transition, the first real frame is
        # replicated into all K slots — mirroring sim's
        # _needs_history_init behaviour and avoiding the OOD zero-
        # history bug.
        self._obs_history = np.zeros(
            (C.OBS_HISTORY_STEPS, C.OBS_FRAME_DIM), dtype=np.float32
        )
        self._needs_history_init = True

        # Standup state
        self._grace_remaining = 0    # in steps
        self._init_height     = C.SPAWN_HEIGHT
        self._env_init_jpos   = self._default_jpos.copy()
        self._difficulty      = 0.0

        # Yaw reference
        self._spawn_yaw = 0.0

        # Commands
        self._vx_cmd = 0.0
        self._yaw_rate_cmd = 0.0

        # READY state: hold the captured pose (kneel) until standup.
        # Uses burst gains — strong enough to resist gravity at kneel,
        # and matches what STANDUP will use for a smooth transition.
        self._hold_pos = self._default_jpos.copy()

        # ── Safety state (review #6, #7) ─────────────────────────
        self._last_tick_time = time.monotonic()
        self._consec_can_drops = 0   # consecutive ticks with 0 replies
        # Expose the last observation frame for external logging.
        self._last_obs_frame = np.zeros(C.OBS_FRAME_DIM, dtype=np.float32)

    # ════════════════════════════════════════════════════════
    # State transitions (called from main loop based on RC)
    # ════════════════════════════════════════════════════════
    def arm(self):
        if self.state != DISARMED:
            return
        print("[Controller] ARM → READY")
        self.motors.enable()
        # Prime encoder cache: send_idle issues a no-torque MIT command
        # which makes each motor reply with its current pos/vel.
        self.motors.send_idle()
        time.sleep(0.02)
        self.motors.send_idle()
        self.motors.read_state()
        # Capture current pose — READY will firmly hold it (no sudden snap)
        self._hold_pos = self.motors.sim_joint_pos.copy()
        self._reset_runtime_state(initial_height=C.SPAWN_HEIGHT)
        self.state = READY

    def disarm(self):
        if self.state == DISARMED:
            return
        print("[Controller] DISARM")
        self.motors.send_idle()
        self.motors.disable()
        self.state = DISARMED

    def trigger_standup(self):
        """Begin a fresh standup sequence from the kneel pose.

        Caller is responsible for ensuring the robot is *physically* near
        the kneel pose first — the policy can recover small offsets but
        not arbitrary postures.
        """
        if self.state == DISARMED:
            print("[Controller] Cannot stand up while disarmed")
            return
        print(f"[Controller] STANDUP triggered (grace = {C.MAX_GRACE_STEPS} steps)")
        self._grace_remaining = C.MAX_GRACE_STEPS
        self._env_init_jpos = self._kneel_jpos.copy()
        self._init_height = C.KNEEL_HEIGHT
        self._difficulty = 1.0
        # Lock commands during grace (policy ignores anyway, but be explicit)
        self._vx_cmd = 0.0
        self._yaw_rate_cmd = 0.0
        self._spawn_yaw = self.imu.yaw()
        # Re-prime history so STANDUP starts from clean state (review #3)
        self._last_action[:] = 0.0
        self._needs_history_init = True
        self.state = STANDUP

    def set_commands(self, vx_cmd, yaw_rate_cmd):
        """Update locomotion commands.

        In READY: any nonzero command transitions straight to WALK
        (assumes robot is already near standing pose).
        In WALK: updates commands normally.
        """
        if self.state == READY:
            if abs(vx_cmd) > 0.02 or abs(yaw_rate_cmd) > 0.05:
                print("[Controller] READY → WALK (command received)")
                self._spawn_yaw = self.imu.yaw()
                # Clean transition: zero actions, re-prime history (review #3)
                self._last_action[:] = 0.0
                self._needs_history_init = True
                self.state = WALK
                self._vx_cmd = float(vx_cmd)
                self._yaw_rate_cmd = float(yaw_rate_cmd)
        elif self.state == WALK:
            self._vx_cmd = float(vx_cmd)
            self._yaw_rate_cmd = float(yaw_rate_cmd)

    # ════════════════════════════════════════════════════════
    # Per-step entry point
    # ════════════════════════════════════════════════════════
    def step(self):
        """Run one 100Hz tick. Reads sensors, builds obs, applies action."""
        now = time.monotonic()

        # ── Watchdog: detect main-loop stalls (review #6) ────────
        tick_dt = now - self._last_tick_time
        self._last_tick_time = now
        if self.state not in (DISARMED, READY) and tick_dt > C.WATCHDOG_TIMEOUT_S:
            print(f"[SAFETY] Watchdog: loop stall {tick_dt*1000:.0f} ms "
                  f"(limit {C.WATCHDOG_TIMEOUT_S*1000:.0f} ms), disarming.")
            self.disarm()
            return

        # Always refresh sensors
        self.imu.update()
        self.motors.read_state()

        if self.state == DISARMED:
            return

        # ── CAN drop tracking (review #7) ────────────────────────
        if self.state not in (DISARMED, READY):
            if self.motors.last_replies_parsed == 0:
                self._consec_can_drops += 1
            else:
                self._consec_can_drops = 0
            if self._consec_can_drops >= C.CAN_CONSEC_DROP_LIMIT:
                print(f"[SAFETY] {self._consec_can_drops} consecutive CAN "
                      f"drop ticks, disarming.")
                self.disarm()
                return

        # ── Safety: tilt check (review #6) ───────────────────────
        # Only in WALK (post-grace).  READY and STANDUP are expected
        # to be tilted (kneel is ~90°).
        if self.state == WALK:
            cos_tilt = self.imu.upright_cos()
            if cos_tilt < math.cos(C.MAX_TILT_RAD):
                print(f"[SAFETY] Tilt exceeded "
                      f"(cos={cos_tilt:.3f} < {math.cos(C.MAX_TILT_RAD):.3f}), "
                      f"disarming.")
                self.disarm()
                return

        # ── Safety: motor temperature (review #6) ────────────────
        if self.state != DISARMED:
            max_temp = float(np.max(self.motors.sim_joint_temperature))
            if max_temp >= C.MAX_MOTOR_TEMP_C:
                hot_idx = int(np.argmax(self.motors.sim_joint_temperature))
                print(f"[SAFETY] Motor {C.MOTOR_IDS[hot_idx]} at {max_temp:.1f}°C "
                      f"(limit {C.MAX_MOTOR_TEMP_C:.0f}°C), disarming.")
                self.disarm()
                return

        # ── READY state: firmly hold captured pose (kneel), no policy ──
        if self.state == READY:
            target = self._hold_pos.copy()
            self.motors.send_target(target, self._burst_kp, self._burst_kd,
                                    effort_limit=self._burst_effort)
            self._step += 1
            return

        # Advance gait clock
        self._gait_phase = (self._gait_phase
                             + 2.0 * math.pi * C.GAIT_FREQ_HZ * C.DT) % (2.0 * math.pi)

        # Standup countdown
        if self.state == STANDUP:
            self._grace_remaining = max(0, self._grace_remaining - 1)
            if self._grace_remaining == 0:
                print("[Controller] STANDUP grace expired → WALK")
                self.state = WALK
                self._spawn_yaw = self.imu.yaw()
                self._difficulty = 0.0
                self._env_init_jpos = self._default_jpos.copy()
                self._init_height = C.SPAWN_HEIGHT

        # Periodic yaw-reference resample (mirrors sim cmd_resample_steps)
        if (self.state == WALK
            and self._step > 0
            and self._step % C.YAW_REF_RESAMPLE_STEPS == 0):
            self._spawn_yaw = self.imu.yaw()

        # Build observation frame, stack history, run policy, apply action
        obs_frame = self._build_obs_frame()
        self._last_obs_frame = obs_frame
        obs_stacked = self._stack_history(obs_frame)
        action = self.policy.forward(obs_stacked)
        self._apply_action(action)
        self._last_action = action
        self._step += 1

    # ════════════════════════════════════════════════════════
    # Observation construction (matches rl_env._get_observations)
    # ════════════════════════════════════════════════════════
    def _build_obs_frame(self):
        """Build a single 36-dim observation frame (no history)."""
        proj_grav = self.imu.projected_gravity()           # 3
        ang_vel   = self.imu.ang_vel_body()                # 3
        lin_vel   = self.imu.lin_vel_body()                # 3 (zero by default)

        joint_pos_norm = (self.motors.sim_joint_pos
                          - self._default_jpos)            # 6
        joint_vel = self.motors.sim_joint_vel              # 6

        gait_sin = math.sin(self._gait_phase)
        gait_cos = math.cos(self._gait_phase)
        grace_norm = self._grace_remaining / C.MAX_GRACE_STEPS

        # Yaw error
        yaw_err = self.imu.yaw() - self._spawn_yaw
        yaw_err_sin = math.sin(yaw_err)
        yaw_err_cos = math.cos(yaw_err)

        frame = np.empty(C.OBS_FRAME_DIM, dtype=np.float32)
        i = 0
        frame[i:i+3] = proj_grav;        i += 3
        frame[i:i+3] = ang_vel;          i += 3
        frame[i:i+3] = lin_vel;          i += 3
        frame[i:i+6] = joint_pos_norm;   i += 6
        frame[i:i+6] = joint_vel;        i += 6
        frame[i:i+6] = self._last_action; i += 6
        frame[i] = gait_sin;             i += 1
        frame[i] = gait_cos;             i += 1
        frame[i] = self._difficulty;     i += 1
        frame[i] = grace_norm;           i += 1
        frame[i] = self._init_height;    i += 1
        frame[i] = self._vx_cmd;         i += 1
        frame[i] = self._yaw_rate_cmd;   i += 1
        frame[i] = yaw_err_sin;          i += 1
        frame[i] = yaw_err_cos;          i += 1
        assert i == C.OBS_FRAME_DIM, \
            f"obs frame dim mismatch: built {i}, expected {C.OBS_FRAME_DIM}"
        return frame

    def _stack_history(self, obs_frame):
        """Roll the history buffer and return a flat (OBS_DIM,) vector.

        On the first call after a reset / state transition,
        replicates obs_frame into all K slots — mirroring sim's
        _needs_history_init.  This prevents the OOD zero-history
        bug (review #2).
        """
        if self._needs_history_init:
            self._obs_history[:] = obs_frame[np.newaxis, :]
            self._needs_history_init = False
        else:
            self._obs_history = np.roll(self._obs_history, shift=1, axis=0)
            self._obs_history[0] = obs_frame
        return self._obs_history.reshape(-1)   # (180,)

    # ════════════════════════════════════════════════════════
    # Action execution (matches rl_env._apply_action)
    # ════════════════════════════════════════════════════════
    # Max position error sent to motor — clamps target to
    # actual ± this value. Prevents violent snaps if coupling
    # or state is wrong.
    _MAX_CMD_ERROR_RAD = 0.8   # loose enough for policy, tight enough for safety

    def _apply_action(self, action):
        # Slide action_center: kneel → stand as grace decays
        grace_frac = self._grace_remaining / max(C.MAX_GRACE_STEPS, 1)
        action_center = (self._env_init_jpos * grace_frac
                         + self._default_jpos * (1.0 - grace_frac))
        targets = action_center + action * self._action_scale

        # SAFETY: clamp target to actual ± max error
        actual = self.motors.sim_joint_pos
        for i in range(6):
            lo = actual[i] - self._MAX_CMD_ERROR_RAD
            hi = actual[i] + self._MAX_CMD_ERROR_RAD
            targets[i] = max(lo, min(hi, targets[i]))

        # Choose burst vs continuous PD gains + effort limits
        if self._grace_remaining > 0:
            kp, kd = self._burst_kp, self._burst_kd
            effort = self._burst_effort
        else:
            kp, kd = self._real_kp, self._real_kd
            effort = self._real_effort

        self.motors.send_target(targets, kp, kd, effort_limit=effort)

    # ════════════════════════════════════════════════════════
    # Internal reset
    # ════════════════════════════════════════════════════════
    def _reset_runtime_state(self, initial_height: float):
        self._step = 0
        self._last_action[:] = 0.0
        self._gait_phase = 0.0
        self._grace_remaining = 0
        self._difficulty = 0.0
        self._init_height = initial_height
        self._env_init_jpos = self._default_jpos.copy()
        self._spawn_yaw = self.imu.yaw()
        self._vx_cmd = 0.0
        self._yaw_rate_cmd = 0.0
        # History will be re-primed from first real frame (review #2)
        self._needs_history_init = True
        self._consec_can_drops = 0
        self._last_tick_time = time.monotonic()