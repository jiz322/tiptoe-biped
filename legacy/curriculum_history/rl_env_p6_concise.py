# CONCISE SUMMARY of rl_env_p6.py — for AI context-sharing only.
# Not runnable — method bodies are docstring-only stubs describing WHAT each
# piece does and WHY, not HOW (no tensor ops, no exact indices/formulas).
# See rl_env_p6.py for the authoritative implementation.

"""
RlEnv(DirectRLEnv) — implements the step/reset/reward/observation logic for
RlEnvCfg (see rl_env_cfg_p6_concise.py for the config-side design intent).
"""

# ── Forward-kinematics helpers ──
def _compute_fk_height(...):
    """Computes toe height above ground via URDF-derived forward kinematics
    (joint angles -> toe world position), used for reset pose correction
    (placing the robot at the right height for a given kneel/stand pose
    without relying on physics settling) and overshoot compensation."""

def _fixed_joint_transform(...): """Rigid offset transform for a fixed (non-actuated) URDF joint."""
def _make_Rz(...): """Rotation matrix about Z (yaw) for FK chain composition."""
def _batched_mat4_mul(...): """Batched 4x4 homogeneous transform multiply across all envs."""
def _slerp(...): """Spherical interpolation between two quaternions (used for kneel<->stand pose blending during reset)."""
def _yaw_from_quat(...): """Extracts yaw angle from a quaternion (for heading-anchor reward and command frame)."""


# ── Mirror / symmetry data augmentation ──
# _OBS_MIRROR_PERMUTATION / _OBS_MIRROR_SIGN, _ACT_MIRROR_PERMUTATION /
# _ACT_MIRROR_SIGN: fixed index-permutation + sign tables that map a
# sagittal-plane left-right reflection of the robot onto a permutation of
# the flat obs/action vectors (swap leg-1/leg-2 channels, negate the
# channels whose sign flips under a mirror, e.g. hip_roll, yaw-rate).
# Used by get_symmetric_states(), which rsl_rl's SymmetryCfg.data_augmentation_func
# calls to generate a mirrored (obs, action) pair for every real transition —
# doubles data efficiency and enforces gait symmetry as an inductive bias.
# A privileged-obs mirror table does the same for the asymmetric critic input.

def mirror_obs(obs): """Applies the obs mirror permutation+sign table."""
def mirror_action(act): """Applies the action mirror permutation+sign table."""
def _verify_mirror_correctness(): """Startup self-check: mirroring twice should return the identity; mirroring a known symmetric reference pose should be a no-op. Catches index-table bugs early."""


class RlEnv(DirectRLEnv):

    def __init__(self, cfg):
        """
        Sets up (beyond the DirectRLEnv base):
          - joint/body name -> index maps for the 6 actuated joints + IMU/foot bodies
          - per-env actuator parameter caches so DR multipliers can be applied
            without re-querying the sim every step
          - DR state tensors: one persistent per-env value per DR axis (actuator
            physics mults, control-loop delays, sensor biases, floor/mass/CoM,
            gravity-tilt direction+magnitude), resampled at each reset
          - actuator-lag 2nd-order prefilter state (per-joint per-env y, y' for
            the command prefilter described in cfg §11)
          - mirror permutation/sign tables (built once, verified via
            _verify_mirror_correctness)
          - pickup state buffers: phase enum, phase-entry step, lift target,
            hold duration, release latency, wobble phase
          - obs delay ring buffers (separate IMU-channel and joint-channel
            buffers, since they're sampled with independent per-env delays)
          - curriculum state (per-env standup difficulty, kneel-branch flag)
          - EMA accumulator buffers for windowed reward terms (tracking,
            anchoring, leg/angle symmetry, stance symmetry)
        """

    def _setup_scene(self):
        """Spawns robot + ground plane + envs per InteractiveSceneCfg; clones/replicates physics across num_envs."""

    def _get_toe_pos_w(self):
        """World-frame toe position per foot, via the FK helpers above — used for swing-clearance reward, no-flight/contact detection, and pickup's contact-based early release."""

    def _pre_physics_step(self, actions):
        """
        Per control-step (100 Hz), before physics substeps:
          - clip raw actions, apply action LPF (alpha=0.6), track raw
            (pre-clip) action stats for the raw_action reward term
          - handle standup-grace / post-grace-stabilize state transitions
            (disables certain rewards/terminations during the kneel->stand
            settling window)
          - apply pickup PD forces (position + orientation) if a pickup
            event is active for that env (see _apply_pickup_events)
          - resample vx/yaw-rate commands every cmd_resample_steps
          - advance the actuator-lag prefilter one step per joint (2nd-order
            ODE integration of the command signal, see cfg §11)
          - push the delayed action into the action-delay ring buffer
        """

    def _advance_motor_target(self):
        """Converts the (delayed, prefiltered) action into a DCMotor PD position target per joint."""

    def _apply_action(self):
        """Writes the current motor targets to the articulation's DCMotor actuators (torque computed internally by DCMotorCfg's speed-torque envelope + friction model)."""

    def _apply_training_pushes(self):
        """(push_enable=False currently) Would apply a randomly-timed external force impulse to the base to simulate a bump/disturbance."""

    def _get_observations(self):
        """
        Builds the 36-D per-frame policy observation:
          base orientation (proj gravity), base angular velocity, optionally
          base linear velocity (usually dropped out per cfg lin_vel_obs_dropout_prob),
          joint positions/velocities, gait clock phase, commanded vx/yaw-rate,
          previous action.
        IMU-channel fields and joint-channel fields are read from their own
        independently-delayed ring buffers (see cfg §12b), then per-step
        noise (cfg §12c) and per-episode bias (cfg §12d) are applied.
        The last obs_history_steps=5 frames are stacked (frame 0 = newest)
        into the flat 180-D actor input.
        """

    def _get_critic_observations(self):
        """
        Builds the 78-D privileged obs for the asymmetric critic: true base
        linear/angular velocity (un-delayed, un-noised), pickup
        phase/support-factor/force, and the full per-env DR parameter vector
        (actuator mults, delays, biases, floor/mass/CoM, gravity tilt,
        actuator-lag mults) — signals the actor is deliberately blind to.
        """

    def _get_rewards(self):
        """
        Gathers all reward inputs (base state, joint state, gait clock phase,
        commands, toe positions/contacts, action history, DR/pickup gates)
        and calls the JIT reward function compute_rewards_vx_yaw(...) which
        returns (total_reward, reward_components[38]) in the fixed order of
        _REWARD_COMPONENT_NAMES. Also updates the EMA accumulators (tracking,
        anchoring, leg/angle/stance symmetry) that feed back into next
        step's reward inputs, and logs ~150 diagnostic scalars to
        self.extras["log"] across categories: per-reward-component values,
        per-DR-axis correlation-with-episode-length, gait/idle stepping
        diagnostics (e.g. idle_toe_z_p10, idle_liftoff_rate), pickup phase
        stats, torque/action asymmetry between legs, clip fractions, etc.
        (Logging is intentionally exhaustive — see cfg dev notes: never
        remove a TensorBoard key once emitted.)
        """

    def _get_dones(self):
        """
        Terminates on: tilt beyond max_tilt_rad, height below min_height,
        any joint beyond its split limit, or (during pickup) support factor
        dropping below pickup_termination_support_threshold. Time-limit
        truncation at episode_length_s.
        """

    def _reset_idx(self, env_ids):
        """
        For each reset env:
          - samples standup difficulty / kneel-branch flag from the
            curriculum (uniform [0, alpha] when curriculum enabled)
          - interpolates joint pose + base quat between kneel pose and
            stand pose per that difficulty (via _slerp for orientation)
          - corrects spawn height using _compute_fk_height so the toes
            land at the right height for the interpolated pose, with an
            overshoot compensation term
          - applies init-state randomization: base roll/pitch noise
            (scaled by 1-difficulty), joint pos/vel noise, base lin/ang
            vel noise
          - resamples all per-episode DR (_sample_dr) and re-applies
            per-episode obs biases and pickup-on-spawn probability
          - resets EMA accumulators, action-delay/obs-delay ring buffers,
            actuator-lag prefilter state, pickup state machine to IDLE
            (or straight into LIFT if a spawn-pickup was rolled)
        """

    def _sample_commands(self, env_ids):
        """Samples vx / yaw-rate commands per cfg §5 ranges and zero-probabilities."""

    def _sample_dr_obs(self, env_ids):
        """Samples per-episode obs-side DR: sensor biases (joint_pos/proj_grav/ang_vel), IMU/joint obs delays, action delay."""

    def _sample_dr_physics(self, env_ids):
        """Samples per-episode physics-side DR: actuator mults (stiffness/damping/effort/armature/friction/voltage), actuator-lag omega/zeta mults, floor friction, mass mult, per-link CoM offset, gravity-tilt direction+magnitude."""

    def _write_dr_physics_to_sim(self, env_ids):
        """Writes the sampled per-env physics DR values into the actual PhysX actuator objects and rigid-body material/mass properties (index_select/index_copy across the GPU-resident per-env buffers)."""

    def _sample_dr(self, env_ids):
        """Convenience wrapper calling _sample_dr_obs + _sample_dr_physics + _write_dr_physics_to_sim, gated by cfg.dr_enable."""

    def _record_episode_lengths(self, env_ids):
        """Logs completed episode lengths, bucketed/correlated against each env's per-episode DR values, to produce the diagnostics/<dr>_eplen_corr series referenced in cfg §12's impact tags."""

    def _apply_pickup_events(self):
        """
        Advances the pickup state machine per env:
          IDLE -> (probability roll, or forced at reset) LIFT
          LIFT -> ramps a PD target for base z upward over pickup_lift_steps
          HOLD -> holds base z/orientation with small wobble for a randomly
                  sampled duration
          LOWER -> ramps base z target back down; contact-based early
                   release if a foot nears the ground
          RELEASE_TRANSITION -> linearly fades PD force authority to zero
                   over a sampled latency window (simulating gradual weight
                   transfer from hand to legs)
          -> IDLE
        Throughout LIFT/HOLD/LOWER, applies closed-loop PD forces on base
        z/xy and orientation (critically-damped 2nd order, torque-capped),
        blended by a support_factor that the critic obs and termination
        check also read.
        """

    def _trigger_initial_pickup(self, env_ids):
        """At reset, for envs rolling under pickup_initial_probability, starts them directly in LIFT with the (higher) initial lift-height range instead of IDLE."""


# ── Reward function (JIT-compiled for speed) ──
_REWARD_COMPONENT_NAMES = [
    # 38 named reward terms in fixed stack order, e.g.:
    "alive", "terminated", "standup_height", "standup_fall",
    "vel_tracking", "yaw_rate_tracking",
    "gait_step", "feet_air", "foot_swing", "foot_progress", "back_leg_step",
    "clock_contact", "swing_clearance",
    "no_flight", "vz_penalty", "stance_symmetry",
    "leg_symmetry", "angle_symmetry",
    "height", "balance", "knee_over_hip", "hip_roll_split", "upright",
    "position_anchor", "heading_anchor", "yaw_drift",
    "thermal", "energy",
    "action_rate", "action_smoothness", "action_snap", "action_magnitude", "raw_action",
    "joint_vel",
    # ... (see rl_env_p6.py for the exact full list / order)
    "gait_ref",  # slot 37 — the A/B fork term, see cfg
]

def compute_rewards_vx_yaw(...):
    """
    Pure-tensor function (no Python control flow per-env, torch.jit.script'd
    for step-time speed at 32768 envs) taking ~30 float reward-scale args
    (from cfg §9) and tensor args (base state, joint state, commands, gait
    clock, toe contacts, action history, EMA accumulators, gates for
    grace/pickup/idle-step). Computes each of the 38 reward components
    described in cfg §9's comments, stacks them in _REWARD_COMPONENT_NAMES
    order, sums with their scales into total_reward, and returns
    (total_reward, reward_components) — the components tensor is purely
    for diagnostic logging in _get_rewards, gradients only flow through
    total_reward.
    """
