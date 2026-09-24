"""
obs_parity_check.py — offline obs/action parity test between the real
deployment obs pipeline (balance_log.py / build_obs) and an independent
re-implementation of the sim's _get_observations (rl_env.py), replayed
against ONE recorded real-hardware log.

This does NOT run Isaac Lab. It re-derives the sim-formula 36D frame
from the RAW per-tick fields already in the log (proj_grav, ang_vel,
sim_joint_pos, sim_joint_vel, action, gait_phase) using a fresh port of
the training-side formula, then compares element-wise against the
obs actually fed to the policy on hardware. Divergence here means the
two obs pipelines disagree even with the SAME sensor stream — no DR
range fixes that.

Usage:
    python3 obs_parity_check.py <policy_basename>_log_<timestamp>.npz [--policy path.pt]

What it checks explicitly (the "classic survivors"):
  1. Per-channel element-wise obs diff, frame-by-frame (pre-history-stack).
  2. History-stack order/convention (frame 0 = newest) — rebuilds the
     180-D stack two ways (roll-then-write vs write-then-roll) and shows
     they are NOT interchangeable, so a mismatch here is a real bug class,
     not a false positive.
  3. Gait-clock phase convention (increment formula + zero-reference).
  4. Default-pose offset used for joint_pos_norm.
  5. Actions: both re-stacked obs vectors run through the same policy,
     actions diffed. This is the number that actually matters.
"""

import argparse
import math
import sys

import numpy as np

import config as C
from policy import Policy


OBS_PER_FRAME = C.OBS_FRAME_DIM          # 36
OBS_HISTORY_STEPS = C.OBS_HISTORY_STEPS  # 5
GAIT_FREQ_HZ = C.GAIT_FREQ_HZ
CONTROL_HZ = C.CONTROL_HZ
DEFAULT_JOINT_POS = np.array(C.DEFAULT_JOINT_POS, dtype=np.float32)
SPAWN_HEIGHT = C.SPAWN_HEIGHT

CHANNEL_LABELS = (
    ["proj_grav_x", "proj_grav_y", "proj_grav_z"]
    + ["ang_vel_x", "ang_vel_y", "ang_vel_z"]
    + ["lin_vel_x", "lin_vel_y", "lin_vel_z"]
    + [f"joint_pos_norm_{i}" for i in range(6)]
    + [f"joint_vel_{i}" for i in range(6)]
    + [f"last_action_{i}" for i in range(6)]
    + ["gait_sin", "gait_cos"]
    + ["difficulty", "grace_norm", "init_h", "vx_cmd", "yaw_rate_cmd",
       "yaw_err_sin", "yaw_err_cos"]
)
assert len(CHANNEL_LABELS) == OBS_PER_FRAME


def rebuild_sim_frame(proj_grav, ang_vel, joint_pos, joint_vel, last_action,
                       gait_phase):
    """Independent re-derivation of rl_env._get_observations' 36D frame,
    for the balance-only case (grace=0, difficulty=0, cmd=0, yaw_err=0),
    computed straight from the log's raw per-tick fields — NOT by calling
    build_obs() again. If this drifts from the logged 'obs' field, the two
    pipelines disagree on the SAME sensor stream.
    """
    joint_pos_norm = joint_pos - DEFAULT_JOINT_POS
    return np.concatenate([
        proj_grav.astype(np.float32),
        ang_vel.astype(np.float32),
        np.zeros(3, dtype=np.float32),          # lin_vel: dropout on hw
        joint_pos_norm.astype(np.float32),
        joint_vel.astype(np.float32),
        last_action.astype(np.float32),
        np.array([math.sin(gait_phase), math.cos(gait_phase)], dtype=np.float32),
        np.array([0.0, 0.0, SPAWN_HEIGHT, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    ]).astype(np.float32)


def stack_history_frame0_newest(frames):
    """frames: list of (36,) in chronological order (oldest..newest).
    Returns the (K, 36) history buffer using the SAME convention as
    rl_env._get_observations / build_obs: roll first, then write newest
    into slot 0. Equivalent to: buf[0]=frames[-1], buf[1]=frames[-2], ...
    """
    K = OBS_HISTORY_STEPS
    buf = np.zeros((K, OBS_PER_FRAME), dtype=np.float32)
    for f in frames:
        buf = np.roll(buf, shift=1, axis=0)
        buf[0] = f
    return buf


def stack_history_wrong_order(frames):
    """Deliberately-wrong alternative: frame 0 = OLDEST. Used only to prove
    the two conventions are numerically distinct (sanity check on the test
    itself, not on the real pipelines)."""
    K = OBS_HISTORY_STEPS
    buf = np.zeros((K, OBS_PER_FRAME), dtype=np.float32)
    recent = frames[-K:]
    for i, f in enumerate(recent):
        buf[i] = f
    return buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_path", help=".npz produced by balance_log.py ('b' to log)")
    ap.add_argument("--policy", default=None, help="override policy path for action diff")
    ap.add_argument("--atol", type=float, default=1e-5)
    args = ap.parse_args()

    data = np.load(args.log_path, allow_pickle=True)
    required = ["obs", "action", "sim_joint_pos", "sim_joint_vel",
                "proj_grav", "ang_vel", "gait_phase"]
    missing = [k for k in required if k not in data]
    if missing:
        print(f"[FATAL] log is missing fields: {missing}")
        sys.exit(1)

    n = len(data["time"])
    print(f"Loaded {args.log_path}: {n} ticks, policy={data['policy'].item() if 'policy' in data else '?'}")

    logged_obs = data["obs"].astype(np.float32)          # (n, 36) — deployment build_obs()
    logged_action = data["action"].astype(np.float32)    # (n, 6)  — policy output on hw
    sim_pos = data["sim_joint_pos"].astype(np.float32)
    sim_vel = data["sim_joint_vel"].astype(np.float32)
    proj_grav = data["proj_grav"].astype(np.float32)
    ang_vel = data["ang_vel"].astype(np.float32)
    gait_phase = data["gait_phase"].astype(np.float32)

    # ── 1. Per-frame element-wise diff (pre-history-stack) ─────────────
    # last_action fed into frame k is the action OUTPUT one tick earlier
    # (both build_obs and rl_env feed last_actions, not the current action).
    # balance_log.py logs gait_phase AFTER incrementing it for next tick
    # (build_obs -> policy.forward -> increment gait_phase -> log append),
    # so the value in the log is one step ahead of what build_obs actually
    # used. Un-increment it here so the rebuild uses the same phase the
    # deployment obs pipeline saw. This is a LOGGING-order artifact, not a
    # obs-pipeline bug -- confirmed by Step 3's per-tick increment matching
    # this diff almost exactly.
    # balance.py/balance_log.py advance gait_phase by a FIXED nominal step
    # (2*pi*GAIT_FREQ_HZ/CONTROL_HZ), NOT the measured loop dt -- see
    # balance.py:540-543. Un-increment with that same fixed constant.
    phase_step = 2.0 * math.pi * GAIT_FREQ_HZ / CONTROL_HZ
    gait_phase_used = (gait_phase - phase_step) % (2.0 * math.pi)

    rebuilt = np.zeros_like(logged_obs)
    prev_action = np.zeros(6, dtype=np.float32)
    for t in range(n):
        rebuilt[t] = rebuild_sim_frame(
            proj_grav[t], ang_vel[t], sim_pos[t], sim_vel[t],
            prev_action, gait_phase_used[t],
        )
        prev_action = logged_action[t]

    frame_diff = np.abs(rebuilt - logged_obs)
    max_per_channel = frame_diff.max(axis=0)
    worst = np.argsort(-max_per_channel)[:10]

    print("\n=== Step 1: per-frame obs parity (pre-history-stack) ===")
    print(f"  max abs diff overall: {frame_diff.max():.6e}")
    print(f"  mean abs diff overall: {frame_diff.mean():.6e}")
    if frame_diff.max() > args.atol:
        print(f"  [FAIL] exceeds atol={args.atol:.1e}. Worst channels:")
        for idx in worst:
            if max_per_channel[idx] > args.atol:
                print(f"    [{idx:2d}] {CHANNEL_LABELS[idx]:<18s}  max_diff={max_per_channel[idx]:.6e}")
    else:
        print(f"  [PASS] all channels within atol={args.atol:.1e}")

    # ── 2. History-stack convention sanity (proves the test itself works) ──
    print("\n=== Step 2: history-stack convention (frame 0 = newest) ===")
    frames_list = [rebuilt[t] for t in range(min(n, OBS_HISTORY_STEPS + 3))]
    if len(frames_list) >= OBS_HISTORY_STEPS:
        correct = stack_history_frame0_newest(frames_list)
        wrong = stack_history_wrong_order(frames_list)
        conv_diff = np.abs(correct - wrong).max()
        print(f"  correct-vs-reversed convention max diff: {conv_diff:.4f} "
              f"({'distinguishable — good, test has power' if conv_diff > 1e-6 else '[WARN] frames look identical, check log has motion'})")
    else:
        print("  [SKIP] log shorter than history length")

    # ── 3. Gait phase convention check ──────────────────────────────────
    # balance.py advances gait_phase by a FIXED nominal step
    # (2*pi*GAIT_FREQ_HZ/CONTROL_HZ) every tick, not measured loop dt
    # (see balance.py:540-543) -- so the expectation here must use the
    # same fixed constant, not the wall-clock dt from the log.
    print("\n=== Step 3: gait-clock phase convention ===")
    if n >= 2:
        expected_step = 2.0 * math.pi * GAIT_FREQ_HZ / CONTROL_HZ
        observed_step = np.diff(np.unwrap(gait_phase[:min(n, 50)])).mean()
        print(f"  expected Δphase/tick (2π·{GAIT_FREQ_HZ}Hz/{CONTROL_HZ}Hz, fixed nominal) = {expected_step:.5f} rad")
        print(f"  observed Δphase/tick (from log)                                        = {observed_step:.5f} rad")
        if abs(expected_step - observed_step) > 1e-3:
            print(f"  [FAIL] gait phase increment mismatch — check GAIT_FREQ_HZ / CONTROL_HZ "
                  f"between config.py and rl_env_cfg_p6.py (cfg.gait_freq).")
        else:
            print(f"  [PASS]")

    # ── 4. Default-pose offset used for joint_pos_norm ──────────────────
    print("\n=== Step 4: default-pose offset (joint_pos_norm reference) ===")
    print(f"  config.py DEFAULT_JOINT_POS = {DEFAULT_JOINT_POS.tolist()}")
    print("  [ACTION REQUIRED] this must equal self.robot.data.default_joint_pos"
          " in rl_env.py, which comes from the USD/robot_cfg default — NOT")
    print("  necessarily this list. config.py:81-88 has a live "
          "'# do we still need this???' comment on this exact constant —")
    print("  verify it against the actual sim default_joint_pos, don't assume it's kept in sync.")

    # ── 5. Action parity: run BOTH obs pipelines through the policy ─────
    print("\n=== Step 5: action parity through the policy ===")
    policy_path = args.policy or (data["policy"].item() if "policy" in data else C.POLICY_PATH)
    policy = Policy(policy_path)

    # (a) replay logged obs verbatim, re-stacked EXACTLY the way
    #     balance_log.py stacks it (first frame replicated into all K slots,
    #     then roll-and-write-slot-0 per tick). logged_obs is the single
    #     36-D frame per tick, not the 180-D policy input -- must stack
    #     before calling forward(). Sanity: should reproduce logged_action
    #     near-exactly since this is the SAME frames through the SAME
    #     stacking logic build_obs used live.
    replay_action = np.zeros_like(logged_action)
    hist_a = np.zeros((OBS_HISTORY_STEPS, OBS_PER_FRAME), dtype=np.float32)
    hist_a[:] = logged_obs[0]
    for t in range(n):
        if t == 0:
            hist_a[:] = logged_obs[0]
        else:
            hist_a = np.roll(hist_a, shift=1, axis=0)
            hist_a[0] = logged_obs[t]
        replay_action[t] = policy.forward(hist_a.reshape(-1))
    replay_diff = np.abs(replay_action - logged_action).max()
    print(f"  (a) logged_obs (re-stacked) -> policy vs logged_action: max diff = {replay_diff:.6e}  "
          f"({'PASS' if replay_diff < args.atol else '[FAIL] policy.forward is non-deterministic or POLICY_PATH mismatch'})")

    # (b) independently-rebuilt sim-formula frames, stacked K deep with the
    #     frame0=newest convention, fed through the SAME policy
    rebuilt_action = np.zeros_like(logged_action)
    hist = np.zeros((OBS_HISTORY_STEPS, OBS_PER_FRAME), dtype=np.float32)
    hist[:] = rebuilt[0]  # replicate first frame into all slots (matches _reset_idx priming)
    for t in range(n):
        hist = np.roll(hist, shift=1, axis=0)
        hist[0] = rebuilt[t]
        stacked = hist.reshape(-1)
        rebuilt_action[t] = policy.forward(stacked)

    action_diff = np.abs(rebuilt_action - logged_action)
    print(f"  (b) rebuilt_obs -> policy vs logged_action: max diff = {action_diff.max():.6e}, "
          f"mean = {action_diff.mean():.6e}")
    per_joint = action_diff.max(axis=0)
    joint_names = ["hr1", "hp1", "kn1", "hr2", "hp2", "kn2"]
    for j, name in zip(range(6), joint_names):
        flag = "" if per_joint[j] <= args.atol else "  <-- DIVERGES"
        print(f"      {name}: max_diff={per_joint[j]:.6e}{flag}")

    if action_diff.max() <= args.atol:
        print("\n[RESULT] Obs parity PASSES: rebuilt sim-formula obs produces bit-close "
              "actions vs the real deployment pipeline. Bad standing behavior is NOT "
              "explained by an obs-construction bug — look at DR range / sysid instead.")
    else:
        print("\n[RESULT] Obs parity FAILS. Fix the obs pipeline before touching DR ranges "
              "— no domain randomization width fixes a systematic obs mismatch.")


if __name__ == "__main__":
    main()
