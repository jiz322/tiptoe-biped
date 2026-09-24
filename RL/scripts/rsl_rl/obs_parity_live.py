"""obs_parity_live.py — real obs-parity test: drives the ACTUAL Isaac Lab
env's _get_observations() from a recorded hardware log, instead of a
hand-typed re-derivation of the formula (which is what
control/obs_parity_check.py does, and why that script is circular).

Usage (from an env with isaaclab, e.g. `conda activate env_isaaclab`):

    python RL/scripts/rsl_rl/obs_parity_live.py \
        --log control/model_4150_log_20260803_140243.npz \
        --policy control/model_4150.pt \
        --num_envs 1 --headless

Checks performed, in order of "cheapest and most likely to be the bug":

  0. CONSTANT PARITY — config.py vs the live sim cfg: gait_freq, control_hz,
     obs_history_steps, action_scales, default_joint_pos, obs dims. A mismatch
     here (esp. gait_freq / control_hz) silently rotates the gait clock and
     mimics bad DR forever.

  1. LOG SELF-CONSISTENCY — the recorded 36D obs vs the raw sensor channels
     recorded alongside it, including the last_action alignment
     (obs[t][21:27] == action[t-1]) and the gait-phase convention
     (logged gait_phase is POST-increment, obs used phase - step).
     This runs before Isaac and needs no GPU.

  2. LIVE OBS PARITY — for every tick, overwrite the env's simulated state
     (joint_pos, joint_vel, IMU proj_grav/ang_vel, last_actions, gait_phase,
     difficulty=0, grace=0, cmd=0, yaw_err=0, init_h=SPAWN_HEIGHT) with the
     recorded values, then call the REAL env._get_observations(). No formula
     is re-typed; this is the training code path itself. Diff frame 0 of the
     returned 180D stack against the logged 36D deployment obs, element-wise.

  3. STACKING-ORDER PARITY — diff the env's OWN 180D history stack against a
     hardware-style np.roll re-stack of the same frames. This is the only
     non-circular test of "frame 0 = newest" on both sides.

  4. POLICY SELF-CONSISTENCY — run the logged obs history through Policy()
     and compare to the logged actions. If this fails, policy LOADING is
     broken (rebuilt-actor activation, normalizer, clamp), not the obs.
     Isolating this first is what makes step 5's verdict meaningful.

  5. ACTION PARITY — feed both the env-produced 180D stack and the logged
     180D obs through the same policy and diff actions.

  6. (--dr-delay) Repeat step 2 with the trained fixed IMU/joint obs delay
     enabled (dr_imu_delay / dr_joint_delay from the cfg) and ALL obs noise
     and per-episode biases forced to zero, to quantify how much of a
     residual is the delay model rather than a formula mismatch. build_obs()
     on hardware has no delay model, so this is expected to differ — the
     point is to measure it, not to pass it.
"""

import argparse
import math
import sys

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--log", type=str, required=True, help="npz from balance_log.py")
parser.add_argument("--policy", type=str, default=None, help="policy path override")
parser.add_argument("--task", type=str, default="Template-standup_curriculum-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--atol", type=float, default=1e-5)
parser.add_argument(
    "--dr-delay", action="store_true", default=False,
    help="Also run a second pass with the trained fixed IMU/joint obs delay "
         "enabled (noise+biases zeroed), to isolate its effect from a real "
         "formula mismatch.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# ── Channel layout of the 36D frame (must match rl_env_p6 _get_observations) ──
CH_SLICES = [
    ("proj_grav",    0,  3),
    ("ang_vel",      3,  6),
    ("lin_vel",      6,  9),
    ("joint_pos_n",  9, 15),
    ("joint_vel",   15, 21),
    ("last_actions",21, 27),
    ("gait_sin_cos",27, 29),
    ("difficulty",  29, 30),
    ("grace_norm",  30, 31),
    ("init_h",      31, 32),
    ("vx_cmd",      32, 33),
    ("yaw_rate_cmd",33, 34),
    ("yaw_err_sin", 34, 35),
    ("yaw_err_cos", 35, 36),
]
JOINT_NAMES = ["hr1", "hp1", "kn1", "hr2", "hp2", "kn2"]


def load_log(path):
    data = np.load(path, allow_pickle=True)
    required = ["obs", "action", "sim_joint_pos", "sim_joint_vel",
                "proj_grav", "ang_vel", "gait_phase"]
    missing = [k for k in required if k not in data.files]
    if missing:
        print(f"[FATAL] log is missing fields: {missing}")
        sys.exit(1)
    return {k: data[k] for k in data.files}


# ══════════════════════════════════════════════════════════════════════
# STEP 1 — log self-consistency. Pure numpy, runs before Isaac starts.
# ══════════════════════════════════════════════════════════════════════

def check_log_self_consistency(log, C, atol):
    """Does the recorded 36D obs actually equal what build_obs() would have
    produced from the raw sensor channels recorded on the same tick?

    This catches history/alignment/phase bugs on the DEPLOYMENT side without
    needing the sim at all, and it pins down the two conventions the live
    pass has to reproduce: last_action lag and gait-phase pre/post increment.
    """
    print("\n=== [1] LOG SELF-CONSISTENCY (deployment side, no sim needed) ===")
    obs = log["obs"].astype(np.float32)
    act = log["action"].astype(np.float32)
    n = len(obs)
    default = np.asarray(C.DEFAULT_JOINT_POS, dtype=np.float32)

    passed = True

    def cmp(label, a, b, tol=atol):
        nonlocal passed
        d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
        good = d.max() <= tol
        print(f"  {'[ OK ]' if good else '[FAIL]'} {label:34s} "
              f"max={d.max():.3e} mean={d.mean():.3e}")
        passed = passed and good
        return d.max()

    cmp("obs[0:3]  == proj_grav",   obs[:, 0:3],  log["proj_grav"])
    cmp("obs[3:6]  == ang_vel",     obs[:, 3:6],  log["ang_vel"])
    cmp("obs[6:9]  == 0 (lin_vel)", obs[:, 6:9],  np.zeros((n, 3)))
    cmp("obs[9:15] == jpos-default", obs[:, 9:15],
        log["sim_joint_pos"].astype(np.float32) - default)
    cmp("obs[15:21] == joint_vel",  obs[:, 15:21], log["sim_joint_vel"])

    # last_actions lag: obs at tick t must carry the action emitted at t-1.
    prev = np.vstack([np.zeros((1, 6), np.float32), act[:-1]])
    d_lag1 = cmp("obs[21:27] == action[t-1]", obs[:, 21:27], prev)
    d_lag0 = np.abs(obs[:, 21:27] - act).max()
    if d_lag1 > atol:
        print(f"         (for reference, vs action[t] (no lag): {d_lag0:.3e})")
        if d_lag0 <= atol:
            print("         >>> log used action[t], i.e. NO lag. The live pass's "
                  "prev_action convention must match this.")

    # Gait phase: balance_log advances the clock AFTER building obs, so the
    # logged gait_phase is one step ahead of the phase baked into obs.
    gp = log["gait_phase"].astype(np.float64)
    step = 2.0 * math.pi * C.GAIT_FREQ_HZ / C.CONTROL_HZ
    pre = (gp - step) % (2.0 * math.pi)
    d_pre = cmp("obs[27:29] == sin/cos(gp - step)",
                obs[:, 27:29], np.stack([np.sin(pre), np.cos(pre)], axis=-1))
    if d_pre > atol:
        d_post = np.abs(
            obs[:, 27:29] - np.stack([np.sin(gp), np.cos(gp)], axis=-1)
        ).max()
        print(f"         (vs sin/cos(gp) directly: {d_post:.3e})")
        if d_post <= atol:
            print("         >>> logged gait_phase is PRE-increment, not post. "
                  "Drop the -step correction in the live pass.")

    # Constant tail.
    tail_expect = np.array(
        [0.0, 0.0, C.SPAWN_HEIGHT, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
    )
    cmp("obs[29:36] == const tail", obs[:, 29:36],
        np.tile(tail_expect, (n, 1)))
    uniq = np.unique(obs[:, 29:36], axis=0)
    if len(uniq) != 1:
        print(f"         [WARN] tail is not constant across the run "
              f"({len(uniq)} distinct rows) — the log mixes regimes.")

    return passed, step


# ══════════════════════════════════════════════════════════════════════
# STEP 0 — constant parity between config.py and the live sim cfg
# ══════════════════════════════════════════════════════════════════════

def check_constants(base_env, C):
    """The classic silent killers: a gait_freq or control_hz that differs
    between sim and hardware rotates the gait clock every tick; an
    obs_history_steps mismatch reshapes the policy input."""
    print("\n=== [0] CONSTANT PARITY (config.py vs live sim cfg) ===")
    cfg = base_env.cfg
    passed = True

    sim_control_hz = 1.0 / (cfg.sim.dt * cfg.decimation)
    checks = [
        ("gait_freq",         cfg.gait_freq,          C.GAIT_FREQ_HZ),
        ("control_hz",        sim_control_hz,         float(C.CONTROL_HZ)),
        ("obs_history_steps", cfg.obs_history_steps,  C.OBS_HISTORY_STEPS),
        ("obs_frame_dim",     cfg.observation_space // cfg.obs_history_steps,
                              C.OBS_FRAME_DIM),
        ("obs_dim",           cfg.observation_space,  C.OBS_DIM),
    ]
    for name, sim_v, cfg_v in checks:
        good = abs(float(sim_v) - float(cfg_v)) < 1e-9
        passed = passed and good
        print(f"  {'[ OK ]' if good else '[FAIL]'} {name:20s} "
              f"sim={sim_v}  config.py={cfg_v}")

    # default_joint_pos is what _get_observations actually subtracts.
    joint_ids = base_env._joint_ids
    sim_default = base_env._default_joint_pos[0, joint_ids].cpu().numpy()
    cfg_default = np.asarray(C.DEFAULT_JOINT_POS, dtype=np.float32)
    dd = np.abs(sim_default - cfg_default)
    good = dd.max() <= 1e-6
    passed = passed and good
    print(f"  {'[ OK ]' if good else '[FAIL]'} {'default_joint_pos':20s} "
          f"max_diff={dd.max():.3e}")
    if not good:
        print(f"         sim:    {sim_default}")
        print(f"         config: {cfg_default}")
        print("         >>> joint_pos_norm carries a constant offset. This alone "
              "shifts every joint channel the policy sees.")

    # action_scales: not an obs channel, but a mismatch here means the same
    # action produces a different joint target on hardware than in training.
    sim_scales = _resolve_action_scales(base_env)
    if sim_scales is not None:
        cs = np.asarray(C.ACTION_SCALES, dtype=np.float32)
        ds = np.abs(sim_scales - cs)
        good = ds.max() <= 1e-6
        print(f"  {'[ OK ]' if good else 'WARN '} {'action_scales':20s} "
              f"max_diff={ds.max():.3e}   (not an obs channel)")
        if not good:
            print(f"         sim:    {sim_scales}")
            print(f"         config: {cs}")
    else:
        print("  [SKIP] action_scales — could not locate on the env")

    return passed


def _resolve_action_scales(base_env):
    for attr in ("_action_scales", "_action_scale"):
        if hasattr(base_env, attr):
            v = getattr(base_env, attr)
            try:
                v = v.detach().cpu().numpy().reshape(-1)
            except AttributeError:
                v = np.asarray(v, dtype=np.float32).reshape(-1)
            if v.size == 6:
                return v.astype(np.float32)
    for attr in ("action_scales", "action_scale"):
        if hasattr(base_env.cfg, attr):
            v = np.asarray(getattr(base_env.cfg, attr), dtype=np.float32).reshape(-1)
            if v.size == 6:
                return v
    return None


# ══════════════════════════════════════════════════════════════════════
# STEP 2/6 — replay the log through the REAL _get_observations()
# ══════════════════════════════════════════════════════════════════════

def replay(base_env, log, C, gait_phase_used, torch, label):
    """Overwrite sim state from the log tick-by-tick and collect the env's
    own obs output. Returns (frames_36, stacks_180)."""
    joint_ids = base_env._joint_ids
    device = base_env.device
    n = len(log["obs"])

    sim_pos = log["sim_joint_pos"].astype(np.float32)
    sim_vel = log["sim_joint_vel"].astype(np.float32)
    proj_grav = log["proj_grav"].astype(np.float32)
    ang_vel = log["ang_vel"].astype(np.float32)
    logged_action = log["action"].astype(np.float32)

    obs_dim = base_env.cfg.observation_space
    frames = np.zeros((n, C.OBS_FRAME_DIM), dtype=np.float32)
    stacks = np.zeros((n, obs_dim), dtype=np.float32)

    # Replay from a clean history/delay state so frame 0 of the run is
    # primed the same way hardware primes it (replicate first real frame).
    base_env._needs_history_init[:] = True
    if hasattr(base_env, "_obs_delay_buffer"):
        base_env._obs_delay_buffer[:] = 0.0

    prev_action = np.zeros(6, dtype=np.float32)
    zeros3 = torch.zeros(3, device=device)

    print(f"\n  replaying {n} ticks through _get_observations() [{label}] ...")
    for t in range(n):
        with torch.no_grad():
            # ── joints ──────────────────────────────────────────────
            # robot.data.joint_pos is a lazily-refreshed cached buffer; it
            # only re-reads physx when _sim_timestamp advances, and we never
            # step the sim, so an in-place write here survives until
            # _get_observations() reads it.
            jp = base_env.robot.data.joint_pos
            jv = base_env.robot.data.joint_vel
            for i, jid in enumerate(joint_ids):
                # sim_joint_pos in the log is already the ABSOLUTE sim-frame
                # angle (bridge.sim_joint_pos), the same quantity
                # robot.data.joint_pos holds. Do NOT re-add default_joint_pos
                # here — that would double-count it and manufacture an offset.
                jp[0, jid] = float(sim_pos[t, i])
                jv[0, jid] = float(sim_vel[t, i])

            # ── IMU ─────────────────────────────────────────────────
            # _get_observations computes quat_apply_inverse(imu_quat, X).
            # Force imu_quat = identity and put the already-body-frame logged
            # values straight into the world-frame fields, since
            # quat_apply_inverse(identity, v) == v. body_quat_w /
            # body_ang_vel_w are tensor VIEWS into the cached pose/vel
            # buffers, so these writes land in the underlying storage.
            q = base_env.robot.data.body_quat_w[:, base_env._imu_idx, :]
            q[:, 0] = 1.0
            q[:, 1:] = 0.0
            base_env.robot.data.body_ang_vel_w[:, base_env._imu_idx, :] = (
                torch.as_tensor(ang_vel[t], device=device)
            )
            base_env.robot.data.body_lin_vel_w[:, base_env._imu_idx, :] = zeros3
            base_env._gravity_vec[:] = torch.as_tensor(proj_grav[t], device=device)

            # ── bookkeeping ─────────────────────────────────────────
            base_env._last_actions[:] = torch.as_tensor(prev_action, device=device)
            base_env._gait_phase[:] = float(gait_phase_used[t])
            base_env._env_difficulty[:] = 0.0
            base_env._grace_remaining[:] = 0
            base_env._vx_cmd[:] = 0.0
            base_env._yaw_rate_cmd[:] = 0.0
            # yaw_error = current_yaw - spawn_yaw; pin spawn to current so the
            # pair is (sin, cos) = (0, 1), matching build_obs's hardcoded tail.
            base_env._spawn_yaw[:] = _yaw_from_quat(base_env.robot.data.root_quat_w)
            # build_obs zeroes lin_vel unconditionally (dropout=1.0 on hw).
            base_env._lin_vel_obs_keep[:] = 0.0
            # build_obs hardcodes init_h = SPAWN_HEIGHT; env.reset() instead
            # sets _initial_height from an FK-computed per-episode value.
            base_env._initial_height[:] = float(C.SPAWN_HEIGHT)

            out = base_env._get_observations()
            stacked = out["policy"][0].detach().cpu().numpy()
            stacks[t] = stacked
            frames[t] = stacked[:C.OBS_FRAME_DIM]   # frame 0 = newest

        prev_action = logged_action[t]

    return frames, stacks


def report_frame_diff(live, logged, atol, skip=0, header=""):
    print(f"\n{header}")
    if skip:
        print(f"  (skipping first {skip} ticks — buffer warm-up)")
    a, b = live[skip:], logged[skip:]
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    print(f"  max abs diff:  {d.max():.6e}")
    print(f"  mean abs diff: {d.mean():.6e}")

    per_ch = d.max(axis=0)
    print(f"  {'channel group':16s} {'max':>12s} {'mean':>12s}")
    for name, lo, hi in CH_SLICES:
        seg = d[:, lo:hi]
        flag = "" if seg.max() <= atol else "   <-- DIVERGES"
        print(f"  {name:16s} {seg.max():12.4e} {seg.mean():12.4e}{flag}")

    bad = np.where(per_ch > atol)[0]
    if len(bad):
        print(f"  [FAIL] {len(bad)} of {live.shape[1]} channels exceed "
              f"atol={atol:.1e}. Worst 12:")
        for idx in np.argsort(-per_ch)[:12]:
            if per_ch[idx] > atol:
                ch = next((nm for nm, lo, hi in CH_SLICES if lo <= idx < hi), "?")
                t_worst = int(np.argmax(d[:, idx]))
                print(f"    [{idx:2d}] {ch:14s} max={per_ch[idx]:.4e} "
                      f"@tick {t_worst + skip}  live={a[t_worst, idx]:+.6f} "
                      f"logged={b[t_worst, idx]:+.6f}")
        return False
    print(f"  [PASS] all channels within atol={atol:.1e}")
    return True


def main():
    log = load_log(args_cli.log)

    # ── STEP 1: free, deployment-side, before Isaac gets involved ──
    log_ok, phase_step = check_log_self_consistency(log, C, args_cli.atol)

    # ── build the real env ─────────────────────────────────────────
    spec = gym.spec(args_cli.task)
    env_cfg = spec.kwargs["env_cfg_entry_point"]
    if isinstance(env_cfg, str):
        mod_name, cls_name = env_cfg.split(":")
        import importlib
        env_cfg = getattr(importlib.import_module(mod_name), cls_name)()

    env_cfg.scene.num_envs = args_cli.num_envs
    # Baseline pass: build_obs on hardware has no obs-noise, no obs-delay and
    # no per-episode bias, so DR off is the correct apples-to-apples setting.
    env_cfg.dr_enable = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    env.reset()

    const_ok = check_constants(base_env, C)

    n = len(log["obs"])
    logged_obs = log["obs"].astype(np.float32)
    logged_action = log["action"].astype(np.float32)
    gp = log["gait_phase"].astype(np.float64)
    gait_phase_used = (gp - phase_step) % (2.0 * math.pi)

    # ── STEP 2: live obs parity, DR off ────────────────────────────
    frames, stacks = replay(base_env, log, C, gait_phase_used, torch, "dr_enable=False")
    obs_ok = report_frame_diff(
        frames, logged_obs, args_cli.atol,
        header="=== [2] LIVE OBS PARITY (real rl_env._get_observations, dr_enable=False) ===",
    )

    # ── STEP 3: stacking-order parity ──────────────────────────────
    # Rebuild a 180D stack the way hardware does (np.roll, frame 0 newest,
    # first frame replicated into all K slots) and diff against the env's own
    # history buffer. Any diff here IS a history-order mismatch.
    print("\n=== [3] HISTORY STACKING-ORDER PARITY (env buffer vs hw np.roll) ===")
    K = C.OBS_HISTORY_STEPS
    hw_stacks = np.zeros_like(stacks)
    hist = np.zeros((K, C.OBS_FRAME_DIM), dtype=np.float32)
    for t in range(n):
        if t == 0:
            hist[:] = frames[0]
        else:
            hist = np.roll(hist, shift=1, axis=0)
            hist[0] = frames[t]
        hw_stacks[t] = hist.reshape(-1)
    ds = np.abs(stacks.astype(np.float64) - hw_stacks.astype(np.float64))
    stack_ok = ds.max() <= args_cli.atol
    print(f"  max abs diff: {ds.max():.6e}   mean: {ds.mean():.6e}")
    if stack_ok:
        print(f"  [PASS] env history order == hardware order (frame 0 = newest).")
    else:
        per_frame = [ds[:, k * 36:(k + 1) * 36].max() for k in range(K)]
        print("  [FAIL] history order differs. Per-frame-slot max diff:")
        for k, v in enumerate(per_frame):
            print(f"    frame {k} (newest=0): {v:.4e}")
        print("  >>> If the diff is small at frame 0 and grows with k, the two "
              "sides roll in opposite directions.")

    # ── load the policy ────────────────────────────────────────────
    policy_path = args_cli.policy or (
        str(log["policy"]) if "policy" in log else C.POLICY_PATH
    )
    if not os.path.isabs(policy_path) and not os.path.exists(policy_path):
        policy_path = os.path.join(_CONTROL_DIR, policy_path)
    if "policy" in log:
        logged_name = os.path.basename(str(log["policy"]))
        used_name = os.path.basename(policy_path)
        if logged_name != used_name:
            print(f"\n  [WARN] log was recorded with '{logged_name}' but this "
                  f"run uses '{used_name}'. Action parity is meaningless "
                  f"across different checkpoints.")
    print()
    policy = Policy(policy_path)

    def stack_from(frames_36):
        h = np.zeros((K, C.OBS_FRAME_DIM), dtype=np.float32)
        out = np.zeros((len(frames_36), K * C.OBS_FRAME_DIM), dtype=np.float32)
        for t in range(len(frames_36)):
            if t == 0:
                h[:] = frames_36[0]
            else:
                h = np.roll(h, shift=1, axis=0)
                h[0] = frames_36[t]
            out[t] = h.reshape(-1)
        return out

    # ── STEP 4: policy self-consistency ────────────────────────────
    # Logged obs -> this Policy() -> should reproduce the logged actions.
    # If this fails, the loader is broken and step 5 tells you nothing.
    print("\n=== [4] POLICY SELF-CONSISTENCY (logged obs -> logged action) ===")
    logged_stacks = stack_from(logged_obs)
    replay_action = np.stack([policy.forward(logged_stacks[t]) for t in range(n)])
    dp = np.abs(replay_action.astype(np.float64) - logged_action.astype(np.float64))
    policy_ok = dp.max() <= args_cli.atol
    print(f"  max abs diff: {dp.max():.6e}   mean: {dp.mean():.6e}")
    for j, name in enumerate(JOINT_NAMES):
        flag = "" if dp[:, j].max() <= args_cli.atol else "   <-- DIVERGES"
        print(f"    {name}: max={dp[:, j].max():.4e}{flag}")
    if policy_ok:
        print("  [PASS] this Policy() reproduces hardware actions bit-close. "
              "Any action diff below is caused by the OBS, not the loader.")
    else:
        print("  [FAIL] the loader itself disagrees with what ran on hardware "
              "(activation / normalizer / clamp / checkpoint mismatch).")
        print("  >>> Fix this BEFORE interpreting step 5.")

    # ── STEP 5: action parity ──────────────────────────────────────
    print("\n=== [5] ACTION PARITY (env-produced obs -> policy) ===")
    live_action = np.stack([policy.forward(stacks[t]) for t in range(n)])
    da = np.abs(live_action.astype(np.float64) - logged_action.astype(np.float64))
    action_ok = da.max() <= args_cli.atol
    print(f"  max abs diff: {da.max():.6e}   mean: {da.mean():.6e}")
    for j, name in enumerate(JOINT_NAMES):
        flag = "" if da[:, j].max() <= args_cli.atol else "   <-- DIVERGES"
        print(f"    {name}: max={da[:, j].max():.4e}{flag}")

    # ── STEP 6: optional delay-DR pass ─────────────────────────────
    delay_max = None
    if args_cli.dr_delay:
        print("\n=== [6] DELAY-DR PASS (obs delay ON, noise+biases forced to 0) ===")
        cfg = base_env.cfg
        imu_d, joint_d = cfg.dr_imu_delay_max, cfg.dr_joint_delay_max
        # Isolate delay: zero every other DR obs term so the only difference
        # from pass 2 is the history-buffer tap.
        for f in ("dr_obs_noise_proj_grav", "dr_obs_noise_ang_vel",
                  "dr_obs_noise_joint_vel", "dr_obs_noise_joint_pos"):
            if hasattr(cfg, f):
                setattr(cfg, f, 0.0)
        base_env._dr_proj_grav_bias[:] = 0.0
        base_env._dr_ang_vel_bias[:] = 0.0
        base_env._dr_joint_pos_bias[:] = 0.0
        base_env._env_imu_delay[:] = imu_d
        base_env._env_joint_delay[:] = joint_d
        cfg.dr_enable = True
        print(f"  imu_delay={imu_d} steps, joint_delay={joint_d} steps, "
              f"max_obs_delay={base_env._max_obs_delay}")

        d_frames, _ = replay(base_env, log, C, gait_phase_used, torch, "delay ON")
        skip = int(base_env._max_obs_delay) + 1
        report_frame_diff(
            d_frames, logged_obs, args_cli.atol, skip=skip,
            header="--- delay-DR obs diff vs logged (EXPECTED to differ) ---",
        )
        dd = np.abs(d_frames[skip:] - logged_obs[skip:])
        delay_max = dd.max()
        print(f"\n  Interpretation: build_obs() has no delay model, so this "
              f"residual ({delay_max:.4e}) is the SIZE of the delay DR the "
              f"policy was trained to tolerate, not a bug. Compare it to "
              f"step 2's residual: if step 2 passed and this is large, the "
              f"delay is the dominant train/deploy gap.")
        cfg.dr_enable = False

    # ── verdict ────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("SUMMARY")
    print("=" * 68)
    for name, ok in [
        ("[0] constant parity (config.py vs sim cfg)", const_ok),
        ("[1] log self-consistency (deployment side)", log_ok),
        ("[2] live obs parity (real env code path)", obs_ok),
        ("[3] history stacking order", stack_ok),
        ("[4] policy loader self-consistency", policy_ok),
        ("[5] action parity", action_ok),
    ]:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if delay_max is not None:
        print(f"  ----  [6] delay-DR residual = {delay_max:.4e} (informational)")

    all_ok = const_ok and log_ok and obs_ok and stack_ok and policy_ok and action_ok
    print()
    if all_ok:
        print("[RESULT] Obs parity PASSES against the actual training env code path.")
        print("         Obs construction is ruled out. The sim2real gap is")
        print("         elsewhere: dynamics/actuator model, latency, or DR ranges.")
    else:
        print("[RESULT] Obs parity FAILS. There is a genuine mismatch between")
        print("         build_obs() and rl_env._get_observations(). No DR range")
        print("         fixes this — fix the obs first.")

    env.close()
    return 0 if all_ok else 1


# ── everything below needs the sim app running ──────────────────────
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import RL.tasks  # noqa: F401  registers Template-standup_curriculum-v0
from RL.tasks.direct.standup_curriculum.rl_env_p6 import _yaw_from_quat  # noqa: E402

# AppLauncher prepends Isaac Sim's bundled site-packages to sys.path (which
# includes a cv2-internal config.py), so a bare `import config` here — or
# inside policy.py, which does the same — resolves to the WRONG module.
# Load control/config.py and control/policy.py by explicit file path and
# seed sys.modules so policy.py's own `import config as C` finds ours.
import importlib.util
import os

_CONTROL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "control")
)


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_CONTROL_DIR, filename)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load_module("config", "config.py")
Policy = _load_module("policy", "policy.py").Policy


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    sys.exit(code)
