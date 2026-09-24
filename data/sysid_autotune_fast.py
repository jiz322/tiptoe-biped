#!/usr/bin/env python3
"""
sysid_autotune_fast.py — VECTORIZED auto-tuner for the sim actuator params.

Same objective as sysid_autotune.py (minimize sim-vs-real RMSE of the replay),
but built for speed: it spawns N parallel IsaacLab environments and evaluates a
whole optimizer *population* in one batched rollout instead of one candidate at
a time.

Why this is faster
------------------
Optuna/TPE is inherently sequential — each candidate is chosen from all previous
results — so it pins you to one environment and leaves the GPU mostly idle. A
population optimizer (CMA-ES) proposes `popsize` independent candidates per
generation, which map one-to-one onto `popsize` parallel envs. Each env gets a
DIFFERENT parameter set (the same per-env actuator write your training env
already does in `_write_per_env_actuators_to_sim`); we replay the SAME recorded
command across all of them and read back `popsize` RMSE values at once. A
generation is ~1 rollout, not `popsize` rollouts, so even a handful of
generations evaluates far more candidates than the serial script in a fraction
of the wall-clock.

Per-group decomposition, warm-start at the current values, and the never-regress
guard are all kept from the serial version.

WHAT IS / ISN'T TESTED
----------------------
The optimizer core (CMA-ES driver, [0,1] normalization, candidate->param
mapping, never-regress) is pure NumPy and was validated standalone with a
synthetic objective. The IsaacLab binding (multi-env spawn + per-env writes +
batched rollout) cannot be exercised without a GPU + your URDF + recordings;
the one block most likely to need a version tweak is marked
`>>> VERIFY FOR YOUR ISAACLAB VERSION <<<`.

Usage (run from the dir with rl_env_cfg_p6.py; use the idle GPU if 0 is busy):
    python sysid_autotune_fast.py sysid_leg1_*.npz --headless --device cuda:1
    python sysid_autotune_fast.py sysid_leg1_*.npz --headless --num-envs 128 --generations 20
    python sysid_autotune_fast.py sysid_leg1_*.npz --headless --optimizer random   # no `cma` dep
    python sysid_autotune_fast.py sysid_leg1_*.npz --headless --groups knee --check-determinism

Speed knobs:
    --num-envs     parallel envs == CMA-ES popsize (more = wider parallel search)
    --generations  CMA-ES iterations (fewer = faster; ~8-12 is usually plenty
                   from a warm start, 15-25 for a thorough search)
"""
from __future__ import annotations
import argparse, json, sys, os, copy, time
import numpy as np

# ── joint bookkeeping (identical to sysid_replay_sim.py) ────────────
GROUPS = ["hip_roll", "hip_pitch", "knee"]
JTYPE  = ["hip_roll", "hip_pitch", "knee", "hip_roll", "hip_pitch", "knee"]
GROUP  = JTYPE
SUB    = [0, 0, 0, 1, 1, 1]
EXCITE = {"hip_roll": (6.0, 0.3), "hip_pitch": (8.0, 0.5), "knee": (8.0, 0.5)}
HOLD   = {"hip_roll": (15.0, 0.8), "hip_pitch": (20.0, 1.2), "knee": (20.0, 1.2)}
CMD_DELAY_STEPS = 0
ARMATURE_MULT   = {"knee": 1.0}     # not tuned here; matches the replay script
_HW_JOINT_ORDER = ["hip_roll1", "hip_pitch1", "knee1",
                   "hip_roll2", "hip_pitch2", "knee2"]

CURRENT = {
    "hip_roll":  dict(omega=15.0, zeta=0.25, friction=0.10,
                      dynamic_friction=0.08, viscous_friction=0.002, armature_mult=1.0),
    "hip_pitch": dict(omega=15.0, zeta=0.50, friction=0.12,
                      dynamic_friction=0.10, viscous_friction=0.002, armature_mult=1.0),
    "knee":      dict(omega=15.0, zeta=0.65, friction=0.12,
                      dynamic_friction=0.10, viscous_friction=0.002, armature_mult=1.0),
}

SEARCH_SPACE = {
    "omega":     (8.0, 25.0),
    "zeta":      (0.10, 1.00),
    "friction":  (0.00, 0.30),
    "dyn_ratio": (0.30, 1.00),   # dynamic = ratio * static  (PhysX: mu_d <= mu_s)
    "viscous":   (0.000, 0.030),
}
_KEYS = ["omega", "zeta", "friction", "dyn_ratio", "viscous"]
_LO   = np.array([SEARCH_SPACE[k][0] for k in _KEYS])
_HI   = np.array([SEARCH_SPACE[k][1] for k in _KEYS])


# ══════════════════════════════════════════════════════════════════
# Pure helpers  (no IsaacLab — unit-testable, see the self-test block)
# ══════════════════════════════════════════════════════════════════
def normalize(raw):                       # raw dict -> vector in [0,1]^5
    x = np.array([raw[k] for k in _KEYS], dtype=float)
    return np.clip((x - _LO) / (_HI - _LO), 0.0, 1.0)

def denormalize(v):                       # vector in [0,1]^5 -> raw dict
    x = _LO + np.clip(np.asarray(v, float), 0.0, 1.0) * (_HI - _LO)
    return {k: float(x[i]) for i, k in enumerate(_KEYS)}

def to_raw(final):                        # final group dict -> raw (with dyn_ratio)
    ratio = (final["dynamic_friction"] / final["friction"]) if final["friction"] > 0 else 1.0
    return dict(omega=final["omega"], zeta=final["zeta"], friction=final["friction"],
                dyn_ratio=float(np.clip(ratio, *SEARCH_SPACE["dyn_ratio"])),
                viscous=final["viscous_friction"])

def from_raw(raw, armature_mult):         # raw dict -> final group dict
    dyn = float(np.clip(raw["dyn_ratio"], 0.0, 1.0)) * raw["friction"]
    return dict(omega=raw["omega"], zeta=raw["zeta"], friction=raw["friction"],
                dynamic_friction=dyn, viscous_friction=raw["viscous"],
                armature_mult=armature_mult)

def build_per_env(raw_list, group, base, N):
    """Tile `raw_list` (the varying group's candidates) across N envs; hold the
    other groups fixed at `base`. Returns {group: {static,dynamic,viscous,
    omega,zeta,armature_mult}} with each value a length-N NumPy array."""
    L   = len(raw_list)
    pick = [raw_list[i % L] for i in range(N)]
    static = np.array([r["friction"] for r in pick])
    ratio  = np.clip([r["dyn_ratio"] for r in pick], 0.0, 1.0)
    per = {}
    for g in GROUPS:
        if g == group:
            per[g] = dict(static=static, dynamic=ratio * static,
                          viscous=np.array([r["viscous"] for r in pick]),
                          omega=np.array([r["omega"] for r in pick]),
                          zeta=np.array([r["zeta"] for r in pick]),
                          armature_mult=np.full(N, base[g]["armature_mult"]))
        else:
            b = base[g]
            per[g] = dict(static=np.full(N, b["friction"]),
                          dynamic=np.full(N, b["dynamic_friction"]),
                          viscous=np.full(N, b["viscous_friction"]),
                          omega=np.full(N, b["omega"]),
                          zeta=np.full(N, b["zeta"]),
                          armature_mult=np.full(N, b["armature_mult"]))
    return per

def per_env_broadcast(params_full, N):
    """Same param set in every env — used for final scoring / plots."""
    per = {}
    for g in GROUPS:
        b = params_full[g]
        per[g] = dict(static=np.full(N, b["friction"]), dynamic=np.full(N, b["dynamic_friction"]),
                      viscous=np.full(N, b["viscous_friction"]), omega=np.full(N, b["omega"]),
                      zeta=np.full(N, b["zeta"]), armature_mult=np.full(N, b["armature_mult"]))
    return per


# ══════════════════════════════════════════════════════════════════
# Population optimizers  (cost_fn: list[np.ndarray] -> np.ndarray, len==popsize)
# ══════════════════════════════════════════════════════════════════
def optimize_group_cmaes(cost_fn, x0, sigma0, generations, popsize, seed):
    import cma
    es = cma.CMAEvolutionStrategy(
        list(x0), sigma0,
        {"bounds": [0, 1], "popsize": popsize, "seed": seed,
         "maxiter": generations, "verbose": -9},
    )
    best_x, best_c = np.array(x0, float), np.inf
    for gen in range(generations):
        X = es.ask()
        if gen == 0:
            X[0] = list(x0)                      # force-measure the warm start
        C = cost_fn([np.asarray(x, float) for x in X])
        es.tell(X, [float(c) for c in C])
        j = int(np.argmin(C))
        if C[j] < best_c:
            best_c, best_x = float(C[j]), np.asarray(X[j], float).copy()
        print(f"    gen {gen:2d}/{generations}  min={float(np.min(C)):.5f}  "
              f"mean={float(np.mean(C)):.5f}  best={best_c:.5f}", flush=True)
        if es.stop():
            break
    return best_x, best_c


def optimize_group_random(cost_fn, x0, sigma0, generations, popsize, seed):
    """Dependency-free fallback: Gaussian cloud around the best, shrinking each
    generation. Fully parallel; good for local polishing from a warm start."""
    rng = np.random.default_rng(seed)
    dim = len(x0)
    best_x, best_c = np.array(x0, float), np.inf
    sigma = sigma0
    for gen in range(generations):
        X = np.clip(best_x[None, :] + sigma * rng.standard_normal((popsize, dim)), 0.0, 1.0)
        if gen == 0:
            X[0] = x0
        C = cost_fn([X[i] for i in range(popsize)])
        j = int(np.argmin(C))
        if C[j] < best_c:
            best_c, best_x = float(C[j]), X[j].copy()
        print(f"    gen {gen:2d}/{generations}  min={float(np.min(C)):.5f}  "
              f"mean={float(np.mean(C)):.5f}  best={best_c:.5f}  sigma={sigma:.3f}", flush=True)
        sigma *= 0.85
    return best_x, best_c


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def load_trials(path):
    d = np.load(path, allow_pickle=True)
    names   = list(d["trial_names"])
    configs = json.loads(str(d["trial_configs"]))
    dpos    = d["default_pos"]
    trials  = {}
    for nm, cfg in zip(names, configs):
        trials[nm] = dict(
            t=d[f"{nm}__time"], p_des=d[f"{nm}__p_des"],
            pos=d[f"{nm}__pos"], vel=d[f"{nm}__vel"], cfg=cfg,
        )
    all_idx = [tr["cfg"]["idx"] for tr in trials.values()]
    off = 0 if min(all_idx) < 3 else 3
    return trials, dpos, off


# ══════════════════════════════════════════════════════════════════
# Vectorized replay engine (N parallel envs)
# ══════════════════════════════════════════════════════════════════
class BatchedReplaySimulator:
    """Builds N parallel envs once; each call evaluates up to N candidates in a
    single batched rollout. Physics logic mirrors sysid_replay_sim.py with a
    leading env (batch) dimension."""

    def __init__(self, npz_path, num_envs, env_spacing):
        # heavy imports happen here (after AppLauncher) so the module stays
        # importable for unit-testing the pure helpers above.
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.sim import SimulationContext
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.utils import configclass
        self.torch = torch

        self.trials, self.dpos, self.off = load_trials(npz_path)
        self.N = int(num_envs)

        env_cfg  = RlEnvCfg()
        self.sim = SimulationContext(env_cfg.sim)

        robot_cfg = env_cfg.robot_cfg
        robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"      # cloned across envs
        robot_cfg.spawn.articulation_props.fix_root_link = True
        for act_cfg in robot_cfg.actuators.values():
            act_cfg.effort_limit_sim   = act_cfg.effort_limit
            act_cfg.velocity_limit_sim = act_cfg.velocity_limit

        # >>> VERIFY FOR YOUR ISAACLAB VERSION <<<
        # Standard multi-env spawn: a configclass scene whose `robot` field is
        # your articulation on the {ENV_REGEX_NS} path. If this errors, the two
        # most common fixes are (a) your namespace is `omni.isaac.lab.scene`
        # instead of `isaaclab.scene`, or (b) build the scene with the manual
        # GridCloner pattern. As a last resort, instantiate your training env
        # (DirectRLEnv) with num_envs and grab env.scene["robot"] — that reuses
        # whatever spawn path your training already uses.
        @configclass
        class _ReplaySceneCfg(InteractiveSceneCfg):
            robot = robot_cfg
            dome  = AssetBaseCfg(
                prim_path="/World/Light",
                spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.75, 0.75, 0.75)),
            )

        self.scene = InteractiveScene(_ReplaySceneCfg(num_envs=self.N, env_spacing=env_spacing))
        self.sim.reset()
        self.robot = self.scene["robot"]
        self.robot.reset()
        # Warm up the GPU pipeline before any STATE write. A couple of steps
        # materialize the physics position/velocity buffers so the first
        # write_joint_state_to_sim (set_dof_positions) doesn't fail with
        # "Failed to set DOF positions in backend". Parameter writes
        # (friction/armature) don't need this; GPU state writes do.
        for _ in range(2):
            self.sim.step(render=False)
        # <<< end verify block >>>

        self.device = self.robot.device
        self.J      = self.robot.num_joints
        self.base_armature = {g: float(robot_cfg.actuators[g].armature) for g in GROUPS}

        hw_to_sim = []
        for jn in _HW_JOINT_ORDER:
            ids, _ = self.robot.find_joints([jn])
            hw_to_sim.append(ids[0])
        self.hw_to_sim_np = np.array(hw_to_sim)
        self.hw_to_sim_t  = torch.tensor(hw_to_sim, device=self.device)
        self.group_ids    = {g: self.robot.actuators[g].joint_indices for g in GROUPS}

        dpos_t = torch.tensor(self.dpos, dtype=torch.float32, device=self.device)
        self.default_sim = torch.zeros(self.N, self.J, dtype=torch.float32, device=self.device)
        self.default_sim[:, self.hw_to_sim_t] = dpos_t.unsqueeze(0).repeat(self.N, 1)
        self.zero_vel_sim = torch.zeros_like(self.default_sim)

        self.DECIMATION = env_cfg.decimation
        self.DT_PHY     = float(env_cfg.sim.dt)
        self.DT_CTRL    = self.DECIMATION * self.DT_PHY
        if max(CURRENT[g]["omega"] for g in GROUPS) * self.DT_CTRL > 1.0:
            print("  [warn] omega*dt large; 2nd-order prefilter may be inaccurate")

    # ── per-env actuator writes (the path your training env uses) ──
    def _write_params(self, per_env):
        torch = self.torch
        for g, act in self.robot.actuators.items():
            p        = per_env[g]
            static_t = torch.as_tensor(p["static"],  dtype=torch.float32, device=self.device)
            dyn_t    = torch.as_tensor(p["dynamic"], dtype=torch.float32, device=self.device)
            dyn_t    = torch.minimum(dyn_t, static_t)             # PhysX: mu_d <= mu_s
            visc_t   = torch.as_tensor(p["viscous"], dtype=torch.float32, device=self.device)
            arm_t    = torch.as_tensor(self.base_armature[g] * p["armature_mult"],
                                       dtype=torch.float32, device=self.device)
            act.friction[:]         = static_t[:, None]
            act.dynamic_friction[:] = dyn_t[:, None]
            act.viscous_friction[:] = visc_t[:, None]
            act.armature[:]         = arm_t[:, None]
            self.robot.write_joint_armature_to_sim(act.armature, joint_ids=act.joint_indices)
            self.robot.write_joint_friction_coefficient_to_sim(
                act.friction,
                joint_dynamic_friction_coeff=act.dynamic_friction,
                joint_viscous_friction_coeff=act.viscous_friction,
                joint_ids=act.joint_indices,
            )

    def _build_prefilter(self, per_env):
        torch = self.torch
        omega = torch.zeros(self.N, self.J, dtype=torch.float32, device=self.device)
        zeta  = torch.ones (self.N, self.J, dtype=torch.float32, device=self.device)
        for g in GROUPS:
            ids = self.group_ids[g]
            omega[:, ids] = torch.as_tensor(per_env[g]["omega"], dtype=torch.float32,
                                            device=self.device)[:, None]
            zeta[:, ids]  = torch.as_tensor(per_env[g]["zeta"],  dtype=torch.float32,
                                            device=self.device)[:, None]
        return omega, zeta

    def _replay_trial(self, trial, omega, zeta):
        """Rollout one trial on all N envs. Returns excited-joint sim positions,
        shape (T, N), as NumPy. (Readback is accumulated on-GPU and copied once.)"""
        torch     = self.torch
        cfg       = trial["cfg"]
        idx       = cfg["idx"]
        hip_float = cfg.get("hip_float")
        sim_idx   = int(self.hw_to_sim_np[idx])
        p_des_rec = trial["p_des"]
        n_steps   = len(trial["t"])

        self.robot.write_joint_state_to_sim(self.default_sim, self.zero_vel_sim)
        self.robot.reset()
        pf_y  = self.default_sim.clone()
        pf_yd = torch.zeros_like(self.default_sim)

        use_lag  = bool((omega > 0).any())
        omega2   = omega ** 2
        two_zw   = 2.0 * zeta * omega
        lag_mask = (omega > 0).float()

        # gains depend on the trial's role, NOT the candidate -> identical across envs
        for j in range(6):
            act = self.robot.actuators[GROUP[j]]
            s   = SUB[j]
            jt  = JTYPE[j]
            if j == idx:
                kp, kd = EXCITE[jt]
            elif hip_float is not None and j == hip_float:
                kp, kd = 0.0, 0.3
            else:
                kp, kd = HOLD[jt]
            act.stiffness[:, s] = kp
            act.damping[:, s]   = kd

        buf = torch.empty(n_steps, self.N, dtype=torch.float32, device=self.device)
        for step in range(n_steps):
            cmd_step = max(0, step - CMD_DELAY_STEPS)
            tgt_J = torch.zeros(self.J, dtype=torch.float32, device=self.device)
            tgt_J[self.hw_to_sim_t] = torch.as_tensor(p_des_rec[cmd_step],
                                                      dtype=torch.float32, device=self.device)
            u = tgt_J.unsqueeze(0).repeat(self.N, 1)                   # (N, J), same cmd all envs

            if use_lag:
                pf_ydd = omega2 * (u - pf_y) - two_zw * pf_yd
                pf_yd  = pf_yd + pf_ydd * self.DT_CTRL
                pf_y   = pf_y  + pf_yd  * self.DT_CTRL
                out    = lag_mask * pf_y + (1.0 - lag_mask) * u
                self.robot.set_joint_position_target(out)
            else:
                self.robot.set_joint_position_target(u)

            for _ in range(self.DECIMATION):
                self.robot.write_data_to_sim()
                self.sim.step()
                self.robot.update(self.DT_PHY)      # full-rate PD; required, no .cpu() here

            buf[step] = self.robot.data.joint_pos[:, sim_idx]

        return buf.cpu().numpy()                    # (T, N)

    def _group_trials(self, group):
        return [(nm, tr) for nm, tr in self.trials.items()
                if JTYPE[tr["cfg"]["idx"]] == group]

    def group_cost_batched(self, per_env, group):
        """Mean per-trial RMSE over `group`'s trials, per env -> (N,)."""
        self._write_params(per_env)
        omega, zeta = self._build_prefilter(per_env)
        trials = self._group_trials(group)
        if not trials:
            return np.full(self.N, np.inf)
        acc = np.zeros(self.N)
        for nm, tr in trials:
            sp   = self._replay_trial(tr, omega, zeta)                 # (T, N)
            real = tr["pos"][:, tr["cfg"]["idx"]]                      # (T,)
            acc += np.sqrt(np.mean((sp - real[:, None]) ** 2, axis=0)) # (N,)
        return acc / len(trials)

    def evaluate_all(self, params_full):
        """Replay every trial with one param set (env 0). Returns overlay data."""
        per_env     = per_env_broadcast(params_full, self.N)
        self._write_params(per_env)
        omega, zeta = self._build_prefilter(per_env)
        out = {}
        for nm, tr in self.trials.items():
            sp  = self._replay_trial(tr, omega, zeta)                  # (T, N)
            idx = tr["cfg"]["idx"]
            out[nm] = dict(sim=sp[:, 0], real=tr["pos"][:, idx],
                           cmd=tr["p_des"][:, idx], t=tr["t"], idx=idx)
        return out


# ══════════════════════════════════════════════════════════════════
# Output
# ══════════════════════════════════════════════════════════════════
def print_paste(best):
    print("\n# ---- paste into rl_env_cfg_p6.py / sysid_replay_sim.py ----")
    print("ACT_LAG_OMEGA = {")
    for g in GROUPS:
        print(f'    "{g}": {best[g]["omega"]:.3f},')
    print("}")
    print("ACT_LAG_ZETA  = {")
    for g in GROUPS:
        print(f'    "{g}": {best[g]["zeta"]:.3f},')
    print("}")
    print("FRICTION_TUNE = {")
    for g in GROUPS:
        b = best[g]
        print(f'    "{g}": dict(friction={b["friction"]:.4f}, '
              f'dynamic_friction={b["dynamic_friction"]:.4f}, '
              f'viscous_friction={b["viscous_friction"]:.4f}),')
    print("}")


def make_plots(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = {}
    for nm, r in results.items():
        jt  = JTYPE[r["idx"]]
        tag = f"{jt}_float" if "float" in nm else jt
        groups.setdefault(tag, []).append(nm)
    for tag, names in groups.items():
        n    = len(names)
        cols = min(n, 4)
        rows = max(1, (n + cols - 1) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4.5*cols, 3*rows), squeeze=False)
        fig.suptitle(f"{tag}: IsaacLab (red) vs hardware (blue) — AUTOTUNED", fontsize=12)
        for i, nm in enumerate(names):
            ax = axes[i // cols][i % cols]
            r  = results[nm]
            ax.plot(r["t"], r["real"], "b-",  lw=0.9, label="real")
            ax.plot(r["t"], r["sim"],  "r--", lw=0.9, label="sim")
            ax.plot(r["t"], r["cmd"],  "k:",  lw=0.4, alpha=0.35, label="cmd")
            ax.set_title(f"{nm}  RMSE={rmse(r['sim'], r['real']):.4f}", fontsize=8)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.legend(fontsize=7)
        for k in range(n, rows * cols):
            axes[k // cols][k % cols].set_visible(False)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf = f"sysid_autotune_{tag}.pdf"
        fig.savefig(pdf, dpi=150)
        plt.close(fig)
        print(f"  -> {pdf}")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
def main():
    t0  = time.time()
    sim = BatchedReplaySimulator(args.npz, args.num_envs, args.env_spacing)
    popsize = sim.N
    print(f"Loaded {len(sim.trials)} trials  (leg {'1' if sim.off == 0 else '2'})  "
          f"| {sim.N} parallel envs == popsize")

    have       = {JTYPE[tr["cfg"]["idx"]] for tr in sim.trials.values()}
    req_groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    skipped    = [g for g in req_groups if g not in have]
    req_groups = [g for g in req_groups if g in have]
    if skipped:
        print(f"  [skip] no trials for: {', '.join(skipped)}")
    print(f"Tuning groups: {req_groups}  (optimizer: {args.optimizer})\n")

    base = copy.deepcopy(CURRENT)

    if args.check_determinism:
        print("Determinism check (start params, 2 batched evals per group):")
        for g in req_groups:
            per = build_per_env([to_raw(CURRENT[g])], g, base, sim.N)
            c1  = float(sim.group_cost_batched(per, g)[0])
            c2  = float(sim.group_cost_batched(per, g)[0])
            flag = "" if abs(c1 - c2) < 1e-4 else "   <-- NOISY: raise --num-envs to average"
            print(f"  {g:10s} {c1:.6f}  {c2:.6f}  |delta|={abs(c1 - c2):.2e}{flag}")
        sim_app.close()
        return

    print("Scoring start (hand-tuned) params on all trials ...")
    start_results = sim.evaluate_all(CURRENT)
    start_mean = float(np.mean([rmse(r["sim"], r["real"]) for r in start_results.values()]))
    print(f"  start mean RMSE (all trials) = {start_mean:.4f}\n")

    for g in req_groups:
        print(f"== optimizing {g} ({args.optimizer}, pop={popsize}) ==")
        if g == "hip_roll":
            print("  [note] hip_roll overlays are the least trustworthy (the real "
                  "tether sits below the fixed root); weight this fit less.")

        def cost_fn(X_list, _g=g, _base=base):
            raws = [denormalize(x) for x in X_list]
            return sim.group_cost_batched(build_per_env(raws, _g, _base, sim.N), _g)

        x0 = normalize(to_raw(CURRENT[g]))
        if args.optimizer == "cmaes":
            best_x, _ = optimize_group_cmaes(cost_fn, x0, args.sigma0,
                                             args.generations, popsize, args.seed)
        else:
            best_x, _ = optimize_group_random(cost_fn, x0, args.sigma0,
                                              args.generations, popsize, args.seed)

        # never-regress: fresh head-to-head of warm-start vs best, both in one rollout
        cur_raw, best_raw = to_raw(CURRENT[g]), denormalize(best_x)
        per = build_per_env([cur_raw, best_raw], g, base, sim.N)        # env0=cur, env1=best
        C   = sim.group_cost_batched(per, g)
        c_cur, c_best = float(C[0]), float(C[1])
        if c_best <= c_cur:
            base[g] = from_raw(best_raw, CURRENT[g]["armature_mult"])
            print(f"  {g}: {c_cur:.4f} -> {c_best:.4f}  "
                  f"(improved {100*(c_cur - c_best)/max(c_cur, 1e-9):.1f}%)\n")
        else:
            print(f"  {g}: kept hand-tuned ({c_cur:.4f}; best {c_best:.4f} did not beat it)\n")

    print("Scoring tuned params on all trials ...")
    results = sim.evaluate_all(base)
    finals  = {nm: rmse(r["sim"], r["real"]) for nm, r in results.items()}
    print("\n" + "=" * 55)
    print(f"{'trial':<28s} {'joint':>5s} {'RMSE':>8s}")
    print("-" * 55)
    for nm in sorted(finals):
        print(f"{nm:<28s} {results[nm]['idx']:>5d} {finals[nm]:8.4f}")
    print("-" * 55)
    final_mean = float(np.mean(list(finals.values())))
    print(f"{'MEAN':<28s} {'':>5s} {final_mean:8.4f}")
    print("=" * 55)
    print(f"start mean {start_mean:.4f}  ->  tuned mean {final_mean:.4f}  "
          f"({100*(start_mean - final_mean)/max(start_mean, 1e-9):+.1f}%)")

    print_paste(base)

    out = args.out or args.npz.replace(".npz", "_autotune.json")
    with open(out, "w") as f:
        json.dump(dict(best=base, start_mean_rmse=start_mean,
                       tuned_mean_rmse=final_mean, per_trial_rmse=finals), f, indent=2)
    print(f"\nSaved best params -> {out}")

    if not args.no_plot:
        print("Overlay plots:")
        make_plots(results)

    print(f"\nDone in {time.time() - t0:.0f} s.")
    sim_app.close()


if __name__ == "__main__":
    # Launch IsaacLab (must precede any isaaclab import) — arg parsing lives here
    # so the module stays importable for the pure-helper self-test.
    from isaaclab.app import AppLauncher
    cli = argparse.ArgumentParser()
    cli.add_argument("npz", help="recorded sysid .npz")
    cli.add_argument("--num-envs", type=int, default=64, help="parallel envs == CMA-ES popsize")
    cli.add_argument("--env-spacing", type=float, default=2.5)
    cli.add_argument("--optimizer", choices=["cmaes", "random"], default="cmaes",
                     help="cmaes (needs `pip install cma`) or random (no dependency)")
    cli.add_argument("--generations", type=int, default=15)
    cli.add_argument("--sigma0", type=float, default=0.20, help="initial step (normalized [0,1] space)")
    cli.add_argument("--groups", default="knee,hip_pitch,hip_roll")
    cli.add_argument("--check-determinism", action="store_true")
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--no-plot", action="store_true")
    cli.add_argument("--out", default=None)
    AppLauncher.add_app_launcher_args(cli)
    args = cli.parse_args()
    launcher = AppLauncher(args)
    sim_app  = launcher.app

    import torch
    sys.path.insert(0, os.getcwd())
    from RL.tasks.direct.standup_curriculum.rl_env_cfg_p6 import RlEnvCfg
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    if args.optimizer == "cmaes":
        try:
            import cma  # noqa: F401
        except ImportError:
            print("[info] `cma` not installed -> falling back to --optimizer random "
                  "(`pip install cma` for faster convergence)")
            args.optimizer = "random"

    main()