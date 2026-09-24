#!/usr/bin/env python3
"""
sysid_autotune.py — Auto-tune the sim actuator parameters to minimize the
sim-vs-real RMSE measured by sysid_replay_sim.py.

Searches the four hand-tuned dicts:
    ACT_LAG_OMEGA, ACT_LAG_ZETA                          (2nd-order prefilter)
    FRICTION_TUNE.friction / .dynamic_friction / .viscous_friction
i.e. 5 numbers per joint group {hip_roll, hip_pitch, knee}.

Key design choices
------------------
* Launch IsaacLab + build the robot ONCE, then re-evaluate thousands of
  parameter sets in the same process. The sim app cannot be cheaply
  restarted and a fresh launch is ~30-60 s, so subprocess-per-eval is a
  non-starter. write_joint_friction_coefficient_to_sim / armature writes are
  cheap and are re-applied per candidate exactly the way rl_env_p6.py applies
  them, so each eval is just (write params -> replay -> RMSE).

* Optimize each joint group INDEPENDENTLY. Every sysid trial excites a single
  joint and holds the others fixed, so a knee trial's RMSE depends (to first
  order) only on the knee parameters. That turns one 15-D search into three
  ~5-D searches, each running only its own (fewer) trials — far faster and
  much easier to converge. (Cross-joint coupling is small because the held
  joints barely move; the final all-trials table below confirms it.)

* Warm-start at the current hand-tuned values and never regress: the start
  point is evaluated as the first trial, and the per-group winner is re-checked
  against the start on a fresh replay (with extra repeats) so a noisy lucky
  eval can't make things worse than what you already have.

* Constraint mu_dynamic <= mu_static (PhysX) is honoured by searching a ratio
  in [0,1] instead of an absolute dynamic coefficient.

Usage (run from the directory that contains rl_env_cfg_p6.py):
    python sysid_autotune.py sysid_leg1_*.npz --headless
    python sysid_autotune.py sysid_leg1_*.npz --headless --groups knee
    python sysid_autotune.py sysid_leg1_*.npz --headless --optimizer nelder-mead
    python sysid_autotune.py sysid_leg1_*.npz --headless --check-determinism

Tunables (armature multiplier, command delay) are left fixed to match the
replay script; see ARMATURE_MULT / CMD_DELAY_STEPS and SEARCH_SPACE to widen.
"""
from __future__ import annotations
import argparse, json, sys, os, copy, time

# ── Launch IsaacLab (must happen before other isaaclab imports) ─────
from isaaclab.app import AppLauncher
cli = argparse.ArgumentParser()
cli.add_argument("npz", help="recorded sysid .npz")
cli.add_argument("--optimizer", choices=["optuna", "nelder-mead"], default="optuna",
                 help="optuna (Bayesian/TPE, needs `pip install optuna`) or nelder-mead (scipy)")
cli.add_argument("--groups", default="knee,hip_pitch,hip_roll",
                 help="comma list of joint groups to tune")
cli.add_argument("--n-trials", type=int, default=60, help="optuna trials per group")
cli.add_argument("--timeout", type=float, default=None, help="optuna wall-clock seconds per group")
cli.add_argument("--maxiter", type=int, default=80, help="nelder-mead iters per group")
cli.add_argument("--eval-repeats", type=int, default=1,
                 help="average RMSE over N replays per eval (raise if the sim is noisy)")
cli.add_argument("--check-determinism", action="store_true",
                 help="evaluate the start params twice per group, report the spread, then exit")
cli.add_argument("--seed", type=int, default=0)
cli.add_argument("--no-plot", action="store_true")
cli.add_argument("--out", default=None, help="json output (default <npz>_autotune.json)")
AppLauncher.add_app_launcher_args(cli)
args = cli.parse_args()
launcher = AppLauncher(args)
sim_app = launcher.app

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation

sys.path.insert(0, os.getcwd())
from RL.tasks.direct.standup_curriculum.rl_env_cfg_p6 import RlEnvCfg

torch.manual_seed(args.seed)
np.random.seed(args.seed)

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

# ── current hand-tuned values = warm-start / fallback floor ─────────
CURRENT = {
    "hip_roll":  dict(omega=15.0, zeta=0.25, friction=0.10,
                      dynamic_friction=0.08, viscous_friction=0.002, armature_mult=1.0),
    "hip_pitch": dict(omega=15.0, zeta=0.50, friction=0.12,
                      dynamic_friction=0.10, viscous_friction=0.002, armature_mult=1.0),
    "knee":      dict(omega=15.0, zeta=0.65, friction=0.12,
                      dynamic_friction=0.10, viscous_friction=0.002, armature_mult=1.0),
}

# ── search bounds (kept fairly tight: we're refining, not exploring) ─
SEARCH_SPACE = {
    "omega":     (8.0, 25.0),    # rad/s   (~0.15-0.35 s rise)
    "zeta":      (0.10, 1.00),   # under- to critically damped
    "friction":  (0.00, 0.30),   # static breakaway
    "dyn_ratio": (0.30, 1.00),   # dynamic = ratio * static  (PhysX: mu_d <= mu_s)
    "viscous":   (0.000, 0.030), # smooth velocity damping
}
_KEYS = ["omega", "zeta", "friction", "dyn_ratio", "viscous"]


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ══════════════════════════════════════════════════════════════════
# Data loading (identical to sysid_replay_sim.py)
# ══════════════════════════════════════════════════════════════════
def load_trials(path):
    d = np.load(path, allow_pickle=True)
    names   = list(d["trial_names"])
    configs = json.loads(str(d["trial_configs"]))
    dpos    = d["default_pos"]
    trials  = {}
    for nm, cfg in zip(names, configs):
        trials[nm] = dict(
            t=d[f"{nm}__time"], p_des=d[f"{nm}__p_des"],
            pos=d[f"{nm}__pos"], vel=d[f"{nm}__vel"],
            kp=d[f"{nm}__kp"], kd=d[f"{nm}__kd"], cfg=cfg,
        )
    all_idx = [tr["cfg"]["idx"] for tr in trials.values()]
    off = 0 if min(all_idx) < 3 else 3
    return trials, dpos, off


# ══════════════════════════════════════════════════════════════════
# Replay engine: build once, evaluate many parameter sets
# ══════════════════════════════════════════════════════════════════
class ReplaySimulator:
    """Owns the single IsaacLab session and replays trials for a given
    parameter set. All physics-relevant logic mirrors sysid_replay_sim.py."""

    def __init__(self, npz_path):
        self.trials, self.dpos, self.off = load_trials(npz_path)

        env_cfg = RlEnvCfg()
        self.sim = SimulationContext(env_cfg.sim)

        robot_cfg = env_cfg.robot_cfg
        robot_cfg.prim_path = "/World/Robot"
        robot_cfg.spawn.articulation_props.fix_root_link = True
        # match sim PhysX clamps to real hardware limits
        for act_cfg in robot_cfg.actuators.values():
            act_cfg.effort_limit_sim   = act_cfg.effort_limit
            act_cfg.velocity_limit_sim = act_cfg.velocity_limit

        self.robot = Articulation(cfg=robot_cfg)
        light = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)
        self.sim.reset()
        self.robot.reset()

        self.device     = self.robot.device
        self.num_joints = self.robot.num_joints
        # datasheet armature per group (the ARMATURE_MULT multiplier is applied on top)
        self.base_armature = {g: float(robot_cfg.actuators[g].armature) for g in GROUPS}

        # hw <-> sim joint index map (IsaacLab orders by type, hardware by leg)
        hw_to_sim = []
        for jn in _HW_JOINT_ORDER:
            ids, _ = self.robot.find_joints([jn])
            hw_to_sim.append(ids[0])
        self.hw_to_sim_np = np.array(hw_to_sim)
        self.hw_to_sim_t  = torch.tensor(hw_to_sim, device=self.device)

        # default pose in sim ordering
        self.default_sim = torch.zeros(1, self.num_joints, dtype=torch.float32, device=self.device)
        self.default_sim[0, self.hw_to_sim_t] = torch.tensor(
            self.dpos, dtype=torch.float32, device=self.device)
        self.zero_vel_sim = torch.zeros_like(self.default_sim)

        self.DECIMATION = env_cfg.decimation
        self.DT_PHY     = float(env_cfg.sim.dt)
        self.DT_CTRL    = self.DECIMATION * self.DT_PHY

        # prime PhysX with the start params so it holds sane values
        self.write_params({g: self._fr(CURRENT[g]) for g in GROUPS})
        if max(CURRENT[g]["omega"] for g in GROUPS) * self.DT_CTRL > 1.0:
            print("  [warn] omega*dt large; 2nd-order prefilter may be inaccurate")

    # ── push armature + 3-coeff joint friction into PhysX (env-exact) ──
    def write_params(self, fr_map):
        for gname, act in self.robot.actuators.items():
            fr       = fr_map[gname]
            static_f = float(fr["friction"])
            dyn_f    = min(float(fr["dynamic_friction"]), static_f)   # PhysX: mu_d <= mu_s
            visc_f   = float(fr["viscous_friction"])
            arm      = self.base_armature[gname] * float(fr.get("armature_mult", 1.0))

            act.friction[:]         = static_f
            act.dynamic_friction[:] = dyn_f
            act.viscous_friction[:] = visc_f
            act.armature[:]         = arm

            self.robot.write_joint_armature_to_sim(act.armature, joint_ids=act.joint_indices)
            self.robot.write_joint_friction_coefficient_to_sim(
                act.friction,
                joint_dynamic_friction_coeff=act.dynamic_friction,
                joint_viscous_friction_coeff=act.viscous_friction,
                joint_ids=act.joint_indices,
            )

    # ── replay one trial, return sim joint positions in hw order ──
    def replay_trial(self, trial, omega_map, zeta_map):
        cfg       = trial["cfg"]
        idx       = cfg["idx"]
        hip_float = cfg.get("hip_float")
        p_des_rec = trial["p_des"]
        n_steps   = len(trial["t"])

        self.robot.write_joint_state_to_sim(self.default_sim, self.zero_vel_sim)
        self.robot.reset()
        pf_y  = self.default_sim.clone()
        pf_yd = torch.zeros_like(self.default_sim)

        # build per-joint prefilter vectors (sim order) from the group dicts
        omega_vec = torch.zeros(1, self.num_joints, dtype=torch.float32, device=self.device)
        zeta_vec  = torch.ones (1, self.num_joints, dtype=torch.float32, device=self.device)
        for gname in GROUPS:
            ids = self.robot.actuators[gname].joint_indices
            omega_vec[0, ids] = float(omega_map[gname])
            zeta_vec [0, ids] = float(zeta_map[gname])
        use_lag  = bool((omega_vec > 0.0).any())
        omega2   = omega_vec ** 2
        two_zw   = 2.0 * zeta_vec * omega_vec
        lag_mask = (omega_vec > 0.0).float()

        # per-joint gains: EXCITE on the driven joint, optional float, else HOLD
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
            act.stiffness[0, s] = kp
            act.damping[0, s]   = kd

        sim_pos_buf = []
        for step in range(n_steps):
            cmd_step  = max(0, step - CMD_DELAY_STEPS)
            target_hw = torch.tensor(p_des_rec[cmd_step], dtype=torch.float32, device=self.device)
            target_sim = torch.zeros(self.num_joints, dtype=torch.float32, device=self.device)
            target_sim[self.hw_to_sim_t] = target_hw

            if use_lag:
                u      = target_sim.unsqueeze(0)
                pf_ydd = omega2 * (u - pf_y) - two_zw * pf_yd
                pf_yd  = pf_yd + pf_ydd * self.DT_CTRL
                pf_y   = pf_y  + pf_yd  * self.DT_CTRL
                out    = lag_mask * pf_y + (1.0 - lag_mask) * u
                self.robot.set_joint_position_target(out)
            else:
                self.robot.set_joint_position_target(target_sim.unsqueeze(0))

            for _ in range(self.DECIMATION):
                self.robot.write_data_to_sim()
                self.sim.step()
                # refresh state every physics substep so the DCMotor PD runs at
                # the full rate (matches DirectRLEnv.step) — critical, see replay
                self.robot.update(self.DT_PHY)

            sim_pos_all = self.robot.data.joint_pos[0].cpu().numpy()
            sim_pos_buf.append(sim_pos_all[self.hw_to_sim_np].copy())

        return np.array(sim_pos_buf)

    # ── helpers ──
    @staticmethod
    def _fr(p):
        return dict(friction=p["friction"], dynamic_friction=p["dynamic_friction"],
                    viscous_friction=p["viscous_friction"],
                    armature_mult=p.get("armature_mult", 1.0))

    def _group_trials(self, group):
        return [(nm, tr) for nm, tr in self.trials.items()
                if JTYPE[tr["cfg"]["idx"]] == group]

    def group_cost(self, params, group, repeats=1):
        """Mean per-trial RMSE over `group`'s trials for a full param set."""
        self.write_params({g: self._fr(params[g]) for g in GROUPS})
        omega_map = {g: params[g]["omega"] for g in GROUPS}
        zeta_map  = {g: params[g]["zeta"]  for g in GROUPS}
        trials = self._group_trials(group)
        if not trials:
            return float("inf")
        runs = []
        for _ in range(max(1, repeats)):
            errs = []
            for nm, tr in trials:
                sp  = self.replay_trial(tr, omega_map, zeta_map)
                idx = tr["cfg"]["idx"]
                errs.append(rmse(sp[:, idx], tr["pos"][:, idx]))
            runs.append(np.mean(errs))
        return float(np.mean(runs))

    def evaluate_all(self, params):
        """Replay every trial once; return per-trial overlay data."""
        self.write_params({g: self._fr(params[g]) for g in GROUPS})
        omega_map = {g: params[g]["omega"] for g in GROUPS}
        zeta_map  = {g: params[g]["zeta"]  for g in GROUPS}
        out = {}
        for nm, tr in self.trials.items():
            sp  = self.replay_trial(tr, omega_map, zeta_map)
            idx = tr["cfg"]["idx"]
            out[nm] = dict(sim_pos=sp, real_pos=tr["pos"], p_des=tr["p_des"],
                           t=tr["t"], idx=idx)
        return out


# ══════════════════════════════════════════════════════════════════
# Parameter (un)packing
# ══════════════════════════════════════════════════════════════════
def _assemble(base_params, group, x):
    """Insert a candidate (dict over _KEYS) for `group` into a full param set."""
    p     = copy.deepcopy(base_params)
    fr    = float(x["friction"])
    ratio = float(min(max(x["dyn_ratio"], 0.0), 1.0))
    p[group] = dict(
        omega=float(x["omega"]), zeta=float(x["zeta"]),
        friction=fr, dynamic_friction=ratio * fr,
        viscous_friction=float(x["viscous"]),
        armature_mult=base_params[group].get("armature_mult", 1.0),
    )
    return p


def _start_x(group):
    cur    = CURRENT[group]
    ratio0 = (cur["dynamic_friction"] / cur["friction"]) if cur["friction"] > 0 else 1.0
    lo, hi = SEARCH_SPACE["dyn_ratio"]
    return {"omega": cur["omega"], "zeta": cur["zeta"], "friction": cur["friction"],
            "dyn_ratio": min(max(ratio0, lo), hi), "viscous": cur["viscous_friction"]}


# ══════════════════════════════════════════════════════════════════
# Optimizers (both return a 6-key group dict)
# ══════════════════════════════════════════════════════════════════
def optimize_group_optuna(sim, group, base_params, repeats, n_trials, timeout, seed):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial(_start_x(group))                      # warm-start = trial 0

    def objective(trial):
        x = {k: trial.suggest_float(k, *SEARCH_SPACE[k]) for k in _KEYS}
        return sim.group_cost(_assemble(base_params, group, x), group, repeats)

    def cb(study, trial):
        v = trial.value if trial.value is not None else float("nan")
        print(f"    [{group}] trial {trial.number:3d}/{n_trials}  "
              f"value={v:.5f}  best={study.best_value:.5f}", flush=True)

    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[cb])
    return _assemble(base_params, group, study.best_params)[group]


def optimize_group_scipy(sim, group, base_params, repeats, maxiter):
    from scipy.optimize import minimize
    x0     = np.array([_start_x(group)[k] for k in _KEYS], dtype=float)
    bounds = [SEARCH_SPACE[k] for k in _KEYS]
    lo     = np.array([b[0] for b in bounds])
    hi     = np.array([b[1] for b in bounds])
    n      = {"i": 0}

    def cost(xv):
        xv = np.minimum(np.maximum(xv, lo), hi)
        c  = sim.group_cost(_assemble(base_params, group, dict(zip(_KEYS, xv))), group, repeats)
        n["i"] += 1
        if n["i"] % 5 == 0:
            print(f"    [{group}] eval {n['i']:3d}  cost={c:.5f}", flush=True)
        return c

    res = minimize(cost, x0, method="Nelder-Mead", bounds=list(zip(lo, hi)),
                   options=dict(maxiter=maxiter, xatol=1e-3, fatol=1e-5, disp=False))
    xb = np.minimum(np.maximum(res.x, lo), hi)
    return _assemble(base_params, group, dict(zip(_KEYS, xb)))[group]


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
    plot_groups = {}
    for name, r in results.items():
        jt  = JTYPE[r["idx"]]
        tag = f"{jt}_float" if "float" in name else jt
        plot_groups.setdefault(tag, []).append(name)
    for tag, names in plot_groups.items():
        n    = len(names)
        cols = min(n, 4)
        rows = max(1, (n + cols - 1) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4.5*cols, 3*rows), squeeze=False)
        fig.suptitle(f"{tag}: IsaacLab (red) vs hardware (blue) — AUTOTUNED", fontsize=12)
        for i, nm in enumerate(names):
            ax = axes[i // cols][i % cols]
            r  = results[nm]
            j  = r["idx"]
            ax.plot(r["t"], r["real_pos"][:, j], "b-",  lw=0.9, label="real")
            ax.plot(r["t"], r["sim_pos"][:, j],  "r--", lw=0.9, label="sim")
            ax.plot(r["t"], r["p_des"][:, j],    "k:",  lw=0.4, alpha=0.35, label="cmd")
            ax.set_title(f"{nm}  RMSE={rmse(r['sim_pos'][:, j], r['real_pos'][:, j]):.4f}", fontsize=8)
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
    sim = ReplaySimulator(args.npz)
    print(f"Loaded {len(sim.trials)} trials  (leg {'1' if sim.off == 0 else '2'})")

    have       = {JTYPE[tr["cfg"]["idx"]] for tr in sim.trials.values()}
    req_groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    skipped    = [g for g in req_groups if g not in have]
    req_groups = [g for g in req_groups if g in have]
    if skipped:
        print(f"  [skip] no trials for: {', '.join(skipped)}")
    print(f"Tuning groups: {req_groups}  (optimizer: {args.optimizer})\n")

    # ── optional determinism probe (pick --eval-repeats from this) ──
    if args.check_determinism:
        print("Determinism check (start params, 2 replays per group):")
        for g in req_groups:
            c1 = sim.group_cost(CURRENT, g, 1)
            c2 = sim.group_cost(CURRENT, g, 1)
            flag = "" if abs(c1 - c2) < 1e-4 else "   <-- NOISY: use --eval-repeats > 1"
            print(f"  {g:10s} {c1:.6f}  {c2:.6f}  |delta|={abs(c1 - c2):.2e}{flag}")
        sim_app.close()
        return

    # ── before: score hand-tuned params on every trial ──
    print("Scoring start (hand-tuned) params on all trials ...")
    start_results = sim.evaluate_all(CURRENT)
    start_mean = float(np.mean([rmse(r["sim_pos"][:, r["idx"]], r["real_pos"][:, r["idx"]])
                                for r in start_results.values()]))
    print(f"  start mean RMSE (all trials) = {start_mean:.4f}\n")

    # ── per-group optimization (progressively updates `base`) ──
    base = copy.deepcopy(CURRENT)
    for g in req_groups:
        print(f"== optimizing {g} ==")
        if g == "hip_roll":
            print("  [note] hip_roll overlays are the least trustworthy (the real "
                  "tether sits below the fixed root); weight this fit less.")
        start_g = sim.group_cost(base, g, max(args.eval_repeats, 3))
        if args.optimizer == "optuna":
            cand = optimize_group_optuna(sim, g, base, args.eval_repeats,
                                         args.n_trials, args.timeout, args.seed)
        else:
            cand = optimize_group_scipy(sim, g, base, args.eval_repeats, args.maxiter)

        # fresh, higher-repeat re-check so noise can't beat the hand-tuned floor
        cand_params = copy.deepcopy(base); cand_params[g] = cand
        cand_g = sim.group_cost(cand_params, g, max(args.eval_repeats, 3))
        if cand_g <= start_g:
            base[g] = cand
            print(f"  {g}: {start_g:.4f} -> {cand_g:.4f}  "
                  f"(improved {100*(start_g - cand_g)/max(start_g, 1e-9):.1f}%)\n")
        else:
            print(f"  {g}: kept hand-tuned ({start_g:.4f}; "
                  f"best candidate {cand_g:.4f} did not beat it)\n")

    # ── after: full all-trials table (catches any cross-coupling) ──
    print("Scoring tuned params on all trials ...")
    results = sim.evaluate_all(base)
    finals  = {nm: rmse(r["sim_pos"][:, r["idx"]], r["real_pos"][:, r["idx"]])
               for nm, r in results.items()}
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
    main()
