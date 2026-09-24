"""sysid_optuna.py — plant-offset system identification against the hardware
balance trials (2026-09-10).

QUESTION. The hardware balance point sits ~3 cm / 5 deg behind the sim twin's
(R1r trim response +6.7 deg vs sim <= 2 deg the other way; S4a stepping falls
at both trims while every sim twin survives). Which small, physically anchored
plant offsets make ONE sim plant reproduce the hardware statistics of the
shuffle-gait policy at three IMU trims — and does that plant then PREDICT the
stepping-gait policy's opening without being fitted to it?

SEARCH SPACE (4 parameters, coherent per robot; L-leg joint convention, the
R leg gets the mirrored sign so the bias is the same physical direction):
    com_x      [m]   whole-body CoM shift, + forward   (dr_com_shift torso row)
    com_z      [m]   whole-body CoM shift, + up
    knee_bias  [rad] true knee = policy's knee + delta (encoder/belt zero error;
                     + = truly MORE flexed than believed; one tooth = -0.21)
    hip_bias   [rad] same for hip pitch (+ = thigh truly more extended/back)
PINNED (not searched): floor friction mult, total mass (5.57 kg weigh-in),
every other DR band as in the S4 measured-plant recipe (passed as env.* on the
CLI by run_s4_sysid_GPU2.ps1). The pg_x obs bias is pinned per cell to -trim.

CELLS. fit  : R1r @ trim -0.089 / 0 / +0.089   (hardware 08-31, 09-03, 09-06)
       pred : S4a @ trim 0 / +0.089            (hardware 09-10) — held out
Per candidate every cell runs `--seeds` envs (fresh DR draw each), all cells of
all candidates in ONE vectorized Isaac process.

CLEAN WINDOW (user, 09-10 eve). On the rig the safety tether can be in contact
from the first deep dip onward, so only the opening of each hardware recording
is trustworthy (0.6 s default, per-trial overrides in data/_s4_sysid_targets.py).
The objective therefore uses ONLY in-window statistics — the mean TRUE pitch
trace on a 0.1 s grid and the fraction of trials whose tilt crossed 25 deg by
0.4 / 0.6 s (the failure onset that precedes any tether action) — applied
identically to sim (sysid_metrics.py; chi-square-like, 4 deg / 0.25
tolerances). Objective = sum over the FIT cells. Search episodes are therefore
short (--episode-s 1.0); the final top-k re-evaluation runs --final-episode-s
(5 s) so the report can show the long-horizon PREDICTION (45-deg fall
fractions) next to the hardware's tether-suspect fall fractions.

OPTIMIZER. pycma CMA-ES (batch ask/tell, one generation = one sim pass) in
normalized [-1,1]^4 coordinates; every evaluated candidate is also recorded in
an Optuna sqlite study (study.db) with the per-cell terms as user attrs, so
optuna-dashboard / importance tools work on it. --sampler tpe uses Optuna's
TPE (constant-liar batches) instead. The nominal plant (all offsets 0) is
injected as a generation-0 candidate (baseline).

--sensitivity: instead of searching, evaluate the nominal plant and one-at-a-
time perturbations (+-3 cm CoM_x/z, +-0.2 rad knee, +-0.15 rad hip) with all
envs and print, per cell, how much each moves the in-window pitch trace
relative to the seed noise and the hardware spread — the "is 0.6 s enough"
check.

PRE-REGISTERED READINGS (written before the first run):
  (a) J_fit of the best candidate < ~2 (in-window pitch within ~4 deg rms,
      25-deg crossing fractions within ~0.25) AND its S4a J_pred < ~3 -> the
      offset is real and gait-independent; centre the next training DR on it.
      Soft check: the fitted plant's 5-s sim should make R1r fall at trims
      -0.089 / 0 and mostly stand at +0.089 (hardware pattern, tether-suspect).
  (b) fit good, S4a prediction bad -> stepping has a second, un-parameterized
      problem (contact / belt compliance / rate).
  (c) no candidate fits the three R1r cells (J_fit >~ 6) -> the gap is not a
      static offset of these four kinds; stop searching this family.
  (d) degeneracy: report the top-k spread; if com_x and knee_bias trade off
      along a ridge, only the real-robot bias sweep can separate them.
  (e) sensitivity: if no single perturbation moves the 0.5-0.6 s pitch by more
      than ~2x the seed-noise of a cell mean, the window is too short to
      identify the offsets and the fit result must not be trusted.
Total mass and friction are pinned by user decision (09-10), so a good fit is
conditional on them.

OUTPUT (<out>/): run.log, study.db, trials.csv, generations.csv, best.json,
report.txt, topk/*.npz (simlog-format 5-s rollouts of the top candidates, one
per cell/seed — readable by the data/_s4_*.py scripts), state.pkl (--resume-run).

Usage: via RL/run_s4_sysid_GPU2.ps1 (sets the Isaac env vars, GPU2, the
S4 plant recipe, and runs in the foreground so the tqdm progress bar with the
remaining-time estimate is visible). --smoke = tiny run to check the pipeline.
"""
from __future__ import annotations

import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Vectorized plant sysid (CMA-ES / Optuna) vs hardware balance trials.")
parser.add_argument("--task", type=str, default="Template-standup_curriculum-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--ckpt-r1r", type=str, required=True, help="R1r checkpoint (.pt), fit policy")
parser.add_argument("--ckpt-s4a", type=str, required=True, help="S4a checkpoint (.pt), held-out policy")
parser.add_argument("--out", type=str, required=True, help="run directory (created)")
parser.add_argument("--targets", type=str, default=None,
                    help="sysid_targets.json (default data/s4_flight_20260910/sysid_targets.json)")
parser.add_argument("--fit-cells", type=str, default="R1r@-0.089,R1r@+0.000,R1r@+0.089")
parser.add_argument("--predict-cells", type=str, default="S4a@+0.000,S4a@+0.089")
parser.add_argument("--pop", type=int, default=24, help="candidates per generation")
parser.add_argument("--seeds", type=int, default=8, help="envs per candidate per cell")
parser.add_argument("--gens", type=int, default=30)
parser.add_argument("--episode-s", type=float, default=1.0, help="search episode length (>= clean window)")
parser.add_argument("--final-episode-s", type=float, default=5.0, help="top-k / sensitivity episode length")
parser.add_argument("--total-mass-kg", type=float, default=5.57, help="pinned total robot mass")
parser.add_argument("--friction-mult", type=float, default=1.0, help="pinned floor friction multiplier")
parser.add_argument("--com-x-max", type=float, default=0.06)
parser.add_argument("--com-z-max", type=float, default=0.06)
parser.add_argument("--knee-bias-max", type=float, default=0.35)
parser.add_argument("--hip-bias-max", type=float, default=0.35)
parser.add_argument("--sigma0", type=float, default=0.3, help="CMA-ES initial std, normalized units")
parser.add_argument("--sampler", type=str, choices=["cma", "tpe"], default="cma")
parser.add_argument("--gait-phase-init", type=float, default=math.pi,
                    help="gait clock at every reset (twin convention: pi)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--topk", type=int, default=4, help="candidates re-evaluated with all envs at the end")
parser.add_argument("--resume-run", action="store_true", help="continue from <out>/state.pkl")
parser.add_argument("--sensitivity", action="store_true", help="one-at-a-time perturbation table instead of a search")
parser.add_argument("--smoke", action="store_true", help="pop 4, seeds 2, gens 2, short episodes")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.smoke:
    # tiny pipeline check; --pop / --seeds given explicitly are kept (timing a full-size generation)
    if "--pop" not in sys.argv:
        args_cli.pop = 4
    if "--seeds" not in sys.argv:
        args_cli.seeds = 2
    args_cli.gens, args_cli.episode_s, args_cli.final_episode_s, args_cli.topk = 2, 1.0, 1.5, 2
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── everything else after the app is up ──────────────────────────────────────
import csv  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import RL.tasks  # noqa: F401,E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sysid_metrics import (EXC_T, GRID, SURV_T, cell_distance, cell_stats,  # noqa: E402
                           format_cell_table, stats_from_json, stats_to_json,
                           trial_features)

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
PARAM_NAMES = ["com_x", "com_z", "knee_bias", "hip_bias"]
BOUNDS = np.array([args_cli.com_x_max, args_cli.com_z_max, args_cli.knee_bias_max, args_cli.hip_bias_max])
POLICY_KEYS = {"R1r": args_cli.ckpt_r1r, "S4a": args_cli.ckpt_s4a}
SENS_STEPS = {"com_x": 0.03, "com_z": 0.03, "knee_bias": 0.20, "hip_bias": 0.15}

# Deploy contract (control/config.py REAL_KP/KD, MotorBridge LPF) as defaults —
# identical to play_sim2real.py; any env.<key>= on the CLI wins.
BALANCE_CONTRACT = {
    "servo_kp_hip_roll": 6.0, "servo_kd_hip_roll": 0.3,
    "servo_kp_hip_pitch": 20.0, "servo_kd_hip_pitch": 1.4,
    "servo_kp_knee": 30.0, "servo_kd_knee": 1.5,
    "action_lpf_alpha": 0.6,
}
SIM_DEFAULT_JOINT_POS = np.array([0.0, 0.05, 0.0, 0.0, -0.05, 0.0], dtype=np.float32)
SIM_ACTION_SCALES = np.array([0.3, 0.75, 1.025, 0.3, 0.75, 1.025], dtype=np.float32)


def _hydra_overrides(key: str) -> bool:
    prefix = f"env.{key}="
    return any(a.startswith(prefix) or a.startswith("+" + prefix) for a in hydra_args)


class Tee:
    """Duplicate print() output into <out>/run.log (tqdm keeps stderr)."""

    def __init__(self, path, stream):
        self.f = open(path, "a", buffering=1, encoding="utf-8")
        self.s = stream

    def write(self, x):
        self.s.write(x)
        self.f.write(x)

    def flush(self):
        self.s.flush()
        self.f.flush()

    def isatty(self):
        return self.s.isatty()

    def __getattr__(self, name):
        return getattr(self.s, name)


def parse_cells(spec, targets):
    cells = []
    for name in [x.strip() for x in spec.split(",") if x.strip()]:
        if name not in targets["cells"]:
            raise SystemExit(f"cell {name!r} not in targets json (have {list(targets['cells'])})")
        c = targets["cells"][name]
        cells.append({"name": name, "policy": c["policy"], "trim": float(c["trim"]),
                      "hw": stats_from_json(c)})
    return cells


def u_to_params(u):
    return np.clip(np.asarray(u, dtype=np.float64), -1.0, 1.0) * BOUNDS


def install_gait_phase_pin(e, phase_init):
    orig = e._reset_idx

    def _pinned(env_ids):
        orig(env_ids)
        if env_ids is None:
            e._gait_phase[:] = phase_init
        else:
            e._gait_phase[env_ids] = phase_init

    e._reset_idx = _pinned


def whole_body_com_base(e):
    masses = e.robot.root_physx_view.get_masses().to(e.device)
    coms = e.robot.root_physx_view.get_coms().to(e.device)[..., :3]
    com_w = e.robot.data.body_pos_w + quat_apply(e.robot.data.body_quat_w, coms)
    tot = (masses.unsqueeze(-1) * com_w).sum(dim=1) / masses.sum(dim=-1, keepdim=True)
    return quat_apply_inverse(e.robot.data.root_quat_w, tot - e.robot.data.root_pos_w)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    out = os.path.abspath(args_cli.out)
    os.makedirs(os.path.join(out, "topk"), exist_ok=True)
    sys.stdout = Tee(os.path.join(out, "run.log"), sys.stdout)
    print(f"\n[sysid] {time.strftime('%Y-%m-%d %H:%M:%S')}  out={out}")
    print(f"[sysid] argv: {' '.join(sys.argv[1:])}")

    targets_path = args_cli.targets or os.path.join(REPO, "data", "s4_flight_20260910", "sysid_targets.json")
    with open(targets_path) as fh:
        targets = json.load(fh)
    if ([round(g, 3) for g in targets["grid"]] != [round(float(g), 3) for g in GRID]
            or list(targets["exc_t"]) != list(EXC_T) or list(targets["surv_t"]) != list(SURV_T)):
        raise SystemExit("targets json grid/exc_t/surv_t differ from sysid_metrics — rerun data/_s4_sysid_targets.py")
    window = float(targets["window_s"])
    fit_cells = parse_cells(args_cli.fit_cells, targets)
    pred_cells = parse_cells(args_cli.predict_cells, targets)
    cells = fit_cells + pred_cells
    n_fit, n_cells = len(fit_cells), len(cells)
    with open(os.path.join(out, "targets.json"), "w") as fh:
        json.dump(targets, fh, indent=1)
    if args_cli.episode_s < window + 0.05:
        raise SystemExit(f"--episode-s {args_cli.episode_s} must cover the clean window {window} s")

    pop, seeds = args_cli.pop, args_cli.seeds
    num_envs = pop * seeds * n_cells

    # ── env cfg: deploy scenario, same as the hardware twins ──────────────────
    agent_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.pickup_enable = False
    env_cfg.pickup_probability = 0.0
    env_cfg.pickup_initial_probability = 0.0
    env_cfg.curriculum_enable = False
    env_cfg.kneel_start_probability = 0.0
    for key, val in BALANCE_CONTRACT.items():
        if not _hydra_overrides(key):
            setattr(env_cfg, key, val)
    # clean PD-held handover pose (play_sim2real --init_noise_scale 0)
    env_cfg.noise_joint_pos = 0.0
    env_cfg.noise_base_vel_lin = 0.0
    env_cfg.noise_base_vel_ang = 0.0
    env_cfg.init_base_orient_noise_rad = 0.0
    env_cfg.log_dir = out

    print(f"[sysid] cells fit={[c['name'] for c in fit_cells]} pred={[c['name'] for c in pred_cells]}; "
          f"clean window {window} s (objective = in-window pitch grid + 25-deg crossing by {EXC_T})")
    print(f"[sysid] pop {pop} x seeds {seeds} x cells {n_cells} = {num_envs} envs; {args_cli.gens} gens x "
          f"{args_cli.episode_s:.1f} s search episodes, {args_cli.final_episode_s:.1f} s final; "
          f"pins: total mass {args_cli.total_mass_kg} kg, friction x{args_cli.friction_mult}")
    print(f"[sysid] bounds: " + ", ".join(f"{n} ±{b:g}" for n, b in zip(PARAM_NAMES, BOUNDS)))

    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    # the env's (disabled) pickup path calls a deprecated articulation API every
    # few seconds; its warning would shred the progress bar
    logging.getLogger("isaaclab.assets.articulation.articulation").setLevel(logging.ERROR)
    install_gait_phase_pin(e, args_cli.gait_phase_init)
    e.manual_cmd_override = True
    e.manual_vx = 0.0
    e.manual_yaw_rate = 0.0
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    dev = e.device
    dt = float(e.step_dt)
    T_search = int(round(args_cli.episode_s / dt))
    T_final = int(round(args_cli.final_episode_s / dt))
    T_max = max(T_search, T_final)
    N = e.num_envs
    jids = e._joint_ids

    # ── two policies on one env ──────────────────────────────────────────────
    policies = {}
    for key, path in POLICY_KEYS.items():
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(path)
        policies[key] = (runner.get_inference_policy(device=dev), runner.alg.policy)
        print(f"[sysid] policy {key}: {path}")

    # ── per-env override tensors (the sysid hook in rl_env_p6._sample_dr_*) ──
    nominal_total = float(e._original_masses[0].sum())
    mass_mult = args_cli.total_mass_kg / nominal_total
    ov = {
        "mass_mult": torch.full((N,), mass_mult, device=dev),
        "floor_friction_mult": torch.full((N,), float(args_cli.friction_mult), device=dev),
        "proj_grav_bias_x": torch.zeros(N, device=dev),
        "knee_bias": torch.zeros(N, 2, device=dev),
        "hip_bias": torch.zeros(N, 2, device=dev),
        "com_shift": torch.zeros(N, 3, device=dev),
    }
    e._dr_override = ov
    print(f"[sysid] nominal PhysX mass {nominal_total:.3f} kg -> mass_mult {mass_mult:.4f}; "
          f"gait clock pinned {args_cli.gait_phase_init:.3f}; control dt {dt:.4f} s")

    def make_layout(n_cands, seeds_per):
        """env i -> (cand, cell, seed); envs beyond n_cands*seeds*n_cells get cand -1."""
        block = seeds_per * n_cells
        i = np.arange(N)
        cand = np.where(i < n_cands * block, i // block, -1)
        rem = i % block
        cell = rem // seeds_per
        seed = rem % seeds_per
        return cand, cell, seed

    def apply_layout(params_phys, cand, cell):
        """params_phys [n_cands, 4]; write the override tensors for every env."""
        p = np.zeros((N, 4))
        ok = cand >= 0
        p[ok] = params_phys[cand[ok]]
        trims = np.array([cells[c]["trim"] for c in cell])
        ov["proj_grav_bias_x"][:] = torch.as_tensor(-trims, dtype=torch.float32, device=dev)
        ov["com_shift"][:, 0] = torch.as_tensor(p[:, 0], dtype=torch.float32, device=dev)
        ov["com_shift"][:, 1] = 0.0
        ov["com_shift"][:, 2] = torch.as_tensor(p[:, 1], dtype=torch.float32, device=dev)
        kb = torch.as_tensor(p[:, 2], dtype=torch.float32, device=dev)
        hb = torch.as_tensor(p[:, 3], dtype=torch.float32, device=dev)
        ov["knee_bias"][:, 0], ov["knee_bias"][:, 1] = kb, -kb     # R leg mirrored
        ov["hip_bias"][:, 0], ov["hip_bias"][:, 1] = hb, -hb
        masks = {}
        for key in POLICY_KEYS:
            m = np.array([cells[c]["policy"] == key for c in cell])
            masks[key] = torch.as_tensor(m, device=dev)
        return masks

    rec = {
        "proj_grav": torch.zeros(T_max, N, 3, device=dev), "ang_vel": torch.zeros(T_max, N, 3, device=dev),
        "joint_pos": torch.zeros(T_max, N, 6, device=dev), "joint_vel": torch.zeros(T_max, N, 6, device=dev),
        "tau": torch.zeros(T_max, N, 6, device=dev), "action": torch.zeros(T_max, N, 6, device=dev),
        "obs36": torch.zeros(T_max, N, 36, device=dev), "gait": torch.zeros(T_max, N, device=dev),
    }

    def rollout(masks, T, bar, after_reset=None):
        with torch.inference_mode():
            obs, _ = env.reset()
            e._gait_phase[:] = args_cli.gait_phase_init
            all_true = torch.ones(N, dtype=torch.bool, device=dev)
            for _, nn_ in policies.values():
                nn_.reset(all_true)
            if after_reset is not None:
                after_reset()
        first_term = torch.full((N,), T, dtype=torch.long, device=dev)
        for s in range(T):
            is_t = isinstance(obs, torch.Tensor)
            ot = obs if is_t else obs["policy"]
            with torch.inference_mode():
                act = torch.zeros(N, 6, device=dev)
                for key, (pol, _) in policies.items():
                    m = masks[key]
                    if m.any():
                        # rsl_rl >= 3 wants the obs-group container; TensorDict masks on batch
                        act[m] = pol(ot[m] if is_t else obs[m])
                rq = e.robot.data.root_quat_w
                rec["proj_grav"][s] = quat_apply_inverse(rq, e._gravity_vec)
                rec["ang_vel"][s] = quat_apply_inverse(rq, e.robot.data.root_ang_vel_w)
                rec["joint_pos"][s] = e.robot.data.joint_pos[:, jids]
                rec["joint_vel"][s] = e.robot.data.joint_vel[:, jids]
                rec["tau"][s] = e.robot.data.applied_torque[:, jids]
                rec["action"][s] = act
                rec["obs36"][s] = ot[:, :36]
                rec["gait"][s] = e._gait_phase
                obs, _, _, _ = env.step(act)
                fresh = (e.reset_terminated | e.reset_time_outs) & (first_term == T)
                first_term[fresh] = s + 1
            bar.update(1)
        return first_term

    def verify_overrides(cand, cell, params_phys):
        """PhysX / obs-bias read-back right after the first reset (spawn pose)."""
        masses = e.robot.root_physx_view.get_masses()
        tot = masses.sum(dim=-1)
        mats = e.robot.root_physx_view.get_material_properties()
        fric = mats[:, :, 0] / e._original_static_fric.clamp(min=1e-9)
        coms = e.robot.root_physx_view.get_coms()[:, e._torso_idx, :3]
        dcom = (coms - e._original_coms[:, e._torso_idx, :3]).to(dev)
        want = ov["com_shift"] * e._com_shift_gain
        pg = e._dr_proj_grav_bias[:, 0].cpu().numpy()
        trims = np.array([cells[c]["trim"] for c in cell])
        tq = e.robot.data.body_quat_w[0, e._torso_idx].cpu().numpy()
        print("[verify] PhysX total mass  min/max {:.3f}/{:.3f} kg (target {:.3f})".format(
            tot.min().item(), tot.max().item(), args_cli.total_mass_kg))
        print("[verify] floor friction mult min/max {:.3f}/{:.3f} (target {:.3f}); static mu env0 body0 = {:.3f}".format(
            fric.min().item(), fric.max().item(), args_cli.friction_mult, mats[0, 0, 0].item()))
        print("[verify] torso CoM row - original - com_shift*gain : max |resid| {:.2e} m "
              "(= the per-link dr_com_offset draw, <= {:.3f}; gain {:.3f})".format(
                  (dcom - want).abs().max().item(), float(e.cfg.dr_com_offset_max), e._com_shift_gain))
        print("[verify] torso body quat at spawn (w,x,y,z) = {} (identity => shift x=forward, z=up)".format(
            np.round(tq, 3)))
        print("[verify] pg_x obs bias == -trim : max |err| {:.2e}".format(np.abs(pg + trims).max()))
        i = int(np.argmax(cand >= 0))
        c = cand[i]
        print("[verify] env {} cand {} params {}: jpos_bias[hip,knee] L {} R {} | exec_bias L {} R {} "
              "(expected -delta / +delta plus the sampled ±jpos/exec DR)".format(
                  i, c, np.round(params_phys[c], 3),
                  np.round(e._dr_joint_pos_bias[i, [1, 2]].cpu().numpy(), 3),
                  np.round(e._dr_joint_pos_bias[i, [4, 5]].cpu().numpy(), 3),
                  np.round(e._exec_target_bias[i, [1, 2]].cpu().numpy(), 3),
                  np.round(e._exec_target_bias[i, [4, 5]].cpu().numpy(), 3)))
        com_b = whole_body_com_base(e).cpu().numpy()
        j = int(np.argmax(cand == (c + 1) % max(cand.max() + 1, 1)))
        d_meas = com_b[i] - com_b[j]
        d_want = ov["com_shift"][i].cpu().numpy() - ov["com_shift"][j].cpu().numpy()
        print("[verify] whole-body CoM (base frame, spawn pose) env {} - env {} = {} m; com_shift diff = {} m; "
              "resid {} m (per-link draw ~<= 5 mm)".format(i, j, np.round(d_meas, 4), np.round(d_want, 4),
                                                           np.round(d_meas - d_want, 4)))

    def evaluate(first_term, T, cand, cell, n_cands):
        """rec[:T] + first_term -> per-candidate per-cell stats, J_fit, J_pred."""
        pg = rec["proj_grav"][:T].cpu().numpy()          # [T, N, 3]
        ft = first_term.cpu().numpy()
        t = np.arange(T) * dt
        feats = [trial_features(t, pg[:, i], t_term=ft[i] * dt if ft[i] < T else np.inf, t_clean=window)
                 for i in range(N)]
        results = []
        for c in range(n_cands):
            per_cell, J_fit, J_pred = [], 0.0, 0.0
            for k, cinfo in enumerate(cells):
                sel = np.where((cand == c) & (cell == k))[0]
                s = cell_stats([feats[i] for i in sel])
                s["pitch_std"] = np.nanstd(np.stack([feats[i]["pitch_grid"] for i in sel]), axis=0)
                J, terms = cell_distance(s, cinfo["hw"])
                per_cell.append({"name": cinfo["name"], "J": J, "stats": s, **terms})
                if k < n_fit:
                    J_fit += J
                else:
                    J_pred += J
            results.append({"J_fit": J_fit, "J_pred": J_pred, "cells": per_cell})
        return results

    def save_npz(i, first_term, T, path, meta):
        n = int(min(first_term[i].item(), T))
        g = lambda k: rec[k][:n, i].cpu().numpy().astype(np.float32)  # noqa: E731
        act = g("action")
        np.savez(path, policy=np.asarray(meta["policy"]), control_hz=np.asarray(1.0 / dt),
                 time=(np.arange(n) * dt).astype(np.float32), obs=g("obs36"), action=act,
                 target=(SIM_DEFAULT_JOINT_POS + act * SIM_ACTION_SCALES).astype(np.float32),
                 sim_joint_pos=g("joint_pos"), sim_joint_vel=g("joint_vel"),
                 motor_tau_cmd=g("tau"), motor_tau_meas=g("tau"), motor_temp=np.zeros((n, 6), np.float32),
                 proj_grav=g("proj_grav"), ang_vel=g("ang_vel"), gait_phase=g("gait"),
                 mode=np.asarray(["NN_CONTROL"] * n),
                 t_term=np.asarray(first_term[i].item() * dt if first_term[i].item() < T else np.inf),
                 **{f"sysid_{k}": np.asarray(v) for k, v in meta.items() if k != "policy"})

    def stats_json_clean(s):
        return stats_to_json({k: v for k, v in s.items() if k != "pitch_std"})

    # ══ SENSITIVITY MODE ═══════════════════════════════════════════════════════
    if args_cli.sensitivity:
        names = ["nominal"]
        P = [np.zeros(4)]
        for j, n in enumerate(PARAM_NAMES):
            for sgn in (+1, -1):
                v = np.zeros(4)
                v[j] = sgn * SENS_STEPS[n]
                P.append(v)
                names.append(f"{n} {sgn * SENS_STEPS[n]:+g}")
        P = np.stack(P)
        k = len(P)
        seeds_s = max(1, N // (k * n_cells))
        cand, cell, _ = make_layout(k, seeds_s)
        T = T_final
        bar = tqdm(total=T, unit="tick", dynamic_ncols=True, smoothing=0.02, mininterval=0.5)
        bar.set_postfix_str(f"sensitivity: {k} plants x {seeds_s} seeds/cell")
        masks = apply_layout(P, cand, cell)
        first_term = rollout(masks, T, bar, after_reset=lambda: verify_overrides(cand, cell, P))
        res = evaluate(first_term, T, cand, cell, k)
        bar.close()
        picks = [0.3, 0.4, 0.5, 0.6]
        gi = [int(np.argmin(np.abs(GRID - p))) for p in picks]
        lines = ["=" * 100, f"SENSITIVITY  {time.strftime('%Y-%m-%d %H:%M:%S')}  {seeds_s} seeds per plant per cell, "
                 f"{T * dt:.1f} s episodes, window {window} s"]
        for kk, cinfo in enumerate(cells):
            hw = cinfo["hw"]
            lines.append("")
            lines.append(f"[{cinfo['name']}]  {'FIT' if kk < n_fit else 'PRED'}   pitch (deg) at "
                         + "/".join(f"{p:.1f}" for p in picks) + " s  |  E25@0.6  |  F45@2.5 F45@5 (5-s sim)  |  J vs hw")
            lines.append("   {:22s} ".format("hardware") + "  ".join(
                f"{hw['pitch_mean'][i]:6.1f}" if np.isfinite(hw["pitch_mean"][i]) else "   nan" for i in gi)
                + f"  |  {hw['E'][-1]:5.2f}    |  {hw['F'][2]:5.2f}  {hw['F'][3]:5.2f} (tether-suspect)")
            for c in range(k):
                cc = res[c]["cells"][kk]
                s = cc["stats"]
                lines.append("   {:22s} ".format(names[c]) + "  ".join(f"{s['pitch_mean'][i]:6.1f}" for i in gi)
                             + f"  |  {s['E'][-1]:5.2f}    |  {s['F'][2]:5.2f}  {s['F'][3]:5.2f}"
                             + f"               |  {cc['J']:5.2f}")
            s0 = res[0]["cells"][kk]["stats"]
            se = s0["pitch_std"][gi] / math.sqrt(seeds_s)
            lines.append("   seed std of the nominal plant at those times: "
                         + "  ".join(f"{s0['pitch_std'][i]:4.1f}" for i in gi)
                         + f"  deg (per env);  std of a {hw['n']}-trial mean: "
                         + "  ".join(f"{s0['pitch_std'][i] / math.sqrt(hw['n']):4.1f}" for i in gi) + " deg")
        lines.append("")
        lines.append("READING (e): a perturbation is identifiable inside the window if it moves the 0.5-0.6 s pitch "
                     "by clearly more than the std of a hardware cell mean (~1.5-2.5 deg); the hardware trim cells "
                     "differ by ~6-13 deg there.")
        rep = "\n".join(lines)
        print(rep)
        with open(os.path.join(out, "sensitivity.txt"), "w", encoding="utf-8") as fh:
            fh.write(rep + "\n")
        with open(os.path.join(out, "sensitivity.json"), "w") as fh:
            json.dump([{"name": names[c], "params": dict(zip(PARAM_NAMES, P[c].tolist())),
                        "J_fit": res[c]["J_fit"], "J_pred": res[c]["J_pred"],
                        "cells": [{"name": cc["name"], "J": cc["J"], **stats_json_clean(cc["stats"])}
                                  for cc in res[c]["cells"]]} for c in range(k)], fh, indent=1)
        env.close()
        return

    # ══ SEARCH ═════════════════════════════════════════════════════════════════
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    dists = {n: optuna.distributions.FloatDistribution(-float(b), float(b)) for n, b in zip(PARAM_NAMES, BOUNDS)}
    storage = f"sqlite:///{os.path.join(out, 'study.db').replace(os.sep, '/')}"
    try:
        study = optuna.create_study(study_name="s4_sysid", storage=storage, load_if_exists=True, direction="minimize",
                                    sampler=optuna.samplers.TPESampler(multivariate=True, constant_liar=True,
                                                                       n_startup_trials=pop, seed=args_cli.seed))
    except Exception as ex:  # sqlite/sqlalchemy trouble -> in-memory, csv still written
        print(f"[sysid] WARN optuna storage failed ({ex}); using in-memory study")
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(multivariate=True, constant_liar=True,
                                                                       n_startup_trials=pop, seed=args_cli.seed))

    state_path = os.path.join(out, "state.pkl")
    es = None
    trials = []          # every evaluated candidate: dict(gen, cand, params, J_fit, J_pred, cells)
    gen0 = 0
    if args_cli.resume_run and os.path.exists(state_path):
        with open(state_path, "rb") as fh:
            st = pickle.load(fh)
        es, trials, gen0 = st["es"], st["trials"], st["gen"]
        print(f"[sysid] resumed at gen {gen0} with {len(trials)} trials")
    if args_cli.sampler == "cma" and es is None:
        import cma
        es = cma.CMAEvolutionStrategy([0.0] * 4, args_cli.sigma0,
                                      {"popsize": pop, "bounds": [-1.0, 1.0], "seed": args_cli.seed + 1,
                                       "verbose": -9})
        try:
            es.inject([np.zeros(4)])   # the nominal plant is candidate 0 of gen 0 (baseline in the report)
        except Exception as ex:
            print(f"[sysid] WARN could not inject the nominal candidate: {ex}")

    trials_csv = os.path.join(out, "trials.csv")
    gens_csv = os.path.join(out, "generations.csv")
    cell_cols = [f"{c['name']}:{k}" for c in cells for k in ("J", "pitch_rms", "E0.4", "E0.6")]
    if not (args_cli.resume_run and os.path.exists(trials_csv)):
        with open(trials_csv, "w", newline="") as fh:
            csv.writer(fh).writerow(["gen", "cand", *PARAM_NAMES, "J_fit", "J_pred", *cell_cols])
        with open(gens_csv, "w", newline="") as fh:
            csv.writer(fh).writerow(["gen", "best_J_fit_gen", "mean_J_fit_gen", "best_J_fit_all",
                                     *[f"best_{n}" for n in PARAM_NAMES], "best_J_pred", "cma_sigma", "sec"])

    def row_cells(res):
        r = []
        for c in res["cells"]:
            r += [f"{c['J']:.4f}", f"{c['pitch_rms_deg']:.3f}", *[f"{v:.3f}" for v in c["stats"]["E"]]]
        return r

    cand, cell, seed = make_layout(pop, seeds)
    total_ticks = (args_cli.gens - gen0) * T_search + T_final
    bar = tqdm(total=total_ticks, unit="tick", dynamic_ncols=True, smoothing=0.02, mininterval=0.5,
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}")
    bar.set_postfix_str(f"gen 1/{args_cli.gens}")
    best = min(trials, key=lambda r: r["J_fit"]) if trials else None

    for gen in range(gen0, args_cli.gens):
        t0 = time.time()
        if args_cli.sampler == "cma":
            U = es.ask()
            P = np.stack([u_to_params(u) for u in U])
            ot_trials = None
        else:
            ot_trials = [study.ask() for _ in range(pop)]
            P = np.stack([[tr.suggest_float(n, -float(b), float(b)) for n, b in zip(PARAM_NAMES, BOUNDS)]
                          for tr in ot_trials])
            U = P / BOUNDS
        masks = apply_layout(P, cand, cell)
        first_term = rollout(masks, T_search, bar,
                             after_reset=(lambda: verify_overrides(cand, cell, P)) if gen == gen0 else None)
        results = evaluate(first_term, T_search, cand, cell, pop)
        J = np.array([r["J_fit"] for r in results])
        if args_cli.sampler == "cma":
            es.tell(list(U), list(J))
        with open(trials_csv, "a", newline="") as fh:
            w = csv.writer(fh)
            for c, r in enumerate(results):
                w.writerow([gen, c, *[f"{v:.5f}" for v in P[c]], f"{r['J_fit']:.4f}", f"{r['J_pred']:.4f}", *row_cells(r)])
        for c, r in enumerate(results):
            attrs = {"gen": gen, "J_pred": r["J_pred"],
                     **{f"{cc['name']}:J": cc["J"] for cc in r["cells"]},
                     **{f"{cc['name']}:E0.6": float(cc["stats"]["E"][-1]) for cc in r["cells"]}}
            try:
                if ot_trials is not None:
                    for kk, v in attrs.items():
                        ot_trials[c].set_user_attr(kk, v)
                    study.tell(ot_trials[c], float(r["J_fit"]))
                else:
                    study.add_trial(optuna.trial.create_trial(
                        params={n: float(v) for n, v in zip(PARAM_NAMES, P[c])}, distributions=dists,
                        value=float(r["J_fit"]), user_attrs=attrs))
            except Exception as ex:
                print(f"[sysid] WARN optuna record failed: {ex}")
            trials.append({"gen": gen, "cand": c, "params": P[c].tolist(), "J_fit": r["J_fit"], "J_pred": r["J_pred"],
                           "cells": [{"name": cc["name"], "J": cc["J"], **stats_json_clean(cc["stats"])}
                                     for cc in r["cells"]]})
        gbest = int(np.argmin(J))
        if best is None or results[gbest]["J_fit"] < best["J_fit"]:
            best = trials[-pop + gbest]
        sig = float(es.sigma) if es is not None else float("nan")
        sec = time.time() - t0
        with open(gens_csv, "a", newline="") as fh:
            csv.writer(fh).writerow([gen, f"{J.min():.4f}", f"{J.mean():.4f}", f"{best['J_fit']:.4f}",
                                     *[f"{v:.5f}" for v in best["params"]], f"{best['J_pred']:.4f}", f"{sig:.4f}", f"{sec:.1f}"])
        with open(state_path, "wb") as fh:
            pickle.dump({"es": es, "trials": trials, "gen": gen + 1}, fh)
        pb = results[gbest]
        bar.write(f"gen {gen + 1:3d}/{args_cli.gens}  J_fit min {J.min():6.2f} mean {J.mean():6.2f} | gen-best "
                  + " ".join(f"{n}={v:+.3f}" for n, v in zip(PARAM_NAMES, P[gbest]))
                  + f"  J_pred {pb['J_pred']:5.2f} | all-time best J_fit {best['J_fit']:.2f} J_pred {best['J_pred']:.2f}"
                  + f" | sigma {sig:.3f} | {sec:4.0f} s")
        bar.set_postfix_str(f"gen {min(gen + 2, args_cli.gens)}/{args_cli.gens} best J_fit {best['J_fit']:.2f}")

    # ── final: re-evaluate the top-k with every env at final length, dump rollouts, report ──
    k = max(1, min(args_cli.topk, len(trials)))
    order = sorted(range(len(trials)), key=lambda i: trials[i]["J_fit"])
    top_idx, seen = [], set()
    for i in order:
        key = tuple(np.round(trials[i]["params"], 4))
        if key not in seen:
            seen.add(key)
            top_idx.append(i)
        if len(top_idx) == k:
            break
    P_top = np.stack([trials[i]["params"] for i in top_idx])
    seeds_final = max(1, N // (k * n_cells))
    cand_f, cell_f, seed_f = make_layout(k, seeds_final)
    bar.set_postfix_str(f"final: top-{k} x {seeds_final} seeds/cell, {args_cli.final_episode_s:.0f} s")
    masks = apply_layout(P_top, cand_f, cell_f)
    first_term = rollout(masks, T_final, bar)
    final = evaluate(first_term, T_final, cand_f, cell_f, k)
    bar.close()

    for r, i in enumerate(top_idx):
        for kk, cinfo in enumerate(cells):
            envs = np.where((cand_f == r) & (cell_f == kk))[0][:2]
            for j, ei in enumerate(envs):
                save_npz(int(ei), first_term, T_final,
                         os.path.join(out, "topk", f"rank{r}_{cinfo['name'].replace('@', '_')}_s{j}.npz"),
                         {"policy": os.path.basename(POLICY_KEYS[cinfo["policy"]]), "cell": cinfo["name"],
                          "trim": cinfo["trim"], **{n: float(v) for n, v in zip(PARAM_NAMES, P_top[r])}})

    lines = []
    lines.append("=" * 78)
    lines.append(f"SYSID REPORT  {time.strftime('%Y-%m-%d %H:%M:%S')}   out={out}")
    lines.append(f"pins: total mass {args_cli.total_mass_kg} kg (mult {mass_mult:.4f}), floor friction x{args_cli.friction_mult}; "
                 f"{len(trials)} candidates in {args_cli.gens} gens; clean window {window} s; "
                 f"final re-eval {seeds_final} seeds/cell x {args_cli.final_episode_s:.0f} s")
    lines.append(f"fit cells: {[c['name'] for c in fit_cells]}   held-out: {[c['name'] for c in pred_cells]}")
    lines.append("")
    nominal = [tr for tr in trials if np.all(np.abs(np.asarray(tr["params"])) < 1e-9)]
    if nominal:
        lines.append(f"NOMINAL plant (all four offsets 0, same pins; {seeds} seeds/cell in the search): "
                     f"J_fit {nominal[0]['J_fit']:.2f}  J_pred {nominal[0]['J_pred']:.2f}   <- baseline")
    lines.append(f"TOP-{k} (by search J_fit; 'final' = re-evaluated with {seeds_final} seeds per cell):")
    lines.append("rank  " + "  ".join(f"{n:>10s}" for n in PARAM_NAMES) + "   J_fit(search)  J_fit(final)  J_pred(final)")
    for r, i in enumerate(top_idx):
        lines.append(f"{r:4d}  " + "  ".join(f"{v:+10.4f}" for v in P_top[r])
                     + f"   {trials[i]['J_fit']:12.2f}  {final[r]['J_fit']:12.2f}  {final[r]['J_pred']:13.2f}")
    if es is not None:
        try:
            m = u_to_params(es.result.xfavorite)
            sd = np.abs(np.asarray(es.result.stds) * BOUNDS)
            lines.append("CMA-ES final distribution (mean ± std, physical units): "
                         + ", ".join(f"{n} {a:+.4f}±{b:.4f}" for n, a, b in zip(PARAM_NAMES, m, sd)))
        except Exception as ex:
            lines.append(f"(CMA-ES distribution unavailable: {ex})")
    lines.append("")
    lines.append("BEST CANDIDATE (rank 0) — hardware vs sim per cell. In-window pitch (deg) and E = crossed 25 deg by "
                 "(objective); F = 45-deg fall fraction (REPORT ONLY: hardware F is tether-suspect, sim F is the 5-s prediction):")
    for kk, (cc, cinfo) in enumerate(zip(final[0]["cells"], cells)):
        tag = "FIT " if kk < n_fit else "PRED"
        lines.append(f"{tag} " + format_cell_table(cinfo["name"], cc["stats"], cinfo["hw"]))
        lines.append(f"      J {cc['J']:.2f}  (pitch term {cc['pitch_term']:.2f}, exceedance term {cc['exc_term']:.2f}, "
                     f"pitch rms {cc['pitch_rms_deg']:.1f} deg over {cc['n_grid_both']} grid pts)")
    try:
        try:
            imp = optuna.importance.get_param_importances(study)
            how = "fANOVA"
        except Exception:
            imp = optuna.importance.get_param_importances(
                study, evaluator=optuna.importance.PedAnovaImportanceEvaluator())
            how = "PED-ANOVA"
        lines.append(f"Optuna param importance ({how} on J_fit): " + ", ".join(f"{kk} {v:.2f}" for kk, v in imp.items()))
    except Exception as ex:
        lines.append(f"(param importance unavailable: {ex})")
    lines.append("")
    lines.append("READING GUIDE (pre-registered in the header): (a) J_fit(final) < ~2 and J_pred < ~3 -> offset real & "
                 "gait-independent (soft check: sim F pattern falls@-0.089/0, stands@+0.089); (b) fit ok, pred bad -> "
                 "stepping has an extra un-parameterized problem; (c) J_fit >~ 6 everywhere -> not a static offset of "
                 "these kinds; (d) top-k spread = degeneracy; (e) see sensitivity.txt for whether the window can identify it.")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(out, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    with open(os.path.join(out, "best.json"), "w") as fh:
        json.dump({"params": dict(zip(PARAM_NAMES, P_top[0].tolist())), "J_fit_final": final[0]["J_fit"],
                   "J_pred_final": final[0]["J_pred"], "window_s": window,
                   "pins": {"total_mass_kg": args_cli.total_mass_kg, "friction_mult": args_cli.friction_mult},
                   "topk": [{"params": dict(zip(PARAM_NAMES, P_top[r].tolist())), "J_fit_final": final[r]["J_fit"],
                             "J_pred_final": final[r]["J_pred"]} for r in range(k)],
                   "cells_final_rank0": [{"name": cc["name"], "J": cc["J"], **stats_json_clean(cc["stats"])}
                                         for cc in final[0]["cells"]]}, fh, indent=1)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
