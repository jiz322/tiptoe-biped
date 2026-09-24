"""Stand-up ramp SHAPE search: candidate keyframe profiles vs the legacy ramp.

Companion to probe_standup_ramp.py / probe_standup_traj.py (09-08 finding: the
legacy linear joint-space ramp topples forward by itself at every d). Each
candidate sets cfg.standup_ramp_mode="keyframe" with its own hip / knee timing
profiles (flat [s,f,...] lists, see rl_env_cfg_p6 §6) — the special name
"legacy" runs the linear P6 ramp for reference and "path" the cfg path mode
(spawn on the path, standup_path_knots). Per (candidate, d, mode) it
spawns every env at exactly d and runs zero actions ("zero" = the ramp alone)
or the checkpoint ("policy"), recording per control step over alive envs:

  z          root height                       pitch   signed torso pitch (deg, kneel < 0)
  wy         body pitch rate (deg/s)           com_off whole-body CoM ahead of toe-mid (m, heading axis)
  com_ball   CoM ahead of the foot-BALL contact mid (m) — the true static margin
  root_off   root ahead of toe-mid (m)         contact toe contact fraction
  knee_h     min shin-origin height above ground (≈0.027 = knee resting)
  knee, hip  mean |joint| (rad)                alive   alive fraction

plus a ground-contact map (fraction of envs with each body's link origin below
--body_ground_z) at phases s = 0, .25, .5, .75, 1 of the ramp. Summary line per
run: state at grace end (alive, z, pitch, |wy|, com_off, contact), peak |com_off|
during the ramp, survival ≥1 s / ≥3 s after grace. Saves everything to one npz
(keys "<cand>/<mode>_d<d>/<key>"); plot with data/_plot_standup_keyframes.py.

Candidate syntax (ONE argument, ';'-separated):  name:hip_s0,f0,s1,f1,...:knee_s0,f0,...
  e.g.  "legacy;linear:0,0,1,1:0,0,1,1;hipsfirst:0,0,0.5,1,1,1:0,0,0.5,0,1,1"

Example (GPU 2, nominal plant):
  python scripts/rsl_rl/probe_standup_keyframes.py --task Template-standup_curriculum-v0 \
      --headless --num_envs 256 --load_run <run> --checkpoint model_4840.pt \
      --alphas 0.7,1.0 --modes zero,policy --out logs/standup_kf.npz --csv logs/standup_kf.csv \
      --candidates "legacy;hipsfirst:0,0,0.5,1,1,1:0,0,0.5,0,1,1" \
      env.dr_enable=false <recipe overrides...>
"""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Stand-up keyframe shape probe.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--alphas", type=str, default="0.7,1.0")
parser.add_argument("--grace_s", type=float, default=None, help="override standup_grace_seconds")
parser.add_argument("--modes", type=str, default="zero,policy")
parser.add_argument("--candidates", type=str, required=True,
                    help="';'-separated list (a single argument, so hydra overrides after it survive)")
parser.add_argument("--horizon_after_grace", type=int, default=300)
parser.add_argument("--body_ground_z", type=float, default=0.06, help="link origin below this = on the ground")
parser.add_argument("--out", type=str, required=True, help="npz path")
parser.add_argument("--csv", type=str, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import RL.tasks  # noqa: F401


def _parse_candidates(specs):
    out = []
    for spec in [s.strip() for s in specs.split(";") if s.strip()]:
        if spec in ("legacy", "path"):
            out.append((spec, None, None))
            continue
        parts = spec.split(":")
        if len(parts) != 3:
            raise SystemExit(f"bad candidate '{spec}' — want name:hip_profile:knee_profile")
        hip = [float(v) for v in parts[1].split(",")]
        knee = [float(v) for v in parts[2].split(",")]
        out.append((parts[0], hip, knee))
    return out


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.curriculum_enable = True
    env_cfg.push_enable = False
    env_cfg.pickup_enable = False
    if args_cli.grace_s is not None:
        env_cfg.standup_grace_seconds = args_cli.grace_s

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint and os.path.isabs(args_cli.checkpoint):
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    raw = env.unwrapped
    raw.cfg.curriculum_alpha_step = 0.0
    raw._force_fixed_difficulty = True
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] loading {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=raw.device)

    n = raw.num_envs
    dt = raw._control_dt
    body_names = list(raw.robot.body_names)
    print(f"[INFO] bodies: {body_names}")
    fwd_b = torch.tensor([1.0, 0.0, 0.0], device=raw.device).expand(n, -1)
    kq = raw._kneel_quat.unsqueeze(0)
    pg_k = quat_apply_inverse(kq, raw._gravity_vec[:1])
    kneel_pitch_sign = -1.0 if math.asin(max(-1.0, min(1.0, -pg_k[0, 0].item()))) > 0 else 1.0

    def signed_pitch_deg(quat):
        pg = quat_apply_inverse(quat, raw._gravity_vec)
        return kneel_pitch_sign * torch.rad2deg(torch.asin((-pg[:, 0]).clamp(-1, 1)))

    # foot BALL collider centres, MEASURED from the live stage.  The nail point from
    # _get_toe_pos_w sits 5.8 cm up the shin → only a contact proxy when the shin is
    # vertical, so CoM margins are referenced to the ball here.  These used to be
    # hardcoded as bbox centres with r=0.0205 (read 09-08); 'part_3' is a 40 mm ball
    # plus a 30.6 mm-reach boss, so the bbox centre is off by 1.03 mm and the bbox
    # radius reads 20.81 mm.  Measured 2026-09-13: R=20.003 mm.
    from RL.tasks.direct.standup_curriculum.pad_contact import measure_ball_geometry
    _geom_paths = tuple(f"/World/envs/env_0/Robot/{nm}"
                        for nm in (raw.cfg.foot_body_name_1, raw.cfg.foot_body_name_2))
    _offs, BALL_R = measure_ball_geometry(
        raw.sim.stage,
        (raw.cfg.foot_body_name_1, raw.cfg.foot_body_name_2),
        _geom_paths)
    ball_off = {raw.cfg.foot_body_name_1: _offs[0].to(raw.device),
                raw.cfg.foot_body_name_2: _offs[1].to(raw.device)}
    ball_ids = [body_names.index(raw.cfg.foot_body_name_1), body_names.index(raw.cfg.foot_body_name_2)]
    print(f"[GEOM] ball measured from the stage: R={BALL_R:.6f} m  offsets="
          f"{ {k: [round(x, 6) for x in v.tolist()] for k, v in ball_off.items()} }")

    def ball_centres():
        out = []
        for nm, b in zip((raw.cfg.foot_body_name_1, raw.cfg.foot_body_name_2), ball_ids):
            pos = raw.robot.data.body_link_pos_w[:, b]
            qq = raw.robot.data.body_link_quat_w[:, b]
            out.append(pos + quat_apply(qq, ball_off[nm].unsqueeze(0).expand(n, -1)))
        return out

    def com_w():
        masses = raw.robot.root_physx_view.get_masses().to(raw.device)          # [N,B]
        com = raw.robot.data.body_com_pos_w                                      # [N,B,3]
        return (com * masses.unsqueeze(-1)).sum(1) / masses.sum(1, keepdim=True)

    cands = _parse_candidates(args_cli.candidates)
    alphas = [float(a) for a in args_cli.alphas.split(",")]
    modes = args_cli.modes.split(",")
    phases = [0.0, 0.25, 0.5, 0.75, 1.0]
    keys = ["z", "pitch", "wy", "com_off", "com_ball", "root_off", "contact", "knee_h", "knee", "hip", "alive"]
    out = {"body_names": np.array(body_names)}
    rows = []
    hdr = ("cand         mode   d    | @grace end: alive  z     pitch   |wy|  com_off com_ball contact | peak|com| | surv1s surv3s")
    print("\n" + hdr)
    for cname, hip_prof, knee_prof in cands:
        if hip_prof is None:
            raw.cfg.standup_ramp_mode = "linear" if cname == "legacy" else "path"
        else:
            raw.cfg.standup_ramp_mode = "keyframe"
            raw.cfg.standup_hip_profile = hip_prof
            raw.cfg.standup_knee_profile = knee_prof
            raw._build_standup_profiles()
        for d in alphas:
            raw._curriculum_alpha = d
            grace = int(d * raw._max_grace_steps)
            horizon = grace + args_cli.horizon_after_grace
            snap_steps = {int(round(p * max(grace - 1, 0))): p for p in phases}
            for mode in modes:
                rec = {k: np.full((horizon, 3), np.nan, dtype=np.float32) for k in keys}
                ground_map = np.full((len(phases), len(body_names)), np.nan, dtype=np.float32)
                first_len = torch.full((n,), -1, dtype=torch.long, device=raw.device)
                with torch.inference_mode():
                    obs, _ = env.reset()
                    alive = torch.ones(n, dtype=torch.bool, device=raw.device)
                    for t in range(horizon):
                        z = raw.robot.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
                        quat = raw.robot.data.root_quat_w
                        pitch = signed_pitch_deg(quat)
                        wy = kneel_pitch_sign * torch.rad2deg(raw.robot.data.root_ang_vel_b[:, 1])
                        toe1, toe2 = raw._get_toe_pos_w()
                        toe_mid = 0.5 * (toe1 + toe2)
                        fwd_w = quat_apply(quat, fwd_b)
                        fwd_w[:, 2] = 0.0
                        fwd_w = fwd_w / fwd_w.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                        root_off = ((raw.robot.data.root_pos_w[:, :2] - toe_mid[:, :2]) * fwd_w[:, :2]).sum(-1)
                        cw = com_w()
                        com_off = ((cw[:, :2] - toe_mid[:, :2]) * fwd_w[:, :2]).sum(-1)
                        bl1, bl2 = ball_centres()
                        ball_mid = 0.5 * (bl1 + bl2)
                        com_ball = ((cw[:, :2] - ball_mid[:, :2]) * fwd_w[:, :2]).sum(-1)
                        knee_h = (raw.robot.data.body_link_pos_w[:, ball_ids, 2].min(dim=-1).values
                                  - raw.scene.env_origins[:, 2])
                        contact = ((toe1[:, 2] < raw.cfg.no_flight_contact_threshold).float()
                                   + (toe2[:, 2] < raw.cfg.no_flight_contact_threshold).float()) * 0.5
                        jp = raw.robot.data.joint_pos[:, raw._joint_ids]
                        knee = 0.5 * (jp[:, 2].abs() + jp[:, 5].abs())
                        hip = 0.5 * (jp[:, 1].abs() + jp[:, 4].abs())
                        if t in snap_steps and alive.any():
                            bz = raw.robot.data.body_link_pos_w[:, :, 2] - raw.scene.env_origins[:, 2:3]
                            ground_map[phases.index(snap_steps[t])] = (
                                (bz[alive] < args_cli.body_ground_z).float().mean(0).cpu().numpy())
                        vals = dict(z=z, pitch=pitch, wy=wy, com_off=com_off, com_ball=com_ball, root_off=root_off,
                                    contact=contact, knee_h=knee_h, knee=knee, hip=hip)
                        if alive.any():
                            for k, v in vals.items():
                                va = v[alive].float()
                                rec[k][t] = (va.mean().item(), torch.quantile(va, 0.1).item(),
                                             torch.quantile(va, 0.9).item())
                        rec["alive"][t] = alive.float().mean().item()
                        if mode == "zero":
                            actions = torch.zeros(n, raw.cfg.action_space, device=raw.device)
                        else:
                            actions = policy(obs)
                        obs, _, dones, _ = env.step(actions)
                        newly = (dones > 0) & alive
                        first_len[newly] = t + 1
                        alive = alive & ~newly
                tag = f"{cname}/{mode}_d{d:.2f}"
                for k in keys:
                    out[f"{tag}/{k}"] = rec[k]
                out[f"{tag}/ground_map"] = ground_map
                out[f"{tag}/grace"] = np.array(grace)
                ep_len = torch.where(first_len < 0, torch.full_like(first_len, horizon), first_len)
                post = (ep_len - grace).clamp(min=0)
                surv1 = (post >= 100).float().mean().item()
                surv3 = (post >= 300).float().mean().item()
                ge = max(grace - 1, 0)
                alive_ge = rec["alive"][ge, 0]
                peak_com = float(np.nanmax(np.abs(rec["com_off"][:max(grace, 1), 0])))
                print(f"{cname:12s} {mode:6s} {d:.2f} | {alive_ge:5.2f} {rec['z'][ge,0]:6.3f} {rec['pitch'][ge,0]:7.1f} "
                      f"{abs(rec['wy'][ge,0]):6.1f} {rec['com_off'][ge,0]:7.3f} {rec['com_ball'][ge,0]:8.3f} {rec['contact'][ge,0]:6.2f} | "
                      f"{peak_com:8.3f} | {surv1:6.2f} {surv3:6.2f}")
                gm = " | ".join(
                    f"s={p:.2f}: " + ",".join(f"{body_names[b]}={ground_map[i, b]:.2f}"
                                             for b in range(len(body_names)) if ground_map[i, b] > 0.2)
                    for i, p in enumerate(phases))
                print(f"    ground: {gm}")
                rows.append((cname, mode, d, alive_ge, rec['z'][ge, 0], rec['pitch'][ge, 0], abs(rec['wy'][ge, 0]),
                             rec['com_off'][ge, 0], rec['contact'][ge, 0], peak_com, surv1, surv3))
    np.savez(args_cli.out, **out)
    print(f"[INFO] wrote {args_cli.out}")
    if args_cli.csv:
        with open(args_cli.csv, "w") as f:
            f.write("cand,mode,d,alive_ge,z_ge,pitch_ge,wy_abs_ge,com_off_ge,contact_ge,peak_com_off,surv1s,surv3s\n")
            for r in rows:
                f.write(",".join(f"{v}" for v in r) + "\n")
        print(f"[INFO] wrote {args_cli.csv}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
