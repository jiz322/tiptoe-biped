"""sweep_rest_absorption.py — S6 foot-restitution ABSORPTION A/B (eval only, NO
training).  Copied from sweep_plant_elasticity.py and edited for exactly one
question: does the S6_restDR policy (trained with foot-ball restitution as a
per-env DR band over 0.4-0.8) stay flat across e, where S4a degrades?

The two policies run side by side in one launch, on the same seeds, against
the same contact cell.  The contact material is created with the prim and
cloned per environment, so it cannot vary within a process: the four
restitution cells are four Isaac launches (the launcher's contact arms), each
holding a rigid foot (stiffness=0, damping=0) with a pinned restitution
{0.0, 0.3, 0.63, 0.8} and combine mode "max".

The belt axis is NOT swept here (pinned rigid, "0:0"), and no pad force is
present: the only thing that differs across cells is foot restitution.  The
physics scene runs bounce_threshold_velocity=0.05 so a low-speed ball contact
actually bounces instead of being damped out.

VERIFICATION (per cell, once, before the rollout is recorded): read back from
the live physics buffer that the two discovered ball rows carry the cell's
restitution (combine mode "max" is authored on the material prim) and that
physxScene:bounceThreshold is 0.05.  A sweep that does not prove this could
silently record the rigid plant on every cell and fake an absorption.

OUTPUT (<out>/): run.log, contact.json, cells.csv (one row per policy x cell x
seed), report.txt (the absorption table + verdict), and rollout.npz (pitch/
pitch-rate/root-vx/action/contact traces).

Usage: via RL/run_s6_eval_sweep_GPU1.ps1, which sets the Isaac env vars and
the S6 plant recipe (belt off, no pad, curriculum off — a fixed-policy
standing-start rollout).  --smoke is a short pipeline check.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fixed-policy belt/foot elasticity sweep.")
parser.add_argument("--task", type=str, default="Template-standup_curriculum-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--out", type=str, required=True, help="run directory (created)")
parser.add_argument("--policy", type=str, action="append", default=[],
                    metavar="NAME=PATH",
                    help="a policy checkpoint, repeatable. Two policies side by side = "
                         "two --policy flags. Several checkpoints under ONE name "
                         "(compare seeds of the same policy) = repeat the name.")
parser.add_argument("--policy-extra", type=str, action="append", default=[],
                    metavar="NAME=PATH",
                    help="additional checkpoint for an already-named policy, repeatable. "
                         "Exists so a launcher never has to pack several paths into one "
                         "argument with a separator character.")
parser.add_argument("--belt", type=str, default="0:0",
                    help="comma-separated joint-referred k:d cells; k=0 is the rigid plant. "
                         "d may be given as a ratio of critical damping with a 'z' suffix, "
                         "e.g. 8:0.7z")
parser.add_argument("--seeds", type=int, default=8, help="envs per policy x belt cell")
parser.add_argument("--num-envs", type=int, default=0,
                    help="scene size. 0 (default) = the exact number the layout needs "
                         "(policies x belt cells x seeds), rounded up. This MUST be set: "
                         "the task cfg's own num_envs is the training size (32768), and "
                         "leaving it there builds a training-scale scene for a fixed-"
                         "policy sweep.")
parser.add_argument("--episode-s", type=float, default=2.0)
parser.add_argument("--gait-phase-init", type=float, default=0.0,
                    help="hardware phase for every reset (the analysis fixes 0.0)")
parser.add_argument("--trim", type=float, default=0.0, help="pg_x obs trim as logged on hardware")
parser.add_argument("--contact-stiffness", type=float, default=0.0, help="N/m, 0 = rigid")
parser.add_argument("--contact-damping", type=float, default=0.0, help="N s/m, explicit")
parser.add_argument("--contact-damping-ratio", type=float, default=None,
                    help="foot contact damping as a ratio of the ball's own contact mode "
                         "(2*zeta*sqrt(k*m_eff)); the readable form across a stiffness arm")
parser.add_argument("--contact-effective-mass", type=float, default=1.0,
                    help="kg participating in a single-support catch (NOT the 5.57 kg body)")
parser.add_argument("--contact-restitution", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dry-run", action="store_true",
                    help="build the env, verify every override read-back, then exit")
parser.add_argument("--smoke", action="store_true", help="2 cells x 2 envs, short episode")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.smoke:
    if "--seeds" not in sys.argv:
        args_cli.seeds = 2
    if "--belt" not in sys.argv:
        args_cli.belt = "0:0"
    args_cli.episode_s = 1.0
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── everything else after the app is up ──────────────────────────────────────
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from pxr import PhysxSchema  # noqa: E402

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse, quat_rotate_inverse  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402

import RL.tasks  # noqa: F401,E402

BELT_G2 = (22.0 / 30.0) ** 2
# Motor-side belt rotor = G^2 * knee armature, and the knee armature is
# I_out/G^2 (belt_knee_actuator.derive_armature), so this is the RS01
# output-shaft inertia itself.  Keep ROTOR_INERTIA_OUTPUT equal to
# belt_knee_actuator.ROTOR_INERTIA_OUTPUT (data/_test_armature_derivation.py
# checks it); the product form spells the identity rather than hiding it.
ROTOR_INERTIA_OUTPUT = 4.2e-3    # vendor spec since 2026-09-18 (was 19.3e-3, bench apparent)
BELT_MOTOR_INERTIA = BELT_G2 * (ROTOR_INERTIA_OUTPUT / BELT_G2)   # = rl_env_cfg_p6._BELT_MOTOR_INERTIA
SERVO_KP_KNEE_DEFAULT = 30.0
SERVO_KD_KNEE_DEFAULT = 1.5
ROBOT_MASS_KG = 5.57          # weigh-in; used only for the critical-damping factor
CONTACT_WINDOW_S = 0.6        # the window the analysis treats as tether-free


class Tee:
    """Duplicate print() into <out>/run.log."""

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


def parse_policies(specs: list[str]) -> dict[str, list[str]]:
    """Collect repeated NAME=PATH flags into {name: [paths...]}.

    One argument per checkpoint, deliberately:  the launchers pass these
    through PowerShell, which splits an argument at whitespace and treats `;`
    as a statement separator, so any 'NAME=a,b,c' packing scheme breaks as
    soon as a second policy is added (observed 2026-09-11 — the first real
    run died with 'checkpoint not found: <R1r path> S4a=<path>').
    """
    out: dict[str, list[str]] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(
                f"--policy expects NAME=PATH, got {spec!r}. Pass one checkpoint per "
                "flag; to add a second checkpoint under the same name, repeat the name."
            )
        name, path = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise SystemExit(f"--policy has an empty name: {spec!r}")
        if not os.path.isfile(path):
            raise SystemExit(f"checkpoint not found for policy {name!r}: {path!r}")
        if name in out and path in out[name]:
            raise SystemExit(f"policy {name!r} lists the same checkpoint twice: {path}")
        out.setdefault(name, []).append(path)
    if not out:
        raise SystemExit("at least one --policy NAME=PATH is required")
    return out


def parse_belt(spec: str, servo_kp: float, servo_kd: float):
    """'k:d' cells -> [(k_joint, d_motor, label)].  FRAMES DIFFER, deliberately:
    k is knee-joint N m/rad (comparable against servo_kp_knee), d is
    MOTOR-side N m s/rad, matching RlEnvCfg.belt_series_* exactly.

    The 'z' form is the one the doc asks for ("sweep damping by damping ratio
    rather than guessing a final viscous coefficient").  zeta is taken against
    the belt's own mode, wn = sqrt(k_motor / I_rotor_motor), so

        d_motor = 2*zeta*sqrt(k_joint * G^2 * I_rotor_motor)

    and zeta = 1 is the transmission's own critical damping.  Note there is no
    G^2 on d itself: a torque-over-velocity coefficient carries no power of G,
    which is exactly why the cfg keeps it motor-side."""
    cells = []
    for token in [t.strip() for t in spec.split(",") if t.strip()]:
        if ":" not in token:
            raise SystemExit(f"--belt cell needs k:d, got {token!r}")
        k_s, d_s = token.split(":", 1)
        k = float(k_s)
        if k < 0.0:
            raise SystemExit(f"belt stiffness must be >= 0, got {k}")
        # Two spellings of the same idea. 'k:z<zeta>' is the readable one the
        # docs ask for; 'k:1z' is the launcher's pre-existing form where the
        # trailing z alone means zeta=1, and it is kept working because the
        # shipped ps1 reads it.
        if d_s.startswith("z") and len(d_s) > 1:
            zeta = float(d_s[1:])
        elif d_s == "z":
            zeta = 1.0
        elif d_s.endswith("z"):
            zeta = float(d_s[:-1])
        else:
            zeta = None
        if zeta is not None:
            if k == 0.0:
                raise SystemExit(f"cell {token}: a damping ratio needs a nonzero stiffness")
            if zeta < 0.0:
                raise SystemExit(f"cell {token}: damping ratio must be >= 0, got {zeta}")
            d = 2.0 * zeta * math.sqrt(k * BELT_G2 * BELT_MOTOR_INERTIA)
            label = f"k{k:g}_z{zeta:g}"
        else:
            d = float(d_s)
            if d < 0.0:
                raise SystemExit(f"belt damping must be >= 0, got {d}")
            label = f"k{k:g}_d{d:g}"
        if k == 0.0:
            label = "rigid"
        cells.append((k, d, label))
    if not cells:
        raise SystemExit("no belt cells")
    return cells


def install_gait_phase_pin(env, phase_init: float):
    orig = env._reset_idx

    def _pinned(env_ids):
        orig(env_ids)
        if env_ids is None:
            env._gait_phase[:] = phase_init
        else:
            env._gait_phase[env_ids] = phase_init

    env._reset_idx = _pinned


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    out = os.path.abspath(args_cli.out)
    os.makedirs(out, exist_ok=True)
    sys.stdout = Tee(os.path.join(out, "run.log"), sys.stdout)
    print(f"\n[sweep] {time.strftime('%Y-%m-%d %H:%M:%S')}  out={out}")
    print(f"[sweep] argv: {' '.join(sys.argv[1:])}")

    policies = parse_policies(args_cli.policy + args_cli.policy_extra)
    belt_cells = parse_belt(args_cli.belt, env_cfg.servo_kp_knee, env_cfg.servo_kd_knee)

    # Contact arm. Damping comes either explicitly (N s/m) or as a ratio of
    # the ball's own mode, and it is resolved HERE, before the stage exists,
    # because the material is created with the prim and cannot be changed from
    # inside the process. The envelope the env will build is the same one:
    # resolved_foot_contact_damping() is the single implementation.
    env_cfg.foot_contact_stiffness = float(args_cli.contact_stiffness)
    env_cfg.foot_restitution = float(args_cli.contact_restitution)
    env_cfg.foot_contact_effective_mass = float(args_cli.contact_effective_mass)
    if args_cli.contact_damping_ratio is not None:
        if args_cli.contact_damping != 0.0:
            raise SystemExit(
                "--contact-damping (explicit) and --contact-damping-ratio both set; "
                "the explicit value would silently win inside the cfg. Pass one."
            )
        env_cfg.foot_contact_damping = 0.0
        env_cfg.foot_contact_damping_ratio = float(args_cli.contact_damping_ratio)
    else:
        env_cfg.foot_contact_damping = float(args_cli.contact_damping)
        env_cfg.foot_contact_damping_ratio = float("nan")
    contact = {
        "stiffness": env_cfg.foot_contact_stiffness,
        "damping": env_cfg.resolved_foot_contact_damping(),
        "damping_ratio": (None if math.isnan(env_cfg.foot_contact_damping_ratio)
                          else env_cfg.foot_contact_damping_ratio),
        "effective_mass": env_cfg.foot_contact_effective_mass,
        "restitution": env_cfg.foot_restitution,
    }
    with open(os.path.join(out, "contact.json"), "w") as fh:
        json.dump(contact, fh, indent=1)

    # ── the sweep must not silently measure a cell it did not set ───────────
    if env_cfg.belt_series_stiffness_min != env_cfg.belt_series_stiffness_max:
        print("[sweep] note: cfg belt stiffness band is a RANGE; every env gets an "
              "explicit pinned value, so the band bounds are ignored here")
    if not env_cfg.belt_series_enable and any(k > 0.0 for k, _, _ in belt_cells):
        raise SystemExit(
            "belt cells request stiffness but env.belt_series_enable is False. "
            "Pass env.belt_series_enable=true (the env seeds the actuator buffers "
            "from it before the articulation is built)."
        )
    if contact["stiffness"] == 0.0 and contact["damping"] != 0.0:
        raise SystemExit("contact damping needs positive contact stiffness")

    # Size the scene to the layout BEFORE the stage is built.  A hidden
    # training-scale default would otherwise spawn 32768 robots for a handful
    # of fixed-policy rollouts and dominate the whole sweep's runtime.
    blocks_n = sum(len(paths) for paths in policies.values())
    need = blocks_n * len(belt_cells) * args_cli.seeds
    n_envs = args_cli.num_envs if args_cli.num_envs > 0 else need
    if n_envs < need:
        raise SystemExit(
            f"--num-envs {n_envs} is smaller than the layout needs ({need} = "
            f"{blocks_n} policy block(s) x {len(belt_cells)} belt cell(s) x {args_cli.seeds} seeds)"
        )
    if env_cfg.scene.num_envs != n_envs:
        print(f"[sweep] scene num_envs {env_cfg.scene.num_envs} -> {n_envs} "
              f"(layout needs {need}; {n_envs - need} spare envs stay unassigned)")
    env_cfg.scene.num_envs = n_envs

    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    # Seed so two contact arms are comparable: the DR draws (stiffness, delay,
    # CoM) must differ only in the arm under test, not in their random draw.
    env.reset(seed=args_cli.seed)
    install_gait_phase_pin(e, args_cli.gait_phase_init)
    e.manual_cmd_override = True
    e.manual_vx = 0.0
    e.manual_yaw_rate = 0.0
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    dev = e.device
    dt = float(e.step_dt)
    N = e.num_envs
    jids = e._joint_ids
    T = int(round(args_cli.episode_s / dt))

    # ── layout: env i -> (policy block, belt cell, seed) ────────────────────
    # (the scene was sized to `need` above, so the only spare envs are the
    #  rounding slack from an explicit --num-envs)
    blocks = [(name, path) for name, paths in policies.items() for path in paths]
    per_block = len(belt_cells) * args_cli.seeds
    need = len(blocks) * per_block
    i = np.arange(N)
    blk = np.where(i < need, i // per_block, -1)
    rem = i % per_block
    cell_of = rem // args_cli.seeds
    seed_of = rem % args_cli.seeds
    print(f"[sweep] {len(blocks)} policy block(s) x {len(belt_cells)} belt cell(s) x "
          f"{args_cli.seeds} seeds = {need} envs (num_envs {N}); episode {args_cli.episode_s} s = {T} steps")
    for b, (name, path) in enumerate(blocks):
        print(f"[sweep]   block {b}: {name} {path}")
    for c, (k, d, label) in enumerate(belt_cells):
        # Frames differ on purpose: k converts with G^2, d does NOT (a torque
        # over velocity carries no power of the reduction).  Printing d*G^2
        # here would advertise a motor-side damping the actuator never sees.
        print(f"[sweep]   cell  {c}: {label}  k={k:g} N m/rad (joint)"
              f"  [motor-side k={k * BELT_G2:.4g}]  d={d:g} N m s/rad (motor-side, no conversion)")
    print(f"[sweep] contact: {contact}  gait phase {args_cli.gait_phase_init}  "
          f"trim {args_cli.trim}  dt {dt:.4f} s")
    for k, d, label in belt_cells:
        if k > 0.0:
            zeta = d / (2.0 * math.sqrt(k * BELT_G2 * BELT_MOTOR_INERTIA))
            print(f"[sweep]   {label}: belt damping ratio zeta = {zeta:.3f} "
                  f"(wn = {math.sqrt(k * BELT_G2 / BELT_MOTOR_INERTIA):.1f} rad/s)")

    # ── per-env overrides through the env's existing hook ────────────────────
    belt_t = torch.zeros(N, 2, device=dev)          # joint-side stiffness
    belt_d = torch.zeros(N, 2, device=dev)          # motor-side damping
    for c, (k, d, _) in enumerate(belt_cells):
        sel = torch.as_tensor(cell_of == c, device=dev)
        belt_t[sel, :] = float(k)
        belt_d[sel, :] = float(d)
    e._dr_override = {
        "belt_series": belt_t,
        "belt_series_damping": belt_d,
        "proj_grav_bias_x": torch.full((N,), -float(args_cli.trim), device=dev),
    }

    # ── one policy instance per block ───────────────────────────────────────
    blocks_pol = {}
    for b, (name, path) in enumerate(blocks):
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(path)
        mask = torch.as_tensor(blk == b, device=dev)
        blocks_pol[b] = (runner.get_inference_policy(device=dev), runner.alg.policy, mask)
        print(f"[sweep] loaded {name} {os.path.basename(path)}")

    rec = {
        "pitch": torch.zeros(T, N, device=dev),
        "pitch_rate": torch.zeros(T, N, device=dev),
        "roll": torch.zeros(T, N, device=dev),
        "joint_pos": torch.zeros(T, N, 6, device=dev),
        "joint_vel": torch.zeros(T, N, 6, device=dev),
        "action": torch.zeros(T, N, 6, device=dev),
        # encoder-derived knee reads (obs contract under elasticity)
        "knee_enc": torch.zeros(T, N, 2, device=dev),
        "knee_spring": torch.zeros(T, N, 2, device=dev),
        "contact": torch.zeros(T, N, 2, device=dev),
        # forward velocity, body-frame x (play_sim2real convention)
        "root_vx": torch.zeros(T, N, device=dev),
    }
    rec["contact_sensors"] = rec.pop("contact")

    # ── verification: the sweep must prove it set what it claims ────────────
    def verify():
        ok = True
        hk = e.robot.actuators.get("hip_knee")
        if hk is not None:
            mask = hk.armature_write_mask()
            # Rigid pairs (k = 0) keep their configured armature: mask == 1.
            is_elastic = belt_t[:, 0] > 0.0
            for leg in range(2):
                got = mask[:, hk._kn_cols[leg].item()] > 0.5      # 1 where rigid
                if not torch.equal(got.cpu(), (~is_elastic).cpu()):
                    print(f"[verify] FAIL armature mask leg {leg} disagrees with the belt cell")
                    ok = False
            el = is_elastic.nonzero(as_tuple=True)[0]
            if len(el) > 0:
                j = int(el[0])
                print("[verify] env {}: belt k={:.4g} d={:.4g} (joint) -> actuator k={:.4g} d={:.4g} (motor), "
                      "enabled={}".format(
                          j, belt_t[j, 0].item(), belt_d[j, 0].item(),
                          hk._series_stiffness[j, 0].item(), hk._series_damping[j, 0].item(),
                          bool(hk._series_enabled[j, 0])))
                if not hk._series_enabled[j, 0]:
                    print("[verify] FAIL an elastic cell did not enable the series path")
                    ok = False
                # float32 storage, so compare at float32 tolerance.
                if not math.isclose(hk._series_stiffness[j, 0].item(),
                                    belt_t[j, 0].item() * BELT_G2, rel_tol=1e-4):
                    print("[verify] FAIL motor-side stiffness is not k_joint * G^2")
                    ok = False
                # Damping must arrive UNCHANGED: it is already motor-side.
                # rel_tol 1e-5 would be tighter than float32, where the stored
                # value already differs from the requested one at ~2e-8.
                if not math.isclose(hk._series_damping[j, 0].item(),
                                    belt_d[j, 0].item(), rel_tol=1e-4):
                    print("[verify] FAIL motor-side damping does not equal the requested value "
                          f"({hk._series_damping[j, 0].item():.6g} vs {belt_d[j, 0].item():.6g})")
                    ok = False
            rigid = (~is_elastic).nonzero(as_tuple=True)[0]
            if len(rigid) > 0:
                j = int(rigid[0])
                if bool(hk._series_enabled[j, 0]):
                    print("[verify] FAIL a rigid cell enabled the series path")
                    ok = False
        pg = e._dr_proj_grav_bias[:, 0].cpu().numpy()
        if not np.allclose(pg, -args_cli.trim, atol=1e-6):
            print("[verify] FAIL pg_x trim bias not applied")
            ok = False
        print(f"[verify] contact material: stiffness={contact['stiffness']} "
              f"damping={contact['damping']} restitution={contact['restitution']} "
              f"(rigid branch: {contact['stiffness'] == contact['damping'] == contact['restitution'] == 0.0})")
        # ── the two things an absorption sweep must prove per cell ──────────
        # (a) the ball rows actually hold the cell's restitution (the material
        #     spawner writes it; combine mode "max" is authored on the prim so
        #     the ball's value wins over the ground's 0).
        try:
            mats = e.robot.root_physx_view.get_material_properties()
            rows = e._foot_restitution_rows if e._foot_restitution_rows else ()
            if not rows:
                # DR band is off in this eval, so the env never discovers the
                # rows: find them the same way, by matching the carrier.
                hit = (mats[:, :, 2].abs() - float(env_cfg.foot_restitution)).abs() < 1e-4
                rows = tuple(int(x) for x in torch.nonzero(hit[0]).flatten().tolist())
            vals = sorted({float(mats[0, r, 2].item()) for r in rows}) if rows else []
            if rows and not all(abs(v - args_cli.contact_restitution) < 1e-4 for v in vals):
                print(f"[verify] FAIL ball-row restitution {vals} != cell "
                      f"{args_cli.contact_restitution:g} (rows {rows})")
                ok = False
            else:
                print(f"[verify] ball rows {rows} restitution {vals} == cell "
                      f"{args_cli.contact_restitution:g}")
            if args_cli.contact_restitution != 0.0 and not rows:
                print("[verify] FAIL nonzero cell but no ball rows matched the carrier")
                ok = False
        except Exception as exc:  # noqa: BLE001
            print(f"[verify] FAIL could not read back ball-row restitution: {exc}")
            ok = False
        # combine mode "max" must be authored on the foot material prim, so the
        # ball's value wins over the ground's 0 (ground combine mode multiply
        # would otherwise zero the bounce).
        try:
            stage = e.sim.stage
            modes = []
            for prim in stage.Traverse():
                if prim.GetName().startswith("foot_contact_material_"):
                    px = PhysxSchema.PhysxMaterialAPI(prim)
                    attr = px.GetRestitutionCombineModeAttr() if px else None
                    modes.append((str(prim.GetPath()),
                                  attr.Get() if attr else None))
            bad = [(p, v) for p, v in modes if v not in ("max", 2)]
            if args_cli.contact_restitution != 0.0 and (not modes or bad):
                print(f"[verify] FAIL foot-material combine modes not all 'max': {modes}")
                ok = False
            else:
                print(f"[verify] foot-material restitution_combine_mode: {modes}")
        except Exception as exc:  # noqa: BLE001
            print(f"[verify] FAIL could not read back combine mode: {exc}")
            ok = False
        # (b) the physics scene runs the low-speed bounce threshold, not the
        #     0.5 default that would flatten the whole curve.
        try:
            stage = e.sim.stage
            prim = stage.GetPrimAtPath(sim_utils.SimulationCfg().physics_prim_path)
            attr = prim.GetAttribute("physxScene:bounceThreshold")
            thr = float(attr.Get()) if attr else None
            if thr is None or abs(thr - 0.05) > 1e-9:
                print(f"[verify] FAIL physxScene:bounceThreshold = {thr!r}, want 0.05")
                ok = False
            else:
                print(f"[verify] physxScene:bounceThreshold = {thr} (0.05)")
        except Exception as exc:  # noqa: BLE001
            print(f"[verify] FAIL could not read back bounceThreshold: {exc}")
            ok = False
        return ok

    # ── rollout ─────────────────────────────────────────────────────────────
    print("[sweep] reset + rollout ...")
    with torch.inference_mode():
        obs, _ = env.reset()
        e._gait_phase[:] = args_cli.gait_phase_init
        all_true = torch.ones(N, dtype=torch.bool, device=dev)
        for _, nn_, _ in blocks_pol.values():
            nn_.reset(all_true)

        if not verify():
            raise SystemExit("override verification failed; not recording a sweep on unverified knobs")

        if args_cli.dry_run:
            print("[sweep] --dry-run: verification passed, exiting before the rollout")
            return

        first_term = torch.full((N,), T, dtype=torch.long, device=dev)
        t0 = time.time()
        for s in range(T):
            is_t = isinstance(obs, torch.Tensor)
            ot = obs if is_t else obs["policy"]
            act = torch.zeros(N, 6, device=dev)
            for _, (pol, _, mask) in blocks_pol.items():
                if mask.any():
                    act[mask] = pol(ot[mask] if is_t else obs[mask])
            rq = e.robot.data.root_quat_w
            pg = quat_apply_inverse(rq, e._gravity_vec)
            # Same convention as sysid_metrics.pitch_tilt_deg and the env's own
            # standup tracker: pitch = asin(pg_x), negative = leaning back.
            rec["pitch"][s] = torch.asin(pg[:, 0].clamp(-1.0, 1.0))
            rec["roll"][s] = torch.asin(pg[:, 1].clamp(-1.0, 1.0))
            # forward velocity in the BASE frame (play_sim2real convention),
            # not world x: the robot can yaw.
            rec["root_vx"][s] = quat_rotate_inverse(rq, e.robot.data.root_lin_vel_w)[:, 0]
            rec["joint_pos"][s] = e.robot.data.joint_pos[:, jids]
            rec["joint_vel"][s] = e.robot.data.joint_vel[:, jids]
            rec["action"][s] = act
            hk = e.robot.actuators.get("hip_knee")
            if hk is not None:
                # Actuator-LOCAL slice: the pair columns index the 4-joint
                # hip_knee group, not the 6-joint array (fixed 2026-09-15).
                q_enc, qd_enc = hk.encoder_joint_state(
                    e.robot.data.joint_pos[:, hk.joint_indices],
                    e.robot.data.joint_vel[:, hk.joint_indices])
                rec["knee_enc"][s] = q_enc
                rec["knee_spring"][s] = hk.series_spring_effort
            obs, _, _, _ = env.step(act)
            fresh = (e.reset_terminated | e.reset_time_outs) & (first_term == T)
            first_term[fresh] = s + 1
            if (s + 1) % max(1, T // 10) == 0:
                print(f"[sweep]   {s + 1}/{T} steps  ({time.time() - t0:.1f} s)")
        rec["pitch_rate"][1:] = (rec["pitch"][1:] - rec["pitch"][:-1]) / dt
        rec["pitch_rate"][0] = rec["pitch_rate"][1]

    # ── per-cell statistics ────────────────────────────────────────────────
    t = np.arange(T) * dt
    pitch = rec["pitch"].cpu().numpy()
    prate = rec["pitch_rate"].cpu().numpy()
    ft = first_term.cpu().numpy()
    tilt = np.degrees(np.abs(pitch))

    def at_time(trace, tt):
        k = int(round(tt / dt))
        return trace[k] if k < len(trace) else np.nan

    rows = []
    for b, (name, path) in enumerate(blocks):
        for c, (k, d, label) in enumerate(belt_cells):
            sel = np.where((blk == b) & (cell_of == c))[0]
            if len(sel) == 0:
                continue
            for j in sel:
                n = int(min(ft[j], T))
                inwin = t[:n] <= CONTACT_WINDOW_S + 1e-9
                cross = np.where(tilt[:n] >= 25.0)[0]
                rows.append({
                    "policy": name,
                    "cell": label,
                    "k_joint": k,
                    "d_joint": d,
                    "seed": int(seed_of[j]),
                    "env": int(j),
                    "survived_s": float(n * dt),
                    "mean_vx": float(np.mean(rec["root_vx"][:n, j].cpu().numpy())) if n else np.nan,
                    "pitch_0.2_deg": float(np.degrees(at_time(pitch[j], 0.2))),
                    "pitch_0.4_deg": float(np.degrees(at_time(pitch[j], 0.4))),
                    "pitch_0.6_deg": float(np.degrees(at_time(pitch[j], 0.6))),
                    "min_rate_deg_s": float(np.degrees(prate[j, :n].min())) if n else np.nan,
                    "tilt25_s": float(t[cross[0]]) if len(cross) else float("inf"),
                    "tilt25_inwin": bool(len(cross) and t[cross[0]] <= CONTACT_WINDOW_S),
                    "in_window_ok": bool(n * dt >= CONTACT_WINDOW_S),
                    "_inwin_ok": bool(inwin.all() or n >= int(CONTACT_WINDOW_S / dt)),
                })

    # ── aggregate: one row per policy, for THIS launch's contact cell ───────
    def agg(policy_name):
        sub = [r for r in rows if r["policy"] == policy_name]
        if not sub:
            return None
        surv = np.array([min(r["survived_s"], args_cli.episode_s) for r in sub])
        return {
            "n": len(sub),
            "e": float(args_cli.contact_restitution),
            "mean_ep_len": float(surv.mean()),
            "fall_rate": float(np.mean([r["survived_s"] < args_cli.episode_s - 1e-9 for r in sub])),
            "mean_vx": float(np.nanmean([r["mean_vx"] for r in sub])),
            "min_rate": float(np.nanmean([r["min_rate_deg_s"] for r in sub])),
        }

    lines = []
    lines.append(f"sweep_rest_absorption  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"contact: {contact}")
    lines.append(f"gait phase {args_cli.gait_phase_init}, trim {args_cli.trim}, "
                 f"{T} steps @ {dt:.4f} s, {args_cli.seeds} seeds/cell")
    lines.append("")
    lines.append(f"── cell e = {args_cli.contact_restitution:g}")
    lines.append(f"{'policy':>8} {'n':>3} {'ep_len_s':>9} {'fall_rate':>10} {'mean_vx':>9} "
                 f"{'min_rate':>9}")
    for name in policies:
        a = agg(name)
        if a is None:
            continue
        lines.append(f"{name:>8} {a['n']:>3} {a['mean_ep_len']:>9.2f} {a['fall_rate']:>10.2f} "
                     f"{a['mean_vx']:>9.3f} {a['min_rate']:>9.1f}")
    lines.append("")
    lines.append("Absorption is computed ACROSS launches by the launcher, which "
                 "collects the per-cell (policy, e) rows above into one table.")
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(out, "report.txt"), "w") as fh:
        fh.write(report + "\n")

    # machine-readable per-cell summary, so the launcher can assemble the
    # cross-cell absorption table without parsing the human report.
    _summ = {name: agg(name) for name in policies}
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump({
            "e": float(args_cli.contact_restitution),
            "episode_s": float(args_cli.episode_s),
            "seeds": int(args_cli.seeds),
            "trim": float(args_cli.trim),
            "policies": {name: v for name, v in _summ.items() if v is not None},
        }, fh, indent=1)

    with open(os.path.join(out, "cells.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[k for k in rows[0].keys() if not k.startswith("_")])
        w.writeheader()
        for r in rows:
            w.writerow({k: v for k, v in r.items() if not k.startswith("_")})

    np.savez_compressed(
        os.path.join(out, "rollout.npz"),
        **{k: v.cpu().numpy().astype(np.float32) for k, v in rec.items()},
        first_term=ft.astype(np.int32),
        block=blk.astype(np.int32),
        cell=cell_of.astype(np.int32),
        seed=seed_of.astype(np.int32),
        belt_stiffness_joint=belt_t.cpu().numpy().astype(np.float32),
        belt_damping_motor=belt_d.cpu().numpy().astype(np.float32),
        dt=np.float32(dt),
        blocks=np.array([f"{n}|{p}" for n, p in blocks]),
        belt_labels=np.array([c[2] for c in belt_cells]),
        contact=json.dumps(contact),
    )
    print(f"[sweep] wrote {out}\\report.txt, cells.csv, rollout.npz")


if __name__ == "__main__":
    main()
    simulation_app.close()
