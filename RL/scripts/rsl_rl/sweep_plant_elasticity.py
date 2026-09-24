"""sweep_plant_elasticity.py — fixed-policy virtual sweep of the two plant
mechanisms the 09-11 S4 analysis left open (docs/s4-stepping-gate-failure-
analysis-20260911.md, "Revised plan: virtual elasticity sweep"):

    1. belt series elasticity  — a motor-side spring/damper between the knee
       motor and the knee joint (BeltKneeDCMotor's series path)
    2. foot normal compliance / restitution — a contact material on the two
       ball collision meshes (foot_contact.FootContactUsdFileCfg)

WHY A SWEEP AND NOT DR TRAINING.  The hardware develops backward sagittal
momentum during the first dynamic catch step that the sim arrests (~-340 to
-443 deg/s on S4 vs ~-175 in the twin).  Both candidates are energy-storage
mechanisms, so they are cheap to evaluate as one-axis and two-axis FIXED
plants before any retraining: the question is only whether either reproduces
the hardware's opening signature and whether it does so for S4 and NOT for
R1r (a candidate that breaks both policies is not a stepping explanation).

CONSTRAINTS THE ENVIRONMENT NOW ENFORCES (see the 09-11 fixes):
  * Belt stiffness/damping are JOINT-referred in the CLI (comparable against
    servo_kp_knee=30, servo_kd_knee=1.5); the env converts once with (22/30)^2.
  * Under elasticity the knee observation channels read the MOTOR ENCODER
    (G*m - c*q_hip), not the belt-free PhysX joint, so the policy sees what the
    deploy stack can actually produce.
  * The reflected rotor armature is removed from the PhysX joint (knee, and
    the coupled hip-pitch diagonal) for elastic envs; the rotor lives inside
    the series integrator instead.

GRID.  One axis at a time, plus the pairings the analysis named.  Arms are
enumerated in the ps1 launcher; this script takes a finite list of (k, d)
belt cells and ONE contact setting, because the contact material is created at
spawn and clone time and therefore cannot vary within a process.

  POLICIES.  Any set of checkpoints, one per env block.  Default R1r@3150
  (the deployed stand-in) and the S4a Stepping policy under test.

  READING (pre-registered before the first run — apply verbatim):

  P1. If belt elasticity is the mechanism, the surviving cells are the SOFT
      ones (k_belt/u_servo <= 1) and they reproduce the S4 opening WITHOUT a
      large static pitch offset: |pitch(0.2 s)| still small, but min pitch
      rate after 0.2 s moves >= 1.5x toward the hardware's -340 deg/s, and the
      same cell leaves R1r recoverable.  A cell that needs k >> servo_kp to
      show any effect is numerically the rigid plant and answers nothing.
  P2. If foot compliance/restitution is the mechanism, the divergence starts
      AT the first impact, not before: pitch traces of rigid and compliant
      cells agree until the first foot touchdown, then separate, and the
      swing-foot contact time grows.  A cell where the traces diverge before
      any contact is not a contact explanation.
  P3. If neither arm moves min pitch rate by >= 1.2x at any cell while S4
      still fails in sim, both mechanisms are out for this failure and the
      report must say so plainly (no "trend in the right direction").
  P4. Ranking is by the hardware-trustworthy signals only: pitch at
      0.2/0.4/0.6 s, min pitch rate, tilt-25-deg crossing time, and survival.
      Do NOT rank by the 45-deg fall time (tether-suspect on hardware).

OUTPUT (<out>/): run.log, contact.json, cells.csv (one row per policy x belt
cell x seed), report.txt (ranked per policy, with the P1-P4 verdict), and
rollout.npz (pitch/pitch-rate/action/contact traces, sysid_optuna layout).

Usage: via RL/run_s4_elasticity_sweep_GPU2.ps1, which sets the Isaac env vars,
the S4 measured-plant recipe, and loops the contact arms.  --smoke is a
2-minute pipeline check (2 cells, 2 envs each).
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
        args_cli.belt = "0:0,8:1"
    args_cli.episode_s = 1.0
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── everything else after the app is up ──────────────────────────────────────
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

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
    }

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
                    "pitch_0.2_deg": float(np.degrees(at_time(pitch[j], 0.2))),
                    "pitch_0.4_deg": float(np.degrees(at_time(pitch[j], 0.4))),
                    "pitch_0.6_deg": float(np.degrees(at_time(pitch[j], 0.6))),
                    "min_rate_deg_s": float(np.degrees(prate[j, :n].min())) if n else np.nan,
                    "tilt25_s": float(t[cross[0]]) if len(cross) else float("inf"),
                    "tilt25_inwin": bool(len(cross) and t[cross[0]] <= CONTACT_WINDOW_S),
                    "in_window_ok": bool(n * dt >= CONTACT_WINDOW_S),
                    "_inwin_ok": bool(inwin.all() or n >= int(CONTACT_WINDOW_S / dt)),
                })

    # ── aggregate + rank ───────────────────────────────────────────────────
    def agg(policy_name):
        per_cell = []
        for c, (k, d, label) in enumerate(belt_cells):
            sub = [r for r in rows if r["policy"] == policy_name and r["cell"] == label]
            if not sub:
                continue
            pr = np.array([r["min_rate_deg_s"] for r in sub], dtype=float)
            per_cell.append({
                "cell": label, "k": k, "d": d, "n": len(sub),
                "pitch_0.2": float(np.nanmean([r["pitch_0.2_deg"] for r in sub])),
                "pitch_0.4": float(np.nanmean([r["pitch_0.4_deg"] for r in sub])),
                "min_rate": float(np.nanmean(pr)),
                "min_rate_p10": float(np.nanpercentile(pr, 10)),
                "tilt25_frac": float(np.mean([r["tilt25_inwin"] for r in sub])),
                "surv": float(np.mean([min(r["survived_s"], args_cli.episode_s) for r in sub])),
            })
        return per_cell

    lines = []
    lines.append(f"sweep_plant_elasticity  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"contact: {contact}")
    lines.append(f"gait phase {args_cli.gait_phase_init}, trim {args_cli.trim}, "
                 f"{T} steps @ {dt:.4f} s, {args_cli.seeds} seeds/cell")
    lines.append("")
    verdicts = {}
    for name in policies:
        per_cell = agg(name)
        if not per_cell:
            continue
        rigid = next((x for x in per_cell if x["k"] == 0.0), None)
        base_rate = rigid["min_rate"] if rigid else float("nan")
        lines.append(f"── policy {name}")
        lines.append(f"{'cell':>12} {'n':>3} {'pitch0.2':>9} {'pitch0.4':>9} "
                     f"{'min_rate':>9} {'p10':>8} {'rate/base':>9} {'tilt25@win':>10} {'surv_s':>7}")
        for x in per_cell:
            ratio = x["min_rate"] / base_rate if rigid and base_rate not in (0.0, float("nan")) else float("nan")
            lines.append(f"{x['cell']:>12} {x['n']:>3} {x['pitch_0.2']:>9.2f} {x['pitch_0.4']:>9.2f} "
                         f"{x['min_rate']:>9.1f} {x['min_rate_p10']:>8.1f} {ratio:>9.2f} "
                         f"{x['tilt25_frac']:>10.2f} {x['surv']:>7.2f}")
        # how much the belt arm moves the metric relative to seed noise
        if rigid and len(per_cell) > 1:
            spread = np.std([x["min_rate"] for x in per_cell if x["k"] > 0.0]) if len(per_cell) > 1 else 0.0
            verdicts[name] = {
                "base_min_rate": base_rate,
                "max_ratio": max((x["min_rate"] / base_rate) if base_rate else float("nan")
                                 for x in per_cell if x["k"] > 0.0) if len(per_cell) > 1 else float("nan"),
                "across_cell_std": float(spread),
            }
        lines.append("")

    lines.append("── P1-P4 verdicts (pre-registered, apply verbatim)")
    if "R1r" in verdicts and verdicts["R1r"].get("max_ratio") == verdicts["R1r"].get("max_ratio"):
        r = verdicts["R1r"]["max_ratio"]
        lines.append(f"P1 (belt mechanism): max min-rate ratio over soft cells = {r:.2f}. "
                     f"{'SUPPORTED' if abs(r) >= 1.5 else ('WEAK' if abs(r) >= 1.2 else 'CONTRADICTED')} "
                     f"by the >=1.5x / >=1.2x bars.")
    else:
        lines.append("P1 (belt mechanism): no R1r rigid baseline in this run — cannot read; "
                     "re-run with the rigid cell included (belt spec must contain '0:0').")
    lines.append("P2 (contact mechanism): compare the pitch traces in rollout.npz — the rigid and "
                 "compliant cells must AGREE until the first foot touchdown and separate after it. "
                 "This script cannot time contacts without a sensor; read it off the traces and the video.")
    lines.append("P3: if no cell moves min pitch rate by >= 1.2x while S4 still fails in sim, say "
                 "plainly that both mechanisms are out for this failure.")
    lines.append("P4: rank on pitch@0.2/0.4/0.6 s, min rate, tilt-25 crossing, survival. Do NOT use "
                 "the 45-deg fall time (tether-suspect).")
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(out, "report.txt"), "w") as fh:
        fh.write(report + "\n")

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
