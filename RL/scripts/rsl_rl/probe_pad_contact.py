"""probe_pad_contact.py — does the ball-pad model actually change the plant?

The pad is a PLANT FORCE (pad_contact.PadContact), so unlike a material edit it
can be pinned per arm and read back as a number in one process.  This probe is
the gate the material path never passed: it runs the SAME scene three times at
different pinned stiffnesses and reports a measured quantity each time.

Modes:

  --geometry_only   Validate measure_ball_geometry against the live stage.  This
                    is the ONLY untested part of the model: the ball's centre
                    offset and radius are measured rather than hardcoded, and a
                    wrong measurement puts the spring at a point that never
                    touches the floor.  Prints both feet's offsets, the radius,
                    and the collision prims it rejected, so a miss is diagnosable
                    without a second Isaac launch.

  --sag_ab          Hold the stand pose with deploy gains (a plain PD law, no
                    policy) and report the settled base height and the pad force
                    for k = 0, soft, stiff.  A live pad moves BOTH numbers; an
                    inert one leaves them identical.

  --drop_ab         Same, but drop the robot from height so the pad sees a real
                    impact.  Reports peak pad force and rebound.

    python RL/scripts/rsl_rl/probe_pad_contact.py \
        --task Template-standup_curriculum-v0 --mode sag_ab --headless

Read the header of RL/run_probe_pad_contact.ps1 for the launch recipe.
"""
from __future__ import annotations

import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Ball-pad contact probe.")
parser.add_argument("--task", type=str, default="Template-standup_curriculum-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--mode", type=str, default="geometry_only",
                    choices=["geometry_only", "sag_ab", "drop_ab", "env_sweep"])
parser.add_argument("--num_envs", type=int, default=1)
# Pinned stiffness arms, N/m. The cfg's own band is NOT used here: this probe is
# deliberately independent of it so a broken cfg cannot make the probe agree.
parser.add_argument("--stiffness", type=float, nargs="+",
                    default=[0.0, 5000.0, 20000.0, 50000.0])
parser.add_argument("--damping_ratio", type=float, default=0.7)
parser.add_argument("--effective_mass", type=float, default=1.0)
parser.add_argument("--pad_thickness", type=float, default=0.002)
parser.add_argument("--steps", type=int, default=400)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.utils.hydra import hydra_task_config

import RL.tasks  # noqa: F401
from RL.tasks.direct.standup_curriculum.pad_contact import (
    PadContact,
    damping_for_ratio_tensor,
    measure_ball_geometry,
)


def _geometry(env):
    cfg = env.cfg
    robot_prim_path = cfg.robot_cfg.prim_path.replace("env_.*", "env_0")
    body_paths = (
        f"{robot_prim_path}/{cfg.foot_body_name_1}",
        f"{robot_prim_path}/{cfg.foot_body_name_2}",
    )
    stage = env.sim.stage
    print("=" * 78)
    print("A. BALL GEOMETRY (measured from the live stage)")
    print("=" * 78)
    print(f"  robot prim path : {robot_prim_path}")
    for name, path in zip((cfg.foot_body_name_1, cfg.foot_body_name_2), body_paths):
        print(f"  {name:20s} -> {path}  valid={stage.GetPrimAtPath(path).IsValid()}")
    offsets, radius = measure_ball_geometry(
        stage, (cfg.foot_body_name_1, cfg.foot_body_name_2), body_paths
    )
    print(f"\n  measured ball radius : {radius:.6f} m  (expect 0.020 = 40 mm mesh)")
    for i, label in enumerate(("left ", "right")):
        o = offsets[i].tolist()
        print(f"  {label} centre offset  : ({o[0]:+.6f}, {o[1]:+.6f}, {o[2]:+.6f}) m in the shin frame")
    # The env's own toe offset is the NAIL origin, documented as 5.8 cm short of
    # the ball.  Print the distance so the difference is visible, not assumed.
    toe = env._toe1_local_offset.tolist() if hasattr(env, "_toe1_local_offset") else None
    if toe is not None:
        d = math.dist(toe, offsets[0].tolist())
        print(f"\n  env's _toe1_local_offset : ({toe[0]:+.6f}, {toe[1]:+.6f}, {toe[2]:+.6f})")
        print(f"  distance nail-origin -> ball centre : {d*1000:.2f} mm")
        print("  (URDF-computed 2026-09-12 at 59.06 mm for BOTH feet; the 2026-09-08")
        print("   note calls it 5.8 cm. A very different number means the ball lookup")
        print("   found the wrong mesh — check the rejected list printed above.)")
    return offsets, radius


def _env_sweep(env):
    """Pin the REAL env's pad, one stiffness per reset, and read it back.

    The other modes build their own PadContact, which proves the model but not
    the wiring.  This one goes through RlEnv._build_pad_contact, the per-episode
    draw in _sample_dr, and the substep force in _apply_action — so a mistake in
    any of those shows up as a number that does not move.
    """
    print("=" * 78)
    print("C. ENV WIRING — the env's own pad, pinned via the DR override")
    print("=" * 78)
    if env._pad is None:
        print("  env._pad is None: the cfg never built a pad. Launch with")
        print("  env.pad_contact_enable=true env.pad_stiffness_min=<k> (or pass")
        print("  the band) and re-run this mode.")
        return
    pad = env._pad
    print(f"  built pad: reference radius {pad.reference_radius:.5f} m, "
          f"ball radius {pad.ball_radius:.5f} m, max deflection "
          f"{pad.max_compression*1e3:.1f} mm")
    print(f"  damping ceiling {pad.damping_cap:.1f} N s/m at m_eff="
          f"{pad.effective_mass:g} kg, dt={pad.physics_dt:.5f} s")

    print(f"\n  {'k pinned (N/m)':>15s} {'pad.stiffness[0,0]':>18s} "
          f"{'pad.damping[0,0]':>17s} {'clamped':>8s}")
    print("  " + "-" * 62)
    for k_value in args_cli.stiffness:
        k_t = torch.full((env.num_envs, 2), float(k_value), device=env.device)
        c_t = damping_for_ratio_tensor(k_t, args_cli.damping_ratio,
                                       env.cfg.pad_effective_mass).to(env.device)
        env._pad.set_parameters(torch.arange(env.num_envs, device=env.device), k_t, c_t)
        print(f"  {k_value:15.1f} {float(pad.stiffness[0,0]):18.1f} "
              f"{float(pad.damping[0,0]):17.1f} {str(bool(pad.clamped[0,0])):>8s}")

    print("\n  These are the values the substep force will use. If pad.stiffness")
    print("  does not track the left column, the DR path is not reaching the")
    print("  model — that is the failure this mode exists to catch.")


def _build_pad(env, offsets, radius, stiffness, damping_ratio, effective_mass,
               pad_thickness):
    pad = PadContact(
        num_envs=env.num_envs, device=env.device,
        ball_local_offset=offsets, ball_radius=radius,
        pad_thickness=pad_thickness, effective_mass=effective_mass,
        physics_dt=env.physics_dt, stability_factor=0.5,
        enabled=stiffness > 0.0,
    )
    n = env.num_envs
    k = torch.full((n, 2), float(stiffness), device=env.device)
    c = damping_for_ratio_tensor(k, damping_ratio, effective_mass).to(env.device)
    pad.set_parameters(torch.arange(n, device=env.device), k, c)
    return pad


def _foot_kinematics(env, pad):
    """Ball centres, their velocities, and the foot origins — same maths as
    RlEnv._apply_pad_contact, deliberately duplicated here so a bug in the env
    wiring cannot hide behind this probe reusing the env's own helper."""
    f1, f2 = env._foot1_idx, env._foot2_idx
    foot_pos = torch.stack(
        [env.robot.data.body_pos_w[:, f1, :], env.robot.data.body_pos_w[:, f2, :]], dim=1
    )
    foot_quat = torch.stack(
        [env.robot.data.body_quat_w[:, f1, :], env.robot.data.body_quat_w[:, f2, :]], dim=1
    )
    offset = pad.ball_local_offset.unsqueeze(0).expand(env.num_envs, -1, -1)
    centre = foot_pos + _quat_apply(foot_quat.reshape(-1, 4), offset.reshape(-1, 3)).reshape(
        env.num_envs, 2, 3
    )
    omega = env.robot.data.body_ang_vel_w[:, [f1, f2], :]
    vel = (env.robot.data.body_lin_vel_w[:, [f1, f2], :]
           + torch.cross(omega, centre - foot_pos, dim=-1))
    return centre, vel, foot_pos


def _quat_apply(quat, vec):
    return quat_apply(quat, vec)


def _hold(env, pad, z0, steps, stand):
    """PD-hold the stand pose with the pad APPLIED to the sim.

    The force is written into the same external-force path the env uses and
    recomputed every physics step, because the pad is exactly the kind of stiff
    spring whose effect disappears if it is evaluated at the control rate only.
    A probe that merely computed the force without applying it would report a
    beautiful table and prove nothing about the plant.
    """
    dt = env.physics_dt
    kp = torch.tensor([[6.0, 20.0, 30.0, 6.0, 20.0, 30.0]], device=env.device)
    kd = torch.tensor([[0.3, 1.4, 1.5, 0.3, 1.4, 1.5]], device=env.device)
    kp = kp.expand(env.num_envs, -1)
    kd = kd.expand(env.num_envs, -1)
    stand = stand.expand(env.num_envs, -1)
    body_ids = [env._foot1_idx, env._foot2_idx]

    root = env.robot.data.root_state_w.clone()
    root[:, 2] = z0
    root[:, 7:13] = 0.0
    env.robot.write_root_state_to_sim(root)
    env.robot.set_joint_position_target(stand, joint_ids=env._joint_ids)

    zero_f = torch.zeros(env.num_envs, env.robot.num_bodies, 3, device=env.device)
    trace, forces = [], []
    # Mirror DirectRLEnv.step: decimation physics steps per control step, with
    # the PD effort and the pad force refreshed before each one.
    for _ in range(steps):
        for _ in range(env.cfg.decimation):
            q = env.robot.data.joint_pos
            qd = env.robot.data.joint_vel
            env.robot.set_joint_effort_target(kp * (stand - q) - kd * qd)
            centre, vel, origin = _foot_kinematics(env, pad)
            f, t = pad.compute(centre, vel, origin)
            if f is None:
                total_f, total_t = zero_f, zero_f
            else:
                total_f = zero_f.clone()
                total_t = zero_f.clone()
                total_f[:, body_ids, :] = f
                total_t[:, body_ids, :] = t
                forces.append(float(f[:, :, 2].max().item()))
            env.robot.set_external_force_and_torque(total_f, total_t)
            env.robot.write_data_to_sim()
            env.sim.step()
            env.robot.update(dt)
        trace.append(env.robot.data.root_pos_w[0, 2].item())
    return trace, forces


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: DirectRLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None).unwrapped

    offsets, radius = _geometry(env)

    if args_cli.mode == "geometry_only":
        print("\n[probe] geometry only — done")
        return

    if args_cli.mode == "env_sweep":
        _env_sweep(env)
        return

    stand = torch.tensor([[0.0, 0.05, 0.0, 0.0, -0.05, 0.0]], device=env.device)

    print("\n" + "=" * 78)
    print(f"B. {'SAG' if args_cli.mode == 'sag_ab' else 'DROP'} A/B — same scene, "
          f"pinned stiffness")
    print("=" * 78)
    print(f"  {'k (N/m)':>10s} {'c (N s/m)':>11s} {'settled z (m)':>14s} "
          f"{'min z (m)':>11s} {'peak |Fpad| (N)':>16s}")
    print("  " + "-" * 68)
    results = []
    for k_value in args_cli.stiffness:
        pad = _build_pad(env, offsets, radius, k_value, args_cli.damping_ratio,
                         args_cli.effective_mass, args_cli.pad_thickness)
        if args_cli.mode == "drop_ab":
            env.reset()
            trace, forces = _hold(env, pad, 0.55, args_cli.steps, stand)
        else:
            env.reset()
            trace, forces = _hold(env, pad, env.cfg.spawn_height if hasattr(env.cfg, "spawn_height") else 0.434,
                                  args_cli.steps, stand)
        settled = sum(trace[-20:]) / 20.0
        c_applied = float(pad.damping[0, 0])
        peak_f = max(forces) if forces else 0.0
        print(f"  {k_value:10.1f} {c_applied:11.1f} {settled:14.4f} "
              f"{min(trace):11.4f} {peak_f:16.2f}")
        results.append((k_value, settled, min(trace), peak_f))

    print("\n  READ THIS:")
    if len({round(r[1], 6) for r in results}) == 1:
        print("  => every arm settled identically. The pad is INERT here: the balls")
        print("     are not loaded in this pose (check the toe height against the")
        print("     env's contact line) or the force never reached the sim.")
    else:
        print("  => the settled height and the peak pad force both move with k.")
        print("     The pad is live. A softer pad must settle LOWER and show a")
        print("     smaller peak force; if that ordering is inverted the force sign")
        print("     or the contact point is wrong.")
    if pad.clamped.any():
        print(f"  NOTE: damping was clipped to the stability ceiling "
              f"{pad.damping_cap:.1f} N s/m for this draw; the applied c is the")
        print("        clipped value, not what the ratio asked for.")
    print(f"\n[probe] mode={args_cli.mode} done")


if __name__ == "__main__":
    main()
    simulation_app.close()
