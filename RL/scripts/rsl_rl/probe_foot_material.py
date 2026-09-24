"""probe_foot_material.py — is the ball-foot contact material really on the stage?

Boots Isaac Lab, spawns ONE robot the way the task cfg does (via
FootContactUsdFileCfg), and reports, from the live stage:

  A. every prim carrying UsdPhysics.CollisionAPI under the robot root, with
     whether it is an instance proxy and which physics material is bound to it
  B. the attributes actually authored on each foot_contact_material_* prim,
     read back through both UsdPhysics.MaterialAPI and PhysxSchema.PhysxMaterialAPI
  C. the ground plane's own material, for the combine-mode cross-check
  D. a rigid-body drop test: the robot's foot is teleported to a set height
     above the ground with zero velocity and the base fixed, and the peak
     rebound height after impact is recorded — the one number that says
     whether restitution changed the contact at all.

Usage (needs the Isaac env vars, so run it through the wrapper .ps1):

    powershell -File RL/run_probe_foot_material.ps1                 # rigid, k=100.1 d=10.1z r=1.0
    powershell -File RL/run_probe_foot_material.ps1 -Stiffness 0 -Rest 1.0
"""
from __future__ import annotations

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Foot contact material stage probe.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--stiffness", type=float, default=100.1, help="N/m, 0 = rigid")
parser.add_argument("--damping_ratio", type=float, default=10.1,
                    help="ratio form; NaN string 'off' disables")
parser.add_argument("--damping", type=float, default=0.0, help="explicit N s/m (wins if >0)")
parser.add_argument("--effective_mass", type=float, default=1.0)
parser.add_argument("--restitution", type=float, default=1.0)
parser.add_argument("--drop_height", type=float, default=0.05,
                    help="metres the ball foot is lifted above its resting contact before release")
parser.add_argument("--steps", type=int, default=400, help="drop-test length (x 2 ms)")
parser.add_argument("--trace_every", type=int, default=40,
                    help="print every Nth step of the contact trace during the run")
parser.add_argument("--no_drop", action="store_true", help="stage inspection only, no physics probe")
parser.add_argument("--reset_ab", action="store_true",
                    help="also test whether a material edit takes effect across sim.reset()")
parser.add_argument("--writer_check", action="store_true",
                    help="exercise play_sim2real.apply_foot_material's attribute names")
parser.add_argument("--contact_sweep", action="store_true",
                    help="settle the limbed robot at each foot setting, one process")
parser.add_argument("--scene_style", action="store_true",
                    help="rebuild the stage in the play env's InteractiveScene layout")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── everything else after the app is up ──────────────────────────────────────
import torch  # noqa: E402
from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import RL.tasks  # noqa: F401,E402
from RL.tasks.direct.standup_curriculum.foot_contact import (  # noqa: E402
    FootContactUsdFileCfg,
)
from RL.tasks.direct.standup_curriculum.rl_env_cfg_p6 import (  # noqa: E402
    ROBOT_USD_PATH,
    RlEnvCfg,
)

ROBOT_PATH = "/World/Robot"


def _material_report(stage, prim):
    """Every material attribute authored on a prim, or 'NO MATERIAL'."""
    # ComputeBoundMaterial returns a UsdShade.Material (a schema object), not a
    # Usd.Prim — get the prim out of it before asking anything of it.
    if isinstance(prim, UsdShade.Material):
        prim = prim.GetPrim()
    if not prim or not prim.IsValid():
        return "  <prim invalid>"
    lines = [f"    authored attributes on the prim ({len(prim.GetAuthoredProperties())}):"]
    for prop in prim.GetAuthoredProperties():
        lines.append(f"      {prop.GetName():38s}: {prop.Get()}")
    lines.append("    ---")
    lines.append(f"    xformOp / type           : {prim.GetTypeName()}")
    usd_api = UsdPhysics.MaterialAPI(prim)
    px_api = PhysxSchema.PhysxMaterialAPI(prim)
    lines.append(f"    is UsdShade.Material     : {prim.IsA(UsdShade.Material)}")
    lines.append(f"    UsdPhysics.MaterialAPI   : {bool(usd_api)}")
    lines.append(f"    PhysxMaterialAPI         : {bool(px_api)}")
    for api, names in (
        (usd_api, ("staticFriction", "dynamicFriction", "restitution")),
        (px_api, ("compliantContactStiffness", "compliantContactDamping",
                  "compliantContactAccelerationSpring", "frictionCombineMode",
                  "restitutionCombineMode", "dampingCombineMode")),
    ):
        if not api:
            continue
        for name in names:
            attr = api.GetPrim().GetAttribute(name)
            if attr and attr.HasAuthoredValue():
                lines.append(f"    {name:34s}: {attr.Get()}")
            else:
                lines.append(f"    {name:34s}: <unauthored>")
    return "\n".join(lines)


def _dump_stage(stage, root: str):
    print("\n" + "=" * 78)
    print("A. COLLISION PRIMS AND THEIR BOUND MATERIALS")
    print("=" * 78)
    prim = stage.GetPrimAtPath(root)
    if not prim.IsValid():
        print(f"  !! {root} is not valid")
        return
    found = 0
    for desc in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not desc.HasAPI(UsdPhysics.CollisionAPI):
            continue
        found += 1
        bound = UsdShade.MaterialBindingAPI(desc).ComputeBoundMaterial(
            materialPurpose="physics")[0]
        print(f"  {desc.GetPath()}")
        print(f"      instance proxy : {desc.IsInstanceProxy()}")
        print(f"      bound material : {bound.GetPath() if bound else '<none>'}")
        if found > 60:
            print("  ... more than 60 collision prims, stopping")
            break
    print(f"  total collision prims under {root}: {found}")


def _dump_materials(stage, root: str):
    print("\n" + "=" * 78)
    print("B. foot_contact_material_* PRIMS AND THEIR AUTHORED ATTRIBUTES")
    print("=" * 78)
    n = 0
    for prim in stage.Traverse():
        if prim.GetName().startswith("foot_contact_material_"):
            n += 1
            print(f"  {prim.GetPath()}")
            print(_material_report(stage, prim))
            # Can the attribute be written at all from here? If Set() works,
            # the schema is present and the failure is upstream in the cfg.
            api = PhysxSchema.PhysxMaterialAPI(prim)
            attr = api.GetPrim().GetAttribute("compliantContactStiffness")
            print(f"    probe Write: attribute exists = {bool(attr)}", end="")
            if attr:
                try:
                    attr.Set(1234.5)
                    print(f", Set() ok, reads back {attr.Get()}")
                except Exception as exc:  # noqa: BLE001
                    print(f", Set() FAILED: {exc}")
    if n == 0:
        print("  NONE CREATED — the spawner never reached its material branch")
    return n


def _dump_ground(stage):
    print("\n" + "=" * 78)
    print("C. GROUND PLANE MATERIAL")
    print("=" * 78)
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if "/ground" not in str(prim.GetPath()):
            continue
        bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(
            materialPurpose="physics")[0]
        print(f"  {prim.GetPath()}  bound: {bound.GetPath() if bound else '<none>'}")
        if bound:
            print(_material_report(stage, bound))


def _settle_height(sim, robot, device, steps: int = 300) -> float:
    """Base height after letting the unactuated robot fall and rest."""
    dt = sim.get_physics_dt()
    sim.reset()
    for _ in range(steps):
        sim.step()
        robot.update(dt)
    return robot.data.root_pos_w[0, 2].item()


def _free_fall(sim, robot, device, args):
    """Drop the free robot and print the base-height trace around impact.

    The one test that separates "the material is live" from "the material is a
    no-op": nothing about the fall depends on the policy, the joints are limp
    on purpose (so the contact carries the whole robot), and the trace shows
    the impact and whatever recoil follows. Drop the same robot in a rigid
    process and a restitution-1.0 process and the traces differ, or they do
    not — no interpretation needed.
    """
    dt = sim.get_physics_dt()
    print("\n" + "=" * 78)
    print("D. FREE-FALL CONTACT TRACE (no policy, limp joints)")
    print("=" * 78)
    sim.reset()
    root_state = robot.data.root_state_w.clone()
    root_state[:, 2] = 0.90
    root_state[:, 7:13] = 0.0
    robot.write_root_state_to_sim(root_state)

    trace = []
    for i in range(args.steps):
        sim.step()
        robot.update(dt)
        trace.append((i * dt, robot.data.root_pos_w[0, 2].item(),
                      robot.data.root_lin_vel_w[0, 2].item()))
        if args.trace_every and i % args.trace_every == 0:
            print(f"      t={i * dt:6.3f}  z={trace[-1][1]:7.4f}  vz={trace[-1][2]:+7.3f}")
    z_min = min(t[1] for t in trace)
    z_final = trace[-1][1]
    i_min = min(range(len(trace)), key=lambda i: trace[i][1])
    t_impact = trace[i_min][0]
    print(f"  released at z=0.900 m, {len(trace)} steps of {dt*1e3:.0f} ms")
    print(f"  lowest base height  : {z_min:.4f} m at t={t_impact:.3f} s")
    print(f"  final base height   : {z_final:.4f} m")
    after = [t for t in trace if t[0] > t_impact]
    if after:
        best = max(after, key=lambda t: t[1])
        print(f"  highest after impact: {best[1]:.4f} m at t={best[0]:.3f} s "
              f"(rise {best[1] - z_min:+.4f} m after bottom)")
        # Restitution equivalent of the rebound, if any: v_up = e * v_down.
        v_at_bottom = trace[i_min][2]
        print(f"  base vz at bottom   : {v_at_bottom:+.3f} m/s")
    step = args.trace_every or 5
    print(f"  trace (t, z, vz), every {step}th step:")
    for t, z, vz in trace[::step]:
        print(f"      t={t:6.3f}  z={z:7.4f}  vz={vz:+7.3f}")


@hydra_task_config("Template-standup_curriculum-v0", "rsl_rl_cfg_entry_point")
def main(env_cfg: RlEnvCfg, agent_cfg):
    device = args_cli.device or "cuda:0"

    # ── build the exact spawn config the env would build ────────────────────
    spawn = env_cfg.robot_cfg.spawn
    spawn.compliant_contact_stiffness = float(args_cli.stiffness)
    spawn.compliant_contact_damping = (
        float(args_cli.damping) if args_cli.damping > 0.0
        else (0.0 if not math.isfinite(args_cli.damping_ratio) or args_cli.damping_ratio == 0.0
              else 2.0 * args_cli.damping_ratio
              * math.sqrt(args_cli.stiffness * args_cli.effective_mass))
    )
    spawn.restitution = float(args_cli.restitution)
    print("\n" + "=" * 78)
    print("PROBE CONFIG")
    print("=" * 78)
    print(f"  is FootContactUsdFileCfg : {type(spawn).__name__}")
    print(f"  usd_path                 : {spawn.usd_path}")
    print(f"  stiffness                : {spawn.compliant_contact_stiffness}")
    print(f"  damping                  : {spawn.compliant_contact_damping}")
    print(f"  restitution              : {spawn.restitution}")
    print(f"  material prim paths      : {list(spawn.physics_material_prim_path)}")
    print(f"  spawn.func               : {spawn.func}")

    # The material prims are built by RigidBodyMaterialCfg.func, which writes
    # cfg.to_dict() through safe_set_attribute_on_usd_schema, and that SILENTLY
    # SKIPS None.  So what the cfg serializes to is the whole question.
    from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg

    mat_cfg = RigidBodyMaterialCfg(
        static_friction=spawn.static_friction,
        dynamic_friction=spawn.dynamic_friction,
        restitution=spawn.restitution,
        friction_combine_mode="multiply",
        restitution_combine_mode=spawn.restitution_combine_mode,
        compliant_contact_stiffness=spawn.compliant_contact_stiffness,
        compliant_contact_damping=spawn.compliant_contact_damping,
    )
    d = mat_cfg.to_dict()
    print(f"  material cfg class       : {type(mat_cfg).__name__}")
    print("  material cfg.to_dict()   :")
    for k, v in d.items():
        print(f"      {k:34s}: {v!r}")
    nulls = [k for k, v in d.items() if v is None]
    print(f"  -> fields serializing to None (SKIPPED by the material writer): {nulls}")

    if args_cli.scene_style:
        # The scene IS the subject; a second stage would collide with it.
        _scene_style(sim_utils, device, env_cfg, args_cli)
        return

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=device))
    sim.set_camera_view([2.0, 0.0, 1.0], [0.0, 0.0, 0.3])
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    robot = Articulation(env_cfg.robot_cfg.replace(prim_path=ROBOT_PATH))
    sim.reset()   # play() + reset(): this is what creates the prims on the stage

    stage = sim.stage
    _dump_stage(stage, ROBOT_PATH)
    n = _dump_materials(stage, ROBOT_PATH)
    _dump_ground(stage)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  foot material prims created : {n}")
    if n == 0:
        print("  => the foot parameters ARE DEAD: no material ever reached the stage.")

    if not args_cli.no_drop:
        _free_fall(sim, robot, device, args_cli)

    if args_cli.reset_ab:
        _reset_ab(sim, robot, device, args_cli)

    if args_cli.writer_check:
        _writer_check(sim, robot, device, args_cli)

    if args_cli.contact_sweep:
        _contact_sweep(sim, robot, device, args_cli)

    if args_cli.scene_style:
        _scene_style(sim, device, env_cfg, args_cli)


def _writer_check(sim, robot, device, args):
    """Exercise the play loop's writer: its three attribute names, verbatim.

    play_sim2real.apply_foot_material() sets exactly
    physxMaterial:compliantContactStiffness / :compliantContactDamping /
    physics:restitution and raises if a name is missing. This runs the same
    write here and then settles, so "the fix works" is a number, not a reading
    of the code.
    """
    print("\n" + "=" * 78)
    print("F. PLAY-LOOP WRITER CHECK (apply_foot_material's attribute names)")
    print("=" * 78)
    dt = sim.get_physics_dt()
    names = ("physxMaterial:compliantContactStiffness",
             "physxMaterial:compliantContactDamping",
             "physics:restitution")
    prims = [p for p in sim.stage.Traverse()
             if p.GetName().startswith("foot_contact_material_")]
    print(f"  material prims found : {[str(p.GetPath()) for p in prims]}")
    if not prims:
        print("  => NO material prim: the writer has nothing to write, and the")
        print("     knobs are inert for a different reason (spawner never ran its")
        print("     material branch).")
        return

    def settle(label):
        sim.reset()
        root = robot.data.root_state_w.clone()
        root[:, 2] = 0.90
        root[:, 7:13] = 0.0
        robot.write_root_state_to_sim(root)
        lows = []
        for _ in range(220):
            sim.step()
            robot.update(dt)
            lows.append(robot.data.root_pos_w[0, 2].item())
        lo = min(lows)
        print(f"  {label:34s}: lowest base height {lo:.4f} m")
        return lo

    # Whatever this process spawned with.
    settle("as spawned")

    for name in names:
        missing = [p for p in prims if not p.GetAttribute(name)]
        print(f"  {name:38s}: {'OK on all prims' if not missing else 'MISSING'}")

    print("  writing rigid (k=20000, d=0, restitution=0) through those names")
    for p in prims:
        p.GetAttribute(names[0]).Set(20000.0)
        p.GetAttribute(names[1]).Set(0.0)
        p.GetAttribute(names[2]).Set(0.0)
    rigid = settle("after writer call")

    print("  writing soft+dead (k=100.1, d=202.1, restitution=0) through those names")
    for p in prims:
        p.GetAttribute(names[0]).Set(100.1)
        p.GetAttribute(names[1]).Set(202.1)
        p.GetAttribute(names[2]).Set(0.0)
    soft = settle("after second writer call")
    print(f"  stiffness moved the settle height by {soft - rigid:+.4f} m")
    print("  (a live writer moves it; a dead attribute name moves nothing)")


def _write_foot_material(sim, stiffness: float, damping: float, restitution: float) -> int:
    """The same three attribute writes play_sim2real.apply_foot_material makes."""
    names = ("physxMaterial:compliantContactStiffness",
             "physxMaterial:compliantContactDamping",
             "physics:restitution")
    n = 0
    for prim in sim.stage.Traverse():
        if not prim.GetName().startswith("foot_contact_material_"):
            continue
        for name, value in zip(names, (stiffness, damping, restitution)):
            attr = prim.GetAttribute(name)
            if not attr:
                raise RuntimeError(f"{prim.GetPath()} has no {name}")
            attr.Set(value)
        n += 1
    return n


def _scene_style(sim_utils, device, env_cfg, args):
    """Reproduce the PLAY scene's structure: /World/envs/env_0, via the scene.

    The probe's first version spawned a bare Articulation at /World/Robot and
    the foot material clearly changed the contact there. The play env builds an
    InteractiveScene, and the same material does NOT change its rollout. This
    builds the scene-shaped stage and puts the ball on the ground with the
    robot's own spawn pose, so "does the material reach the collider here?"
    stops being a guess.
    """
    from pxr import Usd, UsdPhysics, UsdShade
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.assets import ArticulationCfg

    print("\n" + "=" * 78)
    print("I. SCENE-SHAPED STAGE (/World/envs/env_0) — the play layout")
    print("=" * 78)

    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    # Reuse the task's own robot cfg so this is the same spawn path.
    scene_cfg.robot = env_cfg.robot_cfg.replace(prim_path="/World/envs/env_.*/Robot")
    sim2 = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=device))
    sim2.set_camera_view([2.0, 0.0, 1.0], [0.0, 0.0, 0.3])
    from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    scene = InteractiveScene(scene_cfg)
    sim2.reset()

    stage = sim2.stage
    print("  collision prims named *nail*, with their bound material:")
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if "nail" in p and prim.HasAPI(UsdPhysics.CollisionAPI):
            bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(
                materialPurpose="physics")[0]
            print(f"    {p}")
            print(f"        instance proxy {prim.IsInstanceProxy()}  bound "
                  f"{bound.GetPath() if bound else '<none>'}")
    mats = [p for p in stage.Traverse()
            if p.GetName().startswith("foot_contact_material_")]
    print(f"  foot material prims: {[str(p.GetPath()) for p in mats]}")
    for m in mats:
        for name in ("physxMaterial:compliantContactStiffness",
                     "physxMaterial:compliantContactDamping",
                     "physics:restitution",
                     "physics:staticFriction",
                     "physics:dynamicFriction"):
            a = m.GetAttribute(name)
            print(f"      {name:38s} = {a.Get() if a else '<MISSING>'}")

    robot = scene["robot"]
    dt = sim2.get_physics_dt()

    # HOLD THE STAND POSE. Without this the robot crumples and rests on its
    # torso in every arm — the feet carry nothing, so every stiffness gives the
    # same number and the sweep reads as "the material is dead" when it is the
    # TEST that is dead. Deploy gains, one plain PD law, no policy involved.
    import torch
    kp = torch.tensor([[6.0, 20.0, 30.0, 6.0, 20.0, 30.0]], device=device)
    kd = torch.tensor([[0.3, 1.4, 1.5, 0.3, 1.4, 1.5]], device=device)
    stand = torch.tensor([[0.0, 0.05, 0.0, 0.0, -0.05, 0.0]], device=device)

    def hold(z0: float, steps: int):
        st = robot.data.root_state_w.clone()
        st[:, 2] = z0
        st[:, 7:13] = 0.0
        robot.write_root_state_to_sim(st)
        robot.set_joint_position_target(stand)
        trace = []
        for _ in range(steps):
            q = robot.data.joint_pos
            qd = robot.data.joint_vel
            robot.set_joint_effort_target(kp * (stand - q) - kd * qd)
            robot.write_data_to_sim()
            sim2.step()
            robot.update(dt)
            trace.append(robot.data.root_pos_w[0, 2].item())
        return trace

    print("\n  PD-held drop (deploy gains), 60 steps from 0.55 m:")
    tr = hold(0.55, 60)
    print(f"    base z: start {tr[0]:.4f} -> min {min(tr):.4f} -> end {tr[-1]:.4f}")
    print("    contact = min z dips BELOW the free-fall value then recovers")

    # The decisive comparison for "does a FOOT SPRING carry this robot":
    # rest it and see how far it sinks. Rigid contact = base near the
    # straight-leg height; a soft foot = base drops by m*g/k.
    print("\n  rest from the spawn pose, 400 steps, and report the hold:")
    tr2 = hold(0.434, 400)
    print(f"    base z: start {tr2[0]:.4f} -> settled {tr2[-1]:.4f} "
          f"(sank {tr2[0] - tr2[-1]:+.4f} m)")
    print(f"    expected sag from a foot spring of k={args.stiffness:g} N/m "
          f"carrying 5.57 kg: {5.57 * 9.81 / max(args.stiffness, 1e-9):.4f} m")

    # ── FRICTION ────────────────────────────────────────────────────────────
    # First attempt at this shoved a free-standing robot and measured how far it
    # slid, which measured the fall rather than the traction: at friction 1.0 it
    # had already travelled 0.89 m. Second attempt pinned the base kinematic,
    # which made PhysX's own root-transform query throw. This version uses the
    # standard friction measurement instead — put the body on a SLOPE and read
    # how fast it slides. Gravity supplies a bounded shear load, friction
    # resists it, and the terminal speed is a direct read on the ball's
    # traction with no external-force plumbing at all.
    from pxr import UsdGeom as _UsdGeom, Gf as _Gf

    def tilt_ground(deg):
        ground = stage.GetPrimAtPath("/World/ground")
        # UsdGeom.Xformable, not UsdPhysics.XformAPI — the latter does not
        # exist and raises AttributeError.
        xf = _UsdGeom.Xformable(ground)
        m = _Gf.Matrix4d()
        m.SetRotate(_Gf.Rotation(_Gf.Vec3d(1.0, 0.0, 0.0), deg))
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(m)

    def slide(label, deg=25.0, steps=250):
        tilt_ground(deg)
        sim2.reset()
        st = robot.data.root_state_w.clone()
        st[:, 2] = 0.50
        st[:, 7:13] = 0.0
        robot.write_root_state_to_sim(st)
        robot.set_joint_position_target(stand)
        y0 = None
        for _ in range(steps):
            q = robot.data.joint_pos
            qd = robot.data.joint_vel
            robot.set_joint_effort_target(kp * (stand - q) - kd * qd)
            robot.write_data_to_sim()
            sim2.step()
            robot.update(dt)
            if y0 is None:
                y0 = robot.data.root_pos_w[0, 2].item()
        # Down-slope is world -y once the floor is rotated about +x.
        slid = robot.data.root_pos_w[0, 1].item()
        print(f"  {label:32s}: slid {slid:+.5f} m down a {deg:g} deg slope")
        return slid

    def write_friction(sf, df):
        n = 0
        for m in mats:
            for name, value in (("physics:staticFriction", sf),
                                ("physics:dynamicFriction", df)):
                a = m.GetAttribute(name)
                if not a:
                    raise RuntimeError(f"{m.GetPath()} has no {name}")
                a.Set(value)
            n += 1
        return n

    print("\n  friction A/B, robot on a slope (free slide):")
    print("  " + "-" * 66)
    s_hi = slide("spawned friction (expect 1.0)")
    n = write_friction(0.0, 0.0)
    print(f"  wrote friction 0.0 to {n} prim(s); resetting")
    s_lo = slide("friction 0.0 after reset")
    tilt_ground(0.0)
    print(f"\n  the slick foot slid {s_lo - s_hi:+.5f} m further than the grip "
          f"one. A dead\n  friction attribute leaves this at ~0.00 m; a live one "
          f"opens a gap.")


def _contact_sweep(sim, robot, device, args):
    """Settle the limp robot at each foot setting, in ONE process.

    The writer is the one the play loop uses, and the reset between cells is
    what makes PhysX re-read it — the same mechanism the PgUp/Home hotkeys now
    ride on. Printed as a table so the effect is a number, not an impression.
    """
    dt = sim.get_physics_dt()
    print("\n" + "=" * 78)
    print("G. FOOT-PARAMETER SWEEP (one process, written through the play writer)")
    print("=" * 78)

    def cell(k, d, r):
        n = _write_foot_material(sim, k, d, r)
        sim.reset()
        root = robot.data.root_state_w.clone()
        root[:, 2] = 0.90
        root[:, 7:13] = 0.0
        robot.write_root_state_to_sim(root)
        trace = []
        for _ in range(300):
            sim.step()
            robot.update(dt)
            trace.append((robot.data.root_pos_w[0, 2].item(),
                          robot.data.root_lin_vel_w[0, 2].item()))
        lows = [t[0] for t in trace]
        lo = min(lows)
        i_lo = lows.index(lo)
        peak_vz_after = max((t[1] for t in trace[i_lo:]), default=0.0)
        return n, lo, peak_vz_after

    arms = []
    for k in (0.0, 100.1, 1000.0, 20000.0):
        arms.append((f"k={k:g} rigid-ish", k, 0.0, 0.0))
    for r in (0.0, 0.5, 1.0):
        arms.append((f"restitution={r:g}", args.stiffness, args.damping, r))

    print(f"  {'arm':26s} {'prims':>5s} {'lowest z (m)':>13s} {'rebound vz (m/s)':>17s}")
    print("  " + "-" * 66)
    for label, k, d, r in arms:
        n, lo, v = cell(k, d, r)
        print(f"  {label:26s} {n:5d} {lo:13.4f} {v:17.3f}")
    print("\n  (the fall is policy-free and the joints are limp, so the contact")
    print("   carries the whole robot: this is the foot parameter's own effect)")


def _reset_ab(sim, robot, device, args):
    """Does PhysX re-read a material edit across sim.reset()?

    This is the question the play hotkeys depend on. If the answer is yes,
    a foot-material hotkey can work on the next reset (as the belt knobs do).
    If no, the only honest thing a hotkey can do is say "restart the process".
    """
    print("\n" + "=" * 78)
    print("E. MATERIAL EDIT ACROSS sim.reset()  ← what the play hotkeys need")
    print("=" * 78)
    dt = sim.get_physics_dt()
    mat = sim.stage.GetPrimAtPath(f"{ROBOT_PATH}/foot_contact_material_node_STL_BINARY_")
    if not mat.IsValid():
        print("  no material prim; nothing to edit")
        return

    def settle(label):
        sim.reset()
        root = robot.data.root_state_w.clone()
        root[:, 2] = 0.90
        root[:, 7:13] = 0.0
        robot.write_root_state_to_sim(root)
        trace = []
        for _ in range(200):
            sim.step()
            robot.update(dt)
            trace.append(robot.data.root_pos_w[0, 2].item())
        lo = min(trace)
        hi_after = max(trace[trace.index(lo):]) if lo in trace else lo
        print(f"  {label:28s}: lowest {lo:.4f} m, highest after bottom "
              f"{hi_after:.4f} m (rise {hi_after - lo:+.4f} m)")
        return lo, hi_after

    r = mat.GetAttribute("physics:restitution")
    s = mat.GetAttribute("physxMaterial:compliantContactStiffness")
    d = mat.GetAttribute("physxMaterial:compliantContactDamping")
    print(f"  authored now: restitution={r.Get()} stiffness={s.Get()} damping={d.Get()}")
    settle("as authored")

    print("  -> editing restitution to 0.0 and stiffness to 20000")
    r.Set(0.0)
    s.Set(20000.0)
    d.Set(0.0)
    settle("after edit + reset")
    print("  (if the rise barely changes, PhysX parsed the material at stage")
    print("   attach and a mid-process edit cannot reach it)")

    print("\n[probe] done")


if __name__ == "__main__":
    main()
    simulation_app.close()
