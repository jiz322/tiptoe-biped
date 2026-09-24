"""probe_foot_drop.py — Phase 0 wiring gate: reproduce the felt-pad drop test in sim.

    hardware: 200 mm release of the foot on a yoga mat -> 75-85 mm rebound (e ~ 0.63)
    sim:      this script. Same physics dt / solver / iterations as training.

Stock PhysX only. No custom contact model, no force term, no integrator. The
only knobs touched are:
  * RigidBodyMaterialCfg on the FOOT collision prims (via the task's
    FootContactUsdFileCfg, which binds a material to part_3 of each foot)
  * RigidBodyMaterialCfg on the GROUND plane
  * PhysxCfg.bounce_threshold_velocity

Every value the script claims is READ BACK from the live stage (the material
prims' authored attributes) and, for the shape-level view, from
ArticulationView.get_material_properties() / the ground collision prim's bound
material. A cfg default is never printed as evidence.

Usage (needs the Isaac env vars -> run through the wrapper .ps1):

    powershell -File RL/run_probe_foot_drop.ps1                       # e = 0.63
    powershell -File RL/run_probe_foot_drop.ps1 -Rest 0.0
    powershell -File RL/run_probe_foot_drop.ps1 -Rest 0.3
    powershell -File RL/run_probe_foot_drop.ps1 -GroundCombine average   # show the zeroing

Exit status is 0 always; the PASS/FAIL verdict is printed and written to
RL/logs/foot_drop_<rest>.log next to the trace.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase 0 foot restitution drop test.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--restitution", type=float, default=0.63, help="foot material restitution")
parser.add_argument("--drop_height", type=float, default=0.200,
                    help="metres from the foot's lowest ball point to the ground")
parser.add_argument("--steps", type=int, default=500, help="steps of 2 ms")
parser.add_argument("--bounce_threshold", type=float, default=0.05,
                    help="PhysxCfg.bounce_threshold_velocity, m/s (training default 0.5)")
parser.add_argument("--ground_restitution", type=float, default=0.0)
parser.add_argument("--ground_combine", type=str, default="max",
                    choices=["average", "min", "multiply", "max"],
                    help="restitution_combine_mode on the GROUND material")
parser.add_argument("--foot_combine", type=str, default="max",
                    choices=["average", "min", "multiply", "max"])
parser.add_argument("--arm", type=str, default="locked",
                    choices=["locked", "robot", "sphere"],
                    help="locked = the real foot, joints pinned (a true drop test, no "
                         "actuator energy); robot = the real foot held by a PD servo "
                         "(the deployed plant, but the servo pumps energy at the "
                         "contact); sphere = an isolated ball with the foot's "
                         "measured radius/mass")
parser.add_argument("--sphere_radius", type=float, default=0.0,
                    help="sphere arm only; 0 = copy the measured ball radius")
parser.add_argument("--hold_gain", type=float, default=400.0,
                    help="joint-space PD stiffness used to hold the pose (training "
                         "servo is 20-30; a limp leg is not a drop test)")
parser.add_argument("--trace_every", type=int, default=20)
parser.add_argument("--tag", type=str, default="", help="suffix for the log file")
parser.add_argument("--task", type=str, default="Template-standup_curriculum-v0",
                    help="task whose cfg supplies the plant (need not match the GUI task)")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── everything else after the app is up ──────────────────────────────────────
import torch  # noqa: E402
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, RigidObject  # noqa: E402
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane  # noqa: E402
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import RL.tasks  # noqa: F401,E402
from RL.tasks.direct.standup_curriculum.foot_contact import (  # noqa: E402
    BALL_FOOT_COLLISION_PATHS,
)
from RL.tasks.direct.standup_curriculum.rl_env_cfg_p6 import (  # noqa: E402
    ROBOT_USD_PATH,
    RlEnvCfg,
)

ROBOT_PATH = "/World/Robot"
SPHERE_PATH = "/World/Ball"
GROUND_PATH = "/World/ground"

# The two ball-foot collision meshes (foot_contact.py). Part_3 is the ball.
FOOT_BODY_NAMES = ("shin_inner", "nail___fillet1")

REPORT_LINES: list[str] = []


def say(line: str = "") -> None:
    print(line, flush=True)
    REPORT_LINES.append(line)


# ─────────────────────────────────────────────────────────────────────────────
# Read-back helpers — the whole point of this script
# ─────────────────────────────────────────────────────────────────────────────
def _fmt(v) -> str:
    if v is None:
        return "<unauthored>"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _material_rows(prim, label: str, path: str) -> list[str]:
    """Authored physics-material attributes on a material prim, or NO MATERIAL.

    The attributes are reached through the schema objects' own ``Get*Attr()``
    accessors, NOT ``prim.GetAttribute("restitution")``. The material prim the
    spawners create reads back '<unauthored>' for every bare-name lookup even
    when the value is on it: the Get*Attr() accessors resolve the schema's real
    property path and the bare name does not (measured 2026-09-14 — the prim
    carries ``physics:restitution = 0.63`` while ``prim.GetAttribute("restitution")``
    returns an invalid attribute). A read-back that misses this reports "not set"
    for a value that is set, which is exactly the failure this probe exists to
    prevent.

    Authored-ness is deliberately NOT gated on ``HasAuthoredValue()``: it is
    False on this stage for values that PhysX plainly receives (the same prim
    reports restitution 0 so its fallback is in play). The resolved value, which
    is what the solver uses, is printed instead.
    """
    if isinstance(prim, UsdShade.Material):
        prim = prim.GetPrim()
    if not prim or not prim.IsValid():
        return [f"  {label}: <no material prim at {path}>"]
    usd_api = UsdPhysics.MaterialAPI(prim)
    px_api = PhysxSchema.PhysxMaterialAPI(prim)
    rows = [f"  {label} @ {path}"]
    rows.append(f"      UsdShade.Material      : {prim.IsA(UsdShade.Material)}")
    rows.append(f"      UsdPhysics.MaterialAPI : {bool(usd_api)}")
    rows.append(f"      PhysxMaterialAPI       : {bool(px_api)}")
    for api, names in (
        (usd_api, ("staticFriction", "dynamicFriction", "restitution")),
        (px_api, ("frictionCombineMode", "restitutionCombineMode", "dampingCombineMode",
                  "compliantContactStiffness", "compliantContactDamping",
                  "compliantContactAccelerationSpring")),
    ):
        if not api:
            continue
        for name in names:
            attr = getattr(api, f"Get{name[0].upper()}{name[1:]}Attr", None)
            attr = attr() if attr is not None else None
            rows.append(f"      {name:34s}: {_fmt(attr.Get() if attr else None)}")
    return rows


def read_back_foot_materials(stage, root: str) -> list[str]:
    """The material prims the foot spawner created, read off the live stage."""
    rows: list[str] = []
    found = 0
    for prim in stage.Traverse():
        if prim.GetName().startswith("foot_contact_material_"):
            found += 1
            rows += _material_rows(prim, "FOOT material", str(prim.GetPath()))
    if found == 0:
        rows.append("  FOOT material: NONE CREATED — the foot spawner took its legacy "
                    "rigid branch (all five experiment values were at their defaults)")
    return rows


def read_back_bindings(stage, root: str) -> list[str]:
    """Which collision prims exist under the robot and what is bound to them."""
    rows = ["  collision prim -> bound physics material"]
    prim = stage.GetPrimAtPath(root)
    if not prim.IsValid():
        return rows + [f"    !! {root} invalid"]
    for desc in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not desc.HasAPI(UsdPhysics.CollisionAPI):
            continue
        bound = UsdShade.MaterialBindingAPI(desc).ComputeBoundMaterial(materialPurpose="physics")[0]
        rows.append(f"    {str(desc.GetPath()):70s} -> "
                    f"{bound.GetPath() if bound else '<none>'}")
    return rows


def read_back_ground(stage) -> list[str]:
    rows = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not str(prim.GetPath()).startswith(GROUND_PATH):
            continue
        bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")[0]
        rows.append(f"  ground collision prim {prim.GetPath()} -> "
                    f"{bound.GetPath() if bound else '<none>'}")
        if bound:
            rows += _material_rows(bound, "GROUND material", str(bound.GetPath()))
    if not rows:
        rows.append("  ground: NO collision prim found")
    return rows


def read_back_physx(sim, stage) -> list[str]:
    """bounce_threshold_velocity as PhysX actually holds it (the scene prim).

    The attribute name is NOT the cfg field name. ``PhysxCfg.bounce_threshold_velocity``
    lands on the physics-scene prim as ``physxScene:bounceThreshold`` — no
    ``Velocity`` suffix (measured 2026-09-14). Isaac Lab's own
    ``_set_additional_physx_params`` sets a long list of these one by one and
    this one is applied by the omniverse-side scene creation, so the cfg field
    name is the wrong thing to search the stage for.
    """
    rows = [f"  dt (from sim)                    : {sim.get_physics_dt():.6f} s"]
    rows.append(f"  cfg value handed to SimulationCfg : "
                f"{sim.cfg.physx.bounce_threshold_velocity:g} m/s")
    candidates = [sim_utils.SimulationCfg().physics_prim_path, "/physicsScene", "/World/physicsScene"]
    found = False
    for path in dict.fromkeys(candidates):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        found = True
        attr = prim.GetAttribute("physxScene:bounceThreshold")
        rows.append(f"  scene prim                       : {path}")
        rows.append(f"      physxScene:bounceThreshold (what the solver reads) : "
                    f"{_fmt(attr.Get()) if attr else '<absent>'}")
        shape = prim.GetAttribute("physxScene:solverType")
        rows.append(f"      physxScene:solverType                              : "
                    f"{_fmt(shape.Get()) if shape else '<absent>'} (1 = TGS, 0 = PGS)")
    if not found:
        rows.append("  scene prim: NOT FOUND")
    return rows


def read_back_articulation_view(robot) -> list[str]:
    """What PhysX serves the solver: per-body (static, dynamic, restitution)."""
    rows = []
    try:
        mats = robot.root_physx_view.get_material_properties()
    except Exception as exc:  # noqa: BLE001
        return [f"  root_physx_view.get_material_properties() FAILED: {exc}"]
    rows.append(f"  root_physx_view.get_material_properties() shape {tuple(mats.shape)}")
    rows.append(f"    max_shapes per link            : {robot.root_physx_view.max_shapes}")
    try:
        names = robot.body_names
    except Exception:  # noqa: BLE001
        names = [f"body{i}" for i in range(mats.shape[1])]
    m = mats.detach().cpu().reshape(mats.shape[0], -1, 3)
    if m.shape[1] == 1:
        rows.append("    (buffer is one row per articulation: PhysX exposes a single "
                    "per-articulation material here, so the robot's OTHER links share it)")
        for env in range(min(2, m.shape[0])):
            rows.append(f"    env {env}: static={m[env,0,0]:g} dynamic={m[env,0,1]:g} "
                        f"restitution={m[env,0,2]:g}")
    else:
        for body in range(m.shape[1]):
            n = names[body] if body < len(names) else f"body{body}"
            rows.append(f"    {n:34s}: static={m[0,body,0]:g} dynamic={m[0,body,1]:g} "
                        f"restitution={m[0,body,2]:g}")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Geometry: the ball's local offset and radius, straight off the collision mesh
# ─────────────────────────────────────────────────────────────────────────────
def ball_geometry(stage) -> list[dict]:
    """The ball mesh's centre (body frame) and radius, for the two foot bodies.

    The path in BALL_FOOT_COLLISION_PATHS names the collision prim, which is an
    Xform wrapping the actual mesh at ``.../node_STL_BINARY_/mesh`` (measured
    2026-09-14: the collision prim itself is type Xform with NO points, so
    UsdGeom.Mesh() on it is falsy and the ball silently vanishes from the
    geometry). Read the child mesh, and check the Xform carries no transform,
    because a nonzero one would invalidate the body-frame assumption below.

    The ball is a sphere, so each bbox half-extent is its radius: a true sphere
    shows three equal half-extents, and the printout in main() says so or says
    NOT a sphere. -Z is up in the body frame here (the bbox z runs 0 -> 0.04),
    which does not matter: the vertical direction is taken from the world
    transform of the centre, not from the local axes.
    """
    out = []
    for rel in BALL_FOOT_COLLISION_PATHS:
        prim = stage.GetPrimAtPath(f"{ROBOT_PATH}/{rel}")
        if not prim.IsValid():
            continue
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"{ROBOT_PATH}/{rel}/mesh"))
        if not mesh:
            continue
        pts = mesh.GetPointsAttr().Get()
        if not pts:
            continue
        xform_ops = [(op.GetOpName(), op.Get())
                     for op in UsdGeom.Xformable(prim).GetOrderedXformOps()]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]
        lo = (min(xs), min(ys), min(zs))
        hi = (max(xs), max(ys), max(zs))
        out.append({
            "body": rel.split("/")[0],
            "path": f"{ROBOT_PATH}/{rel}",
            "center_local": tuple(0.5 * (lo[i] + hi[i]) for i in range(3)),
            "radii": tuple(0.5 * (hi[i] - lo[i]) for i in range(3)),
            "points": len(pts),
            "max_radius": max(0.5 * (hi[i] - lo[i]) for i in range(3)),
            "xform_ops": xform_ops,
        })
    return out


def _world_offset(robot, body_name: str, local: tuple[float, float, float]) -> torch.Tensor:
    """A fixed local point in a body -> world, as a (3,) tensor."""
    bid = robot.body_names.index(body_name)
    pos = robot.data.body_pos_w[0, bid]          # (3,)
    quat = robot.data.body_quat_w[0, bid]        # (w, x, y, z)
    local_t = torch.tensor(local, device=pos.device, dtype=pos.dtype)
    # quat_apply is Isaac Lab's rotate-vector-by-quaternion
    from isaaclab.utils.math import quat_apply
    return pos + quat_apply(quat.unsqueeze(0), local_t.unsqueeze(0)).squeeze(0)


# ─────────────────────────────────────────────────────────────────────────────
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: RlEnvCfg, agent_cfg):
    device = args_cli.device or "cuda:0"

    # ── spawn config: the foot material, exactly as the env builds it ────────
    spawn = env_cfg.robot_cfg.spawn
    spawn.compliant_contact_stiffness = 0.0      # rigid contact, no custom spring
    spawn.compliant_contact_damping = 0.0
    spawn.restitution = float(args_cli.restitution)
    spawn.restitution_combine_mode = args_cli.foot_combine
    spawn.static_friction = 1.0
    spawn.dynamic_friction = 1.0

    # ── sim: training's own numbers (rl_env_cfg_p6 §2), + the one knob ───────
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 500.0,
        render_interval=env_cfg.decimation,
        physics_material=env_cfg.sim.physics_material,   # same scene material as training
        physx=sim_utils.PhysxCfg(
            gpu_max_rigid_patch_count=env_cfg.sim.physx.gpu_max_rigid_patch_count,
            bounce_threshold_velocity=float(args_cli.bounce_threshold),
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.2, 0.0, 0.9], [0.0, 0.0, 0.2])

    # ── ground: same GroundPlaneCfg the env spawns, restitution pinned ───────
    ground_mat = RigidBodyMaterialCfg(
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=float(args_cli.ground_restitution),
        friction_combine_mode="multiply",
        restitution_combine_mode=args_cli.ground_combine,
    )
    spawn_ground_plane(prim_path=GROUND_PATH, cfg=GroundPlaneCfg(physics_material=ground_mat))

    sphere = None
    robot = None
    if args_cli.arm == "sphere":
        # Isolated ball: same radius / mass / material as the foot, nothing else.
        geom = None
        # The collision meshes only exist once the robot is spawned, so the
        # sphere arm spawns the robot first, measures, then removes it.
        probe = Articulation(env_cfg.robot_cfg.replace(prim_path="/World/ProbeRobot"))
        sim.reset()
        geom = ball_geometry(sim.stage)
        radius = args_cli.sphere_radius or (geom[0]["max_radius"] if geom else 0.02)
        body = geom[0]["body"] if geom else FOOT_BODY_NAMES[0]
        bid = probe.body_names.index(body)
        mass = float(probe.root_physx_view.get_masses()[0, bid])
        say(f"[PHASE0] sphere arm: radius={radius:.5f} m  mass={mass:.5f} kg "
            f"(from {body}, the foot body that carries the ball)")
        say(f"[PHASE0] ball mesh extents = "
            f"{['%.5f' % r for r in geom[0]['radii']] if geom else 'n/a'} "
            f"({'sphere' if geom and max(geom[0]['radii']) - min(geom[0]['radii']) < 0.002 else 'NOT a sphere'})")
        sim.stage.RemovePrim("/World/ProbeRobot")
        sim_utils.delete_prim("/World/ProbeRobot")
        sim.reset()
        sphere_cfg = sim_utils.SphereCfg(
            radius=radius,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            physics_material=RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=float(args_cli.restitution),
                friction_combine_mode="multiply",
                restitution_combine_mode=args_cli.foot_combine,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        sphere_cfg.func(SPHERE_PATH, sphere_cfg)
        radius_used = radius
        robot = None
    else:
        # ── the real robot: real articulation, real ball meshes, real material.
        #    The only difference between the two robot arms is what holds the
        #    joints: a stiff PD servo (deploy-like) or a fixed joint.
        if args_cli.arm == "locked":
            # Nothing welds in this build: JointDrivePropertiesCfg.drive_type is
            # Literal["force", "acceleration"], there is no "fixed" (measured
            # 09-14). So the joints are locked by two means at once — a very
            # stiff implicit PD actuator, AND the joint state re-written to zero
            # every step. Together they hold the pose to ~1e-6 rad, which makes
            # the contact the only place energy can enter or leave. Without this
            # the PD servo does work at the impact and inflates the measured
            # restitution (e = 0.97 at a material 0.63, measured 09-14).
            robot_cfg = env_cfg.robot_cfg.replace(
                prim_path=ROBOT_PATH,
                actuators={
                    "hip_roll": ImplicitActuatorCfg(
                        joint_names_expr=["hip_roll.*"], stiffness=1.0e7,
                        damping=1.0e5, effort_limit_sim=1.0e5),
                    "hip_pitch": ImplicitActuatorCfg(
                        joint_names_expr=["hip_pitch.*"], stiffness=1.0e7,
                        damping=1.0e5, effort_limit_sim=1.0e5),
                    "knee": ImplicitActuatorCfg(
                        joint_names_expr=["knee.*"], stiffness=1.0e7,
                        damping=1.0e5, effort_limit_sim=1.0e5),
                },
            )
        else:
            robot_cfg = env_cfg.robot_cfg.replace(
                prim_path=ROBOT_PATH,
                actuators={
                    "hip_roll": ImplicitActuatorCfg(
                        joint_names_expr=["hip_roll.*"], stiffness=args_cli.hold_gain,
                        damping=0.1 * args_cli.hold_gain, effort_limit_sim=8.0),
                    "hip_pitch": ImplicitActuatorCfg(
                        joint_names_expr=["hip_pitch.*"], stiffness=args_cli.hold_gain,
                        damping=0.1 * args_cli.hold_gain, effort_limit_sim=22.0),
                    "knee": ImplicitActuatorCfg(
                        joint_names_expr=["knee.*"], stiffness=args_cli.hold_gain,
                        damping=0.1 * args_cli.hold_gain, effort_limit_sim=30.0),
                },
            )
        robot = Articulation(robot_cfg)
        sim.reset()
        radius_used = None

    sim.reset()

    # ═════════════════════════════════════════════════════════════════════════
    # READ-BACK BLOCK (constraint 2: nothing is claimed that was not read back)
    # ═════════════════════════════════════════════════════════════════════════
    stage = sim.stage
    say("")
    say("=" * 78)
    say("PHASE 0 — SETTINGS ASKED FOR")
    say("=" * 78)
    say(f"  requested foot restitution   : {args_cli.restitution:g} "
        f"(combine mode {args_cli.foot_combine})")
    say(f"  requested ground restitution : {args_cli.ground_restitution:g} "
        f"(combine mode {args_cli.ground_combine})")
    say(f"  requested bounce threshold   : {args_cli.bounce_threshold:g} m/s")
    say(f"  drop height                  : {args_cli.drop_height * 1e3:.1f} mm")
    say(f"  arm                          : {args_cli.arm}")
    say("")
    say("=" * 78)
    say("PHASE 0 — READ BACK FROM THE LIVE STAGE / PHYSX")
    say("=" * 78)
    say("-- PhysX scene --")
    for row in read_back_physx(sim, stage):
        say(row)
    say("-- ground --")
    for row in read_back_ground(stage):
        say(row)
    if robot is not None:
        say("-- foot material prims --")
        for row in read_back_foot_materials(stage, ROBOT_PATH):
            say(row)
        say("-- collision prim -> material bindings --")
        for row in read_back_bindings(stage, ROBOT_PATH):
            say(row)
        say("-- what PhysX serves the solver --")
        for row in read_back_articulation_view(robot):
            say(row)
    else:
        say("-- sphere material prim --")
        for prim in stage.Traverse():
            if prim.GetName().startswith("physicsMaterial") and SPHERE_PATH in str(prim.GetPath()):
                for row in _material_rows(prim, "SPHERE material", str(prim.GetPath())):
                    say(row)

    # ═════════════════════════════════════════════════════════════════════════
    # THE DROP
    # ═════════════════════════════════════════════════════════════════════════
    dt = sim.get_physics_dt()

    if robot is not None:
        geom = ball_geometry(stage)
        if not geom:
            say("!! no ball collision mesh found — cannot position the drop")
            return
        say("-- ball geometry (from the collision mesh) --")
        for g in geom:
            say(f"  {g['body']:14s} centre_local={tuple(round(c, 5) for c in g['center_local'])} "
                f"radii={tuple(round(r, 5) for r in g['radii'])} pts={g['points']}")
            say(f"                 collision-prim xform: "
                f"{[(n, tuple(round(v, 6) for v in val) if hasattr(val, '__len__') else val) for n, val in g['xform_ops']]}")
        radii = [max(g["radii"]) for g in geom]
        radius_used = max(radii)
        say(f"  using radius {radius_used:.5f} m (max sphere radius across ball meshes)")

        def ball_lowest() -> float:
            """Lowest point of either ball, world z."""
            zs = []
            for g in geom:
                c = _world_offset(robot, g["body"], g["center_local"])
                zs.append(float(c[2].item()) - max(g["radii"]))
            return min(zs)

        robot.update(dt)
        root = robot.data.root_state_w.clone()
        root[:, 2] += args_cli.drop_height - ball_lowest()
        root[:, 7:13] = 0.0
        robot.write_root_state_to_sim(root)
        robot.write_joint_state_to_sim(
            torch.zeros((1, robot.num_joints), device=device),
            torch.zeros((1, robot.num_joints), device=device),
        )
    else:
        # sphere: teleport to drop height, zero velocity
        state = sphere.data.root_state_w.clone()
        state[:, 2] = radius_used + args_cli.drop_height
        state[:, 7:13] = 0.0
        sphere.write_root_state_to_sim(state)

        def ball_lowest() -> float:
            return float(sphere.data.root_pos_w[0, 2].item()) - radius_used

    # one update so the printed start height is the real one
    sim.step()
    if robot is not None:
        robot.update(dt)
    else:
        sphere.update(dt)

    say("")
    say("=" * 78)
    say("PHASE 0 — DROP TRACE")
    say("=" * 78)
    say(f"  t, ball_low_z (m), ball_low_vz (m/s); every {args_cli.trace_every}th step")

    _zero_joints = torch.zeros((1, robot.num_joints), device=device) if (
        robot is not None and args_cli.arm == "locked") else None

    trace: list[tuple[float, float, float]] = []
    for i in range(args_cli.steps):
        sim.step()
        if robot is not None:
            if _zero_joints is not None:
                # Second half of the joint lock, and it must run AFTER the step:
                # zeroing the joint velocity before the step leaves the legs free
                # to fold during it, and the write is not a constraint, so the
                # pose snaps back with the impulse it should have carried (seen
                # 09-14: the body stalled on the leg and sank 34 mm into the
                # floor instead of free-falling).
                robot.write_joint_state_to_sim(_zero_joints, _zero_joints)
            robot.update(dt)
            z_low = ball_lowest()
            # the ball body's own vertical velocity — the honest impact/rebound
            # speed. The ROOT vz is only a proxy, and a bad one whenever the
            # joints move.
            bid = robot.body_names.index(geom[0]["body"])
            vz = float(robot.data.body_lin_vel_w[0, bid, 2].item())
        else:
            sphere.update(dt)
            z_low = ball_lowest()
            vz = float(sphere.data.root_lin_vel_w[0, 2].item())
        trace.append((i * dt, z_low, vz))
        if args_cli.trace_every and i % args_cli.trace_every == 0:
            say(f"    t={i*dt:6.3f}  ball_low_z={z_low:8.5f}  vz={vz:+7.3f}")

    zs = [t[1] for t in trace]
    i_min = min(range(len(zs)), key=lambda i: zs[i])
    t_impact = trace[i_min][0]
    z_min = zs[i_min]
    # rebound: highest ball-low after the bottom, inside a 0.3 s window (a
    # toppling robot's later peak is not a rebound)
    window = [t for t in trace[i_min:] if t[0] <= t_impact + 0.30]
    peak = max(window, key=lambda t: t[1]) if window else trace[i_min]
    rebound = peak[1] - z_min

    # impact speed from the trace (free-fall check)
    v_impact = trace[i_min][2]
    v_theory = -math.sqrt(2.0 * 9.81 * args_cli.drop_height)

    say("")
    say("=" * 78)
    say("PHASE 0 — RESULT")
    say("=" * 78)
    say(f"  released at ball-low z      : {trace[1][1]:.5f} m (requested "
        f"{args_cli.drop_height:.4f} m)")
    say(f"  lowest ball-low z           : {z_min:.5f} m at t={t_impact:.3f} s "
        f"(penetration {-z_min*1e3:+.2f} mm)")
    say(f"  root vz at the bottom       : {v_impact:+.3f} m/s "
        f"(free-fall theory {v_theory:+.3f} m/s)")
    say(f"  highest ball-low after impact: {peak[1]:.5f} m at t={peak[0]:.3f} s")
    say(f"  REBOUND                     : {rebound*1e3:.1f} mm")

    # ── the coefficient itself, from the velocity reversal at the contact ────
    # A single step either side of the bottom is NOT the contact: at 500 Hz the
    # reversal spreads over several steps, and the step-adjacent vz values can
    # both read ≈0 while the body is genuinely moving (measured 09-14). So the
    # contact segment is found by direction, and each side is fitted with the
    # parabola the free-flight samples must lie on (z = z_c + v·(t−t_c) − g/2·(t−t_c)²,
    # with g known), which recovers v at the contact instant rather than at a
    # sample. The height ratio is the independent second estimate.
    t_c = trace[i_min][0]

    def fit_v(sign: int) -> float:
        """Velocity at t_c from the free-flight samples on one side."""
        seg = [(t, z) for t, z, _ in trace[1:]
               if (t - t_c) * sign < 0 and abs(t - t_c) <= 0.020]
        if len(seg) < 3:
            return float("nan")
        # least squares on z = a + b·(t−t_c) + c·(t−t_c)², v = b
        n = len(seg)
        xs = [t - t_c for t, _ in seg]
        zs = [z for _, z in seg]
        s0, s1, s2, s3, s4 = n, sum(xs), sum(x * x for x in xs), \
            sum(x**3 for x in xs), sum(x**4 for x in xs)
        b0, b1, b2 = sum(zs), sum(x * z for x, z in zip(xs, zs)), \
            sum(x * x * z for x, z in zip(xs, zs))
        import numpy as np
        try:
            sol = np.linalg.solve(np.array([[s0, s1, s2], [s1, s2, s3], [s2, s3, s4]]),
                                  np.array([b0, b1, b2], dtype=float))
        except Exception:  # noqa: BLE001
            return float("nan")
        return float(sol[1])

    v_in = fit_v(+1)     # samples before the bottom
    v_out = fit_v(-1)    # samples after
    e_vel = (-v_out / v_in) if (v_in < -1e-3 and math.isfinite(v_out)) else float("nan")
    say(f"  ball vz in  (fit at contact) : {v_in:+.3f} m/s")
    say(f"  ball vz out (fit at contact) : {v_out:+.3f} m/s")
    say(f"  e from the velocity reversal : {e_vel:.3f}   (requested {args_cli.restitution:g})")
    e_eff = math.sqrt(max(rebound, 0.0) / args_cli.drop_height) if args_cli.drop_height > 0 else 0.0
    say(f"  e from the height ratio      : {e_eff:.3f}")

    # ── verdict ──────────────────────────────────────────────────────────────
    say("")
    if args_cli.restitution >= 0.6:
        lo, hi = 70.0, 90.0
    elif abs(args_cli.restitution - 0.3) < 0.05:
        lo, hi = 12.0, 24.0
    else:
        lo = hi = 0.0
    mm = rebound * 1e3
    if lo == hi == 0.0:
        verdict = "INFO"
        say(f"  VERDICT: e={args_cli.restitution:g} has no pre-registered window; "
            f"recorded {mm:.1f} mm.")
    elif lo <= mm <= hi:
        verdict = "PASS"
        say(f"  VERDICT: PASS — rebound {mm:.1f} mm inside the {lo:.0f}-{hi:.0f} mm window "
            f"(e^2 x 200 mm).")
    else:
        verdict = "FAIL"
        say(f"  VERDICT: FAIL — rebound {mm:.1f} mm outside the {lo:.0f}-{hi:.0f} mm window "
            f"(e^2 x 200 mm).")
        if mm < 1.0:
            say("    -> zero rebound. In order: (a) the foot material was not bound to the")
            say("       ball collision prim (see the binding rows above), (b) the combine")
            say("       mode on the ground zeroed it, (c) bounce_threshold_velocity is")
            say("       above the impact speed.")

    # ── write the record ─────────────────────────────────────────────────────
    tag = f"_{args_cli.tag}" if args_cli.tag else ""
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "logs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"foot_drop_{args_cli.arm}_e{args_cli.restitution:g}"
                                     f"{'_thr%g' % args_cli.bounce_threshold}"
                                     f"_{args_cli.ground_combine}{tag}.log")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(REPORT_LINES) + "\n")
        fh.write("\n  full trace (t, ball_low_z, vz):\n")
        for t, z, v in trace:
            fh.write(f"    {t:8.4f} {z:10.6f} {v:+9.4f}\n")
    say(f"  record: {out_path}")
    say(f"  VERDICT CODE: {verdict}")


if __name__ == "__main__":
    main()
    simulation_app.close()
