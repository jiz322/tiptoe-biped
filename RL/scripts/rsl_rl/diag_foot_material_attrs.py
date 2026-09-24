"""One-question diagnostic: what exactly is authored on the foot material prim?

The drop probe's read-back block reported every PhysX/USD material attribute as
"<unauthored>" on a prim that PhysX itself returns restitution 0.63 for, so the
query is wrong (or looking at the wrong prim) rather than the value being absent.

Prints, for the material prim the foot spawner created:
  * every property GetPropertyNames() reports, with its type and value
  * the same list via GetAuthoredProperties()
  * the raw attribute lookups the probe uses, with HasAuthoredValue / Get()
  * the prim's namespace contents and applied schemas
"""
from __future__ import annotations

import os
import sys

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # <repo>/RL
from RL.tasks.direct.standup_curriculum.rl_env_cfg_p6 import (  # noqa: E402
    RlEnvCfg,
)

ROBOT_PATH = "/World/Robot"

cfg = RlEnvCfg()
spawn = cfg.robot_cfg.spawn
spawn.restitution = 0.63
spawn.static_friction = 1.0
spawn.dynamic_friction = 1.0

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 500.0))
spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
robot = Articulation(cfg.robot_cfg.replace(prim_path=ROBOT_PATH))
sim.reset()

stage = sim.stage
print("\n" + "=" * 78, flush=True)
print("MATERIAL PRIM INSPECTION", flush=True)
print("=" * 78, flush=True)
prim = stage.GetPrimAtPath(f"{ROBOT_PATH}/foot_contact_material_node_STL_BINARY_")
print(f"prim          : {prim.GetPath()}")
print(f"valid         : {prim.IsValid()}")
print(f"type          : {prim.GetTypeName()}")
print(f"schemas       : {list(prim.GetAppliedSchemas())}")
print(f"children/names: {[c.GetName() for c in prim.GetChildren()]}")

for title, api in (("UsdPhysics.MaterialAPI", UsdPhysics.MaterialAPI(prim)),
                   ("PhysxSchema.PhysxMaterialAPI", PhysxSchema.PhysxMaterialAPI(prim))):
    print(f"\n-- {title}: truthy={bool(api)}")
    if not api:
        continue
    for name in ("staticFriction", "dynamicFriction", "restitution",
                 "frictionCombineMode", "restitutionCombineMode", "dampingCombineMode",
                 "compliantContactStiffness", "compliantContactDamping",
                 "compliantContactAccelerationSpring"):
        attr = prim.GetAttribute(name)
        exists = bool(attr)
        authored = attr.HasAuthoredValue() if exists else None
        val = attr.Get() if exists else None
        print(f"   {name:34s} exists={str(exists):5s} authored={str(authored):5s} value={val!r}")
    got = api.GetPrim().GetAttribute("restitution")
    print(f"   api.GetPrim() is the same prim : {api.GetPrim() == prim}")

print("\n-- every property on the prim --")
for prop in prim.GetProperties():
    try:
        val = prop.Get()
    except Exception as exc:  # noqa: BLE001
        val = f"<{exc}>"
    print(f"   {prop.GetName():40s} type={prop.GetPropertyTypeName():14s} value={val!r}")

print("\n" + "=" * 78, flush=True)
print("ROBOT MATERIAL BUFFER (root_physx_view)", flush=True)
print("=" * 78, flush=True)
mats = robot.root_physx_view.get_material_properties()
print(f"shape {tuple(mats.shape)}  max_shapes {robot.root_physx_view.max_shapes}")
m = mats.detach().cpu().reshape(mats.shape[0], -1, 3)[0]
for i, n in enumerate(robot.body_names):
    if m[i, 2] != 0.0 or m[i, 0] != 1.0:
        print(f"   {n:34s}: static={m[i,0]:g} dynamic={m[i,1]:g} restitution={m[i,2]:g}")
print("   (only rows that differ from the 1/1/0 default are listed)")

print("\n" + "=" * 78, flush=True)
print("BALL MESH GEOMETRY VIA INSTANCE PROXIES", flush=True)
print("=" * 78, flush=True)
root = stage.GetPrimAtPath(ROBOT_PATH)
for rel in ("shin_inner/collisions/part_3/node_STL_BINARY_",
            "nail___fillet1/collisions/part_3/node_STL_BINARY_"):
    path = f"{ROBOT_PATH}/{rel}"
    p = stage.GetPrimAtPath(path)
    print(f"\n{path}")
    print(f"   valid={p.IsValid()}  type={p.GetTypeName()}  instance_proxy={p.IsInstanceProxy()}")
    # the same prim, reached by traversing instance proxies
    for desc in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if str(desc.GetPath()) == path or (desc.GetName() == "node_STL_BINARY_"
                                           and "/part_3/" in str(desc.GetPath())
                                           and rel.split("/")[0] in str(desc.GetPath())):
            from pxr import UsdGeom
            mesh = UsdGeom.Mesh(desc)
            pts = mesh.GetPointsAttr().Get() if mesh else None
            print(f"   traversed prim {desc.GetPath()} proxy={desc.IsInstanceProxy()} "
                  f"is_mesh={bool(mesh)} points={len(pts) if pts else None}")
            if pts:
                xs = [q[0] for q in pts]
                ys = [q[1] for q in pts]
                zs = [q[2] for q in pts]
                lo = (min(xs), min(ys), min(zs))
                hi = (max(xs), max(ys), max(zs))
                print(f"      bbox lo={tuple(round(v,5) for v in lo)} "
                      f"hi={tuple(round(v,5) for v in hi)}")
                print(f"      centre={tuple(round(0.5*(lo[i]+hi[i]),5) for i in range(3))} "
                      f"halfextent={tuple(round(0.5*(hi[i]-lo[i]),5) for i in range(3))}")
            break

simulation_app.close()
