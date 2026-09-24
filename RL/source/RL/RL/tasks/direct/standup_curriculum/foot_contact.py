"""Per-foot compliant-contact material experiments for the ball-foot robot.

The default (all experiment values zero) returns the legacy USD spawner without
creating or binding a material.  Nonzero values are stage/material properties,
so they are pinned for one run; ArticulationView can randomize only the three
material tensor channels (friction, friction, restitution), not compliant gains.
Run a separate fixed-policy evaluation for each spring/damper pair.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from isaaclab.sim.spawners.from_files import from_files_cfg
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.sim.utils import (
    bind_physics_material,
    clone,
    make_uninstanceable,
    safe_set_attribute_on_usd_schema,
)
from isaaclab.utils import configclass
from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade

# Actual ball collision meshes in robot_physics.usd (left and mirrored right).
# NOT verifiable from the binary USD crate on this machine (robot.usd is a
# 1.5 KB reference to the payload, and the prim names live in the generated
# payload), so these are an ASSUMPTION from the earlier inspection and are
# checked at spawn time instead: a wrong path raises with the paths that DO
# exist under the articulation root, which is the fastest way to correct it.
BALL_FOOT_COLLISION_PATHS: tuple[str, ...] = (
    "shin_inner/collisions/part_3/node_STL_BINARY_",
    "nail___fillet1/collisions/part_3/node_STL_BINARY_",
)


def _spawn_foot_contact(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Implemented below, after the config type it needs exists.

    The spawner and the config refer to each other, so one of them has to be
    late-bound.  It must be late-bound by ASSIGNMENT INSIDE THE CLASS BODY
    (``func = _spawn_foot_contact``), never by writing ``FootContactUsdFileCfg.func
    = ...`` afterwards: @configclass bakes its defaults the same way
    @dataclass does, into the generated __init__, so a post-class attribute
    assignment never reaches instances and every spawned asset sees
    ``func is None`` — which Isaac Lab then calls, raising
    "TypeError: 'NoneType' object is not callable" at Articulation creation.
    Observed exactly once, on the first real sweep launch (2026-09-11).
    """
    return spawn_from_usd_with_foot_contact_material(
        prim_path, cfg, translation, orientation, **kwargs
    )


@configclass
class FootContactUsdFileCfg(from_files_cfg.UsdFileCfg):
    """UsdFileCfg with optional ball-foot compliance and restitution.

    Gains use force-based PhysX units: stiffness N/m and damping N s/m.
    Friction defaults to 1.0 so a compliance experiment is independent of floor
    friction DR; set it to 0 to remove the ball's traction instead. Set
    stiffness to zero to test rigid contact restitution.
    """

    # In the CLASS BODY on purpose — see _spawn_foot_contact's docstring.
    func = _spawn_foot_contact
    compliant_contact_stiffness: float = 0.0
    compliant_contact_damping: float = 0.0
    restitution: float = 0.0
    restitution_combine_mode: Literal["average", "min", "multiply", "max"] = "max"
    static_friction: float = 1.0
    dynamic_friction: float = 1.0
    # Obstacle-course knob, NOT a physics-correctness constraint: PhysX reads a
    # restitution above 1.0 without complaint and the contact then returns more
    # energy than it took in.  That is the point of asking for it here (the
    # user wants an OOD plant to make the policy fail on purpose), so there is
    # deliberately no upper bound.  Set `clamp_foot_restitution_to_unity` when
    # a value that is meant to be physical (a measurement, a DR band) must not
    # silently exceed 1.
    clamp_foot_restitution_to_unity: bool = False
    physics_material_prim_path: Sequence[str] = BALL_FOOT_COLLISION_PATHS


@clone
def spawn_from_usd_with_foot_contact_material(
    prim_path: str,
    cfg: FootContactUsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    _validate_contact_cfg(cfg)
    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    # Preserve the exact legacy path when all experiment parameters are zero.
    if (
        cfg.compliant_contact_stiffness == 0.0
        and cfg.compliant_contact_damping == 0.0
        and cfg.restitution == 0.0
        and cfg.static_friction == 1.0
        and cfg.dynamic_friction == 1.0
    ):
        return prim
    mat_cfg = RigidBodyMaterialCfg(
        static_friction=cfg.static_friction,
        dynamic_friction=cfg.dynamic_friction,
        restitution=cfg.restitution,
        friction_combine_mode="multiply",
        restitution_combine_mode=cfg.restitution_combine_mode,
        compliant_contact_stiffness=cfg.compliant_contact_stiffness,
        compliant_contact_damping=cfg.compliant_contact_damping,
    )
    stage = prim.GetStage()
    for rel in cfg.physics_material_prim_path:
        target = rel if rel.startswith("/") else f"{prim_path}/{rel}"
        collision = stage.GetPrimAtPath(target)
        if not collision.IsValid() or not collision.HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError(
                f"Foot material target is not a collision prim: {target}. "
                f"Collision prims actually present under {prim_path}: "
                f"{_list_collision_prims(stage, prim_path)}"
            )
        # The two feet are USD instance proxies.  Deinstance only this tiny
        # collision subtree before adding a per-foot material binding; leaving
        # it instanced makes bind_physics_material a no-op.
        while collision.IsInstanceProxy():
            ancestor = collision.GetParent()
            while ancestor.IsInstanceProxy():
                ancestor = ancestor.GetParent()
            if not ancestor.IsInstance():
                raise RuntimeError(f"Cannot make foot collision editable: {target}")
            make_uninstanceable(ancestor.GetPath(), stage=stage)
            collision = stage.GetPrimAtPath(target)
        # Material lives in the articulation root, not under the collision
        # prim: PhysX expects a collision prim to be a leaf mesh, and the
        # upstream spawner's placement under the mesh is a known wart.
        node = target.rsplit("/", 1)[-1]
        material_path = f"{prim_path}/foot_contact_material_{node}"
        mat_cfg.func(material_path, mat_cfg)
        material = stage.GetPrimAtPath(material_path)
        if not material.IsValid():
            raise RuntimeError(f"Foot contact material was not created at {material_path}")
        material_api = PhysxSchema.PhysxMaterialAPI(material)
        if not material_api:
            material_api = PhysxSchema.PhysxMaterialAPI.Apply(material)
        # Explicitly use force gains; acceleration gains would hide mass DR.
        # (Isaac Lab's own compliant-contact spawner does not set this, so it
        # silently takes the schema default, which is acceleration-based — a
        # different plant.)  Set through the schema helper rather than the
        # typed Create* accessors: those names differ across PhysX versions,
        # and the helper raises a clear TypeError if this build lacks one.
        safe_set_attribute_on_usd_schema(
            material_api, "compliant_contact_acceleration_spring", False, camel_case=True
        )
        safe_set_attribute_on_usd_schema(
            material_api, "damping_combine_mode", "max", camel_case=True
        )
        safe_set_attribute_on_usd_schema(
            material_api, "friction_combine_mode", "multiply", camel_case=True
        )
        safe_set_attribute_on_usd_schema(
            material_api, "restitution_combine_mode", cfg.restitution_combine_mode, camel_case=True
        )
        bind_physics_material(target, material_path, stage=stage)
        bound = UsdShade.MaterialBindingAPI(collision).ComputeBoundMaterial(materialPurpose="physics")[0]
        if not bound or str(bound.GetPath()) != material_path:
            raise RuntimeError(
                f"Foot contact material binding failed: {target} reports "
                f"{bound.GetPath() if bound else 'no bound material'}"
            )
    return prim


def _list_collision_prims(stage, root: str, limit: int = 40) -> list[str]:
    """Diagnostic: the collision prims that exist under ``root``.

    TraverseInstanceProxies is required, not decorative.  Every link in this
    asset keeps its collision meshes under an `IsInstance()=True` scope, and a
    plain PrimRange does NOT descend into an instanced subtree — so this
    diagnostic used to report an empty list on a perfectly healthy stage,
    which is the opposite of what a diagnostic is for (measured 2026-09-13:
    it printed "Collision prims present: []" for a path that holds five).
    """
    found: list[str] = []
    prim = stage.GetPrimAtPath(root)
    if not prim.IsValid():
        return found
    for descendant in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if descendant.HasAPI(UsdPhysics.CollisionAPI):
            found.append(str(descendant.GetPath()))
            if len(found) >= limit:
                break
    return found


def _validate_contact_cfg(cfg: FootContactUsdFileCfg) -> None:
    """Reject the values PhysX cannot express.  Nothing else.

    This deliberately does NOT bound the parameters to a physically sensible
    range.  Driving the plant out of distribution is a supported use (it is how
    the eval arms check that the policy is actually reading the contact at
    all), and a validator that refuses an extreme value turns an experiment
    into a crash.  Restitution > 1 is energy-gaining contact, which is exactly
    what a "make it fail" arm asks for; the only honest guard is against values
    with no meaning at all.
    """
    for name in ("compliant_contact_stiffness", "compliant_contact_damping", "restitution", "static_friction", "dynamic_friction"):
        value = float(getattr(cfg, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    if not cfg.physics_material_prim_path:
        raise ValueError("physics_material_prim_path must contain at least one mesh")
    if cfg.clamp_foot_restitution_to_unity and cfg.restitution > 1.0:
        raise ValueError(
            f"restitution {cfg.restitution} is above 1 while "
            "clamp_foot_restitution_to_unity is set; a restitution that is "
            "meant to be physical must be in [0, 1]"
        )
    if cfg.compliant_contact_stiffness == 0.0 and cfg.compliant_contact_damping != 0.0:
        raise ValueError("contact damping requires positive stiffness; use damping=0 for rigid restitution")
    if cfg.restitution_combine_mode not in ("average", "min", "multiply", "max"):
        raise ValueError("unsupported restitution combine mode")


def damping_for_ratio(stiffness_n_per_m: float, damping_ratio: float, effective_mass_kg: float) -> float:
    """Return c = 2*zeta*sqrt(k*m), in N s/m."""
    if stiffness_n_per_m < 0.0 or damping_ratio < 0.0 or effective_mass_kg <= 0.0:
        raise ValueError("stiffness and damping ratio must be non-negative; mass must be positive")
    return 2.0 * damping_ratio * math.sqrt(stiffness_n_per_m * effective_mass_kg)


__all__ = ["BALL_FOOT_COLLISION_PATHS", "FootContactUsdFileCfg", "spawn_from_usd_with_foot_contact_material", "damping_for_ratio"]
