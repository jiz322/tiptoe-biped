"""
find_standing_pose_editor.py — Run inside Isaac Sim's Script Editor

Paste this into the Script Editor (Window > Script Editor) and press Run.
The robot will be spawned on a ground plane at the specified joint angles.

Workflow:
    1. Set your USD path below
    2. Tweak JOINT_ANGLES until the robot stands upright
    3. Press Run again after each change (it clears the old robot first)
    4. Press PLAY in the viewport to start physics
"""

import os
import numpy as np
import asyncio
from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema, Sdf, PhysicsSchemaTools
import omni.usd
import omni.kit.app
import omni.kit.commands

# ============================================================
# USER CONFIG — EDIT THESE
# ============================================================

# Path to your robot USD file
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
USD_PATH = os.path.join(_REPO, "assets", "tiptoe_v27", "robot", "robot.usd")

# Height above ground to spawn the robot (meters)
SPAWN_HEIGHT = 0.42

# Joint angles in radians — THIS IS WHAT YOU ITERATE ON
# Tip: change one at a time in 0.1 rad (~5.7 deg) increments
JOINT_ANGLES = {
    "hip_roll1":  0.0,
    "hip_pitch1": 0.0,
    "knee1":      0.0,
    "hip_roll2":  0.0,
    "hip_pitch2": 0.0,
    "knee2":      0.0,
}

# PD drive gains for holding position
JOINT_STIFFNESS = 40.0   # Nm/rad — increase if too floppy
JOINT_DAMPING   = 5.0    # Nm·s/rad — increase if oscillating

# ============================================================
# SCRIPT — no need to edit below unless debugging
# ============================================================

async def setup_robot():
    stage = omni.usd.get_context().get_stage()

    # ---- Clean up previous run ----
    if stage.GetPrimAtPath("/World"):
        omni.kit.commands.execute("DeletePrims", paths=["/World"])
        # Wait a frame for deletion to propagate
        await omni.kit.app.get_app().next_update_async()

    # ---- Create /World scope ----
    UsdGeom.Xform.Define(stage, "/World")

    # ---- Physics scene ----
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)

    # ---- Ground plane ----
    PhysicsSchemaTools.addGroundPlane(
        stage, "/World/GroundPlane", "Z", 10.0,
        Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.5, 0.5, 0.5)
    )

    # ---- Load robot USD as a reference ----
    robot_prim_path = "/World/Robot"
    robot_prim = stage.DefinePrim(robot_prim_path)
    robot_prim.GetReferences().AddReference(USD_PATH)

    # Wait for USD to load
    await omni.kit.app.get_app().next_update_async()
    await omni.kit.app.get_app().next_update_async()

    # Set spawn position (translate up so robot doesn't clip ground)
    xform = UsdGeom.Xformable(robot_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, SPAWN_HEIGHT))

    # ---- Scan stage for joints and configure them ----
    print("\n" + "=" * 60)
    print("SCANNING FOR JOINTS IN THE USD STAGE...")
    print("=" * 60)

    found_joints = []
    for prim in stage.Traverse():
        # Check if this prim has a revolute or prismatic joint schema
        if prim.IsA(UsdPhysics.Joint):
            joint_name = prim.GetName()
            joint_path = str(prim.GetPath())
            found_joints.append(joint_name)
            print(f"  Found joint: {joint_name}  (path: {joint_path})")

            # If this joint is in our config, set its drive target
            if joint_name in JOINT_ANGLES:
                target_rad = JOINT_ANGLES[joint_name]
                target_deg = np.degrees(target_rad)

                # Apply angular drive API for position control
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.CreateTypeAttr("force")
                drive.CreateStiffnessAttr(JOINT_STIFFNESS)
                drive.CreateDampingAttr(JOINT_DAMPING)
                drive.CreateTargetPositionAttr(target_deg)  # PhysX uses degrees

                print(f"    -> Target: {target_rad:.3f} rad ({target_deg:.1f} deg)")
                print(f"    -> Drive: stiffness={JOINT_STIFFNESS}, damping={JOINT_DAMPING}")

    # Also check for generic Joint type
    if len(found_joints) == 0:
        print("\n  No RevoluteJoint prims found. Scanning for generic joints...")
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Joint):
                joint_name = prim.GetName()
                joint_path = str(prim.GetPath())
                found_joints.append(joint_name)
                joint_type = prim.GetTypeName()
                print(f"  Found: {joint_name}  type={joint_type}  (path: {joint_path})")

                if joint_name in JOINT_ANGLES:
                    target_rad = JOINT_ANGLES[joint_name]
                    target_deg = np.degrees(target_rad)

                    drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                    drive.CreateTypeAttr("force")
                    drive.CreateStiffnessAttr(JOINT_STIFFNESS)
                    drive.CreateDampingAttr(JOINT_DAMPING)
                    drive.CreateTargetPositionAttr(target_deg)

                    print(f"    -> Target: {target_rad:.3f} rad ({target_deg:.1f} deg)")

    # ---- Report ----
    print(f"\nTotal joints found: {len(found_joints)}")

    configured = set(JOINT_ANGLES.keys())
    found = set(found_joints)
    missing = configured - found
    if missing:
        print(f"\n*** WARNING ***")
        print(f"  These joints from JOINT_ANGLES were NOT found: {missing}")
        print(f"  Available joint names: {found_joints}")
        print(f"  The URDF importer may have renamed them.")
        print(f"  Update JOINT_ANGLES keys to match the names above.")

    extra = found - configured
    if extra:
        print(f"\n  Note: joints in USD not in your config: {extra}")
        print(f"  (These may be fixed joints like 'imu_frame')")

    print("\n" + "=" * 60)
    print(f"  Spawn height:  {SPAWN_HEIGHT} m")
    print(f"  Joint targets: {JOINT_ANGLES}")
    print("=" * 60)
    print("\n>>> Press PLAY in the toolbar to start physics.")
    print(">>> Watch if the robot stands or falls.")
    print(">>> Adjust JOINT_ANGLES, then re-run this script.\n")


# Launch the async setup
asyncio.ensure_future(setup_robot())