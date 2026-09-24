"""
show_kneel.py — kneeling pose viewer.

Spawns the robot in a kneeling pose at the correct FK-computed height
so it doesn't bounce off the ground. Includes hip-pitch sliders for
real-time tuning and prints joint angles and IMU state once per second.

Two ways to run:

A) Standalone (recommended):
       powershell -File script\run_show_kneel.ps1
   The wrapper sets the Isaac Sim env vars (ISAAC_PATH etc.) and uses the
   env_isaaclab python. Plain `python show_kneel.py` will NOT work — pxr/omni
   modules only exist once SimulationApp is up, which needs those env vars.

B) Inside Isaac Sim's Script Editor (original workflow):
   paste/run, then press PLAY.

Either way:
    1. Set your USD path below
    2. Press PLAY in the viewport — robot kneels steadily
    3. Drag sliders to adjust hip pitch while physics runs
    4. Joint angles and IMU frame state print to console every second
"""

# ---- Standalone bootstrap: must happen BEFORE any pxr/omni import ----
_STANDALONE = __name__ == "__main__"
simulation_app = None
if _STANDALONE:
    try:
        from isaacsim import SimulationApp          # Isaac Sim >= 4.x
    except ImportError:
        from omni.isaac.kit import SimulationApp    # older fallback
    simulation_app = SimulationApp({"headless": False})

import os
import numpy as np
import math
import asyncio
from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema, Sdf, PhysicsSchemaTools
import omni.usd
import omni.kit.app
import omni.kit.commands
import omni.ui as ui

# ============================================================
# USER CONFIG — EDIT THESE
# ============================================================

# Last exercised against the earlier bigthigh-urdf export; the shipped asset has the same link/joint names.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
USD_PATH = os.path.join(_REPO, "assets", "tiptoe_v27", "robot", "robot.usd")

# Name of the IMU link in your URDF/USD
IMU_LINK_PATH = "/World/Robot/joints/imu_frame"

# Joint angles in radians — the kneel pose
JOINT_ANGLES = {
    "hip_roll1":  0.0,
    "hip_pitch1": 3.09,
    "knee1":     -3.06,
    "hip_roll2":  0.0,
    "hip_pitch2": -3.09,
    "knee2":      3.06,
}

# PD drive gains
JOINT_STIFFNESS = 40.0
JOINT_DAMPING   = 5.0

# Orientation quaternion (Y-up URDF → Z-up Isaac)
SPAWN_QUAT = Gf.Quatd(0.7071, 0.0, -0.7071, 0.0)

# Small upward margin so the robot doesn't intersect ground at frame 0
SPAWN_HEIGHT_MARGIN = 0.005  # meters above FK-computed height

# ============================================================
# FORWARD KINEMATICS — copied from rl_env.py (CPU, single-env)
# ============================================================

def _fixed_tf(xyz, rpy):
    """Build a 4×4 homogeneous matrix from xyz + roll-pitch-yaw."""
    x, y, z = xyz
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr, x],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr, y],
        [-sp,   cp*sr,            cp*cr,             z],
        [0.0,   0.0,              0.0,               1.0],
    ])


def _Rz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1],
    ])


def _joint_tf(fixed, angle):
    return fixed @ _Rz(angle)


# URDF chain constants (from rl_env.py _compute_fk_height)
_F_hr1 = _fixed_tf((-0.070392, 0.0800218, -0.0315), (1.5708, 0.0, 1.5708))
_F_hp1 = _fixed_tf((-0.0415, 0.0, -0.071), (0.0, 1.5708, 0.0))
_F_kn1 = _fixed_tf((-0.0592463, -0.172173, 0.0969), (0.0, 0.0, 1.72565))
_T_nail1 = np.array([
    [1, 0, 0, -0.138448], [0, 1, 0, 0.0216628],
    [0, 0, 1, -0.01835],  [0, 0, 0, 1.0],
])

_F_hr2 = _fixed_tf((-0.070392, -0.0799782, -0.0315), (-3.14159, -1.5708, 0.0))
_F_hp2 = _fixed_tf((0.0, -0.0415, -0.071), (-1.5708, 0.908015, 0.0))
_F_kn2 = _fixed_tf((-0.154185, 0.107117, 0.0602), (0.0, 0.0, 0.662781))
_T_nail2 = np.array([
    [1, 0, 0, -4.95155e-05], [0, 1, 0, 0.140191],
    [0, 0, 1, 0.01835],      [0, 0, 0, 1.0],
])

_FK_TARGET_FOOT_Z = 0.0716  # desired foot-tip z on ground


def _quat_to_rot4(qw, qx, qy, qz):
    """Quaternion (w,x,y,z) → 4×4 rotation matrix (no translation)."""
    R = np.zeros((4, 4))
    R[0, 0] = 1 - 2*(qy*qy + qz*qz)
    R[0, 1] = 2*(qx*qy - qw*qz)
    R[0, 2] = 2*(qx*qz + qw*qy)
    R[1, 0] = 2*(qx*qy + qw*qz)
    R[1, 1] = 1 - 2*(qx*qx + qz*qz)
    R[1, 2] = 2*(qy*qz - qw*qx)
    R[2, 0] = 2*(qx*qz - qw*qy)
    R[2, 1] = 2*(qy*qz + qw*qx)
    R[2, 2] = 1 - 2*(qx*qx + qy*qy)
    R[3, 3] = 1.0
    return R


def compute_spawn_height(joint_angles, quat):
    """Return the base z so the lowest foot tip sits at _FK_TARGET_FOOT_Z."""
    qw, qx, qy, qz = quat
    T_base = _quat_to_rot4(qw, qx, qy, qz)

    hr1 = joint_angles["hip_roll1"]
    hp1 = joint_angles["hip_pitch1"]
    kn1 = joint_angles["knee1"]
    hr2 = joint_angles["hip_roll2"]
    hp2 = joint_angles["hip_pitch2"]
    kn2 = joint_angles["knee2"]

    # Leg 1
    T = T_base @ _joint_tf(_F_hr1, hr1) @ _joint_tf(_F_hp1, hp1) \
        @ _joint_tf(_F_kn1, kn1) @ _T_nail1
    foot1_z = T[2, 3]

    # Leg 2
    T = T_base @ _joint_tf(_F_hr2, hr2) @ _joint_tf(_F_hp2, hp2) \
        @ _joint_tf(_F_kn2, kn2) @ _T_nail2
    foot2_z = T[2, 3]

    min_foot_z = min(foot1_z, foot2_z)
    return _FK_TARGET_FOOT_Z - min_foot_z


# ============================================================
# GLOBALS
# ============================================================

_joint_prims = {}       # name -> USD prim
_slider_models = {}     # name -> ui slider model
_window = None
_print_sub = None       # update subscription for periodic printing
_elapsed = 0.0


# ============================================================
# HELPERS
# ============================================================

def _update_joint_target(joint_name, value_rad):
    """Push a new drive target (in degrees) to a joint prim."""
    if joint_name in _joint_prims:
        prim = _joint_prims[joint_name]
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if drive:
            drive.GetTargetPositionAttr().Set(np.degrees(value_rad))
        # Also update JointStateAPI initial position so it's consistent
        joint_state = PhysxSchema.JointStateAPI.Get(prim, "angular")
        if joint_state:
            joint_state.GetPositionAttr().Set(np.degrees(value_rad))


def _read_current_angles():
    """Read actual joint positions from the running physics simulation."""
    angles = {}

    # Try reading from the live simulation via PhysX / DC
    try:
        from omni.isaac.dynamic_control import _dynamic_control
        dc = _dynamic_control.acquire_dynamic_control_interface()

        # Find the articulation (first one under /World/Robot)
        art = dc.get_articulation("/World/Robot")
        if art == _dynamic_control.INVALID_HANDLE:
            for suffix in ["/base_link", "/torso2hip", ""]:
                art = dc.get_articulation("/World/Robot" + suffix)
                if art != _dynamic_control.INVALID_HANDLE:
                    break

        if art != _dynamic_control.INVALID_HANDLE:
            for name in JOINT_ANGLES:
                dof = dc.find_articulation_dof(art, name)
                if dof != _dynamic_control.INVALID_HANDLE:
                    angles[name] = dc.get_dof_position(dof)  # radians
                else:
                    angles[name] = 0.0
            return angles
    except (ImportError, Exception):
        pass

    # Fallback: read from USD JointStateAPI (updated by PhysX each frame)
    for name in JOINT_ANGLES:
        if name in _joint_prims:
            prim = _joint_prims[name]
            joint_state = PhysxSchema.JointStateAPI.Get(prim, "angular")
            if joint_state:
                deg = joint_state.GetPositionAttr().Get()
                if deg is not None:
                    angles[name] = np.radians(deg)
                    continue
            # Last resort: drive target
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if drive:
                deg = drive.GetTargetPositionAttr().Get()
                angles[name] = np.radians(deg) if deg is not None else 0.0
            else:
                angles[name] = 0.0
        else:
            angles[name] = 0.0
    return angles


def _read_imu_state():
    """
    Read the world pose of the IMU frame using pure USD APIs.
    This avoids Isaac Sim 5.1+ deprecation warnings and handles dummy links perfectly.
    """
    from pxr import UsdGeom, Usd
    import omni.usd
    
    global IMU_LINK_PATH
    stage = omni.usd.get_context().get_stage()
    
    # 1. Try the hardcoded path
    prim = stage.GetPrimAtPath(IMU_LINK_PATH)
    
    # 2. If invalid, auto-search for it inside the robot
    if not prim.IsValid():
        for p in stage.TraverseAll():
            if "imu_frame" in p.GetName():
                prim = p
                IMU_LINK_PATH = str(p.GetPath())  # Save correct path for next frame
                break
                
    if not prim.IsValid():
        return None, None
        
    try:
        # Physics writes to the default timecode during live simulation
        time = Usd.TimeCode.Default()
        
        # Compute the 4x4 world transform matrix native to USD
        world_transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time)
        
        # Extract translation (Gf.Vec3d)
        pos = world_transform.ExtractTranslation()
        
        # Extract rotation (Gf.Rotation -> Gf.Quatd)
        rot_quat = world_transform.ExtractRotation().GetQuat()
        
        # Return as standard tuples: (x, y, z), (w, x, y, z)
        return (
            (pos[0], pos[1], pos[2]), 
            (rot_quat.GetReal(), rot_quat.GetImaginary()[0], rot_quat.GetImaginary()[1], rot_quat.GetImaginary()[2])
        )
        
    except Exception as e:
        print(f"Error computing pose for {IMU_LINK_PATH}: {e}")
        return None, None


def _print_state():
    """Print actual joint angles and IMU frame read to the console."""
    angles = _read_current_angles()
    
    print("\n--- Robot State (actual) ---")
    print("Joint Angles:")
    for name, rad in angles.items():
        print(f"  {name:>12s}: {rad:>8.4f} rad  ({np.degrees(rad):>8.2f} deg)")

    pos, rot = _read_imu_state()
    if pos is not None:
        print(f"\nIMU Link ('{IMU_LINK_PATH}'):")
        print(f"  Pos (x,y,z)  : [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        print(f"  Rot (w,x,y,z): [{rot[0]:.4f}, {rot[1]:.4f}, {rot[2]:.4f}, {rot[3]:.4f}]")
    else:
        print(f"\nIMU Link ('{IMU_LINK_PATH}'): [Data could not be read]")
        
    print("----------------------------")


def _build_ui():
    """Build the slider window for hip-pitch joints."""
    global _window

    if _window is not None:
        _window.destroy()
        _window = None

    _window = ui.Window("Kneel Pose — Hip Pitch Tuner", width=480, height=300)

    # Slider configs: (joint_name, display_label, min_rad, max_rad)
    slider_joints = [
        ("hip_pitch1", "hip_pitch1", -3.5, 6.28),
        ("hip_pitch2", "hip_pitch2", -6.28, 3.5),
    ]

    with _window.frame:
        with ui.VStack(spacing=5):
            ui.Label(
                "Adjust hip pitch while physics is running (PLAY)",
                height=25, style={"font_size": 14},
            )
            ui.Separator(height=5)

            for joint_name, label, smin, smax in slider_joints:
                init_val = JOINT_ANGLES.get(joint_name, 0.0)
                with ui.HStack(height=30, spacing=5):
                    ui.Label(label, width=120)
                    slider = ui.FloatSlider(min=smin, max=smax, step=0.01)
                    slider.model.as_float = init_val
                    _slider_models[joint_name] = slider.model

                    value_label = ui.Label(f"{init_val:.3f}", width=70)

                    def _make_cb(jname, vlabel):
                        def _on_changed(model):
                            val = model.as_float
                            vlabel.text = f"{val:.3f}"
                            _update_joint_target(jname, val)
                        return _on_changed

                    slider.model.add_value_changed_fn(_make_cb(joint_name, value_label))

            ui.Separator(height=10)

            with ui.HStack(height=30, spacing=10):
                ui.Button(
                    "Print Current State",
                    clicked_fn=_print_state,
                    style={"font_size": 13},
                )

            ui.Separator(height=5)
            ui.Label(
                "Joints and IMU state print to console every 1 s automatically.",
                style={"font_size": 12, "color": 0xFF888888},
            )


# ============================================================
# PERIODIC ANGLE PRINTER (once per second via update sub)
# ============================================================

def _on_update(e):
    global _elapsed
    _elapsed += e.payload["dt"]
    if _elapsed >= 1.0:
        _elapsed = 0.0
        _print_state()


def _start_periodic_print():
    global _print_sub
    if _print_sub is not None:
        _print_sub = None  # let old one be GC'd
    _print_sub = (
        omni.kit.app.get_app()
        .get_update_event_stream()
        .create_subscription_to_pop(_on_update, name="state_printer")
    )


# ============================================================
# SCENE SETUP
# ============================================================

async def setup_robot():
    global _joint_prims

    stage = omni.usd.get_context().get_stage()

    # ---- Clean up previous run ----
    if stage.GetPrimAtPath("/World"):
        omni.kit.commands.execute("DeletePrims", paths=["/World"])
        await omni.kit.app.get_app().next_update_async()

    # ---- /World scope ----
    UsdGeom.Xform.Define(stage, "/World")

    # ---- Physics scene ----
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)

    # Tighten solver for stable resting contact (no bounce)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/World/PhysicsScene"))
    physx_scene.CreateSolverTypeAttr("TGS")
    physx_scene.CreateMinPositionIterationCountAttr(8)
    physx_scene.CreateMinVelocityIterationCountAttr(4)

    # ---- Ground plane ----
    PhysicsSchemaTools.addGroundPlane(
        stage, "/World/GroundPlane", "Z", 10.0,
        Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.5, 0.5, 0.5),
    )

    # ---- Load robot USD ----
    robot_prim_path = "/World/Robot"
    robot_prim = stage.DefinePrim(robot_prim_path)
    robot_prim.GetReferences().AddReference(USD_PATH)

    await omni.kit.app.get_app().next_update_async()
    await omni.kit.app.get_app().next_update_async()

    # ---- Apply zero-restitution material to all robot colliders ----
    _apply_no_bounce_material(stage, robot_prim_path)

    # ---- Compute correct spawn height via FK ----
    qw, qx, qy, qz = (
        SPAWN_QUAT.GetReal(),
        SPAWN_QUAT.GetImaginary()[0],
        SPAWN_QUAT.GetImaginary()[1],
        SPAWN_QUAT.GetImaginary()[2],
    )
    spawn_h = compute_spawn_height(JOINT_ANGLES, (qw, qx, qy, qz))
    spawn_h += SPAWN_HEIGHT_MARGIN
    print(f"\n[FK] Computed spawn height = {spawn_h:.5f} m  (margin {SPAWN_HEIGHT_MARGIN} m)")

    # ---- Position & orientation ----
    xform = UsdGeom.Xformable(robot_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(0.0, 0.0, float(spawn_h))
    )
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(SPAWN_QUAT)

    # ---- Configure joints ----
    print("\n" + "=" * 60)
    print("SCANNING FOR JOINTS...")
    print("=" * 60)

    _joint_prims = {}
    found_joints = []

    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint_name = prim.GetName()
            joint_path = str(prim.GetPath())
            found_joints.append(joint_name)
            print(f"  Found joint: {joint_name}  (path: {joint_path})")

            if joint_name in JOINT_ANGLES:
                _joint_prims[joint_name] = prim
                target_rad = JOINT_ANGLES[joint_name]
                target_deg = np.degrees(target_rad)

                # Angular drive for position control
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.CreateTypeAttr("force")
                drive.CreateStiffnessAttr(JOINT_STIFFNESS)
                drive.CreateDampingAttr(JOINT_DAMPING)
                drive.CreateTargetPositionAttr(target_deg)

                # Set initial joint state so physics starts at this angle
                joint_state = PhysxSchema.JointStateAPI.Apply(prim, "angular")
                joint_state.CreatePositionAttr().Set(target_deg)
                joint_state.CreateVelocityAttr().Set(0.0)

                print(f"    -> Target: {target_rad:.3f} rad ({target_deg:.1f} deg)")

    # ---- Report ----
    configured = set(JOINT_ANGLES.keys())
    found = set(found_joints)
    missing = configured - found
    if missing:
        print(f"\n*** WARNING: joints not found: {missing}")
        print(f"    Available: {found_joints}")

    print(f"\nTotal joints found: {len(found_joints)}")
    print(f"Configured: {len(_joint_prims)}")
    print(f"Spawn height: {spawn_h:.5f} m")
    print("=" * 60)
    print("\n>>> Press PLAY to start physics. Robot should kneel steadily.")
    print(">>> Use the Hip Pitch Tuner window to adjust in real time.\n")

    # ---- Enable self-collision (after joints, so failure doesn't block setup) ----
    _enable_self_collision(stage, robot_prim_path)

    # ---- Build slider UI ----
    _build_ui()

    # ---- Start periodic angle printer ----
    _start_periodic_print()


def _apply_no_bounce_material(stage, robot_prim_path):
    """Create a zero-restitution physics material and bind it to every
    collision shape under the robot.  This stops thigh-on-shin bouncing."""

    mat_path = "/World/PhysMat_NoBounce"
    UsdPhysics.MaterialAPI.Apply(stage.DefinePrim(mat_path))
    phys_mat = UsdPhysics.MaterialAPI.Get(stage, mat_path)
    phys_mat.CreateRestitutionAttr(0.0)              # no bounce
    phys_mat.CreateStaticFrictionAttr(0.8)
    phys_mat.CreateDynamicFrictionAttr(0.6)

    # PhysX extras: compliant contacts for softer settling
    try:
        px_mat = PhysxSchema.PhysxMaterialAPI.Apply(stage.GetPrimAtPath(mat_path))
        px_mat.CreateRestitutionCombineModeAttr("min")   # min(a,b) → 0 wins
        px_mat.CreateFrictionCombineModeAttr("average")
        px_mat.CreateCompliantContactDampingAttr(100.0)   # dampen resting contact
    except Exception:
        pass  # older Isaac Sim versions may lack some attrs

    # Bind the material to every collision shape in the robot subtree
    from pxr import UsdShade
    mat_prim = stage.GetPrimAtPath(mat_path)
    count = 0
    for prim in stage.Traverse():
        path_str = str(prim.GetPath())
        if not path_str.startswith(robot_prim_path):
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            binding = UsdShade.MaterialBindingAPI.Apply(prim)
            binding.Bind(
                UsdShade.Material(mat_prim),
                UsdShade.Tokens.weakerThanDescendants,
                "physics",
            )
            count += 1

    print(f"  [No-bounce] Bound zero-restitution material to {count} collision shapes")


def _enable_self_collision(stage, robot_prim_path):
    """Walk the robot subtree and enable self-collision on the articulation root."""
    robot_prim = stage.GetPrimAtPath(robot_prim_path)

    # Try to find the prim that carries ArticulationRootAPI
    target_prim = None
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(robot_prim_path):
            continue
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            target_prim = prim
            break

    if target_prim is None:
        print("  [Self-collision] No ArticulationRootAPI found; applying to robot root.")
        target_prim = robot_prim
        UsdPhysics.ArticulationRootAPI.Apply(target_prim)

    # Apply PhysxArticulationAPI and enable self-collisions
    physx_art = PhysxSchema.PhysxArticulationAPI.Apply(target_prim)
    try:
        # Isaac Sim 4.x / PhysX 5 attribute name
        physx_art.CreateEnabledSelfCollisionsAttr().Set(True)
        print(f"  [Self-collision] Enabled on {target_prim.GetPath()}")
    except AttributeError:
        # Fallback: set the raw USD attribute directly
        try:
            attr = target_prim.GetAttribute("physxArticulation:enabledSelfCollisions")
            if not attr.IsValid():
                attr = target_prim.CreateAttribute(
                    "physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool
                )
            attr.Set(True)
            print(f"  [Self-collision] Enabled via raw attr on {target_prim.GetPath()}")
        except Exception as e2:
            print(f"  [Self-collision] WARNING: Could not enable — {e2}")
            print(f"  You can enable it manually: select the ArticulationRoot prim →")
            print(f"  Property panel → PhysxArticulationAPI → enabledSelfCollisions = True")


# ============================================================
# LAUNCH
# ============================================================
try:
    # Preferred: schedules on Kit's own loop; works in Script Editor and standalone.
    from omni.kit.async_engine import run_coroutine
    run_coroutine(setup_robot())
except ImportError:
    asyncio.ensure_future(setup_robot())

if _STANDALONE:
    # Keep the app alive; setup_robot() runs during the first updates,
    # then the user drives PLAY/sliders in the GUI until the window closes.
    while simulation_app.is_running():
        simulation_app.update()
    simulation_app.close()