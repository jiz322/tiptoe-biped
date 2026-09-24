"""
alpha_tester_p6d.py — Test spawn height at every curriculum alpha

Single alpha slider (0.0 -> 1.0) that interpolates between standing
and kneeling poses exactly like the training code in rl_env_p6d.py.

Height strategy:
  FK is accurate when FEET are the ground contact (alpha 0 - ~0.8).
  FK is wrong when SHINS become the ground contact (alpha ~0.9 - 1.0)
  because it still tracks foot tips which are now folded up in the air.

  Fix: use FK everywhere, minus a smooth correction = overshoot * a^3.
  a^3 is negligible below 0.5 (<1mm) and reaches full strength at 1.0.

Workflow:
    1. Press Run in the Script Editor (spawns at alpha=0 = standing)
    2. Drag slider to desired alpha
    3. Press PLAY - robot should sit at ground level, no drop or extrusion
    4. To test another alpha: STOP, drag slider, PLAY again
    5. During PLAY: drag slider to teleport live (DC interface)
"""

import numpy as np
import math
import asyncio
from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema, Sdf, PhysicsSchemaTools
import omni.usd
import omni.kit.app
import omni.kit.commands
import omni.ui as ui

# ============================================================
# USER CONFIG
# ============================================================

USD_PATH = r"<repo>\urdf_onshape_export\bigthigh-urdf\robot\robot.usd"

# Standing pose (URDF default)
STAND_ANGLES = {
    "hip_roll1":  0.0,
    "hip_pitch1": 0.05,
    "knee1":      0.0,
    "hip_roll2":  0.0,
    "hip_pitch2": -0.05,
    "knee2":      0.0,
}

# Kneeling pose (measured in Isaac Sim)
KNEEL_ANGLES = {
    "hip_roll1":  0.0,
    "hip_pitch1": 3.09,
    "knee1":     -3.06,
    "hip_roll2":  0.0,
    "hip_pitch2": -3.09,
    "knee2":      3.06,
}

JOINT_NAMES = ["hip_roll1", "hip_pitch1", "knee1",
               "hip_roll2", "hip_pitch2", "knee2"]

# Heights
SPAWN_HEIGHT = 0.42    # standing height (from training cfg)
KNEEL_HEIGHT = 0.22    # empirical kneeling height (shin contact)

# Quaternions (w, x, y, z)
STAND_QUAT = (1.0, 0.0, 0.0, 0.0)
KNEEL_QUAT = (0.7071, 0.0, -0.7071, 0.0)

# PD drive gains
JOINT_STIFFNESS = 40.0
JOINT_DAMPING   = 5.0

# ============================================================
# MATH HELPERS
# ============================================================

def _slerp_np(q0, q1, t):
    """Spherical linear interpolation between two quaternions (w,x,y,z)."""
    q0 = np.array(q0, dtype=np.float64)
    q1 = np.array(q1, dtype=np.float64)
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    if abs(sin_theta) < 1e-6:
        return (1.0 - t) * q0 + t * q1
    s0 = np.sin((1.0 - t) * theta) / sin_theta
    s1 = np.sin(t * theta) / sin_theta
    result = s0 * q0 + s1 * q1
    return result / np.linalg.norm(result)


def _interp_angles(alpha):
    """Interpolate joint angles: stand*(1-a) + kneel*a."""
    angles = {}
    for name in JOINT_NAMES:
        angles[name] = STAND_ANGLES[name] * (1.0 - alpha) + KNEEL_ANGLES[name] * alpha
    return angles


# ============================================================
# FORWARD KINEMATICS
# ============================================================

def _fixed_tf(xyz, rpy):
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
    return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]])

def _joint_tf(fixed, angle):
    return fixed @ _Rz(angle)

_F_hr1 = _fixed_tf((-0.070392, 0.0800218, -0.0315), (1.5708, 0.0, 1.5708))
_F_hp1 = _fixed_tf((-0.0415, 0.0, -0.071), (0.0, 1.5708, 0.0))
_F_kn1 = _fixed_tf((-0.0592463, -0.172173, 0.0969), (0.0, 0.0, 1.72565))
_T_nail1 = np.array([
    [1,0,0,-0.138448],[0,1,0,0.0216628],[0,0,1,-0.01835],[0,0,0,1.0]])
_F_hr2 = _fixed_tf((-0.070392, -0.0799782, -0.0315), (-3.14159, -1.5708, 0.0))
_F_hp2 = _fixed_tf((0.0, -0.0415, -0.071), (-1.5708, 0.908015, 0.0))
_F_kn2 = _fixed_tf((-0.154185, 0.107117, 0.0602), (0.0, 0.0, 0.662781))
_T_nail2 = np.array([
    [1,0,0,-4.95155e-05],[0,1,0,0.140191],[0,0,1,0.01835],[0,0,0,1.0]])
_FK_TARGET_FOOT_Z = 0.0716

def _quat_to_rot4(qw, qx, qy, qz):
    R = np.zeros((4, 4))
    R[0,0]=1-2*(qy*qy+qz*qz); R[0,1]=2*(qx*qy-qw*qz); R[0,2]=2*(qx*qz+qw*qy)
    R[1,0]=2*(qx*qy+qw*qz);   R[1,1]=1-2*(qx*qx+qz*qz); R[1,2]=2*(qy*qz-qw*qx)
    R[2,0]=2*(qx*qz-qw*qy);   R[2,1]=2*(qy*qz+qw*qx);   R[2,2]=1-2*(qx*qx+qy*qy)
    R[3,3]=1.0
    return R

def compute_fk_height(joint_angles, quat):
    """FK height from foot tips."""
    qw, qx, qy, qz = quat
    T_base = _quat_to_rot4(qw, qx, qy, qz)
    T = T_base @ _joint_tf(_F_hr1, joint_angles["hip_roll1"]) \
        @ _joint_tf(_F_hp1, joint_angles["hip_pitch1"]) \
        @ _joint_tf(_F_kn1, joint_angles["knee1"]) @ _T_nail1
    foot1_z = T[2, 3]
    T = T_base @ _joint_tf(_F_hr2, joint_angles["hip_roll2"]) \
        @ _joint_tf(_F_hp2, joint_angles["hip_pitch2"]) \
        @ _joint_tf(_F_kn2, joint_angles["knee2"]) @ _T_nail2
    foot2_z = T[2, 3]
    return _FK_TARGET_FOOT_Z - min(foot1_z, foot2_z)


# -- Precompute FK overshoot at full kneel --
# FK tracks foot tips which are above the shins in the kneel pose,
# so it returns a height that is too high. The overshoot is the
# difference between FK answer and the empirical correct height.
_FK_AT_KNEEL = compute_fk_height(
    KNEEL_ANGLES,
    tuple(_slerp_np(STAND_QUAT, KNEEL_QUAT, 1.0)),
)
_FK_OVERSHOOT = max(0.0, _FK_AT_KNEEL - KNEEL_HEIGHT)


def compute_spawn_height(alpha, joint_angles, quat):
    """
    P6d height: FK minus smooth correction.

    FK is correct when feet are the ground contact (low alpha).
    FK overshoots when shins become the contact (high alpha) because
    it still tracks foot tips that are now folded up.

    Correction = overshoot x alpha^3
      alpha=0.0 -> 0          (FK untouched)
      alpha=0.3 -> 0.027*ov   (sub-mm, negligible)
      alpha=0.5 -> 0.125*ov   (~0.4mm if ov=3cm)
      alpha=0.7 -> 0.343*ov   (~1.0cm)
      alpha=0.9 -> 0.729*ov   (~2.2cm)
      alpha=1.0 -> 1.000*ov   (full correction -> KNEEL_HEIGHT)
    """
    fk_h = compute_fk_height(joint_angles, quat)
    correction = _FK_OVERSHOOT * (alpha ** 3)
    final_h = fk_h - correction
    return final_h, fk_h, correction


# ============================================================
# GLOBALS
# ============================================================

_joint_prims = {}
_window = None
_print_sub = None
_elapsed = 0.0
_current_alpha = 0.0
_robot_prim_path = "/World/Robot"
_info_label = None
_height_label = None


# ============================================================
# TELEPORT ROBOT TO ALPHA POSE
# ============================================================

def _apply_alpha(alpha):
    """Teleport the robot to the interpolated pose at the given alpha."""
    global _current_alpha
    _current_alpha = alpha

    angles = _interp_angles(alpha)
    quat = _slerp_np(STAND_QUAT, KNEEL_QUAT, alpha)
    spawn_h, fk_h, correction = compute_spawn_height(alpha, angles, tuple(quat))

    qw, qx, qy, qz = quat

    # -- Try Dynamic Control teleport (works during PLAY) --
    teleported = False
    try:
        from omni.isaac.dynamic_control import _dynamic_control
        dc = _dynamic_control.acquire_dynamic_control_interface()

        art = dc.get_articulation(_robot_prim_path)
        if art == _dynamic_control.INVALID_HANDLE:
            for suffix in ["/base_link", "/torso2hip", ""]:
                art = dc.get_articulation(_robot_prim_path + suffix)
                if art != _dynamic_control.INVALID_HANDLE:
                    break

        if art != _dynamic_control.INVALID_HANDLE:
            root = dc.get_articulation_root_body(art)
            if root != _dynamic_control.INVALID_HANDLE:
                from omni.isaac.dynamic_control import _dynamic_control as dc_mod
                tf = dc_mod.Transform()
                tf.p = dc_mod.float3(0.0, 0.0, float(spawn_h))
                tf.r = dc_mod.float4(float(qx), float(qy), float(qz), float(qw))
                dc.set_rigid_body_pose(root, tf)
                zero_vel = dc_mod.float3(0.0, 0.0, 0.0)
                dc.set_rigid_body_linear_velocity(root, zero_vel)
                dc.set_rigid_body_angular_velocity(root, zero_vel)

            for name in JOINT_NAMES:
                dof = dc.find_articulation_dof(art, name)
                if dof != _dynamic_control.INVALID_HANDLE:
                    dc.set_dof_position(dof, angles[name])
                    dc.set_dof_velocity(dof, 0.0)
                    props = dc.get_dof_properties(dof)
                    props.stiffness = JOINT_STIFFNESS
                    props.damping = JOINT_DAMPING
                    dc.set_dof_properties(dof, props)
                    dc.set_dof_position_target(dof, angles[name])

            teleported = True
    except (ImportError, Exception) as e:
        pass

    # -- Fallback: update USD transforms (works before PLAY) --
    if not teleported:
        stage = omni.usd.get_context().get_stage()
        robot_prim = stage.GetPrimAtPath(_robot_prim_path)
        if robot_prim.IsValid():
            xform = UsdGeom.Xformable(robot_prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Vec3d(0.0, 0.0, float(spawn_h))
            )
            spawn_gf_quat = Gf.Quatd(float(qw), Gf.Vec3d(float(qx), float(qy), float(qz)))
            xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(spawn_gf_quat)

        for name in JOINT_NAMES:
            if name in _joint_prims:
                prim = _joint_prims[name]
                target_deg = np.degrees(angles[name])
                drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                if drive:
                    drive.GetTargetPositionAttr().Set(target_deg)
                joint_state = PhysxSchema.JointStateAPI.Get(prim, "angular")
                if joint_state:
                    joint_state.GetPositionAttr().Set(target_deg)

    # -- Update info labels --
    if _height_label is not None:
        _height_label.text = (
            f"Height: {spawn_h:.4f} m  "
            f"[FK={fk_h:.4f}  correction={correction:.4f}  "
            f"overshoot={_FK_OVERSHOOT:.4f}]"
        )
    if _info_label is not None:
        _info_label.text = (
            f"hp1={angles['hip_pitch1']:+.3f}  kn1={angles['knee1']:+.3f}  "
            f"hp2={angles['hip_pitch2']:+.3f}  kn2={angles['knee2']:+.3f}"
        )

    print(f"[ALPHA={alpha:.3f}]  height={spawn_h:.4f}m  "
          f"FK={fk_h:.4f}m  correction={correction:.4f}m  "
          f"hp1={angles['hip_pitch1']:+.3f} kn1={angles['knee1']:+.3f}")


# ============================================================
# READ & PRINT ACTUAL ANGLES
# ============================================================

def _read_current_angles():
    angles = {}
    try:
        from omni.isaac.dynamic_control import _dynamic_control
        dc = _dynamic_control.acquire_dynamic_control_interface()
        art = dc.get_articulation(_robot_prim_path)
        if art == _dynamic_control.INVALID_HANDLE:
            for suffix in ["/base_link", "/torso2hip", ""]:
                art = dc.get_articulation(_robot_prim_path + suffix)
                if art != _dynamic_control.INVALID_HANDLE:
                    break
        if art != _dynamic_control.INVALID_HANDLE:
            for name in JOINT_NAMES:
                dof = dc.find_articulation_dof(art, name)
                if dof != _dynamic_control.INVALID_HANDLE:
                    angles[name] = dc.get_dof_position(dof)
                else:
                    angles[name] = 0.0
            return angles
    except (ImportError, Exception):
        pass

    for name in JOINT_NAMES:
        if name in _joint_prims:
            prim = _joint_prims[name]
            joint_state = PhysxSchema.JointStateAPI.Get(prim, "angular")
            if joint_state:
                deg = joint_state.GetPositionAttr().Get()
                if deg is not None:
                    angles[name] = np.radians(deg)
                    continue
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if drive:
                deg = drive.GetTargetPositionAttr().Get()
                angles[name] = np.radians(deg) if deg is not None else 0.0
            else:
                angles[name] = 0.0
        else:
            angles[name] = 0.0
    return angles


def _print_angles():
    angles = _read_current_angles()
    root_h = "?"
    try:
        from omni.isaac.dynamic_control import _dynamic_control
        dc = _dynamic_control.acquire_dynamic_control_interface()
        art = dc.get_articulation(_robot_prim_path)
        if art == _dynamic_control.INVALID_HANDLE:
            for suffix in ["/base_link", "/torso2hip", ""]:
                art = dc.get_articulation(_robot_prim_path + suffix)
                if art != _dynamic_control.INVALID_HANDLE:
                    break
        if art != _dynamic_control.INVALID_HANDLE:
            root = dc.get_articulation_root_body(art)
            if root != _dynamic_control.INVALID_HANDLE:
                tf = dc.get_rigid_body_pose(root)
                root_h = f"{tf.p.z:.4f}"
    except Exception:
        pass

    print(f"\n--- alpha={_current_alpha:.3f}  root_z={root_h} ---")
    for name, rad in angles.items():
        print(f"  {name:>12s}: {rad:>8.4f} rad  ({np.degrees(rad):>8.2f} deg)")
    print("--------------------------------------------")


# ============================================================
# UI
# ============================================================

def _build_ui():
    global _window, _info_label, _height_label

    if _window is not None:
        _window.destroy()
        _window = None

    _window = ui.Window("Alpha Tester (p6d)", width=580, height=300)

    with _window.frame:
        with ui.VStack(spacing=5):
            ui.Label(
                "Drag alpha to test spawn pose. STOP -> drag -> PLAY to verify height.",
                height=25, style={"font_size": 14},
            )
            ui.Separator(height=5)

            # -- Alpha slider --
            with ui.HStack(height=30, spacing=5):
                ui.Label("alpha", width=60, style={"font_size": 14})
                alpha_slider = ui.FloatSlider(min=0.0, max=1.0, step=0.01)
                alpha_slider.model.as_float = 0.0
                alpha_value_label = ui.Label("0.000", width=50)

            def _on_alpha_changed(model):
                val = model.as_float
                alpha_value_label.text = f"{val:.3f}"
                _apply_alpha(val)

            alpha_slider.model.add_value_changed_fn(_on_alpha_changed)

            ui.Separator(height=5)

            # -- Quick-jump buttons --
            with ui.HStack(height=28, spacing=4):
                for a in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                    def _make_jump(val, slider_model=alpha_slider.model):
                        def _fn():
                            slider_model.as_float = val
                        return _fn
                    ui.Button(
                        f"{a:.1f}",
                        clicked_fn=_make_jump(a),
                        width=42,
                        style={"font_size": 11},
                    )

            ui.Separator(height=5)

            # -- Info labels --
            _height_label = ui.Label(
                "Height: ---",
                height=22, style={"font_size": 12},
            )
            _info_label = ui.Label(
                "Joint angles: ---",
                height=22, style={"font_size": 12},
            )

            ui.Separator(height=8)

            with ui.HStack(height=28, spacing=10):
                ui.Button(
                    "Print Actual Angles",
                    clicked_fn=_print_angles,
                    width=160, style={"font_size": 13},
                )

            ui.Separator(height=5)
            ui.Label(
                f"FK overshoot at kneel: {_FK_OVERSHOOT:.4f} m  "
                f"(FK@kneel={_FK_AT_KNEEL:.4f}, empirical={KNEEL_HEIGHT:.4f})",
                style={"font_size": 11, "color": 0xFFAAAA44},
            )
            ui.Label(
                "Correction = overshoot x alpha^3  (negligible <0.5, full at 1.0)",
                style={"font_size": 11, "color": 0xFF888888},
            )


# ============================================================
# PERIODIC PRINTER
# ============================================================

def _on_update(e):
    global _elapsed
    _elapsed += e.payload["dt"]
    if _elapsed >= 1.0:
        _elapsed = 0.0
        _print_angles()

def _start_periodic_print():
    global _print_sub
    if _print_sub is not None:
        _print_sub = None
    _print_sub = (
        omni.kit.app.get_app()
        .get_update_event_stream()
        .create_subscription_to_pop(_on_update, name="alpha_tester_printer")
    )


# ============================================================
# SCENE SETUP
# ============================================================

async def setup_robot():
    global _joint_prims

    stage = omni.usd.get_context().get_stage()

    if stage.GetPrimAtPath("/World"):
        omni.kit.commands.execute("DeletePrims", paths=["/World"])
        await omni.kit.app.get_app().next_update_async()

    UsdGeom.Xform.Define(stage, "/World")

    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)

    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/World/PhysicsScene"))
    physx_scene.CreateSolverTypeAttr("TGS")
    physx_scene.CreateMinPositionIterationCountAttr(8)
    physx_scene.CreateMinVelocityIterationCountAttr(4)

    PhysicsSchemaTools.addGroundPlane(
        stage, "/World/GroundPlane", "Z", 10.0,
        Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.5, 0.5, 0.5),
    )

    robot_prim = stage.DefinePrim(_robot_prim_path)
    robot_prim.GetReferences().AddReference(USD_PATH)

    await omni.kit.app.get_app().next_update_async()
    await omni.kit.app.get_app().next_update_async()

    _apply_no_bounce_material(stage, _robot_prim_path)

    # -- Initial spawn at alpha=0 (standing) --
    init_angles = _interp_angles(0.0)
    init_h, _, _ = compute_spawn_height(0.0, init_angles, STAND_QUAT)

    xform = UsdGeom.Xformable(robot_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(0.0, 0.0, float(init_h))
    )
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(1.0, 0.0, 0.0, 0.0)
    )

    # -- Configure joints --
    print(f"\n{'=' * 60}")
    print(f"  ALPHA TESTER (p6d) -- FK + alpha^3 correction")
    print(f"{'=' * 60}")
    print(f"  FK at full kneel: {_FK_AT_KNEEL:.5f} m")
    print(f"  Empirical kneel:  {KNEEL_HEIGHT:.5f} m")
    print(f"  Overshoot:        {_FK_OVERSHOOT:.5f} m ({_FK_OVERSHOOT*100:.1f} cm)")
    print(f"  Correction curve: overshoot x alpha^3")

    _joint_prims = {}
    found_joints = []

    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint_name = prim.GetName()
            found_joints.append(joint_name)

            if joint_name in JOINT_NAMES:
                _joint_prims[joint_name] = prim
                target_rad = init_angles[joint_name]
                target_deg = np.degrees(target_rad)

                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.CreateTypeAttr("force")
                drive.CreateStiffnessAttr(JOINT_STIFFNESS)
                drive.CreateDampingAttr(JOINT_DAMPING)
                drive.CreateTargetPositionAttr(target_deg)

                joint_state = PhysxSchema.JointStateAPI.Apply(prim, "angular")
                joint_state.CreatePositionAttr().Set(target_deg)
                joint_state.CreateVelocityAttr().Set(0.0)

    missing = set(JOINT_NAMES) - set(found_joints)
    if missing:
        print(f"\n*** WARNING: joints not found: {missing}")

    # -- Height table at all alphas --
    print(f"\n  {'alpha':>6s}  {'FK':>8s}  {'correct.':>8s}  {'final':>8s}  {'vs old min()':>12s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*12}")
    for test_alpha in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
        ta = _interp_angles(test_alpha)
        tq = tuple(_slerp_np(STAND_QUAT, KNEEL_QUAT, test_alpha))
        h, fk, corr = compute_spawn_height(test_alpha, ta, tq)
        # What min(FK, empirical) would have given (the old broken method)
        old_emp = SPAWN_HEIGHT * (1.0 - test_alpha) + KNEEL_HEIGHT * test_alpha
        old_h = min(fk, old_emp)
        diff_cm = (h - old_h) * 100
        print(f"  {test_alpha:6.2f}  {fk:8.4f}  {corr:8.4f}  {h:8.4f}  {diff_cm:+8.1f} cm")

    print(f"\n  Configured: {len(_joint_prims)} joints")
    print(f"{'=' * 60}")
    print(f"\n>>> Set alpha with slider, then PLAY to verify no drop/extrusion.")
    print(f">>> Or: PLAY first, then drag to teleport live.\n")

    _enable_self_collision(stage, _robot_prim_path)
    _build_ui()
    _start_periodic_print()


# ============================================================
# MATERIAL & SELF-COLLISION
# ============================================================

def _apply_no_bounce_material(stage, robot_prim_path):
    mat_path = "/World/PhysMat_NoBounce"
    UsdPhysics.MaterialAPI.Apply(stage.DefinePrim(mat_path))
    phys_mat = UsdPhysics.MaterialAPI.Get(stage, mat_path)
    phys_mat.CreateRestitutionAttr(0.0)
    phys_mat.CreateStaticFrictionAttr(0.8)
    phys_mat.CreateDynamicFrictionAttr(0.6)
    try:
        px_mat = PhysxSchema.PhysxMaterialAPI.Apply(stage.GetPrimAtPath(mat_path))
        px_mat.CreateRestitutionCombineModeAttr("min")
        px_mat.CreateFrictionCombineModeAttr("average")
        px_mat.CreateCompliantContactDampingAttr(100.0)
    except Exception:
        pass
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
    print(f"  [No-bounce] Bound to {count} collision shapes")


def _enable_self_collision(stage, robot_prim_path):
    target_prim = None
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(robot_prim_path):
            continue
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            target_prim = prim
            break
    if target_prim is None:
        robot_prim = stage.GetPrimAtPath(robot_prim_path)
        target_prim = robot_prim
        UsdPhysics.ArticulationRootAPI.Apply(target_prim)
    physx_art = PhysxSchema.PhysxArticulationAPI.Apply(target_prim)
    try:
        physx_art.CreateEnabledSelfCollisionsAttr().Set(True)
        print(f"  [Self-collision] Enabled on {target_prim.GetPath()}")
    except AttributeError:
        try:
            attr = target_prim.GetAttribute("physxArticulation:enabledSelfCollisions")
            if not attr.IsValid():
                attr = target_prim.CreateAttribute(
                    "physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool
                )
            attr.Set(True)
        except Exception as e2:
            print(f"  [Self-collision] WARNING: {e2}")


# ============================================================
# LAUNCH
# ============================================================
asyncio.ensure_future(setup_robot())
