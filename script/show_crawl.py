"""
show_crawl.py — lay/crawl pose finder.

Spawns the robot and lets you tune hip-pitch and knee angles via sliders
to find a stable laying pose. Uses small PD gains so joints settle
naturally instead of fighting gravity.

Two ways to run:

A) Standalone (recommended):
       powershell -File script\run_show_crawl.ps1
   The wrapper sets the Isaac Sim env vars (ISAAC_PATH etc.) and uses the
   env_isaaclab python. Plain `python show_crawl.py` will NOT work — pxr/omni
   modules only exist once SimulationApp is up, which needs those env vars.

B) Inside Isaac Sim's Script Editor (original workflow):
   paste/run, then press PLAY.

Either way:
    1. Press PLAY in the viewport — robot will flop down and settle
    2. Drag the four sliders to find the lay configuration you want
    3. Joint angles and IMU frame state print to console every second
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

# Canonical asset = what training loads (rl_env_cfg_p6.ROBOT_USD_PATH).
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
USD_PATH = os.path.join(_REPO, "assets", "tiptoe_v27", "robot", "robot.usd")

# IMU *body* prim (URDF link "imu" = rl_env_cfg_p6.imu_body_name, what
# training reads). NOT "/World/Robot/joints/imu_frame": that is the fixed
# JOINT connecting torso2hip->imu; a joint prim has no xform of its own, so
# reading it returns the static /World/Robot spawn transform (until 09-03
# this script printed the spawn quat + FK spawn height and called it the IMU).
IMU_LINK_PATH = "/World/Robot/imu"

# Deploy-side trim (control/config.py IMU_PG_OFFSET): hardware DISPLAYS
# pg_raw - offset. Imported so the printout can show both raw and
# hardware-displayed-equivalent; falls back to [0,0] if the import fails.
_CONTROL_CONFIG = os.path.join(_REPO, "control", "config.py")
try:
    # Load by path: Isaac's python already has an unrelated module named
    # `config` in sys.modules, so `from config import ...` picks that up.
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("bigthigh_control_config", _CONTROL_CONFIG)
    _cfg_mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_cfg_mod)
    _IMU_PG_OFFSET = list(_cfg_mod.IMU_PG_OFFSET)
except Exception as _e:
    print(f"[IMU] control/config.py not loadable ({_e}); trim shown as 0")
    _IMU_PG_OFFSET = [0.0, 0.0]

# Initial joint angles in radians — start neutral and tune via sliders
JOINT_ANGLES = {
    "hip_roll1":  0.0,
    "hip_pitch1": 0.0,
    "knee1":      0.0,
    "hip_roll2":  0.0,
    "hip_pitch2": 0.0,
    "knee2":      0.0,
}

# Small PD gains — let gravity do most of the work
JOINT_STIFFNESS = 5.0
JOINT_DAMPING   = 1.0

# "Crawl mode" button: joints go limp + ground becomes near-frictionless,
# so the robot can be dragged/settled into a true crawl.
CRAWL_STIFFNESS = 0.0
CRAWL_DAMPING   = 0.1    # tiny damping so limbs don't flail
CRAWL_FRICTION  = 0.02

# Normal-mode friction (also the restore values when crawl mode toggles off)
NORMAL_STATIC_FRICTION  = 0.8
NORMAL_DYNAMIC_FRICTION = 0.6

ROBOT_MAT_PATH  = "/World/PhysMat_NoBounce"
GROUND_MAT_PATH = "/World/PhysMat_Ground"

# Crawl-mode push: a steady downward force, ramped in over PUSH_RAMP_S, on
# both thighs and at both foot tips — low friction alone leaves the robot
# inert, this actually settles it into the crawl.
PUSH_THIGH_N = 5.0   # per thigh, world -Z
PUSH_FOOT_N  = 3.0   # per foot tip, world -Z
PUSH_RAMP_S  = 2.0   # seconds from 0 to full force

# Link names from ballfeet robot.urdf (leg1 / leg2):
THIGH_LINKS = ("thigh_outer", "stand_steel_00___fillet1")
# shank link -> foot-tip offset in link frame (= FK _T_nail1/_T_nail2 above)
FOOT_PUSH_POINTS = {
    "shin_inner":     (-0.138448, 0.0216628, -0.01835),
    "nail___fillet1": (-4.95155e-05, 0.140191, 0.01835),
}

# Orientation quaternion (Y-up URDF → Z-up Isaac)
SPAWN_QUAT = Gf.Quatd(0.7071, 0.0, -0.7071, 0.0)

# Small upward margin so the robot doesn't intersect ground at frame 0
SPAWN_HEIGHT_MARGIN = 0.02   # meters above FK-computed height

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

# Matches rl_env_p6: ball foot is ~14 mm taller than the old nail foot.
_FK_TARGET_FOOT_Z = 0.0716 + (0.040 - 0.02598076)  # 0.08561924


def _quat_to_rot4(qw, qx, qy, qz):
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

    T = T_base @ _joint_tf(_F_hr1, hr1) @ _joint_tf(_F_hp1, hp1) \
        @ _joint_tf(_F_kn1, kn1) @ _T_nail1
    foot1_z = T[2, 3]

    T = T_base @ _joint_tf(_F_hr2, hr2) @ _joint_tf(_F_hp2, hp2) \
        @ _joint_tf(_F_kn2, kn2) @ _T_nail2
    foot2_z = T[2, 3]

    return _FK_TARGET_FOOT_Z - min(foot1_z, foot2_z)


# ============================================================
# GLOBALS
# ============================================================

_joint_prims = {}       # name -> USD prim
_slider_models = {}     # name -> ui slider model
_window = None
_print_sub = None
_elapsed = 0.0
_crawl_mode = False
_crawl_btn = None
_push_apis = []        # list of (PhysxForceAPI, magnitude_N)
_push_ramp_t = None    # ramp clock while forces ramp in, else None


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
        joint_state = PhysxSchema.JointStateAPI.Get(prim, "angular")
        if joint_state:
            joint_state.GetPositionAttr().Set(np.degrees(value_rad))


def _read_current_angles():
    """Read actual joint positions from the running physics simulation."""
    angles = {}

    try:
        from omni.isaac.dynamic_control import _dynamic_control
        dc = _dynamic_control.acquire_dynamic_control_interface()

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

    # Fallback: read from USD JointStateAPI
    for name in JOINT_ANGLES:
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


def _read_imu_state():
    """World pose (pos, quat wxyz) of the IMU rigid body.

    Prefers dynamic_control (live physics state); falls back to the USD
    xform, which is only current if physics write-back to USD is on. Both
    paths must target a RIGID BODY prim — a joint prim would silently
    return the static articulation-root transform.
    """
    from pxr import UsdGeom, Usd, UsdPhysics
    import omni.usd

    global IMU_LINK_PATH
    stage = omni.usd.get_context().get_stage()

    prim = stage.GetPrimAtPath(IMU_LINK_PATH)
    if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        for p in stage.TraverseAll():
            if p.GetName() == "imu" and p.HasAPI(UsdPhysics.RigidBodyAPI):
                prim = p
                IMU_LINK_PATH = str(p.GetPath())
                print(f"[IMU] resolved IMU body to {IMU_LINK_PATH}")
                break

    if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        return None, None

    try:
        from omni.isaac.dynamic_control import _dynamic_control
        dc = _dynamic_control.acquire_dynamic_control_interface()
        rb = dc.get_rigid_body(IMU_LINK_PATH)
        if rb != _dynamic_control.INVALID_HANDLE:
            pose = dc.get_rigid_body_pose(rb)
            p, r = pose.p, pose.r  # r is (x, y, z, w)
            return (p.x, p.y, p.z), (r.w, r.x, r.y, r.z)
    except (ImportError, Exception):
        pass

    try:
        time = Usd.TimeCode.Default()
        world_transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time)
        pos = world_transform.ExtractTranslation()
        rot_quat = world_transform.ExtractRotation().GetQuat()
        return (
            (pos[0], pos[1], pos[2]),
            (rot_quat.GetReal(),
             rot_quat.GetImaginary()[0],
             rot_quat.GetImaginary()[1],
             rot_quat.GetImaginary()[2])
        )
    except Exception as e:
        print(f"Error computing pose for {IMU_LINK_PATH}: {e}")
        return None, None


def _projected_gravity(quat_wxyz):
    """Gravity (world -Z) expressed in the IMU body frame — identical to
    rl_env_p6._get_observations' quat_apply_inverse(imu_quat, gravity) and
    to control/imu_adapter.projected_gravity_raw()."""
    R = _quat_to_rot4(*quat_wxyz)[:3, :3]
    return R.T @ np.array([0.0, 0.0, -1.0])


def _print_state():
    """Print actual joint angles and IMU frame read to the console."""
    angles = _read_current_angles()

    print("\n--- Robot State (actual) ---")
    print("Joint Angles:")
    for name, rad in angles.items():
        print(f"  {name:>12s}: {rad:>8.4f} rad  ({np.degrees(rad):>8.2f} deg)")

    pos, rot = _read_imu_state()
    if pos is not None:
        pg = _projected_gravity(rot)
        pg_disp = pg - np.array([_IMU_PG_OFFSET[0], _IMU_PG_OFFSET[1], 0.0])
        pitch_deg = math.degrees(math.asin(max(-1.0, min(1.0, pg[0]))))
        roll_deg = math.degrees(math.asin(max(-1.0, min(1.0, pg[1]))))
        print(f"\nIMU body ('{IMU_LINK_PATH}'):")
        print(f"  Pos (x,y,z)  : [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        print(f"  Rot (w,x,y,z): [{rot[0]:.4f}, {rot[1]:.4f}, {rot[2]:.4f}, {rot[3]:.4f}]")
        print(f"  proj_grav RAW: ({pg[0]:+.3f},{pg[1]:+.3f},{pg[2]:+.3f})"
              f"   <- compare to hardware RAW (crawl_imu_log.py / "
              f"imu.projected_gravity_raw)")
        print(f"  proj_grav as balance_log DISPLAYS it (raw - IMU_PG_OFFSET "
              f"{_IMU_PG_OFFSET}): ({pg_disp[0]:+.3f},{pg_disp[1]:+.3f},{pg_disp[2]:+.3f})")
        print(f"  tilt: pitch {pitch_deg:+.2f} deg (asin pg_x), roll {roll_deg:+.2f} deg (asin pg_y)")
    else:
        print(f"\nIMU body ('{IMU_LINK_PATH}'): [Data could not be read — "
              f"no rigid body named 'imu' found]")

    print("----------------------------")


def _set_material_friction(mat_path, static_f, dynamic_f, combine_mode):
    """Set friction on a physics material prim (USD + PhysX combine mode)."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(mat_path)
    if not prim.IsValid():
        return
    mat = UsdPhysics.MaterialAPI.Get(stage, mat_path)
    if mat:
        mat.CreateStaticFrictionAttr().Set(static_f)
        mat.CreateDynamicFrictionAttr().Set(dynamic_f)
    try:
        px_mat = PhysxSchema.PhysxMaterialAPI.Apply(prim)
        px_mat.CreateFrictionCombineModeAttr().Set(combine_mode)
    except Exception:
        pass


def _setup_push_forces(stage, robot_prim_path):
    """Attach (disabled) downward force emitters to thighs and foot tips."""
    global _push_apis
    _push_apis = []
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(robot_prim_path):
            continue
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        name = prim.GetName()
        if name in THIGH_LINKS:
            api = PhysxSchema.PhysxForceAPI.Apply(prim)
            _push_apis.append((api, PUSH_THIGH_N))
            print(f"  [Push] thigh force point: {prim.GetPath()}")
        elif name in FOOT_PUSH_POINTS:
            # child Xform at the foot tip so the force also torques the
            # shank foot-down instead of just translating it
            tip = UsdGeom.Xform.Define(
                stage, str(prim.GetPath()) + "/crawl_push_tip")
            tip.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Vec3d(*FOOT_PUSH_POINTS[name]))
            api = PhysxSchema.PhysxForceAPI.Apply(tip.GetPrim())
            _push_apis.append((api, PUSH_FOOT_N))
            print(f"  [Push] foot-tip force point: {tip.GetPath()}")

    for api, _ in _push_apis:
        api.CreateModeAttr().Set("force")
        api.CreateWorldFrameEnabledAttr().Set(True)
        api.CreateForceAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        api.CreateForceEnabledAttr().Set(False)

    if not _push_apis:
        print("  [Push] WARNING: no thigh/foot links found — crawl push disabled")


def _set_push_scale(scale):
    """Scale all push forces: 0.0 = off, 1.0 = full configured magnitude."""
    for api, mag in _push_apis:
        api.CreateForceAttr().Set(Gf.Vec3f(0.0, 0.0, -mag * scale))


def _set_crawl_mode(enable):
    """Toggle limp joints + near-zero friction + steady push (crawl) vs normal."""
    global _crawl_mode, _push_ramp_t
    _crawl_mode = enable

    kp = CRAWL_STIFFNESS if enable else JOINT_STIFFNESS
    kd = CRAWL_DAMPING if enable else JOINT_DAMPING
    for prim in _joint_prims.values():
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if drive:
            drive.CreateStiffnessAttr().Set(kp)
            drive.CreateDampingAttr().Set(kd)

    if enable:
        # combine mode "min" so the slippery side wins at every contact
        _set_material_friction(ROBOT_MAT_PATH, CRAWL_FRICTION, CRAWL_FRICTION, "min")
        _set_material_friction(GROUND_MAT_PATH, CRAWL_FRICTION, CRAWL_FRICTION, "min")
    else:
        _set_material_friction(ROBOT_MAT_PATH, NORMAL_STATIC_FRICTION,
                               NORMAL_DYNAMIC_FRICTION, "average")
        _set_material_friction(GROUND_MAT_PATH, NORMAL_STATIC_FRICTION,
                               NORMAL_DYNAMIC_FRICTION, "average")

    # steady push: ramp in on enable, hard-off on disable
    if enable:
        _push_ramp_t = 0.0
        for api, _ in _push_apis:
            api.CreateForceEnabledAttr().Set(True)
        _set_push_scale(0.0)
    else:
        _push_ramp_t = None
        _set_push_scale(0.0)
        for api, _ in _push_apis:
            api.CreateForceEnabledAttr().Set(False)

    if _crawl_btn is not None:
        _crawl_btn.text = ("CRAWL MODE: ON  (limp + slippery — click to restore)"
                           if enable else
                           "Crawl mode: OFF  (click to go limp + slippery)")
    print(f"\n[Crawl mode] {'ON' if enable else 'OFF'}: "
          f"stiffness={kp} damping={kd} "
          f"friction={'%.2f' % (CRAWL_FRICTION if enable else NORMAL_STATIC_FRICTION)} "
          f"push={'ramping to %g/%g N (thigh/foot) over %gs' % (PUSH_THIGH_N, PUSH_FOOT_N, PUSH_RAMP_S) if enable else 'off'}")


def _toggle_crawl_mode():
    _set_crawl_mode(not _crawl_mode)


def _build_ui():
    """Build the slider window for hip-pitch and knee joints."""
    global _window, _crawl_btn

    if _window is not None:
        _window.destroy()
        _window = None

    _window = ui.Window("Lay Pose — Hip Pitch + Knee Tuner", width=520, height=380)

    # Slider configs: (joint_name, display_label, min_rad, max_rad)
    slider_joints = [
        ("hip_pitch1", "hip_pitch1", -3.5, 3.5),
        ("knee1",      "knee1",      -3.5, 3.5),
        ("hip_pitch2", "hip_pitch2", -3.5, 3.5),
        ("knee2",      "knee2",      -3.5, 3.5),
    ]

    with _window.frame:
        with ui.VStack(spacing=5):
            ui.Label(
                "Adjust hip pitch & knee while physics is running (PLAY).",
                height=25, style={"font_size": 14},
            )
            ui.Label(
                f"PD gains: stiffness={JOINT_STIFFNESS}, damping={JOINT_DAMPING}",
                height=20, style={"font_size": 12, "color": 0xFF888888},
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
            with ui.HStack(height=36, spacing=10):
                _crawl_btn = ui.Button(
                    "Crawl mode: OFF  (click to go limp + slippery)",
                    clicked_fn=_toggle_crawl_mode,
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
    global _elapsed, _push_ramp_t
    dt = e.payload["dt"]

    # ramp the crawl push in gently
    if _crawl_mode and _push_ramp_t is not None:
        _push_ramp_t += dt
        scale = min(_push_ramp_t / PUSH_RAMP_S, 1.0)
        _set_push_scale(scale)
        if scale >= 1.0:
            _push_ramp_t = None
            print(f"[Push] full force reached: thigh {PUSH_THIGH_N} N, "
                  f"foot {PUSH_FOOT_N} N (each, world -Z)")

    _elapsed += dt
    if _elapsed >= 1.0:
        _elapsed = 0.0
        _print_state()


def _start_periodic_print():
    global _print_sub
    if _print_sub is not None:
        _print_sub = None
    _print_sub = (
        omni.kit.app.get_app()
        .get_update_event_stream()
        .create_subscription_to_pop(_on_update, name="state_printer_lay")
    )


# ============================================================
# SCENE SETUP
# ============================================================

async def setup_robot():
    global _joint_prims, _crawl_mode
    _crawl_mode = False  # fresh scene always starts in normal (grippy) mode

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

    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/World/PhysicsScene"))
    physx_scene.CreateSolverTypeAttr("TGS")
    physx_scene.CreateMinPositionIterationCountAttr(8)
    physx_scene.CreateMinVelocityIterationCountAttr(4)

    # ---- Ground plane ----
    PhysicsSchemaTools.addGroundPlane(
        stage, "/World/GroundPlane", "Z", 10.0,
        Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.5, 0.5, 0.5),
    )

    # ---- Ground physics material (so crawl mode can drop its friction) ----
    from pxr import UsdShade
    UsdPhysics.MaterialAPI.Apply(stage.DefinePrim(GROUND_MAT_PATH))
    ground_mat = UsdPhysics.MaterialAPI.Get(stage, GROUND_MAT_PATH)
    ground_mat.CreateStaticFrictionAttr(NORMAL_STATIC_FRICTION)
    ground_mat.CreateDynamicFrictionAttr(NORMAL_DYNAMIC_FRICTION)
    ground_mat.CreateRestitutionAttr(0.0)
    ground_prim = stage.GetPrimAtPath("/World/GroundPlane")
    UsdShade.MaterialBindingAPI.Apply(ground_prim).Bind(
        UsdShade.Material(stage.GetPrimAtPath(GROUND_MAT_PATH)),
        UsdShade.Tokens.weakerThanDescendants,
        "physics",
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

                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.CreateTypeAttr("force")
                drive.CreateStiffnessAttr(JOINT_STIFFNESS)
                drive.CreateDampingAttr(JOINT_DAMPING)
                drive.CreateTargetPositionAttr(target_deg)

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
    print(f"PD gains:     stiffness={JOINT_STIFFNESS}  damping={JOINT_DAMPING}")
    print("=" * 60)
    print("\n>>> Press PLAY. Robot will flop down — tune sliders to find lay pose.\n")

    # ---- Enable self-collision ----
    _enable_self_collision(stage, robot_prim_path)

    # ---- Attach crawl-mode push force emitters (disabled until toggled) ----
    _setup_push_forces(stage, robot_prim_path)

    # ---- Build slider UI ----
    _build_ui()

    # ---- Start periodic angle printer ----
    _start_periodic_print()


def _apply_no_bounce_material(stage, robot_prim_path):
    """Zero-restitution material on every robot collider."""
    mat_path = ROBOT_MAT_PATH
    UsdPhysics.MaterialAPI.Apply(stage.DefinePrim(mat_path))
    phys_mat = UsdPhysics.MaterialAPI.Get(stage, mat_path)
    phys_mat.CreateRestitutionAttr(0.0)
    phys_mat.CreateStaticFrictionAttr(NORMAL_STATIC_FRICTION)
    phys_mat.CreateDynamicFrictionAttr(NORMAL_DYNAMIC_FRICTION)

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

    print(f"  [No-bounce] Bound zero-restitution material to {count} collision shapes")


def _enable_self_collision(stage, robot_prim_path):
    """Enable self-collision on the articulation root."""
    robot_prim = stage.GetPrimAtPath(robot_prim_path)

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
            print(f"  [Self-collision] Enabled via raw attr on {target_prim.GetPath()}")
        except Exception as e2:
            print(f"  [Self-collision] WARNING: Could not enable — {e2}")


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