"""
interactive_joint_tuner.py — Run inside Isaac Sim's Script Editor

Creates a floating UI window with sliders for each joint.
Adjust joint angles IN REAL TIME while physics is running.

Workflow:
    1. Set your USD_PATH and SPAWN_HEIGHT below
    2. Run this script in Script Editor
    3. Press PLAY in the toolbar
    4. Drag the sliders to find the standing pose
    5. When the robot stands, click "Print Current Angles" to copy the values
"""

import numpy as np
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
SPAWN_HEIGHT = 0.32

# Joint config: name, initial angle (rad), slider min, slider max, sign multiplier
# Sign multipliers correct the URDF frame inconsistencies so that:
#   +pitch = same direction on both legs
#   +roll  = same direction on both legs
#   +knee  = same direction on both legs
JOINT_CONFIG = [
    ("hip_roll1",  0.0,    -1.5, 1.5,   1),
    ("hip_pitch1", 3.06,   -2.0, 6.28, -1),   # extended to 2π
    ("knee1",      3.01,   -3.0, 6.28, -1),   # extended to 2π
    ("hip_roll2",  0.0,    -1.5, 1.5,  -1),
    ("hip_pitch2", 3.06,   -2.0, 6.28,  1),   # extended to 2π
    ("knee2",      3.01,   -3.0, 6.28,  1),   # extended to 2π
]

# Sign map for quick lookup
SIGN_MAP = {name: sign for name, _, _, _, sign in JOINT_CONFIG}

JOINT_STIFFNESS = 50.0
JOINT_DAMPING   = 10.0

# ============================================================
# GLOBALS
# ============================================================

_joint_prims = {}       # name -> USD prim
_slider_models = {}     # name -> slider model
_window = None
_link_legs = True       # When True, left/right legs move together

# Mirror pairs: moving one auto-sets the other
MIRROR_PAIRS = {
    "hip_roll1":  "hip_roll2",
    "hip_roll2":  "hip_roll1",
    "hip_pitch1": "hip_pitch2",
    "hip_pitch2": "hip_pitch1",
    "knee1":      "knee2",
    "knee2":      "knee1",
}

# ============================================================
# FUNCTIONS
# ============================================================

def update_joint_target(joint_name, value_rad):
    """Update a joint's drive target in real-time, applying sign correction."""
    if joint_name in _joint_prims:
        prim = _joint_prims[joint_name]
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if drive:
            corrected = value_rad * SIGN_MAP.get(joint_name, 1)
            drive.GetTargetPositionAttr().Set(np.degrees(corrected))


def print_current_angles():
    """Print all current slider values as a copy-pasteable dict."""
    print("\n" + "=" * 60)
    print("CURRENT JOINT ANGLES — SYMMETRIC (slider values):")
    print("=" * 60)
    print("SYMMETRIC_ANGLES = {")
    for name, _, _, _, _ in JOINT_CONFIG:
        if name in _slider_models:
            val = _slider_models[name].as_float
            print(f'    "{name}": {val:>8.4f},  # {np.degrees(val):>7.2f} deg')
    print("}")
    print("\n# Actual angles sent to USD (with sign correction):")
    print("RAW_ANGLES = {")
    for name, _, _, _, sign in JOINT_CONFIG:
        if name in _slider_models:
            val = _slider_models[name].as_float
            raw = val * sign
            print(f'    "{name}": {raw:>8.4f},  # {np.degrees(raw):>7.2f} deg (sign={sign:+d})')
    print("}")
    print("=" * 60 + "\n")


def reset_to_zero():
    """Reset all sliders to zero."""
    for name, _, _, _, _ in JOINT_CONFIG:
        if name in _slider_models:
            _slider_models[name].as_float = 0.0
            update_joint_target(name, 0.0)


def build_ui():
    """Build the slider window."""
    global _window

    # Close existing window if re-running
    if _window is not None:
        _window.destroy()
        _window = None

    _window = ui.Window("Joint Tuner", width=450, height=500)

    with _window.frame:
        with ui.VStack(spacing=5):
            ui.Label("Drag sliders while physics is running (PLAY)",
                     height=30, style={"font_size": 14})
            ui.Separator(height=5)

            for joint_name, init_val, smin, smax, sign in JOINT_CONFIG:
                with ui.HStack(height=30, spacing=5):
                    sign_str = "+" if sign == 1 else "-"
                    ui.Label(f"{joint_name} ({sign_str})", width=120)

                    slider = ui.FloatSlider(min=smin, max=smax, step=0.01)
                    slider.model.as_float = init_val
                    _slider_models[joint_name] = slider.model

                    value_label = ui.Label(f"{init_val:.3f}", width=60)

                    # Callback: update joint + label when slider moves
                    def make_callback(jname, vlabel):
                        def on_changed(model):
                            val = model.as_float
                            vlabel.text = f"{val:.3f}"
                            update_joint_target(jname, val)
                            # Mirror to paired joint if linked
                            if _link_legs and jname in MIRROR_PAIRS:
                                pair = MIRROR_PAIRS[jname]
                                if pair in _slider_models:
                                    _slider_models[pair].as_float = val
                                    update_joint_target(pair, val)
                        return on_changed

                    slider.model.add_value_changed_fn(make_callback(joint_name, value_label))

            ui.Separator(height=10)

            with ui.HStack(height=30, spacing=10):
                ui.Button("Print Current Angles",
                          clicked_fn=print_current_angles,
                          style={"font_size": 13})
                ui.Button("Reset to Zero",
                          clicked_fn=reset_to_zero,
                          style={"font_size": 13})

            with ui.HStack(height=25, spacing=5):
                cb = ui.CheckBox(width=20)
                cb.model.as_bool = True
                ui.Label("Link Legs (left & right move together)", style={"font_size": 12})

                def on_link_changed(model):
                    global _link_legs
                    _link_legs = model.as_bool
                cb.model.add_value_changed_fn(on_link_changed)

            ui.Separator(height=5)
            ui.Label("1. Adjust sliders to find standing pose",
                     style={"font_size": 12, "color": 0xFF888888})
            ui.Label("2. Click 'Print Current Angles' to save values",
                     style={"font_size": 12, "color": 0xFF888888})


async def setup_scene():
    """Set up the robot scene."""
    global _joint_prims

    stage = omni.usd.get_context().get_stage()

    # Clean up previous run
    if stage.GetPrimAtPath("/World"):
        omni.kit.commands.execute("DeletePrims", paths=["/World"])
        await omni.kit.app.get_app().next_update_async()

    # Create world
    UsdGeom.Xform.Define(stage, "/World")

    # Physics scene
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)

    # Ground plane
    PhysicsSchemaTools.addGroundPlane(
        stage, "/World/GroundPlane", "Z", 10.0,
        Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.5, 0.5, 0.5)
    )

    # Load robot
    robot_prim_path = "/World/Robot"
    robot_prim = stage.DefinePrim(robot_prim_path)
    robot_prim.GetReferences().AddReference(USD_PATH)
    await omni.kit.app.get_app().next_update_async()
    await omni.kit.app.get_app().next_update_async()

    # Set spawn height
    xform = UsdGeom.Xformable(robot_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, SPAWN_HEIGHT))

    # Find and configure joints
    _joint_prims = {}
    print("\nJoints found:")
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            name = prim.GetName()
            print(f"  {name} -> {prim.GetPath()}")

            if any(name == jc[0] for jc in JOINT_CONFIG):
                _joint_prims[name] = prim

                # Find initial value and sign from config
                init_val = 0.0
                sign = 1
                for jc in JOINT_CONFIG:
                    if jc[0] == name:
                        init_val = jc[1]
                        sign = jc[4]
                        break

                # Apply drive
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.CreateTypeAttr("force")
                drive.CreateStiffnessAttr(JOINT_STIFFNESS)
                drive.CreateDampingAttr(JOINT_DAMPING)
                drive.CreateTargetPositionAttr(np.degrees(init_val * sign))
    
    print(f"\nConfigured {len(_joint_prims)} joints for interactive control.")
    print("Press PLAY, then use the Joint Tuner window to adjust angles.\n")

    # Build the UI
    build_ui()


# Launch
asyncio.ensure_future(setup_scene())