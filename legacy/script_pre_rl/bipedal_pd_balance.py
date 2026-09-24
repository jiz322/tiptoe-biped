"""
bipedal_pd_balance.py — Run inside Isaac Sim's Script Editor

PD balance controller for a point-foot bipedal robot.
Same control architecture as the balanced wheel (pitch PD → motor command),
extended with a roll axis.

Architecture (mirrors your wheel code):
    Wheel:  pitch_error → PD → velocity_target → MIT Kd → torque
    Biped:  tilt_error  → PD → position_offset  → MIT Kp → torque

Control loop (200 Hz physics step callback):
    1. Read torso orientation (body Z-axis projection, same as IMU)
    2. Compute pitch and roll tilt errors
    3. PD → hip_pitch correction (forward/backward balance)
    4. PD → hip_roll correction  (lateral balance)
    5. Send corrected position targets to joints

Tuning guide:
    - If robot falls without reacting:     increase Kp
    - If robot oscillates / vibrates:      decrease Kp or increase Kd
    - If robot falls FASTER with control:  flip PITCH_DIRECTION or swap PITCH_AXIS
    - If lateral balance is off:           flip ROLL_DIRECTION or swap ROLL_AXIS

Usage:
    1. Paste into Script Editor (Window > Script Editor)
    2. Press Run
    3. Press PLAY in the toolbar
    4. Watch debug output — tune gains as needed
"""

import numpy as np
import math
import time
import asyncio
from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema, Sdf, PhysicsSchemaTools
import omni.usd
import omni.kit.app
import omni.kit.commands
import omni.timeline
import omni.physx

# ============================================================
# USER CONFIG
# ============================================================

USD_PATH = r"<repo>\urdf_onshape_export\bigthigh-urdf\robot\robot.usd"
SPAWN_HEIGHT = 0.42
TORSO_PRIM_PATH = "/World/Robot/torso2hip"

# Default standing pose (from your tuning — this is the balance point)
DEFAULT_POSE = {
    "hip_roll1":  0.0,
    "hip_pitch1": 0.05,
    "knee1":      0.0,
    "hip_roll2":  0.0,
    "hip_pitch2": -0.05,
    "knee2":      0.0,
}

# Motor parameters (matched to real hardware)
# Hip pitch drive — stiffer so joints actually track the targets
JOINT_PARAMS = {
    "hip_roll1":  {"stiffness": 5.0,   "damping": 0.3, "effort": 6.0},
    "hip_roll2":  {"stiffness": 5.0,   "damping": 0.3, "effort": 6.0},
    "hip_pitch1": {"stiffness": 20.0,  "damping": 1.0, "effort": 17.0},
    "hip_pitch2": {"stiffness": 20.0,  "damping": 1.0, "effort": 17.0},
    "knee1":      {"stiffness": 38.0,  "damping": 0.5, "effort": 23.2},
    "knee2":      {"stiffness": 38.0,  "damping": 0.5, "effort": 23.2},
}

# Sign mapping: how each joint responds to a correction.
PITCH_JOINT_SIGNS = {"hip_pitch1": 1, "hip_pitch2": -1, "knee1": -0.5, "knee2": 0.5}
ROLL_JOINT_SIGNS  = {"hip_roll1":  1, "hip_roll2":  -1}

# ── Balance PD gains ──────────────────────────────────────────
PITCH_KP = 2.0
PITCH_KD = 0.08    # was 0.3 — at 200Hz you need much less
ROLL_KP  = 0.3
ROLL_KD  = 0.005
# ── Clamp correction to safe range
MAX_CORRECTION = 0.8  # radians — about 17 degrees max hip adjustment
# ── Direction signs ───────────────────────────────────────────
PITCH_DIRECTION = -1
ROLL_DIRECTION  = -1

# ── Tilt axis mapping ────────────────────────────────────────
PITCH_AXIS = 0
ROLL_AXIS  = 1

# ── Low-pass filters ─────────────────────────────────────────
PITCH_FILTER_ALPHA      = 0.3
PITCH_RATE_FILTER_ALPHA = 0.3
ROLL_FILTER_ALPHA       = 0.3
ROLL_RATE_FILTER_ALPHA  = 0.3

# ── Observation delay ────────────────────────────────────────
OBS_DELAY_STEPS = 1

# ── Safety ────────────────────────────────────────────────────
MAX_TILT_DEG    = 45.0
MIN_BASE_HEIGHT = 0.15

# ── Debug ─────────────────────────────────────────────────────
PRINT_EVERY_N = 6  # every 200 steps = 1 second at 200Hz


# ============================================================
# LOW-PASS FILTER (same as your wheel code)
# ============================================================

class LowPassFilter:
    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self.value = 0.0
        self.initialized = False

    def update(self, raw):
        if not self.initialized:
            self.value = raw
            self.initialized = True
        else:
            self.value = self.alpha * raw + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.initialized = False
        self.value = 0.0


# ============================================================
# BALANCE CONTROLLER
# ============================================================

class BalanceController:
    def __init__(self):
        self.pitch_filter      = LowPassFilter(PITCH_FILTER_ALPHA)
        self.pitch_rate_filter = LowPassFilter(PITCH_RATE_FILTER_ALPHA)
        self.roll_filter       = LowPassFilter(ROLL_FILTER_ALPHA)
        self.roll_rate_filter  = LowPassFilter(ROLL_RATE_FILTER_ALPHA)

        self.prev_pitch = 0.0
        self.prev_roll  = 0.0
        self.obs_buffer = []
        self.step       = 0

    def reset(self):
        self.pitch_filter.reset()
        self.pitch_rate_filter.reset()
        self.roll_filter.reset()
        self.roll_rate_filter.reset()
        self.prev_pitch = 0.0
        self.prev_roll  = 0.0
        self.obs_buffer.clear()
        self.step = 0

    def get_tilt(self, world_transform):
        """
        Extract pitch, roll, and height from the torso's world transform.
        Uses body Z-axis projection — same as what an IMU measures.
        """
        body_z_x = world_transform[2][0]
        body_z_y = world_transform[2][1]
        body_z_z = world_transform[2][2]
        base_height = world_transform[3][2]

        pitch = math.atan2(body_z_x if PITCH_AXIS == 0 else body_z_y, body_z_z)
        roll  = math.atan2(body_z_y if ROLL_AXIS  == 1 else body_z_x, body_z_z)

        return pitch, roll, base_height

    def compute(self, pitch_raw, roll_raw, dt):
        """
        Compute joint corrections from current tilt.
        Same structure as your wheel code: error → PD → correction → targets
        """
        # ── Observation delay (simulate real latency) ─────────
        self.obs_buffer.append((pitch_raw, roll_raw))
        if len(self.obs_buffer) > OBS_DELAY_STEPS:
            pitch_obs, roll_obs = self.obs_buffer.pop(0)
        else:
            pitch_obs, roll_obs = pitch_raw, roll_raw

        # ── Filter ────────────────────────────────────────────
        pitch = self.pitch_filter.update(pitch_obs)
        roll  = self.roll_filter.update(roll_obs)

        # ── Angular rates via differentiation ─────────────────
        if self.step == 0:
            pitch_rate = 0.0
            roll_rate  = 0.0
        else:
            pitch_rate = (pitch_obs - self.prev_pitch) / dt
            roll_rate  = (roll_obs  - self.prev_roll)  / dt

        pitch_rate = self.pitch_rate_filter.update(pitch_rate)
        roll_rate  = self.roll_rate_filter.update(roll_rate)

        self.prev_pitch = pitch_obs
        self.prev_roll  = roll_obs

        # ── PD control ────────────────────────────────────────
        pitch_correction = PITCH_DIRECTION * (PITCH_KP * pitch + PITCH_KD * pitch_rate)
        roll_correction  = ROLL_DIRECTION  * (ROLL_KP  * roll  + ROLL_KD  * roll_rate)

        # ── Clamp correction to safe range ────────────────────
        pitch_correction = max(-MAX_CORRECTION, min(MAX_CORRECTION, pitch_correction))
        roll_correction  = max(-MAX_CORRECTION, min(MAX_CORRECTION, roll_correction))

        # ── Build joint targets ───────────────────────────────
        targets = {}
        for name, default in DEFAULT_POSE.items():
            target = default

            if name in PITCH_JOINT_SIGNS:
                target += PITCH_JOINT_SIGNS[name] * pitch_correction

            if name in ROLL_JOINT_SIGNS:
                target += ROLL_JOINT_SIGNS[name] * roll_correction

            targets[name] = target

        self.step += 1

        debug = {
            "pitch_deg":        math.degrees(pitch),
            "roll_deg":         math.degrees(roll),
            "pitch_rate":       pitch_rate,
            "roll_rate":        roll_rate,
            "pitch_correction": pitch_correction,
            "roll_correction":  roll_correction,
        }

        return targets, debug


# ============================================================
# GLOBALS
# ============================================================

_joint_prims   = {}
_controller    = BalanceController()
_was_playing   = False
_physx_sub     = None
_timeline_sub  = None


# ============================================================
# JOINT HELPER
# ============================================================

def set_joint_target(name, target_rad):
    """Update a single joint's drive target position (rad → deg for PhysX)."""
    if name in _joint_prims:
        prim = _joint_prims[name]
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if drive:
            drive.GetTargetPositionAttr().Set(np.degrees(target_rad))


# ============================================================
# TIMELINE EVENT HANDLER
# ============================================================

def _on_timeline_event(event):
    global _was_playing
    if event.type == int(omni.timeline.TimelineEventType.PLAY):
        _controller.reset()
        _was_playing = True
        print(f"\n[Controller] PLAY — reset to step 0, balancing active (200 Hz)")
    elif event.type == int(omni.timeline.TimelineEventType.STOP):
        _was_playing = False
        print(f"[Controller] STOPPED")


# ============================================================
# PHYSICS STEP CALLBACK (200 Hz)
# ============================================================

def physics_step_callback(dt):
    """Called every physics step (200 Hz)."""
    if not _was_playing:
        return

    stage = omni.usd.get_context().get_stage()
    torso_prim = stage.GetPrimAtPath(TORSO_PRIM_PATH)
    if not torso_prim or not torso_prim.IsValid():
        return

    torso_xform = UsdGeom.Xformable(torso_prim)

    try:
        world_transform = torso_xform.ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
    except RuntimeError:
        return

    pitch, roll, base_height = _controller.get_tilt(world_transform)

    # Safety check
    tilt_mag = math.degrees(math.sqrt(pitch * pitch + roll * roll))
    if tilt_mag > MAX_TILT_DEG or base_height < MIN_BASE_HEIGHT:
        for name, default in DEFAULT_POSE.items():
            set_joint_target(name, default)
        if _controller.step % PRINT_EVERY_N == 0:
            status = "TILT" if tilt_mag > MAX_TILT_DEG else "FALLEN"
            print(f"  [{status}] tilt={tilt_mag:.1f}°  "
                  f"height={base_height:.3f}m  — holding default")
        _controller.step += 1
        return

    # Compute and apply
    targets, debug = _controller.compute(pitch, roll, dt)
    for name, target in targets.items():
        set_joint_target(name, target)

    # Debug print (every PRINT_EVERY_N steps)
    if _controller.step % PRINT_EVERY_N == 0:
        print(
            f"  step={_controller.step:5d}  dt={dt:.4f}  "
            f"pitch={debug['pitch_deg']:+6.2f}°  "
            f"roll={debug['roll_deg']:+6.2f}°  "
            f"p_rate={debug['pitch_rate']:+6.3f}  "
            f"r_rate={debug['roll_rate']:+6.3f}  |  "
            f"p_corr={debug['pitch_correction']:+.4f}  "
            f"r_corr={debug['roll_correction']:+.4f}  "
            f"h={base_height:.3f}m"
        )


# ============================================================
# SCENE SETUP
# ============================================================

async def setup_scene():
    global _joint_prims, _physx_sub, _timeline_sub

    stage = omni.usd.get_context().get_stage()

    # Clean up previous run
    if stage.GetPrimAtPath("/World"):
        omni.kit.commands.execute("DeletePrims", paths=["/World"])
        await omni.kit.app.get_app().next_update_async()

    # Create world
    UsdGeom.Xform.Define(stage, "/World")

    # Physics scene with 200 Hz timestep
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)

    physx_api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx_api.CreateTimeStepsPerSecondAttr(200)

    # Ground plane with friction
    PhysicsSchemaTools.addGroundPlane(
        stage, "/World/GroundPlane", "Z", 10.0,
        Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.5, 0.5, 0.5)
    )
    for prim in stage.Traverse():
        if prim.GetPath().pathString.startswith("/World/GroundPlane"):
            mat = UsdPhysics.MaterialAPI.Apply(prim)
            if mat:
                mat.CreateStaticFrictionAttr(0.9)
                mat.CreateDynamicFrictionAttr(0.8)
                mat.CreateRestitutionAttr(0.1)

    # Load robot
    robot_prim_path = "/World/Robot"
    robot_prim = stage.DefinePrim(robot_prim_path)
    robot_prim.GetReferences().AddReference(USD_PATH)
    await omni.kit.app.get_app().next_update_async()
    await omni.kit.app.get_app().next_update_async()

    # Add friction to robot collision surfaces (AFTER robot is loaded)
    for prim in stage.Traverse():
        if prim.GetPath().pathString.startswith("/World/Robot"):
            if prim.HasAPI(UsdPhysics.CollisionAPI) or "collision" in prim.GetName().lower():
                mat = UsdPhysics.MaterialAPI.Apply(prim)
                if mat:
                    mat.CreateStaticFrictionAttr(0.9)
                    mat.CreateDynamicFrictionAttr(0.8)

    # Spawn height
    xform = UsdGeom.Xformable(robot_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, SPAWN_HEIGHT))

    # Configure joints
    _joint_prims = {}

    print("\n" + "=" * 70)
    print("  BIPEDAL PD BALANCE CONTROLLER")
    print("=" * 70)

    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            name = prim.GetName()
            if name in JOINT_PARAMS:
                _joint_prims[name] = prim
                params = JOINT_PARAMS[name]
                target_rad = DEFAULT_POSE.get(name, 0.0)

                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.CreateTypeAttr("force")
                drive.CreateStiffnessAttr(params["stiffness"])
                drive.CreateDampingAttr(params["damping"])
                drive.CreateMaxForceAttr(params["effort"])
                drive.CreateTargetPositionAttr(np.degrees(target_rad))

                print(f"  {name:15s}  default={target_rad:+.3f} rad  "
                      f"Kp={params['stiffness']:<5}  Kd={params['damping']:<5}  "
                      f"max={params['effort']} N·m")

    print(f"\n  Joints configured: {len(_joint_prims)}")
    print(f"  Physics timestep:  200 Hz")
    print(f"  Control rate:      200 Hz (physics callback)")
    print(f"  Obs delay:         {OBS_DELAY_STEPS} steps")
    print(f"  ──────────────────────────────────────────")
    print(f"  Pitch PD:  Kp={PITCH_KP}  Kd={PITCH_KD}  dir={PITCH_DIRECTION:+d}  axis={PITCH_AXIS}")
    print(f"  Roll  PD:  Kp={ROLL_KP}  Kd={ROLL_KD}  dir={ROLL_DIRECTION:+d}  axis={ROLL_AXIS}")
    print(f"  Filter α:  {PITCH_FILTER_ALPHA}")
    print(f"  Safety:    max tilt ±{MAX_TILT_DEG}°  min height {MIN_BASE_HEIGHT}m")
    print(f"  ──────────────────────────────────────────")
    print(f"  If robot falls FASTER with control:")
    print(f"    1. Try PITCH_DIRECTION = +1")
    print(f"    2. Try swapping PITCH_AXIS / ROLL_AXIS")
    print(f"\n  >>> Press PLAY to start balancing <<<")
    print("=" * 70 + "\n")

    # Register physics step callback
    _physx_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(
        physics_step_callback
    )

    # Register timeline event for play/stop detection
    timeline = omni.timeline.get_timeline_interface()
    stream = timeline.get_timeline_event_stream()
    _timeline_sub = stream.create_subscription_to_pop(_on_timeline_event)

    print("[Controller] Physics callback + timeline events registered")


# ============================================================
# LAUNCH
# ============================================================

asyncio.ensure_future(setup_scene())