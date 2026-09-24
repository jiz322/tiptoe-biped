"""
show_standup_path.py — interactive tester for the stand-up path knots
(rl_env_cfg_p6.standup_path_knots, `standup_ramp_mode=path`).

Loads the SAME knot table training uses (parsed straight out of
rl_env_cfg_p6.py, no Isaac Lab import) and lets you check, pose by pose,
whether the joint angles / pitch / root height are mutually consistent
with the ground and with each other:

  * one button per knot  — spawn the robot ON the path at that knot exactly
    the way the env does (joints [0,hip,knee,0,-hip,-knee], root quat =
    rotation about +y by pitch_deg, z = root_h + spawn clearance), then
    watch what the plant does under the PD drives;
  * phase slider s       — spawn at an interpolated phase, or only move the
    PD targets there without teleporting (watch the plant follow the path);
  * difficulty slider d  — spawn at s0 = 1 - d and play the zero-action
    grace ramp s0 -> 1 over d * standup_grace_seconds (constant path speed),
    i.e. the motion the env's action center performs with a silent policy;
  * gain presets         — training gains (kp 8 / kd 0.5 Nm/rad + real
    effort limits) or a stiff hold (kp 200) to separate "pose is
    kinematically wrong" from "pose sags / topples under trained PD".

After every spawn a settle report prints at 0.5 / 1 / 2 s: root z and
pitch vs the knot, per-joint tracking error, root speed and a verdict
(STABLE / DRIFTING / FELL). The live status label in the window shows the
same numbers continuously.

Two ways to run:

A) Standalone (recommended):
       powershell -File script\run_show_standup_path.ps1
   Optional flags after the script name (the .ps1 passes them through):
       --cfg <path>    rl_env_cfg_p6.py to read the knots from
       --smoke         headless self-test: spawn at every knot + one d=1.0
                       ramp, print the reports, exit
   Plain `python show_standup_path.py` will NOT work — pxr/omni modules
   only exist once SimulationApp is up, which needs the env vars the .ps1 sets.

B) Inside Isaac Sim's Script Editor: paste/run, then press PLAY.
"""

import argparse
import ast
import math
import os
import sys

# ---- Standalone bootstrap: must happen BEFORE any pxr/omni import ----
_STANDALONE = __name__ == "__main__"
simulation_app = None
_ARGS = argparse.Namespace(cfg=None, smoke=False, headless=False)
if _STANDALONE:
    _p = argparse.ArgumentParser()
    _p.add_argument("--cfg", default=None, help="path to rl_env_cfg_p6.py")
    _p.add_argument("--smoke", action="store_true", help="headless self-test then exit")
    _p.add_argument("--scan", default=None,
                    help="headless spawn-consistency scan 'start:stop:step' over s (stiff hold, "
                         "+0.1 s snapshot only), e.g. 0.0:0.2:0.02")
    _p.add_argument("--headless", action="store_true")
    _ARGS = _p.parse_args()
    if _ARGS.smoke or _ARGS.scan:
        _ARGS.headless = True
    try:
        from isaacsim import SimulationApp          # Isaac Sim >= 4.x
    except ImportError:
        from omni.isaac.kit import SimulationApp    # older fallback
    simulation_app = SimulationApp({"headless": _ARGS.headless})

import numpy as np
import asyncio
from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema, Sdf, PhysicsSchemaTools, UsdShade
import omni.usd
import omni.kit.app
import omni.kit.commands
import omni.timeline
if not _ARGS.headless:
    import omni.ui as ui

# ============================================================
# CONFIG
# ============================================================

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
CFG_PATH = _ARGS.cfg or os.path.join(
    _REPO, "RL", "source", "RL", "RL", "tasks", "direct", "standup_curriculum", "rl_env_cfg_p6.py")

# Canonical asset = what training loads (rl_env_cfg_p6.ROBOT_USD_PATH); overridden
# below if the cfg file has a literal ROBOT_USD_PATH.
USD_PATH = os.path.join(_REPO, "assets", "tiptoe_v27", "robot", "robot.usd")

# IMU *body* prim (URDF link "imu" = rl_env_cfg_p6.imu_body_name). Must be a
# rigid body, not the joint prim of the same name.
IMU_LINK_PATH = "/World/Robot/imu"

JOINT_NAMES = ["hip_roll1", "hip_pitch1", "knee1", "hip_roll2", "hip_pitch2", "knee2"]

# Training actuator gains, JOINT side, Nm/rad (rl_env_cfg_p6 make_belt_actuators /
# hip_roll group) and effort limits. USD DriveAPI angular stiffness is per DEGREE:
# Isaac Sim writes kp_usd = kp_rad * pi/180 (isaacsim.core.prims articulation.py).
GAIN_PRESETS = {
    "train": {
        "hip_roll":  (6.0, 0.3, 8.0),
        "hip_pitch": (8.0, 0.5, 22.0),
        "knee":      (8.0, 0.5, 30.0),
    },
    "hold": {
        "hip_roll":  (200.0, 10.0, 1.0e6),
        "hip_pitch": (200.0, 10.0, 1.0e6),
        "knee":      (200.0, 10.0, 1.0e6),
    },
}
_RAD2USD = math.pi / 180.0

# Training physics (rl_env_cfg_p6 §2/§3)
PHYSICS_HZ = 500
GROUND_STATIC_FRICTION = 1.0
GROUND_DYNAMIC_FRICTION = 1.0

ROBOT_MAT_PATH = "/World/PhysMat_Robot"
GROUND_MAT_PATH = "/World/PhysMat_Ground"

# Deploy-side trim (control/config.py IMU_PG_OFFSET): hardware DISPLAYS pg_raw - offset.
_CONTROL_CONFIG = os.path.join(_REPO, "control", "config.py")
try:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("bigthigh_control_config", _CONTROL_CONFIG)
    _cfg_mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_cfg_mod)
    _IMU_PG_OFFSET = list(_cfg_mod.IMU_PG_OFFSET)
except Exception as _e:
    print(f"[IMU] control/config.py not loadable ({_e}); trim shown as 0")
    _IMU_PG_OFFSET = [0.0, 0.0]


# ============================================================
# CFG PARSING (AST — the cfg module itself imports Isaac Lab)
# ============================================================

# Fallback copy of the 09-08 default table, used only if the cfg can't be parsed.
_FALLBACK_KNOTS = [
    0.00, 3.090, -3.060, -90.0, 0.235,
    0.10, 2.740, -3.040, -60.6, 0.222,
    0.20, 2.260, -2.960, -38.2, 0.212,
    0.30, 1.810, -2.820, -20.7, 0.202,
    0.40, 1.360, -2.700,  -3.2, 0.184,
    0.45, 1.100, -2.630,   8.0, 0.170,
    0.5875, 0.838, -1.875,  9.7, 0.305,
    0.725, 0.575, -1.250,   8.4, 0.375,
    0.8625, 0.312, -0.625,  2.3, 0.424,
    1.00, 0.050,  0.000,   0.0, 0.439,
]


def load_cfg_values(path):
    """Pull the path-mode constants out of rl_env_cfg_p6.py without importing it."""
    vals = {
        "standup_path_knots": _FALLBACK_KNOTS,
        "standup_path_spawn_clearance_m": 0.003,
        "standup_grace_seconds": 4.0,
        "stand_joint_pos": None,
        "stand_pos": None,
        "usd_path": None,
    }
    try:
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
    except Exception as e:
        print(f"[cfg] could not parse {path} ({e}); using fallback constants")
        return vals

    scalar_keys = {"standup_path_knots", "standup_path_spawn_clearance_m", "standup_grace_seconds"}
    for node in ast.walk(tree):
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0].id, node.value
        if target in scalar_keys and value is not None:
            try:
                vals[target] = ast.literal_eval(value)
            except Exception:
                pass
        elif target == "ROBOT_USD_PATH" and value is not None:
            try:
                vals["usd_path"] = ast.literal_eval(value)
            except Exception:
                pass
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "InitialStateCfg":
            for kw in node.keywords:
                try:
                    if kw.arg == "joint_pos":
                        vals["stand_joint_pos"] = ast.literal_eval(kw.value)
                    elif kw.arg == "pos":
                        vals["stand_pos"] = ast.literal_eval(kw.value)
                except Exception:
                    pass
    return vals


_CFG = load_cfg_values(CFG_PATH)
if _CFG["usd_path"] and os.path.exists(_CFG["usd_path"]):
    USD_PATH = _CFG["usd_path"]

_rows = [float(v) for v in _CFG["standup_path_knots"]]
# 7-column table (09-08 eve): [s, hip_target, knee_target, pitch_deg, root_h, hip_rest, knee_rest];
# the older 5-column form [s, hip, knee, pitch_deg, root_h] is padded with rest = target.
if len(_rows) >= 14 and len(_rows) % 7 == 0:
    KNOTS = np.asarray(_rows).reshape(-1, 7)
elif len(_rows) >= 10 and len(_rows) % 5 == 0:
    _k5 = np.asarray(_rows).reshape(-1, 5)
    KNOTS = np.concatenate([_k5, _k5[:, 1:3]], axis=1)
else:
    raise ValueError("standup_path_knots: need flat rows [s,hip,knee,pitch_deg,root_h(,hip_rest,knee_rest)]")
if abs(KNOTS[0, 0]) > 1e-6 or abs(KNOTS[-1, 0] - 1.0) > 1e-6 or np.any(np.diff(KNOTS[:, 0]) < 0):
    raise ValueError(f"standup_path_knots: s must ascend from 0 to 1, got {KNOTS[:, 0].tolist()}")
SPAWN_CLEARANCE = float(_CFG["standup_path_spawn_clearance_m"])
GRACE_SECONDS = float(_CFG["standup_grace_seconds"])
STAND_JOINT_POS = _CFG["stand_joint_pos"] or {
    "hip_roll1": 0.0, "hip_pitch1": 0.05, "knee1": 0.0,
    "hip_roll2": 0.0, "hip_pitch2": -0.05, "knee2": 0.0,
}
STAND_Z = float((_CFG["stand_pos"] or (0.0, 0.0, 0.434))[2])


# ============================================================
# PATH MATH (mirrors rl_env_p6._pwl / _path_joint_pos / reset quat)
# ============================================================

def path_pose(s):
    """(hip, knee, pitch_deg, root_h) on the path at phase s — piecewise-linear like the env."""
    s = float(np.clip(s, 0.0, 1.0))
    hip = float(np.interp(s, KNOTS[:, 0], KNOTS[:, 1]))
    knee = float(np.interp(s, KNOTS[:, 0], KNOTS[:, 2]))
    pitch = float(np.interp(s, KNOTS[:, 0], KNOTS[:, 3]))
    h = float(np.interp(s, KNOTS[:, 0], KNOTS[:, 4]))
    return hip, knee, pitch, h


def path_joint_targets(s, rest=False):
    """PD-target joints on the path (rest=True → the joints the plant rests at under that
    target; the env writes THOSE as the spawn state, mirrors rl_env_p6._path_joint_pos)."""
    s = float(np.clip(s, 0.0, 1.0))
    if rest:
        hip = float(np.interp(s, KNOTS[:, 0], KNOTS[:, 5]))
        knee = float(np.interp(s, KNOTS[:, 0], KNOTS[:, 6]))
    else:
        hip, knee, _, _ = path_pose(s)
    return {
        "hip_roll1": 0.0, "hip_pitch1": hip, "knee1": knee,
        "hip_roll2": 0.0, "hip_pitch2": -hip, "knee2": -knee,
    }


def pitch_quat(pitch_deg):
    """Root quaternion the env spawns with: rotation about +y by pitch (−90° = kneel_quat)."""
    half = 0.5 * math.radians(pitch_deg)
    return Gf.Quatd(math.cos(half), Gf.Vec3d(0.0, math.sin(half), 0.0))


def _quat_to_rot3(qw, qx, qy, qz):
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy)],
    ])


def projected_gravity(quat_wxyz):
    """Gravity (world −Z) in the body frame — same as rl_env_p6 quat_apply_inverse(q, g)."""
    return _quat_to_rot3(*quat_wxyz).T @ np.array([0.0, 0.0, -1.0])


def pitch_deg_from_quat(quat_wxyz):
    """Pitch about +y consistent with pitch_quat(): −90° ↔ kneel_quat, 0 ↔ standing."""
    pg = projected_gravity(quat_wxyz)
    return math.degrees(math.atan2(pg[0], -pg[2]))


def _gain_group(joint_name):
    return "hip_roll" if joint_name.startswith("hip_roll") else (
        "hip_pitch" if joint_name.startswith("hip_pitch") else "knee")


# ============================================================
# GLOBALS
# ============================================================

_joint_prims = {}          # joint name -> USD prim
_robot_prim = None
_art_root_path = None      # prim carrying ArticulationRootAPI
_window = None
_status_label = None
_gain_label = None
_gain_preset = "train"
_update_sub = None

_sim_time = 0.0            # accumulated timeline dt while playing
_was_playing = False
_sim_view = None
_art_view = None
_imu_view = None
_dof_index = {}            # joint name -> column in the tensor view

_cur_targets = path_joint_targets(1.0)   # what the drives are currently asked for
_cur_s_cmd = 1.0
_spawn_ref = None          # dict(label, s, pitch, h, t0) of the last spawn, for the settle report
_settle_marks = []         # sim times at which to print a report
_ramp = None               # dict(s0, T, t0, speed) while a ramp is playing
_smoke_done = False
_last_report = {}

_print_period = 1.0
_print_acc = 0.0


# ============================================================
# LIVE STATE (omni.physics.tensors; USD fallback)
# ============================================================

def _ensure_views():
    """Create tensor views lazily once physics is running; None if unavailable."""
    global _sim_view, _art_view, _imu_view, _dof_index
    if _art_view is not None:
        return True
    if _art_root_path is None:
        return False
    try:
        import omni.physics.tensors as pt
        _sim_view = pt.create_simulation_view("numpy")
        _sim_view.set_subspace_roots("/")
        _art_view = _sim_view.create_articulation_view(_art_root_path)
        names = list(_art_view.shared_metatype.dof_names)
        _dof_index = {n: i for i, n in enumerate(names)}
        try:
            _imu_view = _sim_view.create_rigid_body_view(IMU_LINK_PATH)
        except Exception:
            _imu_view = None
        print(f"[state] tensor view on {_art_root_path}: dofs={names}"
              f"{'' if _imu_view is not None else '  (IMU body view unavailable)'}")
        return True
    except Exception as e:
        print(f"[state] omni.physics.tensors unavailable ({e}); falling back to USD reads")
        _sim_view = _art_view = _imu_view = None
        return False


def _drop_views():
    global _sim_view, _art_view, _imu_view
    _sim_view = _art_view = _imu_view = None


def read_state():
    """dict: q (name->rad), root_pos, root_quat_wxyz, root_lin_vel, imu_quat_wxyz — or None."""
    st = {"q": {}, "root_pos": None, "root_quat": None, "root_vel": None, "imu_quat": None}
    if _ensure_views():
        try:
            q = np.asarray(_art_view.get_dof_positions())[0]
            for n in JOINT_NAMES:
                if n in _dof_index:
                    st["q"][n] = float(q[_dof_index[n]])
            rt = np.asarray(_art_view.get_root_transforms())[0]          # x y z qx qy qz qw
            st["root_pos"] = rt[:3].copy()
            st["root_quat"] = np.array([rt[6], rt[3], rt[4], rt[5]])
            rv = np.asarray(_art_view.get_root_velocities())[0]
            st["root_vel"] = rv[:3].copy()
            if _imu_view is not None:
                it = np.asarray(_imu_view.get_transforms())[0]
                st["imu_quat"] = np.array([it[6], it[3], it[4], it[5]])
            return st
        except Exception as e:
            print(f"[state] tensor read failed ({e}); dropping views")
            _drop_views()

    # USD fallback (only current if physics write-back to USD is enabled)
    stage = omni.usd.get_context().get_stage()
    for n in JOINT_NAMES:
        prim = _joint_prims.get(n)
        val = None
        if prim is not None:
            js = PhysxSchema.JointStateAPI.Get(prim, "angular")
            if js:
                deg = js.GetPositionAttr().Get()
                if deg is not None:
                    val = math.radians(deg)
        st["q"][n] = val if val is not None else float("nan")
    try:
        from pxr import Usd
        root = stage.GetPrimAtPath(_art_root_path or "/World/Robot")
        m = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        p = m.ExtractTranslation()
        r = m.ExtractRotation().GetQuat()
        st["root_pos"] = np.array([p[0], p[1], p[2]])
        st["root_quat"] = np.array([r.GetReal(), *r.GetImaginary()])
        imu = stage.GetPrimAtPath(IMU_LINK_PATH)
        if imu.IsValid():
            mi = UsdGeom.Xformable(imu).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            ri = mi.ExtractRotation().GetQuat()
            st["imu_quat"] = np.array([ri.GetReal(), *ri.GetImaginary()])
    except Exception:
        pass
    return st


# ============================================================
# DRIVES
# ============================================================

def _apply_gains(preset):
    global _gain_preset
    _gain_preset = preset
    for name, prim in _joint_prims.items():
        kp, kd, tau = GAIN_PRESETS[preset][_gain_group(name)]
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive:
            continue
        drive.CreateStiffnessAttr().Set(kp * _RAD2USD)
        drive.CreateDampingAttr().Set(kd * _RAD2USD)
        drive.CreateMaxForceAttr().Set(tau)
    if _gain_label is not None:
        _gain_label.text = _gain_text()
    g = GAIN_PRESETS[preset]
    print(f"[gains] {preset}: hip_roll kp/kd/τmax {g['hip_roll']}, hip_pitch {g['hip_pitch']}, "
          f"knee {g['knee']}  (Nm/rad, Nm·s/rad, Nm — joint side)")


def _gain_text():
    g = GAIN_PRESETS[_gain_preset]
    return (f"gains = {_gain_preset.upper()}   hip_pitch/knee kp {g['hip_pitch'][0]:g} kd {g['hip_pitch'][1]:g} "
            f"| hip_roll kp {g['hip_roll'][0]:g} | τmax hip {g['hip_pitch'][2]:g} knee {g['knee'][2]:g} Nm")


def set_drive_targets(targets, s_cmd=None):
    """Move only the PD targets (no teleport)."""
    global _cur_targets, _cur_s_cmd
    _cur_targets = dict(targets)
    if s_cmd is not None:
        _cur_s_cmd = float(s_cmd)
    for name, rad in targets.items():
        prim = _joint_prims.get(name)
        if prim is None:
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if drive:
            drive.GetTargetPositionAttr().Set(math.degrees(rad))


def _write_initial_joint_state(targets):
    for name, rad in targets.items():
        prim = _joint_prims.get(name)
        if prim is None:
            continue
        js = PhysxSchema.JointStateAPI.Get(prim, "angular")
        if not js:
            js = PhysxSchema.JointStateAPI.Apply(prim, "angular")
        js.CreatePositionAttr().Set(math.degrees(rad))
        js.CreateVelocityAttr().Set(0.0)


def _write_root_pose(pitch_deg, z):
    xform = UsdGeom.Xformable(_robot_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, float(z)))
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(pitch_quat(pitch_deg))


# ============================================================
# SPAWN / RAMP
# ============================================================

async def spawn_at_s(s, label=None):
    """Stop physics, author the env's spawn state for phase s, restart physics."""
    global _spawn_ref, _settle_marks, _ramp, _sim_time
    s = float(np.clip(s, 0.0, 1.0))
    hip, knee, pitch, h = path_pose(s)
    targets = path_joint_targets(s)
    label = label or f"s={s:.4f}"
    _ramp = None

    tl = omni.timeline.get_timeline_interface()
    tl.stop()
    _drop_views()
    await omni.kit.app.get_app().next_update_async()

    z = h + SPAWN_CLEARANCE
    _write_root_pose(pitch, z)
    _write_initial_joint_state(path_joint_targets(s, rest=True))   # env writes the REST joints
    set_drive_targets(targets, s_cmd=s)                            # ...and drives the TARGET path
    await omni.kit.app.get_app().next_update_async()

    _sim_time = 0.0
    _spawn_ref = {"label": label, "s": s, "pitch": pitch, "h": z, "hip": hip, "knee": knee}
    _settle_marks = [0.1, 0.5, 1.0, 2.0]
    print(f"\n[spawn] {label}: hip {hip:+.3f} knee {knee:+.3f} rad | pitch {pitch:+.1f}° | "
          f"z {z:.4f} m (root_h {h:.3f} + clearance {SPAWN_CLEARANCE:.3f})")
    tl.play()


async def spawn_at_difficulty_and_ramp(d, speed=1.0):
    """Env contract: spawn at s0 = 1 − d, then the action center travels s0→1 over d·grace."""
    global _ramp
    d = float(np.clip(d, 0.0, 1.0))
    s0 = 1.0 - d
    await spawn_at_s(s0, label=f"d={d:.3f} (s0={s0:.3f})")
    T = d * GRACE_SECONDS / max(speed, 1e-6)
    _ramp = {"s0": s0, "T": T, "t0": _sim_time, "speed": speed, "done": False}
    print(f"[ramp] s {s0:.3f} → 1.000 over {T:.2f} s (grace {GRACE_SECONDS:g} s × d, speed ×{speed:g}); "
          f"settle report at +0.5/1/2 s after arrival")


def stop_ramp():
    global _ramp
    if _ramp is not None:
        print(f"[ramp] stopped at s_cmd={_cur_s_cmd:.3f}")
    _ramp = None


def _tick_ramp():
    global _ramp, _settle_marks
    if _ramp is None or _ramp["done"]:
        return
    frac = 1.0 if _ramp["T"] <= 0 else min((_sim_time - _ramp["t0"]) / _ramp["T"], 1.0)
    s = _ramp["s0"] + (1.0 - _ramp["s0"]) * frac
    set_drive_targets(path_joint_targets(s), s_cmd=s)
    if frac >= 1.0:
        _ramp["done"] = True
        _spawn_ref.update({"label": _spawn_ref["label"] + " → ramp arrived s=1",
                           "s": 1.0, "pitch": 0.0, "h": STAND_Z, "hip": path_pose(1.0)[0],
                           "knee": path_pose(1.0)[1]})
        _settle_marks = [_sim_time + 0.5, _sim_time + 1.0, _sim_time + 2.0]
        print(f"[ramp] arrived at s=1 (standing target) at t={_sim_time:.2f} s")


# ============================================================
# REPORTING
# ============================================================

def _assess(st):
    """Compare live state to the reference pose; returns (lines, verdict)."""
    global _last_report
    ref = _spawn_ref
    lines = []
    if st["root_pos"] is None or ref is None:
        return ["(no state)"], "?"
    z = float(st["root_pos"][2])
    pitch = pitch_deg_from_quat(st["root_quat"])
    dz = z - ref["h"]
    dp = pitch - ref["pitch"]
    speed = float(np.linalg.norm(st["root_vel"])) if st["root_vel"] is not None else float("nan")

    errs = {}
    for n in JOINT_NAMES:
        q = st["q"].get(n, float("nan"))
        if n in _cur_targets and not math.isnan(q):
            errs[n] = q - _cur_targets[n]
    worst = max(errs, key=lambda k: abs(errs[k])) if errs else None

    lines.append(f"s_cmd {_cur_s_cmd:.3f}   ref: {ref['label']}")
    lines.append(f"root z {z:.4f} m (ref {ref['h']:.4f}, Δ {dz*1000:+.1f} mm)   "
                 f"pitch {pitch:+.1f}° (ref {ref['pitch']:+.1f}°, Δ {dp:+.1f}°)   |v| {speed:.3f} m/s")
    short = {"hip_roll1": "hr1", "hip_pitch1": "hp1", "knee1": "k1",
             "hip_roll2": "hr2", "hip_pitch2": "hp2", "knee2": "k2"}
    lines.append("q [rad]: " + "  ".join(f"{short[n]} {st['q'].get(n, float('nan')):+.3f}" for n in JOINT_NAMES))
    if worst is not None:
        lines.append(f"tracking err vs target: max {abs(errs[worst]):.3f} rad ({worst}); "
                     + "  ".join(f"{short[n]} {e:+.3f}" for n, e in errs.items()))

    if st["imu_quat"] is not None:
        pg = projected_gravity(st["imu_quat"])
        pgd = pg - np.array([_IMU_PG_OFFSET[0], _IMU_PG_OFFSET[1], 0.0])
        lines.append(f"IMU proj_grav RAW ({pg[0]:+.3f},{pg[1]:+.3f},{pg[2]:+.3f})   "
                     f"as balance_log displays (raw − {_IMU_PG_OFFSET}): ({pgd[0]:+.3f},{pgd[1]:+.3f},{pgd[2]:+.3f})")

    if abs(dp) > 30.0 or z < 0.6 * ref["h"]:
        verdict = "FELL"
    elif abs(dp) > 3.0 or abs(dz) > 0.015 or (not math.isnan(speed) and speed > 0.05):
        verdict = "DRIFTING"
    else:
        verdict = "STABLE"
    _last_report = {"z": z, "dz": dz, "pitch": pitch, "dp": dp, "speed": speed,
                    "max_err": abs(errs[worst]) if worst else float("nan"), "verdict": verdict}
    return lines, verdict


def print_report(tag=""):
    st = read_state()
    lines, verdict = _assess(st)
    print(f"\n--- {tag or 'state'} @ t={_sim_time:.2f} s  →  {verdict} ---")
    for ln in lines:
        print("  " + ln)


def _refresh_status_label():
    if _status_label is None:
        return
    st = read_state()
    lines, verdict = _assess(st)
    ramp_txt = ""
    if _ramp is not None and not _ramp["done"]:
        ramp_txt = f"   RAMP {100*min((_sim_time-_ramp['t0'])/max(_ramp['T'],1e-6),1):.0f} %"
    _status_label.text = f"t={_sim_time:.2f} s   verdict: {verdict}{ramp_txt}\n" + "\n".join(lines)


# ============================================================
# UPDATE LOOP
# ============================================================

def _on_update(e):
    global _sim_time, _was_playing, _print_acc, _settle_marks
    tl = omni.timeline.get_timeline_interface()
    playing = tl.is_playing()
    if playing and not _was_playing:
        _drop_views()
    _was_playing = playing
    if not playing:
        return
    dt = e.payload["dt"]
    _sim_time += dt
    _tick_ramp()

    due = [m for m in _settle_marks if _sim_time >= m]
    if due:
        _settle_marks = [m for m in _settle_marks if _sim_time < m]
        print_report(tag=f"settle +{due[-1]:.1f}s ({_spawn_ref['label'] if _spawn_ref else ''})")

    _print_acc += dt
    if _print_acc >= 0.1:
        _print_acc = 0.0
        _refresh_status_label()


def _start_update_sub():
    global _update_sub
    _update_sub = (
        omni.kit.app.get_app()
        .get_update_event_stream()
        .create_subscription_to_pop(_on_update, name="standup_path_tester")
    )


# ============================================================
# UI
# ============================================================

def _run(coro):
    try:
        from omni.kit.async_engine import run_coroutine
        run_coroutine(coro)
    except ImportError:
        asyncio.ensure_future(coro)


def _build_ui():
    global _window, _status_label, _gain_label
    if _window is not None:
        _window.destroy()
        _window = None

    _window = ui.Window("Stand-up Path Tester", width=760, height=720)
    s_model = ui.SimpleFloatModel(1.0)
    d_model = ui.SimpleFloatModel(1.0)
    speed_model = ui.SimpleFloatModel(1.0)

    with _window.frame:
        with ui.ScrollingFrame():
            with ui.VStack(spacing=6):
                ui.Label(f"knots from {os.path.basename(CFG_PATH)}  |  clearance {SPAWN_CLEARANCE*1000:.0f} mm  |  "
                         f"grace {GRACE_SECONDS:g} s × d  |  physics {PHYSICS_HZ} Hz",
                         height=20, style={"font_size": 12, "color": 0xFF888888})

                # ---- gains ----
                with ui.HStack(height=28, spacing=8):
                    ui.Button("Training gains (kp 8 / kd 0.5)", clicked_fn=lambda: _apply_gains("train"))
                    ui.Button("Stiff hold (kp 200)", clicked_fn=lambda: _apply_gains("hold"))
                _gain_label = ui.Label(_gain_text(), height=18, style={"font_size": 12})
                ui.Separator(height=6)

                # ---- knot buttons ----
                ui.Label("Spawn ON the path at a knot (env spawn contract: joints, pitch quat, z = h + clearance):",
                         height=20, style={"font_size": 13})
                for i in range(0, len(KNOTS), 2):
                    with ui.HStack(height=30, spacing=6):
                        for k in range(i, min(i + 2, len(KNOTS))):
                            s, hip, knee, pitch, h = KNOTS[k, :5]

                            def _mk(kk, ss):
                                return lambda: _run(spawn_at_s(ss, label=f"K{kk} s={ss:.4f}"))
                            ui.Button(f"K{k}  s={s:.3f}   hip {hip:+.3f}  knee {knee:+.3f}   "
                                      f"pitch {pitch:+.1f}°   h {h:.3f}",
                                      clicked_fn=_mk(k, float(s)), style={"font_size": 12})
                ui.Separator(height=6)

                # ---- phase slider ----
                ui.Label("Arbitrary phase s (piecewise-linear between knots, like the env):",
                         height=20, style={"font_size": 13})
                with ui.HStack(height=28, spacing=6):
                    ui.Label("s", width=20)
                    ui.FloatSlider(model=s_model, min=0.0, max=1.0, step=0.005)
                    s_lbl = ui.Label("", width=260, style={"font_size": 12})

                def _s_changed(m):
                    hip, knee, pitch, h = path_pose(m.as_float)
                    s_lbl.text = f"hip {hip:+.3f} knee {knee:+.3f} | {pitch:+.1f}° | h {h:.3f}"
                s_model.add_value_changed_fn(_s_changed)
                _s_changed(s_model)
                with ui.HStack(height=28, spacing=8):
                    ui.Button("Spawn at s", clicked_fn=lambda: _run(spawn_at_s(s_model.as_float)))
                    ui.Button("Drive PD targets to s (no teleport)",
                              clicked_fn=lambda: (stop_ramp(),
                                                  set_drive_targets(path_joint_targets(s_model.as_float),
                                                                    s_cmd=s_model.as_float),
                                                  print(f"[drive] targets → s={s_model.as_float:.3f}")))
                ui.Separator(height=6)

                # ---- difficulty ramp ----
                ui.Label("Zero-action grace ramp: spawn at s0 = 1 − d, PD center travels s0 → 1 over d × grace:",
                         height=20, style={"font_size": 13})
                with ui.HStack(height=28, spacing=6):
                    ui.Label("d", width=20)
                    ui.FloatSlider(model=d_model, min=0.0, max=1.0, step=0.01)
                    d_lbl = ui.Label("", width=260, style={"font_size": 12})
                    ui.Label("speed ×", width=50)
                    ui.FloatField(model=speed_model, width=50)

                def _d_changed(m):
                    d = m.as_float
                    d_lbl.text = f"s0 {1-d:.3f}, ramp {d*GRACE_SECONDS:.2f} s"
                d_model.add_value_changed_fn(_d_changed)
                _d_changed(d_model)
                with ui.HStack(height=28, spacing=8):
                    ui.Button("Spawn at d and run ramp",
                              clicked_fn=lambda: _run(spawn_at_difficulty_and_ramp(
                                  d_model.as_float, max(speed_model.as_float, 0.01))))
                    ui.Button("Stop ramp (hold current target)", clicked_fn=stop_ramp)
                    ui.Button("Print state now", clicked_fn=lambda: print_report("manual"))
                ui.Separator(height=6)

                _status_label = ui.Label("(press PLAY or spawn)", style={"font_size": 12}, word_wrap=True)


# ============================================================
# SCENE SETUP
# ============================================================

async def setup_robot():
    global _joint_prims, _robot_prim, _art_root_path

    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath("/World"):
        omni.kit.commands.execute("DeletePrims", paths=["/World"])
        await omni.kit.app.get_app().next_update_async()
    UsdGeom.Xform.Define(stage, "/World")

    # ---- physics scene: training solver, 500 Hz, CPU (leave the GPUs to the training runs) ----
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/World/PhysicsScene"))
    physx_scene.CreateSolverTypeAttr("TGS")
    physx_scene.CreateMinPositionIterationCountAttr(8)
    physx_scene.CreateMinVelocityIterationCountAttr(4)
    physx_scene.CreateTimeStepsPerSecondAttr(PHYSICS_HZ)
    physx_scene.CreateEnableGPUDynamicsAttr(False)
    physx_scene.CreateBroadphaseTypeAttr("MBP")

    # ---- ground + training friction ----
    PhysicsSchemaTools.addGroundPlane(stage, "/World/GroundPlane", "Z", 10.0,
                                      Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.5, 0.5, 0.5))
    ground_mat = _make_material(stage, GROUND_MAT_PATH)
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/World/GroundPlane")).Bind(
        UsdShade.Material(ground_mat), UsdShade.Tokens.weakerThanDescendants, "physics")

    # ---- robot ----
    _robot_prim = stage.DefinePrim("/World/Robot")
    _robot_prim.GetReferences().AddReference(USD_PATH)
    await omni.kit.app.get_app().next_update_async()
    await omni.kit.app.get_app().next_update_async()
    _apply_robot_material(stage, "/World/Robot")

    _art_root_path = _enable_self_collision(stage, "/World/Robot")

    # ---- joints: drives + initial state at the standing knot (s = 1) ----
    _joint_prims = {}
    found = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint) and str(prim.GetPath()).startswith("/World/Robot"):
            found.append(prim.GetName())
            if prim.GetName() in JOINT_NAMES:
                _joint_prims[prim.GetName()] = prim
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.CreateTypeAttr("force")
                PhysxSchema.JointStateAPI.Apply(prim, "angular")
    missing = set(JOINT_NAMES) - set(found)
    if missing:
        print(f"*** WARNING: joints not found: {missing}  (available: {found})")
    _apply_gains("train")

    s1 = path_joint_targets(1.0)
    _write_root_pose(0.0, path_pose(1.0)[3] + SPAWN_CLEARANCE)
    _write_initial_joint_state(s1)
    set_drive_targets(s1, s_cmd=1.0)

    # ---- knot table ----
    print("\n" + "=" * 78)
    print(f"STAND-UP PATH KNOTS  ({CFG_PATH})")
    print(f"  clearance {SPAWN_CLEARANCE} m | grace {GRACE_SECONDS} s × d | stand z {STAND_Z} | usd {USD_PATH}")
    print(f"  {'k':>2} {'s':>7} {'hip_T':>7} {'knee_T':>7} {'pitch°':>7} {'root_h':>7} {'hip_R':>7} {'knee_R':>7}"
          f"   T = PD target, R = rest joints written at spawn; leg2 = (−hip, −knee), hip_roll = 0")
    for k, (s, hip, knee, pitch, h, hr, kr) in enumerate(KNOTS):
        print(f"  {k:>2} {s:7.4f} {hip:+7.3f} {knee:+7.3f} {pitch:+7.1f} {h:7.3f} {hr:+7.3f} {kr:+7.3f}")
    print("=" * 78)
    print(">>> Press PLAY, then click a knot / run a ramp. Reports print at +0.5/1/2 s after each spawn.\n")

    if not _ARGS.headless:
        _build_ui()
    _start_update_sub()


def _make_material(stage, path):
    UsdPhysics.MaterialAPI.Apply(stage.DefinePrim(path))
    m = UsdPhysics.MaterialAPI.Get(stage, path)
    m.CreateRestitutionAttr(0.0)
    m.CreateStaticFrictionAttr(GROUND_STATIC_FRICTION)
    m.CreateDynamicFrictionAttr(GROUND_DYNAMIC_FRICTION)
    try:
        px = PhysxSchema.PhysxMaterialAPI.Apply(stage.GetPrimAtPath(path))
        px.CreateRestitutionCombineModeAttr("multiply")
        px.CreateFrictionCombineModeAttr("multiply")
    except Exception:
        pass
    return stage.GetPrimAtPath(path)


def _apply_robot_material(stage, robot_prim_path):
    # The ballfeet USD's colliders are instanceable — stage.Traverse() never sees them
    # and instance proxies can't be authored — so bind once on the robot root; physics
    # material bindings inherit down the hierarchy.
    mat_prim = _make_material(stage, ROBOT_MAT_PATH)
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(robot_prim_path)).Bind(
        UsdShade.Material(mat_prim), UsdShade.Tokens.weakerThanDescendants, "physics")
    from pxr import Usd
    count = sum(1 for p in Usd.PrimRange(stage.GetPrimAtPath(robot_prim_path), Usd.TraverseInstanceProxies())
                if p.HasAPI(UsdPhysics.CollisionAPI))
    print(f"  [material] friction {GROUND_STATIC_FRICTION}/{GROUND_DYNAMIC_FRICTION}, restitution 0, "
          f"combine multiply (as training) bound on {robot_prim_path} ({count} colliders incl. instance proxies)")


def _enable_self_collision(stage, robot_prim_path):
    """Enable self-collision; returns the articulation-root prim path."""
    target = None
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(robot_prim_path) and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            target = prim
            break
    if target is None:
        print("  [Self-collision] No ArticulationRootAPI found; applying to robot root.")
        target = stage.GetPrimAtPath(robot_prim_path)
        UsdPhysics.ArticulationRootAPI.Apply(target)
    physx_art = PhysxSchema.PhysxArticulationAPI.Apply(target)
    try:
        physx_art.CreateEnabledSelfCollisionsAttr().Set(True)
    except AttributeError:
        attr = target.GetAttribute("physxArticulation:enabledSelfCollisions")
        if not attr.IsValid():
            attr = target.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool)
        attr.Set(True)
    print(f"  [Self-collision] enabled on articulation root {target.GetPath()}")
    return str(target.GetPath())


# ============================================================
# HEADLESS SMOKE TEST
# ============================================================

async def _sleep_sim(seconds):
    target = _sim_time + seconds
    guard = 0
    while _sim_time < target and guard < 20000:
        await omni.kit.app.get_app().next_update_async()
        guard += 1


async def smoke():
    global _smoke_done
    await setup_robot()
    tl = omni.timeline.get_timeline_interface()
    tl.play()
    await _sleep_sim(0.5)
    summaries = {}
    for preset in ("hold", "train"):
        _apply_gains(preset)
        rows = []
        for k, row in enumerate(KNOTS):
            await spawn_at_s(float(row[0]), label=f"K{k} s={row[0]:.4f}")
            await _sleep_sim(0.12)
            early = dict(_last_report)
            await _sleep_sim(2.0)
            rows.append((f"K{k} s={row[0]:.3f}", early, dict(_last_report)))
        await spawn_at_difficulty_and_ramp(1.0, 1.0)
        await _sleep_sim(0.12)
        early = dict(_last_report)
        await _sleep_sim(GRACE_SECONDS + 2.1)
        rows.append(("ramp d=1.0", early, dict(_last_report)))
        summaries[preset] = rows

    print("\n" + "=" * 96)
    print("SMOKE SUMMARY — per knot: +0.1 s snapshot (spawn consistency) and +2 s settle; "
          "ramp = zero-action d=1.0 → s=1")
    for preset, rows in summaries.items():
        g = GAIN_PRESETS[preset]["hip_pitch"]
        print(f"\n  [{preset.upper()} gains: kp {g[0]:g} kd {g[1]:g} Nm/rad]"
              + ("  — kinematic consistency of the knot itself" if preset == "hold"
                 else "  — what the plant does under trained PD with a silent policy"))
        print(f"  {'pose':<14} | {'+0.1s Δz mm':>11} {'Δpitch°':>8} {'|v|':>6} | "
              f"{'+2s verdict':<11} {'Δz mm':>7} {'pitch°':>7} {'Δpitch°':>8} {'|v|':>6} {'max q err':>10}")
        for name, e, r in rows:
            if not r or not e:
                print(f"  {name:<14} (no report)")
                continue
            print(f"  {name:<14} | {e['dz']*1000:+11.1f} {e['dp']:+8.1f} {e['speed']:6.3f} | "
                  f"{r['verdict']:<11} {r['dz']*1000:+7.1f} {r['pitch']:+7.1f} {r['dp']:+8.1f} "
                  f"{r['speed']:6.3f} {r['max_err']:10.3f}")
    print("=" * 96)
    _smoke_done = True


async def scan(spec):
    """Spawn-consistency sweep: stiff hold, +0.1 s snapshot at each s. A pose that
    intersects the ground shows up as a large +Δz / |v| ejection; a pose hanging in
    the air as a large −Δz drop with |v| ~ g·t."""
    global _smoke_done
    a, b, step = (float(v) for v in spec.split(":"))
    ss = np.arange(a, b + 0.5 * step, step)
    await setup_robot()
    tl = omni.timeline.get_timeline_interface()
    tl.play()
    await _sleep_sim(0.5)
    _apply_gains("hold")
    rows = []
    for s in ss:
        await spawn_at_s(float(s))
        await _sleep_sim(0.12)
        rows.append((float(s), dict(_last_report)))
    print("\n" + "=" * 78)
    print("SPAWN CONSISTENCY SCAN (stiff hold, snapshot at +0.1 s; free fall would be Δz ≈ −49 mm, |v| ≈ 1 m/s)")
    print(f"  {'s':>6} {'hip':>7} {'knee':>7} {'pitch°':>7} {'h':>6} | {'Δz mm':>8} {'Δpitch°':>8} {'|v| m/s':>8}  note")
    for s, r in rows:
        hip, knee, pitch, h = path_pose(s)
        if not r:
            print(f"  {s:6.3f} (no report)")
            continue
        note = ("EJECTED (penetrating ground)" if r["dz"] > 0.05 or r["speed"] > 1.5
                else "dropping (spawned in the air)" if r["dz"] < -0.03
                else "ok")
        print(f"  {s:6.3f} {hip:+7.3f} {knee:+7.3f} {pitch:+7.1f} {h:6.3f} | {r['dz']*1000:+8.1f} {r['dp']:+8.1f} "
              f"{r['speed']:8.3f}  {note}")
    print("=" * 78)
    _smoke_done = True


# ============================================================
# LAUNCH
# ============================================================
if _STANDALONE and (_ARGS.smoke or _ARGS.scan):
    _run(scan(_ARGS.scan) if _ARGS.scan else smoke())
    while simulation_app.is_running() and not _smoke_done:
        simulation_app.update()
    simulation_app.close()
else:
    _run(setup_robot())
    if _STANDALONE:
        while simulation_app.is_running():
            simulation_app.update()
        simulation_app.close()
