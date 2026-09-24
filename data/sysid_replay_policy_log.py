"""
sysid_replay_policy_log.py — replay a balance_log.py run through the TRAINING
env's full action chain in IsaacLab, and overlay sim vs real joint response.

This is the action->motion half of the parity story. obs_parity_live.py proved
sensing->action is bit-close (obs <=1.8e-7, actions <=9.5e-7) — but it never
steps the sim. This script closes the other half: feed the LOGGED action
stream through exactly what the training env believes happens after the policy

    action (post-clamp) -> LPF(alpha) -> transport delay -> u = default + a*scale
    -> 2nd-order actuator-lag prefilter (omega/zeta from cfg) -> DCMotor PD
    -> PhysX @ 500 Hz (decimation 5)

and compare the resulting joint pos/vel against what the real robot did.
The balance log is HELD-OUT data for the actuator model: the June 22 sysid fit
used scripted steps/chirps, never the policy's own command distribution.

Differences from sysid_replay_sim.py (which replays sysid p_des trials):
  * consumes a balance_log NPZ (raw actions, not p_des) and runs the env's
    LPF + delay + prefilter stages that the sysid replay skips;
  * omega/zeta/friction/armature/scales are read from the LIVE RlEnvCfg — no
    local copies to drift out of sync (sysid_replay_sim.py still carries the
    old omega/zeta 13.15/14.08/15.94);
  * PD gains default to the values RECOVERED from the log's own tau_cmd
    recursion (the 20260803 run had knee Kp=4.0/Kd=0.3, NOT the trained
    8.0/0.5 — using cfg gains here would confound the actuator-model test);
  * the base is NOT fixed at spawn: the real robot hung from a harness and
    swung (mean tilt ~34 deg, wander ~12 deg in the 0803 log). Two modes:
      --base fixed   root kinematically pinned at the run's MEAN orientation
      --base driven  root orientation/ang-vel replayed from the logged IMU
                     (roll/pitch from proj_grav, yaw integrated from gyro)
    Both are implemented as per-substep root pose+velocity writes (the
    fix_root_link fallback documented in sysid_replay_sim.py), so both run
    in one process. Trust conclusions where both modes agree.

No ground plane is spawned — the robot hangs in empty space, like the run.

Usage (Isaac Lab python):
    isaaclab.bat -p data/sysid_replay_policy_log.py \
        --npz control/model_4150_log_20260803_140243.npz --headless
Options:
    --base both|fixed|driven   (default both)
    --delays 0,1,2             action transport delay sweep, ticks (default 0,1,2)
    --alpha 0.6                action LPF alpha (default: cfg.action_lpf_alpha)
    --kp/--kd "6,8,4,6,8,4"    override recovered gains (sim joint space)
    --no-prefilter --no-lpf    ablations to attribute residuals per stage
    --self-check               run the first config twice, assert identical
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading

# ── Launch IsaacLab (must happen before other isaaclab imports) ─────
from isaaclab.app import AppLauncher

cli = argparse.ArgumentParser()
cli.add_argument("--npz", required=True, help="balance_log.py NPZ")
cli.add_argument("--base", choices=["both", "fixed", "driven"], default="both")
cli.add_argument("--delays", default="0,1,2",
                 help="comma list of action transport delays to sweep (ticks)")
cli.add_argument("--alpha", type=float, default=None,
                 help="action LPF alpha (default: cfg.action_lpf_alpha)")
cli.add_argument("--kp", default=None, help="6 comma values, sim joint space")
cli.add_argument("--kd", default=None, help="6 comma values, sim joint space")
cli.add_argument("--no-prefilter", action="store_true",
                 help="disable the 2nd-order actuator-lag prefilter")
cli.add_argument("--no-lpf", action="store_true", help="disable the action LPF")
cli.add_argument("--warmup", type=int, default=20,
                 help="ticks excluded from RMSE (filter-state settling)")
cli.add_argument("--teacher-forcing", action="store_true",
                 help="reset sim joints to the LOGGED state at every control "
                      "tick and measure ONE-STEP prediction error instead of "
                      "free-run drift. Separates 'model locally wrong' from "
                      "'open-loop divergence compounding'.")
cli.add_argument("--self-check", action="store_true",
                 help="run first config twice and require identical output")
cli.add_argument("--coupling", action="store_true",
                 help="ALSO run every (base, delay) config with the real belt "
                      "drivetrain model: knee PD computed in motor coordinates "
                      "(kd fights c*qd_hp belt drag; kp sees the stale hip "
                      "decoupling baked at send time) and the belt reaction "
                      "torque c*tau_knee applied to hip pitch. Compare RMSE "
                      "against the uncoupled baseline to test whether the "
                      "coupled model shrinks the sim2real gap.")
cli.add_argument("--coupling-terms", default="knee,hip",
                 help="which coupling torques the --coupling pass injects: "
                      "'knee' (belt drag on the knee PD), 'hip' (belt reaction "
                      "on hip pitch), or 'knee,hip' (both). Ablation knob: the "
                      "two are independent physics claims.")
cli.add_argument("--actuator", choices=["belt", "legacy"], default="belt",
                 help="'belt' = BeltKneeDCMotor firmware replica (the training "
                      "cfg default: motor-space knee PD, stale-hip hold, "
                      "motor-velocity envelope); 'legacy' = pre-belt plain "
                      "DCMotor groups (the model every pre-2026-08 policy was "
                      "trained on). Baselines should use 'legacy'.")
cli.add_argument("--knee-ff", choices=["on", "off"], default="on",
                 help="belt actuator only: model the hardware KNEE_DECOUPLE_FF "
                      "v_des (live since 2026-08-06). PROTOCOL: replay logs "
                      "recorded BEFORE the patch with 'off', after with 'on'.")
cli.add_argument("--no-stale-hip", action="store_true",
                 help="belt actuator only: disable the stale-hip target hold "
                      "(ablation).")
cli.add_argument("--hip-reaction", action="store_true",
                 help="belt actuator only: enable the ideal J^T belt reaction "
                      "on hip pitch (empirically refuted 2026-08-06 — ablation "
                      "only).")
cli.add_argument("--hip-armature", type=float, default=None,
                 help="override hip_pitch armature (kg m^2). Defaults: legacy "
                      "4.2e-3, belt 8.4e-3 (reflected knee rotor). Use "
                      "'--actuator legacy --hip-armature 8.4e-3' for the "
                      "armature-only A/B.")
AppLauncher.add_app_launcher_args(cli)
args = cli.parse_args()
launcher = AppLauncher(args)
sim_app = launcher.app

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation

# Path setup: work from any cwd; also covers non-pip-installed RL package.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "RL", "source", "RL"))
from RL.tasks.direct.standup_curriculum.rl_env_cfg_p6 import (  # noqa: E402
    RlEnvCfg, make_belt_actuators, make_legacy_actuators)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

# Hardware joint order (balance_log.py 6-vectors) == env _desired_order.
HW_JOINT_ORDER = ["hip_roll1", "hip_pitch1", "knee1",
                  "hip_roll2", "hip_pitch2", "knee2"]
JOINT_LABELS = ["hr1", "hp1", "kn1", "hr2", "hp2", "kn2"]
# logical joint TYPE per hw slot (cfg param lookup — NOT actuator group keys;
# the belt variant merges hip_pitch+knee into one "hip_knee" group)
GROUP_OF = ["hip_roll", "hip_pitch", "knee", "hip_roll", "hip_pitch", "knee"]
KNEE_IDX = (2, 5)
JOINT_SIGN = np.array([+1, +1, -1, +1, +1, -1], dtype=np.float64)  # config.py
KNEE_GEAR = 22.0 / 30.0
KNEE_COUPLING = 0.733            # config.py KNEE_HIP_COUPLING
# RS01 motor-frame envelope (hip_pitch direct drive; knee via belt)
RS01_SAT, RS01_VLIM = 17.0, 27.5


# ────────────────────────────────────────────────────────────────────
# Gain recovery from the log (twin of balance_forensics.py)
# ────────────────────────────────────────────────────────────────────

def recover_gains(target, pos, vel, tau_cmd_joint, warmup=20):
    """(kp, kd, alpha) per joint from the tau_cmd LPF recursion.

    tau[t] - a*tau[t-1] = kp*[(1-a)u[t] - pos[t] + a*pos[t-1]] + kd*[a*vel[t-1] - vel[t]]
    Exact (float32 noise) whenever the effort soft-clip did not engage.
    """
    alphas = [0.0, 0.5, 0.6, 0.65, 0.7]
    kp = np.zeros(6); kd = np.zeros(6); al = np.zeros(6); rmse = np.zeros(6)
    for i in range(6):
        best = None
        for a in alphas:
            y = (tau_cmd_joint[1:, i] - a * tau_cmd_joint[:-1, i])[warmup:]
            A = ((1 - a) * target[1:, i] - pos[1:, i] + a * pos[:-1, i])[warmup:]
            B = (a * vel[:-1, i] - vel[1:, i])[warmup:]
            M = np.stack([A, B], axis=1)
            coef, *_ = np.linalg.lstsq(M, y, rcond=None)
            r = float(np.sqrt(np.mean((M @ coef - y) ** 2)))
            if best is None or r < best[0]:
                best = (r, coef[0], coef[1], a)
        rmse[i], kp[i], kd[i], al[i] = best
    return kp, kd, al, rmse


# ────────────────────────────────────────────────────────────────────
# Base-orientation reconstruction from the logged IMU
# ────────────────────────────────────────────────────────────────────
# The URDF mounts the imu link on the root (torso2hip) with identity rotation,
# so IMU body frame == root body frame and the logged proj_grav / ang_vel are
# directly the root's. Roll/pitch come from proj_grav (world z in body frame
# = -proj_grav); yaw is gravity-unobservable, so integrate the gyro's world-z
# rate. Yaw only spins the hanging robot about world z — it has no effect on
# joint gravity loading — but keeping it makes the ang-vel replay consistent.

def _rot_to_quat(R):
    """3x3 rotation (body->world) -> quat (w,x,y,z)."""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s; y = 0.25 * s
        x = (R[0, 1] + R[1, 0]) / s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s; z = 0.25 * s
        x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _tilt_rotation(g_body):
    """Rows of R (body->world) with third row = -g_body (zero-yaw choice)."""
    r3 = -g_body / np.linalg.norm(g_body)          # world z in body frame
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, r3)) > 0.95:                 # near-degenerate: use y
        ref = np.array([0.0, 1.0, 0.0])
    r1 = ref - np.dot(ref, r3) * r3
    r1 /= np.linalg.norm(r1)
    r2 = np.cross(r3, r1)
    return np.stack([r1, r2, r3], axis=0)


def build_base_quats(proj_grav, ang_vel, dt):
    """Per-tick root quats (w,x,y,z) and world ang-vel from logged IMU."""
    n = proj_grav.shape[0]
    quats = np.zeros((n, 4))
    ang_w = np.zeros((n, 3))
    yaw = 0.0
    for t in range(n):
        R_tilt = _tilt_rotation(proj_grav[t])
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        R = Rz @ R_tilt
        quats[t] = _rot_to_quat(R)
        ang_w[t] = R @ ang_vel[t]
        yaw += ang_w[t][2] * dt
    return quats, ang_w


def _slerp(q0, q1, s):
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1 = -q1; d = -d
    if d > 0.9995:
        q = q0 + s * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(np.clip(d, -1.0, 1.0))
    return (np.sin((1 - s) * th) * q0 + np.sin(s * th) * q1) / np.sin(th)


# ────────────────────────────────────────────────────────────────────
# Replay
# ────────────────────────────────────────────────────────────────────

def main():
    d = np.load(args.npz, allow_pickle=True)
    n = len(d["time"])
    action = d["action"].astype(np.float64)          # (n,6) post-clamp
    target_log = d["target"].astype(np.float64)      # default + a*scale (check)
    pos_log = d["sim_joint_pos"].astype(np.float64)
    vel_log = d["sim_joint_vel"].astype(np.float64)
    tau_cmd_log = d["motor_tau_cmd"].astype(np.float64)   # motor frame
    tau_meas_log = d["motor_tau_meas"].astype(np.float64)  # motor frame
    proj_grav = d["proj_grav"].astype(np.float64)
    ang_vel = d["ang_vel"].astype(np.float64)
    w = min(args.warmup, n // 4)

    tau_cmd_joint = tau_cmd_log * JOINT_SIGN
    tau_cmd_joint[:, KNEE_IDX[0]] /= KNEE_GEAR
    tau_cmd_joint[:, KNEE_IDX[1]] /= KNEE_GEAR
    tau_meas_joint = tau_meas_log * JOINT_SIGN
    tau_meas_joint[:, KNEE_IDX[0]] /= KNEE_GEAR
    tau_meas_joint[:, KNEE_IDX[1]] /= KNEE_GEAR

    # ── gains: log channels > CLI > recovered from tau_cmd recursion ──
    if "kp" in d.files:
        kp_hw = d["kp"][0].astype(np.float64)
        kd_hw = d["kd"][0].astype(np.float64)
        gains_src = "log channels"
    elif args.kp and args.kd:
        kp_hw = np.array([float(x) for x in args.kp.split(",")])
        kd_hw = np.array([float(x) for x in args.kd.split(",")])
        gains_src = "cli"
    else:
        kp_hw, kd_hw, al_rec, g_rmse = recover_gains(
            target_log, pos_log, vel_log, tau_cmd_joint, warmup=w)
        gains_src = f"recovered from tau_cmd recursion (fit rmse max {g_rmse.max():.1e} Nm)"
        if g_rmse.max() > 0.02:
            print("[warn] gain recovery residual is large — effort clip may have "
                  "engaged; pass --kp/--kd explicitly")

    # ── training-side constants from the LIVE cfg ─────────────────────
    env_cfg = RlEnvCfg()
    DECIMATION = int(env_cfg.decimation)
    DT_PHY = float(env_cfg.sim.dt)
    DT_CTRL = DECIMATION * DT_PHY
    alpha = float(env_cfg.action_lpf_alpha if args.alpha is None else args.alpha)
    if args.no_lpf:
        alpha = 0.0
    scales_grp = {"hip_roll": env_cfg.real_action_scale_hip_roll,
                  "hip_pitch": env_cfg.real_action_scale_hip_pitch,
                  "knee": env_cfg.real_action_scale_knee}
    scales_hw = np.array([scales_grp[g] for g in GROUP_OF])
    omega_grp = {"hip_roll": env_cfg.act_lag_omega_hip_roll,
                 "hip_pitch": env_cfg.act_lag_omega_hip_pitch,
                 "knee": env_cfg.act_lag_omega_knee}
    zeta_grp = {"hip_roll": env_cfg.act_lag_zeta_hip_roll,
                "hip_pitch": env_cfg.act_lag_zeta_hip_pitch,
                "knee": env_cfg.act_lag_zeta_knee}
    use_prefilter = bool(env_cfg.act_lag_enable) and not args.no_prefilter
    default_hw = np.array([env_cfg.robot_cfg.init_state.joint_pos[j]
                           for j in HW_JOINT_ORDER])

    # target-identity sanity: logged target must equal default + a*scale,
    # otherwise the log came from different scales/default than this cfg.
    tgt_err = np.abs(target_log - (default_hw + action * scales_hw)).max()
    if tgt_err > 1e-4:
        print(f"[FAIL] logged target != default + action*scale (max {tgt_err:.2e}) "
              f"— cfg scales/default do not match this log. Aborting.")
        sim_app.close(); sys.exit(1)

    delays = [int(x) for x in args.delays.split(",")]
    base_modes = ["fixed", "driven"] if args.base == "both" else [args.base]

    print(f"\n=== policy-log replay: {os.path.basename(args.npz)} ===")
    print(f"  ticks={n} ({n*DT_CTRL:.2f} s)  decimation={DECIMATION} dt_phy={DT_PHY}")
    print(f"  gains ({gains_src}):")
    for i in range(6):
        print(f"    {JOINT_LABELS[i]}: kp={kp_hw[i]:6.3f} kd={kd_hw[i]:6.3f}")
    print(f"  action LPF alpha={alpha}   prefilter={'ON' if use_prefilter else 'OFF'}"
          + (f" omega={list(omega_grp.values())} zeta={list(zeta_grp.values())}"
             if use_prefilter else ""))
    print(f"  base modes={base_modes}   delay sweep={delays}\n")

    # ── build sim + robot (single articulation, root left FREE; both base
    #    modes are per-substep kinematic root writes) ───────────────────
    sim = SimulationContext(env_cfg.sim)
    robot_cfg = env_cfg.robot_cfg
    robot_cfg.prim_path = "/World/Robot"
    # Hang the robot high with no ground plane — nothing to touch, like the run.
    ROOT_POS = (0.0, 0.0, 0.8)
    robot_cfg.init_state.pos = ROOT_POS
    # NOTE: deliberately NOT overriding effort_limit_sim/velocity_limit_sim the
    # way sysid_replay_sim.py does — this replay validates the TRAINING model,
    # so the DCMotor + PhysX clamp config must be exactly the cfg's.

    # ── actuator variant (belt firmware replica vs legacy pre-belt) ────
    if args.coupling and args.actuator == "belt":
        print("[FAIL] --coupling injects the belt torques externally; the belt "
              "actuator already models them internally (double-counting). Use "
              "--coupling with --actuator legacy only.")
        sim_app.close(); sys.exit(1)
    if args.actuator == "legacy":
        _kw = {}
        if args.hip_armature is not None:
            _kw["hip_pitch_armature"] = args.hip_armature
        robot_cfg.actuators = make_legacy_actuators(**_kw)
    else:
        _kw = dict(
            knee_decouple_ff=(args.knee_ff == "on"),
            hip_reaction=args.hip_reaction,
            stale_hip_hold=not args.no_stale_hip,
        )
        if args.hip_armature is not None:
            _kw["hip_pitch_armature"] = args.hip_armature
        robot_cfg.actuators = make_belt_actuators(**_kw)
        robot_cfg.actuators["hip_knee"].hold_decimation = DECIMATION
    print(f"  actuator variant: {args.actuator}"
          + (f" (knee_ff={args.knee_ff} stale_hip={not args.no_stale_hip} "
             f"hip_reaction={args.hip_reaction})" if args.actuator == "belt" else "")
          + (f"  hip_armature={args.hip_armature}" if args.hip_armature is not None
             else ""))

    robot = Articulation(cfg=robot_cfg)
    light = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.75, 0.75, 0.75))
    light.func("/World/Light", light)
    sim.reset()
    robot.reset()
    device = robot.device

    # ── push armature + 3-coeff joint friction into PhysX ─────────────
    # Same critical block as sysid_replay_sim.py:210-244 / rl_env_p6.py
    # _write_per_env_actuators_to_sim: cfg construction alone does NOT write
    # the dynamic/viscous friction model (or armature) into the simulation.
    # Reads the actuator's parsed [1, J] tensors, not the cfg fields — the
    # belt group's cfg values are per-joint dicts.
    print("[replay] writing armature + joint friction to PhysX:")
    for gname, act in robot.actuators.items():
        act.dynamic_friction[:] = torch.minimum(act.dynamic_friction, act.friction)
        robot.write_joint_armature_to_sim(act.armature, joint_ids=act.joint_indices)
        robot.write_joint_friction_coefficient_to_sim(
            act.friction,
            joint_dynamic_friction_coeff=act.dynamic_friction,
            joint_viscous_friction_coeff=act.viscous_friction,
            joint_ids=act.joint_indices,
        )
        for s, jn in enumerate(act.joint_names):
            print(f"    {gname:10s} {jn:10s} "
                  f"armature={float(act.armature[0, s]):.4e}  "
                  f"friction={float(act.friction[0, s]):.3f}"
                  f"/{float(act.dynamic_friction[0, s]):.3f}"
                  f"/{float(act.viscous_friction[0, s]):.4f}")

    # ── hw <-> sim joint index map ─────────────────────────────────────
    hw_to_sim = []
    for jn in HW_JOINT_ORDER:
        ids, _ = robot.find_joints([jn])
        hw_to_sim.append(ids[0])
    hw_to_sim_np = np.array(hw_to_sim)
    hw_to_sim_t = torch.tensor(hw_to_sim, device=device)
    print(f"  hw->sim index map: {hw_to_sim}")

    def joint_group(jn):
        """(actuator, local column) of a joint — group-layout agnostic."""
        for _gname, act in robot.actuators.items():
            if jn in act.joint_names:
                return act, act.joint_names.index(jn)
        raise KeyError(f"no actuator group covers joint '{jn}'")

    def apply_gains():
        """Set the hardware-run PD gains (sim joint space). Re-applied after
        every robot.reset() so no pass can run on stale/default gains."""
        for j in range(6):
            act, s = joint_group(HW_JOINT_ORDER[j])
            act.stiffness[0, s] = float(kp_hw[j])
            act.damping[0, s] = float(kd_hw[j])

    # ── per-joint prefilter coefficient vectors (sim order) ───────────
    nj = robot.num_joints
    omega_vec = torch.zeros(1, nj, device=device)
    zeta_vec = torch.ones(1, nj, device=device)
    for j in range(6):
        omega_vec[0, hw_to_sim[j]] = float(omega_grp[GROUP_OF[j]])
        zeta_vec[0, hw_to_sim[j]] = float(zeta_grp[GROUP_OF[j]])
    omega2_vec = omega_vec ** 2
    two_zw_vec = 2.0 * zeta_vec * omega_vec
    lag_mask = (omega_vec > 0.0).float()

    def to_sim(vec_hw):
        out = torch.zeros(1, nj, device=device)
        out[0, hw_to_sim_t] = torch.tensor(vec_hw, dtype=torch.float32, device=device)
        return out

    default_sim_t = to_sim(default_hw)
    pos0_sim_t = to_sim(pos_log[0])
    vel0_sim_t = to_sim(vel_log[0])

    # ── base orientation series ────────────────────────────────────────
    quats, ang_w = build_base_quats(proj_grav, ang_vel, DT_CTRL)
    # sanity: reconstructed quats must reproduce proj_grav
    R0 = _tilt_rotation(proj_grav[0])
    g_check = R0.T @ np.array([0.0, 0.0, -1.0])
    g_err = np.abs(g_check - proj_grav[0] / np.linalg.norm(proj_grav[0])).max()
    assert g_err < 1e-6, f"base-quat reconstruction inconsistent ({g_err:.2e})"
    g_mean = proj_grav.mean(axis=0)
    q_fixed = _rot_to_quat(_tilt_rotation(g_mean))
    print(f"  mean hang tilt from vertical: "
          f"{np.degrees(np.arccos(min(1.0, -g_mean[2] / np.linalg.norm(g_mean)))):.1f} deg\n")

    root_pos_t = torch.tensor([ROOT_POS], dtype=torch.float32, device=device)
    zero6 = torch.zeros(1, 6, device=device)

    def root_state_at(mode, tick, sub):
        """(pose[1,7], vel[1,6]) to pin the root with at this substep."""
        if mode == "fixed":
            q = q_fixed
            av = np.zeros(3)
        else:
            q0 = quats[tick]
            q1 = quats[min(tick + 1, n - 1)]
            q = _slerp(q0, q1, sub / DECIMATION)
            av = ang_w[tick]
        pose = torch.cat([root_pos_t,
                          torch.tensor([q], dtype=torch.float32, device=device)], dim=1)
        vel = zero6.clone()
        vel[0, 3:] = torch.tensor(av, dtype=torch.float32, device=device)
        return pose, vel

    # ── one replay pass ────────────────────────────────────────────────
    # hw-order index pairs for the belt coupling: (knee, its hip_pitch)
    KNEE_HIP_PAIRS = [(2, 1), (5, 4)]
    _coup_terms = {s.strip() for s in args.coupling_terms.split(",") if s.strip()}
    coup_knee_on = "knee" in _coup_terms
    coup_hip_on = "hip" in _coup_terms

    def run_pass(mode, delay, coupled=False):
        """coupled=True adds the real belt-drivetrain torques on top of the
        uncoupled DCMotor PD, as per-substep feedforward efforts.

        Firmware identity (motor coord m = (q_kn + c*q_hp)/G, kp_m = kp*G²):
          tau_kn_joint = kp*(tgt_kn - q_kn) - kd*qd_kn        [uncoupled PD]
                       + kp*c*(q_hp_send - q_hp) - kd*c*qd_hp [coupling extra]
          tau_hp_extra = c * tau_kn_joint                     [belt reaction]
        q_hp_send is the hip pitch position at target-computation time (tick
        start) — motor_bridge bakes it into the knee motor target, firmware
        holds it, so within-tick hip motion creates a phantom knee error.
        NOTE: the DCMotor clip sees PD+feedforward against the joint-frame
        envelope; at these logs' torque levels (p99 << saturation) the clip
        never binds, so the approximation is exact in practice."""
        # reset robot to the logged initial joint state
        pose, rvel = root_state_at(mode, 0, 0)
        robot.write_root_pose_to_sim(pose)
        robot.write_root_velocity_to_sim(rvel)
        robot.write_joint_state_to_sim(pos0_sim_t, vel0_sim_t)
        robot.reset()
        apply_gains()
        # clear any effort feedforward left by a previous (coupled) pass
        robot.set_joint_effort_target(torch.zeros(1, nj, device=device))
        robot.update(DT_PHY)

        # action-LPF state: the hardware filter carried over from HOLD; its
        # pre-run state is recoverable from tau_cmd[0]:
        #   filt0 = tau_j[0]/kp + kd*vel[0]/kp + pos[0]  (state AFTER tick-0 update)
        # In env terms f = (filt - default)/scale. Invert one LPF step for the
        # pre-tick state; with alpha=0 no state is needed.
        filt0_hw = tau_cmd_joint[0] / kp_hw + kd_hw * vel_log[0] / kp_hw + pos_log[0]
        f_after0 = (filt0_hw - default_hw) / scales_hw
        if alpha > 0.0:
            f_state = (f_after0 - (1.0 - alpha) * action[0]) / alpha
        else:
            f_state = np.zeros(6)
        f_state = f_state.copy()

        filt_hist = np.zeros((n, 6))
        # prefilter state: env resets pf_y to the spawn joint pos (pf_yd=0)
        pf_y = pos0_sim_t.clone()
        pf_yd = torch.zeros_like(pf_y)

        sim_pos = np.zeros((n, 6))
        sim_vel = np.zeros((n, 6))
        sim_tau = np.zeros((n, 6))   # applied_torque, joint frame, hw order

        for t in range(n):
            # teacher forcing: pin the joints to the LOGGED state at the top
            # of every tick — sim then predicts one 10 ms step ahead, and
            # sim_pos[t] is the prediction of pos_log[t+1] (compare shifted).
            if args.teacher_forcing and t > 0:
                robot.write_joint_state_to_sim(to_sim(pos_log[t]), to_sim(vel_log[t]))
                robot.update(DT_PHY)
            # 1. action LPF (env: rl_env_p6.py:1089)
            if alpha > 0.0:
                f_state = alpha * f_state + (1.0 - alpha) * action[t]
            else:
                f_state = action[t].copy()
            filt_hist[t] = f_state
            # 2. transport delay (hold the first sample; env buffer is
            #    zero-initialized at reset, but this run continues from HOLD)
            a_delayed = filt_hist[max(0, t - delay)]
            # 3. PD target (grace=0 -> center = default pose)
            u = to_sim(default_hw + a_delayed * scales_hw)
            # 4. 2nd-order actuator-lag prefilter at the control rate
            #    (env: _advance_motor_target, held across substeps)
            if use_prefilter:
                pf_ydd = omega2_vec * (u - pf_y) - two_zw_vec * pf_yd
                pf_yd = pf_yd + pf_ydd * DT_CTRL
                pf_y = pf_y + pf_yd * DT_CTRL
                tgt = lag_mask * pf_y + (1.0 - lag_mask) * u
            else:
                tgt = u
            robot.set_joint_position_target(tgt)
            if coupled:
                tgt_hw = tgt[0].cpu().numpy()[hw_to_sim_np]
                # hip pitch baked into the knee motor target at send time
                # (motor_bridge._sim_to_motor uses the tick-start read_state)
                jp_send = robot.data.joint_pos[0].cpu().numpy()[hw_to_sim_np]

            for sub in range(DECIMATION):
                if coupled:
                    jp = robot.data.joint_pos[0].cpu().numpy()[hw_to_sim_np]
                    jv = robot.data.joint_vel[0].cpu().numpy()[hw_to_sim_np]
                    ff = np.zeros(6)
                    for kn, hp in KNEE_HIP_PAIRS:
                        extra_kn = KNEE_COUPLING * (
                            kp_hw[kn] * (jp_send[hp] - jp[hp])
                            - kd_hw[kn] * jv[hp])
                        tau_kn = (kp_hw[kn] * (tgt_hw[kn] - jp[kn])
                                  - kd_hw[kn] * jv[kn] + extra_kn)
                        if coup_knee_on:
                            ff[kn] += extra_kn               # belt drag on knee PD
                        if coup_hip_on:
                            ff[hp] += KNEE_COUPLING * tau_kn  # belt reaction on hip
                    robot.set_joint_effort_target(to_sim(ff))
                pose, rvel = root_state_at(mode, t, sub)
                robot.write_root_pose_to_sim(pose)
                robot.write_root_velocity_to_sim(rvel)
                robot.write_data_to_sim()
                sim.step()
                # refresh joint state EVERY substep — else the DCMotor PD sees
                # state frozen at the control tick (sysid_replay_sim.py:368-376)
                robot.update(DT_PHY)

            jp = robot.data.joint_pos[0].cpu().numpy()
            jv = robot.data.joint_vel[0].cpu().numpy()
            ta = robot.data.applied_torque[0].cpu().numpy()
            sim_pos[t] = jp[hw_to_sim_np]
            sim_vel[t] = jv[hw_to_sim_np]
            sim_tau[t] = ta[hw_to_sim_np]

        return sim_pos, sim_vel, sim_tau

    # ── sweep ─────────────────────────────────────────────────────────
    results = {}
    if args.self_check:
        m0, d0 = base_modes[0], delays[0]
        p1, v1, _ = run_pass(m0, d0)
        p2, v2, _ = run_pass(m0, d0)
        dmax = np.abs(p1 - p2).max()
        # GPU PhysX is not bitwise deterministic (atomics ordering); ~1e-5 rad
        # pass-to-pass jitter is normal and 4 orders below the residuals we
        # measure. Only flag drift that could actually bias a verdict.
        print(f"[self-check] two identical passes ({m0}, delay={d0}): "
              f"max pos diff {dmax:.2e} rad -> "
              f"{'PASS' if dmax < 1e-3 else 'FAIL (state leaks between passes!)'}")

    for mode in base_modes:
        for delay in delays:
            variants = [False, True] if args.coupling else [False]
            for coupled in variants:
                sim_pos, sim_vel, sim_tau = run_pass(mode, delay, coupled=coupled)
                if args.teacher_forcing:
                    # sim_pos[t] was stepped FROM the logged state at tick t, so it
                    # predicts pos_log[t+1]: compare one-step-ahead, shifted.
                    err_p = sim_pos[:-1] - pos_log[1:]
                    err_v = sim_vel[:-1] - vel_log[1:]
                else:
                    err_p = sim_pos - pos_log
                    err_v = sim_vel - vel_log
                rmse_p = np.sqrt((err_p[w:] ** 2).mean(axis=0))
                rmse_v = np.sqrt((err_v[w:] ** 2).mean(axis=0))
                # torque RMSE, motor frame, best lag 0..2 ticks (sim_tau[t] is
                # the tick-t applied torque; the CAN tau_meas reply can lag a
                # tick — mirror coupling_torque_analysis.py's best-lag fit).
                # Compare against tau_MEAS: the log's motor_tau_cmd channel
                # under-reports the executed knee torque on post-FF logs
                # (balance_log.py's reconstruction predates the v_des patch).
                stm = sim_tau * JOINT_SIGN
                stm[:, KNEE_IDX[0]] *= KNEE_GEAR
                stm[:, KNEE_IDX[1]] *= KNEE_GEAR
                rmse_t = np.zeros(6)
                lag_t = np.zeros(6, dtype=int)
                for i in range(6):
                    cands = []
                    for k in range(3):
                        e = stm[w:n - k, i] - tau_meas_log[w + k:, i]
                        cands.append((float(np.sqrt(np.mean(e ** 2))), k))
                    rmse_t[i], lag_t[i] = min(cands)
                key = (mode + ("+coup" if coupled else ""), delay)
                results[key] = dict(sim_pos=sim_pos, sim_vel=sim_vel,
                                    sim_tau=sim_tau, rmse_p=rmse_p,
                                    rmse_v=rmse_v, rmse_t=rmse_t, lag_t=lag_t)
                print(f"  [{key[0]:11s} delay={delay}] pos RMSE  "
                      + "  ".join(f"{JOINT_LABELS[i]}={rmse_p[i]:.4f}" for i in range(6))
                      + f"   mean={rmse_p.mean():.4f}  tau={rmse_t.mean():.3f}Nm")

    # ── summary table ─────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print(f"{'base':<8s}{'delay':>6s}{'pos RMSE mean':>15s}{'vel RMSE mean':>15s}"
          f"{'tau RMSE mean':>15s}   per-joint pos RMSE")
    print("-" * 92)
    best = None
    for (mode, delay), r in results.items():
        mp, mv, mt = r["rmse_p"].mean(), r["rmse_v"].mean(), r["rmse_t"].mean()
        print(f"{mode:<12s}{delay:>6d}{mp:>15.4f}{mv:>15.4f}{mt:>15.3f}   "
              + " ".join(f"{x:.3f}" for x in r["rmse_p"]))
        if best is None or mp < best[2]:
            best = (mode, delay, mp)
    print("=" * 92)
    print(f"best: base={best[0]} delay={best[1]}  pos RMSE mean={best[2]:.4f} rad")
    print("  (June-22 hanging sysid fit baseline: ~0.018-0.024 rad pos RMSE)")

    if args.coupling:
        print("\ncoupled vs uncoupled (same base/delay), per-joint pos RMSE delta "
              "(negative = coupling model closer to real):")
        for (mode, delay) in list(results):
            if mode.endswith("+coup"):
                continue
            twin = (mode + "+coup", delay)
            if twin not in results:
                continue
            d_pj = results[twin]["rmse_p"] - results[(mode, delay)]["rmse_p"]
            d_m = d_pj.mean()
            print(f"  [{mode:6s} delay={delay}] mean {d_m:+.4f}  "
                  + "  ".join(f"{JOINT_LABELS[i]}={d_pj[i]:+.4f}" for i in range(6)))

    # ── plots for the best config ─────────────────────────────────────
    variant = "_" + args.actuator
    if args.actuator == "belt":
        if args.knee_ff == "off":
            variant += "_ffoff"
        if args.no_stale_hip:
            variant += "_nostale"
        if args.hip_reaction:
            variant += "_hiprx"
    if args.hip_armature is not None:
        variant += f"_arm{args.hip_armature:g}"
    if args.coupling:
        variant += "_coup"
    if args.no_prefilter:
        variant += "_nopf"
    if args.no_lpf:
        variant += "_nolpf"
    if args.teacher_forcing:
        variant += "_tf"
    stem = os.path.splitext(args.npz)[0] + "_simreplay" + variant
    rbest = results[(best[0], best[1])]
    t_axis = np.arange(n) * DT_CTRL
    sim_tau_motor = rbest["sim_tau"] * JOINT_SIGN
    sim_tau_motor[:, KNEE_IDX[0]] *= KNEE_GEAR
    sim_tau_motor[:, KNEE_IDX[1]] *= KNEE_GEAR

    with PdfPages(stem + ".pdf") as pdf:
        for title, sim_arr, log_arr, unit in [
            ("joint position", rbest["sim_pos"], pos_log, "rad"),
            ("joint velocity", rbest["sim_vel"], vel_log, "rad/s"),
        ]:
            fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
            for i in range(6):
                ax = axes[i % 3][i // 3]
                ax.plot(t_axis, log_arr[:, i], "b-", lw=0.9, label="real")
                ax.plot(t_axis, sim_arr[:, i], "r--", lw=0.9, label="sim replay")
                if title.startswith("joint position"):
                    ax.plot(t_axis, target_log[:, i], "k:", lw=0.5, alpha=0.4,
                            label="target (pre-LPF)")
                rm = np.sqrt(((sim_arr - log_arr)[w:, i] ** 2).mean())
                ax.set_title(f"{JOINT_LABELS[i]}  RMSE={rm:.4f}", fontsize=9)
                if i == 0:
                    ax.legend(fontsize=7)
            fig.suptitle(f"{title}: sim (training chain, base={best[0]}, "
                         f"delay={best[1]}) vs real  [{unit}]")
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
        for i in range(6):
            ax = axes[i % 3][i // 3]
            ax.plot(t_axis, tau_meas_log[:, i], "b-", lw=0.8, label="tau_meas (real)")
            ax.plot(t_axis, sim_tau_motor[:, i], "r--", lw=0.8,
                    label="applied_torque (sim)")
            ax.set_title(JOINT_LABELS[i], fontsize=9)
            if i == 0:
                ax.legend(fontsize=7)
        fig.suptitle("torque, motor frame [Nm] (informational)")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig); plt.close(fig)

    sv = {"base_mode": np.asarray(best[0]), "delay": np.asarray(best[1]),
          "kp": kp_hw, "kd": kd_hw, "alpha": np.asarray(alpha),
          "time": t_axis}
    for (mode, delay), r in results.items():
        tag = f"{mode}_d{delay}"
        sv[f"{tag}__sim_pos"] = r["sim_pos"]
        sv[f"{tag}__sim_vel"] = r["sim_vel"]
        sv[f"{tag}__sim_tau"] = r["sim_tau"]
        sv[f"{tag}__rmse_pos"] = r["rmse_p"]
        sv[f"{tag}__rmse_vel"] = r["rmse_v"]
        sv[f"{tag}__rmse_tau"] = r["rmse_t"]
        sv[f"{tag}__tau_lag"] = r["lag_t"]
    np.savez(stem + ".npz", **sv)
    with open(stem + ".json", "w") as f:
        json.dump({f"{m}_d{dl}": {"rmse_pos": r["rmse_p"].tolist(),
                                  "rmse_vel": r["rmse_v"].tolist(),
                                  "rmse_tau": r["rmse_t"].tolist(),
                                  "tau_lag": r["lag_t"].tolist()}
                   for (m, dl), r in results.items()}
                  | {"best": {"base": best[0], "delay": best[1],
                              "rmse_pos_mean": best[2]},
                     "gains_source": gains_src,
                     "kp": kp_hw.tolist(), "kd": kd_hw.tolist(),
                     "actuator": args.actuator,
                     "knee_ff": args.knee_ff,
                     "stale_hip": not args.no_stale_hip,
                     "hip_reaction": args.hip_reaction,
                     "hip_armature": args.hip_armature,
                     "teacher_forcing": args.teacher_forcing}, f, indent=2)
    print(f"\n  plots -> {stem}.pdf\n  data  -> {stem}.npz / {stem}.json")
    sys.stdout.flush()

    # sim_app.close() intermittently hangs on Windows (the 20260805 run sat
    # for hours post-results). Results are already on disk; force-exit if
    # close() doesn't return promptly.
    t = threading.Timer(60.0, lambda: os._exit(0))
    t.daemon = True
    t.start()
    sim_app.close()


if __name__ == "__main__":
    main()
