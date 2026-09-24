"""
sysid_replay_sim.py — Replay sysid experiments inside IsaacLab.

Spawns the robot with its root body fixed in the air, applies the exact
same p_des commands through the DCMotor actuator model + PhysX link
dynamics, and overlays the sim response on the real recording.

This is the proper test of the full sim pipeline (URDF inertias + gravity
+ DCMotor PD/friction/envelope) versus reality, without the lumped-model
approximation of sysid_fit.py.

ATTACHMENT POINT NOTE:
    fix_root_link fixes the root body (torso2hip). The real tether is on
    the motor chassis (chasis_motor2motor), one joint (hip_roll) below
    the root. For hip_pitch and knee tests this difference is negligible
    (the hip_roll PD keeps the chassis nearly fixed). For hip_roll tests,
    the sim moves the chassis while the real robot moves the torso against
    the tether — the effective inertia may differ. Keep this in mind when
    interpreting hip_roll overlays.

INDEX REMAPPING NOTE:
    IsaacLab orders joints by type (hip_roll1/2, hip_pitch1/2, knee1/2)
    while hardware orders by leg (hr1, hp1, kn1, hr2, hp2, kn2). We use
    find_joints() to build an explicit hw↔sim mapping so targets, readbacks,
    and default poses all use the correct indices.

Usage (run from the directory that contains rl_env_cfg_p6.py):
    python sysid_replay_sim.py sysid_leg1_*.npz --headless
"""

from __future__ import annotations
import argparse, json, sys, os

# ── Launch IsaacLab (must happen before other isaaclab imports) ─────
from isaaclab.app import AppLauncher
cli = argparse.ArgumentParser()
cli.add_argument("npz", help="recorded sysid .npz")
AppLauncher.add_app_launcher_args(cli)
args = cli.parse_args()
launcher = AppLauncher(args)
sim_app = launcher.app

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation

sys.path.insert(0, os.getcwd())
from RL.tasks.direct.standup_curriculum.rl_env_cfg_p6 import RlEnvCfg         # reuse the robot / actuator cfg

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JTYPE = ["hip_roll", "hip_pitch", "knee",
         "hip_roll", "hip_pitch", "knee"]
GROUP  = JTYPE                                # actuator-group name per joint
SUB    = [0, 0, 0, 1, 1, 1]                  # index inside the group

# Gains matching sysid_experiment.py
EXCITE = {"hip_roll": (6.0, 0.3), "hip_pitch": (8.0, 0.5), "knee": (8.0, 0.5)}
HOLD   = {"hip_roll": (15.0, 0.8), "hip_pitch": (20.0, 1.2), "knee": (20.0, 1.2)}

# ── Friction tuning (sysid iteration) ────────────────────────
# Edit values here, rerun replay, compare RMSE.
# Once converged, copy final numbers back into rl_env_cfg_p6.py.
#
# Your trajectory SHAPES show the cfg friction is a bit too high:
#   * steps settle SHORT of the command (sim 0.31 vs cmd 0.35) — a static
#     dead-band of mu_static/Kp ~ 0.4/8 ~ 0.05 rad, which matches.
#   * hr triangle shows visible stair-steps (stick-slip) and the knee triangle
#     has flat platforms at the reversals — both are static>>dynamic stiction.
# Hardware reaches (and overshoots) the command, so its breakaway is small.
# So: LOWER static, and CLOSE the static/dynamic gap (set dynamic ~ static) to
# kill the stick-slip; keep viscous tiny (it damps the overshoot you want).
FRICTION_TUNE = {
    "hip_roll":  dict(friction=0.12, dynamic_friction=0.10, viscous_friction=0.003),
    "hip_pitch": dict(friction=0.12, dynamic_friction=0.10, viscous_friction=0.003),
    "knee":      dict(friction=0.06, dynamic_friction=0.05, viscous_friction=0.003),
}
# Overshoot knob: with the command lag in place, friction now GATES overshoot
# (high static = no overshoot, undershoot; low static = the loop rings through).
# Keep static/dynamic low + roughly equal (kills dead-band + stick-slip), and
# use VISCOUS to dial the overshoot MAGNITUDE down to match — viscous is smooth
# velocity damping (like extra Kd), so it trims ringing without adding stick-slip.
# If sim overshoots more than the real ~3%, nudge viscous up (e.g. 0.01-0.03).

# ── Inertia tuning (the knob that ACTUALLY moves step rise-time) ──────
# Multiplier on the cfg armature (reflected rotor inertia). Friction barely
# touches a step rise because Kp*step (~2.4 Nm for a 0.3 rad knee step) dwarfs
# friction (~0.5 Nm) — the controller just powers through it. Rise time is set
# by inertia and Kd, so if sim leads real on a step, raise armature here.
# Knee reflected inertia = I_rotor / gear^2 = I_rotor*(30/22)^2; an error in
# that reflection (or an optimistic datasheet I_rotor) shows up as sim-too-fast.
ARMATURE_MULT = {
    "knee": 1.0,        # try 1.5, 2.0, 3.0 and watch the step rise slow down
    # "hip_pitch": 1.0,
    # "hip_roll": 1.0,
}

# ── Command latency (the OTHER reason sim leads real) ────────────────
# The real loop has command + serial + current-loop lag the replay omits
# (100 Hz cmd over USB-serial, ~1 ms/frame pacing, motor MIT loop). The
# training env models this as a 1–7 step action delay; the replay applies
# torque instantly. Delay the replayed command by N control steps (10 ms each)
# to push the sim response later and line it up with the real recording.
# 0 = off. Start at 1 if a clean step still leads real after armature is set.
CMD_DELAY_STEPS = 0

# ── Actuator bandwidth lag (SECOND-ORDER command prefilter) ──────────
# The actuator's closed-loop position response is a 2nd-order system, and the
# diagnostics pin it down:
#   * bulk lag is UNIFORM across joints  -> it's actuator-level (the prefilter),
#     not the rigid loop (which is inertia-dependent).
#   * the step rises gradually from t=0 (no flat dead-time) -> a lag, not a
#     pure transport delay.
#   * unmuted, the rigid loops already overshoot to ~the real peak -> inertia /
#     zeta are correct; we just need the bulk lag without smoothing the ring off.
# A 1st-order low-pass smooths the ring away (overdamped, zeta_a=inf). A 2nd-
# order prefilter with zeta_a<1 delays the command AND keeps a small overshoot,
# which the per-joint loops then shape (hips ring clearly, knee barely — matches
# "knee overshoots a hair only at +0.6").
#   y'' + 2*zeta_a*omega_a*y' + omega_a^2*y = omega_a^2 * u
# Per-JOINT now: each joint's actuator+load closed-loop has its own damping
# (hip_roll is an EL05; hip_pitch/knee are RS01; the knee adds a belt + load).
# A single zeta couldn't satisfy "hip_roll wants more ring, knee wants less".
# TUNING:  omega -> rise time / bulk lag (~14-16 rad/s for ~0.2 s rise).
#          zeta  -> overshoot: LOWER = more ring. ~0.50->16%, 0.65->7%, 0.80->2%
#          (per-joint hip/knee overshoot also depends on each joint's own loop).
# Set a group's omega to 0 to disable the prefilter on that group.
ACT_LAG_OMEGA = {"hip_roll": 13.15, "hip_pitch": 14.08, "knee": 15.94}
ACT_LAG_ZETA  = {"hip_roll": 0.26,  "hip_pitch": 0.54,  "knee": 0.68}



# Hardware joint order (must match sysid_experiment.py)
_HW_JOINT_ORDER = [
    "hip_roll1", "hip_pitch1", "knee1",
    "hip_roll2", "hip_pitch2", "knee2",
]


# ══════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════
def load_trials(path):
    d = np.load(path, allow_pickle=True)
    names   = list(d["trial_names"])
    configs = json.loads(str(d["trial_configs"]))
    dpos    = d["default_pos"]
    trials  = {}
    for nm, cfg in zip(names, configs):
        trials[nm] = dict(
            t=d[f"{nm}__time"], p_des=d[f"{nm}__p_des"],
            pos=d[f"{nm}__pos"], vel=d[f"{nm}__vel"],
            kp=d[f"{nm}__kp"], kd=d[f"{nm}__kd"], cfg=cfg,
        )
    all_idx = [tr["cfg"]["idx"] for tr in trials.values()]
    off = 0 if min(all_idx) < 3 else 3
    return trials, dpos, off


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
def main():
    trials, dpos, off = load_trials(args.npz)
    print(f"Loaded {len(trials)} trials  (leg {'1' if off==0 else '2'})\n")

    # ── build sim + robot ──────────────────────────────────────
    env_cfg  = RlEnvCfg()
    sim = SimulationContext(env_cfg.sim)

    robot_cfg = env_cfg.robot_cfg
    robot_cfg.prim_path = "/World/Robot"
    # Fix the root body in the air (approximates tether / pickup).
    # If your IsaacLab version doesn't have fix_root_link, see the
    # fallback comment inside the physics loop.
    robot_cfg.spawn.articulation_props.fix_root_link = True

    # ── match sim PhysX clamps to real hardware limits ────────
    for act_cfg in robot_cfg.actuators.values():
        act_cfg.effort_limit_sim   = act_cfg.effort_limit
        act_cfg.velocity_limit_sim = act_cfg.velocity_limit

    # ── apply friction tuning overrides ───────────────────────
    for grp, overrides in FRICTION_TUNE.items():
        act_cfg = robot_cfg.actuators[grp]
        for attr, val in overrides.items():
            setattr(act_cfg, attr, val)
        print(f"  [tune] {grp}: {overrides}")

    # ── actuator bandwidth modeled as a 2nd-order command prefilter ──
    # (No actuator swap. The torque-lag operator was wrong — it rings without
    #  delaying. The lag is a 2nd-order low-pass on the position target in the
    #  replay loop; see ACT_LAG_OMEGA / ACT_LAG_ZETA and the pf_y/pf_yd update.)

    robot = Articulation(cfg=robot_cfg)

    # Optional dome light so the viewer isn't pitch black
    light = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.75, 0.75, 0.75))
    light.func("/World/Light", light)

    sim.reset()
    robot.reset()

    device = robot.device

    # ── push armature + 3-coeff joint friction into PhysX ─────
    # CRITICAL: creating the Articulation from cfg does NOT reliably write
    # the joint-friction model to PhysX. The basic init path writes (at
    # most) the single static `friction` coefficient; the dynamic and
    # viscous terms are part of PhysX's newer joint-friction model and are
    # only ever applied in the training env through an explicit
    # write_joint_friction_coefficient_to_sim() call (see rl_env_p6.py
    # _write_per_env_actuators_to_sim). Armature is in the same boat.
    #
    # Without this block, editing FRICTION_TUNE above changes the cfg object
    # but never touches the simulation — the friction knob is disconnected,
    # which is why tuning it does nothing to the RMSE. We replicate the env's
    # exact write so the replay matches what the policy trains against.
    print("\n[replay] writing armature + joint friction to PhysX:")
    for gname, act in robot.actuators.items():
        acfg     = robot_cfg.actuators[gname]
        static_f = float(acfg.friction)
        dyn_f    = min(float(acfg.dynamic_friction), static_f)   # PhysX: mu_d <= mu_s
        visc_f   = float(acfg.viscous_friction)
        arm      = float(acfg.armature) * float(ARMATURE_MULT.get(gname, 1.0))

        act.friction[:]         = static_f
        act.dynamic_friction[:] = dyn_f
        act.viscous_friction[:] = visc_f
        act.armature[:]         = arm

        robot.write_joint_armature_to_sim(act.armature, joint_ids=act.joint_indices)
        robot.write_joint_friction_coefficient_to_sim(
            act.friction,
            joint_dynamic_friction_coeff=act.dynamic_friction,
            joint_viscous_friction_coeff=act.viscous_friction,
            joint_ids=act.joint_indices,
        )
        print(f"    {gname:10s} armature={arm:.4e} (x{ARMATURE_MULT.get(gname, 1.0):.2f})  "
              f"friction(static/dyn/visc)={static_f:.3f}/{dyn_f:.3f}/{visc_f:.4f}")

    # ── build hw ↔ sim index mapping ──────────────────────────
    # IsaacLab internal order (by type): [hr1,hr2, hp1,hp2, kn1,kn2]
    # Hardware order (by leg):           [hr1,hp1,kn1, hr2,hp2,kn2]
    # hw_to_sim[i] = sim index of hardware joint i
    hw_to_sim = []
    for jn in _HW_JOINT_ORDER:
        ids, _ = robot.find_joints([jn])
        hw_to_sim.append(ids[0])
    hw_to_sim_np = np.array(hw_to_sim)
    hw_to_sim_t  = torch.tensor(hw_to_sim, device=device)

    print(f"hw→sim index map: {hw_to_sim}")
    for gname in ["hip_roll", "hip_pitch", "knee"]:
        act = robot.actuators[gname]
        print(f"  {gname}: joint_names={act.joint_names}, "
              f"joint_indices={act.joint_indices}")

    # ── default pose in sim ordering ──────────────────────────
    default_sim  = torch.zeros(1, robot.num_joints, dtype=torch.float32, device=device)
    default_sim[0, hw_to_sim_t] = torch.tensor(dpos, dtype=torch.float32, device=device)
    zero_vel_sim = torch.zeros_like(default_sim)

    DECIMATION = env_cfg.decimation          # 5
    DT_PHY     = float(env_cfg.sim.dt)       # 0.002
    DT_CTRL    = DECIMATION * DT_PHY          # 0.01 s (100 Hz command rate)
    # 2nd-order command prefilter coefficients, per joint (sim order).
    # Build [1, num_joints] omega/zeta vectors from the per-group dicts.
    omega_vec = torch.zeros(1, robot.num_joints, dtype=torch.float32, device=device)
    zeta_vec  = torch.ones (1, robot.num_joints, dtype=torch.float32, device=device)
    for gname in ["hip_roll", "hip_pitch", "knee"]:
        ids = robot.actuators[gname].joint_indices
        omega_vec[0, ids] = float(ACT_LAG_OMEGA[gname])
        zeta_vec [0, ids] = float(ACT_LAG_ZETA[gname])
    USE_LAG = bool((omega_vec > 0.0).any())
    omega2_vec = omega_vec ** 2
    two_zw_vec = 2.0 * zeta_vec * omega_vec
    # Joints with omega==0 -> passthrough (handled by mask below).
    lag_mask = (omega_vec > 0.0).float()
    if USE_LAG:
        print("Command prefilter (2nd order), per joint:")
        for gname in ["hip_roll", "hip_pitch", "knee"]:
            w, z = ACT_LAG_OMEGA[gname], ACT_LAG_ZETA[gname]
            print(f"  {gname:10s} omega={w:5.1f} rad/s ({w/2/np.pi:.2f} Hz)  zeta={z:.2f}")
        if float(omega_vec.max()) * DT_CTRL > 1.0:
            print(f"  [warn] omega*dt={float(omega_vec.max())*DT_CTRL:.2f} large; "
                  f"prefilter may be inaccurate/unstable")

    print(f"\nJoint names : {robot.data.joint_names}")
    print(f"Joint shape : {robot.data.joint_pos.shape}")
    print(f"Decimation  : {DECIMATION}   dt_phy={DT_PHY}\n")

    results = {}

    for name, trial in trials.items():
        cfg       = trial["cfg"]
        idx       = cfg["idx"]
        hip_float = cfg.get("hip_float")
        p_des_rec = trial["p_des"]           # (N, 6) in hw order
        n_steps   = len(trial["t"])

        print(f"  {name:30s}  joint {idx}  {n_steps} steps ... ", end="", flush=True)

        # ── reset to default ──────────────────────────────────
        robot.write_joint_state_to_sim(default_sim, zero_vel_sim)
        robot.reset()
        # Init the 2nd-order command prefilter at the pre-trial pose (zero
        # velocity) so the step rises gradually from default instead of jumping.
        pf_y  = default_sim.clone()              # filtered position target
        pf_yd = torch.zeros_like(default_sim)    # filtered target velocity

        # ── set per-joint gains ───────────────────────────────
        # Gains go through actuator group names → correct regardless
        # of internal joint ordering.
        for j in range(6):
            act = robot.actuators[GROUP[j]]
            s   = SUB[j]
            jt  = JTYPE[j]
            if j == idx:
                kp, kd = EXCITE[jt]
            elif hip_float is not None and j == hip_float:
                kp, kd = 0.0, 0.3
            else:
                kp, kd = HOLD[jt]
            act.stiffness[0, s] = kp
            act.damping[0, s]   = kd

        # ── replay the recorded command sequence ──────────────
        sim_pos_buf, sim_vel_buf = [], []

        for step in range(n_steps):
            # Optional command latency: feed the command from CMD_DELAY_STEPS
            # ago (holds p_des[0] for the first few steps). Models the real
            # comm + current-loop lag the replay otherwise skips, shifting the
            # sim response later to line up with the recording.
            cmd_step = max(0, step - CMD_DELAY_STEPS)
            # Reorder target from hw order → sim order
            target_hw  = torch.tensor(
                p_des_rec[cmd_step], dtype=torch.float32, device=device
            )
            target_sim = torch.zeros(
                robot.num_joints, dtype=torch.float32, device=device
            )
            target_sim[hw_to_sim_t] = target_hw
            # 2nd-order command prefilter (per joint, actuator closed-loop):
            #   y'' = w^2 (u - y) - 2 z w y'   integrated at the control rate.
            # omega==0 joints pass the raw command through via lag_mask.
            if USE_LAG:
                u = target_sim.unsqueeze(0)
                pf_ydd = omega2_vec * (u - pf_y) - two_zw_vec * pf_yd
                pf_yd  = pf_yd + pf_ydd * DT_CTRL
                pf_y   = pf_y  + pf_yd  * DT_CTRL
                out_target = lag_mask * pf_y + (1.0 - lag_mask) * u
                robot.set_joint_position_target(out_target)
            else:
                robot.set_joint_position_target(target_sim.unsqueeze(0))

            for sub in range(DECIMATION):
                robot.write_data_to_sim()
                if step == 0 and sub == 0:
                    act = robot.actuators["knee"]
                    print(f"  KNEE GAINS CHECK: Kp={act.stiffness}, Kd={act.damping}")
                sim.step()
                # Refresh joint state EVERY physics substep. The DCMotor model
                # recomputes torque from robot.data.joint_pos/joint_vel inside
                # write_data_to_sim(); if we don't update() between substeps the
                # controller sees state frozen at the start of the 10 ms control
                # tick and effectively runs its PD at 100 Hz instead of 500 Hz.
                # That is NOT what the policy trains against — DirectRLEnv.step()
                # calls scene.update() every substep — and the stale velocity term
                # is exactly what makes held joints drift/oscillate ("hip roll
                # spinning", "all joints moving") and the excited joint overshoot.
                robot.update(DT_PHY)
                # ── fallback if fix_root_link is not available ──
                # Uncomment the next two lines to force-clamp the root:
                # robot.write_root_pose_to_sim(root_pose)
                # robot.write_root_velocity_to_sim(zero_root_vel)
            # (no robot.update() here — state is already current from the
            #  last per-substep update above)

            # Reorder readback from sim order → hw order
            sim_pos_all = robot.data.joint_pos[0].cpu().numpy()
            sim_pos_buf.append(sim_pos_all[hw_to_sim_np].copy())
            sim_vel_all = robot.data.joint_vel[0].cpu().numpy()
            sim_vel_buf.append(sim_vel_all[hw_to_sim_np].copy())

        sim_pos = np.array(sim_pos_buf)    # (N, 6) in hw order
        sim_vel = np.array(sim_vel_buf)
        rmse = np.sqrt(np.mean(
            (sim_pos[:, idx] - trial["pos"][:, idx])**2))
        print(f"RMSE = {rmse:.4f}")

        results[name] = dict(
            sim_pos=sim_pos, sim_vel=sim_vel,
            real_pos=trial["pos"], real_vel=trial["vel"],
            p_des=p_des_rec, t=trial["t"], idx=idx,
        )

    # ── group trials by joint type for plotting ───────────────
    plot_groups: dict[str, list[str]] = {}
    for name, r in results.items():
        jt = JTYPE[r["idx"]]
        tag = f"{jt}_float" if "float" in name else jt
        plot_groups.setdefault(tag, []).append(name)

    for tag, names in plot_groups.items():
        n    = len(names)
        cols = min(n, 4)
        rows = max(1, (n + cols - 1) // cols)
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(4.5*cols, 3*rows), squeeze=False)
        fig.suptitle(
            f"{tag}: IsaacLab (red) vs hardware (blue)", fontsize=12)

        for i, nm in enumerate(names):
            ax = axes[i // cols][i % cols]
            r  = results[nm]
            j  = r["idx"]
            ax.plot(r["t"], r["real_pos"][:, j], "b-",  lw=0.9, label="real")
            ax.plot(r["t"], r["sim_pos"][:, j],  "r--", lw=0.9, label="sim")
            ax.plot(r["t"], r["p_des"][:, j],    "k:",  lw=0.4,
                    alpha=0.35, label="cmd")
            rmse = np.sqrt(np.mean(
                (r["sim_pos"][:, j] - r["real_pos"][:, j])**2))
            ax.set_title(f"{nm}  RMSE={rmse:.4f}", fontsize=8)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.legend(fontsize=7)

        for k in range(n, rows * cols):
            axes[k // cols][k % cols].set_visible(False)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf = f"sysid_replay_{tag}.pdf"
        fig.savefig(pdf, dpi=150)
        plt.close(fig)
        print(f"  -> {pdf}")

    # ── summary table ─────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"{'trial':<28s} {'joint':>5s} {'RMSE':>8s}")
    print("-" * 55)
    for nm in sorted(results):
        r = results[nm]
        j = r["idx"]
        rmse = np.sqrt(np.mean(
            (r["sim_pos"][:, j] - r["real_pos"][:, j])**2))
        print(f"{nm:<28s} {j:>5d} {rmse:8.4f}")
    print("=" * 55)

    # ── save sim trajectories ─────────────────────────────────
    out = args.npz.replace(".npz", "_replay.npz")
    sv = {}
    for nm, r in results.items():
        sv[f"{nm}__sim_pos"] = r["sim_pos"]
        sv[f"{nm}__sim_vel"] = r["sim_vel"]
    np.savez(out, **sv)
    print(f"\nSim data -> {out}")

    sim_app.close()


if __name__ == "__main__":
    main()