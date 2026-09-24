"""
lqr_balance_controller_com.py — LQR Balance with COM-over-Foot Objective
Run inside Isaac Sim's Script Editor (Window > Script Editor)

=== KEY CHANGE FROM FIXED-POSE LQR ===

The original controller penalizes deviations from a single equilibrium
pose [θ1_eq, θ2_eq, θ3_eq]. This has two problems:

  1. If the FK model is even slightly wrong, the "balanced" pose isn't
     actually balanced in sim → steady-state lean → fall.

  2. It wastes control effort fighting perturbations that redistribute
     joint angles but DON'T affect balance (e.g., torso tilts back while
     knee bends forward by the right amount to keep COM over foot).

THE FIX: Instead of Q = diag([w_θ1, w_θ2, ...]), we construct Q from
the COM-over-foot Jacobian. We compute a vector h such that:

    COM_x − foot_x  ≈  h · δθ     (linearized)

Then penalize  (h · δθ)²  heavily — the BALANCE objective — while only
lightly regularizing individual angles (POSTURE). This lets the robot
freely redistribute posture as long as COM stays over the foot.

Everything else (dynamics model, Isaac Sim integration, state measurement)
is unchanged.

=== APPROACH ===

1. Model the robot as a 3-link planar pendulum in the sagittal (XZ) plane:
     Link 1 (shin):  foot contact → knee        [UNACTUATED pivot at foot]
     Link 2 (thigh): knee → hip_pitch            [actuated: knee joint]
     Link 3 (upper): hip_pitch → upper body COM  [actuated: hip_pitch joint]

2. Both legs are treated symmetrically (same commands, doubled mass/inertia).

3. The state vector is 6D:  x = [θ1, θ2, θ3, θ̇1, θ̇2, θ̇3]
   where θi are absolute angles from vertical in the sagittal plane.

4. The control vector is 2D: u = [τ_knee, τ_hip_pitch]
   These are torques between adjacent links.

5. We linearize the dynamics around the standing equilibrium and solve
   the continuous-time algebraic Riccati equation (CARE) to get K.

6. The Q matrix is built from the COM Jacobian, NOT from per-angle weights:
     Q = w_bal * h_pos @ h_pos^T        (penalize COM-foot offset)
       + w_vel * h_vel @ h_vel^T        (penalize COM velocity)
       + diag([w_reg, ...])             (light posture regularization)

7. At each physics step:
     - Measure absolute link angles from USD world transforms
     - Compute angular velocities via windowed differentiation
     - Compute δx = x - x_equilibrium
     - Compute δτ = -K @ δx
     - Convert torques to joint position offsets: δq = δτ / kp
     - Apply: q_target = q_default + δq

=== USAGE ===

1. Paste into Script Editor
2. Press Run
3. Press PLAY in the toolbar
4. Monitor console output — tune w_bal / w_vel / w_reg / R as needed
"""

import numpy as np
import math
import asyncio
from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema, Sdf, PhysicsSchemaTools
import omni.usd
import omni.kit.app
import omni.kit.commands
import omni.timeline
import omni.physx


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 1: CONFIGURATION                                      ║
# ╚══════════════════════════════════════════════════════════════════╝

USD_PATH = r"<repo>\urdf_onshape_export\bigthigh-urdf\robot\robot.usd"
SPAWN_HEIGHT = 0.42
TORSO_PRIM_PATH = "/World/Robot/torso2hip"

# Default standing pose (will be overwritten by balanced-pose search)
DEFAULT_POSE = {
    "hip_roll1":  0.0,
    "hip_pitch1": 0.05,
    "knee1":      0.0,
    "hip_roll2":  0.0,
    "hip_pitch2": -0.05,
    "knee2":      0.0,
}

# Motor PD parameters — these are the INNER loop gains.
# The LQR provides position offsets ON TOP of the default pose.
JOINT_PARAMS = {
    "hip_roll1":  {"stiffness": 8.0,   "damping": 0.5, "effort": 6.0},
    "hip_roll2":  {"stiffness": 8.0,   "damping": 0.5, "effort": 6.0},
    "hip_pitch1": {"stiffness": 80.0,  "damping": 3.0, "effort": 17.0},
    "hip_pitch2": {"stiffness": 80.0,  "damping": 3.0, "effort": 17.0},
    "knee1":      {"stiffness": 60.0,  "damping": 2.0, "effort": 23.2},
    "knee2":      {"stiffness": 60.0,  "damping": 2.0, "effort": 23.2},
}

# Link prim paths (must match your USD stage hierarchy)
LINK_PRIM_PATHS = {
    "torso2hip":                         "/World/Robot/torso2hip",
    "chasis_motor2motor":                "/World/Robot/chasis_motor2motor",
    "thigh_outer":                       "/World/Robot/thigh_outer",
    "shin_inner":                        "/World/Robot/shin_inner",
    "lid_bearing_outer___boss_extrude3": "/World/Robot/lid_bearing_outer___boss_extrude3",
    "stand_steel_00___fillet1":          "/World/Robot/stand_steel_00___fillet1",
    "nail___fillet1":                    "/World/Robot/nail___fillet1",
}

# Safety limits
MAX_TILT_DEG    = 45.0
MIN_BASE_HEIGHT = 0.08
PRINT_EVERY_N   = 5

# ╔══════════════════════════════════════════════════════════════════╗
# ║  COM-OVER-FOOT LQR TUNING                                      ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# These replace the old per-angle Q diagonal.
#
# TUNING GUIDE:
#   w_balance  — how aggressively to keep COM over foot.
#                Increase → tighter balance, more aggressive torques.
#                Decrease → looser balance, gentler.
#
#   w_com_vel  — how aggressively to damp COM horizontal velocity.
#                Increase → more damping, less overshoot.
#                Decrease → faster response, more oscillation.
#
#   w_posture  — how tightly to hold individual joint angles at equilibrium.
#                Increase → stiffer posture (closer to fixed-pose LQR).
#                Decrease → freer posture, robot finds its own balance config.
#                Set to 0 for pure COM-tracking with no posture preference.
#
#   w_vel_reg  — damping on individual joint velocities (prevents jitter).
#
#   R          — control cost. Increase for gentler actuator use.

W_BALANCE  = 5000.0   # COM-foot offset penalty
W_COM_VEL  = 5000.0    # COM velocity penalty
W_POSTURE  = 5.0       # individual angle regularization (LIGHT)
W_VEL_REG  = 2.0       # individual velocity regularization

R_KNEE = 8000.0        # knee torque cost
R_HIP  = 8000.0        # hip torque cost


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 2: FORWARD KINEMATICS FROM URDF                       ║
# ╚══════════════════════════════════════════════════════════════════╝

def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float64)

def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float64)

def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)

def _rpy(r, p, y):
    return _rz(y) @ _ry(p) @ _rx(r)

def _tf(xyz, rpy):
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = _rpy(*rpy)
    T[:3, 3] = xyz
    return T

def _tf_rz(q):
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = _rz(q)
    return T

def _pos(T):
    return T[:3, 3].copy()

def _rot(T):
    return T[:3,:3].copy()


def compute_fk(pose):
    """
    Compute world-frame positions of all joints and link COMs at a given pose.
    """
    T_torso = np.eye(4, dtype=np.float64)
    T_torso[2, 3] = SPAWN_HEIGHT

    # Leg 1
    T_hr1_origin = _tf([-0.070392, 0.0800218, -0.0315], [1.5708, 0, 1.5708])
    T_chassis1 = T_torso @ T_hr1_origin @ _tf_rz(pose["hip_roll1"])

    T_hp1_origin = _tf([-0.0415, 0, -0.071], [0, 1.5708, 0])
    T_hp1_joint = T_chassis1 @ T_hp1_origin
    T_thigh1 = T_hp1_joint @ _tf_rz(pose["hip_pitch1"])

    T_k1_origin = _tf([-0.0592463, -0.172173, 0.0969], [0, 0, 1.72565])
    T_k1_joint = T_thigh1 @ T_k1_origin
    T_shin1 = T_k1_joint @ _tf_rz(pose["knee1"])

    # Leg 2
    T_hr2_origin = _tf([-0.070392, -0.0799782, -0.0315], [-3.14159, -1.5708, 0])
    T_chassis2 = T_torso @ T_hr2_origin @ _tf_rz(pose["hip_roll2"])

    T_hp2_origin = _tf([0, -0.0415, -0.071], [-1.5708, 0.908015, 0])
    T_hp2_joint = T_chassis2 @ T_hp2_origin
    T_thigh2 = T_hp2_joint @ _tf_rz(pose["hip_pitch2"])

    T_k2_origin = _tf([-0.154185, 0.107117, 0.0602], [0, 0, 0.662781])
    T_k2_joint = T_thigh2 @ T_k2_origin
    T_shin2 = T_k2_joint @ _tf_rz(pose["knee2"])

    # Link COMs
    torso_com_world = _rot(T_torso) @ np.array([-0.0110578, 2.08706e-05, -0.0316562]) + _pos(T_torso)
    c1_com_world = _rot(T_chassis1) @ np.array([0.0, 0.0, -0.0623004]) + _pos(T_chassis1)
    c2_com_world = _rot(T_chassis2) @ np.array([0.0, 0.0, -0.0623004]) + _pos(T_chassis2)
    th1_com_world = _rot(T_thigh1) @ np.array([-0.0223297, -0.118239, 0.0709555]) + _pos(T_thigh1)
    th2_com_world = _rot(T_thigh2) @ np.array([-0.0903635, 0.0795028, 0.0709556]) + _pos(T_thigh2)
    sh1_com_world = _rot(T_shin1) @ np.array([-0.0782005, 0.0113971, -0.0185949]) + _pos(T_shin1)
    sh2_com_world = _rot(T_shin2) @ np.array([0.000800703, 0.0790291, 0.0181051]) + _pos(T_shin2)

    foot1_world = _rot(T_shin1) @ np.array([-0.175, 0.038, -0.038]) + _pos(T_shin1)
    foot2_world = _rot(T_shin2) @ np.array([-0.010, 0.179, 0.038]) + _pos(T_shin2)

    m_torso, m_c1, m_c2 = 1.698, 1.1141, 1.1141
    m_upper = m_torso + m_c1 + m_c2
    upper_com = (m_torso * torso_com_world + m_c1 * c1_com_world + m_c2 * c2_com_world) / m_upper

    m_th, m_sh = 0.434, 0.1905
    total_mass = m_torso + 2*m_c1 + 2*m_th + 2*m_sh
    whole_com = (m_torso * torso_com_world
                 + m_c1 * (c1_com_world + c2_com_world)
                 + m_th * (th1_com_world + th2_com_world)
                 + m_sh * (sh1_com_world + sh2_com_world)) / total_mass

    hip_pos    = 0.5 * (_pos(T_hp1_joint) + _pos(T_hp2_joint))
    knee_pos   = 0.5 * (_pos(T_k1_joint) + _pos(T_k2_joint))
    foot_pos   = 0.5 * (foot1_world + foot2_world)
    shin_com   = 0.5 * (sh1_com_world + sh2_com_world)
    thigh_com  = 0.5 * (th1_com_world + th2_com_world)

    hp1_axis_world = _rot(T_hp1_joint) @ np.array([0, 0, 1])
    k1_axis_world  = _rot(T_k1_joint) @ np.array([0, 0, 1])

    return {
        "hip_pos": hip_pos, "knee_pos": knee_pos, "foot_pos": foot_pos,
        "shin_com": shin_com, "thigh_com": thigh_com, "upper_com": upper_com,
        "torso_com": torso_com_world, "whole_com": whole_com,
        "hp_axis": hp1_axis_world, "k_axis": k1_axis_world,
        "torso_z": _pos(T_torso)[2],
    }


def find_balanced_pose():
    """
    Numerically find the joint angles where COM_x = foot_contact_x.
    """
    from scipy.optimize import minimize

    def cost(x):
        hp, kn = x[0], x[1]
        pose = {
            "hip_roll1": 0.0, "hip_roll2": 0.0,
            "hip_pitch1": hp, "hip_pitch2": -hp,
            "knee1": kn, "knee2": -kn,
        }
        fk = compute_fk(pose)
        com_x  = fk["whole_com"][0]
        foot_x = fk["foot_pos"][0]
        foot_z = fk["foot_pos"][2]

        balance_err = (com_x - foot_x) ** 2
        ground_err = (foot_z - 0.0) ** 2
        reg = 0.001 * (hp**2 + kn**2)
        return 1e6 * balance_err + 1e4 * ground_err + reg

    best_result = None
    best_cost = float('inf')
    for hp0 in np.linspace(-0.5, 0.5, 9):
        for kn0 in np.linspace(-0.5, 0.5, 9):
            try:
                result = minimize(cost, [hp0, kn0], method='Nelder-Mead',
                                  options={'xatol': 1e-6, 'fatol': 1e-10, 'maxiter': 5000})
                if result.fun < best_cost:
                    best_cost = result.fun
                    best_result = result
            except Exception:
                continue

    hp_opt, kn_opt = best_result.x
    balanced_pose = {
        "hip_roll1": 0.0, "hip_roll2": 0.0,
        "hip_pitch1": hp_opt, "hip_pitch2": -hp_opt,
        "knee1": kn_opt, "knee2": -kn_opt,
    }

    fk = compute_fk(balanced_pose)
    com_x  = fk["whole_com"][0]
    foot_x = fk["foot_pos"][0]
    com_z  = fk["whole_com"][2]
    foot_z = fk["foot_pos"][2]

    print("\n" + "=" * 70)
    print("  BALANCED POSE SEARCH")
    print("=" * 70)
    print(f"  Optimized pose:")
    print(f"    hip_pitch = {hp_opt:+.4f} rad ({math.degrees(hp_opt):+.2f}°)")
    print(f"    knee      = {kn_opt:+.4f} rad ({math.degrees(kn_opt):+.2f}°)")
    print(f"  Balance check:")
    print(f"    COM_x  = {com_x*1000:+.1f} mm")
    print(f"    foot_x = {foot_x*1000:+.1f} mm")
    print(f"    offset = {(com_x - foot_x)*1000:+.1f} mm  {'✓ BALANCED' if abs(com_x - foot_x) < 0.002 else '✗ NOT BALANCED'}")
    print(f"  Geometry:")
    print(f"    COM_z  = {com_z*1000:.1f} mm")
    print(f"    foot_z = {foot_z*1000:.1f} mm  {'(on ground)' if foot_z < 0.01 else '(ABOVE GROUND — check foot model)'}")

    print(f"\n  FK at balanced pose:")
    for name in ["hip_pos", "knee_pos", "foot_pos", "shin_com", "thigh_com", "upper_com", "whole_com"]:
        p = fk[name]
        print(f"    {name:12s}: X={p[0]*1000:+7.1f}mm  Y={p[1]*1000:+7.1f}mm  Z={p[2]*1000:+7.1f}mm")
    print("=" * 70)

    return balanced_pose, fk


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 3: PLANAR MODEL EXTRACTION                            ║
# ╚══════════════════════════════════════════════════════════════════╝

def extract_planar_params(fk):
    """
    From the FK positions, extract the 3-link planar pendulum parameters.
    """
    foot = fk["foot_pos"]
    knee = fk["knee_pos"]
    hip  = fk["hip_pos"]
    scom = fk["shin_com"]
    tcom = fk["thigh_com"]
    ucom = fk["upper_com"]

    dx1 = knee[0] - foot[0]
    dz1 = knee[2] - foot[2]
    l1  = math.sqrt(dx1**2 + dz1**2)

    dx1c = scom[0] - foot[0]
    dz1c = scom[2] - foot[2]
    lc1  = math.sqrt(dx1c**2 + dz1c**2)

    dx2 = hip[0] - knee[0]
    dz2 = hip[2] - knee[2]
    l2  = math.sqrt(dx2**2 + dz2**2)

    dx2c = tcom[0] - knee[0]
    dz2c = tcom[2] - knee[2]
    lc2  = math.sqrt(dx2c**2 + dz2c**2)

    dx3 = ucom[0] - hip[0]
    dz3 = ucom[2] - hip[2]
    lc3  = math.sqrt(dx3**2 + dz3**2)

    theta1_eq = math.atan2(dx1, dz1)
    theta2_eq = math.atan2(dx2, dz2)
    theta3_eq = math.atan2(dx3, dz3)

    m1 = 2 * 0.1905
    m2 = 2 * 0.434
    m3 = 1.698 + 2 * 1.1141

    I1 = 2 * 0.000888
    I2 = 2 * 0.002334

    m_torso = 1.698
    m_chassis = 1.1141
    I3_torso = 0.001971
    I3_chassis = 0.001281
    I3 = I3_torso + 2 * I3_chassis
    I3 += 2 * m_chassis * 0.003
    I3 += m_torso * 0.003

    params = {
        "l1": l1, "l2": l2,
        "lc1": lc1, "lc2": lc2, "lc3": lc3,
        "m1": m1, "m2": m2, "m3": m3,
        "I1": I1, "I2": I2, "I3": I3,
        "theta_eq": np.array([theta1_eq, theta2_eq, theta3_eq]),
    }

    print("\n" + "=" * 70)
    print("  PLANAR MODEL PARAMETERS")
    print("=" * 70)
    print(f"  Shin:  l1={l1:.4f}m  lc1={lc1:.4f}m  m1={m1:.3f}kg  I1={I1:.6f}")
    print(f"  Thigh: l2={l2:.4f}m  lc2={lc2:.4f}m  m2={m2:.3f}kg  I2={I2:.6f}")
    print(f"  Upper: lc3={lc3:.4f}m  m3={m3:.3f}kg  I3={I3:.6f}")
    print(f"  θ_eq = [{math.degrees(theta1_eq):+.1f}°, "
          f"{math.degrees(theta2_eq):+.1f}°, "
          f"{math.degrees(theta3_eq):+.1f}°]")

    return params


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 4: 3-LINK PLANAR PENDULUM DYNAMICS                    ║
# ╚══════════════════════════════════════════════════════════════════╝

class PlanarDynamics:
    """3-link planar pendulum dynamics."""

    def __init__(self, params):
        self.l  = np.array([params["l1"], params["l2"], 0.0])
        self.lc = np.array([params["lc1"], params["lc2"], params["lc3"]])
        self.m  = np.array([params["m1"], params["m2"], params["m3"]])
        self.I  = np.array([params["I1"], params["I2"], params["I3"]])
        self.g  = 9.81
        self.n  = 3

        self.B_tau = np.array([
            [-1.0,  0.0],
            [ 1.0, -1.0],
            [ 0.0,  1.0],
        ])

    def _alpha(self, j, k):
        if j == k:
            return self.lc[j]
        elif j < k:
            return self.l[j]
        else:
            return 0.0

    def mass_matrix(self, q):
        n = self.n
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                val = 0.0
                for k in range(max(i, j), n):
                    val += self.m[k] * self._alpha(i, k) * self._alpha(j, k)
                M[i, j] = val * math.cos(q[i] - q[j])
                if i == j:
                    M[i, j] += self.I[i]
        return M

    def gravity_vector(self, q):
        n = self.n
        G = np.zeros(n)
        for i in range(n):
            s = 0.0
            for k in range(i, n):
                s += self.m[k] * self._alpha(i, k)
            G[i] = -self.g * math.sin(q[i]) * s
        return G


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 5: COM JACOBIAN, LINEARIZATION, AND LQR               ║
# ╚══════════════════════════════════════════════════════════════════╝

def compute_com_jacobian(params):
    """
    Compute the 3-vector h such that:

        COM_x - foot_x  ≈  h · δθ    (linearized around θ_eq)

    For a 3-link planar pendulum with absolute angles from vertical,
    the X position of mass k's COM is:

        x_k = Σ_{j=0}^{k} α(j,k) · sin(θ_j)

    where α(j,k) = lc_k if j==k, l_j if j<k, 0 if j>k.

    Total COM_x = (1/M) Σ_k  m_k · x_k

    The Jacobian of COM_x w.r.t. θ_i is:

        h_i = ∂(COM_x)/∂θ_i = (1/M) Σ_{k≥i} m_k · α(i,k) · cos(θ_i)

    At the equilibrium angles θ_eq.
    """
    m  = np.array([params["m1"], params["m2"], params["m3"]])
    l  = np.array([params["l1"], params["l2"], 0.0])
    lc = np.array([params["lc1"], params["lc2"], params["lc3"]])
    theta_eq = params["theta_eq"]
    M_total = m.sum()

    h = np.zeros(3)
    for i in range(3):
        weighted_arm = 0.0
        for k in range(i, 3):
            alpha_ik = lc[k] if i == k else l[i]
            weighted_arm += m[k] * alpha_ik
        h[i] = weighted_arm * math.cos(theta_eq[i]) / M_total

    return h


def compute_com_height_jacobian(params):
    """
    Compute the 3-vector h_z such that:

        COM_z - COM_z_eq  ≈  h_z · δθ    (linearized)

    h_z_i = ∂(COM_z)/∂θ_i = -(1/M) Σ_{k≥i} m_k · α(i,k) · sin(θ_i)

    This is used for a mild height-maintenance penalty (optional).
    At small equilibrium angles, h_z is small → height is insensitive.
    """
    m  = np.array([params["m1"], params["m2"], params["m3"]])
    l  = np.array([params["l1"], params["l2"], 0.0])
    lc = np.array([params["lc1"], params["lc2"], params["lc3"]])
    theta_eq = params["theta_eq"]
    M_total = m.sum()

    h_z = np.zeros(3)
    for i in range(3):
        weighted_arm = 0.0
        for k in range(i, 3):
            alpha_ik = lc[k] if i == k else l[i]
            weighted_arm += m[k] * alpha_ik
        h_z[i] = -weighted_arm * math.sin(theta_eq[i]) / M_total

    return h_z


def solve_care_numpy(A, B, Q, R):
    """
    Solve CARE via Hamiltonian eigendecomposition (numpy only).
    """
    n = A.shape[0]
    R_inv = np.linalg.inv(R)
    BR_inv_BT = B @ R_inv @ B.T

    H = np.zeros((2*n, 2*n))
    H[:n, :n] = A
    H[:n, n:] = -BR_inv_BT
    H[n:, :n] = -Q
    H[n:, n:] = -A.T

    eigvals, eigvecs = np.linalg.eig(H)
    stable_idx = np.where(np.real(eigvals) < 0)[0]

    if len(stable_idx) != n:
        print(f"  [LQR] WARNING: Expected {n} stable eigenvalues, "
              f"found {len(stable_idx)}")
        sorted_idx = np.argsort(np.real(eigvals))
        stable_idx = sorted_idx[:n]

    V = eigvecs[:, stable_idx]
    U1 = V[:n, :]
    U2 = V[n:, :]
    P = np.real(U2 @ np.linalg.inv(U1))
    P = 0.5 * (P + P.T)
    return P


def compute_lqr(dynamics, params):
    """
    Linearize the 3-link pendulum and solve LQR with COM-over-foot cost.

    KEY DIFFERENCE from the fixed-pose version:
    Q is built from the COM Jacobian, not per-angle weights.

    Returns:
        K       : 2×6 gain matrix (u = -K @ δx)
        A       : 6×6 state matrix
        B_lin   : 6×2 input matrix
        tau_eq  : 2-vector of equilibrium torques
        h_com   : 3-vector COM_x Jacobian (for debug output)
    """
    n = 3
    theta_eq = params["theta_eq"]
    q_eq = theta_eq.copy()

    # ── Equilibrium torque ──
    G_eq = dynamics.gravity_vector(q_eq)
    B_tau = dynamics.B_tau
    tau_eq, res, _, _ = np.linalg.lstsq(B_tau, G_eq, rcond=None)
    G_residual = G_eq - B_tau @ tau_eq

    print(f"\n  [LQR] Equilibrium torques: τ_knee={tau_eq[0]:+.3f} N·m, "
          f"τ_hip={tau_eq[1]:+.3f} N·m")
    print(f"  [LQR] Gravity residual: {np.linalg.norm(G_residual):.4f}")

    # ── Linearize dynamics ──
    M_eq = dynamics.mass_matrix(q_eq)
    M_inv = np.linalg.inv(M_eq)

    eps = 1e-6
    dGdq = np.zeros((n, n))
    for j in range(n):
        qp = q_eq.copy(); qp[j] += eps
        qm = q_eq.copy(); qm[j] -= eps
        dGdq[:, j] = (dynamics.gravity_vector(qp) - dynamics.gravity_vector(qm)) / (2*eps)

    A21 = -M_inv @ dGdq

    A = np.zeros((2*n, 2*n))
    A[:n, n:] = np.eye(n)
    A[n:, :n] = A21

    B_lin = np.zeros((2*n, 2))
    B_lin[n:, :] = M_inv @ B_tau

    # ── Controllability ──
    C_mat = B_lin.copy()
    AB = B_lin.copy()
    for i in range(2*n - 1):
        AB = A @ AB
        C_mat = np.hstack([C_mat, AB])
    rank = np.linalg.matrix_rank(C_mat)
    print(f"  [LQR] Controllability rank: {rank} / {2*n} "
          f"({'CONTROLLABLE' if rank == 2*n else 'NOT CONTROLLABLE'})")

    # ── Open-loop eigenvalues ──
    ol_eigs = np.linalg.eigvals(A)
    print(f"  [LQR] Open-loop eigenvalues:")
    for e in sorted(ol_eigs, key=lambda x: -np.real(x)):
        print(f"         {np.real(e):+.4f} {'+' if np.imag(e) >= 0 else ''}"
              f"{np.imag(e):.4f}j")

    # ══════════════════════════════════════════════════════════════
    #  BUILD Q FROM COM JACOBIAN  (this is the key change)
    # ══════════════════════════════════════════════════════════════
    #
    # h_com is a 3-vector: COM_x − foot_x ≈ h_com · δθ
    #
    # We want to penalize:
    #   (h_com · δθ)²          — balance error
    #   (h_com · δθ̇)²         — balance error rate
    #   ||δθ||²                — posture (light)
    #   ||δθ̇||²               — velocity (light)
    #
    # In the 6D state space [δθ; δθ̇]:
    #   h_pos = [h_com; 0,0,0]     → h_pos · x = COM offset
    #   h_vel = [0,0,0; h_com]     → h_vel · x = COM velocity
    #
    # Q = w_bal * h_pos @ h_pos^T
    #   + w_vel * h_vel @ h_vel^T
    #   + diag(w_reg_pos, w_reg_vel)

    h_com = compute_com_jacobian(params)

    # Extend to 6D state
    h_pos = np.zeros(2*n)
    h_pos[:n] = h_com

    h_vel = np.zeros(2*n)
    h_vel[n:] = h_com

    # Build Q
    Q = (W_BALANCE * np.outer(h_pos, h_pos)
       + W_COM_VEL * np.outer(h_vel, h_vel)
       + np.diag([W_POSTURE]*n + [W_VEL_REG]*n))

    # Optional: add mild COM height penalty to discourage squatting
    h_z = compute_com_height_jacobian(params)
    h_z_pos = np.zeros(2*n)
    h_z_pos[:n] = h_z
    W_HEIGHT = 500.0  # much lighter than balance
    Q += W_HEIGHT * np.outer(h_z_pos, h_z_pos)

    R = np.diag([R_KNEE, R_HIP])

    # ── Print Q structure for debugging ──
    print(f"\n  [LQR] COM-over-foot Q construction:")
    print(f"    h_com (COM_x Jacobian) = [{h_com[0]:.4f}, {h_com[1]:.4f}, {h_com[2]:.4f}]")
    print(f"    h_z   (COM_z Jacobian) = [{h_z[0]:.4f}, {h_z[1]:.4f}, {h_z[2]:.4f}]")
    print(f"    w_balance = {W_BALANCE:.0f}")
    print(f"    w_com_vel = {W_COM_VEL:.0f}")
    print(f"    w_height  = {W_HEIGHT:.0f}")
    print(f"    w_posture = {W_POSTURE:.1f}")
    print(f"    w_vel_reg = {W_VEL_REG:.1f}")
    print(f"    R = diag({R_KNEE:.0f}, {R_HIP:.0f})")
    print(f"\n    Effective Q diagonal: [{', '.join(f'{Q[i,i]:.1f}' for i in range(2*n))}]")
    print(f"    (note: off-diagonal terms carry the COM coupling!)")

    # Compare to what the old per-angle Q would have been:
    print(f"\n    For comparison, OLD per-angle Q would be:")
    print(f"      diag(200, 50, 50, 20, 5, 5)")
    print(f"    NEW Q captures COM-foot coupling in the off-diagonals,")
    print(f"    so angles that don't affect balance are lightly penalized.")

    # ── Solve CARE ──
    try:
        from scipy.linalg import solve_continuous_are
        P = solve_continuous_are(A, B_lin, Q, R)
        print("\n  [LQR] Solved CARE using scipy")
    except ImportError:
        P = solve_care_numpy(A, B_lin, Q, R)
        print("\n  [LQR] Solved CARE using numpy Hamiltonian method")

    K = np.linalg.inv(R) @ B_lin.T @ P

    # ── Closed-loop eigenvalues ──
    cl_eigs = np.linalg.eigvals(A - B_lin @ K)
    print(f"\n  [LQR] Closed-loop eigenvalues:")
    all_stable = True
    for e in sorted(cl_eigs, key=lambda x: -np.real(x)):
        stable = np.real(e) < 0
        all_stable = all_stable and stable
        marker = "✓" if stable else "✗ UNSTABLE"
        print(f"         {np.real(e):+.4f} {'+' if np.imag(e) >= 0 else ''}"
              f"{np.imag(e):.4f}j  {marker}")

    if not all_stable:
        print("\n  *** WARNING: Closed loop is NOT stable! ***")

    print(f"\n  [LQR] Gain matrix K (2×6):")
    labels = ["θ1", "θ2", "θ3", "θ̇1", "θ̇2", "θ̇3"]
    for i, name in enumerate(["τ_knee", "τ_hip "]):
        row = "  " + name + " = "
        terms = [f"{K[i,j]:+7.3f}·{labels[j]}" for j in range(6)]
        row += " ".join(terms)
        print(row)

    # ── Show what K "cares about" ──
    # K @ h_pos^T tells us the torque response to a unit COM offset
    tau_per_com = K @ h_pos
    print(f"\n  [LQR] Torque response to 1mm COM-foot offset:")
    print(f"    τ_knee = {tau_per_com[0]*0.001:+.4f} N·m/mm")
    print(f"    τ_hip  = {tau_per_com[1]*0.001:+.4f} N·m/mm")

    return K, A, B_lin, tau_eq, h_com


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 6: ISAAC SIM STATE MEASUREMENT                        ║
# ╚══════════════════════════════════════════════════════════════════╝

class SagittalStateMeasurer:
    """
    Measures the 6D state [θ1, θ2, θ3, θ̇1, θ̇2, θ̇3] from USD transforms.
    Also computes COM_x - foot_x for monitoring balance quality.
    """

    RATE_WINDOW = 8

    def __init__(self, stage, fk_result, planar_params, h_com):
        self.stage = stage
        self.theta_eq = planar_params["theta_eq"]
        self.h_com = h_com  # COM Jacobian for computing balance error

        self.torso_prim   = stage.GetPrimAtPath(LINK_PRIM_PATHS["torso2hip"])
        self.chassis1_prim = stage.GetPrimAtPath(LINK_PRIM_PATHS["chasis_motor2motor"])
        self.chassis2_prim = stage.GetPrimAtPath(LINK_PRIM_PATHS["lid_bearing_outer___boss_extrude3"])
        self.thigh1_prim  = stage.GetPrimAtPath(LINK_PRIM_PATHS["thigh_outer"])
        self.thigh2_prim  = stage.GetPrimAtPath(LINK_PRIM_PATHS["stand_steel_00___fillet1"])
        self.shin1_prim   = stage.GetPrimAtPath(LINK_PRIM_PATHS["shin_inner"])
        self.shin2_prim   = stage.GetPrimAtPath(LINK_PRIM_PATHS["nail___fillet1"])

        self.m_torso = 1.698
        self.m_chassis = 1.1141
        self.m_thigh = 0.434
        self.m_shin = 0.1905

        self.torso_com_local = np.array([-0.0110578, 2.08706e-05, -0.0316562])
        self.c1_com_local = np.array([0, 0, -0.0623004])
        self.c2_com_local = np.array([0, 0, -0.0623004])
        self.th1_com_local = np.array([-0.0223297, -0.118239, 0.0709555])
        self.th2_com_local = np.array([-0.0903635, 0.0795028, 0.0709556])
        self.sh1_com_local = np.array([-0.0782005, 0.0113971, -0.0185949])
        self.sh2_com_local = np.array([0.000800703, 0.0790291, 0.0181051])

        self.foot_contact_xz = None
        self.contact_captured = False
        self.capture_countdown = 40
        self.theta_history = []

    def reset(self):
        self.foot_contact_xz = None
        self.contact_captured = False
        self.capture_countdown = 40
        self.theta_history.clear()

    def _get_world_pos_and_rot(self, prim):
        xform = UsdGeom.Xformable(prim)
        try:
            M = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        except RuntimeError:
            return None, None
        pos = np.array([M[3][0], M[3][1], M[3][2]])
        rot = np.array([
            [M[0][0], M[0][1], M[0][2]],
            [M[1][0], M[1][1], M[1][2]],
            [M[2][0], M[2][1], M[2][2]],
        ])
        return pos, rot

    def _link_com_world(self, prim, local_com):
        pos, rot = self._get_world_pos_and_rot(prim)
        if pos is None:
            return None
        return rot @ local_com + pos

    def measure(self, dt):
        """
        Measure the 6D state.
        Returns (state_6d, debug_dict) or (None, None) if measurement fails.
        """
        torso_pos, torso_rot = self._get_world_pos_and_rot(self.torso_prim)
        if torso_pos is None:
            return None, None

        shin1_pos, _ = self._get_world_pos_and_rot(self.shin1_prim)
        shin2_pos, _ = self._get_world_pos_and_rot(self.shin2_prim)
        thigh1_pos, _ = self._get_world_pos_and_rot(self.thigh1_prim)
        thigh2_pos, _ = self._get_world_pos_and_rot(self.thigh2_prim)

        if any(x is None for x in [shin1_pos, shin2_pos, thigh1_pos, thigh2_pos]):
            return None, None

        knee_xz = np.array([
            0.5 * (shin1_pos[0] + shin2_pos[0]),
            0.5 * (shin1_pos[2] + shin2_pos[2]),
        ])
        hip_xz = np.array([
            0.5 * (thigh1_pos[0] + thigh2_pos[0]),
            0.5 * (thigh1_pos[2] + thigh2_pos[2]),
        ])

        tc = self._link_com_world(self.torso_prim, self.torso_com_local)
        c1c = self._link_com_world(self.chassis1_prim, self.c1_com_local)
        c2c = self._link_com_world(self.chassis2_prim, self.c2_com_local)
        if any(x is None for x in [tc, c1c, c2c]):
            return None, None

        m_upper = self.m_torso + 2 * self.m_chassis
        upper_com = (self.m_torso * tc + self.m_chassis * c1c
                     + self.m_chassis * c2c) / m_upper
        upper_xz = np.array([upper_com[0], upper_com[2]])

        # Foot contact
        if not self.contact_captured:
            sh1c = self._link_com_world(self.shin1_prim, self.sh1_com_local)
            sh2c = self._link_com_world(self.shin2_prim, self.sh2_com_local)
            if sh1c is None or sh2c is None:
                return None, None
            avg_shin_x = 0.5 * (sh1c[0] + sh2c[0])
            self.foot_contact_xz = np.array([avg_shin_x, 0.0])
            self.capture_countdown -= 1
            if self.capture_countdown <= 0:
                self.contact_captured = True
                print(f"  [State] Foot contact frozen at X={self.foot_contact_xz[0]*1000:.1f}mm")

        foot_xz = self.foot_contact_xz

        # Absolute angles
        dx1 = knee_xz[0] - foot_xz[0]
        dz1 = knee_xz[1] - foot_xz[1]
        theta1 = math.atan2(dx1, dz1)

        dx2 = hip_xz[0] - knee_xz[0]
        dz2 = hip_xz[1] - knee_xz[1]
        theta2 = math.atan2(dx2, dz2)

        dx3 = upper_xz[0] - hip_xz[0]
        dz3 = upper_xz[1] - hip_xz[1]
        theta3 = math.atan2(dx3, dz3)

        theta = np.array([theta1, theta2, theta3])

        # Angular velocities
        self.theta_history.append(theta.copy())
        if len(self.theta_history) > self.RATE_WINDOW:
            self.theta_history.pop(0)

        if len(self.theta_history) >= 2:
            oldest = self.theta_history[0]
            newest = self.theta_history[-1]
            window_dt = (len(self.theta_history) - 1) * dt
            if window_dt > 1e-9:
                theta_dot = (newest - oldest) / window_dt
            else:
                theta_dot = np.zeros(3)
        else:
            theta_dot = np.zeros(3)

        delta_theta = theta - self.theta_eq
        state = np.concatenate([delta_theta, theta_dot])

        # ── COM-foot balance error (linearized) ──
        com_foot_offset = self.h_com @ delta_theta   # meters
        com_foot_vel    = self.h_com @ theta_dot      # m/s

        # Full-body COM for debug
        th1c = self._link_com_world(self.thigh1_prim, self.th1_com_local)
        th2c = self._link_com_world(self.thigh2_prim, self.th2_com_local)
        sh1c = self._link_com_world(self.shin1_prim, self.sh1_com_local)
        sh2c = self._link_com_world(self.shin2_prim, self.sh2_com_local)

        total_mass = self.m_torso + 2*(self.m_chassis + self.m_thigh + self.m_shin)
        if all(x is not None for x in [th1c, th2c, sh1c, sh2c]):
            com_full = (self.m_torso * tc
                       + self.m_chassis * (c1c + c2c)
                       + self.m_thigh * (th1c + th2c)
                       + self.m_shin * (sh1c + sh2c)) / total_mass
        else:
            com_full = upper_com

        # Actual (nonlinear) COM_x - foot_x for comparison
        actual_com_foot = com_full[0] - foot_xz[0]

        debug = {
            "theta": theta,
            "theta_dot": theta_dot,
            "delta_theta": delta_theta,
            "com_x": com_full[0],
            "com_z": com_full[2],
            "foot_x": foot_xz[0],
            "torso_z": torso_pos[2],
            "com_foot_lin": com_foot_offset,   # linearized COM-foot offset
            "com_foot_vel": com_foot_vel,       # linearized COM velocity
            "com_foot_actual": actual_com_foot, # actual (nonlinear) offset
        }

        return state, debug


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 7: LQR CONTROLLER FOR ISAAC SIM                       ║
# ╚══════════════════════════════════════════════════════════════════╝

class LQRController:
    """
    Applies LQR full-state feedback with COM-over-foot objective.
    """

    def __init__(self, K, tau_eq, planar_params, h_com):
        self.K = K
        self.tau_eq = tau_eq
        self.h_com = h_com
        self.step = 0

        self.kp_knee = JOINT_PARAMS["knee1"]["stiffness"]
        self.kp_hip  = JOINT_PARAMS["hip_pitch1"]["stiffness"]

        self.max_offset = 0.25

        self.knee_signs = {"knee1": -1.0, "knee2": 1.0}
        self.hip_signs  = {"hip_pitch1": -1.0, "hip_pitch2": +1.0}

        # Sign diagnosis — now based on COM-foot offset instead of δθ1
        self.com_offset_history = []
        self.sign_diagnosed = False

    def compute_targets(self, state_6d, debug=None):
        """
        Compute joint targets from LQR state feedback.
        """
        delta_tau = -self.K @ state_6d
        tau_knee_raw = delta_tau[0]
        tau_hip_raw  = delta_tau[1]

        dq_knee_raw = tau_knee_raw / self.kp_knee
        dq_hip_raw  = tau_hip_raw  / self.kp_hip

        dq_knee = np.clip(dq_knee_raw, -self.max_offset, self.max_offset)
        dq_hip  = np.clip(dq_hip_raw,  -self.max_offset, self.max_offset)

        knee_saturated = abs(dq_knee_raw) > self.max_offset
        hip_saturated  = abs(dq_hip_raw)  > self.max_offset

        targets = {}
        for name, default in DEFAULT_POSE.items():
            target = default
            if name in self.knee_signs:
                target += self.knee_signs[name] * dq_knee
            if name in self.hip_signs:
                target += self.hip_signs[name] * dq_hip
            targets[name] = target

        self.step += 1

        ctrl_debug = {
            "tau_knee": tau_knee_raw,
            "tau_hip": tau_hip_raw,
            "dq_knee": dq_knee,
            "dq_hip": dq_hip,
            "knee_sat": knee_saturated,
            "hip_sat": hip_saturated,
        }

        # ── Sign diagnosis using COM-foot offset ──
        if debug is not None:
            com_offset = debug.get("com_foot_actual", 0.0)
            self.com_offset_history.append(com_offset)
            if len(self.com_offset_history) > 40:
                self.com_offset_history.pop(0)

        if not self.sign_diagnosed and self.step == SETTLE_STEPS + 35:
            self.sign_diagnosed = True
            hist = self.com_offset_history[-25:]
            if len(hist) >= 20:
                diffs = [hist[i+1] - hist[i] for i in range(len(hist)-1)]
                pos_diffs = sum(1 for d in diffs if d > 0)
                total = len(diffs)
                growing = abs(hist[-1]) > abs(hist[0]) * 1.5

                print("\n" + "=" * 70)
                print("  AUTOMATIC SIGN DIAGNOSIS (COM-foot based)")
                print("=" * 70)
                print(f"  COM-foot offset: start={hist[0]*1000:+.1f}mm  end={hist[-1]*1000:+.1f}mm  "
                      f"{'DIVERGING' if growing else 'stable'}")
                print(f"  Trend: {pos_diffs}/{total} steps growing")
                if growing:
                    print(f"\n  >>> COM-foot offset DIVERGING — try flipping sign dicts <<<")
                else:
                    print(f"\n  >>> COM-foot offset stable — signs appear correct <<<")
                print("=" * 70 + "\n")

        return targets, ctrl_debug


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 8: ISAAC SIM INTEGRATION                              ║
# ╚══════════════════════════════════════════════════════════════════╝

_joint_prims    = {}
_lqr_ctrl       = None
_state_measurer = None
_was_playing    = False
_physx_sub      = None
_timeline_sub   = None


def set_joint_target(name, target_rad):
    if name in _joint_prims:
        prim = _joint_prims[name]
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if drive:
            drive.GetTargetPositionAttr().Set(np.degrees(target_rad))


def get_torso_state(stage):
    prim = stage.GetPrimAtPath(TORSO_PRIM_PATH)
    if not prim or not prim.IsValid():
        return None, None
    xform = UsdGeom.Xformable(prim)
    try:
        M = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    except RuntimeError:
        return None, None
    bz_x, bz_y, bz_z = M[2][0], M[2][1], M[2][2]
    pitch = math.atan2(bz_x, bz_z)
    roll  = math.atan2(bz_y, bz_z)
    tilt  = math.degrees(math.sqrt(pitch**2 + roll**2))
    height = M[3][2]
    return tilt, height


def _on_timeline_event(event):
    global _was_playing
    if event.type == int(omni.timeline.TimelineEventType.PLAY):
        if _state_measurer is not None:
            _state_measurer.reset()
        if _lqr_ctrl is not None:
            _lqr_ctrl.step = 0
            _lqr_ctrl.com_offset_history.clear()
            _lqr_ctrl.sign_diagnosed = False
        _was_playing = True
        print(f"\n[LQR-COM] PLAY — COM-over-foot balance controller active (200 Hz)")
    elif event.type == int(omni.timeline.TimelineEventType.STOP):
        _was_playing = False
        print(f"[LQR-COM] STOPPED")


SETTLE_STEPS = 50

def physics_step_callback(dt):
    global _lqr_ctrl, _state_measurer

    if not _was_playing or _lqr_ctrl is None or _state_measurer is None:
        return

    stage = omni.usd.get_context().get_stage()

    tilt, height = get_torso_state(stage)
    if tilt is None:
        return

    if tilt > MAX_TILT_DEG or height < MIN_BASE_HEIGHT:
        for name, default in DEFAULT_POSE.items():
            set_joint_target(name, default)
        if _lqr_ctrl.step % PRINT_EVERY_N == 0:
            status = "TILT" if tilt > MAX_TILT_DEG else "FALLEN"
            print(f"  [{status}] tilt={tilt:.1f}° height={height:.3f}m — holding default")
        _lqr_ctrl.step += 1
        return

    # Settling phase
    if _lqr_ctrl.step < SETTLE_STEPS:
        for name, default in DEFAULT_POSE.items():
            set_joint_target(name, default)

        state, debug = _state_measurer.measure(dt)

        if _lqr_ctrl.step % PRINT_EVERY_N == 0 and debug is not None:
            dth = debug["delta_theta"]
            cf = debug["com_foot_actual"]
            print(f"  [SETTLE {_lqr_ctrl.step:3d}/{SETTLE_STEPS}]  "
                  f"δθ=[{math.degrees(dth[0]):+5.1f} {math.degrees(dth[1]):+5.1f} {math.degrees(dth[2]):+5.1f}]°  "
                  f"COM-foot={cf*1000:+6.1f}mm  h={debug['torso_z']:.3f}m")

        if _lqr_ctrl.step == SETTLE_STEPS - 1:
            print(f"\n  >>> SETTLING COMPLETE — COM-OVER-FOOT LQR ACTIVE <<<\n")

        _lqr_ctrl.step += 1
        return

    # ── Measure state ──
    state, debug = _state_measurer.measure(dt)
    if state is None:
        return

    # ── Compute LQR targets ──
    targets, ctrl = _lqr_ctrl.compute_targets(state, debug)

    for name, target in targets.items():
        set_joint_target(name, target)

    # ── Debug print ──
    if _lqr_ctrl.step % PRINT_EVERY_N == 0:
        dth = debug["delta_theta"]
        thd = debug["theta_dot"]
        th  = debug["theta"]

        ksat = "SAT!" if ctrl["knee_sat"] else "    "
        hsat = "SAT!" if ctrl["hip_sat"] else "    "

        # The KEY metric: COM-foot offset (both linearized and actual)
        cf_lin = debug["com_foot_lin"]
        cf_act = debug["com_foot_actual"]
        cf_vel = debug["com_foot_vel"]

        print(f"\n  ── step {_lqr_ctrl.step:5d} ──────────────────────────────────────")
        print(f"  ★ BALANCE:  COM-foot = {cf_act*1000:+6.1f}mm (actual)  "
              f"{cf_lin*1000:+6.1f}mm (linearized)  "
              f"vel={cf_vel*1000:+5.1f}mm/s")
        print(f"  STATE (abs):  θ=[{math.degrees(th[0]):+6.1f} {math.degrees(th[1]):+6.1f} {math.degrees(th[2]):+6.1f}]°")
        print(f"  STATE (δ):    δθ=[{math.degrees(dth[0]):+6.1f} {math.degrees(dth[1]):+6.1f} {math.degrees(dth[2]):+6.1f}]°")
        print(f"  VELOCITY:     θ̇=[{thd[0]:+6.2f} {thd[1]:+6.2f} {thd[2]:+6.2f}] rad/s")
        print(f"  LQR TORQUES:  τ_kn={ctrl['tau_knee']:+7.2f}  τ_hp={ctrl['tau_hip']:+7.2f} N·m")
        print(f"  POS OFFSETS:  Δq_kn={ctrl['dq_knee']:+.4f} {ksat}   Δq_hp={ctrl['dq_hip']:+.4f} {hsat}")
        print(f"  TARGETS:      hp1={targets['hip_pitch1']:+.4f}  hp2={targets['hip_pitch2']:+.4f}  "
              f"kn1={targets['knee1']:+.4f}  kn2={targets['knee2']:+.4f}")
        print(f"  GEOMETRY:     COM_x={debug['com_x']*1000:+6.1f}mm  foot_x={debug['foot_x']*1000:+6.1f}mm  "
              f"h={debug['torso_z']:.3f}m")

        # Divergence warning
        if abs(cf_act) > 0.020:  # >20mm offset
            print(f"  ⚠ COM-foot offset > 20mm — balance degrading!")
        if abs(cf_act) > 0.050:  # >50mm offset
            print(f"  ⚠⚠ COM-foot offset > 50mm — likely falling! Check signs.")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 9: SCENE SETUP                                        ║
# ╚══════════════════════════════════════════════════════════════════╝

async def setup_scene():
    global _joint_prims, _physx_sub, _timeline_sub
    global _lqr_ctrl, _state_measurer
    global DEFAULT_POSE

    print("\n" + "=" * 70)
    print("  LQR-COM BALANCE CONTROLLER — COM-OVER-FOOT OBJECTIVE")
    print("=" * 70)

    # Step 1: Find balanced pose (still needed for linearization point)
    print("\n  Original pose:")
    orig_fk = compute_fk(DEFAULT_POSE)
    com_x_orig = orig_fk["whole_com"][0]
    foot_x_orig = orig_fk["foot_pos"][0]
    print(f"    COM_x={com_x_orig*1000:+.1f}mm  foot_x={foot_x_orig*1000:+.1f}mm  "
          f"offset={(com_x_orig-foot_x_orig)*1000:+.1f}mm")

    balanced_pose, balanced_fk = find_balanced_pose()
    DEFAULT_POSE = balanced_pose

    # Step 2: Compute LQR with COM-over-foot cost
    print("\n" + "=" * 70)
    print("  COMPUTING COM-OVER-FOOT LQR GAINS")
    print("=" * 70)

    fk = balanced_fk
    params = extract_planar_params(fk)
    dynamics = PlanarDynamics(params)

    K, A, B_lin, tau_eq, h_com = compute_lqr(dynamics, params)

    # Set up Isaac Sim scene
    stage = omni.usd.get_context().get_stage()

    if stage.GetPrimAtPath("/World"):
        omni.kit.commands.execute("DeletePrims", paths=["/World"])
        await omni.kit.app.get_app().next_update_async()

    UsdGeom.Xform.Define(stage, "/World")

    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    physx_api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx_api.CreateTimeStepsPerSecondAttr(200)

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

    robot_prim = stage.DefinePrim("/World/Robot")
    robot_prim.GetReferences().AddReference(USD_PATH)
    await omni.kit.app.get_app().next_update_async()
    await omni.kit.app.get_app().next_update_async()

    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if path.startswith("/World/Robot"):
            if prim.HasAPI(UsdPhysics.CollisionAPI) or "collision" in prim.GetName().lower():
                mat = UsdPhysics.MaterialAPI.Apply(prim)
                if mat:
                    mat.CreateStaticFrictionAttr(0.9)
                    mat.CreateDynamicFrictionAttr(0.8)

    xform = UsdGeom.Xformable(robot_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, SPAWN_HEIGHT))

    # Configure joints
    _joint_prims = {}
    print("\n  JOINT CONFIGURATION (balanced pose):")
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            name = prim.GetName()
            if name in JOINT_PARAMS:
                _joint_prims[name] = prim
                p = JOINT_PARAMS[name]
                target_rad = DEFAULT_POSE.get(name, 0.0)

                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.CreateTypeAttr("force")
                drive.CreateStiffnessAttr(p["stiffness"])
                drive.CreateDampingAttr(p["damping"])
                drive.CreateMaxForceAttr(p["effort"])
                drive.CreateTargetPositionAttr(np.degrees(target_rad))

                joint_state = PhysxSchema.JointStateAPI.Apply(prim, "angular")
                if joint_state:
                    joint_state.CreatePositionAttr().Set(np.degrees(target_rad))
                    joint_state.CreateVelocityAttr().Set(0.0)

                print(f"    {name:15s}  target={target_rad:+.3f} rad ({np.degrees(target_rad):+.1f}°)  "
                      f"Kp={p['stiffness']:<5}  Kd={p['damping']:<5}")

    # Initialize controller and measurer
    _state_measurer = SagittalStateMeasurer(stage, fk, params, h_com)
    _lqr_ctrl = LQRController(K, tau_eq, params, h_com)

    # Summary
    print(f"\n  Joints configured: {len(_joint_prims)}")
    print(f"  Physics timestep:  200 Hz")

    print(f"\n  COM JACOBIAN h = [{h_com[0]:.4f}, {h_com[1]:.4f}, {h_com[2]:.4f}]")
    print(f"  Interpretation: a 1° change in θ1 moves COM by {h_com[0]*math.radians(1)*1000:.2f}mm")
    print(f"                  a 1° change in θ2 moves COM by {h_com[1]*math.radians(1)*1000:.2f}mm")
    print(f"                  a 1° change in θ3 moves COM by {h_com[2]*math.radians(1)*1000:.2f}mm")

    print(f"\n  LQR GAIN MATRIX (2×6):")
    labels = ["θ1", "θ2", "θ3", "θ̇1", "θ̇2", "θ̇3"]
    for i, name in enumerate(["τ_knee", "τ_hip "]):
        terms = [f"{K[i,j]:+7.3f}·{labels[j]}" for j in range(6)]
        print(f"    {name}: {' '.join(terms)}")

    print(f"\n  ──────────────────────────────────────────")
    print(f"  TUNING GUIDE (COM-over-foot version):")
    print(f"    The controller now targets COM_x = foot_x,")
    print(f"    NOT a fixed joint configuration.")
    print(f"")
    print(f"    If falling:  → Check signs (same as before)")
    print(f"    If sluggish: → Increase W_BALANCE ({W_BALANCE:.0f})")
    print(f"                   or decrease R_KNEE/R_HIP")
    print(f"    If oscillating: → Increase W_COM_VEL ({W_COM_VEL:.0f})")
    print(f"                      or increase R")
    print(f"    If squatting:   → Increase W_POSTURE ({W_POSTURE:.1f})")
    print(f"                      or increase W_HEIGHT")
    print(f"    If stiff/unnatural: → Decrease W_POSTURE")
    print(f"  ──────────────────────────────────────────")
    print(f"\n  >>> Press PLAY to start COM-over-foot LQR <<<")
    print("=" * 70 + "\n")

    _physx_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(
        physics_step_callback
    )
    timeline = omni.timeline.get_timeline_interface()
    stream = timeline.get_timeline_event_stream()
    _timeline_sub = stream.create_subscription_to_pop(_on_timeline_event)

    print("[LQR-COM] Physics callback + timeline events registered")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  LAUNCH                                                        ║
# ╚══════════════════════════════════════════════════════════════════╝

asyncio.ensure_future(setup_scene())
