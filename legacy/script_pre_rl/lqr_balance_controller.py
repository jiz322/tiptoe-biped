"""
lqr_balance_controller.py — LQR Balance for Point-Foot Bipedal Robot
Run inside Isaac Sim's Script Editor (Window > Script Editor)

=== WHY LQR INSTEAD OF HAND-TUNED PD ===

This robot is an underactuated inverted pendulum: the foot-ground contact
is a passive (unactuated) pivot, while hip_pitch and knee are actuated.
Balance requires coordinated multi-joint control — the hip correction MUST
account for the knee state and vice versa, because they're coupled through
the equations of motion.

Hand-tuned PD with decoupled loops (balance loop, height loop, posture loop)
cannot capture this coupling. LQR solves the optimal full-state feedback
problem: given a cost on state deviation and control effort, it produces a
gain matrix K that makes every joint respond optimally to every state variable.

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

6. At each physics step:
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
4. Monitor console output — tune Q/R matrices as needed
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

# Default standing pose
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
# Higher stiffness = tighter tracking of LQR commands.
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
# ║  SECTION 2: FORWARD KINEMATICS FROM URDF                       ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# We compute the world-frame positions of every joint and link COM
# at the default pose (all joints ≈ 0). This gives us the sagittal-
# plane geometry needed for the planar model.
#
# All joint axes map to world-Y (verified from URDF transforms),
# confirming the sagittal plane is XZ.

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
    """4×4 rotation about Z by angle q."""
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
    pose: dict with keys like DEFAULT_POSE (hip_roll1, hip_pitch1, knee1, etc.)
    Returns a dict with positions for the sagittal plane model + total COM.
    """
    # ── World frame: torso at spawn height ──
    T_torso = np.eye(4, dtype=np.float64)
    T_torso[2, 3] = SPAWN_HEIGHT  # Z = 0.42

    # ── Leg 1 chain: torso → hip_roll1 → chassis → hip_pitch1 → thigh → knee1 → shin ──
    T_hr1_origin = _tf([-0.070392, 0.0800218, -0.0315], [1.5708, 0, 1.5708])
    T_chassis1 = T_torso @ T_hr1_origin @ _tf_rz(pose["hip_roll1"])

    T_hp1_origin = _tf([-0.0415, 0, -0.071], [0, 1.5708, 0])
    T_hp1_joint = T_chassis1 @ T_hp1_origin
    T_thigh1 = T_hp1_joint @ _tf_rz(pose["hip_pitch1"])

    T_k1_origin = _tf([-0.0592463, -0.172173, 0.0969], [0, 0, 1.72565])
    T_k1_joint = T_thigh1 @ T_k1_origin
    T_shin1 = T_k1_joint @ _tf_rz(pose["knee1"])

    # ── Leg 2 chain ──
    T_hr2_origin = _tf([-0.070392, -0.0799782, -0.0315], [-3.14159, -1.5708, 0])
    T_chassis2 = T_torso @ T_hr2_origin @ _tf_rz(pose["hip_roll2"])

    T_hp2_origin = _tf([0, -0.0415, -0.071], [-1.5708, 0.908015, 0])
    T_hp2_joint = T_chassis2 @ T_hp2_origin
    T_thigh2 = T_hp2_joint @ _tf_rz(pose["hip_pitch2"])

    T_k2_origin = _tf([-0.154185, 0.107117, 0.0602], [0, 0, 0.662781])
    T_k2_joint = T_thigh2 @ T_k2_origin
    T_shin2 = T_k2_joint @ _tf_rz(pose["knee2"])

    # ── Link COMs in world frame ──
    torso_com_world = _rot(T_torso) @ np.array([-0.0110578, 2.08706e-05, -0.0316562]) + _pos(T_torso)
    c1_com_world = _rot(T_chassis1) @ np.array([0.0, 0.0, -0.0623004]) + _pos(T_chassis1)
    c2_com_world = _rot(T_chassis2) @ np.array([0.0, 0.0, -0.0623004]) + _pos(T_chassis2)
    th1_com_world = _rot(T_thigh1) @ np.array([-0.0223297, -0.118239, 0.0709555]) + _pos(T_thigh1)
    th2_com_world = _rot(T_thigh2) @ np.array([-0.0903635, 0.0795028, 0.0709556]) + _pos(T_thigh2)
    sh1_com_world = _rot(T_shin1) @ np.array([-0.0782005, 0.0113971, -0.0185949]) + _pos(T_shin1)
    sh2_com_world = _rot(T_shin2) @ np.array([0.000800703, 0.0790291, 0.0181051]) + _pos(T_shin2)

    # Foot contact estimates
    foot1_world = _rot(T_shin1) @ np.array([-0.175, 0.038, -0.038]) + _pos(T_shin1)
    foot2_world = _rot(T_shin2) @ np.array([-0.010, 0.179, 0.038]) + _pos(T_shin2)

    # ── Combined upper body COM (torso + both chassis) ──
    m_torso, m_c1, m_c2 = 1.698, 1.1141, 1.1141
    m_upper = m_torso + m_c1 + m_c2
    upper_com = (m_torso * torso_com_world + m_c1 * c1_com_world + m_c2 * c2_com_world) / m_upper

    # ── Whole-body COM ──
    m_th, m_sh = 0.434, 0.1905
    total_mass = m_torso + 2*m_c1 + 2*m_th + 2*m_sh
    whole_com = (m_torso * torso_com_world
                 + m_c1 * (c1_com_world + c2_com_world)
                 + m_th * (th1_com_world + th2_com_world)
                 + m_sh * (sh1_com_world + sh2_com_world)) / total_mass

    # ── Average positions for symmetric model ──
    hip_pos    = 0.5 * (_pos(T_hp1_joint) + _pos(T_hp2_joint))
    knee_pos   = 0.5 * (_pos(T_k1_joint) + _pos(T_k2_joint))
    foot_pos   = 0.5 * (foot1_world + foot2_world)
    shin_com   = 0.5 * (sh1_com_world + sh2_com_world)
    thigh_com  = 0.5 * (th1_com_world + th2_com_world)

    # Joint axes in world frame
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

    Searches over hip_pitch and knee angles (symmetric: hp2=-hp1, kn2=-kn1)
    to minimize |COM_x - foot_x| while keeping the robot at a reasonable height.

    Returns the balanced DEFAULT_POSE dict.
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
        com_z  = fk["whole_com"][2]
        foot_z = fk["foot_pos"][2]

        # Primary: COM_x must equal foot_x (balance)
        balance_err = (com_x - foot_x) ** 2

        # Secondary: foot should be near the ground (foot_z ≈ 0)
        ground_err = (foot_z - 0.0) ** 2

        # Tertiary: prefer small joint angles (regularization)
        reg = 0.001 * (hp**2 + kn**2)

        return 1e6 * balance_err + 1e4 * ground_err + reg

    # Search from multiple starting points
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

    # Verify
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

    # Also print FK details for the balanced pose
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
    From the FK positions, extract the 3-link planar pendulum parameters
    in the sagittal (XZ) plane.

    Returns dict with:
        l1, l2      : link lengths (shin, thigh)
        lc1, lc2, lc3 : COM distances from proximal joint
        m1, m2, m3  : link masses (combined for both legs)
        I1, I2, I3  : sagittal-plane moments of inertia
        theta_eq    : equilibrium absolute angles [θ1, θ2, θ3]
    """
    foot = fk["foot_pos"]
    knee = fk["knee_pos"]
    hip  = fk["hip_pos"]
    scom = fk["shin_com"]
    tcom = fk["thigh_com"]
    ucom = fk["upper_com"]

    # ── Sagittal plane (XZ) projections ──
    # Link 1: foot → knee (shin)
    dx1 = knee[0] - foot[0]
    dz1 = knee[2] - foot[2]
    l1  = math.sqrt(dx1**2 + dz1**2)

    # COM distance from foot along shin
    dx1c = scom[0] - foot[0]
    dz1c = scom[2] - foot[2]
    lc1  = math.sqrt(dx1c**2 + dz1c**2)

    # Link 2: knee → hip (thigh)
    dx2 = hip[0] - knee[0]
    dz2 = hip[2] - knee[2]
    l2  = math.sqrt(dx2**2 + dz2**2)

    # COM distance from knee along thigh
    dx2c = tcom[0] - knee[0]
    dz2c = tcom[2] - knee[2]
    lc2  = math.sqrt(dx2c**2 + dz2c**2)

    # Link 3: hip → upper body COM
    dx3 = ucom[0] - hip[0]
    dz3 = ucom[2] - hip[2]
    lc3  = math.sqrt(dx3**2 + dz3**2)

    # ── Equilibrium absolute angles (from vertical) ──
    # Positive θ = lean in +X direction
    theta1_eq = math.atan2(dx1, dz1)
    theta2_eq = math.atan2(dx2, dz2)
    theta3_eq = math.atan2(dx3, dz3)

    # ── Masses (both legs combined) ──
    m1 = 2 * 0.1905     # both shins
    m2 = 2 * 0.434      # both thighs
    m3 = 1.698 + 2 * 1.1141  # torso + both chassis

    # ── Sagittal-plane inertias ──
    # Joint axes are world-Y. In each link's local frame, world-Y maps to
    # a specific local axis. The moment of inertia about that axis is what
    # we need.
    #
    # From URDF analysis:
    #   shin local Z ≈ world Y  → use Izz
    #   thigh local Z ≈ world Y → use Izz
    #   torso local Y ≈ world Y → use Iyy (torso frame ≈ world frame)
    #   chassis local Z ≈ world Y → use Izz
    #
    # Combined with parallel axis theorem for offset COMs.

    # Shin: Izz_shin_inner=0.000888, Izz_nail=0.000888
    I1 = 2 * 0.000888

    # Thigh: Izz_thigh=0.002333, Izz_stand_steel=0.002334
    I2 = 2 * 0.002334

    # Upper body: torso Iyy + 2 × chassis Izz + parallel axis terms
    # Torso Iyy = 0.001971
    # Chassis Izz = 0.001281
    # Parallel axis: m * d² where d is XZ-plane distance from sub-COM to combined COM
    m_torso = 1.698
    m_chassis = 1.1141

    # We approximate the parallel axis contribution
    # (it's small compared to the direct inertia for this robot)
    I3_torso = 0.001971
    I3_chassis = 0.001281
    I3 = I3_torso + 2 * I3_chassis

    # Add parallel axis terms for chassis offset from combined upper COM
    # d² ≈ (chassis_com_x - upper_com_x)² + (chassis_com_z - upper_com_z)²
    # These are typically a few cm → d² ≈ 0.002-0.005
    # m_chassis * d² ≈ 1.11 * 0.003 ≈ 0.003
    I3 += 2 * m_chassis * 0.003  # approximate parallel axis
    I3 += m_torso * 0.003        # torso parallel axis

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
#
# Standard Lagrangian mechanics for an n-link planar pendulum with
# absolute angles θ_i measured from vertical.
#
# State:   q = [θ1, θ2, θ3]  (absolute angles from vertical)
# Control: τ = [τ_knee, τ_hip]
#
# Equations of motion:  M(q) q̈ + C(q,q̇) q̇ + G(q) = B_tau @ τ
#
# where B_tau = [[-1,  0],    knee torque acts between link 1 and 2
#                [ 1, -1],    hip torque acts between link 2 and 3
#                [ 0,  1]]

class PlanarDynamics:
    """3-link planar pendulum dynamics."""

    def __init__(self, params):
        self.l  = np.array([params["l1"], params["l2"], 0.0])  # link lengths
        self.lc = np.array([params["lc1"], params["lc2"], params["lc3"]])
        self.m  = np.array([params["m1"], params["m2"], params["m3"]])
        self.I  = np.array([params["I1"], params["I2"], params["I3"]])
        self.g  = 9.81
        self.n  = 3  # number of links

        # Actuation matrix: τ_applied = B_tau @ u
        # u = [τ_knee, τ_hip]
        # τ_knee is between links 1-2, τ_hip between links 2-3
        self.B_tau = np.array([
            [-1.0,  0.0],
            [ 1.0, -1.0],
            [ 0.0,  1.0],
        ])

    def _alpha(self, j, k):
        """
        Coefficient α(j,k) for COM position computation.
        α(j,k) = lc_k if j == k, l_j if j < k, 0 if j > k
        """
        if j == k:
            return self.lc[j]
        elif j < k:
            return self.l[j]
        else:
            return 0.0

    def mass_matrix(self, q):
        """
        Compute 3×3 mass matrix M(q).

        M_ij = δ_ij I_i + Σ_{k=max(i,j)}^{n-1} m_k α(i,k) α(j,k) cos(q_i - q_j)
        """
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
        """
        Compute gravity vector G(q).

        G_i = -g sin(θ_i) Σ_{k=i}^{n-1} m_k α(i,k)

        Convention: M q̈ + G = B_tau τ  (G opposes gravity for inverted pendulum)
        For inverted pendulum upright at θ=0, G is negative for small positive θ,
        meaning gravity pulls the system away from equilibrium.
        """
        n = self.n
        G = np.zeros(n)
        for i in range(n):
            s = 0.0
            for k in range(i, n):
                s += self.m[k] * self._alpha(i, k)
            G[i] = -self.g * math.sin(q[i]) * s
        return G

    def coriolis_matrix(self, q, qdot):
        """
        Compute Coriolis/centrifugal matrix C(q, q̇) using Christoffel symbols.
        At q̇ = 0 this is zero, so it doesn't affect the linearization.
        """
        n = self.n
        C = np.zeros((n, n))
        eps = 1e-8

        # Numerical Christoffel: c_ijk = 0.5 * (dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i)
        # C_ij = Σ_k c_ijk q̇_k
        M0 = self.mass_matrix(q)
        for k in range(n):
            qp = q.copy(); qp[k] += eps
            qm = q.copy(); qm[k] -= eps
            dMdqk = (self.mass_matrix(qp) - self.mass_matrix(qm)) / (2 * eps)
            for i in range(n):
                for j in range(n):
                    C[i, j] += 0.5 * dMdqk[i, j] * qdot[k]

        # Antisymmetric part
        for k in range(n):
            qp = q.copy(); qp[k] += eps
            qm = q.copy(); qm[k] -= eps
            dMdqk = (self.mass_matrix(qp) - self.mass_matrix(qm)) / (2 * eps)
            for i in range(n):
                for j in range(n):
                    C[i, j] -= 0.5 * dMdqk[j, k] * qdot[k]  # this isn't quite right

        # Simpler: just use numerical C directly
        # C(q,qdot) q̇ = M̊(q) q̇ - 0.5 * (d/dq)(q̇ᵀ M q̇)
        # At qdot=0, C=0, which is all we need for linearization
        return C

    def dynamics(self, q, qdot, tau):
        """
        Compute q̈ = M⁻¹ (B_tau @ τ - C q̇ - G)
        """
        M = self.mass_matrix(q)
        G = self.gravity_vector(q)
        # C is zero at qdot=0 (linearization point), include for completeness
        C = np.zeros((3, 3))  # skip for now, only needed at nonzero velocity
        rhs = self.B_tau @ tau - G  # - C @ qdot
        return np.linalg.solve(M, rhs)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 5: LINEARIZATION AND LQR                              ║
# ╚══════════════════════════════════════════════════════════════════╝

def solve_care_numpy(A, B, Q, R):
    """
    Solve the continuous-time algebraic Riccati equation:
        A'P + PA - PBR⁻¹B'P + Q = 0

    Uses the Hamiltonian eigendecomposition method (works with numpy only).

    Returns P (positive semi-definite solution).
    """
    n = A.shape[0]

    # Form the 2n × 2n Hamiltonian matrix
    R_inv = np.linalg.inv(R)
    BR_inv_BT = B @ R_inv @ B.T

    H = np.zeros((2*n, 2*n))
    H[:n, :n] = A
    H[:n, n:] = -BR_inv_BT
    H[n:, :n] = -Q
    H[n:, n:] = -A.T

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eig(H)

    # Select eigenvectors corresponding to stable eigenvalues (Re < 0)
    stable_idx = np.where(np.real(eigvals) < 0)[0]

    if len(stable_idx) != n:
        print(f"  [LQR] WARNING: Expected {n} stable eigenvalues, "
              f"found {len(stable_idx)}")
        print(f"  [LQR] Eigenvalues: {np.sort(np.real(eigvals))}")
        # Try to proceed anyway with the most negative ones
        sorted_idx = np.argsort(np.real(eigvals))
        stable_idx = sorted_idx[:n]

    # Extract stable eigenspace
    V = eigvecs[:, stable_idx]
    U1 = V[:n, :]    # top n rows
    U2 = V[n:, :]    # bottom n rows

    # P = U2 @ U1⁻¹
    P = np.real(U2 @ np.linalg.inv(U1))

    # Symmetrize (numerical cleanup)
    P = 0.5 * (P + P.T)

    return P


def compute_lqr(dynamics, theta_eq):
    """
    Linearize the 3-link pendulum around the standing equilibrium
    and solve the LQR problem.

    Returns:
        K  : 2×6 gain matrix (u = -K @ δx)
        A  : 6×6 state matrix
        B_lin : 6×2 input matrix
    """
    n = 3   # DOF
    q_eq = theta_eq.copy()
    qdot_eq = np.zeros(n)

    # ── Equilibrium torque ──
    # At equilibrium: M q̈ = 0, q̇ = 0
    # So: B_tau @ τ_eq = G(q_eq)
    G_eq = dynamics.gravity_vector(q_eq)
    B_tau = dynamics.B_tau

    # Least-squares solution (B_tau is 3×2, overdetermined)
    tau_eq, res, _, _ = np.linalg.lstsq(B_tau, G_eq, rcond=None)
    print(f"\n  [LQR] Equilibrium torques: τ_knee={tau_eq[0]:+.3f} N·m, "
          f"τ_hip={tau_eq[1]:+.3f} N·m")

    # Check residual — if large, the system can't be statically balanced
    # with just knee + hip torques
    G_residual = G_eq - B_tau @ tau_eq
    print(f"  [LQR] Gravity residual (should be ~0): {np.linalg.norm(G_residual):.4f}")

    # ── Linearization using numerical differentiation ──
    M_eq = dynamics.mass_matrix(q_eq)
    M_inv = np.linalg.inv(M_eq)

    # A matrix (6×6):
    # ẋ = [q̇; q̈] = [q̇; M⁻¹(B_tau τ - G(q))]
    #
    # At equilibrium (q̇=0, τ=τ_eq):
    # A = [[0, I], [A21, A22]]
    # where A21 = -M⁻¹ ∂G/∂q, A22 = 0 (C vanishes at qdot=0)

    # Numerical gradient of G w.r.t. q
    eps = 1e-6
    dGdq = np.zeros((n, n))
    for j in range(n):
        qp = q_eq.copy(); qp[j] += eps
        qm = q_eq.copy(); qm[j] -= eps
        dGdq[:, j] = (dynamics.gravity_vector(qp) - dynamics.gravity_vector(qm)) / (2*eps)

    A21 = -M_inv @ dGdq  # this gives the acceleration due to position perturbation
    A22 = np.zeros((n, n))  # Coriolis vanishes at qdot=0

    A = np.zeros((2*n, 2*n))
    A[:n, n:] = np.eye(n)       # dq/dt = qdot
    A[n:, :n] = A21             # dqdot/dt = -M⁻¹ ∂G/∂q * δq
    A[n:, n:] = A22

    # B matrix (6×2):
    B_lin = np.zeros((2*n, 2))
    B_lin[n:, :] = M_inv @ B_tau

    # ── Check controllability ──
    C_mat = B_lin.copy()
    AB = B_lin.copy()
    for i in range(2*n - 1):
        AB = A @ AB
        C_mat = np.hstack([C_mat, AB])
    rank = np.linalg.matrix_rank(C_mat)
    print(f"  [LQR] Controllability rank: {rank} / {2*n} "
          f"({'CONTROLLABLE' if rank == 2*n else 'NOT CONTROLLABLE'})")

    # ── Check open-loop eigenvalues ──
    ol_eigs = np.linalg.eigvals(A)
    print(f"  [LQR] Open-loop eigenvalues:")
    for e in sorted(ol_eigs, key=lambda x: -np.real(x)):
        print(f"         {np.real(e):+.4f} {'+' if np.imag(e) >= 0 else ''}"
              f"{np.imag(e):.4f}j")

    # ── LQR cost matrices ──
    # Q penalizes state deviations, R penalizes control effort
    #
    # TUNING GUIDE:
    #   - Increase Q[0,0] (θ1 weight) for tighter lean control
    #   - Increase Q[3,3] (θ̇1 weight) for more damping
    #   - Decrease R for more aggressive control (may cause oscillation)
    #   - Increase R for smoother but slower response
    #
    # θ1 (shin lean) is the most critical — it's the unactuated DOF
    # that directly corresponds to falling over.

    Q = np.diag([
        200.0,   # θ1 — shin lean
        50.0,    # θ2 — thigh angle (relaxed)
        50.0,    # θ3 — upper body (relaxed)
        20.0,    # θ̇1 — shin velocity
        5.0,     # θ̇2 — thigh velocity
        5.0,     # θ̇3 — upper body velocity
    ])

    R = np.diag([
        1000.0,   # knee torque cost — MASSIVE INCREASE for gentler motion
        8000.0,   # hip torque cost  — MASSIVE INCREASE for gentler motion
    ])

    # ── Solve CARE ──
    try:
        from scipy.linalg import solve_continuous_are
        P = solve_continuous_are(A, B_lin, Q, R)
        print("  [LQR] Solved CARE using scipy")
    except ImportError:
        P = solve_care_numpy(A, B_lin, Q, R)
        print("  [LQR] Solved CARE using numpy Hamiltonian method")

    # K = R⁻¹ B' P
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
        print("  Try adjusting Q/R weights or checking model parameters.")

    print(f"\n  [LQR] Gain matrix K (2×6):")
    labels = ["θ1", "θ2", "θ3", "θ̇1", "θ̇2", "θ̇3"]
    for i, name in enumerate(["τ_knee", "τ_hip "]):
        row = "  " + name + " = "
        terms = [f"{K[i,j]:+7.3f}·{labels[j]}" for j in range(6)]
        row += " ".join(terms)
        print(row)

    return K, A, B_lin, tau_eq


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 6: ISAAC SIM STATE MEASUREMENT                        ║
# ╚══════════════════════════════════════════════════════════════════╝

class SagittalStateMeasurer:
    """
    Measures the 6D state [θ1, θ2, θ3, θ̇1, θ̇2, θ̇3] from USD transforms.

    θ_i are absolute angles from vertical in the sagittal (XZ) plane,
    computed from the world-frame positions of foot, knee, hip, and
    upper body COM.

    Angular velocities are computed via windowed differentiation
    (spans ~8 physics steps to smooth out render-rate aliasing).
    """

    RATE_WINDOW = 8  # physics steps for velocity estimation

    def __init__(self, stage, fk_result, planar_params):
        self.stage = stage
        self.theta_eq = planar_params["theta_eq"]

        # Prim references for position readout
        self.torso_prim   = stage.GetPrimAtPath(LINK_PRIM_PATHS["torso2hip"])
        self.chassis1_prim = stage.GetPrimAtPath(LINK_PRIM_PATHS["chasis_motor2motor"])
        self.chassis2_prim = stage.GetPrimAtPath(LINK_PRIM_PATHS["lid_bearing_outer___boss_extrude3"])
        self.thigh1_prim  = stage.GetPrimAtPath(LINK_PRIM_PATHS["thigh_outer"])
        self.thigh2_prim  = stage.GetPrimAtPath(LINK_PRIM_PATHS["stand_steel_00___fillet1"])
        self.shin1_prim   = stage.GetPrimAtPath(LINK_PRIM_PATHS["shin_inner"])
        self.shin2_prim   = stage.GetPrimAtPath(LINK_PRIM_PATHS["nail___fillet1"])

        # Link masses for COM computation
        self.m_torso = 1.698
        self.m_chassis = 1.1141
        self.m_thigh = 0.434
        self.m_shin = 0.1905

        # Local COM offsets
        self.torso_com_local = np.array([-0.0110578, 2.08706e-05, -0.0316562])
        self.c1_com_local = np.array([0, 0, -0.0623004])
        self.c2_com_local = np.array([0, 0, -0.0623004])
        self.th1_com_local = np.array([-0.0223297, -0.118239, 0.0709555])
        self.th2_com_local = np.array([-0.0903635, 0.0795028, 0.0709556])
        self.sh1_com_local = np.array([-0.0782005, 0.0113971, -0.0185949])
        self.sh2_com_local = np.array([0.000800703, 0.0790291, 0.0181051])

        # Foot contact (frozen after settling)
        self.foot_contact_xz = None
        self.contact_captured = False
        self.capture_countdown = 40

        # Windowed differentiation buffers
        self.theta_history = []

    def reset(self):
        self.foot_contact_xz = None
        self.contact_captured = False
        self.capture_countdown = 40
        self.theta_history.clear()

    def _get_world_pos_and_rot(self, prim):
        """Get world position and rotation matrix for a prim."""
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

        state_6d = [δθ1, δθ2, δθ3, θ̇1, θ̇2, θ̇3]
        where δθ_i = θ_i - θ_i_eq
        """
        # ── Get link positions ──
        torso_pos, torso_rot = self._get_world_pos_and_rot(self.torso_prim)
        if torso_pos is None:
            return None, None

        # Shin link positions (to find knee positions)
        shin1_pos, _ = self._get_world_pos_and_rot(self.shin1_prim)
        shin2_pos, _ = self._get_world_pos_and_rot(self.shin2_prim)

        # Thigh link positions (to find hip positions — use thigh origin ≈ hip_pitch location)
        thigh1_pos, _ = self._get_world_pos_and_rot(self.thigh1_prim)
        thigh2_pos, _ = self._get_world_pos_and_rot(self.thigh2_prim)

        if any(x is None for x in [shin1_pos, shin2_pos, thigh1_pos, thigh2_pos]):
            return None, None

        # Average for symmetric model
        knee_xz = np.array([
            0.5 * (shin1_pos[0] + shin2_pos[0]),
            0.5 * (shin1_pos[2] + shin2_pos[2]),
        ])

        # The thigh prim origin is at the hip_pitch joint (after joint rotation)
        hip_xz = np.array([
            0.5 * (thigh1_pos[0] + thigh2_pos[0]),
            0.5 * (thigh1_pos[2] + thigh2_pos[2]),
        ])

        # Upper body COM
        tc = self._link_com_world(self.torso_prim, self.torso_com_local)
        c1c = self._link_com_world(self.chassis1_prim, self.c1_com_local)
        c2c = self._link_com_world(self.chassis2_prim, self.c2_com_local)
        if any(x is None for x in [tc, c1c, c2c]):
            return None, None

        m_upper = self.m_torso + 2 * self.m_chassis
        upper_com = (self.m_torso * tc + self.m_chassis * c1c
                     + self.m_chassis * c2c) / m_upper
        upper_xz = np.array([upper_com[0], upper_com[2]])

        # ── Foot contact ──
        if not self.contact_captured:
            # Estimate foot position from shin positions (they're near the ground)
            # Use shin COMs to extrapolate down
            sh1c = self._link_com_world(self.shin1_prim, self.sh1_com_local)
            sh2c = self._link_com_world(self.shin2_prim, self.sh2_com_local)
            if sh1c is None or sh2c is None:
                return None, None

            # Foot contact X ≈ average shin COM X (shins are nearly vertical)
            # Foot contact Z ≈ 0 (ground)
            avg_shin_x = 0.5 * (sh1c[0] + sh2c[0])
            self.foot_contact_xz = np.array([avg_shin_x, 0.0])

            self.capture_countdown -= 1
            if self.capture_countdown <= 0:
                self.contact_captured = True
                print(f"  [State] Foot contact frozen at X={self.foot_contact_xz[0]*1000:.1f}mm")

        foot_xz = self.foot_contact_xz

        # ── Compute absolute angles ──
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

        # ── Angular velocities via windowed differentiation ──
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

        # ── State deviation from equilibrium ──
        delta_theta = theta - self.theta_eq
        state = np.concatenate([delta_theta, theta_dot])

        # ── Full-body COM for debug ──
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

        debug = {
            "theta": theta,
            "theta_dot": theta_dot,
            "delta_theta": delta_theta,
            "com_x": com_full[0],
            "com_z": com_full[2],
            "foot_x": foot_xz[0],
            "torso_z": torso_pos[2],
        }

        return state, debug


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 7: LQR CONTROLLER FOR ISAAC SIM                       ║
# ╚══════════════════════════════════════════════════════════════════╝

class LQRController:
    """
    Applies LQR full-state feedback to the bipedal robot.

    At each step:
      1. Measure state x = [δθ1, δθ2, δθ3, θ̇1, θ̇2, θ̇3]
      2. Compute desired torques: δτ = -K @ x
      3. Convert to joint position offsets: δq = δτ / kp
      4. Apply: q_target = q_default + δq (mapped to both legs)
    """

    def __init__(self, K, tau_eq, planar_params):
        self.K = K
        self.tau_eq = tau_eq
        self.step = 0

        # Joint stiffnesses for torque → position conversion
        self.kp_knee = JOINT_PARAMS["knee1"]["stiffness"]
        self.kp_hip  = JOINT_PARAMS["hip_pitch1"]["stiffness"]

        # Clamp limits for position offsets (rad)
        # Must be large enough for the LQR to actually command meaningful corrections.
        # The joints have ±3.14 to ±6.28 range, so 1.5 is well within limits.
        # <--- CHANGED FROM 1.5 TO 0.25
        self.max_offset = 0.25

        # Sign mapping — FLIPPED based on run diagnosis:
        # At step 5, δθ1=+5° (leaning fwd), controller sent kn1=+0.3 hp1=+0.35,
        # and lean ACCELERATED. So those signs were pushing the robot FURTHER.
        self.knee_signs = {"knee1": -1.0, "knee2": 1.0}
        self.hip_signs  = {"hip_pitch1": -1.0, "hip_pitch2": +1.0}

        # Sign diagnosis: track δθ1 history to detect wrong signs
        self.dtheta1_history = []
        self.sign_diagnosed = False

    def compute_targets(self, state_6d):
        """
        Compute joint targets from LQR state feedback.

        Returns (targets_dict, ctrl_debug_dict)
        """
        # LQR control law: δτ = -K @ x
        delta_tau = -self.K @ state_6d
        tau_knee_raw = delta_tau[0]
        tau_hip_raw  = delta_tau[1]

        # Convert torques to position offsets
        # τ ≈ kp * δq → δq = τ / kp
        dq_knee_raw = tau_knee_raw / self.kp_knee
        dq_hip_raw  = tau_hip_raw  / self.kp_hip

        dq_knee = np.clip(dq_knee_raw, -self.max_offset, self.max_offset)
        dq_hip  = np.clip(dq_hip_raw,  -self.max_offset, self.max_offset)

        knee_saturated = abs(dq_knee_raw) > self.max_offset
        hip_saturated  = abs(dq_hip_raw)  > self.max_offset

        # Apply to all joints
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

        # ── Automatic sign diagnosis ──
        # Track δθ1 over time. If it monotonically diverges for 30 steps,
        # the signs are almost certainly wrong.
        delta_theta1 = state_6d[0]  # this is δθ1
        self.dtheta1_history.append(delta_theta1)
        if len(self.dtheta1_history) > 40:
            self.dtheta1_history.pop(0)

        if not self.sign_diagnosed and self.step == SETTLE_STEPS + 35:
            self.sign_diagnosed = True
            hist = self.dtheta1_history[-25:]  # last 25 samples
            if len(hist) >= 20:
                # Check if δθ1 is monotonically increasing or decreasing
                diffs = [hist[i+1] - hist[i] for i in range(len(hist)-1)]
                pos_diffs = sum(1 for d in diffs if d > 0)
                neg_diffs = sum(1 for d in diffs if d < 0)
                total = len(diffs)

                growing = abs(hist[-1]) > abs(hist[0]) * 1.5  # diverging?

                print("\n" + "=" * 70)
                print("  AUTOMATIC SIGN DIAGNOSIS (after 35 steps)")
                print("=" * 70)
                print(f"  δθ1 trend: {pos_diffs}/{total} steps increasing, {neg_diffs}/{total} decreasing")
                print(f"  δθ1 start={math.degrees(hist[0]):+.1f}°  end={math.degrees(hist[-1]):+.1f}°  "
                      f"{'DIVERGING' if growing else 'stable'}")

                if growing and pos_diffs > 0.7 * total:
                    print(f"\n  >>> δθ1 is GROWING POSITIVE (leaning more +X) <<<")
                    print(f"  >>> RECOMMENDATION: Flip BOTH sign dicts:")
                    print(f"  >>>   knee_signs = {{'knee1': -1.0, 'knee2': 1.0}}")
                    print(f"  >>>   hip_signs  = {{'hip_pitch1': -1.0, 'hip_pitch2': 1.0}}")
                elif growing and neg_diffs > 0.7 * total:
                    print(f"\n  >>> δθ1 is GROWING NEGATIVE (leaning more -X) <<<")
                    print(f"  >>> RECOMMENDATION: Flip BOTH sign dicts:")
                    print(f"  >>>   knee_signs = {{'knee1': -1.0, 'knee2': 1.0}}")
                    print(f"  >>>   hip_signs  = {{'hip_pitch1': -1.0, 'hip_pitch2': 1.0}}")
                elif growing:
                    print(f"\n  >>> δθ1 is diverging but not monotonic — may need partial flip")
                    print(f"  >>> Try flipping knee_signs ONLY first, then hip_signs ONLY")
                else:
                    print(f"\n  >>> δθ1 appears stable — signs may be correct!")
                    print(f"  >>> If still falling, the issue is elsewhere (model/gains)")
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
            _lqr_ctrl.dtheta1_history.clear()
            _lqr_ctrl.sign_diagnosed = False
        _was_playing = True
        print(f"\n[LQR] PLAY — LQR balance controller active (200 Hz)")
    elif event.type == int(omni.timeline.TimelineEventType.STOP):
        _was_playing = False
        print(f"[LQR] STOPPED")


SETTLE_STEPS = 50  # hold balanced pose for 50 steps (0.25s at 200Hz) before LQR

def physics_step_callback(dt):
    global _lqr_ctrl, _state_measurer

    if not _was_playing or _lqr_ctrl is None or _state_measurer is None:
        return

    stage = omni.usd.get_context().get_stage()

    # ── Safety check ──
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

    # ── Settling phase: hold balanced pose, let PD stabilize ──
    if _lqr_ctrl.step < SETTLE_STEPS:
        for name, default in DEFAULT_POSE.items():
            set_joint_target(name, default)

        # Still measure state so the foot contact gets captured
        # and the velocity buffer fills up
        state, debug = _state_measurer.measure(dt)

        if _lqr_ctrl.step % PRINT_EVERY_N == 0 and debug is not None:
            th = debug["theta"]
            dth = debug["delta_theta"]
            print(f"  [SETTLE {_lqr_ctrl.step:3d}/{SETTLE_STEPS}]  "
                  f"δθ=[{math.degrees(dth[0]):+5.1f} {math.degrees(dth[1]):+5.1f} {math.degrees(dth[2]):+5.1f}]°  "
                  f"COM_x={debug['com_x']*1000:+6.1f}mm  foot_x={debug['foot_x']*1000:+6.1f}mm  "
                  f"h={debug['torso_z']:.3f}m")

        if _lqr_ctrl.step == SETTLE_STEPS - 1:
            print(f"\n  >>> SETTLING COMPLETE — LQR ACTIVE <<<\n")

        _lqr_ctrl.step += 1
        return

    # ── Measure state ──
    state, debug = _state_measurer.measure(dt)
    if state is None:
        return

    # ── Compute LQR targets ──
    targets, ctrl = _lqr_ctrl.compute_targets(state)

    for name, target in targets.items():
        set_joint_target(name, target)

    # ── Debug print ──
    if _lqr_ctrl.step % PRINT_EVERY_N == 0:
        dth = debug["delta_theta"]  # deviation from equilibrium
        thd = debug["theta_dot"]
        th  = debug["theta"]

        # Saturation markers
        ksat = "SAT!" if ctrl["knee_sat"] else "    "
        hsat = "SAT!" if ctrl["hip_sat"] else "    "

        print(f"\n  ── step {_lqr_ctrl.step:5d} ──────────────────────────────────────")
        print(f"  STATE (absolute):  θ=[{math.degrees(th[0]):+6.1f} {math.degrees(th[1]):+6.1f} {math.degrees(th[2]):+6.1f}]°")
        print(f"  STATE (δ from eq): δθ=[{math.degrees(dth[0]):+6.1f} {math.degrees(dth[1]):+6.1f} {math.degrees(dth[2]):+6.1f}]°")
        print(f"  VELOCITY:          θ̇=[{thd[0]:+6.2f} {thd[1]:+6.2f} {thd[2]:+6.2f}] rad/s")
        print(f"  LQR TORQUES:       τ_knee={ctrl['tau_knee']:+7.2f}  τ_hip={ctrl['tau_hip']:+7.2f} N·m")
        print(f"  POSITION OFFSETS:  Δq_knee={ctrl['dq_knee']:+.4f} {ksat}   Δq_hip={ctrl['dq_hip']:+.4f} {hsat}")
        print(f"  JOINT TARGETS:     hp1={targets['hip_pitch1']:+.4f}  hp2={targets['hip_pitch2']:+.4f}  kn1={targets['knee1']:+.4f}  kn2={targets['knee2']:+.4f}")
        print(f"  GEOMETRY:          COM_x={debug['com_x']*1000:+6.1f}mm  foot_x={debug['foot_x']*1000:+6.1f}mm  h={debug['torso_z']:.3f}m")

        # ── SIGN DIAGNOSIS ──
        # The critical question: is δθ1 (lean) being opposed or reinforced?
        # If the controller is correct: when δθ1 > 0 (leaning forward),
        # the next step should show δθ1 decreasing (or at least θ̇1 < 0).
        #
        # We check: is the torque trying to push AGAINST the lean?
        lean_dir = "FWD(+X)" if dth[0] > 0 else "BACK(-X)"
        corr_dir_k = "+" if ctrl['tau_knee'] > 0 else "-"
        corr_dir_h = "+" if ctrl['tau_hip'] > 0 else "-"

        # Simple sign check: are δθ1 and θ̇1 same sign? (diverging = bad)
        if abs(dth[0]) > 0.01:  # only diagnose if meaningful lean
            diverging = (dth[0] * thd[0]) > 0  # lean and velocity same sign = getting worse
            if diverging and abs(thd[0]) > 0.1:
                print(f"  *** SIGN CHECK: Leaning {lean_dir} AND accelerating that way (θ̇1={thd[0]:+.2f})")
                print(f"  *** Controller outputs τ_knee={corr_dir_k} τ_hip={corr_dir_h} — is this opposing the lean?")
                print(f"  *** If this persists for >5 steps, FLIP SIGNS (knee_signs or hip_signs)")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 9: SCENE SETUP                                        ║
# ╚══════════════════════════════════════════════════════════════════╝

async def setup_scene():
    global _joint_prims, _physx_sub, _timeline_sub
    global _lqr_ctrl, _state_measurer
    global DEFAULT_POSE

    # ── Step 1: Find the balanced standing pose ──
    print("\n" + "=" * 70)
    print("  LQR BALANCE CONTROLLER — FINDING BALANCED POSE")
    print("=" * 70)

    # First show what the original pose looks like
    print("\n  Original pose:")
    orig_fk = compute_fk(DEFAULT_POSE)
    com_x_orig = orig_fk["whole_com"][0]
    foot_x_orig = orig_fk["foot_pos"][0]
    print(f"    COM_x={com_x_orig*1000:+.1f}mm  foot_x={foot_x_orig*1000:+.1f}mm  "
          f"offset={(com_x_orig-foot_x_orig)*1000:+.1f}mm")

    # Find balanced pose
    balanced_pose, balanced_fk = find_balanced_pose()
    DEFAULT_POSE = balanced_pose  # update global

    # ── Step 2: Compute LQR gains at the balanced pose ──
    print("\n" + "=" * 70)
    print("  COMPUTING LQR GAINS AT BALANCED POSE")
    print("=" * 70)

    fk = balanced_fk
    params = extract_planar_params(fk)
    dynamics = PlanarDynamics(params)

    K, A, B_lin, tau_eq = compute_lqr(dynamics, params["theta_eq"])

    # ── Now set up the Isaac Sim scene ──
    stage = omni.usd.get_context().get_stage()

    if stage.GetPrimAtPath("/World"):
        omni.kit.commands.execute("DeletePrims", paths=["/World"])
        await omni.kit.app.get_app().next_update_async()

    UsdGeom.Xform.Define(stage, "/World")

    # Physics scene (200 Hz)
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    physx_api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx_api.CreateTimeStepsPerSecondAttr(200)

    # Ground plane
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
    robot_prim = stage.DefinePrim("/World/Robot")
    robot_prim.GetReferences().AddReference(USD_PATH)
    await omni.kit.app.get_app().next_update_async()
    await omni.kit.app.get_app().next_update_async()

    # Friction on robot
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if path.startswith("/World/Robot"):
            if prim.HasAPI(UsdPhysics.CollisionAPI) or "collision" in prim.GetName().lower():
                mat = UsdPhysics.MaterialAPI.Apply(prim)
                if mat:
                    mat.CreateStaticFrictionAttr(0.9)
                    mat.CreateDynamicFrictionAttr(0.8)

    # Spawn position
    xform = UsdGeom.Xformable(robot_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, SPAWN_HEIGHT))

    # ── Configure joints ──
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

                # Fix 3: Set the INITIAL joint position to the balanced pose.
                # Without this, joints start at 0 and the PD has to drive them
                # to the target, creating a large transient where the robot is
                # in a completely wrong configuration.
                joint_state = PhysxSchema.JointStateAPI.Apply(prim, "angular")
                if joint_state:
                    joint_state.CreatePositionAttr().Set(np.degrees(target_rad))
                    joint_state.CreateVelocityAttr().Set(0.0)

                print(f"    {name:15s}  target={target_rad:+.3f} rad ({np.degrees(target_rad):+.1f}°)  "
                      f"Kp={p['stiffness']:<5}  Kd={p['damping']:<5}")

    # ── Initialize state measurer and controller ──
    _state_measurer = SagittalStateMeasurer(stage, fk, params)
    _lqr_ctrl = LQRController(K, tau_eq, params)

    # ── Print summary ──
    print(f"\n  Joints configured: {len(_joint_prims)}")
    print(f"  Physics timestep:  200 Hz")
    print(f"\n  LQR GAIN MATRIX (2×6):")
    labels = ["θ1", "θ2", "θ3", "θ̇1", "θ̇2", "θ̇3"]
    for i, name in enumerate(["τ_knee", "τ_hip "]):
        terms = [f"{K[i,j]:+7.3f}·{labels[j]}" for j in range(6)]
        print(f"    {name}: {' '.join(terms)}")

    print(f"\n  ──────────────────────────────────────────")
    print(f"  TUNING GUIDE:")
    print(f"    If robot falls FASTER with controller on:")
    print(f"      → Flip signs in LQRController.knee_signs or .hip_signs")
    print(f"    If too sluggish:")
    print(f"      → Increase Q[0,0] (θ1 weight) or decrease R")
    print(f"    If oscillating:")
    print(f"      → Increase R or increase Q[3,3] (θ̇1 damping)")
    print(f"    If steady-state lean:")
    print(f"      → Adjust DEFAULT_POSE hip_pitch values")
    print(f"  ──────────────────────────────────────────")
    print(f"\n  >>> Press PLAY to start LQR balancing <<<")
    print("=" * 70 + "\n")

    # Register callbacks
    _physx_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(
        physics_step_callback
    )
    timeline = omni.timeline.get_timeline_interface()
    stream = timeline.get_timeline_event_stream()
    _timeline_sub = stream.create_subscription_to_pop(_on_timeline_event)

    print("[LQR] Physics callback + timeline events registered")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  LAUNCH                                                        ║
# ╚══════════════════════════════════════════════════════════════════╝

asyncio.ensure_future(setup_scene())