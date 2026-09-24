"""pad_contact.py — per-environment ball-foot pad compliance, as a PLANT FORCE.

Why this module exists
----------------------
The ball foot's compliance cannot be randomized through the contact material.
PhysX compliant-contact gains (`physxMaterial:compliantContactStiffness` /
`:compliantContactDamping`) are MATERIAL-level properties, and the material prim
is shared by every environment in the stage.  The articulation view can write
only the three per-shape tensor channels (static friction, dynamic friction,
restitution), so a per-env pad stiffness has no home in the physics material at
all — measured 2026-09-12: editing the material mid-process reaches nothing
without a full stage re-attach, which is why the play hotkeys needed a restart.

So the pad is modelled where the DR can reach: as an external force applied to
the shin body from the environment step, one stiffness/damping pair per env per
foot — the same shape as belt_knee_actuator's series path, and for the same
reason.

Model
-----
The ball is a rigid sphere of radius R whose centre trails the shin body origin
by a fixed body-frame offset (both measured from the USD at reset, not
hardcoded — see `measure_ball_geometry`).  A pad at the contact carries a
damped spring in series with the rigid contact, so the force law is

    F = -(k * x + c * v_n)          along the floor normal (world +z here)

with `x` the pad compression and `v_n` the ball centre's normal velocity.
`c` is entered as a damping RATIO of the pad's own mode with the mass that
actually participates in a single-support catch (`pad_effective_mass`, ~1 kg —
NOT the whole body, which would over-damp by sqrt(m_robot/m_eff)).

`x` is advanced once per PHYSICS STEP by the semi-implicit integral the project
already uses for the belt (see `PadContact._advance`), and the damping gain is
capped at `pad_damping_stability_factor * m / dt`, because an explicit damped
contact oscillates and diverges above `2*m/dt`.  A DR band that reaches the cap
is a band that silently stops meaning what it says, so the cap is exposed as
`PadContact.clamped` and the env warns once per run instead of hiding it.

With `pad_stiffness_max = 0` (the default) the module is inert: it returns
zero force and never touches the state, so the cfg default reproduces the
legacy plant bit-for-bit.
"""

from __future__ import annotations

import math

import torch

FLOOR_NORMAL = (0.0, 0.0, 1.0)


def pad_damping_for_ratio(stiffness_n_per_m: float, damping_ratio: float,
                          effective_mass_kg: float) -> float:
    """Damping (N s/m) for a target pad damping ratio: c = 2*zeta*sqrt(k*m)."""
    if stiffness_n_per_m < 0.0:
        raise ValueError(f"pad stiffness must be >= 0, got {stiffness_n_per_m!r}")
    if damping_ratio < 0.0 or not math.isfinite(damping_ratio):
        raise ValueError(f"pad damping ratio must be finite and >= 0, got {damping_ratio!r}")
    if effective_mass_kg <= 0.0:
        raise ValueError(f"effective mass must be > 0, got {effective_mass_kg!r}")
    return 2.0 * damping_ratio * math.sqrt(stiffness_n_per_m * effective_mass_kg)


def damping_stability_cap(effective_mass_kg: float, physics_dt: float,
                          factor: float) -> float:
    """Largest damping this integrator can carry: factor * m / dt.

    An explicit damped contact is stable only while c*dt/m < 2; the factor
    keeps a margin below that (default 0.5, i.e. a quarter of the hard limit)
    because the pad state is advanced at the physics rate while the rest of the
    plant is integrated by PhysX at the same rate with its own solver.
    """
    if physics_dt <= 0.0:
        raise ValueError(f"physics dt must be > 0, got {physics_dt!r}")
    if factor <= 0.0 or factor >= 1.0:
        raise ValueError(f"stability factor must be in (0, 1), got {factor!r}")
    return float(factor) * float(effective_mass_kg) / float(physics_dt)


class PadContact:
    """Per-env, per-foot ball-pad springs.  Inert while `stiffness_max == 0`.

    Buffers are (num_envs, 2) — column 0 left foot, column 1 right foot — and
    follow the env's joint/body ordering (L then R), like every other DR array
    in rl_env_p6.
    """

    #: Pad compression beyond this is a modelling failure, not a plant state.
    #: `x` is reset (not clipped) so a blown-up env recovers instead of pinning
    #: the foot to the floor with a huge force.
    MAX_COMPRESSION_FACTOR = 4.0

    def __init__(self, num_envs: int, device, ball_local_offset: torch.Tensor,

                 ball_radius: float, pad_thickness: float = 0.002,
                 stiffness: torch.Tensor = None,
                 damping: torch.Tensor = None, effective_mass: float = 1.0,
                 physics_dt: float = 1.0 / 500.0, stability_factor: float = 0.5,
                 enabled: bool = False):
        self.num_envs = int(num_envs)
        self.device = device
        self.enabled = bool(enabled)
        self.ball_radius = float(ball_radius)
        # The pad's UNDEFORMED contact plane: the ball's radius less the pad's
        # own thickness.  The ball rests here in a relaxed stand, and the pad
        # compresses below it under load, so a stand is not a loaded contact.
        # This is the offset PhysX calls the rest offset, and it is what makes
        # the pad able to carry the robot AT ALL: with the reference at the bare
        # radius the pad can only push while the ball penetrates the floor, so
        # its equilibrium force is zero and the measured sag came out 6x too
        # small.  `pad_thickness` is the one geometry number this model cannot
        # measure — the ball's own radius comes from the USD — so it is a cfg
        # field to be pinned from the bench.
        self.pad_thickness = float(pad_thickness)
        self.reference_radius = self.ball_radius - self.pad_thickness
        # Physical floor on a meaningful stiffness: a sag larger than the ball
        # itself is not a pad, it is a modelling error, and a DR band that
        # contains one would put the robot through the floor.
        self.max_compression = self.MAX_COMPRESSION_FACTOR * self.ball_radius
        self.effective_mass = float(effective_mass)
        self.physics_dt = float(physics_dt)
        #: Force ceiling, refreshed by `set_parameters` from the drawn stiffness.
        #: Zero until a stiffness is written, and a rigid pad never reaches it.
        self.max_force = 0.0

        # (2, 3) body-frame offsets from each shin origin to its ball centre.
        self.ball_local_offset = torch.as_tensor(
            ball_local_offset, dtype=torch.float32, device=device
        ).reshape(2, 3)
        # (num_envs, 2) — resolved spring pair per env per foot.
        self.stiffness = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=device)
        self.damping = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=device)
        if stiffness is not None:
            self.stiffness.copy_(stiffness)
        if damping is not None:
            self.damping.copy_(damping)
        self._cap = damping_stability_cap(self.effective_mass, self.physics_dt, stability_factor)
        self.clamped = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=device)
        #: Cumulative count of foot-steps whose deflection ran past the ball's
        #: geometry.  The env prints it once instead of silently resetting.
        self.blowups = 0

        # Pad compression per foot, advanced at the physics rate, plus the
        # previous value: the state's own rate is what tells a separating ball
        # from a pressing one.
        self.compression = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=device)
        self._prev_compression = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=device)
        # Live contact readings, for logging / the smoke test.
        self.last_penetration = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=device)
        self.last_normal_vel = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=device)
        self.last_normal_force = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=device)

    # ── configuration ────────────────────────────────────────────────────
    @property
    def damping_cap(self) -> float:
        """Largest damping this integrator can carry, N s/m (see set_parameters)."""
        return self._cap

    def set_parameters(self, env_ids, stiffness: torch.Tensor, damping: torch.Tensor):
        """Write a resolved (k, c) pair for `env_ids`; caps the damping.

        The force ceiling moves with the stiffness: it is the force the pad
        carries at the extreme of its own travel, so a soft pad gets a soft
        ceiling and a stiff one is allowed the force its travel implies.
        """
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.stiffness[env_ids] = stiffness
        capped = torch.clamp(damping, max=self._cap)
        self.clamped[env_ids] = capped < damping
        self.damping[env_ids] = capped
        k_max = float(self.stiffness.max().item()) if self.stiffness.numel() else 0.0
        self.max_force = k_max * self.max_compression

    def resolved_kc(self, stiffness: torch.Tensor, damping_ratio) -> tuple[torch.Tensor, torch.Tensor]:
        """(k, c) for a drawn stiffness, in the ratio form the cfg exposes.

        ``damping_ratio`` is a scalar for the pinned form, or a per-env tensor
        for the S7 DR band — the two forms exist because a band whose every
        episode draws a different zeta cannot be expressed as one scalar, and
        because the ratio is the parameter that maps to a measurable quantity
        while an explicit N s/m band does not (it means a different zeta at
        every stiffness, which is what the S5 damping band got wrong).

        k = 0 keeps c = 0: a rigid cell has no spring to damp, and asking for a
        ratio there would be a silent division by zero.
        """
        k = stiffness.to(torch.float32)
        if torch.is_tensor(damping_ratio):
            ratio = damping_ratio.to(device=k.device, dtype=torch.float32)
            c = 2.0 * ratio * torch.sqrt(torch.clamp(k, min=0.0) * self.effective_mass)
            c = torch.where(k > 0.0, c, torch.zeros_like(c))
        else:
            c = damping_for_ratio_tensor(k, damping_ratio, self.effective_mass)
        return k, c

    def reset(self, env_ids=None):
        """Clear the pad state.  Called on episode reset: the pad is a plant
        state, so carrying it across episodes would leak energy between them."""
        if env_ids is None:
            self.compression.zero_()
            self._prev_compression.zero_()
        else:
            self.compression[env_ids] = 0.0
            self._prev_compression[env_ids] = 0.0

    # ── the force ────────────────────────────────────────────────────────
    def compute(self, contact_pos_w: torch.Tensor, contact_vel_w: torch.Tensor,
                body_origin_w: torch.Tensor):
        """Forces and torques for the two feet, in world frame.

        Args:
            contact_pos_w: (N, 2, 3) ball CENTRE world positions.
            contact_vel_w: (N, 2, 3) ball centre world velocities.
            body_origin_w: (N, 2, 3) shin body origins, for the moment arm.

        Returns:
            (forces, torques), each (N, num_bodies, 3), to be summed into the
            env's external-force buffers.  A rigid pad returns zeros and leaves
            the state untouched.
        """
        n = self.num_envs
        if not self.enabled or not bool((self.stiffness > 0.0).any()):
            return None, None

        # Penetration of the ball INTO THE PAD, not into the floor: measured
        # from the pad's undeformed contact plane, so a relaxed stand sits at
        # zero compression instead of zero force.  See `reference_radius`.
        # The floor is the z=0 plane of the env's own frame: env origins sit on
        # it and every contact threshold in the env
        # (`no_flight_contact_threshold`) is measured the same way.
        penetration = self.reference_radius - contact_pos_w[..., 2]
        penetrating = penetration > 0.0

        vn = contact_vel_w[..., 2]
        step = self._advance(penetration, vn, penetrating)

        # The floor pushes the foot UP (+z), and `step` is already a magnitude,
        # so the world-frame force is +z.  Applied at the CONTACT POINT:
        # `contact_pos_w` is the ball centre, so the moment arm runs from the
        # shin origin to the centre, and the small difference between the centre
        # and the deformed contact point is inside the pad's own deflection.
        force_z = torch.where(penetrating, step, torch.zeros_like(step))
        self.last_penetration.copy_(penetration)
        self.last_normal_vel.copy_(vn)
        self.last_normal_force.copy_(force_z)

        forces = torch.zeros(n, 2, 3, device=self.device)
        forces[..., 2] = force_z
        lever = contact_pos_w - body_origin_w
        torques = torch.cross(lever, forces, dim=-1)
        return forces, torques

    def _advance(self, penetration: torch.Tensor, vn: torch.Tensor,
                 penetrating: torch.Tensor) -> torch.Tensor:
        """Update the pad state one physics step; return the normal force.

        The pad's deflection `x` is carried as its own state, driven by the
        MEASURED overlap of the ball into the pad surface, and advanced
        semi-implicitly at the physics rate — the same shape as the belt series
        integrator, and for the same reason: the solver must stay stable at the
        stiff end of the band, not just where the band happens to start.

        Three properties fall out, each one here because its absence was
        measured:

        * Steady load: at rest the damping term is zero and `k*x` carries the
          weight, so a loaded pad sinks by exactly `m*g/k` and a relaxed one
          rests on its reference plane.  Deriving the force from raw floor
          penetration instead made the settled sag 6x too small — with the
          reference at the bare ball radius the pad can only push while the ball
          is INSIDE the floor, so its equilibrium force is zero.
        * Damping acts on the relative approach speed `(approach - x_rate)` and
          is then clamped at zero: a pad resists being compressed and releases
          freely.  Damping the absolute speed turned the pad into glue that
          hauled the foot back down as the spring relaxed (a 500 N pull,
          measured), and clamping only the input left a doubled damping term
          that halved the settled sag (245 um of the analytic 490 um).
        * The damping is implicit in the state update, so `c` cannot push the
          integration over the stability limit however stiff the band is.
        """
        dt = self.physics_dt
        m = self.effective_mass
        k, c = self.stiffness, self.damping

        # vn is the ball centre's world +z velocity: a descending ball has
        # vn < 0, so the compression advances along -vn and the damping acts on
        # that same approach speed.
        approach = torch.clamp(-vn, min=0.0)
        overlap = torch.clamp(penetration, min=0.0)
        x_prev = self.compression

        # The state relaxes toward the measured overlap.  Writing the update
        # this way — rather than integrating the deflection against its own
        # spring force — is what makes the equilibrium exact: at rest the target
        # IS the deflection, so nothing moves and `k*x` carries the weight that
        # the overlap implies.  Integrating it as a free state instead left the
        # force pinned at m*g while the deflection ran away, because nothing
        # tied it to where the ball actually was (measured 2026-09-12).
        #
        # A relaxation is also what a real pad does: its surface cannot respond
        # instantly, and `tau = m/c` is its own response time.
        tau = torch.clamp(m / torch.clamp(c, min=1e-9), min=4.0 * dt)
        alpha = torch.clamp(dt / tau, max=1.0)
        x_new = x_prev + alpha * (overlap - x_prev)
        x_new = torch.clamp(x_new, min=0.0)

        # A deflection past the geometry is a modelling failure: reset it rather
        # than applying a force the pad could not physically carry.
        blown = x_new > self.max_compression
        x_new = torch.where(blown, torch.zeros_like(x_new), x_new)

        # Force = spring alone: the damping already determined HOW x_new got
        # here (through tau), and adding `c * v` on top would count it twice —
        # measured as a halved settled sag (245 um of the analytic 490 um at
        # k=20000, m=1 kg, zeta=1, exactly the factor of two a doubled damping
        # term predicts).
        force = k * x_new
        force = torch.clamp(force, max=self.max_force)
        force = torch.where(penetrating, force, torch.zeros_like(force))

        self._prev_compression = x_prev
        self.compression = x_new
        if bool(blown.any()):
            self.blowups += int(blown.sum().item())
        return force


def damping_for_ratio_tensor(stiffness: torch.Tensor, damping_ratio: float,
                             effective_mass: float) -> torch.Tensor:
    """Elementwise c = 2*zeta*sqrt(k*m), safe at k = 0 (returns 0 there)."""
    if damping_ratio != damping_ratio or damping_ratio < 0.0:   # NaN or negative
        raise ValueError(f"pad damping ratio must be finite and >= 0, got {damping_ratio!r}")
    c = 2.0 * damping_ratio * torch.sqrt(torch.clamp(stiffness, min=0.0) * effective_mass)
    return torch.where(stiffness > 0.0, c, torch.zeros_like(c))


def fit_sphere(points) -> tuple[tuple[float, float, float], float, int, int, float]:
    """Least-squares sphere through ``points`` (an (N, 3) float array).

    A sphere through a cloud is linear in (cx, cy, cz, R^2 - |c|^2) after
    expanding |p - c|^2 = R^2, so one lstsq solves it.  The fit is then
    re-run on the inliers only: the first pass is dragged by the far outliers
    (this mesh's flange), and the ball's true centre is what the pad force
    direction depends on.

    Returns:
        (centre, radius, n_inliers, n_distinct, max_abs_residual_over_inliers),
        where `n_distinct` is the duplicate-collapsed point count.  Callers
        must judge the fit against `n_distinct`, never against `len(points)`:
        a triangle-soup mesh repeats each corner once per incident triangle.

    Why not the bounding box: `part_3` is a 40 mm sphere PLUS a boss that
    reaches r = 30.6 mm, so the AABB is a 41.6 mm cube whose centre sits
    1.85 mm off the true centre.  The AABB passes a cube-ness test (its
    extents differ by 3.9 %), so a bbox-based radius and centre look
    plausible and are both wrong — 20.8 mm instead of 20.0, and a contact
    point displaced by nearly 2 mm, which is 10 % of the ball.
    """
    import numpy as np

    pts = np.asarray(points, dtype=float)
    # Triangle soup: the same corner appears once per incident triangle, so a
    # 5438-triangle mesh arrives as 16314 points with every vertex repeated
    # (this is why the count looks alarming next to the STL's 2721 unique
    # vertices — it is the same surface).  Dedupe before fitting, or the
    # duplicated corners weight the fit by local triangle density.
    pts = np.unique(pts.round(7), axis=0)
    if len(pts) < 4:
        raise ValueError(f"need >= 4 distinct points to fit a sphere, got {len(pts)}")

    def _solve(P):
        A = np.hstack([2.0 * P, np.ones((len(P), 1))])
        rhs = (P ** 2).sum(axis=1)
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        c = sol[:3]
        return c, float(np.sqrt(max(sol[3] + float((c ** 2).sum()), 0.0)))

    tol = 0.0004

    def _trim(seed_c, seed_R, work):
        """Refit on ``work``, keeping only points within ``tol`` of the sphere.

        Trim-to-convergence on a cloud with a minority contaminant: the boss
        points fall outside ``tol`` and drop out after a few passes.  Runs on
        a SUBSAMPLE — the inlier set is what is being found here, and a fifth
        of the cloud settles the same model (the final refit below uses every
        point, so the returned numbers are not subsample-limited).
        """
        c, R = seed_c, seed_R
        keep = None
        for _ in range(12):
            d = np.linalg.norm(work - c, axis=1)
            keep = np.abs(d - R) < tol
            if keep.sum() < 4:
                return None
            c_new, R_new = _solve(work[keep])
            if np.allclose([*c_new, R_new], [*c, R], atol=1e-10):
                c, R = c_new, R_new
                break
            c, R = c_new, R_new
        if not (np.isfinite(R) and 0.0 < R < 1.0):
            return None
        # Score against the WHOLE cloud, so a lucky subsample cannot win.
        d_all = np.linalg.norm(pts - c, axis=1)
        return c, R, int((np.abs(d_all - R) < tol).sum())

    # The composite is a ball plus a boss, and a single least-squares pass is
    # dragged between the two: with a fat enough boss the converged "sphere"
    # sits partway out and matches neither surface (measured on this mesh: it
    # lands at R = 21.6 mm with 1942 inliers instead of R = 20.0 with 2201).
    # So SEED FROM THE BALL ITSELF rather than from the outer shell — a sphere
    # through any four surface points is exact, and the ball is the majority
    # of the mesh, so four-point seeds land on it often and the boss seeds die
    # in the trim.  Keep the seed with the largest inlier set.
    rng = np.random.default_rng(0)
    n_work = min(len(pts), 1500)
    work = pts if n_work >= len(pts) else pts[rng.choice(len(pts), size=n_work, replace=False)]
    best = None
    for _ in range(120):
        idx = rng.choice(len(work), size=4, replace=False)
        seed = work[idx]
        # Four near-coplanar points define a wildly wrong sphere; |det| of the
        # three edge vectors is 6x the tetra volume, so it is the test.
        if abs(np.linalg.det(seed[1:] - seed[0])) < 1e-14:
            continue
        try:
            c0, R0 = _solve(seed)
        except np.linalg.LinAlgError:
            continue
        if not (np.isfinite(R0) and 0.0 < R0 < 1.0):
            continue
        got = _trim(c0, R0, work)
        if got is None:
            continue
        c, R, n_in = got
        if best is None or n_in > best[2]:
            best = (c, R, n_in)
        # Early exit: on this asset the ball is ~80 % of the distinct
        # vertices, so a model already explaining three quarters of them
        # cannot be beaten by a meaningful margin, and the remaining seeds
        # cost an lstsq each.  Measured 2026-09-13: the winning model appears
        # within the first few seeds and the answer is identical either way.
        if best[2] >= 0.75 * len(pts):
            break
    if best is None:
        raise ValueError("sphere fit found no consistent model in the point cloud")

    # Refit on every inlier of the winning model: the answer must come from
    # the whole cloud, not from the subsample that happened to find it.
    c, R, _ = best
    d_all = np.linalg.norm(pts - c, axis=1)
    final_keep = np.abs(d_all - R) < tol
    if final_keep.sum() < 4:
        raise ValueError("sphere fit collapsed to fewer than 4 inliers")
    c, R = _solve(pts[final_keep])

    final_res = np.abs(np.linalg.norm(pts - c, axis=1) - R)
    inlier_mask = final_res < tol
    return ((float(c[0]), float(c[1]), float(c[2])), float(R),
            int(inlier_mask.sum()), int(len(pts)), float(final_res[inlier_mask].max()))


def _measure_ball_on_body(stage, body_path: str, ball_prim_hints: tuple[str, ...]) -> tuple[tuple[float, float, float], float, str]:
    """Fitted ball centre (body frame) + radius for ONE foot link."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    xform_cache = UsdGeom.XformCache()
    body_prim = stage.GetPrimAtPath(body_path)
    if not body_prim.IsValid():
        raise RuntimeError(f"foot body prim not found: {body_path}")
    body_to_world = xform_cache.GetLocalToWorldTransform(body_prim)

    def collision_prims():
        # TraverseInstanceProxies is REQUIRED, not decorative: every link's
        # `collisions` and `visuals` scope in this asset is IsInstance()=True,
        # and a plain PrimRange does NOT descend into an instanced subtree.
        # Without it this walk finds zero collision prims on a healthy stage
        # (measured 2026-09-13) and raises "no spherical ball collision prim
        # found", which reads like a missing asset and is not.
        return [p for p in Usd.PrimRange(body_prim, Usd.TraverseInstanceProxies())
                if p.HasAPI(UsdPhysics.CollisionAPI)]

    present = collision_prims()
    chosen = None
    for prim in present:
        if any(hint in prim.GetName() or hint in str(prim.GetPath()) for hint in ball_prim_hints):
            chosen = prim
            break
    if chosen is None:
        raise RuntimeError(
            f"No ball collision prim found under {body_path}; looked for {ball_prim_hints}. "
            f"Collision prims present ({len(present)}): {[str(p.GetPath()) for p in present][:20]}"
        )

    # Read the prim's own mesh points (NOT ComputeLocalBound: the flange makes
    # the AABB centre wrong; and PhysX's convexHull approximation is a
    # different thing again from the authored mesh).
    points = None
    for desc in Usd.PrimRange(chosen, Usd.TraverseInstanceProxies()):
        if desc.IsA(UsdGeom.Mesh):
            attr = UsdGeom.Mesh(desc).GetPointsAttr().Get()
            if attr:
                points = [(float(p[0]), float(p[1]), float(p[2])) for p in attr]
                break
    if not points:
        raise RuntimeError(f"ball collision prim {chosen.GetPath()} carries no mesh points")

    centre_local, radius_here, n_in, n_distinct, max_res = fit_sphere(points)
    if not math.isfinite(radius_here) or radius_here <= 0.0:
        raise RuntimeError(f"ball radius came out non-positive ({radius_here}) from {chosen.GetPath()}")
    # A sphere fit that only explains a small fraction of the surface is not a
    # ball, whatever the prim is called: better to fail than to place the
    # contact point on the boss.  Judged against the DEDUPED count -- the raw
    # point list is triangle soup and would make every mesh look 1/3 spherical.
    if n_in < 0.5 * n_distinct:
        raise RuntimeError(
            f"ball collision mesh {chosen.GetPath()} is not sphere-like: only "
            f"{n_in}/{n_distinct} distinct vertices lie within 0.4 mm of the "
            f"best-fit sphere (R={radius_here:.6f}, max residual {max_res:.6f} m)"
        )

    centre_world = xform_cache.GetLocalToWorldTransform(chosen).Transform(
        Gf.Vec3d(*centre_local))
    centre_in_body = body_to_world.GetInverse().Transform(centre_world)
    return (float(centre_in_body[0]), float(centre_in_body[1]), float(centre_in_body[2])), radius_here, str(chosen.GetPath())


def measure_ball_geometry(stage, foot_body_names: tuple[str, str],
                          body_paths: tuple[str, str],
                          ball_prim_hints: tuple[str, ...] = ("part_3",),
                          verbose: bool = False) -> tuple[torch.Tensor, float]:
    """Ball centre offset (body frame) and radius, read from the live stage.

    The ball is `part_3`: a **40 mm spherical cap on a boss**, NOT a clean
    sphere.  Measured from the authored STL 2026-09-13: a least-squares sphere
    of R = 20.003 mm carries 2201 of 2721 unique vertices; the other 520 sit up
    to 10.6 mm outside it (the boss/flange, at both ends of the mesh's local
    z).  So the radius and the centre are both FITTED here rather than taken
    from the bounding box — the bbox is a 41.6 mm cube, passes a cube-ness
    test at 3.9 %, and yields R = 20.8 mm with a centre 1.85 mm off, which is
    9 % of the ball radius of error in the contact point.

    Carried as a SEPARATE collision on each foot — off the `shin_inner` link on
    the left and off `nail___fillet1` on the right, which is itself a link
    under the knee joint rather than a child of the shin.  It is measured
    rather than hardcoded for two reasons: the env's `_toe*_local_offset` is
    the NAIL origin, 59.06 mm short of the ball on both feet (URDF-computed
    2026-09-12; the 2026-09-08 note calls it 5.8 cm), and composing the
    shin->nail chain by hand is the easy way to put the right foot's contact
    point ~9 cm out.  A future asset change should move this number with it
    instead of silently invalidating the contact model.

    Returns:
        (offsets, radius) with offsets (2, 3) in each shin's body frame, ordered
        left, right as `foot_body_names` / `body_paths` are given.
        Raises if the ball prim cannot be identified, because a pad with the
        wrong contact point is worse than no pad.
    """
    offsets = []
    radius = None
    for body_path in body_paths:
        centre, radius_here, prim_path = _measure_ball_on_body(stage, body_path, ball_prim_hints)
        if verbose:
            print(f"    ball on {body_path}: centre {centre} R={radius_here:.6f} from {prim_path}")
        offsets.append(centre)
        radius = radius_here if radius is None else radius

    if radius is None or radius <= 0.0:
        raise RuntimeError("ball radius came out non-positive")
    return torch.tensor(offsets, dtype=torch.float32), float(radius)


__all__ = [
    "PadContact",
    "pad_damping_for_ratio",
    "damping_for_ratio_tensor",
    "damping_stability_cap",
    "measure_ball_geometry",
]
