r"""Where the bow shock is, before anything is solved.

The pipeline's four failures reduced to one cause -- an isotropic tetrahedral
mesh cannot hold a bow shock still -- and the measurement that proved it was
that the finest grid had **zero** nodes between the shock and the body
→ :mod:`aether.cfd.forces` and ``dataset-shock-standoff-vs-billig``.

Fixing that needs the shock's location *before* a solution exists, so cells can
be put where it will be. :cite:`billig1967shock` supplies exactly that for
spherically blunted cones: assume the detached shock is a **hyperbola
asymptotic to the downstream shock angle**, and pin it with two experimentally
correlated numbers -- the standoff distance and the vertex radius of curvature.

That is the whole reason this is worth having. LAURA-style adaptation solves,
finds the shock, re-meshes and re-solves. For this body family Billig places it
analytically, so a shock-fitted grid is a **one-pass** construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Protocol

import numpy as np

_Array = np.ndarray

__all__ = [
    "BilligShock",
    "ConicalShock",
    "MachCone",
    "Oriented",
    "ShockEnvelope",
    "Validity",
    "WedgeShock",
    "billig_shock",
    "envelope_for",
    "freestream_axis",
]


@dataclass(frozen=True)
class BilligShock:
    r"""A hyperbolic bow shock, in the body frame.

    Billig writes the shock with :math:`x` measured from the **centre of the
    nose sphere, positive upstream**, so his vertex sits at :math:`x = R +
    \Delta`. This pipeline's meshes put the nose *tip* at the origin with
    :math:`x` positive **downstream**, so the two differ by
    :math:`x_{\text{body}} = R - x_{\text{Billig}}` and the vertex lands at
    :math:`-\Delta`, which is where a standoff distance ought to be. Both are
    exposed rather than one being silently assumed.
    """

    nose_radius: float
    standoff: float
    """:math:`\\Delta`, Ambrosio & Wortman Eq. (2) as quoted by Billig."""

    vertex_radius: float
    """:math:`R_c`, Billig Eq. (4). The shock's curvature at its nose."""

    asymptote: float
    """:math:`\\theta` in radians -- the angle the hyperbola tends to far downstream.

    The freestream Mach angle for a body with no conical afterbody, or **the
    attached cone shock angle** when there is one. Billig is explicit that it is
    the latter for "a cone or wedge afterbody", and the difference is not small:
    a 10 degree cone at Mach 8 turns its shock through about 14 degrees where
    the Mach angle is 7.2, so using the Mach angle would put the shock at twice
    the distance from the body by mid-length.
    """

    mach: float

    def radius_at(self, x_body: np.ndarray | float) -> np.ndarray:
        """Shock radius :math:`y` at a body-frame axial station. NaN upstream of the vertex."""
        x_billig = self.nose_radius - np.asarray(x_body, dtype=np.float64)
        t = np.tan(self.asymptote)
        # Invert Eq. (1) for y.
        inner = (self.nose_radius + self.standoff - x_billig) * t**2 / self.vertex_radius + 1.0
        # NaN where the station is ahead of the vertex and no shock radius
        # exists. np.maximum(v, np.nan) would propagate NaN everywhere instead.
        squared = np.where(inner >= 1.0, inner**2 - 1.0, np.nan)
        return np.sqrt(squared) * self.vertex_radius / t

    def axial_at(self, y: np.ndarray | float) -> np.ndarray:
        r"""Body-frame :math:`x` of the shock at radius :math:`y` -- Billig Eq. (1).

        .. math::

            x = R + \Delta - R_c \cot^2\theta
                \left[\left(1 + \frac{y^2\tan^2\theta}{R_c^2}\right)^{1/2} - 1\right]
        """
        y = np.asarray(y, dtype=np.float64)
        t = math.tan(self.asymptote)
        x_billig = (
            self.nose_radius
            + self.standoff
            - self.vertex_radius / t**2 * (np.sqrt(1.0 + y**2 * t**2 / self.vertex_radius**2) - 1.0)
        )
        return self.nose_radius - x_billig


def billig_shock(
    mach: float,
    nose_radius: float,
    cone_half_angle: float | None = None,
    gamma: float = 1.4,
) -> BilligShock:
    r"""Place the bow shock from the correlations alone.

    ``cone_half_angle`` in **degrees**, or ``None`` for a body with no conical
    afterbody, in which case the asymptote is the freestream Mach angle. When a
    cone angle is given the asymptote comes from a Taylor-Maccoll solve
    (:func:`aether.aerodynamics.solve_cone`), which is also this project's one
    exact external anchor -- so the same solution validates the flank and aims
    the hyperbola.

    Validity, in Billig's words: the correlations rest on tests *"made at
    relatively low temperature"* and so agree with perfect-gas :math:`\gamma =
    1.4`; real-gas effects touch the shape *"only at Mach numbers greater than
    about 8 and then not too significantly"*. Above roughly Mach 10 treat the
    result as an **envelope to mesh inside**, not a prediction.

    Eq. (2) is **not Billig's** -- he credits Ambrosio & Wortman (1962) in the
    text, and it is routinely miscited. Eq. (4) is his.
    """
    if mach <= 1.0:
        msg = f"a detached bow shock needs supersonic flow, got M = {mach:g}"
        raise ValueError(msg)
    # Ambrosio & Wortman Eq. (2), cone-spheres.
    standoff = 0.143 * math.exp(3.24 / mach**2) * nose_radius
    # Billig Eq. (4), sphere-cone.
    vertex_radius = 1.143 * math.exp(0.54 / (mach - 1.0) ** 1.2) * nose_radius

    if cone_half_angle is None:
        asymptote = math.asin(1.0 / mach)
    else:
        from aether.aerodynamics import solve_cone

        solution = solve_cone(mach, math.radians(cone_half_angle), gamma=gamma)
        if not solution.converged:
            msg = (
                f"Taylor-Maccoll did not converge for M = {mach:g} at a "
                f"{cone_half_angle:g} degree half-angle, so the asymptote is unknown"
            )
            raise ValueError(msg)
        asymptote = solution.shock_angle
    return BilligShock(
        nose_radius=nose_radius,
        standoff=standoff,
        vertex_radius=vertex_radius,
        asymptote=asymptote,
        mach=mach,
    )


# ---------------------------------------------------------------- the interface


@dataclass(frozen=True)
class Validity:
    """The domain a shock provider is entitled to speak about.

    Carried rather than assumed, because a correlation applied outside its
    range is the failure mode this project has a rule about: an estimate
    wearing a bound's clothing. ``why`` is written onto the run record so a
    mesh can be traced to the reasoning that sized it.
    """

    provider: str
    why: str
    mach: tuple[float, float]
    trustworthy: bool = True
    """False marks an *envelope* -- safe to mesh inside, not a prediction."""

    caveats: tuple[str, ...] = ()

    def check(self, mach: float) -> None:
        low, high = self.mach
        if not low <= mach <= high:
            msg = (
                f"{self.provider} is valid for Mach {low:g}-{high:g} and was asked "
                f"for {mach:g}. Refusing rather than extrapolating: a shock placed "
                "outside a correlation's range puts the grid clustering somewhere "
                "the shock is not, and nothing downstream would notice."
            )
            raise ValueError(msg)


class ShockEnvelope(Protocol):
    """Where the shock is, in a form the mesher can march to.

    The common currency is a scalar field, positive on the **freestream** side
    of the shock and negative between shock and body, so the extruder finds the
    shock by marching outward along a surface normal until the sign flips. Each
    provider writes it in whatever coordinates suit its own geometry.
    """

    validity: Validity

    def outside(self, x: np.ndarray, r: np.ndarray) -> np.ndarray:
        """Positive outside the shock, negative inside, zero on it."""
        ...


@dataclass(frozen=True)
class Azimuthal:
    r"""A Billig hyperboloid whose asymptote varies around the flow axis.

    A single asymptote is a body of revolution, and at incidence the shock is
    not one: it is steeper on the windward meridian and shallower on the
    leeward. Enclosing both with one angle means taking the windward value
    everywhere, which does enclose the shock -- and puts the leeward boundary
    hundreds of millimetres outside a shock that is tens away. On the
    sphere-cone at :math:`\alpha = 6^\circ` that left 6.5 % of columns unable
    to reach their own envelope at all, every one of them leeward and aft.

    So the asymptote is a function of azimuth. The local cone angle seen by the
    flow on a meridian at :math:`\psi` from windward is

    .. math:: \delta_{\text{eff}}(\psi) = \delta + \alpha\cos\psi

    which is the standard equivalent-cone approximation, and each meridian gets
    the Taylor-Maccoll shock angle for *its own* cone. Windward and leeward come
    out right, and everything between is interpolated from a table computed once.

    Still an **envelope**, not a prediction: the equivalent-cone rule is a
    first-order statement about a genuinely three-dimensional shock, and the
    validity says so.
    """

    nose_radius: float
    standoff: float
    vertex_radius: float
    mach: float
    azimuths: _Array
    """Sample azimuths from the windward meridian, radians, ascending over a turn."""

    asymptotes: _Array
    """Shock angle at each sampled azimuth, radians."""

    validity: Validity

    def asymptote_at(self, psi: _Array) -> _Array:
        """Shock angle on the meridian at azimuth ``psi``, by interpolation."""
        return np.interp(
            np.mod(np.asarray(psi, dtype=np.float64), 2.0 * math.pi),
            self.azimuths,
            self.asymptotes,
            period=2.0 * math.pi,
        )

    def outside(self, x: _Array, r: _Array, psi: _Array | None = None) -> _Array:
        """Positive upstream of the hyperboloid. ``psi`` selects the meridian."""
        x = np.asarray(x, dtype=np.float64)
        r = np.asarray(r, dtype=np.float64)
        theta = (
            np.full_like(r, float(np.mean(self.asymptotes)))
            if psi is None
            else self.asymptote_at(psi)
        )
        t = np.tan(theta)
        axial = -self.standoff + self.vertex_radius / t**2 * (
            np.sqrt(1.0 + r**2 * t**2 / self.vertex_radius**2) - 1.0
        )
        return axial - x

    def axial_at(self, y: _Array) -> _Array:
        """The windward meridian, which is the one worth drawing."""
        y = np.asarray(y, dtype=np.float64)
        t = math.tan(float(self.asymptotes[0]))
        return -self.standoff + self.vertex_radius / t**2 * (
            np.sqrt(1.0 + y**2 * t**2 / self.vertex_radius**2) - 1.0
        )


def azimuthal_billig(
    mach: float,
    nose_radius: float,
    cone_half_angle: float,
    incidence: float,
    gamma: float = 1.4,
    samples: int = 37,
) -> Azimuthal:
    """Billig with a per-meridian asymptote from Taylor-Maccoll.

    ``cone_half_angle`` in degrees, ``incidence`` in radians. Falls back to the
    Mach angle on any meridian whose equivalent cone is past detachment, which
    is what a cone that steep actually does.
    """
    from aether.aerodynamics import maximum_cone_angle, solve_cone

    psis = np.linspace(0.0, 2.0 * math.pi, samples)
    detach, _ = maximum_cone_angle(mach, gamma)
    mach_angle = math.asin(1.0 / mach)
    angles = []
    for psi in psis:
        local = math.radians(cone_half_angle) + incidence * math.cos(psi)
        if local <= 1.0e-4 or local >= detach:
            angles.append(max(mach_angle, local + mach_angle))
            continue
        solution = solve_cone(mach, local, gamma=gamma)
        angles.append(
            float(solution.shock_angle) if solution.converged else local + mach_angle
        )
    return Azimuthal(
        nose_radius=float(nose_radius),
        standoff=0.143 * math.exp(3.24 / mach**2) * float(nose_radius),
        vertex_radius=1.143 * math.exp(0.54 / (mach - 1.0) ** 1.2) * float(nose_radius),
        mach=float(mach),
        azimuths=psis,
        asymptotes=np.asarray(angles, dtype=np.float64),
        validity=None,  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class Oriented:
    r"""A shock envelope expressed about the **freestream**, not the body axis.

    Every provider here is a body of revolution, and each writes its field in
    its own natural frame: :math:`x` downstream from the stagnation point along
    the oncoming flow, :math:`r` from that axis. At zero incidence those axes
    coincide with the body's and the distinction does not arise.

    At incidence it is the whole problem. The bow shock stands about the
    **velocity vector**, not the body centreline, so an envelope built on the
    body axis is wrong by the incidence angle -- on a 2 m sphere-cone at
    :math:`\alpha = 6^\circ` that is :math:`2\tan 6^\circ = 0.21` m of
    displacement against a 9 mm standoff. It was wrong by exactly that from the
    day the shock-fitted mesher was written, and it went unseen because every
    run that reached the mesher was axial: the non-axial ones had already been
    failing for a year → [[every-non-axial-case-dies-in-the-prelude]].

    It stopped being survivable when the outer boundary began to **impose**
    freestream → [[a-fitted-boundary-is-not-a-far-field]]. A misplaced envelope
    then asserts undisturbed flow inside the shock layer on the windward side,
    which is not a small error in a boundary condition but a false statement.

    The stagnation point moves too. On a spherical nose of radius :math:`R_n`
    centred at :math:`C` it is exactly :math:`C - R_n\hat v`, which is where the
    surface normal is anti-parallel to the flow.
    """

    inner: ShockEnvelope
    origin: _Array
    """Stagnation point: the envelope's own nose, in body coordinates."""

    axis: _Array
    """Unit freestream direction. The envelope is a surface of revolution about
    the line through :attr:`origin` along this."""

    validity: Validity

    def at(self, points: _Array) -> _Array:
        """Positive outside the shock, negative inside. ``points`` is ``(N, 3)``."""
        offset = np.asarray(points, dtype=np.float64).reshape(-1, 3) - self.origin
        along = offset @ self.axis
        lateral = offset - along[:, None] * self.axis[None, :]
        radial = np.linalg.norm(lateral, axis=1)
        if getattr(self.inner, "azimuths", None) is None:
            return np.asarray(self.inner.outside(along, radial), dtype=np.float64)
        # Azimuth from the windward meridian, about the flow axis. Windward is
        # the side the flow is turned *into*, which for a +alpha rotation of the
        # freestream is -z.
        windward = self.windward
        across = np.cross(self.axis, windward)
        psi = np.arctan2(lateral @ across, lateral @ windward)
        return np.asarray(self.inner.outside(along, radial, psi), dtype=np.float64)

    def __getattr__(self, name: str) -> Any:
        """Anything not the frame belongs to the provider being oriented.

        A wrapper that hides its subject is a wrapper that breaks every caller
        reading `standoff`, `asymptote` or `vertex_radius` off an envelope --
        and those are read by the mesher, the preview and the report. Delegating
        is what makes this an added frame rather than a replacement type.
        """
        try:
            return getattr(object.__getattribute__(self, "inner"), name)
        except AttributeError:
            raise AttributeError(
                f"neither the oriented frame nor {type(self.inner).__name__} has {name!r}"
            ) from None

    @property
    def windward(self) -> _Array:
        """Unit vector on the windward meridian, perpendicular to the flow axis."""
        reference = np.array([0.0, 0.0, -1.0])
        lateral = reference - (reference @ self.axis) * self.axis
        length = float(np.linalg.norm(lateral))
        if length < 1.0e-9:
            lateral = np.array([0.0, 1.0, 0.0])
            length = 1.0
        return lateral / length

    def outside(self, x: _Array, r: _Array) -> _Array:
        """Body-axis form, kept so an axial caller reads unchanged.

        Only correct when the envelope's axis *is* the body axis, so it refuses
        rather than quietly answering the wrong question when it is not.
        """
        if abs(float(self.axis[0]) - 1.0) > 1.0e-12:
            msg = (
                "this envelope is built about the freestream, which is at "
                f"{math.degrees(math.acos(min(1.0, abs(float(self.axis[0]))))):.2f} "
                "degrees to the body axis, so an axisymmetric (x, r) query is "
                "not well posed. Use `at(points)`."
            )
            raise ValueError(msg)
        x = np.asarray(x, dtype=np.float64)
        r = np.asarray(r, dtype=np.float64)
        return np.asarray(self.inner.outside(x - self.origin[0], r), dtype=np.float64)

    def trace(self, radii: _Array) -> _Array:
        """The shock in the symmetry plane, as ``(N, 3)`` points, for drawing."""
        radii = np.asarray(radii, dtype=np.float64)
        axial = np.asarray(self.inner.axial_at(radii), dtype=np.float64)
        # In-plane normal to the axis, in the x-z symmetry plane.
        normal = np.array([-self.axis[2], 0.0, self.axis[0]])
        length = float(np.linalg.norm(normal))
        normal = normal / length if length > 1.0e-12 else np.array([0.0, 0.0, 1.0])
        return (
            self.origin[None, :]
            + axial[:, None] * self.axis[None, :]
            + radii[:, None] * normal[None, :]
        )


def freestream_axis(alpha: float, beta: float) -> _Array:
    """Unit freestream direction for an incidence pair, in body coordinates.

    The same convention :class:`~aether.cfd.config.FlowConditions` uses
    and SU2 is given: ``alpha`` about the y axis, ``beta`` about z.
    """
    return np.array(
        [
            math.cos(alpha) * math.cos(beta),
            -math.sin(beta),
            math.sin(alpha) * math.cos(beta),
        ],
        dtype=np.float64,
    )


def _march_to_shock(
    envelope: ShockEnvelope,
    origin: np.ndarray,
    direction: np.ndarray,
    limit: float,
    samples: int = 256,
) -> float:
    """Distance from ``origin`` along ``direction`` to the shock, by bisection.

    Returns ``limit`` when the ray never crosses -- a surface point whose
    outward normal runs down the body rather than out through the shock, which
    happens on the aft flank and is not an error.
    """
    ts = np.linspace(0.0, limit, samples)
    points = origin[None, :] + ts[:, None] * direction[None, :]
    values = _field(envelope, points)
    crossings = np.nonzero(np.diff(np.signbit(values)))[0]
    if not crossings.size:
        return limit
    lo, hi = ts[crossings[0]], ts[crossings[0] + 1]
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        value = float(_field(envelope, (origin + mid * direction)[None, :])[0])
        if np.signbit(value) == np.signbit(values[crossings[0]]):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _field(envelope: Any, points: _Array) -> _Array:
    """Evaluate an envelope at ``(N, 3)`` points, oriented or not.

    One place, because the alternative is every caller remembering which kind it
    holds -- and a caller that forgets gets an answer rather than an error.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    reader = getattr(envelope, "at", None)
    if reader is not None:
        return np.asarray(reader(points), dtype=np.float64)
    return np.asarray(
        envelope.outside(points[:, 0], np.hypot(points[:, 1], points[:, 2])),
        dtype=np.float64,
    )


# ---------------------------------------------------------------- the providers


@dataclass(frozen=True)
class ConicalShock:
    """An attached conical shock from a sharp cone -- Taylor-Maccoll.

    Exact, not correlated, which makes it the strongest provider here. It
    applies only below the detachment angle; above it the shock stands off and
    a blunt-body provider is required instead.
    """

    apex: float
    shock_angle: float
    validity: Validity

    def outside(self, x: np.ndarray, r: np.ndarray) -> np.ndarray:
        return np.asarray(r) - np.maximum(np.asarray(x) - self.apex, 0.0) * math.tan(
            self.shock_angle
        )


@dataclass(frozen=True)
class WedgeShock:
    """An attached oblique shock from a wedge-derived lifting body.

    A caret waverider is **designed** from this shock: its leading edges ride
    it, which is what makes the compression close underneath. Treating such a
    body with a detached-shock correlation would place the clustering where the
    shock is not.
    """

    apex: float
    shock_angle: float
    validity: Validity

    def outside(self, x: np.ndarray, r: np.ndarray) -> np.ndarray:
        return np.asarray(r) - np.maximum(np.asarray(x) - self.apex, 0.0) * math.tan(
            self.shock_angle
        )


@dataclass(frozen=True)
class MachCone:
    """The freestream Mach cone, padded. **An envelope, never a prediction.**

    Every real shock on a lifting or blunted body stands outside the Mach cone,
    so meshing to a padded Mach cone is conservative: the shock is guaranteed
    to be inside the domain. It is the fallback for bodies no better provider
    covers, and it is marked ``trustworthy=False`` so nothing quotes a standoff
    from it.
    """

    apex: float
    mach: float
    padding: float = 1.5
    validity: Validity = None  # type: ignore[assignment]

    def outside(self, x: np.ndarray, r: np.ndarray) -> np.ndarray:
        angle = math.asin(1.0 / self.mach) * self.padding
        return np.asarray(r) - np.maximum(np.asarray(x) - self.apex, 0.0) * math.tan(
            min(angle, 0.5 * math.pi - 1e-3)
        )


def _billig_outside(self: BilligShock, x: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Positive upstream of the hyperbola, negative between it and the body."""
    return np.asarray(self.axial_at(r)) - np.asarray(x)


BilligShock.outside = _billig_outside  # type: ignore[attr-defined]


#: Fractional slack added to a measured bounding angle.
#:
#: The bound is taken from a *tessellated* surface, so it is the angle the
#: triangles subtend and not quite the angle the exact shape does; refining the
#: surface moves it by a fraction of a percent. Two percent covers that without
#: pushing the boundary appreciably off the shock -- and an envelope a little
#: too wide costs cells, while one a little too narrow puts the body outside
#: its own domain.
ENVELOPE_MARGIN = 0.02

_BODY_BOUNDS: dict[tuple[str, tuple[tuple[str, float], ...]], tuple[Any, float]] = {}


def body_bound(body: str, geometry: dict[str, float], axis: Any) -> tuple[Any, float]:
    r"""The body's apex on the freestream line, and the half-angle it subtends there.

    An envelope has one job: contain the body. Two of the providers here could
    not, and for the same reason in different disguises.

    * The **waveriders are built off-axis.** A waverider rides under its own
      shock, so its design frame puts the body below the centreline: the caret
      occupies :math:`z \in [-0.829, -0.002]` and the osculating-cone body
      starts at :math:`x = 0.987` with :math:`z \in [0.063, 0.814]`. The
      envelope was anchored at the coordinate **origin**, which is not on either
      body, so 15 840 of 26 592 caret vertices lay outside their own shock.
    * The **slender bodies bulge.** ``atan2(radius, length)`` is the angle from
      the apex to the *base rim*, and a Sears-Haack spindle carries its maximum
      radius at mid-length -- 12.6 degrees where 6.4 was being used. A cone of
      the same fineness ratio does not bound a contour that is convex upward,
      whatever the comment claimed.

    Measured rather than derived, because the shapes are generated and their
    apparent half-angle is a property of the generated surface and not of the
    parameter dictionary. Cached on the parameter set, since the preflight, the
    mesher and the force anchor all ask for the same envelope.

    Returns ``(apex, half_angle)`` -- the most upstream point along ``axis``,
    and the largest angle any vertex subtends about the line through it.
    """
    import numpy as _np

    key = (body, tuple(sorted((k, float(v)) for k, v in geometry.items())))
    hit = _BODY_BOUNDS.get(key)
    if hit is not None:
        return hit

    from aether.cfd.geometries import build_surface

    vertices = _np.asarray(build_surface(body, **geometry).vertices, dtype=float)
    unit = _np.asarray(axis, dtype=float)
    unit = unit / _np.linalg.norm(unit)
    along_all = vertices @ unit
    apex = vertices[int(_np.argmin(along_all))]
    offset = vertices - apex
    along = offset @ unit
    lateral = _np.linalg.norm(offset - _np.outer(along, unit), axis=1)
    # Vertices level with the apex subtend 90 degrees on any blunt nose, which
    # is true and useless: a cone from the apex cannot bound a blunt nose, and
    # the bodies that reach this helper are the sharp-nosed ones. Measure where
    # the cone is meaningful -- a percent of the body length downstream.
    span = float(along.max())
    keep = along > 0.01 * span
    angle = float(_np.arctan2(lateral[keep], along[keep]).max()) if keep.any() else 0.0
    _BODY_BOUNDS[key] = (apex, angle)
    return apex, angle


def effective_tip_radius(body: str, geometry: dict[str, float], axis: Any) -> float:
    r"""Radius of curvature at the nose, measured from the surface.

    ``nosecap.nose_sphere_radius`` returns ``None`` for these shapes and is
    right to: an ogive or a power-law body is not spherical at the tip, so the
    rings disagree and a single sphere is not what the nose is. But it is not
    *sharp* either, and that is what matters for the envelope -- a cone anchored
    on a blunt nose has zero radius where the body has millimetres, so it cannot
    enclose it, and a cone shifted far enough upstream to enclose it puts its
    apex metres ahead of the vehicle.

    The number wanted is the osculating radius, which the same algebra gives
    without requiring the rings to agree:

    .. math:: R = \frac{r^2 + u^2}{2u}

    taken as the median over the forward few percent, which is robust to the
    tessellation of a rounded tip.
    """
    import numpy as _np

    from aether.cfd.geometries import build_surface

    unit = _np.asarray(axis, dtype=float)
    unit = unit / _np.linalg.norm(unit)
    vertices = _np.asarray(build_surface(body, **geometry).vertices, dtype=float)
    offset = vertices - vertices[int(_np.argmin(vertices @ unit))]
    along = offset @ unit
    lateral = _np.linalg.norm(offset - _np.outer(along, unit), axis=1)
    span = float(along.max())
    near = (along > 1.0e-6 * span) & (along < 0.02 * span)
    if not near.any():
        return 0.0
    return float(_np.median((lateral[near] ** 2 + along[near] ** 2) / (2.0 * along[near])))


def cone_standoff(body: str, geometry: dict[str, float], axis: Any, angle: float) -> float:
    r"""How far upstream a cone of half-angle ``angle`` must start to contain the body.

    A cone anchored exactly on the apex has zero radius there, so any body whose
    nose is **blunt** -- even slightly -- pokes out of it. The caret waverider's
    3 mm leading-edge radius puts 95 vertices outside over the first 0.9 % of
    its length, and the osculating-cone body's apex station has finite lateral
    extent, so three vertices sit outside a cone that is a point where they are.

    Shifting the apex upstream is what a detached shock does, and the shift that
    suffices is exact rather than tuned: the cone contains a vertex when
    :math:`\ell \le (a + \Delta)\tan\theta`, so

    .. math:: \Delta = \max\left(0,\; \max_i\left[
              \frac{\ell_i}{\tan\theta} - a_i \right]\right)

    with :math:`a` measured along the freestream from the apex and
    :math:`\ell` perpendicular to it.
    """
    import numpy as _np

    from aether.cfd.geometries import build_surface

    unit = _np.asarray(axis, dtype=float)
    unit = unit / _np.linalg.norm(unit)
    vertices = _np.asarray(build_surface(body, **geometry).vertices, dtype=float)
    apex = vertices[int(_np.argmin(vertices @ unit))]
    offset = vertices - apex
    along = offset @ unit
    lateral = _np.linalg.norm(offset - _np.outer(along, unit), axis=1)
    tangent = math.tan(float(angle))
    if tangent <= 0.0:
        return 0.0
    return float(max(0.0, (lateral / tangent - along).max()))


def enclosing_nose_radius(
    body: str,
    geometry: dict[str, float],
    axis: Any,
    mach: float,
    half_angle: float,
    gamma: float = 1.4,
    ceiling: float = 64.0,
) -> tuple[float, float]:
    r"""Effective nose radius whose Billig hyperbola contains the body.

    Billig's standoff and vertex radius are **sphere** forms: both scale with
    the nose radius, so a hyperbola drawn for a 0.1 m sphere is the shock of a
    0.1 m sphere and nothing else. A wide blunt body is far blunter than its
    nose radius suggests in the span plane -- the lifting body fattens to a 1 m
    half-span over 2.9 m, and its forward quarter sat up to 70 mm outside a
    hyperbola built on 0.1 m, across 160 vertices.

    The effective radius is found rather than assumed: double it until the body
    fits, then bisect. A blunter body stands its shock further off, so this is
    the physical knob and not a fudge -- but the number that comes out is an
    *envelope* radius and not the body's geometric nose radius, and the validity
    says so.

    Returns ``(radius, inflation)`` -- the radius used and its ratio to the
    geometric one.
    """
    import numpy as _np

    from aether.cfd.geometries import build_surface

    unit = _np.asarray(axis, dtype=float)
    unit = unit / _np.linalg.norm(unit)
    vertices = _np.asarray(build_surface(body, **geometry).vertices, dtype=float)
    geometric = float(geometry.get("nose_radius", 0.1))

    def encloses(radius: float) -> bool:
        shock = billig_shock(mach, radius, cone_half_angle=half_angle, gamma=gamma)
        centre = _np.array([radius, 0.0, 0.0])
        # A bare provider carries no Validity; this probe only needs geometry.
        probe = Oriented(
            inner=shock,
            origin=centre - radius * unit,
            axis=unit,
            validity=Validity(provider="probe", why="enclosure search", mach=(1.0, 60.0)),
        )
        return bool(_np.all(_np.asarray(probe.at(vertices)) <= 0.0))

    if encloses(geometric):
        return geometric, 1.0
    low, high = geometric, geometric * 2.0
    while high <= geometric * ceiling and not encloses(high):
        low, high = high, high * 2.0
    if not encloses(high):
        msg = (
            f"no Billig hyperbola up to {ceiling:g}x the {geometric:g} m nose "
            f"radius contains {body}; it is too far from a blunt body of "
            "revolution for this envelope family"
        )
        raise ValueError(msg)
    for _ in range(40):
        mid = 0.5 * (low + high)
        if encloses(mid):
            high = mid
        else:
            low = mid
    return float(high), float(high / geometric)


def envelope_for(
    body: str,
    geometry: dict[str, float],
    mach: float,
    gamma: float = 1.4,
    alpha: float = 0.0,
    beta: float = 0.0,
) -> ShockEnvelope:
    r"""Choose a shock provider for a body, or refuse.

    The refusal matters more than the choice. Silently falling back to a Mach
    cone for a body nobody has thought about would put cells in the wrong
    place and produce a mesh that looks fine.

    ``alpha`` and ``beta`` in **radians**. They do two things, and neither is
    optional once the outer boundary imposes freestream rather than
    extrapolating it:

    * the envelope is built about the **freestream vector through the
      stagnation point**, not the body axis -- see :class:`Oriented`;
    * where the provider has an asymptote taken from an afterbody, that
      afterbody's effective angle is the **windward** one, :math:`\delta + |\alpha|`.
      The shock is steeper on the windward side and shallower on the leeward, so
      the windward value is the one that encloses both. That makes the result an
      *envelope* rather than a prediction, and the validity says so.
    """
    from aether.aerodynamics import maximum_cone_angle, solve_cone, wedge_shock_angle

    incidence = math.hypot(alpha, beta)
    axis = freestream_axis(alpha, beta)

    def oriented(
        provider: Any, nose_radius: float | None, origin: Any = None
    ) -> ShockEnvelope:
        """Re-express a provider about the freestream through the stagnation point.

        For a spherical nose of radius R centred at (R, 0, 0) the stagnation
        point is exactly ``centre - R * axis``: the place where the surface
        normal is anti-parallel to the flow. Without a spherical nose -- a
        waverider's leading edge -- the apex is the origin and does not move.
        """
        if origin is None:
            if nose_radius:
                centre = np.array([float(nose_radius), 0.0, 0.0])
                origin = centre - float(nose_radius) * axis
            else:
                origin = np.zeros(3)
        origin = np.asarray(origin, dtype=np.float64)
        validity = provider.validity
        if incidence > 1.0e-9:
            validity = replace(
                validity,
                trustworthy=False,
                caveats=(
                    *validity.caveats,
                    f"at {math.degrees(incidence):.2f} degrees of incidence the "
                    "shock is not a body of revolution; this is the windward "
                    "envelope, which encloses the leeward side with room to "
                    "spare and is not a standoff to quote",
                ),
            )
        return Oriented(inner=provider, origin=origin, axis=axis, validity=validity)

    if body in ("sphere-cone", "biconic"):
        # Blunt nose: the shock is detached there whatever the afterbody does.
        # The afterbody sets the asymptote, and for a biconic that is the
        # *first* cone -- the second is further downstream than the hyperbola
        # has settled, and taking the shallower one would cut inside the shock.
        half_angle = geometry.get("half_angle", geometry.get("half_angle_1", 10.0))
        nose = geometry.get("nose_radius", 0.05)
        # At incidence the shock is steeper windward and shallower leeward, so
        # one asymptote cannot describe it. Taking the windward value everywhere
        # does enclose the shock and puts the leeward boundary hundreds of
        # millimetres outside one that is tens away -- measured, 6.5 % of
        # columns could not reach their own envelope, all of them leeward.
        shock: Any
        if incidence > 1.0e-6:
            shock = azimuthal_billig(
                mach, nose, cone_half_angle=half_angle, incidence=incidence, gamma=gamma
            )
        else:
            shock = billig_shock(mach, nose, cone_half_angle=half_angle, gamma=gamma)
        caveats: tuple[str, ...] = ()
        if body == "biconic":
            caveats += (
                "asymptote taken from the forward cone; the aft cone is shallower "
                "and the hyperbola is conservative against it",
            )
        if mach > 10.0:
            caveats += (
                "above Mach 10 Billig's perfect-gas correlations are an envelope "
                "to mesh inside, not a standoff to quote",
            )
        object.__setattr__(
            shock,
            "validity",
            Validity(
                provider="BilligShock",
                why=f"{body} has a {nose:g} m blunt nose, so the bow shock is detached",
                mach=(1.5, 30.0),
                trustworthy=mach <= 10.0,
                caveats=caveats,
            ),
        )
        return oriented(shock, nose)

    if body == "lifting-body":
        # Blunt-nosed and **not** a body of revolution: half-span 1.0 m against
        # half-thickness 0.3 m, so the shock stands much further out in the span
        # plane than in the thickness plane and no single hyperbola is *the*
        # shock. An envelope does not have to be the shock; it has to enclose
        # it. So the asymptote is taken from the **widest** semi-apex angle --
        # the span plane -- which encloses the thickness plane with room to
        # spare. The cost is cells outboard of the shock above and below; the
        # alternative is a boundary cutting inside it, which imposes freestream
        # on shocked gas and is the failure this envelope exists to avoid.
        geometric_nose = float(geometry.get("nose_radius", 0.1))
        span = float(geometry.get("base_half_span", 1.0))
        length = float(geometry.get("length", 2.9))
        half_angle = math.degrees(math.atan2(span, length))
        # Solved, not assumed: the span-plane hyperbola on the geometric nose
        # radius left the forward quarter of this body outside its own shock.
        nose, inflation = enclosing_nose_radius(
            body, geometry, axis, mach, half_angle, gamma=gamma
        )
        if incidence > 1.0e-6:
            shock = azimuthal_billig(
                mach, nose, cone_half_angle=half_angle, incidence=incidence, gamma=gamma
            )
        else:
            shock = billig_shock(mach, nose, cone_half_angle=half_angle, gamma=gamma)
        object.__setattr__(
            shock,
            "validity",
            Validity(
                provider="BilligShock",
                why=(
                    f"lifting-body has a {geometric_nose:g} m blunt nose, so the bow "
                    f"shock is detached; asymptote from the span plane at "
                    f"{half_angle:.1f} deg, hyperbola opened to an effective "
                    f"{nose:.3f} m ({inflation:.1f}x) to enclose the body"
                ),
                mach=(1.5, 30.0),
                # Never a standoff to quote on this body: the correlation is a
                # body-of-revolution form and this shape is not one.
                trustworthy=False,
                caveats=(
                    "not a body of revolution: the envelope is the span-plane "
                    "hyperbola applied in every azimuth, so it stands well "
                    "outside the shock in the thickness plane and the standoff "
                    "it reports there is the envelope's, not the flow's",
                    f"effective nose radius {nose:.3f} m is {inflation:.1f}x the "
                    "geometric one, solved for enclosure; it is a meshing radius "
                    "and not this body's bluntness",
                )
                + (
                    (
                        "above Mach 10 Billig's perfect-gas correlations are an "
                        "envelope to mesh inside, not a standoff to quote",
                    )
                    if mach > 10.0
                    else ()
                ),
            ),
        )
        return oriented(shock, nose)

    if body == "hemisphere":
        # The verification body, and the only one the correlations were fitted
        # on directly: Ambrosio & Wortman Eq. (2) and Billig Eq. (4) are both
        # *sphere* forms, so here they are used as intended rather than carried
        # onto a cone-sphere. No afterbody, so no Taylor-Maccoll asymptote --
        # past the equator there is nothing to turn the flow and the shock
        # relaxes onto the Mach wave, which is what ``cone_half_angle=None``
        # asks for.
        radius = geometry.get("radius", 0.045)
        shock = billig_shock(mach, radius, cone_half_angle=None, gamma=gamma)
        object.__setattr__(
            shock,
            "validity",
            Validity(
                provider="BilligShock",
                why=(
                    f"sphere of radius {radius:g} m; the standoff and vertex-radius "
                    "correlations are the sphere forms, applied to a sphere"
                ),
                mach=(1.5, 30.0),
                trustworthy=mach <= 10.0,
                caveats=(
                    (
                        "above Mach 10 Billig's perfect-gas correlations are an "
                        "envelope to mesh inside, not a standoff to quote",
                    )
                    if mach > 10.0
                    else ()
                )
                + (
                    "the base is flat and separated; the hyperbola says nothing "
                    "about the wake and the domain is sized past it instead",
                ),
            ),
        )
        return oriented(shock, radius)

    if body in ("caret-waverider", "osculating-cone"):
        wedge = geometry.get("wedge_angle")
        if wedge is not None:
            angle = wedge_shock_angle(mach, math.radians(wedge), gamma=gamma)
            why = f"caret waverider designed on a {wedge:g} degree wedge; shock is attached"
        else:
            angle = math.radians(geometry.get("shock_angle", 14.0))
            why = "osculating-cone waverider carries its design shock angle explicitly"
        # Anchored on the body's own apex, and never narrower than the body.
        # The design shock angle describes the *compression surface*; the
        # planform is wider than that, and an envelope that ignores the span
        # leaves the leading edges outside their own shock.
        apex, bound = body_bound(body, geometry, axis)
        widened = bound + incidence > angle
        angle = max(float(angle), float(bound + incidence)) * (1.0 + ENVELOPE_MARGIN)
        return oriented(WedgeShock(
            apex=0.0,
            shock_angle=float(angle),
            validity=Validity(
                provider="WedgeShock",
                why=why,
                mach=(1.5, 30.0),
                trustworthy=not widened,
                caveats=(
                    "attached only on-design; a blunted leading edge detaches the "
                    "shock locally and at incidence the windward angle grows",
                )
                + (
                    (
                        f"opened from the design angle to {math.degrees(angle):.1f} "
                        "degrees so the planform fits inside it; this is an "
                        "envelope to mesh in, not the shock angle",
                    )
                    if widened
                    else ()
                ),
            ),
        ), None, origin=apex - cone_standoff(body, geometry, axis, angle) * axis)

    if body in ("sears-haack", "von-karman-ogive", "power-law-newtonian", "power-law-blast"):
        # Slender axisymmetric bodies with **slightly blunt** noses, which is
        # what a tessellated ogive or power-law tip is: 12 to 18 mm of osculating
        # radius, not a point. That distinction decides the envelope family.
        #
        # A pure cone was used here and cannot work. Anchored on the nose it has
        # zero radius where the body has millimetres, so it does not enclose;
        # shifted upstream far enough to enclose, its apex lands metres ahead of
        # the vehicle and ``_march_to_shock`` -- looking upstream from the nose
        # for a surface that is now behind it -- ran to its 10 m limit and
        # returned that as the standoff. The column cap then sized itself at
        # 17.5 m, and the extrusion produced inverted cells.
        #
        # Billig is the family for a blunt nose: a standoff at the tip relaxing
        # onto a conical asymptote. Both numbers are measured -- the tip radius
        # from the surface, the asymptote from the widest station the contour
        # reaches -- and the radius is then opened until the body is inside.
        tip = effective_tip_radius(body, geometry, axis)
        apex, bound = body_bound(body, geometry, axis)
        asymptote = math.degrees(bound + incidence)
        detach, _ = maximum_cone_angle(mach, gamma)
        if tip > 0.0:
            radius, inflation = enclosing_nose_radius(
                body, geometry, axis, mach, asymptote, gamma=gamma
            )
            shock = billig_shock(mach, radius, cone_half_angle=asymptote, gamma=gamma)
            object.__setattr__(
                shock,
                "validity",
                Validity(
                    provider="BilligShock",
                    why=(
                        f"{body} has a {1000 * tip:.1f} mm nose radius measured from "
                        f"its own surface, so the shock is detached; asymptote "
                        f"{asymptote:.1f} deg from the widest station"
                    ),
                    mach=(1.5, 30.0),
                    trustworthy=mach <= 10.0,
                    caveats=(
                        "the asymptote bounds the contour rather than following "
                        "it, so the envelope stands outside the shock over the "
                        "afterbody",
                    )
                    + (
                        (
                            f"hyperbola opened to {1000 * radius:.1f} mm "
                            f"({inflation:.1f}x the measured tip) to enclose the body",
                        )
                        if inflation > 1.001
                        else ()
                    )
                    + (
                        (
                            "above Mach 10 Billig's perfect-gas correlations are an "
                            "envelope to mesh inside, not a standoff to quote",
                        )
                        if mach > 10.0
                        else ()
                    ),
                ),
            )
            return oriented(shock, radius)

        if bound + incidence < detach:
            solution = solve_cone(mach, bound + incidence, gamma=gamma)
            if solution.converged:
                return oriented(
                    ConicalShock(
                        apex=0.0,
                        shock_angle=float(solution.shock_angle),
                        validity=Validity(
                            provider="ConicalShock",
                            why=(
                                f"{body} is sharp and slender; the "
                                f"{math.degrees(bound + incidence):.1f} degree cone "
                                "that bounds its widest station encloses it"
                            ),
                            mach=(1.05, 30.0),
                            caveats=(
                                "bounds the body rather than following its contour",
                            ),
                        ),
                    ),
                    None,
                    origin=apex
                    - cone_standoff(body, geometry, axis, float(solution.shock_angle)) * axis,
                )
        return oriented(
            MachCone(
                apex=0.0,
                mach=mach,
                validity=Validity(
                    provider="MachCone",
                    why=f"{body} at Mach {mach:g} exceeds cone detachment; no fitted provider",
                    mach=(1.05, 60.0),
                    trustworthy=False,
                    caveats=("a padded envelope, not a standoff",),
                ),
            ),
            None,
            origin=apex - cone_standoff(body, geometry, axis, math.asin(1.0 / mach)) * axis,
        )

    msg = (
        f"no shock provider covers {body!r}. Add one, or give it an explicit "
        "envelope -- do not let it fall through to a Mach cone, because a mesh "
        "clustered in the wrong place looks exactly like a mesh clustered in "
        "the right one."
    )
    raise ValueError(msg)
