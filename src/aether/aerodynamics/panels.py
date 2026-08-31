"""Panel model, integrated loads, and trim (Paper II, §3.3).

The distributed forcing on the collocation grid is
:math:`\\mathbf{Q}_{\\mathrm{aero}} = q_{\\mathrm{dyn}}\\mathbf{C}_p`
(Eq. 3.6). This module integrates that over a panelized surface to give
normal force and pitching moment, and solves for the trim incidence —
the quantity II-V4 measures against blend width.

The bundled :func:`curved_lifting_body` is a *generic demonstration
geometry*: a cambered surface with elliptical spanwise loading, chosen
because its shoulder line sweeps through :math:`\\delta_c = 0` as
incidence varies, which is exactly the configuration where the blending
seam of :mod:`aether.aerodynamics.closure` can affect integrated loads.
It corresponds to no vehicle and carries no design data.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import scipy.optimize
from numpy.typing import NDArray

from aether.aerodynamics.closure import blended_pressure_coefficient
from aether.geometry.profiles import (
    sphere_cone_closure,
    sphere_cone_meridian,
    sphere_cone_tangency,
)

__all__ = [
    "PanelModel",
    "SurfaceGrid",
    "TrimSolution",
    "blunted_multiconic",
    "caret_lifting_body",
    "curved_lifting_body",
    "exact_mitered_bent_biconic",
    "smooth_bent_biconic",
    "spatular_wedge",
    "sphere_cone",
    "sphere_cone_closure",
]

_FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SurfaceGrid:
    """The structured vertex net a parametric generator built its panels from.

    A :class:`PanelModel` is centroids, normals and areas, which is everything
    impact theory reads and nothing a mesh generator can use. Panels carry no
    connectivity: two neighbouring triangles are two independent rows, and
    there is no way back from that to the closed surface they came from. A CFD
    domain needs the closed surface — gmsh has to know which edges are shared
    before it can put a volume outside them — so the generators that build
    their panels from a structured net keep the net rather than discarding it.

    Attributes
    ----------
    vertices:
        Shape ``(n_axial, n_circumferential + 1, 3)``, ordered nose-first
        along the first axis and once around the body along the second. The
        last circumferential column repeats the first, which is what makes
        the seam closable without a search for coincident points.
    """

    vertices: _FloatArray = field(repr=False)

    def __post_init__(self) -> None:
        v = np.asarray(self.vertices, dtype=np.float64)
        if v.ndim != 3 or v.shape[2] != 3:
            msg = f"vertices must have shape (n_axial, n_circ + 1, 3), got {v.shape}"
            raise ValueError(msg)
        if v.shape[0] < 2 or v.shape[1] < 4:
            msg = f"a surface net needs at least 2 x 4 vertices, got {v.shape[:2]}"
            raise ValueError(msg)

    @property
    def n_axial(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_circumferential(self) -> int:
        """Distinct circumferential stations — one fewer than the array width."""
        return int(self.vertices.shape[1]) - 1

    @property
    def seam_closed(self) -> bool:
        """Does the last circumferential column repeat the first?

        Checked rather than assumed: a generator that sampled ``psi`` on
        ``linspace(0, 2*pi, n + 1)`` closes the seam exactly, and one that used
        ``endpoint=False`` does not. Triangulating the second as though it were
        the first leaves a full-length slit down the body, which is invisible
        in a panel integration — the slit has no area — and fatal to a mesh
        generator, which will happily fill the vehicle's interior with cells.
        """
        gap = self.vertices[:, -1, :] - self.vertices[:, 0, :]
        scale = float(np.max(np.abs(self.vertices))) or 1.0
        return bool(np.max(np.abs(gap)) <= 1.0e-9 * scale)


@dataclass(frozen=True)
class TrimSolution:
    """Result of a trim solve about a moment reference point."""

    incidence: float
    """Trim angle of attack (rad)."""
    normal_force: float
    """Normal force at trim (N)."""
    axial_force: float
    """Axial force at trim (N)."""
    pitching_moment: float
    """Residual pitching moment at trim (N·m), zero to solver tolerance."""
    converged: bool


@dataclass(frozen=True)
class PanelModel:
    """A panelized surface with outward normals and areas.

    Attributes
    ----------
    centroids:
        Panel centroids in body axes, shape ``(n, 3)`` (m).
    normals:
        Outward unit normals, shape ``(n, 3)``.
    areas:
        Panel areas, shape ``(n,)`` (m²).
    reference_point:
        Moment reference point in body axes (m).
    """

    centroids: _FloatArray = field(repr=False)
    normals: _FloatArray = field(repr=False)
    areas: _FloatArray = field(repr=False)
    reference_point: _FloatArray = field(repr=False, default_factory=lambda: np.zeros(3))
    surface: SurfaceGrid | None = field(repr=False, default=None)
    """The net the panels were cut from, when the generator had one.

    Optional because it is not always available — :func:`caret_lifting_body` and
    :func:`sphere_cone` assemble triangles directly and :func:`curved_lifting_body`
    is an open two-sheet surface with no inside — and because nothing in the
    panel integration needs it. It exists for the mesh generator, which does.
    """

    def __post_init__(self) -> None:
        c = np.asarray(self.centroids, dtype=np.float64)
        n = np.asarray(self.normals, dtype=np.float64)
        a = np.asarray(self.areas, dtype=np.float64)
        if c.ndim != 2 or c.shape[1] != 3:
            raise ValueError(f"centroids must have shape (n, 3), got {c.shape}")
        if n.shape != c.shape:
            raise ValueError(f"normals shape {n.shape} does not match centroids {c.shape}")
        if a.shape != (c.shape[0],):
            raise ValueError(f"areas must have shape ({c.shape[0]},), got {a.shape}")
        if np.any(a <= 0.0):
            raise ValueError("panel areas must be strictly positive")
        norms = np.linalg.norm(n, axis=1)
        if np.max(np.abs(norms - 1.0)) > 1e-9:
            raise ValueError("normals must be unit vectors")

    @property
    def n_panels(self) -> int:
        return int(self.areas.size)

    @property
    def total_area(self) -> float:
        return float(np.sum(self.areas))

    def velocity_direction(self, incidence: float, sideslip: float = 0.0) -> _FloatArray:
        """Unit freestream direction in body axes for a given attitude.

        With :math:`\\alpha` positive nose-up, the flow arrives from
        ahead and below: :math:`\\hat{\\mathbf{v}}_B =
        (\\cos\\alpha\\cos\\beta, \\sin\\beta, \\sin\\alpha)`
        normalized.
        """
        a, b = float(incidence), float(sideslip)
        v = np.array([np.cos(a) * np.cos(b), np.sin(b), np.sin(a) * np.cos(b)])
        return np.asarray(v / np.linalg.norm(v))

    def incidences(self, incidence: float, sideslip: float = 0.0) -> _FloatArray:
        """Per-panel local incidence (rad); positive windward."""
        v_hat = self.velocity_direction(incidence, sideslip)
        sin_delta = -(self.normals @ v_hat)
        return np.asarray(np.arcsin(np.clip(sin_delta, -1.0, 1.0)))

    def loads(
        self,
        incidence: float,
        mach: float,
        dynamic_pressure: float,
        gamma: float = 1.4,
        blend_width: float = 0.02,
        sideslip: float = 0.0,
        cp_max: float | None = None,
    ) -> tuple[_FloatArray, _FloatArray]:
        """Integrated force and moment in body axes.

        ``cp_max`` overrides the perfect-gas Rayleigh-Pitot stagnation value.
        Above about Mach 8 that value is wrong — equilibrium air reaches
        1.93 at Mach 20 where the perfect gas is stuck near its 1.839
        asymptote — and :class:`~aether.aerodynamics.realgas.EquilibriumAir`
        supplies the right one.

        Pressure acts along :math:`-\\mathbf{n}`, so the panel force is
        :math:`-C_p\\,q_{\\mathrm{dyn}}A\\,\\mathbf{n}`.

        Returns
        -------
        tuple
            ``(force, moment)``, each shape ``(3,)``, in N and N·m.
        """
        if not (np.isfinite(dynamic_pressure) and dynamic_pressure > 0.0):
            raise ValueError(f"dynamic_pressure must be finite and > 0, got {dynamic_pressure}")
        delta = self.incidences(incidence, sideslip)
        cp = blended_pressure_coefficient(
            delta, mach, gamma=gamma, blend_width=blend_width, cp_max=cp_max
        )
        panel_force = -(cp * dynamic_pressure * self.areas)[:, np.newaxis] * self.normals
        force = np.sum(panel_force, axis=0)
        arms = self.centroids - self.reference_point
        moment = np.sum(np.cross(arms, panel_force), axis=0)
        return np.asarray(force), np.asarray(moment)

    def pitching_moment(
        self,
        incidence: float,
        mach: float,
        dynamic_pressure: float,
        gamma: float = 1.4,
        blend_width: float = 0.02,
        cp_max: float | None = None,
    ) -> float:
        """Body-axis pitching moment (about :math:`y`) at a given incidence."""
        _, moment = self.loads(
            incidence,
            mach,
            dynamic_pressure,
            gamma=gamma,
            blend_width=blend_width,
            cp_max=cp_max,
        )
        return float(moment[1])

    def trim(
        self,
        mach: float,
        dynamic_pressure: float,
        gamma: float = 1.4,
        blend_width: float = 0.02,
        bracket: tuple[float, float] = (np.deg2rad(-20.0), np.deg2rad(30.0)),
    ) -> TrimSolution:
        """Incidence at which the pitching moment vanishes.

        Brent's method on the stated bracket. A missing sign change is
        reported as an error rather than returning a bracket endpoint:
        an untrimmable configuration is a real result, and silently
        returning the edge of the search would hide it.
        """
        lo, hi = float(bracket[0]), float(bracket[1])
        if not lo < hi:
            raise ValueError(f"bracket must satisfy lo < hi, got {bracket}")

        def moment(alpha: float) -> float:
            return self.pitching_moment(
                alpha, mach, dynamic_pressure, gamma=gamma, blend_width=blend_width
            )

        m_lo, m_hi = moment(lo), moment(hi)
        if m_lo * m_hi > 0.0:
            raise ValueError(
                f"no trim point on [{np.rad2deg(lo):.2f}, {np.rad2deg(hi):.2f}] deg: "
                f"pitching moment is {m_lo:.4e} and {m_hi:.4e} N·m at the ends"
            )
        alpha, info = scipy.optimize.brentq(
            moment, lo, hi, xtol=1e-12, rtol=8.9e-16, full_output=True
        )
        force, mom = self.loads(alpha, mach, dynamic_pressure, gamma=gamma, blend_width=blend_width)
        return TrimSolution(
            incidence=float(alpha),
            normal_force=float(force[2]),
            axial_force=float(force[0]),
            pitching_moment=float(mom[1]),
            converged=bool(info.converged),
        )


def curved_lifting_body(
    length: float = 6.0,
    span: float = 3.0,
    upper_thickness: float = 0.45,
    lower_thickness: float = 0.15,
    n_chord: int = 40,
    n_span: int = 24,
    reference_fraction: float = 0.5,
) -> PanelModel:
    """Generic cambered lifting body — a demonstration geometry.

    Upper and lower surfaces are
    :math:`z = \\pm t\\,s(x)\\,c(y)` with a smooth chordwise shape
    :math:`s` and elliptical spanwise distribution :math:`c`. The camber
    (``upper_thickness`` > ``lower_thickness``) gives a non-zero moment
    at zero incidence, so the trim point is non-trivial, and the
    curvature makes the shoulder line sweep through
    :math:`\\delta_c = 0` — the configuration where the blending seam
    can matter.

    Corresponds to no vehicle; carries no design or performance data.
    """
    for name, val in (
        ("length", length),
        ("span", span),
        ("upper_thickness", upper_thickness),
        ("lower_thickness", lower_thickness),
    ):
        if not (np.isfinite(val) and val > 0.0):
            raise ValueError(f"{name} must be finite and > 0, got {val}")
    if n_chord < 4 or n_span < 4:
        raise ValueError(f"need n_chord, n_span >= 4, got ({n_chord}, {n_span})")

    xi = np.linspace(0.0, 1.0, n_chord + 1)
    eta = np.linspace(-1.0, 1.0, n_span + 1)

    def surface(u: _FloatArray, v: _FloatArray, thickness: float, sign: float) -> _FloatArray:
        shape = np.sin(np.pi * u) ** 0.9  # blunt-nosed, smoothly closing aft
        spanwise = np.sqrt(np.maximum(1.0 - v * v, 0.0))
        return sign * thickness * shape * spanwise

    centroids: list[_FloatArray] = []
    normals: list[_FloatArray] = []
    areas: list[float] = []

    for thickness, sign in ((upper_thickness, 1.0), (lower_thickness, -1.0)):
        for i in range(n_chord):
            for j in range(n_span):
                u = np.array([xi[i], xi[i + 1], xi[i + 1], xi[i]])
                v = np.array([eta[j], eta[j], eta[j + 1], eta[j + 1]])
                corners = np.column_stack(
                    [length * u, 0.5 * span * v, surface(u, v, thickness, sign)]
                )
                # split the quad into two triangles for an exact area/normal
                for tri in ((0, 1, 2), (0, 2, 3)):
                    p = corners[list(tri)]
                    cross = np.cross(p[1] - p[0], p[2] - p[0])
                    area = 0.5 * float(np.linalg.norm(cross))
                    if area <= 0.0:
                        continue
                    normal = cross / np.linalg.norm(cross)
                    if normal[2] * sign < 0.0:  # orient outward
                        normal = -normal
                    centroids.append(p.mean(axis=0))
                    normals.append(normal)
                    areas.append(area)

    return PanelModel(
        centroids=np.asarray(centroids),
        normals=np.asarray(normals),
        areas=np.asarray(areas),
        reference_point=np.array([reference_fraction * length, 0.0, 0.0]),
    )


def sphere_cone(
    length: float | None = 1.75,
    base_radius: float | None = 0.277,
    nose_radius: float | None = None,
    half_angle: float | None = np.radians(8.2),
    n_axial: int = 48,
    n_circ: int = 48,
    reference_fraction: float = 0.45,
    include_base: bool = False,
) -> PanelModel:
    """Blunted sphere--cone: a spherical cap tangent to a conical frustum.

    The canonical high-ballistic-coefficient entry shape, and the one the
    reachability series uses as its certified exemplar --- chosen because
    the sharp-cone limit has an *exact* solution (Taylor--Maccoll,
    :mod:`aether.aerodynamics.conical`) to validate against, which no
    lifting body offers.

    Defaults are a generic slender re-entry class: :math:`L=1.75`~m,
    :math:`R_b=0.277`~m, :math:`\\theta_c=8.2^\\circ`, whence
    :math:`R_n\\approx2.9`~cm and a blunting ratio
    :math:`R_n/R_b\\approx0.10`. They correspond to no specific vehicle and
    carry no design or performance data; three of the four are supplied
    and the fourth follows from :func:`sphere_cone_closure`.

    Unlike :func:`curved_lifting_body` this shape **trims**: it is
    axisymmetric, so :math:`\\alpha=0` is an equilibrium by symmetry, and
    it is statically stable for any centre of mass forward of the centre
    of pressure. That is what makes an attitude critical manifold exist
    at all, and the demonstration lifting body has no such manifold at
    any reference point.

    Parameters
    ----------
    length, base_radius, nose_radius, half_angle:
        Exactly three must be given; pass ``None`` for the one to solve.
        Lengths in metres, ``half_angle`` in radians.
    n_axial, n_circ:
        Panel counts along the meridian and around the circumference.
    reference_fraction:
        Moment reference point as a fraction of ``length`` along the axis.
    include_base:
        Whether to panel the flat base. Newtonian pressure on a base in
        shadow is zero, so it changes no force, but it closes the surface
        for anything that needs a watertight mesh.
    """
    length, base_radius, nose_radius, half_angle = sphere_cone_closure(
        length, base_radius, nose_radius, half_angle
    )
    if n_axial < 4 or n_circ < 8:
        raise ValueError(f"need n_axial >= 4, n_circ >= 8, got ({n_axial}, {n_circ})")

    phi_tangent = 0.5 * np.pi - half_angle
    x_tangent, _ = sphere_cone_tangency(nose_radius, half_angle)
    if length <= x_tangent:
        raise ValueError("length must exceed the sphere--cone tangency station")

    # Split the meridian between cap and frustum in proportion to arclength,
    # so neither is starved when the body is very slender or very blunt.
    cap_arc = nose_radius * phi_tangent
    cone_arc = (length - x_tangent) / np.cos(half_angle)
    n_cap = max(2, round(n_axial * cap_arc / (cap_arc + cone_arc)))
    n_cone = max(2, n_axial - n_cap)

    station, profile = sphere_cone_meridian(length, nose_radius, half_angle, n_cap, n_cone)
    nodes = list(zip(station, profile, strict=True))
    psi = np.linspace(0.0, 2.0 * np.pi, n_circ + 1)

    centroids: list[_FloatArray] = []
    normals: list[_FloatArray] = []
    areas: list[float] = []

    def add(triangle: _FloatArray, fixed_normal: _FloatArray | None = None) -> None:
        cross = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        area = 0.5 * float(np.linalg.norm(cross))
        if area <= 1e-15:
            return
        mid = triangle.mean(axis=0)
        if fixed_normal is not None:
            # The base is flat and axial: its outward normal is +x, and the
            # radial test below would wrongly rotate it into the meridian,
            # turning a shadowed disc into a windward one.
            normal = fixed_normal
        else:
            normal = cross / np.linalg.norm(cross)
            # Outward means away from the axis; on the nose cap, where the
            # radial reference degenerates, it means forward along -x.
            radial = np.array([0.0, mid[1], mid[2]])
            if np.linalg.norm(radial) > 1e-9:
                if normal @ (radial / np.linalg.norm(radial)) < 0.0:
                    normal = -normal
            elif normal[0] > 0.0:
                normal = -normal
        centroids.append(mid)
        normals.append(normal)
        areas.append(area)

    for (x0, r0), (x1, r1) in itertools.pairwise(nodes):
        for j in range(n_circ):
            a, b = psi[j], psi[j + 1]
            quad = np.array(
                [
                    [x0, r0 * np.cos(a), r0 * np.sin(a)],
                    [x1, r1 * np.cos(a), r1 * np.sin(a)],
                    [x1, r1 * np.cos(b), r1 * np.sin(b)],
                    [x0, r0 * np.cos(b), r0 * np.sin(b)],
                ]
            )
            add(quad[[0, 1, 2]])
            add(quad[[0, 2, 3]])

    if include_base:
        for j in range(n_circ):
            a, b = psi[j], psi[j + 1]
            add(
                np.array(
                    [
                        [length, 0.0, 0.0],
                        [length, base_radius * np.cos(b), base_radius * np.sin(b)],
                        [length, base_radius * np.cos(a), base_radius * np.sin(a)],
                    ]
                ),
                fixed_normal=np.array([1.0, 0.0, 0.0]),
            )

    # The same net the panels above were cut from, kept rather than discarded.
    # This generator predates :class:`SurfaceGrid` and used to hand back panels
    # only, which made it the one sphere-cone in the package that could not be
    # meshed -- so the meshable sphere-cone had to be spelled as a
    # single-segment :func:`blunted_multiconic`, and which function to call
    # depended on what you meant to do with the answer. There was never a
    # reason for that beyond the order the two were written in.
    profile = np.asarray(nodes, dtype=np.float64)
    net = np.stack(
        [
            profile[:, 0][:, None] * np.ones_like(psi)[None, :],
            profile[:, 1][:, None] * np.cos(psi)[None, :],
            profile[:, 1][:, None] * np.sin(psi)[None, :],
        ],
        axis=-1,
    )

    return PanelModel(
        centroids=np.asarray(centroids),
        normals=np.asarray(normals),
        areas=np.asarray(areas),
        reference_point=np.array([reference_fraction * length, 0.0, 0.0]),
        surface=SurfaceGrid(vertices=net),
    )


def caret_lifting_body(
    length: float = 4.0,
    semi_span: float = 1.2,
    keel_depth: float = 0.32,
    n_chord: int = 40,
    n_span: int = 24,
    reference_fraction: float = 0.50,
    include_base: bool = True,
) -> PanelModel:
    """Caret-form lifting body: the high-L/D counterpart to :func:`sphere_cone`.

    .. warning::

       **Not on-design, and not the same shape as**
       :func:`aether.geometry.bodies.caret_lifting_body`. This one is
       parameterised by span and keel depth directly, which means its leading
       edges do not generally lie on the shock its own compression surface
       makes: at the defaults, the implied deflection is 4.57 degrees, the
       Mach 8 shock angle is 10.49 degrees, and the edges sit 0.74 m *above*
       the shock. Flow spills around them, so the lift-to-drag ratio is not
       representative of a real waverider.

       That is deliberate. This function is the panel method's *lifting
       exemplar* — the shape used in Paper I §6 to exercise objects an
       axisymmetric body cannot, because a cone trims at zero incidence where
       it makes no lift. It needs to lift and to have a shoulder line that
       sweeps through zero deflection; it does not need to be on-design.

       For a body whose numbers are meant to represent the waverider *class* —
       anything fed to CFD, or quoted in a comparison — use
       :func:`aether.geometry.bodies.caret_lifting_body`, which solves the
       oblique-shock relation and places the leading edges in the shock plane
       by construction.


    A waverider rides its own attached shock --- the leading edges lie *on*
    the shock surface, so the high pressure behind it cannot spill around
    to the upper surface, and the lift-to-drag ratio escapes the barrier
    that limits blunt bodies. The caret is the simplest member of the
    family: a planar-shock design whose lower surface is a straight
    dihedral V and whose upper surface is a freestream-aligned plate.

    Geometry is a ruled surface between two straight lines from the nose:
    the leading edge to :math:`(L,\\pm b,0)` and the keel to
    :math:`(L,0,-d)`. Every chordwise station is therefore a straight
    segment, which is what keeps the shock planar and the design closed
    form.

    **Why the series needs this shape as well as the cone.** An
    axisymmetric cone trims at zero incidence, where it generates exactly
    no lift, so it flies a ballistic arc: there is no equilibrium glide
    manifold, no skip, and nothing for the quasi-equilibrium skeleton or
    the boundary-layer bubbles to describe. The reduction those objects
    support is only exercised by a lifting vehicle, and this is the
    generic one. The two shapes are different *letters* of the same
    hybrid alphabet, not competing exemplars.

    Parameters
    ----------
    length, semi_span, keel_depth:
        Metres. ``keel_depth`` sets the compression: the lower surface
        deflects the flow by roughly :math:`\\arctan(d/L)`, and L/D falls
        as it grows.
    n_chord, n_span:
        Panel counts along and across the chord.
    reference_fraction:
        Moment reference as a fraction of ``length`` along the axis.
    include_base:
        Panel the blunt base. Unlike the cone the base here is a real
        area, so it is closed by default.
    """
    for name, value in (("length", length), ("semi_span", semi_span), ("keel_depth", keel_depth)):
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be finite and > 0, got {value}")
    if n_chord < 4 or n_span < 4:
        raise ValueError(f"need n_chord, n_span >= 4, got ({n_chord}, {n_span})")

    centroids: list[_FloatArray] = []
    normals: list[_FloatArray] = []
    areas: list[float] = []

    def add(triangle: _FloatArray, outward_hint: _FloatArray) -> None:
        cross = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        area = 0.5 * float(np.linalg.norm(cross))
        if area <= 1e-15:
            return
        normal = cross / np.linalg.norm(cross)
        if normal @ outward_hint < 0.0:
            normal = -normal
        centroids.append(triangle.mean(axis=0))
        normals.append(normal)
        areas.append(area)

    chord = np.linspace(0.0, 1.0, n_chord + 1)
    span = np.linspace(0.0, 1.0, n_span + 1)
    up, down = np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])

    for sign in (1.0, -1.0):
        for i in range(n_chord):
            u0, u1 = chord[i], chord[i + 1]
            for j in range(n_span):
                v0, v1 = span[j], span[j + 1]

                def lower(u: float, v: float, side: float = sign) -> _FloatArray:
                    # ruled from keel (v=0) to leading edge (v=1)
                    return np.array(
                        [
                            u * length,
                            side * u * semi_span * v,
                            -u * keel_depth * (1.0 - v),
                        ]
                    )

                def upper(u: float, v: float, side: float = sign) -> _FloatArray:
                    return np.array([u * length, side * u * semi_span * v, 0.0])

                quad_l = np.array([lower(u0, v0), lower(u1, v0), lower(u1, v1), lower(u0, v1)])
                add(quad_l[[0, 1, 2]], down)
                add(quad_l[[0, 2, 3]], down)
                quad_u = np.array([upper(u0, v0), upper(u1, v0), upper(u1, v1), upper(u0, v1)])
                add(quad_u[[0, 1, 2]], up)
                add(quad_u[[0, 2, 3]], up)

    if include_base:
        aft = np.array([1.0, 0.0, 0.0])
        for sign in (1.0, -1.0):
            for j in range(n_span):
                v0, v1 = span[j], span[j + 1]
                keel_0 = np.array([length, sign * semi_span * v0, -keel_depth * (1.0 - v0)])
                keel_1 = np.array([length, sign * semi_span * v1, -keel_depth * (1.0 - v1)])
                top_0 = np.array([length, sign * semi_span * v0, 0.0])
                top_1 = np.array([length, sign * semi_span * v1, 0.0])
                add(np.array([keel_0, keel_1, top_1]), aft)
                add(np.array([keel_0, top_1, top_0]), aft)

    return PanelModel(
        centroids=np.asarray(centroids),
        normals=np.asarray(normals),
        areas=np.asarray(areas),
        reference_point=np.array([reference_fraction * length, 0.0, 0.0]),
    )


def _grid_to_panels(
    vertices: _FloatArray, reference_point: _FloatArray, outward_hint: _FloatArray | None = None
) -> PanelModel:
    """Helper to convert an (N_axial, N_circ, 3) vertex grid into a PanelModel."""
    n_ax, n_circ, _ = vertices.shape
    centroids, normals, areas = [], [], []

    def add(tri: _FloatArray) -> None:
        cross = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        area = 0.5 * float(np.linalg.norm(cross))
        if area <= 1e-15:
            return
        normal = cross / np.linalg.norm(cross)
        mid = tri.mean(axis=0)

        radial = mid.copy()
        radial[0] = 0.0
        if outward_hint is not None:
            if normal @ outward_hint < 0:
                normal = -normal
        elif np.linalg.norm(radial) > 1e-9:
            if normal @ (radial / np.linalg.norm(radial)) < 0.0:
                normal = -normal
        elif normal[0] > 0.0:
            normal = -normal

        centroids.append(mid)
        normals.append(normal)
        areas.append(area)

    for i in range(n_ax - 1):
        for j in range(n_circ - 1):
            p00, p10 = vertices[i, j], vertices[i + 1, j]
            p11, p01 = vertices[i + 1, j + 1], vertices[i, j + 1]

            quad = np.array([p00, p10, p11, p01])
            add(quad[[0, 1, 2]])
            add(quad[[0, 2, 3]])

    return PanelModel(
        centroids=np.asarray(centroids),
        normals=np.asarray(normals),
        areas=np.asarray(areas),
        reference_point=reference_point,
        surface=SurfaceGrid(vertices=np.asarray(vertices, dtype=np.float64)),
    )


def blunted_multiconic(
    nose_radius: float = 0.05,
    lengths: Sequence[float] | None = None,
    half_angles: Sequence[float] | None = None,
    fillet_radii: Sequence[float] | None = None,
    n_axial_per_segment: int = 40,
    n_circ: int = 48,
    reference_fraction: float = 0.5,
) -> PanelModel:
    """Generates a C1 continuous biconic, triconic, or n-conic with tangent fillets."""
    # Defaults built here, not in the signature: a list literal in a default
    # is shared by every call that takes it, so one caller mutating it changes
    # the geometry every later caller gets.
    lengths = [1.0, 1.5] if lengths is None else list(lengths)
    half_angles = [np.radians(12.0), np.radians(7.0)] if half_angles is None else list(half_angles)
    fillet_radii = [0.1] if fillet_radii is None else list(fillet_radii)
    if len(lengths) != len(half_angles):
        raise ValueError("Lengths and half_angles arrays must match.")
    if len(fillet_radii) != len(lengths) - 1:
        raise ValueError("Must provide exactly one fillet radius per junction.")

    n_segments = len(lengths)
    psi = np.linspace(0.0, 2.0 * np.pi, n_circ + 1)

    x_profile, r_profile = [], []

    theta_0 = half_angles[0]
    phi_tangent = 0.5 * np.pi - theta_0
    for phi in np.linspace(0, phi_tangent, n_axial_per_segment):
        x_profile.append(nose_radius * (1.0 - np.cos(phi)))
        r_profile.append(nose_radius * np.sin(phi))

    x_current = nose_radius * (1.0 - np.sin(theta_0))
    r_current = nose_radius * np.cos(theta_0)

    for i in range(n_segments):
        theta = half_angles[i]

        if i < n_segments - 1:
            theta_next = half_angles[i + 1]
            R_f = fillet_radii[i]

            L_seg = lengths[i]
            x_int = x_current + L_seg
            r_int = r_current + L_seg * np.tan(theta)

            half_delta = abs(theta - theta_next) / 2.0
            # Tangent length of a circular arc of radius R_f blending two lines
            # that meet at deflection angle delta: T = R tan(delta/2). The
            # reciprocal, R/tan(delta/2), is the *cotangent* form and diverges
            # as the cones become parallel — which is the usual case, since
            # consecutive cone angles differ by a few degrees. At the default
            # 12/7-degree junction it returns 2.29 m of tangent for a 1.0 m
            # segment, so both tangency points land outside their own frusta and
            # the profile folds back through the nose. That fold is invisible to
            # a panel integration, which sums unordered faces, and fatal to a
            # mesh generator, which sees overlapping facets.
            L_tan = R_f * np.tan(half_delta) if half_delta > 1e-6 else 0.0

            x_end_frustum = x_int - L_tan * np.cos(theta)
            r_end_frustum = r_int - L_tan * np.sin(theta)

            x_start_next = x_int + L_tan * np.cos(theta_next)
            r_start_next = r_int + L_tan * np.sin(theta_next)

            x_c = x_end_frustum + R_f * np.sin(theta)
            r_c = r_end_frustum - R_f * np.cos(theta)

            x_frust = np.linspace(x_current, x_end_frustum, n_axial_per_segment)[1:]
            r_frust = np.linspace(r_current, r_end_frustum, n_axial_per_segment)[1:]
            x_profile.extend(x_frust)
            r_profile.extend(r_frust)

            angles = np.linspace(
                np.pi / 2 - theta,
                np.pi / 2 - theta_next,
                max(4, n_axial_per_segment // 2),
            )[1:]
            for a in angles:
                x_profile.append(x_c - R_f * np.cos(a))
                r_profile.append(r_c + R_f * np.sin(a))

            x_current, r_current = x_start_next, r_start_next

        else:
            x_end = x_current + lengths[i]
            r_end = r_current + lengths[i] * np.tan(theta)
            x_frust = np.linspace(x_current, x_end, n_axial_per_segment)[1:]
            r_frust = np.linspace(r_current, r_end, n_axial_per_segment)[1:]
            x_profile.extend(x_frust)
            r_profile.extend(r_frust)

    x_prof, r_prof = np.array(x_profile), np.array(r_profile)
    vertices = np.zeros((len(x_prof), len(psi), 3))

    for i, (x, r) in enumerate(zip(x_prof, r_prof, strict=True)):
        for j, p in enumerate(psi):
            vertices[i, j] = [x, r * np.cos(p), r * np.sin(p)]

    ref_pt = np.array([reference_fraction * np.max(x_prof), 0.0, 0.0])
    return _grid_to_panels(vertices, ref_pt)


def exact_mitered_bent_biconic(
    nose_radius: float = 0.05,
    L1: float = 1.0,
    theta1: float = np.radians(10.0),
    L2: float = 1.5,
    theta2: float = np.radians(6.0),
    bend_angle: float = np.radians(5.0),
    n_axial: int = 60,
    n_circ: int = 48,
    reference_fraction: float = 0.5,
) -> PanelModel:
    """Watertight bent biconic using exact 3D ray-plane miter intersections."""
    psi = np.linspace(0.0, 2.0 * np.pi, n_circ + 1)
    vertices = np.zeros((n_axial, len(psi), 3))
    n_fwd = n_axial // 2
    n_aft = n_axial - n_fwd

    x_tan = nose_radius * (1.0 - np.sin(theta1))
    r_tan = nose_radius * np.cos(theta1)
    x_apex = x_tan - r_tan / np.tan(theta1)
    v_apex = np.array([x_apex, 0.0, 0.0])

    hinge = np.array([x_tan + L1, 0.0, 0.0])
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([np.cos(bend_angle), 0.0, np.sin(bend_angle)])

    n_plane = v1 - v2
    n_plane /= np.linalg.norm(n_plane)

    P_junction = np.zeros((len(psi), 3))

    for j, p in enumerate(psi):
        d = np.array([np.cos(theta1), np.sin(theta1) * np.cos(p), np.sin(theta1) * np.sin(p)])
        t_junc = np.dot(hinge - v_apex, n_plane) / np.dot(d, n_plane)
        t_tan = r_tan / np.sin(theta1)

        P_junction[j] = v_apex + t_junc * d

        for i in range(n_fwd):
            if i < n_fwd // 3:
                phi = (0.5 * np.pi - theta1) * (i / max(1, (n_fwd // 3 - 1)))
                vertices[i, j] = [
                    nose_radius * (1.0 - np.cos(phi)),
                    nose_radius * np.sin(phi) * np.cos(p),
                    nose_radius * np.sin(phi) * np.sin(p),
                ]
            else:
                u = (i - n_fwd // 3 + 1) / (n_fwd - n_fwd // 3)
                vertices[i, j] = v_apex + (t_tan + u * (t_junc - t_tan)) * d

    R_nom = r_tan + L1 * np.tan(theta1)
    R_base = R_nom + L2 * np.tan(theta2)
    C_base = hinge + L2 * v2

    u2 = np.array([0.0, 1.0, 0.0])
    w2 = np.array([-np.sin(bend_angle), 0.0, np.cos(bend_angle)])

    for j, p in enumerate(psi):
        B = C_base + R_base * (np.cos(p) * u2 + np.sin(p) * w2)
        for i in range(n_aft):
            vertices[n_fwd + i, j] = P_junction[j] + ((i + 1) / n_aft) * (B - P_junction[j])

    total_len = hinge[0] + L2 * np.cos(bend_angle)
    ref_pt = np.array([reference_fraction * total_len, 0.0, 0.0])

    return _grid_to_panels(vertices, ref_pt)


def smooth_bent_biconic(
    nose_radius: float = 0.05,
    L1: float = 1.0,
    theta1: float = np.radians(10.0),
    L2: float = 1.5,
    theta2: float = np.radians(6.0),
    bend_angle: float = np.radians(5.0),
    spine_bend_radius: float = 0.5,
    n_axial: int = 80,
    n_circ: int = 48,
    reference_fraction: float = 0.5,
) -> PanelModel:
    """Watertight C1 continuous bent biconic utilizing a planar Bishop frame loft."""
    psi = np.linspace(0.0, 2.0 * np.pi, n_circ + 1)
    vertices = np.zeros((n_axial, len(psi), 3))

    x_tan = nose_radius * (1.0 - np.sin(theta1))
    r_tan = nose_radius * np.cos(theta1)
    x_apex = x_tan - r_tan / np.tan(theta1)

    L_fwd_spine = L1 - spine_bend_radius * np.tan(bend_angle / 2.0)
    if L_fwd_spine <= x_tan:
        raise ValueError("Spine bend radius is too large; intersects nose cap.")

    arc_length = spine_bend_radius * bend_angle

    n_nose = max(4, n_axial // 10)
    n_arc = max(4, n_axial // 6)
    n_fwd = (n_axial - n_nose - n_arc) // 2
    n_aft = n_axial - n_nose - n_fwd - n_arc

    s_vals = []
    phi_vals = np.linspace(0, 0.5 * np.pi - theta1, n_nose, endpoint=False)
    for phi in phi_vals:
        s_vals.append((nose_radius * (1.0 - np.cos(phi))) - x_apex)

    s_fwd_end = L_fwd_spine - x_apex
    s_vals.extend(np.linspace(s_vals[-1], s_fwd_end, n_fwd, endpoint=False)[1:])

    s_arc_end = s_fwd_end + arc_length
    s_vals.extend(np.linspace(s_fwd_end, s_arc_end, n_arc, endpoint=False))

    s_aft_end = s_arc_end + (L2 - spine_bend_radius * np.tan(bend_angle / 2.0))
    s_vals.extend(np.linspace(s_arc_end, s_aft_end, n_aft))

    B_vec = np.array([0.0, 1.0, 0.0])

    for i, s in enumerate(s_vals):
        if i < n_nose:
            phi = phi_vals[i]
            x_c = nose_radius * (1.0 - np.cos(phi))
            r_c = nose_radius * np.sin(phi)
            for j, p in enumerate(psi):
                vertices[i, j] = [x_c, r_c * np.cos(p), r_c * np.sin(p)]
            continue

        if s <= s_fwd_end:
            gamma = np.array([x_apex + s, 0.0, 0.0])
            N_vec = np.array([0.0, 0.0, 1.0])
            R = r_tan + (s - (x_tan - x_apex)) * np.tan(theta1)

        elif s <= s_arc_end:
            theta_local = (s - s_fwd_end) / spine_bend_radius
            gamma = np.array(
                [
                    x_apex + s_fwd_end + spine_bend_radius * np.sin(theta_local),
                    0.0,
                    spine_bend_radius - spine_bend_radius * np.cos(theta_local),
                ]
            )
            N_vec = np.array([-np.sin(theta_local), 0.0, np.cos(theta_local)])

            u_arc = (s - s_fwd_end) / arc_length
            theta_eff = theta1 * (1 - u_arc) + theta2 * u_arc
            R = (
                r_tan
                + (s_fwd_end - (x_tan - x_apex)) * np.tan(theta1)
                + (s - s_fwd_end) * np.tan(theta_eff)
            )

        else:
            s_local = s - s_arc_end
            gamma_arc_end = np.array(
                [
                    x_apex + s_fwd_end + spine_bend_radius * np.sin(bend_angle),
                    0.0,
                    spine_bend_radius - spine_bend_radius * np.cos(bend_angle),
                ]
            )
            T_vec = np.array([np.cos(bend_angle), 0.0, np.sin(bend_angle)])
            N_vec = np.array([-np.sin(bend_angle), 0.0, np.cos(bend_angle)])

            gamma = gamma_arc_end + s_local * T_vec

            R_arc_end = (
                r_tan
                + (s_fwd_end - (x_tan - x_apex)) * np.tan(theta1)
                + arc_length * np.tan(0.5 * (theta1 + theta2))
            )
            R = R_arc_end + s_local * np.tan(theta2)

        for j, p in enumerate(psi):
            vertices[i, j] = gamma + R * (np.cos(p) * B_vec + np.sin(p) * N_vec)

    ref_pt = np.array([reference_fraction * vertices[-1, 0, 0], 0.0, 0.0])
    return _grid_to_panels(vertices, ref_pt)


def spatular_wedge(
    length: float = 3.0,
    nose_radius_y: float = 0.1,
    nose_radius_z: float = 0.1,
    base_half_span: float = 1.0,
    base_half_thickness: float = 0.3,
    p_span: float = 0.5,
    p_thickness: float = 0.75,
    n_power_nose: float = 2.0,
    n_power_base: float = 1.2,
    p_n_exp: float = 2.0,
    n_axial: int = 60,
    n_circ: int = 48,
    reference_fraction: float = 0.5,
) -> PanelModel:
    """Exact power-law blended lifting body using rigorous Lamé curve transitions."""
    psi = np.linspace(0.0, 2.0 * np.pi, n_circ + 1)
    vertices = np.zeros((n_axial, len(psi), 3))

    u_vals = np.linspace(1e-5, 1.0, n_axial)

    for i, u in enumerate(u_vals):
        x = u * length

        a = nose_radius_y + (base_half_span - nose_radius_y) * (u**p_span)
        b = nose_radius_z + (base_half_thickness - nose_radius_z) * (u**p_thickness)
        n_exp = n_power_nose + (n_power_base - n_power_nose) * (u**p_n_exp)

        for j, p in enumerate(psi):
            cos_p = np.cos(p)
            sin_p = np.sin(p)

            y = a * (np.abs(cos_p) ** (2.0 / n_exp)) * np.sign(cos_p)
            z = b * (np.abs(sin_p) ** (2.0 / n_exp)) * np.sign(sin_p)

            rad_fraction = (y / a) ** 2 + (z / b) ** 2
            x_offset = nose_radius_y * (1.0 - np.sqrt(max(0.0, 1.0 - rad_fraction)))

            x_local = x + x_offset * (1.0 - u)
            vertices[i, j] = [x_local, y, z]

    ref_pt = np.array([reference_fraction * length, 0.0, 0.0])
    return _grid_to_panels(vertices, ref_pt)
