"""Osculating-cone waverider design.

A waverider is defined by the flow it rides. The caret in
:func:`aether.geometry.bodies.caret_waverider` rides a *plane* shock, which is
the simplest case and fixes the whole shape once the design Mach number and
one deflection are chosen. The osculating-cone method generalises that: the
shock is allowed to be a **curved** surface, prescribed by the designer, and
the body is whatever rides it.

The construction
----------------

Everything is specified in the **base plane** at :math:`x = L`, looking
upstream:

* The **shockwave profile curve** (SPC) is the trace of the shock in that
  plane -- a curve :math:`z_s(y)` the designer draws.
* At each point :math:`P` of the SPC, the **osculating plane** contains the
  freestream direction and the local normal to the SPC. In that plane the flow
  is taken to be the flow over an axisymmetric cone, whose axis lies at the
  SPC's local **centre of curvature** and whose shock angle is the design
  angle :math:`\\beta`.
* The **inlet capture curve** (ICC) fixes where the leading edge sits inside
  each osculating plane. Here it is given as a fraction of the local
  osculating radius, which is what makes the leading edge lie *on* the shock
  by construction rather than by iteration.

The lower surface is then traced, one osculating plane at a time, by following
a streamline of the local conical field from the leading edge back to the base
plane. The upper surface is freestream-aligned: from the leading edge straight
back, so it generates no compression and the high pressure underneath cannot
spill over the edge.

Why the shock angle is constant along the SPC
---------------------------------------------

Because that is the assumption the method is built on, and it is what makes it
cheap: with :math:`\\beta` fixed, every osculating plane sees the *same*
Taylor--Maccoll solution. Conical flow is self-similar in radius, so one
integration serves the whole span and the planes differ only in where their
axis is and how far away it is. Letting :math:`\\beta` vary would be a
different -- and much more expensive -- method.

What this does not claim
------------------------

Inviscid design. The traced surface is a stream surface of the *inviscid*
conical field, so the body rides its shock exactly at the design point and
nowhere else: off-design, and with a boundary layer, the flow spills. Leading
edges are sharp for the same reason, and any practical version is blunted,
which is a departure from the design rather than a refinement of it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from aether.aerodynamics.panels import SurfaceGrid
from aether.geometry.mesh import VehicleMesh

_FloatArray = NDArray[np.float64]
_IntArray = NDArray[np.int64]

__all__ = [
    "OsculatingConeWaverider",
    "circular_shock_curve",
    "osculating_cone_waverider",
    "power_shock_curve",
]


def circular_shock_curve(radius: float) -> Callable[[_FloatArray], _FloatArray]:
    """A circular-arc SPC of the given radius, lowest at the centreline.

    Constant curvature, so every osculating plane shares one osculating
    radius and the design collapses to the **cone-derived** waverider -- the
    same body a single cone's stream surface would give. That degeneracy is
    useful: it is the case with an independent answer, and the tests use it
    as one.
    """

    def curve(y: _FloatArray) -> _FloatArray:
        if np.any(np.abs(y) >= radius):
            raise ValueError(
                f"a circular shock curve of radius {radius:g} cannot span "
                f"|y| = {float(np.abs(y).max()):g}; the span must stay inside the arc"
            )
        return np.asarray(radius - np.sqrt(radius**2 - y**2))

    return curve


def power_shock_curve(
    depth: float, half_span: float, exponent: float = 2.0
) -> Callable[[_FloatArray], _FloatArray]:
    """A power-law SPC: :math:`z_s = d\\,(|y|/b)^{n}`, flat-bottomed for ``n > 2``.

    Curvature varies along the span, which is the case the osculating-cone
    method exists for -- the local cone is a different cone at every station.
    """

    def curve(y: _FloatArray) -> _FloatArray:
        return np.asarray(depth * (np.abs(y) / half_span) ** exponent)

    return curve


@dataclass(frozen=True)
class OsculatingConeWaverider:
    """A designed waverider, and the design it came from."""

    surface: SurfaceGrid
    """The two surfaces as one ring net, for callers that want a grid."""

    lower: _FloatArray
    """``(n_axial, n_span, 3)`` compression surface; row 0 is the leading edge."""

    upper: _FloatArray
    """``(n_axial, n_span, 3)`` freestream surface; row 0 is the leading edge."""

    leading_edge: _FloatArray
    """``(n_span, 3)`` leading-edge points, which lie on the shock."""
    shock_points: _FloatArray
    """``(n_span, 3)`` the SPC, in the base plane."""
    osculating_radius: _FloatArray
    """``(n_span,)`` local radius of curvature of the SPC."""
    design_mach: float
    shock_angle: float
    cone_angle: float
    """The cone half-angle whose attached shock sits at :attr:`shock_angle`."""
    length: float

    def to_mesh(self, name: str = "osculating-cone-waverider") -> VehicleMesh:
        """Stitch the design into a watertight, outward-wound triangle mesh.

        Built from patches rather than from the ring net, because the ring is
        the wrong topology for this body. A structured ring assumes every
        station is a closed loop, and a waverider's first station is not a
        loop -- it is the leading edge, where the two surfaces *meet*. Handed
        that, the weld collapses the row onto itself and leaves the edges
        along it shared by three faces: no open boundary, and still not a
        solid.

        Five patches close it: the compression surface, the freestream
        surface, the base, and a flat panel at each span tip where the two
        surfaces separate downstream of the edge they share.
        """
        n_axial, n_span = self.lower.shape[:2]
        edge = self.lower[0]
        lower_rows, upper_rows = self.lower[1:], self.upper[1:]
        vertices = np.concatenate([edge, lower_rows.reshape(-1, 3), upper_rows.reshape(-1, 3)])
        lower_base, upper_base = n_span, n_span + (n_axial - 1) * n_span

        def node(row: int, column: int, upper: bool) -> _IntArray:
            if row == 0:
                return np.asarray(column, dtype=np.int64)
            offset = upper_base if upper else lower_base
            return np.asarray(offset + (row - 1) * n_span + column, dtype=np.int64)

        faces: list[list[int]] = []
        for row in range(n_axial - 1):
            for column in range(n_span - 1):
                a, b = node(row, column, False), node(row, column + 1, False)
                c, d = node(row + 1, column + 1, False), node(row + 1, column, False)
                faces += [[int(a), int(c), int(b)], [int(a), int(d), int(c)]]
                a, b = node(row, column, True), node(row, column + 1, True)
                c, d = node(row + 1, column + 1, True), node(row + 1, column, True)
                faces += [[int(a), int(b), int(c)], [int(a), int(c), int(d)]]

        # The base, closing the two surfaces at x = L. Wound against the
        # compression surface's direction, so that its outward normal points
        # aft rather than back into the body -- which is what makes the shell
        # consistently oriented and not merely closed.
        last = n_axial - 1
        for column in range(n_span - 1):
            a, b = node(last, column, False), node(last, column + 1, False)
            c, d = node(last, column + 1, True), node(last, column, True)
            faces += [[int(a), int(c), int(b)], [int(a), int(d), int(c)]]

        for column, flip in ((0, True), (n_span - 1, False)):
            for row in range(n_axial - 1):
                a, b = node(row, column, False), node(row + 1, column, False)
                c, d = node(row + 1, column, True), node(row, column, True)
                # At the leading edge the two surfaces share their vertex, so
                # the tip panel is a triangle there and not a quadrilateral.
                # Emitting the quad anyway leaves a face with a repeated
                # corner: zero area, invisible, and enough to make the solid
                # non-manifold along the one edge that matters.
                quad = (
                    [[int(a), int(b), int(c)]]
                    if row == 0
                    else [[int(a), int(b), int(c)], [int(a), int(c), int(d)]]
                )
                faces += [face[::-1] for face in quad] if flip else quad

        mesh = VehicleMesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64), name=name)
        triangles = mesh.triangles
        volume = float(
            np.einsum(
                "ij,ij->i",
                triangles[:, 0],
                np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            ).sum()
            / 6.0
        )
        if volume < 0.0:
            mesh = VehicleMesh(
                vertices=vertices, faces=np.asarray(faces, dtype=np.int64)[:, ::-1], name=name
            )
        return mesh


def _curvature(y: _FloatArray, z: _FloatArray) -> tuple[_FloatArray, _FloatArray]:
    """Radius of curvature and the unit normal toward the centre, for ``z(y)``.

    Computed as the **osculating circle through three consecutive points**,
    which is what an osculating radius is, rather than by differentiating the
    sampled curve twice. The distinction is not pedantic: a second derivative
    from :func:`numpy.gradient` falls back to one-sided differences at the
    ends of the span, and on a circular arc -- whose curvature is exactly
    constant -- that reported a radius varying by a factor of two from the
    centreline to the tips. Every osculating plane near the tip would then
    have been given the wrong cone.

    Through three points the circumradius is
    :math:`R = \\frac{abc}{4A}`, exact for any arc that really is circular and
    second-order accurate otherwise. The endpoints reuse their neighbouring
    triple, which is exact for a circle and the best available elsewhere.
    """
    points = np.column_stack([y, z])
    index = np.arange(points.shape[0])
    left = np.clip(index - 1, 0, points.shape[0] - 1)
    right = np.clip(index + 1, 0, points.shape[0] - 1)
    # The endpoints have no neighbour on one side; step their stencil inward
    # so it stays three distinct points rather than collapsing to two.
    left[0], right[0] = 0, 2
    left[-1], right[-1] = points.shape[0] - 3, points.shape[0] - 1
    middle = np.where(
        index == 0, 1, np.where(index == points.shape[0] - 1, points.shape[0] - 2, index)
    )

    a, b, c = points[left], points[middle], points[right]
    ab, bc, ca = b - a, c - b, a - c
    area = 0.5 * (ab[:, 0] * (-ca[:, 1]) - ab[:, 1] * (-ca[:, 0]))
    lengths = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1) * np.linalg.norm(ca, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        radius = np.where(np.abs(area) > 0.0, lengths / (4.0 * np.abs(area)), np.inf)

    # Circumcentre, from the perpendicular bisectors, then the unit normal
    # from each point toward it -- the concave side, where the cone's axis is.
    d = 2.0 * (
        a[:, 0] * (b[:, 1] - c[:, 1])
        + b[:, 0] * (c[:, 1] - a[:, 1])
        + c[:, 0] * (a[:, 1] - b[:, 1])
    )
    sq = (points**2).sum(axis=1)
    sq_a, sq_b, sq_c = (
        (points[left] ** 2).sum(1),
        (points[middle] ** 2).sum(1),
        (points[right] ** 2).sum(1),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        centre_x = (
            sq_a * (b[:, 1] - c[:, 1]) + sq_b * (c[:, 1] - a[:, 1]) + sq_c * (a[:, 1] - b[:, 1])
        ) / d
        centre_y = (
            sq_a * (c[:, 0] - b[:, 0]) + sq_b * (a[:, 0] - c[:, 0]) + sq_c * (b[:, 0] - a[:, 0])
        ) / d
    del sq
    offset = np.column_stack([centre_x, centre_y]) - points
    norm = np.linalg.norm(offset, axis=1)
    normal = offset / np.where(norm > 0.0, norm, 1.0)[:, None]
    return radius, normal


def osculating_cone_waverider(
    design_mach: float = 8.0,
    shock_angle: float = np.radians(14.0),
    length: float = 4.0,
    half_span: float | None = None,
    shock_curve: Callable[[_FloatArray], _FloatArray] | None = None,
    capture_fraction: float = 0.55,
    n_span: int = 41,
    n_axial: int = 41,
    gamma: float = 1.4,
) -> OsculatingConeWaverider:
    """Design a waverider from a prescribed shock, by osculating cones.

    Parameters
    ----------
    design_mach, shock_angle:
        The flow the body is to ride. ``shock_angle`` must be attachable at
        this Mach number -- above the maximum-deflection angle no cone
        supports it and there is nothing to trace.
    length, half_span:
        Axial length from the apex to the base plane, and the half-width of
        the shockwave profile curve. ``half_span`` defaults to a fraction of
        the natural scale below.
    shock_curve:
        ``z_s(y)`` in the base plane. Defaults to a circular arc of radius
        :math:`L\tan\beta`, which is the cone-derived case.

    The natural scale
    -----------------

    The osculating radius is not free once the length and shock angle are
    chosen. The local shock runs from its apex to the base plane at
    :math:`\beta`, so an osculating radius :math:`R` puts that apex
    :math:`R/\tan\beta` upstream of the base -- and a body whose apex is
    where the design wants it therefore has :math:`R \approx L\tan\beta`.
    Choosing :math:`R` independently is not wrong, it just describes a
    different (and usually much longer) vehicle than the ``length`` suggests,
    which is why both defaults are derived from that scale rather than left as
    free numbers to be set inconsistently.
    capture_fraction:
        Where the leading edge sits in each osculating plane, as a fraction of
        the local osculating radius measured from the local axis. Zero puts it
        on the axis and one puts it on the shock trace itself, so both ends
        are degenerate.
    """
    from aether.aerodynamics.conical import conical_field, solve_cone

    if not 0.0 < capture_fraction < 1.0:
        raise ValueError(f"capture_fraction must lie in (0, 1), got {capture_fraction}")
    if length <= 0.0:
        raise ValueError(f"length must be positive, got {length}")
    scale = length * np.tan(shock_angle)
    if half_span is None:
        half_span = 0.6 * scale
    if half_span <= 0.0:
        raise ValueError(f"half_span must be positive, got {half_span}")

    # The cone whose attached shock sits at the design angle. Solved once:
    # every osculating plane sees the same conical solution, scaled.
    cone_angle = _cone_angle_for_shock(design_mach, shock_angle, gamma)
    field = conical_field(design_mach, cone_angle, gamma)
    solution = solve_cone(design_mach, cone_angle, gamma)
    if abs(solution.shock_angle - shock_angle) > 1e-6:
        raise ValueError(
            f"shock angle {np.degrees(shock_angle):.4f} deg is not attachable at Mach "
            f"{design_mach}; the closest attached solution is "
            f"{np.degrees(solution.shock_angle):.4f} deg"
        )

    curve = shock_curve if shock_curve is not None else circular_shock_curve(scale)
    span = np.linspace(-half_span, half_span, int(n_span))
    height = np.asarray(curve(span), dtype=np.float64)
    radius, normal = _curvature(span, height)
    if not np.all(np.isfinite(radius)):
        raise ValueError(
            "the shockwave profile curve is straight somewhere, so it has no osculating "
            "cone there; a straight segment is a caret, not an osculating-cone design"
        )

    lower = np.empty((int(n_axial), int(n_span), 3), dtype=np.float64)
    upper = np.empty_like(lower)
    leading = np.empty((int(n_span), 3), dtype=np.float64)

    for index in range(int(n_span)):
        osculating = float(radius[index])
        # The local cone: axis parallel to +x, through the centre of
        # curvature, apex placed so its shock reaches the SPC point at x = L.
        centre = np.array([span[index], height[index]]) + osculating * normal[index]
        apex_x = length - osculating / np.tan(shock_angle)
        outward = -normal[index]

        # Leading edge: on the shock, at `capture_fraction` of the way out.
        edge_radius = capture_fraction * osculating
        edge_x = apex_x + edge_radius / np.tan(shock_angle)
        edge_yz = centre + edge_radius * outward
        leading[index] = (edge_x, edge_yz[0], edge_yz[1])

        # Lower surface: the streamline through that point, to the base plane.
        station, offset = field.streamline_to_station(
            shock_radius=edge_radius / np.sin(shock_angle),
            station=length - apex_x,
            samples=4000,
        )
        fraction = np.linspace(0.0, 1.0, int(n_axial))
        sampled_x = edge_x + fraction * (length - edge_x)
        sampled_r = np.interp(sampled_x - apex_x, station, offset)
        lower[:, index, 0] = sampled_x
        lower[:, index, 1:] = centre + sampled_r[:, None] * outward

        # Upper surface: freestream-aligned, so it holds the edge's (y, z).
        upper[:, index, 0] = sampled_x
        upper[:, index, 1] = edge_yz[0]
        upper[:, index, 2] = edge_yz[1]

    # One closed section per streamwise station: out along the lower surface,
    # back along the upper. At the leading edge the two coincide, which is
    # what makes the edge an edge.
    net = np.concatenate([lower, upper[:, ::-1, :], lower[:, :1, :]], axis=1)
    return OsculatingConeWaverider(
        surface=SurfaceGrid(vertices=net),
        lower=lower,
        upper=upper,
        leading_edge=leading,
        shock_points=np.column_stack([np.full_like(span, length), span, height]),
        osculating_radius=radius,
        design_mach=float(design_mach),
        shock_angle=float(shock_angle),
        cone_angle=float(cone_angle),
        length=float(length),
    )


def _cone_angle_for_shock(mach: float, shock_angle: float, gamma: float) -> float:
    """The cone whose attached shock sits at ``shock_angle``.

    :func:`~aether.aerodynamics.conical.solve_cone` goes the other way, so
    this inverts it on the weak branch, where the surface angle rises
    monotonically with the shock angle up to the detachment maximum.
    """
    import scipy.optimize

    from aether.aerodynamics.conical import mach_angle, maximum_cone_angle, solve_cone

    largest, at_shock = maximum_cone_angle(mach, gamma)
    if not mach_angle(mach) < shock_angle <= at_shock:
        raise ValueError(
            f"shock angle {np.degrees(shock_angle):.4f} deg is not on the attached weak "
            f"branch at Mach {mach}: it must exceed the Mach angle "
            f"({np.degrees(mach_angle(mach)):.4f} deg) and not exceed the detachment "
            f"shock angle ({np.degrees(at_shock):.4f} deg)"
        )

    def residual(cone: float) -> float:
        return float(solve_cone(mach, cone, gamma).shock_angle - shock_angle)

    # The bracket's lower end is a small cone rather than a vanishing one: as
    # the cone angle goes to zero its shock approaches the Mach angle, and the
    # root find inside `solve_cone` loses its own bracket before it gets
    # there. A hundredth of a degree is already flat against the Mach angle.
    smallest = np.radians(0.01)
    if residual(smallest) > 0.0:
        raise ValueError(
            f"shock angle {np.degrees(shock_angle):.4f} deg is at or below the Mach angle "
            f"at Mach {mach}; no cone produces it"
        )
    return float(scipy.optimize.brentq(residual, smallest, largest, xtol=1e-12))
