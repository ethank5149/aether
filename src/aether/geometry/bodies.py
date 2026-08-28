"""Parameterised vehicle classes, as exact solids.

The same shape families :mod:`aether.aerodynamics.panels` builds panels from,
expressed as :mod:`aether.geometry.brep` descriptions instead of sampled
triangles. Every parameter is preserved: these are the generators, not a
reduced set of them, and a body defined here can be meshed at any density,
exported to STEP, or measured exactly.

Which representation each family wants is set by its geometry, not by taste:

:func:`sphere_cone` and :func:`blunted_multiconic`
    Bodies of revolution. Built with :class:`~aether.geometry.brep.Revolve`,
    which is **exact in the round** — a true surface of revolution rather
    than a polygon swept through some number of azimuths.
:func:`spatular_wedge`
    Super-elliptical cross-sections whose Lamé exponent varies along the
    body. Lofted through smooth closed B-splines, because every section is a
    smooth closed curve.
:func:`caret_waverider`
    Lofted through **polylines**, and exactly so: the caret's cross-section is
    a triangle at every station, so three points describe it without error.
    Splining it would round the leading edge, which is the one feature the
    shape exists to have.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from aether.geometry.brep import Loft, Revolve
from aether.geometry.edges import Arc, Contour, Segment, round_corners, rounded_contour

__all__ = [
    "blunted_multiconic",
    "caret_waverider",
    "spatular_wedge",
    "sphere_cone",
]

_FloatArray = NDArray[np.float64]



def _shoulder(
    station: float, radius_at_corner: float, half_angle: float,
    fillet: float, samples: int = 8,
) -> _FloatArray:
    r"""Meridian points rounding the corner where a cone meets its base disc.

    A base shoulder is the second sharpest feature on a re-entry body after the
    nose, and left as a mathematical corner it is unbuildable and a singular
    expansion to any solver that meets it.

    The fillet is tangent to the cone on one side and to the base plane on the
    other. Equal distance to both puts its centre at

    .. math::

        x_c = x_b - R, \qquad r_c = r_b - R\,\frac{1 + \sin\theta}{\cos\theta},

    and the arc then sweeps from :math:`\phi = \pi/2 + \theta`, where it meets
    the cone, round to :math:`\phi = 0`, where it meets the base plane
    travelling radially inward so the flat base closes it. Station increases
    monotonically across that sweep, which is what the profile requires.
    """
    if fillet <= 0.0:
        return np.zeros((0, 2))
    offset = fillet * (1.0 + np.sin(half_angle)) / np.cos(half_angle)
    centre_r = radius_at_corner - offset
    centre_x = station - fillet
    if centre_r <= 0.0:
        msg = (
            f"shoulder radius {fillet:g} is too large for a base radius of "
            f"{radius_at_corner:g} at a {np.rad2deg(half_angle):.1f} degree cone"
        )
        raise ValueError(msg)
    phi = np.linspace(0.5 * np.pi + half_angle, 0.0, max(3, int(samples)))
    return np.column_stack([centre_x + fillet * np.cos(phi),
                            centre_r + fillet * np.sin(phi)])


def _shoulder_arc(
    station: float, radius_at_corner: float, half_angle: float, fillet: float
) -> Arc:
    """The shoulder fillet of :func:`_shoulder`, as one exact arc.

    Same centre and same endpoints; what changes is that the revolved solid
    gets a single toroidal face instead of one conical face per sample.
    """
    if fillet <= 0.0:
        raise ValueError("shoulder fillet must be positive")
    offset = fillet * (1.0 + np.sin(half_angle)) / np.cos(half_angle)
    centre_r = radius_at_corner - offset
    centre_x = station - fillet
    if centre_r <= 0.0:
        msg = (
            f"shoulder radius {fillet:g} is too large for a base radius of "
            f"{radius_at_corner:g} at a {np.rad2deg(half_angle):.1f} degree cone"
        )
        raise ValueError(msg)
    centre = np.array([centre_x, centre_r])
    begin = 0.5 * np.pi + half_angle
    return Arc(
        start=centre + fillet * np.array([np.cos(begin), np.sin(begin)]),
        end=centre + fillet * np.array([1.0, 0.0]),
        centre=centre,
    )


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
    n_sections: int = 24,
    n_perimeter: int = 96,
    nose_station: float = 0.01,
    name: str = "spatular-wedge",
) -> Loft:
    """Power-law blended lifting body with Lamé cross-sections.

    Semi-axes and the super-ellipse exponent all vary as independent powers of
    the fractional station, which is what lets the section morph from a
    near-circular nose to a flattened base without a blending function:

    .. math::

        a(u) = R_{n,y} + (S - R_{n,y})u^{p_y}, \\quad
        b(u) = R_{n,z} + (T - R_{n,z})u^{p_z}, \\quad
        n(u) = n_0 + (n_1 - n_0)u^{p_n}

    ``nose_station`` is where the loft starts. It cannot be zero — the section
    there is a point and a loft needs a curve — and the flat cap it leaves is
    that section's own size, so a small value gives a small cap. A genuinely
    pointed body is a :func:`sphere_cone` and should be revolved instead.
    """
    if not (0.0 < nose_station < 1.0):
        msg = f"nose_station must lie in (0, 1), got {nose_station}"
        raise ValueError(msg)

    def section(u: float) -> _FloatArray:
        a = nose_radius_y + (base_half_span - nose_radius_y) * u**p_span
        b = nose_radius_z + (base_half_thickness - nose_radius_z) * u**p_thickness
        exponent = n_power_nose + (n_power_base - n_power_nose) * u**p_n_exp
        psi = np.linspace(0.0, 2.0 * np.pi, int(n_perimeter), endpoint=False)
        cos, sin = np.cos(psi), np.sin(psi)
        return np.column_stack([
            np.full(psi.size, u * length),
            a * np.abs(cos) ** (2.0 / exponent) * np.sign(cos),
            b * np.abs(sin) ** (2.0 / exponent) * np.sign(sin),
        ])

    # Clustered toward the nose, where both the semi-axes and the exponent
    # move fastest: u^0.5 has infinite slope at the origin, so uniform
    # stations under-resolve exactly the region that sets the stagnation flow.
    fraction = np.linspace(0.0, 1.0, int(n_sections)) ** 1.6
    return Loft(
        section=section,
        stations=nose_station + (1.0 - nose_station) * fraction,
        smooth=True,
        name=name,
    )


def caret_waverider(
    design_mach: float = 8.0,
    wedge_angle: float = np.deg2rad(6.0),
    length: float = 4.0,
    span_fraction: float = 0.55,
    leading_edge_radius: float = 0.0,
    ridge_radius: float = 0.0,
    edge_samples: int = 6,
    n_sections: int = 24,
    nose_station: float = 0.004,
    name: str = "caret-waverider",
) -> Loft:
    r"""Caret (Nonweiler) waverider, constructed **on-design** from its shock.

    A waverider is defined by the flow it rides, not by a set of dimensions.
    Given a design Mach number and the deflection of its compression surface,
    the oblique-shock relation fixes the shock angle :math:`\beta`, and the
    leading edges must lie **in that shock plane** — that is the entire point
    of the shape. The high pressure behind the shock is then trapped under the
    body instead of spilling around the edges, and the lift-to-drag ratio
    escapes the barrier that limits blunt bodies.

    Parameterising it by span and keel depth instead — as an earlier version of
    this function did, and as
    :func:`aether.aerodynamics.panels.caret_lifting_body` still does under a
    name that no longer claims otherwise — produces a caret-*shaped* body that is not
    a waverider.
    At the values that version defaulted to — 4 m long, 1.2 m semi-span, 0.32 m
    keel — the implied deflection is 4.57°, the Mach 8 shock angle is 10.49°,
    and the leading edges sit **0.74 m above the shock**. The flow spills over
    them, and the L/D that comes back is not representative of the class.

    Geometry
    --------

    With the apex at the origin and the freestream along :math:`+x`, at station
    :math:`x`:

    * the **shock plane** is :math:`z = -x\tan\beta`;
    * the **leading edges** lie in it, at :math:`(x, \pm x s \tan\beta, -x\tan\beta)`
      where ``span_fraction`` :math:`s` sets how wide the caret is opened;
    * the **ridge** — the compression surface's upper line — is at
      :math:`(x, 0, -x\tan\delta)`, above the shock because
      :math:`\delta < \beta`;
    * the **upper surface** is freestream-aligned, closing the section from
      each leading edge back to the ridge.

    Every section is a triangle, so ``smooth=False`` reproduces the ruled
    surface exactly. That matters: the leading edge is a crease, and splining
    through it would round the one feature the shape exists to have.

    Parameters
    ----------
    design_mach, wedge_angle:
        The design point. The shock angle is solved from them, so the body is
        on-design at ``design_mach`` and only there — which is true of real
        waveriders and is why off-design performance is the interesting
        question about them.
    span_fraction:
        Lateral opening of the caret as a fraction of the shock's depth. One
        gives leading edges at 45° in the base plane; smaller is a narrower,
        deeper caret.
    leading_edge_radius, ridge_radius:
        Blunting radii (m) for the two edge families, applied in each
        cross-section. Zero — the default — keeps the ideal sharp shape.

        They are separate because they do different jobs. The **leading edge**
        is where the shock attaches, and blunting it is a direct trade: a sharp
        edge carries infinite stagnation heating (:math:`q \propto 1/\sqrt R`),
        while a blunt one lets flow spill and moves the shock off the edge by
        roughly the radius. One to five millimetres is the range real vehicles
        live in, and it wants optimising rather than assuming. The **ridge** is
        interior to the shock layer, sees far milder conditions, and can
        usually be blunted more freely — it is separate so that it can be.

        The radius scales with the section, so it is applied at the base and
        tapered forward with the section itself; a constant-radius edge would
        need a radius larger than the section near the apex.
    edge_samples:
        Points per fillet arc.
    """
    from aether.aerodynamics.conical import wedge_shock_angle

    for label, value in (("length", length), ("design_mach", design_mach),
                         ("wedge_angle", wedge_angle), ("span_fraction", span_fraction)):
        if not (np.isfinite(value) and value > 0.0):
            msg = f"{label} must be finite and > 0, got {value}"
            raise ValueError(msg)
    if design_mach <= 1.0:
        msg = f"a waverider needs supersonic design flow, got Mach {design_mach}"
        raise ValueError(msg)

    beta = float(wedge_shock_angle(float(design_mach), float(wedge_angle)))
    if beta <= float(wedge_angle):
        msg = (
            f"shock angle {np.rad2deg(beta):.2f} deg does not exceed the wedge "
            f"angle {np.rad2deg(wedge_angle):.2f} deg; the shock has detached"
        )
        raise ValueError(msg)
    shock_slope = np.tan(beta)
    ridge_slope = np.tan(float(wedge_angle))

    for label, value in (("leading_edge_radius", leading_edge_radius),
                         ("ridge_radius", ridge_radius)):
        if not (np.isfinite(value) and value >= 0.0):
            msg = f"{label} must be finite and >= 0, got {value}"
            raise ValueError(msg)

    def section(u: float) -> _FloatArray:
        x = u * length
        shock_depth = x * shock_slope
        corners = np.array([
            [x, +span_fraction * shock_depth, -shock_depth],  # starboard edge, on the shock
            [x, -span_fraction * shock_depth, -shock_depth],  # port edge, on the shock
            [x, 0.0, -x * ridge_slope],                       # ridge, inside the shock layer
        ])
        if leading_edge_radius <= 0.0 and ridge_radius <= 0.0:
            return corners
        # Radii scale with the station: the section shrinks to a point at the
        # apex, and a constant radius would exceed it there.
        return round_corners(
            corners,
            u * np.array([leading_edge_radius, leading_edge_radius, ridge_radius]),
            samples=int(edge_samples),
            closed=True,
        )

    if not (0.0 < nose_station < 1.0):
        msg = f"nose_station must lie in (0, 1), got {nose_station}"
        raise ValueError(msg)
    # Started just off the apex: the section there is a point and a loft needs
    # a curve. Clustered forward, where the compression surface does its work.
    fraction = np.linspace(0.0, 1.0, int(n_sections)) ** 1.3
    stations = nose_station + (1.0 - nose_station) * fraction
    def section_contour(u: float) -> Contour:
        """The same section, as arcs and lines rather than chords."""
        x = u * length
        shock_depth = x * shock_slope
        corners = np.array([
            [x, +span_fraction * shock_depth, -shock_depth],
            [x, -span_fraction * shock_depth, -shock_depth],
            [x, 0.0, -x * ridge_slope],
        ])
        return rounded_contour(
            corners,
            u * np.array([leading_edge_radius, leading_edge_radius, ridge_radius]),
            closed=True,
        )

    sharp = leading_edge_radius <= 0.0 and ridge_radius <= 0.0
    return Loft(
        section=section,
        stations=stations,
        section_contour=None if sharp else section_contour,
        smooth=False,
        name=name,
    )


def sphere_cone(
    half_angle: float = np.deg2rad(10.0),
    nose_radius: float = 0.05,
    length: float = 2.0,
    shoulder_radius: float = 0.0,
    n_cap: int = 40,
    n_cone: int = 60,
    n_shoulder: int = 10,
    name: str = "sphere-cone",
) -> Revolve:
    """Spherically blunted cone, tangent at the shoulder.

    The cap runs to the tangency angle :math:`\\pi/2 - \\theta_c`, where the
    sphere's own slope already equals the cone's, so the join is
    :math:`C^1` by construction rather than by blending.
    """
    if not (0.0 < half_angle < 0.5 * np.pi):
        msg = f"half_angle must lie in (0, pi/2), got {half_angle}"
        raise ValueError(msg)

    phi = np.linspace(0.0, 0.5 * np.pi - half_angle, int(n_cap))
    cap_x = nose_radius * (1.0 - np.cos(phi))
    cap_r = nose_radius * np.sin(phi)

    x_tangent = float(cap_x[-1])
    r_tangent = float(cap_r[-1])
    if length <= x_tangent:
        msg = f"length {length} does not reach past the nose cap at {x_tangent:g}"
        raise ValueError(msg)
    cone_x = np.linspace(x_tangent, length, int(n_cone))[1:]
    cone_r = r_tangent + (cone_x - x_tangent) * np.tan(half_angle)

    station = np.concatenate([cap_x, cone_x])
    radius = np.concatenate([cap_r, cone_r])
    if shoulder_radius > 0.0:
        arc = _shoulder(float(station[-1]), float(radius[-1]), half_angle,
                        float(shoulder_radius), int(n_shoulder))
        keep = station < arc[0, 0]
        station = np.concatenate([station[keep], arc[:, 0]])
        radius = np.concatenate([radius[keep], arc[:, 1]])

    # The exact meridian: cap arc, cone, and the shoulder fillet if there is
    # one. The sampled arrays above are kept because they are the profile the
    # panel and sizing code reads; the contour is what the solid is built from.
    cap = Arc(
        start=np.array([0.0, 0.0]),
        end=np.array([x_tangent, r_tangent]),
        centre=np.array([nose_radius, 0.0]),
    )
    base_radius = r_tangent + (length - x_tangent) * np.tan(half_angle)
    primitives: tuple[Arc | Segment, ...]
    if shoulder_radius > 0.0:
        fillet = _shoulder_arc(length, base_radius, half_angle, float(shoulder_radius))
        primitives = (cap, Segment(cap.end, fillet.start), fillet)
    else:
        primitives = (cap, Segment(cap.end, np.array([length, base_radius])))
    return Revolve(
        station=station, radius=radius,
        contour=Contour(primitives, closed=False), name=name,
    )


def blunted_multiconic(
    nose_radius: float = 0.05,
    lengths: tuple[float, ...] = (1.0, 1.5),
    half_angles: tuple[float, ...] = (np.deg2rad(12.0), np.deg2rad(7.0)),
    fillet_radii: tuple[float, ...] = (0.1,),
    shoulder_radius: float = 0.0,
    n_cap: int = 40,
    n_segment: int = 40,
    n_fillet: int = 20,
    n_shoulder: int = 10,
    name: str = "blunted-multiconic",
) -> Revolve:
    """Biconic, triconic or n-conic with tangent fillets at every junction.

    The fillet tangent length is :math:`T = R\\tan(\\Delta/2)` for a deflection
    :math:`\\Delta` between consecutive cones — the standard circular-arc
    relation. The reciprocal, :math:`R/\\tan(\\Delta/2)`, is a cotangent that
    diverges as the cones become parallel, which is the usual case here since
    consecutive half-angles differ by a few degrees; at the default 12°/7°
    junction it asks for 2.29 m of tangent on a 1.0 m segment and folds the
    profile back through the nose.
    """
    if len(lengths) != len(half_angles):
        msg = f"got {len(lengths)} lengths for {len(half_angles)} half-angles"
        raise ValueError(msg)
    if len(fillet_radii) != len(lengths) - 1:
        msg = f"need one fillet radius per junction: {len(lengths) - 1}, got {len(fillet_radii)}"
        raise ValueError(msg)

    theta_0 = float(half_angles[0])
    phi = np.linspace(0.0, 0.5 * np.pi - theta_0, int(n_cap))
    station = list(nose_radius * (1.0 - np.cos(phi)))
    radius = list(nose_radius * np.sin(phi))

    # Built alongside the sampled profile: same geometry, exact primitives.
    cap = Arc(
        start=np.array([0.0, 0.0]),
        end=np.array([station[-1], radius[-1]]),
        centre=np.array([nose_radius, 0.0]),
    )
    prims: list[Arc | Segment] = [cap]
    here = cap.end

    x_current, r_current = station[-1], radius[-1]
    for index, (segment_length, theta) in enumerate(
        zip(lengths, half_angles, strict=True)
    ):
        theta = float(theta)
        if index == len(lengths) - 1:
            x_end = x_current + float(segment_length)
            r_end = r_current + float(segment_length) * np.tan(theta)
            station += list(np.linspace(x_current, x_end, int(n_segment))[1:])
            radius += list(np.linspace(r_current, r_end, int(n_segment))[1:])
            prims.append(Segment(here, np.array([x_end, r_end])))
            here = np.array([x_end, r_end])
            continue

        theta_next = float(half_angles[index + 1])
        fillet = float(fillet_radii[index])
        half_delta = abs(theta - theta_next) / 2.0
        tangent = fillet * np.tan(half_delta) if half_delta > 1e-6 else 0.0

        x_corner = x_current + float(segment_length)
        r_corner = r_current + float(segment_length) * np.tan(theta)
        x_stop = x_corner - tangent * np.cos(theta)
        r_stop = r_corner - tangent * np.sin(theta)
        station += list(np.linspace(x_current, x_stop, int(n_segment))[1:])
        radius += list(np.linspace(r_current, r_stop, int(n_segment))[1:])

        centre_x = x_stop + fillet * np.sin(theta)
        centre_r = r_stop - fillet * np.cos(theta)
        angles = np.linspace(0.5 * np.pi - theta, 0.5 * np.pi - theta_next,
                             max(4, int(n_fillet)))[1:]
        station += list(centre_x - fillet * np.cos(angles))
        radius += list(centre_r + fillet * np.sin(angles))

        x_current = x_corner + tangent * np.cos(theta_next)
        r_current = r_corner + tangent * np.sin(theta_next)

        stop = np.array([x_stop, r_stop])
        if float(np.linalg.norm(stop - here)) > 1e-12:
            prims.append(Segment(here, stop))
        here = np.array([x_current, r_current])
        if tangent > 0.0:
            prims.append(Arc(stop, here, np.array([centre_x, centre_r])))

    station_array = np.asarray(station, dtype=np.float64)
    radius_array = np.asarray(radius, dtype=np.float64)
    if shoulder_radius > 0.0:
        arc = _shoulder(float(station_array[-1]), float(radius_array[-1]),
                        float(half_angles[-1]), float(shoulder_radius), int(n_shoulder))
        keep = station_array < arc[0, 0]
        station_array = np.concatenate([station_array[keep], arc[:, 0]])
        radius_array = np.concatenate([radius_array[keep], arc[:, 1]])
        shoulder = _shoulder_arc(float(here[0]), float(here[1]),
                                 float(half_angles[-1]), float(shoulder_radius))
        # Trim the final cone back to where the shoulder picks it up.
        trimmed = prims[-1]
        if isinstance(trimmed, Segment):
            prims[-1] = Segment(trimmed.start, shoulder.start)
        prims.append(shoulder)
    return Revolve(
        station=station_array, radius=radius_array,
        contour=Contour(tuple(prims), closed=False), name=name,
    )
