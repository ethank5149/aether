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

__all__ = [
    "blunted_multiconic",
    "caret_waverider",
    "spatular_wedge",
    "sphere_cone",
]

_FloatArray = NDArray[np.float64]


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
    length: float = 4.0,
    semi_span: float = 1.2,
    keel_depth: float = 0.32,
    n_sections: int = 24,
    nose_station: float = 0.004,
    name: str = "caret-waverider",
) -> Loft:
    """Caret waverider — exact, because its sections are triangles.

    The lower surface is a straight dihedral V and the upper surface a
    freestream-aligned plate, so the cross-section at every station is the
    triangle joining the two leading edges and the keel. Three points describe
    it with no error at all, and lofting them with ``smooth=False`` reproduces
    the ruled surface exactly rather than approximating it.

    That exactness is the point. A waverider rides its own attached shock, and
    the leading edge has to be a crease for the high pressure underneath to be
    trapped; rounding it — which any spline through these three points would
    do — is not a small geometric error but a different vehicle.
    """
    for label, value in (("length", length), ("semi_span", semi_span),
                         ("keel_depth", keel_depth)):
        if not (np.isfinite(value) and value > 0.0):
            msg = f"{label} must be finite and > 0, got {value}"
            raise ValueError(msg)

    def section(u: float) -> _FloatArray:
        span = u * semi_span
        depth = u * keel_depth
        x = u * length
        return np.array([
            [x, +span, 0.0],   # starboard leading edge
            [x, -span, 0.0],   # port leading edge
            [x, 0.0, -depth],  # keel
        ])

    # Started just off the apex rather than at it: the section there is a
    # point and a loft needs a curve. Left at 1/n the truncation costs 5 % of
    # the length, which on a waverider is 5 % of the compression surface.
    if not (0.0 < nose_station < 1.0):
        msg = f"nose_station must lie in (0, 1), got {nose_station}"
        raise ValueError(msg)
    fraction = np.linspace(0.0, 1.0, int(n_sections)) ** 1.3
    stations = nose_station + (1.0 - nose_station) * fraction
    return Loft(section=section, stations=stations, smooth=False, name=name)


def sphere_cone(
    half_angle: float = np.deg2rad(10.0),
    nose_radius: float = 0.05,
    length: float = 2.0,
    n_cap: int = 40,
    n_cone: int = 60,
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

    return Revolve(
        station=np.concatenate([cap_x, cone_x]),
        radius=np.concatenate([cap_r, cone_r]),
        name=name,
    )


def blunted_multiconic(
    nose_radius: float = 0.05,
    lengths: tuple[float, ...] = (1.0, 1.5),
    half_angles: tuple[float, ...] = (np.deg2rad(12.0), np.deg2rad(7.0)),
    fillet_radii: tuple[float, ...] = (0.1,),
    n_cap: int = 40,
    n_segment: int = 40,
    n_fillet: int = 20,
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

    return Revolve(
        station=np.asarray(station, dtype=np.float64),
        radius=np.asarray(radius, dtype=np.float64),
        name=name,
    )
