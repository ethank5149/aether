"""Meridian profiles: the shape maths, in one place.

A body of revolution is a curve in the meridian plane, and everything else --
panels for impact theory, an exact solid for meshing, a sizing profile for the
axisymmetric CFD path -- is a *representation* of that curve. The curve itself
should therefore exist once.

The multiconic makes the point sharply. Its junctions are blended by fillets,
and the solid needs those as exact arcs while the panel method needs them
sampled. That is not two curves; it is one curve and two things to do with it,
so :func:`multiconic_meridian` returns the sampling *and* the junction
geometry the arcs are built from.

For a while it did not. :func:`aether.aerodynamics.panels.sphere_cone` and
:func:`aether.geometry.bodies.sphere_cone` each derived the blunted-cone
meridian from scratch, with the same algebra written twice and sampled by two
different conventions, and the same was true of the multiconic. Two
consequences, and the second is the expensive one: a correction to the
geometry had to be made in both places to take effect, and "the sphere-cone"
meant different things depending on which module you imported it from.

This module holds the curves. It imports nothing from
:mod:`aether.aerodynamics`, so the aerodynamic and geometric layers can both
depend on it without either depending on the other.

Sampling convention
-------------------

The functions here count **intervals**, not points, because that is the
convention under which the two historical samplings coincide: a caller
wanting ``n`` points along a segment asks for ``n - 1`` intervals. Every
profile starts at the nose and its stations increase strictly, which is what
:class:`~aether.geometry.brep.Revolve` requires and what the panel sweep
assumes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.optimize
from numpy.typing import NDArray

_FloatArray = NDArray[np.float64]

__all__ = [
    "MulticonicJunction",
    "multiconic_meridian",
    "sphere_cone_closure",
    "sphere_cone_meridian",
    "sphere_cone_tangency",
]


def sphere_cone_closure(
    length: float | None = None,
    base_radius: float | None = None,
    nose_radius: float | None = None,
    half_angle: float | None = None,
) -> tuple[float, float, float, float]:
    """Close the sphere--cone geometry from any three of its four parameters.

    A spherical nose cap tangent to a conical frustum has one relation among
    its four defining lengths,

    .. math::

        L = R_b\\cot\\theta_c - R_n\\csc\\theta_c + R_n ,

    obtained by walking the axis from nose to base through the tangency
    station. Supplying three parameters therefore determines the fourth, and
    this function solves for whichever is left out. Over-specifying is
    rejected rather than silently resolved: a mesh built from four mutually
    inconsistent numbers is not the shape any of them describe.
    """
    given = [length, base_radius, nose_radius, half_angle]
    if sum(value is not None for value in given) != 3:
        raise ValueError(
            "exactly three of (length, base_radius, nose_radius, half_angle) "
            f"must be given, got {sum(v is not None for v in given)}"
        )

    if length is None:
        assert base_radius is not None and nose_radius is not None
        assert half_angle is not None
        cot, csc = 1.0 / np.tan(half_angle), 1.0 / np.sin(half_angle)
        out_length = base_radius * cot - nose_radius * csc + nose_radius
        out_base, out_nose, out_angle = base_radius, nose_radius, half_angle
    elif base_radius is None:
        assert nose_radius is not None and half_angle is not None
        cot, csc = 1.0 / np.tan(half_angle), 1.0 / np.sin(half_angle)
        out_base = (length + nose_radius * csc - nose_radius) / cot
        out_length, out_nose, out_angle = length, nose_radius, half_angle
    elif nose_radius is None:
        assert base_radius is not None and half_angle is not None
        cot, csc = 1.0 / np.tan(half_angle), 1.0 / np.sin(half_angle)
        out_nose = (base_radius * cot - length) / (csc - 1.0)
        out_length, out_base, out_angle = length, base_radius, half_angle
    else:
        span_base, span_nose, span_len = base_radius, nose_radius, length

        def residual(theta: float) -> float:
            return float(
                span_base / np.tan(theta) - span_nose / np.sin(theta) + span_nose - span_len
            )

        lo, hi = np.radians(0.05), np.radians(89.0)
        if residual(lo) * residual(hi) > 0.0:
            raise ValueError("no half-angle closes this (length, base_radius, nose_radius)")
        out_angle = float(scipy.optimize.brentq(residual, lo, hi, xtol=1e-14))
        out_length, out_base, out_nose = length, base_radius, nose_radius

    if not (np.isfinite(out_length) and out_length > 0.0):
        raise ValueError(f"closure gave a non-physical length {out_length}")
    if not (np.isfinite(out_nose) and 0.0 < out_nose <= out_base):
        raise ValueError(
            f"closure gave a non-physical nose radius {out_nose}; the nose "
            f"cannot be blunter than the base ({out_base})"
        )
    return float(out_length), float(out_base), float(out_nose), float(out_angle)


def sphere_cone_tangency(nose_radius: float, half_angle: float) -> tuple[float, float]:
    """``(x, r)`` where the nose cap meets the cone.

    The cap runs to :math:`\\phi = \\pi/2 - \\theta_c`, where the sphere's own
    slope already equals the cone's, so the join is :math:`C^1` by
    construction rather than by blending.
    """
    return (
        float(nose_radius * (1.0 - np.sin(half_angle))),
        float(nose_radius * np.cos(half_angle)),
    )


def sphere_cone_meridian(
    length: float,
    nose_radius: float,
    half_angle: float,
    cap_intervals: int,
    cone_intervals: int,
) -> tuple[_FloatArray, _FloatArray]:
    """The blunted-cone meridian, nose first. Returns ``(station, radius)``.

    Cap nodes are equally spaced in the polar angle of the sphere and cone
    nodes equally spaced along the axis, which is what both historical
    samplings did; passing ``n - 1`` intervals reproduces either of them
    exactly.
    """
    if cap_intervals < 1 or cone_intervals < 1:
        raise ValueError(
            f"need at least one interval on each segment, got cap={cap_intervals}, "
            f"cone={cone_intervals}"
        )
    x_tangent, r_tangent = sphere_cone_tangency(nose_radius, half_angle)
    if length <= x_tangent:
        raise ValueError(
            f"length {length} does not reach past the nose cap, which ends at {x_tangent:g}"
        )

    phi_tangent = 0.5 * np.pi - half_angle
    phi = phi_tangent * np.arange(cap_intervals + 1) / cap_intervals
    cap_x = nose_radius * (1.0 - np.cos(phi))
    cap_r = nose_radius * np.sin(phi)

    span = np.arange(1, cone_intervals + 1) / cone_intervals
    cone_x = x_tangent + span * (length - x_tangent)
    cone_r = r_tangent + span * (length - x_tangent) * np.tan(half_angle)

    return np.concatenate([cap_x, cone_x]), np.concatenate([cap_r, cone_r])


@dataclass(frozen=True)
class MulticonicJunction:
    """Where one cone hands over to the next, through a tangent fillet.

    The sampled profile passes through these points, and the exact solid is
    built from them -- a straight segment up to :attr:`stop`, an arc about
    :attr:`centre` to :attr:`resume`. Returning them is what lets the two
    representations share a derivation instead of each computing the blend
    from the same formula and hoping.
    """

    stop: _FloatArray
    """``(x, r)`` where the frustum ends and the fillet begins."""

    centre: _FloatArray
    """``(x, r)`` centre of the fillet arc."""

    resume: _FloatArray
    """``(x, r)`` where the fillet ends and the next frustum begins."""

    radius: float
    """Fillet radius; zero when the two cones are parallel and there is no blend."""


def multiconic_meridian(
    nose_radius: float,
    lengths: Sequence[float],
    half_angles: Sequence[float],
    fillet_radii: Sequence[float],
    cap_intervals: int,
    segment_intervals: int,
    fillet_intervals: int,
) -> tuple[_FloatArray, _FloatArray, tuple[MulticonicJunction, ...]]:
    """A blunted cone of one or more segments, blended at each junction.

    ``lengths`` are measured **along the axis from the previous segment's
    start**, not as absolute stations, and the first is measured from the cap
    tangency. Junction fillets are tangent to both cones, so the profile is
    :math:`C^1` everywhere.

    The tangent length
    ------------------

    A circular arc of radius :math:`R_f` blending two lines that meet at
    deflection :math:`\\delta` stands off the corner by
    :math:`T = R_f\\tan(\\delta/2)`. The reciprocal form,
    :math:`R_f/\\tan(\\delta/2)`, is the one that looks equally plausible and
    is wrong: it *diverges* as the cones become parallel, which is the usual
    case, since consecutive cone angles differ by a few degrees. At the
    twelve-to-seven-degree junction these functions default to, it returns
    2.29 m of tangent for a 1.0 m segment, so both tangency points land
    outside their own frusta and the profile folds back through the nose.
    That fold is invisible to a panel integration, which sums unordered
    faces, and fatal to a mesh generator, which sees overlapping facets.
    """
    lengths = list(lengths)
    half_angles = list(half_angles)
    fillet_radii = list(fillet_radii)
    if len(lengths) != len(half_angles):
        raise ValueError(f"got {len(lengths)} lengths for {len(half_angles)} half-angles")
    if len(fillet_radii) != len(lengths) - 1:
        raise ValueError(
            f"need one fillet radius per junction: {len(lengths) - 1}, got {len(fillet_radii)}"
        )
    if not lengths:
        raise ValueError("a multiconic needs at least one segment")

    phi_tangent = 0.5 * np.pi - half_angles[0]
    phi = np.linspace(0.0, phi_tangent, cap_intervals + 1)
    station = [nose_radius * (1.0 - np.cos(phi))]
    radius = [nose_radius * np.sin(phi)]

    here = np.array(sphere_cone_tangency(nose_radius, half_angles[0]))
    junctions: list[MulticonicJunction] = []

    for index, (length, angle) in enumerate(zip(lengths, half_angles, strict=True)):
        corner = here + float(length) * np.array([1.0, np.tan(angle)])
        if index == len(lengths) - 1:
            station.append(np.linspace(here[0], corner[0], segment_intervals + 1)[1:])
            radius.append(np.linspace(here[1], corner[1], segment_intervals + 1)[1:])
            here = corner
            continue

        following = half_angles[index + 1]
        fillet = float(fillet_radii[index])
        half_delta = abs(angle - following) / 2.0
        tangent = fillet * np.tan(half_delta) if half_delta > 1e-6 else 0.0

        stop = corner - tangent * np.array([np.cos(angle), np.sin(angle)])
        centre = stop + fillet * np.array([np.sin(angle), -np.cos(angle)])
        resume = corner + tangent * np.array([np.cos(following), np.sin(following)])

        station.append(np.linspace(here[0], stop[0], segment_intervals + 1)[1:])
        radius.append(np.linspace(here[1], stop[1], segment_intervals + 1)[1:])

        sweep = np.linspace(
            0.5 * np.pi - angle, 0.5 * np.pi - following, max(4, fillet_intervals + 1)
        )[1:]
        station.append(centre[0] - fillet * np.cos(sweep))
        radius.append(centre[1] + fillet * np.sin(sweep))

        junctions.append(
            MulticonicJunction(stop=stop, centre=centre, resume=resume, radius=tangent and fillet)
        )
        here = resume

    return np.concatenate(station), np.concatenate(radius), tuple(junctions)
