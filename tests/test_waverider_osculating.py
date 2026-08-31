"""Osculating-cone waverider design.

A waverider is defined by the flow it rides, so the tests ask whether it rides
it. Two checks carry most of the weight:

* The **leading edge lies on the shock**. That is the whole definition, and it
  is checkable exactly because the local shock is a cone of known angle about
  a known axis.
* With a **circular** shockwave profile curve the curvature is constant, every
  osculating plane shares one cone, and the design must therefore collapse to
  the cone-derived waverider -- a single stream surface, sampled many times.
  That is an independent answer for a method that otherwise has none.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from aether.geometry.waverider import (
    _curvature,
    circular_shock_curve,
    osculating_cone_waverider,
    power_shock_curve,
)

_MACH, _BETA, _LENGTH = 8.0, np.radians(14.0), 4.0


@pytest.fixture(scope="module")
def cone_derived():
    """The circular-SPC design: constant curvature, so one cone."""
    return osculating_cone_waverider(design_mach=_MACH, shock_angle=_BETA, length=_LENGTH)


@pytest.fixture(scope="module")
def variable():
    """A power-law SPC, whose curvature genuinely varies along the span."""
    return osculating_cone_waverider(
        design_mach=_MACH,
        shock_angle=_BETA,
        length=_LENGTH,
        half_span=0.55,
        shock_curve=power_shock_curve(0.35, 0.55, 2.6),
    )


def _axes(design):
    """Local cone axes and apex stations, rebuilt from the design's own SPC."""
    y, z = design.shock_points[:, 1], design.shock_points[:, 2]
    radius, normal = _curvature(y, z)
    centre = np.column_stack([y, z]) + radius[:, None] * normal
    return centre, design.length - radius / np.tan(design.shock_angle), radius


# ------------------------------------------------------------- the curvature


def test_a_circular_arc_has_exactly_constant_curvature() -> None:
    """The osculating circle through three points, not a second derivative.

    Differentiating the sampled curve twice falls back to one-sided
    differences at the span ends, which on a circular arc reported a radius
    varying by a factor of two -- so every osculating plane near the tip got
    the wrong cone.
    """
    y = np.linspace(-0.6, 0.6, 41)
    radius, _ = _curvature(y, circular_shock_curve(1.0)(y))
    assert radius == pytest.approx(np.ones_like(radius), rel=1e-9)


def test_the_normal_points_to_the_concave_side() -> None:
    """The cone's axis is where the shock curves toward, not away from."""
    y = np.linspace(-0.6, 0.6, 41)
    _, normal = _curvature(y, circular_shock_curve(1.0)(y))
    assert normal[20] == pytest.approx([0.0, 1.0], abs=1e-9)
    assert np.all(normal[:, 1] > 0.0)


# ------------------------------------------------------------- riding a shock


def test_the_leading_edge_lies_on_the_shock(cone_derived) -> None:
    """The definition of a waverider, measured.

    In each osculating plane the shock is a cone of half-angle beta about the
    local axis, so an edge point on it satisfies
    :math:`r/(x - x_{apex}) = \\tan\\beta` exactly.
    """
    centre, apex, _ = _axes(cone_derived)
    edge = cone_derived.leading_edge
    radial = np.linalg.norm(edge[:, 1:] - centre, axis=1)
    assert radial / (edge[:, 0] - apex) == pytest.approx(np.tan(_BETA), rel=1e-8)


def test_the_leading_edge_lies_on_the_shock_for_a_curved_one(variable) -> None:
    """And it must still hold when every plane has a different cone."""
    centre, apex, _ = _axes(variable)
    edge = variable.leading_edge
    radial = np.linalg.norm(edge[:, 1:] - centre, axis=1)
    assert radial / (edge[:, 0] - apex) == pytest.approx(np.tan(_BETA), rel=1e-6)


def test_a_circular_shock_curve_collapses_to_a_single_cone(cone_derived) -> None:
    """Constant curvature means one cone, so one stream surface.

    Every osculating plane's compression surface, measured from its own axis,
    must be the same curve. This is the closest thing the method has to an
    independent answer.
    """
    centre, _apex, radius = _axes(cone_derived)
    assert np.ptp(radius) / radius.mean() < 1e-9

    radial = np.linalg.norm(cone_derived.lower[..., 1:] - centre[None, :, :], axis=2)
    spread = np.ptp(radial, axis=1) / np.maximum(radial.mean(axis=1), 1e-12)
    assert spread.max() < 1e-9


def test_a_power_law_shock_curve_really_varies(variable) -> None:
    """Otherwise it exercises nothing the circular case does not."""
    radius = variable.osculating_radius
    assert radius.max() / radius.min() > 1.5


def test_the_upper_surface_is_aligned_with_the_freestream(variable) -> None:
    """It must generate no compression, or the shock is not the design shock."""
    lateral = variable.upper[..., 1:]
    assert np.ptp(lateral, axis=0) == pytest.approx(np.zeros_like(lateral[0]), abs=1e-12)


def test_the_design_is_symmetric_about_the_centreline(variable) -> None:
    assert variable.lower[:, :, 1] == pytest.approx(-variable.lower[:, ::-1, 1], abs=1e-12)
    assert variable.lower[:, :, 2] == pytest.approx(variable.lower[:, ::-1, 2], abs=1e-12)


# ------------------------------------------------------------------ the solid


@pytest.mark.parametrize("name", ["cone_derived", "variable"])
def test_the_design_closes_into_an_oriented_solid(name, request) -> None:
    """Watertight, outward, and consistently wound -- three separate things.

    A ring net gets the first and fails the others: a waverider's first
    station is its leading edge, where the two surfaces meet rather than
    closing a loop, so welding it leaves edges shared by three faces. These
    count the edges instead of trusting the flag.
    """
    mesh = request.getfixturevalue(name).to_mesh()

    undirected: Counter = Counter()
    directed: Counter = Counter()
    for triangle in mesh.faces:
        corners = [int(v) for v in triangle]
        for a, b in zip(corners, corners[1:] + corners[:1], strict=True):
            undirected[tuple(sorted((a, b)))] += 1
            directed[(a, b)] += 1

    assert all(count == 2 for count in undirected.values()), "not a closed manifold"
    assert all(count == 1 for count in directed.values()), "inconsistent winding"
    assert mesh.is_closed

    triangles = mesh.triangles
    volume = float(
        np.einsum(
            "ij,ij->i",
            triangles[:, 0],
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        ).sum()
        / 6.0
    )
    assert volume > 0.0, "normals wound inward"


# ------------------------------------------------------------------ refusals


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.2, 1.5])
def test_a_degenerate_capture_fraction_is_refused(fraction: float) -> None:
    with pytest.raises(ValueError, match="capture_fraction"):
        osculating_cone_waverider(capture_fraction=fraction)


def test_a_shock_the_flow_cannot_support_is_refused() -> None:
    """Below the Mach angle there is no shock; above detachment, no cone."""
    with pytest.raises(ValueError, match=r"Mach angle|attached weak branch"):
        osculating_cone_waverider(design_mach=8.0, shock_angle=np.radians(3.0))
    with pytest.raises(ValueError, match="attached weak branch"):
        osculating_cone_waverider(design_mach=8.0, shock_angle=np.radians(85.0))


def test_a_span_wider_than_its_own_arc_is_refused() -> None:
    with pytest.raises(ValueError, match="must stay inside the arc"):
        osculating_cone_waverider(design_mach=8.0, shock_angle=_BETA, length=_LENGTH, half_span=5.0)
