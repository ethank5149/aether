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
        shock_curve=power_shock_curve(0.35, 0.55, 2.0),
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


def test_a_shock_curve_flat_at_the_centreline_is_refused() -> None:
    """The design degenerates there, quietly, and still produces a body.

    For :math:`z_s = d(|y|/b)^n` the second derivative goes as
    :math:`|y|^{n-2}`, so above :math:`n = 2` the centreline curvature is zero
    and the osculating radius unbounded. The local apex then runs away
    upstream and the centreline chord grows with every refinement of the span:
    the body still closes and still meshes, it just is not a convergent
    design. At :math:`n = 2.6` the apex sat 6.5 m ahead of a 4 m body.
    """
    for exponent in (2.6, 3.0, 4.0):
        with pytest.raises(ValueError, match="osculating radius is unbounded"):
            power_shock_curve(0.35, 0.55, exponent)
    for exponent in (0.0, -1.0):
        with pytest.raises(ValueError, match="exponent must lie"):
            power_shock_curve(0.35, 0.55, exponent)


def test_the_parabola_keeps_the_body_downstream_of_its_apex(variable) -> None:
    """The consequence of that bound, at the default the study now uses.

    Every leading-edge station is downstream of the nose, and the osculating
    radius varies by a bounded factor rather than an unbounded one -- which is
    what keeps the chord, and therefore the cell aspect ratio, in hand.
    """
    assert variable.leading_edge[:, 0].min() > 0.0
    radius = variable.osculating_radius
    assert radius.max() / radius.min() < 6.0


# ------------------------------------------------------- closing the body


def test_the_base_is_banded_rather_than_bridged(cone_derived) -> None:
    """A single strip across the base guarantees slivers, whatever the shape.

    The base spans the body's whole thickness -- half a metre on the default
    design -- while neighbouring planes are forty millimetres apart, so one
    row of triangles across it is thirteen to one before any geometry is
    considered. That, not the compression surface, was where the worst cell on
    this body lived.
    """
    mesh = cone_derived.to_mesh()
    base = np.abs(mesh.vertices[:, 0] - cone_derived.length) < 1e-9
    corners = mesh.faces[base[mesh.faces].all(axis=1)]
    assert len(corners) > 0

    triangles = mesh.vertices[corners]
    edges = np.stack(
        [
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        ],
        axis=1,
    )
    assert (edges.max(axis=1) / edges.min(axis=1)).max() < 8.0


def test_the_thickness_banding_can_be_set_and_changes_the_count(cone_derived) -> None:
    coarse = cone_derived.to_mesh(thickness_divisions=1)
    fine = cone_derived.to_mesh(thickness_divisions=8)
    assert len(fine.faces) > len(coarse.faces)
    for mesh in (coarse, fine):
        assert mesh.is_closed


def test_banding_the_base_does_not_move_the_body(cone_derived) -> None:
    """More faces across the base, the same solid: the volume is the check."""
    volumes = []
    for divisions in (1, 4, 12):
        mesh = cone_derived.to_mesh(thickness_divisions=divisions)
        triangles = mesh.triangles
        volumes.append(
            float(
                np.einsum(
                    "ij,ij->i",
                    triangles[:, 0],
                    np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
                ).sum()
                / 6.0
            )
        )
    assert volumes[0] == pytest.approx(volumes[1], rel=1e-9)
    assert volumes[0] == pytest.approx(volumes[2], rel=1e-9)


def test_the_shell_is_oriented_by_propagation_not_by_guesswork(variable) -> None:
    """Four patches meet here and none of them agrees on winding by luck.

    Flipping each by hand is guesswork that must be redone whenever a patch is
    added, and one wrong guess leaves a shell that still closes and reports a
    nonsense volume -- which is what happened: 242 directed edges traversed
    twice, and a volume three times the truth.
    """
    mesh = variable.to_mesh()
    directed: Counter = Counter()
    for triangle in mesh.faces:
        corners = [int(v) for v in triangle]
        for a, b in zip(corners, corners[1:] + corners[:1], strict=True):
            directed[(a, b)] += 1
    assert all(count == 1 for count in directed.values())


def test_a_face_that_collapses_in_the_weld_is_dropped(variable) -> None:
    """Degeneracy has to be removed after welding, not before.

    A patch whose corners differ by less than the weld tolerance has area as
    coordinates and a repeated vertex as indices. Dropping on area first
    leaves a face that collapses a moment later, and a face with a repeated
    corner is a self-edge -- non-manifold at exactly the point the shell looks
    closed. The span tip's leading edge is where that happens: the two
    surfaces meet and the panel between them has nowhere to go.
    """
    faces = variable.to_mesh().faces
    assert np.all(faces[:, 0] != faces[:, 1])
    assert np.all(faces[:, 1] != faces[:, 2])
    assert np.all(faces[:, 2] != faces[:, 0])


def test_the_chord_follows_the_osculating_radius(variable) -> None:
    """Why this shape is intrinsically harder to mesh evenly than a cone.

    :math:`\\text{chord} = (1-f)\\,R/\\tan\\beta`, so a shock curve whose
    curvature varies *must* give a chord that varies with it -- the cell size
    along the body follows the design, not the mesher. Only a circular shock
    curve, which is the cone-derived case, has a constant chord.
    """
    chord = variable.length - variable.leading_edge[:, 0]
    radius = variable.osculating_radius
    ratio = chord / radius
    assert np.allclose(ratio, ratio[0], rtol=1e-6)
    assert chord.max() / chord.min() == pytest.approx(radius.max() / radius.min(), rel=1e-6)
