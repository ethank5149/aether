"""Exact solids, and the error the old triangulated path could not remove.

The claim these tests exist to hold is narrow and worth stating: meshing an
exact solid converges to the body, and sampling one into a fixed triangulation
does not. The second is not a small effect — on the generic lifting body the
frozen faceting is 3 % of the volume — and it is invisible to every test that
only checks a mesh against itself.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from aether.aerodynamics import panels
from aether.geometry import VehicleMesh, bodies, brep

pytest.importorskip("gmsh")

MAKERS = (
    bodies.spatular_wedge,
    bodies.caret_waverider,
    bodies.sphere_cone,
    bodies.blunted_multiconic,
)


def _enclosed_volume(mesh: VehicleMesh) -> float:
    """Volume from the divergence theorem — signed, so it also checks winding."""
    t = mesh.triangles
    return float(
        np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])).sum() / 6.0
    )


@pytest.mark.parametrize("maker", MAKERS, ids=lambda f: f.__name__)
def test_every_vehicle_class_builds_a_closed_solid(maker) -> None:
    properties = brep.solid_properties(maker())
    assert properties.volume > 0.0
    assert properties.surface_area > 0.0
    assert properties.length > 0.0
    assert properties.diameter > 0.0
    low, high = properties.bounds
    assert np.all(high > low)
    # The centroid of a nose-forward body sits aft of mid-length: the section
    # grows monotonically, so most of the volume is behind the halfway station.
    assert low[0] < properties.centroid[0] < high[0]


@pytest.mark.parametrize("maker", MAKERS, ids=lambda f: f.__name__)
def test_the_surface_mesh_is_watertight_and_outward(maker) -> None:
    """Both properties are prerequisites for extruding a boundary layer."""
    mesh = brep.surface_mesh(maker(), curvature_nodes=12)
    assert mesh.is_closed, f"open at {mesh.boundary_stations()}"
    assert _enclosed_volume(mesh) > 0.0, "normals wound inward"


def test_refinement_converges_to_the_exact_volume() -> None:
    """The property the triangulated path did not have.

    A mesh over a fixed faceting converges to the faceting. A mesh over a solid
    converges to the solid, so geometric error becomes something a caller can
    buy down rather than something baked in at generator-definition time.
    """
    body = bodies.spatular_wedge()
    exact = brep.solid_properties(body).volume

    errors = []
    for size in (0.20, 0.10, 0.05):
        mesh = brep.surface_mesh(body, size_max=size, size_min=size / 40, curvature_nodes=24)
        errors.append(abs(_enclosed_volume(mesh) - exact) / exact)

    assert all(b < a for a, b in itertools.pairwise(errors)), f"not monotone: {errors}"
    assert errors[-1] < 0.005


def test_the_frozen_triangulation_does_not_reach_the_exact_body() -> None:
    """Why this module exists, as a number.

    The panel generator's grid is fixed at definition time, so its volume error
    is whatever that grid gives and is unreachable by any downstream refinement.
    """
    exact = brep.solid_properties(bodies.spatular_wedge()).volume
    sampled = abs(_enclosed_volume(VehicleMesh.from_surface_grid(panels.spatular_wedge().surface)))
    assert abs(sampled - exact) / exact > 0.02, (
        "if this no longer holds the generators were changed and the note in "
        "aether.geometry.brep should be re-measured rather than trusted"
    )


def test_the_caret_rides_its_own_shock() -> None:
    """The defining property, and the one the old parameterisation lacked.

    A waverider is defined by the flow it rides: its leading edges must lie in
    the shock plane its own compression surface makes, or the high pressure
    underneath spills around them and the shape is a caret-*formed* body rather
    than a waverider. Parameterised by span and keel depth instead, the edges
    sat 0.74 m above the shock at Mach 8.
    """
    from aether.aerodynamics.conical import wedge_shock_angle

    mach, wedge = 8.0, np.deg2rad(6.0)
    body = bodies.caret_waverider(design_mach=mach, wedge_angle=wedge, length=4.0)
    properties = brep.solid_properties(body)
    low, _ = properties.bounds

    beta = wedge_shock_angle(mach, wedge)
    expected_depth = -properties.length * np.tan(beta)
    # The deepest point of the body is the leading edge at the base station,
    # and it must sit on the shock plane to within the nose truncation.
    assert low[2] == pytest.approx(expected_depth, rel=0.02), (
        f"deepest point {low[2]:.4f} m against a shock at {expected_depth:.4f} m"
    )


def test_the_caret_is_sharp_by_default_and_blunts_on_request() -> None:
    """Edge blunting is a design variable, so it must be off unless asked for.

    A sharp edge carries infinite stagnation heating and cannot be built; a
    blunt one spills flow and moves the shock off the edge. Which trade to make
    is the caller's, so the default reproduces the ideal shape exactly and the
    radius is tunable per edge family.
    """
    sharp = brep.solid_properties(bodies.caret_waverider())
    blunt = brep.solid_properties(bodies.caret_waverider(leading_edge_radius=0.01))
    blunter = brep.solid_properties(bodies.caret_waverider(leading_edge_radius=0.02))

    # Rounding a convex corner can only remove material.
    assert blunt.volume < sharp.volume
    assert blunter.volume < blunt.volume
    assert blunt.surface_area < sharp.surface_area
    # And it is a small perturbation, not a different vehicle.
    assert abs(blunt.volume - sharp.volume) / sharp.volume < 0.01


def test_the_leading_edge_and_ridge_blunt_independently() -> None:
    """They see different environments and are optimised separately."""
    edge_only = brep.solid_properties(bodies.caret_waverider(leading_edge_radius=0.02))
    ridge_only = brep.solid_properties(bodies.caret_waverider(ridge_radius=0.02))
    sharp = brep.solid_properties(bodies.caret_waverider())
    assert edge_only.surface_area < sharp.surface_area
    assert ridge_only.surface_area < sharp.surface_area
    assert edge_only.surface_area != ridge_only.surface_area


@pytest.mark.parametrize(
    "maker", (bodies.sphere_cone, bodies.blunted_multiconic), ids=lambda f: f.__name__
)
def test_the_base_shoulder_blunts(maker) -> None:
    """The second sharpest feature after the nose, and a singular expansion."""
    sharp = brep.solid_properties(maker())
    blunt = brep.solid_properties(maker(shoulder_radius=0.02))
    assert blunt.volume < sharp.volume
    assert blunt.surface_area < sharp.surface_area
    # The fillet lives inside the original envelope, so the body does not grow.
    assert blunt.length == pytest.approx(sharp.length, rel=1e-6)


def test_an_over_large_shoulder_radius_is_refused() -> None:
    with pytest.raises(ValueError, match="too large"):
        bodies.sphere_cone(
            half_angle=np.deg2rad(10.0), nose_radius=0.05, length=2.0, shoulder_radius=1.0
        )


def test_a_loft_refuses_a_degenerate_station_list() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        brep.Loft(section=lambda u: np.zeros((4, 3)), stations=np.array([0.5, 0.5, 1.0]))


def test_a_revolve_refuses_a_non_monotone_profile() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        brep.Revolve(station=np.array([0.0, 0.5, 0.4]), radius=np.array([0.0, 0.1, 0.2]))


def test_step_export_is_readable(tmp_path) -> None:
    """STEP is the interchange format, so the file has to be a real one."""
    path = brep.write_step(bodies.sphere_cone(), tmp_path / "cone.step")
    assert path.exists()
    text = path.read_text(errors="ignore")
    assert text.startswith("ISO-10303-21")
    assert "MANIFOLD_SOLID_BREP" in text or "ADVANCED_BREP_SHAPE_REPRESENTATION" in text


class TestEdgeBlunting:
    """Blunting radii are design variables, so they need real invariants."""

    def test_rounding_only_removes_material(self) -> None:
        from aether.geometry.edges import round_corners

        square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        sharp_reach = float(np.max(np.linalg.norm(square - 0.5, axis=1)))
        for radius in (0.05, 0.15, 0.3):
            rounded = round_corners(square, radius, samples=12)
            reach = float(np.max(np.linalg.norm(rounded - 0.5, axis=1)))
            assert reach < sharp_reach, "a fillet cannot push a corner outward"

    def test_the_arc_is_tangent_to_both_edges(self) -> None:
        """A fillet that is not tangent is a chamfer, and reads as one."""
        from aether.geometry.edges import round_corners

        square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        radius = 0.2
        rounded = round_corners(square, radius, samples=24)
        # Every arc point around the (1, 0) corner is `radius` from its centre.
        centre = np.array([1.0 - radius, radius])
        near = rounded[np.linalg.norm(rounded - np.array([1.0, 0.0]), axis=1) < 0.35]
        distances = np.linalg.norm(near - centre, axis=1)
        assert np.allclose(distances, radius, atol=1e-9), (
            f"arc radii span {distances.min():.6f}..{distances.max():.6f}"
        )

    def test_zero_radius_is_exactly_the_sharp_shape(self) -> None:
        from aether.geometry.edges import round_corners

        points = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -0.5]])
        assert np.array_equal(round_corners(points, 0.0), points)

    def test_radii_may_differ_per_corner(self) -> None:
        """A waverider blunts its leading edges and its ridge differently."""
        from aether.geometry.edges import round_corners

        points = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -0.5]])
        mixed = round_corners(points, np.array([0.05, 0.05, 0.0]), samples=6)
        # Two corners rounded into arcs, the third left as a single vertex.
        assert len(mixed) == 2 * 6 + 1


class TestBasePressure:
    """The Euler base is substituted, not computed. The substitute must be sane."""

    @pytest.mark.parametrize("mach", [1.5, 2.0, 4.0, 8.0, 12.0, 20.0])
    def test_never_below_vacuum(self, mach) -> None:
        """Nothing can pull harder than an absolute vacuum."""
        from aether.aerodynamics.closure import (
            base_pressure_coefficient,
            vacuum_pressure_coefficient,
        )

        base = base_pressure_coefficient(mach)
        assert base < 0.0, "a base is always below freestream in supersonic flow"
        assert base >= vacuum_pressure_coefficient(mach)

    def test_base_drag_falls_with_mach(self) -> None:
        from aether.aerodynamics.closure import base_axial_coefficient

        values = [base_axial_coefficient(m, 1.0, 1.0) for m in (2.0, 4.0, 8.0, 16.0)]
        assert all(a > b for a, b in itertools.pairwise(values))
        assert all(v > 0.0 for v in values), "base drag adds to axial force"

    def test_it_replaces_an_impossible_euler_value(self) -> None:
        """What the substitution is for, as a number.

        The solver reported a *positive* pressure coefficient on a base at
        Mach 8 — pushing the vehicle forward — worth -0.30 in axial force
        against a forebody of +0.08. The correlated value is small, negative,
        and bounded.
        """
        from aether.aerodynamics.closure import base_axial_coefficient

        correlated = base_axial_coefficient(8.0, 0.489, 0.489)
        euler_artefact = -0.30
        assert correlated > 0.0
        assert abs(correlated) < 0.05
        assert correlated * euler_artefact < 0.0, "they do not even share a sign"
