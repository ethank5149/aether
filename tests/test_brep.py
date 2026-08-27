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
        np.einsum(
            "ij,ij->i", t[:, 0], np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        ).sum()
        / 6.0
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
        mesh = brep.surface_mesh(body, size_max=size, size_min=size / 40,
                                 curvature_nodes=24)
        errors.append(abs(_enclosed_volume(mesh) - exact) / exact)

    assert all(b < a for a, b in itertools.pairwise(errors)), (
        f"not monotone: {errors}"
    )
    assert errors[-1] < 0.005


def test_the_frozen_triangulation_does_not_reach_the_exact_body() -> None:
    """Why this module exists, as a number.

    The panel generator's grid is fixed at definition time, so its volume error
    is whatever that grid gives and is unreachable by any downstream refinement.
    """
    exact = brep.solid_properties(bodies.spatular_wedge()).volume
    sampled = abs(_enclosed_volume(
        VehicleMesh.from_surface_grid(panels.spatular_wedge().surface)
    ))
    assert abs(sampled - exact) / exact > 0.02, (
        "if this no longer holds the generators were changed and the note in "
        "aether.geometry.brep should be re-measured rather than trusted"
    )


def test_the_caret_keeps_its_leading_edge_sharp() -> None:
    """A waverider whose leading edge is rounded is a different vehicle.

    The section is a triangle, so the exact solid must reproduce the corner.
    Measured as the fraction of the base section's span that is within a
    hair of the z = 0 upper plate: a sharp caret has a straight upper surface
    all the way out to the edge, and a splined one bulges away from it.
    """
    body = bodies.caret_waverider()
    mesh = brep.surface_mesh(body, curvature_nodes=8)
    v = np.asarray(mesh.vertices)
    aft = v[v[:, 0] > 0.9 * v[:, 0].max()]
    assert aft.size > 0
    on_plate = np.count_nonzero(np.abs(aft[:, 2]) < 1.0e-9)
    assert on_plate > 0.2 * len(aft), (
        "the upper surface is not flat; the section was splined rather than "
        "built from straight edges"
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
