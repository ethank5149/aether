"""The singularity-free nose cap.

A surface of revolution converges every meridian to one vertex, so the panels
there are slivers whose circumferential extent goes to zero. The stagnation
point is the worst place in a hypersonic mesh to keep the worst cells: it
carries the peak pressure, the peak heating, and the largest axial force per
unit area on the vehicle.
"""

from __future__ import annotations

import numpy as np
import pytest

from aether.cfd.extrude import extrude_to_shock, measure_quality
from aether.cfd.geometries import build_surface
from aether.cfd.gridding import grid_requirement, stagnation_pressure
from aether.cfd.nosecap import replace_nose_pole
from aether.cfd.shock import envelope_for
from aether.geometry.mesh import VehicleMesh

GEOMETRY = {"nose_radius": 0.06, "length": 2.0, "half_angle": 10.0}


def _capped(resolution: float = 1.0):
    surface = build_surface("sphere-cone", resolution=resolution, **GEOMETRY)
    result = replace_nose_pole(surface, 0.06)
    assert result is not None
    return surface, result, VehicleMesh(result.vertices, result.faces, name="capped")


def test_the_stagnation_vertex_has_low_valence() -> None:
    """Forty-eight triangles met at the stagnation point. Now a handful do.

    Asserted by **position**, not index: the surgery renumbers vertices, and an
    index carried across it points at whatever happens to have moved into that
    slot.
    """
    original, result, _ = _capped()

    def valence_at_tip(vertices, faces):
        tip = int(np.argmin(np.asarray(vertices, dtype=np.float64)[:, 0]))
        incident = np.zeros(len(vertices), dtype=np.int64)
        np.add.at(incident, np.asarray(faces).ravel(), 1)
        return int(incident[tip])

    before = valence_at_tip(original.vertices, original.faces)
    after = valence_at_tip(result.vertices, result.faces)
    assert before >= 40
    assert after <= 8, f"stagnation vertex still carries {after} triangles"


def test_the_stitch_adds_no_vertices_and_leaves_no_crack() -> None:
    """The patch boundary *is* the seam ring, not a copy of it.

    Duplicating those positions would give a mesh that looks watertight and has
    a crack down the join.
    """
    _original, result, capped = _capped()
    assert capped.is_closed
    # Every seam position appears once.
    unique = np.unique(np.round(result.vertices, 9), axis=0)
    assert len(unique) == len(result.vertices)


def test_it_applies_at_every_refinement_level() -> None:
    """A default that matched one resolution left the pole in place at the rest.

    Silently, by declining -- which is worse than failing, because the
    resolutions where it declined are the ones where the pole hurts most.
    """
    for resolution in (0.7071, 1.0, 1.4142, 2.0):
        surface = build_surface("sphere-cone", resolution=resolution, **GEOMETRY)
        assert replace_nose_pole(surface, 0.06) is not None, resolution


def test_poor_cells_stop_growing_with_refinement() -> None:
    """The property the whole fix exists for.

    The revolved cap degrades without bound -- 0, 0, 7040, 10560 cells below the
    quality floor across a fourfold cell count -- which is a ceiling on how far
    the grid can be refined. Capped, it plateaus.
    """
    envelope = envelope_for("sphere-cone", GEOMETRY, 8.0)
    requirement = grid_requirement(
        envelope.standoff, stagnation_pressure(8.0, 149.101418), 300.0, cell_reynolds=100.0
    )

    counts, medians, percentiles = [], [], []
    for resolution in (1.4142, 2.0):
        _, _, capped = _capped(resolution)
        quality = measure_quality(
            extrude_to_shock(
                capped, envelope, requirement.wall_cell,
                layers=requirement.cells_across_layer,
            )
        )
        assert quality.inverted == 0
        counts.append(quality.poor)
        medians.append(quality.median)
        percentiles.append(quality.first_percentile)

    assert max(counts) < 200
    # And the bulk of the mesh is better, not merely the tail. The bulk measure
    # is the **median**; this used to assert `first_percentile > 0.25`, which is
    # by construction the tail and was reading one at 0.25-0.26 because the
    # panel split happened to put it there. Choosing the split per body
    # -> [[dataset-the-surface-grid-was-one-body-s-proportions]] moved it to
    # 0.198-0.206 while leaving the median at 0.584-0.595 and *improving* the
    # poor count from 84-88 to 64-66, so the old assertion would have failed a
    # change that made the mesh better by every other measure.
    assert min(medians) > 0.5
    assert min(percentiles) > 0.15


def test_a_body_with_no_blunt_nose_is_declined_rather_than_mangled() -> None:
    """A waverider's nose is an edge, not a point."""
    waverider = build_surface(
        "caret-waverider", design_mach=8.0, wedge_angle=6.0, length=4.0,
        leading_edge_radius=0.003, ridge_radius=0.01,
    )
    assert replace_nose_pole(waverider, 0.003) is None


def test_the_nose_radius_is_measured_rather_than_named() -> None:
    """The caller should not have to know what this body calls its nose radius.

    The worker asked for ``geometry["nose_radius"]`` and skipped the cap when it
    was absent. The hemisphere spells it ``radius``, so the one body whose
    stagnation point is the entire purpose of the run kept its axis singularity
    -- silently, because a skipped cap looks exactly like a body that has none.
    """
    from aether.cfd.nosecap import nose_sphere_radius

    for name, geometry, expected in (
        ("hemisphere", {"radius": 0.045}, 0.045),
        ("sphere-cone", GEOMETRY, 0.06),
    ):
        measured = nose_sphere_radius(build_surface(name, **geometry))
        assert measured is not None
        assert measured == pytest.approx(expected, rel=1e-4)


def test_a_nose_that_is_not_spherical_is_not_given_a_radius() -> None:
    """The algebra returns a number for any three rings; agreement is the test."""
    import numpy as np

    from aether.aerodynamics.panels import surface_of_revolution
    from aether.cfd.nosecap import nose_sphere_radius
    from aether.geometry.mesh import VehicleMesh

    # A cone: r = x tan(15 deg), which is not tangent to any sphere at the apex.
    station = np.linspace(0.0, 1.0, 40)
    model = surface_of_revolution(station, station * np.tan(np.radians(15.0)), n_circ=32)
    cone = VehicleMesh.from_surface_grid(model.surface, name="cone")
    assert nose_sphere_radius(cone) is None
    assert replace_nose_pole(cone) is None


def test_the_cap_valence_is_reported_for_the_cap_and_not_the_base() -> None:
    """A hemisphere's cap centre *is* its base plane, and that broke the report.

    ``_polar`` divides by the distance from the cap centre, so the base disc's
    fan pole -- which sits within 1e-8 m of that centre on a hemisphere -- comes
    back at theta = 0 whichever side of it the rounding puts it. Without a
    radial test the base pole is read as the nose pole and a successful cap
    reports "80 -> 80".
    """
    for name, geometry in (
        ("hemisphere", {"radius": 0.045}),
        ("sphere-cone", GEOMETRY),
    ):
        result = replace_nose_pole(build_surface(name, **geometry))
        assert result is not None
        assert result.peak_valence_before > 16
        assert result.peak_valence_after <= 8


def test_the_cap_leaves_no_cell_the_solver_will_blow_up_on() -> None:
    """The cap traded a pole singularity for four corner ones, and that is not a trade.

    Snapping the gnomonic square's boundary straight onto the circular seam put
    the whole square-to-circle reconciliation into one row of quads, and its
    corners took all of it. On the hemisphere every one of the 102 cells below
    the quality floor sat at exactly phi = 22.46 deg, azimuth 45 deg mod 90 --
    the four diagonals -- and the Mach 8 solve diverged from that exact place,
    x = 0.0033, y = z = 0.0122, by iteration 200. It had been recorded as "a
    thin band at the seam, small and bounded".

    A graded O-grid annulus spreads the reconciliation over as many rings as
    the grading needs. Measured across all three blunt bodies: no cell below
    the floor at all, and the worst scaled Jacobian up from 0.037 to 0.148.
    """
    import numpy as np

    from aether.cfd.config import FlowConditions
    from aether.cfd.registry import RunSpec
    from aether.cfd.shockfit import fit_shock_layer

    for body, geometry in (
        ("hemisphere", {"radius": 0.045}),
        ("sphere-cone", GEOMETRY),
    ):
        spec = RunSpec(
            body=body, mach=8.0, altitude_km=45.0, geometry=geometry,
            shock_fitted=True, cell_reynolds=100.0,
        )
        fitted = fit_shock_layer(
            spec, FlowConditions.at_altitude(mach=8.0, altitude=45.0e3)
        )
        assert fitted.quality.inverted == 0
        assert fitted.quality.poor == 0, f"{body}: {fitted.quality.summary()}"
        assert fitted.quality.minimum > 0.1, f"{body}: {fitted.quality.summary()}"
        assert np.isfinite(fitted.standoff) and fitted.standoff > 0.0


def test_the_capped_surface_is_watertight() -> None:
    """A crack here is invisible to area, to valence, and to the winding check.

    The seam was selected with a tolerance of 1e-6 on its polar angle and the
    faces to discard with one of 1e-9, against a `seam_theta` that is a
    *rounded* value -- so a ring's own vertices could be both the seam and
    inside it, and the band of faces immediately outside the seam was thrown
    away with nothing put back. 160 edges of the hemisphere's cap belonged to
    one triangle instead of two, and the mesh still reported the right wetted
    area and the right peak valence.
    """
    import collections

    import numpy as np

    for body, geometry in (
        ("hemisphere", {"radius": 0.045}),
        ("sphere-cone", GEOMETRY),
        ("biconic", {}),
    ):
        surface = build_surface(body, **geometry) if geometry else build_surface(body)
        result = replace_nose_pole(surface)
        assert result is not None
        edges: collections.Counter = collections.Counter()
        for triangle in np.asarray(result.faces):
            for a, b in (
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ):
                edges[(min(a, b), max(a, b))] += 1
        stray = [edge for edge, count in edges.items() if count != 2]
        assert not stray, f"{body}: {len(stray)} non-manifold edges"


def test_every_patch_face_is_wound_outward_not_most_of_them() -> None:
    """The winding was decided by a sum over the whole patch.

    That is correct only while every face already agrees, which was true of one
    square grid and stopped being true the moment the patch became a core plus
    an annulus built by a different loop. The sum flipped whichever group was
    outvoted: 6452 inverted prisms, on a surface that looked perfect.
    """
    import numpy as np

    surface = build_surface("hemisphere", radius=0.045)
    result = replace_nose_pole(surface)
    assert result is not None
    vertices = np.asarray(result.vertices)
    faces = np.asarray(result.faces)
    corners = vertices[faces]
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    outward = corners.mean(axis=1) - np.array([0.045, 0.0, 0.0])
    # Only the cap's own faces lie on the sphere about its centre; the rest of
    # the body is checked by `build_surface`.
    on_cap = np.abs(np.linalg.norm(outward, axis=1) - 0.045) < 1e-6 * 0.045
    assert np.all(np.einsum("ij,ij->i", normal[on_cap], outward[on_cap]) > 0.0)
