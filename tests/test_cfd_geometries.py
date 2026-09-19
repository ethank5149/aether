"""The reference bodies, and the properties a mesher needs from them.

Checked against geometry rather than against stored vertex counts: a body is
closed or it is not, wound outward or it is not, and those are the two
properties whose failure a picture does not show.
"""

from __future__ import annotations

import numpy as np
import pytest

from aether.cfd.geometries import REFERENCE_BODIES, build_surface
from aether.geometry.mesh import VehicleMesh


def _signed_volume(mesh) -> float:
    t = mesh.triangles
    return float(
        np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])).sum() / 6.0
    )


def test_the_study_spans_more_than_one_family_of_shape() -> None:
    """A survey of cones at one condition is a mesh study, not a survey."""
    assert set(REFERENCE_BODIES) == {
        "sphere-cone",
        # Not a survey shape. It is the verification body: the only entry whose
        # exact answer is known without a solver -- modified Newtonian on a
        # hemisphere is C_pmax/2 -- and the one with published standoff data to
        # check the shock against.
        "hemisphere",
        "biconic",
        "caret-waverider",
        "lifting-body",
        "osculating-cone",
        "sears-haack",
        "von-karman-ogive",
        "power-law-newtonian",
        "power-law-blast",
    }
    lifting = {"caret-waverider", "lifting-body", "osculating-cone"}
    assert all(not REFERENCE_BODIES[name].axisymmetric for name in lifting)
    assert all(REFERENCE_BODIES[name].axisymmetric for name in set(REFERENCE_BODIES) - lifting)


@pytest.mark.parametrize("name", sorted(REFERENCE_BODIES))
def test_every_reference_body_closes_into_something_meshable(name: str) -> None:
    """Watertight and outward-wound -- the two the domain mesher depends on.

    An inward-wound body passes every other check and makes the mesher fill
    the vehicle instead of the space around it.
    """
    mesh = build_surface(name)
    assert mesh.is_closed, f"open at stations {mesh.boundary_stations()}"
    assert _signed_volume(mesh) > 0.0


@pytest.mark.parametrize("name", sorted(REFERENCE_BODIES))
def test_a_reference_body_has_extent_in_every_direction(name: str) -> None:
    vertices = build_surface(name).vertices
    assert np.all(np.ptp(vertices, axis=0) > 0.0)


@pytest.mark.parametrize(
    "name",
    [n for n, b in sorted(REFERENCE_BODIES.items()) if not b.axisymmetric],
)
def test_the_non_axisymmetric_bodies_really_are_not_round(name: str) -> None:
    """Otherwise they test nothing the cones do not already test.

    Measured as the *shape of a cross-section*, not as span against height.
    The obvious proxy -- width differing from depth -- says nothing useful: a
    body can be exactly as tall as it is wide and still be nothing like a body
    of revolution, which is what the osculating-cone design happens to be at
    its defaults (0.716 m across, 0.750 m deep). It failed a test that was
    asking the wrong question.

    So the section is taken at mid-body and its radius measured about its own
    centroid. A body of revolution gives one radius at every azimuth; anything
    else does not.
    """
    vertices = build_surface(name).vertices
    stations = vertices[:, 0]
    middle = stations.min() + 0.5 * np.ptp(stations)
    band = np.abs(stations - middle) < 0.02 * np.ptp(stations)
    section = vertices[band][:, 1:]
    assert len(section) > 20, "no cross-section to measure"

    radius = np.linalg.norm(section - section.mean(axis=0), axis=1)
    # A circle has one radius; a lifting body's varies by a large factor.
    assert radius.max() / max(radius.min(), 1e-9) > 2.0


def test_the_waverider_is_tessellated_from_the_exact_solid() -> None:
    """A B-rep body, so the mesh converges to the shape rather than to a facet.

    The check that it is really the solid and not a resampled approximation:
    the tessellated volume must equal the volume OpenCASCADE measures on the
    exact body, which a triangle soup baked at some fixed density would not.
    """
    from aether.geometry.brep import solid_properties

    body = REFERENCE_BODIES["caret-waverider"]
    exact = solid_properties(body.build())
    assert _signed_volume(build_surface(body)) == pytest.approx(exact.volume, rel=1e-3)


@pytest.mark.parametrize(
    "name",
    [n for n, b in sorted(REFERENCE_BODIES.items()) if b.axisymmetric],
)
def test_the_axisymmetric_bodies_really_are_axisymmetric(name: str) -> None:
    vertices = build_surface(name).vertices
    assert np.ptp(vertices[:, 1]) == pytest.approx(np.ptp(vertices[:, 2]), rel=0.02)


def test_the_biconic_has_two_cone_angles_not_one() -> None:
    """A biconic that meshed as a single cone would pass every other test.

    Measured as the slope of the profile's outer radius against station: one
    cone gives a constant slope, two give a step down at the junction.
    """
    vertices = build_surface("biconic").vertices
    x = vertices[:, 0]
    radius = np.linalg.norm(vertices[:, 1:], axis=1)
    stations = np.linspace(x.min() + 0.35, x.max() - 0.1, 40)
    outer = np.array([radius[np.abs(x - s) < 0.05].max() for s in stations])
    slope = np.gradient(outer, stations)
    forward, aft = slope[:12].mean(), slope[-12:].mean()
    assert forward > aft > 0.0, f"forward slope {forward:.4f}, aft {aft:.4f}"


def test_the_analytic_sphere_cone_can_also_be_meshed() -> None:
    """``panels.sphere_cone`` used to be the one sphere-cone that could not.

    It built its panels directly and never kept the vertex net they were cut
    from, so the meshable sphere-cone had to be spelled as a single-segment
    ``blunted_multiconic`` and which function you called depended on what you
    meant to do with the answer. That was an accident of the order the two
    generators were written in, not a distinction worth keeping, and this
    guards the repair.
    """
    from aether.aerodynamics.panels import sphere_cone

    model = sphere_cone()
    assert model.surface is not None
    mesh = VehicleMesh.from_surface_grid(model.surface, name="sphere_cone")
    assert mesh.is_closed
    assert _signed_volume(mesh) > 0.0


def test_a_generator_with_no_surface_grid_is_refused_with_the_reason() -> None:
    """Some generators are still panel-only, and the failure has to say so.

    ``caret_lifting_body`` is one: an open lifting surface for impact theory
    with no closed body behind it. A bare ``AttributeError`` on ``None`` three
    frames down is not a useful way to learn that.
    """
    from aether.aerodynamics.panels import caret_lifting_body
    from aether.cfd.geometries import ReferenceBody

    body = ReferenceBody("open", caret_lifting_body, False, "panel-only")
    with pytest.raises(ValueError, match="no surface grid"):
        build_surface(body)


# ------------------------------------------- defaults that used to be silent


def test_a_flight_condition_cannot_leave_its_freestream_unstated() -> None:
    """The trap this closes cost a 71-minute solve at the wrong altitude.

    ``FlowConditions(mach=8.0)`` used to be accepted and quietly meant sea
    level. Nothing downstream complains: the case renders, converges, and
    reports coefficients to six figures against a dynamic pressure two orders
    of magnitude away from the one intended.
    """
    from aether.cfd.config import FlowConditions

    with pytest.raises(TypeError, match="temperature"):
        FlowConditions(mach=8.0)  # type: ignore[call-arg]


def test_a_flight_condition_from_an_altitude_matches_the_atmosphere() -> None:
    from aether.atmosphere import earth_atmosphere
    from aether.cfd.config import FlowConditions

    altitude = 45_000.0
    state = earth_atmosphere().state(altitude)
    flow = FlowConditions.at_altitude(mach=20.0, altitude=altitude)
    assert flow.temperature == pytest.approx(float(np.atleast_1d(state.temperature)[0]))
    assert flow.pressure == pytest.approx(float(np.atleast_1d(state.pressure)[0]))
    assert flow.pressure < 1000.0, "45 km is not sea level"


def test_a_meshed_domain_separates_the_base_by_default() -> None:
    """An Euler base pressure is meaningless, so it must be separable.

    Folded into one coefficient it does not degrade the answer, it replaces
    it: on a Mach 8 sphere-cone the forebody gives C_A = +0.069 and the base
    -0.254, so the reported drag is negative and the body appears to be under
    thrust. The split has to be the default because the failure is invisible
    -- the run converges and reports six figures.
    """
    import inspect

    from aether.aerodynamics.cfd.meshing import inviscid_domain

    for entry in (inviscid_domain,):
        assert inspect.signature(entry).parameters["split_base"].default is True


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("sears-haack", lambda: 3.0 * np.pi**2 * 0.28**2 * 2.5 / 16.0),
        ("power-law-newtonian", lambda: np.pi * 0.32**2 * 2.5 / (2.0 * 0.75 + 1.0)),
        ("power-law-blast", lambda: np.pi * 0.32**2 * 2.5 / (2.0 * (2.0 / 3.0) + 1.0)),
    ],
)
def test_an_analytic_body_meshes_to_the_volume_its_formula_gives(name, expected) -> None:
    """The reason these shapes are in the study: they carry their own answer.

    Checked on the *meshed* body rather than on the profile, so it ties the
    geometry that reaches the solver back to the closed form -- the tolerance
    is the tessellation's, not the formula's.
    """
    assert _signed_volume(build_surface(name)) == pytest.approx(expected(), rel=0.02)


# --------------------------------------------------------- the reference frame


@pytest.mark.parametrize(
    ("name", "radius", "tolerance"),
    [
        # Monotone forebodies: the maximum radius is at the base, which is
        # always a sample point, so the area is exact to rounding.
        ("power-law-newtonian", 0.32, 1e-12),
        ("power-law-blast", 0.32, 1e-12),
        ("von-karman-ogive", 0.32, 1e-12),
        # Sears-Haack peaks mid-body, and its peak falls *between* stations,
        # so the vertex maximum is a slight under-estimate rather than the
        # true one. Small, one-sided, and worth stating rather than hiding
        # behind a loose tolerance everywhere.
        ("sears-haack", 0.28, 1e-4),
    ],
)
def test_the_reference_area_is_exact_for_a_body_of_revolution(
    name: str, radius: float, tolerance: float
) -> None:
    """Taken from vertices on the true surface, so mesh density cannot move it.

    A reference area that drifts with tessellation is a bad reference area:
    every coefficient computed against it drifts too, and the drift looks like
    physics.
    """
    from aether.cfd.geometries import reference_frame

    assert reference_frame(name).area == pytest.approx(np.pi * radius**2, rel=tolerance)


def test_a_body_peaking_between_stations_under_estimates_rather_than_over() -> None:
    """The one-sidedness matters: a reference area may be low, never high.

    Sampling a smooth maximum can only miss it from below, so the frontal area
    of such a body is a lower bound and every coefficient built on it is a
    slight over-estimate -- which is the safe direction, and knowable.
    """
    from aether.cfd.geometries import reference_frame

    assert reference_frame("sears-haack").area <= np.pi * 0.28**2


def test_every_body_gets_a_reference_of_its_own() -> None:
    """One hard-coded area would be wrong for most of them, and invisibly so."""
    from aether.cfd.geometries import reference_frame

    areas = {name: reference_frame(name).area for name in REFERENCE_BODIES}
    assert all(area > 0.0 for area in areas.values())
    assert max(areas.values()) / min(areas.values()) > 2.0


@pytest.mark.parametrize("name", sorted(REFERENCE_BODIES))
def test_the_reference_length_and_moment_point_span_the_body(name: str) -> None:
    from aether.cfd.geometries import reference_frame

    stations = build_surface(name).vertices[:, 0]
    frame = reference_frame(name, moment_fraction=0.5)
    assert frame.length == pytest.approx(np.ptp(stations))
    assert stations.min() < frame.moment_reference[0] < stations.max()


def test_the_projected_area_of_a_cone_is_its_base() -> None:
    """The identity the non-axisymmetric branch rests on, checked where it can be.

    Half the summed ``|n_x| A`` over a closed body is its silhouette, for any
    shape whose outline an axial ray crosses twice. A cone is such a shape and
    has a base area that can be written down.
    """
    mesh = build_surface("sphere-cone")
    triangles = mesh.triangles
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    projected = 0.25 * float(np.abs(normals[:, 0]).sum())
    radius = np.linalg.norm(mesh.vertices[:, 1:], axis=1).max()
    assert projected == pytest.approx(np.pi * radius**2, rel=0.01)


def test_the_panel_budget_is_split_to_suit_each_body() -> None:
    """A fixed count pair is one body's proportions imposed on every body.

    Circumferential spacing is 2*pi*rbar/n_circ and meridian spacing is L/n_ax,
    so a fixed pair squares the cells of exactly one shape. At 80 x 48 the
    hemisphere came out at 2.62 -- a median triangle edge ratio of 4.92 against
    the sphere-cone's 1.75, and a median scaled Jacobian of 0.20 against 0.58 in
    the mesh extruded from it.
    """
    import numpy as np

    from aether.cfd.geometries import surface_grid

    for name in ("hemisphere", "sphere-cone", "biconic"):
        body = REFERENCE_BODIES[name]
        grid = surface_grid(body, body.defaults())
        surface = build_surface(name, **body.defaults())
        vertices = np.asarray(surface.vertices, dtype=float)
        triangle = vertices[np.asarray(surface.faces, dtype=int)]
        edge = np.stack(
            [
                np.linalg.norm(triangle[:, 1] - triangle[:, 0], axis=1),
                np.linalg.norm(triangle[:, 2] - triangle[:, 1], axis=1),
                np.linalg.norm(triangle[:, 0] - triangle[:, 2], axis=1),
            ],
            axis=1,
        )
        ratio = float(np.median(edge.max(axis=1) / np.maximum(edge.min(axis=1), 1e-30)))
        assert ratio < 2.0, f"{name} panels are slivers: {ratio:.2f}, grid {grid}"
        # The nose cap needs an even count per side of its square patch.
        assert grid["n_circ"] % 8 == 0


def test_the_budget_is_held_when_the_split_moves() -> None:
    """Re-splitting must not quietly buy quality with run time."""
    from aether.cfd.geometries import BASE_PANELS, surface_grid

    body = REFERENCE_BODIES["sphere-cone"]
    grid = surface_grid(body, body.defaults())
    # Cap plus one frustum, so two segments' worth of axial budget.
    before = BASE_PANELS["n_axial_per_segment"] * 2 * BASE_PANELS["n_circ"]
    after = grid["n_axial_per_segment"] * 2 * grid["n_circ"]
    assert 0.9 < after / before < 1.1


def test_a_hemisphere_is_a_hemisphere() -> None:
    """The verification body, against the only two numbers it can be wrong about.

    Its wetted area is 2*pi*R^2 and its modified-Newtonian axial coefficient is
    exactly C_pmax/2 -- no integration, no fitted constant. Both are checked as
    *sequences*, because a single level cannot tell a converging discretisation
    error from a constant offset, and that distinction is precisely what was
    wrong here: `newtonian_forebody` was not passing `resolution` through, so
    its error sat at -0.4278 % on every level of every grid study.
    """
    import numpy as np

    from aether.cfd.forces import newtonian_forebody

    body = REFERENCE_BODIES["hemisphere"]
    radius = body.defaults()["radius"]
    wetted = 2.0 * np.pi * radius**2
    # Rayleigh pitot at Mach 8, gamma 1.4.
    exact = 1.8273543 / 2.0

    areas, drags = [], []
    for resolution in (1.0, 2.0, 4.0):
        surface = build_surface(body, resolution=resolution, **body.defaults())
        normal = np.asarray(surface.normals, dtype=float)
        area = np.asarray(surface.areas, dtype=float)
        areas.append(abs(float(area[normal[:, 0] < 0.0].sum()) / wetted - 1.0))
        drags.append(
            abs(
                newtonian_forebody(
                    body, body.defaults(), 8.0, np.pi * radius**2, resolution=resolution
                )
                / exact
                - 1.0
            )
        )

    assert areas[0] < 2.0e-3 and drags[0] < 3.0e-3
    # Both converge, and neither stalls on a floor.
    assert areas[0] > areas[1] > areas[2]
    assert drags[0] > drags[1] > drags[2]
    assert drags[0] / drags[2] > 8.0


@pytest.mark.parametrize("name", sorted(REFERENCE_BODIES))
def test_every_body_can_be_refined(name: str) -> None:
    """Six of the nine refused refinement outright, so no grid-convergence
    study was possible on the waveriders, the ogive or the power-law bodies --
    every coefficient they produced was a single-grid number with no error bar.

    Each underlying generator had accepted its element counts all along; the
    wrappers dropped them. What the wrappers could not reach was the gmsh path,
    where refinement is a target *size* rather than a count, and for a
    curvature-limited shape a size **floor** as well: the caret waverider's 3 mm
    leading edge holds most of its faces at the floor, so halving ``size_max``
    alone moved it 1.20x where a level wants 4.
    """
    body = REFERENCE_BODIES[name]
    geometry = body.defaults()
    coarse = len(np.asarray(build_surface(name, resolution=1.0, **geometry).faces))
    fine = len(np.asarray(build_surface(name, resolution=2.0, **geometry).faces))
    # Doubling the resolution refines two surface directions, so the face count
    # goes as r^2. Bodies that mix a size-limited region with a curvature-limited
    # one land below 4; anything near 1 is a level that did not refine at all.
    assert fine / coarse > 2.5, (
        f"{name} refined only {fine / coarse:.2f}x ({coarse} -> {fine} faces); "
        "a refinement study on levels this close reports an order about nothing"
    )
    assert fine / coarse < 5.0, f"{name} refined {fine / coarse:.2f}x, beyond r^2"


def test_a_refinement_request_is_honoured_or_refused_never_ignored() -> None:
    """The guard that made the refusal visible must stay: silently returning the
    baseline mesh is what turns an unrefinable body into a convergence order
    about nothing."""
    body = REFERENCE_BODIES["sphere-cone"]
    with pytest.raises(ValueError, match="resolution must be positive"):
        build_surface("sphere-cone", resolution=0.0, **body.defaults())
