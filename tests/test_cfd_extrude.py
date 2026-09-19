"""The wall-normal extruder, checked against the analysis that sizes it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aether.cfd.extrude import (
    MARGIN,
    extrude_to_shock,
    geometric_spacing,
    vertex_normals,
)
from aether.cfd.geometries import build_surface
from aether.cfd.gridding import grid_requirement
from aether.cfd.shock import envelope_for

GEOMETRY = {"nose_radius": 0.06, "length": 2.0, "half_angle": 10.0}
STAGNATION_PRESSURE = 149.101418 * 82.87


@pytest.fixture(scope="module")
def surface():
    return build_surface("sphere-cone", **GEOMETRY)


@pytest.fixture(scope="module")
def layer(surface):
    envelope = envelope_for("sphere-cone", GEOMETRY, 8.0)
    requirement = grid_requirement(envelope.standoff, STAGNATION_PRESSURE, 300.0)
    return (
        extrude_to_shock(
            surface, envelope, requirement.wall_cell, layers=requirement.cells_across_layer
        ),
        envelope,
        requirement,
    )


def test_geometric_spacing_spans_zero_to_one_with_the_requested_first_cell() -> None:
    positions = geometric_spacing(1e-6, 1.0, 100, growth=1.15)
    assert positions[0] == 0.0
    assert positions[-1] == 1.0
    assert positions[1] == pytest.approx(1e-6)
    assert np.all(np.diff(positions) > 0.0)


def test_geometric_spacing_never_overshoots_the_total() -> None:
    """An unchecked geometric series blows past the total.

    A grid whose outer cells exceed the shock layer resolves nothing at the one
    place the physics lives, so the cap is not cosmetic.
    """
    for layers in (5, 20, 200):
        positions = geometric_spacing(1e-7, 1.0, layers, growth=1.5)
        assert positions[-1] == 1.0
        assert np.all(positions <= 1.0 + 1e-12)
        assert np.all(np.diff(positions) > 0.0)


def test_geometric_spacing_refuses_a_first_cell_larger_than_the_total() -> None:
    with pytest.raises(ValueError, match="first cell"):
        geometric_spacing(2.0, 1.0, 10)
    with pytest.raises(ValueError, match="at least one layer"):
        geometric_spacing(0.1, 1.0, 0)


def test_the_nose_column_reaches_exactly_the_billig_standoff_times_the_margin(layer) -> None:
    """The one node whose answer is known in closed form."""
    extruded, envelope, _ = layer
    nose = int(np.argmin(extruded.nodes[:, 0, 0]))
    assert extruded.thickness[nose] == pytest.approx(MARGIN * envelope.standoff, rel=1e-3)


def test_the_first_cell_is_the_one_laura_asked_for(layer) -> None:
    extruded, _, requirement = layer
    heights = np.linalg.norm(extruded.nodes[:, 1] - extruded.nodes[:, 0], axis=1)
    assert np.allclose(heights, requirement.wall_cell, rtol=1e-9)
    assert requirement.wall_cell < 1e-6


def test_no_column_crosses_another(layer) -> None:
    """Marched normals that collide make negative-volume cells.

    Zero is the only acceptable answer; anything else is a mesh that will fail
    somewhere downstream with a message about the solver.
    """
    extruded, _, _ = layer
    assert extruded.crossings == 0


def test_the_layer_thickens_from_nose_to_base_as_the_shock_diverges(layer) -> None:
    """A 13 degree shock over a 10 degree cone opens a widening gap.

    Measured: the shock stands 9.03 mm off the nose and 298 mm off the base, so
    the layer thickens by a factor of 18 between the forward tenth of the body
    and the aft tenth. A constant-thickness extrusion would be wrong by that
    factor at one end or the other.
    """
    extruded, _, _ = layer
    x = extruded.nodes[:, 0, 0]
    forward = extruded.thickness[x < 0.1].mean()
    aft = extruded.thickness[x > 1.8].mean()
    assert aft / forward > 15.0
    assert extruded.thickness.min() == pytest.approx(MARGIN * 0.00903, rel=1e-2)


def test_the_base_disc_is_not_extruded(layer, surface) -> None:
    """Its centre vertex has no forebody normal, and marching it would be silent.

    Counted against the surface rather than written down. The literal used to be
    3841, which was the count at one panel split and stopped being true the
    moment the split was chosen per body -- a test that pins an incidental
    number fails for the wrong reason and says nothing when the real invariant
    breaks. The invariant is: exactly one vertex, the base centre, is dropped.
    """
    extruded, _, _ = layer
    assert extruded.nodes.shape[0] == len(np.asarray(surface.vertices)) - 1
    assert extruded.faces.max() == extruded.nodes.shape[0] - 1
    assert extruded.faces.min() == 0


def test_smoothing_leaves_the_normals_unit_length() -> None:
    surface = build_surface("sphere-cone", **GEOMETRY)
    directions = vertex_normals(
        np.asarray(surface.vertices),
        np.asarray(surface.faces),
        np.asarray(surface.areas),
        np.asarray(surface.normals),
        smoothing=4,
    )
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)


# ------------------------------------------------------------------ mesh output


def _closure(points, markers):
    """Sum of outward area vectors over a closed surface. Should vanish."""
    total = np.zeros(3)
    for cells in markers:
        p = points[cells]
        if cells.shape[1] == 3:
            total += np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]).sum(axis=0) / 2.0
        else:
            total += np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1]).sum(axis=0) / 2.0
    return total


def test_the_written_mesh_is_a_closed_domain(tmp_path, layer) -> None:
    """Divergence theorem: outward area vectors over a closed surface sum to zero.

    This is the check that catches a missing boundary. Sweeping the rim edges
    outward is what seals the downstream end, and forgetting them would leave a
    hole that SU2 reports much later as something else entirely.
    """
    from aether.cfd.extrude import boundary_edges, write_su2

    extruded, _, _ = layer
    info = write_su2(extruded, tmp_path / "m.su2")
    assert info["min_volume"] > 0.0

    faces, nodes = extruded.faces, extruded.nodes
    n, levels = nodes.shape[0], nodes.shape[1]
    points = np.concatenate([nodes[:, k, :] for k in range(levels)], axis=0)
    rim = boundary_edges(faces)
    sides = np.concatenate(
        [
            np.stack(
                [
                    rim[:, 0] + k * n, rim[:, 1] + k * n,
                    rim[:, 1] + (k + 1) * n, rim[:, 0] + (k + 1) * n,
                ],
                axis=1,
            )
            for k in range(levels - 1)
        ]
    )
    residual = _closure(
        points,
        [faces + (levels - 1) * n, faces[:, ::-1], sides],
    )
    scale = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))) ** 2
    assert np.linalg.norm(residual) / scale < 1e-9


def test_the_rim_is_the_base_ring_and_seals_the_downstream_end(layer, surface) -> None:
    from aether.cfd.extrude import boundary_edges

    extruded, _, _ = layer
    rim = boundary_edges(extruded.faces)
    # One edge per circumferential station, counted off the base disc's own fan
    # rather than assumed: the split is chosen per body now, so the number is a
    # property of this surface and not of the mesher.
    vertices = np.asarray(surface.vertices)
    aft = vertices[:, 0] > vertices[:, 0].max() - 1e-9
    assert len(rim) == int(aft.sum()) - 1
    # Every rim vertex sits on the aft plane.
    x = extruded.nodes[:, 0, 0]
    assert np.ptp(x[np.unique(rim)]) < 1e-6


def test_counts_are_consistent(tmp_path, layer) -> None:
    from aether.cfd.extrude import write_su2

    extruded, _, _ = layer
    info = write_su2(extruded, tmp_path / "m.su2")
    assert info["prisms"] == extruded.faces.shape[0] * extruded.layers
    assert info["points"] == extruded.nodes.shape[0] * (extruded.layers + 1)
    from aether.cfd.extrude import boundary_edges

    assert info["outflow"] == len(boundary_edges(extruded.faces)) * extruded.layers

    text = (tmp_path / "m.su2").read_text().splitlines()
    assert text[0] == "NDIME= 3"
    assert text[1] == f"NELEM= {info['prisms']}"
    assert text[2].split()[0] == "13"
    assert sum(1 for line in text if line.startswith("MARKER_TAG=")) == 3


# ------------------------------------------------- thickness across geometries


def test_the_gradient_limiter_only_ever_grows_a_column() -> None:
    """Shortening could pull the outer boundary inside the shock; growing cannot.

    That asymmetry is the whole design. A short column lengthened gains
    freestream above it and loses a little resolution; a long column shortened
    silently truncates the physics it was sized to contain.
    """
    from aether.cfd.extrude import limit_thickness_gradient

    faces = np.array([[0, 1, 2], [1, 2, 3]])
    thickness = np.array([1.0, 0.001, 0.5, 0.25])
    limited = limit_thickness_gradient(thickness, faces, ratio=1.5)
    assert np.all(limited >= thickness - 1e-15)
    edges = np.array([[0, 1], [1, 2], [2, 0], [1, 3], [2, 3]])
    ratio = np.maximum(
        limited[edges[:, 0]] / limited[edges[:, 1]],
        limited[edges[:, 1]] / limited[edges[:, 0]],
    )
    assert ratio.max() <= 1.5 + 1e-9


def test_the_limiter_refuses_a_ratio_that_cannot_converge() -> None:
    from aether.cfd.extrude import limit_thickness_gradient

    with pytest.raises(ValueError, match="ratio must exceed 1"):
        limit_thickness_gradient(np.array([1.0, 2.0]), np.array([[0, 1, 0]]), ratio=1.0)


@pytest.mark.parametrize(
    ("body", "geometry", "expect_capped"),
    [
        ("sphere-cone", {"nose_radius": 0.06, "length": 2.0, "half_angle": 10.0}, False),
        (
            "biconic",
            {"nose_radius": 0.05, "length_1": 1.0, "length_2": 1.5,
             "half_angle_1": 12.0, "half_angle_2": 7.0, "fillet_radius": 0.1},
            False,
        ),
        (
            "caret-waverider",
            {"design_mach": 8.0, "wedge_angle": 6.0, "length": 4.0,
             "leading_edge_radius": 0.003, "ridge_radius": 0.01},
            True,
        ),
        ("sears-haack", {"length": 2.5, "max_radius": 0.28}, True),
    ],
)
def test_every_geometry_marches_to_a_bounded_well_graded_layer(
    body, geometry, expect_capped
) -> None:
    """Detached-shock bodies are bounded by their envelope; attached ones are not.

    A caret waverider's upper surface is a *freestream* surface and a sharp
    body's shock is attached at the apex, so many normals never meet a shock at
    all -- physics, not a bug, but it must be reported rather than silently
    producing 19.9 m columns on a 4 m vehicle.

    The assertion for a blunt body is **enclosure, not the absence of capping**.
    Those were the same thing at a margin of 1.25 and stopped being so at 1.75:
    beside the free edge at the base the smoothed normal tilts aft, the column
    runs nearly parallel to the shock rather than across it, and caps. On the
    sphere-cone every one of the 240 such columns is in the last 6 % of the body
    and the shock there is 60 mm off a wall whose column was cut at 512. What
    matters is that the column ends **outside** the shock, and it does.
    """
    from aether.cfd.extrude import march_thickness
    from aether.cfd.geometries import build_surface
    from aether.cfd.shock import envelope_for

    surface = build_surface(body, **geometry)
    envelope = envelope_for(body, geometry, 8.0)
    vertices, faces, directions, thickness, capped = march_thickness(surface, envelope)

    span = float(np.ptp(vertices[:, 0]))
    assert thickness.max() <= 0.25 * span + 1e-12
    assert thickness.min() > 0.0
    assert (capped > 0) is expect_capped or not expect_capped

    if not expect_capped:
        top = vertices + directions * thickness[:, None]
        side = envelope.outside(top[:, 0], np.hypot(top[:, 1], top[:, 2]))
        assert np.all(np.asarray(side) > 0.0), (
            f"{body}: {int(np.sum(np.asarray(side) <= 0.0))} columns end behind "
            "the shock"
        )

    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    ratio = np.maximum(
        thickness[edges[:, 0]] / thickness[edges[:, 1]],
        thickness[edges[:, 1]] / thickness[edges[:, 0]],
    )
    assert ratio.max() <= 1.5 + 1e-9


def test_the_blunt_nose_standoff_survives_the_limiter() -> None:
    """The one column whose value is known must not be inflated by smoothing."""
    from aether.cfd.extrude import march_thickness
    from aether.cfd.geometries import build_surface
    from aether.cfd.shock import envelope_for

    geometry = {"nose_radius": 0.06, "length": 2.0, "half_angle": 10.0}
    surface = build_surface("sphere-cone", **geometry)
    envelope = envelope_for("sphere-cone", geometry, 8.0)
    _, _, _, thickness, _ = march_thickness(surface, envelope)
    assert thickness.min() == pytest.approx(MARGIN * envelope.standoff, rel=1e-3)


def test_the_shock_layer_is_resolved_everywhere_along_the_body() -> None:
    """The property the whole rebuild exists to obtain.

    The isotropic tetrahedral mesh put **zero** cells between the body and the
    shock at the nose, at 1.36 million elements, and refining it 9.5-fold moved
    that count from 2 to 0. This asserts the replacement puts tens of cells
    there at *every* station, not just at the nose -- which it gets for free
    from each column's length following the shock rather than being constant.
    """
    from aether.cfd.extrude import extrude_to_shock
    from aether.cfd.geometries import build_surface
    from aether.cfd.gridding import grid_requirement, stagnation_pressure
    from aether.cfd.shock import envelope_for

    geometry = {"nose_radius": 0.06, "length": 2.0, "half_angle": 10.0}
    surface = build_surface("sphere-cone", **geometry)
    envelope = envelope_for("sphere-cone", geometry, 8.0)
    requirement = grid_requirement(
        envelope.standoff, stagnation_pressure(8.0, 149.101418), 300.0, cell_reynolds=100.0
    )
    extruded = extrude_to_shock(
        surface, envelope, requirement.wall_cell, layers=requirement.cells_across_layer
    )

    nodes = extruded.nodes
    radius = np.hypot(nodes[..., 1], nodes[..., 2])
    inside = envelope.outside(nodes[..., 0].ravel(), radius.ravel()).reshape(
        nodes.shape[:2]
    ) < 0
    cells = inside.sum(axis=1) - 1

    # Thirty is the low end of what hypersonic practice asks for across a shock
    # layer; the nose is the tightest column and still clears it.
    assert cells.min() >= 30
    # Roughly uniform along the body, because the column follows the shock.
    # The band widened from 25 % to 50 % when the marching margin went to 1.75:
    # every column carries the same layer count, so a column that has to reach
    # further past its shock spends more of that count in the freestream beyond
    # it. Measured 35 to 50 cells across the shock layer -- the tightest is
    # better resolved than the old uniform value, and the spread is in the part
    # of the domain where nothing happens.
    assert cells.max() - cells.min() < 0.5 * cells.min()


def test_quality_is_signed_so_stretched_and_inverted_are_distinguishable(layer) -> None:
    """A wall-normal grid is deliberately stretched; only inversion is a fault.

    Aspect ratios here run to 10^5 by design, so a low scaled Jacobian is
    expected and says nothing on its own. What must be nil is the count at or
    below zero -- and no one-sided measure separates the two.
    """
    from aether.cfd.extrude import measure_quality

    extruded, _, _ = layer
    quality = measure_quality(extruded)

    assert quality.measure == "scaledJacobian"
    assert quality.inverted == 0
    assert quality.usable
    assert quality.elements == extruded.faces.shape[0] * extruded.layers
    assert 0.0 < quality.minimum <= quality.median <= 1.0


def test_an_inverted_cell_is_reported_as_inverted() -> None:
    """Turn one column inside out and the measure must catch it."""
    import dataclasses

    from aether.cfd.extrude import ExtrudedLayer, measure_quality

    faces = np.array([[0, 1, 2]])
    base = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    good = np.stack([base, base + np.array([0.0, 0.0, 1.0])], axis=1)
    healthy = ExtrudedLayer(
        nodes=good, thickness=np.ones(3), wall_cell=1.0, crossings=0, faces=faces
    )
    assert measure_quality(healthy).inverted == 0

    # Push the top face below the bottom one: the cell folds through itself.
    folded = dataclasses.replace(
        healthy, nodes=np.stack([base, base - np.array([0.0, 0.0, 1.0])], axis=1)
    )
    assert measure_quality(folded).inverted == 1
    assert not measure_quality(folded).usable


def test_every_column_is_graded_across_its_own_thickness() -> None:
    r"""One growth ratio cannot serve a shock layer that varies by fifty times.

    The sphere-cone's marched thickness is a 9 mm standoff at the nose and half
    a metre at the base. `growth` was solved once, from the nose, and handed to
    every column -- so the geometric series spanned 10.68 mm at *every* station
    and `geometric_spacing`'s overshoot guard dumped the rest into the final
    cell. Measured on the Mach 8 alpha-4 mesh: a cell-to-cell jump of 841x and
    97.8 % of the shock layer inside one cell, which is why the bow shock had
    nowhere to live.
    """
    import numpy as np

    from aether.cfd.extrude import geometric_spacing

    wall, layers = 37.054e-6, 55
    thin, thick = 0.0158, 0.445           # nose standoff and base shock layer

    # The old behaviour, reproduced: one ratio, solved for the thin column.
    from aether.cfd.gridding import growth_for

    one_ratio = growth_for(wall, thin, layers)
    shared = geometric_spacing(wall, thick, layers, one_ratio) * thick
    assert (shared[-1] - shared[-2]) / (shared[-2] - shared[-3]) > 100.0, (
        "the defect must still be reproducible, or this test proves nothing"
    )

    # The fix: each column gets the ratio that spans its own thickness.
    for total in (thin, thick):
        graded = geometric_spacing(wall, total, layers, growth_for(wall, total, layers)) * total
        steps = np.diff(graded)
        assert steps[0] == pytest.approx(wall, rel=0.05), "wall cell is physics; keep it"
        ratios = steps[1:] / steps[:-1]
        assert ratios.max() < 1.25, f"grading {ratios.max():.3f} is too aggressive"
        assert ratios.min() > 0.99, "the series must be monotone, not capped flat"
        # And the graded levels must actually reach the shock, not 2 % of the way.
        assert graded[-2] / total > 0.8, (
            f"only {100 * graded[-2] / total:.1f}% of the layer is resolved"
        )


# ------------------------------------------------------------- the wake block
#
# A shock-fitted block ends at the base plane, which is right for forebody
# aerodynamics and cannot produce a base pressure. That came from
# `closure.base_axial_coefficient` -- an empirical C_pb = -1/M^2 worth 7 % of
# total drag at Mach 10 and 32 % at Mach 4.

WAKE_BODIES = ["sphere-cone", "hemisphere", "biconic", "lifting-body",
               "von-karman-ogive", "power-law-blast"]


def _fitted(name: str, *, wake: bool):
    from aether.cfd.config import FlowConditions
    from aether.cfd.geometries import REFERENCE_BODIES
    from aether.cfd.registry import RunSpec
    from aether.cfd.shockfit import fit_shock_layer

    spec = RunSpec(
        body=name, geometry=REFERENCE_BODIES[name].defaults(), mach=8.0,
        shock_fitted=True, wake=wake, wall_refinement=0.05, cell_reynolds=100.0,
        convective="HLLC", inviscid=False, turbulence=None,
    )
    flow = FlowConditions.at_altitude(mach=8.0, altitude=45.0e3, alpha=0.0)
    return spec, fit_shock_layer(spec, flow, say=lambda *_: None, strict=False)


def _face_census(path):
    """Every cell face, counted. Interior faces appear twice, boundary once."""
    import numpy as np

    lines = Path(path).read_text().splitlines()
    nelem = int(lines[1].split("=")[1])
    tris, quads = [], []
    for k in range(nelem):
        parts = lines[2 + k].split()
        kind, nodes = int(parts[0]), [int(v) for v in parts[1:-1]]
        if kind == 13:
            a, b, c, d, e, f = nodes
            tris += [(a, b, c), (d, e, f)]
            quads += [(a, b, e, d), (b, c, f, e), (c, a, d, f)]
        elif kind == 12:
            a, b, c, d, e, f, g, h = nodes
            quads += [(a, b, c, d), (e, f, g, h), (a, b, f, e),
                      (b, c, g, f), (c, d, h, g), (d, a, e, h)]
        else:
            raise AssertionError(f"unexpected cell type {kind}")
    j = 2 + nelem
    npoin = int(lines[j].split("=")[1])
    j += 1 + npoin
    nmark = int(lines[j].split("=")[1])
    j += 1
    declared = set()
    for _ in range(nmark):
        j += 1
        count = int(lines[j].split("=")[1])
        j += 1
        for k in range(count):
            declared.add(tuple(sorted(int(v) for v in lines[j + k].split()[1:])))
        j += count

    def boundary(faces):
        key = np.sort(np.array(faces, dtype=np.int64), axis=1)
        uniq, counts = np.unique(key, axis=0, return_counts=True)
        return {tuple(r) for r in uniq[counts == 1]}, int((counts > 2).sum())

    tri_b, tri_over = boundary(tris)
    quad_b, quad_over = boundary(quads)
    return tri_b | quad_b, declared, tri_over + quad_over


@pytest.mark.parametrize("name", WAKE_BODIES)
def test_a_merged_forebody_and_wake_is_watertight(name: str, tmp_path: Path) -> None:
    """The acceptance test, because every way of getting this wrong produces a
    mesh that passes cell-quality checks and leaks.

    Four did, on the way here: triangulating the forebody's quad annulus (13 680
    unmarked faces), welding the cap's two halves separately (a slit right round
    the base), taking the rim over the quads alone (1 920 interior faces declared
    farfield), and a quantised-hash weld that missed sixteen node pairs 1e-12
    apart (thirty-two coincident quads).
    """
    from aether.cfd.shockfit import write

    _, fitted = _fitted(name, wake=True)
    assert fitted.wake is not None, f"{name} has a base and should have a wake block"
    path = tmp_path / f"{name}.su2"
    write(fitted, path)

    boundary, declared, over = _face_census(path)
    assert over == 0, f"{over} faces are shared by more than two cells"
    assert not (boundary - declared), (
        f"{len(boundary - declared)} boundary faces belong to no marker: the "
        "domain has a hole in it"
    )
    assert not (declared - boundary), (
        f"{len(declared - boundary)} marker faces are interior, so a boundary "
        "condition would be imposed in the middle of the flow"
    )


@pytest.mark.parametrize("name", WAKE_BODIES)
def test_the_base_disc_becomes_wall_only_when_the_wake_is_resolved(name: str) -> None:
    """Without a wake the base is not in the mesh at all; with one it carries a
    computed pressure, which is the entire point."""
    import numpy as np

    _, plain = _fitted(name, wake=False)
    _, resolved = _fitted(name, wake=True)
    assert plain.wake is None
    assert resolved.wake is not None
    assert resolved.wake.base_faces > 0
    # The annulus is kept as quads, never triangulated: it has to meet the
    # forebody's own quad faces one-for-one.
    assert len(np.asarray(resolved.wake.quads)) > 0


def test_the_wake_reaches_a_base_diameter_scale_not_a_body_length() -> None:
    """A multiple of body length is the wrong ruler: it makes a stubby body's
    wake short and a slender body's needlessly long."""
    import numpy as np

    _, fitted = _fitted("sphere-cone", wake=True)
    nodes = np.asarray(fitted.wake.nodes)
    reach = float(nodes[:, -1, 0].max() - nodes[:, 0, 0].min())
    assert 6.0 < reach < 8.0, f"reach {reach:.2f} m is not eight base diameters"


def test_a_body_that_closes_to_a_point_gets_no_wake() -> None:
    """Sears-Haack is closed at both ends. There is no base, so there is no base
    flow, and sweeping its degenerate aft sliver would size a run off a rounding
    error."""
    _, fitted = _fitted("sears-haack", wake=True)
    assert fitted.wake is None


def test_the_recirculation_zone_carries_near_isotropic_cells() -> None:
    r"""A detached-eddy model resolves the large scales where the flow separates,
    and that is only meaningful on cells that are not badly stretched.

    Measured over the **core** -- inside half the base radius -- because that is
    where the recirculation bubble lives. The base grid is deliberately not
    isotropic near its rim: the boundary layer arrives there on a tens-of-micron
    wall cell and separates, and the shear layer it becomes needs that scale for
    a while, so the outer rings are graded down to meet it. Those cells are
    anisotropic the way a boundary layer's are -- thin across the layer, long
    along it -- which is resolution, not a defect.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    _, fitted = _fitted("sphere-cone", wake=True)
    wake = fitted.wake
    nodes = np.asarray(wake.nodes)
    level0 = nodes[:, 0, :]
    radius = np.hypot(level0[:, 1], level0[:, 2])

    ring_nodes = np.unique(np.asarray(wake.quads)[: wake.base_quads])
    rim = float(radius[ring_nodes].max())
    core = ring_nodes[radius[ring_nodes] < 0.5 * rim]
    assert len(core) > 40, "the core of the base grid is nearly empty again"

    spacing, _ = cKDTree(level0[core]).query(level0[core], k=2)
    lateral = float(np.median(spacing[:, 1]))
    axial = np.linalg.norm(np.diff(nodes, axis=1), axis=2)
    near = [
        k
        for k in range(nodes.shape[1] - 1)
        if nodes[core, k, 0].mean() < nodes[core, 0, 0].mean() + 2.0 * rim
    ]
    aspect = float(np.median(axial[core][:, near])) / lateral
    # The bound sits where the topology puts it, not at 1. In the core the
    # nearest-neighbour direction is circumferential, and that is capped at
    # 2*pi*r/48 by the body's own azimuthal panel count -- 13 mm at r = 0.1 m
    # against a ~55 mm axial step is 4.2 before anything has gone wrong. The
    # recorded cure is the O-grid the nose cap already uses; until then this
    # guards against a *worse* core, and the sharp regressions -- an empty core,
    # a cliff at the rim -- have their own assertions above and below.
    assert aspect < 5.0, (
        f"cells in the recirculation core are stretched {aspect:.1f}:1; a "
        "detached-eddy model on these filters what it is meant to resolve"
    )


def test_the_base_grid_meets_the_annulus_at_its_own_scale() -> None:
    r"""The regression guard for the 75x jump.

    The uniform base grid put a 49 mm cell against the annulus's sub-millimetre
    one at the exact radius where the boundary layer separates. The Mach 6 run
    answered with 65,000 and 72,000 K spikes riding the shear layer -- the same
    signature, for the same reason, as the 841x jump at the shock before the
    per-column grading. The base rings are graded now, and the outermost radial
    step has to match the wall-normal step arriving from the annulus to within
    a small factor, growing inward at a civil ratio.
    """
    import numpy as np

    _, fitted = _fitted("sphere-cone", wake=True)
    wake = fitted.wake
    level0 = np.asarray(wake.nodes)[:, 0, :]
    radius = np.hypot(level0[:, 1], level0[:, 2])

    ring_nodes = np.unique(np.asarray(wake.quads)[: wake.base_quads])
    radii = np.unique(np.round(radius[ring_nodes], 8))
    steps = np.diff(radii)
    assert len(steps) >= 8, "the graded base needs enough rings to bridge the scales"

    # The annulus's wall-normal arrival scale: its strip quads' [1]->[2] edge.
    annulus = np.asarray(wake.quads)[wake.base_quads :]
    arriving = float(
        np.min(np.linalg.norm(level0[annulus[:, 2]] - level0[annulus[:, 1]], axis=1))
    )
    outermost = float(steps[-1])
    assert outermost < 3.0 * arriving, (
        f"the base grid's outermost step ({1000 * outermost:.3f} mm) does not meet "
        f"the annulus's arriving scale ({1000 * arriving:.3f} mm); the shear layer "
        "separates across that jump"
    )
    # Grading civil: steps grow inward (descending radius) by a bounded ratio.
    inward = steps[::-1]
    ratios = inward[1:] / inward[:-1]
    assert float(ratios.max()) < 1.5, (
        f"base ring growth reaches {ratios.max():.2f}; the grading is a cliff again"
    )
