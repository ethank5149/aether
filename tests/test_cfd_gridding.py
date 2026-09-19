"""LAURA's sizing rules, and what they say about the mesh we have.

Defaults are NASA's shipped values from ``algnshk_vars.strt``; the tests pin
them so a change reads as a deliberate departure rather than a drifting
constant.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aether.cfd.gridding import (
    grid_requirement,
    outer_boundary_distance,
    wall_spacing,
)

#: Post-shock stagnation pressure at Mach 8, 45 km (Rayleigh pitot x p_inf).
STAGNATION_PRESSURE = 149.101418 * 82.87


def test_wall_spacing_inverts_the_cell_reynolds_definition() -> None:
    """Re_cell = rho a Ds / mu, so Ds = Re_cell mu / (rho a) exactly."""
    spacing = wall_spacing(STAGNATION_PRESSURE, 300.0, cell_reynolds=1.0)
    density = STAGNATION_PRESSURE / (287.058 * 300.0)
    sound = math.sqrt(1.4 * 287.058 * 300.0)
    viscosity = 1.716e-5 * (300.0 / 273.15) ** 1.5 * (273.15 + 110.4) / (300.0 + 110.4)
    assert spacing == pytest.approx(viscosity / (density * sound))
    # Sub-micron, which is the whole point.
    assert 1e-7 < spacing < 1e-6


def test_wall_spacing_is_linear_in_the_requested_cell_reynolds() -> None:
    """Relaxing to Re_cell = 10 buys exactly one order, not more."""
    tight = wall_spacing(STAGNATION_PRESSURE, 300.0, cell_reynolds=1.0)
    loose = wall_spacing(STAGNATION_PRESSURE, 300.0, cell_reynolds=10.0)
    assert loose == pytest.approx(10.0 * tight)


def test_wall_spacing_tightens_with_density_so_it_cannot_be_a_constant() -> None:
    """A hand-typed wall length is wrong somewhere in any real envelope.

    Denser air at lower altitude needs a finer cell for the same cell Reynolds
    number, and across this vehicle's altitude band the requirement moves by
    orders of magnitude.
    """
    thin = wall_spacing(STAGNATION_PRESSURE, 300.0)
    thick = wall_spacing(STAGNATION_PRESSURE * 37.0, 300.0)  # ~45 km -> ~20 km
    assert thick == pytest.approx(thin / 37.0, rel=1e-9)
    assert thin / thick > 30.0


def test_outer_boundary_follows_laura_fsh() -> None:
    """fsh = 0.8 puts the boundary at 1.25 standoffs, per algnshk_vars.strt."""
    assert outer_boundary_distance(0.00903) == pytest.approx(0.00903 / 0.8)
    assert outer_boundary_distance(0.00903, 0.8) == pytest.approx(0.0112875)
    for bad in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="shock fraction"):
            outer_boundary_distance(0.00903, bad)


def test_the_scale_ratio_is_what_rules_out_isotropic_cells() -> None:
    """One wall-normal column spans four to five orders of magnitude.

    Cubing that is the isotropic cell count, and it is astronomically
    unaffordable -- which is why stretched wall-normal cells are structural
    here and not an optimisation.
    """
    requirement = grid_requirement(0.00903, STAGNATION_PRESSURE, 300.0)
    assert requirement.scale_ratio > 1e4
    isotropic = (requirement.outer_boundary / requirement.wall_cell) ** 3
    assert isotropic > 1e12
    # Stretched, the same column is an ordinary number of cells.
    assert 60 < requirement.cells_across_layer < 300


def test_shortfall_reports_how_far_off_the_current_mesh_is() -> None:
    """The 2026-09-11 grid study's finest level, measured."""
    requirement = grid_requirement(0.00903, STAGNATION_PRESSURE, 300.0)
    assert requirement.shortfall(1.7e-3) > 1000.0
    assert requirement.shortfall(requirement.wall_cell) == pytest.approx(1.0)


def test_a_gentler_growth_ratio_costs_cells_monotonically() -> None:
    counts = [
        grid_requirement(0.00903, STAGNATION_PRESSURE, 300.0, growth=g).cells_across_layer
        for g in (1.30, 1.20, 1.15, 1.10)
    ]
    assert counts == sorted(counts)


def test_nonphysical_wall_state_is_refused() -> None:
    for pressure, temperature in ((0.0, 300.0), (1000.0, 0.0), (-1.0, 300.0)):
        with pytest.raises(ValueError, match="positive"):
            wall_spacing(pressure, temperature)


def test_growth_for_fits_exactly_the_requested_layer_count() -> None:
    """Solves sum(first * g^k) = total, so the layer count is the input."""
    from aether.cfd.gridding import growth_for

    first, total = 3.7e-5, 0.0113
    for layers in (20, 55, 200):
        g = growth_for(first, total, layers)
        span = first * (g**layers - 1.0) / (g - 1.0)
        assert span == pytest.approx(total, rel=1e-9)
        assert g > 1.0


def test_growth_for_handles_a_layer_that_needs_no_stretching() -> None:
    """More cells than the thickness needs means g below one, not a failure."""
    from aether.cfd.gridding import growth_for

    # 100 cells of 1e-3 would span 0.1; asked to span 0.05 they must shrink.
    g = growth_for(1e-3, 0.05, 100)
    assert 0.0 < g < 1.0
    assert 1e-3 * (g**100 - 1.0) / (g - 1.0) == pytest.approx(0.05, rel=1e-9)


def test_growth_for_refuses_an_impossible_request() -> None:
    from aether.cfd.gridding import growth_for

    with pytest.raises(ValueError, match="cannot fit"):
        growth_for(1.0, 0.5, 10)
    with pytest.raises(ValueError, match="cannot fit"):
        growth_for(1e-6, 0.01, 1)


def test_refinement_scales_surface_and_normal_by_the_same_ratio() -> None:
    """A study whose refinement ratio differs by axis has no single r.

    Choosing the growth ratio and taking whatever layer count falls out gave a
    normal refinement of 1.28x where the surface refined 1.41x -- measured.
    Setting the layer count and solving for the growth gives both.
    """
    from aether.cfd.extrude import extrude_to_shock
    from aether.cfd.geometries import build_surface
    from aether.cfd.gridding import grid_requirement, growth_for, stagnation_pressure
    from aether.cfd.shock import envelope_for

    geometry = {"nose_radius": 0.06, "length": 2.0, "half_angle": 10.0}
    envelope = envelope_for("sphere-cone", geometry, 8.0)
    base = grid_requirement(
        envelope.standoff, stagnation_pressure(8.0, 149.101418), 300.0, cell_reynolds=100.0
    )

    counts = []
    for level in (0.7071, 1.0, 1.4142):
        surface = build_surface("sphere-cone", resolution=level, **geometry)
        layers = max(2, round(base.cells_across_layer * level))
        growth = growth_for(base.wall_cell, base.outer_boundary, layers)
        extruded = extrude_to_shock(
            surface, envelope, base.wall_cell, layers=layers, growth=growth
        )
        # The wall cell is physics and must not move with resolution.
        heights = np.linalg.norm(extruded.nodes[:, 1] - extruded.nodes[:, 0], axis=1)
        assert np.allclose(heights, base.wall_cell, rtol=1e-9)
        counts.append(extruded.faces.shape[0] * extruded.layers)

    effective = [(n / counts[0]) ** (1 / 3) for n in counts]
    steps = [effective[1] / effective[0], effective[2] / effective[1]]
    # Both axes refine together, which is the property being asserted. The
    # ratio is not exactly root two because the circumferential count is
    # snapped to a multiple of eight so a singularity-free nose cap always
    # fits, and that quantises it -- so a study takes *r* from the cell counts
    # it got rather than the level it asked for. Levels of 0.5, 1 and 2 land
    # exactly; these do not, and are within a few percent.
    assert all(step == pytest.approx(1.4142, rel=0.05) for step in steps)
    assert steps[0] / steps[1] == pytest.approx(1.0, rel=0.12)


def test_doubling_levels_land_exactly_on_the_snapped_counts() -> None:
    """0.5, 1 and 2 give a clean factor of two per direction per level.

    Which is grid doubling, and what a Richardson extrapolation wants.
    """
    from aether.cfd.geometries import build_surface

    geometry = {"nose_radius": 0.06, "length": 2.0, "half_angle": 10.0}
    faces = [
        len(np.asarray(build_surface("sphere-cone", resolution=r, **geometry).faces))
        for r in (0.5, 1.0, 2.0)
    ]
    assert faces[1] / faces[0] == pytest.approx(4.0, rel=0.01)
    assert faces[2] / faces[1] == pytest.approx(4.0, rel=0.01)
