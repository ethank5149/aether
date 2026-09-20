"""Entry footprints: the frame, the hull, and the bank-schedule sweep."""

from __future__ import annotations

import numpy as np
import pytest

from aether.geodesy import GeodeticPosition, great_circle_bearing, great_circle_range
from aether.guidance.entry import EntryBody
from aether.guidance.entry_ocp import EntryPathLimits
from aether.guidance.footprint import (
    assemble_footprint,
    convex_hull,
    entry_footprint,
    hull_contains,
    to_local,
)

_R_EARTH = 6378137.0
_MU = 3.986004418e14

#: Limits loose enough that the sweep is decided by the dynamics, not by them.
LOOSE = EntryPathLimits(max_heat_flux_w_cm2=5000.0, max_g_load=50.0, nose_radius_m=1.0)
BODY = EntryBody(ballistic_coefficient=400.0, lift_to_drag=2.0)


@pytest.fixture
def origin():
    return GeodeticPosition(np.deg2rad(20.0), np.deg2rad(-150.0), 0.0)


@pytest.fixture
def site():
    return GeodeticPosition(np.deg2rad(32.7), np.deg2rad(-117.2), 0.0)


@pytest.fixture
def interface_state(origin, site):
    """An entry body at entry interface, heading for the site."""
    return np.array([
        _R_EARTH + 80.0e3,
        origin.longitude,
        origin.latitude,
        7000.0,
        np.deg2rad(-1.0),
        float(great_circle_bearing(origin, site)),
    ])


class TestFrame:
    def test_point_dead_ahead_has_no_crossrange(self, origin, site):
        heading = float(great_circle_bearing(origin, site))
        downrange, crossrange = to_local(origin, heading, site)
        assert crossrange == pytest.approx(0.0, abs=1.0)
        assert downrange == pytest.approx(float(great_circle_range(origin, site)), rel=1e-9)

    def test_origin_maps_to_zero(self, origin):
        assert to_local(origin, 0.0, origin) == (0.0, 0.0)

    def test_crossrange_sign_follows_bearing(self, origin, site):
        """A point to the right of the nose reads positive crossrange."""
        heading = float(great_circle_bearing(origin, site)) - np.deg2rad(20.0)
        _, crossrange = to_local(origin, heading, site)
        assert crossrange > 0.0


class TestConvexHull:
    def test_square(self):
        pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)]
        hull = convex_hull(pts)
        assert len(hull) == 4
        assert 4 not in hull  # the interior point is not on the hull

    def test_degenerate(self):
        assert len(convex_hull([(0.0, 0.0), (1.0, 1.0)])) == 2

    def test_containment_is_inside_only(self):
        square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        assert hull_contains(square, (0.5, 0.5))
        assert not hull_contains(square, (1.5, 0.5))
        assert not hull_contains(square[:2], (0.5, 0.0))  # no area, contains nothing


class TestEntryFootprint:
    def test_max_downrange_matches_equilibrium_glide(self, interface_state):
        r"""The downrange edge is Sanger's closed-form equilibrium-glide range.

        A wings-level flight is the longest one, and its range has an analytic
        answer, :math:`R = (L/D)(R_E/2)\ln[1/(1 - (V/V_c)^2)]`. The sweep
        includes zero bank, so its maximum downrange has to land on that number;
        if it does not, either the integrator or the sweep is wrong.
        """
        footprint = entry_footprint(
            interface_state, BODY, limits=LOOSE, terminal_speed=300.0,
            n_constant=9, n_reversals=0, step=10.0,
        )
        speed = float(interface_state[3])
        circular = np.sqrt(_MU / float(interface_state[0]))
        analytic = (
            BODY.lift_to_drag * (_R_EARTH / 2.0) * np.log(1.0 / (1.0 - (speed / circular) ** 2))
        )
        assert footprint.max_downrange_m == pytest.approx(analytic, rel=0.05)

    def test_higher_lift_reaches_further(self, interface_state):
        poor = entry_footprint(
            interface_state, EntryBody(400.0, 1.0), limits=LOOSE, terminal_speed=300.0,
            n_constant=5, n_reversals=0, step=10.0,
        )
        good = entry_footprint(
            interface_state, EntryBody(400.0, 2.0), limits=LOOSE, terminal_speed=300.0,
            n_constant=5, n_reversals=0, step=10.0,
        )
        assert good.max_downrange_m > poor.max_downrange_m
        assert good.area_km2 > poor.area_km2

    def test_bank_sweep_is_symmetric_in_crossrange(self, interface_state):
        """Left and right banks are mirror images, so the hull straddles the axis."""
        footprint = entry_footprint(
            interface_state, BODY, limits=LOOSE, terminal_speed=300.0,
            n_constant=9, n_reversals=0, step=10.0,
        )
        crossranges = [footprint.local(p)[1] for p in footprint.samples]
        assert min(crossranges) < 0.0 < max(crossranges)
        assert len(footprint.boundary) >= 3
        assert footprint.area_km2 > 0.0

    def test_impossible_limits_empty_the_footprint_with_a_reason(self, interface_state):
        """A footprint nothing survives still has to say what bound it."""
        footprint = entry_footprint(
            interface_state, BODY,
            limits=EntryPathLimits(max_heat_flux_w_cm2=1.0, max_g_load=50.0, nose_radius_m=1.0),
            terminal_speed=300.0, n_constant=3, n_reversals=0, step=20.0,
        )
        assert footprint.samples == ()
        assert footprint.area_km2 == 0.0
        assert footprint.rejected == 3
        assert footprint.binding_constraints[0] == "heat_flux"
        assert footprint.worst_heat_flux_w_cm2 > 1.0
        assert "worst heat flux" in footprint.notes

    def test_the_footprint_contains_what_it_reaches(self, interface_state, origin):
        footprint = entry_footprint(
            interface_state, BODY, limits=LOOSE, terminal_speed=300.0,
            n_constant=7, n_reversals=2, step=10.0,
        )
        # Reversal schedules land inside the hull the constant banks span.
        interior = [p for p in footprint.samples if p not in footprint.boundary]
        assert interior
        assert any(footprint.contains(p) for p in interior)
        assert all(footprint.distance_to(p) == 0.0 for p in interior if footprint.contains(p))
        # Somewhere behind the body is not reachable.
        behind = GeodeticPosition(origin.latitude - np.deg2rad(10.0), origin.longitude, 0.0)
        assert not footprint.contains(behind)
        assert footprint.distance_to(behind) > 0.0


class TestEmptyFootprint:
    def test_degenerate_footprint_answers_honestly(self, origin):
        empty = assemble_footprint("none", 0.0, origin, 0.0, [], rejected=4)
        assert not empty.contains(origin)
        assert empty.distance_to(origin) == float("inf")
        assert empty.area_km2 == 0.0
        assert "no control history" in empty.notes
