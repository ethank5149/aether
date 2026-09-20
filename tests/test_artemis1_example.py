"""The Artemis I example: its derived aerodynamics, its flight, and its footprints."""

from __future__ import annotations

import numpy as np
import pytest
from examples.artemis1.entry import (
    NMI,
    certified_outer_bound,
    fly,
    footprints,
    orion_aerodynamics,
)


@pytest.fixture(scope="module")
def aero():
    return orion_aerodynamics()


@pytest.fixture(scope="module")
def flown(aero):
    return fly(aero)


def test_the_design_lift_lands_inside_the_published_trim_band(aero):
    """L/D 0.30 was the only input; the database puts trim at 157-162 degrees."""
    assert 157.0 <= 180.0 - np.rad2deg(aero.angle_of_attack) <= 162.0


def test_six_reversals_like_the_flight(flown):
    """Artemis I reversed six times (AAS 24-174, Table 3). The corridor
    coefficients are fitted to that count, so this pins the fit. The timing is
    not fitted: with the body rolling at 15 deg/s the reversals land a mean
    28 s from the flight's, against 53 s when the bank was instantaneous."""
    result, _ = flown
    assert result.reversals == 6


def test_the_flight_meets_the_landing_requirement(flown):
    result, _ = flown
    assert result.miss_distance < 5.4 * NMI


def test_the_target_stays_reachable_as_the_footprint_shrinks(flown, aero):
    result, _ = flown
    prints = footprints(result, aero)
    areas = [fp.area_km2 for _, _, fp, _ in prints]
    assert all(fp.contains(target) for _, _, fp, target in prints)
    assert areas == sorted(areas, reverse=True)


def test_the_certified_bound_contains_what_was_flown_and_what_was_swept(flown, aero):
    """The sandwich: a sampled sweep bounds the reachable set from inside, the
    certificate bounds it from outside, and the flown trajectory sits between.
    A sample outside the certificate would falsify one of the two, which is the
    only check that can catch an error in both."""
    result, _ = flown
    bound, distance, _ceiling = certified_outer_bound(result, aero)
    inner = next(
        fp.max_downrange_m
        for label, _, fp, _ in footprints(result, aero)
        if label == "final phase"
    )
    assert distance <= inner <= bound.max_downrange_m
