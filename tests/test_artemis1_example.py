"""The Artemis I example: its derived aerodynamics, its flight, and its footprints."""

from __future__ import annotations

import numpy as np
import pytest
from examples.artemis1.entry import (
    FLOWN,
    FT,
    NMI,
    PREDICTED,
    certified_corridor,
    certified_outer_bound,
    flown_model,
    fly,
    footprints,
    guidance_model,
    interface,
    orion_aerodynamics,
)

from aether.atmosphere.model import earth_atmosphere, tabulate
from aether.symbolic import find_c_compiler


@pytest.fixture(scope="module")
def aero():
    return orion_aerodynamics()


@pytest.fixture(scope="module")
def flown(aero):
    return fly(aero)


@pytest.fixture(scope="module")
def prints(flown, aero):
    result, _ = flown
    return footprints(result, aero)


def _reversal_times(result):
    flips = np.nonzero(np.diff(np.sign(result.bank)))[0] + 1
    return result.times[flips]


def test_the_design_lift_lands_inside_the_published_trim_band(aero):
    """L/D 0.30 was the only input; the database puts trim at 157-162 degrees."""
    assert 157.0 <= 180.0 - np.rad2deg(aero.angle_of_attack) <= 162.0


def test_the_flight_is_flown_by_the_compiled_symbolic_field(flown, aero):
    """The plant is the routine printed from the symbolic field, and matches its expressions.

    It carries eight components -- the six of the entry state, the ground-track
    arc and the heat load -- which is how one can tell it from the hand-written
    equations, whose state is the six.
    """
    result, _ = flown
    plant = flown_model(aero)
    assert plant.backend == ("c" if find_c_compiler() else "numpy")
    assert result.plant_states is not None and result.plant_states.shape[1] == 8
    for sample in result.plant_states[:: max(1, len(result.times) // 12)]:
        assert plant.audit(sample, 0.4) < 1e-10


def test_the_entry_interface_is_the_published_one_in_the_fields_coordinates(aero):
    """400 kft above the ellipsoid; slower and shallower than the inertial figures.

    Relative to the rotating Earth the body is 26 m/s slower than its inertial
    36,062.66 ft/s, and against the geocentric horizon it descends 0.13 degrees
    less steeply than the published 5.664 against the geodetic one.
    """
    start = interface()
    assert guidance_model(aero).altitude(start) == pytest.approx(400000.0 * FT, abs=0.5)
    assert start[3] == pytest.approx(36062.6568 * FT - 26.2, abs=0.5)
    assert np.rad2deg(start[4]) == pytest.approx(-5.530, abs=0.002)


def test_six_reversals_like_the_flight(flown):
    """Artemis I reversed six times (AAS 24-174, Table 3). The corridor
    coefficients are fitted to that count, so this pins the fit; the fitted
    quantity is the mean distance of the six times from the flight's, 27 s."""
    result, _ = flown
    assert result.reversals == 6
    difference = np.abs(_reversal_times(result) - np.asarray(FLOWN["reversal times (s)"]))
    assert difference.mean() < 30.0


def test_the_reversals_fall_where_nasas_own_simulation_put_them(flown):
    """Not fitted, and closer than the fit: a mean of 6 s from NASA's prediction.

    The paper's best simulation missed the flight's second reversal by 142 s,
    and says why: it comes in thin air on the way out, where its timing is
    sensitive to everything. This simulation puts it where that one did.
    """
    result, _ = flown
    difference = np.abs(_reversal_times(result) - np.asarray(PREDICTED["reversal times (s)"]))
    assert difference.mean() < 10.0
    assert difference.max() < 20.0


def test_the_skip_is_within_a_few_percent_of_the_flights(flown):
    """Apogee 5 % under the flight's, the Final phase 10 % early, Terminal 2 % early."""
    result, _ = flown
    at = {phase.value: t for t, phase in result.phases}
    assert result.skip_apogee_altitude / FT / 1e3 == pytest.approx(
        FLOWN["skip apogee (kft)"], rel=0.06
    )
    assert at["final"] == pytest.approx(FLOWN["final phase (s)"], rel=0.11)
    assert at["terminal"] == pytest.approx(FLOWN["terminal phase (s)"], rel=0.03)


def test_the_flight_meets_the_landing_requirement(flown):
    result, _ = flown
    assert result.miss_distance < 5.4 * NMI


def test_the_target_stays_reachable_as_the_footprint_shrinks(prints):
    areas = [fp.area_km2 for _, _, fp, _ in prints]
    assert all(fp.contains(target) for _, _, fp, target in prints)
    assert areas == sorted(areas, reverse=True)


def test_the_certified_bound_contains_what_was_flown_and_what_was_swept(flown, aero, prints):
    """The sandwich: a sampled sweep bounds the reachable set from inside, the
    certificate bounds it from outside, and the flown trajectory sits between.
    A sample outside the certificate would falsify one of the two, which is the
    only check that can catch an error in both."""
    result, _ = flown
    bound, distance, _ceiling = certified_outer_bound(result, aero)
    inner = next(fp.max_downrange_m for label, _, fp, _ in prints if label == "final phase")
    assert distance <= inner <= bound.max_downrange_m


def test_the_air_the_flight_met_is_in_the_class_the_bound_is_certified_for(flown, aero):
    """The certificate covers atmospheres 0.85 to 1.15 times the tabulated one.

    The flight flew through 0.9 times the symbolic atmosphere, which is the
    standard exactly where the tabulation interpolates it. Across the
    corridor the ratio of the two has to stay inside the class, and does with
    room to spare.
    """
    result, _ = flown
    corridor = certified_corridor(result)
    plant = flown_model(aero)
    table = tabulate(earth_atmosphere())
    heights = np.linspace(corridor.altitude_floor_m, corridor.altitude_ceiling_m, 2601)
    radius = float(plant.planet.radius)
    probes = np.column_stack([
        radius + heights, np.zeros_like(heights), np.zeros_like(heights),
        np.ones_like(heights), np.zeros_like(heights), np.zeros_like(heights),
    ])
    flown_density = plant.output_table(probes)["density"]
    # At the equator the ellipsoid is the equatorial radius, so the probes are
    # at the altitudes asked for.
    assert plant.output_table(probes)["altitude"] == pytest.approx(heights, abs=1e-6)
    ratio = flown_density / np.asarray(table.state(heights).density)
    assert ratio.min() > 0.85 and ratio.max() < 1.15
    assert ratio == pytest.approx(0.9, rel=1e-3)
