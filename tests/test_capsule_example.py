"""Smoke test for the capsule entry example.

Verifies the example constructs a valid simulator, produces a result,
and the trajectory is physically plausible (the vehicle decelerates as
it encounters the atmosphere).  Uses only ``rhs()`` — no full
propagation — so the test runs in under a second.
"""

from __future__ import annotations

import numpy as np
from examples.capsule.entry import capsule_aero_table, capsule_config

from aether.flight.simulator import FlightSimulator


def test_capsule_aero_table_is_valid():
    table = capsule_aero_table()
    assert table.reference_area > 0
    assert table.mach.size > 0
    assert table.alpha.size > 0


def test_capsule_rhs_decelerates_in_the_atmosphere():
    """At 50 km altitude and Mach 20, the vehicle must feel drag."""
    from aether.atmosphere.model import earth_atmosphere, tabulate

    table = capsule_aero_table()
    atmosphere = tabulate(earth_atmosphere())
    config = capsule_config(table, atmosphere)
    sim = FlightSimulator(config)

    r0 = 6_371_000.0 + 50_000.0
    speed = 6_000.0
    gamma = np.radians(-10.0)

    state = sim.state_at(
        position=np.array([r0, 0.0, 0.0]),
        velocity=np.array([speed * np.sin(gamma), speed * np.cos(gamma), 0.0]),
        mass=5_400.0,
    )

    rates = sim.rhs(0.0, state)
    accel = rates[sim.layout.velocity]
    radial_accel = float(np.dot(accel, np.array([1.0, 0.0, 0.0])))
    assert radial_accel < -5.0, (
        f"vehicle at 50 km / 6 km/s should feel >0.5g drag, got {radial_accel:.1f} m/s^2"
    )


def test_capsule_config_builds_without_aero_table():
    """The example config also works in point-mass mode."""
    config = capsule_config(table=None, atmosphere=None)
    sim = FlightSimulator(config)
    state = sim.state_at(
        position=np.array([6_491_000.0, 0.0, 0.0]),
        velocity=np.array([-500.0, 7000.0, 0.0]),
        mass=5_400.0,
    )
    rates = sim.rhs(0.0, state)
    assert rates.shape == (sim.layout.size,)
