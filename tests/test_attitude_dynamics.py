"""The attitude responds to the air, which it never did before.

``out[layout.angular_rate] = 0.0  # torque-free in this configuration`` was the
whole of the rotational dynamics. The quaternion was integrated, so the vehicle
*had* an attitude -- it simply never changed in response to anything. No trim,
no static margin, no pitch oscillation, and no way for the re-entry aerodynamics
the rest of the package computes to act on the body that generates them.
"""

from __future__ import annotations

import numpy as np

from aether.aerodynamics.tables import AeroTable
from aether.flight.simulator import FlightConfiguration, FlightSimulator


def _table(*, cm_alpha: float, name: str = "t") -> AeroTable:
    """A body with a linear pitching moment: statically stable if cm_alpha < 0."""
    mach = np.array([4.0, 8.0, 20.0])
    alpha = np.array([-8.0, -4.0, 0.0, 4.0, 8.0])
    axial = np.full((mach.size, alpha.size), 0.10)
    normal = np.outer(np.ones(mach.size), 0.02 * alpha)
    moment = np.outer(np.ones(mach.size), cm_alpha * alpha)
    return AeroTable(
        name=name, mach=mach, alpha=alpha, axial=axial, normal=normal,
        pitching_moment=moment, reference_area=0.2374, reference_length=2.0,
        solver="synthetic", metadata={"altitude_km": 45.0},
    )


def _simulator(**kwargs) -> FlightSimulator:
    return FlightSimulator(
        FlightConfiguration(
            n_modes=2, nose_radius=0.175, reference_area=0.2374,
            ballistic_coefficient=15000.0, **kwargs
        )
    )


def test_without_an_inertia_the_attitude_is_still_torque_free() -> None:
    """The old behaviour is kept exactly, and reached by not asking for the new.

    A moment coefficient with no inertia has nothing to accelerate. Rather than
    invent one, the configuration says so by leaving it None and the vehicle
    flies as it always did.
    """
    simulator = _simulator(aero_table=_table(cm_alpha=-0.02))
    state = simulator.state_at(
        position=np.array([6_500_000.0, 0.0, 0.0]),
        velocity=np.array([-500.0, 7000.0, 0.0]),
        mass=350.0,
        angular_rate=np.array([0.0, 0.3, 0.0]),
    )
    rates = simulator.rhs(0.0, state)
    assert np.allclose(rates[simulator.layout.angular_rate], 0.0)


def test_a_statically_stable_body_is_pushed_back_toward_trim() -> None:
    """C_m_alpha < 0 is what static stability *means*, and it must show up as a
    moment opposing the incidence rather than as a number in a table nobody
    reads."""
    simulator = _simulator(
        aero_table=_table(cm_alpha=-0.02),
        inertia=(40.0, 400.0, 400.0),
        reference_length=2.0,
    )
    # Nose pitched away from the velocity: a positive incidence in the x-z plane.
    tilt = np.radians(6.0)
    quaternion = np.array([np.cos(0.5 * tilt), 0.0, np.sin(0.5 * tilt), 0.0])
    state = simulator.state_at(
        position=np.array([6_420_000.0, 0.0, 0.0]),
        velocity=np.array([-1000.0, 5000.0, 0.0]),
        mass=350.0,
        quaternion=quaternion,
    )
    rates = simulator.rhs(0.0, state)
    angular_acceleration = rates[simulator.layout.angular_rate]
    assert np.any(np.abs(angular_acceleration) > 1e-9), (
        "a statically stable body at incidence must feel a restoring moment"
    )


def test_the_drag_is_not_applied_twice() -> None:
    """Two force paths, and running both would look like a plausible trajectory
    for a vehicle with half the ballistic coefficient."""
    table = _table(cm_alpha=-0.02)
    position = np.array([6_420_000.0, 0.0, 0.0])
    velocity = np.array([-1000.0, 5000.0, 0.0])

    point_mass = _simulator(aero_table=table)
    six_dof = _simulator(
        aero_table=table, inertia=(40.0, 400.0, 400.0), reference_length=2.0
    )
    # At zero incidence the body axis and the velocity coincide, so the two
    # force models must agree -- and they only can if exactly one of them ran.
    axis = velocity / np.linalg.norm(velocity)
    angle = np.arccos(np.clip(axis @ np.array([1.0, 0.0, 0.0]), -1.0, 1.0))
    rotation = np.cross(np.array([1.0, 0.0, 0.0]), axis)
    rotation = rotation / max(float(np.linalg.norm(rotation)), 1e-12)
    quaternion = np.concatenate(
        [[np.cos(0.5 * angle)], np.sin(0.5 * angle) * rotation]
    )

    accelerations = []
    for simulator in (point_mass, six_dof):
        state = simulator.state_at(
            position=position, velocity=velocity, mass=350.0, quaternion=quaternion
        )
        rates = simulator.rhs(0.0, state)
        accelerations.append(rates[simulator.layout.velocity].copy())

    assert np.allclose(accelerations[0], accelerations[1], rtol=1e-6, atol=1e-9)
