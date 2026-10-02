"""Entry flown from the compiled symbolic field.

The field is proved to be Newton's law by the derivations, and checked
against its expressions by the audit. What is tested here is the flight built
on it: that integrating the compiled routine reproduces an independent
integration of the same physics in other coordinates, that the bookkeeping
beside the motion -- the energy, the heat load, the ground-track arc -- is
what it says it is, and that the pieces a guidance loop uses stop where they
should.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.integrate

pytest.importorskip("sympy")

from aether.certification.rotating_field import EARTH, surface_radius
from aether.flight.entry import (
    EARTH_WGS84,
    EntryModel,
    EntryVehicle,
    geocentric_site,
    interface_state,
)
from aether.guidance.skip import SKIP_OUT_ERROR
from aether.orbital.gravity import gravitational_acceleration
from aether.symbolic import find_c_compiler

FT = 0.3048
_VEHICLE = EntryVehicle(
    ballistic_coefficient=281.0, lift_to_drag=0.3, nose_radius=6.0, max_bank=np.radians(165.0)
)
#: Artemis I's published entry interface, used here only as a realistic state.
_PUBLISHED = (
    400000.0 * FT, np.radians(-120.08071), np.radians(-25.82847),
    36062.6568 * FT, np.radians(-5.66367), np.radians(4.65389),
)


@pytest.fixture(scope="module")
def model():
    return EntryModel(_VEHICLE)


@pytest.fixture(scope="module")
def interface():
    return interface_state(*_PUBLISHED)


def _cartesian(state, rotation_rate):
    """Inertial position and velocity of a model state, frames aligned at this instant."""
    radius, longitude, latitude, speed, gamma, heading = state[:6]
    up = np.array([
        np.cos(latitude) * np.cos(longitude), np.cos(latitude) * np.sin(longitude),
        np.sin(latitude),
    ])
    east = np.array([-np.sin(longitude), np.cos(longitude), 0.0])
    north = np.cross(up, east)
    position = radius * up
    relative = speed * (
        np.cos(gamma) * np.cos(heading) * north + np.cos(gamma) * np.sin(heading) * east
        + np.sin(gamma) * up
    )
    return position, relative + np.cross([0.0, 0.0, rotation_rate], position)


def _model_state(position, velocity, elapsed, rotation_rate):
    """The model's six coordinates of an inertial state, ``elapsed`` seconds after alignment."""
    angle = rotation_rate * elapsed
    turn = np.array([
        [np.cos(angle), np.sin(angle), 0.0], [-np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    fixed = turn @ position
    relative = turn @ (velocity - np.cross([0.0, 0.0, rotation_rate], position))
    radius = np.linalg.norm(fixed)
    up = fixed / radius
    longitude = np.arctan2(fixed[1], fixed[0])
    east = np.array([-np.sin(longitude), np.cos(longitude), 0.0])
    north = np.cross(up, east)
    speed = np.linalg.norm(relative)
    return np.array([
        radius, longitude, np.arcsin(up[2]), speed,
        np.arcsin(relative @ up / speed), np.arctan2(relative @ east, relative @ north),
    ])


class TestTheFlightIsTheFieldIntegrated:
    def test_the_routine_is_compiled_and_is_its_expressions(self, model, interface):
        assert model.backend == ("c" if find_c_compiler() else "numpy")
        deep = model.advance(interface, 0.5, 90.0)
        for state in (interface, deep):
            # The altitude is a difference of two radii near 6,400 km, so a part
            # in 1e16 of either is a part in 1e11 of the density.
            assert model.audit(state, 0.5) < 1e-10

    def test_an_inertial_cartesian_integration_lands_in_the_same_place(self, model, interface):
        """Newton's law in an inertial frame, with the kernel's Cartesian gravity.

        Shares with the model only the density it reads and the definition of
        altitude: the frame, the coordinates, the gravity routine and the
        integrator are all different. Two hundred seconds through the first
        pass, banked, where every term is exercised.
        """
        bank, duration = 0.9, 200.0
        rate = EARTH.rotation_rate
        vehicle = model.vehicle

        def acceleration(t, y):
            position, velocity = y[:3], y[3:]
            angle = rate * t
            turn = np.array([
                [np.cos(angle), np.sin(angle), 0.0], [-np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ])
            fixed = turn @ position
            gravity = turn.T @ gravitational_acceleration(fixed)
            relative = velocity - np.cross([0.0, 0.0, rate], position)
            speed = np.linalg.norm(relative)
            radius = np.linalg.norm(position)
            up, along = position / radius, relative / speed
            height = radius - surface_radius(np.arcsin(fixed[2] / radius), planet=EARTH_WGS84)
            probe = np.array([EARTH.radius + height, 0.0, 0.0, 1.0, 0.0, 0.0])
            density = EntryModel(vehicle, planet=EARTH.spherical().non_rotating()).outputs(
                probe
            )["density"]
            drag = density * speed**2 / (2.0 * vehicle.ballistic_coefficient)
            lift = vehicle.lift_to_drag * drag
            normal = up - (up @ along) * along
            normal /= np.linalg.norm(normal)
            lateral = np.cross(along, normal)
            aero = -drag * along + lift * (np.cos(bank) * normal + np.sin(bank) * lateral)
            return np.concatenate([velocity, gravity + aero])

        position, velocity = _cartesian(interface, rate)
        solution = scipy.integrate.solve_ivp(
            acceleration, (0.0, duration), np.concatenate([position, velocity]),
            method="DOP853", rtol=1e-11, atol=1e-6,
        )
        expected = _model_state(solution.y[:3, -1], solution.y[3:, -1], duration, rate)
        flown = model.advance(interface, bank, duration)[:6]
        assert flown[0] == pytest.approx(expected[0], abs=0.05)  # metres of radius
        assert flown[1:3] == pytest.approx(expected[1:3], abs=1e-8)  # 6 cm on the ground
        assert flown[3] == pytest.approx(expected[3], abs=1e-4)
        assert flown[4:6] == pytest.approx(expected[4:6], abs=1e-8)

    def test_the_energy_lost_is_the_drag_power_integrated(self, model, interface):
        """The Jacobi integral falls at D V and at nothing else, whatever the bank."""
        times, rows = [0.0], [model.initial_state(interface)]
        for step, bank in enumerate(np.linspace(-1.2, 1.2, 960)):
            rows.append(model.advance(rows[-1], float(bank), 0.25))
            times.append(0.25 * (step + 1))
        table = model.output_table(np.asarray(rows))
        dissipated = scipy.integrate.simpson(table["drag_power"], x=np.asarray(times))
        lost = table["energy"][0] - table["energy"][-1]
        assert lost > 1.0e7
        # What is left is the quadrature: the standard's lapse rate has corners,
        # and Simpson's rule sees them.
        assert dissipated == pytest.approx(lost, rel=1e-6)

    def test_the_heat_load_is_the_heat_flux_integrated(self, model, interface):
        times, rows = [0.0], [model.initial_state(interface)]
        for step in range(180):
            rows.append(model.advance(rows[-1], 0.3, 1.0))
            times.append(step + 1.0)
        samples = np.asarray(rows)
        flux = model.output_table(samples)["heat_flux"]
        assert samples[-1, 7] == pytest.approx(
            scipy.integrate.simpson(flux, x=np.asarray(times)), rel=1e-6
        )
        # ... and the arc is the ground track: V cos(gamma) / r, integrated.
        rate = samples[:, 3] * np.cos(samples[:, 4]) / samples[:, 0]
        assert samples[-1, 6] == pytest.approx(
            scipy.integrate.simpson(rate, x=np.asarray(times)), rel=1e-9
        )

    def test_the_interface_is_at_its_published_altitude(self, model, interface):
        """400 kft geodetic, to the 0.4 m by which height along the radius exceeds it."""
        assert model.altitude(interface) == pytest.approx(400000.0 * FT, abs=0.5)
        assert model.altitude(interface) > 400000.0 * FT

    def test_the_two_readings_of_the_published_angle_differ_by_the_deflection(self, interface):
        geocentric = interface_state(*_PUBLISHED, geodetic_horizon=False)
        assert np.degrees(interface[4] - geocentric[4]) == pytest.approx(0.148, abs=0.003)
        assert interface[:3] == pytest.approx(geocentric[:3], rel=1e-15)

    def test_a_ground_point_is_reported_in_geodetic_latitude(self, model, interface):
        point = model.ground_point(interface)
        assert np.degrees(point.latitude) == pytest.approx(-25.82847, abs=3e-3)
        assert np.degrees(point.longitude) == pytest.approx(-120.08071, abs=1e-9)
        site = geocentric_site(np.radians(-118.10181), np.radians(27.34852))
        assert np.degrees(site[1]) == pytest.approx(27.19179, abs=1e-4)


class TestWhatTheGuidanceUsesOfIt:
    def test_a_prediction_ends_at_the_terminal_speed_or_says_it_escaped(self, model, interface):
        site = geocentric_site(np.radians(-118.10181), np.radians(27.34852))
        lift_down = model.predict(interface, 0.0, site, -0.5, terminal_speed=1000.0 * FT)
        assert -6.0e6 < lift_down.range_error < -3.0e6  # falls short by thousands of km
        assert 100.0 < lift_down.flight_time < 400.0
        lift_up = model.predict(interface, 0.0, site, 1.0, terminal_speed=1000.0 * FT)
        assert lift_up.range_error == SKIP_OUT_ERROR

    def test_more_vertical_lift_flies_further(self, model, interface):
        """The ordering the corrector's bracketing rests on."""
        site = geocentric_site(np.radians(-118.10181), np.radians(27.34852))
        errors = [
            model.predict(interface, 0.0, site, u, terminal_speed=1000.0 * FT).range_error
            for u in (-0.8, -0.4, 0.0, 0.2)
        ]
        assert errors == sorted(errors)

    def test_a_prediction_is_the_plant_flown_without_lateral_lift(self, model, interface):
        """Lift straight up, where there is none to leave out: the two must agree."""
        site = (float(interface[1]), float(interface[2]))
        steep = interface.copy()
        steep[4] = np.radians(-7.5)  # steep enough to come down without a skip
        predicted = model.predict(steep, 0.0, site, 1.0, terminal_speed=1000.0 * FT)
        flown = model.fly_schedule(
            steep, [(np.inf, 0.0)], terminal_speed=1000.0 * FT, max_time=3000.0, step=10.0
        )
        assert flown is not None
        assert predicted.flight_time == pytest.approx(flown.times[-1], abs=0.5)
        assert predicted.range_error == pytest.approx(flown.final[6] * EARTH.radius, rel=2e-4)

    def test_a_schedule_is_not_flown_past_its_end(self, model, interface):
        """Lift-down ends in a dive toward the vertical, where the heading has no rate."""
        flown = model.fly_schedule(
            interface, [(np.inf, np.radians(165.0))], terminal_speed=1000.0 * FT,
            max_time=2000.0, step=10.0,
        )
        assert flown is not None
        assert flown.final[3] == pytest.approx(1000.0 * FT, abs=0.05)
        assert flown.times[-1] < 200.0 and flown.times[-1] % 10.0 != 0.0
        assert flown.load_factor is not None and flown.load_factor.max() > 20.0

    def test_a_thinner_atmosphere_is_a_number_and_not_another_field(self, interface):
        """Two bodies, one compiled routine: the density scale is a parameter."""
        thin = EntryModel(
            EntryVehicle(281.0, 0.3, 6.0, np.radians(165.0), density_scale=0.9)
        )
        standard = EntryModel(_VEHICLE)
        deep = standard.advance(interface, 0.5, 80.0)
        assert thin.outputs(deep)["density"] == pytest.approx(
            0.9 * standard.outputs(deep)["density"], rel=1e-14
        )
        assert thin.sensed(deep)[0] == pytest.approx(0.9 * standard.sensed(deep)[0], rel=1e-14)
