"""Tests for the entry field over a rotating, oblate planet.

The classical six-state field already has a suite. What is tested here is what
this one adds to it -- oblate gravity and the rotating frame -- and, first of
all, that it adds *nothing else*: with those two switched off it must be the
classical field exactly, or it is a second model rather than an extension.

The derivation of the field from Newton's law is not tested here. It is proved
symbolically, in ``aether.symbolic.derivations``; these tests are about the
numbers.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.integrate

from aether.certification import NUMPY_OPS, enclose_field
from aether.certification.entry_field import entry_field
from aether.certification.rotating_field import (
    EARTH,
    Planet,
    exponential_density,
    j2_gravity,
    jacobi_energy,
    point_mass_field,
    rotating_entry_field,
    vertical_balance_terms,
)
from aether.orbital.gravity import EARTH as CARTESIAN_EARTH
from aether.orbital.gravity import gravitational_acceleration

_BETA, _LD = 4000.0, 2.5


def _state(altitude=45e3, latitude=0.54, speed=5200.0, gamma=-0.024, heading=1.22):
    return np.array([EARTH.radius + altitude, 0.21, latitude, speed, gamma, heading])


def _random_states(count, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        yield np.array(
            [
                EARTH.radius + rng.uniform(30e3, 90e3),
                rng.uniform(-3.0, 3.0),
                rng.uniform(-1.3, 1.3),
                rng.uniform(2000.0, 7800.0),
                rng.uniform(-0.3, 0.3),
                rng.uniform(-3.0, 3.0),
            ]
        )


class TestItExtendsTheClassicalFieldRatherThanReplacingIt:
    @pytest.mark.parametrize("bank", [0.0, 0.4, -0.9])
    def test_a_spherical_non_rotating_planet_gives_the_classical_field_bit_for_bit(self, bank):
        """Not `approx`. The added terms must vanish, not merely be small."""
        classical = point_mass_field(_BETA, _LD, planet=EARTH.spherical().non_rotating())
        for state in _random_states(50):
            ours = np.array(classical(list(state), bank, NUMPY_OPS), dtype=float)
            theirs = np.array(entry_field(list(state), bank, _BETA, _LD, NUMPY_OPS), dtype=float)
            assert np.array_equal(ours, theirs)

    def test_rotation_and_oblateness_each_change_it(self):
        """So the test above is not passing because the terms are dead code."""
        state = list(_state())
        full = np.array(point_mass_field(_BETA, _LD)(state, 0.4, NUMPY_OPS))
        no_spin = np.array(
            point_mass_field(_BETA, _LD, planet=EARTH.non_rotating())(state, 0.4, NUMPY_OPS)
        )
        round_earth = np.array(
            point_mass_field(_BETA, _LD, planet=EARTH.spherical())(state, 0.4, NUMPY_OPS)
        )
        assert not np.allclose(full[3:], no_spin[3:], rtol=1e-6)
        assert not np.allclose(full[3:], round_earth[3:], rtol=1e-9)


class TestGravity:
    def test_the_spherical_components_are_the_cartesian_field_resolved_locally(self):
        """Against `aether.orbital.gravity`, which was written and verified independently."""
        rng = np.random.default_rng(1)
        for _ in range(200):
            radius = EARTH.radius + rng.uniform(0.0, 400e3)
            lam, phi = rng.uniform(-np.pi, np.pi), rng.uniform(-1.5, 1.5)
            up = np.array([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)])
            north = np.array([-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)])
            east = np.array([-np.sin(lam), np.cos(lam), 0.0])
            cartesian = gravitational_acceleration(radius * up, CARTESIAN_EARTH)
            g_down, g_north = j2_gravity(radius, phi)
            assert -cartesian @ up == pytest.approx(g_down, rel=1e-12)
            assert cartesian @ north == pytest.approx(g_north, abs=1e-12)
            assert cartesian @ east == pytest.approx(0.0, abs=1e-12)

    def test_the_radial_correction_changes_sign_at_35_degrees(self):
        radius = EARTH.radius + 50e3
        central = EARTH.mu / radius**2
        equator, _ = j2_gravity(radius, 0.0)
        pole, _ = j2_gravity(radius, np.pi / 2)
        null, _ = j2_gravity(radius, np.arcsin(1 / np.sqrt(3)))
        assert equator > central > pole
        assert null == pytest.approx(central, rel=1e-14)
        # Twice as large at the pole as at the equator, and of the opposite sign.
        assert (pole - central) / (equator - central) == pytest.approx(-2.0, rel=1e-10)


class TestEnergy:
    def test_lift_alone_conserves_the_jacobi_energy(self):
        """Lift does no work, in the rotating frame and with oblateness kept."""

        def field(_t, state):
            lift = 6.0
            return rotating_entry_field(list(state), (0.0, lift * 0.8, lift * 0.6), NUMPY_OPS)

        start = _state(altitude=70e3, speed=7000.0, gamma=-0.01)
        flown = scipy.integrate.solve_ivp(field, (0.0, 600.0), start, rtol=1e-12, atol=1e-9)
        energies = np.array([jacobi_energy(list(y)) for y in flown.y.T])
        assert np.ptp(energies) < 1e-9 * abs(energies[0])
        # And the test would notice: the state itself moved a long way.
        assert abs(flown.y[5, -1] - start[5]) > 0.1

    def test_drag_removes_energy_at_the_drag_power(self):
        field = point_mass_field(_BETA, _LD)
        state = _state()
        density = exponential_density()(state[0] - EARTH.radius, NUMPY_OPS)
        drag = 0.5 * density * state[3] ** 2 / _BETA
        step = 1e-3
        advanced = state + step * np.array(field(list(state), 0.4, NUMPY_OPS))
        rate = (jacobi_energy(list(advanced)) - jacobi_energy(list(state))) / step
        assert rate == pytest.approx(-drag * state[3], rel=1e-4)
        assert rate < 0.0


class TestTheGlideBalance:
    def test_the_terms_sum_to_the_lift_that_holds_the_flight_path(self):
        for state in _random_states(100, seed=2):
            terms = vertical_balance_terms(list(state))
            required = sum(terms[k] for k in ("central", "oblate_north", "coriolis", "centrifugal"))
            rates = rotating_entry_field(list(state), (0.0, required, 0.0), NUMPY_OPS)
            assert rates[4] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize(
        ("speed", "latitude_deg", "low", "high"),
        [(7000.0, 0.0, 0.45, 0.55), (7000.0, 45.0, 0.30, 0.40), (5000.0, 0.0, 0.10, 0.15)],
    )
    def test_coriolis_is_tens_of_percent_of_what_lift_balances(
        self, speed, latitude_deg, low, high
    ):
        """Due east at 50 km. Not a refinement of the balance: a leading term of it."""
        state = _state(altitude=50e3, latitude=np.radians(latitude_deg), speed=speed, gamma=0.0,
                       heading=np.pi / 2)
        terms = vertical_balance_terms(list(state))
        assert low < abs(terms["coriolis"]) / terms["central"] < high

    def test_oblateness_is_a_percent_of_it_at_most(self):
        """Largest at the pole and near orbital speed, where the balance itself is small."""
        worst = 0.0
        for latitude in (0.0, 0.6, 1.2, 1.5):
            for speed in (3000.0, 5000.0, 7000.0):
                state = _state(altitude=50e3, latitude=latitude, speed=speed, gamma=0.0,
                               heading=np.pi / 2)
                terms = vertical_balance_terms(list(state))
                worst = max(worst, abs(terms["oblate_radial"]) / terms["central"])
        assert 0.01 < worst < 0.02

    def test_coriolis_dominates_oblateness_away_from_the_poles(self):
        for latitude in (0.0, 0.6, 1.2):
            for speed in (3000.0, 5000.0, 7000.0):
                state = _state(altitude=50e3, latitude=latitude, speed=speed, gamma=0.0,
                               heading=np.pi / 2)
                terms = vertical_balance_terms(list(state))
                assert abs(terms["coriolis"]) > 5.0 * abs(terms["oblate_radial"])


class TestEveryArithmetic:
    def test_a_planet_may_be_given_as_plain_numbers(self):
        """`Planet` is untyped so that it can also be given as balls or symbols."""
        moon = Planet(mu=4.9048695e12, radius=1737.4e3, j2=2.027e-4, rotation_rate=2.6617e-6)
        rates = rotating_entry_field(
            [moon.radius + 20e3, 0.0, 0.3, 1600.0, -0.01, 1.0], (0.0, 0.0, 0.0), planet=moon
        )
        assert np.all(np.isfinite(rates))

    def test_the_arb_enclosure_contains_every_sampled_evaluation(self):
        pytest.importorskip("flint")
        box = [
            (EARTH.radius + 40e3, EARTH.radius + 50e3), (0.0, 0.4), (0.3, 0.7),
            (4800.0, 5600.0), (-0.05, -0.01), (1.0, 1.4),
        ]
        bank = (0.2, 0.6)
        field = point_mass_field(_BETA, _LD)
        enclosure = enclose_field(field, box, bank)
        rng = np.random.default_rng(3)
        for _ in range(2000):
            state = [rng.uniform(low, high) for low, high in box]
            rates = field(state, rng.uniform(*bank), NUMPY_OPS)
            for value, (lower, upper) in zip(rates, enclosure, strict=True):
                assert lower <= value <= upper

    def test_an_enclosure_reaching_below_the_surface_is_refused(self):
        with pytest.raises(ValueError, match="below the surface"):
            point_mass_field(_BETA, _LD)(list(_state(altitude=-10.0)), 0.0, NUMPY_OPS)
