"""Tests for the entry glide as a certified-reachability problem.

The machinery -- enclosures, verified tubes, Arb Jacobians -- is tested
generically in ``aether``'s own suite, against fields with closed-form
solutions. What is tested here is the part that is about a vehicle: that the
glide field is the same field the simulator flies, and that binding it to the
machinery produces bounds the trajectory actually respects.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.integrate

from aether.certification import (
    NUMPY_OPS,
    jacobian_enclosure,
    mean_value_step,
    reachable_step,
    reachable_tube,
    tube_widths,
)
from aether.certification.entry_field import (
    enclose_entry_field,
    entry_field,
    entry_vector_field,
)
from aether.guidance.entry import _R_EARTH, EntryBody, entry_dynamics

_BETA, _LD = 4000.0, 2.5
_BANK = (0.2, 0.6)


def _vehicle():
    return EntryBody(_BETA, _LD)


def _field():
    return entry_vector_field(_BETA, _LD)


def _box():
    return [
        (_R_EARTH + 45e3, _R_EARTH + 45.2e3), (0.0, 0.002), (0.4, 0.402),
        (5200.0, 5210.0), (-0.021, -0.020), (1.20, 1.202),
    ]


def _wide_box():
    return [
        (_R_EARTH + 40e3, _R_EARTH + 50e3), (0.0, 0.4), (0.3, 0.7),
        (4800.0, 5600.0), (-0.05, -0.01), (1.0, 1.4),
    ]


class TestTheGlideFieldIsTheSameField:
    """A second expression of the dynamics is only safe if it is not a second model."""

    @pytest.mark.parametrize("bank", [0.0, 0.4, -0.9])
    def test_it_reproduces_glide_dynamics_bit_for_bit(self, bank):
        """Not `approx`: the same expressions in the same order must agree exactly.

        Anything looser would let the two definitions drift by amounts a
        tolerance hides -- the failure the seven duplicated copies of this field
        already demonstrated once.
        """
        state = np.array([_R_EARTH + 45e3, 0.21, 0.54, 5200.0, -0.024, 1.22])
        generic = np.array(
            entry_field(list(state), bank, _BETA, _LD, NUMPY_OPS), dtype=float
        )
        assert np.array_equal(generic, entry_dynamics(state, bank, _vehicle()))

    def test_it_takes_vehicle_parameters_as_numbers(self):
        """So a certificate can range over coefficient envelopes, not one vehicle."""
        state = [_R_EARTH + 45e3, 0.0, 0.0, 5000.0, -0.02, 1.0]
        blunt = entry_field(state, 0.0, _BETA, 0.5, NUMPY_OPS)
        sharp = entry_field(state, 0.0, _BETA, 3.0, NUMPY_OPS)
        assert sharp[4] > blunt[4]


class TestSoundnessOnTheGlide:
    def test_the_enclosure_contains_every_sampled_evaluation(self):
        enclosure = enclose_entry_field(_wide_box(), _BANK, _BETA, _LD)
        vehicle = _vehicle()
        rng = np.random.default_rng(0)
        for _ in range(3000):
            state = np.array([rng.uniform(lo, hi) for lo, hi in _wide_box()])
            rates = entry_dynamics(state, rng.uniform(*_BANK), vehicle)
            for value, (lower, upper) in zip(rates, enclosure, strict=True):
                assert lower <= value <= upper

    def test_arbitrary_bank_histories_never_leave_the_tube(self):
        """What distinguishes a reachable set from a bundle of trajectories.

        The bank is a time-varying interpolant of random knots inside the
        admissible interval -- deliberately not one of the piecewise-constant
        schedules a footprint sweep uses. A tube built by enclosing the field
        over the whole control interval must contain these too.
        """
        vehicle = _vehicle()
        dt, steps = 0.5, 12
        tube = reachable_tube(_field(), _box(), _BANK, dt, steps)
        rng = np.random.default_rng(0)
        for _ in range(30):
            state = np.array([rng.uniform(lo, hi) for lo, hi in _box()])
            knots = rng.uniform(*_BANK, size=6)
            grid = np.linspace(0.0, steps * dt, 6)
            for k in range(steps):
                bank = float(np.interp(k * dt, grid, knots))
                flown = scipy.integrate.solve_ivp(
                    lambda _t, y, b=bank: entry_dynamics(y, b, vehicle),
                    (0.0, dt), state, rtol=1e-11, atol=1e-11,
                )
                state = flown.y[:, -1]
                for value, (lower, upper) in zip(state, tube[k], strict=True):
                    assert lower <= value <= upper

    @pytest.mark.parametrize("dt", [0.1, 0.5])
    def test_the_mean_value_step_contains_the_flown_solution(self, dt):
        vehicle = _vehicle()
        result = mean_value_step(_field(), _box(), _BANK, dt)
        rng = np.random.default_rng(1)
        for _ in range(300):
            state = np.array([rng.uniform(lo, hi) for lo, hi in _box()])
            bank = rng.uniform(*_BANK)
            flown = scipy.integrate.solve_ivp(
                lambda _t, y, b=bank: entry_dynamics(y, b, vehicle),
                (0.0, dt), state, rtol=1e-11, atol=1e-11,
            )
            for value, (lower, upper) in zip(flown.y[:, -1], result, strict=True):
                assert lower <= value <= upper

    def test_the_jacobian_encloses_the_finite_difference_one(self):
        """Checked against an independent numerical path.

        Finite differences are not how the enclosure is computed -- Arb power
        series are -- so agreement is evidence about both.
        """
        vehicle = _vehicle()
        centre = np.array([_R_EARTH + 45e3, 0.2, 0.5, 5200.0, -0.021, 1.20])
        enclosure = jacobian_enclosure(
            _field(), [(v, v) for v in centre], (0.4, 0.4)
        )
        for j in range(6):
            step = 1e-6 * max(abs(centre[j]), 1.0)
            forward, backward = centre.copy(), centre.copy()
            forward[j] += step
            backward[j] -= step
            column = (
                entry_dynamics(forward, 0.4, vehicle)
                - entry_dynamics(backward, 0.4, vehicle)
            ) / (2.0 * step)
            for i in range(6):
                lower, upper = enclosure[i][j]
                assert lower - 1e-5 <= column[i] <= upper + 1e-5


class TestWhatTheGlideCostsInWrapping:
    def test_the_mean_value_form_is_tighter_on_the_components_that_carry_range(self):
        """Speed and flight path angle; longitude and latitude gain almost nothing."""
        dt = 0.5
        naive = reachable_step(_field(), _box(), _BANK, dt)
        refined = mean_value_step(_field(), _box(), _BANK, dt)
        for i in (3, 4):
            assert (refined[i][1] - refined[i][0]) < 0.9 * (naive[i][1] - naive[i][0])

    def test_the_advantage_compounds_over_a_tube(self):
        dt, steps = 0.5, 12
        naive = [(float(a), float(b)) for a, b in _box()]
        refined = [(float(a), float(b)) for a, b in _box()]
        for _ in range(steps):
            naive = reachable_step(_field(), naive, _BANK, dt)
            refined = mean_value_step(_field(), refined, _BANK, dt)
        gain = (naive[3][1] - naive[3][0]) / (refined[3][1] - refined[3][0])
        assert gain > 2.0

    def test_the_tube_only_grows(self):
        widths = tube_widths(reachable_tube(_field(), _box(), _BANK, 0.5, 10))
        for component in range(widths.shape[1]):
            assert np.all(np.diff(widths[:, component]) >= -1e-12)


class TestRefusalsRatherThanSilentWidening:
    def test_a_box_reaching_below_the_surface_is_refused(self):
        """The float field clamps with `max(h, 0)`; an interval cannot branch on sign."""
        box = _wide_box()
        box[0] = (_R_EARTH - 1000.0, _R_EARTH + 50e3)
        with pytest.raises(ValueError, match="below the surface"):
            enclose_entry_field(box, _BANK, _BETA, _LD)

    def test_a_box_of_the_wrong_length_is_refused(self):
        with pytest.raises(ValueError, match="six components"):
            enclose_entry_field([(0.0, 1.0)] * 4, _BANK, _BETA, _LD)
