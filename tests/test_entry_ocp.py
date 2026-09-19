"""Tests for the entry glide posed as an optimal control problem.

There is no closed form for a constrained lifting entry, so these check the
pieces against things derived independently: Sanger's equilibrium-glide range,
the unscaled dynamics, and the physics of the Sutton-Graves correlation.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.integrate

from aether.guidance.entry import (
    _R_EARTH,
    EntryBody,
    EntryState,
    entry_dynamics,
)
from aether.guidance.entry_ocp import (
    _V_C,
    EntryPathLimits,
    EntryScaling,
    bank_argmin,
    entry_path_constraints,
    flown_guess,
    g_load,
    max_range_problem,
    sanger_range,
    stagnation_heat_flux,
)


def _vehicle():
    return EntryBody(ballistic_coefficient=4000.0, lift_to_drag=2.5)


def _initial(speed=6000.0, altitude=60_000.0):
    return EntryState(
        radius=_R_EARTH + altitude, longitude=0.0, latitude=0.0,
        speed=speed, flight_path_angle=np.deg2rad(-0.5),
        heading=np.deg2rad(90.0),
    )


def _arc(physical):
    """Great-circle distance from the origin of the frame."""
    return _R_EARTH * np.arccos(
        np.clip(np.cos(physical[2]) * np.cos(physical[1]), -1.0, 1.0)
    )


class TestScaling:
    def test_it_round_trips(self):
        scaling = EntryScaling()
        state = _initial().as_array()
        assert np.allclose(
            scaling.to_physical(scaling.to_scaled(state)), state, rtol=1e-15
        )

    def test_the_angles_are_left_alone(self):
        """Longitude, latitude, flight path angle and heading are already O(1)."""
        assert np.allclose(EntryScaling().factors[[1, 2, 4, 5]], 1.0)

    def test_the_scaled_dynamics_are_the_similarity_transform(self):
        r""":math:`\tilde f = S^{-1} f(S\tilde x)`, checked against the unscaled call."""
        vehicle, scaling = _vehicle(), EntryScaling()
        state = _initial().as_array()
        scaled = scaling.to_scaled(state)
        got = scaling.dynamics(vehicle)(scaled, np.array([0.3]))
        expected = entry_dynamics(state, 0.3, vehicle) / scaling.factors
        assert np.allclose(got, expected, rtol=1e-15)

    def test_the_scaled_states_are_of_order_one(self):
        """The reason for the transform: SLSQP works on the raw variables."""
        scaled = EntryScaling().to_scaled(_initial().as_array())
        assert 0.5 < abs(scaled[0]) < 2.0
        assert 0.0 < abs(scaled[3]) < 2.0


class TestSangerRange:
    def test_it_matches_the_closed_form(self):
        vehicle = _vehicle()
        speed = 6000.0
        expected = (
            vehicle.lift_to_drag * 0.5 * _R_EARTH
            * np.log(1.0 / (1.0 - (speed / _V_C) ** 2))
        )
        assert sanger_range(vehicle, speed) == pytest.approx(expected, rel=1e-12)

    def test_range_scales_with_lift_to_drag(self):
        """L/D enters linearly, which is the whole reason a glider outranges an RV."""
        low = sanger_range(EntryBody(4000.0, 1.0), 6000.0)
        high = sanger_range(EntryBody(4000.0, 3.0), 6000.0)
        assert high == pytest.approx(3.0 * low, rel=1e-12)

    def test_it_refuses_circular_speed(self):
        """At orbital speed the glide does not descend and the range diverges."""
        with pytest.raises(ValueError, match="equilibrium glide"):
            sanger_range(_vehicle(), _V_C)

    def test_a_flown_glide_agrees_with_it(self):
        """The independent check the plan asked for, as a number.

        Sanger assumes an unconstrained equilibrium glide at zero bank. Flying
        the actual six-state EOM at zero bank must land in the same place; the
        two share no code, so agreement is evidence about both.
        """
        vehicle = _vehicle()
        states, _, _ = flown_guess(
            vehicle, _initial(6000.0), 1500.0, bank=0.0, n_nodes=20
        )
        flown = _arc(EntryScaling().to_physical(states[-1]))
        closed = sanger_range(vehicle, 6000.0) - sanger_range(vehicle, 1500.0)
        assert flown == pytest.approx(closed, rel=0.05)

    def test_the_closed_form_is_the_looser_bound(self):
        """Sanger glides to zero speed unconstrained, so it cannot be beaten by much.

        A solved trajectory landing far *above* it would mean the transcription
        is admitting something the physics does not.
        """
        vehicle = _vehicle()
        states, _, _ = flown_guess(
            vehicle, _initial(6000.0), 1500.0, bank=0.0, n_nodes=20
        )
        flown = _arc(EntryScaling().to_physical(states[-1]))
        assert flown < sanger_range(vehicle, 6000.0)


class TestPathConstraints:
    """The sign convention, which was documented backwards and does not fail loudly."""

    def test_feasible_states_are_positive(self):
        """``g >= 0`` is feasible -- SLSQP's ``"ineq"`` convention.

        The regression guard on the real bug: ``OCPProblem`` documented
        ``c <= 0`` while the transcription enforced ``g >= 0`` without a sign
        change, which turns every ceiling into a requirement to *exceed* it.
        A comfortable glide at 60 km must show margin on all three rows.
        """
        vehicle, scaling = _vehicle(), EntryScaling()
        constraints = entry_path_constraints(vehicle, EntryPathLimits(), scaling)
        margins = constraints(scaling.to_scaled(_initial().as_array()), np.array([0.0]))
        assert np.all(margins > 0.0)

    def test_a_violating_state_goes_negative(self):
        """Deep and fast: heat flux and g-load both blow through their limits."""
        vehicle, scaling = _vehicle(), EntryScaling()
        constraints = entry_path_constraints(vehicle, EntryPathLimits(), scaling)
        deep = EntryState(
            _R_EARTH + 22_000.0, 0.0, 0.0, 6000.0, 0.0, 0.0
        ).as_array()
        margins = constraints(scaling.to_scaled(deep), np.array([0.0]))
        assert margins[0] < 0.0      # heat flux
        assert margins[1] < 0.0      # g load

    def test_the_altitude_floor_is_the_third_row(self):
        vehicle, scaling = _vehicle(), EntryScaling()
        limits = EntryPathLimits(min_altitude_m=50_000.0)
        constraints = entry_path_constraints(vehicle, limits, scaling)
        # The nominal start is at 60 km, comfortably above a 50 km floor.
        assert constraints(
            scaling.to_scaled(_initial().as_array()), np.array([0.0])
        )[2] > 0.0
        # Raise the floor above it and the same state is now refused.
        below = entry_path_constraints(
            vehicle, EntryPathLimits(min_altitude_m=70_000.0), scaling
        )
        assert below(
            scaling.to_scaled(_initial().as_array()), np.array([0.0])
        )[2] < 0.0

    def test_the_rows_are_normalised_to_their_own_limits(self):
        """So an SQP solver weights them against each other sensibly."""
        vehicle, scaling = _vehicle(), EntryScaling()
        constraints = entry_path_constraints(vehicle, EntryPathLimits(), scaling)
        margins = constraints(scaling.to_scaled(_initial().as_array()), np.array([0.0]))
        assert np.all(np.abs(margins) < 100.0)

    def test_limits_are_validated(self):
        with pytest.raises(ValueError, match="max_g_load"):
            EntryPathLimits(max_g_load=0.0)
        with pytest.raises(ValueError, match="nose_radius_m"):
            EntryPathLimits(nose_radius_m=-1.0)


class TestHeatFlux:
    def test_it_follows_the_sutton_graves_scaling(self):
        r""":math:`\dot q \propto V^3` at fixed density."""
        base = stagnation_heat_flux(60_000.0, 3000.0, 0.1)
        doubled = stagnation_heat_flux(60_000.0, 6000.0, 0.1)
        assert doubled == pytest.approx(8.0 * base, rel=1e-9)

    def test_a_blunter_nose_heats_less(self):
        r""":math:`\dot q \propto R_n^{-1/2}`, which is why re-entry bodies are blunt."""
        sharp = stagnation_heat_flux(60_000.0, 6000.0, 0.1)
        blunt = stagnation_heat_flux(60_000.0, 6000.0, 0.4)
        assert blunt == pytest.approx(sharp / 2.0, rel=1e-9)

    def test_g_load_is_drag_over_standard_gravity(self):
        vehicle = _vehicle()
        assert g_load(vehicle, 40_000.0, 5000.0) == pytest.approx(
            vehicle.drag_acceleration(40_000.0, 5000.0) / 9.80665, rel=1e-12
        )


class TestFlownGuess:
    def test_it_stops_on_the_terminal_speed(self):
        states, _, _ = flown_guess(_vehicle(), _initial(), 1500.0, n_nodes=20)
        assert EntryScaling().to_physical(states[-1])[3] == pytest.approx(
            1500.0, rel=1e-6
        )

    def test_the_shapes_match_the_node_count(self):
        states, controls, tf = flown_guess(_vehicle(), _initial(), 1500.0, n_nodes=12)
        assert states.shape == (13, 6)
        assert controls.shape == (13, 1)
        assert tf > 0.0

    def test_it_actually_satisfies_the_dynamics(self):
        """The point of a warm start: it begins on the manifold, not near it.

        Re-integrating from the first node must reproduce the last one. The
        cold start it replaces holds longitude constant, so the vehicle never
        moves downrange and every defect is violated at iteration zero.
        """
        vehicle, scaling = _vehicle(), EntryScaling()
        states, _controls, tf = flown_guess(
            vehicle, _initial(), 1500.0, bank=0.2, n_nodes=20
        )
        run = scipy.integrate.solve_ivp(
            lambda _t, y: entry_dynamics(y, 0.2, vehicle),
            (0.0, tf), _initial().as_array(), rtol=1e-11, atol=1e-11,
        )
        terminal = scaling.to_physical(states[-1])
        assert _arc(run.y[:, -1]) == pytest.approx(_arc(terminal), rel=1e-6)

    def test_it_refuses_a_vehicle_already_below_the_terminal_speed(self):
        with pytest.raises(ValueError, match="terminal speed"):
            flown_guess(_vehicle(), _initial(speed=1000.0), 1500.0, n_nodes=10)

    def test_more_bank_costs_range(self):
        """Banking tips lift out of the vertical; the glide falls short."""
        vehicle, scaling = _vehicle(), EntryScaling()
        level, _, _ = flown_guess(vehicle, _initial(), 1500.0, bank=0.0, n_nodes=20)
        banked, _, _ = flown_guess(
            vehicle, _initial(), 1500.0, bank=np.deg2rad(30.0), n_nodes=20
        )
        assert _arc(scaling.to_physical(banked[-1])) < _arc(
            scaling.to_physical(level[-1])
        )


class TestMaxRangeProblem:
    def test_bank_is_the_only_control_and_it_is_bounded(self):
        vehicle = _vehicle()
        problem = max_range_problem(vehicle, _initial(), 1500.0)
        assert problem.nu == 1
        lower, upper = problem.u_bounds
        assert upper[0] == pytest.approx(vehicle.max_bank)
        assert lower[0] == pytest.approx(-vehicle.max_bank)

    def test_only_the_terminal_speed_is_targeted(self):
        """Where the vehicle ends up is the answer, not the question."""
        problem = max_range_problem(_vehicle(), _initial(), 1500.0)
        finite = np.isfinite(problem.xf_target)
        assert finite.tolist() == [False, False, False, True, False, False]
        assert problem.xf_target[3] == pytest.approx(1500.0 / _V_C)

    def test_a_crossrange_requirement_adds_a_latitude_target(self):
        problem = max_range_problem(
            _vehicle(), _initial(), 1500.0, crossrange_m=-1.6e6
        )
        finite = np.isfinite(problem.xf_target)
        assert finite.tolist() == [False, False, True, True, False, False]
        assert problem.xf_target[2] == pytest.approx(-1.6e6 / _R_EARTH)

    def test_the_objective_is_pure_range(self):
        """No time weight: a glide is not trying to be quick."""
        problem = max_range_problem(_vehicle(), _initial(), 1500.0)
        assert problem.time_weight == 0.0
        assert problem.running_cost is not None

    def test_the_running_cost_integrates_to_arc_length(self):
        r""":math:`-\int V\cos\gamma\,\mathrm{d}t / R_E` is the arc flown, in Earth radii."""
        problem = max_range_problem(_vehicle(), _initial(), 1500.0)
        scaled = EntryScaling().to_scaled(_initial().as_array())
        value = problem.running_cost(scaled, np.array([0.0]))
        expected = -6000.0 * np.cos(np.deg2rad(-0.5)) / _R_EARTH
        assert value == pytest.approx(expected, rel=1e-12)


class TestTheDegeneracyWorthKnowingAbout:
    """Why an unconstrained maximum-range solve has nothing to solve."""

    def test_zero_bank_maximises_range(self):
        """Its optimum is known in advance, so it is no test of an optimiser.

        Every degree of bank tips lift out of the vertical, so the
        unconstrained maximum-range glide flies wings-level -- which is exactly
        the trajectory a constant-bank warm start already supplies. Handed its
        own optimum, SLSQP reports "positive directional derivative" without
        moving, which reads as failure and is really a fixed point. A
        crossrange requirement is what makes the problem an optimisation.
        """
        vehicle, scaling = _vehicle(), EntryScaling()
        ranges = []
        for bank_deg in (0.0, 10.0, 20.0, 30.0):
            states, _, _ = flown_guess(
                vehicle, _initial(), 1500.0, bank=np.deg2rad(bank_deg), n_nodes=20
            )
            ranges.append(_arc(scaling.to_physical(states[-1])))
        assert ranges == sorted(ranges, reverse=True)

    def test_crossrange_costs_range(self):
        """The trade the constrained problem is actually making."""
        vehicle, scaling = _vehicle(), EntryScaling()
        level, _, _ = flown_guess(vehicle, _initial(), 1500.0, bank=0.0, n_nodes=20)
        turned, _, _ = flown_guess(
            vehicle, _initial(), 1500.0, bank=np.deg2rad(20.0), n_nodes=20
        )
        assert abs(scaling.to_physical(turned[-1])[2]) > 0.1
        assert _arc(scaling.to_physical(turned[-1])) < _arc(
            scaling.to_physical(level[-1])
        )


class TestBankArgmin:
    """The closed-form Hamiltonian minimiser, and why it is not a clip."""

    @staticmethod
    def _coefficients(costate, x=None):
        """``A`` and ``B`` in ``H(sigma) = const + A cos(sigma) + B sin(sigma)``."""
        vehicle, scaling = _vehicle(), EntryScaling()
        x = scaling.to_scaled(_initial().as_array()) if x is None else x
        physical = scaling.to_physical(x)
        lift = vehicle.lift_to_drag * vehicle.drag_acceleration(
            physical[0] - _R_EARTH, physical[3]
        )
        a = lift * costate[4] / physical[3]
        b = lift * costate[5] / (physical[3] * np.cos(physical[4]))
        return a, b

    def test_it_beats_a_dense_search(self):
        """The claim, checked the slow way at several costates."""
        vehicle, scaling = _vehicle(), EntryScaling()
        argmin = bank_argmin(vehicle, scaling, vehicle.max_bank)
        state = scaling.to_scaled(_initial().as_array())
        grid = np.linspace(-vehicle.max_bank, vehicle.max_bank, 20001)
        for costate in (
            np.array([0.0, 0.0, 0.0, 0.0, -1.0, 0.5]),
            np.array([0.0, 0.0, 0.0, 0.0, 1.0, -2.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.3, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, -0.2, -0.9]),
        ):
            a, b = self._coefficients(costate)
            best = argmin(state, costate)[0]
            values = a * np.cos(grid) + b * np.sin(grid)
            assert a * np.cos(best) + b * np.sin(best) <= values.min() + 1e-9

    def test_it_stays_inside_the_bank_limit(self):
        vehicle, scaling = _vehicle(), EntryScaling()
        argmin = bank_argmin(vehicle, scaling, vehicle.max_bank)
        state = scaling.to_scaled(_initial().as_array())
        rng = np.random.default_rng(0)
        for _ in range(50):
            costate = np.concatenate([np.zeros(4), rng.normal(size=2)])
            assert abs(argmin(state, costate)[0]) <= vehicle.max_bank + 1e-9

    def test_it_does_not_merely_clip_the_stationary_point(self):
        r"""The trap a projected stationarity condition falls into.

        :math:`A\cos\sigma + B\sin\sigma` is a sinusoid, not convex, so when the
        unconstrained minimiser lies outside the bank limit the constrained
        minimum is *not* generally the nearest bound. Clipping would return a
        maximum on the arcs where the costates change sign; comparing endpoints
        against the stationary point is the minimum principle.
        """
        vehicle, scaling = _vehicle(), EntryScaling()
        narrow = np.deg2rad(20.0)
        argmin = bank_argmin(vehicle, scaling, narrow)
        state = scaling.to_scaled(_initial().as_array())
        grid = np.linspace(-narrow, narrow, 20001)
        rng = np.random.default_rng(7)
        clipped_would_differ = 0
        for _ in range(200):
            costate = np.concatenate([np.zeros(4), rng.normal(size=2)])
            a, b = self._coefficients(costate)
            best = argmin(state, costate)[0]
            stationary = np.arctan2(-b, -a)
            if abs(stationary) > narrow:
                clipped_would_differ += 1
                clipped = np.clip(stationary, -narrow, narrow)
                # Whatever clipping gives, the returned value is no worse.
                assert a * np.cos(best) + b * np.sin(best) <= (
                    a * np.cos(clipped) + b * np.sin(clipped) + 1e-12
                )
            values = a * np.cos(grid) + b * np.sin(grid)
            assert a * np.cos(best) + b * np.sin(best) <= values.min() + 1e-9
        assert clipped_would_differ > 0, "the saturated branch was never exercised"

    def test_the_problem_carries_it(self):
        """So the indirect stage picks it up without the caller wiring anything."""
        problem = max_range_problem(_vehicle(), _initial(), 1500.0)
        assert problem.control_argmin is not None

    def test_minimising_control_uses_it(self):
        """``indirect.minimising_control`` must prefer the closed form."""
        from aether.optimal_control.indirect import minimising_control

        problem = max_range_problem(_vehicle(), _initial(), 1500.0)
        scaled = EntryScaling().to_scaled(_initial().as_array())
        costate = np.array([0.0, 0.0, 0.0, 0.0, -1.0, 0.5])
        assert minimising_control(problem, scaled, costate) == pytest.approx(
            problem.control_argmin(scaled, costate)
        )


class TestTheIndirectStageCannotSeedThisProblem:
    """A structural limit found by trying it, recorded rather than worked around."""

    def test_the_covector_estimate_returns_zero(self):
        r"""``covector_estimate`` needs the running cost to depend on the control.

        It solves :math:`\partial L/\partial u + (\partial f/\partial u)^{\mathsf T}
        \lambda = 0` in least squares. Maximum range has
        :math:`L = -V\cos\gamma/R_E`, which does not contain the bank at all, so
        :math:`\partial L/\partial u = 0` and the system is homogeneous — its
        minimum-norm solution is :math:`\lambda \equiv 0`.

        This is the same degeneracy as seeding a backward sweep with
        :math:`\lambda(t_f)=0`, arrived at from the other direction, and it is
        why the two-state validation did not catch it: the minimum-energy double
        integrator has :math:`L = u^2/2`, which does depend on the control.
        """
        from aether.optimal_control.indirect import covector_estimate
        from aether.optimal_control.pseudospectral import OCPSolution

        vehicle = _vehicle()
        states, controls, tf = flown_guess(
            vehicle, _initial(), 1500.0, bank=0.3, n_nodes=15
        )
        problem = max_range_problem(vehicle, _initial(), 1500.0, n_nodes=15)
        seed = OCPSolution(
            x=states, u=controls, t=np.linspace(0.0, tf, 16), tf=tf,
            cost=0.0, success=True, message="", nfev=0, njev=0,
        )
        assert np.linalg.norm(covector_estimate(problem, seed)) == 0.0

    def test_the_running_cost_is_what_makes_the_difference(self):
        """Stated as the property, so a future fix has something to satisfy."""
        problem = max_range_problem(_vehicle(), _initial(), 1500.0)
        scaled = EntryScaling().to_scaled(_initial().as_array())
        step = 1e-6
        gradient = (
            problem.running_cost(scaled, np.array([0.3 + step]))
            - problem.running_cost(scaled, np.array([0.3 - step]))
        ) / (2.0 * step)
        assert gradient == 0.0
