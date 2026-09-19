"""Declarative phase-termination triggers as interior boundary conditions."""

from __future__ import annotations

import numpy as np
import pytest

from aether.optimal_control.pseudospectral import (
    OCPProblem,
    _boundary_constraints,
    solve_ocp,
)
from aether.trajectory.triggers import (
    PhaseTrigger,
    TriggerParameter,
    altitude_trigger,
    mass_trigger,
    speed_trigger,
    state_component_trigger,
)

_R = 6.371e6


class TestTriggerResidual:
    def test_altitude_residual(self):
        trig = altitude_trigger(600.0e3)
        x = np.array([_R + 600e3, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert trig.residual(x) == pytest.approx(0.0, abs=1e-6)
        high = np.array([_R + 700e3, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert trig.residual(high) == pytest.approx(100.0e3, rel=1e-9)

    def test_speed_residual(self):
        trig = speed_trigger(7000.0)
        x = np.array([_R, 0.0, 0.0, 0.0, 7000.0, 0.0])
        assert trig.residual(x) == pytest.approx(0.0, abs=1e-6)

    def test_mass_and_component_residual(self):
        trig = mass_trigger(mass_index=6, burnout_mass=15000.0)
        x = np.array([_R, 0.0, 0.0, 0.0, 0.0, 0.0, 15000.0])
        assert trig.residual(x) == pytest.approx(0.0, abs=1e-6)
        comp = state_component_trigger(6, 12000.0)
        assert comp.residual(x) == pytest.approx(3000.0, rel=1e-9)

    def test_derived_trigger_needs_no_index_component_does(self):
        assert speed_trigger(100.0).component_index is None
        with pytest.raises(ValueError, match="component_index"):
            PhaseTrigger(TriggerParameter.MASS, 1.0)


class TestTriggerInSolver:
    def test_empty_triggers_leave_the_problem_unchanged(self):
        # A problem with no terminal constraints must solve identically whether
        # the (empty) tuple is passed or defaulted.
        def dyn(x, u):
            return np.array([x[2], x[3], u[0], u[1]])

        def path(x, u):
            return np.array([1.0])

        common = {
            "dynamics": dyn, "path_constraints": path,
            "x0": np.array([0.0, 0.0, 50.0, 0.0]),
            "xf_target": np.array([1000.0, 0.0, np.inf, np.inf]),
            "nx": 4, "nu": 2, "N": 15, "tf_guess": 20.0,
            "u_bounds": (np.array([-20.0, -20.0]), np.array([20.0, 20.0])),
        }
        a = solve_ocp(OCPProblem(**common))
        b = solve_ocp(OCPProblem(**common, terminal_constraints=()))
        assert np.allclose(a.x[-1], b.x[-1], atol=1e-9)

    def test_trigger_enters_the_boundary_constraint_vector(self):
        # White-box: the trigger residual g(x_terminal) is appended to the
        # equality boundary constraints the NLP drives to zero, so the solver
        # closes the phase exactly when the condition holds.
        def dyn(x, u):
            return np.array([x[2], x[3], u[0], u[1]])

        def path(x, u):
            return np.array([1.0])

        n = 8
        nx = 4
        base = {
            "dynamics": dyn, "path_constraints": path,
            "x0": np.array([0.0, 0.0, 100.0, 0.0]),
            "xf_target": np.full(4, np.inf), "nx": nx, "nu": 2, "N": n,
        }
        # A z with a known terminal state; the trigger fires on component 2 == 100.
        z = np.zeros((n + 1) * nx + (n + 1) * 2 + 1)
        term = np.array([500.0, 0.0, 150.0, 200.0])
        z[n * nx:(n + 1) * nx] = term

        trig = state_component_trigger(2, 100.0).as_constraint()
        without = _boundary_constraints(z, OCPProblem(**base))
        withtrig = _boundary_constraints(
            z, OCPProblem(**base, terminal_constraints=(trig,)))
        # Exactly one extra equality, equal to the residual (150 - 100).
        assert withtrig.size == without.size + 1
        assert withtrig[-1] == pytest.approx(50.0)
