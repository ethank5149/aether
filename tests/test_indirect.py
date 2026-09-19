"""Tests for the indirect (Pontryagin) refinement of a direct solution.

The double integrator with minimum control energy is used throughout because it
has a closed-form optimum, so "did the solver work" is a question with an exact
answer rather than a plausible-looking curve.
"""

from __future__ import annotations

import numpy as np
import pytest

from aether.optimal_control.indirect import (
    HamiltonianSystem,
    control_jacobian,
    costate_dynamics,
    covector_estimate,
    hamiltonian,
    minimising_control,
    refine_indirect,
)
from aether.optimal_control.pseudospectral import OCPProblem, OCPSolution

_T = 2.0


def _problem(u_bounds=None):
    """xdot = v, vdot = u; minimise int u^2/2 from rest to x=1 at rest."""
    return OCPProblem(
        dynamics=lambda x, u: np.array([x[1], u[0]]),
        path_constraints=lambda x, u: np.array([]),
        x0=np.array([0.0, 0.0]),
        xf_target=np.array([1.0, 0.0]),
        nx=2, nu=1, tf_guess=_T,
        running_cost=lambda x, u: 0.5 * float(u[0]) ** 2,
        time_weight=0.0,
        u_bounds=u_bounds,
    )


def _analytic(t):
    """u*(t) = 6/T^2 - 12 t/T^3 and the state it produces."""
    u = 6.0 / _T**2 - 12.0 * t / _T**3
    x = 3.0 * t**2 / _T**2 - 2.0 * t**3 / _T**3
    return u, x


def _seed(scale=0.7, wobble=0.05):
    """A deliberately mediocre direct solution to start from."""
    t = np.linspace(0.0, _T, 41)
    u, x = _analytic(t)
    return OCPSolution(
        x=np.column_stack([x + wobble * np.sin(3.0 * t), np.gradient(x, t)]),
        u=(u * scale).reshape(-1, 1), t=t, tf=_T,
        cost=0.0, success=True, message="seed", nfev=0, njev=0,
    )


class TestHamiltonian:
    def test_it_is_cost_plus_costate_times_dynamics(self):
        problem = _problem()
        state = np.array([0.5, 1.0])
        control = np.array([2.0])
        costate = np.array([3.0, 4.0])
        expected = 0.5 * 4.0 + (3.0 * 1.0 + 4.0 * 2.0)
        assert hamiltonian(problem, state, control, costate) == pytest.approx(expected)

    def test_costate_dynamics_match_the_analytic_partials(self):
        r"""For this problem dH/dx = [0, lambda_x], so lambda-dot = [0, -lambda_x]."""
        problem = _problem()
        rate = costate_dynamics(
            problem, np.array([0.5, 1.0]), np.array([2.0]), np.array([3.0, 4.0])
        )
        assert rate[0] == pytest.approx(0.0, abs=1e-6)
        assert rate[1] == pytest.approx(-3.0, rel=1e-5)

    def test_control_jacobian(self):
        problem = _problem()
        jac = control_jacobian(problem, np.array([0.0, 0.0]), np.array([1.0]))
        assert jac.shape == (2, 1)
        assert jac[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert jac[1, 0] == pytest.approx(1.0, rel=1e-6)


class TestMinimisingControl:
    def test_unbounded_control_solves_stationarity(self):
        r"""H = u^2/2 + lambda_v u is minimised at u = -lambda_v."""
        problem = _problem()
        control = minimising_control(
            problem, np.array([0.0, 0.0]), np.array([0.0, 1.5])
        )
        assert control[0] == pytest.approx(-1.5, abs=1e-5)

    def test_a_bound_is_respected(self):
        """Pontryagin asks for the minimum over the admissible set.

        A stationarity solver would return -1.5 here and violate the bound; the
        minimum principle returns the bound itself, which is where bang-bang
        arcs live.
        """
        problem = _problem(u_bounds=(np.array([-0.5]), np.array([0.5])))
        control = minimising_control(
            problem, np.array([0.0, 0.0]), np.array([0.0, 1.5])
        )
        assert control[0] == pytest.approx(-0.5, abs=1e-6)


class TestCovectorEstimate:
    def test_it_recovers_the_costate_from_stationarity(self):
        r"""dL/du + (df/du)^T lambda = 0 gives lambda_v = -u."""
        problem = _problem()
        guess = _seed(scale=1.0, wobble=0.0)
        costates = covector_estimate(problem, guess)
        assert costates.shape == guess.x.shape
        assert np.allclose(costates[:, 1], -guess.u[:, 0], atol=1e-5)

    def test_it_does_not_return_the_zero_covector(self):
        """The failure this replaced.

        Seeding a backward sweep with lambda(tf) = 0 -- a transversality
        condition that holds only for *free* terminal states -- made lambda
        vanish identically for this fixed-endpoint problem, so u came out zero
        and the boundary value solver met a singular Jacobian at a non-solution.
        """
        costates = covector_estimate(_problem(), _seed())
        assert np.linalg.norm(costates) > 1.0


class TestRefineIndirect:
    @pytest.fixture(scope="class")
    @classmethod
    def solved(cls):
        return refine_indirect(_problem(), _seed())

    def test_it_converges(self, solved):
        assert solved.converged
        assert solved.residual < 1e-6

    def test_it_improves_on_the_direct_seed(self, solved):
        """The whole point: the indirect stage has to be worth running."""
        seed = _seed()
        seed_error = np.sqrt(
            np.mean((seed.u[:, 0] - _analytic(seed.t)[0]) ** 2)
        )
        refined_error = np.sqrt(
            np.mean((solved.u[:, 0] - _analytic(solved.t)[0]) ** 2)
        )
        assert refined_error < seed_error / 5.0

    def test_the_hamiltonian_is_constant(self, solved):
        """The invariant that certifies an autonomous optimal solution.

        H has no explicit time dependence here, so along an optimal trajectory
        it must be conserved. This is an independent check -- nothing in the
        solver enforces it -- and a violation would mean the costates are wrong
        however well the endpoints match.
        """
        problem = _problem()
        values = [
            hamiltonian(problem, solved.x[i], solved.u[i], solved.costates[i])
            for i in range(len(solved.t))
        ]
        assert max(values) - min(values) < 1e-6

    def test_the_costate_has_its_analytic_structure(self, solved):
        r"""lambda-dot_v = -lambda_x with lambda_x constant, so lambda_v is linear."""
        fit = np.polyval(np.polyfit(solved.t, solved.costates[:, 1], 1), solved.t)
        assert np.max(np.abs(solved.costates[:, 1] - fit)) < 1e-8

    def test_the_state_matches_the_analytic_trajectory(self, solved):
        _, exact = _analytic(solved.t)
        assert np.sqrt(np.mean((solved.x[:, 0] - exact) ** 2)) < 1e-2

    def test_a_failed_solve_reports_rather_than_raises(self):
        """An unconverged refinement still carries the direct answer it began with."""
        broken = OCPProblem(
            dynamics=lambda x, u: np.array([np.nan, np.nan]),
            path_constraints=lambda x, u: np.array([]),
            x0=np.array([0.0, 0.0]), xf_target=np.array([1.0, 0.0]),
            nx=2, nu=1, tf_guess=_T,
            running_cost=lambda x, u: 0.5 * float(u[0]) ** 2, time_weight=0.0,
        )
        result = refine_indirect(broken, _seed())
        assert not result.converged
        assert result.x.shape == _seed().x.shape


class TestHamiltonianSystem:
    def test_the_right_hand_side_is_vectorised(self):
        """solve_bvp evaluates on the whole mesh at once.

        A right-hand side written for one column silently receives a matrix,
        which is how the first version of this failed.
        """
        system = HamiltonianSystem(_problem())
        mesh = np.tile(np.array([0.1, 0.2, 0.3, 0.4]).reshape(4, 1), (1, 7))
        out = system.derivatives(np.zeros(7), mesh)
        assert out.shape == (4, 7)

    def test_single_and_vectorised_agree(self):
        system = HamiltonianSystem(_problem())
        column = np.array([0.1, 0.2, 0.3, 0.4])
        single = system.derivatives(0.0, column)
        batched = system.derivatives(np.zeros(1), column.reshape(4, 1))
        assert np.allclose(single, batched[:, 0])
