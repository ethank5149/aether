"""LGL collocation, checked against things with known answers.

This module had no test that could distinguish a working transcription from
a broken one, and it was broken in four independent ways at once. The
checks here are chosen so that each of those four failures is caught by a
specific assertion rather than by a trajectory looking wrong later.
"""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import pytest
import scipy.integrate
from numpy.polynomial.legendre import Legendre

from aether.optimal_control.pseudospectral import (
    OCPProblem,
    OCPSolution,
    differentiation_matrix,
    lgl_nodes,
    map_to_physical,
    map_to_tau,
    mesh_error,
    solve_ocp,
)


def _double_integrator() -> OCPProblem:
    """Rest to rest over unit distance with unit acceleration authority.

    The minimum-time solution is bang-bang — full acceleration to the
    halfway point, full braking after — and takes ``2 sqrt(d / a) = 2 s``.
    """
    return OCPProblem(
        dynamics=lambda x, u: np.array([x[1], u[0]]),
        path_constraints=lambda x, u: np.zeros(0),
        x0=np.array([0.0, 0.0]),
        xf_target=np.array([1.0, 0.0]),
        nx=2, nu=1, N=20, t0=0.0, tf_guess=3.0,
        u_bounds=(np.array([-1.0]), np.array([1.0])),
        max_iter=400,
    )


class TestLGLNodes:
    """The quadrature rule, against its own definition."""

    @pytest.mark.parametrize("n", [3, 4, 5, 6, 8, 12])
    def test_the_weights_integrate_a_constant(self, n):
        """The measure of ``[-1, 1]`` is 2. The previous implementation gave
        1.67 at n=3 and **4.32 at n=8** — and the error grew with n, so
        refining the discretisation made the quadrature worse."""
        _, weights = lgl_nodes(n)
        assert float(weights.sum()) == pytest.approx(2.0, abs=1e-12)

    @pytest.mark.parametrize("n", [3, 4, 5, 6, 8, 12])
    def test_it_is_exact_to_degree_two_n_minus_three(self, n):
        """The defining property of Lobatto quadrature."""
        nodes, weights = lgl_nodes(n)
        for degree in range(2 * n - 2):
            exact = 0.0 if degree % 2 else 2.0 / (degree + 1)
            assert float(weights @ nodes**degree) == pytest.approx(exact, abs=1e-11)

    @pytest.mark.parametrize("n", [4, 5, 6, 8])
    def test_the_interior_nodes_are_the_roots_of_the_derivative(self, n):
        """LGL interior nodes are the roots of ``P'_{n-1}``. They were taken
        from ``leggauss(n - 2)`` — the *Gauss* nodes, roots of ``P_{n-2}`` —
        which is a different set: out by 0.13 at n=4 and 0.096 at n=6."""
        nodes, _ = lgl_nodes(n)
        expected = np.sort(np.real(Legendre.basis(n - 1).deriv().roots()))
        assert np.allclose(nodes[1:-1], expected, atol=1e-12)

    @pytest.mark.parametrize("n", [3, 5, 9])
    def test_the_endpoints_are_included_and_the_set_is_symmetric(self, n):
        nodes, weights = lgl_nodes(n)
        assert nodes[0] == pytest.approx(-1.0)
        assert nodes[-1] == pytest.approx(1.0)
        assert np.allclose(nodes, -nodes[::-1], atol=1e-12)
        assert np.allclose(weights, weights[::-1], atol=1e-12)
        assert np.all(weights > 0.0)

    def test_two_nodes_is_the_trapezium_rule(self):
        nodes, weights = lgl_nodes(2)
        assert np.allclose(nodes, [-1.0, 1.0])
        assert np.allclose(weights, [1.0, 1.0])

    def test_fewer_than_two_nodes_is_refused(self):
        with pytest.raises(ValueError, match="n must be >= 2"):
            lgl_nodes(1)


class TestDifferentiationMatrix:
    """Exact on the polynomial space it spans."""

    @pytest.mark.parametrize("n", [4, 6, 9])
    def test_it_differentiates_polynomials_exactly(self, n):
        nodes, _ = lgl_nodes(n)
        matrix = differentiation_matrix(nodes)
        for degree in range(1, n):
            got = matrix @ nodes**degree
            assert np.allclose(got, degree * nodes ** (degree - 1), atol=1e-9)

    @pytest.mark.parametrize("n", [3, 5, 8, 12])
    def test_a_constant_has_zero_derivative(self, n):
        """Row sums vanish — the "negative sum trick". Without it a constant
        state acquires a spurious rate and every defect constraint carries a
        bias that no amount of iteration removes."""
        nodes, _ = lgl_nodes(n)
        matrix = differentiation_matrix(nodes)
        assert np.abs(matrix.sum(axis=1)).max() < 1e-12


class TestTimeMapping:
    def test_the_maps_are_inverses(self):
        tau = np.linspace(-1.0, 1.0, 11)
        physical = map_to_physical(tau, 12.0, 400.0)
        assert np.allclose(map_to_tau(physical, 12.0, 400.0), tau, atol=1e-12)

    def test_the_endpoints_land_on_the_horizon(self):
        physical = map_to_physical(np.array([-1.0, 1.0]), 5.0, 95.0)
        assert physical[0] == pytest.approx(5.0)
        assert physical[-1] == pytest.approx(95.0)


class TestMinimumTimeTranscription:
    """The whole point: does it find the right answer to a known problem?"""

    def test_it_recovers_the_analytic_minimum_time(self):
        """Rest to rest, unit distance, unit acceleration: 2 s.

        The previous transcription returned **1.000000 s** here — exactly
        its own lower bound on ``tf`` — and reported ``success=True``. The
        cause was that the defect residual was formed as
        ``(f - D x) / scale``, which vanishes when ``D x = f`` regardless of
        ``tf``: the final time appeared in the objective and in no
        constraint, so minimising it drove it to the bound with nothing
        pushing back.
        """
        solution = solve_ocp(_double_integrator())
        assert solution.success, solution.message
        assert solution.tf == pytest.approx(2.0, rel=0.05)

    def test_the_terminal_state_is_actually_reached(self):
        solution = solve_ocp(_double_integrator())
        assert np.allclose(solution.x[-1], [1.0, 0.0], atol=1e-6)

    def test_the_initial_state_is_held(self):
        solution = solve_ocp(_double_integrator())
        assert np.allclose(solution.x[0], [0.0, 0.0], atol=1e-9)

    def test_refining_the_grid_does_not_make_it_worse(self):
        """Bang-bang control is discontinuous, so a global polynomial basis
        gives algebraic rather than spectral convergence here — Gibbs, not a
        defect. What must hold either way is that the error falls."""
        errors = []
        for n in (8, 12, 20, 30):
            problem = OCPProblem(
                dynamics=lambda x, u: np.array([x[1], u[0]]),
                path_constraints=lambda x, u: np.zeros(0),
                x0=np.array([0.0, 0.0]), xf_target=np.array([1.0, 0.0]),
                nx=2, nu=1, N=n, tf_guess=3.0,
                u_bounds=(np.array([-1.0]), np.array([1.0])), max_iter=400,
            )
            solution = solve_ocp(problem)
            assert solution.success, f"N={n}: {solution.message}"
            errors.append(abs(solution.tf - 2.0))
        assert errors == sorted(errors, reverse=True), (
            f"error must fall as the grid refines, got {errors}"
        )

    def test_the_dynamics_are_satisfied_along_the_solution(self):
        """Independent of the solver's own residual: differentiate the
        returned states with the same matrix and compare with the dynamics
        in *seconds*, which is where the missing time scale showed up."""
        solution = solve_ocp(_double_integrator())
        nodes, _ = lgl_nodes(solution.x.shape[0])
        matrix = differentiation_matrix(nodes)
        scale = 0.5 * (solution.tf - 0.0)
        rates = matrix @ solution.x / scale
        expected = np.column_stack([solution.x[:, 1], solution.u[:, 0]])
        # Endpoints carry the polynomial's edge error; the interior is the
        # part the collocation actually enforces.
        assert np.abs(rates[1:-1] - expected[1:-1]).max() < 5e-2

    def test_the_control_bounds_are_respected(self):
        solution = solve_ocp(_double_integrator())
        assert solution.u.min() >= -1.0 - 1e-6
        assert solution.u.max() <= 1.0 + 1e-6


class TestRunningCost:
    """The Lagrange term, which is what the quadrature weights are for."""

    def test_a_running_cost_is_integrated_with_the_weights(self):
        """A constant integrand of 1 over ``[0, tf]`` costs ``tf``, so with
        ``time_weight=0`` the objective must come back as the horizon."""
        horizon = 7.0
        problem = OCPProblem(
            dynamics=lambda x, u: np.array([0.0]),
            path_constraints=lambda x, u: np.zeros(0),
            x0=np.array([0.0]), xf_target=np.array([np.nan]),
            nx=1, nu=1, N=8, tf_guess=horizon,
            u_bounds=(np.array([0.0]), np.array([0.0])),
            running_cost=lambda x, u: 1.0, time_weight=0.0, max_iter=50,
        )
        from aether.optimal_control.pseudospectral import _objective

        z = np.zeros((problem.N + 1) * (problem.nx + problem.nu) + 1)
        z[-1] = horizon
        assert _objective(z, problem) == pytest.approx(horizon, rel=1e-9)

    def test_no_running_cost_is_pure_minimum_time(self):
        from aether.optimal_control.pseudospectral import _objective

        problem = _double_integrator()
        z = np.zeros((problem.N + 1) * (problem.nx + problem.nu) + 1)
        z[-1] = 42.0
        assert _objective(z, problem) == pytest.approx(42.0)


class TestInitialGuess:
    """Where a cold start begins."""

    def test_it_interpolates_towards_the_target_not_the_origin(self):
        """It used to interpolate from ``x0`` towards ``np.zeros(nx)``. For a
        state in ECI metres that is the centre of the Earth, so every cold
        start marched the guess through the planet."""
        from aether.optimal_control.pseudospectral import _build_nlp_variables

        start = np.array([7.0e6, 0.0])
        target = np.array([7.1e6, 100.0])
        problem = OCPProblem(
            dynamics=lambda x, u: np.zeros(2),
            path_constraints=lambda x, u: np.zeros(0),
            x0=start, xf_target=target, nx=2, nu=1, N=6,
        )
        z0, _, _ = _build_nlp_variables(problem)
        states = z0[: (problem.N + 1) * problem.nx].reshape(problem.N + 1, 2)
        assert np.allclose(states[0], start)
        assert np.allclose(states[-1], target)
        assert states[:, 0].min() > 6.9e6, "the guess must not pass through zero"

    def test_free_terminal_components_hold_their_initial_value(self):
        from aether.optimal_control.pseudospectral import _build_nlp_variables

        problem = OCPProblem(
            dynamics=lambda x, u: np.zeros(2),
            path_constraints=lambda x, u: np.zeros(0),
            x0=np.array([5.0, 9.0]),
            xf_target=np.array([np.nan, 3.0]),
            nx=2, nu=1, N=4,
        )
        z0, _, _ = _build_nlp_variables(problem)
        states = z0[: (problem.N + 1) * problem.nx].reshape(problem.N + 1, 2)
        assert np.allclose(states[:, 0], 5.0)
        assert states[-1, 1] == pytest.approx(3.0)


class TestSolverSelection:
    def test_an_unknown_solver_is_refused(self):
        with pytest.raises(ValueError, match="unknown solver"):
            solve_ocp(_double_integrator(), solver="nonesuch")


class TestScvxAdapter:
    """The second solver the module advertises.

    ``solve_ocp(problem, solver="scvx")`` could not execute. The adapter
    called :func:`~aether.optimal_control.scvx.solve_scvx` with six keyword
    arguments it does not accept, passed a three-argument dynamics function
    where a two-argument one is required, and then read ``x``, ``u``, ``t``,
    ``cost``, ``message`` and ``n_iter`` off an ``SCvxResult`` that carries
    ``states``, ``controls``, ``iterations``, ``virtual_norm``,
    ``trust_radius``, ``converged`` and ``history``. A bare
    ``except Exception`` turned the resulting ``TypeError`` into
    ``success=False``, so the path looked like a solver that never
    converged rather than one that never ran.
    """

    def test_it_returns_a_trajectory_of_the_right_shape(self):
        problem = _double_integrator()
        solution = solve_ocp(problem, solver="scvx")
        assert solution.x.shape == (problem.N + 1, problem.nx)
        assert solution.u.shape == (problem.N + 1, problem.nu)
        assert solution.t.shape == (problem.N + 1,)

    def test_it_does_not_report_a_type_error_as_non_convergence(self):
        solution = solve_ocp(_double_integrator(), solver="scvx")
        assert "TypeError" not in solution.message
        assert "unexpected keyword" not in solution.message

    def test_it_drives_the_state_towards_the_target(self):
        """Convergence of the trust-region loop is a separate question from
        whether the adapter wires the two sides together correctly; this
        checks the wiring, by asking whether the terminal state moved."""
        solution = solve_ocp(_double_integrator(), solver="scvx")
        assert abs(float(solution.x[-1, 0]) - 1.0) < 0.2

    def test_the_time_grid_spans_the_horizon(self):
        problem = _double_integrator()
        solution = solve_ocp(problem, solver="scvx")
        assert solution.t[0] == pytest.approx(problem.t0)
        assert solution.t[-1] == pytest.approx(problem.tf_guess)


class TestEveryNodeIsCollocated:
    """The initial collocation defect is a constraint, not a redundancy.

    It used to be skipped, on the stated grounds that the boundary condition
    covered it. The boundary condition fixes the state's *value* at node 0; the
    defect fixes its *derivative*. Dropping it left the polynomial's initial
    slope free and ``u[0]`` in no constraint at all, making the transcription a
    strict relaxation of the problem.
    """

    @staticmethod
    def _min_time_problem(n=20, max_iter=400):
        amax = 20.0
        return OCPProblem(
            dynamics=lambda s, u: np.array([s[2], s[3], u[0], u[1]]),
            path_constraints=lambda s, u: np.zeros(0),
            x0=np.zeros(4),
            xf_target=np.array([1000.0, 500.0, np.inf, np.inf]),
            nx=4, nu=2, N=n, t0=0.0, tf_guess=15.0,
            u_bounds=(np.array([-amax, -amax]), np.array([amax, amax])),
            time_weight=1.0, max_iter=max_iter,
        )

    def test_it_cannot_beat_the_physically_attainable_minimum(self):
        """The check that exposed the relaxation, and the tightest one available.

        Accelerating from rest at 20 m/s^2 with the terminal velocity free,
        1000 m cannot be covered in less than sqrt(2*1000/20) = 10 s exactly.
        The relaxed transcription returned 9.6903 s -- 3.1% below a bound that
        no physical trajectory can cross, which is why "does it reach the
        target" could never catch it: the terminal position is an equality
        constraint and was always satisfied.
        """
        solution = solve_ocp(self._min_time_problem())
        assert solution.success, solution.message
        assert solution.tf >= 10.0 - 1e-6
        assert solution.tf == pytest.approx(10.0, abs=1e-4)

    def test_the_initial_defect_is_satisfied(self):
        """Structural: the collocation condition holds at node 0 like any other."""
        problem = self._min_time_problem()
        solution = solve_ocp(problem)
        # lgl_nodes(k) returns k nodes, and the transcription carries N + 1
        # states, so the matrix the solver builds is lgl_nodes(N + 1).
        nodes, _ = lgl_nodes(problem.N + 1)
        diff = differentiation_matrix(nodes)
        states = np.asarray(solution.x, dtype=np.float64)
        controls = np.asarray(solution.u, dtype=np.float64)
        scale = 0.5 * (solution.tf - problem.t0)
        residual = diff[0] @ states - scale * problem.dynamics(states[0], controls[0])
        assert np.max(np.abs(residual)) < 1e-6

    def test_convergence_is_from_above_not_below(self):
        """A valid transcription restricts the problem; it must not relax it.

        A global polynomial cannot represent a bang-bang control exactly, so
        the discrete optimum should be *worse* than the continuous one and
        approach it from above as the mesh refines. Before the fix this
        sequence ran 1.923, 1.951, 1.973, 1.980 -- below the true 2.0 s at
        every mesh, which is impossible for a restriction and is the tell that
        infeasible trajectories were being admitted. A refinement test that
        only checked the error *falls* passed either way.
        """
        times = []
        for n in (8, 12, 20, 30):
            problem = OCPProblem(
                dynamics=lambda x, u: np.array([x[1], u[0]]),
                path_constraints=lambda x, u: np.zeros(0),
                x0=np.array([0.0, 0.0]), xf_target=np.array([1.0, 0.0]),
                nx=2, nu=1, N=n, tf_guess=3.0,
                u_bounds=(np.array([-1.0]), np.array([1.0])), max_iter=400,
            )
            solution = solve_ocp(problem)
            assert solution.success, solution.message
            times.append(solution.tf)
        assert all(t >= 2.0 for t in times)
        assert all(a > b for a, b in itertools.pairwise(times))
        assert times[-1] == pytest.approx(2.0, rel=2e-3)

    def test_an_independent_integration_reproduces_the_terminal_state(self):
        """The transcription's answer must survive being re-flown outside it.

        Interpolating the returned control and integrating the dynamics with a
        general-purpose integrator is a path that shares no code with the
        collocation, so agreement is evidence the discrete solution is a real
        trajectory rather than an artefact of its own mesh.
        """
        problem = self._min_time_problem()
        solution = solve_ocp(problem)
        times = np.asarray(solution.t, dtype=np.float64)
        controls = np.asarray(solution.u, dtype=np.float64)

        def rhs(t, state):
            control = np.array([
                np.interp(t, times, controls[:, j]) for j in range(problem.nu)
            ])
            return problem.dynamics(state, control)

        flown = scipy.integrate.solve_ivp(
            rhs, (times[0], times[-1]), problem.x0, rtol=1e-10, atol=1e-10,
        )
        assert flown.success
        assert flown.y[:2, -1] == pytest.approx(
            np.asarray(solution.x)[-1, :2], abs=5.0
        )


class TestWarmStart:
    """The optional ``initial_guess``, and why the cold start needed a companion.

    The straight-line guess interpolates from ``x0`` toward ``xf_target``,
    holding components the target leaves free. That is adequate whenever the
    target says where the trajectory is going, and useless when it does not: an
    entry glide closing on terminal *speed* alone gets a guess with longitude
    held constant -- a vehicle that never moves downrange -- so every defect is
    violated at iteration zero and SLSQP stops without leaving the guess.
    """

    @staticmethod
    def _nodes(problem):
        tau, _ = lgl_nodes(problem.N + 1)
        return map_to_physical(tau, problem.t0, problem.tf_guess)

    def test_a_supplied_guess_is_used_verbatim(self):
        """White-box: the guess must reach the NLP unmodified."""
        import dataclasses

        from aether.optimal_control.pseudospectral import _build_nlp_variables

        problem = _double_integrator()
        n, nx, nu = problem.N, problem.nx, problem.nu
        states = np.arange((n + 1) * nx, dtype=float).reshape(n + 1, nx)
        controls = np.full((n + 1, nu), 0.25)
        z0, _, _ = _build_nlp_variables(
            dataclasses.replace(problem, initial_guess=(states, controls, 2.5))
        )
        assert np.allclose(z0[: (n + 1) * nx], states.reshape(-1))
        assert np.allclose(
            z0[(n + 1) * nx : (n + 1) * nx + (n + 1) * nu], controls.reshape(-1)
        )
        assert z0[-1] == pytest.approx(2.5)

    def test_omitting_it_keeps_the_cold_start(self):
        """The default path is unchanged, which is what makes this additive."""
        import dataclasses

        from aether.optimal_control.pseudospectral import _build_nlp_variables

        problem = _double_integrator()
        bare, _, _ = _build_nlp_variables(problem)
        explicit, _, _ = _build_nlp_variables(
            dataclasses.replace(problem, initial_guess=None)
        )
        assert np.allclose(bare, explicit)

    def test_it_still_reaches_the_same_optimum(self):
        """A warm start changes where the search begins, not where it ends."""
        import dataclasses

        problem = _double_integrator()
        cold = solve_ocp(problem)
        times = self._nodes(problem)
        states = np.column_stack([times / times[-1], np.ones_like(times)])
        warm = solve_ocp(
            dataclasses.replace(
                problem,
                initial_guess=(states, np.zeros((problem.N + 1, 1)), 2.2),
            )
        )
        assert warm.success
        assert warm.tf == pytest.approx(cold.tf, rel=1e-3)

    @pytest.mark.parametrize("bad_shape", [(5, 2), (21, 3)])
    def test_a_mismatched_state_guess_is_refused(self, bad_shape):
        """Silently reshaping a wrong guess would scramble states into controls."""
        import dataclasses

        from aether.optimal_control.pseudospectral import _build_nlp_variables

        problem = _double_integrator()
        with pytest.raises(ValueError, match="initial_guess states"):
            _build_nlp_variables(
                dataclasses.replace(
                    problem,
                    initial_guess=(
                        np.zeros(bad_shape), np.zeros((problem.N + 1, 1)), 2.0
                    ),
                )
            )

    def test_a_mismatched_control_guess_is_refused(self):
        import dataclasses

        from aether.optimal_control.pseudospectral import _build_nlp_variables

        problem = _double_integrator()
        with pytest.raises(ValueError, match="initial_guess controls"):
            _build_nlp_variables(
                dataclasses.replace(
                    problem,
                    initial_guess=(
                        np.zeros((problem.N + 1, 2)), np.zeros((7, 1)), 2.0
                    ),
                )
            )


class TestMeshError:
    """The instrument that separates a converged solve from a believable one.

    A transcription enforces the dynamics at the nodes and nowhere else, so the
    NLP always solves a relaxation of the continuous problem. That is what
    discretisation *means*, not a defect to repair. What can be repaired is not
    knowing how big the gap is -- and the gap is invisible from the solver's own
    diagnostics, which report success either way.
    """

    def test_it_falls_with_the_node_count(self):
        """The signature of a convergent discretisation.

        Bang-bang control cannot be represented by a polynomial at all, so a
        non-zero error here is honest. What matters is the trend: 2.1e-03 at
        N=8 down to 6.2e-05 at N=30. The maximum-range entry glide does the
        opposite -- 8.2 km at N=20 rising to 33.4 km at N=30 -- which is how
        collocation ringing announces itself.
        """
        errors = []
        for n in (8, 12, 20, 30):
            problem = dataclasses.replace(_double_integrator(), N=n)
            errors.append(
                float(np.linalg.norm(mesh_error(problem, solve_ocp(problem))))
            )
        assert errors == sorted(errors, reverse=True)
        assert errors[-1] < errors[0] / 10.0

    def test_it_returns_a_vector_not_a_norm(self):
        """States carry different units; one threshold across them means nothing."""
        problem = _double_integrator()
        error = mesh_error(problem, solve_ocp(problem))
        assert error.shape == (problem.nx,)

    def test_it_is_near_zero_on_a_trajectory_that_is_exactly_right(self):
        """The measurement floor, established without involving the solver.

        Constant unit acceleration from rest for 2 s lands at ``x = 2, v = 2``
        exactly. Handing that analytic trajectory to the indicator must return
        essentially zero, which bounds what it can resolve -- anything larger in
        a real solve is the solution, not the instrument.
        """
        problem = OCPProblem(
            dynamics=lambda x, u: np.array([x[1], u[0]]),
            path_constraints=lambda x, u: np.zeros(0),
            x0=np.array([0.0, 0.0]), xf_target=np.array([2.0, 2.0]),
            nx=2, nu=1, N=10, tf_guess=2.0,
        )
        tau, _ = lgl_nodes(problem.N + 1)
        times = map_to_physical(tau, 0.0, 2.0)
        exact = OCPSolution(
            x=np.column_stack([0.5 * times**2, times]),
            u=np.ones((times.size, 1)),
            t=times, tf=2.0, cost=0.0, success=True,
            message="analytic", nfev=0, njev=0,
        )
        assert np.max(np.abs(mesh_error(problem, exact))) < 1e-8

    def test_it_detects_a_control_that_does_not_fly(self):
        """Directly: corrupt the control, and the indicator must notice.

        The entry-glide failure in miniature. The state history still says the
        trajectory arrives; the control history no longer takes it there, and
        nothing but an independent integration can tell.
        """
        problem = _double_integrator()
        good = solve_ocp(problem)
        clean = float(np.linalg.norm(mesh_error(problem, good)))
        rung = dataclasses.replace(
            good,
            u=good.u + 0.4 * np.sign(np.sin(9.0 * np.arange(good.u.shape[0]))).reshape(-1, 1),
        )
        assert float(np.linalg.norm(mesh_error(problem, rung))) > 20.0 * clean

    def test_a_single_node_has_nothing_to_integrate(self):
        problem = _double_integrator()
        solution = solve_ocp(problem)
        with pytest.raises(ValueError, match="at least two nodes"):
            mesh_error(
                problem,
                dataclasses.replace(
                    solution, t=solution.t[:1], u=solution.u[:1], x=solution.x[:1]
                ),
            )
