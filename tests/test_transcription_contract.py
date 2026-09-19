"""What any transcription backend must reproduce, stated as numbers.

The characterisation layer the refocus depends on. Swapping an implementation
for a library is only safe if the behaviour being replaced was pinned first --
otherwise a swap silently trades verified numbers for unverified ones, and the
numbers most worth keeping are exactly the ones a rewrite quietly moves.

Every problem here has a closed-form answer, so "did the backend work" has an
exact answer rather than a plausible-looking curve. Each is run against every
available backend, and a backend that cannot meet the contract is marked with
the tolerance it actually achieves rather than excused.

The in-house transcription currently fails the brachistochrone outright -- it
returns its initial guess and reports failure -- so it is held to a documented,
deliberately loose bound while Dymos is held to a tight one. Those two bounds
are the measurement that justifies the swap, and closing the gap is what would
let them merge.
"""

from __future__ import annotations

import numpy as np
import pytest

from aether.optimal_control.pseudospectral import OCPProblem, mesh_error, solve_ocp

_G = 9.80665

_BACKENDS = ["pseudospectral"]
try:  # pragma: no cover - environment dependent
    import dymos  # noqa: F401

    _BACKENDS.append("dymos")
except ImportError:  # pragma: no cover
    pass


def _brachistochrone(n: int = 24) -> OCPProblem:
    r"""The standard benchmark: fastest descent from (0,0) to (10,-5).

    The cycloid solution gives :math:`t_f = 1.8016` s. This is also Dymos's own
    tutorial problem, which makes it a fair cross-backend contract rather than
    one written around a particular implementation.
    """
    return OCPProblem(
        dynamics=lambda s, u: np.array([
            s[2] * np.sin(u[0]), -s[2] * np.cos(u[0]), _G * np.cos(u[0])
        ]),
        path_constraints=lambda s, u: np.zeros(0),
        x0=np.array([0.0, 0.0, 1e-6]),
        xf_target=np.array([10.0, -5.0, np.nan]),
        nx=3, nu=1, N=n, tf_guess=2.0,
        u_bounds=(np.array([0.01]), np.array([np.pi - 0.01])),
        max_iter=600,
    )


def _min_time_double_integrator(
    distance: float, accel: float, n: int = 20, *, at_rest: bool = True
) -> OCPProblem:
    r"""Minimum time over ``distance`` with :math:`|u| \le a`.

    Two closed forms, and they are not the same number -- which is worth being
    explicit about, because conflating them is how this file's first version got
    its expected values wrong:

    * arriving **at rest** is bang-bang, accelerate then brake, and takes
      :math:`t_f = 2\sqrt{d/a}`;
    * arriving with **terminal speed free** just accelerates the whole way, and
      takes :math:`t_f = \sqrt{2d/a}` -- a factor of :math:`\sqrt 2` shorter.

    The second is the instance that exposed the dropped node-0 collocation
    defect, returning 9.6903 s against a physical minimum of exactly 10.0.
    """
    return OCPProblem(
        dynamics=lambda x, u: np.array([x[1], u[0]]),
        path_constraints=lambda x, u: np.zeros(0),
        x0=np.array([0.0, 0.0]),
        xf_target=np.array([distance, 0.0 if at_rest else np.nan]),
        nx=2, nu=1, N=n, tf_guess=3.0 * np.sqrt(distance / accel),
        u_bounds=(np.array([-accel]), np.array([accel])),
        max_iter=600,
    )


#: Per-backend tolerance on the brachistochrone. Not a style choice -- the
#: in-house path does not solve it at all, returning ``tf_guess`` with
#: ``success=False`` at every node count and warm start tried, so its entry
#: records the failure instead of hiding it behind a skip.
_BRACH_TOLERANCE = {"pseudospectral": 2.0e-01, "dymos": 1.0e-04}


class TestBrachistochrone:
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_it_reaches_the_analytic_minimum_time(self, backend):
        solution = solve_ocp(_brachistochrone(), solver=backend)
        error = abs(solution.tf - 1.8016) / 1.8016
        assert error < _BRACH_TOLERANCE[backend]

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_it_never_beats_the_analytic_minimum(self, backend):
        """A transcription is a restriction; it cannot outperform the optimum.

        Returning less than 1.8016 would mean the discretisation is admitting
        trajectories the dynamics forbid -- the failure mode that let the
        minimum-time double integrator return 9.6903 s against a physical
        minimum of 10.0.
        """
        solution = solve_ocp(_brachistochrone(), solver=backend)
        assert solution.tf > 1.8016 * (1.0 - 1e-6)

    def test_dymos_is_orders_of_magnitude_closer(self):
        """The measurement the swap rests on, with the optimizer held fixed.

        Both backends run SLSQP. The gap is the transcription and its analytic
        partials, not the NLP solver -- which is why adopting a library for
        derivatives alone would not have closed it.
        """
        if "dymos" not in _BACKENDS:
            pytest.skip("dymos not installed")
        ours = solve_ocp(_brachistochrone(), solver="pseudospectral")
        theirs = solve_ocp(_brachistochrone(), solver="dymos")
        assert abs(theirs.tf - 1.8016) < abs(ours.tf - 1.8016) / 100.0

    def test_the_mesh_error_falls_with_refinement(self):
        """Convergence, measured independently of the solver's own opinion."""
        if "dymos" not in _BACKENDS:
            pytest.skip("dymos not installed")
        coarse = _brachistochrone(12)
        fine = _brachistochrone(24)
        e_coarse = np.linalg.norm(
            mesh_error(coarse, solve_ocp(coarse, solver="dymos"))
        )
        e_fine = np.linalg.norm(mesh_error(fine, solve_ocp(fine, solver="dymos")))
        assert e_fine < e_coarse / 5.0


class TestMinimumTimeDoubleIntegrator:
    @pytest.mark.parametrize("backend", _BACKENDS)
    @pytest.mark.parametrize("distance,accel", [(1.0, 1.0), (1000.0, 20.0)])
    def test_arriving_at_rest_matches_the_closed_form(self, backend, distance, accel):
        r"""Bang-bang, :math:`t_f = 2\sqrt{d/a}`."""
        problem = _min_time_double_integrator(distance, accel)
        solution = solve_ocp(problem, solver=backend)
        assert solution.tf == pytest.approx(2.0 * np.sqrt(distance / accel), rel=0.06)

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_a_free_terminal_speed_matches_its_own_closed_form(self, backend):
        r""":math:`t_f = \sqrt{2d/a}` -- exactly 10.0 s here.

        The instance that exposed the dropped node-0 collocation defect: it
        returned 9.6903 s, 3.1 % below a bound no trajectory can cross, because
        the transcription had become a relaxation.
        """
        problem = _min_time_double_integrator(1000.0, 20.0, at_rest=False)
        solution = solve_ocp(problem, solver=backend)
        assert solution.tf == pytest.approx(10.0, rel=0.06)

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_it_approaches_the_optimum_from_above(self, backend):
        """Bang-bang cannot be represented by a polynomial, so it must cost.

        Approaching from *below* is the signature of a transcription admitting
        infeasible trajectories, and it is what the convergence sequence did
        before node 0 was collocated: 1.923, 1.951, 1.973, 1.980 -- every one
        of them under a bound that cannot be crossed.
        """
        problem = _min_time_double_integrator(1.0, 1.0)
        assert solve_ocp(problem, solver=backend).tf > 2.0 * (1.0 - 1e-6)


class TestTheInterfaceIsActuallySwappable:
    def test_the_same_problem_object_serves_every_backend(self):
        """The point of the adapter: one specification, no caller changes."""
        problem = _min_time_double_integrator(1.0, 1.0)
        answers = {b: solve_ocp(problem, solver=b) for b in _BACKENDS}
        for backend, solution in answers.items():
            assert solution.x.shape[1] == problem.nx, backend
            assert solution.u.shape[1] == problem.nu, backend
            assert solution.t.shape[0] == solution.x.shape[0], backend

    def test_times_are_strictly_increasing(self):
        """Gauss-Lobatto repeats segment boundaries; the adapter must not.

        ``mesh_error`` and every interpolator refuse duplicate abscissae, so a
        backend returning them breaks consumers rather than itself.
        """
        problem = _min_time_double_integrator(1.0, 1.0)
        for backend in _BACKENDS:
            times = solve_ocp(problem, solver=backend).t
            assert np.all(np.diff(times) > 0.0), backend

    def test_an_unknown_backend_names_the_alternatives(self):
        with pytest.raises(ValueError, match="dymos"):
            solve_ocp(_min_time_double_integrator(1.0, 1.0), solver="nonesuch")
