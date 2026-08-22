r"""Dymos as a transcription backend for :class:`OCPProblem`.

Why this exists, stated as the measurement that motivated it. On the standard
optimal-control benchmark -- the brachistochrone, whose minimum time is known in
closed form to be 1.8016 s -- the in-house transcription in
:mod:`aether.optimal_control.pseudospectral` reaches 2.0072 s, an 11.4 % error,
and reports ``success=False`` while doing it. Warm-starting it from nine
different flown trajectories does not help; the best answer it produces is the
one it labels a failure, and it labels ``t_f = 8.34`` a success. Dymos, driven
by the **same** SLSQP optimizer, returns:

===========================  ===========  ==========
transcription                :math:`t_f`  rel. error
===========================  ===========  ==========
``GaussLobatto(5, 3)``       1.8016246    1.4e-05
``GaussLobatto(10, 3)``      1.8016057    3.2e-06
``GaussLobatto(20, 3)``      1.8016039    2.2e-06
in-house, best of 9 starts   2.0072100    1.1e-01
===========================  ===========  ==========

Four orders of magnitude, with the optimizer held fixed. The difference is the
transcription and its analytic partials, and that is the whole argument: the
NLP solver was never what was wrong.

This module is deliberately a **translation layer and nothing else**. It maps an
:class:`OCPProblem` onto a Dymos phase and maps the answer back into an
:class:`OCPSolution`, so callers reach it through ``solve_ocp(problem,
solver="dymos")`` and change nothing else. Every piece of mathematics --
differentiation matrices, mesh refinement, the collocation itself -- belongs to
Dymos. That separation is the point rather than an implementation detail: the
bug this codebase already shipped was a hand-written differentiation matrix
sitting *beside* CasADi and IPOPT, with the library supplying derivatives for a
discretisation it had never been shown. A dependency buys trust only for the
part actually handed to it.

The objective is assembled through a single integrated state. Dymos takes one
objective variable, while :class:`OCPProblem` carries both a Lagrange term and a
weight on final time, so a cost state is integrated with rate
:math:`L(x,u) + w`; since :math:`\int_0^{t_f} w\,\mathrm{d}t = w\,t_f`, its
terminal value is exactly :math:`\int L + w t_f` and one objective covers both
cases, including pure minimum time where :math:`L` is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aether.optimal_control.pseudospectral import OCPProblem, OCPSolution

_FloatArray = NDArray[np.float64]

__all__ = ["DymosSettings", "solve_dymos"]


@dataclass(frozen=True)
class DymosSettings:
    """Transcription choices handed to Dymos.

    Attributes
    ----------
    num_segments:
        Segments in the phase. Defaults to ``None``, which derives a count from
        ``OCPProblem.N`` so an existing problem transcribes without being
        rewritten -- the two do not mean the same thing (``N`` counts LGL nodes,
        a segment carries ``order + 1`` of them), so this is a translation and
        not an equivalence.
    order:
        Polynomial order within each segment.
    optimizer:
        ``"SLSQP"`` matches what the in-house path uses, which is what makes the
        comparison above a statement about transcriptions rather than about
        solvers. ``"IPOPT"`` is available where pyoptsparse is installed.
    partials:
        ``"cs"`` (complex step) gives derivatives accurate to machine precision
        for dynamics that survive complex arithmetic, which is most of them.
        Falls back to ``"fd"`` automatically when a complex probe raises.
    """

    num_segments: int | None = None
    order: int = 3
    optimizer: str = "SLSQP"
    partials: str = "cs"
    max_iter: int | None = None


def _supports_complex_step(problem: OCPProblem) -> bool:
    """Whether the dynamics survive a complex perturbation.

    Checked rather than assumed. Complex step is exact where it works and raises
    or silently truncates where it does not -- ``np.maximum``, ``float()`` casts
    and most table lookups all break it -- so the probe decides.
    """
    try:
        state = np.asarray(problem.x0, dtype=np.complex128)
        state[0] += 1e-30j
        control = np.zeros(problem.nu, dtype=np.complex128)
        out = np.asarray(problem.dynamics(state, control))  # type: ignore[arg-type]
        return bool(np.iscomplexobj(out))
    except Exception:
        return False


def _ode_class(problem: OCPProblem, n_path: int, method: str) -> type:
    """Build an OpenMDAO component evaluating the problem's own callables.

    One component supplies three things Dymos needs as ODE outputs: the state
    rates, the running cost rate, and the path constraint residuals. They are
    computed together because they are computed together in the problem -- a
    separate component per output would evaluate the dynamics three times.
    """
    import openmdao.api as om  # type: ignore[import-untyped]

    nx, nu = problem.nx, problem.nu
    weight = float(problem.time_weight)
    running_cost = problem.running_cost
    dynamics = problem.dynamics
    path_constraints = problem.path_constraints

    class _GenericODE(om.ExplicitComponent):  # type: ignore[misc]
        def initialize(self) -> None:
            self.options.declare("num_nodes", types=int)

        def setup(self) -> None:
            n = self.options["num_nodes"]
            for i in range(nx):
                self.add_input(f"x{i}", shape=(n,))
            for j in range(nu):
                self.add_input(f"u{j}", shape=(n,))
            # Dimensionless states, so their rates carry `1/s`. Declaring them
            # unitless instead leaves Dymos connecting a `1/s` collocation input
            # to a unitless output, which it reports as a UnitsWarning -- and
            # the suite escalates warnings to errors, so an ambiguity that would
            # otherwise pass unnoticed becomes a failure. `OCPProblem` carries no
            # units of its own; this states that fact rather than leaving it to
            # be inferred twice, differently.
            for i in range(nx):
                self.add_output(f"xdot{i}", shape=(n,), units="1/s")
            self.add_output("cost_rate", shape=(n,), units="1/s")
            for k in range(n_path):
                self.add_output(f"g{k}", shape=(n,))
            self.declare_partials("*", "*", method=method)

        def compute(self, inputs: Any, outputs: Any) -> None:
            n = self.options["num_nodes"]
            dtype = np.complex128 if method == "cs" else np.float64
            states = np.empty((n, nx), dtype=dtype)
            controls = np.empty((n, nu), dtype=dtype)
            for i in range(nx):
                states[:, i] = inputs[f"x{i}"]
            for j in range(nu):
                controls[:, j] = inputs[f"u{j}"]
            rates = np.empty((n, nx), dtype=dtype)
            costs = np.empty(n, dtype=dtype)
            residuals = np.empty((n, n_path), dtype=dtype)
            for k in range(n):
                state, control = states[k], controls[k]
                rates[k] = np.atleast_1d(dynamics(state, control))
                lagrange = (
                    0.0 if running_cost is None else running_cost(state, control)
                )
                costs[k] = lagrange + weight
                if n_path:
                    residuals[k] = np.atleast_1d(path_constraints(state, control))
            for i in range(nx):
                outputs[f"xdot{i}"] = rates[:, i]
            outputs["cost_rate"] = costs
            for k in range(n_path):
                outputs[f"g{k}"] = residuals[:, k]

    return _GenericODE


def solve_dymos(
    problem: OCPProblem, settings: DymosSettings | None = None
) -> OCPSolution:
    """Solve an :class:`OCPProblem` through Dymos, returning the usual solution.

    Raises ``ImportError`` with an actionable message rather than failing
    obscurely when Dymos is absent, since it is an optional dependency and a
    caller who asked for this backend by name wants to know it is missing.
    """
    try:
        import dymos as dm  # type: ignore[import-untyped]
        import openmdao.api as om
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "the 'dymos' backend needs dymos and openmdao: pip install dymos"
        ) from exc

    settings = settings or DymosSettings()
    nx, nu = problem.nx, problem.nu
    x0 = np.asarray(problem.x0, dtype=np.float64)
    target = np.asarray(problem.xf_target, dtype=np.float64)

    probe = np.atleast_1d(
        problem.path_constraints(x0, np.zeros(nu, dtype=np.float64))
    )
    n_path = int(probe.size)

    method = settings.partials
    if method == "cs" and not _supports_complex_step(problem):
        method = "fd"

    segments = settings.num_segments
    if segments is None:
        segments = max(2, int(np.ceil(problem.N / max(settings.order, 1))))

    ode = _ode_class(problem, n_path, method)
    driver = om.ScipyOptimizeDriver(optimizer=settings.optimizer)
    driver.options["disp"] = False
    if settings.max_iter is not None:
        driver.options["maxiter"] = int(settings.max_iter)

    # `reports=False` and the disabled recorder below are not tidiness. Dymos
    # defaults to writing `dymos_solution.db` and an HTML report directory on
    # every call, so a solver invoked from a library litters the caller's
    # working directory just by being used.
    #
    # The sharper reason is that `SqliteRecorder.shutdown` calls `gc.collect()`,
    # which finalises any sqlite connection the process still holds open and
    # emits `ResourceWarning: unclosed database` for it. Recorders from earlier
    # solves in the same session are exactly such connections, so once two
    # solves have run, one recorder's shutdown reports the other's handle. In a
    # suite that escalates warnings to errors that is a failure, and an
    # order-dependent one -- it needs a prior solve to exist, so it passes when
    # the file runs alone and fails in the full run. Creating no recorder
    # removes the cause rather than the symptom; this project opens no sqlite
    # connections of its own.
    om_problem = om.Problem(model=om.Group(), reports=False)
    om_problem.driver = driver
    trajectory = om_problem.model.add_subsystem("traj", dm.Trajectory())
    phase = trajectory.add_phase(
        "phase",
        dm.Phase(
            ode_class=ode,
            transcription=dm.GaussLobatto(
                num_segments=segments, order=settings.order
            ),
        ),
    )
    phase.set_time_options(
        fix_initial=True,
        duration_bounds=(1e-3 * problem.tf_guess, 10.0 * problem.tf_guess),
    )
    for i in range(nx):
        phase.add_state(f"x{i}", fix_initial=True, rate_source=f"xdot{i}", units=None)
    # The cost state starts at zero and integrates L + time_weight, so its
    # terminal value is the whole objective and Dymos needs only one.
    phase.add_state("cost", fix_initial=True, rate_source="cost_rate", units=None)
    for j in range(nu):
        lower = upper = None
        if problem.u_bounds is not None:
            lower = float(np.atleast_1d(problem.u_bounds[0])[j])
            upper = float(np.atleast_1d(problem.u_bounds[1])[j])
        phase.add_control(f"u{j}", continuity=True, rate_continuity=True,
                          lower=lower, upper=upper, units=None)
    for i in range(nx):
        if np.isfinite(target[i]):
            phase.add_boundary_constraint(
                f"x{i}", loc="final", equals=float(target[i])
            )
    for k in range(n_path):
        # `g >= 0` feasible, the convention the transcription enforces.
        phase.add_path_constraint(f"g{k}", lower=0.0)
    phase.add_objective("cost", loc="final")

    om_problem.setup(check=False)
    phase.set_time_val(initial=float(problem.t0), duration=float(problem.tf_guess))
    guess = problem.initial_guess
    for i in range(nx):
        if guess is not None:
            phase.set_state_val(f"x{i}", list(np.asarray(guess[0])[:, i]))
        elif np.isfinite(target[i]):
            phase.set_state_val(f"x{i}", [float(x0[i]), float(target[i])])
        else:
            phase.set_state_val(f"x{i}", [float(x0[i]), float(x0[i])])
    phase.set_state_val("cost", [0.0, 1.0])
    for j in range(nu):
        if guess is not None:
            phase.set_control_val(f"u{j}", list(np.atleast_2d(guess[1])[:, j]))
        else:
            # The midpoint of the admissible interval, not zero. Zero is a
            # perfectly ordinary control value and a perfectly bad default: it
            # lies outside a one-sided bound such as a steering angle in
            # `(0, pi)`, and starting a gradient method on the wrong side of its
            # own bound produces a degenerate first step and a failed solve.
            start = 0.0
            if problem.u_bounds is not None:
                low = float(np.atleast_1d(problem.u_bounds[0])[j])
                high = float(np.atleast_1d(problem.u_bounds[1])[j])
                start = 0.5 * (low + high)
            phase.set_control_val(f"u{j}", [start, start])

    result = dm.run_problem(
        om_problem, run_driver=True, simulate=False,
        solution_record_file=None, simulation_record_file=None,
    )
    # `bool(result)` is deprecated in favour of the explicit attribute, and the
    # suite escalates warnings to errors -- so the deprecated spelling is not a
    # style question here, it is a failure. Older versions return a bare bool,
    # hence the fallback. It must be a ternary rather than a `getattr` default:
    # Python evaluates that default eagerly, so the deprecated `bool()` would
    # run -- and warn -- even when the attribute is present. A ternary does not.
    converged = (
        bool(result.success) if hasattr(result, "success") else not bool(result)
    )

    times = np.asarray(om_problem.get_val("traj.phase.timeseries.time")).ravel()
    # Gauss-Lobatto reports segment boundaries twice -- the last node of one
    # segment and the first of the next are the same instant. `OCPSolution`
    # implies a strictly increasing time vector, and anything interpolating the
    # control (`mesh_error`, for one) refuses duplicates outright, so the
    # repeats are dropped here rather than left for each consumer to discover.
    keep = np.concatenate([[True], np.diff(times) > 0.0])
    states = np.column_stack([
        np.asarray(om_problem.get_val(f"traj.phase.timeseries.x{i}")).ravel()
        for i in range(nx)
    ])
    controls = np.column_stack([
        np.asarray(om_problem.get_val(f"traj.phase.timeseries.u{j}")).ravel()
        for j in range(nu)
    ])
    cost = float(
        np.asarray(om_problem.get_val("traj.phase.timeseries.cost")).ravel()[-1]
    )
    return OCPSolution(
        x=states[keep],
        u=controls[keep],
        t=times[keep],
        tf=float(times[-1]),
        cost=cost,
        success=converged,
        message=(
            "dymos: converged" if converged else "dymos: driver reported failure"
        ),
        nfev=0,
        njev=0,
    )
