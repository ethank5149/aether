"""Solve a :class:`TrajectoryProblem`, returning a :class:`ReferenceTrajectory`.

This is the dispatch layer: it transcribes the problem onto a backend and wraps
the result in the pseudospectral-polynomial reference the rest of the module
speaks. The default and only backend wired here is LGL pseudospectral
collocation (:mod:`aether.optimal_control.pseudospectral`); the successive-
convexification and stochastic-MPPI backends already in ``guidance`` slot in
behind the same ``solve`` signature and are left as declared extension points so
the foundation lands without duplicating them.

Multi-phase problems are solved phase by phase with explicit linkage: each phase
is transcribed as its own optimal-control problem, its terminal state is mapped
through the phase's ``linkage`` into the next phase's initial state, and the
phases are concatenated (with monotonically increasing absolute time) into one
multi-phase reference. This is the linked-phase approach standard in trajectory
optimization when the phases have distinct dynamics regimes; a fully
simultaneous multi-phase transcription is a later refinement behind this same
entry point.

Only Bolza-form objectives (flight time, control effort) map onto the
transcription's cost; a trajectory-functional objective (undetected time,
pursuit time) is not something a single collocation solve minimises, so it is
carried in the returned trajectory's metadata for the generator / engagement
layer to score, and the inner solve minimises time. That division — inner OCP,
outer functional search — is the mature pattern and matches the existing
``optimize_trajectories`` builder registry.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from aether.optimal_control.pseudospectral import OCPProblem, solve_ocp
from aether.trajectory.objectives import (
    MinControlEffort,
    MinFlightTime,
    Objective,
    WeightedSum,
)
from aether.trajectory.problem import PhaseSpec, TrajectoryProblem
from aether.trajectory.representation import PhasePolynomial, ReferenceTrajectory

_FloatArray = NDArray[np.float64]

__all__ = ["SolveError", "solve"]


class SolveError(RuntimeError):
    """A phase failed to converge."""


def _effort_cost(weight: float) -> Callable[[_FloatArray, _FloatArray], float]:
    def running_cost(x: _FloatArray, u: _FloatArray) -> float:
        return weight * float(np.dot(u, u)) if u.size else 0.0

    return running_cost


def _bolza(objective: Objective) -> tuple[object, float]:
    """Map an objective to a pseudospectral ``(running_cost, time_weight)``.

    Recognised Bolza forms:

    * :class:`MinFlightTime` -> minimise the final time.
    * :class:`MinControlEffort` -> minimise integrated control effort. Note this
      is degenerate on its own with a *free* final time (a longer flight always
      lowers the effort), so it is meant to be composed with a time term.
    * :class:`WeightedSum` of the two -> both, with the given weights and scales;
      this is the well-posed minimum-effort-in-bounded-time form.

    Any other (trajectory-functional) objective: the inner solve minimises time
    and the outer generator layer scores the functional.
    """
    if isinstance(objective, MinControlEffort):
        return _effort_cost(1.0), 0.0
    if isinstance(objective, WeightedSum):
        time_weight = 0.0
        effort_weight = 0.0
        for obj, weight, scale in objective.terms:
            if isinstance(obj, MinFlightTime):
                time_weight += weight / scale
            elif isinstance(obj, MinControlEffort):
                effort_weight += weight / scale
        if effort_weight > 0.0:
            return _effort_cost(effort_weight), time_weight
        return None, (time_weight or 1.0)
    # MinFlightTime and any functional objective: minimise time.
    return None, 1.0


def _solve_phase(
    phase: PhaseSpec,
    x0: _FloatArray,
    xf_target: _FloatArray,
    nx: int,
    objective: Objective,
    x_bounds: tuple[_FloatArray, _FloatArray] | None,
    t0: float,
    backend: str = "pseudospectral",
) -> PhasePolynomial:
    """Transcribe and solve one phase; return it as a polynomial segment."""
    running_cost, time_weight = _bolza(objective)
    ocp = OCPProblem(
        dynamics=phase.dynamics,
        path_constraints=phase.path_constraints,
        x0=x0,
        xf_target=xf_target,
        nx=nx,
        nu=phase.nu,
        N=phase.nodes,
        t0=0.0,
        tf_guess=phase.duration_guess,
        u_bounds=phase.u_bounds,
        x_bounds=x_bounds,
        running_cost=running_cost,  # type: ignore[arg-type]
        time_weight=time_weight,
        max_iter=phase.max_iter,
        terminal_constraints=phase.terminal_constraints,
        progress=phase.progress,
        initial_guess=phase.initial_guess,
    )
    solution = solve_ocp(ocp, solver=backend)
    if not solution.success:
        raise SolveError(f"phase {phase.label!r} did not converge: {solution.message}")
    t = np.asarray(solution.t, dtype=np.float64)
    control = np.asarray(solution.u, dtype=np.float64)
    if phase.nu == 0:
        control = np.zeros((t.size, 0))
    return PhasePolynomial(
        t0=t0,
        t1=t0 + float(t[-1] - t[0]),
        nodes=t0 + (t - t[0]),
        state_values=np.asarray(solution.x, dtype=np.float64),
        control_values=control,
        label=phase.label,
    )


def solve(
    problem: TrajectoryProblem,
    method: str = "pseudospectral",
    progress: Callable[[int, PhaseSpec, str], None] | None = None,
) -> ReferenceTrajectory:
    """Solve ``problem`` and return the optimised reference trajectory.

    Parameters
    ----------
    problem:
        The trajectory-optimization problem.
    method:
        Backend. ``"pseudospectral"`` (default) is wired; ``"scvx"`` and
        ``"mppi"`` are recognised names reserved for the existing ``guidance``
        solvers and currently raise :class:`NotImplementedError` so a caller
        gets a clear signal rather than a silent substitution.
    """
    if method == "dymos":
        pass
    elif method in {"scvx", "mppi"}:
        raise NotImplementedError(
            f"the {method!r} backend is a declared extension point; the "
            "foundation wires 'pseudospectral'. Use guidance.solve_scvx / the "
            "stochastic optimiser directly until it is bridged here."
        )
    elif method != "pseudospectral":
        raise ValueError(f"unknown method {method!r}")

    def announce(index: int, phase: PhaseSpec, event: str) -> None:
        if progress is not None:
            progress(index, phase, event)

    segments: list[PhasePolynomial] = []
    x0 = np.asarray(problem.x0, dtype=np.float64)
    t_cursor = 0.0
    last = len(problem.phases) - 1
    for i, phase in enumerate(problem.phases):
        # Interior phases run to a free terminal (all-free target); only the
        # final phase carries the problem's terminal target.
        xf = problem.xf_target if i == last else np.full(problem.nx, np.inf)
        announce(i, phase, "started")
        try:
            segment = _solve_phase(
                phase, x0, xf, problem.nx, problem.objective, problem.x_bounds,
                t_cursor, backend=method,
            )
        except SolveError:
            # Reported before it is re-raised, so a watcher learns *which* phase
            # failed rather than only that the solve did.
            announce(i, phase, "failed")
            raise
        announce(i, phase, "converged")
        segments.append(segment)
        t_cursor = segment.t1
        if i != last:
            terminal = segment.state(segment.t1)
            link = phase.linkage or (lambda x: x)
            x0 = np.asarray(link(terminal), dtype=np.float64)

    metadata: dict[str, object] = {
        "label": problem.label,
        "objective": problem.objective.name,
        "method": method,
        "staging_times": [s.t1 for s in segments[:-1]],
        "n_phases": len(segments),
    }
    return ReferenceTrajectory.from_phases(segments, metadata=metadata)
