"""Trajectory-optimization problem specification.

A :class:`TrajectoryProblem` states *what* to optimise — dynamics, boundary
conditions, path constraints, control bounds, and a staging structure — without
committing to *how* it is solved; :mod:`aether.trajectory.solve` maps it
onto a concrete backend. This mirrors the separation every mature optimal-control
package keeps between the problem and the transcription (Betts; Rao's GPOPS-II).

Two things are first-class here because the engagement problem needs them:

* **Multi-phase staging.** A trajectory is a sequence of :class:`PhaseSpec` s —
  boost, coast, glide, terminal — each with its own dynamics and free duration.
  The terminal state of one phase is mapped into the initial state of the next
  by a *linkage* (identity by default; a mass drop at staging, a fairing jettison,
  a bus deployment otherwise). Free phase durations make the staging *schedule*
  itself a decision variable, which is what "optimal staging timing" asks for.

* **Path constraints as functions.** Waypoints and no-fly zones (Jorris, *Common
  Aero Vehicle Autonomous Reentry Trajectory Optimization*, AIAA 2007) and the
  thermal / g-load limits of :mod:`aether.guidance.entry_ocp` are all
  expressed as ``g(x, u) >= 0`` (a feasibility margin) and enforced at the collocation nodes.

Time-varying dynamics (a rotating Earth, an altitude-dependent atmosphere) enter
through the standard device of augmenting the state with time itself
(``dt/dtau = 1``) rather than threading ``t`` through every callback, keeping the
dynamics signature ``(x, u) -> xdot`` compatible with the collocation backend.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from aether.trajectory.objectives import MinFlightTime, Objective

_FloatArray = NDArray[np.float64]

__all__ = ["PhaseSpec", "TrajectoryProblem"]

#: ``dynamics(x, u) -> xdot`` and ``path_constraints(x, u) -> g`` with ``g >= 0`` feasible.
DynamicsFn = Callable[[_FloatArray, _FloatArray], _FloatArray]
ConstraintFn = Callable[[_FloatArray, _FloatArray], _FloatArray]
#: ``linkage(x_terminal) -> x_initial_next`` across a phase boundary.
LinkageFn = Callable[[_FloatArray], _FloatArray]


def _no_constraints(x: _FloatArray, u: _FloatArray) -> _FloatArray:
    return np.zeros(0)


@dataclass(frozen=True)
class PhaseSpec:
    """One phase of a staged trajectory.

    Attributes
    ----------
    label:
        Phase name (``"boost-1"``, ``"coast"``, ``"glide"``, ``"terminal"``).
    dynamics:
        ``f(x, u) -> xdot`` on this phase.
    nu:
        Number of controls on this phase (``0`` for an unpowered coast).
    duration_guess:
        Initial guess for the free phase duration (s).
    path_constraints:
        ``g(x, u) >= 0`` (feasibility margin) enforced at the nodes on this phase.
    u_bounds:
        ``(lower, upper)`` control bounds, or ``None``.
    running_cost:
        Optional Lagrange integrand ``L(x, u) -> float`` for this phase.
    nodes:
        LGL node count for the phase's collocation.
    max_iter:
        NLP iteration budget for this phase. Exposed because it is a real
        tuning knob and was previously reachable only by editing
        :mod:`aether.optimal_control.pseudospectral` -- a phase that needed
        more iterations than the default had no way to ask for them, and
        failed with ``SolveError: Iteration limit reached`` rather than with
        anything that pointed at the cause.
    linkage:
        Maps this phase's terminal state to the next phase's initial state.
        Identity by default; a staging event overrides it (drop a stage's mass,
        change the dynamics regime). Ignored on the final phase.
    """

    label: str
    dynamics: DynamicsFn
    nu: int = 2
    duration_guess: float = 100.0
    path_constraints: ConstraintFn = _no_constraints
    u_bounds: tuple[_FloatArray, _FloatArray] | None = None
    running_cost: Callable[[_FloatArray, _FloatArray], float] | None = None
    nodes: int = 30
    max_iter: int = 400
    linkage: LinkageFn | None = None
    #: Declarative event triggers ``g(x_terminal) == 0`` that close this phase on
    #: a physical condition (mass depletion, an altitude crossing, a target
    #: speed) rather than at a freely chosen time. See
    #: :mod:`aether.trajectory.triggers`. Empty leaves the phase's terminal
    #: time free, the existing behaviour.
    terminal_constraints: tuple[Callable[[_FloatArray], float], ...] = ()
    #: Called once per NLP iteration on this phase as
    #: ``progress(iteration, cost, violation)``. Passed straight through to the
    #: transcription. Worth setting on anything a person is waiting for: a
    #: four-hundred-iteration phase is otherwise silent from start to finish.
    progress: Callable[[int, float, float], None] | None = None
    #: Warm start ``(x, u, tf)`` sampled at this phase's LGL nodes, passed
    #: through to the transcription.
    #:
    #: Not optional in practice for an interior phase. Interior phases are
    #: solved to an all-free terminal target, and the transcription's
    #: straight-line cold start needs a target to draw a line to -- without one
    #: it hands SLSQP a guess that never leaves the pad, which surfaces as
    #: ``Inequality constraints incompatible`` rather than as anything pointing
    #: at the guess. One forward integration fixes it.
    initial_guess: tuple[_FloatArray, _FloatArray, float] | None = None


@dataclass(frozen=True)
class TrajectoryProblem:
    """A complete trajectory-optimization problem.

    Attributes
    ----------
    x0:
        Initial state (m, m/s, ... — whatever the dynamics use), shape ``(nx,)``.
    xf_target:
        Terminal target on the final phase. Entries may be ``inf`` to leave that
        component free (a target position with free terminal velocity, say).
    nx:
        State dimension, shared across phases.
    phases:
        Ordered phases. A single-phase problem is a one-element tuple.
    objective:
        The cost to minimise. Bolza-form objectives (time, control effort) map
        directly onto the collocation transcription; trajectory-functional
        objectives (undetected time, pursuit time) are scored by the generator
        / engagement layer, which searches problem parameters with the inner
        solver. Defaults to minimum flight time.
    x_bounds:
        Optional global ``(lower, upper)`` state bounds.
    label:
        Human-readable problem name, carried into the trajectory metadata.
    """

    x0: _FloatArray
    xf_target: _FloatArray
    nx: int
    phases: tuple[PhaseSpec, ...]
    objective: Objective = field(default_factory=MinFlightTime)
    x_bounds: tuple[_FloatArray, _FloatArray] | None = None
    label: str = "trajectory"

    def __post_init__(self) -> None:
        if not self.phases:
            raise ValueError("a problem needs at least one phase")
        object.__setattr__(self, "x0", np.asarray(self.x0, dtype=np.float64))
        object.__setattr__(self, "xf_target", np.asarray(self.xf_target, dtype=np.float64))
        if self.x0.shape != (self.nx,):
            raise ValueError(f"x0 must have shape ({self.nx},), got {self.x0.shape}")

    @property
    def is_multiphase(self) -> bool:
        return len(self.phases) > 1

    @property
    def total_duration_guess(self) -> float:
        return float(sum(p.duration_guess for p in self.phases))
