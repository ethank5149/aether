r"""Indirect optimal control: Pontryagin's principle, started from a direct solve.

A direct method discretises the trajectory and hands the whole thing to a
nonlinear program. It is robust and it converges from poor guesses, which is why
:mod:`aether.optimal_control.pseudospectral` exists -- but its answer satisfies
the dynamics only at the collocation nodes, and its accuracy is the accuracy of
the mesh. An indirect method instead writes down the conditions an optimal
trajectory must satisfy and solves those, so its answer is exact up to the
integrator. What it cannot do is find its own starting point: the costates have
no physical meaning to guess at, and a shooting method launched from a bad
initial covector diverges immediately.

So the two are used together, in the order the architecture note specifies:
pseudospectral first for a guess that is close, Pontryagin second for an answer
that is right.

The bridge between them is not improvised. Ross & Fahroo's **covector mapping
theorem** ("A Pseudospectral Transformation of the Covectors of Optimal Control
Systems", IFAC 2001; and Ross & Fahroo, *J. Guidance, Control, and Dynamics*
27(3), 2004) establishes that the KKT multipliers of the discretised problem
converge to the costates of the continuous one under a mesh-dependent
transformation -- so the direct solve does not merely supply a trajectory, it
supplies an estimate of :math:`\lambda` itself. :func:`covector_estimate`
implements that map; without it there is nothing to initialise the shooter with.

The conditions being solved, for a problem with running cost :math:`L` and
dynamics :math:`\dot x = f(x,u)`:

.. math::

    H(x, u, \lambda) &= L(x,u) + \lambda^{\mathsf T} f(x,u), \\
    \dot x &= \partial H/\partial\lambda = f(x,u), \\
    \dot\lambda &= -\partial H/\partial x, \\
    u^\star &= \arg\min_u H,

with :math:`\lambda(t_f)` fixed by the terminal conditions. The minimisation is
Pontryagin's *minimum* principle rather than a stationarity condition, which
matters wherever the control is bounded: at a bound the derivative
:math:`\partial H/\partial u` need not vanish, and a solver that only zeroes it
will miss bang-bang and singular arcs entirely. :func:`minimising_control`
therefore minimises over the admissible set rather than solving for a stationary
point.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.integrate
import scipy.optimize
from numpy.typing import NDArray

from aether.optimal_control.pseudospectral import OCPProblem, OCPSolution

_FloatArray = NDArray[np.float64]

__all__ = [
    "HamiltonianSystem",
    "IndirectSolution",
    "control_jacobian",
    "costate_dynamics",
    "covector_estimate",
    "hamiltonian",
    "minimising_control",
    "refine_indirect",
    "refine_indirect_free_time",
]


def hamiltonian(
    problem: OCPProblem,
    state: _FloatArray,
    control: _FloatArray,
    costate: _FloatArray,
) -> float:
    r""":math:`H = L(x,u) + \lambda^{\mathsf T} f(x,u)`."""
    running = 0.0
    if problem.running_cost is not None:
        running = float(problem.running_cost(state, control))
    return float(running + np.dot(costate, problem.dynamics(state, control)))


def minimising_control(
    problem: OCPProblem,
    state: _FloatArray,
    costate: _FloatArray,
    *,
    guess: _FloatArray | None = None,
) -> _FloatArray:
    """The control minimising the Hamiltonian at this state and costate.

    Minimised over the admissible set, not solved for a stationary point.
    Pontryagin's principle asks for the minimum, and on a bounded control the
    two differ exactly where the interesting behaviour is: a bang-bang arc sits
    against its bound with a non-zero gradient, and a stationarity solver walks
    straight past it.

    Uses ``problem.control_argmin`` when the problem supplies one. This is not
    an optimisation of convenience: the minimum is needed at every mesh node on
    every iteration of the boundary value solve, so a numerical search here is
    the cost of the whole indirect stage rather than a detail of it.
    """
    if problem.control_argmin is not None:
        return np.atleast_1d(
            np.asarray(problem.control_argmin(state, costate), dtype=np.float64)
        )
    start = np.zeros(problem.nu) if guess is None else np.asarray(guess, float)
    bounds = None
    if problem.u_bounds is not None:
        lower, upper = problem.u_bounds
        bounds = list(zip(np.atleast_1d(lower), np.atleast_1d(upper), strict=True))
        start = np.clip(start, np.atleast_1d(lower), np.atleast_1d(upper))

    result = scipy.optimize.minimize(
        lambda u: hamiltonian(problem, state, np.atleast_1d(u), costate),
        start, method="L-BFGS-B" if bounds else "BFGS", bounds=bounds,
    )
    return np.atleast_1d(np.asarray(result.x, dtype=np.float64))


def costate_dynamics(
    problem: OCPProblem,
    state: _FloatArray,
    control: _FloatArray,
    costate: _FloatArray,
    *,
    step: float = 1e-6,
) -> _FloatArray:
    r""":math:`\dot\lambda = -\partial H/\partial x`, by central differences.

    Central rather than forward differencing: the costate equation is integrated
    backwards over the whole trajectory, so a first-order truncation error here
    accumulates into the initial covector the shooter is trying to find.
    """
    state = np.asarray(state, dtype=np.float64)
    gradient = np.empty(problem.nx, dtype=np.float64)
    for i in range(problem.nx):
        h = step * max(abs(float(state[i])), 1.0)
        forward, backward = state.copy(), state.copy()
        forward[i] += h
        backward[i] -= h
        gradient[i] = (
            hamiltonian(problem, forward, control, costate)
            - hamiltonian(problem, backward, control, costate)
        ) / (2.0 * h)
    return -gradient


@dataclass(frozen=True)
class HamiltonianSystem:
    """The coupled state-costate system Pontryagin's conditions define."""

    problem: OCPProblem

    def derivatives(self, _t: float | _FloatArray, y: _FloatArray) -> _FloatArray:
        """``[x, lambda]`` derivatives with the minimising control substituted.

        Accepts a single state or a whole mesh. ``scipy.integrate.solve_bvp``
        evaluates the right-hand side vectorised -- ``y`` arrives shaped
        ``(2 nx, m)`` for the m mesh nodes -- and a function written for one
        column silently receives a matrix, which is how this first failed.
        """
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            return self._one(y)
        return np.column_stack([self._one(y[:, k]) for k in range(y.shape[1])])

    def _one(self, y: _FloatArray) -> _FloatArray:
        """Derivatives at a single state-costate point."""
        nx = self.problem.nx
        state, costate = y[:nx], y[nx:]
        control = minimising_control(self.problem, state, costate)
        return np.concatenate([
            self.problem.dynamics(state, control),
            costate_dynamics(self.problem, state, control, costate),
        ])

    def controls_along(
        self, states: _FloatArray, costates: _FloatArray
    ) -> _FloatArray:
        """Recover the optimal control history from a state-costate trajectory."""
        return np.array([
            minimising_control(self.problem, x, lam)
            for x, lam in zip(states, costates, strict=True)
        ])


def control_jacobian(
    problem: OCPProblem,
    state: _FloatArray,
    control: _FloatArray,
    *,
    step: float = 1e-6,
) -> _FloatArray:
    r""":math:`\partial f/\partial u`, shape ``(nx, nu)``, by central differences."""
    control = np.asarray(control, dtype=np.float64)
    jacobian = np.empty((problem.nx, problem.nu), dtype=np.float64)
    for j in range(problem.nu):
        h = step * max(abs(float(control[j])), 1.0)
        forward, backward = control.copy(), control.copy()
        forward[j] += h
        backward[j] -= h
        jacobian[:, j] = (
            problem.dynamics(state, forward) - problem.dynamics(state, backward)
        ) / (2.0 * h)
    return jacobian


def covector_estimate(
    problem: OCPProblem,
    solution: OCPSolution,
) -> _FloatArray:
    r"""Costates at each node, estimated from a converged direct solution.

    The covector mapping in practice. The direct solve already satisfies the
    optimality conditions to its mesh accuracy, so the costate is recoverable
    from the one condition that links it to the control -- stationarity of the
    Hamiltonian,

    .. math::

        \frac{\partial L}{\partial u}
        + \left(\frac{\partial f}{\partial u}\right)^{\mathsf T}\lambda = 0,

    solved in least squares at every node. Where the controls do not span the
    state space this pins only some components of :math:`\lambda` and the
    minimum-norm solution supplies the rest; the boundary value solve then
    corrects them. That is the intended division of labour, and it is why this
    is called an estimate.

    **Not** a backward sweep seeded with :math:`\lambda(t_f)=0`. That
    transversality condition holds only for terminal states left *free*, and
    applying it to a problem whose terminal state is fixed is not a small error:
    for a minimum-energy double integrator it makes :math:`\lambda` vanish
    identically, hence :math:`u \equiv 0`, and hands the boundary value solver a
    singular Jacobian at a point that is not a solution.
    """
    states = np.asarray(solution.x, dtype=np.float64)
    controls = np.atleast_2d(np.asarray(solution.u, dtype=np.float64))
    if controls.shape[0] != states.shape[0]:
        controls = controls.T
    n_nodes, nx = states.shape
    costates = np.zeros((n_nodes, nx), dtype=np.float64)

    for k in range(n_nodes):
        state, control = states[k], np.atleast_1d(controls[k])
        jacobian = control_jacobian(problem, state, control)
        # dL/du by central differences, so a cost with no closed form still works.
        cost_gradient = np.zeros(problem.nu, dtype=np.float64)
        if problem.running_cost is not None:
            for j in range(problem.nu):
                h = 1e-6 * max(abs(float(control[j])), 1.0)
                forward, backward = control.copy(), control.copy()
                forward[j] += h
                backward[j] -= h
                cost_gradient[j] = (
                    float(problem.running_cost(state, forward))
                    - float(problem.running_cost(state, backward))
                ) / (2.0 * h)
        costates[k] = np.linalg.lstsq(jacobian.T, -cost_gradient, rcond=None)[0]

    return costates


@dataclass(frozen=True)
class IndirectSolution:
    """A trajectory satisfying Pontryagin's conditions."""

    t: _FloatArray
    x: _FloatArray
    u: _FloatArray
    costates: _FloatArray
    converged: bool
    residual: float
    message: str = ""


def refine_indirect(
    problem: OCPProblem,
    guess: OCPSolution,
    *,
    tol: float = 1e-6,
    max_nodes: int = 20000,
) -> IndirectSolution:
    """Refine a direct solution by solving the two-point boundary value problem.

    Takes the pseudospectral trajectory and its covector estimate as the initial
    mesh, then solves the state-costate system with the terminal state as the
    boundary condition. Returns with ``converged`` false rather than raising if
    the BVP solver fails -- an unconverged indirect solve still carries the
    direct answer it started from, and the caller needs to know which one it is
    holding.
    """
    system = HamiltonianSystem(problem)
    nx = problem.nx
    times = np.asarray(guess.t, dtype=np.float64)
    states = np.asarray(guess.x, dtype=np.float64)
    target = np.asarray(problem.xf_target, dtype=np.float64)
    # Estimating the covector is part of setting up the shot, so a failure there
    # is reported the same way a failure to converge is -- the caller still holds
    # the direct solution and needs to know it was not improved on.
    try:
        costates = covector_estimate(problem, guess)
    except Exception as exc:
        return IndirectSolution(
            t=times, x=states, u=np.asarray(guess.u, dtype=np.float64),
            costates=np.zeros_like(states), converged=False,
            residual=float("inf"),
            message=f"covector estimate failed: {exc}",
        )
    y0 = np.vstack([states.T, costates.T])

    def boundary(ya: _FloatArray, yb: _FloatArray) -> _FloatArray:
        """Initial state pinned, terminal state driven to the target."""
        return np.concatenate([ya[:nx] - states[0], yb[:nx] - target])

    try:
        result = scipy.integrate.solve_bvp(
            system.derivatives, boundary, times, y0,
            tol=tol, max_nodes=max_nodes,
        )
    except Exception as exc:
        return IndirectSolution(
            t=times, x=states, u=np.asarray(guess.u, dtype=np.float64),
            costates=costates, converged=False, residual=float("inf"),
            message=f"boundary value solve raised: {exc}",
        )

    refined_states = result.y[:nx].T
    refined_costates = result.y[nx:].T
    controls = system.controls_along(refined_states, refined_costates)
    miss = float(np.linalg.norm(refined_states[-1] - target))
    return IndirectSolution(
        t=np.asarray(result.x, dtype=np.float64),
        x=refined_states,
        u=controls,
        costates=refined_costates,
        converged=bool(result.success),
        residual=miss,
        message=str(result.message),
    )


def refine_indirect_free_time(
    problem: OCPProblem,
    guess: OCPSolution,
    *,
    costate_guess: _FloatArray | None = None,
    smoothing: tuple[float, ...] = (),
    tol: float = 1e-8,
    max_nodes: int = 60000,
) -> IndirectSolution:
    r"""Indirect solve for a **free final time**, using transversality instead of an estimate.

    The recovery from :func:`covector_estimate`'s structural limit. That
    function reads the costates off the stationarity condition
    :math:`\partial L/\partial u + (\partial f/\partial u)^{\mathsf T}\lambda = 0`,
    which is a homogeneous system whenever the running cost does not contain the
    control -- so minimum-time (:math:`L = 1`) and maximum-range
    (:math:`L = -V\cos\gamma/R_E`) both hand it :math:`\lambda \equiv 0`, the
    same degenerate seed that :math:`\lambda(t_f)=0` produced on a fixed
    endpoint. Since those are two of the three objectives anyone actually poses,
    the limit is not a corner case.

    This route never estimates the costates at all. For a free final time the
    transversality conditions supply them outright:

    * :math:`\lambda_j(t_f) = 0` for every terminal component left free, and
      :math:`x_j(t_f) = x_{f,j}` for every one that is fixed -- exactly ``nx``
      conditions between them, whichever way each component falls;
    * :math:`H(t_f) = 0`, because :math:`t_f` is itself a decision variable.

    That last condition is what makes the difference. The costate equations are
    homogeneous in :math:`\lambda`, so :math:`\lambda` and :math:`c\lambda`
    satisfy the same dynamics and the same zero terminal conditions -- the scale
    is genuinely undetermined until something fixes it, and
    :math:`H(t_f) = 0` is that something. With autonomous dynamics :math:`H` is
    conserved, so this is not merely a terminal condition but the statement
    :math:`H \equiv 0` along the whole trajectory, which is simultaneously the
    strongest available check on the answer.

    Counting: ``nx`` initial conditions, ``nx`` terminal, one for
    :math:`H(t_f)=0`, against ``2 nx`` state-costate components plus the unknown
    :math:`t_f`. Square, with no free parameter left to guess.

    ``smoothing`` runs a continuation on a bang-bang control. Where the control
    saturates, :math:`u^\star` jumps and the boundary value solver meets a
    right-hand side with no derivative -- it reports a singular Jacobian and
    stops. Replacing the switch by a graded one and tightening it in stages
    walks the solution in. Pass a decreasing sequence such as
    ``(1.0, 0.3, 0.1)``; leave it empty for a problem whose minimising control
    is already smooth in the costates, which is the common case and includes the
    entry glide, where :math:`\sigma^\star = \operatorname{atan2}(-B,-A)`.

    **Cost.** :func:`costate_dynamics` central-differences
    :math:`\partial H/\partial x`, which is ``2 nx`` dynamics evaluations per
    node per Newton step, and the boundary value solver refines its own mesh on
    top of that. It is comfortable on the two-state problems and expensive at
    six; an analytic :math:`\partial H/\partial x` is the fix, and is not
    implemented.
    """
    nx = problem.nx
    target = np.asarray(problem.xf_target, dtype=np.float64)
    fixed = np.isfinite(target)
    x0 = np.asarray(problem.x0, dtype=np.float64)
    states = np.asarray(guess.x, dtype=np.float64)

    if costate_guess is None:
        costates = np.zeros_like(states)
        # Any non-zero seed beats the zero covector, which is a fixed point of
        # the whole system rather than a poor starting point.
        costates[:, min(nx - 1, 3)] = -1.0
    else:
        costates = np.asarray(costate_guess, dtype=np.float64)
        if costates.shape != states.shape:
            raise ValueError(
                f"costate_guess must match the state history's shape "
                f"{states.shape}, got {costates.shape}"
            )

    times = np.asarray(guess.t, dtype=np.float64)
    span = float(times[-1] - times[0])
    if span <= 0.0:
        raise ValueError(
            f"the guess must cover a positive time span to be rescaled onto "
            f"[0, 1]; got t from {times[0]} to {times[-1]}"
        )
    normalised = (times - times[0]) / span

    def control_at(state: _FloatArray, costate: _FloatArray, rho: float) -> _FloatArray:
        r"""The minimising control, optionally graded to make the switch smooth.

        The smoothing has to act on the control's *dependence on the costate*,
        not on its magnitude. A first attempt scaled the saturated command back
        toward the interior -- which changes how large the control is while
        leaving :math:`u^\star` jumping from one bound to the other exactly
        where it did before, so the right-hand side is no more differentiable
        than it was and the solver reports the same singular Jacobian.

        What is graded instead is the switching function
        :math:`g = \partial H/\partial u`. For a Hamiltonian linear in the
        control the exact minimiser is :math:`u_j = \text{mid}_j -
        \text{half}_j\operatorname{sign}(g_j)`, and replacing the sign by
        :math:`\tanh(g_j/\rho)` gives a control that is smooth in
        :math:`\lambda` and recovers the bang-bang law as
        :math:`\rho \to 0`.
        """
        if rho <= 0.0 or problem.u_bounds is None:
            return minimising_control(problem, state, costate)
        lower = np.atleast_1d(np.asarray(problem.u_bounds[0], dtype=np.float64))
        upper = np.atleast_1d(np.asarray(problem.u_bounds[1], dtype=np.float64))
        mid = 0.5 * (lower + upper)
        half = 0.5 * (upper - lower)
        switching = np.empty(problem.nu, dtype=np.float64)
        for j in range(problem.nu):
            step = 1e-6 * max(abs(float(mid[j])), 1.0)
            forward, backward = mid.copy(), mid.copy()
            forward[j] += step
            backward[j] -= step
            switching[j] = (
                hamiltonian(problem, state, forward, costate)
                - hamiltonian(problem, state, backward, costate)
            ) / (2.0 * step)
        return np.asarray(mid - half * np.tanh(switching / rho), dtype=np.float64)

    def make(
        rho: float,
    ) -> tuple[
        Callable[[_FloatArray, _FloatArray, _FloatArray], _FloatArray],
        Callable[[_FloatArray, _FloatArray, _FloatArray], _FloatArray],
    ]:
        def derivatives(_s: _FloatArray, y: _FloatArray, par: _FloatArray) -> _FloatArray:
            duration = float(par[0])
            out = np.empty_like(y)
            for k in range(y.shape[1]):
                state, costate = y[:nx, k], y[nx:, k]
                control = control_at(state, costate, rho)
                out[:nx, k] = problem.dynamics(state, control)
                out[nx:, k] = costate_dynamics(problem, state, control, costate)
            return duration * out

        def boundary(
            ya: _FloatArray, yb: _FloatArray, _par: _FloatArray
        ) -> _FloatArray:
            conditions = list(ya[:nx] - x0)
            for j in range(nx):
                # Fixed component: pin the state. Free component: transversality
                # pins its costate to zero instead. Exactly one applies to each.
                conditions.append(yb[j] - target[j] if fixed[j] else yb[nx + j])
            terminal_control = control_at(yb[:nx], yb[nx:], rho)
            conditions.append(
                hamiltonian(problem, yb[:nx], terminal_control, yb[nx:])
            )
            return np.asarray(conditions, dtype=np.float64)

        return derivatives, boundary

    mesh = normalised
    values = np.vstack([states.T, costates.T])
    duration = np.array([span])
    result = None
    for rho in (*smoothing, 0.0):
        derivatives, boundary = make(rho)
        try:
            result = scipy.integrate.solve_bvp(
                derivatives, boundary, mesh, values, p=duration,
                tol=tol, max_nodes=max_nodes,
            )
        except Exception as exc:
            return IndirectSolution(
                t=times, x=states, u=np.asarray(guess.u, dtype=np.float64),
                costates=costates, converged=False, residual=float("inf"),
                message=f"boundary value solve raised at rho={rho}: {exc}",
            )
        if not result.success:
            return IndirectSolution(
                t=times, x=states, u=np.asarray(guess.u, dtype=np.float64),
                costates=costates, converged=False, residual=float("inf"),
                message=f"did not converge at rho={rho}: {result.message}",
            )
        mesh, values, duration = result.x, result.y, result.p

    assert result is not None
    final = float(duration[0])
    refined_states = result.y[:nx].T
    refined_costates = result.y[nx:].T
    controls = np.array([
        minimising_control(problem, x, lam)
        for x, lam in zip(refined_states, refined_costates, strict=True)
    ])
    # The residual reported is the Hamiltonian drift, not a terminal miss: the
    # endpoints are boundary conditions and are met by construction, whereas
    # H = 0 is enforced only at t_f and must hold everywhere if the answer is
    # right. It is the one number here the solver did not fit.
    energies = [
        hamiltonian(problem, refined_states[k], controls[k], refined_costates[k])
        for k in range(refined_states.shape[0])
    ]
    return IndirectSolution(
        t=times[0] + final * np.asarray(result.x, dtype=np.float64),
        x=refined_states,
        u=controls,
        costates=refined_costates,
        converged=True,
        residual=float(max(energies) - min(energies)),
        message=f"converged with t_f = {final:.6f}",
    )
