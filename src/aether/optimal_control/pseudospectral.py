"""Pseudospectral direct transcription for optimal control problems.

This module solves trajectory optimization problems using Legendre-Gauss-Lobatto
(LGL) pseudospectral collocation. The method converts the infinite-dimensional
continuous-time optimal control problem into a finite-dimensional nonlinear
programming (NLP) problem by:

1. Discretizing the time domain into LGL nodes (clustered at endpoints)
2. Approximating states and controls with Lagrange polynomials
3. Enforcing dynamics via defect constraints at interior nodes
4. Enforcing path constraints at all nodes
5. Solving the resulting NLP with ``scipy.optimize.minimize`` (SLSQP)

Swappable solver interface
--------------------------

Both :mod:`aether.optimal_control.pseudospectral` and :mod:`aether.optimal_control.scvx`
implement the same protocol:

* **Problem specification:** a :class:`OCPProblem` dataclass containing
  dynamics, path constraints, boundary conditions, and bounds.
* **Solution:** an :class:`OCPSolution` dataclass containing the optimized
  state/control histories, cost, and solver diagnostics.
* **Solver entry point:** :func:`solve_ocp` accepts an ``OCPProblem`` and
  returns an ``OCPSolution``.

Callers can swap solvers by passing a different ``solver`` argument to
:func:`solve_ocp`. Supported values are ``"pseudospectral"`` (default) and
``"scvx"``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.integrate
import scipy.interpolate
import scipy.optimize
from numpy.polynomial.legendre import Legendre
from numpy.typing import NDArray

__all__ = [
    "OCPProblem",
    "OCPSolution",
    "mesh_error",
    "solve_ocp",
]

_FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OCPProblem:
    """Optimal control problem specification.

    Attributes
    ----------
    dynamics:
        ``f(x, u) -> xdot``. Must accept and return 1-D arrays of length
        ``nx`` and ``nu`` respectively.
    path_constraints:
        ``g(x, u) -> NDArray``. Inequality constraints, **feasible where
        ``g >= 0``**, evaluated at each node. Return an empty array if there are
        no path constraints.

        This docstring said ``c <= 0`` until it was checked against the code.
        The values are handed to SLSQP as an ``"ineq"`` constraint without a
        sign change, and SLSQP's ``"ineq"`` means non-negative, so the
        documented convention was the exact inverse of the enforced one — while
        a sibling problem definition, which feeds the same callable
        straight through, documented ``g >= 0`` correctly. Nothing caught it
        because every constraint in the suite was either empty or a positive
        constant. A limit written to the wrong convention here does not fail
        loudly: it becomes a *requirement to exceed* the limit, and the solver
        will happily fly the trajectory into it.
    x0:
        Initial state vector, shape ``(nx,)``.
    xf_target:
        Target terminal state. NaN entries are treated as free (unconstrained).
    nx:
        State dimension.
    nu:
        Control dimension.
    N:
        Number of LGL collocation nodes.
    t0:
        Initial time (s).
    tf_guess:
        Initial guess for final time (s).
    u_bounds:
        ``(u_min, u_max)`` each of shape ``(nu,)``. Controls are bounded.
    x_bounds:
        ``(x_min, x_max)`` each of shape ``(nx,)``. States are bounded.
    running_cost:
        ``L(x, u) -> float``, integrated over the arc with the LGL
        quadrature weights and added to the objective. ``None`` leaves the
        problem as pure minimum time, which is what it was fixed at before.
        This is where a chance-constraint penalty belongs — see TODO 9.13.
    time_weight:
        Weight on the final time in the objective. Set it to zero to
        optimise a running cost at a fixed horizon.
    max_iter:
        Maximum NLP iterations. 400 rather than 200 because collocating at
        every node -- including the initial one, which this transcription used
        to skip -- adds ``nx`` equality constraints and costs iterations to
        satisfy them. The minimum-time double integrator lands on its analytic
        optimum of exactly 10.000000 s in under 400 and reaches only 10.000882
        in 200, so the old budget cut off just short of convergence and
        reported ``success=False`` from a nearly-right answer.
    """

    dynamics: Callable[[_FloatArray, _FloatArray], _FloatArray]
    path_constraints: Callable[[_FloatArray, _FloatArray], _FloatArray]
    x0: _FloatArray
    xf_target: _FloatArray
    nx: int = 6
    nu: int = 2
    N: int = 30
    t0: float = 0.0
    tf_guess: float = 300.0
    u_bounds: tuple[_FloatArray, _FloatArray] | None = None
    x_bounds: tuple[_FloatArray, _FloatArray] | None = None
    running_cost: Callable[[_FloatArray, _FloatArray], float] | None = None
    time_weight: float = 1.0
    max_iter: int = 400
    #: Extra terminal event constraints ``g(x_terminal) == 0``, beyond the
    #: per-component ``xf_target``. Each closes a phase on a physical condition —
    #: mass depletion, an altitude crossing, a target speed — rather than at a
    #: freely chosen time. Empty by default, so the transcription is unchanged.
    terminal_constraints: tuple[Callable[[_FloatArray], float], ...] = ()
    #: Called once per NLP iteration as ``progress(iteration, cost, violation)``.
    #: ``None`` by default, so the transcription is unchanged.
    #:
    #: Worth passing on anything long. A phase is allowed four hundred SLSQP
    #: iterations and emits nothing until it stops, which from outside is
    #: indistinguishable from a hang -- the same problem the mesher solved by
    #: reporting stages. The callback is given the running cost and the largest
    #: equality-constraint violation, which together say whether an iteration
    #: budget is being spent converging or thrashing.
    progress: Callable[[int, float, float], None] | None = None

    #: Optional warm start ``(x, u, tf)`` with ``x`` shaped ``(N+1, nx)`` and
    #: ``u`` shaped ``(N+1, nu)``, both sampled at the LGL nodes. ``None`` keeps
    #: the straight-line cold start, which is adequate whenever ``xf_target``
    #: says where the trajectory is going and useless when it does not — a
    #: problem closing on terminal speed alone gets a guess that never moves
    #: downrange. Supplying a flown trajectory costs one integration and is the
    #: difference between converging and not.
    initial_guess: tuple[_FloatArray, _FloatArray, float] | None = None
    #: Optional closed-form minimiser of the Hamiltonian,
    #: ``argmin(x, costate) -> u``, used by
    #: :func:`~aether.optimal_control.indirect.minimising_control` in place of a
    #: numerical search. Pontryagin's principle needs that minimum at *every*
    #: mesh node on *every* iteration of the boundary value solve, so a
    #: per-node BFGS is the bottleneck of an indirect solve long before the
    #: dynamics are. Where the control enters the Hamiltonian in a form that can
    #: be minimised by hand -- a bank angle appears only as
    #: ``A cos(sigma) + B sin(sigma)`` -- supplying it is the difference between
    #: an indirect stage that runs and one that does not. ``None`` keeps the
    #: numerical search, which is correct but slow.
    control_argmin: Callable[[_FloatArray, _FloatArray], _FloatArray] | None = None


@dataclass(frozen=True)
class OCPSolution:
    """Solution of an optimal control problem.

    Attributes
    ----------
    x:
        Optimal state history, shape ``(N, nx)``.
    u:
        Optimal control history, shape ``(N, nu)``.
    t:
        Time nodes, shape ``(N,)``.
    tf:
        Final time (s).
    cost:
        Final objective value.
    success:
        Whether the NLP converged.
    message:
        Solver diagnostic message.
    nfev:
        Number of objective evaluations.
    njev:
        Number of Jacobian evaluations.
    """

    x: _FloatArray
    u: _FloatArray
    t: _FloatArray
    tf: float
    cost: float
    success: bool
    message: str
    nfev: int = 0
    njev: int = 0


# ---------------------------------------------------------------------------
# LGL nodes and differentiation matrix
# ---------------------------------------------------------------------------


def lgl_nodes(n: int) -> tuple[_FloatArray, _FloatArray]:
    """Legendre-Gauss-Lobatto nodes and weights on ``[-1, 1]``.

    Parameters
    ----------
    n:
        Number of nodes. Must be >= 2.

    Returns
    -------
    nodes:
        LGL nodes, shape ``(n,)``, clustered at endpoints.
    weights:
        Quadrature weights, shape ``(n,)``.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")

    if n == 2:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])

    # The interior LGL nodes are the roots of P'_{n-1}, **not** the
    # Gauss-Legendre nodes. Taking `leggauss(n - 2)` for them — which is what
    # this did — places them wrong by up to 0.13 at n = 4 and 0.096 at n = 6,
    # and the collocation built on them is no longer spectrally accurate.
    legendre = Legendre.basis(n - 1)
    interior = np.sort(np.real(legendre.deriv().roots()))
    nodes = np.concatenate([[-1.0], interior, [1.0]])

    # w_i = 2 / (n (n-1) [P_{n-1}(x_i)]^2), at **every** node including the
    # endpoints: there P_{n-1}(+-1) = +-1, so the formula returns the familiar
    # 2 / (n (n-1)) rather than the 1 / (n (n-1)) written here before. Those
    # two errors compounded: the weights summed to 4.32 at n = 8 against the
    # 2.0 that integrating the constant function over [-1, 1] requires, and
    # the discrepancy *grew* with n — refinement made the quadrature worse.
    values = legendre(nodes)
    weights = 2.0 / (n * (n - 1) * values**2)

    return nodes.astype(np.float64), weights.astype(np.float64)


def differentiation_matrix(nodes: _FloatArray) -> _FloatArray:
    """Pseudospectral differentiation matrix for given nodes.

    Computes the Lagrange interpolation derivative matrix such that
    ``f'(x_i) = sum_j D_ij * f(x_j)``.

    Parameters
    ----------
    nodes:
        Collocation nodes, shape ``(n,)``.

    Returns
    -------
    D:
        Differentiation matrix, shape ``(n, n)``.
    """
    n = nodes.shape[0]
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i, j] = _lagrange_weight(nodes, i) / (
                    _lagrange_weight(nodes, j) * (nodes[i] - nodes[j])
                )

    # Diagonal entries: negative sum of off-diagonal. This is the "negative
    # sum trick", and it is what makes the row sums vanish to machine
    # precision so that a constant state has zero derivative.
    for i in range(n):
        matrix[i, i] = -np.sum(matrix[i, :])

    return matrix


def _lagrange_weight(nodes: _FloatArray, i: int) -> float:
    """Compute the Lagrange polynomial weight for node i."""
    n = nodes.shape[0]
    prod = 1.0
    xi = nodes[i]
    for j in range(n):
        if j != i:
            prod *= xi - nodes[j]
    return float(prod)


# ---------------------------------------------------------------------------
# Time mapping
# ---------------------------------------------------------------------------


def map_to_physical(tau: _FloatArray, t0: float, tf: float) -> _FloatArray:
    """Map LGL nodes from [-1, 1] to [t0, tf]."""
    return 0.5 * (tf - t0) * (tau + 1.0) + t0


def map_to_tau(t: _FloatArray, t0: float, tf: float) -> _FloatArray:
    """Map physical times to LGL domain [-1, 1]."""
    return 2.0 * (t - t0) / (tf - t0) - 1.0


# ---------------------------------------------------------------------------
# NLP formulation
# ---------------------------------------------------------------------------


def _build_nlp_variables(
    problem: OCPProblem,
) -> tuple[_FloatArray, _FloatArray, _FloatArray]:
    """Build initial NLP variable vector and bounds.

    The NLP variable vector is:
        z = [x_0, ..., x_N, u_0, ..., u_N, tf]

    Returns
    -------
    z0:
        Initial guess.
    lb:
        Lower bounds.
    ub:
        Upper bounds.
    """
    n_nodes = problem.N
    nx = problem.nx
    nu = problem.nu
    n_vars = (n_nodes + 1) * nx + (n_nodes + 1) * nu + 1  # +1 for tf

    z0 = np.zeros(n_vars)

    # A supplied guess wins outright. The straight line below is a reasonable
    # cold start for a problem whose target says where it is going, and no
    # start at all for one whose terminal state is mostly free: the entry glide
    # in `guidance.entry_ocp` targets only its terminal speed, so interpolation
    # holds longitude fixed and guesses a vehicle that never moves downrange.
    # Every defect is then violated at iteration zero and SLSQP stops on
    # "positive directional derivative" without leaving the guess.
    if problem.initial_guess is not None:
        guess_x, guess_u, guess_tf = problem.initial_guess
        guess_x = np.asarray(guess_x, dtype=np.float64)
        guess_u = np.atleast_2d(np.asarray(guess_u, dtype=np.float64))
        if guess_x.shape != (n_nodes + 1, nx):
            raise ValueError(
                f"initial_guess states must have shape {(n_nodes + 1, nx)} to match "
                f"N={n_nodes} and nx={nx}, got {guess_x.shape}"
            )
        if guess_u.shape != (n_nodes + 1, nu):
            raise ValueError(
                f"initial_guess controls must have shape {(n_nodes + 1, nu)}, got {guess_u.shape}"
            )
        z0[: (n_nodes + 1) * nx] = guess_x.reshape(-1)
        offset = (n_nodes + 1) * nx
        z0[offset : offset + (n_nodes + 1) * nu] = guess_u.reshape(-1)
        z0[-1] = float(guess_tf)
    else:
        # Initial state
        z0[:nx] = problem.x0

        # Remaining states: straight line from x0 to the stated terminal target,
        # holding whichever components are free. It used to interpolate towards
        # `np.zeros(nx)` — for a state in ECI metres that is the centre of the
        # Earth, so every cold start marched the guess into the planet.
        target = np.where(np.isfinite(problem.xf_target), problem.xf_target, problem.x0)
        for i in range(1, n_nodes + 1):
            z0[i * nx : (i + 1) * nx] = problem.x0 + (i / n_nodes) * (target - problem.x0)

        # Controls: zeros
        offset = (n_nodes + 1) * nx
        z0[offset : offset + (n_nodes + 1) * nu] = 0.0

        # Final time
        z0[-1] = problem.tf_guess

    # Bounds
    lb = np.full(n_vars, -np.inf)
    ub = np.full(n_vars, np.inf)

    # State bounds
    if problem.x_bounds is not None:
        x_min, x_max = problem.x_bounds
        for i in range(n_nodes + 1):
            lb[i * nx : (i + 1) * nx] = x_min
            ub[i * nx : (i + 1) * nx] = x_max

    # Control bounds
    if problem.u_bounds is not None:
        u_min, u_max = problem.u_bounds
        offset = (n_nodes + 1) * nx
        for i in range(n_nodes + 1):
            lb[offset + i * nu : offset + (i + 1) * nu] = u_min
            ub[offset + i * nu : offset + (i + 1) * nu] = u_max

    # Final time must be positive
    lb[-1] = max(problem.t0 + 1.0, 0.1)
    ub[-1] = problem.tf_guess * 10.0

    return z0, lb, ub


def _objective(z: _FloatArray, problem: OCPProblem) -> float:
    """Minimum final time, plus a running cost when the problem states one.

    The Lagrange term is integrated with the LGL quadrature weights, which
    is the only reason :func:`lgl_nodes` returns them. They were computed,
    returned, discarded at the call site, and wrong — so nothing caught that
    they summed to 4.32 instead of 2.0 at ``N = 8``. Using them is what
    makes that a testable claim rather than a latent one.
    """
    n_nodes = problem.N
    nx = problem.nx
    nu = problem.nu
    tf = float(z[-1])
    if problem.running_cost is None:
        return tf
    x = z[: (n_nodes + 1) * nx].reshape(n_nodes + 1, nx)
    u = z[(n_nodes + 1) * nx : (n_nodes + 1) * nx + (n_nodes + 1) * nu].reshape(n_nodes + 1, nu)
    _, weights = lgl_nodes(n_nodes + 1)
    scale = 0.5 * (tf - problem.t0)
    running = float(
        scale * sum(w * float(problem.running_cost(x[i], u[i])) for i, w in enumerate(weights))
    )
    return problem.time_weight * tf + running


def _dynamics_constraints(
    z: _FloatArray,
    problem: OCPProblem,
    tau: _FloatArray,
    diff_matrix: _FloatArray,
) -> _FloatArray:
    """Evaluate dynamics defect constraints.

    The collocation condition on the LGL domain is

    .. math::

        \\sum_j D_{ij}\\, x_j \\;=\\; \\frac{t_f - t_0}{2}\\, f(x_i, u_i),

    because the states are polynomials in :math:`\\tau \\in [-1, 1]` while the
    dynamics are in seconds. Three things were wrong with this and each was
    enough on its own:

    * **The time scale was dropped.** The residual formed was
      ``(f_i - D[i+1] @ x) / scale``, which vanishes when ``D x = f`` — the
      factor of :math:`(t_f - t_0)/2` is divided out of *both* terms instead
      of multiplying one of them. So the solver enforced dynamics on a
      two-unit interval whatever ``tf`` said, ``tf`` entered the constraints
      not at all, and a minimum-time objective with nothing pushing back on
      it simply drove ``tf`` to its lower bound.
    * **The node indices did not match.** ``D[i + 1] @ x`` is the derivative
      at node :math:`i+1` and ``f(x[i], u[i])`` is the dynamics at node
      :math:`i`. A defect is a statement about one node.
    * **Two nodes were left out.** The loop ran ``i`` over ``0..N-1`` and
      then skipped ``i == 0``, giving :math:`N-1` conditions for :math:`N`
      unknown state vectors. With a terminal target that is mostly NaN —
      which is the normal case here — the last state was pinned by nothing.

    Returns a flat vector of all defect constraints,
    :math:`(N+1) \\times n_x` of them -- one at **every** node, the initial one
    included.

    The initial defect used to be skipped, on the stated grounds that "the
    boundary condition fixes" it. It does not. The boundary condition
    :math:`x_0 - x(t_0) = 0` fixes the state's *value* at node 0; the
    collocation defect :math:`D_0 x = \\tfrac{1}{2}(t_f-t_0) f(x_0, u_0)`
    fixes its *derivative*, and they are independent statements. Dropping it
    left the interpolating polynomial's slope at :math:`\\tau = -1` free and
    made :math:`u_0` an orphan variable appearing in no constraint at all --
    so the transcription was a strict *relaxation* of the problem, and the
    optimiser duly exploited it.

    The symptom was a minimum time below the physically attainable one. The
    double integrator of ``test_min_time_reaches_the_target`` accelerates at
    20 m/s^2 to 1000 m from rest with the terminal velocity free, so it cannot
    arrive before :math:`\\sqrt{2 \\cdot 1000/20} = 10` s exactly; the solver
    returned 9.6903 s. Every minimum-time answer this module produced was
    optimistically low, approaching the true value from below as the mesh
    refined (9.11, 9.42, 9.69, 9.78 s at N = 6, 10, 20, 30) -- which is why a
    grid-refinement test that only checks the error *falls* could not see it.
    """
    n_nodes = problem.N
    nx = problem.nx
    nu = problem.nu
    tf = z[-1]
    t0 = problem.t0

    x = z[: (n_nodes + 1) * nx].reshape(n_nodes + 1, nx)
    u = z[(n_nodes + 1) * nx : (n_nodes + 1) * nx + (n_nodes + 1) * nu].reshape(n_nodes + 1, nu)

    # dt/dtau
    scale = 0.5 * (tf - t0)

    defects = [
        diff_matrix[i] @ x - scale * np.atleast_1d(problem.dynamics(x[i], u[i]))
        for i in range(n_nodes + 1)
    ]
    return np.concatenate([np.atleast_1d(d) for d in defects])


def _path_constraints_flat(
    z: _FloatArray,
    problem: OCPProblem,
) -> _FloatArray:
    """Evaluate path constraints at all nodes."""
    n_nodes = problem.N
    nx = problem.nx
    nu = problem.nu

    x = z[: (n_nodes + 1) * nx].reshape(n_nodes + 1, nx)
    u = z[(n_nodes + 1) * nx : (n_nodes + 1) * nx + (n_nodes + 1) * nu].reshape(n_nodes + 1, nu)

    constraints = []
    for i in range(n_nodes + 1):
        c = problem.path_constraints(x[i], u[i])
        constraints.append(c)

    return np.concatenate([np.atleast_1d(c) for c in constraints])


def _boundary_constraints(z: _FloatArray, problem: OCPProblem) -> _FloatArray:
    """Evaluate boundary condition constraints."""
    n_nodes = problem.N
    nx = problem.nx

    x = z[: (n_nodes + 1) * nx].reshape(n_nodes + 1, nx)

    constraints = []

    # Initial state constraint: x_0 = x0
    constraints.append(x[0] - problem.x0)

    # Terminal state constraint: x_N = xf_target (only for non-NaN entries)
    xf_target = problem.xf_target
    for j in range(nx):
        if np.isfinite(xf_target[j]):
            constraints.append(x[n_nodes, j] - xf_target[j])

    # Terminal event constraints g(x_N) == 0 — the declarative phase triggers.
    for g in problem.terminal_constraints:
        constraints.append(np.atleast_1d(np.float64(g(x[n_nodes]))))

    return np.concatenate([np.atleast_1d(c) for c in constraints])


def _nlp_constraints(
    z: _FloatArray,
    problem: OCPProblem,
    tau: _FloatArray,
    diff_matrix: _FloatArray,
) -> _FloatArray:
    """Combine all equality and inequality constraints."""
    eq_dynamics = _dynamics_constraints(z, problem, tau, diff_matrix)
    eq_boundary = _boundary_constraints(z, problem)
    ineq_path = _path_constraints_flat(z, problem)

    # Stack: equality first, then inequality
    return np.concatenate([eq_dynamics, eq_boundary, ineq_path])


def solve_ocp(
    problem: OCPProblem,
    solver: str = "pseudospectral",
) -> OCPSolution:
    """Solve an optimal control problem.

    Parameters
    ----------
    problem:
        OCP specification.
    solver:
        Solver choice. ``"pseudospectral"`` uses the LGL collocation
        method implemented in this module. ``"dymos"`` delegates the whole
        transcription to Dymos (:mod:`aether.optimal_control.dymos_backend`),
        which is the accurate option and, on the standard brachistochrone
        benchmark under the *same* SLSQP optimizer, four orders of magnitude
        closer to the analytic answer than the in-house path. ``"scvx"``
        delegates to :mod:`aether.optimal_control.scvx` (if available).

    Returns
    -------
    OCPSolution:
        Optimal trajectory and diagnostics.
    """
    if solver == "scvx":
        return _solve_scvx(problem)
    if solver == "dymos":
        from aether.optimal_control.dymos_backend import solve_dymos

        return solve_dymos(problem)
    if solver == "pseudospectral":
        return _solve_pseudospectral(problem)
    raise ValueError(f"unknown solver: {solver!r}. Choose 'pseudospectral', 'dymos' or 'scvx'")


def _solve_pseudospectral(problem: OCPProblem) -> OCPSolution:
    """Solve using pseudospectral collocation."""
    n_nodes = problem.N
    nx = problem.nx
    nu = problem.nu

    # Generate LGL nodes and differentiation matrix
    tau, _weights = lgl_nodes(n_nodes + 1)
    diff_matrix = differentiation_matrix(tau)

    # Build initial guess and bounds
    z0, lb, ub = _build_nlp_variables(problem)

    # Constraint counts, matching what the assemblers actually return: one
    # defect at every node, and path constraints at every node. This count has
    # now been short twice -- first at N - 1, then at N - and nothing catches
    # it, because only `n_eq` is used to split the stacked vector and `n_ineq`
    # is never read. It must track `_dynamics_constraints` exactly.
    n_dynamics = (n_nodes + 1) * nx
    n_boundary = (
        nx + int(np.sum(np.isfinite(problem.xf_target))) + len(problem.terminal_constraints)
    )
    n_eq = n_dynamics + n_boundary

    # SLSQP requires separate equality and inequality functions
    def eq_constraints(z: _FloatArray) -> _FloatArray:
        return _nlp_constraints(z, problem, tau, diff_matrix)[:n_eq]

    def ineq_constraints(z: _FloatArray) -> _FloatArray:
        return _nlp_constraints(z, problem, tau, diff_matrix)[n_eq:]

    # SLSQP's callback is given only the iterate, so the iteration is counted
    # here. The two numbers reported are the ones that distinguish converging
    # from thrashing: the objective, and how far the collocation defects still
    # are from zero.
    reported = {"n": 0}

    def _report(z: _FloatArray, *_ignored: object) -> None:
        reported["n"] += 1
        assert problem.progress is not None
        violation = float(np.max(np.abs(eq_constraints(z)))) if n_eq else 0.0
        problem.progress(reported["n"], float(_objective(z, problem)), violation)

    try:
        result = scipy.optimize.minimize(
            fun=lambda z: _objective(z, problem),
            x0=z0,
            method="SLSQP",
            bounds=list(zip(lb, ub, strict=True)),
            constraints=[
                {"type": "eq", "fun": eq_constraints},
                {"type": "ineq", "fun": ineq_constraints},
            ],
            options={"maxiter": problem.max_iter, "ftol": 1e-9, "disp": False, "iprint": 0},
            callback=_report if problem.progress is not None else None,
        )
    except Exception as exc:
        return OCPSolution(
            x=np.zeros((n_nodes + 1, nx)),
            u=np.zeros((n_nodes + 1, nu)),
            t=np.zeros(n_nodes + 1),
            tf=problem.tf_guess,
            cost=np.inf,
            success=False,
            message=f"NLP failed: {exc}",
            nfev=0,
            njev=0,
        )

    # Extract solution
    tf_opt = float(result.x[-1])
    x_opt = result.x[: (n_nodes + 1) * nx].reshape(n_nodes + 1, nx)
    controls = result.x[(n_nodes + 1) * nx : (n_nodes + 1) * nx + (n_nodes + 1) * nu]
    u_opt = controls.reshape(n_nodes + 1, nu)
    t_opt = map_to_physical(tau, problem.t0, tf_opt)

    return OCPSolution(
        x=x_opt,
        u=u_opt,
        t=t_opt,
        tf=tf_opt,
        cost=float(result.fun),
        success=result.success,
        message=result.message,
        nfev=int(result.nfev),
        njev=int(result.njev) if hasattr(result, "njev") else 0,
    )


def _solve_scvx(problem: OCPProblem) -> OCPSolution:
    """Solve using successive convexification (SCvx)."""
    try:
        from aether.optimal_control.scvx import SCvxConfig, solve_scvx
    except ImportError as exc:
        raise ImportError("SCvx solver requires aether.optimal_control.scvx") from exc

    # Map OCPProblem to SCvx format
    config = SCvxConfig(max_iterations=min(problem.max_iter, 80))

    # Build linearized trajectory reference
    n_nodes = problem.N
    nx = problem.nx
    nu = problem.nu
    n_steps = n_nodes
    horizon = float(problem.tf_guess - problem.t0)
    dt = horizon / max(n_steps, 1)
    times = np.asarray(problem.t0 + dt * np.arange(n_steps + 1), dtype=np.float64)

    # SCvx needs Jacobians and `OCPProblem` carries only the dynamics, so they
    # are taken by central differences. Declared rather than hidden: a
    # finite-difference Jacobian costs 2 (nx + nu) dynamics evaluations per
    # node per iteration, and its accuracy sets how large a trust region the
    # loop can accept.
    def jacobians(x: _FloatArray, u: _FloatArray) -> tuple[_FloatArray, _FloatArray]:
        step = 1.0e-6
        a_mat = np.empty((nx, nx))
        b_mat = np.empty((nx, nu))
        for k in range(nx):
            delta = np.zeros(nx)
            delta[k] = step * max(abs(float(x[k])), 1.0)
            a_mat[:, k] = (problem.dynamics(x + delta, u) - problem.dynamics(x - delta, u)) / (
                2.0 * delta[k]
            )
        for k in range(nu):
            delta = np.zeros(nu)
            delta[k] = step * max(abs(float(u[k])), 1.0)
            b_mat[:, k] = (problem.dynamics(x, u + delta) - problem.dynamics(x, u - delta)) / (
                2.0 * delta[k]
            )
        return a_mat, b_mat

    # A scalar authority bound is what SCvx takes; the OCP carries a box.
    if problem.u_bounds is None:
        control_limit = 1.0
    else:
        lower, upper = problem.u_bounds
        control_limit = float(np.max(np.abs(np.concatenate([lower, upper]))))

    # Free terminal components have no target for SCvx to drive to, so they
    # hold their initial value rather than being read as a target of NaN.
    target = np.where(np.isfinite(problem.xf_target), problem.xf_target, problem.x0)

    try:
        scvx_result = solve_scvx(
            dynamics=problem.dynamics,
            jacobians=jacobians,
            x0=problem.x0,
            x_target=target,
            n_steps=n_steps,
            dt=dt,
            control_limit=control_limit,
            n_controls=nu,
            config=config,
        )
    except Exception as exc:
        return OCPSolution(
            x=np.zeros((n_steps + 1, nx)),
            u=np.zeros((n_steps + 1, nu)),
            t=times,
            tf=problem.tf_guess,
            cost=np.inf,
            success=False,
            message=f"SCvx failed: {exc}",
            nfev=0,
            njev=0,
        )

    # SCvx returns one control per *interval*; the OCP contract is one per
    # node. The last is repeated rather than invented.
    controls = np.asarray(scvx_result.controls, dtype=np.float64)
    if controls.shape[0] == n_steps:
        controls = np.vstack([controls, controls[-1:]])

    return OCPSolution(
        x=np.asarray(scvx_result.states, dtype=np.float64),
        u=controls,
        t=times,
        tf=float(times[-1]),
        cost=float(scvx_result.virtual_norm),
        success=bool(scvx_result.converged),
        message=(
            "SCvx converged"
            if scvx_result.converged
            else f"SCvx stopped after {scvx_result.iterations} iterations"
        ),
        nfev=int(scvx_result.iterations),
        njev=int(scvx_result.iterations),
    )


def mesh_error(
    problem: OCPProblem,
    solution: OCPSolution,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> _FloatArray:
    r"""How far the collocated answer is from a trajectory that actually flies.

    Takes the returned control history, interpolates it, integrates the dynamics
    under it with an independent adaptive integrator, and returns the
    component-wise discrepancy at the terminal state. **This is the only thing
    standing between a converged solve and a wrong answer**, and it is not
    optional instrumentation.

    A direct transcription enforces the dynamics at the collocation nodes and
    nowhere else. Between nodes the polynomial is free, so the NLP is always
    solving a *relaxation* of the continuous problem -- and an optimiser handed
    a relaxation will find whatever the relaxation permits. That is not a flaw
    to be repaired; it is what discretisation means. What can be repaired is not
    knowing how large the gap is.

    The failure this exists to catch has a distinctive shape, and reporting
    ``success=True`` is part of it. On the maximum-range entry glide the solver
    converged cleanly and returned a bank history oscillating between 3 and 61
    degrees; the terminal state disagreed with an independent integration by
    **8.2 km at N=20 and 33.4 km at N=30**, and the reported arc sat *below* the
    best constant-bank trajectory meeting the same terminal conditions -- an
    optimum worse than a trivial feasible point, which is only possible if the
    reported trajectory is not one. Refining the mesh made it worse rather than
    better, which is the diagnostic: a convergent discretisation has this error
    falling with ``N``, and one that rises means the extra nodes are buying the
    optimiser freedom rather than fidelity.

    The mechanism there was ringing in the control. ``dJ/du`` is *exactly* zero
    for a maximum-range objective -- the control reaches the cost only through
    the dynamics -- so nothing in the NLP penalises a rough control history, and
    each added node is another degree of freedom to oscillate in. The structural
    remedy is a change of variables rather than a penalty: carry the control as
    a state and command its rate, which makes the history continuous by
    construction instead of merely discouraging roughness. This function does
    not fix any of that. It tells you whether you need to.

    Returns
    -------
    NDArray
        ``x_integrated(t_f) - x_collocated(t_f)``, in the problem's own state
        units. Compare against the scale of each state; there is no single
        threshold that is meaningful across a mixed-unit state vector, which is
        why this returns the vector rather than a norm.
    """
    times = np.asarray(solution.t, dtype=np.float64)
    controls = np.atleast_2d(np.asarray(solution.u, dtype=np.float64))
    if controls.shape[0] != times.size:
        controls = controls.T
    if times.size < 2:
        raise ValueError(
            f"a mesh error needs at least two nodes to integrate between, got {times.size}"
        )

    # Cubic where there are enough nodes to support it, linear otherwise. The
    # interpolant is part of the measurement: a control the integrator reads
    # differently from the transcription would show up as an error that is not
    # there.
    kind = "cubic" if times.size >= 4 else "linear"
    interpolants = [
        scipy.interpolate.interp1d(
            times,
            controls[:, j],
            kind=kind,
            bounds_error=False,
            fill_value=(controls[0, j], controls[-1, j]),
        )
        for j in range(controls.shape[1])
    ]

    def derivatives(t: float, x: _FloatArray) -> _FloatArray:
        control = np.array([float(f(t)) for f in interpolants])
        return np.atleast_1d(problem.dynamics(x, control))

    flown = scipy.integrate.solve_ivp(
        derivatives,
        (float(times[0]), float(times[-1])),
        np.asarray(problem.x0, dtype=np.float64),
        rtol=rtol,
        atol=atol,
    )
    if not flown.success:
        raise RuntimeError(
            f"the independent integration failed, so the mesh error is unknown "
            f"rather than small: {flown.message}"
        )
    terminal = np.asarray(solution.x, dtype=np.float64)[-1]
    return np.asarray(flown.y[:, -1] - terminal, dtype=np.float64)
