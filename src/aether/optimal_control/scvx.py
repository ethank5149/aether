"""Successive convexification with exact :math:`\\ell_1` penalty (Paper II, §6.1).

At iteration :math:`k` the nonlinear dynamics are linearized about the
current reference and discretized (Eq. 6.1),

.. math::

    \\mathbf{x}_{i+1} = \\mathbf{A}_i\\mathbf{x}_i + \\mathbf{B}_i
    \\mathbf{u}_i + \\mathbf{z}_i + \\bm{\\nu}_i,

with **free** virtual controls :math:`\\bm{\\nu}_i \\in \\mathbb{R}^{n_x}`.

**Virtual controls are unconstrained, not sign-constrained** (Paper II,
Remark 1). Constraining them to be non-negative — by false analogy with
inequality slacks — makes the subproblem infeasible in exactly the cases
the virtual control exists to rescue, since linearization error has no
preferred sign along any state axis, and it invalidates the exact-penalty
argument, which needs the penalty to be exact on a neighbourhood of zero
*in all directions*. This implementation splits each
:math:`\\bm{\\nu}_i` into non-negative parts only as an LP encoding of
:math:`|\\cdot|`; the variable itself remains free.

**The penalty is exact.** The :math:`\\ell_1` form (Eq. 6.2)

.. math::

    J_\\nu = w_\\nu \\sum_i \\|\\bm{\\nu}_i\\|_1

drives :math:`\\bm{\\nu}_i` to **exactly** zero above a finite threshold
related to the dual variables of the original problem, where a quadratic
penalty only approaches zero as :math:`w_\\nu \\to \\infty` while
degrading conditioning. :func:`solve_subproblem` and
:func:`solve_subproblem_l2` provide both so the difference can be
measured, which is verification task II-V7.

The trust region is accepted or rejected on the ratio of actual to
predicted cost reduction (Eq. 6.3), with a non-positive predicted
reduction terminating the iteration as subproblem stagnation rather than
being divided through.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import scipy.optimize
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "SCvxConfig",
    "SCvxResult",
    "SubproblemSolution",
    "linearize_trajectory",
    "solve_scvx",
    "solve_subproblem",
    "solve_subproblem_l2",
]

_FloatArray = NDArray[np.float64]
#: Continuous dynamics f(x, u) -> xdot
Dynamics = Callable[[_FloatArray, _FloatArray], _FloatArray]


@dataclass(frozen=True)
class SCvxConfig:
    """Trust-region and penalty parameters (Paper II, §6.1)."""

    penalty_weight: float = 1.0e3
    """:math:`w_\\nu` of Eq. (6.2)."""
    trust_radius: float = 2.0
    """Initial trust-region radius on the state, in state units."""
    trust_min: float = 1.0e-6
    trust_max: float = 1.0e3
    rho_reject: float = 0.0
    """:math:`\\rho_0`: below this the step is rejected and the radius contracts."""
    rho_contract: float = 0.25
    """:math:`\\rho_1`: accept with contraction below this."""
    rho_expand: float = 0.7
    """:math:`\\rho_2`: accept with expansion at or above this."""
    contract_factor: float = 0.5
    expand_factor: float = 2.0
    max_iterations: int = 80
    virtual_control_tol: float = 1.0e-9
    """Convergence: :math:`\\|\\bm{\\nu}\\|_1` at or below this counts as zero."""
    step_tol: float = 1.0e-6
    """Convergence: max state change relative to the state scale."""
    cost_tol: float = 1.0e-8
    """Convergence: relative change in the true cost between accepted steps."""

    def __post_init__(self) -> None:
        if not (np.isfinite(self.penalty_weight) and self.penalty_weight > 0.0):
            raise ValueError("penalty_weight must be finite and > 0")
        if not 0.0 < self.trust_min <= self.trust_radius <= self.trust_max:
            raise ValueError("need 0 < trust_min <= trust_radius <= trust_max")
        if not self.rho_reject <= self.rho_contract <= self.rho_expand:
            raise ValueError("need rho_reject <= rho_contract <= rho_expand")
        if not 0.0 < self.contract_factor < 1.0 < self.expand_factor:
            raise ValueError("need 0 < contract_factor < 1 < expand_factor")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")


@dataclass(frozen=True)
class SubproblemSolution:
    """One convex subproblem solve."""

    states: _FloatArray = field(repr=False)
    controls: _FloatArray = field(repr=False)
    virtual_controls: _FloatArray = field(repr=False)
    model_cost: float
    control_cost: float
    virtual_norm: float
    """:math:`\\sum_i\\|\\bm{\\nu}_i\\|_1`."""
    success: bool


@dataclass(frozen=True)
class SCvxResult:
    """Outcome of the SCvx loop."""

    states: _FloatArray = field(repr=False)
    controls: _FloatArray = field(repr=False)
    iterations: int
    virtual_norm: float
    trust_radius: float
    converged: bool
    history: list[dict[str, float]] = field(default_factory=list, repr=False)


def linearize_trajectory(
    dynamics: Dynamics,
    jacobians: Callable[[_FloatArray, _FloatArray], tuple[_FloatArray, _FloatArray]],
    states: _FloatArray,
    controls: _FloatArray,
    dt: float,
) -> tuple[_FloatArray, _FloatArray, _FloatArray]:
    """First-order discrete linearization about a reference trajectory.

    Returns :math:`(\\mathbf{A}_i, \\mathbf{B}_i, \\mathbf{z}_i)` of
    Eq. (6.1) using an Euler discretization of the Jacobians, with
    :math:`\\mathbf{z}_i` the residual that makes the linearization
    exact at the reference point.
    """
    n_steps = controls.shape[0]
    n_x = states.shape[1]
    n_u = controls.shape[1]
    a_mats = np.empty((n_steps, n_x, n_x))
    b_mats = np.empty((n_steps, n_x, n_u))
    z_vecs = np.empty((n_steps, n_x))
    eye = np.eye(n_x)
    for i in range(n_steps):
        x_i, u_i = states[i], controls[i]
        j_x, j_u = jacobians(x_i, u_i)
        a_mats[i] = eye + dt * j_x
        b_mats[i] = dt * j_u
        x_next = x_i + dt * dynamics(x_i, u_i)
        z_vecs[i] = x_next - a_mats[i] @ x_i - b_mats[i] @ u_i
    return a_mats, b_mats, z_vecs


def _build_lp(
    a_mats: _FloatArray,
    b_mats: _FloatArray,
    z_vecs: _FloatArray,
    x0: _FloatArray,
    x_target: _FloatArray,
    reference_states: _FloatArray,
    control_limit: float,
    penalty_weight: float,
    trust_radius: float,
) -> tuple[
    _FloatArray,
    _FloatArray,
    _FloatArray,
    list[tuple[float | None, float | None]],
    int,
    int,
    _FloatArray,
    _FloatArray,
]:
    """Assemble the penalized subproblem as a linear program.

    Decision vector: states, controls, the split :math:`|u|` epigraph
    variables, and the split :math:`\\nu^+, \\nu^- \\ge 0` encoding
    :math:`\\bm{\\nu} = \\bm{\\nu}^+ - \\bm{\\nu}^-` — free in sign, as
    Remark 1 requires, with :math:`\\|\\bm{\\nu}\\|_1 = \\sum(\\nu^+ +
    \\nu^-)` exactly at the optimum.
    """
    n_steps, n_x, n_u = b_mats.shape
    n_nodes = n_steps + 1
    n_state_vars = n_nodes * n_x
    n_ctrl_vars = n_steps * n_u
    n_abs_vars = n_steps * n_u
    n_nu_vars = n_steps * n_x

    offs_x = 0
    offs_u = n_state_vars
    offs_t = offs_u + n_ctrl_vars
    offs_nu_p = offs_t + n_abs_vars
    offs_nu_m = offs_nu_p + n_nu_vars
    n_vars = offs_nu_m + n_nu_vars

    cost = np.zeros(n_vars)
    cost[offs_t : offs_t + n_abs_vars] = 1.0  # minimize control effort ||u||_1
    cost[offs_nu_p:] = penalty_weight  # exact l1 penalty on the virtual control

    rows: list[_FloatArray] = []
    rhs: list[float] = []
    for i in range(n_steps):
        for k in range(n_x):
            row = np.zeros(n_vars)
            row[offs_x + (i + 1) * n_x + k] = 1.0
            row[offs_x + i * n_x : offs_x + (i + 1) * n_x] -= a_mats[i, k, :]
            row[offs_u + i * n_u : offs_u + (i + 1) * n_u] -= b_mats[i, k, :]
            row[offs_nu_p + i * n_x + k] = -1.0
            row[offs_nu_m + i * n_x + k] = 1.0
            rows.append(row)
            rhs.append(float(z_vecs[i, k]))
    for k in range(n_x):  # initial condition
        row = np.zeros(n_vars)
        row[offs_x + k] = 1.0
        rows.append(row)
        rhs.append(float(x0[k]))
    for k in range(n_x):  # terminal condition
        row = np.zeros(n_vars)
        row[offs_x + n_steps * n_x + k] = 1.0
        rows.append(row)
        rhs.append(float(x_target[k]))

    a_eq = np.asarray(rows)
    b_eq = np.asarray(rhs)

    # |u| epigraph:  -t <= u <= t
    ub_rows: list[_FloatArray] = []
    ub_rhs: list[float] = []
    for i in range(n_steps):
        for k in range(n_u):
            for sign in (1.0, -1.0):
                row = np.zeros(n_vars)
                row[offs_u + i * n_u + k] = sign
                row[offs_t + i * n_u + k] = -1.0
                ub_rows.append(row)
                ub_rhs.append(0.0)
    a_ub = np.asarray(ub_rows)
    b_ub = np.asarray(ub_rhs)

    bounds: list[tuple[float | None, float | None]] = []
    for i in range(n_nodes):
        for k in range(n_x):
            centre = float(reference_states[i, k])
            bounds.append((centre - trust_radius, centre + trust_radius))
    bounds += [(-control_limit, control_limit)] * n_ctrl_vars
    bounds += [(0.0, control_limit)] * n_abs_vars
    bounds += [(0.0, None)] * (2 * n_nu_vars)
    return cost, a_eq, b_eq, bounds, offs_u, offs_nu_p, a_ub, b_ub


def solve_subproblem(
    a_mats: _FloatArray,
    b_mats: _FloatArray,
    z_vecs: _FloatArray,
    x0: ArrayLike,
    x_target: ArrayLike,
    reference_states: _FloatArray,
    control_limit: float,
    penalty_weight: float,
    trust_radius: float,
) -> SubproblemSolution:
    """Solve the :math:`\\ell_1`-penalized convex subproblem as an LP."""
    x_init = np.asarray(x0, dtype=np.float64)
    x_end = np.asarray(x_target, dtype=np.float64)
    n_steps, n_x, n_u = b_mats.shape
    cost, a_eq, b_eq, bounds, offs_u, offs_nu_p, a_ub, b_ub = _build_lp(
        a_mats,
        b_mats,
        z_vecs,
        x_init,
        x_end,
        reference_states,
        control_limit,
        penalty_weight,
        trust_radius,
    )
    res = scipy.optimize.linprog(
        cost, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )
    if not res.success:
        return SubproblemSolution(
            states=reference_states.copy(),
            controls=np.zeros((n_steps, n_u)),
            virtual_controls=np.zeros((n_steps, n_x)),
            model_cost=np.inf,
            control_cost=np.inf,
            virtual_norm=np.inf,
            success=False,
        )
    sol = np.asarray(res.x)
    states = sol[: (n_steps + 1) * n_x].reshape(n_steps + 1, n_x)
    controls = sol[offs_u : offs_u + n_steps * n_u].reshape(n_steps, n_u)
    nu_p = sol[offs_nu_p : offs_nu_p + n_steps * n_x]
    nu_m = sol[offs_nu_p + n_steps * n_x : offs_nu_p + 2 * n_steps * n_x]
    nu = (nu_p - nu_m).reshape(n_steps, n_x)
    control_cost = float(np.sum(np.abs(controls)))
    virtual_norm = float(np.sum(np.abs(nu)))
    return SubproblemSolution(
        states=states,
        controls=controls,
        virtual_controls=nu,
        model_cost=float(res.fun),
        control_cost=control_cost,
        virtual_norm=virtual_norm,
        success=True,
    )


def solve_subproblem_l2(
    a_mats: _FloatArray,
    b_mats: _FloatArray,
    z_vecs: _FloatArray,
    x0: ArrayLike,
    x_target: ArrayLike,
    reference_states: _FloatArray,
    control_limit: float,
    penalty_weight: float,
    trust_radius: float,
) -> SubproblemSolution:
    """The same subproblem under a *quadratic* virtual-control penalty.

    Provided solely as the II-V7 comparison: a quadratic penalty drives
    :math:`\\bm{\\nu}` to zero only as :math:`w_\\nu \\to \\infty`, so
    its residual norm should scale as :math:`1/w_\\nu` where the
    :math:`\\ell_1` form reaches exactly zero at finite weight.
    """
    x_init = np.asarray(x0, dtype=np.float64)
    x_end = np.asarray(x_target, dtype=np.float64)
    n_steps, n_x, n_u = b_mats.shape
    n_nodes = n_steps + 1

    def unpack(v: _FloatArray) -> tuple[_FloatArray, _FloatArray]:
        return (
            v[: n_nodes * n_x].reshape(n_nodes, n_x),
            v[n_nodes * n_x :].reshape(n_steps, n_u),
        )

    def defects(v: _FloatArray) -> _FloatArray:
        x, u = unpack(v)
        out = np.empty((n_steps, n_x))
        for i in range(n_steps):
            out[i] = x[i + 1] - (a_mats[i] @ x[i] + b_mats[i] @ u[i] + z_vecs[i])
        return out

    def objective(v: _FloatArray) -> float:
        _, u = unpack(v)
        nu = defects(v)
        return float(np.sum(u * u) + penalty_weight * np.sum(nu * nu))

    constraints = [
        {"type": "eq", "fun": lambda v: unpack(v)[0][0] - x_init},
        {"type": "eq", "fun": lambda v: unpack(v)[0][-1] - x_end},
    ]
    bounds = [
        (float(reference_states[i, k] - trust_radius), float(reference_states[i, k] + trust_radius))
        for i in range(n_nodes)
        for k in range(n_x)
    ] + [(-control_limit, control_limit)] * (n_steps * n_u)

    v0 = np.concatenate([reference_states.reshape(-1), np.zeros(n_steps * n_u)])
    res = scipy.optimize.minimize(
        objective,
        v0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 400, "ftol": 1e-14},
    )
    x, u = unpack(np.asarray(res.x))
    nu = defects(np.asarray(res.x))
    return SubproblemSolution(
        states=x,
        controls=u,
        virtual_controls=nu,
        model_cost=float(res.fun),
        control_cost=float(np.sum(np.abs(u))),
        virtual_norm=float(np.sum(np.abs(nu))),
        success=bool(res.success),
    )


def solve_scvx(
    dynamics: Dynamics,
    jacobians: Callable[[_FloatArray, _FloatArray], tuple[_FloatArray, _FloatArray]],
    x0: ArrayLike,
    x_target: ArrayLike,
    n_steps: int,
    dt: float,
    control_limit: float,
    n_controls: int,
    config: SCvxConfig | None = None,
) -> SCvxResult:
    """Run the SCvx loop to convergence.

    The trust region is updated on :math:`\\rho_k` of Eq. (6.3), and a
    non-positive predicted reduction terminates the iteration as
    subproblem stagnation rather than being divided through — the
    well-posedness condition the paper states alongside the ratio.
    """
    cfg = config or SCvxConfig()
    x_init = np.asarray(x0, dtype=np.float64)
    x_end = np.asarray(x_target, dtype=np.float64)
    if x_end.shape != x_init.shape:
        raise ValueError("x0 and x_target must have the same dimension")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if n_controls < 1:
        raise ValueError(f"n_controls must be >= 1, got {n_controls}")
    if not (np.isfinite(control_limit) and control_limit > 0.0):
        raise ValueError(f"control_limit must be finite and > 0, got {control_limit}")
    if not (np.isfinite(dt) and dt > 0.0):
        raise ValueError(f"dt must be finite and > 0, got {dt}")

    states = np.linspace(x_init, x_end, n_steps + 1)
    controls = np.zeros((n_steps, n_controls))

    def nonlinear_defect(x: _FloatArray, u: _FloatArray) -> float:
        total = 0.0
        for i in range(n_steps):
            predicted = x[i] + dt * dynamics(x[i], u[i])
            total += float(np.sum(np.abs(x[i + 1] - predicted)))
        return total

    def true_cost(x: _FloatArray, u: _FloatArray) -> float:
        return float(np.sum(np.abs(u))) + cfg.penalty_weight * nonlinear_defect(x, u)

    radius = cfg.trust_radius
    history: list[dict[str, float]] = []
    converged = False
    iterations = 0
    virtual_norm = np.inf
    previous_cost = np.inf

    for iteration in range(1, cfg.max_iterations + 1):
        iterations = iteration
        a_mats, b_mats, z_vecs = linearize_trajectory(dynamics, jacobians, states, controls, dt)
        sub = solve_subproblem(
            a_mats,
            b_mats,
            z_vecs,
            x_init,
            x_end,
            states,
            control_limit,
            cfg.penalty_weight,
            radius,
        )
        if not sub.success:
            radius *= cfg.contract_factor
            history.append({"iteration": iteration, "status": -1.0, "radius": radius})
            if radius < cfg.trust_min:
                break
            continue

        j_current = true_cost(states, controls)
        j_candidate = true_cost(sub.states, sub.controls)
        model_current = float(np.sum(np.abs(controls))) + cfg.penalty_weight * float(
            nonlinear_defect(states, controls)
        )
        predicted_reduction = model_current - sub.model_cost
        if predicted_reduction <= 0.0:
            history.append(
                {
                    "iteration": iteration,
                    "rho": float("nan"),
                    "radius": radius,
                    "virtual_norm": sub.virtual_norm,
                    "stagnated": 1.0,
                }
            )
            virtual_norm = sub.virtual_norm
            converged = sub.virtual_norm <= cfg.virtual_control_tol
            break

        rho = (j_current - j_candidate) / predicted_reduction
        state_scale = max(1.0, float(np.max(np.abs(states))))
        step = float(np.max(np.abs(sub.states - states))) / state_scale
        accepted = rho >= cfg.rho_reject
        if accepted:
            states, controls = sub.states, sub.controls
            virtual_norm = sub.virtual_norm
        if rho < cfg.rho_contract:
            radius = max(cfg.trust_min, radius * cfg.contract_factor)
        elif rho >= cfg.rho_expand:
            radius = min(cfg.trust_max, radius * cfg.expand_factor)
        history.append(
            {
                "iteration": iteration,
                "rho": float(rho),
                "radius": float(radius),
                "virtual_norm": float(sub.virtual_norm),
                "step": step,
                "accepted": float(accepted),
            }
        )
        # Converged when the virtual controls have vanished — the exact
        # penalty makes that an equality, not a limit — and the iteration
        # is no longer buying anything, whether measured by the step or by
        # the true cost. The cost test is the operative one: the LP returns
        # a vertex, so the state can keep jittering between adjacent
        # vertices at fixed cost long after the solution has settled.
        if accepted and sub.virtual_norm <= cfg.virtual_control_tol:
            cost_change = abs(previous_cost - j_candidate) / max(1.0, abs(j_candidate))
            if step < cfg.step_tol or cost_change < cfg.cost_tol:
                converged = True
                previous_cost = j_candidate
                break
            previous_cost = j_candidate

    return SCvxResult(
        states=states,
        controls=controls,
        iterations=iterations,
        virtual_norm=float(virtual_norm),
        trust_radius=float(radius),
        converged=converged,
        history=history,
    )
