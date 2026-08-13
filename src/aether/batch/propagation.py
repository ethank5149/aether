"""Common-outer-grid batched integration (Paper I, §5.2).

One classical RK4 sweep advances the whole replicate ensemble on a
shared fixed time grid — the mitigation the paper's Remark 9 prescribes
for warp divergence: replicates never adapt their steps independently,
so the batch stays coherent on both CPU SIMD and GPU warps. The stage
values are materialized as the rank-3 tensor ``(replicate, state,
stage)`` that §5.2 identifies as the native shape of the computation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from aether.batch.backend import Backend, get_array_module

__all__ = ["rk4_batch"]

#: RHS signature: (t, states (n_rep, n_state), xp) -> derivatives, same shape.
BatchRHS = Callable[[float, Any, Any], Any]


def rk4_batch(
    rhs: BatchRHS,
    initial_states: Any,
    t_start: float,
    t_end: float,
    n_steps: int,
    backend: Backend = "numpy",
    callback: Callable[[float, Any], None] | None = None,
) -> Any:
    """Advance a replicate ensemble with fixed-step classical RK4.

    Parameters
    ----------
    rhs:
        Batched right-hand side ``f(t, y, xp) -> dy/dt`` operating on the
        full ``(n_rep, n_state)`` block with the backend array module
        ``xp`` — one function evaluation advances every replicate.
    initial_states:
        Ensemble initial conditions, shape ``(n_rep, n_state)``.
    t_start, t_end:
        Integration window, ``t_end > t_start``.
    n_steps:
        Number of fixed steps on the common outer grid.
    backend:
        ``"numpy"`` or ``"cupy"``.
    callback:
        Optional observer called as ``callback(t, y)`` after every step
        (used for event detection by the entry model).

    Returns
    -------
    array
        Final ensemble states on the requested backend.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if not t_end > t_start:
        raise ValueError(f"need t_end > t_start, got ({t_start}, {t_end})")
    xp = get_array_module(backend)
    y = xp.array(initial_states, dtype=xp.float64)
    if y.ndim != 2:
        raise ValueError(
            f"initial_states must have shape (n_rep, n_state), got {y.shape}"
        )
    dt = (t_end - t_start) / n_steps
    # rank-3 stage tensor: (replicate, state, stage)
    stages = xp.empty((*y.shape, 4), dtype=xp.float64)

    t = t_start
    for _ in range(n_steps):
        stages[:, :, 0] = rhs(t, y, xp)
        stages[:, :, 1] = rhs(t + 0.5 * dt, y + 0.5 * dt * stages[:, :, 0], xp)
        stages[:, :, 2] = rhs(t + 0.5 * dt, y + 0.5 * dt * stages[:, :, 1], xp)
        stages[:, :, 3] = rhs(t + dt, y + dt * stages[:, :, 2], xp)
        y = y + (dt / 6.0) * (
            stages[:, :, 0] + 2.0 * stages[:, :, 1] + 2.0 * stages[:, :, 2] + stages[:, :, 3]
        )
        t += dt
        if callback is not None:
            callback(t, y)
    return y
