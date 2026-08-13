"""Temporal integration strategies for the structural block (Paper I, §3.6).

Three strategies appear in verification task V3:

1. **Explicit** — any explicit RK method obeys the imaginary-axis limit
   :math:`\\Delta t \\le C_{\\mathrm{RK}}/\\omega_{\\max}` of Paper I,
   Prop. 2, with :math:`\\omega_{\\max} = \\mathcal{O}(N^4)`.
   :func:`explicit_dt_limit` computes the bound; the V3 runner drives
   SciPy's adaptive RK45 to *measure* the step it actually selects.
2. **Modal truncation** — :class:`ModalPropagator` advances a truncated
   modal basis *exactly* (rotation of each modal oscillator, zero-order
   hold on forcing), which is the strongest form of the mitigation: the
   only error is truncation itself.
3. **IMEX** — :class:`NewmarkIntegrator` treats the linear structural
   block implicitly with a factorization computed once and reused across
   every step and every Monte Carlo replicate (the columns of a batched
   state), which is what preserves the batching argument of Paper I, §5.
   Newmark with :math:`(\\beta, \\gamma) = (1/4, 1/2)` is
   unconditionally stable and non-dissipative, so it removes the CFL
   constraint without damping the modes it under-resolves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from aether.structures.modal import ModalBasis

__all__ = ["ModalPropagator", "NewmarkIntegrator", "explicit_dt_limit", "omega_max"]

_FloatArray = NDArray[np.float64]

#: Imaginary-axis stability-interval length of the fifth-order member of
#: the RKF45 pair (Paper I, §3.6 quotes C_RK ≈ 3.0).
C_RK_FEHLBERG5 = 3.0


def omega_max(k_hat: _FloatArray, m_hat: _FloatArray) -> float:
    """Largest undamped natural frequency of the reduced pencil.

    :math:`\\omega_{\\max} = \\sqrt{\\lambda_{\\max}(\\hat{\\mathbf{K}},
    \\hat{\\mathbf{M}})}`, computed by the QZ algorithm on the full
    pencil — no shortcut via a norm bound, since V3's scaling law is fit
    against this exact quantity.
    """
    lam = scipy.linalg.eigvals(k_hat, m_hat)
    lam_max = float(np.max(lam.real))
    if lam_max <= 0.0:
        raise ValueError("reduced pencil has no positive eigenvalue; nothing to bound")
    return float(np.sqrt(lam_max))


def explicit_dt_limit(omega_maximum: float, c_rk: float = C_RK_FEHLBERG5) -> float:
    """Explicit stability limit :math:`\\Delta t \\le C_{\\mathrm{RK}}/\\omega_{\\max}`
    (Paper I, Eq. 3.24)."""
    if not (np.isfinite(omega_maximum) and omega_maximum > 0.0):
        raise ValueError(f"omega_maximum must be finite and > 0, got {omega_maximum}")
    if not (np.isfinite(c_rk) and c_rk > 0.0):
        raise ValueError(f"c_rk must be finite and > 0, got {c_rk}")
    return c_rk / omega_maximum


def _as_state(x: NDArray[np.floating], n: int, name: str) -> _FloatArray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.shape[0] != n or arr.ndim not in (1, 2):
        raise ValueError(
            f"{name} must have shape ({n},) or ({n}, n_batch), got {arr.shape}"
        )
    return arr


class NewmarkIntegrator:
    """Implicit Newmark-:math:`\\beta` scheme with a cached factorization.

    Advances :math:`\\hat{\\mathbf{M}}\\ddot{\\mathbf{u}} +
    \\hat{\\mathbf{C}}\\dot{\\mathbf{u}} + \\hat{\\mathbf{K}}\\mathbf{u}
    = \\mathbf{f}(t)` at fixed :math:`\\Delta t`. The effective operator
    :math:`\\mathbf{S} = \\hat{\\mathbf{M}} + \\gamma\\Delta t\\,
    \\hat{\\mathbf{C}} + \\beta\\Delta t^2 \\hat{\\mathbf{K}}` is LU-
    factorized once at construction; every subsequent step — and every
    replicate column of a batched state — costs one pair of triangular
    solves, which is the IMEX cost model of Paper I, §3.6.

    The default :math:`(\\beta, \\gamma) = (1/4, 1/2)` (average constant
    acceleration) is unconditionally stable and introduces no algorithmic
    dissipation. :math:`\\gamma > 1/2` adds high-frequency dissipation at
    the cost of first-order error in the damped modes; it is permitted
    but not silently chosen.

    Parameters
    ----------
    k_hat, m_hat:
        Reduced stiffness and SPD reduced mass.
    dt:
        Fixed step size (s).
    damping:
        Optional reduced damping matrix :math:`\\hat{\\mathbf{C}}`.
    beta, gamma:
        Newmark parameters; unconditional stability requires
        :math:`\\gamma \\ge 1/2` and :math:`\\beta \\ge \\gamma/2`,
        enforced here because this class exists specifically to be the
        unconditionally stable branch of V3.
    """

    def __init__(
        self,
        k_hat: _FloatArray,
        m_hat: _FloatArray,
        dt: float,
        damping: _FloatArray | None = None,
        beta: float = 0.25,
        gamma: float = 0.5,
    ) -> None:
        k = np.asarray(k_hat, dtype=np.float64)
        m = np.asarray(m_hat, dtype=np.float64)
        if k.ndim != 2 or k.shape[0] != k.shape[1]:
            raise ValueError(f"k_hat must be square, got shape {k.shape}")
        if m.shape != k.shape:
            raise ValueError(f"m_hat shape {m.shape} does not match k_hat shape {k.shape}")
        if not (np.isfinite(dt) and dt > 0.0):
            raise ValueError(f"dt must be finite and > 0, got {dt}")
        if not gamma >= 0.5:
            raise ValueError(f"unconditional stability requires gamma >= 1/2, got {gamma}")
        if not beta >= gamma / 2.0:
            raise ValueError(
                f"unconditional stability requires beta >= gamma/2 = {gamma / 2.0}, got {beta}"
            )
        c = (
            np.zeros_like(k)
            if damping is None
            else np.asarray(damping, dtype=np.float64)
        )
        if c.shape != k.shape:
            raise ValueError(f"damping shape {c.shape} does not match k_hat shape {k.shape}")

        self._n = int(k.shape[0])
        self._k = k
        self._c = c
        self._dt = float(dt)
        self._beta = float(beta)
        self._gamma = float(gamma)

        # M is SPD by construction of the null-space reduction.
        self._m_cho = scipy.linalg.cho_factor(m, lower=True, check_finite=True)
        s = m + gamma * dt * c + beta * dt * dt * k
        self._s_lu = scipy.linalg.lu_factor(s, check_finite=True)

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def dim(self) -> int:
        return self._n

    def initial_acceleration(
        self, u0: _FloatArray, v0: _FloatArray, f0: _FloatArray | None = None
    ) -> _FloatArray:
        """Consistent :math:`\\ddot{\\mathbf{u}}_0 = \\hat{\\mathbf{M}}^{-1}
        (\\mathbf{f}_0 - \\hat{\\mathbf{C}}\\mathbf{v}_0 -
        \\hat{\\mathbf{K}}\\mathbf{u}_0)`."""
        u = _as_state(u0, self._n, "u0")
        v = _as_state(v0, self._n, "v0")
        rhs = -(self._k @ u) - self._c @ v
        if f0 is not None:
            rhs = rhs + _as_state(f0, self._n, "f0")
        return cast(_FloatArray, scipy.linalg.cho_solve(self._m_cho, rhs))

    def step(
        self,
        u: _FloatArray,
        v: _FloatArray,
        a: _FloatArray,
        f_next: _FloatArray | None = None,
    ) -> tuple[_FloatArray, _FloatArray, _FloatArray]:
        """One Newmark step; state arrays may carry a trailing batch axis.

        Parameters
        ----------
        u, v, a:
            Displacement, velocity, acceleration at time :math:`t_k`.
        f_next:
            Forcing evaluated at :math:`t_{k+1}` (zero if omitted).

        Returns
        -------
        tuple
            ``(u_next, v_next, a_next)``.
        """
        dt, beta, gamma = self._dt, self._beta, self._gamma
        u = _as_state(u, self._n, "u")
        v = _as_state(v, self._n, "v")
        a = _as_state(a, self._n, "a")

        u_pred = u + dt * v + (0.5 - beta) * dt * dt * a
        v_pred = v + (1.0 - gamma) * dt * a
        rhs = -(self._k @ u_pred) - self._c @ v_pred
        if f_next is not None:
            rhs = rhs + _as_state(f_next, self._n, "f_next")
        a_next = scipy.linalg.lu_solve(self._s_lu, rhs)
        u_next = u_pred + beta * dt * dt * a_next
        v_next = v_pred + gamma * dt * a_next
        return u_next, v_next, a_next


@dataclass(frozen=True)
class ModalPropagator:
    """Exact fixed-step propagator for a truncated modal basis.

    Each retained elastic mode is an undamped oscillator
    :math:`\\ddot{q}_i + \\omega_i^2 q_i = f_i`; over one step with the
    forcing held at its start-of-step value (zero-order hold) the update
    is the exact rotation

    .. math::

        q^+ = q\\cos\\omega\\Delta t + \\frac{\\dot q}{\\omega}
              \\sin\\omega\\Delta t + \\frac{1 - \\cos\\omega\\Delta t}
              {\\omega^2} f,

    with the matching velocity row, and rigid modes
    (:math:`\\omega = 0`) take the polynomial limit. There is no
    stability limit and no discretization error beyond the ZOH — the
    error of the strategy is confined to the truncation itself, which is
    what V3 isolates.
    """

    basis: ModalBasis
    dt: float

    def __post_init__(self) -> None:
        if not (np.isfinite(self.dt) and self.dt > 0.0):
            raise ValueError(f"dt must be finite and > 0, got {self.dt}")

    def step(
        self,
        q: _FloatArray,
        q_dot: _FloatArray,
        f_modal: _FloatArray | None = None,
    ) -> tuple[_FloatArray, _FloatArray]:
        """Advance modal coordinates one step; trailing batch axes allowed."""
        omega = self.basis.omega
        n = omega.size
        q = np.asarray(q, dtype=np.float64)
        q_dot = np.asarray(q_dot, dtype=np.float64)
        if q.shape[0] != n or q_dot.shape != q.shape:
            raise ValueError(
                f"q and q_dot must share shape ({n},) or ({n}, n_batch), "
                f"got {q.shape} and {q_dot.shape}"
            )
        f = np.zeros_like(q) if f_modal is None else np.asarray(f_modal, dtype=np.float64)
        if f.shape != q.shape:
            raise ValueError(f"f_modal shape {f.shape} does not match q shape {q.shape}")

        dt = self.dt
        elastic = omega > 0.0
        w = omega[elastic]
        shape_tail = (1,) * (q.ndim - 1)
        w_c = w.reshape(w.shape + shape_tail)
        cos_wt = np.cos(w_c * dt)
        sin_wt = np.sin(w_c * dt)

        q_next = np.empty_like(q)
        v_next = np.empty_like(q_dot)

        q_next[elastic] = (
            q[elastic] * cos_wt
            + q_dot[elastic] * sin_wt / w_c
            + f[elastic] * (1.0 - cos_wt) / w_c**2
        )
        v_next[elastic] = (
            -q[elastic] * w_c * sin_wt + q_dot[elastic] * cos_wt + f[elastic] * sin_wt / w_c
        )

        rigid = ~elastic
        q_next[rigid] = q[rigid] + dt * q_dot[rigid] + 0.5 * dt * dt * f[rigid]
        v_next[rigid] = q_dot[rigid] + dt * f[rigid]
        return q_next, v_next
