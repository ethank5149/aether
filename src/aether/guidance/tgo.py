"""Numerically stable time-to-go (Paper I, §4.3, Prop. 4 and Remark 6).

Under the constant-closing-acceleration extrapolation
:math:`R(t) = R_{\\mathrm{LOS}} - \\hat{V}_c t - \\tfrac{1}{2}\\hat{A}_c t^2`,
the time-to-go is the smallest positive root of :math:`R(t) = 0`. The
textbook root

.. math::

    t_{go} = \\frac{-\\hat{V}_c + \\sqrt{\\hat{V}_c^2
             + 2\\hat{A}_c R_{\\mathrm{LOS}}}}{\\hat{A}_c}

is algebraically correct and numerically unusable: as
:math:`\\hat{A}_c \\to 0` — the near-constant-closing-rate condition
that holds over most of terminal flight — it evaluates :math:`0/0` by
catastrophic cancellation. This module evaluates the conjugate form
(Paper I, Eq. 4.16),

.. math::

    t_{go} = \\frac{2 R_{\\mathrm{LOS}}}{\\hat{V}_c
             + \\sqrt{\\hat{V}_c^2 + 2\\hat{A}_c R_{\\mathrm{LOS}}}},

whose denominator is a sum of non-negative terms bounded below by
:math:`\\hat{V}_c`, together with the complete case analysis the paper's
:math:`\\hat{V}_c > 0` assumption leaves open:

========================  =========================================
condition                 result
========================  =========================================
:math:`R = 0`             :math:`t_{go} = 0` (``INTERCEPT_NOW``)
:math:`D \\ge 0`, closure  smallest positive root, cancellation-free
                          branch chosen by :math:`\\mathrm{sign}(V_c)`
:math:`D < 0, V_c > 0`    linear fallback :math:`R/V_c`
                          (``LINEAR_FALLBACK``, Remark 6 guard)
no positive root at all   :math:`+\\infty` (``NO_CLOSURE``)
========================  =========================================

For :math:`V_c < 0` with :math:`A_c > 0` (opening now, accelerating
closure) the conjugate form's *denominator* becomes the cancelling
difference, so that branch evaluates the equivalent
:math:`(-V_c + \\sqrt{D})/A_c`, which there is a sum of positives. Every
branch is therefore subtraction-free in its own domain.

All arithmetic preserves the input floating-point dtype — the V6
comparison exercises exactly this path in ``float32`` and ``float64``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["TgoResult", "TgoStatus", "time_to_go", "time_to_go_naive"]


class TgoStatus(IntEnum):
    """Disposition of the time-to-go computation, recorded per replicate.

    Paper I, Remark 6: a systematic pattern of ``LINEAR_FALLBACK`` or
    ``NO_CLOSURE`` across a Monte Carlo ensemble indicates an
    energy-management problem upstream, not a guidance problem — so the
    flag must survive batching rather than being collapsed to a scalar.
    """

    OK = 0
    #: Discriminant negative with positive closure: the constant-A_c
    #: extrapolation predicts no intercept; clamped to R/V_c.
    LINEAR_FALLBACK = 1
    #: No closure now and none predicted (V_c <= 0 with non-positive
    #: discriminant contribution); t_go is +inf.
    NO_CLOSURE = 2
    #: Zero range: intercept condition already satisfied.
    INTERCEPT_NOW = 3
    #: A non-finite input (NaN/inf) propagated; t_go is NaN.
    INVALID_INPUT = 4


@dataclass(frozen=True)
class TgoResult:
    """Vectorized time-to-go with per-element status.

    ``t_go`` preserves the dtype and shape of the broadcast inputs;
    ``status`` holds :class:`TgoStatus` values as an integer array of the
    same shape. ``feasible`` is true where the quadratic extrapolation
    itself yielded the root (status ``OK`` or ``INTERCEPT_NOW``).
    """

    t_go: NDArray[np.floating]
    status: NDArray[np.int64]
    discriminant: NDArray[np.floating]

    @property
    def feasible(self) -> NDArray[np.bool_]:
        return cast(
            NDArray[np.bool_],
            (self.status == TgoStatus.OK) | (self.status == TgoStatus.INTERCEPT_NOW),
        )

    def item(self) -> tuple[float, TgoStatus]:
        """Scalar convenience accessor for 0-d results."""
        if self.t_go.shape != ():
            raise ValueError(f"item() requires scalar inputs, result has shape {self.t_go.shape}")
        return float(self.t_go), TgoStatus(int(self.status))


def _prepare(
    r_los: ArrayLike, v_c: ArrayLike, a_c: ArrayLike
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    r = np.asarray(r_los)
    v = np.asarray(v_c)
    a = np.asarray(a_c)
    dtype = np.result_type(r, v, a, np.float32)  # floor at float32, never below
    if not np.issubdtype(dtype, np.floating):
        dtype = np.result_type(dtype, np.float64)
    r, v, a = (x.astype(dtype, copy=False) for x in (r, v, a))
    r, v, a = np.broadcast_arrays(r, v, a)
    if np.any(r[np.isfinite(r)] < 0.0):
        raise ValueError("r_los is a range (a norm) and must be non-negative")
    return r, v, a


def time_to_go(r_los: ArrayLike, v_c: ArrayLike, a_c: ArrayLike) -> TgoResult:
    """Stable time-to-go with the non-intercept guard (Paper I, Eq. 4.16).

    Parameters
    ----------
    r_los:
        Range to target :math:`R_{\\mathrm{LOS}} \\ge 0` (m).
    v_c:
        Estimated closing velocity :math:`\\hat{V}_c = -\\dot{R}` (m/s);
        positive when closing.
    a_c:
        Estimated closing acceleration :math:`\\hat{A}_c = \\dot{V}_c`
        (m/s²).

    Inputs broadcast; computation runs in the broadcast floating dtype
    (``float32`` stays ``float32``).
    """
    r, v, a = _prepare(r_los, v_c, a_c)
    dtype = r.dtype
    zero = dtype.type(0.0)
    two = dtype.type(2.0)

    disc = v * v + two * a * r
    with np.errstate(invalid="ignore", divide="ignore"):
        sqrt_d = np.sqrt(np.where(disc >= zero, disc, zero))

        # Branch values, each evaluated only where its mask holds; the
        # np.where(mask, expr, safe) pattern keeps every lane finite so no
        # spurious FP exceptions leak from inactive branches.
        closing = v > zero
        opening_accel = (~closing) & (a > zero) & (disc >= zero)

        denom = v + sqrt_d
        conj = np.where(denom > zero, two * r / np.where(denom > zero, denom, 1), np.inf)
        direct = np.where(opening_accel, (-v + sqrt_d) / np.where(opening_accel, a, 1), np.inf)
        linear = np.where(closing, r / np.where(closing, v, 1), np.inf)

    invalid = ~(np.isfinite(r) & np.isfinite(v) & np.isfinite(a))
    intercept_now = (r == zero) & ~invalid
    quad_ok = ~invalid & ~intercept_now & (disc >= zero) & (closing | opening_accel)
    fallback = ~invalid & ~intercept_now & (disc < zero) & closing
    no_closure = ~invalid & ~intercept_now & ~quad_ok & ~fallback

    t_go = np.empty_like(r)
    t_go[invalid] = np.nan
    t_go[intercept_now] = zero
    use_conj = quad_ok & closing
    use_direct = quad_ok & ~closing
    t_go[use_conj] = conj[use_conj]
    t_go[use_direct] = direct[use_direct]
    t_go[fallback] = linear[fallback]
    t_go[no_closure] = np.inf

    status = np.full(r.shape, int(TgoStatus.OK), dtype=np.int64)
    status[intercept_now] = int(TgoStatus.INTERCEPT_NOW)
    status[fallback] = int(TgoStatus.LINEAR_FALLBACK)
    status[no_closure] = int(TgoStatus.NO_CLOSURE)
    status[invalid] = int(TgoStatus.INVALID_INPUT)

    disc = np.ascontiguousarray(disc)
    for arr in (t_go, status, disc):
        arr.flags.writeable = False
    return TgoResult(t_go=t_go, status=status, discriminant=disc)


def time_to_go_naive(r_los: ArrayLike, v_c: ArrayLike, a_c: ArrayLike) -> NDArray[np.floating]:
    """The textbook root (Paper I, Eq. 4.15) — retained solely as the V6
    comparison baseline.

    Deliberately reproduces the catastrophic cancellation as
    :math:`\\hat{A}_c \\to 0`; do **not** use for guidance. NaN and inf
    propagate as IEEE arithmetic dictates.
    """
    r, v, a = _prepare(r_los, v_c, a_c)
    with np.errstate(invalid="ignore", divide="ignore"):
        return cast(NDArray[np.floating], (-v + np.sqrt(v * v + r.dtype.type(2.0) * a * r)) / a)
