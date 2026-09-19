r"""Rigorous Jacobians, and the mean-value step they buy.

The naive step of :mod:`aether.certification.tube` evaluates :math:`f` over a
whole box, so every occurrence of a variable is treated as independent. The
**mean-value form** keeps the correlation instead. For any :math:`x \in X` with
centre :math:`c`,

.. math::

    f(x) \in f(c) + J(X)\,(x - c),

so the step becomes

.. math::

    X' \subseteq c + h f(c) \;+\; \bigl(I + h J(Y)\bigr)(X - c)
        \;+\; \tfrac{h^2}{2}\,\bigl(J f\bigr)(Y),

in which the linear part acts on the *deviation* coherently and the centre is
carried as a point. Measured on a six-state entry glide over ten seconds this
is several times tighter than re-boxing the field each step -- 6.2x on speed,
3.9x on flight path angle -- and the growth rate falls correspondingly.

**The final term is not optional and dropping it is not conservative.** Without
it the step is a first-order Taylor expansion missing its remainder, so it
*under*-estimates: an earlier version omitted it and produced boxes 0.65-1.00x
the width of the naive step which were wrong 51 times in 18 000 containment
checks. A certified bound that is too narrow is worse than no bound, because
nothing downstream can detect it -- which is why soundness is tested against
sampled trajectories rather than against a previous implementation.

Getting :math:`J` rigorously
----------------------------

Not by finite differences: a difference quotient of enclosures is an enclosure
of a difference quotient, which is not a derivative and carries a truncation
error nothing bounds. Arb's truncated power series give the real thing.
Evaluate the field on :math:`x_j + \epsilon` -- a series whose constant term is
the ball for component :math:`j` and whose linear coefficient is one -- and the
linear coefficient of the result *is* :math:`\partial f/\partial x_j`, enclosed
to the same standard as everything else. One evaluation per state component.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from aether.certification.rigorous import (
    MathOps,
    VectorField,
    enclose_field,
)
from aether.certification.tube import Box, rough_enclosure

__all__ = ["SERIES_OPS", "jacobian_enclosure", "mean_value_step"]

#: The operation set on Arb power series rather than on balls.
SERIES_OPS = MathOps(
    sin=lambda x: x.sin(),
    cos=lambda x: x.cos(),
    tan=lambda x: x.tan(),
    exp=lambda x: x.exp(),
    sqrt=lambda x: x.sqrt(),
)


def jacobian_enclosure(
    field: VectorField,
    box: Sequence[tuple[float, float]],
    control: tuple[float, float],
) -> list[list[tuple[float, float]]]:
    r"""Rigorous bounds on :math:`\partial f_i/\partial x_j` over the box.

    Returns an ``n x n`` grid of ``(lower, upper)`` pairs, each provably
    containing the partial derivative everywhere in the box, where ``n`` is the
    state dimension implied by ``box``. Obtained by automatic differentiation in
    Arb power series, so the enclosure covers the derivative itself rather than
    a difference quotient standing in for it.
    """
    import flint

    n = len(box)
    if n == 0:
        raise ValueError("the box has no components, so there is no Jacobian")
    balls = [flint.arb(0.5 * (lo + hi), 0.5 * (hi - lo)) for lo, hi in box]
    control_ball = flint.arb(0.5 * (control[0] + control[1]), 0.5 * (control[1] - control[0]))

    rows: list[list[tuple[float, float]]] = [[] for _ in range(n)]
    for j in range(n):
        # x_j + eps, every other component constant: the linear coefficient of
        # the result is the partial with respect to component j.
        state: list[Any] = list(balls)
        state[j] = flint.arb_series([balls[j], 1])
        rates = field(state, control_ball, SERIES_OPS)
        if len(rates) != n:
            raise ValueError(
                f"the field returned {len(rates)} derivatives for a {n}-component "
                f"state; a Jacobian needs them to match"
            )
        for i in range(n):
            entry = rates[i]
            derivative = entry[1] if hasattr(entry, "__getitem__") else flint.arb(0)
            rows[i].append((float(derivative.lower()), float(derivative.upper())))
    return rows


def mean_value_step(
    field: VectorField,
    box: Sequence[tuple[float, float]],
    control: tuple[float, float],
    dt: float,
) -> Box:
    """One verified step in mean-value form, keeping the centre as a point.

    Same guarantee as :func:`~aether.certification.tube.reachable_step` -- every
    solution starting in ``box`` lands in the result -- but the deviation from
    the centre is propagated through the Jacobian rather than folded into a
    single enclosure of the field, so correlated variation stays correlated.

    The rough enclosure is still what makes the step legitimate: it is the
    Picard containment proving a solution exists across the step at all, and the
    Jacobian is taken over it because the mean-value theorem needs the derivative
    on the segment travelled, not merely at the endpoints.
    """
    n = len(box)
    enclosure = rough_enclosure(field, box, control, dt)
    jacobian = jacobian_enclosure(field, enclosure, control)

    centre = np.array([0.5 * (lo + hi) for lo, hi in box], dtype=np.float64)
    half = np.array([0.5 * (hi - lo) for lo, hi in box], dtype=np.float64)
    # f at the centre, with the control still an interval: it is not a deviation
    # from a nominal, it ranges over its whole admissible set.
    centre_field = enclose_field(field, [(float(c), float(c)) for c in centre], control)
    # Second-order remainder (h^2/2)(J f)(Y) on the rough enclosure. This is the
    # Taylor remainder; without it the expansion is not an enclosure at all.
    field_on_enclosure = enclose_field(field, enclosure, control)

    result: Box = []
    for i in range(n):
        low = centre[i] + dt * centre_field[i][0]
        high = centre[i] + dt * centre_field[i][1]
        spread = 0.0
        remainder = 0.0
        for j in range(n):
            lo_j, hi_j = jacobian[i][j]
            magnitude = max(abs(lo_j), abs(hi_j))
            # (I + h J)(X - c): an interval matrix on a symmetric box, so each
            # column contributes |entry| * half_j to the half-width.
            coefficient = (
                max(abs(1.0 + dt * lo_j), abs(1.0 + dt * hi_j)) if i == j else dt * magnitude
            )
            spread += coefficient * half[j]
            lo_f, hi_f = field_on_enclosure[j]
            remainder += magnitude * max(abs(lo_f), abs(hi_f))
        result.append(
            (low - spread - 0.5 * dt * dt * remainder, high + spread + 0.5 * dt * dt * remainder)
        )
    return result
