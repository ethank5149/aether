r"""Rigorous enclosures of a vector field, by Arb ball arithmetic.

A certificate is an inequality asserted over a whole set, and a floating-point
evaluation cannot establish one: rounding can exceed the margin being checked,
and the failure is silent and one-sided, because nothing in a float computation
announces that the true value lay on the other side of zero.

Ball arithmetic supplies the missing guarantee. Every quantity is a midpoint
with a radius, every operation returns a ball provably containing the true
result, and an inequality is discharged only when the *entire* ball lies on the
right side of zero. What comes back is either a proof or an honest "cannot tell
at this precision" -- never a wrong answer with a confident face. Arb
(Johansson, "Arb: Efficient Arbitrary-Precision Midpoint-Radius Interval
Arithmetic", *IEEE Trans. Computers* 66(8), 2017) is the implementation, reached
through ``python-flint``.

The design constraint this imposes
----------------------------------

A field cannot be written against NumPy. ``np.sin`` does not accept an ``arb``,
so a field expressed in NumPy calls evaluates in floating point and nowhere
else. The same constraint arrives from the other direction whenever derivatives
are wanted: a symbolic backend needs its own types too, and writing the dynamics
a second time in those types is how implementations drift apart.

So a field is written once against an *operation set* and evaluated through
whichever backend is needed: floats to simulate, Arb to prove, symbolic types to
differentiate. One definition, several arithmetics. The alternative is one copy
per arithmetic, and the copies drift.
"""

from __future__ import annotations

import numbers
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "ARB_OPS",
    "NUMPY_OPS",
    "MathOps",
    "VectorField",
    "enclose_field",
    "interval",
]


@dataclass(frozen=True)
class MathOps:
    """The elementary operations a vector field is allowed to use.

    Deliberately small. Every function here exists in floating point, in Arb, and
    in the symbolic types an automatic-differentiation backend supplies, so a
    field written against this set evaluates in all three. A field reaching past
    it -- for a table lookup, a branch on sign, a ``float()`` cast -- is
    restricted to floats again, which is why the boundary is named rather than
    left implicit.
    """

    sin: Callable[[Any], Any]
    cos: Callable[[Any], Any]
    tan: Callable[[Any], Any]
    exp: Callable[[Any], Any]
    sqrt: Callable[[Any], Any]


NUMPY_OPS = MathOps(sin=np.sin, cos=np.cos, tan=np.tan, exp=np.exp, sqrt=np.sqrt)

#: The same operations on Arb balls, as bound methods.
ARB_OPS = MathOps(
    sin=lambda x: x.sin(),
    cos=lambda x: x.cos(),
    tan=lambda x: x.tan(),
    exp=lambda x: x.exp(),
    sqrt=lambda x: x.sqrt(),
)

#: A vector field written against :class:`MathOps`: ``f(state, control, ops)``
#: returning one derivative per state component. Everything here is written
#: against this rather than against any particular field.
VectorField = Callable[[Sequence[Any], Any, MathOps], list[Any]]


def lower_bound(value: Any) -> float:
    """Least value ``value`` can take, across every backend a field runs on.

    Three cases, because a field is evaluated in three arithmetics: a real
    number is its own bound, an Arb ball exposes ``lower()``, and an Arb power
    series has no bound of its own -- its constant coefficient is the ball, and
    the higher coefficients describe variation in the perturbation rather than
    in the state.

    Ordered by type rather than by attribute. A NumPy scalar *has*
    ``__getitem__`` -- it raises ``IndexError`` when used -- so a ``hasattr``
    check matches it and then fails, which is how an earlier version broke the
    float path while fixing the series one.
    """
    if isinstance(value, numbers.Real):
        return float(value)
    if hasattr(value, "lower"):
        return float(value.lower())
    return float(value[0].lower())


def interval(lower: float, upper: float) -> Any:
    """An Arb ball covering ``[lower, upper]`` exactly.

    Arb is midpoint-radius rather than endpoint-based, so an interval is given
    as its centre and half-width. The radius is rounded *outward* by Arb itself,
    which keeps the enclosure sound when the midpoint is not representable.
    """
    import flint

    if not (np.isfinite(lower) and np.isfinite(upper)):
        raise ValueError(
            f"an enclosure needs finite endpoints, got [{lower}, {upper}]"
        )
    if lower > upper:
        raise ValueError(
            f"the interval [{lower}, {upper}] is empty; endpoints are "
            f"(lower, upper) and this pair is reversed"
        )
    return flint.arb(0.5 * (lower + upper), 0.5 * (upper - lower))


def enclose_field(
    field: VectorField,
    box: Sequence[tuple[float, float]],
    control: tuple[float, float],
) -> list[tuple[float, float]]:
    r"""Rigorous bounds on :math:`f(x, u)` over a box of states and controls.

    The primitive every certificate rests on. A barrier condition
    :math:`\nabla B \cdot f \le 0` on a set is discharged by covering the set
    with boxes and showing the *enclosure* of the left side is non-positive on
    each, so a sound enclosure of :math:`f` is what turns the argument from
    numerical evidence into proof.

    Sound but not tight. Interval arithmetic evaluates each operation
    independently and so loses the correlation between repeated occurrences of a
    variable; the consequence is over-estimation growing with box width, which is
    why certified reachability subdivides and why
    :func:`~aether.certification.jacobian.mean_value_step` exists. It never
    under-estimates, and that asymmetry is the point: an answer too wide costs
    effort, an answer too narrow costs the proof.
    """
    state = [interval(low, high) for low, high in box]
    values = field(state, interval(*control), ARB_OPS)
    return [(float(value.lower()), float(value.upper())) for value in values]
