r"""Certified reachable tubes, by verified one-step propagation.

:func:`~aether.certification.rigorous.enclose_field` bounds a field over a box.
This turns that into a bound on where the *trajectory* goes, which is what a
reachable set is.

The step is the classical first-order verified scheme (Moore, *Interval
Analysis*, 1966; Lohner, "Enclosing the solutions of ordinary initial and
boundary value problems", 1987). It has two halves, and the first carries the
proof:

**Existence.** A *rough enclosure* :math:`Y` for the step is a box satisfying

.. math::

    X + [0,h]\,f(Y) \subseteq Y.

By the Picard-Lindelöf argument this containment does more than bound the
solution -- it *establishes that a solution exists* across the whole step and
stays inside :math:`Y`. Without it the second half would be bounding something
not yet known to be there. When the containment cannot be achieved the honest
answer is refusal, not a wider guess.

**Propagation.** Given such a :math:`Y`,

.. math::

    x(h) = x(0) + \int_0^h f(x(s))\,\mathrm{d}s \in X + h\,f(Y),

because :math:`f(x(s)) \in f(Y)` for every :math:`s` in the step. Sound, and
first order in :math:`h`.

Quantifying over the control
----------------------------

``control`` is an interval and the field is enclosed over all of it at every
step, so a tube covers *every* admissible control history -- not one, and not a
finite family. That is the difference between a reachable set and a bundle of
trajectories, and the reason a sampled sweep can only ever bound a reachable set
from inside.

The wrapping effect
-------------------

Boxes grow, and not only because the true reachable set grows. Representing a
set by an axis-aligned box discards its shape, so a set that rotates under the
flow is re-covered each step by a box large enough to hold its rotated extent,
and the excess compounds. This is a property of the *representation*, not of the
dynamics: it is why serious certified reachability carries zonotopes or Taylor
models. :func:`tube_widths` reports the growth so a reader can see where the
representation stops paying, rather than being handed a tube that has quietly
become vacuous.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from aether.certification.rigorous import VectorField, enclose_field

__all__ = [
    "Box",
    "reachable_step",
    "reachable_tube",
    "rough_enclosure",
    "tube_widths",
]

#: A box is a list of ``(lower, upper)`` pairs, one per state component.
Box = list[tuple[float, float]]


def _contains(
    outer: Sequence[tuple[float, float]], inner: Sequence[tuple[float, float]]
) -> bool:
    return all(
        lo_o <= lo_i and hi_i <= hi_o
        for (lo_o, hi_o), (lo_i, hi_i) in zip(outer, inner, strict=True)
    )


def _inflate(box: Sequence[tuple[float, float]], factor: float, floor: float) -> Box:
    """Widen a box about its centre, with an absolute floor for degenerate sides.

    The floor matters: a component of zero width stays zero under a purely
    multiplicative inflation, so a Picard iteration started from a point box
    would never grow and never verify.
    """
    out: Box = []
    for lower, upper in box:
        centre = 0.5 * (lower + upper)
        half = max(0.5 * (upper - lower) * factor, floor)
        out.append((centre - half, centre + half))
    return out


def rough_enclosure(
    field: VectorField,
    box: Sequence[tuple[float, float]],
    control: tuple[float, float],
    dt: float,
    *,
    inflation: float = 1.5,
    max_iterations: int = 40,
) -> Box:
    r"""A box provably containing the solution for the whole step.

    Iterates :math:`Y \leftarrow X + [0,h] f(Y)` from an inflated start until
    :math:`X + [0,h] f(Y) \subseteq Y` holds. That containment is the
    Picard-Lindelöf hypothesis on this step, so achieving it proves a solution
    exists here and remains in :math:`Y` -- the enclosure is a by-product of an
    existence proof rather than an assumption.

    Raises when it cannot be achieved within ``max_iterations``. That is the
    correct outcome and not a failure to work around: the step is too long for
    how much the field varies over the box, and the answer is a shorter step,
    not a wider box asserted without justification.
    """
    if dt <= 0.0:
        raise ValueError(f"the step must be positive, got {dt}")
    widths = [hi - lo for lo, hi in box]
    candidate = _inflate(box, inflation, max(1e-6, 1e-3 * max(widths, default=1.0)))
    for _ in range(max_iterations):
        rates = enclose_field(field, candidate, control)
        # X + [0, h] f(Y): the interval [0, h] keeps the whole step, not its end.
        picard: Box = [
            (lo_x + min(0.0, dt * lo_f), hi_x + max(0.0, dt * hi_f))
            for (lo_x, hi_x), (lo_f, hi_f) in zip(box, rates, strict=True)
        ]
        if _contains(candidate, picard):
            return candidate
        candidate = _inflate(
            picard, inflation, max(1e-9, 1e-6 * max(widths, default=1.0))
        )
    raise ValueError(
        f"no rough enclosure verified in {max_iterations} iterations at dt={dt}; "
        f"the step is too long for how much the field varies over this box -- "
        f"shorten it rather than widening the box by fiat"
    )


def reachable_step(
    field: VectorField,
    box: Sequence[tuple[float, float]],
    control: tuple[float, float],
    dt: float,
    **kwargs: float,
) -> Box:
    """One verified step: every solution starting in ``box`` lands in the result."""
    enclosure = rough_enclosure(field, box, control, dt, **kwargs)  # type: ignore[arg-type]
    rates = enclose_field(field, enclosure, control)
    return [
        (lo_x + dt * lo_f, hi_x + dt * hi_f)
        for (lo_x, hi_x), (lo_f, hi_f) in zip(box, rates, strict=True)
    ]


def reachable_tube(
    field: VectorField,
    box: Sequence[tuple[float, float]],
    control: tuple[float, float],
    dt: float,
    n_steps: int,
) -> list[Box]:
    """Successive verified boxes, each containing every solution at that time.

    The outer half of a verification sandwich: a sampling method bounds a
    reachable set from inside, this bounds it from outside, and a sampled
    trajectory found outside would falsify one of the two -- the only kind of
    check that can catch an error in both.
    """
    tube: list[Box] = []
    current: Box = [(float(lo), float(hi)) for lo, hi in box]
    for _ in range(n_steps):
        current = reachable_step(field, current, control, dt)
        tube.append(current)
    return tube


def tube_widths(tube: Sequence[Sequence[tuple[float, float]]]) -> np.ndarray:
    """Per-step component widths, for seeing where the representation stops paying."""
    return np.array([[hi - lo for lo, hi in box] for box in tube], dtype=np.float64)
