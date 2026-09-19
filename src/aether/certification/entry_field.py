r"""The six-state entry field, written for every arithmetic at once.

The application half of the certification split. The machinery that encloses and
propagates a field is in :mod:`aether.certification` and is about interval
arithmetic; this is about an entry body.

Written against :class:`~aether.certification.rigorous.MathOps` rather than
against NumPy, so the same definition evaluates in floats to simulate, in Arb
balls to prove, and in Arb power series to differentiate. It reproduces
:func:`aether.guidance.entry.entry_dynamics` **bit for bit** on floats,
asserted with ``array_equal`` rather than a tolerance so the two cannot drift.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from aether.certification import (
    NUMPY_OPS,
    MathOps,
    VectorField,
    enclose_field,
    lower_bound,
)

__all__ = ["enclose_entry_field", "entry_field", "entry_vector_field"]

#: Exponential atmosphere, matching :mod:`aether.guidance.entry`.
_RHO0 = 1.225
_H_SCALE = 8500.0
_MU = 3.986004418e14
_R_EARTH = 6378137.0


def entry_field(
    state: Sequence[Any],
    bank: Any,
    ballistic_coefficient: float,
    lift_to_drag: float,
    ops: MathOps = NUMPY_OPS,
    *,
    body_radius: float = _R_EARTH,
) -> list[Any]:
    r"""The six-state entry vector field, over any arithmetic ``ops`` supports.

    Numerically identical to
    :func:`aether.guidance.entry.entry_dynamics` for float inputs -- the
    same expressions in the same order -- but written against :class:`MathOps`
    rather than against NumPy, so the same definition also evaluates in Arb.

    ``ballistic_coefficient`` and ``lift_to_drag`` are taken as plain numbers
    rather than an :class:`~aether.guidance.entry.EntryBody`, so that
    a certificate can be discharged over a *range* of body parameters by
    passing balls for them. The coefficient envelopes the proposal carries are
    exactly that kind of range.

    The altitude is required to stay above the surface. The float field clamps
    it with ``max(h, 0)``, and a clamp is a branch on sign -- which an interval
    straddling zero cannot take. Refusing is the honest behaviour: the enclosure
    that a clamp would produce is sound but so wide as to be useless, and a
    silent widening is worse than a stated limit.
    """
    radius, _longitude, latitude, speed, gamma, heading = state
    altitude = radius - body_radius
    lower_altitude = lower_bound(altitude)
    if lower_altitude < 0.0:
        raise ValueError(
            f"the enclosure reaches below the surface (altitude down to "
            f"{lower_altitude:.1f} m); the float field clamps this with "
            f"max(h, 0), which is a branch on sign that an interval spanning "
            f"zero cannot take"
        )

    density = _RHO0 * ops.exp(-altitude / _H_SCALE)
    drag = 0.5 * density * speed * speed / ballistic_coefficient
    lift = lift_to_drag * drag
    gravity = _MU / (radius * radius)

    cos_gamma = ops.cos(gamma)
    sin_gamma = ops.sin(gamma)
    return [
        speed * sin_gamma,
        speed * cos_gamma * ops.sin(heading) / (radius * ops.cos(latitude)),
        speed * cos_gamma * ops.cos(heading) / radius,
        -drag - gravity * sin_gamma,
        (lift * ops.cos(bank) + (speed * speed / radius - gravity) * cos_gamma) / speed,
        (
            lift * ops.sin(bank) / cos_gamma
            + speed * speed * cos_gamma * ops.sin(heading) * ops.tan(latitude) / radius
        )
        / speed,
    ]


def entry_vector_field(
    ballistic_coefficient: float,
    lift_to_drag: float,
    *,
    body_radius: float = _R_EARTH,
) -> VectorField:
    """:func:`entry_field` closed over its body parameters, as a bare ``f(x, u, ops)``.

    The boundary between the body and the mathematics. Everything upstream of
    this takes a field; everything downstream is about an entry body.
    """

    def field(state: Sequence[Any], control: Any, ops: MathOps) -> list[Any]:
        return entry_field(
            state, control, ballistic_coefficient, lift_to_drag, ops,
            body_radius=body_radius,
        )

    return field


def enclose_entry_field(
    box: Sequence[tuple[float, float]],
    bank: tuple[float, float],
    ballistic_coefficient: float,
    lift_to_drag: float,
    *,
    body_radius: float = _R_EARTH,
) -> list[tuple[float, float]]:
    """Rigorous bounds on the entry field over a box of states and bank angles."""
    if len(box) != 6:
        raise ValueError(f"the entry state has six components, got {len(box)}")
    return enclose_field(
        entry_vector_field(ballistic_coefficient, lift_to_drag, body_radius=body_radius),
        box, bank,
    )
