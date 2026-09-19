"""Rigorous machinery for certified reachability.

Everything here answers a question of the form "is this inequality *provably*
true", as distinct from "did a float computation suggest so". That distinction
is the whole content of a certified bound: a barrier certificate discharged in
floating point is evidence, not proof, because the arithmetic that checked it
can be wrong by more than the margin it was checking.

The package knows nothing about any particular system. Every entry point takes a
:data:`~aether.certification.rigorous.VectorField` -- ``f(state, control, ops)``
written against :class:`~aether.certification.rigorous.MathOps` -- so the field
and the machinery that certifies it stay separable.
"""

from aether.certification.jacobian import (
    SERIES_OPS,
    jacobian_enclosure,
    mean_value_step,
)
from aether.certification.rigorous import (
    ARB_OPS,
    NUMPY_OPS,
    MathOps,
    VectorField,
    enclose_field,
    interval,
    lower_bound,
)
from aether.certification.tube import (
    Box,
    reachable_step,
    reachable_tube,
    rough_enclosure,
    tube_widths,
)

__all__ = [
    "ARB_OPS",
    "NUMPY_OPS",
    "SERIES_OPS",
    "Box",
    "MathOps",
    "VectorField",
    "enclose_field",
    "interval",
    "jacobian_enclosure",
    "lower_bound",
    "mean_value_step",
    "reachable_step",
    "reachable_tube",
    "rough_enclosure",
    "tube_widths",
]
