r"""SymPy as an arithmetic, and the two operations a derivation is made of.

:class:`~aether.certification.rigorous.MathOps` exists so that a field is
written once and evaluated in whichever arithmetic is needed. Floats simulate
and Arb balls prove; this adds the third the design anticipated. A field
evaluated on symbols returns the expressions themselves, which is what lets a
derivation be checked against the object that is integrated instead of against
a second copy typed out for the purpose.

A derivation, as the manuscript uses the word, is then one of two things:
the rate of some quantity along the flow (:func:`lie_derivative`), or the
assertion that two expressions are the same (:func:`is_identically_zero`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import sympy as sp

from aether.certification.rigorous import MathOps

__all__ = ["SYMPY_OPS", "is_identically_zero", "lie_derivative", "reduce"]

#: The operation set on SymPy expressions.
SYMPY_OPS = MathOps(sin=sp.sin, cos=sp.cos, tan=sp.tan, exp=sp.exp, sqrt=sp.sqrt)


def lie_derivative(expression: Any, state: Sequence[Any], rates: Sequence[Any]) -> Any:
    r"""Rate of ``expression`` along the flow, :math:`\sum_i \partial_{x_i}e\;f_i`.

    Entrywise on a matrix. This is the chain rule and nothing else, which is
    the reason to route every time derivative through it: a derivation that
    differentiates along the flow by hand has a second place to be wrong.
    """
    if len(state) != len(rates):
        raise ValueError(
            f"one rate per state component: {len(state)} components, {len(rates)} rates"
        )

    def along_flow(entry: Any) -> Any:
        return sum(
            (sp.diff(entry, x) * rate for x, rate in zip(state, rates, strict=True)),
            sp.S.Zero,
        )

    if isinstance(expression, sp.MatrixBase):
        return expression.applyfunc(along_flow)
    return along_flow(sp.sympify(expression))


#: Exact rewrites, cheapest first. Each maps an expression to an equal one, so
#: an expression any of them sends to zero *is* zero; none of them is a
#: numerical test.
_REWRITES: tuple[Callable[[Any], Any], ...] = (
    sp.expand,
    lambda e: sp.trigsimp(sp.expand(e, trig=True)),
    sp.simplify,
    lambda e: sp.simplify(sp.expand_trig(e)),
)


def reduce(expression: Any) -> Any:
    """The simplest form the exact rewrites reach: zero if any of them finds it."""
    entry = sp.sympify(expression)
    if entry == 0:
        return sp.S.Zero
    best = entry
    for rewrite in _REWRITES:
        candidate = rewrite(entry)
        if candidate == 0:
            return sp.S.Zero
        if sp.count_ops(candidate) < sp.count_ops(best):
            best = candidate
    return best


def _entries(expression: Any) -> Iterable[Any]:
    if isinstance(expression, sp.MatrixBase):
        return list(expression)
    if isinstance(expression, (list, tuple)):
        return [entry for item in expression for entry in _entries(item)]
    return [expression]


def is_identically_zero(expression: Any) -> bool:
    """Whether every entry of ``expression`` reduces to zero by exact rewriting.

    One-sided, and the side matters. ``True`` is a proof: each rewrite preserves
    equality, so reaching zero establishes the identity. ``False`` means only
    that these rewrites did not get there, which is how an identity that is
    true but awkwardly stated shows up -- restate it rather than weaken the
    test. Deciding whether an arbitrary elementary expression vanishes is not
    computable in general, so no version of this can promise more.
    """
    return all(reduce(entry) == 0 for entry in _entries(expression))
