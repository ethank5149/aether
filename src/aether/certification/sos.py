r"""Sum-of-squares polynomial machinery for certified bounds.

A polynomial :math:`p(x) \ge 0` everywhere on a semialgebraic set is established
by writing :math:`p = \sigma_0 + \sum \sigma_i g_i` where each :math:`\sigma_i`
is a sum of squares and each :math:`g_i \ge 0` defines the set. The
decomposition is found by semidefinite programming and verified in exact
arithmetic -- the SDP provides the candidate, and Arb confirms it.

The SOS condition :math:`\sigma = z^\top Q z` (with :math:`z` a vector of
monomials and :math:`Q \succeq 0`) is the Gram matrix parameterisation: the
polynomial's coefficients are linear in the entries of :math:`Q`, and positive
semidefiniteness is an LMI constraint. This module handles the bookkeeping --
enumerating monomials, building the coefficient map, and wiring the SDP -- so
the caller works with polynomials and constraints rather than with matrices.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from itertools import combinations_with_replacement
from typing import Any

import cvxpy as cp
import numpy as np

__all__ = [
    "Monomial",
    "Polynomial",
    "sos_polynomial",
    "monomial_basis",
    "poly_multiply",
    "poly_add",
    "poly_differentiate",
    "poly_evaluate",
    "poly_substitute_affine",
    "poly_degree",
]

Monomial = tuple[int, ...]
Polynomial = dict[Monomial, Any]


@lru_cache(maxsize=64)
def monomial_basis(n_vars: int, max_degree: int) -> tuple[Monomial, ...]:
    """Every monomial in ``n_vars`` variables up to total degree ``max_degree``."""
    if n_vars == 0:
        return ((),)
    result: list[Monomial] = []
    for degree in range(max_degree + 1):
        for combo in combinations_with_replacement(range(n_vars), degree):
            exponents = [0] * n_vars
            for index in combo:
                exponents[index] += 1
            result.append(tuple(exponents))
    return tuple(result)


def _mono_add(a: Monomial, b: Monomial) -> Monomial:
    return tuple(ai + bi for ai, bi in zip(a, b))


def poly_multiply(p: Polynomial, q: Polynomial) -> Polynomial:
    """Multiply two polynomials, both represented as coefficient dicts."""
    result: Polynomial = defaultdict(lambda: 0.0)
    for mono_p, coeff_p in p.items():
        for mono_q, coeff_q in q.items():
            result[_mono_add(mono_p, mono_q)] += coeff_p * coeff_q
    return dict(result)


def poly_add(p: Polynomial, q: Polynomial, *, scale_q: Any = 1.0) -> Polynomial:
    """Add ``p + scale_q * q``."""
    result: Polynomial = dict(p)
    for mono, coeff in q.items():
        if mono in result:
            result[mono] = result[mono] + scale_q * coeff
        else:
            result[mono] = scale_q * coeff
    return result


def poly_differentiate(p: Polynomial, var: int) -> Polynomial:
    """Partial derivative of ``p`` with respect to variable ``var``."""
    result: Polynomial = {}
    for mono, coeff in p.items():
        if mono[var] == 0:
            continue
        new_mono = list(mono)
        power = new_mono[var]
        new_mono[var] -= 1
        result[tuple(new_mono)] = coeff * power
    return result


def poly_evaluate(p: Polynomial, values: dict[int, float] | list[float]) -> float:
    """Evaluate a polynomial at a point."""
    if isinstance(values, dict):
        vals = values
    else:
        vals = {i: v for i, v in enumerate(values)}
    total = 0.0
    for mono, coeff in p.items():
        term = float(coeff)
        for var, power in enumerate(mono):
            if power > 0:
                term *= vals[var] ** power
        total += term
    return total


def sos_polynomial(
    n_vars: int,
    degree: int,
    name: str = "Q",
) -> tuple[cp.Variable, Polynomial]:
    """A sum-of-squares polynomial as a cvxpy decision variable.

    Returns ``(Q, coefficients)`` where ``Q`` is a symmetric PSD matrix variable
    and ``coefficients`` maps each monomial to the linear expression in ``Q``
    entries that gives its coefficient. The caller adds ``Q >> 0`` to the SDP.

    The polynomial :math:`\\sigma(x) = z(x)^\\top Q\\, z(x)` where :math:`z` is
    the vector of monomials up to degree ``degree // 2``.
    """
    half = degree // 2
    basis = monomial_basis(n_vars, half)
    n = len(basis)
    Q = cp.Variable((n, n), symmetric=True, name=name)

    coefficients: Polynomial = defaultdict(lambda: 0.0)
    for i in range(n):
        for j in range(n):
            product_mono = _mono_add(basis[i], basis[j])
            coefficients[product_mono] = coefficients[product_mono] + Q[i, j]

    return Q, dict(coefficients)


def _zero_mono(n_vars: int) -> Monomial:
    return (0,) * n_vars


def constant_poly(n_vars: int, value: Any) -> Polynomial:
    """A constant polynomial."""
    return {_zero_mono(n_vars): value}


def variable_poly(n_vars: int, var: int, *, scale: float = 1.0, offset: float = 0.0) -> Polynomial:
    """The polynomial ``scale * x_var + offset``."""
    mono = [0] * n_vars
    mono[var] = 1
    result: Polynomial = {tuple(mono): scale}
    if offset != 0.0:
        result[_zero_mono(n_vars)] = offset
    return result


def poly_degree(p: Polynomial) -> int:
    """Total degree of the polynomial."""
    return max(sum(mono) for mono in p) if p else 0


def poly_substitute_affine(
    p: Polynomial,
    offsets: list[float],
    scales: list[float],
) -> Polynomial:
    """Substitute ``x_i -> offsets[i] + scales[i] * x_bar_i`` into *p*."""
    from math import comb

    n_vars = len(offsets)
    zero: Monomial = (0,) * n_vars
    result: Polynomial = {}
    for mono, coeff in p.items():
        term: Polynomial = {zero: coeff}
        for i, exp_i in enumerate(mono):
            if exp_i == 0:
                continue
            factor: Polynomial = {}
            for k in range(exp_i + 1):
                c = comb(exp_i, k) * offsets[i] ** (exp_i - k) * scales[i] ** k
                e = [0] * n_vars
                e[i] = k
                factor[tuple(e)] = c
            term = poly_multiply(term, factor)
        result = poly_add(result, term)
    return result
