"""Ultraspherical (Gegenbauer) spectral method (Paper II, §5.4 and Appendix A).

Coefficient-space spectral discretization after Olver & Townsend (2013):
the solution lives in the Chebyshev basis, its :math:`k`-th derivative
in the ultraspherical basis :math:`C^{(k)}`, where differentiation is a
single-band operator and basis conversion is tridiagonal. Variable
coefficients enter as banded multiplication operators built from the
Jacobi (multiplication-by-:math:`x`) recurrence, so an assembled
:math:`k`-th order variable-coefficient operator is banded — in contrast
to the dense :math:`\\mathcal{O}(N^{2k})`-conditioned collocation
matrices of Paper I — with boundary conditions appended as dense rows
rather than substituted.
"""

from __future__ import annotations

from aether.ultraspherical.assembly import UltrasphericalBVP, VariableCoefficientOperator
from aether.ultraspherical.operators import (
    chebyshev_coefficients,
    chebyshev_values,
    conversion_chain,
    conversion_operator,
    derivative_in_basis,
    diff_operator,
    evaluation_row,
    jacobi_operator,
    multiplication_operator,
)

__all__ = [
    "UltrasphericalBVP",
    "VariableCoefficientOperator",
    "chebyshev_coefficients",
    "chebyshev_values",
    "conversion_chain",
    "conversion_operator",
    "derivative_in_basis",
    "diff_operator",
    "evaluation_row",
    "jacobi_operator",
    "multiplication_operator",
]
