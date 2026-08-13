"""Sparse ultraspherical operator primitives (Paper II, Appendix A).

All operators act on *coefficient vectors*. The solution is expanded in
Chebyshev polynomials :math:`T_m`; its :math:`k`-th derivative is
represented in the ultraspherical basis :math:`C^{(k)}` via the
derivative relation (Paper II, Eq. A.1)

.. math::

    \\frac{d^k}{dx^k} T_m = 2^{k-1}(k-1)!\\; m\\, C_{m-k}^{(k)}
    \\quad (m \\ge k),

which makes :math:`\\mathcal{D}_k` a single superdiagonal (Eq. 5.18).
Conversion between adjacent bases (Eq. A.2) gives the tridiagonal
:math:`\\mathcal{S}_\\lambda`, and multiplication by a function with a
decaying Chebyshev expansion is a banded operator built from the
Gegenbauer three-term recurrence applied to the Jacobi operator
(multiplication by :math:`x`) — the operators are polynomials in a
tridiagonal matrix, hence banded with bandwidth equal to the coefficient
truncation length.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import scipy.fft
import scipy.sparse
from numpy.typing import ArrayLike, NDArray

__all__ = [
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

_FloatArray = NDArray[np.float64]
# scipy.sparse.csr_matrix; scipy ships no stubs, so the alias is Any to mypy
_Sparse = Any


def _validate_size(n: int) -> None:
    if not (isinstance(n, (int, np.integer)) and not isinstance(n, bool)) or n < 2:
        raise ValueError(f"coefficient-space dimension n must be an integer >= 2, got {n}")


def chebyshev_coefficients(values: ArrayLike) -> _FloatArray:
    """Chebyshev coefficients of the interpolant through CGL nodal values.

    Input values are sampled at the *descending* CGL nodes
    :math:`\\xi_j = \\cos(j\\pi/N)` (the convention of
    :mod:`aether.spectral`); output is :math:`a` with
    :math:`f = \\sum_k a_k T_k`. Uses the type-I DCT, exact for the
    interpolant.
    """
    f = np.asarray(values, dtype=np.float64)
    if f.ndim != 1 or f.size < 2:
        raise ValueError(f"values must be a 1-D array of >= 2 samples, got shape {f.shape}")
    n = f.size - 1
    x = scipy.fft.dct(f, type=1)
    coeffs = x / n
    coeffs[0] *= 0.5
    coeffs[-1] *= 0.5
    return np.asarray(coeffs)


def chebyshev_values(coeffs: ArrayLike, points: ArrayLike) -> _FloatArray:
    """Evaluate a Chebyshev series at arbitrary points (Clenshaw)."""
    a = np.asarray(coeffs, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"coeffs must be 1-D, got shape {a.shape}")
    return np.asarray(np.polynomial.chebyshev.chebval(np.asarray(points, dtype=np.float64), a))


def diff_operator(n: int, order: int) -> _Sparse:
    """The sparse differentiation operator :math:`\\mathcal{D}_k`
    (Paper II, Eq. 5.18): Chebyshev coefficients in, :math:`C^{(k)}`
    coefficients of the :math:`k`-th derivative out.

    Single superdiagonal at offset ``order`` with entries
    :math:`2^{k-1}(k-1)!\\,m` for column :math:`m = k, k+1, \\dots`.
    """
    _validate_size(n)
    if not 1 <= order < n:
        raise ValueError(f"derivative order must satisfy 1 <= order < n = {n}, got {order}")
    scale = 2.0 ** (order - 1) * float(math.factorial(order - 1)) if order > 1 else 1.0
    cols = np.arange(order, n)
    data = scale * cols.astype(np.float64)
    return scipy.sparse.csr_matrix(
        (data, (cols - order, cols)), shape=(n, n)
    )


def conversion_operator(n: int, lam: int) -> _Sparse:
    """The tridiagonal conversion :math:`\\mathcal{S}_\\lambda`.

    ``lam = 0`` maps Chebyshev :math:`T` to :math:`C^{(1)}`; ``lam >= 1``
    maps :math:`C^{(\\lambda)}` to :math:`C^{(\\lambda+1)}` via Paper II,
    Eq. (A.2). Both are upper-triangular with a main diagonal and one
    superdiagonal at offset 2.
    """
    _validate_size(n)
    if lam < 0:
        raise ValueError(f"lambda must be >= 0, got {lam}")
    i = np.arange(n, dtype=np.float64)
    if lam == 0:
        diag = np.where(i == 0, 1.0, 0.5)
        upper = np.full(n - 2, -0.5)
    else:
        diag = lam / (lam + i)
        upper = -(lam / (lam + i[2:]))
    return scipy.sparse.diags_array(
        [diag, upper], offsets=[0, 2], shape=(n, n), format="csr"
    )


def conversion_chain(n: int, lam_from: int, lam_to: int) -> _Sparse:
    """Product :math:`\\mathcal{S}_{\\lambda_{to}-1}\\cdots
    \\mathcal{S}_{\\lambda_{from}}` mapping basis
    :math:`\\lambda_{from}` (0 meaning Chebyshev) to
    :math:`\\lambda_{to}`; identity when equal."""
    _validate_size(n)
    if not 0 <= lam_from <= lam_to:
        raise ValueError(f"need 0 <= lam_from <= lam_to, got ({lam_from}, {lam_to})")
    op = scipy.sparse.identity(n, format="csr")
    for lam in range(lam_from, lam_to):
        op = conversion_operator(n, lam) @ op
    return scipy.sparse.csr_matrix(op)


def derivative_in_basis(n: int, derivative: int, basis: int, scale: float = 1.0) -> _Sparse:
    """:math:`d^j/dx^j` mapping Chebyshev coefficients into the
    :math:`C^{(\\lambda)}` basis, i.e.
    :math:`\\mathcal{S}_{\\lambda-1}\\cdots\\mathcal{S}_j\\,\\mathcal{D}_j`
    (:math:`j = 0` gives the pure conversion chain).

    This is the per-direction building block of the tensor-product
    assembly of Paper II, Eq. (5.19): every term of a bivariate operator
    is a Kronecker product of two such factors, so all terms land in one
    common output basis :math:`C^{(\\lambda)} \\otimes C^{(\\lambda)}`.

    Parameters
    ----------
    n:
        Coefficient-space dimension.
    derivative:
        Derivative order :math:`j`, :math:`0 \\le j \\le \\lambda`.
    basis:
        Target ultraspherical index :math:`\\lambda \\ge 1`.
    scale:
        Physical scaling applied as ``scale ** derivative`` — the
        :math:`(2/L)^j` factor of the affine map.
    """
    _validate_size(n)
    if basis < 1:
        raise ValueError(f"target basis must be >= 1, got {basis}")
    if not 0 <= derivative <= basis:
        raise ValueError(
            f"derivative must satisfy 0 <= j <= basis = {basis}, got {derivative}"
        )
    factor = float(scale) ** derivative
    if derivative == 0:
        return scipy.sparse.csr_matrix(factor * conversion_chain(n, 0, basis))
    op = conversion_chain(n, derivative, basis) @ diff_operator(n, derivative)
    return scipy.sparse.csr_matrix(factor * op)


def jacobi_operator(n: int, lam: int) -> _Sparse:
    """Multiplication by :math:`x` on coefficient vectors in basis
    :math:`C^{(\\lambda)}` (``lam = 0``: Chebyshev).

    From the three-term recurrence
    :math:`x\\,C^{(\\lambda)}_m = A_m C^{(\\lambda)}_{m+1} +
    B_m C^{(\\lambda)}_{m-1}` with
    :math:`A_m = (m+1)/(2(m+\\lambda))`,
    :math:`B_m = (m + 2\\lambda - 1)/(2(m+\\lambda))`; the Chebyshev
    case is the usual :math:`x T_0 = T_1`,
    :math:`x T_m = (T_{m+1} + T_{m-1})/2`.
    """
    _validate_size(n)
    if lam < 0:
        raise ValueError(f"lambda must be >= 0, got {lam}")
    m = np.arange(n, dtype=np.float64)
    if lam == 0:
        a = np.where(m == 0, 1.0, 0.5)  # T_{m} -> T_{m+1} weight
        b = np.full(n, 0.5)  # T_{m} -> T_{m-1} weight
    else:
        a = (m + 1.0) / (2.0 * (m + lam))
        b = (m + 2.0 * lam - 1.0) / (2.0 * (m + lam))
    # (X u)_i receives A_{i-1} u_{i-1} and B_{i+1} u_{i+1}
    return scipy.sparse.diags_array(
        [a[:-1], b[1:]], offsets=[-1, 1], shape=(n, n), format="csr"
    )


def multiplication_operator(
    coeff_cheb: ArrayLike,
    n: int,
    lam: int,
    tol: float = 1.0e-14,
) -> _Sparse:
    """Banded multiplication operator :math:`\\mathcal{M}^{(\\lambda)}[a]`.

    Given the *Chebyshev* coefficients of :math:`a(x)`, converts them to
    the :math:`C^{(\\lambda)}` basis and forms
    :math:`\\sum_j \\hat a_j\\, C_j^{(\\lambda)}(X_\\lambda)` by the
    Gegenbauer recurrence in the (tridiagonal) Jacobi operator
    :math:`X_\\lambda` — a polynomial in a tridiagonal matrix, hence
    banded with bandwidth equal to the truncation length. Coefficients
    below ``tol`` (relative to the largest) are truncated; slowly
    decaying coefficient expansions therefore cost bandwidth, which is
    Paper II's stated reason for hyperbolically blended property fields.
    """
    _validate_size(n)
    if lam < 0:
        raise ValueError(f"lambda must be >= 0, got {lam}")
    if not (np.isfinite(tol) and tol >= 0.0):
        raise ValueError(f"tol must be finite and >= 0, got {tol}")
    a_cheb = np.asarray(coeff_cheb, dtype=np.float64)
    if a_cheb.ndim != 1 or a_cheb.size == 0:
        raise ValueError(f"coeff_cheb must be a non-empty 1-D array, got shape {a_cheb.shape}")

    # coefficients of a in the C^(lam) basis
    if a_cheb.size < n:
        a_cheb = np.pad(a_cheb, (0, n - a_cheb.size))
    a_lam = np.asarray(conversion_chain(a_cheb.size, 0, lam) @ a_cheb)

    scale = float(np.max(np.abs(a_lam)))
    if scale == 0.0:
        return scipy.sparse.csr_matrix((n, n))
    keep = np.nonzero(np.abs(a_lam) > tol * scale)[0]
    m_trunc = int(keep[-1]) + 1 if keep.size else 1

    x_op = jacobi_operator(n, lam)
    p_prev = scipy.sparse.identity(n, format="csr")  # C_0(X) = I
    op = a_lam[0] * p_prev
    if m_trunc > 1:
        p_cur = (2.0 * lam) * x_op if lam >= 1 else x_op  # C_1 = 2*lam*x; T_1 = x
        op = op + a_lam[1] * p_cur
        for j in range(1, m_trunc - 1):
            if lam >= 1:
                p_next = (2.0 * (j + lam) / (j + 1.0)) * (x_op @ p_cur) - (
                    (j + 2.0 * lam - 1.0) / (j + 1.0)
                ) * p_prev
            else:
                p_next = 2.0 * (x_op @ p_cur) - p_prev
            op = op + a_lam[j + 1] * p_next
            p_prev, p_cur = p_cur, p_next
    return scipy.sparse.csr_matrix(op)


def evaluation_row(n: int, endpoint: int, derivative: int = 0) -> _FloatArray:
    """Dense boundary row: the functional :math:`u^{(d)}(\\pm 1)` acting
    on Chebyshev coefficients.

    Uses the closed forms :math:`T_m^{(d)}(1) = \\prod_{i=0}^{d-1}
    \\frac{m^2 - i^2}{2i + 1}` and
    :math:`T_m^{(d)}(-1) = (-1)^{m+d}\\,T_m^{(d)}(1)`; boundary
    conditions are *appended* as such rows, never substituted into the
    banded interior (Paper II, §5.4).
    """
    _validate_size(n)
    if endpoint not in (-1, 1):
        raise ValueError(f"endpoint must be -1 or +1, got {endpoint}")
    if derivative < 0:
        raise ValueError(f"derivative must be >= 0, got {derivative}")
    m = np.arange(n, dtype=np.float64)
    row = np.ones(n)
    for i in range(derivative):
        row *= (m * m - i * i) / (2.0 * i + 1.0)
    if endpoint == -1:
        row *= np.where((np.arange(n) + derivative) % 2 == 0, 1.0, -1.0)
    return row
