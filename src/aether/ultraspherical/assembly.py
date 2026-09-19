"""Variable-coefficient assembly and bordered solve (Paper II, Eq. A.3).

A :math:`k`-th order operator :math:`\\mathcal{L}u = \\sum_j a_j(x)
u^{(j)}` on an interval assembles as

.. math::

    \\mathcal{L} = \\sum_{j=0}^{k} \\left(\\tfrac{2}{b-a}\\right)^{j}
        \\mathcal{M}^{(k)}[a_j]\\;
        \\mathcal{S}_{k-1}\\cdots\\mathcal{S}_j\\, \\mathcal{D}_j ,

mapping Chebyshev coefficients of :math:`u` to :math:`C^{(k)}`
coefficients of :math:`\\mathcal{L}u` — banded throughout. Boundary
conditions are appended as dense rows on top of the interior operator,
with the trailing interior rows dropped to keep the system square
(boundary bordering), preserving the bandedness of everything below the
first ``n_bc`` rows.

The :math:`\\mathcal{O}(1)` conditioning claim of Olver & Townsend
applies to the preconditioned ultraspherical operator; this module's
:meth:`~UltrasphericalBVP.condition_number` therefore reports both the
raw bordered system and its column-equilibrated form, and the II-V1
runner records both — asserting only what the citation establishes and
measuring the rest, per the Remark in Paper II §5.4.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import ArrayLike, NDArray

from aether.ultraspherical.operators import (
    chebyshev_coefficients,
    chebyshev_values,
    conversion_chain,
    diff_operator,
    evaluation_row,
    multiplication_operator,
)

__all__ = ["BoundaryCondition", "UltrasphericalBVP", "VariableCoefficientOperator"]

_FloatArray = NDArray[np.float64]
_ScalarField = Callable[[_FloatArray], _FloatArray]


def _cgl_nodes(n: int) -> _FloatArray:
    j = np.arange(n)
    return np.asarray(np.sin(np.pi * (n - 1 - 2.0 * j) / (2.0 * (n - 1))))


@dataclass(frozen=True)
class BoundaryCondition:
    """One boundary functional :math:`\\sum_d c_d\\,u^{(d)}(x_b) = v`.

    Attributes
    ----------
    endpoint:
        ``-1`` or ``+1`` in reference coordinates.
    weights:
        ``{derivative_order: coefficient}`` in *physical* derivatives
        (the interval scaling is applied during assembly).
    value:
        Right-hand side of the condition.
    """

    endpoint: int
    weights: dict[int, float]
    value: float = 0.0

    def __post_init__(self) -> None:
        if self.endpoint not in (-1, 1):
            raise ValueError(f"endpoint must be -1 or +1, got {self.endpoint}")
        if not self.weights:
            raise ValueError("boundary condition needs at least one derivative weight")
        for d, c in self.weights.items():
            if d < 0 or not np.isfinite(c):
                raise ValueError(f"invalid boundary weight {c} on derivative {d}")


class VariableCoefficientOperator:
    """Banded ultraspherical form of :math:`\\sum_j a_j(x)\\,u^{(j)}`.

    Parameters
    ----------
    coefficients:
        ``coefficients[j]`` is :math:`a_j(x)` as a callable on the
        physical interval; ``None`` entries are absent terms. The list
        length fixes the operator order ``k = len(coefficients) - 1``
        and ``coefficients[k]`` must be present.
    n:
        Coefficient-space dimension (polynomial degree ``n - 1``).
    interval:
        Physical interval ``(a, b)``.
    coefficient_tol:
        Relative truncation tolerance for the coefficient expansions
        entering the multiplication operators (bandwidth control).
    """

    def __init__(
        self,
        coefficients: Sequence[_ScalarField | None],
        n: int,
        interval: tuple[float, float] = (-1.0, 1.0),
        coefficient_tol: float = 1.0e-14,
    ) -> None:
        if len(coefficients) < 2:
            raise ValueError("need at least a first-order operator (two coefficient slots)")
        if coefficients[-1] is None:
            raise ValueError("the leading-order coefficient must be present")
        a, b = float(interval[0]), float(interval[1])
        if not (np.isfinite(a) and np.isfinite(b) and a < b):
            raise ValueError(f"invalid interval {interval}")
        order = len(coefficients) - 1
        if not 1 <= order < n:
            raise ValueError(f"operator order {order} must satisfy 1 <= order < n = {n}")

        self._n = int(n)
        self._order = order
        self._interval = (a, b)
        scale = 2.0 / (b - a)

        xi = _cgl_nodes(n)
        x_phys = a + (b - a) * (xi + 1.0) / 2.0

        op = scipy.sparse.csr_matrix((n, n))
        for j, fn in enumerate(coefficients):
            if fn is None:
                continue
            coeff_vals = np.asarray(fn(x_phys), dtype=np.float64)
            if coeff_vals.shape != x_phys.shape or not np.all(np.isfinite(coeff_vals)):
                raise ValueError(f"coefficient {j} must return finite values on the grid")
            a_cheb = chebyshev_coefficients(coeff_vals)
            mult = multiplication_operator(a_cheb, n, order, tol=coefficient_tol)
            if j == 0:
                deriv_part = conversion_chain(n, 0, order)
            else:
                deriv_part = conversion_chain(n, j, order) @ diff_operator(n, j)
            op = op + (scale**j) * (mult @ deriv_part)
        self._matrix = scipy.sparse.csr_matrix(op)

    @property
    def n(self) -> int:
        return self._n

    @property
    def order(self) -> int:
        return self._order

    @property
    def interval(self) -> tuple[float, float]:
        return self._interval

    @property
    def matrix(self) -> scipy.sparse.csr_matrix:
        """Interior operator: T-coefficients in, :math:`C^{(k)}`
        coefficients out."""
        return self._matrix

    def apply(self, u_coeffs: ArrayLike) -> _FloatArray:
        """Apply to Chebyshev coefficients; returns :math:`C^{(k)}`
        coefficients of :math:`\\mathcal{L}u`."""
        u = np.asarray(u_coeffs, dtype=np.float64)
        if u.shape != (self._n,):
            raise ValueError(f"u_coeffs must have shape ({self._n},), got {u.shape}")
        return np.asarray(self._matrix @ u)

    def interior_condition_number(self, preconditioned: bool = False) -> float:
        """:math:`\\sigma_{\\max}/\\sigma_{\\min}` of the rectangular
        banded interior (the trailing ``order`` rows dropped, as in the
        bordered solve) — the quantity the Olver–Townsend conditioning
        claim is *about*, as distinct from the bordered system.

        With ``preconditioned = True`` each column :math:`m` is scaled by
        the inverse of the leading differentiation diagonal,
        :math:`1/\\max(1,\\,2^{k-1}(k-1)!\\,m)` — the natural right
        preconditioner under which the interior operator is
        :math:`\\mathcal{O}(1)`-conditioned.
        """
        import math as _math

        k = self._order
        dense = self._matrix[: self._n - k, :].toarray()
        if preconditioned:
            m = np.arange(self._n, dtype=np.float64)
            lead = 2.0 ** (k - 1) * float(_math.factorial(k - 1)) if k > 1 else 1.0
            dense = dense / np.maximum(1.0, lead * m)[np.newaxis, :]
        s = np.linalg.svd(dense, compute_uv=False)
        return float(s[0] / s[-1])


class UltrasphericalBVP:
    """Bordered square system: boundary rows over a truncated interior.

    Parameters
    ----------
    operator:
        The assembled interior operator.
    boundary_conditions:
        Exactly ``operator.order`` conditions for a well-posed BVP.
    """

    def __init__(
        self,
        operator: VariableCoefficientOperator,
        boundary_conditions: Sequence[BoundaryCondition],
    ) -> None:
        k = operator.order
        if len(boundary_conditions) != k:
            raise ValueError(
                f"a {k}-th order operator needs exactly {k} boundary conditions, "
                f"got {len(boundary_conditions)}"
            )
        self._op = operator
        self._bcs = list(boundary_conditions)
        n = operator.n
        a, b = operator.interval
        scale = 2.0 / (b - a)

        rows = []
        for bc in self._bcs:
            row = np.zeros(n)
            for d, c in bc.weights.items():
                row += c * (scale**d) * evaluation_row(n, bc.endpoint, d)
            rows.append(row)
        self._bc_rows = np.vstack(rows)
        interior = operator.matrix[: n - k, :]
        self._system = scipy.sparse.vstack(
            [scipy.sparse.csr_matrix(self._bc_rows), interior], format="csc"
        )
        try:
            self._lu = scipy.sparse.linalg.splu(self._system)
        except RuntimeError as exc:
            raise np.linalg.LinAlgError(
                "the bordered system is singular: the boundary conditions do not "
                "eliminate the operator's null space. Free-free conditions on a "
                "bending operator are the canonical case — they leave the two "
                "rigid-body modes, so the BVP is genuinely ill-posed and the "
                "problem must be posed as an eigenproblem instead"
            ) from exc

    @property
    def system(self) -> scipy.sparse.csc_matrix:
        """The bordered square system matrix."""
        return self._system

    def condition_number(self, equilibrated: bool = True) -> float:
        """:math:`\\kappa_2` of the bordered system.

        ``equilibrated = True`` scales each column to unit 2-norm first
        (the diagonal right-preconditioner under which the Olver–Townsend
        well-conditioning statement applies); ``False`` gives the raw
        bordered matrix. Computed densely — the II-V1 sweep sizes are
        modest by construction.
        """
        dense = self._system.toarray()
        if equilibrated:
            norms = np.linalg.norm(dense, axis=0)
            norms[norms == 0.0] = 1.0
            dense = dense / norms
        s = np.linalg.svd(dense, compute_uv=False)
        return float(s[0] / s[-1])

    def solve(self, rhs: _ScalarField) -> _FloatArray:
        """Solve :math:`\\mathcal{L}u = f` with the stored boundary values.

        Parameters
        ----------
        rhs:
            :math:`f(x)` on the physical interval.

        Returns
        -------
        numpy.ndarray
            Chebyshev coefficients of the solution.
        """
        n, k = self._op.n, self._op.order
        a, b = self._op.interval
        xi = _cgl_nodes(n)
        x_phys = a + (b - a) * (xi + 1.0) / 2.0
        f_vals = np.asarray(rhs(x_phys), dtype=np.float64)
        if f_vals.shape != x_phys.shape or not np.all(np.isfinite(f_vals)):
            raise ValueError("rhs must return finite values on the grid")
        # f expressed in the C^(k) basis, truncated to the interior rows
        f_ck = np.asarray(conversion_chain(n, 0, k) @ chebyshev_coefficients(f_vals))
        vec = np.concatenate([[bc.value for bc in self._bcs], f_ck[: n - k]])
        return np.asarray(self._lu.solve(vec))

    def solve_values(self, rhs: _ScalarField, points: ArrayLike) -> _FloatArray:
        """Convenience: solve and evaluate at physical points."""
        coeffs = self.solve(rhs)
        a, b = self._op.interval
        xi = 2.0 * (np.asarray(points, dtype=np.float64) - a) / (b - a) - 1.0
        return chebyshev_values(coeffs, xi)
