"""Chebyshev–Gauss–Lobatto collocation operators.

Implements Paper I, Appendix A, including both accuracy remedies stated
there:

1. every derivative matrix is constructed *directly* from the
   Weideman–Reddy recurrence (Weideman & Reddy, ACM TOMS 26(4), 2000)
   rather than by repeated matrix multiplication, so rounding error is
   not amplified by :math:`\\kappa(\\mathbf{D})` at each power; and
2. diagonal entries are computed by the negative-sum trick,
   :math:`D_{jj} = -\\sum_{i \\ne j} D_{ji}`, which enforces the exact
   annihilation of constants.

Node differences are evaluated with the trigonometric identity
:math:`\\xi_i - \\xi_j = 2 \\sin\\tfrac{(i+j)\\pi}{2N} \\sin\\tfrac{(j-i)\\pi}{2N}`
and the "flipping" symmetry trick, both of which avoid the cancellation
incurred by forming :math:`\\cos\\theta_i - \\cos\\theta_j` directly near
the domain endpoints.

The node convention follows the papers: :math:`\\xi_j = \\cos(j\\pi/N)`,
*descending* from :math:`+1` at :math:`j = 0` to :math:`-1` at
:math:`j = N`. Under the affine map of Paper I, Eq. (3.3),
:math:`x = \\tfrac{L}{2}(\\xi + 1)`, index 0 is therefore the :math:`x = L`
end and index :math:`N` the :math:`x = 0` end.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "ChebyshevGrid",
    "barycentric_interpolate",
    "barycentric_weights",
    "chebyshev_diffmats",
    "clenshaw_curtis_weights",
    "gauss_lobatto_nodes",
]

_FloatArray = NDArray[np.float64]


def _validate_order(n: int) -> None:
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool):
        raise TypeError(f"polynomial order n must be an integer, got {type(n).__name__}")
    if n < 1:
        raise ValueError(f"polynomial order n must be >= 1, got {n}")


def gauss_lobatto_nodes(n: int) -> _FloatArray:
    """Chebyshev–Gauss–Lobatto nodes :math:`\\xi_j = \\cos(j\\pi/N)`, descending.

    Computed as :math:`\\sin(\\pi(N - 2j)/(2N))`, which is algebraically
    identical but exactly antisymmetric about the domain midpoint in
    floating point, so that :math:`\\xi_j = -\\xi_{N-j}` holds to the bit.

    Parameters
    ----------
    n:
        Polynomial order :math:`N`; the grid has :math:`N + 1` nodes.
    """
    _validate_order(n)
    j = np.arange(n + 1)
    nodes = np.sin(np.pi * (n - 2.0 * j) / (2.0 * n))
    nodes.flags.writeable = False
    return nodes


def chebyshev_diffmats(n: int, max_order: int) -> _FloatArray:
    """Differentiation matrices :math:`\\mathbf{D}^{(1)}, \\dots, \\mathbf{D}^{(M)}`.

    Direct construction on the descending CGL nodes via the
    Weideman–Reddy recurrence with negative-sum diagonals; see the module
    docstring for why this is used instead of matrix powers.

    Parameters
    ----------
    n:
        Polynomial order :math:`N`.
    max_order:
        Highest derivative :math:`M` to construct, :math:`1 \\le M \\le N`.

    Returns
    -------
    numpy.ndarray
        Read-only array of shape ``(max_order, n + 1, n + 1)``;
        ``result[k - 1]`` represents :math:`d^k/d\\xi^k` on :math:`[-1, 1]`.
    """
    _validate_order(n)
    if not 1 <= max_order <= n:
        raise ValueError(
            f"max_order must satisfy 1 <= max_order <= n = {n}, got {max_order}: "
            f"derivatives above order N annihilate the collocation space"
        )

    m = n + 1
    k = np.arange(m)
    theta_half = (k * np.pi / n) / 2.0

    # Node differences xi_i - xi_j by the product-of-sines identity
    # (exact antisymmetry, no endpoint cancellation), with the flipping
    # trick: the lower half of the matrix is the rotated negation of the
    # upper half, so both halves inherit the accuracy of the
    # better-conditioned upper-half evaluation.
    t_col = theta_half[:, np.newaxis]
    t_row = theta_half[np.newaxis, :]
    dxi = 2.0 * np.sin(t_row + t_col) * np.sin(t_row - t_col)
    n_lower = m // 2  # bottom-half rows, exactly the rotated negation of the top half
    dxi[m - n_lower :, :] = -dxi[:n_lower, :][::-1, ::-1]
    np.fill_diagonal(dxi, 1.0)  # placeholder: reciprocal taken next, diagonal unused

    # C[i, j] = c_i (-1)^{i+j} / c_j with c_0 = c_N = 2, else 1.
    sign = np.where((k[:, np.newaxis] + k[np.newaxis, :]) % 2 == 0, 1.0, -1.0)
    c = np.ones(m)
    c[0] = c[-1] = 2.0
    coeff = sign * (c[:, np.newaxis] / c[np.newaxis, :])

    inv_dxi = 1.0 / dxi
    np.fill_diagonal(inv_dxi, 0.0)

    diffmats = np.empty((max_order, m, m))
    d = np.eye(m)
    for order in range(1, max_order + 1):
        d = order * inv_dxi * (coeff * np.diag(d)[:, np.newaxis] - d)
        np.fill_diagonal(d, 0.0)
        np.fill_diagonal(d, -d.sum(axis=1))  # negative-sum trick
        diffmats[order - 1] = d

    diffmats.flags.writeable = False
    return diffmats


def clenshaw_curtis_weights(n: int) -> _FloatArray:
    """Clenshaw–Curtis quadrature weights on the CGL nodes (reference domain).

    Exact for polynomials of degree :math:`\\le N`; the weights are the
    integrals over :math:`[-1, 1]` of the Lagrange cardinal functions of
    the grid. These are the :math:`w_i^{\\mathrm{CC}}` of Paper I,
    Eq. (3.13). Ordering matches :func:`gauss_lobatto_nodes` (the weights
    are symmetric, so ordering is observable only through indexing).
    """
    _validate_order(n)
    j = np.arange(1, n)
    theta = j * np.pi / n
    interior = np.ones(n - 1)
    if n % 2 == 0:
        for kk in range(1, n // 2):
            interior -= 2.0 * np.cos(2.0 * kk * theta) / (4.0 * kk**2 - 1.0)
        interior -= np.cos(n * theta) / (n**2 - 1.0)
        endpoint = 1.0 / (n**2 - 1.0)
    else:
        for kk in range(1, (n - 1) // 2 + 1):
            interior -= 2.0 * np.cos(2.0 * kk * theta) / (4.0 * kk**2 - 1.0)
        endpoint = 1.0 / n**2
    weights = np.empty(n + 1)
    weights[0] = weights[-1] = endpoint
    weights[1:-1] = 2.0 * interior / n
    weights.flags.writeable = False
    return weights


def barycentric_weights(n: int) -> _FloatArray:
    """Barycentric interpolation weights for the CGL grid.

    :math:`\\lambda_j = (-1)^j \\delta_j` with :math:`\\delta_0 =
    \\delta_N = 1/2`, :math:`\\delta_j = 1` otherwise (Berrut &
    Trefethen, SIAM Rev. 46(3), 2004). Any common scaling cancels in the
    barycentric formula.
    """
    _validate_order(n)
    lam = np.where(np.arange(n + 1) % 2 == 0, 1.0, -1.0)
    lam[0] *= 0.5
    lam[-1] *= 0.5
    lam.flags.writeable = False
    return lam


def barycentric_interpolate(
    nodes: ArrayLike,
    values: ArrayLike,
    x_eval: ArrayLike,
    weights: ArrayLike | None = None,
) -> _FloatArray:
    """Evaluate the interpolant of nodal ``values`` at ``x_eval``.

    Uses the second (true) barycentric formula, which is backward stable
    on Chebyshev grids. Evaluation points that coincide exactly with a
    node return the nodal value, avoiding the 0/0 in the formula.

    Parameters
    ----------
    nodes:
        Interpolation nodes, shape ``(n + 1,)``.
    values:
        Nodal values, shape ``(n + 1,)`` or ``(n + 1, m)`` for ``m``
        stacked fields.
    x_eval:
        Evaluation points, any shape.
    weights:
        Barycentric weights for ``nodes``. Defaults to the CGL weights
        of :func:`barycentric_weights`, which are correct only if
        ``nodes`` is a (possibly affinely mapped) CGL grid.
    """
    nodes_arr = np.asarray(nodes, dtype=np.float64)
    values_arr = np.asarray(values, dtype=np.float64)
    x_arr = np.asarray(x_eval, dtype=np.float64)
    if nodes_arr.ndim != 1 or nodes_arr.size < 2:
        raise ValueError(f"nodes must be a 1-D array of >= 2 points, got shape {nodes_arr.shape}")
    if values_arr.shape[0] != nodes_arr.size:
        raise ValueError(
            f"values first dimension {values_arr.shape[0]} does not match "
            f"{nodes_arr.size} nodes"
        )
    lam = (
        np.asarray(weights, dtype=np.float64)
        if weights is not None
        else barycentric_weights(nodes_arr.size - 1)
    )
    if lam.shape != nodes_arr.shape:
        raise ValueError(f"weights shape {lam.shape} does not match nodes shape {nodes_arr.shape}")

    flat = x_arr.reshape(-1)
    diff = flat[:, np.newaxis] - nodes_arr[np.newaxis, :]
    exact_pt, exact_node = np.nonzero(diff == 0.0)
    diff[exact_pt, exact_node] = 1.0  # silenced below by direct assignment
    ratio = lam[np.newaxis, :] / diff
    denom = ratio.sum(axis=1)
    result = (ratio @ values_arr) / denom.reshape(denom.shape + (1,) * (values_arr.ndim - 1))
    if exact_pt.size:
        result[exact_pt] = values_arr[exact_node]
    return result.reshape(x_arr.shape + values_arr.shape[1:])


class ChebyshevGrid:
    """CGL collocation grid on an interval, with physically scaled operators.

    Encapsulates the affine map of Paper I, Eq. (3.3): a grid of order
    :math:`N` on :math:`\\xi \\in [-1, 1]` mapped to
    :math:`x \\in [a, b]`, with every derivative matrix carrying its
    :math:`(2/(b-a))^k` factor explicitly — the factor Paper I flags as
    "a common source of error" — and quadrature weights carrying
    :math:`(b-a)/2`.

    All exposed arrays are read-only views of internally owned storage.

    Parameters
    ----------
    n:
        Polynomial order :math:`N`; the grid has :math:`N + 1` nodes.
    interval:
        Physical interval :math:`(a, b)` with :math:`a < b`. Defaults to
        the reference domain :math:`(-1, 1)`.
    max_derivative:
        Highest derivative operator to construct (4 for the
        Euler–Bernoulli operator).
    """

    def __init__(
        self,
        n: int,
        interval: tuple[float, float] = (-1.0, 1.0),
        max_derivative: int = 4,
    ) -> None:
        _validate_order(n)
        a, b = float(interval[0]), float(interval[1])
        if not (np.isfinite(a) and np.isfinite(b)):
            raise ValueError(f"interval endpoints must be finite, got {interval}")
        if not a < b:
            raise ValueError(f"interval must satisfy a < b, got {interval}")

        self._n = int(n)
        self._interval = (a, b)
        self._xi = gauss_lobatto_nodes(n)
        self._ref_diffmats = chebyshev_diffmats(n, max_derivative)
        self._ref_weights = clenshaw_curtis_weights(n)

        half_len = (b - a) / 2.0
        x = a + half_len * (self._xi + 1.0)
        x.flags.writeable = False
        self._x = x

        scale = 1.0 / half_len  # d/dx = (2/(b-a)) d/dxi
        phys = np.array(
            [scale ** (k + 1) * self._ref_diffmats[k] for k in range(max_derivative)]
        )
        phys.flags.writeable = False
        self._diffmats = phys

        w = half_len * self._ref_weights
        w.flags.writeable = False
        self._weights = w

    @property
    def n(self) -> int:
        """Polynomial order :math:`N` (the grid has :math:`N + 1` nodes)."""
        return self._n

    @property
    def size(self) -> int:
        """Number of nodes, :math:`N + 1`."""
        return self._n + 1

    @property
    def interval(self) -> tuple[float, float]:
        """Physical interval :math:`(a, b)`."""
        return self._interval

    @property
    def length(self) -> float:
        """Interval length :math:`b - a`."""
        return self._interval[1] - self._interval[0]

    @property
    def xi(self) -> _FloatArray:
        """Reference nodes on :math:`[-1, 1]`, descending."""
        return self._xi

    @property
    def x(self) -> _FloatArray:
        """Physical nodes on :math:`[a, b]`; ``x[0] = b``, ``x[-1] = a``."""
        return self._x

    @property
    def weights(self) -> _FloatArray:
        """Clenshaw–Curtis weights on the physical interval."""
        return self._weights

    @property
    def max_derivative(self) -> int:
        """Highest derivative operator constructed."""
        return int(self._diffmats.shape[0])

    def diffmat(self, order: int) -> _FloatArray:
        """Physically scaled derivative matrix :math:`d^k/dx^k`.

        Parameters
        ----------
        order:
            Derivative order :math:`k`, ``1 <= k <= max_derivative``.
        """
        if not 1 <= order <= self.max_derivative:
            raise ValueError(
                f"derivative order must be in [1, {self.max_derivative}], got {order}"
            )
        return cast(_FloatArray, self._diffmats[order - 1])

    def __repr__(self) -> str:
        a, b = self._interval
        return f"ChebyshevGrid(n={self._n}, interval=({a:g}, {b:g}))"
