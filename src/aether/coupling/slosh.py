"""Quadrature-consistent regularization of localized slosh forces.

Paper I, §3.3: a point load on a spectral grid is a Dirac delta whose
Chebyshev coefficients do not decay, producing Gibbs oscillations across
the entire domain. The load is regularized by a Gaussian kernel, with
two details that determine admissibility:

**Bandwidth** (Paper I, "Bandwidth" paragraph). CGL nodes are not
uniformly spaced — local spacing is :math:`h(\\xi) \\approx
\\pi\\sqrt{1 - \\xi^2}/N`, which is :math:`\\mathcal{O}(N^{-1})` at the
center and :math:`\\mathcal{O}(N^{-2})` at the ends — so a single global
:math:`\\sigma` is either under-resolved at the center or wastefully
wide at the ends. The bandwidth is chosen per station as
:math:`\\sigma^{(k)} = \\gamma\\,h(x_s^{(k)})` with
:math:`\\gamma \\in [1, 2]`.

**Exact force transfer** (Paper I, Eq. 3.13 and Prop. 1). The Gaussian
integrates to unity over :math:`\\mathbb{R}`, not over the truncated
domain, and the discrete system sees the quadrature sum, not the
integral. The kernel is therefore normalized *discretely* against the
Clenshaw–Curtis weights,

.. math::

    \\tilde{\\delta}_\\sigma(x_j - x_s) =
    \\frac{\\delta_\\sigma(x_j - x_s)}
         {\\sum_i w_i^{\\mathrm{CC}}\\,\\delta_\\sigma(x_i - x_s)},

so the total transverse force delivered to the discrete structure equals
:math:`\\sum_k F^{(k)}` exactly — independent of :math:`\\sigma`,
:math:`N`, and the station. Only the zeroth moment is exact: the first
moment (applied bending moment) is transferred to
:math:`\\mathcal{O}(\\sigma^2)` in the domain interior and degrades for
stations within :math:`\\sim 2\\sigma` of an endpoint, where the kernel
is truncated asymmetrically. Quantifying that is verification task V2.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aether.spectral import ChebyshevGrid

__all__ = [
    "SloshCoupling",
    "kernel_bandwidth",
    "local_node_spacing",
    "normalized_kernel",
]

_FloatArray = NDArray[np.float64]


def local_node_spacing(grid: ChebyshevGrid, station: ArrayLike) -> _FloatArray:
    """Local CGL node spacing :math:`h(x)` at physical station(s).

    Evaluates the paper's estimate :math:`h(\\xi) = \\pi\\sqrt{1 -
    \\xi^2}/N` mapped to the physical interval, floored at the exact
    endpoint gap :math:`2\\sin^2(\\pi/2N)` (the estimate vanishes at
    :math:`\\xi = \\pm 1` while the true spacing does not).
    """
    a, b = grid.interval
    x = np.asarray(station, dtype=np.float64)
    if np.any(x < a) or np.any(x > b):
        raise ValueError(f"station {np.array2string(x)} outside grid interval [{a}, {b}]")
    xi = 2.0 * (x - a) / (b - a) - 1.0
    xi = np.clip(xi, -1.0, 1.0)  # guard rounding at the endpoints
    h_ref = np.pi * np.sqrt(1.0 - xi * xi) / grid.n
    end_gap = 2.0 * np.sin(np.pi / (2.0 * grid.n)) ** 2  # exact 1 - cos(pi/N)
    return np.asarray((b - a) / 2.0 * np.maximum(h_ref, end_gap))


def kernel_bandwidth(grid: ChebyshevGrid, station: ArrayLike, gamma: float = 1.5) -> _FloatArray:
    """Station-adapted bandwidth :math:`\\sigma = \\gamma\\,h(x_s)`.

    Paper I prescribes :math:`\\gamma \\in [1, 2]`: below 1 the kernel
    is unresolved and reintroduces the oscillations it exists to
    suppress; large values over-smear the load and artificially stiffen
    the response. Values outside that band are rejected rather than
    silently accepted.
    """
    if not (np.isfinite(gamma) and 1.0 <= gamma <= 2.0):
        raise ValueError(
            f"gamma must lie in the resolvability band [1, 2] of Paper I §3.3, got {gamma}"
        )
    return gamma * local_node_spacing(grid, station)


def normalized_kernel(grid: ChebyshevGrid, station: float, sigma: float) -> _FloatArray:
    """Discretely normalized Gaussian kernel :math:`\\tilde{\\delta}_\\sigma`
    (Paper I, Eq. 3.13) sampled at the grid nodes.

    Satisfies :math:`\\sum_j w_j^{\\mathrm{CC}} \\tilde{\\delta}_j = 1`
    to rounding by construction. The (analytically cancelling) Gaussian
    prefactor is omitted so the exponentials are evaluated at their
    natural scale.
    """
    a, b = grid.interval
    x_s = float(station)
    if not a <= x_s <= b:
        raise ValueError(f"station {station} outside grid interval [{a}, {b}]")
    if not (np.isfinite(sigma) and sigma > 0.0):
        raise ValueError(f"sigma must be finite and > 0, got {sigma}")
    z = (grid.x - x_s) / sigma
    kernel = np.exp(-0.5 * z * z)
    denom = float(grid.weights @ kernel)
    if denom <= 0.0 or not np.isfinite(denom):  # pragma: no cover - defensive
        raise FloatingPointError(f"kernel normalization degenerate at station {x_s}, sigma {sigma}")
    return kernel / denom


class SloshCoupling:
    """Regularized load operator for a set of tank stations.

    Precomputes one normalized kernel column per station; the nodal
    distributed load for instantaneous slosh forces
    :math:`F^{(k)}(t)` is then a single matrix–vector product,

    .. math::

        \\mathbf{q}_{\\text{slosh}} = \\tilde{\\mathbf{\\Delta}}\\,\\mathbf{F},

    and batched force histories (one column per Monte Carlo replicate)
    reuse the same operator — the stations move only when the tank
    drains, at which point :meth:`with_stations` rebuilds the columns.

    Parameters
    ----------
    grid:
        Structural Chebyshev grid (the beam grid).
    stations:
        Tank stations :math:`x_s^{(k)}`, strictly inside the domain.
    gamma:
        Bandwidth multiple in the paper's admissible band ``[1, 2]``.
    """

    def __init__(
        self,
        grid: ChebyshevGrid,
        stations: ArrayLike,
        gamma: float = 1.5,
    ) -> None:
        x_s = np.atleast_1d(np.asarray(stations, dtype=np.float64))
        if x_s.ndim != 1 or x_s.size == 0:
            raise ValueError(f"stations must be a non-empty 1-D array, got shape {x_s.shape}")
        a, b = grid.interval
        if np.any(x_s <= a) or np.any(x_s >= b):
            raise ValueError(f"tank stations must lie strictly inside ({a}, {b}), got {x_s}")
        self._grid = grid
        self._gamma = float(gamma)
        self._stations = x_s.copy()
        self._stations.flags.writeable = False
        self._sigma = kernel_bandwidth(grid, x_s, gamma)
        self._sigma.flags.writeable = False
        cols = [
            normalized_kernel(grid, float(s), float(sig))
            for s, sig in zip(x_s, self._sigma, strict=True)
        ]
        self._kernels = np.column_stack(cols)
        self._kernels.flags.writeable = False

    @property
    def grid(self) -> ChebyshevGrid:
        return self._grid

    @property
    def stations(self) -> _FloatArray:
        return self._stations

    @property
    def sigma(self) -> _FloatArray:
        """Per-station bandwidths actually used."""
        return self._sigma

    @property
    def kernels(self) -> _FloatArray:
        """Normalized kernel columns, shape ``(N + 1, n_tank)``."""
        return self._kernels

    @property
    def n_tanks(self) -> int:
        return int(self._stations.size)

    def with_stations(self, stations: ArrayLike) -> SloshCoupling:
        """New operator for updated tank stations (drain tracking)."""
        return SloshCoupling(self._grid, stations, self._gamma)

    def load(self, forces: ArrayLike) -> _FloatArray:
        """Nodal distributed load for slosh forces (Paper I, Eq. 3.12).

        Parameters
        ----------
        forces:
            Shape ``(n_tank,)`` or ``(n_tank, n_batch)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(N + 1,)`` or ``(N + 1, n_batch)``.
        """
        f = np.asarray(forces, dtype=np.float64)
        if f.shape[0] != self.n_tanks or f.ndim not in (1, 2):
            raise ValueError(
                f"forces must have shape ({self.n_tanks},) or ({self.n_tanks}, n_batch), "
                f"got {f.shape}"
            )
        return self._kernels @ f

    def transferred_force(self, load: ArrayLike) -> _FloatArray:
        """Discrete total force :math:`\\sum_j w_j q_j` — equals
        :math:`\\sum_k F^{(k)}` exactly per Prop. 1."""
        return np.asarray(self._grid.weights @ np.asarray(load, dtype=np.float64))

    def transferred_moment(self, load: ArrayLike, x_ref: float) -> _FloatArray:
        """Discrete first moment about ``x_ref``,
        :math:`\\sum_j w_j (x_j - x_{\\mathrm{ref}}) q_j`.

        Exact only to :math:`\\mathcal{O}(\\sigma^2)` in the interior
        (Paper I, Remark after Prop. 1); V2 measures the constant and
        the endpoint degradation.
        """
        lever = self._grid.x - float(x_ref)
        return np.asarray((self._grid.weights * lever) @ np.asarray(load, dtype=np.float64))
