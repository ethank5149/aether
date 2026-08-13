"""Landau transformation to a fixed computational domain (Paper I, §3.4.2).

With recession depth :math:`s(t)`, the physical domain
:math:`y \\in [s(t), L_{\\mathrm{TPS}}]` shrinks as the surface recedes;
the coordinate

.. math::

    \\eta = \\frac{y - s(t)}{L_{\\mathrm{TPS}} - s(t)} \\in [0, 1]

renders the front stationary in computational coordinates for all
:math:`s(t) < L_{\\mathrm{TPS}}`. The time derivative at fixed
:math:`\\eta` acquires the grid-velocity term of Eq. (3.16),

.. math::

    \\partial_t|_y = \\partial_t|_\\eta
    - \\frac{\\dot s (1 - \\eta)}{L_{\\mathrm{TPS}} - s}\\,\\partial_\\eta ,

which is explicit, local, and cheap — the entire price of eliminating
front tracking. This module owns the transform so its signs and
degeneracy guard (:math:`s \\to L_{\\mathrm{TPS}}` is TPS burn-through
and must fail loudly) live in exactly one tested place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["LandauFrame"]

_FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LandauFrame:
    """The fixed-domain transform for one TPS stack of thickness
    ``total_thickness`` (m)."""

    total_thickness: float
    #: Fraction of the original thickness at which the solver refuses to
    #: continue; the transform degenerates as ell -> 0 and a burn-through
    #: must surface as a physics event, not a division overflow.
    min_thickness_fraction: float = 0.02

    def __post_init__(self) -> None:
        if not (np.isfinite(self.total_thickness) and self.total_thickness > 0.0):
            raise ValueError(
                f"total_thickness must be finite and > 0, got {self.total_thickness}"
            )
        if not 0.0 < self.min_thickness_fraction < 1.0:
            raise ValueError(
                f"min_thickness_fraction must be in (0, 1), got {self.min_thickness_fraction}"
            )

    def thickness(self, recession: float) -> float:
        """Remaining thickness :math:`\\ell = L_{\\mathrm{TPS}} - s`.

        Raises
        ------
        ValueError
            If the recession is negative or the remaining thickness is
            below the burn-through guard.
        """
        s = float(recession)
        if not np.isfinite(s) or s < 0.0:
            raise ValueError(f"recession depth must be finite and >= 0, got {s}")
        ell = self.total_thickness - s
        if ell <= self.min_thickness_fraction * self.total_thickness:
            raise ValueError(
                f"TPS burn-through: remaining thickness {ell:.4e} m is below the "
                f"{self.min_thickness_fraction:.0%} guard of "
                f"{self.total_thickness:.4e} m"
            )
        return ell

    def physical_coordinate(self, eta: ArrayLike, recession: float) -> _FloatArray:
        """Map :math:`\\eta \\mapsto y = s + \\eta\\,\\ell`."""
        ell = self.thickness(recession)
        return np.asarray(
            float(recession) + np.asarray(eta, dtype=np.float64) * ell
        )

    def grid_velocity_coefficient(
        self, eta: ArrayLike, recession: float, recession_rate: float
    ) -> _FloatArray:
        """The advection coefficient :math:`\\dot s (1 - \\eta)/\\ell` of
        Eq. (3.16): :math:`\\partial_t|_\\eta = \\partial_t|_y +
        \\dot s(1-\\eta)/\\ell\\,\\partial_\\eta` for a field carried by
        the material."""
        ell = self.thickness(recession)
        return np.asarray(
            float(recession_rate) * (1.0 - np.asarray(eta, dtype=np.float64)) / ell
        )
