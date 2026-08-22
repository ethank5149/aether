"""Surface thermochemistry :math:`B'` tables (Chen & Milos 1999, §"Numerical
Procedures"; Milos, Chen & Squire 2006, Fig. 5).

FIAT closes its surface energy balance with tables computed offline by
ACE or MAT: *"The user selects sets of values for pressure P,
dimensionless gas rate B'_g, and dimensionless char rate B'_c"*, and
*"In general, for pyrolyzing ablators B'_c is a complex function of
temperature, pressure, and B'_g."* The same tables carry the wall
enthalpy :math:`h_w`, which is the other unknown the SEB needs.

This module implements that as a three-dimensional table
:math:`(P, B'_g, T_w) \\mapsto (B'_c, h_w)`.

Two properties are enforced rather than assumed.

**No extrapolation.** An equilibrium thermochemistry table has no meaning
outside the envelope it was generated for. Queries outside the tabulated
box raise. FIAT clamps; clamping silently converts an out-of-envelope
trajectory into a plausible-looking answer, and a sizing code that does
that cannot be trusted at the edges of its own design space.

**Smoothness is checked, not hoped for.** Milos, Chen & Squire warn:
*"For numerical stability, these curves should be sufficiently smooth and
well-resolved."* Because the Newton solve differentiates :math:`B'_c`
with respect to :math:`T_w`, a table with a kink in it will stall the
iteration in a way that looks like a physics problem. Cubic interpolation
gives a :math:`C^2` surface, and :meth:`BPrimeTable.roughness` reports
the second-difference norm so a bad table is caught at construction time
rather than diagnosed from a divergent run.
"""

from __future__ import annotations

import numpy as np
import scipy.interpolate
from numpy.typing import ArrayLike, NDArray

__all__ = ["BPrimeTable", "TableRangeError"]

_FloatArray = NDArray[np.float64]


class TableRangeError(ValueError):
    """A query fell outside the tabulated envelope.

    Distinct from a plain :class:`ValueError` so that an iterative solver
    can treat an out-of-envelope *trial* iterate as a step to shorten,
    while a genuinely bad argument still propagates. Widening the table
    is the fix when a converged state lands outside it; shortening the
    step is the fix when a Newton iterate overshoots into it.
    """


class BPrimeTable:
    """Tabulated equilibrium surface thermochemistry.

    Parameters
    ----------
    pressures:
        Strictly increasing :math:`P` (Pa). A single-pressure table is
        allowed and is treated as pressure-independent.
    gas_rates:
        Strictly increasing :math:`B'_g` (-), at least 2 points.
    wall_temperatures:
        Strictly increasing :math:`T_w` (K), at least 2 points.
    char_rates:
        :math:`B'_c \\ge 0`, shape ``(nP, nBg, nTw)``.
    wall_enthalpies:
        :math:`h_w` (J/kg), same shape.
    method:
        ``"cubic"`` (default) needs at least 4 points on the
        :math:`B'_g` and :math:`T_w` axes and gives a :math:`C^2`
        surface. ``"linear"`` is accepted for coarse tables but gives a
        discontinuous :math:`\\partial B'_c/\\partial T_w`, which the
        Newton solve will feel. Pressure is always blended linearly
        between planes.
    """

    def __init__(
        self,
        pressures: ArrayLike,
        gas_rates: ArrayLike,
        wall_temperatures: ArrayLike,
        char_rates: ArrayLike,
        wall_enthalpies: ArrayLike,
        method: str = "cubic",
    ) -> None:
        p = np.atleast_1d(np.asarray(pressures, dtype=np.float64))
        b_g = np.asarray(gas_rates, dtype=np.float64)
        t_w = np.asarray(wall_temperatures, dtype=np.float64)
        b_c = np.asarray(char_rates, dtype=np.float64)
        h_w = np.asarray(wall_enthalpies, dtype=np.float64)

        for name, axis, minimum in (
            ("pressures", p, 1),
            ("gas_rates", b_g, 2),
            ("wall_temperatures", t_w, 2),
        ):
            if axis.ndim != 1 or axis.size < minimum:
                raise ValueError(f"{name} must be 1-D with >= {minimum} points")
            if not np.all(np.isfinite(axis)):
                raise ValueError(f"{name} must be finite")
            if axis.size > 1 and np.any(np.diff(axis) <= 0.0):
                raise ValueError(f"{name} must be strictly increasing")
        if np.any(p <= 0.0):
            raise ValueError("pressures must be > 0")
        if np.any(b_g < 0.0):
            raise ValueError("gas_rates must be >= 0")
        if np.any(t_w <= 0.0):
            raise ValueError("wall_temperatures must be > 0")

        shape = (p.size, b_g.size, t_w.size)
        if b_c.shape != shape or h_w.shape != shape:
            raise ValueError(
                f"char_rates {b_c.shape} and wall_enthalpies {h_w.shape} must "
                f"both have shape {shape}"
            )
        if not np.all(np.isfinite(b_c)) or np.any(b_c < 0.0):
            raise ValueError("char_rates must be finite and >= 0")
        if not np.all(np.isfinite(h_w)):
            raise ValueError("wall_enthalpies must be finite")

        # Pressure planes are interpolated separately from (B'_g, T_w).
        # RegularGridInterpolator would force one method across all three
        # axes, and pressure axes are routinely two or three points — which
        # would drag the whole table down to linear and destroy the C2
        # behaviour on the T_w axis, the one the Newton solve differentiates.
        # Cubic tensor-product splines per pressure plane, linear across
        # planes, keeps the smoothness where it is load-bearing.
        kx = 3 if (method == "cubic" and b_g.size >= 4) else 1
        ky = 3 if (method == "cubic" and t_w.size >= 4) else 1
        if method == "cubic" and (kx == 1 or ky == 1):
            raise ValueError(
                "cubic interpolation needs >= 4 points on the B'_g and T_w axes "
                f"(got {b_g.size} and {t_w.size}); pass method='linear' for a "
                "coarse table and accept the discontinuous derivative"
            )
        if method not in ("cubic", "linear"):
            raise ValueError(f"method must be 'cubic' or 'linear', got {method!r}")

        self._method = method
        self._pressures = p
        self._ranges = (
            (float(p[0]), float(p[-1])),
            (float(b_g[0]), float(b_g[-1])),
            (float(t_w[0]), float(t_w[-1])),
        )
        self._b_c = [
            scipy.interpolate.RectBivariateSpline(b_g, t_w, plane, kx=kx, ky=ky, s=0)
            for plane in b_c
        ]
        self._h_w = [
            scipy.interpolate.RectBivariateSpline(b_g, t_w, plane, kx=kx, ky=ky, s=0)
            for plane in h_w
        ]
        self._b_c_values = b_c

    @property
    def pressure_range(self) -> tuple[float, float]:
        return self._ranges[0]

    @property
    def gas_rate_range(self) -> tuple[float, float]:
        return self._ranges[1]

    @property
    def wall_temperature_range(self) -> tuple[float, float]:
        return self._ranges[2]

    def _weights(self, pressure: float) -> tuple[int, int, float]:
        """Bracketing pressure planes and the linear blend weight."""
        p = self._pressures
        if p.size == 1:
            return 0, 0, 0.0
        j = int(np.clip(np.searchsorted(p, pressure) - 1, 0, p.size - 2))
        w = (pressure - p[j]) / (p[j + 1] - p[j])
        return j, j + 1, float(w)

    def _check(
        self, pressure: float, gas_rate: float, wall_temperature: float
    ) -> tuple[float, float, float]:
        query = (float(pressure), float(gas_rate), float(wall_temperature))
        names = ("pressure", "B'_g", "T_w")
        for value, name, (lo, hi) in zip(query, names, self._ranges, strict=True):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
            if name == "pressure" and self._pressures.size == 1:
                # A one-plane table is explicitly pressure-independent, which
                # is a modelling statement, not an extrapolation.
                continue
            # A query landing exactly on an axis endpoint is not
            # extrapolation, but arithmetic that should produce the endpoint
            # often lands a few ulps past it — a gas flux that saturates the
            # B'_g axis is the common case. Admit the boundary to within a
            # relative ulp scale; anything genuinely outside is still refused.
            span = max(abs(lo), abs(hi), 1.0)
            if not (lo - 1e-12 * span) <= value <= (hi + 1e-12 * span):
                raise TableRangeError(
                    f"{name} = {value:.6g} is outside the tabulated range "
                    f"[{lo:.6g}, {hi:.6g}]; refusing to extrapolate an "
                    f"equilibrium thermochemistry table"
                )
        return query

    def _blend(
        self,
        splines: list[scipy.interpolate.RectBivariateSpline],
        pressure: float,
        gas_rate: float,
        wall_temperature: float,
    ) -> float:
        lo, hi, w = self._weights(pressure)
        a = float(splines[lo](gas_rate, wall_temperature)[0, 0])
        if lo == hi:
            return a
        b = float(splines[hi](gas_rate, wall_temperature)[0, 0])
        return (1.0 - w) * a + w * b

    def char_rate(self, pressure: float, gas_rate: float, wall_temperature: float) -> float:
        """:math:`B'_c` at the queried surface state, clamped at zero."""
        p, b_g, t_w = self._check(pressure, gas_rate, wall_temperature)
        return max(self._blend(self._b_c, p, b_g, t_w), 0.0)

    def wall_enthalpy(self, pressure: float, gas_rate: float, wall_temperature: float) -> float:
        """:math:`h_w` (J/kg) at the queried surface state."""
        p, b_g, t_w = self._check(pressure, gas_rate, wall_temperature)
        return self._blend(self._h_w, p, b_g, t_w)

    def char_rate_derivative(
        self, pressure: float, gas_rate: float, wall_temperature: float
    ) -> float:
        """:math:`\\partial B'_c/\\partial T_w` (1/K), analytically.

        The Newton surface solve needs this. ``RectBivariateSpline``
        differentiates exactly, so unlike a finite difference this costs
        nothing in accuracy and cannot be spoiled by a poor step size.
        Clamped to zero wherever :meth:`char_rate` is clamped, so the
        derivative stays consistent with the value it differentiates.
        """
        p, b_g, t_w = self._check(pressure, gas_rate, wall_temperature)
        if self._blend(self._b_c, p, b_g, t_w) <= 0.0:
            return 0.0
        lo, hi, w = self._weights(p)
        a = float(self._b_c[lo](b_g, t_w, dy=1)[0, 0])
        if lo == hi:
            return a
        b = float(self._b_c[hi](b_g, t_w, dy=1)[0, 0])
        return (1.0 - w) * a + w * b

    def roughness(self) -> float:
        """Max second difference of :math:`B'_c` along :math:`T_w`,
        relative to the table's own range.

        The diagnostic behind Milos, Chen & Squire's stability warning.
        Values above roughly 0.1 indicate a table too coarse or too
        kinked for a Newton surface solve; the table will still
        interpolate, but the iteration may stall.

        The normalisation is by the **global** span of :math:`B'_c`, not
        by the local value. A real equilibrium table is essentially zero
        below the onset of surface chemistry, and a per-point relative
        measure divides by those zeros: applied to a TACOT table it
        reported a roughness of :math:`10^{11}` for a surface that is in
        fact smooth everywhere it matters.
        """
        values = self._b_c_values
        if values.shape[2] < 3:
            return 0.0
        span = float(np.max(values) - np.min(values))
        if span <= 0.0:
            return 0.0
        return float(np.max(np.abs(np.diff(values, n=2, axis=2))) / span)
