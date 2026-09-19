"""Tauber–Sutton shock-layer radiative heating (Paper II, §4.2).

.. math::

    \\dot q_{s,\\mathrm{rad}} = C\\, R_{\\mathrm{eff}}^{\\,a}\\,
    \\rho_\\infty^{\\,b}\\, f(V_\\infty),

with :math:`a \\approx 1.0`, :math:`b \\approx 1.22` for Earth entry and
:math:`f` a **tabulated** velocity function rising effectively as
:math:`V^8` to :math:`V^{10}`. The velocity table is a *required input
with stated provenance* — the published Tauber–Sutton values must be
transcribed from the source (Tauber & Sutton, J. Spacecraft 28(1),
1991) by the user; this module deliberately ships no numbers claiming to
be theirs, consistent with the repository's citation-audit posture.

The :math:`R_{\\mathrm{eff}}` dependence is of *opposite sign* to the
convective correlation: blunting increases radiative heating while
reducing convective, so the optimal nose radius is a genuine trade
(Paper II, §4.2) — the II-V8 runner demonstrates the interior optimum.
"""

from __future__ import annotations

import numpy as np
import scipy.interpolate
from numpy.typing import ArrayLike, NDArray

__all__ = ["TauberSuttonRadiation"]

_FloatArray = NDArray[np.float64]


class TauberSuttonRadiation:
    """Radiative stagnation heating with a user-supplied velocity table.

    Parameters
    ----------
    velocities:
        Strictly increasing tabulated :math:`V_\\infty` grid (m/s), at
        least 4 points.
    velocity_function:
        Tabulated :math:`f(V_\\infty)` values, same length, positive and
        non-decreasing (the published function rises steeply and
        monotonically over its range).
    radius_exponent, density_exponent:
        :math:`a`, :math:`b`; Earth-entry defaults 1.0 and 1.22.
    coefficient:
        Leading constant :math:`C` consistent with the units of the
        supplied table.
    provenance:
        Free-text statement of where the table came from; required and
        recorded so a synthetic surrogate can never masquerade as the
        published data.
    """

    def __init__(
        self,
        velocities: ArrayLike,
        velocity_function: ArrayLike,
        coefficient: float,
        radius_exponent: float = 1.0,
        density_exponent: float = 1.22,
        provenance: str = "",
    ) -> None:
        v = np.asarray(velocities, dtype=np.float64)
        f = np.asarray(velocity_function, dtype=np.float64)
        if v.ndim != 1 or v.size < 4 or np.any(np.diff(v) <= 0.0):
            raise ValueError("velocities must be strictly increasing with >= 4 points")
        if f.shape != v.shape:
            raise ValueError(f"velocity_function shape {f.shape} does not match {v.shape}")
        if np.any(f <= 0.0) or not np.all(np.isfinite(f)):
            raise ValueError("velocity function values must be finite and > 0")
        if np.any(np.diff(f) < 0.0):
            raise ValueError("the tabulated velocity function must be non-decreasing")
        if not (np.isfinite(coefficient) and coefficient > 0.0):
            raise ValueError(f"coefficient must be finite and > 0, got {coefficient}")
        if not provenance.strip():
            raise ValueError(
                "a provenance statement for the velocity table is required; "
                "see the module docstring"
            )
        self._v_range = (float(v[0]), float(v[-1]))
        # monotone C1 interpolation: no overshoot between table points
        self._interp = scipy.interpolate.PchipInterpolator(v, f, extrapolate=False)
        self._coefficient = float(coefficient)
        self._a = float(radius_exponent)
        self._b = float(density_exponent)
        self._provenance = provenance

    @property
    def provenance(self) -> str:
        return self._provenance

    @property
    def velocity_range(self) -> tuple[float, float]:
        return self._v_range

    def heat_flux(
        self,
        effective_radius: ArrayLike,
        freestream_density: ArrayLike,
        freestream_velocity: ArrayLike,
    ) -> _FloatArray:
        """Radiative stagnation heat flux (Paper II, Eq. 4.4), W/m².

        Velocities outside the tabulated range raise — the correlation
        is a fit with no validity beyond its table.
        """
        r_eff = np.asarray(effective_radius, dtype=np.float64)
        rho = np.asarray(freestream_density, dtype=np.float64)
        v = np.asarray(freestream_velocity, dtype=np.float64)
        if np.any(r_eff <= 0.0) or np.any(rho <= 0.0):
            raise ValueError("effective_radius and freestream_density must be > 0")
        if np.any(v < self._v_range[0]) or np.any(v > self._v_range[1]):
            raise ValueError(
                f"velocity outside tabulated range {self._v_range}; refusing to "
                f"extrapolate the Tauber–Sutton velocity function"
            )
        f_v = self._interp(v)
        return np.asarray(self._coefficient * r_eff**self._a * rho**self._b * f_v)
