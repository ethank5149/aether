"""Thermosphere and exosphere: NRLMSIS, with the solar inputs made explicit.

Above 86 km the 1976 standard abandons a well-mixed gas and integrates each
species under diffusive equilibrium, which makes it a model of *one*
atmosphere — the one with an exospheric temperature of 1000 K, corresponding
to high solar activity. Real upper-atmosphere density at 300 km swings by
more than an order of magnitude between solar minimum and maximum, and any
trajectory with a long low-orbit or upper-mesospheric leg is sensitive to
exactly that.

So the upper model here is NRLMSIS, through :mod:`pymsis`, which is a
compiled reference implementation rather than a transcription. What this
module adds is the discipline around it:

* **The solar and geomagnetic inputs are arguments with stated defaults, not
  a download.** Called without indices, ``pymsis`` fetches the historical
  F10.7 and ap series over the network. That makes a result depend on the
  day it was computed and on having a network, neither of which belongs in a
  simulation whose whole point is reproducibility. Every call here passes
  indices explicitly, so nothing is ever fetched.
* **The defaults are a design condition, not a date.** F10.7 = 150 with
  ap = 4 is the moderate-activity case; :data:`SOLAR_MINIMUM` and
  :data:`SOLAR_MAXIMUM` bracket it. Which one to fly is an engineering
  choice and it is made in the open.

The seam with the lower model is real and is not hidden: at 86 km
NRLMSISE-00 gives 7.06e-6 kg/m³ against the standard's 6.958e-6, a 1.5 %
step, and MSIS 2.1 gives 6.40e-6, an 8 % step the other way.
:class:`~aether.atmosphere.model.LayeredAtmosphere` blends across it rather
than jumping, because a discontinuity in density is a discontinuity in drag
and an adaptive integrator will chase it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aether.atmosphere.standard import AVOGADRO, BOLTZMANN, AtmosphereState

__all__ = [
    "MODERATE_ACTIVITY",
    "SOLAR_MAXIMUM",
    "SOLAR_MINIMUM",
    "MSISAtmosphere",
    "SolarActivity",
]

_FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SolarActivity:
    """The solar and geomagnetic inputs MSIS needs, as a named condition.

    Attributes
    ----------
    f107:
        Solar 10.7 cm radio flux on the previous day, in solar flux units.
    f107a:
        81-day centred average of the same.
    ap:
        Daily geomagnetic index. MSIS accepts a seven-element history; a
        single value is broadcast to all seven, which is the quiet-and-steady
        assumption and is what a design condition means.
    """

    name: str
    f107: float
    f107a: float
    ap: float

    def as_arrays(self) -> tuple[float, float, list[list[float]]]:
        return float(self.f107), float(self.f107a), [[float(self.ap)] * 7]


#: Deep solar minimum — the thin thermosphere. Long orbital lifetimes.
SOLAR_MINIMUM = SolarActivity("solar minimum", f107=70.0, f107a=70.0, ap=4.0)
#: The usual design condition, and this module's default.
MODERATE_ACTIVITY = SolarActivity("moderate activity", f107=150.0, f107a=150.0, ap=4.0)
#: Solar maximum — roughly an order of magnitude denser at 400 km.
SOLAR_MAXIMUM = SolarActivity("solar maximum", f107=250.0, f107a=250.0, ap=15.0)


@dataclass(frozen=True)
class MSISAtmosphere:
    """NRLMSIS as an :class:`~aether.atmosphere.standard.AtmosphereState` source.

    Attributes
    ----------
    activity:
        Solar and geomagnetic condition. Explicit, always.
    epoch:
        Date used for the seasonal and diurnal terms. A date is unavoidable —
        MSIS has an annual cycle — so it is pinned to an equinox by default,
        which is the mid-range condition rather than an extreme.
    latitude, longitude:
        Degrees. The default is the equator at local midnight-ish; MSIS's
        diurnal variation at 400 km is roughly a factor of two between the
        day and night bulges, so this is not a negligible choice either.
    version:
        ``0`` selects NRLMSISE-00, ``2.1`` the current MSIS. The default is
        NRLMSISE-00: it is the model most entry and lifetime work is written
        against, and it happens to match the 1976 standard within 1.5 % at
        the 86 km seam where MSIS 2.1 is 8 % off.
    """

    activity: SolarActivity = MODERATE_ACTIVITY
    epoch: str = "2000-03-21T12:00"
    latitude: float = 0.0
    longitude: float = 0.0
    version: float = 0.0
    name: str = field(default="", compare=False)

    #: Geometric altitude bounds MSIS is valid over (m).
    floor: float = 0.0
    ceiling: float = 1000.0e3

    def __post_init__(self) -> None:
        if self.version not in (0.0, 2.0, 2.1):
            msg = f"MSIS version must be 0 (NRLMSISE-00), 2.0 or 2.1, got {self.version}"
            raise ValueError(msg)
        label = "NRLMSISE-00" if self.version == 0.0 else f"MSIS {self.version:g}"
        object.__setattr__(self, "name", f"{label} ({self.activity.name})")

    def state(self, altitude: ArrayLike) -> AtmosphereState:
        """Full thermodynamic state at geometric altitude (m)."""
        z = np.atleast_1d(np.asarray(altitude, dtype=np.float64))
        if np.any(~np.isfinite(z)):
            msg = "altitude must be finite"
            raise ValueError(msg)
        if np.any(z < self.floor) or np.any(z > self.ceiling):
            msg = (
                f"{self.name} is defined from {self.floor / 1e3:g} to "
                f"{self.ceiling / 1e3:g} km; asked for "
                f"[{float(np.min(z)) / 1e3:g}, {float(np.max(z)) / 1e3:g}] km"
            )
            raise ValueError(msg)

        f107, f107a, aps = self.activity.as_arrays()
        raw = self._calculate(z / 1.0e3, f107, f107a, aps)

        density = np.asarray(raw[..., 0], dtype=np.float64)
        temperature = np.asarray(raw[..., 10], dtype=np.float64)
        # Mean molar mass from the species mix. Anomalous oxygen (index 8) is
        # a hot, non-thermalised population that MSIS carries for satellite
        # drag bookkeeping; it does not belong in a mean molecular weight and
        # is left out.
        species = np.asarray(raw[..., [1, 2, 3, 4, 5, 6, 7, 9]], dtype=np.float64)
        total_number = np.nansum(species, axis=-1)
        molar_mass = density * AVOGADRO / total_number

        # Pressure from the species count and the *same* Boltzmann constant
        # the lower model uses, so that p = rho R T holds identically on both
        # sides of the seam and the blend between them inherits it.
        pressure = total_number * BOLTZMANN * temperature

        shape = np.shape(altitude)

        def restore(values: NDArray[np.float64]) -> _FloatArray:
            return np.asarray(values.reshape(shape), dtype=np.float64)

        return AtmosphereState(
            altitude=restore(z),
            temperature=restore(temperature),
            pressure=restore(np.asarray(pressure)),
            density=restore(density),
            molar_mass=restore(molar_mass),
        )

    def _calculate(
        self, altitude_km: _FloatArray, f107: float, f107a: float, aps: list[list[float]]
    ) -> NDArray[np.float64]:
        try:
            import pymsis
        except ImportError as error:  # pragma: no cover - dependency is declared
            msg = (
                "the upper atmosphere needs pymsis (pip install pymsis). The "
                "1976 standard's own upper model is a species-wise diffusive "
                "equilibrium that this package does not reimplement, because a "
                "compiled reference model is checkable and a transcription of "
                "one is not."
            )
            raise ImportError(msg) from error
        result: Any = pymsis.calculate(
            np.datetime64(self.epoch),
            self.longitude,
            self.latitude,
            altitude_km,
            f107,
            f107a,
            aps,
            # pymsis matches the version against a tuple containing the
            # *integer* 0, and 0.0 == 0 is not enough for its membership test.
            version=0 if self.version == 0.0 else self.version,
        )
        # pymsis returns (n_dates, n_lons, n_lats, n_alts, n_variables) when
        # it builds a grid, but drops to (n_points, n_variables) when every
        # input has the same length — which happens whenever a single
        # altitude is queried, since the date, longitude and latitude are all
        # scalars too. Reshaping to (-1, n_variables) is correct in both
        # cases; indexing the leading axes is correct in only one, and the
        # one it is wrong for is the one that appears the moment a blend band
        # contains exactly one sample.
        variables = np.asarray(result, dtype=np.float64)
        return variables.reshape(-1, variables.shape[-1])
