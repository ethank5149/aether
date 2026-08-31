"""US Standard Atmosphere 1976, 0 to 86 km, as defined rather than as fitted.

Every aerodynamic regime this codebase covers is a function of the gas the
vehicle is flying through, and up to now that gas has been
:class:`~aether.flight.ballistic_entry.ExponentialAtmosphere` — one density,
no temperature. That is enough for Allen–Eggers, where only :math:`\\rho`
appears, and it is not enough for anything else. Skin friction needs
Reynolds number, so it needs viscosity and therefore temperature. Rarefied
aerodynamics needs the mean free path, so it needs number density and a
collision cross-section. A Mach number needs a speed of sound.

So this is the standard itself: the seven geopotential layers of
NASA-TM-X-74335 with their defined lapse rates, the closed-form hydrostatic
pressure in each, and the molecular-weight correction between 80 and 86 km.
The layer base temperatures and pressures are **computed by the recursion**
at import rather than transcribed, because a transcribed constant is a
number nobody can check and a recursion is one that checks itself — the
published values then become a test rather than an input.

Two details that are easy to get wrong and are not cosmetic:

* The standard's gas constant is :math:`R^* = 8314.32`
  J kmol\\ :sup:`-1` K\\ :sup:`-1`, not the current CODATA value. Substituting
  the modern number moves the pressure at 86 km by about 0.02 %, which is
  small, but it means the model is no longer the 1976 standard and no longer
  reproduces its table. The standard is a *definition*; it is reproduced as
  written.
* Pressure is hydrostatic in the **molecular-scale** temperature
  :math:`T_M`, and the kinetic temperature is :math:`T = T_M M/M_0`. Between
  80 and 86 km these differ, and density follows :math:`p M_0/(R^* T_M)`, so
  the correction changes temperature — and hence viscosity, sound speed and
  mean free path — while leaving density alone.

Above 86 km the standard switches to a species-by-species diffusive
equilibrium that this module does not implement; see
:mod:`aether.atmosphere.upper`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "BOLTZMANN",
    "COLLISION_DIAMETER",
    "EARTH_RADIUS_EFFECTIVE",
    "GAMMA_AIR",
    "MOLAR_MASS_SEA_LEVEL",
    "SEA_LEVEL_GRAVITY",
    "UNIVERSAL_GAS_CONSTANT",
    "AtmosphereState",
    "USStandard1976",
    "geometric_altitude",
    "geopotential_altitude",
]

_FloatArray = NDArray[np.float64]

#: :math:`R^*`, J kmol\ :sup:`-1` K\ :sup:`-1` (the standard's value).
UNIVERSAL_GAS_CONSTANT = 8.31432e3
#: :math:`M_0`, kg/kmol — sea-level mean molar mass.
MOLAR_MASS_SEA_LEVEL = 28.9644
#: :math:`g_0`, m/s², the standard's defined value.
SEA_LEVEL_GRAVITY = 9.80665
#: :math:`r_0`, m — the effective Earth radius the geopotential is built on.
#: Not an Earth radius in any geodetic sense; it is chosen so that
#: :math:`g_0 r_0/(r_0+Z)^2` reproduces the standard's gravity profile.
EARTH_RADIUS_EFFECTIVE = 6356766.0
#: Avogadro's number, kmol\ :sup:`-1` (the standard's value).
AVOGADRO = 6.022169e26
#: :math:`k = R^*/N_A`, J/K.
BOLTZMANN = UNIVERSAL_GAS_CONSTANT / AVOGADRO
#: Effective collision diameter of air, m — sets the mean free path.
COLLISION_DIAMETER = 3.65e-10
#: Ratio of specific heats, taken constant by the standard.
GAMMA_AIR = 1.40
#: Sutherland's law constants, kg m\ :sup:`-1` s\ :sup:`-1` K\ :sup:`-1/2` and K.
SUTHERLAND_BETA = 1.458e-6
SUTHERLAND_S = 110.4

#: Geopotential altitudes of the layer bases (m).
_LAYER_BASE = np.array([0.0, 11.0, 20.0, 32.0, 47.0, 51.0, 71.0, 84.8520]) * 1.0e3
#: Molecular-scale temperature gradients within each layer (K/m).
_LAPSE_RATE = np.array([-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0]) * 1.0e-3
#: The model is defined from -5 km geopotential upward.
_LOWEST_GEOPOTENTIAL = -5.0e3
#: Geometric altitude at which the lower model ends (m).
TOP_OF_LOWER_ATMOSPHERE = 86.0e3

_SEA_LEVEL_TEMPERATURE = 288.15
_SEA_LEVEL_PRESSURE = 101325.0


def _base_temperatures() -> _FloatArray:
    """Layer base molecular-scale temperatures, by the defining recursion."""
    temperatures = np.empty(_LAYER_BASE.size)
    temperatures[0] = _SEA_LEVEL_TEMPERATURE
    for index in range(_LAPSE_RATE.size):
        span = _LAYER_BASE[index + 1] - _LAYER_BASE[index]
        temperatures[index + 1] = temperatures[index] + _LAPSE_RATE[index] * span
    return temperatures


def _base_pressures(temperatures: _FloatArray) -> _FloatArray:
    """Layer base pressures, by hydrostatic integration through each layer."""
    pressures = np.empty(_LAYER_BASE.size)
    pressures[0] = _SEA_LEVEL_PRESSURE
    coefficient = SEA_LEVEL_GRAVITY * MOLAR_MASS_SEA_LEVEL / UNIVERSAL_GAS_CONSTANT
    for index in range(_LAPSE_RATE.size):
        span = _LAYER_BASE[index + 1] - _LAYER_BASE[index]
        lapse = _LAPSE_RATE[index]
        base_t = temperatures[index]
        if lapse == 0.0:
            pressures[index + 1] = pressures[index] * np.exp(-coefficient * span / base_t)
        else:
            top_t = base_t + lapse * span
            pressures[index + 1] = pressures[index] * (base_t / top_t) ** (coefficient / lapse)
    return pressures


_BASE_TEMPERATURE = _base_temperatures()
_BASE_PRESSURE = _base_pressures(_BASE_TEMPERATURE)

#: :math:`M/M_0` between 80 and 86 km geometric, at 0.5 km spacing (Table 8).
#:
#: The endpoint is a check on the whole table rather than a free parameter:
#: :math:`T_M(86\\,\\mathrm{km}) = 186.946` K times 0.999579 is 186.8673 K,
#: which is exactly the kinetic temperature the *upper* model starts from.
#: The two halves of the standard are stitched at that number.
_MOLAR_RATIO_ALTITUDE = np.arange(80.0, 86.5, 0.5) * 1.0e3
_MOLAR_RATIO = np.array(
    [
        1.000000,
        0.999996,
        0.999989,
        0.999971,
        0.999941,
        0.999909,
        0.999870,
        0.999829,
        0.999786,
        0.999741,
        0.999694,
        0.999641,
        0.999579,
    ]
)


def geopotential_altitude(geometric: ArrayLike) -> _FloatArray:
    """:math:`H = r_0 Z/(r_0 + Z)` (m from m).

    Geopotential altitude absorbs the variation of gravity with height into
    the vertical coordinate, which is what lets the hydrostatic equation be
    integrated in closed form with a constant :math:`g_0`.
    """
    z = np.asarray(geometric, dtype=np.float64)
    return np.asarray(EARTH_RADIUS_EFFECTIVE * z / (EARTH_RADIUS_EFFECTIVE + z))


def geometric_altitude(geopotential: ArrayLike) -> _FloatArray:
    """Inverse of :func:`geopotential_altitude` (m from m)."""
    h = np.asarray(geopotential, dtype=np.float64)
    return np.asarray(EARTH_RADIUS_EFFECTIVE * h / (EARTH_RADIUS_EFFECTIVE - h))


@dataclass(frozen=True)
class AtmosphereState:
    """Thermodynamic state of the ambient gas, vectorised over altitude.

    The four stored fields are the ones a model computes; everything else is
    derived here so that two atmosphere models cannot disagree about what a
    Reynolds number means.
    """

    altitude: _FloatArray
    """Geometric altitude (m)."""
    temperature: _FloatArray
    """Kinetic temperature (K)."""
    pressure: _FloatArray
    """Static pressure (Pa)."""
    density: _FloatArray
    """Mass density (kg/m³)."""
    molar_mass: _FloatArray
    """Local mean molar mass (kg/kmol)."""

    @property
    def gas_constant(self) -> _FloatArray:
        """Specific gas constant :math:`R = R^*/M` (J kg⁻¹ K⁻¹)."""
        return np.asarray(UNIVERSAL_GAS_CONSTANT / self.molar_mass)

    @property
    def speed_of_sound(self) -> _FloatArray:
        """:math:`a = \\sqrt{\\gamma R T}` (m/s).

        Uses the *local* molar mass. The standard defines the speed of sound
        only below 86 km, where :math:`M = M_0`; above that the gas is
        rarefied enough that a Mach number is no longer the parameter that
        governs the flow — Knudsen number is — and the continuum sound speed
        is carried on as a bookkeeping quantity, not a claim.
        """
        return np.asarray(np.sqrt(GAMMA_AIR * self.gas_constant * self.temperature))

    @property
    def viscosity(self) -> _FloatArray:
        """Dynamic viscosity by Sutherland's law (Pa·s).

        Valid where the gas is a continuum of well-mixed air, which is the
        lower model's whole range. Above 86 km it is an extrapolation, and it
        is retained because Reynolds number there is small enough that
        viscous forces have stopped mattering before the correlation stops
        being right.
        """
        t = self.temperature
        return np.asarray(SUTHERLAND_BETA * t**1.5 / (t + SUTHERLAND_S))

    @property
    def kinematic_viscosity(self) -> _FloatArray:
        """:math:`\\nu = \\mu/\\rho` (m²/s)."""
        return np.asarray(self.viscosity / self.density)

    @property
    def number_density(self) -> _FloatArray:
        """Particle number density (m⁻³)."""
        return np.asarray(self.density * AVOGADRO / self.molar_mass)

    @property
    def mean_free_path(self) -> _FloatArray:
        """:math:`\\lambda = 1/(\\sqrt{2}\\pi\\sigma^2 n)` (m).

        6.6328e-8 m at sea level, which is the standard's tabulated value and
        the reason :data:`COLLISION_DIAMETER` is 3.65 Å rather than one of the
        several other effective diameters in circulation.
        """
        return np.asarray(
            1.0 / (np.sqrt(2.0) * np.pi * COLLISION_DIAMETER**2 * self.number_density)
        )

    @property
    def scale_height(self) -> _FloatArray:
        """:math:`H = RT/g` (m) — the local e-folding height of density."""
        return np.asarray(self.gas_constant * self.temperature / gravity(self.altitude))


def gravity(altitude: ArrayLike) -> _FloatArray:
    """Gravitational acceleration by the standard's inverse-square law (m/s²)."""
    z = np.asarray(altitude, dtype=np.float64)
    return np.asarray(
        SEA_LEVEL_GRAVITY * (EARTH_RADIUS_EFFECTIVE / (EARTH_RADIUS_EFFECTIVE + z)) ** 2
    )


@dataclass(frozen=True)
class USStandard1976:
    """The 1976 standard's lower model: 0 to 86 km geometric.

    Stateless and vectorised — :meth:`state` accepts a scalar or an array of
    altitudes and does the layer lookup with a single ``searchsorted``, so
    evaluating it along a whole trajectory costs one pass, not one call per
    sample.
    """

    name: str = "US Standard Atmosphere 1976"

    #: Geometric altitude above which :meth:`state` refuses to answer.
    ceiling: float = TOP_OF_LOWER_ATMOSPHERE
    #: Geometric altitude below which :meth:`state` refuses to answer.
    floor: float = -5.0e3

    def molecular_temperature(self, altitude: ArrayLike) -> _FloatArray:
        """Molecular-scale temperature :math:`T_M` (K).

        This — not the kinetic temperature — is what the hydrostatic
        integration uses, and the two differ above 80 km.
        """
        h = geopotential_altitude(altitude)
        index = self._layer_index(h)
        return np.asarray(_BASE_TEMPERATURE[index] + _LAPSE_RATE[index] * (h - _LAYER_BASE[index]))

    def molar_mass(self, altitude: ArrayLike) -> _FloatArray:
        """Mean molar mass (kg/kmol); :math:`M_0` below 80 km."""
        z = np.asarray(altitude, dtype=np.float64)
        ratio = np.interp(z, _MOLAR_RATIO_ALTITUDE, _MOLAR_RATIO, left=1.0, right=_MOLAR_RATIO[-1])
        return np.asarray(MOLAR_MASS_SEA_LEVEL * ratio)

    def pressure(self, altitude: ArrayLike) -> _FloatArray:
        """Static pressure (Pa), in closed form within each layer."""
        h = geopotential_altitude(altitude)
        index = self._layer_index(h)
        base_h = _LAYER_BASE[index]
        base_t = _BASE_TEMPERATURE[index]
        base_p = _BASE_PRESSURE[index]
        lapse = _LAPSE_RATE[index]
        coefficient = SEA_LEVEL_GRAVITY * MOLAR_MASS_SEA_LEVEL / UNIVERSAL_GAS_CONSTANT

        # Both branches are evaluated everywhere and selected between, so the
        # expression stays vectorised. The isothermal branch is guarded with a
        # non-zero stand-in lapse rate purely to keep the unused half finite;
        # `np.where` evaluates both arms regardless of the condition.
        isothermal = lapse == 0.0
        safe_lapse = np.where(isothermal, 1.0, lapse)
        top_t = base_t + safe_lapse * (h - base_h)
        gradient_branch = base_p * (base_t / top_t) ** (coefficient / safe_lapse)
        isothermal_branch = base_p * np.exp(-coefficient * (h - base_h) / base_t)
        return np.asarray(np.where(isothermal, isothermal_branch, gradient_branch))

    def state(self, altitude: ArrayLike) -> AtmosphereState:
        """Full thermodynamic state at geometric altitude (m)."""
        z = np.asarray(altitude, dtype=np.float64)
        if np.any(~np.isfinite(z)):
            msg = "altitude must be finite"
            raise ValueError(msg)
        if np.any(z < self.floor) or np.any(z > self.ceiling):
            low, high = float(np.min(z)), float(np.max(z))
            msg = (
                f"{self.name} is defined from {self.floor / 1e3:g} to "
                f"{self.ceiling / 1e3:g} km geometric; asked for "
                f"[{low / 1e3:g}, {high / 1e3:g}] km. Above 86 km the standard "
                f"switches to species-wise diffusive equilibrium — use "
                f"aether.atmosphere.EARTH, which stitches an upper model on."
            )
            raise ValueError(msg)
        molecular_t = self.molecular_temperature(z)
        pressure = self.pressure(z)
        molar = self.molar_mass(z)
        # Density follows the molecular-scale temperature and M0, so the
        # 80-86 km molar correction leaves it untouched and moves only T.
        density = pressure * MOLAR_MASS_SEA_LEVEL / (UNIVERSAL_GAS_CONSTANT * molecular_t)
        kinetic_t = molecular_t * molar / MOLAR_MASS_SEA_LEVEL
        return AtmosphereState(
            altitude=z,
            temperature=kinetic_t,
            pressure=pressure,
            density=np.asarray(density),
            molar_mass=molar,
        )

    @staticmethod
    def _layer_index(geopotential: _FloatArray) -> NDArray[np.intp]:
        """Which layer each geopotential altitude falls in."""
        h = np.asarray(geopotential, dtype=np.float64)
        if np.any(h < _LOWEST_GEOPOTENTIAL):
            msg = "the standard is defined from -5 km geopotential upward"
            raise ValueError(msg)
        index = np.searchsorted(_LAYER_BASE, h, side="right") - 1
        return np.asarray(np.clip(index, 0, _LAPSE_RATE.size - 1))
