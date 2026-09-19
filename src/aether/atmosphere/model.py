"""One atmosphere from two models, and the similarity parameters it implies.

Nothing downstream should have to know where 86 km is. :class:`LayeredAtmosphere`
takes a lower model and an upper one and returns a single state that is
continuous, thermodynamically consistent and defined from the ground to the
exosphere; :class:`Freestream` turns that state plus a speed into the three
numbers that decide which aerodynamic theory applies.

**Why the blend is on log-density and temperature.** Blending pressure and
density independently would put the mixture off the gas law in the seam
band, and a solver that recomputes one from the other would then see a
different state than the one it was handed. Blending :math:`\\ln\\rho`,
:math:`T` and :math:`M` and *deriving* :math:`p = \\rho R T` keeps
:math:`p = \\rho R T` exact everywhere, and reproduces each model exactly
outside the band. Log-density because density falls exponentially: a linear
blend of two exponentials across 6 km of a 6 km scale height is not close to
either.

**Why the seam is blended at all.** Drag is proportional to density, so a
step in density is a step in acceleration. An adaptive integrator meeting one
will cut its step size until it either resolves the discontinuity or gives
up, and either way the cost lands on every trajectory that crosses 86 km —
which, for this vehicle, is all of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import scipy.interpolate
from numpy.typing import ArrayLike, NDArray

from aether.atmosphere.standard import (
    GAMMA_AIR,
    AtmosphereState,
    USStandard1976,
    gravity,
)
from aether.atmosphere.upper import MODERATE_ACTIVITY, MSISAtmosphere, SolarActivity
from aether.blending import smoothstep

__all__ = [
    "Atmosphere",
    "Freestream",
    "LayeredAtmosphere",
    "TabulatedAtmosphere",
    "earth_atmosphere",
    "tabulate",
]

_FloatArray = NDArray[np.float64]


class Atmosphere(Protocol):
    """Anything that can report the gas state at an altitude.

    ``name`` is a read-only property rather than an attribute so that frozen
    dataclasses satisfy it; a Protocol declaring ``name: str`` demands a
    settable one and every model here is immutable.
    """

    @property
    def name(self) -> str: ...

    def state(self, altitude: ArrayLike) -> AtmosphereState:  # pragma: no cover
        ...


@dataclass(frozen=True)
class Freestream:
    """Ambient state plus a speed: the similarity parameters of the flow.

    The three that matter, and what each one decides:

    ``mach``
        Whether the flow is compressible, and which pressure closure applies.
    ``reynolds``
        Whether the boundary layer is laminar or turbulent, and how much of
        the drag is friction.
    ``knudsen``
        Whether there is a boundary layer at all. Above :math:`Kn \\approx 10`
        molecules do not collide with each other near the body and the
        continuum equations are not merely inaccurate, they are inapplicable.
    """

    state: AtmosphereState
    speed: _FloatArray
    """Freestream speed (m/s)."""
    reference_length: float
    """Length the Reynolds and Knudsen numbers are formed on (m)."""

    def __post_init__(self) -> None:
        if not (np.isfinite(self.reference_length) and self.reference_length > 0.0):
            msg = f"reference_length must be finite and > 0, got {self.reference_length}"
            raise ValueError(msg)
        if np.any(np.asarray(self.speed) < 0.0):
            msg = "freestream speed must be non-negative"
            raise ValueError(msg)

    @property
    def mach(self) -> _FloatArray:
        return np.asarray(self.speed / self.state.speed_of_sound)

    @property
    def dynamic_pressure(self) -> _FloatArray:
        """:math:`q = \\tfrac12\\rho V^2` (Pa)."""
        return np.asarray(0.5 * self.state.density * self.speed**2)

    @property
    def reynolds(self) -> _FloatArray:
        """:math:`Re_L = \\rho V L/\\mu`."""
        return np.asarray(
            self.state.density * self.speed * self.reference_length / self.state.viscosity
        )

    @property
    def knudsen(self) -> _FloatArray:
        """:math:`Kn = \\lambda/L`."""
        return np.asarray(self.state.mean_free_path / self.reference_length)

    @property
    def speed_ratio(self) -> _FloatArray:
        """:math:`S = V/\\sqrt{2RT}` — the molecular speed ratio.

        The hypersonic analogue of Mach number for a rarefied gas: it is the
        freestream speed against the most probable thermal speed, and it is
        what the free-molecular integrals are written in.
        Related to Mach by :math:`S = M\\sqrt{\\gamma/2}`.
        """
        thermal = np.sqrt(2.0 * self.state.gas_constant * self.state.temperature)
        return np.asarray(self.speed / thermal)

    @property
    def total_temperature(self) -> _FloatArray:
        """:math:`T_0 = T(1 + \\tfrac12(\\gamma-1)M^2)` (K), perfect gas.

        At Mach 20 this is above 20,000 K, which is the point: a perfect-gas
        stagnation temperature is a *statement that the perfect gas
        assumption has failed*, since air dissociates well before it. It is
        computed anyway because it is the number the real-gas model has to
        replace, and seeing it makes the replacement legible.
        """
        return np.asarray(self.state.temperature * (1.0 + 0.5 * (GAMMA_AIR - 1.0) * self.mach**2))


@dataclass(frozen=True)
class LayeredAtmosphere:
    """A lower and an upper model, blended over a band and continuous across it.

    Attributes
    ----------
    lower, upper:
        Models valid below and above the band. ``lower`` is asked for nothing
        above ``blend_top`` and ``upper`` for nothing below ``blend_bottom``,
        so each is used only where it is defined.
    blend_bottom, blend_top:
        Geometric altitudes bounding the transition (m).
    """

    lower: Atmosphere = field(default_factory=USStandard1976)
    upper: Atmosphere = field(default_factory=MSISAtmosphere)
    blend_bottom: float = 80.0e3
    blend_top: float = 86.0e3
    name: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if not self.blend_bottom < self.blend_top:
            msg = (
                f"blend band must satisfy bottom < top, got ({self.blend_bottom}, {self.blend_top})"
            )
            raise ValueError(msg)
        object.__setattr__(self, "name", f"{self.lower.name} / {self.upper.name}")

    def state(self, altitude: ArrayLike) -> AtmosphereState:
        """Blended gas state at geometric altitude (m)."""
        z = np.asarray(altitude, dtype=np.float64)
        shape = z.shape
        flat = np.atleast_1d(z)

        temperature = np.empty_like(flat)
        density = np.empty_like(flat)
        molar_mass = np.empty_like(flat)

        below = flat < self.blend_bottom
        above = flat > self.blend_top
        band = ~(below | above)

        if np.any(below):
            lower = self.lower.state(flat[below])
            temperature[below] = lower.temperature
            density[below] = lower.density
            molar_mass[below] = lower.molar_mass
        if np.any(above):
            upper = self.upper.state(flat[above])
            temperature[above] = upper.temperature
            density[above] = upper.density
            molar_mass[above] = upper.molar_mass
        if np.any(band):
            heights = flat[band]
            low = self.lower.state(heights)
            high = self.upper.state(heights)
            weight = smoothstep(
                (heights - self.blend_bottom) / (self.blend_top - self.blend_bottom)
            )
            temperature[band] = (1.0 - weight) * low.temperature + weight * high.temperature
            molar_mass[band] = (1.0 - weight) * low.molar_mass + weight * high.molar_mass
            density[band] = np.exp(
                (1.0 - weight) * np.log(low.density) + weight * np.log(high.density)
            )

        from aether.atmosphere.standard import UNIVERSAL_GAS_CONSTANT

        pressure = density * (UNIVERSAL_GAS_CONSTANT / molar_mass) * temperature
        return AtmosphereState(
            altitude=np.asarray(flat.reshape(shape)),
            temperature=np.asarray(temperature.reshape(shape)),
            pressure=np.asarray(pressure.reshape(shape)),
            density=np.asarray(density.reshape(shape)),
            molar_mass=np.asarray(molar_mass.reshape(shape)),
        )

    def freestream(
        self, altitude: ArrayLike, speed: ArrayLike, reference_length: float
    ) -> Freestream:
        """Gas state and similarity parameters in one call."""
        return Freestream(
            state=self.state(altitude),
            speed=np.asarray(speed, dtype=np.float64),
            reference_length=float(reference_length),
        )

    def gravity(self, altitude: ArrayLike) -> _FloatArray:
        """Gravitational acceleration by the standard's inverse-square law (m/s²)."""
        return gravity(altitude)


@dataclass(frozen=True)
class TabulatedAtmosphere:
    """A sampled atmosphere that can be evaluated inside an integrator.

    :class:`LayeredAtmosphere` calls into NRLMSIS, which is a Fortran model
    with a per-call cost of tens of microseconds. That is nothing once and
    everything inside a right-hand side that an implicit integrator will
    evaluate a hundred thousand times. So it is sampled once onto a grid and
    interpolated afterwards.

    Interpolation is monotone (PCHIP) on :math:`\\ln\\rho`, :math:`T` and
    :math:`M`, with pressure *derived* from the gas law rather than
    interpolated, so :math:`p = \\rho R T` holds exactly at every point and
    not merely at the samples. Log-density because density spans twenty
    decades over the tabulated range, and monotone because an overshoot in
    density is a spurious drag pulse.

    Above the tabulated ceiling the density is continued exponentially at the
    local scale height rather than clamped. Clamping would leave a satellite
    at 1200 km flying through 1000 km air, which over an orbit is a large
    error in the one direction that matters.
    """

    altitude: _FloatArray
    log_density: _FloatArray
    temperature_samples: _FloatArray
    molar_mass_samples: _FloatArray
    name: str = "tabulated atmosphere"
    _interpolants: tuple[Any, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        z = np.asarray(self.altitude, dtype=np.float64)
        if z.ndim != 1 or z.size < 4 or np.any(np.diff(z) <= 0.0):
            msg = "altitude must be a strictly increasing 1-D array of 4+ samples"
            raise ValueError(msg)
        object.__setattr__(
            self,
            "_interpolants",
            tuple(
                scipy.interpolate.PchipInterpolator(z, values, extrapolate=False)
                for values in (
                    self.log_density,
                    self.temperature_samples,
                    self.molar_mass_samples,
                )
            ),
        )

    @property
    def ceiling(self) -> float:
        return float(np.asarray(self.altitude)[-1])

    @property
    def floor(self) -> float:
        return float(np.asarray(self.altitude)[0])

    def state(self, altitude: ArrayLike) -> AtmosphereState:
        z = np.asarray(altitude, dtype=np.float64)
        clamped = np.clip(z, self.floor, self.ceiling)
        density = np.exp(self._interpolants[0](clamped))
        temperature = np.asarray(self._interpolants[1](clamped))
        molar_mass = np.asarray(self._interpolants[2](clamped))

        # Exponential continuation above the ceiling, at the scale height the
        # top two samples imply.
        top = np.asarray(self.altitude)[-2:]
        log_top = np.asarray(self.log_density)[-2:]
        decay = (log_top[1] - log_top[0]) / (top[1] - top[0])
        above = z > self.ceiling
        if np.any(above):
            density = np.where(above, density * np.exp(decay * (z - self.ceiling)), density)

        from aether.atmosphere.standard import UNIVERSAL_GAS_CONSTANT

        pressure = density * (UNIVERSAL_GAS_CONSTANT / molar_mass) * temperature
        return AtmosphereState(
            altitude=np.asarray(z),
            temperature=temperature,
            pressure=np.asarray(pressure),
            density=np.asarray(density),
            molar_mass=molar_mass,
        )

    def freestream(
        self, altitude: ArrayLike, speed: ArrayLike, reference_length: float
    ) -> Freestream:
        return Freestream(
            state=self.state(altitude),
            speed=np.asarray(speed, dtype=np.float64),
            reference_length=float(reference_length),
        )


def tabulate(
    atmosphere: Atmosphere,
    floor: float = 0.0,
    ceiling: float = 1000.0e3,
    samples: int = 4001,
) -> TabulatedAtmosphere:
    """Sample an atmosphere onto a grid fine enough to interpolate exactly.

    Four thousand points over a thousand kilometres is 250 m spacing, against
    a scale height that is never below 6 km. Monotone cubic interpolation on
    the log of an exponential at 1/24 of its scale height is accurate to
    parts in :math:`10^9`; the grid is not where the error is.
    """
    z = np.linspace(float(floor), float(ceiling), int(samples))
    sampled = atmosphere.state(z)
    return TabulatedAtmosphere(
        altitude=z,
        log_density=np.log(np.asarray(sampled.density)),
        temperature_samples=np.asarray(sampled.temperature),
        molar_mass_samples=np.asarray(sampled.molar_mass),
        name=f"{atmosphere.name} (tabulated)",
    )


def earth_atmosphere(
    activity: SolarActivity = MODERATE_ACTIVITY, version: float = 0.0
) -> LayeredAtmosphere:
    """The default Earth atmosphere: the 1976 standard under NRLMSIS.

    ``activity`` is the design condition for everything above 86 km and is
    worth choosing deliberately — the density at 300 km differs by more than
    a factor of ten between :data:`~aether.atmosphere.upper.SOLAR_MINIMUM`
    and :data:`~aether.atmosphere.upper.SOLAR_MAXIMUM`.
    """
    return LayeredAtmosphere(
        lower=USStandard1976(),
        upper=MSISAtmosphere(activity=activity, version=version),
    )
