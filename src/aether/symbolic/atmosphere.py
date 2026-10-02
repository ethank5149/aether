r"""The atmosphere on symbols: the 1976 standard as defined, and a measured fit above it.

A field that is to be printed to C cannot call an interpolant. Whatever the
atmosphere supplies to it -- density for the drag, pressure and temperature
for the speed of sound -- has to be an expression in the altitude. This
module builds that expression, in two parts that are not of the same
standing, and says which is which.

*Below 86 km* the US Standard Atmosphere 1976 is a definition: seven layers
of geopotential altitude, a constant lapse rate of the molecular-scale
temperature in each, and hydrostatic equilibrium of a perfect gas. In each
layer the pressure is then a closed form, and :func:`us_standard_1976`
states it as the standard does. Nothing is fitted. The base values are
produced by the standard's own recursion, in exact rational arithmetic for
the temperatures, and the derivations check that each layer satisfies the
hydrostatic equation and that the layers join.

*Above 86 km* the standard is a species-by-species diffusive equilibrium
that has no closed form, and the reference is a model evaluated numerically
(:class:`aether.atmosphere.upper.MSISAtmosphere`). What is used here is a
fit: a monotone cubic in :math:`\ln\rho` and in :math:`\ln p` through a few
dozen knots sampled from that reference. It is data, with an error against
its source that is measured and returned (:attr:`StandardAtmosphere.fit_error`),
and it is the piecewise-exponential form the dynamics appendix asks of the
density.

Between 80 and 86 km the two are blended on the logarithm with the quintic
smoothstep, exactly as :class:`aether.atmosphere.model.LayeredAtmosphere`
blends them, so that this expression and that kernel are the same function
and the tests can hold one against the other.

The altitude is geometric height above the surface the planet model
supplies; :mod:`aether.symbolic.figure` says what that surface is.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any

import numpy as np
import sympy as sp

__all__ = [
    "StandardAtmosphere",
    "StandardLayer",
    "standard_atmosphere",
    "us_standard_1976_layers",
]

#: The standard's defining constants (NASA-TM-X-74335), exactly as it gives them.
_GAS_CONSTANT = sp.Rational(831432, 100)  # R*, J/(kmol K)
_MOLAR_MASS = sp.Rational(289644, 10000)  # M_0, kg/kmol
_GRAVITY = sp.Rational(980665, 100000)  # g_0, m/s^2
_EARTH_RADIUS = sp.Integer(6356766)  # r_0, m: the radius the geopotential is built on
_HEAT_RATIO = sp.Rational(7, 5)
_SEA_LEVEL_TEMPERATURE = sp.Rational(28815, 100)  # K
_SEA_LEVEL_PRESSURE = sp.Integer(101325)  # Pa
#: Base of each layer in geopotential metres, and the lapse rate above it (K/m).
_LAYER_BASES = tuple(sp.Rational(v) for v in (0, 11000, 20000, 32000, 47000, 51000, 71000))
_LAPSE_RATES = tuple(
    sp.Rational(v, 10000) for v in (-65, 0, 10, 28, 0, -28, -20)
)
_TOP_GEOPOTENTIAL = sp.Rational(84852)

#: The band over which the lower and upper models are blended (m, geometric).
_BLEND_BOTTOM, _BLEND_TOP = 80.0e3, 86.0e3

#: Knots of the upper fit (km): spacing follows the scale height, which grows
#: from six kilometres at the mesopause to sixty in the thermosphere.
_UPPER_KNOT_SPEC = ((80, 124, 2), (124, 160, 3), (160, 220, 6), (220, 400, 15), (400, 1000, 50))

#: Digits kept when an exact base pressure is turned into a constant of the
#: generated code: more than a double holds, so the constant is correctly rounded.
_DIGITS = 25


@dataclass(frozen=True)
class StandardLayer:
    """One layer of the 1976 standard, in geopotential altitude.

    The fields may be exact rationals, as the standard defines them; symbols,
    for a derivation that should hold of any layer; or floats, for the
    expression that is printed.
    """

    base: Any
    """Geopotential altitude of the layer's base (m)."""
    lapse_rate: Any
    """Gradient of the molecular-scale temperature (K/m); zero if isothermal."""
    base_temperature: Any
    base_pressure: Any
    hydrostatic_constant: Any = _GRAVITY * _MOLAR_MASS / _GAS_CONSTANT
    r""":math:`g_0 M_0/R^*` (K/m): what the gas law and gravity contribute."""
    isothermal: bool | None = None
    """Whether the lapse rate is zero. Read off the lapse rate when it is a
    number; has to be said when it is a symbol."""

    @property
    def _isothermal(self) -> bool:
        return bool(self.lapse_rate == 0) if self.isothermal is None else self.isothermal

    def temperature(self, geopotential: Any) -> Any:
        r""":math:`T_M = T_b + L\,(H - H_b)`."""
        if self._isothermal:
            return self.base_temperature
        return self.base_temperature + self.lapse_rate * (geopotential - self.base)

    def pressure(self, geopotential: Any) -> Any:
        r"""The hydrostatic pressure of a perfect gas in this layer.

        :math:`p_b\,(T_b/T_M)^{g_0 M_0/(R^* L)}` where the temperature has a
        gradient, and :math:`p_b\exp(-g_0 M_0 (H - H_b)/(R^* T_b))` where it
        has none.
        """
        if self._isothermal:
            return self.base_pressure * sp.exp(
                -self.hydrostatic_constant * (geopotential - self.base) / self.base_temperature
            )
        ratio = self.base_temperature / self.temperature(geopotential)
        return self.base_pressure * ratio ** (self.hydrostatic_constant / self.lapse_rate)

    def density(self, geopotential: Any) -> Any:
        r""":math:`\rho = p\,M_0/(R^* T_M)`, written with :math:`g_0 M_0/R^*` over
        :math:`g_0`."""
        return (
            self.pressure(geopotential)
            * self.hydrostatic_constant
            / (_GRAVITY * self.temperature(geopotential))
        )


def geopotential(altitude: Any) -> Any:
    r""":math:`H = r_0 Z/(r_0 + Z)`: the altitude that makes gravity constant."""
    return _EARTH_RADIUS * altitude / (_EARTH_RADIUS + altitude)


def geometric(geopotential_altitude: Any) -> Any:
    r""":math:`Z = r_0 H/(r_0 - H)`, the inverse."""
    return _EARTH_RADIUS * geopotential_altitude / (_EARTH_RADIUS - geopotential_altitude)


def gravity(altitude: Any) -> Any:
    r""":math:`g(Z) = g_0\,(r_0/(r_0 + Z))^2`, the standard's gravity."""
    return _GRAVITY * (_EARTH_RADIUS / (_EARTH_RADIUS + altitude)) ** 2


@lru_cache(maxsize=1)
def us_standard_1976_layers() -> tuple[StandardLayer, ...]:
    """The seven layers, with base values produced by the standard's recursion.

    The base temperatures are exact rationals. Each base pressure is the
    pressure of the layer below at its top, evaluated in thirty-digit
    arithmetic: the published table is then something to test against, not
    something typed in.
    """
    layers: list[StandardLayer] = []
    temperature: Any = _SEA_LEVEL_TEMPERATURE
    pressure: Any = sp.Float(_SEA_LEVEL_PRESSURE, _DIGITS + 5)
    tops = (*_LAYER_BASES[1:], _TOP_GEOPOTENTIAL)
    for base, lapse, top in zip(_LAYER_BASES, _LAPSE_RATES, tops, strict=True):
        layer = StandardLayer(base, lapse, temperature, pressure)
        layers.append(layer)
        exact = StandardLayer(base, lapse, temperature, sp.Integer(1))
        pressure = pressure * sp.N(exact.pressure(top), _DIGITS + 5)
        temperature = layer.temperature(top)
    return tuple(layers)


def _upper_knots() -> np.ndarray:
    knots: list[float] = []
    for low, high, step in _UPPER_KNOT_SPEC:
        knots.extend(np.arange(low, high, step))
    knots.append(_UPPER_KNOT_SPEC[-1][1])
    return np.asarray(knots, dtype=np.float64) * 1.0e3


def _smoothstep(t: Any) -> Any:
    """The quintic :math:`6t^5 - 15t^4 + 10t^3` of :mod:`aether.blending`."""
    return t * t * t * (t * (6 * t - 15) + 10)


@dataclass(frozen=True)
class _LogCubic:
    """A monotone piecewise cubic through knots, as coefficients about each knot."""

    knots: tuple[float, ...]
    coefficients: tuple[tuple[float, float, float, float], ...]
    """Per segment, descending powers of the distance above its lower knot."""

    @classmethod
    def through(cls, knots: Any, values: Any) -> _LogCubic:
        from scipy.interpolate import PchipInterpolator

        fit = PchipInterpolator(np.asarray(knots), np.asarray(values))
        columns = np.asarray(fit.c).T
        return cls(
            tuple(float(k) for k in knots),
            tuple((float(a), float(b), float(c), float(d)) for a, b, c, d in columns),
        )

    def segment(self, index: int, altitude: Any) -> Any:
        a, b, c, d = self.coefficients[index]
        above = altitude - sp.Float(self.knots[index])
        return ((sp.Float(a) * above + sp.Float(b)) * above + sp.Float(c)) * above + sp.Float(d)

    def continued(self, altitude: Any) -> Any:
        """Above the last knot: the value there, continued at the slope there."""
        a, b, c, d = self.coefficients[-1]
        span = self.knots[-1] - self.knots[-2]
        value = ((a * span + b) * span + c) * span + d
        slope = (3.0 * a * span + 2.0 * b) * span + c
        return sp.Float(value) + sp.Float(slope) * (altitude - sp.Float(self.knots[-1]))

    def evaluate(self, altitude: np.ndarray) -> np.ndarray:
        """The same function on floats, for measuring the fit."""
        z = np.asarray(altitude, dtype=np.float64)
        knots = np.asarray(self.knots)
        index = np.clip(np.searchsorted(knots, z, side="right") - 1, 0, knots.size - 2)
        above = z - knots[index]
        a, b, c, d = np.asarray(self.coefficients).T
        return np.asarray(((a[index] * above + b[index]) * above + c[index]) * above + d[index])


@dataclass(frozen=True)
class StandardAtmosphere:
    r"""Density, pressure and speed of sound as expressions in the altitude.

    Built by :func:`standard_atmosphere`. The three expressions share one
    partition of the altitude: the seven layers of the standard up to 80 km,
    the blend band to 86 km, and the segments of the upper fit above it.

    Attributes
    ----------
    upper_density, upper_pressure:
        The fits to the upper reference, in :math:`\ln\rho` and :math:`\ln p`.
    fit_error:
        Largest :math:`\lvert\Delta\ln\rho\rvert` between the fit and the
        reference it was sampled from, on a 250 m grid above 80 km. A
        relative density error, to first order.
    reference:
        Name of the upper reference model.
    """

    upper_density: _LogCubic
    upper_pressure: _LogCubic
    fit_error: float
    reference: str

    # -- the lower model, as defined ------------------------------------------------

    @staticmethod
    def _rounded(layer: StandardLayer) -> StandardLayer:
        """The layer with its constants as floats, for the expression that is printed.

        Left as rationals, the powers would be split by SymPy into products
        of prime powers of very different sizes, which is exact and prints
        badly.
        """

        def number(value: Any) -> Any:
            return sp.Float(sp.N(value, _DIGITS), _DIGITS)

        return StandardLayer(
            number(layer.base),
            number(layer.lapse_rate),
            number(layer.base_temperature),
            number(layer.base_pressure),
            number(layer.hydrostatic_constant),
            isothermal=layer.lapse_rate == 0,
        )

    @cached_property
    def layers(self) -> tuple[StandardLayer, ...]:
        """The standard's layers, base pressures rounded for printing."""
        return tuple(self._rounded(layer) for layer in us_standard_1976_layers())

    @cached_property
    def layer_tops(self) -> tuple[Any, ...]:
        """Geometric altitude at which each layer below 80 km hands over (m)."""
        tops = [geometric(base) for base in _LAYER_BASES[1:]]
        return tuple(sp.Float(sp.N(top, _DIGITS), _DIGITS) for top in tops)

    # -- assembling one quantity ------------------------------------------------------

    def _assemble(self, altitude: Any, lower: str, upper: _LogCubic) -> Any:
        height = geopotential(altitude)
        pieces: list[tuple[Any, Any]] = []
        for layer, top in zip(self.layers[:-1], self.layer_tops, strict=True):
            pieces.append((getattr(layer, lower)(height), altitude < top))
        last = getattr(self.layers[-1], lower)(height)
        pieces.append((last, altitude < _BLEND_BOTTOM))

        weight = _smoothstep((altitude - _BLEND_BOTTOM) / (_BLEND_TOP - _BLEND_BOTTOM))
        for index, top in enumerate(upper.knots[1:]):
            fitted = upper.segment(index, altitude)
            if top <= _BLEND_TOP:
                blended = sp.exp((1 - weight) * sp.log(last) + weight * fitted)
                pieces.append((blended, altitude < top))
            else:
                pieces.append((sp.exp(fitted), altitude < top))
        pieces.append((sp.exp(upper.continued(altitude)), True))
        return sp.Piecewise(*pieces)

    def density(self, altitude: Any) -> Any:
        r""":math:`\rho(h)` (kg/m³)."""
        return self._assemble(altitude, "density", self.upper_density)

    def pressure(self, altitude: Any) -> Any:
        r""":math:`p(h)` (Pa)."""
        return self._assemble(altitude, "pressure", self.upper_pressure)

    def sound_speed(self, altitude: Any) -> Any:
        r""":math:`a = \sqrt{\kappa p/\rho}` with :math:`\kappa = 1.4` (m/s).

        Below 86 km this is :math:`\sqrt{\kappa R^* T_M/M_0}` exactly, the
        molecular-scale temperature being the one the gas law is written in.
        """
        return sp.sqrt(_HEAT_RATIO * self.pressure(altitude) / self.density(altitude))

    def as_density(self, scale: Any = 1) -> Any:
        """``density(altitude, ops)`` for a field written against ``MathOps``.

        ``scale`` multiplies the density: a number, or a symbol that the
        generated code then takes as a parameter. It is how an atmosphere
        thinner or thicker than the standard is flown through.
        """

        def density(altitude: Any, _ops: Any) -> Any:
            return scale * self.density(altitude)

        return density


@lru_cache(maxsize=4)
def standard_atmosphere(reference: Any = None) -> StandardAtmosphere:
    """The symbolic atmosphere: the 1976 standard under a fit to ``reference``.

    ``reference`` is the upper model, anything with a ``state(altitude)``
    returning density and pressure; it defaults to
    :class:`aether.atmosphere.upper.MSISAtmosphere`, the upper half of
    :func:`aether.atmosphere.model.earth_atmosphere`. It is sampled at the
    knots, and on a 250 m grid to measure what the fit costs.
    """
    if reference is None:
        from aether.atmosphere.upper import MSISAtmosphere

        reference = MSISAtmosphere()
    knots = _upper_knots()
    sampled = reference.state(knots)
    density = _LogCubic.through(knots, np.log(np.asarray(sampled.density)))
    pressure = _LogCubic.through(knots, np.log(np.asarray(sampled.pressure)))
    grid = np.arange(knots[0], knots[-1] + 1.0, 250.0)
    truth = np.log(np.asarray(reference.state(grid).density))
    error = float(np.max(np.abs(density.evaluate(grid) - truth)))
    return StandardAtmosphere(density, pressure, error, str(reference.name))
