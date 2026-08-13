"""Charring material model (Paper I, §3.4.1).

The solid is three components — two resin constituents (A, B) and a
filler (C) — each with an independent Arrhenius decomposition law; the
bulk density is :math:`\\rho = \\Gamma(\\rho_A + \\rho_B) +
(1 - \\Gamma)\\rho_C` for resin volume fraction :math:`\\Gamma`, and the
transport properties are interpolated between virgin and char values by
the degree of char :math:`\\beta = (\\rho_v - \\rho)/(\\rho_v - \\rho_c)`.

Property functions carry *analytic derivatives* with respect to both
temperature and degree of char. That is not decoration: the V4 method of
manufactured solutions needs
:math:`\\partial_\\eta\\big(k(T,\\beta)\\,\\partial_\\eta T\\big)` in
closed form, which requires :math:`\\partial k/\\partial T` and
:math:`\\partial k/\\partial \\beta` exactly, not by finite differences.

The bundled :func:`demo_material` is synthetic: generic magnitudes
representative of a charring phenolic-class ablator, constructed for
verification exercises. It is traceable to no real material system and
must not be used for design.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "ArrheniusComponent",
    "CharringMaterial",
    "LinearBlendProperty",
    "demo_material",
]

_FloatArray = NDArray[np.float64]

#: Universal gas constant, J/(mol K).
GAS_CONSTANT = 8.31446261815324


@dataclass(frozen=True)
class ArrheniusComponent:
    """One decomposing solid component of Paper I, Eq. (3.14).

    Attributes
    ----------
    pre_exponential:
        :math:`A_i` (1/s).
    activation_energy:
        :math:`E_i` (J/mol).
    reaction_order:
        :math:`n_i` (–).
    virgin_density:
        :math:`\\rho_{v,i}` (kg/m³).
    char_density:
        :math:`\\rho_{c,i}` (kg/m³), with
        ``char_density < virgin_density``.
    """

    pre_exponential: float
    activation_energy: float
    reaction_order: float
    virgin_density: float
    char_density: float

    def __post_init__(self) -> None:
        if not (np.isfinite(self.pre_exponential) and self.pre_exponential >= 0.0):
            raise ValueError(f"pre_exponential must be finite and >= 0, got {self.pre_exponential}")
        if not (np.isfinite(self.activation_energy) and self.activation_energy > 0.0):
            raise ValueError(
                f"activation_energy must be finite and > 0, got {self.activation_energy}"
            )
        if not (np.isfinite(self.reaction_order) and self.reaction_order >= 0.0):
            raise ValueError(f"reaction_order must be finite and >= 0, got {self.reaction_order}")
        if not 0.0 <= self.char_density < self.virgin_density:
            raise ValueError(
                f"need 0 <= char_density < virgin_density, got "
                f"{self.char_density} / {self.virgin_density}"
            )


@dataclass(frozen=True)
class LinearBlendProperty:
    """Property affine in T for each phase, blended linearly in char fraction.

    .. math::

        p(T, \\beta) = (1 - \\beta)\\,(p_v + p_v' T) + \\beta\\,(p_c + p_c' T)

    Affine-in-T phases keep :math:`\\partial p/\\partial T` and
    :math:`\\partial p/\\partial \\beta` exact one-liners for the MMS
    machinery while retaining the virgin/char asymmetry that matters
    physically.
    """

    virgin_intercept: float
    virgin_slope: float
    char_intercept: float
    char_slope: float

    def value(self, temperature: ArrayLike, char_fraction: ArrayLike) -> _FloatArray:
        t = np.asarray(temperature, dtype=np.float64)
        b = np.asarray(char_fraction, dtype=np.float64)
        virgin = self.virgin_intercept + self.virgin_slope * t
        char = self.char_intercept + self.char_slope * t
        return np.asarray((1.0 - b) * virgin + b * char)

    def d_temperature(self, temperature: ArrayLike, char_fraction: ArrayLike) -> _FloatArray:
        t = np.asarray(temperature, dtype=np.float64)
        b = np.asarray(char_fraction, dtype=np.float64)
        return np.asarray(
            (1.0 - b) * self.virgin_slope + b * self.char_slope + 0.0 * t
        )

    def d_char_fraction(self, temperature: ArrayLike, char_fraction: ArrayLike) -> _FloatArray:
        t = np.asarray(temperature, dtype=np.float64)
        b = np.asarray(char_fraction, dtype=np.float64)
        return np.asarray(
            (self.char_intercept + self.char_slope * t)
            - (self.virgin_intercept + self.virgin_slope * t)
            + 0.0 * b
        )


@dataclass(frozen=True)
class CharringMaterial:
    """Complete charring-ablator model consumed by the thermal solver.

    Attributes
    ----------
    resin_a, resin_b, filler:
        The three Arrhenius components (A, B resin; C filler).
    resin_fraction:
        Resin volume fraction :math:`\\Gamma \\in (0, 1)`.
    conductivity:
        :math:`k(T, \\beta)` (W/(m K)).
    specific_heat:
        :math:`c_p(T, \\beta)` (J/(kg K)).
    gas_specific_heat:
        Pyrolysis-gas :math:`c_{p_g}` (J/(kg K)), constant.
    gas_enthalpy_offset, gas_enthalpy_slope:
        Pyrolysis-gas enthalpy :math:`h_g(T) = h_{g,0} + c\\,T` (J/kg).
    solid_enthalpy_offset, solid_enthalpy_slope:
        Solid enthalpy :math:`\\bar h(T) = \\bar h_0 + c\\,T` (J/kg).
    emissivity_virgin, emissivity_char:
        Surface emissivity endpoints, blended by :math:`\\beta`.
    """

    resin_a: ArrheniusComponent
    resin_b: ArrheniusComponent
    filler: ArrheniusComponent
    resin_fraction: float
    conductivity: LinearBlendProperty
    specific_heat: LinearBlendProperty
    gas_specific_heat: float
    gas_enthalpy_offset: float
    gas_enthalpy_slope: float
    solid_enthalpy_offset: float
    solid_enthalpy_slope: float
    emissivity_virgin: float
    emissivity_char: float

    def __post_init__(self) -> None:
        if not 0.0 < self.resin_fraction < 1.0:
            raise ValueError(f"resin_fraction must be in (0, 1), got {self.resin_fraction}")
        if not (np.isfinite(self.gas_specific_heat) and self.gas_specific_heat > 0.0):
            raise ValueError(f"gas_specific_heat must be > 0, got {self.gas_specific_heat}")
        for name in ("emissivity_virgin", "emissivity_char"):
            val = getattr(self, name)
            if not 0.0 < val <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {val}")

    @property
    def components(self) -> tuple[ArrheniusComponent, ArrheniusComponent, ArrheniusComponent]:
        return (self.resin_a, self.resin_b, self.filler)

    @property
    def virgin_bulk_density(self) -> float:
        """Bulk density of fully virgin material."""
        g = self.resin_fraction
        return g * (self.resin_a.virgin_density + self.resin_b.virgin_density) + (
            1.0 - g
        ) * self.filler.virgin_density

    @property
    def char_bulk_density(self) -> float:
        """Bulk density of fully charred material."""
        g = self.resin_fraction
        return g * (self.resin_a.char_density + self.resin_b.char_density) + (
            1.0 - g
        ) * self.filler.char_density

    def gas_enthalpy(self, temperature: ArrayLike) -> _FloatArray:
        return np.asarray(
            self.gas_enthalpy_offset
            + self.gas_enthalpy_slope * np.asarray(temperature, dtype=np.float64)
        )

    def solid_enthalpy(self, temperature: ArrayLike) -> _FloatArray:
        return np.asarray(
            self.solid_enthalpy_offset
            + self.solid_enthalpy_slope * np.asarray(temperature, dtype=np.float64)
        )

    def emissivity(self, char_fraction: ArrayLike) -> _FloatArray:
        b = np.asarray(char_fraction, dtype=np.float64)
        return np.asarray((1.0 - b) * self.emissivity_virgin + b * self.emissivity_char)


def demo_material() -> CharringMaterial:
    """Synthetic charring ablator for verification exercises.

    Magnitudes are generic phenolic-class values assembled from open
    textbook ranges; the material corresponds to no real system and
    exists so the solver can be exercised and verified without any
    proprietary database. Kinetic parameters are chosen so the three
    components decompose over distinct, overlapping temperature bands.
    """
    return CharringMaterial(
        resin_a=ArrheniusComponent(
            pre_exponential=1.0e4,
            activation_energy=7.0e4,
            reaction_order=1.0,
            virgin_density=350.0,
            char_density=200.0,
        ),
        resin_b=ArrheniusComponent(
            pre_exponential=5.0e6,
            activation_energy=1.2e5,
            reaction_order=2.0,
            virgin_density=150.0,
            char_density=50.0,
        ),
        filler=ArrheniusComponent(
            pre_exponential=2.0e3,
            activation_energy=9.0e4,
            reaction_order=1.0,
            virgin_density=1100.0,
            char_density=900.0,
        ),
        resin_fraction=0.5,
        conductivity=LinearBlendProperty(0.35, 2.0e-4, 0.60, 4.0e-4),
        specific_heat=LinearBlendProperty(1200.0, 0.25, 1600.0, 0.15),
        gas_specific_heat=1800.0,
        gas_enthalpy_offset=-1.5e6,
        gas_enthalpy_slope=1800.0,
        solid_enthalpy_offset=-8.0e5,
        solid_enthalpy_slope=1300.0,
        emissivity_virgin=0.80,
        emissivity_char=0.90,
    )
