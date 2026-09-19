"""Three-component Arrhenius decomposition kinetics (Paper I, Eq. 3.14).

Each component decomposes in the material frame by

.. math::

    \\left.\\frac{\\partial \\rho_i}{\\partial t}\\right|_{y}
    = -A_i\\,\\rho_{v,i}
      \\left(\\frac{\\rho_i - \\rho_{c,i}}{\\rho_{v,i}}\\right)^{n_i}
      \\exp\\!\\left(-\\frac{E_i}{\\mathcal{R} T}\\right),

which stops identically at the char density. Densities that fall to (or
by transient integrator overshoot, marginally below) :math:`\\rho_{c,i}`
produce exactly zero rate rather than a NaN from a fractional power of a
negative number — decomposition is irreversible and the char state is
absorbing.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aether.thermal.material import GAS_CONSTANT, ArrheniusComponent, CharringMaterial

__all__ = ["bulk_density", "decomposition_rate", "degree_of_char"]

_FloatArray = NDArray[np.float64]


def decomposition_rate(
    component: ArrheniusComponent,
    density: ArrayLike,
    temperature: ArrayLike,
) -> _FloatArray:
    """Material-frame density rate :math:`\\partial\\rho_i/\\partial t|_y`
    (negative or zero), vectorized over nodes.

    Parameters
    ----------
    component:
        Kinetic parameters for this component.
    density:
        Current partial density :math:`\\rho_i` (kg/m³).
    temperature:
        Temperature (K), strictly positive.
    """
    rho = np.asarray(density, dtype=np.float64)
    t = np.asarray(temperature, dtype=np.float64)
    if np.any(t <= 0.0):
        raise ValueError("temperature must be strictly positive for Arrhenius kinetics")
    extent = (rho - component.char_density) / component.virgin_density
    extent = np.maximum(extent, 0.0)  # char state is absorbing; no NaN from overshoot
    rate = (
        -component.pre_exponential
        * component.virgin_density
        * extent**component.reaction_order
        * np.exp(-component.activation_energy / (GAS_CONSTANT * t))
    )
    return np.asarray(rate)


def bulk_density(
    material: CharringMaterial,
    rho_a: ArrayLike,
    rho_b: ArrayLike,
    rho_c: ArrayLike,
) -> _FloatArray:
    """Bulk density :math:`\\rho = \\Gamma(\\rho_A + \\rho_B) +
    (1 - \\Gamma)\\rho_C`."""
    g = material.resin_fraction
    return np.asarray(
        g * (np.asarray(rho_a, dtype=np.float64) + np.asarray(rho_b, dtype=np.float64))
        + (1.0 - g) * np.asarray(rho_c, dtype=np.float64)
    )


def degree_of_char(material: CharringMaterial, rho_bulk: ArrayLike) -> _FloatArray:
    """Degree of char :math:`\\beta = (\\rho_v - \\rho)/(\\rho_v - \\rho_c)`,
    clipped to :math:`[0, 1]` against integrator overshoot."""
    rho = np.asarray(rho_bulk, dtype=np.float64)
    beta = (material.virgin_bulk_density - rho) / (
        material.virgin_bulk_density - material.char_bulk_density
    )
    return np.asarray(np.clip(beta, 0.0, 1.0))
