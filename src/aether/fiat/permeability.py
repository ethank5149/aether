"""Measured gas permeability of rigid fibrous insulations.

Implements the model and data of

* J. Marschall and F. S. Milos, "Gas Permeability of Rigid Fibrous
  Refractory Insulations," *J. Thermophysics and Heat Transfer* **12**(4),
  1998, pp. 528-535, doi:10.2514/2.6372.

This module exists to replace an assumption. :mod:`aether.fiat.pore_pressure`
previously took permeability as an argument with no default, because the
value for PICA was not held here and inventing one would have been worse
than sweeping. Marschall & Milos measure **FiberForm**, the commercial
carbon-fibre preform that PICA is made from, and that measurement is the
right one to use -- for a reason worth stating rather than assuming.

Why FiberForm is the correct material, not an approximation to it
----------------------------------------------------------------

PICA is FiberForm impregnated with phenolic resin. The virgin composite is
therefore *less* permeable than the bare preform, and quoting FiberForm for
virgin PICA would be optimistic. But the pressure that
:mod:`~aether.fiat.pore_pressure` solves for is set by the
resistance pyrolysis gas meets **on its way out**, and that path runs
outward through the char. The char is resin-free carbon preform: the resin
that used to fill its pores is precisely the gas now flowing through them.
So FiberForm is not a stand-in for the charred transport path -- it *is*
that path, and the virgin material below contributes little because the gas
generated there must cross the char regardless.

The remaining approximation is that charring does not otherwise alter the
preform microstructure. Marschall & Milos do not test that, and this module
does not claim it.

Klinkenberg slip, which is not a small correction here
-----------------------------------------------------

Effective permeability is pressure dependent (their Eq. 2):

.. math:: K(P) = K_0 \\left[1 + b/P\\right]

with :math:`K_0` the continuum-flow permeability and :math:`b` a slip
parameter. The characteristic pore size in these materials is microns to
tens of microns, so the local Knudsen number reaches the slip regime
(:math:`Kn \\sim 0.1`-:math:`1`) at low pressure or high temperature -- both
of which describe in-depth flow during entry. At 1 kPa with FiberForm's
:math:`b \\approx 700` Pa the effective permeability is already 1.7 times
the continuum value, and the ratio grows as the gas heats.

The slip parameter scales with gas and temperature (their Eq. 5):

.. math::

    \\frac{b(T, M, \\mu)}{b_\\mathrm{ref}}
      = \\frac{\\mu}{\\mu_\\mathrm{ref}}
        \\sqrt{\\frac{T}{T_\\mathrm{ref}} \\frac{M_\\mathrm{ref}}{M}}

Both halves of that relation are checked in this repository against the
authors' own experiments -- helium against air for the gas dependence, and
a 293-1200 K furnace series for the temperature dependence. See
:func:`slip_parameter` and the tests that accompany it.

:math:`K_0` itself is *not* a function of temperature or gas: the authors
demonstrate it directly (their Table 2, ratios within 5 % of unity over
293-1200 K) and it is a purely geometric property of the microstructure.

Anisotropy is a design decision, not a scatter band
---------------------------------------------------

These materials are made by pressing a fibre slurry, so the fibres align
preferentially normal to the pressing axis. Permeability along the pressing
axis ("transverse", also called through-thickness) is markedly lower than
in-plane. For FiberForm the ratio is about **6**, far stronger than the 1.25
to 1.5 seen in the ceramic tiles, because its 15 micron carbon fibres make a
coarser and more directional structure.

A heatshield is cut with the through-thickness direction pointing out
through the surface, which is the direction pyrolysis gas must travel. The
conservative choice for a heatshield is therefore the *transverse* value,
and that is what :func:`fiberform_permeability` returns by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "AIR_MOLAR_MASS",
    "AIR_REFERENCE_TEMPERATURE",
    "AIR_REFERENCE_VISCOSITY",
    "FIBERFORM_SAMPLES",
    "MARSCHALL_MILOS_SAMPLES",
    "PERMEABILITY_UNCERTAINTY",
    "SLIP_PARAMETER_UNCERTAINTY",
    "Orientation",
    "PermeabilitySample",
    "effective_permeability",
    "fiberform_permeability",
    "knudsen_regime_pressure",
    "slip_parameter",
]

_FloatArray = NDArray[np.float64]

Orientation = Literal["transverse", "in-plane"]

#: Fractional uncertainty on :math:`K_0`, as (low, high) -- Marschall &
#: Milos give +11 %/-16 % for every sample in their Table 1.
PERMEABILITY_UNCERTAINTY: tuple[float, float] = (-0.16, +0.11)

#: Fractional uncertainty on :math:`b`, as (low, high): +7 %/-4 %.
SLIP_PARAMETER_UNCERTAINTY: tuple[float, float] = (-0.04, +0.07)

#: Air properties at the conditions of their room-temperature measurements.
#: Their measurements span 285-300 K; they state 290 K when computing the
#: helium comparison, and this is the reference the tabulated ``b`` values
#: belong to.
AIR_REFERENCE_TEMPERATURE = 290.0
AIR_REFERENCE_VISCOSITY = 18.05e-6
AIR_MOLAR_MASS = 0.02897


@dataclass(frozen=True)
class PermeabilitySample:
    """One measured specimen from Marschall & Milos Table 1.

    Attributes
    ----------
    material:
        Insulation trade name.
    density:
        Specimen bulk density (kg/m^3). Measured per specimen rather than
        taken from the nominal tile density, because inter- and intra-billet
        variation is significant -- and for FiberForm it is dramatic: an
        11 % density spread moves the transverse permeability by 4.5x.
    orientation:
        ``"transverse"`` (along the pressing axis, i.e. through-thickness)
        or ``"in-plane"``.
    continuum_permeability:
        :math:`K_0` (m^2), the permeability in the continuum-flow limit.
    slip_parameter:
        :math:`b` (Pa) for air at room temperature.
    """

    material: str
    density: float
    orientation: Orientation
    continuum_permeability: float
    slip_parameter: float


def _sample(
    material: str,
    density: float,
    orientation: Orientation,
    k0_e12: float,
    b: float,
) -> PermeabilitySample:
    return PermeabilitySample(
        material=material,
        density=density,
        orientation=orientation,
        continuum_permeability=k0_e12 * 1e-12,
        slip_parameter=b,
    )


#: Marschall & Milos Table 1 in full, transcribed as published.
#:
#: Retained whole rather than reduced to a fit, because the within-material
#: spread is itself information: it is what specimen-to-specimen density
#: variation does, and it is larger than any uncertainty band quoted on a
#: single measurement.
MARSCHALL_MILOS_SAMPLES: tuple[PermeabilitySample, ...] = (
    # LI-900 (silica, nominal 144 kg/m^3)
    _sample("LI-900", 127.0, "transverse", 21.5, 3910.0),
    _sample("LI-900", 130.0, "transverse", 19.6, 4200.0),
    _sample("LI-900", 152.0, "transverse", 12.0, 4940.0),
    _sample("LI-900", 136.0, "in-plane", 29.8, 3420.0),
    _sample("LI-900", 137.0, "in-plane", 29.0, 3690.0),
    _sample("LI-900", 139.0, "in-plane", 30.2, 3400.0),
    # LI-2200 (silica, nominal 352 kg/m^3)
    _sample("LI-2200", 366.0, "transverse", 2.90, 7640.0),
    _sample("LI-2200", 370.0, "transverse", 2.86, 7940.0),
    _sample("LI-2200", 373.0, "transverse", 2.86, 7230.0),
    _sample("LI-2200", 380.0, "in-plane", 3.59, 6950.0),
    _sample("LI-2200", 386.0, "in-plane", 3.03, 7420.0),
    _sample("LI-2200", 390.0, "in-plane", 3.01, 7430.0),
    # AIM-18 (silica, nominal 288 kg/m^3) -- transverse only
    _sample("AIM-18", 295.0, "transverse", 4.63, 5970.0),
    _sample("AIM-18", 297.0, "transverse", 4.32, 6850.0),
    _sample("AIM-18", 306.0, "transverse", 3.86, 7110.0),
    _sample("AIM-18", 314.0, "transverse", 4.12, 6750.0),
    # FRCI-12 (silica + Nextel, nominal 192 kg/m^3)
    _sample("FRCI-12", 197.0, "transverse", 14.8, 4200.0),
    _sample("FRCI-12", 206.0, "transverse", 22.6, 3620.0),
    _sample("FRCI-12", 213.0, "transverse", 14.3, 4160.0),
    _sample("FRCI-12", 219.0, "transverse", 13.3, 4440.0),
    _sample("FRCI-12", 217.0, "in-plane", 20.1, 3740.0),
    _sample("FRCI-12", 217.0, "in-plane", 20.0, 3740.0),
    # AETB-12 (silica + alumina + Nextel, nominal 192 kg/m^3)
    _sample("AETB-12", 178.0, "transverse", 31.5, 3120.0),
    _sample("AETB-12", 182.0, "transverse", 28.8, 3260.0),
    _sample("AETB-12", 187.0, "transverse", 23.5, 3650.0),
    _sample("AETB-12", 196.0, "transverse", 20.3, 3590.0),
    _sample("AETB-12", 159.0, "in-plane", 62.9, 2390.0),
    _sample("AETB-12", 184.0, "in-plane", 47.0, 2550.0),
    _sample("AETB-12", 187.0, "in-plane", 33.7, 3260.0),
    _sample("AETB-12", 235.0, "in-plane", 35.4, 2850.0),
    # FiberForm (carbon, 152-176 kg/m^3) -- the PICA preform
    _sample("FiberForm", 145.0, "transverse", 359.0, 703.0),
    _sample("FiberForm", 149.0, "transverse", 323.0, 877.0),
    _sample("FiberForm", 157.0, "transverse", 82.3, 1100.0),
    _sample("FiberForm", 161.0, "transverse", 79.1, 1150.0),
    _sample("FiberForm", 149.0, "in-plane", 528.0, 784.0),
    _sample("FiberForm", 152.0, "in-plane", 549.0, 603.0),
    _sample("FiberForm", 155.0, "in-plane", 514.0, 662.0),
    _sample("FiberForm", 157.0, "in-plane", 498.0, 725.0),
)

#: The FiberForm subset -- the only material here relevant to PICA.
FIBERFORM_SAMPLES: tuple[PermeabilitySample, ...] = tuple(
    s for s in MARSCHALL_MILOS_SAMPLES if s.material == "FiberForm"
)


def slip_parameter(
    reference: float,
    *,
    temperature: float,
    viscosity: float,
    molar_mass: float,
    reference_temperature: float = AIR_REFERENCE_TEMPERATURE,
    reference_viscosity: float = AIR_REFERENCE_VISCOSITY,
    reference_molar_mass: float = AIR_MOLAR_MASS,
) -> float:
    """Scale a measured slip parameter to another gas and temperature.

    Marschall & Milos Eq. (5):

    .. math::

        b = b_\\mathrm{ref}\\,\\frac{\\mu}{\\mu_\\mathrm{ref}}
            \\sqrt{\\frac{T\\,M_\\mathrm{ref}}{T_\\mathrm{ref}\\,M}}

    The lighter and hotter the gas, the larger ``b``, and therefore the
    larger the slip enhancement at a given pressure. Pyrolysis gas is both:
    it is dominated by hydrogen and light carbon species at thousands of
    kelvin, so ``b`` is far above the tabulated room-temperature air value
    and slip is correspondingly more important in service than in the
    laboratory.

    Parameters
    ----------
    reference:
        Measured ``b`` (Pa) at the reference gas and temperature.
    temperature, viscosity, molar_mass:
        Target gas state: K, N s/m^2, kg/mol.
    reference_temperature, reference_viscosity, reference_molar_mass:
        The conditions ``reference`` was measured at. Default to the
        authors' room-temperature air.

    Returns
    -------
    float
        ``b`` (Pa) at the target conditions.
    """
    if reference < 0.0:
        msg = f"slip parameter must be non-negative, got {reference}"
        raise ValueError(msg)
    for name, value in (
        ("temperature", temperature),
        ("viscosity", viscosity),
        ("molar_mass", molar_mass),
        ("reference_temperature", reference_temperature),
        ("reference_viscosity", reference_viscosity),
        ("reference_molar_mass", reference_molar_mass),
    ):
        if value <= 0.0:
            msg = f"{name} must be positive, got {value}"
            raise ValueError(msg)

    viscosity_ratio = viscosity / reference_viscosity
    thermal_ratio = np.sqrt(
        (temperature * reference_molar_mass) / (reference_temperature * molar_mass)
    )
    return float(reference * viscosity_ratio * thermal_ratio)


def effective_permeability(
    continuum: float,
    slip: float,
    pressure: ArrayLike,
) -> _FloatArray:
    """Klinkenberg effective permeability :math:`K_0[1 + b/P]`.

    Marschall & Milos Eq. (2). The correction is *always* an enhancement:
    rarefaction lets gas slip along pore walls rather than sticking, so a
    porous medium passes more flow at low pressure than Darcy alone
    predicts. Ignoring it therefore over-predicts pore pressure.

    Parameters
    ----------
    continuum:
        :math:`K_0` (m^2).
    slip:
        :math:`b` (Pa) at the gas and temperature of interest -- pass the
        result of :func:`slip_parameter` if that differs from the
        measurement conditions.
    pressure:
        Local gas pressure (Pa), scalar or array.

    Returns
    -------
    numpy.ndarray
        Effective permeability (m^2), broadcast against ``pressure``.
    """
    if continuum <= 0.0:
        msg = f"continuum permeability must be positive, got {continuum}"
        raise ValueError(msg)
    if slip < 0.0:
        msg = f"slip parameter must be non-negative, got {slip}"
        raise ValueError(msg)

    p = np.atleast_1d(np.asarray(pressure, dtype=np.float64))
    if np.any(p <= 0.0):
        msg = "pressure must be positive everywhere"
        raise ValueError(msg)
    return np.asarray(continuum * (1.0 + slip / p), dtype=np.float64)


def fiberform_permeability(
    density: float = 155.0,
    orientation: Orientation = "transverse",
) -> tuple[float, float]:
    """Continuum permeability and slip parameter for FiberForm.

    Interpolates the four measured specimens of the requested orientation
    against density, in :math:`\\log K_0` -- the measured variation is
    multiplicative and steep (a factor of 4.5 across an 11 % density change
    in the transverse direction), so a linear interpolation of :math:`K_0`
    itself would be badly biased. ``b`` varies far more gently and is
    interpolated directly.

    Outside the measured density range the endpoint values are held rather
    than extrapolated. Extrapolating a fit this steep past its data would
    manufacture precision that the four points do not support.

    Parameters
    ----------
    density:
        Bulk density (kg/m^3). Defaults to 155, mid-range of the measured
        FiberForm specimens.
    orientation:
        ``"transverse"`` (default -- the through-thickness direction a
        heatshield is cut for, and the lower of the two) or ``"in-plane"``.

    Returns
    -------
    tuple[float, float]
        ``(K_0, b)`` in m^2 and Pa, with ``b`` at room-temperature air.

    Notes
    -----
    The transverse specimens split into two clear groups -- 359 and 323
    at 145-149 kg/m^3, then 82.3 and 79.1 at 157-161 -- so the
    interpolation across that gap is doing real work and should be treated
    as the coarse estimate it is. The published +11 %/-16 % applies to a
    single specimen, not to this interpolation.
    """
    samples = sorted(
        (s for s in FIBERFORM_SAMPLES if s.orientation == orientation),
        key=lambda s: s.density,
    )
    if not samples:
        msg = f"unknown orientation {orientation!r}"
        raise ValueError(msg)

    densities = np.array([s.density for s in samples])
    log_k0 = np.log(np.array([s.continuum_permeability for s in samples]))
    slips = np.array([s.slip_parameter for s in samples])

    k0 = float(np.exp(np.interp(density, densities, log_k0)))
    b = float(np.interp(density, densities, slips))
    return k0, b


def knudsen_regime_pressure(slip: float, enhancement: float = 2.0) -> float:
    """Pressure at which slip inflates permeability by ``enhancement``.

    From :math:`K/K_0 = 1 + b/P`, the pressure is :math:`b/(K/K_0 - 1)`.
    Above it the flow is effectively continuum; below it, rarefaction
    dominates and a Darcy solve using :math:`K_0` alone will over-predict
    pore pressure.

    Parameters
    ----------
    slip:
        :math:`b` (Pa) at the conditions of interest.
    enhancement:
        Target :math:`K/K_0`, greater than 1. Default 2, i.e. the pressure
        at which slip doubles the permeability.

    Returns
    -------
    float
        Pressure (Pa).
    """
    if slip <= 0.0:
        msg = f"slip parameter must be positive, got {slip}"
        raise ValueError(msg)
    if enhancement <= 1.0:
        msg = f"enhancement must exceed 1, got {enhancement}"
        raise ValueError(msg)
    return float(slip / (enhancement - 1.0))
