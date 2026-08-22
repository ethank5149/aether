"""In-depth pore pressure from Darcy flow of pyrolysis gas.

The framework carries a *pressure-dependent* conductivity for PICA — the
MEDLI2 model of :mod:`aether.fiat.materials`, which changes by a
factor of 4.1 between 1 atm and 0.001 atm — and until now handed it the
**boundary-layer edge pressure** at every depth. That is the wrong
pressure. The conductivity is a function of the gas pressure *in the
pores*, and pore pressure is set by pyrolysis gas fighting its way out
through the material's own permeability. It peaks inside the pyrolysis
zone and can exceed surface pressure substantially; Ahn et al., via
Rabinovitch §IV.A, put it at "a factor of two or higher" for the Pioneer
Venus probe.

The governing form is quadratic in pressure
-------------------------------------------

Steady compressible Darcy flow through a porous slab, with the gas obeying
the ideal law :math:`\\rho_g = PM/(RT)`:

.. math::

    \\dot m = -\\frac{\\rho_g K}{\\mu}\\frac{\\mathrm{d}P}{\\mathrm{d}x}
            = -\\frac{P M K}{R T \\mu}\\frac{\\mathrm{d}P}{\\mathrm{d}x}
    \\;\\Longrightarrow\\;
    \\frac{\\mathrm{d}(P^2)}{\\mathrm{d}x} = -\\frac{2 \\dot m R T \\mu}{M K}.

Solving for :math:`P^2` rather than :math:`P` is not a trick — it is the
natural variable, because the gas density is itself proportional to
pressure and the two factors combine. Integrating inward from the surface,

.. math::

    P^2(x) = P_s^2 + \\int_0^x \\frac{2 \\dot m(x') R T(x') \\mu}{M K}
             \\,\\mathrm{d}x',

with :math:`\\dot m(x)` the accumulated gas mass flux, which is zero at the
back face and grows as the flow gathers everything decomposing above it.

The permeability, which is now measured
---------------------------------------

This module was originally written around a *missing* number. Park &
Lawrence's measurements (in ``reference/``) are for MX4926 carbon *cloth*
phenolic — a dense rocket nozzle liner near 1.45 g/cm³ — and give
:math:`10^{-17}` to :math:`10^{-21}` m²; using a nozzle-liner value for a
90 %-porous preform would be wrong in the direction that matters most. So
permeability was an explicit argument with no default, and
:func:`pore_pressure_sensitivity` existed to answer the question that
absence raises: *does it change anything?*

The number has since arrived. Marschall & Milos measure **FiberForm**, the
carbon preform PICA is made from, and
:mod:`aether.fiat.permeability` carries their table — including
why the preform rather than the composite is the right material for *this*
calculation, since the transport path that sets pore pressure runs outward
through resin-free char.

**The measured value is 79 to 550 × 10⁻¹² m², two to three orders of
magnitude above where the sweep's lower bound sat.** The old placeholder
bracket of :math:`10^{-12}`–:math:`10^{-10}` m², inferred from porosity,
had roughly the right top end and was 79× too tight at the bottom.

The measured answer: it is real, and it is smaller than the sweep suggested
--------------------------------------------------------------------------

Re-run against real pyrolysis profiles from the solver on two Milos & Chen
arcjet conditions, at the measured permeability rather than a swept one:

.. code-block:: text

    surface p    orientation    K_0 (m^2)   peak P/Ps   virgin k error
    27.3 kPa     transverse     1.16e-10        1.020          0.07 %
    27.3 kPa     in-plane       5.14e-10        1.004          0.02 %
     2.3 kPa     transverse     1.16e-10        1.793          2.43 %
     2.3 kPa     in-plane       5.14e-10        1.224          0.84 %

**The worst case is 2.4 %**, against the 8.4 % the speculative bracket
produced — and that is the *continuum* figure. Including Klinkenberg slip
(:func:`~aether.fiat.permeability.effective_permeability`) cuts the
2.3 kPa transverse ratio further, 1.79 → 1.39, because pyrolysis gas is
light and hot and therefore deep in the slip regime: its slip parameter is
around 11 kPa at 2000 K, so permeability is more than doubled anywhere the
pore pressure is below that. Rarefaction lets gas escape more easily than
Darcy alone allows, so neglecting it is conservative.

Three things bound the error. The MEDLI2 model interpolates in *log*
pressure, so even a large pressure ratio is a fraction of one decade out of
the three its anchors span. It **clamps** at the 1 atm anchor rather than
extrapolating — a decision taken for unrelated reasons, which caps this
error too. And the measured permeability is high enough that the pressure
ratio itself never gets far from unity at flight-relevant pressures.

**This module is therefore a diagnostic, not a correction, and is
deliberately not wired into the solver.** That decision was originally
made to avoid injecting an *unmeasured* parameter into the main solve path.
The measurement has since removed that objection but strengthened the
conclusion it was protecting: the effect is 2.4 % at worst, against 27 %
experimental scatter in the recession data the comparison lives inside.
Wiring it in would add a Darcy solve per step to move an answer by a tenth
of its own uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "PORE_PRESSURE_REFERENCES",
    "PorePressureProfile",
    "pore_pressure",
    "pore_pressure_sensitivity",
]

_FloatArray = NDArray[np.float64]

#: Universal gas constant (J/mol/K).
_R_UNIVERSAL = 8.314462618

#: Measured permeabilities, with the material each belongs to.
#:
#: The FiberForm entry is the one that applies to PICA, and it is a
#: measurement. The other two are kept for contrast: Park & Lawrence
#: because it is the wrong material and was once mistaken for the right
#: one, and the retired placeholder because the gap between an inferred
#: bracket and the measurement that replaced it is worth being able to see.
PORE_PRESSURE_REFERENCES: dict[str, tuple[float, float, str]] = {
    "FiberForm (PICA carbon preform)": (
        7.91e-11,
        5.49e-10,
        "Marschall & Milos, JTHT 12(4) 1998, Table 1. MEASURED, room "
        "temperature air, four specimens per orientation over 145-161 "
        "kg/m3. Transverse (through-thickness, the heatshield direction) "
        "spans 7.9e-11 to 3.6e-10; in-plane 5.0e-10 to 5.5e-10. K_0 "
        "uncertainty +11%/-16%. See aether.fiat.permeability.",
    ),
    "MX4926 carbon cloth phenolic": (
        1e-21,
        1e-17,
        "Park & Lawrence, AIAA 2003-5242, measured 22-260 C. Dense RSRM "
        "nozzle liner near 1.45 g/cm3 -- NOT PICA, and not a substitute "
        "for it. Retained as a counterexample: it is a real measurement of "
        "the wrong material, which is the failure mode this entry exists "
        "to make visible.",
    ),
    "PICA (retired placeholder bracket)": (
        1e-12,
        1e-10,
        "SUPERSEDED by the FiberForm entry above. An order-of-magnitude "
        "bracket inferred from porosity, used only to sweep before the "
        "measurement was held. Its top end was within 5.5x; its bottom end "
        "was 79x too tight, which is why the swept worst case (8.4% "
        "conductivity error) overstated the measured one (2.4%).",
    ),
}


@dataclass(frozen=True)
class PorePressureProfile:
    """Pore pressure through the stack, and where it peaks."""

    depth: _FloatArray
    """Distance from the heated surface (m), increasing inward."""
    pressure: _FloatArray
    """Pore gas pressure (Pa) at each depth."""
    surface_pressure: float
    permeability: float

    @property
    def peak_pressure(self) -> float:
        return float(np.max(self.pressure))

    @property
    def peak_ratio(self) -> float:
        """Peak pore pressure over surface pressure.

        The single number that says whether feeding surface pressure to a
        pressure-dependent property is defensible. At 1.0 it is exact; the
        further above, the worse the substitution.
        """
        return self.peak_pressure / self.surface_pressure

    @property
    def peak_depth(self) -> float:
        return float(self.depth[int(np.argmax(self.pressure))])


def pore_pressure(
    depth: ArrayLike,
    gas_source: ArrayLike,
    temperature: ArrayLike,
    surface_pressure: float,
    permeability: float,
    viscosity: float = 4.0e-5,
    molar_mass: float = 0.0103,
) -> PorePressureProfile:
    """Pore pressure profile from a pyrolysis gas source distribution.

    Parameters
    ----------
    depth:
        Distance from the heated surface (m), increasing inward and
        strictly increasing. ``depth[0]`` is the surface.
    gas_source:
        Volumetric pyrolysis gas generation rate (kg/m³/s) at each depth,
        i.e. :math:`-\\partial \\rho / \\partial t` from the solid.
    temperature:
        Local temperature (K) at each depth.
    surface_pressure:
        Boundary-layer edge pressure (Pa), the outflow boundary condition.
    permeability:
        Darcy permeability (m²). **No default** — see the module docstring
        for why, and :data:`PORE_PRESSURE_REFERENCES` for what is and is
        not measured.
    viscosity:
        Pyrolysis gas dynamic viscosity (Pa s). The default is a
        representative high-temperature value; the result depends on it
        only linearly and through the same combination as permeability, so
        the two cannot be separated by this model.
    molar_mass:
        Gas molar mass (kg/mol). The default is what the equilibrium solve
        returns for PICA pyrolysis gas at 2000 K, which is light because
        the mixture is mostly H2.

    Notes
    -----
    Quasi-steady: the gas is assumed to leave as fast as it is made, with
    no storage term. That is the standard treatment and it is good wherever
    the pyrolysis front moves slowly against the gas transit time, which is
    everywhere except the first instants of a step heat load.
    """
    x = np.asarray(depth, dtype=np.float64)
    source = np.asarray(gas_source, dtype=np.float64)
    temp = np.asarray(temperature, dtype=np.float64)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("depth must be a 1-D array with at least two points")
    if source.shape != x.shape or temp.shape != x.shape:
        raise ValueError(
            f"gas_source and temperature must match depth: got "
            f"{source.shape} and {temp.shape} against {x.shape}"
        )
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("depth must be strictly increasing from the surface")
    if np.any(source < 0.0):
        raise ValueError(
            "gas_source must be non-negative; a negative source is gas being "
            "absorbed by the solid, which pyrolysis does not do"
        )
    if not (np.isfinite(surface_pressure) and surface_pressure > 0.0):
        raise ValueError(f"surface_pressure must be finite and > 0, got {surface_pressure}")
    for name, value in (
        ("permeability", permeability),
        ("viscosity", viscosity),
        ("molar_mass", molar_mass),
    ):
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be finite and > 0, got {value}")

    # Accumulated mass flux at each depth: everything generated *below*
    # this point has already passed through it on its way out, so the flux
    # is the reverse cumulative integral of the source. Zero at the back
    # face by construction, which is the impermeable-backface condition.
    widths = np.diff(x)
    generated = 0.5 * (source[:-1] + source[1:]) * widths
    # The zero belongs at the *back* face, not the surface: nothing has
    # passed through the deepest node, and the flux grows outward as the
    # flow gathers everything decomposing behind it. Putting it at the
    # surface instead inverts the profile.
    flux = np.concatenate([np.cumsum(generated[::-1])[::-1], [0.0]])

    # Integrate d(P^2)/dx inward. The integrand is evaluated at midpoints
    # and accumulated, so P^2 is exact for a piecewise-linear source.
    coefficient = 2.0 * _R_UNIVERSAL * viscosity / (molar_mass * permeability)
    integrand = coefficient * flux * temp
    midpoint_integrand = 0.5 * (integrand[:-1] + integrand[1:])
    squared = np.concatenate(
        [[surface_pressure**2], surface_pressure**2 + np.cumsum(midpoint_integrand * widths)]
    )
    return PorePressureProfile(
        depth=x,
        pressure=np.sqrt(squared),
        surface_pressure=float(surface_pressure),
        permeability=float(permeability),
    )


def pore_pressure_sensitivity(
    depth: ArrayLike,
    gas_source: ArrayLike,
    temperature: ArrayLike,
    surface_pressure: float,
    permeabilities: ArrayLike,
    **kwargs: float,
) -> dict[float, float]:
    """Peak-to-surface pressure ratio against permeability.

    The question a missing measurement actually raises is not "what is the
    number" but "does the answer depend on it". Returns the peak ratio for
    each permeability, so the sweep can be read directly: a ratio near 1
    everywhere means feeding surface pressure to a pressure-dependent
    property was harmless, and a ratio that climbs means it was not.
    """
    values = np.atleast_1d(np.asarray(permeabilities, dtype=np.float64))
    return {
        float(k): pore_pressure(
            depth, gas_source, temperature, surface_pressure, float(k), **kwargs
        ).peak_ratio
        for k in values
    }
