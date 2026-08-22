"""Hypersonic aerothermodynamic correlations (Paper II, §4).

Stagnation convective heating by Fay–Riddell with the Lewis exponent
stated (Eq. 4.1), the modified-Newtonian stagnation velocity gradient
(Eq. 4.2), the Sutton–Graves correlation kept explicitly distinct as the
screening fallback, Tauber–Sutton shock-layer radiation with its
tabulated velocity function as a required input (Eq. 4.4), the Lees
distribution over the planform with stagnation-region matching
(Eq. 4.5), and the single-temperature recession balance for
non-pyrolyzing leading edges (Eq. 4.6). All quantities SI.
"""

from __future__ import annotations

from aether.aerothermal.distribution import lees_distribution
from aether.aerothermal.radiative import TauberSuttonRadiation
from aether.aerothermal.recession import stefan_recession_rate
from aether.aerothermal.stagnation import (
    FAY_RIDDELL_COEFFICIENT_EXACT_094,
    FAY_RIDDELL_COEFFICIENT_FROM_NUSSELT,
    FAY_RIDDELL_COEFFICIENT_LITERATURE,
    FAY_RIDDELL_COEFFICIENT_SOURCE,
    FAY_RIDDELL_FACTOR_PR071,
    FAY_RIDDELL_NUSSELT_COEFFICIENT,
    WallCatalycity,
    catalycity_bracket,
    fay_riddell,
    newtonian_velocity_gradient,
    sutton_graves,
)
from aether.aerothermal.tauber_sutton import (
    EARTH_EXPONENT_CAPS,
    EARTH_VELOCITY_FUNCTION,
    MARS_VELOCITY_FUNCTION,
    TAUBER_SUTTON_PROVENANCE,
    earth_radiative_heat_flux,
    earth_radiative_heating_exponent,
    earth_velocity_function,
    mars_radiative_heat_flux,
    radiative_heat_transfer_coefficient,
)

__all__ = [
    "EARTH_EXPONENT_CAPS",
    "EARTH_VELOCITY_FUNCTION",
    "FAY_RIDDELL_COEFFICIENT_EXACT_094",
    "FAY_RIDDELL_COEFFICIENT_FROM_NUSSELT",
    "FAY_RIDDELL_COEFFICIENT_LITERATURE",
    "FAY_RIDDELL_COEFFICIENT_SOURCE",
    "FAY_RIDDELL_FACTOR_PR071",
    "FAY_RIDDELL_NUSSELT_COEFFICIENT",
    "MARS_VELOCITY_FUNCTION",
    "TAUBER_SUTTON_PROVENANCE",
    "TauberSuttonRadiation",
    "WallCatalycity",
    "catalycity_bracket",
    "earth_radiative_heat_flux",
    "earth_radiative_heating_exponent",
    "earth_velocity_function",
    "fay_riddell",
    "lees_distribution",
    "mars_radiative_heat_flux",
    "newtonian_velocity_gradient",
    "radiative_heat_transfer_coefficient",
    "stefan_recession_rate",
    "sutton_graves",
]
