"""Leading-edge single-temperature recession balance (Paper II, §4.4).

Non-pyrolyzing refractory leading edges (C/C, ZrB₂–SiC) recede by
surface oxidation and mechanical removal with no in-depth reaction zone
and no pyrolysis gas; for that material class the Stefan-type balance

.. math::

    \\rho_m \\Delta H_{\\mathrm{abl}}\\,\\dot s
    = \\dot q_{\\mathrm{total}} - \\varepsilon\\sigma_S T_w^4
    - k_{\\mathrm{mat}}\\,\\partial T/\\partial n|_{\\mathrm{wall}}

is the *physically correct* model, not a simplification — the paper's
Remark resolves the apparent inconsistency with Paper I explicitly: the
charring formulation of :mod:`aether.thermal` applies to acreage
phenolics, this balance to refractory edges, selected per region from
the material map. Applying either to the other class is wrong in a
specific, stated way.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aether.thermal.surface import STEFAN_BOLTZMANN

__all__ = ["stefan_recession_rate"]

_FloatArray = NDArray[np.float64]


def stefan_recession_rate(
    total_heat_flux: ArrayLike,
    wall_temperature: ArrayLike,
    emissivity: float,
    conduction_flux: ArrayLike,
    material_density: float,
    ablation_enthalpy: float,
) -> _FloatArray:
    """Recession rate :math:`\\dot s` of Paper II, Eq. (4.6), m/s.

    Parameters
    ----------
    total_heat_flux:
        :math:`\\dot q_{\\mathrm{total}}` from the Lees distribution
        (W/m²).
    wall_temperature:
        :math:`T_w` (K), > 0.
    emissivity:
        Surface :math:`\\varepsilon \\in (0, 1]`.
    conduction_flux:
        :math:`k_{\\mathrm{mat}}\\,\\partial T/\\partial n` into the
        solid (W/m²).
    material_density, ablation_enthalpy:
        :math:`\\rho_m` (kg/m³), :math:`\\Delta H_{\\mathrm{abl}}`
        (J/kg), both > 0.

    Notes
    -----
    Oxidative recession is irreversible: where the net flux is negative
    (surface cooling faster than it is heated) the rate is clamped to
    zero rather than allowed to "un-ablate", and the clamp is a physical
    statement, not a numerical guard.
    """
    if not 0.0 < emissivity <= 1.0:
        raise ValueError(f"emissivity must be in (0, 1], got {emissivity}")
    for name, val in (
        ("material_density", material_density),
        ("ablation_enthalpy", ablation_enthalpy),
    ):
        if not (np.isfinite(val) and val > 0.0):
            raise ValueError(f"{name} must be finite and > 0, got {val}")
    q = np.asarray(total_heat_flux, dtype=np.float64)
    t_w = np.asarray(wall_temperature, dtype=np.float64)
    q_cond = np.asarray(conduction_flux, dtype=np.float64)
    if np.any(q < 0.0) or not np.all(np.isfinite(q)):
        raise ValueError("total_heat_flux must be finite and >= 0")
    if np.any(t_w <= 0.0) or not np.all(np.isfinite(t_w)):
        raise ValueError("wall_temperature must be finite and > 0")
    if not np.all(np.isfinite(q_cond)):
        raise ValueError("conduction_flux must be finite")

    net = q - emissivity * STEFAN_BOLTZMANN * t_w**4 - q_cond
    return np.asarray(np.maximum(net, 0.0) / (material_density * ablation_enthalpy))
