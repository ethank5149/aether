"""Lees heat-flux distribution over the planform (Paper II, §4.3).

.. math::

    \\dot q(\\xi, \\eta) = \\dot q_s\\,\\sin\\delta_c(\\xi,\\eta)
    \\left(\\frac{R_{\\mathrm{eff}}}{x(\\xi)}\\right)^{1/2},

with the :math:`x \\to 0` singularity handled exactly as the paper
states: within the stagnation region :math:`x < R_{\\mathrm{eff}}` the
running-length factor is held at its matching value, so the distribution
is continuous at :math:`x = R_{\\mathrm{eff}}`. Leeward nodes
(:math:`\\delta_c \\le 0`) receive zero — the correlation is a windward
model. The laminar caveat of the paper's Remark stands: transition
multiplies local heating by 3–5× and should be dispersed, not fixed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["lees_distribution"]

_FloatArray = NDArray[np.float64]


def lees_distribution(
    stagnation_flux: float,
    incidence_angle: ArrayLike,
    running_length: ArrayLike,
    effective_radius: float,
) -> _FloatArray:
    """Local heat flux from the stagnation value (Paper II, Eq. 4.5), W/m².

    Parameters
    ----------
    stagnation_flux:
        Total stagnation heating :math:`\\dot q_{s,\\mathrm{conv}} +
        \\dot q_{s,\\mathrm{rad}}` (W/m²).
    incidence_angle:
        Local incidence :math:`\\delta_c` (rad); non-positive values are
        leeward and receive zero flux.
    running_length:
        Streamline distance :math:`x` from the stagnation point (m),
        non-negative.
    effective_radius:
        :math:`R_{\\mathrm{eff}} > 0` (m).
    """
    if not (np.isfinite(stagnation_flux) and stagnation_flux >= 0.0):
        raise ValueError(f"stagnation_flux must be finite and >= 0, got {stagnation_flux}")
    if not (np.isfinite(effective_radius) and effective_radius > 0.0):
        raise ValueError(f"effective_radius must be finite and > 0, got {effective_radius}")
    delta = np.asarray(incidence_angle, dtype=np.float64)
    x = np.asarray(running_length, dtype=np.float64)
    if np.any(x < 0.0) or not np.all(np.isfinite(x)):
        raise ValueError("running_length must be finite and >= 0")
    if not np.all(np.isfinite(delta)):
        raise ValueError("incidence_angle must be finite")

    # (R_eff/x)^(1/2), capped at unity inside the stagnation region so the
    # distribution matches continuously at x = R_eff
    with np.errstate(divide="ignore"):
        length_factor = np.where(
            x < effective_radius, 1.0, np.sqrt(effective_radius / np.maximum(x, 1e-300))
        )
    windward = np.maximum(np.sin(delta), 0.0)
    return np.asarray(stagnation_flux * windward * length_factor)
