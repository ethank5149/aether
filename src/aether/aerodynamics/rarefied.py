"""Free-molecular and transitional aerodynamics.

Above about 90 km a molecule that bounces off the vehicle flies away without
hitting another one, so there is no shock, no boundary layer and no
continuum. The Navier–Stokes equations are not inaccurate there; they are
inapplicable, and so is every method built on them — the panel method's
Prandtl–Meyer branch and the reference-temperature boundary layer alike.

This matters here for two specific things. A FOBS parking orbit at 170 km
spends its whole coast in free-molecular flow, and the drag there sets how
much the orbit decays before the deorbit burn. And an HGV entering at Mach
25 crosses the entire transitional regime between roughly 120 and 70 km,
which is where its trajectory is bent and where continuum theory would
overpredict drag by a factor of order two at the top of that band.

Free-molecular closure
----------------------

Schaaf and Chambré's result for a surface element in a Maxwellian stream,
with diffuse re-emission at accommodation coefficients :math:`\\sigma_n` and
:math:`\\sigma_t`, molecular speed ratio :math:`S = V/\\sqrt{2RT}` and
:math:`s = S\\sin\\delta` the normal component:

.. math::

    C_p = \\frac{1}{S^2}\\Bigg\\{
      \\left[\\frac{2-\\sigma_n}{\\sqrt\\pi}s
      + \\frac{\\sigma_n}{2}\\sqrt{\\frac{T_w}{T_\\infty}}\\right]e^{-s^2}
      + \\left[(2-\\sigma_n)\\left(s^2+\\tfrac12\\right)
      + \\frac{\\sigma_n}{2}\\sqrt{\\frac{\\pi T_w}{T_\\infty}}\\,s\\right]
      \\big[1+\\mathrm{erf}(s)\\big]\\Bigg\\}

.. math::

    C_\\tau = \\frac{\\sigma_t\\cos\\delta}{S\\sqrt\\pi}
      \\Big\\{e^{-s^2} + \\sqrt\\pi\\,s\\big[1+\\mathrm{erf}(s)\\big]\\Big\\}

Two things follow that are worth stating because they are what makes this
regime different rather than merely thinner:

* **There is no shadow.** Every element sees the stream, including one
  facing directly aft, because the distribution is Maxwellian and its tail
  reaches around. The formulae are evaluated on every panel with no ray
  casting; the leeward branch is :math:`\\mathrm{erf}` of a negative
  argument, not a separate model.
* **Shear is half the drag.** In the hyperthermal limit the pressure branch
  goes to :math:`2\\sin^2\\delta` and the shear branch to
  :math:`2\\sigma_t\\sin\\delta\\cos\\delta`, and on a sphere those integrate
  to exactly 1 and exactly 1 for a total :math:`C_D` of 2 — the classical
  free-molecular sphere result. A method that carried pressure alone would
  be short by half. That identity is the test this module is checked with.

The transitional bridge
-----------------------

Between :math:`Kn \\approx 10^{-3}` and :math:`10` neither limit holds and
there is no theory, only correlation. :func:`sine_squared_bridge` is the
Wilmoth–Blanchard form used across the entry literature. It is
**empirical**, it is :math:`C^1` and not :math:`C^2`, and it is the weakest
link in this whole pipeline; the honest description is that it interpolates
between two things that are each defensible, and asserts nothing of its own.
Direct simulation Monte Carlo is what actually resolves this band, and it is
not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.special
from numpy.typing import ArrayLike, NDArray

from aether.aerodynamics.tables import Coefficients

__all__ = [
    "CONTINUUM_KNUDSEN",
    "FREE_MOLECULAR_KNUDSEN",
    "FreeMolecularSolver",
    "free_molecular_coefficients",
    "sine_squared_bridge",
    "sphere_free_molecular_drag",
]

_FloatArray = NDArray[np.float64]

#: Below this Knudsen number the flow is taken as continuum.
CONTINUUM_KNUDSEN = 1.0e-3
#: Above this Knudsen number the flow is taken as free molecular.
FREE_MOLECULAR_KNUDSEN = 10.0


def free_molecular_coefficients(
    incidence: ArrayLike,
    speed_ratio: float,
    wall_temperature_ratio: float = 1.0,
    normal_accommodation: float = 1.0,
    tangential_accommodation: float = 1.0,
) -> tuple[_FloatArray, _FloatArray]:
    """Schaaf–Chambré pressure and shear coefficients on a surface element.

    Parameters
    ----------
    incidence:
        Local incidence :math:`\\delta` (rad), positive windward — the same
        sign convention as :mod:`aether.aerodynamics.closure`.
    speed_ratio:
        :math:`S = V_\\infty/\\sqrt{2RT_\\infty}`. Equal to
        :math:`M\\sqrt{\\gamma/2}`, so Mach 25 is about :math:`S = 21`.
    wall_temperature_ratio:
        :math:`T_w/T_\\infty`. Enters only through re-emission, as
        :math:`\\sqrt{T_w/T_\\infty}/S`, so at :math:`S = 20` a factor-of-four
        error in it moves :math:`C_p` by under two per cent — which is why a
        fixed wall temperature is defensible here and is not on a boundary
        layer.
    normal_accommodation, tangential_accommodation:
        :math:`\\sigma_n, \\sigma_t`. Unity is full diffuse accommodation, the
        usual assumption for an engineering surface. Zero normal
        accommodation is specular reflection, which doubles the pressure
        because the molecule's normal momentum is reversed rather than
        absorbed.

    Returns
    -------
    tuple
        ``(cp, c_tau)``. ``c_tau`` is positive along the surface-tangential
        component of the freestream direction.
    """
    s_ratio = float(speed_ratio)
    if not (np.isfinite(s_ratio) and s_ratio > 0.0):
        msg = f"speed ratio must be finite and > 0, got {speed_ratio}"
        raise ValueError(msg)
    sigma_n = float(normal_accommodation)
    sigma_t = float(tangential_accommodation)
    for label, value in (("normal", sigma_n), ("tangential", sigma_t)):
        if not 0.0 <= value <= 1.0:
            msg = f"{label} accommodation must be in [0, 1], got {value}"
            raise ValueError(msg)
    ratio = float(wall_temperature_ratio)
    # Zero is admitted, not excluded: T_w/T_inf = 0 is the standard cold-wall
    # idealisation in which re-emitted molecules carry away no momentum, and
    # it is the limit the hyperthermal results 2 sin^2(delta) and
    # 2 sin(delta) cos(delta) are stated in.
    if not (np.isfinite(ratio) and ratio >= 0.0):
        msg = f"wall temperature ratio must be finite and >= 0, got {ratio}"
        raise ValueError(msg)

    delta = np.asarray(incidence, dtype=np.float64)
    sin_delta = np.sin(delta)
    cos_delta = np.cos(delta)
    s = s_ratio * sin_delta

    # exp(-s^2) underflows to zero for |s| beyond about 27, which is the
    # correct limit and not a loss: the erf term carries everything there.
    gaussian = np.exp(-np.minimum(s * s, 700.0))
    complementary = 1.0 + scipy.special.erf(s)
    wall = np.sqrt(ratio)

    pressure = (
        ((2.0 - sigma_n) / np.sqrt(np.pi) * s + 0.5 * sigma_n * wall) * gaussian
        + ((2.0 - sigma_n) * (s * s + 0.5) + 0.5 * sigma_n * np.sqrt(np.pi) * wall * s)
        * complementary
    ) / (s_ratio * s_ratio)

    shear = (
        sigma_t
        * cos_delta
        / (s_ratio * np.sqrt(np.pi))
        * (gaussian + np.sqrt(np.pi) * s * complementary)
    )
    return np.asarray(pressure), np.asarray(shear)


def sphere_free_molecular_drag(speed_ratio: float, wall_temperature_ratio: float = 1.0) -> float:
    """Closed-form free-molecular sphere drag, diffuse reflection.

    .. math::

        C_D = \\frac{2S^2+1}{\\sqrt\\pi S^3}e^{-S^2}
        + \\frac{4S^4+4S^2-1}{2S^4}\\mathrm{erf}(S)
        + \\frac{2\\sqrt\\pi}{3S}\\sqrt{\\frac{T_w}{T_\\infty}}

    On the sphere's frontal area. Present as an **independent check on the
    panel integration**, not as a model of anything in this codebase: nothing
    here is a sphere. Its hyperthermal limit is exactly 2, and reproducing
    that by integrating :func:`free_molecular_coefficients` over a
    discretised sphere is what verifies the surface closure — including the
    shear branch, which contributes exactly half of it and which a
    pressure-only check would miss entirely.
    """
    s = float(speed_ratio)
    if not (np.isfinite(s) and s > 0.0):
        msg = f"speed ratio must be finite and > 0, got {speed_ratio}"
        raise ValueError(msg)
    gaussian = float(np.exp(-min(s * s, 700.0)))
    return float(
        (2.0 * s * s + 1.0) / (np.sqrt(np.pi) * s**3) * gaussian
        + (4.0 * s**4 + 4.0 * s * s - 1.0) / (2.0 * s**4) * scipy.special.erf(s)
        + 2.0 * np.sqrt(np.pi) / (3.0 * s) * np.sqrt(float(wall_temperature_ratio))
    )


def sine_squared_bridge(
    knudsen: ArrayLike,
    continuum: float = CONTINUUM_KNUDSEN,
    free_molecular: float = FREE_MOLECULAR_KNUDSEN,
) -> _FloatArray:
    """Weight on the free-molecular limit, 0 to 1, across the transitional band.

    .. math::

        f = \\sin^2\\left(\\frac{\\pi}{2}\\,
        \\frac{\\log_{10} Kn - \\log_{10} Kn_c}
             {\\log_{10} Kn_{fm} - \\log_{10} Kn_c}\\right)

    clamped outside the band. Logarithmic because the band spans four decades
    of Knudsen number and nothing about it is linear in :math:`Kn`.
    """
    low, high = float(continuum), float(free_molecular)
    if not 0.0 < low < high:
        msg = f"bridge needs 0 < continuum < free_molecular, got ({low}, {high})"
        raise ValueError(msg)
    kn = np.asarray(knudsen, dtype=np.float64)
    fraction = np.clip(
        (np.log10(np.maximum(kn, 1.0e-300)) - np.log10(low)) / (np.log10(high) - np.log10(low)),
        0.0,
        1.0,
    )
    return np.asarray(np.sin(0.5 * np.pi * fraction) ** 2)


@dataclass
class FreeMolecularSolver:
    """Integrated free-molecular coefficients over a panelised mesh.

    Attributes
    ----------
    mesh:
        Anything with ``panel_model(reference_point)``.
    reference_area, reference_length:
        Non-dimensionalisation, shared with the continuum tables so the two
        can be bridged without renormalising.
    wall_temperature:
        Surface temperature (K). See
        :func:`free_molecular_coefficients` on why a fixed value is adequate
        here.
    """

    mesh: Any
    reference_area: float
    reference_length: float
    reference_point: _FloatArray | None = None
    wall_temperature: float = 300.0
    normal_accommodation: float = 1.0
    tangential_accommodation: float = 1.0
    gamma: float = 1.4
    name: str = "free-molecular"

    def __post_init__(self) -> None:
        self._model = self.mesh.panel_model(self.reference_point)

    def solve(self, mach: float, alpha: float, temperature: float = 250.0) -> Coefficients:
        """Coefficients at a Mach number, incidence and ambient temperature.

        Ambient temperature enters twice — through the speed ratio
        :math:`S = M\\sqrt{\\gamma/2}`, which it does *not* affect, and
        through :math:`T_w/T_\\infty`, which it does. Only the second is a
        real dependence, and it is weak.
        """
        model = self._model
        speed_ratio = float(mach) * np.sqrt(self.gamma / 2.0)
        delta = model.incidences(float(alpha))
        cp, c_tau = free_molecular_coefficients(
            delta,
            speed_ratio,
            wall_temperature_ratio=float(self.wall_temperature) / float(temperature),
            normal_accommodation=self.normal_accommodation,
            tangential_accommodation=self.tangential_accommodation,
        )

        normals = np.asarray(model.normals)
        areas = np.asarray(model.areas)
        v_hat = model.velocity_direction(float(alpha))

        # Shear acts along the surface-tangential projection of the freestream
        # direction. Where a panel is exactly normal to the flow that
        # projection vanishes; so does cos(delta), so the zero-length vector
        # is multiplied by a zero coefficient and the guarded normalisation
        # only keeps the arithmetic finite.
        tangential = v_hat[np.newaxis, :] - (normals @ v_hat)[:, np.newaxis] * normals
        magnitude = np.linalg.norm(tangential, axis=1)
        direction = tangential / np.where(magnitude > 1e-12, magnitude, 1.0)[:, np.newaxis]

        force = np.sum(
            (-(cp * areas))[:, np.newaxis] * normals + (c_tau * areas)[:, np.newaxis] * direction,
            axis=0,
        )
        arms = np.asarray(model.centroids) - np.asarray(model.reference_point)
        panel_force = (-(cp * areas))[:, np.newaxis] * normals + (c_tau * areas)[
            :, np.newaxis
        ] * direction
        moment = np.sum(np.cross(arms, panel_force), axis=0)

        scale = self.reference_area
        return Coefficients(
            axial=float(force[0] / scale),
            normal=float(force[2] / scale),
            pitching_moment=float(moment[1] / (scale * self.reference_length)),
        )
