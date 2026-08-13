"""Surface energy balance and blowing reduction (Paper I, §3.4.4).

The surface balance (Eq. 3.19) equates net incoming flux — blowing-
reduced convection, absorbed radiation, and the mass-transfer enthalpy
of char and pyrolysis gas leaving the surface — to re-radiation plus
conduction into the solid. The blowing correction (Eq. 3.20)

.. math::

    \\phi = \\frac{\\ln(1 + 2\\lambda B')}{2\\lambda B'}

is evaluated through ``log1p`` so that :math:`\\phi \\to 1` as
:math:`B' \\to 0` without cancellation — the same numerical posture as
the time-to-go conjugate form, and flagged by the paper for exactly the
same reason.

Per the Remark in §3.4.4, the surface thermochemistry (the map from wall
state to char consumption rate) is a **table input** generated offline
by equilibrium minimization; it is interpolated with a :math:`C^2`
tensor-product spline so the interpolation discontinuity sits in the
third derivative. This module refuses silent extrapolation outside the
table.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.interpolate
import scipy.optimize
from numpy.typing import ArrayLike, NDArray

from aether.thermal.material import CharringMaterial

__all__ = [
    "STEFAN_BOLTZMANN",
    "SurfaceEnergyBalance",
    "SurfaceEnvironment",
    "SurfaceThermochemistry",
    "blowing_correction",
]

_FloatArray = NDArray[np.float64]

#: Stefan–Boltzmann constant, W/(m² K⁴).
STEFAN_BOLTZMANN = 5.670374419e-8


def blowing_correction(b_prime: ArrayLike, lam: float = 0.5) -> _FloatArray:
    """Blowing reduction :math:`\\phi(B')` of Paper I, Eq. (3.20).

    Parameters
    ----------
    b_prime:
        Non-dimensional blowing rate :math:`B' \\ge 0`.
    lam:
        :math:`\\lambda \\approx 0.5` laminar, :math:`0.4` turbulent.

    Notes
    -----
    Evaluated as ``log1p(x)/x`` with :math:`x = 2\\lambda B'`, which is
    cancellation-free for small :math:`x` (``log1p`` carries the full
    relative accuracy of its argument); the removable singularity at
    :math:`x = 0` is filled with the exact limit :math:`\\phi = 1`.
    """
    if not (np.isfinite(lam) and 0.0 < lam <= 1.0):
        raise ValueError(f"lambda must be finite in (0, 1], got {lam}")
    b = np.asarray(b_prime, dtype=np.float64)
    if np.any(b < 0.0) or not np.all(np.isfinite(b)):
        raise ValueError("B' must be finite and >= 0")
    x = 2.0 * lam * b
    safe = np.where(x > 0.0, x, 1.0)
    return np.asarray(np.where(x > 0.0, np.log1p(safe) / safe, 1.0))


@dataclass(frozen=True)
class SurfaceEnvironment:
    """Boundary-layer edge conditions feeding the surface balance.

    Attributes
    ----------
    film_coefficient:
        :math:`\\rho_e u_e C_H` (kg/(m² s)).
    recovery_enthalpy:
        :math:`h_r` (J/kg).
    radiative_flux:
        Incoming radiative flux :math:`\\dot q_{\\mathrm{rad,in}}`
        (W/m²).
    absorptivity:
        Surface absorptivity :math:`\\alpha_w \\in (0, 1]`.
    wall_enthalpy:
        Boundary-layer gas enthalpy evaluated at wall temperature,
        :math:`h_w(T_w)` (J/kg).
    blowing_lambda:
        :math:`\\lambda` of Eq. (3.20); 0.5 laminar, 0.4 turbulent.
    """

    film_coefficient: float
    recovery_enthalpy: float
    radiative_flux: float
    absorptivity: float
    wall_enthalpy: Callable[[float], float]
    blowing_lambda: float = 0.5

    def __post_init__(self) -> None:
        if not (np.isfinite(self.film_coefficient) and self.film_coefficient > 0.0):
            raise ValueError(f"film_coefficient must be > 0, got {self.film_coefficient}")
        if not 0.0 < self.absorptivity <= 1.0:
            raise ValueError(f"absorptivity must be in (0, 1], got {self.absorptivity}")
        if not (np.isfinite(self.radiative_flux) and self.radiative_flux >= 0.0):
            raise ValueError(f"radiative_flux must be >= 0, got {self.radiative_flux}")


@dataclass(frozen=True)
class SurfaceEnergyBalance:
    """Residual form and scalar solve of Paper I, Eq. (3.19)."""

    material: CharringMaterial
    environment: SurfaceEnvironment

    def residual(
        self,
        wall_temperature: float,
        char_flux: float,
        gas_flux: float,
        conduction_flux: float,
        char_fraction: float = 1.0,
    ) -> float:
        """Net surface flux imbalance (W/m²); zero at the balanced wall state.

        Parameters
        ----------
        wall_temperature:
            Trial :math:`T_w` (K).
        char_flux, gas_flux:
            :math:`\\dot m_c = \\rho_c \\dot s` and :math:`\\dot m_g`
            at the surface (kg/(m² s)).
        conduction_flux:
            :math:`(k/\\ell)\\,\\partial T/\\partial\\eta|_{\\eta=0}`
            into the solid (W/m²).
        char_fraction:
            Surface :math:`\\beta` for the emissivity blend (the
            ablating surface is normally fully charred).
        """
        t_w = float(wall_temperature)
        if not (np.isfinite(t_w) and t_w > 0.0):
            raise ValueError(f"wall temperature must be finite and > 0, got {t_w}")
        if char_flux < 0.0 or gas_flux < 0.0:
            raise ValueError("mass fluxes must be >= 0")
        env = self.environment
        mat = self.material

        total_flux = char_flux + gas_flux
        b_prime = total_flux / env.film_coefficient
        phi = float(blowing_correction(b_prime, env.blowing_lambda))
        h_w = float(env.wall_enthalpy(t_w))

        convective = env.film_coefficient * (env.recovery_enthalpy - h_w) * phi
        absorbed = env.absorptivity * env.radiative_flux
        mass_transfer = (
            char_flux * float(mat.solid_enthalpy(t_w))
            + gas_flux * float(mat.gas_enthalpy(t_w))
            - total_flux * h_w
        )
        emissivity = float(mat.emissivity(char_fraction))
        reradiation = emissivity * STEFAN_BOLTZMANN * t_w**4
        return convective + absorbed + mass_transfer - reradiation - float(conduction_flux)

    def solve_wall_temperature(
        self,
        char_flux: float,
        gas_flux: float,
        conduction_flux: float,
        char_fraction: float = 1.0,
        bracket: tuple[float, float] = (200.0, 6000.0),
    ) -> float:
        """Wall temperature balancing Eq. (3.19), by bracketed root find.

        Brent's method on the stated bracket: the residual is strictly
        decreasing in :math:`T_w` for physical inputs (re-radiation grows
        as :math:`T_w^4`, wall enthalpy grows with :math:`T_w`), so a
        sign change over the bracket implies a unique root. No sign
        change is an input error and is reported as such.
        """
        lo, hi = float(bracket[0]), float(bracket[1])
        if not 0.0 < lo < hi:
            raise ValueError(f"bracket must satisfy 0 < lo < hi, got {bracket}")

        def f(t_w: float) -> float:
            return self.residual(t_w, char_flux, gas_flux, conduction_flux, char_fraction)

        f_lo, f_hi = f(lo), f(hi)
        if f_lo * f_hi > 0.0:
            raise ValueError(
                f"surface balance has no sign change on [{lo}, {hi}] K "
                f"(residuals {f_lo:.4e}, {f_hi:.4e}); check environment magnitudes"
            )
        result = scipy.optimize.brentq(f, lo, hi, xtol=1e-10, rtol=8.9e-16, full_output=True)
        root, info = result
        if not info.converged:  # pragma: no cover - brentq converges on a valid bracket
            raise RuntimeError("Brent iteration failed to converge on surface balance")
        return float(root)


class SurfaceThermochemistry:
    """:math:`C^2` spline interpolant of an offline thermochemistry table.

    Maps :math:`(T_w, B'_g) \\mapsto B'_c` — the non-dimensional char
    consumption rate from equilibrium surface chemistry — per the Remark
    in Paper I §3.4.4. Cubic tensor-product B-splines are :math:`C^2`,
    placing the interpolation discontinuity in the third derivative.
    Queries outside the tabulated rectangle raise rather than
    extrapolate: an equilibrium table is meaningless outside its
    generation envelope.

    Parameters
    ----------
    wall_temperatures:
        Strictly increasing grid of :math:`T_w` (K), at least 4 points.
    gas_blowing_rates:
        Strictly increasing grid of :math:`B'_g`, at least 4 points.
    char_blowing_table:
        :math:`B'_c \\ge 0` values, shape
        ``(len(wall_temperatures), len(gas_blowing_rates))``.
    """

    def __init__(
        self,
        wall_temperatures: ArrayLike,
        gas_blowing_rates: ArrayLike,
        char_blowing_table: ArrayLike,
    ) -> None:
        t = np.asarray(wall_temperatures, dtype=np.float64)
        b = np.asarray(gas_blowing_rates, dtype=np.float64)
        z = np.asarray(char_blowing_table, dtype=np.float64)
        if t.ndim != 1 or t.size < 4 or np.any(np.diff(t) <= 0.0):
            raise ValueError("wall_temperatures must be strictly increasing with >= 4 points")
        if b.ndim != 1 or b.size < 4 or np.any(np.diff(b) <= 0.0):
            raise ValueError("gas_blowing_rates must be strictly increasing with >= 4 points")
        if z.shape != (t.size, b.size):
            raise ValueError(
                f"char_blowing_table shape {z.shape} does not match grids "
                f"({t.size}, {b.size})"
            )
        if np.any(z < 0.0) or not np.all(np.isfinite(z)):
            raise ValueError("char blowing rates must be finite and >= 0")
        self._t_range = (float(t[0]), float(t[-1]))
        self._b_range = (float(b[0]), float(b[-1]))
        self._spline = scipy.interpolate.RectBivariateSpline(t, b, z, kx=3, ky=3, s=0)

    def char_blowing_rate(self, wall_temperature: float, gas_blowing_rate: float) -> float:
        """Interpolated :math:`B'_c` at the queried wall state."""
        t_w, b_g = float(wall_temperature), float(gas_blowing_rate)
        if not self._t_range[0] <= t_w <= self._t_range[1]:
            raise ValueError(
                f"T_w = {t_w} K outside tabulated range {self._t_range}; "
                f"refusing to extrapolate an equilibrium table"
            )
        if not self._b_range[0] <= b_g <= self._b_range[1]:
            raise ValueError(
                f"B'_g = {b_g} outside tabulated range {self._b_range}; "
                f"refusing to extrapolate an equilibrium table"
            )
        return max(float(self._spline(t_w, b_g)[0, 0]), 0.0)

    def recession_rate(
        self,
        wall_temperature: float,
        gas_blowing_rate: float,
        film_coefficient: float,
        char_density: float,
    ) -> float:
        """:math:`\\dot s = B'_c\\,\\rho_e u_e C_H / \\rho_c` (m/s), from
        :math:`\\dot m_c = \\rho_c \\dot s` (Paper I, below Eq. 3.19)."""
        if not (np.isfinite(film_coefficient) and film_coefficient > 0.0):
            raise ValueError(f"film_coefficient must be > 0, got {film_coefficient}")
        if not (np.isfinite(char_density) and char_density > 0.0):
            raise ValueError(f"char_density must be > 0, got {char_density}")
        b_c = self.char_blowing_rate(wall_temperature, gas_blowing_rate)
        return b_c * film_coefficient / char_density
