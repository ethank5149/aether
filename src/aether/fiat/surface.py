"""Surface and backface boundary conditions (Chen & Milos 1999, Eqs. 10–11).

**Surface energy balance.** Chen & Milos write Eq. (10) in
convective-transfer-coefficient form; Milos, Chen & Squire restate it as
a flux balance, which is the arrangement used here:

.. math::

    \\rho_e u_e C_H (H_r - h_w) + \\dot m_c h_c + \\dot m_g h_g
      + \\alpha_w q_{\\mathrm{rad}}
    = (\\dot m_c + \\dot m_g) h_w + F\\sigma\\varepsilon_w T_w^4 + q .

Reading left to right: sensible convective input, chemical energy carried
in by the receding char and the pyrolysis gas, absorbed shock-layer
radiation; against energy carried away by the blown mixture at wall
enthalpy, reradiation, and conduction into the TPS.

**Blowing correction, Eq. (11).**

.. math::

    \\frac{C_H}{C_{H_0}} = \\frac{\\ln(1 + 2\\lambda B')}{2\\lambda B'},
    \\qquad B' = \\frac{\\dot m_c + \\dot m_g}{\\rho_e u_e C_M} .

:math:`\\lambda = 0.5` for laminar flow — at which Eq. (11) reduces to
the classical laminar result — and 0.2 to 0.4 for transitional or
turbulent flow.

.. note::

   The archived scan of Chen & Milos Eq. (11) drops the logarithm,
   extracting as ``(1+2λB')/(2λB')``. That form does not tend to 1 as
   :math:`B' \\to 0` and does not reduce to the classical laminar
   correction at :math:`\\lambda = 1/2` as the surrounding text states it
   must. Milos, Chen & Squire print the same equation intact, with the
   logarithm and with the equivalent
   :math:`2\\lambda B'_1/(e^{2\\lambda B'_1} - 1)`. The logarithmic form
   is used here, and :func:`blowing_reduction` is tested against both of
   the source's own algebraic statements.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aether.fiat.bprime import BPrimeTable
from aether.thermal.surface import STEFAN_BOLTZMANN

__all__ = [
    "AerothermalEnvironment",
    "BackfaceCondition",
    "BackfaceKind",
    "SurfaceState",
    "blowing_reduction",
    "solve_surface",
]

_FloatArray = NDArray[np.float64]


def blowing_reduction(b_prime: ArrayLike, lam: float = 0.5) -> _FloatArray:
    """:math:`C_H/C_{H_0} = \\ln(1 + 2\\lambda B')/(2\\lambda B')`, Eq. (11).

    Returns 1 at :math:`B' = 0` by the removable singularity, evaluated
    through :func:`numpy.log1p` so that small :math:`B'` does not lose
    precision to cancellation.
    """
    b = np.asarray(b_prime, dtype=np.float64)
    if not np.all(np.isfinite(b)) or np.any(b < 0.0):
        raise ValueError("b_prime must be finite and >= 0")
    if not (np.isfinite(lam) and lam > 0.0):
        raise ValueError(f"lam must be finite and > 0, got {lam}")
    x = 2.0 * lam * b
    # log1p(x)/x -> 1 - x/2 + x^2/3 as x -> 0; the series is used below the
    # threshold where the quotient starts losing significant digits.
    small = x < 1e-8
    out = np.empty_like(x)
    out[small] = 1.0 - 0.5 * x[small]
    out[~small] = np.log1p(x[~small]) / x[~small]
    return out


class BackfaceKind(Enum):
    """Which backface condition a stack is run with."""

    ADIABATIC = "adiabatic"
    FIXED_TEMPERATURE = "fixed_temperature"
    RADIATING = "radiating"


@dataclass(frozen=True)
class BackfaceCondition:
    """Condition at the inner face of the stack.

    The backface choice moves recession and in-depth temperature by more
    than most material-property uncertainties, so it is an explicit
    object rather than a default.

    Attributes
    ----------
    kind:
        Which of the three conditions applies.
    temperature:
        Imposed :math:`T` (K) for :attr:`BackfaceKind.FIXED_TEMPERATURE`.
    emissivity, sink_temperature:
        For :attr:`BackfaceKind.RADIATING`, the backface emissivity and
        the temperature (K) of the space it radiates to.
    """

    kind: BackfaceKind = BackfaceKind.ADIABATIC
    temperature: float | None = None
    emissivity: float = 0.8
    sink_temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.kind is BackfaceKind.FIXED_TEMPERATURE and (
            self.temperature is None
            or not (np.isfinite(self.temperature) and self.temperature > 0.0)
        ):
            raise ValueError("FIXED_TEMPERATURE needs a finite positive temperature")
        if self.kind is BackfaceKind.RADIATING:
            if not 0.0 < self.emissivity <= 1.0:
                raise ValueError(f"emissivity must be in (0, 1], got {self.emissivity}")
            if not (np.isfinite(self.sink_temperature) and self.sink_temperature >= 0.0):
                raise ValueError("sink_temperature must be finite and >= 0")

    def closure(self, t_last: float, t_back: float, k_last: float, half: float) -> float:
        """Residual of the equation that determines the backface temperature.

        Written so that all three conditions present the same interface
        to the Newton solve, with :math:`T_b` always an unknown.
        """
        conduction = k_last * (t_last - t_back) / half
        if self.kind is BackfaceKind.ADIABATIC:
            return conduction
        if self.kind is BackfaceKind.FIXED_TEMPERATURE:
            assert self.temperature is not None
            return t_back - self.temperature
        radiated = self.emissivity * STEFAN_BOLTZMANN * (
            t_back**4 - self.sink_temperature**4
        )
        return conduction - radiated

    def closure_jacobian(
        self, t_last: float, t_back: float, k_last: float, half: float
    ) -> tuple[float, float]:
        """``(d closure / d T_last, d closure / d T_back)``."""
        if self.kind is BackfaceKind.ADIABATIC:
            return k_last / half, -k_last / half
        if self.kind is BackfaceKind.FIXED_TEMPERATURE:
            return 0.0, 1.0
        d_rad = 4.0 * self.emissivity * STEFAN_BOLTZMANN * t_back**3
        return k_last / half, -k_last / half - d_rad

    def flux(self, t_last: float, t_back: float, k_last: float, half: float) -> float:
        """Conduction flux leaving the stack at the backface, W/m².

        Adiabatic is exactly zero by construction; the other two conduct
        to the backface temperature the closure has determined.
        """
        if self.kind is BackfaceKind.ADIABATIC:
            return 0.0
        return k_last * (t_last - t_back) / half

    def flux_jacobian(self, k_last: float, half: float) -> tuple[float, float]:
        """``(d flux / d T_last, d flux / d T_back)``."""
        if self.kind is BackfaceKind.ADIABATIC:
            return 0.0, 0.0
        return k_last / half, -k_last / half


@dataclass(frozen=True)
class AerothermalEnvironment:
    """Boundary-layer edge conditions at one instant.

    This is FIAT's ``envir.inp`` row: *"File envir.inp provides the time
    history of boundary condition data at the heated surface."*

    Attributes
    ----------
    film_coefficient:
        Unblown :math:`\\rho_e u_e C_{H_0}` (kg/(m² s)).
    mass_transfer_coefficient:
        Unblown :math:`\\rho_e u_e C_{M_0}` (kg/(m² s)). Defaults to
        :attr:`film_coefficient`, i.e. unit Lewis number, which is
        FIAT's usual assumption but is stated here rather than hidden.
    recovery_enthalpy:
        :math:`H_r` (J/kg).
    pressure:
        Surface pressure (Pa), for the B' table lookup.
    radiative_flux:
        Incident shock-layer radiation :math:`q_{\\mathrm{rad}}` (W/m²).
    wall_absorptance, wall_emissivity, view_factor:
        :math:`\\alpha_w`, :math:`\\varepsilon_w`, :math:`F`.
    blowing_parameter:
        :math:`\\lambda`: 0.5 laminar, 0.2–0.4 transitional/turbulent.
    """

    film_coefficient: float
    recovery_enthalpy: float
    pressure: float
    mass_transfer_coefficient: float | None = None
    radiative_flux: float = 0.0
    wall_absorptance: float = 0.9
    wall_emissivity: float = 0.9
    view_factor: float = 1.0
    blowing_parameter: float = 0.5

    def __post_init__(self) -> None:
        if not (np.isfinite(self.film_coefficient) and self.film_coefficient >= 0.0):
            raise ValueError("film_coefficient must be finite and >= 0")
        if self.mass_transfer_coefficient is not None and not (
            np.isfinite(self.mass_transfer_coefficient)
            and self.mass_transfer_coefficient >= 0.0
        ):
            raise ValueError("mass_transfer_coefficient must be finite and >= 0")
        if not np.isfinite(self.recovery_enthalpy):
            raise ValueError("recovery_enthalpy must be finite")
        if not (np.isfinite(self.pressure) and self.pressure > 0.0):
            raise ValueError("pressure must be finite and > 0")
        if not (np.isfinite(self.radiative_flux) and self.radiative_flux >= 0.0):
            raise ValueError("radiative_flux must be finite and >= 0")
        for name in ("wall_absorptance", "wall_emissivity"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if not 0.0 <= self.view_factor <= 1.0:
            raise ValueError(f"view_factor must be in [0, 1], got {self.view_factor}")
        if not (np.isfinite(self.blowing_parameter) and self.blowing_parameter > 0.0):
            raise ValueError("blowing_parameter must be finite and > 0")

    @property
    def mass_coefficient(self) -> float:
        """:math:`\\rho_e u_e C_{M_0}`, defaulting to the film coefficient."""
        if self.mass_transfer_coefficient is None:
            return self.film_coefficient
        return self.mass_transfer_coefficient


@dataclass(frozen=True)
class SurfaceState:
    """Converged surface solution at one instant."""

    wall_temperature: float
    char_rate: float
    """:math:`B'_c` (-)."""
    gas_rate: float
    """:math:`B'_g` (-)."""
    wall_enthalpy: float
    """:math:`h_w` (J/kg)."""
    char_mass_flux: float
    """:math:`\\dot m_c` (kg/(m² s))."""
    gas_mass_flux: float
    """:math:`\\dot m_g` (kg/(m² s))."""
    blowing_reduction: float
    """:math:`C_H/C_{H_0}` (-)."""
    conduction_flux: float
    """:math:`q` into the solid (W/m²)."""
    recession_rate: float
    """:math:`\\dot s` (m/s)."""


def solve_surface(
    wall_temperature: float,
    gas_mass_flux: float,
    conduction_flux: float,
    environment: AerothermalEnvironment,
    table: BPrimeTable,
    char_density: float,
    char_enthalpy: float,
    gas_enthalpy: float,
) -> tuple[float, SurfaceState]:
    """Evaluate the surface energy balance residual and its state.

    Returns ``(residual, state)``. The residual is Eq. (10) rearranged to
    zero; the Newton solve in :mod:`aether.fiat.solver` drives it
    with :math:`T_w` as the unknown, so this function does no iteration
    of its own.

    Parameters
    ----------
    wall_temperature:
        Current :math:`T_w` iterate (K).
    gas_mass_flux:
        :math:`\\dot m_g` arriving at the surface (kg/(m² s)), from the
        in-depth mass balance.
    conduction_flux:
        :math:`q` into the solid (W/m²), from the in-depth solution.
    char_density:
        :math:`\\rho_c` of the ablating ply, for
        :math:`\\dot s = \\dot m_c/\\rho_c`.
    char_enthalpy, gas_enthalpy:
        :math:`h_c` and :math:`h_g` at the wall (J/kg).
    """
    t_w = float(wall_temperature)
    m_g = max(float(gas_mass_flux), 0.0)
    c_m = environment.mass_coefficient
    b_g = m_g / c_m if c_m > 0.0 else 0.0

    b_c = table.char_rate(environment.pressure, b_g, t_w)
    h_w = table.wall_enthalpy(environment.pressure, b_g, t_w)
    m_c = b_c * c_m
    phi = float(blowing_reduction(b_g + b_c, environment.blowing_parameter))
    c_h = environment.film_coefficient * phi

    residual = (
        c_h * (environment.recovery_enthalpy - h_w)
        + m_c * char_enthalpy
        + m_g * gas_enthalpy
        + environment.wall_absorptance * environment.radiative_flux
        - (m_c + m_g) * h_w
        - environment.view_factor
        * STEFAN_BOLTZMANN
        * environment.wall_emissivity
        * t_w**4
        - conduction_flux
    )
    state = SurfaceState(
        wall_temperature=t_w,
        char_rate=b_c,
        gas_rate=b_g,
        wall_enthalpy=h_w,
        char_mass_flux=m_c,
        gas_mass_flux=m_g,
        blowing_reduction=phi,
        conduction_flux=float(conduction_flux),
        recession_rate=m_c / char_density,
    )
    return float(residual), state
