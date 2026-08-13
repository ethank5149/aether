"""Manufactured solutions for the charring-ablation system (V4, first leg).

Constructs smooth non-polynomial fields — so spectral convergence is
visible rather than trivially exact — with every partial derivative in
closed form, and the equation-level sources that make them an exact
solution of Paper I, Eqs. (3.14)–(3.18) under the solver's
``eta_frame`` density-rate convention:

.. math::

    T^*(\\eta, t) &= T_0 + \\Delta T\\, e^{-2\\eta}
                     \\big(1 + a_T \\sin \\omega t\\big), \\\\
    \\rho_i^*(\\eta, t) &= \\rho_{c,i} + \\Delta_i
                     \\big(a + b\\, e^{-t/\\tau} P(\\eta)\\big),
    \\qquad P(\\eta) = \\tfrac{1}{2}\\big(1 + \\cos \\pi\\eta\\big), \\\\
    s^*(t) &= s_0 + v_s t .

The char-extent factor :math:`x = a + b e^{-t/\\tau} P` stays inside
:math:`(0, 1)`, so the degree-of-char clip and the kinetics extent clamp
are both inactive on the manufactured trajectory — the sources are exact,
not exact-up-to-clipping. Because every component shares the same
:math:`x`, the bulk density is :math:`\\rho^* = \\rho_c^{\\mathrm{bulk}}
+ x\\,(\\rho_v^{\\mathrm{bulk}} - \\rho_c^{\\mathrm{bulk}})` and the
degree of char is simply :math:`\\beta^* = 1 - x`.

The manufactured gas flux follows from Eq. (3.18) in closed form,

.. math::

    \\dot m_g^*(\\eta, t) = \\ell(t) \\int_\\eta^1
        \\partial_t \\rho^*\\, d\\eta'
    = -\\ell(t)\\, C(t)\\, \\tfrac{1}{2}\\Big[(1 - \\eta)
        - \\tfrac{\\sin \\pi\\eta}{\\pi}\\Big],

with :math:`C(t) = b\\,(\\rho_v - \\rho_c)^{\\mathrm{bulk}}
e^{-t/\\tau}/\\tau`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from aether.thermal import (
    CharringMaterial,
    LandauFrame,
    ThermalState,
    decomposition_rate,
)
from aether.thermal.solver import SourceField

__all__ = ["ManufacturedAblation"]

_FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ManufacturedAblation:
    """Closed-form manufactured trajectory and its MMS sources."""

    material: CharringMaterial
    frame: LandauFrame
    recession_initial: float = 0.0
    recession_velocity: float = 2.0e-4
    temperature_base: float = 400.0
    temperature_span: float = 1200.0
    temperature_wobble: float = 0.1
    omega: float = 1.0
    char_extent_base: float = 0.3
    char_extent_amplitude: float = 0.5
    tau: float = 5.0

    def __post_init__(self) -> None:
        lo = self.char_extent_base
        hi = self.char_extent_base + self.char_extent_amplitude
        if not 0.0 < lo < hi < 1.0:
            raise ValueError(
                f"char extent range ({lo}, {hi}) must sit strictly inside (0, 1) "
                f"so the clip and clamp stay inactive"
            )

    # ----------------------------------------------------------- recession
    def recession(self, t: float) -> float:
        return self.recession_initial + self.recession_velocity * t

    def recession_rate(self, _t: float) -> float:
        return self.recession_velocity

    def thickness(self, t: float) -> float:
        return self.frame.thickness(self.recession(t))

    # --------------------------------------------------------- temperature
    def temperature(self, eta: _FloatArray, t: float) -> _FloatArray:
        wob = 1.0 + self.temperature_wobble * np.sin(self.omega * t)
        return np.asarray(self.temperature_base + self.temperature_span * np.exp(-2.0 * eta) * wob)

    def temperature_t(self, eta: _FloatArray, t: float) -> _FloatArray:
        return np.asarray(
            self.temperature_span
            * np.exp(-2.0 * eta)
            * self.temperature_wobble
            * self.omega
            * np.cos(self.omega * t)
        )

    def temperature_eta(self, eta: _FloatArray, t: float) -> _FloatArray:
        wob = 1.0 + self.temperature_wobble * np.sin(self.omega * t)
        return np.asarray(-2.0 * self.temperature_span * np.exp(-2.0 * eta) * wob)

    def temperature_etaeta(self, eta: _FloatArray, t: float) -> _FloatArray:
        wob = 1.0 + self.temperature_wobble * np.sin(self.omega * t)
        return np.asarray(4.0 * self.temperature_span * np.exp(-2.0 * eta) * wob)

    # ------------------------------------------------------------- density
    @staticmethod
    def _shape(eta: _FloatArray) -> _FloatArray:
        return np.asarray(0.5 * (1.0 + np.cos(np.pi * eta)))

    @staticmethod
    def _shape_eta(eta: _FloatArray) -> _FloatArray:
        return np.asarray(-0.5 * np.pi * np.sin(np.pi * eta))

    @staticmethod
    def _shape_integral(eta: _FloatArray) -> _FloatArray:
        """:math:`\\int_\\eta^1 P\\,d\\eta' = \\tfrac{1}{2}[(1-\\eta) -
        \\sin(\\pi\\eta)/\\pi]`."""
        return np.asarray(0.5 * ((1.0 - eta) - np.sin(np.pi * eta) / np.pi))

    def char_extent(self, eta: _FloatArray, t: float) -> _FloatArray:
        return np.asarray(
            self.char_extent_base
            + self.char_extent_amplitude * np.exp(-t / self.tau) * self._shape(eta)
        )

    def partial_density(self, i: int, eta: _FloatArray, t: float) -> _FloatArray:
        comp = self.material.components[i]
        delta = comp.virgin_density - comp.char_density
        return np.asarray(comp.char_density + delta * self.char_extent(eta, t))

    def partial_density_t(self, i: int, eta: _FloatArray, t: float) -> _FloatArray:
        comp = self.material.components[i]
        delta = comp.virgin_density - comp.char_density
        return np.asarray(
            -delta
            * self.char_extent_amplitude
            / self.tau
            * np.exp(-t / self.tau)
            * self._shape(eta)
        )

    def partial_density_eta(self, i: int, eta: _FloatArray, t: float) -> _FloatArray:
        comp = self.material.components[i]
        delta = comp.virgin_density - comp.char_density
        return np.asarray(
            delta * self.char_extent_amplitude * np.exp(-t / self.tau) * self._shape_eta(eta)
        )

    @cached_property
    def _bulk_delta(self) -> float:
        return self.material.virgin_bulk_density - self.material.char_bulk_density

    def bulk_density_t(self, eta: _FloatArray, t: float) -> _FloatArray:
        return np.asarray(
            -self._bulk_delta
            * self.char_extent_amplitude
            / self.tau
            * np.exp(-t / self.tau)
            * self._shape(eta)
        )

    def char_fraction(self, eta: _FloatArray, t: float) -> _FloatArray:
        return np.asarray(1.0 - self.char_extent(eta, t))

    def char_fraction_eta(self, eta: _FloatArray, t: float) -> _FloatArray:
        return np.asarray(
            -self.char_extent_amplitude * np.exp(-t / self.tau) * self._shape_eta(eta)
        )

    def gas_flux(self, eta: _FloatArray, t: float) -> _FloatArray:
        """Manufactured :math:`\\dot m_g^*` from the closed-form integral of
        Eq. (3.18)."""
        c_t = (
            self._bulk_delta * self.char_extent_amplitude * np.exp(-t / self.tau) / self.tau
        )
        return np.asarray(-self.thickness(t) * c_t * self._shape_integral(eta))

    # ------------------------------------------------------------- sources
    def energy_source(self) -> SourceField:
        """Equation-level source :math:`g_T` making :math:`T^*` exact in
        Eq. (3.17) (W/m³)."""
        mat = self.material

        def g_t(eta: _FloatArray, t: float) -> _FloatArray:
            temp = self.temperature(eta, t)
            beta = self.char_fraction(eta, t)
            x = self.char_extent(eta, t)
            rho = mat.char_bulk_density + self._bulk_delta * x
            ell = self.thickness(t)
            sdot = self.recession_velocity

            t_eta = self.temperature_eta(eta, t)
            k = mat.conductivity.value(temp, beta)
            k_eta = mat.conductivity.d_temperature(temp, beta) * t_eta + (
                mat.conductivity.d_char_fraction(temp, beta) * self.char_fraction_eta(eta, t)
            )
            conduction = (k * self.temperature_etaeta(eta, t) + k_eta * t_eta) / (ell * ell)

            cp = mat.specific_heat.value(temp, beta)
            convection = (
                (self.gas_flux(eta, t) * mat.gas_specific_heat + rho * cp * sdot * (1.0 - eta))
                * t_eta
                / ell
            )
            pyrolysis = (mat.gas_enthalpy(temp) - mat.solid_enthalpy(temp)) * self.bulk_density_t(
                eta, t
            )
            return np.asarray(
                rho * cp * self.temperature_t(eta, t) - conduction - convection - pyrolysis
            )

        return g_t

    def density_source(self, i: int) -> SourceField:
        """Equation-level source :math:`g_{\\rho_i}` making
        :math:`\\rho_i^*` exact in Eqs. (3.14)+(3.16) (kg/(m³ s))."""
        comp = self.material.components[i]

        def g_rho(eta: _FloatArray, t: float) -> _FloatArray:
            temp = self.temperature(eta, t)
            rho_i = self.partial_density(i, eta, t)
            rate = decomposition_rate(comp, rho_i, temp)
            advect = self.frame.grid_velocity_coefficient(
                eta, self.recession(t), self.recession_velocity
            )
            return np.asarray(
                self.partial_density_t(i, eta, t)
                - rate
                - advect * self.partial_density_eta(i, eta, t)
            )

        return g_rho

    # ------------------------------------------------------------ closures
    def surface_rate(self, t: float) -> float:
        """Manufactured :math:`dT/dt` at the surface node (η = 0)."""
        return float(self.temperature_t(np.asarray(0.0), t))

    def back_face_rate(self, t: float) -> float:
        """Manufactured :math:`dT/dt` at the back face (η = 1)."""
        return float(self.temperature_t(np.asarray(1.0), t))

    def initial_state(self, eta: _FloatArray) -> ThermalState:
        """Exact manufactured state at :math:`t = 0` on the given nodes."""
        rho = np.vstack([self.partial_density(i, eta, 0.0) for i in range(3)])
        return ThermalState(
            temperature=self.temperature(eta, 0.0),
            partial_densities=rho,
            recession=self.recession_initial,
        )
