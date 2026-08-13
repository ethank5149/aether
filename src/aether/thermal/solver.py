"""Semi-discrete charring-ablation solver on the fixed Landau grid.

Method of lines for Paper I, Eqs. (3.14)–(3.18) on a Chebyshev grid over
:math:`\\eta \\in [0, 1]`. Node convention follows the descending CGL
ordering of :mod:`aether.spectral`: **index 0 is the back face**
(:math:`\\eta = 1`) and **index** :math:`N_T` **is the ablating surface**
(:math:`\\eta = 0`).

The state advanced in time is

.. math::

    \\big[\\mathbf{T},\\ \\rho_A,\\ \\rho_B,\\ \\rho_C,\\ s\\big]
    \\in \\mathbb{R}^{4(N_T+1)+1},

with every field stored at fixed :math:`\\eta` nodes; no node is ever
created, destroyed, or interpolated (Paper I, §3.4.2). The energy
equation (3.17) is collocated directly — the conduction term as the
nested spectral derivative :math:`\\partial_\\eta(k\\,\\partial_\\eta T)`
— and the gas-flux continuity equation (3.18) is solved spectrally each
evaluation: the first-derivative operator with the back-face condition
:math:`\\dot m_g(\\eta{=}1) = 0` imposed in place of its (rank-deficient)
constant mode is LU-factorized once at construction.

**Density-rate convention.** Eqs. (3.17)–(3.18) as printed source both
the pyrolysis-gas continuity and the decomposition enthalpy with
:math:`\\partial\\rho/\\partial t|_\\eta` — the full computational-frame
rate, including the grid-advection contribution. The material-frame
alternative (sourcing them with the Arrhenius rate alone, the CMA
convention) differs at :math:`\\mathcal{O}(\\dot s\\,\\rho_\\eta/\\ell)`.
This solver implements the paper's letter by default and exposes the
choice as ``density_rate_convention`` rather than deciding silently;
V4's manufactured solutions verify the default.

The density transport itself needs no boundary closure: its advection
velocity :math:`-\\dot s(1-\\eta)/\\ell` vanishes at the inflow face
:math:`\\eta = 1`, so there is no inflow characteristic to feed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from aether.spectral import ChebyshevGrid
from aether.thermal.kinetics import bulk_density, decomposition_rate, degree_of_char
from aether.thermal.landau import LandauFrame
from aether.thermal.material import CharringMaterial

__all__ = ["CharringThermalSolver", "ThermalState"]

_FloatArray = NDArray[np.float64]
#: (t, state) -> recession rate sdot (m/s)
RecessionRate = Callable[[float, "ThermalState"], float]
#: (t,) -> dT/dt (K/s) imposed at a boundary node (Dirichlet-rate closure)
BoundaryRate = Callable[[float], float]
#: (eta, t) -> equation-level source, added to the RHS of its PDE
SourceField = Callable[[_FloatArray, float], _FloatArray]


@dataclass(frozen=True)
class ThermalState:
    """Unpacked thermal state at one instant.

    Attributes
    ----------
    temperature:
        Nodal temperatures (K), back face first (descending-η CGL order).
    partial_densities:
        Component densities, shape ``(3, N_T + 1)`` ordered (A, B, C).
    recession:
        Recession depth :math:`s` (m).
    """

    temperature: _FloatArray
    partial_densities: _FloatArray
    recession: float

    @property
    def surface_temperature(self) -> float:
        """Temperature at the ablating surface (η = 0, last node)."""
        return float(self.temperature[-1])


class CharringThermalSolver:
    """Method-of-lines right-hand side for the in-depth ablation system.

    Parameters
    ----------
    grid:
        Chebyshev grid with ``interval == (0, 1)`` and at least first
        and second derivative operators.
    material:
        Charring material model.
    frame:
        Landau transform for the TPS stack thickness.
    density_rate_convention:
        ``"eta_frame"`` (Paper I, Eqs. 3.17–3.18 as printed, default) or
        ``"material_frame"`` (CMA convention); see the module docstring.
    """

    def __init__(
        self,
        grid: ChebyshevGrid,
        material: CharringMaterial,
        frame: LandauFrame,
        density_rate_convention: Literal["eta_frame", "material_frame"] = "eta_frame",
    ) -> None:
        if grid.interval != (0.0, 1.0):
            raise ValueError(
                f"thermal grid must live on the Landau domain (0, 1), got {grid.interval}"
            )
        if grid.max_derivative < 2:
            raise ValueError("thermal grid needs derivative operators up to order 2")
        if density_rate_convention not in ("eta_frame", "material_frame"):
            raise ValueError(
                f"density_rate_convention must be 'eta_frame' or 'material_frame', "
                f"got {density_rate_convention!r}"
            )
        self._grid = grid
        self._material = material
        self._frame = frame
        self._convention = density_rate_convention
        self._eta = grid.x  # descending: eta[0] = 1 (back face), eta[-1] = 0 (surface)
        self._d1 = grid.diffmat(1)

        # Gas-flux operator: d(mdot)/d(eta) rows with the back-face row
        # replaced by the boundary condition mdot(eta = 1) = 0.
        flux_op = self._d1.copy()
        flux_op[0, :] = 0.0
        flux_op[0, 0] = 1.0
        self._flux_lu = scipy.linalg.lu_factor(flux_op, check_finite=True)

    @property
    def grid(self) -> ChebyshevGrid:
        return self._grid

    @property
    def material(self) -> CharringMaterial:
        return self._material

    @property
    def frame(self) -> LandauFrame:
        return self._frame

    @property
    def state_size(self) -> int:
        return 4 * self._grid.size + 1

    # ------------------------------------------------------------ state I/O
    def pack(self, state: ThermalState) -> _FloatArray:
        """Flatten a :class:`ThermalState` into the ODE vector."""
        if state.temperature.shape != (self._grid.size,):
            raise ValueError(
                f"temperature must have shape ({self._grid.size},), "
                f"got {state.temperature.shape}"
            )
        if state.partial_densities.shape != (3, self._grid.size):
            raise ValueError(
                f"partial_densities must have shape (3, {self._grid.size}), "
                f"got {state.partial_densities.shape}"
            )
        return np.concatenate(
            [state.temperature, state.partial_densities.reshape(-1), [state.recession]]
        )

    def unpack(self, vector: _FloatArray) -> ThermalState:
        """Inverse of :meth:`pack`."""
        m = self._grid.size
        vec = np.asarray(vector, dtype=np.float64)
        if vec.shape != (self.state_size,):
            raise ValueError(f"state vector must have shape ({self.state_size},), got {vec.shape}")
        return ThermalState(
            temperature=vec[:m],
            partial_densities=vec[m : 4 * m].reshape(3, m),
            recession=float(vec[-1]),
        )

    # ------------------------------------------------------------ operators
    def gas_flux(self, density_rate_eta: _FloatArray, thickness: float) -> _FloatArray:
        """Pyrolysis gas flux from Eq. (3.18):
        :math:`\\ell^{-1}\\partial_\\eta \\dot m_g = -\\dot\\rho`,
        integrated spectrally from :math:`\\dot m_g(\\eta{=}1) = 0`."""
        rhs = -float(thickness) * np.asarray(density_rate_eta, dtype=np.float64)
        rhs = rhs.copy()
        rhs[0] = 0.0  # back-face boundary condition row
        return cast(_FloatArray, scipy.linalg.lu_solve(self._flux_lu, rhs))

    # ------------------------------------------------------------------ RHS
    def rhs(
        self,
        t: float,
        y: _FloatArray,
        recession_rate: RecessionRate,
        surface_rate: BoundaryRate | None = None,
        back_face_rate: BoundaryRate | None = None,
        energy_source: SourceField | None = None,
        density_sources: tuple[SourceField, SourceField, SourceField] | None = None,
    ) -> _FloatArray:
        """Time derivative of the packed state (Paper I, Eqs. 3.14–3.18).

        Parameters
        ----------
        t, y:
            Time and packed state (``solve_ivp`` signature via closure).
        recession_rate:
            Supplies :math:`\\dot s` from the surface model (thermo-
            chemistry table in flight; prescribed in verification).
        surface_rate, back_face_rate:
            Dirichlet-rate closures: when given, the energy ODE at the
            corresponding boundary node is replaced by the prescribed
            :math:`dT/dt`. Verification imposes the manufactured rates;
            flight operation will close the surface node with the
            surface-energy-balance solve instead.
        energy_source, density_sources:
            Equation-level manufactured sources: ``energy_source`` adds
            to the right side of Eq. (3.17) in W/m³; each density source
            adds to the right side of its component's Eq. (3.14)+(3.16)
            rate in kg/(m³ s).
        """
        state = self.unpack(y)
        temp = state.temperature
        if np.any(temp <= 0.0) or not np.all(np.isfinite(temp)):
            raise FloatingPointError(
                "non-physical temperature field encountered; the integrator has "
                "left the model's domain of validity"
            )
        mat = self._material
        eta = self._eta
        ell = self._frame.thickness(state.recession)
        sdot = float(recession_rate(t, state))

        rho_a, rho_b, rho_c = state.partial_densities
        rho = bulk_density(mat, rho_a, rho_b, rho_c)
        beta = degree_of_char(mat, rho)

        # --- component density rates: Arrhenius (material frame) plus the
        # grid-velocity advection of Eq. (3.16), plus manufactured sources.
        advect = self._frame.grid_velocity_coefficient(eta, state.recession, sdot)
        rho_dot_eta = np.empty_like(state.partial_densities)
        rho_dot_material = np.empty_like(state.partial_densities)
        for i, (component, rho_i) in enumerate(
            zip(mat.components, (rho_a, rho_b, rho_c), strict=True)
        ):
            rate = decomposition_rate(component, rho_i, temp)
            rho_dot_material[i] = rate
            rho_dot_eta[i] = rate + advect * (self._d1 @ rho_i)
            if density_sources is not None:
                rho_dot_eta[i] = rho_dot_eta[i] + density_sources[i](eta, t)

        g = mat.resin_fraction
        bulk_rate_eta = g * (rho_dot_eta[0] + rho_dot_eta[1]) + (1.0 - g) * rho_dot_eta[2]
        bulk_rate_material = (
            g * (rho_dot_material[0] + rho_dot_material[1]) + (1.0 - g) * rho_dot_material[2]
        )
        source_rate = bulk_rate_eta if self._convention == "eta_frame" else bulk_rate_material

        # --- pyrolysis gas flux, Eq. (3.18)
        mdot_g = self.gas_flux(source_rate, ell)

        # --- energy equation, Eq. (3.17)
        k = mat.conductivity.value(temp, beta)
        cp = mat.specific_heat.value(temp, beta)
        t_eta = self._d1 @ temp
        conduction = (self._d1 @ (k * t_eta)) / (ell * ell)
        convection = (mdot_g * mat.gas_specific_heat + rho * cp * sdot * (1.0 - eta)) * t_eta / ell
        pyrolysis = (mat.gas_enthalpy(temp) - mat.solid_enthalpy(temp)) * source_rate
        rhs_energy = conduction + convection + pyrolysis
        if energy_source is not None:
            rhs_energy = rhs_energy + energy_source(eta, t)
        temp_dot = rhs_energy / (rho * cp)

        if back_face_rate is not None:
            temp_dot[0] = back_face_rate(t)
        if surface_rate is not None:
            temp_dot[-1] = surface_rate(t)

        return np.concatenate([temp_dot, rho_dot_eta.reshape(-1), [sdot]])
