"""Fully implicit ablation and thermal response solver.

An independent implementation of the formulation published in Chen &
Milos, *J. Spacecraft and Rockets* **36**(3), 1999, pp. 475–483, and
Milos, Chen & Squire, TFAWS06-1008, 2006. FIAT itself is
US-government-controlled software and is not used, referenced at
runtime, or reproduced here; what follows is written from the governing
equations and numerical description in those two open-literature papers.

Equations solved, in the source's numbering
-------------------------------------------

**Eq. (1), internal energy balance**, in the surface-fixed coordinate
:math:`x = y - s`:

.. math::

    \\rho c_p \\frac{\\partial T}{\\partial\\theta}\\bigg|_x =
      \\frac{\\partial}{\\partial x}\\!\\left(k\\frac{\\partial T}{\\partial x}
      - q_R\\right)_\\theta
    + (h_g - \\bar h)\\frac{\\partial\\rho}{\\partial\\theta}\\bigg|_y
    + \\dot s\\,\\rho c_p \\frac{\\partial T}{\\partial x}\\bigg|_\\theta
    + \\dot m_g \\frac{\\partial h_g}{\\partial x}\\bigg|_\\theta

**Eq. (7), composite density** and **Eq. (8), decomposition**:

.. math::

    \\rho = \\Gamma(\\rho_A + \\rho_B) + (1-\\Gamma)\\rho_C, \\qquad
    \\frac{\\partial \\rho_i}{\\partial\\theta}\\bigg|_y =
      -A_i e^{-E_i/RT}\\rho_{v i}
      \\left(\\frac{\\rho_i - \\rho_{r i}}{\\rho_{v i}}\\right)^{\\psi_i}

**Eq. (9), internal mass balance**, quasi-steady with an impermeable
backface: :math:`\\partial \\dot m_g/\\partial y = \\partial\\rho/\\partial\\theta`.

**Eqs. (4)–(6)**, the virgin/char mixing rules, and **Eq. (10)/(11)**,
the surface energy balance with its blowing correction, are in
:mod:`aether.fiat.surface`.

Discretisation
--------------

Finite volume in space, backward Euler in time, and a Newton solve over
the coupled unknowns — matching FIAT's "fully implicit" defining
property, which is the whole reason it exists: *"in the CMA code, the
in-depth energy equation is linked explicitly to the internal
decomposition and ablating surface equations ... Thus, CMA solutions are
sensitive to the user-specified time step as well as to grid size."*

Three implementation choices differ from the published description and
are recorded rather than buried.

**The Jacobian is analytic and dense, not block-tridiagonal.** FIAT keeps
:math:`(T, q_c, \\rho, g, \\rho_i)` in a block-tridiagonal system. Here
:math:`\\rho_i` is eliminated per cell by implicit differentiation of
Eq. (8) and :math:`\\dot m_g` by the Eq. (9) quadrature, leaving
:math:`(T, T_w, T_b)` as the Newton unknowns. Two couplings that
elimination introduces are carried in full rather than lagged to
preserve a band: the dependence of each cell's gas flux on decomposition
in every cell below it, and the surface row's dependence on the whole
temperature field through the same integral. The surface closure runs
through interpolated :math:`B'` tables, so its two derivatives are taken
by central difference on the closure itself — the tables *are* the
model, and there is no analytic form underneath them to prefer.

The result agrees with a central difference of the residual to the
difference's own truncation error, which the test suite checks. Getting
there mattered: an earlier version omitted the gas-convection and
grid-advection face weights, and Newton stalled at a scaled residual of
:math:`10^{-1}` in exactly the regime — an active pyrolysis front — the
solver exists for.

**Recession stretches the top ply rather than consuming cells** — see
:mod:`aether.fiat.stack`.

**Recession is converged, not lagged.** The grid geometry depends on
:math:`s`, which depends on :math:`T_w`, which is a Newton unknown on
that geometry. Rather than take one pass, the step iterates grid and
Newton solve to a stated tolerance, so the scheme is implicit in
recession too.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from aether.fiat.bprime import BPrimeTable, TableRangeError
from aether.fiat.materials import (
    MultiComponentMaterial,
    PressureConductivity,
    TabulatedConductivity,
)
from aether.fiat.radiation import (
    gray_radiative_flux,
    optical_depth,
    rosseland_conductivity,
)
from aether.fiat.stack import MaterialLike, MaterialStack
from aether.fiat.surface import (
    AerothermalEnvironment,
    BackfaceCondition,
    BackfaceKind,
    SurfaceState,
    solve_surface,
)
from aether.thermal.material import GAS_CONSTANT, ArrheniusComponent
from aether.thermal.surface import STEFAN_BOLTZMANN

__all__ = [
    "FiatSolution",
    "FiatSolver",
    "FiatStep",
    "RadiationMode",
    "SolverOptions",
]

_FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class _StepContext:
    """Everything held fixed while the Newton iteration runs."""

    previous_temperature: _FloatArray
    previous_components: _FloatArray
    dt: float
    recession_rate: float
    widths: _FloatArray
    centers: _FloatArray
    environment: AerothermalEnvironment
    backface: BackfaceCondition
    table: BPrimeTable


#: In-depth radiation treatment.
RadiationMode = str  # "none" | "rosseland" | "gray"
_RADIATION_MODES = ("none", "rosseland", "gray")


@dataclass(frozen=True)
class SolverOptions:
    """Numerical controls.

    Attributes
    ----------
    newton_tolerance:
        Convergence threshold on the scaled energy residual (-).
    max_newton_iterations:
        Cap per outer recession iteration.
    recession_tolerance:
        Relative convergence threshold on :math:`s` across the outer
        iteration. Set to ``inf`` to lag recession by one step (CMA-like)
        — allowed for comparison studies, never the default.
    max_recession_iterations:
        Cap on the outer loop.
    radiation:
        ``"none"``, ``"rosseland"`` (Eq. 3, implicit) or ``"gray"``
        (Eq. 2, explicit source).
    min_temperature:
        Floor applied to Newton iterates (K). Arrhenius rates and
        :math:`T^4` reradiation both misbehave for non-physical negative
        temperatures that a Newton step can transiently produce.
    """

    newton_tolerance: float = 1.0e-9
    max_newton_iterations: int = 50
    recession_tolerance: float = 1.0e-8
    max_recession_iterations: int = 20
    radiation: RadiationMode = "none"
    min_temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.radiation not in _RADIATION_MODES:
            raise ValueError(
                f"radiation must be one of {_RADIATION_MODES}, got {self.radiation!r}"
            )
        for name in ("newton_tolerance", "recession_tolerance", "min_temperature"):
            value = getattr(self, name)
            if not (value > 0.0 and not np.isnan(value)):
                raise ValueError(f"{name} must be > 0, got {value}")
        for name in ("max_newton_iterations", "max_recession_iterations"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")


@dataclass
class FiatStep:
    """State after one converged time step."""

    time: float
    temperature: _FloatArray
    """Cell-centre temperatures (K), length ``n_cells``."""
    component_density: _FloatArray
    """Per-component densities (kg/m³), shape ``(n_cells, 3)``."""
    wall_temperature: float
    backface_temperature: float
    recession: float
    recession_rate: float
    gas_flux: _FloatArray
    """:math:`\\dot m_g` at cell centres (kg/(m² s)), length ``n_cells``."""
    surface: SurfaceState
    newton_iterations: int
    recession_iterations: int


@dataclass
class FiatSolution:
    """A whole run."""

    times: _FloatArray
    steps: list[FiatStep] = field(default_factory=list)

    @property
    def recession(self) -> _FloatArray:
        return np.array([s.recession for s in self.steps])

    @property
    def wall_temperature(self) -> _FloatArray:
        return np.array([s.wall_temperature for s in self.steps])

    @property
    def backface_temperature(self) -> _FloatArray:
        return np.array([s.backface_temperature for s in self.steps])

    def temperature_history(self) -> _FloatArray:
        """Shape ``(n_steps, n_cells)``."""
        return np.array([s.temperature for s in self.steps])


def _components(material: MaterialLike) -> tuple[ArrheniusComponent, ...]:
    """Solid components of a material, however many it has."""
    if isinstance(material, MultiComponentMaterial):
        return material.components
    return (material.resin_a, material.resin_b, material.filler)


def _component_weights(material: MaterialLike) -> tuple[float, ...]:
    """:math:`w_i` of the generalised Eq. (7).

    The classical three-component form is the special case
    :math:`w = (\\Gamma, \\Gamma, 1-\\Gamma)`.
    """
    if isinstance(material, MultiComponentMaterial):
        return material.weights
    g = material.resin_fraction
    return (g, g, 1.0 - g)


def _virgin_density(material: MaterialLike) -> float:
    return float(
        sum(
            w * c.virgin_density
            for w, c in zip(_component_weights(material), _components(material), strict=True)
        )
    )


def _char_density(material: MaterialLike) -> float:
    return float(
        sum(
            w * c.char_density
            for w, c in zip(_component_weights(material), _components(material), strict=True)
        )
    )


class FiatSolver:
    """Implicit ablation and thermal response of a multilayer stack.

    Parameters
    ----------
    stack:
        The ply stack, heated face first.
    options:
        Numerical controls.
    """

    def __init__(self, stack: MaterialStack, options: SolverOptions | None = None) -> None:
        self._stack = stack
        self._opt = options or SolverOptions()
        self._materials = stack.cell_materials()
        self._n = stack.n_cells
        self._virgin = np.array([_virgin_density(m) for m in self._materials])
        self._char = np.array([_char_density(m) for m in self._materials])
        self._n_top = stack.plies[0].n_cells
        self._pressure_k = [
            stack.plies[i].pressure_conductivity for i in stack.grid(0.0).ply_index
        ]
        self._any_pressure_k = any(c is not None for c in self._pressure_k)
        # Per-cell Arrhenius parameter arrays, so the kinetics update is a
        # vectorised (n_cells, 3) operation rather than a Python loop.
        comps = [_components(m) for m in self._materials]
        self._pre = np.array([[c.pre_exponential for c in row] for row in comps])
        self._act = np.array([[c.activation_energy for c in row] for row in comps])
        self._order = np.array([[c.reaction_order for c in row] for row in comps])
        self._rho_v = np.array([[c.virgin_density for c in row] for row in comps])
        self._rho_r = np.array([[c.char_density for c in row] for row in comps])
        n_components = len(comps[0])
        if any(len(row) != n_components for row in comps):
            raise ValueError(
                "every ply must carry the same number of solid components; "
                "mixing a 3-component and a 7-component material in one stack "
                "would make the packed state ambiguous"
            )
        self._weights = np.array(
            [list(_component_weights(m)) for m in self._materials]
        )
        self._gas_slope = np.array([m.gas_enthalpy_slope for m in self._materials])
        self._solid_slope = np.array([m.solid_enthalpy_slope for m in self._materials])
        ext = [
            stack.plies[i].extinction_coefficient for i in stack.grid(0.0).ply_index
        ]
        self._opaque = any(e is None for e in ext)
        self._extinction = np.array([e if e is not None else np.inf for e in ext])

    @property
    def stack(self) -> MaterialStack:
        return self._stack

    @property
    def n_cells(self) -> int:
        return self._n

    # -- constitutive helpers ------------------------------------------------

    def virgin_mass_fraction(self, bulk_density: _FloatArray) -> _FloatArray:
        """FIAT Eq. (5), :math:`\\tau = (1-\\rho_c/\\rho)/(1-\\rho_c/\\rho_v)`.

        Clipped to :math:`[0, 1]`: a converged density always lies between
        char and virgin, but a Newton iterate en route need not, and an
        out-of-range :math:`\\tau` would push the blended properties into
        physically meaningless territory before the iteration recovers.
        """
        rho = np.maximum(bulk_density, 1e-12)
        tau = (1.0 - self._char / rho) / (1.0 - self._char / self._virgin)
        return np.asarray(np.clip(tau, 0.0, 1.0))

    def _properties(
        self,
        temperature: _FloatArray,
        bulk_density: _FloatArray,
        pressure: float | None = None,
    ) -> tuple[_FloatArray, _FloatArray]:
        """Conductivity (W/(m K)) and specific heat (J/(kg K)) per cell.

        FIAT Eq. (4) blends by virgin mass fraction :math:`\\tau`; the
        material model here is parameterised by char fraction
        :math:`\\beta = 1 - \\tau`, which is the same rule written the
        other way round.
        """
        beta = 1.0 - self.virgin_mass_fraction(bulk_density)
        k = np.array(
            [
                m.conductivity.value(t, b)
                for m, t, b in zip(self._materials, temperature, beta, strict=True)
            ]
        )
        if self._any_pressure_k:
            if pressure is None:
                raise ValueError(
                    "a ply carries a pressure_conductivity, so _properties needs "
                    "the surface pressure; this is a solver bug if it surfaces"
                )
            for i, prop in enumerate(self._pressure_k):
                if prop is not None:
                    k[i] = float(prop.value(temperature[i], beta[i], pressure))
        cp = np.array(
            [
                m.specific_heat.value(t, b)
                for m, t, b in zip(self._materials, temperature, beta, strict=True)
            ]
        )
        return k, cp

    def _gas_enthalpy(self, temperature: _FloatArray) -> _FloatArray:
        offset = np.array([m.gas_enthalpy_offset for m in self._materials])
        return np.asarray(offset + self._gas_slope * temperature)

    def _partial_char_enthalpy(self, temperature: _FloatArray) -> _FloatArray:
        """:math:`\\bar h` of FIAT Eq. (6), supplied directly by the model."""
        offset = np.array([m.solid_enthalpy_offset for m in self._materials])
        return np.asarray(offset + self._solid_slope * temperature)

    # -- decomposition -------------------------------------------------------

    def decompose(
        self, temperature: _FloatArray, previous: _FloatArray, dt: float
    ) -> tuple[_FloatArray, _FloatArray]:
        """Implicit Eq. (8) update and its temperature sensitivity.

        Returns ``(rho_i, d rho_i / dT)``, both shape ``(n_cells, 3)``.

        Backward Euler on Eq. (8) is scalar and independent per component
        per cell, so it is solved by a damped Newton confined to the
        physically admissible interval :math:`[\\rho_{r i},
        \\rho_i^{\\,\\mathrm{old}}]` — density can only fall, and never
        below the residual char. The sensitivity comes from implicit
        differentiation of the converged relation, which is exact rather
        than a difference of two solves.
        """
        t = np.maximum(temperature, self._opt.min_temperature)[:, None]
        rho = previous.copy()
        floor = self._rho_r
        active = (self._pre > 0.0) & (previous > floor + 1e-12)

        rate_const = np.zeros_like(rho)
        np.divide(-self._act, GAS_CONSTANT * t, out=rate_const, where=active)
        k_arr = np.where(active, self._pre * np.exp(rate_const), 0.0)

        def rate(r: _FloatArray) -> _FloatArray:
            excess = np.maximum(r - floor, 0.0) / self._rho_v
            return np.asarray(k_arr * self._rho_v * excess**self._order)

        def drate(r: _FloatArray) -> _FloatArray:
            excess = np.maximum(r - floor, 0.0) / self._rho_v
            with np.errstate(divide="ignore", invalid="ignore"):
                d = np.where(
                    excess > 0.0,
                    k_arr * self._order * excess ** np.maximum(self._order - 1.0, 0.0),
                    0.0,
                )
            return np.asarray(np.nan_to_num(d))

        for _ in range(60):
            residual = rho - previous + dt * rate(rho)
            jac = 1.0 + dt * drate(rho)
            step = np.where(active, residual / np.maximum(jac, 1e-30), 0.0)
            rho_new = np.clip(rho - step, floor, previous)
            if np.max(np.abs(rho_new - rho)) <= 1e-14 * np.maximum(previous, 1.0).max():
                rho = rho_new
                break
            rho = rho_new

        # d rho / dT from F(rho, T) = rho - rho_old + dt * f(rho, T) = 0,
        # with d f / d T = f * E / (R T^2).
        f = rate(rho)
        df_dt = f * self._act / (GAS_CONSTANT * t**2)
        drho_dt = np.where(active, -dt * df_dt / (1.0 + dt * drate(rho)), 0.0)
        return rho, np.asarray(drho_dt)

    def bulk_density(self, component_density: _FloatArray) -> _FloatArray:
        """FIAT Eq. (7)."""
        return np.asarray(np.sum(self._weights * component_density, axis=1))

    # -- residual ------------------------------------------------------------

    def _face_conductivity(self, k: _FloatArray, widths: _FloatArray) -> _FloatArray:
        """Harmonic-mean conductivity at interior faces.

        The harmonic mean is not a smoothing choice — it is the only
        average that makes the discrete flux exact for a steady
        one-dimensional problem across a conductivity jump, which is
        precisely what every ply interface in the stack is. An arithmetic
        mean would leak energy there, and the leak would be largest at the
        bondline, where the answer matters most.
        """
        half_left, half_right = 0.5 * widths[:-1], 0.5 * widths[1:]
        return np.asarray(
            (half_left + half_right) / (half_left / k[:-1] + half_right / k[1:])
        )


    def _residual(
        self, unknowns: _FloatArray, ctx: _StepContext
    ) -> tuple[_FloatArray, dict[str, object]]:
        """Full residual over ``[T_0..T_{n-1}, T_w, T_b]``.

        Rows ``0..n-1`` are the finite-volume form of Eq. (1), row ``n``
        is the surface energy balance Eq. (10), and row ``n+1`` is the
        backface closure.
        """
        n = self._n
        opt = self._opt
        t_cell = np.maximum(unknowns[:n], opt.min_temperature)
        t_wall = float(max(unknowns[n], opt.min_temperature))
        t_back = float(max(unknowns[n + 1], opt.min_temperature))
        widths, centers = ctx.widths, ctx.centers

        rho_i, drho_i_dt = self.decompose(t_cell, ctx.previous_components, ctx.dt)
        rho = self.bulk_density(rho_i)
        drho_dt = np.sum(self._weights * drho_i_dt, axis=1)
        k, cp = self._properties(t_cell, rho, ctx.environment.pressure)
        if opt.radiation == "rosseland":
            k = k + rosseland_conductivity(t_cell, self._extinction)

        # Eq. (8) integrated over the step; omega <= 0.
        omega = (rho - self.bulk_density(ctx.previous_components)) / ctx.dt
        domega_dt = drho_dt / ctx.dt

        # Eq. (9), integrated up from an impermeable backface.
        cell_source = omega * widths
        below = np.concatenate([np.cumsum(cell_source[::-1])[::-1][1:], [0.0]])
        gas_flux = -(below + 0.5 * cell_source)
        surface_gas_flux = float(-np.sum(cell_source))

        h_g = self._gas_enthalpy(t_cell)
        h_bar = self._partial_char_enthalpy(t_cell)
        gas_slope = self._gas_slope
        top = self._materials[0]

        k_face = self._face_conductivity(k, widths)
        distance = centers[1:] - centers[:-1]
        w_right = 0.5 * widths[:-1] / distance
        t_face = np.empty(n + 1)
        t_face[0] = t_wall
        t_face[1:n] = (1.0 - w_right) * t_cell[:-1] + w_right * t_cell[1:]
        t_face[n] = t_back

        q = np.empty(n + 1)
        q[0] = k[0] * (t_wall - t_cell[0]) / (0.5 * widths[0])
        q[1:n] = -k_face * (t_cell[1:] - t_cell[:-1]) / distance
        q[n] = ctx.backface.flux(float(t_cell[-1]), t_back, float(k[-1]), 0.5 * widths[-1])

        if opt.radiation == "gray":
            kappa = optical_depth(widths, self._extinction)
            q = q + gray_radiative_flux(
                kappa,
                t_cell,
                front_intensity=ctx.environment.wall_emissivity
                * STEFAN_BOLTZMANN
                * t_wall**4
                / np.pi,
            )

        # Grid motion: only the top ply moves relative to the material, with a
        # coefficient tapering from s-dot at the surface to zero at its inner
        # face, below which every node is glued to the material.
        advect = np.zeros(n)
        if ctx.recession_rate != 0.0 and self._n_top > 0:
            top_thickness = float(np.sum(widths[: self._n_top]))
            eta = centers[: self._n_top] / top_thickness
            advect[: self._n_top] = ctx.recession_rate * (1.0 - eta)

        h_face = np.empty(n + 1)
        h_face[0] = top.gas_enthalpy_offset + gas_slope[0] * t_wall
        h_face[1:n] = (1.0 - w_right) * h_g[:-1] + w_right * h_g[1:]
        h_face[n] = h_g[-1]

        storage = rho * cp * widths / ctx.dt
        residual = np.empty(n + 2)
        residual[:n] = (
            storage * (t_cell - ctx.previous_temperature)
            - (q[:-1] - q[1:])
            - (h_g - h_bar) * omega * widths
            - advect * rho * cp * (t_face[1:] - t_face[:-1])
            - gas_flux * (h_face[1:] - h_face[:-1])
        )

        residual[n], surface = self._surface(
            t_wall, surface_gas_flux, float(q[0]), ctx
        )
        residual[n + 1] = ctx.backface.closure(
            float(t_cell[-1]), t_back, float(k[-1]), 0.5 * widths[-1]
        )

        info: dict[str, object] = {
            "component_density": rho_i,
            "bulk_density": rho,
            "gas_flux": gas_flux,
            "surface_gas_flux": surface_gas_flux,
            "conductivity": k,
            "specific_heat": cp,
            "storage": storage,
            "k_face": k_face,
            "distance": distance,
            "domega_dt": domega_dt,
            "omega": omega,
            "h_g": h_g,
            "h_bar": h_bar,
            "surface": surface,
            "wall_flux": float(q[0]),
            "w_right": w_right,
            "advect": advect,
            "t_face": t_face,
            "h_face": h_face,
            "gas_slope": gas_slope,
        }
        return residual, info

    def _surface(
        self, t_wall: float, gas_mass_flux: float, conduction_flux: float, ctx: _StepContext
    ) -> tuple[float, SurfaceState]:
        """Eq. (10) at the current surface state."""
        top = self._materials[0]
        return solve_surface(
            t_wall,
            gas_mass_flux,
            conduction_flux,
            ctx.environment,
            ctx.table,
            self._char[0],
            char_enthalpy=float(
                top.solid_enthalpy_offset + top.solid_enthalpy_slope * t_wall
            ),
            gas_enthalpy=float(
                top.gas_enthalpy_offset + top.gas_enthalpy_slope * t_wall
            ),
        )

    def _face_weight_matrix(
        self, w_right: _FloatArray, slope: _FloatArray | None
    ) -> _FloatArray:
        """:math:`\\partial(\\text{face value})/\\partial u`, shape ``(n+1, n+2)``.

        Face values are width-weighted interpolants of the two adjacent
        cells, with the heated face taking :math:`T_w` and the backface
        :math:`T_b`. ``slope`` converts a temperature sensitivity into an
        enthalpy one; ``None`` means the face value *is* temperature.
        """
        n = self._n
        m = np.zeros((n + 1, n + 2))
        d = np.ones(n) if slope is None else slope
        m[0, n] = d[0]
        rows = np.arange(1, n)
        m[rows, rows - 1] = (1.0 - w_right) * d[:-1]
        m[rows, rows] = w_right * d[1:]
        m[n, n - 1] = d[-1]
        return m

    def _gas_flux_sensitivity(
        self, domega_dt: _FloatArray, widths: _FloatArray
    ) -> _FloatArray:
        """:math:`\\partial \\dot m_g/\\partial u`, shape ``(n, n+2)``.

        Eq. (9) integrates decomposition upward from the backface, so the
        gas flux at one cell depends on every cell below it. That is the
        lower-triangular fill FIAT avoids by keeping ``g`` as a block
        unknown; carrying it explicitly is cheaper here than restructuring
        the solve around it, and dropping it was what stalled the Newton
        iteration during development.
        """
        n = self._n
        contribution = domega_dt * widths
        m = np.zeros((n, n + 2))
        # gas_flux[j] = -(sum_{i>j} omega_i w_i + 0.5 omega_j w_j)
        upper = np.triu(np.ones((n, n)), k=1)
        m[:, :n] = -(upper * contribution[None, :])
        m[np.arange(n), np.arange(n)] = -0.5 * contribution
        return m

    def _jacobian(
        self, unknowns: _FloatArray, ctx: _StepContext, info: dict[str, object]
    ) -> _FloatArray:
        """Jacobian of :meth:`_residual` over ``[T, T_w, T_b]``.

        Every term of the residual is differentiated except the
        temperature dependence of the thermal conductivity, which enters
        only through the harmonic face mean and whose slope is three
        orders of magnitude below the conductivity itself for the
        materials this solver is used on.

        That makes the scheme very slightly *inexact* Newton, and the
        distinction is worth being precise about: the converged state is
        the one that drives :meth:`_residual` below
        :attr:`SolverOptions.newton_tolerance`, and the residual carries
        every term of Eqs. (1) and (7)–(11) exactly. An omitted Jacobian
        term costs iterations and cannot bias the answer.
        """
        n = self._n
        widths = ctx.widths
        storage = np.asarray(info["storage"])
        k = np.asarray(info["conductivity"])
        cp = np.asarray(info["specific_heat"])
        rho = np.asarray(info["bulk_density"])
        k_face = np.asarray(info["k_face"])
        distance = np.asarray(info["distance"])
        domega_dt = np.asarray(info["domega_dt"])
        omega = np.asarray(info["omega"])
        h_g = np.asarray(info["h_g"])
        h_bar = np.asarray(info["h_bar"])
        gas_flux = np.asarray(info["gas_flux"])
        w_right = np.asarray(info["w_right"])
        advect = np.asarray(info["advect"])
        h_face = np.asarray(info["h_face"])
        t_face = np.asarray(info["t_face"])
        surface: SurfaceState = info["surface"]  # type: ignore[assignment]
        t_wall = float(max(unknowns[n], self._opt.min_temperature))
        t_back = float(max(unknowns[n + 1], self._opt.min_temperature))
        t_cell = np.maximum(unknowns[:n], self._opt.min_temperature)

        jac = np.zeros((n + 2, n + 2))
        idx = np.arange(n)

        # Storage, including the density and specific-heat sensitivities.
        drho_dt = domega_dt * ctx.dt
        dcp_dt = self._property_derivative("specific_heat", t_cell, rho, drho_dt)
        dk_dt = self._property_derivative(
            "conductivity", t_cell, rho, drho_dt, ctx.environment.pressure
        )
        jac[idx, idx] += storage + (
            (drho_dt * cp + rho * dcp_dt) * widths / ctx.dt
        ) * (t_cell - ctx.previous_temperature)

        # Pyrolysis source, -(h_g - h_bar) * omega * width.
        jac[idx, idx] -= (
            (self._gas_slope - self._solid_slope) * omega
            + (h_g - h_bar) * domega_dt
        ) * widths

        # Conduction, including the temperature dependence of k. At interior
        # faces that term is a small correction; at the heated face, where the
        # gradient is (T_w - T_0) over half a cell, it is not, and leaving it
        # out visibly degrades Newton convergence during a heat pulse.
        half0 = 0.5 * widths[0]
        halfn = 0.5 * widths[-1]
        for f in range(1, n):
            j, i = f, f - 1
            d = distance[f - 1]
            delta_t = t_cell[j] - t_cell[i]
            k_f = k_face[f - 1]
            a, b = 0.5 * widths[i], 0.5 * widths[j]
            # d(harmonic mean)/d k_i and /d k_j.
            denom = a / k[i] + b / k[j]
            dk_f_di = (a / k[i] ** 2) * (a + b) / denom**2 * dk_dt[i]
            dk_f_dj = (b / k[j] ** 2) * (a + b) / denom**2 * dk_dt[j]
            dq_di = k_f / d - (delta_t / d) * dk_f_di
            dq_dj = -k_f / d - (delta_t / d) * dk_f_dj
            # Residual of cell i carries +q_f; residual of cell j carries -q_f.
            jac[i, i] += dq_di
            jac[i, j] += dq_dj
            jac[j, i] -= dq_di
            jac[j, j] -= dq_dj

        # Heated face: q_0 = k_0 (T_w - T_0)/(w_0/2), entering cell 0 as -q_0
        # and the surface energy balance as -q_0.
        dq0_dt0 = -k[0] / half0 + ((t_wall - t_cell[0]) / half0) * dk_dt[0]
        dq0_dtw = k[0] / half0
        jac[0, 0] -= dq0_dt0
        jac[0, n] -= dq0_dtw

        # Backface.
        if ctx.backface.kind is not BackfaceKind.ADIABATIC:
            dqn_dtl = k[-1] / halfn + ((t_cell[-1] - t_back) / halfn) * dk_dt[-1]
            jac[n - 1, n - 1] += dqn_dtl
            jac[n - 1, n + 1] += -k[-1] / halfn

        # Grid-motion advection and pyrolysis-gas convection. Both are face
        # differences, so both pick up the interpolation weights of the two
        # neighbouring cells; these off-diagonals are comparable to the
        # conduction ones whenever the gas flux is appreciable.
        d_tface = self._face_weight_matrix(w_right, None)
        d_hface = self._face_weight_matrix(w_right, self._gas_slope)
        d_gas = self._gas_flux_sensitivity(domega_dt, widths)
        jac[:n, :] -= (advect * rho * cp)[:, None] * (d_tface[1:] - d_tface[:-1])
        # rho and cp in that coefficient are themselves functions of T, and
        # with an active pyrolysis front drho/dT is not small.
        jac[idx, idx] -= (
            advect * (drho_dt * cp + rho * dcp_dt) * (t_face[1:] - t_face[:-1])
        )
        jac[:n, :] -= gas_flux[:, None] * (d_hface[1:] - d_hface[:-1])
        jac[:n, :] -= (h_face[1:] - h_face[:-1])[:, None] * d_gas

        # Surface energy balance row.
        #
        # Eq. (10) closes through the B' tables, so T_w and m_g reach the
        # residual by way of B'_c(P, B'_g, T_w), h_w(P, B'_g, T_w) and the
        # blowing reduction phi(B'_g + B'_c) — three interpolants and a
        # logarithm, composed. There is no analytic derivative to write
        # down: the tables *are* the model. Central differences on the
        # closure itself are therefore not an approximation of something
        # better, they are the derivative, and four extra table lookups
        # cost nothing beside the Newton solve.
        env = ctx.environment

        def closure(t_w: float, m_g: float) -> float:
            """Eq. (10) with the conduction term removed."""
            value, _ = self._surface(t_w, m_g, 0.0, ctx)
            return value

        def probe(
            f: float,
            lo: float,
            hi: float,
            step: float,
            args: Callable[[float], tuple[float, float]],
        ) -> float:
            """Central difference where the table allows it, one-sided where
            it does not.

            A converged state can sit exactly on an axis edge — a gas flux
            that saturates the B'_g axis is the ordinary case, not a
            pathology — and a symmetric probe around it then steps outside
            a table that correctly refuses to extrapolate. Falling back to
            the interior side costs one order of accuracy in a Jacobian
            entry and nothing in the converged answer.
            """
            for a, b in ((f + step, f - step), (f, f - step), (f + step, f)):
                a_c, b_c = min(max(a, lo), hi), min(max(b, lo), hi)
                if a_c == b_c:
                    continue
                try:
                    return (closure(*args(a_c)) - closure(*args(b_c))) / (a_c - b_c)
                except TableRangeError:
                    continue
            return 0.0

        m_g = surface.gas_mass_flux
        c_m = env.mass_coefficient
        t_lo, t_hi = ctx.table.wall_temperature_range
        g_lo, g_hi = ctx.table.gas_rate_range
        m_lo, m_hi = (g_lo * c_m, g_hi * c_m) if c_m > 0.0 else (0.0, np.inf)

        d_seb_dtw = probe(
            t_wall, t_lo, t_hi, 1.0e-3 * max(t_wall, 1.0), lambda v: (v, m_g)
        )
        d_seb_dmg = probe(
            m_g, m_lo, m_hi, 1.0e-6 * max(m_g, c_m, 1.0e-6), lambda v: (t_wall, v)
        )

        jac[n, n] = d_seb_dtw - dq0_dtw
        jac[n, 0] = -dq0_dt0
        jac[n, :n] += d_seb_dmg * (-domega_dt * widths)

        # Backface closure row.
        c_last, c_back = ctx.backface.closure_jacobian(
            float(t_cell[-1]), t_back, float(k[-1]), 0.5 * widths[-1]
        )
        jac[n + 1, n - 1] = c_last
        jac[n + 1, n + 1] = c_back
        return jac

    def _property_derivative(
        self,
        which: str,
        temperature: _FloatArray,
        rho: _FloatArray,
        drho_dt: _FloatArray,
        pressure: float | None = None,
    ) -> _FloatArray:
        """Total :math:`dp/dT` of a blended property.

        Blended properties depend on temperature twice over: directly,
        and through the char fraction, which is itself a function of the
        density the kinetics are driving down. Both paths are carried.
        """
        beta = 1.0 - self.virgin_mass_fraction(rho)
        prop: list[object] = [getattr(m, which) for m in self._materials]
        if which == "conductivity" and self._any_pressure_k:
            for i, override in enumerate(self._pressure_k):
                if override is not None:
                    prop[i] = override
        extra = () if pressure is None else (pressure,)

        def _call(obj: object, name: str, t: float, b: float) -> float:
            method = getattr(obj, name)
            if isinstance(obj, PressureConductivity | TabulatedConductivity):
                return float(method(t, b, *extra))
            return float(method(t, b))

        direct = np.array(
            [
                _call(o, "d_temperature", t, b)
                for o, t, b in zip(prop, temperature, beta, strict=True)
            ]
        )
        d_beta = np.array(
            [
                _call(o, "d_char_fraction", t, b)
                for o, t, b in zip(prop, temperature, beta, strict=True)
            ]
        )
        # beta = 1 - tau, tau = (1 - rho_c/rho)/(1 - rho_c/rho_v).
        safe = np.maximum(rho, 1e-12)
        dbeta_drho = -(self._char / safe**2) / (1.0 - self._char / self._virgin)
        inside = (rho > self._char) & (rho < self._virgin)
        return np.asarray(direct + np.where(inside, d_beta * dbeta_drho * drho_dt, 0.0))

    # -- time stepping -------------------------------------------------------

    def initial_state(self, temperature: float) -> tuple[_FloatArray, _FloatArray]:
        """Uniform virgin material at ``temperature``."""
        if not (np.isfinite(temperature) and temperature > 0.0):
            raise ValueError(f"temperature must be finite and > 0, got {temperature}")
        return (
            np.full(self._n, float(temperature)),
            self._rho_v.copy(),
        )

    def step(
        self,
        temperature: _FloatArray,
        component_density: _FloatArray,
        recession: float,
        dt: float,
        environment: AerothermalEnvironment,
        table: BPrimeTable,
        backface: BackfaceCondition,
        wall_temperature: float | None = None,
        backface_temperature: float | None = None,
        time: float = 0.0,
    ) -> FiatStep:
        """Advance one fully implicit time step.

        The outer loop converges the recession, which sets the grid, against
        the inner Newton solve, which sets :math:`T_w` and hence the
        recession. Without it the scheme would be implicit in temperature
        and explicit in geometry, which is exactly the CMA weakness FIAT
        was written to remove.
        """
        if not (np.isfinite(dt) and dt > 0.0):
            raise ValueError(f"dt must be finite and > 0, got {dt}")
        opt = self._opt
        t_w = float(wall_temperature if wall_temperature is not None else temperature[0])
        t_b = float(
            backface_temperature if backface_temperature is not None else temperature[-1]
        )
        unknowns = np.concatenate([np.asarray(temperature, dtype=np.float64), [t_w, t_b]])

        s_new = float(recession)
        s_dot = 0.0
        newton_total = 0
        outer = 0
        change = 0.0
        for outer in range(1, opt.max_recession_iterations + 1):  # noqa: B007
            grid = self._stack.grid(s_new)
            ctx = _StepContext(
                previous_temperature=np.asarray(temperature, dtype=np.float64),
                previous_components=np.asarray(component_density, dtype=np.float64),
                dt=float(dt),
                recession_rate=s_dot,
                widths=grid.widths,
                centers=grid.centers,
                environment=environment,
                backface=backface,
                table=table,
            )
            unknowns, iterations, info = self._newton(unknowns, ctx)
            newton_total += iterations
            surface: SurfaceState = info["surface"]  # type: ignore[assignment]
            s_dot = surface.recession_rate
            s_next = float(recession) + s_dot * float(dt)
            change = abs(s_next - s_new) / max(abs(s_next), 1e-12)
            s_new = s_next
            if change <= opt.recession_tolerance or not np.isfinite(opt.recession_tolerance):
                break
        else:
            raise RuntimeError(
                f"recession did not converge in {opt.max_recession_iterations} outer "
                f"iterations (last relative change {change:.3e}); reduce dt"
            )

        return FiatStep(
            time=float(time) + float(dt),
            temperature=unknowns[: self._n].copy(),
            component_density=np.asarray(info["component_density"]).copy(),
            wall_temperature=float(unknowns[self._n]),
            backface_temperature=float(unknowns[self._n + 1]),
            recession=s_new,
            recession_rate=s_dot,
            gas_flux=np.asarray(info["gas_flux"]).copy(),
            surface=surface,
            newton_iterations=newton_total,
            recession_iterations=outer,
        )

    def _newton(
        self, unknowns: _FloatArray, ctx: _StepContext
    ) -> tuple[_FloatArray, int, dict[str, object]]:
        opt = self._opt
        u = unknowns.copy()
        residual, info = self._residual(u, ctx)
        scale = max(float(np.max(np.abs(np.asarray(info["storage"])))) * 1.0, 1.0)
        for iteration in range(1, opt.max_newton_iterations + 1):
            norm = float(np.max(np.abs(residual))) / scale
            if norm <= opt.newton_tolerance:
                return u, iteration - 1, info
            jac = self._jacobian(u, ctx, info)
            try:
                delta = scipy.linalg.solve(jac, -residual)
            except scipy.linalg.LinAlgError as exc:  # pragma: no cover - singular
                raise RuntimeError(
                    "the Newton system became singular; this usually means a "
                    "material property or B' table has gone non-physical"
                ) from exc
            # Damped update: a full step can overshoot into negative
            # temperature early in a heat pulse, where the T^4 reradiation
            # term is stiff.
            alpha = 1.0
            accepted = False
            for _ in range(60):
                trial = u + alpha * delta
                if np.all(trial > opt.min_temperature):
                    try:
                        trial_residual, trial_info = self._residual(trial, ctx)
                    except TableRangeError:
                        # A Newton step that lands off the B' table is an
                        # overshoot, not a modelling error: shorten it. Only
                        # a *converged* state outside the table means the
                        # table is too small for the trajectory, and that
                        # surfaces below as a genuine failure.
                        alpha *= 0.5
                        continue
                    if float(np.max(np.abs(trial_residual))) <= float(
                        np.max(np.abs(residual))
                    ) * (1.0 - 1.0e-4 * alpha) or alpha < 1e-8:
                        u, residual, info = trial, trial_residual, trial_info
                        accepted = True
                        break
                alpha *= 0.5
            if not accepted:
                raise RuntimeError(
                    "line search failed to find an admissible step; reduce dt, "
                    "widen the B' table, or check the environment history"
                )
        raise RuntimeError(
            f"Newton did not converge in {opt.max_newton_iterations} iterations "
            f"(scaled residual {norm:.3e} > {opt.newton_tolerance:.3e})"
        )

    def solve(
        self,
        times: NDArray[np.float64],
        environments: list[AerothermalEnvironment],
        table: BPrimeTable,
        backface: BackfaceCondition,
        initial_temperature: float = 300.0,
    ) -> FiatSolution:
        """Integrate a full trajectory.

        ``environments[i]`` applies over the step ending at ``times[i+1]``,
        so ``len(environments) == len(times) - 1``. Piecewise-constant
        environments over a step is what FIAT's tabulated ``envir.inp``
        amounts to; interpolate before calling if a finer history is
        wanted.
        """
        t = np.asarray(times, dtype=np.float64)
        if t.ndim != 1 or t.size < 2 or np.any(np.diff(t) <= 0.0):
            raise ValueError("times must be strictly increasing with >= 2 entries")
        if len(environments) != t.size - 1:
            raise ValueError(
                f"need {t.size - 1} environments for {t.size} times, "
                f"got {len(environments)}"
            )
        temperature, components = self.initial_state(initial_temperature)
        solution = FiatSolution(times=t)
        recession = 0.0
        t_w = float(initial_temperature)
        t_b = float(initial_temperature)
        for i in range(t.size - 1):
            step = self.step(
                temperature,
                components,
                recession,
                float(t[i + 1] - t[i]),
                environments[i],
                table,
                backface,
                wall_temperature=t_w,
                backface_temperature=t_b,
                time=float(t[i]),
            )
            temperature = step.temperature
            components = step.component_density
            recession = step.recession
            t_w, t_b = step.wall_temperature, step.backface_temperature
            solution.steps.append(step)
        return solution
