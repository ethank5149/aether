r"""The heat shield in depth, on symbols: two conservation laws, a moving surface, and cells.

The dynamics appendix once carried the wall as a lumped temperature and the
recession as a multiple of the heat load. Neither was a statement that
anything is conserved. This module is the statement, in three parts, each
checked by :mod:`aether.symbolic.derivations`.

*The slab.* Along the inward normal :math:`y`, a decomposing solid of bulk
density :math:`\rho` through which pyrolysis gas flows with mass flux
:math:`j` satisfies conservation of mass and of energy,

.. math::

    \partial_t\rho + \partial_y j = 0, \qquad
    \partial_t(\rho h) + \partial_y\bigl(j\,h_g - k\,\partial_y T\bigr) = 0 ,

with the solid's enthalpy that of a mixture of virgin material and char
(:class:`CharringSlab`). Two things are assumed and are said here once: the
gas is at the solid's temperature, and it leaves as it is made, so that no
gas is stored. Everything else in the familiar temperature equation of a
charring ablator -- the decomposition enthalpy
:math:`\bar h = (\rho_v h_v - \rho_c h_c)/(\rho_v - \rho_c)`, the sign of
the gas's convection -- is a consequence, and the derivation ``ablation-energy``
is that consequence drawn.

*The surface.* It recedes at :math:`\dot s` into a domain of fixed back
face, :math:`s(t)\le y\le L`. Leibniz's rule then gives what the slab as a
whole loses: its mass falls at the char removed plus the gas blown, and its
enthalpy changes by the heat conducted in less what those two carry out.
The surface energy balance closes the second, and :class:`SurfaceBalance`
states it. The coordinate that holds the surface still,
:math:`\eta = (y - s)/(L - s)`, is Landau's.

*The cells.* :class:`CharringCells` is the same pair of laws on
:math:`N` cells of the Landau coordinate, written so that what leaves one
cell enters the next: a finite-volume semi-discretization. Its states are a
temperature and a density per cell and the recession, and its total mass
and total enthalpy change at exactly the rates the continuum gives, for
any number of cells. One cell, with no decomposition and no recession, is
the lumped wall the appendix used to state; the difference between the two
is now written down instead of left out.

What is constitutive, and stays a named function: the conductivity, the
enthalpies of virgin material, char and gas, the decomposition rate, the
film coefficient and the recovery enthalpy of the boundary layer, the wall
enthalpy of the gas there, and the char's rate of removal by the chemistry
of the surface. Each has a source and a range in the appendix. None is a
conservation law, and none is asked to be.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import sympy as sp

__all__ = ["CharringCells", "CharringSlab", "SurfaceBalance", "landau_coordinate"]

_DEPTH, _TIME = sp.symbols("y t", real=True)
_THICKNESS = sp.Symbol("L", positive=True)
_VIRGIN_DENSITY, _CHAR_DENSITY = sp.symbols("rho_v rho_c", positive=True)
_EMISSIVITY, _ABSORPTIVITY = sp.symbols("epsilon_w alpha_w", positive=True)
_STEFAN_BOLTZMANN = sp.Symbol("sigma_SB", positive=True)
_BLOWING_SHAPE = sp.Symbol("lambda_b", positive=True)
_FILM_COEFFICIENT = sp.Symbol("g_h", positive=True)
_RECOVERY_ENTHALPY, _WALL_ENTHALPY = sp.symbols("h_r h_w", real=True)
_RADIATIVE_FLUX = sp.Symbol("q_rad", nonnegative=True)
_WALL_TEMPERATURE = sp.Symbol("T_w", positive=True)
_CHAR_FLUX, _GAS_FLUX = sp.symbols("mdot_c mdot_g", nonnegative=True)
_CHAR_ENTHALPY, _GAS_ENTHALPY = sp.symbols("h_s h_gs", real=True)
_RECESSION = sp.Symbol("s", real=True)
_SURFACE_CONDUCTION = sp.Symbol("q_c", real=True)


def landau_coordinate(depth: Any, recession: Any, thickness: Any) -> Any:
    r""":math:`\eta = (y - s)/(L - s)`: zero on the receding surface, one on the back face."""
    return (depth - recession) / (thickness - recession)


@dataclass(frozen=True)
class CharringSlab:
    r"""A decomposing solid in depth, as fields of :math:`(y, t)` and what they conserve.

    The temperature, the bulk density and the gas flux are free functions.
    The gas flux :math:`j` is along :math:`+y`, into the body, so the gas
    that flows to the surface has :math:`j < 0` and the mass leaving there
    per unit area is :math:`\dot m_g = -j(s, t)`.
    """

    depth: Any = _DEPTH
    time: Any = _TIME
    thickness: Any = _THICKNESS
    virgin_density: Any = _VIRGIN_DENSITY
    char_density: Any = _CHAR_DENSITY

    # -- the fields -----------------------------------------------------------------

    @cached_property
    def temperature(self) -> Any:
        return sp.Function("T", positive=True)(self.depth, self.time)

    @cached_property
    def density(self) -> Any:
        return sp.Function("rho", positive=True)(self.depth, self.time)

    @cached_property
    def gas_flux(self) -> Any:
        return sp.Function("j", real=True)(self.depth, self.time)

    @cached_property
    def surface(self) -> Any:
        r""":math:`s(t)`: how far the surface has receded."""
        return sp.Function("s", real=True)(self.time)

    # -- the material -----------------------------------------------------------------

    @staticmethod
    def conductivity(temperature: Any, density: Any) -> Any:
        return sp.Function("k", positive=True)(temperature, density)

    @staticmethod
    def gas_enthalpy(temperature: Any) -> Any:
        return sp.Function("h_g")(temperature)

    def virgin_fraction(self, density: Any) -> Any:
        r""":math:`\epsilon_v = (\rho - \rho_c)/(\rho_v - \rho_c)`: the volume still virgin."""
        return (density - self.char_density) / (self.virgin_density - self.char_density)

    def enthalpy_density(self, temperature: Any, density: Any) -> Any:
        r""":math:`\rho h = \epsilon_v\rho_v h_v(T) + (1 - \epsilon_v)\rho_c h_c(T)`.

        The solid is a mixture of virgin material and char at one
        temperature, each with its own enthalpy.
        """
        virgin = self.virgin_fraction(density)
        return (
            virgin * self.virgin_density * sp.Function("h_v")(temperature)
            + (1 - virgin) * self.char_density * sp.Function("h_c")(temperature)
        )

    def heat_capacity(self, temperature: Any, density: Any) -> Any:
        r""":math:`\rho c_p = \partial(\rho h)/\partial T` at fixed density."""
        probe = sp.Symbol("T_", positive=True)
        return sp.diff(self.enthalpy_density(probe, density), probe).subs(probe, temperature)

    def decomposition_enthalpy(self, temperature: Any, density: Any) -> Any:
        r""":math:`\bar h = \partial(\rho h)/\partial\rho` at fixed temperature.

        It comes out as :math:`(\rho_v h_v - \rho_c h_c)/(\rho_v - \rho_c)`:
        the quantity the charring-ablator literature defines, here the
        derivative it is.
        """
        probe = sp.Symbol("rho_", positive=True)
        return sp.diff(self.enthalpy_density(temperature, probe), probe).subs(probe, density)

    # -- the conservation laws, as residuals ---------------------------------------------

    @property
    def mass_residual(self) -> Any:
        r""":math:`\partial_t\rho + \partial_y j`."""
        return sp.diff(self.density, self.time) + sp.diff(self.gas_flux, self.depth)

    @property
    def energy_flux(self) -> Any:
        r""":math:`j\,h_g(T) - k\,\partial_y T`: enthalpy carried by the gas, and conduction."""
        temperature = self.temperature
        return self.gas_flux * self.gas_enthalpy(temperature) - self.conductivity(
            temperature, self.density
        ) * sp.diff(temperature, self.depth)

    @property
    def energy_residual(self) -> Any:
        r""":math:`\partial_t(\rho h) + \partial_y(j\,h_g - k\,\partial_y T)`."""
        stored = self.enthalpy_density(self.temperature, self.density)
        return sp.diff(stored, self.time) + sp.diff(self.energy_flux, self.depth)

    @property
    def temperature_residual(self) -> Any:
        r"""The energy law as an equation for the temperature:

        .. math::

            \rho c_p\,\partial_t T - \partial_y(k\,\partial_y T)
            - (h_g - \bar h)\,\partial_t\rho + j\,c_{p,g}\,\partial_y T .

        With :math:`\dot m_g = -j` this is the equation a charring-ablator
        code integrates: decomposition absorbs :math:`h_g - \bar h` per unit
        mass, and gas flowing toward a hotter surface cools the solid it
        passes through.
        """
        temperature, density = self.temperature, self.density
        probe = sp.Symbol("T_", positive=True)
        gas_capacity = sp.diff(self.gas_enthalpy(probe), probe).subs(probe, temperature)
        return (
            self.heat_capacity(temperature, density) * sp.diff(temperature, self.time)
            - sp.diff(
                self.conductivity(temperature, density) * sp.diff(temperature, self.depth),
                self.depth,
            )
            - (
                self.gas_enthalpy(temperature)
                - self.decomposition_enthalpy(temperature, density)
            )
            * sp.diff(density, self.time)
            + self.gas_flux * gas_capacity * sp.diff(temperature, self.depth)
        )

    # -- the slab as a whole ----------------------------------------------------------------

    def content(self, quantity: Any) -> Any:
        r""":math:`\int_{s(t)}^{L} u\,dy`: what the slab holds of a density :math:`u`."""
        return sp.Integral(quantity, (self.depth, self.surface, self.thickness))

    def balance_residual(self, quantity: Any, flux: Any) -> Any:
        r"""Leibniz's rule for a conservation law on the receding slab, as a residual.

        For any density :math:`u` and flux :math:`\Phi`,

        .. math::

            \frac{d}{dt}\int_{s}^{L} u\,dy
            = -u(s)\,\dot s - \bigl[\Phi\bigr]_{s}^{L}
              + \int_{s}^{L}\bigl(\partial_t u + \partial_y\Phi\bigr)\,dy ,

        and the last integral vanishes when the law holds.
        """
        surface, back = self.surface, self.thickness
        law = sp.diff(quantity, self.time) + sp.diff(flux, self.depth)
        stated = (
            -quantity.subs(self.depth, surface) * sp.diff(surface, self.time)
            - (flux.subs(self.depth, back) - flux.subs(self.depth, surface))
            + self.content(law)
        )
        return (sp.diff(self.content(quantity), self.time) - stated).doit()


@dataclass(frozen=True)
class SurfaceBalance:
    r"""What crosses the ablating surface, per unit area, and the balance that closes it.

    The boundary layer delivers heat through a film coefficient
    :math:`g_h = \rho_e u_e C_H`, that of the layer with nothing blown into
    it, and a recovery enthalpy :math:`h_r`; the surface loses char at
    :math:`\dot m_c` and blows pyrolysis gas at :math:`\dot m_g`, both of
    which leave at the wall enthalpy :math:`h_w` of the boundary-layer gas;
    and it radiates. Blowing mass into the boundary layer thickens it and
    reduces the transfer by

    .. math:: \Omega = \frac{\Phi}{e^{\Phi} - 1},
              \qquad \Phi = \frac{2\lambda_b(\dot m_c + \dot m_g)}{g_h} ,

    which in the coefficient of the *blown* layer, :math:`g_h\Omega`, is the
    familiar :math:`\ln(1 + 2\lambda_b B')/(2\lambda_b B')`.
    """

    film_coefficient: Any = _FILM_COEFFICIENT
    recovery_enthalpy: Any = _RECOVERY_ENTHALPY
    wall_enthalpy: Any = _WALL_ENTHALPY
    radiative_flux: Any = _RADIATIVE_FLUX
    wall_temperature: Any = _WALL_TEMPERATURE
    char_flux: Any = _CHAR_FLUX
    gas_flux: Any = _GAS_FLUX
    char_enthalpy: Any = _CHAR_ENTHALPY
    """Enthalpy of the solid at the surface, per unit mass: what the char brings there."""
    gas_enthalpy: Any = _GAS_ENTHALPY
    """Enthalpy of the pyrolysis gas at the surface."""
    emissivity: Any = _EMISSIVITY
    absorptivity: Any = _ABSORPTIVITY

    @property
    def blowing_parameter(self) -> Any:
        r""":math:`\Phi = 2\lambda_b(\dot m_c + \dot m_g)/g_h`."""
        return 2 * _BLOWING_SHAPE * (self.char_flux + self.gas_flux) / self.film_coefficient

    @property
    def blowing_correction(self) -> Any:
        r""":math:`\Omega = \Phi/(e^{\Phi} - 1)`, which tends to one as the blowing
        vanishes."""
        return self.blowing_parameter / (sp.exp(self.blowing_parameter) - 1)

    @property
    def convective_flux(self) -> Any:
        r""":math:`g_h\,\Omega\,(h_r - h_w)`."""
        return (
            self.film_coefficient
            * self.blowing_correction
            * (self.recovery_enthalpy - self.wall_enthalpy)
        )

    @property
    def net_heating(self) -> Any:
        r"""Convection, absorbed radiation, less re-radiation:
        :math:`g_h\Omega(h_r - h_w) + \alpha_w q_{\mathrm{rad}} -
        \epsilon_w\sigma T_w^4`."""
        return (
            self.convective_flux
            + self.absorptivity * self.radiative_flux
            - self.emissivity * _STEFAN_BOLTZMANN * self.wall_temperature**4
        )

    @property
    def conduction(self) -> Any:
        r"""Heat conducted into the solid, :math:`-k\,\partial_y T` at the surface.

        The surface stores nothing, so what arrives is what leaves: the net
        heating, plus the enthalpy the char and the gas bring to the surface
        from inside, less the enthalpy they carry away at wall conditions.
        """
        arriving = self.char_flux * self.char_enthalpy + self.gas_flux * self.gas_enthalpy
        leaving = (self.char_flux + self.gas_flux) * self.wall_enthalpy
        return self.net_heating + arriving - leaving


@dataclass(frozen=True)
class CharringCells:
    r"""The slab on :math:`N` cells of the Landau coordinate: a conservative scheme.

    Faces :math:`0 = \eta_0 < \eta_1 < \dots < \eta_N = 1`; cell :math:`i`
    lies between faces :math:`i-1` and :math:`i` and holds a temperature
    :math:`T_i` and a density :math:`\rho_i`. The faces move with the
    coordinate, at :math:`v_i = \dot s\,(1 - \eta_i)`, so material crosses
    them toward the surface and each face takes its values from the cell
    behind it. The surface is the first cell: its temperature is the wall
    temperature, which is first-order in the cell's thickness and is why the
    cells are meant to be thin there.

    Attributes
    ----------
    cells:
        :math:`N`.
    faces:
        The :math:`N + 1` face coordinates; uniform when left unset.
    """

    cells: int = 2
    slab: CharringSlab = field(default_factory=CharringSlab)
    faces: tuple[Any, ...] | None = None
    recession: Any = _RECESSION
    char_removal: Any = _CHAR_FLUX
    r"""Char removed at the surface, :math:`\dot m_c` (kg m⁻² s⁻¹): supplied by the
    chemistry of the surface, and what the recession rate is defined by."""
    surface_conduction: Any = _SURFACE_CONDUCTION
    """Heat conducted into the first cell from the surface (W/m²)."""

    def __post_init__(self) -> None:
        if self.cells < 1:
            raise ValueError(f"at least one cell is needed, got {self.cells}")
        if self.faces is not None and len(self.faces) != self.cells + 1:
            raise ValueError(f"{self.cells} cells have {self.cells + 1} faces")

    # -- geometry ---------------------------------------------------------------------

    @cached_property
    def face_coordinates(self) -> tuple[Any, ...]:
        if self.faces is not None:
            return tuple(self.faces)
        return tuple(sp.Rational(i, self.cells) for i in range(self.cells + 1))

    @cached_property
    def widths(self) -> tuple[Any, ...]:
        eta = self.face_coordinates
        return tuple(eta[i + 1] - eta[i] for i in range(self.cells))

    @property
    def remaining(self) -> Any:
        r""":math:`\ell = L - s`: the thickness left."""
        return self.slab.thickness - self.recession

    # -- the state --------------------------------------------------------------------

    @cached_property
    def temperatures(self) -> tuple[Any, ...]:
        return tuple(sp.Symbol(f"T_{i}", positive=True) for i in range(1, self.cells + 1))

    @cached_property
    def densities(self) -> tuple[Any, ...]:
        return tuple(sp.Symbol(f"rho_{i}", positive=True) for i in range(1, self.cells + 1))

    @property
    def state(self) -> tuple[Any, ...]:
        """Temperatures, densities, recession: :math:`2N + 1` components."""
        return (*self.temperatures, *self.densities, self.recession)

    # -- closures, at the cells -----------------------------------------------------------

    @staticmethod
    def decomposition_rate(density: Any, temperature: Any) -> Any:
        r""":math:`R(\rho, T) \ge 0`: the rate at which a material point loses density."""
        return sp.Function("R", nonnegative=True)(density, temperature)

    @cached_property
    def recession_rate(self) -> Any:
        r""":math:`\dot s = \dot m_c/\rho_1`: char leaves at the surface cell's density."""
        return self.char_removal / self.densities[0]

    @cached_property
    def face_velocities(self) -> tuple[Any, ...]:
        r""":math:`v_i = \dot s\,(1 - \eta_i)`: zero at the back face, :math:`\dot s` at the
        surface."""
        return tuple(self.recession_rate * (1 - eta) for eta in self.face_coordinates)

    @cached_property
    def gas_fluxes(self) -> tuple[Any, ...]:
        r"""Gas flux at each face, along :math:`+y`: :math:`J_N = 0` and
        :math:`J_{i-1} = J_i - \ell\,\Delta_i R_i`.

        All of it is toward the surface, and :math:`-J_0` is the gas blown.
        """
        fluxes: list[Any] = [sp.S.Zero]
        for index in reversed(range(self.cells)):
            produced = self.remaining * self.widths[index] * self.decomposition_rate(
                self.densities[index], self.temperatures[index]
            )
            fluxes.append(fluxes[-1] - produced)
        return tuple(reversed(fluxes))

    @property
    def gas_blown(self) -> Any:
        r""":math:`\dot m_g = -J_0 = \ell\sum_i\Delta_i R_i`."""
        return -self.gas_fluxes[0]

    def _behind(self, values: tuple[Any, ...], face: int) -> Any:
        """The value a face takes: that of the cell on its inward side, or the last cell's."""
        return values[min(face, self.cells - 1)]

    # -- the rates ----------------------------------------------------------------------

    @cached_property
    def mass_rates(self) -> tuple[Any, ...]:
        r""":math:`\frac{d}{dt}(\ell\Delta_i\rho_i)`: decomposition, and what the moving
        faces sweep in and out."""
        speeds = self.face_velocities
        rates = []
        for index in range(self.cells):
            lost = self.remaining * self.widths[index] * self.decomposition_rate(
                self.densities[index], self.temperatures[index]
            )
            swept = speeds[index + 1] * self._behind(self.densities, index + 1) - speeds[
                index
            ] * self._behind(self.densities, index)
            rates.append(-lost + swept)
        return tuple(rates)

    @cached_property
    def enthalpy_densities(self) -> tuple[Any, ...]:
        return tuple(
            self.slab.enthalpy_density(temperature, density)
            for temperature, density in zip(self.temperatures, self.densities, strict=True)
        )

    @cached_property
    def energy_fluxes(self) -> tuple[Any, ...]:
        r""":math:`\Phi_i = J_i\,h_g - k\,\partial_y T` at each face.

        At the surface, the gas leaves at the first cell's temperature and
        the conduction is the heat supplied; between cells the gas carries
        the enthalpy of the cell it comes from and the conduction is a
        difference over the distance between centres; at the back face
        nothing crosses.
        """
        eta = self.face_coordinates
        centres = [(eta[i] + eta[i + 1]) / 2 for i in range(self.cells)]
        fluxes = [
            self.gas_fluxes[0] * self.slab.gas_enthalpy(self.temperatures[0])
            + self.surface_conduction
        ]
        for face in range(1, self.cells):
            inner, outer = self.temperatures[face], self.temperatures[face - 1]
            mean_temperature = (inner + outer) / 2
            mean_density = (self.densities[face] + self.densities[face - 1]) / 2
            gradient = (inner - outer) / (self.remaining * (centres[face] - centres[face - 1]))
            fluxes.append(
                self.gas_fluxes[face] * self.slab.gas_enthalpy(inner)
                - self.slab.conductivity(mean_temperature, mean_density) * gradient
            )
        fluxes.append(sp.S.Zero)
        return tuple(fluxes)

    @cached_property
    def enthalpy_rates(self) -> tuple[Any, ...]:
        r""":math:`\frac{d}{dt}(\ell\Delta_i\,\rho h_i)`: flux in less flux out, and what
        the moving faces sweep."""
        speeds, fluxes, stored = self.face_velocities, self.energy_fluxes, self.enthalpy_densities
        return tuple(
            -(fluxes[index + 1] - fluxes[index])
            + speeds[index + 1] * self._behind(stored, index + 1)
            - speeds[index] * self._behind(stored, index)
            for index in range(self.cells)
        )

    @cached_property
    def density_rates(self) -> tuple[Any, ...]:
        r""":math:`\dot\rho_i`, from the cell's mass and its shrinking thickness."""
        return tuple(
            (rate + self.recession_rate * width * density) / (self.remaining * width)
            for rate, width, density in zip(self.mass_rates, self.widths, self.densities,
                                            strict=True)
        )

    @cached_property
    def temperature_rates(self) -> tuple[Any, ...]:
        r""":math:`\dot T_i = \bigl(\dot{(\rho h)}_i - \bar h_i\,\dot\rho_i\bigr)/(\rho c_p)_i`."""
        rates = []
        for index in range(self.cells):
            temperature, density = self.temperatures[index], self.densities[index]
            width = self.widths[index]
            stored_rate = (
                self.enthalpy_rates[index]
                + self.recession_rate * width * self.enthalpy_densities[index]
            ) / (self.remaining * width)
            rates.append(
                (
                    stored_rate
                    - self.slab.decomposition_enthalpy(temperature, density)
                    * self.density_rates[index]
                )
                / self.slab.heat_capacity(temperature, density)
            )
        return tuple(rates)

    @property
    def rates(self) -> tuple[Any, ...]:
        """The rate of every component of :attr:`state`, in the same order."""
        return (*self.temperature_rates, *self.density_rates, self.recession_rate)

    # -- what the whole slab holds ----------------------------------------------------------

    @property
    def areal_mass(self) -> Any:
        r""":math:`\ell\sum_i\Delta_i\rho_i`."""
        return self.remaining * sum(
            (width * density for width, density in zip(self.widths, self.densities, strict=True)),
            sp.S.Zero,
        )

    @property
    def areal_enthalpy(self) -> Any:
        r""":math:`\ell\sum_i\Delta_i\,(\rho h)_i`."""
        return self.remaining * sum(
            (width * stored for width, stored in zip(self.widths, self.enthalpy_densities,
                                                     strict=True)),
            sp.S.Zero,
        )
