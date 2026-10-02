r"""The coupled state of the dynamics appendix, and its field, on symbols.

The appendix carries one state vector,

.. math::

    x = \bigl(r,\lambda,\phi,V,\gamma,\psi,\;
              q_0,q_1,q_2,q_3,\,p,q_b,r_b,\;
              Q_{\mathrm{tot}},m,s,\,T_1,\dots,T_{N_T},\,\rho_1,\dots,\rho_{N_T},\;
              \eta_1,\dots,\eta_{N_m},\dot\eta_1,\dots,\dot\eta_{N_m}\bigr),

of dimension :math:`16 + 2N_T + 2N_m`, and an equation for each block. This
module assembles them into one field so that the claims the manuscript makes
about the *coupled* system -- that the quaternion stays on the unit sphere,
that the energy still dissipates at the drag power once the force comes
through the attitude, that the vehicle loses exactly the mass its heat shield
does -- are checked on the coupled system and not on a block of it.

The blocks are of four kinds, and the difference is what a check can mean.

*Translation* is the field of record,
:func:`aether.certification.rotating_field.rotating_entry_field`, evaluated on
symbols and driven by the force that :class:`AttitudeSymbols` resolves on the
wind triad. Nothing about it is restated here.

*Attitude* -- the quaternion kinematics and Euler's equations -- is stated as
the appendix displays it. Its derivations are checks of the statement.

*The heat shield* is two conservation laws in depth, of mass and of energy,
on :math:`N_T` cells that move with the receding surface
(:class:`~aether.symbolic.thermal.CharringCells`), closed at the surface by
an energy balance (:class:`~aether.symbolic.thermal.SurfaceBalance`). What it
conserves is derived. What is constitutive in it -- the conductivity and the
enthalpies of the material, its decomposition rate, the film coefficient of
the boundary layer, the rate at which the surface's chemistry removes char --
is a named function, as the aerodynamic coefficients, the moment, the
generalized force of the undeformed body's pressure and the density are:
in the manuscript each is a selection from an enclosure and not a formula.

*The structural modes* are a constitutive oscillator, forced by the one
aerodynamic term with a derivation behind it: first-order piston pressure,
from :mod:`aether.symbolic.piston`, projected on the mode shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import sympy as sp

from aether.certification.rotating_field import rotating_entry_field
from aether.symbolic.attitude import AttitudeSymbols
from aether.symbolic.ops import SYMPY_OPS
from aether.symbolic.thermal import CharringCells, SurfaceBalance

__all__ = ["CoupledSymbols"]

_MASS = sp.Symbol("m", positive=True)
_HEAT_LOAD = sp.Symbol("Q_tot", nonnegative=True)
_RECESSION = sp.Symbol("s", real=True)
_REFERENCE_AREA = sp.Symbol("S", positive=True)
_MOMENT = sp.symbols("M_1 M_2 M_3", real=True)
_INERTIA = sp.symbols("I_xx I_yy I_zz I_xy I_xz I_yz", real=True)
_INERTIA_RATE = sp.symbols("J_xx J_yy J_zz J_xy J_xz J_yz", real=True)
_HEATING_CONSTANT, _NOSE_RADIUS = sp.symbols("k_Q R_n", positive=True)
_ABLATING_AREA = sp.Symbol("A_s", positive=True)
_FREESTREAM_ENTHALPY, _REFERENCE_WALL_ENTHALPY = sp.symbols("h_inf h_w0", real=True)
_RADIATIVE_FLUX = sp.Symbol("q_rad", nonnegative=True)


def _symmetric(entries: tuple[Any, ...]) -> Any:
    xx, yy, zz, xy, xz, yz = entries
    return sp.Matrix([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


@dataclass(frozen=True)
class CoupledSymbols:
    """The appendix's full state, and the rate of every component of it."""

    attitude: AttitudeSymbols = field(default_factory=AttitudeSymbols)
    modes: int = 2
    """Number of structural modes retained, :math:`N_m`."""
    cells: int = 2
    """Number of cells the heat shield is resolved on in depth, :math:`N_T`."""
    mass: Any = _MASS
    heat_load: Any = _HEAT_LOAD
    recession: Any = _RECESSION
    reference_area: Any = _REFERENCE_AREA
    moment: tuple[Any, Any, Any] = _MOMENT
    """Total external moment in body axes, aerodynamic and reaction-control."""

    # -- named pieces -----------------------------------------------------------

    @property
    def entry(self) -> Any:
        """The translational symbols the attitude was built on."""
        return self.attitude.entry

    @cached_property
    def density(self) -> Any:
        r""":math:`\rho(h)`: any positive function of the altitude :math:`h = r - R(\phi)`."""
        return sp.Function("rho", positive=True)(self.entry.altitude)

    @property
    def pressure_area(self) -> Any:
        r""":math:`q_{\mathrm{dyn}} S = \tfrac12\rho V^2 S`."""
        return self.density * self.entry.speed**2 * self.reference_area / 2

    @cached_property
    def inertia(self) -> Any:
        r"""The inertia tensor :math:`\mathbf I` in body axes: symmetric, otherwise free."""
        return _symmetric(_INERTIA)

    @cached_property
    def inertia_rate(self) -> Any:
        r"""Rate of change of :math:`\mathbf I` at fixed mass: the deformation's share.

        Only this part of :math:`\dot{\mathbf I}` appears in Euler's
        equations. What recession removes from the inertia leaves with the
        angular momentum it had, and contributes no term: see the derivation
        ``euler-equations``.
        """
        return _symmetric(_INERTIA_RATE)

    @cached_property
    def modal_displacements(self) -> tuple[Any, ...]:
        return tuple(sp.Symbol(f"eta_{i}", real=True) for i in range(1, self.modes + 1))

    @cached_property
    def modal_velocities(self) -> tuple[Any, ...]:
        return tuple(sp.Symbol(f"etadot_{i}", real=True) for i in range(1, self.modes + 1))

    @cached_property
    def modal_frequencies(self) -> tuple[Any, ...]:
        r""":math:`\omega_i(T_b)`: each a positive function of the structure's temperature.

        The structure lies behind the heat shield, so the temperature that
        softens it is the back face's and not the ablating wall's.
        """
        return tuple(
            sp.Function(f"omega_{i}", positive=True)(self.structure_temperature)
            for i in range(1, self.modes + 1)
        )

    @cached_property
    def modal_damping(self) -> tuple[Any, ...]:
        return tuple(sp.Symbol(f"zeta_{i}", positive=True) for i in range(1, self.modes + 1))

    @cached_property
    def sound_speed(self) -> Any:
        r""":math:`a_\infty(h)`: any positive function of altitude."""
        return sp.Function("a_inf", positive=True)(self.entry.altitude)

    @cached_property
    def rigid_forcing(self) -> tuple[Any, ...]:
        r""":math:`Q_i^{(0)}`: the generalized force of the pressure on the undeformed
        body, left free.

        It comes from the same pressure distribution as the aerodynamic
        coefficients, and is enclosed with them.
        """
        return tuple(sp.Symbol(f"Q0_{i}", real=True) for i in range(1, self.modes + 1))

    @cached_property
    def damping_integrals(self) -> Any:
        r""":math:`c_{ij} = \int b\,W_i W_j\,d\xi`: a Gram matrix, so symmetric."""
        return sp.Matrix(
            self.modes,
            self.modes,
            lambda i, j: sp.Symbol(f"c_{min(i, j) + 1}_{max(i, j) + 1}", real=True),
        )

    @cached_property
    def stiffness_integrals(self) -> Any:
        r""":math:`g_{ij} = \int b\,W_i W_j'\,d\xi`: not symmetric."""
        return sp.Matrix(
            self.modes, self.modes, lambda i, j: sp.Symbol(f"g_{i + 1}_{j + 1}", real=True)
        )

    @cached_property
    def modal_forcing(self) -> tuple[Any, ...]:
        r"""Generalized forces, with first-order piston pressure on the deformation:

        .. math:: Q_i = Q_i^{(0)} - 2\rho\,a_\infty\sum_j
                  \bigl(c_{ij}\,\dot\eta_j + V\,g_{ij}\,\eta_j\bigr).

        Derived in :mod:`aether.symbolic.piston`; the integrals are those of
        :class:`~aether.symbolic.piston.StripSymbols` under a uniform stream.
        """
        impedance = 2 * self.density * self.sound_speed
        velocities = sp.Matrix(self.modal_velocities)
        displacements = sp.Matrix(self.modal_displacements)
        aerodynamic = impedance * (
            self.damping_integrals * velocities
            + self.entry.speed * self.stiffness_integrals * displacements
        )
        return tuple(
            rigid - elastic for rigid, elastic in zip(self.rigid_forcing, aerodynamic, strict=True)
        )

    # -- the state ----------------------------------------------------------------

    @cached_property
    def state(self) -> tuple[Any, ...]:
        """All :math:`16 + 2N_T + 2N_m` components, in the appendix's order."""
        shield = self.heat_shield
        return (
            *self.entry.state,
            *self.attitude.quaternion,
            *self.attitude.body_rate,
            self.heat_load,
            self.mass,
            self.recession,
            *shield.temperatures,
            *shield.densities,
            *self.modal_displacements,
            *self.modal_velocities,
        )

    # -- the blocks of the field ----------------------------------------------------

    @cached_property
    def wind_force(self) -> tuple[Any, Any, Any]:
        r""":math:`(F_V, F_\gamma, F_\psi)` from the quaternion, through the body-to-local
        rotation."""
        bound = AttitudeSymbols(
            entry=self.attitude.entry,
            quaternion=self.attitude.quaternion,
            body_rate=self.attitude.body_rate,
            coefficients=self.attitude.coefficients,
            pressure_area=self.pressure_area,
        )
        return bound.wind_force(bound.body_to_local)

    @cached_property
    def translational_rates(self) -> tuple[Any, ...]:
        """The field of record, driven by the force the attitude supplies."""
        specific = tuple(component / self.mass for component in self.wind_force)
        return tuple(
            rotating_entry_field(self.entry.state, specific, SYMPY_OPS, planet=self.entry.planet)
        )

    @cached_property
    def quaternion_rates(self) -> tuple[Any, ...]:
        r""":math:`\dot{\mathbf q} =
        \tfrac12\boldsymbol\Omega(\boldsymbol\omega_{\mathcal B\mathcal E})\,\mathbf q`."""
        return tuple(self.attitude.quaternion_rate(self.attitude.relative_rate))

    @cached_property
    def body_rate_rates(self) -> tuple[Any, ...]:
        r"""Euler's equations solved for :math:`\dot{\boldsymbol\omega}`."""
        omega = sp.Matrix(self.attitude.body_rate)
        torque = (
            sp.Matrix(self.moment)
            - omega.cross(self.inertia * omega)
            - self.inertia_rate * omega
        )
        return tuple(self.inertia.adjugate() * torque / self.inertia.det())

    # -- heating, and the heat shield --------------------------------------------------

    @property
    def heat_flux(self) -> Any:
        r"""Stagnation-point flux to a cold wall, :math:`\dot q_s = k_Q\sqrt{\rho/R_n}\,V^3`.

        A correlation, with the envelope the appendix states. It supplies
        the corridor's heating limit and the heat load, and through
        :attr:`film_coefficient` it is what the heat shield's surface is
        heated by.
        """
        return _HEATING_CONSTANT * sp.sqrt(self.density / _NOSE_RADIUS) * self.entry.speed**3

    @property
    def recovery_enthalpy(self) -> Any:
        r""":math:`h_r = h_\infty + \tfrac12 V^2`: the stream's total enthalpy."""
        return _FREESTREAM_ENTHALPY + self.entry.speed**2 / 2

    @property
    def film_coefficient(self) -> Any:
        r""":math:`g_h = \dot q_s/(h_r - h_{w,0})`.

        The correlation is a flux to a wall of enthalpy :math:`h_{w,0}`; the
        coefficient is what makes the same boundary layer deliver
        :math:`g_h\,(h_r - h_w)` to a wall at any other.
        """
        return self.heat_flux / (self.recovery_enthalpy - _REFERENCE_WALL_ENTHALPY)

    @cached_property
    def _bare_shield(self) -> CharringCells:
        """The cells with their closures unbound: where the state symbols come from."""
        return CharringCells(cells=self.cells, recession=self.recession)

    @cached_property
    def char_removal(self) -> Any:
        r""":math:`\dot m_c = g_h\,B'_c(T_w, B'_g)`: what the surface's chemistry removes.

        The dimensionless rate :math:`B'_c` is a function of the wall
        temperature and of the gas blown, :math:`B'_g = \dot m_g/g_h`,
        supplied by the equilibrium of the char with the boundary-layer gas.
        """
        bare = self._bare_shield
        film = self.film_coefficient
        return film * sp.Function("B_c", nonnegative=True)(
            bare.temperatures[0], bare.gas_blown / film
        )

    @cached_property
    def surface_balance(self) -> SurfaceBalance:
        """The energy balance of the ablating surface, at this state."""
        bare = self._bare_shield
        wall = bare.temperatures[0]
        return SurfaceBalance(
            film_coefficient=self.film_coefficient,
            recovery_enthalpy=self.recovery_enthalpy,
            wall_enthalpy=sp.Function("h_w")(wall),
            radiative_flux=_RADIATIVE_FLUX,
            wall_temperature=wall,
            char_flux=self.char_removal,
            gas_flux=bare.gas_blown,
            char_enthalpy=bare.enthalpy_densities[0] / bare.densities[0],
            gas_enthalpy=bare.slab.gas_enthalpy(wall),
        )

    @cached_property
    def heat_shield(self) -> CharringCells:
        """The heat shield in depth, heated and eroded as this state's surface balance says."""
        return CharringCells(
            cells=self.cells,
            recession=self.recession,
            char_removal=self.char_removal,
            surface_conduction=self.surface_balance.conduction,
        )

    @property
    def wall_temperature(self) -> Any:
        r""":math:`T_w = T_1`: the temperature of the cell at the ablating surface."""
        return self._bare_shield.temperatures[0]

    @property
    def structure_temperature(self) -> Any:
        r""":math:`T_b = T_{N_T}`: the temperature of the cell at the back face."""
        return self._bare_shield.temperatures[-1]

    @property
    def ablating_area(self) -> Any:
        r""":math:`A_s`: the area of heat shield this one station stands for."""
        return _ABLATING_AREA

    @property
    def mass_rate(self) -> Any:
        r""":math:`\dot m = -A_s\,(\dot m_c + \dot m_g)`: the vehicle loses what its heat
        shield loses."""
        return -self.ablating_area * (self.char_removal + self.heat_shield.gas_blown)

    @cached_property
    def thermal_rates(self) -> tuple[Any, ...]:
        r"""Rates of :math:`(Q_{\mathrm{tot}}, m, s, T_1,\dots, \rho_1,\dots)`."""
        shield = self.heat_shield
        return (
            self.heat_flux,
            self.mass_rate,
            shield.recession_rate,
            *shield.temperature_rates,
            *shield.density_rates,
        )

    @cached_property
    def modal_rates(self) -> tuple[Any, ...]:
        r"""Each mode as the pair
        :math:`(\dot\eta_i,\ Q_i - 2\zeta_i\omega_i\dot\eta_i - \omega_i^2\eta_i)`."""
        accelerations = tuple(
            forcing - 2 * damping * frequency * velocity - frequency**2 * displacement
            for forcing, damping, frequency, velocity, displacement in zip(
                self.modal_forcing,
                self.modal_damping,
                self.modal_frequencies,
                self.modal_velocities,
                self.modal_displacements,
                strict=True,
            )
        )
        return (*self.modal_velocities, *accelerations)

    @cached_property
    def rates(self) -> tuple[Any, ...]:
        """The rate of every component of :attr:`state`, in the same order."""
        return (
            *self.translational_rates,
            *self.quaternion_rates,
            *self.body_rate_rates,
            *self.thermal_rates,
            *self.modal_rates,
        )
