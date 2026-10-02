r"""The entry field on symbols, and the geometry its derivation needs.

Nothing here restates the equations of motion. The rates are
:func:`aether.certification.rotating_field.rotating_entry_field` evaluated on
symbols, so the expressions the derivations reason about are the ones the
simulator and the enclosures evaluate. What this module adds is the frame
geometry that field was *derived* from -- the position vector, the local
north-east-down basis, the wind triad -- which the field itself has no reason
to carry, and without which "these are Newton's equations in a rotating frame"
could only be asserted.

Notation follows the manuscript's dynamics appendix: geocentric radius
:math:`r`, longitude :math:`\lambda`, latitude :math:`\phi`, planet-relative
speed :math:`V`, flight-path angle :math:`\gamma` (positive climbing), heading
:math:`\psi` (clockwise from north).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import sympy as sp

from aether.certification.rotating_field import (
    Planet,
    altitude,
    j2_gravity,
    jacobi_energy,
    rotating_entry_field,
    surface_radius,
)
from aether.symbolic.ops import SYMPY_OPS

__all__ = ["EntrySymbols"]

_RADIUS = sp.Symbol("r", positive=True)
_LONGITUDE = sp.Symbol("lambda", real=True)
_LATITUDE = sp.Symbol("phi", real=True)
_SPEED = sp.Symbol("V", positive=True)
_GAMMA = sp.Symbol("gamma", real=True)
_HEADING = sp.Symbol("psi", real=True)
_FORCE_V = sp.Symbol("a_V", real=True)
_FORCE_GAMMA = sp.Symbol("a_gamma", real=True)
_FORCE_PSI = sp.Symbol("a_psi", real=True)
_PLANET = Planet(
    mu=sp.Symbol("mu", positive=True),
    radius=sp.Symbol("R_e", positive=True),
    j2=sp.Symbol("J_2", real=True),
    rotation_rate=sp.Symbol("omega_E", real=True),
    flattening=sp.Symbol("f_E", nonnegative=True),
)


@dataclass(frozen=True)
class EntrySymbols:
    """The symbols of the translational entry problem, and what is built on them.

    Positivity is declared where the physics supplies it -- a radius, a speed,
    a gravitational parameter -- because SymPy will not cancel a square root or
    divide through by a quantity it has not been told the sign of, and an
    identity that fails for want of an assumption looks exactly like one that
    is false.
    """

    radius: Any = _RADIUS
    longitude: Any = _LONGITUDE
    latitude: Any = _LATITUDE
    speed: Any = _SPEED
    gamma: Any = _GAMMA
    heading: Any = _HEADING
    force_v: Any = _FORCE_V
    force_gamma: Any = _FORCE_GAMMA
    force_psi: Any = _FORCE_PSI
    planet: Planet = _PLANET

    @property
    def state(self) -> tuple[Any, ...]:
        r""":math:`(r, \lambda, \phi, V, \gamma, \psi)`."""
        return (
            self.radius,
            self.longitude,
            self.latitude,
            self.speed,
            self.gamma,
            self.heading,
        )

    @property
    def specific_force(self) -> tuple[Any, Any, Any]:
        r""":math:`(F_V, F_\gamma, F_\psi)/m`, the aerodynamic force on the wind triad."""
        return (self.force_v, self.force_gamma, self.force_psi)

    # -- the field, evaluated rather than restated ---------------------------

    @cached_property
    def rates(self) -> tuple[Any, ...]:
        """The six rates: the field of record, on symbols."""
        return tuple(
            rotating_entry_field(self.state, self.specific_force, SYMPY_OPS, planet=self.planet)
        )

    @cached_property
    def gravity(self) -> tuple[Any, Any]:
        """:math:`(g_D, g_N)`: downward and northward gravity."""
        return j2_gravity(self.radius, self.latitude, SYMPY_OPS, planet=self.planet)

    @cached_property
    def energy(self) -> Any:
        """The Jacobi integral: specific mechanical energy in the rotating frame."""
        return jacobi_energy(self.state, SYMPY_OPS, planet=self.planet)

    @cached_property
    def surface_radius(self) -> Any:
        r""":math:`R(\phi)`: the reference ellipsoid, at this geocentric latitude."""
        return surface_radius(self.latitude, SYMPY_OPS, planet=self.planet)

    @cached_property
    def altitude(self) -> Any:
        r""":math:`h = r - R(\phi)`: height above the reference ellipsoid, along the radius.

        This is the argument of everything the atmosphere supplies. On a
        planet of zero flattening it is :math:`r - R_e`.
        """
        return altitude(self.radius, self.latitude, SYMPY_OPS, planet=self.planet)

    @cached_property
    def potential(self) -> Any:
        r"""The :math:`J_2` potential :math:`\Phi_g(r, \phi)`, stated independently.

        Written out here rather than recovered from :attr:`gravity`, because the
        claim to be checked is that the one is the gradient of the other.
        """
        planet = self.planet
        ratio = (planet.radius / self.radius) ** 2
        zonal = sp.Rational(1, 2) * planet.j2 * ratio * (3 * sp.sin(self.latitude) ** 2 - 1)
        return -planet.mu / self.radius * (1 - zonal)

    # -- frame geometry, in planet-fixed Cartesian components -----------------

    @cached_property
    def up(self) -> Any:
        r"""Radial unit vector :math:`\hat e_r = -\hat e_D`."""
        lam, phi = self.longitude, self.latitude
        return sp.Matrix([sp.cos(phi) * sp.cos(lam), sp.cos(phi) * sp.sin(lam), sp.sin(phi)])

    @cached_property
    def north(self) -> Any:
        r""":math:`\hat e_N`."""
        lam, phi = self.longitude, self.latitude
        return sp.Matrix([-sp.sin(phi) * sp.cos(lam), -sp.sin(phi) * sp.sin(lam), sp.cos(phi)])

    @cached_property
    def east(self) -> Any:
        r""":math:`\hat e_E`."""
        lam = self.longitude
        return sp.Matrix([-sp.sin(lam), sp.cos(lam), 0])

    @cached_property
    def position(self) -> Any:
        """The position vector in the rotating, planet-fixed frame."""
        return self.radius * self.up

    def from_ned(self, north: Any, east: Any, down: Any) -> Any:
        """A vector given by north-east-down components, in planet-fixed ones."""
        return north * self.north + east * self.east - down * self.up

    @cached_property
    def wind_triad_ned(self) -> tuple[Any, Any, Any]:
        r""":math:`(\hat V, \hat n_\gamma, \hat n_\psi)` in north-east-down components.

        The first two are the manuscript's definitions; the third is their
        cross product, computed rather than typed.
        """
        gam, psi = self.gamma, self.heading
        v_hat = sp.Matrix([sp.cos(gam) * sp.cos(psi), sp.cos(gam) * sp.sin(psi), -sp.sin(gam)])
        n_gamma = sp.Matrix(
            [-sp.sin(gam) * sp.cos(psi), -sp.sin(gam) * sp.sin(psi), -sp.cos(gam)]
        )
        return v_hat, n_gamma, v_hat.cross(n_gamma).applyfunc(sp.trigsimp)

    @cached_property
    def wind_triad(self) -> tuple[Any, Any, Any]:
        """The same triad in planet-fixed Cartesian components."""
        v_hat, n_gamma, n_psi = self.wind_triad_ned
        return self.from_ned(*v_hat), self.from_ned(*n_gamma), self.from_ned(*n_psi)

    @cached_property
    def rotation(self) -> Any:
        """The planet's angular velocity vector."""
        return sp.Matrix([0, 0, self.planet.rotation_rate])
