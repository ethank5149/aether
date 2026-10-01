r"""The planar glide on symbols: equilibrium, phugoid, and what damps it.

The proposal's skeleton is the equilibrium glide, and the size of everything
left over is set by how a trajectory moves *about* it. That motion is one
oscillator -- altitude against flight-path angle, the phugoid -- and three
facts about it carry the argument:

- its frequency is :math:`\omega^2 = (g - V^2/r)/H_s`, so its period shrinks
  like :math:`\sqrt{\varepsilon}` against the slow time, with
  :math:`\varepsilon = H_s/R_e`;
- its damping rate does **not** shrink. It is :math:`g/(V\,E_v)` with
  :math:`E_v = (L/D)\cos\sigma`, independent of the scale height, so the
  damping *ratio* is :math:`O(\sqrt\varepsilon)` and the glide is approached
  over a time that stays fixed as the layer thins. The equilibrium glide is
  therefore not an attracting slow manifold in the thin-atmosphere limit, and
  a boundary-layer estimate of Tikhonov type is not available for it;
- the lag of a trajectory behind the descending glide is
  :math:`C_{\mathcal R}\sqrt\varepsilon` scale heights, and
  :math:`C_{\mathcal R}\sqrt\varepsilon = 2\zeta` exactly: the residual
  constant and the damping ratio are one number.

Everything here is obtained from
:func:`aether.certification.rotating_field.point_mass_field` on a spherical,
non-rotating planet -- the planar components of the field of record, not a
planar model written for the occasion.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import sympy as sp

from aether.certification.rotating_field import Planet, exponential_density, point_mass_field
from aether.symbolic.ops import SYMPY_OPS

__all__ = ["GlideSymbols"]

_RADIUS = sp.Symbol("r", positive=True)
_SPEED = sp.Symbol("V", positive=True)
_GAMMA = sp.Symbol("gamma", real=True)
_BANK = sp.Symbol("sigma", real=True)
_MU = sp.Symbol("mu", positive=True)
_BODY_RADIUS = sp.Symbol("R_e", positive=True)
_SCALE_HEIGHT = sp.Symbol("H_s", positive=True)
_REFERENCE_DENSITY = sp.Symbol("rho_0", positive=True)
_BALLISTIC_COEFFICIENT = sp.Symbol("beta", positive=True)
_LIFT_TO_DRAG = sp.Symbol("E", positive=True)


@dataclass(frozen=True)
class GlideSymbols:
    """The planar point-mass glide, and the quantities derived from it."""

    radius: Any = _RADIUS
    speed: Any = _SPEED
    gamma: Any = _GAMMA
    bank: Any = _BANK
    mu: Any = _MU
    body_radius: Any = _BODY_RADIUS
    scale_height: Any = _SCALE_HEIGHT
    reference_density: Any = _REFERENCE_DENSITY
    ballistic_coefficient: Any = _BALLISTIC_COEFFICIENT
    lift_to_drag: Any = _LIFT_TO_DRAG

    @property
    def state(self) -> tuple[Any, Any, Any]:
        r""":math:`(r, V, \gamma)`."""
        return (self.radius, self.speed, self.gamma)

    @cached_property
    def rates(self) -> Any:
        r"""Rates of :math:`(r, V, \gamma)`: the planar components of the field of record.

        Longitude, latitude and heading are set to zero and discarded. On a
        spherical, non-rotating planet nothing in the remaining three rates
        depends on them, which the derivation checks assert rather than assume.
        """
        planet = Planet(mu=self.mu, radius=self.body_radius, j2=0, rotation_rate=0)
        field = point_mass_field(
            self.ballistic_coefficient,
            self.lift_to_drag,
            density=exponential_density(self.reference_density, self.scale_height),
            planet=planet,
        )
        rates = field((self.radius, 0, 0, self.speed, self.gamma, 0), self.bank, SYMPY_OPS)
        return sp.Matrix([rates[0], rates[3], rates[4]])

    # -- named quantities -----------------------------------------------------

    @property
    def gravity(self) -> Any:
        r""":math:`g = \mu/r^2`."""
        return self.mu / self.radius**2

    @property
    def net_gravity(self) -> Any:
        r""":math:`g - V^2/r`: gravity less the centrifugal relief, what lift balances."""
        return self.gravity - self.speed**2 / self.radius

    @property
    def vertical_lift_to_drag(self) -> Any:
        r""":math:`E_v = (L/D)\cos\sigma`."""
        return self.lift_to_drag * sp.cos(self.bank)

    @property
    def epsilon(self) -> Any:
        r"""The thin-atmosphere parameter :math:`\varepsilon = H_s/R_e`."""
        return self.scale_height / self.body_radius

    # -- the equilibrium glide --------------------------------------------------

    @cached_property
    def glide_ballistic_coefficient(self) -> Any:
        r"""The :math:`\beta` at which the glide balance holds at :math:`(r, V)`.

        Solving the balance for a vehicle parameter rather than for the
        altitude keeps :math:`(r, V)` free, so the linearisation can be taken
        at an arbitrary point of the glide instead of along one solution.
        """
        level = self.rates[2].subs(self.gamma, 0)
        (solution,) = sp.solve(sp.Eq(level, 0), self.ballistic_coefficient)
        return sp.simplify(solution)

    def on_glide(self, expression: Any) -> Any:
        r"""``expression`` at :math:`\gamma = 0` on the equilibrium glide."""
        return sp.sympify(expression).subs(self.gamma, 0).subs(
            self.ballistic_coefficient, self.glide_ballistic_coefficient
        )

    @cached_property
    def glide_descent_rate(self) -> Any:
        r""":math:`\dot h` of the equilibrium-glide altitude as speed bleeds off.

        By implicit differentiation of the balance :math:`B(r, V) = 0` along
        :math:`\dot V = -D/m`. Exact for this model; its thin-atmosphere limit
        is :math:`-2 H_s g / (V E_v)`.
        """
        balance = self.rates[2].subs(self.gamma, 0) * self.speed
        altitude_per_speed = -sp.diff(balance, self.speed) / sp.diff(balance, self.radius)
        deceleration = self.rates[1].subs(self.gamma, 0)
        return sp.simplify(self.on_glide(altitude_per_speed * deceleration))

    # -- the phugoid ----------------------------------------------------------

    @cached_property
    def jacobian(self) -> Any:
        r"""Jacobian of the planar field in :math:`(r, V, \gamma)`, on the glide."""
        frozen = self.rates.jacobian(sp.Matrix(self.state))
        return frozen.applyfunc(lambda entry: sp.simplify(self.on_glide(entry)))

    @cached_property
    def characteristic_coefficients(self) -> tuple[Any, Any, Any]:
        r""":math:`(a_2, a_1, a_0)` of :math:`\lambda^3 + a_2\lambda^2 + a_1\lambda + a_0`.

        The frozen-coefficient spectrum is one real root and the phugoid pair.
        Their sum is :math:`-a_2`, so the pair's decay rate is
        :math:`(a_2 + \lambda_{\mathrm{slow}})/2` *exactly*, and to leading
        order in the scale height :math:`\lambda_{\mathrm{slow}} = -a_0/a_1`.
        """
        lam = sp.Symbol("lambda")
        polynomial = sp.Poly((lam * sp.eye(3) - self.jacobian).det(), lam)
        _, a2, a1, a0 = (sp.simplify(c) for c in polynomial.all_coeffs())
        return a2, a1, a0

    @property
    def phugoid_frequency_squared(self) -> Any:
        r"""Leading order: :math:`\omega^2 = (g - V^2/r)/H_s`."""
        return self.net_gravity / self.scale_height

    @property
    def phugoid_damping_rate(self) -> Any:
        r"""Leading order, frozen coefficients: :math:`g/(V E_v)`, free of :math:`H_s`."""
        return self.gravity / (self.speed * self.vertical_lift_to_drag)

    @property
    def damping_ratio(self) -> Any:
        r""":math:`\zeta`: damping rate over frequency, :math:`O(\sqrt\varepsilon)`."""
        return self.phugoid_damping_rate / sp.sqrt(self.phugoid_frequency_squared)

    @property
    def residual_constant(self) -> Any:
        r""":math:`C_{\mathcal R}`: the lag behind the glide, per :math:`\sqrt\varepsilon`.

        Measured in scale heights. The closed form verification task R1-V4
        measures against,

        .. math::

            C_{\mathcal R} = \frac{2\sqrt{g R_e}}
                {V\,E_v\,\sqrt{1 - V^2/(g r)}} .
        """
        relief = 1 - self.speed**2 / (self.gravity * self.radius)
        return (
            2
            * sp.sqrt(self.gravity * self.body_radius)
            / (self.speed * self.vertical_lift_to_drag * sp.sqrt(relief))
        )
