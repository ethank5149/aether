r"""Piston theory on symbols: the pressure on a wall that moves under a fast stream.

The structural modes of the dynamics appendix are forced by the air, and the
appendix names the theory that supplies the force. This module is the
derivation behind the name, from one-dimensional gas dynamics to the
generalized forces on the modes, written so that each step is an identity
:mod:`aether.symbolic.derivations` can reduce to zero. The two sources are
Lighthill, *Oscillating Airfoils at High Mach Number* (J. Aeronaut. Sci. 20,
1953), and Ashley and Zartarian, *Piston Theory -- A New Aerodynamic Tool for
the Aeroelastician* (J. Aeronaut. Sci. 23, 1956); where they print a number
the tests reproduce it.

The chain has four links, and they are not of equal standing.

*The slab.* At high Mach number a plane slab of fluid normal to the stream
stays plane and moves in its own plane as a one-dimensional unsteady flow. A
wall :math:`z = Z(x, t)` then acts on the slab as a piston of velocity
:math:`w = \partial_t Z + U\,\partial_x Z`. The second sentence is the chain
rule (:func:`downwash`). The first is an asymptotic statement about the
equations of gas dynamics, and what this module can do for it is exhibit it
exactly where an exact solution exists: for a travelling wave of the wall
under a uniform stream, linearized (:class:`WavyWall`), the pressure is the
piston value times :math:`n/\sqrt{n^2 - 1}`, with :math:`n` the Mach number of
the stream relative to the wave. Beyond first order it is taken from the
sources and listed as such.

*The piston.* One-dimensional isentropic flow has two Riemann invariants
(:class:`IsentropicFlow`). Next to gas at rest one of them is uniform, which
fixes the speed of sound on the piston face from the piston's speed alone, and
with it the pressure (:meth:`PistonSymbols.simple_wave`). When a shock runs
ahead of the piston the pressure is a different function of the same speed
(:meth:`PistonSymbols.shock`), and the two agree through second order.

*The expansion.* First order is the acoustic impedance,
:math:`p - p_\infty = \rho_\infty a_\infty w`; third order is Lighthill's
cubic (:meth:`PistonSymbols.expansion`). The remainder after the cubic has a
sign and a size (:attr:`PistonSymbols.remainder_coefficient`).

*The load.* On a thin strip in bending the first-order pressure on the two
faces is a force per unit length opposing the downwash, and projecting it on
the mode shapes gives a damping matrix that is a Gram matrix and a stiffness
matrix that is not symmetric (:class:`StripSymbols`).

The ratio of specific heats is written :math:`\kappa`. The sources write
:math:`\gamma`, which in this package and in the manuscript is the
flight-path angle.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import sympy as sp

__all__ = ["IsentropicFlow", "PistonSymbols", "StripSymbols", "WavyWall", "downwash"]

_HEAT_RATIO = sp.Symbol("kappa", positive=True)
_PRESSURE = sp.Symbol("p_inf", positive=True)
_DENSITY = sp.Symbol("rho_inf", positive=True)
_PISTON_MACH = sp.Symbol("mu_w", real=True)
_X, _Z, _T = sp.symbols("x z t", real=True)
_STREAM = sp.Symbol("U", positive=True)
_ENTROPY_CONSTANT = sp.Symbol("K", positive=True)
_SOUND_SPEED = sp.Symbol("a_inf", positive=True)
_WAVENUMBER = sp.Symbol("k_x", positive=True)
_RELATIVE_MACH = sp.Symbol("n", positive=True)
_STRIP_STATION = sp.Symbol("xi", real=True)


def downwash(wall: Any, station: Any, time: Any, stream: Any) -> Any:
    r"""The piston velocity of a wall under a stream,
    :math:`w = \partial_t Z + U\,\partial_x Z`.

    ``wall`` is :math:`Z(x, t)`, measured from the surface into the fluid.
    Lighthill's Eq. (1); Ashley and Zartarian's Eq. (6).
    """
    return sp.diff(wall, time) + stream * sp.diff(wall, station)


@dataclass(frozen=True)
class IsentropicFlow:
    r"""One-dimensional isentropic flow of a perfect gas, as fields of :math:`(x, t)`.

    The density and the velocity are free functions. The pressure is
    :math:`p = K\rho^\kappa` and the speed of sound
    :math:`a = \sqrt{\kappa p/\rho}`, so what is assumed of the gas is in
    those two lines and nowhere else.
    """

    heat_ratio: Any = _HEAT_RATIO
    entropy_constant: Any = _ENTROPY_CONSTANT
    station: Any = _X
    time: Any = _T

    @cached_property
    def density(self) -> Any:
        return sp.Function("rho", positive=True)(self.station, self.time)

    @cached_property
    def velocity(self) -> Any:
        return sp.Function("u", real=True)(self.station, self.time)

    @property
    def pressure(self) -> Any:
        r""":math:`p = K\rho^\kappa`."""
        return self.entropy_constant * self.density**self.heat_ratio

    @property
    def sound_speed(self) -> Any:
        r""":math:`a = \sqrt{\kappa p/\rho}`."""
        return sp.sqrt(self.heat_ratio * self.pressure / self.density)

    @property
    def continuity(self) -> Any:
        r""":math:`\partial_t\rho + \partial_x(\rho u)`, which the flow sets to zero."""
        return sp.diff(self.density, self.time) + sp.diff(
            self.density * self.velocity, self.station
        )

    @property
    def momentum(self) -> Any:
        r""":math:`\partial_t u + u\,\partial_x u + \partial_x p/\rho`, likewise zero."""
        return (
            sp.diff(self.velocity, self.time)
            + self.velocity * sp.diff(self.velocity, self.station)
            + sp.diff(self.pressure, self.station) / self.density
        )

    def invariant(self, sign: int) -> Any:
        r"""The Riemann invariant :math:`u \pm 2a/(\kappa - 1)`."""
        return self.velocity + sign * 2 * self.sound_speed / (self.heat_ratio - 1)

    def along_characteristic(self, expression: Any, sign: int) -> Any:
        r"""Rate of ``expression`` along :math:`dx/dt = u \pm a`."""
        speed = self.velocity + sign * self.sound_speed
        return sp.diff(expression, self.time) + speed * sp.diff(expression, self.station)


@dataclass(frozen=True)
class PistonSymbols:
    r"""The pressure on a piston as a function of its speed, three ways.

    Everything is a ratio to the undisturbed pressure, as a function of
    :math:`w/a_\infty`: the piston's speed into the gas over the undisturbed
    speed of sound. Positive is compression.
    """

    heat_ratio: Any = _HEAT_RATIO
    pressure: Any = _PRESSURE
    density: Any = _DENSITY
    piston_mach: Any = _PISTON_MACH

    @property
    def sound_speed(self) -> Any:
        r""":math:`a_\infty = \sqrt{\kappa p_\infty/\rho_\infty}`."""
        return sp.sqrt(self.heat_ratio * self.pressure / self.density)

    @property
    def impedance(self) -> Any:
        r"""The acoustic impedance :math:`\rho_\infty a_\infty`."""
        return self.density * self.sound_speed

    @property
    def exponent(self) -> Any:
        r""":math:`2\kappa/(\kappa - 1)`: seven, for :math:`\kappa = 1.4`."""
        return 2 * self.heat_ratio / (self.heat_ratio - 1)

    # -- the simple wave ----------------------------------------------------------

    def sound_ratio(self, mach: Any = None) -> Any:
        r""":math:`a/a_\infty = 1 + \tfrac{\kappa - 1}{2}\,w/a_\infty` on the piston face.

        The invariant :math:`u - 2a/(\kappa - 1)` is carried to the face from
        gas at rest, where it is :math:`-2a_\infty/(\kappa - 1)`; on the face
        :math:`u = w`.
        """
        mach = self.piston_mach if mach is None else mach
        return 1 + (self.heat_ratio - 1) * mach / 2

    def simple_wave(self, mach: Any = None) -> Any:
        r""":math:`p/p_\infty =
        \bigl(1 + \tfrac{\kappa-1}{2}\,w/a_\infty\bigr)^{2\kappa/(\kappa-1)}`.

        Exact while the piston generates only simple waves. Lighthill's
        Eq. (4); Ashley and Zartarian's Eq. (1).
        """
        return self.sound_ratio(mach) ** self.exponent

    # -- the shock ----------------------------------------------------------------

    def shock_mach(self, mach: Any = None) -> Any:
        r"""Mach number of the shock an advancing piston drives into gas at rest.

        :math:`M_s = b + \sqrt{1 + b^2}` with
        :math:`b = (\kappa + 1)\,w/(4a_\infty)`: the positive root of
        :math:`w/a_\infty = \tfrac{2}{\kappa + 1}(M_s - 1/M_s)`.
        """
        mach = self.piston_mach if mach is None else mach
        b = (self.heat_ratio + 1) * mach / 4
        return b + sp.sqrt(1 + b**2)

    def shock(self, mach: Any = None) -> Any:
        r""":math:`p/p_\infty = 1 + \kappa\,(w/a_\infty)\,M_s` behind that shock.

        Meaningful for an advancing piston only. A receding one sends an
        expansion, which is a simple wave.
        """
        mach = self.piston_mach if mach is None else mach
        return 1 + self.heat_ratio * mach * self.shock_mach(mach)

    def shock_jump(self, shock_mach: Any) -> tuple[Any, Any, Any]:
        r"""Speed, pressure and density behind a shock of Mach number :math:`M_s`.

        Returned as :math:`(w/a_\infty,\ p/p_\infty,\ \rho/\rho_\infty)`, the
        gas ahead being at rest. These are the relations the three
        conservation laws across the shock are checked against.
        """
        k = self.heat_ratio
        speed = 2 / (k + 1) * (shock_mach - 1 / shock_mach)
        pressure = 1 + 2 * k / (k + 1) * (shock_mach**2 - 1)
        density = (k + 1) * shock_mach**2 / ((k - 1) * shock_mach**2 + 2)
        return speed, pressure, density

    def shock_then_expansion(self, mach: Any, shock_at: Any) -> Any:
        r"""Pressure when a shock formed at piston speed ``shock_at`` and a simple wave
        followed.

        The shock sets the state behind it; from there the invariant is
        carried as before, starting from that state instead of from rest.
        This is the "shock-expansion" column of Lighthill's Table 1, whose
        entries take the shock at :math:`w/a_\infty = 1`.
        """
        _, pressure, density = self.shock_jump(self.shock_mach(shock_at))
        sound = sp.sqrt(pressure / density)
        return pressure * (
            (sound + (self.heat_ratio - 1) * (mach - shock_at) / 2) / sound
        ) ** self.exponent

    # -- the expansions -----------------------------------------------------------

    def coefficients(self, order: int = 3, *, through_shock: bool = False) -> tuple[Any, ...]:
        r"""Coefficients of :math:`(w/a_\infty)^k`, :math:`k = 0..` ``order``, in
        :math:`p/p_\infty`.

        Stated, not computed: these are the numbers the sources print, and
        the derivations check them against the Taylor coefficients of
        :meth:`simple_wave` and :meth:`shock`. The two expansions share their
        first three terms and part at the fourth.
        """
        if not 0 <= order <= 3:
            raise ValueError(f"the sources carry the expansion to third order, not {order}")
        k = self.heat_ratio
        third = k * (k + 1) ** 2 / 32 if through_shock else k * (k + 1) / 12
        return (sp.S.One, k, k * (k + 1) / 4, third)[: order + 1]

    def expansion(
        self, order: int = 3, mach: Any = None, *, through_shock: bool = False
    ) -> Any:
        r"""The expansion of :math:`p/p_\infty` to ``order`` in :math:`w/a_\infty`.

        Order one is the small-disturbance relation
        :math:`p - p_\infty = \rho_\infty a_\infty w`; order two resembles
        Busemann's; order three is Lighthill's Eq. (5), Ashley and
        Zartarian's Eq. (4).
        """
        mach = self.piston_mach if mach is None else mach
        terms = self.coefficients(order, through_shock=through_shock)
        return sum(
            (coefficient * mach**power for power, coefficient in enumerate(terms)), sp.S.Zero
        )

    @property
    def remainder_coefficient(self) -> Any:
        r""":math:`\kappa(\kappa + 1)(3 - \kappa)/96`: the fourth-order coefficient.

        The simple wave less its cubic is this times
        :math:`(w/a_\infty)^4\,(a_\xi/a_\infty)^{2\kappa/(\kappa-1) - 4}` at
        some intermediate piston speed, by Taylor's theorem. It is positive
        for :math:`1 < \kappa < 3`, so the cubic lies below the simple wave
        for compression and expansion alike.
        """
        k = self.heat_ratio
        return k * (k + 1) * (3 - k) / 96


@dataclass(frozen=True)
class WavyWall:
    r"""Linearized supersonic flow over a travelling wave of the wall, solved exactly.

    The perturbation potential of a uniform stream :math:`U` along
    :math:`x` satisfies :math:`\varphi_{xx} + \varphi_{zz} = a^{-2}
    (\partial_t + U\partial_x)^2\varphi`. The wall is a wave of arbitrary
    profile, :math:`Z = G(k_x x - \omega t)`, and

    .. math:: n = \frac{U - \omega/k_x}{a}

    is the Mach number of the stream relative to it. The frequency is
    carried through ``relative_mach`` rather than the other way round, so
    that the square root below is of a named quantity. For :math:`n > 1`
    the solution that radiates away from the wall is

    .. math:: \varphi = -\frac{a\,n}{\sqrt{n^2 - 1}}\;
              G\bigl(k_x x - k_x\sqrt{n^2 - 1}\,z - \omega t\bigr).

    For a wall at rest :math:`n = M` and the pressure is Ackeret's. Piston
    theory is the limit :math:`n \to \infty`, and this is the one case in
    which what it leaves out can be written down.
    """

    stream: Any = _STREAM
    sound_speed: Any = _SOUND_SPEED
    density: Any = _DENSITY
    wavenumber: Any = _WAVENUMBER
    relative_mach: Any = _RELATIVE_MACH
    station: Any = _X
    normal: Any = _Z
    time: Any = _T

    @property
    def frequency(self) -> Any:
        r""":math:`\omega = k_x\,(U - n a)`."""
        return self.wavenumber * (self.stream - self.relative_mach * self.sound_speed)

    @property
    def mach_slope(self) -> Any:
        r""":math:`\sqrt{n^2 - 1}`: how far downstream a front leans per unit height."""
        return sp.sqrt(self.relative_mach**2 - 1)

    def _profile(self, height: Any) -> Any:
        phase = (
            self.wavenumber * self.station
            - self.wavenumber * self.mach_slope * height
            - self.frequency * self.time
        )
        return sp.Function("G")(phase)

    @cached_property
    def wall(self) -> Any:
        r""":math:`Z(x, t) = G(k_x x - \omega t)`."""
        return self._profile(sp.S.Zero)

    @cached_property
    def potential(self) -> Any:
        return -self.sound_speed * self.relative_mach / self.mach_slope * self._profile(
            self.normal
        )

    def convected(self, expression: Any) -> Any:
        r""":math:`(\partial_t + U\,\partial_x)` of ``expression``."""
        return sp.diff(expression, self.time) + self.stream * sp.diff(expression, self.station)

    @property
    def wave_equation(self) -> Any:
        """The linearized potential equation, as a residual."""
        phi = self.potential
        return (
            sp.diff(phi, self.station, 2)
            + sp.diff(phi, self.normal, 2)
            - self.convected(self.convected(phi)) / self.sound_speed**2
        )

    @property
    def wall_velocity(self) -> Any:
        r"""Normal velocity of the fluid at the wall, :math:`\partial_z\varphi` at
        :math:`z = 0`."""
        return sp.diff(self.potential, self.normal).subs(self.normal, 0)

    @property
    def wall_downwash(self) -> Any:
        r"""What the wall imposes there, :math:`\partial_t Z + U\,\partial_x Z`."""
        return downwash(self.wall, self.station, self.time, self.stream)

    @property
    def wall_slope(self) -> Any:
        r""":math:`\partial_x Z`."""
        return sp.diff(self.wall, self.station)

    @property
    def wall_pressure(self) -> Any:
        r"""Perturbation pressure at the wall,
        :math:`-\rho\,(\partial_t + U\partial_x)\varphi`."""
        return (-self.density * self.convected(self.potential)).subs(self.normal, 0)

    @property
    def piston_factor(self) -> Any:
        r""":math:`n/\sqrt{n^2 - 1}`: the exact pressure over the piston value."""
        return self.relative_mach / self.mach_slope

    @property
    def piston_excess(self) -> Any:
        r"""The factor less one, written so that its size is visible:

        .. math:: \frac{n}{\sqrt{n^2-1}} - 1
                  = \frac{1}{\sqrt{n^2-1}\,\bigl(n + \sqrt{n^2-1}\bigr)}
                  \le \frac{1}{2\,(n^2 - 1)} .
        """
        return 1 / (self.mach_slope * (self.relative_mach + self.mach_slope))


@dataclass(frozen=True)
class StripSymbols:
    r"""A thin strip in bending, with first-order piston pressure on both faces.

    The strip lies along the stream, station :math:`\xi` running from the
    leading edge aft, with width :math:`b(\xi)`. Its deflection normal to
    itself is a sum over mode shapes, :math:`\zeta(\xi, t) =
    \sum_j \eta_j(t)\,W_j(\xi)`. Each face sees the other's downwash with
    the sign reversed, so to first order the two pressures add to a force
    per unit length

    .. math:: f = -\,b\,z_a\,\bigl(\partial_t\zeta + U\,\partial_\xi\zeta\bigr),

    opposing the downwash, where :math:`z_a` is the sum of the acoustic
    impedances :math:`\rho a` on the two faces and :math:`U` the speed of
    the stream along the strip. Under a uniform stream
    :math:`z_a = 2\rho_\infty a_\infty` and :math:`U = V`; both are left as
    functions of the station, because first order is the statement
    :math:`dp = \rho a\,dw` and that holds about any state of the gas, not
    only the undisturbed one.
    """

    modes: int = 2
    station: Any = _STRIP_STATION
    time: Any = _T

    @cached_property
    def width(self) -> Any:
        return sp.Function("b", positive=True)(self.station)

    @cached_property
    def impedance(self) -> Any:
        r""":math:`z_a(\xi)`: the two faces' :math:`\rho a`, added."""
        return sp.Function("z_a", positive=True)(self.station)

    @cached_property
    def stream(self) -> Any:
        r""":math:`U(\xi)`: speed of the stream along the strip."""
        return sp.Function("U", positive=True)(self.station)

    @cached_property
    def shapes(self) -> tuple[Any, ...]:
        return tuple(
            sp.Function(f"W_{i}", real=True)(self.station) for i in range(1, self.modes + 1)
        )

    @cached_property
    def amplitudes(self) -> tuple[Any, ...]:
        return tuple(
            sp.Function(f"eta_{i}", real=True)(self.time) for i in range(1, self.modes + 1)
        )

    @cached_property
    def deflection(self) -> Any:
        r""":math:`\zeta(\xi, t) = \sum_j \eta_j(t)\,W_j(\xi)`."""
        return sum(
            (eta * shape for eta, shape in zip(self.amplitudes, self.shapes, strict=True)),
            sp.S.Zero,
        )

    @property
    def downwash(self) -> Any:
        r""":math:`\partial_t\zeta + U\,\partial_\xi\zeta`."""
        return downwash(self.deflection, self.station, self.time, self.stream)

    @property
    def load(self) -> Any:
        """Force per unit length on the strip, along its deflection."""
        return -self.width * self.impedance * self.downwash

    def damping_density(self, i: int, j: int) -> Any:
        r"""Integrand of :math:`C_{ij} = \int b\,z_a\,W_i W_j\,d\xi`."""
        return self.width * self.impedance * self.shapes[i] * self.shapes[j]

    def stiffness_density(self, i: int, j: int) -> Any:
        r"""Integrand of :math:`G_{ij} = \int b\,z_a\,U\,W_i\,W_j'\,d\xi`."""
        return (
            self.width
            * self.impedance
            * self.stream
            * self.shapes[i]
            * sp.diff(self.shapes[j], self.station)
        )

    def forcing_density(self, i: int) -> Any:
        r"""Integrand of the generalized force :math:`Q_i = \int f\,W_i\,d\xi`."""
        return self.load * self.shapes[i]
