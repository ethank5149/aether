r"""The translational entry field over a rotating, oblate planet, for every arithmetic.

:mod:`aether.certification.entry_field` is the classical field: spherical
gravity and no planetary rotation. This is the field the proposal's dynamics
appendix states -- geocentric spherical position :math:`(r, \lambda, \phi)`,
planet-relative velocity in wind-frame components :math:`(V, \gamma, \psi)`,
:math:`J_2` gravity, and the Coriolis and centrifugal terms of the rotating
frame -- and it is written, like that one, against
:class:`~aether.certification.rigorous.MathOps`.

The reason to add an arithmetic rather than a second implementation is the same
as before, and here it is the whole point. Evaluated on floats the field
simulates; on Arb balls it encloses; and on SymPy symbols
(:data:`aether.symbolic.SYMPY_OPS`) it *is* the displayed equations, so the
manuscript's derivations are checked against the object the code integrates
rather than against a transcription of it, and the compiled routines
:mod:`aether.symbolic.codegen` emits are printed from that same expression.

The aerodynamic force enters as a specific force resolved on the wind triad,
:math:`(F_V, F_\gamma, F_\psi)/m`, which is where the appendix separates the
translational equations from whatever produces the force: a lift-and-drag
point mass (:func:`point_mass_field`) or, in the six-degree-of-freedom
formulation, the attitude quaternion.

What is deliberately retained
-----------------------------

Planetary rotation is not a refinement of the vertical force balance. At 7 km/s and
50 km, flying east, the Coriolis term :math:`2\omega_E V\cos\phi\sin\psi` is a
third to a half of :math:`g - V^2/r`, the quantity the lift has to balance;
the :math:`J_2` terms are a percent of it at most. Both are carried, and
:func:`vertical_balance_terms` reports them separately so that statement can be
checked rather than quoted.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from aether.certification.rigorous import NUMPY_OPS, MathOps, VectorField, lower_bound
from aether.orbital.gravity import EARTH as _EARTH_GRAVITY

__all__ = [
    "EARTH",
    "Planet",
    "exponential_density",
    "j2_gravity",
    "jacobi_energy",
    "point_mass_field",
    "rotating_entry_field",
    "vertical_balance_terms",
]

#: Exponential atmosphere, matching :mod:`aether.certification.entry_field`.
_RHO0 = 1.225
_H_SCALE = 8500.0


@dataclass(frozen=True)
class Planet:
    """Central-body constants, as numbers, Arb balls or symbols.

    Untyped on purpose. :class:`aether.orbital.gravity.GravityModel` validates
    its fields as finite floats, which is right for a simulation and rules out
    exactly the two uses this field exists for: a derivation carried out in
    symbols, and a certificate discharged over a range of constants.
    """

    mu: Any
    """Standard gravitational parameter (m³/s²)."""
    radius: Any
    """Equatorial reference radius (m)."""
    j2: Any
    """Second zonal harmonic coefficient."""
    rotation_rate: Any
    """Sidereal rotation rate (rad/s)."""

    def spherical(self) -> Planet:
        """The same body with the oblateness removed."""
        return replace(self, j2=0)

    def non_rotating(self) -> Planet:
        """The same body with the rotation removed."""
        return replace(self, rotation_rate=0)


#: Earth: the gravity constants of :data:`aether.orbital.gravity.EARTH`, and the
#: sidereal rotation rate (IERS conventions) :mod:`aether.geodesy` uses.
EARTH = Planet(
    mu=_EARTH_GRAVITY.mu,
    radius=_EARTH_GRAVITY.radius,
    j2=_EARTH_GRAVITY.j2,
    rotation_rate=7.292115e-5,
)


def j2_gravity(
    radius: Any, latitude: Any, ops: MathOps = NUMPY_OPS, *, planet: Planet = EARTH
) -> tuple[Any, Any]:
    r"""Downward and northward gravity, :math:`(g_D, g_N)`, of the :math:`J_2` field.

    The gradient of

    .. math::

        \Phi_g(r, \phi) = -\frac{\mu}{r}\Bigl[1 - \tfrac12 J_2
            \Bigl(\frac{R_e}{r}\Bigr)^2 (3\sin^2\phi - 1)\Bigr]

    resolved in the local north-east-down frame, :math:`g_D = \partial_r\Phi_g`
    and :math:`g_N = -r^{-1}\partial_\phi\Phi_g`. There is no eastward
    component, by axial symmetry.

    Explicit in :math:`(r, \phi)`, with no square root of a sum of squares, so
    an interval evaluation does not pay the dependency a Cartesian
    :math:`\lVert\mathbf r\rVert` would cost. It is the same field as
    :func:`aether.orbital.gravity.gravitational_acceleration`, and the tests
    hold the two against each other.
    """
    sin_lat = ops.sin(latitude)
    cos_lat = ops.cos(latitude)
    central = planet.mu / (radius * radius)
    oblateness = planet.j2 * (planet.radius / radius) * (planet.radius / radius)
    g_down = central * (1 + 3 * oblateness * (1 - 3 * sin_lat * sin_lat) / 2)
    g_north = -3 * central * oblateness * sin_lat * cos_lat
    return g_down, g_north


def rotating_entry_field(
    state: Sequence[Any],
    specific_force: Sequence[Any],
    ops: MathOps = NUMPY_OPS,
    *,
    planet: Planet = EARTH,
) -> list[Any]:
    r"""Rates of :math:`(r, \lambda, \phi, V, \gamma, \psi)` over any arithmetic.

    ``specific_force`` is :math:`(F_V, F_\gamma, F_\psi)/m`: along the
    velocity, normal to it in the vertical plane (positive up at level flight),
    and lateral (positive to the right of the velocity). For a point mass these
    are :math:`(-D, L\cos\sigma, L\sin\sigma)/m`.

    ``planet`` is an argument rather than a set of module constants so that a
    derivation can be carried out in symbols and a certificate discharged over
    a range of them. ``planet.spherical().non_rotating()`` recovers the
    classical equations of
    :func:`aether.certification.entry_field.entry_field`.

    Two coordinate singularities are inherited from the chart and are not
    removed here: :math:`\cos\phi = 0` at the poles, in both the longitude and
    the heading rate, and :math:`\cos\gamma = 0` in vertical flight, in the
    heading rate. An enclosure reaching either returns an unbounded interval
    rather than a wrong one.
    """
    radius, _longitude, latitude, speed, gamma, heading = state
    force_v, force_gamma, force_psi = specific_force
    g_down, g_north = j2_gravity(radius, latitude, ops, planet=planet)

    sin_gamma, cos_gamma = ops.sin(gamma), ops.cos(gamma)
    sin_psi, cos_psi = ops.sin(heading), ops.cos(heading)
    sin_lat, cos_lat = ops.sin(latitude), ops.cos(latitude)
    # Centrifugal acceleration has magnitude w^2 r cos(phi), directed away from
    # the rotation axis; the Coriolis terms are linear in the rate.
    centrifugal = planet.rotation_rate * planet.rotation_rate * radius * cos_lat
    coriolis = 2 * planet.rotation_rate * speed

    return [
        speed * sin_gamma,
        speed * cos_gamma * sin_psi / (radius * cos_lat),
        speed * cos_gamma * cos_psi / radius,
        force_v
        - g_down * sin_gamma
        + g_north * cos_gamma * cos_psi
        + centrifugal * (sin_gamma * cos_lat - cos_gamma * cos_psi * sin_lat),
        (
            force_gamma
            - (g_down - speed * speed / radius) * cos_gamma
            - g_north * sin_gamma * cos_psi
            + coriolis * cos_lat * sin_psi
            + centrifugal * (cos_gamma * cos_lat + sin_gamma * cos_psi * sin_lat)
        )
        / speed,
        # Grouped as the classical field groups it -- the lateral force and the
        # rotating-frame terms each over cos(gamma), the great-circle term not --
        # so that on a spherical, non-rotating planet the two agree to the bit.
        (
            force_psi / cos_gamma
            + speed * speed * cos_gamma * sin_psi * ops.tan(latitude) / radius
            + (
                -g_north * sin_psi
                + coriolis * (sin_lat * cos_gamma - sin_gamma * cos_lat * cos_psi)
                + centrifugal * sin_lat * sin_psi
            )
            / cos_gamma
        )
        / speed,
    ]


def jacobi_energy(
    state: Sequence[Any], ops: MathOps = NUMPY_OPS, *, planet: Planet = EARTH
) -> Any:
    r"""Specific mechanical energy in the rotating frame (J/kg).

    .. math::

        \mathcal E = \tfrac12 V^2 + \Phi_g(r, \phi)
            - \tfrac12\,\omega_E^2 r^2\cos^2\phi .

    The Jacobi integral: Coriolis force does no work, the centrifugal force is
    conservative, and so along :func:`rotating_entry_field`

    .. math::

        \dot{\mathcal E} = \frac{F_V}{m}\,V ,

    whatever the lift, the bank, or the gravity model's oblateness. With
    :math:`F_V = -D \le 0` the energy is non-increasing along every trajectory
    of every selection of an aerodynamic enclosure, which is what bounds the
    total dissipation -- and with it the number of atmospheric passes that can
    each dissipate a given amount -- without reference to a heating
    correlation.
    """
    radius, _longitude, latitude, speed, _gamma, _heading = state
    sin_lat, cos_lat = ops.sin(latitude), ops.cos(latitude)
    ratio = (planet.radius / radius) * (planet.radius / radius)
    potential = -planet.mu / radius * (1 - planet.j2 * ratio * (3 * sin_lat * sin_lat - 1) / 2)
    spin = planet.rotation_rate * radius * cos_lat
    return speed * speed / 2 + potential - spin * spin / 2


def vertical_balance_terms(
    state: Sequence[Any], ops: MathOps = NUMPY_OPS, *, planet: Planet = EARTH
) -> dict[str, Any]:
    r"""The terms the vertical lift must balance for :math:`\dot\gamma = 0`.

    Setting the flight-path-angle rate of :func:`rotating_entry_field` to zero
    gives the condition for the flight path to hold its angle,

    .. math::

        \frac{F_\gamma}{m} = \Bigl(g_D - \frac{V^2}{r}\Bigr)\cos\gamma
            + g_N\sin\gamma\cos\psi
            - 2\omega_E V\cos\phi\sin\psi
            - \omega_E^2 r\cos\phi\,
              (\cos\gamma\cos\phi + \sin\gamma\cos\psi\sin\phi),

    and the entries returned are its four right-hand terms in that order, with
    their signs, under the keys ``"central"``, ``"oblate_north"``,
    ``"coriolis"`` and ``"centrifugal"``. Their sum is the required
    :math:`F_\gamma/m`; ``"oblate_radial"`` is reported separately as the part
    of ``"central"`` that :math:`J_2` contributes, so the sizes can be compared.
    """
    radius, _longitude, latitude, speed, gamma, heading = state
    g_down, g_north = j2_gravity(radius, latitude, ops, planet=planet)
    sin_gamma, cos_gamma = ops.sin(gamma), ops.cos(gamma)
    sin_lat, cos_lat = ops.sin(latitude), ops.cos(latitude)
    centrifugal = planet.rotation_rate * planet.rotation_rate * radius * cos_lat
    spherical = planet.mu / (radius * radius)
    return {
        "central": (g_down - speed * speed / radius) * cos_gamma,
        "oblate_north": g_north * sin_gamma * ops.cos(heading),
        "coriolis": -2 * planet.rotation_rate * speed * cos_lat * ops.sin(heading),
        "centrifugal": -centrifugal
        * (cos_gamma * cos_lat + sin_gamma * ops.cos(heading) * sin_lat),
        "oblate_radial": (g_down - spherical) * cos_gamma,
    }


def exponential_density(
    reference_density: Any = _RHO0, scale_height: Any = _H_SCALE
) -> Callable[[Any, MathOps], Any]:
    r"""An isothermal atmosphere, :math:`\rho = \rho_0\,e^{-h/H_s}`, as ``density(h, ops)``."""

    def density(altitude: Any, ops: MathOps) -> Any:
        return reference_density * ops.exp(-altitude / scale_height)

    return density


def _refuse_below_surface(altitude: Any) -> None:
    """Refuse an enclosure that reaches below the surface, as the classical field does.

    A symbolic altitude carries no bound to test and is let through: the
    refusal exists to stop an interval straddling zero from being silently
    widened, and a symbol is not an interval.
    """
    if getattr(altitude, "free_symbols", None):
        return
    lowest = lower_bound(altitude)
    if lowest < 0.0:
        raise ValueError(
            f"the enclosure reaches below the surface (altitude down to "
            f"{lowest:.1f} m); an interval spanning zero cannot be clamped "
            f"without a branch on sign"
        )


def point_mass_field(
    ballistic_coefficient: Any,
    lift_to_drag: Any,
    *,
    density: Callable[[Any, MathOps], Any] | None = None,
    planet: Planet = EARTH,
) -> VectorField:
    r""":func:`rotating_entry_field` closed over a lift-and-drag point mass.

    Returns ``f(state, bank, ops)`` with
    :math:`(F_V, F_\gamma, F_\psi)/m = (-D, L\cos\sigma, L\sin\sigma)/m`,
    :math:`D/m = \rho V^2/(2\beta)` and :math:`L = (L/D)\,D`. ``density`` is
    ``density(altitude, ops)`` and defaults to the exponential atmosphere of
    :mod:`aether.certification.entry_field`, so that on
    ``planet.spherical().non_rotating()`` this is that module's field.
    """
    atmosphere = exponential_density() if density is None else density

    def field(state: Sequence[Any], bank: Any, ops: MathOps) -> list[Any]:
        radius, speed = state[0], state[3]
        altitude = radius - planet.radius
        _refuse_below_surface(altitude)
        drag = atmosphere(altitude, ops) * speed * speed / (2 * ballistic_coefficient)
        lift = lift_to_drag * drag
        return rotating_entry_field(
            state,
            (-drag, lift * ops.cos(bank), lift * ops.sin(bank)),
            ops,
            planet=planet,
        )

    return field
