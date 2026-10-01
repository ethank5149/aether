r"""The derivations the manuscript relies on, each as an identity to be reduced to zero.

A derivation checked by reading is checked once, by one reader, against
whatever was on the page that day. Each entry below instead states a claim the
proposal makes as a *residual* -- left side minus right side, built from the
field of record -- and :func:`check` reduces it by exact rewriting. Zero is a
proof; anything else is printed, and is either a wrong claim or a claim stated
so that SymPy cannot see it, which amounts to the same instruction: fix it.

The registry is the contract with the manuscript. ``where`` names the place
the claim is used, so that a changed equation has a named check to fail, and a
check that fails names the paragraph to reread.

Two kinds of entry are worth distinguishing, because they license different
sentences:

*Identities* hold exactly for the model as written -- the equations of motion
are Newton's law, the energy dissipates at the drag power.

*Limits* hold to leading order in the scale height -- the phugoid frequency
and damping -- and are stated as the limit of an exact expression, so that
what is neglected is visible rather than implied.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import sympy as sp

from aether.symbolic.entry import EntrySymbols
from aether.symbolic.glide import GlideSymbols
from aether.symbolic.ops import is_identically_zero, lie_derivative, reduce

__all__ = ["DERIVATIONS", "Derivation", "DerivationResult", "check", "check_all"]


@dataclass(frozen=True)
class Derivation:
    """One claim, where it is used, and the residual that must vanish."""

    key: str
    claim: str
    where: str
    residual: Callable[[], Any]
    #: Identities hold exactly; limits hold to leading order in the scale height.
    kind: str = "identity"


@dataclass(frozen=True)
class DerivationResult:
    """The outcome of reducing one residual."""

    derivation: Derivation
    passed: bool
    seconds: float
    #: What the residual reduced to when it did not reach zero, for the report.
    remainder: str = ""


# -- gravity ------------------------------------------------------------------


def _gravity_is_the_gradient_of_the_potential() -> Any:
    s = EntrySymbols()
    g_down, g_north = s.gravity
    return sp.Matrix(
        [
            sp.diff(s.potential, s.radius) - g_down,
            -sp.diff(s.potential, s.latitude) / s.radius - g_north,
        ]
    )


def _spherical_gravity_is_the_cartesian_j2_field() -> Any:
    r"""The Cartesian :math:`J_2` perturbation, projected, is the spherical one.

    The Cartesian field carries :math:`(1 - 5z^2/r^2)` on its equatorial
    components and :math:`(3 - 5z^2/r^2)` on its polar one; projected on the
    radial direction the pair collapses to :math:`(1 - 3\sin^2\phi)`. Carrying
    the equatorial factor into the radial component instead doubles the polar
    correction, which is the slip this check exists to catch.
    """
    s = EntrySymbols()
    planet = s.planet
    x, y, z = s.position
    z_ratio = 5 * z**2 / s.radius**2
    factor = -sp.Rational(3, 2) * planet.j2 * planet.mu * planet.radius**2 / s.radius**5
    cartesian = factor * sp.Matrix([x * (1 - z_ratio), y * (1 - z_ratio), z * (3 - z_ratio)])
    g_down, g_north = s.gravity
    oblate_down = g_down - planet.mu / s.radius**2
    return sp.Matrix(
        [
            cartesian.dot(s.up) + oblate_down,
            cartesian.dot(s.north) - g_north,
            cartesian.dot(s.east),
        ]
    )


def _radial_oblateness_changes_sign_at_35_degrees() -> Any:
    r"""The radial :math:`J_2` factor :math:`1 - 3\sin^2\phi`: where it vanishes, and its range.

    Zero at :math:`\arcsin(1/\sqrt3)`; :math:`+1` at the equator and
    :math:`-2` at the poles, so the correction is twice as large, and of the
    opposite sign, at the pole.
    """
    phi = sp.Symbol("phi", real=True)
    factor = 1 - 3 * sp.sin(phi) ** 2
    return sp.Matrix(
        [
            factor.subs(phi, sp.asin(1 / sp.sqrt(3))),
            factor.subs(phi, 0) - 1,
            factor.subs(phi, sp.pi / 2) + 2,
        ]
    )


# -- the equations of motion ----------------------------------------------------


def _position_rates_are_the_velocity() -> Any:
    """The three position rates are the planet-relative velocity, resolved."""
    s = EntrySymbols()
    v_hat, _, _ = s.wind_triad
    return lie_derivative(s.position, s.state, s.rates) - s.speed * v_hat


def _the_field_is_newtons_law_in_the_rotating_frame() -> Any:
    r"""The three velocity rates are Newton's second law in the rotating frame.

    Differentiate the planet-relative velocity :math:`V\hat V` along the flow
    in planet-fixed Cartesian components, where no basis vector moves, and
    compare with

    .. math::

        \frac{\mathbf F_a}{m} + \mathbf g
        - 2\,\boldsymbol\omega_E\times\mathbf v
        - \boldsymbol\omega_E\times(\boldsymbol\omega_E\times\mathbf r).

    Nothing about the moving local frame is assumed: the transport terms that
    make the three wind-axis equations carry different powers of
    :math:`\cos\gamma` come out of the differentiation, which is the point of
    doing it this way.

    The difference is resolved on the local north-east-up basis, where the
    longitude drops out and the reduction is ten times cheaper. A vector that
    vanishes on one orthonormal basis vanishes on every other, the wind triad
    included.
    """
    s = EntrySymbols()
    v_hat, n_gamma, n_psi = s.wind_triad
    velocity = s.speed * v_hat
    acceleration = lie_derivative(velocity, s.state, s.rates)
    g_down, g_north = s.gravity
    gravity = -g_down * s.up + g_north * s.north
    force_v, force_gamma, force_psi = s.specific_force
    applied = force_v * v_hat + force_gamma * n_gamma + force_psi * n_psi
    coriolis = -2 * s.rotation.cross(velocity)
    centrifugal = -s.rotation.cross(s.rotation.cross(s.position))
    residual = acceleration - (applied + gravity + coriolis + centrifugal)
    return sp.Matrix([residual.dot(axis) for axis in (s.north, s.east, s.up)])


def _the_field_reduces_to_vinhs_equations() -> Any:
    """Spherical gravity and no rotation give the classical entry equations."""
    s = EntrySymbols()
    classical = EntrySymbols(planet=s.planet.spherical().non_rotating())
    gravity = s.planet.mu / s.radius**2
    curvature = s.speed**2 / s.radius
    force_v, force_gamma, force_psi = s.specific_force
    vinh = sp.Matrix(
        [
            force_v - gravity * sp.sin(s.gamma),
            (force_gamma - (gravity - curvature) * sp.cos(s.gamma)) / s.speed,
            (force_psi + curvature * sp.cos(s.gamma) ** 2 * sp.sin(s.heading) * sp.tan(s.latitude))
            / (s.speed * sp.cos(s.gamma)),
        ]
    )
    return sp.Matrix(classical.rates[3:]) - vinh


def _level_eastward_flight_turns_toward_the_equator() -> Any:
    r"""The sign test for the curvature term: :math:`\dot\psi = (V/r)\tan\phi`.

    Level, due east, no lateral force, classical planet. A positive rate in
    the northern hemisphere turns the velocity from east toward south, which
    is how a great circle bends.
    """
    s = EntrySymbols()
    classical = EntrySymbols(planet=s.planet.spherical().non_rotating())
    heading_rate = classical.rates[5].subs(
        {s.gamma: 0, s.heading: sp.pi / 2, s.force_psi: 0}
    )
    return heading_rate - s.speed / s.radius * sp.tan(s.latitude)


def _wind_triad_is_orthonormal_and_right_handed() -> Any:
    r"""The triad is orthonormal, and :math:`\hat n_\gamma` is up at level flight.

    Also :math:`\hat n_\gamma = \partial_\gamma\hat V` and
    :math:`\cos\gamma\,\hat n_\psi = \partial_\psi\hat V`, which is what makes the
    two normal components of the acceleration the rates of :math:`\gamma` and
    :math:`\psi`.
    """
    s = EntrySymbols()
    v_hat, n_gamma, n_psi = s.wind_triad_ned
    return [
        v_hat.dot(v_hat) - 1,
        n_gamma.dot(n_gamma) - 1,
        v_hat.dot(n_gamma),
        n_psi - sp.Matrix([-sp.sin(s.heading), sp.cos(s.heading), 0]),
        n_gamma.subs(s.gamma, 0) - sp.Matrix([0, 0, -1]),
        sp.diff(v_hat, s.gamma) - n_gamma,
        sp.diff(v_hat, s.heading) - sp.cos(s.gamma) * n_psi,
    ]


# -- energy and the glide balance ------------------------------------------------


def _energy_dissipates_at_the_drag_power() -> Any:
    r""":math:`\dot{\mathcal E} = (F_V/m)\,V`: only the along-track force does work.

    With oblateness and rotation retained, and for any lift and any lateral
    force. This is what makes the total dissipation a bounded, monotone
    quantity over every selection of an aerodynamic enclosure.
    """
    s = EntrySymbols()
    return lie_derivative(s.energy, s.state, s.rates) - s.force_v * s.speed


def _the_glide_balance_retains_rotation() -> Any:
    r"""The lift required for :math:`\dot\gamma = 0`, with every term of the field kept."""
    s = EntrySymbols()
    (required,) = sp.solve(sp.Eq(s.rates[4], 0), s.force_gamma)
    g_down, g_north = s.gravity
    omega = s.planet.rotation_rate
    stated = (
        (g_down - s.speed**2 / s.radius) * sp.cos(s.gamma)
        + g_north * sp.sin(s.gamma) * sp.cos(s.heading)
        - 2 * omega * s.speed * sp.cos(s.latitude) * sp.sin(s.heading)
        - omega**2
        * s.radius
        * sp.cos(s.latitude)
        * (
            sp.cos(s.gamma) * sp.cos(s.latitude)
            + sp.sin(s.gamma) * sp.cos(s.heading) * sp.sin(s.latitude)
        )
    )
    return required - stated


# -- attitude -------------------------------------------------------------------


def _quaternion_kinematics_rotate_the_body_frame() -> Any:
    r""":math:`\dot{\mathbf q} = \tfrac12\Omega(\boldsymbol\omega)\mathbf q` gives
    :math:`\dot{\mathbf R} = \mathbf R\,[\boldsymbol\omega\times]`, and preserves the norm.

    With the direction-cosine matrix and :math:`\Omega` exactly as the appendix
    displays them. The first entry is what fixes the convention: the rates are
    body-frame components and :math:`\mathbf R` carries body to reference.
    """
    q0, q1, q2, q3, w1, w2, w3 = sp.symbols("q0 q1 q2 q3 omega1 omega2 omega3", real=True)
    dcm = sp.Matrix(
        [
            [q0**2 + q1**2 - q2**2 - q3**2, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
            [2 * (q1 * q2 + q0 * q3), q0**2 - q1**2 + q2**2 - q3**2, 2 * (q2 * q3 - q0 * q1)],
            [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), q0**2 - q1**2 - q2**2 + q3**2],
        ]
    )
    omega = sp.Matrix(
        [[0, -w1, -w2, -w3], [w1, 0, w3, -w2], [w2, -w3, 0, w1], [w3, w2, -w1, 0]]
    )
    quaternion = sp.Matrix([q0, q1, q2, q3])
    rate = omega * quaternion / 2
    cross = sp.Matrix([[0, -w3, w2], [w3, 0, -w1], [-w2, w1, 0]])
    norm_squared = quaternion.dot(quaternion)
    return [
        lie_derivative(dcm, list(quaternion), list(rate)) - dcm * cross,
        dcm * dcm.T - norm_squared**2 * sp.eye(3),
        lie_derivative(norm_squared, list(quaternion), list(rate)),
    ]


# -- the phugoid ----------------------------------------------------------------


def _phugoid_frequency_and_damping() -> Any:
    r"""Frozen-coefficient spectrum on the glide, to leading order in :math:`H_s`.

    :math:`H_s\,a_1 \to g - V^2/r` gives the frequency. The pair's decay rate
    is :math:`(a_2 + \lambda_{\mathrm{slow}})/2` with
    :math:`\lambda_{\mathrm{slow}} \to -a_0/a_1`, and its limit is
    :math:`g/(V E_v)`: the scale height has dropped out.
    """
    s = GlideSymbols()
    a2, a1, a0 = s.characteristic_coefficients
    height = s.scale_height
    return [
        sp.limit(height * a1, height, 0) - s.net_gravity,
        sp.limit((a2 - a0 / a1) / 2, height, 0) - s.phugoid_damping_rate,
        sp.diff(s.phugoid_damping_rate, height),
    ]


def _residual_constant_is_twice_the_damping_ratio() -> Any:
    r""":math:`C_{\mathcal R}\sqrt\varepsilon = 2\zeta`, and the lag it measures.

    The second entry is the mechanism: the glide altitude descends at
    :math:`2 H_s g/(V E_v)` and the trajectory trails it by one phugoid
    response time :math:`1/\omega`, which in scale heights is
    :math:`C_{\mathcal R}\sqrt\varepsilon`.
    """
    s = GlideSymbols()
    lag = (2 * s.scale_height * s.phugoid_damping_rate) / sp.sqrt(s.phugoid_frequency_squared)
    return [
        s.residual_constant * sp.sqrt(s.epsilon) - 2 * s.damping_ratio,
        lag / s.scale_height - s.residual_constant * sp.sqrt(s.epsilon),
    ]


def _glide_descent_rate() -> Any:
    r""":math:`\dot h_{\mathrm{glide}}/H_s \to -2g/(V E_v)` as :math:`H_s \to 0`."""
    s = GlideSymbols()
    height = s.scale_height
    return sp.limit(s.glide_descent_rate / height, height, 0) + 2 * s.phugoid_damping_rate


def _slow_variation_amplitude_law() -> Any:
    r"""The amplitude of a slowly varying oscillator obeys :math:`\dot A/A = -p/2 - \dot Q/(4Q)`.

    For :math:`\ddot\eta + p\,\dot\eta + (Q/\delta^2)\,\eta = 0` with
    :math:`\delta` small, from

    the two leading orders of :math:`\eta = A\,e^{iS/\delta}`. Order
    :math:`\delta^{-2}` is the eikonal equation :math:`\dot S^2 = Q`; order
    :math:`\delta^{-1}` is the transport equation, and the claimed amplitude
    law is its solution.
    """
    t, delta = sp.symbols("t delta", positive=True)
    amplitude, phase, damping, stiffness = (sp.Function(n)(t) for n in ("A", "S", "p", "Q"))
    eta = amplitude * sp.exp(sp.I * phase / delta)
    equation = sp.diff(eta, t, 2) + damping * sp.diff(eta, t) + stiffness / delta**2 * eta
    orders = sp.expand(sp.simplify(equation / sp.exp(sp.I * phase / delta)))
    eikonal = {
        sp.Derivative(phase, (t, 2)): sp.diff(sp.sqrt(stiffness), t),
        sp.Derivative(phase, t): sp.sqrt(stiffness),
    }
    law = amplitude * (-damping / 2 - sp.diff(stiffness, t) / (4 * stiffness))
    transport = orders.coeff(delta, -1).subs(eikonal).subs(sp.Derivative(amplitude, t), law)
    return [orders.coeff(delta, -2).subs(eikonal), transport.doit()]


def _phugoid_amplitudes_decay_at_three_halves_and_one_half() -> Any:
    r"""With the glide's own drift, altitude decays at 3/2 of the frozen rate and angle at 1/2.

    Linearised about the descending glide, with :math:`\eta` the altitude
    deviation in scale heights,

    .. math::

        \dot\eta = a\,\gamma - c\,\eta, \qquad \dot\gamma = -b\,\eta,
        \qquad a = \frac{V}{H_s},\quad b = \frac{g - V^2/r}{V},\quad
        c = \frac{2g}{V E_v},

    and :math:`V` itself decaying at :math:`\dot V = -(g - V^2/r)/E_v`. The
    frozen eigenvalue gives both amplitudes the rate :math:`c/2 = g/(V E_v)`;
    the drift of the coefficients redistributes it. The frequency rises as the
    vehicle slows, so the altitude amplitude decays faster and the
    flight-path-angle amplitude slower, and their product -- the oscillator's
    action -- decays at exactly :math:`c`.
    """
    t = sp.Symbol("t", positive=True)
    gravity, radius, height, vertical = sp.symbols("g r H_s E_v", positive=True)
    speed = sp.Function("V")(t)
    net = gravity - speed**2 / radius
    a, b, c = speed / height, net / speed, 2 * gravity / (speed * vertical)

    def rate(expression: Any) -> Any:
        return sp.diff(expression, t).subs(sp.Derivative(speed, t), -net / vertical)

    # eta'' + (c - a'/a) eta' + (a b + ...) eta = 0, then the amplitude law.
    altitude_rate = (c - rate(a) / a) / 2 + rate(a * b) / (4 * a * b)
    # gamma ~ eta'/a, so its amplitude is the altitude's times sqrt(b/a).
    angle_rate = altitude_rate - (rate(b) / b - rate(a) / a) / 2
    frozen = gravity / (speed * vertical)
    return [
        altitude_rate - sp.Rational(3, 2) * frozen,
        angle_rate - sp.Rational(1, 2) * frozen,
        altitude_rate + angle_rate - c,
    ]


def _first_order_pair_is_the_damped_oscillator() -> Any:
    r"""The reduction the amplitude law is applied to.

    :math:`\dot\eta = a\gamma - c\eta`, :math:`\dot\gamma = -b\eta` with
    time-dependent coefficients is
    :math:`\ddot\eta + (c - \dot a/a)\dot\eta + (ab + \dot c - c\,\dot a/a)\eta = 0`.
    """
    t = sp.Symbol("t", positive=True)
    a, b, c, eta = (sp.Function(n)(t) for n in ("a", "b", "c", "eta"))
    gamma = (sp.diff(eta, t) + c * eta) / a
    second_order = (
        sp.diff(eta, t, 2)
        + (c - sp.diff(a, t) / a) * sp.diff(eta, t)
        + (a * b + sp.diff(c, t) - c * sp.diff(a, t) / a) * eta
    )
    return a * (sp.diff(gamma, t) + b * eta) - second_order


def _a_pass_dissipates_in_proportion_to_its_turn() -> Any:
    r"""Where the aerodynamic force dominates, :math:`d\ln V/d\gamma = -1/E_v`.

    The ratio of the speed and flight-path-angle rates, in the limit of small
    ballistic coefficient -- gravity and curvature negligible against lift and
    drag, which is the inner region of a pass. Integrated across a pass that
    turns the flight path through :math:`\Delta\gamma` it gives the classical
    :math:`V_{\mathrm{out}}/V_{\mathrm{in}} = e^{-\Delta\gamma/E_v}`, so the
    energy a pass dissipates is fixed by how far it turns the velocity. Lift
    cannot be had without drag, and that is what lets the dissipation stand
    in for everything a pass does to the state.
    """
    s = GlideSymbols()
    ratio = s.rates[1] / (s.speed * s.rates[2])
    return sp.limit(ratio, s.ballistic_coefficient, 0, "+") + 1 / s.vertical_lift_to_drag


def _the_depth_of_a_pass_is_set_by_its_entry_angle() -> Any:
    r"""Along a pass, :math:`\cos\gamma - E_v H_s\rho/(2\beta)` changes only through gravity.

    Its rate along the planar field is exactly
    :math:`(g - V^2/r)\sin\gamma\cos\gamma/V`: the lift and the density
    gradient cancel. Where the aerodynamic force dominates the quantity is
    therefore conserved, and a pass entered from thin air at an angle
    :math:`\gamma_e` levels out at the density
    :math:`\rho_b = 2\beta(1-\cos\gamma_e)/(E_v H_s)`. The vertical lift
    acceleration there is :math:`V^2(1-\cos\gamma_e)/H_s`, about
    :math:`V^2\gamma_e^2/(2H_s)`: this is how the load in a pass comes to
    scale as :math:`\gamma_e^2/\varepsilon` against gravity, and why holding
    the load fixed as the layer thins forces :math:`\gamma_e \sim
    \sqrt\varepsilon`.
    """
    s = GlideSymbols()
    density = s.reference_density * sp.exp(-(s.radius - s.body_radius) / s.scale_height)
    invariant = sp.cos(s.gamma) - s.vertical_lift_to_drag * s.scale_height * density / (
        2 * s.ballistic_coefficient
    )
    rate = lie_derivative(invariant, s.state, list(s.rates))
    entry_angle = sp.Symbol("gamma_e", positive=True)
    bottom_density = (
        2 * s.ballistic_coefficient * (1 - sp.cos(entry_angle))
        / (s.vertical_lift_to_drag * s.scale_height)
    )
    lift_at_the_bottom = (
        s.vertical_lift_to_drag * bottom_density * s.speed**2 / (2 * s.ballistic_coefficient)
    )
    return [
        rate - s.net_gravity * sp.sin(s.gamma) * sp.cos(s.gamma) / s.speed,
        lift_at_the_bottom - s.speed**2 * (1 - sp.cos(entry_angle)) / s.scale_height,
    ]


def _glide_phugoid_and_skip_are_one_oscillator() -> Any:
    r"""Level flight :math:`z` scale heights off the glide accelerates at :math:`G(e^{-z}-1)`.

    With :math:`G = g - V^2/r` and the ballistic coefficient that puts the
    glide at :math:`(r, V)`, the vertical acceleration :math:`V\dot\gamma` at
    level flight and radius :math:`r + H_s z` tends to :math:`G(e^{-z} - 1)`
    as :math:`H_s \to 0` with :math:`z` fixed: lift falls off exponentially
    with altitude and what it has to balance does not. Since
    :math:`\ddot h = V\dot\gamma` at level flight, the altitude in scale
    heights obeys :math:`\ddot z = k\,(e^{-z} - 1)`, :math:`k = G/H_s`, to
    leading order, at frozen speed and with the damping left out.

    The remaining entries are what that equation says. Its energy
    :math:`\tfrac12\dot z^2 + k\,(e^{-z} + z)` is conserved; the curvature of
    the potential at its minimum is :math:`k`, the phugoid frequency squared,
    so the phugoid is its small oscillation; far above the glide the
    acceleration tends to :math:`-k`, a ballistic arc under :math:`-G`, so a
    skip is its large one; and an oscillation that crosses the glide altitude
    at a flight-path angle :math:`\gamma_0` turns where
    :math:`e^{-z} + z - 1 = V^2\gamma_0^2/(2 G H_s)`, which is of order
    :math:`\gamma_0^2/\varepsilon`.
    """
    s = GlideSymbols()
    height = s.scale_height
    z, z_rate, angle = sp.symbols("z z_dot gamma_0", real=True)
    level = (s.speed * s.rates[2]).subs(s.gamma, 0)
    displaced = level.subs(s.radius, s.radius + height * z).subs(
        s.ballistic_coefficient, s.glide_ballistic_coefficient
    )
    stiffness = s.phugoid_frequency_squared
    potential = stiffness * (sp.exp(-z) + z)
    restoring = stiffness * (sp.exp(-z) - 1)
    energy = z_rate**2 / 2 + potential
    crossing_rate = s.speed * angle / height
    return [
        sp.limit(sp.simplify(displaced), height, 0, "+") - s.net_gravity * (sp.exp(-z) - 1),
        lie_derivative(energy, (z, z_rate), (z_rate, restoring)),
        sp.diff(potential, z, 2).subs(z, 0) - stiffness,
        sp.limit(restoring, z, sp.oo) + stiffness,
        crossing_rate**2 / (2 * stiffness)
        - s.speed**2 * angle**2 / (2 * s.net_gravity * height),
    ]


# -- the corridor, and wrapping ---------------------------------------------------


def _corridor_limits_are_monotone_in_altitude_and_speed() -> Any:
    r"""Dynamic pressure and stagnation heating fall with altitude and rise with speed.

    Stated through logarithmic derivatives, whose signs are then fixed:
    :math:`\partial_h\ln\bar q = -1/H_s`, :math:`\partial_V\ln\bar q = 2/V`,
    and for the Sutton--Graves form :math:`k\sqrt{\rho/R_n}\,V^3`,
    :math:`-1/(2H_s)` and :math:`3/V`. Each limit is therefore a curve
    :math:`h \ge h_{\min}(V)`, not a face of a box -- and contracting a box
    against a constraint monotone in each variable returns its exact hull.
    """
    altitude = sp.Symbol("h", real=True)
    speed, height, density, nose, constant = sp.symbols("V H_s rho_0 R_n k", positive=True)
    rho = density * sp.exp(-altitude / height)
    pressure = rho * speed**2 / 2
    heating = constant * sp.sqrt(rho / nose) * speed**3
    return [
        sp.diff(pressure, altitude) / pressure + 1 / height,
        sp.diff(pressure, speed) / pressure - 2 / speed,
        sp.diff(heating, altitude) / heating + 1 / (2 * height),
        sp.diff(heating, speed) / heating - 3 / speed,
    ]


def _interval_euler_wraps_a_rotation_by_exp_two_pi() -> Any:
    r"""Re-boxing a rotation every step inflates a box by :math:`e^{2\pi}` per turn.

    Interval Euler on :math:`\dot x = y`, :math:`\dot y = -x` maps the widths
    by :math:`\begin{psmallmatrix}1 & h\\ h & 1\end{psmallmatrix}`, whose
    largest eigenvalue is :math:`1 + h`. Over one revolution, :math:`2\pi/h`
    steps, the growth is :math:`(1+h)^{2\pi/h}`, which *increases* to
    :math:`e^{2\pi} \approx 535` as the step is refined. Step size does not
    control wrapping; the representation of the set does.
    """
    step = sp.Symbol("h", positive=True)
    widths = sp.Matrix([[1, step], [step, 1]])
    growth = (1 + step) ** (2 * sp.pi / step)
    return [
        max(widths.eigenvals(), key=lambda value: value.subs(step, 1)) - (1 + step),
        sp.limit(growth, step, 0) - sp.exp(2 * sp.pi),
    ]


DERIVATIONS: tuple[Derivation, ...] = (
    Derivation(
        "gravity-gradient",
        "g_D and g_N are the gradient of the J2 potential",
        "apx:dyn, eq:gD, eq:gN",
        _gravity_is_the_gradient_of_the_potential,
    ),
    Derivation(
        "gravity-cartesian",
        "the spherical gravity components are the Cartesian J2 field, projected",
        "apx:dyn, rem:gravity-spherical",
        _spherical_gravity_is_the_cartesian_j2_field,
    ),
    Derivation(
        "gravity-sign-change",
        "the radial J2 correction vanishes at arcsin(1/sqrt 3) and is -2x as large at the pole",
        "apx:dyn, rem:j2-glide",
        _radial_oblateness_changes_sign_at_35_degrees,
    ),
    Derivation(
        "kinematics",
        "the position rates are the planet-relative velocity",
        "apx:dyn, eq:rdot to eq:phidot",
        _position_rates_are_the_velocity,
    ),
    Derivation(
        "newton-rotating-frame",
        "the velocity rates are Newton's law in the rotating frame: J2, Coriolis, centrifugal",
        "apx:dyn, eq:Vdot to eq:psidot",
        _the_field_is_newtons_law_in_the_rotating_frame,
    ),
    Derivation(
        "vinh-reduction",
        "spherical gravity and no rotation recover the classical entry equations",
        "apx:dyn, rem:vinh-reduction",
        _the_field_reduces_to_vinhs_equations,
    ),
    Derivation(
        "great-circle",
        "level eastward flight has heading rate (V/r) tan(phi)",
        "apx:dyn, rem:great-circle",
        _level_eastward_flight_turns_toward_the_equator,
    ),
    Derivation(
        "wind-triad",
        "the wind triad is orthonormal, right-handed, and n_gamma is up at level flight",
        "apx:dyn, eq:Vhat to eq:npsi",
        _wind_triad_is_orthonormal_and_right_handed,
    ),
    Derivation(
        "energy-dissipation",
        "the Jacobi energy changes at the along-track specific power, dE/dt = (F_V/m) V",
        "sec:profile; apx:time, eq:dissipation",
        _energy_dissipates_at_the_drag_power,
    ),
    Derivation(
        "glide-balance",
        "the glide condition with rotation and oblateness retained",
        "apx:dyn, eq:glide-balance, eq:glide-full",
        _the_glide_balance_retains_rotation,
    ),
    Derivation(
        "quaternion-kinematics",
        "the quaternion rate rotates the body frame at the body rate and preserves the norm",
        "apx:dyn, eq:DCM, eq:qdot, eq:Omega",
        _quaternion_kinematics_rotate_the_body_frame,
    ),
    Derivation(
        "phugoid-spectrum",
        "omega^2 -> (g - V^2/r)/H_s and the damping rate -> g/(V E_v), independent of H_s",
        "apx:residual, eq:phugoid, eq:zeta",
        _phugoid_frequency_and_damping,
        kind="limit",
    ),
    Derivation(
        "residual-constant",
        "C_R sqrt(eps) = 2 zeta, and it is the lag behind the descending glide in scale heights",
        "apx:residual, eq:CR",
        _residual_constant_is_twice_the_damping_ratio,
    ),
    Derivation(
        "glide-descent",
        "the glide altitude descends at 2 H_s g/(V E_v)",
        "apx:residual, eq:glide-descent",
        _glide_descent_rate,
        kind="limit",
    ),
    Derivation(
        "amplitude-law",
        "a slowly varying oscillator has dA/A = -(p/2 + Q'/(4Q)) dt",
        "apx:residual, eq:decay",
        _slow_variation_amplitude_law,
        kind="limit",
    ),
    Derivation(
        "oscillator-reduction",
        "the linearised altitude/angle pair is a damped oscillator with the stated coefficients",
        "apx:residual, eq:decay",
        _first_order_pair_is_the_damped_oscillator,
    ),
    Derivation(
        "phugoid-decay",
        "altitude and flight-path-angle amplitudes decay at 3/2 and 1/2 of g/(V E_v)",
        "apx:residual, eq:decay",
        _phugoid_amplitudes_decay_at_three_halves_and_one_half,
        kind="limit",
    ),
    Derivation(
        "pass-turning",
        "in a pass, d ln V / d gamma = -1/E_v: dissipation is proportional to the turn",
        "apx:residual, eq:turning; apx:time, lem:impulse",
        _a_pass_dissipates_in_proportion_to_its_turn,
        kind="limit",
    ),
    Derivation(
        "pass-depth",
        "cos(gamma) - E_v H_s rho/(2 beta) changes only through gravity, so a pass entered at "
        "gamma_e bottoms out where the lift is V^2 (1 - cos gamma_e)/H_s",
        "sec:profile; apx:residual, eq:pass-depth",
        _the_depth_of_a_pass_is_set_by_its_entry_angle,
    ),
    Derivation(
        "one-oscillator",
        "z scale heights off the glide, level flight accelerates at G(e^-z - 1): glide, phugoid "
        "and skip are the equilibrium, small and large oscillations of one equation",
        "sec:bound, eq:oscillator-body; apx:residual, eq:oscillator",
        _glide_phugoid_and_skip_are_one_oscillator,
        kind="limit",
    ),
    Derivation(
        "corridor-monotone",
        "dynamic-pressure and heating limits are monotone in altitude and in speed",
        "sec:bound; apx:dyn, sec:coord-rationale",
        _corridor_limits_are_monotone_in_altitude_and_speed,
    ),
    Derivation(
        "wrapping",
        "naive interval propagation inflates a rotating box by e^(2 pi) per turn as h -> 0",
        "sec:provenance; apx:dyn, sec:summary-table",
        _interval_euler_wraps_a_rotation_by_exp_two_pi,
        kind="limit",
    ),
)


def check(derivation: Derivation) -> DerivationResult:
    """Reduce one residual and report whether it reached zero."""
    start = time.perf_counter()
    residual = derivation.residual()
    passed = is_identically_zero(residual)
    remainder = "" if passed else _describe(residual)
    return DerivationResult(derivation, passed, time.perf_counter() - start, remainder)


def check_all(derivations: Sequence[Derivation] = DERIVATIONS) -> list[DerivationResult]:
    """Reduce every residual in the registry."""
    return [check(derivation) for derivation in derivations]


def _describe(residual: Any) -> str:
    """The entries that did not reduce, compactly, for a failure message."""
    entries: list[Any] = []
    stack = [residual]
    while stack:
        item = stack.pop()
        if isinstance(item, sp.MatrixBase):
            entries.extend(item)
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        else:
            entries.append(item)
    left = [reduce(entry) for entry in entries]
    return "; ".join(str(entry) for entry in left if entry != 0)
