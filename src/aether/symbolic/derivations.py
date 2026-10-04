r"""The derivations the manuscript relies on, each as an identity to be reduced to zero.

A derivation checked by reading is checked once, by one reader, against
whatever was on the page that day. Each entry below instead states a claim the
proposal makes as a *residual* -- left side minus right side, built from the
field of record -- and :func:`check` reduces it by exact rewriting. Zero is a
proof; anything else is printed, and is either a wrong claim or a claim stated
so that SymPy cannot see it, which amounts to the same instruction: fix it.

The registry is the contract with the manuscript. ``where`` names the place
the claim is used, so that a changed equation has a named check to fail, and a
check that fails names the paragraph to reread. The contract runs both ways:
every displayed equation, lemma, proposition and remark of the manuscript is
named by an entry here or listed in :data:`STATED_NOT_DERIVED` with the reason
it is not algebra, and the tests fail when a label appears that is neither.

It spans the whole of the dynamics appendix, and the entries are of three
standings. The *translational* ones are checked against the field of record,
the function the enclosures and the simulations evaluate. The *attitude* ones
-- the quaternion, the transport of the planet's rotation, Euler's equations,
the passage from the body force to the wind axes -- are checked against the
appendix's own statement of them, in :mod:`aether.symbolic.attitude`, because
no field written for every arithmetic carries the attitude. The *heating
correlation and the structural oscillator* are constitutive laws: there is
nothing to derive them from, and what is checked is what follows from them.

Two blocks that were once constitutive are derived. The heat shield is two
conservation laws in depth, and the ``ablation-*`` entries take them from
their conservation form to the equation a thermal-response code integrates,
across the receding surface, and onto the cells the coupled field carries
(:mod:`aether.symbolic.thermal`); what stays constitutive there is the
material and the chemistry of the surface, and is named. And the air's force
on the structural modes: the ``piston-*`` entries take it from the equations
of one-dimensional gas dynamics to the generalized forces
(:mod:`aether.symbolic.piston`). What those do not derive -- that a slab of
fluid in a fast stream moves as a one-dimensional flow at all, beyond the
linearized case that can be solved exactly -- is listed in
:data:`STATED_NOT_DERIVED` with its sources.

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

from aether.certification.rotating_field import (
    exponential_density,
    point_mass_field,
    rotating_entry_field,
)
from aether.symbolic.atmosphere import StandardLayer, geopotential, gravity
from aether.symbolic.attitude import AttitudeSymbols, skew
from aether.symbolic.coupled import CoupledSymbols
from aether.symbolic.entry import EntrySymbols
from aether.symbolic.figure import FigureSymbols
from aether.symbolic.glide import GlideSymbols
from aether.symbolic.ops import SYMPY_OPS, is_identically_zero, lie_derivative, reduce
from aether.symbolic.piston import (
    IsentropicFlow,
    PistonSymbols,
    StripSymbols,
    WavyWall,
    downwash,
)
from aether.symbolic.thermal import (
    CharringCells,
    CharringSlab,
    SurfaceBalance,
    landau_coordinate,
)

__all__ = [
    "DERIVATIONS",
    "STATED_NOT_DERIVED",
    "Derivation",
    "DerivationResult",
    "check",
    "check_all",
]


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
    opposite sign, at the pole. The last entry is the slip the manuscript
    warns of: carrying the equatorial Cartesian factor :math:`1 - 5\sin^2\phi`
    into the radial component moves the sign change to
    :math:`\arcsin(1/\sqrt5)`.
    """
    phi = sp.Symbol("phi", real=True)
    factor = 1 - 3 * sp.sin(phi) ** 2
    mistaken = 1 - 5 * sp.sin(phi) ** 2
    return sp.Matrix(
        [
            factor.subs(phi, sp.asin(1 / sp.sqrt(3))),
            factor.subs(phi, 0) - 1,
            factor.subs(phi, sp.pi / 2) + 2,
            mistaken.subs(phi, sp.asin(1 / sp.sqrt(5))),
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


def _the_dissipation_density_has_the_mass_of_the_dissipation() -> Any:
    r""":math:`\int\sqrt\varepsilon\,P(\sqrt\varepsilon\,\tau)\,d\tau = \int P\,dt`.

    The density the manuscript reads the loss of compactness on is the drag
    power on the fast-time axis :math:`\tau = t/\sqrt\varepsilon`. The change
    of variable carries the horizon :math:`[0, T]` to
    :math:`[0, T/\sqrt\varepsilon]` and leaves the mass alone, so dividing by
    the total dissipation gives a density of unit mass for every
    :math:`\varepsilon`: the one hypothesis of the lemma applied to it.
    """
    epsilon, horizon = sp.symbols("epsilon T", positive=True)
    fast, slow = sp.symbols("tau t", real=True)
    power = sp.Function("P")
    on_the_fast_axis = sp.Integral(
        sp.sqrt(epsilon) * power(sp.sqrt(epsilon) * fast), (fast, 0, horizon / sp.sqrt(epsilon))
    )
    return on_the_fast_axis.transform(fast, (slow / sp.sqrt(epsilon), slow)) - sp.Integral(
        power(slow), (slow, 0, horizon)
    )


def _glide_balance_as_stated(s: EntrySymbols) -> Any:
    """The right-hand side of the manuscript's glide balance, typed as it is displayed."""
    g_down, g_north = s.gravity
    omega = s.planet.rotation_rate
    return (
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


def _the_glide_balance_retains_rotation() -> Any:
    r"""The lift required for :math:`\dot\gamma = 0`, with every term of the field kept.

    This is the *translational* balance: the normal specific force that holds
    the flight-path angle, whatever supplies it. That it is supplied by the
    attitude, and what the balance then says, is ``glide-balance-attitude``.
    """
    s = EntrySymbols()
    (required,) = sp.solve(sp.Eq(s.rates[4], 0), s.force_gamma)
    return required - _glide_balance_as_stated(s)


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


# -- frames, rotation, and the passage from attitude to force -----------------------


def _the_local_frame_is_the_one_the_field_was_derived_on() -> Any:
    r"""The north-east-down basis the appendix displays is the basis of the derivation.

    Its three vectors are the ones the translational field was derived on,
    they are orthonormal, and north cross east is down. The last is what
    fixes the handedness of every rotation that follows.
    """
    a = AttitudeSymbols()
    s = a.entry
    frame = a.local_frame
    north, east, down = frame[:, 0], frame[:, 1], frame[:, 2]
    return [
        north - s.north,
        east - s.east,
        down + s.up,
        frame.T * frame - sp.eye(3),
        north.cross(east) - down,
    ]


def _the_direction_cosine_matrix_is_a_proper_rotation() -> Any:
    r""":math:`\det\mathbf R = \lVert\mathbf q\rVert^6`, and :math:`\mathbf R` carries cross
    products.

    On the unit sphere the determinant is :math:`+1`: a rotation and not a
    reflection. The second entry is the rule every change of frame uses,
    :math:`\mathbf R^{\mathsf T}[\mathbf a\times]\mathbf R =
    [(\mathbf R^{\mathsf T}\mathbf a)\times]`, with the power of the norm it
    carries off the sphere made explicit. The third is that the body-to-local
    matrix, the local frame's transpose times :math:`\mathbf R`, is orthogonal
    with it.
    """
    a = AttitudeSymbols()
    arbitrary = sp.Matrix(sp.symbols("a_1 a_2 a_3", real=True))
    # R_LB^T R_LB = R^T (R_EL R_EL^T) R. The middle factor is reduced first: it
    # is trigonometric and small, and what is left is a polynomial in q.
    frame = (a.local_frame * a.local_frame.T).applyfunc(sp.trigsimp)
    return [
        a.dcm.det() - a.norm_squared**3,
        a.dcm.T * skew(arbitrary) * a.dcm - a.norm_squared * skew(a.dcm.T * arbitrary),
        a.dcm.T * frame * a.dcm - a.norm_squared**2 * sp.eye(3),
    ]


def _the_planets_rotation_is_carried_into_the_body_rate() -> Any:
    r"""Driving the quaternion at :math:`\boldsymbol\omega -
    \mathbf R^{\mathsf T}\boldsymbol\omega_E` turns the body at :math:`\boldsymbol\omega`
    in inertial space.

    The quaternion is taken against the rotating frame, whose own rate in
    inertial space is :math:`\boldsymbol\omega_E`. The inertial rate of the
    body frame, in body components, is the rate relative to the rotating
    frame plus the planet's rate carried into body axes,
    :math:`\mathbf R^{\mathsf T}[\boldsymbol\omega_E\times]\mathbf R +
    [\boldsymbol\omega_{\mathcal B\mathcal E}\times]`. The first entry says
    that this is :math:`[\boldsymbol\omega\times]` on the unit sphere, with
    what is left off it shown. The other two say that the kinematics driven
    by the relative rate turn :math:`\mathbf R` at that rate and keep the norm.
    """
    a = AttitudeSymbols()
    planet, relative = a.entry.rotation, a.relative_rate
    quaternion = list(a.quaternion)
    rate = list(a.quaternion_rate(relative))
    return [
        a.dcm.T * skew(planet) * a.dcm
        + skew(relative)
        - skew(a.body_rate)
        - (a.norm_squared - 1) * skew(a.dcm.T * planet),
        lie_derivative(a.dcm, quaternion, rate) - a.dcm * skew(relative),
        lie_derivative(a.norm_squared, quaternion, rate),
    ]


def _eulers_equations_with_a_changing_inertia() -> Any:
    r"""Which :math:`\dot{\mathbf I}` belongs in Euler's equations, and which does not.

    Angular momentum about the mass centre, in inertial components, is
    :math:`\mathbf R\,\mathbf I\,\boldsymbol\omega` when the body axes carry no
    angular momentum of the deformation (mean axes), with
    :math:`\dot{\mathbf R} = \mathbf R[\boldsymbol\omega\times]`.

    *A closed body that deforms.* Its angular momentum changes at the moment
    applied, and that is the manuscript's equation,
    :math:`\mathbf I\dot{\boldsymbol\omega} = \mathbf M -
    \boldsymbol\omega\times\mathbf I\boldsymbol\omega -
    \dot{\mathbf I}\boldsymbol\omega`. The first entry is the equivalence.

    *A body that loses mass from its surface.* An element of mass
    :math:`\mu` at body position :math:`\boldsymbol\rho`, leaving with the
    velocity of the surface it leaves, takes the angular momentum
    :math:`\boldsymbol\rho\times(\boldsymbol\omega\times\boldsymbol\rho)` per
    unit mass with it. The inertia falls at
    :math:`\dot\mu\,(\lVert\boldsymbol\rho\rVert^2\mathbf 1 -
    \boldsymbol\rho\boldsymbol\rho^{\mathsf T})`, and
    :math:`\dot{\mathbf I}\boldsymbol\omega` is exactly what the departing
    mass carries off: the two cancel, and the balance is
    :math:`\mathbf I\dot{\boldsymbol\omega} = \mathbf M -
    \boldsymbol\omega\times\mathbf I\boldsymbol\omega` with no
    :math:`\dot{\mathbf I}` term at all. This is the cancellation of the
    inertia-variation moment against the jet-damping moment in the general
    equation for variable-mass bodies, for the case in which mass leaves from
    where it sat. So only the deformation's share of :math:`\dot{\mathbf I}`
    appears, and recession contributes none.
    """
    t = sp.Symbol("t", real=True)
    rotation = sp.Matrix(3, 3, lambda i, j: sp.Function(f"R_{i}{j}")(t))
    omega = sp.Matrix([sp.Function(f"omega_{i}")(t) for i in (1, 2, 3)])
    moment = sp.Matrix(sp.symbols("M_1 M_2 M_3", real=True))
    turning = {
        sp.Derivative(rotation[i, j], t): (rotation * skew(omega))[i, j]
        for i in range(3)
        for j in range(3)
    }

    def momentum_rate(inertia: Any) -> Any:
        return sp.diff(rotation * inertia * omega, t).subs(turning)

    names = ("xx", "yy", "zz", "xy", "xz", "yz")
    xx, yy, zz, xy, xz, yz = (sp.Function(f"I_{n}")(t) for n in names)
    deforming = sp.Matrix([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    closed = (
        momentum_rate(deforming)
        - rotation * moment
        - rotation
        * (
            deforming * sp.diff(omega, t)
            - (moment - omega.cross(deforming * omega) - sp.diff(deforming, t) * omega)
        )
    )

    rigid = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"B_{min(i, j)}{max(i, j)}", real=True))
    position = sp.Matrix(sp.symbols("rho_1 rho_2 rho_3", real=True))
    element = sp.Function("mu")(t)
    shedding = rigid + element * (position.dot(position) * sp.eye(3) - position * position.T)
    carried_off = sp.diff(element, t) * rotation * position.cross(omega.cross(position))
    open_system = (
        momentum_rate(shedding)
        - carried_off
        - rotation * moment
        - rotation * (shedding * sp.diff(omega, t) - (moment - omega.cross(shedding * omega)))
    )
    return [closed, open_system]


def _the_body_force_on_the_wind_axes() -> Any:
    r"""At incidence :math:`\alpha` and bank :math:`\sigma`:
    :math:`F_V = -D`, :math:`F_\gamma = L\cos\sigma`, :math:`F_\psi = L\sin\sigma`.

    The body force :math:`q_{\mathrm{dyn}}S\,(-C_A, C_Y, -C_N)` is rotated to
    local axes by the attitude and projected on the wind triad, as the
    appendix sets out, for the attitude that has no sideslip. With
    :math:`D = q_{\mathrm{dyn}}S\,(C_A\cos\alpha + C_N\sin\alpha)` and
    :math:`L = q_{\mathrm{dyn}}S\,(C_N\cos\alpha - C_A\sin\alpha)` the three
    components are the point-mass ones, and a side force :math:`Y` enters the
    two normal components as :math:`-Y\sin\sigma` and :math:`+Y\cos\sigma`.
    Without it :math:`(F_\gamma, F_\psi) = L(\cos\sigma, \sin\sigma)`, which
    is what makes :math:`\operatorname{atan2}(F_\psi, F_\gamma)` the bank
    angle. The first two entries say the attitude is a rotation.
    """
    a = AttitudeSymbols()
    attitude = a.banked_attitude.subs(a.sideslip, 0)
    along, normal, lateral = a.wind_force(attitude)
    side = a.pressure_area * a.coefficients[1]
    return [
        attitude.T * attitude - sp.eye(3),
        attitude.det() - 1,
        along + a.drag,
        normal - (a.lift * sp.cos(a.bank) - side * sp.sin(a.bank)),
        lateral - (a.lift * sp.sin(a.bank) + side * sp.cos(a.bank)),
    ]


def _incidence_angles_are_read_off_one_vector() -> Any:
    r"""The body-frame velocity is :math:`V(\cos\alpha\cos\beta,\ \sin\beta,\ \sin\alpha\cos\beta)`.

    So :math:`\operatorname{atan2}(V_z, V_x)` and :math:`\arcsin(V_y/V)`
    return the angle of attack and the sideslip the attitude was set at, both
    referred to the velocity. Referred to the freestream instead, which is
    minus the velocity, purely axial flight reads :math:`\pi` and not zero,
    and the sideslip changes sign: the last three entries are the
    manuscript's warning about mixing the two.
    """
    a = AttitudeSymbols()
    forward, right, down = a.body_velocity(a.banked_attitude)
    speed = a.entry.speed
    axial, lateral = sp.Symbol("u", positive=True), sp.Symbol("v", real=True)
    return [
        forward - speed * sp.cos(a.alpha) * sp.cos(a.sideslip),
        right - speed * sp.sin(a.sideslip),
        down - speed * sp.sin(a.alpha) * sp.cos(a.sideslip),
        sp.atan2(0, axial),
        sp.atan2(0, -axial) - sp.pi,
        sp.asin(-lateral / axial) + sp.asin(lateral / axial),
    ]


def _the_glide_balance_constrains_the_attitude() -> Any:
    r"""The six-degree-of-freedom glide balance, and its point-mass and classical forms.

    With the normal force supplied by the attitude, the flight-path angle
    holds exactly when that force equals the right-hand side of the
    translational balance: the first entry is
    :math:`V\dot\gamma = F_\gamma/m - (\text{that right-hand side})` for the
    field driven through the attitude. At incidence and bank, without side
    force, :math:`F_\gamma = L\cos\sigma`. And on a spherical, non-rotating
    planet at level flight the balance is :math:`\Lambda = 1` with
    :math:`\Lambda = L\cos\sigma/[m(g - V^2/r)]`: the third entry is
    :math:`V\dot\gamma = (g - V^2/r)(\Lambda - 1)`.
    """
    a = AttitudeSymbols()
    s = a.entry
    mass = sp.Symbol("m", positive=True)
    attitude = a.banked_attitude.subs(a.sideslip, 0)
    force = [component.subs(a.coefficients[1], 0) for component in a.wind_force(attitude)]
    specific = [component / mass for component in force]
    driven = rotating_entry_field(s.state, specific, SYMPY_OPS, planet=s.planet)
    classical = s.planet.spherical().non_rotating()
    level = rotating_entry_field(s.state, specific, SYMPY_OPS, planet=classical)[4].subs(
        s.gamma, 0
    )
    net_gravity = s.planet.mu / s.radius**2 - s.speed**2 / s.radius
    loading = a.lift * sp.cos(a.bank) / (mass * net_gravity)
    return [
        s.speed * driven[4] - (specific[1] - _glide_balance_as_stated(s)),
        force[1] - a.lift * sp.cos(a.bank),
        s.speed * level - net_gravity * (loading - 1),
    ]


def _in_trim_the_field_is_the_point_mass_field() -> Any:
    r"""Driven through an attitude at incidence and bank, the field is the point-mass one.

    With the ballistic coefficient :math:`m/(C_D S)` and the lift-to-drag
    ratio :math:`C_L/C_D` that the body-axis coefficients give at the angle
    of attack, the six translational rates are those of the point-mass field
    of record, rotation and oblateness included. This is the sense in which
    the point-mass model is the six-degree-of-freedom one in trim.
    """
    a = AttitudeSymbols()
    s = a.entry
    mass, area = sp.symbols("m S", positive=True)
    density = exponential_density(*sp.symbols("rho_0 H_s", positive=True))
    pressure_area = density(s.altitude, SYMPY_OPS) * s.speed**2 * area / 2
    attitude = a.banked_attitude.subs(a.sideslip, 0)
    force = [
        component.subs(a.coefficients[1], 0).subs(a.pressure_area, pressure_area) / mass
        for component in a.wind_force(attitude)
    ]
    axial, _, normal = a.coefficients
    drag_coefficient = axial * sp.cos(a.alpha) + normal * sp.sin(a.alpha)
    lift_coefficient = normal * sp.cos(a.alpha) - axial * sp.sin(a.alpha)
    point_mass = point_mass_field(
        mass / (drag_coefficient * area),
        lift_coefficient / drag_coefficient,
        density=density,
        planet=s.planet,
    )
    driven = rotating_entry_field(s.state, force, SYMPY_OPS, planet=s.planet)
    return sp.Matrix(driven) - sp.Matrix(point_mass(s.state, a.bank, SYMPY_OPS))


def _the_aerodynamic_impulse_is_bounded_by_the_dissipation() -> Any:
    r""":math:`C_a^2P^2 - \lvert\mathbf a\rvert^2` is a sum of two terms that cannot be negative.

    The aerodynamic acceleration has the squared magnitude
    :math:`a_V^2 + a_\gamma^2 + a_\psi^2` on the orthonormal wind triad, the
    drag power is :math:`P = -a_V V`, and
    :math:`C_a = \sqrt{1 + E_{\max}^2}/V_{\min}`. The difference is

    .. math::

        (1 + E_{\max}^2)\,a_V^2\,\frac{V^2 - V_{\min}^2}{V_{\min}^2}
        \;+\; \bigl(E_{\max}^2 a_V^2 - a_\gamma^2 - a_\psi^2\bigr),

    and each term is nonnegative on the corridor under the dissipative
    enclosure: the first because :math:`V \ge V_{\min}`, the second because
    lift is bounded by :math:`E_{\max}` times drag.
    """
    s = EntrySymbols()
    along, normal, lateral = s.specific_force
    lowest, ratio = sp.symbols("V_min E_max", positive=True)
    power = -along * s.speed
    bound = sp.sqrt(1 + ratio**2) / lowest
    return (
        bound**2 * power**2
        - (along**2 + normal**2 + lateral**2)
        - (
            (1 + ratio**2) * along**2 * (s.speed**2 - lowest**2) / lowest**2
            + (ratio**2 * along**2 - normal**2 - lateral**2)
        )
    )


def _oblateness_bends_the_heading_with_the_sign_of_the_hemisphere() -> Any:
    r"""The northward :math:`J_2` gravity is :math:`\propto -\sin 2\phi`; the radial part is even.

    :math:`g_N = -\tfrac32 J_2 (\mu/r^2)(R_e/r)^2\sin 2\phi`: odd in latitude,
    zero at the equator and at the poles, and largest at :math:`45^\circ`,
    where its size is :math:`\tfrac32 J_2 (R_e/r)^2` of the central term. The
    radial correction is that same fraction times :math:`1 - 3\sin^2\phi`,
    which is even. So the heading drift keeps one sign within a hemisphere
    and the radial correction does not change with the hemisphere: the two
    cannot be bounded by one worst-case latitude.
    """
    s = EntrySymbols()
    planet, phi = s.planet, s.latitude
    g_down, g_north = s.gravity
    central = planet.mu / s.radius**2
    fraction = sp.Rational(3, 2) * planet.j2 * (planet.radius / s.radius) ** 2
    return [
        g_north + fraction * central * sp.sin(2 * phi),
        g_north + g_north.subs(phi, -phi),
        sp.diff(g_north, phi).subs(phi, sp.pi / 4),
        g_north.subs(phi, sp.pi / 4) + fraction * central,
        (g_down - central) - fraction * central * (1 - 3 * sp.sin(phi) ** 2),
        (g_down - central) - (g_down - central).subs(phi, -phi),
    ]


# -- the coupled state: heating, recession, structure -----------------------------


def _what_the_coupled_field_conserves_and_dissipates() -> Any:
    r"""On the full state: the count, the unit quaternion, the energy, the mass.

    The state has :math:`16 + 2N_T + 2N_m` components. Along the coupled
    field the norm of the quaternion does not change, so the unit sphere is
    invariant and the state lives on a manifold one dimension smaller. The
    energy still changes at :math:`(F_V/m)\,V` when :math:`F_V` is the
    component the attitude puts along the velocity and the mass varies: the
    dissipation that counts the passes is untouched by the coupling. And
    :math:`m - A_s\,\ell\sum_i\Delta_i\rho_i` is constant: the vehicle loses
    exactly the mass its heat shield loses, as char removed and as gas
    blown, and no more.
    """
    c = CoupledSymbols()
    s = c.entry
    state, rates = list(c.state), list(c.rates)
    along = c.wind_force[0] / c.mass
    return [
        sp.Integer(len(state) - (16 + 2 * c.cells + 2 * c.modes)),
        sp.Integer(len(rates) - len(state)),
        lie_derivative(c.attitude.norm_squared, state, rates),
        lie_derivative(s.energy, state, rates) - along * s.speed,
        lie_derivative(c.mass - c.ablating_area * c.heat_shield.areal_mass, state, rates),
    ]


def _the_heating_laws_are_monotone_and_the_film_coefficient_returns_the_correlation() -> Any:
    r"""Each term of the heating moves one way in each argument, and the wall sees it
    through a film coefficient.

    The stagnation flux :math:`k_Q\sqrt{\rho/R_n}\,V^3` has logarithmic
    derivatives :math:`1/(2\rho)` in density and :math:`3/V` in speed, both
    positive on the corridor, which is what lets it be enclosed by
    evaluating it at the ends of an interval. The film coefficient
    :math:`g_h = \dot q_s/(h_r - h_{w,0})` is defined so that a wall at the
    correlation's own enthalpy, with nothing blown, receives the correlation:
    the third entry. A hotter wall receives less, in the ratio
    :math:`(h_r - h_w)/(h_r - h_{w,0})`: the fourth.

    The last three are the blowing correction. :math:`\Omega =
    \Phi/(e^\Phi - 1)` tends to one as the blowing vanishes and falls from
    it at the rate one half in :math:`\Phi`; and it is the same function as
    the :math:`\ln(1 + x)/x` of the charring-ablator literature, with
    :math:`x = 2\lambda_b B'` formed on the coefficient of the blown layer,
    since then :math:`x = \Phi/\Omega = e^\Phi - 1`.
    """
    c = CoupledSymbols()
    density = sp.Symbol("rho", positive=True)
    flux = c.heat_flux.subs(c.density, density)
    balance = c.surface_balance
    reference = sp.Symbol("h_w0", real=True)
    wall = balance.wall_enthalpy
    unblown = balance.film_coefficient * (balance.recovery_enthalpy - wall)
    parameter = sp.Symbol("Phi", positive=True)
    correction = SurfaceBalance().blowing_correction
    correction = correction.subs(SurfaceBalance().blowing_parameter, parameter)
    blown = parameter / correction
    return [
        sp.diff(flux, density) / flux - 1 / (2 * density),
        sp.diff(flux, c.entry.speed) / flux - 3 / c.entry.speed,
        unblown.subs(wall, reference) - c.heat_flux,
        unblown / c.heat_flux
        - (balance.recovery_enthalpy - wall) / (balance.recovery_enthalpy - reference),
        sp.limit(correction, parameter, 0) - 1,
        sp.limit(sp.diff(correction, parameter), parameter, 0) + sp.Rational(1, 2),
        sp.log(1 + blown) / blown - correction,
    ]


def _a_structural_mode_is_a_damped_oscillator_whose_stiffness_can_do_work() -> Any:
    r"""The modal equation in energy form, with the softening term it implies.

    For :math:`\ddot\eta + 2\zeta\omega\dot\eta + \omega^2\eta = Q` with
    :math:`\omega = \omega(T_b)`, the modal energy
    :math:`\tfrac12\dot\eta^2 + \tfrac12\omega^2\eta^2` changes at
    :math:`Q\dot\eta - 2\zeta\omega\dot\eta^2 +
    \omega\,\omega'(T_b)\,\dot T_b\,\eta^2`: the forcing's power, less the
    damping, plus what the changing stiffness does. A structure that warms
    while it is deflected takes energy out of the mode when the frequency
    falls with temperature. That last term is the thermal half of the
    aerothermoelastic coupling, and it is in the equation as displayed. The
    temperature is the back face's, :math:`T_b = T_{N_T}`, whose rate is the
    heat shield's own.

    The second entry opens the forcing's power. With first-order piston
    pressure on the deformation it is :math:`Q^{(0)}\dot\eta -
    2\rho a_\infty c\,\dot\eta^2 - 2\rho a_\infty V g\,\eta\dot\eta`: the air
    adds :math:`2\rho a_\infty c` to the structure's own damping, and the
    term in :math:`g` is the work of an aerodynamic stiffness.
    """
    c = CoupledSymbols(modes=1)
    state, rates = list(c.state), list(c.rates)
    (displacement,), (velocity,) = c.modal_displacements, c.modal_velocities
    (frequency,), (damping,), (forcing,) = c.modal_frequencies, c.modal_damping, c.modal_forcing
    energy = velocity**2 / 2 + frequency**2 * displacement**2 / 2
    softening = (
        frequency
        * sp.diff(frequency, c.structure_temperature)
        * lie_derivative(c.structure_temperature, state, rates)
        * displacement**2
    )
    (rigid,) = c.rigid_forcing
    impedance = 2 * c.density * c.sound_speed
    power = (
        rigid * velocity
        - impedance * c.damping_integrals[0, 0] * velocity**2
        - impedance * c.entry.speed * c.stiffness_integrals[0, 0] * displacement * velocity
    )
    return [
        lie_derivative(energy, state, rates)
        - (forcing * velocity - 2 * damping * frequency * velocity**2 + softening),
        forcing * velocity - power,
    ]


# -- the heat shield ---------------------------------------------------------------------


def _the_temperature_equation_is_conservation_of_energy() -> Any:
    r"""The equation a charring-ablator code integrates, from the two conservation laws.

    With the solid a mixture of virgin material and char at one temperature,
    conservation of energy in the form
    :math:`\partial_t(\rho h) + \partial_y(j h_g - k\,\partial_y T) = 0`
    equals the temperature equation plus :math:`h_g` times conservation of
    mass, identically. So where mass is conserved the two say the same
    thing, and the temperature equation's two unobvious features are forced:
    the decomposition enthalpy is
    :math:`\bar h = (\rho_v h_v - \rho_c h_c)/(\rho_v - \rho_c)`, the second
    entry; and the gas's term is :math:`+\,j\,c_{p,g}\,\partial_y T` with
    :math:`j` the flux *into* the body, so that gas flowing out toward a
    hotter surface cools what it passes through.
    """
    slab = CharringSlab()
    temperature, density = slab.temperature, slab.density
    virgin, char = sp.Function("h_v")(temperature), sp.Function("h_c")(temperature)
    return [
        slab.energy_residual
        - (slab.temperature_residual + slab.gas_enthalpy(temperature) * slab.mass_residual),
        slab.decomposition_enthalpy(temperature, density)
        - (slab.virgin_density * virgin - slab.char_density * char)
        / (slab.virgin_density - slab.char_density),
    ]


def _the_landau_coordinate_holds_the_surface_still() -> Any:
    r"""In :math:`\eta = (y - s)/(L - s)` a time derivative at fixed depth picks up a drift.

    For a field given as :math:`G(\eta, t)`,

    .. math::

        \partial_t\big|_y = \partial_t\big|_\eta
            - \frac{\dot s\,(1 - \eta)}{L - s}\,\partial_\eta ,
        \qquad
        \partial_y = \frac{1}{L - s}\,\partial_\eta .

    Both are linear in :math:`G`, and are checked on :math:`a(t)\,\eta^n`
    for an arbitrary coefficient and exponent, whose sums are every field.
    """
    slab = CharringSlab()
    depth, clock, surface = slab.depth, slab.time, slab.surface
    eta = landau_coordinate(depth, surface, slab.thickness)
    coordinate = sp.Symbol("eta", positive=True)
    exponent = sp.Symbol("n", positive=True)
    in_frame = sp.Function("a")(clock) * coordinate**exponent
    at_depth = in_frame.subs(coordinate, eta)
    remaining = slab.thickness - surface
    drift = sp.diff(surface, clock) * (1 - coordinate) / remaining
    return [
        sp.diff(at_depth, clock)
        - (sp.diff(in_frame, clock) - drift * sp.diff(in_frame, coordinate)).subs(coordinate, eta),
        sp.diff(at_depth, depth)
        - (sp.diff(in_frame, coordinate) / remaining).subs(coordinate, eta),
    ]


def _what_the_slab_loses_through_its_receding_surface() -> Any:
    r"""Leibniz's rule on the receding slab, and the surface balance that closes it.

    For any conservation law :math:`\partial_t u + \partial_y\Phi = 0` on
    :math:`s(t)\le y\le L`, the content changes at
    :math:`-u(s)\dot s - [\Phi]_s^L`: the first entry, with the law's own
    residual kept as a remainder. With an impermeable, adiabatic back face
    that gives, per unit area, a mass falling at the char removed plus the
    gas blown, :math:`\dot m_c = \rho_s\dot s` and :math:`\dot m_g = -j(s)`;
    and an enthalpy changing at the heat conducted in, less what the char
    and the gas carry to the surface. The surface stores nothing, so the
    conduction is the net heating plus what arrives less what leaves; put
    in, the slab's enthalpy changes at the net heating less
    :math:`(\dot m_c + \dot m_g)\,h_w`: what is delivered, less the
    enthalpy of everything that departs, at the wall.
    """
    slab = CharringSlab()
    quantity = sp.Function("u")(slab.depth, slab.time)
    flux = sp.Function("Phi")(slab.depth, slab.time)
    balance = SurfaceBalance()
    surface_density, recession_rate, surface_flux = sp.symbols("rho_s sdot j_s", real=True)
    conducted = sp.Symbol("q_c", real=True)
    char, gas = surface_density * recession_rate, -surface_flux
    mass_rate = -surface_density * recession_rate + surface_flux
    solid, vapour = balance.char_enthalpy, balance.gas_enthalpy
    enthalpy_rate = -surface_density * solid * recession_rate + (surface_flux * vapour + conducted)
    fluxes = {balance.char_flux: char, balance.gas_flux: gas}
    return [
        slab.balance_residual(quantity, flux),
        mass_rate + (char + gas),
        enthalpy_rate - (conducted - char * solid - gas * vapour),
        enthalpy_rate.subs(conducted, balance.conduction.subs(fluxes))
        - (balance.net_heating.subs(fluxes) - (char + gas) * balance.wall_enthalpy),
    ]


def _the_cells_conserve_what_the_slab_conserves() -> Any:
    r"""On three unequal cells: the scheme's totals change at the continuum's rates.

    Summed over the cells, the mass rates telescope to
    :math:`-(\dot m_c + \dot m_g)` and the enthalpy rates to
    :math:`q_c - \dot m_c h_1 - \dot m_g h_g(T_1)`: nothing is created
    between cells, whatever the conductivity, the enthalpies and the
    decomposition rate are. The last two entries tie the rates of the
    *state* to those totals: the Lie derivative of the areal mass along the
    cells' field is the sum of the mass rates, and likewise the enthalpy,
    so the temperatures and densities that are integrated are the ones that
    conserve.
    """
    cells = CharringCells(
        cells=3, faces=(sp.S.Zero, sp.Rational(1, 6), sp.Rational(1, 2), sp.S.One)
    )
    state, rates = list(cells.state), list(cells.rates)
    mass, enthalpy = sum(cells.mass_rates), sum(cells.enthalpy_rates)
    surface = cells.temperatures[0]
    surface_enthalpy = cells.enthalpy_densities[0] / cells.densities[0]
    return [
        mass + cells.char_removal + cells.gas_blown,
        enthalpy
        - (
            cells.surface_conduction
            - cells.char_removal * surface_enthalpy
            - cells.gas_blown * cells.slab.gas_enthalpy(surface)
        ),
        lie_derivative(cells.areal_mass, state, rates) - mass,
        lie_derivative(cells.areal_enthalpy, state, rates) - enthalpy,
    ]


def _the_lumped_wall_is_two_cells_that_neither_decompose_nor_recede() -> Any:
    r"""What the lumped wall was: :math:`C_w\dot T_w = q_c - C_w (T_w - T_b)/\tau`.

    Two cells, no decomposition, no recession. The first cell's temperature
    then changes at :math:`q_c/C_w - (T_1 - T_2)/\tau_{\mathrm{cond}}` with

    .. math::

        C_w = \rho c_p\,\ell\,\Delta_1, \qquad
        \tau_{\mathrm{cond}} = \frac{C_w\,\ell\,(\eta^c_2 - \eta^c_1)}{k} ,

    which names the two constants the lumped model left free, and says what
    it left out: the decomposition, the gas, the motion of the surface, and
    every cell behind the second.
    """
    cells = CharringCells(cells=2)
    surface, back = cells.temperatures
    quiet = cells.temperature_rates[0].subs(cells.char_removal, 0).replace(
        sp.Function("R", nonnegative=True), lambda *_: sp.S.Zero
    )
    capacity = (
        cells.remaining * cells.widths[0] * cells.slab.heat_capacity(surface, cells.densities[0])
    )
    conductivity = cells.slab.conductivity(
        (surface + back) / 2, (cells.densities[0] + cells.densities[1]) / 2
    )
    time_constant = capacity * cells.remaining * sp.Rational(1, 2) / conductivity
    return quiet - (cells.surface_conduction / capacity - (surface - back) / time_constant)


# -- the figure of the planet, and its atmosphere --------------------------------------


def _the_surface_is_the_reference_ellipse_in_polar_form() -> Any:
    r"""The point at :math:`R(\phi)` along the radius lies on
    :math:`x^2/a^2 + z^2/b^2 = 1`.

    :math:`R(\phi)` is what the field of record subtracts from the radius to
    get the altitude, so this is the statement that the atmosphere is put on
    the reference ellipsoid and on nothing else.
    """
    return FigureSymbols().ellipse_residual(sp.Symbol("phi", real=True))


def _height_along_the_radius_is_geodetic_altitude_to_second_order() -> Any:
    r"""What measuring altitude along the radius costs, and what it cannot be used for.

    For a point at geodetic latitude :math:`\varphi` and altitude
    :math:`h_g`, the height along the radius less :math:`h_g` vanishes at
    orders zero and one in the flattening, and at order two it is
    :math:`\tfrac12\,a h_g\sin^2 2\varphi/(a + h_g)`. The altitude is
    written as the distance from the centre at zero flattening less the
    equatorial radius, :math:`h_g = s - a`, so that the square root there is
    of a quantity whose sign is known.

    The last two entries are the other face of the same geometry: the angle
    from the radius to the ellipsoid's normal vanishes on a sphere -- stated
    through its sine, since the angle itself is an inverse tangent whose
    branch SymPy will not choose -- and is :math:`f\,(a/s)\sin 2\varphi`
    to first order, which is not small beside an entry angle, and is why a
    direction published against the geodetic horizon has to be turned before
    it is the field's.
    """
    planet = EntrySymbols().planet
    distance = sp.Symbol("s", positive=True)
    figure = FigureSymbols(geodetic_altitude=distance - planet.radius)
    flattening = planet.flattening

    def order(expression: Any, power: int) -> Any:
        # Factored before the trigonometry is collected: expanding first is slow.
        coefficient = sp.diff(expression, flattening, power).subs(flattening, 0)
        return sp.trigsimp(sp.factor(coefficient)) / sp.factorial(power)

    double = sp.sin(2 * figure.geodetic_latitude)
    return [
        order(figure.radial_height_excess, 0),
        order(figure.radial_height_excess, 1),
        order(figure.radial_height_excess, 2)
        - planet.radius * (distance - planet.radius) * double**2 / (2 * distance),
        sp.expand_trig(sp.sin(figure.deflection.subs(flattening, 0))),
        order(figure.deflection, 1) - planet.radius * double / distance,
    ]


def _each_layer_of_the_standard_atmosphere_is_hydrostatic() -> Any:
    r"""In a layer of constant lapse rate, the stated pressure satisfies
    :math:`dp/dH = -\rho g_0`.

    With the layer's base values, lapse rate and :math:`g_0 M_0/R^*` left as
    symbols, for the layer with a temperature gradient and for the
    isothermal one. The last entry is why the layers are closed forms at
    all: :math:`dH/dZ = g(Z)/g_0`, so that in the geopotential altitude the
    hydrostatic equation has a constant gravity.
    """
    height, altitude = sp.Symbol("H", real=True), sp.Symbol("Z", positive=True)
    base, lapse = sp.symbols("H_b L", real=True)
    temperature, pressure, constant = sp.symbols("T_b p_b c_h", positive=True)
    standard_gravity = sp.Rational(980665, 100000)
    residuals = []
    for isothermal in (False, True):
        layer = StandardLayer(base, lapse, temperature, pressure, constant, isothermal)
        residuals.append(
            sp.diff(layer.pressure(height), height) + layer.density(height) * standard_gravity
        )
    residuals.append(
        sp.diff(geopotential(altitude), altitude) - gravity(altitude) / standard_gravity
    )
    return residuals


# -- piston theory ------------------------------------------------------------------


def _one_dimensional_isentropic_flow_has_two_riemann_invariants() -> Any:
    r"""Along :math:`dx/dt = u \pm a` the quantity :math:`u \pm 2a/(\kappa-1)` is carried.

    Stated as an identity between differential expressions, with the density
    and the velocity free: the rate of each invariant along its characteristic
    is the momentum equation plus or minus :math:`a/\rho` times the
    continuity equation. Where the flow satisfies both, the rate is zero.
    Nothing is assumed of the gas beyond :math:`p = K\rho^\kappa`.
    """
    gas = IsentropicFlow()
    return [
        gas.along_characteristic(gas.invariant(sign), sign)
        - (gas.momentum + sign * gas.sound_speed / gas.density * gas.continuity)
        for sign in (+1, -1)
    ]


def _the_pressure_on_a_piston_in_a_simple_wave() -> Any:
    r"""With one invariant uniform, the piston's speed fixes the pressure on its face.

    The state on the face is parametrized by :math:`\theta = a/a_\infty`, so
    that no power of an unsigned quantity appears. The invariant carried
    from gas at rest gives :math:`w/a_\infty = 2(\theta - 1)/(\kappa - 1)`;
    an isentropic change has :math:`\rho \propto a^{2/(\kappa-1)}`. The
    entries check, in order, that the piston relation returns
    :math:`\theta`; that the stated pressure and density have
    :math:`a^2 = \kappa p/\rho`; that they have :math:`p \propto
    \rho^\kappa`; and that the slope of the pressure against the piston
    speed is :math:`\rho a` *at the state on the face*, whatever that state
    is. The last is the first-order relation in its local form: it is what
    licenses applying :math:`dp = \rho a\,dw` to a perturbation about a flow
    that is not the undisturbed stream.
    """
    p = PistonSymbols()
    k = p.heat_ratio
    theta = sp.Symbol("theta", positive=True)
    mach = 2 * (theta - 1) / (k - 1)
    density_ratio = theta ** (2 / (k - 1))
    pressure_ratio = p.simple_wave(mach)

    speed = sp.Symbol("w", real=True)
    pressure = p.pressure * p.simple_wave(speed / p.sound_speed)
    slope = sp.diff(pressure, speed).subs(speed, mach * p.sound_speed)
    local_impedance = (p.density * density_ratio) * (p.sound_speed * theta)
    return [
        p.sound_ratio(mach) - theta,
        sp.powsimp(pressure_ratio / density_ratio) - theta**2,
        sp.expand_log(sp.log(pressure_ratio) - k * sp.log(density_ratio), force=True),
        sp.powsimp(sp.simplify(slope / local_impedance)) - 1,
    ]


def _the_pressure_behind_a_piston_driven_shock() -> Any:
    r"""Mass, momentum and energy across the shock, and the shock eliminated.

    In the frame of the gas ahead, at rest, a shock of speed :math:`U_s =
    M_s a_\infty` leaves gas moving with the piston at :math:`w`. The first
    three entries are the conservation laws across it,

    .. math::

        \rho_\infty U_s = \rho_2 (U_s - w), \qquad
        p_2 - p_\infty = \rho_\infty U_s w, \qquad
        p_2 w = \rho_\infty U_s \bigl(e_2 - e_\infty + \tfrac12 w^2\bigr),

    with :math:`e = p/((\kappa-1)\rho)`, satisfied by the jump relations
    :meth:`~aether.symbolic.piston.PistonSymbols.shock_jump` states. The last
    two eliminate the shock Mach number in favour of the piston speed.
    """
    p = PistonSymbols()
    k = p.heat_ratio
    shock_mach = sp.Symbol("M_s", positive=True)
    speed_ratio, pressure_ratio, density_ratio = p.shock_jump(shock_mach)
    sound = p.sound_speed
    shock_speed, piston_speed = shock_mach * sound, speed_ratio * sound
    behind_pressure, behind_density = pressure_ratio * p.pressure, density_ratio * p.density

    def internal_energy(pressure: Any, density: Any) -> Any:
        return pressure / ((k - 1) * density)

    advancing = sp.Symbol("mu_w", positive=True)
    return [
        p.density * shock_speed - behind_density * (shock_speed - piston_speed),
        (behind_pressure - p.pressure) - p.density * shock_speed * piston_speed,
        behind_pressure * piston_speed
        - p.density
        * shock_speed
        * (
            internal_energy(behind_pressure, behind_density)
            - internal_energy(p.pressure, p.density)
            + piston_speed**2 / 2
        ),
        speed_ratio.subs(shock_mach, p.shock_mach(advancing)) - advancing,
        pressure_ratio.subs(shock_mach, p.shock_mach(advancing)) - p.shock(advancing),
    ]


def _the_piston_expansions_agree_to_second_order_and_part_at_third() -> Any:
    r"""Taylor coefficients of the simple wave and of the shock, against the sources'.

    Both expand as :math:`1 + \kappa\mu + \tfrac14\kappa(\kappa+1)\mu^2 +
    \dots` in :math:`\mu = w/a_\infty`. The third-order coefficient is
    :math:`\kappa(\kappa+1)/12` for the simple wave and
    :math:`\kappa(\kappa+1)^2/32` behind a shock, which is the statement
    that a weak shock is isentropic to second order in its strength and not
    to third. The last entry is the gap between them,
    :math:`\kappa(\kappa+1)(5-3\kappa)/96`: a tenth of the coefficient for
    :math:`\kappa = 1.4`, as Lighthill remarks.
    """
    p = PistonSymbols()
    mach, k = p.piston_mach, p.heat_ratio
    residuals: list[Any] = []
    for through_shock, exact in ((False, p.simple_wave()), (True, p.shock())):
        stated = p.coefficients(3, through_shock=through_shock)
        residuals.extend(
            sp.diff(exact, mach, order).subs(mach, 0) / sp.factorial(order) - coefficient
            for order, coefficient in enumerate(stated)
        )
    simple, shocked = p.coefficients(3), p.coefficients(3, through_shock=True)
    residuals.append(simple[3] - shocked[3] - k * (k + 1) * (5 - 3 * k) / 96)
    return residuals


def _the_cubic_lies_below_the_simple_wave() -> Any:
    r"""The fourth derivative of the simple-wave pressure, in closed form.

    With :math:`\theta = 1 + \tfrac{\kappa-1}{2}\mu`, the pressure ratio is
    :math:`\theta^{2\kappa/(\kappa-1)}` and its fourth derivative in
    :math:`\mu` is :math:`\tfrac14\kappa(\kappa+1)(3-\kappa)\,
    \theta^{2\kappa/(\kappa-1)-4}`. Taylor's theorem then gives the simple
    wave less its cubic as :math:`\tfrac{1}{96}\kappa(\kappa+1)(3-\kappa)\,
    \mu^4\,\theta_\xi^{2\kappa/(\kappa-1)-4}` at an intermediate
    :math:`\theta_\xi`: positive for :math:`1 < \kappa < 3`, and bounded on
    any interval of piston speeds by the value of the last factor at its
    ends. Stated as a ratio, so that the powers of :math:`\theta` cancel
    before anything is expanded.
    """
    p = PistonSymbols()
    k = p.heat_ratio
    theta = sp.Symbol("theta", positive=True)
    fourth = ((k - 1) / 2) ** 4 * sp.diff(theta**p.exponent, theta, 4)
    stated = 24 * p.remainder_coefficient * theta ** (p.exponent - 4)
    return sp.powsimp(fourth / stated) - 1


def _a_convected_slab_meets_the_wall_as_a_piston() -> Any:
    r"""A slab carried at :math:`U` sees the wall move at :math:`\partial_t Z + U\partial_x Z`.

    The slab that was at :math:`x_0` is at :math:`x_0 + Ut`, where the wall
    stands at :math:`Z(x_0 + Ut, t)`. The identity is linear in :math:`Z`,
    and it is checked on a travelling wave of arbitrary profile, wavenumber
    and frequency, whose superpositions are every wall.
    """
    station, start, clock = sp.symbols("x x_0 t", real=True)
    stream, wavenumber, frequency = sp.symbols("U k_x omega", real=True)
    wall = sp.Function("G")(wavenumber * station - frequency * clock)
    carried = start + stream * clock
    return sp.diff(wall.subs(station, carried), clock) - downwash(
        wall, station, clock, stream
    ).subs(station, carried)


def _piston_theory_is_the_limit_of_the_linearized_flow_over_a_wavy_wall() -> Any:
    r"""Over a travelling wave of the wall, the exact linear pressure is the piston
    value times :math:`n/\sqrt{n^2-1}`.

    :math:`n` is the Mach number of the stream relative to the wave. In
    order: the potential satisfies the linearized equation; the fluid at
    the wall has the velocity the wall imposes; the pressure there is
    :math:`\rho a\,w\,n/\sqrt{n^2-1}`; equivalently it is
    :math:`\rho (U - c)^2\,\partial_x Z/\sqrt{n^2-1}` with :math:`c` the
    speed of the wave, which for a wall at rest is Ackeret's formula; and
    the factor exceeds one by
    :math:`1/\bigl(\sqrt{n^2-1}\,(n+\sqrt{n^2-1})\bigr)`, which is at most
    :math:`1/(2(n^2-1))`. First-order piston theory is the limit
    :math:`n\to\infty`, and its relative error is that excess: two percent
    at :math:`n = 5`. The last entry is the wall at rest, whose frequency
    vanishes when :math:`n` is the flight Mach number.
    """
    flow = WavyWall()
    relative_speed = flow.relative_mach * flow.sound_speed
    at_rest = WavyWall(relative_mach=flow.stream / flow.sound_speed)
    return [
        flow.wave_equation,
        flow.wall_velocity - flow.wall_downwash,
        flow.wall_pressure
        - flow.density * flow.sound_speed * flow.wall_downwash * flow.piston_factor,
        flow.wall_pressure
        - flow.density * relative_speed**2 * flow.wall_slope / flow.mach_slope,
        flow.piston_factor - 1 - flow.piston_excess,
        at_rest.frequency,
    ]


def _the_load_on_a_thin_surface_has_no_term_in_the_square_of_its_motion() -> Any:
    r"""Lighthill's Eq. (8): the cubic on the lower face less the cubic on the upper.

    A symmetric section gives both faces the piston speed :math:`w_s` of
    its thickness, and a motion gives them :math:`\pm w_i`. The difference
    of the two cubics is odd in :math:`w_i`: thickness changes the
    coefficient of the motion, by a factor that depends on Mach number and
    local slope, and no term in :math:`w_i^2` survives. To first order the
    load is :math:`2\kappa p_\infty w_i/a_\infty = 2\rho_\infty a_\infty
    w_i`, which is the second entry.
    """
    p = PistonSymbols()
    k = p.heat_ratio
    thickness, motion = sp.symbols("mu_s mu_i", real=True)
    stated = (
        2 * k + k * (k + 1) * thickness + k * (k + 1) / 2 * thickness**2
    ) * motion + k * (k + 1) / 6 * motion**3
    first_order = p.pressure * (p.expansion(1, motion) - p.expansion(1, -motion))
    return [
        p.expansion(3, thickness + motion) - p.expansion(3, thickness - motion) - stated,
        first_order - 2 * p.impedance * (motion * p.sound_speed),
    ]


def _piston_pressure_projected_on_the_modes_damps_and_stiffens() -> Any:
    r"""The generalized forces of first-order piston pressure, and what they do.

    For :math:`\zeta = \sum_j\eta_j W_j` the load
    :math:`f = -b\,z_a(\partial_t\zeta + U\partial_\xi\zeta)` has, mode by
    mode, the integrand :math:`f\,W_i = -\sum_j(c_{ij}\dot\eta_j +
    g_{ij}\eta_j)` with :math:`c_{ij} = b z_a W_i W_j` and
    :math:`g_{ij} = b z_a U\,W_i W_j'`: the first entries. The next is the
    power, :math:`\sum_i f W_i\dot\eta_i = -b z_a(\partial_t\zeta)^2 -
    b z_a U\,\partial_\xi\zeta\,\partial_t\zeta`, whose first term is a
    square: the damping can only dissipate, being the radiation of sound
    by a moving wall. The rest are the integration by parts that splits
    the stiffness: :math:`g_{ij} + g_{ji} = \partial_\xi(b z_a U\,W_i W_j)
    - \partial_\xi(b z_a U)\,W_i W_j`, so its symmetric part is a boundary
    term and a term in the gradient of the stream, and what is left is
    antisymmetric. An antisymmetric stiffness has no potential; it is the
    part that can move energy between two modes, and with it the
    mechanism of flutter is in the equations.
    """
    strip = StripSymbols(modes=2)
    indices = range(strip.modes)
    rates = [sp.diff(amplitude, strip.time) for amplitude in strip.amplitudes]
    residuals = [
        strip.forcing_density(i)
        + sum(
            strip.damping_density(i, j) * rates[j]
            + strip.stiffness_density(i, j) * strip.amplitudes[j]
            for j in indices
        )
        for i in indices
    ]
    velocity = sp.diff(strip.deflection, strip.time)
    slope = sp.diff(strip.deflection, strip.station)
    power = sum(strip.forcing_density(i) * rates[i] for i in indices)
    residuals.append(
        power + strip.width * strip.impedance * (velocity**2 + strip.stream * slope * velocity)
    )
    carried = strip.width * strip.impedance * strip.stream
    for i in indices:
        for j in range(i, strip.modes):
            product = strip.shapes[i] * strip.shapes[j]
            residuals.append(
                strip.stiffness_density(i, j)
                + strip.stiffness_density(j, i)
                - (
                    sp.diff(carried * product, strip.station)
                    - sp.diff(carried, strip.station) * product
                )
            )
    return residuals


def _a_flat_strip_at_incidence_has_normal_force_four_alpha_over_mach() -> Any:
    r"""First-order piston pressure on a flat strip: :math:`C_N = 4\sin\alpha/M`.

    The undeformed strip at incidence :math:`\alpha` presents the normal
    velocity :math:`V\sin\alpha` to the stream on one face and its negative
    on the other, so the force per unit area is
    :math:`2\rho_\infty a_\infty V\sin\alpha`. Referred to
    :math:`\tfrac12\rho_\infty V^2` that is :math:`4\sin\alpha/M`, the
    high-Mach-number form of the linear result
    :math:`4\alpha/\sqrt{M^2-1}`, which the wavy-wall factor
    :math:`M/\sqrt{M^2-1}` restores. The second entry says so.
    """
    p = PistonSymbols()
    alpha = sp.Symbol("alpha", real=True)
    speed = sp.Symbol("V", positive=True)
    mach = speed / p.sound_speed
    pressure_difference = 2 * p.impedance * speed * sp.sin(alpha)
    dynamic_pressure = p.density * speed**2 / 2
    flow = WavyWall(relative_mach=sp.Symbol("M", positive=True))
    linear_theory = 4 * sp.sin(alpha) / flow.mach_slope
    return [
        pressure_difference / dynamic_pressure - 4 * sp.sin(alpha) / mach,
        (4 * sp.sin(alpha) / flow.relative_mach) * flow.piston_factor - linear_theory,
    ]


# -- the phugoid ----------------------------------------------------------------


def _phugoid_frequency_and_damping() -> Any:
    r"""Frozen-coefficient spectrum on the glide, to leading order in :math:`H_s`.

    :math:`H_s\,a_1 \to g - V^2/r` gives the frequency. The pair's decay rate
    is :math:`(a_2 + \lambda_{\mathrm{slow}})/2` with
    :math:`\lambda_{\mathrm{slow}} \to -a_0/a_1`, and its limit is
    :math:`g/(V E_v)`: the scale height has dropped out. The last two entries
    are the two coefficients the manuscript quotes on the way:
    :math:`a_2 = 2(g - V^2/r)/(E_v V)` exactly, and
    :math:`a_0/a_1 \to -2V/(r E_v)`.
    """
    s = GlideSymbols()
    a2, a1, a0 = s.characteristic_coefficients
    height = s.scale_height
    return [
        sp.limit(height * a1, height, 0) - s.net_gravity,
        sp.limit((a2 - a0 / a1) / 2, height, 0) - s.phugoid_damping_rate,
        sp.diff(s.phugoid_damping_rate, height),
        a2 - 2 * s.net_gravity / (s.vertical_lift_to_drag * s.speed),
        sp.limit(a0 / a1, height, 0) + 2 * s.speed / (s.radius * s.vertical_lift_to_drag),
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


def _a_bank_history_pumps_or_damps_the_oscillation() -> Any:
    r"""Scaling the vertical lift by :math:`c` changes the oscillator's energy at
    :math:`k\,(c-1)\,e^{-z}\,\dot z`.

    Fly at a bank :math:`\sigma` a vehicle whose glide at :math:`(r, V)` was
    set at a reference bank :math:`\sigma_{\mathrm{ref}}`. The level-flight
    vertical acceleration :math:`z` scale heights off that glide tends to
    :math:`G\,(c\,e^{-z} - 1)` with :math:`c = \cos\sigma/\cos\sigma_{\mathrm{ref}}`,
    so the equilibrium has moved to :math:`z = \ln c`: a fixed fraction of a
    scale height, however thin the layer. Along
    :math:`\ddot z = k\,(c\,e^{-z} - 1)` the energy of the reference
    oscillator, :math:`\tfrac12\dot z^2 + k\,(e^{-z} + z)`, changes at exactly
    :math:`k\,(c-1)\,e^{-z}\,\dot z`. More lift while climbing and less while
    descending raises it at every instant, and the opposite lowers it. The
    amplitude of the oscillation about the glide is therefore something the
    bank history acts on throughout, not something it excites once.
    """
    s = GlideSymbols()
    height = s.scale_height
    z, z_rate = sp.symbols("z z_dot", real=True)
    reference_bank = sp.Symbol("sigma_ref", real=True)
    lift_ratio = sp.cos(s.bank) / sp.cos(reference_bank)
    reference_coefficient = s.glide_ballistic_coefficient.subs(s.bank, reference_bank)
    level = (s.speed * s.rates[2]).subs(s.gamma, 0)
    displaced = level.subs(s.radius, s.radius + height * z).subs(
        s.ballistic_coefficient, reference_coefficient
    )
    stiffness = s.phugoid_frequency_squared
    forced = stiffness * (lift_ratio * sp.exp(-z) - 1)
    energy = z_rate**2 / 2 + stiffness * (sp.exp(-z) + z)
    return [
        sp.limit(sp.simplify(displaced), height, 0, "+")
        - s.net_gravity * (lift_ratio * sp.exp(-z) - 1),
        forced.subs(z, sp.log(lift_ratio)),
        lie_derivative(energy, (z, z_rate), (z_rate, forced))
        - stiffness * (lift_ratio - 1) * sp.exp(-z) * z_rate,
    ]


# -- the corridor, and wrapping ---------------------------------------------------


def _the_range_is_bounded_by_the_energy_to_be_lost() -> Any:
    r"""Under an exponential density floor, :math:`s \le (2\beta/\rho_c)[\ln(V_0/V_f) + gH/V_f^2]`.

    The ground track advances no faster than the vehicle and the energy
    :math:`E = V^2/2 + \Phi` falls at the drag power (``energy-dissipation``),
    so :math:`\mathrm ds \le -\mathrm dE/(D/m)`. What is checked here is the
    algebra between that inequality and the bound. The air is at least
    :math:`\rho_c` where the potential is :math:`\Phi_{\max}` and thickens
    below at least as fast as :math:`e^{(\Phi_{\max}-\Phi)/(gH)}`, so the drag
    is at least :math:`(\rho_c/\beta)\,e^{(\Phi_{\max}-\Phi)/(gH)}(E-\Phi)`.
    At a given energy that floor falls as the potential rises (first entry),
    so it is least at the largest potential the state can have: the ceiling
    while :math:`E \ge E^* = \Phi_{\max} + V_f^2/2`, and below that the level
    at which the speed is :math:`V_f` (second entry). Integrated in the
    energy, the reciprocal of the floor gives the logarithm above
    :math:`E^*` and :math:`gH/V_f^2` below it. The last entry says the
    logarithm is not slack: the path flown level at the ceiling density,
    :math:`\mathrm dV/\mathrm ds = -\rho_c V/(2\beta)`, is that long.
    """
    energy, top = sp.symbols("E Phi_max", real=True)
    potential = sp.Symbol("Phi", real=True)
    density, beta, first, last, speed = sp.symbols("rho_c beta V_0 V_f V", positive=True)
    kinetic, fold = sp.symbols("u gH", positive=True)
    floor = (density / beta) * sp.exp((top - potential) / fold) * (energy - potential)
    threshold = top + last**2 / 2
    slowest = floor.subs(potential, energy - last**2 / 2)
    bound = (2 * beta / density) * (sp.log(first / last) + fold / last**2)
    above = sp.integrate(beta / (density * kinetic), (kinetic, last**2 / 2, first**2 / 2))
    below = sp.integrate(
        (2 * beta / (density * last**2)) * sp.exp(-(threshold - energy) / fold),
        (energy, -sp.oo, threshold),
    )
    level = (2 * beta / density) * sp.log(first / speed)
    return [
        sp.diff(floor, potential) + floor * (1 / fold + 1 / (energy - potential)),
        slowest - (density / (2 * beta)) * last**2 * sp.exp((threshold - energy) / fold),
        sp.expand_log(above + below - bound, force=True),
        sp.diff(level, speed) + 2 * beta / (density * speed),
    ]


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
        "apx:dyn, rem:j2-glide, rem:gravity-spherical",
        _radial_oblateness_changes_sign_at_35_degrees,
    ),
    Derivation(
        "oblateness-heading",
        "g_N is -(3/2) J2 (mu/r^2)(R_e/r)^2 sin(2 phi): odd in latitude and largest at 45 "
        "degrees, while the radial correction is even",
        "apx:dyn, rem:j2-skip",
        _oblateness_bends_the_heading_with_the_sign_of_the_hemisphere,
    ),
    Derivation(
        "kinematics",
        "the position rates are the planet-relative velocity",
        "apx:dyn, eq:rdot, eq:lamdot, eq:phidot",
        _position_rates_are_the_velocity,
    ),
    Derivation(
        "newton-rotating-frame",
        "the velocity rates are Newton's law in the rotating frame: J2, Coriolis, centrifugal",
        "apx:dyn, eq:Vdot, eq:gamdot, eq:psidot",
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
        "apx:dyn, eq:Vhat, eq:ngam, eq:npsi",
        _wind_triad_is_orthonormal_and_right_handed,
    ),
    Derivation(
        "energy-dissipation",
        "the Jacobi energy changes at the along-track specific power, dE/dt = (F_V/m) V",
        "sec:loss, eq:kmax; apx:time, lem:energy, eq:dissipation",
        _energy_dissipates_at_the_drag_power,
    ),
    Derivation(
        "defect-normalised",
        "the drag power on the fast-time axis has the mass of the dissipation, so the "
        "defect density has unit mass for every epsilon",
        "sec:loss, eq:defect",
        _the_dissipation_density_has_the_mass_of_the_dissipation,
    ),
    Derivation(
        "impulse-bound",
        "C_a^2 P^2 - |a|^2 is a sum of two terms that are nonnegative on the corridor under "
        "the dissipative enclosure",
        "apx:time, lem:impulse, asm:dissipative",
        _the_aerodynamic_impulse_is_bounded_by_the_dissipation,
    ),
    Derivation(
        "glide-balance",
        "the translational glide balance with rotation and oblateness retained",
        "apx:dyn, eq:glide-balance",
        _the_glide_balance_retains_rotation,
    ),
    Derivation(
        "quaternion-kinematics",
        "the quaternion rate rotates the body frame at the body rate and preserves the norm",
        "apx:dyn, eq:DCM, eq:qdot, eq:Omega",
        _quaternion_kinematics_rotate_the_body_frame,
    ),
    Derivation(
        "local-frame",
        "the north-east-down basis displayed is orthonormal, right-handed, and the one the "
        "field was derived on",
        "apx:dyn, sec:frames",
        _the_local_frame_is_the_one_the_field_was_derived_on,
    ),
    Derivation(
        "rotation-matrix",
        "det R(q) = |q|^6 and R^T [a x] R = |q|^2 [(R^T a) x]: a proper rotation on the unit "
        "sphere",
        "apx:dyn, eq:DCM, eq:RLB",
        _the_direction_cosine_matrix_is_a_proper_rotation,
    ),
    Derivation(
        "omega-transport",
        "driving the quaternion at omega - R^T omega_E turns the body at omega in inertial "
        "space, and keeps the norm",
        "apx:dyn, eq:omega-rel, rem:omega-transport",
        _the_planets_rotation_is_carried_into_the_body_rate,
    ),
    Derivation(
        "euler-equations",
        "I omega' = M - omega x I omega - I' omega for a deforming body; mass lost from the "
        "surface carries off exactly its I' omega",
        "apx:dyn, eq:euler, rem:euler-mass-loss",
        _eulers_equations_with_a_changing_inertia,
    ),
    Derivation(
        "attitude-to-force",
        "at incidence alpha and bank sigma the body force projects to F_V = -D, "
        "F_gamma = L cos sigma, F_psi = L sin sigma",
        "apx:dyn, eq:Fa-body, eq:Fa-LVLH, eq:F-projections, rem:bank-angle",
        _the_body_force_on_the_wind_axes,
    ),
    Derivation(
        "incidence-angles",
        "the body-frame velocity is V (cos alpha cos beta, sin beta, sin alpha cos beta); "
        "the freestream convention shifts alpha by pi",
        "apx:dyn, eq:aoa-sideslip, rem:aoa-sideslip",
        _incidence_angles_are_read_off_one_vector,
    ),
    Derivation(
        "glide-balance-attitude",
        "with the normal force supplied by the attitude the balance is F_gamma/m = its "
        "right-hand side; in trim and on a classical planet, Lambda = 1",
        "sec:loss; apx:dyn, eq:glide-full, eq:glide-3dof, rem:manifold-attitude",
        _the_glide_balance_constrains_the_attitude,
    ),
    Derivation(
        "trim-reduction",
        "driven through an attitude at incidence and bank, the field is the point-mass "
        "field with beta = m/(C_D S) and L/D = C_L/C_D",
        "apx:dyn, sec:aero-force; apx:residual, sec:residual-glide",
        _in_trim_the_field_is_the_point_mass_field,
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
        "sec:bound; apx:residual, eq:CR",
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
        "sec:loss; apx:residual, eq:turning; apx:time, rem:regimes",
        _a_pass_dissipates_in_proportion_to_its_turn,
        kind="limit",
    ),
    Derivation(
        "pass-depth",
        "cos(gamma) - E_v H_s rho/(2 beta) changes only through gravity, so a pass entered at "
        "gamma_e bottoms out where the lift is V^2 (1 - cos gamma_e)/H_s",
        "sec:loss, sec:composition; apx:residual, eq:pass-depth",
        _the_depth_of_a_pass_is_set_by_its_entry_angle,
    ),
    Derivation(
        "one-oscillator",
        "z scale heights off the glide, level flight accelerates at G(e^-z - 1): glide, phugoid "
        "and skip are the equilibrium, small and large oscillations of one equation",
        "sec:bound, eq:oscillator-body; apx:residual, eq:oscillator, eq:amplitude",
        _glide_phugoid_and_skip_are_one_oscillator,
        kind="limit",
    ),
    Derivation(
        "bank-pumping",
        "a bank change moves the glide by ln(cos sigma/cos sigma_ref) scale heights, and the "
        "oscillator's energy changes at k (c - 1) e^-z dz/dt: in step it pumps, opposed it damps",
        "sec:bound; apx:residual, eq:pumping",
        _a_bank_history_pumps_or_damps_the_oscillation,
        kind="limit",
    ),
    Derivation(
        "coupled-field",
        "on the full state: 16 + 2 N_T + 2 N_m components, the quaternion norm is invariant, "
        "the energy dissipates at (F_V/m) V, and the vehicle loses exactly the mass its heat "
        "shield loses",
        "sec:application, eq:state-rigid; apx:dyn, eq:state-full, eq:mass-rate",
        _what_the_coupled_field_conserves_and_dissipates,
    ),
    Derivation(
        "thermal-laws",
        "the stagnation flux is monotone in density and speed; the film coefficient returns "
        "the correlation at its own wall enthalpy and less to a hotter wall; the blowing "
        "correction tends to one",
        "apx:dyn, sec:thermal, eq:Qdot, eq:film, eq:blowing",
        _the_heating_laws_are_monotone_and_the_film_coefficient_returns_the_correlation,
    ),
    Derivation(
        "ablation-energy",
        "conservation of energy in the decomposing solid is the temperature equation plus h_g "
        "times conservation of mass; the decomposition enthalpy is (rho_v h_v - rho_c h_c)/"
        "(rho_v - rho_c)",
        "apx:dyn, sec:thermal, eq:slab-conservation, eq:slab-temperature",
        _the_temperature_equation_is_conservation_of_energy,
    ),
    Derivation(
        "ablation-landau",
        "in eta = (y - s)/(L - s) the time derivative at fixed depth is that at fixed eta less "
        "sdot (1 - eta)/(L - s) times the eta-derivative",
        "apx:dyn, sec:thermal, eq:landau",
        _the_landau_coordinate_holds_the_surface_still,
    ),
    Derivation(
        "ablation-balance",
        "the receding slab's mass falls at mdot_c + mdot_g, and its enthalpy changes at the "
        "net heating less (mdot_c + mdot_g) h_w",
        "apx:dyn, sec:thermal, eq:slab-balance, eq:surface-balance, eq:recession",
        _what_the_slab_loses_through_its_receding_surface,
    ),
    Derivation(
        "ablation-cells",
        "on cells that move with the surface the total mass and the total enthalpy change at "
        "the continuum's rates, and the integrated temperatures and densities are the ones "
        "that conserve",
        "apx:dyn, sec:thermal, eq:cells",
        _the_cells_conserve_what_the_slab_conserves,
    ),
    Derivation(
        "ablation-lumped-wall",
        "the lumped wall is two cells with no decomposition and no recession, with C_w = "
        "rho c_p l Delta_1 and tau_cond = C_w l (eta_2 - eta_1)/k",
        "apx:dyn, sec:thermal, rem:lumped-wall",
        _the_lumped_wall_is_two_cells_that_neither_decompose_nor_recede,
    ),
    Derivation(
        "modal-energy",
        "a mode's energy changes at Q eta' - 2 zeta omega eta'^2 + omega omega'(T_b) T_b' "
        "eta^2: forcing, damping, and the work of a changing stiffness; the forcing's power "
        "is Q0 eta' - 2 rho a c eta'^2 - 2 rho a V g eta eta'",
        "apx:dyn, sec:modal, eq:modal, eq:modal-forcing",
        _a_structural_mode_is_a_damped_oscillator_whose_stiffness_can_do_work,
    ),
    Derivation(
        "piston-characteristics",
        "along dx/dt = u +- a the quantity u +- 2a/(kappa-1) changes at the momentum "
        "equation +- (a/rho) times continuity: the Riemann invariants of one-dimensional "
        "isentropic flow",
        "apx:dyn, sec:piston, eq:riemann",
        _one_dimensional_isentropic_flow_has_two_riemann_invariants,
    ),
    Derivation(
        "piston-simple-wave",
        "next to gas at rest the pressure on the piston is p_inf (1 + (kappa-1) w/(2 a_inf))^"
        "(2 kappa/(kappa-1)), and dp/dw = rho a at every state of the wave",
        "apx:dyn, sec:piston, eq:piston-exact, eq:piston-linear, rem:piston-local",
        _the_pressure_on_a_piston_in_a_simple_wave,
    ),
    Derivation(
        "piston-shock",
        "behind the shock an advancing piston drives, p/p_inf = 1 + kappa (w/a_inf) M_s with "
        "M_s = b + sqrt(1 + b^2) and b = (kappa+1) w/(4 a_inf): mass, momentum and energy "
        "across it",
        "apx:dyn, sec:piston, eq:piston-shock",
        _the_pressure_behind_a_piston_driven_shock,
    ),
    Derivation(
        "piston-expansion",
        "simple wave and shock share 1 + kappa mu + kappa(kappa+1) mu^2/4 and part at third "
        "order, kappa(kappa+1)/12 against kappa(kappa+1)^2/32",
        "apx:dyn, sec:piston, eq:piston-cubic",
        _the_piston_expansions_agree_to_second_order_and_part_at_third,
    ),
    Derivation(
        "piston-remainder",
        "the simple wave exceeds its cubic by kappa(kappa+1)(3-kappa)/96 mu^4 times a positive "
        "power of a/a_inf at an intermediate piston speed",
        "apx:dyn, sec:piston, eq:piston-remainder",
        _the_cubic_lies_below_the_simple_wave,
    ),
    Derivation(
        "piston-slab",
        "a slab convected at U meets the wall Z(x, t) as a piston of velocity dZ/dt + U dZ/dx",
        "apx:dyn, sec:piston, eq:downwash",
        _a_convected_slab_meets_the_wall_as_a_piston,
    ),
    Derivation(
        "piston-linear-limit",
        "over a travelling wave of the wall the linearized pressure is rho a w n/sqrt(n^2-1), "
        "n the Mach number of the stream relative to the wave: Ackeret's formula for a wall "
        "at rest, piston theory as n grows, in excess by at most 1/(2(n^2-1))",
        "apx:dyn, sec:piston, eq:piston-wavy",
        _piston_theory_is_the_limit_of_the_linearized_flow_over_a_wavy_wall,
    ),
    Derivation(
        "piston-load",
        "lower face less upper is [2 kappa + kappa(kappa+1) mu_s + kappa(kappa+1) mu_s^2/2] mu_i "
        "+ kappa(kappa+1) mu_i^3/6: no term in the square of the motion, and 2 rho a w to "
        "first order",
        "apx:dyn, sec:piston, eq:piston-load",
        _the_load_on_a_thin_surface_has_no_term_in_the_square_of_its_motion,
    ),
    Derivation(
        "piston-modal-forces",
        "projected on the modes, first-order piston pressure is -(C eta' + G eta): C a Gram "
        "matrix, which can only dissipate, and G with a symmetric part that is a boundary term",
        "apx:dyn, sec:piston, eq:piston-strip, eq:aero-matrices; sec:modal, eq:modal-forcing",
        _piston_pressure_projected_on_the_modes_damps_and_stiffens,
    ),
    Derivation(
        "piston-flat-strip",
        "a flat strip at incidence has C_N = 4 sin(alpha)/M, the high-Mach-number form of "
        "4 alpha/sqrt(M^2-1)",
        "apx:dyn, sec:piston, eq:piston-flat",
        _a_flat_strip_at_incidence_has_normal_force_four_alpha_over_mach,
    ),
    Derivation(
        "figure-ellipsoid",
        "the surface the altitude is measured from, a(1-f)/sqrt(1 - f(2-f) cos^2 phi), is the "
        "reference ellipse in polar form about its centre",
        "apx:dyn, sec:density, eq:ellipsoid",
        _the_surface_is_the_reference_ellipse_in_polar_form,
    ),
    Derivation(
        "figure-altitude",
        "height along the radius exceeds the geodetic altitude by a h sin^2(2 phi)/(2(a+h)) f^2 "
        "and by nothing at first order; the two verticals differ by f (a/(a+h)) sin(2 phi)",
        "apx:dyn, sec:density, eq:altitude, rem:geodetic-altitude",
        _height_along_the_radius_is_geodetic_altitude_to_second_order,
    ),
    Derivation(
        "atmosphere-hydrostatic",
        "each layer of the 1976 standard satisfies dp/dH = -rho g_0, and dH/dZ = g(Z)/g_0",
        "apx:dyn, sec:density, eq:standard-layer",
        _each_layer_of_the_standard_atmosphere_is_hydrostatic,
    ),
    Derivation(
        "range-bound",
        "with the density at least rho_c at the ceiling and at least exponential below it, "
        "the ground track is at most (2 beta/rho_c)[ln(V_0/V_f) + g H/V_f^2], and the path of "
        "level flight at the ceiling density is the logarithm",
        "sec:first-bound, prop:range-bound, eq:range-bound",
        _the_range_is_bounded_by_the_energy_to_be_lost,
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


#: Labels of the manuscript that no derivation backs, each with the reason.
#:
#: The registry is one half of the contract with the manuscript and this is
#: the other. A displayed equation, lemma, proposition or remark is either
#: named by a derivation above or listed here, and the tests hold the two
#: lists against the sources in both directions: a new equation has to be
#: given a check or a reason, and a reason goes stale the day its label does.
#: What is listed is analysis or argument, not algebra. None of it is a model
#: equation that went unchecked.
STATED_NOT_DERIVED: dict[str, str] = {
    "eq:target": (
        "the containment the program aims at; a statement to be proved, not an identity"
    ),
    "eq:bv": (
        "a bound on the limiting velocity obtained along a subsequence (Helly's theorem); "
        "the law of motion it starts from is `newton-rotating-frame` and the constant in it "
        "is `impulse-bound`"
    ),
    "prop:slow": (
        "compactness of energies, positions and velocities in slow time; an argument about "
        "limits of sequences, resting on `energy-dissipation` and `impulse-bound`"
    ),
    "eq:liouville-limit": (
        "the limit of the Liouville equation, by the chain rule for functions of bounded "
        "variation applied to the limit trajectory"
    ),
    "prop:liouville": "the proposition that eq:liouville-limit states; analysis, as above",
    "eq:additivity": (
        "additivity of the dissipation over windows that are eventually disjoint; a property "
        "of a measure"
    ),
    "rem:psi-singularity": (
        "says where the chart is singular and that the corridor excludes it; the equation it "
        "refers to is covered by `newton-rotating-frame`"
    ),
    "rem:sutton-graves-circularity": (
        "an argument about which way a heating correlation would bias the count of passes; "
        "the count itself rests on `energy-dissipation`"
    ),
    "rem:thermal-embedding": (
        "a note on the polynomial recasting the main line no longer uses"
    ),
    "eq:piston-validity": (
        "the conditions under which a slab of a fast stream moves as a one-dimensional flow, "
        "and the orders of what the expansion omits; taken from Lighthill (after Hayes and "
        "Goldsworthy) and from Ashley and Zartarian (after Landahl), not derived. The "
        "linearized case, where the omission can be written down, is `piston-linear-limit`"
    ),
}


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
