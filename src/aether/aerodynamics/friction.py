"""Skin friction: the part of drag an inviscid method cannot see.

A panel method integrates pressure. An Euler solution integrates pressure.
Neither contains a boundary layer, so neither contains friction — and on a
body with a fineness ratio near twelve, friction is not negligible. The
reference-heavy stack has 305 m² of wetted area against 7.07 m² of frontal area, a
ratio of 43, so every count of skin friction is 43 counts of :math:`C_A`.

Measured on the stack along an ascent schedule, friction is **5 % of axial
force at Mach 3 and 7 % at Mach 5**, rising as the vehicle climbs and the
Reynolds number falls. That is smaller than a first guess suggests, because
the guess usually reaches for a wind-tunnel :math:`c_f` near 0.003 while an
ascending launch vehicle sits at :math:`Re_L \\sim 10^8`, where the
compressible turbulent value is nearer 0.0012. It is still several times the
error budget of the pressure methods, and it is signed: leaving it out always
underpredicts drag.

The method is Eckert's reference temperature, as given by Anderson,
*Hypersonic and High-Temperature Gas Dynamics*, 2nd ed., §6.9: evaluate the
incompressible flat-plate correlations at a temperature representative of
conditions inside the boundary layer,

.. math::

    \\frac{T^*}{T_e} = 1 + 0.032 M_e^2 + 0.58\\left(\\frac{T_w}{T_e} - 1\\right)
    \\qquad\\text{(Eq. 6.159)}

and use :math:`c_f^* = 0.664/\\sqrt{Re^*_x}` laminar (Eq. 6.155) and
:math:`c_f^* = 0.0592/(Re^*_x)^{0.2}` turbulent (Eq. 6.161), with
:math:`\\sqrt{3}` on the laminar branch for a cone (the Mangler factor,
§6.9). Anderson's own note on carrying it to three dimensions is what this
module implements: :math:`Re^*_x` is formed on the **running length along a
streamline**, which for a body of revolution at incidence is the meridian arc
length from the nose.

What is checked and what is trusted
-----------------------------------

The laminar branch is verified against a *solution*, not another
correlation: :func:`compressible_blasius` integrates the transformed laminar
boundary-layer equations

.. math::

    (C f'')' + f f'' = 0, \\qquad
    \\left(\\frac{C}{Pr} g'\\right)' + f g'
    + C\\,(\\gamma-1)M_e^2\\,(f'')^2 = 0

as a two-point boundary-value problem with :math:`C = \\rho\\mu/\\rho_e\\mu_e`
from Sutherland's law. That is first-principles laminar compressible flow,
and the agreement between it and the reference-temperature method is
measured rather than assumed.

The turbulent branch has no such check available and is a correlation
standing on its own. Its incompressible limit is pinned — 0.0592 Re⁻⁰·² is
within 2 % of Schultz–Grunow over 10⁵ to 10⁷ — but its compressible
behaviour is trusted, not verified, and that asymmetry is why the two
branches are separate functions with separate docstrings.

Transition is a **band, not a switch**. A hard laminar-to-turbulent jump puts
a step in drag as a vehicle climbs, and an adaptive integrator finds every
one of them. The two branches are blended over a Reynolds-number band with
the same :math:`C^2` smoothstep the pressure closure uses. The band is an
engineering smoothing of a real bifurcation, not a transition model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import scipy.integrate
import scipy.interpolate
from numpy.typing import ArrayLike, NDArray

from aether.atmosphere.standard import (
    SUTHERLAND_BETA,
    SUTHERLAND_S,
)
from aether.blending import smoothstep

__all__ = [
    "CONE_MANGLER_FACTOR",
    "PRANDTL_AIR",
    "SPECIFIC_HEAT_AIR",
    "STEFAN_BOLTZMANN",
    "AdiabaticWall",
    "BlasiusSolution",
    "BoundaryLayer",
    "FixedWall",
    "RadiativeEquilibriumWall",
    "WallCondition",
    "adiabatic_wall_temperature",
    "compressible_blasius",
    "eckert_reference_temperature",
    "laminar_skin_friction",
    "recovery_factor",
    "reference_temperature",
    "reference_temperature_of",
    "turbulent_skin_friction",
]

_FloatArray = NDArray[np.float64]

#: Prandtl number of air, taken constant. Anderson's exact laminar solutions
#: use 0.75; 0.71 is the value at moderate temperature and is the one the
#: reference-temperature correlations were fitted with.
PRANDTL_AIR = 0.71
#: Specific heat at constant pressure for air, J kg⁻¹ K⁻¹ (perfect gas).
SPECIFIC_HEAT_AIR = 1004.5
#: :math:`\sigma`, W m⁻² K⁻⁴.
STEFAN_BOLTZMANN = 5.670374419e-8
#: :math:`\sqrt{3}` — the laminar Mangler factor for a cone (Anderson §6.9).
CONE_MANGLER_FACTOR = float(np.sqrt(3.0))


def _sutherland(temperature: ArrayLike) -> _FloatArray:
    t = np.asarray(temperature, dtype=np.float64)
    return np.asarray(SUTHERLAND_BETA * t**1.5 / (t + SUTHERLAND_S))


def recovery_factor(prandtl: float = PRANDTL_AIR, turbulent: bool = False) -> float:
    """:math:`r = \\sqrt{Pr}` laminar, :math:`Pr^{1/3}` turbulent.

    The fraction of the freestream kinetic energy that appears as temperature
    rise at an insulated wall. For air these are 0.843 and 0.892, so a
    turbulent boundary layer recovers slightly more — which is why a
    turbulent adiabatic wall runs hotter than a laminar one at the same Mach.
    """
    p = float(prandtl)
    if not (np.isfinite(p) and p > 0.0):
        msg = f"Prandtl number must be finite and > 0, got {prandtl}"
        raise ValueError(msg)
    return float(p ** (1.0 / 3.0) if turbulent else np.sqrt(p))


def adiabatic_wall_temperature(
    edge_temperature: ArrayLike,
    edge_mach: ArrayLike,
    recovery: float,
    gamma: float = 1.4,
) -> _FloatArray:
    """:math:`T_{aw} = T_e\\left(1 + r\\frac{\\gamma-1}{2}M_e^2\\right)` (K)."""
    t = np.asarray(edge_temperature, dtype=np.float64)
    m = np.asarray(edge_mach, dtype=np.float64)
    return np.asarray(t * (1.0 + float(recovery) * 0.5 * (gamma - 1.0) * m * m))


def reference_temperature(
    edge_temperature: ArrayLike,
    edge_mach: ArrayLike,
    wall_temperature: ArrayLike,
) -> _FloatArray:
    """:math:`T^*` by Anderson Eq. 6.159 (K).

    .. math::

        \\frac{T^*}{T_e} = 1 + 0.032 M_e^2
        + 0.58\\left(\\frac{T_w}{T_e} - 1\\right)

    Transcribed from the reference rather than reconstructed: the coefficients
    0.032 and 0.58 are not the 0.22r(γ−1)/2 and 0.5 of the form usually
    quoted from Eckert, and the difference is a few per cent in :math:`c_f` at
    Mach 20. Use :func:`eckert_reference_temperature` for that other form —
    both are provided because they are genuinely different correlations and
    picking one silently would hide which.
    """
    t_e = np.asarray(edge_temperature, dtype=np.float64)
    m_e = np.asarray(edge_mach, dtype=np.float64)
    t_w = np.asarray(wall_temperature, dtype=np.float64)
    return np.asarray(t_e * (1.0 + 0.032 * m_e * m_e) + 0.58 * (t_w - t_e))


def eckert_reference_temperature(
    edge_temperature: ArrayLike,
    edge_mach: ArrayLike,
    wall_temperature: ArrayLike,
    recovery: float = 0.843,
    gamma: float = 1.4,
) -> _FloatArray:
    """:math:`T^* = 0.5(T_e + T_w) + 0.22\\,r\\frac{\\gamma-1}{2}M_e^2 T_e` (K).

    The form with an explicit recovery factor, which is what makes it usable
    on a turbulent layer where :math:`r = Pr^{1/3}` rather than
    :math:`\\sqrt{Pr}`.
    """
    t_e = np.asarray(edge_temperature, dtype=np.float64)
    m_e = np.asarray(edge_mach, dtype=np.float64)
    t_w = np.asarray(wall_temperature, dtype=np.float64)
    return np.asarray(
        0.5 * (t_e + t_w) + 0.22 * float(recovery) * 0.5 * (gamma - 1.0) * m_e**2 * t_e
    )


def reference_temperature_of(
    model: str,
    edge_temperature: ArrayLike,
    edge_mach: ArrayLike,
    wall_temperature: ArrayLike,
    recovery: float = 0.843,
    gamma: float = 1.4,
) -> _FloatArray:
    """Dispatch between the two reference-temperature correlations.

    ``"eckert"`` is the default because it is measurably the better of the
    two: against :func:`compressible_blasius` on an adiabatic flat plate it
    is 1.2 % low at Mach 5 and 4.3 % low at Mach 25, where Anderson's
    Eq. 6.159 is 2.3 % and 5.9 % low. Both are within engineering tolerance
    to Mach 10 and both drift the same way — the correlations were fitted at
    moderate supersonic speeds and neither knows about a Mach 25 boundary
    layer.
    """
    if model == "eckert":
        return eckert_reference_temperature(
            edge_temperature, edge_mach, wall_temperature, recovery, gamma
        )
    if model == "anderson":
        return reference_temperature(edge_temperature, edge_mach, wall_temperature)
    msg = f"reference-temperature model must be 'eckert' or 'anderson', got {model!r}"
    raise ValueError(msg)


def laminar_skin_friction(reynolds: ArrayLike, mangler: float = 1.0) -> _FloatArray:
    """:math:`c_f^* = 0.664\\,k/\\sqrt{Re^*_x}` (Anderson Eq. 6.155).

    ``mangler`` is 1 for a flat plate and :math:`\\sqrt{3}` for a cone, where
    the three-dimensional relieving effect thins the boundary layer and
    raises the wall gradient.
    """
    re = np.asarray(reynolds, dtype=np.float64)
    return np.asarray(0.664 * float(mangler) / np.sqrt(np.maximum(re, 1.0)))


def turbulent_skin_friction(reynolds: ArrayLike, mangler: float = 1.0) -> _FloatArray:
    """:math:`c_f^* = 0.0592\\,k/(Re^*_x)^{0.2}` (Anderson Eq. 6.161)."""
    re = np.asarray(reynolds, dtype=np.float64)
    return np.asarray(0.0592 * float(mangler) / np.maximum(re, 1.0) ** 0.2)


# --------------------------------------------------------------------------
# Wall thermal conditions
# --------------------------------------------------------------------------


class WallCondition(Protocol):
    """How the wall temperature is decided at each point of the surface."""

    @property
    def name(self) -> str:
        """A read-only property, not an attribute: every implementation here
        is a frozen dataclass, and a frozen attribute does not satisfy a
        Protocol that declares a settable one."""

    def temperature(
        self,
        edge_temperature: _FloatArray,
        edge_mach: _FloatArray,
        edge_density: _FloatArray,
        edge_speed: _FloatArray,
        edge_pressure: _FloatArray,
        running_length: _FloatArray,
        turbulent: _FloatArray,
    ) -> _FloatArray:  # pragma: no cover
        ...


@dataclass(frozen=True)
class FixedWall:
    """A wall held at one temperature — a cold structure, or a test case."""

    wall_temperature: float = 300.0
    name: str = "fixed wall"

    def temperature(
        self,
        edge_temperature: _FloatArray,
        edge_mach: _FloatArray,
        edge_density: _FloatArray,
        edge_speed: _FloatArray,
        edge_pressure: _FloatArray,
        running_length: _FloatArray,
        turbulent: _FloatArray,
    ) -> _FloatArray:
        return np.full_like(edge_temperature, float(self.wall_temperature))


@dataclass(frozen=True)
class AdiabaticWall:
    """An insulated wall: :math:`T_w = T_{aw}`, and no heat flux.

    The hottest the wall can get from convection alone, and therefore the
    lowest skin friction — a hot wall thickens the boundary layer. Physically
    unreachable on a real vehicle, which radiates, so it brackets rather than
    predicts.
    """

    prandtl: float = PRANDTL_AIR
    gamma: float = 1.4
    name: str = "adiabatic wall"

    def temperature(
        self,
        edge_temperature: _FloatArray,
        edge_mach: _FloatArray,
        edge_density: _FloatArray,
        edge_speed: _FloatArray,
        edge_pressure: _FloatArray,
        running_length: _FloatArray,
        turbulent: _FloatArray,
    ) -> _FloatArray:
        laminar_r = recovery_factor(self.prandtl, turbulent=False)
        turbulent_r = recovery_factor(self.prandtl, turbulent=True)
        blended = laminar_r + (turbulent_r - laminar_r) * turbulent
        return np.asarray(
            edge_temperature * (1.0 + blended * 0.5 * (self.gamma - 1.0) * edge_mach**2)
        )


@dataclass(frozen=True)
class RadiativeEquilibriumWall:
    """:math:`\\epsilon\\sigma T_w^4 = q_w` — what a real hot structure does.

    Convective heating in, radiation out, nothing stored. This is the right
    default for both ends of the flight envelope and it is the reason it is
    worth solving implicitly: the wall temperature enters :math:`T^*`, which
    sets :math:`c_f`, which sets the heating that sets the wall temperature.

    Solved by **vectorised bisection** on :math:`[T_e, T_{aw}]`. The residual
    :math:`\\epsilon\\sigma T_w^4 - q_w(T_w)` is monotonically increasing —
    radiation rises as :math:`T_w^4` while the driving potential
    :math:`T_{aw} - T_w` falls — so the bracket is guaranteed and bisection
    cannot fail. A Newton iteration would be faster per step and would need a
    fallback; over sixty thousand panels the difference is milliseconds and
    the guarantee is worth more.
    """

    emissivity: float = 0.85
    prandtl: float = PRANDTL_AIR
    gamma: float = 1.4
    iterations: int = 60
    reference_model: str = "eckert"
    name: str = "radiative equilibrium wall"

    def __post_init__(self) -> None:
        if not 0.0 < self.emissivity <= 1.0:
            msg = f"emissivity must be in (0, 1], got {self.emissivity}"
            raise ValueError(msg)

    def temperature(
        self,
        edge_temperature: _FloatArray,
        edge_mach: _FloatArray,
        edge_density: _FloatArray,
        edge_speed: _FloatArray,
        edge_pressure: _FloatArray,
        running_length: _FloatArray,
        turbulent: _FloatArray,
    ) -> _FloatArray:
        adiabatic = AdiabaticWall(self.prandtl, self.gamma).temperature(
            edge_temperature,
            edge_mach,
            edge_density,
            edge_speed,
            edge_pressure,
            running_length,
            turbulent,
        )
        gas_constant = edge_pressure / (edge_density * edge_temperature)

        laminar_r = recovery_factor(self.prandtl, turbulent=False)
        turbulent_r = recovery_factor(self.prandtl, turbulent=True)
        blended_recovery = float(np.mean(laminar_r + (turbulent_r - laminar_r) * turbulent))

        def heat_flux(wall: _FloatArray) -> _FloatArray:
            star = reference_temperature_of(
                self.reference_model,
                edge_temperature,
                edge_mach,
                wall,
                blended_recovery,
                self.gamma,
            )
            star = np.maximum(star, 1.0)
            density_star = edge_pressure / (gas_constant * star)
            reynolds = density_star * edge_speed * running_length / _sutherland(star)
            friction = (1.0 - turbulent) * laminar_skin_friction(reynolds) + (
                turbulent * turbulent_skin_friction(reynolds)
            )
            # Reynolds analogy: St = c_f/2 * Pr^{-2/3}.
            stanton = 0.5 * friction * self.prandtl ** (-2.0 / 3.0)
            return np.asarray(
                density_star * edge_speed * stanton * SPECIFIC_HEAT_AIR * (adiabatic - wall)
            )

        def residual(wall: _FloatArray) -> _FloatArray:
            return np.asarray(self.emissivity * STEFAN_BOLTZMANN * wall**4 - heat_flux(wall))

        low = np.minimum(edge_temperature, adiabatic)
        high = np.maximum(adiabatic, low + 1.0)
        for _ in range(int(self.iterations)):
            middle = 0.5 * (low + high)
            increasing = residual(middle) > 0.0
            high = np.where(increasing, middle, high)
            low = np.where(increasing, low, middle)
        return np.asarray(0.5 * (low + high))


# --------------------------------------------------------------------------
# The exact laminar solution, used to check the correlation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BlasiusSolution:
    """Similarity solution of the compressible laminar boundary layer."""

    eta: _FloatArray
    """Transformed wall-normal coordinate."""
    velocity: _FloatArray
    """:math:`f' = u/u_e`."""
    enthalpy: _FloatArray
    """:math:`g = h/h_e`."""
    wall_shear_parameter: float
    """:math:`(C f'')_w`, the quantity :math:`c_f\\sqrt{Re_x}` is built from."""
    wall_heat_parameter: float
    """:math:`(C g'/Pr)_w`; zero for an adiabatic wall."""
    converged: bool

    @property
    def skin_friction_coefficient(self) -> float:
        """:math:`c_f\\sqrt{Re_x}` — the similarity-invariant combination.

        For incompressible flow this is Blasius's 0.664. Everything the
        reference-temperature method claims about the laminar branch is a
        claim about this number.
        """
        return float(np.sqrt(2.0) * self.wall_shear_parameter)


def compressible_blasius(
    edge_mach: float,
    wall_enthalpy_ratio: float | None = None,
    edge_temperature: float = 288.15,
    prandtl: float = PRANDTL_AIR,
    gamma: float = 1.4,
    eta_max: float = 12.0,
    n_nodes: int = 401,
    mach_step: float = 1.0,
) -> BlasiusSolution:
    """Solve the transformed laminar boundary layer from first principles.

    The Illingworth–Levy–Lees form on a flat plate with zero pressure
    gradient:

    .. math::

        (C f'')' + f f'' = 0, \\qquad
        \\left(\\frac{C}{Pr}g'\\right)' + f g'
        + C(\\gamma-1)M_e^2 (f'')^2 = 0

    with :math:`C = \\rho\\mu/(\\rho_e\\mu_e)`, evaluated from Sutherland's law
    at the local temperature rather than assumed unity — the Chapman–Rubesin
    approximation :math:`C = 1` is what makes the usual textbook version
    tractable by hand, and dropping it is the point of solving numerically.

    Parameters
    ----------
    wall_enthalpy_ratio:
        :math:`h_w/h_e`. ``None`` selects an adiabatic wall, imposed as
        :math:`g'(0) = 0`.
    mach_step:
        Largest Mach increment taken by the continuation. The problem is
        stiff at hypersonic speeds — an adiabatic wall at Mach 20 sits at an
        enthalpy ratio near 70 — and a cold start from an analytic guess
        diverges there. Solving up a ladder in Mach and carrying each
        solution forward as the next guess converges to Mach 25 without
        difficulty; setting this larger than about 2 stops being reliable.

    Notes
    -----
    Solved with :func:`scipy.integrate.solve_bvp`, which refines its own mesh;
    ``n_nodes`` sets the initial guess only. The returned
    :attr:`~BlasiusSolution.converged` flag is the solver's, and it is
    returned rather than raised on so that a sweep over a Mach range can
    record where the solution stopped converging instead of stopping.
    """
    m_e = float(edge_mach)
    if not (np.isfinite(m_e) and m_e >= 0.0):
        msg = f"edge Mach must be finite and >= 0, got {edge_mach}"
        raise ValueError(msg)
    pr = float(prandtl)
    adiabatic = wall_enthalpy_ratio is None
    g_wall = 0.0 if wall_enthalpy_ratio is None else float(wall_enthalpy_ratio)
    if not adiabatic and g_wall <= 0.0:
        msg = f"wall_enthalpy_ratio must be > 0, got {wall_enthalpy_ratio}"
        raise ValueError(msg)

    t_e = float(edge_temperature)
    mu_e = float(_sutherland(t_e))

    def chapman_rubesin(g: _FloatArray) -> _FloatArray:
        """:math:`C = \\rho\\mu/(\\rho_e\\mu_e) = \\mu/(\\mu_e\\, T/T_e)`.

        At constant pressure :math:`\\rho/\\rho_e = T_e/T`, so the density
        ratio cancels one power of the temperature ratio.
        """
        temperature = np.maximum(g * t_e, 1.0)
        return np.asarray(_sutherland(temperature) / (mu_e * (temperature / t_e)))

    # State: [f, f', C f'', g, (C/Pr) g']
    def system(mach: float) -> Callable[[_FloatArray, _FloatArray], _FloatArray]:
        def rates(eta: _FloatArray, y: _FloatArray) -> _FloatArray:
            f, f_prime, cf_second, g, cg_prime = y
            c = chapman_rubesin(g)
            f_second = cf_second / c
            g_prime = cg_prime * pr / c
            return np.vstack(
                [
                    f_prime,
                    f_second,
                    -f * f_second,
                    g_prime,
                    -f * g_prime - c * (gamma - 1.0) * mach**2 * f_second**2,
                ]
            )

        return rates

    def boundary(ya: _FloatArray, yb: _FloatArray) -> _FloatArray:
        wall = ya[4] if adiabatic else ya[3] - g_wall
        return np.array([ya[0], ya[1], wall, yb[1] - 1.0, yb[3] - 1.0])

    eta = np.linspace(0.0, float(eta_max), int(n_nodes))
    guess = np.zeros((5, eta.size))
    guess[0] = eta - 1.2 * np.tanh(eta)
    guess[1] = np.tanh(eta)
    guess[2] = 0.47 / np.cosh(eta) ** 2
    guess[3] = np.ones_like(eta) if adiabatic else g_wall + (1.0 - g_wall) * np.tanh(eta)
    guess[4] = 0.0 if adiabatic else (1.0 - g_wall) / np.cosh(eta) ** 2

    # Continuation in Mach. Each rung starts from the previous solution, so
    # the guess is always close and the Newton iteration inside solve_bvp
    # stays in its basin.
    steps = max(int(np.ceil(m_e / max(float(mach_step), 1e-6))), 1)
    ladder = np.linspace(0.0, m_e, steps + 1)[1:] if m_e > 0.0 else np.array([0.0])

    nodes, values, status = eta, guess, 0
    for mach in ladder:
        solution = scipy.integrate.solve_bvp(
            system(float(mach)), boundary, nodes, values, tol=1e-8, max_nodes=200000
        )
        status = int(solution.status)
        if status != 0:
            break
        nodes, values = np.asarray(solution.x), np.asarray(solution.y)

    sampled = np.asarray(scipy.interpolate.PchipInterpolator(nodes, values, axis=1)(eta))
    return BlasiusSolution(
        eta=eta,
        velocity=np.asarray(sampled[1]),
        enthalpy=np.asarray(sampled[3]),
        wall_shear_parameter=float(values[2][0]),
        wall_heat_parameter=float(values[4][0]),
        converged=status == 0,
    )


# --------------------------------------------------------------------------
# The model that gets used
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundaryLayer:
    """Everything needed to turn edge conditions into a wall shear stress.

    Attributes
    ----------
    wall:
        How the wall temperature is decided.
    transition_reynolds:
        ``(start, end)`` of the transition band on running-length Reynolds
        number. The default 1e6 to 5e6 is a smooth slender body in flight,
        an order of magnitude above the 5e5 of a wind-tunnel flat plate;
        flight transition Reynolds numbers are higher because flight
        freestream disturbance levels are lower.
    mangler:
        Laminar three-dimensional factor. :data:`CONE_MANGLER_FACTOR` for a
        body of revolution, 1 for a plate.
    """

    wall: WallCondition = field(default_factory=FixedWall)
    transition_reynolds: tuple[float, float] = (1.0e6, 5.0e6)
    mangler: float = CONE_MANGLER_FACTOR
    prandtl: float = PRANDTL_AIR
    gamma: float = 1.4
    reference_model: str = "eckert"
    """``"eckert"`` or ``"anderson"`` — see :func:`reference_temperature_of`."""

    def __post_init__(self) -> None:
        start, end = self.transition_reynolds
        if not 0.0 < start < end:
            msg = f"transition band must satisfy 0 < start < end, got {self.transition_reynolds}"
            raise ValueError(msg)

    def turbulent_fraction(self, reynolds: ArrayLike) -> _FloatArray:
        """Weight on the turbulent branch, 0 to 1, :math:`C^2` in ``reynolds``."""
        start, end = self.transition_reynolds
        re = np.asarray(reynolds, dtype=np.float64)
        return smoothstep((re - start) / (end - start))

    def skin_friction(
        self,
        edge_temperature: ArrayLike,
        edge_mach: ArrayLike,
        edge_density: ArrayLike,
        edge_speed: ArrayLike,
        edge_pressure: ArrayLike,
        running_length: ArrayLike,
    ) -> tuple[_FloatArray, _FloatArray, _FloatArray]:
        """Local :math:`c_f^*`, wall temperature and wall shear stress.

        Returns
        -------
        tuple
            ``(skin_friction_coefficient, wall_temperature, shear_stress)``.
            The shear stress is :math:`\\tau_w = c_f^*\\,\\tfrac12\\rho^* u_e^2`
            — note :math:`\\rho^*`, not :math:`\\rho_e`: the reference-temperature
            method defines its coefficient on the reference density
            (Anderson Eq. 6.157a), and using the edge density instead would
            overstate the shear by the density ratio, which at Mach 20 is a
            factor of several.
        """
        t_e = np.asarray(edge_temperature, dtype=np.float64)
        m_e = np.asarray(edge_mach, dtype=np.float64)
        rho_e = np.asarray(edge_density, dtype=np.float64)
        u_e = np.asarray(edge_speed, dtype=np.float64)
        p_e = np.asarray(edge_pressure, dtype=np.float64)
        length = np.maximum(np.asarray(running_length, dtype=np.float64), 1.0e-6)

        gas_constant = p_e / (rho_e * t_e)

        # Transition is decided on the *edge* Reynolds number, not the
        # reference one: the reference Reynolds number depends on the wall
        # temperature, which depends on whether the layer is turbulent, and
        # closing that loop would make the transition location a fixed point
        # with no physical content.
        reynolds_edge = rho_e * u_e * length / _sutherland(t_e)
        turbulent = self.turbulent_fraction(reynolds_edge)

        wall = self.wall.temperature(t_e, m_e, rho_e, u_e, p_e, length, turbulent)
        laminar_r = recovery_factor(self.prandtl, turbulent=False)
        turbulent_r = recovery_factor(self.prandtl, turbulent=True)
        recovery = float(np.mean(laminar_r + (turbulent_r - laminar_r) * turbulent))
        star = np.maximum(
            reference_temperature_of(self.reference_model, t_e, m_e, wall, recovery, self.gamma),
            1.0,
        )
        density_star = p_e / (gas_constant * star)
        reynolds_star = density_star * u_e * length / _sutherland(star)

        laminar = laminar_skin_friction(reynolds_star, self.mangler)
        turbulent_branch = turbulent_skin_friction(reynolds_star)
        friction = (1.0 - turbulent) * laminar + turbulent * turbulent_branch
        shear = friction * 0.5 * density_star * u_e**2
        return np.asarray(friction), np.asarray(wall), np.asarray(shear)
