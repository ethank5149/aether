"""Taylor–Maccoll conical flow: an exact answer to check approximate ones with.

Supersonic flow over a sharp circular cone at zero incidence is one of the
very few compressible flows with an exact solution, and it is exact in the
strong sense — no linearisation, no thin-body assumption, no perfect-gas
approximation beyond a constant :math:`\\gamma`. Because the flow is conical,
every quantity depends on the polar angle alone, the partial differential
equations collapse to a single ordinary one, and what remains is quadrature.

That makes it the right instrument for validating a CFD pipeline. A
finite-volume Euler solver on an unstructured mesh and a shooting method on
a second-order ODE share no code, no discretisation and no formulation; if
they agree on the shock angle and the surface pressure to a few tenths of a
per cent, both are probably right, and if they do not, exactly one of them is
wrong and the ODE is the one with fewer places to hide.

The equation
------------

With :math:`V' = V/V_{\\max}` and :math:`f(\\theta) = V_r'`, so that
:math:`f' = V_\\theta'`,

.. math::

    \\frac{\\gamma-1}{2}\\left(1 - f^2 - f'^2\\right)
    \\left(2f + f'\\cot\\theta + f''\\right) = f'\\left(f f' + f' f''\\right)

which rearranges to an explicit second derivative. Integration runs
**inward** from the shock: the shock angle is guessed, the oblique-shock jump
gives :math:`f` and :math:`f'` there, and the surface is wherever
:math:`f' = 0` — the cone is a streamline, so the polar velocity component
vanishes on it. The outer loop is a root find on the shock angle for the
requested cone angle.

Integrating inward rather than outward is not a preference. The surface
condition :math:`f' = 0` makes :math:`\\theta` a coordinate singularity of
the outward problem, and starting there loses the leading digits before the
integration begins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.integrate
import scipy.optimize

__all__ = [
    "ConeSolution",
    "ObliqueShock",
    "mach_angle",
    "oblique_shock",
    "solve_cone",
    "wedge_shock_angle",
]


def mach_angle(mach: float) -> float:
    """:math:`\\mu = \\arcsin(1/M)` (rad) — the weakest possible shock angle."""
    m = float(mach)
    if not (np.isfinite(m) and m > 1.0):
        msg = f"Mach angle needs supersonic flow, got {mach}"
        raise ValueError(msg)
    return float(np.arcsin(1.0 / m))


@dataclass(frozen=True)
class ObliqueShock:
    """Jump across a straight oblique shock."""

    mach: float
    """Upstream Mach number."""
    angle: float
    """Shock angle :math:`\\beta` from the freestream (rad)."""
    deflection: float
    """Flow deflection :math:`\\delta` through the shock (rad)."""
    downstream_mach: float
    pressure_ratio: float
    density_ratio: float
    temperature_ratio: float


def oblique_shock(mach: float, angle: float, gamma: float = 1.4) -> ObliqueShock:
    """Jump conditions across an oblique shock at a given wave angle."""
    m1 = float(mach)
    beta = float(angle)
    g = float(gamma)
    if not (np.isfinite(m1) and m1 > 1.0):
        msg = f"oblique shock needs supersonic upstream flow, got {mach}"
        raise ValueError(msg)
    normal = m1 * np.sin(beta)
    if normal < 1.0:
        msg = (
            f"shock angle {np.rad2deg(beta):.4f} deg is below the Mach angle "
            f"{np.rad2deg(mach_angle(m1)):.4f} deg at Mach {m1:g}; the normal "
            f"component {normal:.4f} is subsonic and there is no shock"
        )
        raise ValueError(msg)
    normal2 = normal * normal
    deflection = float(
        np.arctan(
            2.0
            / np.tan(beta)
            * (normal2 - 1.0)
            / (m1 * m1 * (g + np.cos(2.0 * beta)) + 2.0)
        )
    )
    downstream_normal = np.sqrt(
        (1.0 + 0.5 * (g - 1.0) * normal2) / (g * normal2 - 0.5 * (g - 1.0))
    )
    pressure_ratio = 1.0 + 2.0 * g / (g + 1.0) * (normal2 - 1.0)
    density_ratio = (g + 1.0) * normal2 / ((g - 1.0) * normal2 + 2.0)
    return ObliqueShock(
        mach=m1,
        angle=beta,
        deflection=deflection,
        downstream_mach=float(downstream_normal / np.sin(beta - deflection)),
        pressure_ratio=float(pressure_ratio),
        density_ratio=float(density_ratio),
        temperature_ratio=float(pressure_ratio / density_ratio),
    )


def wedge_shock_angle(
    mach: float, deflection: float, gamma: float = 1.4, weak: bool = True
) -> float:
    """Shock angle on a two-dimensional wedge (rad).

    Here so that the cone can be compared against it: at the same turning
    angle a cone's shock is always the *weaker* of the two, because the flow
    behind a conical shock keeps expanding as it moves away from the tip
    while the flow behind a wedge shock does not. That inequality is a
    qualitative check on :func:`solve_cone` that needs no tables.
    """
    m = float(mach)
    target = float(deflection)
    mu = mach_angle(m)
    detachment = scipy.optimize.minimize_scalar(
        lambda b: -oblique_shock(m, b, gamma).deflection,
        bounds=(mu + 1e-9, 0.5 * np.pi - 1e-9),
        method="bounded",
        options={"xatol": 1e-12},
    )
    peak = float(detachment.x)
    if -float(detachment.fun) < target:
        msg = (
            f"deflection {np.rad2deg(target):.3f} deg exceeds the detachment "
            f"angle {np.rad2deg(-float(detachment.fun)):.3f} deg at Mach {m:g}; "
            f"the shock is detached and no attached solution exists"
        )
        raise ValueError(msg)
    bracket = (mu + 1e-12, peak) if weak else (peak, 0.5 * np.pi - 1e-12)
    return float(
        scipy.optimize.brentq(
            lambda b: oblique_shock(m, b, gamma).deflection - target,
            *bracket,
            xtol=1e-14,
        )
    )


@dataclass(frozen=True)
class ConeSolution:
    """Exact conical flow over a sharp cone at zero incidence."""

    mach: float
    """Freestream Mach number."""
    cone_angle: float
    """Cone half-angle (rad)."""
    shock_angle: float
    """Conical shock angle (rad)."""
    surface_mach: float
    """Mach number on the cone surface."""
    surface_pressure_ratio: float
    """:math:`p_c/p_\\infty`."""
    gamma: float
    converged: bool

    @property
    def pressure_coefficient(self) -> float:
        """:math:`C_p = \\frac{2}{\\gamma M^2}(p_c/p_\\infty - 1)` on the cone."""
        return float(
            2.0
            / (self.gamma * self.mach**2)
            * (self.surface_pressure_ratio - 1.0)
        )

    @property
    def wave_drag_coefficient(self) -> float:
        """:math:`C_D` on the cone's base area, pressure only.

        For a cone the surface pressure is uniform, so this is just
        :math:`C_p` — the axial projection of a uniform pressure over a cone
        equals the pressure times the base area. Stated separately because it
        is the number a drag comparison wants and the identity is easy to
        doubt.
        """
        return self.pressure_coefficient


def _velocity_ratio(mach: float, gamma: float) -> float:
    """:math:`V/V_{\\max} = [1 + 2/((\\gamma-1)M^2)]^{-1/2}`."""
    return float(1.0 / np.sqrt(1.0 + 2.0 / ((gamma - 1.0) * mach * mach)))


def _mach_from_velocity_ratio(ratio: float, gamma: float) -> float:
    """Inverse of :func:`_velocity_ratio`."""
    r2 = float(ratio) ** 2
    return float(np.sqrt(2.0 * r2 / ((gamma - 1.0) * (1.0 - r2))))


def _integrate_from_shock(
    mach: float, shock_angle: float, gamma: float
) -> tuple[float, float, bool]:
    """Integrate inward from a shock; return the surface angle and velocity.

    Returns ``(theta_surface, f_surface, found)``. ``found`` is False when the
    integration reached the axis without :math:`f'` changing sign, which is
    what happens for a shock angle too weak to support any cone.
    """
    jump = oblique_shock(mach, shock_angle, gamma)
    ratio = _velocity_ratio(jump.downstream_mach, gamma)
    # The flow behind the shock is turned by `deflection` from the freestream,
    # so its angle to the local radial direction at theta = beta is
    # beta - deflection.
    swept = shock_angle - jump.deflection
    initial = np.array([ratio * np.cos(swept), -ratio * np.sin(swept)])

    def rates(theta: float, y: np.ndarray) -> np.ndarray:
        f, df = y
        a = 0.5 * (gamma - 1.0) * (1.0 - f * f - df * df)
        denominator = a - df * df
        # The denominator vanishes where the local Mach number reaches one in
        # the polar direction, which cannot happen inside a valid conical
        # field; the floor keeps a probing root-finder from producing a NaN
        # instead of a large residual.
        if abs(denominator) < 1e-14:
            denominator = np.copysign(1e-14, denominator)
        second = (df * df * f - a * (2.0 * f + df / np.tan(theta))) / denominator
        return np.array([df, second])

    def surface(theta: float, y: np.ndarray) -> float:
        return float(y[1])

    surface.terminal = True  # type: ignore[attr-defined]
    surface.direction = 0.0  # type: ignore[attr-defined]

    solution = scipy.integrate.solve_ivp(
        rates,
        (shock_angle, 1e-6),
        initial,
        events=surface,
        rtol=1e-11,
        atol=1e-13,
        dense_output=True,
        method="DOP853",
    )
    if solution.t_events[0].size == 0:
        return 0.0, 0.0, False
    theta_surface = float(solution.t_events[0][0])
    f_surface = float(solution.y_events[0][0][0])
    return theta_surface, f_surface, True


def maximum_cone_angle(mach: float, gamma: float = 1.4) -> tuple[float, float]:
    """Largest cone the flow will stay attached to, and its shock angle (rad).

    Returns ``(cone_angle, shock_angle)``. Past this angle the shock detaches
    and the flow stops being conical, so the Taylor–Maccoll solution stops
    existing rather than becoming inaccurate.

    This is also what makes :func:`solve_cone` need it: the surface angle is
    **not monotone** in the shock angle. It rises from zero at the Mach
    angle to this maximum and falls again toward the normal-shock limit, so
    every attached cone has two solutions — a weak shock and a strong one —
    and a root find over the whole range would land on whichever the
    bisection happened to bracket.
    """
    mu = mach_angle(float(mach))

    def surface_angle(beta: float) -> float:
        return _integrate_from_shock(float(mach), beta, float(gamma))[0]

    peak = scipy.optimize.minimize_scalar(
        lambda beta: -surface_angle(beta),
        bounds=(mu + 1e-7, 0.5 * np.pi - 1e-7),
        method="bounded",
        options={"xatol": 1e-11},
    )
    return float(-float(peak.fun)), float(peak.x)


def solve_cone(
    mach: float, cone_angle: float, gamma: float = 1.4, weak: bool = True
) -> ConeSolution:
    """Exact Taylor–Maccoll solution for a sharp cone at zero incidence.

    Parameters
    ----------
    mach:
        Freestream Mach number, strictly supersonic.
    cone_angle:
        Cone half-angle (rad).
    weak:
        Select the weak-shock branch. Both branches are mathematically valid
        Taylor–Maccoll solutions; the weak one is what an unconfined external
        flow produces and is the physical answer unless downstream pressure
        forces the strong one.

    Raises
    ------
    ValueError
        If no attached conical shock exists at that Mach and angle. This is a
        real physical outcome — the shock detaches past a Mach-dependent
        limit given by :func:`maximum_cone_angle` — and it is raised rather
        than returned as a nearby answer.
    """
    m = float(mach)
    theta_c = float(cone_angle)
    g = float(gamma)
    if not (np.isfinite(m) and m > 1.0):
        msg = f"conical flow needs supersonic freestream, got {mach}"
        raise ValueError(msg)
    if not 0.0 < theta_c < 0.5 * np.pi:
        msg = f"cone half-angle must be in (0, 90) deg, got {np.rad2deg(theta_c):g}"
        raise ValueError(msg)

    def residual(beta: float) -> float:
        return _integrate_from_shock(m, beta, g)[0] - theta_c

    mu = mach_angle(m)
    detached_angle, detached_shock = maximum_cone_angle(m, g)
    if theta_c > detached_angle:
        msg = (
            f"no attached conical shock at Mach {m:g} for a "
            f"{np.rad2deg(theta_c):.3f} deg cone: the shock detaches above "
            f"{np.rad2deg(detached_angle):.3f} deg at this Mach number"
        )
        raise ValueError(msg)

    bracket = (
        (mu + 1e-12, detached_shock)
        if weak
        else (detached_shock, 0.5 * np.pi - 1e-9)
    )
    beta = float(scipy.optimize.brentq(residual, *bracket, xtol=1e-13, rtol=1e-14))
    theta_surface, f_surface, found = _integrate_from_shock(m, beta, g)
    jump = oblique_shock(m, beta, g)
    surface_mach = _mach_from_velocity_ratio(f_surface, g)
    # Isentropic from just behind the shock to the surface: the flow is
    # shocked once and then compresses smoothly, so stagnation pressure is
    # constant between the two.
    isentropic = (
        (1.0 + 0.5 * (g - 1.0) * jump.downstream_mach**2)
        / (1.0 + 0.5 * (g - 1.0) * surface_mach**2)
    ) ** (g / (g - 1.0))
    return ConeSolution(
        mach=m,
        cone_angle=theta_c,
        shock_angle=beta,
        surface_mach=surface_mach,
        surface_pressure_ratio=float(jump.pressure_ratio * isentropic),
        gamma=g,
        converged=found and abs(theta_surface - theta_c) < 1e-9,
    )
