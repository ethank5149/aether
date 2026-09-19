r"""The maximum-range entry as an optimal control problem.

This is where three separately built pieces meet: the six-state entry dynamics
of :func:`aether.guidance.entry.entry_dynamics`, the direct transcription in
:mod:`aether.optimal_control.pseudospectral`, and the Pontryagin refinement in
:mod:`aether.optimal_control.indirect`. A prescribed bank history -- a steering
law, or a swept schedule -- answers "where does the body end up"; this module
asks the opposite question: what bank history is optimal, and how far can the
body actually go.

The answer bounds the entry footprint. Sweeping prescribed schedules can only
find what those schedules happen to reach, which is an *inner* bound; an
optimised trajectory gives the outer edge directly, so the two together bracket
the true footprint.

Scaling
-------

The six states span seven orders of magnitude -- radius near
:math:`6.4\times10^6` m, flight path angle near :math:`10^{-2}` rad -- and
SLSQP works on the raw variables, so an unscaled problem asks it to resolve a
part in :math:`10^9` of the radius to see a degree of pitch. States are
therefore carried as :math:`\tilde r = r/R_E` and :math:`\tilde V = V/V_c` with
:math:`V_c=\sqrt{gR_E}`, the angles left alone, and the dynamics wrapped as
:math:`\tilde f(\tilde x, u) = S^{-1} f(S\tilde x, u)`.

**Time is deliberately left in seconds.** The canonical entry
non-dimensionalisation scales it too, by :math:`R_E/V_c \approx 807` s, and
that would collide with the transcription's floor of ``t0 + 1.0`` on the final
time (:mod:`~aether.optimal_control.pseudospectral`, ``lb[-1]``) -- a bound
meant to keep the time scale positive would silently become a 13-minute minimum
flight. Scaling the states and not the clock keeps both well conditioned.

Objective
---------

Maximum range is written as a running cost rather than a terminal one, because
the transcription integrates :math:`L` and has no Mayer term:

.. math::

    J = -\int_0^{t_f} \frac{V\cos\gamma}{R_E}\,\mathrm{d}t,

whose value is the great-circle arc flown, in Earth radii. The horizon closes on
a terminal *speed* rather than a terminal time, so the body flies until its
energy is spent and :math:`t_f` is free.

Validation without a closed form
--------------------------------

There is no analytic optimum for a constrained lifting entry, so correctness is
argued three ways, none of which the solver enforces:

* **Hamiltonian constancy.** The dynamics carry no explicit time dependence, so
  :math:`H` is conserved along an optimum. This is what caught the costate error
  in the two-state case.
* **The Sanger equilibrium-glide range**, which a swept prescribed
  schedule should reproduce closely. An optimised unconstrained entry must land in the
  same place; :func:`sanger_range` states it for comparison.
* **Cost monotonicity.** The indirect refinement is seeded from the direct
  solution and must not come back worse.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.integrate
from numpy.typing import NDArray

from aether.aerothermal.stagnation import sutton_graves
from aether.guidance.entry import (
    EntryBody,
    EntryState,
    atmospheric_density,
    entry_dynamics,
)
from aether.optimal_control.pseudospectral import OCPProblem, lgl_nodes

_FloatArray = NDArray[np.float64]

_MU = 3.986004418e14
_R_EARTH = 6378137.0
_G0 = 9.80665
#: Circular speed at the surface, the natural speed scale for an entry.
_V_C = float(np.sqrt(_MU / _R_EARTH))

__all__ = [
    "EntryPathLimits",
    "EntryScaling",
    "bank_argmin",
    "entry_path_constraints",
    "flown_guess",
    "g_load",
    "max_range_problem",
    "sanger_range",
    "stagnation_heat_flux",
]


@dataclass(frozen=True)
class EntryScaling:
    """Non-dimensionalisation of the entry state.

    Only the two dimensional states are scaled; longitude, latitude, flight
    path angle and heading are radians and already of order one. Time stays in
    seconds -- see the module docstring for why.
    """

    length: float = _R_EARTH
    speed: float = _V_C

    @property
    def factors(self) -> _FloatArray:
        """Diagonal of :math:`S`, so ``x = factors * x_scaled``."""
        return np.array([self.length, 1.0, 1.0, self.speed, 1.0, 1.0])

    def to_physical(self, scaled: _FloatArray) -> _FloatArray:
        return np.asarray(scaled, dtype=np.float64) * self.factors

    def to_scaled(self, physical: _FloatArray) -> _FloatArray:
        return np.asarray(physical, dtype=np.float64) / self.factors

    def dynamics(
        self, body: EntryBody, *, body_radius: float = _R_EARTH
    ) -> Callable[[_FloatArray, _FloatArray], _FloatArray]:
        r""":math:`\tilde f(\tilde x, u) = S^{-1} f(S\tilde x, u)`, as ``f(x, u)``."""
        factors = self.factors

        def scaled_dynamics(x: _FloatArray, u: _FloatArray) -> _FloatArray:
            physical = np.asarray(x, dtype=np.float64) * factors
            return entry_dynamics(physical, u, body, body_radius=body_radius) / factors

        return scaled_dynamics


@dataclass(frozen=True)
class EntryPathLimits:
    """Pointwise limits an optimised entry must respect at every node.

    Trajectory-level checks take whole flown histories and report a peak. A
    transcription needs the same physics evaluated at a single state, so it is
    stated here pointwise.

    One difference is deliberate and worth stating: those checkers evaluate
    against ``LayeredAtmosphere`` while :func:`~aether.guidance.entry.entry_dynamics`
    flies the exponential atmosphere of
    :func:`~aether.guidance.entry.atmospheric_density`. Mixing the two
    inside one optimisation would let the optimiser satisfy a heat-flux limit in
    an atmosphere it is not flying through, which is a good way to produce a
    trajectory that is feasible on paper and not in the air. These constraints
    use the same atmosphere as the dynamics.

    Attributes
    ----------
    max_heat_flux_w_cm2:
        Stagnation-point convective flux ceiling (W/cm^2), Sutton-Graves.
    max_g_load:
        Deceleration ceiling in units of standard gravity.
    min_altitude_m:
        Floor below which the trajectory has left the atmospheric phase.
    nose_radius_m:
        Effective stagnation radius. Not a property of
        :class:`~aether.guidance.entry.EntryBody`, which is a point
        mass, so it is carried with the limit that needs it.
    """

    max_heat_flux_w_cm2: float = 500.0
    max_g_load: float = 30.0
    min_altitude_m: float = 20_000.0
    nose_radius_m: float = 0.1

    def __post_init__(self) -> None:
        for name in ("max_heat_flux_w_cm2", "max_g_load", "nose_radius_m"):
            value = float(getattr(self, name))
            if not (np.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and > 0, got {value}")


def stagnation_heat_flux(
    altitude: float, speed: float, nose_radius_m: float
) -> float:
    r"""Sutton-Graves convective stagnation flux (W/cm^2).

    :math:`\dot q = k\sqrt{\rho/R_n}\,V^3`, converted from W/m^2. Radiative
    heating is omitted: below about 7 km/s it is a few percent of convective,
    and the Tauber-Sutton correlation used elsewhere is not differentiable
    enough to hand an SQP solver. At lunar-return speeds (about 11 km/s)
    radiative heating is a large fraction of the total and this limit
    under-counts it.
    """
    density = max(float(atmospheric_density(altitude)), 1e-12)
    flux = float(sutton_graves(density, nose_radius_m, max(float(speed), 1.0)))
    return flux / 1.0e4


def g_load(body: EntryBody, altitude: float, speed: float) -> float:
    """Aerodynamic deceleration in units of standard gravity."""
    return body.drag_acceleration(altitude, speed) / _G0


def entry_path_constraints(
    body: EntryBody,
    limits: EntryPathLimits,
    scaling: EntryScaling,
    *,
    body_radius: float = _R_EARTH,
) -> Callable[[_FloatArray, _FloatArray], _FloatArray]:
    """Path constraints ``g(x, u) >= 0`` on the scaled state.

    Each row is the *remaining margin* against one limit, normalised by that
    limit so all three are of order one and positive while feasible. The
    normalisation matters: an SQP solver weights constraint violations against
    each other, and a heat flux measured in hundreds against an altitude
    measured in tens of thousands would make the altitude floor the only one it
    really sees.

    The sign convention is the one the transcription enforces --
    non-negative-is-feasible -- which is *not* what
    :class:`~aether.optimal_control.pseudospectral.OCPProblem` documented when
    this was first written. Getting it backwards does not raise: it turns every
    ceiling into a floor, and the first version of this function asked the
    optimiser to fly a trajectory that exceeded 500 W/cm^2 and 30 g.
    """
    factors = scaling.factors

    def constraints(x: _FloatArray, _u: _FloatArray) -> _FloatArray:
        physical = np.asarray(x, dtype=np.float64) * factors
        altitude = float(physical[0] - body_radius)
        speed = float(physical[3])
        return np.array([
            1.0
            - stagnation_heat_flux(altitude, speed, limits.nose_radius_m)
            / limits.max_heat_flux_w_cm2,
            1.0 - g_load(body, altitude, speed) / limits.max_g_load,
            (altitude - limits.min_altitude_m) / limits.min_altitude_m,
        ])

    return constraints


def sanger_range(body: EntryBody, speed: float) -> float:
    r"""Equilibrium-glide range (m) from Sanger's closed form.

    .. math::

        R = \frac{L}{D}\,\frac{R_E}{2}\,
            \ln\!\left[\frac{1}{1-(V/V_c)^2}\right]

    The reference the optimised solution is checked against. It assumes an
    unconstrained equilibrium glide flown at zero bank all the way to zero
    speed, so it is an *upper* bound on any real trajectory: a body that must
    bank to steer, respect a heat-flux limit, or stop at a finite terminal speed
    flies shorter. A solved maximum-range trajectory landing above this number would
    mean the transcription is admitting something the physics does not.
    """
    ratio = (float(speed) / _V_C) ** 2
    if not 0.0 <= ratio < 1.0:
        raise ValueError(
            f"speed must lie in [0, {_V_C:.1f}) m/s for an equilibrium glide; "
            f"at circular speed the glide does not descend and the range diverges. "
            f"Got {speed}"
        )
    return float(body.lift_to_drag * 0.5 * _R_EARTH * np.log(1.0 / (1.0 - ratio)))


def bank_argmin(
    body: EntryBody,
    scaling: EntryScaling,
    max_bank: float,
    *,
    body_radius: float = _R_EARTH,
) -> Callable[[_FloatArray, _FloatArray], _FloatArray]:
    r"""Closed-form minimiser of the Hamiltonian over bank.

    Bank enters the entry dynamics in exactly two places -- :math:`\cos\sigma`
    in :math:`\dot\gamma` and :math:`\sin\sigma` in :math:`\dot\psi` -- and
    nowhere in the running cost, which depends on speed and flight path angle
    alone. So the whole bank dependence of :math:`H = L + \lambda^{\mathsf T}f`
    collapses to

    .. math::

        H(\sigma) = \text{const} + A\cos\sigma + B\sin\sigma,
        \qquad A = \frac{L\lambda_\gamma}{V},
        \qquad B = \frac{L\lambda_\psi}{V\cos\gamma},

    whose unconstrained minimiser is :math:`\sigma^\star =
    \operatorname{atan2}(-B, -A)`. That is the plan's own contingency, and it is
    worth taking: :func:`~aether.optimal_control.indirect.minimising_control`
    is called at every mesh node on every iteration of the boundary value
    solve, so a per-node BFGS over six states is the cost of the indirect stage
    rather than an overhead on it.

    The bound is applied by comparing candidates rather than by clipping.
    :math:`A\cos\sigma + B\sin\sigma` is a sinusoid, not a convex function, so
    an interior stationary point outside :math:`[-\sigma_{\max},
    \sigma_{\max}]` does *not* imply the constrained minimum sits at the
    nearest bound -- clipping would silently return a maximum on the arcs where
    the costates change sign. The endpoints are evaluated alongside the
    stationary point and the best is returned, which is the minimum principle
    rather than a projected stationarity condition.
    """
    factors = scaling.factors

    def argmin(x: _FloatArray, costate: _FloatArray) -> _FloatArray:
        physical = np.asarray(x, dtype=np.float64) * factors
        radius, speed, gamma = physical[0], physical[3], physical[4]
        lift = body.lift_to_drag * body.drag_acceleration(
            radius - body_radius, speed
        )
        # The costates are in scaled coordinates, so the coefficients pick up
        # the same factors the scaled dynamics divide by.
        a = lift * float(costate[4]) / speed
        b = lift * float(costate[5]) / (speed * float(np.cos(gamma)))
        candidates = [
            float(np.arctan2(-b, -a)),
            -float(max_bank),
            float(max_bank),
        ]
        admissible = [
            c for c in candidates if -max_bank - 1e-12 <= c <= max_bank + 1e-12
        ]
        best = min(admissible, key=lambda s: a * np.cos(s) + b * np.sin(s))
        return np.array([best])

    return argmin


def flown_guess(
    body: EntryBody,
    initial: EntryState,
    terminal_speed: float,
    *,
    bank: float = 0.0,
    n_nodes: int = 20,
    scaling: EntryScaling | None = None,
    max_time: float = 4000.0,
    body_radius: float = _R_EARTH,
) -> tuple[_FloatArray, _FloatArray, float]:
    """A warm start built by actually flying the body at constant bank.

    Integrates the same dynamics the transcription collocates, stops on the
    terminal speed, and resamples onto the LGL nodes. The result already
    satisfies the dynamics, so the solver starts from a feasible point and
    spends its iterations on optimality rather than on finding the manifold.

    Constant bank is deliberate: it is the trajectory whose range the Sanger
    closed form describes, so the optimiser starts from the analytic reference
    and whatever it gains over it is the value of steering.
    """
    scaling = scaling or EntryScaling()
    # Checked up front rather than after integrating. The terminating event is
    # a downward crossing of the terminal speed, so a body that starts below
    # it never triggers -- the integration runs to `max_time` and returns a
    # full-length trajectory for an entry that was over before it began.
    if initial.speed <= float(terminal_speed):
        raise ValueError(
            f"initial speed {initial.speed} m/s is at or below the terminal "
            f"speed of {terminal_speed} m/s, so there is nothing to optimise"
        )
    nodes, _ = lgl_nodes(n_nodes + 1)

    def derivatives(_t: float, y: _FloatArray) -> _FloatArray:
        return entry_dynamics(y, bank, body, body_radius=body_radius)

    def spent(_t: float, y: _FloatArray) -> float:
        return float(y[3]) - float(terminal_speed)

    spent.terminal = True  # type: ignore[attr-defined]
    spent.direction = -1.0  # type: ignore[attr-defined]

    flight = scipy.integrate.solve_ivp(
        derivatives, (0.0, max_time), initial.as_array(),
        events=spent, rtol=1e-9, atol=1e-9, dense_output=True,
    )
    if not flight.success:
        raise RuntimeError(f"constant-bank guess failed to integrate: {flight.message}")
    tf = float(flight.t[-1])
    if tf <= 0.0:
        raise ValueError(
            f"the body is already at or below the terminal speed of "
            f"{terminal_speed} m/s, so there is nothing to optimise"
        )

    # LGL nodes live on [-1, 1]; map them onto the flown interval.
    times = 0.5 * (np.asarray(nodes, dtype=np.float64) + 1.0) * tf
    states = np.array([scaling.to_scaled(flight.sol(t)) for t in times])
    controls = np.full((n_nodes + 1, 1), float(bank))
    return states, controls, tf


def max_range_problem(
    body: EntryBody,
    initial: EntryState,
    terminal_speed: float,
    *,
    limits: EntryPathLimits | None = None,
    scaling: EntryScaling | None = None,
    crossrange_m: float | None = None,
    n_nodes: int = 20,
    tf_guess: float = 1200.0,
    body_radius: float = _R_EARTH,
) -> OCPProblem:
    """The maximum-range entry, transcribed.

    Bank is the single control, bounded by the body's own ``max_bank``. The
    terminal condition is the speed alone -- every other component of the
    terminal state is left free, which is the point: where the body ends up
    is the answer, not the question.

    ``terminal_speed`` closes the horizon. Without it the free final time has
    nothing to push against, since the objective rewards flying longer and the
    only thing that stops the flight is running out of energy.

    ``crossrange_m`` is what makes this an optimisation rather than a
    formality. **Without it the answer is known in advance: zero bank.** Every
    degree of bank tips lift out of the vertical, so an unconstrained
    maximum-range entry flies wings-level, which is precisely the trajectory
    Sanger's closed form describes and precisely the trajectory
    :func:`flown_guess` already supplies. A solver handed its own optimum has
    nothing to do and says so -- SLSQP stops on "positive directional
    derivative for linesearch" without moving, which reads as a failure and is
    really a fixed point.

    Demanding a lateral offset breaks that degeneracy. The body must now
    bank, and *how* it banks matters: turning early costs range for the whole
    remaining flight while turning late has less speed to work with, so the
    optimum is a genuine bank history that no constant bank reproduces. That is
    the comparison worth making, and it needs no closed form -- the optimised
    trajectory must reach the same crossrange as the best constant bank while flying
    measurably further.

    Passed as a signed displacement in metres along the initial local
    horizontal normal to the flight path, converted to a terminal latitude
    target. This assumes the entry begins on the equator heading east, the
    natural frame for a footprint.
    """
    limits = limits or EntryPathLimits()
    scaling = scaling or EntryScaling()

    x0 = scaling.to_scaled(initial.as_array())
    target = np.full(6, np.nan)
    target[3] = float(terminal_speed) / scaling.speed
    if crossrange_m is not None:
        target[2] = float(crossrange_m) / _R_EARTH

    length = scaling.length

    def running_cost(x: _FloatArray, _u: _FloatArray) -> float:
        """Negative ground-track rate: minimising it maximises range flown."""
        speed = float(x[3]) * scaling.speed
        return -speed * float(np.cos(x[4])) / length

    return OCPProblem(
        dynamics=scaling.dynamics(body, body_radius=body_radius),
        path_constraints=entry_path_constraints(
            body, limits, scaling, body_radius=body_radius
        ),
        x0=x0,
        xf_target=target,
        nx=6,
        nu=1,
        N=n_nodes,
        t0=0.0,
        tf_guess=tf_guess,
        u_bounds=(np.array([-body.max_bank]), np.array([body.max_bank])),
        running_cost=running_cost,
        time_weight=0.0,
        control_argmin=bank_argmin(
            body, scaling, body.max_bank, body_radius=body_radius
        ),
    )
