"""Real-world constraint builders — what separates a physical trajectory from a curve.

A bare ``dynamics + boundary + objective`` optimal-control problem is
underspecified against its real counterpart: nothing stops it flying forever,
pulling infinite g, diving through the atmosphere until the airframe fails, or
an interceptor arriving after the target has already struck. These builders add
the missing physics as composable constraints, reusing the modules that already
own each quantity (``systems.mass``, ``aerothermal``, ``hypersonic``,
``atmosphere``) rather than re-deriving it.

Two flavours, because the two kinds of real constraint transcribe differently:

* **Path constraints** hold at every collocation node — ``g(x, u) >= 0`` when
  satisfied (a *margin*: the convention the pseudospectral backend enforces).
  Attach these to a :class:`~aether.trajectory.problem.PhaseSpec`.
  Max-Q, q-alpha, g-load, heat flux, no-fly zones, the reentry corridor, and
  propellant non-negativity are path constraints. :func:`combine` stacks several
  into one callback.

* **Trajectory constraints** are functionals of the *whole* trajectory (and, for
  the two-sided ones, the opposing trajectory): a returned scalar margin,
  ``>= 0`` when feasible. Divert-budget, intercept-before-defended-asset, seeker
  acquisition, launch window. These are checked by the generator / engagement
  layer, not the single-phase transcription.

Physical quantities the caller must supply (``dynamic_pressure_of``,
``heat_flux_of``, ``angle_of_attack_of``, ...) are passed in, so a builder stays
decoupled from any one state layout or atmosphere — exactly as the objectives do.
The mass state and Tsiolkovsky depletion, which is what actually bounds range,
energy, and time, are provided as dynamics/linkage helpers here too.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from aether.trajectory.representation import ReferenceTrajectory

_FloatArray = NDArray[np.float64]

#: ``g(x, u) -> margins`` with each entry ``>= 0`` when the constraint holds.
PathConstraint = Callable[[_FloatArray, _FloatArray], _FloatArray]
#: ``f(x) -> value`` reading a scalar physical quantity from a state.
StateScalar = Callable[[_FloatArray], float]

__all__ = [
    "PathConstraint",
    "combine",
    "divert_budget",
    "g_load_limit",
    "heat_flux_limit",
    "heat_load_limit",
    "intercept_before",
    "launch_window",
    "mass_depletion_dynamics",
    "max_dynamic_pressure",
    "no_fly_zone",
    "propellant_nonnegative",
    "q_alpha_limit",
    "reentry_corridor",
    "seeker_acquisition",
    "staging_mass_drop",
    "terminal_flight_path_angle",
    "terminal_speed_at_least",
]


# =====================================================================
#  Mass and propellant — the constraint that bounds range, energy, time
# =====================================================================

def mass_depletion_dynamics(
    base_dynamics: Callable[[_FloatArray, _FloatArray], _FloatArray],
    thrust_magnitude: Callable[[_FloatArray, _FloatArray], float],
    isp: float,
    mass_index: int,
    g0: float = 9.80665,
) -> Callable[[_FloatArray, _FloatArray], _FloatArray]:
    """Augment a dynamics function with Tsiolkovsky mass depletion.

    ``base_dynamics(x, u)`` returns the derivative of every state *except* mass;
    this wraps it so the mass component at ``mass_index`` decreases as
    ``mdot = -||T|| / (Isp * g0)``. Carrying mass as a state — rather than
    assuming it away — is what makes propellant finite, and finite propellant is
    what gives every objective a bounded optimum (POST/OTIS carry mass as a
    state for exactly this reason).
    """
    ve = isp * g0

    def dynamics(x: _FloatArray, u: _FloatArray) -> _FloatArray:
        xdot = np.array(base_dynamics(x, u), dtype=np.float64)
        xdot[mass_index] = -float(thrust_magnitude(x, u)) / ve
        return xdot

    return dynamics


def staging_mass_drop(
    dry_mass_dropped: float, mass_index: int
) -> Callable[[_FloatArray], _FloatArray]:
    """Phase linkage that jettisons a spent stage's dry mass at separation."""

    def linkage(x_terminal: _FloatArray) -> _FloatArray:
        x = np.array(x_terminal, dtype=np.float64)
        x[mass_index] = x[mass_index] - dry_mass_dropped
        return x

    return linkage


def propellant_nonnegative(dry_mass: float, mass_index: int) -> PathConstraint:
    """Mass may not fall below the stage dry mass (no negative propellant)."""

    def constraint(x: _FloatArray, u: _FloatArray) -> _FloatArray:
        return np.array([x[mass_index] - dry_mass])

    return constraint


# =====================================================================
#  Structural / thermal survival
# =====================================================================

def max_dynamic_pressure(
    q_max: float, dynamic_pressure_of: Callable[[_FloatArray], float]
) -> PathConstraint:
    """Dynamic pressure may not exceed ``q_max`` (Pa) — the airframe load limit."""

    def constraint(x: _FloatArray, u: _FloatArray) -> _FloatArray:
        return np.array([q_max - dynamic_pressure_of(x)])

    return constraint


def q_alpha_limit(
    qalpha_max: float,
    dynamic_pressure_of: Callable[[_FloatArray], float],
    angle_of_attack_of: Callable[[_FloatArray, _FloatArray], float],
) -> PathConstraint:
    """Bound ``q * alpha`` (Pa-rad) — the bending-moment load through max-Q."""

    def constraint(x: _FloatArray, u: _FloatArray) -> _FloatArray:
        return np.array([qalpha_max - dynamic_pressure_of(x) * abs(angle_of_attack_of(x, u))])

    return constraint


def g_load_limit(
    n_max: float,
    accel_magnitude_of: Callable[[_FloatArray, _FloatArray], float],
    g0: float = 9.80665,
) -> PathConstraint:
    """Sensed acceleration may not exceed ``n_max`` g."""

    def constraint(x: _FloatArray, u: _FloatArray) -> _FloatArray:
        return np.array([n_max * g0 - accel_magnitude_of(x, u)])

    return constraint


def heat_flux_limit(
    qdot_max: float, heat_flux_of: Callable[[_FloatArray], float]
) -> PathConstraint:
    """Stagnation-point heat flux may not exceed ``qdot_max`` (W/m^2)."""

    def constraint(x: _FloatArray, u: _FloatArray) -> _FloatArray:
        return np.array([qdot_max - heat_flux_of(x)])

    return constraint


def heat_load_limit(
    load_max: float, heat_flux_of: Callable[[_FloatArray], float]
) -> Callable[[ReferenceTrajectory], float]:
    """Integrated heat load may not exceed ``load_max`` (J/m^2) — a trajectory constraint.

    The TPS survives a peak flux *and* a total load; the second is a functional
    of the whole descent, so it is checked on the finished trajectory.
    """

    def check(trajectory: ReferenceTrajectory, samples: int = 300) -> float:
        times = np.linspace(trajectory.t_start, trajectory.t_end, samples)
        dt = (trajectory.t_end - trajectory.t_start) / max(samples - 1, 1)
        load = sum(heat_flux_of(trajectory.state(float(t))) for t in times) * dt
        return float(load_max - load)

    return check


# =====================================================================
#  Terminal geometry / reentry corridor  (attach to the terminal phase)
# =====================================================================

def terminal_flight_path_angle(
    gamma_min: float,
    flight_path_angle_of: Callable[[_FloatArray], float],
) -> PathConstraint:
    """Require a dive at least ``gamma_min`` (rad, magnitude) — impact steepness.

    Applied on the terminal phase it constrains the approach to arrive steeply
    enough to be lethal rather than skimming in flat.
    """

    def constraint(x: _FloatArray, u: _FloatArray) -> _FloatArray:
        return np.array([abs(flight_path_angle_of(x)) - gamma_min])

    return constraint


def terminal_speed_at_least(
    v_min: float, speed_of: Callable[[_FloatArray], float]
) -> PathConstraint:
    """Require terminal speed >= ``v_min`` (m/s) — penetration / closing energy."""

    def constraint(x: _FloatArray, u: _FloatArray) -> _FloatArray:
        return np.array([speed_of(x) - v_min])

    return constraint


def reentry_corridor(
    gamma_lo: float,
    gamma_hi: float,
    flight_path_angle_of: Callable[[_FloatArray], float],
) -> PathConstraint:
    """Keep the flight-path angle within ``[gamma_lo, gamma_hi]`` (rad).

    Too shallow and the vehicle skips back out; too steep and it over-heats /
    over-g's. Returns two margins, both ``>= 0`` inside the corridor.
    """

    def constraint(x: _FloatArray, u: _FloatArray) -> _FloatArray:
        gamma = flight_path_angle_of(x)
        return np.array([gamma - gamma_lo, gamma_hi - gamma])

    return constraint


# =====================================================================
#  Geography / launch  (path or trajectory level)
# =====================================================================

def no_fly_zone(
    center: _FloatArray,
    radius: float,
    position_of: Callable[[_FloatArray], _FloatArray] = lambda x: x[:3],
) -> PathConstraint:
    """Stay at least ``radius`` (m) from a keep-out ``center`` (Jorris no-fly zone)."""

    center = np.asarray(center, dtype=np.float64)

    def constraint(x: _FloatArray, u: _FloatArray) -> _FloatArray:
        return np.array([np.linalg.norm(position_of(x) - center) - radius])

    return constraint


def launch_window(
    epoch_min: float,
    epoch_max: float,
) -> Callable[[float], float]:
    """Feasible launch-epoch window ``[epoch_min, epoch_max]`` (s).

    Returns ``min(t - epoch_min, epoch_max - t)`` — ``>= 0`` inside the window.
    Consumed where the launch epoch is a decision variable.
    """

    def check(epoch: float) -> float:
        return float(min(epoch - epoch_min, epoch_max - epoch))

    return check


# =====================================================================
#  Two-sided engagement closers  (trajectory level; pass to the engagement layer)
# =====================================================================

def divert_budget(
    deltav_max: float,
    thrust_accel_of: Callable[[_FloatArray, _FloatArray], _FloatArray],
) -> Callable[[ReferenceTrajectory], float]:
    """Interceptor divert Delta-V may not exceed ``deltav_max`` (m/s).

    A kill vehicle carries a few hundred m/s of lateral divert; integrating the
    commanded thrust acceleration over the flight and capping it is what bounds
    the interceptor's reachable set.
    """

    def check(trajectory: ReferenceTrajectory, samples: int = 300) -> float:
        times = np.linspace(trajectory.t_start, trajectory.t_end, samples)
        dt = (trajectory.t_end - trajectory.t_start) / max(samples - 1, 1)
        used = 0.0
        for t in times:
            u = trajectory.control(float(t))
            used += float(np.linalg.norm(thrust_accel_of(trajectory.state(float(t)), u))) * dt
        return float(deltav_max - used)


    return check


def intercept_before(
    defended_position: _FloatArray,
    threat: ReferenceTrajectory,
    lethal_radius: float,
    position_of: Callable[[_FloatArray], _FloatArray] = lambda x: x[:3],
) -> Callable[[ReferenceTrajectory], float]:
    """The interceptor must close on the threat *before* the threat reaches the
    defended point.

    Returns ``t_threat_arrival - t_intercept`` — ``>= 0`` when the intercept
    happens in time. This temporal inequality is what makes 'minimise pursuit
    time' a feasible problem rather than an open-ended one.
    """

    defended = np.asarray(defended_position, dtype=np.float64)

    def _threat_arrival(samples: int) -> float:
        times = np.linspace(threat.t_start, threat.t_end, samples)
        for t in times:
            if np.linalg.norm(position_of(threat.state(float(t))) - defended) <= lethal_radius:
                return float(t)
        return float(threat.t_end)

    def check(interceptor: ReferenceTrajectory, samples: int = 400) -> float:
        arrival = _threat_arrival(samples)
        t0 = max(interceptor.t_start, threat.t_start)
        t1 = min(interceptor.t_end, threat.t_end)
        times = np.linspace(t0, t1, samples) if t1 > t0 else np.array([t0])
        for t in times:
            sep = np.linalg.norm(
                position_of(interceptor.state(float(t))) - position_of(threat.state(float(t)))
            )
            if sep <= lethal_radius:
                return float(arrival - t)
        return float(-(t1 - t0) - 1.0)  # never intercepts: infeasible

    return check


def seeker_acquisition(
    max_range: float,
    half_fov: float,
    threat: ReferenceTrajectory,
    position_of: Callable[[_FloatArray], _FloatArray] = lambda x: x[:3],
    boresight_of: Callable[[_FloatArray], _FloatArray] | None = None,
) -> Callable[[ReferenceTrajectory], float]:
    """The interceptor's seeker must acquire the threat: within ``max_range`` (m)
    and inside the half-angle field of view ``half_fov`` (rad) at some instant.

    Returns a positive margin once acquisition is possible; negative if the
    geometry never allows it.
    """

    def check(interceptor: ReferenceTrajectory, samples: int = 400) -> float:
        t0 = max(interceptor.t_start, threat.t_start)
        t1 = min(interceptor.t_end, threat.t_end)
        times = np.linspace(t0, t1, samples) if t1 > t0 else np.array([t0])
        best = -np.inf
        for t in times:
            xi = interceptor.state(float(t))
            los = position_of(threat.state(float(t))) - position_of(xi)
            rng = float(np.linalg.norm(los))
            range_margin = max_range - rng
            if boresight_of is not None and rng > 0.0:
                b = boresight_of(xi)
                cos_off = float(np.dot(los, b) / (rng * (np.linalg.norm(b) or 1.0)))
                fov_margin = cos_off - np.cos(half_fov)
            else:
                fov_margin = 0.0
            best = max(best, min(range_margin, fov_margin))
        return float(best)

    return check


# =====================================================================
#  Composition
# =====================================================================

def combine(*constraints: PathConstraint) -> PathConstraint:
    """Stack several path constraints into one ``g(x, u)`` returning all margins."""

    def combined(x: _FloatArray, u: _FloatArray) -> _FloatArray:
        parts = [np.atleast_1d(np.asarray(c(x, u), dtype=np.float64)) for c in constraints]
        return np.concatenate(parts) if parts else np.zeros(0)

    return combined
