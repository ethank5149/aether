"""Midcourse trajectory correction on the exoatmospheric arc.

A correction maneuver answers one question: given where the vehicle
actually is, what velocity change puts it back on a trajectory that
arrives where it is supposed to, when it is supposed to? Lambert's problem
answers the "what velocity" part; everything here is about *when* to ask,
*how well* the answer can be executed, and what the residual miss is.

The central trade
-----------------

Correcting a position error :math:`\\delta r` at time-to-go :math:`t_{go}`
costs roughly

.. math::

    |\\Delta v| \\sim \\frac{|\\delta r|}{t_{go}},

so **fuel argues for burning as early as possible**. Knowledge argues the
opposite: early in the arc the navigation covariance is still large, so an
early burn corrects an error that is largely estimation noise and must be
redone. The optimum is interior, and it is not a matter of taste — it
falls out of the covariance history, which is why
:func:`schedule_corrections` takes one.

Two failure modes hide in that trade and both are represented here rather
than assumed away. **Execution error** means a commanded
:math:`\\Delta \\mathbf{v}` is not the one applied: real thrusters carry a
proportional magnitude error and a pointing error, so a large early burn
injects a large new error, which partly cancels the benefit of burning
early. **Knowledge error** means the correction is computed from an
estimate, so even a perfectly executed burn leaves the estimation error
uncorrected. The first scales with the burn, the second does not.

Fixed versus variable time of arrival
-------------------------------------

What is implemented here is **fixed time of arrival**: the maneuver is
sized to reach the aimpoint at a specified epoch, not merely to reach it.
That is the expensive option, and it is the one a coordinated arrival
requires.

Variable time of arrival — targeting the position alone and accepting
whatever arrival epoch results — is typically cheaper, sometimes by a
large factor, because it does not fight the along-track component of the
error, which is usually the largest. It is **not** implemented, and the
difference is not a tuning parameter: it is a different boundary-value
problem with one fewer constraint. Sweeping ``time_of_flight`` and taking
the cheapest is a serviceable stand-in, and it makes the cost of holding
the arrival epoch explicit rather than hidden.

Scope
-----

Targeting is seeded by a two-body Lambert solve and then refined against
whatever dynamics the propagator carries, so the achieved trajectory
includes :math:`J_2`. It is *not* an optimal-control solution: a genuinely
fuel-optimal multi-burn plan is a different problem, and
:mod:`aether.optimal_control.scvx` is the tool for that. What this module does is
the classical, auditable thing — Lambert targeting with an explicit error
budget — which is what a correction plan is actually built from.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aether.orbital.coast import propagate_coast
from aether.orbital.gravity import EARTH, GravityModel
from aether.orbital.lambert import lambert

__all__ = [
    "CorrectionPlan",
    "CorrectionResult",
    "ExecutionErrorModel",
    "correction_maneuver",
    "miss_sensitivity",
    "schedule_corrections",
]

_FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ExecutionErrorModel:
    """How badly a commanded :math:`\\Delta \\mathbf{v}` is applied.

    Attributes
    ----------
    magnitude_fraction:
        One-sigma proportional error on :math:`|\\Delta \\mathbf{v}|`.
        Scales with the burn, so it is what penalises correcting early
        with a large maneuver.
    pointing_sigma:
        One-sigma pointing error (rad), applied about an axis normal to
        the commanded direction. Also scales with the burn.
    fixed_sigma:
        One-sigma error (m/s) independent of burn size — thruster
        resolution, tail-off, minimum impulse bit. This is what makes a
        large number of tiny corrections *worse* than a few larger ones,
        and it is the reason `schedule_corrections` does not simply
        recommend correcting continuously.
    """

    magnitude_fraction: float = 0.01
    pointing_sigma: float = 2.0e-3
    fixed_sigma: float = 0.0

    def __post_init__(self) -> None:
        for name in ("magnitude_fraction", "pointing_sigma", "fixed_sigma"):
            value = float(getattr(self, name))
            if not (np.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be finite and >= 0, got {value}")

    def covariance(self, delta_v: ArrayLike) -> _FloatArray:
        """Execution-error covariance (m²/s²) for a commanded burn.

        The magnitude error acts along the burn, the pointing error in the
        two directions normal to it, and the fixed error isotropically.
        Building it in the burn frame and rotating out is what keeps the
        anisotropy — a burn is much better known along its own axis than
        across it, and treating the error as isotropic would hide that.
        """
        dv = np.asarray(delta_v, dtype=np.float64)
        if dv.shape != (3,):
            raise ValueError("delta_v must be a 3-vector")
        magnitude = float(np.linalg.norm(dv))
        isotropic = self.fixed_sigma**2 * np.eye(3)
        if magnitude == 0.0:
            return np.asarray(isotropic)
        axis = dv / magnitude
        along = (self.magnitude_fraction * magnitude) ** 2
        across = (self.pointing_sigma * magnitude) ** 2
        projector = np.outer(axis, axis)
        return np.asarray(along * projector + across * (np.eye(3) - projector) + isotropic)


@dataclass(frozen=True)
class CorrectionResult:
    """A single correction maneuver and what it is expected to leave behind.

    Attributes
    ----------
    delta_v:
        Commanded velocity change (m/s) in the inertial frame.
    time_of_flight:
        Remaining flight time (s) the correction was targeted over.
    v_required:
        Velocity the vehicle must have after the burn.
    residual_miss:
        One-sigma expected miss distance (m) at arrival, combining the
        knowledge error that the burn cannot remove with the execution
        error the burn itself injects. ``nan`` if no covariance was given.
    """

    delta_v: _FloatArray
    time_of_flight: float
    v_required: _FloatArray
    residual_miss: float = float("nan")
    refinements: int = 0
    """Newton passes taken against the real propagator, for diagnostics."""

    @property
    def cost(self) -> float:
        """Maneuver magnitude (m/s)."""
        return float(np.linalg.norm(self.delta_v))


@dataclass(frozen=True)
class CorrectionPlan:
    """A sequence of corrections and its aggregate cost.

    Attributes
    ----------
    corrections:
        Maneuvers in execution order.
    times:
        Time from the start of the arc (s) at which each is applied.
    total_cost:
        Sum of maneuver magnitudes (m/s). This is the fuel figure of
        merit, and it is *not* the same as the norm of the vector sum:
        corrections that partly undo each other still cost propellant.
    final_miss:
        One-sigma expected miss (m) after the last correction.
    """

    corrections: tuple[CorrectionResult, ...]
    times: _FloatArray = field(repr=False)
    total_cost: float
    final_miss: float


def correction_maneuver(
    position: ArrayLike,
    velocity: ArrayLike,
    target_position: ArrayLike,
    time_of_flight: float,
    model: GravityModel = EARTH,
    prograde: bool = True,
    state_covariance: ArrayLike | None = None,
    execution: ExecutionErrorModel | None = None,
    include_j2: bool = True,
    refine: bool = True,
    tolerance: float = 1.0,
    max_refinements: int = 8,
) -> CorrectionResult:
    """Velocity change that retargets the vehicle to an aimpoint.

    Fixed time of arrival: the maneuver is sized so that the vehicle
    reaches ``target_position`` exactly ``time_of_flight`` seconds later.

    Parameters
    ----------
    position, velocity:
        Current estimated inertial state (m, m/s).
    target_position:
        Inertial aimpoint (m) at the arrival epoch.
    time_of_flight:
        Remaining time (s) to arrival. Positive.
    state_covariance:
        Optional 6x6 covariance of the current state estimate, ordered
        position then velocity. Supplying it turns on the residual-miss
        estimate; without it the maneuver is still computed, but the
        result reports ``nan`` for the miss rather than pretending to a
        number it cannot support.
    execution:
        Execution-error model. Defaults to a perfect burn, which is a
        deliberate choice: a silently assumed 1% error would make every
        cost figure in this module quietly optimistic.
    """
    r = np.asarray(position, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    target = np.asarray(target_position, dtype=np.float64)
    for name, vec in (("position", r), ("velocity", v), ("target_position", target)):
        if vec.shape != (3,):
            raise ValueError(f"{name} must be a 3-vector")
        if not np.isfinite(vec).all():
            raise ValueError(f"{name} must be finite")
    tof = float(time_of_flight)
    if not (np.isfinite(tof) and tof > 0.0):
        raise ValueError(f"time_of_flight must be finite and > 0, got {tof}")

    # Seed selection, then differential correction against the real
    # dynamics.
    #
    # Lambert is a *two-body* solve, but the trajectory actually flown
    # carries J2 (and drag, if the propagator is configured for it). Using
    # Lambert's answer directly leaves a miss of kilometres over a
    # half-hour arc -- negligible as a fraction of range, and far too large
    # as a correction residual. Lambert therefore only seeds a Newton
    # iteration against the propagator, which drives the true miss to
    # centimetres in two or three aether.
    #
    # The seed list deliberately begins with the velocity the vehicle
    # already has. A correction is by construction a small perturbation of
    # the current motion, and near the end of an arc the vehicle is nearly
    # collinear with its own aimpoint -- a geometry where Lambert's
    # transfer plane is barely determined and both of its branches can be
    # nonsense (one case here asked for 127 km/s). Seeding from the current
    # velocity is well conditioned exactly where Lambert is not, and the
    # two failure regions do not overlap, so carrying both is what makes
    # the whole arc covered.
    seeds: list[_FloatArray] = [np.asarray(v, dtype=np.float64)]
    for direction in (True, False):
        try:
            seeds.append(lambert(r, target, tof, model=model, prograde=direction).v1)
        except (ValueError, RuntimeError):
            continue

    def terminal_miss(candidate: _FloatArray) -> float:
        try:
            reached = propagate_coast(
                r,
                candidate,
                tof,
                model=model,
                include_j2=include_j2,
                rtol=1e-12,
                atol=1e-6,
                n_output=2,
            )
        except RuntimeError:
            # A seed that flies into the central body is not a candidate.
            return float("inf")
        return float(np.linalg.norm(np.asarray(reached.states[:3, -1]) - target))

    scored = [(terminal_miss(seed), i, seed) for i, seed in enumerate(seeds)]
    best = min(scored, key=lambda item: (item[0], item[1]))
    if not np.isfinite(best[0]):
        raise ValueError(
            "no seed trajectory reaches the aimpoint region; the target is "
            "not reachable from this state in the time available"
        )
    v_required = best[2]

    refine_iterations = 0
    if refine:
        for _ in range(max_refinements):
            refine_iterations += 1
            reached = propagate_coast(
                r,
                v_required,
                tof,
                model=model,
                include_j2=include_j2,
                rtol=1e-12,
                atol=1e-6,
                n_output=2,
            )
            residual = np.asarray(reached.states[:3, -1]) - target
            if float(np.linalg.norm(residual)) <= tolerance:
                break
            sensitivity = miss_sensitivity(r, v_required, tof, model=model, include_j2=include_j2)
            try:
                v_required = v_required - np.linalg.solve(sensitivity.velocity, residual)
            except np.linalg.LinAlgError as exc:  # pragma: no cover - degenerate
                raise RuntimeError(
                    "terminal sensitivity is singular; the aimpoint is not "
                    "reachable by a velocity change at this point on the arc"
                ) from exc

    delta_v = v_required - v

    miss = float("nan")
    if state_covariance is not None:
        covariance = np.asarray(state_covariance, dtype=np.float64)
        if covariance.shape != (6, 6):
            raise ValueError(
                f"state_covariance must be 6x6 (position then velocity), got {covariance.shape}"
            )
        sensitivity = miss_sensitivity(r, v_required, tof, model=model, include_j2=include_j2)
        # Knowledge error maps to a miss two ways: a position error moves
        # where the burn starts from, and a velocity error is *not* removed
        # by the burn, because the burn was computed from the estimate and
        # so inherits it. Both propagate through the same terminal
        # sensitivity, which is why they combine rather than compete.
        transition = np.hstack([sensitivity.position, sensitivity.velocity])
        total = transition @ covariance @ transition.T
        if execution is not None:
            # Execution error enters only through the velocity block, and
            # only in proportion to the burn -- this is the term that makes
            # a large early correction self-defeating.
            total = total + (
                sensitivity.velocity @ execution.covariance(delta_v) @ sensitivity.velocity.T
            )
        miss = float(np.sqrt(max(0.0, np.trace(total))))

    return CorrectionResult(
        delta_v=np.asarray(delta_v),
        time_of_flight=tof,
        v_required=np.asarray(v_required),
        residual_miss=miss,
        refinements=refine_iterations,
    )


@dataclass(frozen=True)
class _Sensitivity:
    """Terminal-position sensitivity to the departure state."""

    position: _FloatArray
    """:math:`\\partial \\mathbf{r}_f / \\partial \\mathbf{r}_0`, dimensionless."""
    velocity: _FloatArray
    """:math:`\\partial \\mathbf{r}_f / \\partial \\mathbf{v}_0` (s)."""


def miss_sensitivity(
    position: ArrayLike,
    velocity: ArrayLike,
    time_of_flight: float,
    model: GravityModel = EARTH,
    include_j2: bool = True,
    step_position: float = 1.0,
    step_velocity: float = 1.0e-3,
) -> _Sensitivity:
    """Terminal-position sensitivity to the departure state.

    The upper-right block :math:`\\partial \\mathbf{r}_f /
    \\partial \\mathbf{v}_0` is the one that matters for correction sizing:
    inverted, it converts a desired terminal move into the velocity change
    that produces it, and its smallest singular value is what says how
    expensive the *worst* direction of correction is.

    Evaluated by central differences of the propagator rather than by
    integrating the variational equations. That is a deliberate trade: it
    costs twelve extra propagations, and in exchange the sensitivity is
    guaranteed consistent with the trajectory model actually being flown,
    including :math:`J_2` and drag, with no second derivation to keep in
    step. The default steps are chosen well above the propagator's own
    tolerance and well below the scale on which the dynamics curve.
    """
    r = np.asarray(position, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    if r.shape != (3,) or v.shape != (3,):
        raise ValueError("position and velocity must both be 3-vectors")
    tof = float(time_of_flight)
    if not (np.isfinite(tof) and tof > 0.0):
        raise ValueError(f"time_of_flight must be finite and > 0, got {tof}")
    for name, step in (("step_position", step_position), ("step_velocity", step_velocity)):
        if not (np.isfinite(step) and step > 0.0):
            raise ValueError(f"{name} must be finite and > 0, got {step}")

    def terminal(r0: _FloatArray, v0: _FloatArray) -> _FloatArray:
        result = propagate_coast(
            r0,
            v0,
            tof,
            model=model,
            include_j2=include_j2,
            rtol=1e-12,
            atol=1e-6,
            n_output=2,
        )
        return np.asarray(result.states[:3, -1])

    d_position = np.zeros((3, 3))
    d_velocity = np.zeros((3, 3))
    for axis in range(3):
        perturb = np.zeros(3)
        perturb[axis] = step_position
        d_position[:, axis] = (terminal(r + perturb, v) - terminal(r - perturb, v)) / (
            2.0 * step_position
        )
        perturb = np.zeros(3)
        perturb[axis] = step_velocity
        d_velocity[:, axis] = (terminal(r, v + perturb) - terminal(r, v - perturb)) / (
            2.0 * step_velocity
        )
    return _Sensitivity(position=d_position, velocity=d_velocity)


def schedule_corrections(
    position: ArrayLike,
    velocity: ArrayLike,
    target_position: ArrayLike,
    total_time: float,
    burn_times: ArrayLike,
    model: GravityModel = EARTH,
    execution: ExecutionErrorModel | None = None,
    covariance_at: Callable[[float], ArrayLike] | None = None,
    prograde: bool = True,
) -> CorrectionPlan:
    """Cost a fixed schedule of corrections by flying it.

    Each correction is computed from the state actually reached, so the
    plan accounts for the fact that a later burn has to clean up what an
    earlier one left — including the error the earlier burn injected.

    Parameters
    ----------
    burn_times:
        Times from the start of the arc (s) at which to correct, strictly
        increasing and all strictly inside ``(0, total_time)``.
    covariance_at:
        Optional callable mapping a time (s) to a 6x6 state covariance.
        This is how the knowledge side of the trade enters: pass the
        filter's covariance history and early burns are correctly charged
        for correcting an estimate rather than the truth.

    Notes
    -----
    This *evaluates* a schedule; it does not optimise one. Sweeping
    :func:`schedule_corrections` over candidate ``burn_times`` and taking
    the minimum is the honest way to find a good schedule, and it makes
    the shape of the trade visible instead of hiding it in an optimiser.
    """
    r = np.asarray(position, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    target = np.asarray(target_position, dtype=np.float64)
    times = np.atleast_1d(np.asarray(burn_times, dtype=np.float64))
    duration = float(total_time)
    if not (np.isfinite(duration) and duration > 0.0):
        raise ValueError(f"total_time must be finite and > 0, got {duration}")
    if times.ndim != 1 or times.size == 0:
        raise ValueError("burn_times must be a non-empty 1-D sequence")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("burn_times must be strictly increasing")
    if times[0] <= 0.0 or times[-1] >= duration:
        raise ValueError(
            f"burn_times must lie strictly inside (0, {duration:.6g}); got "
            f"[{times[0]:.6g}, {times[-1]:.6g}]"
        )

    corrections: list[CorrectionResult] = []
    total_cost = 0.0
    state_r, state_v = r, v
    clock = 0.0
    for burn_time in times:
        coast = propagate_coast(
            state_r,
            state_v,
            float(burn_time - clock),
            model=model,
            rtol=1e-12,
            atol=1e-6,
            n_output=2,
        )
        state_r = np.asarray(coast.states[:3, -1])
        state_v = np.asarray(coast.states[3:, -1])
        clock = float(burn_time)
        covariance = None if covariance_at is None else covariance_at(clock)
        correction = correction_maneuver(
            state_r,
            state_v,
            target,
            duration - clock,
            model=model,
            prograde=prograde,
            state_covariance=covariance,
            execution=execution,
        )
        corrections.append(correction)
        total_cost += correction.cost
        state_v = correction.v_required

    return CorrectionPlan(
        corrections=tuple(corrections),
        times=times,
        total_cost=float(total_cost),
        final_miss=corrections[-1].residual_miss,
    )
