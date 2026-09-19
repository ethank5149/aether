"""Objectives for trajectory optimization.

An objective maps a :class:`~aether.trajectory.representation.ReferenceTrajectory`
and an :class:`EvaluationContext` to a scalar cost, *always in minimise sense*
(a maximise objective negates internally, so the solver only ever minimises).
Objectives compose the two ways the trajectory-optimization literature composes
them (Betts, *Practical Methods for Optimal Control*, §4.9): a normalised
weighted sum, or a lexicographic priority order where a lower-priority term is
optimised only within the tolerance of the higher one.

The concrete objectives here cover the cases named for this module — minimum
flight time, maximum undetected (blind) time, minimum pursuit time — plus the
usual control-effort and terminal-miss regularisers. Undetected-time and
pursuit-time need external structure (a sensor picture, an interceptor), which
they take through the context rather than reaching into the engagement layer,
so the objective set stays decoupled from any one solver or scenario type.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from aether.trajectory.representation import ReferenceTrajectory

__all__ = [
    "EvaluationContext",
    "Lexicographic",
    "MaxPeakAltitude",
    "MaxTerminalSpeed",
    "MaxUndetectedTime",
    "MinBoostDuration",
    "MinControlEffort",
    "MinFlightTime",
    "MinPeakAltitude",
    "MinPeakDynamicPressure",
    "MinPursuitTime",
    "MinTerminalMiss",
    "Objective",
    "WeightedSum",
]


def _default_altitude(body_radius: float) -> Callable[[np.ndarray], float]:
    def altitude(state: np.ndarray) -> float:
        return float(np.linalg.norm(state[:3]) - body_radius)

    return altitude


def _default_speed(state: np.ndarray) -> float:
    return float(np.linalg.norm(state[3:6])) if state.size >= 6 else 0.0


@dataclass
class EvaluationContext:
    """Everything an objective needs beyond the trajectory itself.

    Fields are optional so an objective that does not need one ignores it; an
    objective that does raises a clear error when its field is absent. The
    physical *accessors* (altitude, speed, dynamic pressure) keep the objective
    set decoupled from any one state convention: an objective asks the context
    for "the altitude of this state" rather than assuming a layout.
    """

    body_radius: float = 6.371e6
    earth_rotation: float = 7.292115e-5
    #: ``detected(position_eci, t) -> bool`` — True when any sensor holds the
    #: vehicle. Supplied by the caller (wired to the radar/IR pipeline) so this
    #: module does not depend on the sensor internals.
    detected: Callable[[np.ndarray, float], bool] | None = None
    #: Aim point / target state for terminal-miss objectives (ECI, m).
    target: np.ndarray | None = None
    #: Interceptor reference for pursuit-time; set by the engagement layer.
    pursuer: ReferenceTrajectory | None = None
    #: ``altitude(state) -> m`` and ``speed(state) -> m/s``. Default to a
    #: geocentric-position / inertial-velocity reading of ``[r, v]``.
    altitude_of: Callable[[np.ndarray], float] | None = None
    speed_of: Callable[[np.ndarray], float] = _default_speed
    #: ``dynamic_pressure(state, t) -> Pa`` — no default (needs an atmosphere).
    dynamic_pressure_of: Callable[[np.ndarray, float], float] | None = None
    #: Phase-label prefixes that count as powered/boost, for IR-signature terms.
    boost_labels: tuple[str, ...] = ("boost", "ascent", "powered")
    #: Sample count for time-integrated objectives.
    samples: int = 200
    extras: dict[str, object] = field(default_factory=dict)

    def altitude(self, state: np.ndarray) -> float:
        fn = self.altitude_of or _default_altitude(self.body_radius)
        return fn(state)


@runtime_checkable
class Objective(Protocol):
    """A scalar cost on a trajectory, in minimise sense."""

    @property
    def name(self) -> str:
        """Short identifier, carried into trajectory metadata."""
        ...

    def evaluate(self, trajectory: ReferenceTrajectory, context: EvaluationContext) -> float:
        ...


@dataclass(frozen=True)
class MinFlightTime:
    """Minimise total time of flight — the launch-to-impact duration."""

    name: str = "min-flight-time"

    def evaluate(self, trajectory: ReferenceTrajectory, context: EvaluationContext) -> float:
        return float(trajectory.t_end - trajectory.t_start)


@dataclass(frozen=True)
class MinControlEffort:
    """Minimise the integrated squared control — the usual smoothness term."""

    name: str = "min-control-effort"

    def evaluate(self, trajectory: ReferenceTrajectory, context: EvaluationContext) -> float:
        times = np.linspace(trajectory.t_start, trajectory.t_end, context.samples)
        total = 0.0
        for t in times:
            u = trajectory.control(float(t))
            if u.size:
                total += float(np.dot(u, u))
        return total * (trajectory.t_end - trajectory.t_start) / context.samples


@dataclass(frozen=True)
class MinTerminalMiss:
    """Minimise the terminal position miss against ``context.target``."""

    name: str = "min-terminal-miss"

    def evaluate(self, trajectory: ReferenceTrajectory, context: EvaluationContext) -> float:
        if context.target is None:
            raise ValueError("MinTerminalMiss needs context.target")
        end = trajectory.state(trajectory.t_end)[: context.target.size]
        return float(np.linalg.norm(end - np.asarray(context.target, dtype=np.float64)))


@dataclass(frozen=True)
class MaxUndetectedTime:
    """Maximise the time the vehicle is *not* held by any sensor.

    Returned in minimise sense as ``-(undetected time)``. The detection test is
    supplied through ``context.detected`` so this objective is agnostic to the
    radar/IR implementation; without it the objective cannot be evaluated and
    says so rather than assuming the vehicle is invisible.
    """

    name: str = "max-undetected-time"

    def evaluate(self, trajectory: ReferenceTrajectory, context: EvaluationContext) -> float:
        if context.detected is None:
            raise ValueError(
                "MaxUndetectedTime needs context.detected (the sensor picture); "
                "it will not assume the vehicle is undetected"
            )
        times = np.linspace(trajectory.t_start, trajectory.t_end, context.samples)
        dt = (trajectory.t_end - trajectory.t_start) / max(context.samples - 1, 1)
        undetected = 0.0
        for t in times:
            pos = trajectory.state(float(t))[:3]
            if not context.detected(pos, float(t)):
                undetected += dt
        return -float(undetected)


@dataclass(frozen=True)
class MinPursuitTime:
    """Minimise time-to-intercept against ``context.pursuer``.

    This is the engagement-level objective: it needs the opposing (interceptor)
    reference trajectory, which the two-sided engagement layer supplies. The
    pursuit time is the first instant the two references close within
    ``capture_radius``; if they never do, the full flight time is returned as a
    (large) miss cost so the optimiser is pushed toward closing geometries.
    """

    name: str = "min-pursuit-time"
    capture_radius: float = 5.0e3

    def evaluate(self, trajectory: ReferenceTrajectory, context: EvaluationContext) -> float:
        if context.pursuer is None:
            raise ValueError("MinPursuitTime needs context.pursuer (the interceptor reference)")
        pursuer = context.pursuer
        t0 = max(trajectory.t_start, pursuer.t_start)
        t1 = min(trajectory.t_end, pursuer.t_end)
        if not t1 > t0:
            return float(trajectory.t_end - trajectory.t_start)
        times = np.linspace(t0, t1, context.samples)
        for t in times:
            sep = np.linalg.norm(trajectory.state(float(t))[:3] - pursuer.state(float(t))[:3])
            if sep <= self.capture_radius:
                return float(t - t0)
        return float(t1 - t0)


@dataclass(frozen=True)
class MinBoostDuration:
    """Minimise total powered-flight time — the IR-signature term.

    A boosting stage is a bright, hot, detectable plume; the shorter the total
    burn, the smaller the window a boost-phase IR sensor has to acquire and the
    boost-phase interceptor has to react. Sums the durations of the phases whose
    label marks them powered (``context.boost_labels``). With nothing else in
    the objective this drives a hot, high-thrust, short boost — so it is
    normally one term of a composite against range/energy feasibility.
    """

    name: str = "min-boost-duration"

    def evaluate(self, trajectory: ReferenceTrajectory, context: EvaluationContext) -> float:
        total = 0.0
        for phase in trajectory.phases:
            if any(phase.label.startswith(p) for p in context.boost_labels):
                total += phase.t1 - phase.t0
        return float(total)


@dataclass(frozen=True)
class MinPeakAltitude:
    """Minimise the trajectory apogee — this *is* the depressed trajectory.

    A depressed (flattened) trajectory trades the efficient minimum-energy
    lofted arc for a lower, faster path that spends less time high and exposed
    and compresses the defender's timeline. Minimising the peak altitude is the
    objective that produces it; bounding range or energy elsewhere keeps it
    feasible.
    """

    name: str = "min-peak-altitude"

    def evaluate(self, trajectory: ReferenceTrajectory, context: EvaluationContext) -> float:
        times = np.linspace(trajectory.t_start, trajectory.t_end, context.samples)
        return float(max(context.altitude(trajectory.state(float(t))) for t in times))


@dataclass(frozen=True)
class MaxPeakAltitude:
    """Maximise the apogee — the lofted trajectory, the inverse of depressed.

    Returned as ``-(peak altitude)`` (minimise sense). A lofted profile raises
    the apogee and steepens reentry to compress the terminal window at the cost
    of energy and heating; minimising ``-apogee`` is the objective that shapes
    it, the mathematical mirror of :class:`MinPeakAltitude`.
    """

    name: str = "max-peak-altitude"

    def evaluate(self, trajectory: ReferenceTrajectory, context: EvaluationContext) -> float:
        times = np.linspace(trajectory.t_start, trajectory.t_end, context.samples)
        return -float(max(context.altitude(trajectory.state(float(t))) for t in times))


@dataclass(frozen=True)
class MaxTerminalSpeed:
    """Maximise terminal speed — penetration energy, or interceptor closing.

    Returned as ``-speed`` (minimise sense). For a threat it is impact energy /
    penetration; for an interceptor it is closing velocity into the kill.
    """

    name: str = "max-terminal-speed"

    def evaluate(self, trajectory: ReferenceTrajectory, context: EvaluationContext) -> float:
        return -float(context.speed_of(trajectory.state(trajectory.t_end)))


@dataclass(frozen=True)
class MinPeakDynamicPressure:
    """Minimise peak dynamic pressure — a structural / survivability limit.

    Needs ``context.dynamic_pressure_of`` (an atmosphere); it will not guess a
    ``q`` profile. Useful as a constraint-shaping objective on a depressed or
    skip trajectory that would otherwise dive through dense air too fast.
    """

    name: str = "min-peak-dynamic-pressure"

    def evaluate(self, trajectory: ReferenceTrajectory, context: EvaluationContext) -> float:
        if context.dynamic_pressure_of is None:
            raise ValueError("MinPeakDynamicPressure needs context.dynamic_pressure_of")
        times = np.linspace(trajectory.t_start, trajectory.t_end, context.samples)
        return float(
            max(context.dynamic_pressure_of(trajectory.state(float(t)), float(t)) for t in times)
        )


@dataclass(frozen=True)
class WeightedSum:
    """Normalised weighted sum of objectives (Betts §4.9).

    Each term is divided by a ``scale`` (its characteristic magnitude) so one
    physical unit does not dominate by its size alone, then weighted.
    """

    terms: tuple[tuple[Objective, float, float], ...]  # (objective, weight, scale)
    name: str = "weighted-sum"

    def evaluate(self, trajectory: ReferenceTrajectory, context: EvaluationContext) -> float:
        return float(
            sum(
                weight * obj.evaluate(trajectory, context) / scale
                for obj, weight, scale in self.terms
            )
        )


@dataclass(frozen=True)
class Lexicographic:
    """Lexicographic objectives: earlier terms dominate later ones.

    Evaluation returns a scalar that orders trajectories the same way the
    priority list does, by packing each term into a decreasing-magnitude
    decade band. This avoids the optimiser trading a large drop in a
    high-priority term for a tiny gain in a low-priority one — the failure mode
    a naive weighted sum has when scales are misjudged (Betts §4.9).
    """

    priorities: tuple[tuple[Objective, float], ...]  # (objective, band_scale)
    name: str = "lexicographic"

    def evaluate(self, trajectory: ReferenceTrajectory, context: EvaluationContext) -> float:
        total = 0.0
        band = 1.0
        for obj, scale in self.priorities:
            total += band * (obj.evaluate(trajectory, context) / scale)
            band *= 1.0e-3
        return float(total)
