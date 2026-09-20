r"""Skip entry guidance by numerical predictor-corrector, in the PredGuid lineage.

A capsule returning from the Moon arrives near 11 km/s with a lift-to-drag
ratio of about 0.3. Flown as a direct entry that buys roughly 2500 nmi of
range; reaching a fixed landing site from anywhere the return geometry puts
the entry interface needs more, and the way to get it is to **skip**: dip
into the atmosphere, shed some energy, use the lift to climb back out, coast
a ballistic arc, and enter a second time closer to the site. Orion's entry
guidance, PredGuid, does this with a numerical predictor-corrector (NPC)
built on the Apollo entry guidance; Artemis I flew it in December 2022 and
landed within 2 nmi of its target (Orion Artemis I Entry Performance,
AAS 24-174).

The algorithm here follows the structure that paper describes, simplified
where stated.

The predictor
-------------

Each guidance cycle asks one question: flown from the present state at a
*constant* vertical lift fraction :math:`u = \cos\sigma`, where does the body
end up? The prediction integrates the planar entry equations -- radius,
downrange angle, speed and flight-path angle -- to the terminal speed, and
compares the predicted downrange with the distance to the target site. A
constant-bank trajectory is not what will be flown, but it is a
one-parameter family that spans the achievable range, and re-solving every
cycle from the current state absorbs the difference. That is the same
reasoning as the Apollo and PredGuid predictors, whose constant-bank problem
this is.

The target site rotates with the Earth, and the entry equations here are
inertial. Over a skip entry of about 16 minutes the Earth turns about four
degrees, several hundred kilometres at mid latitudes, so the predictor aims
at where the site *will be* at the predicted arrival time rather than where
it is now.

The corrector
-------------

The range error is monotone in :math:`u` over the region that matters: more
vertical lift flies further. A secant iteration on :math:`u`, warm-started
from the previous cycle's solution and bracketed by full lift-up and the
body's bank limit, converges in one to three predictions once the solution
settles, matching what the flight data shows for PredGuid. When even full
lift-up falls short the command saturates there, and likewise for lift-down.

What is simplified
------------------

* PredGuid's Energy Management, Up Control and Ballistic phases each run the
  NPC with slightly different terminal conditions; here one NPC phase covers
  all three, predicting to the terminal speed throughout.
* PredGuid's Final phase replaces the NPC with Apollo's reference-trajectory
  terminal point controller. Here the NPC continues to the terminal speed.
* The predictor is planar and the atmosphere does not co-rotate. Both are
  stated limitations of the plant and the predictor alike, so they bias the
  answer rather than destabilise the loop.

The estimators are PredGuid's: a density scale factor and a lift-to-drag
estimate, each a first-order filter on the sensed aerodynamic acceleration,
active once drag exceeds a threshold, and fed to the predictor. They are what
let the guidance absorb an atmosphere and an L/D that differ from its model,
which on Artemis I they did by about 10 % and 5 %.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import scipy.integrate
from numpy.typing import NDArray

from aether.guidance.entry import (
    _MU,
    _R_EARTH,
    EntryBody,
    EntryState,
    entry_dynamics,
)

__all__ = [
    "EARTH_ROTATION_RATE",
    "SKIP_OUT_ALTITUDE",
    "SKIP_OUT_ERROR",
    "Prediction",
    "SkipEntryResult",
    "SkipGuidance",
    "SkipGuidanceConfig",
    "SkipPhase",
    "fly_skip_entry",
    "predict_constant_bank",
]

_FloatArray = NDArray[np.float64]

#: Sidereal rotation rate of the Earth (rad/s), WGS84.
EARTH_ROTATION_RATE = 7.2921150e-5


class SkipPhase(Enum):
    """Guidance phases, in the order they are flown."""

    INITIAL_ROLL = "initial roll"
    PREDICTOR_CORRECTOR = "predictor-corrector"
    FINAL = "final"
    TERMINAL = "terminal"


@dataclass(frozen=True)
class SkipGuidanceConfig:
    """Parameters of the guidance. None is defaulted to a particular body.

    Attributes
    ----------
    initial_bank:
        Bank magnitude (rad) held through the first pass until the body starts
        to pull up. Artemis I held 0 degrees limited to 15 to preserve lateral
        authority.
    initial_roll_exit_altitude_rate:
        The Initial Roll phase ends when the altitude rate rises above this
        (m/s, negative); Artemis I used -700 ft/s.
    terminal_speed:
        Speed (m/s) that ends guided flight; Artemis I used 1000 ft/s.
    range_tolerance:
        Converged when the predicted range error is inside this (m);
        Artemis I used 25 nmi.
    max_iterations:
        Predictions allowed per guidance cycle; Artemis I allowed 5.
    estimator_drag_threshold:
        Drag (m/s²) above which the density and L/D estimators run;
        Artemis I used 1.6 ft/s².
    corridor_constant, corridor_quadratic:
        The lateral corridor on the crossrange angle to the target site, in the
        Apollo-heritage form :math:`c_0 + c_1 V^2` (rad, and rad s²/m²): a bank
        reversal is commanded when the crossrange angle leaves it (Smith,
        *Predictive Lateral Logic for Numerical Entry Guidance Algorithms*,
        AAS preprint, NTRS 20160001182, Eq. 1). Wide at high speed, where a
        heading error is cheap to correct later, and narrowing quadratically
        as the energy to correct with runs out. The two coefficients are tuned
        per mission to set the number of reversals; PredGuid uses this form with
        its own corridor definition, whose coefficients are not published.
    final_phase_drag:
        Drag (m/s²) that marks the second entry: once drag has fallen below it
        on the way out of the atmosphere, rising back through it starts the
        Final phase. Artemis I used 6 ft/s² (D Upc End).
    final_vertical_lift_fraction:
        Vertical lift fraction :math:`\\cos\\sigma` the predictor assumes after
        the second entry while solving the skip. It is the reserve the Final
        phase keeps in both directions: a skip solved to a lift-neutral second
        entry arrives with no authority left to correct with.
    final_range_tolerance:
        Range tolerance (m) in the Final phase, where the corrector re-solves
        one constant bank to the terminal speed; ``range_tolerance`` if zero.
    roll_rate:
        Largest bank rate the body can slew at (rad/s), or zero for an
        instantaneous bank. A reversal sweeps the lift vector through
        wings-level, where the vertical component is largest, so a finite rate
        both delays the reversal and lofts the body while it turns. Orion's
        handling-qualities work flew reversals at 9 deg/s, and 15 deg/s was
        the rate the lunar-return skip entry scenarios were run at
        (NTRS 20110013203).
    guidance_period:
        Seconds between guidance cycles.
    estimator_time_constant:
        Time constant (s) of the first-order estimator filters.
    """

    initial_bank: float
    initial_roll_exit_altitude_rate: float
    terminal_speed: float
    range_tolerance: float
    max_iterations: int
    estimator_drag_threshold: float
    corridor_constant: float
    corridor_quadratic: float
    final_phase_drag: float = 0.0
    final_vertical_lift_fraction: float = 0.5
    final_range_tolerance: float = 0.0
    roll_rate: float = 0.0
    guidance_period: float = 2.0
    estimator_time_constant: float = 10.0


@dataclass(frozen=True)
class Prediction:
    """One constant-bank prediction."""

    vertical_lift_fraction: float
    range_error: float
    """Predicted downrange past the target (m); negative falls short."""
    flight_time: float
    """Predicted time (s) from now to the terminal speed."""


# ---------------------------------------------------------------------------
# Geometry on the sphere
# ---------------------------------------------------------------------------


def _central_angle(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def _bearing(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return float(np.arctan2(y, x))


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _target_inertial(target: tuple[float, float], epoch: float) -> tuple[float, float]:
    """Target (lon, lat) in the inertial frame aligned with the Earth at epoch 0."""
    return target[0] + EARTH_ROTATION_RATE * epoch, target[1]


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


#: Altitude (m) above which a prediction is declared a skip-out. Well above
#: any skip apogee a guided entry flies (Artemis I peaked near 88 km) and well
#: below the orbits an uncontrolled skip escapes to.
SKIP_OUT_ALTITUDE = 400.0e3

#: Range error (m) reported for a prediction that skips out: an overshoot
#: larger than any downrange a real trajectory can fly, so the monotone
#: ordering the corrector relies on still holds.
SKIP_OUT_ERROR = 4.0e7


def predict_constant_bank(
    state: _FloatArray,
    epoch: float,
    target: tuple[float, float],
    body: EntryBody,
    vertical_lift_fraction: float,
    *,
    terminal_speed: float,
    lift_to_drag: float | None = None,
    density_factor: float = 1.0,
    max_time: float = 3000.0,
    final_phase_drag: float = 0.0,
    final_vertical_lift_fraction: float | None = None,
    bank_now: float | None = None,
    roll_rate: float = 0.0,
) -> Prediction:
    """Fly the planar entry equations at constant :math:`\\cos\\sigma` to the terminal speed.

    ``state`` is the six-element entry vector ``[r, lon, lat, V, gamma, psi]``
    in the inertial frame, ``epoch`` the time (s) since that frame was aligned
    with the Earth. The range error compares the predicted downrange with the
    great-circle distance to the target site *at the predicted arrival time*.

    With ``final_vertical_lift_fraction`` given, the prediction is two-segment,
    the skip problem PredGuid's Up Control solves: ``vertical_lift_fraction``
    until the second entry -- drag falling below ``final_phase_drag`` on the
    way out and rising back through it -- and the final fraction after it.

    With ``roll_rate`` and ``bank_now`` given, the prediction starts at the bank
    the body is actually holding and slews toward the commanded one at that
    rate, instead of assuming the command is reached instantly. The lag is
    worth predicting rather than discovering: while the body rolls it is not
    yet flying the vertical lift the solution asked for, and at lunar-return
    speed a few seconds of the wrong lift moves the skip apogee by kilometres.

    A prediction that climbs past :data:`SKIP_OUT_ALTITUDE`, or has not slowed
    to the terminal speed by ``max_time``, never comes back on its own: its
    range error is :data:`SKIP_OUT_ERROR`, an overshoot, rather than whatever
    downrange it happened to reach when the integration stopped.
    """
    r0, lon, lat, v0, g0, _psi = (float(x) for x in state)
    ld = body.lift_to_drag if lift_to_drag is None else float(lift_to_drag)
    u_skip = float(np.clip(vertical_lift_fraction, -1.0, 1.0))

    def vertical(t: float, u_target: float) -> float:
        """Vertical lift fraction at time ``t`` into the prediction.

        Instantaneous when no roll rate is given. Otherwise the bank magnitude
        slews from the one now held toward ``arccos(u_target)``, and the
        vertical fraction is the cosine of where it has got to.
        """
        if roll_rate <= 0.0 or bank_now is None:
            return u_target
        start = abs(float(bank_now))
        target = float(np.arccos(np.clip(u_target, -1.0, 1.0)))
        reached = start + np.clip(target - start, -roll_rate * t, roll_rate * t)
        return float(np.cos(reached))

    def drag_of(y: _FloatArray) -> float:
        rho = density_factor * body.air_density(float(y[0]) - _R_EARTH)
        return 0.5 * rho * float(y[2]) ** 2 / body.ballistic_coefficient

    def rhs(t: float, y: _FloatArray, u_target: float) -> _FloatArray:
        r, _s, v, gamma = y
        drag = drag_of(y)
        gravity = _MU / (r * r)
        u = vertical(t, u_target)
        return np.array([
            v * np.sin(gamma),
            v * np.cos(gamma) / r,
            -drag - gravity * np.sin(gamma),
            (ld * drag * u + (v * v / r - gravity) * np.cos(gamma)) / v,
        ])

    def slow(_t: float, y: _FloatArray, _u: float) -> float:
        return float(y[2] - terminal_speed)

    def ground(_t: float, y: _FloatArray, _u: float) -> float:
        return float(y[0] - _R_EARTH)

    def escaped(_t: float, y: _FloatArray, _u: float) -> float:
        return float(_R_EARTH + SKIP_OUT_ALTITUDE - y[0])

    def drag_crossing(_t: float, y: _FloatArray, _u: float) -> float:
        return drag_of(y) - final_phase_drag

    for event in (slow, ground, escaped, drag_crossing):
        event.terminal = True  # type: ignore[attr-defined]
    stop = (slow, ground, escaped)

    y = np.array([r0, 0.0, v0, g0])
    clock = 0.0
    segments: list[tuple[float, tuple[object, ...]]] = []
    if final_vertical_lift_fraction is None or final_phase_drag <= 0.0:
        segments.append((u_skip, stop))
    else:
        u_final = float(np.clip(final_vertical_lift_fraction, -1.0, 1.0))
        exited = drag_of(y) < final_phase_drag
        drag_crossing.direction = -1.0  # type: ignore[attr-defined]
        if not exited:
            segments.append((u_skip, (*stop, drag_crossing)))  # out of the first pass
        segments.append((u_skip, (*stop, _rising(drag_crossing))))  # to the second entry
        segments.append((u_final, stop))

    landed = False
    for u, events in segments:
        solution = scipy.integrate.solve_ivp(
            rhs, (clock, max_time), y, args=(u,), events=events,
            rtol=1e-6, atol=1e-6, max_step=20.0,
        )
        y = np.asarray(solution.y[:, -1])
        clock = float(solution.t[-1])
        hit = [i for i, times in enumerate(solution.t_events) if times.size]
        if not hit:
            break  # ran out of time
        first = min(hit, key=lambda i: solution.t_events[i][0])
        if first in (0, 1):
            landed = True
            break
        if first == 2:
            break  # skipped out
        # otherwise a drag crossing: go on to the next segment
    if not landed:
        return Prediction(u_skip, SKIP_OUT_ERROR, clock)
    flown = float(y[1]) * _R_EARTH
    t_lon, t_lat = _target_inertial(target, epoch + clock)
    remaining = _central_angle(lon, lat, t_lon, t_lat) * _R_EARTH
    return Prediction(u_skip, flown - remaining, clock)


def _rising(event: object) -> object:
    """``event`` as a terminal event that fires only on an upward crossing."""

    def rising(t: float, y: _FloatArray, u: float) -> float:
        return float(event(t, y, u))  # type: ignore[operator]

    rising.terminal = True  # type: ignore[attr-defined]
    rising.direction = 1.0  # type: ignore[attr-defined]
    return rising


# ---------------------------------------------------------------------------
# Guidance
# ---------------------------------------------------------------------------


@dataclass
class SkipGuidance:
    """The guidance loop: phase logic, estimators, corrector and lateral channel.

    Stateful by design, like the flight software it models: it carries the
    current phase, the warm-start solution, the bank sign and the estimator
    states from one cycle to the next. Call :meth:`command` once per cycle.
    """

    body: EntryBody
    target: tuple[float, float]
    config: SkipGuidanceConfig
    phase: SkipPhase = SkipPhase.INITIAL_ROLL
    vertical_lift_fraction: float = 1.0
    bank_sign: float = 0.0
    density_factor: float = 1.0
    flown_bank: float = 0.0
    lift_to_drag_estimate: float = field(default=0.0)
    last_prediction: Prediction | None = None
    reversals: int = 0
    _exited: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.lift_to_drag_estimate <= 0.0:
            self.lift_to_drag_estimate = self.body.lift_to_drag
        self.vertical_lift_fraction = float(np.cos(self.config.initial_bank))

    def _estimate(self, state: _FloatArray, sensed_drag: float, sensed_lift: float) -> None:
        if sensed_drag < self.config.estimator_drag_threshold:
            return
        altitude = float(state[0]) - _R_EARTH
        model_drag = self.body.drag_acceleration(altitude, float(state[3]))
        gain = min(self.config.guidance_period / self.config.estimator_time_constant, 1.0)
        self.density_factor += gain * (sensed_drag / model_drag - self.density_factor)
        self.lift_to_drag_estimate += gain * (
            sensed_lift / sensed_drag - self.lift_to_drag_estimate
        )

    def _predict(self, state: _FloatArray, epoch: float, u: float) -> Prediction:
        skipping = self.phase is not SkipPhase.FINAL and self.config.final_phase_drag > 0.0
        return predict_constant_bank(
            state, epoch, self.target, self.body, u,
            terminal_speed=self.config.terminal_speed,
            lift_to_drag=self.lift_to_drag_estimate,
            density_factor=self.density_factor,
            final_phase_drag=self.config.final_phase_drag,
            final_vertical_lift_fraction=(
                self.config.final_vertical_lift_fraction if skipping else None
            ),
            bank_now=self.flown_bank,
            roll_rate=self.config.roll_rate,
        )

    def _correct(self, state: _FloatArray, epoch: float) -> None:
        """Solve for the vertical lift fraction that nulls the range error.

        Range error rises with :math:`u` (more vertical lift flies further),
        so the solution is bracketed by the bank limit and full lift-up. The
        warm start is tried first; if it misses, the bound on the side the
        error points to is checked -- saturating there if even it cannot
        close the gap -- and the bracket is then narrowed by false position
        with the Illinois modification, bisecting instead whenever an
        endpoint is a skip-out, whose error is a sentinel, not a distance.
        """
        tolerance = self.config.range_tolerance
        if self.phase is SkipPhase.FINAL and self.config.final_range_tolerance > 0.0:
            tolerance = self.config.final_range_tolerance
        lower = float(np.cos(self.body.max_bank))
        u0 = float(np.clip(self.vertical_lift_fraction, lower, 1.0))
        warm = self._predict(state, epoch, u0)
        best = warm
        budget = self.config.max_iterations - 1
        if abs(warm.range_error) > tolerance and budget > 0:
            bound = self._predict(state, epoch, lower if warm.range_error > 0.0 else 1.0)
            budget -= 1
            if abs(bound.range_error) < abs(best.range_error):
                best = bound
            if np.sign(bound.range_error) == np.sign(warm.range_error):
                best = bound  # cannot close the gap: saturate at the bound
            else:
                lo, hi = (bound, warm) if bound.vertical_lift_fraction < u0 else (warm, bound)
                side = 0
                for _ in range(budget):
                    if abs(best.range_error) <= tolerance:
                        break
                    ua, ub = lo.vertical_lift_fraction, hi.vertical_lift_fraction
                    ea, eb = lo.range_error, hi.range_error
                    if SKIP_OUT_ERROR in (abs(ea), abs(eb)) or eb == ea:
                        um = 0.5 * (ua + ub)
                    else:
                        um = ub - eb * (ub - ua) / (eb - ea)
                    mid = self._predict(state, epoch, um)
                    if abs(mid.range_error) < abs(best.range_error):
                        best = mid
                    if mid.range_error > 0.0:
                        hi = mid
                        if side == 1:
                            lo = Prediction(lo.vertical_lift_fraction, 0.5 * lo.range_error,
                                            lo.flight_time)
                        side = 1
                    else:
                        lo = mid
                        if side == -1:
                            hi = Prediction(hi.vertical_lift_fraction, 0.5 * hi.range_error,
                                            hi.flight_time)
                        side = -1
        self.vertical_lift_fraction = best.vertical_lift_fraction
        self.last_prediction = best

    def _lateral(self, state: _FloatArray, epoch: float) -> None:
        _r, lon, lat, speed, _gamma, heading = (float(x) for x in state)
        # Steer for where the site will be on arrival, as the predictor does:
        # the Earth carries it several hundred kilometres east over a skip
        # entry, and a lateral channel aiming at where it is now chases it.
        arrival = epoch + (self.last_prediction.flight_time if self.last_prediction else 0.0)
        t_lon, t_lat = _target_inertial(self.target, arrival)
        distance = _central_angle(lon, lat, t_lon, t_lat)
        offset = _wrap(_bearing(lon, lat, t_lon, t_lat) - heading)
        crossrange_angle = float(np.arcsin(np.sin(distance) * np.sin(offset)))
        deadband = self.config.corridor_constant + self.config.corridor_quadratic * speed**2
        if self.bank_sign == 0.0:
            # The first decision picks a side -- toward the site -- and is not
            # a reversal: nothing has been flown the other way yet.
            self.bank_sign = float(np.sign(crossrange_angle)) or 1.0
        elif abs(crossrange_angle) > deadband:
            wanted = float(np.sign(crossrange_angle)) or 1.0
            if wanted != self.bank_sign:
                self.bank_sign = wanted
                self.reversals += 1

    def command(
        self,
        state: _FloatArray,
        epoch: float,
        sensed_drag: float,
        sensed_lift: float,
        flown_bank: float | None = None,
    ) -> float:
        """Signed bank command (rad) for this cycle.

        ``flown_bank`` is the bank the body is actually holding, which the
        predictor starts its slew from; omitted, the command is assumed to be
        held already.
        """
        if flown_bank is not None:
            self.flown_bank = float(flown_bank)
        speed = float(state[3])
        altitude_rate = speed * float(np.sin(state[4]))
        if speed <= self.config.terminal_speed:
            self.phase = SkipPhase.TERMINAL
        elif (
            self.phase is SkipPhase.INITIAL_ROLL
            and altitude_rate > self.config.initial_roll_exit_altitude_rate
            and sensed_drag > self.config.estimator_drag_threshold
        ):
            self.phase = SkipPhase.PREDICTOR_CORRECTOR
        elif self.phase is SkipPhase.PREDICTOR_CORRECTOR and self.config.final_phase_drag > 0.0:
            if sensed_drag < self.config.final_phase_drag:
                self._exited = True
            elif self._exited:
                self.phase = SkipPhase.FINAL

        if self.phase in (SkipPhase.PREDICTOR_CORRECTOR, SkipPhase.FINAL):
            self._estimate(state, sensed_drag, sensed_lift)
            self._correct(state, epoch)
        self._lateral(state, epoch)
        magnitude = float(np.arccos(np.clip(self.vertical_lift_fraction, -1.0, 1.0)))
        return self.bank_sign * min(magnitude, self.body.max_bank)


# ---------------------------------------------------------------------------
# Closed loop
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkipEntryResult:
    """A closed-loop skip entry."""

    times: _FloatArray
    states: _FloatArray
    """Shape ``(6, n)``: radius, longitude, latitude, speed, gamma, heading (inertial)."""
    bank: _FloatArray
    phases: tuple[tuple[float, SkipPhase], ...]
    """``(time, phase)`` at each phase transition."""
    reversals: int
    """Sign changes in the bank the body actually flew."""
    commanded_reversals: int
    """Reversals the guidance asked for. Larger than ``reversals`` when a
    command is superseded while the body is still rolling through one."""
    miss_distance: float
    """Great-circle distance (m) from the terminal point to the target site."""
    skip_apogee_altitude: float
    """Highest altitude (m) between the first and second atmospheric passes."""
    peak_deceleration_g: float

    @property
    def altitudes(self) -> _FloatArray:
        return np.asarray(self.states[0] - _R_EARTH)


def fly_skip_entry(
    truth: EntryBody,
    initial: EntryState,
    guidance: SkipGuidance,
    *,
    max_time: float = 3000.0,
) -> SkipEntryResult:
    """Fly ``truth`` from ``initial`` under ``guidance`` to the terminal speed.

    ``truth`` is the plant: its lift-to-drag ratio and atmosphere are what the
    body actually flies, and may differ from the ``guidance.body`` model, which
    is what the estimators are for. Epoch 0 is ``initial``; the inertial frame
    is aligned with the Earth there.
    """
    period = guidance.config.guidance_period
    rate = guidance.config.roll_rate
    flown_bank = 0.0
    state = initial.as_array()
    clock = 0.0
    times, history, banks = [0.0], [state.copy()], []
    phases: list[tuple[float, SkipPhase]] = [(0.0, guidance.phase)]
    peak_g = 0.0

    def rhs(_t: float, y: _FloatArray, bank: float) -> _FloatArray:
        return entry_dynamics(y, bank, truth)

    while clock < max_time and state[0] > _R_EARTH:
        altitude = float(state[0]) - _R_EARTH
        drag = truth.drag_acceleration(altitude, float(state[3]))
        lift = truth.lift_to_drag * drag
        peak_g = max(peak_g, float(np.hypot(drag, lift)) / 9.80665)
        before = guidance.phase
        bank = guidance.command(state, clock, drag, lift, flown_bank=flown_bank)
        if guidance.phase is not before:
            phases.append((clock, guidance.phase))
        if _finished(guidance):
            break
        if rate > 0.0:
            # Slew toward the command at the body's roll rate rather than
            # snapping to it: what is flown over the step is the bank the
            # body could actually reach in it.
            step_limit = rate * period
            flown_bank += float(np.clip(bank - flown_bank, -step_limit, step_limit))
            bank = flown_bank
        else:
            flown_bank = bank
        banks.append(bank)
        solution = scipy.integrate.solve_ivp(
            rhs, (clock, clock + period), state, args=(bank,), rtol=1e-8, atol=1e-8,
        )
        state = np.asarray(solution.y[:, -1])
        clock = float(solution.t[-1])
        times.append(clock)
        history.append(state.copy())

    trajectory = np.asarray(history).T
    altitudes = trajectory[0] - _R_EARTH
    flown = np.asarray(banks)
    banked = flown[flown != 0.0]
    flown_reversals = int(np.count_nonzero(np.diff(np.sign(banked)))) if banked.size else 0
    t_lon, t_lat = _target_inertial(guidance.target, clock)
    miss = _central_angle(float(state[1]), float(state[2]), t_lon, t_lat) * _R_EARTH
    return SkipEntryResult(
        times=np.asarray(times),
        states=trajectory,
        bank=np.asarray(banks),
        phases=tuple(phases),
        reversals=flown_reversals,
        commanded_reversals=guidance.reversals,
        miss_distance=float(miss),
        skip_apogee_altitude=_skip_apogee(altitudes),
        peak_deceleration_g=peak_g,
    )


def _finished(guidance: SkipGuidance) -> bool:
    """Whether guided flight is over. A function so the check reads the phase
    :meth:`SkipGuidance.command` has just set, not a narrowed copy of it."""
    return guidance.phase is SkipPhase.TERMINAL


def _skip_apogee(altitudes: _FloatArray) -> float:
    """Highest altitude after the first minimum and before the next descent below it."""
    h = np.asarray(altitudes)
    if h.size < 3:
        return float("nan")
    first_min = int(np.argmin(h[: max(3, h.size // 2)]))
    after = h[first_min:]
    return float(after.max()) if after.size else float("nan")
