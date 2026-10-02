r"""Entry flown from the compiled symbolic field.

Every number of motion this module produces comes out of a routine printed
from an expression in :mod:`aether.symbolic`: the translational field of
record over a rotating, oblate planet, the atmosphere on the reference
ellipsoid, the drag and lift a body at trim puts on it. Nothing here restates
an equation of motion. What is here is what a field is not: which body, which
planet, how long to integrate, and when to stop.

:class:`EntryModel` is a body on a planet. It plays the two parts a guided
entry has. As the *plant* it advances a state under a held bank angle and
reports what an accelerometer would sense. As the guidance's *model* it
predicts where a constant vertical lift fraction leads, from the same field
with the lateral lift left out. A flight that differs from its guidance's
model -- thinner air, less lift -- is two :class:`EntryModel` objects with
different numbers and one compiled field.

The state is the field's, :math:`(r, \lambda, \phi, V, \gamma, \psi)`:
geocentric radius, longitude and geocentric latitude, and the speed,
flight-path angle and heading of the velocity *relative to the rotating
planet*. A landing site is therefore a fixed point, and an entry interface
published as a geodetic position and an inertial velocity has to be brought
into this state first; :func:`interface_state` does it, from the expressions
of :mod:`aether.symbolic.figure`.

What is not generated is the integration scheme (a fixed-step Runge--Kutta
routine the generator emits beside every field, driven with step halving),
the location of events between its steps, and the conversion of a state to
geodetic coordinates for a report, which is the kernel of
:mod:`aether.ellipsoid`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aether.aerothermal.stagnation import SUTTON_GRAVES_EARTH
from aether.certification.rotating_field import EARTH, WGS84_FLATTENING, Planet
from aether.ellipsoid import Ellipsoid, ecef_to_geodetic
from aether.geodesy import GeodeticPosition
from aether.guidance.footprint import FlownSchedule
from aether.guidance.skip import SKIP_OUT_ALTITUDE, SKIP_OUT_ERROR, Prediction
from aether.symbolic.codegen import CompiledField, LambdifiedField, compile_field
from aether.symbolic.figure import FigureSymbols
from aether.symbolic.system import CompiledSystem
from aether.symbolic.trim import entry_system, prediction_system

__all__ = [
    "EARTH_WGS84",
    "EntryModel",
    "EntryVehicle",
    "geocentric_site",
    "interface_state",
]

_FloatArray = NDArray[np.float64]

#: Earth with its altitude measured from the WGS84 ellipsoid.
EARTH_WGS84 = EARTH.on_ellipsoid(WGS84_FLATTENING)

@dataclass(frozen=True)
class EntryVehicle:
    """What the trimmed field needs to know of a body.

    Attributes
    ----------
    ballistic_coefficient:
        :math:`m/(C_D S)` (kg/m²).
    lift_to_drag:
        :math:`L/D` at trim.
    nose_radius:
        Radius (m) the stagnation heating correlation is evaluated at.
    max_bank:
        Largest bank magnitude the body will hold (rad).
    density_scale:
        Density of the air this body flies through over the model
        atmosphere's. One for the atmosphere as modelled.
    heating_constant:
        Constant of the convective correlation, SI; Sutton and Graves's value
        for air.
    """

    ballistic_coefficient: float
    lift_to_drag: float
    nose_radius: float
    max_bank: float = float(np.pi)
    density_scale: float = 1.0
    heating_constant: float = SUTTON_GRAVES_EARTH

    def __post_init__(self) -> None:
        for name in ("ballistic_coefficient", "lift_to_drag", "nose_radius", "density_scale"):
            value = getattr(self, name)
            if not (np.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and > 0, got {value}")
        if not 0.0 < self.max_bank <= np.pi:
            raise ValueError(f"max_bank must lie in (0, pi], got {self.max_bank}")


@lru_cache(maxsize=1)
def _plant() -> CompiledSystem:
    return entry_system().compile(jacobian=False)


@lru_cache(maxsize=1)
def _predictor() -> CompiledSystem:
    return prediction_system().compile(jacobian=False)


@lru_cache(maxsize=2)
def _interface(geodetic_horizon: bool) -> CompiledField | LambdifiedField:
    figure = FigureSymbols(geodetic_horizon=geodetic_horizon)
    planet = figure.planet
    return compile_field(
        figure.model_state,
        figure.published,
        (planet.mu, planet.radius, planet.j2, planet.rotation_rate, planet.flattening),
        name="interface_state",
        jacobian=False,
    )


def _planet_values(planet: Planet) -> dict[str, float]:
    return {
        "mu": float(planet.mu),
        "R_e": float(planet.radius),
        "J_2": float(planet.j2),
        "omega_E": float(planet.rotation_rate),
        "f_E": float(planet.flattening),
    }


def interface_state(
    geodetic_altitude: float,
    longitude: float,
    geodetic_latitude: float,
    inertial_speed: float,
    inertial_flight_path_angle: float,
    inertial_azimuth: float,
    *,
    planet: Planet = EARTH_WGS84,
    geodetic_horizon: bool = True,
) -> _FloatArray:
    r"""The model's state from an entry interface as a flight report gives it.

    In: geodetic altitude (m) and latitude, longitude, and the inertial
    velocity by its magnitude, flight-path angle and azimuth in the
    topocentric frame. Out: :math:`(r, \lambda, \phi, V, \gamma, \psi)`,
    geocentric and relative to the rotating planet.

    ``geodetic_horizon`` says what the published angles are measured from. A
    topocentric frame has the ellipsoid's normal for its vertical, which is
    the default; pass ``False`` for angles referred to the geocentric
    horizon. At mid latitudes the two differ by :math:`0.15^\circ` in the
    flight-path angle of a body flying along a meridian, which is not small
    for an entry.
    """
    published = [
        geodetic_altitude, longitude, geodetic_latitude,
        inertial_speed, inertial_flight_path_angle, inertial_azimuth,
    ]
    values = _planet_values(planet)
    constants = [values[name] for name in ("mu", "R_e", "J_2", "omega_E", "f_E")]
    return _interface(bool(geodetic_horizon))(published, constants)


def geocentric_site(
    longitude: float, geodetic_latitude: float, *, planet: Planet = EARTH_WGS84
) -> tuple[float, float]:
    """A surface site as the guidance needs it: ``(longitude, geocentric latitude)``.

    The geocentric latitude is that of the point on the ellipsoid, not of the
    geodetic latitude read as an angle at the centre; the two are 17 km apart
    on the ground at mid latitudes.
    """
    state = interface_state(
        0.0, longitude, geodetic_latitude, 1.0, 0.0, 0.0, planet=planet
    )
    return float(state[1]), float(state[2])


@dataclass
class EntryModel:
    """A body at trim on a planet, flown from the compiled field.

    Attributes
    ----------
    vehicle:
        The body.
    planet:
        The planet, with the flattening of the surface its atmosphere lies on.
    step:
        Largest step (s) the plant is integrated at. It is halved until the
        result holds to ``rtol``, so this is a starting point and not an
        accuracy.
    prediction_step:
        Fixed step (s) of the guidance's predictions, which are solved many
        times a cycle and to a tolerance of kilometres.
    """

    vehicle: EntryVehicle
    planet: Planet = EARTH_WGS84
    step: float = 0.25
    rtol: float = 1e-10
    prediction_step: float = 1.0
    #: A site fixed to the planet does not move in this model's frame.
    target_rotation_rate: float = field(default=0.0, init=False)

    # -- what the guidance reads of its model --------------------------------------

    @property
    def lift_to_drag(self) -> float:
        return self.vehicle.lift_to_drag

    @property
    def max_bank(self) -> float:
        return self.vehicle.max_bank

    @property
    def ballistic_coefficient(self) -> float:
        return self.vehicle.ballistic_coefficient

    @property
    def backend(self) -> str:
        """``"c"`` when the field was compiled, ``"numpy"`` when it runs through NumPy."""
        return _plant().backend

    # -- parameters --------------------------------------------------------------------

    def _plant_parameters(self, bank: float) -> _FloatArray:
        vehicle = self.vehicle
        return _plant().parameter_vector(
            _planet_values(self.planet),
            sigma=bank,
            beta=vehicle.ballistic_coefficient,
            E=vehicle.lift_to_drag,
            k_rho=vehicle.density_scale,
            R_n=vehicle.nose_radius,
            k_Q=vehicle.heating_constant,
        )

    def _prediction_parameters(
        self, bank_now: float, bank_goal: float, roll_rate: float,
        lift_to_drag: float, density_factor: float,
    ) -> _FloatArray:
        return _predictor().parameter_vector(
            _planet_values(self.planet),
            sigma_0=bank_now,
            sigma_c=bank_goal,
            sigmadot_max=roll_rate,
            beta=self.vehicle.ballistic_coefficient,
            E=lift_to_drag,
            k_rho=density_factor * self.vehicle.density_scale,
        )

    # -- the plant ---------------------------------------------------------------------

    @staticmethod
    def initial_state(entry: ArrayLike) -> _FloatArray:
        """The plant's state from a six-element entry vector: arc and heat load at zero."""
        vector = np.asarray(
            entry.as_array() if hasattr(entry, "as_array") else entry, dtype=np.float64
        )
        if vector.shape == (8,):
            return vector.copy()
        if vector.shape != (6,):
            raise ValueError(f"an entry state has six components, got shape {vector.shape}")
        return np.concatenate([vector, [0.0, 0.0]])

    def advance(self, state: ArrayLike, bank: float, duration: float) -> _FloatArray:
        """The plant's state after ``duration`` seconds at a held bank."""
        return _plant().advance(
            self.initial_state(state), self._plant_parameters(bank), duration,
            step=self.step, rtol=self.rtol,
        )

    def rates(self, state: ArrayLike, bank: float) -> _FloatArray:
        """Rates of the plant's state, straight from the compiled field."""
        return _plant().rates(self.initial_state(state), self._plant_parameters(bank))

    def outputs(self, state: ArrayLike) -> dict[str, float]:
        """Altitude, density, sensed accelerations, loads and energy at a state."""
        return _plant().outputs(self.initial_state(state), self._plant_parameters(0.0))

    def output_table(self, states: ArrayLike) -> dict[str, _FloatArray]:
        """The same, along a trajectory given as rows of plant states."""
        rows = np.asarray(states, dtype=np.float64)
        if rows.ndim != 2:
            raise ValueError(f"states must be rows of plant states, got shape {rows.shape}")
        if rows.shape[1] == 6:
            rows = np.hstack([rows, np.zeros((rows.shape[0], 2))])
        return _plant().output_table(rows, self._plant_parameters(0.0))

    def sensed(self, state: ArrayLike) -> tuple[float, float]:
        """Drag and lift accelerations (m/s²) an accelerometer would report."""
        values = self.outputs(state)
        return values["drag"], values["lift"]

    def altitude(self, state: ArrayLike) -> float:
        """Height above the reference ellipsoid (m)."""
        return self.outputs(state)["altitude"]

    def model_drag(self, state: ArrayLike) -> float:
        """Drag acceleration this model expects at a state: what an estimator compares with."""
        return self.outputs(state)["drag"]

    def audit(self, state: ArrayLike, bank: float = 0.0) -> float:
        """Largest relative gap between the compiled plant and its expressions."""
        return _plant().audit(self.initial_state(state), self._plant_parameters(bank))

    def ground_point(self, state: ArrayLike, label: str = "") -> GeodeticPosition:
        """The geodetic point a state is over, for a report or a map."""
        radius, longitude, latitude = (float(x) for x in np.asarray(state)[:3])
        position = radius * np.array(
            [
                np.cos(latitude) * np.cos(longitude),
                np.cos(latitude) * np.sin(longitude),
                np.sin(latitude),
            ]
        )
        ellipsoid = Ellipsoid(float(self.planet.radius), float(self.planet.flattening))
        geodetic_latitude, geodetic_longitude, _ = ecef_to_geodetic(position, ellipsoid)
        return GeodeticPosition(
            latitude=float(geodetic_latitude), longitude=float(geodetic_longitude), label=label
        )

    # -- the guidance's predictor --------------------------------------------------------

    def predict(
        self,
        state: ArrayLike,
        epoch: float,
        target: tuple[float, float],
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
        r"""Fly the field at constant :math:`\cos\sigma` to the terminal speed.

        ``target`` is ``(longitude, geocentric latitude)`` of a site fixed to
        the planet; ``epoch`` is accepted for the guidance's sake and unused,
        because in this frame the site does not move. The range error is the
        ground-track arc flown, less the central angle from the present
        position to the site, both in metres of equatorial radius.

        With ``final_vertical_lift_fraction`` the prediction has two parts,
        as a skip does: the given fraction until the second entry -- the drag
        falling below ``final_phase_drag`` on the way out and rising back
        through it -- and the final fraction after it.

        With ``roll_rate`` and ``bank_now`` the prediction starts at the bank
        the body holds and slews toward the commanded one at that rate.

        A prediction that climbs past :data:`SKIP_OUT_ALTITUDE`, or has not
        slowed to the terminal speed by ``max_time``, reports
        :data:`SKIP_OUT_ERROR`: an overshoot.
        """
        del epoch
        entry = np.asarray(state, dtype=np.float64)[:6]
        ratio = self.vehicle.lift_to_drag if lift_to_drag is None else float(lift_to_drag)
        u_skip = float(np.clip(vertical_lift_fraction, -1.0, 1.0))
        slewing = roll_rate > 0.0 and bank_now is not None

        def parameters(u_target: float) -> _FloatArray:
            goal = float(np.arccos(np.clip(u_target, -1.0, 1.0)))
            start = abs(float(bank_now)) if slewing and bank_now is not None else goal
            rate = float(roll_rate) if slewing else 1.0
            return self._prediction_parameters(start, goal, rate, ratio, density_factor)

        def drag_at(y: _FloatArray, p: _FloatArray) -> float:
            return _predictor().outputs(y, p)["drag"]

        y = np.concatenate([entry, [0.0, 0.0]])
        segments: list[tuple[float, str | None]] = []
        if final_vertical_lift_fraction is None or final_phase_drag <= 0.0:
            segments.append((u_skip, None))
        else:
            u_final = float(np.clip(final_vertical_lift_fraction, -1.0, 1.0))
            if drag_at(y, parameters(u_skip)) >= final_phase_drag:
                segments.append((u_skip, "falling"))  # out of the first pass
            segments.append((u_skip, "rising"))  # to the second entry
            segments.append((u_final, None))

        landed = False
        for u, crossing in segments:
            y, outcome = self._predict_segment(
                y, parameters(u), terminal_speed, max_time,
                final_phase_drag if crossing else 0.0, crossing,
            )
            if outcome == "landed":
                landed = True
                break
            if outcome != "crossed":
                break  # skipped out, or ran out of time
        clock = float(y[7])
        if not landed:
            return Prediction(u_skip, SKIP_OUT_ERROR, clock)
        radius = float(self.planet.radius)
        remaining = _central_angle(float(entry[1]), float(entry[2]), target[0], target[1])
        return Prediction(u_skip, (float(y[6]) - remaining) * radius, clock)

    def _predict_segment(
        self,
        y: _FloatArray,
        p: _FloatArray,
        terminal_speed: float,
        max_time: float,
        drag_level: float,
        crossing: str | None,
        chunk: int = 128,
    ) -> tuple[_FloatArray, str]:
        """Integrate one segment of a prediction to its first event.

        Returns the state at the event and which it was: ``"landed"`` (slow
        enough, or on the ground), ``"escaped"``, ``"crossed"`` (the drag
        level, in the direction asked for) or ``"timeout"``.
        """
        system = _predictor()
        dt = self.prediction_step

        def events(rows: _FloatArray) -> _FloatArray:
            table = system.output_table(rows, p)
            columns = [
                rows[:, 3] - terminal_speed,
                table["altitude"],
                SKIP_OUT_ALTITUDE - table["altitude"],
            ]
            if crossing == "falling":
                columns.append(table["drag"] - drag_level)
            elif crossing == "rising":
                columns.append(drag_level - table["drag"])
            return np.column_stack(columns)

        names = ["landed", "landed", "escaped", "crossed"]
        previous = events(y[np.newaxis, :])[0]
        while float(y[7]) < max_time:
            steps = max(1, min(chunk, int(np.ceil((max_time - float(y[7])) / dt))))
            rows = system.field.rk4(y, p, dt, steps, 1)
            values = events(rows)
            # An event fires where its function, positive before, is no longer.
            before = np.vstack([previous, values[:-1]])
            fired = (before > 0.0) & (values <= 0.0)
            if not fired.any():
                y, previous = rows[-1], values[-1]
                continue
            row = int(np.argmax(fired.any(axis=1)))
            start = y if row == 0 else rows[row - 1]
            which = int(np.argmax(fired[row]))
            g0, g1 = float(before[row, which]), float(values[row, which])
            fraction = g0 / (g0 - g1)
            landing = system.field.rk4(start, p, fraction * dt / 4.0, 4, 4)[-1]
            # One step of false position on the event function, inside whichever
            # half of the step still brackets it.
            g_mid = float(events(landing[np.newaxis, :])[0, which])
            if g_mid > 0.0:
                fraction += (1.0 - fraction) * g_mid / (g_mid - g1)
            elif g_mid < 0.0:
                fraction *= g0 / (g0 - g_mid)
            if g_mid != 0.0 and fraction > 0.0:
                landing = system.field.rk4(start, p, fraction * dt / 4.0, 4, 4)[-1]
            return landing, names[which]
        return y, "timeout"

    # -- sweeps --------------------------------------------------------------------------

    def fly_schedule(
        self,
        state: ArrayLike,
        bank_schedule: Sequence[tuple[float, float]],
        *,
        terminal_speed: float,
        max_time: float,
        step: float,
    ) -> FlownSchedule | None:
        """Fly a piecewise-constant bank history, sampled every ``step`` seconds.

        ``bank_schedule`` is ``[(until_time, bank), ...]``. The flight ends
        where the speed falls to the terminal speed or the altitude to zero,
        located inside the sample interval in which it happens, or at
        ``max_time``. It is not flown past its end: below the terminal speed
        a capsule under lift-down pitches over toward the vertical, where the
        heading is not defined and its rate is not bounded. Returns ``None``
        if an interval could not be integrated and did not contain the end.
        """
        y = self.initial_state(state)
        clock = 0.0
        times, rows = [0.0], [y.copy()]
        while clock < max_time and y[3] > terminal_speed and self.altitude(y) > 0.0:
            bank = bank_schedule[-1][1]
            for until, value in bank_schedule:
                if clock < until:
                    bank = value
                    break
            span = min(step, max_time - clock)
            try:
                after = self.advance(y, bank, span)
                ended = after[3] <= terminal_speed or self.altitude(after) <= 0.0
            except RuntimeError:
                after, ended = None, True
            if ended:
                found = self._end_of_flight(y, bank, span, terminal_speed)
                if found is None:
                    return None
                elapsed, y = found
                clock += elapsed
                times.append(clock)
                rows.append(y.copy())
                break
            assert after is not None
            y = after
            clock += span
            if not np.all(np.isfinite(y)):
                return None
            times.append(clock)
            rows.append(y.copy())
        table = self.output_table(np.asarray(rows))
        return FlownSchedule(
            times=np.asarray(times, dtype=np.float64),
            altitudes=table["altitude"],
            speeds=np.asarray([row[3] for row in rows], dtype=np.float64),
            final=y,
            heat_flux=table["heat_flux"],
            load_factor=table["load_factor"],
        )

    def _end_of_flight(
        self, state: _FloatArray, bank: float, span: float, terminal_speed: float,
        fine: float = 0.05,
    ) -> tuple[float, _FloatArray] | None:
        """Where, inside ``span``, the flight ends: the time into it and the state there.

        The interval is stepped finely and the first step across the terminal
        speed or the ground is cut at the crossing, by one step of false
        position on the quantity that crossed. ``None`` if neither is crossed
        before the state stops being finite.
        """
        system = _plant()
        parameters = self._plant_parameters(bank)
        steps = max(1, int(np.ceil(span / fine)))
        dt = span / steps
        samples = system.field.rk4(state, parameters, dt, steps, 1)
        finite = np.all(np.isfinite(samples), axis=1)
        usable = int(np.argmin(finite)) if not finite.all() else steps
        samples = samples[:usable]
        if not samples.size:
            return None
        altitude = system.output_table(samples, parameters)["altitude"]
        margins = np.column_stack([samples[:, 3] - terminal_speed, altitude])
        crossed = (margins <= 0.0).any(axis=1)
        if not crossed.any():
            return None
        row = int(np.argmax(crossed))
        which = int(np.argmax(margins[row] <= 0.0))
        before = state if row == 0 else samples[row - 1]
        start_margin = (
            before[3] - terminal_speed if which == 0
            else system.outputs(before, parameters)["altitude"]
        )
        fraction = float(start_margin / (start_margin - margins[row, which]))
        fraction = min(max(fraction, 0.0), 1.0)
        landing = before if fraction == 0.0 else system.field.rk4(
            before, parameters, fraction * dt / 4.0, 4, 4
        )[-1]
        return (row + fraction) * dt, np.asarray(landing, dtype=np.float64)


def _central_angle(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))

