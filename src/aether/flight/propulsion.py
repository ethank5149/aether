"""Finite-thrust arcs for the coupled flight simulator.

Why this exists
---------------

:class:`~aether.flight.simulator.FlightSimulator` integrated an *unpowered*
vehicle: ``out[layout.mass] = 0.0``, no thrust term, torque-free. That is
the right model for entry and for a ballistic coast, and it is why the
fractional-orbital animations were driven by
a stitched-geometry trajectory model instead — one of
Keplerian conics with a prescribed pitch program.

That split is the same defect a replay-oriented history class
was built to remove, one level up. A picture rendered from the geometry
model is faithful to the geometry model, and the geometry model is not the
physics engine: it carries no drag, no J2, no attitude, no mass depletion
and no gravity loss. This module supplies the missing piece so a fractional
orbital profile can be *flown* — boost, coast, deorbit burn, coast, entry —
through the one integrator that carries all of it.

What a burn is here
-------------------

A constant-thrust, constant-exhaust-velocity arc with a steering law. Mass
is depleted at :math:`\\dot m = -T/c`, so the acceleration grows through the
burn exactly as it does on a real stage, and the burn is *not* an impulse:
a 182 m/s deorbit at 30 kN takes about ten seconds, over which the vehicle
moves 70 km. Treating it as instantaneous is a common and usually harmless
approximation; it is not made here, because the whole point is that the
animation shows what the integrator did.

Steering laws
-------------

Three, and the choice matters more than the thrust magnitude:

* ``"prograde"`` / ``"retrograde"`` — along or against the inertial
  velocity. Correct for orbital manoeuvres, undefined at zero speed.
* ``"gravity_turn"`` — what a launch vehicle actually flies: hold vertical
  through the dense low atmosphere, pitch over by a small commanded kick,
  then steer **prograde** and let gravity do the turning. It needs a
  *plane* because at lift-off the velocity is zero and cannot define one.
* ``"pitch"`` — a commanded flight-path-angle program, vertical at ignition
  and running to ``final_angle`` as :math:`(1-\tau)^{n}`.

  ``"pitch"`` is available and is **not** what a launcher flies, which is
  worth stating because using it here failed instructively: commanding the
  angle as a fraction of *burn time* puts the thrust 17 degrees above
  horizontal at half the burn, by which point the vehicle is still at
  10 km. It then accelerates horizontally through dense air, dynamic
  pressure reaches 400 kPa against a real max-Q near 30, and drag exceeds
  thrust. The vehicle reached 13 km and 1,050 m/s in 245 s. A gravity turn
  does not have that failure mode because after the kick the thrust follows
  the velocity, so the trajectory turns only as fast as gravity turns it.

Segments are integrated one at a time rather than gated inside a single
solve. Thrust switching is a genuine discontinuity in the right-hand side,
and a stiff implicit integrator asked to step across one either rejects
steps until it crawls or quietly smears the event over a step. Splitting at
the discontinuity is both faster and more accurate, and it makes the burn
boundaries exact rather than resolved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["STANDARD_GRAVITY", "Burn", "thrust_direction"]

_FloatArray = NDArray[np.float64]

#: Standard gravity used to convert specific impulse to exhaust velocity.
STANDARD_GRAVITY = 9.80665


@dataclass(frozen=True)
class Burn:
    """One constant-thrust arc.

    Attributes
    ----------
    duration:
        Burn time (s).
    thrust:
        Thrust magnitude (N), constant through the arc.
    exhaust_velocity:
        Effective exhaust velocity :math:`c = I_{sp} g_0` (m/s).
    steering:
        ``"pitch"``, ``"prograde"`` or ``"retrograde"``.
    plane_normal:
        Unit normal of the trajectory plane, required by ``"pitch"``. The
        thrust stays in this plane throughout, which is what keeps a
        launch on the great circle through its aim point.
    final_angle:
        Commanded flight-path angle at the end of a ``"pitch"`` burn (rad),
        measured above the local horizontal. Near zero for a circular
        insertion.
    pitch_exponent:
        Shape of the pitch-over: the commanded angle runs as
        :math:`\\gamma_f + (\\pi/2 - \\gamma_f)(1-\\tau)^{n}`. Larger holds
        the vehicle vertical longer.
    label:
        What this burn is, carried through to the trajectory's events.
    """

    duration: float
    thrust: float
    exhaust_velocity: float
    steering: str = "prograde"
    plane_normal: _FloatArray | None = None
    final_angle: float = 0.0
    pitch_exponent: float = 2.5
    vertical_time: float = 12.0
    kick_time: float = 18.0
    kick_angle: float = 0.12
    label: str = "burn"

    def __post_init__(self) -> None:
        for name in ("duration", "thrust", "exhaust_velocity"):
            value = float(getattr(self, name))
            if not (np.isfinite(value) and value > 0.0):
                msg = f"{name} must be finite and > 0, got {value}"
                raise ValueError(msg)
        if self.steering not in ("gravity_turn", "pitch", "prograde", "retrograde"):
            msg = (
                f"steering must be 'gravity_turn', 'pitch', 'prograde' or "
                f"'retrograde', got {self.steering!r}"
            )
            raise ValueError(msg)
        if self.steering in ("gravity_turn", "pitch"):
            if self.plane_normal is None:
                msg = (
                    f"a {self.steering!r} burn needs plane_normal: at lift-off the "
                    "velocity is zero, so the trajectory plane cannot be inferred "
                    "from it"
                )
                raise ValueError(msg)
            normal = np.asarray(self.plane_normal, dtype=np.float64)
            if normal.shape != (3,) or not np.isfinite(normal).all():
                msg = f"plane_normal must be a finite 3-vector, got {self.plane_normal!r}"
                raise ValueError(msg)
            if float(np.linalg.norm(normal)) < 1e-12:
                msg = "plane_normal must be non-zero"
                raise ValueError(msg)

    @property
    def mass_flow(self) -> float:
        """Propellant consumption rate (kg/s)."""
        return float(self.thrust / self.exhaust_velocity)

    def ideal_delta_v(self, initial_mass: float) -> float:
        """Tsiolkovsky :math:`\\Delta v` for this arc, ignoring losses.

        Useful as a *bound*: the flown velocity change is always smaller,
        because gravity and drag take their share. Comparing the two is how
        the gravity loss of a boost gets measured rather than assumed.
        """
        spent = self.mass_flow * self.duration
        if spent >= initial_mass:
            msg = (
                f"burn consumes {spent:,.0f} kg of a {initial_mass:,.0f} kg vehicle; "
                "the mass would go non-positive before burnout"
            )
            raise ValueError(msg)
        return float(
            self.exhaust_velocity * np.log(initial_mass / (initial_mass - spent))
        )


def thrust_direction(
    burn: Burn,
    elapsed: float,
    position: _FloatArray,
    velocity: _FloatArray,
) -> _FloatArray:
    """Unit thrust direction for ``burn`` at ``elapsed`` seconds into it.

    Parameters
    ----------
    burn:
        The arc being flown.
    elapsed:
        Seconds since ignition, used only by the pitch program.
    position, velocity:
        Inertial state (m, m/s).

    Notes
    -----
    The pitch program builds its frame from the *position*, not the
    velocity, so it is defined on the pad. The horizontal direction is
    ``normalize(cross(plane_normal, r_hat))``, which stays in the
    trajectory plane and rotates with the vehicle as it flies downrange.
    """
    r = np.asarray(position, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)

    if burn.steering in ("prograde", "retrograde"):
        # Softened normalisation rather than a guard that raises. An
        # implicit integrator evaluates the right-hand side at *trial*
        # states inside its Newton iteration, and those can be far from any
        # physical state — including through zero velocity. Raising there
        # aborts a solve that would have converged. At any real state the
        # epsilon is 1e-6 m/s against orbital speed and the direction is
        # exact to machine precision; a retrograde burn genuinely starting
        # from rest is a setup error and is caught before integration.
        sign = 1.0 if burn.steering == "prograde" else -1.0
        return np.asarray(sign * v / np.sqrt(float(v @ v) + 1.0e-12))

    radius = float(np.linalg.norm(r))
    if radius < 1e-9:
        msg = "pitch steering is undefined at the body centre"
        raise ValueError(msg)
    up = r / radius
    normal = np.asarray(burn.plane_normal, dtype=np.float64)
    normal = normal / float(np.linalg.norm(normal))
    horizontal = np.cross(normal, up)
    span = float(np.linalg.norm(horizontal))
    if span < 1e-12:  # pragma: no cover - launch from the plane's pole
        msg = "plane_normal is parallel to the launch radius; no downrange exists"
        raise ValueError(msg)
    horizontal = horizontal / span

    if burn.steering == "pitch":
        fraction = float(np.clip(elapsed / burn.duration, 0.0, 1.0))
        angle = burn.final_angle + (0.5 * np.pi - burn.final_angle) * (
            1.0 - fraction
        ) ** burn.pitch_exponent
        return np.asarray(np.sin(angle) * up + np.cos(angle) * horizontal)

    # --- gravity turn: vertical, kick, then let gravity steer.
    if elapsed <= burn.vertical_time:
        return np.asarray(up)
    kicked = elapsed - burn.vertical_time
    if kicked < burn.kick_time:
        angle = 0.5 * np.pi - burn.kick_angle * (kicked / burn.kick_time)
        return np.asarray(np.sin(angle) * up + np.cos(angle) * horizontal)
    # After the kick the thrust follows the velocity. This is the whole
    # mechanism: the trajectory turns only as fast as gravity turns it, so
    # the vehicle cannot be commanded into the atmosphere sideways.
    return np.asarray(v / np.sqrt(float(v @ v) + 1.0e-12))
