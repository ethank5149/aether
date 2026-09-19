"""Declarative phase termination triggers — closing a phase on a condition.

A staged trajectory does not end its phases at arbitrary clock times: a boost
phase ends when the propellant is gone, a coast ends at a deorbit altitude, a
glide hands over at a target speed. The scenario schema for a defence-grade
engine states these as declarative *termination triggers* — a parameter, a
comparison, a value — and the multi-phase propagator runs each phase until its
trigger fires (the sensor-fusion reference's phase state machine).

In a pseudospectral transcription with free phase durations, a termination
trigger is exactly an *interior boundary condition*: the phase's terminal state
must satisfy ``g(x_terminal) == 0`` (mass equals the burnout mass, altitude
equals the deorbit altitude, speed equals the handover speed). This module turns
a declarative :class:`PhaseTrigger` into that residual, which
:class:`~aether.trajectory.problem.PhaseSpec` carries as a
``terminal_constraints`` entry and the solver enforces at the phase boundary. The
free duration then adjusts until the condition is met — the phase ends when
physics says it does, not at a guessed time.

The state convention matches the trajectory module's dynamics: ``x[0:3]`` is the
inertial position and ``x[3:6]`` the inertial velocity; a mass or custom
component is addressed by its index.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

_FloatArray = NDArray[np.float64]

_EARTH_RADIUS = 6.371e6

__all__ = [
    "PhaseTrigger",
    "TriggerParameter",
    "altitude_trigger",
    "mass_trigger",
    "speed_trigger",
    "state_component_trigger",
]


class TriggerParameter(Enum):
    """The physical quantity a phase terminates on."""

    STATE_COMPONENT = "state-component"  # a raw state index reaches a value
    ALTITUDE = "altitude"                # |r| - R_earth reaches a value (m)
    SPEED = "speed"                      # |v| reaches a value (m/s)
    MASS = "mass"                        # the mass component reaches a value (kg)


@dataclass(frozen=True)
class PhaseTrigger:
    """A declarative condition that closes a phase when its residual is zero.

    Attributes
    ----------
    parameter:
        Which quantity terminates the phase.
    value:
        The value it must reach at the phase boundary (units of the parameter).
    component_index:
        The state index for :attr:`TriggerParameter.STATE_COMPONENT` and
        :attr:`TriggerParameter.MASS`.
    body_radius:
        Reference radius for an altitude trigger (m).
    """

    parameter: TriggerParameter
    value: float
    component_index: int | None = None
    body_radius: float = _EARTH_RADIUS

    def __post_init__(self) -> None:
        needs_index = self.parameter in (
            TriggerParameter.STATE_COMPONENT,
            TriggerParameter.MASS,
        )
        if needs_index and self.component_index is None:
            raise ValueError(f"{self.parameter.value} trigger needs a component_index")

    def residual(self, x_terminal: _FloatArray) -> float:
        """``g(x) == 0`` at the trigger — the interior boundary condition."""
        x = np.asarray(x_terminal, dtype=np.float64)
        if self.parameter is TriggerParameter.ALTITUDE:
            return float(np.linalg.norm(x[0:3]) - self.body_radius - self.value)
        if self.parameter is TriggerParameter.SPEED:
            return float(np.linalg.norm(x[3:6]) - self.value)
        # STATE_COMPONENT and MASS both address a raw component.
        assert self.component_index is not None  # guarded in __post_init__
        return float(x[self.component_index] - self.value)

    def as_constraint(self) -> Callable[[_FloatArray], float]:
        """The residual as a plain callable for ``PhaseSpec.terminal_constraints``."""
        return self.residual


def mass_trigger(mass_index: int, burnout_mass: float) -> PhaseTrigger:
    """Close a phase when the mass component reaches ``burnout_mass`` (kg)."""
    return PhaseTrigger(TriggerParameter.MASS, burnout_mass, component_index=mass_index)


def altitude_trigger(altitude_m: float, body_radius: float = _EARTH_RADIUS) -> PhaseTrigger:
    """Close a phase when the vehicle crosses ``altitude_m`` (m)."""
    return PhaseTrigger(TriggerParameter.ALTITUDE, altitude_m, body_radius=body_radius)


def speed_trigger(speed_mps: float) -> PhaseTrigger:
    """Close a phase when the speed reaches ``speed_mps`` (m/s)."""
    return PhaseTrigger(TriggerParameter.SPEED, speed_mps)


def state_component_trigger(index: int, value: float) -> PhaseTrigger:
    """Close a phase when state component ``index`` reaches ``value``."""
    return PhaseTrigger(TriggerParameter.STATE_COMPONENT, value, component_index=index)
