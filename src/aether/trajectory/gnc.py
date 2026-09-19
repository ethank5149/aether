"""Hand an offline reference trajectory to a GNC loop as the feed-forward nominal.

A tracking guidance law needs, at each instant, the reference state to track and
the reference *rates* to feed forward — the nominal velocity and acceleration
that the control must produce even with zero tracking error (Zarchan, *Tactical
and Strategic Missile Guidance*, on the two-degree-of-freedom structure: a
feed-forward reference plus feedback on the error). This module is the thin
adapter that turns a :class:`ReferenceTrajectory` into exactly that, so the
optimised, serialised trajectory becomes the nominal a controller closes around.

It deliberately does no control: it reports the reference and its derivatives.
The feedback law (APN, midcourse, an LQR autopilot) consumes
:class:`ReferenceCommand` and adds its own error term.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from aether.trajectory.representation import ReferenceTrajectory

_FloatArray = NDArray[np.float64]

__all__ = ["ReferenceCommand", "TrajectoryFollower"]


@dataclass(frozen=True)
class ReferenceCommand:
    """The reference the controller feeds forward at one instant.

    Attributes
    ----------
    time:
        Query time (s).
    state:
        Full reference state at ``time``.
    velocity:
        First time-derivative of the state (reference rates).
    acceleration:
        Second time-derivative (reference feed-forward acceleration).
    control:
        Nominal control at ``time`` (empty for an unpowered phase).
    """

    time: float
    state: _FloatArray
    velocity: _FloatArray
    acceleration: _FloatArray
    control: _FloatArray


class TrajectoryFollower:
    """Serve feed-forward reference commands from a reference trajectory.

    Parameters
    ----------
    trajectory:
        The optimised reference (typically loaded from a
        :class:`~aether.trajectory.library.TrajectoryLibrary`).
    position_slice:
        Slice of the state vector that is position, used by
        :meth:`tracking_error`. Defaults to the first three components.
    """

    def __init__(
        self, trajectory: ReferenceTrajectory, position_slice: slice = slice(0, 3)
    ) -> None:
        self.trajectory = trajectory
        self.position_slice = position_slice

    @property
    def duration(self) -> float:
        return self.trajectory.t_end - self.trajectory.t_start

    def command(self, t: float) -> ReferenceCommand:
        """Reference state, rates, and nominal control at time ``t``."""
        return ReferenceCommand(
            time=float(t),
            state=self.trajectory.state(t),
            velocity=self.trajectory.derivative(t, 1),
            acceleration=self.trajectory.derivative(t, 2),
            control=self.trajectory.control(t),
        )

    def tracking_error(self, t: float, measured_state: _FloatArray) -> _FloatArray:
        """Position error ``measured - reference`` for the feedback law."""
        ref = self.trajectory.state(t)[self.position_slice]
        return np.asarray(measured_state, dtype=np.float64)[self.position_slice] - ref

    def is_active(self, t: float) -> bool:
        """Whether ``t`` lies within the reference's valid span."""
        return self.trajectory.t_start <= t <= self.trajectory.t_end
