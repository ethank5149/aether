"""The global state vector (Paper I, §3.5).

The bulk rigid-body kinematics are coupled to the spectral grids in a
*single* state vector (Eq. 3.20):

.. math::

    \\mathbf{X}_{\\mathrm{global}} = \\big[\\mathbf{r}_E,\\ \\mathbf{v}_E,\\
    \\mathbf{q}_{E2B},\\ \\bm{\\omega}_B,\\ m_{\\mathrm{bulk}},\\
    \\hat{\\mathbf{w}},\\ \\dot{\\hat{\\mathbf{w}}},\\ \\mathbf{H}\\big],

with :math:`\\hat{\\mathbf{w}}` the null-space-reduced structural
coordinate and the thermal block carried alongside. Six-degree-of-freedom
rotational states are retained because the coupling between structural
bending slope and IMU orientation is a first-order effect on the
navigation solution and cannot be recovered from translational states
alone.

**The dimension is fixed for the entire trajectory.** That is the whole
computational argument of the framework, and this module is where it
becomes checkable: :attr:`GlobalState.size` is a property of the
configuration, not of the flight regime, and
:meth:`FlightConfiguration.layout` is computed once at construction.
Nothing in the right-hand side can add or remove a degree of freedom.

This module carries *enthalpy-adjacent* temperature rather than enthalpy
as the thermal state. Paper I argues for enthalpy because it stays
continuous across the pyrolysis zone where :math:`c_p` varies sharply;
the charring solver of :mod:`aether.thermal` integrates temperature
directly, so temperature is what is carried here and the enthalpy
formulation is noted as the refinement it is, not silently claimed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = ["GlobalState", "StateLayout"]

_FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class StateLayout:
    """Fixed index layout of the global state vector.

    Computed once from the configuration; every slice below is a compile
    -time constant of the trajectory, which is what makes the batched
    rank-3 tensor argument of Paper I §5.2 available.
    """

    n_modes: int
    """Retained structural modes :math:`n_m`."""
    n_thermal: int
    """Thermal grid nodes :math:`N_T + 1`."""

    def __post_init__(self) -> None:
        if self.n_modes < 0:
            raise ValueError(f"n_modes must be >= 0, got {self.n_modes}")
        if self.n_thermal < 2:
            raise ValueError(f"n_thermal must be >= 2, got {self.n_thermal}")

    @property
    def position(self) -> slice:
        return slice(0, 3)

    @property
    def velocity(self) -> slice:
        return slice(3, 6)

    @property
    def quaternion(self) -> slice:
        return slice(6, 10)

    @property
    def angular_rate(self) -> slice:
        return slice(10, 13)

    @property
    def mass(self) -> int:
        return 13

    @property
    def modal_displacement(self) -> slice:
        return slice(14, 14 + self.n_modes)

    @property
    def modal_velocity(self) -> slice:
        start = 14 + self.n_modes
        return slice(start, start + self.n_modes)

    @property
    def temperature(self) -> slice:
        start = 14 + 2 * self.n_modes
        return slice(start, start + self.n_thermal)

    @property
    def densities(self) -> slice:
        start = 14 + 2 * self.n_modes + self.n_thermal
        return slice(start, start + 3 * self.n_thermal)

    @property
    def recession(self) -> int:
        return 14 + 2 * self.n_modes + 4 * self.n_thermal

    @property
    def size(self) -> int:
        """Total state dimension — constant for the whole trajectory."""
        return 15 + 2 * self.n_modes + 4 * self.n_thermal


@dataclass(frozen=True)
class GlobalState:
    """Unpacked view of the global state at one instant."""

    position: _FloatArray
    velocity: _FloatArray
    quaternion: _FloatArray
    angular_rate: _FloatArray
    mass: float
    modal_displacement: _FloatArray = field(repr=False)
    modal_velocity: _FloatArray = field(repr=False)
    temperature: _FloatArray = field(repr=False)
    densities: _FloatArray = field(repr=False)
    recession: float

    @classmethod
    def unpack(cls, vector: _FloatArray, layout: StateLayout) -> GlobalState:
        """Split the flat ODE vector into named blocks."""
        y = np.asarray(vector, dtype=np.float64)
        if y.shape != (layout.size,):
            raise ValueError(
                f"state vector must have shape ({layout.size},), got {y.shape}"
            )
        return cls(
            position=y[layout.position],
            velocity=y[layout.velocity],
            quaternion=y[layout.quaternion],
            angular_rate=y[layout.angular_rate],
            mass=float(y[layout.mass]),
            modal_displacement=y[layout.modal_displacement],
            modal_velocity=y[layout.modal_velocity],
            temperature=y[layout.temperature],
            densities=y[layout.densities].reshape(3, layout.n_thermal),
            recession=float(y[layout.recession]),
        )

    def pack(self, layout: StateLayout) -> _FloatArray:
        """Flatten back into the ODE vector."""
        y = np.empty(layout.size)
        y[layout.position] = self.position
        y[layout.velocity] = self.velocity
        y[layout.quaternion] = self.quaternion
        y[layout.angular_rate] = self.angular_rate
        y[layout.mass] = self.mass
        y[layout.modal_displacement] = self.modal_displacement
        y[layout.modal_velocity] = self.modal_velocity
        y[layout.temperature] = self.temperature
        y[layout.densities] = np.asarray(self.densities).reshape(-1)
        y[layout.recession] = self.recession
        return y

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def surface_temperature(self) -> float:
        """Wall temperature at the ablating surface (last thermal node)."""
        return float(self.temperature[-1])
