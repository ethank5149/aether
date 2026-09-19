"""Three-dimensional CFD for the aero tables and the dynamic derivatives.

Where this sits relative to :mod:`aether.aerodynamics.cfd`: that package is
the **axisymmetric** path — a body of revolution at zero incidence, solved in
the meridian plane, which is what makes the sub- and transonic band affordable
and what confines it to :math:`\\alpha = 0`. This one is the three-dimensional
path, and exists for the two things the meridian plane cannot reach: incidence,
and unsteady motion.

The pipeline, end to end::

    panels.spatular_wedge(...)          parametric body -> PanelModel + SurfaceGrid
      -> VehicleMesh.from_surface_grid  watertight, outward-wound triangles
      -> cfd.viscous_domain             prism boundary layer + farfield, .su2
      -> cfd.config.SU2Case             SU2 v8 case for the flight condition
      -> cfd.solver.SU2Solver           run, read forces  -> Coefficients
      -> tables.SweepRun                checkpointed sweep -> AeroTable

The last step is the point of the whole arrangement: an
:class:`~aether.aerodynamics.tables.AeroTable` from CFD is the same object as
one from the panel method, so everything downstream of it cannot tell which
solver produced a table and does not need to.
"""

from __future__ import annotations

from aether.cfd.ablation import (
    AblationSolver,
    AblationState,
    TPSMaterial,
    UnderResolvedGrid,
)
from aether.cfd.config import (
    FlowConditions,
    LowMach,
    Nonequilibrium,
    Numerics,
    PerfectGas,
    PitchingMotion,
    ReferenceFrame,
    Regime,
    SU2Case,
    regime_for,
)
from aether.cfd.damping import (
    PitchDamping,
    pitch_damping,
    pitch_damping_from_history,
)
from aether.cfd.geometries import (
    REFERENCE_BODIES,
    ReferenceBody,
    build_domain,
    build_surface,
)
from aether.cfd.plasma import (
    BlackoutWindow,
    attenuation_db,
    collision_frequency,
    critical_density,
    electron_density_from_solution,
    plasma_frequency,
    saha_electron_density,
)
from aether.cfd.registry import (
    RunRecord,
    RunSpec,
    enqueue,
    load_all,
    reconcile,
    request_cancel,
)
from aether.cfd.run import SolveJob, SolveProgress, read_progress
from aether.cfd.solver import SU2Run, SU2Solver, read_history
from aether.cfd.sweep import cfd_grid, sweep_cfd_table, write_hdf5

__all__ = [
    "REFERENCE_BODIES",
    "AblationSolver",
    "AblationState",
    "BlackoutWindow",
    "FlowConditions",
    "LowMach",
    "Nonequilibrium",
    "Numerics",
    "PerfectGas",
    "PitchDamping",
    "PitchingMotion",
    "ReferenceBody",
    "ReferenceFrame",
    "Regime",
    "RunRecord",
    "RunSpec",
    "SU2Case",
    "SU2Run",
    "SU2Solver",
    "SolveJob",
    "SolveProgress",
    "TPSMaterial",
    "UnderResolvedGrid",
    "attenuation_db",
    "build_domain",
    "build_surface",
    "cfd_grid",
    "collision_frequency",
    "critical_density",
    "electron_density_from_solution",
    "enqueue",
    "load_all",
    "pitch_damping",
    "pitch_damping_from_history",
    "plasma_frequency",
    "read_history",
    "read_progress",
    "reconcile",
    "regime_for",
    "request_cancel",
    "saha_electron_density",
    "sweep_cfd_table",
    "write_hdf5",
]
