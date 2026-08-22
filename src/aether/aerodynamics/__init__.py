"""Aerodynamics from the launch pad to Mach 25 in free-molecular flow.

Five methods, each valid somewhere and none valid everywhere:

``closure`` / ``panels``
    Blended modified-Newtonian and Prandtl-Meyer impact theory. Supersonic
    and above; milliseconds a point.
``cfd``
    Axisymmetric Euler through SU2 and gmsh, for the sub- and transonic
    band the panel method cannot describe. Validated against
    :func:`~aether.aerodynamics.conical.solve_cone`.
``friction``
    Compressible boundary layer by the reference-temperature method, with
    the laminar branch checked against a solution of the boundary-layer
    equations rather than another correlation.
``realgas``
    Equilibrium normal shock over eleven-species ionising air, which is
    where :math:`C_{p,\\max}` stops being 1.839.
``rarefied``
    Schaaf-Chambre free-molecular flow, bridged across the transitional
    regime.

:class:`~aether.aerodynamics.composite.PatchedSolver` assembles them into one
coefficient source and reports which one it used.
"""

from __future__ import annotations

from aether.aerodynamics.closure import (
    blended_pressure_coefficient,
    newtonian_pressure_coefficient,
    prandtl_meyer_angle,
    prandtl_meyer_pressure_coefficient,
    rayleigh_pitot_cp_max,
    smoothstep,
    vacuum_pressure_coefficient,
)
from aether.aerodynamics.composite import (
    PatchedSolver,
    SkinFrictionModel,
    meridian_running_length,
)
from aether.aerodynamics.conical import (
    ConeSolution,
    ObliqueShock,
    mach_angle,
    maximum_cone_angle,
    oblique_shock,
    solve_cone,
    wedge_shock_angle,
)
from aether.aerodynamics.friction import (
    AdiabaticWall,
    BlasiusSolution,
    BoundaryLayer,
    FixedWall,
    RadiativeEquilibriumWall,
    adiabatic_wall_temperature,
    compressible_blasius,
    eckert_reference_temperature,
    laminar_skin_friction,
    recovery_factor,
    reference_temperature,
    turbulent_skin_friction,
)
from aether.aerodynamics.panels import PanelModel, TrimSolution, curved_lifting_body
from aether.aerodynamics.rarefied import (
    FreeMolecularSolver,
    free_molecular_coefficients,
    sine_squared_bridge,
    sphere_free_molecular_drag,
)
from aether.aerodynamics.realgas import (
    EquilibriumAir,
    NormalShock,
    perfect_gas_normal_shock,
)
from aether.aerodynamics.tables import (
    AeroTable,
    Coefficients,
    PanelSolver,
    SweepGrid,
    SweepRun,
    console_progress,
)

__all__ = [
    "AdiabaticWall",
    "AeroTable",
    "BlasiusSolution",
    "BoundaryLayer",
    "Coefficients",
    "ConeSolution",
    "EquilibriumAir",
    "FixedWall",
    "FreeMolecularSolver",
    "NormalShock",
    "ObliqueShock",
    "PanelModel",
    "PanelSolver",
    "PatchedSolver",
    "RadiativeEquilibriumWall",
    "SkinFrictionModel",
    "SweepGrid",
    "SweepRun",
    "TrimSolution",
    "adiabatic_wall_temperature",
    "blended_pressure_coefficient",
    "compressible_blasius",
    "console_progress",
    "curved_lifting_body",
    "eckert_reference_temperature",
    "free_molecular_coefficients",
    "laminar_skin_friction",
    "mach_angle",
    "maximum_cone_angle",
    "meridian_running_length",
    "newtonian_pressure_coefficient",
    "oblique_shock",
    "perfect_gas_normal_shock",
    "prandtl_meyer_angle",
    "prandtl_meyer_pressure_coefficient",
    "rayleigh_pitot_cp_max",
    "recovery_factor",
    "reference_temperature",
    "sine_squared_bridge",
    "smoothstep",
    "solve_cone",
    "sphere_free_molecular_drag",
    "turbulent_skin_friction",
    "vacuum_pressure_coefficient",
    "wedge_shock_angle",
]
