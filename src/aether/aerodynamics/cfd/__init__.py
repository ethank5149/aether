"""Axisymmetric Euler CFD: the sub- and transonic part of the envelope.

gmsh builds the meridian-plane domain, SU2 solves it, and the wall pressure
is integrated here rather than read from SU2's own force output. Validated
against :func:`aether.aerodynamics.conical.solve_cone` — a 15-degree cone at
Mach 3 comes back within 0.08 % of the exact Taylor–Maccoll pressure
coefficient on the finest mesh, with a Richardson limit 0.06 % from it.
"""

from __future__ import annotations

from aether.aerodynamics.cfd.meshing import (
    BodyProfile,
    BoundaryLayerTruncated,
    DomainSizing,
    MeshResult,
    ViscousSizing,
    axisymmetric_domain,
    boundary_layer_thickness,
    cone_profile,
    inviscid_domain,
    profile_from_arrays,
    viscous_domain,
    wall_spacing_for_y_plus,
)
from aether.aerodynamics.cfd.solver import (
    EulerSolver,
    GridConvergence,
    grid_convergence,
)
from aether.aerodynamics.cfd.su2 import (
    SU2Result,
    SU2Settings,
    find_su2,
    run_su2,
    surface_axial_force,
)

__all__ = [
    "BodyProfile",
    "BoundaryLayerTruncated",
    "DomainSizing",
    "EulerSolver",
    "GridConvergence",
    "MeshResult",
    "SU2Result",
    "SU2Settings",
    "ViscousSizing",
    "axisymmetric_domain",
    "boundary_layer_thickness",
    "cone_profile",
    "find_su2",
    "grid_convergence",
    "inviscid_domain",
    "profile_from_arrays",
    "run_su2",
    "surface_axial_force",
    "viscous_domain",
    "wall_spacing_for_y_plus",
]
