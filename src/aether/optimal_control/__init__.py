"""Trajectory optimisation: direct transcription and Pontryagin refinement.

General methods over an arbitrary :class:`OCPProblem`, none of which knows about
any particular vehicle. Legendre-Gauss-Lobatto pseudospectral collocation,
successive convexification, Pontryagin's minimum principle, and a Dymos backend
for problems where accuracy matters -- on the standard brachistochrone
benchmark, under the same SLSQP optimizer, Dymos is four orders of magnitude
closer to the analytic optimum than the in-house transcription.
"""

from aether.optimal_control.indirect import (
    HamiltonianSystem,
    IndirectSolution,
    control_jacobian,
    costate_dynamics,
    covector_estimate,
    hamiltonian,
    minimising_control,
    refine_indirect,
    refine_indirect_free_time,
)
from aether.optimal_control.pseudospectral import (
    OCPProblem,
    OCPSolution,
    differentiation_matrix,
    lgl_nodes,
    map_to_physical,
    map_to_tau,
    mesh_error,
    solve_ocp,
)

__all__ = [
    "HamiltonianSystem",
    "IndirectSolution",
    "OCPProblem",
    "OCPSolution",
    "control_jacobian",
    "costate_dynamics",
    "covector_estimate",
    "differentiation_matrix",
    "hamiltonian",
    "lgl_nodes",
    "map_to_physical",
    "map_to_tau",
    "mesh_error",
    "minimising_control",
    "refine_indirect",
    "refine_indirect_free_time",
    "solve_ocp",
]
