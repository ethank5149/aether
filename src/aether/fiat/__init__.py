"""Fully implicit ablation and thermal response of a multilayer TPS stack.

An independent implementation of the formulation published in

* Y.-K. Chen and F. S. Milos, "Ablation and Thermal Response Program for
  Spacecraft Heatshield Analysis," *J. Spacecraft and Rockets* **36**(3),
  1999, pp. 475–483, doi:10.2514/2.3469;
* F. S. Milos, Y.-K. Chen and T. H. Squire, "Updated Ablation and Thermal
  Response Program for Spacecraft Heatshield Analysis," TFAWS06-1008,
  17th Thermal and Fluids Analysis Workshop, 2006.

FIAT itself is US-government-controlled software. It is not used,
invoked, or reproduced here; this package is written from the governing
equations and numerical description in the two open-literature papers
above, which is why it exists — a code whose verification criterion is
stated against FIAT cannot be closed without an independent solver of
the same equations.

Comparisons produced with this package are **code-to-code
cross-verification against an independent implementation of FIAT's
formulation**, and must be reported as such. They are not FIAT results.
"""

from __future__ import annotations

from aether.fiat.analysis import (
    DepthProbe,
    InterfaceHistory,
    ThicknessResult,
    interface_histories,
    optimize_ply_thickness,
    probe_depths,
    scale_environments,
    sized_stack,
)
from aether.fiat.bprime import BPrimeTable, TableRangeError
from aether.fiat.kinetics import (
    TgaTargets,
    calibrated_components,
    fit_arrhenius,
    peak_rate_temperature,
    tga_mass_fraction,
)
from aether.fiat.materials import (
    HERITAGE_PICA_CONDUCTIVITY,
    MEDLI2_PICA_CONDUCTIVITY,
    PressureConductivity,
    pica_like_material,
    structural_material,
)
from aether.fiat.permeability import (
    FIBERFORM_SAMPLES,
    MARSCHALL_MILOS_SAMPLES,
    PERMEABILITY_UNCERTAINTY,
    SLIP_PARAMETER_UNCERTAINTY,
    PermeabilitySample,
    effective_permeability,
    fiberform_permeability,
    knudsen_regime_pressure,
    slip_parameter,
)
from aether.fiat.pica_kinetics import (
    COMPETITIVE_PICA_BAYESIAN,
    COMPETITIVE_PICA_DETERMINISTIC,
    PARALLEL_PICA_RESIN,
    CompetitivePica,
    ParallelReaction,
    advancement_to_fiat_rate,
    competitive_mass_fraction,
    parallel_pica_resin,
)
from aether.fiat.pore_pressure import (
    PORE_PRESSURE_REFERENCES,
    PorePressureProfile,
    pore_pressure,
    pore_pressure_sensitivity,
)
from aether.fiat.radiation import (
    gray_radiative_flux,
    optical_depth,
    rosseland_conductivity,
    rosseland_flux,
)
from aether.fiat.solver import (
    FiatSolution,
    FiatSolver,
    FiatStep,
    SolverOptions,
)
from aether.fiat.stack import MaterialStack, Ply, StackGrid
from aether.fiat.surface import (
    AerothermalEnvironment,
    BackfaceCondition,
    BackfaceKind,
    SurfaceState,
    blowing_reduction,
    solve_surface,
)

__all__ = [
    "COMPETITIVE_PICA_BAYESIAN",
    "COMPETITIVE_PICA_DETERMINISTIC",
    "FIBERFORM_SAMPLES",
    "HERITAGE_PICA_CONDUCTIVITY",
    "MARSCHALL_MILOS_SAMPLES",
    "MEDLI2_PICA_CONDUCTIVITY",
    "PARALLEL_PICA_RESIN",
    "PERMEABILITY_UNCERTAINTY",
    "PORE_PRESSURE_REFERENCES",
    "SLIP_PARAMETER_UNCERTAINTY",
    "AerothermalEnvironment",
    "BPrimeTable",
    "BackfaceCondition",
    "BackfaceKind",
    "CompetitivePica",
    "DepthProbe",
    "FiatSolution",
    "FiatSolver",
    "FiatStep",
    "InterfaceHistory",
    "MaterialStack",
    "ParallelReaction",
    "PermeabilitySample",
    "Ply",
    "PorePressureProfile",
    "PressureConductivity",
    "SolverOptions",
    "StackGrid",
    "SurfaceState",
    "TableRangeError",
    "TgaTargets",
    "ThicknessResult",
    "advancement_to_fiat_rate",
    "blowing_reduction",
    "calibrated_components",
    "competitive_mass_fraction",
    "effective_permeability",
    "fiberform_permeability",
    "fit_arrhenius",
    "gray_radiative_flux",
    "interface_histories",
    "knudsen_regime_pressure",
    "optical_depth",
    "optimize_ply_thickness",
    "parallel_pica_resin",
    "peak_rate_temperature",
    "pica_like_material",
    "pore_pressure",
    "pore_pressure_sensitivity",
    "probe_depths",
    "rosseland_conductivity",
    "rosseland_flux",
    "scale_environments",
    "sized_stack",
    "slip_parameter",
    "solve_surface",
    "structural_material",
    "tga_mass_fraction",
]
