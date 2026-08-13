"""Batched Monte Carlo layer.

The computational claim of the manuscript: because every replicate shares
one fixed-dimension, fixed-sparsity ODE, a Monte Carlo batch is a rank-3
tensor operation (replicate × state × stage) with no per-replicate
remesh. This package provides the array-backend abstraction (NumPy on
CPU, CuPy on CUDA), reproducible per-batch dispersion sampling, the
common-outer-grid batched integrator, and the occupancy model that
relates achieved to theoretical device utilisation.

Terminal *dispersion statistics* — CEP, containment radii, and the
associated normality tests — are deliberately not part of this package.
"""

from __future__ import annotations

from aether.batch.backend import cuda_available, get_array_module, to_numpy
from aether.batch.entry_demo import EntryDispersionModel
from aether.batch.occupancy import (
    AchievedOccupancy,
    OccupancyReport,
    achieved_occupancy,
    device_limits,
    theoretical_occupancy,
)
from aether.batch.propagation import rk4_batch
from aether.batch.sampling import DispersionSpec, sample_dispersions

__all__ = [
    "AchievedOccupancy",
    "DispersionSpec",
    "EntryDispersionModel",
    "OccupancyReport",
    "achieved_occupancy",
    "cuda_available",
    "device_limits",
    "get_array_module",
    "rk4_batch",
    "sample_dispersions",
    "theoretical_occupancy",
    "to_numpy",
]
