"""AETHER — Aero-thermo-Elastic Trajectory & Hypersonic Estimation Research.

The public numerics kernel: a fixed-grid spectral formulation for coupled
moving-boundary problems, with batched uncertainty quantification. This is
the implementation accompanying Knox (2026), *A Fixed-Grid Spectral Method
for Coupled Moving-Boundary Problems, with Batched Uncertainty
Quantification* (``manuscript/``).

- :mod:`aether.spectral` — Chebyshev–Gauss–Lobatto collocation operators
  built by the direct recurrence with the negative-sum trick,
  Clenshaw–Curtis quadrature, and barycentric interpolation.
- :mod:`aether.ultraspherical` — the well-conditioned Olver–Townsend
  ultraspherical formulation and its banded operator assembly.
- :mod:`aether.thermal` — the Landau boundary-immobilization frame and the
  semi-discrete charring/Stefan solver on the resulting fixed grid.
- :mod:`aether.structures`, :mod:`aether.plates` — the variable-rigidity
  Euler–Bernoulli operator with free-free boundary conditions by null-space
  projection, Mindlin plates, and laminate stiffness.
- :mod:`aether.coupling` — quadrature-normalized kernel force transfer.
- :mod:`aether.batch` — the array-backend abstraction and the batched
  common-outer-grid integrator underlying the UQ speedup result.
- :mod:`aether.atmosphere`, :mod:`aether.dynamics` — standard published
  atmosphere models and rigid-body attitude/incidence kinematics.
- :mod:`aether.verification` — the executable verification tasks.

Scope note
----------
This package is deliberately limited to numerical-methods and
standard-model content. Applied flight-systems capability — guidance,
trajectory optimization, sensing, tracking, and engagement modelling —
is maintained separately and is not distributed with this package.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
