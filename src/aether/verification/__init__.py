"""Executable verification tasks for the numerics kernel.

Each module runs one verification task against an acceptance criterion
stated before any results existed, and writes a markdown report plus
machine-readable CSV into a results directory:

- :mod:`aether.verification.v1_structural` — V1: conditioning of the
  reduced stiffness operator versus :math:`N`; free-free frequencies
  against the analytic uniform-beam solution.
- :mod:`aether.verification.v2_slosh` — V2: exact force transfer under the
  collocation quadrature rule.
- :mod:`aether.verification.v3_integrators` — V3: achieved step size and
  wall clock for explicit, modally truncated, and IMEX strategies.
- :mod:`aether.verification.v4_thermal` — V4: manufactured-solution
  convergence for the charring/Stefan system on the immobilized grid,
  using :mod:`aether.verification.mms_ablation`.
- :mod:`aether.verification.v8_throughput` — V8: batched Monte Carlo
  throughput and occupancy against replicate count.
- :mod:`aether.verification.p2v1_ultraspherical` — conditioning of the
  ultraspherical formulation versus collocation.
- :mod:`aether.verification.p2v123_plates` — Mindlin plate and laminate
  benchmarks.

Run them all with ``python -m aether.verification``.
"""

from __future__ import annotations

from aether.verification.common import VerificationReport

__all__ = ["VerificationReport"]
