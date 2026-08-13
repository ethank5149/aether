"""Reproducible dispersion sampling for Monte Carlo batches.

Sampling uses the counter-based Philox generator: given ``(seed,
n_replicates)`` the draw is bit-reproducible across runs and platforms,
which is the reproducibility posture of Paper I, §5.1 applied to the
statistical layer. Parameters are sampled *jointly* in a fixed
specification order, so adding a parameter to the end of a spec does not
perturb the draws of the parameters before it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["DispersionSpec", "sample_dispersions"]

_FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DispersionSpec:
    """One dispersed scalar parameter.

    Attributes
    ----------
    name:
        Key under which samples are returned.
    nominal:
        Mean value.
    sigma:
        Standard deviation of the Gaussian dispersion; zero pins the
        parameter at nominal (deterministic).
    lower, upper:
        Optional physical truncation bounds, applied by resampling
        (rejection), not clipping — clipping piles probability mass on
        the bound and distorts the tails that dispersion analysis
        exists to resolve.
    """

    name: str
    nominal: float
    sigma: float
    lower: float = -np.inf
    upper: float = np.inf

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("dispersion name must be non-empty")
        if not (np.isfinite(self.nominal)):
            raise ValueError(f"nominal must be finite, got {self.nominal}")
        if not (np.isfinite(self.sigma) and self.sigma >= 0.0):
            raise ValueError(f"sigma must be finite and >= 0, got {self.sigma}")
        if not self.lower < self.upper:
            raise ValueError(f"need lower < upper, got ({self.lower}, {self.upper})")
        if not self.lower <= self.nominal <= self.upper:
            raise ValueError(
                f"nominal {self.nominal} outside bounds ({self.lower}, {self.upper})"
            )


def sample_dispersions(
    specs: Sequence[DispersionSpec],
    n_replicates: int,
    seed: int,
) -> dict[str, _FloatArray]:
    """Draw all dispersed parameters for a batch.

    Returns
    -------
    dict
        ``name -> samples`` with each array of shape ``(n_replicates,)``.
    """
    if n_replicates < 1:
        raise ValueError(f"n_replicates must be >= 1, got {n_replicates}")
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate dispersion names in spec: {names}")
    rng = np.random.Generator(np.random.Philox(seed))
    out: dict[str, _FloatArray] = {}
    for spec in specs:
        if spec.sigma == 0.0:
            out[spec.name] = np.full(n_replicates, spec.nominal)
            continue
        draws = rng.normal(spec.nominal, spec.sigma, n_replicates)
        bad = (draws < spec.lower) | (draws > spec.upper)
        guard = 0
        while np.any(bad):
            draws[bad] = rng.normal(spec.nominal, spec.sigma, int(bad.sum()))
            bad = (draws < spec.lower) | (draws > spec.upper)
            guard += 1
            if guard > 1000:
                raise RuntimeError(
                    f"rejection sampling for '{spec.name}' failed to terminate; "
                    f"bounds ({spec.lower}, {spec.upper}) are too tight for "
                    f"N({spec.nominal}, {spec.sigma}²)"
                )
        out[spec.name] = draws
    return out
