"""Adaptive state estimation: χ² anomaly gating and IAE (Paper I, §4.1–4.2)."""

from __future__ import annotations

from aether.estimation.adaptive import (
    AdaptiveConfig,
    AdaptiveKalmanFilter,
    FilterStepDiagnostics,
    LinearModel,
    chi_square_gate,
    inflation_factor,
)

__all__ = [
    "AdaptiveConfig",
    "AdaptiveKalmanFilter",
    "FilterStepDiagnostics",
    "LinearModel",
    "chi_square_gate",
    "inflation_factor",
]
