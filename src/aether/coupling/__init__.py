"""Structure–fluid coupling: quadrature-consistent slosh force regularization."""

from __future__ import annotations

from aether.coupling.slosh import (
    SloshCoupling,
    kernel_bandwidth,
    local_node_spacing,
    normalized_kernel,
)

__all__ = [
    "SloshCoupling",
    "kernel_bandwidth",
    "local_node_spacing",
    "normalized_kernel",
]
