"""Rigid-body attitude kinematics and deformed-surface incidence.

Quaternion kinematics with norm-error diagnostics, the direction cosine
matrix, and the local incidence angle on a deformed surface — the
kinematic layer the structural and thermal models need in order to be
driven by an attitude history.
"""

from __future__ import annotations

from aether.dynamics.attitude import (
    dcm_from_quaternion,
    quaternion_derivative,
    quaternion_norm_error,
)
from aether.dynamics.incidence import deformed_normal, local_incidence

__all__ = [
    "dcm_from_quaternion",
    "deformed_normal",
    "local_incidence",
    "quaternion_derivative",
    "quaternion_norm_error",
]
