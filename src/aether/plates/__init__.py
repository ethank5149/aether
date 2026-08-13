"""Mindlin–Reissner anisotropic plate kernel (Paper II, §5).

Three-field :math:`(w, \\phi_x, \\phi_y)` first-order shear-deformation
plates on a fixed bivariate ultraspherical grid: the moment/shear
constitutive relations of Eqs. (5.3)–(5.4), the governing system
Eqs. (5.5)–(5.7), the three *independent* free-edge conditions of
Eq. (5.9), and Kronecker-structured assembly per Eq. (5.19).
"""

from __future__ import annotations

from aether.plates.laminate import OrthotropicLaminate, isotropic_laminate
from aether.plates.mindlin import (
    MindlinPlate,
    PlateModes,
    kirchhoff_free_free_reference,
    simply_supported_exact,
    solve_plate_modes,
)

__all__ = [
    "MindlinPlate",
    "OrthotropicLaminate",
    "PlateModes",
    "isotropic_laminate",
    "kirchhoff_free_free_reference",
    "simply_supported_exact",
    "solve_plate_modes",
]
