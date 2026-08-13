"""Orthotropic laminate rigidities for the Mindlin–Reissner kernel.

Paper II, Eqs. (5.3)–(5.4): the moment resultants follow from the
flexural rigidity tensor :math:`\\mathbf{D}_{ij}` and the transverse
shears from :math:`\\kappa_s^2 G_{xz} h`, :math:`\\kappa_s^2 G_{yz} h`,
with the shear correction factor :math:`\\kappa_s^2 = 5/6` following
Reissner. Mindlin's alternative :math:`\\pi^2/12` differs by under 2%
and is available as a named constant rather than a magic number, since
the paper states both.

Rigidities are held constant over the planform in this implementation.
Paper II constructs :math:`\\mathbf{D}_{ij}(\\xi, \\eta)` by hyperbolic
blending across material interfaces; the Kronecker assembly in
:mod:`aether.plates.mindlin` multiplies each term by a scalar
coefficient, and extending it to separable (rank-1) coefficient fields
means substituting a per-direction multiplication operator for the
scalar — the structure is in place, the variable-coefficient path is
not yet exercised and is stated as such rather than implied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SHEAR_CORRECTION_MINDLIN",
    "SHEAR_CORRECTION_REISSNER",
    "OrthotropicLaminate",
    "isotropic_laminate",
]

#: Reissner's shear correction factor (Paper II, §5.2 default).
SHEAR_CORRECTION_REISSNER = 5.0 / 6.0
#: Mindlin's alternative; differs from Reissner's by under 2%.
SHEAR_CORRECTION_MINDLIN = np.pi**2 / 12.0


@dataclass(frozen=True)
class OrthotropicLaminate:
    """Flexural and transverse-shear rigidities of a plate section.

    Attributes
    ----------
    d11, d12, d22, d66:
        Flexural rigidity components (N·m) of Paper II, Eq. (5.3).
    shear_xz, shear_yz:
        Transverse shear moduli :math:`G_{xz}`, :math:`G_{yz}` (Pa).
    thickness:
        Section thickness :math:`h` (m).
    density:
        Material density :math:`\\rho` (kg/m³).
    shear_correction:
        :math:`\\kappa_s^2`; defaults to Reissner's 5/6.
    """

    d11: float
    d12: float
    d22: float
    d66: float
    shear_xz: float
    shear_yz: float
    thickness: float
    density: float
    shear_correction: float = SHEAR_CORRECTION_REISSNER

    def __post_init__(self) -> None:
        for name in ("d11", "d22", "d66", "shear_xz", "shear_yz", "thickness", "density"):
            val = float(getattr(self, name))
            if not (np.isfinite(val) and val > 0.0):
                raise ValueError(f"{name} must be finite and > 0, got {val}")
        if not np.isfinite(self.d12):
            raise ValueError(f"d12 must be finite, got {self.d12}")
        # positive-definiteness of the bending block [[d11, d12], [d12, d22]]
        if self.d11 * self.d22 <= self.d12**2:
            raise ValueError(
                f"bending rigidity block is not positive definite: "
                f"d11*d22 = {self.d11 * self.d22:.6g} must exceed d12² = {self.d12**2:.6g}"
            )
        if not 0.0 < self.shear_correction <= 1.0:
            raise ValueError(
                f"shear_correction must be in (0, 1], got {self.shear_correction}"
            )

    @property
    def shear_stiffness_x(self) -> float:
        """:math:`\\kappa_s^2 G_{xz} h` (N/m)."""
        return self.shear_correction * self.shear_xz * self.thickness

    @property
    def shear_stiffness_y(self) -> float:
        """:math:`\\kappa_s^2 G_{yz} h` (N/m)."""
        return self.shear_correction * self.shear_yz * self.thickness

    @property
    def mass_per_area(self) -> float:
        """:math:`\\rho h` (kg/m²), the transverse inertia."""
        return self.density * self.thickness

    @property
    def rotary_inertia(self) -> float:
        """:math:`\\rho h^3/12` (kg), the rotary inertia of Eqs. (5.6)–(5.7)."""
        return self.density * self.thickness**3 / 12.0

    def with_thickness(self, thickness: float) -> OrthotropicLaminate:
        """Same material at a different thickness.

        Flexural rigidities scale as :math:`h^3` and shear stiffness as
        :math:`h`, which is precisely the :math:`h^{-1}`-growing
        stiffness ratio that drives shear locking (Paper II, Remark 3) —
        so this is the constructor the II-V2 thickness sweep uses.
        """
        ratio = (float(thickness) / self.thickness) ** 3
        return OrthotropicLaminate(
            d11=self.d11 * ratio,
            d12=self.d12 * ratio,
            d22=self.d22 * ratio,
            d66=self.d66 * ratio,
            shear_xz=self.shear_xz,
            shear_yz=self.shear_yz,
            thickness=float(thickness),
            density=self.density,
            shear_correction=self.shear_correction,
        )


def isotropic_laminate(
    youngs_modulus: float,
    poisson_ratio: float,
    thickness: float,
    density: float,
    shear_correction: float = SHEAR_CORRECTION_REISSNER,
) -> OrthotropicLaminate:
    """Isotropic plate: the II-V2/II-V3 reference configuration.

    :math:`D = Eh^3/(12(1-\\nu^2))`, with
    :math:`D_{11} = D_{22} = D`, :math:`D_{12} = \\nu D`,
    :math:`D_{66} = (1-\\nu)D/2`, and :math:`G = E/(2(1+\\nu))`.
    """
    e = float(youngs_modulus)
    nu = float(poisson_ratio)
    h = float(thickness)
    if not (np.isfinite(e) and e > 0.0):
        raise ValueError(f"youngs_modulus must be finite and > 0, got {e}")
    if not -1.0 < nu < 0.5:
        raise ValueError(f"poisson_ratio must be in (-1, 0.5), got {nu}")
    flexural = e * h**3 / (12.0 * (1.0 - nu * nu))
    shear_modulus = e / (2.0 * (1.0 + nu))
    return OrthotropicLaminate(
        d11=flexural,
        d12=nu * flexural,
        d22=flexural,
        d66=(1.0 - nu) * flexural / 2.0,
        shear_xz=shear_modulus,
        shear_yz=shear_modulus,
        thickness=h,
        density=float(density),
        shear_correction=shear_correction,
    )
