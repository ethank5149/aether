"""In-depth radiation transport (Chen & Milos 1999, Eqs. 2 and 3).

Lightweight ablators are semi-transparent: radiation penetrates the char
and moves energy independently of conduction. FIAT offers two treatments
and this module implements both.

**Optically thick limit, Eq. (3).**

.. math::

    q_R = -\\frac{4\\sigma}{3(a + \\sigma_s)}\\frac{\\partial T^4}{\\partial x}
        = -\\frac{16\\sigma T^3}{3K}\\frac{\\partial T}{\\partial x},
    \\qquad K = a + \\sigma_s .

This is a conductivity augmentation, so it folds into the diffusion
operator and is solved *implicitly*. FIAT does the same: "For an
optically thick medium, the radiative flux term of Eq. (3) is used
instead of Eq. (2), and this term is coded implicitly within Eq. (1)."

**Gray medium with isotropic scattering, Eq. (2).** The
integrodifferential form, which FIAT treats as an explicit source
because "a fully implicit treatment of Eq. (2) would require the
inversion of a full matrix."

.. math::

    q_R(\\kappa) = 2\\pi\\Big[
        I^{+}(0)E_3(\\kappa) - I^{-}(\\kappa_D)E_3(\\kappa_D - \\kappa)
        + \\int_0^{\\kappa} I_0 E_2(\\kappa - \\kappa')\\,d\\kappa'
        - \\int_{\\kappa}^{\\kappa_D} I_0 E_2(\\kappa' - \\kappa)\\,d\\kappa'
    \\Big]

with :math:`I_0(\\kappa) = \\sigma T^4/\\pi` at radiative equilibrium.

The integrals here are evaluated **exactly**, not by quadrature. Taking
the source function piecewise constant over each finite-volume cell —
which is the accuracy the surrounding discretisation carries anyway —
and using :math:`dE_3/dt = -E_2`, each cell's contribution collapses to a
difference of :math:`E_3`. That removes quadrature error entirely and,
more importantly, is well behaved as :math:`\\kappa' \\to \\kappa`, where
the :math:`E_2` kernel has an integrable logarithmic singularity that
ordinary quadrature handles badly.
"""

from __future__ import annotations

import numpy as np
import scipy.special
from numpy.typing import ArrayLike, NDArray

from aether.thermal.surface import STEFAN_BOLTZMANN

__all__ = [
    "gray_radiative_flux",
    "optical_depth",
    "rosseland_conductivity",
    "rosseland_flux",
]

_FloatArray = NDArray[np.float64]


def _checked(value: ArrayLike, name: str, *, positive: bool = True) -> _FloatArray:
    arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    if positive and np.any(arr <= 0.0):
        raise ValueError(f"{name} must be > 0")
    return arr


def rosseland_conductivity(
    temperature: ArrayLike, extinction_coefficient: ArrayLike
) -> _FloatArray:
    """Radiative conductivity :math:`16\\sigma T^3/(3K)` (W/(m K)).

    Add this to the solid conductivity to obtain Eq. (3)'s contribution
    to the diffusion operator. Both arguments are strictly positive: a
    zero extinction coefficient is a transparent medium, for which the
    diffusion approximation does not exist and
    :func:`gray_radiative_flux` must be used instead.
    """
    t = _checked(temperature, "temperature")
    k_ext = _checked(extinction_coefficient, "extinction_coefficient")
    return np.asarray(16.0 * STEFAN_BOLTZMANN * t**3 / (3.0 * k_ext))


def rosseland_flux(
    temperature: ArrayLike,
    extinction_coefficient: ArrayLike,
    temperature_gradient: ArrayLike,
) -> _FloatArray:
    """Eq. (3) evaluated directly, W/m². Positive means flux toward +x."""
    k_rad = rosseland_conductivity(temperature, extinction_coefficient)
    grad = np.asarray(temperature_gradient, dtype=np.float64)
    if not np.all(np.isfinite(grad)):
        raise ValueError("temperature_gradient must be finite")
    return np.asarray(-k_rad * grad)


def optical_depth(
    cell_widths: ArrayLike, extinction_coefficient: ArrayLike
) -> _FloatArray:
    """Cumulative optical depth at cell faces, length ``n_cells + 1``.

    :math:`\\kappa(x) = \\int_0^x K\\,dx'`, starting at zero on the
    heated face.
    """
    widths = _checked(cell_widths, "cell_widths")
    k_ext = _checked(extinction_coefficient, "extinction_coefficient")
    if widths.ndim != 1:
        raise ValueError("cell_widths must be one-dimensional")
    k_ext = np.broadcast_to(k_ext, widths.shape)
    return np.concatenate([[0.0], np.cumsum(widths * k_ext)])


def gray_radiative_flux(
    face_optical_depth: ArrayLike,
    cell_temperature: ArrayLike,
    front_intensity: float = 0.0,
    back_intensity: float = 0.0,
) -> _FloatArray:
    """Eq. (2) at every cell face, W/m². Positive means flux toward +x.

    Parameters
    ----------
    face_optical_depth:
        :math:`\\kappa` at each face, from :func:`optical_depth`;
        non-decreasing, length ``n_cells + 1``.
    cell_temperature:
        Cell temperatures (K), length ``n_cells``. The source function
        is :math:`I_0 = \\sigma T^4/\\pi`, piecewise constant per cell.
    front_intensity:
        :math:`I^{+}(0)`, the intensity entering the heated face and
        travelling inward (W/(m² sr)). For an opaque emitting wall at
        :math:`T_w` this is :math:`\\varepsilon_w \\sigma T_w^4/\\pi`.
    back_intensity:
        :math:`I^{-}(\\kappa_D)`, entering the backface and travelling
        outward.

    Notes
    -----
    Contributions are accumulated as an explicit double sum over cells.
    That is :math:`O(n^2)`, which is the honest cost of the kernel —
    every cell radiates to every other — and is why FIAT declines to
    treat this term implicitly. At the few hundred cells a TPS stack
    needs, the cost is negligible against the Newton solve.
    """
    kappa = np.asarray(face_optical_depth, dtype=np.float64)
    t_cell = _checked(cell_temperature, "cell_temperature")
    if kappa.ndim != 1 or t_cell.ndim != 1 or kappa.size != t_cell.size + 1:
        raise ValueError(
            f"need face_optical_depth of length n_cells + 1; got "
            f"{kappa.size} faces for {t_cell.size} cells"
        )
    if not np.all(np.isfinite(kappa)) or np.any(np.diff(kappa) < 0.0):
        raise ValueError("face_optical_depth must be finite and non-decreasing")
    for name, value in (("front_intensity", front_intensity), ("back_intensity", back_intensity)):
        if not (np.isfinite(value) and value >= 0.0):
            raise ValueError(f"{name} must be finite and >= 0, got {value}")

    source = STEFAN_BOLTZMANN * t_cell**4 / np.pi
    kappa_d = float(kappa[-1])

    # E_3 of every (face, face) optical-depth separation, computed once.
    # separation[f, i] = kappa[f] - kappa[i], signed.
    separation = kappa[:, None] - kappa[None, :]
    e3 = scipy.special.expn(3, np.abs(separation))

    n_faces = kappa.size
    flux = np.empty(n_faces, dtype=np.float64)
    for f in range(n_faces):
        # Cells strictly above this face: integral_0^kappa I_0 E_2(kappa - kappa').
        # Per cell i, that is E_3(kappa_f - kappa_{i+1}) - E_3(kappa_f - kappa_i).
        above = source[:f] @ (e3[f, 1 : f + 1] - e3[f, :f]) if f > 0 else 0.0
        # Cells below: integral_kappa^kappa_D, entering with the opposite sign.
        # Per cell i, E_3(kappa_i - kappa_f) - E_3(kappa_{i+1} - kappa_f).
        below = source[f:] @ (e3[f, f:-1] - e3[f, f + 1 :]) if f < n_faces - 1 else 0.0
        boundary = front_intensity * scipy.special.expn(3, kappa[f]) - (
            back_intensity * scipy.special.expn(3, kappa_d - kappa[f])
        )
        flux[f] = 2.0 * np.pi * (boundary + above - below)
    return flux
