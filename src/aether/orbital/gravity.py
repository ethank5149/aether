""":math:`J_2`-perturbed gravity (Paper II, §7.1).

Over a fractional orbit a spherical gravity model accumulates
unacceptable secular error; the dominant correction is the :math:`J_2`
zonal harmonic (Eq. 7.1–7.2). In ECI Cartesian coordinates,

.. math::

    \\mathbf{g}(\\mathbf{r}) = -\\frac{\\mu}{r^3}\\mathbf{r}
      - \\frac{3 J_2 \\mu R_\\oplus^2}{2 r^5}
      \\begin{bmatrix}
        x(1 - 5z^2/r^2) \\\\ y(1 - 5z^2/r^2) \\\\ z(3 - 5z^2/r^2)
      \\end{bmatrix}.

The field is conservative and time-independent, so specific orbital
energy is an exact invariant of the flow — which is what makes the
energy-drift criterion of II-V5 a meaningful integrator diagnostic
rather than a physics question. It is also axisymmetric about the polar
axis, so the polar component of specific angular momentum is a second
invariant; both are provided, because a scheme can conserve one and not
the other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "EARTH",
    "GravityModel",
    "j2_acceleration",
    "specific_angular_momentum_z",
    "specific_energy",
    "two_body_acceleration",
]

_FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GravityModel:
    """Central-body constants (Paper II, §7.1 values)."""

    mu: float
    """Standard gravitational parameter (m³/s²)."""
    radius: float
    """Equatorial reference radius (m)."""
    j2: float
    """Second zonal harmonic coefficient."""

    def __post_init__(self) -> None:
        for name in ("mu", "radius"):
            val = float(getattr(self, name))
            if not (np.isfinite(val) and val > 0.0):
                raise ValueError(f"{name} must be finite and > 0, got {val}")
        if not np.isfinite(self.j2):
            raise ValueError(f"j2 must be finite, got {self.j2}")


#: Earth constants exactly as quoted in Paper II, §7.1.
EARTH = GravityModel(mu=3.986004418e14, radius=6378137.0, j2=1.08263e-3)


def _as_position(position: ArrayLike) -> _FloatArray:
    r = np.asarray(position, dtype=np.float64)
    if r.shape[-1:] != (3,):
        raise ValueError(f"position must have trailing dimension 3, got {r.shape}")
    if not np.all(np.isfinite(r)):
        raise ValueError("position must be finite")
    radius = np.linalg.norm(r, axis=-1)
    if np.any(radius == 0.0):
        raise ValueError("gravity is singular at the origin")
    return r


def two_body_acceleration(position: ArrayLike, model: GravityModel = EARTH) -> _FloatArray:
    """Spherical term :math:`-\\mu\\mathbf{r}/r^3` (m/s²)."""
    r = _as_position(position)
    radius = np.linalg.norm(r, axis=-1, keepdims=True)
    return np.asarray(-model.mu * r / radius**3)


def j2_acceleration(position: ArrayLike, model: GravityModel = EARTH) -> _FloatArray:
    """The :math:`J_2` perturbation of Paper II, Eq. (7.2) (m/s²)."""
    r = _as_position(position)
    radius = np.linalg.norm(r, axis=-1, keepdims=True)
    z_ratio = 5.0 * (r[..., 2:3] / radius) ** 2
    factor = -1.5 * model.j2 * model.mu * model.radius**2 / radius**5
    bracket = np.concatenate(
        [
            r[..., 0:1] * (1.0 - z_ratio),
            r[..., 1:2] * (1.0 - z_ratio),
            r[..., 2:3] * (3.0 - z_ratio),
        ],
        axis=-1,
    )
    return np.asarray(factor * bracket)


def gravitational_acceleration(
    position: ArrayLike, model: GravityModel = EARTH, include_j2: bool = True
) -> _FloatArray:
    """Total gravity (Paper II, Eq. 7.1); ``include_j2 = False`` gives the
    spherical model the paper measures the secular difference against."""
    accel = two_body_acceleration(position, model)
    if include_j2:
        accel = accel + j2_acceleration(position, model)
    return accel


def gravitational_potential(position: ArrayLike, model: GravityModel = EARTH) -> _FloatArray:
    """Specific potential energy :math:`U` (J/kg) whose negative gradient is
    :func:`gravitational_acceleration`.

    .. math::

        U(\\mathbf{r}) = -\\frac{\\mu}{r}
          + \\frac{\\mu J_2 R_\\oplus^2}{2}\\,\\frac{3z^2 - r^2}{r^5}.
    """
    r = _as_position(position)
    radius = np.linalg.norm(r, axis=-1)
    z = r[..., 2]
    keplerian = -model.mu / radius
    oblate = (
        0.5 * model.mu * model.j2 * model.radius**2 * (3.0 * z**2 - radius**2) / radius**5
    )
    return np.asarray(keplerian + oblate)


def specific_energy(
    position: ArrayLike, velocity: ArrayLike, model: GravityModel = EARTH
) -> _FloatArray:
    """Specific orbital energy :math:`v^2/2 + U` (J/kg) — an exact invariant
    of the :math:`J_2` flow, and the II-V5 measurand."""
    v = np.asarray(velocity, dtype=np.float64)
    if v.shape[-1:] != (3,):
        raise ValueError(f"velocity must have trailing dimension 3, got {v.shape}")
    kinetic = 0.5 * np.sum(v * v, axis=-1)
    return np.asarray(kinetic + gravitational_potential(position, model))


def specific_angular_momentum_z(position: ArrayLike, velocity: ArrayLike) -> _FloatArray:
    """Polar component :math:`h_z = x v_y - y v_x` (m²/s).

    Conserved because the :math:`J_2` field is axisymmetric about the
    polar axis — a second, independent invariant, so a scheme that
    conserves energy but leaks :math:`h_z` is still detectably wrong.
    """
    r = _as_position(position)
    v = np.asarray(velocity, dtype=np.float64)
    if v.shape != r.shape:
        raise ValueError(f"velocity shape {v.shape} does not match position {r.shape}")
    return np.asarray(r[..., 0] * v[..., 1] - r[..., 1] * v[..., 0])
