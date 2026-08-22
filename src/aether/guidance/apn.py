"""Aerodynamically-Compensated Augmented Proportional Navigation.

Paper I, Eq. (4.18): the commanded lateral acceleration is

.. math::

    \\mathbf{a}_n = N'\\,(\\dot{\\bm{\\lambda}} \\times \\mathbf{v}_{\\mathrm{rel}})
        + \\tfrac{N'}{2}\\,\\mathbf{a}_T - \\mathbf{g}_\\perp,

with line-of-sight rate :math:`\\dot{\\bm{\\lambda}} = (\\mathbf{r}
\\times \\mathbf{v}_{\\mathrm{rel}}) / \\|\\mathbf{r}\\|^2`, estimated
target acceleration :math:`\\mathbf{a}_T`, and :math:`\\mathbf{g}_\\perp`
the component of gravity normal to the relative velocity. The gravity
term is feed-forward, not corrective: omitting it shows up as a
systematic downrange offset in the dispersion footprint (Paper I, §4.3).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["apn_acceleration", "los_rate"]

_FloatArray = NDArray[np.floating]


def _as_vec3(x: ArrayLike, name: str) -> _FloatArray:
    arr = np.asarray(x)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)
    if arr.shape[-1:] != (3,):
        raise ValueError(f"{name} must have trailing dimension 3, got shape {arr.shape}")
    return arr


def los_rate(r_los: ArrayLike, v_rel: ArrayLike) -> _FloatArray:
    """Line-of-sight angular rate :math:`\\dot{\\bm{\\lambda}}` (rad/s).

    Broadcasts over leading (batch) axes. Raises on zero range — the
    LOS direction is undefined there and guidance upstream must have
    already terminated on ``INTERCEPT_NOW``.
    """
    r = _as_vec3(r_los, "r_los")
    v = _as_vec3(v_rel, "v_rel")
    r_sq = np.sum(r * r, axis=-1, keepdims=True)
    if np.any(r_sq == 0.0):
        raise ValueError(
            "los_rate is undefined at zero range; terminal logic must handle "
            "INTERCEPT_NOW before requesting a LOS rate"
        )
    return np.cross(r, v) / r_sq


def apn_acceleration(
    r_los: ArrayLike,
    v_rel: ArrayLike,
    target_accel: ArrayLike,
    gravity: ArrayLike,
    nav_gain: float = 3.0,
) -> _FloatArray:
    """AC-APN commanded lateral acceleration (Paper I, Eq. 4.18).

    Parameters
    ----------
    r_los:
        Relative position, target minus vehicle, shape ``(..., 3)`` (m).
    v_rel:
        Relative velocity, shape ``(..., 3)`` (m/s).
    target_accel:
        Estimated target acceleration :math:`\\mathbf{a}_T` (m/s²).
    gravity:
        Local gravity vector :math:`\\mathbf{g}` (m/s²); the component
        normal to :math:`\\mathbf{v}_{\\mathrm{rel}}` is fed forward.
    nav_gain:
        Navigation constant :math:`N'`, conventionally 3–5.

    Returns
    -------
    numpy.ndarray
        Commanded acceleration, shape ``(..., 3)``. The PN term is
        normal to :math:`\\mathbf{v}_{\\mathrm{rel}}` by construction of
        the cross product; the target-acceleration augmentation is
        applied unprojected, as written in the paper.
    """
    if not (np.isfinite(nav_gain) and nav_gain > 0.0):
        raise ValueError(f"nav_gain must be finite and > 0, got {nav_gain}")
    r = _as_vec3(r_los, "r_los")
    v = _as_vec3(v_rel, "v_rel")
    a_t = _as_vec3(target_accel, "target_accel")
    g = _as_vec3(gravity, "gravity")

    v_sq = np.sum(v * v, axis=-1, keepdims=True)
    if np.any(v_sq == 0.0):
        raise ValueError(
            "apn_acceleration is undefined at zero relative velocity: the "
            "normal-to-velocity decomposition of gravity has no meaning there"
        )
    lam_dot = los_rate(r, v)
    g_parallel = (np.sum(g * v, axis=-1, keepdims=True) / v_sq) * v
    g_perp = g - g_parallel
    return nav_gain * np.cross(lam_dot, v) + (nav_gain / 2.0) * a_t - g_perp
