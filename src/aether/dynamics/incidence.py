"""Local incidence on the deformed surface (Paper II, §3.2).

Because a lifting body has significant planform curvature and deforms
aeroelastically in flight, a single vehicle angle of attack does not
determine the local flow condition at a given surface station. The
incidence is defined pointwise from the deformed outward normal and the
unit body-frame velocity (Eq. 3.2):

.. math::

    \\sin\\delta_c(\\xi, \\eta, t) = -\\,\\mathbf{n}(\\xi,\\eta,t)
    \\cdot \\hat{\\mathbf{v}}_B .

**The negative sign is required and is easy to lose.** With
:math:`\\mathbf{n}` the *outward* normal, a windward panel — one facing
into the flow — has :math:`\\mathbf{n}\\cdot\\hat{\\mathbf{v}}_B < 0`.
Dropping the negation makes windward panels register as leeward,
silently inverting the entire pressure and heating distribution. The
convention here is :math:`\\delta_c > 0` windward, :math:`\\delta_c \\le
0` leeward, and the sign is asserted in the tests rather than trusted.

The normal is computed from the deformed *outer mold line*, which for
Mindlin kinematics sits a half-thickness along the rotated
cross-section normal:

.. math::

    \\mathbf{r}(x,y) = \\big(x - \\tfrac{h}{2}\\phi_x,\\;
    y - \\tfrac{h}{2}\\phi_y,\\; w + \\tfrac{h}{2}\\big),

so :math:`\\mathbf{n}` depends on all three structural fields and their
first derivatives — this is the coupling that makes the aerodynamic load
a function of the deformation rather than of rigid-body attitude alone.
With ``offset = 0`` the expression collapses to the midsurface normal
:math:`\\propto (-w_{,x}, -w_{,y}, 1)`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["deformed_normal", "local_incidence"]

_FloatArray = NDArray[np.float64]


def deformed_normal(
    w_x: ArrayLike,
    w_y: ArrayLike,
    phi_x: ArrayLike | None = None,
    phi_y: ArrayLike | None = None,
    phi_x_x: ArrayLike | None = None,
    phi_x_y: ArrayLike | None = None,
    phi_y_x: ArrayLike | None = None,
    phi_y_y: ArrayLike | None = None,
    offset: float = 0.0,
) -> _FloatArray:
    """Unit outward normal of the deformed outer mold line.

    Parameters
    ----------
    w_x, w_y:
        Midsurface slopes :math:`\\partial w/\\partial x`,
        :math:`\\partial w/\\partial y`.
    phi_x, phi_y:
        Cross-section rotations; required when ``offset`` is non-zero.
    phi_x_x, phi_x_y, phi_y_x, phi_y_y:
        First derivatives of the rotations; required when ``offset`` is
        non-zero.
    offset:
        Half-thickness :math:`h/2` of the surface above the midsurface.
        Zero gives the midsurface normal.

    Returns
    -------
    numpy.ndarray
        Outward unit normals, shape ``(..., 3)``, oriented to the
        :math:`+z` side.
    """
    wx = np.asarray(w_x, dtype=np.float64)
    wy = np.asarray(w_y, dtype=np.float64)
    if wx.shape != wy.shape:
        raise ValueError(f"w_x and w_y must share shape, got {wx.shape} and {wy.shape}")
    off = float(offset)
    if not np.isfinite(off):
        raise ValueError(f"offset must be finite, got {offset}")

    if off == 0.0:
        tangent_x = np.stack([np.ones_like(wx), np.zeros_like(wx), wx], axis=-1)
        tangent_y = np.stack([np.zeros_like(wy), np.ones_like(wy), wy], axis=-1)
    else:
        required = (phi_x_x, phi_x_y, phi_y_x, phi_y_y)
        if any(v is None for v in required):
            raise ValueError(
                "a non-zero offset needs all four rotation derivatives "
                "(phi_x_x, phi_x_y, phi_y_x, phi_y_y)"
            )
        pxx = np.asarray(phi_x_x, dtype=np.float64)
        pxy = np.asarray(phi_x_y, dtype=np.float64)
        pyx = np.asarray(phi_y_x, dtype=np.float64)
        pyy = np.asarray(phi_y_y, dtype=np.float64)
        tangent_x = np.stack([1.0 - off * pxx, -off * pyx, wx], axis=-1)
        tangent_y = np.stack([-off * pxy, 1.0 - off * pyy, wy], axis=-1)

    normal = np.cross(tangent_x, tangent_y)
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    if np.any(norm == 0.0):
        raise ValueError(
            "degenerate surface: tangent vectors are parallel, so the normal "
            "is undefined"
        )
    normal = normal / norm
    # orient outward (+z side); the cross product above already does so for
    # small deformations, but a large rotation could flip it
    flip = np.where(normal[..., 2:3] < 0.0, -1.0, 1.0)
    return np.asarray(normal * flip)


def local_incidence(normal: ArrayLike, velocity_body: ArrayLike) -> _FloatArray:
    """Local incidence :math:`\\delta_c` (rad) from Paper II, Eq. (3.2).

    Positive windward, non-positive leeward. ``velocity_body`` need not
    be normalized.
    """
    n = np.asarray(normal, dtype=np.float64)
    v = np.asarray(velocity_body, dtype=np.float64)
    if n.shape[-1:] != (3,) or v.shape[-1:] != (3,):
        raise ValueError(
            f"normal and velocity_body need trailing dimension 3, got "
            f"{n.shape} and {v.shape}"
        )
    speed = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(speed == 0.0):
        raise ValueError("velocity_body must be non-zero to define an incidence")
    v_hat = v / speed
    sin_delta = -np.sum(n * v_hat, axis=-1)
    return np.asarray(np.arcsin(np.clip(sin_delta, -1.0, 1.0)))
