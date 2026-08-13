"""Attitude kinematics with Baumgarte stabilization (Paper II, §3.1).

The quaternion evolves by :math:`\\dot{\\mathbf{q}} = \\tfrac{1}{2}
\\bm{\\Omega}(\\bm{\\omega})\\mathbf{q}` (Eq. 3.1). The unit-norm
constraint is an invariant of the exact flow but is *not* preserved by a
general Runge–Kutta method, which drifts off the constraint manifold at
the order of the local truncation error and accumulates over a long
trajectory. Rather than renormalizing after each step — cheap, but it
silently perturbs the embedded error estimate the adaptive controller
relies on — the correction is applied inside the right-hand side
(Paper I, Eq. 3.30):

.. math::

    \\dot{\\mathbf{q}} = \\tfrac{1}{2}\\bm{\\Omega}(\\bm{\\omega})\\mathbf{q}
    + k_q\\left(1 - \\|\\mathbf{q}\\|^2\\right)\\mathbf{q},

which renders the unit sphere *attracting* rather than merely invariant
and keeps the correction visible to the error controller.

**Convention.** Scalar part first, :math:`\\mathbf{q} = [q_0,
\\bm{q}_{1:3}]`, Hamilton (not JPL) product, mapping ECI to body axes
(Paper II, Appendix B). Mixing Hamilton and JPL produces a transpose
error that is easy to introduce and hard to detect, since both yield
valid rotation matrices — which is why the convention is asserted in
the tests rather than left to a comment.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["dcm_from_quaternion", "quaternion_derivative", "quaternion_norm_error"]

_FloatArray = NDArray[np.float64]


def _as_quaternion(q: ArrayLike) -> _FloatArray:
    arr = np.asarray(q, dtype=np.float64)
    if arr.shape[-1:] != (4,):
        raise ValueError(f"quaternion must have trailing dimension 4, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("quaternion must be finite")
    return arr


def quaternion_derivative(
    quaternion: ArrayLike,
    angular_rate: ArrayLike,
    baumgarte_gain: float = 1.0,
) -> _FloatArray:
    """:math:`\\dot{\\mathbf{q}}` with Baumgarte norm stabilization.

    Parameters
    ----------
    quaternion:
        :math:`[q_0, q_1, q_2, q_3]`, scalar first, shape ``(..., 4)``.
    angular_rate:
        Body angular velocity :math:`\\bm{\\omega}_B` (rad/s), shape
        ``(..., 3)``.
    baumgarte_gain:
        :math:`k_q > 0`, chosen so :math:`1/k_q` is short relative to
        the trajectory but long relative to the step size. Zero
        recovers the unstabilized kinematics exactly.
    """
    if not (np.isfinite(baumgarte_gain) and baumgarte_gain >= 0.0):
        raise ValueError(f"baumgarte_gain must be finite and >= 0, got {baumgarte_gain}")
    q = _as_quaternion(quaternion)
    w = np.asarray(angular_rate, dtype=np.float64)
    if w.shape[-1:] != (3,):
        raise ValueError(f"angular_rate must have trailing dimension 3, got {w.shape}")

    q0 = q[..., 0]
    qv = q[..., 1:]
    # Omega(w) q with Omega = [[0, -w^T], [w, -[w x]]]
    d0 = -np.sum(qv * w, axis=-1)
    dv = q0[..., np.newaxis] * w - np.cross(w, qv)
    dq = 0.5 * np.concatenate([d0[..., np.newaxis], dv], axis=-1)

    if baumgarte_gain > 0.0:
        norm_sq = np.sum(q * q, axis=-1, keepdims=True)
        dq = dq + baumgarte_gain * (1.0 - norm_sq) * q
    return np.asarray(dq)


def quaternion_norm_error(quaternion: ArrayLike) -> _FloatArray:
    """:math:`\\|\\mathbf{q}\\| - 1`, the constraint residual."""
    q = _as_quaternion(quaternion)
    return np.asarray(np.linalg.norm(q, axis=-1) - 1.0)


def dcm_from_quaternion(quaternion: ArrayLike) -> _FloatArray:
    """Direction cosine matrix :math:`\\mathbf{C}_E^B` (Paper II, Eq. B.1).

    .. math::

        \\mathbf{C}_E^B = (q_0^2 - \\bm{q}^\\top\\bm{q})\\mathbf{I}
        + 2\\bm{q}\\bm{q}^\\top - 2q_0[\\bm{q}\\times],

    mapping ECI components to body components. Returns shape
    ``(..., 3, 3)``.
    """
    q = _as_quaternion(quaternion)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm == 0.0):
        raise ValueError("quaternion must be non-zero to define a rotation")
    q = q / norm
    q0 = q[..., 0]
    qv = q[..., 1:]
    eye = np.eye(3)
    outer = qv[..., :, np.newaxis] * qv[..., np.newaxis, :]
    skew = np.zeros((*qv.shape[:-1], 3, 3))
    skew[..., 0, 1] = -qv[..., 2]
    skew[..., 0, 2] = qv[..., 1]
    skew[..., 1, 0] = qv[..., 2]
    skew[..., 1, 2] = -qv[..., 0]
    skew[..., 2, 0] = -qv[..., 1]
    skew[..., 2, 1] = qv[..., 0]
    scalar = q0**2 - np.sum(qv * qv, axis=-1)
    return np.asarray(
        scalar[..., np.newaxis, np.newaxis] * eye
        + 2.0 * outer
        - 2.0 * q0[..., np.newaxis, np.newaxis] * skew
    )
