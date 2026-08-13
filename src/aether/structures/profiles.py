"""Material property fields for the variable-rigidity beam.

Assumption 1 of Paper I requires :math:`EI(x) \\in C^2([0, L])` with
:math:`EI(x) > 0` — the regularity that licenses the product-rule
expansion of the stiffness operator. Stepped configurations (stage
joints) are therefore constructed by *hyperbolic blending* between
segment values rather than by piecewise assignment, exactly as the paper
prescribes: a :math:`\\tanh` transition of finite width is :math:`C^\\infty`,
so no spectral accuracy is forfeited to a jump discontinuity.

Analytic first and second derivatives are carried alongside each field so
tests can distinguish spectral-differentiation error from model error;
the operator assembly itself uses spectral derivatives of the sampled
field, as written in Paper I, Eq. (3.5).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["MaterialProfile", "stepped_profile", "uniform_profile"]

_FloatArray = NDArray[np.float64]
_Field = Callable[[_FloatArray], _FloatArray]


@dataclass(frozen=True)
class MaterialProfile:
    """Flexural rigidity and mass-per-length fields on :math:`[0, L]`.

    Attributes
    ----------
    ei:
        Flexural rigidity :math:`EI(x)` in N·m²; must be positive.
    mass:
        Mass per unit length :math:`m(x)` in kg/m; must be positive.
    d_ei:
        Analytic :math:`dEI/dx`, available for every profile constructed
        by this module.
    d2_ei:
        Analytic :math:`d^2 EI/dx^2`.
    label:
        Short name used in verification reports.
    """

    ei: _Field
    mass: _Field
    d_ei: _Field
    d2_ei: _Field
    label: str

    def validate_on(self, x: ArrayLike) -> None:
        """Check positivity of both fields at the given stations."""
        x_arr = np.asarray(x, dtype=np.float64)
        ei = self.ei(x_arr)
        m = self.mass(x_arr)
        if not np.all(np.isfinite(ei)) or np.any(ei <= 0.0):
            raise ValueError(f"profile '{self.label}': EI(x) must be finite and > 0 everywhere")
        if not np.all(np.isfinite(m)) or np.any(m <= 0.0):
            raise ValueError(f"profile '{self.label}': m(x) must be finite and > 0 everywhere")


def uniform_profile(ei: float, mass: float, label: str = "uniform") -> MaterialProfile:
    """Constant :math:`EI` and :math:`m` — the V1 analytic reference case."""
    ei_val = float(ei)
    m_val = float(mass)
    if not (np.isfinite(ei_val) and ei_val > 0.0):
        raise ValueError(f"EI must be finite and > 0, got {ei}")
    if not (np.isfinite(m_val) and m_val > 0.0):
        raise ValueError(f"mass must be finite and > 0, got {mass}")
    return MaterialProfile(
        ei=lambda x: np.full_like(x, ei_val),
        mass=lambda x: np.full_like(x, m_val),
        d_ei=lambda x: np.zeros_like(x),
        d2_ei=lambda x: np.zeros_like(x),
        label=label,
    )


def stepped_profile(
    segment_ei: Sequence[float],
    segment_mass: Sequence[float],
    joints: Sequence[float],
    blend_width: float,
    label: str = "stepped",
) -> MaterialProfile:
    """Multi-segment profile with :math:`C^\\infty` hyperbolic blending.

    The fields are

    .. math::

        EI(x) = EI_1 + \\sum_{k} \\frac{EI_{k+1} - EI_k}{2}
                \\left[1 + \\tanh\\frac{x - x_k}{\\delta}\\right],

    and likewise for :math:`m(x)`: far from every joint the field sits on
    the segment value, and each joint is a smooth step of width
    :math:`\\mathcal{O}(\\delta)`.

    Parameters
    ----------
    segment_ei:
        Rigidity of each segment, length ``len(joints) + 1``, ordered by
        increasing :math:`x`.
    segment_mass:
        Mass per length of each segment, same length as ``segment_ei``.
    joints:
        Joint stations :math:`x_k`, strictly increasing.
    blend_width:
        Transition scale :math:`\\delta > 0`. Resolving the blend
        requires the local Chebyshev spacing to be below
        :math:`\\sim \\delta`; an under-resolved blend degrades spectral
        convergence exactly as an unresolved boundary layer would.
    """
    ei_seg = np.asarray(segment_ei, dtype=np.float64)
    m_seg = np.asarray(segment_mass, dtype=np.float64)
    x_j = np.asarray(joints, dtype=np.float64)
    delta = float(blend_width)

    if ei_seg.ndim != 1 or ei_seg.size < 2:
        raise ValueError(f"segment_ei must be 1-D with >= 2 segments, got shape {ei_seg.shape}")
    if m_seg.shape != ei_seg.shape:
        raise ValueError(
            f"segment_mass shape {m_seg.shape} does not match segment_ei shape {ei_seg.shape}"
        )
    if x_j.shape != (ei_seg.size - 1,):
        raise ValueError(
            f"need exactly {ei_seg.size - 1} joints for {ei_seg.size} segments, "
            f"got {x_j.size}"
        )
    if x_j.size > 1 and not np.all(np.diff(x_j) > 0.0):
        raise ValueError("joints must be strictly increasing")
    if not (np.isfinite(delta) and delta > 0.0):
        raise ValueError(f"blend_width must be finite and > 0, got {blend_width}")
    if np.any(ei_seg <= 0.0) or np.any(m_seg <= 0.0):
        raise ValueError("all segment EI and mass values must be > 0")

    d_ei = np.diff(ei_seg)  # step heights at each joint
    d_m = np.diff(m_seg)

    def _field(base: float, steps: _FloatArray) -> _Field:
        def field(x: _FloatArray) -> _FloatArray:
            xs = np.asarray(x, dtype=np.float64)
            arg = (xs[..., np.newaxis] - x_j) / delta
            return base + 0.5 * np.sum(steps * (1.0 + np.tanh(arg)), axis=-1)

        return field

    def _dfield(steps: _FloatArray) -> _Field:
        def dfield(x: _FloatArray) -> _FloatArray:
            xs = np.asarray(x, dtype=np.float64)
            arg = (xs[..., np.newaxis] - x_j) / delta
            return np.sum(steps / (2.0 * delta) * np.cosh(arg) ** -2, axis=-1)

        return dfield

    def _d2field(steps: _FloatArray) -> _Field:
        def d2field(x: _FloatArray) -> _FloatArray:
            xs = np.asarray(x, dtype=np.float64)
            arg = (xs[..., np.newaxis] - x_j) / delta
            sech2 = np.cosh(arg) ** -2
            return np.sum(-steps / delta**2 * sech2 * np.tanh(arg), axis=-1)

        return d2field

    return MaterialProfile(
        ei=_field(float(ei_seg[0]), d_ei),
        mass=_field(float(m_seg[0]), d_m),
        d_ei=_dfield(d_ei),
        d2_ei=_d2field(d_ei),
        label=label,
    )
