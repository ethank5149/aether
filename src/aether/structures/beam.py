"""Assembly of the collocated variable-rigidity Euler–Bernoulli operator.

Paper I, Eq. (3.5): the stiffness operator retains the full product-rule
expansion,

.. math::

    \\mathbf{K} = \\mathrm{diag}(\\mathbf{EI})\\,\\mathbf{D}_x^4
        + 2\\,\\mathrm{diag}(\\mathbf{D}_x \\mathbf{EI})\\,\\mathbf{D}_x^3
        + \\mathrm{diag}(\\mathbf{D}_x^2 \\mathbf{EI})\\,\\mathbf{D}_x^2,

with mass operator :math:`\\mathbf{M} = \\mathrm{diag}(\\mathbf{m})`.
Dropping the second and third terms is valid only where :math:`EI` is
near-constant over a bending wavelength, which fails across stage joints
and in regions of thermal softening — so all three terms are kept.

All derivative matrices here are *physically scaled*: the
:math:`(2/L)^k` factors of Paper I, Eq. (3.3) are carried inside
:class:`~aether.spectral.ChebyshevGrid`, so no scaling factor appears in
this module. That is deliberate — the affine factor is flagged by the
paper as "a common source of error", and centralizing it in one place is
the defense.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from aether.spectral import ChebyshevGrid
from aether.structures.profiles import MaterialProfile

__all__ = ["BeamOperators", "assemble_beam"]

_FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BeamOperators:
    """Collocated full-grid operators for one beam configuration.

    Attributes
    ----------
    grid:
        The Chebyshev grid on :math:`[0, L]`; note the descending node
        convention (``grid.x[0] = L``).
    profile:
        The material profile the operators were assembled from.
    ei:
        Nodal rigidity samples :math:`\\mathbf{EI}`.
    mass:
        Nodal mass-per-length samples :math:`\\mathbf{m}`.
    stiffness:
        The unconstrained collocation operator :math:`\\mathbf{K}`,
        shape ``(N + 1, N + 1)``. Not symmetric (Paper I, Remark 1).
    mass_matrix:
        :math:`\\mathbf{M} = \\mathrm{diag}(\\mathbf{m})`, stored dense
        for uniformity with the projected algebra downstream.
    """

    grid: ChebyshevGrid
    profile: MaterialProfile
    ei: _FloatArray = field(repr=False)
    mass: _FloatArray = field(repr=False)
    stiffness: _FloatArray = field(repr=False)
    mass_matrix: _FloatArray = field(repr=False)

    @property
    def size(self) -> int:
        """Full-grid dimension :math:`N + 1`."""
        return self.grid.size


def assemble_beam(grid: ChebyshevGrid, profile: MaterialProfile) -> BeamOperators:
    """Assemble :math:`\\mathbf{K}` and :math:`\\mathbf{M}` on ``grid``.

    The rigidity derivatives entering the product-rule terms are computed
    spectrally from the nodal samples (``diag(D EI)``, ``diag(D² EI)``),
    exactly as written in Paper I, Eq. (3.5) — not from the profile's
    analytic derivatives, which exist so that tests can quantify the
    difference.

    Parameters
    ----------
    grid:
        Chebyshev grid whose interval is the physical beam domain; must
        carry derivative operators up to fourth order.
    profile:
        Material fields; positivity is validated at the nodes.
    """
    if grid.max_derivative < 4:
        raise ValueError(
            f"grid must carry derivatives up to order 4 for the bending operator, "
            f"has {grid.max_derivative}"
        )
    a, _ = grid.interval
    if a != 0.0:
        raise ValueError(
            f"beam domain must start at x = 0 per Paper I convention, got interval {grid.interval}"
        )
    profile.validate_on(grid.x)

    d1 = grid.diffmat(1)
    d2 = grid.diffmat(2)
    d3 = grid.diffmat(3)
    d4 = grid.diffmat(4)

    ei = np.ascontiguousarray(profile.ei(grid.x))
    m = np.ascontiguousarray(profile.mass(grid.x))

    d_ei = d1 @ ei
    d2_ei = d2 @ ei

    stiffness = ei[:, np.newaxis] * d4 + 2.0 * d_ei[:, np.newaxis] * d3 + d2_ei[:, np.newaxis] * d2
    mass_matrix = np.diag(m)

    for arr in (ei, m, stiffness, mass_matrix):
        arr.flags.writeable = False

    return BeamOperators(
        grid=grid,
        profile=profile,
        ei=ei,
        mass=m,
        stiffness=stiffness,
        mass_matrix=mass_matrix,
    )
