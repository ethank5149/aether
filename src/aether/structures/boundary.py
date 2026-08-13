"""Free-free boundary conditions by null-space projection (Paper I, §3.2).

The free-free conditions — zero bending moment and zero shear at both
ends,

.. math::

    EI\\,w''\\big|_{x=0,L} = 0, \\qquad
    \\big(EI\\,w''\\big)'\\big|_{x=0,L} = 0,

collocate to four linear constraints :math:`\\mathbf{B}\\mathbf{w} = 0`.
The conventional row-replacement treatment destroys operator symmetry
and produces spurious growing modes; instead an orthonormal basis
:math:`\\mathbf{Z}` for :math:`\\ker\\mathbf{B}` is computed from the
trailing right singular vectors of :math:`\\mathbf{B}`, and the dynamics
are restricted to :math:`\\mathbf{w} = \\mathbf{Z}\\hat{\\mathbf{w}}`,
which satisfies the boundary conditions identically since
:math:`\\mathbf{B}\\mathbf{Z} = 0` by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from aether.structures.beam import BeamOperators

__all__ = ["FreeFreeProjection", "free_free_constraints", "project_free_free"]

_FloatArray = NDArray[np.float64]

_N_CONSTRAINTS = 4


@dataclass(frozen=True)
class FreeFreeProjection:
    """Null-space reduction of a beam configuration.

    Attributes
    ----------
    beam:
        The full-grid operators being reduced.
    constraints:
        :math:`\\mathbf{B} \\in \\mathbb{R}^{4 \\times (N+1)}`, rows
        scaled to unit norm (row scaling leaves the kernel unchanged and
        makes the SVD rank decision meaningful across operator orders).
    basis:
        :math:`\\mathbf{Z} \\in \\mathbb{R}^{(N+1) \\times (N-3)}` with
        orthonormal columns spanning :math:`\\ker\\mathbf{B}`.
    reduced_stiffness:
        :math:`\\hat{\\mathbf{K}} = \\mathbf{Z}^\\top\\mathbf{K}\\mathbf{Z}`.
    reduced_mass:
        :math:`\\hat{\\mathbf{M}} = \\mathbf{Z}^\\top\\mathbf{M}\\mathbf{Z}`,
        symmetric positive definite whenever :math:`\\mathbf{M} \\succ 0`.
    constraint_singular_values:
        The four singular values of the (row-scaled) constraint matrix,
        retained for diagnostics.
    """

    beam: BeamOperators
    constraints: _FloatArray = field(repr=False)
    basis: _FloatArray = field(repr=False)
    reduced_stiffness: _FloatArray = field(repr=False)
    reduced_mass: _FloatArray = field(repr=False)
    constraint_singular_values: _FloatArray = field(repr=False)

    @property
    def reduced_dim(self) -> int:
        """Dimension of the constrained subspace, :math:`N - 3`."""
        return int(self.basis.shape[1])

    def reduce(self, w: _FloatArray) -> _FloatArray:
        """Project a full-grid vector into reduced coordinates,
        :math:`\\hat{\\mathbf{w}} = \\mathbf{Z}^\\top \\mathbf{w}`.

        Exact (not merely least-squares) when ``w`` satisfies the
        boundary conditions, since such vectors lie in
        :math:`\\mathrm{range}\\,\\mathbf{Z}`.
        """
        return self.basis.T @ w

    def expand(self, w_hat: _FloatArray) -> _FloatArray:
        """Lift reduced coordinates to the full grid,
        :math:`\\mathbf{w} = \\mathbf{Z}\\hat{\\mathbf{w}}`."""
        return self.basis @ w_hat

    def boundary_residual(self, w: _FloatArray) -> float:
        """Max-norm residual of the four boundary conditions at ``w``."""
        return float(np.max(np.abs(self.constraints @ w)))


def free_free_constraints(beam: BeamOperators) -> _FloatArray:
    """Collocated free-free constraint matrix :math:`\\mathbf{B}`.

    Rows, in order: moment at :math:`x = L` (node 0), moment at
    :math:`x = 0` (node :math:`N`), shear at :math:`x = L`, shear at
    :math:`x = 0`. The shear rows expand
    :math:`(EI\\,w'')' = EI\\,w''' + EI'\\,w''` with the *spectral*
    derivative of the sampled rigidity, consistent with the stiffness
    assembly. Each row is scaled to unit Euclidean norm.
    """
    grid = beam.grid
    if grid.n < 5:
        raise ValueError(
            f"free-free projection needs polynomial order n >= 5 "
            f"(4 constraints + 2 rigid modes + at least 1 elastic dof), got n = {grid.n}"
        )
    d1, d2, d3 = grid.diffmat(1), grid.diffmat(2), grid.diffmat(3)
    ei = beam.ei
    d_ei = d1 @ ei

    end_nodes = (0, grid.n)  # x = L and x = 0 under the descending convention
    rows = []
    for j in end_nodes:
        rows.append(ei[j] * d2[j, :])  # bending moment
    for j in end_nodes:
        rows.append(ei[j] * d3[j, :] + d_ei[j] * d2[j, :])  # shear
    b = np.vstack(rows)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    b.flags.writeable = False
    return b


def project_free_free(beam: BeamOperators) -> FreeFreeProjection:
    """Build the null-space basis and the reduced operator pair.

    The basis is taken from the trailing right singular vectors of
    :math:`\\mathbf{B}`; the constraint rank is verified to be exactly 4
    against a standard singular-value threshold before the split, so a
    degenerate constraint set fails loudly rather than silently widening
    the admissible space.
    """
    b = free_free_constraints(beam)
    _, s, vt = scipy.linalg.svd(b, full_matrices=True, lapack_driver="gesdd")
    tol = max(b.shape) * np.finfo(np.float64).eps * s[0]
    rank = int(np.count_nonzero(s > tol))
    if rank != _N_CONSTRAINTS:
        raise np.linalg.LinAlgError(
            f"free-free constraint matrix has numerical rank {rank}, expected "
            f"{_N_CONSTRAINTS}; singular values {s}"
        )
    z = np.ascontiguousarray(vt[_N_CONSTRAINTS:, :].T)

    k_hat = z.T @ beam.stiffness @ z
    m_hat = z.T @ beam.mass_matrix @ z
    # Zᵀ diag(m) Z is symmetric in exact arithmetic; enforce it so the
    # SPD property survives rounding for the Cholesky-based consumers.
    m_hat = 0.5 * (m_hat + m_hat.T)

    s = np.ascontiguousarray(s)
    for arr in (z, k_hat, m_hat, s):
        arr.flags.writeable = False

    return FreeFreeProjection(
        beam=beam,
        constraints=b,
        basis=z,
        reduced_stiffness=k_hat,
        reduced_mass=m_hat,
        constraint_singular_values=s,
    )
