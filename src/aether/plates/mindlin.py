"""Bivariate Mindlin–Reissner plate operator (Paper II, §5).

The three-field system Eqs. (5.5)–(5.7) is *second order in each field*
— the structural reason the formulation suits spectral discretization,
since the highest derivative order is halved relative to Kirchhoff and
with it the conditioning penalty. Writing the shear and moment
resultants out, the governing operator in :math:`\\mathbf{M}\\ddot{
\\mathbf{u}} + \\mathbf{K}\\mathbf{u} = \\mathbf{Q}` form is the
:math:`3\\times 3` block

.. math::

    \\mathbf{K} = -\\begin{bmatrix}
      S_x \\partial_x^2 + S_y \\partial_y^2 & S_x \\partial_x & S_y \\partial_y \\\\
      -S_x \\partial_x & D_{11}\\partial_x^2 + D_{66}\\partial_y^2 - S_x
        & (D_{12}+D_{66})\\partial_{xy} \\\\
      -S_y \\partial_y & (D_{12}+D_{66})\\partial_{xy}
        & D_{66}\\partial_x^2 + D_{22}\\partial_y^2 - S_y
    \\end{bmatrix},

with :math:`S_x = \\kappa_s^2 G_{xz} h`, :math:`S_y = \\kappa_s^2 G_{yz}
h`. Every entry is a sum of separable terms, so each assembles as a
Kronecker product :math:`\\mathbf{A}_j \\otimes \\mathbf{B}_j` of
one-dimensional ultraspherical factors (Paper II, Eq. 5.19) — the
Sylvester structure is preserved in the assembly rather than being
destroyed by forming a dense bivariate matrix.

**Boundary conditions.** Three *independent* free-edge conditions are
imposed per edge — :math:`M_x = M_{xy} = Q_x = 0` on an edge normal to
:math:`x` (Eq. 5.9) — never the Kirchhoff effective-shear pair, which
over-constrains the perimeter. The third condition is evaluated
independently of the geometric slope: collapsing it onto
:math:`\\partial w/\\partial x` would reintroduce exactly the constraint
the three-field formulation exists to relax.

**Corner redundancy.** The edge-condition set is rank deficient at the
corners, and the deficiency is *measured* rather than assumed — see
:attr:`MindlinPlate.corner_redundancy`. For a uniformly free perimeter
it is four, one per corner, for a specific reason: the twist condition
:math:`M_{xy} = 0` belongs to the free-edge triple of *both* edges
meeting at a corner, so it is imposed twice there, while
:math:`M_x \\ne M_y` and :math:`Q_x \\ne Q_y` are independent. A
uniformly simply-supported perimeter gives twelve. A wrong boundary
assembly surfaces as an out-of-range deficiency rather than as quietly
wrong frequencies.

**Constraint handling.** Conditions are imposed by *null-space
projection*, the same treatment Paper I §3.2 applies to the free-free
beam and for the same reason: it degrades gracefully under the corner
redundancy where row replacement does not. The projection is a genuine
Rayleigh–Ritz restriction because the residual is first mapped back to
the Chebyshev basis, which also renders the inertia operator exactly
diagonal — so the projected mass matrix is SPD by construction and the
pencil is well posed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.linalg
import scipy.sparse
from numpy.typing import NDArray

from aether.plates.laminate import OrthotropicLaminate
from aether.ultraspherical import (
    chebyshev_coefficients,
    conversion_chain,
    derivative_in_basis,
    evaluation_row,
)

__all__ = [
    "MindlinPlate",
    "PlateModes",
    "bivariate_coefficients",
    "kirchhoff_free_free_reference",
    "simply_supported_exact",
    "solve_plate_modes",
]

_FloatArray = NDArray[np.float64]
_Sparse = Any

#: Free-free plates carry three rigid-body modes: transverse translation
#: and the two out-of-plane rotations.
_N_RIGID = 3


def _apply_scaling(matrix: _FloatArray, scaling: str, sweeps: int = 12) -> _FloatArray:
    """Diagonal rescaling before a condition-number measurement.

    Ruiz equilibration alternately normalizes rows and columns to unit
    max-norm, converging to a two-sided scaling under which the
    condition number reflects the operator rather than the unit system
    its blocks happen to be expressed in.
    """
    if scaling == "none":
        return matrix
    if scaling == "column":
        norms = np.linalg.norm(matrix, axis=0)
        norms[norms == 0.0] = 1.0
        return np.asarray(matrix / norms)
    if scaling != "ruiz":
        raise ValueError(f"scaling must be 'ruiz', 'column' or 'none', got {scaling!r}")
    scaled = matrix.copy()
    for _ in range(sweeps):
        rows = np.sqrt(np.linalg.norm(scaled, np.inf, axis=1))
        rows[rows == 0.0] = 1.0
        scaled = scaled / rows[:, np.newaxis]
        cols = np.sqrt(np.linalg.norm(scaled, np.inf, axis=0))
        cols[cols == 0.0] = 1.0
        scaled = scaled / cols[np.newaxis, :]
    return np.asarray(scaled)


def _kron2(a: _Sparse, b: _Sparse) -> _Sparse:
    """Kronecker product on the sparse *array* interface (no legacy warning)."""
    return scipy.sparse.kron(scipy.sparse.csr_array(a), scipy.sparse.csr_array(b), format="csr")


def _cgl(n: int) -> _FloatArray:
    j = np.arange(n)
    return np.asarray(np.sin(np.pi * (n - 1 - 2.0 * j) / (2.0 * (n - 1))))


def bivariate_coefficients(values: _FloatArray) -> _FloatArray:
    """Tensor-product Chebyshev coefficients of nodal values on the CGL grid.

    ``values`` has shape ``(n_x, n_y)`` sampled at the descending CGL
    grids of each direction.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError(f"values must be 2-D, got shape {v.shape}")
    stage = np.apply_along_axis(chebyshev_coefficients, 0, v)
    return np.asarray(np.apply_along_axis(chebyshev_coefficients, 1, stage))


@dataclass(frozen=True)
class PlateModes:
    """Free-vibration spectrum of a plate configuration.

    Attributes
    ----------
    frequencies:
        Natural frequencies :math:`\\omega` (rad/s), ascending, rigid
        modes included as the leading (near-zero) entries.
    eigenvalues:
        Raw :math:`\\lambda = \\omega^2` values — rigid entries are *not*
        snapped to zero, so their magnitude stays visible as a quality
        metric.
    n_rigid:
        Number of rigid-body modes detected (3 for a free-free plate).
    max_imag_ratio:
        Relative imaginary contamination of the computed spectrum before
        the real part was taken.
    """

    frequencies: _FloatArray
    eigenvalues: _FloatArray
    n_rigid: int
    max_imag_ratio: float
    modes_reduced: _FloatArray = field(repr=False, default_factory=lambda: np.empty(0))

    @property
    def elastic_frequencies(self) -> _FloatArray:
        """Frequencies of the elastic modes only."""
        return self.frequencies[self.n_rigid :]

    def nondimensional(self, length: float, laminate: OrthotropicLaminate) -> _FloatArray:
        """:math:`\\lambda = \\omega a^2\\sqrt{\\rho h/D_{11}}` for the elastic
        modes — the form plate frequencies are conventionally tabulated in."""
        return np.asarray(
            self.elastic_frequencies * length**2 * np.sqrt(laminate.mass_per_area / laminate.d11)
        )


class MindlinPlate:
    """Kronecker-assembled Mindlin–Reissner operator on a rectangle.

    Parameters
    ----------
    laminate:
        Section rigidities and inertias (constant over the planform;
        see :mod:`aether.plates.laminate` on the variable-coefficient
        extension path).
    n_x, n_y:
        Coefficient-space dimensions per direction, at least 5.
    length_x, length_y:
        Planform dimensions (m).
    """

    #: Ultraspherical output basis: the system is second order in each field.
    BASIS = 2

    def __init__(
        self,
        laminate: OrthotropicLaminate,
        n_x: int,
        n_y: int,
        length_x: float,
        length_y: float,
        edges: tuple[str, str, str, str] = ("free", "free", "free", "free"),
    ) -> None:
        if n_x < 5 or n_y < 5:
            raise ValueError(f"need n_x, n_y >= 5, got ({n_x}, {n_y})")
        for name, val in (("length_x", length_x), ("length_y", length_y)):
            if not (np.isfinite(val) and val > 0.0):
                raise ValueError(f"{name} must be finite and > 0, got {val}")
        if len(edges) != 4 or any(e not in ("free", "simply_supported") for e in edges):
            raise ValueError(
                f"edges must be four entries from ('free', 'simply_supported'), got {edges}"
            )

        self._lam = laminate
        self._nx = int(n_x)
        self._ny = int(n_y)
        self._lx = float(length_x)
        self._ly = float(length_y)
        self._sx = 2.0 / self._lx
        self._sy = 2.0 / self._ly
        self._edges = tuple(edges)
        self._constraint_rank = 0
        self._corner_redundancy = 0

        self._stiffness = self._assemble_stiffness(chebyshev_output=False)
        self._stiffness_cheb = self._assemble_stiffness(chebyshev_output=True)
        self._constraints = self._assemble_boundary()
        self._basis = self._null_space(self._constraints)

    # ------------------------------------------------------------- geometry
    @property
    def laminate(self) -> OrthotropicLaminate:
        return self._lam

    @property
    def shape(self) -> tuple[int, int]:
        return (self._nx, self._ny)

    @property
    def dimensions(self) -> tuple[float, float]:
        return (self._lx, self._ly)

    @property
    def n_dof(self) -> int:
        """Total coefficient-space dimension, :math:`3 n_x n_y`."""
        return 3 * self._nx * self._ny

    @property
    def reduced_dim(self) -> int:
        """Dimension after free-edge projection."""
        return int(self._basis.shape[1])

    def grid(self) -> tuple[_FloatArray, _FloatArray]:
        """Physical CGL node coordinates ``(x, y)`` of each direction."""
        x = self._lx * (_cgl(self._nx) + 1.0) / 2.0
        y = self._ly * (_cgl(self._ny) + 1.0) / 2.0
        return x, y

    # ------------------------------------------------------------ assembly
    def _direction_ops(
        self, n: int, scale: float, chebyshev_output: bool
    ) -> tuple[_FloatArray, _FloatArray, _FloatArray]:
        """Per-direction factors for derivative orders 0, 1, 2.

        With ``chebyshev_output`` the conversion chain is divided out, so
        the factors map Chebyshev coefficients to Chebyshev coefficients:
        left-multiplying the whole pencil by the invertible
        :math:`(\\mathcal{S}_x \\otimes \\mathcal{S}_y)^{-1}` leaves the
        generalized eigenproblem unchanged, and — the reason for doing it
        — renders the inertia operator exactly diagonal.
        """
        ops = [derivative_in_basis(n, k, self.BASIS, scale).toarray() for k in range(3)]
        if chebyshev_output:
            conv = conversion_chain(n, 0, self.BASIS).toarray()
            ops = [scipy.linalg.solve_triangular(conv, op, lower=False) for op in ops]
        return ops[0], ops[1], ops[2]

    def _assemble_stiffness(self, chebyshev_output: bool) -> _Sparse:
        """The 3×3 block operator; every entry a sum of Kronecker terms."""
        lam = self._lam
        s_x, s_y = lam.shear_stiffness_x, lam.shear_stiffness_y
        d11, d12, d22, d66 = lam.d11, lam.d12, lam.d22, lam.d66

        x0, x1, x2 = self._direction_ops(self._nx, self._sx, chebyshev_output)
        y0, y1, y2 = self._direction_ops(self._ny, self._sy, chebyshev_output)

        def kr(a: _FloatArray, b: _FloatArray) -> _Sparse:
            return scipy.sparse.kron(
                scipy.sparse.csr_array(a), scipy.sparse.csr_array(b), format="csr"
            )

        k11 = -(s_x * kr(x2, y0) + s_y * kr(x0, y2))
        k12 = -s_x * kr(x1, y0)
        k13 = -s_y * kr(x0, y1)
        k21 = s_x * kr(x1, y0)
        k22 = -d11 * kr(x2, y0) - d66 * kr(x0, y2) + s_x * kr(x0, y0)
        k23 = -(d12 + d66) * kr(x1, y1)
        k31 = s_y * kr(x0, y1)
        k32 = -(d12 + d66) * kr(x1, y1)
        k33 = -d66 * kr(x2, y0) - d22 * kr(x0, y2) + s_y * kr(x0, y0)
        return scipy.sparse.bmat([[k11, k12, k13], [k21, k22, k23], [k31, k32, k33]], format="csr")

    def _mass_diagonal(self) -> _FloatArray:
        """Inertia in the Chebyshev-output form: exactly diagonal."""
        block = self._nx * self._ny
        lam = self._lam
        return np.concatenate(
            [
                np.full(block, lam.mass_per_area),
                np.full(block, lam.rotary_inertia),
                np.full(block, lam.rotary_inertia),
            ]
        )

    def _edge_row_x(self, endpoint: int, derivative: int) -> _FloatArray:
        return evaluation_row(self._nx, endpoint, derivative) * self._sx**derivative

    def _edge_row_y(self, endpoint: int, derivative: int) -> _FloatArray:
        return evaluation_row(self._ny, endpoint, derivative) * self._sy**derivative

    def _assemble_boundary(self) -> _FloatArray:
        """Boundary conditions as dense rows, three per edge.

        Free edges impose Paper II, Eq. (5.9); simply-supported edges
        impose the hard-support triple (:math:`w = 0`, tangential
        rotation zero, normal moment zero), which admits a closed-form
        Mindlin solution and therefore supplies the exact reference the
        verification leans on.

        Along each edge the condition is a function of the tangential
        coordinate; it is expanded in the tangential basis (converted to
        :math:`C^{(1)}` so all terms share one basis) and every
        coefficient set to zero.
        """
        lam = self._lam
        nx, ny = self._nx, self._ny
        n_block = nx * ny
        d11, d12, d22, d66 = lam.d11, lam.d12, lam.d22, lam.d66
        s_x, s_y = lam.shear_stiffness_x, lam.shear_stiffness_y

        ty0 = derivative_in_basis(ny, 0, 1, self._sy)
        ty1 = derivative_in_basis(ny, 1, 1, self._sy)
        tx0 = derivative_in_basis(nx, 0, 1, self._sx)
        tx1 = derivative_in_basis(nx, 1, 1, self._sx)

        rows: list[_Sparse] = []

        for end, edge in ((-1, self._edges[0]), (1, self._edges[1])):  # normal to x
            ex0 = scipy.sparse.csr_array(self._edge_row_x(end, 0).reshape(1, -1))
            ex1 = scipy.sparse.csr_array(self._edge_row_x(end, 1).reshape(1, -1))
            zero_blk = scipy.sparse.csr_array((ny, n_block))
            if edge == "free":
                # M_x = D11 dphix/dx + D12 dphiy/dy = 0
                rows.append(
                    scipy.sparse.hstack(
                        [zero_blk, d11 * _kron2(ex1, ty0), d12 * _kron2(ex0, ty1)], format="csr"
                    )
                )
                # M_xy = D66 (dphix/dy + dphiy/dx) = 0
                rows.append(
                    scipy.sparse.hstack(
                        [zero_blk, d66 * _kron2(ex0, ty1), d66 * _kron2(ex1, ty0)], format="csr"
                    )
                )
                # Q_x = kappa^2 Gxz h (phix + dw/dx) = 0, evaluated
                # independently of the geometric slope (Paper II, §5.3)
                rows.append(
                    scipy.sparse.hstack(
                        [s_x * _kron2(ex1, ty0), s_x * _kron2(ex0, ty0), zero_blk], format="csr"
                    )
                )
            else:  # hard simple support: w = 0, phi_y = 0, M_x = 0
                rows.append(
                    scipy.sparse.hstack([_kron2(ex0, ty0), zero_blk, zero_blk], format="csr")
                )
                rows.append(
                    scipy.sparse.hstack([zero_blk, zero_blk, _kron2(ex0, ty0)], format="csr")
                )
                rows.append(
                    scipy.sparse.hstack(
                        [zero_blk, d11 * _kron2(ex1, ty0), d12 * _kron2(ex0, ty1)], format="csr"
                    )
                )

        for end, edge in ((-1, self._edges[2]), (1, self._edges[3])):  # normal to y
            ey0 = scipy.sparse.csr_array(self._edge_row_y(end, 0).reshape(1, -1))
            ey1 = scipy.sparse.csr_array(self._edge_row_y(end, 1).reshape(1, -1))
            zero_blk = scipy.sparse.csr_array((nx, n_block))
            if edge == "free":
                # M_y = D12 dphix/dx + D22 dphiy/dy = 0
                rows.append(
                    scipy.sparse.hstack(
                        [zero_blk, d12 * _kron2(tx1, ey0), d22 * _kron2(tx0, ey1)], format="csr"
                    )
                )
                # M_xy = 0
                rows.append(
                    scipy.sparse.hstack(
                        [zero_blk, d66 * _kron2(tx0, ey1), d66 * _kron2(tx1, ey0)], format="csr"
                    )
                )
                # Q_y = kappa^2 Gyz h (phiy + dw/dy) = 0
                rows.append(
                    scipy.sparse.hstack(
                        [s_y * _kron2(tx0, ey1), zero_blk, s_y * _kron2(tx0, ey0)], format="csr"
                    )
                )
            else:  # hard simple support: w = 0, phi_x = 0, M_y = 0
                rows.append(
                    scipy.sparse.hstack([_kron2(tx0, ey0), zero_blk, zero_blk], format="csr")
                )
                rows.append(
                    scipy.sparse.hstack([zero_blk, _kron2(tx0, ey0), zero_blk], format="csr")
                )
                rows.append(
                    scipy.sparse.hstack(
                        [zero_blk, d12 * _kron2(tx1, ey0), d22 * _kron2(tx0, ey1)], format="csr"
                    )
                )

        block = scipy.sparse.vstack(rows, format="csr").toarray()
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return np.asarray(block / norms)

    def _null_space(self, constraints: _FloatArray) -> _FloatArray:
        """Orthonormal basis of :math:`\\ker\\mathbf{B}`, with the corner
        rank deficiency measured rather than assumed."""
        _, s, vt = scipy.linalg.svd(constraints, full_matrices=True)
        tol = max(constraints.shape) * np.finfo(np.float64).eps * s[0]
        rank = int(np.count_nonzero(s > tol))
        self._constraint_rank = rank
        self._corner_redundancy = int(constraints.shape[0] - rank)
        if not 0 <= self._corner_redundancy <= 16:
            raise np.linalg.LinAlgError(
                f"boundary condition set has rank deficiency "
                f"{self._corner_redundancy}, outside the 0–16 range four corners "
                f"can produce with three conditions per edge; the assembly is wrong"
            )
        interior_rows = 3 * (self._nx - 2) * (self._ny - 2)
        null_dim = constraints.shape[1] - rank
        if interior_rows < null_dim:
            raise np.linalg.LinAlgError(
                f"under-determined system: {interior_rows} resolved residual rows "
                f"for {null_dim} admissible degrees of freedom"
            )
        return np.ascontiguousarray(vt[rank:, :].T)

    # ------------------------------------------------------------ operators
    @property
    def stiffness(self) -> _Sparse:
        """Assembled block operator in the :math:`C^{(2)}\\otimes C^{(2)}`
        output basis — the banded Kronecker form of Eq. (5.19)."""
        return self._stiffness

    @property
    def constraints(self) -> _FloatArray:
        """Row-normalized boundary condition matrix."""
        return self._constraints

    @property
    def basis(self) -> _FloatArray:
        """Orthonormal basis of the boundary-condition null space."""
        return self._basis

    @property
    def edges(self) -> tuple[str, ...]:
        """Edge conditions, ordered ``(x_min, x_max, y_min, y_max)``."""
        return self._edges

    @property
    def all_free(self) -> bool:
        return all(e == "free" for e in self._edges)

    @property
    def corner_redundancy(self) -> int:
        """Measured rank deficiency of the boundary set.

        Four for a uniformly free perimeter — the twist condition
        :math:`M_{xy} = 0` belongs to the free-edge triple of *both*
        edges meeting at a corner, so it is imposed twice there, while
        :math:`M_x \\ne M_y` and :math:`Q_x \\ne Q_y` are not redundant.
        Twelve for a uniformly simply-supported perimeter, where the
        support conditions overlap far more strongly at the corners.
        Measured from the rank of the assembled set, never assumed: a
        wrong boundary assembly surfaces here rather than as quietly
        wrong frequencies.
        """
        return self._corner_redundancy

    def _interior_mask(self) -> NDArray[np.bool_]:
        """Residual rows retained: two dropped per direction per equation.

        Only the *resolved* residual coefficients are enforced. The
        dropped tail is where the truncated expansion aliases, and
        enforcing it is what makes a tau-type discretization converge
        slowly — measured directly during development, where enforcing
        the full residual cost roughly an order of magnitude in accuracy
        at equal :math:`N`.
        """
        keep = np.zeros((self._nx, self._ny), dtype=bool)
        keep[: self._nx - 2, : self._ny - 2] = True
        flat = keep.reshape(-1)
        return np.concatenate([flat, flat, flat])

    def projected_operators(self) -> tuple[_FloatArray, _FloatArray]:
        """Rectangular projected pencil on the free-edge null space.

        Rows are the resolved residual coefficients, columns the
        free-edge-admissible subspace: shape ``(m, d)`` with
        :math:`m = 3(n_x-2)(n_y-2)` and
        :math:`d = 3n_xn_y - \\mathrm{rank}\\,\\mathbf{B}`. The eight-row
        excess of :math:`m` over :math:`d` is resolved in the
        minimum-residual sense by :meth:`reduced_matrix`, not by dropping
        rows on an arbitrary rule.
        """
        mask = self._interior_mask()
        z = self._basis
        k_r = self._stiffness_cheb.toarray()[mask, :] @ z
        m_r = (self._mass_diagonal()[:, np.newaxis] * z)[mask, :]
        return np.asarray(k_r), np.asarray(m_r)

    def reduced_matrix(self) -> _FloatArray:
        """Square :math:`d \\times d` matrix whose spectrum is the plate's.

        The rectangular pencil :math:`(\\mathbf{K}_r, \\mathbf{M}_r)` is
        squared down through the thin QR factorization
        :math:`\\mathbf{M}_r = \\mathbf{Q}\\mathbf{R}`, giving
        :math:`\\mathbf{A} = \\mathbf{R}^{-1}\\mathbf{Q}^\\top
        \\mathbf{K}_r` — the minimum-residual reduction, which is exact
        whenever the eigenvector lies in the admissible subspace.
        """
        k_r, m_r = self.projected_operators()
        q, r = np.linalg.qr(m_r)
        if np.linalg.matrix_rank(r) < r.shape[0]:
            raise np.linalg.LinAlgError(
                "projected inertia operator is rank deficient; the free-edge "
                "null space and the resolved residual rows are inconsistent"
            )
        return np.asarray(scipy.linalg.solve_triangular(r, q.T @ k_r, lower=False))

    def condition_number(self, scaling: str = "ruiz") -> float:
        """:math:`\\kappa_2` of the projected block operator.

        For a uniformly free perimeter the three rigid-body directions
        are excluded: the operator annihilates them exactly, so the raw
        ratio is a rounding-floor artefact carrying no information about
        the discretization (Paper I's V1 documents the same trap for the
        free-free beam).

        Parameters
        ----------
        scaling:
            ``"ruiz"`` (default) applies symmetric two-sided
            equilibration, ``"column"`` scales columns only, ``"none"``
            measures the raw matrix. Two-sided is the meaningful default
            for a *block* operator, whose entries carry different
            physical units — shear stiffness in N/m against bending
            rigidity in N·m — so an unscaled condition number reports the
            unit system rather than the discretization.
        """
        k_hat, _ = self.projected_operators()
        dense = _apply_scaling(np.asarray(k_hat), scaling)
        s = np.linalg.svd(dense, compute_uv=False)
        drop = _N_RIGID if self.all_free else 0
        if s.size <= drop:  # pragma: no cover - guarded by n_x, n_y >= 5
            raise ValueError("operator too small to exclude the rigid null space")
        return float(s[0] / s[-(drop + 1)])

    def assembled_condition_number(self, scaling: str = "ruiz") -> float:
        """:math:`\\sigma_1/\\sigma_m` of the assembled banded interior — the
        rectangular block whose rows are the *resolved* residual
        coefficients, before boundary rows or projection.

        The full square operator is not the right object to measure: the
        differentiation operators shift by their order, so its trailing
        rows are structurally zero and its :math:`\\kappa_2` is infinite
        at every :math:`N` regardless of the discretization's quality.
        The resolved interior is the banded object the Olver–Townsend
        conditioning statement is actually about, and is the same
        quantity the univariate leg of II-V1 reports.
        """
        mask = self._interior_mask()
        dense = _apply_scaling(self._stiffness.toarray()[mask, :], scaling)
        s = np.linalg.svd(dense, compute_uv=False)
        return float(s[0] / s[-1])

    def apply(
        self, w: _FloatArray, phi_x: _FloatArray, phi_y: _FloatArray
    ) -> tuple[_FloatArray, _FloatArray, _FloatArray]:
        """Apply :math:`\\mathbf{K}` to Chebyshev coefficient blocks.

        Inputs are ``(n_x, n_y)`` coefficient matrices; the result is the
        three Chebyshev coefficient blocks of :math:`\\mathbf{K}\\mathbf{u}`
        — that is, of :math:`-(\\text{spatial operator})\\mathbf{u}`, which
        is what the manufactured-solution check compares against.
        """
        for name, arr in (("w", w), ("phi_x", phi_x), ("phi_y", phi_y)):
            if np.asarray(arr).shape != (self._nx, self._ny):
                raise ValueError(
                    f"{name} must have shape ({self._nx}, {self._ny}), got {np.asarray(arr).shape}"
                )
        vec = np.concatenate(
            [np.asarray(a, dtype=np.float64).reshape(-1) for a in (w, phi_x, phi_y)]
        )
        out = np.asarray(self._stiffness_cheb @ vec)
        blocks = np.split(out, 3)
        return (
            blocks[0].reshape(self._nx, self._ny),
            blocks[1].reshape(self._nx, self._ny),
            blocks[2].reshape(self._nx, self._ny),
        )


def solve_plate_modes(
    plate: MindlinPlate,
    imag_tol: float = 1.0e-8,
    rigid_tol: float = 1.0e-9,
    strict: bool = True,
    resolved_fraction: float = 0.25,
) -> PlateModes:
    """Free-vibration spectrum of the projected plate pencil.

    The strong-form operator is not symmetric (the same situation as
    Paper I, Remark 1), so the spectrum's reality and non-negativity are
    *verified* rather than assumed. The check is applied to the
    **resolved** part of the spectrum: the unresolved high tail of a
    truncated strong-form discretization does acquire complex pairs, and
    demanding otherwise would be demanding accuracy the discretization
    never claimed. ``resolved_fraction`` sets how much of the spectrum
    counts as resolved.
    """
    a_mat = plate.reduced_matrix()
    lam_c, phi_c = scipy.linalg.eig(a_mat)
    if not np.all(np.isfinite(lam_c)):
        raise np.linalg.LinAlgError("projected plate pencil returned non-finite eigenvalues")
    scale = float(np.max(np.abs(lam_c.real)))
    order_all = np.argsort(lam_c.real)
    n_resolved = max(1, int(resolved_fraction * lam_c.size))
    resolved = lam_c[order_all][:n_resolved]
    max_imag_ratio = float(np.max(np.abs(resolved.imag)) / scale) if scale > 0.0 else 0.0
    if strict and max_imag_ratio > imag_tol:
        raise np.linalg.LinAlgError(
            f"resolved plate spectrum has relative imaginary contamination "
            f"{max_imag_ratio:.3e} > {imag_tol:.3e}"
        )
    order = order_all
    lam = np.ascontiguousarray(lam_c.real[order])
    modes = np.ascontiguousarray(phi_c.real[:, order])

    # A uniformly free perimeter admits exactly three rigid-body modes
    # (transverse translation and the two out-of-plane rotations); any
    # support removes all of them. The count is a property of the
    # boundary configuration, so it is asserted structurally and then
    # *verified* by the separation from the first elastic eigenvalue —
    # a magnitude threshold relative to the spectral radius fails here,
    # because the radius scales as h^-2 and swallows real low modes for
    # thin plates.
    n_rigid = _N_RIGID if plate.all_free else 0
    magnitudes = np.sort(np.abs(lam))
    if n_rigid:
        first_elastic = magnitudes[n_rigid]
        separation = magnitudes[n_rigid - 1] / first_elastic if first_elastic > 0 else np.inf
        if strict and separation > rigid_tol:
            raise np.linalg.LinAlgError(
                f"rigid-body modes are not cleanly separated: "
                f"|lambda_{n_rigid}|/|lambda_{n_rigid + 1}| = {separation:.3e} "
                f"exceeds {rigid_tol:.3e}; the free-edge projection is suspect"
            )
    if strict and np.any(lam[n_rigid:] <= 0.0):
        raise np.linalg.LinAlgError(
            "negative elastic eigenvalue: the free-free plate operator must be "
            "positive semi-definite on the constrained subspace"
        )
    freqs = np.sqrt(np.maximum(lam, 0.0))
    for arr in (lam, freqs, modes):
        arr.flags.writeable = False
    return PlateModes(
        frequencies=freqs,
        eigenvalues=lam,
        n_rigid=n_rigid,
        max_imag_ratio=max_imag_ratio,
        modes_reduced=modes,
    )


def simply_supported_exact(
    laminate: OrthotropicLaminate,
    length_x: float,
    length_y: float,
    modes: int = 4,
) -> _FloatArray:
    """Exact Mindlin frequencies of an all-round simply-supported plate.

    For hard simple supports the three-field system separates exactly on
    :math:`w = W\\sin\\alpha x\\sin\\beta y`,
    :math:`\\phi_x = \\Phi_x\\cos\\alpha x\\sin\\beta y`,
    :math:`\\phi_y = \\Phi_y\\sin\\alpha x\\cos\\beta y` with
    :math:`\\alpha = p\\pi/a`, :math:`\\beta = q\\pi/b`, reducing each
    :math:`(p, q)` to the symmetric :math:`3\\times 3` pencil

    .. math::

        \\begin{bmatrix}
          S_x\\alpha^2 + S_y\\beta^2 & S_x\\alpha & S_y\\beta \\\\
          S_x\\alpha & D_{11}\\alpha^2 + D_{66}\\beta^2 + S_x
            & (D_{12}+D_{66})\\alpha\\beta \\\\
          S_y\\beta & (D_{12}+D_{66})\\alpha\\beta
            & D_{66}\\alpha^2 + D_{22}\\beta^2 + S_y
        \\end{bmatrix}
        \\mathbf{a} = \\omega^2\\,\\mathrm{diag}(\\rho h, I, I)\\,\\mathbf{a}.

    This is a *closed-form* reference for the complete operator —
    transverse shear, rotary inertia and all — and is therefore the
    verification anchor the free-free comparison cannot be, since the
    latter rests on tabulated values this repository cannot audit.

    Returns
    -------
    numpy.ndarray
        The lowest ``modes`` natural frequencies (rad/s), ascending,
        gathered over :math:`p, q \\ge 1`.
    """
    if modes < 1:
        raise ValueError(f"modes must be >= 1, got {modes}")
    lam = laminate
    span = max(4, modes + 2)
    freqs: list[float] = []
    for p in range(1, span + 1):
        for q in range(1, span + 1):
            alpha = p * np.pi / float(length_x)
            beta = q * np.pi / float(length_y)
            s_x, s_y = lam.shear_stiffness_x, lam.shear_stiffness_y
            k = np.array(
                [
                    [s_x * alpha**2 + s_y * beta**2, s_x * alpha, s_y * beta],
                    [
                        s_x * alpha,
                        lam.d11 * alpha**2 + lam.d66 * beta**2 + s_x,
                        (lam.d12 + lam.d66) * alpha * beta,
                    ],
                    [
                        s_y * beta,
                        (lam.d12 + lam.d66) * alpha * beta,
                        lam.d66 * alpha**2 + lam.d22 * beta**2 + s_y,
                    ],
                ]
            )
            m = np.diag([lam.mass_per_area, lam.rotary_inertia, lam.rotary_inertia])
            vals = scipy.linalg.eigvalsh(k, m)
            freqs.extend(np.sqrt(np.maximum(vals, 0.0)).tolist())
    out = np.sort(np.asarray(freqs))[:modes]
    out.flags.writeable = False
    return out


def kirchhoff_free_free_reference() -> _FloatArray:
    """Frequency parameters :math:`\\lambda = \\omega a^2\\sqrt{\\rho h/D}`
    commonly cited for the thin free-free square plate at :math:`\\nu = 0.3`.

    .. warning::

        These are the figures that circulate in the plate-vibration
        literature for the FFFF square plate. They are recorded as an
        *orientation* value only and are **not** verified against a
        publisher record, so they do not satisfy Paper II's V3 criterion
        ("published Rayleigh–Ritz values") under this repository's
        citation-audit standard. The verification runner treats them as
        an unverified cross-check and rests its verdict on the
        manufactured-solution leg and on self-convergence.
    """
    values = np.array([13.489, 19.789, 24.432, 35.024, 35.024, 61.526])
    values.flags.writeable = False
    return values
