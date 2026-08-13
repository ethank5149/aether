"""Free-free modal solution of the reduced pencil (Paper I, §3.2, §3.6).

Solves the generalized eigenproblem
:math:`\\hat{\\mathbf{K}}\\hat{\\mathbf{w}} = \\lambda\\hat{\\mathbf{M}}\\hat{\\mathbf{w}}`
by the QZ algorithm. Because the collocation stiffness operator is not
symmetric (Paper I, Remark 1), the computed spectrum is nominally
complex; physical correctness requires it to be real and non-negative up
to rounding, and this module *verifies* that instead of assuming it,
recording the worst imaginary contamination as a diagnostic and failing
loudly when it exceeds tolerance.

A free-free beam retains exactly two rigid-body modes (translation and
rotation) after projection — the null-space basis removes boundary
*constraint violations*, not rigid motion. The solver identifies and
counts them.

Mode normalization uses the quadrature mass norm
:math:`\\sum_j w_j^{\\mathrm{CC}} m_j W_j^2 = 1` (the discrete
:math:`\\int m w^2\\,dx`), which is the physically meaningful inner
product on a collocation grid and makes the effective-mass participation
used for modal truncation (Paper I, §3.6) exact by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from aether.structures.beam import BeamOperators
from aether.structures.boundary import FreeFreeProjection, free_free_constraints

__all__ = [
    "ModalBasis",
    "ModalSolution",
    "free_free_analytic_frequencies",
    "row_replacement_spectrum",
    "solve_free_free_modes",
]

_FloatArray = NDArray[np.float64]
_N_RIGID_EXPECTED = 2


@dataclass(frozen=True)
class ModalBasis:
    """A truncated set of mass-normalized modes for time integration.

    Attributes
    ----------
    omega:
        Natural frequencies (rad/s), ascending; rigid modes contribute
        exact zeros.
    modes_reduced:
        Mode shapes in reduced (null-space) coordinates, one per column.
    modes_full:
        The same modes lifted to the full grid, one per column.
    """

    omega: _FloatArray
    modes_reduced: _FloatArray = field(repr=False)
    modes_full: _FloatArray = field(repr=False)

    @property
    def n_modes(self) -> int:
        return int(self.omega.size)


@dataclass(frozen=True)
class ModalSolution:
    """Complete spectral data for one free-free beam configuration.

    Attributes
    ----------
    projection:
        The null-space reduction the pencil was built from.
    eigenvalues:
        Generalized eigenvalues :math:`\\lambda_i = \\omega_i^2`
        (rad²/s²), real, ascending. Rigid-mode entries are the raw
        near-zero computed values — *not* snapped to zero — so their
        magnitude remains visible as a quality metric.
    frequencies:
        :math:`\\omega_i = \\sqrt{\\max(\\lambda_i, 0)}` in rad/s.
    modes_reduced, modes_full:
        Quadrature-mass-normalized eigenvectors (columns), in reduced
        coordinates and lifted to the grid.
    n_rigid:
        Number of rigid-body modes detected (must be 2 in strict mode).
    stiffness_condition:
        Raw :math:`\\kappa_2(\\hat{\\mathbf{K}})` — the V1 measurand. For
        a free-free beam :math:`\\hat{\\mathbf{K}}` retains the two
        *physical* rigid-body null directions, so this saturates at the
        reciprocal rounding floor (:math:`\\sim 1/\\varepsilon`)
        regardless of :math:`N`; it is reported because V1 asks for it,
        and interpreted alongside the elastic value.
    stiffness_condition_elastic:
        :math:`\\sigma_1/\\sigma_{n-2}` — the condition number of
        :math:`\\hat{\\mathbf{K}}` restricted to its regular (elastic)
        part, excluding the two rigid null directions. This is the
        quantity whose growth with :math:`N` is physically meaningful.
    max_imag_ratio:
        :math:`\\max_i |\\mathrm{Im}\\,\\lambda_i| / \\max_i |\\lambda_i|`
        before the real part was taken.
    translation_participation:
        Effective-mass fraction of each mode for unit rigid translation;
        sums to 1 over the complete mode set.
    """

    projection: FreeFreeProjection
    eigenvalues: _FloatArray
    frequencies: _FloatArray
    modes_reduced: _FloatArray = field(repr=False)
    modes_full: _FloatArray = field(repr=False)
    n_rigid: int
    stiffness_condition: float
    stiffness_condition_elastic: float
    max_imag_ratio: float
    translation_participation: _FloatArray = field(repr=False)

    @property
    def elastic_frequencies(self) -> _FloatArray:
        """Frequencies of the elastic modes only."""
        return self.frequencies[self.n_rigid :]

    def truncate(self, n_modes: int) -> ModalBasis:
        """Retain the lowest ``n_modes`` modes (rigid modes included).

        This is the modal-truncation mitigation of Paper I, §3.6: it
        caps :math:`\\omega_{\\max}` at :math:`\\omega_{n_m}` and thereby
        restores a workable explicit step size. The retained-mode
        translation participation quantifies the approximation.
        """
        if not 1 <= n_modes <= self.frequencies.size:
            raise ValueError(
                f"n_modes must be in [1, {self.frequencies.size}], got {n_modes}"
            )
        omega = self.frequencies[:n_modes].copy()
        omega[: self.n_rigid] = 0.0  # rigid modes propagate as exact drift
        omega.flags.writeable = False
        return ModalBasis(
            omega=omega,
            modes_reduced=self.modes_reduced[:, :n_modes],
            modes_full=self.modes_full[:, :n_modes],
        )

    def retained_participation(self, n_modes: int) -> float:
        """Cumulative translation effective-mass fraction of the lowest
        ``n_modes`` modes."""
        if not 1 <= n_modes <= self.frequencies.size:
            raise ValueError(
                f"n_modes must be in [1, {self.frequencies.size}], got {n_modes}"
            )
        return float(np.sum(self.translation_participation[:n_modes]))


def solve_free_free_modes(
    projection: FreeFreeProjection,
    imag_tol: float = 1e-8,
    rigid_tol: float = 1e-12,
    strict: bool = True,
) -> ModalSolution:
    """Solve the reduced pencil and classify the spectrum.

    Parameters
    ----------
    projection:
        Null-space reduction from
        :func:`~aether.structures.boundary.project_free_free`.
    imag_tol:
        Maximum tolerated :math:`|\\mathrm{Im}\\,\\lambda|` relative to
        the spectral radius before the solve is declared failed.
    rigid_tol:
        Eigenvalues below ``rigid_tol`` × (largest eigenvalue) are
        classified rigid. The default separates the two populations by
        about three decades on each side: computed rigid eigenvalues sit
        at the rounding floor (:math:`\\sim 10^{-15}` relative), while
        the first elastic eigenvalue is fixed at
        :math:`\\beta_1^4 EI/(mL^4)` and therefore *falls* relative to
        :math:`\\lambda_{\\max} = \\mathcal{O}(N^8)` — reaching
        :math:`\\sim 10^{-9}` relative around :math:`N = 64`. A looser
        threshold silently absorbs elastic modes at large :math:`N`.
    strict:
        If true (default), raise unless exactly two rigid modes are
        found and the spectrum is real and non-negative within
        tolerance. Setting false records the anomalies in the returned
        diagnostics instead, for use by verification sweeps that *study*
        failure.
    """
    k_hat = projection.reduced_stiffness
    m_hat = projection.reduced_mass

    lam_c, phi_c = scipy.linalg.eig(k_hat, m_hat)
    if np.any(~np.isfinite(lam_c)):
        raise np.linalg.LinAlgError(
            "reduced pencil returned non-finite eigenvalues; the reduced mass "
            "matrix should be SPD and cannot produce an infinite spectrum"
        )

    scale = float(np.max(np.abs(lam_c.real))) if lam_c.size else 0.0
    max_imag_ratio = float(np.max(np.abs(lam_c.imag)) / scale) if scale > 0.0 else 0.0
    if strict and max_imag_ratio > imag_tol:
        raise np.linalg.LinAlgError(
            f"spectrum has relative imaginary contamination {max_imag_ratio:.3e} "
            f"> imag_tol {imag_tol:.3e}; the projected free-free pencil should be "
            f"real — suspect the constraint basis or the operator assembly"
        )

    order = np.argsort(lam_c.real)
    lam = np.ascontiguousarray(lam_c.real[order])
    phi = np.ascontiguousarray(phi_c.real[:, order])

    n_rigid = int(np.count_nonzero(np.abs(lam) < rigid_tol * scale)) if scale > 0.0 else lam.size
    if strict and n_rigid != _N_RIGID_EXPECTED:
        raise np.linalg.LinAlgError(
            f"detected {n_rigid} rigid-body modes, expected {_N_RIGID_EXPECTED} "
            f"(translation and rotation); smallest |eigenvalues|: "
            f"{np.sort(np.abs(lam))[:4]}"
        )
    if strict and np.any(lam[n_rigid:] <= 0.0):
        raise np.linalg.LinAlgError(
            "negative elastic eigenvalue: the free-free operator must be "
            "positive semi-definite on the constrained subspace"
        )

    # Rigid modes from QZ span the rigid subspace but are an arbitrary
    # basis of it; replace them with the physical translation/rotation
    # pair, quadrature-mass-orthogonalized, so participation is exact.
    beam = projection.beam
    grid = beam.grid
    w_cc = grid.weights
    m_nodal = beam.mass
    mass_quad = w_cc * m_nodal  # diagonal of the quadrature mass form

    def _mass_norm(cols: _FloatArray) -> _FloatArray:
        return cast(_FloatArray, np.sqrt(np.einsum("jk,j,jk->k", cols, mass_quad, cols)))

    z = projection.basis
    modes_full = z @ phi
    if n_rigid == _N_RIGID_EXPECTED:
        total_mass = float(mass_quad.sum())
        translation = np.ones(grid.size)
        x_cg = float((mass_quad @ grid.x) / total_mass)
        rotation = grid.x - x_cg  # mass-orthogonal to translation by choice of x_cg
        rigid = np.column_stack([translation, rotation])
        rigid /= _mass_norm(rigid)
        modes_full[:, :n_rigid] = rigid

    norms = _mass_norm(modes_full)
    if np.any(norms == 0.0):
        raise np.linalg.LinAlgError("zero-norm eigenvector returned by QZ")
    modes_full = modes_full / norms
    # Sign convention: largest-magnitude nodal component positive.
    idx = np.argmax(np.abs(modes_full), axis=0)
    signs = np.sign(modes_full[idx, np.arange(modes_full.shape[1])])
    modes_full *= signs
    modes_reduced = z.T @ modes_full

    freqs = np.sqrt(np.maximum(lam, 0.0))

    # Effective-mass participation for unit rigid translation:
    # Gamma_i = int m W_i dx; effective mass Gamma_i^2 sums to total mass
    # over the complete (mass-orthonormal) set.
    gamma = modes_full.T @ mass_quad
    participation = gamma**2 / mass_quad.sum()

    sing = scipy.linalg.svdvals(k_hat)
    kappa = float(sing[0] / sing[-1])
    kappa_elastic = float(sing[0] / sing[-3]) if sing.size >= 3 else kappa

    for arr in (lam, freqs, modes_reduced, modes_full, participation):
        arr.flags.writeable = False

    return ModalSolution(
        projection=projection,
        eigenvalues=lam,
        frequencies=freqs,
        modes_reduced=modes_reduced,
        modes_full=modes_full,
        n_rigid=n_rigid,
        stiffness_condition=kappa,
        stiffness_condition_elastic=kappa_elastic,
        max_imag_ratio=max_imag_ratio,
        translation_participation=participation,
    )


def free_free_analytic_frequencies(
    n_modes: int,
    length: float,
    ei: float,
    mass: float,
) -> _FloatArray:
    """Elastic natural frequencies of the uniform free-free beam.

    The characteristic equation is :math:`\\cos\\beta\\cosh\\beta = 1`
    with :math:`\\beta = \\beta_n L`; it is solved here in the
    numerically benign form :math:`\\cos\\beta - \\mathrm{sech}\\,\\beta = 0`
    (identical roots, no overflow of :math:`\\cosh`) by Newton iteration
    from the asymptotic guesses :math:`\\beta_n \\approx (n + 1/2)\\pi`,
    polished to machine precision. Frequencies follow as
    :math:`\\omega_n = \\beta_n^2 \\sqrt{EI/(m L^4)}`.

    Parameters
    ----------
    n_modes:
        Number of elastic modes requested.
    length, ei, mass:
        Beam length (m), rigidity (N·m²), and mass per length (kg/m).
    """
    if n_modes < 1:
        raise ValueError(f"n_modes must be >= 1, got {n_modes}")
    for name, val in (("length", length), ("ei", ei), ("mass", mass)):
        if not (np.isfinite(val) and val > 0.0):
            raise ValueError(f"{name} must be finite and > 0, got {val}")

    roots = np.empty(n_modes)
    for k in range(1, n_modes + 1):
        beta = (k + 0.5) * np.pi
        for _ in range(60):
            sech = 1.0 / np.cosh(beta)
            f = np.cos(beta) - sech
            df = -np.sin(beta) + np.tanh(beta) * sech
            step = f / df
            beta -= step
            if abs(step) < 1e-15 * beta:
                break
        else:  # pragma: no cover - Newton on this function converges in < 10 steps
            raise RuntimeError(f"Newton failed to converge for mode {k}")
        roots[k - 1] = beta

    freqs = cast(_FloatArray, roots**2 * np.sqrt(ei / (mass * length**4)))
    freqs.flags.writeable = False
    return freqs


def row_replacement_spectrum(beam: BeamOperators) -> NDArray[np.complex128]:
    """Spectrum of the *conventional* row-replacement treatment.

    Provided as the counterexample motivating §3.2: four rows of
    :math:`\\mathbf{K}` are overwritten with the constraint rows (and the
    matching mass rows zeroed), and the full complex spectrum of the
    resulting pencil is returned — infinite eigenvalues from the
    constraint rows filtered out. On free-free problems this treatment
    produces complex pairs and/or eigenvalues with positive real parts,
    which manifest as spurious growth in time integration. Not used by
    the solver path; exists so V1 can measure the pathology.
    """
    b = free_free_constraints(beam)
    grid = beam.grid
    n = grid.n
    replace_rows = (0, 1, n - 1, n)  # two rows at each end of the grid
    k_rr = beam.stiffness.copy()
    m_rr = beam.mass_matrix.copy()
    k_rr[list(replace_rows), :] = b
    m_rr[list(replace_rows), :] = 0.0
    lam = scipy.linalg.eig(k_rr, m_rr, right=False)
    finite = lam[np.isfinite(lam)]
    return cast(NDArray[np.complex128], np.ascontiguousarray(finite[np.argsort(finite.real)]))
