"""Structural kernel: profiles, operator assembly, null-space projection, modes.

The decisive test is the V1 acceptance criterion itself: free-free
frequencies of the uniform beam against the analytic solution with
relative error <= 1e-6 at N = 32.
"""

import numpy as np
import pytest
import scipy.linalg

from aether.spectral import ChebyshevGrid
from aether.structures import (
    assemble_beam,
    free_free_analytic_frequencies,
    project_free_free,
    solve_free_free_modes,
    stepped_profile,
    uniform_profile,
)
from aether.structures.modal import row_replacement_spectrum

EPS = np.finfo(np.float64).eps


def make_uniform(n=32, length=1.0, ei=1.0, mass=1.0):
    grid = ChebyshevGrid(n, interval=(0.0, length))
    return assemble_beam(grid, uniform_profile(ei, mass))


def make_stepped(n=48, length=10.0):
    # blend_width must exceed the mid-domain node spacing pi*L/(2N) or the
    # tanh transition is unresolved and spectral convergence stalls
    # (Paper I flags the same requirement for the slosh kernel bandwidth).
    profile = stepped_profile(
        segment_ei=[5.0e6, 1.2e6, 4.0e5],
        segment_mass=[300.0, 120.0, 60.0],
        joints=[4.0, 7.0],
        blend_width=0.8,
    )
    grid = ChebyshevGrid(n, interval=(0.0, length))
    return assemble_beam(grid, profile)


class TestProfiles:
    def test_uniform_fields(self):
        p = uniform_profile(2.5, 1.5)
        x = np.linspace(0, 3, 7)
        assert np.all(p.ei(x) == 2.5)
        assert np.all(p.mass(x) == 1.5)
        assert np.all(p.d_ei(x) == 0.0)

    def test_stepped_segment_values_far_from_joints(self):
        p = stepped_profile([2.0, 8.0], [1.0, 3.0], joints=[5.0], blend_width=0.1)
        assert p.ei(np.array(1.0)) == pytest.approx(2.0, rel=1e-12)
        assert p.ei(np.array(9.0)) == pytest.approx(8.0, rel=1e-12)
        assert p.ei(np.array(5.0)) == pytest.approx(5.0, rel=1e-12)  # midpoint of the blend
        assert p.mass(np.array(9.0)) == pytest.approx(3.0, rel=1e-12)

    def test_stepped_analytic_derivatives_match_finite_differences(self):
        p = stepped_profile([2.0, 8.0, 3.0], [1.0, 2.0, 1.5], joints=[3.0, 6.0], blend_width=0.5)
        x = np.linspace(0.5, 8.5, 33)
        h = 1e-4  # balances truncation (h^2 f'''') against cancellation (eps f / h^2)
        fd1 = (p.ei(x + h) - p.ei(x - h)) / (2 * h)
        fd2 = (p.ei(x + h) - 2 * p.ei(x) + p.ei(x - h)) / h**2
        assert np.allclose(p.d_ei(x), fd1, rtol=1e-7, atol=1e-7)
        assert np.allclose(p.d2_ei(x), fd2, rtol=1e-5, atol=1e-5)

    def test_validation_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            uniform_profile(0.0, 1.0)
        with pytest.raises(ValueError):
            uniform_profile(1.0, -2.0)
        with pytest.raises(ValueError):
            stepped_profile([1.0, -1.0], [1.0, 1.0], joints=[1.0], blend_width=0.1)
        with pytest.raises(ValueError):
            stepped_profile([1.0, 2.0], [1.0, 1.0], joints=[1.0], blend_width=0.0)
        with pytest.raises(ValueError):
            stepped_profile([1.0, 2.0, 3.0], [1.0] * 3, joints=[5.0, 2.0], blend_width=0.1)


class TestBeamAssembly:
    def test_uniform_reduces_to_leading_term(self):
        """With constant EI the product-rule terms must vanish to rounding."""
        beam = make_uniform(n=20, length=2.0, ei=3.0)
        d4 = beam.grid.diffmat(4)
        scale = np.linalg.norm(3.0 * d4, np.inf)
        assert np.max(np.abs(beam.stiffness - 3.0 * d4)) <= 1e3 * EPS * scale

    def test_manufactured_solution_variable_ei(self):
        """K w must reproduce (EI w'')'' for polynomial EI and w, where the
        collocation is exact up to conditioning."""
        length = 2.0
        grid = ChebyshevGrid(24, interval=(0.0, length))
        x = grid.x

        # EI = 2 + x^2, w = x^5:  (EI w'')'' = d^2/dx^2 (40x^3 + 20x^5)
        #                                    = 240 x + 400 x^3
        from aether.structures.profiles import MaterialProfile

        profile = MaterialProfile(
            ei=lambda s: 2.0 + s**2,
            mass=lambda s: np.ones_like(s),
            d_ei=lambda s: 2.0 * s,
            d2_ei=lambda s: np.full_like(s, 2.0),
            label="poly",
        )
        beam = assemble_beam(grid, profile)
        got = beam.stiffness @ x**5
        exact = 240.0 * x + 400.0 * x**3
        floor = 1e3 * EPS * np.linalg.norm(beam.stiffness, np.inf) * np.max(np.abs(x**5))
        assert np.max(np.abs(got - exact)) <= floor

    def test_rejects_wrong_domain_and_low_order_grid(self):
        grid = ChebyshevGrid(16, interval=(1.0, 2.0))
        with pytest.raises(ValueError, match="x = 0"):
            assemble_beam(grid, uniform_profile(1.0, 1.0))
        grid3 = ChebyshevGrid(16, interval=(0.0, 1.0), max_derivative=3)
        with pytest.raises(ValueError, match="order 4"):
            assemble_beam(grid3, uniform_profile(1.0, 1.0))


class TestNullSpaceProjection:
    @pytest.mark.parametrize("factory", [make_uniform, make_stepped])
    def test_basis_invariants(self, factory):
        proj = project_free_free(factory())
        z, b = proj.basis, proj.constraints
        n1 = proj.beam.size
        assert z.shape == (n1, n1 - 4)
        assert np.max(np.abs(b @ z)) <= 100 * EPS, "B Z = 0 must hold to rounding"
        assert np.max(np.abs(z.T @ z - np.eye(n1 - 4))) <= 100 * EPS

    def test_constraint_rank_is_four(self):
        proj = project_free_free(make_uniform())
        s = proj.constraint_singular_values
        assert s.size == 4
        assert s[-1] > 1e-8 * s[0]

    def test_reduced_mass_spd(self):
        proj = project_free_free(make_stepped())
        scipy.linalg.cholesky(proj.reduced_mass)  # raises if not SPD

    def test_rigid_modes_lie_in_kernel(self):
        """Constant and linear deflections satisfy free-free BCs, so both
        must be representable exactly in the reduced basis."""
        proj = project_free_free(make_uniform(n=24))
        x = proj.beam.grid.x
        for vec in (np.ones_like(x), x - x.mean()):
            recon = proj.expand(proj.reduce(vec))
            assert np.max(np.abs(recon - vec)) <= 1e3 * EPS * np.max(np.abs(vec))
            assert proj.boundary_residual(vec) <= 1e3 * EPS

    def test_expanded_vectors_satisfy_bcs(self):
        proj = project_free_free(make_stepped())
        rng = np.random.default_rng(7)
        w = proj.expand(rng.standard_normal(proj.reduced_dim))
        assert proj.boundary_residual(w) <= 1e3 * EPS * np.max(np.abs(w))

    def test_too_small_grid_raises(self):
        with pytest.raises(ValueError, match="n >= 5"):
            project_free_free(make_uniform(n=4))


class TestModalSolution:
    def test_v1_acceptance_uniform_n32(self):
        """The V1 failure criterion: relative frequency error > 1e-6 at
        N = 32 for the uniform case fails verification."""
        length, ei, mass = 1.0, 1.0, 1.0
        sol = solve_free_free_modes(project_free_free(make_uniform(32, length, ei, mass)))
        analytic = free_free_analytic_frequencies(5, length, ei, mass)
        got = sol.elastic_frequencies[:5]
        rel = np.abs(got - analytic) / analytic
        assert np.max(rel) <= 1e-6, f"V1 acceptance violated: {rel}"

    def test_two_rigid_modes_near_zero(self):
        sol = solve_free_free_modes(project_free_free(make_uniform(24)))
        assert sol.n_rigid == 2
        lam_scale = sol.eigenvalues[-1]
        assert np.all(np.abs(sol.eigenvalues[:2]) <= 1e-8 * lam_scale)

    def test_dimensional_scaling(self):
        """omega ~ sqrt(EI/m)/L^2: frequencies must scale exactly."""
        base = solve_free_free_modes(project_free_free(make_uniform(28, 1.0, 1.0, 1.0)))
        scaled = solve_free_free_modes(project_free_free(make_uniform(28, 2.0, 8.0, 0.5)))
        # sqrt(8/0.5)/4 = 1: identical spectra
        assert np.allclose(
            scaled.elastic_frequencies[:6], base.elastic_frequencies[:6], rtol=1e-9
        )

    def test_stepped_profile_real_nonnegative_spectrum(self):
        sol = solve_free_free_modes(project_free_free(make_stepped()))
        assert sol.n_rigid == 2
        assert np.all(sol.eigenvalues[2:] > 0)
        assert sol.max_imag_ratio <= 1e-10

    def test_stepped_self_convergence(self):
        """Without an analytic reference, the stepped case must converge in N,
        and the increments must shrink (spectral, not stalled)."""

        def freqs(n):
            return solve_free_free_modes(
                project_free_free(make_stepped(n=n))
            ).elastic_frequencies[:4]

        f48, f64, f80 = freqs(48), freqs(64), freqs(80)
        d_coarse = np.max(np.abs(f48 - f64) / f64)
        d_fine = np.max(np.abs(f64 - f80) / f80)
        assert d_fine < d_coarse, "refinement must reduce the increment"
        assert d_fine < 1e-5, f"stepped case not converged: {d_fine:.3e}"

    def test_translation_participation_completeness(self):
        sol = solve_free_free_modes(project_free_free(make_uniform(24)))
        # translation rigid mode carries all translational effective mass
        # exactly (it is inserted analytically and mass-normalized)
        assert sol.translation_participation[0] == pytest.approx(1.0, abs=1e-10)
        # completeness over the collocation eigenbasis is approximate: the
        # non-symmetric operator's eigenvectors are only near-mass-orthogonal
        assert sol.retained_participation(sol.frequencies.size) == pytest.approx(1.0, abs=1e-2)
        # elastic participation of low modes of a uniform beam is ~0
        assert np.all(sol.translation_participation[2:8] < 1e-6)

    def test_mode_shapes_satisfy_bcs_and_orthogonality(self):
        sol = solve_free_free_modes(project_free_free(make_uniform(32)))
        proj = sol.projection
        for i in range(6):
            assert proj.boundary_residual(sol.modes_full[:, i]) < 1e-8
        # quadrature-mass orthonormality for distinct elastic modes
        grid = proj.beam.grid
        mq = grid.weights * proj.beam.mass
        gram = sol.modes_full[:, 2:8].T @ (mq[:, None] * sol.modes_full[:, 2:8])
        assert np.max(np.abs(gram - np.eye(6))) < 1e-7

    def test_analytic_frequencies_known_values(self):
        """beta_n L roots must match the classical tabulated values."""
        w = free_free_analytic_frequencies(3, 1.0, 1.0, 1.0)
        beta = np.sqrt(w)
        assert beta[0] == pytest.approx(4.730040744862704, rel=1e-12)
        assert beta[1] == pytest.approx(7.853204624095838, rel=1e-12)
        assert beta[2] == pytest.approx(10.995607838001671, rel=1e-12)

    def test_truncation_basis(self):
        sol = solve_free_free_modes(project_free_free(make_uniform(24)))
        basis = sol.truncate(6)
        assert basis.n_modes == 6
        assert np.all(basis.omega[:2] == 0.0)
        assert np.all(np.diff(basis.omega[2:]) > 0)
        with pytest.raises(ValueError):
            sol.truncate(0)

    def test_row_replacement_pathology(self):
        """The counterexample of §3.2: row replacement must exhibit spectral
        contamination (complex pairs and/or negative/complex stiffness
        eigenvalues) that the null-space treatment avoids."""
        beam = make_uniform(32)
        lam_rr = row_replacement_spectrum(beam)
        sol = solve_free_free_modes(project_free_free(beam))
        # Null-space spectrum: clean (checked in strict mode inside the call).
        assert sol.max_imag_ratio <= 1e-10
        # Row replacement: the same physical problem shows imaginary
        # contamination orders of magnitude above the projected solve.
        scale = np.max(np.abs(lam_rr.real))
        rr_imag = np.max(np.abs(lam_rr.imag)) / scale
        assert rr_imag > 1e2 * sol.max_imag_ratio
