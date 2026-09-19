"""Mindlin–Reissner plate kernel (Paper II, §5)."""

import numpy as np
import pytest

from aether.plates import (
    MindlinPlate,
    OrthotropicLaminate,
    isotropic_laminate,
    simply_supported_exact,
    solve_plate_modes,
)
from aether.plates.laminate import SHEAR_CORRECTION_MINDLIN, SHEAR_CORRECTION_REISSNER
from aether.plates.mindlin import bivariate_coefficients

SS = ("simply_supported",) * 4


@pytest.fixture(scope="module")
def laminate():
    return isotropic_laminate(70e9, 0.3, 0.05, 2700.0)


class TestLaminate:
    def test_isotropic_relations(self):
        e, nu, h = 70e9, 0.3, 0.02
        lam = isotropic_laminate(e, nu, h, 2700.0)
        d = e * h**3 / (12 * (1 - nu**2))
        assert lam.d11 == pytest.approx(d)
        assert lam.d22 == pytest.approx(d)
        assert lam.d12 == pytest.approx(nu * d)
        assert lam.d66 == pytest.approx((1 - nu) * d / 2)
        assert lam.shear_xz == pytest.approx(e / (2 * (1 + nu)))
        assert lam.shear_correction == pytest.approx(5.0 / 6.0)

    def test_inertia_properties(self, laminate):
        assert laminate.mass_per_area == pytest.approx(2700.0 * 0.05)
        assert laminate.rotary_inertia == pytest.approx(2700.0 * 0.05**3 / 12)
        assert laminate.shear_stiffness_x == pytest.approx(
            (5 / 6) * laminate.shear_xz * 0.05
        )

    def test_shear_correction_constants_differ_under_two_percent(self):
        """Paper II: Mindlin's pi^2/12 differs from Reissner's 5/6 by under 2%."""
        rel = abs(SHEAR_CORRECTION_MINDLIN - SHEAR_CORRECTION_REISSNER) / (
            SHEAR_CORRECTION_REISSNER
        )
        assert rel < 0.02

    def test_with_thickness_scaling(self, laminate):
        thick = laminate.with_thickness(0.1)
        assert thick.d11 / laminate.d11 == pytest.approx(8.0)  # h^3
        assert thick.shear_stiffness_x / laminate.shear_stiffness_x == pytest.approx(2.0)

    def test_validation(self):
        with pytest.raises(ValueError, match="positive definite"):
            OrthotropicLaminate(1.0, 2.0, 1.0, 1.0, 1e9, 1e9, 0.01, 2700.0)
        with pytest.raises(ValueError, match="poisson_ratio"):
            isotropic_laminate(70e9, 0.7, 0.01, 2700.0)
        with pytest.raises(ValueError, match="shear_correction"):
            isotropic_laminate(70e9, 0.3, 0.01, 2700.0, shear_correction=0.0)


class TestOperatorAssembly:
    def test_manufactured_operator_exact(self, laminate):
        """MMS on Eqs. (5.5)–(5.7): the assembled operator must reproduce
        the analytic residual to rounding."""
        a, b = 1.3, 0.9
        s_x, s_y = laminate.shear_stiffness_x, laminate.shear_stiffness_y
        d11, d12, d22, d66 = laminate.d11, laminate.d12, laminate.d22, laminate.d66
        plate = MindlinPlate(laminate, 18, 18, a, b)
        x, y = plate.grid()
        xx, yy = np.meshgrid(x, y, indexing="ij")

        w = np.sin(1.3 * xx) * np.cos(0.7 * yy) + 0.2 * xx * yy
        px = np.cos(0.9 * xx) * np.sin(1.1 * yy)
        py = np.exp(0.3 * xx) * np.cos(0.5 * yy)
        w_x = 1.3 * np.cos(1.3 * xx) * np.cos(0.7 * yy) + 0.2 * yy
        w_y = -0.7 * np.sin(1.3 * xx) * np.sin(0.7 * yy) + 0.2 * xx
        w_xx = -1.69 * np.sin(1.3 * xx) * np.cos(0.7 * yy)
        w_yy = -0.49 * np.sin(1.3 * xx) * np.cos(0.7 * yy)
        px_x = -0.9 * np.sin(0.9 * xx) * np.sin(1.1 * yy)
        px_yy = -1.21 * np.cos(0.9 * xx) * np.sin(1.1 * yy)
        px_xx = -0.81 * np.cos(0.9 * xx) * np.sin(1.1 * yy)
        px_xy = -0.99 * np.sin(0.9 * xx) * np.cos(1.1 * yy)
        py_y = -0.5 * np.exp(0.3 * xx) * np.sin(0.5 * yy)
        py_xx = 0.09 * np.exp(0.3 * xx) * np.cos(0.5 * yy)
        py_yy = -0.25 * np.exp(0.3 * xx) * np.cos(0.5 * yy)
        py_xy = -0.15 * np.exp(0.3 * xx) * np.sin(0.5 * yy)

        r1 = -(s_x * (w_xx + px_x) + s_y * (w_yy + py_y))
        r2 = -(d11 * px_xx + d12 * py_xy + d66 * (px_yy + py_xy) - s_x * (px + w_x))
        r3 = -(d66 * (px_xy + py_xx) + d12 * px_xy + d22 * py_yy - s_y * (py + w_y))

        got = plate.apply(
            bivariate_coefficients(w),
            bivariate_coefficients(px),
            bivariate_coefficients(py),
        )
        for g, exact in zip(got, (r1, r2, r3), strict=True):
            ref = bivariate_coefficients(exact)
            assert np.max(np.abs(g - ref)) / np.max(np.abs(ref)) < 1e-10

    def test_rigid_modes_annihilated(self, laminate):
        """The three rigid-body motions must lie in the operator's null
        space exactly: w = 1; w = -x with phi_x = 1; w = -y with phi_y = 1."""
        plate = MindlinPlate(laminate, 12, 12, 1.0, 1.0)
        x, y = plate.grid()
        xx, yy = np.meshgrid(x, y, indexing="ij")
        ones = np.ones_like(xx)
        zeros = np.zeros_like(xx)
        rigid = [
            (ones, zeros, zeros),
            (-xx, ones, zeros),
            (-yy, zeros, ones),
        ]
        for w, px, py in rigid:
            out = plate.apply(
                bivariate_coefficients(w),
                bivariate_coefficients(px),
                bivariate_coefficients(py),
            )
            scale = max(laminate.shear_stiffness_x, laminate.d11)
            assert max(np.max(np.abs(o)) for o in out) < 1e-8 * scale

    def test_free_edge_corner_redundancy_is_four(self, laminate):
        """The twist condition is shared by both edges at each corner."""
        plate = MindlinPlate(laminate, 12, 14, 1.0, 1.2)
        assert plate.corner_redundancy == 4
        assert plate.all_free

    def test_simply_supported_corner_redundancy(self, laminate):
        plate = MindlinPlate(laminate, 12, 12, 1.0, 1.0, edges=SS)
        assert plate.corner_redundancy == 12
        assert not plate.all_free

    def test_constraints_annihilated_by_basis(self, laminate):
        plate = MindlinPlate(laminate, 10, 10, 1.0, 1.0)
        assert np.max(np.abs(plate.constraints @ plate.basis)) < 1e-9
        z = plate.basis
        assert np.max(np.abs(z.T @ z - np.eye(z.shape[1]))) < 1e-10

    def test_validation(self, laminate):
        with pytest.raises(ValueError, match="n_x, n_y >= 5"):
            MindlinPlate(laminate, 4, 10, 1.0, 1.0)
        with pytest.raises(ValueError, match="length_x"):
            MindlinPlate(laminate, 10, 10, 0.0, 1.0)
        with pytest.raises(ValueError, match="edges"):
            MindlinPlate(laminate, 10, 10, 1.0, 1.0, edges=("clamped",) * 4)
        with pytest.raises(ValueError, match="shape"):
            plate = MindlinPlate(laminate, 10, 10, 1.0, 1.0)
            plate.apply(np.zeros((3, 3)), np.zeros((10, 10)), np.zeros((10, 10)))


class TestExactSimplySupported:
    def test_spectral_convergence_to_exact_mindlin(self, laminate):
        """Against the closed-form Mindlin solution the error must contract
        exponentially and clear Paper II's 1e-5 criterion."""
        a, b = 1.0, 1.3
        exact = simply_supported_exact(laminate, a, b, 6)
        errs = []
        for n in (10, 14, 18):
            plate = MindlinPlate(laminate, n, n, a, b, edges=SS)
            modes = solve_plate_modes(plate, strict=False)
            errs.append(float(np.max(np.abs(modes.frequencies[:6] - exact) / exact)))
        assert errs[0] > errs[1] > errs[2], f"not converging: {errs}"
        assert errs[-1] < 1e-8, f"finest error {errs[-1]:.2e}"
        assert errs[1] < 1e-5, "must clear the II-V3 criterion by n = 14"

    def test_exact_reference_matches_thin_plate_limit(self):
        """As h -> 0 the exact Mindlin SS solution must approach the
        classical Navier result omega = (alpha^2+beta^2) sqrt(D/(rho h))."""
        a = b = 1.0
        thin = isotropic_laminate(70e9, 0.3, 1e-4, 2700.0)
        got = simply_supported_exact(thin, a, b, 1)[0]
        alpha = beta = np.pi
        classical = (alpha**2 + beta**2) * np.sqrt(
            thin.d11 / thin.mass_per_area
        )
        assert got == pytest.approx(classical, rel=1e-6)

    def test_no_rigid_modes_when_supported(self, laminate):
        plate = MindlinPlate(laminate, 12, 12, 1.0, 1.0, edges=SS)
        modes = solve_plate_modes(plate, strict=False)
        assert modes.n_rigid == 0
        assert np.all(modes.frequencies[:5] > 0.0)


class TestFreeFreeModes:
    def test_three_rigid_modes_cleanly_separated(self, laminate):
        plate = MindlinPlate(laminate, 14, 14, 1.0, 1.0)
        modes = solve_plate_modes(plate)
        assert modes.n_rigid == 3
        mags = np.sort(np.abs(modes.eigenvalues))
        assert mags[2] / mags[3] < 1e-9

    def test_spectrum_real_and_positive(self, laminate):
        plate = MindlinPlate(laminate, 14, 14, 1.0, 1.0)
        modes = solve_plate_modes(plate)
        assert modes.max_imag_ratio < 1e-8
        assert np.all(modes.eigenvalues[3:] > 0.0)

    def test_degenerate_pair_on_square_plate(self, laminate):
        """A square isotropic plate has a symmetry-degenerate mode pair;
        the discretization must reproduce the degeneracy."""
        plate = MindlinPlate(laminate, 16, 16, 1.0, 1.0)
        nd = solve_plate_modes(plate, strict=False).nondimensional(1.0, laminate)
        assert nd[3] == pytest.approx(nd[4], rel=1e-6)

    def test_self_convergence(self, laminate):
        """No analytic reference for FFFF, so refinement must contract."""

        def freqs(n):
            plate = MindlinPlate(laminate, n, n, 1.0, 1.0)
            return solve_plate_modes(plate, strict=False).nondimensional(1.0, laminate)[:3]

        f12, f16, f20 = freqs(12), freqs(16), freqs(20)
        d_coarse = np.max(np.abs(f12 - f16) / f16)
        d_fine = np.max(np.abs(f16 - f20) / f20)
        assert d_fine < d_coarse
        assert d_fine < 0.01

    def test_anisotropic_plate_runs(self):
        """An orthotropic laminate must produce a clean spectrum too."""
        lam = OrthotropicLaminate(
            d11=5.0e4, d12=1.0e4, d22=2.0e4, d66=1.2e4,
            shear_xz=4.0e9, shear_yz=3.0e9, thickness=0.04, density=1600.0,
        )
        modes = solve_plate_modes(MindlinPlate(lam, 12, 12, 1.0, 1.4))
        assert modes.n_rigid == 3
        assert np.all(modes.elastic_frequencies > 0.0)


class TestShearLocking:
    def test_no_locking_across_thickness_decades(self, laminate):
        """Paper II, Remark 3 / II-V2: high-order spectral discretizations
        are 'markedly less susceptible' to locking — measured, not assumed."""
        a = 1.0
        results = {}
        for ratio in (0.05, 0.01, 0.005, 0.001):
            lam = isotropic_laminate(70e9, 0.3, ratio * a, 2700.0)
            plate = MindlinPlate(lam, 16, 16, a, a)
            nd = solve_plate_modes(plate, strict=False).nondimensional(a, lam)
            results[ratio] = float(nd[0])
        thin_limit = results[0.005]
        # no spurious stiffening as h/L falls three decades
        assert abs(results[0.001] - thin_limit) / thin_limit < 0.01
        # and the physical softening at thick sections is present
        assert results[0.05] < thin_limit
