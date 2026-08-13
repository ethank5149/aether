"""Ultraspherical spectral core (Paper II, §5.4, Appendix A)."""

import numpy as np
import pytest
import scipy.linalg
import scipy.special

from aether.spectral import ChebyshevGrid
from aether.structures import (
    assemble_beam,
    free_free_analytic_frequencies,
    project_free_free,
    solve_free_free_modes,
)
from aether.ultraspherical import (
    UltrasphericalBVP,
    VariableCoefficientOperator,
    chebyshev_coefficients,
    chebyshev_values,
    conversion_chain,
    conversion_operator,
    diff_operator,
    evaluation_row,
    jacobi_operator,
    multiplication_operator,
)
from aether.ultraspherical.assembly import BoundaryCondition


def cgl(n):
    return np.sin(np.pi * (n - 1 - 2 * np.arange(n)) / (2 * (n - 1)))


def gegenbauer_series(coeffs, lam, pts):
    return sum(
        float(c) * scipy.special.eval_gegenbauer(m, lam, pts) for m, c in enumerate(coeffs)
    )


class TestTransforms:
    def test_round_trip(self):
        n = 33
        x = cgl(n)
        f = np.exp(x) * np.sin(2 * x)
        a = chebyshev_coefficients(f)
        assert np.max(np.abs(chebyshev_values(a, x) - f)) < 1e-14

    def test_polynomial_coefficients_exact(self):
        n = 12
        x = cgl(n)
        a = chebyshev_coefficients(2.0 * x**2 - 1.0)  # = T_2
        expected = np.zeros(n)
        expected[2] = 1.0
        assert np.allclose(a, expected, atol=1e-14)

    def test_smooth_coefficients_decay(self):
        a = chebyshev_coefficients(np.exp(cgl(40)))
        assert abs(a[30]) < 1e-15 * abs(a[0])


class TestOperators:
    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    def test_diff_operator_spectrally_exact(self, k):
        """D_k maps exp(x) to its k-th derivative in C^(k).

        Evaluated on interior points: the Gegenbauer reference summation
        itself amplifies coefficient rounding at x = ±1, where
        C_m^{(k)}(±1) grows like m^{2k-1}."""
        n = 40
        a = chebyshev_coefficients(np.exp(cgl(n)))
        ck = np.asarray(diff_operator(n, k) @ a)
        pts = np.linspace(-0.9, 0.9, 17)
        assert np.max(np.abs(gegenbauer_series(ck, k, pts) - np.exp(pts))) < 1e-7

    def test_diff_operator_single_band(self):
        d = diff_operator(20, 3).toarray()
        i, j = np.nonzero(d)
        assert np.all(j - i == 3), "D_k must have exactly one superdiagonal"
        # entries 2^(k-1)(k-1)! * m at column m
        assert d[0, 3] == pytest.approx(4 * 2 * 3)
        assert d[1, 4] == pytest.approx(4 * 2 * 4)

    def test_conversion_operator_identities(self):
        """S_0 maps T-coefficients to C^(1)-coefficients of the same function."""
        n = 30
        a = chebyshev_coefficients(np.cos(3 * cgl(n)))
        c1 = np.asarray(conversion_operator(n, 0) @ a)
        pts = np.linspace(-1, 1, 15)
        assert np.max(np.abs(gegenbauer_series(c1, 1, pts) - np.cos(3 * pts))) < 1e-12

    @pytest.mark.parametrize("lam_to", [1, 2, 3, 4])
    def test_conversion_chain_preserves_function(self, lam_to):
        n = 36
        f = lambda s: np.exp(-s) + s**3  # noqa: E731
        a = chebyshev_coefficients(f(cgl(n)))
        c = np.asarray(conversion_chain(n, 0, lam_to) @ a)
        pts = np.linspace(-0.95, 0.95, 11)
        assert np.max(np.abs(gegenbauer_series(c, lam_to, pts) - f(pts))) < 1e-11

    @pytest.mark.parametrize("lam", [0, 1, 2, 4])
    def test_jacobi_operator_multiplies_by_x(self, lam):
        n = 36
        f = lambda s: np.sin(2 * s) + 0.3 * s**2  # noqa: E731
        a = chebyshev_coefficients(f(cgl(n)))
        u = np.asarray(conversion_chain(n, 0, lam) @ a)
        xu = np.asarray(jacobi_operator(n, lam) @ u)
        pts = np.linspace(-0.9, 0.9, 13)
        target = pts * f(pts)
        got = gegenbauer_series(xu, lam, pts) if lam else chebyshev_values(xu, pts)
        assert np.max(np.abs(got - target)) < 1e-12

    @pytest.mark.parametrize("lam", [0, 2, 4])
    def test_multiplication_operator(self, lam):
        n = 48
        a_fn = lambda s: 2.0 + np.tanh(2 * s)  # noqa: E731
        u_fn = lambda s: np.exp(s)  # noqa: E731
        x = cgl(n)
        m_op = multiplication_operator(chebyshev_coefficients(a_fn(x)), n, lam)
        u = np.asarray(conversion_chain(n, 0, lam) @ chebyshev_coefficients(u_fn(x)))
        prod = np.asarray(m_op @ u)
        pts = np.linspace(-0.85, 0.85, 11)
        got = gegenbauer_series(prod, lam, pts) if lam else chebyshev_values(prod, pts)
        assert np.max(np.abs(got - a_fn(pts) * u_fn(pts))) < 1e-9

    def test_multiplication_bandwidth_tracks_coefficient_decay(self):
        """A narrow tanh blend needs more bandwidth than a wide one —
        the Paper II Appendix A statement about slowly converging fields."""
        n = 96
        x = cgl(n)
        narrow = multiplication_operator(chebyshev_coefficients(np.tanh(4.0 * x)), n, 2)
        wide = multiplication_operator(chebyshev_coefficients(np.tanh(0.5 * x)), n, 2)
        assert narrow.nnz > 2 * wide.nnz

    def test_evaluation_rows(self):
        n = 20
        f = lambda s: s**4 - 2 * s  # noqa: E731
        a = chebyshev_coefficients(f(cgl(n)))
        assert evaluation_row(n, 1, 0) @ a == pytest.approx(f(1.0), abs=1e-13)
        assert evaluation_row(n, -1, 0) @ a == pytest.approx(f(-1.0), abs=1e-13)
        assert evaluation_row(n, 1, 1) @ a == pytest.approx(4 - 2, abs=1e-12)
        # f'' = 12 x^2 is even: f''(-1) = +12
        assert evaluation_row(n, -1, 2) @ a == pytest.approx(12.0, abs=1e-10)

    def test_validation(self):
        with pytest.raises(ValueError):
            diff_operator(10, 0)
        with pytest.raises(ValueError):
            diff_operator(10, 10)
        with pytest.raises(ValueError):
            conversion_operator(10, -1)
        with pytest.raises(ValueError):
            evaluation_row(10, 0, 1)


class TestBVP:
    def test_constant_coefficient_fourth_order_exact(self):
        """u'''' = f, clamped both ends, polynomial solution: exact."""
        length = 2.0
        p = np.poly1d([-1, 6, -12, 8, 0, 0, 0])  # x^3 (2-x)^3 expanded
        p4 = np.polyder(p, 4)
        op = VariableCoefficientOperator(
            [None] * 4 + [lambda s: np.ones_like(s)], 32, (0.0, length)
        )
        bcs = [
            BoundaryCondition(-1, {0: 1.0}),
            BoundaryCondition(-1, {1: 1.0}),
            BoundaryCondition(1, {0: 1.0}),
            BoundaryCondition(1, {1: 1.0}),
        ]
        xs = np.linspace(0.0, length, 21)
        got = UltrasphericalBVP(op, bcs).solve_values(lambda s: p4(s), xs)
        assert np.max(np.abs(got - p(xs))) < 1e-12

    def test_variable_coefficient_manufactured(self):
        """(EI u'')'' with EI = 2 + x^2 and u = sin x: spectral accuracy."""
        length = 2.0
        op = VariableCoefficientOperator(
            [None, None, lambda s: np.full_like(s, 2.0), lambda s: 4.0 * s,
             lambda s: 2.0 + s**2],
            48,
            (0.0, length),
        )
        bcs = [
            BoundaryCondition(-1, {0: 1.0}, 0.0),
            BoundaryCondition(-1, {1: 1.0}, 1.0),
            BoundaryCondition(1, {0: 1.0}, float(np.sin(length))),
            BoundaryCondition(1, {1: 1.0}, float(np.cos(length))),
        ]

        def rhs(s):
            return (2 + s**2) * np.sin(s) - 4 * s * np.cos(s) - 2 * np.sin(s)

        xs = np.linspace(0.0, length, 21)
        got = UltrasphericalBVP(op, bcs).solve_values(rhs, xs)
        assert np.max(np.abs(got - np.sin(xs))) < 1e-13

    def test_second_order_dirichlet(self):
        """u'' - u = -2 sin x, u(±1) = sin(±1): u = sin x."""
        op = VariableCoefficientOperator(
            [lambda s: np.full_like(s, -1.0), None, lambda s: np.ones_like(s)], 24
        )
        bcs = [
            BoundaryCondition(-1, {0: 1.0}, float(np.sin(-1.0))),
            BoundaryCondition(1, {0: 1.0}, float(np.sin(1.0))),
        ]
        xs = np.linspace(-1, 1, 15)
        got = UltrasphericalBVP(op, bcs).solve_values(lambda s: -2.0 * np.sin(s), xs)
        assert np.max(np.abs(got - np.sin(xs))) < 1e-14

    def test_interior_conditioning_o1_preconditioned_on_beam_operator(self):
        """The Olver–Townsend claim: the preconditioned banded interior is
        O(1)-conditioned; raw interior grows only O(N)."""
        kappas_pre, kappas_raw = [], []
        for n in (32, 64, 128):
            op = VariableCoefficientOperator(
                [None, None, lambda s: np.full_like(s, 2.0), lambda s: 4.0 * s,
                 lambda s: 2.0 + s**2],
                n,
                (0.0, 2.0),
            )
            kappas_pre.append(op.interior_condition_number(preconditioned=True))
            kappas_raw.append(op.interior_condition_number(preconditioned=False))
        assert max(kappas_pre) < 10.0, f"preconditioned interior not O(1): {kappas_pre}"
        raw_slope = np.polyfit(np.log([32, 64, 128]), np.log(kappas_raw), 1)[0]
        assert raw_slope < 1.3, f"raw interior slope {raw_slope}, expected ~1 (O(N))"

    def test_free_free_beam_eigenvalues_match_analytic(self):
        """Cross-validation of the two spectral methods: ultraspherical
        free-free beam frequencies must match Paper I's analytic values.

        Homogeneous boundary rows are normalized to unit norm before the
        QZ solve: raw d = 3 evaluation rows carry m^6-scale entries that
        wreck QZ balancing against the O(m) banded interior."""
        n = 32
        length, ei, mass = 1.0, 1.0, 1.0
        op = VariableCoefficientOperator(
            [None] * 4 + [lambda s: np.full_like(s, ei)], n, (0.0, length)
        )
        scale = 2.0 / length
        rows = []
        for e in (-1, 1):
            for d in (2, 3):  # zero moment and zero shear (constant EI)
                r = evaluation_row(n, e, d) * scale**d
                rows.append(r / np.linalg.norm(r))
        k_mat = np.vstack([*rows, op.matrix.toarray()[: n - 4, :]])
        mass_conv = conversion_chain(n, 0, 4).toarray() * mass
        m_mat = np.vstack([np.zeros((4, n)), mass_conv[: n - 4, :]])
        lam = scipy.linalg.eig(k_mat, m_mat, right=False)
        lam = np.sort(lam[np.isfinite(lam)].real)
        lam = lam[lam > 1e-3]  # drop the two rigid modes
        analytic = free_free_analytic_frequencies(5, length, ei, mass) ** 2
        rel = np.abs(lam[:5] - analytic) / analytic
        assert np.max(rel) < 1e-6, f"ultraspherical beam eigenvalues off: {rel}"

    def test_matches_collocation_kernel_frequencies(self):
        """Ultraspherical and Paper-I collocation agree on a variable-EI
        beam where no analytic reference exists."""
        n_coll = 40
        length = 1.0
        ei_fn = lambda s: 1.0 + 0.5 * s  # noqa: E731
        grid = ChebyshevGrid(n_coll, interval=(0.0, length))
        from aether.structures.profiles import MaterialProfile

        profile = MaterialProfile(
            ei=ei_fn,
            mass=lambda s: np.ones_like(s),
            d_ei=lambda s: np.full_like(s, 0.5),
            d2_ei=lambda s: np.zeros_like(s),
            label="linear-EI",
        )
        coll = solve_free_free_modes(project_free_free(assemble_beam(grid, profile)))
        f_coll = coll.elastic_frequencies[:3]

        n = 32
        op = VariableCoefficientOperator(
            [None, None, lambda s: np.zeros_like(s), lambda s: np.ones_like(s), ei_fn],
            n,
            (0.0, length),
        )
        scale = 2.0 / length
        rows = []
        for e in (-1, 1):
            ei_end = float(ei_fn(np.array([0.0 if e == -1 else length]))[0])
            row_m = ei_end * evaluation_row(n, e, 2) * scale**2
            row_s = (
                ei_end * evaluation_row(n, e, 3) * scale**3
                + 0.5 * evaluation_row(n, e, 2) * scale**2
            )
            rows.append(row_m / np.linalg.norm(row_m))
            rows.append(row_s / np.linalg.norm(row_s))
        k_mat = np.vstack([*rows, op.matrix.toarray()[: n - 4, :]])
        m_mat = np.vstack(
            [np.zeros((4, n)), conversion_chain(n, 0, 4).toarray()[: n - 4, :]]
        )
        lam = scipy.linalg.eig(k_mat, m_mat, right=False)
        lam = np.sort(lam[np.isfinite(lam)].real)
        lam = lam[lam > 1e-3]
        f_ultra = np.sqrt(lam[:3])
        assert np.allclose(f_ultra, f_coll, rtol=1e-6), (f_ultra, f_coll)

    def test_free_free_bordered_system_is_singular(self):
        """Free-free BCs leave the rigid-body null space: the bordered BVP
        is genuinely ill-posed and must say so, not return garbage."""
        op = VariableCoefficientOperator(
            [None] * 4 + [lambda s: np.ones_like(s)], 24, (0.0, 1.0)
        )
        bcs = [
            BoundaryCondition(e, {d: 1.0}) for e in (-1, 1) for d in (2, 3)
        ]
        with pytest.raises(np.linalg.LinAlgError, match="rigid-body"):
            UltrasphericalBVP(op, bcs)

    def test_bvp_validation(self):
        op = VariableCoefficientOperator([None, lambda s: np.ones_like(s)], 16)
        with pytest.raises(ValueError, match="exactly 1"):
            UltrasphericalBVP(op, [])
        with pytest.raises(ValueError, match="leading-order"):
            VariableCoefficientOperator([lambda s: s, None], 16)
        with pytest.raises(ValueError, match="finite"):
            bvp = UltrasphericalBVP(op, [BoundaryCondition(-1, {0: 1.0})])
            bvp.solve(lambda s: np.full_like(s, np.nan))
