"""Spectral primitives: exactness, conditioning-aware accuracy, invariants."""

import numpy as np
import pytest

from aether.spectral import (
    ChebyshevGrid,
    barycentric_interpolate,
    barycentric_weights,
    chebyshev_diffmats,
    clenshaw_curtis_weights,
    gauss_lobatto_nodes,
)

EPS = np.finfo(np.float64).eps


class TestNodes:
    @pytest.mark.parametrize("n", [1, 2, 5, 8, 16, 33, 64])
    def test_matches_cosine_definition(self, n):
        nodes = gauss_lobatto_nodes(n)
        assert np.allclose(nodes, np.cos(np.arange(n + 1) * np.pi / n), atol=4 * EPS)

    @pytest.mark.parametrize("n", [2, 5, 8, 16, 33])
    def test_exact_antisymmetry(self, n):
        nodes = gauss_lobatto_nodes(n)
        assert np.all(nodes == -nodes[::-1]), "sine construction must be antisymmetric to the bit"

    def test_endpoints_exact(self):
        nodes = gauss_lobatto_nodes(17)
        assert nodes[0] == 1.0
        assert nodes[-1] == -1.0

    def test_descending(self):
        assert np.all(np.diff(gauss_lobatto_nodes(12)) < 0)

    @pytest.mark.parametrize("bad", [0, -3])
    def test_rejects_bad_order(self, bad):
        with pytest.raises(ValueError):
            gauss_lobatto_nodes(bad)

    def test_rejects_non_integer(self):
        with pytest.raises(TypeError):
            gauss_lobatto_nodes(8.0)


class TestDifferentiationMatrices:
    @pytest.mark.parametrize("n", [4, 8, 16, 32])
    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    def test_polynomial_exactness(self, n, k):
        """D^(k) must differentiate every monomial of degree <= N to within
        the conditioning floor eps * ||D^(k)|| * ||x^p||."""
        dm = chebyshev_diffmats(n, 4)
        xi = gauss_lobatto_nodes(n)
        floor = 50 * EPS * np.linalg.norm(dm[k - 1], np.inf)
        for p in range(0, n + 1):
            coeffs = np.zeros(p + 1)
            coeffs[0] = 1.0
            exact = np.polyval(np.polyder(coeffs, k), xi) if p >= k else np.zeros(n + 1)
            err = np.max(np.abs(dm[k - 1] @ xi**p - exact))
            assert err <= floor * max(1.0, np.max(np.abs(xi**p))), (k, p, err, floor)

    @pytest.mark.parametrize("n", [4, 7, 12, 25])
    def test_matches_matrix_powers_at_small_n(self, n):
        """Direct recurrence and matrix powers agree to the rounding floor."""
        dm = chebyshev_diffmats(n, 4)
        d1 = dm[0]
        for k in (2, 3, 4):
            power = np.linalg.matrix_power(d1, k)
            scale = np.linalg.norm(power, np.inf)
            assert np.max(np.abs(dm[k - 1] - power)) <= 1e3 * EPS * scale

    @pytest.mark.parametrize("n", [5, 16, 33])
    def test_negative_sum_rows(self, n):
        """Constants are annihilated to the rounding floor of each row."""
        dm = chebyshev_diffmats(n, 4)
        for k in range(4):
            row_scale = np.max(np.abs(dm[k]), axis=1)
            assert np.all(np.abs(dm[k].sum(axis=1)) <= 20 * EPS * row_scale)

    def test_first_matrix_matches_trefethen_formula(self):
        """Spot-check the closed-form entries of Paper I, Eq. (A.1)."""
        n = 10
        d = chebyshev_diffmats(n, 1)[0]
        assert d[0, 0] == pytest.approx((2 * n**2 + 1) / 6.0, rel=1e-13)
        assert d[n, n] == pytest.approx(-(2 * n**2 + 1) / 6.0, rel=1e-13)
        xi = np.cos(np.arange(n + 1) * np.pi / n)
        c = np.where((np.arange(n + 1) == 0) | (np.arange(n + 1) == n), 2.0, 1.0)
        i, j = 3, 7
        expected = (c[i] / c[j]) * (-1.0) ** (i + j) / (xi[i] - xi[j])
        assert d[i, j] == pytest.approx(expected, rel=1e-12)

    def test_rejects_order_above_n(self):
        with pytest.raises(ValueError):
            chebyshev_diffmats(3, 4)

    def test_read_only(self):
        dm = chebyshev_diffmats(8, 2)
        with pytest.raises(ValueError):
            dm[0][0, 0] = 1.0


class TestClenshawCurtis:
    @pytest.mark.parametrize("n", [1, 2, 3, 8, 15, 16, 33, 64])
    def test_polynomial_exactness(self, n):
        """CC weights integrate every monomial of degree <= N exactly."""
        w = clenshaw_curtis_weights(n)
        xi = gauss_lobatto_nodes(n)
        for p in range(n + 1):
            exact = 0.0 if p % 2 else 2.0 / (p + 1)
            assert w @ xi**p == pytest.approx(exact, abs=200 * EPS)

    @pytest.mark.parametrize("n", [2, 9, 32])
    def test_positive_and_symmetric(self, n):
        w = clenshaw_curtis_weights(n)
        assert np.all(w > 0)
        assert np.allclose(w, w[::-1], rtol=0, atol=4 * EPS)

    def test_smooth_nonpolynomial_convergence(self):
        exact = np.exp(1.0) - np.exp(-1.0)
        w = clenshaw_curtis_weights(20)
        xi = gauss_lobatto_nodes(20)
        assert w @ np.exp(xi) == pytest.approx(exact, rel=1e-14)


class TestBarycentric:
    def test_reproduces_polynomials(self):
        n = 14
        xi = gauss_lobatto_nodes(n)
        f = 3 * xi**7 - xi**2 + 0.5
        x_eval = np.linspace(-1, 1, 41)
        got = barycentric_interpolate(xi, f, x_eval)
        assert np.allclose(got, 3 * x_eval**7 - x_eval**2 + 0.5, atol=1e-12)

    def test_exact_at_nodes(self):
        n = 9
        xi = gauss_lobatto_nodes(n)
        f = np.sin(xi)
        got = barycentric_interpolate(xi, f, xi)
        assert np.all(got == f), "node coincidence must return nodal values exactly"

    def test_stacked_fields_and_shape(self):
        n = 8
        xi = gauss_lobatto_nodes(n)
        vals = np.column_stack([xi**2, xi**3])
        x_eval = np.array([[0.1, -0.4], [0.9, 0.0]])
        got = barycentric_interpolate(xi, vals, x_eval)
        assert got.shape == (2, 2, 2)
        assert got[0, 0, 0] == pytest.approx(0.01, abs=1e-13)
        assert got[0, 1, 1] == pytest.approx(-0.064, abs=1e-13)

    def test_weights_shape_mismatch_raises(self):
        xi = gauss_lobatto_nodes(4)
        with pytest.raises(ValueError):
            barycentric_interpolate(xi, xi, 0.0, weights=np.ones(3))

    def test_explicit_cgl_weight_pattern(self):
        lam = barycentric_weights(5)
        assert lam[0] == 0.5 and lam[-1] == -0.5
        assert np.all(np.abs(lam[1:-1]) == 1.0)


class TestChebyshevGrid:
    def test_mapped_operators(self):
        length = 2.5
        g = ChebyshevGrid(16, interval=(0.0, length))
        x = g.x
        assert x[0] == pytest.approx(length) and x[-1] == pytest.approx(0.0)
        assert np.allclose(g.diffmat(1) @ x**3, 3 * x**2, atol=1e-10)
        assert np.allclose(g.diffmat(4) @ x**5, 120 * x, atol=1e-6)
        assert g.weights.sum() == pytest.approx(length, rel=1e-14)

    def test_scaling_factor_carried(self):
        """The (2/L)^k factor of Paper I Eq. (3.3): physical D^k equals
        (2/L)^k times the reference operator."""
        g = ChebyshevGrid(10, interval=(0.0, 4.0))
        ref = chebyshev_diffmats(10, 4)
        for k in range(1, 5):
            assert np.allclose(g.diffmat(k), (2.0 / 4.0) ** k * ref[k - 1], rtol=1e-15)

    def test_invalid_interval(self):
        with pytest.raises(ValueError):
            ChebyshevGrid(8, interval=(1.0, 1.0))
        with pytest.raises(ValueError):
            ChebyshevGrid(8, interval=(2.0, -1.0))
        with pytest.raises(ValueError):
            ChebyshevGrid(8, interval=(0.0, np.inf))

    def test_diffmat_order_bounds(self):
        g = ChebyshevGrid(8, max_derivative=3)
        with pytest.raises(ValueError):
            g.diffmat(4)
        with pytest.raises(ValueError):
            g.diffmat(0)

    def test_arrays_read_only(self):
        g = ChebyshevGrid(8)
        for arr in (g.x, g.xi, g.weights, g.diffmat(1)):
            with pytest.raises(ValueError):
                arr[0] = 0.0
