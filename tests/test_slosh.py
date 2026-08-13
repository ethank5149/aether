"""Slosh regularization (Paper I, §3.3): bandwidth, normalization, transfer."""

import numpy as np
import pytest

from aether.coupling import (
    SloshCoupling,
    kernel_bandwidth,
    local_node_spacing,
    normalized_kernel,
)
from aether.spectral import ChebyshevGrid

L = 10.0


@pytest.fixture(scope="module")
def grid():
    return ChebyshevGrid(64, interval=(0.0, L))


class TestBandwidth:
    def test_spacing_matches_gap_scale(self, grid):
        """The h(x) estimate tracks the actual local node gap within a
        factor of ~2 across the domain."""
        x = grid.x
        gaps = np.abs(np.diff(x))
        mid = 0.5 * (x[:-1] + x[1:])
        h = local_node_spacing(grid, mid)
        ratio = h / gaps
        assert np.all(ratio > 0.4) and np.all(ratio < 2.5)

    def test_spacing_floors_at_endpoints(self, grid):
        h_end = local_node_spacing(grid, np.array([0.0, L]))
        end_gap = (L / 2.0) * 2.0 * np.sin(np.pi / (2 * grid.n)) ** 2
        assert np.allclose(h_end, end_gap, rtol=1e-12)
        assert np.all(h_end > 0)

    def test_spacing_maximal_at_center(self, grid):
        h = local_node_spacing(grid, np.array([0.1 * L, 0.5 * L, 0.9 * L]))
        assert h[1] > h[0] and h[1] > h[2]

    def test_station_outside_domain_raises(self, grid):
        with pytest.raises(ValueError, match="outside"):
            local_node_spacing(grid, -0.1)

    def test_gamma_band_enforced(self, grid):
        for bad in (0.5, 0.99, 2.01, 5.0, np.nan):
            with pytest.raises(ValueError, match="gamma"):
                kernel_bandwidth(grid, 0.5 * L, gamma=bad)
        sigma = kernel_bandwidth(grid, 0.5 * L, gamma=1.0)
        assert sigma == pytest.approx(local_node_spacing(grid, 0.5 * L))


class TestNormalizedKernel:
    @pytest.mark.parametrize("station", [0.05 * L, 0.31 * L, 0.5 * L, 0.87 * L])
    @pytest.mark.parametrize("sigma", [0.05, 0.3, 1.0])
    def test_unit_discrete_integral(self, grid, station, sigma):
        d = normalized_kernel(grid, station, sigma)
        assert float(grid.weights @ d) == pytest.approx(1.0, abs=5e-15)
        assert np.all(d >= 0.0)

    def test_peak_near_station(self, grid):
        d = normalized_kernel(grid, 0.42 * L, 0.3)
        assert abs(grid.x[np.argmax(d)] - 0.42 * L) < 0.3

    def test_validation(self, grid):
        with pytest.raises(ValueError, match="sigma"):
            normalized_kernel(grid, 0.5 * L, 0.0)
        with pytest.raises(ValueError, match="outside"):
            normalized_kernel(grid, 1.5 * L, 0.1)


class TestSloshCoupling:
    def test_exact_force_transfer_sweep(self):
        """Prop. 1: total force error at rounding level, independent of
        N, gamma, and station — including near the domain ends."""
        rng = np.random.default_rng(4)
        worst = 0.0
        for n in (16, 24, 32, 64):
            g = ChebyshevGrid(n, interval=(0.0, L))
            for gamma in (1.0, 1.5, 2.0):
                stations = np.array([0.02, 0.11, 0.35, 0.5, 0.77, 0.93]) * L
                coupling = SloshCoupling(g, stations, gamma=gamma)
                forces = rng.uniform(-1e4, 1e4, stations.size)
                q = coupling.load(forces)
                err = abs(float(coupling.transferred_force(q)) - forces.sum())
                worst = max(worst, err / np.sum(np.abs(forces)))
        assert worst < 5e-15, f"force transfer error {worst:.2e} above machine precision"

    def test_moment_near_machine_for_resolved_interior_kernel(self, grid):
        c = SloshCoupling(grid, [0.43 * L], gamma=1.6)
        # widen sigma well into the resolved regime by using the kernel directly
        d = normalized_kernel(grid, 0.43 * L, 0.4)
        moment = float((grid.weights * (grid.x - 0.5 * L)) @ d)
        assert abs(moment - (0.43 * L - 0.5 * L)) / L < 1e-10
        assert c.n_tanks == 1

    def test_moment_bias_grows_near_endpoint(self, grid):
        """Paper I remark: stations within ~2 sigma of an end acquire a
        moment bias absent in the interior."""
        sigma = 0.4
        x_ref = 0.5 * L
        d_int = normalized_kernel(grid, 0.43 * L, sigma)
        d_end = normalized_kernel(grid, 0.03 * L, sigma)  # well inside 2*sigma of x=0
        err_int = abs(float((grid.weights * (grid.x - x_ref)) @ d_int) - (0.43 * L - x_ref))
        err_end = abs(float((grid.weights * (grid.x - x_ref)) @ d_end) - (0.03 * L - x_ref))
        assert err_end > 1e3 * err_int

    def test_endpoint_moment_bias_grows_with_sigma(self, grid):
        x_s, x_ref = 0.3, 0.5 * L
        errs = []
        for sigma in (0.1, 0.2, 0.4, 0.8):
            d = normalized_kernel(grid, x_s, sigma)
            errs.append(abs(float((grid.weights * (grid.x - x_ref)) @ d) - (x_s - x_ref)))
        assert np.all(np.diff(errs) > 0), f"expected monotone growth, got {errs}"

    def test_batched_forces(self, grid):
        c = SloshCoupling(grid, [0.3 * L, 0.7 * L])
        f = np.array([[1.0, 2.0, 0.0], [3.0, -1.0, 5.0]])
        q = c.load(f)
        assert q.shape == (grid.size, 3)
        totals = c.transferred_force(q)
        assert np.allclose(totals, f.sum(axis=0), rtol=1e-14, atol=1e-12)
        for j in range(3):
            assert np.allclose(q[:, j], c.load(f[:, j]), rtol=1e-15)

    def test_station_strictly_interior(self, grid):
        with pytest.raises(ValueError, match="strictly inside"):
            SloshCoupling(grid, [0.0])
        with pytest.raises(ValueError, match="strictly inside"):
            SloshCoupling(grid, [0.5 * L, L])

    def test_with_stations_rebuilds(self, grid):
        c0 = SloshCoupling(grid, [0.6 * L])
        c1 = c0.with_stations([0.4 * L])
        assert c1.stations[0] == pytest.approx(0.4 * L)
        assert not np.allclose(c0.kernels, c1.kernels)

    def test_force_shape_validation(self, grid):
        c = SloshCoupling(grid, [0.3 * L, 0.7 * L])
        with pytest.raises(ValueError, match="forces"):
            c.load(np.ones(3))

    def test_kernels_read_only(self, grid):
        c = SloshCoupling(grid, [0.5 * L])
        with pytest.raises(ValueError):
            c.kernels[0, 0] = 1.0
