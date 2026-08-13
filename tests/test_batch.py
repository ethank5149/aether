"""Batch layer: sampling, propagation, and the batched entry model."""

import numpy as np
import pytest

from aether.batch import (
    DispersionSpec,
    EntryDispersionModel,
    cuda_available,
    rk4_batch,
    sample_dispersions,
)


class TestSampling:
    def test_deterministic_given_seed(self):
        specs = [DispersionSpec("a", 1.0, 0.1), DispersionSpec("b", 5.0, 0.5)]
        s1 = sample_dispersions(specs, 100, seed=42)
        s2 = sample_dispersions(specs, 100, seed=42)
        s3 = sample_dispersions(specs, 100, seed=43)
        assert np.array_equal(s1["a"], s2["a"]) and np.array_equal(s1["b"], s2["b"])
        assert not np.array_equal(s1["a"], s3["a"])

    def test_appending_spec_preserves_earlier_draws(self):
        base = [DispersionSpec("a", 1.0, 0.1)]
        extended = [*base, DispersionSpec("c", 0.0, 1.0)]
        assert np.array_equal(
            sample_dispersions(base, 50, 7)["a"], sample_dispersions(extended, 50, 7)["a"]
        )

    def test_zero_sigma_pins_nominal(self):
        out = sample_dispersions([DispersionSpec("a", 3.5, 0.0)], 20, 0)
        assert np.all(out["a"] == 3.5)

    def test_truncation_by_rejection(self):
        spec = DispersionSpec("a", 0.0, 1.0, lower=-1.0, upper=1.0)
        draws = sample_dispersions([spec], 5000, 1)["a"]
        assert np.all((draws >= -1.0) & (draws <= 1.0))
        # rejection, not clipping: no mass piled on the bounds
        assert np.sum(np.isclose(draws, 1.0, atol=1e-6)) < 5

    def test_validation(self):
        with pytest.raises(ValueError, match="sigma"):
            DispersionSpec("a", 0.0, -1.0)
        with pytest.raises(ValueError, match="lower < upper"):
            DispersionSpec("a", 0.0, 1.0, lower=2.0, upper=1.0)
        with pytest.raises(ValueError, match="duplicate"):
            sample_dispersions(
                [DispersionSpec("a", 0.0, 1.0), DispersionSpec("a", 1.0, 1.0)], 10, 0
            )


class TestRk4Batch:
    def test_fourth_order_convergence(self):
        """dy/dt = -lambda y with per-replicate lambda: global error O(dt^4)."""
        lam = np.array([0.5, 1.0, 2.0])
        y0 = np.ones((3, 1))

        def rhs(_t, y, _xp):
            return -lam[:, None] * y

        errs = []
        for n_steps in (20, 40, 80):
            y = rk4_batch(rhs, y0, 0.0, 1.0, n_steps)
            errs.append(np.max(np.abs(y[:, 0] - np.exp(-lam))))
        rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
        assert np.all(rates > 3.7), f"rates {rates}, expected ~4"

    def test_batch_matches_per_replicate(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0.5, 2.0, 5)

        def rhs(t, y, _xp):
            return a[:, None] * np.cos(t) * y

        y0 = rng.uniform(0.5, 1.5, (5, 1))
        batch = rk4_batch(rhs, y0, 0.0, 2.0, 100)
        for i in range(5):
            def rhs_i(t, y, xp, i=i):
                return a[i] * np.cos(t) * y

            single = rk4_batch(rhs_i, y0[i : i + 1], 0.0, 2.0, 100)
            assert np.allclose(batch[i], single[0], rtol=1e-14)

    @pytest.mark.skipif(not cuda_available(), reason="no CUDA device")
    def test_cupy_backend_matches_numpy(self):
        lam = np.array([0.5, 1.0, 2.0, 4.0])

        def rhs(_t, y, xp):
            return -xp.asarray(lam)[:, None] * y

        y0 = np.ones((4, 2))
        y_cpu = rk4_batch(rhs, y0, 0.0, 1.0, 50, backend="numpy")
        y_gpu = rk4_batch(rhs, y0, 0.0, 1.0, 50, backend="cupy")
        from aether.batch import to_numpy

        assert np.allclose(y_cpu, to_numpy(y_gpu), rtol=1e-13)

    def test_validation(self):
        def rhs(_t, y, _xp):
            return y

        with pytest.raises(ValueError, match="n_steps"):
            rk4_batch(rhs, np.ones((2, 1)), 0.0, 1.0, 0)
        with pytest.raises(ValueError, match="t_end"):
            rk4_batch(rhs, np.ones((2, 1)), 1.0, 0.0, 10)
        with pytest.raises(ValueError, match="shape"):
            rk4_batch(rhs, np.ones(3), 0.0, 1.0, 10)


class TestEntryModel:
    def test_impacts_finite_and_downrange(self):
        model = EntryDispersionModel()
        pts = model.fly(200, seed=1)
        assert pts.shape == (200, 2)
        assert np.all(np.isfinite(pts))
        assert np.all(pts[:, 0] > 0.0), "impacts must be downrange of release"

    def test_zero_dispersion_collapses_footprint(self):
        model = EntryDispersionModel(
            beta_rel_sigma=0.0,
            speed_rel_sigma=0.0,
            flight_path_sigma_deg=0.0,
            azimuth_sigma_deg=0.0,
            density_bias_rel_sigma=0.0,
            wind_sigma=0.0,
        )
        pts = model.fly(20, seed=5)
        assert np.max(np.abs(pts - pts[0])) < 1e-9

    def test_reproducible(self):
        model = EntryDispersionModel()
        assert np.array_equal(model.fly(50, seed=9), model.fly(50, seed=9))

    def test_heavier_ballistic_coefficient_flies_farther(self):
        light = EntryDispersionModel(beta_nominal=3000.0)
        heavy = EntryDispersionModel(beta_nominal=20000.0)
        x_light = np.mean(light.fly(100, seed=2)[:, 0])
        x_heavy = np.mean(heavy.fly(100, seed=2)[:, 0])
        assert x_heavy > x_light

    @pytest.mark.skipif(not cuda_available(), reason="no CUDA device")
    def test_gpu_matches_cpu(self):
        model = EntryDispersionModel()
        cpu = model.fly(64, seed=3, backend="numpy")
        gpu = model.fly(64, seed=3, backend="cupy")
        assert np.allclose(cpu, gpu, rtol=1e-10, atol=1e-6)


