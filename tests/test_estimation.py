"""Adaptive filter (Paper I, §4.1–4.2): gate calibration, inflation bounds."""

import numpy as np
import pytest
import scipy.stats

from aether.estimation import (
    AdaptiveConfig,
    AdaptiveKalmanFilter,
    LinearModel,
    chi_square_gate,
    inflation_factor,
)

DT = 0.05


def make_cv_model(q_psd=1.0, sigma_z=2.0):
    """1-D constant-velocity tracker: state [pos, vel], position measured."""
    f = np.array([[1.0, DT], [0.0, 1.0]])
    h = np.array([[1.0, 0.0]])
    q = q_psd * np.array(
        [[DT**3 / 3.0, DT**2 / 2.0], [DT**2 / 2.0, DT]]
    )
    r = np.array([[sigma_z**2]])
    return LinearModel(f, h, q, r)


def simulate_truth(model, n_steps, n_batch, rng, jump_step=None, jump_dv=0.0):
    """Truth trajectories and measurements consistent with the model."""
    n = model.state_dim
    x = np.zeros((n_batch, n))
    chol_q = np.linalg.cholesky(model.process_noise + 1e-15 * np.eye(n))
    sigma_z = np.sqrt(model.measurement_noise[0, 0])
    truths, meas = [], []
    for k in range(n_steps):
        x = x @ model.transition.T + rng.standard_normal((n_batch, n)) @ chol_q.T
        if jump_step is not None and k == jump_step:
            x[:, 1] += jump_dv  # unmodeled separation transient
        z = x @ model.observation.T + sigma_z * rng.standard_normal((n_batch, 1))
        truths.append(x.copy())
        meas.append(z)
    return truths, meas


class TestGateAndInflation:
    def test_gate_threshold_matches_chi2(self):
        assert chi_square_gate(1, 1e-3) == pytest.approx(scipy.stats.chi2.ppf(0.999, 1))
        assert chi_square_gate(3, 0.01) == pytest.approx(scipy.stats.chi2.ppf(0.99, 3))
        with pytest.raises(ValueError):
            chi_square_gate(0, 1e-3)
        with pytest.raises(ValueError):
            chi_square_gate(1, 1.5)

    def test_inflation_bounds_property(self):
        """Prop. 5: alpha in [1, alpha_max] for arbitrary inputs."""
        rng = np.random.default_rng(0)
        c_tr = rng.uniform(-100.0, 1e6, 10_000)  # includes covariance deficits
        p_tr = rng.uniform(0.0, 1e3, 10_000)
        alpha = inflation_factor(c_tr, 5.0, p_tr, alpha_max=50.0, epsilon=1e-12)
        assert np.all(alpha >= 1.0)
        assert np.all(alpha <= 50.0)

    def test_inflation_identity_when_consistent(self):
        """Quiet innovations (C_hat <= R) must give exactly alpha = 1."""
        alpha = inflation_factor([3.0], 5.0, [10.0], alpha_max=100.0, epsilon=1e-12)
        assert alpha[0] == 1.0

    def test_inflation_scales_with_excess(self):
        alpha = inflation_factor([25.0], 5.0, [2.0], alpha_max=1e6, epsilon=1e-12)
        assert alpha[0] == pytest.approx(10.0, rel=1e-9)


class TestConfigValidation:
    def test_rejects_bad_parameters(self):
        with pytest.raises(ValueError, match="false_alarm"):
            AdaptiveConfig(false_alarm_probability=0.0)
        with pytest.raises(ValueError, match="window_length"):
            AdaptiveConfig(window_length=1)
        with pytest.raises(ValueError, match="alpha_max"):
            AdaptiveConfig(alpha_max=0.5)
        with pytest.raises(ValueError, match="epsilon"):
            AdaptiveConfig(epsilon=0.0)

    def test_model_validation(self):
        f = np.eye(2)
        h = np.array([[1.0, 0.0]])
        with pytest.raises(ValueError, match="symmetric"):
            LinearModel(f, h, np.array([[1.0, 0.5], [0.0, 1.0]]), np.eye(1))
        with pytest.raises(ValueError, match="positive definite"):
            LinearModel(f, h, np.eye(2), np.zeros((1, 1)))
        with pytest.raises(ValueError, match="observation"):
            LinearModel(f, np.ones((1, 3)), np.eye(2), np.eye(1))


class TestAdaptiveKalmanFilter:
    def test_nominal_false_alarm_rate_matches_design(self):
        """Under the nominal hypothesis the gate must fire at ~p."""
        model = make_cv_model()
        p = 1e-2  # larger p so the rate estimate is well resolved
        filt = AdaptiveKalmanFilter(model, AdaptiveConfig(false_alarm_probability=p))
        n_batch, n_steps, warmup = 200, 400, 50
        filt.reset(np.zeros(2), np.diag([25.0, 4.0]), n_batch)
        rng = np.random.default_rng(7)
        _, meas = simulate_truth(model, n_steps, n_batch, rng)
        fires = 0
        total = 0
        for k, z in enumerate(meas):
            diag = filt.step(z)
            if k >= warmup:
                fires += int(diag.gate_triggered.sum())
                total += n_batch
        rate = fires / total
        # binomial 4-sigma band around p with total samples
        band = 4.0 * np.sqrt(p * (1.0 - p) / total)
        assert abs(rate - p) < band, f"false alarm rate {rate:.4f} vs design {p}"

    def test_nis_is_chi_squared_under_nominal(self):
        """Mean NIS ~ m = 1 in steady state (filter consistency)."""
        model = make_cv_model()
        filt = AdaptiveKalmanFilter(model, AdaptiveConfig())
        n_batch, n_steps = 100, 300
        filt.reset(np.zeros(2), np.diag([25.0, 4.0]), n_batch)
        rng = np.random.default_rng(3)
        _, meas = simulate_truth(model, n_steps, n_batch, rng)
        nis_tail = []
        for k, z in enumerate(meas):
            diag = filt.step(z)
            if k >= 100:
                nis_tail.append(diag.nis)
        mean_nis = float(np.mean(nis_tail))
        assert 0.9 < mean_nis < 1.1, f"mean NIS {mean_nis}, expected ~1"

    def test_transient_triggers_inflation_and_recovery(self):
        """A velocity jump must trip the gate, drive alpha > 1, and the
        NIS must return below the gate afterward — with no divergence."""
        model = make_cv_model()
        cfg = AdaptiveConfig(false_alarm_probability=1e-3, window_length=20, alpha_max=500.0)
        filt = AdaptiveKalmanFilter(model, cfg)
        n_batch, n_steps, jump_step = 50, 500, 200
        filt.reset(np.zeros(2), np.diag([25.0, 4.0]), n_batch)
        rng = np.random.default_rng(11)
        _, meas = simulate_truth(model, n_steps, n_batch, rng, jump_step, jump_dv=30.0)
        alpha_seen = np.zeros(n_batch)
        fired_at_jump = np.zeros(n_batch, dtype=bool)
        late_nis = []
        for k, z in enumerate(meas):
            diag = filt.step(z)
            # the innovation accumulates ~1.5 m/step against a ~2.2 m sigma,
            # so the gate trips a handful of steps after the jump
            if jump_step <= k < jump_step + 20:
                fired_at_jump |= diag.gate_triggered
                alpha_seen = np.maximum(alpha_seen, diag.alpha)
            if k >= n_steps - 100:
                late_nis.append(diag.nis)
        assert np.all(fired_at_jump), "every replicate must detect a 30 m/s jump"
        assert np.all(alpha_seen > 1.0), "detection must inflate Q"
        assert float(np.mean(late_nis)) < 2.0, "filter must re-converge after transient"
        assert not np.any(filt.diverged)

    def test_adaptive_recovers_faster_than_fixed(self):
        """The point of IAE: recovery time strictly shorter than alpha_max=1."""
        model = make_cv_model()
        recovery = {}
        for label, a_max in (("adaptive", 500.0), ("fixed", 1.0)):
            cfg = AdaptiveConfig(false_alarm_probability=1e-3, alpha_max=a_max)
            filt = AdaptiveKalmanFilter(model, cfg)
            n_batch, n_steps, jump_step = 100, 800, 200
            filt.reset(np.zeros(2), np.diag([25.0, 4.0]), n_batch)
            rng = np.random.default_rng(13)  # same seed: identical measurements
            _, meas = simulate_truth(model, n_steps, n_batch, rng, jump_step, jump_dv=30.0)
            recovered = np.full(n_batch, n_steps - jump_step, dtype=float)
            below = np.zeros(n_batch, dtype=int)
            for k, z in enumerate(meas):
                diag = filt.step(z)
                if k > jump_step:
                    below = np.where(diag.nis < filt.gate_threshold, below + 1, 0)
                    just = (below == 10) & (recovered == n_steps - jump_step)
                    recovered[just] = k - jump_step
            recovery[label] = float(np.median(recovered))
        assert recovery["adaptive"] < recovery["fixed"], recovery

    def test_alpha_disabled_is_exact_kalman(self):
        """alpha_max = 1 must reproduce the plain Kalman filter bitwise."""
        model = make_cv_model()
        filt_a = AdaptiveKalmanFilter(model, AdaptiveConfig(alpha_max=1.0))
        filt_b = AdaptiveKalmanFilter(model, AdaptiveConfig(alpha_max=1.0))
        for filt in (filt_a, filt_b):
            filt.reset(np.zeros(2), np.diag([25.0, 4.0]), 4)
        rng = np.random.default_rng(5)
        _, meas = simulate_truth(model, 50, 4, rng)
        for z in meas:
            da = filt_a.step(z)
            db = filt_b.step(z)
            assert np.array_equal(da.nis, db.nis)
        assert np.array_equal(filt_a.state, filt_b.state)

    def test_covariance_stays_symmetric_psd(self):
        model = make_cv_model()
        filt = AdaptiveKalmanFilter(model, AdaptiveConfig())
        filt.reset(np.zeros(2), np.diag([25.0, 4.0]), 8)
        rng = np.random.default_rng(9)
        _, meas = simulate_truth(model, 200, 8, rng)
        for z in meas:
            filt.step(z)
        p = filt.covariance
        assert np.allclose(p, np.swapaxes(p, 1, 2), atol=1e-14)
        assert np.all(np.linalg.eigvalsh(p) > 0.0)

    def test_batched_matches_sequential(self):
        model = make_cv_model()
        rng = np.random.default_rng(21)
        _, meas = simulate_truth(model, 100, 3, rng)
        batch = AdaptiveKalmanFilter(model, AdaptiveConfig())
        batch.reset(np.zeros(2), np.diag([25.0, 4.0]), 3)
        for z in meas:
            batch.step(z)
        for j in range(3):
            single = AdaptiveKalmanFilter(model, AdaptiveConfig())
            single.reset(np.zeros(2), np.diag([25.0, 4.0]), 1)
            for z in meas:
                single.step(z[j : j + 1])
            assert np.allclose(single.state[0], batch.state[j], rtol=1e-12, atol=1e-12)

    def test_usage_errors(self):
        model = make_cv_model()
        filt = AdaptiveKalmanFilter(model, AdaptiveConfig())
        with pytest.raises(RuntimeError, match="reset"):
            filt.step(np.zeros((1, 1)))
        filt.reset(np.zeros(2), np.eye(2), 2)
        with pytest.raises(ValueError, match="measurements"):
            filt.step(np.zeros((3, 1)))
        with pytest.raises(ValueError, match="initial_state"):
            filt.reset(np.zeros(3), np.eye(2), 2)
