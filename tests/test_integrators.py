"""Temporal integration strategies (Paper I, §3.6) — V3 groundwork."""

import numpy as np
import pytest

from aether.spectral import ChebyshevGrid
from aether.structures import (
    ModalPropagator,
    NewmarkIntegrator,
    assemble_beam,
    explicit_dt_limit,
    project_free_free,
    solve_free_free_modes,
    uniform_profile,
)
from aether.structures.integrators import omega_max


@pytest.fixture(scope="module")
def uniform_modal():
    grid = ChebyshevGrid(24, interval=(0.0, 1.0))
    beam = assemble_beam(grid, uniform_profile(1.0, 1.0))
    proj = project_free_free(beam)
    return proj, solve_free_free_modes(proj)


class TestExplicitLimit:
    def test_omega_max_matches_modal_solve(self, uniform_modal):
        proj, sol = uniform_modal
        w = omega_max(proj.reduced_stiffness, proj.reduced_mass)
        assert w == pytest.approx(sol.frequencies[-1], rel=1e-8)

    def test_dt_limit_value_and_validation(self):
        assert explicit_dt_limit(1.0e6) == pytest.approx(3.0e-6)
        with pytest.raises(ValueError):
            explicit_dt_limit(0.0)
        with pytest.raises(ValueError):
            explicit_dt_limit(1.0, c_rk=-1.0)

    def test_omega_max_n4_scaling(self):
        """Prop. 2: omega_max = O(N^4). The fitted log-log slope across an
        octave of N must sit near 4."""
        ns = [16, 20, 24, 28, 32]
        w = []
        for n in ns:
            grid = ChebyshevGrid(n, interval=(0.0, 1.0))
            proj = project_free_free(assemble_beam(grid, uniform_profile(1.0, 1.0)))
            w.append(omega_max(proj.reduced_stiffness, proj.reduced_mass))
        slope = np.polyfit(np.log(ns), np.log(w), 1)[0]
        assert 3.5 < slope < 4.5, f"omega_max slope {slope}, expected ~4"


class TestNewmark:
    def test_sdof_second_order_convergence(self):
        """u'' + w^2 u = 0, u(0)=1: trapezoidal Newmark converges at O(dt^2)."""
        w = 5.0
        k = np.array([[w**2]])
        m = np.eye(1)
        t_end = 2.0
        errs = []
        for dt in (1e-2, 5e-3, 2.5e-3):
            stepper = NewmarkIntegrator(k, m, dt)
            u, v = np.array([1.0]), np.array([0.0])
            a = stepper.initial_acceleration(u, v)
            for _ in range(round(t_end / dt)):
                u, v, a = stepper.step(u, v, a)
            errs.append(abs(u[0] - np.cos(w * t_end)))
        rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
        assert np.all(rates > 1.8), f"convergence rates {rates}, expected ~2"

    def test_per_mode_amplification_bounded(self):
        """|amplification| <= 1 for omega*dt spanning 12 decades — the
        defining property of the unconditionally stable branch of V3."""
        omegas = 10.0 ** np.arange(-3, 7)
        k = np.diag(omegas**2)
        m = np.eye(omegas.size)
        stepper = NewmarkIntegrator(k, m, dt=1.0)  # omega*dt = omega
        u = np.ones(omegas.size)
        v = np.zeros_like(u)
        a = stepper.initial_acceleration(u, v)
        for _ in range(500):
            u, v, a = stepper.step(u, v, a)
        energy = 0.5 * v**2 + 0.5 * omegas**2 * u**2
        # trapezoidal Newmark conserves this energy exactly in exact
        # arithmetic; allow the rounding accumulated over 500 solves at
        # omega*dt up to 1e6 (observed ~4e-6 relative)
        assert np.all(energy <= 0.5 * omegas**2 * (1.0 + 1e-4)), "modal energy grew"

    def test_unconditional_stability_far_beyond_cfl_elastic_beam(self, uniform_modal):
        """Steps 10^4 x the explicit limit on the *elastic* subspace of the
        beam operator must not grow the solution. The full reduced operator
        is excluded deliberately: its two rigid eigenvalues are zero only
        to the rounding floor of a kappa ~ 1/eps operator, and their
        spurious O(1e-7) negative parts grow under any integrator — that
        is an operator property (recorded by V1), not a scheme property."""
        proj, sol = uniform_modal
        q, _ = np.linalg.qr(sol.modes_reduced[:, 2:])
        k_e = q.T @ proj.reduced_stiffness @ q
        m_e = q.T @ proj.reduced_mass @ q
        m_e = 0.5 * (m_e + m_e.T)
        dt = 1e4 * explicit_dt_limit(sol.frequencies[-1])
        stepper = NewmarkIntegrator(k_e, m_e, dt)
        rng = np.random.default_rng(3)
        u = rng.standard_normal(k_e.shape[0]) * 1e-3
        v = np.zeros_like(u)
        a = stepper.initial_acceleration(u, v)
        amp0 = np.linalg.norm(u)
        for _ in range(2000):
            u, v, a = stepper.step(u, v, a)
        assert np.linalg.norm(u) <= 5.0 * amp0, "unconditionally stable scheme grew"

    def test_batched_columns_match_sequential(self, uniform_modal):
        proj, _ = uniform_modal
        k_hat, m_hat = proj.reduced_stiffness, proj.reduced_mass
        stepper = NewmarkIntegrator(k_hat, m_hat, 1e-4)
        rng = np.random.default_rng(11)
        u0 = rng.standard_normal((proj.reduced_dim, 4))
        v0 = np.zeros_like(u0)
        a0 = stepper.initial_acceleration(u0, v0)
        ub, vb, ab = u0, v0, a0
        for _ in range(10):
            ub, vb, ab = stepper.step(ub, vb, ab)
        for col in range(4):
            u, v, a = u0[:, col], v0[:, col], a0[:, col]
            for _ in range(10):
                u, v, a = stepper.step(u, v, a)
            assert np.allclose(u, ub[:, col], rtol=1e-12, atol=1e-14)

    def test_forcing_static_limit(self):
        """Constant force on a stiff SDOF must settle at f/k (mean of the
        undamped oscillation about the static solution equals f/k)."""
        k = np.array([[100.0]])
        stepper = NewmarkIntegrator(k, np.eye(1), 1e-3)
        u, v = np.zeros(1), np.zeros(1)
        f = np.array([50.0])
        a = stepper.initial_acceleration(u, v, f)
        traj = []
        for _ in range(20000):
            u, v, a = stepper.step(u, v, a, f_next=f)
            traj.append(u[0])
        assert np.mean(traj) == pytest.approx(0.5, rel=1e-2)

    def test_parameter_validation(self):
        k, m = np.eye(2), np.eye(2)
        with pytest.raises(ValueError, match="gamma"):
            NewmarkIntegrator(k, m, 0.1, gamma=0.4)
        with pytest.raises(ValueError, match="beta"):
            NewmarkIntegrator(k, m, 0.1, beta=0.1, gamma=0.6)
        with pytest.raises(ValueError):
            NewmarkIntegrator(k, m, 0.0)
        with pytest.raises(ValueError):
            NewmarkIntegrator(k, np.eye(3), 0.1)


class TestModalPropagator:
    def test_exact_oscillator_rotation(self, uniform_modal):
        _, sol = uniform_modal
        basis = sol.truncate(5)
        dt = 0.01
        prop = ModalPropagator(basis, dt)
        q = np.array([0.0, 0.0, 1.0, 0.5, -0.2])
        qd = np.zeros(5)
        steps = 137
        qn, vn = q.copy(), qd.copy()
        for _ in range(steps):
            qn, vn = prop.step(qn, vn)
        t = steps * dt
        for i in range(2, 5):
            w = basis.omega[i]
            assert qn[i] == pytest.approx(q[i] * np.cos(w * t), abs=1e-9)
            assert vn[i] == pytest.approx(-q[i] * w * np.sin(w * t), abs=1e-9 * w)

    def test_rigid_mode_drift(self, uniform_modal):
        _, sol = uniform_modal
        prop = ModalPropagator(sol.truncate(3), 0.5)
        q = np.zeros(3)
        qd = np.array([2.0, -1.0, 0.0])
        qn, vn = prop.step(q, qd)
        assert qn[0] == pytest.approx(1.0)
        assert qn[1] == pytest.approx(-0.5)
        assert np.all(vn[:2] == qd[:2])

    def test_zoh_forcing_particular_solution(self, uniform_modal):
        """Constant modal force f: steady oscillation about f/w^2."""
        _, sol = uniform_modal
        basis = sol.truncate(3)
        w = basis.omega[2]
        prop = ModalPropagator(basis, 1e-3)
        q, qd = np.zeros(3), np.zeros(3)
        f = np.zeros(3)
        f[2] = 4.0
        n = 1000
        for _ in range(n):
            q, qd = prop.step(q, qd, f_modal=f)
        t = n * 1e-3
        expected = (4.0 / w**2) * (1.0 - np.cos(w * t))
        assert q[2] == pytest.approx(expected, abs=1e-9)

    def test_batched(self, uniform_modal):
        _, sol = uniform_modal
        prop = ModalPropagator(sol.truncate(4), 0.02)
        rng = np.random.default_rng(5)
        q = rng.standard_normal((4, 6))
        qd = rng.standard_normal((4, 6))
        qb, vb = prop.step(q, qd)
        for j in range(6):
            qs, vs = prop.step(q[:, j], qd[:, j])
            assert np.allclose(qb[:, j], qs, rtol=1e-14)
            assert np.allclose(vb[:, j], vs, rtol=1e-14)

    def test_shape_validation(self, uniform_modal):
        _, sol = uniform_modal
        prop = ModalPropagator(sol.truncate(4), 0.02)
        with pytest.raises(ValueError):
            prop.step(np.zeros(3), np.zeros(3))
        with pytest.raises(ValueError):
            prop.step(np.zeros(4), np.zeros(4), f_modal=np.zeros(5))
