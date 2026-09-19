"""Thermal kernel (Paper I, §3.4): kinetics, Landau frame, solver, surface."""

import numpy as np
import pytest
import scipy.integrate

from aether.spectral import ChebyshevGrid
from aether.thermal import (
    ArrheniusComponent,
    CharringThermalSolver,
    LandauFrame,
    SurfaceEnergyBalance,
    SurfaceEnvironment,
    SurfaceThermochemistry,
    ThermalState,
    blowing_correction,
    bulk_density,
    decomposition_rate,
    degree_of_char,
    demo_material,
)
from aether.thermal.material import GAS_CONSTANT
from aether.verification.mms_ablation import ManufacturedAblation


@pytest.fixture(scope="module")
def material():
    return demo_material()


class TestKinetics:
    def test_rate_zero_at_and_below_char_density(self, material):
        comp = material.resin_a
        rho = np.array([comp.char_density, comp.char_density - 1.0])
        rate = decomposition_rate(comp, rho, np.full(2, 1500.0))
        assert np.all(rate == 0.0), "char state must be absorbing, no NaN from overshoot"

    def test_rate_negative_above_char(self, material):
        comp = material.resin_a
        rate = decomposition_rate(comp, comp.virgin_density, 1000.0)
        assert rate < 0.0

    def test_first_order_closed_form(self, material):
        """n = 1: rho(t) = rho_c + (rho_0 - rho_c) exp(-k t) isothermally."""
        comp = material.resin_a
        assert comp.reaction_order == 1.0
        temp = 900.0
        k = comp.pre_exponential * np.exp(-comp.activation_energy / (GAS_CONSTANT * temp))
        t_end = 0.5 / k

        sol = scipy.integrate.solve_ivp(
            lambda _t, r: decomposition_rate(comp, r, temp),
            (0.0, t_end),
            [comp.virgin_density],
            rtol=1e-12,
            atol=1e-12,
        )
        exact = comp.char_density + (comp.virgin_density - comp.char_density) * np.exp(
            -k * t_end
        )
        assert sol.y[0, -1] == pytest.approx(exact, rel=1e-9)

    def test_second_order_closed_form(self, material):
        """n = 2: extent x = (rho - rho_c)/rho_v obeys 1/x - 1/x0 = A e^{-E/RT} t."""
        comp = material.resin_b
        assert comp.reaction_order == 2.0
        temp = 1100.0
        k = comp.pre_exponential * np.exp(-comp.activation_energy / (GAS_CONSTANT * temp))
        x0 = (comp.virgin_density - comp.char_density) / comp.virgin_density
        t_end = 1.0 / (k * x0)

        sol = scipy.integrate.solve_ivp(
            lambda _t, r: decomposition_rate(comp, r, temp),
            (0.0, t_end),
            [comp.virgin_density],
            rtol=1e-12,
            atol=1e-12,
        )
        x_exact = 1.0 / (1.0 / x0 + k * t_end)
        rho_exact = comp.char_density + comp.virgin_density * x_exact
        assert sol.y[0, -1] == pytest.approx(rho_exact, rel=1e-9)

    def test_bulk_density_and_char_fraction(self, material):
        rho_v = bulk_density(
            material,
            material.resin_a.virgin_density,
            material.resin_b.virgin_density,
            material.filler.virgin_density,
        )
        assert float(rho_v) == pytest.approx(material.virgin_bulk_density)
        assert float(degree_of_char(material, rho_v)) == pytest.approx(0.0, abs=1e-14)
        assert float(degree_of_char(material, material.char_bulk_density)) == pytest.approx(1.0)
        # clipping against overshoot
        assert float(degree_of_char(material, material.virgin_bulk_density + 50.0)) == 0.0

    def test_component_validation(self):
        with pytest.raises(ValueError, match="char_density"):
            ArrheniusComponent(1e4, 7e4, 1.0, virgin_density=100.0, char_density=100.0)
        with pytest.raises(ValueError, match="activation_energy"):
            ArrheniusComponent(1e4, -1.0, 1.0, virgin_density=100.0, char_density=50.0)
        with pytest.raises(ValueError):
            decomposition_rate(demo_material().resin_a, 300.0, -10.0)


class TestLandauFrame:
    def test_thickness_and_mapping(self):
        frame = LandauFrame(total_thickness=0.05)
        assert frame.thickness(0.01) == pytest.approx(0.04)
        y = frame.physical_coordinate(np.array([0.0, 0.5, 1.0]), 0.01)
        assert np.allclose(y, [0.01, 0.03, 0.05])

    def test_burn_through_guard(self):
        frame = LandauFrame(total_thickness=0.05)
        with pytest.raises(ValueError, match="burn-through"):
            frame.thickness(0.0495)
        with pytest.raises(ValueError, match="recession"):
            frame.thickness(-0.001)

    def test_grid_velocity_vanishes_at_back_face(self):
        frame = LandauFrame(total_thickness=0.05)
        c = frame.grid_velocity_coefficient(np.array([0.0, 0.5, 1.0]), 0.01, 1e-3)
        assert c[2] == 0.0, "no inflow characteristic at the back face"
        assert c[0] == pytest.approx(1e-3 / 0.04)


class TestBlowingCorrection:
    def test_exact_limit_at_zero(self):
        assert float(blowing_correction(0.0)) == 1.0

    def test_small_argument_second_order(self):
        """phi = 1 - x/2 + x^2/3 - ... with x = 2 lambda B'."""
        for b in (1e-12, 1e-8, 1e-4):
            x = 2 * 0.5 * b
            expected = 1.0 - x / 2.0 + x * x / 3.0
            assert float(blowing_correction(b, 0.5)) == pytest.approx(expected, rel=1e-12)

    def test_no_catastrophic_collapse_at_tiny_b(self):
        """The naive log(1+x)/x rounds 1+x to 1 for x < eps and returns 0;
        the log1p form must not."""
        b = 1e-17
        x = 2 * 0.5 * b
        naive = np.log(1.0 + x) / x if (1.0 + x) != 1.0 else 0.0
        assert naive == 0.0, "premise: the naive form collapses here"
        assert float(blowing_correction(b, 0.5)) == pytest.approx(1.0, rel=1e-12)

    def test_monotone_decreasing(self):
        b = np.linspace(0.0, 5.0, 200)
        phi = blowing_correction(b, 0.5)
        assert np.all(np.diff(phi) < 0.0)
        assert np.all((phi > 0.0) & (phi <= 1.0))

    def test_validation(self):
        with pytest.raises(ValueError, match="lambda"):
            blowing_correction(0.1, lam=0.0)
        with pytest.raises(ValueError, match=">= 0"):
            blowing_correction(-0.1)


class TestSurfaceEnergyBalance:
    @staticmethod
    def _environment(q_rad=2.0e5):
        return SurfaceEnvironment(
            film_coefficient=0.08,
            recovery_enthalpy=1.2e7,
            radiative_flux=q_rad,
            absorptivity=0.9,
            wall_enthalpy=lambda t_w: 1004.5 * t_w,
        )

    def test_radiative_equilibrium_root(self, material):
        """No ablation, no conduction: the solved wall temperature must
        satisfy the balance identically."""
        seb = SurfaceEnergyBalance(material, self._environment())
        t_w = seb.solve_wall_temperature(0.0, 0.0, 0.0)
        assert 200.0 < t_w < 6000.0
        assert abs(seb.residual(t_w, 0.0, 0.0, 0.0)) < 1e-6 * (
            self._environment().film_coefficient * self._environment().recovery_enthalpy
        )

    def test_residual_monotone_decreasing_in_tw(self, material):
        seb = SurfaceEnergyBalance(material, self._environment())
        ts = np.linspace(300.0, 5000.0, 40)
        res = [seb.residual(t, 0.01, 0.02, 5.0e4) for t in ts]
        assert np.all(np.diff(res) < 0.0)

    def test_ablation_cools_the_wall(self, material):
        """Blowing reduces convective input, so the balanced wall
        temperature must drop when mass flux turns on."""
        seb = SurfaceEnergyBalance(material, self._environment())
        t_dry = seb.solve_wall_temperature(0.0, 0.0, 1.0e5)
        t_blown = seb.solve_wall_temperature(0.02, 0.04, 1.0e5)
        assert t_blown < t_dry

    def test_no_bracket_raises(self, material):
        seb = SurfaceEnergyBalance(material, self._environment(q_rad=0.0))
        with pytest.raises(ValueError, match="sign change"):
            seb.solve_wall_temperature(0.0, 0.0, 0.0, bracket=(5000.0, 6000.0))

    def test_input_validation(self, material):
        seb = SurfaceEnergyBalance(material, self._environment())
        with pytest.raises(ValueError, match="mass fluxes"):
            seb.residual(1000.0, -0.1, 0.0, 0.0)
        with pytest.raises(ValueError, match="wall temperature"):
            seb.residual(-10.0, 0.0, 0.0, 0.0)


class TestSurfaceThermochemistry:
    @staticmethod
    def _table():
        t = np.linspace(1000.0, 3500.0, 12)
        b = np.linspace(0.0, 1.0, 9)
        tt, bb = np.meshgrid(t, b, indexing="ij")
        z = 0.1 * (tt / 3500.0) ** 2 * (1.0 + 0.5 * bb)
        return SurfaceThermochemistry(t, b, z), (t, b)

    def test_reproduces_smooth_table_function(self):
        table, _ = self._table()
        for t_w, b_g in ((1500.0, 0.25), (2750.0, 0.6), (3400.0, 0.95)):
            exact = 0.1 * (t_w / 3500.0) ** 2 * (1.0 + 0.5 * b_g)
            assert table.char_blowing_rate(t_w, b_g) == pytest.approx(exact, rel=5e-3)

    def test_refuses_extrapolation(self):
        table, _ = self._table()
        with pytest.raises(ValueError, match="outside tabulated range"):
            table.char_blowing_rate(500.0, 0.5)
        with pytest.raises(ValueError, match="outside tabulated range"):
            table.char_blowing_rate(2000.0, 1.5)

    def test_recession_rate_formula(self):
        table, _ = self._table()
        b_c = table.char_blowing_rate(2000.0, 0.5)
        sdot = table.recession_rate(2000.0, 0.5, film_coefficient=0.08, char_density=250.0)
        assert sdot == pytest.approx(b_c * 0.08 / 250.0)

    def test_grid_validation(self):
        t = np.linspace(1000.0, 3500.0, 12)
        b = np.linspace(0.0, 1.0, 9)
        with pytest.raises(ValueError, match="strictly increasing"):
            SurfaceThermochemistry(t[::-1], b, np.zeros((12, 9)))
        with pytest.raises(ValueError, match="shape"):
            SurfaceThermochemistry(t, b, np.zeros((9, 12)))
        with pytest.raises(ValueError, match="finite"):
            SurfaceThermochemistry(t, b, np.full((12, 9), np.nan))


class TestCharringThermalSolver:
    @staticmethod
    def _solver(n=12, convention="eta_frame"):
        grid = ChebyshevGrid(n, interval=(0.0, 1.0), max_derivative=2)
        frame = LandauFrame(total_thickness=0.05)
        return CharringThermalSolver(grid, demo_material(), frame, convention), grid

    def test_grid_domain_enforced(self):
        grid = ChebyshevGrid(12, interval=(0.0, 2.0))
        with pytest.raises(ValueError, match="Landau domain"):
            CharringThermalSolver(grid, demo_material(), LandauFrame(0.05))

    def test_pack_unpack_roundtrip(self):
        solver, grid = self._solver()
        rng = np.random.default_rng(2)
        state = ThermalState(
            temperature=rng.uniform(300.0, 2000.0, grid.size),
            partial_densities=rng.uniform(100.0, 1000.0, (3, grid.size)),
            recession=1e-3,
        )
        rt = solver.unpack(solver.pack(state))
        assert np.array_equal(rt.temperature, state.temperature)
        assert np.array_equal(rt.partial_densities, state.partial_densities)
        assert rt.recession == state.recession
        assert solver.state_size == 4 * grid.size + 1

    def test_gas_flux_polynomial_exact(self):
        """rho_dot = eta^2 - 0.3: mdot = -ell * int_1^eta rho_dot =
        ell*[ (1-eta^3)/3 - 0.3(1-eta) ] ... solved spectrally exactly."""
        solver, grid = self._solver(n=10)
        eta = grid.x
        ell = 0.04
        rho_dot = eta**2 - 0.3
        mdot = solver.gas_flux(rho_dot, ell)
        # mdot(eta) = mdot(1) - ell*[F(eta) - F(1)] with F the antiderivative
        # of rho_dot: exact = ell*((1 - eta^3)/3 - 0.3(1 - eta))
        exact = ell * ((1.0 - eta**3) / 3.0 - 0.3 * (1.0 - eta))
        assert np.allclose(mdot, exact, atol=1e-13 * ell)

    def test_quiescent_virgin_state_is_stationary(self, material):
        """Uniform cold temperature, virgin densities, no recession: all
        rates must be numerically negligible."""
        solver, grid = self._solver()
        state = ThermalState(
            temperature=np.full(grid.size, 300.0),
            partial_densities=np.vstack(
                [np.full(grid.size, c.virgin_density) for c in material.components]
            ),
            recession=0.0,
        )
        ydot = solver.rhs(0.0, solver.pack(state), lambda t, s: 0.0)
        out = solver.unpack(ydot)
        assert np.max(np.abs(out.temperature)) < 1e-6  # K/s
        assert np.max(np.abs(out.partial_densities)) < 1e-6
        assert out.recession == 0.0

    def test_nonphysical_temperature_raises(self):
        solver, grid = self._solver()
        state = ThermalState(
            temperature=np.full(grid.size, -5.0),
            partial_densities=np.full((3, grid.size), 500.0),
            recession=0.0,
        )
        with pytest.raises(FloatingPointError, match="non-physical"):
            solver.rhs(0.0, solver.pack(state), lambda t, s: 0.0)

    def test_conventions_agree_without_recession_gradient(self, material):
        """With sdot = 0 the eta-frame and material-frame density rates
        coincide, so the two conventions must produce identical RHS."""
        rhs_vals = []
        for convention in ("eta_frame", "material_frame"):
            solver, grid = self._solver(convention=convention)
            state = ThermalState(
                temperature=np.linspace(1600.0, 400.0, grid.size),
                partial_densities=np.vstack(
                    [
                        np.full(grid.size, 0.5 * (c.virgin_density + c.char_density))
                        for c in material.components
                    ]
                ),
                recession=0.0,
            )
            rhs_vals.append(solver.rhs(0.0, solver.pack(state), lambda t, s: 0.0))
        assert np.allclose(rhs_vals[0], rhs_vals[1], rtol=1e-14)

    def test_mms_single_resolution(self):
        """Manufactured solution held to high accuracy at N = 14 over a
        short horizon (the full convergence sweep is the V4 runner)."""
        mat = demo_material()
        frame = LandauFrame(total_thickness=0.05)
        mms = ManufacturedAblation(material=mat, frame=frame)
        grid = ChebyshevGrid(14, interval=(0.0, 1.0), max_derivative=2)
        solver = CharringThermalSolver(grid, mat, frame)
        eta = grid.x
        g_t = mms.energy_source()
        g_rho = (mms.density_source(0), mms.density_source(1), mms.density_source(2))
        t_f = 0.5

        sol = scipy.integrate.solve_ivp(
            lambda t, y: solver.rhs(
                t,
                y,
                lambda tt, s: mms.recession_rate(tt),
                surface_rate=mms.surface_rate,
                back_face_rate=mms.back_face_rate,
                energy_source=g_t,
                density_sources=g_rho,
            ),
            (0.0, t_f),
            solver.pack(mms.initial_state(eta)),
            method="DOP853",
            rtol=1e-11,
            atol=1e-9,
        )
        assert sol.success
        end = solver.unpack(sol.y[:, -1])
        t_err = np.max(np.abs(end.temperature - mms.temperature(eta, t_f)))
        assert t_err / mms.temperature_span < 1e-10
        assert end.recession == pytest.approx(mms.recession(t_f), abs=1e-15)

    def test_manufactured_gas_flux_matches_operator(self):
        """The solver's spectral gas-flux solve must reproduce the
        closed-form manufactured flux."""
        mat = demo_material()
        frame = LandauFrame(total_thickness=0.05)
        mms = ManufacturedAblation(material=mat, frame=frame)
        solver, grid = self._solver(n=20)
        eta = grid.x
        t = 0.7
        mdot = solver.gas_flux(mms.bulk_density_t(eta, t), mms.thickness(t))
        assert np.allclose(mdot, mms.gas_flux(eta, t), atol=1e-12)
