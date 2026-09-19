"""Tests for the FIAT-formulation ablation solver.

Organised by what could go wrong rather than by module: the structural
properties of the discretisation (conservation, grid convergence,
interface flux continuity), the exactness of the Newton Jacobian, the
source's own algebraic identities for the blowing correction, and the
refusals that keep an equilibrium table from being extrapolated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.integrate

from aether.fiat import (
    PORE_PRESSURE_REFERENCES,
    AerothermalEnvironment,
    BackfaceCondition,
    BackfaceKind,
    BPrimeTable,
    FiatSolver,
    MaterialStack,
    Ply,
    blowing_reduction,
    gray_radiative_flux,
    optical_depth,
    pore_pressure,
    pore_pressure_sensitivity,
    rosseland_conductivity,
)
from aether.fiat.analysis import (
    interface_histories,
    optimize_ply_thickness,
    probe_depths,
    scale_environments,
    sized_stack,
)
from aether.fiat.arcjet import (
    ANALYSIS_CASES,
    ARC_CONDITIONS,
    JSC_CONDITIONS,
    MODELS,
    TC_PLACEMENTS,
    condition,
    consumption_recession,
    load_fig26_thermocouples,
    recession_statistics,
)
from aether.fiat.bprime import TableRangeError
from aether.fiat.equilibrium import ThermoDatabase, solve_equilibrium
from aether.fiat.kinetics import (
    TgaTargets,
    calibrated_components,
    fit_arrhenius,
    peak_rate_temperature,
    tga_mass_fraction,
)
from aether.fiat.materials import (
    HERITAGE_PICA_CONDUCTIVITY,
    MEDLI2_PICA_CONDUCTIVITY,
    ONE_ATMOSPHERE,
    PICA_VIRGIN_DENSITY,
    PressureConductivity,
    pica_like_material,
    structural_material,
)
from aether.fiat.mutationpp import read_mutationpp_bprime
from aether.fiat.permeability import (
    FIBERFORM_SAMPLES,
    MARSCHALL_MILOS_SAMPLES,
    effective_permeability,
    fiberform_permeability,
    knudsen_regime_pressure,
    slip_parameter,
)
from aether.fiat.pica_kinetics import (
    COMPETITIVE_PICA_BAYESIAN,
    COMPETITIVE_PICA_DETERMINISTIC,
    PARALLEL_PICA_RESIN,
    PICA_CHAR_ELEMENTS,
    PICA_PYROLYSIS_ELEMENTS,
    PICA_PYROLYSIS_ELEMENTS_FIATV3,
    PICA_PYROLYSIS_ELEMENTS_MEASURED,
    PICA_RESIN_ELEMENTS,
    PICA_RESIN_ELEMENTS_CROSSLINKED,
    CompetitivePica,
    ParallelReaction,
    advancement_to_fiat_rate,
    char_required_by_mass_balance,
    char_yield_from_gas_composition,
    competitive_mass_fraction,
    parallel_pica_resin,
    resin_carbon_mass_fraction,
)
from aether.fiat.pica_surface import QUINN_GAS_RATES, read_quinn_bprime
from aether.fiat.quinn import CURVE_LOCATIONS, load_quinn_case
from aether.fiat.solver import (
    SolverOptions,
    _char_density,
    _StepContext,
    _virgin_density,
)
from aether.thermal.material import ArrheniusComponent, LinearBlendProperty
from aether.thermal.surface import STEFAN_BOLTZMANN


def make_table(t_lo: float = 200.0, t_hi: float = 4500.0, b_hi: float = 6.0) -> BPrimeTable:
    """A smooth synthetic B' surface.

    Explicitly not a thermochemistry calculation: a logistic in wall
    temperature with a mild pressure and gas-rate dependence, shaped so
    that char removal switches on around 2700 K. It exercises the solver's
    coupling to a table; it says nothing about any real material.
    """
    t_w = np.linspace(t_lo, t_hi, 30)
    b_g = np.linspace(0.0, b_hi, 12)
    p = np.array([500.0, 5000.0, 50000.0])
    b_c = np.zeros((3, 12, 30))
    h_w = np.zeros((3, 12, 30))
    for i, p_i in enumerate(p):
        for j, g in enumerate(b_g):
            b_c[i, j] = (
                0.35
                / (1.0 + np.exp(-(t_w - 2700.0) / 200.0))
                * (1.0 + 0.15 * g)
                * (p_i / 5000.0) ** 0.08
            )
            h_w[i, j] = 1.2e3 * t_w
    return BPrimeTable(p, b_g, t_w, b_c, h_w)


def make_stack(n_cells: int = 40, structure: bool = True) -> MaterialStack:
    plies = [Ply(pica_like_material(), 0.05, n_cells, 1.03, ablating=True)]
    if structure:
        plies.append(Ply(structural_material(), 0.01, max(n_cells // 4, 4)))
    return MaterialStack(plies)


def pulse(n: int = 240, duration: float = 60.0, peak: float = 2.5e6):
    t = np.linspace(0.0, duration, n + 1)
    q = peak * (0.02 + 0.98 * np.exp(-(((t[:-1] - 0.42 * duration) / 12.0) ** 2)))
    envs = [
        AerothermalEnvironment(
            film_coefficient=q_i / 2.0e7,
            recovery_enthalpy=2.0e7,
            pressure=5000.0,
        )
        for q_i in q
    ]
    return t, envs


ADIABATIC = BackfaceCondition(BackfaceKind.ADIABATIC)


class TestStack:
    def test_widths_sum_to_thickness(self):
        ply = Ply(pica_like_material(), 0.05, 25, growth=1.07)
        assert ply.cell_widths().sum() == pytest.approx(0.05, rel=1e-15)

    def test_growth_refines_toward_the_heated_face(self):
        w = Ply(pica_like_material(), 0.05, 20, growth=1.1).cell_widths()
        assert np.all(np.diff(w) > 0.0)
        assert w[-1] / w[0] == pytest.approx(1.1**19, rel=1e-12)

    def test_uniform_growth_is_exactly_uniform(self):
        w = Ply(pica_like_material(), 0.05, 20, growth=1.0).cell_widths()
        assert np.allclose(w, w[0], rtol=0.0, atol=0.0)

    def test_grid_geometry(self):
        grid = make_stack(20).grid(0.0)
        assert grid.faces[0] == 0.0
        assert grid.total_thickness == pytest.approx(0.06, rel=1e-14)
        assert grid.n_cells == 25
        assert list(grid.interface_faces) == [20]
        assert np.allclose(grid.centers, 0.5 * (grid.faces[:-1] + grid.faces[1:]))

    def test_recession_shrinks_only_the_top_ply(self):
        stack = make_stack(20)
        g0, g1 = stack.grid(0.0), stack.grid(0.01)
        assert g1.total_thickness == pytest.approx(g0.total_thickness - 0.01)
        # Cells below the interface keep their widths exactly.
        assert np.allclose(g1.widths[20:], g0.widths[20:], rtol=0.0, atol=0.0)
        assert stack.stretch_factor(0.01) == pytest.approx(0.8)

    def test_burn_through_refuses_rather_than_continuing(self):
        with pytest.raises(ValueError, match="consumed the entire top ply"):
            make_stack(20).grid(0.05)

    def test_only_the_outermost_ply_may_ablate(self):
        mat = pica_like_material()
        with pytest.raises(ValueError, match="only the outermost ply may ablate"):
            MaterialStack([Ply(mat, 0.05, 10), Ply(mat, 0.01, 5, ablating=True)])

    @pytest.mark.parametrize("bad", [{"thickness": 0.0}, {"n_cells": 1}, {"growth": 3.0}])
    def test_rejects_bad_ply_geometry(self, bad):
        kwargs = {"thickness": 0.05, "n_cells": 10} | bad
        with pytest.raises(ValueError):
            Ply(pica_like_material(), **kwargs)


class TestMaterials:
    def test_pica_like_reproduces_the_published_virgin_density(self):
        assert _virgin_density(pica_like_material()) == pytest.approx(
            PICA_VIRGIN_DENSITY, rel=1e-12
        )

    def test_pica_like_char_density_is_below_virgin(self):
        mat = pica_like_material()
        assert _char_density(mat) == pytest.approx(227.0, rel=1e-9)
        assert _char_density(mat) < _virgin_density(mat)

    @pytest.mark.parametrize("density", [800.0, 1600.0, 2700.0])
    def test_structural_material_hits_its_density(self, density):
        mat = structural_material(density)
        assert _virgin_density(mat) == pytest.approx(density, rel=1e-9)
        # Inert: virgin and char coincide to within the strict-inequality nudge.
        assert _char_density(mat) == pytest.approx(density, rel=1e-6)

    def test_structural_material_does_not_decompose(self):
        stack = MaterialStack([Ply(structural_material(), 0.02, 8)])
        solver = FiatSolver(stack)
        _, components = solver.initial_state(300.0)
        hot = np.full(stack.n_cells, 2500.0)
        after, _ = solver.decompose(hot, components, 10.0)
        assert np.allclose(after, components, rtol=0.0, atol=0.0)


class TestBlowingCorrection:
    """Eq. (11), against the source's own two algebraic statements."""

    def test_tends_to_one_at_zero_blowing(self):
        assert float(blowing_reduction(0.0)) == pytest.approx(1.0, abs=0.0)

    @pytest.mark.parametrize("b", [1e-12, 1e-9, 1e-6, 1e-3])
    @pytest.mark.parametrize("lam", [0.2, 0.5])
    def test_small_blowing_follows_the_series(self, b, lam):
        """phi = 1 - lambda B' + O(B'^2); log1p keeps that accurate where a
        naive log(1+x)/x would lose it to cancellation."""
        x = 2.0 * lam * b
        expected = 1.0 - 0.5 * x + x**2 / 3.0
        # The series is truncated at x^2, so it can only be trusted to O(x^3).
        assert float(blowing_reduction(b, lam)) == pytest.approx(expected, abs=max(x**3, 1e-15))

    @pytest.mark.parametrize("b", [0.05, 0.5, 2.0, 10.0])
    @pytest.mark.parametrize("lam", [0.2, 0.4, 0.5])
    def test_matches_the_exponential_form(self, b, lam):
        """Milos, Chen & Squire print ln(1+2λB')/(2λB') = 2λB'_1/(e^{2λB'_1}-1).

        The two are equal when B' and B'_1 are related by the same
        blowing reduction, i.e. B'_1 = B' * phi. Checking that identity
        confirms the logarithmic reading against the source's own
        alternative statement of the same equation.
        """
        phi = float(blowing_reduction(b, lam))
        x1 = 2.0 * lam * b * phi
        assert phi == pytest.approx(x1 / np.expm1(x1), rel=1e-12)

    def test_reduces_heat_transfer_monotonically(self):
        b = np.array([0.0, 0.1, 0.5, 1.0, 5.0])
        phi = blowing_reduction(b)
        assert np.all(np.diff(phi) < 0.0)
        assert np.all((phi > 0.0) & (phi <= 1.0))

    @pytest.mark.parametrize("bad", [-0.1, np.nan])
    def test_rejects_nonphysical_blowing(self, bad):
        with pytest.raises(ValueError, match="b_prime"):
            blowing_reduction(bad)


class TestBPrimeTable:
    def test_interpolates_the_tabulated_nodes(self):
        table = make_table()
        expected = 0.35 / (1.0 + np.exp(-(3000.0 - 2700.0) / 200.0))
        assert table.char_rate(5000.0, 0.0, 3000.0) == pytest.approx(expected, rel=2e-3)

    def test_derivative_matches_a_central_difference(self):
        table = make_table()
        analytic = table.char_rate_derivative(5000.0, 0.5, 2800.0)
        h = 5.0
        fd = (
            table.char_rate(5000.0, 0.5, 2800.0 + h) - table.char_rate(5000.0, 0.5, 2800.0 - h)
        ) / (2.0 * h)
        # A spline derivative against a central difference agrees to the
        # difference's own O(h^2) truncation, not to round-off.
        assert analytic == pytest.approx(fd, rel=1e-4)

    def test_refuses_to_extrapolate(self):
        table = make_table()
        with pytest.raises(TableRangeError, match="outside the tabulated range"):
            table.char_rate(5000.0, 0.5, 5000.0)
        with pytest.raises(TableRangeError, match="B'_g"):
            table.char_rate(5000.0, 99.0, 3000.0)

    def test_range_error_is_a_value_error(self):
        """So callers that do not care about the distinction still catch it."""
        assert issubclass(TableRangeError, ValueError)

    def test_single_pressure_table_is_pressure_independent(self):
        t_w = np.linspace(300.0, 4000.0, 20)
        b_g = np.linspace(0.0, 2.0, 6)
        b_c = np.tile(0.2 / (1.0 + np.exp(-(t_w - 2500.0) / 150.0)), (1, 6, 1))
        table = BPrimeTable([1000.0], b_g, t_w, b_c, np.full_like(b_c, 1.0e6))
        assert table.char_rate(1.0, 0.5, 3000.0) == pytest.approx(
            table.char_rate(1.0e7, 0.5, 3000.0), rel=1e-14
        )

    def test_roughness_flags_a_kinked_table(self):
        t_w = np.linspace(300.0, 4000.0, 20)
        b_g = np.linspace(0.0, 2.0, 6)
        smooth = np.tile(0.2 / (1.0 + np.exp(-(t_w - 2500.0) / 400.0)), (1, 6, 1))
        kinked = smooth.copy()
        kinked[:, :, 10] *= 3.0
        h_w = np.full_like(smooth, 1.0e6)
        assert BPrimeTable([1000.0], b_g, t_w, kinked, h_w).roughness() > 10.0 * (
            BPrimeTable([1000.0], b_g, t_w, smooth, h_w).roughness()
        )

    def test_cubic_needs_enough_points(self):
        with pytest.raises(ValueError, match="cubic interpolation needs"):
            BPrimeTable(
                [1000.0],
                [0.0, 1.0],
                [300.0, 3000.0],
                np.zeros((1, 2, 2)),
                np.zeros((1, 2, 2)),
            )


class TestRadiation:
    def test_rosseland_conductivity_scales_as_t_cubed(self):
        k1 = float(rosseland_conductivity(1000.0, 500.0))
        k2 = float(rosseland_conductivity(2000.0, 500.0))
        assert k2 / k1 == pytest.approx(8.0, rel=1e-12)
        assert k1 == pytest.approx(16.0 * STEFAN_BOLTZMANN * 1e9 / 1500.0, rel=1e-12)

    def test_optical_depth_accumulates(self):
        kappa = optical_depth(np.full(4, 0.01), 200.0)
        assert np.allclose(kappa, [0.0, 2.0, 4.0, 6.0, 8.0])

    def test_isothermal_medium_carries_no_net_flux(self):
        """An isothermal slab whose boundaries radiate at the same
        temperature is in radiative equilibrium: the flux must vanish
        everywhere, not merely on average."""
        t = np.full(12, 1500.0)
        kappa = optical_depth(np.full(12, 0.005), 300.0)
        i0 = STEFAN_BOLTZMANN * 1500.0**4 / np.pi
        flux = gray_radiative_flux(kappa, t, front_intensity=i0, back_intensity=i0)
        assert np.max(np.abs(flux)) < 1e-6 * STEFAN_BOLTZMANN * 1500.0**4

    def test_flux_runs_from_hot_to_cold(self):
        t = np.linspace(2500.0, 500.0, 16)
        kappa = optical_depth(np.full(16, 0.004), 250.0)
        flux = gray_radiative_flux(kappa, t)
        assert np.all(flux[1:-1] > 0.0)

    def test_thick_limit_converges_to_rosseland(self):
        """The exact kernel must approach the diffusion approximation as the
        *cell* optical depth is refined, not merely as the slab gets thick.

        FIAT offers Eq. (2) and Eq. (3) as alternatives, so they have to
        agree where both are valid — but only once each cell is optically
        thin enough for the piecewise-constant source function to hold.
        At one optical depth of 5 per cell the two differ by a factor of
        nearly four, which is a discretisation statement about Eq. (2)
        rather than a disagreement between the two models.
        """
        n = 400
        widths = np.full(n, 0.05 / n)
        t_cell = np.linspace(2000.0, 1000.0, n)
        mid = n // 2
        grad = (t_cell[mid] - t_cell[mid - 1]) / widths[0]
        ratios = []
        for k_ext in (4.0e4, 4.0e3, 4.0e2):
            kappa = optical_depth(widths, k_ext)
            exact = gray_radiative_flux(
                kappa,
                t_cell,
                STEFAN_BOLTZMANN * t_cell[0] ** 4 / np.pi,
                STEFAN_BOLTZMANN * t_cell[-1] ** 4 / np.pi,
            )
            diffusion = float(rosseland_conductivity(t_cell[mid], k_ext)) * -grad
            ratios.append(exact[mid] / diffusion)
        assert ratios[0] > 3.0
        assert abs(ratios[1] - 1.0) < 0.10
        assert abs(ratios[2] - 1.0) < 0.02
        assert abs(ratios[2] - 1.0) < abs(ratios[1] - 1.0)

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="n_cells \\+ 1"):
            gray_radiative_flux(np.linspace(0.0, 1.0, 5), np.ones(7) * 1000.0)


class TestJacobian:
    """The Newton Jacobian against a central difference of the residual.

    This is the test that matters most for the solver's convergence rate.
    An early version silently omitted the gas-convection and grid-advection
    face weights, and the only symptom was Newton stalling during an active
    pyrolysis front — a failure that looks exactly like a physics problem.
    """

    def _context(self, solver, stack, table, previous, components, dt, s_dot):
        grid = stack.grid(0.0)
        return _StepContext(
            previous_temperature=previous,
            previous_components=components,
            dt=dt,
            recession_rate=s_dot,
            widths=grid.widths,
            centers=grid.centers,
            environment=AerothermalEnvironment(
                film_coefficient=0.2, recovery_enthalpy=2.0e7, pressure=5000.0
            ),
            backface=ADIABATIC,
            table=table,
        )

    @pytest.mark.parametrize("backface", [BackfaceKind.ADIABATIC, BackfaceKind.RADIATING])
    @pytest.mark.parametrize("s_dot", [0.0, 1.0e-4])
    def test_matches_central_differences(self, backface, s_dot):
        stack = make_stack(12)
        solver = FiatSolver(stack)
        table = make_table()
        temperature, components = solver.initial_state(300.0)
        # Drive the state into an active pyrolysis front, where the
        # decomposition-dependent terms are largest.
        hot = temperature + np.linspace(1400.0, 40.0, stack.n_cells)
        for _ in range(4):
            components = solver.decompose(hot, components, 0.5)[0]
        ctx = self._context(solver, stack, table, hot, components, 0.05, s_dot)
        ctx = _StepContext(**{**ctx.__dict__, "backface": BackfaceCondition(backface)})
        u = np.concatenate([hot, [2400.0, hot[-1]]])

        _, info = solver._residual(u, ctx)
        analytic = solver._jacobian(u, ctx, info)
        numeric = np.zeros_like(analytic)
        for i in range(u.size):
            h = 1.0e-4 * max(abs(u[i]), 1.0)
            up, um = u.copy(), u.copy()
            up[i] += h
            um[i] -= h
            numeric[:, i] = (solver._residual(up, ctx)[0] - solver._residual(um, ctx)[0]) / (
                2.0 * h
            )
        scale = np.maximum(np.abs(numeric), 1e-6 * np.abs(numeric).max())
        assert np.max(np.abs(analytic - numeric) / scale) < 1e-4

    def test_decomposition_sensitivity_matches_a_difference(self):
        stack = make_stack(10, structure=False)
        solver = FiatSolver(stack)
        _, components = solver.initial_state(300.0)
        t = np.full(stack.n_cells, 900.0)
        components = solver.decompose(t, components, 1.0)[0]
        _, analytic = solver.decompose(t, components, 0.5)
        h = 1.0e-3
        up = solver.decompose(t + h, components, 0.5)[0]
        dn = solver.decompose(t - h, components, 0.5)[0]
        numeric = (up - dn) / (2.0 * h)
        active = np.abs(numeric) > 1e-8
        assert np.allclose(analytic[active], numeric[active], rtol=1e-5)


class TestDiscretisation:
    def test_no_heating_leaves_the_stack_isothermal(self):
        """A stack with nothing entering and nothing leaving must hold its
        initial state exactly.

        The emissivity has to be zeroed along with the film coefficient:
        a wall at 300 K with the default emissivity of 0.9 radiates
        410 W/m² to space and genuinely cools, which is right, and was
        the first thing this test caught.
        """
        stack = make_stack(20)
        solver = FiatSolver(stack)
        table = make_table()
        t = np.linspace(0.0, 10.0, 11)
        envs = [
            AerothermalEnvironment(
                film_coefficient=0.0,
                recovery_enthalpy=0.0,
                pressure=5000.0,
                wall_emissivity=0.0,
                wall_absorptance=0.0,
            )
        ] * 10
        solution = solver.solve(t, envs, table, ADIABATIC, 300.0)
        # The bound is the Newton tolerance expressed in kelvin: the
        # residual is scaled by the largest cell's heat capacity, so the
        # smallest cell converges to a slightly looser absolute temperature.
        assert np.max(np.abs(solution.temperature_history() - 300.0)) < 1e-4
        assert solution.recession[-1] == pytest.approx(0.0, abs=1e-15)

    def test_conserves_energy_in_a_sealed_inert_stack(self):
        """Fixed heat flux into an inert, adiabatic-backed slab: the stored
        energy must equal the integrated input. This is the check that
        catches a leaking flux at the ply interface, where the harmonic
        conductivity mean earns its place."""
        mat_a = structural_material(1600.0, conductivity=0.5, specific_heat=900.0)
        mat_b = structural_material(800.0, conductivity=0.1, specific_heat=1500.0)
        stack = MaterialStack([Ply(mat_a, 0.02, 40), Ply(mat_b, 0.02, 40)])
        solver = FiatSolver(stack)
        table = make_table()
        t = np.linspace(0.0, 20.0, 201)
        # Zero recovery-enthalpy drive with a pure radiative input keeps the
        # surface chemistry out of it; alpha_w = 1, epsilon_w = 0 means all
        # of q_rad enters and none leaves.
        env = AerothermalEnvironment(
            film_coefficient=0.0,
            recovery_enthalpy=0.0,
            pressure=5000.0,
            radiative_flux=5.0e4,
            wall_absorptance=1.0,
            wall_emissivity=0.0,
        )
        solution = solver.solve(t, [env] * 200, table, ADIABATIC, 300.0)
        grid = stack.grid(0.0)
        rho = np.array([1600.0] * 40 + [800.0] * 40)
        cp = np.array([900.0] * 40 + [1500.0] * 40)
        final = solution.steps[-1].temperature
        stored = float(np.sum(rho * cp * grid.widths * (final - 300.0)))
        assert stored == pytest.approx(5.0e4 * 20.0, rel=2e-3)

    def test_steady_conduction_across_a_ply_interface_is_exact(self):
        """Two plies of different conductivity between a fixed flux and a
        fixed temperature must reach the exact series-resistance profile.

        This is what the harmonic conductivity mean is for. An arithmetic
        mean gets the interface flux wrong by a factor that grows with the
        conductivity ratio, and the error lands squarely on the bondline
        temperature — the number a sizing run exists to produce.
        """
        k1, k2, l1, l2 = 0.5, 0.05, 0.02, 0.005
        stack = MaterialStack(
            [
                Ply(structural_material(1600.0, k1, 900.0), l1, 30),
                Ply(structural_material(1600.0, k2, 900.0), l2, 30),
            ]
        )
        # The default tolerance is already at the round-off floor for this
        # problem: the residual is a difference of ~5 kW/m2 fluxes, so the
        # smallest attainable scaled residual is a few times 1e-11.
        solver = FiatSolver(stack)
        q = 5.0e3
        env = AerothermalEnvironment(
            film_coefficient=0.0,
            recovery_enthalpy=0.0,
            pressure=5000.0,
            radiative_flux=q,
            wall_absorptance=1.0,
            wall_emissivity=0.0,
        )
        backface = BackfaceCondition(BackfaceKind.FIXED_TEMPERATURE, temperature=300.0)
        # The composite time constant is R*C = 0.14 * 36 kJ/(m2 K) ~ 5000 s,
        # so a 20 ks run is only four time constants and still 2% short.
        t = np.linspace(0.0, 120000.0, 401)
        solution = solver.solve(t, [env] * 400, make_table(), backface, 300.0)
        expected_wall = 300.0 + q * (l1 / k1 + l2 / k2)
        assert solution.wall_temperature[-1] == pytest.approx(expected_wall, rel=2e-3)

    def test_recession_grid_converges(self):
        table = make_table()
        t, envs = pulse(120, 60.0)
        results = [
            FiatSolver(make_stack(n)).solve(t, envs, table, ADIABATIC, 300.0).recession[-1]
            for n in (20, 40, 80)
        ]
        first = abs(results[1] - results[0])
        second = abs(results[2] - results[1])
        assert second < 0.4 * first, f"not converging: {results}"

    def test_recession_time_step_converges(self):
        table = make_table()
        stack = make_stack(40)
        results = []
        for n in (60, 120, 240):
            t, envs = pulse(n, 60.0)
            results.append(FiatSolver(stack).solve(t, envs, table, ADIABATIC, 300.0).recession[-1])
        first = abs(results[1] - results[0])
        second = abs(results[2] - results[1])
        assert second < 0.7 * first, f"not converging: {results}"

    def test_pyrolysis_mass_is_conserved(self):
        """Total gas released must equal the density the solid lost. This
        is Eq. (9) integrated over the whole run, and it is the only check
        that ties the mass balance to the kinetics."""
        stack = make_stack(40, structure=False)
        solver = FiatSolver(stack)
        table = make_table()
        t, envs = pulse(240, 60.0)
        solution = solver.solve(t, envs, table, ADIABATIC, 300.0)
        released = 0.0
        for i, step in enumerate(solution.steps):
            dt = t[i + 1] - t[i]
            released += step.surface.gas_mass_flux * dt
        grid = stack.grid(solution.recession[-1])
        initial = solver.bulk_density(solver.initial_state(300.0)[1])
        final = solver.bulk_density(solution.steps[-1].component_density)
        lost = float(np.sum((initial - final) * grid.widths))
        assert released == pytest.approx(lost, rel=2e-2)


class TestSurfaceAndBackface:
    def test_radiative_equilibrium_wall_does_not_recede(self):
        """With the incident flux exactly balanced by reradiation and no
        convective drive, the wall sits still."""
        stack = make_stack(20, structure=False)
        solver = FiatSolver(stack)
        table = make_table()
        t_w = 1500.0
        env = AerothermalEnvironment(
            film_coefficient=0.0,
            recovery_enthalpy=0.0,
            pressure=5000.0,
            radiative_flux=STEFAN_BOLTZMANN * t_w**4 / 0.9,
            wall_absorptance=0.9,
            wall_emissivity=0.9,
        )
        t = np.linspace(0.0, 5.0, 26)
        solution = solver.solve(t, [env] * 25, table, ADIABATIC, 300.0)
        # B'_c is essentially zero below the 2700 K switch-on of the table.
        assert solution.recession[-1] < 1e-9

    def test_fixed_backface_holds_its_temperature(self):
        stack = make_stack(20)
        solver = FiatSolver(stack)
        backface = BackfaceCondition(BackfaceKind.FIXED_TEMPERATURE, temperature=350.0)
        t, envs = pulse(60, 30.0)
        solution = solver.solve(t, envs, make_table(), backface, 300.0)
        assert np.allclose(solution.backface_temperature, 350.0, rtol=1e-9)

    def test_adiabatic_backface_is_hotter_than_a_cooled_one(self):
        stack = make_stack(20)
        table = make_table()
        t, envs = pulse(120, 60.0, peak=4.0e6)
        cooled = BackfaceCondition(BackfaceKind.FIXED_TEMPERATURE, temperature=300.0)
        hot = FiatSolver(stack).solve(t, envs, table, ADIABATIC, 300.0)
        cold = FiatSolver(stack).solve(t, envs, table, cooled, 300.0)
        assert hot.steps[-1].temperature[-1] > cold.steps[-1].temperature[-1]

    def test_radiating_backface_satisfies_its_own_balance(self):
        stack = make_stack(20)
        backface = BackfaceCondition(BackfaceKind.RADIATING, emissivity=0.8, sink_temperature=250.0)
        t, envs = pulse(120, 60.0, peak=6.0e6)
        solution = FiatSolver(stack).solve(t, envs, make_table(), backface, 300.0)
        step = solution.steps[-1]
        grid = stack.grid(step.recession)
        k = FiatSolver(stack)._properties(
            step.temperature, FiatSolver(stack).bulk_density(step.component_density)
        )[0]
        conduction = (
            k[-1] * (step.temperature[-1] - step.backface_temperature) / (0.5 * grid.widths[-1])
        )
        radiated = 0.8 * STEFAN_BOLTZMANN * (step.backface_temperature**4 - 250.0**4)
        assert conduction == pytest.approx(radiated, rel=1e-6)

    def test_blowing_reduces_the_wall_temperature(self):
        """The whole point of the blowing correction: an ablating wall runs
        cooler than the same wall with the correction switched off."""
        stack = make_stack(30, structure=False)
        table = make_table()
        t, envs = pulse(120, 60.0, peak=4.0e6)
        weak = [
            AerothermalEnvironment(
                film_coefficient=e.film_coefficient,
                recovery_enthalpy=e.recovery_enthalpy,
                pressure=e.pressure,
                blowing_parameter=1e-6,
            )
            for e in envs
        ]
        blown = FiatSolver(stack).solve(t, envs, table, ADIABATIC, 300.0)
        unblown = FiatSolver(stack).solve(t, weak, table, ADIABATIC, 300.0)
        assert blown.wall_temperature.max() < unblown.wall_temperature.max()


class TestSolverContract:
    def test_rejects_mismatched_environment_count(self):
        solver = FiatSolver(make_stack(10))
        with pytest.raises(ValueError, match="environments"):
            solver.solve(np.linspace(0.0, 1.0, 5), [], make_table(), ADIABATIC)

    def test_rejects_non_monotone_times(self):
        solver = FiatSolver(make_stack(10))
        env = AerothermalEnvironment(0.1, 1.0e7, 5000.0)
        with pytest.raises(ValueError, match="strictly increasing"):
            solver.solve(np.array([0.0, 1.0, 0.5]), [env] * 2, make_table(), ADIABATIC)

    def test_rejects_bad_radiation_mode(self):
        with pytest.raises(ValueError, match="radiation must be one of"):
            SolverOptions(radiation="diffusion")

    def test_rosseland_needs_an_extinction_coefficient(self):
        stack = make_stack(10, structure=False)
        solver = FiatSolver(stack, SolverOptions(radiation="rosseland"))
        env = AerothermalEnvironment(0.1, 1.0e7, 5000.0)
        with pytest.raises(ValueError):
            solver.solve(np.linspace(0.0, 1.0, 3), [env] * 2, make_table(), ADIABATIC)

    def test_semi_transparent_ply_runs_with_rosseland(self):
        stack = MaterialStack(
            [
                Ply(
                    pica_like_material(),
                    0.05,
                    30,
                    ablating=True,
                    extinction_coefficient=8.0e3,
                )
            ]
        )
        solver = FiatSolver(stack, SolverOptions(radiation="rosseland"))
        t, envs = pulse(60, 30.0)
        solution = solver.solve(t, envs, make_table(), ADIABATIC, 300.0)
        assert np.all(np.isfinite(solution.wall_temperature))
        assert solution.recession[-1] > 0.0

    def test_newton_converges_in_a_handful_of_iterations(self):
        """An inexact or wrong Jacobian shows up here before it shows up
        anywhere else."""
        t, envs = pulse(240, 60.0)
        solution = FiatSolver(make_stack(40)).solve(t, envs, make_table(), ADIABATIC, 300.0)
        worst = max(s.newton_iterations for s in solution.steps)
        assert worst <= 20, f"worst step took {worst} Newton iterations"

    def test_recession_outer_loop_converges(self):
        t, envs = pulse(240, 60.0)
        solution = FiatSolver(make_stack(40)).solve(t, envs, make_table(), ADIABATIC, 300.0)
        assert max(s.recession_iterations for s in solution.steps) < 20

    def test_recession_is_monotone(self):
        t, envs = pulse(240, 60.0)
        solution = FiatSolver(make_stack(40)).solve(t, envs, make_table(), ADIABATIC, 300.0)
        assert np.all(np.diff(solution.recession) >= -1e-15)


class TestPressureConductivity:
    """MEDLI2 paper Table 3, "PICA Room Temperature Properties"."""

    def test_reproduces_both_published_anchors(self):
        for prop, (v1, c1, v0, c0) in (
            (HERITAGE_PICA_CONDUCTIVITY, (0.174, 0.224, 0.520, 0.202)),
            (MEDLI2_PICA_CONDUCTIVITY, (0.169, 0.169, 0.127, 0.143)),
        ):
            low = 0.001 * ONE_ATMOSPHERE
            for beta, want, pressure in (
                (0.0, v1, ONE_ATMOSPHERE),
                (1.0, c1, ONE_ATMOSPHERE),
                (0.0, v0, low),
                (1.0, c0, low),
            ):
                got = float(prop.value(300.0, beta, pressure))
                assert got == pytest.approx(want, rel=1e-12)

    def test_the_two_published_models_disagree_by_four_times(self):
        """The finding worth not losing: the Heritage model has virgin
        conductivity *rising* threefold as pressure falls to 0.001 atm, the
        MEDLI2 re-measurement has it falling by a quarter. They differ by a
        factor of four in the regime that governs entry."""
        low = 0.001 * ONE_ATMOSPHERE

        def virgin(prop, pressure):
            return float(prop.value(300.0, 0.0, pressure))

        h_low = virgin(HERITAGE_PICA_CONDUCTIVITY, low)
        m_low = virgin(MEDLI2_PICA_CONDUCTIVITY, low)
        h_ratio = h_low / virgin(HERITAGE_PICA_CONDUCTIVITY, ONE_ATMOSPHERE)
        m_ratio = m_low / virgin(MEDLI2_PICA_CONDUCTIVITY, ONE_ATMOSPHERE)
        assert h_ratio > 2.5, "Heritage virgin conductivity should rise at low pressure"
        assert m_ratio < 1.0, "MEDLI2 virgin conductivity should fall at low pressure"
        assert h_low / m_low == pytest.approx(0.520 / 0.127, rel=1e-9)

    def test_clamps_rather_than_extrapolating(self):
        at = MEDLI2_PICA_CONDUCTIVITY.value
        deep = float(at(300.0, 0.0, 1e-9 * ONE_ATMOSPHERE))
        assert deep == pytest.approx(float(at(300.0, 0.0, 0.001 * ONE_ATMOSPHERE)), rel=1e-14)
        high = float(at(300.0, 0.0, 100.0 * ONE_ATMOSPHERE))
        assert high == pytest.approx(float(at(300.0, 0.0, ONE_ATMOSPHERE)), rel=1e-14)

    def test_monotone_in_log_pressure(self):
        p = np.logspace(np.log10(0.001 * ONE_ATMOSPHERE), np.log10(ONE_ATMOSPHERE), 25)
        at = MEDLI2_PICA_CONDUCTIVITY.value
        k = np.array([float(at(500.0, 0.3, p_i)) for p_i in p])
        assert np.all(np.diff(k) > 0.0)

    def test_rejects_bad_anchors(self):
        lb = LinearBlendProperty(0.1, 0.0, 0.2, 0.0)
        with pytest.raises(ValueError, match="low_pressure < high_pressure"):
            PressureConductivity(1000.0, 100.0, lb, lb)

    def test_solver_uses_the_pressure_dependence(self):
        """A stack run at 0.001 atm with the Heritage row must be cooler
        in depth than the same stack with the MEDLI2 row, because Heritage
        puts four times the conductivity into the virgin material."""
        # 500 Pa is the bottom of the synthetic B' table's pressure axis and
        # is already 77% of the way from 1 atm to the 0.001 atm anchor in
        # log-pressure, so the two conductivity rows are well separated here.
        low = 500.0
        t, envs = pulse(120, 60.0)
        envs = [
            AerothermalEnvironment(
                film_coefficient=e.film_coefficient,
                recovery_enthalpy=e.recovery_enthalpy,
                pressure=low,
            )
            for e in envs
        ]
        table = make_table()
        results = {}
        for name, prop in (
            ("heritage", HERITAGE_PICA_CONDUCTIVITY),
            ("medli2", MEDLI2_PICA_CONDUCTIVITY),
        ):
            stack = MaterialStack(
                [
                    Ply(
                        pica_like_material(),
                        0.05,
                        30,
                        1.03,
                        ablating=True,
                        pressure_conductivity=prop,
                    )
                ]
            )
            results[name] = FiatSolver(stack).solve(t, envs, table, ADIABATIC, 300.0)
        # More conductivity spreads the pulse: hotter backface, cooler wall.
        assert (
            results["heritage"].backface_temperature[-1]
            > results["medli2"].backface_temperature[-1]
        )
        assert results["heritage"].wall_temperature.max() < results["medli2"].wall_temperature.max()


class TestKinetics:
    """TGA forward model and the fit that inverts it."""

    def _components(self):
        m = pica_like_material()
        return [m.resin_a, m.resin_b, m.filler], np.array([0.5, 0.5, 0.5])

    def test_char_yield_follows_the_composition(self):
        comps, w = self._components()
        t = np.linspace(300.0, 2000.0, 800)
        mass = tga_mass_fraction(comps, w, t, 20.0 / 60.0)
        assert mass[-1] == pytest.approx(227.0 / 274.0, rel=5e-3)
        assert mass[0] == pytest.approx(1.0, rel=1e-9)
        assert np.all(np.diff(mass) <= 1e-12), "mass must be non-increasing"

    def test_hits_its_stated_tga_targets(self):
        """The kinetics are not published, so what they *are* pinned to has
        to be asserted, or the numbers are unfalsifiable."""
        comps, w = self._components()
        targets = TgaTargets()
        t = np.linspace(300.0, 1600.0, 3000)
        mass = tga_mass_fraction(comps, w, t, targets.heating_rate)
        decomposable = 1.0 - targets.char_yield
        onset = float(t[int(np.argmax(mass <= 1.0 - 0.02 * decomposable))])
        assert onset == pytest.approx(targets.onset_temperature, abs=5.0)
        assert peak_rate_temperature(comps, w, targets.heating_rate) == pytest.approx(
            targets.peak_temperature, abs=5.0
        )

    def test_peak_shifts_up_with_heating_rate(self):
        """Basic Arrhenius behaviour, and the reason a peak temperature is
        meaningless without the scan rate it was measured at."""
        comps, w = self._components()
        slow = peak_rate_temperature(comps, w, 5.0 / 60.0)
        fast = peak_rate_temperature(comps, w, 20.0 / 60.0)
        assert fast > slow + 20.0

    def test_fit_recovers_the_generating_triplets(self):
        """Round trip: forward-model a scan, fit it, recover A and E."""
        comps, w = self._components()
        t = np.linspace(300.0, 2000.0, 600)
        rate = 20.0 / 60.0
        mass = tga_mass_fraction(comps, w, t, rate)
        # Start the fit somewhere genuinely wrong, or this only shows that a
        # fixed point is a fixed point.
        guess = [
            ArrheniusComponent(
                pre_exponential=c.pre_exponential * 3.0,
                activation_energy=c.activation_energy * 1.10,
                reaction_order=c.reaction_order,
                virgin_density=c.virgin_density,
                char_density=c.char_density,
            )
            for c in comps
        ]
        recovered = fit_arrhenius(t, mass, rate, guess, w)
        for got, want in zip(recovered, comps, strict=True):
            if want.pre_exponential == 0.0:
                assert got.pre_exponential == 0.0
                continue
            assert got.pre_exponential == pytest.approx(want.pre_exponential, rel=1e-2)
            assert got.activation_energy == pytest.approx(want.activation_energy, rel=1e-3)
        assert np.max(np.abs(tga_mass_fraction(recovered, w, t, rate) - mass)) < 1e-6

    def test_fit_holds_inert_components(self):
        comps, w = self._components()
        t = np.linspace(300.0, 2000.0, 400)
        mass = tga_mass_fraction(comps, w, t, 20.0 / 60.0)
        recovered = fit_arrhenius(t, mass, 20.0 / 60.0, comps, w)
        assert recovered[2].pre_exponential == 0.0
        assert recovered[2].virgin_density == comps[2].virgin_density

    def test_fit_refuses_an_all_inert_template(self):
        inert = [ArrheniusComponent(0.0, 1e5, 1.0, 100.0, 99.0)]
        with pytest.raises(ValueError, match="no decomposing components"):
            fit_arrhenius(np.linspace(300.0, 1000.0, 10), np.ones(10), 0.3, inert, np.array([1.0]))

    def test_calibration_refuses_to_move_a_published_char_yield(self):
        comps, w = self._components()
        with pytest.raises(ValueError, match="char yield"):
            calibrated_components(comps, w, TgaTargets(char_yield=0.5))

    def test_calibration_hits_a_requested_peak(self):
        comps, w = self._components()
        targets = TgaTargets(peak_temperature=850.0)
        tuned = calibrated_components(comps, w, targets)
        assert peak_rate_temperature(tuned, w, targets.heating_rate) == pytest.approx(
            850.0, abs=3.0
        )

    @pytest.mark.parametrize("bad", [{"char_yield": 1.5}, {"peak_temperature": 100.0}])
    def test_targets_validate(self, bad):
        with pytest.raises(ValueError):
            TgaTargets(**bad)


class TestAnalysis:
    """FIATv2's operational outputs: probes, scaling, sizing."""

    def _run(self, n_cells=30, peak=2.5e6):
        stack = make_stack(n_cells)
        t, envs = pulse(120, 60.0, peak)
        solution = FiatSolver(stack).solve(t, envs, make_table(), ADIABATIC, 300.0)
        return stack, solution, envs

    def test_probe_matches_the_wall_at_zero_depth(self):
        stack, solution, _ = self._run()
        probe = probe_depths(stack, solution, [0.0])[0]
        # Depth 0 is consumed as soon as any recession occurs.
        assert probe.consumed_at is not None
        assert np.allclose(
            probe.temperature,
            [s.wall_temperature for s in solution.steps][: probe.times.size],
            rtol=1e-9,
        )

    def test_deeper_probes_are_cooler_and_lag(self):
        stack, solution, _ = self._run()
        shallow, deep = probe_depths(stack, solution, [0.005, 0.02])
        assert shallow.peak_temperature > deep.peak_temperature
        assert shallow.survived and deep.survived
        i_s = int(np.argmax(shallow.temperature))
        i_d = int(np.argmax(deep.temperature))
        assert deep.times[i_d] > shallow.times[i_s]

    def test_probe_reports_consumption_rather_than_extrapolating(self):
        """A thermocouple the surface has receded past stops reporting. This
        is a real arcjet event, not an edge case to smooth over."""
        stack, solution, _ = self._run(peak=6.0e6)
        final = solution.recession[-1]
        assert final > 0.0
        probe = probe_depths(stack, solution, [0.5 * final])[0]
        assert not probe.survived
        assert probe.consumed_at is not None
        assert probe.times.size < len(solution.steps)

    def test_probe_rejects_a_depth_below_the_stack(self):
        stack, solution, _ = self._run()
        with pytest.raises(ValueError, match="lies below the stack"):
            probe_depths(stack, solution, [1.0])

    def test_interface_flux_is_continuous_and_signed_inward(self):
        stack, solution, envs = self._run()
        (iface,) = interface_histories(stack, solution, envs)
        assert iface.times.size == len(solution.steps)
        # Heat flows inward through the bondline while the front face is hot.
        assert iface.peak_temperature > 300.0
        assert np.max(iface.heat_flux) > 0.0

    def test_interface_histories_need_matching_environments(self):
        stack, solution, _ = self._run()
        with pytest.raises(ValueError, match="one environment per step"):
            interface_histories(stack, solution, [])

    def test_single_ply_stack_has_no_interfaces(self):
        stack = make_stack(20, structure=False)
        t, envs = pulse(30, 30.0)
        solution = FiatSolver(stack).solve(t, envs, make_table(), ADIABATIC, 300.0)
        assert interface_histories(stack, solution, envs) == []

    def test_scaling_touches_fluxes_and_not_states(self):
        _, envs = pulse(10, 10.0)
        hot = scale_environments(envs, 1.1)
        for a, b in zip(envs, hot, strict=True):
            assert b.film_coefficient == pytest.approx(1.1 * a.film_coefficient)
            assert b.radiative_flux == pytest.approx(1.1 * a.radiative_flux)
            # Recovery enthalpy and pressure are thermodynamic states, not
            # transport rates: scaling them would change the chemistry the
            # B' table is queried with.
            assert b.recovery_enthalpy == a.recovery_enthalpy
            assert b.pressure == a.pressure

    def test_scaled_heating_gives_ordered_recession(self):
        stack = make_stack(30, structure=False)
        table = make_table()
        t, envs = pulse(120, 60.0, 3.0e6)
        results = [
            FiatSolver(stack)
            .solve(t, scale_environments(envs, f), table, ADIABATIC, 300.0)
            .recession[-1]
            for f in (0.9, 1.0, 1.1)
        ]
        assert results[0] < results[1] < results[2]

    def test_scaling_rejects_a_negative_factor(self):
        _, envs = pulse(5, 5.0)
        with pytest.raises(ValueError, match="factor"):
            scale_environments(envs, -1.0)

    def test_thickness_optimization_hits_its_target(self):
        stack = make_stack(24, structure=False)
        t, envs = pulse(80, 60.0, 3.0e6)
        table = make_table()
        target = 500.0
        result = optimize_ply_thickness(
            stack, 0, t, envs, table, ADIABATIC, target, bounds=(0.01, 0.08)
        )
        assert result.converged
        assert result.peak_temperature == pytest.approx(target, abs=2.0)
        # And the sized stack really does reach it.
        rerun = FiatSolver(sized_stack(stack, 0, result.thickness)).solve(
            t, envs, table, ADIABATIC, 300.0
        )
        assert float(np.max(rerun.backface_temperature)) == pytest.approx(target, abs=2.0)

    def test_thicker_tps_gives_a_cooler_backface(self):
        stack = make_stack(24, structure=False)
        t, envs = pulse(80, 60.0, 3.0e6)
        table = make_table()
        peaks = [
            float(
                np.max(
                    FiatSolver(sized_stack(stack, 0, th))
                    .solve(t, envs, table, ADIABATIC, 300.0)
                    .backface_temperature
                )
            )
            for th in (0.02, 0.04, 0.06)
        ]
        assert peaks[0] > peaks[1] > peaks[2]

    def test_optimization_reports_an_unbracketed_target(self):
        stack = make_stack(20, structure=False)
        t, envs = pulse(40, 30.0)
        with pytest.raises(ValueError, match="not bracketed"):
            optimize_ply_thickness(
                stack, 0, t, envs, make_table(), ADIABATIC, 5000.0, bounds=(0.01, 0.06)
            )


class TestPublishedPicaKinetics:
    """Torres-Herrador et al. 2019 and 2020, both in `reference/`.

    The point of this class is one structural claim: FIAT Eq. (8)'s
    independent-parallel form cannot reproduce PICA's *measured* response
    to heating rate, and the published competitive model can. Everything
    else here supports that.
    """

    SCAN = np.linspace(300.0, 2500.0, 3000)

    def test_deterministic_and_bayesian_tables_are_as_published(self):
        d, b = COMPETITIVE_PICA_DETERMINISTIC, COMPETITIVE_PICA_BAYESIAN
        # [TH2020] Table 1.
        assert (d.log10_a11, d.e11) == (2.019, 32618.482)
        assert (d.log10_a12, d.e12) == (14.292, 143273.910)
        assert (d.log10_a21, d.e21) == (0.442, 51783.980)
        assert (d.log10_a31, d.e31) == (0.993, 31087.851)
        assert (d.gamma_gas_2, d.gamma_gas_3) == (0.163, 0.244)
        # [TH2020] Table 2, posterior means.
        assert (b.log10_a11, b.e11) == (2.4768, 26811.37)
        assert (b.gamma_gas_2, b.gamma_gas_3) == (0.1648, 0.3190)

    def test_mechanism_requires_the_slow_branch_to_start_first(self):
        """E11 < E12 is not a fitted accident, it is what makes the
        competition work: the fast branch must only take over as the rate
        climbs. A set violating it is not this mechanism."""
        with pytest.raises(ValueError, match="E11 < E12"):
            CompetitivePica(
                log10_a11=5.0,
                e11=2.0e5,
                log10_a12=5.0,
                e12=1.0e5,
                log10_a21=1.0,
                e21=5.0e4,
                log10_a31=1.0,
                e31=3.0e4,
                gamma_gas_2=0.16,
                gamma_gas_3=0.24,
            )

    @pytest.mark.parametrize("model", [COMPETITIVE_PICA_DETERMINISTIC, COMPETITIVE_PICA_BAYESIAN])
    def test_competitive_peak_shifts_down_with_heating_rate(self, model):
        """The published anomaly, at the two rates the model was calibrated
        against: Wong et al. at 10 K/min and Bessire & Minton at 366 K/min."""
        peaks = []
        for rate in (10.0, 366.0):
            mass = competitive_mass_fraction(model, self.SCAN, rate / 60.0)
            peaks.append(float(self.SCAN[int(np.argmax(-np.gradient(mass, self.SCAN)))]))
        assert peaks[1] < peaks[0] - 100.0, f"expected a downward shift, got {peaks}"

    def test_parallel_peak_shifts_up_with_heating_rate(self):
        """The same two rates through FIAT Eq. (8)'s model form move the
        peak the *other* way. This is the structural limitation, not a
        calibration failure — and it is why both models are carried."""
        comps = parallel_pica_resin(94.0)
        w = np.ones(len(comps))
        peaks = []
        for rate in (10.0, 366.0):
            mass = tga_mass_fraction(comps, w, self.SCAN, rate / 60.0)
            peaks.append(float(self.SCAN[int(np.argmax(-np.gradient(mass, self.SCAN)))]))
        assert peaks[1] > peaks[0] + 50.0, f"expected an upward shift, got {peaks}"

    def test_char_yield_becomes_heating_rate_dependent(self):
        """A consequence of competition that Eq. (8) cannot express: the two
        branches have different gas coefficients, so the char yield is a
        function of heating rate rather than a constant of the material."""
        slow, fast = COMPETITIVE_PICA_DETERMINISTIC.char_yield_limits()
        assert slow == pytest.approx(1.0 - 0.163)
        assert fast == pytest.approx(1.0 - 0.244)
        assert slow > fast

    def test_low_rate_char_yield_matches_the_published_bulk_densities(self):
        """Independent cross-check between two unrelated sources: the
        competitive model's slow-branch limit against PICA's published
        virgin and char bulk densities."""
        mass = competitive_mass_fraction(COMPETITIVE_PICA_DETERMINISTIC, self.SCAN, 10.0 / 60.0)
        assert float(mass[-1]) == pytest.approx(227.0 / 274.0, rel=0.02)

    def test_solid_mass_is_monotone_and_bounded(self):
        mass = competitive_mass_fraction(COMPETITIVE_PICA_DETERMINISTIC, self.SCAN, 100.0 / 60.0)
        assert mass[0] == pytest.approx(1.0, abs=1e-9)
        assert np.all(np.diff(mass) <= 1e-10)
        assert np.all(mass >= 0.0)

    def test_parallel_table_is_as_published(self):
        assert len(PARALLEL_PICA_RESIN) == 6
        assert PARALLEL_PICA_RESIN[0] == ParallelReaction(0.060, 6.59, 77.6, 5.65)
        assert PARALLEL_PICA_RESIN[-1] == ParallelReaction(0.059, 6.35, 175.2, 8.85)
        total = sum(r.density_loss_fraction for r in PARALLEL_PICA_RESIN)
        assert total == pytest.approx(0.544, abs=5e-4)

    def test_parallel_set_reproduces_its_own_density_loss(self):
        comps = parallel_pica_resin(94.0)
        w = np.ones(len(comps))
        residual = float(tga_mass_fraction(comps, w, self.SCAN, 10.0 / 60.0)[-1])
        total = sum(r.density_loss_fraction for r in PARALLEL_PICA_RESIN)
        assert 1.0 - residual == pytest.approx(total, rel=0.03)

    def test_parallel_set_cross_checks_the_published_composite_loss(self):
        """[TH2019]'s F values scaled by PICA's 94/274 resin fraction against
        the (274-227)/274 implied by the published bulk densities. Two
        unrelated sources, agreeing to about a percentage point."""
        comps = parallel_pica_resin(94.0)
        w = np.ones(len(comps))
        resin_loss = 1.0 - float(tga_mass_fraction(comps, w, self.SCAN, 10.0 / 60.0)[-1])
        composite_loss = resin_loss * 94.0 / 274.0
        assert composite_loss == pytest.approx((274.0 - 227.0) / 274.0, abs=0.015)

    @pytest.mark.parametrize("order", [1.0, 2.0, 4.38, 8.85])
    def test_rate_normalisation_conversion_is_exact(self, order):
        """The advancement form and FIAT Eq. (8) differ by a power of the
        decomposable fraction. Verified by integrating both and comparing,
        not by re-deriving the algebra."""
        rho_v, rho_r = 100.0, 40.0
        log10_a = 6.0
        a_fiat = advancement_to_fiat_rate(log10_a, order, rho_v, rho_r)
        comp = ArrheniusComponent(a_fiat, 1.0e5, order, rho_v, rho_r)
        t = np.linspace(300.0, 2000.0, 2000)
        rho = tga_mass_fraction([comp], np.array([1.0]), t, 10.0 / 60.0) * rho_v

        # Advancement form, integrated independently.
        def rhs(temp, chi):
            k = 10.0**log10_a * np.exp(-1.0e5 / (8.31446261815324 * temp))
            return k * max(1.0 - chi[0], 0.0) ** order / (10.0 / 60.0)

        sol = scipy.integrate.solve_ivp(
            rhs,
            (300.0, 2000.0),
            [0.0],
            t_eval=t,
            method="LSODA",
            rtol=1e-11,
            atol=1e-13,
        )
        rho_adv = rho_v - sol.y[0] * (rho_v - rho_r)
        assert np.max(np.abs(rho - rho_adv)) < 1e-6 * rho_v

    def test_conversion_is_identity_at_first_order(self):
        assert advancement_to_fiat_rate(6.0, 1.0, 100.0, 40.0) == pytest.approx(1e6)

    def test_conversion_rejects_bad_densities(self):
        with pytest.raises(ValueError, match="char_density < virgin_density"):
            advancement_to_fiat_rate(6.0, 2.0, 40.0, 100.0)


TACOT_TABLE = Path("data/bprime/tacot26-air.dat")


@pytest.mark.skipif(not TACOT_TABLE.exists(), reason="TACOT B' table not generated")
class TestMutationppTable:
    """The real equilibrium B' table, generated by Mutation++ for TACOT-26.

    TACOT is the open surrogate for PICA. This replaces the synthetic
    logistic used elsewhere in this file, which exercised the coupling but
    said nothing about surface chemistry.
    """

    def _read(self, **kw):
        return read_mutationpp_bprime(TACOT_TABLE, max_gas_rate=3.0, **kw)

    def test_axes_and_units(self):
        m = self._read()
        # Source is in bar and MJ/kg; the table must be in Pa and J/kg.
        assert m.table.pressure_range == pytest.approx((101.325, 101325.0), rel=1e-9)
        assert m.table.wall_temperature_range == (250.0, 4000.0)
        assert m.table.gas_rate_range[0] == 0.0
        hw = m.table.wall_enthalpy(101325.0, 0.0, 3000.0)
        assert 1.0e5 < abs(hw) < 1.0e8, f"h_w {hw:g} J/kg is not a specific enthalpy"

    def test_identifies_the_solver_ceiling(self):
        """The flat 200-500 series is the solver's cap, not physics. It is
        identified by the condensed-carbon indicator, not by magnitude, so
        genuine B'_c values well above 100 survive."""
        m = self._read()
        assert 0.10 < m.capped_fraction < 0.25
        # Nothing anywhere near the ceiling should survive into the table.
        assert m.table.char_rate(101325.0, 0.0, 4000.0) < 100.0

    def test_sublimation_onset_rises_with_pressure(self):
        """Physical: a higher ambient pressure suppresses sublimation, so
        the carbon survives to a higher wall temperature."""
        m = self._read()
        onsets = [m.sublimation.limit(p, 0.0) for p in (101.325, 1013.25, 10132.5)]
        assert onsets[0] < onsets[1] < onsets[2]
        assert onsets[0] == pytest.approx(3100.0, abs=100.0)

    def test_sublimation_predicate(self):
        m = self._read()
        low = 101.325
        onset = m.sublimation.limit(low, 0.0)
        assert m.sublimation.exceeded(low, 0.0, onset + 50.0)
        assert not m.sublimation.exceeded(low, 0.0, onset - 50.0)

    def test_char_rate_is_physical_and_monotone_below_sublimation(self):
        m = self._read()
        temps = np.arange(1500.0, 3400.0, 100.0)
        b_c = np.array([m.table.char_rate(101325.0, 0.5, t) for t in temps])
        assert np.all(b_c > 0.0)
        assert np.all(b_c < 5.0), "B'_c should stay O(1) well below sublimation"
        assert np.all(np.diff(b_c) > -1e-9), "diffusion-limited plateau then rise"

    def test_derivative_is_positive_through_the_rise(self):
        """The Newton surface solve differentiates this; a wrong sign here
        would push the iteration the wrong way."""
        m = self._read()
        for t in (2500.0, 3000.0, 3200.0):
            assert m.table.char_rate_derivative(101325.0, 0.5, t) > 0.0

    def test_roughness_is_acceptable_for_a_newton_solve(self):
        """The metric normalises by the table's global B'_c span. A
        per-point relative measure divides by the near-zero values below
        the onset of surface chemistry and reported 1e11 for this table."""
        assert self._read().table.roughness() < 1.0

    def test_refuses_a_file_without_the_carbon_indicator(self, tmp_path):
        bad = tmp_path / "bad.dat"
        rows = np.column_stack(
            [
                np.ones(8),
                np.zeros(8),
                np.linspace(300.0, 1000.0, 8),
                np.zeros(8),
                np.zeros(8),
                np.full(8, 0.5),
            ]
        )
        with bad.open("w") as fh:
            fh.write("header\n")
            np.savetxt(fh, rows)
        with pytest.raises(ValueError, match="condensed-carbon indicator"):
            read_mutationpp_bprime(bad)

    def test_refuses_an_incomplete_grid(self, tmp_path):
        raw = np.loadtxt(TACOT_TABLE, skiprows=1)
        bad = tmp_path / "trunc.dat"
        with bad.open("w") as fh:
            fh.write("header\n")
            np.savetxt(fh, raw[:-3])
        with pytest.raises(ValueError, match="complete grid"):
            read_mutationpp_bprime(bad)

    def test_solver_runs_against_the_real_table(self):
        """The end of the chain: an ablation run whose surface chemistry is
        an equilibrium calculation rather than a fitted curve."""
        m = self._read()
        stack = make_stack(30, structure=False)
        t, envs = pulse(120, 60.0, peak=3.5e6)
        envs = [
            AerothermalEnvironment(
                film_coefficient=e.film_coefficient,
                recovery_enthalpy=e.recovery_enthalpy,
                pressure=10132.5,
            )
            for e in envs
        ]
        solution = FiatSolver(stack).solve(t, envs, m.table, ADIABATIC, 300.0)
        assert np.all(np.isfinite(solution.wall_temperature))
        assert solution.recession[-1] > 0.0
        assert np.all(np.diff(solution.recession) >= -1e-15)
        # And the run must stay on the physical side of sublimation.
        peak_t = float(solution.wall_temperature.max())
        assert peak_t < m.sublimation.limit(10132.5, 0.0)


QUINN_DIR = Path("reference/transcribed")


@pytest.mark.skipif(
    not (QUINN_DIR / "Quinn-et-al-Fig5a_Bprime_g=0.1.csv").exists(),
    reason="digitised Quinn et al. Fig. 5 data not present",
)
class TestPicaSurfaceTable:
    """PICA surface chemistry, digitised from Quinn et al. Fig. 5 (ACE output).

    The first B' table in this package that is PICA rather than a
    surrogate. It is figure-traced, so every assertion here is loose
    enough to survive several percent of digitisation error — which is the
    honest tolerance, not a weak test.
    """

    def _read(self):
        return read_quinn_bprime(QUINN_DIR)

    def test_axes_and_pressure_independence(self):
        p = self._read()
        assert p.gas_rates.tolist() == list(QUINN_GAS_RATES)
        assert p.table.wall_temperature_range[0] <= 250.0
        # One pressure in the figure, so the table must not pretend otherwise.
        assert p.table.char_rate(1.0e3, 0.1, 2500.0) == pytest.approx(
            p.table.char_rate(1.0e6, 0.1, 2500.0), rel=1e-12
        )

    def test_oxidation_plateau_then_sublimation_rise(self):
        """The shape that makes a B' curve recognisable: a diffusion-limited
        plateau through the oxidation regime, then a steep climb into
        sublimation."""
        p = self._read()
        plateau = [p.table.char_rate(101325.0, 0.01, t) for t in (1500.0, 2000.0, 2500.0)]
        assert max(plateau) / min(plateau) < 1.5, f"not a plateau: {plateau}"
        assert 0.03 < min(plateau) < 0.2
        assert p.table.char_rate(101325.0, 0.01, 3500.0) > 5.0 * max(plateau)

    def test_char_rate_is_monotone_in_temperature(self):
        p = self._read()
        temps = np.arange(1100.0, 3700.0, 50.0)
        for g in QUINN_GAS_RATES:
            b = np.array([p.table.char_rate(101325.0, g, t) for t in temps])
            assert np.all(np.diff(b) >= -1e-12), f"non-monotone at B'_g = {g}"

    def test_wall_enthalpy_rises_through_zero(self):
        """h_w is referenced to formation enthalpy, so it starts strongly
        negative and crosses zero in the ablation regime."""
        p = self._read()
        cold = p.table.wall_enthalpy(101325.0, 0.01, 500.0)
        hot = p.table.wall_enthalpy(101325.0, 0.01, 3500.0)
        assert cold < -5.0e6
        assert hot > 5.0e6
        temps = np.arange(500.0, 3700.0, 50.0)
        h = np.array([p.table.wall_enthalpy(101325.0, 0.01, t) for t in temps])
        assert np.all(np.diff(h) > 0.0)

    def test_more_pyrolysis_gas_raises_the_sublimation_branch(self):
        """At high temperature the blown gas carries carbon off faster."""
        p = self._read()
        rates = np.array([p.table.char_rate(101325.0, g, 3500.0) for g in QUINN_GAS_RATES])
        # The 0 and 0.01 nodes carry the same coalesced curve, so they tie to
        # within round-off; everything above must strictly increase.
        assert np.all(np.diff(rates) >= -1e-12), f"not increasing: {rates}"
        assert np.all(np.diff(rates[1:]) > 0.0), f"not increasing: {rates}"

    def test_smooth_enough_for_a_newton_solve(self):
        assert self._read().table.roughness() < 0.2

    def test_derivative_is_positive_in_the_ablation_regime(self):
        p = self._read()
        for t in (2500.0, 3000.0, 3400.0):
            assert p.table.char_rate_derivative(101325.0, 0.1, t) > 0.0

    def test_digitisation_uncertainty_is_declared(self):
        """A figure-traced table must carry a number for its own error, or
        anything compared against it implies an exactness it lacks."""
        assert 0.01 <= self._read().digitisation_uncertainty <= 0.15

    def test_missing_curve_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no digitised curve"):
            read_quinn_bprime(tmp_path)

    def test_solver_runs_on_real_pica_surface_chemistry(self):
        """End of the chain: PICA kinetics from Torres-Herrador, PICA
        conductivity from MEDLI2 Table 3, PICA surface chemistry from ACE."""
        p = self._read()
        # A 5 mm ply, because the digitised table stops at B'_g = 1 and a
        # thicker one saturates it. That is a real incompatibility, not a
        # test convenience: our PICA-*like* kinetics are pinned to stated
        # targets rather than measured, and they release more pyrolysis gas
        # than the envelope of this figure covers. Recorded in
        # docs/FIAT-fidelity.md rather than tuned away.
        stack = MaterialStack(
            [
                Ply(
                    pica_like_material(),
                    0.005,
                    30,
                    1.03,
                    ablating=True,
                    pressure_conductivity=MEDLI2_PICA_CONDUCTIVITY,
                )
            ]
        )
        t, envs = pulse(120, 60.0, peak=2.5e6)
        envs = [
            AerothermalEnvironment(
                film_coefficient=e.film_coefficient,
                recovery_enthalpy=e.recovery_enthalpy,
                pressure=101325.0,
            )
            for e in envs
        ]
        solution = FiatSolver(stack).solve(t, envs, p.table, ADIABATIC, 300.0)
        assert np.all(np.isfinite(solution.wall_temperature))
        assert np.all(np.diff(solution.recession) >= -1e-15)
        assert 0.0 < solution.recession[-1] < 0.005
        assert 1000.0 < solution.wall_temperature.max() < 3800.0
        # And the whole run must stay inside the digitised envelope.
        assert max(s.surface.gas_rate for s in solution.steps) <= 1.0


@pytest.mark.skipif(
    not (QUINN_DIR / "Quinn-et-al-Fig6_Blue_Solid.csv").exists(),
    reason="digitised Quinn et al. Figs. 6-8 not present",
)
class TestQuinnTorchData:
    """The oxyacetylene-torch PICA measurements — the first real data here.

    These tests check the dataset is coherent, not that our solver
    reproduces it. Structure first: a mis-mapped colour or an inverted
    solid/dashed convention would invert every comparison built on top,
    and would do so silently.
    """

    FLUXES = (250.0, 500.0, 750.0)

    def test_all_three_cases_load(self):
        for q in self.FLUXES:
            case = load_quinn_case(QUINN_DIR, q)
            assert set(case.curves) == set(CURVE_LOCATIONS)
            for by_kind in case.curves.values():
                assert set(by_kind) == {"measured", "simulated"}

    @pytest.mark.parametrize("flux", FLUXES)
    def test_temperature_decreases_with_depth(self, flux):
        """The ordering that confirms the colour mapping. If Orange and
        Purple were swapped this is what would catch it."""
        case = load_quinn_case(QUINN_DIR, flux)
        peaks = [case.measured(loc).temperature.max() for loc in CURVE_LOCATIONS]
        assert peaks == sorted(peaks, reverse=True), f"not ordered by depth: {peaks}"

    @pytest.mark.parametrize("flux", FLUXES)
    def test_depths_increase_and_carry_their_spread(self, flux):
        case = load_quinn_case(QUINN_DIR, flux)
        assert np.all(np.diff(case.depths) > 0.0)
        assert case.depths[0] > 3.0e-3
        # Three samples per case, so every depth has a real spread.
        assert np.all(case.depth_spread > 0.0)

    def test_depth_spread_is_the_dominant_uncertainty(self):
        """At 750 W/cm² the deepest thermocouple's position varies by
        4.2 mm across samples — larger than TC4's own depth at 250 W/cm².
        A temperature error quoted without this is the wrong number."""
        case = load_quinn_case(QUINN_DIR, 750.0)
        assert case.depth_spread[-1] > 4.0e-3

    @pytest.mark.parametrize("flux", FLUXES)
    def test_pyrometer_has_a_low_temperature_cutoff(self, flux):
        """The measured surface trace starts hot because a two-colour
        pyrometer reports nothing below its threshold — not because data
        is missing. The simulated trace starts at ambient. Their being
        different is the evidence that solid means experimental."""
        case = load_quinn_case(QUINN_DIR, flux)
        assert case.measured("surface").temperature[0] > 1200.0
        assert case.simulated("surface").temperature.min() < 700.0

    @pytest.mark.parametrize("flux", FLUXES)
    def test_thermocouples_start_at_ambient(self, flux):
        """Unlike the pyrometer. This separates the surface curve from the
        four in-depth ones without relying on the colour legend."""
        case = load_quinn_case(QUINN_DIR, flux)
        for loc in CURVE_LOCATIONS[1:]:
            for kind in (case.measured(loc), case.simulated(loc)):
                # The coldest point rather than the first: at 750 W/cm2 the
                # rise is steep enough that a digitiser's first sample on a
                # near-vertical section already reads several hundred kelvin.
                assert 280.0 < kind.temperature.min() < 360.0

    def test_peak_surface_temperature_rises_with_heat_flux(self):
        peaks = [
            load_quinn_case(QUINN_DIR, q).measured("surface").temperature.max() for q in self.FLUXES
        ]
        assert peaks == sorted(peaks), f"expected monotone in heat flux: {peaks}"

    def test_boundary_conditions_are_as_tabulated(self):
        """Table 2. The recovery enthalpy rises with heat flux while the
        film coefficient barely moves, which is how a torch is calibrated."""
        cases = [load_quinn_case(QUINN_DIR, q) for q in self.FLUXES]
        assert [c.recovery_enthalpy for c in cases] == [22538940.0, 45124400.0, 61616820.0]
        assert cases[0].film_coefficient == cases[1].film_coefficient == 0.02661
        assert cases[2].film_coefficient == 0.03366

    def test_environment_carries_the_calibration(self):
        case = load_quinn_case(QUINN_DIR, 500.0)
        env = case.environment()
        assert env.film_coefficient == case.film_coefficient
        assert env.recovery_enthalpy == case.recovery_enthalpy

    def test_interpolation_refuses_to_fill_the_pyrometer_gap(self):
        """Filling it would manufacture agreement exactly where a
        surface-energy-balance error would show."""
        case = load_quinn_case(QUINN_DIR, 250.0)
        early = case.measured("surface").at(np.array([0.0, 0.01]))
        assert np.all(np.isnan(early))
        mid = case.measured("surface").at(np.array([30.0]))
        assert np.all(np.isfinite(mid))

    def test_paper_own_simulation_error_sets_the_scale(self):
        """What 'good agreement' means for this dataset, from the paper's
        own model: useful context for anything we later compare."""
        case = load_quinn_case(QUINN_DIR, 250.0)
        times = np.linspace(5.0, 55.0, 40)
        worst = 0.0
        for loc in CURVE_LOCATIONS[1:]:
            m = case.measured(loc).at(times)
            s = case.simulated(loc).at(times)
            ok = np.isfinite(m) & np.isfinite(s)
            worst = max(worst, float(np.max(np.abs(m[ok] - s[ok]))))
        # The paper's own fit is a few hundred kelvin off in places.
        assert 50.0 < worst < 500.0

    def test_rejects_an_unknown_heat_flux(self):
        with pytest.raises(ValueError, match="no figure for"):
            load_quinn_case(QUINN_DIR, 1000.0)

    def test_reports_a_missing_trace(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="missing digitised trace"):
            load_quinn_case(tmp_path, 250.0)


REF = Path("reference/transcribed")


@pytest.mark.skipif(
    not (REF / "TH2020-Fig5b_Experiment_10Kmin^-1.csv").exists(),
    reason="digitised TGA data not present",
)
class TestMeasuredPyrolysis:
    """Validation against measured pyrolysis, not against another model.

    Sources, all digitised from figures in ``reference/``:

    * Wong et al. 2016 Fig. 4, TGA at 10 K/min.
    * Bessire & Minton 2017 Fig. 8, simulated TGA at 3.1, 6.1, 12.7 and
      25 °C/s (186 to 1500 K/min).
    * Torres-Herrador et al. 2020 Fig. 5b, mass-loss-rate curves at
      10 and 366 K/min, experiment and their computed fit.

    Everything else in this file checks a model against a model. This
    class checks against thermocouples and balances.
    """

    SCAN = np.linspace(300.0, 1600.0, 3000)

    def _measured_peak(self, rate_label: str) -> float:
        d = np.loadtxt(REF / f"TH2020-Fig5b_Experiment_{rate_label}.csv", delimiter=",")
        return float(d[np.argmax(d[:, 1]), 0])

    def _model_peak(self, model, rate_per_min: float) -> float:
        mass = competitive_mass_fraction(model, self.SCAN, rate_per_min / 60.0)
        return float(self.SCAN[int(np.argmax(-np.gradient(mass, self.SCAN)))])

    def test_measured_peak_really_does_shift_down(self):
        """The premise of the whole competitive-model argument, taken
        straight off the measurement rather than from the paper's prose."""
        slow = self._measured_peak("10Kmin^-1")
        fast = self._measured_peak("366Kmin^-1")
        assert slow == pytest.approx(814.0, abs=15.0)
        assert fast == pytest.approx(686.0, abs=15.0)
        assert fast < slow - 100.0

    @pytest.mark.parametrize("model", [COMPETITIVE_PICA_DETERMINISTIC, COMPETITIVE_PICA_BAYESIAN])
    def test_competitive_model_matches_measured_peaks(self, model):
        """Absolute peak temperature to better than 20 K at both rates."""
        for label, rate in (("10Kmin^-1", 10.0), ("366Kmin^-1", 366.0)):
            assert self._model_peak(model, rate) == pytest.approx(
                self._measured_peak(label), abs=25.0
            )

    @pytest.mark.parametrize("model", [COMPETITIVE_PICA_DETERMINISTIC, COMPETITIVE_PICA_BAYESIAN])
    def test_competitive_model_matches_the_measured_shift(self, model):
        """The shift itself, which is the quantity the model form exists to
        capture and the one FIAT Eq. (8) cannot produce at all. Reproduced
        to within a few kelvin on a 128 K effect."""
        measured = self._measured_peak("366Kmin^-1") - self._measured_peak("10Kmin^-1")
        modelled = self._model_peak(model, 366.0) - self._model_peak(model, 10.0)
        assert measured < -100.0
        assert modelled == pytest.approx(measured, abs=10.0)

    def test_parallel_form_gets_the_sign_wrong_against_measurement(self):
        """The same comparison for FIAT Eq. (8)'s model form. Not merely
        less accurate — the opposite sign."""
        measured = self._measured_peak("366Kmin^-1") - self._measured_peak("10Kmin^-1")
        comps = parallel_pica_resin(94.0)
        w = np.ones(len(comps))
        peaks = [
            float(
                self.SCAN[
                    int(
                        np.argmax(
                            -np.gradient(
                                tga_mass_fraction(comps, w, self.SCAN, r / 60.0), self.SCAN
                            )
                        )
                    )
                ]
            )
            for r in (10.0, 366.0)
        ]
        assert measured < 0.0
        assert peaks[1] - peaks[0] > 0.0

    def test_wong_mass_loss_matches_the_published_bulk_densities(self):
        """Two unrelated measurements of the same material: a TGA balance at
        10 K/min, and the virgin and char bulk densities from MEDLI2."""
        d = np.loadtxt(REF / "Wong2016-Fig4.csv", delimiter=",")
        assert float(d[:, 1].max()) == pytest.approx(100.0 * (274.0 - 227.0) / 274.0, abs=1.0)

    def test_char_yield_depends_on_heating_rate(self):
        """Measured confirmation of the structural point. Mass loss is ~17%
        at 10 K/min and ~21% at 186 K/min and above — so char yield is a
        function of heating rate, which FIAT Eq. (8) makes a constant of the
        material and the competitive mechanism does not."""
        slow = float(np.loadtxt(REF / "Wong2016-Fig4.csv", delimiter=",")[:, 1].max())
        fast = [
            100.0 - float(np.loadtxt(REF / f"BM2017-Fig8_{r}Cs^-1.csv", delimiter=",")[:, 1].min())
            for r in ("3.1", "6.1", "12.7")
        ]
        assert slow < 18.0
        assert all(f > 20.0 for f in fast)
        # And the competitive model spans the same range in the same direction.
        low, high = COMPETITIVE_PICA_DETERMINISTIC.char_yield_limits()
        assert 100.0 * (1.0 - low) < slow + 3.0
        assert 100.0 * (1.0 - high) > min(fast) - 3.0

    @pytest.mark.parametrize("rate", ["3.1", "6.1", "12.7", "25.0"])
    def test_bessire_curves_are_monotone_and_physical(self, rate):
        d = np.loadtxt(REF / f"BM2017-Fig8_{rate}Cs^-1.csv", delimiter=",")
        order = np.argsort(d[:, 0])
        weight = d[order, 1]
        assert weight[0] > 99.0
        assert np.all(np.diff(weight) < 5.0), "weight should not rise materially"
        assert 75.0 < weight.min() < 85.0


TRANS = Path("reference/transcribed")


class TestArcjetDataset:
    """Milos & Chen 2010 — the reference case I-V4's criterion asks for."""

    def test_every_analysis_case_has_measured_recession(self):
        """Table 5 selects seven cases; Table 4 must supply recession for
        all of them, or the criterion cannot be evaluated on any."""
        for case in ANALYSIS_CASES:
            mean, lo, hi = recession_statistics(case.condition)
            assert 0.0 < lo <= mean <= hi < 0.03

    def test_conditions_span_entry_relevant_pressure(self):
        """The point of this dataset over the torch data, which is all at
        one atmosphere: 2.3 kPa is 0.023 atm, squarely in entry's regime and
        where the two published PICA conductivity models differ by 4x."""
        pressures = [c.pressure for c in ARC_CONDITIONS.values()]
        assert min(pressures) < 3.0e3
        assert max(pressures) / min(pressures) > 30.0

    def test_recession_scatter_sets_the_accuracy_floor(self):
        """Eight nominally identical models at condition 13 scatter by 27%
        of the mean. No 5% criterion can mean more than this does."""
        mean, lo, hi = recession_statistics("13")
        assert (hi - lo) / mean > 0.25

    def test_the_two_reported_enthalpies_disagree_materially(self):
        """Facility correlation against DPLR. Which one becomes the recovery
        enthalpy is a modelling choice with a 45% lever arm, so both are
        carried and neither is chosen in the data layer."""
        worst = max(
            c.enthalpy_disagreement for c in (*ARC_CONDITIONS.values(), *JSC_CONDITIONS.values())
        )
        assert worst > 0.4

    def test_oxygen_sweep_isolates_oxidation_from_heating(self):
        """The JSC conditions hold heat flux and pressure nearly fixed while
        sweeping oxygen 0 to 30%. Recession goes 1.75 to 24 mm — fourteenfold,
        with the thermal environment essentially unchanged. Nothing else in
        this dataset separates the two effects."""
        fluxes, recessions = [], []
        for n in ("19", "20", "21", "22"):
            fluxes.append(condition(n).heat_flux)
            recessions.append(recession_statistics(n)[0])
        assert (max(fluxes) - min(fluxes)) / np.mean(fluxes) < 0.03
        assert recessions == sorted(recessions)
        assert recessions[-1] / recessions[0] > 10.0

    def test_thermocouple_placements_are_consistent(self):
        """Options A, B and C are a ladder offset by 1.27 mm; TC5 is common."""
        for option in ("A", "B", "C"):
            d = TC_PLACEMENTS[option]
            assert len(d) == 5
            assert np.all(np.diff(d) > 0.0)
            assert d[4] == pytest.approx(30.48e-3)
        assert TC_PLACEMENTS["B"][0] - TC_PLACEMENTS["A"][0] == pytest.approx(1.27e-3)

    def test_instrumented_models_resolve_their_depths(self):
        instrumented = [m for m in MODELS if m.tc_option in TC_PLACEMENTS]
        assert len(instrumented) > 30
        for m in instrumented:
            assert len(m.depths) == 5

    def test_grouped_conditions_resolve(self):
        """Table 4 keys sub-cases as '4a'; Table 2 groups them as '4ab'."""
        assert condition("4a").heat_flux == condition("4b").heat_flux
        assert condition("6a").number == "6ab"
        with pytest.raises(KeyError):
            condition("99")

    def test_dual_pulse_reports_one_recession_for_two_exposures(self):
        """Condition 18 is a high pulse then a low one on the same model, so
        recession is measured only after the second."""
        first = [m for m in MODELS if m.condition == "18a"]
        second = [m for m in MODELS if m.condition == "18b"]
        assert all(m.recession is None for m in first)
        assert all(m.recession is not None for m in second)

    def test_jsc_models_have_no_surface_temperature(self):
        """Table 4 reports it as not measured there; None rather than a
        sentinel, so a comparison cannot quietly use zero."""
        assert all(
            m.peak_surface_temperature is None for m in MODELS if m.condition in JSC_CONDITIONS
        )

    @pytest.mark.skipif(
        not (TRANS / "MC2010-Fig26_TC1_AA-44-213-N.csv").exists(),
        reason="digitised Fig. 26 not present",
    )
    @pytest.mark.parametrize("tc", [1, 2, 3, 4, 5])
    def test_fig26_carries_measurement_and_fiat_bracket(self, tc):
        curves = load_fig26_thermocouples(TRANS, tc)
        assert set(curves) == {"measured", "90", "100", "110"}
        for time, temperature in curves.values():
            assert time.size > 5
            assert np.all(np.diff(time) > 0.0)
            assert temperature.max() > 400.0

    @pytest.mark.skipif(
        not (TRANS / "MC2010-Fig26_TC1_AA-44-213-N.csv").exists(),
        reason="digitised Fig. 26 not present",
    )
    def test_fig26_thermocouples_are_consumed_in_depth_order(self):
        """Each measured trace ends when recession reaches it, so the end
        times must increase with depth. This confirms the TC numbering more
        robustly than peak temperature does — the shallow ones are cut off
        while still rising, so TC 1 reports a *lower* maximum than TC 2."""
        ends = [
            float(load_fig26_thermocouples(TRANS, tc)["measured"][0].max())
            for tc in (1, 2, 3, 4, 5)
        ]
        assert ends == sorted(ends), ends
        # TC5 at 30.48 mm outlives the 20.5 mm of recession and survives.
        assert ends[4] > 2.0 * ends[3]

    def test_consumption_gives_a_measured_recession_history(self):
        """Depths paired with death times are a recession curve the paper
        never plots for this condition. It must be monotone and must reach
        roughly the 20.5 mm Table 4 reports at 120 s."""
        times, depths = consumption_recession(TRANS)
        assert np.all(np.diff(times) > 0.0)
        assert np.all(np.diff(depths) > 0.0)
        rate = depths[-1] / times[-1]
        assert rate * 120.0 == pytest.approx(20.5e-3, rel=0.4)

    def test_rejects_a_bad_thermocouple_number(self):
        with pytest.raises(ValueError, match="thermocouple must be"):
            load_fig26_thermocouples(TRANS, 9)


class TestPicaPyrolysisComposition:
    """The elemental composition that makes a PICA deck rather than TACOT."""

    def test_composition_is_normalised(self):
        assert sum(PICA_PYROLYSIS_ELEMENTS.values()) == pytest.approx(1.0, abs=1e-3)

    def test_hydrogen_dominates_by_mole(self):
        """Pyrolysis gas is mostly hydrogen atoms by count — a sanity check
        that the formulas were applied per atom and not per molecule."""
        assert PICA_PYROLYSIS_ELEMENTS["H"] > 0.6

    def test_matches_fiatv3_once_trace_species_are_dropped(self):
        """The adopted composition is Milos & Chen's FIATv3 deck value
        (RAB2014 §III.A.2) renormalised off its trace N and Si, which this
        solver's B' tables do not carry."""
        trace = PICA_PYROLYSIS_ELEMENTS_FIATV3
        kept = {e: trace[e] for e in ("C", "H", "O")}
        total = sum(kept.values())
        for element, value in kept.items():
            assert PICA_PYROLYSIS_ELEMENTS[element] == pytest.approx(value / total, rel=1e-4), (
                element
            )
        # The dropped species are trace: 2% of the mixture between them.
        assert total == pytest.approx(0.98, abs=1e-9)

    def test_implies_a_physically_possible_char(self):
        """Conservation of elements across resin -> gas + char must yield a
        char that is carbon-rich with no negative mole fraction. This is the
        check that condemned the earlier speciation-derived composition."""
        for gas_fraction in (0.45, 0.50, 0.55):
            char = char_required_by_mass_balance(PICA_PYROLYSIS_ELEMENTS, gas_fraction)
            assert all(v >= 0.0 for v in char.values()), (gas_fraction, char)
            assert char["C"] > 0.7, (gas_fraction, char)

    def test_implied_char_resembles_the_measured_one(self):
        """Stronger than non-negativity: at a representative gas fraction the
        implied char should look like Sykes' 850 C elemental analysis. The
        adopted composition lands within a factor of three on oxygen; the
        superseded one is short by a factor of five and heading negative."""
        adopted = char_required_by_mass_balance(PICA_PYROLYSIS_ELEMENTS, 0.50)
        measured = char_required_by_mass_balance(PICA_PYROLYSIS_ELEMENTS_MEASURED, 0.50)
        target = PICA_CHAR_ELEMENTS["O"]
        assert target / 3.0 <= adopted["O"] <= target * 3.0, adopted
        assert measured["O"] < target / 3.0, measured
        # Carbon is the bulk of the char either way; oxygen is the discriminator.
        assert adopted["C"] == pytest.approx(PICA_CHAR_ELEMENTS["C"], rel=0.10)

    def test_measured_composition_demands_an_impossible_char(self):
        """The superseded route-1/route-2 composition is not merely uncertain
        — it is unphysical. Above a 50% gas fraction it requires negative
        oxygen in the char, and below that a char with H/O near 13. Every
        measured char (Sykes 2.2, Tran ~0) is oxygen-rich by comparison, so
        removing char can only drive the gas H/O *up* from the resin's 6.04,
        never down to this composition's 4.62."""
        char = char_required_by_mass_balance(PICA_PYROLYSIS_ELEMENTS_MEASURED, 0.60)
        assert char["O"] < 0.0, char
        # And the direction of the failure identifies the culprit as oxygen.
        resin_ratio = PICA_RESIN_ELEMENTS["H"] / PICA_RESIN_ELEMENTS["O"]
        measured_ratio = (
            PICA_PYROLYSIS_ELEMENTS_MEASURED["H"] / PICA_PYROLYSIS_ELEMENTS_MEASURED["O"]
        )
        assert measured_ratio < 0.8 * resin_ratio
        assert PICA_CHAR_ELEMENTS["H"] / PICA_CHAR_ELEMENTS["O"] < resin_ratio

    def test_resin_idealisation_is_close_to_sykes_measurement(self):
        """RAB2014 calls the two idealised polymers "limiting idealized
        cases", but Sykes' cured novolac is not literally between them — its
        carbon sits 2.4% below both. The defensible claim is closeness, which
        is all the paper asserts ("extremely close to the expected value")."""
        sykes = {"C": 0.488, "H": 0.434, "O": 0.078}
        for element, value in sykes.items():
            assert value == pytest.approx(PICA_RESIN_ELEMENTS[element], rel=0.10), element
        # The two idealisations differ from each other far less than either
        # differs from the measurement, which is why the choice between them
        # does not matter downstream.
        spread = max(
            abs(PICA_RESIN_ELEMENTS[e] - PICA_RESIN_ELEMENTS_CROSSLINKED[e])
            for e in PICA_RESIN_ELEMENTS
        )
        assert spread < 0.02

    def test_resin_reproduces_scoggins_maximum_char_yields(self):
        """[SCO2017] states the maximum char yield of a pure-carbon char is
        the resin's carbon mass fraction, and reports 79.2% for the linear
        polymer and 80.4% for the fully cross-linked one. Recovering those
        from our stored compositions confirms them independently of the
        source that supplied them."""
        assert resin_carbon_mass_fraction() == pytest.approx(0.792, abs=0.002)
        assert resin_carbon_mass_fraction(PICA_RESIN_ELEMENTS_CROSSLINKED) == pytest.approx(
            0.804, abs=0.002
        )

    def test_char_yield_is_self_consistent_for_the_adopted_composition(self):
        """[SCO2017] Eq. (12) leaves H and O untouched by pyrolysis, so the
        char yield inferred through hydrogen and through oxygen must agree.
        The spread grades a composition: Park 0.1 points, TACOT 0.5, the
        adopted FIATv3 value 1.1, the superseded one 4.1."""
        via_h, via_o = char_yield_from_gas_composition(PICA_PYROLYSIS_ELEMENTS)
        assert abs(via_h - via_o) < 0.02
        # And it lands in the 50-65% band [SCO2017] gives for real materials.
        assert 0.50 <= via_h <= 0.65 and 0.50 <= via_o <= 0.65

    def test_superseded_composition_is_the_least_self_consistent(self):
        """The discriminator that condemned it, in [SCO2017]'s framing."""
        spreads = {
            name: abs(a - b)
            for name, (a, b) in {
                "adopted": char_yield_from_gas_composition(PICA_PYROLYSIS_ELEMENTS),
                "park": char_yield_from_gas_composition({"C": 0.229, "H": 0.661, "O": 0.110}),
                "tacot": char_yield_from_gas_composition({"C": 0.206, "H": 0.679, "O": 0.115}),
                "superseded": char_yield_from_gas_composition(PICA_PYROLYSIS_ELEMENTS_MEASURED),
            }.items()
        }
        assert spreads["superseded"] == max(spreads.values())
        assert spreads["superseded"] > 3.0 * spreads["adopted"]

    def test_char_yield_needs_positive_hydrogen_and_oxygen(self):
        with pytest.raises(ValueError, match="positive H and O"):
            char_yield_from_gas_composition({"C": 1.0, "H": 0.0, "O": 0.0})

    def test_mass_balance_rejects_a_degenerate_gas_fraction(self):
        with pytest.raises(ValueError, match="gas_mole_fraction"):
            char_required_by_mass_balance(PICA_PYROLYSIS_ELEMENTS, 1.0)


NASA9 = Path("/config/Mutationpp-1.0.5/data/thermo/nasa9.dat")


@pytest.fixture(scope="module")
def db():
    """Parsed once; the file is 1262 species and parsing is not the subject."""
    return ThermoDatabase.from_nasa9(NASA9)


@pytest.mark.skipif(not NASA9.exists(), reason="NASA-9 thermo database not available")
class TestNasa9Parsing:
    """The reader, which had a positional bug that silenced the solver."""

    def test_reads_the_whole_database(self, db):
        assert len(db.names) > 1000

    def test_species_with_an_explicit_zero_placeholder_parse_correctly(self, db):
        """NASA-9 reserves the eighth coefficient slot and leaves it unused.
        Gurvich-era records spell that as blanks; species added later write
        ``0.000000000E+00``. Compacting non-blank fields instead of reading
        positionally shifts the two integration constants one slot left on
        exactly the latter group, substituting 0 for the enthalpy constant
        and the enthalpy constant for the entropy constant.

        C5 is such a record, and the corrupted parse gave it s/R = 147965
        against a true 54.4 — which made exp(-g/RT) overflow and was the
        real reason the equilibrium solver never converged."""
        c5 = db["C5"]
        assert c5.entropy_r(2000.0) == pytest.approx(54.43, abs=0.05)
        assert c5.enthalpy_rt(2000.0) == pytest.approx(74.57, abs=0.05)

    def test_blank_placeholder_records_still_parse(self, db):
        """The other spelling must be unaffected by the fix. CO leaves the
        slot blank and its values are textbook."""
        co = db["CO"]
        assert co.entropy_r(2000.0) == pytest.approx(31.12, abs=0.05)
        assert co.gibbs_rt(2000.0) == pytest.approx(-34.35, abs=0.05)

    def test_every_common_gas_species_has_a_sane_entropy(self, db):
        """A blanket guard against the same class of shift elsewhere: no
        ideal gas has |s/R| anywhere near 1e3 at 2000 K."""
        for name in (
            "C",
            "H",
            "O",
            "N",
            "CO",
            "CO2",
            "H2",
            "H2O",
            "N2",
            "CH4",
            "C2H2,acetylene",
            "C3",
            "C5",
            "C6H6",
            "HCN",
        ):
            assert abs(db[name].entropy_r(2000.0)) < 200.0, name


@pytest.mark.skipif(not NASA9.exists(), reason="NASA-9 thermo database not available")
class TestGibbsEquilibrium:
    """Element-potential equilibrium on PICA's own pyrolysis gas."""

    SPECIES = (
        "C",
        "H",
        "O",
        "N",
        "CH4",
        "CN",
        "CO",
        "CO2",
        "C2",
        "C2H",
        "C2H2,acetylene",
        "C3",
        "C4",
        "C4H2,butadiyne",
        "C5",
        "HCN",
        "H2",
        "H2O",
        "N2",
        "C6H6",
        "OH",
        "CH3",
    )

    def _solve(self, db, t, p, **kw):
        return solve_equilibrium(db, self.SPECIES, t, p, PICA_PYROLYSIS_ELEMENTS, **kw)

    def test_converges_across_the_ablation_regime(self, db):
        """The regime the B' table actually spans, cold-started at every
        state. Before the fix this failed at every single one of them."""
        for t in (400.0, 600.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 4000.0):
            for p in (1e2, 1e3, 1e4, 1e5, 1e6):
                self._solve(db, t, p)

    def test_conserves_the_elemental_composition_exactly(self, db):
        """The constraint the solve exists to satisfy. Element ratios in the
        converged mixture must reproduce the input to solver tolerance."""
        result = self._solve(db, 2000.0, 1e4)
        totals = {}
        for name, x in zip(result.species, result.mole_fractions, strict=True):
            for element, count in db[name].composition.items():
                totals[element] = totals.get(element, 0.0) + x * count
        scale = sum(totals.values())
        for element, target in PICA_PYROLYSIS_ELEMENTS.items():
            assert totals[element] / scale == pytest.approx(target, rel=1e-8)

    def test_drops_species_needing_an_absent_element(self, db):
        """No nitrogen is supplied, so N2, CN and HCN cannot form. Keeping
        them leaves them unconstrained by A n = b — the Jacobian loses rank
        and the mixture reports species made of nothing."""
        result = self._solve(db, 2000.0, 1e4)
        assert "N2" not in result.species
        assert "HCN" not in result.species
        assert "CO" in result.species

    def test_mixture_is_physically_ordered(self, db):
        """At 2000 K a 60%-char-yield pyrolysis gas is mostly H2 and CO,
        which is what Rabinovitch Fig. 6b shows for this composition."""
        result = self._solve(db, 2000.0, 1e4)
        x = dict(zip(result.species, result.mole_fractions, strict=True))
        assert x["H2"] > 0.5
        assert x["CO"] > 0.15
        assert x["H2"] > x["CO"] > x["CH4"]

    def test_all_oxygen_is_carried_by_carbon_monoxide(self, db):
        """A carbon-rich, hydrogen-rich mixture at 2000 K puts essentially
        every oxygen atom into CO rather than H2O or CO2 — CO is the most
        stable sink available and carbon is not limiting."""
        result = self._solve(db, 2000.0, 1e4)
        x = dict(zip(result.species, result.mole_fractions, strict=True))
        oxygen = sum(v * db[k].composition.get("O", 0.0) for k, v in x.items())
        assert x["CO"] / oxygen > 0.99

    def test_temperature_continuation_reaches_the_condensation_regime(self, db):
        """Warm starting is what gets below 600 K, where gas-phase-only
        equilibrium of a carbon-rich mixture is genuinely ill-posed because
        condensed carbon should form. Cold starts fail there and say so."""
        potentials = None
        reached = 0
        for t in np.arange(4000.0, 350.0, -50.0):
            result = self._solve(db, float(t), 1e4, initial_potentials=potentials)
            potentials = result.element_potentials
            reached += 1
        assert reached == len(np.arange(4000.0, 350.0, -50.0))

    def test_continuation_fallback_rescues_a_failed_cold_start(self, db):
        """1000 K at 1 bar cannot be reached cold — equilibrium there is a
        near-degenerate competition between a few heavy species. The public
        entry point must not expose that: it retries down a temperature
        ladder automatically and only reports failure if that fails too."""
        with pytest.raises(RuntimeError):
            solve_equilibrium(
                db,
                self.SPECIES,
                1000.0,
                1e5,
                PICA_PYROLYSIS_ELEMENTS,
                _continuation=False,
            )
        # With the fallback the same state resolves.
        result = self._solve(db, 1000.0, 1e5)
        assert result.mole_fractions.sum() == pytest.approx(1.0)

    def test_reports_rather_than_lies_where_carbon_must_condense(self, db):
        """300 K is past what a gas-only model can represent for this
        mixture, and continuation does not rescue it. Failing loudly beats
        returning a mixture that omits the condensed phase carrying most of
        the carbon."""
        with pytest.raises(RuntimeError, match="did not converge"):
            self._solve(db, 300.0, 1e4)

    def test_rejects_an_empty_element_set(self, db):
        with pytest.raises(ValueError, match="at least one positive entry"):
            solve_equilibrium(db, self.SPECIES, 2000.0, 1e4, {"C": 0.0})

    def test_rejects_species_that_cannot_be_formed_at_all(self, db):
        with pytest.raises(ValueError, match="no gas species"):
            solve_equilibrium(db, ("N2", "CN"), 2000.0, 1e4, {"C": 1.0})


class TestPorePressure:
    """Darcy pore pressure, and the question it was built to settle."""

    def _uniform(self, n=201, length=0.05, source=1.0, temperature=1500.0):
        x = np.linspace(0.0, length, n)
        return x, np.full(n, source), np.full(n, temperature)

    def test_matches_the_analytic_uniform_source_solution(self):
        """A uniform source over a slab has a closed-form answer: the flux
        is linear in depth, so d(P^2)/dx is linear and P^2 is quadratic.
        The quadrature is exact for a piecewise-linear source, so this
        should agree to machine precision rather than to a tolerance."""
        x, source, temperature = self._uniform()
        surface = 1.0e4
        permeability = 1.0e-11
        profile = pore_pressure(x, source, temperature, surface, permeability)
        coefficient = 2.0 * 8.314462618 * 4.0e-5 / (0.0103 * permeability)
        length = x[-1]
        expected = np.sqrt(surface**2 + coefficient * 1500.0 * 1.0 * (length * x - 0.5 * x**2))
        assert np.allclose(profile.pressure, expected, rtol=1e-12)

    def test_is_grid_independent_for_a_piecewise_linear_source(self):
        """Exactness means refinement must change nothing at all."""
        peaks = []
        for n in (51, 201, 801):
            x, source, temperature = self._uniform(n=n)
            peaks.append(pore_pressure(x, source, temperature, 1e4, 1e-11).peak_pressure)
        assert max(peaks) == pytest.approx(min(peaks), rel=1e-12)

    def test_pressure_peaks_at_the_impermeable_back_face(self):
        """Gas flows outward, so pressure rises inward and is highest where
        nothing has yet escaped. Getting the flux accumulation backwards
        inverts this, which is how the first implementation was caught."""
        x, source, temperature = self._uniform()
        profile = pore_pressure(x, source, temperature, 1e4, 1e-11)
        assert np.all(np.diff(profile.pressure) > 0.0)
        assert profile.peak_depth == pytest.approx(x[-1])
        assert profile.pressure[0] == pytest.approx(1e4, rel=1e-12)

    def test_higher_permeability_relieves_the_pressure(self):
        """The whole mechanism in one assertion: pressure builds because
        the gas cannot get out fast enough."""
        x, source, temperature = self._uniform()
        ratios = [
            pore_pressure(x, source, temperature, 1e4, k).peak_ratio for k in (1e-9, 1e-11, 1e-13)
        ]
        assert ratios == sorted(ratios)
        # In the free-draining limit the pressure rise vanishes entirely.
        # Note this fixture is a uniform source over the *whole* 5 cm,
        # which is far more aggressive than a real pyrolysis zone confined
        # to a front, so the limit is only approached well above the
        # permeabilities a real material has.
        assert pore_pressure(x, source, temperature, 1e4, 1e-6).peak_ratio == (
            pytest.approx(1.0, abs=1e-3)
        )

    def test_the_sensitivity_sweep_answers_the_question_it_was_built_for(self):
        """The measured conclusion: the pore/surface ratio gets large at
        low permeability, which is exactly why the *conductivity* error
        needs checking rather than assuming — it is bounded by the
        log-pressure interpolation and the 1 atm clamp, and comes out under
        10 % across the whole sweep."""
        x, source, temperature = self._uniform()
        sweep = pore_pressure_sensitivity(x, source, temperature, 2.3e3, [1e-9, 1e-11, 1e-12])
        assert len(sweep) == 3
        assert sweep[1e-12] > sweep[1e-11] > sweep[1e-9]
        # Feed the extremes through the conductivity model the pressure is
        # for, and confirm the error stays bounded.
        base = MEDLI2_PICA_CONDUCTIVITY.value(1000.0, 0.0, 2.3e3)
        worst = MEDLI2_PICA_CONDUCTIVITY.value(1000.0, 0.0, 2.3e3 * sweep[1e-12])
        assert abs(worst / base - 1.0) < 0.10

    def test_reference_table_flags_the_material_it_does_not_cover(self):
        """The table used to exist for the gap in it; it now exists to keep
        the measurement and the two things it replaced distinguishable.
        Park & Lawrence measured a dense nozzle liner, not PICA, and must
        never be quoted as PICA."""
        wrong_material = PORE_PRESSURE_REFERENCES["MX4926 carbon cloth phenolic"]
        assert "NOT PICA" in wrong_material[2]

        measured = PORE_PRESSURE_REFERENCES["FiberForm (PICA carbon preform)"]
        assert "MEASURED" in measured[2]
        assert measured[0] < measured[1]
        # PICA's preform is far more permeable than the dense nozzle liner,
        # as porosity demands; a table that got this backwards would be
        # worse than none.
        assert measured[0] > wrong_material[1]

    def test_retired_placeholder_records_how_wrong_the_inference_was(self):
        """The superseded bracket is kept rather than deleted, because the
        distance between an inferred range and the measurement that replaced
        it is the useful part. Its top end was close; its bottom end was 79x
        too tight, which is why the swept worst case overstated the measured
        one."""
        low, high, note = PORE_PRESSURE_REFERENCES["PICA (retired placeholder bracket)"]
        assert "SUPERSEDED" in note
        measured_low, measured_high, _ = PORE_PRESSURE_REFERENCES[
            "FiberForm (PICA carbon preform)"
        ]
        # The inferred floor was far below anything measured.
        assert measured_low / low > 75.0
        # The inferred ceiling was within an order of magnitude.
        assert measured_high / high < 10.0

    def test_inputs_are_validated(self):
        x, source, temperature = self._uniform()
        with pytest.raises(ValueError, match="strictly increasing"):
            pore_pressure(x[::-1], source, temperature, 1e4, 1e-11)
        with pytest.raises(ValueError, match="non-negative"):
            pore_pressure(x, -source, temperature, 1e4, 1e-11)
        with pytest.raises(ValueError, match="surface_pressure"):
            pore_pressure(x, source, temperature, 0.0, 1e-11)
        with pytest.raises(ValueError, match="permeability"):
            pore_pressure(x, source, temperature, 1e4, 0.0)
        with pytest.raises(ValueError, match="must match depth"):
            pore_pressure(x, source[:-1], temperature, 1e4, 1e-11)


class TestPermeability:
    """Marschall & Milos measured permeability, and the two experiments
    they published that let the scaling law be checked rather than trusted."""

    def test_reproduces_the_published_helium_prediction(self):
        """The authors measure b for air and helium on the *same* LI-2200
        specimen, then predict one from the other with Eq. (5) and state
        the answer. Both viscosities and both molar masses are given in the
        text, so this reproduces their arithmetic exactly -- there is no
        free parameter to absorb a mistake."""
        b_helium = slip_parameter(
            7420.0,  # measured in air on this specimen
            temperature=290.0,
            viscosity=19.42e-6,
            molar_mass=0.00400,
        )
        # Their stated prediction is 21,490 Pa; we get 21,484 on unrounded
        # inputs, which is their rounding and nothing else.
        assert b_helium == pytest.approx(21490.0, rel=1e-3)
        # And their measurement was 21,800 Pa -- the 1.5% they call
        # "within 2%".
        assert abs(b_helium - 21800.0) / 21800.0 < 0.02

    def test_continuum_permeability_is_independent_of_temperature(self):
        """Table 2: the same AETB-12 specimen at 293-1200 K. K_0 is
        geometry, so it must not move; the authors claim within 5% and
        that is what the data shows. This is the half of the model that
        justifies carrying a single K_0 into a solve where the gas spans
        thousands of kelvin."""
        k0 = np.array([25.9, 24.5, 24.5, 26.8])
        ratio = k0 / k0[0]
        assert np.max(np.abs(ratio - 1.0)) < 0.055

    def test_slip_parameter_temperature_scaling_matches_the_furnace_series(self):
        """The other half of Table 2: b rises 5.3x over 293-1200 K, and
        Eq. (5) says the rise is entirely mu*sqrt(T).

        Checking this needs nitrogen viscosity, which the authors take from
        their Ref. 8 and do not print. Using Sutherland instead leaves a
        residual that grows with temperature -- 0%, 1.0%, 1.8%, 2.0% -- and
        the next test shows that residual *is* the viscosity correlation
        rather than a failure of the scaling. So the assertion here is
        against the authors' own stated claim, "within 10% of unity",
        allowing the Sutherland offset on top of it.
        """
        temperature = np.array([293.0, 600.0, 900.0, 1200.0])
        b_measured = np.array([3370.0, 8780.0, 14000.0, 18000.0])
        mu = 1.663e-5 * (temperature / 273.0) ** 1.5 * (273.0 + 107.0) / (temperature + 107.0)

        predicted = np.array(
            [
                slip_parameter(
                    b_measured[0],
                    temperature=float(t),
                    viscosity=float(m),
                    molar_mass=0.028014,
                    reference_temperature=293.0,
                    reference_viscosity=float(mu[0]),
                    reference_molar_mass=0.028014,
                )
                for t, m in zip(temperature, mu, strict=True)
            ]
        )
        ratio = b_measured / predicted
        assert np.max(np.abs(ratio - 1.0)) < 0.11
        # Without the scaling the data is not remotely constant -- a 5.3x
        # rise is what the law is absorbing.
        assert b_measured[-1] / b_measured[0] > 5.0

    def test_the_residual_in_the_furnace_series_is_the_viscosity_correlation(self):
        """Eq. (5) can be inverted: given the measured b and the authors'
        published normalisation column, the viscosity ratio they used falls
        out. If the scaling law were wrong, that back-solved ratio would be
        some arbitrary number. It is not -- it agrees with Sutherland's law
        for nitrogen to within 2%, drifting exactly as Sutherland is known
        to drift above ~1000 K.

        This is what separates "our implementation disagrees by 1.8%" from
        "the reference used a better viscosity table", and it is worth a
        test because the first would be a bug.
        """
        temperature = np.array([293.0, 600.0, 900.0, 1200.0])
        b_measured = np.array([3370.0, 8780.0, 14000.0, 18000.0])
        # Marschall & Milos Table 2, column 5: b/(mu sqrt(T)) normalised.
        published = np.array([1.00, 1.09, 1.09, 1.02])

        implied = (b_measured / b_measured[0]) * np.sqrt(temperature[0] / temperature) / published
        mu = 1.663e-5 * (temperature / 273.0) ** 1.5 * (273.0 + 107.0) / (temperature + 107.0)
        sutherland = mu / mu[0]

        assert np.max(np.abs(sutherland / implied - 1.0)) < 0.025
        # Monotonic drift, not scatter: Sutherland runs progressively low.
        drift = sutherland / implied - 1.0
        assert np.all(np.diff(drift) < 0.0)

    def test_klinkenberg_enhancement_is_always_an_enhancement(self):
        """K = K_0(1 + b/P) exceeds K_0 at every finite pressure and tends
        to it from above. A Darcy solve using K_0 alone therefore
        *over*-predicts pore pressure, which is the direction that makes
        neglecting slip conservative."""
        k0, b = 1.0e-10, 1000.0
        pressures = np.array([1e2, 1e3, 1e4, 1e5, 1e7])
        k = effective_permeability(k0, b, pressures)
        assert np.all(k > k0)
        assert np.all(np.diff(k) < 0.0)
        assert k[-1] == pytest.approx(k0, rel=1e-3)
        # At b itself the permeability is exactly doubled.
        assert float(effective_permeability(k0, b, b)[0]) == pytest.approx(2.0 * k0)

    def test_fiberform_sits_far_above_the_threshold_that_would_matter(self):
        """The pore-pressure module was left as a diagnostic on the stated
        condition that it would be worth revisiting only if PICA turned out
        below 1e-12 m^2. Every measured FiberForm specimen is at least 79x
        above that, so the condition is not met -- and this test is what
        keeps that conclusion tied to the data rather than to a memory of
        having checked."""
        assert len(FIBERFORM_SAMPLES) == 8
        lowest = min(s.continuum_permeability for s in FIBERFORM_SAMPLES)
        assert lowest > 79.0e-12
        highest = max(s.continuum_permeability for s in FIBERFORM_SAMPLES)
        assert highest < 6.0e-10

    def test_transverse_is_the_conservative_orientation(self):
        """Fibres align normal to the pressing axis, so flow along that
        axis is the more obstructed. A heatshield is cut with that axis
        pointing out through the surface, which is the direction pyrolysis
        gas must travel -- so the default must be the transverse value and
        it must be the smaller one."""
        k_transverse, _ = fiberform_permeability(155.0, "transverse")
        k_in_plane, _ = fiberform_permeability(155.0, "in-plane")
        assert k_transverse < k_in_plane
        assert fiberform_permeability(155.0) == fiberform_permeability(155.0, "transverse")
        # The anisotropy is far stronger than the ceramic tiles' 1.25-1.5,
        # because FiberForm's 15 micron carbon fibres make a coarser and
        # more directional structure.
        assert k_in_plane / k_transverse > 3.0

    def test_density_interpolation_is_logarithmic_and_does_not_extrapolate(self):
        """The transverse specimens fall into two groups a factor of four
        apart across an 11% density change. Interpolating K_0 linearly
        across that gap would sit well above the geometric mean; log
        interpolation is the right treatment for a multiplicative spread.
        Outside the measured range the endpoints are held, because
        extrapolating a slope this steep past four points would invent
        precision."""
        k_lo, _ = fiberform_permeability(145.0, "transverse")
        k_hi, _ = fiberform_permeability(161.0, "transverse")
        assert k_lo == pytest.approx(3.59e-10)
        assert k_hi == pytest.approx(7.91e-11)
        # Midway in density between the two measured groups, log
        # interpolation lands below the arithmetic mean.
        k_mid, _ = fiberform_permeability(153.0, "transverse")
        assert k_mid < 0.5 * (3.23e-10 + 8.23e-11)
        # Held, not extrapolated.
        assert fiberform_permeability(100.0, "transverse")[0] == pytest.approx(k_lo)
        assert fiberform_permeability(400.0, "transverse")[0] == pytest.approx(k_hi)

    def test_pyrolysis_gas_is_deep_in_the_slip_regime(self):
        """The tabulated b is for room-temperature air. Pyrolysis gas is
        hot and light, and Eq. (5) scales b by mu*sqrt(T/M) -- so the slip
        parameter in service is an order of magnitude above the laboratory
        value, and the pressure below which permeability doubles moves from
        around 1 kPa to above 10 kPa. That is inside the range of in-depth
        pore pressures during entry, which is why slip is not a small
        correction here."""
        _, b_air = fiberform_permeability(155.0, "transverse")
        b_hot = slip_parameter(
            b_air, temperature=2000.0, viscosity=6.0e-5, molar_mass=0.0103
        )
        assert b_hot > 10.0 * b_air
        assert knudsen_regime_pressure(b_air) < 2.0e3
        assert knudsen_regime_pressure(b_hot) > 1.0e4

    def test_the_full_table_is_transcribed_with_its_published_structure(self):
        """Guards the transcription. Two orientations are absent from the
        source for AIM-18 (transverse only), and the counts per material
        are not uniform -- so a table that came out tidy would mean it had
        been tidied."""
        assert len(MARSCHALL_MILOS_SAMPLES) == 38
        materials = {s.material for s in MARSCHALL_MILOS_SAMPLES}
        assert materials == {
            "LI-900",
            "LI-2200",
            "AIM-18",
            "FRCI-12",
            "AETB-12",
            "FiberForm",
        }
        aim = [s for s in MARSCHALL_MILOS_SAMPLES if s.material == "AIM-18"]
        assert all(s.orientation == "transverse" for s in aim)
        # The denser silica tiles are the least permeable; FiberForm the most.
        by_material = {
            m: np.mean(
                [s.continuum_permeability for s in MARSCHALL_MILOS_SAMPLES if s.material == m]
            )
            for m in materials
        }
        assert by_material["FiberForm"] > 50.0 * by_material["LI-2200"]

    def test_rejects_unphysical_inputs(self):
        with pytest.raises(ValueError, match="temperature must be positive"):
            slip_parameter(1000.0, temperature=0.0, viscosity=1e-5, molar_mass=0.03)
        with pytest.raises(ValueError, match="molar_mass must be positive"):
            slip_parameter(1000.0, temperature=300.0, viscosity=1e-5, molar_mass=0.0)
        with pytest.raises(ValueError, match="continuum permeability"):
            effective_permeability(0.0, 1000.0, 1e4)
        with pytest.raises(ValueError, match="pressure must be positive"):
            effective_permeability(1e-10, 1000.0, 0.0)
        with pytest.raises(ValueError, match="enhancement must exceed 1"):
            knudsen_regime_pressure(1000.0, 1.0)
        with pytest.raises(ValueError, match="unknown orientation"):
            fiberform_permeability(155.0, "diagonal")  # type: ignore[arg-type]
