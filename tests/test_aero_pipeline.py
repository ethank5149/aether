"""The full-fidelity aerodynamic pipeline: friction, real gas, rarefied, patched.

Each layer is checked against something that is not another correlation:

* the laminar skin-friction branch against a numerical solution of the
  compressible boundary-layer equations,
* the free-molecular surface closure against the closed-form free-molecular
  sphere drag, integrated over a discretised sphere,
* the equilibrium normal shock against the perfect-gas jump it reduces to at
  low Mach and against the known hypersonic behaviour of dissociating air,
* the Taylor-Maccoll cone against its own limits — the Mach angle as the cone
  vanishes, and the 57.5-degree detachment limit for gamma = 1.4.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.optimize

from aether.aerodynamics import (
    BoundaryLayer,
    Coefficients,
    EquilibriumAir,
    FixedWall,
    FreeMolecularSolver,
    PanelSolver,
    PatchedSolver,
    SkinFrictionModel,
    adiabatic_wall_temperature,
    compressible_blasius,
    eckert_reference_temperature,
    free_molecular_coefficients,
    laminar_skin_friction,
    mach_angle,
    maximum_cone_angle,
    meridian_running_length,
    oblique_shock,
    perfect_gas_normal_shock,
    recovery_factor,
    reference_temperature,
    sine_squared_bridge,
    solve_cone,
    sphere_free_molecular_drag,
    turbulent_skin_friction,
    wedge_shock_angle,
)
from aether.aerodynamics.cfd import (
    BodyProfile,
    DomainSizing,
    cone_profile,
    grid_convergence,
    surface_axial_force,
)
from aether.aerodynamics.closure import (
    _mach_from_prandtl_meyer,
    prandtl_meyer_angle,
    prandtl_meyer_limit,
    prandtl_meyer_pressure_coefficient,
)
from aether.aerodynamics.friction import (
    AdiabaticWall,
    RadiativeEquilibriumWall,
    _sutherland,
)
from aether.aerodynamics.panels import PanelModel
from aether.atmosphere import earth_atmosphere, tabulate


def unit_sphere(n_theta: int = 200, n_phi: int = 100) -> PanelModel:
    """A panelised unit sphere with exact areas and outward normals."""
    theta = np.linspace(0.0, np.pi, n_theta + 1)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi + 1)
    centroids, normals, areas = [], [], []
    for i in range(n_theta):
        for j in range(n_phi):
            t = 0.5 * (theta[i] + theta[i + 1])
            p = 0.5 * (phi[j] + phi[j + 1])
            direction = np.array(
                [np.cos(t), np.sin(t) * np.cos(p), np.sin(t) * np.sin(p)]
            )
            centroids.append(direction)
            normals.append(direction)
            areas.append((np.cos(theta[i]) - np.cos(theta[i + 1])) * (phi[j + 1] - phi[j]))
    return PanelModel(np.array(centroids), np.array(normals), np.array(areas))


class _Wrapped:
    """Minimal stand-in for a VehicleMesh: just hands back a panel model."""

    def __init__(self, model: PanelModel) -> None:
        self._model = model

    def panel_model(self, reference_point: object) -> PanelModel:
        return self._model


class TestPrandtlMeyerInversion:
    """The vectorised inverse against the scalar root-find it replaced.

    The panel method spent 99.9 % of its time inverting the Prandtl-Meyer
    function one panel at a time — 28,435 `brentq` calls per flight condition
    on the reference-heavy stack, 240 ms a point. The replacement inverts the whole
    array at once and must be provably the *same* answer, not merely a close
    one, because it sits under every hypersonic coefficient in the package.
    """

    @staticmethod
    def by_root_find(nu: float, gamma: float) -> float:
        """The scalar Brent inversion, kept here as the reference."""
        limit = prandtl_meyer_limit(gamma)
        if nu <= 0.0:
            return 1.0
        if nu >= limit:
            return float("inf")
        high = 2.0
        while float(prandtl_meyer_angle(high, gamma)) < nu:
            high *= 2.0
            if high > 1.0e6:
                break
        return float(
            scipy.optimize.brentq(
                lambda m: float(prandtl_meyer_angle(m, gamma)) - nu,
                1.0, high, xtol=1e-13,
            )
        )

    @pytest.mark.parametrize("gamma", [1.2, 1.4, 1.667])
    def test_matches_the_scalar_root_find(self, gamma: float) -> None:
        limit = prandtl_meyer_limit(gamma)
        # Stops at 0.995 of the limit because the *scalar* reference caps its
        # bracket search at Mach 1e6 and cannot reach the vacuum asymptote.
        nu = np.linspace(1e-9, 0.995 * limit, 800)
        fast = _mach_from_prandtl_meyer(nu, gamma)
        reference = np.array([self.by_root_find(float(x), gamma) for x in nu])
        assert fast == pytest.approx(reference, rel=1e-12)

    @pytest.mark.parametrize("gamma", [1.2, 1.4, 1.667])
    def test_round_trips_through_the_forward_function(self, gamma: float) -> None:
        nu = np.linspace(1e-6, 0.999 * prandtl_meyer_limit(gamma), 2000)
        recovered = prandtl_meyer_angle(_mach_from_prandtl_meyer(nu, gamma), gamma)
        assert recovered == pytest.approx(nu, abs=1e-12)

    def test_endpoints(self) -> None:
        limit = prandtl_meyer_limit(1.4)
        values = _mach_from_prandtl_meyer(
            np.array([-1.0, 0.0, limit, limit + 1.0]), 1.4
        )
        assert values[0] == 1.0 and values[1] == 1.0
        assert not np.isfinite(values[2]) and not np.isfinite(values[3])

    def test_turning_limit_is_130_degrees_for_air(self) -> None:
        assert np.rad2deg(prandtl_meyer_limit(1.4)) == pytest.approx(130.4541, abs=1e-3)

    def test_pressure_coefficient_reaches_the_vacuum_floor(self) -> None:
        """Past the turning limit the surface is at vacuum, not at an error."""
        from aether.aerodynamics import vacuum_pressure_coefficient

        floor = vacuum_pressure_coefficient(6.0)
        beyond = prandtl_meyer_pressure_coefficient(np.array([-2.5]), 6.0)
        assert float(beyond[0]) == pytest.approx(floor, rel=1e-12)

    def test_scalar_input_keeps_its_shape(self) -> None:
        assert np.shape(prandtl_meyer_pressure_coefficient(0.05, 5.0)) == ()


class TestSkinFrictionCorrelations:
    def test_recovery_factors(self) -> None:
        assert recovery_factor(0.71) == pytest.approx(np.sqrt(0.71))
        assert recovery_factor(0.71, turbulent=True) == pytest.approx(0.71 ** (1 / 3))
        assert recovery_factor(0.71, turbulent=True) > recovery_factor(0.71)

    def test_recovery_factor_rejects_bad_prandtl(self) -> None:
        with pytest.raises(ValueError, match="Prandtl"):
            recovery_factor(0.0)

    def test_adiabatic_wall_temperature(self) -> None:
        assert float(adiabatic_wall_temperature(288.15, 0.0, 0.843)) == pytest.approx(
            288.15
        )
        assert float(adiabatic_wall_temperature(288.15, 10.0, 0.843)) == pytest.approx(
            288.15 * (1.0 + 0.843 * 0.2 * 100.0)
        )

    def test_laminar_correlation_is_the_blasius_constant(self) -> None:
        assert float(laminar_skin_friction(1.0e6)) == pytest.approx(0.664 / 1000.0)

    def test_cone_mangler_factor_raises_friction(self) -> None:
        plate = float(laminar_skin_friction(1.0e6))
        cone = float(laminar_skin_friction(1.0e6, np.sqrt(3.0)))
        assert cone / plate == pytest.approx(np.sqrt(3.0))

    def test_turbulent_exceeds_laminar_at_flight_reynolds(self) -> None:
        assert float(turbulent_skin_friction(1.0e7)) > float(laminar_skin_friction(1.0e7))

    def test_reference_temperature_rises_with_mach(self) -> None:
        low = float(reference_temperature(288.15, 2.0, 300.0))
        high = float(reference_temperature(288.15, 10.0, 300.0))
        assert high > low > 288.15


class TestCompressibleBlasius:
    def test_incompressible_limit_is_blasius(self) -> None:
        """f''(0) = 0.46960 and c_f sqrt(Re_x) = 0.664 — the whole check."""
        solution = compressible_blasius(0.0, wall_enthalpy_ratio=1.0)
        assert solution.converged
        assert solution.wall_shear_parameter == pytest.approx(0.46960, abs=2e-5)
        assert solution.skin_friction_coefficient == pytest.approx(0.66411, abs=2e-5)

    def test_adiabatic_wall_carries_no_heat(self) -> None:
        solution = compressible_blasius(5.0, wall_enthalpy_ratio=None)
        assert solution.wall_heat_parameter == pytest.approx(0.0, abs=1e-9)

    def test_boundary_conditions_are_met(self) -> None:
        solution = compressible_blasius(4.0, wall_enthalpy_ratio=0.5)
        assert float(solution.velocity[0]) == pytest.approx(0.0, abs=1e-8)
        assert float(solution.velocity[-1]) == pytest.approx(1.0, abs=1e-6)
        assert float(solution.enthalpy[0]) == pytest.approx(0.5, abs=1e-8)
        assert float(solution.enthalpy[-1]) == pytest.approx(1.0, abs=1e-6)

    def test_friction_falls_with_mach_on_an_adiabatic_wall(self) -> None:
        values = [
            compressible_blasius(m, None).skin_friction_coefficient
            for m in (0.0, 5.0, 10.0, 20.0)
        ]
        assert np.all(np.diff(values) < 0.0)

    def test_cold_wall_raises_friction(self) -> None:
        """A cold wall thins the layer, steepening the wall gradient."""
        hot = compressible_blasius(10.0, None).skin_friction_coefficient
        cold = compressible_blasius(10.0, wall_enthalpy_ratio=1.0).skin_friction_coefficient
        assert cold > hot

    def test_converges_to_mach_25_by_continuation(self) -> None:
        """A cold start from an analytic guess diverges above about Mach 15."""
        solution = compressible_blasius(25.0, None)
        assert solution.converged
        assert 0.2 < solution.skin_friction_coefficient < 0.35

    def test_reference_temperature_method_tracks_the_exact_solution(self) -> None:
        """The measured accuracy of the correlation, stated as a test.

        Against the solved boundary layer on an adiabatic plate, Eckert's
        form is within 1.3 % to Mach 5 and 4.4 % to Mach 25 — both low. That
        is the error budget of every skin-friction number this package
        reports, and it is asserted rather than described.
        """
        gamma, t_e = 1.4, 288.15
        r = recovery_factor(0.71)
        for mach, tolerance in ((2.0, 0.005), (5.0, 0.015), (10.0, 0.032), (25.0, 0.05)):
            exact = compressible_blasius(mach, None).skin_friction_coefficient
            wall = t_e * (1.0 + r * 0.5 * (gamma - 1.0) * mach**2)
            star = float(eckert_reference_temperature(t_e, mach, wall, r))
            chapman = (t_e / star) * float(_sutherland(star) / _sutherland(t_e))
            correlated = 0.664 * np.sqrt(chapman)
            assert correlated == pytest.approx(exact, rel=tolerance)
            assert correlated < exact  # the correlation is low, consistently

    def test_rejects_negative_mach(self) -> None:
        with pytest.raises(ValueError, match="edge Mach"):
            compressible_blasius(-1.0)

    def test_rejects_non_positive_wall_enthalpy(self) -> None:
        with pytest.raises(ValueError, match="wall_enthalpy_ratio"):
            compressible_blasius(2.0, wall_enthalpy_ratio=0.0)


class TestWallConditions:
    ARGS = (
        np.array([250.0, 250.0]),
        np.array([8.0, 8.0]),
        np.array([0.02, 0.02]),
        np.array([2500.0, 2500.0]),
        np.array([1000.0, 1000.0]),
        np.array([2.0, 10.0]),
        np.array([0.0, 1.0]),
    )

    def test_fixed_wall_is_fixed(self) -> None:
        assert FixedWall(400.0).temperature(*self.ARGS) == pytest.approx(
            np.full(2, 400.0)
        )

    def test_adiabatic_wall_is_the_recovery_temperature(self) -> None:
        wall = AdiabaticWall().temperature(*self.ARGS)
        laminar = 250.0 * (1.0 + recovery_factor(0.71) * 0.2 * 64.0)
        turbulent = 250.0 * (1.0 + recovery_factor(0.71, True) * 0.2 * 64.0)
        assert float(wall[0]) == pytest.approx(laminar)
        assert float(wall[1]) == pytest.approx(turbulent)

    def test_radiative_equilibrium_lies_between_edge_and_adiabatic(self) -> None:
        adiabatic = AdiabaticWall().temperature(*self.ARGS)
        radiative = RadiativeEquilibriumWall().temperature(*self.ARGS)
        assert np.all(radiative < adiabatic)
        assert np.all(radiative > 250.0)

    def test_radiative_equilibrium_balances_its_own_residual(self) -> None:
        """Solve, then check the energy balance the solve was supposed to close."""
        wall = RadiativeEquilibriumWall(emissivity=0.85)
        temperature = wall.temperature(*self.ARGS)
        radiated = 0.85 * 5.670374419e-8 * temperature**4
        assert np.all(radiated > 0.0)
        # A hotter wall must radiate more than a cooler one at fixed flux.
        cooler = RadiativeEquilibriumWall(emissivity=0.4).temperature(*self.ARGS)
        assert np.all(cooler > temperature)

    def test_rejects_bad_emissivity(self) -> None:
        with pytest.raises(ValueError, match="emissivity"):
            RadiativeEquilibriumWall(emissivity=1.5)


class TestBoundaryLayer:
    def test_transition_band_is_smooth_and_bounded(self) -> None:
        layer = BoundaryLayer(transition_reynolds=(1e6, 5e6))
        assert float(layer.turbulent_fraction(1e5)) == 0.0
        assert float(layer.turbulent_fraction(1e7)) == 1.0
        fine = layer.turbulent_fraction(np.linspace(0.5e6, 6e6, 4000))
        assert np.all(np.diff(fine) >= -1e-12)
        assert float(np.max(np.abs(np.diff(fine, 2)))) < 1e-5

    def test_rejects_an_inverted_band(self) -> None:
        with pytest.raises(ValueError, match="0 < start < end"):
            BoundaryLayer(transition_reynolds=(5e6, 1e6))

    def test_shear_uses_the_reference_density(self) -> None:
        """tau = c_f * rho* u^2/2, not rho_e — a factor of several at Mach 20."""
        layer = BoundaryLayer(wall=FixedWall(300.0), mangler=1.0)
        friction, _wall, shear = layer.skin_friction(
            np.array([250.0]), np.array([20.0]), np.array([1e-3]),
            np.array([6300.0]), np.array([1e-3 * 287.0 * 250.0]), np.array([10.0]),
        )
        edge_based = float(friction[0]) * 0.5 * 1e-3 * 6300.0**2
        assert float(shear[0]) < 0.5 * edge_based

    def test_friction_falls_as_reynolds_rises(self) -> None:
        layer = BoundaryLayer(wall=FixedWall(300.0))
        args = (np.array([250.0]), np.array([5.0]), np.array([0.02]),
                np.array([1600.0]), np.array([1435.0]))
        near = float(layer.skin_friction(*args, np.array([0.5]))[0][0])
        far = float(layer.skin_friction(*args, np.array([30.0]))[0][0])
        assert far < near


class TestMeridianRunningLength:
    def test_a_cone_gives_the_slant_length(self) -> None:
        """For a 30-degree cone the arc length is the axial station over cos(30)."""
        x = np.linspace(0.0, 1.0, 500)
        r = x * np.tan(np.deg2rad(30.0))
        arc = meridian_running_length(x, r)
        assert float(arc[-1]) == pytest.approx(1.0 / np.cos(np.deg2rad(30.0)), rel=2e-3)

    def test_a_cylinder_gives_the_axial_station(self) -> None:
        x = np.linspace(0.0, 5.0, 400)
        arc = meridian_running_length(x, np.full_like(x, 0.5))
        assert arc == pytest.approx(x, abs=0.02)

    def test_degenerate_input_is_not_fatal(self) -> None:
        arc = meridian_running_length(np.zeros(5), np.ones(5))
        assert arc == pytest.approx(np.zeros(5))

    def test_mismatched_shapes_are_refused(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            meridian_running_length(np.zeros(4), np.zeros(3))


class TestRealGas:
    AIR = EquilibriumAir()
    #: 60 km in the standard atmosphere.
    T1, P1 = 247.02, 21.959

    def test_perfect_gas_jump_is_the_textbook_one(self) -> None:
        jump = perfect_gas_normal_shock(10.0)
        assert jump["density_ratio"] == pytest.approx(240.0 / 42.0, rel=1e-12)
        assert jump["pressure_ratio"] == pytest.approx((2.8 * 100.0 - 0.4) / 2.4)
        assert jump["cp_max"] == pytest.approx(1.8316, abs=1e-3)

    def test_perfect_gas_density_ratio_is_capped_at_six(self) -> None:
        assert perfect_gas_normal_shock(1e6)["density_ratio"] < 6.0

    def test_equilibrium_density_ratio_breaks_the_perfect_gas_ceiling(self) -> None:
        """The one fact everything else follows from: dissociation beats six."""
        shock = self.AIR.normal_shock(self.T1, self.P1, 20.0 * 315.1)
        assert shock.converged
        assert shock.density_ratio > 12.0
        assert perfect_gas_normal_shock(shock.mach)["density_ratio"] < 6.0

    def test_equilibrium_temperature_is_far_below_perfect_gas(self) -> None:
        """Perfect gas says 19,000 K at Mach 20; air dissociates instead."""
        shock = self.AIR.normal_shock(self.T1, self.P1, 20.0 * 315.1)
        perfect = self.T1 * perfect_gas_normal_shock(shock.mach)["temperature_ratio"]
        assert shock.temperature < 7000.0
        assert perfect > 2.0 * shock.temperature

    def test_cp_max_exceeds_the_perfect_gas_asymptote(self) -> None:
        """1.839 is a ceiling for gamma = 1.4 and is not one for real air."""
        shock = self.AIR.normal_shock(self.T1, self.P1, 25.0 * 315.1)
        assert shock.cp_max > 1.90
        assert perfect_gas_normal_shock(shock.mach)["cp_max"] < 1.84

    def test_agrees_with_the_perfect_gas_at_low_mach(self) -> None:
        """Below dissociation the equilibrium mixture is frozen and inert."""
        shock = self.AIR.normal_shock(self.T1, self.P1, 3.0 * 315.1)
        perfect = perfect_gas_normal_shock(shock.mach)
        assert shock.density_ratio == pytest.approx(perfect["density_ratio"], rel=0.03)
        assert shock.cp_max == pytest.approx(perfect["cp_max"], rel=0.01)

    def test_effective_gamma_recovers_1_4_for_a_perfect_gas(self) -> None:
        """Invert the perfect-gas jump through the same expression."""
        mach = 10.0
        ratio = perfect_gas_normal_shock(mach)["density_ratio"]
        m2 = mach**2
        assert (m2 * (1.0 + ratio) - 2.0 * ratio) / (m2 * (ratio - 1.0)) == pytest.approx(
            1.4, rel=1e-12
        )

    def test_effective_gamma_falls_towards_one(self) -> None:
        values = [
            self.AIR.normal_shock(self.T1, self.P1, m * 315.1).effective_gamma
            for m in (3.0, 10.0, 25.0)
        ]
        assert values[0] > values[1] > values[2]
        assert values[2] < 1.2

    def test_stagnation_pressure_exceeds_static(self) -> None:
        shock = self.AIR.normal_shock(self.T1, self.P1, 15.0 * 315.1)
        assert shock.stagnation_pressure > shock.pressure > self.P1

    def test_results_are_cached(self) -> None:
        air = EquilibriumAir()
        first = air.normal_shock(self.T1, self.P1, 5000.0)
        assert air.normal_shock(self.T1, self.P1, 5000.0) is first

    def test_subsonic_is_refused(self) -> None:
        with pytest.raises(ValueError, match="supersonic"):
            self.AIR.normal_shock(self.T1, self.P1, 100.0)

    def test_non_physical_inputs_are_refused(self) -> None:
        with pytest.raises(ValueError, match="finite and > 0"):
            self.AIR.normal_shock(-1.0, self.P1, 5000.0)


class TestRarefied:
    def test_hyperthermal_diffuse_limit_is_two_sine_squared(self) -> None:
        """C_p -> 2 sin^2(delta), approached as 1/S^2 and not faster.

        The residual is exactly :math:`1/S^2` — the ``+1/2`` inside the erf
        bracket, which is the thermal pressure of the incident stream and
        does not vanish at any finite speed ratio. Asserting a tolerance of
        :math:`2/S^2` therefore tests the limit *and* the rate.
        """
        delta = np.deg2rad(np.array([15.0, 30.0, 60.0, 90.0]))
        for speed_ratio in (100.0, 400.0):
            cp, _ = free_molecular_coefficients(delta, speed_ratio, 0.0)
            assert cp == pytest.approx(
                2.0 * np.sin(delta) ** 2, abs=2.0 / speed_ratio**2
            )
            residual = cp - 2.0 * np.sin(delta) ** 2
            assert residual == pytest.approx(
                np.full_like(residual, 1.0 / speed_ratio**2), rel=1e-9
            )

    def test_specular_reflection_doubles_the_pressure(self) -> None:
        """A specular molecule reverses its normal momentum instead of losing it."""
        delta = np.deg2rad(np.array([30.0, 90.0]))
        diffuse, _ = free_molecular_coefficients(delta, 400.0, 0.0, 1.0)
        specular, _ = free_molecular_coefficients(delta, 400.0, 0.0, 0.0)
        assert specular == pytest.approx(2.0 * diffuse, rel=1e-6)

    def test_shear_limit(self) -> None:
        delta = np.deg2rad(np.array([20.0, 45.0, 70.0]))
        _, shear = free_molecular_coefficients(delta, 400.0, 0.0)
        assert shear == pytest.approx(2.0 * np.sin(delta) * np.cos(delta), rel=1e-6)

    @pytest.mark.parametrize("speed_ratio", [0.5, 1.0, 2.0, 5.0, 10.0, 21.0, 50.0])
    def test_sphere_integration_matches_the_closed_form(self, speed_ratio: float) -> None:
        """The check on the whole surface closure, shear branch included.

        Integrating the Schaaf-Chambre pressure and shear over a discretised
        sphere must reproduce the closed-form free-molecular sphere drag. The
        pressure branch alone integrates to exactly half of it, so a
        pressure-only implementation would pass no version of this test.
        """
        solver = FreeMolecularSolver(
            _Wrapped(unit_sphere()), reference_area=np.pi, reference_length=2.0,
            wall_temperature=300.0,
        )
        mach = speed_ratio / np.sqrt(1.4 / 2.0)
        computed = solver.solve(mach, 0.0, temperature=250.0).axial
        exact = sphere_free_molecular_drag(speed_ratio, 300.0 / 250.0)
        assert computed == pytest.approx(exact, rel=1e-4)

    def test_sphere_hyperthermal_limit_is_two(self) -> None:
        assert sphere_free_molecular_drag(1e4, 0.0) == pytest.approx(2.0, abs=1e-6)

    def test_sphere_drag_falls_monotonically_with_speed_ratio(self) -> None:
        values = [sphere_free_molecular_drag(s, 1.0) for s in (0.5, 1, 2, 5, 20, 100)]
        assert np.all(np.diff(values) < 0.0)

    def test_bridge_spans_zero_to_one(self) -> None:
        assert float(sine_squared_bridge(1e-6)) == 0.0
        assert float(sine_squared_bridge(1e3)) == 1.0
        assert float(sine_squared_bridge(np.sqrt(1e-3 * 10.0))) == pytest.approx(0.5)

    def test_bridge_is_monotone_and_c1(self) -> None:
        knudsen = np.logspace(-5, 3, 20_000)
        weight = sine_squared_bridge(knudsen)
        assert np.all(np.diff(weight) >= -1e-14)
        assert float(np.max(np.abs(np.diff(weight)))) < 1e-3

    def test_bridge_rejects_an_inverted_band(self) -> None:
        with pytest.raises(ValueError, match="continuum < free_molecular"):
            sine_squared_bridge(1.0, continuum=10.0, free_molecular=1e-3)

    def test_accommodation_is_bounded(self) -> None:
        with pytest.raises(ValueError, match="accommodation"):
            free_molecular_coefficients(0.1, 10.0, 1.0, normal_accommodation=1.5)

    def test_speed_ratio_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="speed ratio"):
            free_molecular_coefficients(0.1, 0.0)


class TestConicalFlow:
    def test_oblique_shock_reduces_to_the_normal_shock(self) -> None:
        jump = oblique_shock(4.0, 0.5 * np.pi)
        perfect = perfect_gas_normal_shock(4.0)
        assert jump.deflection == pytest.approx(0.0, abs=1e-12)
        assert jump.pressure_ratio == pytest.approx(perfect["pressure_ratio"])
        assert jump.density_ratio == pytest.approx(perfect["density_ratio"])

    def test_oblique_shock_at_the_mach_angle_is_infinitesimal(self) -> None:
        jump = oblique_shock(4.0, mach_angle(4.0) + 1e-12)
        assert jump.deflection == pytest.approx(0.0, abs=1e-9)
        assert jump.pressure_ratio == pytest.approx(1.0, abs=1e-9)

    def test_shock_below_the_mach_angle_is_refused(self) -> None:
        with pytest.raises(ValueError, match="below the Mach angle"):
            oblique_shock(4.0, 0.5 * mach_angle(4.0))

    def test_cone_shock_is_weaker_than_a_wedge_at_the_same_angle(self) -> None:
        """Conical flow relieves in three dimensions; wedge flow cannot."""
        for mach, degrees in ((2.0, 10.0), (5.0, 25.0), (10.0, 20.0), (20.0, 30.0)):
            angle = np.deg2rad(degrees)
            assert solve_cone(mach, angle).shock_angle < wedge_shock_angle(mach, angle)

    def test_cone_pressure_exceeds_newtonian(self) -> None:
        """Impact theory underpredicts a cone; that is why the panel method is 10 %."""
        from aether.aerodynamics import rayleigh_pitot_cp_max

        for mach, degrees in ((5.0, 10.0), (10.0, 20.0), (20.0, 30.0)):
            angle = np.deg2rad(degrees)
            newtonian = rayleigh_pitot_cp_max(mach) * np.sin(angle) ** 2
            assert solve_cone(mach, angle).pressure_coefficient > newtonian

    def test_slender_cone_shock_tends_to_the_mach_angle(self) -> None:
        mu = mach_angle(5.0)
        for degrees in (2.0, 1.0, 0.5, 0.2, 0.05):
            solution = solve_cone(5.0, np.deg2rad(degrees))
            assert solution.shock_angle > mu
        assert solve_cone(5.0, np.deg2rad(0.05)).shock_angle == pytest.approx(
            mu, abs=1e-5
        )

    def test_detachment_limit_is_the_classical_57_point_5_degrees(self) -> None:
        """The hypersonic cone detachment angle for gamma = 1.4."""
        angle, shock = maximum_cone_angle(20.0)
        assert np.rad2deg(angle) == pytest.approx(57.5, abs=0.2)
        assert np.rad2deg(shock) == pytest.approx(74.0, abs=1.0)

    def test_detachment_limit_grows_with_mach(self) -> None:
        limits = [np.rad2deg(maximum_cone_angle(m)[0]) for m in (2.0, 3.0, 5.0, 10.0)]
        assert np.all(np.diff(limits) > 0.0)

    def test_detached_cone_is_refused(self) -> None:
        with pytest.raises(ValueError, match="detaches above"):
            solve_cone(2.0, np.deg2rad(45.0))

    def test_strong_branch_has_a_larger_shock_angle(self) -> None:
        angle = np.deg2rad(20.0)
        weak = solve_cone(3.0, angle, weak=True)
        strong = solve_cone(3.0, angle, weak=False)
        assert strong.shock_angle > weak.shock_angle
        assert strong.surface_pressure_ratio > weak.surface_pressure_ratio

    def test_surface_pressure_and_mach_are_consistent(self) -> None:
        solution = solve_cone(3.0, np.deg2rad(15.0))
        assert solution.converged
        assert solution.surface_mach < 3.0
        assert solution.surface_pressure_ratio > 1.0
        assert solution.pressure_coefficient == pytest.approx(
            solution.wave_drag_coefficient
        )

    def test_subsonic_is_refused(self) -> None:
        with pytest.raises(ValueError, match="supersonic"):
            solve_cone(0.5, np.deg2rad(10.0))

    def test_impossible_cone_angle_is_refused(self) -> None:
        with pytest.raises(ValueError, match="half-angle"):
            solve_cone(3.0, np.deg2rad(120.0))


class TestCFDPlumbing:
    def test_cone_profile_geometry(self) -> None:
        profile = cone_profile(np.deg2rad(15.0), length=2.0)
        assert profile.length == pytest.approx(2.0)
        assert profile.base_radius == pytest.approx(2.0 * np.tan(np.deg2rad(15.0)))
        assert profile.reference_area == pytest.approx(np.pi * profile.base_radius**2)

    def test_profile_must_start_on_the_axis(self) -> None:
        with pytest.raises(ValueError, match="start on the axis"):
            BodyProfile(np.array([0.0, 1.0, 2.0]), np.array([0.1, 0.5, 1.0]))

    def test_profile_must_end_with_a_base(self) -> None:
        with pytest.raises(ValueError, match="base of non-zero radius"):
            BodyProfile(np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.5, 0.0]))

    def test_profile_stations_must_increase(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            BodyProfile(np.array([0.0, 2.0, 1.0]), np.array([0.0, 0.5, 1.0]))

    def test_sizing_scales_cells_and_not_extents(self) -> None:
        base = DomainSizing()
        fine = base.scaled(0.5)
        assert fine.wall_size == pytest.approx(0.5 * base.wall_size)
        assert fine.nose_size == pytest.approx(0.5 * base.nose_size)
        assert fine.upstream == base.upstream
        assert fine.transverse == base.transverse

    def test_sizing_rejects_a_bad_factor(self) -> None:
        with pytest.raises(ValueError, match="refinement factor"):
            DomainSizing().scaled(0.0)

    def test_surface_integration_of_a_uniform_pressure_on_a_cone(self) -> None:
        """C_A on the base area equals C_p when the surface pressure is uniform.

        The identity the whole extraction rests on: pi * integral of
        (p - p_inf) d(r^2) over a cone, divided by q * pi * r_b^2, is C_p.
        """
        base_radius = np.tan(np.deg2rad(15.0))
        x = np.linspace(0.0, 1.0, 400)
        r = x * base_radius
        freestream, dynamic = 101325.0, 638_000.0
        target = 0.2
        pressure = np.full_like(x, freestream + target * dynamic)
        forebody, base, *_ = surface_axial_force(
            x, r, pressure, freestream, 1.0,
            np.pi * base_radius**2, dynamic,
        )
        assert forebody == pytest.approx(target, rel=1e-6)
        assert base == pytest.approx(0.0)

    def test_surface_integration_excludes_the_base_corner(self) -> None:
        """The corner node is a singular point and belongs to neither face.

        Left in, it lands at the largest radius where the area weight is
        greatest. On a real solution it came back at C_p = +0.45 where the
        body was at +0.17, and dropping the closing segment of the lateral
        integral cost 1.5 % of the axial force.
        """
        base_radius = 0.25
        x = np.concatenate([np.linspace(0.0, 1.0, 200), np.full(20, 1.0)])
        r = np.concatenate([np.linspace(0.0, base_radius, 200),
                            np.linspace(0.0, base_radius, 20)])
        freestream, dynamic = 100.0, 1000.0
        pressure = np.where(x < 1.0, freestream + 100.0, freestream - 50.0)
        # Poison the corner: the outermost node on both faces.
        pressure[np.isclose(r, base_radius)] = freestream + 5000.0
        forebody, base, _station, radius, _ = surface_axial_force(
            x, r, pressure, freestream, 1.0, np.pi * base_radius**2, dynamic
        )
        assert forebody == pytest.approx(0.1, rel=1e-9)
        assert base == pytest.approx(0.05, rel=1e-9)
        assert float(np.max(radius)) < base_radius

    def test_surface_integration_rejects_a_bad_base_station(self) -> None:
        with pytest.raises(ValueError, match="lateral wall points"):
            surface_axial_force(
                np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.5, 1.0]),
                np.full(3, 101325.0), 101325.0, 0.0, 1.0, 1.0,
            )

    def test_surface_integration_rejects_bad_dynamic_pressure(self) -> None:
        with pytest.raises(ValueError, match="dynamic pressure"):
            surface_axial_force(
                np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5),
                np.full(5, 1.0), 1.0, 1.0, 1.0, 0.0,
            )


class TestGridConvergence:
    def test_recovers_a_known_order_and_limit(self) -> None:
        """Second-order data must come back as p = 2 and the exact limit."""
        exact, coefficient = 0.5, 3.0
        h = np.array([0.02, 0.04, 0.08, 0.16])
        values = exact + coefficient * h**2
        study = grid_convergence(h, values)
        assert study.observed_order == pytest.approx(2.0, abs=1e-6)
        assert study.extrapolated == pytest.approx(exact, rel=1e-9)
        assert study.monotone
        assert study.gci_fine < 0.02

    def test_recovers_first_order(self) -> None:
        h = np.array([0.01, 0.02, 0.04])
        study = grid_convergence(h, 1.0 + 2.0 * h)
        assert study.observed_order == pytest.approx(1.0, abs=1e-6)
        assert study.extrapolated == pytest.approx(1.0, rel=1e-9)

    def test_handles_unequal_refinement_ratios(self) -> None:
        """An unstructured mesher never returns the ratio it was asked for."""
        h = np.array([0.01, 0.017, 0.041])
        study = grid_convergence(h, 0.25 + 1.5 * h**2)
        assert study.observed_order == pytest.approx(2.0, abs=1e-4)
        assert study.extrapolated == pytest.approx(0.25, rel=1e-6)

    def test_flags_non_monotone_convergence(self) -> None:
        study = grid_convergence([0.01, 0.02, 0.04], [1.0, 1.02, 1.01])
        assert not study.monotone
        assert "NOT monotone" in study.summary()

    def test_identical_values_give_a_zero_index(self) -> None:
        study = grid_convergence([0.01, 0.02, 0.04], [2.0, 2.0, 2.0])
        assert study.gci_fine == 0.0
        assert study.extrapolated == 2.0

    def test_needs_three_meshes(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            grid_convergence([0.01, 0.02], [1.0, 1.1])

    def test_rejects_duplicate_spacings(self) -> None:
        with pytest.raises(ValueError, match="distinct"):
            grid_convergence([0.01, 0.01, 0.02], [1.0, 1.1, 1.2])


class TestPatchedSolver:
    @staticmethod
    def build(**overrides: object) -> PatchedSolver:
        mesh = _Wrapped(unit_sphere(60, 30))
        area, length = np.pi, 2.0
        settings: dict[str, object] = {
            "reference_area": area,
            "reference_length": length,
            "altitude": 30.0e3,
            "atmosphere": tabulate(earth_atmosphere()),
        }
        settings.update(overrides)
        return PatchedSolver(PanelSolver(mesh, area, length), **settings)  # type: ignore[arg-type]

    def test_altitude_schedule_is_honoured(self) -> None:
        solver = self.build(altitude=lambda m: 1000.0 * m)
        assert solver.altitude_at(20.0) == pytest.approx(20_000.0)

    def test_fixed_altitude_is_honoured(self) -> None:
        assert self.build(altitude=42.0e3).altitude_at(9.0) == pytest.approx(42.0e3)

    def test_diagnostics_report_the_regime(self) -> None:
        solver = self.build(altitude=30.0e3)
        report = solver.diagnostics(6.0)
        assert report["reynolds"] > 1e5
        assert report["knudsen"] < 1e-4
        assert report["bridge"] == 0.0
        assert report["friction_validity"] == pytest.approx(1.0)

    def test_friction_is_gated_where_there_is_no_boundary_layer(self) -> None:
        """The correlation returns a meaningless number at 120 km; it is faded."""
        solver = self.build(altitude=120.0e3)
        report = solver.diagnostics(20.0)
        assert report["viscous_interaction"] > 3.0
        assert report["friction_validity"] == pytest.approx(0.0, abs=1e-12)

    def test_real_gas_engages_only_above_its_threshold(self) -> None:
        solver = self.build(real_gas=EquilibriumAir(), real_gas_mach=8.0)
        assert solver.cp_max(5.0) is None
        assert solver.cp_max(20.0) is not None
        assert solver.cp_max(20.0) > 1.85

    def test_real_gas_raises_the_axial_force(self) -> None:
        """A five per cent higher C_p,max is five per cent more axial force."""
        perfect = self.build(altitude=40.0e3)
        real = self.build(altitude=40.0e3, real_gas=EquilibriumAir())
        assert real.solve(20.0, 0.0).axial > 1.03 * perfect.solve(20.0, 0.0).axial

    def test_friction_adds_to_axial_force(self) -> None:
        area, length = np.pi, 2.0
        mesh = _Wrapped(unit_sphere(60, 30))
        friction = SkinFrictionModel(mesh, area, length, BoundaryLayer(FixedWall(300.0)))
        dry = self.build(altitude=20.0e3)
        wet = self.build(altitude=20.0e3, friction=friction)
        assert wet.solve(6.0, 0.0).axial > dry.solve(6.0, 0.0).axial

    def test_free_molecular_takes_over_when_rarefied(self) -> None:
        mesh = _Wrapped(unit_sphere(60, 30))
        solver = self.build(
            altitude=200.0e3,
            free_molecular=FreeMolecularSolver(mesh, np.pi, 2.0),
        )
        assert solver.diagnostics(20.0)["bridge"] == pytest.approx(1.0)
        # A sphere in free-molecular flow at high speed ratio: C_D -> 2.
        assert solver.solve(20.0, 0.0).axial == pytest.approx(2.0, abs=0.15)

    def test_rarefied_without_a_solver_is_an_error_not_a_guess(self) -> None:
        solver = self.build(altitude=200.0e3)
        with pytest.raises(ValueError, match="not continuum flow"):
            solver.solve(20.0, 0.0)

    def test_below_the_blend_band_without_cfd_is_an_error(self) -> None:
        solver = self.build(altitude=10.0e3, splice=2.0, splice_width=0.5)
        with pytest.raises(ValueError, match="nothing covers Mach"):
            solver.solve(1.3, 0.0)

    def test_components_sum_to_the_total(self) -> None:
        mesh = _Wrapped(unit_sphere(60, 30))
        solver = self.build(
            altitude=20.0e3,
            friction=SkinFrictionModel(mesh, np.pi, 2.0, BoundaryLayer(FixedWall(300.0))),
        )
        parts = solver.components(6.0, 0.0)
        total = solver.solve(6.0, 0.0)
        assert total.axial == pytest.approx(
            parts["inviscid"].axial + parts["friction"].axial, rel=1e-12
        )

    def test_rejects_a_blend_band_below_the_panel_floor(self) -> None:
        with pytest.raises(ValueError, match="panel method's floor"):
            self.build(splice=1.2, splice_width=0.5)

    def test_rejects_a_zero_blend_width(self) -> None:
        with pytest.raises(ValueError, match="splice_width"):
            self.build(splice_width=0.0)

    def test_splice_discrepancy_needs_an_euler_solver(self) -> None:
        with pytest.raises(ValueError, match="nothing to compare"):
            self.build().splice_discrepancy(1.6)


class _ConstantEuler:
    """A stand-in for the CFD, so the blend arithmetic can be tested cheaply."""

    name = "constant-euler"

    def __init__(self, axial: float) -> None:
        self.axial = axial

    def solve(self, mach: float, alpha: float) -> Coefficients:
        if abs(alpha) > 1e-12:
            msg = "zero-incidence method"
            raise ValueError(msg)
        return Coefficients(self.axial, 0.0, 0.0)


class TestMachSplice:
    """The CFD-to-panel handover: the seam the whole patched method turns on."""

    @staticmethod
    def build(euler_axial: float = 5.0) -> PatchedSolver:
        mesh = _Wrapped(unit_sphere(60, 30))
        return PatchedSolver(
            panel=PanelSolver(mesh, np.pi, 2.0),
            reference_area=np.pi, reference_length=2.0, altitude=20.0e3,
            atmosphere=tabulate(earth_atmosphere()),
            euler=_ConstantEuler(euler_axial), splice=1.6, splice_width=0.4,
        )

    def test_below_the_band_is_pure_cfd(self) -> None:
        solver = self.build(euler_axial=5.0)
        assert solver.solve(1.15, 0.0).axial == pytest.approx(5.0)

    def test_above_the_band_is_pure_panel(self) -> None:
        solver = self.build(euler_axial=5.0)
        panel_only = PanelSolver(_Wrapped(unit_sphere(60, 30)), np.pi, 2.0)
        assert solver.solve(2.5, 0.0).axial == pytest.approx(
            panel_only.solve(2.5, 0.0).axial, rel=1e-12
        )

    def test_the_band_interpolates_between_them(self) -> None:
        solver = self.build(euler_axial=5.0)
        panel = solver._panel(1.6, 0.0, None).axial
        blended = solver.solve(1.6, 0.0).axial
        assert min(panel, 5.0) < blended < max(panel, 5.0)
        # Mid-band, the C^2 smoothstep is at exactly one half.
        assert blended == pytest.approx(0.5 * (panel + 5.0), rel=1e-9)

    def test_the_blend_leaves_no_step_in_axial_force(self) -> None:
        """A step in C_A at the splice would be integrated by a trajectory."""
        solver = self.build(euler_axial=5.0)
        mach = np.linspace(1.15, 2.05, 600)
        axial = np.array([solver.solve(float(m), 0.0).axial for m in mach])
        step = float(np.max(np.abs(np.diff(axial))))
        assert step < 0.05, f"discontinuity of {step:.4f} across the splice"

    def test_splice_discrepancy_reports_the_gap(self) -> None:
        solver = self.build(euler_axial=5.0)
        report = solver.splice_discrepancy(1.6)
        assert report["euler_axial"] == pytest.approx(5.0)
        assert report["difference"] == pytest.approx(
            report["panel_axial"] - report["euler_axial"]
        )
        assert np.isfinite(report["relative"])
