"""Aerothermal correlations (Paper II, §4)."""

import numpy as np
import pytest

from aether.aerothermal import (
    FAY_RIDDELL_COEFFICIENT_EXACT_094,
    FAY_RIDDELL_COEFFICIENT_FROM_NUSSELT,
    FAY_RIDDELL_COEFFICIENT_LITERATURE,
    FAY_RIDDELL_COEFFICIENT_SOURCE,
    FAY_RIDDELL_FACTOR_PR071,
    FAY_RIDDELL_NUSSELT_COEFFICIENT,
    TauberSuttonRadiation,
    fay_riddell,
    lees_distribution,
    newtonian_velocity_gradient,
    stefan_recession_rate,
    sutton_graves,
)
from aether.aerothermal.stagnation import (
    LEWIS_EXPONENT_EQUILIBRIUM,
    LEWIS_EXPONENT_FROZEN_CATALYTIC,
    SUTTON_GRAVES_EARTH,
    WallCatalycity,
    catalycity_bracket,
)
from aether.thermal.surface import STEFAN_BOLTZMANN

# representative (generic) stagnation inputs used across tests
FR_ARGS = {
    "edge_density": 0.05,
    "edge_viscosity": 6.0e-5,
    "wall_density": 0.3,
    "wall_viscosity": 4.0e-5,
    "velocity_gradient": 5.0e3,
    "total_enthalpy_edge": 2.0e7,
    "wall_enthalpy": 1.5e6,
    "dissociation_enthalpy": 5.0e6,
}


def synthetic_ts():
    """Synthetic velocity-function surrogate, clearly labeled as such."""
    v = np.linspace(9000.0, 16000.0, 15)
    f = (v / 9000.0) ** 8.5
    return TauberSuttonRadiation(
        v, f, coefficient=100.0,
        provenance="synthetic V^8.5 surrogate for unit tests; NOT Tauber–Sutton table data",
    )


class TestVelocityGradient:
    def test_formula(self):
        dudx = newtonian_velocity_gradient(0.5, 5.0e4, 100.0, 0.05)
        assert float(dudx) == pytest.approx(np.sqrt(2 * (5.0e4 - 100.0) / 0.05) / 0.5)

    def test_radius_scaling(self):
        d1 = newtonian_velocity_gradient(0.2, 5.0e4, 100.0, 0.05)
        d2 = newtonian_velocity_gradient(0.4, 5.0e4, 100.0, 0.05)
        assert float(d1 / d2) == pytest.approx(2.0)

    def test_rejects_nonphysical(self):
        with pytest.raises(ValueError, match="exceed"):
            newtonian_velocity_gradient(0.5, 100.0, 200.0, 0.05)
        with pytest.raises(ValueError):
            newtonian_velocity_gradient(-0.5, 5e4, 100.0, 0.05)


class TestFayRiddell:
    def test_heating_scales_as_sqrt_velocity_gradient(self):
        """q ∝ (du/dx)^{1/2} — hence R_eff^{-1/2} through Eq. 4.2."""
        args = dict(FR_ARGS)
        q1 = fay_riddell(**args)
        args["velocity_gradient"] = 4.0 * FR_ARGS["velocity_gradient"]
        q2 = fay_riddell(**args)
        assert float(q2 / q1) == pytest.approx(2.0, rel=1e-12)

    def test_driving_potential_linear(self):
        args = dict(FR_ARGS)
        q1 = fay_riddell(**args)
        args["wall_enthalpy"] = 0.5 * (
            FR_ARGS["total_enthalpy_edge"] + FR_ARGS["wall_enthalpy"]
        )
        q2 = fay_riddell(**args)
        ratio_expected = (FR_ARGS["total_enthalpy_edge"] - args["wall_enthalpy"]) / (
            FR_ARGS["total_enthalpy_edge"] - FR_ARGS["wall_enthalpy"]
        )
        assert float(q2 / q1) == pytest.approx(ratio_expected, rel=1e-12)

    def test_lewis_exponent_choice_matters_by_percent(self):
        """Paper II: the bracketed factor differs by several percent
        between equilibrium and frozen/catalytic."""
        q_eq = fay_riddell(**FR_ARGS, lewis_exponent=LEWIS_EXPONENT_EQUILIBRIUM)
        q_fr = fay_riddell(**FR_ARGS, lewis_exponent=LEWIS_EXPONENT_FROZEN_CATALYTIC)
        diff = float(q_fr / q_eq) - 1.0
        assert 0.002 < diff < 0.1, f"exponent sensitivity {diff:.4f}"

    def test_bracket_reduces_to_unity_without_dissociation(self):
        args = dict(FR_ARGS)
        args["dissociation_enthalpy"] = 0.0
        q_eq = fay_riddell(**args, lewis_exponent=0.52)
        q_fr = fay_riddell(**args, lewis_exponent=0.63)
        assert float(q_eq) == pytest.approx(float(q_fr), rel=1e-14)

    def test_magnitude_is_reentry_scale(self):
        """Generic peak-heating inputs must land in the MW/m² decade —
        a sanity check on units, not a validation claim."""
        q = float(fay_riddell(**FR_ARGS))
        assert 1.0e5 < q < 1.0e8

    def test_validation(self):
        bad = dict(FR_ARGS)
        bad["wall_enthalpy"] = 3.0e7
        with pytest.raises(ValueError, match="below edge total"):
            fay_riddell(**bad)
        bad = dict(FR_ARGS)
        bad["dissociation_enthalpy"] = 3.0e7
        with pytest.raises(ValueError, match="h_D"):
            fay_riddell(**bad)
        with pytest.raises(ValueError, match="lewis_exponent"):
            fay_riddell(**FR_ARGS, lewis_exponent=1.5)


class TestWallCatalycity:
    """Fay & Riddell's third correlation -- the noncatalytic wall -- and
    why it needed a new argument rather than a new exponent."""

    def test_noncatalytic_bracket_is_unreachable_by_any_lewis_exponent(self):
        """The catalytic bracket is 1 + (Le^b - 1)f and the noncatalytic
        one is 1 - f. Matching them for f > 0 would need Le^b = 0, which no
        finite exponent gives for Le > 0. This is the whole reason
        catalycity is an enum: the old signature could not express the
        case at all, and would have failed silently by producing a number
        close to 1 instead of one close to 1 - f."""
        h0e, h_d = 2.5e7, 1.5e7
        noncatalytic = float(catalycity_bracket(h_d, h0e, WallCatalycity.FROZEN_NONCATALYTIC))
        assert noncatalytic == pytest.approx(0.4)
        # Sweep the whole admissible exponent range: nothing comes close.
        for exponent in np.linspace(0.01, 0.99, 99):
            catalytic = 1.0 + (1.4**exponent - 1.0) * h_d / h0e
            assert catalytic > 1.0
            assert abs(catalytic - noncatalytic) > 0.5

    def test_reproduces_andersons_factor_of_two(self):
        """Anderson Fig. 17.5, curve 2: heat transfer drops 'by more than a
        factor of two' between a catalytic and a noncatalytic wall as the
        boundary layer goes frozen. That is a property of the brackets, and
        it holds once dissociation carries about half the edge enthalpy --
        which is the regime Fay & Riddell's 5800-22,800 ft/s calculations
        cover."""
        h0e = 2.5e7
        ratios = {}
        for fraction in (0.3, 0.5, 0.6, 0.9):
            catalytic = float(
                catalycity_bracket(fraction * h0e, h0e, WallCatalycity.FROZEN_CATALYTIC)
            )
            noncatalytic = float(
                catalycity_bracket(fraction * h0e, h0e, WallCatalycity.FROZEN_NONCATALYTIC)
            )
            ratios[fraction] = catalytic / noncatalytic
        assert ratios[0.3] < 2.0
        assert ratios[0.5] > 2.0
        assert ratios[0.6] > 2.5
        # Monotone in dissociation fraction, as it must be.
        assert sorted(ratios.values()) == list(ratios.values())

    def test_all_three_agree_when_nothing_is_dissociated(self):
        """With h_D = 0 there is no chemical energy to deliver or withhold,
        so catalycity cannot matter and all three correlations must
        coincide exactly. This is the limit that says the bracket is about
        chemistry rather than an arbitrary knob."""
        h0e = 2.5e7
        values = [
            float(catalycity_bracket(0.0, h0e, case)) for case in WallCatalycity
        ]
        assert values == pytest.approx([1.0, 1.0, 1.0], rel=1e-15)

    def test_noncatalytic_wall_reduces_the_flux(self):
        """End to end through fay_riddell, not just the bracket: a
        noncatalytic wall must see less heating than a catalytic one at
        identical flow conditions, and the ratio must be exactly the
        bracket ratio because nothing else in the correlation changes."""
        q_catalytic = fay_riddell(**FR_ARGS, catalycity=WallCatalycity.FROZEN_CATALYTIC)
        q_noncatalytic = fay_riddell(
            **FR_ARGS, catalycity=WallCatalycity.FROZEN_NONCATALYTIC
        )
        assert float(q_noncatalytic) < float(q_catalytic)
        expected = float(
            catalycity_bracket(
                FR_ARGS["dissociation_enthalpy"],
                FR_ARGS["total_enthalpy_edge"],
                WallCatalycity.FROZEN_NONCATALYTIC,
            )
            / catalycity_bracket(
                FR_ARGS["dissociation_enthalpy"],
                FR_ARGS["total_enthalpy_edge"],
                WallCatalycity.FROZEN_CATALYTIC,
            )
        )
        assert float(q_noncatalytic / q_catalytic) == pytest.approx(expected, rel=1e-14)

    def test_default_is_unchanged_by_the_new_argument(self):
        """The equilibrium case was the framework's only behaviour before
        catalycity existed. Adding the argument must not have moved it."""
        assert float(fay_riddell(**FR_ARGS)) == pytest.approx(
            float(fay_riddell(**FR_ARGS, lewis_exponent=LEWIS_EXPONENT_EQUILIBRIUM)),
            rel=1e-15,
        )
        assert float(fay_riddell(**FR_ARGS, catalycity=WallCatalycity.EQUILIBRIUM)) == (
            pytest.approx(float(fay_riddell(**FR_ARGS)), rel=1e-15)
        )

    def test_enum_carries_the_published_exponents(self):
        assert WallCatalycity.EQUILIBRIUM.lewis_exponent == LEWIS_EXPONENT_EQUILIBRIUM
        assert (
            WallCatalycity.FROZEN_CATALYTIC.lewis_exponent
            == LEWIS_EXPONENT_FROZEN_CATALYTIC
        )
        # None rather than a placeholder number: the Lewis number genuinely
        # does not appear in Anderson Eq. (17.91).
        assert WallCatalycity.FROZEN_NONCATALYTIC.lewis_exponent is None

    def test_rejects_a_lewis_exponent_for_a_noncatalytic_wall(self):
        """Silently ignoring it would be the dangerous behaviour: the
        caller would believe they had selected a Lewis exponent and get a
        result that does not depend on it."""
        with pytest.raises(ValueError, match="no Lewis exponent"):
            fay_riddell(
                **FR_ARGS,
                catalycity=WallCatalycity.FROZEN_NONCATALYTIC,
                lewis_exponent=0.63,
            )

    def test_bracket_validates_its_inputs(self):
        with pytest.raises(ValueError, match="h_D"):
            catalycity_bracket(3.0e7, 2.5e7)
        with pytest.raises(ValueError, match="h_D"):
            catalycity_bracket(-1.0, 2.5e7)
        with pytest.raises(ValueError, match="lewis"):
            catalycity_bracket(1.0e7, 2.5e7, WallCatalycity.EQUILIBRIUM, lewis=0.0)


class TestSuttonGraves:
    def test_formula_and_constant(self):
        q = sutton_graves(1.0e-3, 0.5, 5000.0)
        assert float(q) == pytest.approx(
            SUTTON_GRAVES_EARTH * np.sqrt(1.0e-3 / 0.5) * 5000.0**3
        )

    def test_velocity_cubed(self):
        assert float(
            sutton_graves(1e-3, 0.5, 6000.0) / sutton_graves(1e-3, 0.5, 3000.0)
        ) == pytest.approx(8.0)

    def test_same_radius_trade_direction_as_fay_riddell(self):
        """Both convective models must decrease with blunting."""
        assert float(sutton_graves(1e-3, 0.8, 5000.0)) < float(
            sutton_graves(1e-3, 0.2, 5000.0)
        )


class TestTauberSutton:
    def test_exponents(self):
        ts = synthetic_ts()
        base = ts.heat_flux(0.5, 1e-3, 12000.0)
        assert float(ts.heat_flux(1.0, 1e-3, 12000.0) / base) == pytest.approx(2.0)
        assert float(ts.heat_flux(0.5, 2e-3, 12000.0) / base) == pytest.approx(
            2.0**1.22, rel=1e-12
        )

    def test_interpolates_table_nodes_exactly(self):
        v = np.linspace(9000.0, 16000.0, 15)
        f = (v / 9000.0) ** 8.5
        ts = synthetic_ts()
        for vi, fi in zip(v, f, strict=True):
            assert float(ts.heat_flux(1.0, 1.0, vi)) == pytest.approx(
                100.0 * fi, rel=1e-12
            )

    def test_opposite_radius_trade_produces_interior_optimum(self):
        """Paper II §4.2: q_conv falls and q_rad rises with R_eff, so the
        total has an interior minimum — the blunting trade. The surrogate
        coefficient is scaled so the two components cross near R = 1 m."""
        rho, v = 3.0e-4, 12000.0
        vv = np.linspace(9000.0, 16000.0, 15)
        ff = (vv / 9000.0) ** 8.5
        q_conv_at_1 = float(sutton_graves(rho, 1.0, v))
        f_at_v = np.interp(v, vv, ff)
        coeff = q_conv_at_1 / (rho**1.22 * f_at_v)  # q_rad(R=1) = q_conv(R=1)
        ts = TauberSuttonRadiation(
            vv, ff, coefficient=coeff,
            provenance="synthetic surrogate scaled for the blunting-trade test",
        )
        radii = np.linspace(0.05, 3.0, 120)
        total = sutton_graves(rho, radii, v) + ts.heat_flux(
            radii, rho, np.full_like(radii, v)
        )
        i_min = int(np.argmin(total))
        assert 0 < i_min < radii.size - 1, "optimum radius must be interior"

    def test_refuses_extrapolation_and_bad_tables(self):
        ts = synthetic_ts()
        with pytest.raises(ValueError, match="outside tabulated"):
            ts.heat_flux(0.5, 1e-3, 8000.0)
        v = np.linspace(9000.0, 16000.0, 15)
        with pytest.raises(ValueError, match="provenance"):
            TauberSuttonRadiation(v, (v / 9000.0) ** 8, coefficient=1.0)
        with pytest.raises(ValueError, match="non-decreasing"):
            TauberSuttonRadiation(
                v, np.linspace(10, 1, 15), coefficient=1.0, provenance="x"
            )


class TestLeesDistribution:
    def test_continuous_at_stagnation_region_boundary(self):
        r_eff = 0.4
        eps = 1e-9
        inner = lees_distribution(1e6, 0.8, r_eff - eps, r_eff)
        outer = lees_distribution(1e6, 0.8, r_eff + eps, r_eff)
        assert float(inner) == pytest.approx(float(outer), rel=1e-6)

    def test_stagnation_region_plateau_and_decay(self):
        x = np.array([0.0, 0.1, 0.4, 1.6, 6.4])
        q = lees_distribution(1e6, np.pi / 2, x, 0.4)
        assert q[0] == q[1] == q[2] == pytest.approx(1e6)
        assert float(q[3]) == pytest.approx(1e6 * 0.5)  # sqrt(0.4/1.6)
        assert float(q[4]) == pytest.approx(1e6 * 0.25)

    def test_leeward_zero(self):
        q = lees_distribution(1e6, np.array([-0.3, 0.0, 0.3]), np.full(3, 1.0), 0.4)
        assert q[0] == 0.0 and q[1] == 0.0 and q[2] > 0.0

    def test_incidence_factor(self):
        q30 = lees_distribution(1e6, np.pi / 6, 2.0, 0.4)
        q90 = lees_distribution(1e6, np.pi / 2, 2.0, 0.4)
        assert float(q30 / q90) == pytest.approx(0.5, rel=1e-12)

    def test_validation(self):
        with pytest.raises(ValueError, match="stagnation_flux"):
            lees_distribution(-1.0, 0.5, 1.0, 0.4)
        with pytest.raises(ValueError, match="running_length"):
            lees_distribution(1e6, 0.5, -1.0, 0.4)


class TestStefanRecession:
    def test_energy_balance(self):
        # T_w chosen so the net flux is positive (re-radiation ~7.7e5 W/m2)
        q, t_w, eps, q_cond = 2.0e6, 2000.0, 0.85, 3.0e5
        rho, dh = 1900.0, 2.0e7
        sdot = stefan_recession_rate(q, t_w, eps, q_cond, rho, dh)
        net = q - eps * STEFAN_BOLTZMANN * t_w**4 - q_cond
        assert float(sdot) == pytest.approx(net / (rho * dh), rel=1e-14)

    def test_irreversibility_clamp(self):
        """Net cooling must give zero recession, not negative."""
        sdot = stefan_recession_rate(1.0e4, 3000.0, 0.9, 0.0, 1900.0, 2.0e7)
        assert float(sdot) == 0.0

    def test_balanced_wall_no_recession(self):
        t_w = 2000.0
        eps = 0.85
        q = eps * STEFAN_BOLTZMANN * t_w**4  # exactly re-radiated
        assert float(stefan_recession_rate(q, t_w, eps, 0.0, 1900.0, 2e7)) == 0.0

    def test_validation(self):
        with pytest.raises(ValueError, match="emissivity"):
            stefan_recession_rate(1e6, 2500.0, 1.5, 0.0, 1900.0, 2e7)
        with pytest.raises(ValueError, match="wall_temperature"):
            stefan_recession_rate(1e6, -1.0, 0.9, 0.0, 1900.0, 2e7)
        with pytest.raises(ValueError, match="material_density"):
            stefan_recession_rate(1e6, 2500.0, 0.9, 0.0, -1.0, 2e7)


class TestPublishedTauberSutton:
    """The transcribed Tauber–Sutton relations (reference/, tauber1991)."""

    def test_table_reproduced_exactly_at_every_node(self):
        from aether.aerothermal import EARTH_VELOCITY_FUNCTION, earth_velocity_function

        for velocity, published in EARTH_VELOCITY_FUNCTION:
            assert float(earth_velocity_function(velocity)) == published

    def test_linear_interpolation_as_the_source_prescribes(self):
        from aether.aerothermal import earth_velocity_function

        # midway between the 11,000 (151) and 11,500 (238) nodes
        assert float(earth_velocity_function(11250.0)) == pytest.approx(194.5)

    def test_table_refuses_extrapolation(self):
        from aether.aerothermal import earth_velocity_function

        with pytest.raises(ValueError, match="tabulated span"):
            earth_velocity_function(8000.0)
        with pytest.raises(ValueError, match="tabulated span"):
            earth_velocity_function(17000.0)

    def test_exponent_caps_match_the_published_conditionals(self):
        """Eq. (2): a <= 0.6 for 1 <= r_n <= 2, a <= 0.5 for 2 < r_n <= 3."""
        from aether.aerothermal import earth_radiative_heating_exponent

        # the low-velocity, low-density corner, where the raw value exceeds both caps
        velocity, density = 10000.0, 6.66e-5
        raw = float(earth_radiative_heating_exponent(velocity, density))
        assert raw > 0.6, "premise: the corner must exceed the caps"
        assert float(earth_radiative_heating_exponent(velocity, density, 0.5)) == raw
        assert float(earth_radiative_heating_exponent(velocity, density, 1.0)) == 0.6
        assert float(earth_radiative_heating_exponent(velocity, density, 2.0)) == 0.6
        assert float(earth_radiative_heating_exponent(velocity, density, 2.5)) == 0.5
        assert float(earth_radiative_heating_exponent(velocity, density, 3.0)) == 0.5

    def test_uncapped_where_the_raw_value_is_already_low(self):
        from aether.aerothermal import earth_radiative_heating_exponent

        raw = float(earth_radiative_heating_exponent(14000.0, 5e-4))
        assert raw < 0.5
        assert float(earth_radiative_heating_exponent(14000.0, 5e-4, 2.5)) == raw

    def test_exponent_always_below_one(self):
        """The source states a < 1 must always be met."""
        from aether.aerothermal import earth_radiative_heating_exponent

        v = np.linspace(10000.0, 16000.0, 25)
        rho = np.geomspace(6.66e-5, 6.31e-4, 25)
        vv, rr = np.meshgrid(v, rho)
        assert np.all(earth_radiative_heating_exponent(vv, rr, 1.0) < 1.0)

    def test_units_are_si_not_w_per_cm2(self):
        """The source returns W/cm²; the SI value must be 1e4 larger."""
        from aether.aerothermal import (
            earth_radiative_heat_flux,
            earth_radiative_heating_exponent,
            earth_velocity_function,
        )

        r_n, rho, v = 1.0, 3.0e-4, 12000.0
        a = float(earth_radiative_heating_exponent(v, rho, r_n))
        expected_wcm2 = 4.736e4 * r_n**a * rho**1.22 * float(earth_velocity_function(v))
        assert float(earth_radiative_heat_flux(r_n, rho, v)) == pytest.approx(
            expected_wcm2 * 1.0e4, rel=1e-12
        )

    def test_validity_envelope_enforced(self):
        from aether.aerothermal import earth_radiative_heat_flux

        for bad in (
            {"nose_radius": 1.0, "density": 3e-4, "velocity": 9500.0},
            {"nose_radius": 1.0, "density": 1e-2, "velocity": 12000.0},
            {"nose_radius": 10.0, "density": 3e-4, "velocity": 12000.0},
        ):
            with pytest.raises(ValueError, match="validity envelope"):
                earth_radiative_heat_flux(**bad)
        # deliberate extrapolation is possible but must be asked for
        assert float(
            earth_radiative_heat_flux(0.1, 3e-4, 12000.0, enforce_envelope=False)
        ) > 0.0

    def test_radiation_exceeds_convection_at_high_speed(self):
        """The regime the paper was written for."""
        from aether.aerothermal import earth_radiative_heat_flux

        rho, r_n = 3.0e-4, 1.0
        q_rad = float(earth_radiative_heat_flux(r_n, rho, 12000.0))
        q_conv = float(sutton_graves(rho, r_n, 12000.0))
        assert q_rad > q_conv

    def test_blunting_trade_has_interior_optimum_on_published_data(self):
        from aether.aerothermal import earth_radiative_heat_flux

        rho, v = 3.0e-4, 12000.0
        radii = np.linspace(0.3, 3.0, 200)
        total = np.asarray(sutton_graves(rho, radii, v)) + np.asarray(
            earth_radiative_heat_flux(radii, np.full_like(radii, rho),
                                      np.full_like(radii, v))
        )
        index = int(np.argmin(total))
        assert 0 < index < radii.size - 1

    def test_mars_correlation(self):
        from aether.aerothermal import mars_radiative_heat_flux

        q = float(mars_radiative_heat_flux(2.0, 5.0e-4, 8000.0))
        expected = 2.35e4 * 2.0**0.526 * (5.0e-4) ** 1.19 * 19.2 * 1.0e4
        assert q == pytest.approx(expected, rel=1e-12)
        with pytest.raises(ValueError, match="validity envelope"):
            mars_radiative_heat_flux(2.0, 5.0e-4, 10000.0)

    def test_heat_transfer_coefficient(self):
        """Eq. (4): C_Hr = q_r / (0.5 rho V^3), dimensionless."""
        from aether.aerothermal import radiative_heat_transfer_coefficient

        rho, v, q = 3.0e-4, 12000.0, 8.5e6
        assert float(radiative_heat_transfer_coefficient(q, rho, v)) == pytest.approx(
            q / (0.5 * rho * v**3)
        )


class TestFayRiddellCoefficientProvenance:
    """The source states its leading constant three times, inconsistently.

    Every constant in Fay & Riddell descends from the single fitted 0.67
    of Eq. (58)/(62), carried at two significant figures. Backing the
    modern Eq. (63) coefficient out of the paper's three statements gives
    0.7684, 0.7654 and 0.76 — a 1.1% band. These tests pin the
    reconstruction and the spread so neither can be quietly "fixed" by
    someone who assumes one of the constants is a typo.
    """

    def test_printed_094_is_the_nusselt_constant_over_prandtl(self) -> None:
        """Eq. (63)'s 0.94 is Eq. (62)'s 0.67 divided by Pr, via Eq. (45).

        This is the check that the whole reconstruction rests on: if it
        holds, the substitution of Eq. (62) into Eq. (45) is the right
        reading of the source, and the (rho_e mu_e)^0.4 (rho_w mu_w)^0.1
        exponent split follows from it rather than from convention.
        """
        implied = FAY_RIDDELL_NUSSELT_COEFFICIENT / 0.71
        assert abs(implied / FAY_RIDDELL_FACTOR_PR071 - 1.0) < 5e-3

    def test_exponents_sum_to_the_half_power_of_rho_mu(self) -> None:
        """0.4 (from the Eq. 62 ratio) + 0.1 (residue of sqrt(rho_w mu_w))."""
        args = dict(FR_ARGS)
        scaled = dict(args)
        for key in ("edge_density", "wall_density"):
            scaled[key] = 4.0 * args[key]
        # Scaling both rho by 4 scales (rho mu)_e^0.4 (rho mu)_w^0.1 by 4^0.5.
        ratio = float(fay_riddell(**scaled)) / float(fay_riddell(**args))
        assert ratio == pytest.approx(2.0, rel=1e-12)

    def test_three_source_statements_disagree_by_about_one_percent(self) -> None:
        readings = [
            FAY_RIDDELL_COEFFICIENT_SOURCE,
            FAY_RIDDELL_COEFFICIENT_EXACT_094,
            FAY_RIDDELL_COEFFICIENT_FROM_NUSSELT,
        ]
        spread = max(readings) / min(readings) - 1.0
        assert 0.005 < spread < 0.02

    def test_back_substitutions_are_arithmetically_correct(self) -> None:
        from_nusselt = FAY_RIDDELL_NUSSELT_COEFFICIENT * 0.71**-0.4
        from_printed = FAY_RIDDELL_FACTOR_PR071 * 0.71**0.6
        assert abs(from_nusselt - FAY_RIDDELL_COEFFICIENT_FROM_NUSSELT) < 5e-5
        assert abs(from_printed - FAY_RIDDELL_COEFFICIENT_EXACT_094) < 5e-5

    def test_literature_value_lies_inside_the_source_spread(self) -> None:
        assert (
            FAY_RIDDELL_COEFFICIENT_SOURCE
            < FAY_RIDDELL_COEFFICIENT_LITERATURE
            < FAY_RIDDELL_COEFFICIENT_FROM_NUSSELT
        )

    def test_coefficient_scales_the_flux_linearly(self) -> None:
        args = dict(FR_ARGS)
        base = float(fay_riddell(**args, coefficient=FAY_RIDDELL_COEFFICIENT_SOURCE))
        exact = float(fay_riddell(**args, coefficient=FAY_RIDDELL_COEFFICIENT_FROM_NUSSELT))
        ratio = FAY_RIDDELL_COEFFICIENT_FROM_NUSSELT / FAY_RIDDELL_COEFFICIENT_SOURCE
        assert exact / base == pytest.approx(ratio, rel=1e-12)
        # The whole provenance question is worth around a percent in flux.
        assert 0.0 < exact / base - 1.0 < 0.02

    def test_default_is_the_literature_value(self) -> None:
        args = dict(FR_ARGS)
        assert float(fay_riddell(**args)) == pytest.approx(
            float(fay_riddell(**args, coefficient=FAY_RIDDELL_COEFFICIENT_LITERATURE)),
            rel=1e-15,
        )

    @pytest.mark.parametrize("bad", [0.0, -0.763, float("nan"), float("inf")])
    def test_rejects_nonphysical_coefficients(self, bad: float) -> None:
        with pytest.raises(ValueError, match="coefficient"):
            fay_riddell(**dict(FR_ARGS), coefficient=bad)
