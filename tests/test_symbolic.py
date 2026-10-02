"""Tests for the symbolic arithmetic: derivations as identities, numerics printed from them.

Two things have to hold for this layer to be worth having, and they are tested
separately. The expressions the derivations reason about must be the field the
rest of the package evaluates -- otherwise an identity proved about them says
nothing about the code. And a routine generated from those expressions must
compute them -- otherwise the speed was bought with the one property that
justified generating it.

A third is tested because it is the easiest to lose: that the checker can
*fail*. A reduction that returned zero for everything would pass every test
below except that one.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest
import scipy.integrate

sp = pytest.importorskip("sympy")

from aether.aerothermal import sutton_graves  # noqa: E402
from aether.certification import NUMPY_OPS  # noqa: E402
from aether.certification.rotating_field import (  # noqa: E402
    EARTH,
    exponential_density,
    point_mass_field,
    rotating_entry_field,
)
from aether.dynamics.attitude import dcm_from_quaternion, quaternion_derivative  # noqa: E402
from aether.symbolic import (  # noqa: E402
    DERIVATIONS,
    STATED_NOT_DERIVED,
    SYMPY_OPS,
    AttitudeSymbols,
    CoupledSymbols,
    Derivation,
    EntrySymbols,
    GlideSymbols,
    LambdifiedField,
    PistonSymbols,
    StripSymbols,
    WavyWall,
    check,
    compile_field,
    find_c_compiler,
    generate_c,
    is_identically_zero,
    lie_derivative,
)
from aether.symbolic import codegen as codegen_module  # noqa: E402
from aether.symbolic.__main__ import point_mass_source  # noqa: E402
from aether.symbolic.thermal import CharringCells, CharringSlab, SurfaceBalance  # noqa: E402

_BETA, _LD, _RHO0, _H_SCALE = 4000.0, 2.5, 1.225, 8500.0
_PARAMETERS = np.array(
    [
        0.4, _BETA, _LD, _RHO0, _H_SCALE,
        EARTH.mu, EARTH.radius, EARTH.j2, EARTH.rotation_rate, EARTH.flattening,
    ]
)

needs_compiler = pytest.mark.skipif(find_c_compiler() is None, reason="no C compiler found")


def _states(count, seed=0):
    rng = np.random.default_rng(seed)
    return [
        np.array(
            [
                EARTH.radius + rng.uniform(30e3, 90e3),
                rng.uniform(-3.0, 3.0),
                rng.uniform(-1.3, 1.3),
                rng.uniform(2000.0, 7800.0),
                rng.uniform(-0.3, 0.3),
                rng.uniform(-3.0, 3.0),
            ]
        )
        for _ in range(count)
    ]


def _reference(state):
    field = point_mass_field(_BETA, _LD, density=exponential_density(_RHO0, _H_SCALE))
    return np.array(field(list(state), _PARAMETERS[0], NUMPY_OPS), dtype=float)


class TestTheSymbolicFieldIsTheSameField:
    def test_the_expressions_evaluate_to_what_the_float_field_returns(self):
        """The identities below are about these expressions, so they had better be the field."""
        s = EntrySymbols()
        planet = s.planet
        constants = {
            planet.mu: EARTH.mu, planet.radius: EARTH.radius,
            planet.j2: EARTH.j2, planet.rotation_rate: EARTH.rotation_rate,
        }
        force = (-3.1, 7.4, 2.2)
        for state in _states(20):
            bind = dict(zip(s.state, state, strict=True)) | constants
            bind |= dict(zip(s.specific_force, force, strict=True))
            symbolic = np.array([float(rate.subs(bind)) for rate in s.rates])
            floats = np.array(rotating_entry_field(list(state), force, NUMPY_OPS), dtype=float)
            assert symbolic == pytest.approx(floats, rel=1e-12)

    def test_it_is_produced_by_evaluating_the_field_not_by_restating_it(self):
        s = EntrySymbols()
        evaluated = rotating_entry_field(s.state, s.specific_force, SYMPY_OPS, planet=s.planet)
        assert list(s.rates) == evaluated

    def test_the_planar_glide_does_not_depend_on_what_it_discards(self):
        """The planar reduction sets longitude, latitude and heading to zero and drops them."""
        s = EntrySymbols()
        classical = EntrySymbols(planet=s.planet.spherical().non_rotating())
        for index in (0, 3, 4):
            rate = classical.rates[index]
            assert rate.free_symbols.isdisjoint({s.longitude, s.latitude, s.heading})


class TestTheDerivationsTheManuscriptReliesOn:
    @pytest.mark.parametrize("derivation", DERIVATIONS, ids=lambda d: d.key)
    def test_the_residual_reduces_to_zero(self, derivation):
        result = check(derivation)
        assert result.passed, (
            f"{derivation.claim} (used in {derivation.where}) did not reduce to "
            f"zero; what was left: {result.remainder}"
        )

    def test_every_derivation_says_where_it_is_used(self):
        """A check nobody can connect to a paragraph protects nothing."""
        keys = [d.key for d in DERIVATIONS]
        assert len(keys) == len(set(keys))
        for derivation in DERIVATIONS:
            assert derivation.where and derivation.claim
            assert derivation.kind in {"identity", "limit"}

    def test_every_label_a_derivation_names_exists_in_the_manuscript(self):
        """The other half of the contract: a renamed equation must not orphan its check.

        ``where`` is prose, but the labels in it are real ``\\label`` names, so
        the link from a check to the paragraph it protects can itself be checked.
        """
        manuscript = pathlib.Path(__file__).resolve().parents[1] / "manuscript" / "proposal"
        if not manuscript.is_dir():
            pytest.skip("the manuscript sources are not part of this checkout")
        source = "".join(path.read_text(encoding="utf-8") for path in manuscript.rglob("*.tex"))
        labels = set(re.findall(r"\\label\{([^}]+)\}", source))
        for derivation in DERIVATIONS:
            named = re.findall(r"[a-z]+:[A-Za-z0-9-]+", derivation.where)
            assert named, f"{derivation.key} names no label in {derivation.where!r}"
            for label in named:
                assert label in labels, (
                    f"{derivation.key} says it is used at {label}, which is not a "
                    f"label in the manuscript"
                )

    def test_every_statement_in_the_manuscript_has_a_check_or_a_reason(self):
        """The contract read from the manuscript's side: nothing displayed goes unaccounted for.

        Every equation, lemma, proposition, remark and assumption the manuscript
        labels is either named by a derivation or listed, with the reason, as
        something that is not algebra. A new equation therefore fails this
        test until it is given one or the other, which is the point: the
        symbolic layer cannot fall behind the manuscript without saying so.
        """
        manuscript = pathlib.Path(__file__).resolve().parents[1] / "manuscript" / "proposal"
        if not manuscript.is_dir():
            pytest.skip("the manuscript sources are not part of this checkout")
        source = "".join(path.read_text(encoding="utf-8") for path in manuscript.rglob("*.tex"))
        statements = {
            label
            for label in re.findall(r"\\label\{([^}]+)\}", source)
            if label.split(":")[0] in {"eq", "lem", "prop", "rem", "asm"}
        }
        backed = {
            label
            for derivation in DERIVATIONS
            for label in re.findall(r"[a-z]+:[A-Za-z0-9-]+", derivation.where)
        }
        unaccounted = sorted(statements - backed - set(STATED_NOT_DERIVED))
        assert not unaccounted, (
            f"the manuscript states {unaccounted} and no derivation names them: give each a "
            f"check, or a line in STATED_NOT_DERIVED saying why it is not one"
        )
        stale = sorted(set(STATED_NOT_DERIVED) - statements)
        assert not stale, f"STATED_NOT_DERIVED explains labels the manuscript lacks: {stale}"
        both = sorted(set(STATED_NOT_DERIVED) & backed)
        assert not both, f"{both} are listed as not derived and are named by a derivation"
        assert all(reason.strip() for reason in STATED_NOT_DERIVED.values())
        # Not vacuous: the manuscript displays dozens of equations.
        assert len(statements) > 50 and len(backed & statements) > 45


class TestTheCheckerCanFail:
    def test_a_false_identity_is_reported_with_what_is_left(self):
        x = sp.Symbol("x", real=True)
        wrong = Derivation(
            "wrong", "sin^2 + cos^2 = 2", "nowhere",
            lambda: sp.sin(x) ** 2 + sp.cos(x) ** 2 - 2,
        )
        result = check(wrong)
        assert not result.passed
        assert result.remainder == "-1"

    def test_a_sign_error_in_the_coriolis_term_is_caught(self):
        """The slip a derivation by eye is most likely to make, made deliberately."""
        s = EntrySymbols()
        flipped = list(s.rates)
        flipped[4] = flipped[4].subs(s.planet.rotation_rate, -s.planet.rotation_rate)
        v_hat, n_gamma, n_psi = s.wind_triad
        velocity = s.speed * v_hat
        g_down, g_north = s.gravity
        force_v, force_gamma, force_psi = s.specific_force
        newton = (
            force_v * v_hat + force_gamma * n_gamma + force_psi * n_psi
            - g_down * s.up + g_north * s.north
            - 2 * s.rotation.cross(velocity)
            - s.rotation.cross(s.rotation.cross(s.position))
        )
        residual = lie_derivative(velocity, s.state, flipped) - newton
        assert not is_identically_zero(residual.dot(s.up))

    def test_one_entry_of_several_is_enough_to_fail(self):
        x = sp.Symbol("x")
        binomial = (x + 1) ** 2 - x**2 - 2 * x - 1
        assert is_identically_zero([sp.Matrix([0, x - x]), binomial])
        assert not is_identically_zero([sp.S.Zero, sp.Matrix([0, x])])

    def test_dropping_what_departing_mass_carries_off_is_caught(self):
        """Euler's equation with the whole of dI/dt is wrong for mass lost from the surface.

        The slip the manuscript once made: a body shedding an element of mass
        that leaves with the surface's velocity does not obey
        ``I w' = M - w x I w - I' w``. The angular momentum balance of the open
        system is satisfied by that equation only if what the element carries
        off is left out, and with it left out the residual is exactly the term
        ``I' w``.
        """
        t = sp.Symbol("t", real=True)
        omega = sp.Matrix([sp.Function(f"omega_{i}")(t) for i in (1, 2, 3)])
        position = sp.Matrix(sp.symbols("rho_1 rho_2 rho_3", real=True))
        element = sp.Function("mu")(t)
        inertia = sp.diag(*sp.symbols("B_1 B_2 B_3", positive=True)) + element * (
            position.dot(position) * sp.eye(3) - position * position.T
        )
        carried_off = sp.diff(element, t) * position.cross(omega.cross(position))
        # Body-axis balance of the open system, d(I w)/dt + w x I w - carried_off = M,
        # less the equation that keeps the whole of I': what is left is I' w.
        acceleration = sp.Matrix(sp.symbols("alpha_1 alpha_2 alpha_3", real=True))
        rate = sp.diff(inertia, t) * omega + inertia * acceleration + omega.cross(inertia * omega)
        with_all_of_it = inertia * acceleration + omega.cross(inertia * omega) + sp.diff(
            inertia, t
        ) * omega
        without = inertia * acceleration + omega.cross(inertia * omega)
        assert is_identically_zero((rate - carried_off) - without)
        assert not is_identically_zero((rate - carried_off) - with_all_of_it)

    def test_a_normal_force_of_the_wrong_sign_does_not_project_to_lift(self):
        a = AttitudeSymbols()
        axial, side, normal = a.coefficients
        flipped = AttitudeSymbols(coefficients=(axial, side, -normal))
        attitude = a.banked_attitude.subs(a.sideslip, 0)
        _, lifted, _ = flipped.wind_force(attitude)
        assert not is_identically_zero(lifted.subs(side, 0) - a.lift * sp.cos(a.bank))

    def test_the_gas_convection_term_of_the_other_sign_is_not_conservation_of_energy(self):
        """A charring-ablator equation in which the gas heats what it passes is caught.

        Not hypothetical: it is the sign the hand-written thermal solver
        carries, which the test below this class's sibling in the thermal
        suite pins.
        """
        slab = CharringSlab()
        temperature = slab.temperature
        probe = sp.Symbol("T_", positive=True)
        gas_capacity = sp.diff(slab.gas_enthalpy(probe), probe).subs(probe, temperature)
        gas_term = slab.gas_flux * gas_capacity * sp.diff(temperature, slab.depth)
        right = slab.temperature_residual + slab.gas_enthalpy(temperature) * slab.mass_residual
        assert is_identically_zero(slab.energy_residual - right)
        assert not is_identically_zero(slab.energy_residual - (right - 2 * gas_term))

    def test_a_scheme_that_takes_its_face_values_from_the_wrong_side_does_not_conserve(self):
        """Dropping what the moving faces sweep leaves the totals short by the char removed."""
        cells = CharringCells(cells=2)
        decomposition = [
            -cells.remaining * width
            * cells.decomposition_rate(density, temperature)
            for width, density, temperature in zip(
                cells.widths, cells.densities, cells.temperatures, strict=True
            )
        ]
        assert is_identically_zero(sum(cells.mass_rates) + cells.char_removal + cells.gas_blown)
        assert not is_identically_zero(sum(decomposition) + cells.char_removal + cells.gas_blown)

    def test_the_third_order_piston_coefficient_is_not_interchangeable(self):
        """Three numbers live at third order, and each is wrong in the others' place.

        kappa(kappa+1)/12 is the simple wave's, kappa(kappa+1)^2/32 the
        shock's, and kappa(kappa+1)/6 belongs to the load on two faces.
        """
        p = PistonSymbols()
        mach, k = p.piston_mach, p.heat_ratio
        simple = sp.diff(p.simple_wave(), mach, 3).subs(mach, 0) / 6
        shocked = sp.diff(p.shock(), mach, 3).subs(mach, 0) / 6
        assert is_identically_zero(simple - k * (k + 1) / 12)
        assert is_identically_zero(shocked - k * (k + 1) ** 2 / 32)
        assert not is_identically_zero(simple - k * (k + 1) / 6)
        assert not is_identically_zero(shocked - k * (k + 1) / 12)


def _unit_quaternions(count, seed=0):
    rng = np.random.default_rng(seed)
    samples = rng.normal(size=(count, 4))
    return samples / np.linalg.norm(samples, axis=1, keepdims=True)


class TestTheSymbolicAttitudeIsTheCodesAttitude:
    """Where the code has the same object, the appendix's statement of it is that object.

    The flight simulator carries its attitude against an inertial frame and the
    appendix against the rotating one, so the two do not share a field. They do
    share the quaternion kinematics, the direction-cosine matrix and the heating
    correlation, and on those the symbolic statements are pinned to the code.
    """

    def test_the_quaternion_rate_is_the_kinematics_the_simulator_integrates(self):
        a = AttitudeSymbols()
        rate = sp.lambdify(
            [a.quaternion, a.body_rate], a.quaternion_rate(a.body_rate), modules="numpy"
        )
        rng = np.random.default_rng(1)
        for q in _unit_quaternions(20):
            w = rng.normal(size=3)
            assert np.asarray(rate(q, w), dtype=float).ravel() == pytest.approx(
                quaternion_derivative(q, w, baumgarte_gain=0.0), rel=1e-13, abs=1e-15
            )

    def test_the_direction_cosine_matrix_is_the_codes_transposed(self):
        """The code's matrix maps reference to body; the appendix's maps body to reference."""
        a = AttitudeSymbols()
        matrix = sp.lambdify([a.quaternion], a.dcm, modules="numpy")
        for q in _unit_quaternions(20, seed=2):
            assert np.asarray(matrix(q), dtype=float) == pytest.approx(
                dcm_from_quaternion(q).T, rel=1e-13, abs=1e-15
            )

    def test_the_heat_flux_is_the_codes_correlation(self):
        c = CoupledSymbols()
        density = sp.Symbol("rho", positive=True)
        constant, nose = sp.symbols("k_Q R_n", positive=True)
        flux = sp.lambdify(
            [density, nose, c.entry.speed, constant], c.heat_flux.subs(c.density, density)
        )
        for rho, radius, speed in ((1e-4, 0.3, 6500.0), (3e-3, 1.2, 4200.0), (2e-5, 6.0, 10900.0)):
            assert flux(rho, radius, speed, 1.7415e-4) == pytest.approx(
                float(sutton_graves(rho, radius, speed, 1.7415e-4)), rel=1e-13
            )

    def test_the_force_on_the_wind_axes_is_what_the_matrices_give_numerically(self):
        """The quaternion-driven translational block against an independent composition.

        The symbolic coupled field resolves the body force through
        ``R_EL^T R(q)`` and hands it to the field of record. Here the same is
        done with numbers: the code's direction-cosine matrix, the local frame
        and the wind triad built by hand, and the float field.
        """
        c = CoupledSymbols()
        s = c.entry
        density = sp.Symbol("rho", positive=True)
        planet = s.planet
        constants = {
            planet.mu: EARTH.mu, planet.radius: EARTH.radius,
            planet.j2: EARTH.j2, planet.rotation_rate: EARTH.rotation_rate,
        }
        symbols = [
            *s.state, *c.attitude.quaternion, *c.attitude.coefficients,
            density, c.reference_area, c.mass,
        ]
        rates = sp.lambdify(
            symbols,
            [rate.subs(c.density, density).subs(constants) for rate in c.translational_rates],
            modules="numpy",
        )
        rng = np.random.default_rng(3)
        for state, q in zip(_states(12, seed=4), _unit_quaternions(12, seed=5), strict=True):
            coefficients = rng.uniform(-1.5, 1.5, size=3)
            rho, area, mass = 2.0e-4, 12.0, 1200.0
            _radius, lam, phi, speed, gam, psi = state
            local_frame = np.array(
                [
                    [-np.sin(phi) * np.cos(lam), -np.sin(lam), -np.cos(phi) * np.cos(lam)],
                    [-np.sin(phi) * np.sin(lam), np.cos(lam), -np.cos(phi) * np.sin(lam)],
                    [np.cos(phi), 0.0, -np.sin(phi)],
                ]
            )
            body_to_local = local_frame.T @ dcm_from_quaternion(q).T
            body_force = 0.5 * rho * speed**2 * area * np.array(
                [-coefficients[0], coefficients[1], -coefficients[2]]
            )
            local_force = body_to_local @ body_force
            v_hat = np.array(
                [np.cos(gam) * np.cos(psi), np.cos(gam) * np.sin(psi), -np.sin(gam)]
            )
            n_gamma = np.array(
                [-np.sin(gam) * np.cos(psi), -np.sin(gam) * np.sin(psi), -np.cos(gam)]
            )
            n_psi = np.cross(v_hat, n_gamma)
            specific = [float(local_force @ axis) / mass for axis in (v_hat, n_gamma, n_psi)]
            expected = np.array(rotating_entry_field(list(state), specific, NUMPY_OPS), dtype=float)
            computed = np.array(rates(*state, *q, *coefficients, rho, area, mass), dtype=float)
            assert computed == pytest.approx(expected, rel=1e-10, abs=1e-12)


class TestWhatTheCoupledDerivationsSay:
    """The numbers the dynamics appendix quotes, read off the expressions."""

    def test_the_state_has_sixteen_components_and_two_per_cell_and_per_mode(self):
        """16 + 2 N_T + 2 N_m: the appendix's count, for any number of cells and modes."""
        for modes, cells in ((0, 1), (1, 2), (4, 3)):
            coupled = CoupledSymbols(modes=modes, cells=cells)
            assert len(coupled.state) == len(coupled.rates) == 16 + 2 * cells + 2 * modes
            assert len(set(coupled.state)) == len(coupled.state)
        names = [str(symbol) for symbol in CoupledSymbols(modes=1, cells=2).state[13:]]
        assert names == ["Q_tot", "m", "s", "T_1", "T_2", "rho_1", "rho_2", "eta_1", "etadot_1"]

    def test_the_structure_is_softened_by_the_back_face_and_not_by_the_wall(self):
        coupled = CoupledSymbols(modes=1, cells=3)
        (frequency,) = coupled.modal_frequencies
        assert str(coupled.wall_temperature) == "T_1"
        assert frequency.free_symbols == {coupled.structure_temperature}
        assert str(coupled.structure_temperature) == "T_3"

    def test_leaving_out_the_planets_rotation_biases_the_attitude_by_six_degrees(self):
        """About 4.18e-3 deg/s, hence about 6 degrees over a 1500 s entry."""
        a = AttitudeSymbols()
        omega = sp.Matrix(a.body_rate)
        # What the kinematics are short of when driven by the inertial rate.
        bias = (omega - a.relative_rate).subs(dict(zip(a.quaternion, (1, 0, 0, 0), strict=True)))
        rate = float(bias.norm().subs(a.entry.planet.rotation_rate, EARTH.rotation_rate))
        assert np.degrees(rate) == pytest.approx(4.18e-3, rel=2e-3)
        assert 6.0 < np.degrees(rate) * 1500.0 < 6.5

    def test_the_oblateness_fractions_are_a_sixth_and_a_third_of_a_percent(self):
        """+0.16 % of the central term at the equator, -0.32 % at the poles, 0.16 % northward."""
        s = EntrySymbols()
        planet = s.planet
        at_the_surface = {
            planet.mu: EARTH.mu, planet.radius: EARTH.radius, planet.j2: EARTH.j2,
            s.radius: EARTH.radius,
        }
        g_down, g_north = s.gravity
        central = planet.mu / s.radius**2
        radial = ((g_down - central) / central).subs(at_the_surface)
        northward = (g_north / central).subs(at_the_surface)
        assert float(radial.subs(s.latitude, 0)) == pytest.approx(0.0016, abs=5e-5)
        assert float(radial.subs(s.latitude, sp.pi / 2)) == pytest.approx(-0.0032, abs=1e-4)
        assert abs(float(northward.subs(s.latitude, sp.pi / 4))) == pytest.approx(0.0016, abs=5e-5)

    def test_the_coriolis_heading_drift_is_twenty_to_fifty_times_the_oblate_one(self):
        """Level and due east at 45 degrees, over the range of entry speeds."""
        s = EntrySymbols()
        planet = s.planet
        level_east = {
            s.gamma: 0, s.heading: sp.pi / 2, s.latitude: sp.pi / 4, s.force_psi: 0,
            s.radius: EARTH.radius + 50.0e3, planet.mu: EARTH.mu, planet.radius: EARTH.radius,
        }
        heading_rate = s.rates[5].subs(level_east)
        on, off = EARTH.rotation_rate, 0
        ratios = []
        for speed in (3000.0, 7000.0):
            at_speed = heading_rate.subs(s.speed, speed)
            both_off = at_speed.subs({planet.rotation_rate: off, planet.j2: 0})
            coriolis = at_speed.subs({planet.rotation_rate: on, planet.j2: 0}) - both_off
            # The centrifugal part is second order in the rotation rate; take it out.
            centrifugal = (
                at_speed.subs({planet.rotation_rate: on, planet.j2: 0})
                + at_speed.subs({planet.rotation_rate: -on, planet.j2: 0})
            ) / 2 - both_off
            oblate = at_speed.subs({planet.rotation_rate: off, planet.j2: EARTH.j2}) - both_off
            ratios.append(abs(float(coriolis - centrifugal)) / abs(float(oblate)))
        assert 17.0 < ratios[0] < 23.0
        assert 40.0 < ratios[1] < 55.0

    def test_a_pass_that_turns_eight_degrees_sheds_an_eighth_of_the_kinetic_energy(self):
        """exp(-dgamma/E_v) in speed: thirteen percent of the energy, fourteen passes per e-fold."""
        turn, vertical = np.radians(8.0), 2.0
        speed_ratio = np.exp(-turn / vertical)
        assert 1.0 - speed_ratio**2 == pytest.approx(0.13, abs=0.005)
        assert vertical / turn == pytest.approx(14.3, abs=0.1)

    def test_the_phugoid_period_is_about_265_seconds(self):
        s = GlideSymbols()
        bind = {
            s.radius: 6371e3 + 60e3, s.speed: 6000.0, s.bank: 0.0, s.mu: 3.986004418e14,
            s.body_radius: 6371e3, s.scale_height: 7200.0, s.lift_to_drag: 2.0,
        }
        period = 2.0 * np.pi / np.sqrt(float(s.phugoid_frequency_squared.subs(bind)))
        assert period == pytest.approx(265.0, abs=3.0)


class TestWhatTheHeatShieldDerivationsSay:
    """The heat shield's statement, held against the kernels and the sources."""

    def test_the_mixture_is_chen_and_milos_mass_weighted_rule(self):
        """Their Eqs. (4)-(5): c_p = tau c_pv + (1 - tau) c_pc with tau the virgin
        mass fraction (1 - rho_c/rho)/(1 - rho_c/rho_v)."""
        slab = CharringSlab()
        temperature, density = sp.symbols("T rho", positive=True)
        virgin, char = slab.virgin_density, slab.char_density
        fraction = (1 - char / density) / (1 - char / virgin)
        virgin_capacity = sp.diff(sp.Function("h_v")(temperature), temperature)
        char_capacity = sp.diff(sp.Function("h_c")(temperature), temperature)
        stated = density * (fraction * virgin_capacity + (1 - fraction) * char_capacity)
        assert is_identically_zero(slab.heat_capacity(temperature, density) - stated)

    def test_the_blowing_correction_is_the_kernels(self):
        """The same function as aether.thermal.surface.blowing_correction, which is
        written in the blown coefficient."""
        from aether.thermal.surface import blowing_correction

        balance = SurfaceBalance()
        shape = next(s for s in balance.blowing_parameter.free_symbols if str(s) == "lambda_b")
        for unblown in (0.05, 0.4, 1.7, 6.0):
            values = {
                balance.film_coefficient: 1.0, balance.char_flux: unblown,
                balance.gas_flux: 0.0, shape: 0.5,
            }
            omega = float(balance.blowing_correction.subs(values))
            blown = unblown / omega  # B' on the blown coefficient
            assert omega == pytest.approx(float(blowing_correction(blown, 0.5)), rel=1e-12)

    def test_the_surface_balance_is_the_kernels_residual(self):
        """aether.thermal.surface.SurfaceEnergyBalance, term for term, and where it parts.

        With almost nothing blown the two agree in every term. With blowing
        they part in the correction alone, and by a known amount: the kernel
        applies the logarithmic form to a blowing rate formed on the
        *unblown* coefficient, which is neither of the two consistent
        statements -- it agrees with both to first order in the blowing and
        overstates the transfer beyond it.
        """
        from aether.thermal import demo_material
        from aether.thermal.surface import (
            STEFAN_BOLTZMANN,
            SurfaceEnergyBalance,
            SurfaceEnvironment,
        )

        material = demo_material()
        film, recovery, radiative = 0.3, 2.1e7, 4.0e5

        def wall_enthalpy(wall: float) -> float:
            return 1.0e3 * wall

        environment = SurfaceEnvironment(
            film_coefficient=film, recovery_enthalpy=recovery, radiative_flux=radiative,
            absorptivity=0.9, wall_enthalpy=wall_enthalpy, blowing_lambda=0.5,
        )
        kernel = SurfaceEnergyBalance(material, environment)
        balance = SurfaceBalance()
        symbols = {str(symbol): symbol for symbol in balance.conduction.free_symbols}

        def conduction(wall: float, char: float, gas: float) -> float:
            values = {
                balance.film_coefficient: film,
                balance.recovery_enthalpy: recovery,
                balance.wall_enthalpy: wall_enthalpy(wall),
                balance.radiative_flux: radiative,
                balance.wall_temperature: wall,
                balance.char_flux: char,
                balance.gas_flux: gas,
                balance.char_enthalpy: float(material.solid_enthalpy(wall)),
                balance.gas_enthalpy: float(material.gas_enthalpy(wall)),
                symbols["epsilon_w"]: float(material.emissivity(1.0)),
                symbols["alpha_w"]: 0.9,
                symbols["sigma_SB"]: STEFAN_BOLTZMANN,
                symbols["lambda_b"]: 0.5,
            }
            return float(balance.conduction.subs(values))

        for wall in (900.0, 1800.0, 2600.0):
            char, gas = 2.0e-6, 3.0e-6  # a blowing parameter of 1.7e-5
            balanced = conduction(wall, char, gas)
            # The kernel's residual is zero at the conduction that balances it.
            assert kernel.residual(wall, char, gas, balanced) == pytest.approx(
                0.0, abs=1e-6 * abs(balanced)
            )
        wall, char, gas = 1800.0, 0.11, 0.07
        blowing = (char + gas) / film  # 2 lambda B' with lambda = 1/2
        logarithmic, exponential = np.log1p(blowing) / blowing, blowing / np.expm1(blowing)
        gap = film * (recovery - wall_enthalpy(wall)) * (logarithmic - exponential)
        assert kernel.residual(wall, char, gas, conduction(wall, char, gas)) == pytest.approx(
            gap, rel=1e-9
        )
        assert 0.05 < (logarithmic - exponential) / exponential < 0.08

    def test_the_cells_integrate_a_slab_that_keeps_its_books(self):
        """Numbers through the scheme: a heated, receding, decomposing slab, and its ledgers.

        Closures bound to simple forms, the field compiled, and the total mass
        and enthalpy followed in time against what crossed the surface.
        """
        cells = CharringCells(cells=4)
        slab = cells.slab
        heating, removal = sp.symbols("q_net mdot_c", positive=True)
        # Bind: constant conductivity, enthalpies linear in T, first-order decomposition.
        constants = {
            slab.virgin_density: 280.0, slab.char_density: 220.0, slab.thickness: 0.05,
        }
        closures = {
            sp.Function("k", positive=True): lambda t, r: sp.Float(0.4),
            sp.Function("h_v"): lambda t: 1500.0 * t,
            sp.Function("h_c"): lambda t: 1200.0 * t + 2.0e5,
            sp.Function("h_g"): lambda t: 2000.0 * t + 1.0e6,
            sp.Function("R", nonnegative=True): lambda r, t: 2.0e-3 * (r - 220.0) * sp.exp(
                -3000.0 / t
            ),
        }
        bound = CharringCells(cells=4, char_removal=removal, surface_conduction=heating)

        def bind(expression):
            for function, form in closures.items():
                expression = expression.replace(function, form)
            return sp.simplify(expression.subs(constants).doit())

        rates = [bind(rate) for rate in bound.rates]
        ledgers = [bind(bound.areal_mass), bind(bound.areal_enthalpy)]
        surface_enthalpy = bind(bound.enthalpy_densities[0] / bound.densities[0])
        outflow = [
            bind(removal + bound.gas_blown),
            bind(
                heating - removal * surface_enthalpy
                - bound.gas_blown * slab.gas_enthalpy(bound.temperatures[0])
            ),
        ]
        field = compile_field(rates, bound.state, (heating, removal), jacobian=False)
        watch = compile_field([*ledgers, *outflow], bound.state, (heating, removal),
                              jacobian=False)
        parameters = [2.0e5, 0.01]
        state = np.array([300.0] * 4 + [280.0] * 4 + [0.0])
        step, steps = 0.01, 4000
        rows = np.vstack([state, field.rk4(state, parameters, step, steps, 1)])
        table = watch.batch(rows, parameters)
        times = step * np.arange(steps + 1)
        mass_lost = scipy.integrate.simpson(table[:, 2], x=times)
        enthalpy_gained = scipy.integrate.simpson(table[:, 3], x=times)
        assert rows[-1, 0] > 600.0 and rows[-1, 8] > 1.0e-3  # it heated, and it receded
        assert rows[-1, 4] < 280.0  # and it decomposed
        assert table[0, 0] - table[-1, 0] == pytest.approx(mass_lost, rel=1e-6)
        assert table[-1, 1] - table[0, 1] == pytest.approx(enthalpy_gained, rel=1e-6)


class TestWhatThePistonDerivationsSay:
    """What Lighthill and Ashley and Zartarian print, read off the expressions."""

    #: Lighthill (1953), Table 1, for a ratio of specific heats of 1.4. By row of
    #: w/a_1: simple wave, shock expansion (shock at w/a_1 = 1), quadratic, cubic.
    _TABLE_1 = (
        ("1", (3.583, 3.473, 3.240, 3.520)),
        ("1/2", (1.949, 1.916, 1.910, 1.945)),
        ("0", (1.000, 1.000, 1.000, 1.000)),
        ("-1/2", (0.478, 0.488, 0.510, 0.475)),
        ("-1", (0.210, 0.220, 0.440, 0.160)),
    )
    _AIR = sp.Rational(7, 5)

    def _columns(self, mach):
        p = PistonSymbols()
        air = {p.heat_ratio: self._AIR}
        return tuple(
            float(expression.subs(air))
            for expression in (
                p.simple_wave(mach),
                p.shock_then_expansion(mach, 1),
                p.expansion(2, mach),
                p.expansion(3, mach),
            )
        )

    def test_lighthills_table_of_piston_pressures_is_reproduced(self):
        """Every entry, to the three decimals the paper prints."""
        for mach, printed in self._TABLE_1:
            assert self._columns(sp.Rational(mach)) == pytest.approx(printed, abs=5.1e-4)

    def test_the_cubic_is_within_six_hundredths_of_both_exact_values(self):
        """"Always within 0.06 p_1 of both values", and the simple wave "in error by 0.11 p_1
        immediately behind the shock wave"."""
        worst_simple = worst_shock = 0.0
        for step in range(-20, 21):
            simple, shock_expansion, _, cubic = self._columns(sp.Rational(step, 20))
            worst_simple = max(worst_simple, abs(cubic - simple))
            worst_shock = max(worst_shock, abs(cubic - shock_expansion))
        assert worst_simple == pytest.approx(0.063, abs=5e-4)
        assert worst_shock == pytest.approx(0.060, abs=1e-3)
        p = PistonSymbols()
        behind = float((p.simple_wave(1) - p.shock(1)).subs(p.heat_ratio, self._AIR))
        assert behind == pytest.approx(0.11, abs=5e-3)

    def test_the_surface_pressure_ranges_from_a_fifth_to_three_and_a_half(self):
        """The band Lighthill gives for the formula: 0.2 to 3.5 times the stream pressure."""
        p = PistonSymbols()
        air = {p.heat_ratio: self._AIR}
        assert float(p.simple_wave(-1).subs(air)) == pytest.approx(0.2, abs=0.01)
        assert float(p.simple_wave(1).subs(air)) == pytest.approx(3.5, abs=0.1)

    def test_the_shock_coefficient_is_nine_tenths_of_the_simple_waves(self):
        """"Differs from that in Eq. (5) by only 10 per cent"."""
        p = PistonSymbols()
        simple, shocked = p.coefficients(3)[3], p.coefficients(3, through_shock=True)[3]
        assert sp.simplify((shocked / simple).subs(p.heat_ratio, self._AIR)) == sp.Rational(9, 10)

    def test_the_remainder_after_the_cubic_is_bounded_as_the_appendix_says(self):
        """At most 0.097 for |w| <= a; the gap itself is 0.063, at w = a; never negative."""
        p = PistonSymbols()
        air = {p.heat_ratio: self._AIR}
        coefficient = float(p.remainder_coefficient.subs(air))
        power = float((p.exponent - 4).subs(air))
        assert coefficient == pytest.approx(0.056) and power == pytest.approx(3.0)
        bound = coefficient * float(p.sound_ratio(1).subs(air)) ** power
        assert bound == pytest.approx(0.097, abs=5e-4)
        gaps = [
            float((p.simple_wave(m) - p.expansion(3, m)).subs(air))
            for m in (sp.Rational(step, 20) for step in range(-20, 21))
        ]
        assert min(gaps) >= 0.0
        assert max(gaps) == gaps[-1] == pytest.approx(0.063, abs=5e-4)
        assert max(gaps) <= bound

    def test_the_linear_factor_is_two_percent_at_mach_five(self):
        flow = WavyWall()
        n = flow.relative_mach
        for value in (1.5, 2.0, 5.0, 20.0):
            excess = float((flow.piston_factor - 1).subs(n, value))
            assert 0.0 < excess <= 1.0 / (2.0 * (value**2 - 1.0))
        assert float((flow.piston_factor - 1).subs(n, 5)) == pytest.approx(0.02, abs=1e-3)

    def test_the_three_conditions_of_ashley_and_zartarian_are_the_terms_of_one_square(self):
        """M^2, k M^2 and k^2 M^2 with k = c/U: a wave running upstream at c."""
        stream, sound, wave = sp.symbols("U a c", positive=True)
        upstream = WavyWall(stream=stream, sound_speed=sound, relative_mach=(stream + wave) / sound)
        mach, reduced = stream / sound, wave / stream
        assert is_identically_zero(
            upstream.relative_mach**2
            - (mach**2 + 2 * reduced * mach**2 + reduced**2 * mach**2)
        )
        # ... and that wave does run upstream, at c.
        assert is_identically_zero(upstream.frequency / upstream.wavenumber + wave)

    def test_a_pitching_flat_plate_has_lighthills_stiffness_and_damping(self):
        """Eqs. (12) and (13) of the paper, with the thickness terms dropped.

        A plate pitching about an axis is one "mode", of shape (x - x0). The
        strip's stiffness and damping integrals, made dimensionless as the
        paper makes them, are then its (1/(M c^2)) int 2 (x - x0) dx and
        (1/(M c^3)) int 2 (x - x0)^2 dx.
        """
        strip = StripSymbols(modes=1)
        chord, axis, mach, density, sound = sp.symbols("c x_0 M rho a", positive=True)
        station, stream = strip.station, mach * sound
        plate = {
            strip.width: 1,
            strip.impedance: 2 * density * sound,
            strip.stream: stream,
            strip.shapes[0]: station - axis,
        }

        def over_the_chord(integrand):
            return sp.integrate(integrand.subs(plate).doit(), (station, 0, chord))

        stiffness = over_the_chord(strip.stiffness_density(0, 0)) / (density * stream**2 * chord**2)
        damping = over_the_chord(strip.damping_density(0, 0)) / (density * stream * chord**3)
        assert is_identically_zero(
            stiffness - sp.integrate(2 * (station - axis), (station, 0, chord)) / (mach * chord**2)
        )
        assert is_identically_zero(
            damping
            - sp.integrate(2 * (station - axis) ** 2, (station, 0, chord)) / (mach * chord**3)
        )
        # Damping about any axis is positive: "there is always damping at high Mach Number".
        assert sp.simplify(damping * mach - 2 * (sp.Rational(1, 3) - axis / chord
                                                 + (axis / chord) ** 2)) == 0

    def test_the_coupled_forcing_is_the_strip_under_a_uniform_stream(self):
        """The modal forcing the coupled field carries is the one the strip derives."""
        coupled, strip = CoupledSymbols(modes=2), StripSymbols(modes=2)
        impedance = 2 * coupled.density * coupled.sound_speed
        uniform = {strip.impedance: impedance, strip.stream: coupled.entry.speed}
        shape_rate = {
            sp.Derivative(amplitude, strip.time): velocity
            for amplitude, velocity in zip(strip.amplitudes, coupled.modal_velocities, strict=True)
        }
        amplitude_value = dict(zip(strip.amplitudes, coupled.modal_displacements, strict=True))
        integrands = {}
        for i in range(2):
            integrands[coupled.rigid_forcing[i]] = 0
            for j in range(2):
                integrands[coupled.damping_integrals[i, j]] = (
                    strip.width * strip.shapes[i] * strip.shapes[j]
                )
                integrands[coupled.stiffness_integrals[i, j]] = (
                    strip.width * strip.shapes[i] * sp.diff(strip.shapes[j], strip.station)
                )
        for i in range(2):
            derived = strip.forcing_density(i).subs(uniform).subs(shape_rate).subs(amplitude_value)
            carried = coupled.modal_forcing[i].subs(integrands)
            assert is_identically_zero(derived - carried)


class TestWhatTheGlideDerivationsSay:
    """The numbers the notes quote, read off the expressions rather than typed in."""

    def _values(self, scale_height=7200.0, speed=6000.0, lift_to_drag=2.0, bank=0.0):
        s = GlideSymbols()
        radius = 6371e3 + 60e3
        bind = {
            s.radius: radius, s.speed: speed, s.bank: bank, s.mu: 3.986004418e14,
            s.body_radius: 6371e3, s.scale_height: scale_height, s.lift_to_drag: lift_to_drag,
        }
        return s, bind

    def test_the_damping_ratio_is_a_few_percent_and_scales_as_root_epsilon(self):
        s, bind = self._values()
        zeta = float(s.damping_ratio.subs(bind))
        assert 0.02 < zeta < 0.05
        thinner = float(s.damping_ratio.subs(bind | {s.scale_height: 7200.0 / 4}))
        assert zeta / thinner == pytest.approx(2.0, rel=1e-12)

    def test_the_damping_time_does_not_depend_on_the_scale_height(self):
        s, bind = self._values()
        assert s.scale_height not in s.phugoid_damping_rate.free_symbols
        # Twenty minutes: the length of the entry, not of a boundary layer.
        assert 1200.0 < 1.0 / float(s.phugoid_damping_rate.subs(bind)) < 1300.0

    def test_the_closed_forms_match_the_exact_frozen_spectrum(self):
        """The limits are of exact expressions, so at finite H_s they can be compared."""
        s, bind = self._values()
        coefficients = [float(sp.sympify(c).subs(bind)) for c in s.characteristic_coefficients]
        roots = np.roots([1.0, *coefficients])
        pair = roots[np.abs(roots.imag) > 0][0]
        assert -pair.real == pytest.approx(float(s.phugoid_damping_rate.subs(bind)), rel=0.02)
        assert abs(pair.imag) ** 2 == pytest.approx(
            float(s.phugoid_frequency_squared.subs(bind)), rel=0.02
        )


class TestGeneratedCode:
    def test_both_routines_are_emitted_and_named_for_what_they_bind(self):
        source = point_mass_source()
        assert "void entry_point_mass(double *x, double *p, double *out)" in source.code
        assert "void entry_point_mass_jacobian(double *x, double *p, double *out)" in source.code
        assert "state      x[0..5] = r, lambda, phi, V, gamma, psi" in source.code
        assert "NOT generated" in source.code

    def test_an_unbound_symbol_is_refused(self):
        x, y, stray = sp.symbols("x y stray")
        with pytest.raises(ValueError, match="stray"):
            generate_c("f", [x + stray], [x], [y])

    def test_a_name_c_cannot_use_is_refused(self):
        x = sp.Symbol("x")
        with pytest.raises(ValueError, match="function name"):
            generate_c("not a name", [x], [x], [])

    def test_small_integer_powers_are_printed_as_products(self):
        x = sp.Symbol("x")
        code = generate_c("f", [x**4 + x**-2 + sp.sqrt(x)], [x], [], jacobian=False).code
        assert "pow(" not in code
        assert "x[0]*x[0]*x[0]*x[0]" in code

    def test_without_a_compiler_the_same_interface_runs_through_numpy(self, monkeypatch):
        monkeypatch.setattr(codegen_module, "find_c_compiler", lambda: None)
        source = point_mass_source()
        field = compile_field(source.expressions, source.state, source.parameters)
        assert isinstance(field, LambdifiedField)
        for state in _states(20):
            assert field(state, _PARAMETERS) == pytest.approx(_reference(state), rel=1e-12)

    @needs_compiler
    def test_the_compiled_field_is_the_field(self, tmp_path):
        source = point_mass_source()
        field = compile_field(
            source.expressions, source.state, source.parameters,
            name="entry_point_mass", build_dir=tmp_path,
        )
        assert field.backend == "c"
        for state in _states(200):
            assert field(state, _PARAMETERS) == pytest.approx(_reference(state), rel=1e-13)
        # The emitted source is kept beside the library, to be read.
        assert list(tmp_path.glob("entry_point_mass-*.c"))

    @needs_compiler
    def test_the_compiled_field_matches_its_expressions_in_forty_digits(self):
        """With rotation and oblateness on: the terms a faulty rewrite once corrupted."""
        source = point_mass_source()
        field = compile_field(source.expressions, source.state, source.parameters)
        for state in _states(10, seed=1):
            assert field.audit(state, _PARAMETERS) < 1e-13

    @needs_compiler
    def test_the_compiled_jacobian_is_the_derivative(self):
        source = point_mass_source()
        compiled = compile_field(source.expressions, source.state, source.parameters)
        reference = LambdifiedField(source)
        for state in _states(50, seed=2):
            assert compiled.jacobian(state, _PARAMETERS) == pytest.approx(
                reference.jacobian(state, _PARAMETERS), rel=1e-12, abs=1e-18
            )
        state = _states(1, seed=3)[0]
        jacobian = compiled.jacobian(state, _PARAMETERS)
        for j in range(6):
            step = 1.0 if j == 0 else 1e-5 * max(abs(state[j]), 1.0)
            shift = np.zeros(6)
            shift[j] = step
            column = (compiled(state + shift, _PARAMETERS) - compiled(state - shift, _PARAMETERS))
            column /= 2.0 * step
            assert jacobian[:, j] == pytest.approx(column, rel=1e-5, abs=1e-9)

    @needs_compiler
    def test_the_fixed_driver_integrates_what_an_independent_integrator_does(self):
        """The Runge-Kutta loop is the one part not generated, so it is tested instead."""
        source = point_mass_source()
        field = compile_field(source.expressions, source.state, source.parameters)
        start = np.array([EARTH.radius + 45e3, 0.21, 0.54, 5200.0, -0.024, 1.22])
        steps = 4000
        compiled = field.rk4(start, _PARAMETERS, 0.05, steps, stride=steps)[-1]
        flown = scipy.integrate.solve_ivp(
            lambda _t, y: _reference(y), (0.0, 200.0), start, rtol=1e-12, atol=1e-12
        )
        assert compiled == pytest.approx(flown.y[:, -1], rel=1e-9)
        assert LambdifiedField(source).rk4(start, _PARAMETERS, 0.05, 40, stride=40)[-1] == (
            pytest.approx(field.rk4(start, _PARAMETERS, 0.05, 40, stride=40)[-1], rel=1e-13)
        )

    @needs_compiler
    def test_the_driver_returns_one_row_per_stride(self):
        source = point_mass_source()
        field = compile_field(source.expressions, source.state, source.parameters)
        start = np.array([EARTH.radius + 45e3, 0.21, 0.54, 5200.0, -0.024, 1.22])
        rows = field.rk4(start, _PARAMETERS, 0.1, 100, stride=10)
        assert rows.shape == (10, 6)
        assert rows[-1] == pytest.approx(field.rk4(start, _PARAMETERS, 0.1, 100, stride=100)[-1])
        with pytest.raises(ValueError, match="positive"):
            field.rk4(start, _PARAMETERS, 0.1, 0)
