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

from aether.certification import NUMPY_OPS  # noqa: E402
from aether.certification.rotating_field import (  # noqa: E402
    EARTH,
    exponential_density,
    point_mass_field,
    rotating_entry_field,
)
from aether.symbolic import (  # noqa: E402
    DERIVATIONS,
    SYMPY_OPS,
    Derivation,
    EntrySymbols,
    GlideSymbols,
    LambdifiedField,
    check,
    compile_field,
    find_c_compiler,
    generate_c,
    is_identically_zero,
    lie_derivative,
)
from aether.symbolic import codegen as codegen_module  # noqa: E402
from aether.symbolic.__main__ import point_mass_source  # noqa: E402

_BETA, _LD, _RHO0, _H_SCALE = 4000.0, 2.5, 1.225, 8500.0
_PARAMETERS = np.array(
    [0.4, _BETA, _LD, _RHO0, _H_SCALE, EARTH.mu, EARTH.radius, EARTH.j2, EARTH.rotation_rate]
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


class TestWhatTheGlideDerivationsSay:
    """The numbers the proposal quotes, read off the expressions rather than typed in."""

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
