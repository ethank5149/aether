"""The closures a compiled field is printed from: atmosphere, figure, and the system at trim.

The derivations say what follows from the equations. These tests are about
the other half of a model, the parts that are data or definition: that the
symbolic atmosphere is the 1976 standard where the standard is a definition
and within a stated distance of its reference where it is a fit; that the
surface the field measures altitude from is the reference ellipsoid; that a
state published as geodetic and inertial arrives in the field's coordinates
as an independent Cartesian computation says it should; and that the system
a simulator compiles is the field of record and nothing else.
"""

from __future__ import annotations

import numpy as np
import pytest

sp = pytest.importorskip("sympy")

from aether.atmosphere.model import earth_atmosphere  # noqa: E402
from aether.atmosphere.standard import (  # noqa: E402
    _BASE_PRESSURE,
    _BASE_TEMPERATURE,
    USStandard1976,
)
from aether.certification.rotating_field import (  # noqa: E402
    EARTH,
    WGS84_FLATTENING,
    altitude,
    point_mass_field,
    surface_radius,
)
from aether.ellipsoid import WGS84, ecef_to_geodetic, geodetic_to_ecef  # noqa: E402
from aether.guidance.entry import EntryBody, entry_dynamics  # noqa: E402
from aether.symbolic import SYMPY_OPS, EntrySymbols, find_c_compiler  # noqa: E402
from aether.symbolic.atmosphere import (  # noqa: E402
    geopotential,
    gravity,
    standard_atmosphere,
    us_standard_1976_layers,
)
from aether.symbolic.figure import FigureSymbols  # noqa: E402
from aether.symbolic.ops import is_identically_zero  # noqa: E402
from aether.symbolic.system import SymbolicSystem  # noqa: E402
from aether.symbolic.trim import entry_system, prediction_system  # noqa: E402

_ELLIPSOID = EARTH.on_ellipsoid(WGS84_FLATTENING)
_PLANET = {
    "mu": EARTH.mu, "R_e": EARTH.radius, "J_2": EARTH.j2,
    "omega_E": EARTH.rotation_rate, "f_E": WGS84_FLATTENING,
}


@pytest.fixture(scope="module")
def atmosphere():
    return standard_atmosphere()


@pytest.fixture(scope="module")
def on_floats(atmosphere):
    """Density, pressure and speed of sound of the symbolic atmosphere, as functions."""
    h = sp.Symbol("h", real=True)
    functions = [
        sp.lambdify(h, expression, "numpy")
        for expression in (
            atmosphere.density(h), atmosphere.pressure(h), atmosphere.sound_speed(h)
        )
    ]

    def evaluate(index, altitudes):
        with np.errstate(all="ignore"):
            return np.asarray(functions[index](np.asarray(altitudes, dtype=float)))

    return evaluate


class TestTheAtmosphereOnSymbols:
    def test_below_eighty_kilometres_it_is_the_standard_as_defined(self, on_floats):
        """Not a fit: the same closed forms as the kernel, to rounding."""
        heights = np.linspace(-2000.0, 79_990.0, 4001)
        kernel = USStandard1976().state(heights)
        assert on_floats(0, heights) == pytest.approx(kernel.density, rel=1e-12)
        assert on_floats(1, heights) == pytest.approx(kernel.pressure, rel=1e-12)
        assert on_floats(2, heights) == pytest.approx(kernel.speed_of_sound, rel=1e-12)

    def test_the_base_values_come_out_of_the_recursion(self):
        """Against the kernel's own recursion, and against the standard's printed table."""
        layers = us_standard_1976_layers()
        for layer, temperature, pressure in zip(
            layers, _BASE_TEMPERATURE, _BASE_PRESSURE, strict=False
        ):
            assert float(layer.base_temperature) == pytest.approx(temperature, abs=1e-10)
            assert float(layer.base_pressure) == pytest.approx(pressure, rel=1e-13)
        printed = (101325.0, 22632.06, 5474.889, 868.0187, 110.9063, 66.93887, 3.956420)
        for layer, pressure in zip(layers, printed, strict=True):
            assert float(layer.base_pressure) == pytest.approx(pressure, rel=2e-6)

    def test_every_layer_is_in_hydrostatic_equilibrium(self):
        """dp/dH = -rho g_0 in each layer, with the layer's constants left as symbols."""
        from aether.symbolic.atmosphere import StandardLayer

        height = sp.Symbol("H", real=True)
        base, lapse = sp.symbols("H_b L", real=True)
        temperature, pressure, constant = sp.symbols("T_b p_b c_h", positive=True)
        for isothermal in (False, True):
            layer = StandardLayer(base, lapse, temperature, pressure, constant, isothermal)
            g0 = sp.Rational(980665, 100000)
            assert is_identically_zero(
                sp.diff(layer.pressure(height), height) + layer.density(height) * g0
            )

    def test_geopotential_altitude_absorbs_the_standards_gravity(self):
        """dH/dZ = g(Z)/g_0, which is what makes the layers closed forms in H."""
        z = sp.Symbol("Z", positive=True)
        g0 = sp.Rational(980665, 100000)
        assert is_identically_zero(sp.diff(geopotential(z), z) - gravity(z) / g0)

    def test_the_fit_above_is_within_a_tenth_of_a_percent_of_its_reference(
        self, atmosphere, on_floats
    ):
        assert atmosphere.fit_error < 1.0e-3
        heights = np.linspace(86.0e3, 1000.0e3, 9001)
        reference = earth_atmosphere().state(heights)
        assert np.max(np.abs(on_floats(0, heights) / reference.density - 1.0)) < 1.0e-3

    def test_the_fit_is_the_one_the_appendix_describes(self, atmosphere):
        """Sixty-nine knots, within 0.08 % of NRLMSISE-00."""
        assert len(atmosphere.upper_density.knots) == 69
        assert atmosphere.fit_error < 8.0e-4
        assert "NRLMSISE-00" in atmosphere.reference

    def test_the_models_do_not_meet_at_the_seam(self):
        """At 86 km: NRLMSISE-00 6 % above the standard, NRLMSIS 2.0 11 % below."""
        from aether.atmosphere.upper import MSISAtmosphere

        standard = float(USStandard1976().state(86.0e3).density)
        older = float(MSISAtmosphere(version=0.0).state(86.0e3).density)
        newer = float(MSISAtmosphere(version=2.0).state(86.0e3).density)
        assert older / standard - 1.0 == pytest.approx(0.062, abs=0.003)
        assert newer / standard - 1.0 == pytest.approx(-0.108, abs=0.003)

    def test_the_blend_between_them_is_the_layered_atmospheres(self, on_floats):
        heights = np.linspace(80.0e3, 86.0e3, 601)
        reference = earth_atmosphere().state(heights)
        assert np.max(np.abs(on_floats(0, heights) / reference.density - 1.0)) < 5.0e-5

    def test_density_is_continuous_and_falls_with_altitude(self, on_floats):
        heights = np.linspace(-1000.0, 1100.0e3, 220_001)
        density = on_floats(0, heights)
        assert np.all(np.diff(density) < 0.0)
        # No step at any seam: over 5 m the logarithm moves by 5 m over a scale height.
        assert np.max(np.abs(np.diff(np.log(density)))) < 5.0 / 4000.0


class TestTheFigureOfThePlanet:
    def test_the_surface_the_field_uses_is_the_reference_ellipsoid(self):
        """Against the geodetic kernel: a point on the ellipsoid, seen from the centre."""
        for geodetic_latitude in np.radians([-80.0, -27.0, 0.0, 13.0, 45.0, 89.0]):
            point = geodetic_to_ecef(geodetic_latitude, 0.3, 0.0, WGS84)
            radius = float(np.linalg.norm(point))
            geocentric = float(np.arcsin(point[2] / radius))
            assert surface_radius(geocentric, planet=_ELLIPSOID) == pytest.approx(radius, rel=1e-14)

    def test_height_along_the_radius_is_geodetic_altitude_to_under_a_metre(self):
        """And the leading term of the difference is the one the derivation gives."""
        f, a = WGS84_FLATTENING, EARTH.radius
        for height in (30.0e3, 86.0e3, 121_920.0):
            for geodetic_latitude in np.radians([-60.0, -25.8, 10.0, 27.3, 45.0, 80.0]):
                point = geodetic_to_ecef(geodetic_latitude, -2.0, height, WGS84)
                radius = float(np.linalg.norm(point))
                geocentric = float(np.arcsin(point[2] / radius))
                excess = altitude(radius, geocentric, planet=_ELLIPSOID) - height
                leading = 0.5 * f**2 * a * height / (a + height) * np.sin(
                    2.0 * geodetic_latitude
                ) ** 2
                assert 0.0 <= excess < 0.75
                assert excess == pytest.approx(leading, rel=0.02, abs=1e-6)
                # ... and the kernel's inverse agrees about which altitude that was.
                assert float(ecef_to_geodetic(point, WGS84)[2]) == pytest.approx(height, abs=1e-6)

    def test_the_exact_excess_has_no_term_of_first_order_in_the_flattening(self):
        """Zero at orders zero and one; at order two, a h sin^2(2 phi) / (2 (a + h)).

        The altitude is written as the distance from the centre less the
        equatorial radius, so that the square root at zero flattening is of a
        quantity SymPy knows the sign of.
        """
        planet = EntrySymbols().planet
        distance = sp.Symbol("s", positive=True)
        figure = FigureSymbols(geodetic_altitude=distance - planet.radius)
        flattening = planet.flattening
        orders = [
            # Factored first: expanding these before the trigonometry is collected
            # is the slow way round.
            sp.trigsimp(sp.factor(
                sp.diff(figure.radial_height_excess, flattening, order).subs(flattening, 0)
            )) / sp.factorial(order)
            for order in range(3)
        ]
        assert orders[0] == 0
        assert orders[1] == 0
        leading = (
            planet.radius * (distance - planet.radius)
            * sp.sin(2 * figure.geodetic_latitude) ** 2 / (2 * distance)
        )
        assert is_identically_zero(orders[2] - leading)

    def test_a_published_state_arrives_as_the_cartesian_kernels_say(self):
        """The same conversion done in Cartesian components, with nothing shared but the inputs."""
        figure = FigureSymbols()
        planet = figure.planet
        constants = {
            planet.mu: EARTH.mu, planet.radius: EARTH.radius, planet.j2: EARTH.j2,
            planet.rotation_rate: EARTH.rotation_rate, planet.flattening: WGS84_FLATTENING,
        }
        convert = sp.lambdify(
            figure.published, [e.subs(constants) for e in figure.model_state], "numpy"
        )
        rng = np.random.default_rng(3)
        for _ in range(25):
            height = rng.uniform(60.0e3, 130.0e3)
            longitude, latitude = rng.uniform(-3.0, 3.0), rng.uniform(-1.3, 1.3)
            speed = rng.uniform(3000.0, 11_500.0)
            gamma, azimuth = rng.uniform(-0.3, 0.1), rng.uniform(-3.0, 3.0)
            state = np.array(convert(height, longitude, latitude, speed, gamma, azimuth))

            position = geodetic_to_ecef(latitude, longitude, height, WGS84)
            up = np.array([
                np.cos(latitude) * np.cos(longitude),
                np.cos(latitude) * np.sin(longitude),
                np.sin(latitude),
            ])
            east = np.array([-np.sin(longitude), np.cos(longitude), 0.0])
            north = np.cross(up, east)
            inertial = speed * (
                np.cos(gamma) * np.cos(azimuth) * north
                + np.cos(gamma) * np.sin(azimuth) * east
                + np.sin(gamma) * up
            )
            relative = inertial - np.cross([0.0, 0.0, EARTH.rotation_rate], position)
            radius = float(np.linalg.norm(position))
            radial = position / radius
            level_north = np.cross(radial, east)
            expected = np.array([
                radius, longitude, np.arcsin(radial[2]), np.linalg.norm(relative),
                np.arcsin(relative @ radial / np.linalg.norm(relative)),
                np.arctan2(relative @ east, relative @ level_north),
            ])
            assert state == pytest.approx(expected, rel=1e-12, abs=1e-12)

    def test_the_two_verticals_are_0_15_degrees_apart_at_orions_interface(self):
        figure = FigureSymbols()
        planet = figure.planet
        deflection = figure.deflection.subs({
            planet.radius: EARTH.radius, planet.flattening: WGS84_FLATTENING,
            figure.geodetic_altitude: 121_920.0, figure.geodetic_latitude: np.radians(-25.82847),
        })
        assert abs(np.degrees(float(deflection))) == pytest.approx(0.148, abs=0.002)


class TestTheSystemASimulatorCompiles:
    def test_its_motion_is_the_field_of_record_and_nothing_else(self, atmosphere):
        """The six rates are point_mass_field, evaluated on the same symbols and closures."""
        system = entry_system(atmosphere)
        symbols = EntrySymbols()
        bank, beta, ratio, scale = (system.parameters[i] for i in range(4))
        field = point_mass_field(
            beta, ratio, density=atmosphere.as_density(scale), planet=symbols.planet
        )
        expected = field(symbols.state, bank, SYMPY_OPS)
        assert list(system.rates[:6]) == expected

    def test_the_arc_and_the_heat_load_are_the_two_integrals_it_adds(self, atmosphere):
        system = entry_system(atmosphere)
        symbols = EntrySymbols()
        assert system.state_names[6:] == ("s", "Q_tot")
        assert is_identically_zero(
            system.rates[6] - symbols.speed * sp.cos(symbols.gamma) / symbols.radius
        )
        assert system.rates[7] == system.outputs["heat_flux"]

    def test_the_predictor_is_the_plant_with_the_lateral_lift_left_out(self, atmosphere):
        """Held at one bank, it differs from the plant in the heading rate alone."""
        plant, predictor = entry_system(atmosphere), prediction_system(atmosphere)
        symbols = EntrySymbols()
        bank = plant.parameters[0]
        now, goal = predictor.parameters[0], predictor.parameters[1]
        # The density is one shared expression of seventy-six pieces; it is put
        # aside as a symbol, since the claim is about how it enters and not what it is.
        air = {atmosphere.density(symbols.altitude): sp.Symbol("rho_air", positive=True)}
        plant_rates = [rate.xreplace(air) for rate in plant.rates]
        held = [rate.xreplace(air).subs({now: bank, goal: bank}) for rate in predictor.rates]
        for index in (0, 1, 2, 3, 4, 6):
            assert is_identically_zero(held[index] - plant_rates[index])
        lift = plant.outputs["lift"].xreplace(air)
        lateral = lift * sp.sin(bank) / (symbols.speed * sp.cos(symbols.gamma))
        assert is_identically_zero(plant_rates[5] - held[5] - lateral)

    def test_on_a_sphere_that_does_not_rotate_it_is_the_hand_written_dynamics(self, atmosphere):
        """The legacy equations, given the same density, to rounding."""
        compiled = entry_system(atmosphere).compile(jacobian=False)
        classical = {**_PLANET, "J_2": 0.0, "omega_E": 0.0, "f_E": 0.0}

        def parameters(bank):
            return compiled.parameter_vector(
                classical, sigma=bank, beta=281.0, E=0.3, k_rho=1.0, R_n=6.0, k_Q=1.7415e-4
            )

        def density(height):
            probe = np.array([EARTH.radius + height, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
            return compiled.outputs(probe, parameters(0.0))["density"]

        body = EntryBody(281.0, 0.3, max_bank=np.pi, density=density)
        rng = np.random.default_rng(5)
        for _ in range(100):
            state = np.array([
                EARTH.radius + rng.uniform(20.0e3, 125.0e3), rng.uniform(-3.0, 3.0),
                rng.uniform(-1.2, 1.2), rng.uniform(500.0, 11_000.0),
                rng.uniform(-0.3, 0.3), rng.uniform(-3.0, 3.0),
            ])
            bank = rng.uniform(-3.0, 3.0)
            generated = compiled.rates(np.concatenate([state, [0.0, 0.0]]), parameters(bank))
            assert generated[:6] == pytest.approx(
                entry_dynamics(state, bank, body), rel=1e-10, abs=1e-14
            )


class TestASymbolicSystem:
    def _oscillator(self):
        x, v = sp.symbols("x v", real=True)
        k = sp.Symbol("k", positive=True)
        return SymbolicSystem(
            "oscillator", (x, v), (k,), (v, -k * x), {"energy": (v**2 + k * x**2) / 2}
        )

    def test_a_symbol_that_is_neither_state_nor_parameter_is_refused(self):
        x, v, stray = sp.symbols("x v stray", real=True)
        with pytest.raises(ValueError, match="neither state nor parameters"):
            SymbolicSystem("bad", (x, v), (), (v, -stray * x))
        with pytest.raises(ValueError, match="one rate per state"):
            SymbolicSystem("bad", (x, v), (), (v,))

    def test_every_parameter_has_to_be_given_by_name(self):
        compiled = self._oscillator().compile()
        with pytest.raises(ValueError, match="needs a value"):
            compiled.parameter_vector()
        with pytest.raises(ValueError, match="not parameters"):
            compiled.parameter_vector(k=1.0, m=2.0)
        assert compiled.parameter_vector(k=4.0) == pytest.approx([4.0])

    def test_advancing_halves_the_step_until_the_answer_holds(self):
        """A quarter period of an oscillator, against the closed form, from a coarse step."""
        compiled = self._oscillator().compile()
        parameters = compiled.parameter_vector(k=4.0)
        end = compiled.advance([1.0, 0.0], parameters, np.pi / 4.0, step=0.5, rtol=1e-12,
                               atol=1e-12)
        assert end == pytest.approx([0.0, -2.0], abs=1e-9)

    def test_a_trajectory_reports_one_row_per_stride_and_its_outputs(self):
        compiled = self._oscillator().compile()
        parameters = compiled.parameter_vector(k=4.0)
        times, rows = compiled.trajectory([1.0, 0.0], parameters, 1.0, step=0.01, stride=10)
        assert times == pytest.approx(np.linspace(0.1, 1.0, 10))
        assert rows[:, 0] == pytest.approx(np.cos(2.0 * times), abs=1e-8)
        table = compiled.output_table(rows, parameters)
        assert table["energy"] == pytest.approx(2.0, rel=1e-8)
        assert compiled.outputs(rows[3], parameters)["energy"] == pytest.approx(table["energy"][3])

    def test_the_compiled_routines_are_their_expressions(self):
        compiled = self._oscillator().compile()
        assert compiled.audit([0.3, -1.1], compiled.parameter_vector(k=4.0)) < 1e-15
        assert compiled.backend == ("c" if find_c_compiler() else "numpy")
