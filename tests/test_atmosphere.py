"""Atmosphere, wind and reanalysis.

The 1976 standard is a *definition*, so the tests are the published table:
the layer base temperatures and pressures are computed by the recursion in
the module and checked against the numbers in NASA-TM-X-74335. Anything that
reproduces those to five figures is the standard; anything that does not is
something else.
"""

from __future__ import annotations

import numpy as np
import pytest

from aether.atmosphere import (
    MODERATE_ACTIVITY,
    SOLAR_MAXIMUM,
    SOLAR_MINIMUM,
    Freestream,
    LayeredAtmosphere,
    MSISAtmosphere,
    NoWind,
    TabulatedWind,
    USStandard1976,
    earth_atmosphere,
    geometric_altitude,
    geopotential_altitude,
    gravity,
    relative_velocity,
    tabulate,
    wind_incidence,
)
from aether.atmosphere.era5 import (
    ERA5Box,
    ERA5Column,
    WindEnsemble,
    reference_level_altitude,
)
from aether.atmosphere.standard import _BASE_PRESSURE, _BASE_TEMPERATURE

#: (geometric altitude m, T K, p Pa, rho kg/m^3) from the published table.
PUBLISHED = [
    (0.0, 288.150, 101325.0, 1.22500),
    (11019.0, 216.650, 22632.1, 0.363918),
    (20063.0, 216.650, 5474.89, 0.0880349),
    (32162.0, 228.650, 868.019, 0.0132250),
    (47350.0, 270.650, 110.906, 0.00142753),
    (51413.0, 270.650, 66.9389, 0.000861600),
    (71802.0, 214.650, 3.95642, 6.42110e-5),
    (86000.0, 186.867, 0.373384, 6.95788e-6),
]


class TestUSStandard1976:
    def test_layer_base_temperatures_match_the_standard(self) -> None:
        expected = [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946]
        assert pytest.approx(expected, abs=1e-3) == _BASE_TEMPERATURE

    def test_layer_base_pressures_match_the_standard(self) -> None:
        expected = [101325.0, 22632.1, 5474.89, 868.019, 110.906, 66.9389, 3.95642,
                    0.373384]
        assert pytest.approx(expected, rel=2e-5) == _BASE_PRESSURE

    @pytest.mark.parametrize(("altitude", "temperature", "pressure", "density"), PUBLISHED)
    def test_published_table(
        self, altitude: float, temperature: float, pressure: float, density: float
    ) -> None:
        state = USStandard1976().state(altitude)
        assert float(state.temperature) == pytest.approx(temperature, rel=1e-5)
        assert float(state.pressure) == pytest.approx(pressure, rel=1e-4)
        assert float(state.density) == pytest.approx(density, rel=1e-4)

    def test_sea_level_derived_quantities(self) -> None:
        state = USStandard1976().state(0.0)
        assert float(state.speed_of_sound) == pytest.approx(340.294, rel=1e-5)
        assert float(state.viscosity) == pytest.approx(1.7894e-5, rel=1e-4)
        assert float(state.mean_free_path) == pytest.approx(6.6328e-8, rel=1e-4)
        assert float(state.number_density) == pytest.approx(2.5470e25, rel=1e-4)

    def test_molar_correction_stitches_the_two_halves(self) -> None:
        """T_M(86 km) * M/M0 must be the upper model's starting temperature.

        186.946 K times 0.999579 is 186.8673 K, which is where the standard's
        thermosphere model begins. The two halves of the standard are joined
        at that number and nothing else pins the 80-86 km molar table.
        """
        standard = USStandard1976()
        assert float(standard.molecular_temperature(86e3)) == pytest.approx(
            186.946, rel=1e-5
        )
        assert float(standard.state(86e3).temperature) == pytest.approx(
            186.8673, rel=1e-5
        )

    def test_density_is_unaffected_by_the_molar_correction(self) -> None:
        """Density follows T_M and M0, so the correction moves only T."""
        standard = USStandard1976()
        z = np.linspace(80e3, 86e3, 25)
        state = standard.state(z)
        expected = (
            state.pressure * 28.9644 / (8.31432e3 * standard.molecular_temperature(z))
        )
        assert state.density == pytest.approx(expected, rel=1e-12)

    def test_gas_law_holds_identically(self) -> None:
        state = USStandard1976().state(np.linspace(-4e3, 86e3, 400))
        residual = state.pressure / (state.density * state.gas_constant * state.temperature)
        assert residual == pytest.approx(np.ones_like(residual), rel=1e-12)

    def test_geopotential_round_trip(self) -> None:
        z = np.linspace(-4e3, 86e3, 50)
        assert geometric_altitude(geopotential_altitude(z)) == pytest.approx(z, abs=1e-6)

    def test_gravity_falls_with_altitude(self) -> None:
        assert float(gravity(0.0)) == pytest.approx(9.80665)
        assert float(gravity(100e3)) < float(gravity(0.0))

    def test_refuses_above_its_ceiling(self) -> None:
        with pytest.raises(ValueError, match=r"86 km|passes\.atmosphere\.EARTH"):
            USStandard1976().state(100e3)

    def test_refuses_non_finite(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            USStandard1976().state(np.nan)

    def test_vectorised_matches_scalar(self) -> None:
        standard = USStandard1976()
        z = np.array([0.0, 12e3, 40e3, 80e3])
        batch = standard.state(z)
        for index, height in enumerate(z):
            one = standard.state(float(height))
            assert float(batch.density[index]) == pytest.approx(float(one.density))

    def test_scalar_input_gives_scalar_shape(self) -> None:
        assert USStandard1976().state(10e3).density.shape == ()


class TestUpperAtmosphere:
    def test_solar_activity_orders_the_thermosphere(self) -> None:
        """Density at 400 km must rise strongly from solar minimum to maximum."""
        heights = 400e3
        quiet = MSISAtmosphere(activity=SOLAR_MINIMUM).state(heights)
        mean = MSISAtmosphere(activity=MODERATE_ACTIVITY).state(heights)
        loud = MSISAtmosphere(activity=SOLAR_MAXIMUM).state(heights)
        assert float(quiet.density) < float(mean.density) < float(loud.density)
        assert float(loud.density) / float(quiet.density) > 5.0

    def test_single_altitude_query(self) -> None:
        """pymsis collapses its output shape when every input has length one."""
        state = MSISAtmosphere().state(300e3)
        assert state.density.shape == ()
        assert 1e-12 < float(state.density) < 1e-10

    def test_molar_mass_falls_with_altitude(self) -> None:
        state = MSISAtmosphere().state(np.array([100e3, 300e3, 700e3]))
        assert np.all(np.diff(state.molar_mass) < 0.0)
        assert float(state.molar_mass[0]) == pytest.approx(28.0, abs=1.5)

    def test_rejects_unknown_version(self) -> None:
        with pytest.raises(ValueError, match="version"):
            MSISAtmosphere(version=1.0)

    def test_refuses_above_its_ceiling(self) -> None:
        with pytest.raises(ValueError, match="defined from"):
            MSISAtmosphere().state(2000e3)


class TestLayeredAtmosphere:
    def test_reproduces_each_model_outside_the_band(self) -> None:
        layered = earth_atmosphere()
        assert float(layered.state(30e3).density) == pytest.approx(
            float(USStandard1976().state(30e3).density), rel=1e-12
        )
        assert float(layered.state(200e3).density) == pytest.approx(
            float(MSISAtmosphere().state(200e3).density), rel=1e-12
        )

    def test_density_is_smooth_across_the_seam(self) -> None:
        """A step in density is a step in drag; the seam must not have one."""
        layered = earth_atmosphere()
        z = np.linspace(75e3, 92e3, 3401)
        curvature = np.abs(np.diff(np.log(layered.state(z).density), 2))
        assert float(np.max(curvature)) < 1e-4

    def test_gas_law_holds_through_the_blend(self) -> None:
        state = earth_atmosphere().state(np.linspace(70e3, 95e3, 200))
        residual = state.pressure / (state.density * state.gas_constant * state.temperature)
        assert residual == pytest.approx(np.ones_like(residual), rel=1e-12)

    def test_density_decreases_monotonically(self) -> None:
        state = earth_atmosphere().state(np.linspace(0.0, 600e3, 2000))
        assert np.all(np.diff(state.density) < 0.0)

    def test_rejects_inverted_band(self) -> None:
        with pytest.raises(ValueError, match="bottom < top"):
            LayeredAtmosphere(blend_bottom=90e3, blend_top=80e3)


class TestTabulatedAtmosphere:
    def test_reproduces_the_layered_model_off_grid(self) -> None:
        layered = earth_atmosphere()
        table = tabulate(layered)
        z = np.array([1234.0, 27_777.0, 63_210.0, 84_321.0, 173_456.0, 456_789.0])
        assert table.state(z).density == pytest.approx(
            layered.state(z).density, rel=2e-4
        )

    def test_gas_law_holds_exactly(self) -> None:
        state = tabulate(earth_atmosphere()).state(np.linspace(0.0, 900e3, 500))
        residual = state.pressure / (state.density * state.gas_constant * state.temperature)
        assert residual == pytest.approx(np.ones_like(residual), rel=1e-12)

    def test_continues_exponentially_above_the_ceiling(self) -> None:
        """Clamping would fly a satellite at 1200 km through 1000 km air."""
        table = tabulate(earth_atmosphere())
        top = float(table.state(1000e3).density)
        beyond = float(table.state(1200e3).density)
        assert 0.0 < beyond < top

    def test_rejects_a_degenerate_grid(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            tabulate(earth_atmosphere(), floor=0.0, ceiling=0.0, samples=5)


class TestFreestream:
    def test_speed_ratio_is_mach_times_sqrt_gamma_over_two(self) -> None:
        stream = earth_atmosphere().freestream(30e3, 1500.0, 3.0)
        assert float(stream.speed_ratio / stream.mach) == pytest.approx(
            np.sqrt(1.4 / 2.0), rel=1e-12
        )

    def test_knudsen_grows_with_altitude(self) -> None:
        atmosphere = earth_atmosphere()
        low = atmosphere.freestream(10e3, 1000.0, 3.0)
        high = atmosphere.freestream(120e3, 1000.0, 3.0)
        assert float(high.knudsen) > 1e4 * float(low.knudsen)

    def test_reynolds_and_knudsen_are_consistent(self) -> None:
        """Kn should be M/Re * sqrt(gamma pi / 2), to within the two models.

        The identity is exact for a hard-sphere gas. Here the mean free path
        comes from a fixed collision diameter and the viscosity from
        Sutherland's law, which are different models of the same molecules,
        so the ratio lands 4 to 14 % high rather than on the nose. Anything
        much outside that would mean one of the two is wrong, not that they
        disagree.
        """
        atmosphere = earth_atmosphere()
        for altitude in (0.0, 20e3, 40e3, 60e3):
            stream = atmosphere.freestream(altitude, 1200.0, 3.0)
            implied = (
                float(stream.mach) / float(stream.reynolds) * np.sqrt(1.4 * np.pi / 2.0)
            )
            assert 1.0 < float(stream.knudsen) / implied < 1.2

    def test_rejects_negative_speed(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            Freestream(earth_atmosphere().state(0.0), np.asarray(-1.0), 3.0)

    def test_rejects_bad_reference_length(self) -> None:
        with pytest.raises(ValueError, match="reference_length"):
            Freestream(earth_atmosphere().state(0.0), np.asarray(1.0), 0.0)


class TestWind:
    @staticmethod
    def profile() -> TabulatedWind:
        z = np.array([100.0, 5e3, 11e3, 20e3, 30e3, 45e3])
        return TabulatedWind(
            altitude=z,
            east=np.array([5.0, 22.0, 60.0, 30.0, 40.0, 10.0]),
            north=np.array([-1.0, 5.0, 2.0, -4.0, 8.0, 0.0]),
        )

    def test_still_air_is_zero_everywhere(self) -> None:
        assert NoWind().velocity(np.array([0.0, 10e3])) == pytest.approx(np.zeros((2, 3)))

    def test_interpolates_through_the_samples(self) -> None:
        wind = self.profile()
        at_samples = wind.velocity(wind.altitude)
        assert at_samples[:, 0] == pytest.approx(wind.east, rel=1e-9)
        assert at_samples[:, 1] == pytest.approx(wind.north, rel=1e-9)

    def test_monotone_interpolation_does_not_overshoot(self) -> None:
        """A spline through a jet core invents a stronger jet just above it.

        The guarantee is per *component*, and only across the span the data
        covers. PCHIP is monotone in each of east and north separately, which
        does not bound the magnitude of the vector they form; and the fade to
        zero above the top sample legitimately leaves the sampled range, since
        an easterly of 5 to 60 m/s has to pass through nothing on its way to
        the ceiling.
        """
        wind = self.profile()
        sampled = wind.velocity(np.linspace(wind.bottom, wind.top, 20_000))
        for axis, values in enumerate((wind.east, wind.north)):
            assert float(np.max(sampled[:, axis])) <= float(np.max(values)) + 1e-9
            assert float(np.min(sampled[:, axis])) >= float(np.min(values)) - 1e-9

    def test_fades_to_zero_above_the_data(self) -> None:
        wind = self.profile()
        assert float(wind.speed(45e3)) > 1.0
        assert float(wind.speed(60e3)) == pytest.approx(0.0, abs=1e-12)

    def test_held_constant_below_the_lowest_sample(self) -> None:
        wind = self.profile()
        assert wind.velocity(0.0) == pytest.approx(wind.velocity(100.0))

    def test_shear_matches_a_finite_difference(self) -> None:
        wind = self.profile()
        z, h = 12e3, 1.0
        numerical = (wind.velocity(z + h) - wind.velocity(z - h)) / (2.0 * h)
        assert wind.shear(z) == pytest.approx(numerical, abs=1e-6)

    def test_shear_is_continuous_through_the_fade(self) -> None:
        """The velocity must be C^1 at the top sample, not merely continuous.

        Applying the fade as a window on a clamped interpolant leaves a slope
        discontinuity of 0.0038 s^-1 exactly at the top sample, because the
        interpolant freezes there while the window is still varying. Building
        the fade into the knot vector removes it.
        """
        wind = self.profile()
        z = np.linspace(44.0e3, 61e3, 20_000)
        jump = float(np.max(np.abs(np.diff(wind.shear(z)[:, 0]))))
        assert jump < 1e-5, f"slope discontinuity of {jump:.3e} at the fade"

    def test_bearing_is_the_direction_the_wind_comes_from(self) -> None:
        """A pure easterly vector (+x) is a wind *from* the west, 270 degrees."""
        wind = TabulatedWind(
            altitude=np.array([0.0, 10e3]),
            east=np.array([10.0, 10.0]),
            north=np.array([0.0, 0.0]),
        )
        assert float(np.rad2deg(wind.bearing(5e3))) == pytest.approx(270.0)

    def test_rejects_non_monotone_altitude(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            TabulatedWind(np.array([0.0, 10.0, 5.0]), np.zeros(3), np.zeros(3))

    def test_rejects_a_ceiling_below_the_data(self) -> None:
        with pytest.raises(ValueError, match="ceiling"):
            TabulatedWind(np.array([0.0, 10e3]), np.zeros(2), np.zeros(2), ceiling=5e3)

    def test_relative_velocity_subtracts_the_wind(self) -> None:
        ground = np.array([100.0, 0.0, 200.0])
        wind = np.array([20.0, 0.0, 0.0])
        assert relative_velocity(ground, wind) == pytest.approx([80.0, 0.0, 200.0])

    def test_wind_incidence_is_the_induced_angle_of_attack(self) -> None:
        """A crosswind of 60 m/s on 400 m/s of flight is 8.53 degrees."""
        ground = np.array([0.0, 0.0, 400.0])
        wind = np.array([60.0, 0.0, 0.0])
        expected = np.arctan2(60.0, 400.0)
        assert float(wind_incidence(ground, wind)) == pytest.approx(expected, rel=1e-9)

    def test_wind_incidence_is_zero_at_rest(self) -> None:
        assert float(wind_incidence(np.zeros(3), np.zeros(3))) == 0.0

    def test_wind_incidence_rejects_non_vectors(self) -> None:
        with pytest.raises(ValueError, match="3-vectors"):
            wind_incidence(np.zeros(2), np.zeros(2))


class TestERA5:
    @staticmethod
    def column() -> ERA5Column:
        pressure = np.array([1000.0, 850.0, 500.0, 250.0, 100.0, 10.0, 1.0]) * 100.0
        return ERA5Column(
            latitude=51.0,
            longitude=60.0,
            epoch="2015-01-01T00:00:00",
            pressure=pressure,
            temperature=np.array([258.0, 257.0, 239.0, 219.0, 210.0, 242.0, 252.0]),
            eastward=np.array([5.0, 11.0, 22.0, 24.0, 25.0, 46.0, -6.0]),
            northward=np.array([-1.0, -7.0, 5.0, 6.0, -11.0, 29.0, 1.0]),
            specific_humidity=np.array([1e-3, 8e-4, 2e-4, 1e-5, 3e-6, 2e-6, 2e-6]),
        )

    def test_reference_level_altitude(self) -> None:
        """1000 hPa sits at about 111 m in the standard atmosphere."""
        assert reference_level_altitude(100000.0) == pytest.approx(110.9, abs=1.0)

    def test_altitudes_increase_and_are_physical(self) -> None:
        z = self.column().altitude
        assert np.all(np.diff(z) > 0.0)
        assert float(z[0]) == pytest.approx(110.9, abs=2.0)
        # 1 hPa is near 48 km in any real atmosphere.
        assert 44e3 < float(z[-1]) < 52e3

    def test_hypsometric_thickness_matches_the_closed_form(self) -> None:
        """The 1000-850 hPa layer must be R T_v ln(p1/p2) / g."""
        column = self.column()
        virtual = column.virtual_temperature
        expected = 287.0528 * 0.5 * (virtual[0] + virtual[1]) / 9.80665 * np.log(
            column.pressure[0] / column.pressure[1]
        )
        geopotential = geopotential_altitude(column.altitude)
        assert float(geopotential[1] - geopotential[0]) == pytest.approx(
            expected, rel=1e-9
        )

    def test_virtual_temperature_exceeds_temperature_in_moist_air(self) -> None:
        column = self.column()
        assert np.all(column.virtual_temperature >= column.temperature)

    def test_wind_profile_round_trips(self) -> None:
        column = self.column()
        wind = column.wind()
        sampled = wind.velocity(column.altitude)
        assert sampled[:, 0] == pytest.approx(column.eastward, rel=1e-9)

    def test_ensemble_statistics(self) -> None:
        columns = []
        for scale in (0.5, 1.0, 1.5, 2.0):
            base = self.column()
            columns.append(
                ERA5Column(
                    latitude=base.latitude,
                    longitude=base.longitude,
                    epoch=base.epoch,
                    pressure=base.pressure,
                    temperature=base.temperature,
                    eastward=base.eastward * scale,
                    northward=base.northward * scale,
                    specific_humidity=base.specific_humidity,
                )
            )
        ensemble = WindEnsemble(tuple(columns))
        assert len(ensemble) == 4
        assert ensemble.speeds().shape == (4, 7)
        median = ensemble.percentile(50.0)
        extreme = ensemble.percentile(100.0)
        assert np.all(extreme >= median)
        # The strongest profile in the jet band is the one scaled by two.
        worst = ensemble.worst((5e3, 16e3))
        assert float(np.max(worst.wind_speed)) == pytest.approx(
            float(np.max(ensemble.speeds())), rel=1e-9
        )

    def test_ensemble_sampling_is_reproducible(self) -> None:
        ensemble = WindEnsemble((self.column(),))
        rng = np.random.default_rng(0)
        assert ensemble.sample(rng) is ensemble.columns[0]

    def test_empty_ensemble_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            WindEnsemble(())

    def test_box_round_trips_through_disk(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        rng = np.random.default_rng(3)
        box = ERA5Box(
            times=np.array(["2015-01-01T00:00:00"], dtype="datetime64[s]"),
            latitude=np.array([53.0, 52.0, 51.0]),
            longitude=np.array([58.0, 59.0, 60.0]),
            pressure=np.array([1e5, 5e4, 1e4]),
            temperature=250.0 + rng.normal(size=(1, 3, 3, 3)),
            eastward=rng.normal(size=(1, 3, 3, 3)),
            northward=rng.normal(size=(1, 3, 3, 3)),
            specific_humidity=np.abs(rng.normal(size=(1, 3, 3, 3))) * 1e-4,
            source="synthetic",
        )
        path = box.save(tmp_path / "box.npz")
        restored = ERA5Box.load(path)
        assert restored.temperature == pytest.approx(box.temperature)
        assert restored.source == "synthetic"

    def test_box_interpolates_on_a_descending_latitude_axis(self) -> None:
        """ERA5 latitude runs 90 to -90; a naive searchsorted gets it backwards."""
        temperature = np.zeros((1, 1, 3, 3))
        temperature[0, 0] = np.array([[300.0] * 3, [200.0] * 3, [100.0] * 3])
        box = ERA5Box(
            times=np.array(["2015-01-01T00:00:00"], dtype="datetime64[s]"),
            latitude=np.array([53.0, 52.0, 51.0]),
            longitude=np.array([58.0, 59.0, 60.0]),
            pressure=np.array([1e5]),
            temperature=temperature,
            eastward=np.zeros((1, 1, 3, 3)),
            northward=np.zeros((1, 1, 3, 3)),
            specific_humidity=np.zeros((1, 1, 3, 3)),
        )
        assert float(box.column(53.0, 59.0).temperature[0]) == pytest.approx(300.0)
        assert float(box.column(52.0, 59.0).temperature[0]) == pytest.approx(200.0)
        assert float(box.column(51.5, 59.0).temperature[0]) == pytest.approx(150.0)
