"""WGS-84 geodesy: the universal format for launch sites and aimpoints."""

import numpy as np
import pytest

from aether.geodesy import (
    WGS84_MEAN_RADIUS,
    WGS84_POLAR_RADIUS,
    WGS84_SEMI_MAJOR_AXIS,
    GeodeticPosition,
    ecef_to_eci,
    ecef_to_geodetic,
    eci_to_ecef,
    geodetic_to_ecef,
    geodetic_to_eci,
    geodetic_to_geocentric_latitude,
    great_circle_bearing,
    great_circle_range,
)


class TestEllipsoid:
    def test_defining_constants_reproduce_the_polar_radius(self):
        assert pytest.approx(21384.7, abs=1.0) == WGS84_SEMI_MAJOR_AXIS - WGS84_POLAR_RADIUS
        assert WGS84_POLAR_RADIUS < WGS84_MEAN_RADIUS < WGS84_SEMI_MAJOR_AXIS

    def test_equatorial_and_polar_points_sit_at_the_right_radii(self):
        equator = geodetic_to_ecef(GeodeticPosition.from_degrees(0.0, 0.0))
        pole = geodetic_to_ecef(GeodeticPosition.from_degrees(90.0, 0.0))
        assert np.linalg.norm(equator) == pytest.approx(WGS84_SEMI_MAJOR_AXIS, rel=1e-12)
        assert np.linalg.norm(pole) == pytest.approx(WGS84_POLAR_RADIUS, rel=1e-9)

    def test_geodetic_and_geocentric_latitude_differ_by_21_km(self):
        """The distinction most often elided, and the one that costs most.
        A survey or GPS reports geodetic; a radius-from-centre calculation
        gives geocentric."""
        geodetic = np.deg2rad(45.0)
        geocentric = float(geodetic_to_geocentric_latitude(geodetic))
        assert np.rad2deg(geodetic - geocentric) == pytest.approx(0.1924, abs=1e-3)
        assert WGS84_MEAN_RADIUS * (geodetic - geocentric) == pytest.approx(21.4e3, rel=0.02)
        # They coincide exactly at the equator and the poles.
        assert float(geodetic_to_geocentric_latitude(0.0)) == pytest.approx(0.0)
        assert float(geodetic_to_geocentric_latitude(np.pi / 2.0)) == pytest.approx(
            np.pi / 2.0, abs=1e-9
        )


class TestConversions:
    def test_round_trip_is_exact_within_the_atmosphere(self):
        rng = np.random.default_rng(3)
        worst = 0.0
        for _ in range(2000):
            point = GeodeticPosition(
                latitude=float(np.arcsin(rng.uniform(-1.0, 1.0))),
                longitude=float(rng.uniform(-np.pi, np.pi)),
                altitude=float(rng.uniform(-500.0, 1.0e5)),
            )
            back = ecef_to_geodetic(geodetic_to_ecef(point))
            worst = max(
                worst,
                abs(back.latitude - point.latitude) * WGS84_MEAN_RADIUS,
                abs(back.altitude - point.altitude),
            )
        assert worst < 1e-8

    def test_round_trip_holds_out_to_geostationary(self):
        """Bowring's closed form alone degrades to 0.32 m here; the
        refinement is what makes the accuracy claim hold at altitude."""
        rng = np.random.default_rng(11)
        worst = 0.0
        for _ in range(2000):
            point = GeodeticPosition(
                latitude=float(np.arcsin(rng.uniform(-1.0, 1.0))),
                longitude=float(rng.uniform(-np.pi, np.pi)),
                altitude=float(rng.uniform(-500.0, 4.2e7)),
            )
            back = ecef_to_geodetic(geodetic_to_ecef(point))
            worst = max(worst, abs(back.altitude - point.altitude))
        assert worst < 1e-6

    def test_polar_axis_is_handled_rather_than_dividing_by_zero(self):
        """Longitude is genuinely undefined there, and reporting zero is
        the only defensible convention."""
        for sign in (1.0, -1.0):
            point = GeodeticPosition(latitude=sign * np.pi / 2.0, longitude=0.0, altitude=1000.0)
            back = ecef_to_geodetic(geodetic_to_ecef(point))
            assert back.latitude == pytest.approx(sign * np.pi / 2.0, abs=1e-9)
            assert back.altitude == pytest.approx(1000.0, abs=1e-6)

    def test_eci_and_ecef_round_trip(self):
        vector = np.array([1.0e6, -2.0e6, 3.0e6])
        for time in (0.0, 1234.5, -600.0):
            assert np.allclose(eci_to_ecef(ecef_to_eci(vector, time), time), vector, atol=1e-6)

    def test_ecef_and_eci_coincide_at_the_epoch(self):
        vector = np.array([1.0e6, -2.0e6, 3.0e6])
        assert np.allclose(ecef_to_eci(vector, 0.0), vector)

    def test_rotation_preserves_length_and_the_polar_component(self):
        vector = np.array([1.0e6, -2.0e6, 3.0e6])
        rotated = ecef_to_eci(vector, 3600.0)
        assert np.linalg.norm(rotated) == pytest.approx(np.linalg.norm(vector))
        assert rotated[2] == pytest.approx(vector[2])

    def test_a_ground_point_moves_through_the_inertial_frame(self):
        """This is why `geodetic_to_eci` demands an arrival epoch rather
        than defaulting one: an equatorial site travels 465 m/s, so a
        one-minute epoch error displaces the aimpoint by 28 km."""
        site = GeodeticPosition.from_degrees(0.0, 0.0)
        displacement = float(
            np.linalg.norm(geodetic_to_eci(site, 60.0) - geodetic_to_eci(site, 0.0))
        )
        assert displacement / 60.0 == pytest.approx(465.0, rel=0.02)
        assert displacement == pytest.approx(27.9e3, rel=0.02)

    def test_a_polar_site_barely_moves(self):
        pole = GeodeticPosition.from_degrees(90.0, 0.0)
        displacement = float(
            np.linalg.norm(geodetic_to_eci(pole, 60.0) - geodetic_to_eci(pole, 0.0))
        )
        assert displacement < 1.0


class TestGreatCircle:
    """Now geodesics on the WGS84 ellipsoid, not great circles on a sphere.

    Both assertions below used to read ``0.5 * pi * WGS84_MEAN_RADIUS``. That is
    the spherical answer, and on an ellipsoid it is wrong for *both* cases in
    opposite directions -- 5.6 km short of a quarter equator and 5.6 km long
    against a quarter meridian. The tests were asserting the model rather than
    the fact, which is invisible while the implementation shares the model.
    """

    def test_a_quarter_of_the_equator_is_a_quarter_of_the_equatorial_circle(self):
        r"""Exactly :math:`\pi a / 2`, since the equator is a circle of radius ``a``.

        An identity rather than a reference value, and matched to the last
        printed digit.
        """
        origin = GeodeticPosition.from_degrees(0.0, 0.0)
        quarter = GeodeticPosition.from_degrees(0.0, 90.0)
        assert great_circle_range(origin, quarter) == pytest.approx(
            0.5 * np.pi * WGS84_SEMI_MAJOR_AXIS, rel=1e-12
        )
        assert great_circle_bearing(origin, quarter) == pytest.approx(np.pi / 2.0, abs=1e-12)

    def test_equator_to_pole_is_the_quarter_meridian(self):
        """10 001 965.7 m, the published WGS84 quarter meridian.

        Shorter than a quarter equator by 16.8 km. That difference *is* the
        oblateness, and a spherical range cannot express it -- one number cannot
        be both.
        """
        origin = GeodeticPosition.from_degrees(0.0, 30.0)
        pole = GeodeticPosition.from_degrees(90.0, 30.0)
        assert great_circle_bearing(origin, pole) == pytest.approx(0.0, abs=1e-12)
        assert great_circle_range(origin, pole) == pytest.approx(
            10_001_965.7, abs=1.0
        )

    def test_the_equator_is_longer_than_the_meridian(self):
        """The property the spherical implementation could not have."""
        equator = great_circle_range(
            GeodeticPosition.from_degrees(0.0, 0.0),
            GeodeticPosition.from_degrees(0.0, 90.0),
        )
        meridian = great_circle_range(
            GeodeticPosition.from_degrees(0.0, 30.0),
            GeodeticPosition.from_degrees(90.0, 30.0),
        )
        assert equator - meridian == pytest.approx(16_788.4, abs=1.0)

    def test_range_is_symmetric_and_zero_for_a_point(self):
        a = GeodeticPosition.from_degrees(45.0, -75.0)
        b = GeodeticPosition.from_degrees(51.5, 0.0)
        assert great_circle_range(a, b) == pytest.approx(great_circle_range(b, a))
        assert great_circle_range(a, a) == pytest.approx(0.0, abs=1e-9)

    def test_coincident_points_have_no_bearing(self):
        a = GeodeticPosition.from_degrees(45.0, -75.0)
        with pytest.raises(ValueError, match="no bearing is defined"):
            great_circle_bearing(a, a)

    def test_bearing_at_arrival_differs_from_departure(self):
        """A great circle does not hold a constant heading, which is why
        the docstring calls this the *initial* bearing."""
        a = GeodeticPosition.from_degrees(60.0, -150.0)
        b = GeodeticPosition.from_degrees(60.0, 150.0)
        outbound = great_circle_bearing(a, b)
        inbound = great_circle_bearing(b, a)
        assert abs(abs(outbound - inbound) - np.pi) > np.deg2rad(5.0)


class TestGeodeticPositionValidation:
    def test_longitude_is_wrapped_and_degrees_round_trip(self):
        point = GeodeticPosition.from_degrees(45.0, 370.0)
        assert point.degrees[1] == pytest.approx(10.0)
        assert point.degrees[0] == pytest.approx(45.0)

    def test_out_of_range_latitude_is_a_mis_ordered_pair_not_a_wrap(self):
        with pytest.raises(ValueError, match="mis-ordered coordinate pair"):
            GeodeticPosition.from_degrees(120.0, 0.0)

    def test_non_finite_inputs_are_rejected(self):
        with pytest.raises(ValueError, match="must be finite"):
            GeodeticPosition(latitude=float("nan"), longitude=0.0)

    def test_label_is_carried_through_conversion(self):
        point = GeodeticPosition.from_degrees(10.0, 20.0, 100.0, label="site")
        assert ecef_to_geodetic(geodetic_to_ecef(point), label="site").label == "site"
