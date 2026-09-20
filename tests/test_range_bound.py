"""The certified downrange bound, and the density enclosure under it."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.integrate

from aether.atmosphere.model import earth_atmosphere, tabulate
from aether.certification.atmosphere import monotone_density
from aether.certification.range_bound import (
    Corridor,
    certified_downrange_bound,
    drag_lower_bound,
)
from aether.certification.rigorous import interval, outward

_TABLE = tabulate(earth_atmosphere())
_DENSITY = monotone_density(_TABLE)
_BETA = 281.0
_R_EARTH = 6378137.0
_MU = 3.986004418e14


class TestDensityEnclosure:
    def test_it_contains_the_table_it_encloses(self):
        """Sampled densities inside the band must lie inside the enclosure."""
        low, high = 30e3, 70e3
        band = outward(_DENSITY(interval(low, high)))
        sampled = [float(_TABLE.state(h).density) for h in np.linspace(low, high, 400)]
        assert band[0] <= min(sampled)
        assert max(sampled) <= band[1]

    def test_a_scale_factor_carries_a_class_of_atmospheres(self):
        """Artemis I met air about 10 % thinner than the standard model."""
        plain = outward(_DENSITY(interval(50e3, 50e3)))
        scaled = outward(monotone_density(_TABLE, scale=(0.85, 1.15))(interval(50e3, 50e3)))
        assert scaled[0] < plain[0]
        assert scaled[1] > plain[1]

    def test_a_non_monotone_table_is_refused(self):
        """The endpoint argument is unsound without monotonicity, so it is checked."""
        broken = tabulate(earth_atmosphere())
        object.__setattr__(broken, "log_density", np.zeros_like(broken.log_density))
        with pytest.raises(ValueError, match="not strictly decreasing"):
            monotone_density(broken)


class TestDragFloor:
    def test_the_floor_is_positive_and_below_the_flown_drag(self):
        corridor = Corridor(
            altitude_ceiling_m=60e3, altitude_floor_m=20e3,
            speed_high_ms=7000.0, speed_low_ms=500.0,
        )
        floor = drag_lower_bound(corridor, _BETA, _DENSITY, subdivisions=16)
        assert floor > 0.0
        # Any state in the corridor produces at least the floor.
        for altitude in (25e3, 40e3, 55e3):
            for speed in (600.0, 3000.0, 6900.0):
                drag = 0.5 * float(_TABLE.state(altitude).density) * speed**2 / _BETA
                assert drag >= floor


class TestCertifiedRange:
    @staticmethod
    def _corridor(ceiling: float = 60e3) -> Corridor:
        return Corridor(
            altitude_ceiling_m=ceiling, altitude_floor_m=20e3,
            speed_high_ms=7387.0, speed_low_ms=304.8,
        )

    def test_it_converges_from_above_to_the_closed_form(self):
        r"""Banding the energy interval drives the kinetic part of the sum down
        to :math:`(2\beta/\rho)\ln(V_{high}/V_{low})`, and never below it. The
        potential term, charged to the weakest drag floor, is added whole."""
        corridor = self._corridor()
        rho = float(_TABLE.state(corridor.altitude_ceiling_m).density)
        kinetic = 2.0 * _BETA / rho * np.log(corridor.speed_high_ms / corridor.speed_low_ms)
        weakest = rho * corridor.speed_low_ms**2 / (2.0 * _BETA)
        potential = _MU * (
            1.0 / (_R_EARTH + corridor.altitude_floor_m)
            - 1.0 / (_R_EARTH + corridor.altitude_ceiling_m)
        )
        closed_form = kinetic + potential / weakest
        bounds = [
            certified_downrange_bound(corridor, _BETA, _DENSITY, subdivisions=n).max_downrange_m
            for n in (16, 256, 2048)
        ]
        assert bounds == sorted(bounds, reverse=True)
        assert bounds[-1] >= closed_form
        assert bounds[-1] == pytest.approx(closed_form, rel=0.01)

    def test_the_potential_term_is_carried(self):
        """A body descending sheds potential energy as well as speed, and range
        bought with it has to be inside the bound. Deepening the corridor at a
        fixed ceiling therefore raises the bound, though nothing about the
        speeds has changed."""
        shallow = Corridor(
            altitude_ceiling_m=60e3, altitude_floor_m=50e3,
            speed_high_ms=7387.0, speed_low_ms=304.8,
        )
        deep = Corridor(
            altitude_ceiling_m=60e3, altitude_floor_m=0.0,
            speed_high_ms=7387.0, speed_low_ms=304.8,
        )
        assert (
            certified_downrange_bound(deep, _BETA, _DENSITY).max_downrange_m
            > certified_downrange_bound(shallow, _BETA, _DENSITY).max_downrange_m
        )

    def test_a_higher_ceiling_buys_more_range(self):
        """Thinner air means less energy lost per kilometre covered."""
        low = certified_downrange_bound(self._corridor(50e3), _BETA, _DENSITY)
        high = certified_downrange_bound(self._corridor(70e3), _BETA, _DENSITY)
        assert high.max_downrange_m > low.max_downrange_m

    def test_it_bounds_trajectories_that_stay_in_the_corridor(self):
        """The point of the thing: fly the planar equations at several constant
        banks, keep the ones that stay inside the corridor, and none of them
        may exceed the bound."""
        corridor = self._corridor(70e3)
        bound = certified_downrange_bound(corridor, _BETA, _DENSITY, subdivisions=1024)
        flown = []
        for u in (-0.6, -0.3, 0.0, 0.3, 0.6):
            def rhs(_t, y, u=u):
                r, _s, v, gamma = y
                altitude = r - _R_EARTH
                drag = 0.5 * float(_TABLE.state(max(altitude, 0.0)).density) * v * v / _BETA
                gravity = _MU / (r * r)
                return [
                    v * np.sin(gamma),
                    v * np.cos(gamma) / r,
                    -drag - gravity * np.sin(gamma),
                    (0.3 * drag * u + (v * v / r - gravity) * np.cos(gamma)) / v,
                ]

            def slow(_t, y):
                return y[2] - corridor.speed_low_ms

            slow.terminal = True
            start = _R_EARTH + 65e3
            solution = scipy.integrate.solve_ivp(
                rhs, (0.0, 3000.0), [start, 0.0, corridor.speed_high_ms, np.deg2rad(-1.0)],
                events=slow, rtol=1e-8, atol=1e-8, max_step=5.0,
            )
            altitudes = solution.y[0] - _R_EARTH
            inside = (
                altitudes.max() <= corridor.altitude_ceiling_m
                and altitudes.min() >= corridor.altitude_floor_m
                and solution.t_events[0].size > 0
            )
            if inside:
                flown.append(float(solution.y[1, -1]) * _R_EARTH)
        assert flown, "no sampled trajectory stayed inside the corridor to compare against"
        assert max(flown) <= bound.max_downrange_m

    def test_the_bound_carries_its_hypothesis(self):
        bound = certified_downrange_bound(self._corridor(), _BETA, _DENSITY)
        assert "altitude in [20, 60] km" in bound.corridor.describe()
        assert "downrange at most" in bound.summary()

    def test_a_ceiling_in_thin_air_makes_the_bound_worthless_not_wrong(self):
        """Raise the ceiling and the bound degrades gracefully rather than
        failing: at 2000 km the air still encloses a positive density, so a
        finite bound follows, but it is astronomically large. The hypothesis is
        what buys a useful number, which is why it travels with it."""
        corridor = Corridor(
            altitude_ceiling_m=2.0e6, altitude_floor_m=20e3,
            speed_high_ms=7000.0, speed_low_ms=500.0,
        )
        bound = certified_downrange_bound(corridor, _BETA, _DENSITY, subdivisions=64)
        assert bound.max_downrange_m > 1.0e12

    def test_zero_density_is_refused_rather_than_divided_by(self):
        """A vacuum encloses no drag floor, and no bound follows from one."""
        corridor = Corridor(
            altitude_ceiling_m=60e3, altitude_floor_m=20e3,
            speed_high_ms=7000.0, speed_low_ms=500.0,
        )
        with pytest.raises(ValueError, match="no finite range bound"):
            certified_downrange_bound(corridor, _BETA, lambda _altitude: interval(0.0, 0.0))
