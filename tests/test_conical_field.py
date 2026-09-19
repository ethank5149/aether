"""The Taylor--Maccoll field between a conical shock and its cone.

:func:`solve_cone` answers what the surface does; this answers what the flow
between the shock and the surface does. It is the machinery an osculating-cone
waverider is designed from, and it is checked here against the things that
must be true of any conical solution rather than against a stored field.

The one that is worth stating because it is easy to mistake for a bug: the
cone surface is an **asymptote**. :math:`V_\\theta` vanishes there -- that is
what "the flow is parallel to the wall" means -- so a streamline approaches
the surface and reaches it only at infinite radius.
"""

from __future__ import annotations

import numpy as np
import pytest

from aether.aerodynamics.conical import (
    ConicalField,
    conical_field,
    oblique_shock,
    solve_cone,
)

_CASES = [(3.0, 15.0), (8.0, 10.0), (8.0, 25.0), (20.0, 12.0)]


@pytest.fixture(scope="module")
def field() -> ConicalField:
    return conical_field(8.0, np.radians(10.0))


@pytest.mark.parametrize(("mach", "degrees"), _CASES)
def test_the_field_agrees_with_the_surface_solution_it_came_from(
    mach: float, degrees: float
) -> None:
    """Same integration, so the boundary values must be identical, not close."""
    angle = np.radians(degrees)
    got, exact = conical_field(mach, angle), solve_cone(mach, angle)
    assert got.shock_angle == pytest.approx(exact.shock_angle, rel=1e-12)
    assert float(got.mach_at(got.cone_angle)[0]) == pytest.approx(exact.surface_mach, rel=1e-9)


@pytest.mark.parametrize(("mach", "degrees"), _CASES)
def test_the_flow_is_parallel_to_the_wall_at_the_wall(mach: float, degrees: float) -> None:
    """:math:`V_\\theta(\\theta_c) = 0` is the boundary condition that defines the cone."""
    got = conical_field(mach, np.radians(degrees))
    _, polar = got.velocity(got.cone_angle)
    assert float(polar.ravel()[0]) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize(("mach", "degrees"), _CASES)
def test_the_flow_turns_inward_behind_the_shock(mach: float, degrees: float) -> None:
    got = conical_field(mach, np.radians(degrees))
    _, polar = got.velocity(got.shock_angle)
    assert float(polar.ravel()[0]) < 0.0


@pytest.mark.parametrize(("mach", "degrees"), _CASES)
def test_mach_falls_monotonically_from_the_shock_to_the_wall(mach: float, degrees: float) -> None:
    """Compression all the way in: no interior extremum in a conical field."""
    got = conical_field(mach, np.radians(degrees))
    angles = np.linspace(got.shock_angle, got.cone_angle, 40)
    local = got.mach_at(angles)
    assert np.all(np.diff(local) < 0.0)
    assert float(local[-1]) == pytest.approx(solve_cone(mach, np.radians(degrees)).surface_mach)


def test_asking_outside_the_field_is_refused_rather_than_extrapolated(
    field: ConicalField,
) -> None:
    for outside in (field.cone_angle - 0.01, field.shock_angle + 0.01):
        with pytest.raises(ValueError, match="theta must lie"):
            field.velocity(outside)


# ------------------------------------------------------------- streamlines


def test_a_streamline_leaves_the_shock_at_the_shock_deflection(field: ConicalField) -> None:
    """Independent check: the trace's initial slope is the oblique-shock turn."""
    x, y = field.streamline_to_station(shock_radius=1.0, station=3.0)
    entry = np.arctan2(y[1] - y[0], x[1] - x[0])
    expected = oblique_shock(field.mach, field.shock_angle).deflection
    assert entry == pytest.approx(expected, abs=1e-4)


def test_a_streamline_approaches_the_cone_angle_but_does_not_reach_it(
    field: ConicalField,
) -> None:
    """The asymptote, measured: further downstream is closer, never equal."""
    angles = []
    for station in (1.5, 3.0, 8.0, 25.0):
        x, y = field.streamline_to_station(shock_radius=1.0, station=station)
        angles.append(np.arctan2(y[-1] - y[-2], x[-1] - x[-2]))
    assert np.all(np.diff(angles) > 0.0)
    assert np.all(np.asarray(angles) < field.cone_angle)
    assert angles[-1] == pytest.approx(field.cone_angle, abs=1e-3)


def test_a_streamline_stays_outside_the_body_and_moves_downstream(
    field: ConicalField,
) -> None:
    x, y = field.streamline_to_station(shock_radius=1.0, station=4.0)
    assert np.all(np.diff(x) > 0.0)
    assert np.all(y > x * np.tan(field.cone_angle))
    assert x[-1] == pytest.approx(4.0, abs=1e-12)


def test_streamlines_are_scaled_copies_of_one_another(field: ConicalField) -> None:
    """Conical flow is self-similar in radius, so the family has one member.

    Doubling the shock crossing radius must double the whole trace; if it
    does not, the integration has picked up a length scale it should not have.
    """
    angles_one, radius_one = field.streamline(shock_radius=1.0)
    angles_two, radius_two = field.streamline(shock_radius=2.5)
    assert angles_one == pytest.approx(angles_two)
    assert radius_two == pytest.approx(2.5 * radius_one, rel=1e-12)


def test_tracing_to_the_wall_is_refused_with_the_reason(field: ConicalField) -> None:
    """It has no answer, and an overflow is not a good way to say so."""
    with pytest.raises(ValueError, match="infinite radius"):
        field.streamline(end_angle=field.cone_angle)


def test_an_unreachable_or_upstream_station_is_refused(field: ConicalField) -> None:
    with pytest.raises(ValueError, match="not downstream"):
        field.streamline_to_station(shock_radius=1.0, station=0.1)
