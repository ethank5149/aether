"""One curve, two representations.

The sphere-cone used to be derived twice -- once in
:mod:`aether.aerodynamics.panels` for impact theory and once in
:mod:`aether.geometry.bodies` for the solid -- with the same algebra written
out in both places under different sampling conventions. They agreed, but
only because nobody had changed one of them yet.

These tests hold the two together. Both now sample
:func:`aether.geometry.profiles.sphere_cone_meridian`, so the check is not
that two derivations happen to match but that neither has quietly acquired a
derivation of its own again.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from aether.geometry.profiles import (
    sphere_cone_closure,
    sphere_cone_meridian,
    sphere_cone_tangency,
)


def test_the_cap_meets_the_cone_with_a_continuous_slope() -> None:
    """C1 by construction: the sphere's slope equals the cone's at tangency.

    That is the whole reason the cap stops where it does, so it is worth
    measuring rather than trusting -- a tangency computed with ``cos`` where
    it wanted ``sin`` still gives a plausible-looking body with a crease in
    it.

    Stated as a *convergence* rather than a tolerance, because the quantity
    available on a sampled profile is a chord slope and a chord is only
    first-order accurate to the tangent it subtends. Asserting a fixed number
    would therefore be asserting a sampling density. What has to hold is that
    the discrepancy is the chord's and vanishes with it: refine the cap
    tenfold and the error falls tenfold. A real crease would not shrink.
    """
    nose_radius, half_angle = 0.06, np.radians(12.0)
    exact = np.tan(half_angle)

    errors = []
    for intervals in (200, 2000, 20000):
        station, radius = sphere_cone_meridian(2.0, nose_radius, half_angle, intervals, 400)
        join = intervals
        approaching = (radius[join] - radius[join - 1]) / (station[join] - station[join - 1])
        leaving = (radius[join + 1] - radius[join]) / (station[join + 1] - station[join])
        # The cone side is a straight line, so it is exact at any sampling.
        assert leaving == pytest.approx(exact, rel=1e-12)
        errors.append(abs(approaching - exact) / exact)

    assert errors[0] < 0.02
    for coarse, fine in itertools.pairwise(errors):
        assert coarse / fine == pytest.approx(10.0, rel=0.05)


def test_the_tangency_point_lies_on_both_the_sphere_and_the_cone() -> None:
    nose_radius, half_angle = 0.05, np.radians(10.0)
    x, r = sphere_cone_tangency(nose_radius, half_angle)
    # on the sphere centred at (nose_radius, 0)
    assert np.hypot(x - nose_radius, r) == pytest.approx(nose_radius)
    # and the cone through it, extended back, has the right half-angle
    assert np.arctan2(r, nose_radius - x) == pytest.approx(0.5 * np.pi - half_angle)


def test_the_profile_starts_at_the_nose_and_advances() -> None:
    station, radius = sphere_cone_meridian(2.0, 0.05, np.radians(10.0), 30, 30)
    assert station[0] == pytest.approx(0.0)
    assert radius[0] == pytest.approx(0.0)
    assert np.all(np.diff(station) > 0.0)
    assert np.all(np.diff(radius) > 0.0)


def test_a_body_that_ends_inside_its_own_nose_is_refused() -> None:
    with pytest.raises(ValueError, match="does not reach past the nose cap"):
        sphere_cone_meridian(0.01, 0.5, np.radians(10.0), 10, 10)


def test_a_segment_with_no_intervals_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one interval"):
        sphere_cone_meridian(2.0, 0.05, np.radians(10.0), 0, 10)


def test_the_panel_body_and_the_solid_body_are_the_same_curve() -> None:
    """The consolidation, asserted where it can actually be seen.

    Built through the two independent public entry points at matching
    parameters, sampled onto a common set of stations, and compared. A future
    edit that re-derived either one would move them apart.
    """
    from aether.aerodynamics.panels import sphere_cone as panel_body
    from aether.geometry.bodies import sphere_cone as solid_body

    length, nose_radius, half_angle = 2.0, 0.05, np.radians(10.0)
    _, base_radius, _, _ = sphere_cone_closure(
        length=length, base_radius=None, nose_radius=nose_radius, half_angle=half_angle
    )

    panels = panel_body(
        length=length,
        base_radius=base_radius,
        nose_radius=None,
        half_angle=half_angle,
        n_axial=200,
        n_circ=64,
    )
    solid = solid_body(half_angle=half_angle, nose_radius=nose_radius, length=length)

    # The panel model's meridian, recovered from its own vertex net.
    net = panels.surface.vertices
    panel_station = net[:, 0, 0]
    panel_radius = np.linalg.norm(net[:, 0, 1:], axis=1)

    stations = np.linspace(0.15, length * 0.98, 60)
    assert np.interp(stations, panel_station, panel_radius) == pytest.approx(
        np.interp(stations, np.asarray(solid.station), np.asarray(solid.radius)), rel=2e-4
    )


def test_the_closure_is_consistent_from_whichever_parameter_is_solved() -> None:
    """Moved here from the panel module with the rest of the shape maths."""
    defaults = (1.75, 0.277, np.radians(8.2))
    length, base, nose, angle = sphere_cone_closure(*defaults[:2], None, defaults[2])
    reference = (length, base, nose, angle)
    for omitted in range(4):
        args: list[float | None] = list(reference)
        args[omitted] = None
        assert sphere_cone_closure(*args) == pytest.approx(reference, rel=1e-6)


def test_over_and_under_specifying_the_closure_is_refused() -> None:
    with pytest.raises(ValueError, match="exactly three"):
        sphere_cone_closure(1.75, 0.277, 0.0286, np.radians(8.2))
    with pytest.raises(ValueError, match="exactly three"):
        sphere_cone_closure(1.75, 0.277, None, None)
