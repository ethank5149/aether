"""Bodies whose shape is the answer to a stated optimisation.

Each of these exists because something about it can be written down, so each
is checked against that rather than against a stored profile. Two kinds of
check appear here and they are not equally strong:

* A **closed-form volume**, compared against the volume OpenCASCADE measures
  on the revolved solid. That tests the geometry pipeline as well as the
  algebra, because the two numbers are arrived at along completely different
  routes -- one from the formula the profile was sampled from, the other from
  the solid built out of that sample.
* An **optimum**, found by sweeping the parameter and locating the minimum
  rather than by asserting the textbook value. A test that hard-codes 3/4
  passes whether or not the shape it built is the one that minimises
  anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from aether.geometry.bodies import (
    power_law_body,
    sears_haack,
    sears_haack_volume,
    sears_haack_wave_drag_area,
    von_karman_ogive,
)

brep = pytest.importorskip("aether.geometry.brep")


def _profile(body):
    return np.asarray(body.station), np.asarray(body.radius)


# ------------------------------------------------------------- power law


@pytest.mark.parametrize("exponent", [0.5, 2.0 / 3.0, 0.75, 1.0])
def test_a_power_law_body_has_the_volume_its_exponent_implies(exponent: float) -> None:
    """:math:`V = \\pi R^2 L/(2n+1)`, measured on the solid, not the formula."""
    length, radius = 2.0, 0.4
    body = power_law_body(exponent=exponent, length=length, base_radius=radius)
    expected = np.pi * radius**2 * length / (2.0 * exponent + 1.0)
    assert brep.solid_properties(body).volume == pytest.approx(expected, rel=1e-3)


def test_the_newtonian_optimum_is_found_rather_than_assumed() -> None:
    """The reason ``exponent=3/4`` is the default: it is where the minimum is.

    Slender Newtonian impact theory gives :math:`C_p = 2(r')^2`, and the
    axial force on a body of revolution is
    :math:`C_D = (2/R^2)\\int C_p\\,r\\,r'\\,dx`. Swept over the family at fixed
    length and base radius, that has an interior minimum, and the minimum is
    the classical 3/4 -- which is a statement this test discovers instead of
    encoding.
    """
    length, radius = 2.0, 0.4

    def drag(exponent: float) -> float:
        x = np.linspace(0.0, length, 40001)[1:]
        r = radius * (x / length) ** exponent
        slope = radius * exponent / length * (x / length) ** (exponent - 1.0)
        return float(2.0 / radius**2 * np.trapezoid(2.0 * slope**2 * r * slope, x))

    exponents = np.linspace(0.55, 0.95, 81)
    best = exponents[int(np.argmin([drag(n) for n in exponents]))]
    assert best == pytest.approx(0.75, abs=0.01)
    # And it really is a minimum, not an endpoint the sweep ran into.
    assert drag(0.75) < drag(0.65)
    assert drag(0.75) < drag(1.0)


def test_the_blast_wave_exponent_is_a_different_shape_from_the_newtonian_one() -> None:
    """Two optima from two arguments; carrying both is only useful if they differ."""
    newtonian = power_law_body(exponent=0.75, base_radius=0.4, length=2.0)
    blast = power_law_body(exponent=2.0 / 3.0, base_radius=0.4, length=2.0)
    _, r_newtonian = _profile(newtonian)
    _, r_blast = _profile(blast)
    # The blast-wave body is fuller forward, everywhere except the endpoints.
    assert np.all(r_blast[1:-1] > r_newtonian[1:-1])


def test_a_power_law_body_is_sharp_at_the_nose_and_meets_its_base() -> None:
    body = power_law_body(exponent=0.75, length=2.0, base_radius=0.4)
    station, radius = _profile(body)
    assert radius[0] == pytest.approx(0.0)
    assert radius[-1] == pytest.approx(0.4)
    assert station[-1] == pytest.approx(2.0)
    assert np.all(np.diff(radius) > 0.0)


@pytest.mark.parametrize("exponent", [0.0, -0.5, 1.5])
def test_an_exponent_outside_the_family_is_refused(exponent: float) -> None:
    with pytest.raises(ValueError, match="exponent must lie"):
        power_law_body(exponent=exponent)


# ----------------------------------------------------------- Sears--Haack


def test_the_sears_haack_volume_matches_its_closed_form() -> None:
    """:math:`V = \\tfrac{3}{16}\\pi^2R_{\\max}^2L`, measured on the solid.

    The formula and the measurement reach the same number along different
    routes -- one from the expression the profile was sampled from, the other
    from the revolved solid OpenCASCADE built out of that sample -- so this
    checks the pipeline and not only the arithmetic.
    """
    length, radius = 2.0, 0.25
    body = sears_haack(length=length, max_radius=radius)
    measured = brep.solid_properties(body).volume
    assert measured == pytest.approx(sears_haack_volume(length, radius), rel=1e-3)


def test_the_two_sears_haack_drag_forms_agree() -> None:
    """:math:`9\\pi^3R^4/(2L^2)` and :math:`128V^2/(\\pi L^4)` are the same number.

    They appear in different places in the literature and it is easy to carry
    one while quoting the other.
    """
    length, radius = 3.0, 0.3
    volume = sears_haack_volume(length, radius)
    assert sears_haack_wave_drag_area(length, radius) == pytest.approx(
        128.0 * volume**2 / (np.pi * length**4)
    )


def test_sears_haack_closes_at_both_ends_and_peaks_in_the_middle() -> None:
    """It is a body, not a forebody -- that is what fixing the volume implies."""
    station, radius = _profile(sears_haack(length=2.0, max_radius=0.25))
    assert radius[0] == pytest.approx(0.0)
    assert radius[-1] == pytest.approx(0.0)
    assert station[int(np.argmax(radius))] == pytest.approx(1.0, abs=0.02)
    assert radius.max() == pytest.approx(0.25, rel=1e-3)


# --------------------------------------------------------- von Karman ogive


def test_the_ogive_meets_its_base_radius_and_is_sharp() -> None:
    """The other constraint: base radius fixed, base left open."""
    length, radius = 2.0, 0.3
    station, profile = _profile(von_karman_ogive(length=length, base_radius=radius))
    assert profile[0] == pytest.approx(0.0)
    assert profile[-1] == pytest.approx(radius, rel=1e-6)
    assert station[-1] == pytest.approx(length)
    assert np.all(np.diff(profile) > 0.0)


def test_the_ogive_is_not_the_same_body_as_sears_haack() -> None:
    """Same variational problem, different constraint, so a different shape.

    Sears--Haack closes at the base because its volume is fixed; the ogive
    carries its full radius there because that radius is the constraint.
    """
    _, ogive = _profile(von_karman_ogive(length=2.0, base_radius=0.3))
    _, haack = _profile(sears_haack(length=2.0, max_radius=0.3))
    assert ogive[-1] > 0.25
    assert haack[-1] < 1e-9


@pytest.mark.parametrize("kwargs", [{"length": 0.0}, {"base_radius": 0.0}, {"length": -1.0}])
def test_degenerate_dimensions_are_refused(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        von_karman_ogive(**kwargs)
