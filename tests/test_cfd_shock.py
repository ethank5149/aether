"""Billig's hyperbola, checked against its own limits and against the paper.

The equation is transcribed from the source rather than recalled, because this
project has already spent a day on a correlation applied to the wrong geometry
and a citation attached to the wrong author.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aether.cfd.geometries import REFERENCE_BODIES
from aether.cfd.shock import billig_shock, envelope_for


def test_the_vertex_sits_exactly_one_standoff_ahead_of_the_nose() -> None:
    """At y = 0 Eq. (1) collapses to x = R + Delta, which is the definition."""
    shock = billig_shock(8.0, 0.06, cone_half_angle=10.0)
    assert shock.standoff == pytest.approx(0.143 * math.exp(3.24 / 64.0) * 0.06)
    assert float(shock.axial_at(0.0)) == pytest.approx(-shock.standoff)


def test_the_hyperbola_tends_to_the_cone_shock_angle_not_the_mach_angle() -> None:
    """Billig: asymptotic to the freestream Mach angle, *or* the attached shock
    angle for a cone or wedge afterbody. The two are not close.

    A 10-degree cone at Mach 8 turns its shock through about 13 degrees where
    the Mach angle is 7.2, so the choice moves the shock by a factor of two in
    radius by mid-body.
    """
    coned = billig_shock(8.0, 0.06, cone_half_angle=10.0)
    blunt = billig_shock(8.0, 0.06, cone_half_angle=None)

    assert blunt.asymptote == pytest.approx(math.asin(1.0 / 8.0))
    assert math.degrees(coned.asymptote) > math.degrees(blunt.asymptote) + 4.0

    # Far downstream the local slope must reach the declared asymptote.
    y = np.array([20.0, 100.0])
    slope = math.degrees(math.atan(np.diff(y)[0] / np.diff(coned.axial_at(y))[0]))
    assert slope == pytest.approx(math.degrees(coned.asymptote), abs=0.05)


def test_standoff_falls_toward_the_mach_independent_limit() -> None:
    """exp(3.24/M^2) -> 1, so Delta/R -> 0.143 from above and never below."""
    ratios = [billig_shock(m, 1.0).standoff for m in (3.0, 6.0, 12.0, 30.0)]
    assert ratios == sorted(ratios, reverse=True)
    assert ratios[-1] > 0.143
    assert ratios[-1] == pytest.approx(0.143, abs=0.002)


def test_radius_and_axial_invert_each_other() -> None:
    shock = billig_shock(8.0, 0.06, cone_half_angle=10.0)
    y = np.array([0.001, 0.01, 0.05, 0.2])
    assert np.allclose(shock.radius_at(shock.axial_at(y)), y, rtol=1e-9)


def test_subsonic_has_no_detached_bow_shock_to_place() -> None:
    with pytest.raises(ValueError, match="supersonic"):
        billig_shock(0.8, 0.06)


def test_there_is_no_shock_radius_ahead_of_the_vertex() -> None:
    """NaN, and only there -- the mask must not leak across the whole array."""
    shock = billig_shock(8.0, 0.06, cone_half_angle=10.0)
    ahead, at, behind = shock.radius_at(np.array([-0.05, -shock.standoff, 0.5]))
    assert np.isnan(ahead)
    assert at == pytest.approx(0.0, abs=1e-9)
    assert np.isfinite(behind) and behind > 0.0


# ------------------------------------------------------------ the provider layer


def test_every_provider_is_negative_inside_and_positive_outside() -> None:
    """The one contract the extruder relies on, checked for all four.

    ``outside`` must be positive on the freestream side and negative between
    shock and body, so marching a normal outward finds the shock by a single
    sign change. A provider that gets this backwards would send the extruder
    the wrong way and produce a mesh with no error.
    """
    from aether.cfd.shock import ConicalShock, MachCone, Validity, WedgeShock

    valid = Validity(provider="t", why="t", mach=(1.0, 30.0))
    providers = [
        billig_shock(8.0, 0.06, cone_half_angle=10.0),
        ConicalShock(apex=0.0, shock_angle=math.radians(13.0), validity=valid),
        WedgeShock(apex=0.0, shock_angle=math.radians(12.0), validity=valid),
        MachCone(apex=0.0, mach=8.0, validity=valid),
    ]
    for provider in providers:
        # A point far out in the freestream is outside; one hugging the axis
        # well downstream is inside.
        far = provider.outside(np.array([1.0]), np.array([5.0]))
        near = provider.outside(np.array([1.0]), np.array([1e-4]))
        assert far[0] > 0.0, type(provider).__name__
        assert near[0] < 0.0, type(provider).__name__


def test_marching_a_normal_recovers_billigs_own_standoff() -> None:
    """The march is only trustworthy if it reproduces the closed form.

    Straight up the stagnation streamline from the nose, the distance to the
    shock must be the standoff -- which the hyperbola knows analytically.
    """
    from aether.cfd.shock import _march_to_shock

    shock = billig_shock(8.0, 0.06, cone_half_angle=10.0)
    nose = np.array([0.0, 0.0, 0.0])
    upstream = np.array([-1.0, 0.0, 0.0])
    distance = _march_to_shock(shock, nose, upstream, limit=0.05)
    assert distance == pytest.approx(shock.standoff, rel=1e-4)


def test_a_normal_that_never_crosses_returns_the_limit_not_an_error() -> None:
    """Aft-facing normals run down the body, not out through the shock."""
    from aether.cfd.shock import _march_to_shock

    shock = billig_shock(8.0, 0.06, cone_half_angle=10.0)
    downstream = np.array([1.0, 0.0, 0.0])
    assert _march_to_shock(shock, np.array([0.5, 0.0, 0.0]), downstream, limit=0.02) == 0.02


def test_the_selector_picks_a_provider_and_says_why() -> None:
    from aether.cfd.shock import envelope_for

    blunt = envelope_for("sphere-cone", {"nose_radius": 0.06, "half_angle": 10.0}, 8.0)
    assert blunt.validity.provider == "BilligShock"
    assert "blunt nose" in blunt.validity.why

    waverider = envelope_for("caret-waverider", {"wedge_angle": 6.0}, 8.0)
    assert waverider.validity.provider == "WedgeShock"
    # A waverider rides an *attached* shock; a detached correlation would be wrong.
    assert waverider.shock_angle < math.radians(20.0)


def test_the_selector_refuses_a_body_nobody_has_covered() -> None:
    """Silence here would mesh the wrong place and look identical to success.

    Named against a body that genuinely has no provider rather than against
    ``lifting-body``, which now has one. Every shape in
    :data:`REFERENCE_BODIES` is covered, so the guard is exercised on a name
    from outside it -- the refusal is the point, not which body triggers it.
    """
    from aether.cfd.shock import envelope_for

    assert "blended-wing-body" not in REFERENCE_BODIES
    with pytest.raises(ValueError, match="no shock provider covers"):
        envelope_for("blended-wing-body", {"length": 2.9}, 8.0)


@pytest.mark.parametrize("name", sorted(REFERENCE_BODIES))
def test_every_reference_body_now_has_a_provider(name: str) -> None:
    """``lifting-body`` had none, so it could not be shock-fitted at all."""
    envelope = envelope_for(name, REFERENCE_BODIES[name].defaults(), 8.0)
    assert envelope.validity.provider


def test_above_mach_ten_billig_is_marked_an_envelope_not_a_prediction() -> None:
    """Billig's correlations rest on perfect-gas data; he says so."""
    from aether.cfd.shock import envelope_for

    geometry = {"nose_radius": 0.06, "half_angle": 10.0}
    assert envelope_for("sphere-cone", geometry, 8.0).validity.trustworthy
    high = envelope_for("sphere-cone", geometry, 16.0).validity
    assert not high.trustworthy
    assert any("envelope" in c for c in high.caveats)


def test_validity_refuses_outside_its_declared_mach_range() -> None:
    from aether.cfd.shock import Validity

    validity = Validity(provider="Test", why="t", mach=(2.0, 10.0))
    validity.check(5.0)
    with pytest.raises(ValueError, match="Refusing rather than extrapolating"):
        validity.check(25.0)


def test_the_envelope_is_built_about_the_freestream_not_the_body_axis() -> None:
    """At incidence those are different lines, and the difference is not small.

    The bow shock stands about the velocity vector. On a 2 m sphere-cone at
    6 degrees, a body-axis envelope is displaced by 2*tan(6) = 0.21 m against a
    9 mm standoff -- and since the outer boundary now *imposes* freestream
    rather than extrapolating it, a misplaced envelope asserts undisturbed flow
    inside the shock layer rather than merely meshing it oddly.

    The stagnation point moves too: on a spherical nose of radius R centred at
    C it is exactly C - R*v, where the surface normal is anti-parallel to the
    flow.
    """
    import numpy as np

    from aether.cfd.shock import envelope_for, freestream_axis

    geometry = {"nose_radius": 0.06, "length": 2.0, "half_angle": 10.0}
    axial = envelope_for("sphere-cone", geometry, 8.0)
    assert np.allclose(axial.origin, [0.0, 0.0, 0.0])
    assert np.allclose(axial.axis, [1.0, 0.0, 0.0])

    alpha = np.radians(6.0)
    tilted = envelope_for("sphere-cone", geometry, 8.0, alpha=alpha)
    assert np.allclose(tilted.axis, freestream_axis(alpha, 0.0))
    # Stagnation point: exactly one nose radius upstream of the sphere centre.
    centre = np.array([0.06, 0.0, 0.0])
    assert np.allclose(tilted.origin, centre - 0.06 * tilted.axis)
    # It moves windward -- the -z side for a positive alpha.
    assert tilted.origin[2] < -1.0e-4

    # An axisymmetric query is refused rather than quietly answered wrongly.
    with pytest.raises(ValueError, match="not well posed"):
        tilted.outside(np.array([0.0]), np.array([0.0]))


def test_the_asymptote_varies_around_the_flow_axis_at_incidence() -> None:
    """One asymptote is a body of revolution; the shock at incidence is not.

    Windward the shock is steeper and leeward shallower, so a single angle can
    only enclose both by taking the windward value everywhere -- which puts the
    leeward boundary hundreds of millimetres outside a shock that is tens away.
    Measured on the sphere-cone at 6 degrees, that left **6.5 % of columns
    unable to reach their own envelope**, every one of them leeward and aft.
    """
    import numpy as np

    from aether.aerodynamics import solve_cone
    from aether.cfd.shock import envelope_for

    geometry = {"nose_radius": 0.06, "length": 2.0, "half_angle": 10.0}
    tilted = envelope_for("sphere-cone", geometry, 8.0, alpha=np.radians(6.0))

    windward = float(tilted.asymptote_at(0.0))
    leeward = float(tilted.asymptote_at(np.pi))
    assert windward > leeward

    # Each meridian carries the Taylor-Maccoll angle for its own equivalent
    # cone: 10 + 6 windward, 10 - 6 leeward.
    assert windward == pytest.approx(solve_cone(8.0, np.radians(16.0)).shock_angle, rel=1e-6)
    assert leeward == pytest.approx(solve_cone(8.0, np.radians(4.0)).shock_angle, rel=1e-6)

    # And it is an envelope, not a standoff to quote.
    assert tilted.validity.trustworthy is False


# --------------------------------------------------- the envelope must enclose
#
# Seven of the ten reference bodies lay partly outside their own shock envelope.
# Two causes, both silent: the waveriders are built off-axis -- a waverider
# rides under its shock, so the caret occupies z in [-0.829, -0.002] and the
# osculating-cone body starts at x = 0.987 -- while the envelope was anchored at
# the coordinate origin; and the slender bodies were bounded by the angle to
# their base rim, which does not bound a Sears-Haack spindle whose widest
# station is at mid-length.
#
# A shock-fitted mesh extrudes from the body to this surface. A body outside it
# is a body outside its own domain.


@pytest.mark.parametrize("name", sorted(REFERENCE_BODIES))
@pytest.mark.parametrize(("mach", "alpha_deg"), [(4.0, 0.0), (8.0, 6.0)])
def test_every_envelope_encloses_its_body(name: str, mach: float, alpha_deg: float) -> None:
    from aether.cfd.geometries import build_surface

    body = REFERENCE_BODIES[name]
    geometry = body.defaults()
    vertices = np.asarray(build_surface(name, **geometry).vertices, dtype=float)
    envelope = envelope_for(name, geometry, mach, alpha=np.radians(alpha_deg))
    signed = np.asarray(envelope.at(vertices))
    outside = int((signed > 0.0).sum())
    assert outside == 0, (
        f"{name} at Mach {mach:g}, alpha {alpha_deg:g} has {outside} of "
        f"{len(vertices)} vertices outside its envelope, worst {signed.max():.4f} m"
    )


def test_a_cone_envelope_starts_upstream_of_a_blunt_nose() -> None:
    """A cone anchored exactly on the apex has zero radius there, so any blunt
    nose pokes out of it -- the caret's 3 mm leading-edge radius put 95 vertices
    outside over the first 0.9 % of its length. The shift that fixes it is
    exact, not tuned."""
    from aether.cfd.shock import cone_standoff, freestream_axis

    axis = freestream_axis(0.0, 0.0)
    geometry = REFERENCE_BODIES["caret-waverider"].defaults()
    shift = cone_standoff("caret-waverider", geometry, axis, np.radians(14.0))
    assert shift > 0.0, "a blunted body needs its cone to start upstream of it"
    # And a sharper cone needs to start further upstream to contain the same body.
    assert cone_standoff("caret-waverider", geometry, axis, np.radians(8.0)) > shift


def test_the_effective_nose_radius_is_solved_and_reported_not_assumed() -> None:
    """The lifting body's forward quarter sat up to 70 mm outside a hyperbola
    built on its 0.1 m geometric nose radius. The radius that encloses it is
    found by search, and what comes out is a meshing radius rather than the
    body's bluntness -- so it has to say so."""
    from aether.cfd.shock import enclosing_nose_radius, freestream_axis

    geometry = REFERENCE_BODIES["lifting-body"].defaults()
    radius, inflation = enclosing_nose_radius(
        "lifting-body", geometry, freestream_axis(0.0, 0.0), 8.0, 19.0
    )
    assert inflation > 1.0, "the geometric radius did not enclose this body"
    assert radius == pytest.approx(geometry["nose_radius"] * inflation, rel=1e-9)
    validity = envelope_for("lifting-body", geometry, 8.0).validity
    assert not validity.trustworthy or "effective" in " ".join(validity.caveats)
    assert any("meshing radius" in c for c in validity.caveats)
