"""Caret waverider geometry (Paper I §6, the lifting exemplar).

The cone is the certified exemplar because its sharp limit has an exact
solution; the waverider is here because the cone cannot exercise the
objects the reduction is about. An axisymmetric body trims at zero
incidence where it makes no lift, so it flies ballistically: no
equilibrium glide manifold, no skip, no bubbles.
"""

from __future__ import annotations

import numpy as np
import pytest

from aether.aerodynamics.closure import blended_pressure_coefficient
from aether.aerodynamics.panels import caret_waverider


def _coefficients(model: object, alpha: float) -> tuple[float, float]:
    flow = np.array([np.cos(alpha), 0.0, np.sin(alpha)])
    cp = blended_pressure_coefficient(
        np.arcsin(np.clip(-(model.normals @ flow), -1.0, 1.0)), 18.0, cp_max=1.93
    )
    force = (-(cp * model.areas)[:, None] * model.normals).sum(axis=0)
    reference = model.total_area / 2.0
    lift = float(force @ np.array([-np.sin(alpha), 0.0, np.cos(alpha)]) / reference)
    drag = float(force @ flow / reference)
    return lift, drag


def test_it_lifts_at_zero_incidence() -> None:
    """The defining waverider property, and what the cone cannot do.

    An axisymmetric body has C_L identically zero at alpha = 0 by
    symmetry. The caret's lower surface is inclined, so it carries lift
    there -- which is why a glide manifold exists for this shape class.
    """
    lift, drag = _coefficients(caret_waverider(), 0.0)
    assert lift > 0.0
    assert drag > 0.0
    assert lift / drag > 4.0


def test_lift_to_drag_falls_with_incidence() -> None:
    """L/D peaks near the design point and decays as drag grows like alpha^2."""
    model = caret_waverider()
    ratios = [
        _coefficients(model, np.radians(a))[0] / _coefficients(model, np.radians(a))[1]
        for a in (0.0, 5.0, 10.0, 20.0)
    ]
    assert ratios == sorted(ratios, reverse=True)
    assert ratios[0] > 6.0


def test_deeper_keel_compresses_more_and_costs_lift_to_drag() -> None:
    """Keel depth sets the flow deflection, so it trades L/D for lift."""
    shallow = _coefficients(caret_waverider(keel_depth=0.16), np.radians(5.0))
    deep = _coefficients(caret_waverider(keel_depth=0.64), np.radians(5.0))
    assert deep[0] > shallow[0]
    assert deep[0] / deep[1] < shallow[0] / shallow[1]


def test_geometry_is_symmetric_about_the_vertical_plane() -> None:
    """A symmetric body makes no side force or rolling moment at zero sideslip."""
    model = caret_waverider()
    flow = np.array([np.cos(0.1), 0.0, np.sin(0.1)])
    cp = blended_pressure_coefficient(
        np.arcsin(np.clip(-(model.normals @ flow), -1.0, 1.0)), 18.0, cp_max=1.93
    )
    panel_force = -(cp * model.areas)[:, None] * model.normals
    force = panel_force.sum(axis=0)
    arms = model.centroids - model.reference_point
    moment = np.cross(arms, panel_force).sum(axis=0)
    scale = float(np.linalg.norm(force))
    assert abs(force[1]) < 1e-9 * scale
    assert abs(moment[0]) < 4e-9 * scale


def test_rejects_degenerate_dimensions() -> None:
    for kwargs in ({"length": 0.0}, {"semi_span": -1.0}, {"keel_depth": np.nan}):
        with pytest.raises(ValueError, match="must be finite"):
            caret_waverider(**kwargs)
    with pytest.raises(ValueError, match="n_chord"):
        caret_waverider(n_chord=2)


def test_normals_are_outward_on_both_surfaces() -> None:
    """Upper panels face +z, lower panels face -z; none may be inverted."""
    model = caret_waverider(include_base=False)
    upper = model.centroids[:, 2] > -1e-9
    assert np.all(model.normals[upper][:, 2] > 0.0)
    assert np.all(model.normals[~upper][:, 2] < 0.0)
