"""Parametric sphere--cone geometry (Paper I §6, the certified exemplar).

The shape is chosen because its sharp-cone limit has an exact solution to
validate against, so these tests check the mesh against analysis rather
than against a stored snapshot.
"""

from __future__ import annotations

import numpy as np
import pytest

from aether.aerodynamics.closure import blended_pressure_coefficient
from aether.aerodynamics.panels import sphere_cone, sphere_cone_closure

_DEFAULTS = (1.75, 0.277, np.radians(8.2))


def test_closure_is_consistent_whichever_parameter_is_solved() -> None:
    """All four branches must return the same shape."""
    length, base, _, angle = sphere_cone_closure(*_DEFAULTS[:2], None, _DEFAULTS[2])
    reference = sphere_cone_closure(length, base, None, angle)
    for omitted in range(4):
        args: list[float | None] = list(reference)
        args[omitted] = None
        got = sphere_cone_closure(*args)
        assert got == pytest.approx(reference, rel=1e-6)


def test_closure_rejects_over_and_under_specification() -> None:
    with pytest.raises(ValueError, match="exactly three"):
        sphere_cone_closure(1.75, 0.277, 0.0286, np.radians(8.2))
    with pytest.raises(ValueError, match="exactly three"):
        sphere_cone_closure(1.75, 0.277, None, None)


def test_nose_cannot_be_blunter_than_the_base() -> None:
    """A closure that would put the nose radius past the base is refused."""
    with pytest.raises(ValueError, match="non-physical nose radius"):
        sphere_cone_closure(length=0.05, base_radius=0.277, nose_radius=None,
                            half_angle=np.radians(8.2))


def test_axial_drag_matches_the_sharp_cone_newtonian_limit() -> None:
    """C_A must approach 2 sin^2(theta) plus the blunt-nose contribution.

    This is the check the shape was chosen for: an independent analytical
    value, not a regression baseline.
    """
    base_radius, nose_radius = 0.277, 0.02865
    for half_angle_deg in (15.0, 25.0, 40.0):
        angle = np.radians(half_angle_deg)
        model = sphere_cone(length=None, base_radius=base_radius,
                            nose_radius=nose_radius, half_angle=angle)
        flow = np.array([1.0, 0.0, 0.0])
        cp = blended_pressure_coefficient(
            np.arcsin(np.clip(-(model.normals @ flow), -1.0, 1.0)), 18.0, cp_max=2.0
        )
        axial = float(
            (-(cp * model.areas)[:, None] * model.normals).sum(axis=0) @ flow
            / (np.pi * base_radius**2)
        )
        sharp = 2.0 * np.sin(angle) ** 2
        nose = (nose_radius * np.cos(angle) / base_radius) ** 2
        assert axial == pytest.approx(sharp + nose, rel=0.02)


def test_axisymmetry_gives_a_trim_at_zero_incidence() -> None:
    """alpha = 0 is an equilibrium by symmetry, and it is statically stable.

    This is the property `curved_lifting_body` lacks at every reference
    point, and without it there is no attitude critical manifold for the
    singular-perturbation argument of §4.5 to reduce onto.
    """
    model = sphere_cone(reference_fraction=0.45)
    arms = model.centroids - model.reference_point

    def pitching(alpha: float) -> float:
        flow = np.array([np.cos(alpha), 0.0, np.sin(alpha)])
        cp = blended_pressure_coefficient(
            np.arcsin(np.clip(-(model.normals @ flow), -1.0, 1.0)), 18.0, cp_max=1.93
        )
        force = -(cp * 1.0e4 * model.areas)[:, None] * model.normals
        return float(np.cross(arms, force).sum(axis=0)[1])

    scale = abs(pitching(np.radians(5.0)))
    assert abs(pitching(0.0)) < 1e-6 * scale
    step = 1e-4
    assert (pitching(step) - pitching(-step)) / (2 * step) < 0.0


def test_normals_point_outward_and_areas_are_positive() -> None:
    model = sphere_cone()
    radial = model.centroids.copy()
    radial[:, 0] = 0.0
    off_axis = np.linalg.norm(radial, axis=1) > 1e-6
    outward = np.einsum(
        "ij,ij->i", model.normals[off_axis],
        radial[off_axis] / np.linalg.norm(radial[off_axis], axis=1)[:, None],
    )
    assert np.all(outward > -1e-9)
    assert np.all(model.areas > 0.0)


def test_base_panels_are_leeward_and_carry_only_expansion_pressure() -> None:
    """The base sits at delta = -90 deg and carries a small suction.

    Not zero: the blended closure hands a leeward panel to the
    Prandtl--Meyer branch, which returns a small *negative* C_p. That is
    physical base suction, and the check is that the closed body's extra
    axial force equals the analytic base term rather than that it equals
    nothing -- an exact-invariance test would pass only if the base had
    been silently rotated into the shadow, which is the bug this catches.
    """
    flow = np.array([1.0, 0.0, 0.0])

    def axial(model: object) -> float:
        cp = blended_pressure_coefficient(
            np.arcsin(np.clip(-(model.normals @ flow), -1.0, 1.0)), 18.0, cp_max=1.93
        )
        return float((-(cp * model.areas)[:, None] * model.normals).sum(axis=0) @ flow)

    open_body = sphere_cone()
    closed = sphere_cone(include_base=True)
    n_base = closed.n_panels - open_body.n_panels
    assert n_base > 0

    base_normals = closed.normals[-n_base:]
    assert np.allclose(base_normals, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(-(base_normals @ flow), -1.0)

    cp_base = float(
        blended_pressure_coefficient(np.array([-0.5 * np.pi]), 18.0, cp_max=1.93)[0]
    )
    assert cp_base < 0.0
    # The base is a 48-gon inscribed in the disc, so its area is short of
    # pi*R^2 by n*sin(2 pi/n)/(2 pi) = 0.9971 -- a 0.3% deficit that is the
    # discretization, not a modelling error.
    n_circ = 48
    polygon = n_circ * np.sin(2.0 * np.pi / n_circ) / (2.0 * np.pi)
    expected = -cp_base * np.pi * 0.277**2 * polygon
    assert axial(closed) - axial(open_body) == pytest.approx(expected, rel=1e-3)
