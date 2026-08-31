"""Drawing a cut field.

The drawing itself is not checked -- an image comparison would fail on a
Matplotlib version bump and pass on a physically wrong picture, which is the
wrong way round. What is checked is everything with a defined answer: the
Schlieren transfer function, the resampling that streamlines need, and the
refusals that stop a mismatched field from being drawn against a mesh it does
not belong to.
"""

from __future__ import annotations

import numpy as np
import pytest

from aether.aerodynamics.cfd.fields import Mesh, plane_cut
from aether.viz import flow

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

_CORNERS = np.array(
    [[x, y, z] for z in (0.0, 1.0) for y in (0.0, 1.0) for x in (0.0, 1.0)], dtype=np.float64
)
_KUHN = np.array(
    [[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]],
    dtype=np.int64,
)


@pytest.fixture
def cube() -> Mesh:
    return Mesh(points=_CORNERS.copy(), tetrahedra=_KUHN.copy(), markers={})


@pytest.fixture
def axes():
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()
    yield ax
    plt.close(figure)


# ------------------------------------------------------- transfer function


def test_undisturbed_flow_is_white() -> None:
    """Zero gradient must map to 1, or the whole image is grey."""
    assert flow.schlieren_intensity(np.zeros(5)) == pytest.approx(np.ones(5))


def test_intensity_falls_to_exp_minus_gain_at_the_reference() -> None:
    got = flow.schlieren_intensity(np.array([1.0]), gain=20.0, reference=1.0)
    assert got[0] == pytest.approx(np.exp(-20.0))


def test_intensity_is_monotone_and_bounded() -> None:
    magnitude = np.linspace(0.0, 10.0, 64)
    intensity = flow.schlieren_intensity(magnitude, reference=1.0)
    assert np.all(np.diff(intensity) <= 0.0)
    assert intensity.min() >= 0.0
    assert intensity.max() <= 1.0


def test_a_uniform_field_does_not_divide_by_zero() -> None:
    """A solution with no gradient anywhere has no reference to scale to.

    It happens: a freestream-initialised case that never advanced. The image
    is white rather than a domain of NaN.
    """
    assert flow.schlieren_intensity(np.zeros(8), reference=None) == pytest.approx(np.ones(8))


def test_the_reference_ignores_a_single_extreme_cell() -> None:
    """The reason the default is a percentile and not the maximum.

    One cell a thousand times the rest must not set the exposure, or every
    real feature lands in the first thousandth of the range and renders white.
    """
    magnitude = np.concatenate([np.ones(999), [1000.0]])
    intensity = flow.schlieren_intensity(magnitude, gain=20.0, percentile=99.5)
    assert intensity[0] < np.exp(-10.0)


def test_schlieren_magnitude_relative_divides_by_density(cube: Mesh) -> None:
    from aether.aerodynamics.cfd.fields import VolumeField

    slope = np.array([3.0, 0.0, 0.0])
    density = _CORNERS @ slope + 10.0
    field = VolumeField(
        points=_CORNERS,
        fields={
            "Density": density,
            "Momentum_x": np.zeros(8),
            "Momentum_y": np.zeros(8),
            "Momentum_z": np.zeros(8),
            "Energy": np.ones(8),
        },
    )
    absolute = flow.schlieren_magnitude(cube, field, relative=False)
    relative = flow.schlieren_magnitude(cube, field, relative=True)
    assert absolute == pytest.approx(np.full(8, 3.0))
    assert relative == pytest.approx(3.0 / density)


# ------------------------------------------------------------- resampling


def test_regrid_reproduces_a_linear_field_inside_the_cut(cube: Mesh) -> None:
    """Linear in, linear out -- the resampling adds no error of its own.

    Written through the plane origin rather than in world coordinates,
    because that is the convention the grid comes back in: in-plane
    coordinates are measured from the origin the cut was asked for, so a
    field expressed in world :math:`x` and :math:`z` has to be shifted back
    onto it. Getting that wrong is a constant offset, which is exactly the
    error that hides in a picture and shows up in a number.
    """
    origin = np.array([0.5, 0.5, 0.5])
    cut = plane_cut(cube, origin, (0.0, 1.0, 0.0))
    slope = np.array([2.0, 0.0, -1.0])
    grid_x, grid_y, sampled = flow.regrid(cut, _CORNERS, _CORNERS @ slope + 1.0, resolution=17)
    inside = np.isfinite(sampled)
    assert inside.any()
    # In-plane axes are x and z; undo the origin shift before comparing.
    world_x = grid_x + origin[0]
    world_z = grid_y + origin[2]
    expected = slope[0] * world_x + slope[2] * world_z + 1.0
    assert sampled[inside] == pytest.approx(expected[inside], abs=1e-9)


def test_regrid_marks_the_outside_rather_than_extrapolating(cube: Mesh) -> None:
    cut = plane_cut(cube, (0.5, 0.5, 0.5), (0.0, 1.0, 0.0))
    _, _, sampled = flow.regrid(
        cut, _CORNERS, np.zeros(8), resolution=9, bounds=(-2.0, 3.0, -2.0, 3.0)
    )
    assert np.isnan(sampled).any()


def test_regrid_keeps_vector_components_on_one_grid(cube: Mesh) -> None:
    cut = plane_cut(cube, (0.5, 0.5, 0.5), (0.0, 1.0, 0.0))
    _, _, sampled = flow.regrid(cut, _CORNERS, _CORNERS, resolution=11)
    assert sampled.shape == (11, 11, 3)


# ------------------------------------------------------------ wall curves


def test_a_wall_curve_comes_back_ordered_by_station() -> None:
    from aether.aerodynamics.cfd.fields import surface_cut

    mesh = Mesh(
        points=_CORNERS.copy(),
        tetrahedra=_KUHN.copy(),
        markers={"wall": np.array([[0, 1, 3], [0, 3, 2]], dtype=np.int64)},
    )
    body = surface_cut(mesh, ["wall"], (0.0, 0.5, 0.0), (0.0, 1.0, 0.0))
    station, values = flow.wall_curve(body, _CORNERS, _CORNERS[:, 0])
    assert np.all(np.diff(station) >= 0.0)
    assert station.size == values.size == body.size


# ---------------------------------------------------------------- refusals


def test_a_field_of_the_wrong_length_is_refused_not_broadcast(cube: Mesh, axes) -> None:
    """The failure this catches draws one case's field on another's mesh."""
    cut = plane_cut(cube, (0.5, 0.5, 0.5), (0.0, 1.0, 0.0))
    with pytest.raises(ValueError, match="values for"):
        flow.field_map(axes, cut, _CORNERS, np.zeros(3))


def test_a_velocity_that_is_not_three_dimensional_is_refused(cube: Mesh, axes) -> None:
    cut = plane_cut(cube, (0.5, 0.5, 0.5), (0.0, 1.0, 0.0))
    with pytest.raises(ValueError, match=r"must be \(N, 3\)"):
        flow.streamlines(axes, cut, _CORNERS, np.zeros((8, 2)))


def test_the_drawing_calls_run_on_a_real_cut(cube: Mesh, axes) -> None:
    """A smoke test: these go through Matplotlib's triangulation machinery,
    which rejects degenerate triangulations, so reaching the end is a check
    that the cut is one."""
    cut = plane_cut(cube, (0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
    assert flow.field_map(axes, cut, _CORNERS, _CORNERS[:, 0], levels=8) is not None
    assert flow.schlieren(axes, cut, _CORNERS, np.abs(_CORNERS[:, 2]), levels=8) is not None
    assert flow.streamlines(axes, cut, _CORNERS, np.ones((8, 3)), resolution=12) is not None


def test_a_cut_is_a_triangulation_matplotlib_can_search(cube: Mesh) -> None:
    """The check the duplicate-vertex bug would have failed.

    ``tricontourf`` tolerates coincident vertices, so the figures looked
    right; the trapezoid-map point locator does not, and it is what every
    resampling and every streamline goes through. Building it is the whole
    assertion -- it raises on an invalid triangulation.
    """
    cut = plane_cut(cube, (0.5, 0.5, 0.5), (0.0, 1.0, 0.0))
    flow.triangulation(cut, _CORNERS).get_trifinder()
