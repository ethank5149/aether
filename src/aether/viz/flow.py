"""Drawing a cut flow field: contours, Schlieren, streamlines, wall curves.

The geometry is already done by the time anything here runs --
:mod:`aether.aerodynamics.cfd.fields` produced a triangulated cross-section
and the body's profile in the same plane. What is left is the part with no
right answer, only better and worse choices, and this module is where those
choices are made once rather than in every notebook.

Two of them are worth stating, because both were arrived at by getting a
picture wrong first.

**The Schlieren scale is set by a percentile, not the maximum.** A numerical
Schlieren image is :math:`\\exp(-\\kappa |\\nabla\\rho| / \\rho_{\\mathrm{ref}})`,
and taking :math:`\\rho_{\\mathrm{ref}}` as the largest gradient in the domain
hands the scale to a single cell. On a captured bow shock that cell is
typically at the stagnation point and is the least resolved one in the mesh,
so the shock -- the thing the picture exists to show -- comes out uniformly
pale behind it. The default reference is the 99.5th percentile.

**Streamlines are drawn on a regular grid, and that grid is a resampling.**
Matplotlib's integrator needs one, so the cut is interpolated onto it. That
step is a second discretisation over the solver's, and it is the only one in
this module: the contours and the wall curves are drawn on the cut's own
triangles and show what the solver produced. Which is why
:func:`regrid` is public and separate -- a resampled field should be visibly
a resampled field at the call site.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from aether.aerodynamics.cfd.fields import (
    Mesh,
    PlaneCut,
    SurfaceCut,
    VolumeField,
    node_gradient,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import LineCollection
    from matplotlib.streamplot import StreamplotSet
    from matplotlib.tri import Triangulation, TriContourSet

_FloatArray = NDArray[np.float64]

__all__ = [
    "field_map",
    "outline",
    "regrid",
    "schlieren",
    "schlieren_intensity",
    "schlieren_magnitude",
    "streamlines",
    "triangulation",
    "wall_curve",
]

#: Sets how fast the Schlieren image darkens with gradient. The value used
#: through most of the shock-capturing literature; larger picks out weak
#: waves and saturates the strong ones.
DEFAULT_SCHLIEREN_GAIN = 20.0


def triangulation(cut: PlaneCut, points: _FloatArray) -> Triangulation:
    """The cut as a Matplotlib triangulation, in the plane's own coordinates."""
    from matplotlib.tri import Triangulation

    planar = cut.coordinates(points)
    return Triangulation(planar[:, 0], planar[:, 1], cut.triangles)


def field_map(
    ax: Axes,
    cut: PlaneCut,
    points: _FloatArray,
    values: _FloatArray,
    *,
    levels: int | NDArray[np.float64] = 128,
    cmap: str = "magma",
    **kwargs: Any,
) -> TriContourSet:
    """Filled contours of a nodal field over the cut.

    Drawn on the cut's own triangles, so what appears is the solution
    restricted to the plane and not a resampling of it. ``values`` is a nodal
    field over the *mesh*; it is cut here rather than by the caller so that
    the field and the triangulation cannot come from different cuts.

    The default level count is high because these are shock-dominated fields:
    a dozen levels turn a captured shock into a staircase of bands whose edges
    look like structure and are not.
    """
    if values.shape[0] != points.shape[0]:
        raise ValueError(f"field has {values.shape[0]} values for {points.shape[0]} nodes")
    return ax.tricontourf(
        triangulation(cut, points), cut.interpolate(values), levels, cmap=cmap, **kwargs
    )


def schlieren_intensity(
    magnitude: _FloatArray,
    *,
    gain: float = DEFAULT_SCHLIEREN_GAIN,
    reference: float | None = None,
    percentile: float = 99.5,
) -> _FloatArray:
    """Map density-gradient magnitude to a Schlieren intensity in ``[0, 1]``.

    :math:`S = \\exp(-\\kappa |\\nabla\\rho| / \\rho_{\\mathrm{ref}})`, so ``1``
    is undisturbed flow and ``0`` is the strongest gradient shown. Rendered
    with a reversed grey map this is the familiar photograph: white ahead of
    the shock, black through it.

    ``reference`` defaults to a high percentile rather than the maximum. The
    maximum is one cell, usually the worst-resolved one in the mesh, and
    letting it set the scale is what makes an otherwise correct solution
    render as a blank field with a single dark speck.
    """
    magnitude = np.asarray(magnitude, dtype=np.float64)
    if reference is None:
        finite = magnitude[np.isfinite(magnitude)]
        reference = float(np.percentile(finite, percentile)) if finite.size else 0.0
    if reference <= 0.0:
        return np.ones_like(magnitude)
    return np.asarray(np.exp(-gain * magnitude / reference), dtype=np.float64)


def schlieren_magnitude(mesh: Mesh, field: VolumeField, *, relative: bool = True) -> _FloatArray:
    """The quantity a Schlieren image is made of: how fast density changes.

    ``relative`` divides by the local density, and defaults to on. It is the
    difference between a picture and a blank one. In a Mach 8 shock layer the
    density spans two orders of magnitude, so the absolute gradient is largest
    where the density already is -- inside the shock layer at the nose -- and
    scaling to that leaves the oblique shock, the shoulder expansion and the
    wake all within the first few percent of the range, which renders as an
    empty field with a dark speck at the nose. Measured on the Mach 8
    sphere-cone: absolute gradient 40.4 at the 99.5th percentile against a
    maximum of 117, and every feature outside the nose below 1.

    Dividing by density asks instead by what *fraction* the density changes
    per unit length, which is the quantity a real Schlieren system responds to
    and which is comparable between the shock layer and the freestream.

    Separated from :func:`schlieren` so an animation computes it once. On the
    largest mesh here the gradient costs about half a second, which is nothing
    for one figure and minutes for a frame sequence.
    """
    magnitude = np.linalg.norm(node_gradient(mesh, field.density), axis=1)
    return np.asarray(magnitude / field.density if relative else magnitude, dtype=np.float64)


def schlieren(
    ax: Axes,
    cut: PlaneCut,
    points: _FloatArray,
    magnitude: _FloatArray,
    *,
    gain: float = DEFAULT_SCHLIEREN_GAIN,
    reference: float | None = None,
    percentile: float = 99.5,
    levels: int = 128,
    cmap: str = "gray",
    **kwargs: Any,
) -> TriContourSet:
    """Numerical Schlieren over the cut.

    ``magnitude`` is a nodal scalar, normally from
    :func:`schlieren_magnitude`. It is taken already reduced rather than as a
    gradient vector so that the choice between the absolute and the relative
    measure is made once, visibly, by the caller -- the two produce very
    different pictures of the same solution and neither is a default worth
    hiding.

    The intensity is computed on the *volume* field and then cut, not computed
    from the cut. That way two cuts through one solution share an exposure,
    which is what makes a sequence of them watchable rather than flickering.
    """
    intensity = schlieren_intensity(
        magnitude, gain=gain, reference=reference, percentile=percentile
    )
    return ax.tricontourf(
        triangulation(cut, points),
        cut.interpolate(intensity),
        levels,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        **kwargs,
    )


def outline(
    ax: Axes,
    cut: SurfaceCut,
    points: _FloatArray,
    *,
    color: str = "white",
    linewidth: float = 1.2,
    **kwargs: Any,
) -> LineCollection:
    """Draw a body profile from a surface cut.

    The segments are not ordered into a path, and are not sorted into one
    here: a marker cut by a plane can be several disjoint curves -- a body and
    its fins, a stack and its separation rings -- and joining them into a
    single polyline draws lines across the gaps that are not surfaces.
    """
    from matplotlib.collections import LineCollection

    planar = cut.coordinates(points)
    collection = LineCollection(
        list(planar[cut.segments]), colors=color, linewidths=linewidth, **kwargs
    )
    ax.add_collection(collection)
    return collection


def wall_curve(
    cut: SurfaceCut, points: _FloatArray, values: _FloatArray
) -> tuple[_FloatArray, _FloatArray]:
    """A surface field along the profile, ordered by station for plotting.

    Returns ``(station, value)`` sorted on the first in-plane axis -- the body
    axis, under the meridian convention. Segment endpoints are shared between
    neighbouring segments and appear twice; both are kept, because dropping
    duplicates would silently merge the windward and leeward branches of a
    profile into one curve that is neither.
    """
    station = cut.coordinates(points)[:, 0]
    sampled = cut.interpolate(values)
    order = np.argsort(station)
    return station[order], sampled[order]


def regrid(
    cut: PlaneCut,
    points: _FloatArray,
    values: _FloatArray,
    *,
    resolution: int = 400,
    bounds: tuple[float, float, float, float] | None = None,
) -> tuple[_FloatArray, _FloatArray, _FloatArray]:
    """Resample a cut field onto a regular grid, ``NaN`` outside the cut.

    A resampling, and named as one. It exists because Matplotlib's streamline
    integrator needs a regular grid; nothing else in this module does, and
    nothing that goes into a quantitative figure should come through here.

    ``values`` may be ``(N,)`` or ``(N, k)``; the trailing axis is preserved,
    so a velocity field regrids in one call and its components stay on the
    same grid.
    """
    from matplotlib.tri import LinearTriInterpolator

    mesh = triangulation(cut, points)
    planar = cut.coordinates(points)
    if bounds is None:
        bounds = (
            float(planar[:, 0].min()),
            float(planar[:, 0].max()),
            float(planar[:, 1].min()),
            float(planar[:, 1].max()),
        )
    left, right, bottom, top = bounds
    xs = np.linspace(left, right, resolution)
    ys = np.linspace(bottom, top, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)

    cut_values = cut.interpolate(values)
    columns = cut_values[:, None] if cut_values.ndim == 1 else cut_values
    sampled = np.empty((resolution, resolution, columns.shape[1]), dtype=np.float64)
    for index in range(columns.shape[1]):
        interpolator = LinearTriInterpolator(mesh, columns[:, index])
        sampled[..., index] = interpolator(grid_x, grid_y).filled(np.nan)
    return grid_x, grid_y, (sampled[..., 0] if cut_values.ndim == 1 else sampled)


def streamlines(
    ax: Axes,
    cut: PlaneCut,
    points: _FloatArray,
    velocity: _FloatArray,
    *,
    resolution: int = 400,
    density: float = 1.4,
    color: str = "white",
    linewidth: float = 0.6,
    **kwargs: Any,
) -> StreamplotSet:
    """Streamlines of the in-plane velocity.

    ``velocity`` is the ``(N, 3)`` nodal field; it is projected onto the
    cutting plane here. Out-of-plane velocity is discarded, which for a
    meridian cut at incidence means the picture shows the projected path of a
    particle rather than its path -- true of every streamline plot on a slice,
    and worth remembering before reading crossflow off one.

    Regions outside the cut are filled with zero rather than left ``NaN``, so
    the integrator stops there instead of failing.
    """
    if velocity.ndim != 2 or velocity.shape[1] != 3:
        raise ValueError(f"velocity must be (N, 3); got {velocity.shape}")
    in_plane = velocity @ cut.basis.T
    grid_x, grid_y, sampled = regrid(cut, points, in_plane, resolution=resolution)
    return ax.streamplot(
        grid_x[0],
        grid_y[:, 0],
        np.nan_to_num(sampled[..., 0]),
        np.nan_to_num(sampled[..., 1]),
        density=density,
        color=color,
        linewidth=linewidth,
        **kwargs,
    )
