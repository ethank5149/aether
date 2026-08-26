"""Ray-traced globe rendering, for trajectory visualisation.

Why this is not a Matplotlib surface plot
-----------------------------------------

The obvious way to draw an Earth in Matplotlib is ``plot_surface`` with a
texture passed as ``facecolors``. It works, and it is why the first version
of the campaign animation looked the way it did: **each mesh quad is filled
with one flat colour**, so the visible resolution is the *mesh* resolution,
not the texture's. A 4096x2048 Blue Marble image downsampled onto a 256x128
mesh throws away 99.6 % of its pixels and leaves visible facets on every
coastline.

Raising the mesh resolution does not fix it. The cost grows as the square,
Matplotlib sorts every quad by depth with a painter's algorithm, and once
trajectory lines are added the sort produces the other familiar artefact:
arcs that pass *through* the planet appearing in front of it, because
Matplotlib compares whole artists rather than fragments.

So this module does the direct thing instead. For every output pixel it
intersects a camera ray with the body, converts the hit point to a geodetic
latitude and longitude, and samples the imagery bilinearly. The result is
limited by the texture and the output size, never by a mesh; occlusion is a
per-point depth test rather than an artist sort; and the whole thing is
vectorised NumPy, so a 1600x900 frame takes tens of milliseconds.

The body is an ellipsoid
------------------------

It used to be a sphere, and that was self-consistent and wrong. The
:mod:`~aether.viz.ellipsoid` note has the measurement: treating geodetic
latitude as geocentric displaces a surface point at 45 degrees by 21.4 km.
The intersection is solved in the scaled space where the ellipsoid *is* the
unit sphere, so it costs the same as the sphere version did, and the
surface normal is the true geodetic vertical rather than the position
direction.

A sphere is still available — pass a float radius, or an
:class:`~aether.viz.ellipsoid.Ellipsoid` with zero flattening — because a
history built on a spherical model must be drawn on the sphere it was built
on. Mixing an ellipsoidal picture with spherical physics moves the ground
out from under the trajectory, which is the failure this whole layer exists
to avoid.

Textures carry their own footprint
----------------------------------

Every image handed in is a :class:`~aether.viz.imagery.Texture`: pixels plus
the degree box they cover. That is what lets a full-globe mosaic and a
native-resolution launch-pad crop be composited in one pass — the renderer
samples the finest texture that covers each pixel and falls back outward,
so a close-up gets 15-arc-second Blue Marble where it has it and the global
mosaic everywhere else, with a feathered join rather than a visible tile
edge.

What is modelled, and what is decoration
----------------------------------------

Two things here are **data**:

* **Relief shading** from GMTED2010. The surface normal is perturbed by the
  measured terrain slope, so the Himalaya and the Andes are lit as they
  are lit, not stippled on.
* **Terrain displacement**, optionally: the ray is marched onto the real
  elevation instead of onto the reference ellipsoid. Off by default,
  because at orbital range Everest is under a pixel; on for close-ups,
  where it is the whole picture.

The rest is presentation and is calibrated against nothing:

* Lambertian diffuse with a soft terminator, because a hard day/night edge
  on a smooth body reads as a rendering bug.
* Night side as a dimmed, blue-shifted day texture. Real night lights would
  need a second image.
* An atmospheric limb, the cheapest cue that the object is a planet with
  air rather than a billiard ball.
* A specular highlight weighted toward dark texels so land does not gleam.

Where it runs
-------------

Every per-pixel expression below is written against the array API that
NumPy and CuPy share, so :func:`render` takes the same ``backend`` argument
as the batched integrator in :mod:`aether.batch.backend` and runs on either
device. Textures must already live on the requested backend — see
:func:`to_device` — because re-uploading them once per frame would cost more
than the render it was meant to accelerate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aether.batch.backend import Backend, get_array_module, to_numpy
from aether.ellipsoid import WGS84, Ellipsoid, ray_ellipsoid
from aether.viz.imagery import Texture
from aether.viz.terrain import ReliefMap

__all__ = [
    "Camera",
    "as_ellipsoid",
    "look_at",
    "project",
    "render",
    "sun_direction",
    "to_device",
]

_FloatArray = NDArray[np.float64]

def as_ellipsoid(surface: Ellipsoid | float) -> Ellipsoid:
    """Coerce a radius or an ellipsoid to an :class:`Ellipsoid`.

    A bare float means a sphere, and it means it exactly: zero flattening,
    so every ellipsoidal expression downstream degenerates to the spherical
    one rather than being special-cased.
    """
    if isinstance(surface, Ellipsoid):
        return surface
    radius = float(surface)
    if not (np.isfinite(radius) and radius > 0.0):
        msg = f"radius must be finite and positive, got {surface}"
        raise ValueError(msg)
    return Ellipsoid(semi_major=radius, flattening=0.0, name="sphere")


def to_device(
    texture: Texture | Sequence[Texture] | ReliefMap | Any, backend: Backend = "numpy"
) -> Any:
    """Place a texture, a stack of them, or a relief map on a backend, once.

    Kept explicit rather than done inside :func:`render` because the upload
    is the expensive part: an 8192x4096 BMNG mosaic is 100 MB as ``uint8``
    and the relief grids another 33 MB apiece, and moving them per frame
    would swamp the render they were meant to speed up. Upload at set-up,
    keep the handle, pass it to every frame.

    ``uint8`` stays ``uint8`` — the renderer normalises when it samples, and
    a float64 copy of the same mosaic is 800 MB.
    """
    xp = get_array_module(backend)
    if isinstance(texture, Texture):
        return texture.with_data(xp.asarray(texture.data))
    if isinstance(texture, ReliefMap):
        # The footprint travels with the grid. Dropping it here silently
        # reinterpreted a 3-degree launch-site patch as a global grid, which
        # stretches 149 m cells across the whole planet — the arrays are the
        # cheap part of a relief map and the bounds are the part that says
        # where it is.
        return ReliefMap(
            elevation=xp.asarray(texture.elevation),
            slope_east=xp.asarray(texture.slope_east),
            slope_north=xp.asarray(texture.slope_north),
            exaggeration=texture.exaggeration,
            south=texture.south,
            north=texture.north,
            west=texture.west,
            east=texture.east,
        )
    if isinstance(texture, (list, tuple)):
        return [to_device(item, backend) for item in texture]
    return xp.asarray(texture)


@dataclass(frozen=True)
class Camera:
    """A pinhole camera looking at a point.

    Attributes
    ----------
    position:
        Eye position in the same frame as the geometry (m).
    target:
        Point the camera looks at (m).
    up:
        World up hint; the true up is re-orthogonalised against the view
        direction, so this only has to be non-parallel to it.
    fov:
        Vertical field of view (rad).
    width, height:
        Output size in pixels.
    """

    position: _FloatArray
    target: _FloatArray
    up: _FloatArray
    fov: float = np.deg2rad(35.0)
    width: int = 1280
    height: int = 720

    def basis(self) -> tuple[_FloatArray, _FloatArray, _FloatArray]:
        """Right, true-up and forward unit vectors.

        Degenerate cases are handled rather than left to produce NaNs: if
        the supplied up is parallel to the view direction the camera would
        have no defined roll, so a fallback axis is substituted.
        """
        forward = np.asarray(self.target, dtype=np.float64) - np.asarray(
            self.position, dtype=np.float64
        )
        norm = float(np.linalg.norm(forward))
        if norm == 0.0:
            msg = "camera position and target coincide, so no view direction is defined"
            raise ValueError(msg)
        forward = forward / norm
        hint = np.asarray(self.up, dtype=np.float64)
        right = np.cross(forward, hint)
        if float(np.linalg.norm(right)) < 1e-12:
            hint = np.array([0.0, 0.0, 1.0])
            right = np.cross(forward, hint)
            if float(np.linalg.norm(right)) < 1e-12:
                right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        right = right / float(np.linalg.norm(right))
        true_up = np.cross(right, forward)
        return right, true_up / float(np.linalg.norm(true_up)), forward

    def ground_resolution(self, distance: float) -> float:
        """Metres per pixel on a surface ``distance`` metres away, face on.

        What a caller compares against
        :attr:`~aether.viz.imagery.Texture.ground_resolution` to decide
        whether a native-resolution crop is worth reading, or whether the
        global mosaic already has more detail than the frame can show.
        """
        return float(2.0 * distance * np.tan(0.5 * self.fov) / max(self.height, 1))


def look_at(
    target: _FloatArray,
    distance: float,
    azimuth: float,
    elevation: float,
    up: _FloatArray | None = None,
    **kwargs: object,
) -> Camera:
    """Build a camera orbiting ``target`` at a given range and direction.

    ``azimuth`` is measured about the world z axis from the world x axis,
    and ``elevation`` above the world xy plane — the usual orbit controls.
    """
    centre = np.asarray(target, dtype=np.float64)
    offset = distance * np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    return Camera(
        position=centre + offset,
        target=centre,
        up=np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, dtype=np.float64),
        **kwargs,  # type: ignore[arg-type]
    )


def sun_direction(hour_angle: float, declination: float = 0.0) -> _FloatArray:
    """Unit vector toward the sun in the body-fixed frame.

    Parameters
    ----------
    hour_angle:
        Longitude of the subsolar point (rad).
    declination:
        Latitude of the subsolar point (rad); 0 is equinox.
    """
    return np.array(
        [
            np.cos(declination) * np.cos(hour_angle),
            np.cos(declination) * np.sin(hour_angle),
            np.sin(declination),
        ]
    )


# -- sampling ------------------------------------------------------------


def _bilinear(
    grid: Any,
    row: Any,
    column: Any,
    wraps: bool,
    xp: ModuleType,
) -> Any:
    """Bilinear lookup at fractional ``(row, column)`` in pixel-centre units.

    ``wraps`` applies to the column axis only. Rows are always clamped: a
    grid stops at the poles, and wrapping there would sample the opposite
    hemisphere.
    """
    rows, cols = int(grid.shape[0]), int(grid.shape[1])
    r = xp.clip(row, 0.0, rows - 1.0)
    r0 = xp.clip(xp.floor(r).astype(xp.int64), 0, rows - 1)
    r1 = xp.minimum(r0 + 1, rows - 1)
    fr = r - r0

    if wraps:
        c0 = xp.mod(xp.floor(column).astype(xp.int64), cols)
        c1 = xp.mod(c0 + 1, cols)
        fc = column - xp.floor(column)
    else:
        c = xp.clip(column, 0.0, cols - 1.0)
        c0 = xp.clip(xp.floor(c).astype(xp.int64), 0, cols - 1)
        c1 = xp.minimum(c0 + 1, cols - 1)
        fc = c - c0

    # Flat gathers rather than two-dimensional fancy indexing. Same result,
    # measured at 5.9 ms against 11.2 ms for a 720x1280 lookup into a
    # 2048x4096 grid — and a displaced frame does fourteen of these, so it
    # is the single largest cost in the renderer.
    flat = grid.reshape(-1) if grid.ndim == 2 else grid.reshape(-1, grid.shape[2])
    base_0, base_1 = r0 * cols, r1 * cols
    if grid.ndim == 3:
        fr, fc = fr[..., None], fc[..., None]
        top = flat[base_0 + c0] * (1.0 - fc) + flat[base_0 + c1] * fc
        bottom = flat[base_1 + c0] * (1.0 - fc) + flat[base_1 + c1] * fc
    else:
        top = xp.take(flat, base_0 + c0) * (1.0 - fc) + xp.take(flat, base_0 + c1) * fc
        bottom = (
            xp.take(flat, base_1 + c0) * (1.0 - fc)
            + xp.take(flat, base_1 + c1) * fc
        )
    return top * (1.0 - fr) + bottom * fr


def _pixel_coordinates(
    texture: Texture, latitude_deg: Any, longitude_deg: Any, xp: ModuleType
) -> tuple[Any, Any]:
    """Fractional row and column of a degree point, in pixel-centre units.

    The bounds are pixel **edges**, so the centre of row 0 is half a pixel
    inside ``north``. Getting this off by half a pixel was worth 2.4 km on
    the old 4096-row texture.
    """
    rows, cols = texture.shape
    d_lat = (texture.north - texture.south) / rows
    d_lon = (texture.east - texture.west) / cols
    row = (texture.north - latitude_deg) / d_lat - 0.5
    if texture.wraps:
        column = xp.mod(longitude_deg - texture.west, 360.0) / d_lon - 0.5
    else:
        column = (longitude_deg - texture.west) / d_lon - 0.5
    return row, column


def _sample_texture(
    texture: Texture, latitude_deg: Any, longitude_deg: Any, xp: ModuleType
) -> Any:
    """Bilinear RGB lookup, normalised to ``[0, 1]``."""
    row, column = _pixel_coordinates(texture, latitude_deg, longitude_deg, xp)
    colour = _bilinear(texture.data, row, column, texture.wraps, xp)
    # dtype is a NumPy dtype even on a device array, so this test is the
    # same on both backends.
    if not np.issubdtype(texture.data.dtype, np.floating):
        return colour / 255.0
    return colour


def _inside(texture: Texture, latitude_deg: Any, longitude_deg: Any, xp: ModuleType) -> Any:
    """Mask of points strictly inside a texture's box, with a half-pixel margin."""
    if texture.wraps:
        return (latitude_deg >= texture.south) & (latitude_deg <= texture.north)
    lon = xp.mod(longitude_deg - texture.west, 360.0) + texture.west
    return (
        (latitude_deg >= texture.south)
        & (latitude_deg <= texture.north)
        & (lon >= texture.west)
        & (lon <= texture.east)
    )


def _edge_fade(
    texture: Texture, latitude_deg: Any, longitude_deg: Any, feather: float, xp: ModuleType
) -> Any:
    """Ramp from 0 at a texture's border to 1 ``feather`` degrees inside.

    Without it the join between a 15-arc-second crop and a global mosaic is
    a hard rectangle, which reads as a rendering artefact rather than as
    detail. The ramp does not invent pixels; it cross-fades between two
    measurements of the same ground.

    The width is capped at a quarter of the crop's shorter side. Crops are
    sized to the camera footprint now and run to ten degrees or more, so a
    ramp wide enough to hide the join on those would consume a one-degree
    crop entirely; the cap lets one number serve both.
    """
    if texture.wraps or feather <= 0.0:
        return _inside(texture, latitude_deg, longitude_deg, xp).astype(xp.float64)
    width = min(
        float(feather),
        0.25 * min(texture.north - texture.south, texture.east - texture.west),
    )
    if width <= 0.0:  # pragma: no cover - a degenerate box
        return _inside(texture, latitude_deg, longitude_deg, xp).astype(xp.float64)
    lon = xp.mod(longitude_deg - texture.west, 360.0) + texture.west
    distance = xp.minimum(
        xp.minimum(latitude_deg - texture.south, texture.north - latitude_deg),
        xp.minimum(lon - texture.west, texture.east - lon),
    )
    return xp.clip(distance / width, 0.0, 1.0)


def _rays(camera: Camera, xp: ModuleType) -> tuple[_FloatArray, Any]:
    """Origin and per-pixel unit directions for the camera.

    The origin stays on the host — it is three numbers, and keeping it there
    lets the ray coefficients be plain Python floats on either backend.
    """
    right, up, forward = camera.basis()
    aspect = camera.width / camera.height
    half_h = np.tan(0.5 * camera.fov)
    half_w = half_h * aspect
    # Pixel centres, y increasing downward so row 0 is the top of the image.
    xs = xp.linspace(-half_w, half_w, camera.width)
    ys = xp.linspace(half_h, -half_h, camera.height)
    grid_x, grid_y = xp.meshgrid(xs, ys)
    directions = (
        xp.asarray(forward)[None, None, :]
        + grid_x[..., None] * xp.asarray(right)[None, None, :]
        + grid_y[..., None] * xp.asarray(up)[None, None, :]
    )
    directions /= xp.linalg.norm(directions, axis=-1, keepdims=True)
    return np.asarray(camera.position, dtype=np.float64), directions


def _geodetic_normal(points: Any, ellipsoid: Ellipsoid, xp: ModuleType) -> Any:
    """Unit gradient of the ellipsoid's implicit form — the geodetic vertical.

    :math:`\\nabla(x^2/a^2 + y^2/a^2 + z^2/b^2)` is parallel to the surface
    normal, and on an ellipsoid that is *not* parallel to the position
    vector: the two differ by up to 0.19 degrees at mid-latitudes.
    """
    gradient = points * xp.asarray(1.0 / ellipsoid.axes**2)
    return gradient / xp.maximum(
        xp.linalg.norm(gradient, axis=-1, keepdims=True), 1.0e-300
    )


def _altitude(points: Any, ellipsoid: Ellipsoid, xp: ModuleType) -> Any:
    """Signed height above the ellipsoid (m), to first order in the gradient.

    :math:`F/|\\nabla F|` for the implicit form, which is the standard
    first-order distance to a smooth level set. The error is
    :math:`O(h^2 \\kappa)`; at 10 km altitude on a 6,378 km body that is
    metres, well inside a pixel at any range where terrain is visible, and
    it is checked numerically in the tests rather than asserted here.

    Used only by the displacement march. Anything that needs a *reported*
    altitude uses :func:`~aether.viz.ellipsoid.ecef_to_geodetic`, which is
    exact.
    """
    scaled = points * xp.asarray(1.0 / ellipsoid.axes**2)
    residual = xp.sum(points * scaled, axis=-1) - 1.0
    return 0.5 * residual / xp.maximum(xp.linalg.norm(scaled, axis=-1), 1.0e-300)


def _relief_coordinates(
    relief: ReliefMap, latitude_deg: Any, longitude_deg: Any, xp: ModuleType
) -> tuple[Any, Any]:
    """Fractional row and column in a relief grid, in cell-centre units."""
    rows, cols = relief.shape
    d_lat = (relief.north - relief.south) / rows
    d_lon = (relief.east - relief.west) / cols
    row = (relief.north - latitude_deg) / d_lat - 0.5
    if relief.wraps:
        column = xp.mod(longitude_deg - relief.west, 360.0) / d_lon - 0.5
    else:
        column = (longitude_deg - relief.west) / d_lon - 0.5
    return row, column


def _relief_stack(relief: ReliefMap | Sequence[ReliefMap] | None) -> list[ReliefMap]:
    """Normalise the relief argument to a coarse-to-fine list."""
    if relief is None:
        return []
    if isinstance(relief, ReliefMap):
        return [relief]
    return list(relief)


def _sample_elevation(
    stack: Sequence[ReliefMap], latitude_deg: Any, longitude_deg: Any, xp: ModuleType
) -> Any:
    """Elevation alone, taking the finest grid that covers each pixel.

    Separate from :func:`_sample_relief` because the displacement march
    needs *only* the height and runs several times per frame. Fetching the
    two slope grids it never looks at tripled the march's gather count and
    took a displaced 1280x720 frame from 1.3 to 3.6 seconds.
    """
    elevation = None
    for grid in stack:
        row, column = _relief_coordinates(grid, latitude_deg, longitude_deg, xp)
        height = _bilinear(grid.elevation, row, column, grid.wraps, xp)
        if elevation is None:
            elevation = height
            continue
        elevation = xp.where(
            _covers(grid, latitude_deg, longitude_deg, xp), height, elevation
        )
    return elevation


def _covers(
    grid: ReliefMap, latitude_deg: Any, longitude_deg: Any, xp: ModuleType
) -> Any:
    """Mask of points inside a relief grid's footprint."""
    if grid.wraps:
        return xp.ones_like(latitude_deg, dtype=bool)
    lon = xp.mod(longitude_deg - grid.west, 360.0) + grid.west
    return (
        (latitude_deg >= grid.south) & (latitude_deg <= grid.north)
        & (lon >= grid.west) & (lon <= grid.east)
    )


def _sample_relief(
    stack: Sequence[ReliefMap], latitude_deg: Any, longitude_deg: Any, xp: ModuleType
) -> tuple[Any, Any, Any]:
    """Elevation and slopes, taking the finest grid that covers each pixel.

    The same rule the imagery follows, and for the same reason: the global
    grid is 9.8 km cells, a local patch is 230 m, and a launch close-up
    needs the second wherever it has it and the first everywhere else.
    """
    elevation = slope_e = slope_n = None
    for grid in stack:
        row, column = _relief_coordinates(grid, latitude_deg, longitude_deg, xp)
        height = _bilinear(grid.elevation, row, column, grid.wraps, xp)
        east = _bilinear(grid.slope_east, row, column, grid.wraps, xp)
        north = _bilinear(grid.slope_north, row, column, grid.wraps, xp)
        if elevation is None:
            elevation, slope_e, slope_n = height, east, north
            continue
        inside = _covers(grid, latitude_deg, longitude_deg, xp)
        elevation = xp.where(inside, height, elevation)
        slope_e = xp.where(inside, east, slope_e)
        slope_n = xp.where(inside, north, slope_n)
    return elevation, slope_e, slope_n


def _march_terrain(
    origin: _FloatArray,
    directions: Any,
    ellipsoid: Ellipsoid,
    relief_stack: Sequence[ReliefMap],
    xp: ModuleType,
    steps: int = 12,
    refinements: int = 4,
) -> tuple[Any, Any]:
    """Distance to the **first** terrain crossing along each ray.

    A bounded linear search followed by bisection, over the segment a ray
    spends inside the terrain envelope — the ellipsoid inflated to the
    highest ground in the loaded grids.

    Why not the cheaper fixed point
    -------------------------------

    The obvious method advances each ray by ``(altitude - ground) / descent``
    where ``descent`` is the rate altitude falls along the ray. It is exact
    in one step for a flat surface and converges quickly looking straight
    down. It also **diverges at oblique incidence**, where ``descent`` goes
    to zero: measured over the Khumbu at 15 degrees above the horizon it
    moved the surface by up to **250 km** and the residual test still
    accepted it, because the ray had simply been thrown somewhere else that
    happened to be near some ground. Guarding the small divisor did not fix
    it either — it turned the divergence into doing nothing, and a chase
    camera near the ground displaced **0 %** of its pixels.

    A search cannot overshoot. It samples the segment, takes the first place
    the ray passes from above the terrain to below it, and bisects there —
    so it also **silhouettes**, which the fixed point never could: a ridge
    now hides the valley behind it, because the first crossing is the one
    that is returned.

    What it still cannot do is resolve terrain thinner than a search step.
    A ridge narrower than the segment divided by ``steps`` can be stepped
    over, which shows as an edge that is in the right place but softer than
    the ground actually is. That is aliasing with a known cause, not the
    unbounded error the fixed point produced.

    Returns
    -------
    tuple
        ``(distance, found)``. ``distance`` is meaningless where ``found``
        is false, and callers fall back to the reference surface there.
    """
    ceiling = max(float(grid.elevation.max()) for grid in relief_stack)
    if not np.isfinite(ceiling) or ceiling <= 0.0:  # pragma: no cover - flat grids
        return xp.zeros(directions.shape[:-1]), xp.zeros(directions.shape[:-1], dtype=bool)

    enter, leave, inside = ray_ellipsoid(
        origin, directions, ellipsoid, xp, inflation=ceiling, both_roots=True
    )
    # A ray that reaches the reference surface stops there; one that only
    # grazes the envelope runs to its far side.
    surface_t, reaches_surface = ray_ellipsoid(origin, directions, ellipsoid, xp)
    leave = xp.where(reaches_surface, xp.minimum(leave, surface_t), leave)
    inside = inside & (leave > enter)
    enter = xp.where(inside, enter, 0.0)
    leave = xp.where(inside, leave, 0.0)

    shape = inside.shape
    distance = xp.zeros(shape)
    found = xp.zeros(shape, dtype=bool)
    if not bool(inside.any()):
        return distance, found

    # Only the rays that enter the envelope are marched. Everything else is
    # sky or already resolved, and at a full-disc view that is most of the
    # frame — the compression is the difference between marching 921,600
    # rays and marching the ones that could possibly hit something.
    picked = xp.nonzero(inside.reshape(-1))[0]
    rays = directions.reshape(-1, 3)[picked]
    eye = xp.asarray(origin)[None, :]
    low = enter.reshape(-1)[picked]
    high = leave.reshape(-1)[picked]
    n_steps = max(int(steps), 2)

    def height_above_ground(t: Any) -> Any:
        points = eye + t[:, None] * rays
        latitude_deg, longitude_deg = _degrees(
            _geodetic_normal(points, ellipsoid, xp), xp
        )
        ground = _sample_elevation(relief_stack, latitude_deg, longitude_deg, xp)
        return _altitude(points, ellipsoid, xp) - ground

    span = (high - low) / n_steps
    previous_t = low
    previous_h = height_above_ground(low)
    lower = xp.zeros_like(low)
    upper = xp.zeros_like(low)
    crossed = xp.zeros_like(low, dtype=bool)
    for index in range(1, n_steps + 1):
        current_t = low + span * index
        current_h = height_above_ground(current_t)
        crossing = ~crossed & (previous_h > 0.0) & (current_h <= 0.0)
        lower = xp.where(crossing, previous_t, lower)
        upper = xp.where(crossing, current_t, upper)
        crossed = crossed | crossing
        previous_t, previous_h = current_t, current_h

    for _ in range(max(int(refinements), 0)):
        middle = 0.5 * (lower + upper)
        above = height_above_ground(middle) > 0.0
        lower = xp.where(above, middle, lower)
        upper = xp.where(above, upper, middle)

    flat_distance = distance.reshape(-1)
    flat_found = found.reshape(-1)
    flat_distance[picked] = 0.5 * (lower + upper)
    flat_found[picked] = crossed
    return flat_distance.reshape(shape), flat_found.reshape(shape)


def _degrees(normals: Any, xp: ModuleType) -> tuple[Any, Any]:
    """Geodetic latitude and longitude in degrees from a surface normal."""
    latitude = xp.rad2deg(xp.arcsin(xp.clip(normals[..., 2], -1.0, 1.0)))
    longitude = xp.rad2deg(xp.arctan2(normals[..., 1], normals[..., 0]))
    return latitude, longitude


def render(
    camera: Camera,
    texture: Texture | Sequence[Texture],
    surface: Ellipsoid | float = WGS84,
    sun: _FloatArray | None = None,
    ambient: float = 0.12,
    atmosphere: float = 0.55,
    night: float = 0.16,
    specular: float = 0.35,
    background: _FloatArray | None = None,
    relief: ReliefMap | Sequence[ReliefMap] | None = None,
    displace: bool = False,
    displace_steps: int = 12,
    displace_refinements: int = 4,
    feather: float = 0.6,
    haze_length: float = 600.0e3,
    backend: Backend = "numpy",
) -> tuple[_FloatArray, _FloatArray]:
    """Render the globe, returning an RGB image and a depth buffer.

    Parameters
    ----------
    camera:
        View.
    texture:
        One :class:`~aether.viz.imagery.Texture`, or several. With several,
        each pixel takes the **last** one that covers it, cross-faded over
        ``feather`` degrees at its border — so the natural order is coarse
        to fine: a global mosaic first, then a native-resolution crop of the
        launch site over the top.
    surface:
        The body. An :class:`~aether.viz.ellipsoid.Ellipsoid`, or a float
        radius meaning a sphere. **Must match whatever produced the states
        being drawn over it**; see the module note.
    sun:
        Unit vector toward the sun in the body frame. ``None`` lights the
        scene from the camera, which removes the terminator entirely and is
        occasionally what a diagram wants.
    ambient:
        Floor on the diffuse term, so the night side is not pure black.
    atmosphere:
        Strength of the limb glow. Zero disables it.
    night:
        Brightness of the unlit side relative to full daylight.
    specular:
        Strength of the ocean glint. Zero disables it, which is what a
        diagram wants when the returned colours are being read back rather
        than looked at — the highlight is additive and would corrupt them.
    background:
        RGB image of shape ``(height, width, 3)`` to composite the globe
        over. ``None`` gives a starfield-free dark background.
    relief:
        GMTED2010 slopes from :meth:`~aether.viz.terrain.Terrain.relief`.
        Perturbs the surface normal, so terrain is *lit* rather than
        painted. ``None`` leaves the body smooth.
    displace:
        Move the intersection onto the terrain instead of onto the
        reference ellipsoid. Needs ``relief``. Off by default because at
        orbital range the whole effect is under a pixel and it costs a few
        extra passes over the frame; on, it is what makes a launch-pad or
        impact close-up land on the right ground.
    displace_steps:
        Samples of the linear search that finds the terrain crossing. Sets
        the finest ridge the march can resolve: the segment inside the
        terrain envelope, divided by this.
    displace_refinements:
        Bisections after the search. Six takes a step of a few hundred
        metres to a few metres, which is under a pixel at any range where
        displacement is switched on.
    feather:
        Degrees over which a finer texture fades in at its border, capped
        at a quarter of the crop's shorter side. 0.05 degrees — 5.5 km — was
        a join rather than a blend, and against a crop ten degrees across
        the eye reads it as an edge; 0.6 degrees is 67 km of cross-fade.
    haze_length:
        Path length (m) over which the atmospheric limb fades in. Below it
        the glow is proportionately weaker, which is what stops a close-up
        from being washed out by an effect that belongs to the edge of a
        disc seen from space.
    backend:
        ``"numpy"`` or ``"cupy"``. On ``"cupy"`` every texture and the
        relief map must already be device-resident from :func:`to_device`;
        the returned image and depth buffer are brought back to the host,
        because their only consumers are Matplotlib and the projection test,
        both of which are host-side.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The image, shape ``(height, width, 3)`` in ``[0, 1]``, and the
        distance from the camera to the surface at each pixel, ``inf`` where
        the ray misses. The depth buffer is what makes correct occlusion of
        overlaid trajectories possible.

    Notes
    -----
    The displacement march takes the **first** terrain crossing along each
    ray, so a ridge does hide the ground behind it. What it cannot do is
    resolve terrain thinner than one search step; see :func:`_march_terrain`
    for what that costs and why the cheaper fixed-point method was removed.
    """
    ellipsoid = as_ellipsoid(surface)
    xp = get_array_module(backend)
    textures = [texture] if isinstance(texture, Texture) else list(texture)
    if not textures:
        msg = "render needs at least one texture"
        raise ValueError(msg)
    if backend != "numpy":
        for item in textures:
            if isinstance(item.data, np.ndarray):
                msg = (
                    f"backend {backend!r} needs device-resident textures; call "
                    "aether.viz.globe.to_device(texture, backend) once at set-up "
                    "rather than uploading the mosaic per frame"
                )
                raise TypeError(msg)
    relief_stack = _relief_stack(relief)
    if displace and not relief_stack:
        msg = "displace=True needs a relief map; pass relief=Terrain(...).relief()"
        raise ValueError(msg)

    origin, directions = _rays(camera, xp)
    start = xp.asarray(origin)[None, None, :]

    t_near, hit = ray_ellipsoid(origin, directions, ellipsoid, xp)

    if displace and relief_stack:
        t_hit, found = _march_terrain(
            origin, directions, ellipsoid, relief_stack, xp,
            steps=displace_steps, refinements=displace_refinements,
        )
        t_near = xp.where(found, t_hit, t_near)
        hit = hit | found

    depth = xp.where(hit, t_near, np.inf)
    points = start + xp.where(hit, t_near, 0.0)[..., None] * directions
    normals = _geodetic_normal(points, ellipsoid, xp)
    latitude_deg, longitude_deg = _degrees(normals, xp)

    albedo = _sample_texture(textures[0], latitude_deg, longitude_deg, xp)
    for finer in textures[1:]:
        blend = _edge_fade(finer, latitude_deg, longitude_deg, feather, xp)[..., None]
        albedo = albedo * (1.0 - blend) + _sample_texture(
            finer, latitude_deg, longitude_deg, xp
        ) * blend

    shading_normals = normals
    if relief_stack:
        # East/north/up at each hit, from the geodetic latitude and
        # longitude that the normal itself defines.
        phi, lam = xp.deg2rad(latitude_deg), xp.deg2rad(longitude_deg)
        sin_phi, cos_phi = xp.sin(phi), xp.cos(phi)
        sin_lam, cos_lam = xp.sin(lam), xp.cos(lam)
        east = xp.stack([-sin_lam, cos_lam, xp.zeros_like(sin_lam)], axis=-1)
        north = xp.stack(
            [-sin_phi * cos_lam, -sin_phi * sin_lam, cos_phi], axis=-1
        )
        _, slope_e, slope_n = _sample_relief(
            relief_stack, latitude_deg, longitude_deg, xp
        )
        tilted = normals - slope_e[..., None] * east - slope_n[..., None] * north
        shading_normals = tilted / xp.maximum(
            xp.linalg.norm(tilted, axis=-1, keepdims=True), 1.0e-300
        )

    light = (
        np.asarray(sun, dtype=np.float64)
        if sun is not None
        else -np.asarray(camera.basis()[2], dtype=np.float64)
    )
    light = light / float(np.linalg.norm(light))
    cosine = xp.einsum("ijk,k->ij", shading_normals, xp.asarray(light))
    # The terminator follows the *reference* surface, not the terrain: a
    # shaded slope that tipped past the terminator would light itself on the
    # night side, which is a rendering artefact rather than alpenglow.
    smooth_cosine = xp.einsum("ijk,k->ij", normals, xp.asarray(light))

    # Soft terminator: a hard step reads as an aliasing bug on a sphere.
    day = xp.clip(cosine, 0.0, 1.0) ** 0.75
    twilight = xp.clip((smooth_cosine + 0.12) / 0.24, 0.0, 1.0)
    diffuse = (ambient + (1.0 - ambient) * day)[..., None]

    lit = albedo * diffuse
    dark = albedo * night * xp.asarray([0.55, 0.65, 1.0])
    shaded = dark + (lit - dark) * twilight[..., None]

    if atmosphere > 0.0:
        # A *disc-edge* effect, and it has to be told so. The grazing term
        # alone is near one for any ray that meets the ground obliquely,
        # including one from a camera 1.6 km above a launch pad looking
        # along the surface — which added the full blue wash over the whole
        # lower half of every close-up and is exactly the haze that made
        # them unreadable.
        #
        # What actually sets the glow is how much air the ray crosses, so it
        # is faded in with path length: negligible over the 25 km between an
        # eye near the ground and the ground, saturated over the thousands
        # of kilometres a limb ray covers. ``haze_length`` is the scale of
        # that fade and is a presentation constant, not an optical depth.
        view = -directions
        grazing = 1.0 - xp.clip(xp.einsum("ijk,ijk->ij", normals, view), 0.0, 1.0)
        reach = xp.clip(xp.where(hit, t_near, haze_length) / haze_length, 0.0, 1.0)
        rim = grazing**3 * xp.clip(smooth_cosine + 0.25, 0.0, 1.0) * reach
        shaded = shaded + atmosphere * rim[..., None] * xp.asarray([0.30, 0.52, 0.95])

    if specular > 0.0:
        # Biased toward dark texels so continents do not gleam.
        halfway = xp.asarray(light) - directions
        halfway /= xp.linalg.norm(halfway, axis=-1, keepdims=True)
        spec = xp.clip(xp.einsum("ijk,ijk->ij", normals, halfway), 0.0, 1.0) ** 48.0
        ocean = xp.clip(1.0 - albedo.mean(axis=-1) * 2.2, 0.0, 1.0)
        shaded = shaded + (
            specular * spec * ocean * xp.clip(smooth_cosine, 0.0, 1.0)
        )[..., None]

    if background is None:
        image = xp.zeros((camera.height, camera.width, 3), dtype=xp.float64)
    else:
        if background.shape != (camera.height, camera.width, 3):
            msg = (
                f"background must have shape {(camera.height, camera.width, 3)}, "
                f"got {background.shape}"
            )
            raise ValueError(msg)
        image = xp.asarray(background, dtype=xp.float64)

    image = xp.where(hit[..., None], xp.clip(shaded, 0.0, 1.0), image)
    if backend == "numpy":
        return np.asarray(image), np.asarray(depth)
    return to_numpy(image), to_numpy(depth)


def project(
    points: _FloatArray,
    camera: Camera,
    surface: Ellipsoid | float | None = None,
) -> tuple[_FloatArray, _FloatArray, NDArray[np.bool_]]:
    """Project world points to pixel coordinates.

    Parameters
    ----------
    points:
        Shape ``(n, 3)`` in the same frame as the camera.
    camera:
        View.
    surface:
        Occluding body — an :class:`~aether.viz.ellipsoid.Ellipsoid` or a
        float radius. When given, a point is marked hidden if the body lies
        between it and the camera: the depth test that Matplotlib's
        painter's algorithm cannot do, and the reason trajectories used to
        appear in front of the planet they were behind.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Pixel ``x`` and ``y`` (float, may fall outside the frame), and a
        boolean mask that is ``True`` where the point is in front of the
        camera and not occluded.

    Notes
    -----
    The occlusion test uses the **reference surface**, not the terrain, even
    when the render displaced it. A trajectory clipped by a mountain rather
    than by the limb would be a stronger claim than this layer can support:
    it would need the ray-terrain occlusion that
    :func:`render`'s displacement march explicitly does not do.
    """
    array = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if array.ndim != 2 or array.shape[1] != 3:
        msg = f"points must have shape (n, 3), got {array.shape}"
        raise ValueError(msg)
    right, up, forward = camera.basis()
    origin = np.asarray(camera.position, dtype=np.float64)
    relative = array - origin

    depth = relative @ forward
    in_front = depth > 0.0
    safe = np.where(in_front, depth, 1.0)
    half_h = np.tan(0.5 * camera.fov)
    half_w = half_h * (camera.width / camera.height)
    ndc_x = (relative @ right) / (safe * half_w)
    ndc_y = (relative @ up) / (safe * half_h)
    px = (0.5 * (ndc_x + 1.0)) * (camera.width - 1)
    py = (0.5 * (1.0 - ndc_y)) * (camera.height - 1)

    visible = in_front
    if surface is not None:
        ellipsoid = as_ellipsoid(surface)
        distance = np.linalg.norm(relative, axis=1)
        direction = relative / np.maximum(distance[:, None], 1e-12)
        t_hit, blocked = ray_ellipsoid(origin, direction, ellipsoid)
        occluded = blocked & (t_hit > 1e-6) & (t_hit < distance - 1e-6)
        visible = visible & ~occluded
    return np.asarray(px), np.asarray(py), np.asarray(visible)
