"""Drawing primitives: a camera, a globe plate, and glyphs over it.

Everything here is geometry and Matplotlib. A camera rig that tracks a moving
point, a textured ellipsoid rendered behind it, polylines and meshes projected
onto the result. None of it knows what the point *is* — a sample of a
trajectory, a satellite, or a marker someone placed by hand all draw the same
way, and the module is written so that it cannot tell.

That is what separates this from the overlays built on top of it. Drawing a
polyline is generic; drawing a *detection sector* is a statement about a sensor
and an engagement, and those live with the layer that owns those concepts.

Two conventions worth stating, because both are declared rather than derived:

:data:`NOSE_AXIS`
    The body-frame direction the vehicle glyph points along. A torque-free
    point mass has no attitude to read, so the glyph needs a convention and
    this is it.

:func:`horizon_ring`
    Drawn at the *target's* radius, not on the ground. The visibility boundary
    is where the target is, and putting it on the surface understates the
    footprint by roughly a factor of three at 150 km.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from aether.batch.backend import Backend
from aether.ellipsoid import (
    WGS84,
    Ellipsoid,
    ecef_to_geodetic,
    geodetic_to_ecef,
    horizon_central_angle,
)
from aether.geodesy import GeodeticPosition
from aether.viz.globe import Camera, as_ellipsoid, project, render, to_device
from aether.viz.imagery import Texture
from aether.viz.terrain import ReliefMap

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "NOSE_AXIS",
    "ChaseRig",
    "SceneStyle",
    "dcm_from_flight_path",
    "draw_horizon_ring",
    "draw_marker",
    "draw_mesh",
    "draw_stack",
    "draw_track",
    "draw_vehicle",
    "ease",
    "geodetic_to_cartesian",
    "globe_plate",
    "glyph_polylines",
    "glyph_world",
    "horizon_ring",
    "stack_world",
    "starfield",
]

_FloatArray = NDArray[np.float64]
_IntArray = NDArray[np.int64]


#: Body-frame nose direction assumed by the vehicle glyph. The flight model
#: is torque-free with drag along the relative velocity, so no force term
#: pins a body axis; this is a presentation convention, declared rather than
#: inferred.
NOSE_AXIS = np.array([1.0, 0.0, 0.0])


@dataclass(frozen=True)
class SceneStyle:
    """Colours and weights, in one place so a notebook does not carry them."""

    track: str = "#FF8A3D"
    trail_width: float = 3.2
    track_width: float = 1.3
    track_alpha: float = 0.30
    vehicle: str = "#FFFFFF"
    jettisoned: str = "#8E9AAF"
    aimpoint: str = "#FFD166"
    launch: str = "#FFFFFF"
    site_idle: str = "#7CFFB2"
    site_active: str = "#FF3B30"
    site_seen: str = "#FFD166"
    text: str = "#FFFFFF"
    heat_cmap: str = "inferno"


def ease(t: float | _FloatArray) -> _FloatArray:
    """Smoothstep on ``[0, 1]``.

    Linear interpolation between camera states reads as mechanical: the
    acceleration is discontinuous at every keyframe and the eye visibly
    snaps. Smoothstep has zero first derivative at both ends, which is the
    cheapest fix that looks intentional.
    """
    x = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    return np.asarray(x * x * (3.0 - 2.0 * x))


def geodetic_to_cartesian(
    position: GeodeticPosition | Iterable[GeodeticPosition],
    surface: Ellipsoid | float = WGS84,
    lift: float = 0.0,
) -> _FloatArray:
    """Geodetic point(s) to Cartesian vectors on ``surface``.

    Parameters
    ----------
    position:
        One :class:`~aether.geodesy.GeodeticPosition` or an iterable of them.
    surface:
        The body the markers sit on — an
        :class:`~aether.ellipsoid.Ellipsoid`, or a float radius meaning
        a sphere.
    lift:
        Extra height (m). A marker drawn exactly on the surface is half
        buried in it and half occluded by the depth test, which reads as a
        rendering glitch rather than as a site; lifting it clear is
        cosmetic and stated.

    Returns
    -------
    numpy.ndarray
        Shape ``(3,)`` for a single position, ``(n, 3)`` for an iterable.

    Notes
    -----
    **This must agree with whatever produced the trajectories drawn beside
    it.** On the WGS84 default the conversion is the proper ellipsoidal one
    and a site lands where a survey puts it. Handed a float radius it
    degenerates exactly to the old spherical form — geodetic latitude used
    as geocentric — which is the right answer for a scene whose physics is
    spherical, because a marker 21 km off the trajectory that hit it is
    worse than a marker 21 km off the truth.
    """
    points: list[GeodeticPosition]
    if isinstance(position, GeodeticPosition):
        single, points = True, [position]
    else:
        single, points = False, list(position)
    ellipsoid = as_ellipsoid(surface)
    heights = np.array([max(float(p.altitude), 0.0) + float(lift) for p in points])
    latitudes = np.array([float(p.latitude) for p in points])
    longitudes = np.array([float(p.longitude) for p in points])
    cartesian = geodetic_to_ecef(latitudes, longitudes, heights, ellipsoid)
    return np.asarray(cartesian[0] if single else cartesian)


_STARFIELDS: dict[tuple[int, int, float, int], _FloatArray] = {}


def starfield(width: int, height: int, density: float = 0.00035, seed: int = 7) -> _FloatArray:
    """A faint fixed starfield, cached per size.

    Deterministic because a resampled field shimmers between frames, and
    cached because it is identical for every frame of a sequence — building
    it per frame was pure waste at 130 frames a run. The returned array is
    marked read-only so a caller cannot poison the cache;
    :func:`aether.viz.globe.render` copies its background, so this costs
    nothing.
    """
    key = (int(width), int(height), float(density), int(seed))
    cached = _STARFIELDS.get(key)
    if cached is not None:
        return cached
    rng = np.random.default_rng(seed)
    sky = np.zeros((height, width, 3), dtype=np.float64)
    count = int(width * height * density)
    rows = rng.integers(0, height, count)
    cols = rng.integers(0, width, count)
    magnitude = rng.power(0.35, count)
    tint = 0.75 + 0.25 * rng.random((count, 3))
    sky[rows, cols] = magnitude[:, None] * tint
    sky.flags.writeable = False
    _STARFIELDS[key] = sky
    return sky


@dataclass(frozen=True)
class ChaseRig:
    """A camera that rides behind and above the vehicle, along its velocity.

    The stand-off **scales with the vehicle's own altitude**, which is what
    lets one rig frame a 150 km fractional parking arc and a 1300 km
    minimum-energy apogee without being retuned. A fixed stand-off makes one
    of the two a line in the corner.

    Heading comes from the sampled **velocity**, not from a finite
    difference over an arbitrary look-ahead in the sample array. That
    matters: the old look-ahead of six samples meant the camera's notion of
    "forward" depended on the sampling density, so the same trajectory at
    400 and 900 samples framed differently.

    Attributes
    ----------
    back_scale, back_offset:
        Stand-off behind the vehicle is ``back_scale * altitude +
        back_offset`` (m). The offset is deliberately small — 25 km — so
        that a vehicle on the pad is framed from 25 km rather than from the
        450 km that an offset sized for orbit implies. Altitude does the
        rest: the same rig stands off 675 km at a 250 km parking orbit and
        3,400 km at a 1,300 km apogee.
    lift_scale, lift_offset:
        Height above the vehicle's local vertical, same form (m).
    min_altitude:
        Altitude floor used in those laws only, so a vehicle at the surface
        still gets a finite stand-off rather than sitting inside the eye.
    tighten:
        Fraction by which the rig closes in over the run. Zero holds the
        stand-off constant.
    fov:
        Vertical field of view (rad).
    floor:
        Minimum eye altitude (m). During boost the velocity is steeply
        radial, so stepping *back* along it also steps *down*: at lift-off
        the eye ended up 96 km underground and the ray tracer returned an
        empty frame — a black screen with a floating trajectory and no Earth
        at all. The eye is pushed back out along its own radius when that
        happens.
    lead:
        How far ahead of the vehicle the camera aims, as a fraction of the
        stand-off. Aiming exactly at the vehicle pins it dead centre, which
        wastes the half of the frame the vehicle is flying into.
    side:
        Cross-track stand-off, as a fraction of ``back``. Directly astern is
        the one viewpoint from which a slender body cannot be seen — the
        frame looks up its own tail and the vehicle reads as a ring with
        cross-hairs. A quarter view shows its length, which is what changes
        at a separation.
    """

    back_scale: float = 2.6
    back_offset: float = 25.0e3
    lift_scale: float = 0.9
    lift_offset: float = 10.0e3
    min_altitude: float = 0.0
    tighten: float = 0.25
    fov: float = float(np.deg2rad(42.0))
    floor: float = 3.0e3
    lead: float = 0.35
    side: float = 0.55

    def camera(
        self,
        position: _FloatArray,
        velocity: _FloatArray,
        width: int,
        height: int,
        progress: float = 0.0,
        surface: Ellipsoid | float = WGS84,
        altitude: float | None = None,
    ) -> Camera:
        """Place the eye for one state.

        Parameters
        ----------
        position, velocity:
            Inertial state (m, m/s), straight from
            a history sample.
        width, height:
            Frame size in pixels.
        progress:
            Fraction of the run completed, used only by ``tighten``.
        surface:
            The body, for the stand-off law and the eye-altitude floor.
            Altitudes are **geodetic**, so the same rig frames a polar and
            an equatorial launch identically instead of standing 21 km
            further back at the pole.
        altitude:
            Altitude (m) to feed the stand-off law, overriding the vehicle's
            actual one. The eye still goes where the true state puts it —
            only how far away is affected. It exists because the stand-off
            law is proportional to altitude and altitude can change very
            fast: over a three-minute boost the stand-off grows from 32 km
            to 470 km, and compressed into a few seconds of video that reads
            as the camera cutting wide rather than following.
            the animator passes a
            version of the altitude that has been rate-limited in video
            time. ``None`` uses the real one.
        """
        centre = np.asarray(position, dtype=np.float64)
        radius = float(np.linalg.norm(centre))
        if radius == 0.0:
            msg = "cannot place a chase camera on a vehicle at the body centre"
            raise ValueError(msg)
        up = centre / radius

        heading = np.asarray(velocity, dtype=np.float64)
        speed = float(np.linalg.norm(heading))
        # A stationary sample has no heading; fall back to the local
        # vertical rather than producing NaNs the renderer would silently
        # turn into an empty frame.
        heading = heading / speed if speed > 1e-6 else up

        # Stand off along the *horizontal* part of the heading, not along
        # the heading itself. On a launch the two are completely different:
        # the velocity is straight up, so stepping back along it steps
        # straight down, the eye clamps to its altitude floor, and the
        # vehicle is left as a dot 500 km away on the horizon. Watching
        # that, a launch looks like it begins in mid-air. Standing off
        # horizontally puts the camera beside the pad looking at a rising
        # vehicle, which is what a launch looks like.
        horizontal = heading - float(heading @ up) * up
        extent = float(np.linalg.norm(horizontal))
        if extent > 1e-6:
            horizontal = horizontal / extent
        else:
            # Purely radial: any horizontal direction will do, so take one
            # from the world axis least aligned with the local vertical.
            seed = np.array([0.0, 0.0, 1.0])
            if abs(float(up @ seed)) > 0.9:
                seed = np.array([1.0, 0.0, 0.0])
            horizontal = np.cross(np.cross(up, seed), up)
            horizontal = horizontal / float(np.linalg.norm(horizontal))
        # Blend: on orbit the heading *is* horizontal and the two agree, so
        # this only bites where it has to.
        back_axis = extent * heading + (1.0 - extent) * horizontal
        back_axis = back_axis / float(np.linalg.norm(back_axis))

        # Stand off to one *side* as well as behind. Directly behind is the
        # one place from which a slender body is invisible: the frame looks
        # straight up its own tail and the vehicle reads as a ring with
        # cross-hairs, which is what the first oriented glyphs looked like.
        # A quarter view shows the length, and the length is the thing that
        # changes at separation.
        lateral = np.cross(back_axis, up)
        span = float(np.linalg.norm(lateral))
        # A back axis parallel to the vertical leaves no cross-track
        # direction; the quarter view is simply not taken there.
        lateral = lateral / span if span > 1e-9 else np.zeros(3)

        ellipsoid = as_ellipsoid(surface)
        # Not named `height`: that is the frame's pixel height, and shadowing
        # it here put the vehicle's altitude into the camera's row count.
        standoff_altitude = (
            float(ecef_to_geodetic(centre, ellipsoid)[2]) if altitude is None else float(altitude)
        )
        standoff_altitude = max(standoff_altitude, self.min_altitude)
        shrink = 1.0 - self.tighten * float(ease(progress))
        back = (self.back_scale * standoff_altitude + self.back_offset) * shrink
        lift = (self.lift_scale * standoff_altitude + self.lift_offset) * shrink

        eye = centre - back * back_axis + lift * up + self.side * back * lateral
        # Radial scaling changes geodetic altitude by very nearly the radial
        # step, but not exactly — the normal is not radial. Two passes take
        # the residual below a millimetre, which is a cheaper fix than
        # solving for the offset surface.
        for _ in range(2):
            eye_altitude = float(ecef_to_geodetic(eye, ellipsoid)[2])
            if eye_altitude >= self.floor:
                break
            eye_radius = float(np.linalg.norm(eye))
            eye = eye * (eye_radius + self.floor - eye_altitude) / eye_radius

        return Camera(
            position=eye,
            target=centre + self.lead * back * back_axis,
            up=up,
            fov=self.fov,
            width=int(width),
            height=int(height),
        )


def globe_plate(
    camera: Camera,
    texture: Texture | Sequence[Texture],
    surface: Ellipsoid | float = WGS84,
    sun: _FloatArray | None = None,
    ambient: float = 0.13,
    atmosphere: float = 0.6,
    night: float = 0.20,
    specular: float = 0.10,
    stars: bool = True,
    relief: ReliefMap | None = None,
    displace: bool = False,
    backend: Backend = "numpy",
) -> tuple[Figure, Axes]:
    """A rendered globe on a figure whose axes are **pixels**.

    Every overlay in this module projects to pixel coordinates, so the
    background has to be drawn in the same ones — otherwise a marker and the
    limb it sits against are in two different spaces and only agree by
    accident.

    Specular is kept low by default: at globe scale a broad ocean highlight
    reads as a smudge rather than as sun glint, and it competes with the
    tracks drawn over it.
    """
    import matplotlib.pyplot as plt

    image, _ = render(
        camera,
        to_device(texture, backend),
        surface,
        sun=sun,
        ambient=ambient,
        atmosphere=atmosphere,
        night=night,
        specular=specular,
        background=starfield(camera.width, camera.height) if stars else None,
        relief=None if relief is None else to_device(relief, backend),
        displace=displace,
        backend=backend,
    )
    figure, ax = plt.subplots(figsize=(camera.width / 100, camera.height / 100), dpi=100)
    figure.patch.set_facecolor("black")
    ax.imshow(np.clip(image, 0.0, 1.0), interpolation="bilinear", zorder=0)
    ax.set_xlim(0, camera.width)
    ax.set_ylim(camera.height, 0)
    ax.axis("off")
    figure.subplots_adjust(0, 0, 1, 1)
    return figure, ax


def draw_track(
    ax: Axes,
    points: _FloatArray,
    camera: Camera,
    color: str = "#FF8A3D",
    width: float = 2.0,
    alpha: float = 1.0,
    zorder: float = 3.0,
    surface: Ellipsoid | float | None = WGS84,
    values: _FloatArray | None = None,
    cmap: str = "inferno",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Draw a projected polyline, broken where the globe occludes it.

    Occluded samples become ``nan``, which breaks the polyline exactly where
    the limb cuts in front of it. This is the depth test Matplotlib's
    painter's algorithm cannot do, and the reason trajectories used to be
    drawn in front of the planet they were behind.

    Parameters
    ----------
    values:
        Optional per-sample scalar — stagnation heat flux, dynamic pressure,
        recession — used to colour the line through ``cmap`` instead of the
        flat ``color``. This is the cheapest way to put a physical quantity
        the simulator actually computed into the picture rather than
        alongside it.
    """
    array = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if array.shape[0] < 2:
        return
    px, py, visible = project(array, camera, surface=surface)
    xs = np.where(visible, px, np.nan)
    ys = np.where(visible, py, np.nan)

    if values is None:
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=width,
            alpha=alpha,
            zorder=zorder,
            solid_capstyle="round",
        )
        return

    from matplotlib.collections import LineCollection

    scalars = np.asarray(values, dtype=np.float64)
    if scalars.shape != (array.shape[0],):
        msg = f"values must have one entry per point, got {scalars.shape} for {array.shape}"
        raise ValueError(msg)
    segments = np.stack(
        [np.column_stack([xs[:-1], ys[:-1]]), np.column_stack([xs[1:], ys[1:]])], axis=1
    )
    midpoint = 0.5 * (scalars[:-1] + scalars[1:])
    collection = LineCollection(
        list(segments),
        cmap=cmap,
        linewidths=width,
        alpha=alpha,
        zorder=zorder,
        capstyle="round",
    )
    collection.set_array(midpoint)
    lo = float(np.nanmin(scalars)) if vmin is None else float(vmin)
    hi = float(np.nanmax(scalars)) if vmax is None else float(vmax)
    collection.set_clim(lo, hi if hi > lo else lo + 1.0)
    ax.add_collection(collection)


def draw_marker(
    ax: Axes,
    position: _FloatArray,
    camera: Camera,
    surface: Ellipsoid | float | None = WGS84,
    **kwargs: Any,
) -> bool:
    """Scatter one world point, skipping it when the globe hides it.

    Returns whether it was drawn, so a caller can tell "behind the planet"
    from "drawn".

    Notes
    -----
    A point sitting **exactly** on the surface is drawn. That sounds
    obvious and it was not the case: markers used to be lifted clear of the
    ground — 15 km for a radar site and 20 km in the notebooks — to keep the
    depth test from swallowing them. At globe scale the lift is invisible;
    at the launch and impact close-ups the whole point of this layer, it put
    the aimpoint star **20 km above** the vehicle that had just landed on
    it, which reads as a miss and is indistinguishable from one.

    The lift is unnecessary. Occlusion in :func:`project` compares the
    ray-surface distance against the point's own distance, and for a point
    on the surface those agree to about a nanometre out of 6,378 km, well
    inside the tolerance. So markers go on the ground, and the picture stops
    claiming an error the physics never made.
    """
    px, py, visible = project(np.atleast_2d(position), camera, surface=surface)
    if not bool(visible[0]):
        return False
    ax.scatter(px[0], py[0], **kwargs)
    return True


# -- vehicle glyph -------------------------------------------------------


def glyph_polylines(n_fins: int = 4) -> list[_FloatArray]:
    """Unit-length body-frame polylines for a slender re-entry body.

    The shape is a cone from a nose at ``+x`` back to a base ring at
    ``-0.35 x``, plus ``n_fins`` fins. Returned in body coordinates with the
    nose at unit distance, so a caller scales once and rotates once.

    Kept as line segments rather than a mesh on purpose: the renderer is a
    sphere ray-tracer with a depth buffer, not a scene graph, and a shaded
    solid would need occlusion against itself that nothing here provides.
    """
    if n_fins < 0:
        msg = f"n_fins must be non-negative, got {n_fins}"
        raise ValueError(msg)
    nose = np.array([1.0, 0.0, 0.0])
    base_x, base_r = -0.35, 0.30

    ring_angles = np.linspace(0.0, 2.0 * np.pi, 25)
    ring = np.stack(
        [
            np.full_like(ring_angles, base_x),
            base_r * np.cos(ring_angles),
            base_r * np.sin(ring_angles),
        ],
        axis=1,
    )
    lines: list[_FloatArray] = [ring]

    # Four generators of the cone, at 90 degrees, so the body reads as a
    # solid of revolution rather than as a circle with a dot.
    for phi in np.linspace(0.0, 2.0 * np.pi, 5)[:-1]:
        rim = np.array([base_x, base_r * np.cos(phi), base_r * np.sin(phi)])
        lines.append(np.stack([nose, rim]))

    for phi in np.linspace(0.0, 2.0 * np.pi, n_fins + 1)[:-1]:
        radial = np.array([0.0, np.cos(phi), np.sin(phi)])
        lines.append(
            np.stack(
                [
                    np.array([base_x + 0.30, 0.0, 0.0]) + base_r * radial,
                    np.array([base_x, 0.0, 0.0]) + 2.0 * base_r * radial,
                    np.array([base_x, 0.0, 0.0]) + base_r * radial,
                ]
            )
        )
    return lines


def glyph_world(
    position: _FloatArray,
    dcm: _FloatArray,
    scale: float,
    n_fins: int = 4,
) -> list[_FloatArray]:
    """The glyph's polylines placed and oriented in the inertial frame.

    Separated from :func:`draw_vehicle` so the orientation can be checked
    arithmetically rather than by looking at a picture: the nose must land
    at ``position + scale * C.T @ NOSE_AXIS`` for the direction cosine
    matrix ``C`` the run actually integrated.

    Parameters
    ----------
    dcm:
        :math:`\\mathbf{C}_E^B`, inertial to body. Body vectors go the other
        way by its transpose, which is its inverse because it is
        orthonormal — a property
        :func:`~aether.dynamics.attitude.dcm_from_quaternion` guarantees and
        which is checked in the tests rather than assumed here.
    """
    centre = np.asarray(position, dtype=np.float64).reshape(3)
    rotation = np.asarray(dcm, dtype=np.float64)
    if rotation.shape != (3, 3):
        msg = f"dcm must be a single 3x3 matrix, got shape {rotation.shape}"
        raise ValueError(msg)
    to_inertial = rotation.T
    return [centre + float(scale) * (line @ to_inertial.T) for line in glyph_polylines(n_fins)]


def draw_vehicle(
    ax: Axes,
    position: _FloatArray,
    dcm: _FloatArray,
    camera: Camera,
    scale: float | None = None,
    color: str = "#FFFFFF",
    width: float = 1.6,
    zorder: float = 8.0,
    surface: Ellipsoid | float | None = WGS84,
    n_fins: int = 4,
    screen_fraction: float = 0.08,
) -> bool:
    """Draw the vehicle oriented by its integrated attitude.

    Parameters
    ----------
    dcm:
        :math:`\\mathbf{C}_E^B`, mapping inertial components to body
        components — exactly what
        :func:`~aether.dynamics.attitude.dcm_from_quaternion` returns and
        what a history sample puts in
        ``state["dcm"]``. Body vectors are taken to inertial by its
        transpose, which for an orthonormal matrix is its inverse.
    scale:
        Nose distance in metres. ``None`` sizes the glyph so it subtends
        ``screen_fraction`` of the frame height at the current camera
        distance.
    screen_fraction:
        Fraction of frame height the glyph occupies when ``scale`` is
        ``None``.

    Returns
    -------
    bool
        Whether anything was drawn. ``False`` when the globe occludes the
        vehicle — a caller that wants a "behind the planet" cue can use it.

    Notes
    -----
    **Not to scale, and it cannot be.** A 15 m vehicle at a 500 km
    stand-off subtends 30 nrad; at a 42-degree field of view over 720 rows
    that is 1/200 000 of a pixel. A true-scale glyph is an empty frame, so
    this one is sized in screen space and the exaggeration is declared here
    rather than implied by the picture.
    """
    centre = np.asarray(position, dtype=np.float64).reshape(3)

    _, _, in_front = project(centre[None, :], camera, surface=surface)
    if not bool(in_front[0]):
        return False

    if scale is None:
        distance = float(np.linalg.norm(centre - np.asarray(camera.position)))
        scale = float(screen_fraction * distance * np.tan(0.5 * camera.fov))

    for world in glyph_world(centre, dcm, scale, n_fins):
        px, py, visible = project(world, camera, surface=surface)
        ax.plot(
            np.where(visible, px, np.nan),
            np.where(visible, py, np.nan),
            color=color,
            linewidth=width,
            zorder=zorder,
            solid_capstyle="round",
        )
    return True


def dcm_from_flight_path(
    velocity: _FloatArray, position: _FloatArray, surface: Ellipsoid | float = WGS84
) -> _FloatArray:
    """A direction cosine matrix that points the nose along the flight path.

    **A stated presentation convention, not a computed attitude.** A
    a point-mass trajectory is a point mass: its
    position is real and its orientation was never calculated, which is why
    the history's attitude flag reports
    ``False`` and why an *invented quaternion* would be a lie.

    This is a different claim, and a weaker one. It says only "the body is
    drawn along its own velocity, rolled upright against the local
    vertical", which is what a track symbol on a chart says. It asserts no
    angle of attack, no trim, and nothing about the rotational state — and
    the alternative on offer is a dot, which tells a viewer less than
    nothing about which way the vehicle is going.

    Where a producer *did* integrate an attitude, that attitude is used and
    this function is not called.

    Returns
    -------
    numpy.ndarray
        :math:`\\mathbf{C}_E^B`, inertial to body — the same convention
        :func:`~aether.dynamics.attitude.dcm_from_quaternion` returns, so
        the glyph code cannot tell the two apart.
    """
    speed_vector = np.asarray(velocity, dtype=np.float64).reshape(3)
    centre = np.asarray(position, dtype=np.float64).reshape(3)
    ellipsoid = as_ellipsoid(surface)
    latitude, longitude, _ = ecef_to_geodetic(centre, ellipsoid)
    vertical = np.asarray(
        [
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ]
    )
    speed = float(np.linalg.norm(speed_vector))
    # A vehicle at rest on the pad has no flight path; it stands up.
    nose = speed_vector / speed if speed > 1.0e-6 else vertical
    upward = vertical - float(vertical @ nose) * nose
    extent = float(np.linalg.norm(upward))
    if extent < 1.0e-9:
        # Flying straight up: any roll will do, so take one from the world
        # axis least aligned with the nose rather than produce NaNs.
        seed = np.array([0.0, 0.0, 1.0])
        if abs(float(nose @ seed)) > 0.9:
            seed = np.array([1.0, 0.0, 0.0])
        upward = seed - float(seed @ nose) * nose
        extent = float(np.linalg.norm(upward))
    upward = upward / extent
    side = np.cross(upward, nose)
    return np.asarray(np.stack([nose, side, upward], axis=0))


def stack_world(
    position: _FloatArray,
    dcm: _FloatArray,
    lines: Sequence[_FloatArray],
    scale: float,
) -> list[_FloatArray]:
    """Place metre-scale body polylines in the inertial frame at ``scale``.

    The counterpart of :func:`glyph_world` for a stack whose shape came from
    the mass model rather than from :func:`glyph_polylines`. ``scale`` is a
    pure magnification: the input is already in metres and in the body
    frame, so ``1.0`` draws the vehicle at true size — which is invisible,
    for the reason :func:`draw_vehicle` states.
    """
    centre = np.asarray(position, dtype=np.float64).reshape(3)
    rotation = np.asarray(dcm, dtype=np.float64)
    if rotation.shape != (3, 3):
        msg = f"dcm must be a single 3x3 matrix, got shape {rotation.shape}"
        raise ValueError(msg)
    to_inertial = rotation.T
    return [
        centre + float(scale) * (np.asarray(line, dtype=np.float64) @ to_inertial.T)
        for line in lines
    ]


def draw_stack(
    ax: Axes,
    position: _FloatArray,
    dcm: _FloatArray,
    camera: Camera,
    lines: Sequence[_FloatArray],
    color: str = "#FFFFFF",
    width: float = 1.4,
    zorder: float = 8.0,
    surface: Ellipsoid | float | None = WGS84,
    screen_fraction: float = 0.10,
) -> bool:
    """Draw a multi-stage stack from body-frame polylines in metres.

    Sized so the **whole remaining stack** subtends ``screen_fraction`` of
    the frame height. That is deliberate and it is the opposite of what a
    first attempt does: normalising each configuration to a fixed screen
    length would make the vehicle appear to *grow* at every separation,
    because a shorter stack drawn to the same size is a bigger one. Fixing
    the magnification to the stack's own length at first sight would be
    worse still — by the time the payload is alone it would be four pixels.

    So the scale is set from the current stack's length, and the *step* at
    separation is real: the picture gets shorter because the vehicle did.

    Returns
    -------
    bool
        Whether anything was drawn; ``False`` when the body occludes it.
    """
    if not lines:
        return False
    centre = np.asarray(position, dtype=np.float64).reshape(3)
    _, _, in_front = project(centre[None, :], camera, surface=surface)
    if not bool(in_front[0]):
        return False

    stacked = np.concatenate([np.asarray(line, dtype=np.float64) for line in lines])
    length = float(np.ptp(stacked[:, 0]))
    if length <= 0.0:  # pragma: no cover - a zero-length stack
        return False
    distance = float(np.linalg.norm(centre - np.asarray(camera.position)))
    scale = screen_fraction * distance * np.tan(0.5 * camera.fov) / length

    for world in stack_world(centre, dcm, lines, scale):
        px, py, visible = project(world, camera, surface=surface)
        ax.plot(
            np.where(visible, px, np.nan),
            np.where(visible, py, np.nan),
            color=color,
            linewidth=width,
            zorder=zorder,
            solid_capstyle="round",
        )
    return True


def draw_mesh(
    ax: Axes,
    vertices: _FloatArray,
    faces: _IntArray,
    position: _FloatArray,
    dcm: _FloatArray,
    camera: Camera,
    color: str = "#FFFFFF",
    width: float = 1.0,
    zorder: float = 8.0,
    surface: Ellipsoid | float | None = WGS84,
    scale: float = 1.0,
    light: _FloatArray | None = None,
) -> bool:
    """Draw a 3D mesh at a given position and orientation.

    Parameters
    ----------
    vertices:
        Shape ``(n, 3)`` in the body frame, metres.
    faces:
        Shape ``(m, 3)`` integer vertex indices.
    position, dcm:
        Vehicle state.
    camera:
        View.
    color:
        Base tint.
    width:
        Unused; kept for signature compatibility with :func:`draw_stack`.
    zorder, surface:
        Passed through to Matplotlib.
    scale:
        Metre-to-pixel factor, the same one :func:`draw_stack` computes.
    light:
        Unit world-space light direction. ``None`` uses flat shading.
    """
    centre = np.asarray(position, dtype=np.float64).reshape(3)
    _, _, in_front = project(centre[None, :], camera, surface=surface)
    if not bool(in_front[0]):
        return False

    rotation = np.asarray(dcm, dtype=np.float64)
    if rotation.shape != (3, 3):
        msg = f"dcm must be a single 3x3 matrix, got shape {rotation.shape}"
        raise ValueError(msg)
    world = centre + float(scale) * (np.asarray(vertices, dtype=np.float64) @ rotation)

    px, py, vert_vis = project(world, camera, surface=surface)
    if not np.any(vert_vis):
        return False

    v0 = world[faces[:, 0]]
    v1 = world[faces[:, 1]]
    v2 = world[faces[:, 2]]
    centres = (v0 + v1 + v2) / 3.0
    _, _, centre_vis = project(centres, camera, surface=surface)

    all_vis = vert_vis[faces].all(axis=1)
    draw = centre_vis & all_vis
    if not np.any(draw):
        return False

    depths = np.linalg.norm(centres[draw] - np.asarray(camera.position), axis=1)
    order = np.argsort(depths)[::-1]

    if light is not None:
        light_dir = np.asarray(light, dtype=np.float64)
        light_dir = light_dir / np.linalg.norm(light_dir)
        normals = np.cross(v1[draw] - v0[draw], v2[draw] - v0[draw])
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.where(norms > 1e-12, norms, 1.0)
        normals = normals / norms
        brightness = np.clip(normals @ light_dir, 0.15, 1.0)
    else:
        brightness = np.ones(np.count_nonzero(draw))

    from matplotlib.colors import to_rgb

    base = np.asarray(to_rgb(color), dtype=np.float64)

    for _, face_idx in enumerate(order):
        f = faces[draw][face_idx]
        face_px = px[f]
        face_py = py[f]
        b = float(brightness[face_idx])
        ax.fill(
            face_px,
            face_py,
            facecolor=base * b,
            edgecolor="none",
            zorder=zorder,
        )
    return True


# -- sensor overlays -----------------------------------------------------


def horizon_ring(
    position: GeodeticPosition,
    mask_elevation: float,
    altitude: float,
    surface: Ellipsoid | float = WGS84,
    samples: int = 181,
) -> _FloatArray:
    """The circle an observer can see out to, at one target altitude.

    Takes a position and a mask elevation rather than a site object. The
    geometry is a spherical cap and does not care what is standing at the
    centre of it; typing the argument as a sensor was what kept this function
    out of the kernel, and it was never using anything else from one.

    A small circle of central angle
    :func:`~aether.ellipsoid.horizon_central_angle` about the site,
    drawn at the vehicle's radius rather than on the ground — because that
    is where the boundary actually is. Drawing it on the surface would show
    a footprint some three times too small at 150 km and understate the
    sensor by exactly the amount the fractional-orbital argument turns on.

    Returns
    -------
    numpy.ndarray
        Shape ``(samples, 3)``, closed (last point equals the first).

    Notes
    -----
    The small circle is built about the site's **local** surface radius, not
    about a global mean one. :func:`~aether.ellipsoid.horizon_central_angle`
    is a spherical relation and stays one; taking its radius from the
    ellipsoid at the site's own latitude removes the part of the error that
    is first order in the flattening, which is the part worth removing. What
    remains is the circle's departure from a true ellipsoidal locus over its
    own extent, second order in ``f`` and under a kilometre at any mask
    angle a warning radar uses.
    """
    ellipsoid = as_ellipsoid(surface)
    local_radius = float(ellipsoid.surface_radius(position.latitude))
    lam = float(horizon_central_angle(altitude, mask_elevation, local_radius))
    normal = geodetic_to_cartesian(position, ellipsoid)
    normal = normal / float(np.linalg.norm(normal))

    # Any pair of vectors spanning the plane perpendicular to the site.
    seed = np.array([0.0, 0.0, 1.0])
    if abs(float(normal @ seed)) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    east = np.cross(normal, seed)
    east = east / float(np.linalg.norm(east))
    north = np.cross(normal, east)

    phi = np.linspace(0.0, 2.0 * np.pi, samples)
    radius = local_radius + max(float(altitude), 0.0)
    ring = radius * (
        np.cos(lam) * normal[None, :]
        + np.sin(lam)
        * (np.cos(phi)[:, None] * east[None, :] + np.sin(phi)[:, None] * north[None, :])
    )
    return np.asarray(ring)


def draw_horizon_ring(
    ax: Axes,
    position: GeodeticPosition,
    mask_elevation: float,
    altitude: float,
    camera: Camera,
    surface: Ellipsoid | float = WGS84,
    color: str = "#FF3B30",
    width: float = 1.2,
    alpha: float = 0.75,
    zorder: float = 4.0,
) -> None:
    """Project an observer's visibility circle at the target's altitude."""
    draw_track(
        ax,
        horizon_ring(position, mask_elevation, altitude, surface),
        camera,
        color=color,
        width=width,
        alpha=alpha,
        zorder=zorder,
        surface=surface,
    )
