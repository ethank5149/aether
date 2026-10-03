#!/usr/bin/env python3
"""Generate the proposal's figures: ``python tools/gen_proposal_figures.py``.

Two figures, and both are illustrations: neither reports a result.

**skip-entry.pdf.** What a skip entry looks like against time. Orion's
Artemis I lunar return as :mod:`examples.artemis1.entry` flies it from the
published entry state -- altitude above, and below it the drag power per unit
mass :math:`P = DV/m`, the rate at which mechanical energy is dissipated. The
intervals on which :math:`P` exceeds a tenth of its peak are shaded: those
are the atmospheric passes. No number is printed on the figure. Before
anything is drawn the script checks that the curve it plots is the
dissipation: the integral of :math:`P` over the flight must equal the energy
lost, which over a rotating planet is the Jacobi integral and is likewise an
output of the field.

**reachable-set.pdf.** The statement the proposal aims at, drawn where a
landing footprint is drawn. The same flight over the globe, in perspective
from above the Americas, with its altitude exaggerated so that the skip can
be seen; and on the ground ahead of it three nested sets: what the vehicle
can reach, what the limit system can reach, and the latter thickened by the
residual, which is the bound. **The three sets are drawn for the eye. They are
shapes typed in below, not the output of any computation**, and the figure
says so on its face.

The coastlines are Natural Earth's, which is in the public domain; the file
is kept in ``tools/data`` so that the figure regenerates without a network.
Geodetic latitudes are drawn as latitudes on a sphere, as on any map at this
scale: the difference is a fifth of a degree at most, under the width of a
line.

The output is tracked beside the manuscript, so the LaTeX build needs no
Python. Regenerate it when the example changes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.path as mpath  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from examples.artemis1.entry import (  # noqa: E402
    flown_model,
    fly,
    guidance_model,
    orion_aerodynamics,
)
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import PathPatch, Rectangle  # noqa: E402

OUT = _ROOT / "manuscript" / "proposal" / "figures"
_LAND = Path(__file__).resolve().parent / "data" / "ne_110m_land.geojson"

#: A pass is where the drag power exceeds this fraction of its peak.
_PASS_THRESHOLD = 0.10
#: The energy identity is exact for the model; this is the quadrature's error.
_ENERGY_TOLERANCE = 1.0e-3

# Two hues on the time history, altitude and dissipation. Text and chrome
# stay in ink and grey, so the only colour on the page is what the figure is
# about. The globe has its own, below.
_BLUE, _ORANGE = "#2a78d6", "#eb6834"
_INK, _INK_SECONDARY = "#0b0b0b", "#52514e"
_GRID, _AXIS, _BAND = "#e1e0d9", "#c3c2b7", "#f0efec"
_LAND_FILL, _COAST = "#ecebe6", "#b9b8ae"

#: Radius (km) of the sphere the globe is drawn on.
_GLOBE_RADIUS = 6371.0
#: The camera of the globe: the point it is above (degrees) and its height
#: (km). East of the track, so that the Americas say where this is and the
#: trajectory rises over open ocean.
_CAMERA = (-102.0, 10.0, 16000.0)
#: Altitudes are drawn this many times too large: 122 km is a fiftieth of the
#: radius, and the skip is a fifth of that.
_EXAGGERATION = 12.0
#: A post from the ground track to the trajectory at each of these (s).
_POST_EVERY = 60.0

# The three sets of the globe, as fractions of the range flown. THESE ARE
# NOT DATA. They are chosen so that the picture says what the target
# statement says: a broad fan ahead of the vehicle, the vehicle's set
# sticking out of the limit system's on one flank, by less than the residual.
_LIMIT_SET = (0.60, 1.34)         # where the limit system's set starts and ends
_LIMIT_HALF_WIDTH = 0.30
_LIMIT_DRIFT = 0.05               # how far its centre line leaves the track
_RESIDUAL = 0.07

# The sets are nested, so they share one hue and differ by its step, light
# for what is reached to dark for the bound (an ordinal ramp; the validator
# of the palette passes it). A warm fill inside cool outlines read as
# something else from across a room.
_REACHED, _LIMIT, _BOUND, _BAND = "#86b6ef", "#256abf", "#104281", "#cde2fb"


@dataclass(frozen=True)
class SkipProfile:
    """The example's flight, as the two figures draw it."""

    times: np.ndarray
    altitudes_km: np.ndarray
    power_kw: np.ndarray
    """Drag power per unit mass, kW/kg."""
    passes: tuple[tuple[float, float], ...]
    """The intervals on which the drag power stays above the threshold."""
    track: np.ndarray
    """Ground track, as (longitude, latitude) in degrees."""
    energy_lost_mj: float
    energy_residual: float
    """Relative difference between the integral of the drag power and the
    mechanical energy lost: the check that the curve plotted is the dissipation."""


def _use_document_fonts() -> None:
    """Set the labels in the manuscript's typeface when TeX can find it.

    The proposal is set in Latin Modern. A figure in another face reads as
    pasted in, so the font is looked up through ``kpsewhich`` and registered;
    where there is no TeX the Computer Modern that ships with matplotlib is
    the same design, and the figure still builds.
    """
    families = ["cmr10"]
    try:
        found = subprocess.run(
            ["kpsewhich", "lmroman10-regular.otf", "lmroman10-italic.otf"],
            capture_output=True, text=True, check=False, timeout=30,
        ).stdout.split()
    except (OSError, subprocess.SubprocessError):
        found = []
    for path in found:
        font_manager.fontManager.addfont(path)
    if found:
        families.insert(0, font_manager.FontProperties(fname=found[0]).get_name())
    plt.rcParams.update({
        "font.family": families,
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "font.size": 9.0,
        "pdf.fonttype": 42,
    })


def skip_profile() -> SkipProfile:
    """Fly the example and reduce it to what the two figures draw."""
    aero = orion_aerodynamics()
    result, _guidance = fly(aero)
    if result.plant_states is None:
        raise RuntimeError("the example did not fly a plant that reports its own outputs")
    along = flown_model(aero).output_table(result.plant_states)
    times = np.asarray(result.times, dtype=float)
    power = np.asarray(along["drag_power"], dtype=float)
    dissipated = float(np.trapezoid(power, times))
    energy = np.asarray(along["energy"], dtype=float)
    lost = float(energy[0] - energy[-1])
    residual = abs(dissipated - lost) / lost
    if residual > _ENERGY_TOLERANCE:
        raise RuntimeError(
            f"the drag power integrates to {dissipated:.4e} J/kg but the flight lost "
            f"{lost:.4e} J/kg of mechanical energy: the curve plotted is not the dissipation"
        )
    model = guidance_model(aero)
    ground = [model.ground_point(state) for state in result.states.T]
    return SkipProfile(
        times=times,
        altitudes_km=np.asarray(result.altitudes, dtype=float) / 1e3,
        power_kw=power / 1e3,
        passes=_passes(times, power),
        track=np.rad2deg([[point.longitude, point.latitude] for point in ground]),
        energy_lost_mj=lost / 1e6,
        energy_residual=residual,
    )


def _passes(times: np.ndarray, power: np.ndarray) -> tuple[tuple[float, float], ...]:
    """Maximal intervals on which the power exceeds the threshold."""
    level = _PASS_THRESHOLD * float(power.max())
    above = power > level
    edges = np.flatnonzero(np.diff(above.astype(int)))
    crossings = [
        float(times[k] + (level - power[k]) * (times[k + 1] - times[k]) / (power[k + 1] - power[k]))
        for k in edges
    ]
    if above[0]:
        crossings.insert(0, float(times[0]))
    if above[-1]:
        crossings.append(float(times[-1]))
    return tuple(zip(crossings[0::2], crossings[1::2], strict=True))


def _save(figure: plt.Figure, path: Path, **options: object) -> None:
    """Write the figure, without a creation date and without a background.

    The PDF is tracked, so regenerating it from unchanged code must give an
    unchanged file; and a white rectangle on a white page shows as a hairline
    at its edge in some viewers.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = path.suffix == ".pdf"
    figure.savefig(path, metadata={"CreationDate": None} if pdf else None,
                   facecolor="none" if pdf else "white", **options)
    plt.close(figure)


def draw_skip_entry(profile: SkipProfile, path: Path) -> None:
    """Altitude over drag power, on one time axis, with the passes shaded."""
    figure, (top, bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(6.5, 3.2), height_ratios=(1.0, 0.85),
        gridspec_kw={"hspace": 0.12},
    )
    for axes in (top, bottom):
        for start, end in profile.passes:
            axes.axvspan(start, end, color=_BAND, linewidth=0.0, zorder=0)
        axes.grid(axis="y", color=_GRID, linewidth=0.5, zorder=1)
        axes.set_axisbelow(True)
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axes.spines[side].set_color(_AXIS)
            axes.spines[side].set_linewidth(0.6)
        axes.tick_params(colors=_INK_SECONDARY, width=0.6, length=3.0)
        axes.yaxis.label.set_color(_INK)
        axes.margins(x=0.0)

    top.plot(profile.times, profile.altitudes_km, color=_BLUE, linewidth=1.6,
             solid_capstyle="round", zorder=3)
    top.set_ylabel("altitude (km)")
    top.set_ylim(0.0, 130.0)
    top.set_yticks([0, 40, 80, 120])
    top.tick_params(axis="x", length=0.0)

    bottom.fill_between(profile.times, profile.power_kw, color=_ORANGE, alpha=0.12,
                        linewidth=0.0, zorder=2)
    bottom.plot(profile.times, profile.power_kw, color=_ORANGE, linewidth=1.6,
                solid_capstyle="round", zorder=3)
    bottom.set_ylabel("drag power $P$ (kW/kg)")
    peak = float(profile.power_kw.max())
    bottom.set_ylim(0.0, 1.12 * peak)
    bottom.set_yticks(np.arange(0.0, 1.02 * peak, 100.0))
    bottom.set_xlabel("time from entry interface (s)", color=_INK)
    bottom.set_xlim(float(profile.times[0]), float(profile.times[-1]))

    # Direct labels, and only these: the passes and what lies between them.
    # They sit on one baseline in the headroom above the altitude curve.
    names = ["first pass", "second pass"]
    marks = [(0.5 * (start + end), name)
             for (start, end), name in zip(profile.passes, names, strict=False)]
    marks += [(0.5 * (before[1] + after[0]), "loft") for before, after in pairwise(profile.passes)]
    for when, name in marks:
        top.text(when, 0.875, name, ha="center", va="baseline", color=_INK_SECONDARY,
                 transform=top.get_xaxis_transform())

    figure.align_ylabels((top, bottom))
    _save(figure, path, bbox_inches="tight", pad_inches=0.02)


def _directions(lonlat: np.ndarray) -> np.ndarray:
    """Unit vectors of points given as (longitude, latitude) in degrees."""
    longitude, latitude = np.deg2rad(np.asarray(lonlat, dtype=float)).T
    return np.column_stack([
        np.cos(latitude) * np.cos(longitude),
        np.cos(latitude) * np.sin(longitude),
        np.sin(latitude),
    ])


def _along_great_circles(directions: np.ndarray, step: float, *, close: bool = False) -> np.ndarray:
    """A polyline of unit vectors, refined so that no leg exceeds ``step`` degrees.

    A leg between two points on the ground is the great circle through them,
    and on a globe that is a curve: it has to be drawn through enough points.
    """
    points = np.vstack([directions, directions[:1]]) if close else directions
    out = [points[:1]]
    for a, b in pairwise(points):
        angle = float(np.arccos(np.clip(a @ b, -1.0, 1.0)))
        if angle < 1e-12:
            continue
        along = np.linspace(0.0, 1.0, max(1, int(np.ceil(np.rad2deg(angle) / step))) + 1)[1:, None]
        out.append((np.sin((1.0 - along) * angle) * a + np.sin(along * angle) * b) / np.sin(angle))
    return np.vstack(out)


@dataclass(frozen=True)
class _Camera:
    """A pinhole above a sphere, looking straight down with north up.

    Image coordinates are tangents of the angle off the axis, so the globe is
    a disc about the origin; what lies beyond its limb cannot be seen.
    """

    position: np.ndarray
    right: np.ndarray
    up: np.ndarray

    @classmethod
    def above(cls, longitude: float, latitude: float, height: float) -> _Camera:
        axis = _directions(np.array([[longitude, latitude]]))[0]
        right = np.cross([0.0, 0.0, 1.0], axis)
        right /= np.linalg.norm(right)
        return cls((_GLOBE_RADIUS + height) * axis, right, np.cross(axis, right))

    @property
    def axis(self) -> np.ndarray:
        return self.position / np.linalg.norm(self.position)

    @property
    def horizon(self) -> float:
        """Arc (rad) from the point below the camera to the limb."""
        return float(np.arccos(_GLOBE_RADIUS / np.linalg.norm(self.position)))

    @property
    def limb_radius(self) -> float:
        """Radius of the globe's disc in image coordinates."""
        return float(1.0 / np.tan(self.horizon))

    def project(self, points: np.ndarray) -> np.ndarray:
        """Image coordinates of points given in kilometres from the centre."""
        seen = np.asarray(points, dtype=float) - self.position
        depth = -(seen @ self.axis)
        return np.column_stack([seen @ self.right, seen @ self.up]) / depth[:, None]

    def ground(self, directions: np.ndarray) -> np.ndarray:
        """Image coordinates of points on the surface."""
        return self.project(_GLOBE_RADIUS * np.asarray(directions, dtype=float))

    def visible(self, directions: np.ndarray) -> np.ndarray:
        return np.asarray(directions) @ self.axis > np.cos(self.horizon)

    def region(self, boundary: np.ndarray) -> np.ndarray | None:
        """Image polygon of the visible part of a region on the ground, or ``None``.

        Each point of the boundary beyond the limb is brought back to the limb
        along the great circle through the point below the camera, which
        leaves the visible part of the region as it was and closes it along
        the limb.
        """
        boundary = np.asarray(boundary, dtype=float)
        cosines = boundary @ self.axis
        edge = self.horizon - 1e-4
        hidden = cosines < np.cos(edge)
        if hidden.all():
            return None
        across = boundary[hidden] - np.outer(cosines[hidden], self.axis)
        length = np.linalg.norm(across, axis=1, keepdims=True)
        across = np.divide(across, length, out=np.zeros_like(across), where=length > 1e-12)
        moved = boundary.copy()
        moved[hidden] = np.cos(edge) * self.axis + np.sin(edge) * across
        return self.ground(moved)

    def runs(self, directions: np.ndarray) -> list[np.ndarray]:
        """Image polylines of the visible stretches of a curve on the ground."""
        seen = self.visible(directions)
        breaks = np.flatnonzero(np.diff(seen.astype(int))) + 1
        return [
            self.ground(directions[piece])
            for piece in np.split(np.arange(len(seen)), breaks)
            if seen[piece[0]] and len(piece) > 1
        ]

    def shading(
        self, extent: tuple[float, float, float, float], shape: tuple[int, int]
    ) -> np.ndarray:
        """Black, with an opacity that grows away from a light over the left shoulder.

        The one raster in the figure: it is what makes the disc a ball.
        """
        left, right, bottom, top = extent
        x, y = np.meshgrid(np.linspace(left, right, shape[1]), np.linspace(top, bottom, shape[0]))
        ray = -self.axis + x[..., None] * self.right + y[..., None] * self.up
        ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
        toward = ray @ self.position
        inside = toward**2 - (self.position @ self.position - _GLOBE_RADIUS**2)
        hit = inside > 0.0
        normal = (
            self.position + (-toward - np.sqrt(np.where(hit, inside, 0.0)))[..., None] * ray
        ) / _GLOBE_RADIUS
        light = -0.45 * self.right + 0.55 * self.up + 0.70 * self.axis
        lit = np.clip(normal @ (light / np.linalg.norm(light)), 0.0, 1.0)
        image = np.zeros((*shape, 4))
        image[..., 3] = np.where(hit, 0.13 * (1.0 - lit) ** 1.5, 0.0)
        return image


def _land() -> list[np.ndarray]:
    """The outer ring of each of Natural Earth's land polygons, as unit vectors."""
    features = json.loads(_LAND.read_text())["features"]
    return [
        _along_great_circles(_directions(np.array(feature["geometry"]["coordinates"][0])), 1.0)
        for feature in features
    ]


def _illustrative_sets(track: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three nested sets on the ground ahead of the flight, drawn and not computed.

    Returned as closed curves of unit vectors: the vehicle's reachable set,
    the limit system's, and the limit system's thickened by the margin. They
    are laid out along the great circle from where the flight starts to where
    it ends, in fractions of that range, from the constants at the top of
    this file. Nothing here comes from a model.
    """
    start, end = _directions(track[[0, -1]])
    flown = float(np.arccos(np.clip(start @ end, -1.0, 1.0)))
    ahead = end - (end @ start) * start
    ahead /= np.linalg.norm(ahead)
    aside = np.cross(ahead, start)

    def on_the_ground(downrange: np.ndarray, crossrange: np.ndarray) -> np.ndarray:
        """Unit vectors of points given along and across the track, in radians of arc."""
        arc = np.hypot(downrange, crossrange)[:, None]
        bearing = np.arctan2(crossrange, downrange)[:, None]
        heading = np.cos(bearing) * ahead + np.sin(bearing) * aside
        return np.cos(arc) * start + np.sin(arc) * heading

    along = np.linspace(0.0, 1.0, 241)
    # A fan: blunt where it begins, widest toward its far end, round there.
    fan = (0.22 + 0.78 * along**0.85) * np.sqrt(1.0 - along**6)
    fan *= _LIMIT_HALF_WIDTH / fan.max()
    drift = _LIMIT_DRIFT * along

    def outline(downrange: np.ndarray, one_side: np.ndarray, other: np.ndarray) -> np.ndarray:
        return np.column_stack([
            np.r_[downrange, downrange[::-1]], np.r_[one_side, other[::-1]],
        ])

    first, last = _LIMIT_SET
    limit = outline(first + (last - first) * along, drift + fan, drift - fan)
    # The vehicle's set: inside on one flank, and out past the limit system's
    # on the other by half the residual, which is what the residual is for.
    bulge = 0.80 + 0.30 * np.exp(-(((along - 0.72) / 0.16) ** 2))
    vehicle = outline(
        first + 0.07 + (last - first - 0.12) * along,
        drift + fan * bulge,
        drift - fan * (0.68 + 0.05 * np.sin(5.0 * along)),
    )
    # The limit system's set plus a disc of the residual's radius. The set is
    # convex, so in each direction the sum's edge is the set's farthest point
    # that way, moved out by the residual: round at the corners, as a sum is.
    around = np.linspace(0.0, 2.0 * np.pi, 721)[:-1]
    outward = np.column_stack([np.cos(around), np.sin(around)])
    thickened = limit[np.argmax(outward @ limit.T, axis=1)] + _RESIDUAL * outward
    return tuple(  # type: ignore[return-value]
        on_the_ground(*(flown * curve).T) for curve in (vehicle, limit, thickened)
    )


def draw_target(profile: SkipProfile, path: Path) -> None:
    """The statement aimed at, on the ground under a skip entry.

    One hue in three steps for the three nested sets; everything else is ink
    and grey. The flight is the example's. The three sets are an
    illustration, and the figure says so.
    """
    width, height = 6.5, 3.3
    middle, radius = np.array([2.72, 1.65]), 1.6      # of the globe, in inches
    camera = _Camera.above(*_CAMERA)
    scale = radius / camera.limb_radius               # inches to the image unit

    def inches(image: np.ndarray) -> np.ndarray:
        return middle + scale * np.asarray(image)

    figure = plt.figure(figsize=(width, height))
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axes.set_xlim(0.0, width)
    axes.set_ylim(0.0, height)
    axes.set_aspect("equal")
    axes.axis("off")

    # -- the globe ---------------------------------------------------------------
    turn = np.linspace(0.0, 2.0 * np.pi, 361)
    limb = middle + radius * np.column_stack([np.cos(turn), np.sin(turn)])
    axes.fill(limb[:, 0], limb[:, 1], color="white", linewidth=0.0, zorder=0)
    disc = PathPatch(mpath.Path(limb), facecolor="none", edgecolor="none")
    axes.add_patch(disc)

    vertices: list[np.ndarray] = []
    codes: list[int] = []
    for ring in _land():
        outline = camera.region(ring)
        if outline is not None:
            vertices.append(inches(outline))
            codes += [mpath.Path.MOVETO, *[mpath.Path.LINETO] * (len(outline) - 2),
                      mpath.Path.CLOSEPOLY]
    axes.add_patch(PathPatch(
        mpath.Path(np.vstack(vertices), codes), facecolor=_LAND_FILL, edgecolor=_COAST,
        linewidth=0.35, zorder=1, clip_path=disc,
    ))
    span = np.linspace(-180.0, 180.0, 721)
    lines = [np.column_stack([np.full(361, longitude), span[::2] / 2.0])
             for longitude in range(-180, 180, 15)]
    lines += [np.column_stack([span, np.full(721, latitude)]) for latitude in range(-75, 90, 15)]
    for line in lines:
        for run in camera.runs(_directions(line)):
            run = inches(run)
            axes.plot(run[:, 0], run[:, 1], color=_GRID, linewidth=0.4, zorder=2, clip_path=disc)
    # The raster covers the disc and nothing else, and leaves the limits alone.
    box = 1.01 * camera.limb_radius
    pixels = int(2.02 * radius * 140)
    axes.imshow(
        camera.shading((-box, box, -box, box), (pixels, pixels)), interpolation="bilinear",
        extent=(middle[0] - scale * box, middle[0] + scale * box,
                middle[1] - scale * box, middle[1] + scale * box),
        zorder=2.5, clip_path=disc,
    )
    axes.set_xlim(0.0, width)
    axes.set_ylim(0.0, height)

    # -- on the ground: the three sets of the statement, drawn and not computed ------
    vehicle, limit, thickened = (
        _along_great_circles(curve, 0.5, close=True)
        for curve in _illustrative_sets(profile.track)
    )
    for curve, style in (
        (thickened, {"color": _BAND, "alpha": 0.55, "zorder": 3}),
        (vehicle, {"color": _REACHED, "alpha": 0.70, "zorder": 4}),
    ):
        filled = camera.region(curve)
        if filled is not None:
            filled = inches(filled)
            axes.fill(filled[:, 0], filled[:, 1], linewidth=0.0, clip_path=disc, **style)
    for curve, style in (
        (limit, {"color": _LIMIT, "linewidth": 1.1, "zorder": 6, "dashes": (4.0, 2.2)}),
        (thickened, {"color": _BOUND, "linewidth": 1.6, "zorder": 6}),
    ):
        for run in camera.runs(curve):
            run = inches(run)
            axes.plot(run[:, 0], run[:, 1], solid_capstyle="round", **style)

    # -- above it: the flight, its altitude exaggerated ----------------------------
    over = _directions(profile.track)
    ground = inches(camera.ground(over))
    flight = inches(camera.project(
        (_GLOBE_RADIUS + _EXAGGERATION * profile.altitudes_km)[:, None] * over
    ))
    axes.fill(np.r_[ground[:, 0], flight[::-1, 0]], np.r_[ground[:, 1], flight[::-1, 1]],
              color=_INK, alpha=0.07, linewidth=0.0, zorder=7)
    for moment in np.arange(0.0, profile.times[-1], _POST_EVERY):
        k = int(np.searchsorted(profile.times, moment))
        axes.plot([ground[k, 0], flight[k, 0]], [ground[k, 1], flight[k, 1]],
                  color=_INK_SECONDARY, linewidth=0.3, zorder=8)
    axes.plot(ground[:, 0], ground[:, 1], color=_INK_SECONDARY, linewidth=0.5, zorder=8)
    axes.plot(flight[:, 0], flight[:, 1], color=_INK, linewidth=1.5, zorder=9,
              solid_capstyle="round")
    axes.scatter([flight[0, 0]], [flight[0, 1]], s=16.0, color=_INK, zorder=10)
    axes.plot(limb[:, 0], limb[:, 1], color=_AXIS, linewidth=0.7, zorder=12)

    # -- words: one label, and a key in the symbols of the statement -----------------
    words = {"color": _INK_SECONDARY, "linespacing": 1.22, "zorder": 20}
    axes.annotate(
        "a vehicle at\nentry interface",
        xy=(flight[0, 0], flight[0, 1]), xytext=(0.86, 0.34), ha="right", va="center",
        arrowprops={"arrowstyle": "-", "color": _INK_SECONDARY, "linewidth": 0.5,
                    "shrinkA": 2.0, "shrinkB": 1.0},
        **words,
    )
    left, top, pitch = 4.62, 2.72, 0.50
    keys = (
        (Rectangle((0.0, 0.0), 0.26, 0.11, facecolor=_REACHED, alpha=0.70, edgecolor="none"),
         r"$\mathcal{R}_\varepsilon$: what the vehicle" "\ncan reach"),
        (Line2D([0.0, 0.26], [0.0, 0.0], color=_LIMIT, linewidth=1.1, dashes=(4.0, 2.2)),
         r"$\Pi(\mathcal{R}_{\mathcal{H}_\delta})$: what the limit" "\nsystem can reach"),
        (Line2D([0.0, 0.26], [0.0, 0.0], color=_BOUND, linewidth=1.6),
         r"the same, thickened by" "\n" r"the residual $\varrho$: the bound"),
    )
    for row, (mark, text) in enumerate(keys):
        y = top - row * pitch                         # the first line of the row's text
        if isinstance(mark, Rectangle):
            mark.set_xy((left, y - 0.055))
            axes.add_patch(mark)
        else:
            mark.set_data([left, left + 0.26], [y, y])
            axes.add_line(mark)
        axes.text(left + 0.36, y + 0.065, text, ha="left", va="top", **words)
    axes.text(left, top - 3 * pitch - 0.10,
              "The three sets are drawn to\nshow the statement; they\nare not computed.",
              ha="left", va="top", style="italic", **words)

    _save(figure, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the proposal's figures")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument(
        "--preview", action="store_true", help="also write a PNG beside each PDF, to look at"
    )
    args = parser.parse_args()

    _use_document_fonts()
    profile = skip_profile()
    drawings = (("skip-entry.pdf", draw_skip_entry), ("reachable-set.pdf", draw_target))
    for name, draw in drawings:
        target = args.output / name
        draw(profile, target)
        if args.preview:
            plt.rcParams["savefig.dpi"] = 200
            draw(profile, target.with_suffix(".png"))
        print(f"wrote {target}")
    print(
        f"  the flight: energy lost {profile.energy_lost_mj:.2f} MJ/kg, and the drag power "
        f"integrates to it within {profile.energy_residual:.1e}; passes at "
        + ", ".join(f"{start:.0f}-{end:.0f} s" for start, end in profile.passes)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
