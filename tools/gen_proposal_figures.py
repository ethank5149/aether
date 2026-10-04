#!/usr/bin/env python3
"""Generate the proposal's figures: ``python tools/gen_proposal_figures.py``.

Four figures. One evaluates a result, the range bound; the others are
illustrations.

**concentration.pdf.** What the thin-atmosphere limit does to the dissipation.
One entry of a model vehicle is flown through an exponential atmosphere
whose scale height is halved three times about a fixed altitude, at a fixed
entry angle. On the left, the drag power of each flight, normalized to unit
mass: the passes sharpen into separate peaks at times that converge. On the
right, the share of the energy dissipated up to each time: the curves close
on a staircase, a jump for each pass, followed by a ramp, which is the
glide. The script checks, for each flight, that the curve it plots is the
dissipation.

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

**range-bound.pdf.** The one result the proposal claims, evaluated. The
whole-Earth view of the next figure, enlarged about the end of the entry so
that the globe runs out of the frame: the flight coming in, the limb across
one side, and around the point below the vehicle two circles. The solid one
is the range bound of the proposal, from the state at which the flight last
descends through 40 km: no bank history that stays below that altitude lands
outside it. The dashed one is how far a sweep of bank histories lands from
the same state. The solid radius is the formula, evaluated here in a few
lines of arithmetic, with its hypothesis on the air checked against the
standard atmosphere; the dashed one is a simulation and is named as one.

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
from functools import lru_cache
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
    EARTH_WGS84,
    FT,
    NOSE_RADIUS,
    TARGET_PUBLISHED,
    HypersonicAerodynamics,
    flown_model,
    fly,
    guidance_model,
    last_descent_through,
    orion_aerodynamics,
)
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import PathPatch, Rectangle  # noqa: E402

from aether.certification.rotating_field import (  # noqa: E402
    EARTH,
    jacobi_energy,
    surface_radius,
)
from aether.geodesy import GeodeticPosition  # noqa: E402
from aether.guidance.entry_ocp import EntryPathLimits  # noqa: E402
from aether.guidance.footprint import entry_footprint  # noqa: E402
from aether.guidance.skip import SkipEntryResult  # noqa: E402
from aether.symbolic.atmosphere import ExponentialAtmosphere, standard_atmosphere  # noqa: E402
from aether.symbolic.trim import entry_system  # noqa: E402

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

#: The model entry of the concentration figure. A lift-to-drag ratio of one
#: skips a few times and then glides, so both kinds of motion are in the
#: picture. Units: kg/m^2, -, m/s, degrees, m.
_MODEL_VEHICLE = {"beta": 300.0, "E": 1.0}
_MODEL_ENTRY = {"speed": 7600.0, "flight_path_angle": -4.0, "altitude": 120.0e3}
#: The scale heights (m) it is flown at, the terrestrial one halved three
#: times, and the altitude (m) at which the four atmospheres share a density.
_SCALE_HEIGHTS = (7200.0, 3600.0, 1800.0, 900.0)
_ANCHOR = 60.0e3
#: The four flights are an ordered family, so they take one hue in four
#: steps, light for the thickest atmosphere (an ordinal ramp).
_THINNING = ("#86b6ef", "#5598e7", "#256abf", "#104281")

#: The corridor the range bound is evaluated on: its ceiling (m), and a scale
#: height (m) no less than the standard density's anywhere under that
#: ceiling, so that the air thickens downward at least as fast as an
#: exponential of that height. The bound grows like the reciprocal of the
#: density at the ceiling, so the ceiling is where it is asked from: at 40 km
#: it is a few hundred kilometres, and the two ceilings of ``_HIGHER`` are
#: evaluated as well, to say how fast it grows.
_CEILING = 40.0e3
_SCALE_HEIGHT = 7.0e3
_HIGHER = ((45.0e3, 7.3e3), (50.0e3, 8.1e3))
#: The view the range bound is drawn in: the whole-Earth view enlarged this
#: many times, with the point below the vehicle put here on the page (inches).
#: The globe then runs out of the frame, and its limb crosses one corner.
_ENLARGED = 2.3
_VEHICLE_AT = (3.80, 1.85)
#: How far below the standard the air may be; and gravity (m/s^2), rounded
#: up, so that a drop in potential is a drop in altitude of at least that
#: drop over gravity.
_DENSITY_MARGIN = 0.85
_GRAVITY = 9.9

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


@lru_cache(maxsize=1)
def _flight() -> tuple[HypersonicAerodynamics, SkipEntryResult]:
    """The example's flight, flown once for every figure that draws it."""
    aero = orion_aerodynamics()
    result, _guidance = fly(aero)
    if result.plant_states is None:
        raise RuntimeError("the example did not fly a plant that reports its own outputs")
    return aero, result


def skip_profile() -> SkipProfile:
    """Reduce the example's flight to what the figures draw of it."""
    aero, result = _flight()
    assert result.plant_states is not None
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


@dataclass(frozen=True)
class Thinning:
    """One entry flown through atmospheres of shrinking scale height, as the figure draws it."""

    ratios: tuple[int, ...]
    """How many times thinner than the terrestrial scale height each flight's is."""
    times: tuple[np.ndarray, ...]
    density: tuple[np.ndarray, ...]
    """Drag power normalized to unit mass (1/s): the dissipation density in time."""
    share: tuple[np.ndarray, ...]
    """Fraction of the flight's dissipation that has occurred by each time."""
    energy_residual: float
    """The worst relative difference, over the flights, between the integral of
    the drag power and the mechanical energy lost."""


def thinning() -> Thinning:
    """Fly the model entry at each scale height, on the compiled symbolic field.

    The planet is a sphere that does not rotate, so the motion is planar, and
    the atmospheres agree at the anchor altitude, where the terrestrial
    exponential atmosphere puts them.
    """
    anchor_density = 1.225 * float(np.exp(-_ANCHOR / _SCALE_HEIGHTS[0]))
    start = np.array([
        EARTH.radius + _MODEL_ENTRY["altitude"], 0.0, 0.0, _MODEL_ENTRY["speed"],
        np.deg2rad(_MODEL_ENTRY["flight_path_angle"]), 0.5 * np.pi, 0.0, 0.0,
    ])
    times, density, share, residuals = [], [], [], []
    for scale_height in _SCALE_HEIGHTS:
        plant = entry_system(
            ExponentialAtmosphere(anchor_density, _ANCHOR, scale_height),
            name=f"model_entry_{scale_height:.0f}",
        ).compile(jacobian=False)
        parameters = plant.parameter_vector(
            sigma=0.0, k_rho=1.0, mu=EARTH.mu, R_e=EARTH.radius, J_2=0.0, omega_E=0.0,
            f_E=0.0, R_n=1.0, k_Q=0.0, **_MODEL_VEHICLE,
        )
        # A pass is as short as the atmosphere is thin, and the step follows it.
        step = 0.02 * scale_height / _SCALE_HEIGHTS[0]
        clock, rows = plant.trajectory(start, parameters, 1400.0, step=step,
                                       stride=round(0.25 / step))
        clock = np.concatenate([[0.0], clock])
        rows = np.vstack([start, rows])
        along = plant.output_table(rows, parameters)
        flying = np.flatnonzero((along["altitude"] <= 0.0) | (rows[:, 3] <= 300.0))
        end = int(flying[0]) if flying.size else len(clock)
        clock, power, energy = clock[:end], along["drag_power"][:end], along["energy"][:end]
        steps = 0.5 * (power[1:] + power[:-1]) * np.diff(clock)
        dissipated = float(steps.sum())
        lost = float(energy[0] - energy[-1])
        residuals.append(abs(dissipated - lost) / lost)
        times.append(clock)
        density.append(power / dissipated)
        share.append(np.concatenate([[0.0], np.cumsum(steps)]) / dissipated)
    worst = max(residuals)
    if worst > _ENERGY_TOLERANCE:
        raise RuntimeError(
            f"the drag power of a model flight integrates to its energy loss only within "
            f"{worst:.1e}: the curves plotted are not the dissipation"
        )
    return Thinning(
        ratios=tuple(round(_SCALE_HEIGHTS[0] / scale) for scale in _SCALE_HEIGHTS),
        times=tuple(times), density=tuple(density), share=tuple(share),
        energy_residual=worst,
    )


def draw_concentration(profile: Thinning, path: Path) -> None:
    """The dissipation density at each scale height, and beside it the share dissipated.

    Small multiples on the left, one row for each flight on a common time
    axis, so that the eye runs down a column and sees the peaks sharpen in
    place; one panel on the right, where the same four flights close on a
    staircase.
    """
    figure = plt.figure(figsize=(6.5, 3.3))
    grid = figure.add_gridspec(len(profile.ratios), 2, width_ratios=(1.08, 1.0),
                               wspace=0.30, hspace=0.22)
    horizon = max(float(clock[-1]) for clock in profile.times)

    def quiet(axes: plt.Axes) -> None:
        for side in ("top", "right", "left"):
            axes.spines[side].set_visible(False)
        axes.spines["bottom"].set_color(_AXIS)
        axes.spines["bottom"].set_linewidth(0.6)
        axes.tick_params(colors=_INK_SECONDARY, width=0.6, length=3.0)
        axes.set_xlim(0.0, horizon)

    rows = []
    for row, (ratio, clock, density, colour) in enumerate(
        zip(profile.ratios, profile.times, profile.density, _THINNING, strict=True)
    ):
        axes = figure.add_subplot(grid[row, 0])
        rows.append(axes)
        quiet(axes)
        peak = float(density.max())
        axes.fill_between(clock, density / peak, color=colour, alpha=0.30, linewidth=0.0)
        axes.plot(clock, density / peak, color=colour, linewidth=1.1)
        axes.set_ylim(0.0, 1.12)
        axes.set_yticks([])
        if row < len(profile.ratios) - 1:
            axes.tick_params(axis="x", labelbottom=False, length=0.0)
        name = r"$\varepsilon$" if ratio == 1 else rf"$\varepsilon/{ratio}$"
        axes.text(0.985, 0.80, name, ha="right", va="center", color=_INK_SECONDARY,
                  transform=axes.transAxes)
    rows[0].set_title("dissipation density", loc="left", fontsize=9.0, color=_INK, pad=4.0)
    rows[-1].set_xlabel("time from entry interface (s)", color=_INK)

    stairs = figure.add_subplot(grid[:, 1])
    quiet(stairs)
    stairs.grid(axis="y", color=_GRID, linewidth=0.5)
    stairs.set_axisbelow(True)
    for clock, share, colour in zip(profile.times, profile.share, _THINNING, strict=True):
        stairs.plot(clock, share, color=colour, linewidth=1.3, solid_capstyle="round")
    stairs.set_ylim(0.0, 1.03)
    stairs.set_yticks([0.0, 0.5, 1.0])
    stairs.set_yticklabels(["0", "50%", "100%"])
    stairs.tick_params(axis="y", length=0.0)
    stairs.set_title("share of the energy dissipated", loc="left", fontsize=9.0, color=_INK,
                     pad=4.0)
    stairs.set_xlabel("time from entry interface (s)", color=_INK)

    # Two labels, tied to the thinnest atmosphere's curve: what a jump is and
    # what the ramp is.
    leader = {"arrowstyle": "-", "color": _INK_SECONDARY, "linewidth": 0.5,
              "shrinkA": 2.0, "shrinkB": 1.0}
    clock, share = profile.times[-1], profile.share[-1]
    rise = int(np.argmax(share > 0.5 * share[np.searchsorted(clock, 0.25 * horizon)]))
    stairs.annotate("a jump at\neach pass", xy=(clock[rise], share[rise]),
                    xytext=(0.30 * horizon, 0.07), ha="left", va="center",
                    color=_INK_SECONDARY, linespacing=1.22, arrowprops=leader)
    late = int(np.searchsorted(clock, 0.80 * horizon))
    stairs.annotate("a ramp on\nthe glide", xy=(clock[late], share[late]),
                    xytext=(0.97 * horizon, 0.62), ha="right", va="center",
                    color=_INK_SECONDARY, linespacing=1.22, arrowprops=leader)

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


@dataclass(frozen=True)
class _Canvas:
    """A figure with the globe drawn on it, and the map from the camera's image to inches."""

    figure: plt.Figure
    axes: plt.Axes
    camera: _Camera
    middle: np.ndarray
    scale: float
    disc: PathPatch

    def inches(self, image: np.ndarray) -> np.ndarray:
        return self.middle + self.scale * np.asarray(image)

    def ground(self, directions: np.ndarray) -> np.ndarray:
        return self.inches(self.camera.ground(directions))

    def fill(self, boundary: np.ndarray, **style: object) -> None:
        """Fill the visible part of a region on the ground."""
        filled = self.camera.region(boundary)
        if filled is not None:
            filled = self.inches(filled)
            self.axes.fill(filled[:, 0], filled[:, 1], linewidth=0.0, clip_path=self.disc,
                           **style)

    def stroke(self, curve: np.ndarray, **style: object) -> None:
        """Draw the visible stretches of a curve on the ground."""
        for run in self.camera.runs(curve):
            run = self.inches(run)
            self.axes.plot(run[:, 0], run[:, 1], solid_capstyle="round", **style)


def _globe(
    width: float, height: float, middle: tuple[float, float], radius: float,
    camera: _Camera | None = None, every: int = 15,
) -> _Canvas:
    """A figure of the given size in inches with the shaded globe on it.

    ``camera`` is the one of the whole-Earth view unless another is given, and
    ``every`` is the spacing of the graticule in degrees.
    """
    if camera is None:
        camera = _Camera.above(*_CAMERA)
    centre = np.array(middle)
    scale = radius / camera.limb_radius
    figure = plt.figure(figsize=(width, height))
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axes.set_aspect("equal")
    axes.axis("off")

    turn = np.linspace(0.0, 2.0 * np.pi, 361)
    limb = centre + radius * np.column_stack([np.cos(turn), np.sin(turn)])
    axes.fill(limb[:, 0], limb[:, 1], color="white", linewidth=0.0, zorder=0)
    disc = PathPatch(mpath.Path(limb), facecolor="none", edgecolor="none")
    axes.add_patch(disc)
    canvas = _Canvas(figure, axes, camera, centre, scale, disc)

    vertices: list[np.ndarray] = []
    codes: list[int] = []
    for ring in _land():
        outline = camera.region(ring)
        if outline is not None:
            vertices.append(canvas.inches(outline))
            codes += [mpath.Path.MOVETO, *[mpath.Path.LINETO] * (len(outline) - 2),
                      mpath.Path.CLOSEPOLY]
    axes.add_patch(PathPatch(
        mpath.Path(np.vstack(vertices), codes), facecolor=_LAND_FILL, edgecolor=_COAST,
        linewidth=0.35, zorder=1, clip_path=disc,
    ))
    span = np.linspace(-180.0, 180.0, 721)
    lines = [np.column_stack([np.full(361, longitude), span[::2] / 2.0])
             for longitude in range(-180, 180, every)]
    lines += [np.column_stack([span, np.full(721, latitude)])
              for latitude in range(-90 + every, 90, every)]
    for line in lines:
        for run in camera.runs(_directions(line)):
            run = canvas.inches(run)
            axes.plot(run[:, 0], run[:, 1], color=_GRID, linewidth=0.4, zorder=2, clip_path=disc)
    # The raster covers the disc and nothing else.
    box = 1.01 * camera.limb_radius
    pixels = int(2.02 * radius * 140)
    axes.imshow(
        camera.shading((-box, box, -box, box), (pixels, pixels)), interpolation="bilinear",
        extent=(centre[0] - scale * box, centre[0] + scale * box,
                centre[1] - scale * box, centre[1] + scale * box),
        zorder=2.5, clip_path=disc,
    )
    axes.plot(limb[:, 0], limb[:, 1], color=_AXIS, linewidth=0.7, zorder=12)
    axes.set_xlim(0.0, width)
    axes.set_ylim(0.0, height)
    return canvas


def _draw_flight(
    canvas: _Canvas, profile: SkipProfile, exaggeration: float = _EXAGGERATION
) -> tuple[np.ndarray, np.ndarray]:
    """The flight above the globe, its altitude exaggerated, with a post each minute.

    Only the stretch over the visible side of the globe is drawn. Returns the
    ground track and the trajectory of every sample, in inches; those of a
    sample that cannot be seen mean nothing.
    """
    axes = canvas.axes
    over = _directions(profile.track)
    seen = np.flatnonzero(canvas.camera.visible(over))
    first, last = int(seen[0]), int(seen[-1]) + 1
    with np.errstate(divide="ignore", invalid="ignore"):
        ground = canvas.ground(over)
        flight = canvas.inches(canvas.camera.project(
            (_GLOBE_RADIUS + exaggeration * profile.altitudes_km)[:, None] * over
        ))
    shown = slice(first, last)
    axes.fill(np.r_[ground[shown, 0], flight[shown, 0][::-1]],
              np.r_[ground[shown, 1], flight[shown, 1][::-1]],
              color=_INK, alpha=0.07, linewidth=0.0, zorder=7)
    for moment in np.arange(0.0, profile.times[-1], _POST_EVERY):
        k = int(np.searchsorted(profile.times, moment))
        if first <= k < last:
            axes.plot([ground[k, 0], flight[k, 0]], [ground[k, 1], flight[k, 1]],
                      color=_INK_SECONDARY, linewidth=0.3, zorder=8)
    axes.plot(ground[shown, 0], ground[shown, 1], color=_INK_SECONDARY, linewidth=0.5, zorder=8)
    axes.plot(flight[shown, 0], flight[shown, 1], color=_INK, linewidth=1.5, zorder=9,
              solid_capstyle="round")
    if first == 0:
        axes.scatter([flight[0, 0]], [flight[0, 1]], s=16.0, color=_INK, zorder=10)
    return ground, flight


_LEADER = {"arrowstyle": "-", "color": _INK_SECONDARY, "linewidth": 0.5,
           "shrinkA": 2.0, "shrinkB": 1.0}
_WORDS = {"color": _INK_SECONDARY, "linespacing": 1.22, "zorder": 20}


def draw_target(profile: SkipProfile, path: Path) -> None:
    """The statement aimed at, on the ground under a skip entry.

    One hue in three steps for the three nested sets; everything else is ink
    and grey. The flight is the example's. The three sets are an
    illustration, and the figure says so.
    """
    canvas = _globe(6.5, 3.3, (2.72, 1.65), 1.6)
    axes = canvas.axes

    # -- on the ground: the three sets of the statement, drawn and not computed ------
    vehicle, limit, thickened = (
        _along_great_circles(curve, 0.5, close=True)
        for curve in _illustrative_sets(profile.track)
    )
    canvas.fill(thickened, color=_BAND, alpha=0.55, zorder=3)
    canvas.fill(vehicle, color=_REACHED, alpha=0.70, zorder=4)
    canvas.stroke(limit, color=_LIMIT, linewidth=1.1, zorder=6, dashes=(4.0, 2.2))
    canvas.stroke(thickened, color=_BOUND, linewidth=1.6, zorder=6)

    _ground, flight = _draw_flight(canvas, profile)

    # -- words: one label, and a key in the symbols of the statement -----------------
    axes.annotate(
        "a vehicle at\nentry interface",
        xy=(flight[0, 0], flight[0, 1]), xytext=(0.86, 0.34), ha="right", va="center",
        arrowprops=_LEADER, **_WORDS,
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
        axes.text(left + 0.36, y + 0.065, text, ha="left", va="top", **_WORDS)
    axes.text(left, top - 3 * pitch - 0.10,
              "The three sets are drawn to\nshow the statement; they\nare not computed.",
              ha="left", va="top", style="italic", **_WORDS)

    _save(canvas.figure, path)


@dataclass(frozen=True)
class RangeBound:
    """The range bound evaluated on the example, and the sweep it is compared with."""

    start: int
    """Sample of the flight at which it last descends through the ceiling."""
    ceiling_km: float
    speed: float
    """m/s there: the bound's V_0."""
    terminal_speed: float
    ballistic_coefficient: float
    density: float
    """kg/m^3: the least the air is taken to be anywhere under the ceiling."""
    scale_height: float
    """m: under the ceiling the air thickens at least as fast as an exponential of this height."""
    gravity: float
    """m/s^2, rounded up: with the scale height, the potential the air e-folds across."""
    thinnest: float
    """The fraction of the standard density at which the bound's hypothesis still holds."""
    bound_km: float
    """The formula: no ground track from there is longer."""
    hull: np.ndarray
    samples: np.ndarray
    """Where the sweep's bank histories land, as (longitude, latitude) in degrees."""
    hull_local: np.ndarray
    samples_local: np.ndarray
    track_local: np.ndarray
    target_local: tuple[float, float]
    """The same, the flown track and the site as (downrange, crossrange) in km from there."""
    reach_km: float
    """Farthest downrange the sweep lands."""
    rejected: int


def _least_density_the_bound_allows(
    ceiling: float, density: float, scale_height: float
) -> float:
    """How thin, as a fraction of the standard, the air may be for the bound to hold.

    The bound assumes that under the ceiling the density is at least
    ``density`` times the exponential of the potential's drop below its
    greatest value there, over gravity times the scale height. This holds the
    standard atmosphere against that floor, with the potential of the field
    the example is flown in, at every latitude and every altitude under the
    ceiling, and returns the largest ratio of floor to standard.
    """
    atmosphere = standard_atmosphere()
    latitudes = np.deg2rad(np.linspace(-90.0, 90.0, 181))
    altitudes = np.concatenate([
        np.linspace(0.0, ceiling - 4.0e3, 93), np.linspace(ceiling - 4.0e3, ceiling, 81)
    ])
    surface = surface_radius(latitudes, planet=EARTH_WGS84)

    def potential(altitude: float) -> np.ndarray:
        at_rest = [surface + altitude, 0.0, latitudes, 0.0, 0.0, 0.0]
        return np.asarray(jacobi_energy(at_rest, planet=EARTH_WGS84))

    highest = float(potential(ceiling).max())
    fold = _GRAVITY * scale_height
    worst = 0.0
    for altitude in altitudes:
        floor = density * np.exp((highest - potential(float(altitude))) / fold)
        worst = max(worst, float(floor.max()) / float(atmosphere.density(float(altitude))))
    return worst


def _evaluate_bound(ceiling: float, scale_height: float) -> dict[str, float]:
    """The closed-form bound from the flight's last descent through ``ceiling``.

    Arithmetic on six numbers, after checking the hypothesis on the air
    against the atmosphere and the potential the example is flown in.
    """
    aero, result = _flight()
    start = last_descent_through(result, ceiling)
    speed = float(result.states[3, start])
    terminal = 1000.0 * FT
    beta = float(aero.ballistic_coefficient)
    density = _DENSITY_MARGIN * float(standard_atmosphere().density(float(ceiling)))
    thinnest = _least_density_the_bound_allows(ceiling, density, scale_height)
    if thinnest > 1.0:
        raise RuntimeError(
            f"under {ceiling / 1e3:.0f} km the standard density falls below the floor the "
            f"bound assumes, by a factor of {thinnest:.3f} somewhere"
        )
    bound = (2.0 * beta / density) * (
        np.log(speed / terminal) + _GRAVITY * scale_height / terminal**2
    )
    return {
        "start": float(start), "speed": speed, "terminal": terminal, "beta": beta,
        "density": density, "thinnest": thinnest, "bound_km": float(bound) / 1e3,
    }


def range_bound(ceiling: float = _CEILING, scale_height: float = _SCALE_HEIGHT) -> RangeBound:
    """Evaluate the closed-form bound from the flight's last descent through ``ceiling``.

    The bound is arithmetic on six numbers. The sweep beside it is the
    example's footprint from the same state: a simulation, for comparison,
    and every history in it is held to the bound's hypothesis, that it stays
    under the ceiling.
    """
    aero, result = _flight()
    numbers = _evaluate_bound(ceiling, scale_height)
    start = int(numbers["start"])
    terminal = numbers["terminal"]

    model = guidance_model(aero)
    # Assumed limits, the example's own: loose on heating, 10 g on load.
    limits = EntryPathLimits(
        max_heat_flux_w_cm2=5000.0, max_g_load=10.0, nose_radius_m=NOSE_RADIUS
    )
    footprint = entry_footprint(
        result.states[:, start], model, limits=limits, terminal_speed=terminal,
        label="descent", epoch=float(result.times[start]), max_time=2000.0, step=10.0,
        n_constant=67, n_reversals=8,
    )

    state = result.states[:, start]
    highest = max(
        float(flown.altitudes.max())
        for flown in (
            model.fly_schedule(state, schedule, terminal_speed=terminal, max_time=2000.0,
                               step=10.0)
            for schedule in footprint.schedules
        )
        if flown is not None
    )
    if highest > ceiling + 1.0:
        raise RuntimeError(
            f"a history of the sweep climbs to {highest / 1e3:.2f} km, over the ceiling the "
            f"bound is conditional on"
        )

    def degrees(points: tuple[GeodeticPosition, ...]) -> np.ndarray:
        return np.rad2deg([[point.longitude, point.latitude] for point in points])

    def local(points: tuple[GeodeticPosition, ...]) -> np.ndarray:
        return np.array([footprint.local(point) for point in points]) / 1e3

    site = GeodeticPosition(latitude=TARGET_PUBLISHED[1], longitude=TARGET_PUBLISHED[0])
    track = tuple(model.ground_point(sample) for sample in result.states[:, start:].T)
    target = local((site,))[0]
    return RangeBound(
        start=start, ceiling_km=ceiling / 1e3, speed=numbers["speed"],
        terminal_speed=terminal, ballistic_coefficient=numbers["beta"],
        density=numbers["density"], scale_height=scale_height, gravity=_GRAVITY,
        thinnest=numbers["thinnest"], bound_km=numbers["bound_km"],
        hull=degrees(footprint.boundary), samples=degrees(footprint.samples),
        hull_local=local(footprint.boundary), samples_local=local(footprint.samples),
        track_local=local(track), target_local=(float(target[0]), float(target[1])),
        reach_km=footprint.max_downrange_m / 1e3, rejected=footprint.rejected,
    )


def draw_range_bound(profile: SkipProfile, bound: RangeBound, path: Path) -> None:
    """What is proved against what is found, from one state of the example flight.

    One picture: the whole-Earth view of the other figure, enlarged about the
    end of the entry until the circle of the bound and the patch where the
    sweep lands can both be seen, so that the globe runs out of the frame. The
    flight comes in from below and the limb crosses a corner. Each thing is
    named beside it; there is no key. One hue: dark for the bound, lighter for
    what the sweep finds.
    """
    width, height = 6.5, 3.3
    camera = _Camera.above(*_CAMERA)
    over = _directions(profile.track)
    here = over[bound.start]
    radius = 1.6 * _ENLARGED
    offset = (radius / camera.limb_radius) * camera.ground(here[None, :])[0]
    canvas = _globe(width, height, (_VEHICLE_AT[0] - offset[0], _VEHICLE_AT[1] - offset[1]),
                    radius, camera=camera)
    axes = canvas.axes

    # -- on the ground: the circle of the bound, and inside it what the sweep finds ---
    east = np.cross([0.0, 0.0, 1.0], here)
    east /= np.linalg.norm(east)
    north = np.cross(here, east)
    turn = np.linspace(0.0, 2.0 * np.pi, 361)

    def ring(kilometres: float) -> np.ndarray:
        angle = kilometres / _GLOBE_RADIUS
        return np.cos(angle) * here + np.sin(angle) * (
            np.outer(np.cos(turn), east) + np.outer(np.sin(turn), north)
        )

    proved, found = ring(bound.bound_km), ring(bound.reach_km)
    canvas.fill(proved, color=_BAND, alpha=0.45, zorder=3)
    canvas.stroke(proved, color=_BOUND, linewidth=1.8, zorder=6)
    hull = _along_great_circles(_directions(bound.hull), 0.05, close=True)
    canvas.fill(hull, color=_REACHED, zorder=5)
    canvas.stroke(found, color=_LIMIT, linewidth=0.9, zorder=6, dashes=(3.0, 1.6))

    ground, flight = _draw_flight(canvas, profile)
    axes.scatter([flight[bound.start, 0]], [flight[bound.start, 1]], s=18.0, color=_INK,
                 zorder=10)

    # -- names: a term, and under it the quantity; each ruled to what it names -------
    outer, inner = canvas.ground(proved), canvas.ground(found)
    notes = (
        ("Proved bound", f"radius {round(bound.bound_km, -1):,.0f} km",
         outer[np.argmin(outer[:, 0] - 0.9 * outer[:, 1])], (0.16, 2.92)),
        ("Vehicle", f"{bound.ceiling_km:.0f} km altitude, {bound.speed / 1e3:.2f} km/s",
         flight[bound.start], (0.16, 2.08)),
        ("Simulated landings",
         f"{len(bound.samples)} bank histories,\nall within {bound.reach_km:,.0f} km",
         inner[np.argmin(inner[:, 0] + 0.9 * inner[:, 1])], (0.16, 1.30)),
    )
    for term, quantity, target, (x, y) in notes:
        axes.annotate(term, xy=(target[0], target[1]), xytext=(x, y), ha="left", va="center",
                      color=_INK, zorder=20, arrowprops=_LEADER)
        axes.text(x, y - 0.095, quantity, ha="left", va="top", fontsize=8.0, **_WORDS)
    k = int(np.searchsorted(profile.times, profile.times[bound.start] - 170.0))
    axes.text(ground[k, 0] + 0.13, ground[k, 1] + 0.08, "Entry trajectory", ha="left",
              va="center", color=_INK, zorder=20)
    axes.text(ground[k, 0] + 0.13, ground[k, 1] - 0.085,
              f"altitude $\\times${_EXAGGERATION:.0f}", ha="left", va="center", fontsize=8.0,
              **_WORDS)
    axes.add_patch(Rectangle((0.0, 0.0), width, height, facecolor="none", edgecolor=_AXIS,
                             linewidth=1.2, zorder=30))

    _save(canvas.figure, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the proposal's figures")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument(
        "--preview", action="store_true", help="also write a PNG beside each PDF, to look at"
    )
    args = parser.parse_args()

    _use_document_fonts()
    if args.preview:
        plt.rcParams["savefig.dpi"] = 200

    sweep = thinning()
    target = args.output / "concentration.pdf"
    draw_concentration(sweep, target)
    if args.preview:
        draw_concentration(sweep, target.with_suffix(".png"))
    print(f"wrote {target}")
    print(
        "  the model entry at scale heights of "
        + ", ".join(f"{scale / 1e3:g}" for scale in _SCALE_HEIGHTS)
        + f" km: each drag power integrates to the energy lost within "
        f"{sweep.energy_residual:.1e}"
    )

    profile = skip_profile()
    drawings = (("skip-entry.pdf", draw_skip_entry), ("reachable-set.pdf", draw_target))
    for name, draw in drawings:
        target = args.output / name
        draw(profile, target)
        if args.preview:
            draw(profile, target.with_suffix(".png"))
        print(f"wrote {target}")

    bound = range_bound()
    target = args.output / "range-bound.pdf"
    draw_range_bound(profile, bound, target)
    if args.preview:
        draw_range_bound(profile, bound, target.with_suffix(".png"))
    print(f"wrote {target}")
    print(
        f"  from the last descent through {bound.ceiling_km:.0f} km, at {bound.speed:,.1f} m/s, "
        f"down to {bound.terminal_speed:.1f} m/s:\n"
        f"  (2 beta / rho_c) [ln(V_0/V_f) + g H/V_f^2] = "
        f"(2 x {bound.ballistic_coefficient:.1f} / {bound.density:.4e}) x "
        f"[{np.log(bound.speed / bound.terminal_speed):.3f} + "
        f"{bound.gravity * bound.scale_height / bound.terminal_speed**2:.3f}] = "
        f"{bound.bound_km:,.0f} km\n"
        f"  its floor on the density holds for any atmosphere at least "
        f"{bound.thinnest:.3f} of the standard\n"
        f"  the sweep lands within {bound.reach_km:,.0f} km ({len(bound.samples)} bank "
        f"histories, {bound.rejected} over the load limit, every one under the ceiling): "
        f"{bound.bound_km / bound.reach_km:.1f} times less"
    )
    for ceiling, scale_height in _HIGHER:
        higher = _evaluate_bound(ceiling, scale_height)
        print(
            f"  from {ceiling / 1e3:.0f} km instead, at {higher['speed']:,.1f} m/s, with "
            f"H = {scale_height / 1e3:.1f} km: {higher['bound_km']:,.0f} km (floor holds to "
            f"{higher['thinnest']:.3f} of the standard)"
        )
    print(
        f"  the flight: energy lost {profile.energy_lost_mj:.2f} MJ/kg, and the drag power "
        f"integrates to it within {profile.energy_residual:.1e}; passes at "
        + ", ".join(f"{start:.0f}-{end:.0f} s" for start, end in profile.passes)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
