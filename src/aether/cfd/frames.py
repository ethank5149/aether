"""Pictures of a run while it is still running.

A hypersonic case reports itself as ten residual curves and a line saying "N
points are not physical". Both are true and neither is legible: on the Mach 20
sphere-cone in this repository the species residuals sat between -2.2 and -3.8
and looked healthy while the energy equation stood at +5.3 and the solver was
clamping six hundred points an iteration. The summary line said "converging"
throughout.

A picture of the field says it immediately. The same case rendered as Mach
number shows a mottle of saturated cells around the nose where a converged run
shows a clean bow shock, and no one has to read a legend to see it.

This module turns the volume snapshot SU2 is already writing into one, and does
it where the work belongs -- in the worker, beside the solve -- so the panel
stays a thing that reads files.

What is rendered, and why not Schlieren
---------------------------------------

Mach number, on levels fixed by the flight condition rather than by the
solution's own extremes. Two consequences, both wanted:

* Frames are comparable. Exposure set per frame holds the shock at constant
  brightness while everything around it pulses, which reads as the solution
  changing when it is only the scale changing. The notebook's sweep animation
  learned this the same way.
* Divergence saturates. Anything above the freestream Mach is off the top of
  the scale and appears as flat white, so an unphysical region announces itself
  rather than quietly rescaling the whole picture around itself.

Numerical Schlieren is the better *picture* and is what the four-panel figure
uses, but it costs about five seconds a frame against a third of a second here,
almost all of it in the density-gradient pass. That is the wrong trade for
something rendered every snapshot for the length of a run.

The sidecar
-----------

Each frame is written with a small JSON file beside it holding the iteration it
belongs to and :func:`~aether.viz.flowfield.hot_spots`' verdict on that
field. The expensive part is reading an 18 MB mesh and a 47 MB solution; having
paid it, the numbers that reading supports are recorded too, so the panel can
say *where* the field is unphysical without opening either file.
"""

from __future__ import annotations

import contextlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "Frame",
    "FrameWriter",
    "completed_snapshots",
    "field_size",
    "frame_dir",
    "frames",
    "latest_frame",
    "render_frame",
    "snapshot_path",
]

#: Retired as the frame size: it is kept only because callers import it. The
#: canvas comes from :func:`field_size` now -- see the argument there for why a
#: fixed size and ``aspect="equal"`` cannot both be had.
FIGSIZE = (7.0, 3.4)
DPI = 120

#: How much of the domain around the body to keep, as a fraction of its length.
#: The farfield is tens of body lengths across and mostly freestream; framed to
#: it the vehicle is a speck, which is what the first attempt at this produced.
MARGIN = 0.45

#: How far past the domain to open the view, in multiples of the nose standoff.
#: Clipped exactly to a shock-fitted block the frame is wall-to-wall shock layer
#: and the outer boundary lies along the frame edge, where it looks like a
#: property of the picture. Some undisturbed freestream on each side is what
#: identifies it as a boundary of the *mesh*.
STANDOFF_PADDING = 2.5

#: Whether to draw the leading-edge inset. **Off.** It was there because a
#: 0.7 mm shock on a 2 m body is a sixth of a pixel at whole-body scale, and
#: that is still true -- but the flow a hypersonic CFD run is being read for
#: happens downfield of the nose, and the inset spends a corner of every frame
#: on the one place the whole-body view is least informative about. Kept as a
#: parameter because an ablation run, which is about the stagnation region and
#: nothing else, wants the opposite default.
DETAIL_INSET = False

#: Seconds a snapshot must have been untouched before it is read. SU2 spends
#: about a second writing forty-five megabytes, and reading into that write
#: fails in the parser rather than politely returning nothing.
QUIESCENT = 2.0

#: How many times one unchanged snapshot is retried before it is written off.
#: A torn read succeeds on the next pass; a file that never parses is a fault
#: worth seeing once rather than every three seconds for the rest of the run.
MAX_ATTEMPTS = 3


#: ``volume_flow_000250.dat`` -- SU2's ``WRT_VOLUME_OVERWRITE= NO`` naming,
#: six digits, zero padded, from ``CConfig::GetFilename_Iter``.
NUMBERED = re.compile(r"^volume_flow_(\d+)\.dat$")


def snapshot_path(case_dir: Path | str) -> Path:
    """The volume file SU2 rewrites every ``volume_every`` iterations.

    The overwrite-mode name, kept for cases solved before numbering was turned
    on and as the fallback when no numbered snapshot has appeared yet.
    """
    return Path(case_dir) / "volume_flow.dat"


def completed_snapshots(case_dir: Path | str) -> list[tuple[int, Path]]:
    """Numbered snapshots that are **provably finished**, oldest first.

    A snapshot is finished when a higher-numbered one exists: SU2 does not
    begin the next dump until the previous is closed. That is an exact test,
    and it replaces a stack of heuristics that could not be made to work.

    The file is overwritten by default, so the previous version of this waited
    for the same path to stop changing -- quiescent for two seconds, and the
    same size on two consecutive polls. On a run producing 35 MB every four
    seconds the window never opened: measured over eight polls, the size read
    35592892, 35592409, 35592409, 20691703, 35593032, 12315160, 35592985,
    12835166 -- the three short reads being mid-write. **Not one frame was
    rendered for the entire run**, and nothing said so, because a torn read is
    swallowed by design and looks exactly like "nothing new yet".

    Asking SU2 to number them costs nothing and removes the race rather than
    narrowing it. :func:`FrameWriter.poll` prunes what it has passed, so the
    disk holds two.
    """
    case = Path(case_dir)
    found: list[tuple[int, Path]] = []
    for path in case.glob("volume_flow_*.dat"):
        match = NUMBERED.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    found.sort()
    # The newest is the one being written, or about to be. Everything before it
    # is closed.
    return found[:-1]


def frame_dir(case_dir: Path | str) -> Path:
    return Path(case_dir) / "frames"


@dataclass(frozen=True)
class Frame:
    """One rendered frame and what was measured while rendering it."""

    iteration: int
    image: Path
    note: str = ""
    """:func:`hot_spots`' one-line verdict on this field."""

    diverged: bool = False

    @property
    def sidecar(self) -> Path:
        return self.image.with_suffix(".json")

    @property
    def schlieren(self) -> Path:
        """The Schlieren view of the same iterate."""
        return self.image.with_name(self.image.stem + "-schlieren.png")

    def view(self, name: str) -> Path:
        """``"mach"`` or ``"schlieren"``, falling back to the Mach field."""
        candidate = self.schlieren if name == "schlieren" else self.image
        return candidate if candidate.is_file() else self.image


#: Matches a frame image and **not** its Schlieren companion, which shares
#: the stem. Without the anchor, every iterate would be listed twice.
_STEM = re.compile(r"^iter_(\d+)\.png$")


def frames(case_dir: Path | str) -> list[Frame]:
    """Every frame rendered for a case, in iteration order."""
    found: list[Frame] = []
    for image in sorted(frame_dir(case_dir).glob("iter_*.png")):
        match = _STEM.match(image.name)
        if match is None:
            continue
        note, diverged = "", False
        try:
            payload = json.loads(image.with_suffix(".json").read_text())
            note = str(payload.get("note", ""))
            diverged = bool(payload.get("diverged", False))
        except (OSError, ValueError):
            pass
        found.append(Frame(int(match.group(1)), image, note, diverged))
    return found


def latest_frame(case_dir: Path | str) -> Frame | None:
    found = frames(case_dir)
    return found[-1] if found else None


def _view_box(
    cut: Any, body: Any, mesh: Any, walls: Any, margin: float
) -> tuple[float, float, float, float]:
    """The body plus a margin, **clipped to the domain that exists**.

    Neither half works alone, which is how this got broken twice:

    * Framing on the body alone -- the original behaviour -- is right for an
      isotropic box domain, whose far field is thirty metres of freestream
      nobody needs to look at. On a shock-fitted block it leaves the solution a
      thin line, because the whole domain is 11 mm deep.
    * Framing on the domain is right for the shock-fitted block and absurd for
      the box: a 39 m domain padded by half its own extent puts the 2 m vehicle
      in the middle of an empty 78 m frame.

    Taking the body's window and then clipping it to the domain gives the first
    where the domain is large and the second where it is small.

    **Then it is opened out again by** :data:`STANDOFF_PADDING`. Clipped exactly
    to the domain, a shock-fitted frame is filled edge to edge by the shock
    layer and shows nothing of what the vehicle is flying into or leaving
    behind: the outer boundary runs along the frame edge, where it reads as a
    property of the picture rather than of the mesh. A little undisturbed
    freestream on each side is what makes it legible as a boundary.

    The padding is a fraction of the **domain's own thickness**, not of the
    body, so a 45 mm hemisphere and a 2 m cone are opened out by comparable
    amounts of what is actually being looked at.
    """
    import numpy as np

    from aether.viz.flowfield import _extent

    wide = _extent(mesh, walls, margin)
    try:
        vertices = np.asarray(cut.interpolate(mesh.points))
    except Exception:
        return wide
    if vertices.size == 0:
        return wide
    x, z = vertices[:, 0], vertices[:, 2]
    box = (
        max(wide[0], float(x.min())), min(wide[1], float(x.max())),
        max(wide[2], float(z.min())), min(wide[3], float(z.max())),
    )
    # The shock layer's own scale: how far the domain stands off the wall at
    # its thinnest, which is the stagnation line.
    try:
        wall = np.asarray(body.interpolate(mesh.points))
        thickness = float(wall[:, 0].min() - x.min())
    except Exception:
        thickness = 0.0
    if not np.isfinite(thickness) or thickness <= 0.0:
        thickness = 0.05 * max(box[1] - box[0], 1e-9)
    pad = STANDOFF_PADDING * thickness
    return (box[0] - pad, box[1] + pad, box[2] - pad, box[3] + pad)


def _fill_body(panel: Any, body: Any, points: Any, color: str = "#6b737b") -> None:
    """Paint the vehicle solid.

    On an isotropic box domain the body is a small hole in a large field and
    reads as a body regardless. A shock-fitted block is **only the shock
    layer**, so the region the outline encloses has no cells in it, and the
    frame is mostly the vehicle's own interior rendered as blank paper. That
    looks exactly like a field that failed to draw -- which is what it was
    repeatedly taken for.

    Filled from the profile's own extent at each station rather than by
    ordering the cut into a path: :func:`aether.viz.flow.outline` declines to
    order its segments, and it is right to, because a marker cut by a plane can
    be several disjoint curves and joining them draws lines across gaps that
    are not surfaces. Taking the silhouette per station needs no ordering.
    """
    import numpy as np

    try:
        planar = np.asarray(body.coordinates(points))
    except Exception:
        return
    if planar.shape[0] < 3:
        return
    x, z = planar[:, 0], planar[:, 1]
    edges = np.linspace(float(x.min()), float(x.max()), 240)
    index = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    # Seeded at the infinities, not at NaN: minimum(NaN, z) is NaN, so a NaN
    # seed accumulates nothing and the fill silently draws empty.
    lower = np.full(len(edges) - 1, np.inf)
    upper = np.full(len(edges) - 1, -np.inf)
    np.minimum.at(lower, index, z)
    np.maximum.at(upper, index, z)
    good = np.isfinite(lower) & np.isfinite(upper)
    if good.sum() < 2:
        return
    centres = 0.5 * (edges[:-1] + edges[1:])
    panel.fill_between(
        centres[good], lower[good], upper[good],
        color=color, zorder=2.5, linewidth=0,
    )


def _nose_box(
    body: Any, mesh: Any, fraction: float = 0.08
) -> tuple[float, float, float, float] | None:
    """A window on the **body's** leading edge, where the structure is sub-pixel.

    The captured shock on the Mach 8 sphere-cone is 0.7 mm thick and stands
    9 mm off a 2 m body: a sixth of a pixel at whole-body scale, so the frame is
    honest and shows nothing.

    Anchored on the *body*, not the domain. Anchoring on the domain put this
    window on the far-field inflow plane fifteen metres upstream, where it
    faithfully showed undisturbed freestream.
    """
    import numpy as np

    try:
        vertices = np.asarray(body.interpolate(mesh.points))
    except Exception:
        return None
    if vertices.size == 0:
        return None
    x = vertices[:, 0]
    span = float(np.ptp(x)) * fraction
    if span <= 0.0:
        return None
    lead = float(x.min())
    return (lead - 0.6 * span, lead + 1.4 * span, -span, span)


#: Inches of the canvas that are not the field: y label and ticks, colorbar and
#: its label, title, x label and ticks plus the caption line. Fixed, because
#: they hold text at a fixed point size -- scaling them with the figure is what
#: makes a small frame's labels collide and a large one's swim in space.
GUTTER_LEFT = 0.66
GUTTER_RIGHT = 0.92

#: The right gutter on a panel with no colour scale: just enough not to clip
#: the last tick label.
GUTTER_PLAIN = 0.18
GUTTER_TOP = 0.38
GUTTER_BOTTOM = 0.62

#: Target area of the field itself, square inches. The *shape* comes from the
#: data; only the area is chosen here, so a squat body and a slender one get
#: frames of similar weight rather than similar width.
FIELD_AREA = 11.5

#: Neither dimension of the field may exceed this, so a very slender body does
#: not produce a frame too wide for a phone.
FIELD_MAX = 6.4
FIELD_MIN = 1.8


def field_size(box: tuple[float, float, float, float]) -> tuple[float, float]:
    """Width and height of the field, in inches, from the data it will hold.

    This is the whole fix for a frame that was *mostly white space*. The figure
    was a fixed 7.0 x 3.4 -- an aspect of 2.06, chosen for a 2 m sphere-cone --
    while every panel sets ``aspect="equal"``, which is not negotiable for a
    flow field. Matplotlib then shrinks the axes in whichever direction has to
    give and leaves the rest of the canvas blank. On the hemisphere, whose view
    box is 52 mm across and 104 mm tall, the axes collapsed to a narrow column
    and **more than half the frame was empty paper**.

    Sizing the canvas from the data instead means the axes fill it by
    construction, for any body, with no fitting and nothing to tune.
    """
    width = max(box[1] - box[0], 1e-12)
    height = max(box[3] - box[2], 1e-12)
    aspect = width / height

    field_w = math.sqrt(FIELD_AREA * aspect)
    field_h = math.sqrt(FIELD_AREA / aspect)

    # Both clamps scale the pair, so the aspect -- and therefore the absence of
    # white space -- survives either. They can disagree, though: no rectangle is
    # both at most FIELD_MAX on its long side and at least FIELD_MIN on its
    # short one once the aspect passes FIELD_MAX / FIELD_MIN. **The cap wins**,
    # so the minimum is applied first and the maximum last. Written the other
    # way round the minimum silently undid the cap and a 40:1 view box asked for
    # a frame six feet wide.
    if field_w < FIELD_MIN:
        field_w, field_h = FIELD_MIN, FIELD_MIN / aspect
    if field_h < FIELD_MIN:
        field_w, field_h = FIELD_MIN * aspect, FIELD_MIN
    if field_w > FIELD_MAX:
        field_w, field_h = FIELD_MAX, FIELD_MAX / aspect
    if field_h > FIELD_MAX:
        field_w, field_h = FIELD_MAX * aspect, FIELD_MAX
    return field_w, field_h


def _emptiest_corner(
    body: Any, points: Any, box: tuple[float, float, float, float],
    size: tuple[float, float],
) -> tuple[float, float]:
    """Axes-fraction origin for the inset, in whichever corner holds least body.

    The inset used to sit at a hard-coded top right. On a sphere-cone that is
    open sky; on a hemisphere it is the middle of the vehicle, and the detail
    was drawn over the thing it was detailing.
    """
    import numpy as np

    wide, tall = size
    corners = {
        (0.02, 1.0 - 0.02 - tall): 0.0,
        (1.0 - 0.02 - wide, 1.0 - 0.02 - tall): 0.0,
        (0.02, 0.02): 0.0,
        (1.0 - 0.02 - wide, 0.02): 0.0,
    }
    try:
        planar = np.asarray(body.coordinates(points))
        x = (planar[:, 0] - box[0]) / max(box[1] - box[0], 1e-30)
        z = (planar[:, 1] - box[2]) / max(box[3] - box[2], 1e-30)
    except Exception:
        return next(iter(corners))
    for origin in corners:
        left, bottom = origin
        inside = (x >= left) & (x <= left + wide) & (z >= bottom) & (z <= bottom + tall)
        corners[origin] = float(inside.sum())
    return min(corners, key=lambda origin: corners[origin])



def render_frame(
    case_dir: Path | str,
    iteration: int,
    *,
    figsize: tuple[float, float] | None = None,
    dpi: int = DPI,
    margin: float = MARGIN,
    mesh: Any = None,
    snapshot: Path | None = None,
    detail: bool = DETAIL_INSET,
) -> Frame | None:
    """Render the case's current volume snapshot. ``None`` if there isn't one.

    Deliberately built on :class:`matplotlib.figure.Figure` rather than
    ``pyplot``: pyplot keeps global figure state that is not safe to drive from
    a worker thread, and would leak a figure per frame for the length of a run.

    ``mesh`` accepts an already-parsed grid. A mesh does not change while its
    case is being solved, so re-reading eighteen megabytes of it every frame is
    pure waste -- about a third of the total cost, and it is the part that grows
    with cell count. :class:`FrameWriter` keeps one per case and passes it here.
    """
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from aether.aerodynamics.cfd.fields import (
        plane_cut,
        read_su2_mesh,
        read_volume,
        surface_cut,
    )
    from aether.viz import flow
    from aether.viz.flowfield import hot_spots, read_case_config

    case = Path(case_dir)
    # Named by the caller when it has already decided which snapshot is
    # finished; otherwise the overwritten one, which is what a direct call
    # from a notebook or a test means.
    volume = snapshot if snapshot is not None else snapshot_path(case)
    if not volume.exists():
        return None
    meshes = sorted(case.glob("*.su2"))
    config = case / "case.cfg"
    if not meshes or not config.exists():
        return None

    if mesh is None:
        mesh = read_su2_mesh(meshes[0])
    field = read_volume(volume)
    if mesh.size != field.size:
        # Caught mid-write, or a mesh and a solution from different runs. Not
        # an error worth stopping a solve for -- the next snapshot will do.
        return None
    conditions = read_case_config(config)

    walls = [tag for tag in conditions.walls if tag in mesh.markers]
    origin, normal = (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)
    cut = plane_cut(mesh, origin, normal)
    body = surface_cut(mesh, walls, origin, normal)
    # The outer boundary of the computed region, so it can be drawn as what it
    # is. On a shock-fitted block this is the marching surface; on an isotropic
    # box it is the far field and falls outside the view, drawing nothing.
    rim = None
    if "farfield" in mesh.markers:
        try:
            rim = surface_cut(mesh, ["farfield"], origin, normal)
        except Exception:
            rim = None
    # Frame what is actually drawn. `_extent` sizes the view from the walls plus
    # a margin, which suited a domain that was a large box around a small body.
    # A shock-fitted block *is* the shock layer -- 11 mm deep at the nose -- so
    # walls-plus-margin leaves the solution a thin sliver in a mostly empty
    # frame, which is what "not everything is being shown" looks like.
    box = _view_box(cut, body, mesh, walls, margin)

    # The solver's own primitives where the run wrote them. Deriving Mach would
    # need a gamma a dissociating shock layer does not have.
    mach = field.mach() if field.has_primitives else field.mach(conditions.require_gas())

    nose = _nose_box(body, mesh)
    # What fraction of the view box the domain actually occupies. Small on a
    # shock-fitted block, which is the case that needs explaining.
    # How much of the frame the solution actually occupies -- the **summed
    # triangle area**, not a bounding box. A shock-fitted block's bounding box
    # is the frame, so comparing boxes says it covers everything while it
    # covers a shell.
    try:
        planar = np.asarray(cut.coordinates(mesh.points))
        corners = planar[cut.triangles]
        edge_a = corners[:, 1] - corners[:, 0]
        edge_b = corners[:, 2] - corners[:, 0]
        area = 0.5 * np.abs(edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0])
        covered = float(area.sum()) / max(
            (box[1] - box[0]) * (box[3] - box[2]), 1e-30
        )
    except Exception:
        covered = 1.0

    field_w, field_h = field_size(box) if figsize is None else (
        figsize[0] - GUTTER_LEFT - GUTTER_RIGHT, figsize[1] - GUTTER_TOP - GUTTER_BOTTOM
    )
    # The right gutter holds the colour scale, so a panel without one does not
    # get it. The Schlieren frame shared the Mach frame's canvas and carried an
    # inch of empty paper down its right edge for it.
    def _canvas(colorbar: bool) -> tuple[tuple[float, float], float]:
        right = GUTTER_RIGHT if colorbar else GUTTER_PLAIN
        return (
            (GUTTER_LEFT + field_w + right, GUTTER_TOP + field_h + GUTTER_BOTTOM),
            right,
        )

    canvas, _ = _canvas(True)

    # How many pixels the shock layer gets in the main view. The inset exists
    # because a captured shock 0.7 mm thick on a 2 m body is a sixth of a pixel
    # at whole-body scale -- but the same shock on a 45 mm hemisphere is nearly
    # forty pixels and the detail is redundant, so this is a measurement.
    #
    # Measured along the stagnation streamline: the domain's upstream-most point
    # near the axis is the fitted shock, and the body's is the nose, so the gap
    # between them is the standoff. Estimating it from the nose *window* instead
    # -- which is a fixed fraction of body length and has nothing to do with the
    # shock -- put the hemisphere at 20 px and drew an inset over the vehicle.
    shock_pixels = float("inf")
    if nose is not None:
        try:
            plane = np.asarray(cut.coordinates(mesh.points))
            wall = np.asarray(body.coordinates(mesh.points))
            band = 0.02 * max(box[3] - box[2], 1e-30)
            axial = plane[np.abs(plane[:, 1]) <= band]
            stagnation = wall[np.abs(wall[:, 1]) <= band]
            if len(axial) and len(stagnation):
                standoff = float(stagnation[:, 0].min() - axial[:, 0].min())
                if standoff > 0.0:
                    shock_pixels = (
                        standoff / max(box[1] - box[0], 1e-30) * field_w * dpi
                    )
        except Exception:
            shock_pixels = float("inf")

    def _draw(figure: Any, paint: Any, title: str, colorbar: bool = False) -> None:
        """One field filling the canvas, with the leading edge inset when it helps.

        The axes are placed explicitly, in inches. ``tight_layout`` cannot do
        this job: it fits the axes to the canvas, and the canvas was the thing
        that was wrong. Here the canvas is built from the field instead, so the
        only free space left is the gutters, which hold the labels.

        It was two panels side by side once, and that was a mistake -- it cost
        the primary view half its width on every frame. The nose detail is worth
        having and not at that price, so it is an inset, in the corner with the
        least vehicle in it, and only when the main view cannot already resolve
        the shock.
        """
        canvas, _right = _canvas(colorbar)
        wide = figure.add_axes([
            GUTTER_LEFT / canvas[0],
            GUTTER_BOTTOM / canvas[1],
            field_w / canvas[0],
            field_h / canvas[1],
        ])
        wide._aether_window = box
        wide.set_facecolor(background)
        legend_keys: list[Any] = []
        wide._aether_legend = legend_keys
        paint(wide)
        wide.set_aspect("equal")
        wide.set_xlim(box[0], box[1])
        wide.set_ylim(box[2], box[3])
        wide.set_xlabel("x [m]", fontsize=8, labelpad=2)
        wide.set_ylabel("z [m]", fontsize=8, labelpad=2)
        wide.tick_params(labelsize=7, length=2.5, pad=1.5)

        # The edge of the computed field, drawn on purpose. Outside it the frame
        # is painted at freestream because there is no solution there, and that
        # fill meets the computed field along the mesh's faceted outer edge --
        # which reads as a sudden change in the flow and is not one. Measured on
        # the Mach 8 sphere-cone along the stagnation line: the boundary is at
        # x = -11.28 mm, the flow is Mach 8.000 for 2.2 mm inside it, and the
        # shock is at -8.3 mm. A quarter of the domain is undisturbed freestream
        # by construction -- the marching margin is 1.25 -- and the apparent
        # jump is the seam, not the shock.
        if rim is not None:
            flow.outline(
                wide, rim, mesh.points, color=edge_colour,
                linewidth=0.7, linestyle=(0, (5, 3)), alpha=0.7, zorder=3.5,
            )
            legend_keys.append(
                Line2D([], [], color=edge_colour, linewidth=0.7,
                       linestyle=(0, (5, 3)), label="edge of the computed domain")
            )
        if legend_keys:
            key = wide.legend(
                handles=legend_keys, loc="lower right", fontsize=5.5,
                framealpha=0.82, borderpad=0.4, handlelength=1.8,
                facecolor="white", edgecolor="none",
            )
            key.set_zorder(5)

        if nose is not None and detail and shock_pixels < 25.0:
            # Square, because ``_nose_box`` is; sized against the *figure* so it
            # stays the same physical size whatever shape the frame came out.
            # A third of the shorter side, so the detail is legible and the
            # field it is a detail of is still the picture. At 1.35 in fixed it
            # was 56 % of a narrow frame's width.
            span = min(1.25, 0.34 * min(field_w, field_h))
            fraction = (span / field_w, span / field_h)
            if fraction[0] < 0.45 and fraction[1] < 0.45:
                left, bottom = _emptiest_corner(body, mesh.points, box, fraction)
                close = wide.inset_axes([left, bottom, *fraction])
                close._aether_window = nose
                close.set_facecolor(background)
                paint(close)
                close.set_aspect("equal")
                close.set_xlim(nose[0], nose[1])
                close.set_ylim(nose[2], nose[3])
                close.set_xticks([])
                close.set_yticks([])
                for spine in close.spines.values():
                    spine.set_color("0.55")
                    spine.set_linewidth(0.8)
                # Labelled inside, on its own plate. As an axes title in light
                # grey it sat on whatever the inset's top edge happened to be --
                # pale freestream on the Mach map -- and could not be read.
                close.text(
                    0.5, 0.985, "leading edge",
                    transform=close.transAxes, fontsize=5.5, color="0.12",
                    va="top", ha="center",
                    bbox={
                        "facecolor": "white", "alpha": 0.78,
                        "edgecolor": "none", "pad": 1.2,
                    },
                )

        # Sized to fit. A fixed 8.5 pt ran "264.164 K" off the right edge of
        # any frame narrower than about six inches, which is every blunt body.
        room = canvas[0] - 0.1
        size = min(8.5, max(5.5, room * 72.0 / (0.52 * max(len(title), 1))))
        figure.text(
            0.5, 1.0 - 0.16 / canvas[1], title,
            fontsize=size, va="top", ha="center",
        )
        # Only when the legend is not already saying it.
        if rim is None and covered < 0.55:
            # In the bottom gutter, not over the field. Inside the axes it was
            # drawn across the solution and clipped by the frame edge.
            room = canvas[0] - 0.1
            figure.text(
                0.5, 0.08 / canvas[1], note,
                fontsize=min(5.8, max(4.2, room * 72.0 / (0.52 * len(note)))),
                color="0.35", va="bottom", ha="center",
            )

    figure = Figure(figsize=canvas, dpi=dpi)
    FigureCanvasAgg(figure)

    drawn: list[Any] = []

    # Outside a shock-fitted block the flow **is** freestream -- that is the
    # physical justification for cutting the domain there, and the boundary
    # condition the solver imposes on it. Leaving it blank drew the vehicle in a
    # field that appeared to stop existing; painting it at the freestream value
    # shows what the computation assumes, which is not the same as inventing a
    # solution and is labelled as such.
    from matplotlib import colormaps

    background = colormaps["magma"](1.0 / 1.05)
    edge_colour = "#3a2a12"
    note = "dashed: edge of the computed domain. Outside is freestream, not solved."

    def _mach(panel: Any) -> None:
        drawn.append(
            flow.field_map(
                panel, cut, mesh.points, mach, cmap="magma",
                levels=np.linspace(0.0, conditions.mach * 1.05, 128), extend="max",
            )
        )
        # Contours, and the sonic line above all.
        #
        # A filled map on a linear scale from 0 to M_inf cannot show a blunt
        # body's shock layer, and the reason is arithmetic: the layer spans
        # Mach 0 to about 2.3 -- **the bottom quarter of the scale** -- so the
        # whole of it lands in the dark end of the colormap and reads as one
        # featureless shell wrapped round the vehicle. It looks like the body
        # is glowing rather than like flow arriving from one side, which is
        # exactly what it was repeatedly reported as.
        #
        # Lines fix that without distorting the field. The sonic line is the
        # one a hypersonicist looks for first -- it separates the subsonic nose
        # region from the flow that has accelerated round the shoulder, and it
        # is the feature that makes a blunt-body picture legible to someone who
        # has never seen one.
        # Labelled in a legend, not inline. `clabel` writes the value repeatedly
        # along a contour, and the sonic line on a slender body runs the whole
        # length of the vehicle just outside its boundary layer -- so the labels
        # turned a clean line into a dotted ribbon and the picture into noise.
        keys: list[Any] = getattr(panel, "_aether_legend", None)
        if keys is None:
            keys = []
        try:
            grid = flow.triangulation(cut, mesh.points)
            planar = cut.interpolate(mach)
            rest = [m for m in (2.0, 4.0, 6.0) if m < conditions.mach]
            if rest:
                panel.tricontour(
                    grid, planar, levels=rest,
                    colors="#b9c6d6", linewidths=0.45, alpha=0.6, zorder=3.1,
                )
                keys.append(
                    Line2D([], [], color="#b9c6d6", linewidth=0.8,
                           label="Mach " + ", ".join(f"{m:g}" for m in rest))
                )
            panel.tricontour(
                grid, planar, levels=[1.0],
                colors="#39d0ff", linewidths=1.6, zorder=3.2,
            )
            keys.insert(
                0,
                Line2D([], [], color="#39d0ff", linewidth=1.6,
                       label="sonic line, Mach 1"),
            )
        except Exception:
            pass
        _fill_body(panel, body, mesh.points)
        flow.outline(panel, body, mesh.points, color="white", linewidth=1.0, zorder=3)

        # Which way the flow is going, and how fast. Obvious to an analyst from
        # the stagnation point; not obvious to anyone else, and the frames are
        # read by both.
        window = getattr(panel, "_aether_window", None)
        if window is not None:
            width = window[1] - window[0]
            height = window[3] - window[2]
            panel.annotate(
                "", xytext=(window[0] + 0.02 * width, window[2] + 0.09 * height),
                xy=(window[0] + 0.16 * width, window[2] + 0.09 * height),
                arrowprops={"arrowstyle": "-|>", "color": "0.15", "linewidth": 1.3},
                zorder=4,
            )
            panel.text(
                window[0] + 0.02 * width, window[2] + 0.115 * height,
                f"M {conditions.mach:g}", fontsize=6.5, color="0.15", zorder=4,
            )

    _draw(figure, _mach, f"iteration {iteration}  --  {conditions.label}", colorbar=True)
    # Without a scale the picture says "there is a gradient" and not what of.
    if drawn:
        # Placed in the right gutter, in inches, rather than taking a fraction
        # of the axes. ``ax=figure.axes`` also handed it the *inset*, so the
        # detail panel was being shrunk to make room for a scale bar.
        main = figure.axes[0]
        position = main.get_position()
        figure.colorbar(
            drawn[0],
            cax=figure.add_axes([
                position.x1 + 0.10 / canvas[0], position.y0,
                0.16 / canvas[0], position.height,
            ]),
            label="Mach",
        ).ax.tick_params(labelsize=7, length=2.5, pad=1.5)

    # Numerical Schlieren alongside the Mach field. A Mach map of a thin
    # shock-fitted domain is nearly uniform with one dark band, because almost
    # all of it *is* freestream -- the quantity that makes a shock legible is
    # the density gradient, which is what a Schlieren photograph shows and what
    # a hypersonicist reads first.
    #
    # `schlieren_magnitude` and `schlieren` have existed in `aether.viz.flow`
    # throughout, documented down to why the exposure reference is a percentile
    # rather than the maximum. Nothing had ever called them.
    try:
        magnitude = flow.schlieren_magnitude(mesh, field, relative=True)
        shadow = Figure(figsize=_canvas(False)[0], dpi=dpi)
        FigureCanvasAgg(shadow)
        # Undisturbed flow is intensity 1, which is white on a plain grey map.
        background = colormaps["gray"](1.0)
        edge_colour = "#7a7a7a"
        note = "dashed: edge of the computed domain. The shock is the dark arc inside it."

        def _shade(panel: Any) -> None:
            # Plain "gray", not reversed: schlieren_intensity returns **1 for
            # undisturbed flow**, so a normal grey map puts white ahead of the
            # shock and black through it, which is the photograph. Reversing it
            # paints the freestream black and saturates the whole shock layer.
            #
            # Each panel gets its own exposure. `schlieren` shares one across
            # cuts so a *sequence* does not flicker, and that is right -- but two
            # panels of one instant at different scales are not a sequence, and a
            # shared exposure leaves the zoomed one solid black, because the
            # gradients it is zoomed into are the ones that set the scale.
            window = getattr(panel, "_aether_window", None)
            reference = None
            if window is not None:
                inside = (
                    (mesh.points[:, 0] >= window[0])
                    & (mesh.points[:, 0] <= window[1])
                    & (mesh.points[:, 2] >= window[2])
                    & (mesh.points[:, 2] <= window[3])
                )
                if int(inside.sum()) > 32:
                    reference = float(np.percentile(magnitude[inside], 99.5))
            flow.schlieren(panel, cut, mesh.points, magnitude, reference=reference)
            _fill_body(panel, body, mesh.points, color="#4a4f55")
            flow.outline(panel, body, mesh.points, color="#e0a458", linewidth=1.0, zorder=3)

        _draw(shadow, _shade, f"iteration {iteration}  --  numerical Schlieren")
    except Exception:
        shadow = None

    spot = hot_spots(mesh, field, conditions)
    target = frame_dir(case)
    target.mkdir(parents=True, exist_ok=True)
    image = target / f"iter_{iteration:06d}.png"
    figure.savefig(image)
    if shadow is not None:
        shadow.savefig(image.with_name(image.stem + "-schlieren.png"))

    payload: dict[str, Any] = {
        "iteration": iteration,
        "note": spot.describe(),
        "diverged": bool(spot.count),
        "mach": conditions.mach,
    }
    image.with_suffix(".json").write_text(json.dumps(payload, indent=2))
    return Frame(iteration, image, payload["note"], payload["diverged"])


class FrameWriter:
    """Renders a frame each time the solver rewrites its volume snapshot.

    Holds the snapshot's modification time -- so a worker that restarts renders
    once more than it strictly had to, which is the right way round for
    something that must never block a solve -- and the parsed mesh, which does
    not change for the life of a case and is a third of the cost of a frame.
    """

    #: Cases whose mesh could not be parsed at all, and the reason. A mesh is
    #: fixed for the life of a run, so this is remembered rather than retried.
    _unreadable: dict[str, str]

    def __init__(self) -> None:
        self._seen: dict[str, float] = {}
        self._meshes: dict[str, Any] = {}
        self._failed: dict[str, tuple[float, int]] = {}
        self._sizes: dict[str, int] = {}
        self._unreadable: dict[str, str] = {}

    def poll(self, case_dir: Path | str, iteration: int) -> Frame | None:
        """Render the newest snapshot SU2 has finished writing, if it is new.

        A numbered snapshot with a higher-numbered sibling is closed --
        :func:`completed_snapshots` -- so there is nothing to wait for and
        nothing to race. ``iteration`` is taken from the **filename** when a
        numbered snapshot is used, because that is the iteration the field
        actually belongs to; the caller's count is the one the history has
        reached by now, which is later and by a varying amount.

        The fallback path, for a case solved before numbering was turned on, is
        the old heuristic: a single overwritten file, left alone until it has
        been still for :data:`QUIESCENT` and the same size on two consecutive
        polls. It is kept because those cases exist on disk, and it is not
        trusted because it did not work -- on a run writing 35 MB every four
        seconds it rendered nothing at all for the whole run.

        Either way the stamp is recorded on *success*, so a failed read is
        retried on the next pass. Retries are capped: a file that never parses
        is a fault worth seeing once rather than every three seconds.
        """
        case = Path(case_dir)
        key = str(case)

        finished = completed_snapshots(case)
        if finished:
            number, volume = finished[-1]
            iteration = number
            # Everything older has been passed over and will not be read again.
            for _, stale in finished[:-1]:
                with contextlib.suppress(OSError):
                    stale.unlink()
            try:
                stamp = volume.stat().st_mtime
            except OSError:
                return None
            if self._seen.get(key) == stamp:
                return None
            previous, attempts = self._failed.get(key, (0.0, 0))
            if previous == stamp and attempts >= MAX_ATTEMPTS:
                return None
        else:
            volume = snapshot_path(case)
            try:
                stamp = volume.stat().st_mtime
            except OSError:
                return None
            if self._seen.get(key) == stamp:
                return None
            if time.time() - stamp < QUIESCENT:
                return None  # still being written; next pass
            previous, attempts = self._failed.get(key, (0.0, 0))
            if previous == stamp and attempts >= MAX_ATTEMPTS:
                return None
            size = volume.stat().st_size
            if self._sizes.get(key) != size:
                self._sizes[key] = size
                return None

        if key in self._unreadable:
            return None
        mesh = self._meshes.get(key)
        if mesh is None:
            from aether.aerodynamics.cfd.fields import read_su2_mesh

            grids = sorted(case.glob("*.su2"))
            if grids:
                try:
                    mesh = self._meshes[key] = read_su2_mesh(grids[0])
                except (ValueError, OSError) as error:
                    # A mesh does not change while its case is being solved, so
                    # one failure is final and retrying is pure noise. This read
                    # used to sit outside the guard below, where the attempt
                    # counter could not see it -- and the counter is keyed on
                    # the *snapshot* mtime, which moves every snapshot, so even
                    # inside it would have reset forever. Measured: an
                    # unreadable mesh logged the identical error every 3.5
                    # seconds for the length of the run.
                    self._unreadable[key] = str(error)
                    raise

        try:
            frame = render_frame(case, iteration, mesh=mesh, snapshot=volume)
        except (ValueError, OSError):
            # Losing the race is expected, not exceptional: the parser fails
            # with something like "node 31065 has 0 values for 24 variables"
            # and the next pass, three seconds later, reads the finished file
            # without trouble. So it is swallowed until the retries run out,
            # at which point the file really is unreadable and worth hearing
            # about exactly once.
            count = attempts + 1 if previous == stamp else 1
            self._failed[key] = (stamp, count)
            if count >= MAX_ATTEMPTS:
                raise
            return None
        self._seen[key] = stamp
        self._failed.pop(key, None)
        return frame

    def forget(self, case_dir: Path | str) -> None:
        """Drop what is held for a case. Called when its run ends.

        The mesh matters more than the mtime here: it is tens of megabytes of
        parsed arrays, and a worker that solved twenty cases without this would
        be holding all twenty.
        """
        key = str(Path(case_dir))
        self._seen.pop(key, None)
        self._meshes.pop(key, None)
        self._failed.pop(key, None)
        self._sizes.pop(key, None)
        self._unreadable.pop(key, None)
