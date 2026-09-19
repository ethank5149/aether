r"""A picture of the grid, before anything is solved on it.

Meshing is the one stage of a run that produced no diagnostics at all. A card
sat on "meshing" for thirty to ninety seconds and then a mesh either worked or
did not, and the single number that came back -- the element count -- says
nothing about **where** the cells went. That is precisely the question this
pipeline got wrong for a fortnight: 1.36 million elements, and *zero* of them
between the body and the bow shock.

A symmetry-plane slice answers it in one glance. Draw the wall-normal columns,
the body, and the shock the mesh was sized against, and a mesh that clusters in
the wrong place is obvious immediately rather than three hours into a solve.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["MeshPreview", "preview_path", "render_mesh_preview"]

#: Retired as the figure size -- the canvas is built from the data by
#: :func:`_fit`. Kept because it is still the fallback when nothing is drawn.
FIGSIZE = (9.0, 4.2)
DPI = 110

#: Inches of canvas that hold text rather than mesh.
PAD_LEFT = 0.62
PAD_RIGHT = 0.16
PAD_TOP = 0.42
PAD_BOTTOM = 0.52
PAD_GAP = 0.55
"""Between the two panels, when there are two."""

#: Height of the drawing area. The widths follow from each panel's own data.
PANEL_HEIGHT = 3.6
PANEL_MIN = 0.9
PANEL_MAX = 6.0


def _fit(figure: Any, panels: list[Any]) -> None:
    """Resize the canvas to the drawn data and place the panels in it.

    Same defect, and same fix, as :func:`aether.cfd.frames.field_size`.
    The figure was a fixed 9.0 x 4.2 split ``width_ratios=(2.0, 1.0)``, which is
    a statement about one body's proportions applied to every body. A hemisphere
    slice is 52 mm by 56 mm -- square -- and got a panel twice as wide as it is
    tall.

    What made that worse than white space is ``adjustable="datalim"``: told to
    keep an equal aspect in a box of the wrong shape, matplotlib **widened the
    data limits** instead of narrowing the axes, so the preview claimed a domain
    running from -0.03 m to +0.07 m when the mesh spans -0.008 to 0.045. The
    axes are ``adjustable="box"`` now and the canvas is what moves.

    Done after drawing rather than before, so the limits are the ones actually
    produced -- the isotropic path autoscales and does not know its own extent
    until its points are in.
    """
    widths = []
    for panel in panels:
        panel.set_aspect("equal", adjustable="box")
        left, right = panel.get_xlim()
        bottom, top = panel.get_ylim()
        aspect = abs(right - left) / max(abs(top - bottom), 1e-30)
        widths.append(min(PANEL_MAX, max(PANEL_MIN, PANEL_HEIGHT * aspect)))

    gaps = PAD_GAP * (len(panels) - 1)
    canvas = (
        PAD_LEFT + sum(widths) + gaps + PAD_RIGHT,
        PAD_TOP + PANEL_HEIGHT + PAD_BOTTOM,
    )
    figure.set_size_inches(*canvas)
    cursor = PAD_LEFT
    for panel, width in zip(panels, widths, strict=True):
        panel.set_position([
            cursor / canvas[0],
            PAD_BOTTOM / canvas[1],
            width / canvas[0],
            PANEL_HEIGHT / canvas[1],
        ])
        cursor += width + PAD_GAP


def _layer_indices(levels: int, wanted: int = 14) -> list[int]:
    """Which layers to draw, spaced so the near-wall grading is visible.

    Evenly spaced indices would put almost every line in the outer half, where
    the cells are large, and none in the first few microns where the whole
    design effort went. Geometric spacing shows the grading instead of hiding
    it.
    """
    import numpy as np

    if levels <= wanted:
        return list(range(levels))
    fractions = np.geomspace(1.0, float(levels - 1), wanted)
    return sorted({0, *(int(round(f)) for f in fractions), levels - 1})  # noqa: RUF046


def preview_path(case_dir: Path | str) -> Path:
    """Where the meshing preview lives. One per case, overwritten."""
    return Path(case_dir) / "mesh.png"


@dataclass(frozen=True)
class MeshPreview:
    path: Path
    cells: int
    in_shock_layer: int | None
    """Cells between wall and shock on the tightest column, when known."""

    wall_cell: float | None


def render_mesh_preview(
    case_dir: Path | str,
    layer: Any = None,
    envelope: Any = None,
    slice_half_width: float = 0.02,
) -> MeshPreview | None:
    """Draw a symmetry-plane slice of the grid. ``None`` if there is nothing to draw.

    ``layer`` is an :class:`~aether.cfd.extrude.ExtrudedLayer` when the
    shock-fitted mesher built this case, in which case the columns are drawn
    directly and exactly. Without one -- the isotropic tetrahedral path -- the
    written ``.su2`` is read back and its node cloud shown instead, which is
    less informative but still answers "is anything near the nose".

    Built on :class:`matplotlib.figure.Figure` rather than ``pyplot`` for the
    same reason :mod:`aether.cfd.frames` is: pyplot holds global state
    that is not safe to drive from the worker thread.
    """
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    case = Path(case_dir)
    figure = Figure(figsize=FIGSIZE, dpi=DPI)
    FigureCanvasAgg(figure)
    # Two panels. The whole body answers "does the domain make sense"; the nose
    # answers the question that actually went wrong, because a 9 mm standoff on
    # a 2 m vehicle is four pixels wide on the left-hand plot and a mesh with
    # nothing in it looks identical to a mesh with a hundred cells in it.
    axes, nose_axes = figure.subplots(1, 2, width_ratios=(2.0, 1.0))
    figure.patch.set_facecolor("#0b0f14")
    for panel in (axes, nose_axes):
        panel.set_facecolor("#0b0f14")

    cells = 0
    tightest: int | None = None
    wall_cell: float | None = None
    #: Nose standoff, metres, when the mesh is one this can be measured on. It
    #: decides whether the nose panel is worth its half of the canvas.
    standoff: float | None = None

    if layer is not None:
        nodes = np.asarray(layer.nodes)
        wall_cell = float(layer.wall_cell)
        cells = int(layer.faces.shape[0] * layer.layers)
        near = np.abs(nodes[:, 0, 2]) < slice_half_width * max(
            1e-9, float(np.ptp(nodes[:, 0, 0]))
        )
        columns = nodes[near]
        # Every tenth column, so a thousand-node body reads as a grid rather
        # than a solid block of ink.
        order = np.argsort(columns[:, 0, 0])
        columns = columns[order]
        outer_radius = float(np.hypot(nodes[:, -1, 1], nodes[:, -1, 2]).max())
        # Along the stagnation streamline: the wall's upstream-most point and
        # the outer boundary's, which on a shock-fitted block is the shock.
        standoff = float(nodes[:, 0, 0].min() - nodes[:, -1, 0].min())
        if envelope is not None:
            from aether.cfd.shock import _field

            inside = _field(envelope, nodes.reshape(-1, 3)).reshape(nodes.shape[:2]) < 0
            tightest = int((inside.sum(axis=1) - 1).min())

        span = float(np.ptp(nodes[:, 0, 0]))
        for panel, every, window in (
            (axes, max(1, len(columns) // 140), None),
            (nose_axes, 1, 0.04 * span),
        ):
            shown = columns if window is None else columns[
                columns[:, 0, 0] < columns[:, 0, 0].min() + window
            ]
            step = every if window is None else max(1, len(shown) // 60)
            drawn = shown[::step]
            # **Both** families of edges, so this reads as a grid rather than as
            # a fan of lines. Wall-normal columns alone are what the mesher
            # computes, but they are not what a mesh looks like, and a picture
            # whose subject is unrecognisable answers no question.
            for column in drawn:
                panel.plot(column[:, 0], np.hypot(column[:, 1], column[:, 2]),
                           color="#3d7ea6", linewidth=0.35, alpha=0.85)
            # Constant-layer curves. Graded, so they crowd at the wall -- which
            # is the whole point of the mesh and should be visible.
            levels = drawn.shape[1]
            for k in _layer_indices(levels, wanted=14):
                panel.plot(drawn[:, k, 0],
                           np.hypot(drawn[:, k, 1], drawn[:, k, 2]),
                           color="#2f6d8f", linewidth=0.3, alpha=0.7)
            panel.plot(shown[:, 0, 0], np.hypot(shown[:, 0, 1], shown[:, 0, 2]),
                       color="#e8e8e8", linewidth=1.4, label="body")
            panel.plot(shown[:, -1, 0], np.hypot(shown[:, -1, 1], shown[:, -1, 2]),
                       color="#6fbf73", linewidth=1.0, label="outer boundary")
            # Drawn through the envelope's own frame, so an incidence case
            # shows the shock where it actually stands rather than where the
            # body axis would put it.
            grid = np.linspace(0.0, outer_radius, 400)
            curve = None
            if envelope is not None and hasattr(envelope, "trace"):
                points = np.asarray(envelope.trace(grid))
                curve = (points[:, 0], np.hypot(points[:, 1], points[:, 2]))
            elif envelope is not None and hasattr(envelope, "axial_at"):
                curve = (envelope.axial_at(grid), grid)
            if curve is not None:
                panel.plot(curve[0], curve[1], color="#e0a458",
                           linewidth=1.3, linestyle="--", label="shock")
            if window is not None:
                panel.set_xlim(columns[:, 0, 0].min() - 0.35 * window,
                               columns[:, 0, 0].min() + window)
                top = float(np.hypot(shown[:, -1, 1], shown[:, -1, 2]).max())
                panel.set_ylim(-0.08 * top, 1.1 * top)
    else:
        meshes = sorted(case.glob("*.su2"))
        if not meshes:
            return None
        points = _read_su2_points(meshes[0])
        if points is None:
            return None
        cells, coordinates = points
        near = np.abs(coordinates[:, 2]) < slice_half_width * max(
            1e-9, float(np.ptp(coordinates[:, 0]))
        )
        shown = coordinates[near]
        radial = np.hypot(shown[:, 1], shown[:, 2])
        span = float(np.ptp(coordinates[:, 0]))
        axes.scatter(shown[:, 0], radial, s=0.4, c="#3d7ea6", alpha=0.6, linewidths=0)
        window = 0.04 * span
        head = shown[:, 0] < shown[:, 0].min() + window
        nose_axes.scatter(shown[head, 0], radial[head], s=2.0, c="#3d7ea6",
                          alpha=0.8, linewidths=0)
        nose_axes.set_xlim(shown[:, 0].min() - 0.35 * window, shown[:, 0].min() + window)

    # The nose panel exists because a 9 mm standoff on a 2 m vehicle is four
    # pixels on the whole-body plot, and a mesh with nothing in it looks
    # identical to a mesh with a hundred cells in it. On a 45 mm hemisphere the
    # same standoff is most of the panel, so the detail is a second copy of the
    # picture beside it -- measured, not assumed, the same way the flow frames
    # decide about their inset.
    span = abs(axes.get_xlim()[1] - axes.get_xlim()[0])
    bottom, top = axes.get_ylim()
    panel_px = PANEL_HEIGHT * (span / max(abs(top - bottom), 1e-30)) * DPI
    detail = standoff is None or span <= 0.0 or (standoff / span * panel_px < 40.0)
    if not detail:
        nose_axes.remove()

    shown_panels = [axes, nose_axes] if detail else [axes]
    titles = ["whole body", "nose"] if detail else ["symmetry-plane slice"]
    for panel, title in zip(shown_panels, titles, strict=True):
        panel.set_xlabel("x (m)", color="#9fb3c8", fontsize=8)
        panel.tick_params(colors="#9fb3c8", labelsize=7)
        for spine in panel.spines.values():
            spine.set_color("#25303b")
        # Inside the axes, on the dark ground. As an axes title it sat in the
        # top gutter and collided with the caption.
        panel.text(
            0.985, 0.975, title, transform=panel.transAxes,
            color="#6d8299", fontsize=7, ha="right", va="top",
        )
    axes.set_ylabel("r (m)", color="#9fb3c8", fontsize=8)
    # Only when something is labelled. The isotropic path draws a bare point
    # cloud, and matplotlib warns rather than drawing an empty legend box.
    if axes.get_legend_handles_labels()[0]:
        legend = axes.legend(loc="upper left", fontsize=7, framealpha=0.0)
        for text in legend.get_texts():
            text.set_color("#9fb3c8")
    caption = f"{cells:,} cells"
    if wall_cell:
        # Said in words because it cannot be shown: the first cell is 0.06 % of
        # the nose radius, so the near-wall layers are sub-pixel at any scale
        # that also shows the body, and they merge into its outline.
        caption += f"   wall cell {wall_cell * 1e6:.3g} um (sub-pixel here)"
    if layer is not None:
        caption += f"   {layer.layers} layers"
    if tightest is not None:
        caption += f"   {tightest} in the shock layer"
    _fit(figure, shown_panels)
    # After `_fit`, so it is placed against the canvas the panels decided on.
    figure.text(
        PAD_LEFT / figure.get_size_inches()[0],
        1.0 - 0.16 / figure.get_size_inches()[1],
        caption, color="#cfe3f7", fontsize=8, va="top", ha="left",
    )

    target = preview_path(case)
    figure.savefig(target, facecolor=figure.get_facecolor())
    return MeshPreview(
        path=target, cells=cells, in_shock_layer=tightest, wall_cell=wall_cell
    )


def _read_su2_points(path: Path, limit: int = 400_000) -> tuple[int, Any] | None:
    """Node coordinates from a ``.su2``, without parsing its connectivity.

    Only the point block is needed for a preview, and skipping the elements is
    what keeps this from costing as much as the mesh generation it illustrates.
    """
    import numpy as np

    cells = 0
    coordinates: list[tuple[float, float, float]] = []
    try:
        with path.open() as handle:
            for line in handle:
                if line.startswith("NELEM="):
                    cells = int(line.split("=")[1])
                elif line.startswith("NPOIN="):
                    count = min(int(line.split("=")[1].split()[0]), limit)
                    for _ in range(count):
                        parts = handle.readline().split()
                        coordinates.append(
                            (float(parts[0]), float(parts[1]), float(parts[2]))
                        )
                    break
    except (OSError, ValueError, IndexError):
        return None
    return (cells, np.asarray(coordinates)) if coordinates else None
