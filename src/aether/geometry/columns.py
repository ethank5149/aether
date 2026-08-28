r"""Wall-normal columns over an arbitrary body — the grid ablation actually needs.

Heat entering a heat shield goes almost entirely one way: **into the wall**. The
temperature gradient normal to the surface is enormous, the gradients along it
are mild, and recession is set by the first few millimetres. A grid that does
not resolve that direction preferentially is spending its cells in the wrong
place, and the symptom is specific and misleading — the surface stays too cool,
recession comes out too low, and the run looks conservative rather than wrong.

A uniform Cartesian grid over the whole vehicle is the worst case of this. To
put a millimetre of resolution at the wall it must put a millimetre everywhere,
including through metres of interior the heat will never reach, so the cell
count goes as the cube of a resolution only needed on a surface.

What this module builds instead
-------------------------------

One **column per wall face**, driven inward along that face's own normal, with
cell widths in geometric progression so the first cell is thin and the last is
not. Resolution scales with the *surface*, not the volume, and it follows the
body: a waverider's thin keel and its blunt base get columns of the right
length automatically because each column's depth is measured by casting the
normal through the body and finding where it leaves.

This is how through-thickness ablation is done in practice — a one-dimensional
solve at many body points — and it is the grid
:class:`~aether.fiat.stack.MaterialStack` already expects, one ply at a time.
:func:`wall_columns` therefore serves both the simple enthalpy solver and the
full Chen–Milos response code without either needing to know about the other.

What it does not do
-------------------

Columns do not talk to each other. Lateral conduction between neighbouring
surface locations is dropped, which is the standard assumption and is good
exactly where the Biot number across a column is large — true for a heat shield
under entry heating, false for a thin, highly conductive skin at low flux. On a
sharp leading edge, where two faces' columns converge on the same material from
opposite sides, treating them independently double-counts the available depth;
:attr:`WallColumnGrid.depth` is clipped by the actual through-body distance,
which bounds that error but does not remove it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["WallColumnGrid", "graded_widths", "solve_growth", "wall_columns"]

_FloatArray = NDArray[np.float64]


def graded_widths(thickness: float, n_cells: int, growth: float) -> _FloatArray:
    """Cell widths in geometric progression, summing exactly to ``thickness``.

    ``growth`` is the ratio of each cell to the one before it going **inward**,
    so values above one refine toward the heated face. Normalised rather than
    closed-formed, which keeps the sum exact at any ratio — the same convention
    :meth:`aether.fiat.stack.Ply.cell_widths` uses, so a column can be handed
    to either solver without re-deriving its grid.
    """
    if not (np.isfinite(thickness) and thickness > 0.0):
        msg = f"thickness must be finite and > 0, got {thickness}"
        raise ValueError(msg)
    if int(n_cells) < 2:
        msg = f"n_cells must be >= 2, got {n_cells}"
        raise ValueError(msg)
    if not (np.isfinite(growth) and 0.5 <= growth <= 2.0):
        msg = f"growth must be finite and within [0.5, 2.0], got {growth}"
        raise ValueError(msg)
    widths = (
        np.ones(int(n_cells))
        if growth == 1.0
        else float(growth) ** np.arange(int(n_cells), dtype=np.float64)
    )
    return np.asarray(widths * (float(thickness) / widths.sum()))


@dataclass(frozen=True)
class WallColumnGrid:
    """Inward columns over a body's wall, one per surface face.

    Attributes
    ----------
    origin:
        Surface points the columns start from, ``(n_columns, 3)``.
    normal:
        **Inward** unit normals, ``(n_columns, 3)``. Inward, not outward: every
        expression here marches into the material, and flipping the sign once
        at construction is better than remembering to flip it at each use.
    area:
        Tributary wall area of each column (m²). What turns a heat *flux* into
        a power, and what makes the mapping conserve energy.
    depth:
        Material available along each normal (m), from casting the normal
        through the body.
    widths:
        Cell widths, ``(n_columns, n_cells)``, thin end at the wall.
    """

    origin: _FloatArray
    normal: _FloatArray
    area: _FloatArray
    depth: _FloatArray
    widths: _FloatArray
    growth: float
    clipped: int = 0
    """Columns whose depth was limited by ``max_depth`` rather than by the body."""

    def __post_init__(self) -> None:
        n = self.origin.shape[0]
        for name, expected in (
            ("normal", (n, 3)), ("area", (n,)), ("depth", (n,)),
        ):
            got = np.asarray(getattr(self, name)).shape
            if got != expected:
                msg = f"{name} must have shape {expected}, got {got}"
                raise ValueError(msg)
        if self.widths.ndim != 2 or self.widths.shape[0] != n:
            msg = f"widths must be (n_columns, n_cells), got {self.widths.shape}"
            raise ValueError(msg)

    @property
    def n_columns(self) -> int:
        return int(self.origin.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.widths.shape[1])

    @property
    def wall_spacing(self) -> _FloatArray:
        """First cell width of each column (m) — the resolution that matters."""
        return np.asarray(self.widths[:, 0])

    @property
    def wetted_area(self) -> float:
        return float(self.area.sum())

    def centers(self) -> _FloatArray:
        """Cell-centre depths below the wall, ``(n_columns, n_cells)``."""
        edges = np.cumsum(self.widths, axis=1)
        return np.asarray(edges - 0.5 * self.widths)

    def positions(self) -> _FloatArray:
        """Cell centres in body axes, ``(n_columns, n_cells, 3)``."""
        return np.asarray(
            self.origin[:, None, :] + self.centers()[:, :, None] * self.normal[:, None, :]
        )

    def volume(self) -> _FloatArray:
        """Cell volumes (m³), as area times width.

        A prism of constant cross-section, which is an approximation and a
        known one: the true column narrows or widens with the surface's
        curvature as it goes in. Over a depth small against the local radius of
        curvature the error is second order, and a heat shield's thickness is
        small against its nose radius by design. It is *not* small on a sharp
        leading edge, which is the same place the independent-column assumption
        is weakest, and both errors point the same way — too much material.
        """
        return np.asarray(self.area[:, None] * self.widths)

    def total_volume(self) -> float:
        return float(self.volume().sum())

    def resolves(self, penetration: float) -> bool:
        """Is every column's first cell finer than a given diffusion depth?"""
        return bool(np.all(self.wall_spacing <= float(penetration)))


def solve_growth(depth: _FloatArray, wall_cell: float, n_cells: int) -> _FloatArray:
    r"""Growth ratio giving each column a first cell of ``wall_cell``.

    Solves :math:`w_0(r^n - 1)/(r - 1) = d` for :math:`r`, per column, by
    bisection on the monotone residual.

    This exists because holding the *cell count* fixed across a body holds the
    wrong thing. Column depth varies by orders of magnitude on any real
    vehicle — a waverider's trailing edge is under a millimetre where its keel
    is fifty — so a fixed count makes the thin columns' wall cells
    proportionally thinner, and the thinnest wall cell anywhere sets the
    explicit deposition limit for the *entire* body. On the caret that was a
    factor of a thousand, and it turned a two-minute trajectory into a quarter
    of a million time steps.

    Holding the wall cell fixed instead gives uniform surface resolution, which
    is also what the physics wants: the near-wall gradient does not care how
    thick the material behind it is.
    """
    d = np.atleast_1d(np.asarray(depth, dtype=np.float64))
    n = int(n_cells)
    w0 = float(wall_cell)

    def total(r: _FloatArray) -> _FloatArray:
        uniform = np.abs(r - 1.0) < 1.0e-9
        safe = np.where(uniform, 1.1, r)
        return np.where(uniform, w0 * n, w0 * (safe**n - 1.0) / (safe - 1.0))

    low = np.full(d.shape, 0.5)
    high = np.full(d.shape, 2.0)
    # Depths outside what any admissible ratio can span are clamped to the
    # nearest achievable grid rather than failing: a column shallower than
    # n * w0 simply gets a finer wall than asked for, which is not a problem.
    for _ in range(60):
        middle = 0.5 * (low + high)
        low = np.where(total(middle) < d, middle, low)
        high = np.where(total(middle) < d, high, middle)
    return np.asarray(np.clip(0.5 * (low + high), 0.5, 2.0))


def wall_columns(
    mesh: Any,
    n_cells: int = 24,
    growth: float = 1.15,
    wall_cell: float | None = None,
    max_depth: float | None = None,
    minimum_depth: float = 1.0e-3,
) -> WallColumnGrid:
    """Build inward graded columns from a closed surface mesh.

    Each face contributes one column, starting at its centroid, running along
    its inward normal, and stopping where that normal leaves the body again.

    Parameters
    ----------
    n_cells, growth:
        Cells per column and their inward growth ratio. ``1.15`` puts roughly
        two thirds of the cells in the outer third of the column, which is
        where the gradient is. ``growth`` is ignored when ``wall_cell`` is
        given.
    wall_cell:
        Target first-cell thickness (m), the same for every column, with each
        column's growth ratio solved to match — see :func:`solve_growth`.
        **Prefer this.** A fixed growth ratio ties the wall cell to the column
        depth, so the thinnest part of the body ends up with a wall cell orders
        of magnitude finer than anywhere else, and that one cell sets the time
        step for the whole vehicle.
    max_depth:
        Cap on column length (m). A heat shield is a skin, not the whole
        vehicle: without a cap, a column driven through a metre-thick body
        spends most of its cells on material that never changes temperature.
        ``None`` means no cap, and every column runs to the far wall.
    minimum_depth:
        Columns shorter than this are dropped. A ray leaving through a nearby
        face — which happens at a sharp trailing edge, where the two surfaces
        nearly touch — produces a degenerate column carrying almost no material
        and contributing almost nothing but cost. A millimetre of heat shield
        is not a heat shield.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if not bool(mesh.is_closed):
        msg = (
            "the body must be closed: a column's depth is the distance to where "
            "its normal leaves the material, which an open surface does not define"
        )
        raise ValueError(msg)

    origin = np.asarray(mesh.centroids, dtype=np.float64)
    inward = -np.asarray(mesh.normals, dtype=np.float64)
    area = np.asarray(mesh.areas, dtype=np.float64)

    depth = _first_exit(origin, inward, vertices[faces])
    if max_depth is not None:
        capped = depth > float(max_depth)
        depth = np.minimum(depth, float(max_depth))
    else:
        capped = np.zeros(depth.shape, dtype=bool)

    keep = np.isfinite(depth) & (depth >= float(minimum_depth))
    if not np.any(keep):
        msg = (
            f"no column reached the minimum depth {minimum_depth:g} m; the body "
            f"is thinner than that everywhere, or its normals point inward"
        )
        raise ValueError(msg)

    origin, inward, area, depth = origin[keep], inward[keep], area[keep], depth[keep]
    if wall_cell is None:
        ratios = np.full(depth.shape, float(growth))
    else:
        ratios = solve_growth(depth, float(wall_cell), int(n_cells))
    widths = np.stack(
        [
            graded_widths(float(d), int(n_cells), float(r))
            for d, r in zip(depth, ratios, strict=True)
        ]
    )
    return WallColumnGrid(
        origin=origin, normal=inward, area=area, depth=depth,
        widths=widths, growth=float(np.median(ratios)),
        clipped=int(np.count_nonzero(capped[keep])),
    )


def _first_exit(
    origin: _FloatArray, direction: _FloatArray, triangles: _FloatArray
) -> _FloatArray:
    """Distance from each point to the next surface crossing along its ray.

    Möller–Trumbore, taking the smallest strictly positive hit. The offset
    below which a hit is ignored is not cosmetic: every ray starts *on* a
    triangle — its own — and would otherwise report a depth of zero for the
    entire body.
    """
    n_points = origin.shape[0]
    best = np.full(n_points, np.inf)
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    edge1, edge2 = v1 - v0, v2 - v0

    scale = float(np.max(np.abs(triangles))) or 1.0
    epsilon = 1.0e-7 * scale

    chunk = max(1, int(2.0e7 // max(triangles.shape[0], 1)))
    for start in range(0, n_points, chunk):
        stop = start + chunk
        rays = direction[start:stop]
        pvec = np.cross(rays[:, None, :], edge2[None, :, :])
        det = np.einsum("ptj,tj->pt", pvec, edge1)
        parallel = np.abs(det) < 1.0e-14
        inv_det = np.where(parallel, 0.0, 1.0 / np.where(parallel, 1.0, det))

        tvec = origin[start:stop, None, :] - v0[None, :, :]
        u = np.einsum("ptj,ptj->pt", tvec, pvec) * inv_det
        qvec = np.cross(tvec, edge1[None, :, :])
        v = np.einsum("ptj,pj->pt", qvec, rays) * inv_det
        t = np.einsum("ptj,tj->pt", qvec, edge2) * inv_det

        hit = (
            ~parallel & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (u + v <= 1.0)
            & (t > epsilon)
        )
        best[start:stop] = np.where(hit, t, np.inf).min(axis=1)
    return best
