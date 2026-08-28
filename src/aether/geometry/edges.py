r"""Edge blunting: replacing a geometric crease with a radius you chose.

A sharp edge is a modelling convenience, not a shape anything can be built
from or flown. Stagnation heating on a leading edge goes as
:math:`q \propto 1/\sqrt{R}`, so a mathematically sharp edge carries infinite
flux; every real hypersonic leading edge is blunted, typically to between one
and five millimetres, and that radius is a **design variable** — it trades
heating against the drag and the shock-attachment the sharpness was bought for.

So it belongs in the parameterisation rather than in a mesher's tolerance.
Each vehicle class exposes a radius per *edge family* — a waverider's leading
edges and its ridge blunt independently, because they do different jobs and
see different environments — and each defaults to zero, which reproduces the
sharp shape exactly.

What this does and does not claim
---------------------------------

:func:`round_corners` replaces a corner with a **circular arc tangent to both
adjacent edges**. That is the right first-order treatment and it is what a
drawing means by a fillet radius. It is not a Riemann-mapped or minimum-drag
leading edge, and on a waverider it does spill some flow: blunting a leading
edge moves the shock off it by an amount of order the radius, which is exactly
the trade being made and is why the radius wants to be small and *tunable*
rather than absent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["Arc", "Contour", "Segment", "round_corners", "rounded_contour"]

_FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Segment:
    """A straight edge between two points."""

    start: _FloatArray
    end: _FloatArray


@dataclass(frozen=True)
class Arc:
    """A circular arc, given by its endpoints and centre.

    Three points fix the arc because a fillet always subtends less than a
    half-turn, which is the same convention OCC's ``addCircleArc`` uses.
    """

    start: _FloatArray
    end: _FloatArray
    centre: _FloatArray

    @property
    def radius(self) -> float:
        return float(np.linalg.norm(self.start - self.centre))


@dataclass(frozen=True)
class Contour:
    """An outline made of *exact* line and arc primitives.

    The point of the type. A sampled polyline looks like the same shape and is
    not the same *model*: every sample becomes an edge in the B-rep, every edge
    becomes a face when the outline is revolved or lofted, and the mesher then
    inherits the sampling it was supposed to be free of. Measured on the bodies
    here, sampled outlines gave the sphere-cone 107 CAD faces, the biconic 146,
    and the caret waverider 558 curves — some of them 5.7 um long, some of zero
    length — where the exact outlines have of order ten each.

    That is not an efficiency argument. It is why mesh size fields, ``size_min``,
    the 2D algorithm and OCC shape healing all had *no effect whatever* on the
    waverider's mesh quality: the faces were the constraint, and no amount of
    sizing can mesh a 5.7 um face well.
    """

    primitives: tuple[Segment | Arc, ...]
    closed: bool = True

    def __post_init__(self) -> None:
        if not self.primitives:
            raise ValueError("a contour needs at least one primitive")

    def points(self, samples: int = 16) -> _FloatArray:
        """Sample the outline, for callers that still want a polyline.

        Sampling *here* is safe — it is a view of the model, not the model.
        """
        out: list[_FloatArray] = []
        for item in self.primitives:
            if isinstance(item, Segment):
                out.append(item.start[None, :])
                continue
            first = item.start - item.centre
            last = item.end - item.centre
            angle = float(
                np.arccos(
                    np.clip(
                        np.dot(first, last)
                        / (np.linalg.norm(first) * np.linalg.norm(last)),
                        -1.0,
                        1.0,
                    )
                )
            )
            perpendicular = last - first * (np.dot(first, last) / np.dot(first, first))
            length = float(np.linalg.norm(perpendicular))
            if length < 1e-14:
                out.append(item.start[None, :])
                continue
            perpendicular = perpendicular / length
            theta = np.linspace(0.0, angle, max(2, int(samples)))[:-1]
            out.append(
                item.centre
                + np.cos(theta)[:, None] * first
                + np.sin(theta)[:, None] * (perpendicular * np.linalg.norm(first))
            )
        sampled = np.concatenate(out, axis=0)
        if not self.closed:
            sampled = np.vstack([sampled, self.primitives[-1].end[None, :]])
        return sampled


def _fillets(
    points: _FloatArray, radii: _FloatArray, closed: bool
) -> tuple[_FloatArray, dict[int, tuple[_FloatArray, _FloatArray, _FloatArray]]]:
    """Tangent lengths and arc frames for every corner.

    Shared by :func:`round_corners` and :func:`rounded_contour` so the sampled
    outline and the exact one cannot describe different shapes.

    Returns ``(tangents, frames)`` with ``frames[i] = (start, end, centre)``
    for each rounded corner.
    """
    n = points.shape[0]
    indices = range(n) if closed else range(1, n - 1)

    # Tangent lengths are computed for every corner first, so a conflict
    # between two fillets sharing an edge is detected before any of the
    # geometry is rebuilt.
    tangents = np.zeros(n)
    for i in indices:
        if radii[i] <= 0.0:
            continue
        before, after = points[(i - 1) % n], points[(i + 1) % n]
        u, v = before - points[i], after - points[i]
        lu, lv = np.linalg.norm(u), np.linalg.norm(v)
        if lu < 1e-14 or lv < 1e-14:
            msg = f"vertex {i} coincides with a neighbour; no corner to round"
            raise ValueError(msg)
        cosine = float(np.clip(np.dot(u / lu, v / lv), -1.0, 1.0))
        half = 0.5 * np.arccos(cosine)
        if half <= 1e-9 or half >= 0.5 * np.pi - 1e-9:
            continue  # straight through, or doubled back: nothing to fillet
        tangents[i] = radii[i] / np.tan(half)

    for i in indices:
        if tangents[i] <= 0.0:
            continue
        for neighbour in ((i - 1) % n, (i + 1) % n):
            edge = float(np.linalg.norm(points[neighbour] - points[i]))
            if tangents[i] + tangents[neighbour] > edge + 1e-12:
                msg = (
                    f"fillet radius {radii[i]:g} at vertex {i} needs "
                    f"{tangents[i]:g} m of tangent, and with {tangents[neighbour]:g} m "
                    f"at vertex {neighbour} that exceeds the {edge:g} m edge between "
                    f"them; reduce the radius"
                )
                raise ValueError(msg)

    frames: dict[int, tuple[_FloatArray, _FloatArray, _FloatArray]] = {}
    for i in range(n):
        if tangents[i] <= 0.0:
            continue
        before, after = points[(i - 1) % n], points[(i + 1) % n]
        u = (before - points[i]) / np.linalg.norm(before - points[i])
        v = (after - points[i]) / np.linalg.norm(after - points[i])
        bisector = u + v
        norm = float(np.linalg.norm(bisector))
        if norm < 1e-14:
            continue
        cosine = float(np.clip(np.dot(u, v), -1.0, 1.0))
        half = 0.5 * np.arccos(cosine)
        frames[i] = (
            points[i] + u * tangents[i],
            points[i] + v * tangents[i],
            points[i] + (bisector / norm) * (radii[i] / np.sin(half)),
        )
    return tangents, frames


def _validated(
    points: _FloatArray, radii: float | _FloatArray
) -> tuple[_FloatArray, _FloatArray]:
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 3:
        msg = f"need at least 3 points of shape (n, d), got {p.shape}"
        raise ValueError(msg)
    r = np.broadcast_to(np.asarray(radii, dtype=np.float64), (p.shape[0],)).copy()
    if np.any(r < 0.0):
        msg = "radii must be non-negative"
        raise ValueError(msg)
    return p, r


def rounded_contour(
    points: _FloatArray, radii: float | _FloatArray, closed: bool = True
) -> Contour:
    """The outline of :func:`round_corners`, as exact primitives.

    Same geometry, different *model*: arcs stay arcs instead of becoming a
    fan of short chords. Prefer this wherever the outline is going to be
    revolved or lofted into a solid — see :class:`Contour` for what sampling
    costs there.
    """
    p, r = _validated(points, radii)
    n = p.shape[0]
    _, frames = _fillets(p, r, closed)

    # Each vertex contributes an arc when it is filleted and a bare point when
    # it is not; straight edges are whatever is left between them.
    ends: list[tuple[_FloatArray, _FloatArray, Arc | None]] = []
    for i in range(n):
        if i in frames:
            begin, finish, centre = frames[i]
            ends.append((begin, finish, Arc(begin, finish, centre)))
        else:
            ends.append((p[i], p[i], None))

    primitives: list[Segment | Arc] = []
    last = n if closed else n - 1
    for i in range(n):
        arc = ends[i][2]
        if arc is not None:
            primitives.append(arc)
        if i >= last:
            break
        here, there = ends[i][1], ends[(i + 1) % n][0]
        if float(np.linalg.norm(there - here)) > 1e-12:
            primitives.append(Segment(here, there))
    return Contour(tuple(primitives), closed=closed)


def round_corners(
    points: _FloatArray,
    radii: float | _FloatArray,
    samples: int = 6,
    closed: bool = True,
) -> _FloatArray:
    """Replace each corner of a polygon or polyline with a tangent arc.

    Parameters
    ----------
    points:
        ``(n, d)`` vertices, in order. For ``closed`` the last does **not**
        repeat the first.
    radii:
        One radius per vertex, or a scalar for all of them. Zero leaves that
        corner sharp, so a shape can blunt its leading edges and keep its
        ridge — which is the usual case.
    samples:
        Points per arc. Six is enough for a fillet whose radius is small
        against the edges it joins; the arc is exact in the limit and the
        error is second order in the subtended angle.
    closed:
        Round every vertex. Open polylines keep their endpoints, since those
        are boundaries rather than corners.

    Raises
    ------
    ValueError
        If a radius needs more tangent length than its edges can give. That is
        a real geometric conflict — two fillets eating the same edge — and
        silently shrinking the radius would return a shape the caller did not
        ask for and would not know it had.
    """
    p, r = _validated(points, radii)
    if not np.any(r > 0.0):
        return p
    # Sampled from the exact outline, so the two can never disagree about the
    # shape — only about how finely it is written down.
    return rounded_contour(p, r, closed=closed).points(samples=max(2, int(samples)))
