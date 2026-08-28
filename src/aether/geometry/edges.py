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

import numpy as np
from numpy.typing import NDArray

__all__ = ["round_corners"]

_FloatArray = NDArray[np.float64]


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
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 3:
        msg = f"need at least 3 points of shape (n, d), got {p.shape}"
        raise ValueError(msg)
    n = p.shape[0]
    r = np.broadcast_to(np.asarray(radii, dtype=np.float64), (n,)).copy()
    if np.any(r < 0.0):
        msg = "radii must be non-negative"
        raise ValueError(msg)
    if not np.any(r > 0.0):
        return p

    indices = range(n) if closed else range(1, n - 1)
    # Tangent lengths are computed for every corner first, so a conflict
    # between two fillets sharing an edge is detected before any of the
    # geometry is rebuilt.
    tangents = np.zeros(n)
    for i in indices:
        if r[i] <= 0.0:
            continue
        before, after = p[(i - 1) % n], p[(i + 1) % n]
        u, v = before - p[i], after - p[i]
        lu, lv = np.linalg.norm(u), np.linalg.norm(v)
        if lu < 1e-14 or lv < 1e-14:
            msg = f"vertex {i} coincides with a neighbour; no corner to round"
            raise ValueError(msg)
        cosine = float(np.clip(np.dot(u / lu, v / lv), -1.0, 1.0))
        half = 0.5 * np.arccos(cosine)
        if half <= 1e-9 or half >= 0.5 * np.pi - 1e-9:
            continue  # straight through, or doubled back: nothing to fillet
        tangents[i] = r[i] / np.tan(half)

    for i in indices:
        if tangents[i] <= 0.0:
            continue
        for neighbour in ((i - 1) % n, (i + 1) % n):
            edge = float(np.linalg.norm(p[neighbour] - p[i]))
            if tangents[i] + tangents[neighbour] > edge + 1e-12:
                msg = (
                    f"fillet radius {r[i]:g} at vertex {i} needs "
                    f"{tangents[i]:g} m of tangent, and with {tangents[neighbour]:g} m "
                    f"at vertex {neighbour} that exceeds the {edge:g} m edge between "
                    f"them; reduce the radius"
                )
                raise ValueError(msg)

    out: list[_FloatArray] = []
    for i in range(n):
        if tangents[i] <= 0.0:
            out.append(p[i][None, :])
            continue
        before, after = p[(i - 1) % n], p[(i + 1) % n]
        u = (before - p[i]) / np.linalg.norm(before - p[i])
        v = (after - p[i]) / np.linalg.norm(after - p[i])
        start, end = p[i] + u * tangents[i], p[i] + v * tangents[i]

        bisector = u + v
        norm = np.linalg.norm(bisector)
        if norm < 1e-14:
            out.append(p[i][None, :])
            continue
        cosine = float(np.clip(np.dot(u, v), -1.0, 1.0))
        half = 0.5 * np.arccos(cosine)
        centre = p[i] + (bisector / norm) * (r[i] / np.sin(half))

        # Swept in the plane of the corner, by rotating the start radius toward
        # the end one — which works in any dimension without needing an axis.
        a, b = start - centre, end - centre
        angle = float(np.arccos(np.clip(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0)))
        perpendicular = b - a * (np.dot(a, b) / np.dot(a, a))
        if np.linalg.norm(perpendicular) < 1e-14:
            out.append(np.stack([start, end]))
            continue
        perpendicular /= np.linalg.norm(perpendicular)
        theta = np.linspace(0.0, angle, max(2, int(samples)))
        arc = centre + np.cos(theta)[:, None] * a + np.sin(theta)[:, None] * (
            perpendicular * np.linalg.norm(a)
        )
        out.append(arc)
    return np.concatenate(out, axis=0)
