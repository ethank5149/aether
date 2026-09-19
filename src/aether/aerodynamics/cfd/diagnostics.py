"""Physical-realism checks on a converged CFD surface solution.

These answer questions that need no reference solution and no experiment: a
wall pressure above the Rayleigh pitot limit is not "in disagreement with
theory", it is unattainable by any flow at that Mach number, so seeing one is
proof the discretisation has failed rather than evidence about the vehicle.

That property is what makes them useful as a *sizing* criterion. Grid
convergence tells you a solution has stopped changing; it cannot tell you the
level it converged to is physical, and on a captured bow shock those are
different questions.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from aether.aerodynamics.closure import rayleigh_pitot_cp_max

_FloatArray = NDArray[np.float64]

__all__ = [
    "PitotViolation",
    "SurfaceQuality",
    "pitot_limit_violation",
    "surface_pressure_coefficient",
    "surface_quality",
]


def surface_pressure_coefficient(
    surface: Path | str, freestream_pressure: float, mach: float, gamma: float = 1.4
) -> tuple[_FloatArray, _FloatArray]:
    """Read an SU2 surface file and return ``(points, Cp)``.

    SU2 writes conservative variables to ``surface_flow.csv``, not pressure, so
    the equation of state is applied here. Verified against SU2's own
    per-marker force integration: reproducing ``CFx(vehicle)`` from these
    values matched the solver's history column to all six figures printed,
    which also establishes that SU2 integrates *gauge* pressure per marker.
    """
    with Path(surface).open() as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"{surface} is empty")
    header = [h.strip().strip('"') for h in rows[0]]
    column = {name: index for index, name in enumerate(header)}
    required = ("x", "y", "z", "Density", "Momentum_x", "Momentum_y", "Momentum_z", "Energy")
    missing = [name for name in required if name not in column]
    if missing:
        raise ValueError(f"{surface} lacks {missing}; needs conservative variables")
    data = np.array([[float(r[column[n]]) for n in required] for r in rows[1:]])
    points = data[:, :3]
    density, mx, my, mz, energy = data[:, 3], data[:, 4], data[:, 5], data[:, 6], data[:, 7]
    pressure = (gamma - 1.0) * (energy - 0.5 * (mx**2 + my**2 + mz**2) / density)
    dynamic = 0.5 * gamma * freestream_pressure * mach**2
    return points, (pressure - freestream_pressure) / dynamic


@dataclass(frozen=True)
class PitotViolation:
    """How much of a wall solution is above the attainable stagnation pressure."""

    fraction: float
    """Fraction of wall nodes exceeding the Rayleigh pitot limit."""
    mean_excess: float
    """Mean Cp above the limit, over the violating nodes."""
    peak: float
    """Largest wall Cp anywhere."""
    limit: float
    """The Rayleigh pitot limit itself, for context."""
    extent: tuple[float, float, float, float]
    """``(x_min, x_max, r_min, r_max)`` of the violating region (m)."""
    nodes: int
    """Number of violating nodes."""

    @property
    def resolved(self) -> bool:
        """No node exceeds the limit by more than round-off."""
        return self.nodes == 0

    def summary(self) -> str:
        if self.resolved:
            return f"no pitot-limit violation (peak Cp {self.peak:.4f} vs limit {self.limit:.4f})"
        return (
            f"{100 * self.fraction:.2f}% of wall nodes above the pitot limit "
            f"{self.limit:.4f} (peak {self.peak:.4f}, mean excess {self.mean_excess:.3f}) "
            f"over x={self.extent[0]:.4f}..{self.extent[1]:.4f} m"
        )


def pitot_limit_violation(
    surface: Path | str,
    freestream_pressure: float,
    mach: float,
    gamma: float = 1.4,
    tolerance: float = 1e-3,
) -> PitotViolation:
    """Fraction of the wall solution that is physically unattainable.

    A blunt body in steady flow cannot produce a surface pressure above the
    stagnation pressure behind a normal shock. Exceeding it means the captured
    shock is spread over too few cells and the reconstruction has overshot, and
    the size of the exceedance is a direct measure of that under-resolution —
    available from a single run, with no grid sequence and no reference.

    Measured on this pipeline at Mach 8: a sphere-cone with 1.5 cells across
    its 7.5 mm nose standoff put 8.05 % of its wall nodes above the limit, mean
    excess Cp 0.73, peak 3.58 against a limit of 1.83 — an inflation of about
    6.5 % in forebody :math:`C_A`, ten times that run's statistical bar. Use
    with :func:`~aether.aerodynamics.cfd.meshing.shock_layer_cell_size` to pick
    the cell size, then confirm with this rather than assuming.

    Parameters
    ----------
    surface:
        SU2 ``surface_flow.csv`` from the converged run.
    freestream_pressure:
        Static pressure (Pa), for both the gauge and the non-dimensionalisation.
    tolerance:
        Relative slack on the limit, so that a node sitting exactly on it is
        not reported.
    """
    points, cp = surface_pressure_coefficient(surface, freestream_pressure, mach, gamma)
    limit = rayleigh_pitot_cp_max(mach, gamma)
    over = cp > limit * (1.0 + tolerance)
    if not np.any(over):
        return PitotViolation(0.0, 0.0, float(cp.max()), float(limit), (0.0, 0.0, 0.0, 0.0), 0)
    x = points[over, 0]
    radius = np.hypot(points[over, 1], points[over, 2])
    return PitotViolation(
        fraction=float(np.mean(over)),
        mean_excess=float(np.mean(cp[over] - limit)),
        peak=float(cp.max()),
        limit=float(limit),
        extent=(float(x.min()), float(x.max()), float(radius.min()), float(radius.max())),
        nodes=int(np.count_nonzero(over)),
    )


@dataclass(frozen=True)
class SurfaceQuality:
    """Triangle shape quality of a wall mesh, before it is ever extruded."""

    minimum: float
    """Worst triangle quality, :math:`4\\sqrt{3}A/\\sum \\ell^2` — 1 equilateral, 0 degenerate."""
    first_percentile: float
    """Quality below which 1 % of triangles fall."""
    median: float
    slivers: int
    """Triangles below the usable threshold."""
    worst_location: tuple[float, float]
    """``(x, r)`` of the worst triangle's centroid (m) — where to go looking."""
    faces: int

    @property
    def usable(self) -> bool:
        """No sliver triangles."""
        return self.slivers == 0

    def summary(self) -> str:
        if self.usable:
            return (
                f"{self.faces} faces, worst quality {self.minimum:.3f}, "
                f"median {self.median:.3f} — no slivers"
            )
        return (
            f"{self.slivers} of {self.faces} faces are slivers (worst "
            f"{self.minimum:.4f}, median {self.median:.3f}), worst at "
            f"x={self.worst_location[0]:.4f} r={self.worst_location[1]:.4f}"
        )


def surface_quality(mesh: object, threshold: float = 0.1) -> SurfaceQuality:
    """Shape quality of a wall triangulation.

    Worth checking *before* extruding, because a prism stack multiplies a bad
    triangle by the layer count and the failure surfaces far from its cause. A
    Mach 8 sphere-cone with a 10 mm shoulder fillet on a 785 mm body carried
    432 needle triangles — 102 mm long, 1.9 mm wide, quality 0.033 — because
    the fillet is resolved across its arc by curvature but around the body only
    by the global maximum size. Extruded, those became the 34 prisms SU2
    reported as distorted, with a control-volume sub-volume ratio of 13131 and
    29 non-physical points before the first iteration.

    The cure is geometric, not numerical: a needle of width :math:`w` and
    length :math:`\\ell` has quality about :math:`\\sqrt{3}w/\\ell`, so a
    feature far smaller than the local mesh size cannot be meshed isotropically
    at any affordable cost. On that body, opening the fillet from 10 mm to
    60 mm removed every sliver and used *fewer* faces.

    Parameters
    ----------
    mesh:
        Anything with ``vertices`` and ``faces`` arrays.
    threshold:
        Quality below which a triangle is counted a sliver.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)  # type: ignore[attr-defined]
    faces = np.asarray(mesh.faces, dtype=np.int64)  # type: ignore[attr-defined]
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    edges = np.stack(
        [
            np.linalg.norm(b - a, axis=1),
            np.linalg.norm(c - b, axis=1),
            np.linalg.norm(a - c, axis=1),
        ],
        axis=1,
    )
    total = np.sum(edges**2, axis=1)
    quality = np.divide(
        4.0 * np.sqrt(3.0) * area, total, out=np.zeros_like(area), where=total > 0.0
    )
    worst = int(np.argmin(quality))
    centroid = (a[worst] + b[worst] + c[worst]) / 3.0
    return SurfaceQuality(
        minimum=float(quality.min()),
        first_percentile=float(np.percentile(quality, 1.0)),
        median=float(np.median(quality)),
        slivers=int(np.count_nonzero(quality < threshold)),
        worst_location=(float(centroid[0]), float(np.hypot(centroid[1], centroid[2]))),
        faces=int(faces.shape[0]),
    )
