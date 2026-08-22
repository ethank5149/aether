"""The CFD solver as a table source, and the grid study that licenses it.

A CFD result that has not been shown to be grid-converged is a number, not a
measurement. :class:`GridConvergence` implements the standard procedure —
three or more systematically refined meshes, an *observed* order of accuracy
solved for rather than assumed, Richardson extrapolation to zero spacing, and
a Grid Convergence Index as an error band. Reporting the GCI is what turns
"the drag is 0.167" into "the drag is 0.173 ± 0.004", and the second is the
only one of those a trajectory should be built on.

The observed order matters on its own. A second-order scheme on a smooth
problem should recover :math:`p \\approx 2`; a shock-capturing scheme on a
flow with a shock in it will not, because a captured discontinuity is first
order however the interior is discretised. Seeing :math:`p` come out near 1
on a shocked case is the expected result and seeing it come out near 2 on
one would be the surprising one.

Scope
-----

The axisymmetric formulation is a **zero-incidence** method. A body of
revolution at :math:`\\alpha = 0` is a two-dimensional problem and that is
the entire reason the transonic band is affordable here; at
:math:`\\alpha \\neq 0` it is not, and no amount of post-processing an
axisymmetric solution recovers a normal force. :meth:`EulerSolver.solve`
therefore refuses a non-zero incidence rather than returning something
plausible, and the composite table records which region of the
:math:`(M, \\alpha)` plane it covers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from aether.aerodynamics.cfd.meshing import (
    BodyProfile,
    DomainSizing,
    MeshResult,
    axisymmetric_domain,
)
from aether.aerodynamics.cfd.su2 import SU2Result, SU2Settings, run_su2
from aether.aerodynamics.tables import Coefficients

__all__ = ["EulerSolver", "GridConvergence", "grid_convergence"]

_FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GridConvergence:
    """Richardson extrapolation and a Grid Convergence Index.

    Attributes
    ----------
    spacing:
        Representative cell sizes, **coarsest last**.
    values:
        The quantity on each mesh, in the same order.
    observed_order:
        :math:`p`, solved for from the three finest values.
    extrapolated:
        The zero-spacing limit.
    gci_fine:
        Grid Convergence Index on the finest mesh, as a fraction. Read it as
        a 95 %-confidence error band, which is what the 1.25 safety factor is
        calibrated to mean.
    """

    spacing: _FloatArray
    values: _FloatArray
    observed_order: float
    extrapolated: float
    gci_fine: float
    monotone: bool

    @property
    def relative_error(self) -> float:
        """Fractional gap between the finest mesh and the extrapolated limit."""
        if self.extrapolated == 0.0:
            return float("nan")
        return float(self.values[0] / self.extrapolated - 1.0)

    def summary(self) -> str:
        return (
            f"p = {self.observed_order:.3f}, extrapolated {self.extrapolated:.6f}, "
            f"finest {self.values[0]:.6f} ({100 * self.relative_error:+.2f} %), "
            f"GCI {100 * self.gci_fine:.2f} %"
            + ("" if self.monotone else "  [NOT monotone — p is not meaningful]")
        )


def grid_convergence(
    spacing: NDArray[np.float64] | list[float],
    values: NDArray[np.float64] | list[float],
    safety_factor: float = 1.25,
) -> GridConvergence:
    """Observed order, Richardson limit and GCI from three or more meshes.

    Implements the procedure of Celik et al. (the ASME *Journal of Fluids
    Engineering* editorial policy), which allows **non-constant refinement
    ratios** — solving

    .. math::

        p = \\frac{1}{\\ln r_{21}}
        \\left|\\ln\\left|\\frac{\\epsilon_{32}}{\\epsilon_{21}}\\right| + q(p)\\right|,
        \\qquad
        q(p) = \\ln\\frac{r_{21}^p - s}{r_{32}^p - s}

    by fixed point. That generality is not decoration: an unstructured
    generator asked for cells half the size does not return four times as
    many, so the ratios are never exactly what was requested and a formula
    that assumes they are will report an order that is off by a lot.

    Non-monotone convergence — the three values not ordered — is *reported*,
    not repaired. It means the asymptotic range has not been reached and the
    extrapolation is not valid, and that is information rather than a
    failure.
    """
    h = np.asarray(spacing, dtype=np.float64)
    f = np.asarray(values, dtype=np.float64)
    if h.shape != f.shape or h.size < 3:
        msg = f"need at least 3 matching meshes, got shapes {h.shape} and {f.shape}"
        raise ValueError(msg)
    order = np.argsort(h)  # finest first
    h, f = h[order], f[order]
    if np.any(np.diff(h) <= 0.0):
        msg = "mesh spacings must be distinct"
        raise ValueError(msg)

    h1, h2, h3 = float(h[0]), float(h[1]), float(h[2])
    f1, f2, f3 = float(f[0]), float(f[1]), float(f[2])
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f2 - f1, f3 - f2

    if e21 == 0.0:
        return GridConvergence(h, f, float("nan"), f1, 0.0, True)
    ratio = e32 / e21
    monotone = ratio > 0.0
    s = float(np.sign(ratio)) if ratio != 0.0 else 1.0

    p = 2.0
    for _ in range(200):
        q = np.log((r21**p - s) / (r32**p - s))
        updated = abs(np.log(abs(ratio)) + q) / np.log(r21)
        if abs(updated - p) < 1e-12:
            p = float(updated)
            break
        # Damped, because the map is not a contraction for ratios near unity
        # and an undamped iteration oscillates without converging.
        p = float(0.5 * p + 0.5 * updated)

    extrapolated = (r21**p * f1 - f2) / (r21**p - 1.0)
    relative = abs(e21 / f1) if f1 != 0.0 else float("nan")
    gci = float(safety_factor * relative / (r21**p - 1.0))
    return GridConvergence(
        spacing=h,
        values=f,
        observed_order=float(p),
        extrapolated=float(extrapolated),
        gci_fine=gci,
        monotone=bool(monotone),
    )


@dataclass
class EulerSolver:
    """Axisymmetric Euler through SU2, presented as a coefficient source.

    Attributes
    ----------
    profile:
        The body's meridian outline.
    workspace:
        Where meshes and cases are written. Kept, not temporary: a CFD point
        that cannot be re-examined afterwards is not reproducible, and the
        directory is the record.
    reference_area, reference_length:
        Shared with the panel tables so the two can be spliced without
        renormalising.
    remesh_per_mach:
        Build a mesh sized for each Mach number rather than reusing one. The
        shock-envelope refinement follows the Mach cone, so a single mesh is
        either wasteful at high Mach or under-resolved at low. Meshing costs
        under a second; a wrong mesh costs the answer.
    """

    profile: BodyProfile
    workspace: Path
    reference_area: float
    reference_length: float
    sizing: DomainSizing = field(default_factory=DomainSizing)
    settings: SU2Settings = field(default_factory=SU2Settings)
    temperature: float = 288.15
    pressure: float = 101325.0
    remesh_per_mach: bool = True
    keep_cases: bool = True
    processes: int = 1
    """MPI ranks per case. See the note in :func:`~aether.aerodynamics.cfd.su2.run_su2`."""
    name: str = "su2-euler-axisymmetric"
    _meshes: dict[str, MeshResult] = field(default_factory=dict, repr=False)
    _results: dict[float, SU2Result] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

    def mesh_for(self, mach: float, sizing: DomainSizing | None = None) -> MeshResult:
        """Build (or reuse) a mesh sized for a Mach number."""
        chosen = sizing if sizing is not None else self.sizing
        # Bucket by Mach so that a sweep of forty points does not build forty
        # nearly identical meshes; the envelope width changes slowly above
        # Mach 2 and not at all below 1.
        bucket = f"{self._bucket(mach):.2f}-{chosen.wall_size:.6g}"
        cached = self._meshes.get(bucket)
        if cached is not None:
            return cached
        built = axisymmetric_domain(
            self.profile,
            self.workspace / "mesh" / f"{self.profile.name}-m{bucket}.su2",
            mach=mach,
            sizing=chosen,
        )
        self._meshes[bucket] = built
        return built

    @staticmethod
    def _bucket(mach: float) -> float:
        """Mach values that share a mesh."""
        m = float(mach)
        if m < 1.2:
            return 1.0
        if m < 2.0:
            return 1.5
        if m < 3.5:
            return 2.5
        if m < 6.0:
            return 4.5
        return 8.0

    def run(self, mach: float, sizing: DomainSizing | None = None) -> SU2Result:
        """Solve one Mach number and return the full CFD result."""
        key = round(float(mach), 6)
        if sizing is None and key in self._results:
            return self._results[key]
        mesh = self.mesh_for(mach, sizing)
        result = run_su2(
            mesh,
            self.profile,
            float(mach),
            self.workspace / "cases" / f"m{mach:.4f}",
            self.settings,
            reference_area=self.reference_area,
            reference_length=self.reference_length,
            temperature=self.temperature,
            pressure=self.pressure,
            keep_output=self.keep_cases,
            processes=self.processes,
        )
        if sizing is None:
            self._results[key] = result
        return result

    def solve(self, mach: float, alpha: float) -> Coefficients:
        """Coefficients at zero incidence. Refuses anything else.

        The normal force and pitching moment are returned as zero rather than
        as ``nan``, because at :math:`\\alpha = 0` on a body of revolution
        they are zero *by symmetry* — that is a result, not a gap.
        """
        if abs(float(alpha)) > 1e-12:
            msg = (
                f"the axisymmetric Euler solver is a zero-incidence method and "
                f"cannot evaluate alpha = {np.rad2deg(alpha):.3f} deg. A body of "
                f"revolution at incidence is a three-dimensional problem; "
                f"build the incidence dependence with the panel method above "
                f"its Mach floor, or run a three-dimensional case."
            )
            raise ValueError(msg)
        result = self.run(mach)
        if not result.converged:
            msg = (
                f"SU2 did not converge at Mach {mach:g} (residual "
                f"{result.residual:.3f} after {result.iterations} iterations); "
                f"case kept at {result.directory}"
            )
            raise RuntimeError(msg)
        return Coefficients(
            axial=result.forebody_axial + result.base_axial,
            normal=0.0,
            pitching_moment=0.0,
        )

    def grid_study(
        self,
        mach: float,
        factors: tuple[float, ...] = (2.0, 1.41, 1.0, 0.71),
        quantity: str = "forebody_axial",
    ) -> tuple[GridConvergence, list[SU2Result]]:
        """Run the same case at several resolutions and extrapolate.

        ``factors`` multiply every cell size, so a factor of 0.71 is roughly
        twice the cell count in two dimensions. They are listed coarsest
        first because that is the order they should be run in: a coarse case
        that diverges tells you the settings are wrong before a fine one
        spends an hour finding out.
        """
        results: list[SU2Result] = []
        for factor in factors:
            results.append(self.run(mach, self.sizing.scaled(factor)))
        spacing = np.array(
            [r.mesh.representative_size if r.mesh else np.nan for r in results]
        )
        values = np.array([float(getattr(r, quantity)) for r in results])
        return grid_convergence(spacing, values), results
