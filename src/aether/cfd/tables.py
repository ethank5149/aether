"""An :class:`AeroTable` built from the runs the queue actually solved.

``sweep.py`` already turns a body and a grid into a table -- by launching its
own blocking solves through ``SU2Solver``. Every coefficient this project has
measured came from the other path, the queue and its worker, and until now
nothing could read those into a table at all. The two execution paths were
never unified, so the one with the data had no reader
→ :doc:`the-simplifications-inventory`.

This closes the gap the inventory calls item A: **the CFD reached nothing**.
``grep -rn "from aether.cfd" trajectory/ flight/ guidance/`` returned
nothing, so every aerodynamic coefficient in the flight model was a literal
written by hand, and a model whose aerodynamic input is a constant cannot be
wrong in a way that shows up.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

_FloatArray = NDArray[np.float64]

__all__ = ["TablePoint", "table_from_registry"]


@dataclass(frozen=True)
class TablePoint:
    """One solved grid point, with the provenance that makes it checkable."""

    mach: float
    alpha: float
    run_id: str
    axial: float
    normal: float
    pitching_moment: float
    sigma: float
    periods: float
    settled: bool
    residual: float
    iterations: int
    elements: int


def _matches(spec: Any, body: str, geometry: dict[str, float], altitude_km: float) -> bool:
    if spec.body != body:
        return False
    if abs(float(spec.altitude_km) - float(altitude_km)) > 1.0e-9:
        return False
    keys = set(geometry) | set(spec.geometry)
    return all(
        abs(float(spec.geometry.get(k, float("nan"))) - float(geometry.get(k, float("nan"))))
        <= 1.0e-9
        for k in keys
    )


def table_from_registry(
    body: str,
    geometry: dict[str, float],
    *,
    altitude_km: float,
    name: str | None = None,
    root: Path | None = None,
    require_settled: bool = True,
) -> Any:
    """Assemble every solved run for one body and condition into an ``AeroTable``.

    Points the queue has not solved come back **NaN**, and
    :attr:`~aether.aerodynamics.tables.AeroTable.complete` reports it. A table
    that silently interpolated across a hole would hide exactly the gap it is
    meant to expose.

    ``require_settled`` drops runs whose force never stopped moving. Off, they
    are included and their ``settled`` flag is carried in the metadata, which is
    what a grid-convergence study wants and what a trajectory must not have.

    Sign convention: negative incidence is not assumed by symmetry. A body that
    is axisymmetric has it and one that is not does not, and the table would
    rather be half empty than half invented.
    """
    from aether.aerodynamics.tables import AeroTable
    from aether.cfd.coefficients import settled_coefficients
    from aether.cfd.geometries import reference_frame
    from aether.cfd.registry import load_all

    points: list[TablePoint] = []
    for record in load_all():
        if record.state != "done" or record.case_dir is None:
            continue
        if not _matches(record.spec, body, geometry, altitude_km):
            continue
        forces = settled_coefficients(
            Path(record.case_dir) / "history.csv",
            window=None,
            tolerance=record.spec.settle_tolerance or None,
        )
        if forces is None:
            continue
        if require_settled and not forces.usable:
            continue
        points.append(
            TablePoint(
                mach=float(record.spec.mach),
                alpha=float(record.spec.alpha_deg),
                run_id=record.run_id,
                axial=forces.axial,
                normal=forces.normal,
                pitching_moment=forces.pitching_moment,
                sigma=forces.sigma,
                periods=forces.periods,
                settled=forces.settled,
                residual=forces.residual,
                iterations=forces.iterations,
                elements=int(record.elements or 0),
            )
        )

    machs = np.array(sorted({p.mach for p in points}), dtype=np.float64)
    alphas = np.array(sorted({p.alpha for p in points}), dtype=np.float64)
    if machs.size == 0 or alphas.size == 0:
        msg = (
            f"no solved runs for {body} at {altitude_km:g} km with that geometry. "
            "A table cannot be invented from an empty queue -- run the sweep."
        )
        raise ValueError(msg)

    shape = (machs.size, alphas.size)
    axial = np.full(shape, np.nan)
    normal = np.full(shape, np.nan)
    moment = np.full(shape, np.nan)
    sigma = np.full(shape, np.nan)
    for point in points:
        i = int(np.argmin(np.abs(machs - point.mach)))
        j = int(np.argmin(np.abs(alphas - point.alpha)))
        axial[i, j] = point.axial
        normal[i, j] = point.normal
        moment[i, j] = point.pitching_moment
        sigma[i, j] = point.sigma

    # A symmetric body at zero incidence must produce no normal force and no
    # pitching moment. Any it does produce is spurious, and its size relative to
    # the axial force is a solution-quality number that costs nothing and needs
    # no reference value -- measured on the biconic at Mach 8, C_N is 1.1 % of
    # C_A where it must be 0.
    symmetry = {
        p.mach: (
            abs(p.normal) / max(abs(p.axial), 1e-30),
            abs(p.pitching_moment) / max(abs(p.axial), 1e-30),
        )
        for p in points
        if abs(p.alpha) < 1.0e-9
    }

    frame = reference_frame(body, **geometry)
    return AeroTable(
        name=name or f"{body}-{altitude_km:g}km",
        mach=machs,
        alpha=alphas,
        axial=axial,
        normal=normal,
        pitching_moment=moment,
        reference_area=float(frame.area),
        reference_length=float(frame.length),
        solver="SU2 8.5.0 laminar Navier-Stokes, shock-fitted",
        metadata={
            "altitude_km": float(altitude_km),
            "geometry": dict(geometry),
            # A CFD table is only meaningful at the Reynolds number it was built
            # at, so the altitude travels with it and a trajectory that reads it
            # somewhere else can be told.
            "source": "aether.cfd.tables.table_from_registry",
            "sigma": sigma,
            "points": [vars(p) for p in points],
            "unsettled": [p.run_id for p in points if not p.settled],
            # {mach: (|C_N|/|C_A|, |C_m|/|C_A|)} at zero incidence, where both
            # must be nil on a symmetric body.
            "symmetry_residual": symmetry,
        },
    )
