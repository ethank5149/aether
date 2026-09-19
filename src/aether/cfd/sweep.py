"""The envelope sweep: a body and a grid in, an ``AeroTable`` out.

This is the drafted ``sweep_aero_database`` rebuilt on the machinery that was
already here, and the differences are mostly about what happens when a
multi-day job goes wrong.

* **Checkpointing.** The draft accumulated results in a list and wrote the
  database at the very end, so a crash on the last point of a 200-point sweep
  lost all 200. :class:`~aether.aerodynamics.tables.SweepRun` appends one JSON
  line per finished point, and a rerun skips what is already there.
* **Failures.** The draft caught a solver failure, printed it, and
  ``continue``\\ d — so a point that diverged simply went missing from the
  table with no record of which one, and the table came back one row short and
  looking complete. Here a non-converged point raises by default
  (:attr:`~aether.cfd.solver.SU2Solver.strict`) or is recorded as
  ``nan`` and reported by :attr:`~aether.aerodynamics.tables.AeroTable.complete`.
* **Reference quantities.** The draft's template fixed ``REF_AREA= 1.0`` and
  ``REF_LENGTH= 1.0`` regardless of the vehicle, which makes every coefficient
  a force in disguise and un-splice-able with the panel tables. Here they come
  from the configuration, exactly as in :mod:`aether.aerodynamics.build`.
* **Cleanup.** The draft deleted every ``*.dat`` and ``*.vtu`` in the run
  directory. That includes the restart file, which is what a case would be
  continued or re-examined from; ``keep_cases`` governs it here.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from aether.aerodynamics.cfd.meshing import RefinementBall, ViscousSizing
from aether.aerodynamics.tables import AeroTable, SweepGrid, SweepRun
from aether.cfd.config import Numerics, ReferenceFrame, Regime
from aether.cfd.solver import SU2Solver

__all__ = [
    "ENVELOPE_ALPHA_DEG",
    "ENVELOPE_MACH",
    "cfd_grid",
    "sweep_cfd_table",
    "write_hdf5",
]


#: Mach stations for a reentry-vehicle envelope, spaced by where the physics
#: turns over rather than uniformly.
#:
#: Dense from 0.8 to 1.5 because the transonic drag rise is the steepest
#: ``dCd/dM`` anywhere in the envelope; a grid that steps 0.9 to 1.2 steps over
#: the peak. Coarse from 1.5 to 8 where the curve is smooth and monotone.
#:
#: **Not** thinned above Mach 12, which is the tempting move: Mach-number
#: independence does hold for perfect-gas ``Cd`` up there, but it is exactly
#: where it fails for pitching moment once the shock layer dissociates — the
#: STS-1 pitch-up anomaly was a real-gas failure of that assumption, at a Mach
#: number where the drag really was flat. A table meant to be trusted for
#: stability, not just drag, has to resolve that band.
#:
#: Subsonic stations are omitted deliberately. SU2's compressible Euler path is
#: badly conditioned below about Mach 0.3 without low-Mach preconditioning, and
#: a reentry vehicle's subsonic behaviour is a terminal-descent question that
#: this table is the wrong instrument for.
ENVELOPE_MACH: tuple[float, ...] = (
    0.8, 1.05, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 16.0, 20.0, 27.0,
)

#: Incidence stations (deg), clustered near zero.
#:
#: The 0/2 pair is deliberate and is the reason this is not a uniform grid:
#: it is what gives a clean finite-difference ``C_N_alpha`` and ``C_m_alpha``.
#: On a uniform 5-degree grid the first interval already spans enough angle for
#: the ``sin^2`` nonlinearity to contaminate the slope, so the static stability
#: derivative read off it is not the derivative at trim.
#:
#: Non-negative only: both reference bodies are laterally symmetric, so the
#: negative half-plane is a mirror and solving it would double the bill to
#: reproduce numbers already in hand.
ENVELOPE_ALPHA_DEG: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, 15.0, 20.0)


def cfd_grid(
    mach: tuple[float, ...] = ENVELOPE_MACH,
    alpha_deg: tuple[float, ...] = ENVELOPE_ALPHA_DEG,
) -> SweepGrid:
    """A coarse envelope grid, sized for how much a CFD point costs.

    Deliberately far coarser in both axes than
    :meth:`~aether.aerodynamics.tables.SweepGrid.default_mach` — that grid is
    built for a panel method at milliseconds per point, and the same grid here
    is a job measured in days. See :data:`ENVELOPE_MACH` and
    :data:`ENVELOPE_ALPHA_DEG` for why the stations sit where they do.
    """
    return SweepGrid(
        mach=np.asarray(mach, dtype=np.float64),
        alpha=np.deg2rad(np.asarray(alpha_deg, dtype=np.float64)),
    )


def sweep_cfd_table(
    body: Any,
    *,
    store_dir: str | Path,
    reference: ReferenceFrame,
    altitude: float,
    grid: SweepGrid | None = None,
    name: str = "cfd",
    regime: Regime | None = None,
    numerics: Numerics | None = None,
    sizing: ViscousSizing | None = None,
    wall_temperature: float = 300.0,
    processes: int = 1,
    workers: int = 1,
    strict: bool = True,
    split_base: bool = False,
    base_station: float | None = None,
    wall_refinement: float = 0.05,
    refinement: tuple[RefinementBall, ...] = (),
    shock_layer_radius: float | None = None,
    alpha_continuation: bool = True,
    remesh_per_mach: bool = True,
    keep_cases: bool = True,
    executable: Path | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    max_points: int | None = None,
    time_budget: float | None = None,
    report_every: int = 25,
) -> AeroTable:
    """Sweep one body over a Mach/incidence grid with SU2, checkpointing as it goes.

    ``max_points`` and ``time_budget`` make the sweep resumable in slices,
    which is the only way a job of this length is actually run: start it, stop
    it, look at what came out, and continue from the checkpoint.

    The returned table carries the altitude and the mesh's boundary-layer
    provenance in its metadata, because a CFD table is only meaningful at the
    Reynolds number it was built at, and a table that does not record its own
    altitude cannot be checked against the trajectory that reads it.

    Parameters
    ----------
    split_base, base_station:
        Separate the base disc from the forebody. **Set these on any body with
        a blunt base** — which both reference RV shapes have. An Euler base
        pressure is not a physical quantity, and folding it into one force
        coefficient inverted the sign of the reported drag on a Mach 8
        sphere-cone. They were previously unreachable through this function,
        so a sweep could only ever produce the folded-in number.
    processes, workers:
        ``processes`` is MPI ranks *per point*; ``workers`` is how many points
        run at once. They multiply, so ``processes * workers`` should sit at or
        under the core count. Measured on the 417k-cell biconic, a single point
        scales about 5x out to 8-12 ranks and then gives it back, so on a
        20-core box ``processes=8, workers=2`` beats ``processes=16,
        workers=1``. Both default to 1: a sweep should not quietly seize the
        machine because it was called with defaults.
    alpha_continuation:
        Start each incidence from the nearest lower one already solved, rather
        than from freestream. Left on unless you are deliberately measuring
        cold-start cost -- at hypersonic conditions the asymmetric bow-shock
        transient is what diverges, and continuation removes it rather than
        papering over it with a globally lower CFL.
    """
    out = Path(store_dir)
    out.mkdir(parents=True, exist_ok=True)

    solver = SU2Solver(
        body=body,
        workspace=out / name,
        reference=reference,
        altitude=float(altitude),
        regime=regime,
        numerics=numerics if numerics is not None else Numerics(),
        sizing=sizing if sizing is not None else ViscousSizing(),
        wall_temperature=wall_temperature,
        processes=processes,
        strict=strict,
        split_base=split_base,
        base_station=base_station,
        wall_refinement=wall_refinement,
        refinement=refinement,
        shock_layer_radius=shock_layer_radius,
        alpha_continuation=alpha_continuation,
        remesh_per_mach=remesh_per_mach,
        keep_cases=keep_cases,
        executable=executable,
        name=f"su2-3d-{name}",
    )
    if workers > 1 and not keep_cases:
        # The continuation seed *is* the previous point's restart file, so
        # deleting cases as they finish breaks the chain the concurrency is
        # there to exploit. Caught here rather than left to surface as every
        # incidence mysteriously starting from freestream.
        msg = "workers > 1 needs keep_cases=True: the continuation seed is the previous restart"
        raise ValueError(msg)
    run = SweepRun(
        name=name,
        grid=grid if grid is not None else cfd_grid(),
        solver=solver,
        store=out / f"{name}.jsonl",
        reference_area=reference.area,
        reference_length=reference.length,
        metadata={
            "altitude_m": float(altitude),
            "wall_temperature_k": float(wall_temperature),
            "moment_reference_m": list(reference.moment_reference),
            "y_plus_target": float(solver.sizing.y_plus),
            "boundary_layer_growth": float(solver.sizing.growth_ratio),
            "split_base": bool(split_base or base_station is not None),
            "mpi_ranks_per_point": int(processes),
            "concurrent_points": int(workers),
        },
    )
    # Grouped by mesh key, which is the contract SweepRun.run asks for: points
    # sharing a mesh share the solver's cached entry for it, so they must not
    # run concurrently. It also happens to be what makes continuation work,
    # since a group holds every incidence at a Mach number, in order.
    return run.run(
        progress=progress,
        max_points=max_points,
        time_budget=time_budget,
        report_every=report_every,
        workers=workers,
        group_key=(lambda mach, _alpha: solver.mesh_key(mach)) if workers > 1 else None,
    )


def write_hdf5(table: AeroTable, path: str | Path) -> Path:
    """Write a table to HDF5, for tools that want the drafts' format.

    Secondary to the JSONL checkpoint and the table itself, not a replacement:
    the checkpoint is what survives a crash, and HDF5 is what a post-processor
    or a plotting notebook would rather read. The reference quantities and the
    metadata travel as attributes, so a file that has been moved away from its
    checkpoint still knows what it is non-dimensionalised on — which the
    drafted ``to_hdf`` call, writing a bare dataframe of Mach/AoA/CL/CD, did not.

    Requires ``h5py`` (``pip install aether-gambit[cfd]``).
    """
    import h5py

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(target, "w") as handle:
        group = handle.create_group("aero_table")
        group.create_dataset("mach", data=np.asarray(table.mach))
        group.create_dataset("alpha_rad", data=np.asarray(table.alpha))
        group.create_dataset("axial", data=np.asarray(table.axial))
        group.create_dataset("normal", data=np.asarray(table.normal))
        group.create_dataset("pitching_moment", data=np.asarray(table.pitching_moment))
        group.attrs["name"] = table.name
        group.attrs["solver"] = table.solver
        group.attrs["reference_area_m2"] = float(table.reference_area)
        group.attrs["reference_length_m"] = float(table.reference_length)
        group.attrs["complete"] = bool(table.complete)
        for key, value in (table.metadata or {}).items():
            group.attrs[key] = value
    return target
