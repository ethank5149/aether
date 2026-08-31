"""Resumable aerodynamic coefficient sweeps.

A coefficient table is a long, embarrassingly parallel sweep over Mach and
incidence, and the two facts that shape this module are that the cheap
solver takes milliseconds per point while the expensive one takes minutes,
and that nobody wants to lose a week of the expensive one to a crash.

So results are **appended to a checkpoint as they complete**, one JSON
object per line, and a run that starts against an existing checkpoint skips
what is already there. That is not a nicety — a table over a fine grid with
a CFD solver is a multi-day job, and a design that can only be run to
completion in one sitting cannot be run at all.

Newline-delimited JSON rather than a binary format or a dataframe: it is
append-only so a crash mid-write loses one line rather than the file, it is
readable while the job is running, and a partial file is still a valid
partial table.

Reference quantities
--------------------

Coefficients are non-dimensionalised on a **fixed** reference area and
length — the full vehicle's maximum cross-section and its diameter — and not
on each configuration's own. Otherwise the axial coefficient of the stack
and of the payload alone are divided by different numbers and cannot be
compared or blended across staging, which is the entire purpose of building
them together.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "AeroTable",
    "Coefficients",
    "PanelSolver",
    "Solver",
    "SweepGrid",
    "SweepRun",
    "console_progress",
    "sequence_of",
]

_FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Coefficients:
    """Force and moment coefficients in body axes.

    Attributes
    ----------
    axial:
        :math:`C_A`, positive aft — drag at zero incidence.
    normal:
        :math:`C_N`, positive with the nose up.
    pitching_moment:
        :math:`C_m` about the reference point, per reference length.
    """

    axial: float
    normal: float
    pitching_moment: float

    def as_dict(self) -> dict[str, float]:
        return {
            "axial": float(self.axial),
            "normal": float(self.normal),
            "pitching_moment": float(self.pitching_moment),
        }


class Solver(Protocol):
    """Anything that can return coefficients at one flight condition."""

    name: str

    def solve(self, mach: float, alpha: float) -> Coefficients:  # pragma: no cover
        ...


@dataclass(frozen=True)
class SweepGrid:
    """The Mach and incidence points a table is built on.

    Both are stated explicitly rather than as a range and a count, because a
    useful Mach grid is not uniform: it needs to be dense through the
    transonic bucket, where the coefficients change fastest, and can be
    coarse above Mach 10, where they barely change at all. Handing this a
    ``linspace`` wastes most of the run.
    """

    mach: _FloatArray
    alpha: _FloatArray
    """Incidence (rad)."""

    def __post_init__(self) -> None:
        for label, values in (("mach", self.mach), ("alpha", self.alpha)):
            array = np.asarray(values, dtype=np.float64)
            if array.ndim != 1 or array.size == 0:
                msg = f"{label} must be a non-empty 1-D array, got shape {array.shape}"
                raise ValueError(msg)
            if np.any(~np.isfinite(array)):
                msg = f"{label} contains non-finite values"
                raise ValueError(msg)
        if np.any(np.asarray(self.mach) <= 0.0):
            msg = "Mach numbers must be positive"
            raise ValueError(msg)

    @property
    def size(self) -> int:
        return int(np.asarray(self.mach).size * np.asarray(self.alpha).size)

    def points(self) -> Iterator[tuple[float, float]]:
        """Every (Mach, incidence) pair, Mach-major."""
        for mach in np.asarray(self.mach, dtype=np.float64):
            for alpha in np.asarray(self.alpha, dtype=np.float64):
                yield float(mach), float(alpha)

    @staticmethod
    def default_mach(minimum: float | None = None) -> _FloatArray:
        """A Mach grid weighted where the physics is, not uniformly.

        Dense from 0.6 to 1.6 because that is the transonic bucket, where
        the axial coefficient roughly triples and then falls; progressively
        coarser above, because beyond about Mach 8 Newtonian theory has
        essentially converged and the curve is flat.
        """
        grid = np.concatenate(
            [
                np.arange(0.3, 0.6, 0.1),
                np.arange(0.6, 1.65, 0.05),
                np.arange(1.8, 3.1, 0.2),
                np.arange(3.5, 6.1, 0.5),
                np.arange(7.0, 12.1, 1.0),
                np.array([14.0, 16.0, 20.0, 25.0]),
            ]
        )
        if minimum is not None:
            grid = grid[grid >= float(minimum)]
        return np.asarray(np.round(grid, 6))

    @staticmethod
    def default_alpha(limit_deg: float = 12.0, step_deg: float = 1.0) -> _FloatArray:
        return np.deg2rad(np.arange(0.0, limit_deg + 0.5 * step_deg, step_deg))


@dataclass(frozen=True)
class AeroTable:
    """A completed or partial sweep, indexed by Mach and incidence."""

    name: str
    mach: _FloatArray
    alpha: _FloatArray
    axial: _FloatArray
    normal: _FloatArray
    pitching_moment: _FloatArray
    reference_area: float
    reference_length: float
    solver: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return not bool(np.any(np.isnan(self.axial)))

    @property
    def filled(self) -> int:
        return int(np.sum(~np.isnan(self.axial)))

    def at(self, mach: float, alpha: float) -> Coefficients:
        """Bilinear interpolation, clamped at the grid edges.

        Clamped rather than extrapolated: a coefficient table asked for
        Mach 30 when it was built to 25 should return the Mach 25 value, not
        a linear guess off the end of a curve that is asymptotic anyway.
        """

        def interpolate(surface: _FloatArray) -> float:
            row = np.interp(
                float(np.clip(alpha, self.alpha[0], self.alpha[-1])),
                self.alpha,
                np.arange(self.alpha.size, dtype=np.float64),
            )
            column = np.interp(
                float(np.clip(mach, self.mach[0], self.mach[-1])),
                self.mach,
                np.arange(self.mach.size, dtype=np.float64),
            )
            i0, j0 = int(np.floor(column)), int(np.floor(row))
            i1 = min(i0 + 1, self.mach.size - 1)
            j1 = min(j0 + 1, self.alpha.size - 1)
            fi, fj = column - i0, row - j0
            return float(
                surface[i0, j0] * (1 - fi) * (1 - fj)
                + surface[i1, j0] * fi * (1 - fj)
                + surface[i0, j1] * (1 - fi) * fj
                + surface[i1, j1] * fi * fj
            )

        return Coefficients(
            interpolate(self.axial),
            interpolate(self.normal),
            interpolate(self.pitching_moment),
        )

    def drag_area(self, mach: float) -> float:
        """:math:`C_D A` at zero incidence (m²) — what a trajectory needs.

        The flight simulator takes a single ``drag_area``. This is where
        that number should come from instead of being typed in.
        """
        return float(self.at(mach, 0.0).axial * self.reference_area)


@dataclass
class PanelSolver:
    """Modified-Newtonian / Prandtl-Meyer panel integration over a mesh.

    Milliseconds per point, so a full table is seconds. Its validity is the
    honest limit: impact theory is the standard preliminary-design method
    above roughly Mach 5 and is good to about 10 % on axial force for a
    slender body there. Below Mach 2 it is not a model of anything, and the
    ``mach_floor`` records where the caller has been told to stop believing
    it rather than leaving that to be remembered.
    """

    mesh: Any
    reference_area: float
    reference_length: float
    reference_point: _FloatArray | None = None
    mach_floor: float = 4.0
    name: str = "panel"

    def __post_init__(self) -> None:
        self._model = self.mesh.panel_model(self.reference_point)

    #: Below this the pressure closure has no supersonic solution at all.
    absolute_floor: float = 1.05

    def solve(self, mach: float, alpha: float, cp_max: float | None = None) -> Coefficients:
        """Coefficients at one condition.

        ``cp_max`` overrides the perfect-gas stagnation pressure coefficient
        with an equilibrium-air value; see
        :class:`aether.aerodynamics.realgas.EquilibriumAir`. It is a keyword
        with a default rather than a required argument so that this still
        satisfies the two-argument :class:`Solver` protocol.
        """
        if float(mach) <= self.absolute_floor:
            msg = (
                f"the panel method is a supersonic theory and cannot evaluate "
                f"Mach {mach:g}. Its pressure closure blends a Newtonian "
                f"windward branch with Prandtl-Meyer expansion, and neither "
                f"exists subsonically. Build the sub- and transonic part of a "
                f"table with a compressible flow solver and splice; "
                f"SweepGrid.default_mach(minimum=...) trims a grid to a "
                f"solver's range."
            )
            raise ValueError(msg)
        # Dynamic pressure cancels out of a coefficient; unity keeps the
        # arithmetic in a sane range.
        force, moment = self._model.loads(float(alpha), float(mach), 1.0, cp_max=cp_max)
        scale = self.reference_area
        return Coefficients(
            axial=float(force[0] / scale),
            normal=float(force[2] / scale),
            pitching_moment=float(moment[1] / (scale * self.reference_length)),
        )


class SweepRun:
    """A checkpointed sweep that can be stopped and resumed.

    Parameters
    ----------
    name:
        Identifies the configuration; written into every checkpoint line so
        one file cannot silently mix two vehicles.
    grid:
        Points to evaluate.
    solver:
        Anything with ``solve(mach, alpha) -> Coefficients``.
    store:
        Checkpoint path. Created if absent, appended to if present.
    reference_area, reference_length:
        Non-dimensionalisation, recorded so a table read back cannot be
        combined with one built on different references.
    """

    def __init__(
        self,
        name: str,
        grid: SweepGrid,
        solver: Solver,
        store: str | Path,
        reference_area: float,
        reference_length: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.grid = grid
        self.solver = solver
        self.store = Path(store)
        self.reference_area = float(reference_area)
        self.reference_length = float(reference_length)
        self.metadata = dict(metadata or {})
        self.store.parent.mkdir(parents=True, exist_ok=True)

    # -- checkpoint --------------------------------------------------------

    @staticmethod
    def _key(mach: float, alpha: float) -> str:
        """A grid point's identity, rounded so float noise cannot duplicate it."""
        return f"{mach:.6f}|{alpha:.9f}"

    def completed(self) -> dict[str, dict[str, Any]]:
        """Everything already in the checkpoint, keyed by grid point.

        Malformed trailing lines are skipped rather than fatal: a run killed
        mid-write leaves a partial last line, and losing the whole table to
        that would defeat the point of checkpointing.
        """
        if not self.store.exists():
            return {}
        done: dict[str, dict[str, Any]] = {}
        with self.store.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("name") != self.name:
                    continue
                done[self._key(record["mach"], record["alpha"])] = record
        return done

    def status(self) -> dict[str, Any]:
        """How much is done, without evaluating anything."""
        done = len(self.completed())
        total = self.grid.size
        return {
            "name": self.name,
            "solver": self.solver.name,
            "completed": done,
            "total": total,
            "remaining": total - done,
            "fraction": done / total if total else 1.0,
            "store": str(self.store),
        }

    # -- the run -----------------------------------------------------------

    def run(
        self,
        progress: Callable[[dict[str, Any]], None] | None = None,
        max_points: int | None = None,
        time_budget: float | None = None,
        report_every: int = 25,
    ) -> AeroTable:
        """Evaluate outstanding points, then return the table so far.

        Parameters
        ----------
        max_points, time_budget:
            Stop after this many new points, or this many seconds. Either
            makes a long sweep something you can run in slices — which is
            the difference between a week-long job that is possible and one
            that is not.
        report_every:
            Points between progress callbacks.

        Notes
        -----
        A ``KeyboardInterrupt`` is caught, the checkpoint flushed, and the
        partial table returned. Stopping a sweep is a normal thing to do,
        not an error, and the work already done must survive it.
        """
        done = self.completed()
        outstanding = [
            (mach, alpha)
            for mach, alpha in self.grid.points()
            if self._key(mach, alpha) not in done
        ]
        started = time.perf_counter()
        evaluated = 0
        # Captured before the loop: `done` is updated in place as points
        # complete, so reading its length inside the loop counts every new
        # point twice and the progress bar sails past 100 %.
        already = len(done)

        def emit(final: bool = False) -> None:
            if progress is None:
                return
            elapsed = time.perf_counter() - started
            rate = evaluated / elapsed if elapsed > 0 else 0.0
            left = len(outstanding) - evaluated
            progress(
                {
                    "name": self.name,
                    "evaluated": evaluated,
                    "outstanding": len(outstanding),
                    "completed": already + evaluated,
                    "total": self.grid.size,
                    "elapsed": elapsed,
                    "rate": rate,
                    "eta": left / rate if rate > 0 else float("nan"),
                    "final": final,
                }
            )

        try:
            with self.store.open("a") as handle:
                for mach, alpha in outstanding:
                    if max_points is not None and evaluated >= max_points:
                        break
                    if time_budget is not None and time.perf_counter() - started >= time_budget:
                        break
                    coefficients = self.solver.solve(mach, alpha)
                    record = {
                        "name": self.name,
                        "solver": self.solver.name,
                        "mach": mach,
                        "alpha": alpha,
                        "reference_area": self.reference_area,
                        "reference_length": self.reference_length,
                        **coefficients.as_dict(),
                    }
                    handle.write(json.dumps(record) + "\n")
                    # Flushed per point on purpose: a sweep whose results
                    # sit in a buffer is not checkpointed, it only looks it.
                    handle.flush()
                    done[self._key(mach, alpha)] = record
                    evaluated += 1
                    if evaluated % max(report_every, 1) == 0:
                        emit()
        except KeyboardInterrupt:
            emit(final=True)
            return self.table(done)
        emit(final=True)
        return self.table(done)

    def table(self, done: dict[str, dict[str, Any]] | None = None) -> AeroTable:
        """Assemble what the checkpoint holds into a gridded table.

        Missing points are ``nan`` rather than interpolated, so a partial
        table cannot be mistaken for a finished one — :attr:`AeroTable.complete`
        is the check.
        """
        records = self.completed() if done is None else done
        mach = np.asarray(self.grid.mach, dtype=np.float64)
        alpha = np.asarray(self.grid.alpha, dtype=np.float64)
        shape = (mach.size, alpha.size)
        surfaces = {key: np.full(shape, np.nan) for key in ("axial", "normal", "pitching_moment")}
        for i, m in enumerate(mach):
            for j, a in enumerate(alpha):
                record = records.get(self._key(float(m), float(a)))
                if record is None:
                    continue
                for key, surface in surfaces.items():
                    surface[i, j] = float(record[key])
        return AeroTable(
            name=self.name,
            mach=mach,
            alpha=alpha,
            axial=surfaces["axial"],
            normal=surfaces["normal"],
            pitching_moment=surfaces["pitching_moment"],
            reference_area=self.reference_area,
            reference_length=self.reference_length,
            solver=self.solver.name,
            metadata=self.metadata,
        )


def console_progress(width: int = 34) -> Callable[[dict[str, Any]], None]:
    """A progress callback that overwrites one line and stays readable.

    Deliberately not a notebook widget: this has to be as useful in a
    detached terminal running a multi-day sweep as it is in a cell.
    """

    def report(state: dict[str, Any]) -> None:
        done = state["completed"]
        total = state["total"]
        fraction = done / total if total else 1.0
        filled = round(fraction * width)
        bar = "#" * filled + "." * (width - filled)
        eta = state["eta"]
        eta_text = "--:--"
        if np.isfinite(eta):
            seconds = int(eta)
            eta_text = f"{seconds // 60:3d}m{seconds % 60:02d}s"
        end = "\n" if state["final"] else "\r"
        line = (
            f"  {state['name']:22s} [{bar}] {done:5d}/{total:<5d} "
            f"{fraction * 100:5.1f}%  {state['rate']:7.2f} pt/s  ETA {eta_text}"
        )
        # Padded so a shorter line cannot leave characters from a longer one
        # behind it when the carriage returns.
        print(line.ljust(96), end=end, flush=True)

    return report


def sequence_of(values: Sequence[float]) -> _FloatArray:
    """Small helper so a notebook can write a Mach list inline."""
    return np.asarray(values, dtype=np.float64)
