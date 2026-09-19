"""Body-axis coefficients from a solved run's history, in one place.

The extraction lived inside ``SU2Solver._read_run``, which is the **blocking
sweep** path -- the one that launches its own solves. Every coefficient this
project has actually measured was produced by the *other* path, the queue and
its worker, and nothing could read those runs into a table because the code
that knows how to read a history was a private method on the wrong class.

That is the shape of the defect recorded in
:doc:`the-simplifications-inventory`: the two execution paths were never
unified, so the one with the data had no reader and the one with the reader had
no data. This module is the reader, and both paths use it.

Body axes, and where they come from
-----------------------------------

:math:`C_A` positive aft, :math:`C_N` positive nose-up, :math:`C_m` about the
reference point. SU2 reports these directly as ``CFx``, ``CFz`` and ``CMy`` in
the frame the mesh was written in, which is the frame the table is defined in.
Reading ``CL``/``CD`` and rotating them back through the angle of attack would
give the same answer and one more place to get the rotation backwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_FloatArray = NDArray[np.float64]

__all__ = ["SettledForces", "settled_coefficients"]


@dataclass(frozen=True)
class SettledForces:
    """What one solved run says, and how well it says it."""

    axial: float
    normal: float
    pitching_moment: float

    settled: bool
    """Did the force stop moving over the full window, by the same criterion the
    worker applies live?"""

    drift: float
    ripple: float
    sigma: float
    """Standard error of the window mean, **divided by the independent-sample
    count** rather than the iterate count -- successive iterates of a limit
    cycle are strongly correlated and ``std/sqrt(window)`` understates the error
    by more than an order of magnitude."""

    periods: float
    """Oscillation periods resolved in the window. Two is thin; the report says
    so rather than quoting a standard error that pretends otherwise."""

    residual: float
    iterations: int

    @property
    def usable(self) -> bool:
        """Finite forces that settled over a window with something in it."""
        return (
            self.settled
            and self.periods >= 2.0
            and all(
                np.isfinite(v)
                for v in (self.axial, self.normal, self.pitching_moment)
            )
        )


def settled_coefficients(
    history: dict[str, _FloatArray] | Path | str,
    *,
    wall_marker: str = "vehicle",
    window: int | None = None,
    tolerance: float | None = None,
) -> SettledForces | None:
    """Body-axis coefficients from a run's history. ``None`` if it has none.

    Reported as the **mean over the convergence window**, not the last iterate:
    the solution ripples about its answer, so a final value carries the full
    amplitude of that ripple while the window mean carries it divided by the
    square root of the independent-sample count -- and the window mean is the
    number the convergence test was applied to in the first place.

    Convergence is judged on the **forebody** where the run separates one, not
    on the total. An unresolved base recirculation has no steady state, so its
    force never settles, and judging the total reports the forebody as
    unconverged on the strength of a quantity nobody should be reading.
    """
    from aether.cfd.config import Numerics
    from aether.cfd.solver import (
        _column,
        _force_settled,
        _mean_uncertainty,
        read_history,
    )

    if isinstance(history, (str, Path)):
        path = Path(history)
        if not path.exists():
            return None
        try:
            history = read_history(path)
        except (OSError, ValueError):
            return None
    if not history:
        return None

    window = int(Numerics.force_window if window is None else window)
    tolerance = float(Numerics.force_tolerance if tolerance is None else tolerance)

    forebody = _column(history, f"cfx({wall_marker})")
    base = _column(history, f"cfx({wall_marker}_base)")
    axial = _column(history, "cfx", '"cfx"')
    if axial is None and forebody is not None and base is not None:
        axial = forebody + base
    normal = _column(history, "cfz", '"cfz"')
    moment = _column(history, "cmy", '"cmy"')
    drag = _column(history, "cd", "drag")
    residual = _column(history, "rms[rho]", "rms_density", "rms[rho_0]")

    if axial is None or normal is None or moment is None:
        return None

    judged = forebody if forebody is not None else (drag if drag is not None else axial)
    settled, drift, ripple = _force_settled(judged, window, tolerance)
    sigma, periods = _mean_uncertainty(judged, window)

    def mean(values: _FloatArray | None) -> float:
        if values is None or values.size == 0:
            return float("nan")
        return float(np.mean(values[-min(window, values.size) :]))

    return SettledForces(
        axial=mean(axial),
        normal=mean(normal),
        pitching_moment=mean(moment),
        settled=bool(settled),
        drift=float(drift),
        ripple=float(ripple),
        sigma=float(sigma),
        periods=float(periods),
        residual=float(residual[-1]) if residual is not None and residual.size else float("nan"),
        iterations=int(axial.size),
    )


@dataclass(frozen=True)
class BasePressure:
    r"""A base pressure coefficient reduced from an unsteady run, with its bar."""

    coefficient: float
    r""":math:`C_{p,b}`, the time-averaged base pressure coefficient."""

    axial: float
    """What it contributes to axial force, :math:`-C_{p,b} S_b / S_{ref}`."""

    sigma: float
    """Standard error of the **mean**, from independent samples rather than steps."""

    scatter: float
    """RMS fluctuation about the mean -- the physical unsteadiness, not an error.

    Quoted separately because the two answer different questions. The scatter
    says how hard the base is breathing; ``sigma`` says how well the *average*
    is known. Reporting the first as an uncertainty overstates it, and reporting
    the second alone hides that the flow is unsteady at all.
    """

    samples: int
    periods: float
    """Independent samples the mean rests on -- the shedding cycles resolved."""

    correlation: str = "n/a"
    """What the :math:`-1/M^2` correlation would have said, for comparison."""

    @property
    def resolved(self) -> bool:
        """Whether enough cycles were averaged for the mean to mean anything.

        Ten is the working minimum. A base pressure averaged over two shedding
        cycles is a sample of the cycle, not a mean of it.
        """
        return bool(np.isfinite(self.periods) and self.periods >= 10.0)

    def summary(self) -> str:
        bar = f"+/- {self.sigma:.5f}" if np.isfinite(self.sigma) else "+/- n/a"
        return (
            f"Cpb {self.coefficient:+.5f} {bar} over {self.periods:.1f} cycles "
            f"({self.samples} samples, scatter {self.scatter:.5f}); "
            f"axial {self.axial:+.5f}"
            + ("" if self.resolved else "  -- TOO FEW CYCLES TO AVERAGE")
        )


def base_pressure(
    history_csv: Path | str,
    base_area: float,
    reference_area: float,
    mach: float,
    start_fraction: float = 0.2,
    marker: str = "vehicle_base",
) -> BasePressure | None:
    r"""Reduce a resolved base pressure from an unsteady history. ``None`` if absent.

    Why this is not a window mean
    -----------------------------

    A separated near wake sheds. The base pressure is therefore a *signal*, and
    the number a trajectory wants is its time average over many cycles together
    with how well that average is known. A steady solve, or an unsteady one
    stopped after two cycles, gives the value of the cycle at the instant it
    stopped -- which is why this refuses to call anything ``resolved`` under ten
    cycles rather than returning a confident-looking number.

    ``start_fraction`` discards the leading part of the record as the starting
    transient, the flow field having been initialised from a steady or uniform
    state that the wake has to forget.

    The error bar divides by **independent** samples, not by steps. Successive
    time steps in a dual-time run are strongly correlated -- there are thirty
    per shedding cycle by construction -- so :math:`\sigma/\sqrt{N}` would
    understate the bar by a factor of five or more. The cycle count is the
    honest divisor, the same argument
    :func:`~aether.cfd.solver._mean_uncertainty` makes for a limit-cycling
    force.
    """
    from aether.cfd.solver import read_history

    path = Path(history_csv)
    if not path.exists():
        return None
    columns = read_history(path)
    lookup = {key.lower(): key for key in columns}
    # SU2 writes per-marker coefficients as CD[i] once AERO_COEFF_SURF is on;
    # the base is the second monitored marker on this pipeline.
    key = (
        lookup.get(f"cd({marker})")
        or lookup.get("cd[1]")
        or lookup.get("cp(vehicle_base)")
    )
    if key is None:
        return None
    series = np.asarray(columns[key], dtype=np.float64)
    if series.size < 8 or not np.all(np.isfinite(series)):
        return None

    start = int(np.floor(start_fraction * series.size))
    tail = series[start:]
    if tail.size < 8:
        return None

    axial = float(np.mean(tail))
    detrended = tail - np.mean(tail)
    periods = float(np.count_nonzero(np.diff(np.signbit(detrended)))) / 2.0
    scatter = float(np.std(tail))
    sigma = scatter / np.sqrt(periods) if periods >= 1.0 else scatter

    # C_A = -C_pb S_b / S_ref, so the coefficient is recovered by inverting it.
    ratio = float(base_area) / float(reference_area) if reference_area else float("nan")
    coefficient = -axial / ratio if ratio else float("nan")
    return BasePressure(
        coefficient=coefficient,
        axial=axial,
        sigma=float(sigma),
        scatter=scatter,
        samples=int(tail.size),
        periods=periods,
        correlation=f"{-1.0 / (mach * mach):+.5f}",
    )
