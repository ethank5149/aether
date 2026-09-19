"""Running SU2 as a background job, and watching it without blocking on it.

A hypersonic case is a long job: the sphere-cone smoke test in this repository
took 71 minutes over 6000 iterations on 417k elements, and a nonequilibrium
case is worse. That length is what shapes this module.

Two things follow from it, and neither is cosmetic:

**The solver runs detached.** ``subprocess.call`` in a notebook cell holds the
kernel for the whole run: no other cell executes, closing the tab kills the
job, and the only feedback is whatever the process prints when it is finished.
Here the process is started and the handle returned, so the notebook stays
live and the run outlives the cell that launched it.

**Progress is read from ``history.csv``, not from stdout.** SU2 writes a row
per iteration to that file regardless of who is watching, so progress is
recoverable after a disconnect, from another process, or for a run started
days ago -- none of which is true of a pipe. It also means the same code
reports on a finished run and a live one.

The estimate
------------

Iteration rate is measured over a trailing window rather than from the start.
A hypersonic case does not run at a constant rate: it begins on a low CFL
number with the adaptive controller ramping up, so an average taken from
iteration zero is biased slow for the whole run and the remaining time never
stops being wrong in the same direction. The window forgets the ramp.
"""

from __future__ import annotations

import contextlib
import csv
import os
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

__all__ = [
    "SolveJob",
    "SolveProgress",
    "default_ranks",
    "find_mpirun",
    "read_history",
    "read_progress",
]


#: Cells per rank below which partitioning costs more than it buys. SU2
#: exchanges a halo at every partition boundary each iteration, so a rank with
#: too little interior spends its time communicating.
#:
#: Measured rather than guessed, on a 417k-cell Mach 8 case over 60 iterations:
#:
#: =====  =======  =========  ==========
#: ranks  seconds    speedup  cells/rank
#: =====  =======  =========  ==========
#:     1     58.5       1.0x     416,610
#:     4     16.1       3.6x     104,152
#:     8     11.8       5.0x      52,076
#:    12     11.8       4.9x      34,717
#:    16     15.3       3.8x      26,038
#:    19     13.9       4.2x      21,926
#: =====  =======  =========  ==========
#:
#: Speedup peaks around eight to twelve and falls away below roughly 35,000
#: cells a rank. The drag coefficient agreed to four decimals at every count,
#: which is the part that had to be checked before any of the rest mattered.
CELLS_PER_RANK = 40_000


def default_ranks(elements: int | None = None, reserve: int = 1) -> int:
    """How many ranks to run on: enough to be quick, not so many as to thrash.

    Measured on a 417k-cell Mach 8 case, 60 iterations: 44 s on one rank
    against 12 s on eight, for the same answer to four figures (CD -0.24680
    against -0.24676). Running serial on a twenty-core machine, which is what
    this package did until now, leaves most of that on the floor.

    ``reserve`` keeps a core back so the machine stays usable while a long
    solve runs -- a data-generation box is usually also the box someone is
    working on.
    """
    import os

    cores = max(1, (os.cpu_count() or 1) - max(0, reserve))
    if elements is None:
        return cores
    return max(1, min(cores, int(elements) // CELLS_PER_RANK))


def find_mpirun() -> str | None:
    """Locate ``mpirun``, preferring the one beside the solver.

    SU2's own environment ships a matched pair; a system ``mpirun`` from a
    different MPI build will launch the ranks and then fail to let them talk,
    which surfaces as a hang rather than an error.
    """
    import shutil

    beside = Path(_find_su2()).parent / "mpirun"
    if beside.exists():
        return str(beside)
    return shutil.which("mpirun") or shutil.which("mpiexec")


#: Columns SU2 may use for the iteration counter, most specific first.
_ITERATION_COLUMNS = ("inner_iter", "outer_iter", "time_iter")


@dataclass(frozen=True)
class SolveProgress:
    """Where a run has got to, as of one reading of its history."""

    iteration: int
    total: int | None
    residual: float | None
    drag: float | None
    elapsed: float
    """Seconds since the job was started, or since the file was first seen."""

    rate: float | None
    """Iterations per second. Averaged since the start unless a
    :class:`SolveJob` supplied a windowed estimate from successive samples."""

    residual_of: str = ""
    """Which equation :attr:`residual` came from -- the least converged of them."""

    residuals: dict[str, float] = field(default_factory=dict)
    """Every residual the run writes, by name, at the latest iteration."""

    drops: dict[str, float] = field(default_factory=dict)
    """Orders of magnitude each residual has fallen from its peak."""

    @property
    def fraction(self) -> float | None:
        """Completed fraction, where the total is known."""
        if not self.total:
            return None
        return min(1.0, self.iteration / self.total)

    @property
    def remaining(self) -> float | None:
        """Estimated seconds to go, or ``None`` when it cannot be estimated."""
        if not self.total or not self.rate or self.rate <= 0.0:
            return None
        return max(0.0, (self.total - self.iteration) / self.rate)

    def describe(self, eta: bool = True) -> str:
        """One line, for a progress display or a log.

        ``eta`` is turned off for a run that has stopped. A remaining time is a
        forecast, and a forecast about a run that already ended is not a stale
        number so much as a meaningless one -- the rate beside it stays, because
        "this run averaged 51 it/min before it died" is worth knowing.
        """
        parts = [f"iter {self.iteration}" + (f"/{self.total}" if self.total else "")]
        if self.residual is not None:
            fallen = self.drops.get(self.residual_of, 0.0)
            parts.append(f"{self.residual_of} {self.residual:+.3f} ({fallen:+.2f} dex)")
        if self.drag is not None:
            parts.append(f"CD {self.drag:+.5f}")
        if self.rate:
            parts.append(f"{self.rate * 60.0:.0f} it/min")
        if eta and self.remaining is not None:
            parts.append(f"ETA {self.remaining / 60.0:.0f} min")
        return "  ".join(parts)


def _column(header: list[str], *names: str) -> int | None:
    lowered = [cell.strip().strip('"').lower() for cell in header]
    for name in names:
        if name in lowered:
            return lowered.index(name)
    return None


def _residual_columns(header: list[str]) -> list[int]:
    """Every residual column a run writes, whatever the solver.

    A perfect-gas run writes ``rms[Rho]``, ``rms[RhoU..W]`` and ``rms[RhoE]``;
    a five-species nonequilibrium run writes one density residual per species
    and a second energy for the vibrational mode. All of them are reported on,
    and the *worst* is the number, because a system is converged only when
    every equation in it is.

    Reading the density alone -- which this did at first, and which is the
    natural thing to reach for -- is actively misleading. On the Mach 20
    biconic the five species residuals sat between -2.9 and -3.5 and looked
    like a healthy run, while ``rms[RhoE]`` stood at +4.5: the energy equation
    had grown four orders of magnitude and the solver was clamping a thousand
    points an iteration. The summary said "converging" throughout.
    """
    lowered = [cell.strip().strip('"').lower() for cell in header]
    return [
        index
        for index, name in enumerate(lowered)
        if name.startswith("rms[") and name.endswith("]")
    ]


def read_progress(
    history: Path | str,
    total: int | None = None,
    started: float | None = None,
    rate: float | None = None,
) -> SolveProgress | None:
    """Read a run's progress from its ``history.csv``.

    Returns ``None`` while the file does not exist or holds no rows yet --
    a normal state for the first seconds of a run, not an error.

    ``rate`` is taken from the caller when given, because a rate cannot be
    measured from a single reading of this file: SU2 writes no timestamps, so
    all one row can support is an average over the whole elapsed time.
    :class:`SolveJob` measures a windowed rate from successive readings and
    passes it in; without one, the average is used and labelled as such.

    The whole file is read. These reach tens of megabytes on a long run and a
    tail-seek would be quicker, but the row count is what makes the iteration
    recoverable when SU2 restarts a case and resets its own counter.
    """
    path = Path(history)
    if not path.exists():
        return None
    with path.open() as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        return None

    header, last = rows[0], rows[-1]
    residual_columns = _residual_columns(header)
    drag_at = _column(header, "cd")

    def value(index: int | None) -> float | None:
        if index is None or index >= len(last):
            return None
        try:
            return float(last[index].strip())
        except ValueError:
            return None

    names = [header[index].strip().strip('"') for index in residual_columns]

    def column(index: int) -> list[float]:
        values: list[float] = []
        for row in rows[1:]:
            if index >= len(row):
                break
            try:
                values.append(float(row[index].strip()))
            except ValueError:
                break
        return values

    traces = {
        name: values
        for name, index in zip(names, residual_columns, strict=True)
        if (values := column(index))
    }
    latest = {name: values[-1] for name, values in traces.items()}
    # Convergence is a drop, and it is measured from the *peak* rather than
    # from the first row. Two things make the first row useless: SU2's
    # residuals are dimensional, so rms[RhoE] sits near +6 on a healthy run
    # while rms[Rho] sits at -3 and the two cannot be compared; and several
    # equations begin at the solver's floor and can only rise. A five-species
    # start carries no NO, N or O, so those residuals begin near -32; a
    # uniform initial field has no cross-flow, so rms[RhoV] and rms[RhoW]
    # begin near -12. Both climb as the physics switches on, and reading that
    # as divergence is reading initialisation as failure.
    drops = {name: max(values) - values[-1] for name, values in traces.items()}

    # The continuity residual is what a converged run is conventionally judged
    # on, and with several species the least-converged of them is the honest
    # one -- a mixture is not converged until every species is.
    density = [name for name in traces if name.lower().startswith(("rms[rho]", "rms[rho_"))]
    least = min(density, key=lambda name: drops[name]) if density else ""
    if not least and drops:
        least = min(drops, key=lambda name: drops[name])

    counted = len(rows) - 1
    elapsed = time.monotonic() - started if started is not None else float("nan")
    if rate is None and started is not None and elapsed > 0.0:
        rate = counted / elapsed

    return SolveProgress(
        iteration=counted,
        total=total,
        residual=latest.get(least),
        residual_of=least,
        residuals=latest,
        drops=drops,
        drag=value(drag_at),
        elapsed=elapsed,
        rate=rate,
    )


@dataclass
class SolveJob:
    """One SU2 run, started in the background and watched through its history."""

    directory: Path
    config: str = "case.cfg"
    total: int | None = None
    executable: str | None = None
    ranks: int | None = None
    """MPI ranks. ``None`` asks :func:`default_ranks`; ``1`` forces serial.

    A parallel run partitions the mesh, so its iteration history is not
    bit-identical to a serial one -- the arithmetic happens in a different
    order. It converges to the same answer, which is the thing that matters,
    but do not expect two runs on different rank counts to agree digit for
    digit part-way through.
    """

    elements: int | None = None
    """Mesh size, used only to keep :func:`default_ranks` from over-partitioning."""
    window: float = 300.0
    """Seconds of history the rate estimate looks back over.

    A hypersonic case does not run at a constant rate -- it starts on a low
    CFL number with the adaptive controller ramping up -- so an average taken
    from iteration zero is biased slow and stays biased in the same direction
    for the whole run. Looking back over a window forgets the ramp.
    """

    _process: subprocess.Popen[bytes] | None = None
    _started: float | None = None
    _samples: list[tuple[float, int]] = field(default_factory=list)

    # Set alongside ``_process`` when this job started the solver, and *instead*
    # of it when the job was attached to one that some earlier process started.
    # A pid is the only handle that crosses a process boundary; a Popen is not.
    _pid: int | None = None
    _pid_birth: int | None = None
    _started_wall: float | None = None

    @property
    def pid(self) -> int | None:
        """The solver's process id -- the handle that outlives this process."""
        if self._process is not None:
            return self._process.pid
        return self._pid

    @property
    def pid_birth(self) -> int | None:
        """The kernel's start time for :attr:`pid`, which makes it unambiguous.

        Stored with the pid in a run record so that a pid recycled onto some
        other process days later is not mistaken for this solver. See
        :func:`~aether.cfd.registry.pid_birth`.
        """
        return self._pid_birth

    @property
    def started_at(self) -> float | None:
        """Wall-clock instant the solver started, for a record or another reader."""
        return self._started_wall

    @classmethod
    def attach(
        cls,
        directory: Path | str,
        *,
        pid: int,
        started_at: float,
        pid_birth: int | None = None,
        total: int | None = None,
        ranks: int | None = None,
    ) -> SolveJob:
        """Watch a solver that some other process started.

        This is what makes a run survive the thing that launched it. The
        history was always readable from anywhere -- the docstring above says
        so -- but the *rate* and the ETA were not, because the only start
        instant kept was a :func:`time.monotonic` reading, which means nothing
        outside the process that took it. Given a wall-clock start, the instant
        is mapped into this process's monotonic frame and everything downstream
        works exactly as it does for a job started here.

        Deliberately not a call to :meth:`start`: that would launch a second
        solver into the same directory and unlink the history it was attaching
        to.
        """
        job = cls(directory=Path(directory), total=total, ranks=ranks)
        job._pid = pid
        job._pid_birth = pid_birth
        job._started_wall = started_at
        job._started = time.monotonic() - max(0.0, time.time() - started_at)
        return job

    @property
    def history(self) -> Path:
        return self.directory / "history.csv"

    @property
    def log(self) -> Path:
        return self.directory / "su2.log"

    def start(self) -> SolveJob:
        """Launch the solver and return immediately.

        The previous run's history is removed first. Leaving it means the next
        poll reads the *old* run's final iteration and reports a job that has
        barely started as complete.
        """
        if self._process is not None or self._pid is not None:
            raise RuntimeError("this job has already been started")
        executable = self.executable or _find_su2()
        ranks = self.ranks if self.ranks is not None else default_ranks(self.elements)
        command = [executable, self.config]
        if ranks > 1:
            launcher = find_mpirun()
            if launcher is None:
                # No launcher: run serial rather than fail. A missing mpirun is
                # a slower machine, not a broken one.
                ranks = 1
            else:
                command = [launcher, "-n", str(ranks), *command]
        self.ranks = ranks

        self.history.unlink(missing_ok=True)
        self._started = time.monotonic()
        self._started_wall = time.time()
        # Closed as soon as the child has it. ``Popen`` dups the descriptor
        # into the child, so the copy kept here does nothing but hold the file
        # open -- for the whole run, and past it. A supervisor that starts a
        # hundred cases would leak a hundred descriptors and eventually fail to
        # launch the hundred-and-first for want of one.
        with self.log.open("wb") as handle:
            self._process = subprocess.Popen(
                command,
                cwd=self.directory,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        # Recorded so a later process can tell this solver from whatever
        # happens to hold the same pid days from now. See
        # :func:`~aether.cfd.registry.pid_birth`.
        from aether.cfd.registry import pid_birth

        self._pid_birth = pid_birth(self._process.pid)
        return self

    @property
    def running(self) -> bool:
        if self._process is not None:
            return self._process.poll() is None
        # An attached job has no wait status to read -- the solver is not this
        # process's child -- so liveness is asked of the operating system.
        from aether.cfd.registry import process_alive

        return process_alive(self._pid, self._pid_birth)

    @property
    def returncode(self) -> int | None:
        """Exit status, or ``None`` while the solver is still going.

        Always ``None`` for an attached job, even a finished one: only a
        parent can reap a child's status, and an attached job is by definition
        not the parent. Whether such a run succeeded is answered from what it
        wrote, by :func:`~aether.cfd.registry.reconcile`.
        """
        return None if self._process is None else self._process.poll()

    def progress(self) -> SolveProgress | None:
        """Current progress, with a rate measured over the trailing window.

        Each call records an ``(instant, iteration)`` sample, so the rate
        improves as the run goes on and reflects what the solver is doing now
        rather than what it did while the CFL number was ramping. A single
        call, or one made before the window has filled, falls back to the
        average.
        """
        reading = read_progress(self.history, total=self.total, started=self._started)
        if reading is None:
            return None

        now = time.monotonic()
        self._samples.append((now, reading.iteration))
        self._samples = [
            sample for sample in self._samples if now - sample[0] <= self.window
        ] or self._samples[-1:]

        oldest, first = self._samples[0]
        span, advanced = now - oldest, reading.iteration - first
        if span <= 0.0 or advanced <= 0:
            return reading
        return replace(reading, rate=advanced / span)

    def stop(self, timeout: float = 30.0) -> None:
        """Terminate the solver, leaving whatever it has already written.

        Safe on a job that was never started, and on one that has already
        finished. Under MPI the signal goes to ``mpirun``, which takes its
        ranks down with it -- signalling the ranks individually would leave the
        launcher waiting on processes that are gone.
        """
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self._process.kill()
            return

        from aether.cfd.registry import process_alive

        if not process_alive(self._pid, self._pid_birth):
            return
        assert self._pid is not None
        import signal

        # An attached solver cannot be waited on, so the wait is a poll. The
        # escalation to SIGKILL matters more here than for a child: SU2 writing
        # a restart file at the moment it is asked to stop will not answer a
        # SIGTERM until that write finishes.
        try:
            os.kill(self._pid, signal.SIGTERM)
        except OSError:
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not process_alive(self._pid, self._pid_birth):
                return
            time.sleep(0.25)
        with contextlib.suppress(OSError):
            os.kill(self._pid, signal.SIGKILL)


def _find_su2() -> str:
    """Locate ``SU2_CFD``.

    Delegates rather than repeating the search. Two copies of "where is the
    solver" drift, and the copy that drifts is the one that finds a different
    binary than the rest of the stack is talking to.
    """
    from aether.aerodynamics.cfd.su2 import find_su2

    return str(find_su2("SU2_CFD"))


def read_history(
    history: Path | str, columns: Sequence[str] | None = None
) -> dict[str, list[float]]:
    """Every numeric column of a run's history, for plotting rather than a number.

    :func:`read_progress` answers "how far along, and is it converging" in one
    line. This returns the whole trace, because a residual that is *falling*
    and a residual that has *stalled at a limit cycle* read identically as a
    single number and are obvious as a curve -- and at half a minute an
    iteration there is no reason not to look.

    ``columns`` selects by name, case-insensitively, matching on a prefix so
    ``rms`` picks up every residual a run happens to write: one for a
    perfect-gas case and five for five-species air.
    """
    path = Path(history)
    if not path.exists():
        return {}
    with path.open() as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        return {}

    names = [cell.strip().strip('"') for cell in rows[0]]
    wanted: list[int] = list(range(len(names)))
    if columns is not None:
        lowered = [name.lower() for name in columns]
        wanted = [
            index
            for index, name in enumerate(names)
            if any(name.lower().startswith(prefix) for prefix in lowered)
        ]

    traces: dict[str, list[float]] = {}
    for index in wanted:
        values: list[float] = []
        for row in rows[1:]:
            if index >= len(row):
                break
            try:
                values.append(float(row[index]))
            except ValueError:
                break
        if values:
            traces[names[index]] = values
    return traces
