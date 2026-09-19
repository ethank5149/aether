"""``python -m aether.cfd.worker`` -- the process that owns the queue.

What this exists to fix is not the solver. :mod:`~aether.cfd.run`
already detaches SU2, and a solve survives whatever launched it. The queue did
not. In the notebook the driver was a daemon thread holding a Python list, so a
kernel restart part-way through a six-point sweep lost the five points that had
not run yet, with no record they had ever been asked for. Meshing was worse: it
ran *inside* the kernel, so a 400k-element domain -- minutes of work, and the
step least distinguishable from a hang -- died with the browser tab.

So the queue lives on disk (:mod:`~aether.cfd.registry`) and this owns
it. Everything that must outlive a disconnect happens here; a UI does nothing
but append records and read history files, which is what lets a UI be closed,
reopened on a different device, or killed by Android reclaiming a background
tab without any of that touching the work.

Meshing serialises, solving does not
------------------------------------

``--slots`` bounds how many solvers run at once, and meshing is outside it: one
domain is built at a time no matter how many slots there are. That is not a
simplification, it is the correct shape. gmsh is not safe to drive from two
places at once, and a mesh build is a saturating single-node job anyway --
overlapping two of them makes both slower and neither finishes sooner. Solves
are separate processes on their own ranks and genuinely do overlap, so those
are what the slot count governs.

``slots * ranks`` should sit at or under the core count, exactly as for
:mod:`~aether.cfd.database`.

Shutting down
-------------

A ``SIGTERM`` stops the loop and **leaves the solvers running**. This is
deliberate and is the whole design working as intended: a solve is recoverable
from its record, so restarting the worker reattaches to what is still going
rather than starting it over. Killing hours of converged iterations because the
supervisor was restarted would be the notebook's failure mode reintroduced at a
different layer. Use ``--drain`` to wait for them instead, or cancel the runs.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import socket
import sys
import time
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from aether.cfd.frames import FrameWriter
from aether.cfd.registry import (
    RunRecord,
    RunSpec,
    cancel_requested,
    clear_cancel,
    history_rows,
    load,
    load_all,
    pid_birth,
    pid_namespace,
    process_alive,
    queue_dir,
    reconcile,
    save,
)
from aether.cfd.run import SolveJob
from aether.paths import cfd_results

__all__ = ["Worker", "case_directory", "case_markers", "main", "worker_pid"]


def _lock_path() -> Path:
    """The file naming the worker that owns this queue."""
    return queue_dir() / "worker.lock"


#: How long a worker's heartbeat stays believable, as a multiple of its poll
#: interval, and a floor in seconds. Only consulted by a reader in a different
#: PID namespace, which has nothing better to go on.
_STALE_MULTIPLE = 5.0
_STALE_FLOOR = 30.0

#: Rise in the log RMS residual, decades per thousand iterations, above which a
#: run is called diverged outright.
#:
#: There is deliberately **no** tighter threshold than this. One was tried, at
#: 0.15, on the reasoning that a climbing residual means a deteriorating field.
#: The campaign disproved it. Measured over the same 2000-iteration force
#: window across six completed sphere-cone runs:
#:
#: ==============================  =======  =======  =========  ==========
#: case                              slope    pitot       Tmax       field
#: ==============================  =======  =======  =========  ==========
#: Mach 4, alpha 0                   +0.12     1.5x     1 238 K       clean
#: Mach 6, alpha 0                   +0.42     1.3x     2 417 K       clean
#: Mach 8, alpha 0                   +0.45    37.9x   590 411 K  CARBUNCLE
#: Mach 8, alpha 2                   -0.07     1.4x     4 100 K       clean
#: Mach 8, alpha 4                   +0.44     1.4x     3 979 K       clean
#: Mach 8, alpha 6                   +0.20     1.5x     4 758 K       clean
#: ==============================  =======  =======  =========  ==========
#:
#: +0.44 with a clean field against +0.45 with a carbuncle: the slope does not
#: separate them, and a 0.15 limit would have held back three of the five good
#: runs. The earlier calibration looked convincing because it was dominated by
#: the NEMO cases, which sit at +3 to +13 and are a different failure entirely.
#:
#: The **pitot ratio** separates cleanly -- 1.3-1.5x on every good run against
#: 37.9x -- so that is the gate, in ``_unphysical`` and ``_diverged``. What
#: survives here is a backstop for gross divergence, which every NEMO blow-up
#: clears by a factor of four and no clean run comes close to.
RESIDUAL_DIVERGENCE_LIMIT = 0.8

#: Multiple of the normal-shock stagnation pressure above which a solution is
#: not a solution. Stagnation pressure behind a normal shock is a hard physical
#: bound, so anything above 1.0 is a captured shock spread over too few cells;
#: 2.0 is the slack that allows for that and nothing more. Converged runs on
#: this queue read 1.1-1.6x, the carbuncled ones 7x, 17x and 45893x.
PITOT_CEILING_RATIO = 2.0


def _base_diameter(spec: Any) -> float:
    r"""Equivalent diameter of the body's aft face, measured from the surface.

    The shedding scale that sizes an unsteady base-flow run is the **base**
    size, not the body length or the reference area, so it is measured rather
    than inferred: the faces lying in the aft plane, their summed area, and
    :math:`D = 2\sqrt{A/\pi}`.

    The equivalent diameter rather than a radius, because three of the nine
    reference bodies have a base that is not a disc -- the caret waverider and
    the osculating-cone waverider have spanwise trailing edges, and the lifting
    body's base is a blended cross-section. An equivalent diameter is the
    standard way to carry a bluff-body scale across those shapes.

    Refuses a body that closes to a point. ``sears-haack`` is closed at both
    ends and has no base at all -- its aft plane holds a single node and zero
    area -- so there is no base flow to integrate and asking for one is a
    mistake worth stopping rather than silently running.
    """
    import numpy as np

    from aether.cfd.geometries import build_surface

    surface = build_surface(spec.body, **spec.geometry)
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    faces = np.asarray(surface.faces, dtype=np.int64)
    centroids = vertices[faces].mean(axis=1)
    aft = float(centroids[:, 0].max())
    base = faces[np.isclose(centroids[:, 0], aft, atol=1e-6)]
    area = 0.0
    if len(base):
        corners = vertices[base]
        area = float(
            0.5
            * np.linalg.norm(
                np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
                axis=1,
            ).sum()
        )
    diameter = float(2.0 * np.sqrt(max(area, 0.0) / np.pi))

    # Judged against the body's own cross-section, not against an area epsilon.
    # A closed body still triangulates to a sliver at its tip -- sears-haack
    # returns 1 mm of equivalent diameter that way, which sizes a 517 kHz
    # shedding frequency and a 0.1 us time step, and every one of those numbers
    # is arithmetic on a rounding error. A base worth integrating is a
    # meaningful fraction of the body it sits on.
    widest = float(2.0 * np.hypot(vertices[:, 1], vertices[:, 2]).max())
    if widest <= 0.0 or diameter < 0.02 * widest:
        msg = (
            f"{spec.body} closes at its aft end: base equivalent diameter "
            f"{1000 * diameter:.2f} mm against a body width of "
            f"{1000 * widest:.1f} mm. It has no base flow to integrate, and an "
            "unsteady base-flow run on it would be sized by a rounding error"
        )
        raise ValueError(msg)
    return diameter


def read_lock() -> dict[str, Any] | None:
    """The current worker's claim on this queue, if there is one."""
    try:
        payload = json.loads(_lock_path().read_text())
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def worker_alive(claim: dict[str, Any] | None) -> bool:
    """Whether the process behind a lock is still running.

    Two ways to answer, because the panel and the worker are usually separate
    containers and therefore separate PID namespaces:

    **Same namespace** -- the pid is ours to check, so it is checked exactly,
    start-time and all.

    **Different namespace** -- the pid is a number about somebody else's
    process table and means nothing here. Checking it would report a healthy
    worker as absent, which is precisely the failure this replaced: a panel in
    one container saying "no worker" while the worker in the next container was
    busy meshing. So the heartbeat is used instead: the worker rewrites its
    claim every pass, and a claim that has stopped being rewritten is stale.
    """
    if not claim:
        return False
    pid = claim.get("pid")
    if not isinstance(pid, int):
        return False
    if claim.get("ns") is not None and claim.get("ns") == pid_namespace():
        birth = claim.get("birth")
        return process_alive(pid, birth if isinstance(birth, int) and birth >= 0 else None)

    beat = claim.get("at")
    if not isinstance(beat, (int, float)):
        return False
    poll = claim.get("poll")
    window = max(_STALE_FLOOR, _STALE_MULTIPLE * (poll if isinstance(poll, (int, float)) else 3.0))
    return (time.time() - float(beat)) <= window


def worker_pid() -> int | None:
    """The pid of the worker owning this queue, or ``None`` if there isn't one.

    Worth showing in any UI that can queue work. A queue with nothing draining
    it accepts runs and then does nothing with them, and "I pressed launch an
    hour ago and it is still queued" is a confusing way to discover that the
    worker was never started.

    The pid is only directly meaningful in the worker's own namespace; from
    elsewhere treat it as an identifier, not something to signal.
    """
    claim = read_lock()
    if claim is None or not worker_alive(claim):
        return None
    pid = claim.get("pid")
    return pid if isinstance(pid, int) else None


def case_directory(record: RunRecord, root: Path | None = None) -> Path:
    """Where this run's mesh, config, history and log go.

    The same layout the notebook wrote, so an existing results tree is still
    addressed by the same names and a re-queued point reuses its own directory
    instead of accumulating a second one beside it.
    """
    base = root if root is not None else cfd_results()
    return base / f"gen-{record.spec.body}" / "cases" / record.spec.case_name()


def _worst_pitot(case: Path) -> float:
    """Largest pitot exceedance the frame detector recorded for this case.

    The detector states its finding as prose in the frame's ``note``; this
    reads the ratio back out rather than re-deriving it. Zero when no frame
    reported one, which is also what a case with no frames returns -- absence
    of evidence, and the other checks still apply.
    """
    import re

    frames = case / "frames"
    if not frames.is_dir():
        return 0.0
    worst = 0.0
    for js in frames.glob("*.json"):
        try:
            note = json.loads(js.read_text()).get("note") or ""
        except (OSError, ValueError):
            continue
        found = re.search(r"peak ([\d.]+)x", note)
        if found:
            worst = max(worst, float(found.group(1)))
    return worst


def case_markers(spec: RunSpec) -> dict[str, Any]:
    """Boundary markers for a spec, in one place because it was in two.

    A shock-fitted block never extrudes the base disc, so it has no
    ``vehicle_base`` and its downstream boundary is an ``outflow`` joined to
    ``MARKER_FAR``. That was applied to the second-order case and **not** to the
    first-order prelude, which builds its own :class:`SU2Case` a hundred lines
    away -- so SU2 aborted on a marker that did not exist and every shock-fitted
    run silently lost the prelude that exists to make it survivable.

    The fallback caught it and did the right thing, which is why the run
    continued at all. But a guard catching a bug is not the same as not having
    the bug, and the fix for one setting maintained in two places is one place.
    """
    fitted = bool(spec.shock_fitted)
    # A shock-fitted block has no base disc **unless a wake was built**, and
    # then it does: the aft face becomes a wall under its own tag, which is also
    # what turns on AERO_COEFF_SURF and makes a base pressure separable from the
    # forebody force at all.
    return {
        "base_marker": "vehicle_base" if (not fitted or spec.wake) else None,
        "outflow_markers": ("outflow",) if fitted else (),
        # Imposed freestream on the outer boundary, and only on a shock-fitted
        # one. That boundary follows the shock, so it runs nearly parallel to
        # the flow and MARKER_FAR's characteristic split is ill-posed on it --
        # 70 % of its nodes above Mach 8.05 and a worst node at 0.00048 K. An
        # isotropic box's far field is broadside to the flow, tens of body
        # lengths out, and MARKER_FAR is right there.
        "imposed_inflow": fitted,
    }


def _limiter_freeze(iterations: int) -> dict[str, str]:
    """``LIMITER_ITER`` as a raw override, or nothing at all.

    Emitted only when asked for, so a record written before the field existed
    renders a config byte-identical to the one it actually ran.
    """
    return {"LIMITER_ITER": str(int(iterations))} if iterations > 0 else {}


@contextmanager
def _heartbeat(record: RunRecord, what: str, every: float = 4.0) -> Any:
    """Write elapsed time into the record while a long call runs.

    Meshing is the one stage that reports nothing while it works. The
    shock-fitted mesher takes a second or two and needs none of this; the
    isotropic one has been seen to take **tens of minutes**, and for all that
    time the card said "meshing" with a clock counting from when the run was
    *queued* -- which on a run that waited an hour in the queue reads as an hour
    of meshing and cannot be told apart from a hang.

    A thread rather than a callback because gmsh owns its own initialise and
    finalise inside ``build_domain`` and its API is not thread-safe to poll, so
    the honest thing to report is elapsed wall time and the stage, not a
    fabricated percentage.
    """
    import threading

    started = time.monotonic()
    done = threading.Event()

    def tick() -> None:
        while not done.wait(every):
            elapsed = time.monotonic() - started
            with contextlib.suppress(Exception):
                save(replace(load(record.run_id) or record, message=f"{what} ({elapsed:.0f}s)"))

    worker = threading.Thread(target=tick, daemon=True)
    worker.start()
    try:
        yield
    finally:
        done.set()
        worker.join(timeout=1.0)


class Worker:
    """The single process that turns queued records into finished runs."""

    def __init__(
        self,
        *,
        slots: int = 1,
        poll: float = 3.0,
        root: Path | None = None,
        verbose: bool = True,
    ) -> None:
        self.slots = max(1, slots)
        self.poll = poll
        self.root = root
        self.verbose = verbose
        self._jobs: dict[str, SolveJob] = {}
        #: Runs this worker stopped on purpose, so their signal exit is not
        #: recorded as a crash.
        self._intentional: set[str] = set()
        self._stopping = False
        self._frames = FrameWriter()

    # ------------------------------------------------------------------ log

    def _say(self, message: str) -> None:
        if self.verbose:
            stamp = time.strftime("%H:%M:%S")
            print(f"[{stamp}] {message}", file=sys.stderr, flush=True)

    # --------------------------------------------------------------- claims

    def acquire(self) -> None:
        """Refuse to start beside another worker.

        Two workers would both claim the same queued record and mesh into the
        same directory, and the second solver would overwrite the first's
        restart file. The lock holds a pid and its start time, so a lock left
        behind by a killed worker is recognised as stale rather than blocking
        the machine until somebody deletes a file.
        """
        held = read_lock()
        if worker_alive(held):
            assert held is not None
            where = held.get("host") or "?"
            raise RuntimeError(
                f"another worker holds the queue (pid {held.get('pid')} on {where}). "
                "Stop it first, or point --root at a different results tree."
            )
        if held:
            self._say(f"clearing a stale lock from pid {held.get('pid')}")
        self.beat()

    def beat(self) -> None:
        """Rewrite the claim. Called every pass -- it *is* the heartbeat.

        A reader in another PID namespace cannot check the pid, so the freshness
        of this file is the only evidence it has that a worker still exists.
        Stop rewriting it and the panel correctly reports no worker.
        """
        path = _lock_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        birth = pid_birth(os.getpid())
        payload = {
            "pid": os.getpid(),
            "birth": birth if birth is not None else -1,
            "ns": pid_namespace(),
            "host": socket.gethostname(),
            "at": time.time(),
            "poll": self.poll,
        }
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, path)

    def release(self) -> None:
        _lock_path().unlink(missing_ok=True)

    # ------------------------------------------------------------ the cycle

    def resume(self) -> None:
        """Reattach to solvers that outlived a previous worker.

        This is the payoff for storing a pid and a wall-clock start. A record
        left in ``solving`` whose process is still alive is picked back up,
        with its elapsed time and rate intact -- not restarted, and not
        reported as an orphan.
        """
        for record in load_all(active_only=True):
            settled = reconcile(record)
            if settled.state != "solving" or settled.pid is None:
                if settled.state != record.state:
                    self._say(f"{settled.run_id}: settled as {settled.state}")
                continue
            self._jobs[settled.run_id] = SolveJob.attach(
                settled.case_dir or ".",
                pid=settled.pid,
                started_at=settled.started_at or time.time(),
                pid_birth=settled.pid_birth,
                total=settled.total,
                ranks=settled.ranks,
            )
            minutes = (settled.elapsed or 0.0) / 60.0
            self._say(
                f"{settled.run_id}: reattached to pid {settled.pid} "
                f"({settled.spec.label()}, {minutes:.0f} min in)"
            )

    def step(self) -> None:
        """One pass: settle what has finished, then start what fits."""
        self.beat()
        solving = 0
        for record in load_all(active_only=True):
            if record.state == "queued":
                if cancel_requested(record.run_id):
                    clear_cancel(record.run_id)
                    save(
                        replace(
                            record,
                            state="stopped",
                            finished_at=time.time(),
                            message="cancelled before it started",
                        )
                    )
                    self._say(f"{record.run_id}: cancelled while queued")
                continue

            # Reconcile only what this worker is *not* already watching.
            # ``reconcile`` reconstructs an outcome from the history, which is
            # all a stranger to the run can do; for its own children this
            # process has the actual wait status, which is better evidence and
            # must not be pre-empted by a guess.
            if record.run_id not in self._jobs:
                record = reconcile(record)
                if not record.active:
                    continue

            if record.state == "solving" and self._supervise(record):
                solving += 1

        if self._stopping or solving >= self.slots:
            return

        queued = [
            r
            for r in load_all(active_only=True)
            if r.state == "queued" and not cancel_requested(r.run_id)
        ]
        if queued:
            self._begin(queued[0])

    def _supervise(self, record: RunRecord) -> bool:
        """Watch one running solve. Returns whether it is still going."""
        job = self._jobs.get(record.run_id)
        if job is None:
            # A record in ``solving`` this worker has no handle for: adopt it,
            # rather than leaving it unwatched for the life of the process.
            # Only where the pid is ours to read -- a pid from a previous
            # container's namespace names nothing here, and attaching to it
            # would watch whatever now holds that number. ``reconcile`` has
            # already settled those, so reaching this with one is a bug guard.
            if record.pid is None or not record.checkable:
                return False
            job = SolveJob.attach(
                record.case_dir or ".",
                pid=record.pid,
                started_at=record.started_at or time.time(),
                pid_birth=record.pid_birth,
                total=record.total,
                ranks=record.ranks,
            )
            self._jobs[record.run_id] = job

        if cancel_requested(record.run_id):
            self._say(f"{record.run_id}: stopping on request")
            job.stop()
            clear_cancel(record.run_id)
            self._jobs.pop(record.run_id, None)
            save(
                replace(
                    record,
                    state="stopped",
                    finished_at=time.time(),
                    message="cancelled",
                )
            )
            return False

        if job.running:
            self._draw(record)
            # Consulted live, and before the settle test, because on a
            # zero-incidence case the field can go bad thousands of iterations
            # before the solver would have stopped on its own -- and every one
            # of those iterations makes the answer worse, not better.
            #
            # The hemisphere verification measured it: against the exact
            # Newtonian 0.913677 it read -0.39 % over iterations 4000-6000 with
            # a pitot ratio of 1.3x, and +17.1 % over 10000-12000 with a ratio
            # of 7.5x. The detector crossed 2.0x at iteration 8000, between the
            # two. Stopping there keeps the answer; running the remaining four
            # thousand iterations destroys it.
            unphysical = self._unphysical(record)
            if unphysical is not None:
                self._say(f"{record.run_id}: {unphysical}")
                self._intentional.add(record.run_id)
                job.stop()
                return True
            reason = self._settled(record)
            if reason is not None:
                self._say(f"{record.run_id}: {reason}")
                self._intentional.add(record.run_id)
                job.stop()
            return True

        self._jobs.pop(record.run_id, None)
        # One last frame: the final snapshot is written as the solver exits,
        # and it is the one showing what the run actually ended up at.
        self._draw(record)
        if record.case_dir is not None:
            self._frames.forget(record.case_dir)
        code = job.returncode
        if code is None:
            # An adopted solver: this worker is not its parent, so there is no
            # exit status to read and ``reconcile`` judges it by whether the
            # history reached the iteration count -- the only evidence left.
            finished = reconcile(replace(record, pid=None, pid_birth=None))
        else:
            intentional = record.run_id in self._intentional
            self._intentional.discard(record.run_id)
            # A solver this worker stopped on purpose exits by signal, which is
            # a non-zero code and not a failure. Without remembering the intent,
            # every early stop would be recorded as a crash.
            state = "done" if code == 0 or intentional else "failed"
            message = (
                "settled early" if intentional and code != 0
                else "" if code == 0
                else f"the solver exited {code}"
            )
            if state == "done":
                # A clean exit is not a converged answer. Ask the history.
                diverged = self._diverged(record)
                if diverged is not None:
                    state, message = "failed", diverged
            finished = save(
                replace(
                    record,
                    state=state,
                    returncode=code,
                    finished_at=time.time(),
                    message=message,
                )
            )
        self._say(f"{finished.run_id}: {finished.state} {finished.message}".rstrip())
        return False

    @staticmethod
    def _unphysical(record: RunRecord) -> str | None:
        """Has the field gone somewhere no steady flow can go? Why to stop, or ``None``.

        The same pitot ceiling ``_diverged`` applies after the fact, applied
        while the solver is still running. Stagnation pressure behind a normal
        shock is a hard physical bound; a ratio a little above it is
        under-resolution at the nose, and the diverged runs on this queue read
        7x, 17x and 45893x.

        Stopping here rather than at the iteration ceiling is not only
        economy. The run is recorded ``failed`` either way -- an intentional
        stop is judged by ``_diverged``, which reads the same ratio -- but the
        snapshots and the history stop near where the solution went bad instead
        of thousands of iterations past it, which is the difference between a
        record that shows the onset and one that shows only the wreckage.
        """
        if not record.case_dir:
            return None
        peak = _worst_pitot(Path(record.case_dir))
        if peak > PITOT_CEILING_RATIO:
            return f"pitot ceiling exceeded {peak:.1f}x; stopping the solver"
        return None

    def _settled(self, record: RunRecord) -> str | None:
        """Has the force stopped moving? Returns why to stop, or ``None``.

        Reads the history the run is writing and applies the same criterion the
        blocking solver path has always used. Off unless the spec asks for it,
        because acting on it terminates a live MPI job.

        The first-order prelude is excluded: it is a different discretisation
        and its forces are not the ones being judged. The window is taken from
        the second-order history alone, which is the only thing ``history.csv``
        contains.
        """
        tolerance = float(record.spec.settle_tolerance)
        if tolerance <= 0.0 or not record.case_dir:
            return None
        history = Path(record.case_dir) / "history.csv"
        if not history.exists():
            return None
        try:
            import numpy as np

            from aether.cfd.config import Numerics
            from aether.cfd.solver import (
                _force_settled,
                read_history,
                residual_trend,
            )

            columns = read_history(history)
            lookup = {key.lower(): key for key in columns}
            key = lookup.get("cd(vehicle)") or lookup.get("cd")
            if key is None:
                return None
            drag = np.asarray(columns[key], dtype=float)
            # The **full** window, never a shortened one. Shrinking it to fit a
            # young history is how this stopped a run at 310 iterations with a
            # drift of 2.2e-3, on a mesh that was still falling one percent per
            # hundred iterations at a thousand: a 200-iteration window cannot
            # see a trend that takes thousands to play out.
            #
            # That is SU2's Cauchy mistake -- measuring smoothness over too
            # short a span and calling it convergence -- reproduced here within
            # hours of diagnosing it. `_force_settled` already refuses a history
            # shorter than its window, so passing the real one is also the
            # minimum-iteration guard.
            window = int(Numerics.force_window)
            settled, drift, ripple = _force_settled(drag, window, tolerance)
            slope, _ = residual_trend(columns, window)
        except Exception:
            return None
        if not settled:
            return None
        # The slope is *reported* and not gated on. It was gated on briefly, at
        # 0.15, and the campaign disproved the threshold: across six completed
        # sphere-cone runs the Mach 6 zero-incidence case reads +0.418 and the
        # Mach 8 alpha-4 case +0.442, both with completely clean fields, while
        # the one that carbuncled reads +0.450. There is no separation to find.
        #
        # What separates them is the pitot ratio -- 1.3-1.5x on every clean run
        # against 37.9x on the carbuncle -- so that is what gates, in
        # `_unphysical` and in `_diverged`. A rising residual here means the
        # field is still moving, which is worth printing beside the drift and
        # is not by itself evidence of a bad answer.
        return (
            f"force settled after {drag.size} iterations "
            f"(drift {drift:.2e}, ripple {ripple:.2e} against {tolerance:g}, "
            f"residual {slope:+.3f} decades/1000); stopping the solver"
        )

    @staticmethod
    def _diverged(record: RunRecord) -> str | None:
        """Did this run blow up despite exiting cleanly? Returns why, or ``None``.

        ``returncode == 0`` is not evidence of convergence. SU2 will run its
        full iteration count, print *Exit Success* and return zero on a solution
        that has diverged to a finite but absurd state. Measured on this queue:
        a Mach 10 sphere-cone returning :math:`C_D = 1.85` and
        :math:`C_L = -0.87` at zero incidence, and two archived Mach 20 runs
        returning :math:`C_D` of 9117 and 64. All three were recorded as
        ``done``, and their numbers were readable as if they were measurements.

        ``SU2Solver._detect_divergence`` does not help here twice over: the
        worker has never called it, and it looks for an ``MPI_ABORT`` or a
        "diverged" line in the log, neither of which a quiet blow-up writes.

        Two checks, both read from the history the run itself wrote:

        * a **non-finite residual** at the last iterate;
        * **lift comparable to drag on an axisymmetric body at zero incidence**,
          where lift is identically zero by symmetry. This one needs no
          threshold tuning and no reference solution -- a wrong answer cannot
          satisfy it -- which is the same property that makes
          ``pitot_limit_violation`` worth having.
        """
        if not record.case_dir:
            return None

        # The pitot ceiling first, and outside the history guard, because it is
        # the only check here that does not read ``history.csv``: it reads the
        # frame detector's own findings. Behind that guard a run that blew up
        # before writing a history skipped the one test that would have caught
        # it. Stagnation pressure behind a normal shock is a hard physical
        # bound; a small exceedance is under-resolution at the nose, but the
        # diverged runs here read 7x, 17x and 45893x against 1.1-1.6x on the
        # converged ones.
        peak = _worst_pitot(Path(record.case_dir))
        if peak > PITOT_CEILING_RATIO:
            return f"diverged: pitot ceiling exceeded {peak:.1f}x"

        history = Path(record.case_dir) / "history.csv"
        if not history.exists():
            return None
        try:
            import numpy as np

            from aether.cfd.geometries import REFERENCE_BODIES
            from aether.cfd.solver import read_history

            columns = read_history(history)
        except Exception:
            return None

        # Any density residual. A perfect-gas run writes ``rms[rho]``; a
        # multi-species run writes ``rms[rho_0]``, ``rms[rho_1]``, ... and
        # looking only for the first name silently checks nothing at all --
        # which is how two archived Mach 20 runs passed this guard while
        # carrying a drag coefficient of 9117.
        for name, series in columns.items():
            if not name.startswith("rms[rho"):
                continue
            if series.size and not np.isfinite(series[-1]):
                return f"diverged: {name} is not finite at the last iterate"

        # Judge the corrected total where it can be had: SU2's own total
        # includes an unresolved base, which on this pipeline contributes
        # *thrust* and has turned a physical run into an unphysical number.
        # → aether.cfd.forces
        drag, lift = columns.get("cd"), columns.get("cl")
        try:
            from aether.cfd.forces import resolved_forces
            from aether.cfd.geometries import reference_frame

            resolved = resolved_forces(
                Path(record.case_dir),
                record.spec.body,
                record.spec.geometry,
                record.spec.mach,
                reference_frame(record.spec.body, **record.spec.geometry).area,
                forebody_only=bool(record.spec.shock_fitted),
                wake_resolved=bool(record.spec.wake),
            )
        except Exception:
            resolved = None
        body = REFERENCE_BODIES.get(record.spec.body)
        if (
            body is not None
            and getattr(body, "axisymmetric", False)
            and abs(record.spec.alpha_deg) < 1.0e-9
            and drag is not None
            and lift is not None
            and drag.size
            and lift.size
        ):
            window = slice(-min(200, drag.size), None)
            mean_drag = float(np.mean(drag[window]))
            mean_lift = float(np.mean(lift[window]))
            if np.isfinite(mean_drag) and np.isfinite(mean_lift):  # noqa: SIM102
                if abs(mean_lift) > 0.25 * abs(mean_drag):
                    return (
                        f"diverged: CL={mean_lift:.3g} against CD={mean_drag:.3g} "
                        "on an axisymmetric body at zero incidence"
                    )

        # And an absolute bound, because the ratio test above says nothing when
        # both coefficients blow up together -- the Mach 20 run that returned
        # CD 9117 alongside CL -377 has a ratio of 0.04 and passes it. No
        # hypersonic body has a drag coefficient in the tens; a continuum
        # reference area cannot produce one.
        if drag is not None and drag.size:
            mean_drag = float(np.mean(drag[slice(-min(200, drag.size), None)]))
            if resolved is not None and np.isfinite(resolved.axial):
                mean_drag = resolved.axial
            if np.isfinite(mean_drag) and abs(mean_drag) > 10.0:
                return f"diverged: CD={mean_drag:.4g} is not a drag coefficient"
            # A body in a flow does not produce thrust. Sign, not magnitude:
            # a Mach 6 case returned CD = -0.117 over 48 resolved oscillation
            # periods and passed every other check here, because they all test
            # how big the number is and none tested whether it can exist.
            if np.isfinite(mean_drag) and mean_drag <= 0.0:
                return f"unphysical: CD={mean_drag:.4g} is not positive"

            # Did the force settle at all? A period-counted standard error the
            # size of the answer means it did not. This catches the partial
            # blow-up that the bounds above are too generous for: a Mach 12 run
            # returned CD 0.288 +/- 0.036 -- four times the Mach 8 value, with
            # a 12.6 % bar -- and passed every check above it.
            try:
                from aether.cfd.solver import _mean_uncertainty

                sigma, _ = _mean_uncertainty(drag, window=700)
            except Exception:
                sigma = float("nan")
            if np.isfinite(sigma) and np.isfinite(mean_drag) and mean_drag:  # noqa: SIM102
                if sigma > 0.1 * abs(mean_drag):
                    return (
                        f"did not settle: CD={mean_drag:.4g} +/- {sigma:.2g} "
                        f"({100 * sigma / abs(mean_drag):.0f} % of the mean)"
                    )

        # A residual that has climbed steeply over a long window is a diverging
        # solution whatever the force says. This is the check that would have
        # named the Mach 8 nose-cap carbuncle for what it was at iteration 3000
        # instead of letting it run to the iteration ceiling and be caught, at
        # the end, by a pitot ratio.
        if record.case_dir:
            try:
                from aether.cfd.config import Numerics
                from aether.cfd.solver import read_history, residual_trend

                columns = read_history(Path(record.case_dir) / "history.csv")
                slope, rise = residual_trend(columns, int(Numerics.force_window))
            except Exception:
                slope = rise = float("nan")
            if np.isfinite(slope) and slope > RESIDUAL_DIVERGENCE_LIMIT:
                return (
                    f"diverged: residual rising {slope:+.2f} decades/1000 "
                    f"({rise:+.2f} from its minimum)"
                )

        return None

    def _draw(self, record: RunRecord) -> None:
        """Render a frame if the solver has rewritten its volume snapshot.

        Done here, in the worker, for the same reason the meshing is: it is
        work, and the panel is not the place for work. Rendered inline rather
        than on a thread -- it costs about two seconds and only fires once per
        ``volume_every`` iterations, which is a minute or more apart on any run
        worth watching, and a thread would buy that back at the price of
        matplotlib state shared across one.

        A failure here is logged and dropped. A picture is a convenience; a
        solve is not, and nothing about drawing one may put a run at risk.
        """
        if record.case_dir is None:
            return
        try:
            frame = self._frames.poll(record.case_dir, history_rows(record.history))
        except Exception as error:  # the solve must not care
            self._say(f"{record.run_id}: frame failed -- {type(error).__name__}: {error}")
            return
        if frame is None:
            return
        note = f" -- {frame.note}" if frame.diverged else ""
        self._say(f"{record.run_id}: frame at iteration {frame.iteration}{note}")

    def _begin(self, record: RunRecord) -> None:
        """Mesh a queued run and launch its solver.

        The mesh is built here, in the worker, which is the point: it is the
        one expensive step that has no detached process of its own, so it has
        to happen somewhere that is not a UI.
        """
        import numpy as np

        from aether.aerodynamics.cfd.meshing import RefinementBall
        from aether.cfd.config import (
            FlowConditions,
            Numerics,
            SU2Case,
            UnsteadyRun,
            regime_for,
            regime_named,
        )
        from aether.cfd.geometries import (
            build_domain,
            build_surface,
            reference_frame,
        )

        spec = record.spec
        # A record can reach the worker without passing through the queue form
        # -- a script, an older client, a requeue. The gate is cheap, so it runs
        # here too rather than trusting that it ran somewhere else.
        from aether.cfd.preflight import preflight

        report = preflight(spec, exclude=record.run_id, deep=False)
        if not report.ok:
            reasons = "; ".join(f"{c.name}: {c.detail}" for c in report.failures)
            save(
                replace(
                    record, state="failed", finished_at=time.time(),
                    message=f"preflight refused it -- {reasons}"[:400],
                )
            )
            self._say(f"{record.run_id}: preflight refused it -- {reasons}")
            return

        directory = case_directory(record, self.root)
        directory.mkdir(parents=True, exist_ok=True)
        self._clear_previous_run(directory)
        record = save(
            replace(record, state="meshing", case_dir=str(directory), message="")
        )
        self._say(f"{record.run_id}: meshing {spec.label()}")

        mesh_path = directory / f"{spec.body}.su2"
        try:
            balls: tuple[Any, ...] = ()
            if spec.nose_cell is not None:
                surface = build_surface(spec.body, **spec.geometry)
                nose = surface.vertices[np.argmin(surface.vertices[:, 0])]
                reach = 0.08 * float(np.ptp(surface.vertices[:, 0]))
                balls = (
                    RefinementBall(
                        center=(float(nose[0]), float(nose[1]), float(nose[2])),
                        radius=reach,
                        size=spec.nose_cell,
                    ),
                )

            started = time.monotonic()
            # The geometry is a body-dependent set of names read from a record,
            # so it can only be passed as ``**``. mypy cannot rule out a key
            # colliding with one of the named parameters above, and neither can
            # anything else: ``RunSpec.geometry`` is whatever the body declares.
            # Resolved before meshing: a viscous mesh's first cell height and
            # layer count are solved for the actual flight condition, and Mach
            # alone does not fix either.
            flow = FlowConditions.at_altitude(
                mach=spec.mach,
                altitude=1.0e3 * spec.altitude_km,
                alpha=float(np.radians(spec.alpha_deg)),
                beta=float(np.radians(spec.sideslip_deg)),
            )

            if spec.shock_fitted:
                result = self._mesh_shock_fitted(record, spec, mesh_path, flow)
            else:
                with _heartbeat(record, "meshing the domain"):
                    result = build_domain(
                        spec.body,
                        mesh_path,
                        mach=spec.mach,
                        refinement=balls,
                        wall_refinement=spec.wall_refinement,
                        optimize=spec.optimize,
                        quality_threshold=spec.quality_threshold,
                        # Orient the mesh against the flow, not the body axis. With
                        # ``balls`` empty this is also what makes ``build_domain``
                        # construct the Mach-sized stagnation region at all.
                        alpha=float(np.radians(spec.alpha_deg)),
                        beta=float(np.radians(spec.sideslip_deg)),
                        viscous=spec.viscous_mesh,
                        temperature=flow.temperature,
                        pressure=flow.pressure,
                        **spec.geometry,  # type: ignore[arg-type]
                    )
            if not spec.shock_fitted:
                self._preview(record, mesh_path.parent)
            quality = result.quality
            self._say(
                f"{record.run_id}: meshed {result.n_elements} cells in "
                f"{time.monotonic() - started:.0f}s -- "
                f"{quality.summary() if quality else 'quality not measured'}"
            )

            # None leaves SU2Case on regime_for(mach), which is the default and
            # not a law -- but a hybrid RANS-LES model has to reach the regime
            # object, so a spec asking for one resolves the regime here rather
            # than deferring it. See RunSpec.regime and RunSpec.hybrid_rans_les.
            if spec.regime:
                resolved_regime = regime_named(
                    spec.regime, spec.inviscid, spec.turbulence, spec.hybrid_rans_les
                )
            elif spec.hybrid_rans_les:
                resolved_regime = regime_for(
                    spec.mach, spec.inviscid, spec.turbulence, spec.hybrid_rans_les
                )
            else:
                resolved_regime = None

            # Step and duration come from the shedding scale, not from the spec:
            # the base diameter and the freestream speed set them, so the
            # physics sizes the run. See RunSpec.unsteady.
            unsteady = None
            if spec.unsteady:
                base_diameter = _base_diameter(spec)
                unsteady = UnsteadyRun.for_base_flow(
                    base_diameter=base_diameter,
                    speed=flow.speed,
                    cycles=spec.unsteady_cycles,
                    transient_cycles=spec.unsteady_transient_cycles,
                    steps_per_cycle=spec.unsteady_steps_per_cycle,
                )
                self._say(
                    f"{record.run_id}: unsteady -- base diameter "
                    f"{1000 * base_diameter:.1f} mm, shedding "
                    f"{0.2 * flow.speed / base_diameter:.0f} Hz, step "
                    f"{1e6 * unsteady.time_step:.1f} us, {unsteady.steps} steps "
                    f"({unsteady.sampled_steps} after the transient), "
                    f"{unsteady.steps * unsteady.inner_iterations:,} implicit solves"
                )

            case = SU2Case(
                flow=flow,
                inviscid=spec.inviscid,
                turbulence=spec.turbulence,
                regime=resolved_regime,
                unsteady=unsteady,
                reference=reference_frame(spec.body, **spec.geometry),
                mesh_filename=mesh_path.name,
                **case_markers(spec),
                numerics=Numerics(
                    iterations=spec.iterations,
                    first_order_iterations=spec.first_order_iterations,
                    cfl_bounds=(0.1, spec.cfl_max),
                    limiter_coefficient=spec.limiter_coefficient,
                    convective=spec.convective,
                    limiter=spec.limiter,
                    entropy_fix=spec.entropy_fix,
                    extra=_limiter_freeze(spec.limiter_iterations),
                ),
                volume_every=spec.volume_every,
            )

            # The prelude runs here, inline, for the same reason meshing does:
            # it is bounded, it has to finish before the job that depends on it
            # can start, and the record should not say "solving" until the
            # thing being watched is the second-order run.
            restart = self._run_prelude(record, spec, directory, result.n_elements)
            (directory / "case.cfg").write_text(
                replace(case, restart=restart).render()
            )

            job = SolveJob(
                directory=directory,
                total=spec.iterations,
                elements=result.n_elements,
                ranks=spec.ranks,
            ).start()
        except Exception as error:
            # A failure here must reach the record. The notebook printed it into
            # a widget, so a point that failed while nobody was looking simply
            # went missing from the sweep.
            save(
                replace(
                    record,
                    state="failed",
                    finished_at=time.time(),
                    message=f"{type(error).__name__}: {error}",
                )
            )
            self._say(f"{record.run_id}: failed -- {type(error).__name__}: {error}")
            return

        self._jobs[record.run_id] = job
        save(
            replace(
                record,
                state="solving",
                pid=job.pid,
                pid_birth=job.pid_birth,
                pid_ns=pid_namespace(),
                started_at=job.started_at,
                ranks=job.ranks,
                elements=result.n_elements,
                total=spec.iterations,
            )
        )
        self._say(
            f"{record.run_id}: solving on {job.ranks} ranks, pid {job.pid} "
            f"({result.n_elements} cells)"
        )

    @staticmethod
    def _clear_previous_run(directory: Path) -> None:
        """Remove artefacts of whatever ran here last.

        Case directories are keyed by geometry and flight condition, so a
        re-run lands in a populated directory. Three things go wrong if it is
        not cleared:

        * ``frames/`` accumulates across runs, and the preview shows whichever
          frame has the highest iteration number -- which after a shortened
          re-run is an image from the *previous, longer* run. The dashboard
          then looks frozen while the solver is running normally.
        * a stale ``restart_flow`` can be picked up as an initial condition.
        * ``history.csv`` from a longer run outlives the shorter one that
          replaced it, and anything reading it measures the wrong case.

        The mesh is deliberately left alone: it is rewritten by the mesher and
        keeping it costs nothing if the run is cancelled before that.
        """
        import shutil

        frames = directory / "frames"
        if frames.is_dir():
            shutil.rmtree(frames, ignore_errors=True)
        # Numbered snapshots, which is the same hazard as the frames above and
        # was missed because this function predates them: a re-run that has
        # reached iteration 376 would find volume_flow_000750.dat from the run
        # before it, read that as the newest finished snapshot, and draw the
        # previous run's flow field onto the current run's card.
        for stale in directory.glob("volume_flow_*.dat"):
            stale.unlink(missing_ok=True)
        for name in (
            "history.csv",
            "restart_flow.dat",
            "restart_flow.csv",
            "solution_flow.dat",
            "solution_flow.csv",
            "surface_flow.csv",
            "volume_flow.dat",
            "su2.log",
            "su2-prelude.log",
            "prelude.cfg",
            "case.cfg",
        ):
            (directory / name).unlink(missing_ok=True)

    def _mesh_shock_fitted(
        self, record: RunRecord, spec: RunSpec, mesh_path: Path, flow: Any
    ) -> Any:
        """A wall-normal block marched to the shock, sized by cell Reynolds number.

        The building is :func:`~aether.cfd.shockfit.fit_shock_layer`, so
        the preflight gate and the worker cannot disagree about what a run will
        produce. They used to, by 8 058 poor cells.
        """
        from types import SimpleNamespace

        from aether.cfd.shockfit import fit_shock_layer, write

        fitted = fit_shock_layer(
            spec, flow, say=lambda message: self._say(f"{record.run_id}: {message}")
        )
        info = write(fitted, mesh_path)
        self._preview(record, mesh_path.parent, layer=fitted.layer, envelope=fitted.envelope)
        return SimpleNamespace(n_elements=info["prisms"], quality=fitted.quality)

    def _preview(
        self, record: RunRecord, directory: Path, layer: Any = None, envelope: Any = None
    ) -> None:
        """Draw the grid before solving on it. Never fatal.

        Meshing was the one stage with no diagnostics at all: a card sat on
        "meshing" and then returned a single element count, which says nothing
        about **where** the cells went -- the exact question this pipeline got
        wrong while reporting 1.36 million of them and zero in the shock layer.

        A preview that fails to draw must not lose a mesh that built fine, so
        every failure here is reported and swallowed.
        """
        from aether.cfd.meshpreview import render_mesh_preview

        try:
            preview = render_mesh_preview(directory, layer=layer, envelope=envelope)
        except Exception as error:
            self._say(f"{record.run_id}: mesh preview failed -- {error}")
            return
        if preview is None:
            return
        note = f"{record.run_id}: mesh preview drawn"
        if preview.in_shock_layer is not None:
            note += f" -- {preview.in_shock_layer} cells in the shock layer"
        self._say(note)

    def _run_prelude(
        self, record: RunRecord, spec: RunSpec, directory: Path, elements: int
    ) -> bool:
        """First-order pass to establish the bow shock. Returns ``restart``.

        Mirrors :meth:`aether.cfd.solver.SU2Solver.run`, which has had
        this since the near-sharp cone work; the worker never gained it, so
        every Mach 8 case queued through the dashboard died on a NaN in the
        startup transient while the fix sat one module away.

        Failure here is deliberately **not** fatal. A prelude that dies leaves
        no restart, the second-order run starts from freestream exactly as it
        did before, and the record carries the reason. That is strictly no
        worse than the old behaviour.
        """
        prelude = int(spec.first_order_iterations)
        if prelude <= 0:
            return False

        import shutil
        import subprocess

        import numpy as np

        from aether.cfd.config import FlowConditions, Numerics, SU2Case
        from aether.cfd.geometries import reference_frame
        from aether.cfd.run import _find_su2, default_ranks, find_mpirun

        starter = SU2Case(
            inviscid=spec.inviscid,
            turbulence=spec.turbulence,
            flow=FlowConditions.at_altitude(
                mach=spec.mach,
                altitude=1.0e3 * spec.altitude_km,
                alpha=float(np.radians(spec.alpha_deg)),
                beta=float(np.radians(spec.sideslip_deg)),
            ),
            reference=reference_frame(spec.body, **spec.geometry),
            mesh_filename=f"{spec.body}.su2",
            **case_markers(spec),
            numerics=Numerics(
                muscl=False,
                cfl_bounds=(0.1, spec.cfl_max),
                limiter_coefficient=spec.limiter_coefficient,
                convective=spec.convective,
                limiter=spec.limiter,
                entropy_fix=spec.entropy_fix,
                iterations=prelude,
                minimum_iterations=max(prelude + 1, 10),
                first_order_iterations=0,
            ),
            volume_every=0,
        )
        (directory / "prelude.cfg").write_text(starter.render())

        binary = _find_su2()
        ranks = spec.ranks if spec.ranks is not None else default_ranks(elements)
        command = [str(binary), "prelude.cfg"]
        if ranks > 1:
            launcher = find_mpirun()
            if launcher is not None:
                command = [launcher, "-n", str(ranks), *command]

        self._say(f"{record.run_id}: prelude, {prelude} iters first order on {ranks} ranks")
        started = time.monotonic()
        launched = time.time()
        try:
            finished = subprocess.run(
                command, cwd=directory, capture_output=True, text=True, check=False
            )
        except Exception as error:
            self._say(f"{record.run_id}: prelude could not launch -- {error}")
            return False
        (directory / "su2-prelude.log").write_text(
            finished.stdout + "\n----- stderr -----\n" + finished.stderr
        )

        # A prelude that diverged must not be promoted. SU2 prints *Exit
        # Success* on a field that has blown up -- at four degrees incidence the
        # prelude ended at a residual of 10^7.4 with a lift coefficient of -1826
        # and wrote a restart anyway, and the second-order run began from it and
        # died 63 iterations later. Falling back to freestream is strictly
        # better than starting from a diverged field: the two-degree case, whose
        # prelude crashed and so left no restart, managed 683 iterations.
        from aether.cfd.forces import (
            diverged_outright,
            residual_excursion,
            screen_residual,
        )

        trace = screen_residual(directory / "su2-prelude.log")
        if trace is not None and (residual_excursion(trace) > 5.0 or diverged_outright(trace)):
            self._say(
                f"{record.run_id}: prelude diverged (rms {trace[-1]:+.2f}, "
                f"{residual_excursion(trace):.1f} decades above its best) -- "
                "not restarting from it"
            )
            return False

        # Only a restart *this* prelude wrote. A case directory is reused
        # across runs, so a stale restart -- possibly from a different Mach
        # number -- would otherwise be handed to the second-order run as its
        # initial condition and silently corrupt the answer.
        produced = next(
            (
                p
                for p in (
                    directory / "restart_flow.dat",
                    directory / "restart_flow.csv",
                )
                if p.exists() and p.stat().st_mtime >= launched
            ),
            None,
        )
        if produced is None:
            self._say(f"{record.run_id}: prelude wrote no restart -- second order from freestream")
            return False

        # SU2 reads SOLUTION_FILENAME and writes RESTART_FILENAME; restarting
        # in place needs the first moved onto the second, or the run silently
        # starts from freestream again and dies exactly as it would have.
        shutil.move(str(produced), str(directory / f"solution_flow{produced.suffix}"))
        self._say(
            f"{record.run_id}: prelude done in {time.monotonic() - started:.0f}s, "
            "second order restarting from it"
        )
        return True

    # ----------------------------------------------------------------- loop

    def serve(self, drain: bool = False) -> int:
        """Run until signalled. Returns a process exit status."""
        self.acquire()
        for received in (signal.SIGINT, signal.SIGTERM):
            signal.signal(received, self._handle_signal)
        self._say(f"worker up: {self.slots} slot(s), queue at {queue_dir()}")
        try:
            self.resume()
            while True:
                self.step()
                if self._stopping and not (drain and self._jobs):
                    break
                time.sleep(self.poll)
        finally:
            self.release()
        left = len(self._jobs)
        self._say(
            f"worker down; {left} solver(s) still running and recoverable"
            if left
            else "worker down"
        )
        return 0

    def _handle_signal(self, _number: int, _frame: object) -> None:
        if self._stopping:
            raise KeyboardInterrupt
        self._stopping = True
        self._say("stopping after the current pass (again to force)")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--slots", type=int, default=1, help="solvers running at once (mesh builds never overlap)"
    )
    parser.add_argument("--poll", type=float, default=3.0, help="seconds between passes")
    parser.add_argument("--root", type=Path, default=None, help="default: $AETHER_RESULTS/cfd")
    parser.add_argument(
        "--drain",
        action="store_true",
        help="on shutdown, wait for running solvers instead of leaving them",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--once", action="store_true", help="one pass and exit, for a cron-style poke"
    )
    args = parser.parse_args(argv)

    worker = Worker(slots=args.slots, poll=args.poll, root=args.root, verbose=not args.quiet)
    if args.once:
        worker.acquire()
        try:
            worker.resume()
            worker.step()
        finally:
            worker.release()
        return 0
    return worker.serve(drain=args.drain)


if __name__ == "__main__":
    raise SystemExit(main())
