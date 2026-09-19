"""The queue driver, end to end, with a stand-in for SU2.

The mesh build is stubbed -- gmsh is optional and a real domain is a minute of
wall time -- but the solver is a *real detached process* writing a *real*
``history.csv``, because the thing under test is precisely the handoff between
processes. A test that faked the subprocess would exercise none of what broke
in the notebook.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from aether.cfd import registry, worker
from aether.cfd.registry import (
    RunRecord,
    RunSpec,
    enqueue,
    load,
    load_all,
    request_cancel,
)
from aether.cfd.worker import Worker, worker_pid

_HEADER = '"Time_Iter","Outer_Iter","Inner_Iter","rms[Rho]","RefForce","CD","CL"\n'


@pytest.fixture(autouse=True)
def results_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("AETHER_RESULTS", str(tmp_path))
    return tmp_path


@pytest.fixture
def fake_solver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A script that writes a history the way SU2 does, then exits.

    ``rows`` and a per-row delay come from the environment so one script serves
    both the run-to-completion case and the still-running case.
    """
    script = tmp_path / "fake_su2.py"
    script.write_text(
        textwrap.dedent(
            """
            import os, sys, time
            rows = int(os.environ.get("FAKE_ROWS", "10"))
            delay = float(os.environ.get("FAKE_DELAY", "0"))
            code = int(os.environ.get("FAKE_CODE", "0"))
            with open("history.csv", "w") as handle:
                handle.write('"Time_Iter","Outer_Iter","Inner_Iter",'
                             '"rms[Rho]","RefForce","CD","CL"\\n')
                for i in range(rows):
                    handle.write(f"0,0,{i},{-0.01 * i:.6f},1.0,0.42000000,0.001\\n")
                    handle.flush()
                    time.sleep(delay)
            sys.exit(code)
            """
        )
    )
    launcher = tmp_path / "SU2_CFD"
    launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    launcher.chmod(0o755)
    monkeypatch.setattr("aether.cfd.run._find_su2", lambda: str(launcher))
    return launcher


@pytest.fixture
def stub_mesher(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip gmsh. The domain build is not what this file is testing."""

    def build_domain(_body: Any, path: Path, **_kwargs: Any) -> Any:
        Path(path).write_text("a mesh, notionally\n")
        return SimpleNamespace(n_elements=417_000, n_nodes=80_000, quality=None)

    monkeypatch.setattr("aether.cfd.geometries.build_domain", build_domain)


def _spec(**overrides: Any) -> RunSpec:
    base: dict[str, Any] = {"body": "sphere-cone", "mach": 8.0, "iterations": 10, "ranks": 1}
    base.update(overrides)
    return RunSpec(**base)


def _abandon(w: Worker) -> None:
    """Simulate the worker process dying, without killing what it started.

    A real worker that is killed takes its ``Popen`` objects with it and leaves
    the solvers running -- which is the situation being tested. Inside one
    process the objects survive, and Python warns at collection about a child
    that was never waited on; marking them reaped silences that without going
    anywhere near the actual processes, which must keep running.
    """
    for job in w._jobs.values():
        if job._process is not None:
            job._process.returncode = 0
    w._jobs.clear()


def _drive(w: Worker, until: Any, limit: float = 30.0) -> None:
    """Step the worker until a condition holds, or fail loudly."""
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        w.step()
        if until():
            return
        time.sleep(0.05)
    raise AssertionError("the worker never reached the expected state")


# ------------------------------------------------------------------- the loop


def test_a_queued_run_is_meshed_solved_and_recorded_done(
    fake_solver: Path, stub_mesher: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_ROWS", "10")
    record = enqueue(_spec())
    w = Worker(verbose=False)

    _drive(w, lambda: (load(record.run_id) or record).state == "done")

    done = load(record.run_id)
    assert done is not None
    assert done.returncode == 0
    assert done.elements == 417_000
    assert done.ranks == 1
    # The durable half: a case directory with the config and history in it.
    directory = Path(done.case_dir or "")
    assert (directory / "case.cfg").exists()
    assert (directory / "history.csv").exists()
    assert (directory / "sphere-cone.su2").exists()
    # And the identity that made it watchable from elsewhere.
    assert done.started_at is not None and done.finished_at is not None
    assert done.elapsed is not None and done.elapsed >= 0.0


def test_a_solver_that_exits_nonzero_is_recorded_failed(
    fake_solver: Path, stub_mesher: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The notebook printed this into a widget and moved on."""
    monkeypatch.setenv("FAKE_ROWS", "3")
    monkeypatch.setenv("FAKE_CODE", "7")
    record = enqueue(_spec())
    w = Worker(verbose=False)

    _drive(w, lambda: not (load(record.run_id) or record).active)

    failed = load(record.run_id)
    assert failed is not None
    assert failed.state == "failed" and failed.returncode == 7
    assert "7" in failed.message


def test_a_mesh_failure_lands_on_the_record_rather_than_a_log(
    fake_solver: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("gmsh said no")

    monkeypatch.setattr("aether.cfd.geometries.build_domain", explode)
    record = enqueue(_spec())
    Worker(verbose=False).step()

    failed = load(record.run_id)
    assert failed is not None
    assert failed.state == "failed"
    assert "gmsh said no" in failed.message


def test_slots_bound_how_many_solvers_run_at_once(
    fake_solver: Path, stub_mesher: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_ROWS", "400")
    monkeypatch.setenv("FAKE_DELAY", "0.05")
    for mach in (4.0, 6.0, 8.0):
        # No prelude: this is about slot accounting, and a prelude is a second
        # solver pass per run. With one it became a race -- the first run
        # finished during the drive, freed its slot, and the third started, so
        # the old "exactly one still queued" read 0 and failed a worker that was
        # behaving correctly.
        enqueue(_spec(mach=mach, iterations=400, first_order_iterations=0))

    w = Worker(slots=2, verbose=False)
    try:
        _drive(w, lambda: sum(r.state == "solving" for r in load_all()) == 2)
        # A third pass must not start the third run while both slots are held.
        w.step()
        states = [r.state for r in load_all()]
        assert states.count("solving") == 2
        # The invariant, stated so it does not depend on how fast a run ends:
        # nothing is running beyond the slots, and no run has been skipped.
        assert states.count("queued") + states.count("done") == 1
    finally:
        for job in list(w._jobs.values()):
            job.stop(timeout=5.0)


# ----------------------------------------------------------------- cancelling


def test_cancelling_a_queued_run_never_starts_it(
    fake_solver: Path, stub_mesher: None
) -> None:
    record = enqueue(_spec())
    request_cancel(record.run_id)
    Worker(verbose=False).step()

    stopped = load(record.run_id)
    assert stopped is not None
    assert stopped.state == "stopped"
    assert stopped.case_dir is None  # never meshed


def test_cancelling_a_running_solve_stops_the_process(
    fake_solver: Path, stub_mesher: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_ROWS", "10000")
    monkeypatch.setenv("FAKE_DELAY", "0.05")
    record = enqueue(_spec(iterations=10000))
    w = Worker(verbose=False)

    _drive(w, lambda: (load(record.run_id) or record).state == "solving")
    live = load(record.run_id)
    assert live is not None and live.pid is not None
    assert registry.process_alive(live.pid, live.pid_birth)

    request_cancel(record.run_id)
    _drive(w, lambda: (load(record.run_id) or record).state == "stopped")

    stopped = load(record.run_id)
    assert stopped is not None
    assert not registry.process_alive(live.pid, live.pid_birth)
    assert not registry.cancel_requested(record.run_id)


# ---------------------------------------------------------------- reattaching


def test_a_new_worker_reattaches_to_a_solve_the_old_one_left_running(
    fake_solver: Path, stub_mesher: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure this whole subsystem exists to fix, reproduced deliberately.

    The first worker is abandoned mid-solve, exactly as a killed kernel would
    be. A second one must find the run, recognise it as live, and finish it --
    not restart it, and not report it as an orphan.
    """
    monkeypatch.setenv("FAKE_ROWS", "60")
    monkeypatch.setenv("FAKE_DELAY", "0.05")
    record = enqueue(_spec(iterations=60))

    first = Worker(verbose=False)
    _drive(first, lambda: (load(record.run_id) or record).state == "solving")
    running = load(record.run_id)
    assert running is not None and running.pid is not None
    _abandon(first)  # the old worker is gone; its handles die with it
    del first

    second = Worker(verbose=False)
    second.resume()
    assert record.run_id in second._jobs
    # Reattached, not restarted: same process, and its clock is intact.
    job = second._jobs[record.run_id]
    assert job.pid == running.pid
    progress = job.progress()
    assert progress is None or progress.elapsed >= 0.0

    _drive(second, lambda: not (load(record.run_id) or record).active)
    finished = load(record.run_id)
    assert finished is not None
    # No exit status survives a parent that is gone, so the history is the
    # evidence: it reached the requested iteration count, so it is done.
    assert finished.state == "done"
    assert finished.returncode is None


def test_a_run_whose_solver_died_unobserved_is_settled_on_the_next_look(
    fake_solver: Path, stub_mesher: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_ROWS", "4")
    record = enqueue(_spec(iterations=6000))

    first = Worker(verbose=False)
    _drive(first, lambda: (load(record.run_id) or record).state == "solving")
    _abandon(first)
    del first

    # The solver ran out of rows and exited while nobody was watching.
    live = load(record.run_id)
    assert live is not None and live.pid is not None
    deadline = time.monotonic() + 20.0
    while registry.process_alive(live.pid, live.pid_birth):
        assert time.monotonic() < deadline, "the stand-in solver never exited"
        time.sleep(0.05)

    Worker(verbose=False).resume()
    settled = load(record.run_id)
    assert settled is not None
    assert settled.state == "failed"
    assert "unobserved" in settled.message


# ---------------------------------------------------------------------- lock


def test_two_workers_cannot_own_one_queue() -> None:
    first = Worker(verbose=False)
    first.acquire()
    try:
        assert worker_pid() is not None
        with pytest.raises(RuntimeError, match="another worker"):
            Worker(verbose=False).acquire()
    finally:
        first.release()
    assert worker_pid() is None


def test_a_lock_left_by_a_dead_worker_is_cleared() -> None:
    """Otherwise a killed worker locks the machine out until somebody notices."""
    lock = registry.queue_dir() / "worker.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("999999999 1\n")
    assert worker_pid() is None

    w = Worker(verbose=False)
    w.acquire()
    try:
        assert worker_pid() is not None
    finally:
        w.release()


def test_the_case_directory_follows_the_notebooks_layout() -> None:
    record = enqueue(_spec(geometry={"nose_radius": 0.06}))
    directory = worker.case_directory(record)
    assert directory.parent.parent.name == "gen-sphere-cone"
    assert directory.parent.name == "cases"
    assert directory.name == "m08.0000-a+0.00-h45.0-nos0.06"


# ------------------------------------------------------- the lock heartbeat


def test_a_worker_in_another_namespace_is_seen_through_its_heartbeat() -> None:
    """The bug this replaced: a panel reporting "no worker" at a busy worker.

    The panel and the worker are normally separate containers, so the worker's
    pid means nothing in the panel's process table. Checking it directly said
    "absent" while the worker was meshing. The claim's freshness is what a
    foreign reader has to go on instead.
    """
    from aether.cfd.worker import _lock_path, worker_alive

    Worker(verbose=False).acquire()
    claim = json.loads(_lock_path().read_text())
    assert claim["ns"] == registry.pid_namespace()
    assert worker_alive(claim)

    # Same claim, read from somewhere the pid cannot be checked. The pid here
    # is 1, which exists in every namespace -- so a naive check would pass for
    # the wrong reason; this must pass for the right one.
    foreign = {**claim, "ns": "pid:[4026500000]", "pid": 1, "at": time.time()}
    assert worker_alive(foreign)

    stale = {**foreign, "at": time.time() - 3600.0}
    assert not worker_alive(stale)


def test_a_heartbeat_that_stopped_is_not_a_worker() -> None:
    from aether.cfd.worker import worker_alive

    assert not worker_alive(None)
    assert not worker_alive({})
    # No timestamp and a pid we cannot check is not evidence of anything.
    assert not worker_alive({"pid": 1, "ns": "pid:[4026500000]"})


def test_the_lock_is_rewritten_every_pass(fake_solver: Path, stub_mesher: None) -> None:
    """Because a foreign reader judges the worker by how fresh the claim is."""
    from aether.cfd.worker import _lock_path

    w = Worker(verbose=False)
    w.acquire()
    first = json.loads(_lock_path().read_text())["at"]
    time.sleep(0.05)
    w.step()
    assert json.loads(_lock_path().read_text())["at"] > first


def test_the_prelude_and_the_solve_agree_on_which_markers_exist() -> None:
    """One setting maintained in two places is how the prelude died.

    A shock-fitted block never extrudes the base disc, so it has no
    ``vehicle_base``. That was applied where the second-order case is built and
    not where the prelude builds its own, a hundred lines away, so SU2 aborted
    on a missing marker and every shock-fitted run lost the prelude that exists
    to make it survivable. The fallback caught it -- but a guard catching a bug
    is not the same as not having it.
    """
    from aether.cfd.registry import RunSpec
    from aether.cfd.worker import case_markers

    geometry = {"nose_radius": 0.06, "length": 2.0, "half_angle": 10.0}
    isotropic = RunSpec(
        body="sphere-cone", mach=8.0, altitude_km=45.0, geometry=geometry
    )
    fitted = dataclasses.replace(isotropic, shock_fitted=True)

    assert case_markers(isotropic) == {
        "base_marker": "vehicle_base",
        "outflow_markers": (),
        "imposed_inflow": False,
    }
    assert case_markers(fitted) == {
        "base_marker": None,
        "outflow_markers": ("outflow",),
        "imposed_inflow": True,
    }


def test_a_shock_fitted_config_never_names_a_marker_its_mesh_lacks() -> None:
    """End to end: render the config and look for the marker that killed it."""
    from aether.cfd.config import FlowConditions, SU2Case
    from aether.cfd.geometries import reference_frame
    from aether.cfd.registry import RunSpec
    from aether.cfd.worker import case_markers

    geometry = {"nose_radius": 0.06, "length": 2.0, "half_angle": 10.0}
    spec = RunSpec(
        body="sphere-cone", mach=8.0, altitude_km=45.0, geometry=geometry,
        turbulence=None, shock_fitted=True,
    )
    rendered = SU2Case(
        flow=FlowConditions.at_altitude(mach=8.0, altitude=45_000.0),
        reference=reference_frame("sphere-cone", **geometry),
        mesh_filename="sphere-cone.su2",
        turbulence=None,
        **case_markers(spec),
    ).render()

    assert "vehicle_base" not in rendered
    # The outer boundary imposes freestream on a shock-fitted mesh and the rim
    # extrapolates -> test_a_shock_fitted_outer_boundary_imposes_freestream...
    assert "MARKER_SUPERSONIC_INLET= ( farfield," in rendered
    assert "MARKER_FAR= ( outflow )" in rendered
    assert "MARKER_ISOTHERMAL= ( vehicle, 300 )" in rendered


def test_the_shock_fitted_mesher_refines_the_surface_not_only_the_layers() -> None:
    """Cell count must scale as r^3, not r.

    This is the test that was missing. ``resolution`` was wired into the wrong
    ``build_surface`` call -- a replace that matched the isotropic path's site
    instead of the shock-fitted one -- so the surface stayed at 7632 faces while
    the layers refined alone. The counts came out 297648, 419760, 595296:
    scaling as r, and recorded under refinement labels of 0.7071, 1.0 and
    1.4142 that they did not have.

    Asserting the plumbing exists would not have caught it. Asserting the cell
    count moves the way a uniform refinement must does.
    """
    from aether.cfd.extrude import extrude_to_shock
    from aether.cfd.geometries import build_surface
    from aether.cfd.gridding import grid_requirement, growth_for, stagnation_pressure
    from aether.cfd.shock import envelope_for

    geometry = {"nose_radius": 0.06, "length": 2.0, "half_angle": 10.0}
    envelope = envelope_for("sphere-cone", geometry, 8.0)
    base = grid_requirement(
        envelope.standoff, stagnation_pressure(8.0, 149.101418), 300.0, cell_reynolds=100.0
    )

    faces, cells = [], []
    for level in (1.0, 1.4142):
        surface = build_surface("sphere-cone", resolution=level, **geometry)
        layers = max(2, round(base.cells_across_layer * level))
        extruded = extrude_to_shock(
            surface,
            envelope,
            base.wall_cell,
            layers=layers,
            growth=growth_for(base.wall_cell, base.outer_boundary, layers),
        )
        faces.append(extruded.faces.shape[0])
        cells.append(extruded.faces.shape[0] * extruded.layers)

    # The surface must move. It did not, and that was the whole bug.
    assert faces[1] / faces[0] == pytest.approx(2.0, rel=0.05)
    # And the total must scale as r^3, not r.
    assert cells[1] / cells[0] == pytest.approx(1.4142**3, rel=0.06)


def test_forgetting_a_record_archives_it_rather_than_deleting_it(tmp_path, monkeypatch) -> None:
    """The record is the only index into a case directory.

    Clearing finished runs reads like tidying, and it is -- but a deleted
    record leaves the results on disk and invisible to the report and the
    dashboard alike, which is a worse outcome than the button suggests. It
    happened: 37 case directories survived a sweep that removed every record
    pointing at them.
    """
    from aether.cfd import registry

    monkeypatch.setenv("AETHER_RESULTS", str(tmp_path))
    spec = registry.RunSpec(
        body="sphere-cone", mach=8.0, altitude_km=45.0,
        geometry={"nose_radius": 0.06, "length": 2.0, "half_angle": 10.0},
    )
    record = registry.enqueue(spec)
    assert [r.run_id for r in registry.load_all()] == [record.run_id]

    registry.forget(record.run_id)

    # Gone from the queue, and from every scan of it.
    assert registry.load_all() == []
    # But recoverable, and the archive is not itself scanned.
    assert registry.archived() == [record.run_id]
    assert registry.restore(record.run_id) is True
    assert [r.run_id for r in registry.load_all()] == [record.run_id]
    assert registry.archived() == []
    assert registry.restore("no-such-run") is False


def _history(path: Path, drag: list[float]) -> None:
    path.write_text(
        "Inner_Iter,rms[Rho],CD(vehicle),CL(vehicle)\n"
        + "\n".join(f"{i},-5.0,{d},0.0" for i, d in enumerate(drag))
    )


def test_a_settled_run_is_stopped_and_recorded_as_done(tmp_path) -> None:
    """A fixed iteration count is a bound, not a criterion.

    It is wrong in both directions: the isotropic mesh settles inside two
    hundred iterations and burns eight hundred more, while a wall-resolved mesh
    was still falling one percent per hundred at the cap.
    """
    from aether.cfd.worker import Worker

    worker = Worker.__new__(Worker)
    spec = RunSpec(
        body="sphere-cone", mach=8.0, altitude_km=45.0,
        geometry={"nose_radius": 0.06}, settle_tolerance=5.0e-3,
    )
    record = SimpleNamespace(run_id="r", case_dir=str(tmp_path), spec=spec)

    # Converged: flat, so the ripple path resolves it and the drift is nil.
    # Long enough to fill the full force window -- a shorter history is refused
    # by design, because a short window cannot see a slow trend.
    _history(tmp_path / "history.csv", [0.1] * 2500)
    reason = worker._settled(record)
    assert reason is not None and "force settled" in reason

    # Still falling one percent per hundred: must not be stopped.
    _history(tmp_path / "history.csv", [0.2 - 0.00005 * i for i in range(2500)])
    assert worker._settled(record) is None


def test_settling_is_off_unless_asked_for(tmp_path) -> None:
    """Acting on it terminates a live MPI job, so it is not inherited."""
    from aether.cfd.worker import Worker

    worker = Worker.__new__(Worker)
    _history(tmp_path / "history.csv", [0.1] * 2500)
    spec = RunSpec(body="sphere-cone", mach=8.0, altitude_km=45.0, geometry={})
    assert spec.settle_tolerance == 0.0
    assert worker._settled(SimpleNamespace(run_id="r", case_dir=str(tmp_path), spec=spec)) is None


def test_an_unreadable_history_never_stops_a_healthy_run(tmp_path) -> None:
    from aether.cfd.worker import Worker

    worker = Worker.__new__(Worker)
    spec = RunSpec(
        body="sphere-cone", mach=8.0, altitude_km=45.0,
        geometry={}, settle_tolerance=5.0e-3,
    )
    record = SimpleNamespace(run_id="r", case_dir=str(tmp_path), spec=spec)
    assert worker._settled(record) is None  # no file at all
    (tmp_path / "history.csv").write_text("not,a,history\n1,2\n")
    assert worker._settled(record) is None


def test_the_meshing_heartbeat_reports_elapsed_and_stops(tmp_path, monkeypatch) -> None:
    """Meshing is the one stage that reports nothing while it works.

    The shock-fitted mesher takes a second and needs none of this. The isotropic
    one has been seen to take tens of minutes, and the card showed a clock
    counting from when the run was *queued* -- so a run that waited an hour for
    a slot read as an hour of meshing, which cannot be told apart from a hang.
    """
    import time as _time

    from aether.cfd import registry, worker

    monkeypatch.setenv("AETHER_RESULTS", str(tmp_path))
    record = registry.enqueue(
        registry.RunSpec(
            body="sphere-cone", mach=8.0, altitude_km=45.0,
            geometry={"nose_radius": 0.06, "length": 2.0, "half_angle": 10.0},
        )
    )

    with worker._heartbeat(record, "meshing the domain", every=0.05):
        _time.sleep(0.25)
        note = (registry.load(record.run_id) or record).message
    assert note.startswith("meshing the domain")
    assert note.endswith("s)")

    # The thread must not outlive the block; nothing further is written.
    settled = (registry.load(record.run_id) or record).message
    _time.sleep(0.2)
    assert (registry.load(record.run_id) or record).message == settled


def test_the_heartbeat_never_lets_a_write_failure_kill_the_mesh(tmp_path, monkeypatch) -> None:
    """A progress note that cannot be saved must not lose a mesh that built."""
    from aether.cfd import worker

    broken = SimpleNamespace(run_id="nope", spec=None)
    with worker._heartbeat(broken, "meshing", every=0.05):
        import time as _time

        _time.sleep(0.15)


def test_settling_needs_a_full_window_not_a_shortened_one(tmp_path) -> None:
    """A short window cannot see a slow trend, and will call it converged.

    Measured: a run stopped at 310 iterations with a drift of 2.2e-3, on a mesh
    still falling one percent per hundred iterations at a thousand. That is
    SU2's Cauchy failure -- smoothness over too short a span read as
    convergence -- reproduced here within hours of diagnosing it.
    """
    from aether.cfd.worker import Worker

    worker = Worker.__new__(Worker)
    spec = RunSpec(
        body="sphere-cone", mach=8.0, altitude_km=45.0,
        geometry={}, settle_tolerance=5.0e-3,
    )
    record = SimpleNamespace(run_id="r", case_dir=str(tmp_path), spec=spec)

    # 400 iterations of a slow, smooth fall: locally flat, globally moving.
    _history(tmp_path / "history.csv", [0.2 - 2.0e-5 * i for i in range(400)])
    assert worker._settled(record) is None, "a young history must not settle"

    # The same drift sustained past the window is still not settled.
    _history(tmp_path / "history.csv", [0.2 - 2.0e-5 * i for i in range(2500)])
    assert worker._settled(record) is None

    # Genuinely flat over the full window does settle.
    _history(tmp_path / "history.csv", [0.1] * 2500)
    assert worker._settled(record) is not None


def test_a_rerun_does_not_inherit_the_previous_run_s_snapshots(tmp_path: Path) -> None:
    """Same hazard the frames cleanup exists for, one level down.

    Case directories are keyed by geometry and flight condition, so a re-run
    lands in a populated one. Numbered volume snapshots were added after that
    cleanup was written and were not added to it, so a re-run at iteration 376
    would find volume_flow_000750.dat from the run before, take it as the newest
    finished snapshot -- it is finished, just not this run's -- and draw the
    previous flow field onto the current card.
    """
    for name in ("volume_flow_000750.dat", "volume_flow_000775.dat", "volume_flow.dat"):
        (tmp_path / name).write_text("stale")
    (tmp_path / "history.csv").write_text("stale")
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / "iter_000750.png").write_bytes(b"stale")
    keep = tmp_path / "sphere-cone.su2"
    keep.write_text("the mesh is rewritten by the mesher and costs nothing to keep")

    Worker._clear_previous_run(tmp_path)

    assert not list(tmp_path.glob("volume_flow*.dat"))
    assert not (tmp_path / "frames").exists()
    assert not (tmp_path / "history.csv").exists()
    assert keep.exists()


def test_a_shock_fitted_outer_boundary_imposes_freestream_rather_than_extrapolating() -> None:
    """MARKER_FAR is a characteristic condition and this boundary defeats it.

    A shock-fitted outer boundary follows the shock, so it runs nearly parallel
    to the flow -- 13 degrees on the Mach 8 sphere-cone. The normal velocity is
    then a small difference of large numbers, the characteristic split is
    ill-posed, and the solution drifts: measured, **70 % of the 4033 boundary
    nodes above Mach 8.05**, a median of 8.53 against a freestream 8.0, and a
    worst node expanded to 0.00048 K and 0.00024 Pa.

    The domain is *defined* as the region where the flow differs from
    freestream, so its outer boundary is a surface on which the flow **is**
    freestream. Imposing it is the construction's own premise, not an
    approximation, and it is well posed at any angle.

    An isotropic box's far field is broadside to the flow and tens of body
    lengths out, where MARKER_FAR is exactly right -- so this is keyed on the
    mesh, not applied everywhere.
    """
    from aether.cfd.config import FlowConditions, ReferenceFrame, SU2Case
    from aether.cfd.worker import case_markers

    flow = FlowConditions.at_altitude(8.0, 45.0e3)
    reference = ReferenceFrame(area=0.5326, length=2.04958)

    fitted = case_markers(_spec(shock_fitted=True))
    assert fitted["imposed_inflow"] is True
    text = SU2Case(flow, reference, "m.su2", **fitted).render()
    assert "MARKER_SUPERSONIC_INLET= ( farfield," in text
    assert "MARKER_FAR= ( outflow )" in text

    box = case_markers(_spec(shock_fitted=False))
    assert box["imposed_inflow"] is False
    assert "MARKER_FAR= ( farfield )" in SU2Case(
        flow, reference, "m.su2", **box
    ).render()


# ------------------------------------------------------- residual health
#
# A Mach 8 sphere-cone ran to 3453 iterations with CD creeping 0.14420 ->
# 0.14459: smooth, monotone, no ripple, which is exactly what a drift-and-ripple
# test reads as settled. Over the same span rms[Rho] climbed -4.88 -> -4.39, and
# the volume snapshot at the stopping point held 670 nodes above 6000 K with a
# peak of 590,410 K -- a few hundred cells in one azimuthal sector of the nose
# cap running away while the other 221,000 sat at a correct 3400 K. The
# integrated force cannot feel a few hundred cells. Nothing was reading the
# residual.


def _record(tmp_path: Path) -> RunRecord:
    """A record pointing at a case directory holding only a history.csv."""
    return RunRecord(
        run_id="residual-health",
        spec=_spec(settle_tolerance=0.005),
        state="solving",
        case_dir=str(tmp_path),
    )


def _residual_history(tmp_path: Path, drag, residual) -> Path:
    import numpy as np

    rows = ["Time_Iter,Outer_Iter,Inner_Iter,rms[Rho],CD"]
    for k, (d, r) in enumerate(zip(np.asarray(drag), np.asarray(residual), strict=True)):
        rows.append(f"0,0,{k},{r},{d}")
    path = tmp_path / "history.csv"
    path.write_text("\n".join(rows) + "\n")
    return path


def test_residual_trend_reads_the_slope_in_decades_per_thousand() -> None:
    import numpy as np

    from aether.cfd.solver import residual_trend

    n = 2000
    rising = {"rms[rho]": np.linspace(-5.0, -4.5, n)}
    slope, rise = residual_trend(rising, n)
    assert slope == pytest.approx(0.25, rel=0.02), "0.5 decades over 2000 iterations"
    assert rise == pytest.approx(0.5, rel=0.02)

    falling = {"rms[rho]": np.linspace(-2.0, -7.0, n)}
    slope, rise = residual_trend(falling, n)
    assert slope == pytest.approx(-2.5, rel=0.02)
    assert rise == pytest.approx(0.0, abs=1e-9), "a falling residual ends at its minimum"


def test_residual_trend_reports_the_worst_equation_not_the_average() -> None:
    """A solution can be losing one equation while the others converge."""
    import numpy as np

    from aether.cfd.solver import residual_trend

    n = 2000
    columns = {
        "rms[rho]": np.linspace(-4.0, -8.0, n),
        "rms[rhoe]": np.linspace(-4.0, -3.0, n),
    }
    slope, _ = residual_trend(columns, n)
    assert slope == pytest.approx(0.5, rel=0.02)


def test_residual_trend_says_nothing_when_there_is_nothing_to_say() -> None:
    import numpy as np

    from aether.cfd.solver import residual_trend

    assert all(np.isnan(x) for x in residual_trend({}, 500))
    assert all(np.isnan(x) for x in residual_trend({"cd": np.zeros(10)}, 500))
    assert all(
        np.isnan(x) for x in residual_trend({"rms[rho]": np.zeros(10)}, 500)
    ), "a history shorter than the window is not a trend"


def test_the_settle_reason_reports_the_residual_trend_without_gating_on_it(
    tmp_path: Path,
) -> None:
    """A gate at 0.15 was tried here and the campaign disproved it: Mach 6
    alpha 0 reads +0.418 and Mach 8 alpha 4 reads +0.442, both with clean
    fields, against +0.450 on the one that carbuncled. The slope does not
    separate them. It is printed because it says the field is still moving; the
    pitot ratio is what decides whether the answer is any good."""
    import numpy as np

    from aether.cfd.config import Numerics
    from aether.cfd.solver import _force_settled

    n = int(Numerics.force_window) + 200
    drag = np.linspace(0.14420, 0.14459, n)
    settled, _, _ = _force_settled(drag, int(Numerics.force_window), 0.005)
    assert settled

    record = _record(tmp_path)
    worker_ = Worker(verbose=False)
    for first, last in ((-4.88, -4.39), (-4.39, -5.50)):
        _residual_history(tmp_path, drag, np.linspace(first, last, n))
        reason = worker_._settled(record)
        assert reason is not None and "force settled" in reason
        assert "residual" in reason, "the reason must quote the trend it read"


def test_a_steeply_rising_residual_is_reported_as_divergence(tmp_path: Path) -> None:
    """Named at the point it happens, rather than left to burn the iteration
    budget and be caught at the end by a pitot ratio."""
    import numpy as np

    from aether.cfd.config import Numerics

    n = int(Numerics.force_window) + 200
    record = _record(tmp_path)

    _residual_history(tmp_path, np.full(n, 0.14), np.linspace(-4.5, -1.0, n))
    reason = Worker._diverged(record)
    assert reason is not None and "residual rising" in reason

    _residual_history(tmp_path, np.full(n, 0.14), np.linspace(-1.0, -4.5, n))
    assert Worker._diverged(record) is None


def _frame_note(tmp_path: Path, iteration: int, peak: float) -> None:
    frames = tmp_path / "frames"
    frames.mkdir(exist_ok=True)
    (frames / f"iter_{iteration:06d}.json").write_text(
        json.dumps(
            {
                "iteration": iteration,
                "note": (
                    f"1200 nodes (0.54%) above the pitot ceiling; peak {peak}x at "
                    "x=-0.0072 y=-0.0041 z=0.0062, T 93241 K; region spans x -0.01..0.0"
                ),
            }
        )
    )


def test_a_field_above_the_pitot_ceiling_stops_the_solver_where_it_went_bad(
    tmp_path: Path,
) -> None:
    """Measured on the hemisphere verification: against the exact Newtonian
    0.913677 it read -0.39 % over iterations 4000-6000 at a pitot ratio of 1.3x,
    and +17.1 % over 10000-12000 at 7.5x. The ratio crossed 2.0 at iteration
    8000, between the two. Every iteration after that made the answer worse.
    """
    record = _record(tmp_path)

    _frame_note(tmp_path, 1000, 1.3)
    _frame_note(tmp_path, 2000, 1.6)
    assert Worker._unphysical(record) is None, "under-resolution at the nose is not a blow-up"

    _frame_note(tmp_path, 3000, 7.5)
    reason = Worker._unphysical(record)
    assert reason is not None
    assert "7.5x" in reason and "stopping" in reason


def test_a_run_stopped_for_an_unphysical_field_is_recorded_failed(
    tmp_path: Path,
) -> None:
    """An intentional stop is otherwise recorded done; this one must not be.
    ``_diverged`` reads the same ratio, so the two agree by construction."""
    record = _record(tmp_path)
    _frame_note(tmp_path, 3000, 7.5)
    assert Worker._unphysical(record) is not None
    diverged = Worker._diverged(record)
    assert diverged is not None and "pitot ceiling exceeded" in diverged


# ------------------------------------------------------------ gas model
#
# `regime_for` is documented as "a default, not a law; pass a regime explicitly
# to overrule it" -- and the queue had no way to. Every case at Mach 10 and
# above went to NEMO, and not one produced a usable number: Mach 10 segfaults
# before the first iteration, and Mach 12/14/16/18/20 diverge to residuals of
# +9 to +13 decades, returning CD of 50, 2642, 26, 228 and -4.4.


def test_a_spec_can_overrule_the_mach_number_default_gas_model() -> None:
    from aether.cfd.config import (
        LowMach,
        Nonequilibrium,
        PerfectGas,
        regime_named,
    )

    assert isinstance(regime_named("perfect-gas"), PerfectGas)
    assert isinstance(regime_named("nonequilibrium"), Nonequilibrium)
    assert isinstance(regime_named("low-mach"), LowMach)
    with pytest.raises(ValueError, match="unknown regime"):
        regime_named("ideal gas, I assume")


def test_nemo_takes_no_turbulence_model_however_it_is_reached() -> None:
    """Asking for SST above Mach 10 is asking for something that does not
    exist. `regime_for` drops it, and NEMO does not carry the attribute at all
    -- so the named form must not smuggle one in either."""
    from aether.cfd.config import regime_for, regime_named

    assert not hasattr(regime_named("nonequilibrium", turbulence="SST"), "turbulence")
    assert not hasattr(regime_for(12.0, turbulence="SST"), "turbulence")
    # And the perfect-gas path does carry it, so the assertion above is not
    # passing merely because nothing anywhere has the attribute.
    assert regime_named("perfect-gas", turbulence="SST").turbulence == "SST"


def test_forcing_perfect_gas_at_mach_ten_changes_the_solver() -> None:
    """The substitution is real and has to be visible in what SU2 is handed."""
    from aether.cfd.config import FlowConditions, Numerics, SU2Case, regime_named
    from aether.cfd.geometries import reference_frame

    flow = FlowConditions.at_altitude(mach=10.0, altitude=45.0e3, alpha=0.0)
    ref = reference_frame("sphere-cone", nose_radius=0.06, length=2.0, half_angle=10.0)

    def solver_line(regime) -> str:
        case = SU2Case(flow=flow, inviscid=False, turbulence=None, reference=ref,
                       mesh_filename="m.su2", regime=regime,
                       numerics=Numerics(iterations=10))
        return next(x for x in case.render().splitlines() if x.startswith("SOLVER="))

    assert solver_line(None) == "SOLVER= NEMO_NAVIER_STOKES"
    laminar = regime_named("perfect-gas", turbulence=None)
    assert solver_line(laminar) == "SOLVER= NAVIER_STOKES"
    # And the turbulence model still reaches the solver when one is asked for.
    assert solver_line(regime_named("perfect-gas", turbulence="SST")) == "SOLVER= RANS"


def test_the_regime_override_survives_a_registry_round_trip() -> None:
    """A spec that cannot be written and read back has not overruled anything.

    Records written before the field existed must still load, and must not
    acquire a regime they never ran with -- the mistake `first_order_iterations`
    made, where a default meant two different things to two callers.
    """
    record = RunRecord(run_id="r", spec=_spec(mach=10.0, regime="perfect-gas"), state="queued")
    assert RunRecord.from_json(record.to_json()).spec.regime == "perfect-gas"

    plain = RunRecord(run_id="r", spec=_spec(mach=8.0), state="queued")
    assert RunRecord.from_json(plain.to_json()).spec.regime is None

    legacy = plain.to_json()
    legacy["spec"].pop("regime", None)
    assert RunRecord.from_json(legacy).spec.regime is None


def test_the_gas_model_reaches_the_case_directory_name() -> None:
    """A perfect-gas Mach 10 and a NEMO Mach 10 are different answers to the
    same question, so they must not share a directory. RunSpec refuses any new
    field that has neither an abbreviation nor a place in RUNTIME_ONLY, which
    is how this was caught."""
    default = _spec(mach=10.0).case_name()
    forced = _spec(mach=10.0, regime="perfect-gas").case_name()
    assert default != forced
    assert "gas" in forced
