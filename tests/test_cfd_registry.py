"""The part that has to be right when the process holding a run goes away.

Nothing here starts SU2, and nothing here meshes. What is tested is the claim
the whole remote-control arrangement rests on: that a run's identity survives
in a file, that a second process can pick it back up with its clock intact, and
that a record is never reported as live when it is not.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from aether.cfd import registry
from aether.cfd.registry import (
    NewerSchema,
    RunRecord,
    RunSpec,
    cancel_requested,
    clear_cancel,
    enqueue,
    forget,
    load,
    load_all,
    pid_birth,
    process_alive,
    reconcile,
    request_cancel,
    save,
)
from aether.cfd.run import SolveJob

_HEADER = '"Time_Iter","Outer_Iter","Inner_Iter","rms[Rho]","RefForce","CD","CL"\n'


@pytest.fixture(autouse=True)
def queue_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the registry at a scratch results tree, not the real share."""
    monkeypatch.setenv("AETHER_RESULTS", str(tmp_path))
    return tmp_path / "cfd" / "queue"


def _history(path: Path, rows: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(
        f"0,0,{i},{-0.001 * i:.6f},1.0,{0.42 + 1e-6 * i:.8f},0.001\n" for i in range(rows)
    )
    path.write_text(_HEADER + body)
    return path


def _spec(**overrides: object) -> RunSpec:
    base = {"body": "sphere-cone", "mach": 8.0, "alpha_deg": 0.0,
            "altitude_km": 45.0, "iterations": 100}
    base.update(overrides)
    return RunSpec(**base)  # type: ignore[arg-type]


def _sleeper() -> subprocess.Popen[bytes]:
    """A live process to stand in for a solver, with a pid that is really ours."""
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])


# ------------------------------------------------------------------- records


def test_a_queued_run_is_readable_by_a_process_that_did_not_queue_it() -> None:
    """The whole point: the record, not a Popen, is the handle."""
    record = enqueue(_spec())
    assert record.path.exists()

    # Re-read from disk rather than reusing the object, which is what a
    # dashboard or a restarted worker actually does.
    again = load(record.run_id)
    assert again is not None
    assert again.spec == record.spec
    assert again.state == "queued"
    assert again.total == 100


def test_records_come_back_in_submission_order() -> None:
    first = enqueue(_spec(mach=4.0))
    time.sleep(0.01)
    second = enqueue(_spec(mach=8.0))
    assert [r.run_id for r in load_all()] == [first.run_id, second.run_id]


def test_a_record_from_an_older_schema_still_loads() -> None:
    """A half-day queue must not be lost to a field that has since been added."""
    record = enqueue(_spec())
    payload = json.loads(record.path.read_text())
    del payload["message"]
    record.path.write_text(json.dumps(payload))

    again = load(record.run_id)
    assert again is not None and again.spec.mach == 8.0


def test_a_record_from_a_newer_schema_is_refused_not_quietly_stripped() -> None:
    """The other direction is not symmetric, and must not be treated as if it were.

    Dropping a field this build does not know, then saving the record back
    without it, erased the flux scheme from eight queued runs on 2026-09-11 and
    collided them onto one case directory. Refusing is the correct answer, and
    it has to be loud enough that a point lookup reports it too -- a ``None``
    here is indistinguishable from a run that was never queued.
    """
    record = enqueue(_spec())
    payload = json.loads(record.path.read_text())
    payload["spec"]["a_field_from_the_future"] = 1
    record.path.write_text(json.dumps(payload))

    with pytest.raises(NewerSchema, match="a_field_from_the_future"):
        load(record.run_id)


def test_a_malformed_record_does_not_take_the_listing_down(queue_in_tmp: Path) -> None:
    """A dashboard polls this on a timer; one bad file must not blank the view."""
    good = enqueue(_spec())
    queue_in_tmp.joinpath("broken.json").write_text("{not json")
    assert [r.run_id for r in load_all()] == [good.run_id]


def test_the_case_name_matches_the_scheme_the_notebook_wrote() -> None:
    """So an existing results tree is still addressed by the same names."""
    spec = _spec(mach=8.0, alpha_deg=-2.5, altitude_km=45.0, geometry={"nose_radius": 0.06})
    assert spec.case_name() == "m08.0000-a-2.50-h45.0-nos0.06"


def test_the_geometry_tag_keeps_declared_order_rather_than_sorting() -> None:
    """Sorting it forks the results tree, which is how this was caught.

    A sphere-cone declares nose_radius, length, half_angle in that order, and
    the notebook wrote the tag in that order. Alphabetising produced
    `hal10-len2-nos0.06` beside an existing `nos0.06-len2-hal10` and quietly
    re-solved a case that was already on disk.
    """
    spec = _spec(
        mach=20.0,
        alpha_deg=0.0,
        geometry={"nose_radius": 0.06, "length": 2.0, "half_angle": 10.0},
    )
    assert spec.case_name() == "m20.0000-a+0.00-h45.0-nos0.06-len2-hal10"


def test_the_declared_order_is_the_order_the_body_lists() -> None:
    """The form builds the dict from `parameters`, so the two must agree."""
    from aether.cfd.geometries import REFERENCE_BODIES

    body = REFERENCE_BODIES["sphere-cone"]
    spec = _spec(mach=20.0, geometry=body.defaults())
    expected = "-".join(f"{p.name[:3]}{p.default:g}" for p in body.parameters)
    assert spec.case_name().endswith(expected)


# ------------------------------------------------------------------ liveness


def test_a_live_process_is_live_and_a_dead_one_is_not() -> None:
    process = _sleeper()
    birth = pid_birth(process.pid)
    try:
        assert process_alive(process.pid, birth)
    finally:
        process.terminate()
        process.wait(timeout=10)
    assert not process_alive(process.pid, birth)


def test_a_recycled_pid_is_not_mistaken_for_the_original_run() -> None:
    """The failure ``pid_birth`` exists to prevent, forced rather than waited for.

    A bare ``os.kill(pid, 0)`` would call this alive and a stop request would
    then signal whatever now holds the pid.
    """
    process = _sleeper()
    try:
        actual = pid_birth(process.pid)
        assert actual is not None
        assert not process_alive(process.pid, actual + 10_000)
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_nothing_is_alive_without_a_pid() -> None:
    assert not process_alive(None)
    assert not process_alive(None, 1234)


# --------------------------------------------------------------- reattaching


def test_an_attached_job_recovers_the_elapsed_time_and_the_rate(tmp_path: Path) -> None:
    """The gap this closes: a monotonic start instant means nothing elsewhere.

    Before, a reader that was not running at launch got ``elapsed = nan`` and
    no rate, so the iteration count survived a disconnect but the ETA did not.
    """
    process = _sleeper()
    try:
        job = SolveJob.attach(
            tmp_path,
            pid=process.pid,
            started_at=time.time() - 60.0,
            pid_birth=pid_birth(process.pid),
            total=6000,
        )
        _history(job.history, 120)
        assert job.running

        progress = job.progress()
        assert progress is not None
        assert progress.elapsed == pytest.approx(60.0, abs=2.0)
        assert progress.rate == pytest.approx(2.0, rel=0.1)
        assert progress.remaining == pytest.approx((6000 - 120) / 2.0, rel=0.1)
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_an_attached_job_stops_the_process_it_was_given(tmp_path: Path) -> None:
    process = _sleeper()
    job = SolveJob.attach(
        tmp_path, pid=process.pid, started_at=time.time(), pid_birth=pid_birth(process.pid)
    )
    job.stop(timeout=10.0)
    assert not job.running
    process.wait(timeout=10)


def test_attaching_does_not_delete_the_history_it_came_for(tmp_path: Path) -> None:
    """``start`` unlinks the history; the reattach path must not go near it."""
    history = _history(tmp_path / "history.csv", 50)
    job = SolveJob.attach(tmp_path, pid=os.getpid(), started_at=time.time())
    assert history.exists()
    with pytest.raises(RuntimeError, match="already been started"):
        job.start()
    assert history.exists()


# -------------------------------------------------------------- reconciling


def test_a_run_whose_solver_died_short_of_the_total_is_failed(tmp_path: Path) -> None:
    record = enqueue(_spec(iterations=6000))
    case = tmp_path / "case"
    _history(case / "history.csv", 900)
    record = save(
        replace(
            record,
            state="solving",
            case_dir=str(case),
            pid=999_999_999,
            pid_birth=1,
            total=6000,
            started_at=time.time() - 600.0,
        )
    )
    settled = reconcile(record)
    assert settled.state == "failed"
    assert "900" in settled.message
    # And it is persisted, not just returned.
    assert load(record.run_id).state == "failed"  # type: ignore[union-attr]


def test_a_run_that_reached_its_total_unobserved_is_done(tmp_path: Path) -> None:
    """No exit status survives the parent, so the history is the only evidence."""
    record = enqueue(_spec(iterations=500))
    case = tmp_path / "case"
    _history(case / "history.csv", 500)
    record = save(
        replace(record, state="solving", case_dir=str(case), pid=999_999_999, total=500)
    )
    assert reconcile(record).state == "done"


def test_a_worker_that_died_while_meshing_leaves_a_failure() -> None:
    """Meshing has no detached process, so there is nothing to have survived."""
    record = enqueue(_spec())
    record = save(replace(record, state="meshing"))
    settled = reconcile(record)
    assert settled.state == "failed" and "meshing" in settled.message


def test_a_live_run_is_left_exactly_as_it_was() -> None:
    process = _sleeper()
    try:
        record = enqueue(_spec())
        record = save(
            replace(
                record, state="solving", pid=process.pid, pid_birth=pid_birth(process.pid)
            )
        )
        assert reconcile(record) == record
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_a_finished_record_is_not_reconciled_twice() -> None:
    record = enqueue(_spec())
    done = save(replace(record, state="done"))
    assert reconcile(done) is done


# ----------------------------------------------------------------- cancelling


def test_a_cancel_is_a_separate_file_so_it_cannot_race_the_owner() -> None:
    record = enqueue(_spec())
    assert not cancel_requested(record.run_id)
    request_cancel(record.run_id)
    assert cancel_requested(record.run_id)
    # The record itself is untouched: the requester is not its owner.
    assert load(record.run_id).state == "queued"  # type: ignore[union-attr]
    clear_cancel(record.run_id)
    assert not cancel_requested(record.run_id)


def test_a_live_run_cannot_be_forgotten() -> None:
    """The record is the only handle on the process; deleting it orphans SU2."""
    process = _sleeper()
    try:
        record = enqueue(_spec())
        save(
            replace(
                record, state="solving", pid=process.pid, pid_birth=pid_birth(process.pid)
            )
        )
        with pytest.raises(RuntimeError, match="still running"):
            forget(record.run_id)
        assert load(record.run_id) is not None
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_a_finished_run_can_be_forgotten() -> None:
    record = enqueue(_spec())
    save(replace(record, state="done"))
    request_cancel(record.run_id)
    forget(record.run_id)
    assert load(record.run_id) is None
    assert not cancel_requested(record.run_id)


def test_a_record_is_never_seen_half_written(queue_in_tmp: Path) -> None:
    """Written to a temporary in the same directory, then replaced in one step."""
    record = enqueue(_spec())
    assert list(queue_in_tmp.glob("*.tmp")) == []
    assert json.loads(record.path.read_text())["spec"]["body"] == "sphere-cone"


def test_the_queue_follows_the_results_root(tmp_path: Path) -> None:
    assert registry.queue_dir() == tmp_path / "cfd" / "queue"


# ------------------------------------------------------- across a namespace

def test_a_pid_from_another_namespace_is_not_judged() -> None:
    """The panel-in-one-container, worker-in-another case.

    A pid is only meaningful inside the namespace that issued it. A reader
    elsewhere must say "I cannot tell" rather than run ``os.kill`` against the
    same number in its own process table -- which is either a wrong answer or,
    worse, a signal to an unrelated process.
    """
    process = _sleeper()
    try:
        record = enqueue(_spec())
        foreign = replace(
            record,
            state="solving",
            pid=process.pid,
            pid_birth=pid_birth(process.pid),
            pid_ns="pid:[4026500000]",  # not this process's namespace
        )
        assert not foreign.checkable
        # Genuinely alive, and still not claimed as such from here.
        assert process_alive(process.pid, foreign.pid_birth)
        assert not foreign.alive()

        native = replace(foreign, pid_ns=registry.pid_namespace())
        assert native.checkable and native.alive()
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_a_record_without_a_namespace_is_still_judged() -> None:
    """Records written before the field existed must not all go unreadable."""
    process = _sleeper()
    try:
        record = replace(
            enqueue(_spec()),
            state="solving",
            pid=process.pid,
            pid_birth=pid_birth(process.pid),
            pid_ns=None,
        )
        assert record.checkable and record.alive()
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_a_foreign_run_is_settled_rather_than_left_hanging(tmp_path: Path) -> None:
    """A foreign namespace means the container that held it is gone.

    Stopping a container takes its pid namespace, and every process in it, so
    the solver really is dead and the record must not sit in ``solving``.
    """
    case = tmp_path / "case"
    _history(case / "history.csv", 173)
    record = save(
        replace(
            enqueue(_spec(iterations=6000)),
            state="solving",
            case_dir=str(case),
            pid=1,
            pid_birth=1,
            pid_ns="pid:[4026500000]",
            total=6000,
        )
    )
    settled = reconcile(record)
    assert settled.state == "failed" and "173" in settled.message


def test_the_geometry_order_survives_the_disk_round_trip() -> None:
    """The blind spot that let the sorting bug back in.

    The first version of this test built a `RunSpec` in memory and asked for
    its case name, which passed while the *record* was being written with
    `sort_keys=True` -- so the worker, which reads the record back off disk
    before meshing, got alphabetised keys and solved into a second directory.
    Anything asserting the case name has to go through `enqueue`/`load`.
    """
    spec = _spec(
        mach=20.0,
        geometry={"nose_radius": 0.06, "length": 2.0, "half_angle": 10.0},
    )
    record = enqueue(spec)
    reloaded = load(record.run_id)
    assert reloaded is not None
    assert list(reloaded.spec.geometry) == ["nose_radius", "length", "half_angle"]
    assert reloaded.spec.case_name() == spec.case_name()
    assert reloaded.spec.case_name().endswith("nos0.06-len2-hal10")


def test_a_new_run_is_laminar_and_an_old_record_still_says_sst() -> None:
    """One default was serving two opposite needs, and the run lost.

    Deserialising a record written before ``turbulence`` existed must yield
    "SST", because that is what it ran. Constructing a new run must yield
    laminar, because Re_L is 6.3e5 on this campaign's bodies and a RANS model
    below transition is modelling turbulence that is not there -- and on
    2026-09-11 it NaN'd every body in a baseline suite at around iteration 217.

    The two are told apart by the key being **absent**, not by its value: every
    record written since the field existed carries it, so a null there is a
    deliberate laminar run.
    """
    assert RunSpec(body="sphere-cone", mach=8.0).turbulence is None

    legacy = RunRecord.from_json({"run_id": "old", "spec": {"body": "sphere-cone", "mach": 8.0}})
    assert legacy.spec.turbulence == "SST"

    deliberate = RunRecord.from_json(
        {"run_id": "new", "spec": {"body": "sphere-cone", "mach": 8.0, "turbulence": None}}
    )
    assert deliberate.spec.turbulence is None


def test_a_round_trip_does_not_turn_a_laminar_run_turbulent() -> None:
    """The legacy fill must not fire on a record this build wrote."""
    spec = RunSpec(body="sphere-cone", mach=8.0)
    record = RunRecord(run_id="r", spec=spec)
    assert RunRecord.from_json(record.to_json()).spec.turbulence is None


def test_the_prelude_and_the_limiter_freeze_are_on_by_default() -> None:
    """Three fields made the same mistake; this pins all three at once.

    Each defaulted to the historical behaviour so that a record written before
    it existed would still describe what it ran. That requirement is real and
    belongs in ``from_json``, keyed on the field being absent -- not in a
    dataclass default, which also answers every *new* run and answers it wrongly.

    The note recording the prelude's addition ends with "a default-valued field
    is silently disabled everywhere it is not explicitly passed". It was then
    left disabled for every run the queue ever made.
    """
    fresh = RunSpec(body="sphere-cone", mach=8.0)
    assert fresh.turbulence is None
    assert fresh.first_order_iterations > 0
    assert fresh.limiter_iterations > 0

    legacy = RunRecord.from_json(
        {"run_id": "old", "spec": {"body": "sphere-cone", "mach": 8.0}}
    ).spec
    assert legacy.turbulence == "SST"
    assert legacy.first_order_iterations == 0
    assert legacy.limiter_iterations == 0


def test_an_explicit_zero_is_not_mistaken_for_an_old_record() -> None:
    """Absence is the signal, not the value -- a deliberate 0 must survive."""
    deliberate = RunRecord.from_json(
        {
            "run_id": "new",
            "spec": {
                "body": "sphere-cone",
                "mach": 8.0,
                "first_order_iterations": 0,
                "limiter_iterations": 0,
                "turbulence": None,
            },
        }
    ).spec
    assert deliberate.first_order_iterations == 0
    assert deliberate.limiter_iterations == 0
    assert deliberate.turbulence is None
