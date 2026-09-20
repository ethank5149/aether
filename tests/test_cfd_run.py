"""Watching a long solve without blocking on it.

Nothing here starts SU2. What is tested is the part that has to be right when
it does run: reading progress out of a history file, estimating a rate from
successive readings, and refusing to report a stale run as a live one.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from aether.cfd.run import SolveJob, SolveProgress, read_progress

_HEADER = '"Time_Iter","Outer_Iter","Inner_Iter","rms[Rho]","RefForce","CD","CL"\n'


def _history(path: Path, rows: int, start: int = 0) -> Path:
    body = "".join(
        f"0,0,{i},{-0.001 * i:.6f},1.0,{0.42 + 1e-6 * i:.8f},0.001\n"
        for i in range(start, start + rows)
    )
    path.write_text(_HEADER + body)
    return path


def test_a_history_that_does_not_exist_yet_is_not_an_error(tmp_path: Path) -> None:
    """The normal state of a run for its first few seconds."""
    assert read_progress(tmp_path / "history.csv") is None
    (tmp_path / "empty.csv").write_text(_HEADER)
    assert read_progress(tmp_path / "empty.csv") is None


def test_progress_counts_the_rows_it_actually_has(tmp_path: Path) -> None:
    progress = read_progress(_history(tmp_path / "h.csv", 250), total=6000)
    assert progress is not None
    assert progress.iteration == 250
    assert progress.total == 6000
    assert progress.fraction == pytest.approx(250 / 6000)
    assert progress.residual == pytest.approx(-0.249)
    assert progress.drag == pytest.approx(0.42 + 1e-6 * 249)


def test_the_row_count_wins_over_a_counter_that_restarted(tmp_path: Path) -> None:
    """SU2 resets ``Inner_Iter`` when a case restarts from a solution.

    Trusting the column would report a run 4000 iterations in as being on its
    fiftieth, and the estimate built on that is wrong by two orders of
    magnitude in the direction that matters.
    """
    progress = read_progress(_history(tmp_path / "h.csv", 4000, start=0), total=6000)
    assert progress is not None
    assert progress.iteration == 4000


def test_a_rate_needs_more_than_one_reading(tmp_path: Path) -> None:
    """A single row gives an average and is labelled as one, not a window."""
    path = _history(tmp_path / "h.csv", 120)
    without = read_progress(path, total=6000)
    assert without is not None and without.rate is None
    started = time.monotonic() - 60.0
    average = read_progress(path, total=6000, started=started)
    assert average is not None and average.rate == pytest.approx(2.0, rel=0.05)


def test_an_estimate_needs_a_total_and_a_rate() -> None:
    """Reporting a remaining time without both would be inventing one."""
    assert SolveProgress(10, None, None, None, 1.0, 2.0).remaining is None
    assert SolveProgress(10, 100, None, None, 1.0, None).remaining is None
    assert SolveProgress(10, 100, None, None, 1.0, 0.0).remaining is None
    assert SolveProgress(10, 100, None, None, 1.0, 2.0).remaining == pytest.approx(45.0)


def test_the_estimate_stops_at_zero_and_the_fraction_at_one() -> None:
    finished = SolveProgress(6000, 6000, -5.0, 0.4, 100.0, 60.0)
    assert finished.fraction == 1.0
    assert finished.remaining == 0.0
    overrun = SolveProgress(6100, 6000, -5.0, 0.4, 100.0, 60.0)
    assert overrun.fraction == 1.0
    assert overrun.remaining == 0.0


def test_the_description_says_only_what_it_knows() -> None:
    bare = SolveProgress(7, None, None, None, 1.0, None).describe()
    assert bare == "iter 7"
    full = SolveProgress(
        120, 6000, -0.25, 0.42, 60.0, 2.0, residual_of="rms[Rho]"
    ).describe()
    for fragment in ("iter 120/6000", "rms[Rho] -0.250", "CD +0.42000", "120 it/min", "ETA"):
        assert fragment in full


def test_an_unnamed_residual_is_reported_without_a_gap() -> None:
    """A run whose history has no recognisable residual column still reports
    the number it read. Labelling it with an empty name would leave a double
    space, which reads as a field that failed to render rather than one that
    was never there."""
    line = SolveProgress(120, 6000, -0.25, None, 60.0, 2.0).describe()
    assert "-0.250 (+0.00 dex)" in line
    # The join separator is two spaces; a stray leading space on the part
    # (from formatting an empty name) would produce three consecutive spaces.
    assert "   " not in line


def test_a_job_measures_its_rate_from_successive_readings(tmp_path: Path) -> None:
    """The windowed estimate, which is the one an ETA should be built on."""
    job = SolveJob(directory=tmp_path, total=6000)
    job._started = time.monotonic() - 10.0

    _history(job.history, 100)
    first = job.progress()
    assert first is not None

    time.sleep(0.05)
    _history(job.history, 200)
    second = job.progress()
    assert second is not None
    assert second.iteration == 200
    # 100 further iterations across the gap between the two readings.
    assert second.rate is not None and second.rate > first.rate  # type: ignore[operator]


def test_a_job_that_was_never_started_reports_nothing(tmp_path: Path) -> None:
    job = SolveJob(directory=tmp_path, total=100)
    assert job.progress() is None
    assert not job.running
    assert job.returncode is None
    job.stop()  # must be safe on a job that never ran


def test_starting_twice_is_refused(tmp_path: Path) -> None:
    """Two solvers in one directory overwrite each other's restart."""
    job = SolveJob(directory=tmp_path, total=10)
    job._process = object()  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="already been started"):
        job.start()


def test_a_missing_solver_says_where_it_looked(monkeypatch, tmp_path: Path) -> None:
    from aether.cfd import run as module

    monkeypatch.setattr(
        module.shutil if hasattr(module, "shutil") else __import__("shutil"),
        "which",
        lambda _name: None,
    )
    monkeypatch.setattr(module.Path, "exists", lambda _self: False)
    with pytest.raises(FileNotFoundError, match="not pip-installable"):
        module._find_su2()
