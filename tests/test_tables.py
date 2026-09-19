"""Resumable coefficient sweeps."""

import json

import numpy as np
import pytest

from aether.aerodynamics.tables import (
    AeroTable,
    Coefficients,
    SweepGrid,
    SweepRun,
    console_progress,
)


class _Counting:
    """A solver that records how often it was asked, so resume is testable."""

    name = "counting"

    def __init__(self):
        self.calls = []

    def solve(self, mach, alpha):
        self.calls.append((mach, alpha))
        return Coefficients(axial=mach, normal=alpha, pitching_moment=mach * alpha)


class _Exploding:
    name = "exploding"

    def __init__(self, after):
        self.after = after
        self.calls = 0

    def solve(self, mach, alpha):
        self.calls += 1
        if self.calls > self.after:
            raise KeyboardInterrupt
        return Coefficients(mach, alpha, 0.0)


def _grid():
    return SweepGrid(mach=np.array([2.0, 3.0, 5.0]), alpha=np.deg2rad([0.0, 4.0]))


class TestSweepGrid:
    def test_it_enumerates_every_pair_mach_major(self):
        points = list(_grid().points())
        assert len(points) == 6 == _grid().size
        assert points[0][0] == 2.0 and points[2][0] == 3.0

    def test_the_default_mach_grid_is_dense_where_the_physics_is(self):
        """A uniform grid spends most of its points above Mach 10, where the
        curve is flat, and skips the transonic bucket where it is not."""
        grid = SweepGrid.default_mach()
        transonic = np.sum((grid >= 0.6) & (grid <= 1.6))
        hypersonic = np.sum(grid >= 10.0)
        assert transonic > 2 * hypersonic

    def test_a_minimum_trims_the_grid_to_a_solver_range(self):
        assert SweepGrid.default_mach(minimum=1.2).min() >= 1.2
        assert SweepGrid.default_mach().min() < 1.0

    def test_rejects_empty_or_non_finite_or_subsonic_nonsense(self):
        with pytest.raises(ValueError, match="non-empty"):
            SweepGrid(mach=np.array([]), alpha=np.array([0.0]))
        with pytest.raises(ValueError, match="non-finite"):
            SweepGrid(mach=np.array([np.nan]), alpha=np.array([0.0]))
        with pytest.raises(ValueError, match="must be positive"):
            SweepGrid(mach=np.array([-1.0]), alpha=np.array([0.0]))


class TestCheckpointing:
    def test_a_completed_run_fills_the_table(self, tmp_path):
        run = SweepRun("v", _grid(), _Counting(), tmp_path / "c.jsonl", 1.0, 1.0)
        table = run.run()
        assert table.complete
        assert table.filled == 6

    def test_resuming_skips_what_is_already_done(self, tmp_path):
        """The whole point. A week-long sweep that can only be run in one
        sitting cannot be run."""
        store = tmp_path / "c.jsonl"
        first = _Counting()
        SweepRun("v", _grid(), first, store, 1.0, 1.0).run(max_points=4)
        assert len(first.calls) == 4

        second = _Counting()
        table = SweepRun("v", _grid(), second, store, 1.0, 1.0).run()
        assert len(second.calls) == 2, "only the outstanding points may be re-evaluated"
        assert table.complete

    def test_a_time_budget_stops_the_run(self, tmp_path):
        import time as clock

        class Slow:
            name = "slow"

            def solve(self, mach, alpha):
                clock.sleep(0.02)
                return Coefficients(mach, alpha, 0.0)

        grid = SweepGrid(mach=np.arange(2.0, 12.0), alpha=np.array([0.0]))
        run = SweepRun("v", grid, Slow(), tmp_path / "c.jsonl", 1.0, 1.0)
        table = run.run(time_budget=0.05)
        assert not table.complete
        assert 0 < table.filled < grid.size

    def test_an_interrupt_keeps_the_work_already_done(self, tmp_path):
        """Stopping a sweep is a normal thing to do, not an error."""
        store = tmp_path / "c.jsonl"
        run = SweepRun("v", _grid(), _Exploding(after=3), store, 1.0, 1.0)
        table = run.run()
        assert table.filled == 3
        assert not table.complete
        assert len(store.read_text().strip().splitlines()) == 3

    def test_each_point_is_flushed_as_it_completes(self, tmp_path):
        """Results sitting in a buffer are not checkpointed, they only look
        it — a kill -9 would lose them."""
        store = tmp_path / "c.jsonl"

        class Watching:
            name = "watching"

            def solve(self, mach, alpha):
                # By the time the second point is solved, the first must
                # already be on disk.
                if mach > 2.0:
                    assert store.exists() and store.read_text().strip()
                return Coefficients(mach, alpha, 0.0)

        SweepRun("v", _grid(), Watching(), store, 1.0, 1.0).run()

    def test_a_truncated_last_line_does_not_destroy_the_table(self, tmp_path):
        """A run killed mid-write leaves a partial line. Losing the whole
        checkpoint to that would defeat the point of having one."""
        store = tmp_path / "c.jsonl"
        SweepRun("v", _grid(), _Counting(), store, 1.0, 1.0).run()
        with store.open("a") as handle:
            handle.write('{"name": "v", "mach": 9.0, "alp')
        run = SweepRun("v", _grid(), _Counting(), store, 1.0, 1.0)
        assert len(run.completed()) == 6
        assert run.table().complete

    def test_a_checkpoint_from_another_configuration_is_ignored(self, tmp_path):
        """One file must not silently mix two vehicles."""
        store = tmp_path / "c.jsonl"
        SweepRun("alpha", _grid(), _Counting(), store, 1.0, 1.0).run()
        other = _Counting()
        SweepRun("beta", _grid(), other, store, 1.0, 1.0).run()
        assert len(other.calls) == 6, "beta must not inherit alpha's results"

    def test_status_reports_without_evaluating(self, tmp_path):
        store = tmp_path / "c.jsonl"
        SweepRun("v", _grid(), _Counting(), store, 1.0, 1.0).run(max_points=2)
        solver = _Counting()
        status = SweepRun("v", _grid(), solver, store, 1.0, 1.0).status()
        assert status["completed"] == 2
        assert status["remaining"] == 4
        assert solver.calls == []

    def test_the_record_carries_its_reference_quantities(self, tmp_path):
        """A table read back must not be combinable with one built on
        different references."""
        store = tmp_path / "c.jsonl"
        SweepRun("v", _grid(), _Counting(), store, 7.07, 3.0).run(max_points=1)
        record = json.loads(store.read_text().splitlines()[0])
        assert record["reference_area"] == pytest.approx(7.07)
        assert record["reference_length"] == pytest.approx(3.0)
        assert record["solver"] == "counting"


class TestProgress:
    def test_the_bar_never_exceeds_the_total(self, tmp_path):
        """It did: `done` is mutated in place as points complete, so reading
        its length inside the loop counted every new point twice and the bar
        sailed to 200 %."""
        seen = []
        run = SweepRun("v", _grid(), _Counting(), tmp_path / "c.jsonl", 1.0, 1.0)
        run.run(progress=seen.append, report_every=1)
        assert seen
        for state in seen:
            assert state["completed"] <= state["total"]
        assert seen[-1]["completed"] == 6

    def test_progress_counts_resume_correctly(self, tmp_path):
        store = tmp_path / "c.jsonl"
        SweepRun("v", _grid(), _Counting(), store, 1.0, 1.0).run(max_points=4)
        seen = []
        SweepRun("v", _grid(), _Counting(), store, 1.0, 1.0).run(
            progress=seen.append, report_every=1
        )
        assert seen[-1]["completed"] == 6
        assert seen[0]["completed"] == 5, "the resumed run starts from four done"

    def test_the_console_reporter_runs(self, tmp_path, capsys):
        run = SweepRun("v", _grid(), _Counting(), tmp_path / "c.jsonl", 1.0, 1.0)
        run.run(progress=console_progress(), report_every=1)
        assert "100.0%" in capsys.readouterr().out


class TestAeroTable:
    @staticmethod
    def _table():
        mach = np.array([2.0, 4.0])
        alpha = np.deg2rad([0.0, 10.0])
        axial = np.array([[1.0, 2.0], [3.0, 4.0]])
        return AeroTable(
            "v", mach, alpha, axial, axial * 10.0, axial * 100.0, 7.07, 3.0, "test"
        )

    def test_it_interpolates_bilinearly_on_the_grid(self):
        table = self._table()
        assert table.at(2.0, 0.0).axial == pytest.approx(1.0)
        assert table.at(4.0, np.deg2rad(10.0)).axial == pytest.approx(4.0)
        assert table.at(3.0, np.deg2rad(5.0)).axial == pytest.approx(2.5)

    def test_it_clamps_rather_than_extrapolating(self):
        """Asked for Mach 30 on a table built to 4, the honest answer is the
        Mach 4 value, not a linear guess off the end of an asymptotic curve."""
        table = self._table()
        assert table.at(99.0, 0.0).axial == pytest.approx(3.0)
        assert table.at(0.1, 0.0).axial == pytest.approx(1.0)

    def test_drag_area_is_the_zero_incidence_axial_times_the_reference(self):
        table = self._table()
        assert table.drag_area(2.0) == pytest.approx(1.0 * 7.07)

    def test_a_partial_table_cannot_pass_as_finished(self, tmp_path):
        run = SweepRun("v", _grid(), _Counting(), tmp_path / "c.jsonl", 1.0, 1.0)
        partial = run.run(max_points=2)
        assert not partial.complete
        assert np.isnan(partial.axial).any()
