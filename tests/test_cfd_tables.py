"""The path from a solved queue to a coefficient table.

Item A of the simplifications inventory: ``grep -rn "from aether.cfd"
trajectory/ flight/ guidance/`` returned **nothing**, so every aerodynamic
coefficient in the flight model was a literal written by hand. A model whose
aerodynamic input is a constant cannot be wrong in a way that shows up, which is
why it survived.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

_HEADER = [
    "Time_Iter", "Outer_Iter", "Inner_Iter", "rms[Rho]",
    "CD", "CL", "CMy", "CFx", "CFz",
]


def _history(path: Path, rows: int, *, axial: float, normal: float, moment: float,
            ripple: float = 0.0) -> Path:
    """A history whose force oscillates about a known mean."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_HEADER)
        for i in range(rows):
            wobble = ripple * np.sin(2.0 * np.pi * i / 97.0)
            writer.writerow([
                0, 0, i, -6.0,
                axial + wobble, normal, moment,
                axial + wobble, normal,
            ])
    return path


def test_the_reader_reports_the_window_mean_not_the_last_iterate(tmp_path) -> None:
    """The solution ripples about its answer.

    A final value carries the full amplitude of that ripple; the window mean
    carries it divided by the square root of the independent-sample count, and
    is the number the convergence test was applied to in the first place.
    """
    from aether.cfd.coefficients import settled_coefficients

    path = _history(tmp_path / "history.csv", 3000, axial=0.125, normal=0.0,
                    moment=0.0, ripple=0.01)
    forces = settled_coefficients(path)
    assert forces is not None
    # The mean survives the ripple; the last iterate would not.
    assert forces.axial == pytest.approx(0.125, abs=2e-4)
    assert forces.iterations == 3000
    assert forces.periods > 2.0


def test_a_run_with_too_few_periods_is_not_usable(tmp_path) -> None:
    """Low scatter over a short window says the ripple died, not the mean.

    Measured on the biconic: 1.5 periods resolved in the window, and the report
    flags it `thin:2per` rather than quoting a standard error that pretends the
    window contained a population.
    """
    from aether.cfd.coefficients import settled_coefficients

    path = _history(tmp_path / "history.csv", 120, axial=0.125, normal=0.0,
                    moment=0.0, ripple=0.01)
    forces = settled_coefficients(path)
    assert forces is not None
    assert forces.periods < 2.0
    assert not forces.usable


def test_a_history_without_body_axis_columns_reports_nothing(tmp_path) -> None:
    """Rather than rotating CL/CD and getting the rotation backwards."""
    from aether.cfd.coefficients import settled_coefficients

    path = tmp_path / "history.csv"
    path.write_text("Inner_Iter,CD,CL\n0,0.1,0.0\n1,0.1,0.0\n")
    assert settled_coefficients(path) is None
    assert settled_coefficients(tmp_path / "absent.csv") is None


def test_a_table_cannot_be_invented_from_an_empty_queue(tmp_path, monkeypatch) -> None:
    """The hole is the point. A table that interpolated across it would hide
    exactly the gap it exists to expose."""
    from aether.cfd.tables import table_from_registry

    monkeypatch.setenv("AETHER_RESULTS", str(tmp_path))
    with pytest.raises(ValueError, match="cannot be invented"):
        table_from_registry(
            "sphere-cone",
            {"nose_radius": 0.06, "length": 2.0, "half_angle": 10.0},
            altitude_km=45.0,
        )
