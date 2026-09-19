"""Figure sets built from a solved case directory.

Built against a synthetic results tree rather than the real one, so the tests
say what the code does rather than what the current contents of
``results-cfd`` happen to be. The one thing deliberately modelled on the real
tree is its untidiness: a run that stopped after the prelude, a marker written
twice, a case with logs and no field. Those are the states the discovery has
to survive, and they are all present upstairs.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from aether.aerodynamics.cfd.fields import PERFECT_AIR, CaloricallyPerfect, VolumeField
from aether.viz.flowfield import (
    REAL_GAS_MACH,
    CaseConditions,
    FieldHealth,
    discover_cases,
    field_health,
    final_residual,
    pitot_cp_max,
    read_case_config,
)

_CFG = """\
SOLVER= EULER
GAMMA_VALUE= 1.4
MACH_NUMBER= 8
AOA= 2.5
FREESTREAM_TEMPERATURE= 226.5090836
FREESTREAM_PRESSURE= 1197.03164
MARKER_EULER= ( vehicle, vehicle_base )
MARKER_PLOTTING= ( vehicle, vehicle_base )
"""


def _conditions(mach: float, gas: CaloricallyPerfect | None = PERFECT_AIR) -> CaseConditions:
    """A flight condition at any speed, on the freestream the cases here use."""
    return CaseConditions(
        mach=mach,
        alpha=0.0,
        pressure=1197.03164,
        temperature=226.5090836,
        walls=("vehicle",),
        gas=gas,
    )


def _write_restart(path: Path, points: int = 4) -> None:
    names = ["x", "y", "z", "Density", "Momentum_x", "Momentum_y", "Momentum_z", "Energy"]
    values = np.zeros((points, len(names)))
    values[:, 3] = 0.05
    values[:, 7] = 3000.0
    header = struct.pack("<5i", 535532, len(names), points, 0, 0)
    padded = b"".join(name.encode().ljust(33, b"\x00") for name in names)
    path.write_bytes(header + padded + values.tobytes())


def _case(root: Path, name: str, *, config: str = "case.cfg", field: str = "restart_flow.dat"):
    directory = root / name / "cases" / "m8.0000-a+0.00"
    directory.mkdir(parents=True)
    (directory / config).write_text(_CFG)
    (directory / "mesh.su2").write_text("NDIME= 3\n")
    if field:
        _write_restart(directory / field)
    return directory


# ------------------------------------------------------------ configuration


def test_the_flight_condition_is_read_from_the_case(tmp_path: Path) -> None:
    (tmp_path / "case.cfg").write_text(_CFG)
    conditions = read_case_config(tmp_path / "case.cfg")
    assert conditions.mach == 8.0
    assert conditions.alpha == 2.5
    assert conditions.pressure == pytest.approx(1197.03164)
    assert conditions.walls == ("vehicle", "vehicle_base")


def test_a_missing_key_is_refused_rather_than_defaulted(tmp_path: Path) -> None:
    """A defaulted freestream pressure silently rescales every C_p drawn.

    Checked both for a config missing almost everything and for one missing
    only the pressure, so the refusal is not resting on whichever key the
    parser happens to reach first.
    """
    (tmp_path / "bare.cfg").write_text("MACH_NUMBER= 8\nAOA= 0\nMARKER_EULER= ( vehicle )\n")
    with pytest.raises(ValueError, match="cannot label itself"):
        read_case_config(tmp_path / "bare.cfg")

    (tmp_path / "unmarked.cfg").write_text(
        _CFG.replace("MARKER_EULER", "X").replace("MARKER_PLOTTING", "Y")
    )
    with pytest.raises(ValueError, match="names no wall markers"):
        read_case_config(tmp_path / "unmarked.cfg")

    without_pressure = "\n".join(
        line for line in _CFG.splitlines() if not line.startswith("FREESTREAM_PRESSURE")
    )
    (tmp_path / "nearly.cfg").write_text(without_pressure + "\n")
    with pytest.raises(ValueError, match="FREESTREAM_PRESSURE"):
        read_case_config(tmp_path / "nearly.cfg")


def test_dynamic_pressure_and_the_coefficient_it_defines() -> None:
    conditions = _conditions(8.0)
    assert conditions.dynamic_pressure == pytest.approx(0.5 * 1.4 * 1197.03164 * 64.0)
    absolute = np.array([1197.03164, 1197.03164 + conditions.dynamic_pressure])
    assert conditions.pressure_coefficient(absolute) == pytest.approx([0.0, 1.0])


# ---------------------------------------------------------------- discovery


def test_every_complete_case_is_found_and_named_by_family(tmp_path: Path) -> None:
    _case(tmp_path, "anchor/coarse")
    _case(tmp_path, "anchor/medium")
    _case(tmp_path, "sphere-cone/coarse")
    names = [case.name for case in discover_cases(tmp_path)]
    assert names == ["anchor/coarse", "anchor/medium", "sphere-cone/coarse"]


def test_a_run_with_no_field_is_skipped_not_reported(tmp_path: Path) -> None:
    """An unfinished run is a normal state of a results tree, not an error."""
    _case(tmp_path, "done")
    _case(tmp_path, "still-going", field="")
    assert [case.name for case in discover_cases(tmp_path)] == ["done"]


def test_a_prelude_only_run_is_found_and_marked(tmp_path: Path) -> None:
    """Readable, worth drawing, and not the same claim as a finished run."""
    _case(tmp_path, "finished")
    _case(tmp_path, "startup", config="prelude.cfg", field="solution_flow.dat")
    cases = {case.name: case for case in discover_cases(tmp_path)}
    assert cases["finished"].stage == "converged"
    assert cases["startup"].stage == "prelude"
    assert cases["startup"].is_prelude
    assert cases["startup"].solution.name == "solution_flow.dat"


def test_the_main_run_is_preferred_over_the_prelude(tmp_path: Path) -> None:
    directory = _case(tmp_path, "both")
    _write_restart(directory / "solution_flow.dat")
    (case,) = discover_cases(tmp_path)
    assert case.solution.name == "restart_flow.dat"
    assert case.stage == "converged"


# ------------------------------------------------------------------- health


def _health(max_ratio: float, min_density: float = 0.05, finite: bool = True) -> FieldHealth:
    return FieldHealth(
        min_density=min_density, max_pressure_ratio=max_ratio, final_residual=None, finite=finite
    )


def test_an_under_resolved_shock_is_not_called_divergence() -> None:
    """The sphere-cone overshoots the pitot limit ~4x and is still a result.

    Conflating the two would either hide a diverged field or discard a
    usable one, and the whole point of the screen is that it separates them.
    """
    assert not _health(3.7).diverged
    assert _health(3.7).note is None


def test_a_field_orders_of_magnitude_past_the_pitot_limit_is_flagged() -> None:
    health = _health(2.9e5, min_density=2.3e-14)
    assert health.diverged
    assert "DIVERGED" in (health.note or "")


def test_a_non_finite_or_negative_field_is_flagged() -> None:
    assert _health(1.0, finite=False).diverged
    assert _health(1.0, min_density=-1.0).diverged


def test_health_measures_pressure_against_the_pitot_bound(tmp_path: Path) -> None:
    """The screen's number, checked against the ratio it is defined as."""
    from aether.aerodynamics.cfd.fields import VolumeField
    from aether.aerodynamics.closure import rayleigh_pitot_cp_max

    conditions = _conditions(8.0)
    pitot = conditions.pressure + rayleigh_pitot_cp_max(8.0, 1.4) * conditions.dynamic_pressure
    density = np.array([0.05, 0.05])
    pressure = np.array([1197.0, 2.0 * pitot])
    energy = pressure / 0.4
    field = VolumeField(
        points=np.zeros((2, 3)),
        fields={
            "Density": density,
            "Momentum_x": np.zeros(2),
            "Momentum_y": np.zeros(2),
            "Momentum_z": np.zeros(2),
            "Energy": energy,
        },
    )
    health = field_health(field, conditions, tmp_path / "absent.csv")
    assert health.max_pressure_ratio == pytest.approx(2.0)
    assert not health.diverged


# ----------------------------------------------------------------- residual


def test_the_last_residual_is_read_without_the_whole_file(tmp_path: Path) -> None:
    path = tmp_path / "history.csv"
    rows = "\n".join(f"{i},{-i * 0.1:.6f}" for i in range(5000))
    path.write_text(f'"Inner_Iter","rms[Rho]"\n{rows}\n')
    assert final_residual(path) == pytest.approx(-499.9)


def test_a_history_without_the_column_or_the_file_gives_nothing(tmp_path: Path) -> None:
    (tmp_path / "h.csv").write_text('"Inner_Iter","CD"\n1,0.5\n')
    assert final_residual(tmp_path / "h.csv") is None
    assert final_residual(tmp_path / "absent.csv") is None


# ---------------------------------------------------- the rest of the envelope

_NEMO_CFG = """\
SOLVER= NEMO_EULER
MATH_PROBLEM= DIRECT
FLUID_MODEL= SU2_NONEQ
GAS_MODEL= AIR-5
MACH_NUMBER= 25
AOA= 0
FREESTREAM_TEMPERATURE= 226.5090836
FREESTREAM_PRESSURE= 1197.03164
MARKER_EULER= ( vehicle )
MARKER_PLOTTING= ( vehicle )
"""


def test_the_gas_constant_is_read_from_the_case_not_assumed(tmp_path: Path) -> None:
    """It is a case setting in SU2, and for a while nothing here read it."""
    text = _CFG.replace("GAMMA_VALUE= 1.4", "GAMMA_VALUE= 1.29\nGAS_CONSTANT= 188.9")
    (tmp_path / "case.cfg").write_text(text)
    gas = read_case_config(tmp_path / "case.cfg").require_gas()
    assert gas.gamma == pytest.approx(1.29)
    assert gas.gas_constant == pytest.approx(188.9)


def test_a_nonequilibrium_case_gets_no_invented_gamma(tmp_path: Path) -> None:
    """The regime above Mach 10, where 1.4 is not imprecise but wrong.

    ``regime_for`` switches the solver to NEMO there. If the post-processing
    quietly carried on with a perfect gas *through the field*, the figure
    would disagree with the run that produced it.
    """
    (tmp_path / "case.cfg").write_text(_NEMO_CFG)
    conditions = read_case_config(tmp_path / "case.cfg")
    assert conditions.mach == 25.0
    assert not conditions.perfect_gas
    assert conditions.gas is None

    with pytest.raises(ValueError, match="no single gamma describes its shock layer"):
        _ = conditions.gamma
    with pytest.raises(ValueError, match="volume file"):
        conditions.require_gas()


def test_a_nonequilibrium_freestream_is_still_a_perfect_gas(tmp_path: Path) -> None:
    """The distinction that took two goes to get right.

    Air at 264 K ahead of the shock is calorically perfect whatever the solver
    is doing downstream, so dynamic pressure, :math:`C_p` and freestream speed
    are all defined for a five-species run. Refusing them -- which an earlier
    version did, by carrying one gas model instead of two -- meant a Mach 20
    biconic could not be told its own dynamic pressure, and the health screen
    raised on the first case it was built for.
    """
    (tmp_path / "case.cfg").write_text(_NEMO_CFG)
    conditions = read_case_config(tmp_path / "case.cfg")

    expected = 0.5 * 1.4 * conditions.pressure * conditions.mach**2
    assert conditions.dynamic_pressure == pytest.approx(expected)
    assert conditions.pressure_coefficient(np.array([conditions.pressure])) == pytest.approx([0.0])
    assert conditions.speed > 0.0
    # And the pitot ceiling, which is a freestream-referenced number, works too.
    assert pitot_cp_max(conditions)[0] > 1.8


def test_a_nonequilibrium_case_says_so_in_its_caption(tmp_path: Path) -> None:
    (tmp_path / "case.cfg").write_text(_NEMO_CFG)
    assert "SU2_NONEQ" in read_case_config(tmp_path / "case.cfg").label


@pytest.mark.parametrize("mach", [3.0, 8.0, 9.9])
def test_below_the_threshold_the_ceiling_is_the_perfect_gas_one(mach: float) -> None:
    from aether.aerodynamics.closure import rayleigh_pitot_cp_max

    limit, model = pitot_cp_max(_conditions(mach))
    assert limit == pytest.approx(rayleigh_pitot_cp_max(mach, 1.4))
    assert "Rayleigh" in model


@pytest.mark.parametrize("mach", [12.0, 20.0, 27.0])
def test_above_the_threshold_the_ceiling_accounts_for_the_real_gas(mach: float) -> None:
    """Dissociation raises the attainable stagnation pressure a few percent.

    Small against the factor of ten the divergence screen looks for, and not
    small against the few percent the wall panel is drawn to resolve -- which
    is why the two now share one ceiling rather than each having its own.

    Falls back to the perfect-gas value when Cantera is absent, and says so,
    so this asserts the direction and the labelling rather than a number that
    depends on whether an optional dependency is installed.
    """
    from aether.aerodynamics.closure import rayleigh_pitot_cp_max

    conditions = _conditions(mach)
    limit, model = pitot_cp_max(conditions)
    perfect = rayleigh_pitot_cp_max(mach, 1.4)
    if "equilibrium" in model:
        assert limit > perfect
        assert limit == pytest.approx(perfect, rel=0.10)
    else:
        assert limit == pytest.approx(perfect)
        assert "no real-gas solve" in model


def test_the_real_gas_threshold_matches_the_one_the_solver_switches_at() -> None:
    """Two different thresholds for the same physics is a future disagreement."""
    from aether.cfd.config import Nonequilibrium, PerfectGas, regime_for

    assert isinstance(regime_for(REAL_GAS_MACH - 0.1), PerfectGas)
    assert isinstance(regime_for(REAL_GAS_MACH), Nonequilibrium)


@pytest.mark.parametrize("mach", [3.0, 8.0, 15.0, 27.0])
def test_the_divergence_screen_is_not_calibrated_to_one_speed(mach: float) -> None:
    """The screen's ratio is dimensionless, so its threshold must travel.

    A field at the ceiling is fine at any Mach number; one five orders past it
    is diverged at any Mach number. If this ever fails it means the screen has
    acquired a speed-dependent calibration by accident.
    """
    conditions = _conditions(mach)
    ceiling = conditions.pressure + pitot_cp_max(conditions)[0] * conditions.dynamic_pressure
    for ratio, diverged in ((0.9, False), (3.7, False), (2.9e5, True)):
        pressure = np.array([conditions.pressure, ratio * ceiling])
        energy = pressure / (conditions.require_gas().gamma - 1.0)
        field = VolumeField(
            points=np.zeros((2, 3)),
            fields={
                "Density": np.full(2, 0.05),
                "Momentum_x": np.zeros(2),
                "Momentum_y": np.zeros(2),
                "Momentum_z": np.zeros(2),
                "Energy": energy,
            },
        )
        assert field_health(field, conditions, Path("absent.csv")).diverged is diverged
