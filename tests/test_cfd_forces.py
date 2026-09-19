"""The guards that read what ``history.csv`` cannot say.

Both functions here exist because a run passed every check the campaign report
had and was still wrong. They are tested against the exact shapes SU2 writes,
not against a convenient normalisation, because the whole failure mode is a
reader that only knows one of the shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

from aether.cfd.forces import density_residual, nonphysical_states


def test_density_residual_finds_the_nemo_per_species_column() -> None:
    """NEMO writes ``rms[rho_0]``..``rms[rho_4]``; a perfect gas writes ``rms[rho]``.

    A reader that knows only the second name returns ``None`` on every NEMO run,
    and a residual guard fed ``None`` is a guard that is switched off.
    """
    perfect = {"rms[rho]": np.array([-1.0, -2.0])}
    assert density_residual(perfect) is not None

    nemo = {f"rms[rho_{i}]": np.array([-1.0, -3.6]) for i in range(5)}
    series = density_residual(nemo)
    assert series is not None
    assert series[-1] == -3.6

    assert density_residual({"cd": np.array([0.1])}) is None


def test_nonphysical_states_reports_a_growing_repair_count(tmp_path) -> None:
    """A clipped-point count with a trend is a divergence no residual reports."""
    log = tmp_path / "su2.log"
    counts = list(range(74, 174))
    log.write_text(
        "\n".join(
            f"Warning. The initial solution contains {n} points that are not physical."
            for n in counts
        )
        + "\nWarning: 916527 reconstructed states for upwinding are non-physical.\n"
    )

    state = nonphysical_states(log)
    assert state is not None
    assert state.points_first == 74
    assert state.points_last == 173
    assert state.iterations == 100
    assert state.reconstructions == 916527
    assert state.growing
    assert not state.clean


def test_nonphysical_states_tolerates_a_fixed_handful(tmp_path) -> None:
    """Steady clipping at a geometric singularity must not condemn a run.

    A check that cries wolf on good data is worse than no check.
    """
    log = tmp_path / "su2.log"
    log.write_text(
        "\n".join(
            "Warning. The initial solution contains 12 points that are not physical."
            for _ in range(100)
        )
    )
    state = nonphysical_states(log)
    assert state is not None
    assert not state.growing


def test_nonphysical_states_distinguishes_missing_from_fine(tmp_path) -> None:
    """No log is *not measured*, which is a different statement from *clean*."""
    assert nonphysical_states(tmp_path / "absent.log") is None

    quiet = tmp_path / "su2.log"
    quiet.write_text("Begin Solver\nExit Success\n")
    state = nonphysical_states(quiet)
    assert state is not None and state.clean


def test_newtonian_forebody_is_mach_independent() -> None:
    """The anchor's whole value is that it barely moves with Mach.

    A CFD forebody coefficient that nearly doubles between Mach 8 and Mach 16 is
    contradicting the Mach independence principle, and this is the number that
    makes that visible.
    """
    from aether.cfd.forces import newtonian_forebody
    from aether.cfd.geometries import reference_frame

    geometry = {"nose_radius": 0.06, "length": 2.0, "half_angle": 10.0}
    area = reference_frame("sphere-cone", **geometry).area
    values = [newtonian_forebody("sphere-cone", geometry, m, area) for m in (8, 12, 16, 20)]

    assert all(0.07 < v < 0.08 for v in values)
    assert max(values) - min(values) < 0.001
    # Monotone toward the hypersonic limit, never past it.
    assert values == sorted(values)


def test_newtonian_forebody_uses_the_rayleigh_pitot_coefficient() -> None:
    """Not the incompressible 2.0 -- that is wrong by 9 percent at Mach 8."""
    from aether.cfd.forces import newtonian_forebody
    from aether.cfd.geometries import reference_frame

    geometry = {"nose_radius": 0.06, "length": 2.0, "half_angle": 10.0}
    area = reference_frame("sphere-cone", **geometry).area
    # C_A is linear in C_p,max, so the ratio between two Mach numbers is the
    # ratio of their stagnation pressure coefficients: 1.8274 and 1.8374.
    low = newtonian_forebody("sphere-cone", geometry, 8.0, area)
    high = newtonian_forebody("sphere-cone", geometry, 20.0, area)
    assert high / low == pytest.approx(1.8374 / 1.8274, rel=1e-3)


def _screen(path, residuals, header="|  Inner_Iter|    rms[Rho]|          CL|          CD|"):
    path.write_text(
        header
        + "\n"
        + "\n".join(
            f"|{i:12d}|{r:12.6f}|{0.05:12.6f}|{0.09:12.6f}|" for i, r in enumerate(residuals)
        )
    )
    return path


def test_screen_residual_reads_the_column_by_name(tmp_path) -> None:
    """The columns are not fixed, so a positional read is a wrong read.

    NEMO prints no residual column at all in the screen table; returning None
    there is the honest answer and guessing at a position is not.
    """
    from aether.cfd.forces import screen_residual

    perfect = _screen(tmp_path / "a.log", [-1.0, -2.0, -3.0])
    assert screen_residual(perfect) == [-1.0, -2.0, -3.0]

    nemo = _screen(tmp_path / "b.log", [-1.0], header="|  Inner_Iter|          CL|          CD|")
    assert screen_residual(nemo) is None


def test_residual_excursion_passes_a_healthy_transient(tmp_path) -> None:
    """A good run digs below the plateau it settles onto, and that is not a fault.

    Measured: the settled Mach 8 case spends 2.72 decades this way. A threshold
    that flags it is a check that cries wolf on good data.
    """
    from aether.cfd.forces import diverged_outright, residual_excursion, screen_residual

    healthy = _screen(tmp_path / "ok.log", [-1.2, -3.7, -2.0, -0.98])
    trace = screen_residual(healthy)
    assert residual_excursion(trace) == pytest.approx(2.72, abs=0.01)
    assert not diverged_outright(trace)


def test_residual_excursion_catches_a_run_that_walked_away(tmp_path) -> None:
    from aether.cfd.forces import diverged_outright, residual_excursion, screen_residual

    diverging = _screen(tmp_path / "bad.log", [-2.4, -3.37, -1.5, 1.5, 6.86])
    trace = screen_residual(diverging)
    assert residual_excursion(trace) > 10.0
    assert diverged_outright(trace)


def test_diverged_outright_catches_a_run_that_started_broken(tmp_path) -> None:
    """Restarted from a diverged prelude, so it never had a better solution.

    Excursion is 0.00 here -- there was nothing to walk away from -- which is
    exactly why the absolute test has to exist alongside it.
    """
    from aether.cfd.forces import diverged_outright, residual_excursion, screen_residual

    born_broken = _screen(tmp_path / "worse.log", [7.424188] * 64)
    trace = screen_residual(born_broken)
    assert residual_excursion(trace) == pytest.approx(0.0)
    assert diverged_outright(trace)


def test_a_record_from_a_newer_schema_is_refused_not_silently_trimmed() -> None:
    """Dropping unknown spec fields on load is how eight runs became one.

    A long-lived worker holding a RunSpec from before ``convective`` existed
    read the queue, dropped that field, and saved each record back without it.
    The flux scheme was erased from disk, every case name lost its suffix, all
    eight collided on one directory, and the first overwrote a converged result
    while running a scheme nobody had asked for.
    """
    from aether.cfd.registry import RunRecord

    spec = {"body": "sphere-cone", "mach": 8.0, "altitude_km": 45.0, "geometry": {}}

    # Older record: missing fields default, and the queue stays usable.
    older = RunRecord.from_json({"run_id": "old", "state": "queued", "spec": spec})
    assert older.spec.convective == "AUSMPLUSUP2"

    # Newer record: refuse, loudly, rather than rewrite someone else's queue.
    with pytest.raises(ValueError, match="newer version"):
        RunRecord.from_json(
            {"run_id": "new", "state": "queued", "spec": {**spec, "unheard_of": 1}}
        )


def test_a_forebody_only_mesh_must_be_declared_not_inferred(tmp_path) -> None:
    """A single-marker history looks the same whether the base is absent or folded in.

    A shock-fitted mesh never extrudes the base disc, so SU2 monitors one marker
    and its total *is* the forebody. An isotropic single-marker run's total
    includes a base that came out as **thrust** on this pipeline -- 2.4x the
    forebody drag at Mach 6. The columns cannot tell them apart, so the caller
    says which, and the default stays the safe refusal.
    """
    from aether.cfd.forces import resolved_forces

    (tmp_path / "history.csv").write_text(
        "Inner_Iter,CD,CL\n" + "\n".join(f"{i},0.1,0.0" for i in range(50))
    )
    geometry = {"nose_radius": 0.06, "length": 2.0, "half_angle": 10.0}

    assert resolved_forces(tmp_path, "sphere-cone", geometry, 8.0, 0.5326) is None

    declared = resolved_forces(
        tmp_path, "sphere-cone", geometry, 8.0, 0.5326, forebody_only=True
    )
    assert declared is not None
    assert declared.forebody_axial == pytest.approx(0.1)
    # The modelled base is still added: it is absent from the *mesh*, not from
    # the vehicle.
    assert declared.base_axial > 0.0
    assert declared.axial == pytest.approx(0.1 + declared.base_axial)


def test_a_drag_split_reports_its_own_sanity_check(tmp_path) -> None:
    """The split is only as good as the normals, so it returns them to be checked.

    ``frontal_area`` is the projected forward-facing area taken from the very
    same normals that weight the pressure integral. It must equal the reference
    area; on the Mach 8 sphere-cone it comes out 0.53148 against 0.5326, a
    faceting difference of 0.21 %. If it did not, neither component would mean
    anything, and the failure would be silent -- a sign error on the nose cap
    alone is worth half the drag on that body.
    """
    from aether.cfd.forces import DragSplit, drag_split

    split = DragSplit(
        pressure=0.08516, friction=0.05167,
        wetted_area=3.02089, frontal_area=0.53148, iteration=1450,
    )
    assert split.total == pytest.approx(0.13683)
    assert split.friction_share == pytest.approx(0.3776, abs=1e-3)
    assert "38% friction" in split.summary()

    # Nothing to read: no mesh, no snapshot. Never raises into a report.
    assert drag_split(tmp_path, 1.0, 100.0, 8.0) is None
    (tmp_path / "body.su2").write_text("NDIME= 3\n")
    assert drag_split(tmp_path, 1.0, 100.0, 8.0) is None


# ------------------------------------------------------- resolved base pressure


def _shedding_history(tmp_path, cycles=30, steps=30, mean=0.018, transient=True):
    """A synthetic base-force signal: shedding, noise, and a starting transient."""
    import numpy as np

    n = cycles * steps
    t = np.arange(n)
    rng = np.random.default_rng(0)
    signal = mean + 0.004 * np.sin(2 * np.pi * t / steps) + 5e-4 * rng.standard_normal(n)
    if transient:
        lead = min(5 * steps, n)
        signal[:lead] += 0.010 * np.exp(-t[:lead] / (1.3 * steps))
    path = tmp_path / "history.csv"
    rows = ["Time_Iter,Inner_Iter,CD,CD[1]"] + [
        f"{k},0,0.14,{v:.8f}" for k, v in enumerate(signal)
    ]
    path.write_text("\n".join(rows) + "\n")
    return path, mean


def test_a_resolved_base_pressure_averages_the_shedding_not_a_sample_of_it(tmp_path):
    from aether.cfd.coefficients import base_pressure

    path, mean = _shedding_history(tmp_path)
    result = base_pressure(path, base_area=0.53108, reference_area=0.5326, mach=8.0)
    assert result is not None
    assert result.axial == pytest.approx(mean, rel=0.02), (
        "the starting transient must be discarded, not averaged in"
    )
    assert result.resolved, f"only {result.periods:.1f} cycles were counted"
    # The bar divides by independent cycles, not by correlated steps: there are
    # thirty steps per cycle by construction, so the two differ by ~5x.
    assert result.sigma == pytest.approx(result.scatter / np.sqrt(result.periods), rel=0.05)
    assert result.sigma < result.scatter


def test_a_run_stopped_before_the_wake_settles_says_so(tmp_path):
    """Three cycles is a sample of the shedding, not a mean of it -- and the
    answer it gives is biased by the transient it never escaped."""
    from aether.cfd.coefficients import base_pressure

    path, mean = _shedding_history(tmp_path, cycles=3)
    result = base_pressure(path, base_area=0.53108, reference_area=0.5326, mach=8.0)
    assert result is not None
    assert not result.resolved
    assert "TOO FEW CYCLES" in result.summary()
    assert abs(result.axial - mean) > 0.1 * mean, (
        "this fixture exists because the short answer is wrong, not merely uncertain"
    )


def test_the_correlation_is_reported_beside_the_resolved_value(tmp_path):
    """So the two can be compared, which is the whole reason for resolving it."""
    from aether.cfd.coefficients import base_pressure

    path, _ = _shedding_history(tmp_path)
    result = base_pressure(path, base_area=0.53108, reference_area=0.5326, mach=8.0)
    assert result.correlation == f"{-1.0 / 64.0:+.5f}"


def test_a_history_with_no_base_marker_yields_nothing(tmp_path):
    """A forebody-only run has no base to reduce, and inventing one from the
    total force is the double-count this module exists to refuse."""
    from aether.cfd.coefficients import base_pressure

    path = tmp_path / "history.csv"
    path.write_text("Time_Iter,Inner_Iter,CD\n" + "\n".join(f"{k},0,0.14" for k in range(200)))
    assert base_pressure(path, base_area=0.5, reference_area=0.5, mach=8.0) is None


def test_a_resolved_base_is_read_not_dropped(tmp_path):
    """The correlation must stop being added *and* the computed base must start.

    The first version of this fix did only the first half: it set the base to
    zero when the wake was resolved, which removes a 32 %-of-total term and
    leaves a forebody-only number that looks entirely plausible. A test that
    asserts only "the correlation went away" passes on that.
    """

    from aether.cfd.forces import resolved_forces
    from aether.cfd.geometries import reference_frame

    geometry = {"nose_radius": 0.06, "length": 2.0, "half_angle": 10.0}
    reference = reference_frame("sphere-cone", **geometry).area
    rows = ["Time_Iter,Inner_Iter,CD,CD(vehicle),CD(vehicle_base)"] + [
        f"0,{k},0.16,0.10362,0.06101" for k in range(900)
    ]
    (tmp_path / "history.csv").write_text("\n".join(rows) + "\n")

    computed = resolved_forces(
        tmp_path, "sphere-cone", geometry, 4.0, reference,
        forebody_only=True, wake_resolved=True,
    )
    assert computed.base_axial == pytest.approx(0.06101, rel=1e-6), (
        "the base force was monitored; it must be read, not zeroed"
    )
    assert computed.axial == pytest.approx(0.10362 + 0.06101, rel=1e-6)

    correlated = resolved_forces(
        tmp_path, "sphere-cone", geometry, 4.0, reference,
        forebody_only=True, wake_resolved=False,
    )
    assert correlated.base_axial > 0.0
    # The two differ by the modelling error, not by the whole base term.
    assert abs(computed.axial - correlated.axial) < 0.25 * computed.base_axial


def test_asking_for_a_resolved_base_that_was_never_solved_is_refused(tmp_path):
    """Returning a forebody-only total would under-report drag by a third and
    look perfectly reasonable doing it."""
    from aether.cfd.forces import resolved_forces

    rows = ["Time_Iter,Inner_Iter,CD,CD(vehicle)"] + [
        f"0,{k},0.14,0.10362" for k in range(900)
    ]
    (tmp_path / "history.csv").write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="no base marker"):
        resolved_forces(
            tmp_path, "sphere-cone",
            {"nose_radius": 0.06, "length": 2.0, "half_angle": 10.0},
            4.0, 0.5326, forebody_only=True, wake_resolved=True,
        )
