"""Reporting progress while a mesh is built.

A 400k-element domain takes half a minute and the 3D pass is nearly all of
it. A build that says nothing for that long is indistinguishable from one
that has hung, and the usual response to a hang is to kill it -- so silence
does not merely annoy, it loses work.

What is checked here is the reporting contract rather than the appearance of
a bar: which stages are announced, in what order, that a caller passing
nothing pays nothing, and that a build can run somewhere other than the main
thread, which is what a notebook needs in order to keep redrawing.
"""

from __future__ import annotations

import io
import threading

import numpy as np
import pytest

from aether.aerodynamics.cfd.meshing import _STAGES, _reporter, console_progress


def test_the_stages_are_announced_in_order_and_end_at_one() -> None:
    seen: list[tuple[str, float]] = []
    report = _reporter(lambda stage, fraction: seen.append((stage, fraction)))
    for stage in _STAGES:
        report(stage)

    assert [stage for stage, _ in seen] == list(_STAGES)
    fractions = [fraction for _, fraction in seen]
    assert fractions == sorted(fractions)
    assert fractions[-1] == pytest.approx(1.0)
    assert 0.0 < fractions[0] < 1.0


def test_reporting_nothing_costs_nothing() -> None:
    """The default path must not build strings or take a branch per stage."""
    report = _reporter(None)
    for stage in _STAGES:
        report(stage)  # must simply not raise


def test_an_unknown_stage_still_reports_rather_than_raising() -> None:
    """A future stage name should not break a caller's display."""
    seen: list[float] = []
    _reporter(lambda _stage, fraction: seen.append(fraction))("something new")
    assert seen == [pytest.approx(1.0)]


def test_the_console_reporter_rewrites_one_line_on_a_terminal() -> None:
    class Terminal(io.StringIO):
        def isatty(self) -> bool:
            return True

    stream = Terminal()
    # A tick far longer than the test, so the background clock cannot fire and
    # make the redraw count depend on timing. Its behaviour is checked
    # separately, where it is the subject rather than an interference.
    report = console_progress("mesh", stream=stream, tick=3600.0)
    for index, stage in enumerate(_STAGES, start=1):
        report(stage, index / len(_STAGES))

    written = stream.getvalue()
    assert written.count("\r") == len(_STAGES)
    # One trailing newline, at the end, so the next output starts cleanly.
    assert written.count("\n") == 1
    assert written.endswith("\n")
    assert "mesh 3D" in written


def test_the_console_reporter_does_not_write_carriage_returns_to_a_file() -> None:
    """A log with a hundred carriage returns on one line is worse than lines."""
    stream = io.StringIO()  # not a tty
    report = console_progress("mesh", stream=stream, tick=0.01)
    for index, stage in enumerate(_STAGES, start=1):
        report(stage, index / len(_STAGES))

    written = stream.getvalue()
    assert "\r" not in written
    assert written.count("\n") == len(_STAGES)


def test_meshing_runs_off_the_main_thread() -> None:
    """What a notebook needs, and what gmsh refuses unless asked correctly.

    ``gmsh.initialize`` installs a SIGINT handler, and Python only allows that
    from the main thread, so a worker gets ``ValueError: signal only works in
    main thread of the main interpreter`` -- an obscure way to be told
    something that is not even true. Meshing on a worker is fine; only Ctrl-C
    is not available there.
    """
    pytest.importorskip("gmsh")
    from aether.geometry.backend import start_gmsh

    failure: list[BaseException] = []

    def initialise() -> None:
        try:
            gmsh = start_gmsh()
            gmsh.finalize()
        except BaseException as error:
            failure.append(error)

    worker = threading.Thread(target=initialise)
    worker.start()
    worker.join(timeout=60)
    assert not worker.is_alive()
    assert not failure, f"gmsh would not start on a worker thread: {failure[0]!r}"


def test_the_main_thread_keeps_running_while_gmsh_works() -> None:
    """The claim the notebook's live clock rests on.

    gmsh is reached through ctypes, which releases the GIL, so Python on
    another thread continues during a meshing call. Without that the elapsed
    counter would freeze for the whole 3D pass and look exactly like the hang
    it exists to disprove.
    """
    pytest.importorskip("gmsh")
    from aether.geometry.backend import start_gmsh

    ticks = 0
    done = threading.Event()

    def work() -> None:
        gmsh = start_gmsh()
        try:
            gmsh.model.add("box")
            gmsh.model.occ.addBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
            gmsh.model.occ.synchronize()
            gmsh.option.setNumber("Mesh.MeshSizeMax", 0.06)
            gmsh.model.mesh.generate(3)
        finally:
            gmsh.finalize()
            done.set()

    worker = threading.Thread(target=work)
    worker.start()
    while not done.wait(0.02):
        ticks += 1
    worker.join(timeout=120)
    assert ticks > 3, f"main thread only ran {ticks} times; the GIL was not released"


def test_the_stage_list_puts_the_expensive_pass_where_it_belongs() -> None:
    """3D dominates the wall time, so it must not be the last stage shown.

    If nothing followed it, a display would sit at 100 % through the longest
    part of the build -- which is the hang this machinery exists to disprove,
    reintroduced at the last moment. Stated as "something comes after it"
    rather than as an index, so adding a stage does not fail a test whose
    point it does not touch.
    """
    assert _STAGES[-1] == "write"
    assert "mesh 3D" in _STAGES
    assert _STAGES.index("mesh 3D") < len(_STAGES) - 1
    assert list(_STAGES).count("mesh 3D") == 1
    # The passes run in order, cheapest first.
    assert _STAGES.index("mesh 1D") < _STAGES.index("mesh 2D") < _STAGES.index("mesh 3D")


def test_a_real_build_reports_every_stage(tmp_path) -> None:
    """End to end on a small domain, so the wiring is checked and not assumed."""
    pytest.importorskip("gmsh")
    from aether.aerodynamics.cfd.meshing import inviscid_domain
    from aether.aerodynamics.panels import blunted_multiconic
    from aether.geometry.mesh import VehicleMesh

    body = VehicleMesh.from_surface_grid(
        blunted_multiconic(
            nose_radius=0.06, lengths=[2.0], half_angles=[np.radians(10.0)], fillet_radii=[]
        ).surface,
        name="sphere-cone",
    )
    seen: list[str] = []
    inviscid_domain(
        body, tmp_path / "m.su2", mach=3.0, progress=lambda stage, _f: seen.append(stage)
    )
    assert seen == list(_STAGES)


def test_the_console_bar_keeps_redrawing_while_a_stage_runs() -> None:
    """The whole point, and the thing a stage callback alone cannot do.

    On a real domain the 3D pass is twenty-six of twenty-seven seconds, so a
    bar that redraws only when the stage changes stands still through almost
    the entire build. What has to be true is that the line keeps moving while
    nothing is being reported.
    """
    import time

    class Terminal(io.StringIO):
        def isatty(self) -> bool:
            return True

    stream = Terminal()
    report = console_progress("mesh", stream=stream, tick=0.02)
    report("mesh 3D", 0.85)
    before = stream.getvalue().count("\r")
    time.sleep(0.3)  # a long stage, during which nothing is reported
    during = stream.getvalue().count("\r")
    report("write", 1.0)

    assert during > before, "the bar stood still while a stage was running"


def test_the_bar_stops_ticking_once_the_build_is_done() -> None:
    """A daemon thread that outlives the build would scribble over later output."""
    import time

    class Terminal(io.StringIO):
        def isatty(self) -> bool:
            return True

    stream = Terminal()
    report = console_progress("mesh", stream=stream, tick=0.02)
    report("write", 1.0)
    settled = stream.getvalue()
    time.sleep(0.2)
    assert stream.getvalue() == settled
    assert settled.endswith("\n")


# --------------------------------------------------------------- quality


def _quality(values: list[float], threshold: float = 0.1):
    """A :class:`MeshQuality` over a given set of element qualities."""
    from aether.aerodynamics.cfd.meshing import MeshQuality

    array = np.asarray(values, dtype=np.float64)
    return MeshQuality(
        measure="minSICN",
        minimum=float(array.min()),
        first_percentile=float(np.percentile(array, 1.0)),
        median=float(np.median(array)),
        inverted=int((array <= 0.0).sum()),
        poor=int((array < threshold).sum()),
        threshold=threshold,
        elements=int(array.size),
    )


def test_an_inverted_element_is_a_stop_and_a_poor_one_is_a_warning() -> None:
    """The distinction the signed measure exists to make.

    A stretched cell is survivable and an inverted cell is not -- SU2 reports
    the second as a non-physical point before its first iteration. A
    single-sided quality measure cannot tell them apart, which is why the
    default is the *signed* inverse condition number.
    """
    poor = _quality([0.05, 0.4, 0.9, 0.95])
    assert poor.poor == 1
    assert poor.inverted == 0
    assert poor.usable

    inverted = _quality([-0.2, 0.4, 0.9, 0.95])
    assert inverted.inverted == 1
    assert not inverted.usable
    assert "INVERTED" in inverted.summary()


def test_zero_quality_counts_as_inverted() -> None:
    """A flat element has no volume to integrate over; it is not merely poor."""
    assert not _quality([0.0, 0.5, 0.9]).usable


def test_the_summary_reports_the_measure_it_used() -> None:
    """A quality number without its measure is not comparable to anything."""
    text = _quality([0.3, 0.6, 0.9]).summary()
    assert "minSICN" in text
    assert "median" in text


def test_quality_is_measured_on_a_real_mesh(tmp_path) -> None:
    """End to end, and the optimiser must demonstrably improve it.

    Not a fixed threshold -- gmsh's output varies with version -- but the
    ordering, which is the property the default rests on: optimisation exists
    to remove the slivers a Delaunay fill leaves behind, so a mesh built
    without it must be measurably worse.
    """
    pytest.importorskip("gmsh")
    from aether.aerodynamics.cfd.meshing import inviscid_domain
    from aether.aerodynamics.panels import blunted_multiconic
    from aether.geometry.mesh import VehicleMesh

    body = VehicleMesh.from_surface_grid(
        blunted_multiconic(
            nose_radius=0.06, lengths=[2.0], half_angles=[np.radians(10.0)], fillet_radii=[]
        ).surface,
        name="sphere-cone",
    )

    measured = {}
    for label, optimise in (("on", True), ("off", False)):
        result = inviscid_domain(body, tmp_path / f"{label}.su2", mach=2.5, optimize=optimise)
        assert result.quality is not None
        assert result.quality.elements == result.n_elements
        measured[label] = result.quality

    assert measured["on"].poor < measured["off"].poor
    assert measured["on"].minimum > measured["off"].minimum
