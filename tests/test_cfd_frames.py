"""Rendering a field while it is still being solved.

The render itself is exercised against a real case where one exists -- there is
no way to fake an SU2 volume file that is worth the effort of faking. What is
always tested is the bookkeeping around it: which frames exist, which is the
newest, and the rule that decides when to spend two seconds drawing another.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path

import pytest

from aether.cfd.frames import (
    Frame,
    FrameWriter,
    frame_dir,
    frames,
    latest_frame,
    snapshot_path,
)


def _frame(case: Path, iteration: int, *, note: str = "", diverged: bool = False) -> Path:
    directory = frame_dir(case)
    directory.mkdir(parents=True, exist_ok=True)
    image = directory / f"iter_{iteration:06d}.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    image.with_suffix(".json").write_text(
        json.dumps({"iteration": iteration, "note": note, "diverged": diverged})
    )
    return image


def test_frames_come_back_in_iteration_order(tmp_path: Path) -> None:
    """Zero-padded names, so a plain sort is numeric order rather than 1, 10, 2."""
    for iteration in (900, 100, 1000, 200):
        _frame(tmp_path, iteration)
    assert [f.iteration for f in frames(tmp_path)] == [100, 200, 900, 1000]
    assert latest_frame(tmp_path).iteration == 1000  # type: ignore[union-attr]


def test_a_case_with_no_frames_is_empty_rather_than_an_error(tmp_path: Path) -> None:
    assert frames(tmp_path) == []
    assert latest_frame(tmp_path) is None


def test_the_verdict_is_read_back_from_the_sidecar(tmp_path: Path) -> None:
    """So the panel can say *where* the field is unphysical without opening
    an 18 MB mesh and a 47 MB solution to find out."""
    _frame(tmp_path, 300, note="1200 nodes above the pitot ceiling", diverged=True)
    frame = latest_frame(tmp_path)
    assert frame is not None
    assert frame.diverged and "pitot" in frame.note


def test_a_frame_whose_sidecar_is_missing_still_shows(tmp_path: Path) -> None:
    """The picture is the point; the note is a bonus and must not gate it."""
    directory = frame_dir(tmp_path)
    directory.mkdir(parents=True)
    (directory / "iter_000400.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    frame = latest_frame(tmp_path)
    assert frame is not None and frame.iteration == 400 and frame.note == ""


def test_files_that_are_not_frames_are_ignored(tmp_path: Path) -> None:
    directory = frame_dir(tmp_path)
    directory.mkdir(parents=True)
    (directory / "iter_000100.png").write_bytes(b"x")
    (directory / "notes.txt").write_text("hello")
    (directory / "iter_bad.png").write_bytes(b"x")
    assert [f.iteration for f in frames(tmp_path)] == [100]


# ---------------------------------------------------------------- rendering


def test_a_case_with_no_snapshot_renders_nothing(tmp_path: Path) -> None:
    """A run queued with `volume_every` 0 writes nothing until it ends."""
    from aether.cfd.frames import render_frame

    assert render_frame(tmp_path, 100) is None


def test_the_writer_only_renders_when_the_snapshot_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard that keeps a two-second render off every three-second poll."""
    drawn: list[int] = []

    def fake_render(case: Path | str, iteration: int, **_: object) -> Frame:
        drawn.append(iteration)
        return Frame(iteration, Path(case) / "x.png")

    monkeypatch.setattr("aether.cfd.frames.render_frame", fake_render)
    snapshot = snapshot_path(tmp_path)
    snapshot.write_text("solution")
    # Backdated: a snapshot is not read until it has stopped being written to.
    os.utime(snapshot, (time.time(), time.time() - 10.0))
    writer = FrameWriter()

    assert writer.poll(tmp_path, 100) is None   # first sight: size not yet known
    assert writer.poll(tmp_path, 100) is not None
    assert writer.poll(tmp_path, 200) is None  # unchanged, so not redrawn
    assert drawn == [100]

    # The solver rewrites it; the next poll draws again.
    os.utime(snapshot, (time.time(), time.time() - 5.0))
    assert writer.poll(tmp_path, 300) is not None
    assert drawn == [100, 300]


def test_a_missing_snapshot_is_not_polled_for(tmp_path: Path) -> None:
    """A queued or meshing run has no volume file and must not error on one."""
    assert FrameWriter().poll(tmp_path, 0) is None


def test_forgetting_a_case_makes_the_next_poll_draw_again(tmp_path: Path) -> None:
    """A re-queued run must not inherit the previous run's 'already drawn'."""
    writer = FrameWriter()
    snapshot = snapshot_path(tmp_path)
    snapshot.write_text("solution")
    os.utime(snapshot, (time.time(), time.time() - 10.0))
    writer.poll(tmp_path, 100)  # first sight records the size
    writer.poll(tmp_path, 100)  # renders nothing real, but records the stamp
    assert str(tmp_path) in writer._seen
    writer.forget(tmp_path)
    assert str(tmp_path) not in writer._seen


@pytest.mark.slow
def test_a_real_case_renders(tmp_path: Path) -> None:
    """Against an actual SU2 volume file, where the results share holds one."""
    from aether.cfd.frames import render_frame
    from aether.paths import cfd_results

    cases = sorted(cfd_results().glob("gen-*/cases/*/volume_flow.dat"))
    if not cases:
        pytest.skip("no solved case with a volume snapshot on this machine")
    frame = render_frame(cases[0].parent, 6000)
    assert frame is not None
    assert frame.image.is_file() and frame.image.stat().st_size > 10_000
    assert frame.sidecar.is_file()


def test_a_snapshot_still_being_written_is_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SU2 spends about a second putting 45 MB down; reading into that fails.

    It failed loudly and repeatedly at a snapshot every twenty-five iterations:
    `node 31065 has 0 values for 24 variables`, straight out of the parser.
    """
    drawn: list[int] = []
    monkeypatch.setattr(
        "aether.cfd.frames.render_frame",
        lambda case, iteration, **_: drawn.append(iteration) or Frame(iteration, Path("x")),
    )
    snapshot_path(tmp_path).write_text("half a solution")
    writer = FrameWriter()

    assert writer.poll(tmp_path, 25) is None  # written just now, so untouched
    assert drawn == []

    stat = snapshot_path(tmp_path).stat()
    os.utime(snapshot_path(tmp_path), (stat.st_atime, time.time() - 10.0))
    assert writer.poll(tmp_path, 25) is None  # size seen for the first time
    assert writer.poll(tmp_path, 25) is not None
    assert drawn == [25]


def test_a_torn_read_is_retried_rather_than_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stamp is recorded on success, so a lost race costs one pass, not a frame."""
    attempts: list[int] = []

    def flaky(case: Path | str, iteration: int, **_: object) -> Frame:
        attempts.append(iteration)
        if len(attempts) == 1:
            raise ValueError("node 31065 has 0 values for 24 variables")
        return Frame(iteration, Path("x"))

    monkeypatch.setattr("aether.cfd.frames.render_frame", flaky)
    snapshot = snapshot_path(tmp_path)
    snapshot.write_text("solution")
    os.utime(snapshot, (time.time(), time.time() - 10.0))
    writer = FrameWriter()
    writer.poll(tmp_path, 25)  # first sight records the size

    # Swallowed: losing the race is expected, and the next pass will win it.
    assert writer.poll(tmp_path, 25) is None
    # Same unchanged snapshot, and it is tried again rather than written off.
    assert writer.poll(tmp_path, 25) is not None
    assert attempts == [25, 25]


def test_a_snapshot_that_never_parses_is_eventually_given_up_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise a genuinely broken file is re-read every three seconds forever."""
    from aether.cfd.frames import MAX_ATTEMPTS

    attempts: list[int] = []

    def always_fails(case: Path | str, iteration: int, **_: object) -> Frame:
        attempts.append(iteration)
        raise ValueError("not a solution at all")

    monkeypatch.setattr("aether.cfd.frames.render_frame", always_fails)
    snapshot = snapshot_path(tmp_path)
    snapshot.write_text("garbage")
    os.utime(snapshot, (time.time(), time.time() - 10.0))
    writer = FrameWriter()
    writer.poll(tmp_path, 25)  # first sight records the size

    for _ in range(MAX_ATTEMPTS + 4):
        with contextlib.suppress(ValueError):
            writer.poll(tmp_path, 25)
    assert len(attempts) == MAX_ATTEMPTS
    # And the last of those is the one that is allowed to escape, so a file
    # that genuinely never parses is heard once rather than never.
    assert writer._failed[str(tmp_path)][1] == MAX_ATTEMPTS


def test_an_unreadable_mesh_is_reported_once_and_not_retried(tmp_path, monkeypatch) -> None:
    """A mesh is fixed for the life of a run, so one failure is final.

    The mesh read used to sit outside the guarded block, where the retry
    counter could not see it -- and that counter is keyed on the *snapshot*
    mtime, which moves every snapshot, so even inside it would have reset
    forever. Measured on the first shock-fitted run: the identical error every
    3.5 seconds for the length of the run.
    """
    from aether.cfd import frames as module

    (tmp_path / "body.su2").write_text("NDIME= 3\n")
    volume = module.snapshot_path(tmp_path)
    volume.write_text("x")
    old = time.time() - 10 * module.QUIESCENT
    os.utime(volume, (old, old))

    reads = []

    def _explode(path):
        reads.append(path)
        raise ValueError("marker 'outflow' holds a face that is not a triangle")

    monkeypatch.setattr(
        "aether.aerodynamics.cfd.fields.read_su2_mesh", _explode, raising=False
    )

    writer = module.FrameWriter()
    # Settle the size check, then the first real attempt raises.
    writer.poll(tmp_path, 100)
    with pytest.raises(ValueError, match="not a triangle"):
        writer.poll(tmp_path, 100)

    # Every later poll is silent, and never touches the file again.
    before = len(reads)
    for iteration in range(200, 260, 10):
        assert writer.poll(tmp_path, iteration) is None
    assert len(reads) == before

    # Until the case is forgotten, at which point it is worth another look.
    writer.forget(tmp_path)
    writer.poll(tmp_path, 300)
    with pytest.raises(ValueError, match="not a triangle"):
        writer.poll(tmp_path, 300)
    assert len(reads) > before


def test_a_schlieren_companion_is_not_listed_as_a_frame_of_its_own(tmp_path) -> None:
    """Both views share a stem, so a loose pattern doubles every iterate."""
    from aether.cfd.frames import frame_dir, frames

    directory = frame_dir(tmp_path)
    directory.mkdir(parents=True)
    (directory / "iter_000100.png").write_bytes(b"x")
    (directory / "iter_000100-schlieren.png").write_bytes(b"x")
    (directory / "iter_000200.png").write_bytes(b"x")

    found = frames(tmp_path)
    assert [f.iteration for f in found] == [100, 200]
    assert found[0].schlieren.name == "iter_000100-schlieren.png"
    assert found[0].view("schlieren").name == "iter_000100-schlieren.png"
    assert found[0].view("mach").name == "iter_000100.png"
    # A frame with no companion falls back rather than serving a missing file.
    assert found[1].view("schlieren").name == "iter_000200.png"


def test_a_snapshot_with_a_newer_sibling_is_the_one_that_is_finished(tmp_path) -> None:
    """The whole point of numbering them: no heuristic, no race.

    SU2 does not open the next dump until the previous is closed, so the
    highest-numbered file is the one being written and everything before it is
    complete. Before this, one overwritten file was left alone until it had been
    quiescent for two seconds *and* the same size on two consecutive polls --
    and on a run putting 35 MB down every four seconds that window never opened.
    Eight consecutive polls read 35592892, 35592409, 35592409, 20691703,
    35593032, 12315160, 35592985, 12835166 bytes, and **not one frame was
    rendered for the entire run**.
    """
    from aether.cfd.frames import completed_snapshots

    assert completed_snapshots(tmp_path) == []

    for number in (25, 50, 75):
        (tmp_path / f"volume_flow_{number:06d}.dat").write_text("x")
    finished = completed_snapshots(tmp_path)
    assert [number for number, _ in finished] == [25, 50]

    # Not confused by the overwrite-mode name sitting beside them.
    (tmp_path / "volume_flow.dat").write_text("x")
    assert [number for number, _ in completed_snapshots(tmp_path)] == [25, 50]


def test_the_reader_prunes_the_snapshots_it_has_passed(tmp_path) -> None:
    """11 GB of Tecplot over an 8000-iteration run is not a preview feature."""
    import contextlib

    from aether.cfd.frames import FrameWriter

    for number in (25, 50, 75, 100):
        (tmp_path / f"volume_flow_{number:06d}.dat").write_text("not a solution")

    # The render fails -- there is no mesh and no config -- but the pruning
    # happens first and is what this is about.
    with contextlib.suppress(Exception):
        FrameWriter().poll(tmp_path, 0)

    left = sorted(p.name for p in tmp_path.glob("volume_flow_*.dat"))
    assert left == ["volume_flow_000075.dat", "volume_flow_000100.dat"]


def test_a_steady_case_numbers_its_snapshots_and_an_unsteady_one_cannot() -> None:
    """SU2 rejects WRT_VOLUME_OVERWRITE=NO on a transient problem, outright."""
    from aether.cfd.config import (
        FlowConditions,
        PitchingMotion,
        ReferenceFrame,
        SU2Case,
    )

    flow = FlowConditions.at_altitude(8.0, 45.0e3)
    reference = ReferenceFrame(area=1.0, length=1.0)

    steady = SU2Case(flow, reference, "m.su2")
    assert "WRT_VOLUME_OVERWRITE= NO" in steady.render()

    moving = SU2Case(
        flow, reference, "m.su2", motion=PitchingMotion(origin=(1.2, 0.0, 0.0))
    )
    assert "WRT_VOLUME_OVERWRITE= YES" in moving.render()


def test_the_canvas_is_sized_from_the_data_not_the_other_way_round() -> None:
    """Where all the white space came from.

    Every panel sets ``aspect="equal"`` -- not negotiable for a flow field --
    so if the canvas aspect does not match the data's, matplotlib shrinks the
    axes and leaves the remainder blank. The canvas was a fixed 7.0 x 3.4,
    chosen for a 2 m sphere-cone, and on the hemisphere's 52 mm x 104 mm view
    box the axes collapsed to a column with **more than half the frame empty**.
    """
    from aether.cfd.frames import FIELD_MAX, FIELD_MIN, field_size

    # A tall box gets a tall field, a wide box a wide one, and in both cases the
    # field's aspect is the data's.
    for box, aspect in (
        ((-0.007, 0.045, -0.052, 0.052), 0.5),      # hemisphere
        ((-0.02, 2.06, -0.75, 0.75), 2.08 / 1.5),   # sphere-cone
        ((0.0, 1.0, 0.0, 1.0), 1.0),
    ):
        width, height = field_size(box)
        assert width / height == pytest.approx(aspect, rel=1e-9)
        assert min(width, height) >= FIELD_MIN - 1e-9
        assert max(width, height) <= FIELD_MAX + 1e-9


def test_a_very_slender_body_does_not_produce_a_frame_too_wide_to_look_at() -> None:
    """Past aspect FIELD_MAX/FIELD_MIN the two clamps cannot both hold, and the
    cap is the one that must: written the other way round, the minimum short
    side undid it and a 40:1 view box asked for a frame six feet wide."""
    from aether.cfd.frames import FIELD_MAX, field_size

    width, height = field_size((0.0, 40.0, -0.5, 0.5))
    assert width == pytest.approx(FIELD_MAX)
    # The aspect is kept whatever else gives, because that is what keeps the
    # axes filling the canvas.
    assert width / height == pytest.approx(40.0, rel=1e-9)


def test_the_panel_without_a_colour_scale_does_not_reserve_room_for_one() -> None:
    """The Schlieren frame shared the Mach frame's canvas and carried an inch
    of empty paper down its right edge."""
    from aether.cfd.frames import GUTTER_PLAIN, GUTTER_RIGHT

    assert GUTTER_PLAIN < GUTTER_RIGHT / 3.0
