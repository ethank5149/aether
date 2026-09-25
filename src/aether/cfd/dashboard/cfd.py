"""The CFD run panel: queue a solve, watch it, and see the field it is making.

One tab of :mod:`~aether_gambit.cfd.dashboard.app`, and deliberately shaped so
the others can be written the same way -- it exports a single
:func:`cfd_panel`, holds no module-level state, and gets everything it draws
from disk on a timer.

The layout is lopsided on purpose. A frame of the flow field and one line
saying whether any of it is unphysical answer "how is this going" in about a
second; ten residual curves and a table of drops answer it too, eventually, and
only if you already know what you are looking at. So the picture is always on
screen and the numbers are one tap away, rather than the other way round.

Nothing here holds state that matters. Close the tab mid-solve, reopen it
tomorrow on a different device, and the page reads the same JSON records and
the same ``history.csv`` the worker is writing, and draws the same thing.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from aether.cfd.config import FlowConditions, regime_for
from aether.cfd.frames import Frame, frame_dir, frames
from aether.cfd.geometries import REFERENCE_BODIES
from aether.cfd.registry import (
    RunRecord,
    RunSpec,
    archived,
    cancel_requested,
    enqueue,
    forget,
    load,
    load_all,
    request_cancel,
    restore,
)
from aether.cfd.run import (
    SolveProgress,
    default_ranks,
    read_history,
    read_progress,
)
from aether.cfd.worker import worker_pid
from nicegui import app, ui

from aether.cfd.dashboard.theme import (
    BAD,
    MUTED,
    STATE_COLOUR,
    axis_style,
    chart_base,
)

__all__ = ["cfd_panel"]

_STATE_COLOUR = STATE_COLOUR

#: Points kept when drawing a trace. A converged run writes tens of thousands
#: of rows; the shape of a residual curve survives decimation, and the zoom
#: below is what recovers detail when detail is actually wanted.
_PLOT_POINTS = 1200

#: Lines of ``su2.log`` shown. Enough to hold the solver's parting words --
#: a non-physical-point report, an MPI abort -- without shipping a 40 MB file
#: to a phone.
_LOG_LINES = 60

#: How often an open diagnostics panel redraws, in seconds. Deliberately slower
#: than the list itself: the numbers above want to be current, but rebuilding a
#: chart underneath someone who is reading it is its own kind of unusable.
_CHART_PERIOD = 8.0


def _decimate(values: Sequence[float], keep: int = _PLOT_POINTS) -> list[list[float]]:
    """``[iteration, value]`` pairs, thinned to something a phone can draw.

    Pairs rather than bare values, because the iteration number has to survive
    the thinning: plotted against a category axis the x labels would silently
    become "index of the points that happened to be kept", which is a different
    quantity that looks identical.
    """
    total = len(values)
    if total <= keep:
        return [[float(i), float(v)] for i, v in enumerate(values)]
    step = total // keep + 1
    points = [[float(i), float(values[i])] for i in range(0, total, step)]
    # The last row is appended unconditionally: it is the current residual, and
    # a stride that happens to skip it makes a live run look stalled.
    if points[-1][0] != total - 1:
        points.append([float(total - 1), float(values[-1])])
    return points


def _clock(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "--"
    seconds = int(seconds)
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def _log_tail(record: RunRecord, lines: int = _LOG_LINES) -> str:
    """The end of ``su2.log``, which is where a failure explains itself.

    Read from the end rather than parsed: these reach tens of megabytes on a
    long run, and the only part anyone wants is the last screenful.
    """
    path = record.log
    if path is None or not path.exists():
        return ""
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            window = min(size, 64 * 1024)
            handle.seek(size - window)
            blob = handle.read(window)
    except OSError:
        return ""
    text = blob.decode("utf-8", "replace")
    if window < size:
        # The first line of the window is almost certainly cut in half.
        text = text.split("\n", 1)[-1]
    return "\n".join(text.splitlines()[-lines:])


def _progress(record: RunRecord) -> SolveProgress | None:
    """This run's progress, read from its history.

    ``started`` comes from ``record.elapsed`` rather than the clock, because
    that stops at ``finished_at``. Measuring a finished run against *now* makes
    its reported rate sag a little further on every refresh.
    """
    if record.history is None:
        return None
    started = None
    elapsed = record.elapsed
    if elapsed is not None:
        started = time.monotonic() - elapsed
    return read_progress(record.history, total=record.total, started=started)


def _progress_line(record: RunRecord) -> tuple[str, float | None]:
    """A run's progress as a sentence and a bar fraction.

    Read from ``history.csv`` rather than from the record, and this is the
    division the whole design rests on: the record carries *identity* -- which
    process, started when -- and the history carries *progress*.
    """
    if record.state == "queued":
        return "waiting for a slot", None
    if record.state == "meshing":
        # The worker's own note when it has one. The clock here counts from
        # `queued_at`, so on a run that waited an hour for a slot it reads as an
        # hour of meshing -- indistinguishable from a hang, which is exactly
        # what it was being asked to rule out.
        return record.message or "meshing", None

    reading = _progress(record)
    if reading is None:
        if record.active:
            return "started; no history yet", None
        return record.message or record.state, None
    finished = not record.active
    line = reading.describe(eta=not finished)
    if finished:
        line = f"{record.message or record.state} -- {line}"
    return line, reading.fraction


def _num(widget: Any, fallback: float) -> float:
    """A number field's value, tolerating the empty box.

    ``ui.number`` holds ``None`` while its box is empty, and a viewer clearing
    a field to retype it passes through that state on every keystroke. Reading
    it as a float raises, and the exception lands in the refresh callback of
    whatever was bound to it -- which is how a form that merely looked
    half-typed started throwing `float() argument must be ... not NoneType`
    into the server log several times a second.
    """
    value = getattr(widget, "value", None)
    if value is None or value == "":
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


# ------------------------------------------------------------------- charts


#: Orders of magnitude a residual must have fallen from its own peak before it
#: stops being flagged. One dex is a low bar deliberately: this marks what is
#: *not moving*, not what has converged.
_STALLED_DEX = 1.0


def _drop(values: Sequence[float]) -> float:
    """Orders of magnitude a trace has fallen from its peak.

    From the peak, not from the first row, and the distinction is the same one
    :func:`~aether.cfd.run.read_progress` makes: several equations begin
    at the solver's floor and can only rise. A five-species start carries no NO,
    N or O, so those residuals begin near -32; a uniform initial field has no
    cross-flow, so RhoV and RhoW begin near -12. Measuring from the first row
    reads initialisation as divergence.
    """
    return max(values) - values[-1]


def _residual_options(trace: dict[str, list[float]]) -> dict[str, Any]:
    """Residual traces against iteration, zoomable, with the laggards marked.

    What gets flagged is a trace that has not *fallen* from its own peak -- not
    one sitting above zero. The difference is not pedantry: SU2's residuals are
    dimensional, so on a perfectly healthy Mach 20 five-species run `rms[RhoE]`
    sits around +5 while `rms[Rho_0]` sits at -3, and colouring by sign paints
    half of a converging run red. Falling by orders of magnitude is the thing
    every equation does when it converges, whatever its units, so that is what
    is measured.

    The least-converged series is always marked, since it is the one gating the
    run and the one the summary line names.
    """
    traces = {
        name: values
        for name, values in trace.items()
        if name.lower().startswith("rms") and values
    }
    drops = {name: _drop(values) for name, values in traces.items()}
    worst = min(drops, key=lambda name: drops[name]) if drops else None

    series: list[dict[str, Any]] = []
    for name, values in traces.items():
        stalled = name == worst or drops[name] < _STALLED_DEX
        entry: dict[str, Any] = {
            "name": f"{name} ({drops[name]:+.1f})",
            "type": "line",
            "showSymbol": False,
            "lineStyle": {"width": 2.6 if stalled else 1.4},
            "data": _decimate(values),
            "z": 3 if stalled else 2,
        }
        if stalled:
            entry["color"] = BAD
        series.append(entry)

    base = chart_base()
    axis = axis_style()
    return {
        **base,
        # Off, and not for taste: an animated redraw of eight series every few
        # seconds is both distracting and, on a phone, genuinely slow.
        "animation": False,
        "grid": {"left": 58, "right": 18, "top": 30, "bottom": 58},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
        "legend": {**base["legend"], "type": "scroll", "top": 0},
        "xAxis": {
            **axis,
            "type": "value",
            "name": "iteration",
            "nameLocation": "middle",
            "nameGap": 26,
            "splitLine": {"show": False},
        },
        "yAxis": {
            **axis,
            "type": "value",
            "name": "log10 residual",
            "scale": True,
        },
        # `inside` is the one that matters on a phone: it is pinch-to-zoom and
        # drag-to-pan. The slider is for a mouse, and shows how much of the run
        # the current view covers.
        "dataZoom": [
            {"type": "inside", "filterMode": "none"},
            {"type": "slider", "height": 18, "bottom": 6, "filterMode": "none"},
        ],
        "series": series,
    }


def _force_options(trace: dict[str, list[float]]) -> dict[str, Any]:
    """CD and CL against iteration, on their own axes.

    Two axes because the two coefficients differ by an order of magnitude or
    more on these bodies -- on one axis the smaller is a flat line along the
    bottom, which is indistinguishable from a coefficient that is not moving.
    """
    present = [name for name in ("CD", "CL") if trace.get(name)]
    series = [
        {
            "name": name,
            "type": "line",
            "showSymbol": False,
            "yAxisIndex": index,
            "lineStyle": {"width": 1.6},
            "data": _decimate(trace[name]),
        }
        for index, name in enumerate(present)
    ]
    base = chart_base()
    axis = axis_style()
    return {
        **base,
        "animation": False,
        "grid": {"left": 62, "right": 62, "top": 26, "bottom": 34},
        "tooltip": {"trigger": "axis"},
        "legend": {**base["legend"], "top": 0},
        "xAxis": {**axis, "type": "value", "splitLine": {"show": False}},
        "yAxis": [
            {
                **axis,
                "type": "value",
                "name": name,
                "scale": True,
                "position": "left" if index == 0 else "right",
            }
            for index, name in enumerate(present)
        ]
        or [{**axis, "type": "value"}],
        "series": series,
    }


@app.get("/frame")
def _frame_image(run: str, iteration: int, view: str = "mach") -> Any:
    """Serve one rendered frame.

    Addressed by run id and iteration, never by path. The panel is
    unauthenticated on a tailnet and this is the one route that takes anything
    resembling a filename; resolving through the registry means there is no
    caller-supplied path segment to traverse with.
    """
    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    record = load(run)
    if record is None or record.case_dir is None:
        raise HTTPException(status_code=404, detail="no such run")
    image = frame_dir(record.case_dir) / f"iter_{iteration:06d}.png"
    if view == "schlieren":
        shaded = image.with_name(image.stem + "-schlieren.png")
        if shaded.is_file():
            image = shaded
    if not image.is_file():
        raise HTTPException(status_code=404, detail="no such frame")
    return FileResponse(image, media_type="image/png")


def sweep_records(states: set[str]) -> int:
    """Forget every finished record whose state is in ``states``. Returns the count.

    **Active runs are never swept, whatever is asked for.** Stopping something
    is a different decision from tidying up after it, and one button that did
    both would eventually kill a run somebody was watching. Case directories
    are left alone either way -- this removes the registry record, not the
    result.
    """
    removed = 0
    for record in load_all():
        if record.state in states and not record.active:
            try:
                forget(record.run_id)
            except RuntimeError:
                continue
            removed += 1
    return removed


@app.get("/meshpreview")
def _mesh_image(run: str, stamp: int = 0) -> Any:
    """Serve the meshing preview. Same resolve-through-the-registry rule as frames.

    ``stamp`` is the file's mtime, supplied by the caller so a re-meshed case
    gets a new URL. Without it a browser would happily show the previous run's
    grid.
    """
    from aether.cfd.meshpreview import preview_path
    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    record = load(run)
    if record is None or record.case_dir is None:
        raise HTTPException(status_code=404, detail="no such run")
    image = preview_path(record.case_dir)
    if not image.is_file():
        raise HTTPException(status_code=404, detail="no mesh preview")
    return FileResponse(image, media_type="image/png")


# --------------------------------------------------------------------- runs


class _RunCard:
    """One run's card, built once and updated in place.

    Rebuilding every card on every poll is what made the panels snap shut every
    three seconds, and it would have thrown away a pinch-zoom just as readily.
    So the elements are created once and their *contents* replaced; the charts
    are touched only while their panel is open, and then only every
    :data:`_CHART_PERIOD` seconds.

    The layout is deliberately lopsided. A frame of the flow field and one line
    saying whether any of it is unphysical answer "how is this going" in about a
    second; ten residual curves and a table of drops answer it too, eventually,
    and only if you already know what you are looking at. So the picture is
    always on screen and the numbers are one tap away, rather than the other way
    round.
    """

    def __init__(
        self,
        record: RunRecord,
        on_change: Callable[[], None],
        open_runs: set[str],
    ) -> None:
        self.record = record
        self._on_change = on_change
        self._open_runs = open_runs
        self._drawn = 0.0
        self._frames: list[Frame] = []
        # The slider's upper bound is a Quasar prop, not a Python attribute, so
        # the frame count is tracked here rather than read back off the widget.
        self._count = 0
        # Whether the scrubber tracks the newest frame. It does until the
        # viewer drags it somewhere else, and again as soon as they drag it
        # back to the end -- so watching a live run needs no interaction, and
        # looking back at iteration 200 is not undone three seconds later.
        self._follow = True

        with ui.card().classes("w-full q-mb-sm q-pa-sm"):
            with ui.row().classes("w-full items-center no-wrap"):
                self.badge = ui.badge(record.state)
                ui.label(record.spec.label()).classes("text-weight-medium")
                ui.space()
                self.stop = ui.button(icon="stop", color="red-8", on_click=self._cancel)
                self.stop.props("flat dense round")
                self.drop = ui.button(icon="delete_outline", color="grey", on_click=self._forget)
                self.drop.props("flat dense round")
                # Finished runs start collapsed. A queue that has been going all
                # day is mostly history, and history should not push the two
                # runs that are actually moving off the bottom of the screen.
                self._collapsed = record.run_id not in open_runs and not record.active
                self.fold = ui.button(icon="expand_less", on_click=self._fold)
                self.fold.props("flat dense round")

            self.bar = ui.linear_progress(value=0.0, show_value=False).props("rounded")
            self.line = ui.label().classes("text-caption font-mono")
            self.meta = ui.label().classes("text-caption").style(f"color: {MUTED}")

            self.body = ui.column().classes("w-full q-gutter-none")

        with self.body:

                # ---------------------------------------------------- the field
                # Tabs, not a replacement. The grid does not stop being worth
                # looking at when the first snapshot lands -- a solve that goes
                # wrong is usually a question about the mesh, and having to wait
                # for the run to end to see it again is the wrong way round.
                with ui.tabs().props("dense align=left").classes("w-full") as self.tabs:
                    self.tab_flow = ui.tab("flow")
                    self.tab_shade = ui.tab("schlieren")
                    self.tab_mesh = ui.tab("mesh")
                self.panels = ui.tab_panels(self.tabs, value=self.tab_flow).classes(
                    "w-full"
                ).props("keep-alive")

                with self.panels, ui.tab_panel(self.tab_flow).classes("q-pa-none"):
                    self.field = ui.column().classes("w-full q-gutter-none")
                with self.field:
                    self.image = ui.image().classes("w-full rounded-borders")
                    with ui.row().classes("w-full items-center no-wrap q-gutter-sm"):
                        self.scrub = ui.slider(min=0, max=0, value=0, on_change=self._scrubbed)
                        self.scrub.props("dense label-always").classes("col")
                        self.live = ui.button(icon="fast_forward", on_click=self._resume_follow)
                        self.live.props("flat dense round").tooltip("follow the newest frame")
                    self.note = ui.label().classes("text-caption")

                with self.panels, ui.tab_panel(self.tab_shade).classes("q-pa-none"):
                    self.shade_image = ui.image().classes("w-full rounded-borders")
                    ui.label(
                        "density gradient: white is undisturbed, dark is a shock"
                    ).classes("text-caption").style(f"color: {MUTED}")

                # The grid is the only view that answers *where the cells went*,
                # which a bare element count cannot and which this pipeline got
                # wrong for a fortnight.
                with self.panels, ui.tab_panel(self.tab_mesh).classes("q-pa-none"):
                    self.mesh_view = ui.column().classes("w-full q-gutter-none")
                with self.mesh_view:
                    self.mesh_image = ui.image().classes("w-full rounded-borders")
                    self.mesh_note = ui.label().classes("text-caption").style(
                        f"color: {MUTED}"
                    )

                self.waiting = ui.label().classes("text-caption").style(f"color: {MUTED}")

                # ------------------------------------------------ the numbers
                self.convergence = ui.expansion(
                    "convergence", icon="show_chart", on_value_change=self._toggled
                ).classes("w-full").props("dense")
                with self.convergence:
                    self.residuals = ui.echart(_residual_options({})).classes("w-full").style(
                        "height: 320px"
                    )
                    ui.label(
                        "pinch or scroll to zoom, drag to pan"
                    ).classes("text-caption").style(f"color: {MUTED}")
                    self.table = ui.table(
                        columns=[
                            {"name": "eq", "label": "equation", "field": "eq", "align": "left"},
                            {"name": "now", "label": "latest", "field": "now", "align": "right"},
                            {"name": "dex", "label": "fallen", "field": "dex", "align": "right"},
                        ],
                        rows=[],
                        row_key="eq",
                    ).classes("w-full").props("dense flat")
                    ui.label("forces").classes("text-caption q-mt-sm").style(f"color: {MUTED}")
                    self.forces = ui.echart(_force_options({})).classes("w-full").style(
                        "height: 190px"
                    )

                self.log_panel = ui.expansion("solver log", icon="terminal").classes(
                    "w-full"
                ).props("dense")
                with self.log_panel, ui.scroll_area().classes("w-full").style("height: 220px"):
                    self.log = ui.label().classes(
                        "text-caption font-mono whitespace-pre"
                    ).style("line-height: 1.35")

        self.update(record, force=True)

    # ------------------------------------------------------------ behaviour

    def _fold(self) -> None:
        self._collapsed = not self._collapsed
        self._apply_fold()

    def _apply_fold(self) -> None:
        self.body.set_visibility(not self._collapsed)
        self.fold.props(f'icon={"expand_more" if self._collapsed else "expand_less"}')

    def _toggled(self, event: Any) -> None:
        if event.value:
            self._open_runs.add(self.record.run_id)
            self.refresh_numbers(force=True)
        else:
            self._open_runs.discard(self.record.run_id)

    def _scrubbed(self, event: Any) -> None:
        index = int(_num(event, 0.0))
        self._follow = index >= self._count - 1
        self.live.set_visibility(not self._follow)
        self._show_frame(index)

    def _resume_follow(self) -> None:
        self._follow = True
        if self._frames:
            self.scrub.value = len(self._frames) - 1
            self._show_frame(len(self._frames) - 1)

    def _cancel(self) -> None:
        request_cancel(self.record.run_id)
        # The worker does the stopping, on its next pass, so this reports what
        # was asked for rather than what has happened.
        ui.notify(f"stop requested for {self.record.spec.label()}", type="warning")
        self.stop.disable()

    def _forget(self) -> None:
        try:
            forget(self.record.run_id)
        except RuntimeError as error:
            ui.notify(str(error), type="negative")
            return
        ui.notify("record removed; the case directory is untouched")
        self._on_change()

    # --------------------------------------------------------------- render

    def update(self, record: RunRecord, force: bool = False) -> None:
        self.record = record

        self.badge.text = record.state
        self.badge.props(f'color={_STATE_COLOUR.get(record.state, "grey")}')

        line, fraction = _progress_line(record)
        self.line.text = line
        self.bar.set_visibility(fraction is not None)
        if fraction is not None:
            self.bar.value = fraction

        bits = []
        if record.ranks:
            bits.append(f"{record.ranks} ranks")
        if record.elements:
            bits.append(f"{record.elements} cells")
        if record.elapsed is not None:
            bits.append(_clock(record.elapsed))
        self.meta.text = "  ".join(bits)
        self.meta.set_visibility(bool(bits))

        self.stop.set_visibility(record.active)
        if record.active and cancel_requested(record.run_id):
            self.stop.disable()
        self.drop.set_visibility(not record.active)

        self._sync_frames()
        self._sync_mesh()
        self._apply_fold()

        has_history = record.history is not None and record.history.exists()
        self.convergence.set_visibility(has_history)
        self.log_panel.set_visibility(has_history)
        if has_history and self.convergence.value:
            self.refresh_numbers(force=force)
        if self.log_panel.value:
            self.log.text = _log_tail(record) or "(no solver log yet)"

    def _sync_frames(self) -> None:
        """Pick up frames the worker has rendered since the last look.

        Cheap enough to do on every poll -- it is a directory listing and a
        few small JSON reads, against the seconds the worker spent producing
        them.
        """
        if self.record.case_dir is None:
            self._frames = []
        else:
            self._frames = frames(self.record.case_dir)

        self.tab_flow.set_visibility(bool(self._frames))
        self.tab_shade.set_visibility(bool(self._frames))
        if not self._frames:
            self.field.set_visibility(False)
            self.waiting.set_visibility(self.record.state in {"solving", "meshing"})
            self.waiting.text = _no_frames_because(self.record)
            return

        self.waiting.set_visibility(False)
        self.field.set_visibility(True)
        self._count = len(self._frames)
        last = self._count - 1
        self.scrub.props(f"min=0 max={last}")
        # A scrubber over a single frame is furniture, not a control.
        self.scrub.set_visibility(last > 0)
        self.live.set_visibility(last > 0 and not self._follow)
        if self._follow:
            self.scrub.value = last
        self._show_frame(int(_num(self.scrub, 0.0)))

    def _sync_mesh(self) -> None:
        """Show the grid whenever there is no flow field to show instead.

        Not only while meshing: a run that failed during its prelude never
        produced a snapshot, and the grid it failed on is exactly what you want
        to look at.
        """
        from aether.cfd.meshpreview import preview_path

        if self.record.case_dir is None:
            self.tab_mesh.set_visibility(False)
            return
        image = preview_path(self.record.case_dir)
        if not image.is_file():
            self.tab_mesh.set_visibility(False)
            return
        self.tab_mesh.set_visibility(True)
        # Before the first snapshot there is no flow to show, so open on the
        # grid rather than on an empty panel.
        if not self._frames and self.panels.value != self.tab_mesh:
            self.panels.value = self.tab_mesh
        # mtime in the URL, so a re-meshed case is never served from cache.
        stamp = int(image.stat().st_mtime)
        self.mesh_image.source = f"/meshpreview?run={self.record.run_id}&stamp={stamp}"
        self.mesh_note.text = "grid, before solving"

    def _show_frame(self, index: int) -> None:
        if not self._frames:
            return
        frame = self._frames[max(0, min(index, len(self._frames) - 1))]
        # The iteration is in the URL, so a new frame is a new URL and no cache
        # anywhere between here and the phone can serve the previous one.
        base = f"/frame?run={self.record.run_id}&iteration={frame.iteration}"
        self.image.source = base
        self.shade_image.source = f"{base}&view=schlieren"
        self.note.text = (
            frame.note if frame.diverged else f"iteration {frame.iteration}: field looks physical"
        )
        self.note.classes(replace="text-caption")
        self.note.style(f"color: {BAD if frame.diverged else MUTED}")

    def refresh_numbers(self, force: bool = False) -> None:
        """Redraw the residual chart, the per-equation table and the forces."""
        now = time.monotonic()
        if not force and now - self._drawn < _CHART_PERIOD:
            return
        self._drawn = now

        history = self.record.history
        trace = read_history(history, columns=["rms", "CD", "CL"]) if history else {}

        self.residuals.options.update(_residual_options(trace))
        self.residuals.update()
        self.forces.options.update(_force_options(trace))
        self.forces.update()

        reading = _progress(self.record)
        if reading is not None:
            # Every equation, not just the worst. A five-species run whose
            # species residuals look healthy while `rms[RhoE]` has climbed four
            # orders is the case this table exists to make visible.
            self.table.rows = [
                {
                    "eq": name,
                    "now": f"{value:+.3f}",
                    "dex": f"{reading.drops.get(name, 0.0):+.2f}",
                }
                for name, value in sorted(reading.residuals.items())
            ]
            self.table.update()


def _no_frames_because(record: RunRecord) -> str:
    """Say why there is no picture, since the usual reason is a setting.

    A run queued with snapshots off writes its volume once, at the end. That is
    a legitimate choice and the cheapest one, but with the field now the main
    thing on screen it is worth saying out loud rather than leaving a blank
    space that looks like a fault.
    """
    if record.spec.volume_every <= 0:
        return "no field snapshots: this run was queued with 'volume snapshot every' set to 0"
    if record.state == "meshing":
        return "meshing; the first snapshot comes after the first iterations"
    return f"first snapshot due at iteration {record.spec.volume_every}"


# ------------------------------------------------------------------- launch


def _launch_form(on_queued: Callable[[], None]) -> None:
    """The queue-a-run form. Its only side effect is one new record on disk."""
    state: dict[str, Any] = {"body": "sphere-cone", "geometry": {}}

    @ui.refreshable
    def geometry_controls() -> None:
        spec = REFERENCE_BODIES[state["body"]]
        state["geometry"] = {p.name: p.default for p in spec.parameters}
        for parameter in spec.parameters:

            # Both bound as defaults: a closure over the loop variable would
            # give every control the *last* parameter's name and default.
            def _set(
                event: Any,
                name: str = parameter.name,
                fallback: float = parameter.default,
            ) -> None:
                state["geometry"][name] = _num(event, fallback)

            unit = f" [{parameter.unit}]" if parameter.unit else ""
            ui.number(
                label=f"{parameter.name}{unit}",
                value=parameter.default,
                min=parameter.low,
                max=parameter.high,
                step=parameter.step,
                on_change=_set,
            ).classes("w-full").props("dense outlined")

    @ui.refreshable
    def condition_note() -> None:
        """What the flight condition actually is, before it is paid for."""
        flow = FlowConditions.at_altitude(
            mach=_num(mach, 8.0),
            altitude=1.0e3 * _num(altitude, 45.0),
            alpha=float(np.radians(_num(alpha, 0.0))),
        )
        ui.label(
            f"{type(regime_for(_num(mach, 8.0))).__name__}  "
            f"p {flow.pressure:.3f} Pa  T {flow.temperature:.2f} K  "
            f"V {flow.speed:.0f} m/s"
        ).classes("text-caption font-mono").style(f"color: {MUTED}")

    def choose_body(event: Any) -> None:
        state["body"] = event.value
        # The parameters are a property of the shape, so the controls are
        # rebuilt rather than re-labelled: a biconic has six and a Sears-Haack
        # body has two, and they are not the same two.
        geometry_controls.refresh()

    ui.select(
        sorted(REFERENCE_BODIES),
        value=state["body"],
        label="body",
        on_change=choose_body,
    ).classes("w-full").props("dense outlined")

    with ui.expansion("geometry").classes("w-full").props("dense"):
        geometry_controls()

    with ui.row().classes("w-full no-wrap q-gutter-sm"):
        mach = (
            ui.number(label="Mach", value=8.0, min=0.5, max=30.0, step=0.5,
                      on_change=lambda _: condition_note.refresh())
            .classes("col").props("dense outlined")
        )
        alpha = (
            ui.number(label="alpha [deg]", value=0.0, min=-25.0, max=25.0, step=0.5,
                      on_change=lambda _: condition_note.refresh())
            .classes("col").props("dense outlined")
        )
    with ui.row().classes("w-full no-wrap q-gutter-sm"):
        altitude = (
            ui.number(label="altitude [km]", value=45.0, min=0.0, max=90.0, step=1.0,
                      on_change=lambda _: condition_note.refresh())
            .classes("col").props("dense outlined")
        )
        iterations = (
            ui.number(label="iterations", value=6000, min=200, max=60000, step=200)
            .classes("col").props("dense outlined")
        )
    condition_note()

    with ui.expansion("mesh and solver", value=True).classes("w-full").props("dense"):
        # Defaults are what has been measured to work, not what the pipeline
        # shipped with. Everything below was the failing configuration until
        # 2026-09-11: an isotropic tetrahedral mesh with zero cells inside the
        # shock standoff, solved with a flux scheme that limit-cycles at a
        # strong bow shock. Runs queued from this form went out that way and
        # died -- a Mach 8 biconic on a NaN, a Mach 16 case on a residual above
        # 1e20 -- while the settings that fix both sat in RunSpec, reachable
        # only from a script.
        fitted = ui.switch(
            "shock-fitted mesh (wall-normal, marched to the bow shock)", value=True
        )
        ui.label(
            "The isotropic mesher put zero cells between the body and the shock "
            "at 1.36M elements. This one puts 45 to 120 there and builds in "
            "about two seconds."
        ).classes("text-caption").style(f"color: {MUTED}")
        cell_reynolds = ui.number(
            label="cell Reynolds number at the wall (LAURA ships 1)",
            value=100.0, min=0.5, max=1000.0, step=10.0, format="%.4g",
        ).classes("w-full").props("dense outlined")
        cell_reynolds.bind_visibility_from(fitted, "value")
        refinement = ui.number(
            label="refinement level (scales surface and layers together)",
            value=1.0, min=0.25, max=3.0, step=0.05, format="%.4f",
        ).classes("w-full").props("dense outlined")
        refinement.bind_visibility_from(fitted, "value")
        flux = ui.select(
            ["HLLC", "SLAU2", "SLAU", "AUSMPLUSUP2", "AUSMPLUSM", "ROE", "MSW"],
            label="convective flux", value="HLLC",
        ).classes("w-full").props("dense outlined")
        ui.label(
            "Measured on one mesh: HLLC settles to a residual of -5.8 with a "
            "drag ripple of 0.0002%, SLAU to -4.0, AUSMPLUSUP2 limit-cycles at "
            "-1.0 and 2%, and Roe diverges with a spurious lift of -0.65 on an "
            "axisymmetric body at zero incidence."
        ).classes("text-caption").style(f"color: {MUTED}")
        viscous = ui.select(
            {
                "": "laminar Navier-Stokes",
                "SST": "RANS, k-omega SST",
                "SA": "RANS, Spalart-Allmaras",
            },
            label="viscous model", value="",
        ).classes("w-full").props("dense outlined")
        ui.label(
            "Laminar for this campaign. Re_L is 6.3e5 on the sphere-cone at "
            "Mach 8 and 45 km and 1.4e4 on the hemisphere, both under the ~1e6 "
            "where hypersonic boundary layers transition -- and a RANS model "
            "asked to run below that NaN'd every body in the 2026-09-11 suite "
            "at around iteration 217. This control did not exist then, which is "
            "why all three took the default."
        ).classes("text-caption").style(f"color: {MUTED}")
        settle = ui.switch("stop when the force settles", value=True)
        ui.label(
            "A fixed iteration count is a bound, not a criterion: the isotropic "
            "mesh settles inside 200 iterations and burns the rest, while a "
            "wall-resolved one was still moving 1% per 100 at the cap."
        ).classes("text-caption").style(f"color: {MUTED}")

        ranks = ui.number(
            label="MPI ranks (0 = size from the mesh)", value=0, min=0, max=64, step=1
        ).classes("w-full").props("dense outlined")
        ui.label(
            f"0 sizes from the mesh at ~40k cells/rank; this machine would take "
            f"{default_ranks()} at most."
        ).classes("text-caption").style(f"color: {MUTED}")
        wall_refinement = ui.number(
            label="wall refinement (isotropic mesher only)",
            value=0.05, min=0.001, max=0.2, step=0.005, format="%.4f",
        ).classes("w-full").props("dense outlined")
        # It sizes nothing on the shock-fitted path, where the wall cell comes
        # from the cell Reynolds number instead. Showing it there invites a
        # refinement study made of identical grids, which is how two runs were
        # wasted.
        wall_refinement.bind_visibility_from(fitted, "value", backward=lambda v: not v)
        quality = ui.number(
            label="quality floor", value=0.1, min=0.01, max=0.5, step=0.01
        ).classes("w-full").props("dense outlined")
        volume_every = ui.number(
            label="field snapshot every N iterations (0 = end only)",
            value=50, min=0, max=2000, step=25,
        ).classes("w-full").props("dense outlined")
        ui.label(
            "Snapshots are what the field view is drawn from, so 0 means no "
            "pictures until the run ends. The file is about 650 bytes a node "
            "and takes a second to write, against tens of seconds an iteration."
        ).classes("text-caption").style(f"color: {MUTED}")
        adapt_cycles = ui.number(
            label="adaptation cycles (0 = off)",
            value=3, min=0, max=10, step=1,
        ).classes("w-full").props("dense outlined")
        ui.label(
            "Solution-adaptive refinement: after each solve, multi-sensor "
            "gradients (density, pressure, temperature, species) drive a new "
            "mesh so every resolved feature gets the resolution the physics "
            "demands. 3 cycles by default."
        ).classes("text-caption").style(f"color: {MUTED}")
        optimize = ui.switch("optimise the mesh (removes slivers)", value=True)
        optimize.bind_visibility_from(fitted, "value", backward=lambda v: not v)
        nose = ui.switch("refine the stagnation region", value=False)
        nose.bind_visibility_from(fitted, "value", backward=lambda v: not v)
        nose_cell = ui.number(
            label="nose cell [m]", value=0.004, min=0.0001, max=0.05, step=0.0005, format="%.4f"
        ).classes("w-full").props("dense outlined")
        nose_cell.bind_visibility_from(nose, "value")

    def submit() -> None:
        spec = RunSpec(
            body=state["body"],
            mach=_num(mach, 8.0),
            alpha_deg=_num(alpha, 0.0),
            altitude_km=_num(altitude, 45.0),
            iterations=int(_num(iterations, 6000)),
            geometry=dict(state["geometry"]),
            wall_refinement=_num(wall_refinement, 0.05),
            optimize=bool(optimize.value),
            quality_threshold=_num(quality, 0.1),
            ranks=int(_num(ranks, 0)) or None,
            volume_every=int(_num(volume_every, 25)),
            nose_cell=_num(nose_cell, 0.004) if nose.value else None,
            shock_fitted=bool(fitted.value),
            cell_reynolds=_num(cell_reynolds, 100.0),
            refinement=_num(refinement, 1.0),
            convective=str(flux.value or "HLLC"),
            turbulence=str(viscous.value) or None,
            settle_tolerance=5.0e-3 if settle.value else 0.0,
            adapt_cycles=int(_num(adapt_cycles, 3)),
        )
        # Refuse it here rather than ten minutes into a solve. Every check this
        # runs corresponds to a failure that actually cost a run today.
        from aether.cfd.preflight import preflight

        report = preflight(spec)
        if not report.ok:
            for failure in report.failures:
                ui.notify(f"{failure.name}: {failure.detail}", type="negative", timeout=12000)
            return
        for warning in report.warnings:
            ui.notify(f"{warning.name}: {warning.detail}", type="warning", timeout=10000)
        enqueue(spec)
        ui.notify(f"queued {spec.label()}", type="positive")
        for check in report.checks:
            if check.name in {"shock layer resolved", "mesh built", "shock provider"}:
                ui.notify(f"{check.name}: {check.detail}", type="info", timeout=6000)
        if worker_pid() is None:
            ui.notify(
                "no worker is running -- this will sit queued until one starts",
                type="warning",
                timeout=8000,
            )
        on_queued()

    ui.button("queue run", on_click=submit, icon="add").classes("w-full q-mt-md").props(
        "color=primary unelevated"
    )


# --------------------------------------------------------------------- page


def cfd_panel(open_runs: set[str]) -> Callable[[], None]:
    """Build the CFD tab and return the function that refreshes it.

    ``open_runs`` is owned by the page rather than by this panel, so which
    diagnostics are expanded survives a card being rebuilt -- and is per client,
    so one person expanding a run does not expand it on somebody else's phone.

    The returned callable is what the page's timer drives. Handing it back
    rather than starting a timer here keeps every panel on one clock, which is
    what stops a page with four tabs from polling the disk four times as often.
    """
    cards: dict[str, _RunCard] = {}
    #: Which states the list is showing. Finished work is on by default, because
    #: hiding it silently is how a failed run stops being noticed.
    shown: set[str] = set(_STATE_COLOUR) | {"done", "failed", "stopped"}

    def _sweep(states: set[str]) -> None:
        removed = sweep_records(states)
        ui.notify(
            f"cleared {removed} record(s) -- archived, not deleted; "
            "'restore cleared' brings them back"
            if removed
            else "nothing to clear",
            type="positive" if removed else "info",
        )
        sync(rebuild=True)

    def _restore_all() -> None:
        ids = archived()
        for run_id in ids:
            restore(run_id)
        ui.notify(
            f"restored {len(ids)} record(s)" if ids else "nothing archived",
            type="positive" if ids else "info",
        )
        sync(rebuild=True)

    def _fold_all(collapsed: bool) -> None:
        for card in cards.values():
            card._collapsed = collapsed
            card._apply_fold()

    with ui.column().classes("w-full q-gutter-sm"):
        with ui.expansion("queue a run", icon="add").classes("w-full").props("dense"):
            _launch_form(lambda: sync(rebuild=True))

        with ui.row().classes("w-full items-center no-wrap q-gutter-xs"):
            counts = ui.label().classes("text-caption font-mono").style(f"color: {MUTED}")
            ui.space()
            collapse = ui.button(icon="unfold_less", on_click=lambda: _fold_all(True))
            collapse.props("flat dense round").tooltip("collapse every card")
            expand = ui.button(icon="unfold_more", on_click=lambda: _fold_all(False))
            expand.props("flat dense round").tooltip("expand every card")
            with ui.button(icon="filter_list").props("flat dense round"):
                with ui.menu(), ui.column().classes("q-pa-sm"):
                    for state in ("queued", "meshing", "solving", "done", "failed", "stopped"):
                        ui.checkbox(
                            state,
                            value=True,
                            on_change=lambda event, name=state: (
                                shown.add(name) if event.value else shown.discard(name),
                                sync(rebuild=True),
                            ),
                        ).props("dense")
            with ui.button(icon="cleaning_services", color="grey").props(
                "flat dense round"
            ), ui.menu(), ui.column().classes("q-pa-xs"):
                ui.item("clear failed", on_click=lambda: _sweep({"failed"}))
                ui.item("clear stopped", on_click=lambda: _sweep({"stopped"}))
                ui.item("clear done", on_click=lambda: _sweep({"done"}))
                ui.separator()
                ui.item(
                    "clear all finished",
                    on_click=lambda: _sweep({"done", "failed", "stopped"}),
                )
                ui.separator()
                # Clearing is reversible: the record is archived, not
                # deleted, because it is the only index into a case
                # directory that is still sitting on disk.
                ui.item("restore cleared", on_click=_restore_all)

        runs_container = ui.column().classes("w-full")

    def sync(rebuild: bool = False) -> None:
        """Update the run list, rebuilding its structure only when it changed.

        The distinction is the whole point. Tearing the list down every few
        seconds is what closed an open diagnostics panel and would have reset a
        pinch-zoom mid-read; a run whose *status* moved only needs its labels
        rewritten, which leaves everything the viewer was interacting with
        exactly where it was.
        """
        every = sorted(load_all(), key=lambda r: r.queued_at, reverse=True)
        tally: dict[str, int] = {}
        for record in every:
            tally[record.state] = tally.get(record.state, 0) + 1
        line = "  ".join(f"{n} {state}" for state, n in sorted(tally.items())) or "empty"
        retired = len(archived())
        counts.text = f"{line}   ({retired} cleared)" if retired else line

        records = [r for r in every if r.state in shown]
        if rebuild or [r.run_id for r in records] != list(cards):
            runs_container.clear()
            cards.clear()
            with runs_container:
                if not records:
                    ui.label(
                        "nothing queued yet" if not every else "nothing matches the filter"
                    ).classes("text-caption q-pa-md").style(f"color: {MUTED}")
                for record in records:
                    cards[record.run_id] = _RunCard(record, sync, open_runs)
            return
        for record in records:
            cards[record.run_id].update(record)

    sync(rebuild=True)
    return sync
