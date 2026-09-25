"""CFD workbench dashboard — stripped to the solve monitor.

One clock, one page, one panel: the CFD run queue with residual charts,
force traces, field snapshots, and schlieren frames.  Everything durable
lives in :mod:`~aether.cfd.registry`; this is a reader that can be
closed, reopened on another device, or killed without touching a run.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

from aether.cfd.registry import load_all
from aether.cfd.worker import worker_pid
from nicegui import ui

from aether.cfd.dashboard.cfd import cfd_panel
from aether.cfd.dashboard.theme import MUTED, apply

__all__ = ["main", "page"]

PERIOD = 1.0


def _header() -> Callable[[], None]:
    with ui.header().classes("items-center justify-between q-px-md").props("elevated"):
        with ui.row().classes("items-center q-gutter-sm no-wrap"):
            ui.icon("scatter_plot", size="1.4rem")
            ui.label("AETHER CFD").classes("text-subtitle1 text-weight-medium")
        with ui.row().classes("items-center q-gutter-xs no-wrap"):
            work = ui.chip("", icon="memory").props("dense square")

    def refresh() -> None:
        pid = worker_pid()
        active = sum(1 for r in load_all() if r.active)
        work.text = (
            f"worker · {active} active" if pid and active else "worker" if pid else "no worker"
        )
        work.props(f'color={"aether-good" if pid else "aether-bad"} text-color=white')

    refresh()
    return refresh


@ui.page("/")
def page() -> None:
    apply()
    open_runs: set[str] = set()
    refreshers: list[Callable[[], None]] = []

    refreshers.append(_header())

    with ui.column().classes("w-full q-pa-sm q-gutter-sm"):
        refreshers.append(cfd_panel(open_runs))

    def tick() -> None:
        for refresh in refreshers:
            refresh()

    ui.timer(PERIOD, tick)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AETHER CFD dashboard")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args(argv)

    ui.run(
        host=args.host,
        port=args.port,
        title="AETHER CFD",
        dark=True,
        show=False,
        reload=False,
        reconnect_timeout=30.0,
        favicon="🔥",
    )
    return 0
