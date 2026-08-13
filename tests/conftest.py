"""Shared fixtures.

Only one thing lives here, and it earns its place: Matplotlib figures made
through ``pyplot`` are held open until closed, and
:meth:`~aether.viz.animator.TrajectoryAnimator.frame_at` makes one whenever
it is not handed an axes to draw into. A test file that renders more than
twenty frames therefore trips Matplotlib's ``figure.max_open_warning``,
which this project's ``filterwarnings = ["error"]`` turns into a failure —
and the failure lands on whichever test happened to be the twenty-first,
not on the ones that leaked. That is a confusing way to find out about a
resource leak, so figures are closed after every test instead.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def close_matplotlib_figures() -> Iterator[None]:
    """Close every open figure after each test."""
    yield
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is optional
        return
    plt.close("all")
