"""``python -m aether.cfd.dashboard`` -- CFD run monitoring panel.

A NiceGUI dashboard for queuing, watching, and inspecting SU2 solves.
All durable state is in :mod:`~aether.cfd.registry`; this is a reader.
"""

from __future__ import annotations

from aether.cfd.dashboard.app import main, page

__all__ = ["main", "page"]
