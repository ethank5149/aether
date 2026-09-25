"""Entry point: ``python -m aether.cfd.dashboard``."""

from __future__ import annotations

from aether.cfd.dashboard.app import main

if __name__ in {"__main__", "__mp_main__"}:
    raise SystemExit(main())
