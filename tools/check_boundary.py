#!/usr/bin/env python3
"""Fail if the public kernel has grown a dependency on controlled code.

The split between this package and the controlled applied layer is only
worth anything if it stays one-way: ``aether_gambit`` may import ``aether``,
never the reverse. A single stray import is enough to make the public
package unbuildable without the controlled one — at which point publishing
it either breaks, or leaks.

This check is deliberately dumb and textual. It does not import anything,
so it runs in a clean checkout with no dependencies installed, and it
cannot be defeated by an import that only executes on some code path.

Run from the repository root::

    python tools/check_boundary.py

Exits non-zero, listing every offending line, if any is found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Any module name that must never appear in this repository. The controlled
# package itself, plus the top-level names that were moved out of it — a
# reference to a bare `guidance`/`orbital` import is just as much a leak as
# one to `aether_gambit`, and usually means a file was copied back by hand.
FORBIDDEN_PACKAGES = ("aether_gambit",)
FORBIDDEN_SUBMODULES = (
    "guidance",
    "flight",
    "orbital",
    "systems",
    "sensor",
    "estimation",
    "viz",
    "aerothermal",
    "aerodynamics",
    "geometry",
    "fiat",
)

IMPORT_RE = re.compile(
    r"^\s*(?:from\s+(?P<from>[\w.]+)|import\s+(?P<import>[\w.]+))",
)

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "build", "dist", ".mypy_cache",
             ".pytest_cache", ".ruff_cache", "manuscript"}


def offending_module(name: str) -> str | None:
    """Return the forbidden module `name` resolves to, or None if it is fine."""
    head, _, _ = name.partition(".")
    if head in FORBIDDEN_PACKAGES:
        return head
    if head in FORBIDDEN_SUBMODULES:
        return head
    # `aether.guidance` would mean a controlled module was moved back in.
    if head == "aether":
        parts = name.split(".")
        if len(parts) > 1 and parts[1] in FORBIDDEN_SUBMODULES:
            return name
    return None


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    failures: list[str] = []

    for path in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:  # pragma: no cover
            failures.append(f"{path}: unreadable ({exc})")
            continue
        for lineno, line in enumerate(lines, start=1):
            match = IMPORT_RE.match(line)
            if match is None:
                continue
            name = match.group("from") or match.group("import")
            bad = offending_module(name)
            if bad is not None:
                rel = path.relative_to(root)
                failures.append(f"{rel}:{lineno}: imports controlled module {bad!r}")

    if failures:
        print("BOUNDARY VIOLATION — the public kernel must not depend on", file=sys.stderr)
        print("controlled code. Offending imports:\n", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        print(
            "\nIf this module genuinely belongs in the public kernel, move it here"
            "\nand delete it from the controlled repository — do not import across"
            "\nthe boundary.",
            file=sys.stderr,
        )
        return 1

    print("boundary OK — no controlled imports in the public kernel")
    return 0


if __name__ == "__main__":
    sys.exit(main())
