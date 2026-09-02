#!/usr/bin/env python3
"""Fail if the public kernel names a specific fielded weapon system.

A companion to ``check_boundary.py``, which deliberately checks the import
direction and nothing else. This one checks *content*: the public split rests
on the argument that what is here is "general scientific, mathematical, or
engineering principles" and "general system descriptions" (22 CFR 120.31(b),
120.33(b)) rather than information "required for the design, development,
production ... of defense articles" (22 CFR 120.33(a)(1)).

A named designation is not by itself technical data -- "reentry vehicle"
appears in the USML itself, and in open literature back to Allen & Eggers
(1958). Tying a *specific parameter set* to a *specific fielded system* is
where the general description stops being general, and that is what this looks
for: not the vocabulary, but the pairing.

So a hit is **not** a violation. It is a place where somebody has to decide,
and record the decision, rather than letting one accumulate silently. Cleared
hits go in ``ALLOWED`` with the reason.

What this cannot do, restated because a tool like this invites more confidence
than it earns: it matches literal strings. It cannot recognise a designation it
has not been told about, an obfuscated one, or -- most importantly -- a
parameter set that is recognisably a specific system without ever naming it.
Subject-matter judgement stays manual. This file is not evidence about it.

Run from the repository root::

    python tools/check_disclosure.py

Exits non-zero, listing every unreviewed hit, if any is found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

#: Designations of specific fielded systems. Matched case-insensitively, on a
#: word boundary, so `mk21` and `Mk-21` both hit and `mark` does not.
DESIGNATIONS = (
    r"mk-?\s?2[01]a?",
    r"mk-?\s?[45]00",
    r"minuteman",
    r"peacekeeper",
    r"trident",
    r"polaris",
    r"poseidon",
    r"sarmat",
    r"avangard",
    r"topol",
    r"df-?\d{1,2}",
    r"dongfeng",
    r"hwasong",
    r"w\d{2}\s+warhead",
)

SEARCHED = ("src", "tests", "docs", "notebooks")
SUFFIXES = (".py", ".md", ".rst", ".ipynb", ".toml", ".cfg")

#: Reviewed and kept, with the reason. A path/designation pair listed here is
#: asserted to be a general system description or published information, not a
#: parameter set required for design. Adding an entry is the act of taking
#: responsibility for that assertion -- it is not a way to quiet the tool.
ALLOWED: dict[tuple[str, str], str] = {
    ("src/aether/viz/vehicle.py", "mk21"): (
        "Published external envelope used to scale a drawing. A general system "
        "description under 120.31(b); no mass properties, no aerothermal or "
        "structural data, nothing required for design. Reviewed 2026-09-02."
    ),
    ("src/aether/viz/vehicle.py", "mk21a"): (
        "As above -- the same published envelope, '-class' qualified. "
        "Reviewed 2026-09-02."
    ),
}


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    pattern = re.compile(r"\b(" + "|".join(DESIGNATIONS) + r")\b", re.IGNORECASE)
    unreviewed: list[str] = []

    for directory in SEARCHED:
        base = root / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix not in SUFFIXES or not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            for number, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                for match in pattern.finditer(line):
                    # Normalised so that "Mk21A", "Mk-21a" and "mk 21a" are one
                    # key: the reviewer decided about the system, not the
                    # spelling that happened to appear first.
                    key = re.sub(r"[-\s]", "", match.group(0)).lower()
                    if (relative, key) in ALLOWED:
                        continue
                    unreviewed.append(f"{relative}:{number}: {match.group(0)}  |  {line.strip()[:90]}")

    if unreviewed:
        print("Unreviewed references to specific fielded systems in the public kernel:\n")
        for hit in unreviewed:
            print(f"  {hit}")
        print(
            f"\n{len(unreviewed)} hit(s). Each needs a decision: remove it, generalise it, "
            "or add it to ALLOWED in this file with the reason it is defensible."
        )
        return 1

    print(f"disclosure OK — no unreviewed system designations ({len(ALLOWED)} reviewed and kept)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
