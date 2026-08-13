"""Shared report plumbing for the verification runners."""

from __future__ import annotations

import csv
import datetime as _dt
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy

__all__ = ["VerificationReport", "write_csv"]


@dataclass
class VerificationReport:
    """One verification task's outcome: criterion, measurements, verdict."""

    task_id: str
    title: str
    criterion: str
    passed: bool
    sections: list[str] = field(default_factory=list)
    #: Source of the stated criterion; Paper II tasks carry the "II-" prefix
    #: in ``task_id`` and are cited against §8 of the companion manuscript.
    source: str = ""

    def add_section(self, heading: str, body: str) -> None:
        self.sections.append(f"## {heading}\n\n{body.rstrip()}\n")

    def add_table(
        self,
        heading: str,
        headers: list[str],
        rows: list[list[str]],
        notes: str | None = None,
    ) -> None:
        """Emit a markdown table, optionally followed by prose.

        ``notes`` keeps the interpretation attached to the numbers it
        interprets; a table whose caveats live in a separate section is a
        table that gets quoted without them.
        """
        lines = [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join("---" for _ in headers) + "|",
        ]
        lines += ["| " + " | ".join(row) + " |" for row in rows]
        if notes is not None:
            lines += ["", notes]
        self.add_section(heading, "\n".join(lines))

    def to_markdown(self) -> str:
        verdict = "**PASS**" if self.passed else "**FAIL**"
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        source = self.source or (
            "Paper II §8" if self.task_id.startswith("II-") else "Paper I §8"
        )
        head = (
            f"# {self.task_id}: {self.title}\n\n"
            f"- **Failure criterion (stated in advance, {source}):** {self.criterion}\n"
            f"- **Verdict:** {verdict}\n"
            f"- **Generated:** {stamp} · numpy {np.__version__} · scipy {scipy.__version__} "
            f"· {platform.python_implementation()} {platform.python_version()} "
            f"({platform.machine()})\n"
        )
        return head + "\n" + "\n".join(self.sections)

    def write(self, output_dir: Path, stem: str) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{stem}.md"
        path.write_text(self.to_markdown(), encoding="utf-8")
        return path


def write_csv(output_dir: Path, stem: str, headers: list[str], rows: list[list[Any]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)
    return path
