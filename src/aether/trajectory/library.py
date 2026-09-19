"""An offline library of optimised reference trajectories.

Trajectory optimization is expensive and is done ahead of time; flight tracks a
*precomputed* reference. This is the store for that: a directory of saved
:class:`~aether.trajectory.representation.ReferenceTrajectory` coefficient
files with a JSON index of their metadata, so a mission can look up "the
minimum-flight-time boost-glide for this vehicle to this target" without
re-solving. The index holds only metadata; the trajectories themselves are the
compact ``.npz`` polynomial files the representation writes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from aether.trajectory.representation import ReferenceTrajectory

__all__ = ["LibraryEntry", "TrajectoryLibrary"]


@dataclass(frozen=True)
class LibraryEntry:
    """One catalogued trajectory: its file and the metadata it was saved with."""

    name: str
    file: str
    metadata: dict[str, object]


class TrajectoryLibrary:
    """A directory-backed catalogue of reference trajectories.

    Parameters
    ----------
    root:
        Directory holding the ``.npz`` trajectory files and ``index.json``.
        Created if absent.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "index.json"
        self._entries: dict[str, LibraryEntry] = {}
        if self._index_path.exists():
            raw = json.loads(self._index_path.read_text())
            self._entries = {
                name: LibraryEntry(name, e["file"], e["metadata"])
                for name, e in raw.items()
            }

    def _flush(self) -> None:
        payload = {
            name: {"file": e.file, "metadata": e.metadata}
            for name, e in self._entries.items()
        }
        self._index_path.write_text(json.dumps(payload, indent=2, default=str))

    def add(self, name: str, trajectory: ReferenceTrajectory) -> LibraryEntry:
        """Save ``trajectory`` under ``name`` and index its metadata."""
        file = f"{name}.npz"
        trajectory.save(self.root / file)
        entry = LibraryEntry(name, file, dict(trajectory.metadata))
        self._entries[name] = entry
        self._flush()
        return entry

    def get(self, name: str) -> ReferenceTrajectory:
        """Load a catalogued trajectory by name."""
        if name not in self._entries:
            raise KeyError(f"no trajectory named {name!r} in {self.root}")
        return ReferenceTrajectory.load(self.root / self._entries[name].file)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def query(self, **fields: object) -> tuple[LibraryEntry, ...]:
        """Entries whose metadata matches every ``field=value`` given.

        ``library.query(side="threat", objective="min-flight-time")`` returns
        every catalogued trajectory flown by the threat under that objective.
        """
        out = []
        for entry in self._entries.values():
            if all(entry.metadata.get(k) == v for k, v in fields.items()):
                out.append(entry)
        return tuple(out)
