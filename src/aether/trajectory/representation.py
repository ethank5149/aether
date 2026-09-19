"""Reference trajectories as pseudospectral polynomials.

The optimizer produces a trajectory as state and control values at
Legendre-Gauss-Lobatto (LGL) nodes; the *solution* between the nodes is the
Lagrange polynomial through them (Trefethen, *Spectral Methods in MATLAB*,
Ch. 5). This module carries that polynomial as a first-class, serialisable
object so the optimised trajectory can be stored offline once and replayed as
the feed-forward reference a GNC loop tracks — evaluated at any time, and
differentiated exactly for reference rates and accelerations.

Evaluation is by the barycentric form of Lagrange interpolation (Berrut &
Trefethen, SIAM Review 2004), which is numerically stable and O(N) per query
once the weights are formed. A multi-stage trajectory is a sequence of
:class:`PhasePolynomial` segments, one per powered/coast/glide phase, joined at
free staging times; :class:`ReferenceTrajectory` dispatches a query to the
segment that owns the time.

The stored artifact is the coefficient set — nodes, node values, and the
phase break times — not a resampling of the trajectory onto a dense grid. That
keeps it compact and exact: re-evaluation reproduces the optimiser's own
polynomial to machine precision at the nodes and spectral accuracy between.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from aether.optimal_control.pseudospectral import OCPSolution

_FloatArray = NDArray[np.float64]

__all__ = [
    "PhasePolynomial",
    "ReferenceTrajectory",
    "barycentric_weights",
]


def barycentric_weights(nodes: _FloatArray) -> _FloatArray:
    """Barycentric interpolation weights for arbitrary distinct ``nodes``.

    ``w_i = 1 / prod_{j != i} (x_i - x_j)``. Only the ratios of the weights
    enter the barycentric formula, so the nodes are first shifted and scaled to
    ``[-1, 1]`` (which keeps the products near unit magnitude and avoids
    overflow for the 10-40 node phases used here) and the result normalised by
    its largest magnitude. Computed as a direct reciprocal product — exact to
    rounding, unlike a log-sum round-trip.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    n = nodes.size
    if n < 2:
        raise ValueError(f"need at least two nodes, got {n}")
    span = float(nodes[-1] - nodes[0]) or 1.0
    scaled = 2.0 * (nodes - nodes[0]) / span - 1.0
    diff = scaled[:, None] - scaled[None, :]
    np.fill_diagonal(diff, 1.0)
    weights = 1.0 / np.prod(diff, axis=1)
    return np.asarray(weights / np.max(np.abs(weights)), dtype=np.float64)


@dataclass(frozen=True)
class PhasePolynomial:
    """One trajectory phase: a Lagrange polynomial on ``[t0, t1]``.

    Attributes
    ----------
    t0, t1:
        Phase start and end times (s), ``t1 > t0``.
    nodes:
        LGL (or other distinct) time nodes on ``[t0, t1]``, shape ``(N,)``.
    state_values:
        State at each node, shape ``(N, nx)``.
    control_values:
        Control at each node, shape ``(N, nu)``. May be empty (``nu = 0``) for
        a coast phase carrying no control.
    label:
        Phase name (``"boost-1"``, ``"coast"``, ``"glide"``, ...).
    """

    t0: float
    t1: float
    nodes: _FloatArray
    state_values: _FloatArray
    control_values: _FloatArray
    label: str = ""
    _weights: _FloatArray = field(default_factory=lambda: np.empty(0), repr=False)

    def __post_init__(self) -> None:
        if not self.t1 > self.t0:
            raise ValueError(f"phase {self.label!r} needs t1 > t0, got {self.t0}, {self.t1}")
        nodes = np.asarray(self.nodes, dtype=np.float64)
        if nodes.ndim != 1 or nodes.size < 2:
            raise ValueError("nodes must be a 1-D array of length >= 2")
        if np.asarray(self.state_values).shape[0] != nodes.size:
            raise ValueError("state_values must have one row per node")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "state_values", np.asarray(self.state_values, dtype=np.float64))
        object.__setattr__(
            self, "control_values", np.asarray(self.control_values, dtype=np.float64)
        )
        if self._weights.size != nodes.size:
            object.__setattr__(self, "_weights", barycentric_weights(nodes))

    # -- evaluation -------------------------------------------------------

    @property
    def nx(self) -> int:
        return int(self.state_values.shape[1]) if self.state_values.ndim == 2 else 0

    @property
    def nu(self) -> int:
        return int(self.control_values.shape[1]) if self.control_values.ndim == 2 else 0

    def _bary(self, values: _FloatArray, t: float) -> _FloatArray:
        """Barycentric Lagrange interpolation of ``values`` at scalar ``t``."""
        d = t - self.nodes
        hit = np.isclose(d, 0.0, atol=1e-12)
        if np.any(hit):
            return np.asarray(values[int(np.argmax(hit))], dtype=np.float64)
        c = self._weights / d
        return np.asarray((c @ values) / np.sum(c), dtype=np.float64)

    def state(self, t: float) -> _FloatArray:
        """State at time ``t`` (clamped to the phase interval)."""
        return self._bary(self.state_values, float(np.clip(t, self.t0, self.t1)))

    def control(self, t: float) -> _FloatArray:
        """Control at time ``t``; empty if the phase carries none."""
        if self.nu == 0:
            return np.empty(0)
        return self._bary(self.control_values, float(np.clip(t, self.t0, self.t1)))

    def derivative(self, t: float, order: int = 1) -> _FloatArray:
        """``order``-th time derivative of the state at ``t`` (finite-precision).

        Uses complex-step-free central differencing on the barycentric
        polynomial; the polynomial is smooth on the open interval, so a small
        step gives the reference rate/acceleration a tracking loop feeds
        forward. For the exact spectral derivative build the differentiation
        matrix from :mod:`aether.optimal_control.pseudospectral`.
        """
        if order < 1:
            raise ValueError("order must be >= 1")
        span = self.t1 - self.t0
        h = 1e-4 * span
        tc = float(np.clip(t, self.t0 + h, self.t1 - h))
        if order == 1:
            return (self.state(tc + h) - self.state(tc - h)) / (2.0 * h)
        return (self.derivative(tc + h, order - 1) - self.derivative(tc - h, order - 1)) / (2.0 * h)


@dataclass(frozen=True)
class ReferenceTrajectory:
    """A multi-phase reference trajectory: ordered :class:`PhasePolynomial` s.

    The phases must tile ``[t_start, t_end]`` without gaps in time order. A
    query dispatches to the phase whose interval contains the time; at a phase
    boundary the earlier phase owns the instant.

    ``metadata`` records provenance a GNC loop or a library index needs: which
    side flew it (``"threat"`` / ``"interceptor"``), the vehicle id, the
    objective it optimised, and the decision variables (staging times, launch
    epoch) the optimiser chose. It is free-form and serialised verbatim.
    """

    phases: tuple[PhasePolynomial, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.phases:
            raise ValueError("a reference trajectory needs at least one phase")
        phases = tuple(sorted(self.phases, key=lambda p: p.t0))
        for a, b in pairwise(phases):
            if not np.isclose(a.t1, b.t0, atol=1e-6):
                raise ValueError(
                    f"phases must tile time without gaps: {a.label!r} ends at "
                    f"{a.t1} but {b.label!r} starts at {b.t0}"
                )
            if a.nx != b.nx:
                raise ValueError("all phases must share the same state dimension")
        object.__setattr__(self, "phases", phases)

    @property
    def t_start(self) -> float:
        return self.phases[0].t0

    @property
    def t_end(self) -> float:
        return self.phases[-1].t1

    @property
    def nx(self) -> int:
        return self.phases[0].nx

    @property
    def staging_times(self) -> tuple[float, ...]:
        """Interior phase-boundary times — the staging schedule."""
        return tuple(p.t1 for p in self.phases[:-1])

    def _phase_at(self, t: float) -> PhasePolynomial:
        tc = float(np.clip(t, self.t_start, self.t_end))
        for phase in self.phases:
            if tc <= phase.t1 + 1e-9:
                return phase
        return self.phases[-1]

    def state(self, t: float) -> _FloatArray:
        """State at time ``t`` (clamped to the trajectory span)."""
        return self._phase_at(t).state(t)

    def control(self, t: float) -> _FloatArray:
        """Control at time ``t``."""
        return self._phase_at(t).control(t)

    def derivative(self, t: float, order: int = 1) -> _FloatArray:
        """State time-derivative at ``t`` — reference rates for feed-forward."""
        return self._phase_at(t).derivative(t, order)

    def sample(self, n: int = 200) -> tuple[_FloatArray, _FloatArray]:
        """Dense ``(times, states)`` for plotting or a legacy sampled consumer."""
        times = np.linspace(self.t_start, self.t_end, n)
        states = np.array([self.state(float(t)) for t in times])
        return times, states

    # -- construction -----------------------------------------------------

    @classmethod
    def from_solution(
        cls,
        solution: OCPSolution,
        t0: float = 0.0,
        label: str = "phase",
        metadata: dict[str, object] | None = None,
    ) -> ReferenceTrajectory:
        """Wrap a single-phase pseudospectral :class:`OCPSolution`."""
        t = np.asarray(solution.t, dtype=np.float64)
        phase = PhasePolynomial(
            t0=float(t[0]) + t0,
            t1=float(t[-1]) + t0,
            nodes=t + t0,
            state_values=np.asarray(solution.x, dtype=np.float64),
            control_values=np.asarray(solution.u, dtype=np.float64),
            label=label,
        )
        return cls(phases=(phase,), metadata=dict(metadata or {}))

    @classmethod
    def from_phases(
        cls, phases: list[PhasePolynomial], metadata: dict[str, object] | None = None
    ) -> ReferenceTrajectory:
        return cls(phases=tuple(phases), metadata=dict(metadata or {}))

    # -- serialisation ----------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write the trajectory to a compressed ``.npz`` (the offline artifact).

        Stores every phase's nodes and node values plus the metadata as JSON;
        no dense resampling, so the file is the optimiser's own polynomial.
        """
        import json

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, _FloatArray] = {}
        index = []
        for i, p in enumerate(self.phases):
            arrays[f"nodes_{i}"] = p.nodes
            arrays[f"state_{i}"] = p.state_values
            arrays[f"control_{i}"] = p.control_values
            index.append({"t0": p.t0, "t1": p.t1, "label": p.label})
        np.savez_compressed(
            destination,
            _index=np.array(json.dumps(index)),
            _metadata=np.array(json.dumps(self.metadata, default=str)),
            n_phases=np.array(len(self.phases)),
            **arrays,  # type: ignore[arg-type]
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> ReferenceTrajectory:
        """Read a trajectory written by :meth:`save`."""
        import json

        with np.load(Path(path), allow_pickle=False) as data:
            index = json.loads(str(data["_index"]))
            metadata = json.loads(str(data["_metadata"]))
            phases = [
                PhasePolynomial(
                    t0=float(entry["t0"]),
                    t1=float(entry["t1"]),
                    nodes=data[f"nodes_{i}"],
                    state_values=data[f"state_{i}"],
                    control_values=data[f"control_{i}"],
                    label=str(entry["label"]),
                )
                for i, entry in enumerate(index)
            ]
        return cls(phases=tuple(phases), metadata=metadata)
