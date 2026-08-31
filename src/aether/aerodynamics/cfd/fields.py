"""The volume solution, recovered from what a finished SU2 run already wrote.

The case configuration in this suite asks for ``OUTPUT_FILES= ( RESTART,
SURFACE_CSV )``. There is no ParaView file, and adding one would have meant
re-running every case to look at any of them. It is not necessary: the binary
restart *is* the volume solution at the mesh nodes, in the mesh's own node
order, and the ``.su2`` file next to it carries the connectivity. Everything
here reads those two and nothing else, so every case already on disk is
visualisable without touching the solver.

What this module does not do is draw. It returns arrays and a triangulation;
:mod:`aether.viz.flow` turns those into a picture. The split is deliberate --
a plane cut through a tetrahedral mesh is a geometry question with a right
answer, and it is testable as one only while no plotting library is involved.

Four things here are load-bearing and easy to get subtly wrong.

**Node order.** The restart stores no connectivity and no coordinates that
anyone should trust for indexing -- it stores fields in mesh node order. Read
a restart against the wrong mesh and every array still has a plausible shape
and every picture is wrong. :func:`read_case` therefore refuses a pairing
whose point counts disagree, which is the only cross-check the file formats
make possible.

**Cut topology, computed once.** A plane cut is expressed as a set of
*intersected edges* with interpolation weights, not as a set of coordinates.
Any number of fields can then be cut along the same topology, and they cut
consistently by construction. Interpolating each field independently would
have allowed pressure and density to disagree about where the surface is.

**One point, one vertex.** Marching tetrahedra emits a vertex per cell, so
the same point arrives once for every cell around it, and the cut has to fold
those back together. It matters more than it sounds: coincident vertices draw
perfectly as filled contours and make the triangulation invalid for anything
that has to *locate* a point in it, so the defect is invisible in a figure and
fatal to a streamline. Folding them is exact rather than a tolerance weld,
which took getting the floating-point identities right in three places --
:func:`_signed_distance`, :func:`_merge_shared_edges` and :func:`_lerp`, each
of which says which coincidence it exists to remove.

**Gradients are per-element and exact.** The Schlieren image is a picture of
:math:`|\\nabla\\rho|`, and on a tetrahedral mesh the linear basis has one
constant gradient per element which is exact for a linear field. That is
computed and then volume-averaged to the nodes, rather than differencing
neighbouring node values, which would have been a second discretisation laid
over the solver's own.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_FloatArray = NDArray[np.float64]
_IntArray = NDArray[np.int64]

__all__ = [
    "PERFECT_AIR",
    "CaloricallyPerfect",
    "Mesh",
    "PlaneCut",
    "SurfaceCut",
    "VolumeField",
    "gas_constant_air",
    "node_gradient",
    "plane_cut",
    "read_case",
    "read_restart",
    "read_su2_mesh",
    "read_volume",
    "surface_cut",
]

# SU2 writes its binary restart with this leading magic number; the value is
# the file's only self-identification, so it is also the check that a file is
# a restart at all rather than, say, an ASCII one renamed.
_RESTART_MAGIC = 535532

# Field names are fixed-width in the header. Not a null-terminated stream --
# reading it as one silently consumes the following names as padding, which
# is exactly the failure that produced a field list of ``['x', 'Momentum_x']``
# followed by four entries of binary noise.
_NAME_WIDTH = 33

# VTK cell types, as SU2 writes them into the element sections.
_TETRA = 10
_PRISM = 13
_TRIANGLE = 5

#: Specific gas constant for dry air. Named rather than written as a literal
#: so that a case running on something else is a visible substitution.
gas_constant_air = 287.058


@dataclass(frozen=True)
class CaloricallyPerfect:
    """A constant-:math:`\\gamma`, constant-:math:`R` gas.

    Named and passed rather than defaulted into each formula, because the
    range this suite works over is exactly the range where the assumption
    stops holding. Below about Mach 10 air is calorically perfect to within a
    percent or so; by Mach 20 the shock layer is dissociating, and a perfect
    gas -- having no vibrational modes to excite and no dissociation to absorb
    the enthalpy -- overpredicts the shock-layer temperature by a factor of
    several. :func:`~aether_gambit.cfd.config.regime_for` already switches the
    *solver* to nonequilibrium above Mach 10 for that reason, and this object
    exists so that the post-processing cannot quietly disagree with it.

    It is not a model of high-temperature air and does not pretend to be. Its
    job is to make the assumption an argument, so that a caller who is outside
    its range has to say so.
    """

    gamma: float = 1.4
    gas_constant: float = gas_constant_air

    def __post_init__(self) -> None:
        if self.gamma <= 1.0:
            raise ValueError(f"gamma must exceed 1, got {self.gamma}")
        if self.gas_constant <= 0.0:
            raise ValueError(f"gas constant must be positive, got {self.gas_constant}")


#: Dry air as a calorically perfect gas. The default only because it is what
#: the perfect-gas cases in this suite are run with -- not because it is the
#: regime the suite is about.
PERFECT_AIR = CaloricallyPerfect()

#: What the perfect-gas primitives need a restart to contain.
_CONSERVATIVE = ("Density", "Momentum_x", "Momentum_y", "Momentum_z", "Energy")


@dataclass(frozen=True)
class VolumeField:
    """Conservative variables at the mesh nodes, plus the primitives they imply.

    The stored arrays are what SU2 wrote. Pressure, temperature and Mach are
    derived on request rather than cached, because they depend on the gas
    model and a stale cache computed under a different one would be
    undetectable.

    A restart from the nonequilibrium solver does not carry these variables --
    it carries per-species densities and a second, vibrational-electronic
    energy -- so the primitives here do not apply to it and say so rather than
    failing on a missing key.
    """

    points: _FloatArray
    """``(N, 3)`` node coordinates as recorded in the restart."""

    fields: dict[str, _FloatArray]
    """Non-coordinate fields, keyed by the restart's own names."""

    @property
    def size(self) -> int:
        """Number of nodes."""
        return int(self.points.shape[0])

    @property
    def is_perfect_gas(self) -> bool:
        """Whether this restart carries the single-species conservative set."""
        return all(name in self.fields for name in _CONSERVATIVE)

    def _require_perfect_gas(self) -> None:
        """Refuse, with the reason, rather than raise ``KeyError`` on a name."""
        if self.is_perfect_gas:
            return
        missing = [name for name in _CONSERVATIVE if name not in self.fields]
        species = sorted(n for n in self.fields if n.startswith("Density_"))
        hint = ""
        if species or "Energy_ve" in self.fields:
            hint = (
                " This looks like a NEMO (thermochemical nonequilibrium) restart"
                f" -- it carries {len(species)} species densities"
                f"{' and a vibrational-electronic energy' if 'Energy_ve' in self.fields else ''}."
                " Perfect-gas pressure, temperature and Mach are not defined for it:"
                " the mixture's composition sets its gas constant and its specific"
                " heats, and both vary through the shock layer."
            )
        raise ValueError(
            f"restart lacks {missing}, so the perfect-gas primitives do not apply."
            f"{hint} Present fields: {sorted(self.fields)}"
        )

    @property
    def density(self) -> _FloatArray:
        """Mixture density, in the solver's dimensional units."""
        self._require_perfect_gas()
        return self.fields["Density"]

    @property
    def momentum(self) -> _FloatArray:
        """``(N, 3)`` momentum vector."""
        self._require_perfect_gas()
        return np.column_stack(
            [self.fields["Momentum_x"], self.fields["Momentum_y"], self.fields["Momentum_z"]]
        )

    @property
    def has_primitives(self) -> bool:
        """Whether the solver wrote pressure, temperature and Mach itself.

        True for a volume file, false for a restart. It is the difference
        between reading a number and reconstructing one, and above about Mach
        10 it is the difference between a right number and a wrong one.
        """
        return all(name in self.fields for name in ("Pressure", "Temperature", "Mach"))

    @property
    def velocity(self) -> _FloatArray:
        """``(N, 3)`` velocity."""
        if all(f"Velocity_{axis}" in self.fields for axis in "xyz"):
            return np.column_stack([self.fields[f"Velocity_{axis}"] for axis in "xyz"])
        return self.momentum / self.density[:, None]

    def pressure(self, gas: CaloricallyPerfect = PERFECT_AIR) -> _FloatArray:
        """Static pressure -- the solver's own where it wrote one.

        ``gas`` is used only to reconstruct pressure from the conservative
        variables of a restart. Where the field carries ``Pressure`` it is
        returned untouched and the gas model is not consulted, which is what
        lets one code path serve a perfect-gas run and a nonequilibrium one.
        """
        if "Pressure" in self.fields:
            return self.fields["Pressure"]
        kinetic = 0.5 * (self.momentum**2).sum(axis=1) / self.density
        return np.asarray((gas.gamma - 1.0) * (self.fields["Energy"] - kinetic))

    def temperature(self, gas: CaloricallyPerfect = PERFECT_AIR) -> _FloatArray:
        """Static temperature -- the solver's own where it wrote one.

        The primitive that degrades first and worst as speed rises, because it
        is the one dissociation buys down: a perfect-gas value at reentry
        speeds is wrong by a factor, not a few percent. Which is the whole
        argument for preferring the solver's.
        """
        if "Temperature" in self.fields:
            return self.fields["Temperature"]
        return np.asarray(self.pressure(gas) / (self.density * gas.gas_constant))

    def mach(self, gas: CaloricallyPerfect = PERFECT_AIR) -> _FloatArray:
        """Local Mach number -- the solver's own where it wrote one."""
        if "Mach" in self.fields:
            return self.fields["Mach"]
        speed = np.linalg.norm(self.velocity, axis=1)
        sound = np.sqrt(gas.gamma * gas.gas_constant * self.temperature(gas))
        return np.asarray(speed / sound)


@dataclass(frozen=True)
class Mesh:
    """An SU2 volume mesh: nodes, tetrahedra, and the boundary markers.

    Prisms are decomposed into tetrahedra on read (see :func:`read_su2_mesh`),
    so ``tetrahedra`` is the whole volume however the mesh was generated and
    everything downstream has one element type to handle.
    """

    points: _FloatArray
    """``(N, 3)`` node coordinates."""

    tetrahedra: _IntArray
    """``(M, 4)`` node indices."""

    markers: dict[str, _IntArray]
    """Boundary triangles per marker tag, ``(K, 3)`` node indices."""

    @property
    def size(self) -> int:
        """Number of nodes."""
        return int(self.points.shape[0])

    def marker_points(self, *tags: str) -> _FloatArray:
        """Coordinates of every node used by the named markers.

        Convenient for drawing a body outline without carrying the
        triangulation around.
        """
        used = np.unique(np.concatenate([self.markers[tag].ravel() for tag in tags]))
        return np.asarray(self.points[used], dtype=np.float64)


@dataclass(frozen=True)
class _Cut:
    """What a plane cut of any element type has in common.

    Each vertex of a cut lies on one mesh edge at a known parameter, so the
    cut is stored as that -- edges and weights -- rather than as coordinates.
    Any number of fields then cut along the same topology for the cost of a
    lerp, and they agree about the geometry because they share it, rather
    than because several independent computations rounded the same way.
    """

    edges: _IntArray
    """``(V, 2)`` node indices whose connecting edge each vertex lies on."""

    weights: _FloatArray
    """``(V,)`` parameter along the edge, ``0`` at ``edges[:, 0]``."""

    origin: _FloatArray
    """A point on the cutting plane."""

    normal: _FloatArray
    """Unit normal of the cutting plane."""

    basis: _FloatArray
    """``(2, 3)`` in-plane axes, ``u`` then ``v``; see :func:`plane_cut`."""

    @property
    def size(self) -> int:
        """Number of cut vertices."""
        return int(self.edges.shape[0])

    def interpolate(self, values: _FloatArray) -> _FloatArray:
        """Cut a nodal field, linearly along each intersected edge.

        Accepts ``(N,)`` or ``(N, k)``; returns ``(V,)`` or ``(V, k)``.
        """
        values = np.asarray(values, dtype=np.float64)
        weights = self.weights if values.ndim == 1 else self.weights[:, None]
        return _lerp(values[self.edges[:, 0]], values[self.edges[:, 1]], weights)

    def coordinates(self, points: _FloatArray) -> _FloatArray:
        """``(V, 2)`` in-plane coordinates, measured from :attr:`origin`.

        For the meridian cut these cases want -- origin on the axis, normal
        along :math:`\\hat{y}` -- this returns world :math:`x` and :math:`z`
        unchanged, so an axis label reads as a physical station rather than
        an offset from somewhere.
        """
        return np.asarray((self.interpolate(points) - self.origin) @ self.basis.T)


@dataclass(frozen=True)
class PlaneCut(_Cut):
    """A plane cut through the volume: a triangulated cross-section."""

    triangles: _IntArray = field(default_factory=lambda: np.empty((0, 3), dtype=np.int64))
    """``(T, 3)`` indices into the cut's own vertices."""


@dataclass(frozen=True)
class SurfaceCut(_Cut):
    """A plane cut through boundary markers: the body's profile in that plane.

    The outline of a body is not the scatter of its surface nodes near the
    plane -- that is a band whose width is the local cell size, and it reads
    as a fuzzy body rather than a sharp one. It is the exact intersection of
    the surface triangles with the plane, which is what this is.
    """

    segments: _IntArray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int64))
    """``(S, 2)`` indices into the cut's own vertices."""


def read_restart(path: Path | str) -> VolumeField:
    """Read an SU2 binary restart into a :class:`VolumeField`.

    The header is five 32-bit integers -- magic, field count, node count, and
    two unused -- followed by one fixed-width name per field and then the
    values, node-major. Anything SU2 appends after the values (its restart
    metadata) is ignored; it is not part of the solution.
    """
    raw = Path(path).read_bytes()
    if len(raw) < 20:
        raise ValueError(f"{path} is too short to be an SU2 restart")
    header = np.frombuffer(raw[:20], dtype=np.int32)
    if int(header[0]) != _RESTART_MAGIC:
        raise ValueError(
            f"{path} does not begin with the SU2 restart magic number "
            f"{_RESTART_MAGIC} (found {int(header[0])}); an ASCII restart or a "
            "file from another solver will land here"
        )
    count, points = int(header[1]), int(header[2])
    offset = 20
    names = [
        raw[offset + _NAME_WIDTH * i : offset + _NAME_WIDTH * (i + 1)].split(b"\x00")[0].decode()
        for i in range(count)
    ]
    offset += _NAME_WIDTH * count
    expected = count * points * 8
    if len(raw) - offset < expected:
        raise ValueError(
            f"{path} declares {count} fields over {points} nodes but holds "
            f"only {len(raw) - offset} bytes of values, not {expected}"
        )
    values = np.frombuffer(raw[offset : offset + expected], dtype=np.float64).reshape(points, count)
    named = {name: np.ascontiguousarray(values[:, i]) for i, name in enumerate(names)}
    missing = [axis for axis in ("x", "y", "z") if axis not in named]
    if missing:
        raise ValueError(f"{path} lacks coordinate fields {missing}")
    coordinates = np.column_stack([named.pop("x"), named.pop("y"), named.pop("z")])
    return VolumeField(points=coordinates, fields=named)


def _prisms_to_tetrahedra(prisms: _IntArray) -> _IntArray:
    """Split each prism into three tetrahedra.

    Fanned from node 0 over the two faces that do not contain it: the opposite
    triangle ``(3, 4, 5)``, and the quadrilateral ``(1, 2, 5, 4)`` cut along
    ``1--5``. Fanning from a single vertex is what makes the three tetrahedra
    fill the prism exactly, with no sliver and no overlap -- a property the
    tests check by summing volumes rather than by inspection.
    """
    return np.concatenate(
        [
            prisms[:, [0, 3, 4, 5]],
            prisms[:, [0, 1, 2, 5]],
            prisms[:, [0, 1, 5, 4]],
        ]
    )


def _section(tokens: list[str], name: str, start: int) -> tuple[int, int]:
    """Locate ``NAME= value``, returning ``(value, index just past it)``."""
    if tokens[start] != f"{name}=":
        raise ValueError(f"expected {name}= at token {start}, found {tokens[start]!r}")
    return int(tokens[start + 1]), start + 2


def _read_elements(block: _IntArray, count: int) -> tuple[_IntArray, _IntArray]:
    """Split an element block into tetrahedra and prisms.

    SU2 writes an optional trailing element index, and whether it is there is
    not recorded anywhere in the file. It is inferred from the arithmetic: a
    block of one element type divides evenly by its stride, and the type
    column then reads as a single repeated value. That inference is checked,
    not assumed -- if the type column is not constant the block is mixed and
    is walked element by element instead, which is slower and only the
    boundary-layer meshes need it.
    """
    if block.size and int(block[0]) not in (_TETRA, _PRISM):
        raise ValueError(
            f"element type {int(block[0])} is neither a tetrahedron ({_TETRA}) "
            f"nor a prism ({_PRISM}); no element type may be skipped, because a "
            "dropped element is a hole the picture renders as a void"
        )
    for nodes, kind in ((4, _TETRA), (6, _PRISM)):
        for stride in (nodes + 2, nodes + 1):
            if block.size == count * stride and np.all(block[::stride] == kind):
                cells = block.reshape(count, stride)[:, 1 : nodes + 1]
                return (
                    (cells, np.empty((0, 6), dtype=np.int64))
                    if kind == _TETRA
                    else (
                        np.empty((0, 4), dtype=np.int64),
                        cells,
                    )
                )

    # Mixed element types: stride varies per element, so walk it.
    trailing = 1 if block.size == _mixed_span(block, count, 1) else 0
    if trailing == 0 and block.size != _mixed_span(block, count, 0):
        raise ValueError(
            f"element block holds {block.size} integers, which is not {count} "
            "elements of any supported type"
        )
    tetrahedra: list[_IntArray] = []
    prisms: list[_IntArray] = []
    cursor = 0
    for _ in range(count):
        kind = int(block[cursor])
        nodes = 4 if kind == _TETRA else 6 if kind == _PRISM else -1
        if nodes < 0:
            raise ValueError(
                f"element type {kind} is neither a tetrahedron ({_TETRA}) nor "
                f"a prism ({_PRISM}); no element type may be skipped, because a "
                "dropped element is a hole the picture renders as a void"
            )
        (tetrahedra if kind == _TETRA else prisms).append(block[cursor + 1 : cursor + 1 + nodes])
        cursor += 1 + nodes + trailing
    return (
        np.vstack(tetrahedra) if tetrahedra else np.empty((0, 4), dtype=np.int64),
        np.vstack(prisms) if prisms else np.empty((0, 6), dtype=np.int64),
    )


def _mixed_span(block: _IntArray, count: int, trailing: int) -> int:
    """Total integers a mixed block would occupy for a given trailing width."""
    cursor = 0
    for _ in range(count):
        if cursor >= block.size:
            return -1
        kind = int(block[cursor])
        nodes = 4 if kind == _TETRA else 6 if kind == _PRISM else -1
        if nodes < 0:
            return -1
        cursor += 1 + nodes + trailing
    return cursor


def read_su2_mesh(path: Path | str) -> Mesh:
    """Read an SU2 ASCII mesh: nodes, volume elements, boundary markers.

    Handles the two volume element types these cases produce -- tetrahedra
    from the isotropic domains, prisms from the boundary-layer extrusion --
    and triangular boundary faces. Prisms are split into tetrahedra here, so
    callers see one element type regardless of how the mesh was built.

    Sections are located by their headers and then reshaped wholesale rather
    than parsed element by element. The largest mesh in this suite is 1.7
    million elements, and a per-element Python loop over it costs minutes for
    a file that is read every time a figure is drawn.

    Repeated ``MARKER_TAG`` entries are merged. The extruded meshes here write
    ``vehicle_base`` twice, once with no faces, and treating the second as a
    replacement would discard the real ones.
    """
    path = Path(path)
    tokens = path.read_text().split()

    dimension, cursor = _section(tokens, "NDIME", 0)
    if dimension != 3:
        raise ValueError(f"{path} is {dimension}-dimensional; this reader is three-dimensional")

    element_count, cursor = _section(tokens, "NELEM", cursor)
    point_header = tokens.index("NPOIN=", cursor)
    tetrahedra, prisms = _read_elements(
        np.array(tokens[cursor:point_header], dtype=np.int64), element_count
    )

    point_count, cursor = _section(tokens, "NPOIN", point_header)
    marker_header = tokens.index("NMARK=", cursor)
    block = np.array(tokens[cursor:marker_header], dtype=np.float64)
    if block.size % point_count:
        raise ValueError(f"{path}: {block.size} numbers do not divide into {point_count} nodes")
    coordinates = np.ascontiguousarray(block.reshape(point_count, block.size // point_count)[:, :3])

    marker_count, cursor = _section(tokens, "NMARK", marker_header)
    markers: dict[str, list[_IntArray]] = {}
    for _ in range(marker_count):
        if tokens[cursor] != "MARKER_TAG=":
            raise ValueError(f"{path}: expected MARKER_TAG= at token {cursor}")
        tag = tokens[cursor + 1]
        faces, cursor = _section(tokens, "MARKER_ELEMS", cursor + 2)
        table = np.array(tokens[cursor : cursor + faces * 4], dtype=np.int64).reshape(faces, 4)
        if faces and not np.all(table[:, 0] == _TRIANGLE):
            raise ValueError(f"{path}: marker {tag!r} holds a face that is not a triangle")
        markers.setdefault(tag, []).append(table[:, 1:])
        cursor += faces * 4

    volume = [tetrahedra] if tetrahedra.size else []
    if prisms.size:
        volume.append(_prisms_to_tetrahedra(prisms))
    if not volume:
        raise ValueError(f"{path} contains no volume elements")

    return Mesh(
        points=coordinates,
        tetrahedra=np.concatenate(volume),
        markers={tag: np.vstack(faces) for tag, faces in markers.items()},
    )


def read_case(mesh: Path | str, restart: Path | str) -> tuple[Mesh, VolumeField]:
    """Read a mesh and its restart, refusing a pairing that cannot belong together.

    The restart carries no reference to the mesh that produced it, so the node
    counts matching is the only evidence available that these two files
    describe the same discretisation. It is weak evidence, but a mismatch is
    conclusive, and the failure it catches -- fields indexed against the wrong
    connectivity -- produces a picture that looks like a flow rather than an
    error.
    """
    grid = read_su2_mesh(mesh)
    field = read_restart(restart)
    if grid.size != field.size:
        raise ValueError(
            f"{mesh} has {grid.size} nodes but {restart} has {field.size}; "
            "these files are not from the same case"
        )
    return grid, field


def _plane_basis(normal: _FloatArray) -> _FloatArray:
    """In-plane axes for a plane of the given normal.

    ``u`` is the world axis least aligned with the normal, projected into the
    plane; ``v = u x n``. For the meridian plane of these cases -- normal
    :math:`\\hat{y}` -- that gives ``u`` along :math:`\\hat{x}` and ``v`` along
    :math:`\\hat{z}`, so a cut plots with the body axis to the right and up
    where up is, which is the only reason to prefer one convention here.
    """
    axes = np.eye(3)
    reference = axes[int(np.argmin(np.abs(axes @ normal)))]
    u = reference - (reference @ normal) * normal
    u /= np.linalg.norm(u)
    return np.vstack([u, np.cross(u, normal)])


# The six edges of a tetrahedron, as node-pair indices into its connectivity.
_TET_EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=np.int64)

# For each of the 16 sign patterns of the four nodes (bit i set = node i is on
# the positive side), which edges the surface crosses, in an order that walks
# the cut polygon. -1 pads the entries that cut a triangle rather than a quad.
_CUT_TABLE = np.array(
    [
        [-1, -1, -1, -1],  # 0000 no crossing
        [0, 1, 2, -1],  # 1000
        [0, 3, 4, -1],  # 0100
        [1, 2, 4, 3],  # 1100
        [1, 3, 5, -1],  # 0010
        [0, 2, 5, 3],  # 1010
        [0, 4, 5, 1],  # 0110
        [2, 4, 5, -1],  # 1110
        [2, 4, 5, -1],  # 0001
        [0, 4, 5, 1],  # 1001
        [0, 2, 5, 3],  # 0101
        [1, 3, 5, -1],  # 1101
        [1, 2, 4, 3],  # 0011
        [0, 3, 4, -1],  # 1011
        [0, 1, 2, -1],  # 0111
        [-1, -1, -1, -1],  # 1111 no crossing
    ],
    dtype=np.int64,
)


#: A node this close to the cutting plane, relative to the mesh's own size, is
#: taken to be exactly on it. The gap this sits in is enormous and measured,
#: not guessed: on the Mach 8 anchor mesh the symmetry-plane nodes miss the
#: meridian plane by 4e-16 while the shortest tetrahedron edge is 1.5e-3 --
#: nine orders of separation between rounding and the smallest real feature.
_PLANE_TOLERANCE = 1e-12


def _lerp(low: _FloatArray, high: _FloatArray, weights: _FloatArray) -> _FloatArray:
    """Interpolate, returning the endpoint verbatim at a weight of 0 or 1.

    ``low + w * (high - low)`` is not ``low`` when ``w`` is zero, once the two
    endpoints differ by less than their own rounding: the subtraction cancels
    and the addition puts back a slightly different number. On the sphere-cone
    nose that is the difference between :math:`-8.071\times10^{-17}` and
    :math:`-8.066\times10^{-17}` for what is one mesh node reached along two
    edges, and it is enough to leave two vertices where there is one point.

    A vertex whose weight is exactly 0 or 1 *is* a mesh node -- that is what
    :func:`_signed_distance` snapping the on-plane nodes arranges -- so
    returning the node itself is both exact and what was meant.
    """
    interpolated = low + weights * (high - low)
    at_low = weights == 0.0
    at_high = weights == 1.0
    if at_low.any():
        interpolated = np.where(at_low, low, interpolated)
    if at_high.any():
        interpolated = np.where(at_high, high, interpolated)
    return np.asarray(interpolated, dtype=np.float64)


def _signed_distance(points: _FloatArray, origin: _FloatArray, normal: _FloatArray) -> _FloatArray:
    """Distance from each node to the plane, with on-plane nodes snapped to zero.

    The snap is what makes the cut's vertices merge exactly later, and it
    fixes a cause rather than a symptom. A node that lies on the plane by
    construction -- every symmetry-plane node, under the meridian cut every
    figure here takes -- lands at a distance of order 1e-16 instead of zero.
    Each edge leading out of it then crosses at a parameter of order 1e-17
    rather than at 0, so the several edges that should all produce that one
    node produce several points scattered within 1e-12 of it. They are the
    same point and no exact comparison says so, which leaves a triangulation
    Matplotlib's triangle finder rejects.

    Snapping the distance makes those parameters exactly zero, the positions
    exactly the node, and the merge exact -- with no welding tolerance
    anywhere downstream, and so no possibility of merging two features that
    are merely close.
    """
    distance = (points - origin) @ normal
    scale = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    if scale > 0.0:
        distance[np.abs(distance) <= _PLANE_TOLERANCE * scale] = 0.0
    return np.asarray(distance, dtype=np.float64)


_Point = NDArray[np.float64] | tuple[float, float, float]


def plane_cut(mesh: Mesh, origin: _Point, normal: _Point) -> PlaneCut:
    """Cut a tetrahedral mesh with a plane, returning the cut's topology.

    Marching tetrahedra: each element's four nodes are classified by the sign
    of their signed distance to the plane, and the resulting polygon -- a
    triangle when one node is separated from three, a quadrilateral when the
    split is two and two -- is triangulated. Because the tetrahedron is the
    simplex, the linear interpolant is exact on it and the cut of a linear
    field is that field restricted to the plane, with no discretisation of its
    own. The tests rest on exactly that.

    Nodes lying on the plane are counted as negative, which can produce
    zero-area triangles; those are dropped rather than perturbed, because
    moving the plane to avoid a degeneracy moves the picture.
    """
    origin = np.asarray(origin, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    length = np.linalg.norm(normal)
    if length == 0.0:
        raise ValueError("the cutting plane's normal is the zero vector")
    normal = normal / length

    cells = mesh.tetrahedra
    distance = _signed_distance(mesh.points, origin, normal)
    signs = distance[cells] > 0.0
    pattern = signs @ np.array([1, 2, 4, 8], dtype=np.int64)

    crossing = (pattern != 0) & (pattern != 15)
    cells = cells[crossing]
    if cells.size == 0:
        empty_edges = np.empty((0, 2), dtype=np.int64)
        return PlaneCut(
            edges=empty_edges,
            weights=np.empty(0, dtype=np.float64),
            origin=origin,
            normal=normal,
            basis=_plane_basis(normal),
            triangles=np.empty((0, 3), dtype=np.int64),
        )
    table = _CUT_TABLE[pattern[crossing]]

    # Every crossing cell contributes at least three vertices; a quad
    # contributes a fourth. Build all four slots, then discard the padding.
    local = _TET_EDGES[table.clip(min=0)]  # (C, 4, 2) local node indices
    node_pairs = np.take_along_axis(cells[:, None, :], local, axis=2)  # (C, 4, 2) global

    valid = table >= 0
    vertex_index = np.full(table.shape, -1, dtype=np.int64)
    vertex_index[valid] = np.arange(int(valid.sum()))
    edges = node_pairs[valid]

    # Triangle 0-1-2 always exists; 0-2-3 exists only for the quad cases.
    triangles = [vertex_index[:, [0, 1, 2]]]
    quad = valid[:, 3]
    if quad.any():
        triangles.append(vertex_index[quad][:, [0, 2, 3]])
    edges, lerp, remap = _merge_shared_edges(edges, distance, mesh.points)
    cut = PlaneCut(
        edges=edges,
        weights=lerp,
        origin=origin,
        normal=normal,
        basis=_plane_basis(normal),
        triangles=remap[np.concatenate(triangles)],
    )
    return _drop_degenerate(cut, mesh.points)


def _merge_shared_edges(
    edges: _IntArray, distance: _FloatArray, points: _FloatArray
) -> tuple[_IntArray, _FloatArray, _IntArray]:
    """Collapse coincident cut vertices, so the cut is a valid triangulation.

    Marching tetrahedra emits a vertex per cell, so the same point arrives
    many times over. Filled contours tolerate that; Matplotlib's triangle
    finder does not -- it rejects the triangulation outright, which is why the
    duplicates stayed invisible until a streamline was asked for on a cut that
    drew perfectly.

    Two coincidences have to be caught, and each needs a different remedy.

    An interior mesh **edge** is shared by every cell around it. Those cells
    do not agree on which end of it comes first, and computing the crossing
    parameter in each cell's own orientation and then flipping it does not
    give back the same number: :math:`1 - t'` and :math:`t` are equal in
    arithmetic and not in floating point, and on the anchor mesh they differ
    in the thirteenth digit -- enough that an exact comparison sees two
    points. So the parameter is not carried in from the cells at all. Edges
    are put in a canonical low-to-high order, reduced to the distinct ones,
    and the parameter computed once from that orientation, which makes every
    occurrence identical by construction rather than by luck.

    A **node lying exactly on the plane** puts every edge leading out of it at
    that node. Those are different edges, so the canonical-edge step cannot
    merge them; the key for that has to be the position. It is exact, because
    :func:`_signed_distance` has already snapped the node's distance to zero,
    making each parameter exactly zero and each position exactly the node.
    """
    ordered = np.where((edges[:, 0] > edges[:, 1])[:, None], edges[:, ::-1], edges)
    distinct, inverse = np.unique(ordered, axis=0, return_inverse=True)

    low, high = distance[distinct[:, 0]], distance[distinct[:, 1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(low != high, low / (low - high), 0.0)

    position = _lerp(points[distinct[:, 0]], points[distinct[:, 1]], weights[:, None])
    _, first, folded = np.unique(position, axis=0, return_index=True, return_inverse=True)
    return (
        distinct[first],
        weights[first],
        folded.ravel()[inverse.ravel()].astype(np.int64),
    )


def _drop_degenerate(cut: PlaneCut, points: _FloatArray) -> PlaneCut:
    """Remove zero-area triangles left by nodes sitting exactly on the plane.

    Filtered rather than perturbed: nudging the plane to dodge a degeneracy
    moves the picture, and a cut that silently sits somewhere other than where
    it was asked for is worse than a few discarded triangles. The vertices
    those triangles used are left in place -- unreferenced, harmless, and
    cheaper than reindexing.
    """
    planar = cut.coordinates(points)
    corners = planar[cut.triangles]
    spans = corners[:, 1:, :] - corners[:, :1, :]
    area = 0.5 * np.abs(spans[:, 0, 0] * spans[:, 1, 1] - spans[:, 0, 1] * spans[:, 1, 0])
    keep = area > 0.0
    if bool(keep.all()):
        return cut
    return PlaneCut(
        edges=cut.edges,
        weights=cut.weights,
        origin=cut.origin,
        normal=cut.normal,
        basis=cut.basis,
        triangles=cut.triangles[keep],
    )


def node_gradient(mesh: Mesh, values: _FloatArray) -> _FloatArray:
    """Volume-weighted nodal gradient of a scalar field.

    On a tetrahedron the linear basis has a single constant gradient, obtained
    by solving the 3x3 system of edge vectors against edge value differences.
    That is exact for a linear field, which makes it the right primitive to
    build a Schlieren image on: what appears in the picture is the solver's
    own solution differentiated, not a second difference stencil laid over it.

    Nodal values come from averaging the elements around each node, weighted
    by element volume so that the boundary layer's flat slivers do not
    outvote the cells that carry the shock.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] != mesh.size:
        raise ValueError(f"field has {values.shape[0]} values for {mesh.size} nodes")

    cells = mesh.tetrahedra
    corners = mesh.points[cells]
    edges = corners[:, 1:, :] - corners[:, :1, :]  # (M, 3, 3)
    differences = values[cells][:, 1:] - values[cells][:, :1]  # (M, 3)

    determinant = np.linalg.det(edges)
    volume = np.abs(determinant) / 6.0
    usable = np.abs(determinant) > 0.0
    gradient = np.zeros((cells.shape[0], 3), dtype=np.float64)
    gradient[usable] = np.linalg.solve(edges[usable], differences[usable][..., None])[..., 0]

    weighted = gradient * volume[:, None]
    total = np.zeros(mesh.size, dtype=np.float64)
    accumulated = np.zeros((mesh.size, 3), dtype=np.float64)
    flat = cells.ravel()
    np.add.at(total, flat, np.repeat(volume, 4))
    for axis in range(3):
        np.add.at(accumulated[:, axis], flat, np.repeat(weighted[:, axis], 4))
    nonzero = total > 0.0
    accumulated[nonzero] /= total[nonzero, None]
    return accumulated


# For each sign pattern of a triangle's three nodes, which of its edges
# ``(0,1)``, ``(0,2)``, ``(1,2)`` the plane crosses. A plane meets a triangle
# in a segment or not at all, so every entry is a pair or empty.
_TRI_EDGES = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
_SEGMENT_TABLE = np.array(
    [[-1, -1], [0, 1], [0, 2], [1, 2], [1, 2], [0, 2], [0, 1], [-1, -1]], dtype=np.int64
)


def surface_cut(mesh: Mesh, tags: Sequence[str], origin: _Point, normal: _Point) -> SurfaceCut:
    """Intersect boundary markers with a plane, giving the body's profile.

    Marching triangles, and the same edge-and-weight representation the volume
    cut uses -- so a surface field can be carried along the outline and
    plotted against station, which is what makes a wall-pressure curve and the
    field behind it the same picture rather than two.

    Raises if a tag is not in the mesh: a misspelled marker that silently
    produced an empty outline would look like a body with no surface.
    """
    origin = np.asarray(origin, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    length = np.linalg.norm(normal)
    if length == 0.0:
        raise ValueError("the cutting plane's normal is the zero vector")
    normal = normal / length

    missing = [tag for tag in tags if tag not in mesh.markers]
    if missing:
        raise ValueError(f"mesh has no marker(s) {missing}; it has {sorted(mesh.markers)}")
    faces = np.vstack([mesh.markers[tag] for tag in tags]) if tags else np.empty((0, 3), np.int64)

    distance = _signed_distance(mesh.points, origin, normal)
    pattern = (distance[faces] > 0.0) @ np.array([1, 2, 4], dtype=np.int64)
    crossing = (pattern != 0) & (pattern != 7)
    faces, table = faces[crossing], _SEGMENT_TABLE[pattern[crossing]]

    basis = _plane_basis(normal)
    if faces.size == 0:
        return SurfaceCut(
            edges=np.empty((0, 2), dtype=np.int64),
            weights=np.empty(0, dtype=np.float64),
            origin=origin,
            normal=normal,
            basis=basis,
            segments=np.empty((0, 2), dtype=np.int64),
        )

    local = _TRI_EDGES[table]  # (C, 2, 2) local node indices
    node_pairs = np.take_along_axis(faces[:, None, :], local, axis=2)  # (C, 2, 2) global
    d0 = distance[node_pairs[..., 0]]
    d1 = distance[node_pairs[..., 1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(d0 != d1, d0 / (d0 - d1), 0.0)

    return SurfaceCut(
        edges=node_pairs.reshape(-1, 2),
        weights=weights.ravel(),
        origin=origin,
        normal=normal,
        basis=basis,
        segments=np.arange(2 * faces.shape[0], dtype=np.int64).reshape(-1, 2),
    )


def read_volume(path: Path | str) -> VolumeField:
    """Read an SU2 Tecplot-ASCII volume file (``volume_flow.dat``).

    The reason to want one. A restart carries conservative variables and
    nothing else, so pressure, temperature and Mach have to be reconstructed
    from it -- and reconstructing them needs a gas model, which is exactly the
    thing that stops being simple above about Mach 10. A volume file carries
    ``Pressure``, ``Temperature`` and ``Mach`` as the solver computed them,
    under whatever gas model was actually running: perfect gas, equilibrium,
    or five-species nonequilibrium. Reading those is how the post-processing
    stops having an opinion about the thermodynamics.

    Only the nodal block is read. SU2 writes the connectivity after it as
    ``FEBRICK`` cells -- every element padded to eight nodes, tetrahedra
    included -- which is a lossy shape to recover a mesh from and unnecessary
    here, because the ``.su2`` file already has the real connectivity and its
    node order is the same. That last property is checked by
    :func:`read_case` rather than assumed.

    Written to be tolerant of the variable list rather than positional: SU2's
    ``VOLUME_OUTPUT`` is configurable, so the columns present depend on the
    case, and a reader that indexed them by position would misread a case that
    asked for one field more.
    """
    path = Path(path)
    with path.open() as handle:
        header: list[str] = []
        for line in handle:
            header.append(line)
            if "ZONE" in line.upper():
                break
        else:
            raise ValueError(f"{path} has no ZONE record; is it a Tecplot volume file?")

        text = "".join(header)
        names = re.findall(r'"([^"]*)"', text[text.upper().index("VARIABLES") :])
        if not names:
            raise ValueError(f"{path} declares no VARIABLES")
        match = re.search(r"NODES\s*=\s*(\d+)", text, re.IGNORECASE)
        if match is None:
            raise ValueError(f"{path} does not say how many nodes its zone has")
        count = int(match.group(1))
        if "POINT" not in text.upper():
            raise ValueError(
                f"{path} is not DATAPACKING=POINT; this reader does not handle block packing"
            )

        rows = np.empty((count, len(names)), dtype=np.float64)
        for index in range(count):
            values = handle.readline().split()
            if len(values) != len(names):
                raise ValueError(
                    f"{path}: node {index} has {len(values)} values for {len(names)} variables"
                )
            rows[index] = values

    named = {name: np.ascontiguousarray(rows[:, i]) for i, name in enumerate(names)}
    missing = [axis for axis in ("x", "y", "z") if axis not in named]
    if missing:
        raise ValueError(f"{path} lacks coordinate variables {missing}")
    coordinates = np.column_stack([named.pop("x"), named.pop("y"), named.pop("z")])
    return VolumeField(points=coordinates, fields=named)
