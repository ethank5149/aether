"""Exact solids from the same parameterisation the panel generators use.

A triangulated body is a *sample* of a shape, and once sampled the shape is
gone. Refining a CFD mesh around a fixed triangulation converges to the
faceted body rather than to the real one, which is worse than it sounds: the
Richardson extrapolation in :mod:`aether.aerodynamics.cfd.solver` assumes the
limit it is extrapolating toward is the true geometry, and it is not. On the
generic lifting body in this package the frozen faceting costs 2.7 % of the
volume, and no amount of mesh refinement removes any of it.

This module keeps the shape. A :class:`Loft` or :class:`Revolve` is a
*description* — the analytic cross-sections or meridian profile the generator
already computes — and OpenCASCADE builds a real B-rep solid from it. Meshing
then places nodes **on** that surface at whatever density is asked for, so
refinement converges to the body rather than to a snapshot of it. The same
description exports to STEP, which is what a tool outside this package will
want to read.

Why the description is data and not a live handle
-------------------------------------------------

gmsh is a process-global singleton with an ``initialize``/``finalize``
lifecycle, so a "solid" that stayed live between calls would be a handle into
mutable global state that any other gmsh user in the process could invalidate.
Every operation here therefore opens its own session, builds the solid from
the description, does one job and closes. Building costs milliseconds — the
lofts in this package are tens of section curves — and the cost of the
alternative is a class of bug that only appears when two callers interleave.

The consequence for a sweep is that the *mesh* is what gets cached, not the
solid: build the surface once with :func:`surface_mesh`, then hand that mesh to
the volume mesher as many times as the sweep needs.

Sharp edges
-----------

``smooth`` decides whether a section is a closed B-spline or a closed
polyline, and it is not a quality setting. A waverider's leading edge is a
crease, and the entire point of the shape is that the shock attaches along it;
fitting a spline through that section rounds the crease and quietly turns a
waverider into a lifting body with a blunt edge. Shapes whose cross-sections
are genuinely piecewise-linear — the caret is a triangle at every station —
are built as polylines and are then *exact*, not approximated.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

from aether.geometry.backend import start_gmsh
from aether.geometry.edges import Contour, Segment
from aether.geometry.mesh import VehicleMesh, _weld

__all__ = [
    "Loft",
    "Revolve",
    "SolidProperties",
    "solid_properties",
    "surface_mesh",
    "write_step",
]

_FloatArray = NDArray[np.float64]
_Result = TypeVar("_Result")


@dataclass(frozen=True)
class SolidProperties:
    """Mass properties read from the exact solid, not summed over triangles.

    The triangulated equivalents converge to these as the mesh refines, and
    the gap between them is the geometric error the CFD would inherit — which
    makes comparing the two the cheapest available check that a body was
    sampled finely enough.
    """

    volume: float
    """m³."""
    surface_area: float
    """m² — the wetted area."""
    centroid: _FloatArray
    """Centre of volume (m), i.e. the centre of mass at uniform density."""
    bounds: tuple[_FloatArray, _FloatArray]

    @property
    def length(self) -> float:
        return float(self.bounds[1][0] - self.bounds[0][0])

    @property
    def diameter(self) -> float:
        """Largest transverse extent (m)."""
        low, high = self.bounds
        return float(max(high[1] - low[1], high[2] - low[2]))


def _contour_curves(gmsh: Any, contour: Contour) -> tuple[list[int], int, int]:
    """Create OCC curves for an exact outline and return their tags.

    Endpoints are shared between consecutive primitives rather than created
    twice, so the wire closes on identical vertices instead of on a pair a
    tolerance apart — which is where micro-edges come from.

    Returns ``(curves, first_point, last_point)``; the endpoint tags let an
    open outline be closed against the axis by the caller.
    """

    def point(value: _FloatArray) -> int:
        coordinates = np.zeros(3)
        coordinates[: value.shape[0]] = value
        return int(gmsh.model.occ.addPoint(*coordinates))

    primitives = contour.primitives
    count = len(primitives)
    starts = [point(item.start) for item in primitives]
    curves: list[int] = []
    last = starts[0]
    for index, item in enumerate(primitives):
        begin = starts[index]
        if index + 1 < count:
            finish = starts[index + 1]
        else:
            finish = starts[0] if contour.closed else point(item.end)
        if isinstance(item, Segment):
            curves.append(int(gmsh.model.occ.addLine(begin, finish)))
        else:
            curves.append(int(gmsh.model.occ.addCircleArc(begin, point(item.centre), finish)))
        last = finish
    return curves, starts[0], last


@dataclass(frozen=True)
class Loft:
    """A solid lofted through closed cross-sections along the body axis.

    Attributes
    ----------
    section:
        ``u -> (n, 3)`` array of points around the closed perimeter at
        fractional station ``u``, **not** repeating the first point. The
        generators in :mod:`aether.aerodynamics.panels` already compute
        exactly this; the loft consumes the same expression rather than a
        sampled copy of it.
    stations:
        Fractional stations to loft through, ascending. More sections cost
        almost nothing to build and control how faithfully the *longitudinal*
        curvature is captured — the transverse curvature is exact between
        them, because each section is.
    smooth:
        Closed B-spline through each section, or closed polyline. See the
        module note on sharp edges.
    """

    stations: _FloatArray
    section: Callable[[float], _FloatArray] | None = field(default=None, repr=False)
    section_contour: Callable[[float], Contour] | None = field(default=None, repr=False)
    """Exact outline at a station, preferred over :attr:`section` when given.

    Supplying this is what keeps the loft's faces proportional to the *shape*
    rather than to the sampling: a filleted triangular section is six
    primitives, so the solid has six longitudinal faces, against the hundreds
    of chord-edges a sampled outline produces. See :class:`Contour`.
    """
    smooth: bool = True
    ruled: bool = False
    """Rule the surface between adjacent sections instead of lofting through them.

    Independent of :attr:`smooth`, which decides how each *section* is built.
    Conflating the two is a trap: a caret waverider needs segment-built
    sections, so that its ridge and leading edges stay sharp corners, and a
    *through-lofted* longitudinal surface. Ruling it instead makes every
    section-to-section strip its own face, which pins the mesh to the section
    spacing — 143 mm along the body against 3 mm across a filleted leading
    edge, a median edge-length ratio of 71:1 and 1625 of 2916 faces below a
    quality of 0.1. Nothing downstream could fix that: mesh size fields,
    ``size_min`` and the 2D algorithm choice all had *no effect at all* on the
    result, because the faces themselves were the constraint.
    """
    name: str = "body"

    def __post_init__(self) -> None:
        u = np.asarray(self.stations, dtype=np.float64)
        if u.ndim != 1 or u.size < 2:
            msg = f"need at least two stations to loft through, got {u.shape}"
            raise ValueError(msg)
        if np.any(np.diff(u) <= 0.0):
            msg = "stations must be strictly increasing"
            raise ValueError(msg)
        if self.section is None and self.section_contour is None:
            msg = (
                "a loft needs its cross-sections: give section_contour for an "
                "exact outline, or section for a sampled one"
            )
            raise ValueError(msg)

    def _build(self, gmsh: Any) -> int:
        wires = []
        sampled = self.section
        for u in np.asarray(self.stations, dtype=np.float64):
            if self.section_contour is not None:
                wires.append(
                    gmsh.model.occ.addWire(_contour_curves(gmsh, self.section_contour(float(u)))[0])
                )
                continue
            assert sampled is not None  # guaranteed by __post_init__
            points = np.asarray(sampled(float(u)), dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
                msg = f"section({u:g}) must return (n >= 3, 3) points, got {points.shape}"
                raise ValueError(msg)
            tags = [gmsh.model.occ.addPoint(*p) for p in points]
            closed = [*tags, tags[0]]
            if self.smooth:
                curves = [gmsh.model.occ.addBSpline(closed)]
            else:
                # One straight edge per side, rather than a polyline primitive:
                # the OCC kernel has no polyline, and building the segments
                # explicitly is also what keeps each corner a real vertex, which
                # is what stops the mesher from smoothing across it.
                curves = [gmsh.model.occ.addLine(a, b) for a, b in itertools.pairwise(closed)]
            wires.append(gmsh.model.occ.addWire(curves))
        # makeSolid caps both ends with planar faces. The aft cap *is* the base
        # disc and is right; the forward one is a small flat facet whose size is
        # the first station's section, so a blunt-nosed body wants its first
        # station close to zero and a pointed one is better built by Revolve.
        made = gmsh.model.occ.addThruSections(wires, makeSolid=True, makeRuled=self.ruled)
        gmsh.model.occ.synchronize()
        volumes = [tag for dim, tag in made if dim == 3]
        if not volumes:
            msg = f"lofting {self.name} produced no solid"
            raise RuntimeError(msg)
        return int(volumes[0])


@dataclass(frozen=True)
class Revolve:
    """A body of revolution swept from its meridian profile.

    Exact in the round — the surface is a true surface of revolution, not a
    polygon swept through a finite number of azimuths — which is why a
    sphere-cone or a multiconic should be built this way rather than lofted.
    The profile itself is a polyline in the meridian plane, so axial
    resolution is whatever the profile carries.
    """

    station: _FloatArray
    radius: _FloatArray
    contour: Contour | None = None
    """Exact meridian outline, preferred over the sampled ``station``/``radius``.

    A spherically blunted cone's meridian is three primitives — cap arc, cone
    line, shoulder arc — and sampling it into 110 points gives the revolved
    solid 107 faces instead of 4. See :class:`Contour`.
    """
    name: str = "body"

    def __post_init__(self) -> None:
        x = np.asarray(self.station, dtype=np.float64)
        r = np.asarray(self.radius, dtype=np.float64)
        if x.shape != r.shape or x.ndim != 1 or x.size < 3:
            msg = f"profile needs matching 1-D arrays of length >= 3, got {x.shape}, {r.shape}"
            raise ValueError(msg)
        if np.any(np.diff(x) <= 0.0):
            msg = "profile stations must be strictly increasing from the nose"
            raise ValueError(msg)
        if np.any(r < 0.0):
            msg = "profile radii must be non-negative"
            raise ValueError(msg)

    def _build(self, gmsh: Any) -> int:
        x = np.asarray(self.station, dtype=np.float64)
        r = np.asarray(self.radius, dtype=np.float64)
        # A meridian outline plus its closing segments on the axis: revolve the
        # closed face, not the open curve, or OCC returns a shell with no inside.
        if self.contour is not None:
            segments, head, tail = _contour_curves(gmsh, self.contour)
            first = self.contour.primitives[0].start
            final = self.contour.primitives[-1].end
            x = np.array([float(first[0]), float(final[0])])
            r = np.array([float(first[1]), float(final[1])])
        else:
            points = [
                gmsh.model.occ.addPoint(float(xi), float(ri), 0.0)
                for xi, ri in zip(x, r, strict=True)
            ]
            segments = [gmsh.model.occ.addLine(a, b) for a, b in itertools.pairwise(points)]
            head, tail = points[0], points[-1]
        base = gmsh.model.occ.addPoint(float(x[-1]), 0.0, 0.0)
        nose = gmsh.model.occ.addPoint(float(x[0]), 0.0, 0.0)
        closing = [gmsh.model.occ.addLine(tail, base)] if r[-1] > 0.0 else []
        closing.append(gmsh.model.occ.addLine(base, nose))
        if r[0] > 0.0:
            closing.append(gmsh.model.occ.addLine(nose, head))
        loop = gmsh.model.occ.addCurveLoop(segments + closing)
        face = gmsh.model.occ.addPlaneSurface([loop])
        made = gmsh.model.occ.revolve([(2, face)], 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0 * np.pi)
        gmsh.model.occ.synchronize()
        volumes = [tag for dim, tag in made if dim == 3]
        if not volumes:
            msg = f"revolving {self.name} produced no solid"
            raise RuntimeError(msg)
        return int(volumes[0])


Body = Loft | Revolve


def _session(body: Body, work: Callable[[Any, int], _Result]) -> _Result:
    """Build ``body`` in a private gmsh session, run ``work``, always close.

    The ``finally`` is the whole point: gmsh's state is process-global, so a
    build that raises halfway — a self-intersecting loft, a degenerate section
    — would otherwise leave a partial model behind for the next unrelated
    caller to trip over.
    """
    gmsh = start_gmsh()

    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(body.name)
        volume = body._build(gmsh)
        return work(gmsh, volume)
    finally:
        gmsh.finalize()


def solid_properties(body: Body) -> SolidProperties:
    """Volume, wetted area, centroid and bounding box of the exact solid."""

    def read(gmsh: Any, volume: int) -> SolidProperties:
        # Bounds from a coarse tessellation, not from OCC's box. OCC bounds a
        # curved face by its control polygon, which is a *loose* enclosure: on
        # the sphere-cone here it returned a diameter of 0.8501 m for a body
        # whose meridian reaches 0.78538 m, an 8 % overestimate. That was
        # invisible while the solid was still built from sampled polylines,
        # because a body faceted into near-planar strips has a tight control
        # polygon — exact arcs are what expose it. The number feeds domain
        # extents and refinement sizes, so it has to be the real one.
        # Sized off OCC's own (loose) box, so this stays a cheap tessellation
        # for measuring a body and does not inherit the default fine sizing —
        # on the waverider, curvature-driven default sizes over a 3 mm leading
        # edge fillet turn "measure this solid" into an unbounded mesh.
        box = gmsh.model.occ.getBoundingBox(3, volume)
        diagonal = float(np.linalg.norm(np.asarray(box[3:]) - np.asarray(box[:3])))
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12.0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", diagonal / 60.0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", diagonal / 600.0)
        gmsh.model.mesh.generate(2)
        _, coordinates, _ = gmsh.model.mesh.getNodes()
        nodes = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
        low = nodes.min(axis=0)
        high = nodes.max(axis=0)
        low_x, low_y, low_z = (float(value) for value in low)
        high_x, high_y, high_z = (float(value) for value in high)
        # Area and centroid from the same tessellation, not from OCC. Asking
        # OCC to integrate over exact spherical, toroidal and BSpline faces is
        # far slower than integrating a triangulation of them: on the caret
        # waverider it had not returned after four minutes of CPU, against a
        # few seconds here. The cost was invisible while these solids were
        # sampled polylines, because integrating a few hundred planar strips
        # is trivial — it is the price of exact faces, and not one worth
        # paying for two numbers a triangulation gives to five figures.
        # Volume stays exact: ``getMass(3, ...)`` is closed-form and immediate.
        node_tags = np.asarray(gmsh.model.mesh.getNodes()[0], dtype=np.int64)
        lookup = np.zeros(int(node_tags.max()) + 1, dtype=np.int64)
        lookup[node_tags] = np.arange(node_tags.size)
        # Only the volume's own boundary. ``getElementsByType`` would return
        # every triangle in the model, and ``revolve`` leaves its generating
        # meridian face behind as a separate entity — including it inflated the
        # sphere-cone's wetted area by 13 %.
        boundary = gmsh.model.getBoundary([(3, volume)], combined=True, oriented=False)
        blocks = [
            np.asarray(block, dtype=np.int64).reshape(-1, 3)
            for _, tag in boundary
            for block in gmsh.model.mesh.getElements(2, abs(tag))[2]
        ]
        if not blocks:
            raise RuntimeError(f"{body.name} produced no boundary triangles")
        triangles = np.concatenate(blocks)
        corners = nodes[lookup[triangles]]
        first, second, third = corners[:, 0], corners[:, 1], corners[:, 2]
        cross = np.cross(second - first, third - first)
        area = float(0.5 * np.linalg.norm(cross, axis=1).sum())
        # Centroid of the enclosed solid, by the divergence theorem over the
        # closed surface rather than by averaging vertices, which would weight
        # by how finely each region happens to be meshed.
        signed = np.einsum("ij,ij->i", first, cross) / 6.0
        total = float(signed.sum())
        centre = (
            np.einsum("i,ij->j", signed, (first + second + third) / 4.0) / total
            if abs(total) > 1e-30
            else nodes.mean(axis=0)
        )
        return SolidProperties(
            volume=float(gmsh.model.occ.getMass(3, volume)),
            surface_area=area,
            centroid=np.asarray(centre, dtype=np.float64),
            bounds=(
                np.array([low_x, low_y, low_z], dtype=np.float64),
                np.array([high_x, high_y, high_z], dtype=np.float64),
            ),
        )

    return _session(body, read)


def surface_mesh(
    body: Body,
    size_max: float | None = None,
    size_min: float | None = None,
    curvature_nodes: float = 20.0,
    weld_tolerance: float = 1.0e-9,
) -> VehicleMesh:
    """Triangulate the exact surface, with nodes lying **on** it.

    This is where the geometric error actually goes away. Every node is placed
    by OCC on the true surface, so asking for a finer mesh converges to the
    body rather than to whatever faceting a generator happened to bake in —
    and the same body can be meshed coarse for a shape sweep and fine for a
    heat-flux run without being redefined.

    Parameters
    ----------
    size_max, size_min:
        Target cell sizes (m). Default to the body's diameter over 20 and over
        400 respectively, which is a reasonable starting point for a shape
        whose transverse scale is its diameter.
    curvature_nodes:
        Cells per :math:`2\\pi` of turning. This, rather than ``size_min``, is
        what resolves a nose: curvature-driven sizing puts cells where the
        surface actually bends and leaves the barrel coarse, which a uniform
        size cannot do without paying for the nose everywhere.
    """
    reference = solid_properties(body)
    diameter = reference.diameter if reference.diameter > 0.0 else reference.length
    coarse = float(size_max if size_max is not None else diameter / 20.0)
    fine = float(size_min if size_min is not None else diameter / 400.0)
    if not (0.0 < fine <= coarse):
        msg = f"need 0 < size_min <= size_max, got {fine} and {coarse}"
        raise ValueError(msg)

    def build(gmsh: Any, volume: int) -> VehicleMesh:
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", float(curvature_nodes))
        gmsh.option.setNumber("Mesh.MeshSizeMax", coarse)
        gmsh.option.setNumber("Mesh.MeshSizeMin", fine)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.model.mesh.generate(2)

        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        lookup = {
            int(tag): np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)[i]
            for i, tag in enumerate(np.asarray(tags))
        }
        # Only the surfaces that bound the solid, and only the ones with area.
        # Revolving a profile that touches its own axis — which every body of
        # revolution does, at the nose and across the base — leaves degenerate
        # faces lying on the axis. They enclose nothing, but gmsh meshes them,
        # and their elements give the axis edges a third incident face: enough
        # to make a perfectly good solid report as not closed.
        blocks = []
        for dim, tag in gmsh.model.getBoundary([(3, volume)], combined=True, oriented=False):
            if dim != 2 or gmsh.model.occ.getMass(2, abs(tag)) <= 1.0e-12:
                continue
            _, _, entity = gmsh.model.mesh.getElements(2, abs(tag))
            blocks += [np.asarray(block).reshape(-1, 3) for block in entity]
        if not blocks:
            msg = f"no surface elements on the boundary of {body.name}"
            raise RuntimeError(msg)
        connected = np.vstack(blocks)
        triangles = np.array(
            [[lookup[int(t)] for t in face] for face in connected], dtype=np.float64
        )
        # Degenerate faces dropped first. A surface of revolution touches its
        # own axis at the nose and across the base disc, and the elements that
        # land exactly on the axis have zero area: they contribute nothing to
        # any integral but they do give some edges a third incident face, which
        # is enough to make a watertight body report as neither closed nor open.
        edge_a = triangles[:, 1] - triangles[:, 0]
        edge_b = triangles[:, 2] - triangles[:, 0]
        keep = np.linalg.norm(np.cross(edge_a, edge_b), axis=1) > 1.0e-14
        triangles = triangles[keep]
        # Welded by *position*, not by gmsh's node ids. A surface of revolution
        # is periodic, and gmsh gives its seam two sets of node ids that occupy
        # the same coordinates; trusting the ids leaves a slit down the whole
        # body, which has no area, passes every check a panel integration makes,
        # and lets the volume mesher into the interior.
        vertices, faces = _weld(triangles, weld_tolerance)
        return VehicleMesh(vertices=vertices, faces=faces, name=body.name)

    mesh = _session(body, build)
    # OCC's face orientation is not guaranteed to agree with the outward
    # convention the boundary-layer extruder needs, and it is cheaper to check
    # the sign of the enclosed volume than to reason about it: the divergence
    # theorem gives a positive volume exactly when the winding is outward.
    triangles = mesh.triangles
    signed = float(
        np.einsum(
            "ij,ij->i",
            triangles[:, 0],
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        ).sum()
        / 6.0
    )
    if signed < 0.0:
        mesh = VehicleMesh(
            vertices=mesh.vertices,
            faces=np.asarray(mesh.faces)[:, [0, 2, 1]],
            name=mesh.name,
        )
    return mesh


def write_step(body: Body, path: str | Path) -> Path:
    """Write the exact solid as ISO 10303 STEP.

    The interchange format, and the reason the shape is worth keeping at all
    beyond this package: a STEP file is what a structural mesher, a CAD
    package or a collaborator's tool can read, and it carries the surface
    rather than a sampling of it.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    def export(gmsh: Any, _volume: int) -> Path:
        gmsh.write(str(target))
        return target

    return _session(body, export)
