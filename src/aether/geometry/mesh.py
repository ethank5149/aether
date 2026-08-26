"""Vehicle outer mould line from a triangle mesh.

Everything the framework knew about vehicle shape was a handful of scalars
typed into :class:`~aether_gambit.flight.simulator.FlightConfiguration`: a length, a
reference area, a nose radius, a ballistic coefficient. Each was a
*stipulation*. A mesh replaces most of them with measurements, and it feeds
the one model already built to consume a surface —
:class:`~aether.aerodynamics.panels.PanelModel`, whose inputs are exactly a
centroid, an outward unit normal and an area per panel. A triangle is that.

What a mesh can and cannot settle
---------------------------------

It settles **shape**: length, frontal area, wetted area, nose radius, the
axial station profile, and — through the panel model — force and moment
coefficients against incidence and Mach, instead of a hand-set drag area.

It does **not** settle **mass**. A surface has no density distribution, so
:meth:`VehicleMesh.mass_properties` returns what a *stated* density model
implies and nothing more; a launcher whose propellant drains from the bottom
is not a uniform solid at any moment of its flight. The volume and inertia
below are honest arithmetic on an assumption the caller supplies.

It also does not settle **closure**. A mesh exported for visualisation is
often not watertight, and volume is meaningless on an open surface: the
divergence-theorem sum still returns a number, and that number can put the
centroid outside the bounding box. :attr:`VehicleMesh.is_closed` is checked
and the mass properties refuse to run when it is false, because a silently
wrong inertia tensor is worse than none.

Axis conventions, which are the easy thing to get wrong
-------------------------------------------------------

Meshes are usually authored with the long axis along **z**.
:class:`PanelModel` puts the nose along **+x** — its
``velocity_direction`` is :math:`(\\cos\\alpha, 0, \\sin\\alpha)` — and so
does the vehicle glyph in :mod:`aether_gambit.viz.scene`. :meth:`to_body_axes`
performs that rotation explicitly rather than leaving it to whoever wires
the two together, and it is tested by checking that the nose really does end
up on +x.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from aether.aerodynamics.panels import PanelModel, SurfaceGrid

__all__ = ["VehicleMesh", "load_stl", "write_stl"]

_FloatArray = NDArray[np.float64]
_IntArray = NDArray[np.int64]


def _weld(triangles: _FloatArray, tolerance: float) -> tuple[_FloatArray, _IntArray]:
    """Merge coincident vertices and return (vertices, faces)."""
    flat = triangles.reshape(-1, 3)
    keys = np.round(flat / tolerance).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    return flat[first], inverse.reshape(-1, 3).astype(np.int64)


def load_stl(
    path: str | Path, weld_tolerance: float = 1.0e-6, name: str = ""
) -> VehicleMesh:
    """Read a binary or ASCII STL.

    STL carries no topology at all — it is a bag of triangles, each with its
    own three vertices — so coincident vertices are welded on load. Without
    that there is no edge adjacency, and without edge adjacency there is no
    closure test, no connected components and no way to tell a hole from a
    seam.

    Degenerate (zero-area) triangles are dropped, and the count is available
    on the result. This file carries three of them, which is normal for an
    export and fatal to a normal computation that divides by the area.
    """
    location = Path(path)
    data = location.read_bytes()
    if len(data) < 84:
        msg = f"{location} is too short to be an STL ({len(data)} bytes)"
        raise ValueError(msg)

    count = struct.unpack("<I", data[80:84])[0]
    if len(data) == 84 + count * 50:
        block = np.frombuffer(data[84 : 84 + count * 50], dtype=np.uint8).reshape(
            count, 50
        )
        triangles = (
            block[:, 12:48].copy().view("<f4").reshape(count, 3, 3).astype(np.float64)
        )
    else:
        text = data.decode("ascii", errors="replace").split()
        coordinates = [
            float(text[i + 1]) for i, word in enumerate(text) if word == "vertex"
        ]
        if not coordinates:
            msg = (
                f"{location} is neither a valid binary STL (header claims "
                f"{count} triangles, which needs {84 + count * 50} bytes against "
                f"{len(data)}) nor an ASCII one (no 'vertex' records)"
            )
            raise ValueError(msg)
        raw = np.array(
            [
                [float(text[i + 1]), float(text[i + 2]), float(text[i + 3])]
                for i, word in enumerate(text)
                if word == "vertex"
            ]
        )
        triangles = raw.reshape(-1, 3, 3)

    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    twice_area = np.linalg.norm(np.cross(edge_a, edge_b), axis=1)
    keep = twice_area > 1.0e-14
    dropped = int((~keep).sum())
    vertices, faces = _weld(triangles[keep], weld_tolerance)
    return VehicleMesh(
        vertices=vertices,
        faces=faces,
        name=name or location.stem,
        degenerate_dropped=dropped,
    )


@dataclass(frozen=True)
class VehicleMesh:
    """A welded triangle mesh with the quantities a vehicle model needs."""

    vertices: _FloatArray
    faces: _IntArray
    name: str = ""
    degenerate_dropped: int = 0

    def __post_init__(self) -> None:
        v = np.asarray(self.vertices, dtype=np.float64)
        f = np.asarray(self.faces, dtype=np.int64)
        if v.ndim != 2 or v.shape[1] != 3:
            msg = f"vertices must have shape (n, 3), got {v.shape}"
            raise ValueError(msg)
        if f.ndim != 2 or f.shape[1] != 3:
            msg = f"faces must have shape (m, 3), got {f.shape}"
            raise ValueError(msg)
        if f.size and (f.min() < 0 or f.max() >= v.shape[0]):
            msg = "faces index vertices outside the vertex array"
            raise ValueError(msg)

    @classmethod
    def from_surface_grid(
        cls,
        grid: SurfaceGrid,
        name: str = "",
        weld_tolerance: float = 1.0e-9,
        cap_ends: bool = True,
    ) -> VehicleMesh:
        """Close a parametric generator's vertex net into a watertight mesh.

        The generators in :mod:`aether.aerodynamics.panels` build an
        ``(n_axial, n_circ + 1, 3)`` net and hand the panels cut from it to
        impact theory, which never asks whether the body is closed — a panel
        integration sums over faces and an absent face contributes nothing.
        A mesh generator does ask, and answers wrongly if the body is open:
        gmsh reads a hole in the base as a way into the interior and fills the
        vehicle with cells.

        Two places need closing, and they are different problems:

        **The seam.** The net's last circumferential column repeats its first,
        so the quads that bridge them would be zero-width. The wrap is done by
        index instead — column ``n_circ - 1`` joins column ``0`` — and the
        duplicate column is dropped. :attr:`SurfaceGrid.seam_closed` is checked
        first, because if the generator sampled with ``endpoint=False`` the
        repeat is not a repeat and dropping it opens a slit down the body.

        **The ends.** Each end ring is a hole unless it is degenerate.
        ``cap_ends`` closes whichever are not, with a triangle fan to the ring's
        centroid, wound so the fan's normal points out of the body at that end.
        Both ends have to be tested rather than assumed: a sphere-cone's nose
        ring collapses to the tip and needs nothing, while
        :func:`~aether.aerodynamics.panels.spatular_wedge` starts its net at
        ``u = 1e-5`` on a blunt elliptical nose and leaves a small but real
        hole there. Capping only the base closes the second body and not the
        first, and the difference is invisible until gmsh meshes the inside.

        A degenerate ring's collapsed triangles are dropped by area, exactly as
        :func:`load_stl` drops them.
        """
        if not grid.seam_closed:
            msg = (
                "the surface grid's last circumferential column does not repeat "
                "its first, so it cannot be wrapped by index; the generator "
                "sampled the circumference with endpoint=False"
            )
            raise ValueError(msg)

        net = np.asarray(grid.vertices, dtype=np.float64)[:, :-1, :]
        n_circ = net.shape[1]

        j = np.arange(n_circ)
        j_next = (j + 1) % n_circ
        lower = net[:-1]
        upper = net[1:]
        # Wound azimuth-first, so each face normal is (azimuthal x axial) and
        # points radially *outward*. The opposite order is the natural one to
        # write and gives an inward-facing body: closed, correct in area, and
        # silently wrong to anything that reads the normals — which includes
        # the boundary-layer extrusion, whose prisms would grow into the
        # vehicle rather than into the flow.
        quads = np.stack(
            [lower[:, j], lower[:, j_next], upper[:, j_next], upper[:, j]], axis=2
        ).reshape(-1, 4, 3)
        triangles = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])

        if cap_ends:
            scale = float(np.max(np.abs(net))) or 1.0
            caps = []
            # The nose fan is wound opposite to the base fan: both must turn
            # their normals away from the body, and the two ends face opposite
            # directions along the axis.
            for ring, forward in ((net[0], False), (net[-1], True)):
                if float(np.max(np.abs(ring - ring.mean(axis=0)))) <= 1.0e-9 * scale:
                    continue  # degenerate: the ring is already a point
                hub = np.broadcast_to(ring.mean(axis=0), ring.shape)
                fan = (
                    np.stack([ring[j], ring[j_next], hub], axis=1)
                    if forward
                    else np.stack([ring[j_next], ring[j], hub], axis=1)
                )
                caps.append(fan)
            if caps:
                triangles = np.concatenate([triangles, *caps])

        edge_a = triangles[:, 1] - triangles[:, 0]
        edge_b = triangles[:, 2] - triangles[:, 0]
        twice_area = np.linalg.norm(np.cross(edge_a, edge_b), axis=1)
        keep = twice_area > 1.0e-14
        vertices, faces = _weld(triangles[keep], weld_tolerance)
        return cls(
            vertices=vertices,
            faces=faces,
            name=name,
            degenerate_dropped=int((~keep).sum()),
        )

    # -- basic geometry --------------------------------------------------

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    @property
    def triangles(self) -> _FloatArray:
        return np.asarray(self.vertices[self.faces])

    @property
    def centroids(self) -> _FloatArray:
        return np.asarray(self.triangles.mean(axis=1))

    def _cross(self) -> _FloatArray:
        t = self.triangles
        return np.asarray(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]))

    @property
    def areas(self) -> _FloatArray:
        return np.asarray(0.5 * np.linalg.norm(self._cross(), axis=1))

    @property
    def normals(self) -> _FloatArray:
        """Outward unit normals, from the triangle winding.

        STL also stores a normal per facet, but it is advisory and exporters
        disagree with their own winding often enough that trusting it is a
        mistake. The winding is used and the stored value ignored; for this
        file the two agree on every facet, which is how that was checked
        rather than assumed.
        """
        cross = self._cross()
        length = np.linalg.norm(cross, axis=1, keepdims=True)
        return np.asarray(cross / np.maximum(length, 1.0e-300))

    @property
    def wetted_area(self) -> float:
        return float(self.areas.sum())

    @property
    def bounds(self) -> tuple[_FloatArray, _FloatArray]:
        return (
            np.asarray(self.vertices.min(axis=0)),
            np.asarray(self.vertices.max(axis=0)),
        )

    @property
    def extent(self) -> _FloatArray:
        lo, hi = self.bounds
        return np.asarray(hi - lo)

    @property
    def axis(self) -> int:
        """Index of the longest principal extent — the vehicle's long axis."""
        return int(np.argmax(self.extent))

    @property
    def length(self) -> float:
        return float(self.extent[self.axis])

    # -- topology --------------------------------------------------------

    def edge_counts(self) -> tuple[_IntArray, _IntArray]:
        """Unique undirected edges and how many faces use each."""
        f = self.faces
        edges = np.sort(
            np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1
        )
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        return unique.astype(np.int64), counts.astype(np.int64)

    @property
    def is_closed(self) -> bool:
        """Whether every edge is shared by exactly two faces.

        The gate on :meth:`mass_properties`. An open surface still yields a
        number from the divergence theorem, and that number can place the
        centroid outside the bounding box — which is exactly what this
        file's does, and how the openness was noticed.
        """
        _, counts = self.edge_counts()
        return bool(np.all(counts == 2))

    def boundary_stations(self) -> _FloatArray:
        """Axial positions of edges used by only one face — the holes."""
        edges, counts = self.edge_counts()
        open_edges = edges[counts == 1]
        if open_edges.size == 0:
            return np.zeros(0)
        return np.asarray(
            np.unique(np.round(self.vertices[open_edges][..., self.axis].mean(axis=1), 6))
        )

    # -- shape measurements ----------------------------------------------

    def frontal_area(self, resolution: int = 512) -> float:
        """Projected silhouette area along the long axis (m²), by rasterising.

        The cheap analytic form — half the sum of
        :math:`|\\hat{n}\\cdot\\hat{a}|A`, on the argument that every ray
        crosses a closed body an even number of times — is wrong here, and
        wrong by a factor of 2.4. This mesh contains **internal bulkheads**:
        full-diameter disks at two axial stations that are not part of the
        outer surface at all. Each contributes its whole area to that sum,
        and the formula reported 26.29 m² for a body whose true frontal area
        is 10.93.

        Rasterising the projected triangles and measuring the covered area
        is immune to that, because an internal disk projects *inside* the
        silhouette and covers nothing new. It is also correct for a
        non-convex outline, which the analytic form is not.

        Parameters
        ----------
        resolution:
            Grid cells across the larger transverse extent. The default
            gives about 7 mm on this vehicle; the estimate converges from
            below as it rises, since a partially covered cell is not
            counted.
        """
        others = [i for i in range(3) if i != self.axis]
        projected = self.triangles[:, :, others]
        lo = projected.reshape(-1, 2).min(axis=0)
        hi = projected.reshape(-1, 2).max(axis=0)
        span = hi - lo
        if np.any(span <= 0.0):  # pragma: no cover - a degenerate outline
            return 0.0
        step = float(span.max()) / int(resolution)
        shape = (int(np.ceil(span[1] / step)) + 2, int(np.ceil(span[0] / step)) + 2)
        covered = np.zeros(shape, dtype=bool)

        grid = (projected - lo) / step
        for corners in grid:
            u0, v0 = np.floor(corners.min(axis=0)).astype(int)
            u1, v1 = np.ceil(corners.max(axis=0)).astype(int) + 1
            us, vs = np.meshgrid(
                np.arange(u0, u1) + 0.5, np.arange(v0, v1) + 0.5, indexing="xy"
            )
            p0, p1, p2 = corners
            denominator = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (
                p0[1] - p2[1]
            )
            if abs(denominator) < 1e-15:
                continue
            w0 = (
                (p1[1] - p2[1]) * (us - p2[0]) + (p2[0] - p1[0]) * (vs - p2[1])
            ) / denominator
            w1 = (
                (p2[1] - p0[1]) * (us - p2[0]) + (p0[0] - p2[0]) * (vs - p2[1])
            ) / denominator
            inside = (w0 >= 0.0) & (w1 >= 0.0) & (w0 + w1 <= 1.0)
            if inside.any():
                covered[v0 : v0 + inside.shape[0], u0 : u0 + inside.shape[1]] |= inside
        return float(covered.sum() * step * step)

    def station_profile(self, decimals: int = 3) -> tuple[_FloatArray, _FloatArray]:
        """Distinct axial stations and the maximum radius at each.

        A body of revolution meshed for export has very sparse axial
        tessellation — long triangles spanning metres — so binning vertices
        into uniform slices leaves most bins empty. The *distinct vertex
        stations* are the design stations, and reading those directly is
        both exact and far more informative.
        """
        z = np.round(self.vertices[:, self.axis], decimals)
        radial = np.delete(self.vertices, self.axis, axis=1)
        radius = np.hypot(radial[:, 0], radial[:, 1])
        stations = np.unique(z)
        peak = np.array([radius[z == s].max() for s in stations])
        return np.asarray(stations), np.asarray(peak)

    def _nose_profile(self) -> tuple[_FloatArray, _FloatArray]:
        """Depth behind the tip and maximum radius, per axial station."""
        z = self.vertices[:, self.axis]
        others = [i for i in range(3) if i != self.axis]
        transverse = self.vertices[:, others]
        radius = np.hypot(transverse[:, 0], transverse[:, 1])
        depth = float(z.max()) - z
        stations = np.unique(np.round(depth, 6))
        stations = stations[stations > 0.0]
        peak = np.array(
            [radius[np.isclose(np.round(depth, 6), s)].max() for s in stations]
        )
        return np.asarray(stations), np.asarray(peak)

    def nose_exponent(self, stations: int = 12) -> float:
        """Exponent :math:`p` of a power-law nose :math:`r = k d^{p}`.

        The shape test, and it should be run before anything asks for a
        nose *radius*. A hemisphere gives :math:`p = 1/2` and a cone
        :math:`p = 1`; anything between is an ogive or a power-series body,
        for which no single radius of curvature describes the surface.

        This vehicle returns **0.59**, and the consequence is concrete: its
        :math:`r^2/2d` — the quantity that *would* be the radius if the nose
        were spherical — climbs monotonically from 0.27 m to 0.60 m over the
        first 0.8 m of depth and never settles. Any "fitted nose radius" is
        then a statement about the fit window, not about the vehicle.
        """
        depth, radius = self._nose_profile()
        take = min(int(stations), depth.size)
        if take < 3:
            msg = f"need at least three nose stations to fit an exponent, got {take}"
            raise ValueError(msg)
        slope = np.polyfit(np.log(depth[:take]), np.log(radius[:take]), 1)[0]
        return float(slope)

    def nose_radius(self, window: float | None = None) -> float:
        """Spherical-cap radius fitted over ``window`` metres behind the tip.

        For a sphere the profile near the apex is :math:`r^2 = 2R_n d`, so a
        least-squares fit of :math:`r^2` against :math:`2d` gives
        :math:`R_n`. **That model only applies if the nose is spherical**;
        check :meth:`nose_exponent` first.

        There is deliberately no clever automatic window. An earlier version
        swept windows and took the longest run where the answer was stable
        to 2 %, which sounds robust and is not: on a coarsely tessellated
        tip several small windows catch the *same two vertex rings* and
        therefore agree exactly, so the sweep found a two-ring "plateau" at
        0.35 m and reported it with confidence. The default here is the
        smallest window spanning at least three stations — the most local
        estimate the mesh can support — and it is a **bound on the tip
        curvature**, not a measurement of a sphere that is not there.

        Feeds :math:`R_{\\mathrm{eff}}` in Sutton-Graves, where heating goes
        as :math:`R_n^{-1/2}`.
        """
        depth, radius = self._nose_profile()
        if window is None:
            if depth.size < 3:
                msg = (
                    f"{self.name!r} has {depth.size} nose stations; too few to "
                    "estimate a tip curvature"
                )
                raise ValueError(msg)
            span = float(depth[2])
        else:
            span = float(window)
        near = depth <= span * (1.0 + 1e-9)
        if int(near.sum()) < 2:
            msg = (
                f"only {int(near.sum())} nose stations within {span:g} m of the "
                "tip; widen the window or supply a finer mesh"
            )
            raise ValueError(msg)
        d = 2.0 * depth[near]
        return float(np.sum(radius[near] ** 2 * d) / np.sum(d * d))

    # -- mass ------------------------------------------------------------

    def mass_properties(self, density: float) -> dict[str, object]:
        """Volume, centroid and inertia for a **uniform** solid of ``density``.

        Stated, not measured. A surface carries no density distribution, and
        a launcher is emphatically not uniform — its propellant drains from
        the bottom, moving the centre of mass metres during a burn. This is
        the arithmetic that a uniform-solid assumption implies, offered so
        the assumption can be replaced rather than left implicit.

        Raises
        ------
        ValueError
            If the mesh is not closed, because the divergence-theorem sums
            below are then meaningless while still returning numbers.
        """
        if not (np.isfinite(density) and density > 0.0):
            msg = f"density must be finite and > 0, got {density}"
            raise ValueError(msg)
        if not self.is_closed:
            holes = self.boundary_stations()
            msg = (
                f"mesh {self.name!r} is not closed, so volume and inertia are "
                f"undefined; open edges lie at axial stations "
                f"{np.round(holes, 3).tolist()}. Cap the holes first."
            )
            raise ValueError(msg)

        t = self.triangles
        a, b, c = t[:, 0], t[:, 1], t[:, 2]
        signed = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
        volume = float(signed.sum())
        centroid = np.asarray(((a + b + c) / 4.0 * signed[:, None]).sum(0) / volume)

        # Inertia of the union of origin-based tetrahedra, each with the
        # sign of its own volume. Standard covariance-of-a-tetrahedron form.
        inertia = np.zeros((3, 3))
        for pa, pb, pc, vol in zip(a, b, c, signed, strict=True):
            points = np.stack([pa, pb, pc])
            # Second-moment integral of the tetrahedron (origin, a, b, c):
            # int x x^T dV = V/20 (sum_i p_i p_i^T + (sum_i p_i)(sum_i p_i)^T),
            # with V the *signed* volume already. Multiplying by six here as
            # well — as if `signed` were the determinant rather than the
            # volume — inflates the tensor 6x, which on a unit cube gives
            # 3,500 where the closed form m a^2 / 6 says 166.7.
            covariance = (
                points.T @ points + np.outer(points.sum(0), points.sum(0))
            ) / 20.0
            inertia += vol * covariance
        trace = np.trace(inertia)
        tensor = density * (trace * np.eye(3) - inertia)
        mass = density * volume
        shift = mass * (
            float(centroid @ centroid) * np.eye(3) - np.outer(centroid, centroid)
        )
        return {
            "volume": volume,
            "mass": mass,
            "centroid": centroid,
            "inertia_about_origin": tensor,
            "inertia_about_centroid": tensor - shift,
        }

    def scaled(self, axial: float = 1.0, radial: float = 1.0) -> VehicleMesh:
        """Scale along the long axis and across it independently.

        Anisotropic on purpose. The bundled mesh's two dimensional errors
        against the published vehicle are independent — length low by 1.1 %,
        diameter high by 15.6 % — so a single uniform factor cannot fix
        both, and applying one would trade a small length error for a large
        one. Scaling the axis and the radius separately lands both, and
        lands the fineness ratio with them.

        What this preserves and what it does not: every axial *proportion*
        survives, so the separation rings stay at the same fraction of the
        body and the nose keeps its shape in the axial sense. The nose
        *exponent* survives too. What changes is every angle — a 28-degree
        cone half-angle becomes 24.6 degrees under a 0.865 radial squeeze —
        and therefore every Newtonian pressure coefficient. That is correct
        rather than a side effect: the real vehicle's cone really is
        shallower than this mesh's.
        """
        for label, value in (("axial", axial), ("radial", radial)):
            if not (np.isfinite(value) and value > 0.0):
                msg = f"{label} scale must be finite and > 0, got {value}"
                raise ValueError(msg)
        factors = np.full(3, float(radial))
        factors[self.axis] = float(axial)
        return VehicleMesh(
            vertices=self.vertices * factors,
            faces=self.faces,
            name=self.name,
            degenerate_dropped=self.degenerate_dropped,
        )

    def scaled_to(
        self, length: float | None = None, diameter: float | None = None
    ) -> VehicleMesh:
        """Scale so the body matches a stated length and maximum diameter.

        The maximum diameter is taken over the *whole* body, raised bands
        included, because that is what a published diameter figure refers
        to on a vehicle whose rings are structural rather than aerodynamic.
        """
        current_length = self.length
        others = [i for i in range(3) if i != self.axis]
        transverse = self.vertices[:, others]
        current_diameter = 2.0 * float(np.hypot(transverse[:, 0], transverse[:, 1]).max())
        return self.scaled(
            axial=1.0 if length is None else float(length) / current_length,
            radial=1.0 if diameter is None else float(diameter) / current_diameter,
        )

    # -- frames and consumers --------------------------------------------

    def to_body_axes(self, origin: str = "nose") -> VehicleMesh:
        """Rotate so the nose points along **+x**, the framework convention.

        :class:`~aether.aerodynamics.panels.PanelModel` and the vehicle
        glyph in :mod:`aether_gambit.viz.scene` both take the nose along +x; meshes
        are usually authored along +z. Doing the rotation here, once and
        explicitly, is what stops the two conventions meeting silently in
        somebody's notebook.

        Parameters
        ----------
        origin:
            ``"nose"`` puts the tip at the origin with the body on -x,
            ``"centroid"`` centres the bounding box, ``"keep"`` translates
            nothing.
        """
        source = np.zeros(3)
        source[self.axis] = 1.0
        # The nose is the end with the smaller radius; if the mesh runs the
        # other way the axis is flipped so +x is always forward.
        _, peak = self.station_profile()
        if peak[0] < peak[-1]:
            source = -source
        target = np.array([1.0, 0.0, 0.0])

        rotation = _shortest_arc(source, target)
        rotated = self.vertices @ rotation.T
        if origin == "nose":
            rotated = rotated - np.array([rotated[:, 0].max(), 0.0, 0.0])
        elif origin == "centroid":
            rotated = rotated - 0.5 * (rotated.min(axis=0) + rotated.max(axis=0))
        elif origin != "keep":
            msg = f"origin must be 'nose', 'centroid' or 'keep', got {origin!r}"
            raise ValueError(msg)
        return VehicleMesh(
            vertices=rotated, faces=self.faces, name=self.name,
            degenerate_dropped=self.degenerate_dropped,
        )

    def panel_model(self, reference_point: _FloatArray | None = None) -> PanelModel:
        """Feed the mesh to the existing panel aerodynamics.

        A triangle *is* a panel: centroid, outward unit normal, area. No
        approximation is introduced by this step, which is the reason the
        panel model was worth having a mesh for.

        The mesh is expected nose-along-**+x** (:meth:`to_body_axes`), and
        this method flips it, because :class:`PanelModel` uses the opposite
        convention and the difference is a vehicle flown backwards.

        Its ``velocity_direction`` is :math:`(\\cos\\alpha, 0, \\sin\\alpha)`
        and its incidence is :math:`\\arcsin(-\\hat n\\cdot\\hat v)`, so a
        panel is windward when its normal points along **-x**: that vector
        is the direction the *flow travels*, not the direction the nose
        points. Checked against a flat plate, which is unambiguous — normal
        along -x collects 1,831 N of a 1,000 Pa freestream at Mach 10,
        normal along +x collects 14 N.

        Getting this backwards is not subtle in its consequences and is
        entirely silent: the axial force coefficient came out at **-5.7**,
        a drag coefficient that is both negative and an order of magnitude
        too large, because the vehicle was being flown tail first with its
        internal bulkheads presented to the stream.
        """
        flip = np.diag([-1.0, 1.0, -1.0])  # 180 degrees about y: a rotation
        return PanelModel(
            centroids=self.centroids @ flip.T,
            normals=self.normals @ flip.T,
            areas=self.areas,
            reference_point=(
                np.zeros(3) if reference_point is None
                else np.asarray(reference_point, dtype=np.float64) @ flip.T
            ),
        )

    def _ray_hits(
        self, origins: _FloatArray, directions: _FloatArray, backend: str
    ) -> NDArray[np.int64]:
        """Count mesh intersections for a batch of rays. Möller-Trumbore."""
        from aether.batch.backend import get_array_module, to_numpy

        xp = get_array_module(backend)  # type: ignore[arg-type]
        tri = xp.asarray(self.triangles)
        v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
        edge1, edge2 = v1 - v0, v2 - v0
        origin = xp.asarray(origins)
        direction = xp.asarray(directions)

        counts = xp.zeros(origin.shape[0], dtype=xp.int64)
        chunk = max(1, int(4_000_000 // max(self.n_faces, 1)))
        for start in range(0, origin.shape[0], chunk):
            stop = min(start + chunk, origin.shape[0])
            d = direction[start:stop][:, None, :]
            o = origin[start:stop][:, None, :]
            pvec = xp.cross(d, edge2[None, :, :])
            det = xp.sum(edge1[None, :, :] * pvec, axis=2)
            parallel = xp.abs(det) < 1.0e-12
            inv_det = 1.0 / xp.where(parallel, 1.0, det)
            tvec = o - v0[None, :, :]
            u = xp.sum(tvec * pvec, axis=2) * inv_det
            qvec = xp.cross(tvec, edge1[None, :, :])
            v = xp.sum(d * qvec, axis=2) * inv_det
            t = xp.sum(edge2[None, :, :] * qvec, axis=2) * inv_det
            inside = (
                (~parallel) & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1.0e-9)
            )
            counts[start:stop] = xp.sum(inside, axis=1)
        return np.asarray(to_numpy(counts).astype(np.int64))

    def oriented(self, backend: str = "numpy", probes: int = 7) -> VehicleMesh:
        """A copy whose faces are all wound so their normals point outward.

        **This mesh needs it.** Its winding is inconsistent: the barrel is
        97 % outward while all 2,000 nose facets are wound *inward*, and the
        STL's own stored normals agree with the bad winding, so there is no
        second opinion in the file. Every pressure integration over the raw
        mesh is wrong, and silently: the nose — the one part of a slender
        body that carries most of its axial force — pushes the wrong way.

        The test is parity, per face and independent of adjacency, which
        matters because this mesh is neither closed nor manifold and a
        breadth-first winding propagation would stall at the first bad edge.
        A point just off the surface along the normal is inside the solid if
        a ray from it crosses the surface an odd number of times.

        Parity is used with a **majority vote over several directions**
        rather than once. A single ray is not reliable here: the vehicle
        contains internal bulkheads, so a ray fired aft from a point ahead
        of one crosses it and then the base — two crossings, "outside" —
        while the same point fired forward crosses only the nose and reports
        "inside". Voting removes that, and it removes grazing hits at
        triangle edges at the same time.

        Parameters
        ----------
        backend:
            ``"numpy"`` or ``"cupy"``; the cost is ``probes`` times
            :math:`\\mathcal{O}(F^2)`.
        probes:
            Ray directions per face. Odd, so the vote cannot tie.
        """
        if probes % 2 == 0:
            msg = f"probes must be odd so the vote cannot tie, got {probes}"
            raise ValueError(msg)
        scale = float(np.max(self.extent))
        origins = self.centroids + 1.0e-6 * scale * self.normals

        rng = np.random.default_rng(12345)
        votes = np.zeros(self.n_faces, dtype=np.int64)
        for _ in range(int(probes)):
            direction = rng.normal(size=3)
            direction = direction / float(np.linalg.norm(direction))
            counts = self._ray_hits(
                origins, np.broadcast_to(direction, (self.n_faces, 3)).copy(), backend
            )
            votes += (counts % 2).astype(np.int64)  # odd => inside the solid
        inside = votes * 2 > int(probes)

        faces = self.faces.copy()
        faces[inside] = faces[inside][:, ::-1]
        return VehicleMesh(
            vertices=self.vertices, faces=faces, name=self.name,
            degenerate_dropped=self.degenerate_dropped,
        )

    def outward_fraction(self) -> float:
        """Fraction of faces whose normal points away from the long axis.

        A cheap winding-consistency check for a body of revolution: on a
        correctly wound one it is near 1, and the shortfall is only the
        undercut faces of any raised band. This mesh scores 0.71 before
        :meth:`oriented` and 0.97 after.
        """
        others = [i for i in range(3) if i != self.axis]
        radial = self.centroids[:, others]
        span = np.linalg.norm(radial, axis=1)
        good = span > 1.0e-9
        if not good.any():  # pragma: no cover - a degenerate body
            return 1.0
        unit = radial[good] / span[good, None]
        return float((np.einsum("ij,ij->i", self.normals[good][:, others], unit) > 0).mean())

    def exterior_faces(self, backend: str = "numpy", offset: float = 1.0e-4) -> NDArray[np.bool_]:
        """Which faces can see infinity along their own outward normal.

        The aero integration needs this. A mesh authored for visualisation
        carries structure that is not outer surface — this vehicle has two
        full-diameter **internal bulkheads** — and a panel method integrates
        pressure over whatever it is handed. Measured on this vehicle the
        bulkheads add **+0.89 to the axial force coefficient at Mach 2**,
        falling to +0.009 at Mach 20 as Newtonian shading takes over. Low
        supersonic is exactly where the coefficient is least reliable
        anyway, so contaminating it further is not acceptable.

        The test is a ray cast: from each face centroid, step off the
        surface along the normal and shoot to infinity. A face that hits the
        mesh again is enclosed by it. Möller-Trumbore, every ray against
        every triangle.

        Parameters
        ----------
        backend:
            ``"numpy"`` or ``"cupy"``. The test is
            :math:`\\mathcal{O}(F^2)` — 48 million ray-triangle
            intersections on this mesh — and is the kind of dense, uniform
            arithmetic a GPU is for.
        offset:
            How far to step off the surface before shooting, as an absolute
            distance (m). Without it every ray hits its own triangle.

        Notes
        -----
        This removes *enclosed* geometry. It does **not** model concave
        shadowing, where an external panel is shielded from the freestream
        by another external panel at a particular attitude — that is
        attitude-dependent and a different calculation. Newtonian theory
        handles convex shadowing on its own through negative incidence.
        """
        counts = self._ray_hits(
            self.centroids + offset * self.normals, self.normals, backend
        )
        return np.asarray(counts == 0)

    def exterior(self, backend: str = "numpy") -> VehicleMesh:
        """The sub-mesh of faces that can see infinity — see
        :meth:`exterior_faces`."""
        keep = self.exterior_faces(backend)
        faces = self.faces[keep]
        used, remapped = np.unique(faces, return_inverse=True)
        return VehicleMesh(
            vertices=self.vertices[used],
            faces=remapped.reshape(faces.shape).astype(np.int64),
            name=f"{self.name} (exterior)",
        )

    # -- sub-structure ---------------------------------------------------

    def raised_bands(self, prominence: float = 0.02) -> _FloatArray:
        """Axial stations of rings standing proud of the local body radius.

        A launcher's separation planes are almost always marked by a raised
        band — a raceway, a retro-rocket fairing, a separation-joint ring —
        because that hardware has to sit somewhere. Finding them is
        therefore a good first guess at where the stages divide, and it is
        only a guess: this returns geometry, not a staging sequence.

        Parameters
        ----------
        prominence:
            Minimum radius excess over the surrounding body, as a fraction
            of the maximum radius.
        """
        stations, peak = self.station_profile()
        if stations.size < 5:
            return np.zeros(0)
        threshold = float(peak.max()) * (1.0 - prominence)
        proud = peak >= threshold
        # Group consecutive proud stations into bands and take each centre.
        bands: list[float] = []
        run: list[float] = []
        for station, flag in zip(stations, proud, strict=True):
            if flag:
                run.append(float(station))
            elif run:
                bands.append(float(np.mean(run)))
                run = []
        if run:
            bands.append(float(np.mean(run)))
        return np.asarray(bands)

    def section(self, low: float, high: float) -> VehicleMesh:
        """The sub-mesh whose face centroids lie in ``[low, high]`` on the axis.

        A cut by *whole faces*, not a true boolean slice: no triangle is
        split and no cap is added, so a section is open at both ends and its
        volume is undefined. That is the honest tool for asking "what is the
        wetted area and frontal area of stage two"; it is not a tool for
        asking "what does stage two weigh".
        """
        if not high > low:
            msg = f"need high > low, got {low} and {high}"
            raise ValueError(msg)
        keep = (self.centroids[:, self.axis] >= low) & (
            self.centroids[:, self.axis] <= high
        )
        if not keep.any():
            msg = f"no faces between {low} and {high} on axis {self.axis}"
            raise ValueError(msg)
        faces = self.faces[keep]
        used, remapped = np.unique(faces, return_inverse=True)
        return VehicleMesh(
            vertices=self.vertices[used],
            faces=remapped.reshape(faces.shape).astype(np.int64),
            name=f"{self.name}[{low:g}:{high:g}]",
        )


def write_stl(mesh: VehicleMesh, path: str | Path, name: str | None = None) -> Path:
    """Write a binary STL.

    Normals are written from the winding, so a file produced here is
    self-consistent — unlike the one that started this, whose stored normals
    faithfully recorded a broken winding.
    """
    location = Path(path)
    location.parent.mkdir(parents=True, exist_ok=True)
    triangles = mesh.triangles
    normals = mesh.normals
    header = (name or mesh.name or location.stem).encode("ascii", "replace")[:79]
    with location.open("wb") as handle:
        handle.write(header.ljust(80, b"\0"))
        handle.write(struct.pack("<I", mesh.n_faces))
        for normal, triangle in zip(normals, triangles, strict=True):
            handle.write(struct.pack("<3f", *normal))
            for vertex in triangle:
                handle.write(struct.pack("<3f", *vertex))
            handle.write(b"\0\0")
    return location


def _shortest_arc(source: _FloatArray, target: _FloatArray) -> _FloatArray:
    """Rotation matrix taking unit ``source`` to unit ``target``."""
    a = np.asarray(source, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    a = a / float(np.linalg.norm(a))
    b = b / float(np.linalg.norm(b))
    cosine = float(a @ b)
    if cosine > 1.0 - 1e-12:
        return np.eye(3)
    if cosine < -1.0 + 1e-12:
        # Antiparallel: any axis perpendicular to a gives a 180-degree turn.
        seed = np.array([1.0, 0.0, 0.0])
        if abs(float(a @ seed)) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, seed)
        axis = axis / float(np.linalg.norm(axis))
        return np.asarray(2.0 * np.outer(axis, axis) - np.eye(3))
    axis = np.cross(a, b)
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.asarray(np.eye(3) + skew + skew @ skew / (1.0 + cosine))
