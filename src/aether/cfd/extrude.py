r"""Grow a wall-normal grid from the body out to just beyond the shock.

The domain a blunt hypersonic body needs is **thin**. Everything outside the
bow shock is undisturbed freestream, so LAURA places its outer boundary at
:math:`fsh = 0.8` of the body-to-inflow line -- just outside the captured shock
-- and the whole domain is one wall-normal block
→ :mod:`aether.cfd.gridding`. That is the opposite of what this pipeline
was doing, which was filling a large box with isotropic tetrahedra and spending
9.5x more cells to put **zero** of them inside a 9 mm shock layer.

The layer thickness is not constant and must not be. On the Mach 8 sphere-cone
the shock stands 9 mm off the nose and roughly 50-100 mm off the base, because
a 13 degree shock diverges from a 10 degree cone. Marching each surface node to
the shock envelope along its own normal is what follows that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "ExtrudedLayer",
    "WakeBlock",
    "boundary_edges",
    "extrude_to_shock",
    "geometric_spacing",
    "limit_thickness_gradient",
    "march_thickness",
    "measure_quality",
    "vertex_normals",
    "wake_block",
    "write_su2",
]


def geometric_spacing(
    first: float, total: float, layers: int, growth: float = 1.15
) -> np.ndarray:
    r"""Normalised node positions from 0 to 1, geometrically graded.

    The first interval is ``first/total`` and each subsequent one is ``growth``
    times its predecessor, until the sum would exceed the total -- after which
    the remaining intervals are uniform. That cap matters: an unchecked
    geometric series overshoots wildly, and a grid whose outer cells are larger
    than the shock layer resolves nothing at the one place the physics lives.

    Returns ``layers + 1`` values, the first exactly 0 and the last exactly 1.
    """
    if layers < 1:
        msg = f"need at least one layer, got {layers}"
        raise ValueError(msg)
    if not 0.0 < first < total:
        msg = f"first cell {first:g} must be positive and below the total {total:g}"
        raise ValueError(msg)
    step = first / total
    steps = np.empty(layers, dtype=np.float64)
    remaining = 1.0
    for i in range(layers):
        left = layers - i
        # Never take a step so large that the uniform remainder would have to
        # shrink; that is the overshoot guard.
        steps[i] = min(step, remaining / left) if left > 1 else remaining
        remaining -= steps[i]
        step *= growth
    positions = np.concatenate([[0.0], np.cumsum(steps)])
    positions[-1] = 1.0
    return positions


def vertex_normals(
    vertices: np.ndarray, faces: np.ndarray, areas: np.ndarray, normals: np.ndarray,
    smoothing: int = 2,
) -> np.ndarray:
    """Area-weighted vertex normals, Laplacian-smoothed and re-normalised.

    Smoothing is what keeps marched columns from crossing where the surface
    turns quickly. It is applied to the **normal field**, not the geometry, so
    the wall itself is untouched -- a smoothed body would be a different
    vehicle.
    """
    accumulated = np.zeros_like(vertices)
    for corner in range(faces.shape[1]):
        np.add.at(accumulated, faces[:, corner], normals * areas[:, None])
    for _ in range(max(0, smoothing)):
        neighbours = np.zeros_like(accumulated)
        counts = np.zeros(len(vertices))
        for a, b in ((0, 1), (1, 2), (2, 0)):
            np.add.at(neighbours, faces[:, a], accumulated[faces[:, b]])
            np.add.at(neighbours, faces[:, b], accumulated[faces[:, a]])
            np.add.at(counts, faces[:, a], 1.0)
            np.add.at(counts, faces[:, b], 1.0)
        moving = counts > 0
        accumulated[moving] = 0.5 * accumulated[moving] + 0.5 * (
            neighbours[moving] / counts[moving, None]
        )
    lengths = np.linalg.norm(accumulated, axis=1)
    lengths[lengths == 0.0] = 1.0
    return accumulated / lengths[:, None]


def limit_thickness_gradient(
    thickness: np.ndarray, faces: np.ndarray, ratio: float = 1.5, sweeps: int = 50
) -> np.ndarray:
    """Stop adjacent columns differing by more than ``ratio``, by **growing** the short ones.

    A caret waverider's leading edge has the shock a couple of millimetres off
    the compression side and nothing at all above, so raw marched thicknesses
    there differ by a measured **530x** between neighbouring vertices. Cells
    spanning that are unusable whatever the solver.

    The direction matters and is not symmetric. Shortening a long column could
    pull the outer boundary **inside** the shock, which silently truncates the
    physics; lengthening a short one only adds freestream above it, which costs
    a little resolution and nothing else. So this only ever grows, and the
    result is guaranteed to still contain everything the march found.
    """
    if ratio <= 1.0:
        msg = f"ratio must exceed 1, got {ratio:g}"
        raise ValueError(msg)
    out = np.array(thickness, dtype=np.float64)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    a, b = edges[:, 0], edges[:, 1]
    for _ in range(sweeps):
        before = out.copy()
        np.maximum.at(out, a, before[b] / ratio)
        np.maximum.at(out, b, before[a] / ratio)
        if np.allclose(out, before, rtol=1e-12, atol=0.0):
            break
    return out


#: How far past the predicted shock the outer boundary is marched, as a
#: multiple of the distance to it.
#:
#: **1.75, raised from 1.25.** The margin has to cover the error in the
#: *prediction*, and Billig's is not small on a bluff body: the hemisphere's
#: measured standoff is 7.5 mm against a predicted 6.77, 11 % out, which at 1.25
#: left the shock sitting 7.5 mm inside a boundary at 8.5 -- one millimetre, two
#: or three cells, of freestream between them.
#:
#: That was survivable while the boundary extrapolated. It is not survivable now
#: that it **imposes** freestream: a condition asserting undisturbed flow two
#: cells from a shock is asserting something false, and the hemisphere answered
#: with 83 % of its boundary nodes over-speed in the band from 40 to 60 degrees.
#:
#: The cost is cells spread over a thicker domain, and it is paid in the part of
#: it that is uniform freestream.
MARGIN = 1.75


def march_thickness(
    surface: Any,
    envelope: Any,
    margin: float = MARGIN,
    smoothing: int = 2,
    cap: float | None = None,
    floor: float = 0.0,
    gradient: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Distance from every forebody vertex to the shock, along its own normal.

    Returns ``(vertices, faces, directions, thickness, capped)``.

    **Not every normal reaches a shock, and that is physics rather than a bug.**
    A caret waverider's upper surface is a *freestream* surface: it is shaped so
    the flow passes it undisturbed, so its outward normals run away from the
    attached shock and never cross it. A sharp-nosed slender body has its shock
    attached at the apex, so the nose standoff is zero and marching upstream
    from the tip crosses nothing either.

    Measured before this cap existed: the caret waverider produced columns
    19.9 m long and the sharp bodies 12.5 m, against vehicles 4 m and 2.5 m
    long. ``capped`` counts those columns, because a large count means the
    envelope does not bound this body and the mesh should be questioned rather
    than used.
    """
    from aether.cfd.shock import _march_to_shock

    vertices = np.asarray(surface.vertices, dtype=np.float64)
    faces = np.asarray(surface.faces, dtype=np.int64)
    areas = np.asarray(surface.areas, dtype=np.float64)
    normals = np.asarray(surface.normals, dtype=np.float64)

    centroids = vertices[faces].mean(axis=1)
    forebody = ~np.isclose(centroids[:, 0], centroids[:, 0].max(), atol=1e-6)
    faces, areas, normals = faces[forebody], areas[forebody], normals[forebody]

    used = np.unique(faces)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    faces = remap[faces]
    vertices = vertices[used]

    directions = vertex_normals(vertices, faces, areas, normals, smoothing=smoothing)
    if not np.all(np.isfinite(directions)) or np.any(
        np.linalg.norm(directions, axis=1) < 0.5
    ):
        msg = "some surface vertex has no usable normal after dropping the base"
        raise ValueError(msg)

    span = float(np.ptp(vertices[:, 0]))
    if cap is None:
        # A quarter of the body's **axial length** is the wrong scale for a
        # blunt body, and it silently truncated 42.6 % of a hemisphere's
        # columns. The envelope encloses that body perfectly well -- at the
        # equator radius the shock stands at x = 11.9 mm against a body station
        # of 45 mm -- but the shock layer along the shoulder normal is genuinely
        # 30 mm, and a quarter of a 45 mm span is 11.25 mm. On a 2 m sphere-cone
        # the same rule gives 515 mm and never binds, which is why this survived.
        #
        # The standoff is the envelope's own length scale, so the cap is taken
        # from it where the caller knows it, and the span rule is kept as the
        # floor for bodies where it is the larger of the two.
        cap = 0.25 * span
    reach = 4.0 * span
    raw = np.array(
        [
            _march_to_shock(envelope, vertices[i], directions[i], limit=reach)
            for i in range(len(vertices))
        ]
    )
    capped = int(np.sum(raw * margin > cap))
    thickness = np.clip(raw * margin, max(floor, 0.0), cap)
    thickness = np.minimum(limit_thickness_gradient(thickness, faces, gradient), cap)
    return vertices, faces, directions, thickness, capped


@dataclass(frozen=True)
class ExtrudedLayer:
    """Node positions of a wall-normal block, and how healthy it is."""

    nodes: np.ndarray
    """``(n_vertices, layers + 1, 3)`` -- each surface node's column outward."""

    thickness: np.ndarray
    """``(n_vertices,)`` distance marched, metres."""

    wall_cell: float
    crossings: int
    """Columns whose cells invert, i.e. normals that collided. **Must be 0.**"""

    faces: np.ndarray
    """``(n_faces, 3)`` surface connectivity, re-indexed onto the kept vertices."""

    capped: int = 0
    """Columns whose normal never reached the shock and took the cap instead.

    Expected to be non-zero on a waverider's freestream surface. A large
    fraction means the envelope does not bound the body.
    """

    growth: np.ndarray | None = None
    """``(n_vertices,)`` geometric ratio actually used on each column.

    Per-column, because one ratio cannot span both a 9 mm nose standoff and a
    half-metre base shock layer. Carried so the mesh report can state the range
    rather than a nominal figure no column used.
    """

    @property
    def layers(self) -> int:
        return self.nodes.shape[1] - 1

    @property
    def cells(self) -> int:
        return (self.nodes.shape[0]) * self.layers


def extrude_to_shock(
    surface: Any,
    envelope: Any,
    wall_cell: float,
    layers: int,
    margin: float = MARGIN,
    growth: float = 1.15,
    smoothing: int = 2,
    cap: float | None = None,
) -> ExtrudedLayer:
    """March every surface node to the shock and grade the column.

    ``margin`` is LAURA's :math:`1/fsh`: the outer boundary stands this multiple
    of the shock distance off the body, 1.25 at the shipped ``fsh = 0.8``.

    The base disc is excluded -- it is not a resolved surface on this pipeline
    and extruding it would march columns straight into the wake
    → :mod:`aether.cfd.forces`.
    """
    vertices, faces, directions, thickness, capped = march_thickness(
        surface, envelope, margin=margin, smoothing=smoothing, cap=cap,
        floor=4.0 * wall_cell,
    )
    # The growth is solved **per column**, from that column's own marched
    # thickness. A single ratio cannot serve them all: the shock layer on a
    # sphere-cone is a 9 mm standoff at the nose and half a metre at the base,
    # and a geometric series sized for the first spans 2 % of the last.
    #
    # What that produced, measured on the Mach 8 alpha-4 run: 55 graded levels
    # packed into 10.68 mm at *every* station, then one cell carrying the
    # remaining 8 mm at the nose and 482 mm at the base -- a cell-to-cell jump
    # of 841x, with 97.8 % of the shock layer inside a single cell. The bow
    # shock had no cells to live in, so the numerical Schlieren was blank, the
    # Mach field was flat between the boundary layer and the outer boundary,
    # the outermost nodes read Mach 9.5 against a freestream of 8.0, and 1-2 %
    # of the domain carried nonphysical states -- one at 3e-05 K, which is a
    # vacuum. `geometric_spacing` warns about precisely this outcome in its own
    # docstring; its overshoot guard defends against a series that is too long
    # and there was nothing defending against one far too short.
    #
    # ``growth`` survives as the fallback for a degenerate column that
    # ``growth_for`` cannot solve, and as the documented default.
    from aether.cfd.gridding import growth_for

    ratios = np.empty(len(thickness), dtype=np.float64)
    for i, t in enumerate(thickness):
        try:
            ratios[i] = growth_for(wall_cell, float(t), layers)
        except ValueError:
            ratios[i] = growth
    fractions = np.stack(
        [
            geometric_spacing(wall_cell, t, layers, g)
            for t, g in zip(thickness, ratios, strict=True)
        ]
    )
    nodes = vertices[:, None, :] + fractions[:, :, None] * (
        thickness[:, None, None] * directions[:, None, :]
    )
    steps = np.diff(nodes, axis=1)
    advance = np.einsum("ijk,ik->ij", steps, directions)
    crossings = int(np.sum(np.any(advance <= 0.0, axis=1)))
    return ExtrudedLayer(
        nodes=nodes,
        thickness=thickness,
        wall_cell=wall_cell,
        crossings=crossings,
        faces=faces,
        capped=capped,
        growth=ratios,
    )


# ------------------------------------------------------------------- quality

#: Corners of a prism as (apex, three neighbours), right-handed so a well-formed
#: cell gives a positive determinant at every one.
_CORNERS: tuple[tuple[int, int, int, int], ...] = (
    (0, 1, 2, 3), (1, 2, 0, 4), (2, 0, 1, 5),
    (3, 5, 4, 0), (4, 3, 5, 1), (5, 4, 3, 2),
)


def measure_quality(layer: ExtrudedLayer, threshold: float = 0.1) -> Any:
    """Scaled Jacobian of every prism, as a :class:`MeshQuality`.

    The isotropic path reports gmsh's ``minSICN``, which is gmsh's to compute
    and unavailable here. The scaled Jacobian is the standard analogue and
    shares the property that makes ``minSICN`` the right default: **it is signed**,
    one for a perfect cell and at or below zero for an inverted one. A merely
    stretched cell is survivable and an inverted cell is not, and no one-sided
    measure separates them.

    Reusing :class:`MeshQuality` rather than inventing a parallel type means the
    worker's log line, the record, and anything reading it downstream do not
    have to care which mesher ran. The ``measure`` field says which number it is.

    A wall-normal grid is *deliberately* stretched -- aspect ratios here are
    hundreds to hundreds of thousands -- so a low scaled Jacobian is expected
    and is not by itself a fault. The number that matters is the count at or
    below zero, which must be nil.
    """
    from aether.aerodynamics.cfd.meshing import MeshQuality

    nodes, faces = layer.nodes, layer.faces
    n_surface, n_levels = nodes.shape[0], nodes.shape[1]
    layers = n_levels - 1
    flat = np.concatenate([nodes[:, k, :] for k in range(n_levels)], axis=0)
    lower = np.concatenate([faces + k * n_surface for k in range(layers)], axis=0)
    upper = np.concatenate([faces + (k + 1) * n_surface for k in range(layers)], axis=0)
    cells = flat[np.concatenate([lower, upper], axis=1)]

    best = np.full(len(cells), np.inf)
    for apex, a, b, c in _CORNERS:
        e1 = cells[:, a] - cells[:, apex]
        e2 = cells[:, b] - cells[:, apex]
        e3 = cells[:, c] - cells[:, apex]
        lengths = (
            np.linalg.norm(e1, axis=1)
            * np.linalg.norm(e2, axis=1)
            * np.linalg.norm(e3, axis=1)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            jacobian = np.einsum("ij,ij->i", np.cross(e1, e2), e3) / lengths
        best = np.minimum(best, np.where(lengths > 0.0, jacobian, -1.0))

    return MeshQuality(
        measure="scaledJacobian",
        minimum=float(best.min()),
        first_percentile=float(np.percentile(best, 1.0)),
        median=float(np.median(best)),
        inverted=int(np.sum(best <= 0.0)),
        poor=int(np.sum(best < threshold)),
        threshold=threshold,
        elements=len(cells),
    )



# --------------------------------------------------------------------- wake


@dataclass(frozen=True)
class WakeBlock:
    """Prisms aft of the base plane, swept downstream."""

    nodes: np.ndarray
    """``(n_cap_nodes, layers + 1, 3)``."""

    faces: np.ndarray
    """Base-disc triangles, indexed into the first axis of :attr:`nodes`."""

    quads: np.ndarray
    """Annulus quadrilaterals -- the forebody block's own outflow faces.

    Kept as quads and **not** triangulated. The forebody block is prisms, so its
    lateral boundary is quadrilateral; splitting each into two triangles here
    produces faces that are geometrically coincident with it and topologically
    different, and the two blocks then touch without joining. Measured on the
    sphere-cone: 13 680 boundary faces belonging to no marker, which is a domain
    with a hole in it that every cell-quality measure reports as healthy.

    Swept downstream these give **hexahedra**, so the interface is one quad
    against one quad.
    """

    base_faces: int
    """How many of :attr:`faces` are the base disc itself -- a wall, not an inlet."""

    base_quads: int = 0
    """How many of :attr:`quads` belong to the base disc rather than the annulus.

    The refined base is a radial grid, so most of it is quadrilateral and only
    the centre is a fan. Both parts are wall; the annulus that follows them is
    not, and the counts are what separate them when the markers are written.
    """

    @property
    def layers(self) -> int:
        return self.nodes.shape[1] - 1


def _envelope_radius(envelope: Any, station: float, limit: float = 1.0e3) -> float | None:
    r"""Radius of the shock envelope at an axial station, by bisection.

    The providers expose a *signed* membership test -- positive outside the
    shock, negative inside -- rather than a radius, because a general envelope
    need not be a function of :math:`x` at all. Bisecting that test is the one
    way to get a radius out of any of them without knowing which it is, and it
    is cheap: forty halvings on a bracket found by doubling.

    ``None`` when the station lies ahead of the shock, where there is no radius
    to report.
    """
    probe = np.zeros((1, 3), dtype=np.float64)

    def outside(radius: float) -> bool:
        probe[0, 0] = station
        probe[0, 2] = radius
        return bool(np.asarray(envelope.at(probe))[0] > 0.0)

    if outside(0.0):
        return None
    low, high = 0.0, 1.0e-3
    while high < limit and not outside(high):
        low, high = high, high * 2.0
    if not outside(high):
        return None
    for _ in range(40):
        mid = 0.5 * (low + high)
        if outside(mid):
            high = mid
        else:
            low = mid
    return float(0.5 * (low + high))


def refine_base_disc(
    vertices: np.ndarray,
    base: np.ndarray,
    rings: int | None = None,
    rim_cell: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Replace the base triangulation with a radial grid of the same rim.

    The body's own base disc is a **fan**: one centre node joined to every rim
    node, so the base radius is spanned by a single cell. Measured on the
    sphere-cone, the base carries two distinct radii, 0 and 0.4117 m.

    Swept downstream that becomes a wake whose core cannot hold a recirculation
    bubble -- there is nothing there to resolve it with. In the first
    wake-resolved run the core held **three nodes** between x = 3.0 and 3.5 m,
    the minimum axial velocity anywhere behind the base was +465 m/s, and the
    reversed flow that defines a separated base never appeared. The shear layer
    was resolved beautifully by the 2640-node annulus and the recirculation not
    at all.

    The rim is kept **exactly** -- the same nodes, in the same places -- because
    it is welded to the annulus, which is welded to the forebody. Only the
    interior is added, as ``rings`` rings interpolated toward the rim's
    centroid, so this works on a base that is not a circle.

    ``rings`` defaults to the isotropic choice. With :math:`n` nodes around a
    rim of radius :math:`R`, the circumferential spacing is :math:`2\pi R/n`,
    so matching it radially wants :math:`R/(2\pi R/n) = n/2\pi` rings -- eight
    on a 48-node rim, which also matches the axial step the sweep starts at.

    Returns ``(points, quads, triangles)``: the new interior points appended
    after the originals, the ring quadrilaterals, and the centre fan.
    """
    rim = boundary_edges(base)
    if not len(rim):
        return np.empty((0, 3)), np.empty((0, 4), dtype=np.int64), base
    nodes = np.unique(rim)
    count = max(1, rings if rings is not None else round(len(nodes) / (2.0 * np.pi)))
    if count < 2:
        return np.empty((0, 3)), np.empty((0, 4), dtype=np.int64), base

    centre = vertices[nodes].mean(axis=0)
    # **Graded**, fine at the rim. Uniform rings put a 49 mm cell against the
    # annulus's 0.66 mm one -- a 75x jump at the exact radius where the boundary
    # layer leaves the shoulder and becomes the free shear layer. The Mach 6
    # run produced 65 000 and 72 000 K spikes in the wake because of it, in the
    # same way the 841x jump at the shock produced them at the nose.
    #
    # The first ring therefore matches the scale arriving from the annulus and
    # the series grows inward, where nothing is happening and cells may be
    # coarse. Solved with the same `growth_for` the wall-normal grading uses.
    radius = float(np.linalg.norm(vertices[nodes] - centre, axis=1).max())
    first = min(rim_cell if rim_cell else radius / count, 0.5 * radius)
    try:
        from aether.cfd.gridding import growth_for

        ratio = growth_for(first, radius, count)
    except (ImportError, ValueError):
        ratio = 1.0
    steps = geometric_spacing(first, radius, count, ratio)
    # `geometric_spacing` runs 0 -> 1 from the *rim*; the fraction wanted is how
    # far out each ring sits, so it counts down from 1.
    fractions = 1.0 - np.asarray(steps)

    offset = len(vertices)
    added: list[np.ndarray] = []
    index = {0: {int(n): int(n) for n in nodes}}
    width = len(nodes)
    for k in range(1, count):
        fraction = float(fractions[k])
        ring = centre + (vertices[nodes] - centre) * fraction
        # Stride by the ring's **width**, not by the number of rings appended so
        # far: the latter aliases every ring onto nearly the same indices and
        # collapses eight of them into three.
        index[k] = {int(n): offset + (k - 1) * width + i for i, n in enumerate(nodes)}
        added.append(ring)
    apex = offset + (count - 1) * width
    added.append(centre.reshape(1, 3))

    quads: list[list[int]] = []
    fan: list[list[int]] = []
    for a, b in rim:
        for k in range(count - 1):
            quads.append(
                [index[k][int(a)], index[k][int(b)],
                 index[k + 1][int(b)], index[k + 1][int(a)]]
            )
        fan.append([index[count - 1][int(a)], index[count - 1][int(b)], apex])
    return (
        np.vstack(added),
        np.asarray(quads, dtype=np.int64),
        np.asarray(fan, dtype=np.int64),
    )


def wake_block(
    surface: Any,
    layer: ExtrudedLayer,
    envelope: Any = None,
    length: float | None = None,
    diameters: float = 8.0,
    layers: int = 40,
    growth: float = 1.08,
    isotropic: bool = True,
    base_rings: int | None = None,
) -> WakeBlock | None:
    """Sweep the aft face downstream, so the base and near wake are computed.

    A shock-fitted forebody block ends at the base plane. That is what LAURA and
    DPLR do and it is right for forebody aerodynamics -- everything outside the
    bow shock is freestream, so meshing it buys nothing. What it cannot do is
    give you a **base pressure**, because there is no wake in the domain: the
    coefficient has to come from a correlation instead, which is what
    ``closure.base_axial_coefficient`` has been supplying all along.

    Industry adds an afterbody block for exactly this. The cap swept downstream
    is the base disc (a **wall**) plus the annulus the forebody block's outflow
    face traces from the base rim out to the shock envelope, so the two blocks
    share a face and the domain is continuous across it.

    The sweep is along ``+x`` rather than along normals: the wake is not a
    boundary layer, nothing is being resolved normal to a surface, and a uniform
    direction cannot produce crossing columns.

    Three things this has to get right, and a straight sweep gets none of them
    ----------------------------------------------------------------------------

    **It has to flare.** The bow shock keeps opening downstream -- 13 degrees on
    the Mach 8 sphere-cone -- so a block swept at constant radius ends up
    *inside* it, and imposing freestream on shocked gas is the same defect the
    forebody envelope exists to avoid. ``envelope`` is the shock surface the
    forebody block was built against; the outer rim follows it, and the flare is
    blended to nothing at the axis so the base disc keeps its own size.

    **Its length is a physical scale, not a fraction of the body.** The near
    wake closes a few base diameters downstream and the recirculation is longer
    still, so the reach is ``diameters`` times the **base** diameter -- eight by
    default, far enough that the outflow does not sit on the recirculation. A
    multiple of body length is the wrong ruler: it makes a stubby body's wake
    short and a slender body's needlessly long.

    **Its cells have to be near-isotropic.** A detached-eddy model resolves the
    large scales in the separated region, and that is only meaningful on cells
    that are not badly stretched: a long thin cell filters the very structures
    DDES exists to capture, and the model then behaves like a poor RANS. With
    ``isotropic`` the first axial step matches the cap's own edge length, so the
    cells at the base start square and grow slowly.
    """
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    faces = np.asarray(surface.faces, dtype=np.int64)
    centroids = vertices[faces].mean(axis=1)
    aft_plane = float(centroids[:, 0].max())
    base = faces[np.isclose(centroids[:, 0], aft_plane, atol=1e-6)]
    if not len(base):
        return None

    # A body closed at its aft end still triangulates to a sliver there, and a
    # sliver is not a base: Sears-Haack yields one face of 1e-6 m^2 and a
    # millimetre of equivalent diameter, which would size a 500 kHz shedding
    # frequency and a 0.1 us time step out of a rounding error. Judged against
    # the body's own width, which is the only scale that makes "negligible"
    # mean anything.
    corners = vertices[base]
    base_area = float(
        0.5
        * np.linalg.norm(
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
            axis=1,
        ).sum()
    )
    equivalent = 2.0 * np.sqrt(max(base_area, 0.0) / np.pi)
    widest = 2.0 * float(np.hypot(vertices[:, 1], vertices[:, 2]).max())
    if widest <= 0.0 or equivalent < 0.02 * widest:
        return None

    # The forebody block's outflow face: its rim edges swept through its layers.
    rim = boundary_edges(layer.faces)
    if not len(rim):
        return None
    columns = layer.nodes
    strip = np.concatenate(
        [
            np.stack(
                [
                    columns[rim[:, 0], k], columns[rim[:, 1], k],
                    columns[rim[:, 1], k + 1], columns[rim[:, 0], k + 1],
                ],
                axis=1,
            )
            for k in range(layer.layers)
        ],
        axis=0,
    )

    # The body's base disc is a fan -- one cell across the whole radius -- and a
    # wake swept from it cannot hold a recirculation bubble. Replaced by a
    # radial grid of the same rim before anything is swept.
    # The scale the shear layer arrives at: the annulus's own first cell, which
    # is the forebody's wall spacing. The base grid starts there and coarsens
    # inward, so the boundary layer does not step across a discontinuity at the
    # exact radius where it separates.
    # Edge [1]->[2] of the strip quad, not [0]->[1]: the first is the
    # wall-normal step the boundary layer is resolved on, the second is the
    # circumferential spacing, and they differ by two orders of magnitude.
    arriving = float(np.min(np.linalg.norm(strip[:, 2] - strip[:, 1], axis=1)))
    rim_cell = max(arriving, 1.0e-5)
    corners = vertices[np.unique(base)]
    span = float(np.linalg.norm(corners - corners.mean(axis=0), axis=1).max())
    if base_rings is None:
        # Enough rings that the growth stays civil: solve for the count that
        # bridges rim cell to base radius at a ratio of about 1.25.
        base_rings = max(8, int(np.ceil(np.log1p(span / rim_cell * 0.25) / np.log(1.25))))
    interior, disc_quads, disc_fan = refine_base_disc(
        vertices, base, rings=base_rings, rim_cell=rim_cell
    )
    if len(interior):
        vertices = np.vstack([vertices, interior])
        base_faces_local = disc_fan
    else:
        disc_quads = np.empty((0, 4), dtype=np.int64)
        base_faces_local = base

    # One node list for the cap: the base disc's own vertices, then the strip's.
    used = np.unique(
        np.concatenate([base_faces_local.ravel(), disc_quads.ravel()]).astype(np.int64)
    )
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))

    # One dedup across **both** halves of the cap, not one per half. The base
    # disc's rim and the annulus's inner rim are the same circle of points, and
    # welding them separately leaves two coincident rings with different
    # indices: the base-disc prisms then never join the annulus hexes, and the
    # 1920 quads between them are boundary faces belonging to no marker -- a
    # slit running right around the base, inside a mesh that reports no bad
    # cells at all.
    strip_points = strip.reshape(-1, 3)
    everything = np.vstack([vertices[used], strip_points])
    unique, inverse = np.unique(np.round(everything, 12), axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    triangles = inverse[remap[base_faces_local]]
    annulus = inverse[len(used) + np.arange(len(strip_points))].reshape(-1, 4)
    # The base's own ring quads join the annulus quads: both sweep to hexahedra,
    # and both must be quads so the interfaces stay one face against one face.
    quads = (
        np.vstack([inverse[remap[disc_quads]], annulus])
        if len(disc_quads)
        else annulus
    )
    base_quad_count = len(disc_quads)
    points = unique

    # --- how far downstream, from the base scale rather than the body's -------
    base_corners = vertices[base]  # the original fan still spans the true area
    base_area = float(
        0.5
        * np.linalg.norm(
            np.cross(
                base_corners[:, 1] - base_corners[:, 0],
                base_corners[:, 2] - base_corners[:, 0],
            ),
            axis=1,
        ).sum()
    )
    base_diameter = float(2.0 * np.sqrt(max(base_area, 0.0) / np.pi))
    reach = float(length) if length is not None else diameters * base_diameter
    if reach <= 0.0:
        return None

    # --- axial spacing: square cells at the base, growing slowly --------------
    edges = np.concatenate(
        [
            np.linalg.norm(points[triangles[:, 1]] - points[triangles[:, 0]], axis=1),
            np.linalg.norm(points[quads[:, 1]] - points[quads[:, 0]], axis=1),
        ]
    )
    first = float(np.median(edges)) if isotropic and len(edges) else reach / (layers * 2.0)
    first = min(max(first, reach * 1.0e-4), reach / layers)
    steps = geometric_spacing(first, reach, layers, growth)

    # --- flare, blended from the rim inward -----------------------------------
    axial = points[:, 0]
    lateral = points[:, 1:]
    radius = np.hypot(lateral[:, 0], lateral[:, 1])
    outer = float(radius.max())
    # The flare belongs to the **annulus** alone. Weighting it by r/r_max spread
    # the base disc outward too -- 45 % of a threefold expansion on the
    # sphere-cone -- which carries the core resolution away from the axis just
    # where a closing wake needs it. The shock opens; the base does not, and the
    # recirculation behind it is about a base diameter wide however far
    # downstream the shock has got to.
    inner = float(np.hypot(*vertices[np.unique(base)][:, 1:].T).max())
    span = max(outer - inner, 1.0e-12)
    weight = np.clip((radius - inner) / span, 0.0, 1.0)

    nodes = np.empty((len(points), len(steps), 3), dtype=np.float64)
    for k, step in enumerate(steps * reach if steps.max() <= 1.0 else steps):
        x = axial + step
        scale = 1.0
        if envelope is not None and outer > 0.0:
            here = _envelope_radius(envelope, float(np.median(axial) + step))
            there = _envelope_radius(envelope, float(np.median(axial)))
            if here is not None and there is not None and there > 0.0:
                scale = float(here / there)
        grow = 1.0 + weight * (scale - 1.0)
        nodes[:, k, 0] = x
        nodes[:, k, 1] = lateral[:, 0] * grow
        nodes[:, k, 2] = lateral[:, 1] * grow
    return WakeBlock(
        nodes=nodes,
        faces=triangles,
        quads=quads,
        base_faces=len(triangles),
        base_quads=base_quad_count,
    )


# --------------------------------------------------------------- writing it out

#: VTK cell types, as SU2 reads them from a ``.su2`` file.
TRIANGLE, QUADRILATERAL, HEXAHEDRON, PRISM = 5, 9, 12, 13


def boundary_edges(faces: np.ndarray) -> np.ndarray:
    """Edges used by exactly one face -- the rim left by removing the base.

    Swept outward these become the downstream boundary of the block, so getting
    them wrong leaves a hole in the domain rather than a bad cell.
    """
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    ordered = np.sort(edges, axis=1)
    _, index, counts = np.unique(ordered, axis=0, return_index=True, return_counts=True)
    return edges[index[counts == 1]]


def _prism_volumes(nodes: np.ndarray) -> np.ndarray:
    """Signed volumes of ``(n, 6, 3)`` prisms, by the standard three-tet split."""
    a, b, c, d, e, f = (nodes[:, i, :] for i in range(6))
    def tet(p0, p1, p2, p3):
        return np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0
    return tet(a, b, c, d) + tet(b, c, d, e) + tet(c, d, e, f)


def _hex_volumes(nodes: np.ndarray) -> np.ndarray:
    """Signed volume of each hexahedron, by splitting it into six tetrahedra.

    Only the sign is load-bearing here: a negative one means the swept quad was
    wound the other way, and the whole block flips together or not at all.
    """
    a, b, c, d, e, f, g, h = (nodes[:, k, :] for k in range(8))
    tets = (
        (a, b, d, e), (b, c, d, g), (b, d, e, g),
        (b, e, f, g), (d, e, g, h), (b, c, g, f),
    )
    total = np.zeros(len(nodes), dtype=np.float64)
    for p0, p1, p2, p3 in tets:
        total += np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0
    return total


def _cap_boundary_edges(triangles: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Edges on the rim of the **combined** cap -- triangles and quads together.

    Taken over the quads alone it returns two rings: the annulus's outer rim,
    which is a real boundary, and its inner rim, which is not -- the base disc
    is welded to it. Sweeping both declares 1920 interior faces as farfield, and
    SU2 is then asked to impose freestream in the middle of the wake.

    The cap is one surface. Its boundary is the outer rim and nothing else.
    """
    edges = np.concatenate(
        [
            triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]],
            quads[:, [0, 1]], quads[:, [1, 2]], quads[:, [2, 3]], quads[:, [3, 0]],
        ],
        axis=0,
    )
    ordered = np.sort(edges, axis=1)
    _, index, counts = np.unique(ordered, axis=0, return_index=True, return_counts=True)
    return edges[index[counts == 1]]


def _join_wake(
    prisms: np.ndarray,
    flat: np.ndarray,
    wake: WakeBlock,
    wall_faces: np.ndarray,
    far_faces: np.ndarray,
    tolerance: float = 1.0e-9,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, int]]:
    """Weld the afterbody block onto the forebody block and retag the boundaries.

    The weld is by coordinate, because that is the only thing the two blocks
    share: the wake's cap is *built* from the forebody's column nodes, so every
    strip node has an exact twin already in ``flat`` and must become that index
    rather than a new one. A duplicate there is not a small error -- SU2 reads
    two coincident faces as two boundaries and the domain leaks between them,
    which no cell-quality measure can see.

    Retagging, in the order the physics dictates:

    * the base disc, which was no boundary at all, becomes **wall** -- it now
      carries a computed pressure instead of a correlation;
    * the forebody's outflow annulus becomes **interior**, because the wake sits
      against it;
    * the wake's downstream face becomes the new **outflow**;
    * the wake's outer rim joins the **farfield**, where the flared envelope
      keeps it outside the shock.
    """
    cap, levels = wake.nodes.shape[0], wake.nodes.shape[1]
    faces = np.asarray(wake.faces, dtype=np.int64)

    # --- weld level 0 onto whatever already exists at the same coordinate ----
    #
    # By proximity, not by a quantised hash. Rounding coordinates into buckets
    # fails whenever a pair straddles a bucket edge, and two points 1e-12 apart
    # then land in different cells and are never joined: sixteen node pairs
    # survived that way, leaving thirty-two coincident quads just aft of the
    # base plane -- two faces at the same place, each used once, which is a slit
    # rather than an interface.
    from scipy.spatial import cKDTree

    tree = cKDTree(flat)
    distance, nearest = tree.query(wake.nodes[:, 0, :], k=1)
    matched = distance <= tolerance

    points = [flat]
    offset = len(flat)
    remap = np.empty((cap, levels), dtype=np.int64)
    welded = int(matched.sum())
    remap[matched, 0] = nearest[matched]
    unmatched = np.flatnonzero(~matched)
    remap[unmatched, 0] = offset + np.arange(len(unmatched))
    if len(unmatched):
        points.append(wake.nodes[unmatched, 0, :])
    offset += len(unmatched)
    for level in range(1, levels):
        remap[:, level] = offset + np.arange(cap)
        points.append(wake.nodes[:, level, :])
        offset += cap
    merged = np.vstack(points)

    if welded == 0:
        msg = (
            "the wake block shares no node with the forebody block, so the two "
            "are separate meshes that happen to touch; the cap must be built "
            "from the forebody's own column nodes"
        )
        raise ValueError(msg)

    # --- sweep: triangles give prisms, quads give hexahedra -------------------
    lower = np.concatenate([remap[faces, k] for k in range(levels - 1)], axis=0)
    upper = np.concatenate([remap[faces, k + 1] for k in range(levels - 1)], axis=0)
    wake_prisms = np.concatenate([lower, upper], axis=1)
    volumes = _prism_volumes(merged[wake_prisms])
    if np.all(volumes < 0.0):
        wake_prisms = np.concatenate([upper, lower], axis=1)
    elif np.any(volumes < 0.0):
        msg = (
            f"{int(np.sum(volumes < 0))} of {len(volumes)} wake prisms are "
            "inverted while others are not, so the cap winding is inconsistent"
        )
        raise ValueError(msg)

    quad_cap = np.asarray(wake.quads, dtype=np.int64)
    q_low = np.concatenate([remap[quad_cap, k] for k in range(levels - 1)], axis=0)
    q_high = np.concatenate([remap[quad_cap, k + 1] for k in range(levels - 1)], axis=0)
    wake_hexes = np.concatenate([q_low, q_high], axis=1)
    if _hex_volumes(merged[wake_hexes]).min() < 0.0:
        wake_hexes = np.concatenate([q_high, q_low], axis=1)

    # --- boundaries -----------------------------------------------------------
    base = remap[faces[: wake.base_faces], 0]          # the base fan, now wall
    base_rings = (
        remap[quad_cap[: wake.base_quads], 0]
        if wake.base_quads
        else np.empty((0, 4), dtype=np.int64)
    )
    downstream_tri = remap[faces, levels - 1]           # outflow, triangles
    downstream_quad = remap[quad_cap, levels - 1]       # outflow, quads
    # The block's outer skin is the boundary of the *quad* annulus only: the
    # base disc is interior to the cap, so its rim is welded to the annulus and
    # is not an outer edge.
    rim = _cap_boundary_edges(faces, quad_cap)
    outer = np.concatenate(
        [
            np.stack(
                [
                    remap[rim[:, 0], k], remap[rim[:, 1], k],
                    remap[rim[:, 1], k + 1], remap[rim[:, 0], k + 1],
                ],
                axis=1,
            )
            for k in range(levels - 1)
        ],
        axis=0,
    )
    # The base disc under its **own** tag, not folded into the vehicle wall.
    # Both are walls and SU2 would solve them identically, but a base pressure
    # that cannot be separated from the forebody force is not a base pressure --
    # and `AERO_COEFF_SURF`, which writes the per-marker coefficients, is only
    # emitted when a base marker exists. Resolving the base and then summing it
    # back into the forebody would leave exactly the number the correlation was
    # supplying, with more steps.
    extra = {
        "wall": wall_faces,
        "wall_quads": np.empty((0, 4), dtype=np.int64),
        "base": base,
        "base_quads": base_rings,
        "farfield": far_faces,
        "farfield_quads": outer,
        "outflow": downstream_tri,
        "outflow_quads": downstream_quad,
    }
    counts = {
        "welded": welded,
        "wake_prisms": len(wake_prisms),
        "wake_hexes": len(wake_hexes),
    }
    return (
        np.concatenate([prisms, wake_prisms], axis=0),
        wake_hexes,
        merged,
        extra,
        counts,
    )


def write_su2(layer: ExtrudedLayer, path: Any, wall: str = "vehicle",
              farfield: str = "farfield", outflow: str = "outflow",
              wake: WakeBlock | None = None, base: str = "vehicle_base") -> dict[str, int]:
    """Write the extruded block as an SU2 mesh. Returns the element counts.

    Three boundaries, and the third is the one worth explaining. The block is
    cut off at the base plane, where the flow is **supersonic and outgoing**, so
    every characteristic leaves the domain and SU2's characteristic far-field
    condition extrapolates all of them out. That makes ``MARKER_FAR`` correct
    there; it is emitted under its own tag anyway so the choice is visible in
    the config rather than hidden in the mesh.

    With a ``wake`` the third boundary is not a boundary at all
    ---------------------------------------------------------

    The afterbody block's inlet face **is** the forebody block's outflow face,
    so joining them turns that surface into an interior one and the flow leaves
    through the wake's downstream end instead. The base disc joins the wall,
    which is the whole point: a resolved base carries a pressure, and a pressure
    on a wall is a force, where before it was
    ``closure.base_axial_coefficient`` -- an empirical :math:`C_{p,b} = -1/M^2`
    worth 7 % of total drag at Mach 10 and **32 % at Mach 4**.

    The two blocks have to share nodes, not merely touch. The strip the wake
    sweeps from is built out of the forebody's own column nodes, so those arrive
    duplicated by coordinate and are welded back onto their originals here; the
    base disc's vertices are genuinely new, because ``extrude_to_shock``
    excludes the base and the forebody block has a hole where it sits. A weld
    that misses leaves two meshes in contact, which passes every quality check
    and leaks.

    Without a ``wake`` the base disc is not a boundary of this mesh at all -- it
    was never extruded, because that pipeline models the base rather than
    resolving it → :mod:`aether.cfd.forces`.
    """
    from pathlib import Path as _Path

    nodes, faces = layer.nodes, layer.faces
    n_surface, n_levels = nodes.shape[0], nodes.shape[1]
    layers = n_levels - 1

    def index(vertex: np.ndarray, level: int) -> np.ndarray:
        return level * n_surface + vertex

    # Prisms: the surface triangle at one level, the same triangle at the next.
    lower = np.concatenate([index(faces, k) for k in range(layers)], axis=0)
    upper = np.concatenate([index(faces, k + 1) for k in range(layers)], axis=0)
    prisms = np.concatenate([lower, upper], axis=1)

    flat = np.concatenate([nodes[:, k, :] for k in range(n_levels)], axis=0)

    volumes = _prism_volumes(flat[prisms])
    if np.any(volumes < 0.0):
        # Reversing the two triangles swaps the sign; do it once, globally,
        # rather than per cell -- a mesh with mixed orientation is a bug, not a
        # thing to patch cell by cell.
        if np.all(volumes < 0.0):
            prisms = np.concatenate([upper, lower], axis=1)
            volumes = _prism_volumes(flat[prisms])
        else:
            msg = (
                f"{int(np.sum(volumes < 0))} of {len(volumes)} prisms are inverted "
                "while others are not, so the surface winding is inconsistent "
                "rather than merely reversed"
            )
            raise ValueError(msg)

    rim = boundary_edges(faces)
    sides = np.concatenate(
        [
            np.stack(
                [
                    index(rim[:, 0], k), index(rim[:, 1], k),
                    index(rim[:, 1], k + 1), index(rim[:, 0], k + 1),
                ],
                axis=1,
            )
            for k in range(layers)
        ],
        axis=0,
    )

    wall_faces = index(faces, 0)
    far_faces = index(faces, layers)
    boundaries: list[tuple[str, np.ndarray, int]] = []

    hexes = np.empty((0, 8), dtype=np.int64)
    if wake is None:
        boundaries = [
            (wall, wall_faces, TRIANGLE),
            (farfield, far_faces, TRIANGLE),
            (outflow, sides, QUADRILATERAL),
        ]
        counts = {wall: len(faces), farfield: len(faces), outflow: len(sides)}
    else:
        prisms, hexes, flat, extra, counts = _join_wake(
            prisms, flat, wake, wall_faces, far_faces
        )
        volumes = _prism_volumes(flat[prisms])
        boundaries = [
            (wall, extra["wall"], TRIANGLE),
            (wall, extra["wall_quads"], QUADRILATERAL),
            (base, extra["base"], TRIANGLE),
            (base, extra["base_quads"], QUADRILATERAL),
            (farfield, extra["farfield"], TRIANGLE),
            (farfield, extra["farfield_quads"], QUADRILATERAL),
            (outflow, extra["outflow"], TRIANGLE),
            (outflow, extra["outflow_quads"], QUADRILATERAL),
        ]
        counts = {
            wall: len(extra["wall"]) + len(extra["wall_quads"]),
            base: len(extra["base"]) + len(extra["base_quads"]),
            farfield: len(extra["farfield"]) + len(extra["farfield_quads"]),
            outflow: len(extra["outflow"]) + len(extra["outflow_quads"]),
        }

    out = ["NDIME= 3", f"NELEM= {len(prisms) + len(hexes)}"]
    out += [
        f"{PRISM} " + " ".join(map(str, row)) + f" {i}" for i, row in enumerate(prisms)
    ]
    out += [
        f"{HEXAHEDRON} " + " ".join(map(str, row)) + f" {len(prisms) + i}"
        for i, row in enumerate(hexes)
    ]
    out.append(f"NPOIN= {len(flat)}")
    out += [f"{x:.16g} {y:.16g} {z:.16g} {i}" for i, (x, y, z) in enumerate(flat)]
    # One MARKER_TAG per boundary, with both face kinds under it: SU2 reads a
    # repeated tag as a second marker and then complains it was not configured.
    grouped: dict[str, list[tuple[np.ndarray, int]]] = {}
    for tag, cells, kind in boundaries:
        if len(cells):
            grouped.setdefault(tag, []).append((cells, kind))
    out.append(f"NMARK= {len(grouped)}")
    for tag, parts in grouped.items():
        total = sum(len(cells) for cells, _ in parts)
        out += [f"MARKER_TAG= {tag}", f"MARKER_ELEMS= {total}"]
        for cells, kind in parts:
            out += [f"{kind} " + " ".join(map(str, row)) for row in cells]
    _Path(path).write_text("\n".join(out) + "\n")
    return {
        "prisms": len(prisms),
        "hexes": len(hexes),
        "points": len(flat),
        **counts,
        "min_volume": float(volumes.min()),
    }
