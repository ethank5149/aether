"""Axisymmetric flow domains, built from a body profile with gmsh.

A body of revolution at zero incidence is a two-dimensional problem, and that
is the whole reason the transonic gap is closeable at all: the same physics
that would need ten million cells in three dimensions needs thirty thousand
in the meridian plane, and a Mach sweep that would be a month becomes an
afternoon.

The domain
----------

The meridian half-plane, with the body sitting on the axis:

.. code-block:: text

     y
     |    +---------------------------- farfield -----------+
     |    |                                                 |
     |    |            _____________________                |
     |    |        ,--'                     |               |
     |    +-------o----- body ------------- + base          + outlet
     |    inlet   ^ nose                    |               |
     +----+-------+------- axis ------------+---- axis -----+---> x

Four boundary groups, and each is a different physical statement:

``axis``
    The line :math:`y = 0` fore and aft of the body. Under
    ``AXISYMMETRIC= YES`` this is where SU2 applies the singular-axis
    treatment; it is a symmetry marker, not a wall.
``wall``
    Body surface plus the base disc. A slip wall for Euler.
``farfield``
    Everything else. Characteristic boundary conditions, so the same domain
    works subsonic, transonic and hypersonic without changing type.

Sizing
------

Three length scales, all set relative to the body diameter so a sweep over
configurations does not need re-tuning:

* **Nose.** The stagnation region carries the largest gradients on the body
  and is where an under-resolved mesh shows up first, as a pressure
  coefficient that does not reach :math:`C_{p,\\max}`.
* **Shock envelope.** A supersonic solution puts a bow shock somewhere
  between the body and about one body length off it, and a shock smeared
  over twenty cells is a shock in the wrong place. The refinement box
  follows the Mach cone: its transverse extent is set from
  :math:`1/\\sqrt{M^2-1}`, so it narrows as the Mach number rises rather
  than wasting cells on a region the shock has left.
* **Farfield.** Coarse, and *far* — thirty body lengths by default. A
  transonic solution is sensitive to domain size in a way a supersonic one
  is not, because disturbances propagate upstream; a domain sized for Mach 3
  and reused at Mach 0.9 gives a blockage error that looks like physics.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "BodyProfile",
    "BoundaryLayerTruncated",
    "DomainSizing",
    "MeshResult",
    "RefinementBall",
    "ViscousSizing",
    "axisymmetric_domain",
    "boundary_layer_thickness",
    "cone_profile",
    "inviscid_domain",
    "profile_from_arrays",
    "resolvable_radius",
    "shock_layer_cell_size",
    "shock_layer_sizing",
    "shock_standoff",
    "stagnation_refinement",
    "viscous_domain",
    "wall_spacing_for_y_plus",
]

_FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BodyProfile:
    """The meridian outline of a body of revolution.

    Attributes
    ----------
    station:
        Axial coordinate (m), strictly increasing, starting at the nose.
    radius:
        Radius at each station (m). The first entry is the tip and must be
        zero — a blunt nose is expressed as a very small first radius with a
        near-vertical first segment, not as a non-zero tip, because the mesh
        generator needs to know where the axis attaches.
    """

    station: _FloatArray
    radius: _FloatArray
    name: str = "body"

    def __post_init__(self) -> None:
        x = np.asarray(self.station, dtype=np.float64)
        r = np.asarray(self.radius, dtype=np.float64)
        if x.ndim != 1 or x.size < 3:
            msg = f"profile needs at least 3 stations, got shape {x.shape}"
            raise ValueError(msg)
        if r.shape != x.shape:
            msg = f"radius shape {r.shape} does not match station shape {x.shape}"
            raise ValueError(msg)
        if np.any(np.diff(x) <= 0.0):
            msg = "stations must be strictly increasing from the nose"
            raise ValueError(msg)
        if r[0] != 0.0:
            msg = f"the profile must start on the axis; got radius {r[0]:g} at the tip"
            raise ValueError(msg)
        if np.any(r < 0.0):
            msg = "radii must be non-negative"
            raise ValueError(msg)
        if r[-1] <= 0.0:
            msg = (
                "the profile must end at a base of non-zero radius; a body "
                "closing to a point has no base disc and the domain would be "
                "degenerate there"
            )
            raise ValueError(msg)

    @property
    def length(self) -> float:
        return float(self.station[-1] - self.station[0])

    @property
    def base_radius(self) -> float:
        return float(self.radius[-1])

    @property
    def maximum_radius(self) -> float:
        return float(np.max(self.radius))

    @property
    def reference_area(self) -> float:
        """:math:`\\pi r_{\\max}^2` (m²) — the maximum cross-section."""
        return float(np.pi * self.maximum_radius**2)

    @classmethod
    def from_mesh(
        cls,
        mesh: Any,
        n_stations: int = 160,
        name: str = "body",
    ) -> BodyProfile:
        """Extract the meridian outline from a :class:`~aether.geometry.VehicleMesh`.

        The mesh's own stations are the design stations and are irregularly
        spaced — dense where the geometry has features and empty over metres
        of parallel barrel. Resampling onto a cosine-clustered grid puts
        points where curvature is, which is at the nose and at the shoulders,
        and is what the mesh generator wants anyway.
        """
        stations, radii = mesh.station_profile()
        stations = np.asarray(stations, dtype=np.float64)
        radii = np.asarray(radii, dtype=np.float64)
        # The mesh's body axis may run either way; orient nose-first, which is
        # the end with the smallest radius.
        if radii[0] > radii[-1]:
            stations, radii = -stations[::-1], radii[::-1]
        stations = stations - stations[0]

        # Cosine clustering toward both ends.
        fraction = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, int(n_stations))))
        sampled_x = stations[0] + fraction * (stations[-1] - stations[0])
        sampled_r = np.interp(sampled_x, stations, radii)
        sampled_r[0] = 0.0
        return cls(station=sampled_x, radius=sampled_r, name=name)


def cone_profile(half_angle: float, length: float = 1.0, n_stations: int = 60) -> BodyProfile:
    """A sharp cone — the geometry the CFD is validated on.

    Its exact answer is :func:`aether.aerodynamics.conical.solve_cone`, and
    the point of having a generator for it here is that the *same* meshing
    and solver path produces both the validation case and the production
    case. A validation that runs through a different code path validates that
    code path.
    """
    angle = float(half_angle)
    if not 0.0 < angle < 0.5 * np.pi:
        msg = f"cone half-angle must be in (0, 90) deg, got {np.rad2deg(angle):g}"
        raise ValueError(msg)
    x = np.linspace(0.0, float(length), int(n_stations))
    return BodyProfile(station=x, radius=x * np.tan(angle), name="cone")


@dataclass(frozen=True)
class DomainSizing:
    """Domain extent and cell sizes, all relative to the body's diameter.

    Attributes
    ----------
    upstream, downstream, transverse:
        Domain extent in body lengths (upstream, downstream of the base) and
        body diameters (transverse).
    nose_size, wall_size, shock_size, farfield_size:
        Target cell sizes as fractions of the body diameter.
    shock_margin:
        How far beyond the Mach cone the refined region extends, in body
        diameters. The cone is where the shock is for a slender body; the
        margin covers a blunt nose's detached bow shock, which stands off
        ahead of it.
    """

    upstream: float = 8.0
    downstream: float = 8.0
    transverse: float = 12.0
    nose_size: float = 0.004
    wall_size: float = 0.02
    shock_size: float = 0.05
    farfield_size: float = 2.0
    shock_margin: float = 0.6
    nose_refinement_length: float = 0.15
    """Length of the nose refinement zone, in body diameters."""

    def scaled(self, factor: float) -> DomainSizing:
        """The same domain at a different resolution — the grid-convergence knob.

        Multiplies every cell size by ``factor`` and leaves every extent
        alone. A grid-convergence study that also changed the domain size
        would be measuring two things at once and reporting one number.
        """
        scale = float(factor)
        if not (np.isfinite(scale) and scale > 0.0):
            msg = f"refinement factor must be finite and > 0, got {factor}"
            raise ValueError(msg)
        return DomainSizing(
            upstream=self.upstream,
            downstream=self.downstream,
            transverse=self.transverse,
            nose_size=self.nose_size * scale,
            wall_size=self.wall_size * scale,
            shock_size=self.shock_size * scale,
            farfield_size=self.farfield_size * scale,
            shock_margin=self.shock_margin,
            nose_refinement_length=self.nose_refinement_length,
        )


@dataclass(frozen=True)
class MeshResult:
    """A written ``.su2`` mesh and what it cost."""

    path: Path
    n_nodes: int
    n_elements: int
    sizing: DomainSizing
    mach: float
    dimension: int = 2
    """Spatial dimension of the domain. Sets how ``representative_size`` is formed."""
    n_prisms: int = 0
    """Boundary-layer cells, when the mesh has a boundary layer. Zero otherwise."""
    first_cell_height: float = float("nan")
    """Wall-normal size of the first cell (m), or NaN for an inviscid mesh."""

    @property
    def representative_size(self) -> float:
        """:math:`h = N^{-1/d}` — the length scale a grid study needs.

        Richardson extrapolation is written in a mesh spacing, and for an
        unstructured grid the only defensible one is the average cell size
        implied by the count. In two dimensions that is the inverse square
        root and in three the inverse cube root; using the 2-D form on a 3-D
        mesh reports an observed order that is wrong by a factor of 3/2, which
        looks like a plausible order and is not one.
        """
        return float(max(self.n_elements, 1) ** (-1.0 / float(self.dimension)))


def axisymmetric_domain(
    profile: BodyProfile,
    path: str | Path,
    mach: float = 3.0,
    sizing: DomainSizing | None = None,
    verbose: bool = False,
) -> MeshResult:
    """Build and write the meridian-plane domain around ``profile``.

    Parameters
    ----------
    mach:
        Used only for sizing — it aims the refined region along the Mach
        cone. The mesh itself is valid at any Mach number; it is merely
        wasteful at one far from what it was sized for.

    Notes
    -----
    gmsh is a process-global singleton, so this initialises and finalises it
    around each call rather than leaving it open. That costs a few
    milliseconds and means two meshes built in the same session cannot
    corrupt each other's model, which a long checkpointed sweep will
    eventually try to do.
    """
    try:
        import gmsh
    except ImportError as error:  # pragma: no cover - dependency declared
        msg = "axisymmetric meshing needs gmsh (pip install gmsh)"
        raise ImportError(msg) from error

    sizing = sizing if sizing is not None else DomainSizing()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    diameter = 2.0 * profile.maximum_radius
    length = profile.length
    nose_x = float(profile.station[0])
    base_x = float(profile.station[-1])

    x_min = nose_x - sizing.upstream * length
    x_max = base_x + sizing.downstream * length
    y_max = sizing.transverse * diameter

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add(profile.name)
        geo = gmsh.model.geo

        coarse = sizing.farfield_size * diameter
        fine = sizing.wall_size * diameter

        # --- boundary points, walked anticlockwise around the fluid --------
        inlet_bottom = geo.addPoint(x_min, 0.0, 0.0, coarse)
        body_points = [
            geo.addPoint(float(x), float(r), 0.0, fine)
            for x, r in zip(profile.station, profile.radius, strict=True)
        ]
        base_bottom = geo.addPoint(base_x, 0.0, 0.0, fine)
        outlet_bottom = geo.addPoint(x_max, 0.0, 0.0, coarse)
        outlet_top = geo.addPoint(x_max, y_max, 0.0, coarse)
        inlet_top = geo.addPoint(x_min, y_max, 0.0, coarse)

        axis_forward = geo.addLine(inlet_bottom, body_points[0])
        # A spline through the profile rather than a polyline: the surface
        # pressure is integrated against d(r^2), and a faceted body would put
        # a step in dr/dx at every station and a corresponding spike in the
        # pressure the solver has to resolve.
        body_curve = geo.addSpline(body_points)
        base_line = geo.addLine(body_points[-1], base_bottom)
        axis_aft = geo.addLine(base_bottom, outlet_bottom)
        outlet = geo.addLine(outlet_bottom, outlet_top)
        top = geo.addLine(outlet_top, inlet_top)
        inlet = geo.addLine(inlet_top, inlet_bottom)

        loop = geo.addCurveLoop([axis_forward, body_curve, base_line, axis_aft, outlet, top, inlet])
        surface = geo.addPlaneSurface([loop])
        geo.synchronize()

        gmsh.model.addPhysicalGroup(1, [axis_forward, axis_aft], name="axis")
        gmsh.model.addPhysicalGroup(1, [body_curve, base_line], name="wall")
        gmsh.model.addPhysicalGroup(1, [outlet, top, inlet], name="farfield")
        gmsh.model.addPhysicalGroup(2, [surface], name="fluid")

        _apply_sizing(gmsh, profile, sizing, mach, body_curve, base_line, diameter)

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.removeDuplicateNodes()
        gmsh.write(str(destination))

        node_tags, _, _ = gmsh.model.mesh.getNodes()
        _, element_tags, _ = gmsh.model.mesh.getElements(2)
        n_elements = int(sum(len(tags) for tags in element_tags))
        return MeshResult(
            path=destination,
            n_nodes=len(node_tags),
            n_elements=n_elements,
            sizing=sizing,
            mach=float(mach),
        )
    finally:
        gmsh.finalize()


def _apply_sizing(
    gmsh: Any,
    profile: BodyProfile,
    sizing: DomainSizing,
    mach: float,
    body_curve: int,
    base_line: int,
    diameter: float,
) -> None:
    """Distance-and-threshold size fields: nose, wall, shock envelope."""
    fields: list[int] = []

    wall_distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(wall_distance, "CurvesList", [body_curve, base_line])
    gmsh.model.mesh.field.setNumber(wall_distance, "Sampling", 400)

    wall_threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(wall_threshold, "InField", wall_distance)
    gmsh.model.mesh.field.setNumber(wall_threshold, "SizeMin", sizing.wall_size * diameter)
    gmsh.model.mesh.field.setNumber(wall_threshold, "SizeMax", sizing.farfield_size * diameter)
    gmsh.model.mesh.field.setNumber(wall_threshold, "DistMin", 0.15 * diameter)
    gmsh.model.mesh.field.setNumber(wall_threshold, "DistMax", 6.0 * diameter)
    fields.append(wall_threshold)

    # Nose: a ball rather than a distance field, because the stagnation
    # region needs the fine size everywhere inside it and not just on the
    # surface — the bow shock stands off ahead of the tip.
    nose_ball = gmsh.model.mesh.field.add("Ball")
    gmsh.model.mesh.field.setNumber(nose_ball, "XCenter", float(profile.station[0]))
    gmsh.model.mesh.field.setNumber(nose_ball, "YCenter", 0.0)
    gmsh.model.mesh.field.setNumber(nose_ball, "ZCenter", 0.0)
    gmsh.model.mesh.field.setNumber(nose_ball, "Radius", sizing.nose_refinement_length * diameter)
    gmsh.model.mesh.field.setNumber(nose_ball, "Thickness", 0.4 * diameter)
    gmsh.model.mesh.field.setNumber(nose_ball, "VIn", sizing.nose_size * diameter)
    gmsh.model.mesh.field.setNumber(nose_ball, "VOut", sizing.farfield_size * diameter)
    fields.append(nose_ball)

    # Shock envelope: a wedge along the Mach cone from the nose. Subsonic
    # freestreams have no Mach cone, so the box degenerates to a shallow
    # region around the body, which is the right place for the extra cells
    # transonically anyway.
    spread = 1.0 / np.sqrt(mach * mach - 1.0) if mach > 1.05 else 2.0
    reach = profile.length + sizing.downstream * profile.length
    envelope = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(envelope, "XMin", float(profile.station[0]) - 0.5 * diameter)
    gmsh.model.mesh.field.setNumber(envelope, "XMax", float(profile.station[-1]) + 0.5 * reach)
    gmsh.model.mesh.field.setNumber(envelope, "YMin", 0.0)
    gmsh.model.mesh.field.setNumber(
        envelope,
        "YMax",
        float(spread * profile.length + sizing.shock_margin * diameter),
    )
    gmsh.model.mesh.field.setNumber(envelope, "ZMin", -1.0)
    gmsh.model.mesh.field.setNumber(envelope, "ZMax", 1.0)
    gmsh.model.mesh.field.setNumber(envelope, "VIn", sizing.shock_size * diameter)
    gmsh.model.mesh.field.setNumber(envelope, "VOut", sizing.farfield_size * diameter)
    gmsh.model.mesh.field.setNumber(envelope, "Thickness", 1.5 * diameter)
    fields.append(envelope)

    minimum = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

    # With a background field driving the size, the point-attached and
    # curvature-driven sizes must be switched off or gmsh takes the minimum
    # of all of them and the field stops being the thing in control.
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 5)  # Delaunay


def profile_from_arrays(station: ArrayLike, radius: ArrayLike, name: str = "body") -> BodyProfile:
    """Convenience wrapper so a notebook can build a profile inline."""
    return BodyProfile(
        station=np.asarray(station, dtype=np.float64),
        radius=np.asarray(radius, dtype=np.float64),
        name=name,
    )


# ---------------------------------------------------------------------------
# Three-dimensional viscous domains
# ---------------------------------------------------------------------------
#
# Everything above is the meridian plane, and the meridian plane is only
# available at zero incidence. A vehicle that flies at an angle of attack, or
# whose cross-section is not a circle, is a three-dimensional problem and has
# to be meshed as one.
#
# The expensive part of that is not the volume — it is the boundary layer.
# Wall heat flux and skin friction are set by the gradient at the wall, and
# resolving that gradient means a first cell inside the viscous sublayer,
# y+ of order one, which on a hypersonic vehicle is a few microns against a
# body metres long. Six orders of magnitude of anisotropy cannot be reached by
# refining tetrahedra: the cell count explodes and the elements degenerate.
# It is reached by *extruding prisms* along the wall normal, so the cells are
# thin in the direction the gradient is large and coarse in the two directions
# it is not.


@dataclass(frozen=True)
class ViscousSizing:
    """A three-dimensional domain with a prism boundary layer on the body.

    Attributes
    ----------
    y_plus:
        Target wall coordinate for the *first* cell centre. One is the
        conventional target for a wall-resolved solution; a wall-function
        turbulence model wants 30 or more, and SST as configured here does not
        use one.
    growth_ratio:
        Geometric growth from each layer to the next. Above about 1.3 the
        truncation error of the wall-normal gradient starts to matter; below
        about 1.05 the layer count needed to escape the boundary layer becomes
        the dominant cost.
    n_layers:
        Prism layers. Wanted large enough that the outermost layer is
        comparable to the local tetrahedron size, or the mesh has a
        discontinuity in cell size where the prisms stop —
        :meth:`total_thickness` is what to check that against.
    upstream, downstream, transverse:
        Farfield extent, in body lengths (axial) and body diameters
        (transverse), measured from the body's bounding box.
    farfield_size, wake_size:
        Cell sizes at the outer boundary and in the wake, as fractions of the
        body diameter.
    """

    y_plus: float = 1.0
    growth_ratio: float = 1.2
    n_layers: int = 25
    upstream: float = 8.0
    downstream: float = 10.0
    transverse: float = 10.0
    subsonic_growth: float = 6.0
    """Multiplier applied to every extent below Mach 1. See :meth:`for_mach`."""
    farfield_size: float = 3.0
    wake_size: float = 0.25

    def __post_init__(self) -> None:
        if not (np.isfinite(self.growth_ratio) and self.growth_ratio > 1.0):
            msg = f"growth ratio must be > 1, got {self.growth_ratio}"
            raise ValueError(msg)
        if int(self.n_layers) < 1:
            msg = f"need at least one prism layer, got {self.n_layers}"
            raise ValueError(msg)
        if not (np.isfinite(self.y_plus) and self.y_plus > 0.0):
            msg = f"y_plus must be finite and > 0, got {self.y_plus}"
            raise ValueError(msg)

    def for_mach(self, mach: float) -> ViscousSizing:
        """The same sizing with its extents set for a Mach number.

        Domain size is not a Mach-independent choice, and getting it wrong is
        silent. Supersonically the farfield can sit a few body lengths away
        because nothing propagates upstream: the bow shock confines the
        disturbance and a characteristic boundary just outside it is harmless.
        **Subsonically that is false.** Pressure disturbances reach every
        boundary, and a farfield placed where a Mach 8 case would put it
        produces a blockage error that looks exactly like physics — a lift and
        drag shifted by percent, converging beautifully, wrong.

        For a database spanning Mach 0 to 27 the extents therefore have to move
        with the regime. Below Mach 1 they are grown by ``subsonic_growth``;
        transonically the growth is tapered off between Mach 1 and 1.5, where
        the upstream influence shrinks but has not vanished.
        """
        mach = float(mach)
        if mach >= 1.5:
            factor = 1.0
        elif mach <= 1.0:
            factor = float(self.subsonic_growth)
        else:
            # Linear taper across the transonic band rather than a step: a
            # sweep that jumps domain size between two neighbouring Mach
            # numbers puts a discontinuity into the table that is the mesh's,
            # not the flow's.
            blend = (mach - 1.0) / 0.5
            factor = float(self.subsonic_growth) * (1.0 - blend) + blend
        return ViscousSizing(
            y_plus=self.y_plus,
            growth_ratio=self.growth_ratio,
            n_layers=self.n_layers,
            upstream=self.upstream * factor,
            downstream=self.downstream * factor,
            transverse=self.transverse * factor,
            farfield_size=self.farfield_size * factor,
            wake_size=self.wake_size,
            subsonic_growth=self.subsonic_growth,
        )

    def layer_heights(self, first_cell: float) -> list[float]:
        """Cumulative heights of each layer — what gmsh's extruder wants.

        Cumulative, not per-layer: gmsh reads the vector as the position of
        each layer's outer face measured from the wall, so a geometric series
        of *thicknesses* has to be handed over as its partial sums.
        """
        ratio = float(self.growth_ratio)
        heights: list[float] = []
        height = float(first_cell)
        total = 0.0
        for _ in range(int(self.n_layers)):
            total += height
            heights.append(total)
            height *= ratio
        return heights

    def total_thickness(self, first_cell: float) -> float:
        """How far the prisms reach from the wall (m)."""
        return self.layer_heights(first_cell)[-1]

    def first_cell_for_thickness(self, thickness: float) -> float:
        """Wall cell that makes the stack reach exactly ``thickness`` (m).

        The inverse of :meth:`total_thickness`, which is linear in the wall
        cell, so the whole stack simply scales. Use it when the layer is sized
        to a *geometric* target — a shock standoff — rather than to a viscous
        wall law, where the wall cell is set by :math:`y^+` and the thickness
        is whatever follows.
        """
        unit = self.total_thickness(1.0)
        if unit <= 0.0:
            raise ValueError("degenerate layer stack")
        return float(thickness) / unit

    def layers_to_span(self, first_cell: float, thickness: float) -> int:
        """Layers needed for the stack to reach ``thickness``.

        Inverting the geometric series
        :math:`h_0(r^n - 1)/(r - 1) \\ge \\delta`. Use it to set
        :attr:`n_layers` from :func:`boundary_layer_thickness` rather than by
        eye: the count that spans the layer changes by a factor of two across
        the Mach range, because both the first cell and the layer thickness
        move, and they move in opposite directions.
        """
        ratio = float(self.growth_ratio)
        needed = 1.0 + float(thickness) * (ratio - 1.0) / max(float(first_cell), 1e-300)
        return int(max(1, np.ceil(np.log(needed) / np.log(ratio))))

    def with_layers(self, n_layers: int) -> ViscousSizing:
        """The same sizing with a different layer count."""
        return ViscousSizing(
            y_plus=self.y_plus,
            growth_ratio=self.growth_ratio,
            n_layers=int(n_layers),
            upstream=self.upstream,
            downstream=self.downstream,
            transverse=self.transverse,
            farfield_size=self.farfield_size,
            wake_size=self.wake_size,
        )


def wall_spacing_for_y_plus(
    mach: float,
    reference_length: float,
    temperature: float = 288.15,
    pressure: float = 101325.0,
    y_plus: float = 1.0,
    wall_temperature: float | None = None,
) -> float:
    """First-cell height (m) that lands the first cell centre near ``y_plus``.

    A flat-plate estimate, and worth being explicit about what that means: it
    fixes the *order* of the spacing, which is the part that matters, because
    the difference between a mesh that resolves the sublayer and one that does
    not is a factor of a hundred and not a factor of two. The achieved y+ on a
    real body varies along it — largest just aft of the stagnation region,
    where the boundary layer is thinnest — so this is a starting point to be
    checked against the solution, not a guarantee.

    Turbulent skin friction from the 1/7-power correlation
    :math:`C_f = 0.026\\,\\mathrm{Re}_x^{-1/7}`, evaluated at **Eckert's
    reference temperature**

    .. math::

        T^* = T_\\infty + 0.5(T_w - T_\\infty) + 0.22(T_r - T_\\infty)

    rather than at the freestream. That correction is not optional at
    hypersonic speed: at Mach 10 the recovery temperature is an order of
    magnitude above the freestream, the reference-temperature density is
    correspondingly lower, and an uncorrected estimate asks for a first cell
    roughly an order of magnitude too small — which does not fail, it just
    builds a mesh that costs ten times what it needed to.

    ``wall_temperature`` defaults to the recovery temperature (an adiabatic
    wall), the hottest case and so the one giving the largest reference
    temperature and the coarsest spacing.
    """
    gamma, gas_constant = 1.4, 287.058
    mach = float(mach)
    if not (np.isfinite(mach) and mach > 0.0):
        msg = f"Mach number must be finite and > 0, got {mach}"
        raise ValueError(msg)

    speed = mach * np.sqrt(gamma * gas_constant * temperature)
    # Recovery factor for a turbulent boundary layer, Pr^(1/3) with Pr = 0.71.
    recovery = 0.71 ** (1.0 / 3.0)
    t_recovery = temperature * (1.0 + recovery * 0.5 * (gamma - 1.0) * mach**2)
    t_wall = t_recovery if wall_temperature is None else float(wall_temperature)
    t_star = temperature + 0.5 * (t_wall - temperature) + 0.22 * (t_recovery - temperature)

    def sutherland(t: float) -> float:
        return float(1.716e-5 * (t / 273.15) ** 1.5 * (273.15 + 110.4) / (t + 110.4))

    # Pressure is constant across the layer, so the reference-temperature
    # density follows from the ideal gas law at the same pressure.
    density_star = pressure / (gas_constant * t_star)
    viscosity_star = sutherland(t_star)

    reynolds = density_star * speed * float(reference_length) / viscosity_star
    skin_friction = 0.026 * reynolds ** (-1.0 / 7.0)
    wall_shear = 0.5 * density_star * speed**2 * skin_friction
    friction_velocity = np.sqrt(wall_shear / density_star)
    # y+ is defined on the first cell *centre*, so the cell is twice as tall.
    return float(2.0 * y_plus * viscosity_star / (density_star * friction_velocity))


class BoundaryLayerTruncated(UserWarning):
    """The prism stack had to be shortened to keep the extrusion valid.

    Extruding along the wall normal is only injective while the offset stays
    below the local radius of curvature on a concave patch: past that, normals
    from opposite sides of the concavity cross and the outer surface passes
    through itself. gmsh reports this as ``PLC Error: a segment and a facet
    intersect``, which names the symptom and not the cause.

    A truncated stack is not a failed mesh, but it is a *worse* one, and how
    much worse depends on where the prisms stop relative to the boundary
    layer. If they stop inside it, the wall-normal gradient is being carried
    partly by tetrahedra at the interface. The warning reports the thickness
    achieved against the thickness estimated, so that ratio is visible rather
    than implied.
    """


@dataclass(frozen=True)
class RefinementBall:
    """A sphere of prescribed cell size, imposed on the volume mesh.

    The unit of *local* control. A single global setting cannot serve a body
    with more than one blunt feature: on the caret waverider studied here, the
    curvature setting that gives the 3 mm leading edge its correct 0.79 mm
    cells would have to be seventeen times stronger to give the 50 mm nose its
    0.75 mm, and applying that everywhere drives the leading edge to 45 um and
    the mesh out of reach. Each feature gets its own region instead.

    Attributes
    ----------
    center:
        Centre of the region, in body coordinates (m).
    radius:
        Radius within which ``size`` applies (m).
    size:
        Target cell size inside the region (m). Take it from
        :func:`shock_layer_cell_size` for a feature with a detached shock.
    thickness:
        Width of the transition out to the surrounding size (m). Defaults to
        ``radius``, which spreads the jump over an octave rather than putting
        a discontinuity in the size field where the mesher will make slivers.
    """

    center: tuple[float, float, float]
    radius: float
    size: float
    thickness: float | None = None

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError(f"radius must be positive, got {self.radius}")
        if self.size <= 0.0:
            raise ValueError(f"size must be positive, got {self.size}")


def stagnation_refinement(
    mesh: Any,
    mach: float,
    *,
    nose_radius: float,
    cells: int = 4,
    reach: float = 3.0,
) -> RefinementBall:
    """A :class:`RefinementBall` resolving a blunt nose's captured bow shock.

    Places the region at the body's most upstream point and sizes it from
    :func:`shock_layer_cell_size`, extending ``reach`` standoffs back so the
    shock is resolved where it stands as well as at the wall.

    ``nose_radius`` is passed rather than inferred: the discrete surface's
    curvature at the apex is a property of its triangulation, and reading the
    resolution requirement off the thing being resolved is circular.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    apex = vertices[int(np.argmin(vertices[:, 0]))]
    standoff = shock_standoff(mach, nose_radius, geometry="sphere")
    size = shock_layer_cell_size(mach, nose_radius, cells=cells, geometry="sphere")
    if not np.isfinite(standoff):
        standoff = nose_radius
        size = nose_radius / max(int(cells), 1)
    span = float(nose_radius + reach * standoff)
    return RefinementBall(
        center=(float(apex[0]) + 0.5 * span, float(apex[1]), float(apex[2])),
        radius=span,
        size=float(size),
    )


def shock_standoff(
    mach: float,
    radius: float,
    *,
    geometry: str = "sphere",
    sweep: float = 0.0,
    gamma: float = 1.4,
) -> float:
    r"""Bow-shock standoff distance ahead of a blunt feature (Billig, 1967).

    .. math::
        \frac{\Delta}{R} = k\,\exp\!\left(\frac{c}{M_n^2}\right)

    with :math:`(k, c) = (0.143, 3.24)` for a sphere and :math:`(0.386, 4.67)`
    for a cylinder, and :math:`M_n = M\cos\Lambda` the component normal to a
    swept edge.

    Why this lives in the meshing module
    ------------------------------------
    It is the length that sets the cell size on a blunt nose or leading edge.
    Sizing those from a fraction of the global wall size instead is what let a
    Mach 8 sphere-cone run with **1.5 cells across its nose shock layer**: 8 %
    of the wall nodes then returned a pressure above the Rayleigh pitot limit
    (see :func:`~aether.aerodynamics.closure.rayleigh_pitot_cp_max`) — a
    physically unattainable value — inflating forebody :math:`C_A` by about
    6.5 %, ten times the run's statistical uncertainty. The same sizing left a
    3 mm waverider leading edge with 0.09 cells across its shock layer.

    The distinction between the two correlations is not cosmetic, and neither
    is the sweep. A 3 mm leading edge at Mach 8 stands off 0.45 mm read as an
    unswept sphere and 1.25 mm read as an unswept cylinder. Read correctly —
    as the cylinder it is, swept 83.5 deg on the caret waverider here — the
    normal Mach number is 0.90 and **no bow shock forms at all**, so the cell
    size there is set by resolving the fillet arc and not by a standoff. Using
    the sphere form on that edge understates the required size by a factor of
    order ten and misdiagnoses a resolved edge as hopeless.

    Parameters
    ----------
    mach:
        Freestream Mach number.
    radius:
        Local radius of the blunt feature (m) — nose radius, or the leading
        edge fillet radius.
    geometry:
        ``"sphere"`` for a nose cap, ``"cylinder"`` for an edge.
    sweep:
        Leading-edge sweep (rad), measured from normal to the flow. Only the
        normal Mach component drives the shock.

    Returns
    -------
    float
        Standoff distance in metres, or ``inf`` where the normal Mach number
        is subsonic and no bow shock forms.
    """
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}")
    if geometry not in ("sphere", "cylinder"):
        raise ValueError(f"geometry must be 'sphere' or 'cylinder', got {geometry!r}")
    del gamma  # correlation is fitted for air; kept for signature symmetry
    normal_mach = float(mach) * np.cos(float(sweep))
    if normal_mach <= 1.0:
        return float("inf")
    k, c = (0.143, 3.24) if geometry == "sphere" else (0.386, 4.67)
    return float(radius * k * np.exp(c / normal_mach**2))


def shock_layer_cell_size(
    mach: float,
    radius: float,
    *,
    cells: int = 4,
    geometry: str = "sphere",
    sweep: float = 0.0,
) -> float:
    """Cell size that puts ``cells`` cells across a blunt feature's shock layer.

    Four, measured rather than assumed. Ten is the conventional figure for a
    captured (non-fitted) bow shock, and on this pipeline it is unaffordable —
    ten cells across a Mach 8 sphere-cone's 7.5 mm standoff extrapolates to
    about 4M isotropic elements and 66 hours a run.

    What the sweep found, judged by
    :func:`aether.aerodynamics.cfd.diagnostics.pitot_limit_violation`, which
    needs no reference solution:

    ==========  ==========  ============  ==================
    nose cells  elements    peak C_p      above pitot limit
    ==========  ==========  ============  ==================
    0 (none)    77801       2.501         0.30 %
    4           561261      1.761         none
    ==========  ==========  ============  ==================

    At four cells the peak wall pressure is *below* the Rayleigh limit of
    1.827, where it must be, and that run was also the first to satisfy the
    force-convergence test. Refining further buys no physics: the criterion is
    already met, and the remaining discretisation error belongs to the global
    grid sequence rather than to the nose.

    Note what the same table says about geometry. The 0-cell row was 8.05 %
    over the limit with a peak of 3.579 while the body was built from sampled
    polylines, and 0.30 % with a peak of 2.501 once it was built from exact
    arcs — at the same cell count. Most of an apparent resolution problem was
    a geometry-representation problem.
    """
    standoff = shock_standoff(mach, radius, geometry=geometry, sweep=sweep)
    if not np.isfinite(standoff):
        return float("inf")
    return standoff / max(int(cells), 1)


def resolvable_radius(
    cell_size: float,
    mach: float,
    *,
    cells: int = 4,
    geometry: str = "cylinder",
    sweep: float = 0.0,
) -> float:
    """Smallest blunt radius a given cell size can actually resolve.

    The inverse of :func:`shock_layer_cell_size`, and the honest way to choose
    an edge radius. Bluntness is a free parameter of these bodies; picking one
    below this value does not produce a sharper vehicle, it produces an
    unresolved one, and the resulting leading-edge pressures are set by the
    mesh rather than by the flow.
    """
    normal_mach = float(mach) * np.cos(float(sweep))
    if normal_mach <= 1.0:
        return 0.0
    k, c = (0.143, 3.24) if geometry == "sphere" else (0.386, 4.67)
    return float(cell_size * max(int(cells), 1) / (k * np.exp(c / normal_mach**2)))


def boundary_layer_thickness(
    mach: float,
    distance: float,
    temperature: float = 288.15,
    pressure: float = 101325.0,
    wall_temperature: float | None = None,
) -> float:
    """Turbulent boundary-layer thickness (m) a distance ``x`` from the nose.

    :math:`\\delta = 0.37\\,x\\,\\mathrm{Re}_x^{-1/5}`, evaluated at Eckert's
    reference temperature for the same reason
    :func:`wall_spacing_for_y_plus` is — at hypersonic speed the layer is far
    thicker than the incompressible correlation says, because the gas in it is
    hot and thin. At Mach 10 the correction is roughly a factor of three, and a
    prism stack sized without it stops well inside the layer it was meant to
    span.

    This is what :attr:`ViscousSizing.n_layers` should be chosen against:
    prisms that stop short of :math:`\\delta` leave the wall-normal gradient to
    tetrahedra, which is the thing the boundary layer existed to avoid.
    """
    gamma, gas_constant = 1.4, 287.058
    mach = float(mach)
    speed = mach * np.sqrt(gamma * gas_constant * temperature)
    recovery = 0.71 ** (1.0 / 3.0)
    t_recovery = temperature * (1.0 + recovery * 0.5 * (gamma - 1.0) * mach**2)
    t_wall = t_recovery if wall_temperature is None else float(wall_temperature)
    t_star = temperature + 0.5 * (t_wall - temperature) + 0.22 * (t_recovery - temperature)
    viscosity = 1.716e-5 * (t_star / 273.15) ** 1.5 * (273.15 + 110.4) / (t_star + 110.4)
    density = pressure / (gas_constant * t_star)
    reynolds = max(density * speed * float(distance) / viscosity, 1.0)
    return float(0.37 * float(distance) * reynolds ** (-0.2))


def shock_layer_sizing(
    mach: float,
    nose_radius: float,
    sizing: ViscousSizing,
    *,
    cells: int = 16,
    span: float = 1.5,
    growth: float = 1.05,
) -> tuple[ViscousSizing, float]:
    """Size a prism stack to resolve a captured bow shock, not a viscous layer.

    Returns ``(sizing, first_cell_height)`` ready for :func:`viscous_domain`.

    Why this exists
    ---------------
    Resolving a bow shock is a *wall-normal* requirement, and paying for it
    with isotropic cells means filling a solid ball around the nose with them.
    Measured on a Mach 8 sphere-cone, whose 50 mm nose stands its shock off
    7.5 mm: 8 cells across that standoff cost 2.0M elements and 10 cells
    extrapolate to about 4M, against 77k unrefined — a run of roughly 66 hours
    where the unrefined one took 20 minutes. Stretched cells buy the same
    resolution for ``cells x n_faces`` prisms; on the same body, 7448 faces and
    16 layers is about 119k prisms, comparable to the *4-cell* isotropic mesh
    while resolving thirteen cells across the standoff.

    ``span`` puts the outermost layer beyond the shock so it is captured
    inside the structured stack rather than on the seam where the prisms hand
    over to tetrahedra.

    Why the growth ratio is near one
    --------------------------------
    A viscous stack clusters hard at the wall, because a boundary layer's
    gradients live there and :math:`y^+` sets the first cell. A **captured
    shock has no such preference**: it sits somewhere out in the stack and
    wants uniform resolution across the standoff. Carrying the viscous default
    of 1.2 over to a shock-layer stack put the wall cell at 0.129 mm under a
    5 mm surface cell and gave SU2 a control-volume face-area aspect ratio of
    30834 and a sub-volume ratio of 21897 — six non-physical points before the
    first iteration, and a Mach 8 Euler solution that diverged at iteration 207
    with the geometry otherwise clean. At 1.05 the same sixteen layers put
    twelve cells through the standoff with a wall cell near 0.48 mm, an aspect
    ratio around ten.

    This is the same argument that makes every hypersonic code march layers,
    and the reason it is worth doing properly: see the note on hyperbolic
    marching in :func:`_marching_vectors`.
    """
    standoff = shock_standoff(mach, nose_radius, geometry="sphere")
    if not np.isfinite(standoff):
        standoff = float(nose_radius)
    layered = replace(sizing.with_layers(int(cells)), growth_ratio=float(growth))
    return layered, layered.first_cell_for_thickness(float(span) * standoff)


def _split_su2_marker(
    path: Path,
    wall_marker: str,
    centroids: _FloatArray,
    aft: NDArray[np.bool_],
) -> None:
    """Split a written ``.su2`` wall marker into forebody and base patches.

    Done on the file rather than in the mesher because it is a *labelling*
    operation: the wall triangulation is handed to gmsh as a discrete surface
    and comes back unchanged, so every wall element in the file is one of the
    input faces under a different number, and the split is exact. Attempting
    the same thing topologically — two discrete surfaces, two prism stacks —
    fails to weld along the rim; see the note in :func:`_build_viscous_domain`.

    Elements are matched to input faces by centroid, on a rounded grid rather
    than by nearest neighbour, so a mismatch raises instead of silently
    assigning a face to the wrong patch and corrupting the force split.
    """
    text = path.read_text().split("\n")
    index = 0
    points: _FloatArray | None = None
    while index < len(text):
        line = text[index].strip()
        if line.startswith("NPOIN="):
            count = int(line.split("=")[1].split()[0])
            points = np.array(
                [
                    [float(value) for value in text[index + 1 + offset].split()[:3]]
                    for offset in range(count)
                ],
                dtype=np.float64,
            )
            break
        index += 1
    if points is None:
        raise ValueError(f"{path} has no NPOIN block")

    lookup = {
        tuple(np.round(centroid, 9)): bool(flag)
        for centroid, flag in zip(centroids, aft, strict=True)
    }

    out: list[str] = []
    index = 0
    replaced = False
    while index < len(text):
        line = text[index]
        stripped = line.strip()
        if stripped.startswith("NMARK="):
            out.append(f"NMARK= {int(stripped.split('=')[1]) + 1}")
            index += 1
            continue
        if stripped.startswith("MARKER_TAG=") and stripped.split("=")[1].strip() == wall_marker:
            count = int(text[index + 1].split("=")[1])
            rows = text[index + 2 : index + 2 + count]
            fore: list[str] = []
            base: list[str] = []
            for row in rows:
                nodes = [int(token) for token in row.split()[1:4]]
                key = tuple(np.round(points[nodes].mean(axis=0), 9))
                flag = lookup.get(key)
                if flag is None:
                    raise ValueError(
                        f"wall element at {key} in {path.name} matches no input "
                        f"face; the surface was re-meshed and the base split "
                        f"cannot be trusted"
                    )
                (base if flag else fore).append(row)
            out.append(f"MARKER_TAG= {wall_marker}")
            out.append(f"MARKER_ELEMS= {len(fore)}")
            out.extend(fore)
            out.append(f"MARKER_TAG= {wall_marker}_base")
            out.append(f"MARKER_ELEMS= {len(base)}")
            out.extend(base)
            index += 2 + count
            replaced = True
            continue
        out.append(line)
        index += 1
    if not replaced:
        raise ValueError(f"{path} has no marker named {wall_marker!r}")
    path.write_text("\n".join(out))


def _base_faces(
    mesh: Any,
    n_faces: int,
    split_base: bool,
    base_station: float | None,
    base_cone_deg: float = 10.0,
) -> NDArray[np.bool_]:
    """Which wall faces belong to the base patch.

    Shared by both domain builders so the two cannot drift apart: the forces
    are split on whichever marker the mesh was written with, and a viscous
    mesh disagreeing with an inviscid one about where the base starts would
    make the two force histories incomparable.

    The default test is that the outward normal lies within ``base_cone_deg``
    of the axis. The bare *sign* of the axial component is not enough: a caret
    waverider's upper surface is a freestream surface with :math:`n_x \approx
    0`, so its sign is decided by rounding, and the sign test put 1534 of 2916
    faces — 53 % of the body, the entire upper surface — on the base. The same
    body under any threshold from 0.3 to 0.94 returns the same 60 faces, all
    at the base station, and the two blunt bodies are likewise identical
    between 0.90 and 0.94. The answer is insensitive to the threshold and very
    sensitive to having one.

    Ten degrees, not twenty, and the difference is measurable rather than a
    matter of taste. On the sphere-cone the analytic base disc is 0.46009 m^2;
    the patch recovered at 2, 5 and 10 degrees is 0.46008-0.46010, and at 20
    degrees it is 0.47055 — 2.3 % high, because the tolerance has begun eating
    into the shoulder fillet, whose flow is attached and whose pressure the
    Euler solution gets right. Charging that ring to the base would hand it to
    a correlation instead. The same holds on the biconic, and the waverider's
    base is flat enough that 5, 10 and 20 degrees return the identical 88
    faces.

    The wide default was harmless while the fillet was a band of sliver
    triangles that mostly failed the test anyway; resolving the fillet properly
    is what made the tolerance matter.

    This draws the base at the base *plane*, leaving the shoulder fillet on the
    forebody. ``base_station`` overrides the whole test with an axial cut for
    bodies whose base is not normal to the axis.
    """
    if not split_base and base_station is None:
        return np.zeros(n_faces, dtype=bool)
    centroids = np.asarray(mesh.centroids, dtype=np.float64)
    if base_station is not None:
        aft = centroids[:, 0] >= float(base_station)
        if not aft.any() or aft.all():
            msg = (
                f"base_station {base_station:g} puts {int(aft.sum())} of "
                f"{aft.size} faces on the base; it belongs just forward of "
                f"the base disc, inside the body's axial extent"
            )
            raise ValueError(msg)
        return aft
    normals = np.asarray(mesh.normals, dtype=np.float64)
    lengths = np.linalg.norm(normals, axis=1)
    axial = np.divide(normals[:, 0], lengths, out=np.zeros_like(lengths), where=lengths > 0.0)
    aft = axial > float(np.cos(np.deg2rad(base_cone_deg)))
    if not aft.any():
        msg = (
            f"no wall face points aft within {base_cone_deg:g} deg of the axis, "
            "so there is no base to split off; the body closes to a point, or "
            "its normals are inward"
        )
        raise ValueError(msg)
    return _aft_component(np.asarray(mesh.faces, dtype=np.int64), centroids, aft)


def _aft_component(
    faces: NDArray[np.int64], centroids: _FloatArray, aft: NDArray[np.bool_]
) -> NDArray[np.bool_]:
    """Keep only the aft-facing faces that actually form the base.

    "Points aft" alone is not a base test. A triangulated nose apex carries a
    fan of near-degenerate slivers whose computed normal is numerically
    meaningless, and on the sphere-cone here seven of them came out with a
    positive axial component — putting faces at ``x = 2e-5`` into the same
    patch as the base disc at ``x = 2.0``.

    Charged to the force split that error was negligible, 0.0018 % of the base
    area and 0.04 % of C_A. Charged to a *mesh* it is fatal: growing one prism
    stack off a patch with a piece at each end of the body fails outright, with
    ``Could not find extruded node`` naming a coordinate 11 mm ahead of the
    nose, which reads as an extrusion bug rather than a classification one.

    So the base is the aft-facing region *connected* to the aft-most face,
    found by flood fill across shared edges. Lobes are kept when they too
    reach within a tenth of a body length of the aft-most point, so a body
    whose base is split by a fin or a notch keeps all of it.
    """
    keep = np.flatnonzero(aft)
    if keep.size == 0:
        return aft
    span = float(centroids[:, 0].max() - centroids[:, 0].min()) or 1.0

    # Edge -> incident aft faces, giving adjacency without an O(n^2) scan.
    edges: dict[tuple[int, int], list[int]] = {}
    for index in keep:
        tri = faces[index]
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edges.setdefault((min(a, b), max(a, b)), []).append(int(index))

    unvisited = set(keep.tolist())
    selected = np.zeros(aft.shape[0], dtype=bool)
    rear = float(centroids[keep, 0].max())
    while unvisited:
        stack = [next(iter(unvisited))]
        component: list[int] = []
        while stack:
            face = stack.pop()
            if face not in unvisited:
                continue
            unvisited.discard(face)
            component.append(face)
            tri = faces[face]
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                for neighbour in edges.get((min(a, b), max(a, b)), ()):
                    if neighbour in unvisited:
                        stack.append(neighbour)
        if float(centroids[component, 0].max()) >= rear - 0.1 * span:
            selected[component] = True
    if not selected.any():
        selected[keep] = True
    return selected


def viscous_domain(
    mesh: Any,
    path: str | Path,
    mach: float,
    sizing: ViscousSizing | None = None,
    temperature: float = 288.15,
    pressure: float = 101325.0,
    wall_temperature: float | None = None,
    first_cell_height: float | None = None,
    wall_marker: str = "vehicle",
    farfield_marker: str = "farfield",
    shrink_on_failure: bool = True,
    split_base: bool = True,
    base_station: float | None = None,
) -> MeshResult:
    """Wrap a closed body in a prism boundary layer and a tetrahedral farfield.

    ``mesh`` is a :class:`~aether.geometry.VehicleMesh`; its triangulation
    **is** the wall mesh, so the surface resolution is whatever the generator
    that built it was asked for, and is not re-meshed here. That is deliberate:
    a parametric body already carries the clustering its designer chose — dense
    at the nose, coarse down the barrel — and re-triangulating it from an STL
    would throw that away and then try to guess it back from dihedral angles.

    The body is handed to gmsh as a single **discrete surface** built from the
    vertex and face arrays directly, rather than written to STL and
    re-imported. Round-tripping through STL means
    ``classifySurfaces``, which splits a smooth closed body into patches by
    dihedral angle; each patch is then extruded independently and the
    extrusions overlap along the seams, which gmsh reports as an invalid
    boundary mesh at the point where the volume mesher gives up.

    ``shrink_on_failure`` retries with a shorter prism stack when the extrusion
    self-intersects, warning :class:`BoundaryLayerTruncated` with what it
    settled for. Normal extrusion is injective only while the offset stays
    under the local concave radius of curvature, and on a real vehicle — a bent
    biconic, a shoulder, a flare — that bound can sit well below the boundary
    layer's own thickness. The alternative to retrying is that a sweep dies at
    the one Mach number whose layer got too thick, hours in, having produced
    nothing.

    Raises
    ------
    ValueError
        If the body is not closed. An open surface has no inside, so "extrude
        along the outward normal" is undefined on it, and the failure downstream
        is a meshed vehicle interior rather than an error.
    """
    sizing = sizing if sizing is not None else ViscousSizing()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if not bool(mesh.is_closed):
        holes = np.asarray(mesh.boundary_stations())
        msg = (
            f"the body is not closed — {holes.size} open boundary station(s)"
            f"{' at x = ' + np.array2string(holes, precision=4) if holes.size else ''}. "
            f"A boundary layer is extruded along the outward normal, which an "
            f"open surface does not have; VehicleMesh.from_surface_grid closes "
            f"the ends for a parametric body."
        )
        raise ValueError(msg)

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    low, high = mesh.bounds
    body_length = float(high[0] - low[0])
    diameter = float(max(high[1] - low[1], high[2] - low[2]))
    if not (body_length > 0.0 and diameter > 0.0):
        msg = f"degenerate body bounding box: length {body_length:g}, diameter {diameter:g}"
        raise ValueError(msg)

    if first_cell_height is None:
        first_cell_height = wall_spacing_for_y_plus(
            mach,
            body_length,
            temperature=temperature,
            pressure=pressure,
            y_plus=sizing.y_plus,
            wall_temperature=wall_temperature,
        )
    requested_layers = int(sizing.n_layers)
    requested_thickness = sizing.total_thickness(first_cell_height)

    # Local termination only. The layer count is never reduced: where a column
    # cannot reach full height it stops individually, which is what an
    # industrial mesher does and what keeps the rest of the wall fully layered.
    # Shortening the layer everywhere to accommodate one region is the crude
    # alternative and it is not offered — it cost this package's biconic 12 of
    # its 22 layers over a defect confined to 2.5 % of the wall.
    #
    # The retry backs off the *safety margin* on the measured fold limit, which
    # tightens the field locally where it is already binding and leaves the
    # unconstrained majority at full height.
    # Relaxation strength is part of the search, not a fixed choice. Smoothing
    # the marching field is what lets the biconic reach full height — it goes
    # from 10 layers to 22 — and the same smoothing makes the waverider fail,
    # because that shape is mostly crease and relaxing its direction field is
    # the wrong thing to do to it. Neither setting is right for every geometry,
    # so the builder tries the strong one first and backs off. The layer
    # *count* is never reduced in any of these attempts.
    # The last three rungs hand gmsh a *scalar* field — our per-node heights
    # applied to its own marching normals — instead of a vector field carrying
    # our directions too. On a crease-dominated shape like the waverider that
    # is the better choice: an area-weighted vertex normal at a sharp edge
    # averages two very different face normals into a direction pointing into
    # the crease, where gmsh handles the edge itself. On the biconic the
    # opposite holds and our smoothed directions are what reach full height.
    last_error: Exception | None = None
    aft = _base_faces(mesh, faces.shape[0], split_base, base_station)
    ladder = (
        (6, 0.6, True),
        (2, 0.6, True),
        (0, 0.6, True),
        (0, 0.6, False),
        (0, 0.4, False),
        (0, 0.25, False),
    )
    for attempt, (passes, safety, vector) in enumerate(ladder):
        try:
            result = _build_viscous_domain(
                vertices=vertices,
                faces=faces,
                low=low,
                high=high,
                body_length=body_length,
                diameter=diameter,
                mach=mach,
                sizing=sizing,
                first_cell_height=first_cell_height,
                target=target,
                name=getattr(mesh, "name", None) or "vehicle",
                wall_marker=wall_marker,
                farfield_marker=farfield_marker,
                safety=safety,
                smoothing_passes=passes,
                vector_field=vector,
                aft=aft,
            )
        except Exception as error:
            last_error = error
            if not shrink_on_failure:
                break
            continue
        if attempt > 0:
            warnings.warn(
                f"boundary layer built with relaxed settings "
                f"({'directed' if vector else 'height-only'} field, {passes} "
                f"smoothing passes, fold margin {safety:g}) after "
                f"{attempt} stricter attempt(s) failed. All "
                f"{requested_layers} layers are present; where a column would "
                f"have folded it stops short of the requested "
                f"{requested_thickness:.4g} m.",
                BoundaryLayerTruncated,
                stacklevel=2,
            )
        return result

    msg = (
        f"could not build a viscous domain for "
        f"{getattr(mesh, 'name', 'the body')} at {requested_layers} layers "
        f"({requested_thickness:.4g} m, first cell {first_cell_height:.3e} m) "
        f"under any combination of marching-field smoothing and fold margin. "
        f"The last gmsh error "
        f"was: {last_error}\n"
        f"A PLC error names the *volume* mesher, not the extruder: the layer "
        f"may extrude cleanly in isolation and fail only when the outer "
        f"tetrahedra are filled against it."
    )
    raise RuntimeError(msg)


def _marching_vectors(
    vertices: _FloatArray,
    faces: NDArray[np.int64],
    passes: int = 6,
    blend: float = 0.5,
    max_turn_deg: float = 20.0,
    feature_deg: float = 40.0,
) -> _FloatArray:
    """Smoothed unit directions for the boundary layer to march along.

    Raw vertex normals are the obvious choice and they are what makes columns
    collide: where the surface turns, adjacent normals diverge or converge and
    their columns cross within a few cells. Every industrial extruder marches
    along a *relaxed* direction field instead — Pointwise, AFLR3, and the
    hyperbolic grid generators all smooth the marching vectors before using
    them.

    Smoothing is **feature-preserving**: neighbours across an edge whose faces
    meet at more than ``feature_deg`` are excluded from the average. A crease
    is not noise to be relaxed away — on a waverider the leading edge is the
    entire point of the shape, and relaxing across it both rounds the feature
    and produces marching directions that fail outright. Unfiltered smoothing
    fixed this package's biconic and broke its waverider in the same change.

    Two further guards, both learned the hard way. Neighbour averaging is skipped where
    the neighbours nearly cancel: at a base rim, cone normals and base normals
    oppose across a single edge, their mean is near zero, and normalising it
    returns a direction with no relation to the surface. And the result is
    **clamped** to ``max_turn_deg`` from the original normal, because
    unclamped smoothing on this package's biconic reversed some normals
    outright — a measured 168 degrees of turn, which would extrude that column
    straight into the body.
    """
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    face_normal = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-300)

    accumulated = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(accumulated, faces[:, corner], face_normal * area[:, None])
    original = accumulated / np.maximum(np.linalg.norm(accumulated, axis=1, keepdims=True), 1e-300)

    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    owner = np.concatenate([np.arange(faces.shape[0])] * 3)
    edges = np.concatenate([edges, edges[:, ::-1]])
    owner = np.concatenate([owner, owner])

    # An edge is a feature when the normals of the two faces meeting along it
    # differ by more than the threshold. Keyed on the sorted node pair, so both
    # directed copies of the edge agree about it.
    key = np.sort(edges[: edges.shape[0] // 2], axis=1)
    lookup: dict[tuple[int, int], list[int]] = {}
    for position, pair in enumerate(map(tuple, key)):
        lookup.setdefault(pair, []).append(position)
    sharp = np.zeros(edges.shape[0] // 2, dtype=bool)
    limit_feature = np.cos(np.deg2rad(float(feature_deg)))
    for positions in lookup.values():
        if len(positions) != 2:
            continue
        left, right = owner[positions[0]], owner[positions[1]]
        if float(np.dot(face_normal[left], face_normal[right])) < limit_feature:
            sharp[positions] = True
    keep = ~np.concatenate([sharp, sharp])
    edges = edges[keep]

    degree = np.bincount(edges[:, 0], minlength=vertices.shape[0]).astype(np.float64)
    degree[degree == 0.0] = 1.0

    marching = original.copy()
    for _ in range(int(passes)):
        neighbour = np.zeros_like(marching)
        for axis in range(3):
            neighbour[:, axis] = np.bincount(
                edges[:, 0],
                weights=marching[edges[:, 1], axis],
                minlength=vertices.shape[0],
            )
        neighbour /= degree[:, None]
        # Where neighbours cancel there is no meaningful average to move toward.
        coherent = np.linalg.norm(neighbour, axis=1) > 0.3
        candidate = np.where(
            coherent[:, None], (1.0 - blend) * marching + blend * neighbour, marching
        )
        length = np.linalg.norm(candidate, axis=1, keepdims=True)
        marching = candidate / np.maximum(length, 1e-300)

    # Clamp the total turn away from the surface normal.
    limit = np.cos(np.deg2rad(float(max_turn_deg)))
    alignment = np.einsum("ij,ij->i", marching, original)
    too_far = alignment < limit
    if np.any(too_far):
        marching[too_far] = original[too_far]
    return marching


def _local_height_scale(
    vertices: _FloatArray,
    faces: NDArray[np.int64],
    total_height: float,
    safety: float = 0.6,
    floor: float = 0.02,
    samples: int = 28,
    smoothing_passes: int = 8,
) -> _FloatArray:
    """Per-node fraction of the boundary-layer height that extrudes without folding.

    This is **local layer termination**, the mechanism industrial meshers use
    (Pointwise T-Rex, NASA's AFLR3): the columns that would degenerate stop
    short while every other column grows to full height. Shortening the layer
    everywhere because one region conflicts is the crude alternative, and it
    cost this package's biconic 12 of its 22 layers over a defect confined to
    2.5 % of the wall.

    The limit is **measured, not estimated**. Offsetting the surface along its
    vertex normals by a distance ``t``, a face folds when its normal reverses;
    the largest ``t`` at which every incident face is still forward-facing is
    that node's safe height. Walking a geometric ladder of ``t`` and keeping
    the last clean rung per node gives the field directly.

    Estimating it from local curvature does not work, and it is worth saying
    why: a curvature-derived field left 99.9 % of the biconic's nodes at full
    height while 115 of its 10,328 faces were already folding at a 20 mm
    offset. The folds are not where curvature is extreme — they are spread
    across the body with a median at mid-length, in the ordinary anisotropy of
    a curvature-adapted surface mesh. Only measurement finds them.

    ``safety`` backs off from the fold point, since a face that is merely
    *nearly* folded still makes a bad prism.

    The field is then **smoothed**, which matters as much as measuring it. Left
    raw it steps from 0.03 to 1.0 between neighbouring nodes, and a layer whose
    height collapses across one edge is as distorted as one that folds.
    Smoothing spreads each constraint over its neighbourhood; taking the
    running minimum keeps every pass conservative, so a node's height can only
    fall as the constraint propagates, never rise above what was measured.
    """
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    face_normal = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-300)

    accumulated = np.zeros_like(vertices)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    for corner in range(3):
        np.add.at(accumulated, faces[:, corner], face_normal * area[:, None])
    vertex_normal = accumulated / np.maximum(
        np.linalg.norm(accumulated, axis=1, keepdims=True), 1e-300
    )

    ladder = np.geomspace(max(total_height, 1e-12) * 1.0e-3, max(total_height, 1e-12), int(samples))
    safe_face = np.full(faces.shape[0], ladder[0])
    for step in ladder:
        moved = vertices + step * vertex_normal
        shifted = moved[faces]
        turned = np.cross(shifted[:, 1] - shifted[:, 0], shifted[:, 2] - shifted[:, 0])
        upright = np.einsum("ij,ij->i", turned, face_normal) > 0.0
        safe_face = np.where(upright, step, safe_face)

    safe_node = np.full(vertices.shape[0], np.inf)
    for corner in range(3):
        np.minimum.at(safe_node, faces[:, corner], safe_face)
    safe_node[~np.isfinite(safe_node)] = total_height

    # A node still upright at the top of the ladder is not constrained and must
    # extrude to full height; backing it off by the safety factor as well would
    # shorten the whole layer on account of nodes never in difficulty, which is
    # global truncation wearing a local disguise.
    unconstrained = safe_node >= total_height * (1.0 - 1.0e-9)
    scaled = np.clip(safety * safe_node / max(total_height, 1e-300), floor, 1.0)
    field = np.where(unconstrained, 1.0, scaled)

    if int(smoothing_passes) > 0:
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        edges = np.concatenate([edges, edges[:, ::-1]])
        degree = np.bincount(edges[:, 0], minlength=vertices.shape[0]).astype(np.float64)
        degree[degree == 0.0] = 1.0
        for _ in range(int(smoothing_passes)):
            neighbour_sum = np.bincount(
                edges[:, 0], weights=field[edges[:, 1]], minlength=vertices.shape[0]
            )
            field = np.minimum(field, neighbour_sum / degree)
    return np.clip(field, floor, 1.0)


def _build_viscous_domain(
    vertices: _FloatArray,
    faces: NDArray[np.int64],
    low: _FloatArray,
    high: _FloatArray,
    body_length: float,
    diameter: float,
    mach: float,
    sizing: ViscousSizing,
    first_cell_height: float,
    target: Path,
    name: str,
    wall_marker: str,
    farfield_marker: str,
    safety: float = 0.6,
    smoothing_passes: int = 6,
    vector_field: bool = True,
    aft: NDArray[np.bool_] | None = None,
) -> MeshResult:
    """One extrusion attempt. Separated so a failure can be retried cleanly.

    gmsh keeps global state, so a failed ``generate`` leaves a half-built model
    behind; the retry has to start from a fresh ``initialize``/``finalize``
    pair rather than from the wreckage of the last one.
    """
    import gmsh

    heights = sizing.layer_heights(first_cell_height)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(name)

        # Split before extruding, not after. The prism stack has to be grown
        # off both patches in one call so the two share the nodes along their
        # seam; extruding them separately gives each its own copy of the rim
        # and leaves a crack down the middle of the boundary layer.
        # The whole closed body is extruded as one discrete surface, and the
        # base marker is separated afterwards on the written file by
        # :func:`_split_su2_marker`.
        #
        # Not extruding the base is the better idea and does not work here:
        # gmsh's boundary-layer extruder lives in the *geo* kernel while the
        # wall is a *discrete* entity, so the un-extruded base disc cannot be
        # named in the geo surface loop that bounds the tetrahedral region.
        # The mesh writes without complaint and SU2 rejects it on load —
        # "The surface element (1, 2) doesn't have an associated volume
        # element" — because no tetrahedron was ever grown against the disc.
        #
        # Superseded note, kept because the measurement stands: growing a stack
        # off the base as well is what
        # the obvious reading of "wrap the body in a boundary layer" gives, and
        # it is wrong twice over. Physically, the base is the one region an
        # Euler solution cannot represent — it is replaced by a correlation
        # downstream — so layers there resolve nothing. Numerically, the two
        # stacks fan apart around the convex shoulder and the tetrahedra left
        # to fill the wedge between them are slivers: on this sphere-cone the
        # volume mesh spanned 4.7e-9 to 8.9 m^3, a ratio of 2e9, with the worst
        # cell at x = 2.0109, r = 0.375 — just aft of the base rim — and SU2
        # took a NaN there after 1159 iterations however low the CFL went.
        #
        # Leaving the base as a plain surface also gives the base *marker*
        # topologically, so the force split needs no file surgery.
        wall = gmsh.model.addDiscreteEntity(2)
        gmsh.model.mesh.addNodes(
            2, wall, list(range(1, vertices.shape[0] + 1)), vertices.ravel().tolist()
        )
        gmsh.model.mesh.addElementsByType(wall, 2, [], (faces + 1).ravel().tolist())

        # A *vector* field, not a scalar one: gmsh reads it as the marching
        # direction itself rather than as a multiplier on its own normals, so
        # both the relaxed direction and the local height come from here.
        scale = _local_height_scale(vertices, faces, heights[-1], safety=safety)
        tags = list(range(1, vertices.shape[0] + 1))
        view = gmsh.view.add("boundary-layer-marching")
        if vector_field:
            marching = _marching_vectors(vertices, faces, passes=smoothing_passes) * scale[:, None]
            gmsh.view.addHomogeneousModelData(
                view,
                0,
                name,
                "NodeData",
                tags,
                marching.ravel().tolist(),
                numComponents=3,
            )
        else:
            gmsh.view.addHomogeneousModelData(view, 0, name, "NodeData", tags, scale.tolist())
        extruded = gmsh.model.geo.extrudeBoundaryLayer(
            [(2, wall)], [1] * len(heights), heights, True, viewIndex=view
        )
        gmsh.model.geo.synchronize()

        # The prisms' outer face is found as the boundary of the boundary-layer
        # volumes with the wall removed, rather than by position in the
        # extruder's return value: that vector interleaves top, volume and
        # lateral entities per input surface, and the number of laterals is not
        # fixed, so indexing into it is only right for the cases it was tried on.
        layer_volumes = [dim_tag for dim_tag in extruded if dim_tag[0] == 3]
        boundary = gmsh.model.getBoundary(layer_volumes, combined=True, oriented=False)
        outer = [dim_tag for dim_tag in boundary if dim_tag != (2, wall)]
        if not outer:
            msg = "the boundary-layer extrusion produced no outer surface"
            raise RuntimeError(msg)

        box = _farfield_box(gmsh, low, high, body_length, diameter, sizing)
        # Two loops, not one. A surface loop is a single closed shell: the
        # farfield box is one, and the boundary layer's outer surface is
        # another that bounds a hole in the fluid. Merging them into one loop
        # is topologically invalid — gmsh tolerates it on some geometries and
        # rejects it on others with a PLC error that names the volume mesher,
        # which is what made this look like a boundary-layer problem for as
        # long as it did. The extrusion was never at fault.
        shell = gmsh.model.geo.addSurfaceLoop(box)
        cavity = gmsh.model.geo.addSurfaceLoop([tag for _, tag in outer])
        volume = gmsh.model.geo.addVolume([shell, cavity])
        gmsh.model.geo.synchronize()

        gmsh.model.addPhysicalGroup(2, [wall], name=wall_marker)
        gmsh.model.addPhysicalGroup(2, box, name=farfield_marker)
        gmsh.model.addPhysicalGroup(3, [tag for _, tag in layer_volumes] + [volume], name="fluid")

        _apply_volume_sizing(gmsh, low, high, body_length, diameter, mach, sizing)
        gmsh.model.mesh.generate(3)

        node_tags, _, _ = gmsh.model.mesh.getNodes()
        prisms = 0
        elements = 0
        for element_type in gmsh.model.mesh.getElementTypes(3):
            count = len(gmsh.model.mesh.getElementsByType(element_type)[0])
            elements += count
            if int(element_type) == 6:  # 6-node prism
                prisms += count
        gmsh.write(str(target))
    finally:
        gmsh.finalize()

    return MeshResult(
        path=target,
        n_nodes=len(node_tags),
        n_elements=int(elements),
        sizing=DomainSizing(
            upstream=sizing.upstream,
            downstream=sizing.downstream,
            transverse=sizing.transverse,
            farfield_size=sizing.farfield_size,
        ),
        mach=float(mach),
        dimension=3,
        n_prisms=int(prisms),
        first_cell_height=float(first_cell_height),
    )


def _farfield_box(
    gmsh: Any,
    low: _FloatArray,
    high: _FloatArray,
    body_length: float,
    diameter: float,
    sizing: ViscousSizing,
) -> list[int]:
    """Six plane surfaces enclosing the body, in the built-in CAD kernel.

    Built point-by-point rather than with an OpenCASCADE box because the
    boundary layer lives in the built-in kernel — that is where
    ``extrudeBoundaryLayer`` puts it — and the two kernels do not share
    entities. A box from the OCC kernel cannot be joined to a built-in surface
    loop, and the volume mesher would be handed two unrelated models.
    """
    x_low = float(low[0]) - sizing.upstream * body_length
    x_high = float(high[0]) + sizing.downstream * body_length
    span = sizing.transverse * diameter
    y_low, y_high = float(low[1]) - span, float(high[1]) + span
    z_low, z_high = float(low[2]) - span, float(high[2]) + span
    size = sizing.farfield_size * diameter

    corners = [
        (x_low, y_low, z_low),
        (x_high, y_low, z_low),
        (x_high, y_high, z_low),
        (x_low, y_high, z_low),
        (x_low, y_low, z_high),
        (x_high, y_low, z_high),
        (x_high, y_high, z_high),
        (x_low, y_high, z_high),
    ]
    points = [gmsh.model.geo.addPoint(x, y, z, size) for x, y, z in corners]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    lines = [gmsh.model.geo.addLine(points[a], points[b]) for a, b in edges]
    faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 4, -9, -0],
        [9, 5, -10, -1],
        [10, 6, -11, -2],
        [11, 7, -8, -3],
    ]
    surfaces = []
    for face in faces:
        loop = gmsh.model.geo.addCurveLoop(
            [lines[abs(i)] * (1 if i >= 0 else -1) for i in face], reorient=True
        )
        surfaces.append(gmsh.model.geo.addPlaneSurface([loop]))
    return surfaces


def _apply_volume_sizing(
    gmsh: Any,
    low: _FloatArray,
    high: _FloatArray,
    body_length: float,
    diameter: float,
    mach: float,
    sizing: ViscousSizing,
) -> None:
    """Refine the shock envelope and the near wake, coarsen everywhere else.

    The same Mach-cone argument as the axisymmetric path: the bow shock lies
    within roughly :math:`1/\\sqrt{M^2-1}` of the body, so the refined box
    narrows as Mach rises instead of paying for a region the shock has left.
    Subsonic and transonic cases get a box sized on the body instead, because
    the cone is imaginary there and there is no shock to follow.
    """
    half_angle = 1.0 / np.sqrt(max(mach**2 - 1.0, 1.0e-6)) if mach > 1.05 else 1.0
    envelope = min(
        float(np.clip(half_angle * body_length, 0.5 * diameter, 4.0 * diameter)),
        sizing.transverse * diameter,
    )

    field = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(field, "VIn", sizing.wake_size * diameter)
    gmsh.model.mesh.field.setNumber(field, "VOut", sizing.farfield_size * diameter)
    gmsh.model.mesh.field.setNumber(field, "XMin", float(low[0]) - 0.5 * diameter)
    gmsh.model.mesh.field.setNumber(field, "XMax", float(high[0]) + 2.0 * body_length)
    gmsh.model.mesh.field.setNumber(field, "YMin", float(low[1]) - envelope)
    gmsh.model.mesh.field.setNumber(field, "YMax", float(high[1]) + envelope)
    gmsh.model.mesh.field.setNumber(field, "ZMin", float(low[2]) - envelope)
    gmsh.model.mesh.field.setNumber(field, "ZMax", float(high[2]) + envelope)
    gmsh.model.mesh.field.setNumber(field, "Thickness", envelope)
    gmsh.model.mesh.field.setAsBackgroundMesh(field)

    # Sizes *are* extended from the boundary here, which looks like it should
    # let the wall's few-micron spacing escape into the farfield and does not:
    # the wall is interior to the prism volume, and the tetrahedral region is
    # bounded by the boundary layer's *outer* surface, whose triangles carry
    # the surface mesh's own sizes. Without this the tets against that surface
    # are sized only by the background field — a jump of four to ninety times
    # across the interface, which is what a volume mesher rejects.
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)


def inviscid_domain(
    mesh: Any,
    path: str | Path,
    mach: float,
    sizing: ViscousSizing | None = None,
    wall_refinement: float = 0.05,
    wall_band: float = 0.5,
    wall_marker: str = "vehicle",
    farfield_marker: str = "farfield",
    split_base: bool = True,
    base_station: float | None = None,
    refinement: Sequence[RefinementBall] = (),
) -> MeshResult:
    """Isotropic tetrahedral domain around a closed body — no boundary layer.

    For an Euler run, and the distinction is not an optimisation. A prism layer
    exists to resolve a viscous wall gradient that an Euler solution does not
    have, and extruding one anyway is actively harmful: it puts cells of
    hundred-to-one aspect ratio where the flow has no gradient to justify them,
    and where the body's radius of curvature is smaller than the layer is thick
    — a 0.2 m layer on a 0.05 m nose radius — the extrusion's own normals
    converge and the layer has to be truncated, leaving badly skewed cells
    exactly at the stagnation region.

    On this package's sphere-cone that combination diverged every Mach 8 case
    that was tried against it, at both first and second order, in a way that
    looked like a solver problem and was a meshing one.

    Cells at the wall take their size from the **surface mesh**, which
    :func:`~aether.geometry.brep.surface_mesh` has already refined by
    curvature. That is what resolves a blunt nose, and it is not optional: with
    the background field alone, near-wall size derives from the body
    *diameter*, and a re-entry body's nose radius is an order of magnitude
    smaller. On this package's sphere-cone — 0.05 m of nose — a three-level
    study had near-wall cells of 0.118, 0.084 and 0.059 m, every one larger
    than the feature. The forebody axial force then climbed monotonically
    *past* the exact answer as the mesh refined, because refinement was still
    discovering the nose rather than converging on it, and the Richardson
    extrapolation of that sequence was meaningless while looking perfectly well
    behaved.

    Farther out, resolution comes from an isotropic distance field:
    ``wall_refinement`` of the body diameter within ``wall_band`` diameters of
    the surface. That gives the shock and the stagnation region cells small in
    *every* direction, which is what a hyperbolic scheme with a limiter wants.

    ``split_base`` divides the wall into two markers — ``<wall_marker>`` and
    ``<wall_marker>_base`` — so the solver reports their forces separately,
    **every iteration**. On by default, and the default is the point: an Euler
    solution has no valid base pressure, so folding the base into a single
    force coefficient does not degrade the answer, it replaces it. Measured on
    a Mach 8 sphere-cone here, the forebody contributes :math:`C_A = +0.069`
    and the base :math:`-0.254`, for a reported drag of :math:`-0.184` — a
    body under thrust. Nothing about that run looks wrong otherwise: it
    converges, and the number is reported to six figures. A body with no blunt
    base simply gets an empty second marker, which costs nothing.

    The division is by **outward normal**: a face is base when it faces aft.
    That is geometric, needs no magic number, and on a blunted shoulder falls
    where the physics does, at the widest point where the fillet's normal turns
    aft and the flow separates. ``base_station`` overrides it with an axial cut,
    which on a 10 mm-shouldered sphere-cone swept a band of cone into the base
    and made the patch 1.14 times the area of the disc.

    The reason is that an Euler solution has no valid base pressure: real base
    flow is set by a separated viscous shear layer the equations do not
    contain, so the solver returns whatever its unresolved base region settles
    at, and that region does not settle. On a Mach 8 sphere-cone the base came
    back at :math:`C_p = +0.34` against a physical value near :math:`-0.02`,
    swamping a forebody worth :math:`+0.08` and turning the reported drag
    negative — and its value moved 27 % between grid levels while the forebody
    moved 2 %. Folded together, that unsteadiness also contaminates any
    convergence test applied to the total force.

    The split has to be made **before** the volume is meshed. Re-tagging faces
    afterwards leaves surface elements with no adjacent volume element and SU2
    rejects the mesh outright. Both entities share one node set, so the wall
    stays conformal across the seam.
    """
    import gmsh

    sizing = sizing if sizing is not None else ViscousSizing()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if not bool(mesh.is_closed):
        msg = "the body must be closed to have an outside to mesh"
        raise ValueError(msg)

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    low, high = mesh.bounds
    body_length = float(high[0] - low[0])
    diameter = float(max(high[1] - low[1], high[2] - low[2]))

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(getattr(mesh, "name", None) or "vehicle")
        aft = _base_faces(mesh, faces.shape[0], split_base, base_station)

        wall = gmsh.model.addDiscreteEntity(2)
        # Every node goes on the first entity; the second refers to them by
        # their global tags, which is what keeps the two patches conformal
        # across the seam instead of each carrying its own copy of it.
        gmsh.model.mesh.addNodes(
            2, wall, list(range(1, vertices.shape[0] + 1)), vertices.ravel().tolist()
        )
        gmsh.model.mesh.addElementsByType(wall, 2, [], (faces[~aft] + 1).ravel().tolist())
        patches = [wall]
        if aft.any():
            disc = gmsh.model.addDiscreteEntity(2)
            gmsh.model.mesh.addElementsByType(disc, 2, [], (faces[aft] + 1).ravel().tolist())
            patches.append(disc)
        gmsh.model.geo.synchronize()

        box = _farfield_box(gmsh, low, high, body_length, diameter, sizing)
        outer = gmsh.model.geo.addSurfaceLoop(box)
        inner = gmsh.model.geo.addSurfaceLoop(patches)
        # Two loops: the body is a hole in the fluid, not a second region.
        volume = gmsh.model.geo.addVolume([outer, inner])
        gmsh.model.geo.synchronize()

        gmsh.model.addPhysicalGroup(2, [wall], name=wall_marker)
        if len(patches) > 1:
            gmsh.model.addPhysicalGroup(2, [patches[1]], name=f"{wall_marker}_base")
        gmsh.model.addPhysicalGroup(2, box, name=farfield_marker)
        gmsh.model.addPhysicalGroup(3, [volume], name="fluid")

        # Nested Box fields on coordinates, not a Distance field on the wall.
        # gmsh's Distance field samples parametric surfaces, and the wall here
        # is a *discrete* entity built from arrays — it returns nothing usable,
        # and a Threshold reading it silently produces no refinement at all.
        # That failure is quiet in the worst way: the mesher succeeds, the mesh
        # is uniform at the farfield size, and the run converges to a drag
        # coefficient with the wrong sign.
        near = gmsh.model.mesh.field.add("Box")
        band = wall_band * diameter
        gmsh.model.mesh.field.setNumber(near, "VIn", wall_refinement * diameter)
        gmsh.model.mesh.field.setNumber(near, "VOut", sizing.farfield_size * diameter)
        gmsh.model.mesh.field.setNumber(near, "XMin", float(low[0]) - band)
        gmsh.model.mesh.field.setNumber(near, "XMax", float(high[0]) + band)
        gmsh.model.mesh.field.setNumber(near, "YMin", float(low[1]) - band)
        gmsh.model.mesh.field.setNumber(near, "YMax", float(high[1]) + band)
        gmsh.model.mesh.field.setNumber(near, "ZMin", float(low[2]) - band)
        gmsh.model.mesh.field.setNumber(near, "ZMax", float(high[2]) + band)
        gmsh.model.mesh.field.setNumber(near, "Thickness", band)

        # A second, wider box follows the shock envelope downstream, on the
        # same Mach-cone argument the axisymmetric path uses.
        envelope = 1.0 / np.sqrt(max(mach**2 - 1.0, 1.0e-6)) if mach > 1.05 else 1.0
        reach = float(np.clip(envelope * body_length, 0.5 * diameter, 4.0 * diameter))
        wake = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(wake, "VIn", sizing.wake_size * diameter)
        gmsh.model.mesh.field.setNumber(wake, "VOut", sizing.farfield_size * diameter)
        gmsh.model.mesh.field.setNumber(wake, "XMin", float(low[0]) - 0.5 * diameter)
        gmsh.model.mesh.field.setNumber(wake, "XMax", float(high[0]) + 2.0 * body_length)
        gmsh.model.mesh.field.setNumber(wake, "YMin", float(low[1]) - reach)
        gmsh.model.mesh.field.setNumber(wake, "YMax", float(high[1]) + reach)
        gmsh.model.mesh.field.setNumber(wake, "ZMin", float(low[2]) - reach)
        gmsh.model.mesh.field.setNumber(wake, "ZMax", float(high[2]) + reach)
        gmsh.model.mesh.field.setNumber(wake, "Thickness", reach)

        # Local feature regions, each a Ball taking precedence through the Min
        # below. These are what resolve a captured bow shock: without one the
        # nose cell size is inherited from the body diameter, and a Mach 8
        # sphere-cone ran with 1.5 cells across its 7.5 mm standoff, putting
        # 8 % of its wall nodes above the Rayleigh pitot limit.
        balls = []
        for region in refinement:
            ball = gmsh.model.mesh.field.add("Ball")
            gmsh.model.mesh.field.setNumber(ball, "XCenter", float(region.center[0]))
            gmsh.model.mesh.field.setNumber(ball, "YCenter", float(region.center[1]))
            gmsh.model.mesh.field.setNumber(ball, "ZCenter", float(region.center[2]))
            gmsh.model.mesh.field.setNumber(ball, "Radius", float(region.radius))
            gmsh.model.mesh.field.setNumber(ball, "VIn", float(region.size))
            gmsh.model.mesh.field.setNumber(ball, "VOut", sizing.farfield_size * diameter)
            gmsh.model.mesh.field.setNumber(
                ball,
                "Thickness",
                float(region.thickness if region.thickness is not None else region.radius),
            )
            balls.append(ball)

        smallest = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(smallest, "FieldsList", [near, wake, *balls])
        gmsh.model.mesh.field.setAsBackgroundMesh(smallest)

        # Cells near the wall inherit the *surface* mesh's own sizes, which are
        # already curvature-refined — 6.7 mm across a 50 mm nose radius against
        # 33 mm along the barrel. Without this the background box overrides
        # them with a single body-diameter-derived size, and a blunt nose is
        # simply never resolved: on the sphere-cone the near-wall cell was
        # larger than the nose radius at every level of a three-level study,
        # and the forebody force climbed monotonically past the exact answer
        # because refinement was still discovering the nose.
        #
        # The viscous path must *not* do this — there the wall spacing is a few
        # microns and propagating it outward would fill the domain — which is
        # why the two builders differ here rather than sharing a default.
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.model.mesh.generate(3)

        node_tags, _, _ = gmsh.model.mesh.getNodes()
        elements = sum(
            len(gmsh.model.mesh.getElementsByType(kind)[0])
            for kind in gmsh.model.mesh.getElementTypes(3)
        )
        gmsh.write(str(target))
    finally:
        gmsh.finalize()

    if aft is not None and bool(np.any(aft)):
        _split_su2_marker(target, wall_marker, vertices[faces].mean(axis=1), np.asarray(aft))

    return MeshResult(
        path=target,
        n_nodes=len(node_tags),
        n_elements=int(elements),
        sizing=DomainSizing(
            upstream=sizing.upstream,
            downstream=sizing.downstream,
            transverse=sizing.transverse,
            farfield_size=sizing.farfield_size,
        ),
        mach=float(mach),
        dimension=3,
        n_prisms=0,
        first_cell_height=float(wall_refinement * diameter),
    )
