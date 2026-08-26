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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "BodyProfile",
    "BoundaryLayerTruncated",
    "DomainSizing",
    "MeshResult",
    "ViscousSizing",
    "axisymmetric_domain",
    "boundary_layer_thickness",
    "cone_profile",
    "profile_from_arrays",
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


def cone_profile(
    half_angle: float, length: float = 1.0, n_stations: int = 60
) -> BodyProfile:
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

        loop = geo.addCurveLoop(
            [axis_forward, body_curve, base_line, axis_aft, outlet, top, inlet]
        )
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
    gmsh.model.mesh.field.setNumbers(
        wall_distance, "CurvesList", [body_curve, base_line]
    )
    gmsh.model.mesh.field.setNumber(wall_distance, "Sampling", 400)

    wall_threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(wall_threshold, "InField", wall_distance)
    gmsh.model.mesh.field.setNumber(wall_threshold, "SizeMin", sizing.wall_size * diameter)
    gmsh.model.mesh.field.setNumber(
        wall_threshold, "SizeMax", sizing.farfield_size * diameter
    )
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
    gmsh.model.mesh.field.setNumber(
        nose_ball, "Radius", sizing.nose_refinement_length * diameter
    )
    gmsh.model.mesh.field.setNumber(nose_ball, "Thickness", 0.4 * diameter)
    gmsh.model.mesh.field.setNumber(nose_ball, "VIn", sizing.nose_size * diameter)
    gmsh.model.mesh.field.setNumber(
        nose_ball, "VOut", sizing.farfield_size * diameter
    )
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


def profile_from_arrays(
    station: ArrayLike, radius: ArrayLike, name: str = "body"
) -> BodyProfile:
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
    t_star = (
        temperature
        + 0.5 * (t_wall - temperature)
        + 0.22 * (t_recovery - temperature)
    )

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

    attempts: list[int] = []
    layers = requested_layers
    while layers >= 1:
        attempts.append(layers)
        if not shrink_on_failure:
            break
        layers = int(layers * 0.8) if int(layers * 0.8) < layers else layers - 1

    last_error: Exception | None = None
    for index, n_layers in enumerate(attempts):
        try:
            result = _build_viscous_domain(
                vertices=vertices,
                faces=faces,
                low=low,
                high=high,
                body_length=body_length,
                diameter=diameter,
                mach=mach,
                sizing=sizing.with_layers(n_layers),
                first_cell_height=first_cell_height,
                target=target,
                name=getattr(mesh, "name", None) or "vehicle",
                wall_marker=wall_marker,
                farfield_marker=farfield_marker,
            )
        except Exception as error:
            last_error = error
            continue
        if index > 0:
            achieved = sizing.with_layers(n_layers).total_thickness(first_cell_height)
            warnings.warn(
                f"boundary layer truncated from {requested_layers} layers "
                f"({requested_thickness:.4g} m) to {n_layers} "
                f"({achieved:.4g} m, {100 * achieved / requested_thickness:.0f} % of "
                f"the requested thickness): the extrusion self-intersected on this "
                f"geometry. Cells beyond {achieved:.4g} m from the wall are "
                f"tetrahedra.",
                BoundaryLayerTruncated,
                stacklevel=2,
            )
        return result

    msg = (
        f"could not extrude a boundary layer on {getattr(mesh, 'name', 'the body')} "
        f"at any layer count down to 1 (first cell {first_cell_height:.3e} m). "
        f"The last gmsh error was: {last_error}"
    )
    raise RuntimeError(msg)


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

        wall = gmsh.model.addDiscreteEntity(2)
        gmsh.model.mesh.addNodes(
            2, wall, list(range(1, vertices.shape[0] + 1)), vertices.ravel().tolist()
        )
        gmsh.model.mesh.addElementsByType(
            wall, 2, [], (faces + 1).ravel().tolist()
        )

        extruded = gmsh.model.geo.extrudeBoundaryLayer(
            [(2, wall)], [1] * len(heights), heights, True
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
        loop = gmsh.model.geo.addSurfaceLoop(box + [tag for _, tag in outer])
        volume = gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()

        gmsh.model.addPhysicalGroup(2, [wall], name=wall_marker)
        gmsh.model.addPhysicalGroup(2, box, name=farfield_marker)
        gmsh.model.addPhysicalGroup(
            3, [tag for _, tag in layer_volumes] + [volume], name="fluid"
        )

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
        (x_low, y_low, z_low), (x_high, y_low, z_low),
        (x_high, y_high, z_low), (x_low, y_high, z_low),
        (x_low, y_low, z_high), (x_high, y_low, z_high),
        (x_high, y_high, z_high), (x_low, y_high, z_high),
    ]
    points = [gmsh.model.geo.addPoint(x, y, z, size) for x, y, z in corners]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    lines = [gmsh.model.geo.addLine(points[a], points[b]) for a, b in edges]
    faces = [
        [0, 1, 2, 3], [4, 5, 6, 7],
        [8, 4, -9, -0], [9, 5, -10, -1],
        [10, 6, -11, -2], [11, 7, -8, -3],
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

    # The background field is the only size authority; without this gmsh also
    # honours the sizes attached to the box corner points and to the extruded
    # boundary-layer nodes, and the wall's few-micron spacing would propagate
    # out into the farfield as a size constraint.
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
