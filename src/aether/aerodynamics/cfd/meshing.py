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

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["BodyProfile", "DomainSizing", "MeshResult", "axisymmetric_domain", "cone_profile"]

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

    @property
    def representative_size(self) -> float:
        """:math:`h = 1/\\sqrt{N}` — the length scale a 2-D grid study needs.

        Richardson extrapolation is written in a mesh spacing, and for an
        unstructured grid the only defensible one is the average cell size
        implied by the count. In two dimensions that is the inverse square
        root, not the inverse cube root.
        """
        return float(1.0 / np.sqrt(max(self.n_elements, 1)))


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
