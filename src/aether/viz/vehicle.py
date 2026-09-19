"""Drawable vehicle shapes, taken from the meshes rather than typed out.

The glyph used to be a hand-written formula — a power-law nose over 3.3 m
and a constant 1.5 m barrel — sitting in the animator as
``DEFAULT_MOULD_LINE``. It was the right *idea*: draw the body the mass
model was built from, not a generic cone. It was still a second model of
the vehicle, and like every second model it disagreed with the first. The
bundled reference-heavy mesh has four raised separation rings standing 100 mm proud
at 6.887, 13.565, 23.354 and 33.319 m, and an **aft shroud** where the
radius falls to 1.234 m at 34.42 m and flares back to 1.365 m at the base.
The formula had none of that: it drew a smooth tube, so the one feature
that makes a stack read as a stack — the ring at the joint — was absent
from the picture and the engine end was square.

So the profiles come from the meshes. :func:`mould_line` reads a
:class:`~aether.geometry.mesh.VehicleMesh`'s own station profile — the
distinct vertex stations and the maximum radius at each — and returns it in
the convention everything downstream uses: **metres aft of the tip**, tip at
zero, which is what the applied layer's reference mass model indexes
its stage stations against.

Scaling, and why it is stated
-----------------------------

The MIRV bus and re-entry vehicle meshes are supplied in **arbitrary
units** — the bus is 103 units across and 37 thick, the RV 30 across and 65
long — and no file says what a unit is. They are therefore scaled to stated
physical targets, and the targets are named here rather than buried:

* the **bus** to the diameter of the stage it rides inside, taken from the
  launcher's own mould line at the bus station, so the two cannot disagree;
* the **RV** to a 0.55 m base and 1.75 m length, the published envelope of
  a Mk21-class vehicle.

Both scalings are **anisotropic** where the source proportions do not match
the target, exactly as the launcher mesh's own manifest records for the
same reason: length and diameter are independent measurements and one
uniform factor cannot satisfy both. :class:`ScaledBody` reports the two
factors it used, so a reader can see how much the source shape was
distorted rather than being told it fitted.

What is **not** claimed: that these meshes are accurate models of any real
hardware. They are representative shapes at declared dimensions. The
dimensions are the part that carries meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from aether.geometry.mesh import VehicleMesh, load_stl

__all__ = [
    "BUS_MODEL",
    "REENTRY_VEHICLE_MODEL",
    "STACK_MODEL",
    "ScaledBody",
    "body_from_mesh",
    "default_bus",
    "default_reentry_vehicle",
    "default_stack",
    "mould_line",
]

_FloatArray = NDArray[np.float64]

#: Where the bundled vehicle meshes live, relative to the repository root.
_VEHICLES = Path(__file__).resolve().parents[3] / "data" / "vehicles"

STACK_MODEL = _VEHICLES / "reference-heavy" / "full.stl"
"""The launcher, **full** rather than exterior-only.

``exterior.stl`` and every ``configuration-*.stl`` derived from it
are missing 608 faces, and they are missing them for a reason that is
correct for aerodynamics and wrong for a picture:
:meth:`~aether.geometry.mesh.VehicleMesh.exterior_faces` drops every facet
that cannot see infinity along its own normal, because a panel method
integrating pressure over an internal bulkhead reports +0.89 on the axial
force coefficient at Mach 2. Measured on this mesh the dropped facets are
**434 in the aft two metres** — the inside of the engine shrouds, which is a
cup and therefore encloses its own rays — and **170 straddling the ring at
6.9 m**, which is the payload/bus adapter structure. Both read as holes when
drawn. The profile here is taken from the full mesh; the aero path keeps the
exterior filter.
"""

BUS_MODEL = _VEHICLES / "mirv" / "mirv-bus.stl"
"""The dispenser deck: 103 units across and 37 thick, so its **axis is the
short extent** and has to be named — see :func:`axial_frame`."""

REENTRY_VEHICLE_MODEL = _VEHICLES / "mirv" / "rv-alt2.stl"
"""One of the supplied alternates, chosen on proportions rather than name.

``rv.stl`` is 30 units across and 65 long: a fineness ratio of **2.17**
against the 3.18 of the Mk21-class envelope the RV is scaled to, so fitting
both stated dimensions would stretch it by 1.46 along the axis and visibly
distort it. ``rv-alt2.stl`` is 6.63 across and 20 long — fineness **3.02** —
so it reaches the same envelope on a nearly isotropic scale, and it carries
3,908 facets against 1,090.
"""

#: Published envelope of a Mk21-class re-entry vehicle (m). Used as the
#: scaling target for the supplied RV mesh, which carries no units.
RV_LENGTH = 1.75
RV_DIAMETER = 0.55


@dataclass(frozen=True)
class ScaledBody:
    """One drawable body: its profile, and what was done to get there.

    Attributes
    ----------
    name:
        What it is.
    stations, radii:
        Outer mould line, metres aft of the tip, tip at zero.
    axial_factor, radial_factor:
        Scale factors applied to the source mesh. Equal means the source
        proportions already matched the target; unequal means they did not
        and the shape was distorted to fit both stated dimensions rather
        than one. Reported rather than hidden — see the module note.
    source:
        File the profile came from, or ``None`` for an analytic fallback.
    """

    name: str
    stations: _FloatArray
    radii: _FloatArray
    axial_factor: float = 1.0
    radial_factor: float = 1.0
    source: Path | None = None

    def __post_init__(self) -> None:
        stations = np.asarray(self.stations, dtype=np.float64)
        radii = np.asarray(self.radii, dtype=np.float64)
        if stations.shape != radii.shape or stations.ndim != 1 or stations.size < 2:
            msg = (
                f"stations and radii must be matching 1-D arrays of 2+ points, "
                f"got {stations.shape} and {radii.shape}"
            )
            raise ValueError(msg)
        if np.any(np.diff(stations) < 0.0):
            msg = "stations must be non-decreasing, measured aft of the tip"
            raise ValueError(msg)
        object.__setattr__(self, "stations", stations)
        object.__setattr__(self, "radii", radii)

    @property
    def length(self) -> float:
        """Nose to base (m)."""
        return float(self.stations[-1] - self.stations[0])

    @property
    def diameter(self) -> float:
        """Largest diameter (m), rings included."""
        return 2.0 * float(np.max(self.radii))

    @property
    def frontal_area(self) -> float:
        """Reference area from the largest radius (m^2)."""
        return float(np.pi * np.max(self.radii) ** 2)

    def line(self) -> tuple[_FloatArray, _FloatArray]:
        """``(stations, radii)`` — the pair every drawing routine takes."""
        return self.stations, self.radii

    def section(self, forward: float, aft: float) -> ScaledBody:
        """The part of this body between two stations, keeping its own points."""
        if not aft > forward:
            msg = f"need aft > forward, got {forward} and {aft}"
            raise ValueError(msg)
        inside = (self.stations > forward) & (self.stations < aft)
        cuts = np.concatenate([[forward], self.stations[inside], [aft]])
        return ScaledBody(
            name=f"{self.name}[{forward:g}:{aft:g}]",
            stations=cuts - forward,
            radii=np.interp(cuts, self.stations, self.radii),
            axial_factor=self.axial_factor,
            radial_factor=self.radial_factor,
            source=self.source,
        )


def axial_frame(mesh: VehicleMesh, axis: int | None = None) -> _FloatArray:
    """Vertices with the body's axis on **+x** and centred on that axis.

    Two corrections that :meth:`~aether.geometry.mesh.VehicleMesh.to_body_axes`
    does not make, and both are needed before a radius means anything.

    **The axis is not always the longest extent.** ``VehicleMesh.axis`` takes
    it, which is right for a launcher and wrong for a dispenser deck: the
    supplied bus is 103 units across and 37 thick, so the longest extent is
    its *diameter*. ``axis`` overrides it.

    **The mesh is not centred on its axis.** Radius is measured from the
    line ``y = z = 0``, and the supplied RV and bus meshes sit tens of units
    off it — the RV's profile came back as a body of radius 58 to 73 rather
    than 0 to 15. The transverse coordinates are moved onto the bounding
    box's own centre line, which for a body of revolution is its axis.

    Returns
    -------
    numpy.ndarray
        ``(n, 3)`` with the tip at the origin and the body on **-x**, the
        same sign convention :data:`~aether.viz.scene.NOSE_AXIS` uses.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    index = mesh.axis if axis is None else int(axis)
    others = [column for column in range(3) if column != index]
    out = np.zeros_like(vertices)
    out[:, 0] = vertices[:, index]
    for slot, column in enumerate(others, start=1):
        offset = 0.5 * (vertices[:, column].min() + vertices[:, column].max())
        out[:, slot] = vertices[:, column] - offset
    # The tip has to end up at the **maximum** x, because the translation
    # below puts that end at the origin and stations are then measured aft
    # of it. So flip when the slender end is the one at low x.
    radius = np.hypot(out[:, 1], out[:, 2])
    low, high = out[:, 0].min(), out[:, 0].max()
    cut = 0.1 * (high - low)
    forward = float(radius[out[:, 0] <= low + cut].mean())
    aft = float(radius[out[:, 0] >= high - cut].mean())
    if forward < aft:
        out[:, 0] = -out[:, 0]
    return np.asarray(out - np.array([out[:, 0].max(), 0.0, 0.0]))


def mould_line(
    mesh: VehicleMesh, axis: int | None = None, stations: int = 256
) -> tuple[_FloatArray, _FloatArray]:
    """A mesh's **outer** profile, in metres aft of the tip.

    Not the maximum *vertex* radius per distinct station, which is what
    :meth:`~aether.geometry.mesh.VehicleMesh.station_profile` reports and
    what a first version of this used. A body of revolution meshed for
    export has very sparse axial tessellation — triangles spanning metres —
    so a station whose only vertices belong to an interior feature reports
    that feature as the outer surface. Measured on the bundled launcher, the
    station 34.454 m aft of the tip came back with a radius of **0.288 m**
    between neighbours at 1.234 and 1.299: the engine nozzle throat,
    reported as the outside of a 3 m vehicle. Revolved, that is a hole in
    the tail.

    What is computed instead is the silhouette: at each station, the largest
    transverse distance reached by any mesh **edge** crossing it. An edge
    spanning the gap contributes the radius of its own crossing point, so
    the long triangles that caused the problem are exactly what fixes it.
    """
    vertices = axial_frame(mesh, axis)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    start = vertices[edges[:, 0]]
    end = vertices[edges[:, 1]]

    grid = np.linspace(vertices[:, 0].min(), 0.0, max(int(stations), 8))
    low = np.minimum(start[:, 0], end[:, 0])
    high = np.maximum(start[:, 0], end[:, 0])
    span = np.where(np.abs(end[:, 0] - start[:, 0]) > 1e-12, end[:, 0] - start[:, 0], np.nan)

    radii = np.zeros_like(grid)
    for index, station in enumerate(grid):
        crossing = (low <= station) & (high >= station)
        if not np.any(crossing):  # pragma: no cover - the grid spans the body
            continue
        blend = np.clip((station - start[crossing, 0]) / span[crossing], 0.0, 1.0)
        blend = np.where(np.isfinite(blend), blend, 0.0)[:, None]
        point = (1.0 - blend) * start[crossing, 1:] + blend * end[crossing, 1:]
        radii[index] = float(np.hypot(point[:, 0], point[:, 1]).max())
    return np.asarray(-grid[::-1]), np.asarray(radii[::-1])


def body_from_mesh(
    path: str | Path,
    name: str,
    length: float | None = None,
    diameter: float | None = None,
    axis: int | None = None,
    stations: int = 256,
) -> ScaledBody:
    """Load a mesh and reduce it to a scaled drawable profile.

    Parameters
    ----------
    length, diameter:
        Physical targets (m). ``None`` for **one** of them takes the source
        mesh's own proportions for that dimension — an isotropic scale,
        which is what to use when the mesh's shape is trusted and only its
        units are unknown. ``None`` for both leaves the file's own units,
        which for a mesh carrying none means a body drawn a hundred metres
        long.
    axis:
        Which column of the vertices is the body axis. ``None`` takes the
        longest extent, which is wrong for a squat body — see
        :func:`axial_frame`.
    """
    location = Path(path)
    mesh = load_stl(location, name=name)
    frame = axial_frame(mesh, axis)
    source_length = float(frame[:, 0].max() - frame[:, 0].min())
    source_diameter = 2.0 * float(np.hypot(frame[:, 1], frame[:, 2]).max())
    if length is None and diameter is None:
        axial = radial = 1.0
    elif length is None:
        axial = radial = float(diameter) / source_diameter  # type: ignore[arg-type]
    elif diameter is None:
        axial = radial = float(length) / source_length
    else:
        axial = float(length) / source_length
        radial = float(diameter) / source_diameter
    profile, radii = mould_line(mesh, axis=axis, stations=stations)
    return ScaledBody(
        name=name,
        stations=profile * axial,
        radii=radii * radial,
        axial_factor=axial,
        radial_factor=radial,
        source=location,
    )


def _analytic_stack() -> ScaledBody:
    """The typed profile, kept only as a fallback when the mesh is absent.

    35.4 m long, 3.0 m across, power-law nose over the first 3.3 m. It is
    the shape the applied layer's reference mass model was anchored
    to, and it has no separation rings and no aft shroud — which is the
    whole reason the mesh is preferred.
    """
    stations = np.linspace(0.0, 35.4, 200)
    radii = np.where(stations < 3.3, 1.5 * (stations / 3.3) ** 0.59, 1.5)
    return ScaledBody("reference stack (analytic)", stations, radii)


@lru_cache(maxsize=1)
def default_stack() -> ScaledBody:
    """The launcher's profile, from the bundled mesh where it exists.

    Falls back to :func:`_analytic_stack` — silently, because unlike the
    imagery fallback this one changes no measurement: the analytic profile
    has the same length and the same maximum diameter, and differs only in
    carrying no rings. Which body it is, is recorded in
    :attr:`ScaledBody.source`.
    """
    try:
        return body_from_mesh(STACK_MODEL, "reference heavy-class stack")
    except (FileNotFoundError, OSError, ValueError):
        return _analytic_stack()


@lru_cache(maxsize=1)
def default_bus(diameter: float | None = None) -> ScaledBody:
    """The post-boost bus, scaled to fit inside the stage that carries it.

    Parameters
    ----------
    diameter:
        Target diameter (m). ``None`` takes the launcher's own mould line at
        the bus station, so the bus is exactly as wide as the bay it comes
        out of instead of being sized independently and hoped about.
    """
    if diameter is None:
        stack = default_stack()
        # The bus rides between the two forward separation rings; take the
        # body radius just aft of the forward one, less a clearance.
        diameter = 2.0 * float(np.interp(8.0, stack.stations, stack.radii)) * 0.92
    try:
        # Axis 2 named explicitly: the deck is wider than it is deep, so the
        # longest-extent rule would take its diameter for its axis and draw
        # a 103-unit-long body.
        return body_from_mesh(BUS_MODEL, "post-boost bus", diameter=diameter, axis=2)
    except (FileNotFoundError, OSError, ValueError):
        stations = np.array([0.0, 0.15, 1.05, 1.20])
        radii = np.array([0.35, 0.5 * diameter, 0.5 * diameter, 0.4])
        return ScaledBody("post-boost bus (analytic)", stations, radii)


@lru_cache(maxsize=1)
def default_reentry_vehicle() -> ScaledBody:
    """One re-entry vehicle at the stated Mk21-class envelope."""
    try:
        return body_from_mesh(
            REENTRY_VEHICLE_MODEL,
            "re-entry vehicle",
            length=RV_LENGTH,
            diameter=RV_DIAMETER,
        )
    except (FileNotFoundError, OSError, ValueError):
        stations = np.linspace(0.0, RV_LENGTH, 40)
        radii = 0.5 * RV_DIAMETER * (stations / RV_LENGTH) ** 0.7
        return ScaledBody("re-entry vehicle (analytic)", stations, radii)
