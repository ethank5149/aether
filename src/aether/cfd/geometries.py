"""The reference bodies a flow-field study spans, and how to mesh them.

Named here rather than left implicit in whatever happens to be in
``results-cfd``. That directory currently holds one blunted cone and one sharp
one, at a single Mach number and zero incidence, which is a mesh-convergence
study rather than a survey of shapes -- and a figure set built only from it
quietly becomes a claim that a cone at :math:`\\alpha = 0` is the subject.

Three shapes, chosen because each breaks something the others do not:

``sphere-cone``
    The one with an anchor. Its sharp-cone limit has an exact solution
    (Taylor--Maccoll), so it is the shape that says whether the *method* is
    right, and it is kept for that reason rather than for its own sake.
``biconic``
    A second cone angle, so a second shock and an expansion at the junction.
    The first shape here whose meridian profile is not monotone, and the one
    that shows whether a cut picks up an interior wave rather than only the
    bow shock.
``caret-waverider``
    The one that is not a body of revolution, and the one built from a flow
    rather than from dimensions: :func:`~aether.geometry.bodies.caret_waverider`
    solves the oblique-shock relation for its design Mach number and puts the
    leading edges *in* the shock plane, which is the entire definition of a
    waverider. Everything upstream of it that silently assumed axisymmetry
    fails here, which is half of why it is carried.
``lifting-body``
    A power-law blended body -- wide, flat, and emphatically not on-design.
    Kept as the contrast: it is caret-*shaped* without riding its own shock,
    so the pair shows what the on-design condition is actually worth.
``osculating-cone``
    A waverider designed from a *curved* shock rather than a plane one, by
    tracing streamlines through the local conical field of each osculating
    plane. The general case the caret is the simplest member of.

Four more are shapes whose form is the answer to a stated optimisation, so
each arrives with a number to check itself against rather than only a picture:
``sears-haack`` (minimum wave drag at fixed length and volume),
``von-karman-ogive`` (the same problem at fixed base radius),
and the two power-law bodies at :math:`n = 3/4` and :math:`n = 2/3` -- the
Newtonian optimum and the blast-wave analogy, which are different arguments
and therefore different shapes.

What is *not* claimed
---------------------

These are free parameters of a study, not attempts to match a vehicle. The
sphere-cone is the exception only because it has a closed-form limit to check
against, and even there the anchor is the exact solution rather than any
published shape.

Two representations, and which one a shape uses is not arbitrary
--------------------------------------------------------------

:mod:`aether.aerodynamics.panels` builds *sampled triangles* for impact
theory. :mod:`aether.geometry.bodies` builds the same families as **exact
solids** -- ``Revolve`` and ``Loft`` B-reps -- which can be tessellated at any
density with every node lying on the true surface, exported to STEP, and
measured exactly. Where a shape exists in both, this module takes the solid:
refining its mesh then converges to the body rather than to whatever faceting
a generator happened to bake in. The caret is the clear case --
:func:`~aether.aerodynamics.panels.caret_lifting_body` is a panel-only
lifting surface that is deliberately *not* on-design, while
:func:`~aether.geometry.bodies.caret_waverider` is the real thing.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from aether.aerodynamics.cfd.meshing import (
    MeshProgress,
    MeshResult,
    RefinementBall,
    inviscid_domain,
)
from aether.aerodynamics.panels import (
    PanelModel,
    blunted_multiconic,
    spatular_wedge,
    surface_of_revolution,
)
from aether.cfd.config import ReferenceFrame
from aether.geometry.bodies import (
    caret_waverider,
    power_law_body,
    sears_haack,
    von_karman_ogive,
)
from aether.geometry.brep import Loft, Revolve, surface_mesh
from aether.geometry.mesh import VehicleMesh
from aether.geometry.waverider import (
    OsculatingConeWaverider,
    osculating_cone_waverider,
    power_shock_curve,
)

__all__ = [
    "REFERENCE_BODIES",
    "BodyParameter",
    "ReferenceBody",
    "build_domain",
    "build_surface",
    "reference_frame",
    "surface_grid",
]


@dataclass(frozen=True)
class BodyParameter:
    """One dimension of a shape, and the range it is sensible over.

    Carried with the body so a caller building a control does not have to
    know what a nose radius is, or guess how far it can be pushed before the
    closure fails. The bounds are what the generator accepts, not what is
    aerodynamically wise.
    """

    name: str
    default: float
    low: float
    high: float
    step: float
    unit: str = "m"
    description: str = ""


@dataclass(frozen=True)
class ReferenceBody:
    """One shape in the study, its parameters, and why it is in it.

    ``build`` takes keyword overrides for anything in :attr:`parameters` and
    returns either a :class:`~aether.aerodynamics.panels.PanelModel` -- sampled
    triangles, from which the structured net is taken -- an exact
    :mod:`~aether.geometry.brep` solid, or a designed waverider.
    :func:`build_surface` accepts all three, so which representation a shape
    has is a property of the shape and not something every caller must know.
    """

    name: str
    build: Callable[..., Any]
    axisymmetric: bool
    note: str
    parameters: tuple[BodyParameter, ...] = ()
    size_max: float | None = None
    """Tessellation cell size for a solid; ignored for a panel generator."""

    size_min: float | None = None
    """Floor on the tessellation cell size, where curvature would otherwise set it.

    The caret waverider needs this and ``size_max`` does not reach it: its 3 mm
    blunted leading edge drives gmsh's curvature adaptation, so most of its
    faces sit at the floor rather than at the target. Halving ``size_max`` alone
    moved it 45 924 -> 54 960 faces, a ratio of 1.20 where a refinement level
    wants 4, while halving ``size_min`` moves it 23 208 -> 53 180 -> 128 616.

    A body whose mesh is curvature-limited and whose floor is not declared
    cannot be refined, however many other knobs are turned.
    """

    grid: Mapping[str, int] = field(default_factory=dict)
    """Generator keywords that are **element counts**, at their baseline values.

    Declared per body rather than inferred from the signature, because a count
    and a shape parameter look identical to :mod:`inspect` and guessing wrong
    silently refines the wrong thing.

    :func:`build_surface` scales each of these by the surface resolution, and
    scales :attr:`size_max` by its reciprocal for the shapes gmsh meshes. Six of
    the nine bodies here declared nothing and therefore **refused refinement
    outright** -- so no grid-convergence study was possible on the waveriders,
    the ogive or the power-law bodies, and any coefficient they produced was a
    single-grid number with no error bar. Every underlying generator had taken
    these counts all along; only the wrappers dropped them.
    """

    def defaults(self) -> dict[str, float]:
        """The parameter set this body is listed at."""
        return {p.name: p.default for p in self.parameters}


def _sphere_cone(
    nose_radius: float = 0.06, length: float = 2.0, half_angle: float = 10.0,
    **grid: int,
) -> PanelModel:
    return blunted_multiconic(
        nose_radius=nose_radius,
        lengths=[length],
        half_angles=[np.radians(half_angle)],
        fillet_radii=[],
        **grid,
    )


def _hemisphere(radius: float = 0.045, **grid: int) -> PanelModel:
    r"""A hemisphere on a flat base: the verification body.

    It is here because it is the one shape with **three independent published
    answers** and no fitted parameters:

    * shock standoff from Ambrosio & Wortman's sphere correlation, which is what
      :func:`~aether.cfd.shock.billig_shock` implements;
    * Newtonian forebody drag, which for a hemisphere is exactly
      :math:`C_{p,\max}/2` -- no integration, no geometry, no room to be subtly
      wrong;
    * Lobb's ballistic-range standoff measurements, held in
      ``/mnt/user/source/gdtk-master``.

    A pipeline whose every internal check passes on a wrong answer needs an
    external one, and this is the cheapest that exists. The default radius is
    the DLR n90 cylinder's, so the two sit at comparable conditions.

    The construction is ``blunted_multiconic`` with a degenerate cone: the cap
    sweeps to :math:`90^\circ - \delta`, and a 0.1-degree, 0.1-millimetre
    frustum closes the equator. The meridian therefore comes from the same
    generator every other body in the study uses, rather than from a second
    code path that could drift away from it.

    .. note::

       This was briefly withdrawn on a measurement that said the cap reached
       120 degrees -- wetted area 0.019054 against
       :math:`2\pi r^2 = 0.012723`. The geometry was right and the check was
       wrong: :func:`build_surface` returns a **closed** body, so 0.019054 is
       :math:`2\pi r^2 + \pi r^2` for the hemisphere *plus its base disc*,
       within 0.16 % of :math:`3\pi r^2`. The forward-facing panels alone sum
       to 0.012711, a 0.1 % faceting deficit against the sphere they chord.
       Compare a closed area to a closed formula.
    """
    # Revolved directly, not built as a degenerate multiconic. The cap in
    # ``blunted_multiconic`` stops at 90 - delta and needs a frustum to reach
    # the equator; at delta = 0.1 degrees that frustum is 78 um of axis against
    # 2.5 mm of circumference, a **sliver ring**, and it carried 448 of the 450
    # poor cells in the extruded mesh -- 5.9 % of the volume, all of it in one
    # band, on the body whose whole purpose is to have no error of its own.
    phi = np.linspace(0.0, 0.5 * np.pi, int(grid.get("n_axial_per_segment", 40)) + 1)
    return surface_of_revolution(
        radius * (1.0 - np.cos(phi)),
        radius * np.sin(phi),
        n_circ=int(grid.get("n_circ", 48)),
    )


def _biconic(
    nose_radius: float = 0.05,
    length_1: float = 1.0,
    length_2: float = 1.5,
    half_angle_1: float = 12.0,
    half_angle_2: float = 7.0,
    fillet_radius: float = 0.1,
    **grid: int,
) -> PanelModel:
    return blunted_multiconic(
        nose_radius=nose_radius,
        lengths=[length_1, length_2],
        half_angles=[np.radians(half_angle_1), np.radians(half_angle_2)],
        fillet_radii=[fillet_radius],
        **grid,
    )


def _caret_waverider(
    design_mach: float = 8.0,
    wedge_angle: float = 6.0,
    length: float = 4.0,
    leading_edge_radius: float = 0.003,
    ridge_radius: float = 0.010,
    **grid: int,
) -> Any:
    r"""Caret waverider with **blunted** edges, which is what a real one has.

    :func:`~aether.geometry.bodies.caret_waverider` has carried these radii all
    along and defaults them to zero -- the ideal sharp shape. This wrapper did
    not pass them, so every waverider meshed here was perfectly sharp, and a
    perfectly sharp leading edge is not a vehicle: it carries infinite
    stagnation heating, :math:`q \propto 1/\sqrt R`.

    It also would not mesh. The sharp body's surface came out with **ten cells
    below the quality threshold** (min SICN 0.047) against zero on the blunt
    axisymmetric bodies, and its second-order run diverged at ~130 iterations
    on every configuration tried.

    Defaults follow that function's own guidance -- *"one to five millimetres is
    the range real vehicles live in, and it wants optimising rather than
    assuming"* -- so 3 mm at the leading edge, mid-range. The ridge sits inside
    the shock layer, sees far milder conditions and "can usually be blunted more
    freely", so 10 mm. **Both are exposed as parameters** because the docstring
    is right that they want optimising: they trade heating against the spillage
    that costs a waverider the compression it exists for.
    """
    return caret_waverider(
        design_mach=design_mach,
        wedge_angle=np.radians(wedge_angle),
        length=length,
        leading_edge_radius=leading_edge_radius,
        ridge_radius=ridge_radius,
        **grid,
    )


def _lifting_body(
    length: float = 2.9,
    base_half_span: float = 1.0,
    base_half_thickness: float = 0.3,
    nose_radius: float = 0.1,
    **grid: int,
) -> PanelModel:
    """Power-law blended body, with its nose bluntness made visible.

    ``spatular_wedge`` has carried ``nose_radius_y`` and ``nose_radius_z`` all
    along and defaults them to 0.1 m. This wrapper never passed them, so the
    bluntness that sets the stagnation heating and detaches the bow shock was
    invisible in the parameter list and could not be varied -- the same shape of
    omission as the caret's leading-edge radii. Exposed as one radius because
    the two are equal in every use here; a blended body with different radii in
    span and thickness is a different study.
    """
    return spatular_wedge(
        length=length,
        base_half_span=base_half_span,
        base_half_thickness=base_half_thickness,
        nose_radius_y=nose_radius,
        nose_radius_z=nose_radius,
        **grid,
    )


def _osculating_cone(
    design_mach: float = 8.0,
    shock_angle: float = 14.0,
    length: float = 4.0,
    half_span: float = 0.55,
    capture_fraction: float = 0.55,
    shock_depth: float = 0.35,
    shock_exponent: float = 2.0,
    **grid: int,
) -> Any:
    """On-design, from a power-law shock whose curvature varies along the span."""
    return osculating_cone_waverider(
        design_mach=design_mach,
        shock_angle=np.radians(shock_angle),
        length=length,
        half_span=half_span,
        capture_fraction=capture_fraction,
        shock_curve=power_shock_curve(shock_depth, half_span, shock_exponent),
        **grid,
    )


_LENGTH = BodyParameter("length", 2.0, 0.5, 8.0, 0.1)

#: The study's shapes, keyed by the name their figures and directories use.
REFERENCE_BODIES: dict[str, ReferenceBody] = {
    "sphere-cone": ReferenceBody(
        name="sphere-cone",
        build=_sphere_cone,
        axisymmetric=True,
        note="blunted single cone; the shape with a closed-form anchor",
        parameters=(
            BodyParameter("nose_radius", 0.06, 0.005, 0.5, 0.005),
            _LENGTH,
            BodyParameter("half_angle", 10.0, 3.0, 45.0, 0.5, "deg"),
        ),
    ),
    "hemisphere": ReferenceBody(
        name="hemisphere",
        build=_hemisphere,
        axisymmetric=True,
        note="verification body: published standoff and an exact Newtonian C_D",
        parameters=(
            BodyParameter(
                "radius", 0.045, 0.005, 0.5, 0.005, "m",
                "0.045 m is the DLR n90 cylinder, so the two sit at like conditions",
            ),
        ),
    ),
    "biconic": ReferenceBody(
        name="biconic",
        build=_biconic,
        axisymmetric=True,
        note="two cone angles, so a junction expansion and a second shock",
        parameters=(
            BodyParameter("nose_radius", 0.05, 0.005, 0.5, 0.005),
            BodyParameter("length_1", 1.0, 0.2, 4.0, 0.1),
            BodyParameter("length_2", 1.5, 0.2, 6.0, 0.1),
            BodyParameter("half_angle_1", 12.0, 3.0, 45.0, 0.5, "deg"),
            BodyParameter("half_angle_2", 7.0, 1.0, 40.0, 0.5, "deg"),
            BodyParameter("fillet_radius", 0.1, 0.01, 1.0, 0.01),
        ),
    ),
    "caret-waverider": ReferenceBody(
        name="caret-waverider",
        build=_caret_waverider,
        axisymmetric=False,
        note="Nonweiler caret, built on-design from its own shock",
        parameters=(
            BodyParameter("design_mach", 8.0, 2.0, 27.0, 0.5, "", "leading edges ride this shock"),
            BodyParameter("wedge_angle", 6.0, 2.0, 25.0, 0.5, "deg"),
            BodyParameter("length", 4.0, 1.0, 10.0, 0.25),
            BodyParameter("leading_edge_radius", 0.003, 0.001, 0.005, 0.0005, "m",
                          "sharp is not a vehicle: q ~ 1/sqrt(R)"),
            BodyParameter("ridge_radius", 0.010, 0.002, 0.050, 0.002, "m",
                          "inside the shock layer, blunt more freely"),
        ),
        size_max=0.06,
        grid={"n_sections": 24, "edge_samples": 6},
        size_min=0.002,
    ),
    "lifting-body": ReferenceBody(
        name="lifting-body",
        build=_lifting_body,
        axisymmetric=False,
        note="power-law blended body; caret-shaped but not on-design",
        parameters=(
            BodyParameter("length", 2.9, 1.0, 8.0, 0.1),
            BodyParameter("base_half_span", 1.0, 0.2, 3.0, 0.05),
            BodyParameter("base_half_thickness", 0.3, 0.05, 1.5, 0.05),
            BodyParameter("nose_radius", 0.1, 0.01, 0.5, 0.01, "m",
                          "sharp is not a vehicle: q ~ 1/sqrt(R)"),
        ),
        grid={"n_axial": 60, "n_circ": 48},
    ),
    "osculating-cone": ReferenceBody(
        name="osculating-cone",
        build=_osculating_cone,
        axisymmetric=False,
        note="waverider designed from a curved shock, by osculating cones",
        parameters=(
            BodyParameter("design_mach", 8.0, 2.0, 27.0, 0.5, ""),
            BodyParameter("shock_angle", 14.0, 8.0, 40.0, 0.5, "deg"),
            BodyParameter("length", 4.0, 1.0, 10.0, 0.25),
            BodyParameter("half_span", 0.55, 0.1, 2.0, 0.05),
            BodyParameter("capture_fraction", 0.55, 0.1, 0.9, 0.05, ""),
            BodyParameter("shock_depth", 0.35, 0.05, 1.5, 0.05),
            BodyParameter(
                "shock_exponent", 2.0, 1.2, 2.0, 0.05, "", "above 2 the centreline degenerates"
            ),
        ),
        grid={"n_span": 41, "n_axial": 41},
    ),
    "sears-haack": ReferenceBody(
        name="sears-haack",
        build=lambda length=2.5, max_radius=0.28, **grid: sears_haack(
            length=length, max_radius=max_radius, **grid
        ),
        axisymmetric=True,
        note="minimum wave drag at fixed length and volume; closed both ends",
        parameters=(
            BodyParameter("length", 2.5, 0.5, 8.0, 0.1),
            BodyParameter("max_radius", 0.28, 0.05, 1.0, 0.01),
        ),
        size_max=0.05,
        grid={"n_station": 160},
    ),
    "von-karman-ogive": ReferenceBody(
        name="von-karman-ogive",
        build=lambda length=2.5, base_radius=0.32, **grid: von_karman_ogive(
            length=length, base_radius=base_radius, **grid
        ),
        axisymmetric=True,
        note="minimum wave drag at fixed length and base radius; a forebody",
        parameters=(
            BodyParameter("length", 2.5, 0.5, 8.0, 0.1),
            BodyParameter("base_radius", 0.32, 0.05, 1.0, 0.01),
        ),
        size_max=0.05,
        grid={"n_station": 160},
    ),
    "power-law-newtonian": ReferenceBody(
        name="power-law-newtonian",
        build=lambda exponent=0.75, length=2.5, base_radius=0.32, **grid: power_law_body(
            exponent=exponent, length=length, base_radius=base_radius, **grid
        ),
        axisymmetric=True,
        note="n = 3/4, the Newtonian impact-theory optimum",
        parameters=(
            BodyParameter("exponent", 0.75, 0.3, 1.0, 0.01, "", "3/4 is the optimum"),
            BodyParameter("length", 2.5, 0.5, 8.0, 0.1),
            BodyParameter("base_radius", 0.32, 0.05, 1.0, 0.01),
        ),
        size_max=0.05,
        grid={"n_station": 120},
    ),
    "power-law-blast": ReferenceBody(
        name="power-law-blast",
        build=lambda exponent=2.0 / 3.0, length=2.5, base_radius=0.32, **grid: power_law_body(
            exponent=exponent, length=length, base_radius=base_radius, **grid
        ),
        axisymmetric=True,
        note="n = 2/3, the blast-wave analogy",
        parameters=(
            BodyParameter("exponent", 2.0 / 3.0, 0.3, 1.0, 0.01, ""),
            BodyParameter("length", 2.5, 0.5, 8.0, 0.1),
            BodyParameter("base_radius", 0.32, 0.05, 1.0, 0.01),
        ),
        size_max=0.05,
        grid={"n_station": 120},
    ),
}


#: Panel counts ``blunted_multiconic`` uses at ``resolution = 1``. A refinement
#: level scales both, so the surface refines uniformly rather than in one
#: direction -- a grid study whose refinement ratio differs by axis has no
#: single ``r`` to compute an observed order against.
#:
#: Read this as a **cell budget**, not as two independent settings. How the
#: budget is split between meridian and circumference is decided per body by
#: :func:`surface_grid`; this pair only fixes the product.
BASE_PANELS: dict[str, int] = {"n_axial_per_segment": 40, "n_circ": 48}

#: Measured ``(meridian arc length, area-weighted mean radius, segments)`` per
#: body and parameter set. The probe is a full panel build, which is
#: milliseconds, but :func:`build_surface` is called repeatedly by the
#: preflight, the mesher and the force anchor on identical arguments.
_PROPORTIONS: dict[
    tuple[str, tuple[tuple[str, float], ...]], tuple[float, float, int]
] = {}


def _proportions(
    body: ReferenceBody, parameters: dict[str, float]
) -> tuple[float, float, int] | None:
    """``(meridian arc length, area-weighted mean radius, segments)``, or ``None``.

    ``None`` for anything that is not a surface of revolution sampled on a
    structured net -- a waverider is built patch by patch and has no meridian
    to measure.

    ``segments`` is how many times the generator spends
    ``n_axial_per_segment``, and it is **measured from the probe** rather than
    inferred from the body's parameter list. The hemisphere is why: it is a
    degenerate two-piece multiconic whose cone angle is hard-coded, so it has
    no ``half_angle`` parameter to count and a name-based rule put its budget
    at half the truth -- which is exactly the sort of quiet factor of two this
    function exists to remove.
    """
    key = (body.name, tuple(sorted(parameters.items())))
    if key in _PROPORTIONS:
        return _PROPORTIONS[key]
    try:
        model = body.build(**parameters, **BASE_PANELS)
    except TypeError:
        return None
    grid = getattr(model, "surface", None)
    if grid is None:
        return None
    vertices = np.asarray(grid.vertices, dtype=np.float64)
    if vertices.ndim != 3 or vertices.shape[0] < 2:
        return None
    station = vertices[:, 0, 0]
    radius = np.hypot(vertices[:, 0, 1], vertices[:, 0, 2])
    step = np.hypot(np.diff(station), np.diff(radius))
    arc = float(step.sum())
    # Area-weighted, because the anisotropy that matters is the anisotropy
    # where the surface actually is. A hemisphere's mean radius is 0.785 R
    # this way and 0.5 R if the bands are weighted equally, and the second
    # number puts the isotropic station in the wrong place.
    mid = 0.5 * (radius[1:] + radius[:-1])
    band = 2.0 * np.pi * mid * step
    total = float(band.sum())
    if arc <= 0.0 or total <= 0.0:
        return None
    mean_radius = float((mid * band).sum() / total)
    if mean_radius <= 0.0:
        return None
    segments = max(1, round((vertices.shape[0] - 1) / BASE_PANELS["n_axial_per_segment"]))
    _PROPORTIONS[key] = (arc, mean_radius, segments)
    return arc, mean_radius, segments


def surface_grid(
    body: ReferenceBody, parameters: dict[str, float], resolution: float = 1.0
) -> dict[str, int]:
    r"""Split the panel budget between meridian and circumference.

    :data:`BASE_PANELS` used to be applied as-is to every body, and that is a
    statement about one body's proportions imposed on all of them. The
    circumferential spacing is :math:`2\pi \bar{r}/n_\psi` and the meridian
    spacing is :math:`L_m/n_x`; a fixed pair makes those equal for exactly one
    shape and skews every other one.

    Measured, at the campaign budget of 80 x 48:

    ==============  =========  ===============  ==========================
    body            :math:`L_m`  :math:`2\pi\bar r`  circ/axial spacing
    ==============  =========  ===============  ==========================
    sphere-cone     2.115 m    1.747 m          0.69
    biconic         2.606 m    2.007 m          1.28
    **hemisphere**  0.071 m    0.222 m          **2.62**
    ==============  =========  ===============  ==========================

    A compact blunt body has a short meridian and a long circumference, so the
    same counts give cells five times longer around the body than along it. On
    the hemisphere that is a median triangle edge ratio of **4.92** against the
    sphere-cone's 1.75, and it shows up downstream as a median scaled Jacobian
    of 0.20 against 0.58 -- not an extrusion fault, the surface it was extruded
    from.

    So the product is held and the split is solved for:

    .. math::

       n_x = \sqrt{B\,L_m / (2\pi\bar r)}, \qquad n_\psi = B / n_x

    which costs nothing -- the budget, and therefore the cell count and the run
    time, is unchanged -- and brings all three bodies to within 5 % of square.

    ``n_circ`` is snapped to a multiple of eight so the singularity-free nose
    cap always fits: its square grid needs an even number of cells per side and
    its seam carries one vertex per circumferential station. Without the snap a
    level lands on 34 or 68 stations and the cap silently declines to apply,
    leaving the pole in place at exactly the resolutions where it hurts most.

    Falls back to :data:`BASE_PANELS` scaled uniformly for anything with no
    meridian to measure.
    """
    if resolution <= 0.0:
        raise ValueError(f"resolution must be positive, got {resolution:g}")
    measured = _proportions(body, parameters)
    if measured is None:
        return {
            "n_axial_per_segment": max(8, round(BASE_PANELS["n_axial_per_segment"] * resolution)),
            "n_circ": max(8, 8 * round(BASE_PANELS["n_circ"] * resolution / 8)),
        }

    # Solved **once, at resolution 1**, and then scaled. Solving it afresh at
    # each level re-rounds a square root at each level, and the ratios between
    # levels stop being the ratio that was asked for: measured on the
    # sphere-cone at 0.5 / 1 / 2, face counts came out at 3.967x and 4.059x
    # instead of 4x. A refinement study whose levels are not in a known ratio
    # has no *r* to take an observed order against, which is the whole reason
    # the counts are snapped in the first place.
    arc, mean_radius, segments = measured
    budget = BASE_PANELS["n_axial_per_segment"] * segments * BASE_PANELS["n_circ"]
    base_axial, base_circ = _best_split(body, parameters, arc, mean_radius, segments, budget)
    return {
        "n_axial_per_segment": max(4, round(base_axial * resolution)),
        "n_circ": max(8, 8 * round(base_circ * resolution / 8)),
    }


#: Best ``(axial per segment, circumferential)`` per body and parameter set.
_SPLITS: dict[tuple[str, tuple[tuple[str, float], ...]], tuple[int, int]] = {}


def _elongation(model: Any) -> float:
    """Median cell elongation on a structured surface grid.

    The ratio of the two in-plane spacings, taken per cell and reduced to the
    long-over-short form so that a cell stretched either way counts against it.
    Computed on the net rather than on the triangles, which is the same quantity
    to within the diagonal and does not need the mesh built.
    """
    grid = getattr(model, "surface", None)
    if grid is None:
        return float("inf")
    net = np.asarray(grid.vertices, dtype=np.float64)
    if net.ndim != 3 or net.shape[0] < 2 or net.shape[1] < 2:
        return float("inf")
    along = np.linalg.norm(net[1:, :-1, :] - net[:-1, :-1, :], axis=-1)
    around = np.linalg.norm(net[:-1, 1:, :] - net[:-1, :-1, :], axis=-1)
    lo = np.minimum(along, around)
    hi = np.maximum(along, around)
    usable = lo > 1e-12
    if not usable.any():
        return float("inf")
    return float(np.median(hi[usable] / lo[usable]))


def _best_split(
    body: ReferenceBody,
    parameters: dict[str, float],
    arc: float,
    mean_radius: float,
    segments: int,
    budget: int,
) -> tuple[int, int]:
    r"""Pick the split by **building the candidates and measuring them**.

    :math:`n_x = \sqrt{B L_m / 2\pi\bar{r}}` puts the cells square at the
    area-weighted mean radius, which is the right idea and not a reliable
    ranking: on the sphere-cone it chooses 64 circumferential stations, and 48
    measures better -- median cell elongation 1.75 against 2.11, and 2.89
    against 5.27 at the ninetieth percentile. The criterion is blind to the
    nose, where the circumferential spacing goes to zero and the elongation is
    worst, and the nose is where the cells that matter are.

    So the estimate picks the neighbourhood and four candidates are measured.
    Each is a panel build of a few milliseconds, and the winner is cached.

    Candidates are multiples of **sixteen**. The nose cap needs a multiple of
    eight at *every* level of a refinement study, and halving a multiple of
    eight does not give one: 56 becomes 28, and the cap silently declines to
    apply on the coarse grid -- which is the level whose poor cells would
    dominate the observed order.
    """
    key = (body.name, tuple(sorted(parameters.items())))
    if key in _SPLITS:
        return _SPLITS[key]

    axial = math.sqrt(budget * arc / (2.0 * math.pi * mean_radius))
    estimate = budget / max(axial, 1.0)
    step = max(1, round(estimate / 16.0))

    best: tuple[float, int, int] | None = None
    for offset in (-1, 0, 1, 2):
        circ = 16 * (step + offset)
        if circ < 16:
            continue
        per_segment = max(4, round(budget / circ / segments))
        try:
            model = body.build(
                **parameters, n_axial_per_segment=per_segment, n_circ=circ
            )
        except (TypeError, ValueError):
            continue
        score = _elongation(model)
        if best is None or score < best[0]:
            best = (score, per_segment, circ)

    if best is None:
        per_segment = max(4, round(budget / 48 / segments))
        _SPLITS[key] = (per_segment, 48)
        return _SPLITS[key]
    _SPLITS[key] = (best[1], best[2])
    return _SPLITS[key]


def build_surface(
    body: ReferenceBody | str, resolution: float = 1.0, **parameters: float
) -> VehicleMesh:
    """Close a reference body into a watertight, outward-wound surface.

    The winding is checked rather than assumed. An inward-wound body is
    closed, has the right area and integrates to the right panel loads --
    and makes the domain mesher fill the vehicle instead of the space around
    it, which is a failure that looks like a meshing bug three steps later.

    ``resolution`` scales the surface panel counts for a grid-refinement study.
    It is **not** supported by every body -- the waverider generators build
    patch by patch from their own parameters -- and asking for it on one that
    cannot honour it raises rather than silently returning the baseline mesh,
    because a refinement study whose levels are secretly identical reports a
    beautiful convergence order about nothing.
    """
    if isinstance(body, str):
        body = REFERENCE_BODIES[body]
    # Named explicitly rather than left to a TypeError three frames down.
    # `build_domain` takes geometry through **kwargs, so a misspelt parameter
    # -- or a mesh option that belongs to the mesher rather than the shape --
    # arrives here looking like a shape parameter and should say so.
    known = {parameter.name for parameter in body.parameters}
    unknown = sorted(set(parameters) - known)
    if unknown:
        raise TypeError(f"{body.name} has no parameter(s) {unknown}; it takes {sorted(known)}")
    if resolution <= 0.0:
        raise ValueError(f"resolution must be positive, got {resolution:g}")
    import inspect

    signature = inspect.signature(body.build)
    accepts = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    ) or set(BASE_PANELS) <= set(signature.parameters)
    declared = dict(body.grid)
    if not accepts and not declared and resolution != 1.0:
        raise TypeError(
            f"{body.name} does not take a surface resolution, so it cannot be "
            f"refined to {resolution:g}x. Refusing rather than returning the "
            "baseline mesh: a refinement study whose levels are identical "
            "reports a convergence order about nothing."
        )
    # Chosen per body rather than taken from BASE_PANELS as a pair. The budget
    # is the same; how it is split between meridian and circumference is a
    # property of the shape's proportions -- see `surface_grid`, and note that
    # this is applied at resolution 1 too, so the baseline mesh is the one the
    # study refines rather than a differently-shaped one.
    #
    # The circumferential snapping quantises the refinement ratio, and that has
    # to be lived with rather than ignored: the effective *r* wobbles a few
    # percent either side of the level that was asked for. **Choose levels that
    # land cleanly**, and where one does not, take *r* from the cell counts the
    # mesher actually produced rather than from the number that was requested.
    # Counts scale with the resolution; the gmsh target size scales with its
    # reciprocal, because one is "how many" and the other is "how big". Both
    # are needed: a revolved body's contour is sampled at `n_station` and then
    # *remeshed* by gmsh at `size_max`, so refining only the first leaves the
    # surface triangles exactly where they were.
    if declared:
        grid = {
            name: max(2, round(count * resolution)) for name, count in declared.items()
        }
    else:
        grid = surface_grid(body, parameters, resolution) if accepts else {}
    size_max = body.size_max
    if size_max is not None:
        size_max = float(size_max) / resolution
    size_min = body.size_min
    if size_min is not None:
        size_min = float(size_min) / resolution
    model = body.build(**parameters, **grid)
    if isinstance(model, OsculatingConeWaverider):
        # Already a stitched surface: a waverider's nose is an edge rather
        # than a point, so it is built patch by patch instead of revolved
        # or lofted through closed sections.
        return _checked(model.to_mesh(name=body.name), body.name)
    if isinstance(model, Loft | Revolve):
        # Both knobs, because gmsh honours whichever binds first. The caret's
        # blunted leading edge is curvature-limited rather than size-limited --
        # halving `size_max` alone moved it 45924 -> 54962 faces, a ratio of
        # 1.20 where a refinement level wants 4 -- so the curvature sampling
        # scales with the resolution too.
        mesh = surface_mesh(
            model,
            size_max=size_max,
            size_min=size_min,
            curvature_nodes=20.0 * resolution,
        )
        return _checked(mesh, body.name)
    if model.surface is None:
        raise ValueError(
            f"{body.name} was built by a generator that carries no surface grid, "
            "so it cannot be meshed. The panel-only generators (sphere_cone, "
            "caret_lifting_body) are open bodies for impact theory; the meshable "
            "shapes come from blunted_multiconic and spatular_wedge."
        )
    return _checked(VehicleMesh.from_surface_grid(model.surface, name=body.name), body.name)


def _checked(mesh: VehicleMesh, name: str) -> VehicleMesh:
    """Refuse a surface the domain mesher would silently misuse."""
    if not mesh.is_closed:
        raise ValueError(f"{name} is not closed; open at {mesh.boundary_stations()}")
    triangles = mesh.triangles
    volume = float(
        np.einsum(
            "ij,ij->i",
            triangles[:, 0],
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        ).sum()
        / 6.0
    )
    if volume <= 0.0:
        raise ValueError(f"{name} has inward-wound normals (signed volume {volume:g})")
    return mesh


def _declared_default(body: ReferenceBody, name: str) -> float:
    """A body's declared default for one parameter, or zero if it has none.

    A sharp body -- the caret waverider has no nose radius at all -- returns
    zero, and the caller skips the stagnation region rather than inventing a
    curvature for a leading edge that has none.
    """
    for parameter in body.parameters:
        if parameter.name == name:
            return float(parameter.default)
    return 0.0


def build_domain(
    body: ReferenceBody | str,
    path: Path | str,
    mach: float,
    refinement: tuple[RefinementBall, ...] = (),
    progress: MeshProgress | None = None,
    wall_refinement: float = 0.05,
    optimize: bool = True,
    quality_threshold: float = 0.1,
    alpha: float = 0.0,
    beta: float = 0.0,
    nose_cells: int = 0,
    viscous: bool = True,
    temperature: float = 288.15,
    pressure: float = 101325.0,
    wall_temperature: float | None = None,
    **parameters: float,
) -> MeshResult:
    """Mesh the flow domain around a reference body at a given Mach number.

    Inviscid: an isotropic tetrahedral domain with no prism layer, which is
    the right choice for an Euler run and not merely the cheap one -- a
    boundary layer that the equations do not have is not worth resolving, and
    extruding one onto a nose whose radius is smaller than the layer is thick
    puts badly skewed cells exactly at the stagnation point.

    The sizing is a function of ``mach``: shock standoff shrinks and the
    stagnation gradient steepens as speed rises, so the nose and shock cell
    sizes follow it. Meshing one Mach number and solving another is therefore
    a real error, not a shortcut, and the Mach number is required here for
    that reason rather than defaulted.

    **That was the intent and it was not the behaviour.** Until 2026-09-10 this
    function passed ``mach`` to :func:`inviscid_domain`, which spends it on a
    far-field Mach-angle envelope and nothing else, while the near-wall size
    came from ``wall_refinement`` alone. Meshes at Mach 6, 8, 10 and 20 came out
    at *418 707 elements each* -- identical to the digit -- and the Mach 10 and
    Mach 20 runs diverged, quietly, into results that recorded as ``done``.
    :func:`~aether.aerodynamics.cfd.meshing.stagnation_refinement` had existed
    the whole time, exported and never called.

    It is reachable here now, but **off by default**, and the reason is a
    measurement rather than caution. Enabling it at ``nose_cells=4`` on the
    sphere-cone did what it promised -- 419k to 897k elements, no cells below
    the quality threshold, the drag error bar halved, pitot violations from 196
    nodes to 49 -- and **cost more than it bought**:

    ==========================  ==================  ==================
    alpha = 0 case              coarse (no region)  refined (4 cells)
    ==========================  ==================  ==================
    sphere-cone M6              converges           **diverges**
    sphere-cone M8              converges           converges
    sphere-cone M8, 30 km       converges           **diverges**
    biconic M8                  converges           **diverges**
    ==========================  ==================  ==================

    Four previously-converging cases were lost for one better-resolved one, and
    the divergences are non-monotonic in Mach -- the signature of a
    configuration sitting on a stability boundary rather than of a resolution
    problem. A finer mesh resolves the shock more sharply, and a sharper shock
    is harder for a MUSCL reconstruction to cross without producing negative
    temperatures.

    So ``nose_cells`` defaults to ``0``. **Set it only together with a
    robustness change** -- a lower CFL ceiling, stronger limiting, or a longer
    first-order prelude -- none of which has been tested yet. Four is the right
    number when it is used; see that function's table, where four clears every
    pitot violation at 561k elements and ten is unaffordable.

    ``alpha`` and ``beta`` (**radians**) orient the refinement against the
    oncoming flow rather than the body axis. At zero they reproduce the axial
    placement exactly.

    ``progress`` is called at each stage. Worth passing: a 400k-element domain
    takes tens of seconds and the 3D pass is nearly all of it, during which a
    silent build is indistinguishable from a hung one.
    :func:`~aether.aerodynamics.cfd.meshing.console_progress` draws a bar on a
    terminal; a notebook should run the build on a thread and update a widget
    from the callback.
    """
    if isinstance(body, str):
        body = REFERENCE_BODIES[body]
    surface = build_surface(body, **parameters)

    if viscous:
        # A prism boundary layer, because the solver this feeds runs RANS with
        # k-omega SST and SST cannot compute a production term without one.
        # ``ViscousSizing`` defaults to y+ = 1 over 25 layers at a growth ratio
        # of 1.2, which is what "wall-resolved" means; the first cell height is
        # solved for the actual flight condition rather than guessed, so
        # temperature and pressure are needed here and not only Mach.
        from dataclasses import replace

        from aether.aerodynamics.cfd.meshing import (
            ViscousSizing,
            boundary_layer_thickness,
            viscous_domain,
            wall_spacing_for_y_plus,
        )

        reference_length = float(np.ptp(surface.vertices[:, 0]))
        base = ViscousSizing()
        first_cell = wall_spacing_for_y_plus(
            mach,
            reference_length,
            temperature=temperature,
            pressure=pressure,
            y_plus=base.y_plus,
            wall_temperature=wall_temperature,
        )
        # Layer count from the boundary layer, not from a constant. The stack
        # has to *cover* the layer and not much more: short and the outer flow
        # sees prism cells stretched across a gradient, long and the extrusion
        # self-intersects on a nose smaller than the stack is tall.
        #
        # The default 25 is wrong nearly everywhere on this body. At Mach 8 it
        # gives a stack of 2.05 m against a 0.109 m boundary layer -- nineteen
        # times too tall, on a 2.05 m body. Solved for 1.25 delta instead:
        # 20 layers at 30 km, 11 at 45 km, 7 at 55 km. One constant cannot
        # serve three altitudes, which is the same lesson the Mach-blind mesh
        # taught earlier today.
        thickness = boundary_layer_thickness(
            mach, reference_length, temperature=temperature, pressure=pressure
        )
        ratio = base.growth_ratio
        layers = int(
            np.ceil(np.log1p(1.25 * thickness * (ratio - 1.0) / first_cell) / np.log(ratio))
        )
        sizing = replace(base, n_layers=max(4, min(layers, base.n_layers)))
        return viscous_domain(
            surface,
            Path(path),
            mach=mach,
            sizing=sizing,
            temperature=temperature,
            pressure=pressure,
            wall_temperature=wall_temperature,
            first_cell_height=first_cell,
        )

    if not refinement and nose_cells > 0:
        nose_radius = float(parameters.get("nose_radius", 0.0)) or _declared_default(
            body, "nose_radius"
        )
        if nose_radius > 0.0:
            from aether.aerodynamics.cfd.meshing import (
                freestream_direction,
                stagnation_refinement,
            )

            ball = stagnation_refinement(
                surface,
                mach,
                nose_radius=nose_radius,
                cells=nose_cells,
                direction=freestream_direction(alpha, beta),
            )
            if np.isfinite(ball.size) and ball.size > 0.0:
                refinement = (ball,)

    return inviscid_domain(
        surface,
        Path(path),
        mach=mach,
        refinement=refinement,
        progress=progress,
        wall_refinement=wall_refinement,
        optimize=optimize,
        quality_threshold=quality_threshold,
    )


def reference_frame(
    body: ReferenceBody | str, moment_fraction: float = 0.5, **parameters: float
) -> ReferenceFrame:
    """Non-dimensionalisation derived from the body, not typed in beside it.

    Coefficients are meaningless without the reference they were divided by,
    and a reference typed in next to the geometry is one edit away from
    belonging to a different shape. These bodies differ by a factor of three
    in frontal area, so one hard-coded number is wrong for most of them --
    and wrong invisibly, because a drag coefficient scaled by the wrong area
    is still a plausible drag coefficient.

    Two ways of getting the frontal area, and which one is used depends on
    what the shape allows:

    For a **body of revolution** it is :math:`\\pi r_{\\max}^2`, taken from the
    vertices. Those lie on the true surface, so the answer does not depend on
    how finely the body happened to be tessellated -- and a reference area
    that moves with mesh density is a bad reference area. It is exact for a
    forebody, whose widest station is its base and therefore a sample point.
    Where the maximum is *interior* -- Sears--Haack peaks mid-body -- sampling
    can only miss it from below, so the area is a slight under-estimate: one
    part in :math:`10^4` at the default resolution, and one-sided, which is
    the knowable direction to be wrong in.

    Otherwise it is the projected area :math:`\\tfrac{1}{2}\\sum |n_x| A`, an
    identity for any closed body whose silhouette every axial ray crosses
    twice. That one *is* sampled, and it is worth knowing by how much: swept
    across tessellation sizes on a power-law body, whose nose slope is
    infinite, it wandered by two percent at one resolution while converging to
    within 0.03 % at the finest. Exact formula, imperfectly sampled surface.

    A caution that outlives this function: when CFD coefficients are to be
    spliced with panel-method tables, both must use the *same* reference, and
    that is the table's -- see :class:`~aether.cfd.config.ReferenceFrame`.
    This derives a self-consistent one for a standalone case, which is not the
    same thing as a shared one.
    """
    if isinstance(body, str):
        body = REFERENCE_BODIES[body]
    mesh = build_surface(body, **parameters)
    vertices = mesh.vertices
    stations = vertices[:, 0]
    length = float(np.ptp(stations))

    if body.axisymmetric:
        area = float(np.pi * np.linalg.norm(vertices[:, 1:], axis=1).max() ** 2)
    else:
        triangles = mesh.triangles
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        area = 0.25 * float(np.abs(normals[:, 0]).sum())

    return ReferenceFrame(
        area=area,
        length=length,
        moment_reference=(float(stations.min() + moment_fraction * length), 0.0, 0.0),
    )
