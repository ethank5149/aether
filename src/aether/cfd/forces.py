"""What a CFD run actually measured, and what has to be supplied around it.

SU2 reports one force per monitoring marker and a total across all of them.
Taking that total is wrong on any body with a blunt base, because the base is
not a region this pipeline resolves: there is no wake model, the mesh carries
no wake refinement, and an inviscid or coarsely-viscous solve lets the flow
stagnate on the base instead of separating off its corner.

Measured on this queue, 2026-09-10:

==========  ===============  ==============  =============
case        forebody C_A     base C_A (SU2)  total (SU2)
==========  ===============  ==============  =============
laminar M8  +0.0826          **-0.0084**     +0.0742
Euler M6    +0.0832          **-0.2004**     **-0.1171**
==========  ===============  ==============  =============

Both bases come out *negative* -- thrust -- and at Mach 6 the base alone is two
and a half times the forebody drag, which is how a converged run with 48
resolved oscillation periods reported a vehicle that accelerates itself.

:func:`~aether.aerodynamics.closure.base_axial_coefficient` exists for exactly
this and says so: *"positive because a base pressure below freestream pulls the
vehicle backwards. Add it to a CFD forebody force to get a total that means
something."* This module is that addition.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "DragSplit",
    "NonPhysical",
    "ResolvedForces",
    "density_residual",
    "diverged_outright",
    "drag_split",
    "newtonian_forebody",
    "nonphysical_states",
    "residual_excursion",
    "resolved_forces",
    "screen_residual",
]


def density_residual(columns: dict[str, Any]) -> Any:
    """The density residual series, whichever gas model wrote it.

    A perfect-gas run writes ``rms[rho]``; NEMO writes ``rms[rho_0]`` through
    ``rms[rho_4]``, one per species. Code that looks only for the first name is
    not merely incomplete -- it is **a guard that is off** for every
    configuration naming it differently, which is how two archived Mach 20 runs
    carrying a drag coefficient of 9117 passed a non-finite-residual check.

    Returns ``None`` when no density residual is present at all, which is a
    different statement from "the residual is fine" and callers must treat it
    as such.
    """
    for name in ("rms[rho]", "rms_density"):
        if name in columns:
            return columns[name]
    species = sorted(k for k in columns if k.startswith("rms[rho_"))
    return columns[species[0]] if species else None


@dataclass(frozen=True)
class NonPhysical:
    """How many states SU2 had to repair, and whether the count was growing.

    SU2 clips any cell whose reconstructed state is unphysical -- a negative
    species density, a negative temperature -- and says so on stdout rather
    than in ``history.csv``. A run that is being repaired a hundred times an
    iteration is not converging to the governing equations; it is converging to
    the equations *plus* whatever the clipper puts back.

    ``history.csv`` cannot see this, which is the whole reason this exists. The
    Mach 16 AIR-5 run archived on 2026-09-10 held ``CD`` flat to four decimals
    with a finite falling residual and 24 resolved oscillation periods -- every
    signal the report had -- while SU2 repaired 916,527 reconstructed states
    and the per-iteration count climbed from 74 to 167 and never came back
    down. Its forebody coefficient was 2.2x Newtonian.

    ``growing`` is the load-bearing field. A fixed handful of points at a
    singular corner is tolerable; a count with a trend is a slow divergence
    that no residual in the history file reports.
    """

    points_first: int
    points_last: int
    points_max: int
    iterations: int
    reconstructions: int

    @property
    def growing(self) -> bool:
        """Whether the repaired-point count trended upward over the run."""
        return self.points_last > 1.5 * max(self.points_first, 1)

    @property
    def clean(self) -> bool:
        """Whether SU2 ran without repairing the solution at all."""
        return self.points_max == 0 and self.reconstructions == 0


_SCREEN_ROW = re.compile(r"^\|\s*(\d+)\|(.*)$")


def screen_residual(log: Path) -> list[float] | None:
    """The density residual series from SU2's screen table.

    ``history.csv`` belongs to whichever run wrote it last, so a prelude's
    convergence is not readable from it -- the second-order run overwrites the
    file. The screen table in ``su2-prelude.log`` is the only record that a
    prelude diverged, and that mattered: at four degrees incidence the prelude
    blew up to a residual of 10^7.4 with a lift coefficient of -1826, printed
    *Exit Success*, and handed its restart to the second-order run as an initial
    condition.

    Header-aware, because the columns are not fixed. A perfect-gas run prints
    ``Inner_Iter | rms[Rho] | CL | CD``; NEMO prints no residual column at all,
    and this returns ``None`` for it rather than guessing at a position.
    """
    try:
        lines = log.read_text(errors="replace").splitlines()
    except OSError:
        return None
    column: int | None = None
    series: list[float] = []
    for line in lines:
        if "Inner_Iter" in line and "|" in line:
            names = [c.strip() for c in line.strip().strip("|").split("|")]
            column = next(
                (i for i, n in enumerate(names) if n.lower().startswith("rms[rho")), None
            )
            continue
        if column is None:
            continue
        matched = _SCREEN_ROW.match(line)
        if matched is None:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) <= column:
            continue
        try:
            series.append(float(cells[column]))
        except ValueError:
            continue
    return series or None


def residual_excursion(series: Any) -> float:
    """How far the run ended above the best residual it ever reached, in decades.

    Not "is the residual finite" -- that check passes a run whose residual is
    ``+6.86``, which is what the two degree incidence case carried for its last
    580 iterations while ``CD`` read a perfectly plausible 0.097 and ``CL`` a
    perfectly plausible 0.052. Finite is not converged.

    Not "is the residual below its starting value" either: a shock-buffeting
    case legitimately plateaus slightly above where it started, and the settled
    Mach 8 runs here end at -0.98 having started at -1.17. Flagging those would
    be a check that cries wolf on good data.

    The statement that separates them is **the solver found a better solution
    earlier and has since walked away from it**. Measured over the whole screen
    trace, including the startup transient:

    ================================  ==========
    case                              excursion
    ================================  ==========
    Mach 8, alpha 0, prelude          2.91
    Mach 8, alpha 0, second order     2.72
    Mach 8, alpha +2, prelude         **17.12**
    Mach 8, alpha +2, second order    **10.23**
    Mach 8, alpha +4, prelude         **11.24**
    ================================  ==========

    A healthy run legitimately spends three decades, because the transient digs
    below the plateau it later settles onto. **Five is the threshold**, and the
    gap either side of it is large enough that no calibration is being fitted.

    One case this cannot catch: alpha +4's *second-order* run, which restarted
    from the diverged prelude above and so began at 10^7.4 and never improved --
    excursion 0.00, because there was never a better solution to walk away from.
    That one is caught by the residual being positive at all
    (:func:`diverged_outright`), and better still by not promoting the prelude.
    """
    values = np.asarray(series, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return 0.0
    return float(values[-1] - values.min())


def diverged_outright(series: Any) -> bool:
    """Whether the run ended with an RMS density residual above unity.

    This reads a **dimensional** residual, which is what this pipeline always
    writes (``REF_DIMENSIONALIZATION= DIMENSIONAL``). Freestream density is
    0.002 kg/m^3 at 45 km and never exceeds 0.02 anywhere in the altitude band,
    so ``log10(rms) > 0`` is a residual some hundreds of times the density it is
    a residual *of*. There is no regime in which that is a converging solution.

    It exists for the one shape :func:`residual_excursion` cannot see: a run that
    started diverged, because it was handed a diverged restart, and therefore
    never walked away from anything.
    """
    values = np.asarray(series, dtype=np.float64)
    values = values[np.isfinite(values)]
    return bool(values.size and values[-1] > 0.0)


_POINTS = re.compile(r"contains (\d+) points that are not physical")
_RECONSTRUCTIONS = re.compile(r"(\d+) reconstructed states for upwinding are non-physical")


def nonphysical_states(log: Path) -> NonPhysical | None:
    """Read ``su2.log`` for the states SU2 had to repair.

    Returns ``None`` when the log is missing or unreadable -- which, as with
    :func:`density_residual`, means *not measured*, not *fine*.
    """
    try:
        text = log.read_text(errors="replace")
    except OSError:
        return None
    points = [int(m) for m in _POINTS.findall(text)]
    reconstructions = sum(int(m) for m in _RECONSTRUCTIONS.findall(text))
    return NonPhysical(
        points_first=points[0] if points else 0,
        points_last=points[-1] if points else 0,
        points_max=max(points) if points else 0,
        iterations=len(points),
        reconstructions=reconstructions,
    )


@dataclass(frozen=True)
class ResolvedForces:
    """Forebody from CFD, base from closure, and the total that means something."""

    forebody_axial: float
    base_axial: float
    axial: float
    normal: float
    base_area: float
    reference_area: float
    su2_total: float
    """What SU2 reported across all markers, kept so the two can be compared."""

    @property
    def base_fraction(self) -> float:
        """Share of the corrected total contributed by the modelled base."""
        return float(self.base_axial / self.axial) if self.axial else float("nan")


def _base_area(body: str, geometry: dict[str, float]) -> float:
    r"""Projected area of the aft-facing surface, from the body's own panels.

    Summed as :math:`\sum A_i n_{x,i}` over panels whose normal points
    downstream, which is the area the base pressure acts on.

    **Not** :math:`\pi r_{\max}^2`. That is only the base area of a body of
    revolution, and this pipeline carries bodies that are not: the caret
    waverider's base plane spans 0.90 m across and 0.41 m deep, so a disc fitted
    to its maximum radius over-counts by a factor of **7.4** -- which inflated
    its base drag from 0.016 to 0.116 and its total from 0.096 to 0.196 before
    this was caught. The sphere-cone and biconic agree with the disc formula to
    four decimal places, which is why the error hid.
    """
    from aether.cfd.geometries import build_surface

    surface = build_surface(body, **geometry)
    normals = np.asarray(surface.normals, dtype=np.float64)
    areas = np.asarray(surface.areas, dtype=np.float64)
    aft = normals[:, 0] > 0.0
    return float(np.sum(areas[aft] * normals[aft, 0]))


def resolved_forces(
    case_dir: Path | str,
    body: str,
    geometry: dict[str, float],
    mach: float,
    reference_area: float,
    window: int = 700,
    forebody_only: bool = False,
    wake_resolved: bool = False,
) -> ResolvedForces | None:
    """CFD forebody plus a base -- modelled, or resolved. ``None`` if it cannot."

    The forebody is the **time-averaged per-marker history**, not a single
    iterate of ``surface_flow.csv``. That follows the sweep path's reasoning:
    the force limit-cycles by several percent, so one iterate carries the full
    ripple where the history carries the mean, and the marker split is the one
    the mesh was written with rather than one re-derived from a cone angle
    about the axis -- which disagrees with it on filleted shoulders by about
    7 % of forebody axial force.
    """
    from aether.aerodynamics.closure import base_axial_coefficient
    from aether.cfd.solver import read_history

    history = Path(case_dir) / "history.csv"
    if not history.exists():
        return None
    columns = read_history(history)
    lookup = {key.lower(): key for key in columns}

    forebody_key = lookup.get("cd(vehicle)")
    if forebody_key is None and forebody_only:
        # A shock-fitted mesh never extrudes the base disc, so SU2 monitors one
        # marker, writes no per-marker breakdown, and its total **is** the
        # forebody. The refusal below is then guarding against something that
        # cannot happen.
        #
        # This has to be told, not inferred. A single-marker history looks
        # identical whether the base is absent from the mesh or present and
        # folded into the total, and those differ by a coefficient that came
        # out as *thrust* on this pipeline -- 2.4x the forebody drag at Mach 6.
        # Guessing between them from the columns is exactly the mistake this
        # module was written to stop.
        forebody_key = lookup.get("cd")
    if forebody_key is None:
        # No per-marker breakdown: the run predates AERO_COEFF_SURF, or the
        # case carried a single marker. Either way the base cannot be
        # separated, and a total that silently includes it is the thing this
        # module exists to refuse.
        return None
    series = columns[forebody_key]
    if series.size == 0:
        return None
    tail = slice(-min(int(window), series.size), None)
    forebody = float(np.mean(series[tail]))

    normal_key = lookup.get("cl(vehicle)") or (lookup.get("cl") if forebody_only else None)
    normal = float(np.mean(columns[normal_key][tail])) if normal_key else float("nan")

    total_key = lookup.get("cd")
    su2_total = float(np.mean(columns[total_key][tail])) if total_key else float("nan")

    base_area = _base_area(body, geometry)
    if wake_resolved:
        # The base disc is a wall in this mesh, so its force is **monitored**:
        # read it, do not model it, and do not drop it either. Setting it to
        # zero here was a worse error than the double-count it replaced -- the
        # correlation is 32 % of total drag at Mach 4, so a total without the
        # base under-reports by more than a third, and it looks perfectly
        # plausible while doing it.
        #
        # Told rather than inferred, for the reason the marker split is: a
        # history looks much the same either way and the two differ by the whole
        # coefficient.
        base_key = lookup.get("cd(vehicle_base)") or lookup.get("cd[1]")
        if base_key is None:
            msg = (
                "wake_resolved was asked for but this history has no base marker "
                "column; the case was solved without one, so its base force was "
                "never separated and cannot be recovered here"
            )
            raise ValueError(msg)
        base = float(np.mean(columns[base_key][tail]))
    else:
        base = base_axial_coefficient(mach, base_area, reference_area)
    return ResolvedForces(
        forebody_axial=forebody,
        base_axial=base,
        axial=forebody + base,
        normal=normal,
        base_area=base_area,
        reference_area=reference_area,
        su2_total=su2_total,
    )


#: Transition bands that force a single regime. ``BoundaryLayer`` models
#: transition as a *band* on running-length Reynolds number, so a band placed
#: entirely below the body's Reynolds number is fully turbulent and one placed
#: entirely above it is fully laminar. Forcing the endpoints this way uses the
#: same closure for both bounds, which is what makes their difference a
#: transition bracket rather than two unrelated numbers.
def newtonian_forebody(
    body: str,
    geometry: dict[str, float],
    mach: float,
    area: float,
    gamma: float = 1.4,
    resolution: float = 1.0,
) -> float:
    """Modified-Newtonian forebody axial coefficient, from the panel geometry.

    This is not a result. It is the **independent anchor** -- it uses no solver,
    no mesh, and no history file, so it is the one number in this module that
    cannot be wrong for the same reason a CFD run is wrong. It says within
    roughly a factor of two what the answer has to be, and that is enough.

    Measured against it on 2026-09-10, the 10-degree sphere-cone:

    ====  ===========  ==========  ===========
    Mach  Newtonian    CFD         ratio
    ====  ===========  ==========  ===========
    8     0.0729       0.0826      1.13
    12    0.0732       0.336       **4.6**
    16    0.0733       0.161       **2.2**
    ====  ===========  ==========  ===========

    Mach 8 sits 13 percent high, which is the right sign and size: a slender
    cone carries more shock-layer overpressure than Newtonian allows, and skin
    friction supplies the rest. The two NEMO runs are not near it at all, which
    is how :func:`nonphysical_states` came to be written.

    Note how little the Newtonian value moves: 0.0729 to 0.0733 from Mach 8 to
    Mach 16. That flatness is the Mach independence principle, and any CFD
    forebody coefficient that nearly doubles across that span is making a claim
    the physics does not support.
    """
    from aether.cfd.geometries import build_surface

    # ``resolution`` has to reach the panel build. It did not, so the anchor
    # always integrated the baseline mesh whatever level the caller was on --
    # which made its own discretisation error look like a fixed offset instead
    # of something that converges. On a hemisphere, where the exact answer is
    # C_pmax/2, the fixed offset was -0.4278 % at every level.
    surface = build_surface(body, resolution=resolution, **geometry)
    normals = np.asarray(surface.normals, dtype=np.float64)
    areas = np.asarray(surface.areas, dtype=np.float64)
    # Rayleigh pitot: the stagnation pressure coefficient behind a normal
    # shock, which is what Newtonian scales its sine-squared law by. Using the
    # incompressible 2.0 instead would be wrong by 9 percent here and wrong by
    # a different 9 percent at every other Mach number.
    m2 = mach * mach
    stagnation = (((gamma + 1.0) ** 2 * m2) / (4.0 * gamma * m2 - 2.0 * (gamma - 1.0))) ** (
        gamma / (gamma - 1.0)
    ) * ((1.0 - gamma + 2.0 * gamma * m2) / (gamma + 1.0))
    cp_max = (2.0 / (gamma * m2)) * (stagnation - 1.0)
    # Only panels the flow can see. The base is shadowed, and Newtonian says
    # nothing about it -- which is the same division of labour the rest of this
    # module makes.
    windward = normals[:, 0] < 0.0
    cp = np.zeros(len(areas))
    cp[windward] = cp_max * normals[windward, 0] ** 2
    return float(np.sum(cp * (-normals[:, 0]) * areas) / area)


FULLY_LAMINAR: tuple[float, float] = (1.0e30, 2.0e30)
FULLY_TURBULENT: tuple[float, float] = (1.0, 2.0)


@dataclass(frozen=True)
class FrictionBracket:
    """Skin-friction axial force at both limits of the transition question."""

    laminar: float
    turbulent: float

    @property
    def width(self) -> float:
        return float(self.turbulent - self.laminar)



@dataclass(frozen=True)
class DragSplit:
    """The solver's own drag, separated into pressure and friction."""

    pressure: float
    friction: float
    wetted_area: float
    frontal_area: float
    """Projected forward-facing area, from the same normals. It should come out
    as the reference area, and if it does not the normals are wrong and neither
    number above means anything."""

    iteration: int

    @property
    def total(self) -> float:
        return self.pressure + self.friction

    @property
    def friction_share(self) -> float:
        return self.friction / self.total if self.total else float("nan")

    def summary(self) -> str:
        return (
            f"CD {self.total:.5f} = pressure {self.pressure:.5f} "
            f"+ friction {self.friction:.5f} ({self.friction_share:.0%} friction)"
        )


def drag_split(
    case_dir: Path | str,
    reference_area: float,
    freestream_pressure: float,
    mach: float,
    gamma: float = 1.4,
    marker: str = "vehicle",
) -> DragSplit | None:
    """Split the solved axial force into pressure and friction. ``None`` if it cannot.

    **This is the number that says whether a coefficient is wrong or merely
    unfamiliar.** The Mach 8 sphere-cone settles at :math:`C_D = 0.136` against
    a modified-Newtonian anchor of 0.073, and a ratio of 1.87 reads as a
    factor-of-two error -- it was called one here before it was measured.
    Newtonian is a **pressure-only** model, and on that case the split is
    0.0852 pressure against 0.0517 friction: the pressure part sits 17 % above
    Newtonian, which is the expected shock-layer excess, and the friction part
    is 38 % of the total because the wall is held 2 700 K below the recovery
    temperature at :math:`Re_L = 6.3\times10^5`.

    Taken from the **volume** snapshot rather than ``surface_flow.csv``, which
    carries only the conserved variables -- no pressure coefficient and no skin
    friction -- so the columns this needs are not in it.

    The normals come from the mesh's own winding, and the projected frontal area
    is returned alongside so a caller can check them: it must equal the
    reference area, and on the sphere-cone it does to 0.7 %.
    """
    import numpy as np

    case = Path(case_dir)
    grids = sorted(case.glob("*.su2"))
    if not grids:
        return None
    from aether.cfd.frames import completed_snapshots, snapshot_path

    finished = completed_snapshots(case)
    if finished:
        iteration, volume = finished[-1]
    else:
        volume, iteration = snapshot_path(case), -1
    if not volume.exists():
        return None

    from aether.aerodynamics.cfd.fields import read_su2_mesh, read_volume

    try:
        mesh = read_su2_mesh(grids[0])
        field = read_volume(volume)
    except (ValueError, OSError):
        return None
    if marker not in mesh.markers or mesh.size != field.size:
        return None

    faces = np.asarray(mesh.markers[marker], dtype=np.int64)
    corners = np.asarray(mesh.points, dtype=np.float64)[faces]
    # Area-weighted normal: its length is the triangle's area, so no separate
    # normalisation is needed and a degenerate face contributes nothing.
    normal = 0.5 * np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])

    dynamic = 0.5 * gamma * freestream_pressure * mach * mach
    pressure = np.asarray(field.pressure(), dtype=np.float64)
    cp = (pressure[faces].mean(axis=1) - freestream_pressure) / dynamic
    drag_pressure = -float(np.sum(cp * normal[:, 0])) / reference_area

    skin = field.fields.get("Skin_Friction_Coefficient_x")
    if skin is None:
        return None
    area = np.linalg.norm(normal, axis=1)
    friction = np.asarray(skin, dtype=np.float64)[faces].mean(axis=1)
    drag_friction = float(np.sum(friction * area)) / reference_area

    return DragSplit(
        pressure=drag_pressure,
        friction=drag_friction,
        wetted_area=float(area.sum()),
        frontal_area=float(np.maximum(-normal[:, 0], 0.0).sum()),
        iteration=int(iteration),
    )

def friction_bracket(
    case_dir: Path | str,
    body: str,
    geometry: dict[str, float],
    mach: float,
    alpha_deg: float,
    altitude_km: float,
    reference_area: float,
    reference_length: float,
) -> FrictionBracket | None:
    """Laminar and fully-turbulent skin friction, on the CFD's own pressure field.

    **Transition is not assumed in either direction.** Where it happens on this
    vehicle is an open question — ``BoundaryLayer`` defaults to a band of
    :math:`10^6` to :math:`5\\times10^6` on running-length Reynolds number and
    the vehicle sits near it — so the honest object is the pair of limits it
    cannot fall outside, which is what
    :doc:`the enclosure filter <the-filter-for-a-new-enclosed-effect>` asks for:
    two limiting physical cases, not a regression between them.

    The pressure distribution comes from the CFD run, so both bounds share the
    solved flow field and differ *only* in the friction closure. That is what
    makes the interval meaningful: a laminar CFD run and a separate turbulent
    CFD run would differ in the pressure field too, and the difference could not
    be attributed.
    """
    import numpy as np

    from aether.aerodynamics.cfd.diagnostics import surface_pressure_coefficient
    from aether.aerodynamics.composite import SkinFrictionModel
    from aether.aerodynamics.friction import BoundaryLayer
    from aether.cfd.config import FlowConditions
    from aether.cfd.geometries import build_surface

    surface_file = Path(case_dir) / "surface_flow.csv"
    if not surface_file.exists():
        return None
    flow = FlowConditions.at_altitude(
        mach=mach, altitude=1.0e3 * altitude_km, alpha=float(np.radians(alpha_deg))
    )
    density = flow.pressure / (287.058 * flow.temperature)

    points, cp_nodes = surface_pressure_coefficient(
        surface_file, flow.pressure, mach
    )
    mesh = build_surface(body, **geometry)
    centroids = np.asarray(mesh.centroids, dtype=np.float64)

    # Nearest surface node to each panel centroid. The CFD surface and the
    # panel model are the same geometry at different resolutions, so nearest
    # is well posed; a panel with no node within a body-length is a mismatch
    # worth failing on rather than interpolating through.
    try:
        from scipy.spatial import cKDTree

        distance, index = cKDTree(points).query(centroids)
    except ImportError:
        deltas = centroids[:, None, :] - points[None, :, :]
        squared = np.einsum("ijk,ijk->ij", deltas, deltas)
        index = np.argmin(squared, axis=1)
        distance = np.sqrt(squared[np.arange(centroids.shape[0]), index])
    if float(np.max(distance)) > float(reference_length):
        return None
    cp_panels = np.asarray(cp_nodes, dtype=np.float64)[index]

    bounds = []
    for band in (FULLY_LAMINAR, FULLY_TURBULENT):
        model = SkinFrictionModel(
            mesh=mesh,
            reference_area=reference_area,
            reference_length=reference_length,
            boundary_layer=BoundaryLayer(transition_reynolds=band),
        )
        result = model.solve(
            mach, float(np.radians(alpha_deg)), cp_panels,
            float(flow.temperature), float(flow.pressure), float(density),
        )
        bounds.append(float(result.axial))
    return FrictionBracket(laminar=bounds[0], turbulent=bounds[1])
