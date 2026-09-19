"""Standard figure sets for a solved CFD case, built from the case directory.

A finished run leaves a mesh, a restart and the configuration it was run
under, all in one directory. That is enough to draw the case *and* to label it
correctly, and labelling it correctly is most of why this module exists: a
flow-field figure whose Mach number is typed into the caption separately from
the one the solver used is a figure that can disagree with its own data, and
eventually does. Everything annotated here -- Mach, incidence, freestream
state, the wall markers, the ratio of specific heats -- is read back out of
``case.cfg``.

What gets drawn, and why those
------------------------------

:func:`flow_panels` draws four views of one case, chosen so that each answers
something the others cannot:

* **Schlieren** -- where the waves are. Shows shock position and structure,
  and shows nothing quantitative.
* **Mach** -- the shock layer, the expansion, and the wake as regions.
* **Pressure**, logarithmically -- spans two decades between wake and
  stagnation, so a linear scale is a black field with a bright dot.
* **Wall pressure against the Rayleigh pitot limit** -- the one panel with an
  answer that can be checked. The limit is not a comparison with theory but a
  ceiling no steady flow can exceed, so a curve crossing it locates a failure
  of the discretisation, not a disagreement about the vehicle.

The fourth is the reason the set is worth drawing at all. On the cases in this
repository it is not decorative: the Mach 8 sphere-cone puts about a tenth of
its wall nodes above a ceiling that no amount of convergence can lift.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from aether.aerodynamics.cfd.fields import (
    PERFECT_AIR,
    CaloricallyPerfect,
    Mesh,
    VolumeField,
    plane_cut,
    read_restart,
    read_su2_mesh,
    read_volume,
    surface_cut,
)
from aether.aerodynamics.closure import rayleigh_pitot_cp_max
from aether.viz import flow

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_FloatArray = NDArray[np.float64]

__all__ = [
    "CaseConditions",
    "FieldHealth",
    "HotSpot",
    "SolvedCase",
    "discover_cases",
    "field_health",
    "final_residual",
    "flow_panels",
    "hot_spots",
    "main",
    "read_case_config",
    "render_all",
    "sweep_animation",
    "sweep_frames",
]

#: The meridian plane: normal along y, so the cut plots x to the right and z
#: up. Every figure here uses it, and at zero incidence it is the symmetry
#: plane, which is the plane the flow is actually two-dimensional in.
MERIDIAN = ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))


@dataclass(frozen=True)
class CaseConditions:
    """The flight condition a case was run at, as the solver was told it.

    Two gas models, because a hypersonic case needs two and conflating them is
    wrong in both directions.

    The **freestream** is cold, undissociated air -- 264 K at 45 km -- and is
    calorically perfect there whatever the solver is doing downstream. So
    dynamic pressure, :math:`C_p` and freestream speed are always defined,
    including for a five-species nonequilibrium run, and refusing to compute
    them is simply wrong.

    The **field** is another matter. Through a Mach 20 shock layer air
    dissociates, no single :math:`\\gamma` describes it, and deriving
    temperature from conservative variables with one is wrong by a factor.
    :attr:`gas` is ``None`` for such a case, and only the operations that
    reach into the field consult it.
    """

    mach: float
    alpha: float
    pressure: float
    temperature: float
    walls: tuple[str, ...]
    gas: CaloricallyPerfect | None = PERFECT_AIR
    """Gas model valid *through the field*, or ``None`` where none is."""

    freestream_gas: CaloricallyPerfect = PERFECT_AIR
    """Gas model at the freestream, which is calorically perfect regardless.

    Separate from :attr:`gas` because the freestream of a nonequilibrium case
    is not itself nonequilibrium. Making them one attribute meant a Mach 20
    biconic could not be told its own dynamic pressure.
    """

    solver: str = "EULER"
    fluid_model: str = "STANDARD_AIR"

    @property
    def perfect_gas(self) -> bool:
        """Whether the perfect-gas relations describe the *field*."""
        return self.gas is not None

    def require_gas(self) -> CaloricallyPerfect:
        """The field's gas model, or a refusal naming why there is not one.

        Only for operations that reach into the shock layer. Freestream
        quantities use :attr:`freestream_gas` and never need this.
        """
        if self.gas is None:
            raise ValueError(
                f"case is {self.fluid_model} on {self.solver}, so no single gamma "
                "describes its shock layer; read the solver's own primitives from "
                "a volume file instead of deriving them from conservative variables"
            )
        return self.gas

    @property
    def gamma(self) -> float:
        """Ratio of specific heats in the field, where one is defined."""
        return self.require_gas().gamma

    @property
    def dynamic_pressure(self) -> float:
        """:math:`q = \\tfrac{1}{2}\\gamma p M^2`, from the freestream."""
        return 0.5 * self.freestream_gas.gamma * self.pressure * self.mach**2

    def pressure_coefficient(self, pressure: _FloatArray) -> _FloatArray:
        """Convert absolute pressure to :math:`C_p`."""
        return np.asarray((pressure - self.pressure) / self.dynamic_pressure, dtype=np.float64)

    @property
    def speed(self) -> float:
        """Freestream speed, needed by the equilibrium-air normal shock."""
        gas = self.freestream_gas
        return self.mach * float(np.sqrt(gas.gamma * gas.gas_constant * self.temperature))

    @property
    def label(self) -> str:
        """A caption fragment, from the configuration rather than from memory."""
        gas = "" if self.perfect_gas else f", {self.fluid_model}"
        return (
            f"M {self.mach:g}, alpha {self.alpha:g} deg, "
            f"{self.pressure:g} Pa, {self.temperature:g} K{gas}"
        )


#: Above this Mach number the perfect-gas stagnation-pressure ceiling is
#: replaced by the equilibrium-air one where Cantera is available. It is the
#: threshold :func:`~aether.cfd.config.regime_for` already switches the
#: solver at, and using a second, different number here would mean the picture
#: and the run disagreed about where real-gas effects begin.
REAL_GAS_MACH = 10.0


def pitot_cp_max(conditions: CaseConditions) -> tuple[float, str]:
    """The largest attainable wall :math:`C_p`, and what model produced it.

    Below :data:`REAL_GAS_MACH` this is the Rayleigh-Pitot relation for a
    perfect gas. Above it, that relation is the wrong ceiling: dissociation in
    the shock layer changes the stagnation state, and the perfect-gas value
    understates the real one -- by about 1.8 % at Mach 8, 4.5 % at Mach 20 and
    5.1 % at Mach 27 on the freestream in this repository. Small next to the
    orders of magnitude the divergence screen looks for, and not small next to
    the few percent the wall panel is drawn to resolve.

    Falls back to the perfect-gas value, saying so in the returned label, when
    Cantera is not installed or the equilibrium solve does not converge. The
    label goes into the figure, because a ceiling drawn from a different model
    than the caption claims is worse than a ceiling that is slightly wrong.
    """
    gas = conditions.freestream_gas
    perfect = rayleigh_pitot_cp_max(conditions.mach, gas.gamma)
    if conditions.mach < REAL_GAS_MACH:
        return perfect, f"Rayleigh pitot limit ($\\gamma$={gas.gamma:g})"
    try:
        from aether.aerodynamics.realgas import EquilibriumAir

        real = EquilibriumAir().cp_max(
            conditions.temperature, conditions.pressure, conditions.speed
        )
    except Exception:
        return perfect, f"pitot limit (perfect gas, $\\gamma$={gas.gamma:g} -- no real-gas solve)"
    return float(real), "pitot limit (equilibrium air)"


def read_case_config(path: Path | str) -> CaseConditions:
    """Read the flight condition and the gas model out of an SU2 ``.cfg``.

    Deliberately not a general configuration parser -- it reads the handful of
    keys a figure needs to label itself honestly, and raises if one is missing
    rather than substituting a default. A default freestream pressure would
    silently rescale every :math:`C_p` in the figure.

    The gas model is read, not assumed. ``GAMMA_VALUE`` and ``GAS_CONSTANT``
    both come from the case, so a run on a different gas -- or on air with a
    different constant -- is drawn on its own terms. Where the case is a
    nonequilibrium one (``FLUID_MODEL= SU2_NONEQ``, the ``NEMO_*`` solvers)
    there is no single ratio of specific heats to read, and none is invented:
    :attr:`CaseConditions.gas` comes back ``None`` and everything that would
    need one refuses. Inventing :math:`\\gamma = 1.4` there is the specific
    failure this guards against -- it is the regime the suite exists for, and
    the regime in which that number is worst.
    """
    text = Path(path).read_text()

    def find(key: str) -> str | None:
        match = re.search(rf"^\s*{key}\s*=\s*(.+?)\s*$", text, re.MULTILINE)
        return None if match is None else match.group(1)

    def value(key: str) -> str:
        found = find(key)
        if found is None:
            raise ValueError(f"{path} has no {key}=; a figure cannot label itself without it")
        return found

    markers = re.findall(r"\w+", find("MARKER_EULER") or find("MARKER_PLOTTING") or "")
    if not markers:
        raise ValueError(f"{path} names no wall markers; a body cannot be outlined without them")

    solver = find("SOLVER") or "EULER"
    fluid_model = find("FLUID_MODEL") or "STANDARD_AIR"
    nonequilibrium = solver.upper().startswith("NEMO") or "NONEQ" in fluid_model.upper()
    gas = (
        None
        if nonequilibrium
        else CaloricallyPerfect(
            gamma=float(find("GAMMA_VALUE") or PERFECT_AIR.gamma),
            gas_constant=float(find("GAS_CONSTANT") or PERFECT_AIR.gas_constant),
        )
    )

    return CaseConditions(
        mach=float(value("MACH_NUMBER")),
        alpha=float(value("AOA")),
        pressure=float(value("FREESTREAM_PRESSURE")),
        temperature=float(value("FREESTREAM_TEMPERATURE")),
        walls=tuple(markers),
        gas=gas,
        solver=solver,
        fluid_model=fluid_model,
    )


@dataclass(frozen=True)
class FieldHealth:
    """Whether a solution is worth drawing as a result.

    A diverged SU2 run does not announce itself. ``anchor/medium`` in this
    repository exits ``Exit Success``, writes a full restart, and carries a
    density of :math:`2\times10^{-14}` next to one of :math:`3.7\times10^{4}`
    with a peak pressure of 29 GPa -- and drawn without comment it produces a
    figure that looks like a flow field. The screen is physical rather than
    numerical, because the numerical one is ambiguous here: these cases
    converge on the drag coefficient, not on the residual, so a residual of
    :math:`10^{-0.1}` is normal and says nothing.

    What is unambiguous is that no steady flow puts static pressure above the
    stagnation pressure behind a normal shock. Exceeding it slightly is an
    under-resolved shock, which is a result with a caveat; exceeding it by
    five orders of magnitude is not a result.
    """

    min_density: float
    max_pressure_ratio: float
    final_residual: float | None
    finite: bool

    #: Above this multiple of the pitot pressure a field is not a solution.
    #: Set well clear of the few-fold overshoot a captured shock produces, so
    #: that it separates divergence from under-resolution rather than
    #: relabelling one as the other.
    threshold = 10.0

    @property
    def diverged(self) -> bool:
        """True when the field cannot be a steady solution of anything."""
        return (
            not self.finite or self.min_density <= 0.0 or self.max_pressure_ratio > self.threshold
        )

    @property
    def note(self) -> str | None:
        """A one-line warning for a figure, or ``None`` when the field is sane."""
        if not self.diverged:
            return None
        residual = "" if self.final_residual is None else f", final rms {self.final_residual:+.2f}"
        return (
            f"DIVERGED -- peak pressure {self.max_pressure_ratio:.3g}x the pitot "
            f"limit, min density {self.min_density:.3g}{residual}"
        )


def final_residual(history: Path | str) -> float | None:
    """The last ``rms[Rho]`` in an SU2 history, without reading the whole file.

    These histories reach 25 MB, and a figure that parses one in full every
    time it is drawn is a figure nobody redraws. Only the header and the last
    line are needed.
    """
    path = Path(history)
    if not path.exists():
        return None
    with path.open("rb") as handle:
        header = handle.readline().decode().split(",")
        try:
            names = [name.strip().strip('"').lower() for name in header]
            # NEMO writes one density residual per species, so "rms[rho]" is
            # absent there; take the first species rather than raising.
            candidates = [n for n in names if n == "rms[rho]"] or sorted(
                n for n in names if n.startswith("rms[rho_")
            )
            if not candidates:
                raise ValueError(f"{path} carries no density residual column")
            column = names.index(candidates[0])
        except ValueError:
            return None
        handle.seek(0, 2)
        size = handle.tell()
        window = min(size, 8192)
        handle.seek(size - window)
        lines = [
            line for line in handle.read().decode(errors="ignore").splitlines() if line.strip()
        ]
    for line in reversed(lines):
        cells = line.split(",")
        if len(cells) > column:
            try:
                return float(cells[column])
            except ValueError:
                continue
    return None


def field_health(
    field: VolumeField, conditions: CaseConditions, history: Path | str
) -> FieldHealth:
    """Screen a solution before it is drawn as though it meant something.

    The ceiling comes from :func:`pitot_cp_max`, so above
    :data:`REAL_GAS_MACH` it is the equilibrium-air one. That matters less
    here than in the wall panel -- the threshold is a factor of ten and the
    real-gas correction is a few percent -- but using two different ceilings
    for the same physical bound would be a way of eventually disagreeing with
    ourselves about which solutions are usable.
    """
    # The solver's own pressure where the case wrote one, which is what lets
    # a nonequilibrium field be screened at all: deriving it would need a
    # gamma the shock layer does not have.
    pressure = (
        field.pressure() if field.has_primitives else field.pressure(conditions.require_gas())
    )
    density = field.density
    pitot = conditions.pressure + pitot_cp_max(conditions)[0] * conditions.dynamic_pressure
    finite = bool(np.isfinite(pressure).all() and np.isfinite(density).all())
    return FieldHealth(
        min_density=float(density.min()),
        max_pressure_ratio=float(pressure.max() / pitot) if finite else float("inf"),
        final_residual=final_residual(history),
        finite=finite,
    )


@dataclass(frozen=True)
class SolvedCase:
    """One run with a field on disk: where its files are, and what it ran at.

    ``stage`` distinguishes a case that finished its main run from one that
    has only got through the prelude -- the robust first-order startup pass.
    Both leave a readable field and both are worth looking at, but they are
    not the same claim, and a figure that does not say which it is invites the
    prelude's shock to be read as the converged one.
    """

    name: str
    mesh: Path
    solution: Path
    config: Path
    conditions: CaseConditions
    stage: str = "converged"

    @property
    def is_prelude(self) -> bool:
        """True when only the startup pass has been written."""
        return self.stage == "prelude"

    @property
    def has_volume(self) -> bool:
        """Whether this case wrote the solver's own primitives."""
        return self.solution.suffix == ".dat" and self.solution.name.startswith("volume_")

    def load(self) -> tuple[Mesh, VolumeField]:
        """Read the mesh and the volume solution, checked against each other.

        A volume file is preferred over a restart wherever one exists, because
        it carries pressure, temperature and Mach as the solver computed them
        rather than as something to be reconstructed under an assumed gas
        model. Both are read against the same ``.su2`` connectivity, and both
        share its node order.
        """
        grid = read_su2_mesh(self.mesh)
        field = read_volume(self.solution) if self.has_volume else read_restart(self.solution)
        if grid.size != field.size:
            raise ValueError(
                f"{self.mesh} has {grid.size} nodes but {self.solution} has "
                f"{field.size}; these files are not from the same case"
            )
        return grid, field

    def health(self, field: VolumeField) -> FieldHealth:
        """Screen this case's field against the pitot bound and its residual."""
        return field_health(field, self.conditions, self.solution.parent / "history.csv")


def discover_cases(root: Path | str) -> list[SolvedCase]:
    """Find every case under ``root`` that has a field that can be drawn.

    A case needs a mesh, a configuration and a solution, and each of the three
    has a fallback that is a different thing rather than a worse copy of the
    same thing:

    * ``case.cfg``, else ``prelude.cfg`` -- both carry the flight condition.
    * ``restart_flow.dat``, else ``solution_flow.dat`` -- the main run's
      output, else the prelude's. The second is marked ``stage="prelude"``
      rather than passed off as the first.

    Runs that never wrote a field at all are skipped silently. An unfinished
    run is a normal state of this directory, not an error in it -- of the
    cases here, one has logs and no solution, and reporting that as a failure
    every time figures are built would train the failure to be ignored.
    """
    root = Path(root)
    found: list[SolvedCase] = []
    seen: dict[str, int] = {}
    directories = sorted({path.parent for path in root.rglob("*.cfg")})
    for directory in directories:
        config = next(
            (directory / n for n in ("case.cfg", "prelude.cfg") if (directory / n).exists()), None
        )
        meshes = sorted(directory.glob("*.su2"))
        volume = directory / "volume_flow.dat"
        restart, solution = directory / "restart_flow.dat", directory / "solution_flow.dat"
        # Preference order, and it is not arbitrary: the volume file is the
        # only one of the three that says what the gas actually did.
        field = next(
            (path for path in (volume, restart, solution) if path.exists()),
            None,
        )
        if config is None or field is None or not meshes:
            continue

        parts = directory.relative_to(root).parts
        name = "/".join(parts[:-2]) if len(parts) > 2 and parts[-2] == "cases" else "/".join(parts)
        # Two flight conditions in one family would otherwise collide; keep
        # the case directory only where it is doing work.
        if name in seen:
            name = f"{name}/{parts[-1]}"
        seen[name] = seen.get(name, 0) + 1

        found.append(
            SolvedCase(
                name=name,
                mesh=meshes[0],
                solution=field,
                config=config,
                conditions=read_case_config(config),
                stage=(
                    "converged"
                    if field in (volume, restart) and config.name == "case.cfg"
                    else "prelude"
                ),
            )
        )
    return found


def _extent(mesh: Mesh, walls: Sequence[str], margin: float) -> tuple[float, float, float, float]:
    """A view box around the body, not around the domain.

    The farfield is twenty body lengths across and empty. Framing on it puts
    the vehicle in a handful of pixels, which is the default a caller gets by
    doing nothing, so the default here is the body instead.
    """
    available = [tag for tag in walls if tag in mesh.markers and mesh.markers[tag].size]
    points = mesh.marker_points(*available) if available else mesh.points
    low, high = points.min(axis=0), points.max(axis=0)
    length = float(high[0] - low[0]) or 1.0
    pad = margin * length
    half = 0.5 * (high[2] + low[2])
    reach = max(pad, 0.5 * float(high[2] - low[2]) + pad)
    return (float(low[0]) - pad, float(high[0]) + 2.0 * pad, half - reach, half + reach)


def flow_panels(
    case: SolvedCase,
    *,
    plane: tuple[Any, Any] = MERIDIAN,
    margin: float = 0.45,
    figsize: tuple[float, float] = (12.0, 8.5),
    streamlines: bool = False,
) -> Figure:
    """The four-panel view of one solved case.

    Returns the figure rather than writing it, so a caller can restyle or
    compose it; :func:`render_all` writes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    mesh, field = case.load()
    origin, normal = plane
    cut = plane_cut(mesh, origin, normal)
    walls = [tag for tag in case.conditions.walls if tag in mesh.markers]
    body = surface_cut(mesh, walls, origin, normal)

    conditions = case.conditions
    # The solver's own primitives where the case wrote them. That is what lets
    # a nonequilibrium field be drawn at all: deriving pressure or Mach would
    # need a gamma its shock layer does not have.
    if field.has_primitives:
        pressure, mach = field.pressure(), field.mach()
    else:
        gas = conditions.require_gas()
        pressure, mach = field.pressure(gas), field.mach(gas)
    magnitude = flow.schlieren_magnitude(mesh, field)

    figure, axes = plt.subplots(2, 2, figsize=figsize)
    stage = "  [prelude pass only]" if case.is_prelude else ""
    figure.suptitle(f"{case.name}  --  {case.conditions.label}{stage}", fontsize=11)
    warning = case.health(field).note
    if warning is not None:
        figure.text(
            0.5,
            0.945,
            warning,
            ha="center",
            fontsize=9,
            color="white",
            bbox={"facecolor": "firebrick", "edgecolor": "none", "pad": 3.0},
        )
    box = _extent(mesh, walls, margin)

    flow.schlieren(axes[0, 0], cut, mesh.points, magnitude)
    axes[0, 0].set_title("numerical Schlieren", fontsize=9)

    contours = flow.field_map(
        axes[0, 1],
        cut,
        mesh.points,
        mach,
        cmap="magma",
        levels=np.linspace(0.0, conditions.mach * 1.05, 128),
        extend="max",
    )
    figure.colorbar(contours, ax=axes[0, 1], label="Mach", fraction=0.046)
    axes[0, 1].set_title("Mach number", fontsize=9)

    # Pressure as a ratio to freestream, on a log scale bounded below by a
    # decade under freestream and above by the Rayleigh pitot ratio -- the
    # highest stagnation pressure the flow can reach. Both bounds come from
    # the flight condition rather than from this solution's own extremes, so
    # two cases at one Mach number are directly comparable and a wake cell at
    # 5 Pa cannot set the exposure for the whole field. Anything above the top
    # bound is saturated, and being above it is exactly what the wall panel
    # measures.
    ratio = pressure / conditions.pressure
    ceiling = 1.0 + pitot_cp_max(conditions)[0] * conditions.dynamic_pressure / conditions.pressure
    pressures = flow.field_map(
        axes[1, 0],
        cut,
        mesh.points,
        ratio,
        cmap="inferno",
        levels=np.geomspace(0.1, ceiling, 128),
        norm=LogNorm(vmin=0.1, vmax=ceiling),
        extend="both",
    )
    decades = [tick for tick in (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0) if 0.1 <= tick <= ceiling]
    bar = figure.colorbar(
        pressures, ax=axes[1, 0], label=r"$p/p_\infty$", fraction=0.046, ticks=decades
    )
    bar.ax.set_yticklabels([f"{tick:g}" for tick in decades])
    axes[1, 0].set_title("pressure, relative to freestream (log)", fontsize=9)

    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        if streamlines:
            flow.streamlines(ax, cut, mesh.points, field.velocity, color="0.35", linewidth=0.5)
        flow.outline(ax, body, mesh.points, color="white", linewidth=1.0)
        ax.set_aspect("equal")
        ax.set_xlim(box[0], box[1])
        ax.set_ylim(box[2], box[3])
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("z [m]", fontsize=8)

    _wall_panel(axes[1, 1], case, mesh, field, body, pressure)
    figure.tight_layout()
    return figure


def _wall_panel(
    ax: Any,
    case: SolvedCase,
    mesh: Mesh,
    field: VolumeField,
    body: Any,
    pressure: _FloatArray,
) -> None:
    """Wall :math:`C_p` against the ceiling no steady flow can exceed."""
    conditions = case.conditions
    station, wall = flow.wall_curve(body, mesh.points, conditions.pressure_coefficient(pressure))
    limit, model = pitot_cp_max(conditions)

    ax.plot(station, wall, ".", markersize=1.6, color="tab:blue", label="wall $C_p$")
    ax.axhline(limit, color="tab:red", linestyle="--", linewidth=1.2, label=model)
    over = wall > limit * (1.0 + 1e-3)
    if over.any():
        ax.plot(station[over], wall[over], ".", markersize=3.0, color="tab:red")
        share = 100.0 * over.sum() / over.size
        ax.set_title(
            f"wall $C_p$ -- {share:.1f}% of the profile is unattainable",
            fontsize=9,
            color="tab:red",
        )
    else:
        ax.set_title("wall $C_p$ -- within the pitot limit everywhere", fontsize=9)
    ax.set_xlabel("x [m]", fontsize=8)
    ax.set_ylabel("$C_p$", fontsize=8)
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.25)


def sweep_frames(
    case: SolvedCase,
    *,
    count: int = 90,
    span: tuple[float, float] | None = None,
    axis: int = 1,
) -> list[tuple[float, Any]]:
    """Cut planes stepped across the body, for an animation of one solution.

    A steady solution has nothing to animate in time, so what moves is the
    observer: the cutting plane travels across the vehicle and the
    cross-section opens and closes with it. It is the one animation these runs
    can honestly support, and it shows something a single slice cannot -- that
    the shock is a surface, not the curve a meridian cut makes it look like.

    ``axis`` selects the sweep direction: ``1`` walks the meridian plane
    sideways through the body, ``0`` takes cross-sections down the body axis
    and shows the shock as a ring.

    Returns ``(offset, cut)`` pairs.
    """
    mesh, _ = case.load()
    normal = np.eye(3)[axis]
    if span is None:
        walls = [tag for tag in case.conditions.walls if tag in mesh.markers]
        column = (mesh.marker_points(*walls) if walls else mesh.points)[:, axis]
        low, high = float(column.min()), float(column.max())
        middle, reach = 0.5 * (low + high), 0.49 * (high - low)
        span = (middle - reach, middle + reach) if reach > 0 else (low, high)
    offsets = np.linspace(span[0], span[1], count)
    return [(float(offset), plane_cut(mesh, normal * offset, normal)) for offset in offsets]


def sweep_animation(
    case: SolvedCase,
    path: Path | str,
    *,
    count: int = 90,
    axis: int = 1,
    fps: int = 20,
    dpi: int = 120,
    figsize: tuple[float, float] = (7.0, 5.0),
    progress: bool = True,
) -> Path:
    """Write a Schlieren sweep through one solution to ``.mp4`` or ``.gif``.

    The exposure is fixed once, from the whole volume, and every frame is
    drawn against it. Rescaling per frame is the obvious thing to do and it is
    wrong: each slice would be normalised to its own strongest gradient, so
    the shock would hold a constant brightness while the flow around it
    pulsed, which reads as the solution changing when it is the exposure
    changing.

    The axis limits are fixed for the same reason. Writer is chosen from the
    suffix -- ``ffmpeg`` for ``.mp4``, Pillow for ``.gif``.

    ``progress`` draws a bar over the frames. On by default: ninety frames of
    a large cut is a minute or more of silence, and silence is
    indistinguishable from a hang.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, PillowWriter

    path = Path(path)
    mesh, field = case.load()
    magnitude = flow.schlieren_magnitude(mesh, field)
    # One exposure for the sequence, from the volume rather than from a slice.
    reference = float(np.percentile(magnitude, 99.5))
    frames = sweep_frames(case, count=count, axis=axis)

    walls = [tag for tag in case.conditions.walls if tag in mesh.markers]
    normal = np.eye(3)[axis]

    # One view box for the whole sweep, in the cutting plane's own axes. It
    # has to come from the basis rather than from world x and z, because for a
    # cross-section sweep those are not the axes the picture is drawn in --
    # and a box in the wrong axes silently frames empty space.
    basis = frames[0][1].basis if frames else np.delete(np.eye(3), axis, axis=0)
    reference_points = mesh.marker_points(*walls) if walls else mesh.points
    planar = reference_points @ basis.T
    low, high = planar.min(axis=0), planar.max(axis=0)
    pad = 0.45 * float(max(high - low, default=1.0) or 1.0)
    box = (low[0] - pad, high[0] + pad, low[1] - pad, high[1] + pad)
    labels = [name for index, name in enumerate("xyz") if index != axis]

    figure, ax = plt.subplots(figsize=figsize)
    writer_class = PillowWriter if path.suffix.lower() == ".gif" else FFMpegWriter
    writer = writer_class(fps=fps)

    stepped = _tracked(frames, f"{case.name} sweep", progress)
    with writer.saving(figure, str(path), dpi):
        for offset, cut in stepped:
            ax.clear()
            if cut.size:
                flow.schlieren(ax, cut, mesh.points, magnitude, reference=reference)
                body = surface_cut(mesh, walls, normal * offset, normal)
                if body.size:
                    flow.outline(ax, body, mesh.points, color="tab:red", linewidth=1.0)
            ax.set_aspect("equal")
            ax.set_xlim(box[0], box[1])
            ax.set_ylim(box[2], box[3])
            ax.set_title(f"{case.name}  --  {'xyz'[axis]} = {offset:+.3f} m", fontsize=10)
            ax.set_xlabel(f"{labels[0]} [m]", fontsize=8)
            ax.set_ylabel(f"{labels[1]} [m]", fontsize=8)
            writer.grab_frame()
    plt.close(figure)
    return path


def render_all(
    root: Path | str,
    output: Path | str,
    *,
    dpi: int = 140,
    formats: Sequence[str] = ("png",),
    streamlines: bool = False,
) -> list[Path]:
    """Draw the panel set for every case under ``root``; return what was written.

    One case failing does not stop the rest. A results tree accumulates runs
    in every state, and a batch that aborts on the first unreadable one is a
    batch nobody runs twice.
    """
    import time

    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    cases = discover_cases(root)
    for index, case in enumerate(cases, start=1):
        stem = case.name.replace("/", "-")
        # Announced before the work, not after. A million-cell case takes
        # several seconds to read and draw, and a line printed on completion
        # means the one you are waiting on is the one you cannot see.
        print(f"  [{index}/{len(cases)}] {case.name} ({case.stage}) ...", end="", flush=True)
        started = time.monotonic()
        try:
            figure = flow_panels(case, streamlines=streamlines)
        except Exception as error:
            print(f" {type(error).__name__}: {error}")
            continue
        for suffix in formats:
            path = output / f"{stem}.{suffix}"
            figure.savefig(path, dpi=dpi, bbox_inches="tight")
            written.append(path)
        plt.close(figure)
        print(f" {time.monotonic() - started:5.1f}s", flush=True)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """Render a results tree from the command line.

    With no ``--animate``, draws the panel set for every case found. With it,
    writes one Schlieren sweep instead.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Flow-field figures for solved CFD cases.")
    from aether.paths import cfd_results

    parser.add_argument("--root", type=Path, default=cfd_results())
    parser.add_argument("--output", type=Path, default=Path("figures/cfd"))
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--format", dest="formats", action="append", default=None)
    parser.add_argument("--streamlines", action="store_true")
    parser.add_argument("--list", action="store_true", help="name the cases and stop")
    parser.add_argument("--animate", metavar="CASE", help="sweep one case instead of drawing all")
    parser.add_argument("--axis", type=int, default=1, choices=(0, 1, 2))
    parser.add_argument("--frames", type=int, default=90)
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")

    if args.list:
        for case in discover_cases(args.root):
            print(f"{case.name}\t{case.stage}\t{case.conditions.label}")
        return 0

    if args.animate:
        cases = {case.name: case for case in discover_cases(args.root)}
        if args.animate not in cases:
            parser.error(f"no case named {args.animate!r}; known: {', '.join(sorted(cases))}")
        args.output.mkdir(parents=True, exist_ok=True)
        stem = args.animate.replace("/", "-")
        suffix = (args.formats or ["mp4"])[0]
        path = sweep_animation(
            cases[args.animate],
            args.output / f"{stem}-sweep.{suffix}",
            axis=args.axis,
            count=args.frames,
            dpi=args.dpi,
        )
        print(path)
        return 0

    written = render_all(
        args.root,
        args.output,
        dpi=args.dpi,
        formats=tuple(args.formats or ("png",)),
        streamlines=args.streamlines,
    )
    print(f"{len(written)} file(s) -> {args.output}")
    return 0 if written else 1


def _tracked(items: Sequence[Any], label: str, enabled: bool) -> Iterable[Any]:
    """Wrap a sequence in a progress bar where one is wanted and available.

    tqdm is a dependency, but a missing or non-interactive one should not stop
    an animation from rendering -- so this degrades to the bare sequence
    rather than failing.
    """
    if not enabled:
        return items
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - tqdm is declared
        return items
    tracked: Iterable[Any] = tqdm(items, desc=label, unit="frame", leave=False)
    return tracked


@dataclass(frozen=True)
class HotSpot:
    """Where a field is doing something no steady flow can do."""

    count: int
    """Nodes above the stagnation-pressure ceiling."""

    fraction: float
    peak_ratio: float
    """Highest pressure, as a multiple of the pitot ceiling."""

    location: tuple[float, float, float]
    """Where that peak is."""

    temperature: float
    """Temperature there, which is usually the more alarming number."""

    extent: tuple[float, float]
    """Axial span of the offending region: how localised it is."""

    def describe(self) -> str:
        if not self.count:
            return "no nodes above the pitot ceiling"
        x, y, z = self.location
        return (
            f"{self.count} nodes ({100 * self.fraction:.2f}%) above the pitot ceiling; "
            f"peak {self.peak_ratio:.1f}x at x={x:.4f} y={y:.4f} z={z:.4f}, "
            f"T {self.temperature:.0f} K; region spans x {self.extent[0]:.4f}"
            f"..{self.extent[1]:.4f}"
        )


def hot_spots(mesh: Mesh, field: VolumeField, conditions: CaseConditions) -> HotSpot:
    """Find where a solution exceeds what steady flow allows, and say where.

    SU2 reports "N points are not physical" and no location, which is the one
    piece of information that would make it actionable. This supplies it: the
    ceiling is the stagnation pressure behind a normal shock -- equilibrium
    air above Mach 10 -- and anything above it is a captured shock spread over
    too few cells rather than a fact about the vehicle.

    The **axial extent** is the number worth reading. A region a few
    millimetres across at the nose is an under-resolved stagnation point,
    cured by refining there. One spread over the whole body is not a
    resolution problem and refining will not fix it.
    """
    pressure = (
        field.pressure() if field.has_primitives else field.pressure(conditions.require_gas())
    )
    ceiling = conditions.pressure + pitot_cp_max(conditions)[0] * conditions.dynamic_pressure
    over = pressure > ceiling
    if not bool(over.any()):
        return HotSpot(0, 0.0, float(pressure.max() / ceiling), (0.0, 0.0, 0.0), 0.0, (0.0, 0.0))

    peak = int(np.argmax(pressure))
    stations = mesh.points[over, 0]
    temperature = field.temperature() if field.has_primitives else None
    return HotSpot(
        count=int(over.sum()),
        fraction=float(over.mean()),
        peak_ratio=float(pressure[peak] / ceiling),
        location=tuple(float(v) for v in mesh.points[peak]),  # type: ignore[arg-type]
        temperature=float(temperature[peak]) if temperature is not None else float("nan"),
        extent=(float(stations.min()), float(stations.max())),
    )
