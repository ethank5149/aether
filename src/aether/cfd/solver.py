"""Three-dimensional SU2 as a coefficient source for the table machinery.

:class:`SU2Solver` satisfies the same two-argument
:class:`~aether.aerodynamics.tables.Solver` protocol the panel method does, so
it drops into :class:`~aether.aerodynamics.tables.SweepRun` and produces an
:class:`~aether.aerodynamics.tables.AeroTable` that is interchangeable with a
panel-built one — including its checkpointing, which is what makes a CFD sweep
survivable. The difference is four orders of magnitude in cost per point, and
that difference is the whole reason the checkpoint exists.

Body axes, and where they come from
-----------------------------------

The table stores body-axis coefficients: :math:`C_A` positive aft,
:math:`C_N` positive nose-up, :math:`C_m` about the reference point. SU2
reports both those (``CFx``, ``CFz``, ``CMy``) and the wind-axis pair
(``CL``, ``CD``), and the body-axis triple is read here rather than the wind
one. Reading ``CL``/``CD`` and rotating them back through the angle of attack
would give the same answer and one more place to get the rotation backwards;
SU2's ``CFx``/``CFz`` are already in the frame the mesh was written in, which
is the frame the table is defined in.

What this does *not* silently supply
------------------------------------

An Euler run has no boundary layer, so its axial force is missing skin
friction entirely. That is not repaired here. It is what
:class:`~aether.aerodynamics.composite.SkinFrictionModel` is for, and
:func:`~aether.aerodynamics.build.build_solver` already composes it; a CFD
solver that quietly added a friction estimate to its own output would make the
two sources disagree in a way that looks like a CFD error.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from aether.aerodynamics.cfd.meshing import (
    MeshResult,
    RefinementBall,
    ViscousSizing,
    inviscid_domain,
    shock_layer_sizing,
    viscous_domain,
)
from aether.aerodynamics.cfd.su2 import find_su2
from aether.aerodynamics.tables import Coefficients
from aether.cfd.config import (
    FlowConditions,
    Numerics,
    PitchingMotion,
    ReferenceFrame,
    Regime,
    SU2Case,
)

__all__ = ["SU2Run", "SU2Solver", "find_mpirun", "mesh_key", "read_history"]

_FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SU2Run:
    """One finished (or failed) SU2 case and what came out of it."""

    mach: float
    alpha: float
    coefficients: Coefficients
    converged: bool
    residual: float
    iterations: int
    directory: Path
    diverged: bool = False
    """The solver blew up, as opposed to merely not settling.

    Worth separating, because the two call for opposite responses and look
    identical in a table of ``converged=False``. A run that has not settled
    wants more iterations; a run that produced a NaN wants a smaller CFL, and
    giving it more iterations wastes hours to reach the same NaN. Its
    coefficients are not a partial answer to be reported with a caveat — they
    are whatever the last finite iterate happened to be, which on this vehicle
    was a *negative* lift coefficient at six degrees of incidence.
    """
    mesh: MeshResult | None = None
    history: dict[str, _FloatArray] = field(default_factory=dict, repr=False)
    reference_area: float = float("nan")
    """The area the coefficients are non-dimensionalised on (m²)."""
    forebody_axial: float = float("nan")
    """:math:`C_A` from the forebody marker alone, when the base was split off.

    The number to compare against theory or to build a table on. The total
    includes an Euler base pressure, which is not a physical quantity.
    """
    base_axial: float = float("nan")
    """:math:`C_A` contributed by the base disc. See :attr:`forebody_axial`."""
    forebody_normal: float = float("nan")
    """:math:`C_N` from the forebody alone — the one to build lift from."""
    base_normal: float = float("nan")
    """:math:`C_N` from the base disc. An artefact at incidence, like the axial term."""
    base_area: float = float("nan")
    """Wetted area of the base patch (m²), for substituting a correlated base drag."""

    def corrected_axial(self, gamma: float = 1.4) -> float:
        r""":math:`C_A` with the Euler base **replaced** by a correlation.

        The forebody force from the solver, plus a base drag from
        :func:`~aether.aerodynamics.closure.base_axial_coefficient`. This is
        the number to quote: the raw total carries an Euler base pressure that
        is not a physical quantity, and on a Mach 8 sphere-cone that term was
        :math:`C_p = +0.34` — positive, on a base — against a correlated
        :math:`-0.016`, and swung 27 % between grid levels.

        Substituting rather than computing is the honest treatment of a
        quantity the governing equations do not contain.
        """
        from aether.aerodynamics.closure import base_axial_coefficient

        if not np.isfinite(self.forebody_axial) or not np.isfinite(self.base_area):
            return float("nan")
        return float(
            self.forebody_axial
            + base_axial_coefficient(self.mach, self.base_area, self.reference_area, gamma)
        )
    forebody_sigma: float = float("nan")
    r"""Standard error of :attr:`forebody_axial` (absolute, coefficient units).

    The run reports a *time-averaged* coefficient, so this is the bar that
    belongs on it. Quote results as
    :math:`C_A = \text{forebody\_axial} \pm \text{forebody\_sigma}`;
    see :func:`_mean_uncertainty` for why the sample count is the number of
    oscillation periods and not the window length.
    """
    base_sigma: float = float("nan")
    """Standard error of :attr:`base_axial`. Diagnostic — the base is replaced,
    not reported. See :meth:`corrected_axial`."""
    settled_periods: float = float("nan")
    """Oscillation periods resolved in the window: the independent-sample count."""
    force_drift: float = float("nan")
    """Fractional drift of the force mean across the convergence window."""
    force_ripple: float = float("nan")
    """Fractional peak-to-peak ripple over the same window. Diagnostic, not a gate."""

    def wind_axes(self, forebody_only: bool = True) -> tuple[float, float]:
        """Lift and drag coefficients at this run's incidence.

        Forebody-only by default. Both the axial *and* the normal contribution
        of an Euler base are artefacts, and an L/D built from them describes
        nothing: on a Mach 8 sphere-cone at six degrees, including them gave a
        negative lift coefficient at positive incidence.
        """
        axial = self.forebody_axial if forebody_only else self.coefficients.axial
        normal = self.forebody_normal if forebody_only else self.coefficients.normal
        drag = axial * np.cos(self.alpha) + normal * np.sin(self.alpha)
        lift = normal * np.cos(self.alpha) - axial * np.sin(self.alpha)
        return float(lift), float(drag)

    @property
    def y_plus_reported(self) -> float:
        """Maximum wall :math:`y^+` if the run recorded one, else NaN.

        Worth reading on the first point of a sweep. The mesh's first cell was
        sized from a flat-plate estimate, and this is the only number that says
        whether the estimate held on the actual body.
        """
        values = self.history.get("max[y+]")
        if values is None or values.size == 0:
            return float("nan")
        return float(values[-1])


def find_mpirun(su2_binary: Path) -> str:
    """Locate the ``mpirun`` that belongs to this SU2 build.

    Looked for **beside the SU2 binary first** and on ``PATH`` second, which is
    the opposite of the usual order and is deliberate. SU2 is not
    pip-installable and lives in its own conda environment, which ships its own
    Open MPI; that environment is normally not activated, so ``mpirun`` is
    absent from ``PATH`` even though the one SU2 was linked against is sitting
    next to it. Worse than absent, if the host happens to have a *different*
    MPI on ``PATH``: launching an Open MPI binary under a foreign ``mpirun``
    gives every rank the same communicator size of one, so the run appears to
    work and silently solves the whole domain four times over.
    """
    beside = Path(su2_binary).parent / "mpirun"
    if beside.exists():
        return str(beside)
    found = shutil.which("mpirun")
    if found is not None:
        return found
    msg = (
        f"mpirun not found beside {su2_binary} or on PATH, but processes > 1 was "
        f"requested. SU2's conda environment ships its own Open MPI; either run "
        f"with processes=1 or point at an environment that has one."
    )
    raise FileNotFoundError(msg)


def mesh_key(mach: float, *, remesh_per_mach: bool = True) -> str:
    """Identifier of the mesh a Mach number will be solved on.

    Public because it is the only safe way to partition a sweep for
    concurrency: two points share a mesh exactly when they share this key, so
    grouping by it is what keeps two workers from building the same file at the
    same time. Deriving that grouping any other way — by Mach band, say — is a
    guess that happens to be right until the buckets move.
    """
    return f"{SU2Solver._bucket(mach) if remesh_per_mach else 0.0:.2f}"


def read_history(path: Path) -> dict[str, _FloatArray]:
    """Every numeric column of an SU2 ``history.csv``, keyed by lower-cased name.

    Read wholesale rather than column-by-column because what SU2 writes depends
    on the solver: a NEMO run has no ``CL`` until it is asked for one, an
    unsteady run indexes on ``Time_Iter`` rather than ``Inner_Iter``, and the
    turbulence residuals appear and disappear with the model. Taking the whole
    file and looking up what is wanted keeps that variation in one place.
    """
    if not path.exists():
        return {}
    with path.open() as handle:
        reader = csv.reader(handle)
        try:
            header = [cell.strip().strip('"').lower() for cell in next(reader)]
        except StopIteration:
            return {}
        rows = [row for row in reader if row and len(row) == len(header)]
    if not rows:
        return {}
    columns: dict[str, _FloatArray] = {}
    for index, name in enumerate(header):
        try:
            columns[name] = np.array([float(row[index]) for row in rows])
        except ValueError:
            continue  # a non-numeric column, of which SU2 writes none today
    return columns


def _column(history: dict[str, _FloatArray], *names: str) -> _FloatArray | None:
    for name in names:
        found = history.get(name)
        if found is not None and found.size:
            return found
    return None


def _force_settled(
    drag: _FloatArray | None, window: int, tolerance: float
) -> tuple[bool, float, float]:
    """Has the reported coefficient stopped moving over the last ``window``?

    The reported coefficient is the **window mean**, so that is what has to be
    shown stable — not the instantaneous value, which on a shock-dominated
    hypersonic case oscillates by several percent indefinitely. The sphere-cone
    this was built against settles to a forebody mean of 0.0717 that moves 0.2 %
    between a 4000- and a 12000-iteration window, while the instantaneous signal
    ripples 5.8 % peak-to-peak with no sign of decaying.

    Convergence is therefore the disagreement between the means of the two
    halves of the window, and the guard that makes that trustworthy is a count
    of oscillation periods rather than a threshold on ripple amplitude.

    Why the period count, and not the amplitude
    -------------------------------------------

    Every statistic here aliases against the ripple when the window is short.
    Measured on one real history: the half-mean difference reads 1.6e-4 over a
    1000-iteration window and 7.1e-3 over a 200-iteration one, and a fitted
    slope reads 1.4e-2 and 7.3e-3 on the same data — neither is a property of
    the solution, both are properties of where the window landed. Over many
    periods both converge on the truth. Requiring ``minimum_periods`` does what
    a ripple threshold cannot: it declares the *measurement* unresolvable rather
    than declaring the *case* unconverged, which is what was making settled
    cases look like failures.

    Returns
    -------
    tuple
        ``(settled, drift, ripple)`` — drift the fractional disagreement between
        half-window means, ripple the fractional peak-to-peak spread.
    """
    minimum_periods = 8
    if drag is None or drag.size < window:
        return False, float("nan"), float("nan")
    tail = drag[-window:]
    level = float(np.mean(np.abs(tail)))
    if level <= 0.0 or not np.all(np.isfinite(tail)):
        return False, float("nan"), float("nan")

    ripple = float((np.max(tail) - np.min(tail)) / level)

    # Oscillations counted on the signal with its linear trend removed, so a
    # steadily drifting case is not mistaken for a slowly oscillating one.
    index = np.arange(tail.size, dtype=np.float64)
    slope, intercept = np.polyfit(index, tail, 1)
    detrended = tail - (slope * index + intercept)
    periods = int(np.count_nonzero(np.diff(np.signbit(detrended)))) / 2.0

    half = window // 2
    drift = float(abs(np.mean(tail[half:]) - np.mean(tail[:half])) / level)

    resolvable = periods >= minimum_periods or ripple < tolerance
    return bool(resolvable and drift < tolerance), drift, ripple


def residual_trend(
    columns: dict[str, _FloatArray], window: int
) -> tuple[float, float]:
    r"""How the RMS residual is moving over the last ``window`` iterations.

    A force plateau is not convergence. SU2 reports its residuals as
    :math:`\log_{10}` of the RMS, so a converging steady solve drives them
    *down*; if they are going up, the field is getting worse no matter how still
    the integrated force looks.

    That combination is not hypothetical, it is what this queue was producing.
    A Mach 8 sphere-cone ran to 3453 iterations with :math:`C_D` creeping from
    0.14420 to 0.14459 -- smooth, monotone, no ripple at all, which is exactly
    what a drift-and-ripple test reads as *settled*. Over the same span
    ``rms[Rho]`` climbed from -4.88 to -4.39, and the volume snapshot at the
    stopping point held 670 nodes above 6000 K with a peak of 590,410 K: a few
    hundred cells in one azimuthal sector of the nose cap were running away
    while the other 221,000 sat at a correct 3400 K. The integrated force cannot
    feel a few hundred cells. The residual can, and nothing was reading it.

    The trend is fitted on the log residual, so the slope is in **decades per
    thousand iterations** and is comparable across cases and mesh sizes. The
    worst (most positive) slope over all ``rms[...]`` columns is returned,
    because a solution may be losing only one equation.

    Returns
    -------
    tuple
        ``(slope, rise)`` -- slope in decades per 1000 iterations over the
        window, and the rise in decades from the window's minimum to its end.
        ``(nan, nan)`` when there is no residual column or too little history.
    """
    keys = [k for k in columns if k.startswith("rms[")]
    if not keys or window < 2:
        return float("nan"), float("nan")
    worst_slope = -float("inf")
    worst_rise = -float("inf")
    for key in keys:
        series = np.asarray(columns[key], dtype=np.float64)
        if series.size < window:
            continue
        tail = series[-window:]
        if not np.all(np.isfinite(tail)):
            continue
        index = np.arange(tail.size, dtype=np.float64)
        slope = float(np.polyfit(index, tail, 1)[0]) * 1000.0
        rise = float(tail[-1] - np.min(tail))
        worst_slope = max(worst_slope, slope)
        worst_rise = max(worst_rise, rise)
    if worst_slope == -float("inf"):
        return float("nan"), float("nan")
    return worst_slope, worst_rise


def _mean_uncertainty(
    values: _FloatArray | None, window: int
) -> tuple[float, float]:
    """Standard error of the window mean of a limit-cycling force signal.

    A shock-dominated hypersonic Euler solution with a base region does not
    reach a steady state; it settles into a limit cycle about one. The window
    mean is then a *statistical* estimate of the underlying value, and quoting
    it bare implies a precision the run does not have — one measured
    sphere-cone forebody history rippled 4.3 % about a mean that was itself
    good to 0.07 % against exact Taylor-Maccoll.

    The subtlety is the sample count. Successive iterates are strongly
    correlated, so ``std/sqrt(window)`` understates the error by more than an
    order of magnitude. The independent-sample count is the number of
    oscillation periods in the window, which is what an autocorrelation time
    recovers on a signal this close to sinusoidal, and it is already counted
    for the resolvability guard in :func:`_force_settled`.

    Returns
    -------
    tuple
        ``(sigma, periods)`` — the absolute standard error of the mean, in
        coefficient units, and the independent-sample count it was divided by.
    """
    if values is None or values.size == 0:
        return float("nan"), float("nan")
    tail = values[-min(window, values.size) :]
    if tail.size < 4 or not np.all(np.isfinite(tail)):
        return float("nan"), float("nan")
    index = np.arange(tail.size, dtype=np.float64)
    slope, intercept = np.polyfit(index, tail, 1)
    detrended = tail - (slope * index + intercept)
    periods = float(np.count_nonzero(np.diff(np.signbit(detrended)))) / 2.0
    if periods < 1.0:
        # Fewer than one period resolved: the spread itself is the honest bar.
        return float(np.std(tail)), periods
    return float(np.std(detrended) / np.sqrt(periods)), periods


@dataclass
class SU2Solver:
    """Three-dimensional SU2, presented as a :class:`Solver`.

    Attributes
    ----------
    body:
        A closed :class:`~aether.geometry.VehicleMesh`. Its triangulation is
        the wall mesh; see
        :func:`~aether.aerodynamics.cfd.meshing.viscous_domain`.
    workspace:
        Where meshes and cases are written, and **kept**. A CFD point that
        cannot be re-examined afterwards is not a measurement, and on a sweep
        this long the directory is the only record of what was actually solved.
    reference:
        Shared with the panel tables, so the two can be spliced without
        renormalising.
    altitude:
        Flight altitude (m), which fixes the freestream and therefore the
        Reynolds number at every point of the sweep. Required rather than
        defaulted: there is no altitude at which a default is right, and the
        symptom of a wrong one is a plausible table.
    regime:
        Pin the gas model, or ``None`` to let
        :func:`~aether.cfd.config.regime_for` choose per Mach number —
        low-Mach preconditioning below 0.3, perfect gas to 10, NEMO above.
    remesh_per_mach:
        Build a mesh per Mach bucket rather than reusing one. Both the shock
        envelope and the wall spacing are functions of Mach — the first cell
        height moves by a factor of thirty across this envelope — so one mesh
        is either wrong at the top or wasteful at the bottom.
    """

    body: object
    workspace: Path
    reference: ReferenceFrame
    altitude: float
    regime: Regime | None = None
    numerics: Numerics = field(default_factory=Numerics)
    sizing: ViscousSizing = field(default_factory=ViscousSizing)
    wall_temperature: float = 300.0
    remesh_per_mach: bool = True
    processes: int = 1
    timeout: float = 21600.0
    keep_cases: bool = True
    wall_refinement: float = 0.05
    """Near-wall cell size for an inviscid mesh, as a fraction of body diameter.

    Only used on the Euler path, where resolution near the body comes from an
    isotropic refinement box rather than a prism layer. This is the knob a grid
    study turns.
    """
    wall_band: float = 0.5
    """How far the near-wall refinement extends, in body diameters."""
    shock_layer_radius: float | None = None
    """Nose radius (m) whose bow shock the Euler mesh must resolve.

    When set, the inviscid path is meshed with a *prism stack* sized to the
    shock standoff instead of isotropic tetrahedra. This is not an optimisation
    detail; it is the difference between a resolved captured shock and an
    unresolved one. Resolving a Mach 8 sphere-cone's 7.5 mm standoff with
    isotropic cells extrapolates to about 4M elements and roughly 66 hours;
    thirteen layers through the same standoff cost 134k elements and build in
    a second. See
    :func:`~aether.aerodynamics.cfd.meshing.shock_layer_sizing`.
    """
    shock_layer_cells: int = 16
    """Layers in the stack; about thirteen of sixteen land inside the standoff."""
    shock_layer_span: float = 1.5
    """Stack height as a multiple of the standoff, so the shock sits inside it."""
    refinement: tuple[RefinementBall, ...] = ()
    """Local feature regions imposed on the volume mesh.

    Build a nose region with
    :func:`~aether.aerodynamics.cfd.meshing.stagnation_refinement`. Without one
    the cell size at a blunt nose comes from the body diameter, which on a
    Mach 8 sphere-cone gave 1.5 cells across a 7.5 mm shock standoff and put
    8 % of the wall above the Rayleigh pitot limit — a pressure no flow at that
    Mach number can produce. Check the result with
    :func:`~aether.aerodynamics.cfd.diagnostics.pitot_limit_violation`.
    """
    wall_marker: str = "vehicle"
    """Wall marker name; the base patch is ``<wall_marker>_base``."""
    iteration_scaling: bool = True
    """Scale the iteration cap with mesh size, instead of holding it fixed.

    A fixed cap is the reason iterative error *grew* with refinement in the
    validation study — drift 4.6e-3, 7.6e-3, 1.1e-2 across three levels — until
    it exceeded the discretisation error it was supposed to be smaller than.
    Information crosses the domain at a finite rate per iteration, so the count
    needed to reach a steady state scales with the linear cell count,
    :math:`N^{1/3}`, not with :math:`N^0`.

    Without this, refining the mesh makes the answer *less* trustworthy while
    appearing to make it more so, and no grid-convergence band computed across
    such a sequence is meaningful.
    """
    reference_cells: int = 50_000
    """Cell count at which ``Numerics.iterations`` is taken at face value."""
    alpha_continuation: bool = True
    """Start an incidence case from the converged zero-incidence solution.

    On by default, because starting from freestream at incidence is what fails.
    A hypersonic case has to form its bow shock and sweep it back across the
    mesh, and doing that asymmetrically is markedly less forgiving than doing
    it symmetrically: the sphere-cone converges cleanly at zero incidence on
    the same mesh, settings and CFL that diverge at six degrees.

    Continuation removes the problem rather than papering over it with a lower
    CFL, which costs iterations at every angle instead of only the hard ones.
    The zero-incidence solution is cached, so a sweep pays for it once.
    """
    split_base: bool = False
    """Separate the base disc from the forebody, by outward normal.

    Set it on any body with a blunt base. Without it the Euler base pressure —
    not a physical quantity — is folded into a single force coefficient, and on
    a Mach 8 sphere-cone that inverted the sign of the reported drag.
    """
    base_station: float | None = None
    """Axial station separating forebody from base disc (m).

    Set it on any body with a blunt base. With it, the wall pressure is
    integrated here and reported as
    :attr:`~aether.cfd.solver.SU2Run.forebody_axial` and
    :attr:`~aether.cfd.solver.SU2Run.base_axial` — because an Euler base
    pressure is not a physical quantity and can dominate the total. Without it,
    SU2's own single force coefficient is used and the base is folded in
    silently.
    """
    strict: bool = True
    """Raise when a case fails to converge. Off records NaN and carries on."""
    name: str = "su2-3d"
    executable: Path | None = None
    _meshes: dict[str, MeshResult] = field(default_factory=dict, repr=False)
    _runs: dict[tuple[float, float], SU2Run] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

    # -- meshing ---------------------------------------------------------

    @staticmethod
    def _bucket(mach: float) -> float:
        """Mach values that share a mesh.

        Coarser at the top than the axisymmetric path's buckets, because up
        there the wall spacing rather than the shock envelope is what changes,
        and it changes smoothly.
        """
        m = float(mach)
        for edge, label in ((0.3, 0.2), (1.2, 1.0), (2.0, 1.5), (3.5, 2.5),
                            (6.0, 4.5), (12.0, 9.0), (20.0, 16.0)):
            if m < edge:
                return label
        return 24.0

    def mesh_key(self, mach: float) -> str:
        """Identifier of the mesh this Mach number will be solved on."""
        return mesh_key(mach, remesh_per_mach=self.remesh_per_mach)

    def mesh_for(self, mach: float) -> MeshResult:
        """Build (or reuse) a viscous mesh sized for this Mach number."""
        key = self.mesh_key(mach)
        cached = self._meshes.get(key)
        if cached is not None:
            return cached
        flow = FlowConditions.at_altitude(
            mach=self._bucket(mach) if self.remesh_per_mach else mach,
            altitude=self.altitude,
            wall_temperature=self.wall_temperature,
        )
        target = self.workspace / "mesh" / f"{self.name}-m{key}.su2"
        # An Euler case gets a pure tetrahedral domain, not a prism layer.
        # Extruding a viscous boundary layer for a solver with no viscous term
        # is not merely wasteful: on a body whose nose radius is smaller than
        # the layer is thick the extrusion self-intersects, is truncated, and
        # leaves badly skewed cells exactly at the stagnation region. Every
        # Mach 8 case run against such a mesh diverged.
        if (
            bool(getattr(self.resolved_regime(flow.mach), "inviscid", False))
            and self.shock_layer_radius is not None
        ):
            layered, first_cell = shock_layer_sizing(
                flow.mach, float(self.shock_layer_radius), self.sizing,
                cells=int(self.shock_layer_cells), span=float(self.shock_layer_span),
            )
            built = viscous_domain(
                self.body, target, mach=flow.mach, sizing=layered,
                temperature=flow.temperature, pressure=flow.pressure,
                first_cell_height=first_cell, wall_marker=self.wall_marker,
                split_base=self.split_base, base_station=self.base_station,
            )
        elif bool(getattr(self.resolved_regime(flow.mach), "inviscid", False)):
            built = inviscid_domain(
                self.body, target, mach=flow.mach, sizing=self.sizing,
                wall_refinement=self.wall_refinement, wall_band=self.wall_band,
                wall_marker=self.wall_marker, split_base=self.split_base,
                base_station=self.base_station, refinement=self.refinement,
            )
        else:
            built = viscous_domain(
                self.body,
                target,
                mach=flow.mach,
                sizing=self.sizing,
                temperature=flow.temperature,
                pressure=flow.pressure,
                wall_temperature=self.wall_temperature,
                wall_marker=self.wall_marker,
            )
        self._meshes[key] = built
        return built

    # -- running ---------------------------------------------------------

    def resolved_regime(self, mach: float) -> Regime:
        """The gas model this solver will use at ``mach``."""
        from aether.cfd.config import regime_for

        return self.regime if self.regime is not None else regime_for(mach)

    def case_for(
        self,
        mach: float,
        alpha: float,
        mesh: MeshResult,
        motion: PitchingMotion | None = None,
    ) -> SU2Case:
        """The case this solver would run at one condition."""
        return SU2Case(
            flow=FlowConditions.at_altitude(
                mach=mach,
                altitude=self.altitude,
                alpha=alpha,
                wall_temperature=self.wall_temperature,
            ),
            reference=self.reference,
            mesh_filename=mesh.path.name,
            regime=self.regime,
            numerics=self.numerics,
            motion=motion,
            wall_marker=self.wall_marker,
            # Set together with the mesh split, so a caller cannot ask for the
            # base to be separated in the geometry and then forget to monitor it.
            base_marker=(
                f"{self.wall_marker}_base"
                if (self.split_base or self.base_station is not None)
                else None
            ),
        )

    def _continuation_source(self, mach: float, alpha: float) -> SU2Run | None:
        """The solved case an incidence run should start from, if any.

        The nearest *smaller* incidence of the same sign, rather than always
        zero. Both are valid starts, but the increment is what costs: a sweep
        walking 0, 2, 5, 10, 15, 20 degrees asks the solver to turn the flow by
        five degrees at a time instead of by twenty, and the transient it has
        to chase is correspondingly shorter. Opposite signs are never mixed —
        seeding +15 from -15 would start the solver from a field whose
        asymmetry points the wrong way, which is worse than freestream.

        Returns ``None`` when nothing suitable has been solved yet, leaving the
        caller to fall back to the zero-incidence case.
        """
        # Rounded to the cache's own precision before comparing. The keys of
        # `_runs` are rounded, so comparing a raw alpha against them makes a
        # point fractionally "larger" than its own stored key and it selects
        # itself as its own seed -- a case that reads as working, because
        # restarting from the converged answer converges immediately.
        target = round(float(alpha), 9)
        key_mach = round(float(mach), 6)
        best: SU2Run | None = None
        best_alpha = 0.0
        for (cached_mach, cached_alpha), run in self._runs.items():
            if cached_mach != key_mach or run.diverged:
                continue
            # Strictly closer to zero, and never on the far side of it. Zero
            # itself is a valid seed for either sign, which is the base of the
            # chain rather than a special case.
            if cached_alpha * target < 0.0 or abs(cached_alpha) >= abs(target):
                continue
            if best is None or abs(cached_alpha) > abs(best_alpha):
                best, best_alpha = run, cached_alpha
        return best

    def run(
        self, mach: float, alpha: float, motion: PitchingMotion | None = None
    ) -> SU2Run:
        """Solve one point and return everything the run produced."""
        key = (round(float(mach), 6), round(float(alpha), 9))
        if motion is None and key in self._runs:
            return self._runs[key]

        mesh = self.mesh_for(mach)
        case = self.case_for(mach, alpha, mesh, motion=motion)
        label = f"m{mach:.4f}-a{np.rad2deg(alpha):+.2f}" + ("-pitch" if motion else "")
        work = self.workspace / "cases" / label
        work.mkdir(parents=True, exist_ok=True)

        local_mesh = work / mesh.path.name
        if local_mesh.resolve() != mesh.path.resolve():
            shutil.copy(mesh.path, local_mesh)

        binary = Path(self.executable) if self.executable else find_su2()
        prelude = int(self.numerics.first_order_iterations)
        restart = False

        seed = None
        if (
            self.alpha_continuation
            and motion is None
            and abs(float(alpha)) > 1.0e-9
        ):
            # Start from the nearest incidence already solved at this Mach
            # number, falling back to the symmetric case. Recursion is bounded:
            # the fallback has alpha == 0 and takes neither branch.
            source = self._continuation_source(mach, alpha)
            if source is None:
                source = self.run(mach, 0.0)
            if not source.diverged:
                seed = next(
                    (p for p in (source.directory / "restart_flow.dat",
                                 source.directory / "restart_flow.csv",
                                 source.directory / "solution_flow.dat")
                     if p.exists()),
                    None,
                )
        if seed is not None:
            shutil.copy(seed, work / f"solution_flow{seed.suffix}")
            restart = True
            prelude = 0  # the seed is already a converged second-order field

        if prelude > 0 and motion is None:
            # Stage one: first order, from freestream, just long enough to put
            # the bow shock where it belongs.
            starter = replace(
                case,
                numerics=replace(
                    self.numerics, muscl=False, iterations=prelude,
                    minimum_iterations=max(prelude + 1, 10),
                ),
            )
            (work / "prelude.cfg").write_text(starter.render())
            first = self._launch(binary, work, "prelude.cfg")
            (work / "su2-prelude.log").write_text(first)
            produced = next(
                (p for p in (work / "restart_flow.dat", work / "restart_flow.csv")
                 if p.exists()),
                None,
            )
            if produced is not None:
                # SU2 reads its initial condition from SOLUTION_FILENAME and
                # writes to RESTART_FILENAME; restarting in place needs the
                # first moved onto the second, or the run silently starts from
                # freestream again and dies exactly as it would have.
                shutil.move(str(produced), str(work / f"solution_flow{produced.suffix}"))
                restart = True

        if self.iteration_scaling and mesh.n_elements > 0:
            factor = (mesh.n_elements / float(self.reference_cells)) ** (1.0 / 3.0)
            scaled = replace(
                case.numerics,
                iterations=int(case.numerics.iterations * max(factor, 1.0)),
            )
            case = replace(case, numerics=scaled)

        config = work / "case.cfg"
        config.write_text(replace(case, restart=restart).render())
        text = self._launch(binary, work, "case.cfg")
        (work / "su2.log").write_text(text)

        history = read_history(work / "history.csv")
        result = self._read_run(mach, alpha, history, work, mesh, 0)
        if (self.split_base or self.base_station is not None) and not result.diverged:
            result = self._with_breakdown(result, work, mach, alpha)
        if not self.keep_cases:
            for name in ("restart_flow.dat", "restart_flow.csv", "volume_flow.vtu"):
                (work / name).unlink(missing_ok=True)
        if motion is None:
            self._runs[key] = result
        return result

    @staticmethod
    def _detect_divergence(work: Path, returncode: int) -> bool:
        """Did SU2 abort on a NaN rather than run out of iterations?

        Read from the log, and **not gated on the exit status**. Both halves of
        that matter. The log is consulted rather than the status because a
        non-zero status also covers a missing mesh, a bad marker name and an
        MPI launch failure, and calling those divergence would send a caller to
        lower the CFL over a typo. The status is not consulted at all because
        it cannot be trusted: this Open MPI build returns **zero** after
        ``MPI_ABORT``, so a run that died on a NaN at iteration 205 comes back
        looking like a clean exit, and its last finite iterate — a negative
        lift coefficient at positive incidence — goes into the table.

        ``returncode`` is kept in the signature because a caller reading a
        failure wants both facts, not because this needs it.
        """
        # *Both* logs. The first-order prelude is a separate SU2 invocation
        # writing a separate log, and reading only the main one lets a
        # diverged prelude through silently: the main run then restarts from
        # a corrupted field and finishes without ever printing "diverged"
        # itself. Measured on a Mach 8 near-sharp cone, whose prelude died
        # with "Residual > 10^20 detected" and MPI_ABORT — the run was
        # reported merely "not settled" and returned a forebody C_A of
        # +59169 against an exact 0.0685, with every wall node above the
        # Rayleigh pitot limit. A number that wrong is obvious; the danger is
        # the same failure arriving one order of magnitude out instead of
        # nine, where nothing but the diagnostic would catch it.
        #
        # Residual blow-up is matched as well as NaN: SU2 reports
        # "Residual > 10^20 detected" for the former, which is divergence by
        # any reading and shares none of the wording of the latter.
        markers = ("diverged", "nan detected", "residual > 10^20")
        for name in ("su2.log", "su2-prelude.log"):
            log = work / name
            if not log.exists():
                continue
            text = log.read_text(errors="ignore").lower()
            if any(marker in text for marker in markers):
                return True
        return False

    def _with_breakdown(
        self, result: SU2Run, work: Path, mach: float, alpha: float
    ) -> SU2Run:
        """Re-integrate the wall pressure and split it at the base station."""
        if not (self.split_base or self.base_station is not None):
            return result
        from dataclasses import replace as _replace

        from aether.aerodynamics.cfd.su2 import surface_force_breakdown
        from aether.aerodynamics.tables import Coefficients

        surface = next(
            (p for p in (work / "surface_flow.csv", work / "surface_flow.dat")
             if p.exists()),
            None,
        )
        if surface is None:
            return result
        flow = FlowConditions.at_altitude(
            mach=mach, altitude=self.altitude, alpha=alpha,
            wall_temperature=self.wall_temperature,
        )
        forces = surface_force_breakdown(
            surface, self.body, flow.pressure, flow.dynamic_pressure,
            self.reference.area, self.reference.length,
            base_station=self.base_station,
            moment_reference=self.reference.moment_reference,
        )
        # Where the mesh carries a real base *marker*, the solver's own
        # per-marker history is authoritative and this re-integration supplies
        # only the base area.
        #
        # Two reasons, and both bit. It is time-averaged: the history value is
        # the mean over the convergence window, while the surface file is one
        # iterate of a signal that limit-cycles by several percent, so
        # overriding replaced a mean carrying +/-0.65 % with a sample carrying
        # the full ripple. And it is consistent: the marker split is the one
        # the mesh was actually written with, whereas this function re-derives
        # a split from a cone angle about the axis and disagrees with it on
        # filleted shoulders. On the Mach 8 sphere-cone with a 10 mm shoulder
        # the two differed by 7 % of forebody C_A (0.0696 against 0.0745 on the
        # same iterate), the re-derived split having charged part of the fillet
        # to the base. The near-sharp validation case hid this because at
        # Rn/Rb = 0.0056 with no shoulder there is no fillet to misassign.
        markers_reported = bool(np.isfinite(result.forebody_axial))
        if markers_reported:
            return _replace(
                result, base_area=forces.base_area, reference_area=self.reference.area
            )
        return _replace(
            result,
            forebody_axial=forces.forebody_axial,
            base_axial=forces.base_axial,
            forebody_normal=forces.forebody_normal,
            base_normal=forces.base_normal,
            base_area=forces.base_area,
            reference_area=self.reference.area,
            coefficients=Coefficients(
                axial=forces.axial, normal=forces.normal,
                pitching_moment=forces.pitching_moment,
            ),
        )

    def _launch(self, binary: Path, work: Path, config: str) -> str:
        """Run SU2 once and return its combined output."""
        command = [str(binary), config]
        if int(self.processes) > 1:
            command = [find_mpirun(binary), "-n", str(int(self.processes)), *command]
        finished = subprocess.run(
            command, cwd=work, capture_output=True, text=True,
            timeout=self.timeout, check=False,
        )
        return finished.stdout + "\n----- stderr -----\n" + finished.stderr

    def _read_run(
        self,
        mach: float,
        alpha: float,
        history: dict[str, _FloatArray],
        work: Path,
        mesh: MeshResult,
        returncode: int,
    ) -> SU2Run:
        # Per-marker columns are named for the marker, not indexed: SU2 writes
        # `CFx(vehicle)`, not `CFx[0]`.
        forebody = _column(history, f"cfx({self.wall_marker})")
        base = _column(history, f"cfx({self.wall_marker}_base)")
        forebody_z = _column(history, f"cfz({self.wall_marker})")
        base_z = _column(history, f"cfz({self.wall_marker}_base)")
        axial = _column(history, "cfx", '"cfx"')
        if axial is None and forebody is not None and base is not None:
            axial = forebody + base
        normal = _column(history, "cfz", '"cfz"')
        moment = _column(history, "cmy", '"cmy"')
        drag = _column(history, "cd", "drag")
        residual = _column(history, "rms[rho]", "rms_density", "rms[rho_0]")

        window = int(self.numerics.force_window)
        # Convergence is judged on the **forebody** where one is separated, not
        # on the total. An Euler base region is an unresolved recirculation with
        # no steady state, so the base force never settles — its value moved
        # 27 % between grid levels while the forebody moved 2 %, and the drift
        # measured on the total got *worse* with refinement because finer
        # meshes resolve more of that unsteadiness. Judging the total therefore
        # reports the forebody as unconverged on the strength of a quantity
        # nobody should be reading, and no iteration count fixes it.
        settled, drift, ripple = _force_settled(
            forebody if forebody is not None else (drag if drag is not None else axial),
            window,
            self.numerics.force_tolerance,
        )
        have_forces = all(c is not None and c.size for c in (axial, normal, moment))
        diverged = self._detect_divergence(work, returncode)
        converged = bool(returncode == 0 and have_forces and settled and not diverged)

        # Reported as the mean over the convergence window rather than the last
        # iterate. The solution ripples about its answer, so the final value is
        # one sample of that ripple and carries its full amplitude; the window
        # mean carries it divided by the square root of the window, and is the
        # number the convergence test was applied to in the first place.
        def _mean(values: _FloatArray | None) -> float:
            if values is None or values.size == 0:
                return float("nan")
            return float(np.mean(values[-min(window, values.size):]))

        coefficients = (
            Coefficients(
                axial=_mean(axial),
                normal=_mean(normal),
                pitching_moment=_mean(moment),
            )
            if have_forces
            else Coefficients(float("nan"), float("nan"), float("nan"))
        )
        forebody_sigma, periods_resolved = _mean_uncertainty(forebody, window)
        base_sigma, _ = _mean_uncertainty(base, window)
        return SU2Run(
            mach=float(mach),
            alpha=float(alpha),
            coefficients=coefficients,
            converged=converged,
            residual=float(residual[-1]) if residual is not None else float("nan"),
            forebody_axial=_mean(forebody),
            base_axial=_mean(base),
            forebody_normal=_mean(forebody_z),
            base_normal=_mean(base_z),
            forebody_sigma=forebody_sigma,
            base_sigma=base_sigma,
            settled_periods=periods_resolved,
            force_drift=drift,
            force_ripple=ripple,
            diverged=diverged,
            iterations=int(next(iter(history.values())).size) if history else 0,
            directory=work,
            mesh=mesh,
            history=history,
        )

    # -- the Solver protocol ---------------------------------------------

    def solve(self, mach: float, alpha: float) -> Coefficients:
        """Body-axis coefficients at one condition.

        With ``strict`` set — the default — a non-converged case raises rather
        than entering the table. A table is read by an integrator that has no
        way to tell a converged coefficient from a stalled one, so a bad point
        that reaches the table is worse than a sweep that stops: the first
        corrupts a trajectory quietly, the second is visible immediately and
        resumes from its checkpoint once fixed.
        """
        result = self.run(mach, alpha)
        if result.diverged:
            # Raised even when `strict` is off. Not settling is a partial
            # answer worth recording as NaN and carrying on from; a NaN in the
            # solver is not an answer at all, and a sweep that records one is
            # building a table whose bad entries are indistinguishable from
            # points that were merely slow.
            msg = (
                f"SU2 diverged at Mach {mach:g}, alpha {np.rad2deg(alpha):.2f} deg "
                f"after {result.iterations} iterations — a NaN, not a slow "
                f"convergence. Lower Numerics.cfl (hypersonic starts from "
                f"freestream need a shock-formation transient at low CFL) or "
                f"tighten the CFL bounds. Case kept at {result.directory}"
            )
            raise RuntimeError(msg)
        if not result.converged:
            if self.strict:
                msg = (
                    f"SU2 did not converge at Mach {mach:g}, alpha "
                    f"{np.rad2deg(alpha):.2f} deg after {result.iterations} "
                    f"iterations: force mean still drifting by "
                    f"{result.force_drift:.2e} across the window (tolerance "
                    f"{self.numerics.force_tolerance:.1e}), ripple "
                    f"{result.force_ripple:.2e}, residual {result.residual:.3f}. "
                    f"Case kept at {result.directory}"
                )
                raise RuntimeError(msg)
            return Coefficients(float("nan"), float("nan"), float("nan"))
        return result.coefficients
