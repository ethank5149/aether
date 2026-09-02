"""Driving SU2, and extracting forces in a way that cannot be misread.

SU2 reports its own force coefficients, and for an axisymmetric case they
depend on a reference-area convention that has changed between releases and
is not the one used anywhere else in this package. Rather than calibrate
against it, this module reads the **surface pressure distribution** and
integrates it here.

For a body of revolution that integration has a closed form worth writing
down, because it removes surface normals, arc lengths and node ordering from
the problem entirely. With the profile :math:`r(x)`, outward normal
:math:`\\hat n = (-r', 1)/\\sqrt{1+r'^2}` and element area
:math:`\\mathrm{d}S = 2\\pi r\\sqrt{1+r'^2}\\,\\mathrm{d}x`,

.. math::

    \\hat n_x\\,\\mathrm{d}S = -2\\pi r r'\\,\\mathrm{d}x
    = -\\pi\\,\\mathrm{d}(r^2)

so the axial force on the lateral surface is

.. math::

    F_x = -\\oint (p - p_\\infty)\\,\\hat n_x\\,\\mathrm{d}S
        = \\pi\\int_{\\text{nose}}^{\\text{base}} (p - p_\\infty)\\,\\mathrm{d}(r^2)

and the base disc contributes :math:`-\\pi r_b^2 (\\bar p_b - p_\\infty)`.
The whole force is a trapezoid rule on :math:`p` against :math:`r^2`, needing
only that the wall points be sortable by :math:`x` — which they are, because
the profile is single-valued by construction. Nothing depends on how SU2
orders its output or on which release wrote it.

What an Euler solution does not contain
---------------------------------------

Two things, and both are supplied elsewhere rather than pretended away:

* **Skin friction.** There is no boundary layer in an Euler solution. On
  this vehicle friction is 5 to 7 % of axial force at supersonic speeds — see
  :mod:`aether.aerodynamics.friction`.
* **Base drag.** The Euler base pressure is not the real base pressure,
  because the real one is set by a separated shear layer that the equations
  do not describe. The base contribution is therefore reported
  *separately* by :attr:`SU2Result.base_axial` rather than folded into the
  total, so that a caller can replace it and can see how much is at stake.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aether.aerodynamics.cfd.meshing import BodyProfile, MeshResult

__all__ = [
    "SU2Result",
    "SU2Settings",
    "SurfaceForces",
    "find_su2",
    "run_su2",
    "surface_axial_force",
    "surface_force_breakdown",
]

_FloatArray = NDArray[np.float64]

#: Last-resort locations, tried only after ``SU2_RUN`` and ``PATH``.
#:
#: Absolute paths in library code are not portable and this list is not how the
#: solver is meant to be found -- ``SU2_RUN`` is SU2's own convention and is
#: what the container sets. What remains here is a courtesy for an interactive
#: checkout on the machine this project grew up on, where neither is set.
_DEFAULT_SU2_PREFIXES = ("/config/miniconda3/envs/su2/bin",)


def find_su2(executable: str = "SU2_CFD") -> Path:
    """Locate the SU2 binary, or say clearly that it is not there.

    In order: the ``SU2_RUN`` environment variable, then ``PATH``, then a
    last-resort prefix list.

    ``SU2_RUN`` is SU2's own convention for the directory its binaries live in,
    and is what makes this portable: an image, a module system or a local build
    each set it to their own location and nothing here has to know where that
    is. It is consulted *before* ``PATH`` so that an environment which has
    deliberately named a solver gets that one, rather than whichever build
    happens to be earlier in the search path.

    SU2 is a compiled dependency that cannot be pip-installed into the working
    environment, so it lives in its own; searching for it is better than
    failing with ``FileNotFoundError`` from somewhere deep in ``subprocess``.
    """
    run_dir = os.environ.get("SU2_RUN")
    if run_dir:
        candidate = Path(run_dir) / executable
        if candidate.exists():
            return candidate
        # Named but wrong: better to say so than to fall through to a different
        # solver, which would silently run a build the caller did not choose.
        msg = (
            f"SU2_RUN is set to {run_dir!r} but {executable} is not there. "
            f"Point it at the directory holding the SU2 binaries, or unset it "
            f"to fall back to PATH."
        )
        raise FileNotFoundError(msg)
    found = shutil.which(executable)
    if found is not None:
        return Path(found)
    for prefix in _DEFAULT_SU2_PREFIXES:
        candidate = Path(prefix) / executable
        if candidate.exists():
            return candidate
    msg = (
        f"{executable} not found via SU2_RUN, on PATH, or in "
        f"{_DEFAULT_SU2_PREFIXES}. SU2 is a compiled solver; install it into "
        f"its own environment (conda create -n su2 -c conda-forge su2) and set "
        f"SU2_RUN to its bin directory, add it to PATH, or pass the path "
        f"explicitly."
    )
    raise FileNotFoundError(msg)


@dataclass(frozen=True)
class SU2Settings:
    """Numerical settings for an axisymmetric Euler run.

    The defaults are chosen for **robustness across the whole envelope**
    rather than speed at any one point, because the thing this is used for is
    an unattended sweep from Mach 0.4 to Mach 6 and a scheme that needs
    hand-holding at Mach 5 is useless in that context.

    ``convective_scheme``
        AUSM rather than Roe. Roe's scheme admits the carbuncle instability
        at a strong bow shock — a spurious, grid-aligned protrusion at the
        stagnation line that does not converge away and that quietly
        corrupts the drag. AUSM does not, at the cost of slightly more
        dissipation on shear layers, of which an Euler solution over a smooth
        body has none.
    ``limiter``
        Venkatakrishnan, whose coefficient is a real parameter and not a
        formality: too large and it stops limiting near shocks, too small and
        it stalls convergence on a smooth field.
    ``cfl_adapt``
        On. A fixed CFL that survives Mach 0.6 will crawl at Mach 6 and one
        that is fast at Mach 6 will diverge at Mach 0.6.
    """

    convective_scheme: str = "AUSM"
    limiter: str = "VENKATAKRISHNAN"
    limiter_coefficient: float = 0.05
    cfl: float = 2.0
    cfl_adapt: bool = True
    cfl_bounds: tuple[float, float] = (0.5, 40.0)
    iterations: int = 8000
    linear_solver_iterations: int = 15
    multigrid_levels: int = 2
    force_tolerance: float = 2.0e-5
    """Relative spread of :math:`C_D` over ``force_window`` that counts as converged."""
    force_window: int = 150
    minimum_iterations: int = 300
    extra: dict[str, str] = field(default_factory=dict)
    """Raw ``KEY= VALUE`` overrides, appended last so they win."""


@dataclass(frozen=True)
class SU2Result:
    """A converged (or not) run and the coefficients read out of it."""

    mach: float
    forebody_axial: float
    """:math:`C_A` from the lateral surface only, on the reference area."""
    base_axial: float
    """:math:`C_A` contribution of the base disc. See the module note."""
    pressure_ratio: _FloatArray
    """:math:`p/p_\\infty` at each wall point, ordered by station."""
    station: _FloatArray
    radius: _FloatArray
    residual: float
    iterations: int
    converged: bool
    directory: Path
    mesh: MeshResult | None = None

    @property
    def axial(self) -> float:
        """Forebody plus base — the Euler answer, base pressure and all."""
        return float(self.forebody_axial + self.base_axial)

    @property
    def pressure_coefficient(self) -> _FloatArray:
        """:math:`C_p` along the body, from the pressure ratio and Mach."""
        gamma = 1.4
        return np.asarray(2.0 / (gamma * self.mach**2) * (self.pressure_ratio - 1.0))


def _config(
    mesh_path: Path,
    mach: float,
    settings: SU2Settings,
    reference_area: float,
    reference_length: float,
    temperature: float,
    pressure: float,
) -> str:
    low, high = settings.cfl_bounds
    lines = [
        "% Axisymmetric Euler, written by aether.aerodynamics.cfd",
        "SOLVER= EULER",
        "MATH_PROBLEM= DIRECT",
        "AXISYMMETRIC= YES",
        "RESTART_SOL= NO",
        "",
        f"MACH_NUMBER= {mach:.10g}",
        "AOA= 0.0",
        "SIDESLIP_ANGLE= 0.0",
        f"FREESTREAM_PRESSURE= {pressure:.10g}",
        f"FREESTREAM_TEMPERATURE= {temperature:.10g}",
        "FLUID_MODEL= STANDARD_AIR",
        "GAMMA_VALUE= 1.4",
        "GAS_CONSTANT= 287.058",
        "",
        f"REF_AREA= {reference_area:.10g}",
        f"REF_LENGTH= {reference_length:.10g}",
        "REF_ORIGIN_MOMENT_X= 0.0",
        "REF_ORIGIN_MOMENT_Y= 0.0",
        "REF_ORIGIN_MOMENT_Z= 0.0",
        "REF_DIMENSIONALIZATION= DIMENSIONAL",
        "",
        "MARKER_EULER= ( wall )",
        "MARKER_FAR= ( farfield )",
        "MARKER_SYM= ( axis )",
        "MARKER_PLOTTING= ( wall )",
        "MARKER_MONITORING= ( wall )",
        "",
        "NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES",
        f"CFL_NUMBER= {settings.cfl:.10g}",
        f"CFL_ADAPT= {'YES' if settings.cfl_adapt else 'NO'}",
        f"CFL_ADAPT_PARAM= ( 0.1, 2.0, {low:.10g}, {high:.10g} )",
        "LINEAR_SOLVER= FGMRES",
        "LINEAR_SOLVER_PREC= ILU",
        "LINEAR_SOLVER_ERROR= 1E-6",
        f"LINEAR_SOLVER_ITER= {settings.linear_solver_iterations}",
        f"MGLEVEL= {settings.multigrid_levels}",
        "MGCYCLE= V_CYCLE",
        "",
        f"CONV_NUM_METHOD_FLOW= {settings.convective_scheme}",
        "MUSCL_FLOW= YES",
        f"SLOPE_LIMITER_FLOW= {settings.limiter}",
        f"VENKAT_LIMITER_COEFF= {settings.limiter_coefficient:.10g}",
        "TIME_DISCRE_FLOW= EULER_IMPLICIT",
        "",
        f"ITER= {settings.iterations}",
        # Convergence on the *force*, not the residual. A limiter on an
        # unstructured mesh makes the density residual stall three decades
        # above machine zero while the integrated drag is stable to five
        # digits; converging on the residual would either run to the
        # iteration cap every time or declare failure on a converged answer.
        "CONV_FIELD= DRAG",
        f"CONV_CAUCHY_EPS= {settings.force_tolerance:.10g}",
        f"CONV_CAUCHY_ELEMS= {settings.force_window}",
        f"CONV_STARTITER= {settings.minimum_iterations}",
        "",
        f"MESH_FILENAME= {mesh_path.name}",
        "MESH_FORMAT= SU2",
        "TABULAR_FORMAT= CSV",
        "CONV_FILENAME= history",
        "OUTPUT_FILES= ( RESTART, SURFACE_CSV )",
        "SURFACE_FILENAME= surface_flow",
        "VOLUME_FILENAME= volume_flow",
        "RESTART_FILENAME= restart_flow",
        "SCREEN_OUTPUT= ( INNER_ITER, RMS_DENSITY, DRAG, LIFT )",
        "HISTORY_OUTPUT= ( ITER, RMS_RES, AERO_COEFF )",
        # PRIMITIVE puts Pressure, Temperature and Mach in the surface file.
        # Without it SU2 writes conservatives only, and pressure has to be
        # reconstructed — which this module can do, but reading a number is
        # better than deriving one.
        "VOLUME_OUTPUT= ( COORDINATES, SOLUTION, PRIMITIVE )",
        "OUTPUT_WRT_FREQ= 100000",
    ]
    lines += [f"{key}= {value}" for key, value in settings.extra.items()]
    return "\n".join(lines) + "\n"


def surface_axial_force(
    station: _FloatArray,
    radius: _FloatArray,
    pressure: _FloatArray,
    freestream_pressure: float,
    base_station: float,
    reference_area: float,
    dynamic_pressure: float,
    tolerance: float = 1.0e-9,
) -> tuple[float, float, _FloatArray, _FloatArray, _FloatArray]:
    """Integrate a wall pressure distribution into an axial-force coefficient.

    Splits the wall points into the lateral surface and the base disc by
    axial station, integrates :math:`\\pi\\int(p-p_\\infty)\\mathrm{d}(r^2)`
    over the former by the trapezoid rule and averages the latter over its
    area.

    Returns
    -------
    tuple
        ``(forebody_ca, base_ca, station, radius, pressure)`` with the last
        three sorted along the lateral surface.
    """
    x = np.asarray(station, dtype=np.float64)
    r = np.asarray(radius, dtype=np.float64)
    p = np.asarray(pressure, dtype=np.float64)
    if not (x.shape == r.shape == p.shape):
        msg = f"wall arrays must match; got {x.shape}, {r.shape}, {p.shape}"
        raise ValueError(msg)
    if not (np.isfinite(dynamic_pressure) and dynamic_pressure > 0.0):
        msg = f"dynamic pressure must be finite and > 0, got {dynamic_pressure}"
        raise ValueError(msg)

    span = float(np.max(x) - np.min(x))
    on_base = x > float(base_station) - max(tolerance, 1.0e-6 * span)

    # The base corner is one node shared by two faces, and its pressure
    # belongs to neither. In an Euler solution it is a singular expansion
    # point: on this cone it comes back at C_p = +0.45 where the body is at
    # +0.17 and the base at -0.10. Left in, it lands on the *outermost*
    # radius, where the area weight d(r^2) is largest, and corrupts both
    # integrals. It is therefore excluded from both, and each integral is
    # closed out to the corner radius by holding its own last interior value.
    corner_radius = float(np.max(r))
    corner = np.isclose(r, corner_radius, rtol=0.0, atol=max(tolerance, 1e-9 * span))
    lateral = ~on_base & ~corner
    base_face = on_base & ~corner

    if np.count_nonzero(lateral) < 3:
        msg = (
            f"only {np.count_nonzero(lateral)} lateral wall points found; the "
            f"base station {base_station:g} may be wrong for this surface"
        )
        raise ValueError(msg)

    order = np.argsort(x[lateral])
    x_lateral = x[lateral][order]
    r_lateral = r[lateral][order]
    p_lateral = p[lateral][order]

    closed_r = np.concatenate([r_lateral, [corner_radius]])
    closed_p = np.concatenate([p_lateral, [p_lateral[-1]]])
    forebody = np.pi * float(np.trapezoid(closed_p - freestream_pressure, closed_r**2))

    base = 0.0
    if np.any(base_face):
        sort = np.argsort(r[base_face])
        base_r = np.concatenate([r[base_face][sort], [corner_radius]])
        base_p = np.concatenate([p[base_face][sort], [p[base_face][sort][-1]]])
        # Area-weighted: the integrand is p * 2 pi r dr, so integrating
        # against r^2 weights the rim by its area, where a plain mean over
        # nodes would weight the axis as heavily and the rim is nearly all of
        # the area.
        base = -np.pi * float(np.trapezoid(base_p - freestream_pressure, base_r**2))

    scale = dynamic_pressure * reference_area
    return (
        float(forebody / scale),
        float(base / scale),
        x_lateral,
        r_lateral,
        p_lateral,
    )


def run_su2(
    mesh: MeshResult,
    profile: BodyProfile,
    mach: float,
    directory: str | Path,
    settings: SU2Settings | None = None,
    reference_area: float | None = None,
    reference_length: float | None = None,
    temperature: float = 288.15,
    pressure: float = 101325.0,
    executable: str | Path | None = None,
    timeout: float = 3600.0,
    keep_output: bool = False,
    processes: int = 1,
) -> SU2Result:
    """Write a case, run SU2 on it, and read the wall pressure back.

    Parameters
    ----------
    keep_output:
        Retain the working directory. A sweep of hundreds of points would
        otherwise leave gigabytes of restart files behind; the surface CSV
        and history are always kept because they are what a result can be
        re-derived from.

    Notes
    -----
    A non-zero exit status, a missing surface file, or a residual above the
    target all produce a result with ``converged=False`` rather than an
    exception. A sweep must be able to record that Mach 0.98 did not converge
    and carry on to Mach 1.0; stopping the run is the one outcome that makes
    a multi-day job unusable.
    """
    settings = settings if settings is not None else SU2Settings()
    work = Path(directory)
    work.mkdir(parents=True, exist_ok=True)
    binary = Path(executable) if executable is not None else find_su2()

    local_mesh = work / mesh.path.name
    if local_mesh.resolve() != mesh.path.resolve():
        shutil.copy(mesh.path, local_mesh)

    area = reference_area if reference_area is not None else profile.reference_area
    length = reference_length if reference_length is not None else 2.0 * profile.maximum_radius
    config = work / "case.cfg"
    config.write_text(_config(local_mesh, mach, settings, area, length, temperature, pressure))

    # One MPI rank per partition. Worth it above roughly 50,000 cells and a
    # net loss below: the domain decomposition and halo exchange cost more
    # than they save on a mesh that fits comfortably in cache. A sweep is
    # usually better parallelised across *points* than within one.
    command = [str(binary), config.name]
    if int(processes) > 1:
        command = ["mpirun", "-n", str(int(processes)), *command]

    completed = subprocess.run(
        command,
        cwd=work,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    (work / "su2.log").write_text(completed.stdout + "\n----- stderr -----\n" + completed.stderr)

    history = _read_history(work / "history.csv")
    settled = history.force_settled(settings.force_window, settings.force_tolerance)
    surface = _find_surface_file(work)
    ran = completed.returncode == 0 and surface is not None

    gamma = 1.4
    gas_constant = 287.058
    density = pressure / (gas_constant * temperature)
    speed = mach * np.sqrt(gamma * gas_constant * temperature)
    dynamic = 0.5 * density * speed**2

    if surface is None:
        empty = np.zeros(0)
        return SU2Result(
            mach=float(mach),
            forebody_axial=float("nan"),
            base_axial=float("nan"),
            pressure_ratio=empty,
            station=empty,
            radius=empty,
            residual=history.residual,
            iterations=history.iterations,
            converged=False,
            directory=work,
            mesh=mesh,
        )

    x, y, p = _read_surface(surface)
    forebody, base, x_lateral, r_lateral, p_lateral = surface_axial_force(
        x, y, p, pressure, float(profile.station[-1]), area, dynamic
    )

    if not keep_output:
        for name in ("restart_flow.dat", "restart_flow.csv", "volume_flow.vtu"):
            (work / name).unlink(missing_ok=True)

    return SU2Result(
        mach=float(mach),
        forebody_axial=forebody,
        base_axial=base,
        pressure_ratio=np.asarray(p_lateral / pressure),
        station=x_lateral,
        radius=r_lateral,
        residual=history.residual,
        iterations=history.iterations,
        converged=bool(ran and settled),
        directory=work,
        mesh=mesh,
    )


def _find_surface_file(work: Path) -> Path | None:
    for name in ("surface_flow.csv", "surface_flow.dat"):
        candidate = work / name
        if candidate.exists():
            return candidate
    matches = sorted(work.glob("surface_flow*.csv"))
    return matches[0] if matches else None


def _read_surface(path: Path) -> tuple[_FloatArray, _FloatArray, _FloatArray]:
    """Read x, y and static pressure from an SU2 surface CSV.

    Column names are matched case-insensitively but **exactly** — never by
    prefix. An earlier version fell back to a prefix search for pressure, and
    on a file that carried conservatives only it matched ``PointID``, read
    node numbers as pascals, and produced a negative axial force on a
    compression cone. A missing column has to be an error, not a near miss.

    If ``Pressure`` is genuinely absent the pressure is *derived* from the
    conservative variables, :math:`p = (\\gamma-1)(\\rho E - |\\rho\\mathbf{u}|^2/2\\rho)`,
    which is exact for the perfect gas SU2 is solving and does not depend on
    which derived quantities a given release chooses to write.
    """
    with path.open() as handle:
        reader = csv.reader(handle)
        header = [cell.strip().strip('"') for cell in next(reader)]
        rows = [row for row in reader if row]
    if not rows:
        msg = f"surface file {path} has a header but no data"
        raise ValueError(msg)

    lowered = [name.strip().lower() for name in header]
    table = np.array([[float(cell) for cell in row] for row in rows])

    def column(*candidates: str) -> int | None:
        for candidate in candidates:
            if candidate in lowered:
                return lowered.index(candidate)
        return None

    def require(*candidates: str) -> int:
        index = column(*candidates)
        if index is None:
            msg = f"none of {candidates} in surface file columns {header}"
            raise KeyError(msg)
        return index

    x = table[:, require("x", "points:0", "coord-x")]
    y = table[:, require("y", "points:1", "coord-y")]

    pressure_index = column("pressure")
    if pressure_index is not None:
        return x, y, table[:, pressure_index]

    density = table[:, require("density")]
    momentum_x = table[:, require("momentum_x")]
    momentum_y = table[:, require("momentum_y")]
    energy = table[:, require("energy")]
    kinetic = 0.5 * (momentum_x**2 + momentum_y**2) / density
    return x, y, np.asarray(0.4 * (energy - kinetic))


@dataclass(frozen=True)
class _History:
    """What ``history.csv`` says about how the run went."""

    residual: float
    iterations: int
    drag: _FloatArray

    def force_settled(self, window: int, tolerance: float) -> bool:
        """Is the drag coefficient stable over the last ``window`` iterations?

        Measured here rather than taken from SU2's own convergence flag,
        because the flag reports on whichever criterion the config selected
        and this is the criterion that matters: peak-to-peak spread of
        :math:`C_D` relative to its mean.
        """
        if self.drag.size < window + 1:
            return False
        tail = self.drag[-window:]
        level = float(np.mean(np.abs(tail)))
        if level <= 0.0:
            return False
        return bool((float(np.max(tail)) - float(np.min(tail))) / level < tolerance)


def _read_history(path: Path) -> _History:
    """Final residual, iteration count and drag history from ``history.csv``."""
    empty = np.zeros(0)
    if not path.exists():
        return _History(float("nan"), 0, empty)
    with path.open() as handle:
        reader = csv.reader(handle)
        try:
            header = [cell.strip().strip('"') for cell in next(reader)]
        except StopIteration:
            return _History(float("nan"), 0, empty)
        rows = [row for row in reader if row]
    if not rows:
        return _History(float("nan"), 0, empty)

    lowered = [name.strip().strip('"').lower() for name in header]

    def find(*candidates: str) -> int | None:
        for candidate in candidates:
            if candidate in lowered:
                return lowered.index(candidate)
        return None

    residual_index = find("rms[rho]", "rms_density")
    drag_index = find("cd", "drag")
    residual = float(rows[-1][residual_index]) if residual_index is not None else float("nan")
    drag = np.array([float(row[drag_index]) for row in rows]) if drag_index is not None else empty
    return _History(residual, len(rows), drag)


@dataclass(frozen=True)
class SurfaceForces:
    """Wall forces split into the part that means something and the part that does not."""

    forebody_axial: float
    """:math:`C_A` from the forebody, on the reference area. Compare this to theory."""
    base_axial: float
    """:math:`C_A` from the base disc.

    Reported separately because in an Euler solution it is **not a physical
    quantity**. Real base pressure is set by a separated viscous shear layer
    that the Euler equations do not contain, so the solver returns whatever its
    unresolved base region happens to settle at. On a Mach 8 sphere-cone that
    was :math:`C_p = +0.34` against a physical value near :math:`-0.02` —
    contributing :math:`-0.34` to an axial force whose forebody was worth
    :math:`+0.08`, and turning the reported drag negative.

    Replace it with a correlation, or add it knowingly.
    """
    normal: float
    """:math:`C_N` over the whole wall, base included."""
    forebody_normal: float
    """:math:`C_N` from the forebody alone.

    Split for the same reason the axial force is, and it matters at incidence
    rather than at zero: a base disc at angle of attack carries a normal force
    driven by the same non-physical Euler base pressure, and folding it in
    corrupts the lift. On a Mach 8 sphere-cone at six degrees it produced a
    *negative* lift coefficient at positive incidence.
    """
    base_normal: float
    pitching_moment: float
    """:math:`C_m` over the whole wall."""
    forebody_moment: float
    """:math:`C_m` from the forebody alone."""
    forebody_area: float
    base_area: float

    @property
    def axial(self) -> float:
        """Forebody plus base — the raw Euler answer, base pressure and all."""
        return float(self.forebody_axial + self.base_axial)

    def wind_axes(self, alpha: float, forebody_only: bool = True) -> tuple[float, float]:
        """Lift and drag coefficients at incidence ``alpha`` (rad).

        ``forebody_only`` by default, which is the honest choice: both the
        axial and the normal contribution of an Euler base are artefacts, and
        a lift-to-drag ratio built from them is not a property of the vehicle.
        """
        axial = self.forebody_axial if forebody_only else self.axial
        normal = self.forebody_normal if forebody_only else self.normal
        drag = axial * np.cos(alpha) + normal * np.sin(alpha)
        lift = normal * np.cos(alpha) - axial * np.sin(alpha)
        return float(lift), float(drag)


def surface_force_breakdown(
    surface_csv: str | Path,
    mesh: Any,
    freestream_pressure: float,
    dynamic_pressure: float,
    reference_area: float,
    reference_length: float,
    base_station: float | None = None,
    base_cone_deg: float = 20.0,
    moment_reference: tuple[float, float, float] = (0.0, 0.0, 0.0),
    match_tolerance: float = 1.0e-7,
) -> SurfaceForces:
    """Integrate a three-dimensional wall pressure distribution, split at a station.

    The pressure is integrated **here** rather than read from SU2's force
    output, which is the same choice :func:`surface_axial_force` makes for the
    axisymmetric case and is made for a further reason in three dimensions:
    SU2 reports one force per monitoring marker, and separating the base
    therefore needs a second marker, which needs the split to exist in the
    mesh. Re-tagging faces after the volume is meshed produces surface elements
    with no adjacent volume element and SU2 refuses the mesh; splitting before
    meshing means two discrete surfaces sharing a seam, which is fragile to
    build and easy to get subtly wrong. Reading the pressure back and
    integrating against the wall triangles we already have avoids all of it.

    ``mesh`` must be the same :class:`~aether.geometry.VehicleMesh` the domain
    was built from: its faces supply the areas and outward normals, and the CSV
    supplies the pressure at their vertices.

    Where the base begins
    ---------------------

    By default a face belongs to the base when its outward normal lies within
    ``base_cone_deg`` of the axis — when it is *nearly* aft-facing. That
    matches how the split is drawn in practice: a wind tunnel measures base
    pressure over the flat aft face with its own taps and subtracts it to
    report forebody drag, and a CFD grid carries the base as its own boundary
    patch, so the base is the face the geometry author called the base.

    Taking every aft-facing face instead — ``base_cone_deg = 90`` — sweeps in
    the downstream half of a shoulder fillet, where in supersonic flow the
    stream is still attached and expanding round the corner. That is afterbody,
    not base, and counting it as base both inflates a correlated base drag and
    removes attached-flow area from the forebody.

    ``base_station`` overrides it with an axial cut. That was the first version
    of this, and on a 10 mm-shouldered sphere-cone it swept a band of cone into
    the base, making the base patch **1.14 times** the area of the disc and
    inflating a correlated base drag by the same fraction.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    with Path(surface_csv).open() as handle:
        reader = csv.reader(handle)
        header = [cell.strip().strip('"').lower() for cell in next(reader)]
        table = np.array([[float(c) for c in row] for row in reader if row])
    index = {name: position for position, name in enumerate(header)}

    def need(*names: str) -> int:
        for name in names:
            if name in index:
                return index[name]
        msg = f"none of {names} in {sorted(index)}"
        raise KeyError(msg)

    points = table[:, [need("x"), need("y"), need("z")]]
    if "pressure" in index:
        pressure = table[:, index["pressure"]]
    else:
        density = table[:, need("density")]
        momentum = table[:, [need("momentum_x"), need("momentum_y"), need("momentum_z")]]
        energy = table[:, need("energy")]
        pressure = 0.4 * (energy - 0.5 * np.sum(momentum**2, axis=1) / density)

    # The CSV nodes are the mesh vertices, but SU2 neither preserves their order
    # nor their numbering, so they are matched on position. Quantising to a
    # tolerance and hashing is O(n) where a nearest-neighbour search would be
    # O(n log n) with a tree this does not otherwise need.
    scale = float(np.max(np.abs(vertices))) or 1.0

    def key(array: _FloatArray) -> Any:
        return np.round(array / (match_tolerance * scale)).astype(np.int64)

    lookup = {tuple(row): position for position, row in enumerate(key(points))}
    matched = np.array([lookup.get(tuple(row), -1) for row in key(vertices)])
    if np.any(matched < 0):
        missing = int(np.count_nonzero(matched < 0))
        msg = (
            f"{missing} of {vertices.shape[0]} wall vertices had no matching "
            f"point in {surface_csv}; the surface mesh and the case are not the "
            f"same geometry"
        )
        raise ValueError(msg)

    nodal = pressure[matched]
    face_pressure = nodal[faces].mean(axis=1)

    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    normal = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-300)
    centroid = triangles.mean(axis=1)

    # Force on the body is -(p - p_inf) n dA with n the outward wall normal.
    load = -((face_pressure - freestream_pressure) * area)[:, None] * normal
    scale_force = dynamic_pressure * reference_area
    aft = (
        normal[:, 0] >= np.cos(np.deg2rad(float(base_cone_deg)))
        if base_station is None
        else centroid[:, 0] >= float(base_station)
    )

    arm = centroid - np.asarray(moment_reference, dtype=np.float64)
    moment = np.cross(arm, load)

    return SurfaceForces(
        forebody_axial=float(load[~aft, 0].sum() / scale_force),
        base_axial=float(load[aft, 0].sum() / scale_force),
        normal=float(load[:, 2].sum() / scale_force),
        forebody_normal=float(load[~aft, 2].sum() / scale_force),
        base_normal=float(load[aft, 2].sum() / scale_force),
        pitching_moment=float(moment[:, 1].sum() / (scale_force * reference_length)),
        forebody_moment=float(moment[~aft, 1].sum() / (scale_force * reference_length)),
        forebody_area=float(area[~aft].sum()),
        base_area=float(area[aft].sum()),
    )
