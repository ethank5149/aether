"""Post-processing and sizing, matching FIATv2's operational outputs.

Milos, Chen & Squire describe three capabilities beyond the solver
itself that a run is actually used for, and all three are here.

**In-depth probes.** *"File interface.out provides temperature and heat
flux at inter-ply boundaries... useful for transfer of FIATv2 output to
another thermal analysis tool."* More important than the file format is
what the probes are *for*: flight instrumentation such as MEDLI's MISP
plugs reports temperature at fixed depths below the **original** surface,
and comparing a prediction against that data is the whole business of
material-response validation. A probe at original depth :math:`d` sits at
:math:`x = d - s` in the receding frame, and once :math:`s > d` the
thermocouple has been consumed — which is a real event in arcjet testing
and is reported as such here rather than silently extrapolated.

**Environment scaling.** *"The top line indicates that three environments
are stacked in this file. These runs will be 90% of nominal heating,
nominal heating (as tabulated in the file), and 110%."* Heating
uncertainty is the dominant term in TPS margin, so a sizing run is
always a bracket, never a single case.

**Thickness optimization.** *"The thickness of any material ply may be
optimized to achieve a specified maximum temperature at a selected
material interface."* This is what a TPS sizing run produces.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import scipy.optimize
from numpy.typing import ArrayLike, NDArray

from aether.fiat.bprime import BPrimeTable
from aether.fiat.solver import FiatSolution, FiatSolver, SolverOptions
from aether.fiat.stack import MaterialStack
from aether.fiat.surface import AerothermalEnvironment, BackfaceCondition

__all__ = [
    "DepthProbe",
    "InterfaceHistory",
    "ThicknessResult",
    "interface_histories",
    "optimize_ply_thickness",
    "probe_depths",
    "scale_environments",
    "sized_stack",
]

_FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DepthProbe:
    """Temperature history at one fixed depth below the original surface.

    Attributes
    ----------
    depth:
        Distance below the *original* surface (m) — the quantity a
        thermocouple installation is specified by.
    times:
        Times at which the probe was still in the material (s).
    temperature:
        Interpolated temperature there (K).
    consumed_at:
        Time at which recession reached this depth (s), or ``None`` if
        the probe survived the run. After this time the probe reports
        nothing, because there is nothing there.
    """

    depth: float
    times: _FloatArray
    temperature: _FloatArray
    consumed_at: float | None

    @property
    def survived(self) -> bool:
        return self.consumed_at is None

    @property
    def peak_temperature(self) -> float:
        """Maximum temperature seen before consumption, K."""
        if self.temperature.size == 0:
            return float("nan")
        return float(np.max(self.temperature))


def probe_depths(
    stack: MaterialStack,
    solution: FiatSolution,
    depths: ArrayLike,
) -> list[DepthProbe]:
    """Temperature histories at fixed depths below the original surface.

    Interpolation is linear between cell centres, with the wall
    temperature carried as the value at the receding surface. Higher-order
    interpolation would be false precision: the underlying discretisation
    is second order in space, and a thermocouple bead has finite size and
    its own installation depth uncertainty.
    """
    d = np.atleast_1d(np.asarray(depths, dtype=np.float64))
    if d.ndim != 1 or np.any(d < 0.0) or not np.all(np.isfinite(d)):
        raise ValueError("depths must be a finite 1-D array of non-negative values")
    limit = stack.initial_thickness
    if np.any(d > limit):
        raise ValueError(
            f"depth {float(np.max(d)):.6g} m lies below the stack "
            f"({limit:.6g} m thick)"
        )

    probes: list[DepthProbe] = []
    for depth in d:
        times: list[float] = []
        values: list[float] = []
        consumed: float | None = None
        for step in solution.steps:
            if step.recession >= depth:
                consumed = step.time
                break
            grid = stack.grid(step.recession)
            x = depth - step.recession
            # The surface itself and the backface bracket the cell centres.
            nodes = np.concatenate([[0.0], grid.centers, [grid.total_thickness]])
            field = np.concatenate(
                [[step.wall_temperature], step.temperature, [step.backface_temperature]]
            )
            times.append(step.time)
            values.append(float(np.interp(x, nodes, field)))
        probes.append(
            DepthProbe(
                depth=float(depth),
                times=np.asarray(times, dtype=np.float64),
                temperature=np.asarray(values, dtype=np.float64),
                consumed_at=consumed,
            )
        )
    return probes


@dataclass(frozen=True)
class InterfaceHistory:
    """Temperature and conduction flux at one ply–ply interface."""

    ply_below: int
    """Index of the ply immediately inboard of this interface."""
    times: _FloatArray
    temperature: _FloatArray
    heat_flux: _FloatArray
    """Conduction flux through the interface (W/m²), positive inward."""

    @property
    def peak_temperature(self) -> float:
        return float(np.max(self.temperature))


def interface_histories(
    stack: MaterialStack,
    solution: FiatSolution,
    environments: list[AerothermalEnvironment],
    options: SolverOptions | None = None,
) -> list[InterfaceHistory]:
    """FIATv2's ``interface.out``: temperature and flux at every ply boundary.

    The bondline temperature is the number a sizing run exists to
    produce, so it is computed from the same harmonic-mean face
    conductivity the solver uses rather than by averaging the two
    adjacent cell temperatures — the two differ by exactly the amount the
    conductivity jump matters.

    ``environments`` must be the same list the solution was produced
    with; it supplies the per-step pressure that a pressure-dependent
    conductivity needs.
    """
    if len(environments) != len(solution.steps):
        raise ValueError(
            f"need one environment per step: {len(environments)} environments "
            f"for {len(solution.steps)} steps"
        )
    solver = FiatSolver(stack, options)
    grid0 = stack.grid(0.0)
    faces = [int(f) for f in grid0.interface_faces]
    if not faces:
        return []

    out: list[InterfaceHistory] = []
    for face in faces:
        times, temps, fluxes = [], [], []
        for env, step in zip(environments, solution.steps, strict=True):
            grid = stack.grid(step.recession)
            rho = solver.bulk_density(step.component_density)
            k, _ = solver._properties(step.temperature, rho, env.pressure)
            lo, hi = face - 1, face
            a, b = 0.5 * grid.widths[lo], 0.5 * grid.widths[hi]
            k_face = (a + b) / (a / k[lo] + b / k[hi])
            distance = grid.centers[hi] - grid.centers[lo]
            flux = -k_face * (step.temperature[hi] - step.temperature[lo]) / distance
            # Interface temperature by continuity of that flux from either side.
            t_iface = step.temperature[lo] - flux * a / k[lo]
            times.append(step.time)
            temps.append(float(t_iface))
            fluxes.append(float(flux))
        out.append(
            InterfaceHistory(
                ply_below=int(np.searchsorted(np.cumsum(
                    [p.n_cells for p in stack.plies]), face, side="right")) + 1,
                times=np.asarray(times, dtype=np.float64),
                temperature=np.asarray(temps, dtype=np.float64),
                heat_flux=np.asarray(fluxes, dtype=np.float64),
            )
        )
    return out


def scale_environments(
    environments: list[AerothermalEnvironment], factor: float
) -> list[AerothermalEnvironment]:
    """Scale the heating of an environment history by ``factor``.

    FIATv2 stacks nominal, low and high heating cases in one input file
    because heating uncertainty dominates TPS margin. Scaling applies to
    the transfer coefficients and the incident radiation — the fluxes —
    and **not** to the recovery enthalpy or the pressure, which are
    thermodynamic states rather than transport rates. Scaling those too
    would silently change the chemistry the B' table is queried with.
    """
    if not (np.isfinite(factor) and factor >= 0.0):
        raise ValueError(f"factor must be finite and >= 0, got {factor}")
    scaled = []
    for env in environments:
        scaled.append(
            replace(
                env,
                film_coefficient=env.film_coefficient * factor,
                mass_transfer_coefficient=(
                    None
                    if env.mass_transfer_coefficient is None
                    else env.mass_transfer_coefficient * factor
                ),
                radiative_flux=env.radiative_flux * factor,
            )
        )
    return scaled


@dataclass(frozen=True)
class ThicknessResult:
    """Outcome of a ply-thickness optimization."""

    thickness: float
    """Converged ply thickness (m)."""
    peak_temperature: float
    """Peak temperature reached at the monitored location (K)."""
    target_temperature: float
    iterations: int
    converged: bool


def optimize_ply_thickness(
    stack: MaterialStack,
    ply_index: int,
    times: ArrayLike,
    environments: list[AerothermalEnvironment],
    table: BPrimeTable,
    backface: BackfaceCondition,
    target_temperature: float,
    bounds: tuple[float, float],
    initial_temperature: float = 300.0,
    options: SolverOptions | None = None,
    tolerance: float = 1.0e-5,
) -> ThicknessResult:
    """Size one ply so the backface reaches a specified peak temperature.

    FIATv2's headline sizing capability. The monitored location is the
    backface of the stack, which is the bondline in the usual
    configuration.

    Thicker TPS gives a cooler backface, so peak temperature is monotone
    decreasing in thickness and Brent's method on the bracket is both
    safe and fast. The bracket is checked before iterating: a target that
    lies outside what the bounds can achieve is a specification error,
    and reporting it as one is more useful than returning whichever
    endpoint was closest.
    """
    if not 0 <= ply_index < stack.n_plies:
        raise ValueError(f"ply_index {ply_index} outside the stack")
    lo, hi = float(bounds[0]), float(bounds[1])
    if not (0.0 < lo < hi):
        raise ValueError(f"need 0 < bounds[0] < bounds[1], got {bounds}")
    if not (np.isfinite(target_temperature) and target_temperature > 0.0):
        raise ValueError("target_temperature must be finite and > 0")

    calls = 0

    def peak(thickness: float) -> float:
        nonlocal calls
        calls += 1
        plies = list(stack.plies)
        plies[ply_index] = replace(plies[ply_index], thickness=thickness)
        solution = FiatSolver(MaterialStack(plies), options).solve(
            np.asarray(times, dtype=np.float64),
            environments,
            table,
            backface,
            initial_temperature,
        )
        return float(np.max(solution.backface_temperature))

    peak_lo, peak_hi = peak(lo), peak(hi)
    if not (peak_hi <= target_temperature <= peak_lo):
        raise ValueError(
            f"target {target_temperature:.1f} K is not bracketed: thickness "
            f"{lo:.4g} m gives {peak_lo:.1f} K and {hi:.4g} m gives "
            f"{peak_hi:.1f} K. Widen the bounds or change the target."
        )

    root, results = scipy.optimize.brentq(
        lambda t: peak(t) - target_temperature,
        lo,
        hi,
        xtol=tolerance,
        full_output=True,
    )
    return ThicknessResult(
        thickness=float(root),
        peak_temperature=peak(float(root)),
        target_temperature=float(target_temperature),
        iterations=calls,
        converged=bool(results.converged),
    )


def sized_stack(stack: MaterialStack, ply_index: int, thickness: float) -> MaterialStack:
    """A copy of ``stack`` with one ply's thickness replaced."""
    plies = list(stack.plies)
    plies[ply_index] = replace(plies[ply_index], thickness=float(thickness))
    return MaterialStack(plies)
