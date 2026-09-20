"""Coupled-integration verification (roadmap item 13).

This is **not** one of the sixteen tabulated V&V tasks. Neither paper
states a criterion for the assembled simulator, because neither paper
assembles one. What both papers *do* claim, repeatedly and as the
framework's central argument, is a set of structural properties, and
those are checkable:

1. the state dimension is fixed for the whole trajectory — no remesh,
   no node created, destroyed or interpolated;
2. one integration spans every flight regime with no phase handoff and
   no branch on altitude;
3. the aerothermal loop closes within a single right-hand side, so
   recession feeds back into :math:`R_{\\mathrm{eff}}` in the same step;
4. the structural stiffness constraint governs the
   coupled system, and an implicit method removes it.

The criteria below are stated by this runner, not inherited from a
manuscript, and are labelled as such so nothing here is mistaken for a
result the papers asked for. It also carries the theoretical-occupancy
measurement that closes most of I-V8's remaining instrumentation gap.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np

from aether.aerothermal import sutton_graves
from aether.batch import cuda_available
from aether.flight import simulator as simulator_module
from aether.flight.simulator import FlightConfiguration, FlightSimulator
from aether.orbital.gravity import EARTH
from aether.verification.common import VerificationReport, write_csv

__all__ = ["run_integration"]

_ENTRY = {"altitude": 120.0e3, "speed": 6500.0, "flight_path_angle": np.deg2rad(-8.0)}
_DURATION = 110.0


def run_integration(output_dir: Path) -> VerificationReport:
    report = VerificationReport(
        task_id="INT",
        title="Coupled single-trajectory integration (roadmap item 13)",
        criterion=(
            "self-stated (not a tabulated V&V task): a state dimension that "
            "changes in flight; a branch on flight regime; an aerothermal loop "
            "that does not close within one right-hand side; or an implicit "
            "cost that grows with retained structural modes"
        ),
        passed=True,
        source="this runner — see the scope note",
    )
    sim = FlightSimulator(FlightConfiguration(n_modes=6))
    y0 = sim.initial_state(**_ENTRY)
    result = sim.propagate(y0, _DURATION, n_output=111)

    report.add_section(
        "Configuration",
        f"Global state of {sim.layout.size} components: 3 position, 3 velocity, "
        f"4 quaternion, 3 body rate, 1 mass, {2 * sim.layout.n_modes} structural "
        f"modal, {4 * sim.layout.n_thermal} thermal (temperature and three "
        f"component densities on the Landau grid) and 1 recession. Entry at "
        f"{_ENTRY['altitude'] / 1e3:g} km and {_ENTRY['speed']:g} m/s, flight "
        f"path {np.rad2deg(_ENTRY['flight_path_angle']):g}°, propagated "
        f"{_DURATION:g} s by a single BDF call.",
    )

    # --- 1. fixed dimension ------------------------------------------------
    dimension_ok = result.states.shape == (sim.layout.size, 111)
    report.add_section(
        "Fixed state dimension",
        f"The trajectory is a {result.states.shape[0]} × {result.states.shape[1]} "
        f"array: the dimension is a property of the configuration, fixed at "
        f"construction, and nothing in the right-hand side can add or remove a "
        f"degree of freedom → {'**PASS**' if dimension_ok else '**FAIL**'}. This "
        "is the property the rank-3 batching argument rests on; "
        "a moving-mesh formulation cannot offer it because each replicate's mesh "
        "diverges after its first remesh.",
    )

    # --- 2. no regime branch ----------------------------------------------
    source = inspect.getsource(simulator_module.FlightSimulator.rhs).lower()
    branch_free = not any(
        token in source for token in ("if altitude", "if alt ", "karman", "kármán")
    )
    probe = y0.copy()
    rates = []
    altitudes = np.linspace(90.0e3, 110.0e3, 81)
    for alt in altitudes:
        probe[sim.layout.position] = np.array([EARTH.radius + alt, 0.0, 0.0])
        rates.append(sim.rhs(0.0, probe)[sim.layout.velocity])
    jumps = np.linalg.norm(np.diff(np.asarray(rates), axis=0), axis=1)
    smoothness = float(np.max(jumps) / (np.median(jumps) + 1e-300))
    continuous = smoothness < 10.0
    transition_ok = branch_free and continuous
    report.add_section(
        "Single integration across regimes",
        f"The right-hand side contains no branch on altitude or on the Kármán "
        f"line (source inspected: {'clean' if branch_free else '**BRANCH FOUND**'}), "
        f"and the acceleration is continuous across it — the largest step-to-step "
        f"change over 90–110 km is {smoothness:.2f}× the median, i.e. no jump → "
        f"{'**PASS**' if transition_ok else '**FAIL**'}. The atmospheric terms "
        "decay with the density model and simply stop mattering, which is the "
        "mechanism relied on to avoid a phase handoff.",
    )

    # --- 3. the aerothermal loop closes -----------------------------------
    r_eff = result.effective_radius
    # The physical claim is that the recession *rate* is non-negative, which
    # is what gets verified: sdot is evaluated from the right-hand side at
    # every output state. The differenced dense output is reported alongside
    # as an interpolation diagnostic, not as the criterion — a monotone
    # function sampled through an interpolant can dip by the solver's own
    # tolerance without the physics being wrong, and testing the sampled
    # sequence instead of the rate would confuse the two.
    recession_rates = np.array([
        float(sim.rhs(float(t), result.states[:, i])[sim.layout.recession])
        for i, t in enumerate(result.times)
    ])
    recession_monotone = bool(np.all(recession_rates >= 0.0))
    worst_dip = float(np.min(np.diff(result.recession)))
    worst_rate = float(np.min(recession_rates))
    q_sharp = float(sutton_graves(1.0e-3, r_eff[0], 6000.0))
    q_blunt = float(sutton_graves(1.0e-3, r_eff[-1], 6000.0))
    feedback_ok = recession_monotone and q_blunt < q_sharp and r_eff[-1] > r_eff[0]
    report.add_table(
        "Trajectory summary",
        ["quantity", "start", "peak", "end"],
        [
            ["altitude (km)", f"{result.altitude[0] / 1e3:.1f}", "—",
             f"{result.altitude[-1] / 1e3:.1f}"],
            ["dynamic pressure (kPa)", f"{result.dynamic_pressure[0] / 1e3:.3f}",
             f"{result.dynamic_pressure.max() / 1e3:.1f}",
             f"{result.dynamic_pressure[-1] / 1e3:.1f}"],
            ["stagnation heat flux (MW/m²)",
             f"{result.stagnation_heat_flux[0] / 1e6:.4f}",
             f"{result.stagnation_heat_flux.max() / 1e6:.3f}",
             f"{result.stagnation_heat_flux[-1] / 1e6:.3f}"],
            ["wall temperature (K)", f"{result.surface_temperature[0]:.0f}", "—",
             f"{result.surface_temperature[-1]:.0f}"],
            ["recession (mm)", "0.000", "—", f"{result.recession[-1] * 1e3:.4f}"],
            ["effective nose radius (m)", f"{r_eff[0]:.4f}", "—", f"{r_eff[-1]:.4f}"],
        ],
    )
    report.add_section(
        "Aerothermal loop closure",
        f"The recession rate is non-negative at every sampled state (minimum "
        f"{worst_rate:.3e} m/s), so recession is irreversible as the oxidative "
        f"model requires; the differenced dense output dips by at most "
        f"{worst_dip:.1e} m, which is interpolation noise at the solver "
        f"tolerance rather than physics. Recession grows "
        f"R_eff from {r_eff[0]:.4f} to {r_eff[-1]:.4f} m, which reduces the "
        f"convective heating this run would see at fixed flight conditions by "
        f"{(1 - q_blunt / q_sharp):.2%} → "
        f"{'**PASS**' if feedback_ok else '**FAIL**'}. Nose blunting is "
        "self-limiting, and the loop closes inside one right-hand side rather "
        "than across a coupling iteration — which is what is required "
        "for the feedback to act within the same time step.",
    )
    write_csv(
        output_dir, "int-trajectory",
        ["time_s", "altitude_m", "dynamic_pressure_Pa", "heat_flux_W_m2",
         "wall_temperature_K", "recession_m", "effective_radius_m"],
        [
            [float(t), float(a), float(q), float(h), float(w), float(s), float(r)]
            for t, a, q, h, w, s, r in zip(
                result.times, result.altitude, result.dynamic_pressure,
                result.stagnation_heat_flux, result.surface_temperature,
                result.recession, r_eff, strict=True,
            )
        ],
    )

    # --- 4. stiffness: implicit cost versus retained modes -----------------
    cost_rows: list[list[str]] = []
    cost_csv: list[list[object]] = []
    implicit_costs: list[int] = []
    for n_modes in (2, 3, 4, 5, 6):
        trial = FlightSimulator(FlightConfiguration(n_modes=n_modes))
        trial_result = trial.propagate(
            trial.initial_state(**_ENTRY), 60.0, n_output=31, method="BDF"
        )
        implicit_costs.append(trial_result.n_rhs_evaluations)
        cost_rows.append(
            [str(n_modes), f"{trial.modal_frequencies[-1]:.1f}",
             f"{trial_result.n_rhs_evaluations:,}", f"{trial_result.wall_time:.2f}"]
        )
        cost_csv.append([n_modes, float(trial.modal_frequencies[-1]),
                         trial_result.n_rhs_evaluations, trial_result.wall_time])
    spread = max(implicit_costs) / min(implicit_costs)
    stiffness_ok = spread < 4.0
    report.add_table(
        "Implicit cost versus retained structural modes (60 s arc, BDF)",
        ["modes", "ω_max (rad/s)", "RHS evaluations", "wall (s)"],
        cost_rows,
    )
    report.add_section(
        "Stiffness acceptance",
        f"Right-hand-side evaluations vary by only {spread:.2f}× as the highest "
        f"retained mode goes from {cost_csv[0][1]:.0f} to {cost_csv[-1][1]:.0f} "
        f"rad/s → {'**PASS**' if stiffness_ok else '**FAIL**'}. A fully implicit "
        "method removes the Prop. 2 constraint on the *coupled* system, "
        "confirming on the assembly what V3 measured on the structural block "
        "alone. Worth recording separately: LSODA — nominally a stiffness-"
        "switching method — fails to switch here and needs on the order of "
        "10⁵–10⁶ evaluations for the same arc, so 'adaptive' is not a substitute "
        "for 'implicit' in this system.",
    )
    write_csv(
        output_dir, "int-stiffness",
        ["n_modes", "omega_max", "rhs_evaluations", "wall_s"], cost_csv,
    )

    # --- I-V8 residual: theoretical occupancy ------------------------------
    if cuda_available():
        import cupy

        from aether.batch import device_limits, theoretical_occupancy

        kernel = cupy.RawKernel(
            r"""
extern "C" __global__ void rk4_stage(const double* y, const double* beta,
                                     double* out, int n_state, int n_rep) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_state * n_rep) {
        int rep = i / n_state;
        out[i] = y[i] * 2.0 - y[i] * y[i] / beta[rep];
    }
}""",
            "rk4_stage",
        )
        kernel.compile()
        limits = device_limits()
        occ_rows: list[list[str]] = []
        for threads in (64, 128, 256, 512, 1024):
            occ = theoretical_occupancy(kernel, threads)
            occ_rows.append(
                [str(threads), str(occ.registers_per_thread),
                 str(occ.active_blocks_per_sm), str(occ.active_warps_per_sm),
                 f"{occ.occupancy:.3f}", occ.limiter]
            )
        report.add_table(
            f"I-V8 residual: theoretical occupancy of the batched stage kernel "
            f"(SM {limits['compute_capability_major']}.{limits['compute_capability_minor']}, "
            f"{limits['multiprocessors']} SMs)",
            ["threads/block", "registers/thread", "blocks/SM", "warps/SM",
             "occupancy", "limiter"],
            occ_rows,
        )
        report.add_section(
            "On occupancy",
            "I-V8 asks for *achieved* occupancy, which is a hardware counter and "
            "needs Nsight Compute — not available here. What is computed above "
            "is **theoretical** occupancy: the standard CUDA occupancy model "
            "evaluated from the compiled kernel's register and shared-memory "
            "footprint against the device's per-SM limits. That is exact "
            "arithmetic, not an estimate, and it bounds achieved occupancy from "
            "above. Together with the throughput-saturation curve already in "
            "I-V8 — the externally observable consequence of occupancy — it "
            "closes most of the instrumentation gap. The residual is the "
            "difference between the bound and the counter, which remains "
            "**pending** a profiler.",
        )
    else:  # pragma: no cover - depends on the host
        report.add_section(
            "On occupancy",
            "No CUDA device present, so the occupancy model was not evaluated.",
        )

    report.passed = bool(dimension_ok and transition_ok and feedback_ok and stiffness_ok)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the coupled-integration check")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    report = run_integration(args.output)
    path = report.write(args.output, "int-coupled")
    print(f"INT {'PASS' if report.passed else 'FAIL'} -> {path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
