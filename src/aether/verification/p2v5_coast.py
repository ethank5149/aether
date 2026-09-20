"""II-V5 — regime transition: coast step size, wall clock, energy conservation.

Acceptance criterion: *"Coast-phase step size and wall-clock, single-integration
versus frozen-structure; energy conservation over the coast. Failure
criterion: secular energy drift exceeding 1e-8 per orbit."*

The question has to be set up honestly: carrying the
structural block through a coast in which it holds no energy leaves the
step controller subject to the :math:`\\mathcal{O}(N^{-4})` structural
bound during a phase whose rigid-body dynamics would permit far larger
steps. Freezing the block once it is provably quiescent reintroduces a
switch. This measures the trade instead of arguing it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from aether.orbital import (
    EARTH,
    compare_coast_strategies,
    orbital_elements,
    propagate_coast,
    regime_transition_profile,
    secular_rates,
)
from aether.verification.common import VerificationReport, write_csv

__all__ = ["run_p2v5"]

_ALTITUDE = 400.0e3
_INCLINATION_DEG = 51.6
_ENERGY_TOL = 1.0e-8
_STRUCTURAL_OMEGA = 100.0
_COAST_SECONDS = 300.0


def _circular_state() -> tuple[np.ndarray, np.ndarray, float]:
    a = EARTH.radius + _ALTITUDE
    inc = np.deg2rad(_INCLINATION_DEG)
    speed = np.sqrt(EARTH.mu / a)
    r = np.array([a, 0.0, 0.0])
    v = np.array([0.0, speed * np.cos(inc), speed * np.sin(inc)])
    return r, v, float(2.0 * np.pi * np.sqrt(a**3 / EARTH.mu))


def run_p2v5(output_dir: Path) -> VerificationReport:
    report = VerificationReport(
        task_id="II-V5",
        title="Regime transition — coast conservation, step size and wall clock",
        criterion="secular energy drift exceeding 1e-8 per orbit",
        passed=True,
    )
    r0, v0, period = _circular_state()
    report.add_section(
        "Configuration",
        f"Circular orbit at {_ALTITUDE / 1e3:g} km altitude, "
        f"{_INCLINATION_DEG}° inclination; period {period:.1f} s. Gravity is the "
        f"J₂ model of Eq. (7.1)–(7.2) with the paper's constants "
        f"(μ = {EARTH.mu:.9e} m³/s², R⊕ = {EARTH.radius:g} m, J₂ = {EARTH.j2:g}).",
    )

    # --- conservation over one orbit --------------------------------------
    rows_md: list[list[str]] = []
    rows_csv: list[list[object]] = []
    worst_energy = 0.0
    for revolutions in (0.25, 0.5, 1.0, 2.0):
        res = propagate_coast(r0, v0, revolutions * period)
        per_orbit = res.energy_drift / max(revolutions, 1e-12)
        worst_energy = max(worst_energy, per_orbit)
        rows_md.append(
            [f"{revolutions:g}", f"{res.energy_drift:.3e}", f"{per_orbit:.3e}",
             f"{res.angular_momentum_drift:.3e}"]
        )
        rows_csv.append([revolutions, res.energy_drift, per_orbit,
                         res.angular_momentum_drift])
    energy_ok = worst_energy <= _ENERGY_TOL
    report.add_table(
        "Invariant drift versus arc length",
        ["revolutions", "relative energy drift", "drift per orbit", "h_z drift"],
        rows_md,
    )
    report.add_section(
        "Conservation acceptance",
        f"Worst secular energy drift: **{worst_energy:.2e} per orbit** against the "
        f"criterion {_ENERGY_TOL:.0e} → {'**PASS**' if energy_ok else '**FAIL**'}, "
        f"a margin of roughly {np.log10(_ENERGY_TOL / max(worst_energy, 1e-300)):.0f} "
        "decades. The polar angular momentum is reported alongside because the "
        "J₂ field is axisymmetric as well as conservative: a scheme can leak one "
        "invariant while holding the other, so both are checked.",
    )
    write_csv(
        output_dir, "p2v5-conservation",
        ["revolutions", "energy_drift", "energy_drift_per_orbit", "hz_drift"], rows_csv,
    )

    # --- secular J2 signature ---------------------------------------------
    res = propagate_coast(r0, v0, period)
    el0 = orbital_elements(r0, v0)
    el1 = orbital_elements(res.states[:3, -1], res.states[3:, -1])
    rates = secular_rates(el0["semi_major_axis"], el0["eccentricity"], el0["inclination"])
    measured = ((el1["raan"] - el0["raan"] + np.pi) % (2 * np.pi)) - np.pi
    predicted = rates["raan_rate"] * period
    raan_ok = abs(measured - predicted) <= 0.05 * abs(predicted)
    spherical = propagate_coast(r0, v0, period / 2, include_j2=False)
    half = propagate_coast(r0, v0, period / 2)
    separation = float(np.linalg.norm(half.states[:3, -1] - spherical.states[:3, -1]))
    report.add_table(
        "Secular J₂ signature over one orbit",
        ["quantity", "measured", "analytic first-order"],
        [
            ["nodal regression (deg/day)",
             f"{np.rad2deg(measured) / period * 86400:.4f}",
             f"{np.rad2deg(rates['raan_rate']) * 86400:.4f}"],
            ["J₂ vs spherical position difference at half an orbit",
             f"{separation / 1e3:.1f} km", "order kilometres (§7.1)"],
        ],
    )
    report.add_section(
        "Reading the secular terms",
        f"Nodal regression agrees with the classical first-order rate to "
        f"{abs(measured - predicted) / abs(predicted):.1%} "
        f"({'**PASS**' if raan_ok else '**FAIL**'}); the residual is the "
        "short-period oscillation the secular average deliberately omits. The "
        f"{separation / 1e3:.0f} km separation from a spherical model over half "
        "an orbit is what §7.1 means by 'large compared with any meaningful "
        "terminal accuracy requirement' — the reason J₂ is not optional here.",
    )

    # --- regime transition without a branch -------------------------------
    profile = regime_transition_profile(7000.0, 8000.0)
    ratio = profile.acceleration_ratio
    log_slope = np.diff(np.log(ratio))
    smooth = float(np.max(np.abs(np.diff(log_slope))) / abs(float(log_slope[0])))
    monotone = bool(np.all(np.diff(ratio) < 0.0))
    transition_ok = monotone and smooth < 1e-3
    report.add_table(
        "Aerodynamic-to-gravitational acceleration ratio versus altitude",
        ["altitude (km)", "ratio"],
        [
            [f"{alt / 1e3:.0f}", f"{float(np.interp(alt, profile.altitudes, ratio)):.2e}"]
            for alt in (0.0, 50e3, 86e3, 100e3, 150e3, 200e3, 300e3, 500e3)
        ],
    )
    report.add_section(
        "Regime transition",
        f"The ratio decays monotonically through "
        f"{np.log10(ratio[0] / ratio[-1]):.0f} decades with a log-slope varying "
        f"by {smooth:.1e} relative between samples — that is, smoothly, with no "
        f"discontinuity anywhere ({'**PASS**' if transition_ok else '**FAIL**'}). "
        "This is the whole mechanism of §7.2: no Kármán-line switch is taken, "
        "the aerodynamic term simply stops mattering, so one integration spans "
        "boost, coast and re-entry without a handoff to reinterpolate across. "
        "The single-scale-height exponential used here over-predicts density "
        "above about 86 km, which makes these crossing altitudes conservative.",
    )

    # --- II-V5 proper: single-integration versus frozen structure ----------
    single, frozen = compare_coast_strategies(
        r0, v0, _COAST_SECONDS, structural_frequency=_STRUCTURAL_OMEGA, rtol=1e-9
    )
    speedup = single.n_rhs_evaluations / max(frozen.n_rhs_evaluations, 1)
    strategy_ok = (
        frozen.energy_drift <= _ENERGY_TOL
        and single.energy_drift <= _ENERGY_TOL
        and frozen.structural_energy_ratio < 1e-5
    )
    report.add_table(
        f"Coast strategies over {_COAST_SECONDS:g} s with a "
        f"{_STRUCTURAL_OMEGA:g} rad/s structural mode",
        ["strategy", "RHS evaluations", "mean step (s)", "wall (s)",
         "energy drift", "final modal energy ratio"],
        [
            [s.label, f"{s.n_rhs_evaluations:,}", f"{s.mean_step:.4f}",
             f"{s.wall_time:.3f}", f"{s.energy_drift:.2e}",
             f"{s.structural_energy_ratio:.2e}"]
            for s in (single, frozen)
        ],
    )
    report.add_section(
        "Strategy acceptance",
        f"Freezing the structural block once its modal energy has fallen to "
        f"{frozen.structural_energy_ratio:.1e} of its initial value costs "
        f"**{speedup:.0f}× fewer right-hand-side evaluations** and "
        f"{single.wall_time / max(frozen.wall_time, 1e-9):.0f}× less wall clock, "
        f"with both strategies holding energy to better than {_ENERGY_TOL:.0e} "
        f"({'**PASS**' if strategy_ok else '**FAIL**'}). The measurement settles "
        "Remark 5 in favour of freezing *for a coast of this length*: the "
        "structural stability bound, not the orbital dynamics, sets the step "
        "while the block is live, and the block is demonstrably quiescent long "
        "before the coast ends. The switch is defensible precisely because the "
        "freeze condition is checked rather than assumed — and note the saving "
        "grows with coast duration, since the ring-down time is fixed while the "
        "coast is not.",
    )
    write_csv(
        output_dir, "p2v5-strategies",
        ["strategy", "rhs_evaluations", "mean_step_s", "wall_s", "energy_drift",
         "modal_energy_ratio"],
        [[s.label, s.n_rhs_evaluations, s.mean_step, s.wall_time, s.energy_drift,
          s.structural_energy_ratio] for s in (single, frozen)],
    )

    report.passed = bool(energy_ok and raan_ok and transition_ok and strategy_ok)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run verification task II-V5")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    report = run_p2v5(args.output)
    path = report.write(args.output, "p2v5-coast")
    print(f"II-V5 {'PASS' if report.passed else 'FAIL'} -> {path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
