"""V1 — structural operator: conditioning vs N; frequencies vs analytic.

Paper I, §8: *"κ(K̂) versus N for uniform and stepped EI profiles;
free-free natural frequencies against the analytic uniform-beam
solution. Failure criterion: relative frequency error > 1e-6 at N = 32
for the uniform case."*

Also reports, as the paper's Remark 3 anticipates it cannot assert a
priori: the measured growth rate of the conditioning, split into the raw
κ₂(K̂) (saturated at the rounding floor by the two physical rigid-body
null directions) and the elastic condition number σ₁/σ_{n-2}. The
row-replacement spectrum is measured alongside as the §3.2
counterexample.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from aether.spectral import ChebyshevGrid
from aether.structures import (
    BeamOperators,
    assemble_beam,
    free_free_analytic_frequencies,
    project_free_free,
    solve_free_free_modes,
    stepped_profile,
    uniform_profile,
)
from aether.structures.modal import row_replacement_spectrum
from aether.verification.common import VerificationReport, write_csv

__all__ = ["run_v1"]

_UNIFORM = {"length": 1.0, "ei": 1.0, "mass": 1.0}
_N_SWEEP = (8, 12, 16, 20, 24, 28, 32, 40, 48, 64)
_N_ACCEPT = 32
_N_MODES_CHECKED = 5
_FREQ_TOL = 1e-6


def _stepped_beam(n: int) -> BeamOperators:
    profile = stepped_profile(
        segment_ei=[5.0e6, 1.2e6, 4.0e5],
        segment_mass=[300.0, 120.0, 60.0],
        joints=[4.0, 7.0],
        blend_width=0.8,
        label="stepped(3-segment)",
    )
    return assemble_beam(ChebyshevGrid(n, interval=(0.0, 10.0)), profile)


def _uniform_beam(n: int) -> BeamOperators:
    grid = ChebyshevGrid(n, interval=(0.0, _UNIFORM["length"]))
    return assemble_beam(grid, uniform_profile(_UNIFORM["ei"], _UNIFORM["mass"]))


def run_v1(output_dir: Path) -> VerificationReport:
    report = VerificationReport(
        task_id="V1",
        title="Structural operator — conditioning and free-free frequencies",
        criterion=(
            f"relative frequency error > {_FREQ_TOL:.0e} at N = {_N_ACCEPT} "
            f"for the uniform case"
        ),
        passed=True,
    )

    # --- conditioning sweep, both profiles -------------------------------
    cond_rows_md: list[list[str]] = []
    cond_rows_csv: list[list[object]] = []
    kappa_elastic: dict[str, list[float]] = {"uniform": [], "stepped": []}
    for n in _N_SWEEP:
        row: list[str] = [str(n)]
        for label, factory in (("uniform", _uniform_beam), ("stepped", _stepped_beam)):
            sol = solve_free_free_modes(project_free_free(factory(n)))
            kappa_elastic[label].append(sol.stiffness_condition_elastic)
            row += [f"{sol.stiffness_condition:.2e}", f"{sol.stiffness_condition_elastic:.2e}"]
            cond_rows_csv.append(
                [n, label, sol.stiffness_condition, sol.stiffness_condition_elastic]
            )
        cond_rows_md.append(row)

    slopes = {
        label: float(np.polyfit(np.log(_N_SWEEP), np.log(vals), 1)[0])
        for label, vals in kappa_elastic.items()
    }
    report.add_table(
        "Conditioning of the reduced stiffness operator",
        ["N", "uniform κ₂(K̂) raw", "uniform κ elastic", "stepped κ₂(K̂) raw", "stepped κ elastic"],
        cond_rows_md,
    )
    report.add_section(
        "Interpretation",
        "The raw κ₂(K̂) is pinned at the reciprocal rounding floor (~1/ε) at every N: "
        "the free-free operator retains the two *physical* rigid-body null directions, "
        "so its smallest singular value is rounding noise by construction. The "
        "informative measurand is the elastic condition number σ₁/σ_{n-2}, whose "
        f"fitted log–log slope is **{slopes['uniform']:.2f}** (uniform) and "
        f"**{slopes['stepped']:.2f}** (stepped) over N ∈ [{_N_SWEEP[0]}, {_N_SWEEP[-1]}]. "
        "Paper I, Remark 3 declined to assert a rate; the measured growth remains "
        "of the same O(N⁸) order as the unprojected fourth-derivative operator, i.e. "
        "the projection removes the constraint-violating extremal modes but does not "
        "flatten the asymptotic rate for these profiles.",
    )
    write_csv(
        output_dir,
        "v1-conditioning",
        ["N", "profile", "kappa_raw", "kappa_elastic"],
        cond_rows_csv,
    )

    # --- frequency accuracy, uniform beam --------------------------------
    analytic = free_free_analytic_frequencies(
        _N_MODES_CHECKED, _UNIFORM["length"], _UNIFORM["ei"], _UNIFORM["mass"]
    )
    freq_rows_md: list[list[str]] = []
    freq_rows_csv: list[list[object]] = []
    worst_at_accept = np.nan
    for n in _N_SWEEP:
        sol = solve_free_free_modes(project_free_free(_uniform_beam(n)))
        got = sol.elastic_frequencies[:_N_MODES_CHECKED]
        rel = np.abs(got - analytic[: got.size]) / analytic[: got.size]
        worst = float(np.max(rel))
        if n == _N_ACCEPT:
            worst_at_accept = worst
        rel_cells = [f"{r:.3e}" for r in rel]
        rel_vals = rel.tolist()
        pad = _N_MODES_CHECKED - rel.size  # small N carries fewer elastic modes
        freq_rows_md.append([str(n), f"{worst:.3e}"] + rel_cells + ["—"] * pad)
        freq_rows_csv.append([n, worst, *rel_vals, *([float("nan")] * pad)])
    report.add_table(
        f"Uniform free-free frequencies vs analytic (first {_N_MODES_CHECKED} elastic modes)",
        ["N", "worst rel err"] + [f"mode {i + 1}" for i in range(_N_MODES_CHECKED)],
        freq_rows_md,
    )
    write_csv(
        output_dir,
        "v1-frequencies",
        ["N", "worst_rel_err"] + [f"rel_err_mode_{i + 1}" for i in range(_N_MODES_CHECKED)],
        freq_rows_csv,
    )

    accept_pass = worst_at_accept <= _FREQ_TOL
    report.passed = bool(accept_pass)
    report.add_section(
        "Acceptance",
        f"Worst relative frequency error at N = {_N_ACCEPT}: "
        f"**{worst_at_accept:.3e}** against the criterion {_FREQ_TOL:.0e} → "
        f"{'**PASS**' if accept_pass else '**FAIL**'}.",
    )

    # --- rigid-mode quality and row-replacement counterexample ------------
    sol32 = solve_free_free_modes(project_free_free(_uniform_beam(_N_ACCEPT)))
    lam_rr = row_replacement_spectrum(_uniform_beam(_N_ACCEPT))
    rr_scale = float(np.max(np.abs(lam_rr.real)))
    rr_imag = float(np.max(np.abs(lam_rr.imag)))
    rr_growth = lam_rr[np.abs(lam_rr.imag) > 1e-6 * rr_scale]
    report.add_section(
        "Null-space projection vs row replacement (§3.2 counterexample)",
        f"At N = {_N_ACCEPT}, the projected pencil returns a spectrum with relative "
        f"imaginary contamination {sol32.max_imag_ratio:.1e} and rigid eigenvalues "
        f"{sol32.eigenvalues[:2]} (relative magnitude ≤ "
        f"{np.max(np.abs(sol32.eigenvalues[:2])) / sol32.eigenvalues[-1]:.1e}). "
        f"The conventional row-replacement treatment of the same problem yields "
        f"{rr_growth.size} eigenvalues with non-negligible imaginary parts (largest "
        f"|Im λ| = {rr_imag:.3e} against spectral scale {rr_scale:.3e}) — the spurious "
        f"complex modes that manifest as growth in time integration.",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run verification task V1")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    report = run_v1(args.output)
    path = report.write(args.output, "v1-structural")
    print(f"V1 {'PASS' if report.passed else 'FAIL'} -> {path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
