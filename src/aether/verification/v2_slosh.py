"""V2 — slosh regularization: exact force transfer, moment error.

Paper I, §8: *"Total force and first moment transferred versus σ, N, and
station x_s; Prop. 1. Failure criterion: force error above machine
precision; moment error not O(σ²) in the interior."*

The moment criterion is a *bound*: the remark after Prop. 1 promises the
first moment within the quadrature error of the kernel's first moment,
O(σ²) in the interior. The measured interior error for resolved kernels
sits far below that bound (near the rounding floor), because the CC
quadrature of a resolved Gaussian is spectrally accurate; the O(σ²)-
scale bias appears exactly where the remark localizes it — stations
within ~2σ of an endpoint, where the kernel is truncated asymmetrically.
Both regimes are measured.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from aether.coupling import SloshCoupling, normalized_kernel
from aether.spectral import ChebyshevGrid
from aether.verification.common import VerificationReport, write_csv

__all__ = ["run_v2"]

_L = 10.0
_N_SWEEP = (16, 24, 32, 48, 64)
_GAMMAS = (1.0, 1.5, 2.0)
_STATIONS_REL = (0.02, 0.11, 0.35, 0.50, 0.77, 0.93, 0.98)
_FORCE_TOL = 5.0e-14
_N_MOMENT = 64
_X_INTERIOR = 0.43 * _L
_X_REF = 0.50 * _L


def run_v2(output_dir: Path) -> VerificationReport:
    report = VerificationReport(
        task_id="V2",
        title="Slosh regularization — exact force transfer and moment error",
        criterion=(
            "force error above machine precision; moment error not O(σ²) in the interior"
        ),
        passed=True,
    )

    # --- force transfer: sigma implied by (N, gamma) at each station -----
    rng = np.random.default_rng(20260731)
    force_rows_md: list[list[str]] = []
    force_rows_csv: list[list[object]] = []
    worst_force = 0.0
    for n in _N_SWEEP:
        grid = ChebyshevGrid(n, interval=(0.0, _L))
        for gamma in _GAMMAS:
            stations = np.array(_STATIONS_REL) * _L
            coupling = SloshCoupling(grid, stations, gamma=gamma)
            forces = rng.uniform(-1.0e4, 1.0e4, stations.size)
            q = coupling.load(forces)
            err = abs(float(coupling.transferred_force(q)) - forces.sum()) / np.sum(
                np.abs(forces)
            )
            worst_force = max(worst_force, err)
            force_rows_md.append(
                [str(n), f"{gamma:.1f}", f"{np.min(coupling.sigma):.3e}",
                 f"{np.max(coupling.sigma):.3e}", f"{err:.2e}"]
            )
            force_rows_csv.append([n, gamma, np.min(coupling.sigma), np.max(coupling.sigma), err])

    force_ok = worst_force <= _FORCE_TOL
    report.add_table(
        f"Total force transfer, {len(_STATIONS_REL)} stations per row "
        f"(relative error vs Σ|F|)",
        ["N", "γ", "σ min (m)", "σ max (m)", "rel force error"],
        force_rows_md,
    )
    report.add_section(
        "Force acceptance",
        f"Worst relative force error across the sweep: **{worst_force:.2e}** against "
        f"the machine-precision criterion {_FORCE_TOL:.0e} → "
        f"{'**PASS**' if force_ok else '**FAIL**'}. Per Prop. 1 the transfer is exact "
        "by construction of the discrete normalization; the residual is the rounding "
        "of one quadrature sum.",
    )
    write_csv(
        output_dir,
        "v2-force-transfer",
        ["N", "gamma", "sigma_min", "sigma_max", "rel_force_err"],
        force_rows_csv,
    )

    # --- moment error: interior station, resolved sigma sweep ------------
    grid = ChebyshevGrid(_N_MOMENT, interval=(0.0, _L))
    lever = grid.weights * (grid.x - _X_REF)
    h_center = np.pi * _L / (2.0 * _N_MOMENT)  # ~mid-domain node spacing
    sigmas = h_center * np.array([1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    interior_rows_md: list[list[str]] = []
    interior_rows_csv: list[list[object]] = []
    interior_ok = True
    for sigma in sigmas:
        kernel = normalized_kernel(grid, _X_INTERIOR, float(sigma))
        moment = float(lever @ kernel)
        err = abs(moment - (_X_INTERIOR - _X_REF)) / _L
        bound = (sigma / _L) ** 2
        interior_ok = interior_ok and err <= bound
        interior_rows_md.append(
            [f"{sigma:.3f}", f"{err:.2e}", f"{bound:.2e}", "yes" if err <= bound else "NO"]
        )
        interior_rows_csv.append([sigma, err, bound])
    report.add_table(
        f"Interior moment error (N = {_N_MOMENT}, x_s = {_X_INTERIOR:g}, "
        f"lever about x = {_X_REF:g})",
        ["σ (m)", "rel moment error", "O(σ²) bound (σ/L)²", "within bound"],
        interior_rows_md,
    )
    report.add_section(
        "Moment acceptance (interior)",
        f"Every resolved interior kernel transfers the first moment within the "
        f"(σ/L)² bound → {'**PASS**' if interior_ok else '**FAIL**'}. The measured "
        "errors sit near the rounding floor, far below the bound: for a kernel "
        "resolved by the grid, Clenshaw–Curtis integrates the Gaussian's first "
        "moment spectrally, so the O(σ²) allowance is consumed only where "
        "truncation breaks the kernel's symmetry (next table).",
    )
    write_csv(
        output_dir,
        "v2-moment-interior",
        ["sigma", "rel_moment_err", "sigma_sq_bound"],
        interior_rows_csv,
    )

    # --- moment bias near the endpoint -----------------------------------
    sigma_end = 0.4
    end_rows_md: list[list[str]] = []
    end_rows_csv: list[list[object]] = []
    for x_s in (2.0, 1.2, 0.8, 0.4, 0.2, 0.1):
        kernel = normalized_kernel(grid, x_s, sigma_end)
        err = abs(float(lever @ kernel) - (x_s - _X_REF)) / _L
        end_rows_md.append([f"{x_s:.1f}", f"{x_s / sigma_end:.1f}", f"{err:.2e}"])
        end_rows_csv.append([x_s, x_s / sigma_end, err])
    report.add_table(
        f"Endpoint moment bias (σ = {sigma_end} m fixed, station approaching x = 0)",
        ["x_s (m)", "x_s/σ", "rel moment error"],
        end_rows_md,
    )
    report.add_section(
        "Reading (endpoint)",
        "The bias switches on as the station enters ~2σ of the end and grows "
        "monotonically as the truncated kernel loses symmetry — the behavior the "
        "remark after Prop. 1 predicts. Tanks that drain toward a vehicle end "
        "should carry this bias in their error budget; the force transfer itself "
        "remains exact there (first table, stations at 0.02L and 0.98L).",
    )
    write_csv(
        output_dir,
        "v2-moment-endpoint",
        ["x_s", "xs_over_sigma", "rel_moment_err"],
        end_rows_csv,
    )

    report.passed = bool(force_ok and interior_ok)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run verification task V2")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    report = run_v2(args.output)
    path = report.write(args.output, "v2-slosh")
    print(f"V2 {'PASS' if report.passed else 'FAIL'} -> {path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
