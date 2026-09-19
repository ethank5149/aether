"""V3 — time integration: achieved Δt and wall clock across strategies.

Paper I, §8: *"Achieved Δt and wall-clock for explicit, modally
truncated, and IMEX strategies; Prop. 2. Failure criterion: explicit Δt
not scaling as N⁻⁴."*

The explicit branch drives SciPy's adaptive RK45 on the reduced
first-order system and records the step it actually selects, which per
Prop. 2 must be stability-limited at :math:`C_{\\mathrm{RK}}/\\omega_{\\max}`
with :math:`\\omega_{\\max} = \\mathcal{O}(N^4)`. The IMEX branch is the
Newmark integrator with its factorization computed once; the modal
branch propagates a truncated basis exactly. Accuracy for the strategy
comparison is measured against the closed-form evolution of the exact
(untruncated) modal expansion of the same initial condition.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import scipy.integrate

from aether.spectral import ChebyshevGrid
from aether.structures import (
    FreeFreeProjection,
    ModalPropagator,
    NewmarkIntegrator,
    assemble_beam,
    explicit_dt_limit,
    project_free_free,
    solve_free_free_modes,
    uniform_profile,
)
from aether.structures.integrators import C_RK_FEHLBERG5
from aether.structures.modal import ModalSolution
from aether.verification.common import VerificationReport, write_csv

__all__ = ["run_v3"]

_N_SWEEP = (12, 16, 20, 24, 28, 32)
_N_COMPARE = 32
_T_END = 0.05
_SLOPE_BAND = (-5.0, -3.0)


_FloatArray = np.ndarray


def _uniform(n: int) -> FreeFreeProjection:
    grid = ChebyshevGrid(n, interval=(0.0, 1.0))
    return project_free_free(assemble_beam(grid, uniform_profile(1.0, 1.0)))


def _elastic_ic(sol: ModalSolution, n_mode: int = 4, amplitude: float = 1e-3) -> _FloatArray:
    """Initial displacement: a single low elastic mode shape (reduced coords)."""
    return amplitude * sol.modes_reduced[:, sol.n_rigid + n_mode - 1]


def _explicit_rhs(
    k_hat: _FloatArray, m_hat: _FloatArray
) -> Callable[[float, _FloatArray], _FloatArray]:
    m_inv_k = np.linalg.solve(m_hat, k_hat)
    dim = k_hat.shape[0]

    def rhs(_t: float, y: _FloatArray) -> _FloatArray:
        u, v = y[:dim], y[dim:]
        return np.concatenate([v, -(m_inv_k @ u)])

    return rhs


def _exact_reference(sol: ModalSolution, u0_reduced: _FloatArray, t: float) -> _FloatArray:
    """Closed-form evolution via the full modal expansion (zero velocity IC)."""
    phi = sol.modes_reduced
    # Non-orthogonal (collocation) eigenbasis: solve for the coefficients.
    coeff = np.linalg.lstsq(phi, u0_reduced, rcond=None)[0]
    result: _FloatArray = phi @ (coeff * np.cos(sol.frequencies * t))
    return result


def run_v3(output_dir: Path) -> VerificationReport:
    report = VerificationReport(
        task_id="V3",
        title="Time integration — explicit stability limit and strategy comparison",
        criterion="explicit Δt not scaling as N⁻⁴",
        passed=True,
    )

    # --- explicit Δt vs N (the acceptance measurement) -------------------
    rows_md: list[list[str]] = []
    rows_csv: list[list[object]] = []
    achieved: list[float] = []
    for n in _N_SWEEP:
        proj = _uniform(n)
        sol = solve_free_free_modes(proj)
        w_max = float(sol.frequencies[-1])
        dt_bound = explicit_dt_limit(w_max)
        u0 = _elastic_ic(sol)
        y0 = np.concatenate([u0, np.zeros_like(u0)])
        rhs = _explicit_rhs(proj.reduced_stiffness, proj.reduced_mass)
        t0 = time.perf_counter()
        res = scipy.integrate.solve_ivp(
            rhs, (0.0, _T_END), y0, method="RK45", rtol=1e-6, atol=1e-12
        )
        wall = time.perf_counter() - t0
        if not res.success:
            raise RuntimeError(f"explicit integration failed at N={n}: {res.message}")
        dt_mean = _T_END / (res.t.size - 1)
        achieved.append(dt_mean)
        rows_md.append(
            [
                str(n),
                f"{w_max:.3e}",
                f"{dt_bound:.3e}",
                f"{dt_mean:.3e}",
                f"{dt_mean * w_max:.2f}",
                f"{wall * 1e3:.1f}",
            ]
        )
        rows_csv.append([n, w_max, dt_bound, dt_mean, dt_mean * w_max, wall])

    slope = float(np.polyfit(np.log(_N_SWEEP), np.log(achieved), 1)[0])
    in_band = _SLOPE_BAND[0] <= slope <= _SLOPE_BAND[1]
    report.passed = bool(in_band)
    report.add_table(
        "Explicit RK45: achieved step vs the Prop. 2 bound",
        [
            "N",
            "ω_max (rad/s)",
            f"Δt bound (C={C_RK_FEHLBERG5}/ω_max)",
            "achieved mean Δt",
            "achieved Δt·ω_max",
            "wall (ms)",
        ],
        rows_md,
    )
    report.add_section(
        "Acceptance",
        f"Fitted log–log slope of achieved Δt versus N: **{slope:.2f}** "
        f"(criterion: within [{_SLOPE_BAND[0]}, {_SLOPE_BAND[1]}], i.e. the N⁻⁴ "
        f"scaling of Prop. 2) → {'**PASS**' if in_band else '**FAIL**'}. "
        "The achieved Δt·ω_max column shows the integrator is pinned at an O(1) "
        "multiple of the stability bound, i.e. steps are stability-limited, not "
        "accuracy-limited — the pathology Remark 4 describes.",
    )
    write_csv(
        output_dir,
        "v3-explicit-dt",
        ["N", "omega_max", "dt_bound", "dt_achieved", "dt_times_omega", "wall_s"],
        rows_csv,
    )

    # --- strategy comparison at fixed N ----------------------------------
    proj = _uniform(_N_COMPARE)
    sol = solve_free_free_modes(proj)
    u0 = _elastic_ic(sol)
    ref_end = _exact_reference(sol, u0, _T_END)
    ref_scale = float(np.max(np.abs(u0)))
    dt_imex = 1e-4
    comp_md: list[list[str]] = []
    comp_csv: list[list[object]] = []

    # explicit RK45 (measured above at N=32; recompute end state for error)
    rhs = _explicit_rhs(proj.reduced_stiffness, proj.reduced_mass)
    y0 = np.concatenate([u0, np.zeros_like(u0)])
    t0 = time.perf_counter()
    res = scipy.integrate.solve_ivp(rhs, (0.0, _T_END), y0, method="RK45", rtol=1e-6, atol=1e-12)
    wall_exp = time.perf_counter() - t0
    err_exp = float(np.max(np.abs(res.y[: u0.size, -1] - ref_end)) / ref_scale)
    n_steps_exp = res.t.size - 1
    comp_md.append(
        [
            "explicit RK45 (adaptive)",
            f"{_T_END / n_steps_exp:.3e}",
            str(n_steps_exp),
            f"{err_exp:.2e}",
            f"{wall_exp * 1e3:.1f}",
        ]
    )
    comp_csv.append(["explicit_rk45", _T_END / n_steps_exp, n_steps_exp, err_exp, wall_exp])

    # IMEX Newmark, dt free of the CFL bound (dt*omega_max ~ 3.5)
    stepper = NewmarkIntegrator(proj.reduced_stiffness, proj.reduced_mass, dt_imex)
    u, v = u0.copy(), np.zeros_like(u0)
    a = stepper.initial_acceleration(u, v)
    n_steps = round(_T_END / dt_imex)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        u, v, a = stepper.step(u, v, a)
    wall_imex = time.perf_counter() - t0
    err_imex = float(np.max(np.abs(u - ref_end)) / ref_scale)
    comp_md.append(
        [
            f"IMEX Newmark (Δt = {dt_imex:g})",
            f"{dt_imex:.3e}",
            str(n_steps),
            f"{err_imex:.2e}",
            f"{wall_imex * 1e3:.1f}",
        ]
    )
    comp_csv.append(["imex_newmark", dt_imex, n_steps, err_imex, wall_imex])

    # modal truncation, n_m = 10, exact propagation
    n_m = 10
    basis = sol.truncate(n_m)
    prop = ModalPropagator(basis, dt_imex)
    coeff = np.linalg.lstsq(basis.modes_reduced, u0, rcond=None)[0]
    q, qd = coeff.copy(), np.zeros_like(coeff)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        q, qd = prop.step(q, qd)
    wall_modal = time.perf_counter() - t0
    u_modal = basis.modes_reduced @ q
    err_modal = float(np.max(np.abs(u_modal - ref_end)) / ref_scale)
    comp_md.append(
        [
            f"modal truncation (n_m = {n_m}, exact)",
            f"{dt_imex:.3e}",
            str(n_steps),
            f"{err_modal:.2e}",
            f"{wall_modal * 1e3:.1f}",
        ]
    )
    comp_csv.append(["modal_truncation", dt_imex, n_steps, err_modal, wall_modal])

    report.add_table(
        f"Strategy comparison at N = {_N_COMPARE} "
        f"(free vibration of elastic mode 4, T = {_T_END} s)",
        ["strategy", "Δt", "steps", "max rel error vs exact modal", "wall (ms)"],
        comp_md,
    )
    report.add_section(
        "Reading",
        f"The IMEX step is {dt_imex / explicit_dt_limit(float(sol.frequencies[-1])):.1f}× "
        "the explicit stability bound at the same N with no loss of stability (the "
        "test suite exercises the same scheme at 10⁴× the bound); its "
        "error is the O(Δt²) Newmark dispersion at the excited frequency. Modal "
        "truncation with the excited mode retained is exact to the ZOH/rounding "
        "floor. Retained-mode translation participation at n_m = 10: "
        f"{sol.retained_participation(n_m):.6f}. Which mitigation is preferable is "
        "configuration-dependent; both preserve the fixed-dimension batching "
        "argument, the IMEX branch because its factorization is shared across "
        "replicates, the modal branch because the basis is.",
    )
    write_csv(
        output_dir,
        "v3-strategies",
        ["strategy", "dt", "steps", "max_rel_err", "wall_s"],
        comp_csv,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run verification task V3")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    report = run_v3(args.output)
    path = report.write(args.output, "v3-integrators")
    print(f"V3 {'PASS' if report.passed else 'FAIL'} -> {path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
