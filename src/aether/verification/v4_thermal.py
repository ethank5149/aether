"""V4 — ablation: method of manufactured solutions; FIAT reference case.

Paper I, §8: *"Method of manufactured solutions on Eqs. (3.17)–(3.18);
in-depth temperature and recession against a FIAT reference case.
Failure criterion: recession disagreement > 5% on the reference case."*

**Scope of this run.** The MMS leg — the part executable from this
repository alone — is run in full: spatial convergence of the coupled
energy/kinetics/gas-flux system on the Landau grid, spectral exactness
of the gas-flux operator, closed-form kinetics checks, and the
cancellation behavior of the blowing correction. The FIAT comparison
requires the external FIAT code (or its published reference-case data),
neither of which is in this repository; that leg is reported as
**PENDING**, and the task's own failure criterion — which is stated
against the FIAT case — remains unevaluated. The verdict below therefore
certifies the MMS leg only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.integrate

from aether.spectral import ChebyshevGrid
from aether.thermal import (
    CharringThermalSolver,
    LandauFrame,
    SurfaceEnergyBalance,
    SurfaceEnvironment,
    blowing_correction,
    demo_material,
)
from aether.thermal.kinetics import decomposition_rate
from aether.thermal.material import GAS_CONSTANT
from aether.verification.common import VerificationReport, write_csv
from aether.verification.mms_ablation import ManufacturedAblation

__all__ = ["run_v4"]

_N_SWEEP = (6, 8, 10, 12, 16, 20)
_T_FINAL = 2.0
_THICKNESS = 0.05
#: MMS acceptance: finest-grid error and the coarse-to-fine contraction.
_ERR_FLOOR = 1.0e-6
_MIN_CONTRACTION = 1.0e3


def _integrate_mms(n: int, mms: ManufacturedAblation) -> tuple[float, float, float]:
    grid = ChebyshevGrid(n, interval=(0.0, 1.0), max_derivative=2)
    solver = CharringThermalSolver(grid, mms.material, mms.frame)
    eta = grid.x
    g_t = mms.energy_source()
    g_rho = (mms.density_source(0), mms.density_source(1), mms.density_source(2))

    sol = scipy.integrate.solve_ivp(
        lambda t, y: solver.rhs(
            t,
            y,
            lambda tt, s: mms.recession_rate(tt),
            surface_rate=mms.surface_rate,
            back_face_rate=mms.back_face_rate,
            energy_source=g_t,
            density_sources=g_rho,
        ),
        (0.0, _T_FINAL),
        solver.pack(mms.initial_state(eta)),
        method="DOP853",
        rtol=1e-11,
        atol=1e-9,
    )
    if not sol.success:
        raise RuntimeError(f"MMS integration failed at N={n}: {sol.message}")
    end = solver.unpack(sol.y[:, -1])
    t_err = float(
        np.max(np.abs(end.temperature - mms.temperature(eta, _T_FINAL))) / mms.temperature_span
    )
    r_err = max(
        float(np.max(np.abs(end.partial_densities[i] - mms.partial_density(i, eta, _T_FINAL))))
        / (mms.material.components[i].virgin_density - mms.material.components[i].char_density)
        for i in range(3)
    )
    s_err = abs(end.recession - mms.recession(_T_FINAL))
    return t_err, r_err, s_err


def run_v4(output_dir: Path) -> VerificationReport:
    report = VerificationReport(
        task_id="V4",
        title="Ablation — manufactured solutions (MMS leg); FIAT case pending",
        criterion=(
            "recession disagreement > 5% on the FIAT reference case "
            "(unevaluated here — see scope); MMS leg: loss of spectral "
            "convergence on Eqs. (3.17)–(3.18)"
        ),
        passed=True,
    )
    mat = demo_material()
    frame = LandauFrame(total_thickness=_THICKNESS)
    mms = ManufacturedAblation(material=mat, frame=frame)

    # --- MMS convergence sweep -------------------------------------------
    rows_md: list[list[str]] = []
    rows_csv: list[list[object]] = []
    t_errs: list[float] = []
    for n in _N_SWEEP:
        t_err, r_err, s_err = _integrate_mms(n, mms)
        t_errs.append(t_err)
        rows_md.append([str(n), f"{t_err:.3e}", f"{r_err:.3e}", f"{s_err:.1e}"])
        rows_csv.append([n, t_err, r_err, s_err])

    contraction = t_errs[0] / max(t_errs[-1], np.finfo(np.float64).tiny)
    mms_ok = t_errs[-1] <= _ERR_FLOOR and contraction >= _MIN_CONTRACTION
    report.add_table(
        f"MMS convergence, coupled T/ρ/ṁ system to t = {_T_FINAL} s "
        f"(normalized max-norm errors at t_f)",
        ["N_T", "T error / ΔT", "ρ error / Δρ", "s error (m)"],
        rows_md,
    )
    report.add_section(
        "MMS acceptance",
        f"Error contracts by **{contraction:.1e}** from N = {_N_SWEEP[0]} to "
        f"N = {_N_SWEEP[-1]}, reaching **{t_errs[-1]:.1e}** against the "
        f"{_ERR_FLOOR:.0e} criterion (floor set by the 1e-11 time tolerance) → "
        f"{'**PASS**' if mms_ok else '**FAIL**'}. The decay is exponential in "
        "N_T until the time-integration floor — the spectral signature that the "
        "collocated operators discretize Eqs. (3.17)–(3.18) consistently, "
        "grid-velocity advection and pyrolysis sources included. The manufactured "
        "fields keep the degree-of-char clip and the kinetics extent clamp "
        "strictly inactive, so the sources are exact.",
    )
    write_csv(
        output_dir,
        "v4-mms-convergence",
        ["N_T", "t_err_rel", "rho_err_rel", "s_err_m"],
        rows_csv,
    )

    # --- supporting closed-form checks -----------------------------------
    # kinetics against exact isothermal solutions
    comp1, temp1 = mat.resin_a, 900.0
    k1 = comp1.pre_exponential * np.exp(-comp1.activation_energy / (GAS_CONSTANT * temp1))
    sol1 = scipy.integrate.solve_ivp(
        lambda _t, r: decomposition_rate(comp1, r, temp1),
        (0.0, 0.5 / k1),
        [comp1.virgin_density],
        rtol=1e-12,
        atol=1e-12,
    )
    exact1 = comp1.char_density + (comp1.virgin_density - comp1.char_density) * np.exp(-0.5)
    kin_err = abs(sol1.y[0, -1] - exact1) / (comp1.virgin_density - comp1.char_density)

    # gas-flux operator on a polynomial source: exact to rounding
    grid = ChebyshevGrid(10, interval=(0.0, 1.0), max_derivative=2)
    solver = CharringThermalSolver(grid, mat, frame)
    eta = grid.x
    mdot = solver.gas_flux(eta**2 - 0.3, 0.04)
    flux_err = float(np.max(np.abs(mdot - 0.04 * ((1.0 - eta**3) / 3.0 - 0.3 * (1.0 - eta)))))

    # blowing correction at collapse-inducing B'
    phi_tiny = float(blowing_correction(1e-17, 0.5))
    supporting_ok = kin_err < 1e-9 and flux_err < 1e-14 and abs(phi_tiny - 1.0) < 1e-12
    report.add_table(
        "Supporting closed-form checks",
        ["check", "measured", "criterion"],
        [
            ["first-order kinetics vs exact exponential", f"{kin_err:.1e}", "< 1e-9"],
            ["gas-flux operator, polynomial source", f"{flux_err:.1e}", "< 1e-14 (exact)"],
            ["blowing φ at B' = 1e-17 (naive form returns 0)", f"{phi_tiny:.15f}", "= 1 ± 1e-12"],
        ],
    )

    # --- SEB smoke: balanced root satisfies the balance ------------------
    env = SurfaceEnvironment(
        film_coefficient=0.08,
        recovery_enthalpy=1.2e7,
        radiative_flux=2.0e5,
        absorptivity=0.9,
        wall_enthalpy=lambda t_w: 1004.5 * t_w,
    )
    seb = SurfaceEnergyBalance(mat, env)
    t_wall = seb.solve_wall_temperature(0.01, 0.02, 5.0e4)
    seb_resid = abs(seb.residual(t_wall, 0.01, 0.02, 5.0e4))
    seb_ok = seb_resid < 1e-4 * env.film_coefficient * env.recovery_enthalpy
    report.add_section(
        "Surface energy balance",
        f"Brent solve of Eq. (3.19) with ablating mass fluxes returns "
        f"T_w = {t_wall:.1f} K with residual {seb_resid:.2e} W/m² "
        f"({'**PASS**' if seb_ok else '**FAIL**'}). The blowing correction enters "
        "through the log1p form, so the non-ablating limit is reached without "
        "cancellation.",
    )

    # --- FIAT leg ---------------------------------------------------------
    report.add_section(
        "FIAT reference comparison — PENDING",
        "The stated failure criterion (recession within 5% of a FIAT reference "
        "case) requires the external FIAT code or its published reference-case "
        "data, neither of which is available in this repository. The comparison "
        "harness accepts any tabulated (t, s, T(y)) reference once one is "
        "supplied; until then V4 is **partially complete** and is *not* counted "
        "as a finished verification task.\n\n"
        "`docs/FIAT-reference-data.md` specifies what a usable case must "
        "contain — the recession curve is not sufficient on its own, because "
        "a recession number without its wall boundary condition and material "
        "property set is not interpretable against a 5% criterion — and where "
        "such a case may be obtainable.",
    )

    report.passed = bool(mms_ok and supporting_ok and seb_ok)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run verification task V4 (MMS leg)")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    report = run_v4(args.output)
    path = report.write(args.output, "v4-thermal")
    print(f"V4 (MMS leg) {'PASS' if report.passed else 'FAIL'} -> {path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
