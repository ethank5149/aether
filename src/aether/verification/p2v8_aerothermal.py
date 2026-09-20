"""II-V8 — aerothermal correlations (implementation-verification leg).

Acceptance criterion: *"Stagnation heating against Fay–Riddell reference cases;
recession against a FIAT comparison. Failure criterion: heating
disagreement > 5% on reference conditions."*

**Reference data.** The published Tauber–Sutton velocity function is
now transcribed from the archived source in ``reference/`` and verified
against it, so the radiative leg is checked against *published* values
rather than a surrogate. The Fay–Riddell reference-case comparison still
needs tabulated equilibrium-air properties at the published reference
states, which the repository does not carry; that leg remains pending.

**Scope of this run.** The executable leg verifies the *implementation*
of the correlations: exact scaling structure of Fay–Riddell through the
modified-Newtonian velocity gradient, the quantified Lewis-exponent
sensitivity, Lees continuity at the stagnation-region boundary, the
opposite-sign radius trade between convective and radiative heating
(with the interior blunting optimum demonstrated), and the leading-edge
recession balance limits. The comparison against *published* Fay–Riddell
reference cases — the stated 5% criterion — requires transcribed
reference data (tabulated conditions with equilibrium-air properties),
which this repository does not yet carry; that leg is **PENDING**, as is
the FIAT recession leg shared with I-V4. No number below is claimed as a
validation against flight or ground-test data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TypedDict

import numpy as np

from aether.aerothermal import (
    EARTH_VELOCITY_FUNCTION,
    TAUBER_SUTTON_PROVENANCE,
    earth_radiative_heat_flux,
    earth_radiative_heating_exponent,
    earth_velocity_function,
    fay_riddell,
    lees_distribution,
    newtonian_velocity_gradient,
    stefan_recession_rate,
    sutton_graves,
)
from aether.aerothermal.stagnation import (
    FAY_RIDDELL_COEFFICIENT_EXACT_094,
    FAY_RIDDELL_COEFFICIENT_FROM_NUSSELT,
    FAY_RIDDELL_COEFFICIENT_LITERATURE,
    FAY_RIDDELL_COEFFICIENT_SOURCE,
    FAY_RIDDELL_FACTOR_PR071,
    FAY_RIDDELL_NUSSELT_COEFFICIENT,
    LEWIS_EXPONENT_EQUILIBRIUM,
    LEWIS_EXPONENT_FROZEN_CATALYTIC,
    WallCatalycity,
)
from aether.thermal.surface import STEFAN_BOLTZMANN
from aether.verification.common import VerificationReport, write_csv

__all__ = ["run_p2v8"]


class _FayRiddellState(TypedDict):
    """The flow state shared by every Fay-Riddell check below.

    A ``TypedDict`` rather than a plain ``dict[str, float]`` so that
    ``**_FR_BASE`` stays checkable: with a bare float-valued dict, mypy must
    assume the unpacking could supply *any* keyword, including the
    ``catalycity`` enum, and rejects the call.
    """

    edge_density: float
    edge_viscosity: float
    wall_density: float
    wall_viscosity: float
    total_enthalpy_edge: float
    wall_enthalpy: float
    dissociation_enthalpy: float


_FR_BASE: _FayRiddellState = {
    "edge_density": 0.05,
    "edge_viscosity": 6.0e-5,
    "wall_density": 0.3,
    "wall_viscosity": 4.0e-5,
    "total_enthalpy_edge": 2.0e7,
    "wall_enthalpy": 1.5e6,
    "dissociation_enthalpy": 5.0e6,
}


def run_p2v8(output_dir: Path) -> VerificationReport:
    report = VerificationReport(
        task_id="II-V8",
        title="Aerothermal correlations — implementation verification leg",
        criterion=(
            "heating disagreement > 5% on published reference conditions "
            "(unevaluated here — see scope); implementation leg: any scaling "
            "or continuity property failing its closed form"
        ),
        passed=True,
    )

    # --- Fay–Riddell structure --------------------------------------------
    checks: list[tuple[str, float, float, bool]] = []

    # R_eff^{-1/2} through the velocity gradient
    p_s, p_inf, rho_s = 5.0e4, 100.0, 0.05
    q_at = {}
    for r_eff in (0.25, 1.0):
        dudx = newtonian_velocity_gradient(r_eff, p_s, p_inf, rho_s)
        q_at[r_eff] = float(fay_riddell(velocity_gradient=float(dudx), **_FR_BASE))
    ratio = q_at[0.25] / q_at[1.0]
    checks.append(("q_conv ratio for R_eff 0.25 vs 1.0 (expect 4^{1/2} = 2)", ratio, 2.0,
                   abs(ratio - 2.0) < 1e-12))

    # driving-potential linearity
    dudx0 = float(newtonian_velocity_gradient(0.5, p_s, p_inf, rho_s))
    q1 = float(fay_riddell(velocity_gradient=dudx0, **_FR_BASE))
    args: _FayRiddellState = {
        **_FR_BASE,
        "wall_enthalpy": 0.5 * (_FR_BASE["total_enthalpy_edge"] + _FR_BASE["wall_enthalpy"]),
    }
    q2 = float(fay_riddell(velocity_gradient=dudx0, **args))
    expect = (_FR_BASE["total_enthalpy_edge"] - args["wall_enthalpy"]) / (
        _FR_BASE["total_enthalpy_edge"] - _FR_BASE["wall_enthalpy"]
    )
    checks.append(("enthalpy-potential linearity ratio", q2 / q1, expect,
                   abs(q2 / q1 - expect) < 1e-12))

    # Lewis-exponent sensitivity (the paper: "differs by several percent")
    q_eq = float(fay_riddell(velocity_gradient=dudx0, **_FR_BASE,
                             lewis_exponent=LEWIS_EXPONENT_EQUILIBRIUM))
    q_fr = float(fay_riddell(velocity_gradient=dudx0, **_FR_BASE,
                             lewis_exponent=LEWIS_EXPONENT_FROZEN_CATALYTIC))
    lewis_delta = q_fr / q_eq - 1.0
    checks.append(("frozen/catalytic vs equilibrium bracket shift (expect ~1%)",
                   lewis_delta, 0.011, 0.002 < lewis_delta < 0.1))

    # Wall catalycity — Fay & Riddell's third correlation, via Anderson
    # Eq. (17.91). Structurally distinct from the other two: no choice of
    # Lewis exponent reaches it, so this check exists to confirm the branch
    # is real rather than a relabelling.
    q_noncat = float(fay_riddell(velocity_gradient=dudx0, **_FR_BASE,
                                 catalycity=WallCatalycity.FROZEN_NONCATALYTIC))
    catalytic_gain = q_fr / q_noncat
    fraction = _FR_BASE["dissociation_enthalpy"] / _FR_BASE["total_enthalpy_edge"]
    expected_gain = (1.0 + (1.4**LEWIS_EXPONENT_FROZEN_CATALYTIC - 1.0) * fraction) / (
        1.0 - fraction
    )
    checks.append(("catalytic/noncatalytic heating ratio at h_D/h_0e = 0.25",
                   catalytic_gain, expected_gain,
                   abs(catalytic_gain - expected_gain) < 1e-12))

    # At h_D/h_0e = 0.6 the ratio must exceed the "more than a factor of
    # two" Anderson reports from Fay & Riddell Fig. 17.5.
    hot: _FayRiddellState = {**_FR_BASE, "dissociation_enthalpy": 1.2e7}
    q_fr_hot = float(fay_riddell(velocity_gradient=dudx0, **hot,
                                 catalycity=WallCatalycity.FROZEN_CATALYTIC))
    q_nc_hot = float(fay_riddell(velocity_gradient=dudx0, **hot,
                                 catalycity=WallCatalycity.FROZEN_NONCATALYTIC))
    hot_ratio = q_fr_hot / q_nc_hot
    checks.append(("catalytic/noncatalytic at h_D/h_0e = 0.6 (Anderson: > 2)",
                   hot_ratio, 2.854, hot_ratio > 2.0))

    # With nothing dissociated, catalycity cannot matter: all three
    # correlations must coincide exactly.
    cold: _FayRiddellState = {**_FR_BASE, "dissociation_enthalpy": 0.0}
    q_cold = [
        float(fay_riddell(velocity_gradient=dudx0, **cold, catalycity=case))
        for case in WallCatalycity
    ]
    cold_spread = max(q_cold) / min(q_cold) - 1.0
    checks.append(("catalycity spread at h_D = 0 (expect exactly 0)",
                   cold_spread, 0.0, cold_spread == 0.0))

    # Lees continuity at x = R_eff
    r_eff = 0.4
    inner = float(lees_distribution(1.0e6, 0.8, r_eff * (1 - 1e-9), r_eff))
    outer = float(lees_distribution(1.0e6, 0.8, r_eff * (1 + 1e-9), r_eff))
    checks.append(("Lees jump across x = R_eff (expect 0)",
                   abs(inner - outer) / inner, 0.0, abs(inner - outer) / inner < 1e-6))

    # leading-edge recession: exactly balanced wall must not recede
    t_w, eps = 2000.0, 0.85
    q_bal = eps * STEFAN_BOLTZMANN * t_w**4
    sdot0 = float(stefan_recession_rate(q_bal, t_w, eps, 0.0, 1900.0, 2.0e7))
    checks.append(("recession at radiative-equilibrium wall (expect 0)", sdot0, 0.0,
                   sdot0 == 0.0))

    rows = [[name, f"{got:.6g}", f"{want:.6g}", "yes" if ok else "NO"]
            for name, got, want, ok in checks]
    all_ok = all(ok for _, _, _, ok in checks)
    report.add_table(
        "Closed-form structure checks",
        ["check", "measured", "expected", "pass"],
        rows,
    )
    write_csv(
        output_dir,
        "p2v8-structure-checks",
        ["check", "measured", "expected", "pass"],
        [[name, got, want, ok] for name, got, want, ok in checks],
    )

    # --- the blunting trade, on the published correlation ------------------
    rho_inf, v_inf = 3.0e-4, 12000.0
    radii = np.linspace(0.3, 3.0, 300)  # the source's stated r_n envelope
    q_conv = np.asarray(sutton_graves(rho_inf, radii, v_inf))
    q_rad = np.asarray(
        earth_radiative_heat_flux(radii, np.full_like(radii, rho_inf),
                                  np.full_like(radii, v_inf))
    )
    total = q_conv + q_rad
    i_min = int(np.argmin(total))
    trade_ok = 0 < i_min < radii.size - 1
    trade_exponent = float(
        earth_radiative_heating_exponent(v_inf, rho_inf, radii[i_min])
    )
    report.add_section(
        "Opposite-sign radius trade — on published data",
        f"With convective heating falling as R_eff^(−1/2) and the published "
        f"Tauber–Sutton correlation rising as R_eff^(+{trade_exponent:.3f}), "
        f"the total heating has an **interior optimum at R_eff ≈ "
        f"{radii[i_min]:.2f} m** at V = {v_inf:g} m/s, ρ = {rho_inf:g} kg/m³ — "
        f"inside the source's own 0.3–3 m nose-radius envelope → "
        f"{'**PASS**' if trade_ok else '**FAIL**'}. At this condition radiation "
        f"({q_rad[i_min] / 1e6:.2f} MW/m²) exceeds convection "
        f"({q_conv[i_min] / 1e6:.2f} MW/m²), which is the regime Tauber and "
        f"Sutton wrote the correlation for. A framework modelling only "
        "convection would drive R_eff to the top of this range — the "
        "over-blunting bias the paper warns of, now demonstrated against "
        "published data rather than a surrogate.",
    )
    write_csv(
        output_dir,
        "p2v8-blunting-trade",
        ["R_eff", "q_conv", "q_rad_published", "q_total"],
        [[float(r), float(c), float(g), float(t)]
         for r, c, g, t in zip(radii, q_conv, q_rad, total, strict=True)],
    )

    # --- published Tauber-Sutton data --------------------------------------
    table_rows = [
        [f"{v:g}", f"{f:g}", f"{float(earth_velocity_function(v)):g}"]
        for v, f in EARTH_VELOCITY_FUNCTION[:4] + EARTH_VELOCITY_FUNCTION[-3:]
    ]
    table_exact = all(
        float(earth_velocity_function(v)) == f for v, f in EARTH_VELOCITY_FUNCTION
    )
    report.add_table(
        "Published velocity function f_E(V), Table 1 (spot check of the transcription)",
        ["V (m/s)", "published f_E", "interpolant at the node"],
        table_rows,
    )
    rho_ref, v_ref, rn_ref = 3.0e-4, 11000.0, 1.0
    exponent = float(earth_radiative_heating_exponent(v_ref, rho_ref, rn_ref))
    q_rad_ref = float(earth_radiative_heat_flux(rn_ref, rho_ref, v_ref))
    q_conv_ref = float(sutton_graves(rho_ref, rn_ref, v_ref))
    envelope_guarded = True
    for bad_radius, bad_velocity in ((1.0, 9500.0), (10.0, v_ref)):
        try:
            earth_radiative_heat_flux(bad_radius, rho_ref, bad_velocity)
            envelope_guarded = False
        except ValueError:
            pass
    published_ok = table_exact and envelope_guarded and 0.0 < exponent < 1.0
    report.add_table(
        f"Published correlation at V = {v_ref:g} m/s, ρ = {rho_ref:g} kg/m³, "
        f"r_n = {rn_ref:g} m",
        ["quantity", "value", "note"],
        [
            ["nose-radius exponent a", f"{exponent:.4f}",
             "source requires a < 1; often summarised as ≈ 1.0"],
            ["radiative flux (Eqs. 1–2)", f"{q_rad_ref / 1e6:.4f} MW/m²",
             "converted from the source's W/cm²"],
            ["convective flux (Sutton–Graves)", f"{q_conv_ref / 1e6:.4f} MW/m²",
             "for scale only"],
            ["transcription exact at every node", "yes" if table_exact else "NO", "—"],
            ["validity envelope enforced", "yes" if envelope_guarded else "NO",
             "10–16 km/s, 6.66e-5–6.31e-4 kg/m³, r_n 0.3–3 m"],
        ],
    )
    report.add_section(
        "Published radiative data — now verified",
        f"The velocity function is transcribed from {TAUBER_SUTTON_PROVENANCE}. "
        f"Every tabulated node is reproduced exactly by the interpolant "
        f"({'confirmed' if table_exact else '**MISMATCH**'}), linear "
        f"interpolation being what the source prescribes, and the stated "
        f"validity envelope is enforced rather than silently extrapolated → "
        f"{'**PASS**' if published_ok else '**FAIL**'}. At the sample condition "
        f"radiative and convective heating are comparable "
        f"({q_rad_ref / q_conv_ref:.2f}×), which is the regime the paper was written "
        f"for.\n\n**Two discrepancies against the usual summary, both recorded "
        f"rather than reconciled.** First, the usual summary gives the nose-radius "
        f"exponent as *a ≈ 1.0*; the source makes it a function of velocity and "
        f"density, *a = 1.072×10⁶ V^(−1.88) ρ^(−0.325)*, and states explicitly "
        f"that **a < 1 must always be met**. It evaluates to {exponent:.3f} here "
        f"— the qualitative claim that blunting increases radiative heating "
        f"survives, but the exponent does not. Second, the source returns "
        f"**W/cm²**, so an SI framework needs a 10⁴ conversion; omitting it is a "
        f"four-order-of-magnitude error that still resembles a heat flux. The "
        f"r_n-dependent conditional clauses of Eq. (2) — a ≤ 0.6 on 1 ≤ r_n ≤ 2, "
        f"a ≤ 0.5 on 2 < r_n ≤ 3 — were unreadable in the archived scan and were "
        f"supplied separately from the published text; they are implemented as "
        f"caps with the band edges exactly as printed (first band closed, second "
        f"half-open) rather than guessed.",
    )

    # --- Fay–Riddell primary source ----------------------------------------
    q_of = {
        name: float(fay_riddell(velocity_gradient=dudx0, **_FR_BASE, coefficient=c))
        for name, c in (
            ("nusselt", FAY_RIDDELL_COEFFICIENT_FROM_NUSSELT),
            ("printed", FAY_RIDDELL_COEFFICIENT_EXACT_094),
            ("footnote", FAY_RIDDELL_COEFFICIENT_SOURCE),
            ("literature", FAY_RIDDELL_COEFFICIENT_LITERATURE),
        )
    }
    spread = max(q_of.values()) / min(q_of.values()) - 1.0
    band = (
        FAY_RIDDELL_COEFFICIENT_FROM_NUSSELT / FAY_RIDDELL_COEFFICIENT_SOURCE - 1.0
    )
    report.add_table(
        "Fay–Riddell leading constant, back-substituted from the original paper",
        ["source statement", "back-substitution", "constant", "q_s here"],
        [
            ["Eq. (58)/(62), factor "
             f"{FAY_RIDDELL_NUSSELT_COEFFICIENT}", "0.67·Pr^(−0.4)",
             f"{FAY_RIDDELL_COEFFICIENT_FROM_NUSSELT:.4f}",
             f"{q_of['nusselt'] / 1e6:.4f} MW/m²"],
            [f"Eq. (63), factor {FAY_RIDDELL_FACTOR_PR071}", "0.94·Pr^(+0.6)",
             f"{FAY_RIDDELL_COEFFICIENT_EXACT_094:.4f}",
             f"{q_of['printed'] / 1e6:.4f} MW/m²"],
            ["footnote to Eq. (63)", "as printed",
             f"{FAY_RIDDELL_COEFFICIENT_SOURCE:.4f}",
             f"{q_of['footnote'] / 1e6:.4f} MW/m²"],
            ["*not in the source*", "literature / this framework",
             f"{FAY_RIDDELL_COEFFICIENT_LITERATURE:.4f}",
             f"{q_of['literature'] / 1e6:.4f} MW/m²"],
        ],
        "**Exponents: derived, not assumed.** The archived scan of Eq. (63) is "
        "too degraded to read its exponents directly, but they are "
        "recoverable. Eq. (45) gives q = (Nu/√Re)·√(ρ_w μ_w (du_e/dx)_s)·"
        "(h_s − h_w)/Pr, and Eq. (58)/(62) correlates Nu/√Re = "
        "0.67·(ρ_s μ_s / ρ_w μ_w)^0.4·{1 + (Le^0.52 − 1) h_D/h_s} — a *ratio*, "
        "because the parameter 'was found to depend only upon the total "
        "variation in ρμ across the boundary layer'. Substituting gives 0.4 "
        "from the ratio and a residual 0.1 from the √(ρ_w μ_w), i.e. exactly "
        "(ρ_e μ_e)^0.4 (ρ_w μ_w)^0.1. The Lewis exponents are confirmed "
        "directly: 0.52 equilibrium (Eq. 60/62), 0.63 frozen with a fully "
        "catalytic wall (Eq. 65, corroborated by Fig. 4).\n\n"
        "**The constant is a different story.** The same substitution shows "
        "every constant descending from one fitted number at two significant "
        f"figures — the {FAY_RIDDELL_NUSSELT_COEFFICIENT} of Eq. (58)/(62). "
        f"The printed {FAY_RIDDELL_FACTOR_PR071} of Eq. (63) is "
        f"{FAY_RIDDELL_NUSSELT_COEFFICIENT}/0.71 = "
        f"{FAY_RIDDELL_NUSSELT_COEFFICIENT / 0.71:.4f} rounded, and the "
        "footnote's Pr^(−0.6) is the correlation's own Pr^(+0.4) against the "
        "Pr^(−1) of Eq. (45). Backing the coefficient out three ways spans "
        f"{band * 100:.1f}%, "
        f"or {spread * 100:.2f}% in flux. The widely quoted 0.763 occurs "
        "**nowhere in the source**, but it lies inside that band and is as "
        "defensible as any of them. This is a provenance finding, not an "
        "accuracy finding: no reading supports a third significant figure, "
        "all four constants are named and selectable, and none is presented "
        "as *the* source value.",
    )

    # --- pending legs -------------------------------------------------------
    report.add_section(
        "Fay–Riddell reference cases and FIAT — PENDING",
        "The radiative half of this task is closed against published data, and "
        "the convective **coefficients** are now checked against the original "
        "Fay & Riddell paper (previous two sections). What remains is the "
        "convective **reference case**: the 5% criterion is stated against "
        "published Fay–Riddell reference conditions, and those need transcribed "
        "equilibrium-air properties (ρ_e μ_e, ρ_w μ_w, h_D at the reference "
        "states). The source presents its numerical results in figures (Figs. "
        "2, 3, 4, 7, 8), not tables, so no reference q values are extractable "
        "from it at the precision a 5% criterion requires. The FIAT recession "
        "comparison shares the pending status of I-V4. Neither dataset is in "
        "this repository, and no synthetic stand-in is presented as either. "
        "II-V8 is therefore **partially complete**: implementation verified, "
        "radiative reference verified, convective coefficients verified, "
        "convective reference case pending tabulated data.",
    )

    report.passed = bool(all_ok and trade_ok and published_ok)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run verification task II-V8 (implementation leg)"
    )
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    report = run_p2v8(args.output)
    path = report.write(args.output, "p2v8-aerothermal")
    print(f"II-V8 (implementation leg) {'PASS' if report.passed else 'FAIL'} -> {path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
