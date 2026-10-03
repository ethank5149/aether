"""V7 — dispersion statistics: convergence, bootstrap, normality.

Acceptance criterion: *"Convergence of CEP and R95 with N_MC; bootstrap
intervals; Henze–Zirkler normality. Failure criterion: CEP not
converging at the 1/√(2N_MC) rate."*

Impacts come from the generic entry-dispersion model (no vehicle data);
the convergence measurement splits one large batch into disjoint
sub-batches per sample size, so the empirical scatter of the CEP
estimator is measured directly rather than assumed.

Besides the verdict, the run writes the raw Monte Carlo draw to CSV for
hand analysis (floats rounded to 6 significant figures):

- ``v7-samples-inputs.csv`` — ``sample_id`` (1..32000) then one column per
  dispersed input in spec order: ``beta``, ``speed``, ``flight_path_deg``,
  ``azimuth_deg``, ``density_bias``, ``wind_x``, ``wind_y``.
- ``v7-samples-impacts.csv`` — ``sample_id``, ``downrange_m``,
  ``crossrange_m``, ``miss_m`` (distance from the batch mean impact
  point), ``inside_r95`` (1/0 from the same R95 ellipse test the verdict
  uses). Rows share the ``sample_id`` keys of the inputs file.
- ``v7-dispersion-spec.csv`` — ``parameter``, ``nominal``, ``sigma``,
  ``lower``, ``upper``, ``units`` (one row per dispersion spec; bounds
  blank where infinite, units blank where the model states none).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from aether.batch import EntryDispersionModel
from aether.batch.containment import containment_radius
from aether.batch.dispersion import summarize_dispersion
from aether.verification.common import VerificationReport, write_csv

__all__ = ["run_v7"]

_N_TOTAL = 32_000
_SEED = 20260731
_SUB_SIZES = (250, 500, 1000, 2000, 4000)
_SLOPE_BAND = (-0.75, -0.25)
_RATIO_BAND = (0.5, 2.0)

#: Units for each dispersion spec, read from EntryDispersionModel's
#: docstrings; blank where the docstring states none (density bias is a
#: dimensionless multiplier, the wind components carry no stated unit).
_SPEC_UNITS = {
    "beta": "kg/m²",
    "speed": "m/s",
    "flight_path_deg": "deg",
    "azimuth_deg": "deg",
    "density_bias": "",
    "wind_x": "",
    "wind_y": "",
}


def _sig6(value: float) -> float:
    """Round to 6 significant figures (deterministic across runs)."""
    x = float(value)
    if x == 0.0 or not math.isfinite(x):
        return 0.0 if x == 0.0 else x
    digits = 5 - math.floor(math.log10(abs(x)))
    return round(x, digits) + 0.0  # + 0.0 normalises any -0.0


def _bound(value: float) -> object:
    """Spec bound for CSV: 6 sig figs, or blank for an infinite bound."""
    return _sig6(value) if math.isfinite(value) else ""


def _write_sample_csvs(
    output_dir: Path,
    model: EntryDispersionModel,
    params: dict[str, np.ndarray],
    impacts: np.ndarray,
    centered: np.ndarray,
    inside_r95: np.ndarray,
) -> None:
    """Export the raw Monte Carlo draw for hand analysis.

    Inputs and impacts share one ``sample_id`` (1-based) index, so a row
    in each file refers to the same replicate of the single 32k draw.
    """
    specs = model.specs()
    names = [s.name for s in specs]
    sample_ids = range(1, len(impacts) + 1)
    miss = np.linalg.norm(centered, axis=1)

    write_csv(
        output_dir,
        "v7-samples-inputs",
        ["sample_id", *names],
        [[sid, *(_sig6(params[n][i]) for n in names)] for i, sid in enumerate(sample_ids)],
    )
    write_csv(
        output_dir,
        "v7-samples-impacts",
        ["sample_id", "downrange_m", "crossrange_m", "miss_m", "inside_r95"],
        [
            [sid, _sig6(impacts[i, 0]), _sig6(impacts[i, 1]), _sig6(miss[i]), int(inside_r95[i])]
            for i, sid in enumerate(sample_ids)
        ],
    )
    write_csv(
        output_dir,
        "v7-dispersion-spec",
        ["parameter", "nominal", "sigma", "lower", "upper", "units"],
        [
            [s.name, _sig6(s.nominal), _sig6(s.sigma), _bound(s.lower), _bound(s.upper),
             _SPEC_UNITS.get(s.name, "")]
            for s in specs
        ],
    )


def run_v7(output_dir: Path) -> VerificationReport:
    report = VerificationReport(
        task_id="V7",
        title="Dispersion statistics — CEP/R95 convergence, bootstrap, normality",
        criterion="CEP not converging at the 1/sqrt(2 N_MC) rate",
        passed=True,
    )
    model = EntryDispersionModel()
    impacts, params = model.fly(_N_TOTAL, seed=_SEED, return_params=True)

    # --- full-batch summary ----------------------------------------------
    rep = summarize_dispersion(impacts, bootstrap_samples=2000, seed=1)
    centered = impacts - rep.mean
    proj = centered @ rep.axes
    inside_r95 = (
        (proj[:, 0] / rep.r95_semi_axes[0]) ** 2
        + (proj[:, 1] / rep.r95_semi_axes[1]) ** 2
        <= 1.0
    )
    containment = float(np.mean(inside_r95))
    _write_sample_csvs(output_dir, model, params, impacts, centered, inside_r95)
    report.add_table(
        f"Full-batch summary (N_MC = {_N_TOTAL:,}, generic entry model)",
        ["metric", "value", "95% bootstrap CI"],
        [
            ["CEP", f"{rep.cep:.1f} m ({rep.cep_method})",
             f"[{rep.cep_ci[0]:.1f}, {rep.cep_ci[1]:.1f}]"],
            ["R95 (scalar, = a95)", f"{rep.r95:.1f} m",
             f"[{rep.r95_ci[0]:.1f}, {rep.r95_ci[1]:.1f}]"],
            ["σ1, σ2", f"{rep.sigma[0]:.1f}, {rep.sigma[1]:.1f} m", "—"],
            ["aspect σ2/σ1", f"{rep.aspect_ratio:.3f}", "—"],
            ["per-σ RSE bound 1/sqrt(2N)", f"{rep.relative_standard_error:.4f}", "—"],
            ["empirical containment of R95 ellipse", f"{containment:.4f}", "target 0.95"],
            ["Henze–Zirkler", f"HZ = {rep.hz_statistic:.3f}, p = {rep.hz_p_value:.3g}", "—"],
        ],
    )
    containment_ok = abs(containment - 0.95) < 0.01
    report.add_section(
        "Reading",
        f"The footprint is downrange-elongated (aspect {rep.aspect_ratio:.3f}), "
        "inside the band Eq. (6.3)'s linear approximation was stated for — though "
        "the CEP reported here is the exact elliptical integral regardless, so the "
        "band now only labels comparability with the classical route. The "
        f"R95 ellipse empirically contains {containment:.1%} of impacts "
        f"({'consistent with' if containment_ok else 'off'} the 95% design). The "
        f"Henze–Zirkler p-value of {rep.hz_p_value:.3g} "
        + (
            "does not reject bivariate normality for this batch."
            if rep.hz_p_value > 0.05
            else "**rejects** bivariate normality — the drag nonlinearity skews the "
            "footprint, exactly the situation Remark 11 warns the elliptical summary "
            "about; the metrics above should be read with that caveat."
        ),
    )

    # --- CEP convergence vs the 1/sqrt(2N) rate ---------------------------
    rows_md: list[list[str]] = []
    rows_csv: list[list[object]] = []
    rse_emp: list[float] = []
    rse_pred: list[float] = []
    for n_sub in _SUB_SIZES:
        n_groups = _N_TOTAL // n_sub
        ceps = []
        for g in range(n_groups):
            block = impacts[g * n_sub : (g + 1) * n_sub]
            sig = np.sqrt(
                np.maximum(np.linalg.eigvalsh(np.cov(block, rowvar=False)), 0.0)
            )[::-1]
            ceps.append(0.5887 * (sig[0] + sig[1]))
        ceps_arr = np.asarray(ceps)
        rse = float(np.std(ceps_arr, ddof=1) / np.mean(ceps_arr))
        predicted = float(1.0 / np.sqrt(2.0 * n_sub))
        rse_emp.append(rse)
        rse_pred.append(predicted)
        rows_md.append(
            [str(n_sub), str(n_groups), f"{np.mean(ceps_arr):.1f}", f"{rse:.4f}",
             f"{predicted:.4f}", f"{rse / predicted:.2f}"]
        )
        rows_csv.append([n_sub, n_groups, float(np.mean(ceps_arr)), rse, predicted])

    slope = float(np.polyfit(np.log(_SUB_SIZES), np.log(rse_emp), 1)[0])
    ratios = [e / p for e, p in zip(rse_emp, rse_pred, strict=True)]
    slope_ok = _SLOPE_BAND[0] <= slope <= _SLOPE_BAND[1]
    ratio_ok = all(_RATIO_BAND[0] <= r <= _RATIO_BAND[1] for r in ratios)
    report.add_table(
        "CEP sampling error vs N_MC (disjoint sub-batches of one 32k draw)",
        ["N_MC", "sub-batches", "mean CEP (m)", "empirical RSE", "1/sqrt(2N)", "ratio"],
        rows_md,
    )
    report.add_section(
        "Convergence acceptance",
        f"Fitted log–log slope of the empirical CEP relative standard error versus "
        f"N_MC: **{slope:.3f}** (criterion band [{_SLOPE_BAND[0]}, {_SLOPE_BAND[1]}], "
        f"the −1/2 of §6.3), with every empirical RSE within a factor "
        f"{_RATIO_BAND[1]:g} of the 1/sqrt(2N) prediction → "
        f"{'**PASS**' if slope_ok and ratio_ok else '**FAIL**'}.",
    )
    write_csv(
        output_dir,
        "v7-cep-convergence",
        ["N_MC", "n_groups", "mean_cep", "rse_empirical", "rse_predicted"],
        rows_csv,
    )

    # --- elongated footprint: exact, where the classical route could not be
    elongated = EntryDispersionModel(azimuth_sigma_deg=0.01, wind_sigma=0.5)
    pts_e = elongated.fly(4000, seed=_SEED + 1)
    rep_e = summarize_dispersion(pts_e, bootstrap_samples=500, seed=2)
    exact_e = float(containment_radius(0.5, float(rep_e.sigma[0]), float(rep_e.sigma[1])))
    linear_e = 0.5887 * (rep_e.sigma[0] + rep_e.sigma[1])
    fallback_ok = (
        rep_e.aspect_ratio < 0.25
        and rep_e.cep_method.startswith("exact-elliptical")
        and "outside Eq. 6.4 band" in rep_e.cep_method
        and abs(rep_e.cep - exact_e) < 1e-9 * exact_e
    )
    report.add_section(
        "Elongated footprint — exact rather than approximated (supersedes Remark 10)",
        f"Suppressing crossrange dispersions produces aspect ratio "
        f"{rep_e.aspect_ratio:.3f}, outside the validity band of Eq. (6.4). "
        f"The classical route falls back to the median radius here; the summary uses "
        f"the exact elliptical containment integral instead, which needs no "
        f"fallback at all (CEP = {rep_e.cep:.1f} m, method "
        f"'{rep_e.cep_method}'), and matches "
        f"`containment_radius` to {abs(rep_e.cep - exact_e):.2e} m → "
        f"{'**PASS**' if fallback_ok else '**FAIL**'}. The linear formula of "
        f"Eq. (6.3) would have claimed {linear_e:.1f} m, "
        f"{100 * (linear_e - exact_e) / exact_e:+.1f}% out. That integral is itself "
        f"verified against 126 published values in Siouris Table 5.2, to within "
        f"one unit in the last printed place.",
    )

    report.passed = bool(slope_ok and ratio_ok and containment_ok and fallback_ok)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run verification task V7")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    report = run_v7(args.output)
    path = report.write(args.output, "v7-dispersion")
    print(f"V7 {'PASS' if report.passed else 'FAIL'} -> {path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
