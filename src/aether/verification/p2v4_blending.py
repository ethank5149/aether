"""II-V4 — aerodynamic blending: integrated loads and trim vs blend width.

Acceptance criterion: *"Integrated normal force and pitching moment versus
:math:`\\delta_{\\mathrm{blend}}`; sensitivity of trim solution. Failure
criterion: trim incidence shifting by more than 0.1° across the
blend-width sweep."*

The blend exists because the windward Newtonian and leeward
Prandtl–Meyer branches are :math:`C^0` but not :math:`C^1` at
:math:`\\delta_c = 0` — a genuine violation of the
smoothness the framework otherwise maintains, occurring exactly at the
shoulder line. The blend is a numerical expedient, so the question this
task settles is whether it is *load-neutral*: if integrated forces and
the trim point move materially with a parameter that carries no physics,
the expedient has become a modelling choice.

Geometry is the generic cambered lifting body of
:mod:`aether.aerodynamics.panels` — chosen because its shoulder line
sweeps through :math:`\\delta_c = 0`, so panels genuinely change branch
as incidence varies. It corresponds to no vehicle.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from aether.aerodynamics import (
    blended_pressure_coefficient,
    curved_lifting_body,
    rayleigh_pitot_cp_max,
)
from aether.verification.common import VerificationReport, write_csv

__all__ = ["run_p2v4"]

_MACH = 8.0
_GAMMA = 1.4
_DYNAMIC_PRESSURE = 0.5 * 0.02 * (_MACH * 300.0) ** 2
#: Blend half-widths swept, in radians (0 is the unblended C0 closure).
_BLEND_SWEEP = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.04)
_TRIM_TOL_DEG = 0.1
_SAMPLE_ALPHA_DEG = 6.0


def run_p2v4(output_dir: Path) -> VerificationReport:
    report = VerificationReport(
        task_id="II-V4",
        title="Aerodynamic blending — integrated loads and trim sensitivity",
        criterion="trim incidence shifting by more than 0.1° across the blend-width sweep",
        passed=True,
    )
    body = curved_lifting_body()
    report.add_section(
        "Configuration",
        f"Generic cambered lifting body, {body.n_panels} panels, wetted area "
        f"{body.total_area:.2f} m²; M∞ = {_MACH:g}, γ = {_GAMMA}, "
        f"q∞ = {_DYNAMIC_PRESSURE / 1e3:.1f} kPa. Moment reference at mid-length. "
        f"C_p,max from the Rayleigh–Pitot relation is "
        f"{rayleigh_pitot_cp_max(_MACH, _GAMMA):.4f}.",
    )

    # --- loads and trim versus blend width --------------------------------
    rows_md: list[list[str]] = []
    rows_csv: list[list[object]] = []
    trims: list[float] = []
    alpha_fixed = np.deg2rad(_SAMPLE_ALPHA_DEG)
    for width in _BLEND_SWEEP:
        force, moment = body.loads(
            alpha_fixed, _MACH, _DYNAMIC_PRESSURE, gamma=_GAMMA, blend_width=width
        )
        trim = body.trim(_MACH, _DYNAMIC_PRESSURE, gamma=_GAMMA, blend_width=width)
        trims.append(trim.incidence)
        rows_md.append(
            [
                f"{width:.4f}" if width else "0 (unblended)",
                f"{np.rad2deg(width):.2f}",
                f"{force[2] / 1e3:.4f}",
                f"{moment[1] / 1e3:.4f}",
                f"{np.rad2deg(trim.incidence):.5f}",
            ]
        )
        rows_csv.append(
            [width, float(force[2]), float(moment[1]), float(np.rad2deg(trim.incidence))]
        )

    trim_shift_deg = float(np.rad2deg(max(trims) - min(trims)))
    trim_ok = trim_shift_deg <= _TRIM_TOL_DEG
    report.add_table(
        f"Integrated loads at α = {_SAMPLE_ALPHA_DEG:g}° and trim incidence "
        f"versus blend width",
        ["δ_blend (rad)", "δ_blend (deg)", "normal force (kN)",
         "pitching moment (kN·m)", "trim α (deg)"],
        rows_md,
    )
    report.add_section(
        "Acceptance",
        f"Trim incidence moves by **{trim_shift_deg:.4f}°** across the full "
        f"sweep — from the unblended closure to a half-width of "
        f"{np.rad2deg(_BLEND_SWEEP[-1]):.1f}° — against the criterion of "
        f"{_TRIM_TOL_DEG}° → {'**PASS**' if trim_ok else '**FAIL**'}. The blend "
        "is load-neutral at the widths that matter: it perturbs only panels "
        "within the seam band, whose pressure contributions are small and whose "
        "moment arms about the reference point largely cancel.",
    )
    write_csv(
        output_dir,
        "p2v4-blend-sweep",
        ["blend_width_rad", "normal_force_N", "pitching_moment_Nm", "trim_alpha_deg"],
        rows_csv,
    )

    # --- monotonicity and the small-width limit ---------------------------
    deltas = np.abs(np.rad2deg(np.asarray(trims) - trims[0]))
    monotone = bool(np.all(np.diff(deltas) >= -1e-9))
    report.add_section(
        "Convergence to the unblended limit",
        f"Trim offset from the unblended closure grows monotonically with the "
        f"blend width ({', '.join(f'{d:.5f}°' for d in deltas[1:])}) and scales "
        f"smoothly to zero as δ_blend → 0 "
        f"({'monotone' if monotone else '**non-monotone — investigate**'}). "
        "That is the behaviour a numerical expedient should show: the blended "
        "closure is a controlled perturbation of the physical one, not an "
        "independent model.",
    )

    # --- the seam the blend exists to fix ---------------------------------
    h = 1.0e-7
    cp_minus = float(blended_pressure_coefficient(-h, _MACH, _GAMMA, blend_width=0.0))
    cp_plus = float(blended_pressure_coefficient(+h, _MACH, _GAMMA, blend_width=0.0))
    slope_lee = (
        cp_minus - float(blended_pressure_coefficient(-3 * h, _MACH, _GAMMA, blend_width=0.0))
    ) / (2 * h)
    slope_wind = (
        float(blended_pressure_coefficient(3 * h, _MACH, _GAMMA, blend_width=0.0)) - cp_plus
    ) / (2 * h)
    width = 0.05
    step = width / 200.0

    def blended_slope(centre: float) -> float:
        return (
            float(blended_pressure_coefficient(centre + step, _MACH, _GAMMA, blend_width=width))
            - float(blended_pressure_coefficient(centre - step, _MACH, _GAMMA, blend_width=width))
        ) / (2 * step)

    seam_jump = abs(cp_plus - cp_minus)
    blended_mismatch = abs(blended_slope(step) - blended_slope(-step)) / max(
        abs(blended_slope(step)), 1e-30
    )
    seam_ok = seam_jump < 1e-6 and blended_mismatch < 0.05
    report.add_table(
        "The seam at δ_c = 0",
        ["quantity", "unblended", "blended (δ_blend = 0.05 rad)"],
        [
            ["C_p jump across the seam", f"{seam_jump:.2e}", "—"],
            ["dC_p/dδ, leeward side", f"{slope_lee:+.4f}", f"{blended_slope(-step):+.4f}"],
            ["dC_p/dδ, windward side", f"{slope_wind:+.4f}", f"{blended_slope(step):+.4f}"],
            [
                "relative slope mismatch",
                f"{abs(slope_lee - slope_wind) / max(abs(slope_lee), 1e-30):.2f}",
                f"{blended_mismatch:.2e}",
            ],
        ],
    )
    report.add_section(
        "Reading the seam",
        f"The unblended closure is continuous ({seam_jump:.1e}) but its slope "
        f"jumps: the Newtonian branch has zero slope at δ_c = 0 while the "
        f"expansion branch does not — the C⁰-but-not-C¹ defect Remark 2 "
        f"identifies. The C² smoothstep removes it, closing the slope mismatch "
        f"to {blended_mismatch:.1e} "
        f"({'**PASS**' if seam_ok else '**FAIL**'}). A cubic smoothstep would "
        "restore C¹ only; the quintic used here also matches second derivatives "
        "at the band edges, which is what keeps the closure inside the "
        "framework's smoothness claim.",
    )

    report.passed = bool(trim_ok and seam_ok)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run verification task II-V4")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    report = run_p2v4(args.output)
    path = report.write(args.output, "p2v4-blending")
    print(f"II-V4 {'PASS' if report.passed else 'FAIL'} -> {path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
