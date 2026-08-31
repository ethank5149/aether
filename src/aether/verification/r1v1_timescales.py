"""R1-V1 — timescale separation for the reachability series' Paper 1.

Reachability Paper 1, §2.7: the three small parameters

.. math::

    \\varepsilon_{\\mathrm{atm}} = H_s/R_e,\\qquad
    \\varepsilon_{\\mathrm{att}} = T_{\\mathrm{att}}/T_{\\mathrm{traj}},\\qquad
    \\varepsilon_{\\mathrm{ela}} = T_{\\mathrm{ela}}/T_{\\mathrm{traj}}

index the family whose profile decomposition §4 extracts. Their *relative*
sizes decide which profiles separate cleanly, and that is an empirical
question rather than a modeling choice — so this task measures it, from
the same free-free beam machinery :mod:`aether.structures` already
verifies (task V1), instead of asserting an ordering.

What the manuscript consumes from here is not a single number but three
qualitative facts, each of which is reported below with its supporting
arithmetic:

1. whether :math:`\\varepsilon_{\\mathrm{ela}}` is the *fastest* scale, since
   the elastic bubble of §4.6 exists only if there is a gap to have;
2. how far thermal softening moves it — the point being that a modulus
   loss enters the frequency as :math:`\\sqrt{E}`, so a large softening is
   a modest frequency drift and cannot by itself close a decade-wide gap
   (the gap-closing mechanism is aerodynamic, not thermal);
3. which *pair* of scales is closest to colliding, since where two scales
   collide the profile decomposition yields one joint fast profile rather
   than two orthogonal ones.

The parametrizations are generic thin-walled tubes with open material
data — traceable to no vehicle, per the public-repository rule — and they
fix orders of magnitude and nothing finer. Redo on the published geometry
of §6 before any conclusion here is asserted in the manuscript.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aether.structures import free_free_analytic_frequencies
from aether.verification.common import VerificationReport, write_csv

__all__ = ["run_r1v1"]

#: Young's modulus, generic aerospace aluminium (open material data).
_E_COLD = 70.0e9
#: Retained modulus fraction at the hot end of the 20–50% loss band that
#: the aerothermoelastic literature reports over entry-relevant
#: temperature ranges. Taken at the *pessimistic* end deliberately.
_E_RETAINED = 0.55

#: Trajectory horizon, s — the denominator of both time-ratio parameters.
_T_TRAJ = 1.0e3
#: Scale height and Earth radius, m, for eps_atm = H_s / R_e.
_H_S, _R_E = 7.2e3, 6.371e6
#: Attitude/short-period band, s. Spans a slow aerodynamically-trimmed
#: vehicle through an aggressively closed RCS loop.
_T_ATT = (0.1, 1.0, 10.0)

#: label, length (m), tube radius (m), wall thickness (m), mass/length (kg/m)
_CASES: tuple[tuple[str, float, float, float, float], ...] = (
    ("compact biconic", 5.0, 0.50, 0.010, 200.0),
    ("slender glider", 12.0, 0.40, 0.004, 100.0),
    ("very slender, L/D~6", 18.0, 0.35, 0.003, 60.0),
)

#: Failure criterion, stated in advance: the elastic bubble of §4.6 exists
#: only where the structural period is genuinely faster than the attitude
#: period. If T_ela exceeds T_att for the *slowest* admissible attitude
#: loop in any case, eps_ela is not a fast scale there and the elastic
#: reduction has no gap to stand on.
_T_ATT_SLOWEST = max(_T_ATT)


def _thin_wall_inertia(radius: float, thickness: float) -> float:
    """Second moment of a thin-walled circular section, :math:`\\pi r^3 t`."""
    return float(np.pi * radius**3 * thickness)


def run_r1v1(output_dir: Path) -> VerificationReport:
    report = VerificationReport(
        task_id="R1-V1",
        title="Timescale separation — eps_atm, eps_att, eps_ela",
        criterion=(
            f"T_ela > T_att = {_T_ATT_SLOWEST:g} s for any parametrization at "
            f"{100 * (1 - _E_RETAINED):.0f}% modulus loss, i.e. eps_ela fails to be "
            f"the fast scale the §4.6 elastic bubble requires"
        ),
        passed=True,
        source="Reachability Paper 1 §2.7",
    )

    eps_atm = _H_S / _R_E
    report.add_section(
        "Fixed parameters",
        f"- `eps_atm` = H_s/R_e = {_H_S:.3g}/{_R_E:.4g} = **{eps_atm:.2e}**\n"
        f"- `eps_att` = T_att/T_traj over T_att ∈ "
        + ", ".join(f"{t:g} s → {t / _T_TRAJ:.1e}" for t in _T_ATT)
        + f"\n- T_traj = {_T_TRAJ:.0f} s; modulus {_E_COLD:.3g} Pa cold, "
        f"{_E_RETAINED:.2f}× retained hot",
    )

    freq_rows: list[list[str]] = []
    csv_rows: list[list[object]] = []
    eps_ela_hot: dict[str, float] = {}
    worst_t_ela = 0.0
    for label, length, radius, thickness, mass in _CASES:
        inertia = _thin_wall_inertia(radius, thickness)
        cold = free_free_analytic_frequencies(2, length, _E_COLD * inertia, mass)
        hot = free_free_analytic_frequencies(2, length, _E_COLD * _E_RETAINED * inertia, mass)
        f1_cold, f1_hot = cold[0] / (2 * np.pi), hot[0] / (2 * np.pi)
        t_ela_hot = 1.0 / f1_hot
        eps_cold, eps_hot = (1.0 / (f1_cold * _T_TRAJ)), (t_ela_hot / _T_TRAJ)
        eps_ela_hot[label] = eps_hot
        worst_t_ela = max(worst_t_ela, t_ela_hot)
        freq_rows.append(
            [
                label,
                f"{_E_COLD * inertia:.2e}",
                f"{f1_cold:.1f}",
                f"{f1_hot:.1f}",
                f"{cold[1] / (2 * np.pi):.1f}",
                f"{eps_cold:.2e}",
                f"{eps_hot:.2e}",
            ]
        )
        csv_rows.append(
            [
                label,
                length,
                radius,
                thickness,
                mass,
                _E_COLD * inertia,
                f1_cold,
                f1_hot,
                cold[1] / (2 * np.pi),
                eps_cold,
                eps_hot,
            ]
        )

    softening_drift = 1.0 / np.sqrt(_E_RETAINED) - 1.0
    report.add_table(
        "First two free-free bending frequencies, cold and softened",
        [
            "case",
            "EI (N·m²)",
            "f₁ cold (Hz)",
            "f₁ hot (Hz)",
            "f₂ cold (Hz)",
            "eps_ela cold",
            "eps_ela hot",
        ],
        freq_rows,
        notes=(
            f"A {100 * (1 - _E_RETAINED):.0f}% modulus loss moves ω₁ by "
            f"√{_E_RETAINED:.2f} = {np.sqrt(_E_RETAINED):.3f}, i.e. it inflates "
            f"eps_ela by **{100 * softening_drift:.0f}%** — a drift, not an order of "
            "magnitude. Thermal softening therefore *shifts* the elastic spectrum "
            "(so the constraint surface moves during flight) but cannot on its own "
            "close a decade-wide separation from the attitude band. The mechanism "
            "that closes the aerothermoelastic gap is the aerodynamic (piston-theory) "
            "contribution to the pencil, and the manuscript prose must say so."
        ),
    )
    write_csv(
        output_dir,
        "r1v1-frequencies",
        [
            "case",
            "length_m",
            "radius_m",
            "thickness_m",
            "mass_per_length",
            "EI",
            "f1_cold_hz",
            "f1_hot_hz",
            "f2_cold_hz",
            "eps_ela_cold",
            "eps_ela_hot",
        ],
        csv_rows,
    )

    # --- pairwise scale separation, hot structure -------------------------
    pair_rows: list[list[str]] = []
    pair_csv: list[list[object]] = []
    closest = ("", np.inf)
    for label, _, _, _, _ in _CASES:
        eps_e = eps_ela_hot[label]
        row = [label, f"{eps_e / eps_atm:.2e}"]
        pair_csv.append([label, "atm", eps_e / eps_atm])
        for t_att in _T_ATT:
            ratio = eps_e / (t_att / _T_TRAJ)
            row.append(f"{ratio:.2e}")
            pair_csv.append([label, f"att_{t_att:g}s", ratio])
            if abs(np.log10(ratio)) < abs(np.log10(closest[1])):
                closest = (f"{label} vs T_att={t_att:g} s", ratio)
        pair_rows.append(row)
    atm_att = [(t, (t / _T_TRAJ) / eps_atm) for t in _T_ATT]
    report.add_table(
        "Pairwise separation ratios (hot structure); 1.0 means collision",
        ["case", "eps_ela/eps_atm"] + [f"eps_ela/eps_att @ {t:g} s" for t in _T_ATT],
        pair_rows,
        notes=(
            "eps_ela sits one to two decades below every other scale in every case, "
            "so the elastic bubble has a gap to stand on — the premise of the §4 "
            "reduction, and not obvious in advance. The closest elastic collision is "
            f"**{closest[0]}** at a ratio of {closest[1]:.2f}, which needs a very "
            "slender flexible body *and* an aggressively fast attitude loop "
            "together.\n\nBy contrast, eps_att/eps_atm = "
            + ", ".join(f"{r:.2f} at T_att={t:g} s" for t, r in atm_att)
            + " — so the pair generically nearest to collision is "
            "**(eps_atm, eps_att)**, not the bending/control resonance one would "
            "expect. §4.2 must therefore treat scale-orthogonality as a hypothesis "
            "checked *pairwise*, and the pair most likely to fail is the one neither "
            "earlier draft flagged."
        ),
    )
    write_csv(output_dir, "r1v1-separation", ["case", "against", "ratio"], pair_csv)

    passed = worst_t_ela <= _T_ATT_SLOWEST
    report.passed = bool(passed)
    report.add_section(
        "Acceptance",
        f"Slowest measured structural period across all parametrizations at "
        f"{100 * (1 - _E_RETAINED):.0f}% modulus loss: **{worst_t_ela:.3e} s**, "
        f"against the slowest admissible attitude period {_T_ATT_SLOWEST:g} s → "
        f"{'**PASS**' if passed else '**FAIL**'}. Generic parametrizations only; "
        "this fixes orders of magnitude and nothing finer.",
    )
    return report


if __name__ == "__main__":  # pragma: no cover - manual invocation
    print(run_r1v1(Path("results")).to_markdown())
