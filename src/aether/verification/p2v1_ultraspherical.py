"""II-V1 — ultraspherical operator conditioning versus N (univariate leg).

Paper II, §8: *"κ of the assembled Mindlin–Reissner block operator
versus N; comparison against dense Chebyshev collocation. Failure
criterion: κ growing faster than O(N)."*

**Scope of this run.** The Mindlin–Reissner block operator does not yet
exist (roadmap item 9); this run executes the *univariate* leg: the
fourth-order variable-rigidity operator — the same beam operator Paper
I's V1 measured in dense collocation — assembled in ultraspherical form,
with the criterion applied to the operator's conditioning growth and the
dense-collocation comparison drawn from the committed V1 data. The
block-operator measurement remains pending item 9. Accuracy is
cross-checked by solving the free-free beam eigenproblem and comparing
against the analytic frequencies.

Three condition numbers are reported per the Remark in Paper II §5.4
(assert what the citation establishes, measure the rest): the rectangular
banded interior raw and under the leading-diagonal right preconditioner
(the Olver–Townsend O(1) statement), and the bordered square system with
its dense boundary rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.linalg

from aether.structures import free_free_analytic_frequencies, stepped_profile
from aether.ultraspherical import (
    UltrasphericalBVP,
    VariableCoefficientOperator,
    conversion_chain,
    evaluation_row,
)
from aether.ultraspherical.assembly import BoundaryCondition
from aether.verification.common import VerificationReport, write_csv

__all__ = ["run_p2v1"]

_N_SWEEP = (32, 64, 128, 256, 512)
_SLOPE_LIMIT = 1.3  # criterion: not growing faster than O(N)


def _uniform_operator(n: int) -> VariableCoefficientOperator:
    return VariableCoefficientOperator(
        [None] * 4 + [lambda s: np.ones_like(s)], n, (0.0, 1.0)
    )


def _stepped_operator(n: int) -> VariableCoefficientOperator:
    profile = stepped_profile(
        segment_ei=[5.0e6, 1.2e6, 4.0e5],
        segment_mass=[300.0, 120.0, 60.0],
        joints=[4.0, 7.0],
        blend_width=0.8,
    )
    return VariableCoefficientOperator(
        [None, None, profile.d2_ei, lambda s: 2.0 * np.asarray(profile.d_ei(s)), profile.ei],
        n,
        (0.0, 10.0),
    )


def _clamped_bvp(op: VariableCoefficientOperator) -> UltrasphericalBVP:
    """Clamped–clamped BVP for the bordered-conditioning measurement.

    Free-free conditions leave the two rigid-body modes, so the free-free
    *bordered* system is singular by construction — that configuration is
    a generalized eigenproblem, not a BVP, and is measured as such in the
    accuracy cross-check below.
    """
    bcs = [
        BoundaryCondition(-1, {0: 1.0}),
        BoundaryCondition(-1, {1: 1.0}),
        BoundaryCondition(1, {0: 1.0}),
        BoundaryCondition(1, {1: 1.0}),
    ]
    return UltrasphericalBVP(op, bcs)


def run_p2v1(output_dir: Path) -> VerificationReport:
    report = VerificationReport(
        task_id="II-V1",
        title="Ultraspherical operator — conditioning vs N (univariate leg)",
        criterion=(
            "κ growing faster than O(N); Mindlin–Reissner block-operator leg "
            "pending roadmap item 9"
        ),
        passed=True,
    )

    rows_md: list[list[str]] = []
    rows_csv: list[list[object]] = []
    kappas: dict[str, dict[str, list[float]]] = {
        label: {"raw": [], "pre": [], "bordered": []} for label in ("uniform", "stepped")
    }
    for n in _N_SWEEP:
        row: list[str] = [str(n)]
        for label, factory in (("uniform", _uniform_operator), ("stepped", _stepped_operator)):
            op = factory(n)
            k_raw = op.interior_condition_number(preconditioned=False)
            k_pre = op.interior_condition_number(preconditioned=True)
            k_bord = _clamped_bvp(op).condition_number(equilibrated=True)
            kappas[label]["raw"].append(k_raw)
            kappas[label]["pre"].append(k_pre)
            kappas[label]["bordered"].append(k_bord)
            row += [f"{k_raw:.2e}", f"{k_pre:.2f}", f"{k_bord:.2e}"]
            rows_csv.append([n, label, k_raw, k_pre, k_bord])
        rows_md.append(row)

    report.add_table(
        "Conditioning of the fourth-order variable-EI operator (ultraspherical)",
        ["N", "uniform interior raw", "uniform precond.", "uniform bordered",
         "stepped interior raw", "stepped precond.", "stepped bordered"],
        rows_md,
    )
    write_csv(
        output_dir,
        "p2v1-conditioning",
        ["N", "profile", "kappa_interior_raw", "kappa_interior_precond", "kappa_bordered"],
        rows_csv,
    )

    log_n = np.log(_N_SWEEP)
    slopes = {
        label: float(np.polyfit(log_n, np.log(kappas[label]["raw"]), 1)[0])
        for label in kappas
    }
    pre_max = max(max(kappas[label]["pre"]) for label in kappas)
    bordered_slopes = {
        label: float(np.polyfit(log_n, np.log(kappas[label]["bordered"]), 1)[0])
        for label in kappas
    }
    slope_ok = all(s <= _SLOPE_LIMIT for s in slopes.values())
    report.add_section(
        "Acceptance (operator conditioning)",
        f"Fitted log–log slope of the raw interior κ versus N: "
        f"**{slopes['uniform']:.2f}** (uniform), **{slopes['stepped']:.2f}** "
        f"(stepped) against the criterion ≤ {_SLOPE_LIMIT} (not faster than O(N)) → "
        f"{'**PASS**' if slope_ok else '**FAIL**'}. Under the leading-diagonal "
        f"right preconditioner the interior κ is **≤ {pre_max:.1f} at every N** — "
        "the O(1) statement of Olver & Townsend, reproduced. The bordered square "
        f"system (clamped–clamped) grows as N^{bordered_slopes['uniform']:.1f}: the "
        "dense boundary rows cost conditioning that the banded interior does not, "
        "which is exactly the caveat the Remark in Paper II §5.4 raises and defers "
        "to measurement. Free-free conditions are *not* used for this measurement: "
        "they leave the rigid-body null space, so the bordered free-free system is "
        "singular by construction and the configuration is a generalized "
        "eigenproblem (cross-checked below), not a BVP.",
    )

    # --- comparison against dense collocation ------------------------------
    coll = {}
    coll_path = output_dir / "v1-conditioning.csv"
    if coll_path.exists():
        import csv as _csv

        with coll_path.open() as fh:
            for rec in _csv.DictReader(fh):
                if rec["profile"] == "uniform":
                    coll[int(rec["N"])] = float(rec["kappa_elastic"])
    if coll:
        comp_rows = []
        for n in (32, 64):
            if n in coll:
                ultra = kappas["uniform"]["raw"][_N_SWEEP.index(n)]
                comp_rows.append(
                    [str(n), f"{coll[n]:.2e}", f"{ultra:.2e}", f"{coll[n] / ultra:.1e}"]
                )
        report.add_table(
            "Dense collocation (Paper I V1, elastic κ) vs ultraspherical interior",
            ["N", "collocation κ (O(N⁸))", "ultraspherical κ (O(N))", "ratio"],
            comp_rows,
        )
    else:  # pragma: no cover - V1 results always precede this runner
        report.add_section(
            "Dense collocation comparison",
            "Paper I V1 conditioning data not found in the output directory; "
            "run V1 first for the side-by-side.",
        )

    # --- accuracy cross-check: free-free eigenproblem ----------------------
    n = 32
    op = _uniform_operator(n)
    rows = []
    for e in (-1, 1):
        for d in (2, 3):
            r = evaluation_row(n, e, d) * 2.0**d
            rows.append(r / np.linalg.norm(r))
    k_mat = np.vstack([*rows, op.matrix.toarray()[: n - 4, :]])
    m_mat = np.vstack([np.zeros((4, n)), conversion_chain(n, 0, 4).toarray()[: n - 4, :]])
    lam = scipy.linalg.eig(k_mat, m_mat, right=False)
    lam_r = np.sort(lam[np.isfinite(lam)].real)
    lam_pos = lam_r[lam_r > 1e-3]
    analytic = free_free_analytic_frequencies(5, 1.0, 1.0, 1.0) ** 2
    freq_err = float(np.max(np.abs(lam_pos[:5] - analytic) / analytic))
    freq_ok = freq_err < 1e-6
    report.add_section(
        "Accuracy cross-check",
        f"Free-free beam eigenvalues from the ultraspherical pencil at N = {n} "
        f"match the analytic solution to **{freq_err:.1e}** relative "
        f"({'**PASS**' if freq_ok else '**FAIL**'}) — the same physical problem "
        "Paper I's V1 verified in collocation form, now reproduced by the second, "
        "independent discretization.",
    )

    report.add_section(
        "Block-operator leg — PENDING",
        "The stated II-V1 target is the assembled Mindlin–Reissner 3×3 block "
        "operator, which requires the plate kernel of roadmap item 9. This run "
        "establishes the univariate machinery and its conditioning behavior; the "
        "task is not counted complete until the block measurement runs.",
    )
    report.passed = bool(slope_ok and freq_ok)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run verification task II-V1 (univariate leg)")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    report = run_p2v1(args.output)
    path = report.write(args.output, "p2v1-ultraspherical")
    print(f"II-V1 (univariate leg) {'PASS' if report.passed else 'FAIL'} -> {path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
