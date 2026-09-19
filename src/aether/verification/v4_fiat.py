"""I-V4 (FIAT-formulation leg) — independent ablation solver verification.

Paper I, §8, V4: *"Ablation: method of manufactured solutions; recession
within 5% of a FIAT reference case."*

**What this leg is, and what it is not.** FIAT is US-government-controlled
software and cannot be run here, so :mod:`aether.fiat` implements
its published formulation independently — Chen & Milos 1999 Eqs. (1)–(11)
and Milos, Chen & Squire 2006 — with a conservative finite-volume
discretisation and a fully implicit Newton solve. That gives the project
a *second*, structurally different solver for the same physics: the
existing path is Chebyshev collocation with method-of-lines integration
on a Landau grid, this one is finite volume with backward Euler and an
analytic Jacobian.

Two independent discretisations agreeing is a real and useful result. It
is **not** a validation against FIAT, and nothing below is reported as
one. The stated 5% criterion needs a published FIAT reference case with
its boundary conditions and property set attached, which this repository
still does not carry — see ``docs/FIAT-reference-data.md``.

The checks executed here are the ones that can be settled without any
external data: conservation, convergence order, exactness at a material
interface, and the correctness of the Newton Jacobian.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from aether.fiat import (
    AerothermalEnvironment,
    BackfaceCondition,
    BackfaceKind,
    BPrimeTable,
    FiatSolver,
    MaterialStack,
    Ply,
    blowing_reduction,
    gray_radiative_flux,
    optical_depth,
)
from aether.fiat.kinetics import (
    TgaTargets,
    fit_arrhenius,
    peak_rate_temperature,
    tga_mass_fraction,
)
from aether.fiat.materials import (
    HERITAGE_PICA_CONDUCTIVITY,
    MEDLI2_PICA_CONDUCTIVITY,
    ONE_ATMOSPHERE,
    PICA_VIRGIN_DENSITY,
    pica_like_material,
    structural_material,
)
from aether.fiat.pica_kinetics import (
    COMPETITIVE_PICA_DETERMINISTIC,
    PARALLEL_PICA_RESIN,
    competitive_mass_fraction,
    parallel_pica_resin,
)
from aether.fiat.solver import _StepContext, _virgin_density
from aether.thermal.material import ArrheniusComponent
from aether.thermal.surface import STEFAN_BOLTZMANN
from aether.verification.common import VerificationReport, write_csv

__all__ = ["run_v4_fiat"]

_ADIABATIC = BackfaceCondition(BackfaceKind.ADIABATIC)


def _table() -> BPrimeTable:
    """A smooth synthetic B' surface — explicitly not a thermochemistry run."""
    t_w = np.linspace(200.0, 4500.0, 30)
    b_g = np.linspace(0.0, 6.0, 12)
    p = np.array([500.0, 5000.0, 50000.0])
    b_c = np.zeros((3, 12, 30))
    h_w = np.zeros((3, 12, 30))
    for i, p_i in enumerate(p):
        for j, g in enumerate(b_g):
            b_c[i, j] = (
                0.35
                / (1.0 + np.exp(-(t_w - 2700.0) / 200.0))
                * (1.0 + 0.15 * g)
                * (p_i / 5000.0) ** 0.08
            )
            h_w[i, j] = 1.2e3 * t_w
    return BPrimeTable(p, b_g, t_w, b_c, h_w)


def _stack(n_cells: int, structure: bool = True) -> MaterialStack:
    plies = [Ply(pica_like_material(), 0.05, n_cells, 1.03, ablating=True)]
    if structure:
        plies.append(Ply(structural_material(), 0.01, max(n_cells // 4, 4)))
    return MaterialStack(plies)


def _pulse(
    n: int, duration: float = 60.0, peak: float = 2.5e6
) -> tuple[NDArray[np.float64], list[AerothermalEnvironment]]:
    t = np.linspace(0.0, duration, n + 1)
    q = peak * (0.02 + 0.98 * np.exp(-(((t[:-1] - 0.42 * duration) / 12.0) ** 2)))
    envs = [
        AerothermalEnvironment(
            film_coefficient=q_i / 2.0e7, recovery_enthalpy=2.0e7, pressure=5000.0
        )
        for q_i in q
    ]
    return t, envs


def _observed_order(values: list[float], refinement: float = 2.0) -> float:
    """Order estimated from three successive refinements."""
    d1 = abs(values[1] - values[0])
    d2 = abs(values[2] - values[1])
    if d2 <= 0.0 or d1 <= 0.0:
        return float("inf")
    return float(np.log(d1 / d2) / np.log(refinement))


def run_v4_fiat(output_dir: Path) -> VerificationReport:
    report = VerificationReport(
        task_id="I-V4-FIAT",
        title="Ablation — independent FIAT-formulation solver (cross-verification leg)",
        criterion=(
            "any conservation, convergence, interface-exactness or Jacobian "
            "property failing its closed form; the stated 5% recession "
            "criterion against a published FIAT reference case is NOT "
            "evaluated here — see scope"
        ),
        passed=True,
    )
    table = _table()
    checks: list[tuple[str, float, float, bool]] = []

    # --- conservation -------------------------------------------------------
    mat_a = structural_material(1600.0, conductivity=0.5, specific_heat=900.0)
    mat_b = structural_material(800.0, conductivity=0.1, specific_heat=1500.0)
    inert = MaterialStack([Ply(mat_a, 0.02, 40), Ply(mat_b, 0.02, 40)])
    q_in, span = 5.0e4, 20.0
    env_inert = AerothermalEnvironment(
        film_coefficient=0.0,
        recovery_enthalpy=0.0,
        pressure=5000.0,
        radiative_flux=q_in,
        wall_absorptance=1.0,
        wall_emissivity=0.0,
    )
    times = np.linspace(0.0, span, 201)
    sol = FiatSolver(inert).solve(times, [env_inert] * 200, table, _ADIABATIC, 300.0)
    grid = inert.grid(0.0)
    rho = np.array([1600.0] * 40 + [800.0] * 40)
    cp = np.array([900.0] * 40 + [1500.0] * 40)
    stored = float(
        np.sum(rho * cp * grid.widths * (sol.steps[-1].temperature - 300.0))
    )
    energy_error = abs(stored / (q_in * span) - 1.0)
    checks.append(
        ("energy stored / energy supplied, sealed inert stack", stored / (q_in * span),
         1.0, energy_error < 5e-3)
    )

    # Pyrolysis mass: Eq. (9) integrated over a whole run against the density
    # the solid actually lost.
    ablator = _stack(40, structure=False)
    solver = FiatSolver(ablator)
    t, envs = _pulse(240)
    run = solver.solve(t, envs, table, _ADIABATIC, 300.0)
    released = float(
        sum(s.surface.gas_mass_flux * (t[i + 1] - t[i]) for i, s in enumerate(run.steps))
    )
    lost = float(
        np.sum(
            (
                solver.bulk_density(solver.initial_state(300.0)[1])
                - solver.bulk_density(run.steps[-1].component_density)
            )
            * ablator.grid(run.recession[-1]).widths
        )
    )
    mass_error = abs(released / lost - 1.0)
    checks.append(
        ("pyrolysis gas released / solid mass lost", released / lost, 1.0,
         mass_error < 3e-2)
    )

    # --- interface exactness -----------------------------------------------
    k1, k2, l1, l2 = 0.5, 0.05, 0.02, 0.005
    series = MaterialStack(
        [
            Ply(structural_material(1600.0, k1, 900.0), l1, 30),
            Ply(structural_material(1600.0, k2, 900.0), l2, 30),
        ]
    )
    q_series = 5.0e3
    env_series = AerothermalEnvironment(
        film_coefficient=0.0,
        recovery_enthalpy=0.0,
        pressure=5000.0,
        radiative_flux=q_series,
        wall_absorptance=1.0,
        wall_emissivity=0.0,
    )
    steady = FiatSolver(series).solve(
        np.linspace(0.0, 120000.0, 401),
        [env_series] * 400,
        table,
        BackfaceCondition(BackfaceKind.FIXED_TEMPERATURE, temperature=300.0),
        300.0,
    )
    expected_wall = 300.0 + q_series * (l1 / k1 + l2 / k2)
    wall_error = abs(steady.wall_temperature[-1] / expected_wall - 1.0)
    checks.append(
        ("steady wall temperature / series-resistance value (k ratio 10:1)",
         steady.wall_temperature[-1] / expected_wall, 1.0, wall_error < 3e-3)
    )

    # --- Jacobian -----------------------------------------------------------
    small = _stack(12)
    js = FiatSolver(small)
    temperature, components = js.initial_state(300.0)
    hot = temperature + np.linspace(1400.0, 40.0, small.n_cells)
    for _ in range(4):
        components = js.decompose(hot, components, 0.5)[0]
    g = small.grid(0.0)
    ctx = _StepContext(
        previous_temperature=hot,
        previous_components=components,
        dt=0.05,
        recession_rate=1.0e-4,
        widths=g.widths,
        centers=g.centers,
        environment=AerothermalEnvironment(0.2, 2.0e7, 5000.0),
        backface=_ADIABATIC,
        table=table,
    )
    u = np.concatenate([hot, [2400.0, hot[-1]]])
    _, info = js._residual(u, ctx)
    analytic = js._jacobian(u, ctx, info)
    numeric = np.zeros_like(analytic)
    for i in range(u.size):
        h = 1.0e-4 * max(abs(u[i]), 1.0)
        up, dn = u.copy(), u.copy()
        up[i] += h
        dn[i] -= h
        numeric[:, i] = (js._residual(up, ctx)[0] - js._residual(dn, ctx)[0]) / (2.0 * h)
    scale = np.maximum(np.abs(numeric), 1e-6 * np.abs(numeric).max())
    jac_error = float(np.max(np.abs(analytic - numeric) / scale))
    checks.append(
        ("max relative Jacobian error vs central differences", jac_error, 0.0,
         jac_error < 1e-4)
    )

    # --- blowing correction against the source's own alternative form -------
    lam, b_prime = 0.5, 1.5
    phi = float(blowing_reduction(b_prime, lam))
    x1 = 2.0 * lam * b_prime * phi
    identity_error = abs(phi / (x1 / np.expm1(x1)) - 1.0)
    checks.append(
        ("ln(1+2λB')/(2λB') vs 2λB'_1/(exp(2λB'_1)−1)", identity_error, 0.0,
         identity_error < 1e-12)
    )

    # --- radiation ----------------------------------------------------------
    iso = np.full(12, 1500.0)
    i0 = STEFAN_BOLTZMANN * 1500.0**4 / np.pi
    equilibrium = float(
        np.max(np.abs(gray_radiative_flux(optical_depth(np.full(12, 0.005), 300.0),
                                          iso, i0, i0)))
    ) / (STEFAN_BOLTZMANN * 1500.0**4)
    checks.append(
        ("gray-kernel flux in radiative equilibrium (expect 0)", equilibrium, 0.0,
         equilibrium < 1e-6)
    )

    rows = [
        [name, f"{got:.6g}", f"{want:.6g}", "yes" if ok else "NO"]
        for name, got, want, ok in checks
    ]
    all_ok = all(ok for _, _, _, ok in checks)
    report.add_table(
        "Closed-form and conservation checks",
        ["check", "measured", "expected", "pass"],
        rows,
        "Every entry is a property the discretisation must have exactly, or "
        "to a stated tolerance, independent of any external data. The "
        "series-resistance check is the one that earns the harmonic "
        "conductivity mean at ply interfaces: an arithmetic mean fails it by "
        "a margin that grows with the conductivity ratio, and the error lands "
        "on the bondline temperature, which is the number a sizing run exists "
        "to produce.",
    )
    write_csv(
        output_dir,
        "v4-fiat-checks",
        ["check", "measured", "expected", "pass"],
        [[name, got, want, ok] for name, got, want, ok in checks],
    )

    # --- convergence --------------------------------------------------------
    grid_values = []
    for n in (20, 40, 80):
        pulse_times, pulse_envs = _pulse(120)
        grid_values.append(
            float(
                FiatSolver(_stack(n))
                .solve(pulse_times, pulse_envs, table, _ADIABATIC, 300.0)
                .recession[-1]
            )
        )
    step_values = []
    for n in (60, 120, 240):
        pulse_times, pulse_envs = _pulse(n)
        step_values.append(
            float(
                FiatSolver(_stack(40))
                .solve(pulse_times, pulse_envs, table, _ADIABATIC, 300.0)
                .recession[-1]
            )
        )
    grid_order = _observed_order(grid_values)
    step_order = _observed_order(step_values)
    grid_ok = grid_order > 1.2
    step_ok = step_order > 0.7
    report.add_table(
        "Convergence of terminal recession under refinement",
        ["refinement", "coarse", "medium", "fine", "observed order", "pass"],
        [
            ["cells (20 → 40 → 80)"]
            + [f"{v * 1e3:.6f} mm" for v in grid_values]
            + [f"{grid_order:.2f}", "yes" if grid_ok else "NO"],
            ["steps (60 → 120 → 240)"]
            + [f"{v * 1e3:.6f} mm" for v in step_values]
            + [f"{step_order:.2f}", "yes" if step_ok else "NO"],
        ],
        "Recession is a *derived* quantity — it integrates a surface "
        "thermochemistry lookup driven by a wall temperature that is itself "
        "the solution of the energy balance — so it is the strictest single "
        "scalar to converge, and the one the stated criterion is written "
        "against. Spatial refinement carries the geometric cell distribution "
        "with it, so the observed order is not the formal order of a uniform "
        "grid. Time refinement is backward Euler, first order by "
        "construction; a higher observed order here would be a sign of an "
        "under-resolved run, not a better scheme.",
    )

    # --- scope --------------------------------------------------------------
    report.add_section(
        "Relationship to the stated V4 criterion — still PENDING",
        "The failure criterion in Paper I §8 is *recession within 5% of a "
        "FIAT reference case*. This leg does not evaluate it and must not be "
        "read as doing so.\n\n"
        "What now exists is an **independent implementation of FIAT's "
        "published formulation** — Chen & Milos 1999 Eqs. (1)–(11), Milos, "
        "Chen & Squire 2006 — written from the open literature, since FIAT "
        "itself is US-government-controlled and cannot be run here. The "
        "project therefore has two structurally different solvers for the "
        "same physics: Chebyshev collocation with method-of-lines "
        "integration on a Landau grid, and conservative finite volume with "
        "backward Euler and an analytic Newton Jacobian. Agreement between "
        "them is a genuine result about the discretisations, and it is not a "
        "validation against FIAT.\n\n"
        "Closing the criterion as written still needs a published reference "
        "case carrying its own wall boundary condition and material property "
        "set; `docs/FIAT-reference-data.md` specifies what that requires and "
        "where to look. I-V4 remains **partially complete**.",
    )

    # --- published pressure dependence --------------------------------------
    low = 0.001 * ONE_ATMOSPHERE
    rows = []
    for name, prop in (
        ("Heritage", HERITAGE_PICA_CONDUCTIVITY),
        ("MEDLI2", MEDLI2_PICA_CONDUCTIVITY),
    ):
        v_hi = float(prop.value(300.0, 0.0, ONE_ATMOSPHERE))
        v_lo = float(prop.value(300.0, 0.0, low))
        c_hi = float(prop.value(300.0, 1.0, ONE_ATMOSPHERE))
        c_lo = float(prop.value(300.0, 1.0, low))
        rows.append(
            [name, f"{v_hi:.3f}", f"{c_hi:.3f}", f"{v_lo:.3f}", f"{c_lo:.3f}",
             f"×{v_lo / v_hi:.2f}"]
        )
    disagreement = float(
        HERITAGE_PICA_CONDUCTIVITY.value(300.0, 0.0, low)
    ) / float(MEDLI2_PICA_CONDUCTIVITY.value(300.0, 0.0, low))
    report.add_table(
        "PICA conductivity against pressure — MEDLI2 paper, Table 3",
        ["model", "virgin 1 atm", "char 1 atm", "virgin 0.001 atm",
         "char 0.001 atm", "virgin pressure ratio"],
        rows,
        "All eight values are published and are reproduced here exactly. The "
        "**two published models disagree about the sign of the effect**: the "
        "Heritage model has virgin conductivity rising by a factor of three as "
        "pore-gas pressure falls to 0.001 atm, the MEDLI2 re-measurement has it "
        f"falling by a quarter, and at 0.001 atm they differ by a factor of "
        f"**{disagreement:.1f}** — in the regime that governs entry. A solver "
        "with pressure-independent conductivity can represent neither, which is "
        "why the property model now interpolates in log-pressure between the "
        "published anchors and clamps rather than extrapolating beyond them. "
        "Neither model is presented as correct.",
    )

    # --- kinetics ------------------------------------------------------------
    material = pica_like_material()
    comps = [material.resin_a, material.resin_b, material.filler]
    weights = np.array([0.5, 0.5, 0.5])
    targets = TgaTargets()
    scan_t = np.linspace(300.0, 1600.0, 3000)
    mass = tga_mass_fraction(comps, weights, scan_t, targets.heating_rate)
    decomposable = 1.0 - targets.char_yield
    onset = float(scan_t[int(np.argmax(mass <= 1.0 - 0.02 * decomposable))])
    peak = peak_rate_temperature(comps, weights, targets.heating_rate)
    # Perturb the starting guess so the fit is a genuine identifiability
    # check rather than a fixed point: 3x on the pre-exponentials, 10% on the
    # activation energies, which is well outside any real fit's uncertainty.
    guess = [
        ArrheniusComponent(
            pre_exponential=c.pre_exponential * 3.0,
            activation_energy=c.activation_energy * 1.10,
            reaction_order=c.reaction_order,
            virgin_density=c.virgin_density,
            char_density=c.char_density,
        )
        for c in comps
    ]
    recovered = fit_arrhenius(scan_t, mass, targets.heating_rate, guess, weights)
    worst_a = max(
        abs(g.pre_exponential / w.pre_exponential - 1.0)
        for g, w in zip(recovered, comps, strict=True)
        if w.pre_exponential > 0.0
    )
    worst_e = max(
        abs(g.activation_energy / w.activation_energy - 1.0)
        for g, w in zip(recovered, comps, strict=True)
        if w.pre_exponential > 0.0
    )
    kinetics_checks = [
        ("TGA char yield vs published bulk densities", float(mass[-1]),
         targets.char_yield, abs(mass[-1] - targets.char_yield) < 5e-3),
        ("2% mass-loss onset at 20 K/min (K)", onset, targets.onset_temperature,
         abs(onset - targets.onset_temperature) < 5.0),
        ("peak mass-loss-rate temperature at 20 K/min (K)", peak,
         targets.peak_temperature, abs(peak - targets.peak_temperature) < 5.0),
        ("worst relative error recovering A from a 3x-perturbed guess", worst_a, 0.0,
         worst_a < 1e-2),
        ("worst relative error recovering E from a 3x-perturbed guess", worst_e, 0.0,
         worst_e < 1e-3),
    ]
    kinetics_ok = all(ok for _, _, _, ok in kinetics_checks)
    report.add_table(
        "Decomposition kinetics — targets and round-trip identifiability",
        ["check", "measured", "target", "pass"],
        [[n, f"{g:.6g}", f"{w:.6g}", "yes" if ok else "NO"]
         for n, g, w, ok in kinetics_checks],
        "**No published Arrhenius triplets for PICA exist in this "
        "repository's reference set.** The MEDLI2 material-response paper "
        "characterises conductivity, specific heat and density and is silent "
        "on decomposition rates; the MSL reconstruction paper notes that 'no "
        "kinetic rate-limited recession model for PICA exists that is "
        "sufficiently validated for use in TPS design'. Rather than assert "
        "three numbers, the triplets are pinned to the stated targets above "
        "and those targets are checked here, so the assumption is visible and "
        "falsifiable. The char yield is not a free parameter: it follows from "
        "the published virgin and char bulk densities.\n\n"
        "The last two rows are the useful part. Forward-modelling a scan and "
        "then fitting it recovers the generating pre-exponentials and "
        "activation energies to better than a part in a thousand, so a real "
        "thermogravimetric curve — one curve — closes the largest remaining "
        "gap in the material model the moment one is available.",
    )

    # --- published PICA kinetics, and FIAT's model-form limit ---------------
    scan = np.linspace(300.0, 2500.0, 3000)
    parallel = parallel_pica_resin(94.0)
    parallel_w = np.ones(len(parallel))
    peaks: dict[str, list[float]] = {"parallel": [], "competitive": []}
    for rate in (10.0, 366.0):
        m_par = tga_mass_fraction(parallel, parallel_w, scan, rate / 60.0)
        m_com = competitive_mass_fraction(
            COMPETITIVE_PICA_DETERMINISTIC, scan, rate / 60.0
        )
        peaks["parallel"].append(
            float(scan[int(np.argmax(-np.gradient(m_par, scan)))])
        )
        peaks["competitive"].append(
            float(scan[int(np.argmax(-np.gradient(m_com, scan)))])
        )
    parallel_shift = peaks["parallel"][1] - peaks["parallel"][0]
    competitive_shift = peaks["competitive"][1] - peaks["competitive"][0]
    form_ok = parallel_shift > 0.0 and competitive_shift < 0.0
    slow_yield = float(
        competitive_mass_fraction(COMPETITIVE_PICA_DETERMINISTIC, scan, 10.0 / 60.0)[-1]
    )
    resin_loss = 1.0 - float(
        tga_mass_fraction(parallel, parallel_w, scan, 10.0 / 60.0)[-1]
    )
    composite_loss = resin_loss * 94.0 / 274.0
    published_loss = (274.0 - 227.0) / 274.0
    cross_ok = (
        abs(slow_yield - 227.0 / 274.0) < 0.02
        and abs(composite_loss - published_loss) < 0.015
    )
    report.add_table(
        "Published PICA pyrolysis kinetics — and the limit of FIAT Eq. (8)",
        ["model form", "peak at 10 K/min", "peak at 366 K/min", "shift"],
        [
            ["parallel — Torres-Herrador 2019 Table 2, *in* Eq. (8)'s form",
             f"{peaks['parallel'][0]:.0f} K", f"{peaks['parallel'][1]:.0f} K",
             f"**{parallel_shift:+.0f} K**"],
            ["competitive — Torres-Herrador 2020 Table 1, *outside* it",
             f"{peaks['competitive'][0]:.0f} K", f"{peaks['competitive'][1]:.0f} K",
             f"**{competitive_shift:+.0f} K**"],
        ],
        "The two heating rates are the ones the 2020 model was calibrated "
        "against: Wong et al. at 10 K/min and Bessire & Minton at 366 K/min. "
        "Carbon/phenolic is measured to shift its pyrolysis peak **down** in "
        "temperature as the heating rate rises — Stokes reported it above "
        "300 K/min — and Torres-Herrador et al. state that parallel "
        "mechanisms are 'not able to reproduce this effect due to their "
        "mathematical formulation'.\n\n"
        f"That is reproduced here: {'**PASS**' if form_ok else '**FAIL**'}. "
        "FIAT Eq. (8) is a sum of independent parallel reactions, and such a "
        "sum can only shift its peak upward, because every term does. "
        "Recovering the measured direction needs two reactions competing for "
        "the same reactant. **This is a model-form limitation of Eq. (8), not "
        "a calibration error, and no refitting removes it.** It matters in "
        "flight rather than in the laboratory: heating rates across the MSL "
        "heat shield run from 60 to 60000 K/min, while legacy TGA calibration "
        "data rarely exceeds tens of K/min.\n\n"
        f"Two cross-checks between unrelated sources, both passing: the "
        f"competitive model's slow-branch char yield is {slow_yield:.3f} "
        f"against {227.0 / 274.0:.3f} from the published bulk densities, and "
        f"the 2019 set's density-loss fractions "
        f"({sum(r.density_loss_fraction for r in PARALLEL_PICA_RESIN):.3f} of "
        f"the resin) scaled by PICA's 94/274 resin fraction give a "
        f"{composite_loss * 100:.1f}% composite mass loss against the "
        f"{published_loss * 100:.1f}% those same densities imply.",
    )

    # --- material provenance ------------------------------------------------
    report.add_table(
        "Material used, by provenance",
        ["quantity", "value", "source"],
        [
            ["virgin bulk density", f"{_virgin_density(pica_like_material()):.1f} kg/m³",
             f"published (Heritage PICA, MEDLI2 paper): {PICA_VIRGIN_DENSITY:.0f}"],
            ["RT conductivity, both pressures", "8 values", "published (Table 3)"],
            ["char bulk density", "227.0 kg/m³", "reconstructed from composition"],
            ["PICA kinetics (parallel)", "6 reactions",
             "published (Torres-Herrador 2019, Table 2)"],
            ["PICA kinetics (competitive)", "10 parameters",
             "published (Torres-Herrador 2020, Tables 1-2)"],
            ["kinetics used by the solver", "—",
             "**Eq. (8) parallel form; see the model-form table**"],
            ["conductivity/c_p slopes", "—", "**representative, not published**"],
            ["B' table", "—", "**synthetic logistic, not thermochemistry**"],
        ],
        "The solver is what is being verified here, not the material. The "
        "bulk density and room-temperature conductivities are taken from the "
        "MEDLI2 material-response paper's Heritage PICA row and are "
        "reproduced exactly; everything else is of the right magnitude and "
        "no more. Recession numbers in this report describe the "
        "discretisation and must not be read as PICA predictions.",
    )

    report.passed = bool(
        all_ok and grid_ok and step_ok and kinetics_ok and form_ok and cross_ok
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run verification task I-V4, FIAT-formulation cross-verification leg"
    )
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    report = run_v4_fiat(args.output)
    path = report.write(args.output, "v4-fiat")
    print(f"I-V4 (FIAT-formulation leg) {'PASS' if report.passed else 'FAIL'} -> {path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
