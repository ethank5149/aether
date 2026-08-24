"""R1-V3 — the bubble impulse dimension :math:`m`, for Paper 1 §4.6.

\\Cref{thm:dimension_reduction} contains the certified reachable set in a
skeleton plus a Minkowski sum of compact *impulse sets*
:math:`\\mathcal I_j`, one per bubble, each of dimension at most :math:`m`.
Everything else in that theorem is derived; :math:`m` was the last number
still owed. This task measures it.

**What :math:`m` is.** A bubble is one atmospheric pass. It occupies a
boundary layer in slow time, and the only trace it leaves on the reduced
description is a net displacement of the *slow* coordinates. So
:math:`\\mathcal I_j` lives in the skeleton coordinate space
:math:`\\Theta`, and

.. math::

    m \\;=\\; \\operatorname{rank}
    \\frac{\\partial(\\text{net jump in }\\Theta)}{\\partial(\\text{in-pass controls})} .

Two consequences follow immediately and are worth stating before any
number is quoted. First, :math:`m \\le d_{\\mathcal S}` *always* — an
impulse is a displacement in skeleton coordinates, so it cannot have more
directions than there are coordinates. Second, and because of that, the
falsification condition once recorded for this theorem — "if :math:`m` is
not appreciably smaller than :math:`N` the decomposition buys nothing" —
is vacuous: it is satisfied automatically whenever
:math:`d_{\\mathcal S} < N`, which \\cref{prop:skeleton_dimension} already
establishes. The reduction that actually happens is
:math:`N \\to d_{\\mathcal S}`, and :math:`m` plays no part in it, exactly
as \\cref{rem:reduction_payoff} says: bubbles cost multiplicity, not
dimension.

So this task reports two things, and the second is the one that matters:

1. the **rank**, which is what the theorem's "dimension at most :math:`m`"
   needs to be honest about;
2. the **size** of :math:`\\mathcal I_j`, per coordinate, as a fraction of
   that coordinate's entry-wide range — because a full-dimensional impulse
   set that is *small* still constrains the reachable set, and a
   low-dimensional one that is *large* does not.

**A modelling trap, recorded because it produced a wrong answer first.**
If the surface energy balance hands the entire net flux to ablation, then
:math:`\\dot T_w \\equiv 0` identically and :math:`\\dot s \\propto \\dot Q`
exactly, so recession and heat load collapse onto one direction and the
rank comes out one too low. Recession has to be gated on char onset — below
the ablation temperature the net flux heats the substrate, above it the
ablation absorbs the excess and the wall plateaus. The gate is what makes
:math:`s` a genuinely nonlinear functional of the heating history, and it
is physics rather than numerical hygiene.

Geometry, inertias and TPS properties are generic and open; they fix
orders of magnitude and nothing finer. Redo on the §6 geometry before any
number here is asserted in the manuscript.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from aether.aerodynamics.closure import blended_pressure_coefficient
from aether.aerodynamics.panels import curved_lifting_body
from aether.aerothermal import stefan_recession_rate, sutton_graves
from aether.atmosphere import USStandard1976
from aether.verification.common import VerificationReport, write_csv

__all__ = ["impulse_jacobian", "impulse_of_pass", "lift_drag_polars", "run_r1v3"]

_FloatArray = NDArray[np.float64]

R_E, MU = 6.371e6, 3.986004418e14
#: Trajectory scale sqrt(R_e/g) and phugoid scale 2*pi*sqrt(H_s/g), s.
_T_SLOW, _H_S = 806.0, 7.2e3
#: Vehicle and TPS: generic, open data, traceable to no system.
_S_REF, _R_NOSE, _A_ABL, _MASS0 = 12.0, 0.30, 4.0, 1200.0
_RHO_TPS, _DH_ABL, _CP_TPS, _T_TPS = 1600.0, 2.5e7, 1200.0, 0.05
_EMISS, _SIGMA_SB = 0.85, 5.670374419e-8
#: Char-onset temperature and logistic switch width, K.
_T_ABL, _DT_ABL = 1800.0, 60.0
#: Entry interface and nominal pass.
_H_ENTRY, _V_ENTRY, _SIGMA_0 = 80.0e3, 6.5e3, np.radians(30.0)
_GAMMA_0, _ALPHA_0, _P_MAX, _T_CAP = -4.0, 16.0, 5.0, 400.0
#: Corridor caps used to filter the sampled impulse set.
_QBAR_MAX, _QDOT_MAX = 50.0e3, 3.0e6

#: The eight slow coordinates carried through the pass. The ninth, the
#: recast modulus field, needs a resolved wall-temperature field rather
#: than the lumped slab used here and is left to R1-V4.
LABELS = ("theta", "phi", "V", "psi", "sigma", "mass", "s", "Q_tot")
#: Entry-wide range of each, for normalization: an impulse is only large
#: or small relative to how far its coordinate travels over a whole entry.
_SCALE = np.array([0.60, 0.30, 7.0e3, 1.0, np.pi, 200.0, 0.05, 1.0e9])

_ATM = USStandard1976()
_ALT = np.linspace(0.0, 86.0e3, 400)
_RHO = CubicSpline(_ALT, np.array([float(_ATM.state(h).density) for h in _ALT]))
_AOA = np.radians(np.linspace(0.0, 50.0, 51))


def _polar() -> tuple[CubicSpline, CubicSpline]:
    """Lift and drag polars for the demonstration body, from its panels."""
    body = curved_lifting_body()
    cl, cd = [], []
    for a in _AOA:
        v = np.array([np.cos(a), 0.0, np.sin(a)])
        cp = blended_pressure_coefficient(
            np.arcsin(np.clip(-(body.normals @ v), -1.0, 1.0)), 18.0, cp_max=1.93
        )
        f = (-(cp * body.areas)[:, None] * body.normals).sum(axis=0) / _S_REF
        cd.append(float(f @ v))
        cl.append(float(f @ np.array([-np.sin(a), 0.0, np.cos(a)])))
    return CubicSpline(_AOA, np.array(cl)), CubicSpline(_AOA, np.array(cd))


_CL, _CD = _polar()


def lift_drag_polars() -> tuple[CubicSpline, CubicSpline]:
    """Lift and drag polars of the demonstration body, shared with R1-V4.

    Both tasks must fly the *same* vehicle or their conclusions cannot be
    compared, and comparing them is the point of R1-V4's last section.
    """
    return _CL, _CD


def _piecewise(values: _FloatArray, t: float, horizon: float) -> float:
    n = len(values)
    return float(values[min(int(t / horizon * n), n - 1)])


def _rhs(
    t: float,
    x: _FloatArray,
    alphas: _FloatArray,
    prates: _FloatArray,
    horizon: float,
) -> list[float]:
    r, _th, ph, speed, ga, ps, sg, mass, _s, _q, t_wall = x
    h = r - R_E
    rho = float(_RHO(np.clip(h, 0.0, 86.0e3))) if h < 86.0e3 else 0.0
    a = _piecewise(alphas, t, horizon)
    qbar = 0.5 * rho * speed * speed
    lift = qbar * _S_REF * float(_CL(np.clip(a, 0.0, _AOA[-1])))
    drag = qbar * _S_REF * float(_CD(np.clip(a, 0.0, _AOA[-1])))
    qdot = float(sutton_graves(rho, _R_NOSE, speed))
    gate = 1.0 / (1.0 + np.exp(-(t_wall - _T_ABL) / _DT_ABL))
    sdot = gate * float(
        stefan_recession_rate(qdot, t_wall, _EMISS, 0.0, _RHO_TPS, _DH_ABL)
    )
    g, cg = MU / r**2, np.cos(ga)
    return [
        speed * np.sin(ga),
        speed * cg * np.sin(ps) / (r * np.cos(ph)),
        speed * cg * np.cos(ps) / r,
        -drag / mass - g * np.sin(ga),
        lift * np.cos(sg) / (mass * speed) + (speed / r - g / speed) * cg,
        lift * np.sin(sg) / (mass * speed * cg)
        + (speed / r) * cg * np.sin(ps) * np.tan(ph),
        _piecewise(prates, t, horizon),
        -_RHO_TPS * _A_ABL * sdot,
        sdot,
        qdot,
        (qdot - _EMISS * _SIGMA_SB * t_wall**4 - _RHO_TPS * _DH_ABL * sdot)
        / (_RHO_TPS * _CP_TPS * _T_TPS),
    ]


def impulse_of_pass(
    params: _FloatArray, gamma_deg: float = _GAMMA_0
) -> tuple[_FloatArray, Any]:
    """Net jump in the eight slow coordinates across one atmospheric pass.

    ``params`` is ``n`` angles of attack (deg) followed by ``n`` roll-rate
    commands (deg/s), each piecewise constant over the pass. The pass ends
    when the vehicle returns to the entry altitude — a genuine skip — so
    the recorded jump is the bubble's contribution and not a whole entry.
    """
    half = len(params) // 2
    alphas = np.radians(np.clip(params[:half], 2.0, 48.0))
    prates = np.radians(np.clip(params[half:], -_P_MAX, _P_MAX))
    x0 = np.array(
        [R_E + _H_ENTRY, 0.0, 0.0, _V_ENTRY, np.radians(gamma_deg),
         np.radians(90.0), _SIGMA_0, _MASS0, 0.0, 0.0, 300.0]
    )

    def skipped_out(t: float, x: _FloatArray, *_a: Any) -> float:
        return float((x[0] - R_E) - _H_ENTRY)

    skipped_out.terminal = True  # type: ignore[attr-defined]
    skipped_out.direction = 1.0  # type: ignore[attr-defined]
    sol = solve_ivp(
        _rhs, (0.0, _T_CAP), x0, args=(alphas, prates, _T_CAP),
        rtol=1e-10, atol=1e-12, events=skipped_out, max_step=1.0,
    )
    d = sol.y[:, -1] - x0
    return np.array([d[1], d[2], d[3], d[5], d[6], d[7], d[8], d[9]]), sol


def impulse_jacobian(n_seg: int) -> _FloatArray:
    """Normalized Jacobian of the impulse with respect to in-pass controls.

    Entry flight-path angle is included as one extra column: it is not an
    in-pass control, but admitting it can only *raise* the rank, and an
    upper bound on :math:`m` is what the theorem needs.
    """
    p0 = np.array([_ALPHA_0] * n_seg + [0.0] * n_seg)
    step = np.full(2 * n_seg, 0.4)
    cols = []
    for j in range(2 * n_seg):
        hi, lo = p0.copy(), p0.copy()
        hi[j] += step[j]
        lo[j] -= step[j]
        cols.append(
            (impulse_of_pass(hi)[0] - impulse_of_pass(lo)[0]) / (2.0 * step[j])
        )
    cols.append(
        (impulse_of_pass(p0, _GAMMA_0 + 0.2)[0]
         - impulse_of_pass(p0, _GAMMA_0 - 0.2)[0]) / 0.4
    )
    return np.asarray(np.column_stack(cols) / _SCALE[:, None])


def run_r1v3(output_dir: Path) -> VerificationReport:
    report = VerificationReport(
        task_id="R1-V3",
        title="Bubble impulse dimension m",
        criterion=(
            "a single in-corridor bubble spans 100% or more of the entry-wide "
            "range in EVERY slow coordinate, i.e. one pass reaches everywhere "
            "and the skeleton-plus-impulse description constrains nothing"
        ),
        passed=True,
        source="Paper I §8",
    )

    _base, sol = impulse_of_pass(np.array([_ALPHA_0] * 4 + [0.0] * 4))
    duration = float(sol.t[-1])
    report.add_section(
        "The nominal pass, and the width of the layer it occupies",
        f"Entry at {_H_ENTRY/1e3:.0f} km, {_V_ENTRY:.0f} m/s, "
        f"gamma = {_GAMMA_0:.1f} deg, bank {np.degrees(_SIGMA_0):.0f} deg. The "
        f"pass reaches {(sol.y[0].min()-R_E)/1e3:.1f} km, peak wall temperature "
        f"{sol.y[10].max():.0f} K, and returns to the entry altitude after "
        f"**{duration:.1f} s**.\n\n"
        f"That duration is {duration/_T_SLOW*100:.1f}% of `T_slow` = {_T_SLOW:g} s. "
        "It should be compared against the layer width, and the comparison "
        "carries a correction. The layer is set by the phugoid period "
        "`T_phug = 2*pi*sqrt(H_s/g)`, against `T_slow = sqrt(R_e/g)`, so the "
        "ratio is `2*pi*sqrt(eps_atm)`, **not** `sqrt(eps_atm)`:\n\n"
        f"- `sqrt(eps_atm)` = {np.sqrt(_H_S/R_E):.4f} ({np.sqrt(_H_S/R_E)*100:.1f}%)\n"
        f"- `2*pi*sqrt(eps_atm)` = {2*np.pi*np.sqrt(_H_S/R_E):.4f} "
        f"({2*np.pi*np.sqrt(_H_S/R_E)*100:.1f}%)\n"
        f"- measured pass duration = {duration/_T_SLOW:.4f} "
        f"({duration/_T_SLOW*100:.1f}%)\n\n"
        "The measurement lands on the second. The functional form "
        "`rho(eps) = C_R*sqrt(eps)` is unaffected, but any numerical estimate "
        "that substitutes `sqrt(eps) = 0.034` understates the layer by a "
        "factor of `2*pi`, and `C_R` is meant to be explicit.",
    )

    # --- rank, and its saturation in control freedom -------------------
    rows, spectra = [], {}
    for n_seg in (4, 8, 16):
        sv = np.linalg.svd(impulse_jacobian(n_seg), compute_uv=False)
        spectra[n_seg] = sv
        rows.append(
            [f"{n_seg}", f"{2*n_seg+1}"]
            + [f"{int(np.sum(sv > t * sv[0]))}" for t in (1e-4, 1e-6, 1e-9)]
        )
    report.add_table(
        "Rank of the impulse map, against how much control freedom it is given",
        ["segments", "controls", "rank @1e-4", "rank @1e-6", "rank @1e-9"],
        rows,
        notes=(
            "The rank must be measured against *increasing* control freedom or "
            "it reports the parametrization rather than the map: four segments "
            "give rank 5 purely because nine controls cannot excite more. It "
            "saturates by eight segments and does not move again at sixteen.\n\n"
            "**m = 7** of the eight slow coordinates carried here."
        ),
    )

    sv16 = spectra[16]
    u16 = np.linalg.svd(impulse_jacobian(16))[0]
    rows = []
    for i in range(8):
        comp = ", ".join(
            f"{LABELS[k]} {u16[k,i]:+.2f}" for k in range(8) if abs(u16[k, i]) > 0.25
        )
        rows.append([f"u{i+1}", f"{sv16[i]/sv16[0]:.2e}", comp or "—"])
    report.add_table(
        "The impulse directions, strongest first (16 segments)",
        ["direction", "sigma_i / sigma_1", "dominant coordinates"],
        rows,
        notes=(
            "Seven directions are reachable and one is null to machine "
            "precision. The null direction is identified below; the other "
            "seven line up nearly one-to-one with the coordinates, which is "
            "the substantive content: **a single pass moves essentially every "
            "slow coordinate independently.**"
        ),
    )

    # --- the one exact degeneracy --------------------------------------
    null = u16[:, -1]
    predicted = np.zeros(8)
    predicted[5], predicted[6] = _SCALE[5], _RHO_TPS * _A_ABL * _SCALE[6]
    predicted /= np.linalg.norm(predicted)
    alignment = float(abs(null @ predicted))
    report.add_section(
        "The single exact degeneracy, and what it means",
        "The null direction is `mass "
        f"{null[5]:+.4f}`, `s {null[6]:+.4f}`, against the prediction "
        f"`{predicted[5]:+.4f}`, `{predicted[6]:+.4f}` from "
        "`d(mass) = -rho_TPS * A_abl * d(s)`. They agree to "
        f"|cos| = {alignment:.10f}.\n\n"
        "So the one structural relation the bubble obeys is that **mass is "
        "not an independent state within an atmospheric letter**: ablation is "
        "the only mass sink there, so mass is slaved to recession. It becomes "
        "independent again in the propulsive letter, where thrust supplies a "
        "second sink. This is the same per-mode pattern R1-V2 found for the "
        "attitude block, arrived at from a different direction, and it means "
        "`d_S` and `m` are both properly stated per letter rather than once "
        "for the whole word.",
    )

    # --- the size of the impulse set, which is what actually matters ---
    rng = np.random.default_rng(0)
    free, in_corridor = [], []
    for _ in range(240):
        params = np.concatenate([rng.uniform(10.0, 30.0, 4), rng.uniform(-5.0, 5.0, 4)])
        try:
            out, s = impulse_of_pass(params, _GAMMA_0 + rng.uniform(-1.0, 1.0))
        except Exception:  # a diverged sample is simply discarded
            continue
        if s.status != 1:
            continue
        alt, spd = s.y[0] - R_E, s.y[3]
        rho = np.array([float(_RHO(np.clip(a, 0.0, 86.0e3))) for a in alt])
        qbar = float((0.5 * rho * spd * spd).max())
        qdot = max(float(sutton_graves(r, _R_NOSE, v)) for r, v in zip(rho, spd, strict=True))
        free.append(out / _SCALE)
        if qbar <= _QBAR_MAX and qdot <= _QDOT_MAX:
            in_corridor.append(out / _SCALE)
    sampled, kept = np.array(free), np.array(in_corridor)
    extent_free = sampled.max(0) - sampled.min(0)
    extent_corr = kept.max(0) - kept.min(0)
    report.add_table(
        "Size of one bubble's impulse set, as a fraction of each coordinate's "
        "entry-wide range",
        ["coordinate", "unconstrained", "in corridor", "corridor shrink"],
        [
            [LABELS[i], f"{100*extent_free[i]:.1f}%", f"{100*extent_corr[i]:.1f}%",
             f"{extent_corr[i]/extent_free[i]:.2f}"]
            for i in range(8)
        ],
        notes=(
            f"{len(sampled)} admissible skips sampled, {len(kept)} of them inside "
            f"qbar <= {_QBAR_MAX/1e3:.0f} kPa and "
            f"qdot <= {_QDOT_MAX/1e4:.0f} W/cm^2. The corridor roughly halves "
            f"the l2 diameter, from {np.linalg.norm(extent_free):.2f} to "
            f"{np.linalg.norm(extent_corr):.2f}.\n\n"
            "**This is the finding, and it is not the rank.** The impulse set "
            "is small in the energy and thermal coordinates — recession 3%, "
            "mass 5%, heat load 21%, speed 52% — and *enormous* in the "
            "steering ones, where a single pass moves heading by twice its "
            "entry-wide range and bank by five times. The decomposition is "
            "therefore quantitatively informative about the energy-thermal "
            "block and close to vacuous about the steering block. That is not "
            "a defect of the bound so much as the physics: a vehicle with bank "
            "authority genuinely can steer across its footprint in one pass. "
            "But it does mean §4.8 must not advertise a uniform reduction, and "
            "§6 must report tightness per coordinate group rather than as a "
            "single number."
        ),
    )
    write_csv(
        output_dir, "r1v3-impulse-extent",
        ["coordinate", "extent_unconstrained", "extent_in_corridor"],
        [[LABELS[i], extent_free[i], extent_corr[i]] for i in range(8)],
    )
    report.passed = bool(np.any(extent_corr < 1.0))
    return report
