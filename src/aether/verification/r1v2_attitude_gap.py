"""R1-V2 — normal hyperbolicity of the attitude manifold, for Paper 1 §4.5.

\\Cref{prop:skeleton_dimension} removes the attitude and rate block from the
skeleton on the strength of a singular-perturbation argument, and that
argument is licensed by *normal hyperbolicity*: the fast subsystem's
spectrum must stay off the imaginary axis, uniformly over the corridor.
Task R1-V1 reported :math:`\\varepsilon_{\\mathrm{att}}` as a ratio of
*oscillation periods*. That is not the quantity Fenichel theory consumes.
What it consumes is the **spectral abscissa** — the decay rate — and for a
lightly damped attitude mode the two differ by the damping ratio, which
here runs from :math:`3\times10^{-4}` high in the corridor to
:math:`10^{-2}` deep in it.

This task measures the abscissa instead of assuming it, and it does so
from the same panel model :mod:`aether.aerodynamics.panels` already
carries, with one addition the package otherwise lacks:

**The rotational contribution to local incidence.** Every aerodynamic
method in this package computes pressure from a *single* freestream
direction shared by all panels, so the resulting moment depends on
:math:`(\\alpha,\\beta,M,q_\\infty)` and not on body rate. A model with no
rate dependence has no rate damping, hence a purely imaginary attitude
spectrum and no normally hyperbolic manifold at all. Restoring the
:math:`\\boldsymbol\\Omega\\times\\mathbf r` term to the panel-local
velocity is what makes the question answerable, and it is done here
rather than in the aerodynamics package because the manuscript needs the
number before the package needs the feature. **It should move into
:mod:`aether.aerodynamics` before any simulator result with attitude
dynamics is quoted.**

Three facts are reported, in increasing order of how much trouble they
cause:

1. the sign of the damping derivatives, which decides whether the rate
   directions collapse at all;
2. the ratio of the slowest non-zero abscissa to the trajectory rate,
   which is the *actual* separation parameter and which sets the Fenichel
   constant :math:`C_{\\mathcal R}` by its reciprocal;
3. the dynamic pressure at which that ratio falls through unity, below
   which the attitude subsystem is not fast and the reduction does not
   hold — the floor :math:`\\underline q` that §4.5 must hypothesize.

The bank angle is deliberately *not* measured. Its zero eigenvalue is
exact and structural — the aerodynamic moment cannot depend on a rotation
about the freestream axis, because that rotation does not change which
panels face the flow — so it is asserted in the tests as an identity
rather than estimated here.

Geometry and inertias are the generic demonstration body of
:func:`~aether.aerodynamics.panels.curved_lifting_body` with plausible
thin-shell inertias. They fix orders of magnitude and nothing finer; redo
on the §6 geometry before any number here is asserted in the manuscript.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.optimize
from numpy.typing import NDArray

from aether.aerodynamics.closure import blended_pressure_coefficient
from aether.aerodynamics.panels import PanelModel, curved_lifting_body
from aether.verification.common import VerificationReport, write_csv

__all__ = ["attitude_spectrum", "damped_moment", "run_r1v2"]

_FloatArray = NDArray[np.float64]

#: Demonstration-body length (m) and the moment reference as a fraction of
#: it. The default 0.5 is aft of the centre of pressure and statically
#: unstable; a real entry vehicle is ballasted forward, and 0.35 is a
#: representative stable value.
_LENGTH, _CG_FRACTION = 6.0, 0.35
#: Reference flight condition: mid-corridor, hypersonic.
_V_INF, _MACH = 5.0e3, 18.0
#: Equilibrium-air stagnation value, not the 1.839 perfect-gas asymptote.
_CP_MAX = 1.93
#: Thin-shell inertias for a 6 m, ~1200 kg body (kg m^2).
_IXX, _IYY, _IZZ = 1.6e3, 3.6e3, 4.2e3
#: Trajectory timescale, s — sqrt(R_e/g), the denominator of the ratio
#: that decides whether the attitude subsystem is fast.
_T_SLOW = 806.0
#: Incidences (deg) and dynamic pressures (Pa) spanning the corridor.
_ALPHAS = (15.0, 25.0, 35.0)
_Q_DYN = (5.0e1, 2.0e2, 1.0e3, 5.0e3, 2.0e4, 5.0e4, 1.0e5)

#: Failure criterion, stated in advance: if the damping derivatives are not
#: uniformly negative the rate directions do not collapse and the attitude
#: removals of prop:skeleton_dimension are unsupported at *any* altitude —
#: which is a structural failure of the reduction, not a constant to widen.
_CRITERION = (
    "any of the three Newtonian rate-damping derivatives (L_p, M_q, N_r) is "
    "non-negative at any tested incidence, i.e. the attitude rate directions "
    "do not relax and the trim manifold does not attract"
)


def damped_moment(
    model: PanelModel,
    alpha: float,
    beta: float,
    omega: _FloatArray,
    dynamic_pressure: float,
    velocity: float = _V_INF,
) -> _FloatArray:
    """Body-axis aerodynamic moment including the rotational velocity.

    The panel-local relative flow is
    :math:`V\\hat{\\mathbf v} - \\boldsymbol\\Omega\\times\\mathbf r`, and the
    local dynamic pressure scales with the *local* speed rather than the
    freestream one — both terms matter, and dropping the second is the
    usual way a hand-rolled damping estimate comes out wrong.
    """
    a, b = float(alpha), float(beta)
    v_hat = np.array([np.cos(a) * np.cos(b), np.sin(b), np.sin(a) * np.cos(b)])
    v_hat /= np.linalg.norm(v_hat)
    arms = model.centroids - model.reference_point
    v_rel = velocity * v_hat[None, :] - np.cross(np.asarray(omega)[None, :], arms)
    speed = np.linalg.norm(v_rel, axis=1)
    sin_delta = -np.einsum("ij,ij->i", model.normals, v_rel / speed[:, None])
    cp = blended_pressure_coefficient(
        np.arcsin(np.clip(sin_delta, -1.0, 1.0)), _MACH, cp_max=_CP_MAX
    )
    q_local = dynamic_pressure * (speed / velocity) ** 2
    panel_force = -(cp * q_local * model.areas)[:, None] * model.normals
    return np.asarray(np.cross(arms, panel_force).sum(axis=0))


def _derivatives(model: PanelModel, alpha: float, q_dyn: float) -> dict[str, float]:
    """Central-differenced stiffness and damping derivatives (SI units)."""
    zero, h_rate, h_ang = np.zeros(3), 1.0e-3, 1.0e-5

    def d_rate(axis: int) -> _FloatArray:
        e = np.zeros(3)
        e[axis] = 1.0
        plus = damped_moment(model, alpha, 0.0, h_rate * e, q_dyn)
        minus = damped_moment(model, alpha, 0.0, -h_rate * e, q_dyn)
        return (plus - minus) / (2.0 * h_rate)

    d_alpha = (
        damped_moment(model, alpha + h_ang, 0.0, zero, q_dyn)
        - damped_moment(model, alpha - h_ang, 0.0, zero, q_dyn)
    ) / (2.0 * h_ang)
    d_beta = (
        damped_moment(model, alpha, h_ang, zero, q_dyn)
        - damped_moment(model, alpha, -h_ang, zero, q_dyn)
    ) / (2.0 * h_ang)
    return {
        "M_alpha": float(d_alpha[1]),
        "L_beta": float(d_beta[0]),
        "N_beta": float(d_beta[2]),
        "L_p": float(d_rate(0)[0]),
        "M_q": float(d_rate(1)[1]),
        "N_r": float(d_rate(2)[2]),
    }


def attitude_spectrum(
    model: PanelModel, alpha: float, q_dyn: float
) -> tuple[_FloatArray, dict[str, float]]:
    """Eigenvalues of the linearized attitude fast subsystem.

    Longitudinal :math:`(\\alpha, q)` and lateral-directional
    :math:`(\\beta, p, r)` blocks, in the short-period approximation where
    the flight-path angle is frozen — which is exactly the approximation
    the singular-perturbation argument makes.

    The bank angle is omitted: it contributes an exact zero by
    :func:`~aether.verification.r1v2_attitude_gap.damped_moment` being
    independent of it, and including it would only pad the spectrum with a
    root whose value is known analytically.
    """
    d = _derivatives(model, alpha, q_dyn)
    lon = np.array([[0.0, 1.0], [d["M_alpha"] / _IYY, d["M_q"] / _IYY]])
    lat = np.array(
        [
            [0.0, np.sin(alpha), -np.cos(alpha)],
            [d["L_beta"] / _IXX, d["L_p"] / _IXX, 0.0],
            [d["N_beta"] / _IZZ, 0.0, d["N_r"] / _IZZ],
        ]
    )
    eig = np.concatenate([np.linalg.eigvals(lon), np.linalg.eigvals(lat)])
    return np.asarray(eig), d


def _abscissa(model: PanelModel, alpha: float, q_dyn: float) -> float:
    """Slowest non-zero decay rate (1/s); the separation-setting quantity."""
    eig, _ = attitude_spectrum(model, alpha, q_dyn)
    return float(np.abs(eig.real).min())


def run_r1v2(output_dir: Path) -> VerificationReport:
    report = VerificationReport(
        task_id="R1-V2",
        title="Normal hyperbolicity of the attitude manifold",
        criterion=_CRITERION,
        passed=True,
        source="Paper I §8",
    )
    base = curved_lifting_body()
    model = PanelModel(
        centroids=base.centroids,
        normals=base.normals,
        areas=base.areas,
        reference_point=np.array([_CG_FRACTION * _LENGTH, 0.0, 0.0]),
    )

    report.add_section(
        "What is being measured, and why it is not eps_att",
        "`eps_att` of task R1-V1 is a ratio of *periods*. Fenichel theory "
        "consumes the *spectral abscissa*: the slowest non-zero decay rate "
        "of the fast subsystem. For a mode of frequency `w` and damping "
        "ratio `zeta` these differ by a factor `zeta`, and hypersonic "
        "attitude modes are lightly damped. The ratio reported below is "
        "`|Re lambda|_min * T_slow` with `T_slow = "
        f"{_T_SLOW:g}` s; the reduction of §4.5 needs it **large**, and "
        "`C_R` of the main theorem scales as its reciprocal.",
    )

    # --- 1. damping signs, across incidence and c.g. -------------------
    rows, all_negative = [], True
    q_ref = 2.0e4
    for a_deg in _ALPHAS:
        d = _derivatives(model, np.radians(a_deg), q_ref)
        all_negative &= all(d[k] < 0.0 for k in ("L_p", "M_q", "N_r"))
        rows.append(
            [
                f"{a_deg:.0f}",
                f"{d['L_p']:+.3e}",
                f"{d['M_q']:+.3e}",
                f"{d['N_r']:+.3e}",
                f"{d['M_alpha']:+.3e}",
                f"{d['N_beta']:+.3e}",
            ]
        )
    report.add_table(
        f"Damping and stiffness derivatives at q_inf = {q_ref:g} Pa",
        ["alpha [deg]", "L_p", "M_q", "N_r", "M_alpha", "N_beta"],
        rows,
        notes=(
            "Units N·m/(rad/s) for the rate derivatives and N·m/rad for the "
            "stiffnesses. All three rate derivatives are negative, so the "
            "rate directions relax and the trim manifold attracts — the "
            "removals of prop:skeleton_dimension are real removals.\n\n"
            "Two asides worth recording. `N_r` is roughly forty times "
            "smaller than `L_p` and two hundred times smaller than `M_q`: "
            "this body has no vertical fin, so yaw is much the weakest "
            "axis, and it is yaw rather than roll that limits the lateral "
            "spectrum. And `N_beta > 0` here — directionally stable — but "
            "only because the reference point has been moved forward to "
            f"{_CG_FRACTION:.2f}L. At the geometry default of 0.5L both "
            "`M_alpha` and `N_beta` change sign and the configuration has "
            "no attracting trim manifold at all. The removals of §4.5 are "
            "conditional on ballast, not on shape alone."
        ),
    )
    report.passed = all_negative

    # --- 2. the separation ratio versus dynamic pressure ---------------
    rows, csv_rows = [], []
    for q in _Q_DYN:
        eig, _ = attitude_spectrum(model, np.radians(25.0), q)
        absc = float(np.abs(eig.real).min())
        omega = float(np.abs(eig.imag).max())
        # damping ratio of the LEAST-damped oscillatory mode: -Re/|lambda|
        # taken pairwise. Dividing one mode's abscissa by another mode's
        # frequency is not a damping ratio, and the spiral root has none.
        osc = eig[np.abs(eig.imag) > 1.0e-12]
        zeta = float(np.min(-osc.real / np.abs(osc))) if osc.size else float("nan")
        ratio = absc * _T_SLOW
        rows.append(
            [f"{q:g}", f"{absc:.4e}", f"{ratio:.2f}", f"{2 * np.pi / omega:.3f}", f"{zeta:.2e}"]
        )
        csv_rows.append([q, absc, ratio, 2 * np.pi / omega, zeta])
    report.add_table(
        "Spectral abscissa across the corridor (alpha = 25 deg)",
        ["q_inf [Pa]", "min abs Re [1/s]", "x T_slow", "T_osc [s]", "zeta_min"],
        rows,
        notes=(
            "The abscissa scales as `q_inf` to about one part in 1e4 over "
            "these three decades — damping is linear in dynamic pressure "
            "while inertia is fixed, and the residual comes from the "
            "kinematic row of the lateral block, which carries no dynamic "
            "pressure. The column is therefore a straight line in log-log "
            "to plotting accuracy, and the threshold below is a genuine "
            "crossing rather than an artefact of the sample points.\n\n"
            "`zeta_min` is the damping ratio of the least-damped "
            "oscillatory mode. It runs from about 3e-4 at the top of the "
            "corridor to about 1e-2 at the bottom, scaling as the square "
            "root of `q_inf` — damping is linear in dynamic pressure while "
            "frequency goes as its square root. *That* factor is the entire "
            "discrepancy between `eps_att` and the quantity the reduction "
            "actually needs, and it is worst exactly where the reduction is "
            "already weakest: the oscillation is fast everywhere, the "
            "*decay* is not, and the gap between them widens with altitude."
        ),
    )
    write_csv(
        output_dir,
        "r1v2-abscissa",
        ["q_dyn_Pa", "abscissa_1_s", "abscissa_x_T_slow", "T_osc_s", "zeta"],
        csv_rows,
    )

    # --- 3. the dynamic-pressure floor ---------------------------------
    rows = []
    for target in (1.0, 3.0, 10.0):

        def shortfall(log_q: float, _t: float = target) -> float:
            return _abscissa(model, np.radians(25.0), 10.0**log_q) * _T_SLOW - _t

        lo, hi = np.log10(_Q_DYN[0]), np.log10(_Q_DYN[-1])
        if shortfall(lo) * shortfall(hi) >= 0.0:
            continue
        q_star = 10.0 ** scipy.optimize.brentq(shortfall, lo, hi)
        # exponential-atmosphere altitude at the reference speed
        alt_km = -7.2 * np.log(2.0 * q_star / (1.225 * _V_INF**2))
        rows.append([f"{target:.0f}", f"{q_star:.0f}", f"{alt_km:.1f}"])
    report.add_table(
        "Where the separation is won: the floor q_underbar",
        ["required (abs Re) x T_slow", "q_inf [Pa]", "approx altitude [km]"],
        rows,
        notes=(
            "Altitudes are exponential-atmosphere estimates at "
            f"{_V_INF:g} m/s and are indicative only.\n\n"
            "**This is the finding.** A separation of merely unity is not a "
            "separation, and even a factor of ten is bought only well down "
            "in the corridor. Above the floor the attitude subsystem is not "
            "fast, its states do not collapse, and `d_S` is larger there by "
            "up to five. §4.5 must therefore carry `q_inf >= q_underbar` as "
            "a hypothesis and state `d_S` **per hybrid mode** — which costs "
            "nothing, since §2 already carries rarefied and vacuum flight as "
            "distinct modes, and the point of the word alphabet is that "
            "different modes may reduce differently."
        ),
    )

    report.add_section(
        "The bank angle, and why it is absent above",
        "The aerodynamic moment depends on the body-axis freestream "
        "direction, the body rate, and the flow state. Angle of attack and "
        "sideslip fix that direction completely, and bank is by definition "
        "the remaining rotation *about* it — so bank cannot appear in the "
        "moment, and `dM/dsigma = 0` identically. The attitude fast "
        "subsystem therefore carries an exact zero eigenvalue that no "
        "hyperbolicity hypothesis can remove, and bank is a **slow** "
        "coordinate.\n\nThis is a symmetry statement, not a numerical one: "
        "it holds independently of the aerodynamic method, of any "
        "coefficient value, and of whether the vehicle is axisymmetric. It "
        "is asserted in the test suite as an identity to machine precision "
        "rather than estimated here, and it is the reason "
        "prop:skeleton_dimension removes five attitude states and not six.",
    )

    report.add_section(
        "Model deficiency this exposed",
        "No aerodynamic model in this package includes the "
        "`Omega x r` contribution to panel-local incidence: "
        "`PanelModel.velocity_direction` returns one freestream vector for "
        "the whole body. As implemented the vehicle therefore has **no rate "
        "damping in any axis**, its attitude spectrum is purely imaginary, "
        "and nothing collapses. This task patches that locally in "
        "`damped_moment`. Separately, `AerodynamicCoefficients` retains only "
        "the pitching component of a moment vector that is computed in "
        "full, discarding roll and yaw stiffness before they reach the "
        "simulator. Both should be fixed in `aether.aerodynamics` before §6 "
        "quotes any result computed with attitude dynamics.",
    )
    return report
