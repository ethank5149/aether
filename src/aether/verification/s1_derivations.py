r"""S1 — the manuscript's derivations, and the numerics generated from them.

The proposal sets out an order of evidence: closed-form constants evaluated
in validated interval arithmetic, then symbolic checks of the derivations
behind them, then a moment-SOS computation as an independent cross-check.
This task is the second tier, run as a build step rather than performed once
by hand.

**What is proved.** Every entry of :data:`aether.symbolic.DERIVATIONS` is a
claim the manuscript makes, stated as a residual built from the field of
record and reduced to zero by exact rewriting: that the equations of motion
are Newton's law in the rotating frame with oblate gravity; that the Jacobi
energy dissipates at the drag power; what the glide balance is once rotation
is kept; what damps the phugoid, and at what rate.

**What is measured, and from what.** The numbers below are not computed by a
second implementation. The field is printed to C from the same expressions by
:mod:`aether.symbolic.codegen`, compiled, and compared against those
expressions in forty digits before anything is integrated with it. So each
table tests a closed form the proposal quotes against the model it was derived
from, and a disagreement cannot be a transcription error in the comparison.

**The three results the proposal leans on.**

1. *Rotation belongs in the glide balance.* Flying east at 7 km/s the Coriolis
   term is half of :math:`g - V^2/r`; the :math:`J_2` terms are a percent of
   it at most.
2. *The glide is not an attracting slow manifold in the thin-atmosphere
   limit.* The phugoid's frequency grows like :math:`\varepsilon^{-1/2}` while
   its damping rate :math:`g/(V E_v)` does not move, so the damping ratio is
   :math:`O(\sqrt\varepsilon)` -- and equals :math:`C_{\mathcal R}
   \sqrt\varepsilon/2`, the residual constant of R1-V4 read a second way. The
   oscillation takes a fixed time to die out however thin the layer.
3. *Step size does not control wrapping.* Re-boxing a rotation every step
   inflates a box by :math:`(1+h)^{2\pi/h}` per revolution, which increases to
   :math:`e^{2\pi}` as the step is refined; and the phugoid is such a
   rotation.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import trapezoid

from aether.certification.rigorous import NUMPY_OPS
from aether.certification.rotating_field import EARTH, vertical_balance_terms
from aether.verification.common import VerificationReport, write_csv

__all__ = ["run_s1"]

_FloatArray = NDArray[np.float64]

#: Mean radius, as the other R1 tasks use, so the residual constant here is the
#: same number R1-V4 measures.
_R_E = 6.371e6
#: Exponential-atmosphere anchor shared with R1-V4: density held at 65 km while
#: the scale height -- hence eps -- is varied.
_H_ANCHOR, _RHO_ANCHOR = 65.0e3, 1.0e-4
_H_PHYSICAL = 7.2e3

#: Failure criteria, stated in advance.
#: A generated routine that differs from its expression by more than this is
#: not a rendering of it. Sixteen digits are available; four are left for the
#: conditioning of the expressions themselves.
_AUDIT_TOLERANCE = 1.0e-12
#: The closed forms are leading order in sqrt(eps) ~ 0.03, so at the physical
#: scale height they are owed agreement to a few percent, not to rounding.
_CLOSED_FORM_TOLERANCE = 0.05


def _reference_density(scale_height: float) -> float:
    """Sea-level density of the exponential atmosphere through the anchor."""
    return float(_RHO_ANCHOR * np.exp(_H_ANCHOR / scale_height))


def _parameters(
    bank: float, beta: float, lift_to_drag: float, scale_height: float
) -> _FloatArray:
    """Parameter vector of the generated field, on a spherical non-rotating planet."""
    return np.array(
        [bank, beta, lift_to_drag, _reference_density(scale_height), scale_height,
         EARTH.mu, _R_E, 0.0, 0.0]
    )


def _glide_beta(altitude: float, speed: float, vertical: float, scale_height: float) -> float:
    """The ballistic coefficient that puts the equilibrium glide at ``altitude``."""
    radius = _R_E + altitude
    net = EARTH.mu / radius**2 - speed**2 / radius
    density = _reference_density(scale_height) * np.exp(-altitude / scale_height)
    return float(density * speed**2 * vertical / (2.0 * net))


def _glide_altitude(
    speed: _FloatArray, beta: float, vertical: float, scale_height: float
) -> _FloatArray:
    """Equilibrium-glide altitude at each speed, by fixed-point iteration.

    The balance is :math:`\\rho(h) V^2 E_v/(2\\beta) = \\mu/r^2 - V^2/r`. Its
    right side moves by :math:`O(H_s/R_e)` over a scale height, so iterating
    the logarithm contracts by that factor each pass.
    """
    lift_scale = _reference_density(scale_height) * speed**2 * vertical / (2.0 * beta)
    altitude = np.full_like(speed, _H_ANCHOR)
    for _ in range(8):
        radius = _R_E + altitude
        altitude = scale_height * np.log(lift_scale / (EARTH.mu / radius**2 - speed**2 / radius))
    return altitude


def _swing_envelope(signal: _FloatArray, t: _FloatArray) -> tuple[_FloatArray, _FloatArray]:
    """Half the swing between successive extrema, and the time each is centred on.

    Peak-to-peak rather than peak, deliberately. The oscillation rides on a
    centre that drifts slowly -- the forced response to the glide's own
    descent -- and a peak measured from zero includes it: maxima then appear
    to decay too slowly and minima too fast, by the same few percent. The
    swing between a maximum and the next minimum cancels the centre.
    """
    rising = signal[1:-1] - signal[:-2]
    falling = signal[2:] - signal[1:-1]
    extrema = np.flatnonzero(rising * falling < 0.0) + 1
    swing = 0.5 * np.abs(np.diff(signal[extrema]))
    centre = 0.5 * (t[extrema][1:] + t[extrema][:-1])
    return swing, centre


def _interval_euler_growth(step: float) -> float:
    """Box growth over one revolution of :math:`\\dot x = y,\\ \\dot y = -x`, re-boxed each step."""
    lower, upper = np.array([0.999, -0.001]), np.array([1.001, 0.001])
    initial = float((upper - lower).max())
    for _ in range(round(2.0 * np.pi / step)):
        lower, upper = (
            np.array([lower[0] + step * lower[1], lower[1] - step * upper[0]]),
            np.array([upper[0] + step * upper[1], upper[1] - step * lower[0]]),
        )
    return float((upper - lower).max()) / initial


def run_s1(output_dir: Path) -> VerificationReport:
    report = VerificationReport(
        task_id="S1",
        title="Symbolic derivation checks and the numerics generated from them",
        criterion=(
            "any registered derivation fails to reduce to zero; or the generated field "
            f"differs from its own expressions by more than {_AUDIT_TOLERANCE:.0e} relative; "
            "or a closed form the proposal quotes for the phugoid differs from direct "
            f"computation by more than {100 * _CLOSED_FORM_TOLERANCE:.0f}% at the physical "
            "scale height"
        ),
        passed=True,
        source="task definition",
    )
    try:
        from aether.symbolic import check_all, compile_field
        from aether.symbolic.__main__ import point_mass_source
    except ImportError as error:
        report.passed = False
        report.add_section(
            "Not run",
            f"SymPy is not installed (`{error}`), so no derivation was checked. That is a "
            "failure, not a skip: a derivation nobody reduced is not known to hold. "
            "Install it with `pip install aether[symbolic]`.",
        )
        return report

    # --- 1. the derivations ------------------------------------------------
    results = check_all()
    report.add_table(
        "The derivations, reduced",
        ["key", "kind", "claim", "used in", "time [s]", "verdict"],
        [
            [
                f"`{r.derivation.key}`",
                r.derivation.kind,
                r.derivation.claim,
                r.derivation.where,
                f"{r.seconds:.2f}",
                "zero" if r.passed else f"**left: {r.remainder}**",
            ]
            for r in results
        ],
        notes=(
            f"{sum(r.passed for r in results)} of {len(results)} residuals reduce to zero by "
            "exact rewriting. An *identity* holds exactly for the model as written; a *limit* "
            "is the leading order in the scale height of an exact expression, so what it "
            "neglects is the next order in `sqrt(eps)` and nothing else.\n\n"
            "`newton-rotating-frame` is the one that carries the appendix: the three "
            "velocity rates are obtained by differentiating the planet-relative velocity "
            "in planet-fixed Cartesian components and are not assumed from the classical "
            "literature, so the Coriolis, centrifugal and oblateness terms are checked "
            "sign by sign."
        ),
    )
    report.passed = all(r.passed for r in results)

    # --- 2. what lift has to balance ---------------------------------------
    rows, csv_rows = [], []
    for speed in (7000.0, 6000.0, 5000.0, 3000.0):
        for latitude_deg in (0.0, 45.0, 80.0):
            state = [
                EARTH.radius + 50.0e3, 0.0, np.radians(latitude_deg), speed, 0.0, np.pi / 2,
            ]
            terms = {k: float(v) for k, v in vertical_balance_terms(state, NUMPY_OPS).items()}
            central = terms["central"]
            rows.append(
                [
                    f"{speed / 1e3:.0f}",
                    f"{latitude_deg:.0f}",
                    f"{central:.3f}",
                    f"{abs(terms['coriolis']):.3f} ({100 * abs(terms['coriolis']) / central:.1f}%)",
                    f"{abs(terms['centrifugal']):.4f} "
                    f"({100 * abs(terms['centrifugal']) / central:.2f}%)",
                    f"{terms['oblate_radial']:+.4f} "
                    f"({100 * terms['oblate_radial'] / central:+.2f}%)",
                ]
            )
            csv_rows.append(
                [speed, latitude_deg, central, terms["coriolis"], terms["centrifugal"],
                 terms["oblate_radial"]]
            )
    report.add_table(
        "What the vertical lift has to balance, at 50 km, level and due east [m/s²]",
        ["V [km/s]", "latitude [deg]", "g − V²/r", "Coriolis", "centrifugal", "J2, radial"],
        rows,
        notes=(
            "Percentages are of `g − V²/r`. Due east is the heading at which the Coriolis "
            "term is largest; it vanishes for flight along a meridian and reverses sign "
            "westbound, so the balance depends on heading once rotation is kept.\n\n"
            "The ordering is not close. Coriolis is tens of percent of the balance at entry "
            "speeds; the centrifugal term and `J2` are one to two percent at most, and that "
            "only where the balance itself is small. **A glide condition that keeps `J2` "
            "and drops rotation has kept the smaller correction.**"
        ),
    )
    write_csv(
        output_dir,
        "s1-glide-balance",
        ["V_m_s", "latitude_deg", "central", "coriolis", "centrifugal", "oblate_radial"],
        csv_rows,
    )

    # --- 3. the generated field, audited before it is used -------------------
    source = point_mass_source()
    field = compile_field(source.expressions, source.state, source.parameters, name=source.name)
    rng = np.random.default_rng(0)
    full = np.array([0.4, 4000.0, 2.5, 1.225, 8500.0, EARTH.mu, EARTH.radius, EARTH.j2,
                     EARTH.rotation_rate])
    worst_audit = 0.0
    for _ in range(25):
        sample = np.array(
            [EARTH.radius + rng.uniform(30e3, 90e3), rng.uniform(-3, 3), rng.uniform(-1.3, 1.3),
             rng.uniform(2000, 7800), rng.uniform(-0.3, 0.3), rng.uniform(-3, 3)]
        )
        worst_audit = max(worst_audit, field.audit(sample, full))
    probe = np.array([EARTH.radius + 45e3, 0.21, 0.54, 5200.0, -0.024, 1.22])
    steps = 20000
    start = time.perf_counter()
    field.rk4(probe, full, 0.01, steps, stride=steps)
    in_loop = (time.perf_counter() - start) / (4 * steps)
    start = time.perf_counter()
    for _ in range(2000):
        field(probe, full)
    per_call = (time.perf_counter() - start) / 2000
    backend_note = (
        "compiled from generated C99"
        if field.backend == "c"
        else "no C compiler found; generated NumPy"
    )
    report.add_section(
        "The generated field against its own expressions",
        f"- backend: **{field.backend}** ({backend_note})\n"
        f"- largest relative gap to the expressions evaluated in 40 digits, over 25 random "
        f"states with rotation and oblateness on: **{worst_audit:.1e}** "
        f"(criterion {_AUDIT_TOLERANCE:.0e})\n"
        f"- one field evaluation inside the driver's loop: **{in_loop * 1e9:.0f} ns**; "
        f"one call from Python: {per_call * 1e6:.1f} µs\n\n"
        "The audit is the check that the printed routine is the checked expression. It is "
        "not a formality: a version of the generator that rewrote integer powers before "
        "eliminating common subexpressions emitted a routine in which the Coriolis factor "
        "`2 ω V` had become `ω² V`. It compiled and ran, and this number was `1e-4`.\n\n"
        "The second line is why the driver exists. A call from Python costs microseconds "
        "whatever it calls, so the compiled field is only fast where the loop around it "
        "is compiled too.",
    )
    report.passed = report.passed and worst_audit <= _AUDIT_TOLERANCE

    # --- 4. the phugoid: frozen spectrum against the closed forms -----------
    rows, csv_rows = [], []
    worst_spectrum = 0.0
    planar = np.ix_([0, 3, 4], [0, 3, 4])
    cases = [
        (h, ld, v, bank)
        for h in (2.0e3, _H_PHYSICAL, 15.0e3)
        for ld, v, bank in (
            (1.5, 6000.0, 0.0), (2.5, 6000.0, 0.0), (2.0, 4000.0, 0.0),
            (2.0, 7000.0, 0.0), (2.0, 6000.0, np.radians(45.0)),
        )
    ]
    for scale_height, lift_to_drag, speed, bank in cases:
        vertical = lift_to_drag * np.cos(bank)
        beta = _glide_beta(_H_ANCHOR, speed, vertical, scale_height)
        radius = _R_E + _H_ANCHOR
        gravity = EARTH.mu / radius**2
        relief = 1.0 - speed**2 / (gravity * radius)
        on_glide = np.array([radius, 0.0, 0.0, speed, 0.0, 0.0])
        jacobian = field.jacobian(on_glide, _parameters(bank, beta, lift_to_drag, scale_height))
        roots = np.linalg.eigvals(jacobian[planar])
        pair = roots[np.abs(roots.imag) > 1e-12][0]
        zeta = -pair.real / abs(pair)
        epsilon = scale_height / _R_E
        c_r = 2.0 * np.sqrt(gravity * _R_E) / (speed * vertical * np.sqrt(relief))
        zeta_closed = c_r * np.sqrt(epsilon) / 2.0
        decay_time, decay_closed = -1.0 / pair.real, speed * vertical / gravity
        if scale_height == _H_PHYSICAL:
            worst_spectrum = max(
                worst_spectrum, abs(zeta / zeta_closed - 1.0), abs(decay_time / decay_closed - 1.0)
            )
        rows.append(
            [
                f"{scale_height / 1e3:.1f}",
                f"{lift_to_drag:.1f}",
                f"{speed:.0f}",
                f"{np.degrees(bank):.0f}",
                f"{2 * np.pi / abs(pair.imag):.0f}",
                f"{zeta:.4f}",
                f"{zeta_closed:.4f}",
                f"{decay_time:.0f}",
                f"{decay_closed:.0f}",
                f"{decay_time * abs(pair.imag) / (2 * np.pi):.1f}",
            ]
        )
        csv_rows.append(
            [scale_height, lift_to_drag, speed, bank, abs(pair.imag), zeta, zeta_closed,
             decay_time, decay_closed]
        )
    report.add_table(
        "The phugoid on the equilibrium glide: frozen spectrum against the closed forms",
        ["H_s [km]", "L/D", "V [m/s]", "bank [deg]", "period [s]", "ζ computed",
         "ζ = C_R√ε / 2", "1/rate [s]", "V E_v / g [s]", "periods per e-fold"],
        rows,
        notes=(
            "Eigenvalues of the generated Jacobian's `(r, V, γ)` block on the glide, against "
            "`ζ = C_R √ε / 2` and the decay time `V E_v / g`, with `E_v = (L/D) cos σ`.\n\n"
            "Read the fourth column from the right against the first. Varying the scale "
            "height by a factor of 7.5 moves the decay time by under two percent: **the "
            "damping rate does not depend on `ε`**. The period shrinks like `√ε`, so the "
            "number of oscillations per e-fold grows like `1/√ε` and the damping ratio "
            "goes to zero. In the thin-atmosphere limit the motion normal to the glide is "
            "an undamped oscillation on its own timescale, and the glide is reached, if at "
            "all, over a time comparable to the entry itself.\n\n"
            "This is what R1-V4's `C_R` measures, from the other side. There the trajectory "
            "is started level on the glide altitude and rings with amplitude "
            "`C_R √ε` scale heights; here that amplitude is twice the damping ratio. One "
            "number, two readings -- and the second says the residual is a ringing "
            "amplitude set by how the trajectory was started, not a rate at which it "
            "settles."
        ),
    )
    write_csv(
        output_dir,
        "s1-phugoid-spectrum",
        ["H_s_m", "lift_to_drag", "V_m_s", "bank_rad", "omega_rad_s", "zeta", "zeta_closed",
         "decay_time_s", "decay_time_closed_s"],
        csv_rows,
    )

    # --- 5. the phugoid: how the amplitudes actually decay -------------------
    rows = []
    worst_decay = 0.0
    for scale_height in (1.8e3, 3.6e3, _H_PHYSICAL):
        lift_to_drag, speed0, bank = 2.0, 6500.0, 0.0
        vertical = lift_to_drag * np.cos(bank)
        beta = _glide_beta(_H_ANCHOR, speed0, vertical, scale_height)
        parameters = _parameters(bank, beta, lift_to_drag, scale_height)
        # Started 0.1 scale heights above the glide: small enough to be linear.
        start_state = np.array(
            [_R_E + _H_ANCHOR + 0.1 * scale_height, 0.0, 0.0, speed0, 0.0, 0.0]
        )
        dt, stride, duration = 0.02, 10, 900.0
        steps = int(duration / dt)
        flown = field.rk4(start_state, parameters, dt, steps, stride=stride)
        t = dt * stride * np.arange(1, flown.shape[0] + 1)
        radii, speeds, gammas = flown[:, 0], flown[:, 3], flown[:, 4]
        gravities = EARTH.mu / radii**2
        frozen = gravities / (speeds * vertical)
        glide = _glide_altitude(speeds, beta, vertical, scale_height)
        eta = (radii - _R_E - glide) / scale_height
        # Measured from the glide's own flight-path angle, which is small and negative.
        theta = gammas + 2.0 * gravities * scale_height / (speeds**2 * vertical)
        measured: list[float] = []
        for signal, factor in ((eta, 1.5), (theta, 0.5)):
            swing, centre = _swing_envelope(signal, t)
            window = (t >= centre[0]) & (t <= centre[-1])
            decrement = np.log(swing[0] / swing[-1])
            predicted = factor * float(trapezoid(frozen[window], t[window]))
            measured.append(decrement / predicted)
        if scale_height == _H_PHYSICAL:
            worst_decay = max(abs(m - 1.0) for m in measured)
        rows.append(
            [
                f"{scale_height / 1e3:.1f}",
                f"{scale_height / _R_E:.2e}",
                f"{measured[0]:.3f}",
                f"{measured[1]:.3f}",
            ]
        )
    report.add_table(
        "How the two amplitudes decay along a descending glide",
        ["H_s [km]", "eps", "altitude: measured / (3/2 rate)", "angle: measured / (1/2 rate)"],
        rows,
        notes=(
            "A trajectory started 0.1 scale heights above the glide and integrated for "
            "900 s with the compiled field. The log-decrement of each envelope is compared "
            "with the integral of the predicted rate over the same interval, where the "
            "rate is `g/(V E_v)` times 3/2 for the altitude deviation and 1/2 for the "
            "flight-path angle.\n\n"
            "The frozen eigenvalue above gives *both* amplitudes the rate `g/(V E_v)`. It "
            "is not what either does, because the oscillator's coefficients drift as the "
            "vehicle slows: the frequency rises, so the altitude amplitude decays faster "
            "and the angle amplitude slower. Their product, the oscillator's action, "
            "decays at twice the frozen rate, which the two columns together confirm. Any "
            "bound on how fast a trajectory approaches the glide has to say which of the "
            "three it means."
        ),
    )
    report.passed = (
        report.passed
        and worst_spectrum <= _CLOSED_FORM_TOLERANCE
        and worst_decay <= _CLOSED_FORM_TOLERANCE
    )

    # --- 6. wrapping -------------------------------------------------------
    rows = [
        [f"{step:g}", f"{_interval_euler_growth(step):.1f}",
         f"{(1.0 + step) ** round(2.0 * np.pi / step):.1f}"]
        for step in (0.1, 0.01, 0.001)
    ]
    report.add_table(
        "Naive interval propagation of a rotation, one revolution",
        ["step h", "box growth, computed", "(1 + h)^(2π/h)"],
        rows,
        notes=(
            f"The limit is `e^(2π) = {np.exp(2 * np.pi):.1f}`. Refining the step makes the "
            "wrapping *worse*: the growth per step is `1 + h` and the number of steps per "
            "revolution is `2π/h`. What controls wrapping is how the set is represented "
            "-- Lohner's QR factorisation, zonotopes, Taylor models, the mean-value form "
            "of `aether.certification.jacobian` -- not how finely time is cut.\n\n"
            "This is the phugoid's problem exactly. In altitude and flight-path angle the "
            "motion about the glide is a rotation damped by a few percent per radian, and "
            "an enclosure re-boxed every step would grow by two orders of magnitude per "
            "period for as long as the glide lasts."
        ),
    )
    return report
