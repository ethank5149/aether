r"""Dynamic stability derivatives from a forced-oscillation URANS run.

A static table gives :math:`C_m(\alpha)` and therefore the *stiffness*
:math:`C_{m_\alpha}`. It says nothing about whether a disturbance decays,
which is set by the **damping** :math:`C_{m_q} + C_{m_{\dot\alpha}}` — and on a
blunt re-entry body at high Mach that sum can go positive, at which point the
vehicle is statically stable and dynamically divergent, and a six-degree-of-
freedom simulation built on the static table alone will not show it.

The measurement
---------------

Oscillate the body in pitch at fixed amplitude and frequency, and integrate
the moment against incidence over a whole number of cycles. Write the linear
moment as

.. math::

    C_m = C_{m_0} + C_{m_\alpha}\alpha
        + \frac{c}{2V}\left(C_{m_q} + C_{m_{\dot\alpha}}\right)\dot\alpha ,

and drive it with :math:`\alpha = A\sin\omega t`. Over one cycle the constant
term integrates to zero and the stiffness term integrates to zero as well —
:math:`\oint \alpha\,\mathrm{d}\alpha = 0`, which is what makes this work: the
loop encloses *only* the damping. What remains is

.. math::

    W \equiv \oint C_m\,\mathrm{d}\alpha
      = \frac{c}{2V}\left(C_{m_q}+C_{m_{\dot\alpha}}\right)\int_0^T \dot\alpha^2\,\mathrm{d}t
      = \pi A^2 k \left(C_{m_q}+C_{m_{\dot\alpha}}\right),

using :math:`\int_0^T \dot\alpha^2\,\mathrm{d}t = \pi A^2\omega` and the
reduced frequency :math:`k = \omega c / 2V`. So

.. math::

    C_{m_q} + C_{m_{\dot\alpha}} = \frac{W}{\pi A^2 k}.

**The reduced frequency does not cancel.** The drafted reduction used
:math:`2W/\pi A^2`, which is the same integral divided by the wrong thing: it
omits :math:`k` and carries a stray factor of two, so it returns
:math:`2k\,(C_{m_q}+C_{m_{\dot\alpha}})` rather than the derivative. For a
metre-class body at Mach 10 and 10 Hz, :math:`k \approx 5\times10^{-3}`, and
the drafted number is smaller than the true one by roughly a factor of a
hundred — small enough to look like a well-damped vehicle whatever the truth
is, and with the right sign, which is what makes it dangerous rather than
merely wrong.

A negative result is a damped vehicle. Positive is dynamic instability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from aether.cfd.config import FlowConditions, PitchingMotion
from aether.cfd.solver import read_history

__all__ = ["PitchDamping", "pitch_damping", "pitch_damping_from_history"]

_FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PitchDamping:
    """The damping sum, and the evidence for it.

    Attributes
    ----------
    damping:
        :math:`C_{m_q} + C_{m_{\\dot\\alpha}}`. Negative is stable.
    reduced_frequency:
        :math:`k = \\omega c / 2V`. Quote it with the derivative: the
        derivative is a function of :math:`k`, and two numbers measured at
        different :math:`k` are not the same quantity.
    cycle_work:
        :math:`\\oint C_m\\,\\mathrm{d}\\alpha` per cycle, in the order the
        cycles were run. The spread across cycles is the error bar — see
        :attr:`cycle_scatter`.
    cycles_used, cycles_discarded:
        How many whole cycles entered the average, and how many leading ones
        were dropped as the starting transient.
    """

    damping: float
    reduced_frequency: float
    amplitude: float
    cycle_work: _FloatArray
    cycles_used: int
    cycles_discarded: int
    alpha: _FloatArray
    moment: _FloatArray

    @property
    def cycle_scatter(self) -> float:
        """Peak-to-peak spread of the per-cycle result, relative to the mean.

        The convergence check that matters for this measurement. A single
        cycle always produces *a* number; only agreement between consecutive
        cycles says the starting transient has left and the loop has closed.
        Above a few percent, run more cycles before believing the derivative.
        """
        if self.cycle_work.size < 2:
            return float("nan")
        level = float(np.mean(np.abs(self.cycle_work)))
        if level <= 0.0:
            return float("nan")
        return float(
            (np.max(self.cycle_work) - np.min(self.cycle_work)) / level
        )

    @property
    def stable(self) -> bool:
        return bool(self.damping < 0.0)

    def summary(self) -> str:
        return (
            f"C_mq + C_m_alphadot = {self.damping:+.4f} at k = "
            f"{self.reduced_frequency:.4g} "
            f"({self.cycles_used} cycles, scatter "
            f"{100 * self.cycle_scatter:.1f} %) — "
            f"{'damped' if self.stable else 'DYNAMICALLY UNSTABLE'}"
        )


def pitch_damping_from_history(
    time: _FloatArray,
    moment: _FloatArray,
    motion: PitchingMotion,
    reference_length: float,
    speed: float,
    discard_cycles: int = 1,
) -> PitchDamping:
    """Reduce a moment time history into the damping sum.

    ``discard_cycles`` drops leading whole cycles as the starting transient.
    One is the minimum that means anything: the flow starts from a uniform
    freestream, and the first cycle is spent building the shock layer rather
    than responding to the motion. The drafted reduction instead took "the last
    1000 steps", which is not a whole number of cycles at any frequency and
    therefore does not close the loop — an unclosed loop leaves the stiffness
    term in the integral, and :math:`C_{m_\\alpha}` is far larger than the
    damping term it would contaminate.
    """
    time = np.asarray(time, dtype=np.float64)
    moment = np.asarray(moment, dtype=np.float64)
    if time.shape != moment.shape:
        msg = f"time and moment must match; got {time.shape} and {moment.shape}"
        raise ValueError(msg)

    period = motion.period
    span = float(time[-1] - time[0])
    total_cycles = int(np.floor(span / period + 1.0e-9))
    usable = total_cycles - int(discard_cycles)
    if usable < 1:
        msg = (
            f"the history spans {span / period:.2f} cycles at "
            f"{motion.frequency:g} rad/s, which leaves {usable} whole cycles "
            f"after discarding {discard_cycles}. A partial cycle does not close "
            f"the hysteresis loop, so no derivative can be taken from it; run "
            f"at least {int(discard_cycles) + 1} cycles."
        )
        raise ValueError(msg)

    # alpha is reconstructed from the commanded motion rather than read back
    # from the solver: it is exactly known, and SU2 does not write the grid
    # angle to the history file.
    start = time[0] + int(discard_cycles) * period
    samples_per_cycle = max(int(np.ceil(period / max(motion.time_step, 1e-15))), 16)
    works: list[float] = []
    for index in range(usable):
        low = start + index * period
        high = low + period
        available = int(np.count_nonzero((time >= low - 1e-12) & (time <= high + 1e-12)))
        if available < 8:
            msg = (
                f"cycle {index} holds only {available} samples; the time step "
                f"is too coarse to integrate a loop"
            )
            raise ValueError(msg)
        # Resampled onto exactly one period rather than integrated over
        # whichever samples happen to fall inside it. The solver's time step
        # almost never divides the period — 2e-5 s into a 60 rad/s cycle goes
        # 5235.99 times — so a sample-selected window spans slightly more or
        # less than a full loop and leaves it open. That matters far more than
        # it looks: the stiffness term this integral is supposed to annihilate
        # is larger than the damping term it would contaminate by a factor of
        # 1/k, several thousand here, so a loop that fails to close by a
        # thousandth of a cycle moves the answer by percent. Closing it exactly
        # costs one interpolation and removes the error entirely.
        t_cycle = np.linspace(low, high, samples_per_cycle + 1)
        moment_cycle = np.interp(t_cycle, time, moment)
        phase = motion.frequency * (t_cycle - time[0]) + motion.phase
        rate = motion.amplitude * motion.frequency * np.cos(phase)
        # Integrated against time with the analytic alpha-dot rather than
        # against the sampled alpha: identical in exact arithmetic, and the
        # trapezoid rule on a smooth periodic integrand converges far faster
        # than it does on a loop traversed in the (alpha, C_m) plane, where the
        # turning points are where the spacing collapses.
        works.append(float(np.trapezoid(moment_cycle * rate, t_cycle)))

    cycle_work = np.asarray(works, dtype=np.float64)
    reduced = motion.reduced_frequency(reference_length, speed)
    denominator = np.pi * motion.amplitude**2 * reduced
    damping = float(np.mean(cycle_work) / denominator)

    full = time >= start
    alpha_full = motion.amplitude * np.sin(
        motion.frequency * (time[full] - time[0]) + motion.phase
    )
    return PitchDamping(
        damping=damping,
        reduced_frequency=reduced,
        amplitude=float(motion.amplitude),
        cycle_work=cycle_work,
        cycles_used=usable,
        cycles_discarded=int(discard_cycles),
        alpha=alpha_full,
        moment=moment[full],
    )


def pitch_damping(
    history_csv: str | Path,
    motion: PitchingMotion,
    flow: FlowConditions,
    reference_length: float,
    discard_cycles: int = 1,
) -> PitchDamping:
    """Read an SU2 unsteady ``history.csv`` and reduce it.

    The speed the reduced frequency is formed on comes from ``flow``, not from
    a separate argument, so it cannot disagree with the case that produced the
    history.
    """
    history = read_history(Path(history_csv))
    if not history:
        msg = f"{history_csv} has no rows; the unsteady run produced no history"
        raise ValueError(msg)

    time = None
    for key in ("time(s)", "time", "cur_time", "time_iter"):
        if key in history:
            time = history[key]
            break
    if time is None:
        msg = (
            f"no time column in {history_csv} (columns: "
            f"{sorted(history)}). An unsteady SU2 run writes one; a steady run "
            f"does not, and a steady history cannot give a damping derivative."
        )
        raise KeyError(msg)

    moment = None
    for key in ("cmy", "cmz"):
        if key in history:
            moment = history[key]
            break
    if moment is None:
        msg = f"no pitching-moment column in {history_csv} (columns: {sorted(history)})"
        raise KeyError(msg)

    return pitch_damping_from_history(
        time, moment, motion, reference_length, flow.speed, discard_cycles
    )
