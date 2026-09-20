r"""A certified upper bound on remaining downrange, conditional on a corridor.

The outer half of the reachability sandwich, for the one functional that a
landing footprint is mostly made of: how far can the body still fly?

The bound
--------

Range and energy are related exactly. With :math:`e = V^2/2 - \mu/r`, drag
deceleration :math:`D` and downrange rate :math:`\dot R = V\cos\gamma`,

.. math::

    \frac{\mathrm{d}e}{\mathrm{d}t} = -DV, \qquad
    \frac{\mathrm{d}R}{\mathrm{d}e} = -\frac{\cos\gamma}{D},

so over an energy interval :math:`[e_f, e_0]`

.. math::

    R_{\text{final}} - R_0 = \int_{e_f}^{e_0} \frac{\cos\gamma}{D}\,\mathrm{d}e
        \;\le\; \frac{e_0 - e_f}{D_{\min}},

using :math:`\cos\gamma \le 1` and any lower bound :math:`D_{\min} > 0` on the
drag encountered. Both steps are inequalities in the safe direction, and
neither depends on the bank history: **the bound holds for every admissible
control**, which is what distinguishes it from a sampled sweep.

The hypothesis, stated rather than assumed
------------------------------------------

:math:`D_{\min}` is only positive if the body stays in air thick enough to
produce drag, so the bound is conditional on a corridor: an altitude ceiling
and a speed window that the trajectory is hypothesised to remain inside. That
hypothesis is exactly what the corridor constraints of the proposal supply --
heating, dynamic pressure and load-factor limits are what make the state set
compact -- and compactness is a hypothesis of the results downstream, not a
conclusion of this function. :class:`RangeBound` carries the hypothesis with
the number so the two cannot be separated by accident.

A body that leaves the ceiling -- a skip -- is outside the hypothesis and the
bound says nothing about it. That is honest rather than conservative: a skip
genuinely has unbounded range in this argument, because arbitrarily thin air
means arbitrarily little energy lost per unit of ground covered.

The arithmetic
--------------

:math:`D_{\min}` is a rigorous enclosure, computed in Arb over the corridor box
and rounded outward, so the bound is a theorem about the model and not a
floating-point suggestion. Subdivision tightens it: drag varies over the
corridor by orders of magnitude, and the infimum over one box is the infimum
over the thinnest air and the slowest speed in it, which no real trajectory
occupies for the whole interval. Splitting the box and bounding each piece
separately reports where the loss is.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from aether.certification.rigorous import interval, outward

__all__ = ["Corridor", "RangeBound", "certified_downrange_bound", "drag_lower_bound"]

_MU = 3.986004418e14
_R_EARTH = 6378137.0


@dataclass(frozen=True)
class Corridor:
    """The hypothesis the bound is conditional on.

    Attributes
    ----------
    altitude_ceiling_m, altitude_floor_m:
        The band the trajectory is hypothesised to stay inside. The ceiling is
        what does the work: it is where the air is thinnest and the drag
        smallest.
    speed_high_ms, speed_low_ms:
        The speed window over which the bound is claimed.
    """

    altitude_ceiling_m: float
    altitude_floor_m: float
    speed_high_ms: float
    speed_low_ms: float

    def __post_init__(self) -> None:
        if not self.altitude_ceiling_m > self.altitude_floor_m:
            raise ValueError(
                f"the ceiling must lie above the floor, got "
                f"{self.altitude_ceiling_m} and {self.altitude_floor_m}"
            )
        if not self.speed_high_ms > self.speed_low_ms > 0.0:
            raise ValueError(
                f"require speed_high > speed_low > 0, got "
                f"{self.speed_high_ms} and {self.speed_low_ms}"
            )

    def describe(self) -> str:
        return (
            f"altitude in [{self.altitude_floor_m / 1e3:.0f}, "
            f"{self.altitude_ceiling_m / 1e3:.0f}] km and speed in "
            f"[{self.speed_low_ms:.0f}, {self.speed_high_ms:.0f}] m/s"
        )


@dataclass(frozen=True)
class RangeBound:
    """A certified bound, inseparable from the hypothesis it rests on."""

    max_downrange_m: float
    drag_lower_bound_ms2: float
    energy_interval_j_kg: float
    corridor: Corridor
    subdivisions: int

    def summary(self) -> str:
        return (
            f"downrange at most {self.max_downrange_m / 1e3:,.0f} km while "
            f"{self.corridor.describe()}, from a certified drag floor of "
            f"{self.drag_lower_bound_ms2:.3f} m/s²"
        )


def drag_lower_bound(
    corridor: Corridor,
    ballistic_coefficient: float,
    density: Callable[[Any], Any],
    *,
    subdivisions: int = 1,
) -> float:
    r"""A rigorous lower bound on :math:`D = \tfrac12 \rho V^2/\beta` over ``corridor``.

    ``density`` maps an altitude interval to a density interval: the monotone
    enclosure of a tabulated atmosphere, or any other sound one. The box is
    split into ``subdivisions`` altitude bands and the bound is the least over
    them, which is tighter than one enclosure of the whole band and equally
    sound.
    """
    if subdivisions < 1:
        raise ValueError(f"subdivisions must be at least 1, got {subdivisions}")
    edges = np.linspace(corridor.altitude_floor_m, corridor.altitude_ceiling_m, subdivisions + 1)
    worst = float("inf")
    for low, high in itertools.pairwise(edges):
        rho_low = outward(density(interval(float(low), float(high))))[0]
        # Drag rises with both density and speed, so its least value over the
        # box is at the thinnest air and the slowest speed. Combining those two
        # *endpoints* matters: multiplying the wide intervals instead would
        # treat each occurrence of the speed as independent, and a ball squared
        # that way spans zero -- [305, 7387] m/s "squared" reaches -2.7e2,
        # which would report a drag floor of zero and refuse to bound anything.
        v_low = interval(corridor.speed_low_ms, corridor.speed_low_ms)
        floor = interval(rho_low, rho_low) * v_low * v_low / (2.0 * ballistic_coefficient)
        worst = min(worst, outward(floor)[0])
    return worst


def certified_downrange_bound(
    corridor: Corridor,
    ballistic_coefficient: float,
    density: Callable[[Any], Any],
    *,
    radius_m: float = _R_EARTH,
    terminal_speed_ms: float | None = None,
    subdivisions: int = 256,
) -> RangeBound:
    r"""Certified upper bound on the downrange still available inside ``corridor``.

    The energy interval is split into ``subdivisions`` speed bands and each
    band is bounded with *its own* drag floor:

    .. math::

        R_{\text{final}} - R_0 \;\le\; \sum_k \frac{\Delta e_k}{D_{\min,k}},
        \qquad
        D_{\min,k} = \frac{\rho_{\text{ceiling}}\, V_{k,\min}^2}{2\beta}.

    Banding is what makes the bound worth having. One global floor takes the
    slowest speed of the whole flight in the thinnest air of the corridor --
    a combination no trajectory spends any time at -- and over a lunar-return
    energy interval that is two orders of magnitude of slack. Per band the
    floor is the one that applies while the body is actually that fast, and the
    sum converges from above to
    :math:`(2\beta/\rho_{\text{ceiling}})\ln(V_{\text{high}}/V_{\text{low}})`.

    The potential energy shed in descending through the corridor is added on
    top, charged to the weakest drag floor, since :math:`e = V^2/2 - \mu/r`
    counts both and a bound on range has to cover range bought with either.

    Every step is still an inequality in the safe direction: the ceiling
    density is a lower bound on the density encountered, each band's slowest
    speed is a lower bound on its speed, the potential term is the deepest
    descent the corridor allows, and :math:`\cos\gamma \le 1`. The result
    holds for every bank history that keeps the trajectory inside the
    corridor.
    """
    final_speed = corridor.speed_low_ms if terminal_speed_ms is None else float(terminal_speed_ms)
    if not corridor.speed_high_ms > final_speed > 0.0:
        raise ValueError(
            f"the terminal speed must lie below the corridor's high speed and "
            f"above zero, got {final_speed} against {corridor.speed_high_ms}"
        )
    if subdivisions < 1:
        raise ValueError(f"subdivisions must be at least 1, got {subdivisions}")
    ceiling = corridor.altitude_ceiling_m
    rho_min = outward(density(interval(ceiling, ceiling)))[0]
    if not rho_min > 0.0:
        raise ValueError(
            f"the air at the {ceiling / 1e3:.0f} km ceiling encloses zero density, so no "
            f"positive drag floor follows and no finite range bound with it. Lower the "
            f"ceiling, or bound the excursion above it some other way"
        )
    edges = np.linspace(corridor.speed_high_ms, final_speed, subdivisions + 1)
    total = interval(0.0, 0.0)
    for fast, slow in itertools.pairwise(edges):
        # The band's kinetic energy interval, and the drag floor across it.
        v_fast = interval(float(fast), float(fast))
        v_slow = interval(float(slow), float(slow))
        delta_e = (v_fast * v_fast - v_slow * v_slow) / 2.0
        floor = interval(rho_min, rho_min) * v_slow * v_slow / (2.0 * ballistic_coefficient)
        total = total + delta_e / floor
    # Potential energy is part of the budget too. The specific energy is
    # e = V^2/2 - mu/r, so a body descending through the corridor sheds
    # mu(1/r_floor - 1/r_ceiling) on top of what it loses in speed, and range
    # bought with that energy is range the bound has to cover. Counting only
    # the kinetic part under-estimates the range, which is the unsafe
    # direction.
    #
    # Where in the descent it is shed is not known, so it is charged to the
    # weakest drag floor in the corridor -- the terminal speed at the ceiling.
    # That is the conservative choice and it is expensive: over a deep corridor
    # this term can exceed the kinetic one. Removing the slack needs the
    # altitude bounded as a function of energy, which is the corridor in the
    # (V, h) plane that the heating, load and dynamic-pressure limits define,
    # not the box used here.
    floor_radius = _R_EARTH + corridor.altitude_floor_m
    radius_floor = interval(floor_radius, floor_radius)
    radius_ceiling = interval(
        _R_EARTH + corridor.altitude_ceiling_m, _R_EARTH + corridor.altitude_ceiling_m
    )
    mu = interval(_MU, _MU)
    potential = mu / radius_floor - mu / radius_ceiling
    weakest = interval(rho_min, rho_min) * interval(final_speed, final_speed) ** 2 / (
        2.0 * ballistic_coefficient
    )
    total = total + potential / weakest
    _, bound = outward(total)
    v_high = interval(corridor.speed_high_ms, corridor.speed_high_ms)
    v_low = interval(final_speed, final_speed)
    _, energy_interval = outward((v_high * v_high - v_low * v_low) / 2.0)
    floor_overall = outward(
        interval(rho_min, rho_min) * v_low * v_low / (2.0 * ballistic_coefficient)
    )[0]
    return RangeBound(
        max_downrange_m=bound,
        drag_lower_bound_ms2=floor_overall,
        energy_interval_j_kg=energy_interval,
        corridor=corridor,
        subdivisions=subdivisions,
    )
