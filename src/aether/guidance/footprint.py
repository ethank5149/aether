r"""Entry footprints: everywhere a lifting entry body can still arrive.

A state says where a body *is*. For a lifting entry the question that matters
next is where it could *get to*: the set of surface points it can still reach
from here, under every admissible bank history, without violating its path
limits. For a body with any real lift that set is a footprint that shrinks as
the body spends its energy, and it has no closed form.

The footprint is mapped the way the literature maps it (Lu & Xue, *Rapid
Generation of Accurate Entry Landing Footprints*, J. Guid. Control Dyn. 33(3),
2010): discretise the control space, integrate the equations of motion to
termination for each control history, and take the hull of the terminal
points. Histories that break the body's heat-flux or load limits are dropped
before they contribute, because a point the body can only reach by exceeding
them is not reachable.

This is an *inner* bound: a sweep finds only what its schedules happen to reach.
:mod:`aether.guidance.entry_ocp` gives the outer edge directly, so the two
together bracket the true footprint.

The footprint is expressed in a downrange/crossrange frame anchored at the
body's sub-point and aligned with its present heading, the frame in which its
boundary is meaningful; :meth:`EntryFootprint.contains` answers whether a
surface point is reachable.

One caveat on reading the numbers. This is a *reachable* set, not a guidance
envelope: the sweep includes sustained maximum-bank turns, so its crossrange
edge is wider than the crossrange usually quoted for a body flown by
near-optimal guidance. The downrange edge has no such ambiguity, and is the
sweep's cleanest check: at zero bank it must reproduce Sanger's closed-form
equilibrium-glide range (:func:`aether.guidance.entry_ocp.sanger_range`).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import scipy.integrate
import scipy.spatial
from numpy.typing import NDArray

from aether.geodesy import GeodeticPosition, great_circle_bearing, great_circle_range
from aether.guidance.entry import EntryBody, entry_dynamics
from aether.guidance.entry_ocp import EntryPathLimits, g_load, stagnation_heat_flux

__all__ = [
    "EntryFootprint",
    "FlownSchedule",
    "assemble_footprint",
    "bank_schedules",
    "convex_hull",
    "entry_footprint",
    "fly_bank_schedule",
    "hull_contains",
    "to_local",
]

_FloatArray = NDArray[np.float64]
_R_EARTH = 6378137.0


# ---------------------------------------------------------------------------
# Frame and hull
# ---------------------------------------------------------------------------


def to_local(
    origin: GeodeticPosition, heading: float, point: GeodeticPosition
) -> tuple[float, float]:
    """(downrange, crossrange) in metres, relative to ``origin`` along ``heading``."""
    arc = float(great_circle_range(origin, point))
    if arc == 0.0:
        return 0.0, 0.0
    bearing = float(great_circle_bearing(origin, point))
    delta = bearing - float(heading)
    return arc * float(np.cos(delta)), arc * float(np.sin(delta))


def convex_hull(points: Sequence[tuple[float, float]]) -> list[int]:
    """Indices of the convex hull of 2-D points, counter-clockwise.

    Delegated to Qhull through :class:`scipy.spatial.ConvexHull`, which handles
    the degenerate inputs a footprint sweep actually produces -- collinear
    terminal points when every schedule flies the same great circle, duplicates
    when schedules coincide.

    Degenerate cases are answered directly rather than handed to Qhull, which
    raises on inputs that have no two-dimensional hull. Fewer than three points,
    or points spanning no area, have no hull to compute and the honest answer is
    the points themselves.
    """
    if len(points) < 3:
        return list(range(len(points)))
    array = np.asarray(points, dtype=np.float64)
    try:
        return [int(i) for i in scipy.spatial.ConvexHull(array).vertices]
    except scipy.spatial.QhullError:
        # Collinear or coincident: no area, so no hull. Return the extreme
        # points along the dominant direction so callers still get a polyline.
        spread = array.max(axis=0) - array.min(axis=0)
        axis = int(np.argmax(spread))
        order = np.argsort(array[:, axis])
        return [int(order[0]), int(order[-1])]


def hull_contains(hull: Sequence[tuple[float, float]], point: tuple[float, float]) -> bool:
    """Whether ``point`` lies inside the polygon ``hull``, by even-odd ray casting.

    A hull with fewer than three vertices has no area and contains nothing.
    """
    if len(hull) < 3:
        return False
    px, py = point
    inside = False
    n = len(hull)
    for i in range(n):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % n]
        if (y1 > py) != (y2 > py):
            x_cross = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if px < x_cross:
                inside = not inside
    return inside


# ---------------------------------------------------------------------------
# The footprint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntryFootprint:
    """Where an entry body can still arrive, given the state it is in now.

    Attributes
    ----------
    label:
        Identifier of the body the footprint was computed for.
    epoch:
        Time (s) of the state the footprint was propagated from.
    origin:
        Sub-body point at ``epoch`` -- the frame's anchor.
    heading:
        Course (rad) at ``epoch`` -- the frame's downrange direction.
    samples:
        Terminal points of every control history that satisfied the path
        limits. The footprint is their hull; these are kept because the density
        inside the hull is itself informative.
    boundary:
        Hull of ``samples``, ordered, as ground points.
    max_downrange_m, min_downrange_m, max_crossrange_m:
        Extent in the anchored frame. Crossrange is reported as the larger of
        the two signed extents.
    rejected:
        How many control histories were dropped.
    binding_constraints:
        Which limits did the rejecting, worst first. A footprint that comes back
        empty is a real answer only if it says which limit bound it.
    worst_heat_flux_w_cm2, worst_g_load:
        Peak values seen across every schedule, kept or rejected, so the margin
        to each limit is visible either way.
    """

    label: str
    epoch: float
    origin: GeodeticPosition
    heading: float
    samples: tuple[GeodeticPosition, ...] = ()
    boundary: tuple[GeodeticPosition, ...] = ()
    max_downrange_m: float = 0.0
    min_downrange_m: float = 0.0
    max_crossrange_m: float = 0.0
    rejected: int = 0
    binding_constraints: tuple[str, ...] = ()
    worst_heat_flux_w_cm2: float = 0.0
    worst_g_load: float = 0.0
    notes: str = ""
    _hull_local: tuple[tuple[float, float], ...] = field(default=(), repr=False)

    @property
    def area_km2(self) -> float:
        """Footprint area (km^2) by the shoelace formula on the hull."""
        if len(self._hull_local) < 3:
            return 0.0
        pts = np.asarray(self._hull_local, dtype=np.float64)
        x, y = pts[:, 0], pts[:, 1]
        return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0) / 1e6

    def local(self, site: GeodeticPosition) -> tuple[float, float]:
        """``site`` as (downrange, crossrange) metres in the anchored frame."""
        return to_local(self.origin, self.heading, site)

    def contains(self, site: GeodeticPosition) -> bool:
        """Whether ``site`` lies inside the footprint hull."""
        return hull_contains(self._hull_local, self.local(site))

    def distance_to(self, site: GeodeticPosition) -> float:
        """Great-circle distance (m) from ``site`` to the nearest reachable point.

        Zero when the site is inside the footprint; infinite when nothing is
        reachable at all.
        """
        if not self.samples:
            return float("inf")
        if self.contains(site):
            return 0.0
        return float(min(great_circle_range(site, p) for p in self.samples))


def assemble_footprint(
    label: str,
    epoch: float,
    origin: GeodeticPosition,
    heading: float,
    terminals: Sequence[GeodeticPosition],
    rejected: int,
    *,
    notes: str = "",
    binding: tuple[str, ...] = (),
    worst_heat_flux_w_cm2: float = 0.0,
    worst_g_load: float = 0.0,
) -> EntryFootprint:
    """Hull the terminal points and measure the footprint's extent."""
    if not terminals:
        return EntryFootprint(
            label=label, epoch=epoch, origin=origin, heading=heading, rejected=rejected,
            binding_constraints=binding, worst_heat_flux_w_cm2=worst_heat_flux_w_cm2,
            worst_g_load=worst_g_load,
            notes=notes or "no control history reached termination within the limits",
        )
    local = [to_local(origin, heading, p) for p in terminals]
    hull_idx = convex_hull(local)
    downrange = [d for d, _ in local]
    crossrange = [c for _, c in local]
    return EntryFootprint(
        label=label,
        epoch=epoch,
        origin=origin,
        heading=heading,
        samples=tuple(terminals),
        boundary=tuple(terminals[i] for i in hull_idx),
        max_downrange_m=float(max(downrange)),
        min_downrange_m=float(min(downrange)),
        max_crossrange_m=float(max(abs(c) for c in crossrange)),
        rejected=rejected,
        binding_constraints=binding,
        worst_heat_flux_w_cm2=worst_heat_flux_w_cm2,
        worst_g_load=worst_g_load,
        notes=notes,
        _hull_local=tuple(local[i] for i in hull_idx),
    )


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------


class FlownSchedule(NamedTuple):
    """One integrated control history: its samples and where it ended up."""

    times: _FloatArray
    altitudes: _FloatArray
    speeds: _FloatArray
    final: _FloatArray


def fly_bank_schedule(
    state: _FloatArray,
    body: EntryBody,
    bank_schedule: Sequence[tuple[float, float]],
    *,
    terminal_speed: float,
    max_time: float,
    step: float,
) -> FlownSchedule | None:
    """Integrate the entry equations of motion under a piecewise-constant bank.

    ``bank_schedule`` is ``[(until_time, bank), ...]``. Returns ``None`` if the
    integrator failed or the state went non-finite.
    """

    def derivatives(_t: float, y: _FloatArray, bank: float) -> _FloatArray:
        return entry_dynamics(y, bank, body, body_radius=_R_EARTH)

    y = np.asarray(state, dtype=np.float64).copy()
    clock = 0.0
    times = [0.0]
    altitudes = [float(y[0] - _R_EARTH)]
    speeds = [float(y[3])]

    while clock < max_time and y[3] > terminal_speed and y[0] > _R_EARTH:
        bank = bank_schedule[-1][1]
        for until, value in bank_schedule:
            if clock < until:
                bank = value
                break
        span = (clock, min(clock + step, max_time))
        solution = scipy.integrate.solve_ivp(
            derivatives, span, y, args=(bank,), rtol=1e-8, atol=1e-8,
        )
        if not solution.success:
            return None
        y = np.asarray(solution.y[:, -1])
        clock = float(solution.t[-1])
        times.append(clock)
        altitudes.append(float(y[0] - _R_EARTH))
        speeds.append(float(y[3]))
        if not np.all(np.isfinite(y)):
            return None

    return FlownSchedule(
        times=np.asarray(times, dtype=np.float64),
        altitudes=np.asarray(altitudes, dtype=np.float64),
        speeds=np.asarray(speeds, dtype=np.float64),
        final=y,
    )


def bank_schedules(
    max_bank: float, n_constant: int, n_reversals: int, max_time: float
) -> list[list[tuple[float, float]]]:
    """Discretise the control space into piecewise-constant bank histories.

    Two families. Constant banks from full-left to full-right sweep the
    crossrange extremes and, at zero, the maximum downrange. Single-reversal
    schedules -- hold one way, then flip -- fill the interior, which is where a
    body steering to a target site actually lives.
    """
    schedules: list[list[tuple[float, float]]] = [
        [(np.inf, float(b))] for b in np.linspace(-max_bank, max_bank, n_constant)
    ]
    if n_reversals > 0:
        for t_rev in np.linspace(0.15, 0.85, n_reversals) * max_time:
            for sign in (1.0, -1.0):
                schedules.append([
                    (float(t_rev), float(sign * max_bank * 0.7)),
                    (np.inf, float(-sign * max_bank * 0.7)),
                ])
    return schedules


def _ground_point(state: _FloatArray) -> GeodeticPosition:
    return GeodeticPosition(
        latitude=float(state[2]),
        longitude=float((float(state[1]) + np.pi) % (2.0 * np.pi) - np.pi),
        altitude=0.0,
    )


def entry_footprint(
    state: _FloatArray,
    body: EntryBody,
    *,
    limits: EntryPathLimits,
    terminal_speed: float,
    label: str = "entry",
    epoch: float = 0.0,
    max_time: float = 4000.0,
    step: float = 5.0,
    n_constant: int = 21,
    n_reversals: int = 6,
) -> EntryFootprint:
    """Map the footprint of an entry body by sweeping its bank schedules.

    ``state`` is the six-element entry vector ``[radius, longitude, latitude,
    speed, flight_path_angle, heading]``. Each schedule is integrated to
    termination; those whose stagnation heat flux or aerodynamic load exceed
    ``limits`` at any sample, and those that skip out of the atmosphere
    without coming back inside ``max_time``, are counted in
    :attr:`EntryFootprint.rejected` and left out of the hull.

    The heat flux is the convective Sutton-Graves estimate of
    :func:`~aether.guidance.entry_ocp.stagnation_heat_flux`, in the same
    exponential atmosphere the dynamics fly; see its note on radiative heating
    at lunar-return speeds.
    """
    origin = _ground_point(state)
    heading = float(state[5])

    terminals: list[GeodeticPosition] = []
    rejected = 0
    violations: Counter[str] = Counter()
    worst_q = 0.0
    worst_g = 0.0
    for schedule in bank_schedules(body.max_bank, n_constant, n_reversals, max_time):
        flown = fly_bank_schedule(
            state, body, schedule, terminal_speed=terminal_speed, max_time=max_time, step=step,
        )
        if flown is None or flown.times.size < 3:
            rejected += 1
            violations["integration"] += 1
            continue
        if flown.final[3] > terminal_speed and flown.final[0] > _R_EARTH:
            # Still fast and still airborne when the clock ran out: a schedule
            # that holds too much lift up from a lunar-return speed climbs out
            # of the atmosphere for good. Where it happens to be then is not a
            # landing point.
            rejected += 1
            violations["skip_out"] += 1
            continue
        peak_q = max(
            stagnation_heat_flux(float(h), float(v), limits.nose_radius_m)
            for h, v in zip(flown.altitudes, flown.speeds, strict=True)
        )
        peak_g = max(
            g_load(body, float(h), float(v))
            for h, v in zip(flown.altitudes, flown.speeds, strict=True)
        )
        worst_q = max(worst_q, peak_q)
        worst_g = max(worst_g, peak_g)
        broken = [
            name
            for name, exceeded in (
                ("heat_flux", peak_q > limits.max_heat_flux_w_cm2),
                ("g_load", peak_g > limits.max_g_load),
            )
            if exceeded
        ]
        if broken:
            rejected += 1
            violations.update(broken)
            continue
        terminals.append(_ground_point(flown.final))

    binding = tuple(name for name, _ in violations.most_common())
    notes = ""
    if not terminals and binding:
        notes = (
            f"every control history violated {' and '.join(binding)}; worst heat flux "
            f"{worst_q:,.0f} W/cm2 against a {limits.max_heat_flux_w_cm2:,.0f} limit, "
            f"worst load {worst_g:.1f} g against {limits.max_g_load:.1f} g"
        )
    return assemble_footprint(
        label, epoch, origin, heading, terminals, rejected,
        notes=notes, binding=binding, worst_heat_flux_w_cm2=worst_q, worst_g_load=worst_g,
    )
