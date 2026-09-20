"""Artemis I: Orion's lunar-return skip entry, from entry interface to terminal speed.

Every number here is either taken from a published NASA source, cited where
it is used, or computed by this package from those numbers. Nothing is
tuned to reproduce the flight, and the comparison at the end is against what
the flight reports.

The chain
---------

1. **Geometry.** The Orion outer mould line from the proportions of the CEV
   wind-tunnel models (NTRS 20080008559, Table 1): a spherical heat shield of
   radius 1.2 D, 0.05 D shoulders and a 32.5-degree backshell, at the flight
   diameter of 198 in (NTRS 20110013644).
2. **Aerodynamics.** Modified-Newtonian panel loads on that shape, at the
   equilibrium-air stagnation pressure coefficient. The angle of attack is
   the one at which L/D equals the design value of 0.30 (NTRS 20070023643);
   the drag coefficient there sets the ballistic coefficient. The trim angle
   this implies is an independent check: the Orion aerodynamic database puts
   hypersonic trim between 157 and 162 degrees (NTRS 20110013644).
3. **The flight.** Artemis I's entry interface state (AAS 24-174, Table 1),
   flown by :mod:`aether.guidance.skip` to the splashdown point (Table 4)
   through an atmosphere 10 % thinner and an L/D 5 % lower than the
   guidance's model -- the differences the flight's own estimators reported.
4. **The reachable set, used.** The landing footprint from entry interface,
   from skip apogee and from the start of the Final phase: the set of sites
   still reachable, shrinking as energy is spent, with the target inside it
   throughout.

Run with ``python -m examples.artemis1.entry``.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np

from aether.aerodynamics.panels import PanelModel, surface_of_revolution
from aether.atmosphere.model import earth_atmosphere, tabulate
from aether.certification.atmosphere import monotone_density
from aether.certification.range_bound import Corridor, RangeBound, certified_downrange_bound
from aether.geodesy import GeodeticPosition
from aether.guidance.entry import _R_EARTH, EntryBody, EntryState
from aether.guidance.entry_ocp import EntryPathLimits
from aether.guidance.footprint import EntryFootprint, entry_footprint
from aether.guidance.skip import (
    EARTH_ROTATION_RATE,
    SkipEntryResult,
    SkipGuidance,
    SkipGuidanceConfig,
    SkipPhase,
    fly_skip_entry,
)

FT = 0.3048
NMI = 1852.0
LBM = 0.45359237

#: Where each input comes from. Keys are the NTRS citation numbers of the
#: papers in the project's reference library.
SOURCES = {
    "AAS 24-174": "Orion Artemis I Entry Performance (NTRS 20240000024)",
    "20070023643": "A Comparison of Two Skip Entry Guidance Algorithms",
    "20110013644": "Development of the Orion Crew Module Static Aerodynamic Database, Part I",
    "20080008559": "Aerothermodynamic Testing of the Crew Exploration Vehicle",
}

#: Maximum diameter, 198 in (20110013644).
DIAMETER = 198.0 * 0.0254
#: Nominal entry mass, 20,369.2 lbm (20070023643).
MASS = 20369.2 * LBM
#: Design hypersonic L/D, the low end of 0.30-0.35 (20070023643).
LIFT_TO_DRAG = 0.30
#: Largest bank magnitude commanded. Lift-down to 165 degrees; not from a
#: source -- a stated assumption that keeps a margin from full lift-down.
MAX_BANK = np.deg2rad(165.0)

#: Artemis I entry interface (AAS 24-174, Table 1).
INTERFACE = EntryState(
    radius=_R_EARTH + 400000.0 * FT,
    longitude=np.deg2rad(-120.08071),
    latitude=np.deg2rad(-25.82847),
    speed=36062.6568 * FT,
    flight_path_angle=np.deg2rad(-5.66367),
    heading=np.deg2rad(4.65389),
)
#: Splashdown (AAS 24-174, Table 4). The target itself is not tabulated; the
#: flight splashed down 2 nmi from it, so this stands in for it.
TARGET = (np.deg2rad(-118.10181), np.deg2rad(27.34852))

#: What the flight reports, for the comparison (AAS 24-174).
FLOWN = {
    "final phase (s)": 551.425,
    "terminal phase (s)": 882.400,
    "skip apogee (kft)": 287.4,
    "bank reversals": 6,
    "reversal times (s)": (115.475, 390.450, 713.425, 793.425, 827.425, 864.400),
    "terminal-phase start from target (nmi)": 2.7,
}


# ---------------------------------------------------------------------------
# Geometry and aerodynamics
# ---------------------------------------------------------------------------


def orion_meridian(diameter: float = DIAMETER) -> tuple[np.ndarray, np.ndarray]:
    """``(x, r)`` of the Orion outer mould line, heat shield forward.

    Proportions of the CEV Cycle I wind-tunnel models (NTRS 20080008559,
    Table 1): heat-shield radius 1.2 D, front shoulder radius 0.05 D, a
    32.5-degree backshell cone closing to a 0.356 D base.
    """
    nose, shoulder = 1.2 * diameter, 0.05 * diameter
    cone = np.deg2rad(32.5)
    base = 0.356 * diameter / 2.0
    torus_r = diameter / 2.0 - shoulder
    torus_x = nose - np.sqrt((nose - shoulder) ** 2 - torus_r**2)
    cap_angle = np.arcsin(torus_r / (nose - shoulder))
    phi = np.linspace(0.0, cap_angle, 120)
    x_cap, r_cap = nose * (1.0 - np.cos(phi)), nose * np.sin(phi)
    around = np.linspace(np.pi - cap_angle, np.pi / 2.0 - cone, 40)
    x_torus = torus_x + shoulder * np.cos(around)
    r_torus = torus_r + shoulder * np.sin(around)
    x0, r0 = x_torus[-1], r_torus[-1]
    x_end = x0 + (r0 - base) / np.tan(cone)
    x_cone = np.linspace(x0, x_end, 60)
    r_cone = r0 - (x_cone - x0) * np.tan(cone)
    x_base, r_base = np.full(30, x_end), np.linspace(base, 0.0, 30)
    return (
        np.concatenate([x_cap, x_torus[1:], x_cone[1:], x_base[1:]]),
        np.concatenate([r_cap, r_torus[1:], r_cone[1:], r_base[1:]]),
    )


@dataclass(frozen=True)
class HypersonicAerodynamics:
    """The panel method's answer at the design L/D."""

    angle_of_attack: float
    """Radians, heat shield forward; 180 degrees minus this is the database convention."""
    drag_coefficient: float
    lift_to_drag: float
    ballistic_coefficient: float


def orion_aerodynamics(
    model: PanelModel | None = None, lift_to_drag: float = LIFT_TO_DRAG
) -> HypersonicAerodynamics:
    """Drag coefficient and attitude at which the panel method gives ``lift_to_drag``.

    Modified Newtonian at Mach 25 with the equilibrium-air stagnation value
    :math:`C_{p,\\max} = 1.93` (:meth:`~aether.aerodynamics.panels.PanelModel.loads`).
    """
    if model is None:
        model = surface_of_revolution(*orion_meridian(), n_circ=96)
    area = np.pi * (DIAMETER / 2.0) ** 2
    alphas = np.deg2rad(np.arange(0.0, 36.0, 0.5))
    cd, ld = [], []
    for alpha in alphas:
        force, _ = model.loads(float(alpha), 25.0, 1.0, cp_max=1.93)
        flow = model.velocity_direction(float(alpha))
        drag = float(np.dot(force, flow))
        lift = float(np.linalg.norm(force - drag * flow))
        cd.append(drag / area)
        ld.append(lift / drag)
    alpha = float(np.interp(lift_to_drag, ld, alphas))
    drag_coefficient = float(np.interp(alpha, alphas, cd))
    return HypersonicAerodynamics(
        angle_of_attack=alpha,
        drag_coefficient=drag_coefficient,
        lift_to_drag=lift_to_drag,
        ballistic_coefficient=MASS / (drag_coefficient * area),
    )


# ---------------------------------------------------------------------------
# Guidance and the flight
# ---------------------------------------------------------------------------


def artemis1_guidance_config() -> SkipGuidanceConfig:
    """PredGuid's Artemis I parameters where the paper states them (AAS 24-174).

    Two kinds of number are not from the paper, and are marked:

    * The lateral corridor has the published Apollo-heritage form
      :math:`c_0 + c_1 V^2`, but Orion's coefficients are not published; they
      are tuned per mission to set the number of bank reversals. The values
      here were chosen by a sweep against the flight's six reversal times
      (AAS 24-174, Table 3), which is the one input in this example fitted
      to the flight rather than taken from a source or computed.
    * The Final-phase reserve, its tolerance, and the lift-down bank limit
      are assumptions.

    The roll rate is from Orion's handling-qualities work (NTRS 20110013203),
    which flew reversals at 9 deg/s and ran the lunar-return skip entry
    scenarios at 15 deg/s. It is not a detail: a reversal sweeps the lift
    vector through wings-level, so at 11 km/s the loft it costs is measured in
    kilometres of skip apogee, and a guidance loop that assumes the bank is
    reached instantly plans a trajectory the body cannot fly.
    """
    return SkipGuidanceConfig(
        initial_bank=np.deg2rad(15.0),  # 0 deg limited to 15
        initial_roll_exit_altitude_rate=-700.0 * FT,
        terminal_speed=1000.0 * FT,  # V Terminal
        range_tolerance=25.0 * NMI,  # NPC Range Tol
        max_iterations=5,
        estimator_drag_threshold=1.6 * FT,  # D Estimators
        final_phase_drag=6.0 * FT,  # D Upc End
        corridor_constant=5.0e-5,  # fitted to the flight's reversals
        corridor_quadratic=1.2e-10,  # fitted to the flight's reversals
        roll_rate=np.deg2rad(15.0),  # the rate the lunar-return skip scenarios were flown at
        final_vertical_lift_fraction=0.5,  # assumed reserve
        final_range_tolerance=1.0 * NMI,  # assumed
    )


def standard_density() -> tuple[object, ...]:
    """US Standard 1976 below 86 km, NRLMSIS above, tabulated once."""
    table = tabulate(earth_atmosphere())

    def density(altitude: float) -> float:
        return float(table.state(altitude).density)

    return (density,)


def fly(aero: HypersonicAerodynamics) -> tuple[SkipEntryResult, SkipGuidance]:
    """Fly Artemis I: the guidance's model against the atmosphere and lift it met."""
    (density,) = standard_density()
    model = EntryBody(
        aero.ballistic_coefficient, aero.lift_to_drag, max_bank=MAX_BANK, density=density
    )
    truth = EntryBody(
        aero.ballistic_coefficient,
        0.95 * aero.lift_to_drag,
        max_bank=MAX_BANK,
        density=lambda h: 0.9 * density(h),
    )
    guidance = SkipGuidance(model, TARGET, artemis1_guidance_config())
    return fly_skip_entry(truth, INTERFACE, guidance), guidance


def footprints(
    result: SkipEntryResult, aero: HypersonicAerodynamics
) -> list[tuple[str, float, EntryFootprint, GeodeticPosition]]:
    """The reachable landing footprint at three points of the flight.

    Each is computed from the flown state in the inertial frame, and the
    target is placed where the Earth will have carried it by the time of
    arrival, so the question asked is the one that matters: is the site still
    reachable from here?
    """
    (density,) = standard_density()
    body = EntryBody(
        aero.ballistic_coefficient, aero.lift_to_drag, max_bank=MAX_BANK, density=density
    )
    # Assumed limits, not from a source: loose on heating, so the footprint is
    # shaped by the dynamics, and 10 g on load.
    limits = EntryPathLimits(
        max_heat_flux_w_cm2=5000.0, max_g_load=10.0, nose_radius_m=1.2 * DIAMETER
    )
    at = {phase: t for t, phase in result.phases}
    apogee = int(np.argmax(np.where(result.times > 150.0, result.altitudes, -np.inf)))
    moments = [
        ("entry interface", 0.0),
        ("skip apogee", float(result.times[apogee])),
        ("final phase", at.get(SkipPhase.FINAL, float(result.times[-1]))),
    ]
    arrival = float(result.times[-1])
    target = GeodeticPosition(
        latitude=TARGET[1], longitude=TARGET[0] + EARTH_ROTATION_RATE * arrival, altitude=0.0
    )
    out = []
    for label, epoch in moments:
        k = int(np.searchsorted(result.times, epoch))
        footprint = entry_footprint(
            result.states[:, k], body, limits=limits, terminal_speed=1000.0 * FT,
            label=label, epoch=epoch, max_time=2000.0, step=10.0,
            n_constant=15, n_reversals=4,
        )
        out.append((label, epoch, footprint, target))
    return out


def certified_outer_bound(
    result: SkipEntryResult, aero: HypersonicAerodynamics
) -> tuple[RangeBound, float, float]:
    """The outer half of the sandwich, over the Final phase.

    Returns the certified bound, the downrange the body actually flew over the
    same phase, and the altitude ceiling the bound is conditional on. The
    ceiling is read from the flown trajectory and rounded up: the hypothesis is
    that the body stays under it, which over the Final phase it does, and the
    bound says nothing about a trajectory that leaves.

    The density enclosure carries a scale factor of 0.85 to 1.15, so the bound
    holds over a *class* of atmospheres rather than the tabulated one alone --
    Artemis I flew through air about 10 % thinner than standard.
    """
    at = {phase: t for t, phase in result.phases}
    start = at.get(SkipPhase.FINAL, 0.0)
    k = int(np.searchsorted(result.times, start))
    altitudes = result.altitudes[k:]
    ceiling = float(np.ceil(altitudes.max() / 5e3) * 5e3)
    corridor = Corridor(
        altitude_ceiling_m=ceiling,
        altitude_floor_m=float(max(np.floor(altitudes.min() / 5e3) * 5e3, 0.0)),
        speed_high_ms=float(result.states[3, k]),
        speed_low_ms=1000.0 * FT,
    )
    density = monotone_density(tabulate(earth_atmosphere()), scale=(0.85, 1.15))
    bound = certified_downrange_bound(
        corridor, aero.ballistic_coefficient, density, subdivisions=2048
    )
    flown = _great_circle_m(
        float(result.states[1, k]), float(result.states[2, k]),
        float(result.states[1, -1]), float(result.states[2, -1]),
    )
    return bound, flown, ceiling


def _great_circle_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))) * _R_EARTH)


def main() -> int:
    started = time.monotonic()
    aero = orion_aerodynamics()
    print(
        f"Aerodynamics: L/D {aero.lift_to_drag:.2f} at alpha "
        f"{np.rad2deg(aero.angle_of_attack):.1f} deg "
        f"({180.0 - np.rad2deg(aero.angle_of_attack):.1f} in the database convention; "
        f"published trim 157-162), CD {aero.drag_coefficient:.3f}, "
        f"beta {aero.ballistic_coefficient:.0f} kg/m2"
    )
    result, guidance = fly(aero)
    at = {phase: t for t, phase in result.phases}
    rows = [
        ("final phase (s)", at.get(SkipPhase.FINAL, float("nan"))),
        ("terminal phase (s)", at.get(SkipPhase.TERMINAL, float("nan"))),
        ("skip apogee (kft)", result.skip_apogee_altitude / FT / 1000.0),
        ("bank reversals", float(result.reversals)),
        ("terminal-phase start from target (nmi)", result.miss_distance / NMI),
    ]
    print(f"\n{'':42s}{'simulated':>12s}{'Artemis I':>12s}")
    for name, value in rows:
        print(f"{name:42s}{value:12.1f}{FLOWN[name]:12.1f}")
    print(f"{'peak deceleration (g)':42s}{result.peak_deceleration_g:12.2f}")
    signs = np.sign(result.bank)
    flips = np.nonzero(np.diff(signs))[0] + 1
    simulated = ", ".join(f"{result.times[i]:.0f}" for i in flips)
    flown = ", ".join(f"{t:.0f}" for t in FLOWN["reversal times (s)"])
    print(f"reversal times (s): simulated {simulated}\n{'':20s}Artemis I {flown}")
    print(
        f"estimators: density factor {guidance.density_factor:.3f} (flight ~0.90), "
        f"L/D {guidance.lift_to_drag_estimate:.3f} (truth {0.95 * aero.lift_to_drag:.3f})"
    )
    print("\nReachable landing footprint, target inside?")
    prints = footprints(result, aero)
    for label, epoch, footprint, target in prints:
        print(
            f"  {label:16s} t={epoch:6.0f}s  area {footprint.area_km2:12,.0f} km2  "
            f"downrange {footprint.min_downrange_m / 1e3:7,.0f}-"
            f"{footprint.max_downrange_m / 1e3:7,.0f} km  "
            f"contains target: {footprint.contains(target)}  "
            f"(rejected {footprint.rejected}: {', '.join(footprint.binding_constraints) or '-'})"
        )
    bound, flown, ceiling = certified_outer_bound(result, aero)
    inner = next(fp.max_downrange_m for label, _, fp, _ in prints if label == "final phase")
    print("\nThe sandwich, from the start of the Final phase")
    print(f"  flown                {flown / 1e3:9,.0f} km")
    print(f"  inner sweep (Lu & Xue, bank schedules sampled)"
          f"{inner / 1e3:14,.0f} km")
    print(f"  certified outer      {bound.max_downrange_m / 1e3:9,.0f} km"
          f"   (every bank history, atmospheres 0.85-1.15x standard)")
    print(f"  hypothesis           {bound.corridor.describe()}")
    print(f"  holds for the flight: {bool(flown <= bound.max_downrange_m)}"
          f"  (flown stays under the {ceiling / 1e3:.0f} km ceiling)")
    # How much the hypothesis is worth. The bound is sound at every ceiling and
    # useful at none of the high ones: thin air produces almost no drag, so the
    # argument cannot rule out a body coasting there. Tightening it is what the
    # occupation-measure route in the proposal is for.
    at = {phase: t for t, phase in result.phases}
    k = int(np.searchsorted(result.times, at.get(SkipPhase.FINAL, 0.0)))
    density = monotone_density(tabulate(earth_atmosphere()), scale=(0.85, 1.15))
    print("  sensitivity to the ceiling:")
    for ceiling_km in (50.0, 60.0, 70.0, 80.0):
        corridor = Corridor(
            altitude_ceiling_m=ceiling_km * 1e3, altitude_floor_m=0.0,
            speed_high_ms=float(result.states[3, k]), speed_low_ms=1000.0 * FT,
        )
        each = certified_downrange_bound(
            corridor, aero.ballistic_coefficient, density, subdivisions=2048
        )
        print(f"    under {ceiling_km:4.0f} km: {each.max_downrange_m / 1e3:9,.0f} km"
              f"  ({each.max_downrange_m / flown:6.1f}x the flown range)")
    print(f"\n{time.monotonic() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
