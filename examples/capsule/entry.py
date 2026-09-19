"""Capsule atmospheric entry — a minimal example of the coupled simulator.

Propagates a blunt-body capsule from the entry interface (80 km) through
peak heating and peak deceleration using the full coupled ODE: rigid-body
translation, Euler attitude dynamics, Chebyshev structural modes, and the
Landau-frame charring thermal solver advanced by one integrator with no
phase handoff.

The vehicle is deliberately generic — an axisymmetric capsule with
published Apollo-era proportions — so no controlled parameters appear.

Usage::

    python -m examples.capsule.entry          # from the repo root
    python examples/capsule/entry.py          # or directly

Produces altitude, speed, surface-temperature, and heat-flux histories
on stdout (and optionally as a plot if matplotlib is available).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from aether.aerodynamics.tables import AeroTable
from aether.atmosphere.model import earth_atmosphere, tabulate
from aether.flight.simulator import FlightConfiguration, FlightSimulator


def capsule_aero_table() -> AeroTable:
    """Blunt capsule coefficient table from published correlations.

    Axial-force coefficient is essentially the modified Newtonian value
    for a sphere-cone at low incidence.  Pitching moment gives mild
    static stability (C_m_alpha < 0), consistent with an aft centre of
    gravity.  Values are representative, not tuned to any specific
    programme.
    """
    mach = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 25.0])
    alpha_deg = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
    n_m, n_a = mach.size, alpha_deg.size

    ca_zero = np.array([0.40, 0.55, 0.65, 0.70, 0.72, 0.73, 0.73])
    axial = np.outer(ca_zero, np.ones(n_a))

    cn_slope = 0.015
    normal = np.outer(np.ones(n_m), cn_slope * alpha_deg)

    cm_slope = -0.005
    pitching_moment = np.outer(np.ones(n_m), cm_slope * alpha_deg)

    diameter = 3.91
    reference_area = 0.25 * np.pi * diameter**2

    return AeroTable(
        name="generic_capsule",
        mach=mach,
        alpha=alpha_deg,
        axial=axial,
        normal=normal,
        pitching_moment=pitching_moment,
        reference_area=reference_area,
        reference_length=diameter,
        solver="published_correlations",
        metadata={"geometry": "sphere-cone 25-deg half-angle, 3.91 m diameter"},
    )


def capsule_config(
    table: AeroTable | None = None,
    atmosphere: tabulate | None = None,
) -> FlightConfiguration:
    """Build the capsule flight configuration."""
    diameter = 3.91
    reference_area = 0.25 * np.pi * diameter**2
    mass = 5_400.0

    return FlightConfiguration(
        length=3.3,
        beam_order=8,
        n_modes=2,
        thermal_order=6,
        tps_thickness=0.04,
        flexural_rigidity=8.0e7,
        mass_per_length=mass / 3.3,
        ballistic_coefficient=mass / (0.70 * reference_area),
        nose_radius=4.69,
        reference_area=reference_area,
        aero_table=table,
        atmosphere=atmosphere,
        inertia=(2_500.0, 3_200.0, 3_200.0),
        reference_length=diameter,
        pitch_damping=-0.3,
        baumgarte_gain=1.0,
    )


def main() -> None:
    table = capsule_aero_table()
    atmosphere = tabulate(earth_atmosphere())
    config = capsule_config(table, atmosphere)
    sim = FlightSimulator(config)

    mass = 5_400.0
    entry_altitude = 80_000.0
    r0 = 6_371_000.0 + entry_altitude
    v_entry = 7_500.0
    gamma_entry = np.radians(-8.0)

    initial_state = sim.state_at(
        position=np.array([r0, 0.0, 0.0]),
        velocity=np.array([
            v_entry * np.sin(gamma_entry),
            v_entry * np.cos(gamma_entry),
            0.0,
        ]),
        mass=mass,
    )

    result = sim.propagate(initial_state, duration=120.0, n_output=241)

    alt_km = result.altitude / 1_000.0
    speed_kms = np.linalg.norm(result.states[sim.layout.velocity, :], axis=0) / 1_000.0
    q_dot = result.stagnation_heat_flux / 1e4
    t_wall = result.surface_temperature

    print(f"{'time_s':>8}  {'alt_km':>8}  {'speed_km/s':>10}  {'q_W/cm2':>8}  {'T_wall_K':>8}")
    for i in range(0, len(result.times), 20):
        print(
            f"{result.times[i]:8.1f}  {alt_km[i]:8.2f}  {speed_kms[i]:10.3f}  "
            f"{q_dot[i]:8.2f}  {t_wall[i]:8.1f}"
        )

    print("\n--- Summary ---")
    print(f"Peak dynamic pressure:  {np.max(result.dynamic_pressure) / 1e3:.1f} kPa")
    print(f"Peak heat flux:         {np.max(q_dot):.1f} W/cm^2")
    print(f"Peak wall temperature:  {np.max(t_wall):.0f} K")
    print(f"Final altitude:         {alt_km[-1]:.1f} km")
    print(f"Final speed:            {speed_kms[-1]:.3f} km/s")
    print(f"Total recession:        {float(result.recession[-1]) * 1e3:.2f} mm")
    print(f"RHS evaluations:        {result.n_rhs_evaluations}")
    print(f"Wall time:              {result.wall_time:.2f} s")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle("Capsule Entry — Coupled Simulation")

        axes[0, 0].plot(result.times, alt_km)
        axes[0, 0].set(xlabel="Time (s)", ylabel="Altitude (km)", title="Altitude")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(result.times, speed_kms)
        axes[0, 1].set(xlabel="Time (s)", ylabel="Speed (km/s)", title="Speed")
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(result.times, q_dot)
        axes[1, 0].set(xlabel="Time (s)", ylabel="Heat flux (W/cm²)", title="Stagnation heat flux")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(result.times, t_wall)
        axes[1, 1].set(xlabel="Time (s)", ylabel="Temperature (K)", title="Surface temperature")
        axes[1, 1].grid(True, alpha=0.3)

        fig.tight_layout()
        out = Path(__file__).parent / "capsule_entry.png"
        fig.savefig(out, dpi=150)
        print(f"\nPlot saved to {out}")
    except ImportError:
        pass


if __name__ == "__main__":
    sys.exit(main() or 0)
