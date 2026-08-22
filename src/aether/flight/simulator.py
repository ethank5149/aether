"""Coupled single-trajectory simulator (Paper I, §3.5; Paper II, §7.2).

This is the assembly the whole framework exists to support: rigid-body
translation and attitude, the null-space-reduced structural block, and
the charring thermal block on its Landau grid, advanced as **one system
of ODEs by one integrator** from exo-atmospheric coast through
atmospheric entry, with no phase handoff and no remesh.

Three couplings are closed here rather than asserted:

1. **Aerothermal → thermal → aerothermal.** Stagnation heating drives
   the surface energy balance, which drives recession, which grows
   :math:`R_{\\mathrm{eff}}`, which *reduces* convective heating as
   :math:`R_{\\mathrm{eff}}^{-1/2}` (Paper II, Eq. 4.2). Nose blunting is
   self-limiting, and capturing that requires the recession to feed back
   within the same time step — which it does, because both live in one
   right-hand side.
2. **Aerodynamic → structural.** Dynamic pressure drives the reduced
   modal coordinates through the mode shapes.
3. **Regime → regime.** The atmosphere decays smoothly, so the
   aerodynamic and aerothermal terms become numerically negligible above
   the sensible atmosphere *without any branch being taken*
   (Paper II, §7.2). There is no `if altitude > ...` anywhere in the
   right-hand side, and the integration-invariance test asserts it.

The state dimension is fixed by :class:`~aether.flight.state.StateLayout`
at construction and cannot change during flight. That is the property
the batching argument of Paper I §5.2 rests on.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace

import numpy as np
import scipy.integrate
from numpy.typing import ArrayLike, NDArray

from aether.aerodynamics.tables import AeroTable
from aether.aerothermal import sutton_graves
from aether.atmosphere.model import (
    TabulatedAtmosphere,
    earth_atmosphere,
    tabulate,
)
from aether.dynamics import quaternion_derivative
from aether.flight.propulsion import Burn, thrust_direction
from aether.flight.state import GlobalState, StateLayout

# Submodule import, not `from aether.orbital import ...`: the latter needs the
# name as an attribute on a package that may still be executing its __init__,
# and orbital -> guidance -> systems -> flight -> orbital is a real cycle.
from aether.orbital.gravity import EARTH, GravityModel, gravitational_acceleration
from aether.spectral import ChebyshevGrid
from aether.structures import (
    ModalSolution,
    assemble_beam,
    project_free_free,
    solve_free_free_modes,
    uniform_profile,
)
from aether.thermal import (
    CharringMaterial,
    CharringThermalSolver,
    LandauFrame,
    ThermalState,
    demo_material,
)
from aether.thermal.surface import STEFAN_BOLTZMANN

__all__ = ["FlightConfiguration", "FlightResult", "FlightSimulator"]

_FloatArray = NDArray[np.float64]

#: Exponential atmosphere: reference density (kg/m³) and scale height (m).
_RHO0 = 1.225
_H_SCALE = 8500.0
#: Effective heat of ablation for the refractory surface model (J/kg).
_ABLATION_ENTHALPY = 2.0e7


@dataclass(frozen=True)
class FlightConfiguration:
    """Vehicle and discretization parameters for the coupled run."""

    length: float = 6.0
    """Structural length (m)."""
    beam_order: int = 16
    """Chebyshev order :math:`N` of the structural grid."""
    n_modes: int = 6
    """Retained structural modes (rigid modes included)."""
    thermal_order: int = 10
    """Chebyshev order of the Landau thermal grid."""
    tps_thickness: float = 0.05
    """TPS stack thickness (m)."""
    flexural_rigidity: float = 4.0e7
    """Uniform :math:`EI` (N·m²)."""
    mass_per_length: float = 220.0
    """Uniform :math:`m(x)` (kg/m)."""
    ballistic_coefficient: float = 6000.0
    """:math:`m/(C_D A)` (kg/m²)."""
    nose_radius: float = 0.30
    """Initial effective nose radius (m)."""
    reference_area: float = 1.2
    """Aerodynamic reference area (m²)."""
    drag_area: float | None = None
    """Effective :math:`C_D A` (m²), or ``None`` to use a fixed ballistic
    coefficient.

    The two are alternatives, not duplicates. An unpowered vehicle has a
    constant mass, so a constant :math:`\beta = m/(C_D A)` is exact and is
    what every entry text tabulates. A *powered* one does not: a launcher
    burns fifteen times its burnout mass, and holding :math:`\beta` fixed
    through that understates the drag deceleration on the light, fast end
    of the ascent by the same factor. Set ``drag_area`` for any mission
    that varies mass, and drag becomes :math:`qC_DA/m` with the mass the
    integrator is actually carrying."""
    aero_table: AeroTable | None = None
    """A coefficient table supplying :math:`C_DA` as a function of Mach.

    The third and best of the three. ``ballistic_coefficient`` is one number,
    ``drag_area`` is one number, and the real thing is a curve: on this
    vehicle :math:`C_A` runs from 1.62 at Mach 1.2 to 0.87 at Mach 25, so a
    single constant is wrong by a factor of two somewhere on every ascent.
    Set this and the drag area is read from the table at the local Mach
    number, which needs a speed of sound and therefore also selects the
    layered atmosphere over the exponential one.

    Takes precedence over both of the others when set."""
    atmosphere: TabulatedAtmosphere | None = None
    """Ambient gas model. ``None`` keeps the exponential atmosphere.

    Only consulted when :attr:`aero_table` is set, because only then does
    anything need a temperature. It must be a *tabulated* atmosphere: the
    right-hand side is evaluated tens of thousands of times per trajectory
    and the layered model calls into NRLMSIS on every one of them."""
    baumgarte_gain: float = 1.0
    """:math:`k_q` of the quaternion stabilization."""
    structural_damping: float = 0.02
    """Modal damping ratio."""
    thermal_baumgarte_gain: float = 50.0
    """Baumgarte gain (1/s) removing thermal boundary-constraint drift."""
    wall_emissivity: float = 0.85
    heat_load_fraction: float = 0.15
    """Fraction of stagnation heating reaching the modelled TPS station."""

    def __post_init__(self) -> None:
        for name in (
            "length", "tps_thickness", "flexural_rigidity", "mass_per_length",
            "ballistic_coefficient", "nose_radius", "reference_area",
        ):
            val = float(getattr(self, name))
            if not (np.isfinite(val) and val > 0.0):
                raise ValueError(f"{name} must be finite and > 0, got {val}")
        if self.beam_order < 6:
            raise ValueError(f"beam_order must be >= 6, got {self.beam_order}")
        if self.thermal_order < 4:
            raise ValueError(f"thermal_order must be >= 4, got {self.thermal_order}")
        if self.n_modes < 1:
            raise ValueError(f"n_modes must be >= 1, got {self.n_modes}")
        if not 0.0 <= self.structural_damping < 1.0:
            raise ValueError("structural_damping must be in [0, 1)")
        if self.drag_area is not None and not (
            np.isfinite(self.drag_area) and self.drag_area > 0.0
        ):
            raise ValueError(f"drag_area must be finite and > 0, got {self.drag_area}")
        if not 0.0 < self.heat_load_fraction <= 1.0:
            raise ValueError("heat_load_fraction must be in (0, 1]")
        if not (np.isfinite(self.thermal_baumgarte_gain)
                and self.thermal_baumgarte_gain > 0.0):
            raise ValueError("thermal_baumgarte_gain must be finite and > 0")


@dataclass(frozen=True)
class FlightResult:
    """Trajectory output with the diagnostics the coupling claims need."""

    times: _FloatArray = field(repr=False)
    states: _FloatArray = field(repr=False)
    """Shape ``(state_size, n_times)``."""
    layout: StateLayout
    n_rhs_evaluations: int
    wall_time: float
    effective_radius: _FloatArray = field(repr=False)
    stagnation_heat_flux: _FloatArray = field(repr=False)
    dynamic_pressure: _FloatArray = field(repr=False)

    @property
    def altitude(self) -> _FloatArray:
        r = np.linalg.norm(self.states[self.layout.position, :], axis=0)
        return np.asarray(r - EARTH.radius)

    @property
    def quaternion_norm_error(self) -> _FloatArray:
        q = self.states[self.layout.quaternion, :]
        return np.asarray(np.abs(np.linalg.norm(q, axis=0) - 1.0))

    @property
    def recession(self) -> _FloatArray:
        return np.asarray(self.states[self.layout.recession, :])

    @property
    def surface_temperature(self) -> _FloatArray:
        return np.asarray(self.states[self.layout.temperature, :][-1, :])


class FlightSimulator:
    """One ODE, one integrator, fixed dimension, all regimes."""

    def __init__(
        self,
        config: FlightConfiguration | None = None,
        material: CharringMaterial | None = None,
        gravity: GravityModel = EARTH,
    ) -> None:
        self._cfg = config or FlightConfiguration()
        self._material = material or demo_material()
        self._gravity = gravity
        # A coefficient table indexed by Mach needs a speed of sound, which
        # the exponential atmosphere does not have. Selecting one implies the
        # other, and building the default here rather than in the dataclass
        # keeps a bare FlightConfiguration free of a 4000-point table nobody
        # asked for.
        self._atmosphere: TabulatedAtmosphere | None = None
        if self._cfg.aero_table is not None:
            self._atmosphere = self._cfg.atmosphere or tabulate(earth_atmosphere())
        elif self._cfg.atmosphere is not None:
            self._atmosphere = self._cfg.atmosphere

        # --- structural kernel, assembled once and held fixed
        beam_grid = ChebyshevGrid(self._cfg.beam_order, interval=(0.0, self._cfg.length))
        beam = assemble_beam(
            beam_grid,
            uniform_profile(self._cfg.flexural_rigidity, self._cfg.mass_per_length),
        )
        self._modes: ModalSolution = solve_free_free_modes(project_free_free(beam))
        # Only the ELASTIC modes belong here. The free-free spectrum's three
        # zero-frequency modes are rigid translation and rotation, which the
        # 6-DOF block already carries: integrating them here too double-counts
        # rigid motion, and because they have no restoring term the modal
        # coordinate then grows without bound under any steady load. The
        # structural state of Paper I Eq. (3.20) is the elastic deformation.
        n_rigid = self._modes.n_rigid
        available = self._modes.frequencies.size - n_rigid
        n_modes = min(self._cfg.n_modes, available)
        if n_modes < 1:
            raise ValueError(
                f"beam_order {self._cfg.beam_order} yields no elastic modes"
            )
        self._elastic_omega = np.array(
            self._modes.frequencies[n_rigid : n_rigid + n_modes], dtype=np.float64
        )
        elastic_shapes = self._modes.modes_full[:, n_rigid : n_rigid + n_modes]
        self._beam_grid = beam_grid
        # modal participation of a uniform transverse load, computed once
        self._modal_load_gain = elastic_shapes.T @ beam_grid.weights

        # --- thermal kernel on the fixed Landau grid
        thermal_grid = ChebyshevGrid(
            self._cfg.thermal_order, interval=(0.0, 1.0), max_derivative=2
        )
        self._frame = LandauFrame(total_thickness=self._cfg.tps_thickness)
        self._thermal = CharringThermalSolver(thermal_grid, self._material, self._frame)
        self._thermal_grid = thermal_grid

        self._thermal_baumgarte = float(self._cfg.thermal_baumgarte_gain)
        self._layout = StateLayout(
            n_modes=n_modes, n_thermal=thermal_grid.size
        )

    @property
    def config(self) -> FlightConfiguration:
        return self._cfg

    @property
    def gravity(self) -> GravityModel:
        """The gravity model this simulator integrates against.

        Exposed because anything reading a result needs the same body
        radius the states were produced with — a visualisation that took
        its radius from somewhere else would place the surface in a
        different spot from the trajectory.
        """
        return self._gravity

    @property
    def layout(self) -> StateLayout:
        return self._layout

    @property
    def modal_frequencies(self) -> _FloatArray:
        """Retained *elastic* natural frequencies (rad/s); rigid-body modes
        live in the 6-DOF block, not here."""
        return self._elastic_omega

    # ------------------------------------------------------------ initial state
    def state_at(
        self,
        position: ArrayLike,
        velocity: ArrayLike,
        mass: float = 1500.0,
        wall_temperature: float = 300.0,
    ) -> _FloatArray:
        """Pack a state at an arbitrary inertial position and velocity.

        :meth:`initial_state` places the vehicle on the ``+x`` axis, which
        is fine for a single entry study and useless for a mission that
        launches from a named site on a named bearing. This takes the
        vectors directly.
        """
        virgin = np.array(
            [c.virgin_density for c in self._material.components]
        )[:, np.newaxis] * np.ones((1, self._layout.n_thermal))
        state = GlobalState(
            position=np.asarray(position, dtype=np.float64),
            velocity=np.asarray(velocity, dtype=np.float64),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_rate=np.zeros(3),
            mass=float(mass),
            modal_displacement=np.zeros(self._layout.n_modes),
            modal_velocity=np.zeros(self._layout.n_modes),
            temperature=np.full(self._layout.n_thermal, float(wall_temperature)),
            densities=virgin,
            recession=0.0,
        )
        return state.pack(self._layout)

    def initial_state(
        self,
        altitude: float,
        speed: float,
        flight_path_angle: float,
        wall_temperature: float = 300.0,
        mass: float = 1500.0,
    ) -> _FloatArray:
        """Build a launch state at the given altitude and flight path."""
        radius = self._gravity.radius + float(altitude)
        position = np.array([radius, 0.0, 0.0])
        gamma = float(flight_path_angle)
        velocity = float(speed) * np.array([np.sin(gamma), np.cos(gamma), 0.0])
        virgin = np.array(
            [c.virgin_density for c in self._material.components]
        )[:, np.newaxis] * np.ones((1, self._layout.n_thermal))
        state = GlobalState(
            position=position,
            velocity=velocity,
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_rate=np.zeros(3),
            mass=float(mass),
            modal_displacement=np.zeros(self._layout.n_modes),
            modal_velocity=np.zeros(self._layout.n_modes),
            temperature=np.full(self._layout.n_thermal, float(wall_temperature)),
            densities=virgin,
            recession=0.0,
        )
        return state.pack(self._layout)

    # -------------------------------------------------------------------- RHS
    def effective_radius(self, recession: float) -> float:
        """:math:`R_{\\mathrm{eff}}` grown by recession — the feedback that
        makes nose blunting self-limiting (Paper II, §4.1)."""
        return self._cfg.nose_radius + float(recession)

    def _close_thermal_boundaries(
        self,
        temperature: _FloatArray,
        pde_rate: _FloatArray,
        net_flux: float,
        thickness: float,
        wall: float,
    ) -> _FloatArray:
        """Impose the thermal boundary conditions on the collocation grid.

        A collocated diffusion operator evaluated at *every* node carries
        no boundary conditions and is ill-posed — the manufactured-solution
        verification supplied both boundary rates explicitly, which is
        exactly why it did not expose this. Flight needs the physical
        pair: adiabatic at the back face and the surface energy balance at
        the wall,

        .. math::

            \\left.\\frac{\\partial T}{\\partial \\eta}\\right|_{\\eta=1} = 0,
            \\qquad
            -\\frac{k}{\\ell}
            \\left.\\frac{\\partial T}{\\partial \\eta}\\right|_{\\eta=0}
            = \\dot q_{\\mathrm{net}} .

        Both are algebraic constraints on a state the integrator advances,
        so they are differentiated in time to give rates for the two
        boundary nodes, and the resulting constraint drift is removed by
        **Baumgarte stabilization** — the same device Paper I applies to
        the quaternion norm, and for the same reason: it keeps the
        correction inside the right-hand side where the error controller
        can see it, instead of projecting after each step.
        """
        d1 = self._thermal_grid.diffmat(1)
        k_wall = float(self._material.conductivity.value(wall, 1.0))
        target_surface = -net_flux * thickness / max(k_wall, 1e-12)

        residual_back = float(d1[0, :] @ temperature)
        residual_surface = float(d1[-1, :] @ temperature) - target_surface

        interior_rate = pde_rate[1:-1]
        rhs_back = -(d1[0, 1:-1] @ interior_rate) - self._thermal_baumgarte * residual_back
        rhs_surface = (
            -(d1[-1, 1:-1] @ interior_rate)
            - self._thermal_baumgarte * residual_surface
        )
        matrix = np.array(
            [[d1[0, 0], d1[0, -1]], [d1[-1, 0], d1[-1, -1]]], dtype=np.float64
        )
        boundary_rates = np.linalg.solve(matrix, np.array([rhs_back, rhs_surface]))

        rate = np.array(pde_rate, copy=True)
        rate[0] = boundary_rates[0]
        rate[-1] = boundary_rates[1]
        return rate

    @contextmanager
    def using_aero_table(self, table: AeroTable | None) -> Iterator[None]:
        """Fly with ``table`` as the drag model for the duration of the block.

        A staged vehicle is not one aerodynamic body: the stack on the pad, the
        stack with its boosters gone, and the re-entry body alone have different
        :math:`C_D A` curves, and the mass step at separation is only half of what
        changes there. This swaps the coefficient table in place — the structural
        and thermal kernels, which are expensive to assemble and unaffected by it,
        are left alone — and restores the previous one on the way out.

        A table needs a speed of sound, so a tabulated atmosphere is selected here
        if the configuration did not already have one, exactly as the constructor
        does. ``None`` is a no-op, so a caller can pass a per-phase table that may
        legitimately be absent without branching.
        """
        if table is None:
            yield
            return
        previous_cfg, previous_atmosphere = self._cfg, self._atmosphere
        self._cfg = replace(self._cfg, aero_table=table)
        if self._atmosphere is None:
            self._atmosphere = self._cfg.atmosphere or tabulate(earth_atmosphere())
        try:
            yield
        finally:
            self._cfg, self._atmosphere = previous_cfg, previous_atmosphere

    def rhs(self, t: float, y: _FloatArray, burn: Burn | None = None) -> _FloatArray:
        """Right-hand side of the complete coupled system.

        Contains **no branch on flight regime**. The atmospheric density
        decays smoothly, so every aerodynamic and aerothermal term
        vanishes numerically above the sensible atmosphere on its own.

        Parameters
        ----------
        burn:
            Optional thrust arc active over this whole call. Thrust is not
            gated on ``t`` inside the right-hand side: switching it on or
            off is a genuine discontinuity, and an implicit integrator
            asked to step across one either crawls or smears it. Segments
            are integrated separately instead, so within any one solve the
            thrust state is constant and ``t`` is measured from ignition.
        """
        cfg = self._cfg
        layout = self._layout
        state = GlobalState.unpack(y, layout)
        out = np.zeros(layout.size)

        radius = float(np.linalg.norm(state.position))
        altitude = radius - self._gravity.radius
        speed = state.speed
        if self._atmosphere is None:
            # smooth, branch-free: exp underflows to zero far above the atmosphere
            density = _RHO0 * np.exp(-max(altitude, 0.0) / _H_SCALE)
            mach = 0.0
        else:
            ambient = self._atmosphere.state(max(altitude, 0.0))
            density = float(ambient.density)
            mach = speed / float(ambient.speed_of_sound)
        q_dyn = 0.5 * density * speed**2

        # --- translation: J2 gravity plus drag along the relative velocity
        accel = gravitational_acceleration(state.position, self._gravity)
        if speed > 0.0:
            if cfg.aero_table is not None:
                drag_area = cfg.aero_table.drag_area(mach)
                deceleration = q_dyn * drag_area / max(float(state.mass), 1e-6)
            elif cfg.drag_area is None:
                deceleration = q_dyn / cfg.ballistic_coefficient
            else:
                deceleration = q_dyn * cfg.drag_area / max(float(state.mass), 1e-6)
            accel = accel - deceleration * (state.velocity / speed)
        # --- thrust, when an arc is active
        #
        # Mass depletion is carried, so the acceleration grows through the
        # burn as it does on a real stage: a 6 t vehicle at 300 kN starts
        # at 5 g and finishes heavier in acceleration than it started.
        # Treating the deorbit as an impulse would move the vehicle 70 km
        # less than the integration says it does.
        if burn is not None:
            mass = max(float(state.mass), 1e-6)
            direction = thrust_direction(burn, t, state.position, state.velocity)
            accel = accel + (burn.thrust / mass) * direction
            out[layout.mass] = -burn.mass_flow
        else:
            out[layout.mass] = 0.0  # unpowered coast

        out[layout.position] = state.velocity
        out[layout.velocity] = accel

        # --- attitude with Baumgarte stabilization
        out[layout.quaternion] = quaternion_derivative(
            state.quaternion, state.angular_rate, cfg.baumgarte_gain
        )
        out[layout.angular_rate] = 0.0  # torque-free in this configuration

        # --- structure: elastic modal oscillators driven by the aero load
        omega = self._elastic_omega
        load = q_dyn * cfg.reference_area / max(cfg.length, 1e-12)
        modal_force = self._modal_load_gain * load
        out[layout.modal_displacement] = state.modal_velocity
        out[layout.modal_velocity] = (
            modal_force
            - 2.0 * cfg.structural_damping * omega * state.modal_velocity
            - omega**2 * state.modal_displacement
        )

        # --- aerothermal: stagnation heating with the recession feedback
        #
        # Clamped at zero at the boundary between the integrator and the
        # physics. Recession is monotone non-decreasing by construction —
        # the rate below is clamped at zero because oxidative recession is
        # irreversible — so a negative value is always round-off, and an
        # implicit solver produces them: a Newton trial state arrived here
        # at -4e-16 m and the Landau frame rightly refused it, aborting a
        # solve that was converging. The validator is correct; applying it
        # to trial states rather than to real ones was not.
        recession = max(float(state.recession), 0.0)
        r_eff = self.effective_radius(recession)
        q_stag = (
            float(sutton_graves(density, r_eff, speed)) if density > 0.0 and speed > 0.0
            else 0.0
        )
        q_surface = cfg.heat_load_fraction * q_stag

        # --- thermal block on the fixed Landau grid
        wall = state.surface_temperature
        reradiated = cfg.wall_emissivity * STEFAN_BOLTZMANN * wall**4
        net_flux = q_surface - reradiated
        thermal_state = ThermalState(
            temperature=state.temperature,
            partial_densities=state.densities,
            recession=recession,
        )
        # recession rate from the net surface flux over the ablation enthalpy,
        # clamped at zero because oxidative recession is irreversible
        sdot = max(net_flux, 0.0) / (
            self._material.char_bulk_density * _ABLATION_ENTHALPY
        )
        thermal_rhs = self._thermal.rhs(
            t, self._thermal.pack(thermal_state), lambda _t, _s: sdot
        )
        unpacked = self._thermal.unpack(thermal_rhs)
        temperature_rate = self._close_thermal_boundaries(
            state.temperature, np.asarray(unpacked.temperature), net_flux,
            self._frame.thickness(recession), wall,
        )

        out[layout.temperature] = temperature_rate
        out[layout.densities] = unpacked.partial_densities.reshape(-1)
        out[layout.recession] = sdot
        return out

    # -------------------------------------------------------------- propagate
    def absolute_tolerances(self) -> _FloatArray:
        """Per-component absolute tolerances for the coupled state.

        The global state spans nine orders of magnitude in units — ECI
        position in millions of metres against elastic modal coordinates
        that a free-free beam keeps near zero, because an elastic mode is
        mass-orthogonal to rigid translation and a uniform load therefore
        barely excites it. A single scalar ``atol`` cannot serve both: set
        it for position and the modal block is unconstrained noise; set it
        for the modal block and the integrator chases rounding in the
        orbital state. Measured, the difference is three orders of
        magnitude in right-hand-side evaluations. Per-block tolerances are
        the standard remedy for a multiphysics state and are what this
        simulator uses by default.
        """
        layout = self._layout
        atol = np.empty(layout.size)
        atol[layout.position] = 1.0e-3       # m
        atol[layout.velocity] = 1.0e-6       # m/s
        atol[layout.quaternion] = 1.0e-10
        atol[layout.angular_rate] = 1.0e-10  # rad/s
        atol[layout.mass] = 1.0e-6           # kg
        atol[layout.modal_displacement] = 1.0e-9
        atol[layout.modal_velocity] = 1.0e-7
        atol[layout.temperature] = 1.0e-4    # K
        atol[layout.densities] = 1.0e-6      # kg/m^3
        atol[layout.recession] = 1.0e-12     # m
        return atol

    def propagate(
        self,
        initial_state: ArrayLike,
        duration: float,
        n_output: int = 201,
        rtol: float = 1e-8,
        atol: ArrayLike | None = None,
        method: str = "BDF",
    ) -> FlightResult:
        """Advance the coupled system with a single integrator call.

        The default is BDF, and the choice is not incidental. The
        structural block makes the coupled system stiff — Paper I,
        Prop. 2 — and a *fully* implicit method removes the constraint
        entirely: measured cost is flat in the retained mode count
        (~400 right-hand-side evaluations whether the highest retained
        mode is 0 or 1432 rad/s). LSODA, which is supposed to detect
        stiffness and switch, does not switch here and needs three orders
        of magnitude more work. That is the same conclusion V3 reached
        for the structural block alone, now confirmed on the coupled
        system.
        """
        y0 = np.asarray(initial_state, dtype=np.float64)
        if y0.shape != (self._layout.size,):
            raise ValueError(
                f"initial_state must have shape ({self._layout.size},), got {y0.shape}"
            )
        if not (np.isfinite(duration) and duration > 0.0):
            raise ValueError(f"duration must be finite and > 0, got {duration}")

        tolerances = self.absolute_tolerances() if atol is None else np.asarray(atol)
        t_eval = np.linspace(0.0, float(duration), int(n_output))
        start = time.perf_counter()
        sol = scipy.integrate.solve_ivp(
            self.rhs, (0.0, float(duration)), y0, method=method,
            t_eval=t_eval, rtol=rtol, atol=tolerances,
        )
        wall = time.perf_counter() - start
        if not sol.success:
            raise RuntimeError(f"coupled propagation failed: {sol.message}")

        states = np.asarray(sol.y)
        recession = states[self._layout.recession, :]
        r_eff = np.asarray([self.effective_radius(float(s)) for s in recession])
        radius = np.linalg.norm(states[self._layout.position, :], axis=0)
        altitude = radius - self._gravity.radius
        density = _RHO0 * np.exp(-np.maximum(altitude, 0.0) / _H_SCALE)
        speed = np.linalg.norm(states[self._layout.velocity, :], axis=0)
        q_stag = np.where(
            (density > 0.0) & (speed > 0.0),
            self._cfg.heat_load_fraction
            * np.asarray(sutton_graves(np.maximum(density, 1e-300), r_eff,
                                       np.maximum(speed, 1e-300))),
            0.0,
        )
        return FlightResult(
            times=np.asarray(sol.t),
            states=states,
            layout=self._layout,
            n_rhs_evaluations=int(sol.nfev),
            wall_time=wall,
            effective_radius=r_eff,
            stagnation_heat_flux=np.asarray(q_stag),
            dynamic_pressure=np.asarray(0.5 * density * speed**2),
        )
