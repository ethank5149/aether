"""Coast propagation and regime transition (Paper II, §7).

The framework advances boost, coast and reentry **within one
integration** rather than handing off between phase-specific
propagators, because handoffs reinterpolate the state at the boundary
and restart the integrator with no history. The mechanism is that
atmospheric terms are not switched off at the Kármán line: they decay
smoothly with the density model, so above roughly 100 km the
aerodynamic contribution to the right-hand side becomes numerically
negligible *without any branch being taken*
(:class:`RegimeTransition` measures exactly that).

Paper II's Remark 5 states the cost honestly rather than presenting the
choice as a pure gain: continuing to integrate a structural block that
carries no energy through coast leaves the step-size controller subject
to the :math:`\\mathcal{O}(N^{-4})` structural stability bound during a
phase whose rigid-body dynamics would permit far larger steps. The
suggested mitigation — freeze the structural block once its modal
energy falls below a threshold, reactivating on re-entry — reintroduces
a switch but confines it to a regime where the frozen state is provably
quiescent. :func:`compare_coast_strategies` measures which is
preferable, which is verification task II-V5.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.integrate
from numpy.typing import ArrayLike, NDArray

from aether.orbital.gravity import (
    EARTH,
    GravityModel,
    gravitational_acceleration,
    specific_angular_momentum_z,
    specific_energy,
)

__all__ = [
    "CoastResult",
    "RegimeTransition",
    "StrategyComparison",
    "compare_coast_strategies",
    "orbital_elements",
    "propagate_coast",
    "regime_transition_profile",
    "secular_rates",
]

_FloatArray = NDArray[np.float64]

#: Exponential-atmosphere reference density (kg/m³) and scale height (m).
_RHO0 = 1.225
_H_SCALE = 8500.0


@dataclass(frozen=True)
class CoastResult:
    """Outcome of a coast propagation."""

    times: _FloatArray = field(repr=False)
    states: _FloatArray = field(repr=False)
    """Shape ``(6, n_times)``: position then velocity."""
    energy: _FloatArray = field(repr=False)
    angular_momentum_z: _FloatArray = field(repr=False)
    n_steps: int
    wall_time: float

    @property
    def energy_drift(self) -> float:
        """Relative secular drift of specific energy over the propagation."""
        e0 = float(self.energy[0])
        return float(abs(self.energy[-1] - e0) / abs(e0))

    @property
    def angular_momentum_drift(self) -> float:
        h0 = float(self.angular_momentum_z[0])
        return float(abs(self.angular_momentum_z[-1] - h0) / abs(h0))

    @property
    def mean_step(self) -> float:
        return float((self.times[-1] - self.times[0]) / max(self.n_steps, 1))


def orbital_elements(
    position: ArrayLike, velocity: ArrayLike, model: GravityModel = EARTH
) -> dict[str, float]:
    """Classical osculating elements from an ECI state.

    Returns semi-major axis (m), eccentricity, inclination, RAAN and
    argument of perigee (rad). Used to expose the secular :math:`J_2`
    drifts the propagation is meant to reproduce.
    """
    r = np.asarray(position, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    if r.shape != (3,) or v.shape != (3,):
        raise ValueError("position and velocity must both be 3-vectors")
    radius = float(np.linalg.norm(r))
    speed = float(np.linalg.norm(v))
    h_vec = np.cross(r, v)
    h_mag = float(np.linalg.norm(h_vec))
    if h_mag == 0.0:
        raise ValueError("degenerate (radial) trajectory has no orbital plane")
    energy = 0.5 * speed**2 - model.mu / radius
    if energy >= 0.0:
        raise ValueError("state is not bound; osculating elements are undefined")
    sma = -model.mu / (2.0 * energy)
    e_vec = np.cross(v, h_vec) / model.mu - r / radius
    ecc = float(np.linalg.norm(e_vec))
    inc = float(np.arccos(np.clip(h_vec[2] / h_mag, -1.0, 1.0)))
    node_vec = np.array([-h_vec[1], h_vec[0], 0.0])
    node_mag = float(np.linalg.norm(node_vec))
    if node_mag < 1e-12 * h_mag:  # equatorial: RAAN and argp are degenerate
        raan = 0.0
        argp = float(np.arctan2(e_vec[1], e_vec[0])) if ecc > 0 else 0.0
    else:
        raan = float(np.arctan2(node_vec[1], node_vec[0]))
        if ecc > 1e-12:
            argp = float(
                np.arccos(np.clip(node_vec @ e_vec / (node_mag * ecc), -1.0, 1.0))
            )
            if e_vec[2] < 0.0:
                argp = 2.0 * np.pi - argp
        else:
            argp = 0.0
    return {
        "semi_major_axis": float(sma),
        "eccentricity": ecc,
        "inclination": inc,
        "raan": raan % (2.0 * np.pi),
        "argument_of_perigee": argp % (2.0 * np.pi),
    }


def secular_rates(
    semi_major_axis: float,
    eccentricity: float,
    inclination: float,
    model: GravityModel = EARTH,
) -> dict[str, float]:
    """First-order secular :math:`J_2` rates (rad/s).

    Nodal regression and apsidal precession — the two effects Paper II
    §7.1 names as the reason a spherical model is inadequate over a
    fractional orbit:

    .. math::

        \\dot\\Omega = -\\tfrac{3}{2}\\,n J_2 (R_\\oplus/p)^2\\cos i,
        \\qquad
        \\dot\\omega = \\tfrac{3}{4}\\,n J_2 (R_\\oplus/p)^2 (5\\cos^2 i - 1).

    These are analytic first-order averages, independent of the
    propagator, so they are a genuine cross-check of it.
    """
    a = float(semi_major_axis)
    e = float(eccentricity)
    if not (np.isfinite(a) and a > 0.0):
        raise ValueError(f"semi_major_axis must be finite and > 0, got {a}")
    if not 0.0 <= e < 1.0:
        raise ValueError(f"eccentricity must be in [0, 1), got {e}")
    p = a * (1.0 - e * e)
    n = np.sqrt(model.mu / a**3)
    common = n * model.j2 * (model.radius / p) ** 2
    cos_i = np.cos(inclination)
    return {
        "mean_motion": float(n),
        "raan_rate": float(-1.5 * common * cos_i),
        "argp_rate": float(0.75 * common * (5.0 * cos_i**2 - 1.0)),
    }


def _atmospheric_density(altitude: _FloatArray | float) -> _FloatArray:
    """Exponential density, evaluated without any branch on altitude.

    The exponential simply underflows to zero far above the sensible
    atmosphere, which is the mechanism Paper II §7.2 relies on: no
    Kármán-line switch is taken, the term just stops mattering.
    """
    alt = np.asarray(altitude, dtype=np.float64)
    return np.asarray(_RHO0 * np.exp(-np.maximum(alt, 0.0) / _H_SCALE))


def propagate_coast(
    position: ArrayLike,
    velocity: ArrayLike,
    duration: float,
    model: GravityModel = EARTH,
    include_j2: bool = True,
    rtol: float = 1e-12,
    atol: float = 1e-9,
    n_output: int = 201,
    ballistic_coefficient: float | None = None,
) -> CoastResult:
    """Propagate a coast arc under :math:`J_2` gravity.

    Parameters
    ----------
    position, velocity:
        Initial ECI state (m, m/s).
    duration:
        Arc length (s).
    include_j2:
        ``False`` propagates spherical gravity, for the secular
        comparison of Paper II §7.1.
    ballistic_coefficient:
        If given, retains atmospheric drag with this :math:`m/(C_D A)`
        (kg/m²) — the paper keeps drag through coast because perigee on
        a fractional orbital profile may be low enough for it to matter.
        The term is *always evaluated*, never switched.
    """
    r0 = np.asarray(position, dtype=np.float64)
    v0 = np.asarray(velocity, dtype=np.float64)
    if r0.shape != (3,) or v0.shape != (3,):
        raise ValueError("position and velocity must both be 3-vectors")
    if not (np.isfinite(duration) and duration > 0.0):
        raise ValueError(f"duration must be finite and > 0, got {duration}")

    def rhs(_t: float, y: _FloatArray) -> _FloatArray:
        r = y[:3]
        v = y[3:]
        accel = gravitational_acceleration(r, model, include_j2=include_j2)
        if ballistic_coefficient is not None:
            altitude = float(np.linalg.norm(r)) - model.radius
            rho = float(_atmospheric_density(altitude))
            speed = float(np.linalg.norm(v))
            accel = accel - 0.5 * rho * speed * v / ballistic_coefficient
        return np.concatenate([v, accel])

    t_eval = np.linspace(0.0, float(duration), int(n_output))
    start = time.perf_counter()
    sol = scipy.integrate.solve_ivp(
        rhs,
        (0.0, float(duration)),
        np.concatenate([r0, v0]),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        t_eval=t_eval,
    )
    wall = time.perf_counter() - start
    if not sol.success:
        raise RuntimeError(f"coast propagation failed: {sol.message}")

    states = np.asarray(sol.y)
    energy = specific_energy(states[:3].T, states[3:].T, model)
    h_z = specific_angular_momentum_z(states[:3].T, states[3:].T)
    return CoastResult(
        times=np.asarray(sol.t),
        states=states,
        energy=np.asarray(energy),
        angular_momentum_z=np.asarray(h_z),
        n_steps=int(sol.nfev),
        wall_time=wall,
    )


@dataclass(frozen=True)
class RegimeTransition:
    """Altitude profile of the aerodynamic term relative to gravity.

    Demonstrates the §7.2 mechanism quantitatively: the ratio of
    aerodynamic to gravitational acceleration falls smoothly through
    many orders of magnitude with no discontinuity, so a single
    integration crosses the Kármán line without a branch.
    """

    altitudes: _FloatArray
    acceleration_ratio: _FloatArray

    def negligible_altitude(self, threshold: float = 1e-10) -> float:
        """Lowest sampled altitude at which the ratio is below ``threshold``."""
        below = np.nonzero(self.acceleration_ratio < threshold)[0]
        if below.size == 0:
            return float("inf")
        return float(self.altitudes[below[0]])


def regime_transition_profile(
    speed: float,
    ballistic_coefficient: float,
    altitudes: ArrayLike | None = None,
    model: GravityModel = EARTH,
) -> RegimeTransition:
    """Aerodynamic-to-gravitational acceleration ratio versus altitude.

    .. note::

        The single-scale-height exponential used here over-predicts
        density above roughly 86 km, where the real atmosphere's scale
        height grows substantially. That makes the crossing altitudes
        reported below *conservative* — a more faithful model reaches
        negligibility lower, not higher. The claim being verified is the
        smoothness of the decay, which holds for any exponential.
    """
    if not (np.isfinite(speed) and speed > 0.0):
        raise ValueError(f"speed must be finite and > 0, got {speed}")
    if not (np.isfinite(ballistic_coefficient) and ballistic_coefficient > 0.0):
        raise ValueError("ballistic_coefficient must be finite and > 0")
    alt = (
        np.linspace(0.0, 500_000.0, 501)
        if altitudes is None
        else np.asarray(altitudes, dtype=np.float64)
    )
    rho = _atmospheric_density(alt)
    drag = 0.5 * rho * speed**2 / ballistic_coefficient
    gravity = model.mu / (model.radius + alt) ** 2
    return RegimeTransition(altitudes=alt, acceleration_ratio=np.asarray(drag / gravity))


@dataclass(frozen=True)
class StrategyComparison:
    """II-V5: single-integration versus frozen-structure over a coast."""

    label: str
    n_rhs_evaluations: int
    mean_step: float
    wall_time: float
    energy_drift: float
    structural_energy_ratio: float
    """Final structural modal energy relative to its initial value."""


def compare_coast_strategies(
    position: ArrayLike,
    velocity: ArrayLike,
    duration: float,
    structural_frequency: float,
    structural_damping: float = 0.02,
    freeze_threshold: float = 1e-6,
    model: GravityModel = EARTH,
    rtol: float = 1e-10,
    atol: float = 1e-9,
) -> tuple[StrategyComparison, StrategyComparison]:
    """Propagate a coast with the structural block live, then frozen.

    The structural block is a damped modal oscillator at
    ``structural_frequency`` — the highest retained mode, which is what
    sets the step-size constraint. In the *single-integration* strategy
    it is carried through the whole coast; in the *frozen-structure*
    strategy it is dropped from the state once its modal energy falls
    below ``freeze_threshold`` of its initial value, which is the
    "provably quiescent" condition Paper II's Remark 5 requires for the
    switch to be defensible.

    Returns
    -------
    tuple
        ``(single_integration, frozen_structure)``.
    """
    r0 = np.asarray(position, dtype=np.float64)
    v0 = np.asarray(velocity, dtype=np.float64)
    omega = float(structural_frequency)
    if not (np.isfinite(omega) and omega > 0.0):
        raise ValueError(f"structural_frequency must be finite and > 0, got {omega}")
    zeta = float(structural_damping)
    if not 0.0 <= zeta < 1.0:
        raise ValueError(f"structural_damping must be in [0, 1), got {zeta}")
    if not 0.0 < freeze_threshold < 1.0:
        raise ValueError(f"freeze_threshold must be in (0, 1), got {freeze_threshold}")

    q0, qdot0 = 1.0e-3, 0.0
    initial_modal_energy = 0.5 * (qdot0**2 + omega**2 * q0**2)

    def orbital_rhs(y: _FloatArray) -> _FloatArray:
        return np.concatenate([y[3:6], gravitational_acceleration(y[:3], model)])

    def coupled_rhs(_t: float, y: _FloatArray) -> _FloatArray:
        out = np.empty(8)
        out[:6] = orbital_rhs(y)
        out[6] = y[7]
        out[7] = -2.0 * zeta * omega * y[7] - omega**2 * y[6]
        return out

    def orbital_only_rhs(_t: float, y: _FloatArray) -> _FloatArray:
        return orbital_rhs(y)

    # --- strategy 1: one integration carrying the structural block
    start = time.perf_counter()
    full = scipy.integrate.solve_ivp(
        coupled_rhs,
        (0.0, float(duration)),
        np.concatenate([r0, v0, [q0, qdot0]]),
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    wall_full = time.perf_counter() - start
    if not full.success:
        raise RuntimeError(f"single-integration coast failed: {full.message}")
    q_end, qd_end = full.y[6, -1], full.y[7, -1]
    modal_ratio_full = (
        0.5 * (qd_end**2 + omega**2 * q_end**2) / initial_modal_energy
        if initial_modal_energy > 0
        else 0.0
    )
    e_full = specific_energy(full.y[:3].T, full.y[3:6].T, model)

    # --- strategy 2: freeze once the modal energy is provably quiescent.
    # The ring-down of a damped oscillator is analytic, so the freeze time
    # is known in closed form rather than found by polling.
    freeze_time = min(
        float(duration), float(-np.log(freeze_threshold) / (2.0 * zeta * omega))
    ) if zeta > 0.0 else float(duration)
    start = time.perf_counter()
    phase_a = scipy.integrate.solve_ivp(
        coupled_rhs,
        (0.0, freeze_time),
        np.concatenate([r0, v0, [q0, qdot0]]),
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not phase_a.success:
        raise RuntimeError(f"frozen-structure phase A failed: {phase_a.message}")
    n_eval = int(phase_a.nfev)
    y_mid = phase_a.y[:6, -1]
    energies = [specific_energy(phase_a.y[:3].T, phase_a.y[3:6].T, model)]
    if freeze_time < float(duration):
        phase_b = scipy.integrate.solve_ivp(
            orbital_only_rhs,
            (freeze_time, float(duration)),
            y_mid,
            method="DOP853",
            rtol=rtol,
            atol=atol,
        )
        if not phase_b.success:
            raise RuntimeError(f"frozen-structure phase B failed: {phase_b.message}")
        n_eval += int(phase_b.nfev)
        energies.append(specific_energy(phase_b.y[:3].T, phase_b.y[3:6].T, model))
    wall_frozen = time.perf_counter() - start
    e_frozen = np.concatenate(energies)
    q_f, qd_f = phase_a.y[6, -1], phase_a.y[7, -1]
    modal_ratio_frozen = (
        0.5 * (qd_f**2 + omega**2 * q_f**2) / initial_modal_energy
        if initial_modal_energy > 0
        else 0.0
    )

    single = StrategyComparison(
        label="single-integration",
        n_rhs_evaluations=int(full.nfev),
        mean_step=float(duration) / max(full.t.size - 1, 1),
        wall_time=wall_full,
        energy_drift=float(abs(e_full[-1] - e_full[0]) / abs(e_full[0])),
        structural_energy_ratio=float(modal_ratio_full),
    )
    frozen = StrategyComparison(
        label="frozen-structure",
        n_rhs_evaluations=n_eval,
        mean_step=float(duration) / max(phase_a.t.size - 1, 1),
        wall_time=wall_frozen,
        energy_drift=float(abs(e_frozen[-1] - e_frozen[0]) / abs(e_frozen[0])),
        structural_energy_ratio=float(modal_ratio_frozen),
    )
    return single, frozen

