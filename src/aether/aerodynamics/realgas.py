"""Equilibrium air: what the gas actually does above about Mach 8.

A perfect-gas stagnation temperature at Mach 20 is 20,000 K. Air does not
reach 20,000 K behind a Mach 20 shock — it reaches about 5,900 K, because
long before then the oxygen has dissociated, then the nitrogen, and the
energy that would have gone into translation went into breaking bonds
instead. Every consequence follows from that one fact:

* **The density ratio across the shock is not 6.** A calorically perfect gas
  is stuck at :math:`(\\gamma+1)/(\\gamma-1) = 6`. Equilibrium air reaches
  15 at Mach 20 and 17 at Mach 25, because the same pressure rise is
  achieved at a much lower temperature and therefore a much higher density.
* **The shock lies closer to the body**, since standoff distance scales with
  the density ratio, and that changes the whole inviscid flowfield.
* **The Newtonian** :math:`C_{p,\\max}` **is not 1.839.** It is 1.93 at Mach
  20 and 1.94 at Mach 25 — a five per cent increase in every windward
  pressure, and therefore in axial force, which the perfect-gas Rayleigh–Pitot
  relation in :mod:`aether.aerodynamics.closure` cannot produce at any Mach
  number because its asymptote is 1.839 by construction.

Five per cent on axial force is not a rounding error on a vehicle whose
range is being predicted, and it goes the way that makes a perfect-gas
estimate optimistic.

How it is computed
------------------

Not from a curve fit. The Rankine–Hugoniot jump conditions are solved
simultaneously with **chemical equilibrium** over the eleven-species
ionising air mixture of Cantera's ``airNASA9`` mechanism — N₂, O₂, NO, N, O,
their singly-charged ions and free electrons, on NASA 9-coefficient
polynomials valid to 20,000 K. The unknown is the density ratio
:math:`\\varepsilon = \\rho_\\infty/\\rho_2`; given it, the jump conditions
close in one step,

.. math::

    u_2 = \\varepsilon V_\\infty, \\qquad
    p_2 = p_\\infty + \\rho_\\infty V_\\infty^2 (1-\\varepsilon), \\qquad
    h_2 = h_\\infty + \\tfrac12 V_\\infty^2 (1-\\varepsilon^2)

and an equilibrium solve at constant :math:`(h, p)` returns a new density
and hence a new :math:`\\varepsilon`. The iteration is damped at one half,
which is not tuning: undamped it oscillates, because raising the density
ratio raises the enthalpy jump which lowers the density ratio.

Argon is absent from the mechanism. It is 0.93 % of air by mole and
chemically inert up to the ionisation limit, so leaving it out moves the
molar mass by 0.3 % and the density ratio by less; it is noted because the
mixture used should be visible, not because it matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.optimize

from aether.aerodynamics.closure import rayleigh_pitot_cp_max

__all__ = [
    "AIR_MOLE_FRACTIONS",
    "EquilibriumAir",
    "NormalShock",
    "perfect_gas_normal_shock",
]


@dataclass(frozen=True)
class NormalShock:
    """State behind a normal shock, and the stagnation state behind it."""

    freestream_temperature: float
    freestream_pressure: float
    freestream_density: float
    speed: float
    """Freestream speed (m/s)."""
    mach: float

    temperature: float
    """:math:`T_2` (K)."""
    pressure: float
    """:math:`p_2` (Pa)."""
    density: float
    """:math:`\\rho_2` (kg/m³)."""
    velocity: float
    """:math:`u_2` (m/s), in the shock-fixed frame."""
    stagnation_pressure: float
    """:math:`p_{02}` (Pa) — isentropic from state 2 to rest, in equilibrium."""
    converged: bool

    @property
    def density_ratio(self) -> float:
        """:math:`\\rho_2/\\rho_\\infty`. Six is the perfect-gas ceiling."""
        return float(self.density / self.freestream_density)

    @property
    def cp_max(self) -> float:
        """:math:`C_{p,\\max} = (p_{02}-p_\\infty)/q_\\infty`.

        The number the modified-Newtonian closure multiplies
        :math:`\\sin^2\\delta` by, and the single place a real-gas correction
        enters an impact-theory pressure distribution.
        """
        dynamic = 0.5 * self.freestream_density * self.speed**2
        return float((self.stagnation_pressure - self.freestream_pressure) / dynamic)

    @property
    def effective_gamma(self) -> float:
        """The :math:`\\gamma` that would give this density ratio at this Mach.

        Inverted from the perfect-gas jump
        :math:`\\rho_2/\\rho_1 = (\\gamma+1)M^2/((\\gamma-1)M^2+2)`. A
        *diagnostic*, not a parameter: no single :math:`\\gamma` reproduces
        equilibrium air across the jump, and quoting one is a way of saying
        how far from perfect the gas has become, not a way of modelling it.
        """
        ratio, m2 = self.density_ratio, self.mach**2
        if ratio <= 1.0 or m2 <= 1.0:
            return float("nan")
        return float((m2 * (1.0 + ratio) - 2.0 * ratio) / (m2 * (ratio - 1.0)))

    @property
    def total_enthalpy(self) -> float:
        """:math:`h_0 = h_\\infty + V_\\infty^2/2` (J/kg) — conserved across the shock."""
        return float(0.5 * self.speed**2)


#: Dry air without argon, by mole. See the module note.
AIR_MOLE_FRACTIONS = "N2:0.79, O2:0.21"


@dataclass
class EquilibriumAir:
    """Chemical-equilibrium air, and the normal shock it produces.

    Attributes
    ----------
    mechanism:
        Cantera mechanism file. The default is the ionising eleven-species
        air set; ``air.yaml`` is a lower-temperature set that tops out at
        3500 K and is not usable above about Mach 8.
    composition:
        Freestream mole fractions.
    damping:
        Relaxation on the density-ratio iteration. One half converges from
        Mach 3 to Mach 30 in about forty iterations; unity oscillates.
    """

    mechanism: str = "airNASA9.yaml"
    composition: str = AIR_MOLE_FRACTIONS
    damping: float = 0.5
    tolerance: float = 1.0e-12
    max_iterations: int = 200
    name: str = "equilibrium air (Cantera airNASA9)"
    _gas: Any = field(default=None, init=False, repr=False, compare=False)
    _cache: dict[tuple[float, float, float], NormalShock] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    @property
    def gas(self) -> Any:
        """The Cantera solution object, created on first use.

        Deferred so that importing this module does not cost a mechanism
        parse, and held per instance because a Cantera ``Solution`` is a
        mutable state that two concurrent solves would corrupt.
        """
        if self._gas is None:
            try:
                import cantera
            except ImportError as error:  # pragma: no cover - dependency declared
                msg = (
                    "equilibrium air needs Cantera (pip install cantera). The "
                    "alternative is a curve fit to somebody's equilibrium "
                    "tables, which cannot be checked; a thermochemistry "
                    "library with published polynomial data can be."
                )
                raise ImportError(msg) from error
            self._gas = cantera.Solution(self.mechanism)
        return self._gas

    def freestream(self, temperature: float, pressure: float) -> tuple[float, float, float]:
        """Density, enthalpy and sound speed of undisturbed air (SI).

        The freestream is itself equilibrated. Below about 2000 K that is a
        no-op — air at 250 K is 79 % N₂ and 21 % O₂ and stays that way — but
        doing it unconditionally means the enthalpy datum on both sides of
        the shock is the same one, which is what the energy jump requires.
        """
        gas = self.gas
        gas.TPX = float(temperature), float(pressure), self.composition
        gas.equilibrate("TP")
        return float(gas.density), float(gas.enthalpy_mass), float(gas.sound_speed)

    def normal_shock(
        self, temperature: float, pressure: float, speed: float
    ) -> NormalShock:
        """Solve the equilibrium normal shock at a freestream condition.

        Parameters
        ----------
        temperature, pressure:
            Freestream static conditions (K, Pa).
        speed:
            Freestream speed (m/s).
        """
        t1, p1, v1 = float(temperature), float(pressure), float(speed)
        for label, value in (("temperature", t1), ("pressure", p1), ("speed", v1)):
            if not (np.isfinite(value) and value > 0.0):
                msg = f"freestream {label} must be finite and > 0, got {value}"
                raise ValueError(msg)

        key = (round(t1, 6), round(p1, 9), round(v1, 6))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        gas = self.gas
        rho1, h1, sound = self.freestream(t1, p1)
        mach = v1 / sound
        if mach <= 1.0:
            msg = (
                f"a normal shock needs supersonic freestream; Mach {mach:.4f} at "
                f"{t1:g} K gives a sound speed of {sound:.1f} m/s"
            )
            raise ValueError(msg)

        epsilon = 1.0 / 6.0
        converged = False
        pressure_2 = p1
        enthalpy_2 = h1
        for _ in range(int(self.max_iterations)):
            pressure_2 = p1 + rho1 * v1**2 * (1.0 - epsilon)
            enthalpy_2 = h1 + 0.5 * v1**2 * (1.0 - epsilon**2)
            gas.HP = enthalpy_2, pressure_2
            gas.equilibrate("HP")
            update = rho1 / float(gas.density)
            if abs(update - epsilon) < self.tolerance:
                epsilon = update
                converged = True
                break
            epsilon = (1.0 - self.damping) * epsilon + self.damping * update

        velocity_2 = epsilon * v1
        temperature_2 = float(gas.T)
        density_2 = float(gas.density)
        entropy_2 = float(gas.entropy_mass)
        stagnation_enthalpy = enthalpy_2 + 0.5 * velocity_2**2

        def residual(log_pressure: float) -> float:
            gas.SP = entropy_2, float(np.exp(log_pressure))
            gas.equilibrate("SP")
            return float(gas.enthalpy_mass) - stagnation_enthalpy

        # Brent on log pressure: the stagnation pressure is between p2 and a
        # bounded multiple of it (the subsonic compression from u2 to rest is
        # weak), and searching in the log keeps the bracket well conditioned
        # across five decades of freestream pressure.
        log_p = scipy.optimize.brentq(
            residual, float(np.log(pressure_2)), float(np.log(pressure_2 * 100.0)),
            xtol=1.0e-10,
        )

        solution = NormalShock(
            freestream_temperature=t1,
            freestream_pressure=p1,
            freestream_density=rho1,
            speed=v1,
            mach=mach,
            temperature=temperature_2,
            pressure=pressure_2,
            density=density_2,
            velocity=velocity_2,
            stagnation_pressure=float(np.exp(log_p)),
            converged=converged,
        )
        self._cache[key] = solution
        return solution

    def cp_max(self, temperature: float, pressure: float, speed: float) -> float:
        """:math:`C_{p,\\max}` for the modified-Newtonian closure."""
        return self.normal_shock(temperature, pressure, speed).cp_max


def perfect_gas_normal_shock(mach: float, gamma: float = 1.4) -> dict[str, float]:
    """Closed-form perfect-gas jump, for comparison against the equilibrium one.

    Returns the density ratio, the temperature ratio and
    :math:`C_{p,\\max}`. This is the thing the equilibrium solution is meant
    to replace, and having it here makes the size of the replacement visible
    at any Mach number rather than asserted at one.
    """
    m = float(mach)
    g = float(gamma)
    if not (np.isfinite(m) and m > 1.0):
        msg = f"normal shock requires supersonic Mach, got {mach}"
        raise ValueError(msg)
    m2 = m * m
    density_ratio = (g + 1.0) * m2 / ((g - 1.0) * m2 + 2.0)
    pressure_ratio = (2.0 * g * m2 - (g - 1.0)) / (g + 1.0)
    return {
        "density_ratio": float(density_ratio),
        "pressure_ratio": float(pressure_ratio),
        "temperature_ratio": float(pressure_ratio / density_ratio),
        "cp_max": float(rayleigh_pitot_cp_max(m, g)),
    }
