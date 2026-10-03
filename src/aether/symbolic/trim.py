r"""The entry field at trim, with its closures bound: what a simulator compiles.

The dynamics appendix carries a quaternion, and
:mod:`aether.symbolic.attitude` shows that an attitude held at incidence and
bank returns the lift-and-drag point mass (the derivation
``trim-reduction``). A capsule flies at trim and steers by bank alone, so for
it the reduced field is the model, and this module states it as a
:class:`~aether.symbolic.system.SymbolicSystem`: the translational field of
record, :func:`aether.certification.rotating_field.point_mass_field`,
evaluated on symbols over a rotating, oblate planet whose atmosphere lies on
the reference ellipsoid, with every closure an expression.

Nothing about the motion is restated here. The six rates are the field's.
What is added is what a flight needs beside them:

* the arc of ground track flown, :math:`\dot s = V\cos\gamma/r`, which is
  how far the body has gone over the planet and not through inertial space;
* the stagnation-point heat load, the integral of the convective flux
  :math:`k_Q\sqrt{\rho/R_n}\,V^3`, a screening correlation whose envelope
  the appendix states;
* and the quantities read along a trajectory: altitude, density, the drag and
  lift accelerations an accelerometer senses, the load factor, the dynamic
  pressure, the heat flux, and the energy that drag dissipates.

Two systems are built, and they are the same field. :func:`entry_system` is
the plant: the bank angle is a parameter, held over a guidance cycle.
:func:`prediction_system` is the same field as a guidance law's predictor
flies it: no lateral force, and a vertical lift fraction that slews toward
its commanded value at the body's roll rate, so that the prediction starts
from the bank the body is holding and not from the one it was asked for.

The parameters are named, and a simulator binds them by name:

==========  ==================================================================
``sigma``   bank angle (rad)
``beta``    ballistic coefficient :math:`m/(C_D S)` (kg/m²)
``E``       lift-to-drag ratio
``k_rho``   density of the air flown through, over the model atmosphere's
``mu``      gravitational parameter (m³/s²)
``R_e``     equatorial radius (m)
``J_2``     second zonal harmonic
``omega_E`` rotation rate (rad/s)
``f_E``     flattening of the reference ellipsoid
``R_n``     nose radius of the heating correlation (m)
``k_Q``     constant of the heating correlation (SI)
==========  ==================================================================
"""

from __future__ import annotations

from typing import Any

import sympy as sp

from aether.certification.rotating_field import jacobi_energy, rotating_entry_field
from aether.symbolic.atmosphere import Atmosphere, standard_atmosphere
from aether.symbolic.entry import EntrySymbols
from aether.symbolic.ops import SYMPY_OPS
from aether.symbolic.system import SymbolicSystem

__all__ = ["STANDARD_GRAVITY", "entry_system", "prediction_system"]

#: Standard gravity (m/s²): the unit a load factor is quoted in. A definition,
#: not the local gravity.
STANDARD_GRAVITY = sp.Rational(980665, 100000)

_BANK = sp.Symbol("sigma", real=True)
_BALLISTIC = sp.Symbol("beta", positive=True)
_LIFT_TO_DRAG = sp.Symbol("E", positive=True)
_DENSITY_SCALE = sp.Symbol("k_rho", positive=True)
_NOSE_RADIUS = sp.Symbol("R_n", positive=True)
_HEATING_CONSTANT = sp.Symbol("k_Q", positive=True)
_RANGE = sp.Symbol("s", real=True)
_HEAT_LOAD = sp.Symbol("Q_tot", nonnegative=True)
_CLOCK = sp.Symbol("tau", nonnegative=True)
_BANK_NOW = sp.Symbol("sigma_0", nonnegative=True)
_BANK_GOAL = sp.Symbol("sigma_c", nonnegative=True)
_ROLL_RATE = sp.Symbol("sigmadot_max", positive=True)


def _planet_parameters(symbols: EntrySymbols) -> tuple[Any, ...]:
    planet = symbols.planet
    return (planet.mu, planet.radius, planet.j2, planet.rotation_rate, planet.flattening)


def _aerodynamics(symbols: EntrySymbols, atmosphere: Atmosphere) -> dict[str, Any]:
    """Density and the two aerodynamic accelerations, from the bound closures."""
    density = _DENSITY_SCALE * atmosphere.density(symbols.altitude)
    drag = density * symbols.speed**2 / (2 * _BALLISTIC)
    return {"density": density, "drag": drag, "lift": _LIFT_TO_DRAG * drag}


def _outputs(symbols: EntrySymbols, aero: dict[str, Any]) -> dict[str, Any]:
    drag, lift, density = aero["drag"], aero["lift"], aero["density"]
    return {
        "altitude": symbols.altitude,
        "altitude_rate": symbols.speed * sp.sin(symbols.gamma),
        "density": density,
        "drag": drag,
        "lift": lift,
        "load_factor": sp.sqrt(drag**2 + lift**2) / STANDARD_GRAVITY,
        "dynamic_pressure": density * symbols.speed**2 / 2,
        "heat_flux": _HEATING_CONSTANT * sp.sqrt(density / _NOSE_RADIUS) * symbols.speed**3,
        "drag_power": drag * symbols.speed,
        "energy": jacobi_energy(symbols.state, SYMPY_OPS, planet=symbols.planet),
    }


def entry_system(
    atmosphere: Atmosphere | None = None, *, name: str = "entry_at_trim"
) -> SymbolicSystem:
    r"""The plant: :math:`(r, \lambda, \phi, V, \gamma, \psi, s, Q_{\mathrm{tot}})` at a held bank.

    The first six rates are :func:`~aether.certification.rotating_field.rotating_entry_field`
    under the force :math:`(-D,\ L\cos\sigma,\ L\sin\sigma)/m` with
    :math:`D/m = \rho V^2/(2\beta)` and :math:`L = E\,D`; the last two are
    the ground-track arc and the heat load.
    """
    symbols = EntrySymbols()
    atmosphere = standard_atmosphere() if atmosphere is None else atmosphere
    aero = _aerodynamics(symbols, atmosphere)
    force = (-aero["drag"], aero["lift"] * sp.cos(_BANK), aero["lift"] * sp.sin(_BANK))
    motion = rotating_entry_field(symbols.state, force, SYMPY_OPS, planet=symbols.planet)
    outputs = _outputs(symbols, aero)
    return SymbolicSystem(
        name=name,
        state=(*symbols.state, _RANGE, _HEAT_LOAD),
        parameters=(
            _BANK, _BALLISTIC, _LIFT_TO_DRAG, _DENSITY_SCALE,
            *_planet_parameters(symbols), _NOSE_RADIUS, _HEATING_CONSTANT,
        ),
        rates=(
            *motion,
            symbols.speed * sp.cos(symbols.gamma) / symbols.radius,
            outputs["heat_flux"],
        ),
        outputs=outputs,
    )


def prediction_system(
    atmosphere: Atmosphere | None = None, *, name: str = "entry_prediction"
) -> SymbolicSystem:
    r"""The same field as a predictor flies it: planar lift, slewed at the roll rate.

    State :math:`(r, \lambda, \phi, V, \gamma, \psi, s, \tau)`, with
    :math:`\tau` the time into the prediction. The bank magnitude starts at
    :math:`\sigma_0`, the one the body holds, and moves toward the commanded
    :math:`\sigma_c` no faster than :math:`\dot\sigma_{\max}`:

    .. math::

        \lvert\sigma\rvert(\tau) = \sigma_0
            + \max\bigl(-\dot\sigma_{\max}\tau,\
              \min(\dot\sigma_{\max}\tau,\ \sigma_c - \sigma_0)\bigr),

    and the force is :math:`(-D,\ L\cos\lvert\sigma\rvert,\ 0)/m`: all of the
    lift that is not vertical is left out, which is what makes the prediction
    a one-parameter family in the vertical lift fraction.
    """
    symbols = EntrySymbols()
    atmosphere = standard_atmosphere() if atmosphere is None else atmosphere
    aero = _aerodynamics(symbols, atmosphere)
    travel = sp.Max(-_ROLL_RATE * _CLOCK, sp.Min(_ROLL_RATE * _CLOCK, _BANK_GOAL - _BANK_NOW))
    vertical = sp.cos(_BANK_NOW + travel)
    force = (-aero["drag"], aero["lift"] * vertical, sp.S.Zero)
    motion = rotating_entry_field(symbols.state, force, SYMPY_OPS, planet=symbols.planet)
    outputs = _outputs(symbols, aero)
    outputs = {key: outputs[key] for key in ("altitude", "drag", "lift", "load_factor")}
    return SymbolicSystem(
        name=name,
        state=(*symbols.state, _RANGE, _CLOCK),
        parameters=(
            _BANK_NOW, _BANK_GOAL, _ROLL_RATE, _BALLISTIC, _LIFT_TO_DRAG, _DENSITY_SCALE,
            *_planet_parameters(symbols),
        ),
        rates=(
            *motion,
            symbols.speed * sp.cos(symbols.gamma) / symbols.radius,
            sp.S.One,
        ),
        outputs=outputs,
    )
