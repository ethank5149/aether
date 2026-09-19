r"""Plasma sheath and RF blackout, from a nonequilibrium shock layer.

A re-entry body at Mach 20 is wrapped in air hot enough to ionise. The free
electrons in that sheath refract and absorb radio waves, and above a critical
density they reflect them outright — the vehicle stops being able to talk, and
stops being visible to anything that works by radio. When it starts and stops
is a signature, and it is set by the shock-layer state the NEMO solver already
computes.

Two ways in, and the difference matters
---------------------------------------

:func:`electron_density_from_solution` reads :math:`n_e` from a NEMO solution
that carried electrons as a species. That is the honest number: it comes from
the same finite-rate chemistry that set the temperature.

:func:`saha_electron_density` estimates it from temperature and pressure alone
by equilibrium ionisation. Use it for an envelope, a sanity check, or a case
solved without electrons — never in preference to a solution that has them.
Behind a strong shock the flow is *not* in equilibrium: ionisation lags the
temperature rise by a relaxation length that at 60 km is a good fraction of the
nose radius, so Saha overpredicts near the shock and underpredicts in the
expansion.

The drafted version of this had the two-constant bug that makes Saha look
plausible and be wrong: :math:`k_B` in eV/K and :math:`h` in eV·s substituted
into a thermal-de-Broglie expression that also carried a conversion to joules,
so the exponent was in eV while the prefactor was in a mixture. It is written
here in SI throughout, with the ionisation energy converted once, on entry.

Attenuation
-----------

The Appleton–Hartree result for a collisional, unmagnetised plasma. With
plasma frequency :math:`\omega_p^2 = n_e e^2/\varepsilon_0 m_e` and collision
frequency :math:`\nu`, the wave number acquires an imaginary part and the
one-way power loss over a path :math:`s` is

.. math::

    \alpha = \frac{\omega_p^2 \nu}
                  {c\,(\omega^2 + \nu^2)}\;\text{Np/m}, \qquad
    L_{\mathrm{dB}} = 8.686 \int \alpha\,\mathrm{d}s .

Two regimes fall out of that expression and both matter:

* :math:`\omega < \omega_p` — the wave is **evanescent**. This is blackout
  proper: not attenuation but reflection, and no link margin reaches through
  it. :func:`critical_density` is where that starts.
* :math:`\omega \gg \omega_p` — attenuation falls as :math:`\omega^{-2}`,
  which is the whole argument for putting a telemetry link at Ka band rather
  than S band.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "BlackoutWindow",
    "attenuation_db",
    "collision_frequency",
    "critical_density",
    "electron_density_from_solution",
    "plasma_frequency",
    "saha_electron_density",
]

_FloatArray = NDArray[np.float64]

_ELEMENTARY_CHARGE = 1.602176634e-19       # C
_ELECTRON_MASS = 9.1093837015e-31          # kg
_VACUUM_PERMITTIVITY = 8.8541878128e-12    # F/m
_BOLTZMANN = 1.380649e-23                  # J/K
_PLANCK = 6.62607015e-34                   # J·s
_LIGHT = 2.99792458e8                      # m/s

#: First ionisation energies (eV). Nitric oxide first: NO ionises at 9.26 eV,
#: far below N or O, so it is the dominant electron source through the
#: 2000–6000 K band a re-entry shock layer actually spends most of its time in.
IONISATION_EV = {"NO": 9.264, "O": 13.618, "N": 14.534, "O2": 12.070, "N2": 15.581}


def saha_electron_density(
    temperature: _FloatArray | float,
    pressure: _FloatArray | float,
    species: str = "NO",
    mole_fraction: float = 0.01,
    degeneracy_ratio: float = 0.5,
) -> _FloatArray:
    r"""Equilibrium electron density (m⁻³) from the Saha equation.

    For a singly ionised trace species in a neutral bath, with :math:`n_e =
    n_i` and the neutral density fixed by the total pressure,

    .. math::

        \frac{n_e n_i}{n_n} = \frac{2g_i}{g_n}
            \left(\frac{2\pi m_e k_B T}{h^2}\right)^{3/2} e^{-\chi/k_B T}

    so :math:`n_e = \sqrt{K n_n}` when :math:`n_e \ll n_n`, which holds
    wherever the ionisation fraction is below a percent or so — true for
    everything below about 8000 K.

    ``species`` picks the ionisation potential. The default is **NO**, not N or
    O: nitric oxide forms readily behind the shock and ionises four to five
    electron-volts lower than either atom, so it dominates the electron budget
    across most of the trajectory. Choosing N, as the drafted version did,
    understates :math:`n_e` by orders of magnitude at 4000 K.

    ``mole_fraction`` is the ionising species' share of the mixture, and it is
    not a refinement. :math:`n_e` scales as :math:`\sqrt{n_n}`, so taking the
    *total* density as the NO density — which is the natural way to write this
    and is wrong — overpredicts the electron density by a factor of ten at the
    1 % that equilibrium air actually produces. The default is that 1 %; pass
    the value from a NEMO solution's composition when there is one, or better,
    use :func:`electron_density_from_solution` and skip this entirely.
    """
    t = np.atleast_1d(np.asarray(temperature, dtype=np.float64))
    p = np.atleast_1d(np.asarray(pressure, dtype=np.float64))
    if np.any(t <= 0.0) or np.any(p < 0.0):
        msg = "temperature must be > 0 and pressure >= 0"
        raise ValueError(msg)
    if species not in IONISATION_EV:
        msg = f"unknown species {species!r}; have {sorted(IONISATION_EV)}"
        raise KeyError(msg)

    chi = IONISATION_EV[species] * _ELEMENTARY_CHARGE          # J, converted once
    thermal = (2.0 * np.pi * _ELECTRON_MASS * _BOLTZMANN * t) / _PLANCK**2
    # exp underflows to zero far below the ionisation temperature, which is the
    # correct answer there; np.errstate keeps it from being announced.
    with np.errstate(over="ignore", under="ignore"):
        saha = 2.0 * degeneracy_ratio * thermal**1.5 * np.exp(-chi / (_BOLTZMANN * t))
        neutral = float(mole_fraction) * p / (_BOLTZMANN * t)
        return np.asarray(np.sqrt(np.maximum(saha * neutral, 0.0)))


def electron_density_from_solution(
    number_densities: _FloatArray, species_index: int
) -> _FloatArray:
    """Pull :math:`n_e` straight out of a NEMO solution's species field.

    Preferred over :func:`saha_electron_density` whenever the case carried
    electrons, because it inherits the finite-rate chemistry rather than
    assuming equilibrium.
    """
    field = np.asarray(number_densities, dtype=np.float64)
    if field.ndim != 2:
        msg = f"number_densities must be (n_points, n_species), got {field.shape}"
        raise ValueError(msg)
    if not 0 <= species_index < field.shape[1]:
        msg = f"species_index {species_index} outside 0..{field.shape[1] - 1}"
        raise IndexError(msg)
    return np.asarray(field[:, species_index])


def plasma_frequency(electron_density: _FloatArray | float) -> _FloatArray:
    r""":math:`\omega_p = \sqrt{n_e e^2 / \varepsilon_0 m_e}` (rad/s)."""
    n = np.atleast_1d(np.asarray(electron_density, dtype=np.float64))
    return np.asarray(
        np.sqrt(
            np.maximum(n, 0.0)
            * _ELEMENTARY_CHARGE**2
            / (_VACUUM_PERMITTIVITY * _ELECTRON_MASS)
        )
    )


def critical_density(frequency_hz: float) -> float:
    r"""Electron density at which a wave of this frequency stops propagating.

    :math:`n_c = \varepsilon_0 m_e \omega^2 / e^2`. Above it the plasma
    frequency exceeds the signal frequency, the refractive index goes
    imaginary, and the link is not attenuated but *cut off*.
    """
    omega = 2.0 * np.pi * float(frequency_hz)
    return float(
        _VACUUM_PERMITTIVITY * _ELECTRON_MASS * omega**2 / _ELEMENTARY_CHARGE**2
    )


def collision_frequency(
    temperature: _FloatArray | float, pressure: _FloatArray | float
) -> _FloatArray:
    r"""Electron–neutral momentum-transfer collision frequency (s⁻¹).

    :math:`\nu_{en} \approx n_n \sigma \bar v_e` with a momentum-transfer cross
    section of :math:`10^{-19}\,\mathrm{m^2}`, near enough constant for air
    over the 2000–10000 K band, and a Maxwellian mean electron speed. Good to a
    factor of two, which is enough: :math:`\nu` enters the attenuation only
    through :math:`\nu/(\omega^2+\nu^2)`, and at microwave frequencies
    :math:`\omega \gg \nu`, so the result is nearly linear in it.
    """
    t = np.atleast_1d(np.asarray(temperature, dtype=np.float64))
    p = np.atleast_1d(np.asarray(pressure, dtype=np.float64))
    neutral = p / (_BOLTZMANN * t)
    mean_speed = np.sqrt(8.0 * _BOLTZMANN * t / (np.pi * _ELECTRON_MASS))
    return np.asarray(neutral * 1.0e-19 * mean_speed)


def attenuation_db(
    electron_density: _FloatArray,
    collision_rate: _FloatArray,
    frequency_hz: float,
    path_length: _FloatArray | float,
) -> _FloatArray:
    r"""One-way attenuation (dB) accumulated along a path through the sheath.

    Each sample contributes :math:`8.686\,\alpha\,\Delta s` decibels. Where the
    wave is evanescent (:math:`\omega_p > \omega`) the propagating solution does
    not exist and the returned value is :data:`numpy.inf` for that sample rather
    than a large number: a cut-off link is a different physical situation from a
    heavily attenuated one, and reporting it as, say, 300 dB invites a caller to
    imagine a link budget that closes it.
    """
    n_e = np.atleast_1d(np.asarray(electron_density, dtype=np.float64))
    nu = np.atleast_1d(np.asarray(collision_rate, dtype=np.float64))
    ds = np.broadcast_to(np.asarray(path_length, dtype=np.float64), n_e.shape)
    omega = 2.0 * np.pi * float(frequency_hz)
    omega_p = plasma_frequency(n_e)

    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = (omega_p**2 * nu) / (_LIGHT * (omega**2 + nu**2))
        loss = 8.686 * alpha * ds
    return np.asarray(np.where(omega_p >= omega, np.inf, loss))


@dataclass(frozen=True)
class BlackoutWindow:
    """Whether a link closes through the sheath, and by how much it fails."""

    frequency_hz: float
    total_loss_db: float
    peak_electron_density: float
    critical_density: float

    @property
    def cut_off(self) -> bool:
        """Is any part of the path evanescent?"""
        return bool(
            np.isinf(self.total_loss_db)
            or self.peak_electron_density >= self.critical_density
        )

    @property
    def margin_db(self) -> float:
        """Loss the link must tolerate; ``inf`` when cut off."""
        return float("inf") if self.cut_off else float(self.total_loss_db)

    def summary(self) -> str:
        band = f"{self.frequency_hz / 1e9:.2f} GHz"
        if self.cut_off:
            ratio = self.peak_electron_density / max(self.critical_density, 1.0)
            return (
                f"{band}: CUT OFF — peak n_e {self.peak_electron_density:.3e} m^-3 "
                f"is {ratio:.1f}x critical"
            )
        return (
            f"{band}: {self.total_loss_db:.1f} dB through the sheath "
            f"(peak n_e {self.peak_electron_density:.3e} m^-3)"
        )
