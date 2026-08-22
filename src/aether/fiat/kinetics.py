"""Decomposition kinetics: calibration against thermogravimetric data.

FIAT Eq. (8) needs an Arrhenius triplet per decomposing component,

.. math::

    \\frac{\\partial \\rho_i}{\\partial \\theta}\\bigg|_y =
      -A_i e^{-E_i/RT}\\rho_{v i}
      \\left(\\frac{\\rho_i - \\rho_{r i}}{\\rho_{v i}}\\right)^{\\psi_i},

and those triplets are the least transferable part of a material model.

**Published PICA kinetics now exist in ``reference/`` and are implemented
in** :mod:`aether.fiat.pica_kinetics` — Torres-Herrador et al.
2019 (a six-reaction parallel set, in FIAT Eq. (8)'s form) and 2020 (a
competitive scheme, outside it). Use those for PICA.

This module remains the general machinery: it calibrates *any* material
against a thermogravimetric scan, which is what you need for a material
whose kinetics are not published — the ordinary case.

So this module does not assert triplets it cannot source. It provides
the two things that are actually useful:

* :func:`tga_mass_fraction` — forward-model a thermogravimetric scan
  (constant heating rate, mass fraction against temperature) from a set
  of components, which is the measurement kinetics are calibrated
  against;
* :func:`fit_arrhenius` — recover the triplets from such a scan.

Given a TGA curve for a real material, those two close the loop. Without
one, :func:`calibrated_components` builds a set from three *stated,
checkable* targets — onset temperature, peak-rate temperature and char
yield — so that the resulting material is reproducible and its
assumptions are visible, rather than being three magic numbers.

.. warning::

   Everything here assumes **independent parallel reactions**, because
   that is what FIAT Eq. (8) is. For carbon/phenolic that assumption is
   known to fail across heating rates: the measured pyrolysis peak moves
   *down* in temperature as the rate rises, and no parallel set can
   reproduce that. See :mod:`aether.fiat.pica_kinetics`. A fit
   from this module is valid over the range of heating rates it was
   calibrated on and should not be extrapolated far outside it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.integrate
import scipy.optimize
from numpy.typing import ArrayLike, NDArray

from aether.thermal.material import GAS_CONSTANT, ArrheniusComponent

__all__ = [
    "TgaTargets",
    "calibrated_components",
    "fit_arrhenius",
    "peak_rate_temperature",
    "tga_mass_fraction",
]

_FloatArray = NDArray[np.float64]


def tga_mass_fraction(
    components: list[ArrheniusComponent],
    weights: ArrayLike,
    temperatures: ArrayLike,
    heating_rate: float,
    initial_temperature: float = 300.0,
) -> _FloatArray:
    """Residual mass fraction of a constant-rate TGA scan.

    Integrates Eq. (8) along :math:`T = T_0 + \\beta\\theta`, which turns
    the kinetics into an ODE in temperature — the form a scan is actually
    reported in.

    Parameters
    ----------
    components:
        Decomposing components.
    weights:
        Mass weight of each component in the composite, same length. For
        FIAT Eq. (7) these are :math:`(\\Gamma, \\Gamma, 1-\\Gamma)`.
    temperatures:
        Strictly increasing scan temperatures (K).
    heating_rate:
        :math:`\\beta` (K/s). TGA scans are usually 5–20 K/min; the
        recovered activation energy depends on it, which is why a single
        scan under-determines the triplet and why
        :func:`fit_arrhenius` accepts several.
    initial_temperature:
        Scan start (K).
    """
    t = np.asarray(temperatures, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if t.ndim != 1 or t.size < 2 or np.any(np.diff(t) <= 0.0):
        raise ValueError("temperatures must be strictly increasing with >= 2 points")
    if w.shape != (len(components),):
        raise ValueError(f"weights must have one entry per component, got {w.shape}")
    if not (np.isfinite(heating_rate) and heating_rate > 0.0):
        raise ValueError(f"heating_rate must be finite and > 0, got {heating_rate}")
    if t[0] < initial_temperature:
        raise ValueError("temperatures must start at or above initial_temperature")

    rho_v = np.array([c.virgin_density for c in components])
    rho_r = np.array([c.char_density for c in components])
    pre = np.array([c.pre_exponential for c in components])
    act = np.array([c.activation_energy for c in components])
    order = np.array([c.reaction_order for c in components])

    def rhs(temp: float, rho: _FloatArray) -> _FloatArray:
        excess = np.clip((rho - rho_r) / rho_v, 0.0, None)
        rate = pre * np.exp(-act / (GAS_CONSTANT * temp)) * rho_v * excess**order
        # d rho / dT = (d rho / d theta) / beta.
        return np.asarray(-rate / heating_rate)

    solution = scipy.integrate.solve_ivp(
        rhs,
        (initial_temperature, float(t[-1])),
        rho_v,
        t_eval=t,
        method="LSODA",
        rtol=1e-10,
        atol=1e-12,
    )
    if not solution.success:  # pragma: no cover - integrator failure
        raise RuntimeError(f"TGA integration failed: {solution.message}")
    total0 = float(np.sum(w * rho_v))
    return np.asarray(np.sum(w[:, None] * solution.y, axis=0) / total0)


def peak_rate_temperature(
    components: list[ArrheniusComponent],
    weights: ArrayLike,
    heating_rate: float,
    bracket: tuple[float, float] = (300.0, 2000.0),
    points: int = 2000,
) -> float:
    """Temperature of maximum mass-loss rate in a TGA scan (K).

    The single most repeatable number in a thermogravimetric
    measurement, and the one a kinetics fit is most tightly constrained
    by.
    """
    t = np.linspace(bracket[0], bracket[1], points)
    mass = tga_mass_fraction(components, weights, t, heating_rate, bracket[0])
    rate = -np.gradient(mass, t)
    return float(t[int(np.argmax(rate))])


def fit_arrhenius(
    temperatures: ArrayLike,
    mass_fraction: ArrayLike,
    heating_rate: float,
    template: list[ArrheniusComponent],
    weights: ArrayLike,
    fit_order: bool = False,
) -> list[ArrheniusComponent]:
    """Recover Arrhenius triplets from a measured TGA scan.

    ``template`` supplies the component count and the virgin/char
    densities — the composition, which comes from elsewhere — and its
    triplets are used as the starting guess. Only the pre-exponentials
    and activation energies are fitted unless ``fit_order`` is set;
    reaction order is poorly identified from a single scan and is
    conventionally fixed at an integer.

    :math:`\\log A` is fitted rather than :math:`A` because
    pre-exponentials span decades, and the two are strongly correlated
    with :math:`E` along the classic kinetic compensation direction —
    which is exactly why a fit to one heating rate should not be
    reported as *the* kinetics of a material.
    """
    t = np.asarray(temperatures, dtype=np.float64)
    y = np.asarray(mass_fraction, dtype=np.float64)
    if t.shape != y.shape:
        raise ValueError("temperatures and mass_fraction must have the same shape")
    if np.any(y > 1.0 + 1e-9) or np.any(y < 0.0):
        raise ValueError("mass_fraction must lie in [0, 1]")
    # Inert components (a zero pre-exponential, which is how FIAT writes a
    # non-decomposing constituent) are held fixed. Fitting them would ask the
    # optimiser to find log(0), and there is nothing in a TGA scan that
    # constrains a reaction that never happens.
    active = [i for i, c in enumerate(template) if c.pre_exponential > 0.0]
    if not active:
        raise ValueError("no decomposing components to fit: all pre-exponentials are 0")
    n = len(active)

    def unpack(x: _FloatArray) -> list[ArrheniusComponent]:
        out = list(template)
        for j, i in enumerate(active):
            c = template[i]
            order = x[2 * n + j] if fit_order else c.reaction_order
            out[i] = ArrheniusComponent(
                pre_exponential=float(np.exp(x[j])),
                activation_energy=float(x[n + j]),
                reaction_order=float(order),
                virgin_density=c.virgin_density,
                char_density=c.char_density,
            )
        return out

    def residual(x: _FloatArray) -> _FloatArray:
        try:
            model = tga_mass_fraction(unpack(x), weights, t, heating_rate, float(t[0]))
        except (ValueError, RuntimeError):  # pragma: no cover - bad iterate
            return np.full(t.size, 1e3)
        return np.asarray(model - y)

    blocks: list[_FloatArray] = [
        np.log(np.array([template[i].pre_exponential for i in active])),
        np.array([template[i].activation_energy for i in active]),
    ]
    lowers: list[_FloatArray] = [np.full(n, -5.0), np.full(n, 1.0e4)]
    uppers: list[_FloatArray] = [np.full(n, 60.0), np.full(n, 5.0e5)]
    if fit_order:
        blocks.append(np.array([template[i].reaction_order for i in active]))
        lowers.append(np.full(n, 0.5))
        uppers.append(np.full(n, 4.0))
    x0 = np.concatenate(blocks)
    lower = np.concatenate(lowers)
    upper = np.concatenate(uppers)
    result = scipy.optimize.least_squares(
        residual, x0, bounds=(lower, upper), xtol=1e-12, ftol=1e-12
    )
    return unpack(result.x)


@dataclass(frozen=True)
class TgaTargets:
    """Stated, checkable targets a kinetics set is built to hit.

    Attributes
    ----------
    onset_temperature:
        Temperature at which 2% of the decomposable mass has been lost
        (K), at :attr:`heating_rate`.
    peak_temperature:
        Temperature of maximum mass-loss rate (K).
    char_yield:
        Residual mass fraction of the whole composite once decomposition
        is complete.
    heating_rate:
        Scan rate the first two are quoted at (K/s). Defaults to
        20 K/min, a common TGA rate. Onset and peak both shift upward
        with rate — for this material, 515 K and 762 K at 5 K/min
        against 557 K and 799 K at 20 K/min — so quoting either without
        its rate is meaningless.

    The defaults describe
    :func:`~aether.fiat.materials.pica_like_material`, and are
    the *targets it was built to*, not measurements of PICA. They put
    decomposition in the band a phenolic resin occupies and pin the char
    yield to the published virgin and char bulk densities. Substitute a
    real scan and :func:`fit_arrhenius` the moment one is available.
    """

    onset_temperature: float = 557.0
    peak_temperature: float = 799.0
    char_yield: float = 227.0 / 274.0
    heating_rate: float = 20.0 / 60.0

    def __post_init__(self) -> None:
        if not 0.0 < self.char_yield < 1.0:
            raise ValueError(f"char_yield must be in (0, 1), got {self.char_yield}")
        if not 0.0 < self.onset_temperature < self.peak_temperature:
            raise ValueError("need 0 < onset_temperature < peak_temperature")
        if not self.heating_rate > 0.0:
            raise ValueError("heating_rate must be > 0")


def calibrated_components(
    template: list[ArrheniusComponent],
    weights: ArrayLike,
    targets: TgaTargets,
) -> list[ArrheniusComponent]:
    """Adjust a template's activation energies to hit stated TGA targets.

    A minimal, honest substitute for measured kinetics: it does not
    invent a triplet, it shifts a template until the resulting scan has
    the stated onset and peak-rate temperatures. The char yield is a
    property of the *composition* — the virgin and char densities in the
    template — and is checked rather than fitted, because moving it
    would contradict the published bulk densities.

    The pre-exponentials are held and only :math:`E_i` is moved, by a
    single shared multiplicative factor per component, because that is
    the direction the peak temperature is actually sensitive to; fitting
    both from two scalar targets would be under-determined.
    """
    w = np.asarray(weights, dtype=np.float64)
    implied_yield = float(
        np.sum(w * [c.char_density for c in template])
        / np.sum(w * [c.virgin_density for c in template])
    )
    if abs(implied_yield - targets.char_yield) > 0.02:
        raise ValueError(
            f"the template's composition implies a char yield of "
            f"{implied_yield:.4f}, but the target is {targets.char_yield:.4f}. "
            f"Char yield follows from the virgin and char densities, which are "
            f"set by the published bulk densities; change those, not the kinetics."
        )

    def scan(scale: float) -> tuple[float, float]:
        scaled = [
            ArrheniusComponent(
                pre_exponential=c.pre_exponential,
                activation_energy=c.activation_energy * scale,
                reaction_order=c.reaction_order,
                virgin_density=c.virgin_density,
                char_density=c.char_density,
            )
            for c in template
        ]
        t = np.linspace(300.0, 2000.0, 1500)
        mass = tga_mass_fraction(scaled, w, t, targets.heating_rate, 300.0)
        decomposable = 1.0 - implied_yield
        onset_idx = int(np.argmax(mass <= 1.0 - 0.02 * decomposable))
        peak = float(t[int(np.argmax(-np.gradient(mass, t)))])
        return float(t[onset_idx]), peak

    def error(scale: float) -> float:
        _, peak = scan(scale)
        return peak - targets.peak_temperature

    lo, hi = 0.5, 2.0
    if error(lo) * error(hi) > 0.0:
        raise ValueError(
            f"the target peak temperature {targets.peak_temperature:.0f} K is not "
            f"reachable by scaling this template's activation energies over "
            f"[{lo}, {hi}]; the pre-exponentials need changing too."
        )
    scale = float(scipy.optimize.brentq(error, lo, hi, xtol=1e-10))
    return [
        ArrheniusComponent(
            pre_exponential=c.pre_exponential,
            activation_energy=c.activation_energy * scale,
            reaction_order=c.reaction_order,
            virgin_density=c.virgin_density,
            char_density=c.char_density,
        )
        for c in template
    ]
