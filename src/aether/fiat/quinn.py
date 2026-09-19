"""Oxyacetylene-torch PICA experiments — the first measured validation data.

Quinn, Pickard, Bernstein, Yee, Koo & Radovitzky, "Validation of a Charring
Ablator Material Response Code Against Oxyacetylene Torch Experiments on
PICA Samples", report in-depth temperature histories from cylindrical PICA
samples at three heat fluxes, with the boundary conditions they used
tabulated alongside.

Everything else this package has been checked against is a *model*: FIAT's
published equations, Mutation++'s equilibrium, another code's kinetics.
This is a thermocouple.

What the figures contain
------------------------

Figures 6, 7 and 8 are 250, 500 and 750 W/cm² respectively. Each carries
five colours, and the paper states that **solid lines are experimental
and dotted lines are simulation** — the reverse of the usual convention,
and worth stating plainly because getting it backwards would invert every
comparison made here.

===========  =====================================================
Blue         surface, by two-colour IR pyrometer
Orange       TC 1, shallowest
Green        TC 2
Red          TC 3
Purple       TC 4, deepest
===========  =====================================================

The experimental **surface** trace begins near 1300 K rather than at
ambient. That is not a missing measurement: a two-colour pyrometer has a
low-temperature cutoff and simply reports nothing below it. The
simulated surface trace starts at ambient, and the two meeting where the
pyrometer switches on is a consistency check rather than a discrepancy.

Thermocouple depths
-------------------

Table 1 gives per-sample depths, three samples per heat flux. The paper
averages: *"The experimental data for each case ... was averaged for the
calibration, including the probe locations and the temperature response
at each thermocouple."* The depths here are therefore the mean over each
group of three, and :data:`SAMPLE_DEPTHS` keeps the individual values so
the spread — which reaches 1.9 mm on TC 4 at 750 W/cm² — can be carried
as an uncertainty rather than discarded.

That spread is the dominant experimental uncertainty in this dataset. A
thermocouple 1.9 mm deeper than assumed reads several hundred kelvin
cooler at peak, so any comparison that quotes a temperature error without
also quoting a depth error is quoting the wrong number.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from aether.fiat.surface import AerothermalEnvironment

__all__ = [
    "CALIBRATED_BOUNDARY",
    "CURVE_LOCATIONS",
    "SAMPLE_DEPTHS",
    "QuinnCase",
    "TorchCurve",
    "load_quinn_case",
    "mean_depths",
]

_FloatArray = NDArray[np.float64]

#: Figure colour to measurement location, in order of increasing depth.
CURVE_LOCATIONS = ("surface", "TC1", "TC2", "TC3", "TC4")

_COLOURS = {
    "surface": "Blue",
    "TC1": "Orange",
    "TC2": "Green",
    "TC3": "Red",
    "TC4": "Purple",
}

_FIGURES = {250.0: "Fig6", 500.0: "Fig7", 750.0: "Fig8"}

#: Table 1, per sample: heat flux (W/cm²) and the four TC depths (m).
SAMPLE_DEPTHS: dict[float, tuple[tuple[float, ...], ...]] = {
    250.0: (
        (3.15e-3, 4.52e-3, 5.90e-3, 9.15e-3),
        (3.33e-3, 4.68e-3, 7.32e-3, 9.61e-3),
        (3.64e-3, 5.16e-3, 6.41e-3, 8.17e-3),
    ),
    500.0: (
        (4.35e-3, 6.28e-3, 8.61e-3, 11.41e-3),
        (3.15e-3, 4.29e-3, 7.01e-3, 9.62e-3),
        (3.91e-3, 5.19e-3, 6.83e-3, 11.52e-3),
    ),
    750.0: (
        (3.60e-3, 5.44e-3, 7.54e-3, 10.11e-3),
        (4.60e-3, 7.49e-3, 10.42e-3, 13.24e-3),
        (3.17e-3, 5.19e-3, 7.43e-3, 9.01e-3),
    ),
}

#: Table 2, the paper's calibrated boundary conditions:
#: heat flux (W/cm²) to (rho_e u_e C_H in kg/(m² s), h_r in J/kg).
CALIBRATED_BOUNDARY: dict[float, tuple[float, float]] = {
    250.0: (0.02661, 22538940.0),
    500.0: (0.02661, 45124400.0),
    750.0: (0.03366, 61616820.0),
}


def mean_depths(heat_flux: float) -> _FloatArray:
    """Mean TC depths (m) over the three samples at this heat flux."""
    if heat_flux not in SAMPLE_DEPTHS:
        raise ValueError(f"no samples at {heat_flux} W/cm²; have {sorted(SAMPLE_DEPTHS)}")
    return np.asarray(SAMPLE_DEPTHS[heat_flux], dtype=np.float64).mean(axis=0)


def depth_spread(heat_flux: float) -> _FloatArray:
    """Peak-to-peak TC depth spread (m) across the three samples.

    The experimental uncertainty that matters most, and the one a
    temperature comparison has to be read against.
    """
    d = np.asarray(SAMPLE_DEPTHS[heat_flux], dtype=np.float64)
    return np.asarray(d.max(axis=0) - d.min(axis=0))


@dataclass(frozen=True)
class TorchCurve:
    """One digitised trace."""

    location: str
    depth: float
    """Mean depth below the original surface (m); 0 for the surface."""
    time: _FloatArray
    temperature: _FloatArray
    measured: bool
    """True for the experimental (solid) trace, False for simulation."""

    def at(self, times: _FloatArray) -> _FloatArray:
        """Interpolate onto ``times``, NaN outside the trace's own span.

        NaN rather than a held endpoint: the pyrometer genuinely has no
        reading below its cutoff, and filling that gap with the first
        value it did report would manufacture agreement at exactly the
        place a surface-energy-balance error would show.
        """
        t = np.asarray(times, dtype=np.float64)
        out = np.interp(t, self.time, self.temperature)
        return np.asarray(np.where((t < self.time[0]) | (t > self.time[-1]), np.nan, out))


@dataclass(frozen=True)
class QuinnCase:
    """One heat-flux case: measured and simulated traces plus its BCs."""

    heat_flux: float
    """W/cm², as the paper labels it."""
    curves: dict[str, dict[str, TorchCurve]]
    """``curves[location]["measured" | "simulated"]``."""
    depths: _FloatArray
    depth_spread: _FloatArray
    film_coefficient: float
    recovery_enthalpy: float

    @property
    def duration(self) -> float:
        return float(max(c.time[-1] for by_kind in self.curves.values() for c in by_kind.values()))

    def environment(self, pressure: float = 101325.0) -> AerothermalEnvironment:
        """The paper's calibrated boundary condition, as an environment.

        Constant in time: Table 2 gives one film coefficient and one
        recovery enthalpy per case, and the torch is steady.
        """
        return AerothermalEnvironment(
            film_coefficient=self.film_coefficient,
            recovery_enthalpy=self.recovery_enthalpy,
            pressure=pressure,
        )

    def measured(self, location: str) -> TorchCurve:
        return self.curves[location]["measured"]

    def simulated(self, location: str) -> TorchCurve:
        return self.curves[location]["simulated"]


def load_quinn_case(directory: str | Path, heat_flux: float) -> QuinnCase:
    """Load one figure's ten digitised traces.

    Parameters
    ----------
    directory:
        Holds ``Quinn-et-al-Fig{6,7,8}_{Colour}_{Solid,Dashed}.csv``,
        each a headerless ``time, temperature`` CSV.
    heat_flux:
        250, 500 or 750 W/cm².
    """
    if heat_flux not in _FIGURES:
        raise ValueError(f"no figure for {heat_flux} W/cm²; have {sorted(_FIGURES)}")
    root = Path(directory)
    figure = _FIGURES[heat_flux]
    depths = mean_depths(heat_flux)

    curves: dict[str, dict[str, TorchCurve]] = {}
    for index, location in enumerate(CURVE_LOCATIONS):
        colour = _COLOURS[location]
        by_kind: dict[str, TorchCurve] = {}
        # Solid is experimental and dashed is simulation — the paper's own
        # convention, and the reverse of the usual one.
        for kind, suffix in (("measured", "Solid"), ("simulated", "Dashed")):
            path = root / f"Quinn-et-al-{figure}_{colour}_{suffix}.csv"
            if not path.exists():
                raise FileNotFoundError(f"missing digitised trace {path}")
            raw = np.loadtxt(path, delimiter=",")
            order = np.argsort(raw[:, 0])
            time, temperature = raw[order, 0], raw[order, 1]
            # Collapse the repeated abscissae a digitiser produces on steep
            # sections, so the trace is a function of time.
            unique, inverse = np.unique(np.round(time, 6), return_inverse=True)
            mean_t = np.zeros(unique.size)
            np.add.at(mean_t, inverse, temperature)
            mean_t /= np.bincount(inverse, minlength=unique.size)
            by_kind[kind] = TorchCurve(
                location=location,
                depth=0.0 if location == "surface" else float(depths[index - 1]),
                time=unique,
                temperature=mean_t,
                measured=kind == "measured",
            )
        curves[location] = by_kind

    film, h_r = CALIBRATED_BOUNDARY[heat_flux]
    return QuinnCase(
        heat_flux=heat_flux,
        curves=curves,
        depths=depths,
        depth_spread=depth_spread(heat_flux),
        film_coefficient=film,
        recovery_enthalpy=h_r,
    )
