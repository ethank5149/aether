"""NASA arc-jet PICA tests — measured recession, and FIAT's own predictions.

Milos, F. S. and Chen, Y.-K., "Ablation and Thermal Response Property Model
Validation for Phenolic Impregnated Carbon Ablator," *Journal of Spacecraft
and Rockets* **47**(5), 2010, pp. 786–805, doi:10.2514/1.42949 (also AIAA
2009-0262).

This is what verification task I-V4 has been waiting for. Its criterion is
*recession within 5% of a FIAT reference case*, and until now the project
had neither the reference case nor any way to run FIAT. This paper supplies
**72 arc-jet models** with measured centreline recession across 22
conditions, together with the stagnation heat flux, pressure and enthalpy
each was run at — and, for two conditions, FIAT's own predicted curves at
90, 100 and 110% of nominal heating.

Comparing against those predicted curves is as close to the stated
criterion as is reachable without the export-controlled program itself:
FIAT's answer, on a published case, at a stated environment.

Why this dataset and not the torch data
---------------------------------------

The oxyacetylene-torch experiments in :mod:`aether.fiat.quinn` are
all at roughly one atmosphere. These span **2.3 to 84.4 kPa** — three
decades below ambient at the low end, which is entry's regime and exactly
where the Heritage and MEDLI2 conductivity models disagree by a factor of
four. Heat flux spans 107 to 1102 W/cm².

Uncertainties worth carrying
----------------------------

*Two enthalpies per condition.* The paper reports a facility estimate,
from laminar heat-transfer correlations, and a DPLR (CFD) centreline
value. They differ by up to **31%** — condition 19 is 26.6 against
18.4 MJ/kg. Which one is used as the recovery enthalpy is a modelling
choice with a large lever arm, so both are kept and neither is chosen
here.

*Argon dilution is bracketed, not known.* The ARC tests run air/argon
mixtures, and the paper reports a minimum argon fraction assuming perfect
mixing and a maximum assuming none. Where those differ the true gas
composition is unknown between them; condition 5ab spans 0.082 to 0.257.

*Model-to-model scatter.* Several conditions were run with up to eight
models. Condition 13 gives recessions from 11.58 to 15.33 mm — a 28%
spread on nominally identical tests. That scatter, not our numerics, sets
the floor on what any 5% criterion can mean here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ANALYSIS_CASES",
    "ARC_CONDITIONS",
    "JSC_CONDITIONS",
    "MODELS",
    "TC_PLACEMENTS",
    "AnalysisCase",
    "ArcjetCondition",
    "ArcjetModel",
    "condition",
    "load_fig26_thermocouples",
    "models_for",
    "recession_statistics",
]

_FloatArray = NDArray[np.float64]

#: Table 1. Axial thermocouple depth (m) by placement option, TC 1..5.
#:
#: Options A, B and C share a 3.81 mm ladder offset by 1.27 mm steps; TC 5
#: sits at 30.48 mm in every option. Option D is the off-axis arrangement of
#: Fig. 7 and carries TC 6..10 instead, at radii out to 44.45 mm — those are
#: not centreline measurements and are excluded here.
TC_PLACEMENTS: dict[str, tuple[float, ...]] = {
    "A": (3.81e-3, 7.62e-3, 11.43e-3, 15.24e-3, 30.48e-3),
    "B": (5.08e-3, 8.89e-3, 12.70e-3, 16.51e-3, 30.48e-3),
    "C": (6.35e-3, 10.16e-3, 13.97e-3, 17.78e-3, 30.48e-3),
}


@dataclass(frozen=True)
class ArcjetCondition:
    """One tabulated test environment (Table 2 or 3)."""

    number: str
    facility: str
    heat_flux: float
    """Stagnation-point cold-wall heat flux, W/m²."""
    pressure: float
    """Stagnation pressure, Pa."""
    facility_enthalpy: float
    """Centreline enthalpy from facility correlations, J/kg."""
    dplr_enthalpy: float
    """Centreline enthalpy from DPLR, J/kg."""
    argon_fraction: tuple[float, float] | None
    """(perfectly mixed, unmixed) bounds; ``None`` for the JSC tests."""
    oxygen_fraction: float | None
    """JSC only: oxygen mass fraction, perfectly mixed."""

    @property
    def enthalpy_disagreement(self) -> float:
        """Relative gap between the two reported enthalpies."""
        return abs(self.facility_enthalpy / self.dplr_enthalpy - 1.0)


def _arc(
    number: str,
    facility: str,
    q: float,
    p: float,
    h_fac: float,
    h_dplr: float,
    ar_min: float,
    ar_max: float,
) -> ArcjetCondition:
    return ArcjetCondition(
        number=number,
        facility=facility,
        heat_flux=q * 1.0e4,
        pressure=p * 1.0e3,
        facility_enthalpy=h_fac * 1.0e6,
        dplr_enthalpy=h_dplr * 1.0e6,
        argon_fraction=(ar_min, ar_max),
        oxygen_fraction=None,
    )


#: Table 2 — NASA Ames, air/argon mixtures.
ARC_CONDITIONS: dict[str, ArcjetCondition] = {
    c.number: c
    for c in (
        _arc("1", "AHF 18", 107, 2.3, 15.9, 15.2, 0.276, 0.276),
        _arc("2ab", "AHF 18", 143, 3.8, 17.0, 16.5, 0.173, 0.173),
        _arc("3ab", "AHF 7", 154, 13.3, 9.7, 8.4, 0.188, 0.269),
        _arc("4ab", "AHF 18", 169, 5.0, 17.8, 17.0, 0.142, 0.142),
        _arc("5ab", "AHF 7", 190, 45.7, 7.2, 7.3, 0.082, 0.257),
        _arc("6ab", "AHF 18", 246, 8.5, 20.0, 19.3, 0.108, 0.108),
        _arc("7ab", "AHF 7", 255, 29.8, 11.1, 11.1, 0.113, 0.186),
        _arc("8", "IHF 13", 395, 17.2, 22.8, 21.4, 0.080, 0.080),
        _arc("9", "IHF 13", 430, 17.9, 24.1, 22.4, 0.096, 0.152),
        _arc("10ab", "AHF 7", 480, 31.9, 20.7, 19.0, 0.113, 0.131),
        _arc("11ab", "IHF 13", 548, 19.4, 29.6, 27.2, 0.080, 0.080),
        _arc("12", "IHF 13", 552, 27.3, 25.3, 23.3, 0.076, 0.076),
        _arc("13", "IHF 13", 712, 33.2, 29.4, 26.7, 0.093, 0.098),
        _arc("14", "IHF 13", 744, 31.0, 32.0, 29.2, 0.076, 0.080),
        _arc("15", "IHF 6", 762, 46.6, 26.6, 23.7, 0.084, 0.153),
        _arc("16ab", "IHF 8", 970, 63.4, 25.2, 23.6, 0.085, 0.150),
        _arc("17", "IHF 6", 1102, 84.4, 28.7, 25.6, 0.078, 0.146),
        # Condition 18 is the dual-pulse test: a high-flux pulse followed by
        # a lower one on the same model, which is why Table 4 reports two
        # exposures and only one recession for it.
        _arc("18a", "IHF 13", 425, 16.2, 23.8, 22.2, 0.088, 0.132),
        _arc("18b", "IHF 13", 161, 9.2, 12.6, 15.5, 0.093, 0.193),
    )
}

#: Table 3 — NASA Johnson, nitrogen/oxygen mixtures.
#:
#: The oxygen fraction is swept from 0 to 0.30 at nearly constant heat flux
#: and pressure, which isolates oxidation from heating: the measured
#: recession runs 1.75 mm at zero oxygen to 24.1 mm at 30%, a **fourteenfold**
#: change with the thermal environment essentially fixed. No test in the
#: Ames set separates the two effects like this.
JSC_CONDITIONS: dict[str, ArcjetCondition] = {
    c.number: c
    for c in (
        ArcjetCondition("19", "TP2 5", 416e4, 17.9e3, 26.6e6, 18.4e6, None, 0.00),
        ArcjetCondition("20", "TP2 5", 408e4, 18.5e3, 25.5e6, 17.8e6, None, 0.10),
        ArcjetCondition("21", "TP2 5", 407e4, 18.5e3, 25.2e6, 17.8e6, None, 0.23),
        ArcjetCondition("22", "TP2 5", 415e4, 18.4e3, 26.3e6, 18.3e6, None, 0.30),
    )
}


def condition(number: str) -> ArcjetCondition:
    """Look up a condition in either facility's table."""
    if number in ARC_CONDITIONS:
        return ARC_CONDITIONS[number]
    if number in JSC_CONDITIONS:
        return JSC_CONDITIONS[number]
    # Table 4 keys sub-cases as e.g. "4a"; Table 2 groups them as "4ab".
    for key in ARC_CONDITIONS:
        if key.endswith("ab") and number[:-1] == key[:-2] and number[-1] in "ab":
            return ARC_CONDITIONS[key]
    raise KeyError(f"no arcjet condition {number!r}")


@dataclass(frozen=True)
class ArcjetModel:
    """One tested model (Table 4)."""

    condition: str
    exposure: float
    """s."""
    model_id: str
    tc_option: str | None
    recession: float | None
    """Centreline recession (m); ``None`` where not measured."""
    peak_surface_temperature: float | None
    """K; ``None`` where not measured."""

    @property
    def depths(self) -> tuple[float, ...]:
        """Thermocouple depths (m), empty for uninstrumented models."""
        if self.tc_option is None or self.tc_option not in TC_PLACEMENTS:
            return ()
        return TC_PLACEMENTS[self.tc_option]


def _m(c: str, t: float, i: str, o: str | None, s: float | None, k: float | None) -> ArcjetModel:
    return ArcjetModel(c, t, i, o, None if s is None else s * 1e-3, k)


#: Table 4 — every model tested, with measured centreline recession.
MODELS: tuple[ArcjetModel, ...] = (
    _m("1", 55, "AA-43-209-N", "A", 2.33, 2033),
    _m("1", 55, "AA-43-210-N", "A", 2.27, 2041),
    _m("2a", 200, "AT-008", None, 12.66, 2177),
    _m("2b", 400, "AT-007", None, 24.72, 2196),
    _m("3a", 33, "AA-44-210-N", "B", 2.93, 2161),
    _m("3a", 33, "AA-44-211-N", "B", 2.92, 2174),
    _m("3b", 66, "AA-44-212-N", "B", 5.61, 2163),
    _m("4a", 33, "AA-43-211-N", "A", 2.27, 2243),
    _m("4a", 33, "AA-43-212-N", "A", 2.20, 2245),
    _m("4b", 60, "AA-43-208-N", "A", 4.28, 2248),
    _m("4b", 60, "1403", "D", 4.58, 2259),
    _m("4b", 60, "1404", "D", 4.76, 2284),
    _m("5a", 30, "AA-44-204-N", "B", 4.78, 1989),
    _m("5a", 30, "AA-44-205-N", "B", 3.79, 2004),
    _m("5b", 60, "AA-44-206-N", "B", 8.81, 2065),
    _m("6a", 42, "AA-43-213-N", "A", 3.71, 2407),
    _m("6a", 42, "AA-43-215-N", "A", 3.65, 2414),
    _m("6a", 42, "AA-43-207-N", "A", 3.95, 2401),
    _m("6a", 42, "AA-43-216-N", "B", 3.52, 2432),
    _m("6a", 42, "AA-43-227-N", "B", 3.68, 2419),
    _m("6a", 42, "1405", "D", 4.12, 2437),
    _m("6a", 42, "1406", "D", 3.69, 2417),
    _m("6b", 60, "AA-43-223-N", "A", 5.21, 2434),
    _m("6b", 60, "AA-43-228-N", "A", 5.23, 2409),
    _m("7a", 30, "AA-44-201-N", "B", 3.87, 2331),
    _m("7a", 30, "AA-44-202-N", "B", 4.33, 2320),
    _m("7b", 60, "AA-44-203-N", "B", 8.52, 2359),
    _m("8", 34, "AA-43-219-N", "B", 4.61, 2730),
    _m("8", 34, "AA-43-218-N", "B", 4.43, 2735),
    _m("9", 120, "AT-011", None, 16.45, 2703),
    _m("9", 120, "AT-019", None, 18.66, 2718),
    _m("9", 120, "AT-025", None, 19.06, 2713),
    _m("9", 120, "AT-026", None, 19.03, 2697),
    _m("9", 120, "AA-40-001", None, 17.71, 2682),
    _m("9", 120, "AA-40-002", None, 17.64, 2677),
    _m("9", 120, "AA-40-003-N", None, 16.25, 2662),
    _m("10a", 34, "AA-44-207-N", "B", 6.32, 2757),
    _m("10a", 34, "AA-44-208-N", "B", 6.58, 2764),
    _m("10b", 68, "AA-44-209-N", "B", 13.63, 2738),
    _m("11a", 30, "AA-43-221-N", "B", 4.18, 2990),
    _m("11a", 30, "AA-43-224-N", "B", 4.55, 2923),
    _m("11a", 30, "AA-43-226-N", "B", 4.30, 2943),
    _m("11b", 40, "AA-43-222-N", "B", 5.89, 2988),
    _m("11b", 40, "AA-43-225-N", "B", 5.92, 2934),
    _m("12", 30, "AA-43-233-N", "B", 5.41, 2963),
    _m("12", 30, "AA-43-234-N", "B", 4.97, 2963),
    _m("13", 70, "AT-012", None, 12.57, 2984),
    _m("13", 70, "AT-020", None, 14.56, 3030),
    _m("13", 70, "AT-027", None, 15.00, 3053),
    _m("13", 70, "AT-028", None, 15.33, 3015),
    _m("13", 70, "AA-40-003", None, 13.82, 2994),
    _m("13", 70, "AA-40-004", None, 11.58, 2994),
    _m("13", 70, "AA-40-001-N", None, 13.79, 2969),
    _m("13", 70, "AA-40-002-N", None, 12.83, 2984),
    _m("14", 27, "AA-44-218-N", "C", 5.13, 3030),
    _m("15", 45, "AA-44-223-N", "C", 12.74, 3098),
    _m("15", 45, "AA-44-224-N", "C", 12.91, 3098),
    _m("16a", 30, "AT-001", None, 11.97, 3020),
    _m("16a", 30, "AT-004", None, 11.33, 3051),
    _m("16b", 40, "AT-003", None, 15.15, 3020),
    _m("16b", 40, "AT-005", None, 15.56, 3030),
    _m("17", 10, "AA-44-219-N", "C", 4.84, 3233),
    _m("18a", 113, "AT-009", None, None, 2728),
    _m("18b", 128, "AT-009", None, 26.95, 2218),
    _m("18a", 113, "AT-010", None, None, 2738),
    _m("18b", 128, "AT-010", None, 27.05, 2213),
    _m("19", 120, "AA-44-214-N", "B", 1.75, None),
    _m("20", 120, "AA-44-216-N", "B", 12.0, None),
    _m("21", 120, "AA-44-213-N", "B", 20.5, None),
    _m("22", 120, "AA-44-215-N", "B", 24.1, None),
    _m("22", 120, "AA-44-229-N", "B", 23.9, None),
)


@dataclass(frozen=True)
class AnalysisCase:
    """One of the seven cases the paper analyses in detail (Table 5)."""

    number: int
    condition: str
    heat_flux: float
    """W/m²."""
    pressure: float
    """Pa."""
    exposure: float
    """s."""


#: Table 5, spanning a decade in heat flux and 36x in pressure.
ANALYSIS_CASES: tuple[AnalysisCase, ...] = (
    AnalysisCase(1, "1", 107e4, 2.3e3, 55.0),
    AnalysisCase(2, "4b", 169e4, 5.0e3, 60.0),
    AnalysisCase(3, "6a", 246e4, 8.5e3, 42.0),
    AnalysisCase(4, "8", 395e4, 17.2e3, 34.0),
    AnalysisCase(5, "12", 552e4, 27.3e3, 30.0),
    AnalysisCase(6, "14", 744e4, 31.0e3, 27.0),
    AnalysisCase(7, "17", 1102e4, 84.4e3, 10.0),
)


def models_for(condition_number: str) -> tuple[ArcjetModel, ...]:
    """Every model tested at a condition."""
    return tuple(m for m in MODELS if m.condition == condition_number)


def recession_statistics(condition_number: str) -> tuple[float, float, float]:
    """``(mean, min, max)`` measured recession (m) at a condition.

    The spread is the experimental scatter on nominally identical tests,
    and it is what any accuracy claim has to be read against. At condition
    13, eight models scatter by 28% of the mean.
    """
    values = [m.recession for m in models_for(condition_number) if m.recession is not None]
    if not values:
        raise ValueError(f"no measured recession at condition {condition_number!r}")
    return float(np.mean(values)), float(min(values)), float(max(values))


def consumption_recession(
    directory: str | Path, tc_option: str = "B"
) -> tuple[_FloatArray, _FloatArray]:
    """Recession history inferred from when each thermocouple dies.

    Each measured trace in Fig. 26 ends when the receding surface reaches
    that thermocouple, so the end times paired with the known installation
    depths are a recession-versus-time curve — one the paper does not plot
    for this condition. It is coarse, four points and a survivor, but it
    is a *measured* recession history at 18.5 kPa, and the closest thing
    this dataset offers to a transient recession check.
    """
    depths = TC_PLACEMENTS[tc_option]
    times, values = [], []
    for tc in (1, 2, 3, 4):
        curves = load_fig26_thermocouples(directory, tc)
        times.append(float(curves["measured"][0].max()))
        values.append(depths[tc - 1])
    return np.asarray(times), np.asarray(values)


def load_fig26_thermocouples(
    directory: str | Path, thermocouple: int
) -> dict[str, tuple[_FloatArray, _FloatArray]]:
    """Figure 26: condition 21, model AA-44-213-N, TC 1 to 5.

    Returns ``{"measured" | "90" | "100" | "110": (time_s, temperature_K)}``
    — the measurement plus FIAT's prediction at three heating levels. The
    heating bracket is the paper's own way of expressing environmental
    uncertainty, and it is wide: comparing against the 100% curve alone
    overstates how sharply the prediction is defined.

    The measured traces stop at different times, and that is data rather
    than missing data. Condition 21 recedes 20.5 mm in 120 s past
    placement-B thermocouples at 5.08, 8.89, 12.70, 16.51 and 30.48 mm, so
    each is consumed in turn — see :func:`consumption_recession`. A
    consequence worth noting before comparing peaks: the shallow
    thermocouples are cut off while still rising, so TC 1 reports a *lower*
    maximum than TC 2 despite being hotter throughout its life.
    """
    if not 1 <= thermocouple <= 5:
        raise ValueError(f"thermocouple must be 1..5, got {thermocouple}")
    root = Path(directory)
    stem = f"MC2010-Fig26_TC{thermocouple}"
    sources = {
        "measured": f"{stem}_AA-44-213-N.csv",
        "90": f"{stem}_90percent-heating.csv",
        "100": f"{stem}_100percent-heating.csv",
        "110": f"{stem}_110percent-heating.csv",
    }
    out: dict[str, tuple[_FloatArray, _FloatArray]] = {}
    for key, name in sources.items():
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"missing digitised trace {path}")
        raw = np.loadtxt(path, delimiter=",")
        order = np.argsort(raw[:, 0])
        time, temperature = raw[order, 0], raw[order, 1]
        unique, inverse = np.unique(np.round(time, 6), return_inverse=True)
        mean = np.zeros(unique.size)
        np.add.at(mean, inverse, temperature)
        mean /= np.bincount(inverse, minlength=unique.size)
        out[key] = (unique, mean)
    return out
