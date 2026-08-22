"""Published PICA pyrolysis kinetics, and why FIAT's Eq. (8) cannot hold them all.

Two calibrated models from Torres-Herrador and co-workers are
implemented here, together with the conversion needed to express either
in FIAT's rate normalisation.

Sources, both in ``reference/``:

* **[TH2020]** Torres-Herrador, Coheur, Panerai, Magin, Arnst, Mansour &
  Blondeau, "Competitive kinetic model for the pyrolysis of the Phenolic
  Impregnated Carbon Ablator," *Aerospace Science and Technology* **100**
  (2020) 105784.
* **[TH2019]** Torres-Herrador, Meurisse, Panerai, Blondeau, Lachaud,
  Bessire, Magin & Mansour, "A high heating rate pyrolysis model for the
  Phenolic Impregnated Carbon Ablator (PICA) based on mass spectroscopy
  experiments," *J. Analytical and Applied Pyrolysis* **141** (2019)
  104625.
* **[RAB2014]** Rabinovitch, Marx & Blanquart, "Pyrolysis Gas Composition
  for a Phenolic Impregnated Carbon Ablator Heatshield," AIAA 2014-2246.
  Source of the elemental bookkeeping below, including FIATv3's own
  pyrolysis-gas composition.
* **[SCO2017]** Scoggins, Leroy, Bellas-Chatzigeorgis, Dias & Magin,
  "Thermodynamic properties of carbon-phenolic gas mixtures," *Aerospace
  Science and Technology* **66** (2017) 177-192. Source of the single-step
  pure-carbon-char pyrolysis relation used to audit those compositions.

The structural finding
----------------------

FIAT Eq. (8) models pyrolysis as **independent parallel reactions**, one
per solid component. [TH2020] states plainly that this form cannot
reproduce PICA's measured behaviour across heating rates:

    "In solid-phase pyrolysis, it is usually observed that as the heating
    rate increases, the decomposition curves shift towards higher
    temperatures. This behavior is commonly attributed to the thermal lag
    effects and can be usually reproduced assuming independent parallel
    reactions. However, different experimental evidences show that this
    is not the case for the pyrolysis of carbon/phenolic. For example,
    Stokes observed that at heating rates higher than 300 K/min the
    pyrolysis peak shifted towards *lower* temperatures."

and, of the parallel formulation, that it is

    "not able to reproduce this effect due to their mathematical
    formulation."

This is not a calibration problem. A sum of independent first-order-ish
Arrhenius terms always shifts its peak *up* with heating rate, because
each term does. Reproducing a downward shift requires two reactions
**competing for the same reactant**, so that the faster,
higher-activation-energy path steals reactant from the slower one as the
rate climbs.

It matters for entry rather than for laboratory work. [TH2020] notes
that across the MSL heat shield "values as high as 60000 K/min and as low
as 60 K/min can be found", while "most of flight heating rates are
outside the realm of legacy TGA measurements used for calibration, rarely
exceeding tens of K/min". A parallel model calibrated at 10 K/min is
being extrapolated three or four decades.

:func:`competitive_mass_fraction` integrates [TH2020]'s scheme, which is outside
FIAT Eq. (8)'s model form and is provided as an independent integrator.
:func:`parallel_pica_resin` implements [TH2019]'s six-reaction parallel
set, which *is* in Eq. (8)'s form and drops straight into the solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.integrate
from numpy.typing import ArrayLike, NDArray

from aether.thermal.material import GAS_CONSTANT, ArrheniusComponent

__all__ = [
    "COMPETITIVE_PICA_BAYESIAN",
    "COMPETITIVE_PICA_DETERMINISTIC",
    "PARALLEL_PICA_RESIN",
    "PICA_CHAR_ELEMENTS",
    "PICA_PYROLYSIS_ELEMENTS",
    "PICA_PYROLYSIS_ELEMENTS_FIATV3",
    "PICA_PYROLYSIS_ELEMENTS_MEASURED",
    "PICA_PYROLYSIS_ELEMENTS_RANGE",
    "PICA_RESIN_ELEMENTS",
    "PICA_RESIN_ELEMENTS_CROSSLINKED",
    "CompetitivePica",
    "ParallelReaction",
    "advancement_to_fiat_rate",
    "char_required_by_mass_balance",
    "char_yield_from_gas_composition",
    "competitive_mass_fraction",
    "parallel_pica_resin",
    "resin_carbon_mass_fraction",
]

_FloatArray = NDArray[np.float64]


# --------------------------------------------------------------------------
# [TH2020] competitive scheme
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CompetitivePica:
    """[TH2020] competitive mechanism for PICA pyrolysis.

    The reaction network of its Fig. 4 and Eq. (22):

    .. code-block:: text

        rho_1  --k11-->  rho_2*  --k21-->  (1-g5) rho_4  +  g5 rho_5^gas
          |
          +----k12-->    rho_3*  --k31-->  (1-g7) rho_6  +  g7 rho_7^gas

    ``k11`` is the slow, low-activation-energy path that dominates at low
    heating rate; ``k12`` is the fast, high-activation-energy path that
    takes over at high heating rate and *starves* the first branch. That
    competition for the shared reactant ``rho_1`` is the whole mechanism,
    and it is what produces the downward peak shift.

    Attributes are :math:`\\log_{10} A` (s⁻¹) and :math:`E` (J/mol), as
    tabulated, plus the two independent gas mass coefficients. Mass
    conservation fixes the solid coefficients: [TH2020] states
    :math:`\\gamma_{i,j,l+1} = 1 - \\gamma_{i,j,l}`.
    """

    log10_a11: float
    e11: float
    log10_a12: float
    e12: float
    log10_a21: float
    e21: float
    log10_a31: float
    e31: float
    gamma_gas_2: float
    """:math:`\\gamma_{2,1,5}`, gas fraction of the low-rate branch."""
    gamma_gas_3: float
    """:math:`\\gamma_{3,1,7}`, gas fraction of the high-rate branch."""
    provenance: str = ""

    def __post_init__(self) -> None:
        for name in ("gamma_gas_2", "gamma_gas_3"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1), got {value}")
        if self.e11 >= self.e12:
            raise ValueError(
                "the competitive mechanism requires E11 < E12: the slow branch "
                "must start earlier for the fast branch to take over as the "
                "heating rate rises. Got "
                f"E11={self.e11:.6g}, E12={self.e12:.6g}"
            )

    def rates(self, temperature: float) -> tuple[float, float, float, float]:
        """``(k11, k12, k21, k31)`` at ``temperature`` (1/s)."""
        rt = GAS_CONSTANT * float(temperature)
        k = [
            10.0**log_a * float(np.exp(-e / rt))
            for log_a, e in (
                (self.log10_a11, self.e11),
                (self.log10_a12, self.e12),
                (self.log10_a21, self.e21),
                (self.log10_a31, self.e31),
            )
        ]
        return k[0], k[1], k[2], k[3]

    def char_yield_limits(self) -> tuple[float, float]:
        """Char yield in the slow-branch and fast-branch limits.

        A consequence of the mechanism worth noticing: because the two
        branches have different gas coefficients, **char yield is a
        function of heating rate**. FIAT Eq. (8) makes it a constant of
        the material.
        """
        return 1.0 - self.gamma_gas_2, 1.0 - self.gamma_gas_3


#: [TH2020] Table 1 — deterministic optimisation. E in J/mol.
COMPETITIVE_PICA_DETERMINISTIC = CompetitivePica(
    log10_a11=2.019,
    e11=32618.482,
    log10_a12=14.292,
    e12=143273.910,
    log10_a21=0.442,
    e21=51783.980,
    log10_a31=0.993,
    e31=31087.851,
    gamma_gas_2=0.163,
    gamma_gas_3=0.244,
    provenance="Torres-Herrador et al. 2020, Table 1 (deterministic optimisation)",
)

#: [TH2020] Table 2 — posterior means from Bayesian inference. E in J/mol.
COMPETITIVE_PICA_BAYESIAN = CompetitivePica(
    log10_a11=2.4768,
    e11=26811.37,
    log10_a12=23.4935,
    e12=183938.42,
    log10_a21=0.2219,
    e21=48796.41,
    log10_a31=1.1969,
    e31=33566.43,
    gamma_gas_2=0.1648,
    gamma_gas_3=0.3190,
    provenance="Torres-Herrador et al. 2020, Table 2 (Bayesian posterior mean)",
)

#: [TH2020] Table 2 — posterior standard deviations, same ordering.
#:
#: Kept because the paper's own conclusion is that two of these are badly
#: identified: the coefficient of variation on :math:`A_{2,1}` is 0.56,
#: and the paper attributes the high correlation between :math:`A` and
#: :math:`E` for reactions (2,1) and (3,1) to "the kinetic compensation
#: effect". Propagating the means alone would hide that.
COMPETITIVE_PICA_UNCERTAINTY = {
    "log10_a11": 0.3027,
    "e11": 893.61,
    "log10_a12": 1.1618,
    "e12": 2369.64,
    "log10_a21": 0.1238,
    "e21": 1723.16,
    "log10_a31": 0.0821,
    "e31": 976.07,
    "gamma_gas_2": 0.0038,
    "gamma_gas_3": 0.0703,
}


def competitive_mass_fraction(
    model: CompetitivePica,
    temperatures: ArrayLike,
    heating_rate: float,
    initial_temperature: float = 300.0,
) -> _FloatArray:
    """Residual solid mass fraction of a constant-rate scan, [TH2020] Eq. (22).

    Integrates the seven-species network along
    :math:`T = T_0 + \\beta\\theta` and returns the solid fraction
    :math:`\\rho_1 + \\rho_2^* + \\rho_3^* + \\rho_4 + \\rho_6`, normalised
    to the initial reactant.
    """
    t = np.asarray(temperatures, dtype=np.float64)
    if t.ndim != 1 or t.size < 2 or np.any(np.diff(t) <= 0.0):
        raise ValueError("temperatures must be strictly increasing with >= 2 points")
    if not (np.isfinite(heating_rate) and heating_rate > 0.0):
        raise ValueError(f"heating_rate must be finite and > 0, got {heating_rate}")
    if t[0] < initial_temperature:
        raise ValueError("temperatures must start at or above initial_temperature")

    g5, g7 = model.gamma_gas_2, model.gamma_gas_3

    def rhs(temp: float, y: _FloatArray) -> _FloatArray:
        k11, k12, k21, k31 = model.rates(temp)
        r1, r2, r3 = y[0], y[1], y[2]
        return (
            np.array(
                [
                    -(k11 + k12) * r1,
                    k11 * r1 - k21 * r2,
                    k12 * r1 - k31 * r3,
                    (1.0 - g5) * k21 * r2,
                    g5 * k21 * r2,
                    (1.0 - g7) * k31 * r3,
                    g7 * k31 * r3,
                ]
            )
            / heating_rate
        )

    solution = scipy.integrate.solve_ivp(
        rhs,
        (initial_temperature, float(t[-1])),
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        t_eval=t,
        method="LSODA",
        rtol=1e-10,
        atol=1e-13,
    )
    if not solution.success:  # pragma: no cover - integrator failure
        raise RuntimeError(f"competitive integration failed: {solution.message}")
    solid = solution.y[[0, 1, 2, 3, 5], :].sum(axis=0)
    return np.asarray(solid)


# --------------------------------------------------------------------------
# [TH2019] parallel scheme — FIAT Eq. (8) compatible
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ParallelReaction:
    """One row of [TH2019] Table 2.

    Attributes
    ----------
    density_loss_fraction:
        :math:`F_{i,j}`, "the fraction of density that is lost when
        reaction :math:`R_{i,j}` reaches completion" ([TH2019] Eq. 5).
    log10_a:
        :math:`\\log_{10} A` (s⁻¹).
    activation_energy_kj:
        :math:`E` (kJ/mol), **as tabulated** — note the unit.
    order:
        :math:`n`, the reaction order.
    """

    density_loss_fraction: float
    log10_a: float
    activation_energy_kj: float
    order: float


#: [TH2019] Table 2 — species-based model, six parallel reactions.
#:
#: Calibrated against the high-heating-rate mass-spectrometry data of
#: Bessire and Minton. The activation energies are in **kJ/mol** in the
#: source and are converted on use; the reaction orders are unusually
#: high (4.2 to 10), which is characteristic of a lumped multi-species
#: fit and is faithful to the table.
PARALLEL_PICA_RESIN = (
    ParallelReaction(0.060, 6.59, 77.6, 5.65),
    ParallelReaction(0.009, 6.96, 61.3, 9.96),
    ParallelReaction(0.203, 6.71, 95.1, 4.23),
    ParallelReaction(0.187, 6.67, 103.0, 4.38),
    ParallelReaction(0.026, 6.58, 113.9, 6.68),
    ParallelReaction(0.059, 6.35, 175.2, 8.85),
)


def advancement_to_fiat_rate(
    log10_a: float, order: float, virgin_density: float, char_density: float
) -> float:
    """Convert an advancement-form pre-exponential to FIAT's normalisation.

    The two literatures normalise the rate differently, and the
    difference is a pure power of the decomposable fraction — silent,
    dimensionally invisible, and worth several orders of magnitude at the
    reaction orders [TH2019] reports.

    [TH2019] and the biomass literature write the rate in terms of
    reaction advancement :math:`\\chi`:

    .. math::

        \\frac{d\\chi}{dt} = A_P e^{-E/RT}(1-\\chi)^n, \\qquad
        \\rho = \\rho_v - \\chi(\\rho_v - \\rho_r).

    FIAT Eq. (8) writes it against the *virgin* density:

    .. math::

        \\frac{d\\rho}{dt} = -A_F e^{-E/RT}\\rho_v
        \\left(\\frac{\\rho - \\rho_r}{\\rho_v}\\right)^{n}.

    Equating the two gives

    .. math::

        A_F = A_P\\left(\\frac{\\rho_v}{\\rho_v - \\rho_r}\\right)^{n-1}.

    At :math:`n = 1` they coincide, which is why the trap only bites for
    the high-order fits.
    """
    if not char_density < virgin_density:
        raise ValueError("need char_density < virgin_density")
    if not virgin_density > 0.0:
        raise ValueError("virgin_density must be > 0")
    ratio = virgin_density / (virgin_density - char_density)
    return float(10.0**log10_a * ratio ** (order - 1.0))


def parallel_pica_resin(
    resin_density: float,
    reactions: tuple[ParallelReaction, ...] = PARALLEL_PICA_RESIN,
) -> list[ArrheniusComponent]:
    """[TH2019] Table 2 as FIAT Eq. (8) components.

    Each tabulated reaction becomes one
    :class:`~aether.thermal.material.ArrheniusComponent` carrying
    :math:`F_{i,j}` of the resin mass, with its char density set so the
    component loses exactly that fraction, and its pre-exponential
    converted through :func:`advancement_to_fiat_rate`.

    Parameters
    ----------
    resin_density:
        Mass of decomposing resin per unit volume of the *resin phase*
        (kg/m³). [TH2019]'s :math:`F` values sum to 0.544, so this model
        describes the resin, not the composite: applied to PICA's
        published 94 kg/m³ of phenolic in 274 kg/m³ of material it
        implies a composite mass loss of
        :math:`0.544 \\times 94/274 = 18.7\\%`, against the
        :math:`(274-227)/274 = 17.2\\%` implied by the published virgin
        and char bulk densities. Those agree to about a percent and a
        half, which is a genuine cross-check between two unrelated
        sources — but the identification of :math:`F` with a
        resin-normalised fraction is an inference from Eq. (5), not a
        statement the paper makes in words.
    """
    if not (np.isfinite(resin_density) and resin_density > 0.0):
        raise ValueError(f"resin_density must be finite and > 0, got {resin_density}")
    total = sum(r.density_loss_fraction for r in reactions)
    if not 0.0 < total < 1.0:
        raise ValueError(f"density loss fractions must sum into (0, 1), got {total}")

    components: list[ArrheniusComponent] = []
    for r in reactions:
        # This component holds the share of resin mass its own F represents,
        # and loses all of it; splitting the resin by F rather than giving
        # every component the whole resin is what makes the F values mean
        # what Eq. (5) says they mean.
        share = r.density_loss_fraction / total
        virgin = resin_density * share
        # Each reaction runs to completion, so the residual is what the
        # reaction does not volatilise; with the F-weighted split above that
        # is the non-decomposing remainder of this component's share.
        char = virgin * (1.0 - total)
        components.append(
            ArrheniusComponent(
                pre_exponential=advancement_to_fiat_rate(r.log10_a, r.order, virgin, char),
                activation_energy=r.activation_energy_kj * 1.0e3,
                reaction_order=r.order,
                virgin_density=virgin,
                char_density=char,
            )
        )
    return components


# --------------------------------------------------------------------------
# Elemental bookkeeping  [RAB2014]
# --------------------------------------------------------------------------
#
# Rabinovitch, Marx & Blanquart, "Pyrolysis Gas Composition for a Phenolic
# Impregnated Carbon Ablator Heatshield," AIAA 2014-2246 (in ``reference/``)
# compiles the solid-side numbers that close the elemental books on PICA.
# They are what let a candidate pyrolysis-gas composition be *checked*
# rather than merely quoted.

#: Elemental composition of cured phenolic resin, mole fractions.
#:
#: [RAB2014] §II.B.1: the repeating unit of the idealised linear polymer is
#: C₇H₆O. A fully cross-linked molecule gives 0.517:0.414:0.069 instead, and
#: the paper offers the pair as "limiting idealized cases". Sykes' elemental
#: analysis of a real cured novolac lands at 0.488:0.434:0.078 once trapped
#: ammonia is removed, i.e. between them.
PICA_RESIN_ELEMENTS = {"C": 0.500, "H": 0.429, "O": 0.071}

#: Cross-linked limit of the resin composition, the other bracket.
PICA_RESIN_ELEMENTS_CROSSLINKED = {"C": 0.517, "H": 0.414, "O": 0.069}

#: Elemental composition of char, mole fractions — Sykes at 850 °C.
#:
#: [RAB2014] §II.C, from 92.6% C / 0.9% H / 6.5% O by mass. Tran's XPS of an
#: arc-jet-tested PICA char instead gives 98% C with 1.7% O and no reported
#: hydrogen; [RAB2014] attributes the gap to Tran's sample being fully
#: pyrolysed and calls the discrepancy unresolved. Both chars are *oxygen*
#: rich relative to hydrogen compared with the resin, which is the fact
#: :func:`char_required_by_mass_balance` turns into a falsifiable test.
PICA_CHAR_ELEMENTS = {"C": 0.856, "H": 0.099, "O": 0.045}

#: Elemental composition of PICA pyrolysis gas, mole fractions.
#:
#: **This is FIAT's own value**, and using it is the point. [RAB2014] §III.A.2
#: reports that Milos & Chen's FIATv3 PICA deck injects pyrolysis gas at
#: C:H:N:O:Si = 0.18:0.68:0.014:0.12:0.006 by mole, essentially constant over
#: a surface-temperature excursion from 200 °C to 3000 °C. Dropping the trace
#: N and Si — which this solver's B′ tables do not carry — and renormalising
#: gives the numbers below. It corresponds to a char yield near 65%.
#:
#: **It replaces a measurement-derived composition that could not be right.**
#: Earlier revisions carried C 0.1745, H 0.6785, O 0.1470, obtained two ways:
#: weighting [TH2019] Table 2's species yields by their density-loss
#: fractions, and integrating Bessire & Minton Fig. 7's mass yields over
#: temperature at four heating rates. The two agreed closely, which is why
#: they were trusted. They agree because they share a bias, not because they
#: are right: both are gas-phase speciation measurements, and both are
#: subject to the same under-recovery of H₂ and over-recovery of desorbed
#: water. :func:`char_required_by_mass_balance` shows the consequence — that
#: composition demands a char with H/O ≈ 13, hydrogen-rich, and at any char
#: yield above 50% it demands *negative* oxygen in the char. No char can do
#: that. See :data:`PICA_PYROLYSIS_ELEMENTS_MEASURED`.
PICA_PYROLYSIS_ELEMENTS = {"C": 0.183673, "H": 0.693878, "O": 0.122449}

#: The unrenormalised FIATv3 composition, trace species included.
#:
#: [RAB2014] attributes the nitrogen to resin-synthesis impurities and notes
#: that the origin of the silicon is unclear even to its authors.
PICA_PYROLYSIS_ELEMENTS_FIATV3 = {
    "C": 0.18,
    "H": 0.68,
    "N": 0.014,
    "O": 0.12,
    "Si": 0.006,
}

#: The superseded speciation-derived composition, kept for the audit trail.
#:
#: Retained so that :func:`char_required_by_mass_balance` can be exercised on
#: a composition known to fail, and so the earlier arc-jet results remain
#: reproducible. Do not use it to build a B′ table.
PICA_PYROLYSIS_ELEMENTS_MEASURED = {"C": 0.1745, "H": 0.6785, "O": 0.1470}

#: Range spanned by the four Bessire & Minton heating rates, for reference.
PICA_PYROLYSIS_ELEMENTS_RANGE = {
    "C": (0.1668, 0.1870),
    "H": (0.6613, 0.6782),
    "O": (0.1457, 0.1593),
}


_MOLAR_MASS = {"C": 12.011, "H": 1.008, "O": 15.999}


def resin_carbon_mass_fraction(resin: dict[str, float] | None = None) -> float:
    """Carbon mass fraction of the resin, [SCO2017] Eq. (13).

    Scoggins, Leroy, Bellas-Chatzigeorgis, Dias & Magin, "Thermodynamic
    properties of carbon-phenolic gas mixtures," *Aerospace Science and
    Technology* **66** (2017) 177-192, notes that this is also the *maximum*
    attainable char yield: a pure-carbon char cannot carry away more carbon
    than the resin contains. It evaluates to 79.3% for the linear polymer and
    80.3% for the fully cross-linked one, against the 79.2% and 80.4% the
    paper reports — an independent confirmation of
    :data:`PICA_RESIN_ELEMENTS`.
    """
    base = PICA_RESIN_ELEMENTS if resin is None else resin
    masses = {e: base[e] * _MOLAR_MASS[e] for e in base}
    return masses["C"] / sum(masses.values())


def char_yield_from_gas_composition(
    gas: dict[str, float],
    resin: dict[str, float] | None = None,
) -> tuple[float, float]:
    """Char yield implied by a pyrolysis-gas composition, two ways.

    [SCO2017] Eq. (12) treats pyrolysis as one step producing a pure-carbon
    char, ``CaHbOc -> C(alpha)HbOc + C(a-alpha)``, so hydrogen and oxygen pass
    through **unchanged** and Eq. (13) gives the char yield

    .. math::

        \\chi = y^{\\mathrm{resin}}_C \\left(1 - \\frac{\\alpha}{a}\\right).

    Because H and O are untouched, :math:`\\alpha` can be recovered from the
    gas either through hydrogen or through oxygen, and a self-consistent
    composition gives the same answer both ways. The **spread between the two
    is the diagnostic**: it is zero for a composition that conserves elements
    and grows with whichever of H or O was mismeasured.

    Returns
    -------
    tuple[float, float]
        Char yield inferred via hydrogen and via oxygen, as mass fractions.
    """
    base = PICA_RESIN_ELEMENTS if resin is None else resin
    total = sum(gas.values())
    if not (np.isfinite(total) and total > 0.0):
        raise ValueError("gas composition must have a finite positive sum")
    fractions = {e: gas[e] / total for e in gas}
    if fractions.get("H", 0.0) <= 0.0 or fractions.get("O", 0.0) <= 0.0:
        raise ValueError("gas composition needs positive H and O to invert")
    carbon_mass = resin_carbon_mass_fraction(base)
    yields = []
    for element in ("H", "O"):
        alpha = fractions["C"] / fractions[element] * base[element]
        yields.append(carbon_mass * (1.0 - alpha / base["C"]))
    return yields[0], yields[1]


def char_required_by_mass_balance(
    gas: dict[str, float],
    gas_mole_fraction: float,
    resin: dict[str, float] | None = None,
) -> dict[str, float]:
    """Char composition implied by a pyrolysis-gas composition.

    One mole of resin atoms splits into ``gas_mole_fraction`` moles of gas
    atoms and the remainder as char, so conservation of each element fixes
    the char outright:

    .. math::

        x_{i,\\mathrm{char}} =
            \\frac{x_{i,\\mathrm{resin}} - g\\,x_{i,\\mathrm{gas}}}{1 - g}

    This is a *test*, not a model. A gas composition consistent with the
    resin yields a char that is carbon-rich and has every mole fraction
    non-negative. One that is not yields a char that is impossible, and the
    sign of the violation says which element was mismeasured.

    Parameters
    ----------
    gas:
        Pyrolysis-gas mole fractions, keyed by element symbol.
    gas_mole_fraction:
        Moles of gas atoms per mole of resin atoms, in ``(0, 1)``.
    resin:
        Resin mole fractions; defaults to :data:`PICA_RESIN_ELEMENTS`.

    Returns
    -------
    dict[str, float]
        Char mole fractions, which are **not** clipped and may be negative
        when the input gas composition violates the balance.
    """
    g = float(gas_mole_fraction)
    if not (np.isfinite(g) and 0.0 < g < 1.0):
        raise ValueError(f"gas_mole_fraction must be finite and in (0, 1), got {g}")
    base = PICA_RESIN_ELEMENTS if resin is None else resin
    total = sum(gas.values())
    if not (np.isfinite(total) and total > 0.0):
        raise ValueError("gas composition must have a finite positive sum")
    return {
        element: (base.get(element, 0.0) - g * gas[element] / total) / (1.0 - g) for element in gas
    }
