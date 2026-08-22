"""A PICA-like charring ablator for FIAT-formulation runs.

The synthetic material in :func:`aether.thermal.material.demo_material`
exists to exercise the spectral solver's manufactured solutions, and its
kinetics are deliberately fast so that decomposition is well resolved in
a short run. That makes it useless for realistic ablation: it releases
225 kg/m³ of pyrolysis gas with a millisecond time constant, driving
:math:`B'_g` into the tens, an order of magnitude above anything a real
ablator produces.

This module supplies a material of realistic *magnitude*, so that
recession, gas flux and :math:`B'` land where a phenolic-impregnated
carbon ablator actually puts them.

Provenance, stated per property
-------------------------------

**Published, from the MEDLI2 material-response paper** (Monk et al.,
"MEDLI2 Material Response Model Development and Validation", in
``reference/``), Heritage PICA model row:

===========================================  ==========
virgin density                                274 kg/m³
room-temperature virgin conductivity          0.174 W/(m K)
room-temperature char conductivity            0.224 W/(m K)
===========================================  ==========

**Reconstructed** from the published composition of PICA — a FiberForm
carbon preform impregnated with phenolic resin — by splitting the
274 kg/m³ into a non-decomposing carbon skeleton and a resin that chars
to roughly half its mass. The split reproduces the published virgin
density exactly and puts the char density at 227 kg/m³.

**Representative, not published**: the Arrhenius triplets, the
temperature slopes on conductivity and specific heat, and the pyrolysis
gas enthalpy.

The kinetics deserve a specific warning. No published Arrhenius triplets
for PICA appear anywhere in ``reference/`` — the MEDLI2 paper
characterises conductivity, specific heat and density and says nothing
about decomposition rates, and the MSL reconstruction paper notes that
"no kinetic rate-limited recession model for PICA exists that is
sufficiently validated for use in TPS design". Rather than invent three
numbers, the triplets here are pinned to *stated, checkable* targets and
those targets are asserted in the test suite: at a 20 K/min scan the
composite loses 2% of its decomposable mass by **557 K**, peaks in rate
at **799 K**, and leaves a char yield of **227/274 = 0.8285** — the last
being a consequence of the published bulk densities rather than a free
parameter. :mod:`aether.fiat.kinetics` provides the forward TGA
model and the fitter to replace all of this the moment a real scan is
available.

.. warning::

   This is a *PICA-like* material, not PICA. Results computed with it
   describe the solver, not the material, and must not be reported as
   PICA predictions. Closing a recession comparison against a published
   PICA case requires that case's own property set — see
   ``docs/FIAT-reference-data.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aether.thermal.material import (
    ArrheniusComponent,
    CharringMaterial,
    LinearBlendProperty,
)

__all__ = [
    "HERITAGE_PICA_CONDUCTIVITY",
    "MEDLI2_PICA_CONDUCTIVITY",
    "MEDLI2_PICA_VIRGIN_DENSITY",
    "ONE_ATMOSPHERE",
    "PICA_CHAR_CONDUCTIVITY_RT",
    "PICA_LIKE_CHAR_DENSITY",
    "PICA_VIRGIN_CONDUCTIVITY_RT",
    "PICA_VIRGIN_DENSITY",
    "MultiComponentMaterial",
    "PressureConductivity",
    "TabulatedConductivity",
    "pica_fiat_material",
    "pica_like_material",
    "read_tran_conductivity",
    "read_tran_specific_heat",
    "structural_material",
]

_FloatArray = NDArray[np.float64]

#: Standard atmosphere, Pa — the upper anchor of the published table.
ONE_ATMOSPHERE = 101325.0
#: MEDLI2 flight-lot PICA virgin bulk density, kg/m³ — published.
MEDLI2_PICA_VIRGIN_DENSITY = 292.0


@dataclass(frozen=True)
class PressureConductivity:
    """Conductivity of a porous ablator as a function of pressure.

    PICA is a carbon preform with most of its volume as gas-filled pore
    space, so its conductivity depends on the pressure of the gas in
    those pores as well as on temperature and char fraction. The MEDLI2
    material-response paper tabulates that dependence directly (Table 3,
    "PICA Room Temperature Properties"), giving virgin and char
    conductivity at **both** 1 atm and 0.001 atm.

    Interpolation is linear in :math:`\\log p`, which is the natural
    variable for a Knudsen-regime transition and the only defensible
    choice given two anchors three decades apart. Outside the anchors the
    value is held constant rather than extrapolated: a two-point fit says
    nothing about the behaviour beyond its own endpoints, and entry
    trajectories routinely go below 0.001 atm.

    Attributes
    ----------
    low_pressure, high_pressure:
        The two tabulated pressures (Pa), ``low < high``.
    low, high:
        :class:`~aether.thermal.material.LinearBlendProperty` at each.
    """

    low_pressure: float
    high_pressure: float
    low: LinearBlendProperty
    high: LinearBlendProperty

    def __post_init__(self) -> None:
        if not (0.0 < self.low_pressure < self.high_pressure):
            raise ValueError(
                f"need 0 < low_pressure < high_pressure, got "
                f"{self.low_pressure} / {self.high_pressure}"
            )

    def value(
        self, temperature: ArrayLike, char_fraction: ArrayLike, pressure: float
    ) -> _FloatArray:
        """Conductivity (W/(m K)) at the given state and pressure."""
        if not (np.isfinite(pressure) and pressure > 0.0):
            raise ValueError(f"pressure must be finite and > 0, got {pressure}")
        span = np.log(self.high_pressure / self.low_pressure)
        w = float(np.clip(np.log(pressure / self.low_pressure) / span, 0.0, 1.0))
        return np.asarray(
            (1.0 - w) * self.low.value(temperature, char_fraction)
            + w * self.high.value(temperature, char_fraction)
        )

    def d_temperature(
        self, temperature: ArrayLike, char_fraction: ArrayLike, pressure: float
    ) -> _FloatArray:
        span = np.log(self.high_pressure / self.low_pressure)
        w = float(np.clip(np.log(pressure / self.low_pressure) / span, 0.0, 1.0))
        return np.asarray(
            (1.0 - w) * self.low.d_temperature(temperature, char_fraction)
            + w * self.high.d_temperature(temperature, char_fraction)
        )

    def d_char_fraction(
        self, temperature: ArrayLike, char_fraction: ArrayLike, pressure: float
    ) -> _FloatArray:
        span = np.log(self.high_pressure / self.low_pressure)
        w = float(np.clip(np.log(pressure / self.low_pressure) / span, 0.0, 1.0))
        return np.asarray(
            (1.0 - w) * self.low.d_char_fraction(temperature, char_fraction)
            + w * self.high.d_char_fraction(temperature, char_fraction)
        )


def _rt_conductivity(
    virgin_1atm: float,
    char_1atm: float,
    virgin_low: float,
    char_low: float,
    virgin_slope: float = 1.5e-4,
    char_slope: float = 4.5e-4,
) -> PressureConductivity:
    """Build a pressure-dependent conductivity from one row of Table 3.

    The table gives room-temperature values only, so the temperature
    slopes are supplied separately and are *representative*: the
    published data pins the 300 K intercepts at both pressures and
    nothing else.
    """
    return PressureConductivity(
        low_pressure=0.001 * ONE_ATMOSPHERE,
        high_pressure=ONE_ATMOSPHERE,
        low=LinearBlendProperty(
            virgin_low - 300.0 * virgin_slope,
            virgin_slope,
            char_low - 300.0 * char_slope,
            char_slope,
        ),
        high=LinearBlendProperty(
            virgin_1atm - 300.0 * virgin_slope,
            virgin_slope,
            char_1atm - 300.0 * char_slope,
            char_slope,
        ),
    )


#: Heritage PICA model, MEDLI2 paper Table 3 — all four values published.
#:
#: Note the direction. This model has virgin conductivity **rising** from
#: 0.174 to 0.520 W/(m K) as pressure falls from 1 atm to 0.001 atm — a
#: factor of three *increase* with decreasing pore-gas pressure, which is
#: the opposite of what a porous medium normally does. The MEDLI2
#: re-measurement in the same table has it falling by 25%, the expected
#: direction. The two disagree by a **factor of 4.1** at 0.001 atm, which
#: is the regime that governs entry. Both are provided; neither is
#: presented as correct.
HERITAGE_PICA_CONDUCTIVITY = _rt_conductivity(0.174, 0.224, 0.520, 0.202)
#: MEDLI2 re-measured PICA model, same table — all four values published.
MEDLI2_PICA_CONDUCTIVITY = _rt_conductivity(0.169, 0.169, 0.127, 0.143)

#: Heritage PICA virgin bulk density, kg/m³ — MEDLI2 paper, published.
PICA_VIRGIN_DENSITY = 274.0
#: Heritage PICA room-temperature virgin conductivity, W/(m K) — published.
PICA_VIRGIN_CONDUCTIVITY_RT = 0.174
#: Heritage PICA room-temperature char conductivity, W/(m K) — published.
PICA_CHAR_CONDUCTIVITY_RT = 0.224
#: Char density of the reconstructed composition, kg/m³ — *not* published.
PICA_LIKE_CHAR_DENSITY = 227.0

# Composition reconstruction. FIAT Eq. (7) is
# rho = Gamma (rho_A + rho_B) + (1 - Gamma) rho_C, so with Gamma = 1/2 the
# resin contributes (rho_A + rho_B)/2 and the carbon skeleton rho_C/2. Taking
# the FiberForm preform at 180 kg/m3 and the phenolic at 94 kg/m3 reproduces
# the published 274 exactly; the resin charring to half its mass puts the char
# density at 180 + 47 = 227 kg/m3.
_GAMMA = 0.5
_RESIN_TOTAL = 2.0 * 94.0
_CARBON_TOTAL = 2.0 * 180.0


def pica_like_material(heritage: bool = False) -> CharringMaterial:
    """A charring ablator with PICA's published bulk properties.

    Parameters
    ----------
    heritage:
        Use the Heritage PICA conductivity row of Table 3 instead of the
        MEDLI2 re-measured row. They differ by a factor of four at
        0.001 atm — see :data:`HERITAGE_PICA_CONDUCTIVITY`.

    Two resin components decomposing over overlapping temperature bands,
    plus a non-decomposing carbon skeleton — the three-component model of
    FIAT Eq. (7), used as intended rather than as three arbitrary
    reactions.
    """
    return CharringMaterial(
        # Low-temperature resin fraction: the lighter volatiles, off by ~700 K.
        resin_a=ArrheniusComponent(
            pre_exponential=1.4e4,
            activation_energy=7.1e4,
            reaction_order=3.0,
            virgin_density=0.30 * _RESIN_TOTAL,
            char_density=0.15 * _RESIN_TOTAL,
        ),
        # High-temperature resin fraction: the phenolic backbone, ~1100 K.
        resin_b=ArrheniusComponent(
            pre_exponential=4.5e9,
            activation_energy=1.70e5,
            reaction_order=3.0,
            virgin_density=0.70 * _RESIN_TOTAL,
            char_density=0.35 * _RESIN_TOTAL,
        ),
        # Carbon preform: present in Eq. (7) but inert. A zero
        # pre-exponential is FIAT's own way of writing a non-decomposing
        # component; the char density is held a hair below virgin only
        # because the material model requires a strict inequality.
        filler=ArrheniusComponent(
            pre_exponential=0.0,
            activation_energy=1.0e5,
            reaction_order=1.0,
            virgin_density=_CARBON_TOTAL,
            char_density=_CARBON_TOTAL * (1.0 - 1e-9),
        ),
        resin_fraction=_GAMMA,
        # The 1 atm column of Table 3. A ply given
        # `conductivity=MEDLI2_PICA_CONDUCTIVITY` overrides this with the
        # full pressure dependence; this field is the sea-level fallback for
        # code paths that have no pressure to hand.
        conductivity=(
            HERITAGE_PICA_CONDUCTIVITY if heritage else MEDLI2_PICA_CONDUCTIVITY
        ).high,
        specific_heat=LinearBlendProperty(1100.0, 0.32, 1250.0, 0.30),
        gas_specific_heat=2100.0,
        gas_enthalpy_offset=-2.2e6,
        gas_enthalpy_slope=2100.0,
        solid_enthalpy_offset=-1.1e6,
        solid_enthalpy_slope=1400.0,
        emissivity_virgin=0.85,
        emissivity_char=0.90,
    )


def structural_material(
    density: float = 1600.0,
    conductivity: float = 0.5,
    specific_heat: float = 900.0,
) -> CharringMaterial:
    """A non-decomposing substructure ply (bondline, honeycomb, laminate).

    FIAT's stacks routinely end in structure that conducts and stores
    heat but neither pyrolyses nor ablates. Expressing that as a
    :class:`~aether.thermal.material.CharringMaterial` with zero
    pre-exponentials — rather than as a separate type — is how FIAT's own
    material database does it, and it keeps one code path through the
    solver.
    """
    if not density > 0.0:
        raise ValueError(f"density must be > 0, got {density}")
    if not conductivity > 0.0:
        raise ValueError(f"conductivity must be > 0, got {conductivity}")
    if not specific_heat > 0.0:
        raise ValueError(f"specific_heat must be > 0, got {specific_heat}")
    # Eq. (7) is rho = Gamma(rho_A + rho_B) + (1 - Gamma) rho_C, and the
    # material model forbids a zero virgin density, so the two resin slots
    # carry a vanishing mass and the carbon slot carries the rest. With
    # Gamma = 1/2 that reproduces `density` to a part in 10^6.
    trace = 1.0e-6 * density
    inert = ArrheniusComponent(
        pre_exponential=0.0,
        activation_energy=1.0e5,
        reaction_order=1.0,
        virgin_density=trace,
        char_density=trace * (1.0 - 1e-9),
    )
    return CharringMaterial(
        resin_a=inert,
        resin_b=inert,
        filler=ArrheniusComponent(
            pre_exponential=0.0,
            activation_energy=1.0e5,
            reaction_order=1.0,
            virgin_density=2.0 * (density - trace),
            char_density=2.0 * (density - trace) * (1.0 - 1e-9),
        ),
        resin_fraction=0.5,
        conductivity=LinearBlendProperty(conductivity, 0.0, conductivity, 0.0),
        specific_heat=LinearBlendProperty(specific_heat, 0.0, specific_heat, 0.0),
        gas_specific_heat=1000.0,
        gas_enthalpy_offset=0.0,
        gas_enthalpy_slope=1000.0,
        solid_enthalpy_offset=0.0,
        solid_enthalpy_slope=specific_heat,
        emissivity_virgin=0.85,
        emissivity_char=0.85,
    )


#: Btu·in/(hr·ft²·°F) to W/(m K) — the unit of Tran et al. Fig. 9.
_BTU_IN_PER_HR_FT2_F = 0.1442279
#: Btu/(lbm·°F) to J/(kg K) — the unit of Tran et al. Fig. 7.
_BTU_PER_LBM_F = 4186.8


def _fahrenheit(x: ArrayLike) -> _FloatArray:
    return np.asarray((np.asarray(x, dtype=np.float64) - 32.0) * 5.0 / 9.0 + 273.15)


@dataclass(frozen=True)
class TabulatedConductivity:
    """Measured conductivity on a temperature/pressure grid.

    Tran et al., NASA TM-110440 (1997), Fig. 9, "Thermal conductivity of
    PICA-15", digitised at 0.001, 0.01 and 0.05 atm and spanning roughly
    440 to 2950 K. It replaces the affine-in-temperature blend used
    elsewhere, whose slopes above room temperature were invented: the
    published MEDLI2 data pins only the 300 K intercepts.

    Two honest limitations.

    **Virgin or char is not stated.** The source labels the figure
    "PICA-15" and says nothing more. It cannot be virgin throughout —
    virgin PICA does not survive to 2950 K — so the curve is best read as
    the conductivity of the material *as heated*, virgin at the cold end
    and char at the hot end, with the transition folded in. It is
    therefore applied to both phases here rather than assigned to one, and
    :meth:`d_char_fraction` returns zero. That is a real loss of the
    virgin/char distinction, traded for measured magnitudes over
    2500 K of range.

    **It corroborates MEDLI2 against Heritage.** Conductivity rises with
    pressure at every temperature in this data — 0.248 to 0.296 W/(m K)
    at 450 K going from 0.001 to 0.05 atm. The Heritage PICA model has
    virgin conductivity *rising threefold as pressure falls*, which is
    backwards for a porous medium. Two independent measurements 24 years
    apart now agree against it.
    """

    pressures: _FloatArray
    temperatures: _FloatArray
    values: _FloatArray
    """W/(m K), shape ``(n_pressures, n_temperatures)``."""

    def _interpolate(self, temperature: ArrayLike, pressure: float) -> _FloatArray:
        if not (np.isfinite(pressure) and pressure > 0.0):
            raise ValueError(f"pressure must be finite and > 0, got {pressure}")
        t = np.asarray(temperature, dtype=np.float64)
        # Linear in log-pressure between planes, held outside them: three
        # anchors over two decades say nothing about the fourth.
        lp = np.log(np.clip(pressure, self.pressures[0], self.pressures[-1]))
        planes = np.log(self.pressures)
        j = int(np.clip(np.searchsorted(planes, lp) - 1, 0, planes.size - 2))
        w = (lp - planes[j]) / (planes[j + 1] - planes[j])
        lo = np.interp(t, self.temperatures, self.values[j])
        hi = np.interp(t, self.temperatures, self.values[j + 1])
        return np.asarray((1.0 - w) * lo + w * hi)

    def value(
        self, temperature: ArrayLike, char_fraction: ArrayLike, pressure: float
    ) -> _FloatArray:
        return np.asarray(
            self._interpolate(temperature, pressure) + 0.0 * np.asarray(char_fraction)
        )

    def d_temperature(
        self, temperature: ArrayLike, char_fraction: ArrayLike, pressure: float
    ) -> _FloatArray:
        t = np.asarray(temperature, dtype=np.float64)
        h = 1.0
        up = self._interpolate(t + h, pressure)
        dn = self._interpolate(t - h, pressure)
        return np.asarray((up - dn) / (2.0 * h) + 0.0 * np.asarray(char_fraction))

    def d_char_fraction(
        self, temperature: ArrayLike, char_fraction: ArrayLike, pressure: float
    ) -> _FloatArray:
        return np.zeros_like(np.asarray(temperature, dtype=np.float64) * 1.0)


def read_tran_conductivity(directory: str | Path) -> TabulatedConductivity:
    """Tran Fig. 9, digitised at three pressures."""
    root = Path(directory)
    atm = (0.001, 0.01, 0.05)
    grid = np.arange(300.0, 3001.0, 25.0)
    values = np.zeros((len(atm), grid.size))
    for i, a in enumerate(atm):
        path = root / f"Tran1997-Fig9-{a:g}atm.csv"
        if not path.exists():
            raise FileNotFoundError(f"no digitised Fig. 9 curve at {path}")
        raw = np.loadtxt(path, delimiter=",")
        t = _fahrenheit(raw[:, 0])
        k = raw[:, 1] * _BTU_IN_PER_HR_FT2_F
        order = np.argsort(t)
        values[i] = np.interp(grid, t[order], k[order])
    return TabulatedConductivity(
        pressures=np.asarray(atm) * ONE_ATMOSPHERE,
        temperatures=grid,
        values=values,
    )


def read_tran_specific_heat(directory: str | Path) -> LinearBlendProperty:
    """Tran Fig. 7, "Heat capacity of PICA-15", as an affine blend.

    The measured curve is close to linear in temperature over its range,
    so a least-squares affine fit loses little and keeps the property in
    the same form the rest of the material model uses. As with the
    conductivity, no virgin/char split is available, so both phases carry
    the same fit.
    """
    path = Path(directory) / "Tran1997-Fig7.csv"
    if not path.exists():
        raise FileNotFoundError(f"no digitised Fig. 7 curve at {path}")
    raw = np.loadtxt(path, delimiter=",")
    t = _fahrenheit(raw[:, 0])
    cp = raw[:, 1] * _BTU_PER_LBM_F
    slope, intercept = np.polyfit(t, cp, 1)
    return LinearBlendProperty(
        float(intercept), float(slope), float(intercept), float(slope)
    )


@dataclass(frozen=True)
class MultiComponentMaterial:
    """FIAT Eq. (7) generalised from three solid components to :math:`N`.

    Chen & Milos write the composite density as
    :math:`\\rho = \\Gamma(\\rho_A + \\rho_B) + (1-\\Gamma)\\rho_C` — two
    resin components and one reinforcement, which is what the CMA lineage
    has always used. That is a modelling convention, not a physical
    limit, and it becomes a hard constraint the moment a published
    mechanism has a different number of reactions.

    It does here. Torres-Herrador's measured PICA set has **six** parallel
    reactions. Lumping six into two discards exactly the resolution that
    made the measurement worth having, so Eq. (7) is generalised instead:

    .. math::

        \\rho = \\sum_i w_i \\rho_i,

    with the three-component form recovered by
    :math:`w = (\\Gamma, \\Gamma, 1-\\Gamma)`.

    Attributes
    ----------
    components:
        The decomposing (and inert) solid components.
    weights:
        :math:`w_i` in the sum above, same length.
    """

    components: tuple[ArrheniusComponent, ...]
    weights: tuple[float, ...]
    conductivity: LinearBlendProperty
    specific_heat: LinearBlendProperty
    gas_specific_heat: float
    gas_enthalpy_offset: float
    gas_enthalpy_slope: float
    solid_enthalpy_offset: float
    solid_enthalpy_slope: float
    emissivity_virgin: float
    emissivity_char: float

    def __post_init__(self) -> None:
        if len(self.components) != len(self.weights):
            raise ValueError(
                f"need one weight per component, got {len(self.weights)} "
                f"weights for {len(self.components)} components"
            )
        if not self.components:
            raise ValueError("a material needs at least one component")
        if any(w <= 0.0 or not np.isfinite(w) for w in self.weights):
            raise ValueError("weights must be finite and > 0")

    @property
    def virgin_density(self) -> float:
        return float(
            sum(w * c.virgin_density for w, c in zip(self.weights, self.components, strict=True))
        )

    @property
    def char_density(self) -> float:
        return float(
            sum(w * c.char_density for w, c in zip(self.weights, self.components, strict=True))
        )


def pica_fiat_material(
    directory: str | Path | None = None,
    resin_density: float = 94.0,
    total_density: float = PICA_VIRGIN_DENSITY,
) -> MultiComponentMaterial:
    """PICA with **measured** kinetics — the full deck.

    Seven components: Torres-Herrador 2019 Table 2's six parallel
    reactions, plus the non-decomposing remainder (carbon preform and
    residual resin char).

    Each reaction carries :math:`F_i` of the resin mass and volatilises
    all of it, which is what [TH2019] Eq. (5) says :math:`F` means: "the
    fraction of density that is lost when reaction :math:`R_{i,j}`
    reaches completion". Setting each component's char density to zero
    also makes the advancement and FIAT rate normalisations coincide
    exactly — the conversion factor
    :math:`(\\rho_v/(\\rho_v-\\rho_r))^{n-1}` is unity — so the published
    pre-exponentials go in unmodified.

    The composition closes on published numbers without tuning: the six
    :math:`F` values sum to 0.544 of a 94 kg/m³ resin fraction, which is
    51.1 kg/m³ of volatiles out of 274 kg/m³ total. That leaves a char
    density of 222.9 kg/m³ against the 227 implied by the published bulk
    densities — 1.8% apart, from two unrelated sources.

    Conductivity and specific heat come from Tran Fig. 9 and Fig. 7 when
    ``directory`` is given, and fall back to the MEDLI2 one-atmosphere
    row otherwise.
    """
    from aether.fiat.pica_kinetics import (
        PARALLEL_PICA_RESIN,
        advancement_to_fiat_rate,
    )

    total_f = sum(r.density_loss_fraction for r in PARALLEL_PICA_RESIN)
    volatile = total_f * resin_density
    if volatile >= total_density:
        raise ValueError(
            f"volatile mass {volatile:.1f} kg/m³ exceeds the total "
            f"{total_density:.1f}; check resin_density"
        )

    components = []
    weights = []
    for r in PARALLEL_PICA_RESIN:
        virgin = r.density_loss_fraction * resin_density
        components.append(
            ArrheniusComponent(
                # char_density = 0 makes the two rate normalisations
                # identical, so the published log10(A) is used as printed.
                pre_exponential=advancement_to_fiat_rate(r.log10_a, r.order, virgin, 0.0),
                activation_energy=r.activation_energy_kj * 1.0e3,
                reaction_order=r.order,
                virgin_density=virgin,
                char_density=0.0,
            )
        )
        weights.append(1.0)

    inert = total_density - volatile
    components.append(
        ArrheniusComponent(
            pre_exponential=0.0,
            activation_energy=1.0e5,
            reaction_order=1.0,
            virgin_density=inert,
            char_density=inert * (1.0 - 1e-9),
        )
    )
    weights.append(1.0)

    if directory is not None:
        k = read_tran_conductivity(directory).values.mean(axis=0)
        t = read_tran_conductivity(directory).temperatures
        slope, intercept = np.polyfit(t, k, 1)
        conductivity = LinearBlendProperty(
            float(intercept), float(slope), float(intercept), float(slope)
        )
        specific_heat = read_tran_specific_heat(directory)
    else:
        conductivity = MEDLI2_PICA_CONDUCTIVITY.high
        specific_heat = LinearBlendProperty(1100.0, 0.32, 1250.0, 0.30)

    return MultiComponentMaterial(
        components=tuple(components),
        weights=tuple(weights),
        conductivity=conductivity,
        specific_heat=specific_heat,
        gas_specific_heat=2100.0,
        gas_enthalpy_offset=-2.2e6,
        gas_enthalpy_slope=2100.0,
        solid_enthalpy_offset=-1.1e6,
        solid_enthalpy_slope=1400.0,
        emissivity_virgin=0.85,
        emissivity_char=0.90,
    )
