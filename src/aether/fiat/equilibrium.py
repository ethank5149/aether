"""A multiphase equilibrium :math:`B'` solver, written from first principles.

FIAT's surface closure comes from ACE or MAT; the previous step here used
Mutation++ as an open stand-in. Both are external, and both fix the
material through a deck we do not control. This module implements the
same calculation directly, which is what makes a **PICA** table possible
rather than a TACOT one — see :mod:`aether.fiat.pica_surface`.

What a B' table actually is
---------------------------

At a steadily ablating surface, three streams enter — boundary-layer edge
gas, pyrolysis gas from below, and char from the receding solid — and one
leaves. Normalising every rate by :math:`\\rho_e u_e C_M` and writing
:math:`\\tilde y_k` for the *elemental* mass fraction of element
:math:`k`, elemental conservation is

.. math::

    (1 + B'_g + B'_c)\\,\\tilde y_{k,w}
      = \\tilde y_{k,e} + B'_g\\,\\tilde y_{k,g} + B'_c\\,\\tilde y_{k,c}.

For a given :math:`(P, T_w, B'_g)` this leaves one unknown, :math:`B'_c`,
and one condition to fix it: the surface must be **in equilibrium with
its own condensed carbon**. Solid carbon is neither accumulating nor
absent — it is exactly saturated. That is the closure ACE applies, and
it is what makes :math:`B'_c` a property of the surface rather than a
free parameter.

How the equilibrium is solved
-----------------------------

Gibbs energy minimisation by the **element-potential** method. Minimising
:math:`G = \\sum_j n_j\\left(g_j^\\circ + RT\\ln\\frac{n_j P}{n P^\\circ}\\right)`
subject to :math:`\\sum_j a_{kj} n_j = b_k` gives, at the stationary
point,

.. math::

    \\frac{n_j}{n} = \\exp\\!\\Big(\\sum_k \\lambda_k a_{kj}
      - \\frac{g_j^\\circ}{RT} - \\ln\\frac{P}{P^\\circ}\\Big),

so the whole composition collapses onto one Lagrange multiplier
:math:`\\lambda_k` per element — four here, however many species are
carried. Those are found by minimising the convex dual, not by
root-finding the stationarity conditions; see
:func:`solve_equilibrium` for why that distinction turned out to
matter.

The condensed phase enters as a *condition* rather than an unknown. A
pure condensed species has unit activity, so it is present precisely when

.. math::

    \\sum_k \\lambda_k a_{kc} = \\frac{g_c^\\circ}{RT},

and the saturation residual :math:`\\lambda_C - g_{C(gr)}^\\circ/RT` is
the scalar that :func:`solve_bprime` drives to zero in :math:`B'_c`. No
condensed amount is ever computed, which is the point: the B' condition
is about saturation, not inventory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.optimize
from numpy.typing import NDArray

__all__ = [
    "ATOMIC_MASS",
    "EquilibriumResult",
    "Nasa9Species",
    "SurfaceComposition",
    "ThermoDatabase",
    "solve_bprime",
    "solve_equilibrium",
]

_FloatArray = NDArray[np.float64]

#: Universal gas constant, J/(mol K).
R_UNIVERSAL = 8.31446261815324
#: Reference pressure of the NASA-9 Gibbs polynomials, Pa.
P_REFERENCE = 101325.0

#: Atomic masses, kg/mol. Only the elements a carbon-phenolic surface needs.
ATOMIC_MASS = {"C": 0.0120107, "H": 0.00100794, "O": 0.0159994, "N": 0.0140067}


@dataclass(frozen=True)
class Nasa9Species:
    """One species' NASA 9-coefficient polynomial fit.

    Attributes
    ----------
    name:
        Database name, e.g. ``"CO2"`` or ``"C(gr)"``.
    composition:
        Element symbol to atom count.
    ranges:
        Temperature intervals ``(T_low, T_high)`` (K), ascending.
    coefficients:
        Nine :math:`a_i` per range.
    integration:
        The two integration constants ``(b1, b2)`` per range.
    """

    name: str
    composition: dict[str, float]
    ranges: tuple[tuple[float, float], ...]
    coefficients: tuple[tuple[float, ...], ...]
    integration: tuple[tuple[float, float], ...]

    @property
    def molar_mass(self) -> float:
        """kg/mol."""
        return sum(ATOMIC_MASS[e] * n for e, n in self.composition.items())

    def _interval(self, temperature: float) -> int:
        for i, (lo, hi) in enumerate(self.ranges):
            if lo - 1e-9 <= temperature <= hi + 1e-9:
                return i
        # Outside every fitted range: hold the nearest. NASA-9 fits diverge
        # violently beyond their endpoints, and a silently extrapolated
        # Gibbs energy poisons the whole equilibrium rather than failing.
        return 0 if temperature < self.ranges[0][0] else len(self.ranges) - 1

    def enthalpy_rt(self, temperature: float) -> float:
        """:math:`H^\\circ/(RT)`, dimensionless."""
        t = float(temperature)
        a = self.coefficients[self._interval(t)]
        b1 = self.integration[self._interval(t)][0]
        return float(
            -a[0] / t**2
            + a[1] * np.log(t) / t
            + a[2]
            + a[3] * t / 2.0
            + a[4] * t**2 / 3.0
            + a[5] * t**3 / 4.0
            + a[6] * t**4 / 5.0
            + b1 / t
        )

    def entropy_r(self, temperature: float) -> float:
        """:math:`S^\\circ/R`, dimensionless."""
        t = float(temperature)
        a = self.coefficients[self._interval(t)]
        b2 = self.integration[self._interval(t)][1]
        return float(
            -a[0] / (2.0 * t**2)
            - a[1] / t
            + a[2] * np.log(t)
            + a[3] * t
            + a[4] * t**2 / 2.0
            + a[5] * t**3 / 3.0
            + a[6] * t**4 / 4.0
            + b2
        )

    def gibbs_rt(self, temperature: float) -> float:
        """:math:`G^\\circ/(RT) = H^\\circ/(RT) - S^\\circ/R`."""
        return self.enthalpy_rt(temperature) - self.entropy_r(temperature)


_COMPOSITION = re.compile(r"([A-Z][a-z]?)\s*([0-9.]+)")


class ThermoDatabase:
    """NASA-9 polynomial data, parsed from a ``nasa9.dat``-format file.

    The format is fixed-column and comment-prefixed with ``!``. Each
    species is a name/comment line, a count-and-composition line, then
    three lines per temperature range.
    """

    def __init__(self, species: dict[str, Nasa9Species]) -> None:
        self._species = species

    def __contains__(self, name: str) -> bool:
        return name in self._species

    def __getitem__(self, name: str) -> Nasa9Species:
        if name not in self._species:
            raise KeyError(
                f"species {name!r} is not in the thermodynamic database; "
                f"the B' result would silently omit it"
            )
        return self._species[name]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._species)

    @classmethod
    def from_nasa9(cls, path: str | Path) -> ThermoDatabase:
        """Parse a NASA-9 database file."""
        lines = [
            line.rstrip("\n")
            for line in Path(path).read_text(errors="replace").splitlines()
            if not line.startswith("!")
        ]
        species: dict[str, Nasa9Species] = {}
        i = 0
        while i < len(lines) - 1:
            header = lines[i]
            if not header.strip() or header.startswith(" "):
                i += 1
                continue
            name = header.split()[0]
            info = lines[i + 1]
            try:
                n_ranges = int(info[:2])
            except ValueError:
                i += 1
                continue
            if n_ranges < 1:
                # Zero-interval entries are reference species with no fit.
                i += 3
                continue
            composition: dict[str, float] = {}
            for element, count in _COMPOSITION.findall(info[10:50]):
                if element in ATOMIC_MASS:
                    composition[element] = composition.get(element, 0.0) + float(count)
            ranges, coeffs, integrations = [], [], []
            j = i + 2
            ok = True
            for _ in range(n_ranges):
                if j + 2 >= len(lines):
                    ok = False
                    break
                bounds = lines[j]
                try:
                    lo, hi = float(bounds[:11]), float(bounds[11:22])
                except ValueError:
                    ok = False
                    break
                # NASA-9 lays the second coefficient line out as
                # ``a6 a7 <unused> b1 b2`` in five fixed-width fields. The
                # slot for the eighth coefficient is *reserved but unused*,
                # and files disagree on how to spell "empty" there: the
                # Gurvich/McBride records leave it blank, while species added
                # later (C5, C5H, and the rest of the Mutation++ additions)
                # write an explicit ``0.000000000E+00``. Parsing positionally
                # rather than by compacting non-blank fields is what makes
                # both spellings read the same. Compacting shifts b1 and b2
                # one slot left on exactly the records that fill the slot,
                # which silently substitutes 0 for the enthalpy constant and
                # the enthalpy constant for the entropy constant — an error
                # of order 1e5 in s/R that no bounds check would catch.
                head = _fixed_fields(lines[j + 1], 5)
                tail = _fixed_fields(lines[j + 2], 5)
                a = head[:5] + tail[:2]
                b1, b2 = tail[3], tail[4]
                if any(v is None for v in a) or b1 is None or b2 is None:
                    ok = False
                    break
                ranges.append((lo, hi))
                coeffs.append(tuple(float(v) for v in a))  # type: ignore[arg-type]
                integrations.append((b1, b2))
                j += 3
            if ok and composition and ranges:
                species[name] = Nasa9Species(
                    name=name,
                    composition=composition,
                    ranges=tuple(ranges),
                    coefficients=tuple(coeffs),
                    integration=tuple(integrations),
                )
            i = j if ok else i + 1
        if not species:
            raise ValueError(f"no species parsed from {path}")
        return cls(species)


def _fixed_fields(line: str, count: int, width: int = 16) -> list[float | None]:
    """Parse Fortran ``D``-exponent fixed-width reals, **keeping positions**.

    Returns exactly ``count`` entries, ``None`` where the field is blank or
    unparseable. Preserving the slot is the point: in NASA-9 the meaning of a
    number is its column, so a blank must occupy its place rather than let
    the next value slide into it.
    """
    out: list[float | None] = []
    for k in range(count):
        field = line[k * width : (k + 1) * width].strip()
        if not field:
            out.append(None)
            continue
        try:
            out.append(float(field.replace("D", "E").replace("d", "e")))
        except ValueError:
            out.append(None)
    return out


@dataclass(frozen=True)
class EquilibriumResult:
    """Gas-phase equilibrium at a given state."""

    temperature: float
    pressure: float
    species: tuple[str, ...]
    mole_fractions: _FloatArray
    element_potentials: dict[str, float]
    """:math:`\\lambda_k`, dimensionless (already divided by RT)."""
    enthalpy: float
    """Mixture specific enthalpy, J/kg."""
    molar_mass: float
    """Mixture molar mass, kg/mol."""

    def carbon_saturation(self, graphite: Nasa9Species) -> float:
        """:math:`\\lambda_C - g^\\circ_{C(gr)}/RT`.

        Zero when the gas is exactly saturated with respect to solid
        carbon, positive when carbon would condense, negative when any
        solid present would dissolve. This is the residual the B' solve
        drives to zero.
        """
        return self.element_potentials["C"] - graphite.gibbs_rt(self.temperature)


def solve_equilibrium(
    database: ThermoDatabase,
    species: tuple[str, ...],
    temperature: float,
    pressure: float,
    element_moles: dict[str, float],
    initial_potentials: dict[str, float] | None = None,
    *,
    _continuation: bool = True,
) -> EquilibriumResult:
    """Gas-phase equilibrium by the element-potential method.

    Parameters
    ----------
    element_moles:
        Relative elemental abundance. Only ratios matter; the result is
        returned as mole fractions.
    initial_potentials:
        Warm start. Continuation along a temperature or :math:`B'_c`
        sweep is worth several Newton iterations per point and, more
        importantly, keeps the solve on the same branch.
    """
    if not (np.isfinite(temperature) and temperature > 0.0):
        raise ValueError(f"temperature must be finite and > 0, got {temperature}")
    if not (np.isfinite(pressure) and pressure > 0.0):
        raise ValueError(f"pressure must be finite and > 0, got {pressure}")
    elements = tuple(sorted(k for k, v in element_moles.items() if v > 0.0))
    if not elements:
        raise ValueError("element_moles must contain at least one positive entry")

    # A species built from an element that is absent from `element_moles` has
    # no row constraining it: nothing in A n = b can hold it down, the
    # Jacobian loses rank, and the solve either wanders or reports a mixture
    # containing a species made of nothing. Dropping such species is the
    # physically correct reading — you cannot form N2 with no nitrogen.
    available = {k for k, v in element_moles.items() if v > 0.0}
    gas = [
        database[s] for s in species if "(" not in s and set(database[s].composition) <= available
    ]
    if not gas:
        raise ValueError(
            f"no gas species can be formed from elements {sorted(available)}; "
            f"every species in `species` needs an element that is absent"
        )
    names = tuple(s.name for s in gas)
    a = np.array(
        [[s.composition.get(e, 0.0) for s in gas] for e in elements]
    )  # (n_elements, n_species)
    g_rt = np.array([s.gibbs_rt(temperature) for s in gas])
    ln_p = np.log(pressure / P_REFERENCE)
    b = np.array([element_moles[e] for e in elements])
    b = b / b.sum()

    # Solved as a Newton root-find on the element-potential conditions in
    # **log space**, jointly in the potentials and the total moles.
    #
    #     nu_j = exp(lambda . a_j + c_j + ln n),   c_j = -g_j/RT - ln(P/P0)
    #
    # The conditions are (A nu = b) and (sum_j nu_j = n). Written directly
    # those are sums of exponentials, and Newton on a sum of exponentials
    # descends by exactly *one e-fold per iteration* when it starts far above
    # the root — the step solves (a a^T) d = -a, so a.d = -1 regardless of
    # how large the residual is. An earlier version did exactly that and
    # needed ~120 iterations merely to walk the initial guess down to the
    # right order of magnitude, which is why it hit its iteration cap and
    # reported non-convergence at every single state.
    #
    # Taking logs removes the pathology, because log-sum-exp is close to
    # piecewise linear in lambda:
    #
    #     F_k    = ln n + ln (A nu')_k - ln b_k,    k = 1 .. n_elements
    #     F_last = ln (sum_j nu'_j)
    #
    # where nu' is nu without the ln n shift. F_last is independent of ln n,
    # and each F_k depends on it linearly, so the Jacobian is exact and
    # cheap. Convergence is 4-8 iterations from a cold start.
    #
    # Everything is evaluated against the running maximum exponent, so no
    # intermediate overflows even where c spans 400 e-folds (T = 400 K).
    c = -g_rt - ln_p
    n_el = len(elements)
    ln_b = np.log(b)

    def _residual(
        lam: _FloatArray, ln_n: float
    ) -> tuple[_FloatArray, _FloatArray, _FloatArray, float] | None:
        z = lam @ a + c
        shift = float(z.max())
        mu = np.exp(z - shift)
        a_mu = a @ mu
        if not np.isfinite(mu).all() or np.any(a_mu <= 0.0):
            return None
        total = float(mu.sum())
        f = np.append(ln_n + shift + np.log(a_mu) - ln_b, shift + np.log(total))
        if not np.isfinite(f).all():
            return None
        return f, mu, a_mu, total

    if initial_potentials is not None:
        lam = np.array([initial_potentials.get(e, 0.0) for e in elements])
    else:
        # Place every species near equal abundance. This is deliberately not
        # a thermodynamic guess: the obvious one — put each element's whole
        # abundance on its monatomic species — is catastrophic for carbon,
        # because atomic C is so unstable that reproducing its abundance
        # demands a huge potential, which then overflows every polyatomic
        # carbon species by hundreds of e-folds.
        lam = np.linalg.lstsq(a.T, -c, rcond=None)[0]

    ln_n = 0.0
    converged = False
    for _ in range(200):
        state = _residual(lam, ln_n)
        if state is None:
            break
        f, mu, a_mu, total = state
        if float(np.max(np.abs(f))) < 1e-11:
            converged = True
            break
        jac = np.zeros((n_el + 1, n_el + 1))
        jac[:n_el, :n_el] = ((a * mu) @ a.T) / a_mu[:, None]
        jac[:n_el, n_el] = 1.0
        jac[n_el, :n_el] = a_mu / total
        # A ridge keeps the step finite when one element is present only in
        # species that have gone extinct, collapsing that row.
        jac += np.eye(n_el + 1) * (1e-12 * max(float(np.abs(jac).max()), 1.0))
        try:
            step = np.linalg.solve(jac, -f)
        except np.linalg.LinAlgError:  # pragma: no cover - singular
            break
        # Backtracking on the residual norm. The line search is what makes
        # the cold start safe: a full step from far away can push an exponent
        # out of range, and halving until the residual actually falls both
        # rejects those and handles the mild non-convexity of the log form.
        norm = float(np.linalg.norm(f))
        t = 1.0
        for _ in range(60):
            trial = _residual(lam + t * step[:n_el], ln_n + t * step[n_el])
            if trial is not None and float(np.linalg.norm(trial[0])) < norm:
                break
            t *= 0.5
        else:
            break
        lam = lam + t * step[:n_el]
        ln_n = float(ln_n + t * step[n_el])
    if not converged:
        # A cold start cannot always reach a carbon-rich mixture at low
        # temperature or high pressure, because equilibrium there is a near
        # degenerate competition between a few heavy species. Continuation
        # fixes it without the caller needing to know: solves are easy at
        # high temperature, and each rung is a warm start for the next.
        if _continuation and initial_potentials is None:
            anchor = max(4000.0, 2.0 * float(temperature))
            ladder = np.geomspace(anchor, float(temperature), 24)
            potentials: dict[str, float] | None = None
            result: EquilibriumResult | None = None
            for rung in ladder:
                try:
                    result = solve_equilibrium(
                        database,
                        species,
                        float(rung),
                        pressure,
                        element_moles,
                        potentials,
                        _continuation=False,
                    )
                except RuntimeError:
                    result = None
                    break
                potentials = result.element_potentials
            if result is not None:
                return result
        raise RuntimeError(
            f"equilibrium did not converge at T = {temperature:.6g} K, "
            f"P = {pressure:.6g} Pa, including by continuation from "
            f"{max(4000.0, 2.0 * float(temperature)):.6g} K. Gas-phase-only "
            f"equilibrium of a carbon-rich mixture is genuinely ill-posed "
            f"where condensed carbon should form; either admit a condensed "
            f"phase or restrict the state range."
        )

    final = _residual(lam, ln_n)
    assert final is not None  # converged, so it evaluates
    nu = final[1]

    x_frac = np.asarray(nu / float(nu.sum()))
    molar_mass = float(sum(x * s.molar_mass for x, s in zip(x_frac, gas, strict=True)))
    h_molar = float(
        sum(
            x * s.enthalpy_rt(temperature) * R_UNIVERSAL * temperature
            for x, s in zip(x_frac, gas, strict=True)
        )
    )
    return EquilibriumResult(
        temperature=float(temperature),
        pressure=float(pressure),
        species=names,
        mole_fractions=x_frac,
        element_potentials=dict(zip(elements, lam, strict=True)),
        enthalpy=h_molar / molar_mass,
        molar_mass=molar_mass,
    )


@dataclass(frozen=True)
class SurfaceComposition:
    """Elemental mass fractions of the three streams at an ablating surface.

    Attributes
    ----------
    edge, pyrolysis, char:
        Element symbol to **mass** fraction, each summing to 1.
    species:
        Gas species carried in the equilibrium.
    """

    edge: dict[str, float]
    pyrolysis: dict[str, float]
    char: dict[str, float]
    species: tuple[str, ...]
    name: str = ""

    def __post_init__(self) -> None:
        for label in ("edge", "pyrolysis", "char"):
            stream = getattr(self, label)
            total = sum(stream.values())
            if not np.isclose(total, 1.0, atol=1e-6):
                raise ValueError(f"{label} elemental mass fractions must sum to 1, got {total:.6g}")
            if any(v < 0.0 for v in stream.values()):
                raise ValueError(f"{label} mass fractions must be >= 0")

    def wall_elements(self, b_g: float, b_c: float) -> dict[str, float]:
        """Elemental mass fractions leaving the wall, from the B' balance."""
        denominator = 1.0 + b_g + b_c
        out: dict[str, float] = {}
        for e in ATOMIC_MASS:
            out[e] = (
                self.edge.get(e, 0.0)
                + b_g * self.pyrolysis.get(e, 0.0)
                + b_c * self.char.get(e, 0.0)
            ) / denominator
        return out


def _element_moles(mass_fractions: dict[str, float]) -> dict[str, float]:
    return {e: y / ATOMIC_MASS[e] for e, y in mass_fractions.items() if y > 0.0}


def solve_bprime(
    database: ThermoDatabase,
    composition: SurfaceComposition,
    temperature: float,
    pressure: float,
    gas_rate: float,
    bracket: tuple[float, float] = (1.0e-8, 50.0),
    graphite: str = "C(gr)",
) -> tuple[float, EquilibriumResult]:
    """Solve for :math:`B'_c` at one surface state.

    Returns ``(b_c, equilibrium)``. The equilibrium is the wall gas at
    the converged :math:`B'_c`, so its ``enthalpy`` is :math:`h_w`.

    Raises
    ------
    ValueError
        If the surface is *not* saturated with carbon anywhere in the
        bracket. Above the sublimation temperature no steady ablation
        rate exists — the carbon is gone at any :math:`B'_c` — and there
        is no root to find. That is a physical boundary, and reporting it
        as one is the whole reason this returns an error rather than the
        bracket endpoint the way a table generator's ceiling does.
    """
    solid = database[graphite]

    def saturation(b_c: float) -> float:
        wall = composition.wall_elements(gas_rate, b_c)
        result = solve_equilibrium(
            database,
            composition.species,
            temperature,
            pressure,
            _element_moles(wall),
        )
        return result.carbon_saturation(solid)

    lo, hi = bracket
    f_lo, f_hi = saturation(lo), saturation(hi)
    if f_lo * f_hi > 0.0:
        raise ValueError(
            f"no carbon-saturated state at T_w = {temperature:.6g} K, "
            f"P = {pressure:.6g} Pa, B'_g = {gas_rate:.6g}: the saturation "
            f"residual is {f_lo:+.3e} at B'_c = {lo:g} and {f_hi:+.3e} at "
            f"{hi:g}. Above the sublimation boundary this is expected and "
            f"means no steady ablation rate exists."
        )
    b_c = float(scipy.optimize.brentq(saturation, lo, hi, xtol=1e-12, rtol=1e-13))
    wall = composition.wall_elements(gas_rate, b_c)
    return b_c, solve_equilibrium(
        database, composition.species, temperature, pressure, _element_moles(wall)
    )
