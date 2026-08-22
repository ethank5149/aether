"""Read equilibrium :math:`B'` tables produced by Mutation++.

FIAT closes its surface energy balance with tables "computed by the
program ACE or the program MAT" (Chen & Milos 1999). Neither is
available here, but **Mutation++** is open source and computes the same
thing: a multiphase equilibrium of the ablating surface, giving the
non-dimensional char consumption rate :math:`B'_c` and the wall enthalpy
:math:`h_w` over a grid of pressure, pyrolysis gas rate :math:`B'_g` and
wall temperature.

This module reads the output of ``generate_bprime_table``
(Padovan et al., "An extended B' formulation for ablating-surface
boundary conditions," *Int. J. Heat and Mass Transfer*, 2023), which is
a modified form of Mutation++'s own ``bprime`` application. The material
normally used with it is **TACOT** — the Theoretical Ablative Composite
for Open Testing, an open surrogate for PICA that exists precisely
because PICA's real property set is restricted.

The sublimation ceiling
-----------------------

The raw output contains a flat series of :math:`B'_c` values — 200, 210,
… 500 — that are **not physical**. They are the solver's ceiling,
reported where the equilibrium has consumed all the solid carbon and no
steady ablation rate exists. They track the :math:`B'_g` grid rather than
temperature, which gives them away.

The clean discriminator is not a threshold on :math:`B'_c` but the
graphite mole fraction :math:`x_{C(gr)}` in the same row: it is exactly
1 where a condensed carbon phase is present and exactly 0 where it is
not, and in a 15300-point TACOT table that partition coincides with the
capped rows to the last entry. So this reader classifies by carbon
presence, not by magnitude — genuine :math:`B'_c` values as high as 147
survive, and every capped value is removed.

What is done with the capped region matters. Interpolating through it
would be indefensible: a cubic spline crossing a jump from 3 to 200
oscillates violently, and the fictitious values it produces are what a
solver would then chase. Instead the sublimation boundary is recorded
as a surface in :math:`(P, B'_g)`, the capped entries are replaced by
extrapolation of the last physical value, and
:meth:`SublimationLimit.exceeded` lets a caller ask whether a state has
crossed it. Crossing it means the surface material is gone, which is a
modelling event, not an interpolation problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from aether.fiat.bprime import BPrimeTable

__all__ = [
    "MUTATIONPP_COLUMNS",
    "MutationppTable",
    "SublimationLimit",
    "read_mutationpp_bprime",
]

_FloatArray = NDArray[np.float64]

#: The five leading columns of ``generate_bprime_table`` output.
#:
#: Note the units, both of which differ from what this package uses
#: internally: pressure is in **bar** and wall enthalpy in **MJ/kg**.
#: (The tool's own README says kJ/kg; its column header says MJ/kg, and
#: the magnitudes — a few tens at most — confirm the header.)
MUTATIONPP_COLUMNS = ("P[bar]", "B'g", "Tw[K]", "B'c", "hw[MJ/kg]")

_BAR_TO_PA = 1.0e5
_MJ_PER_KG_TO_J_PER_KG = 1.0e6


@dataclass(frozen=True)
class SublimationLimit:
    """Where the equilibrium runs out of condensed carbon.

    Attributes
    ----------
    pressures, gas_rates:
        The table's :math:`P` (Pa) and :math:`B'_g` axes.
    temperature:
        Lowest wall temperature (K) at which no condensed carbon phase
        remains, for each ``(pressure, gas_rate)``. ``inf`` where the
        table never reaches sublimation.
    """

    pressures: _FloatArray
    gas_rates: _FloatArray
    temperature: _FloatArray

    def limit(self, pressure: float, gas_rate: float) -> float:
        """Sublimation onset temperature (K) at the nearest tabulated axes.

        Nearest-node rather than interpolated on purpose: this is the
        edge of a physical regime, and smoothing it would invent a
        boundary the equilibrium solve never reported.
        """
        i = int(np.argmin(np.abs(self.pressures - float(pressure))))
        j = int(np.argmin(np.abs(self.gas_rates - float(gas_rate))))
        return float(self.temperature[i, j])

    def exceeded(self, pressure: float, gas_rate: float, wall_temperature: float) -> bool:
        """Whether a surface state has passed the sublimation boundary."""
        return float(wall_temperature) >= self.limit(pressure, gas_rate)


@dataclass(frozen=True)
class MutationppTable:
    """A parsed Mutation++ B' table plus the physics the parse discovered."""

    table: BPrimeTable
    """Ready for the surface energy balance, in SI units."""
    sublimation: SublimationLimit
    capped_fraction: float
    """Share of raw rows that were solver ceiling rather than physics."""
    provenance: str


def read_mutationpp_bprime(
    path: str | Path,
    *,
    method: str = "cubic",
    max_gas_rate: float | None = None,
    provenance: str = "",
) -> MutationppTable:
    """Parse ``generate_bprime_table`` output into a :class:`BPrimeTable`.

    Parameters
    ----------
    path:
        The tool's stdout, one header line then one row per state.
    method:
        Interpolation order for the resulting table.
    max_gas_rate:
        Optionally truncate the :math:`B'_g` axis. Tables are routinely
        generated far wider than a run needs, and a narrower axis makes
        the no-extrapolation guard bite where it is useful.
    provenance:
        Free text recorded on the result. Give it the material and the
        generating command; a B' table without those is not reusable.

    Notes
    -----
    Rows are required to form a complete rectangular grid. The tool emits
    one, but a truncated or concatenated file will not, and silently
    reshaping such a file would scramble the axes rather than fail.
    """
    raw = np.loadtxt(Path(path), skiprows=1)
    if raw.ndim != 2 or raw.shape[1] < 6:
        raise ValueError(
            f"expected the 5 B' columns plus at least one species column, "
            f"got shape {raw.shape}"
        )
    pressure_bar, gas_rate, wall_t, char_rate, hw_mj = raw[:, :5].T
    # The final column is the condensed-carbon mole fraction: 1 where a
    # graphite phase exists, 0 where the equilibrium has consumed it.
    carbon = raw[:, -1]
    if not np.all(np.isin(carbon, (0.0, 1.0))):
        raise ValueError(
            "the last column is not a 0/1 condensed-carbon indicator; this "
            "reader identifies the sublimation ceiling by carbon presence and "
            "cannot classify without it"
        )

    p_axis = np.unique(pressure_bar)
    b_axis = np.unique(gas_rate)
    t_axis = np.unique(wall_t)
    if p_axis.size * b_axis.size * t_axis.size != raw.shape[0]:
        raise ValueError(
            f"rows do not form a complete grid: {raw.shape[0]} rows against "
            f"{p_axis.size} x {b_axis.size} x {t_axis.size} axis values"
        )

    shape = (p_axis.size, b_axis.size, t_axis.size)
    idx = (
        np.searchsorted(p_axis, pressure_bar),
        np.searchsorted(b_axis, gas_rate),
        np.searchsorted(t_axis, wall_t),
    )
    b_c = np.full(shape, np.nan)
    h_w = np.full(shape, np.nan)
    has_carbon = np.zeros(shape, dtype=bool)
    b_c[idx] = char_rate
    h_w[idx] = hw_mj * _MJ_PER_KG_TO_J_PER_KG
    has_carbon[idx] = carbon > 0.5
    if np.any(np.isnan(b_c)):  # pragma: no cover - guarded by the grid check
        raise ValueError("grid has holes after reshaping")

    capped = ~has_carbon
    capped_fraction = float(capped.mean())

    # Sublimation onset: the lowest temperature with no condensed carbon.
    onset = np.full(shape[:2], np.inf)
    for i in range(shape[0]):
        for j in range(shape[1]):
            hit = np.flatnonzero(capped[i, j])
            if hit.size:
                onset[i, j] = t_axis[hit[0]]

    # Replace the ceiling with a hold of the last physical value. The result
    # is never used below the sublimation boundary, and holding rather than
    # zeroing keeps the interpolant monotone across the edge so a Newton
    # iterate that strays past it is pushed back rather than pulled onward.
    for i in range(shape[0]):
        for j in range(shape[1]):
            bad = capped[i, j]
            if not bad.any():
                continue
            good = np.flatnonzero(~bad)
            if good.size == 0:
                raise ValueError(
                    f"no physical B'_c anywhere at P={p_axis[i]:.4g} bar, "
                    f"B'_g={b_axis[j]:.4g}: the table is unusable there"
                )
            b_c[i, j, bad] = b_c[i, j, good[-1]]
            h_w[i, j, bad] = h_w[i, j, good[-1]]

    if max_gas_rate is not None:
        keep = b_axis <= float(max_gas_rate)
        if keep.sum() < 4:
            raise ValueError("max_gas_rate leaves fewer than 4 B'_g nodes")
        b_axis, b_c, h_w = b_axis[keep], b_c[:, keep], h_w[:, keep]
        onset = onset[:, keep]

    table = BPrimeTable(
        p_axis * _BAR_TO_PA,
        b_axis,
        t_axis,
        np.maximum(b_c, 0.0),
        h_w,
        method=method,
    )
    return MutationppTable(
        table=table,
        sublimation=SublimationLimit(
            pressures=p_axis * _BAR_TO_PA,
            gas_rates=b_axis,
            temperature=onset,
        ),
        capped_fraction=capped_fraction,
        provenance=provenance or f"Mutation++ B' table read from {Path(path).name}",
    )
