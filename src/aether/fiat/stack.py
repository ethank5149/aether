"""Multilayer TPS stack and its finite-volume grid.

FIAT's defining geometric capability is "a multilayer stack of isotropic
materials and structure that can ablate from a front surface and decompose
in depth" (Chen & Milos 1999, abstract). Only the *top* ply ablates; the
plies beneath it are fixed in thickness and may be non-decomposing
structure (honeycomb, bondline, substructure).

Two choices here differ from FIAT and are deliberate.

**Recession is handled by a Landau stretch of the top ply, not by
consuming cells.** FIAT shrinks the first ply and re-grids as the surface
recedes, which drops nodes discretely. Stretching the top ply onto a fixed
computational coordinate is an exact change of variable, keeps the node
count constant, and makes the grid motion smooth enough for an exact
Newton Jacobian. The two converge to the same answer under refinement;
this one converges monotonically.

**Cell widths are geometric, refined toward the heated face.** The
pyrolysis front is thin and travels inward from the surface, so uniform
spacing wastes resolution at the backface. ``growth`` is the ratio of
successive cell widths going inward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from aether.fiat.materials import (
    MultiComponentMaterial,
    PressureConductivity,
    TabulatedConductivity,
)
from aether.thermal.material import CharringMaterial

#: A three-component CMA material or the generalised N-component form.
MaterialLike = CharringMaterial | MultiComponentMaterial

__all__ = ["MaterialLike", "MaterialStack", "Ply", "StackGrid"]

_FloatArray = NDArray[np.float64]
_IntArray = NDArray[np.intp]


@dataclass(frozen=True)
class Ply:
    """One material layer in the stack.

    Attributes
    ----------
    material:
        The charring model for this layer. A non-decomposing structural
        layer is expressed as a :class:`CharringMaterial` whose Arrhenius
        pre-exponentials are zero, which is exactly how FIAT's material
        database represents structure.
    thickness:
        Initial thickness (m), :math:`> 0`.
    n_cells:
        Finite-volume cells in this ply, :math:`\\ge 2`.
    growth:
        Ratio of successive cell widths going *inward*. ``1.0`` is
        uniform; ``1.05`` refines toward the heated face of this ply.
    ablating:
        Whether this ply may recede. Only the top ply may set this.
    pressure_conductivity:
        Optional :class:`~aether.fiat.materials.PressureConductivity`
        replacing ``material.conductivity``. A porous ablator's
        conductivity depends on the pressure of the gas in its pores as
        well as on temperature and char fraction, and for PICA that
        dependence is published and large — see
        :data:`~aether.fiat.materials.MEDLI2_PICA_CONDUCTIVITY`.
        ``None`` keeps the pressure-independent property on the material.
    extinction_coefficient:
        :math:`K = a + \\sigma_s` (1/m) for in-depth radiation transport.
        ``None`` — the default — means the ply is opaque and carries no
        internal radiative flux, which is FIAT's behaviour for dense
        materials and structure. Lightweight ablators are
        semi-transparent and need a value here.
    """

    material: MaterialLike
    thickness: float
    n_cells: int
    growth: float = 1.0
    ablating: bool = False
    pressure_conductivity: PressureConductivity | TabulatedConductivity | None = None
    extinction_coefficient: float | None = None

    def __post_init__(self) -> None:
        if self.extinction_coefficient is not None and not (
            np.isfinite(self.extinction_coefficient) and self.extinction_coefficient > 0.0
        ):
            raise ValueError(
                f"extinction_coefficient must be finite and > 0 when given, "
                f"got {self.extinction_coefficient}"
            )
        if not (np.isfinite(self.thickness) and self.thickness > 0.0):
            raise ValueError(f"thickness must be finite and > 0, got {self.thickness}")
        if self.n_cells < 2:
            raise ValueError(f"n_cells must be >= 2, got {self.n_cells}")
        if not (np.isfinite(self.growth) and 0.5 <= self.growth <= 2.0):
            raise ValueError(
                f"growth must be finite and in [0.5, 2.0]; values outside that "
                f"range make the cell-width distribution pathological, got {self.growth}"
            )

    def cell_widths(self, thickness: float | None = None) -> _FloatArray:
        """Geometric cell widths (m), summing exactly to the thickness."""
        total = self.thickness if thickness is None else float(thickness)
        if not (np.isfinite(total) and total > 0.0):
            raise ValueError(f"thickness must be finite and > 0, got {total}")
        if self.growth == 1.0:
            widths = np.full(self.n_cells, 1.0)
        else:
            widths = self.growth ** np.arange(self.n_cells, dtype=np.float64)
        # Normalising rather than closed-forming the geometric sum keeps the
        # widths summing to `total` to machine precision at any growth ratio.
        return np.asarray(widths * (total / widths.sum()))


@dataclass(frozen=True)
class StackGrid:
    """Finite-volume geometry of a stack at one instant.

    All coordinates are measured from the *current* heated surface,
    increasing inward — FIAT's moving :math:`x` coordinate.

    Attributes
    ----------
    faces:
        Cell-face positions (m), length ``n_cells + 1``, ``faces[0] == 0``.
    centers:
        Cell-center positions (m), length ``n_cells``.
    widths:
        Cell widths (m), length ``n_cells``.
    ply_index:
        Owning ply of each cell, length ``n_cells``.
    interface_faces:
        Indices into ``faces`` that lie on a ply–ply interface.
    """

    faces: _FloatArray
    centers: _FloatArray
    widths: _FloatArray
    ply_index: _IntArray
    interface_faces: _IntArray

    @property
    def n_cells(self) -> int:
        return int(self.widths.size)

    @property
    def total_thickness(self) -> float:
        return float(self.faces[-1])


class MaterialStack:
    """An ordered stack of plies, heated face first.

    Parameters
    ----------
    plies:
        Outermost (heated) ply first. Exactly one ply may be ablating,
        and if any is, it must be ``plies[0]`` — a buried layer cannot
        recede while the layer above it is intact.
    """

    def __init__(self, plies: list[Ply]) -> None:
        if not plies:
            raise ValueError("a stack needs at least one ply")
        ablating = [i for i, p in enumerate(plies) if p.ablating]
        if len(ablating) > 1:
            raise ValueError(f"at most one ply may ablate, got {len(ablating)}")
        if ablating and ablating[0] != 0:
            raise ValueError(
                f"only the outermost ply may ablate; ply {ablating[0]} is marked "
                f"ablating but lies beneath {ablating[0]} intact ply(s)"
            )
        self._plies = tuple(plies)

    @property
    def plies(self) -> tuple[Ply, ...]:
        return self._plies

    @property
    def n_plies(self) -> int:
        return len(self._plies)

    @property
    def n_cells(self) -> int:
        return sum(p.n_cells for p in self._plies)

    @property
    def ablating(self) -> bool:
        """Whether the stack can recede at all."""
        return self._plies[0].ablating

    @property
    def initial_thickness(self) -> float:
        return float(sum(p.thickness for p in self._plies))

    def top_thickness(self, recession: float) -> float:
        """Remaining thickness of the top ply after ``recession`` metres.

        Raises once the top ply is fully consumed. FIAT's behaviour there
        is to expose the next ply; that is a genuinely different problem
        (the surface thermochemistry table changes material) and this
        implementation refuses it rather than silently continuing with
        the wrong B' table.
        """
        s = float(recession)
        if not np.isfinite(s) or s < 0.0:
            raise ValueError(f"recession must be finite and >= 0, got {s}")
        remaining = self._plies[0].thickness - s
        if remaining <= 0.0:
            raise ValueError(
                f"recession {s:.6g} m has consumed the entire top ply "
                f"({self._plies[0].thickness:.6g} m). Burn-through exposes the "
                f"next ply, whose surface thermochemistry differs; this solver "
                f"does not continue past that point."
            )
        return float(remaining)

    def grid(self, recession: float = 0.0) -> StackGrid:
        """Build the finite-volume geometry at the given recession."""
        widths_per_ply = []
        for i, ply in enumerate(self._plies):
            thickness = self.top_thickness(recession) if i == 0 else ply.thickness
            widths_per_ply.append(ply.cell_widths(thickness))
        widths = np.concatenate(widths_per_ply)
        faces = np.concatenate([[0.0], np.cumsum(widths)])
        centers = 0.5 * (faces[:-1] + faces[1:])
        ply_index = np.concatenate(
            [np.full(p.n_cells, i, dtype=np.intp) for i, p in enumerate(self._plies)]
        )
        interface_faces = np.cumsum([p.n_cells for p in self._plies[:-1]], dtype=np.intp)
        return StackGrid(
            faces=faces,
            centers=centers,
            widths=widths,
            ply_index=ply_index,
            interface_faces=np.asarray(interface_faces, dtype=np.intp),
        )

    def cell_materials(self) -> list[MaterialLike]:
        """Per-cell material, in grid order."""
        out: list[MaterialLike] = []
        for ply in self._plies:
            out.extend([ply.material] * ply.n_cells)
        return out

    def stretch_factor(self, recession: float) -> float:
        """Landau stretch :math:`\\sigma = (L_1 - s)/L_1` for the top ply.

        Cell widths in the top ply scale by this; everything below is
        unaffected. Its time derivative supplies the grid-motion term of
        Eq. (1).
        """
        return self.top_thickness(recession) / self._plies[0].thickness
