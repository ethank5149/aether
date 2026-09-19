r"""Recession and centre-of-gravity travel, on a wall-normal graded grid.

A heat shield that ablates does two things to a trajectory. It absorbs energy,
which is what it is for, and it **moves the centre of gravity**, which nobody
budgets for. A re-entry body that loses a few kilograms off its nose shifts its
CG aft by centimetres, and a centimetre against a metre-class body is a percent
of reference length — the same order as the static margin it was trimmed on.
The trim angle moves, the trim :math:`L/D` moves with it, and the footprint
moves with that.

Verification
------------

Steady ablation under a constant flux has a closed-form recession rate, and it
is **not** :math:`q/\rho h_{\mathrm{eff}}`. The incoming flux has to supply the
latent heat *and* raise virgin material from its initial temperature to the
ablation temperature as the surface sweeps through it:

.. math::

    \dot s = \frac{q}{\rho\,(h_{\mathrm{eff}} + c_p \Delta T)} .

Dropping the sensible term over-predicts the rate by 14 % for the properties
used in the tests, which is the sort of discrepancy easily mistaken for a
numerical deficiency in the solver — it was, here, until the balance was
written out. This module reproduces the expression above to better than a
percent, and grid-converges to 0.4 % across a fourfold refinement of the wall
cell.

The grid
--------

One-dimensional through the thickness, at every wall face, on the graded
columns :func:`~aether.geometry.columns.wall_columns` builds. That is not an
optimisation, it is the only affordable way to resolve the problem: heat
entering a heat shield goes into the wall, the gradient normal to the surface
is enormous and the gradients along it are mild, and recession is set by the
first millimetre or two.

This replaced a uniform Cartesian voxel grid, and the difference is worth
stating because the old version looked like it worked. To put a millimetre at
the wall a Cartesian grid must put a millimetre everywhere, so at any affordable
size its cells were centimetres; the surface never got hot enough, and a body
under six megawatts per square metre reported four percent of the energy doing
removal work while the rest bulk-heated voxels far thicker than any real
ablation layer. Recession came out three orders of magnitude low, and it came
out low *smoothly*, which is what made it easy to believe.

The columns are those of :class:`~aether.fiat.stack.MaterialStack`, deliberately.
This module solves a single-material enthalpy formulation — fast, and enough for
CG bookkeeping across a trajectory. When the answer needs pyrolysis gas, char
mixing rules and a real surface energy balance, the same grid feeds
:class:`~aether.fiat.solver.FiatSolver`, which implements the published
Chen–Milos formulation properly.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from aether.geometry.columns import WallColumnGrid

__all__ = [
    "AblationSolver",
    "AblationState",
    "TPSMaterial",
    "UnderResolvedGrid",
]

_FloatArray = NDArray[np.float64]


class UnderResolvedGrid(UserWarning):
    """The wall cell is coarser than the thermal layer it has to resolve.

    Heat diffuses :math:`\\sqrt{\\alpha t}` in time :math:`t`. A first cell
    thicker than that averages the temperature over material the heat has not
    reached, so the surface stays too cool and recession comes out too low —
    while the run converges and looks entirely healthy. Refining the *time* step
    does not help; this is a spatial limit.
    """


@dataclass(frozen=True)
class TPSMaterial:
    """Thermal-protection material, as the enthalpy formulation needs it.

    ``heat_of_ablation`` is the **effective** heat of ablation: energy per
    kilogram actually removed, lumping pyrolysis, char formation and blowing
    into one number. That lumping is this module's central modelling
    assumption and a real one — a charring ablator's effective value is a
    function of local flux and pressure rather than a constant, so a figure
    quoted at peak heating over-predicts removal in the tail of a trajectory.
    :mod:`aether.fiat` exists because that assumption eventually has to go.
    """

    density: float
    """kg/m³."""
    specific_heat: float
    """J/(kg·K)."""
    conductivity: float
    """W/(m·K)."""
    heat_of_ablation: float
    """J/kg, effective."""
    ablation_temperature: float
    """K, surface temperature at which material leaves."""
    emissivity: float = 0.85
    """Surface emissivity for re-radiation.

    Not decoration: at 3000 K a black surface re-radiates about 4.6 MW/m²,
    which is the same order as the convective heating it is balancing. A model
    that omits it ablates a vehicle that would really have come to radiative
    equilibrium and barely receded at all.
    """
    mushy_range: float = 10.0
    """Width in K over which the transition is smeared — numerics, not physics."""

    def __post_init__(self) -> None:
        for name in ("density", "specific_heat", "conductivity",
                     "heat_of_ablation", "ablation_temperature"):
            value = float(getattr(self, name))
            if not (np.isfinite(value) and value > 0.0):
                msg = f"{name} must be finite and > 0, got {value}"
                raise ValueError(msg)
        if not 0.0 <= float(self.emissivity) <= 1.0:
            msg = f"emissivity must lie in [0, 1], got {self.emissivity}"
            raise ValueError(msg)

    @property
    def latent_heat(self) -> float:
        """Volumetric energy to remove a unit volume of material (J/m³)."""
        return float(self.density * self.heat_of_ablation)

    @property
    def diffusivity(self) -> float:
        """:math:`\\alpha = k/\\rho c_p` (m²/s)."""
        return float(self.conductivity / (self.density * self.specific_heat))


@dataclass(frozen=True)
class AblationState:
    """Mass properties at one instant, and what they moved from."""

    time: float
    mass: float
    centroid: _FloatArray
    recession: _FloatArray
    """Per-column recession (m)."""
    peak_recession: float
    mass_lost_fraction: float
    peak_wall_temperature: float

    def summary(self) -> str:
        return (
            f"t={self.time:6.1f}s  m={self.mass:8.3f} kg "
            f"({100 * self.mass_lost_fraction:5.2f} % lost)  "
            f"cg_x={self.centroid[0]:.4f} m  "
            f"peak recession {1e3 * self.peak_recession:6.3f} mm  "
            f"Tw_max {self.peak_wall_temperature:6.0f} K"
        )


class AblationSolver:
    """Through-thickness enthalpy solver on wall-normal columns.

    Vectorised across columns rather than looped: every column has the same
    cell count, so the whole body is one ``(n_columns, n_cells)`` array and a
    step is a handful of array operations. A body with a few thousand wall
    faces steps in milliseconds, which is what makes it usable *inside* a
    trajectory rather than as an offline post-process.
    """

    def __init__(
        self,
        material: TPSMaterial,
        columns: WallColumnGrid,
        initial_temperature: float = 300.0,
    ) -> None:
        self.material = material
        self.columns = columns
        self.time = 0.0

        shape = (columns.n_columns, columns.n_cells)
        self.temperature = np.full(shape, float(initial_temperature))
        self.phi = np.ones(shape)
        self._volume = columns.volume()
        self._initial_mass = self.mass
        self._widths = np.asarray(columns.widths)
        # Distance between neighbouring cell centres, for the conduction flux.
        self._gap = 0.5 * (self._widths[:, :-1] + self._widths[:, 1:])

    # -- state -----------------------------------------------------------

    @property
    def enthalpy(self) -> _FloatArray:
        m = self.material
        return np.asarray(
            m.density * m.specific_heat * self.temperature
            + m.latent_heat * (1.0 - self.phi)
        )

    @property
    def mass(self) -> float:
        return float(self.material.density * (self.phi * self._volume).sum())

    @property
    def wall_temperature(self) -> _FloatArray:
        """Temperature of the outermost cell still holding material."""
        alive = self.phi > 0.0
        first = np.argmax(alive, axis=1)
        empty = ~alive.any(axis=1)
        value = self.temperature[np.arange(self.temperature.shape[0]), first]
        # A fully consumed column has no wall to report a temperature for.
        # Reporting its last dead cell would put a runaway number into the peak.
        return np.asarray(np.where(empty, np.nan, value))

    def recession(self) -> _FloatArray:
        """How far each column's surface has moved inward (m).

        Sub-cell, and it has to be. Counting whole emptied cells makes
        recession a staircase that reads zero until a cell clears completely,
        which on any affordable grid is most of a trajectory — a body that has
        demonstrably lost mass would report no recession at all. The depleted
        fraction of every cell is already carried by the diffuse interface, so
        summing it is both continuous and free.
        """
        return np.asarray(((1.0 - self.phi) * self._widths).sum(axis=1))

    def centroid(self) -> _FloatArray:
        """Centre of mass of the remaining material (m).

        Only what the columns hold. A vehicle is mostly structure and payload
        that never ablates, so this is the *heat shield's* centroid; combine it
        with the inert mass to get the vehicle's, which is what
        :meth:`vehicle_centroid` does.
        """
        weight = self.phi * self._volume
        total = weight.sum()
        if total <= 0.0:
            return np.zeros(3)
        positions = self.columns.positions()
        return np.asarray(
            (positions * weight[:, :, None]).sum(axis=(0, 1)) / total
        )

    def vehicle_centroid(
        self, inert_mass: float, inert_centroid: _FloatArray
    ) -> _FloatArray:
        """Combined CG of the ablating shield and everything that is not.

        The number a trajectory actually needs. Shield recession moves the
        vehicle CG by the shield's mass fraction times its own CG travel, so
        reporting the shield's shift alone overstates the effect by however
        much of the vehicle is inert — typically a factor of several.
        """
        shield = self.mass
        total = shield + float(inert_mass)
        if total <= 0.0:
            msg = "total mass must be > 0"
            raise ValueError(msg)
        return np.asarray(
            (shield * self.centroid() + float(inert_mass) * np.asarray(inert_centroid))
            / total
        )

    def state(self) -> AblationState:
        lost = (
            0.0 if self._initial_mass <= 0.0 else 1.0 - self.mass / self._initial_mass
        )
        recession = self.recession()
        return AblationState(
            time=self.time,
            mass=self.mass,
            centroid=self.centroid(),
            recession=recession,
            peak_recession=float(recession.max()) if recession.size else 0.0,
            mass_lost_fraction=float(lost),
            peak_wall_temperature=float(np.nanmax(self.wall_temperature)),
        )

    # -- stepping --------------------------------------------------------

    def stable_step(self, wall_flux: _FloatArray | float) -> float:
        """Largest time step this state can take.

        **Diffusion does not appear here**, and that is the point of solving it
        implicitly. The explicit limit is :math:`w^2/2\\alpha` on the *thinnest*
        cell, and grading puts that cell at the wall by construction — so the
        refinement that buys accuracy is exactly what destroys the step. On the
        waverider's 271 µm wall cell it is 10 ms, which is twelve thousand
        steps for two minutes of flight, per configuration. Backward Euler is
        unconditionally stable and costs one tridiagonal solve.

        What remains is **deposition**: a step must not deliver more energy to
        the wall cell than it takes to consume it. Overshooting does not
        diverge, it silently books energy as sensible heat that should have
        removed material — mass loss too *small* for the energy absorbed, which
        reads as conservative rather than wrong.
        """
        m = self.material
        thinnest = float(np.min(self._widths))
        peak = float(np.max(np.abs(np.asarray(wall_flux))))
        # No heating: nothing bounds the step, so cap it at something a caller
        # can still see progress through rather than returning infinity.
        if peak <= 0.0:
            return 10.0
        budget = (
            m.latent_heat + m.density * m.specific_heat * m.ablation_temperature
        ) * thinnest
        return float(0.5 * budget / peak)

    def step(self, wall_flux: _FloatArray, dt: float) -> AblationState:
        """Advance conduction, re-radiation and phase change by ``dt``."""
        flux = np.broadcast_to(
            np.asarray(wall_flux, dtype=np.float64), (self.columns.n_columns,)
        )
        if dt <= 0.0:
            msg = f"time step must be > 0, got {dt}"
            raise ValueError(msg)
        stable = self.stable_step(flux)
        if dt > stable:
            msg = (
                f"dt = {dt:g} s exceeds the stable step {stable:g} s for this "
                f"grid, material and heating. Use AblationSolver.advance, which "
                f"sub-cycles for you."
            )
            raise ValueError(msg)

        m = self.material

        # Re-radiation is evaluated at the old wall temperature. Lagging it is
        # safe where the implicit diffusion is not: it is a *loss* that grows
        # steeply with temperature, so lagging under-estimates cooling and errs
        # toward more ablation rather than less.
        net_wall = flux - (
            m.emissivity * 5.670374419e-8 * (self.wall_temperature**4 - 300.0**4)
        )

        # Cells whose material has gone are frozen out of the system entirely,
        # and the wall flux is applied to the first cell that still has any.
        # Leaving them in is not a small error: a consumed cell keeps absorbing
        # and conducting, and since the enthalpy inversion recovers its
        # temperature as (H - L)/rho c_p with nothing left to melt, that
        # temperature runs away — 221,000 K on this vehicle, while the energy
        # it swallowed went missing from the recession it should have driven.
        alive = self.phi > 0.0
        front = np.argmax(alive, axis=1)
        rows = np.arange(alive.shape[0])
        has_material = alive.any(axis=1)

        # Backward Euler on conduction. The finite-volume flux between
        # neighbours uses the centre-to-centre distance, not a cell width, or
        # the grading itself introduces a spurious source.
        capacity = m.density * m.specific_heat * self._widths / dt
        both_alive = alive[:, :-1] & alive[:, 1:]
        coupling = np.where(both_alive, m.conductivity / self._gap, 0.0)

        lower = np.zeros_like(self._widths)
        upper = np.zeros_like(self._widths)
        lower[:, 1:] = -coupling
        upper[:, :-1] = -coupling
        diagonal = capacity.copy()
        diagonal[:, :-1] += coupling
        diagonal[:, 1:] += coupling
        # Back face adiabatic: a heat shield is bonded to structure cold and
        # thick against the layer being solved, and letting heat leave there
        # would make recession depend on where the column was truncated.

        right = capacity * self.temperature
        right[rows[has_material], front[has_material]] += net_wall[has_material]

        # A dead cell is held at its own temperature by a trivial equation, so
        # the matrix stays the same shape and the solve stays one sweep.
        dead = ~alive
        diagonal[dead] = 1.0
        lower[dead] = 0.0
        upper[dead] = 0.0
        right[dead] = self.temperature[dead]

        solved = _solve_tridiagonal(lower, diagonal, upper, right)
        # Latent heat is carried by the enthalpy, not by the linear solve: the
        # diffusion step moves sensible energy, and the inversion below decides
        # what that does to the solid fraction.
        enthalpy = m.density * m.specific_heat * solved + m.latent_heat * (1.0 - self.phi)
        self._invert(enthalpy)
        self.time += float(dt)
        return self.state()

    def advance(self, wall_flux: _FloatArray, duration: float) -> AblationState:
        """Advance by ``duration``, sub-cycling at whatever is stable.

        The interface a coupled run wants: the CFD sets a coupling interval —
        how often it is worth re-solving the flow — and that has nothing to do
        with what conduction and deposition allow. Making the caller reconcile
        the two is making them rediscover :meth:`stable_step` at every call site.
        """
        if duration <= 0.0:
            msg = f"duration must be > 0, got {duration}"
            raise ValueError(msg)

        penetration = float(np.sqrt(self.material.diffusivity * float(duration)))
        coarsest = float(np.max(self.columns.wall_spacing))
        if coarsest > penetration:
            warnings.warn(
                f"the coarsest wall cell is {1e3 * coarsest:.2f} mm against "
                f"{1e3 * penetration:.2f} mm of thermal penetration in "
                f"{duration:g} s; recession and CG shift will be "
                f"under-predicted. Raise n_cells or growth in wall_columns, or "
                f"read the result as a lower bound.",
                UnderResolvedGrid,
                stacklevel=2,
            )

        flux = np.broadcast_to(
            np.asarray(wall_flux, dtype=np.float64), (self.columns.n_columns,)
        )
        target = self.time + float(duration)
        state = self.state()
        while self.time < target - 1.0e-12:
            state = self.step(flux, min(0.9 * self.stable_step(flux), target - self.time))
        return state

    def _invert(self, enthalpy: _FloatArray) -> None:
        """Recover temperature and solid fraction from total enthalpy."""
        m = self.material
        previous = self.phi
        half = 0.5 * m.mushy_range
        solid_max = m.density * m.specific_heat * (m.ablation_temperature - half)
        gone_min = (
            m.density * m.specific_heat * (m.ablation_temperature + half) + m.latent_heat
        )

        solid = enthalpy <= solid_max
        gone = enthalpy >= gone_min
        mushy = ~(solid | gone)

        self.temperature = np.empty_like(enthalpy)
        self.phi = np.empty_like(enthalpy)
        self.temperature[solid] = enthalpy[solid] / (m.density * m.specific_heat)
        self.phi[solid] = 1.0
        self.temperature[gone] = (enthalpy[gone] - m.latent_heat) / (
            m.density * m.specific_heat
        )
        self.phi[gone] = 0.0
        if np.any(mushy):
            fraction = (enthalpy[mushy] - solid_max) / (gone_min - solid_max)
            self.temperature[mushy] = (
                m.ablation_temperature - half + fraction * m.mushy_range
            )
            self.phi[mushy] = 1.0 - fraction

        # Ablation is one-way. Enforced rather than assumed, because the
        # inversion cannot tell a cell that ablated from one that was never
        # there: both have zero solid fraction, but the "gone" branch assumes
        # the cell is at least at the ablation temperature, which cold material
        # is not. On the previous voxel grid that turned 838 kg into 1730 kg
        # while the body was supposedly losing mass.
        self.phi = np.clip(np.minimum(self.phi, previous), 0.0, 1.0)

    def save(self, path: str | Path) -> Path:
        """Write the current field to ``.npz`` — no optional dependency needed."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        state = self.state()
        np.savez_compressed(
            target,
            temperature=self.temperature,
            phi=self.phi,
            widths=self._widths,
            origin=self.columns.origin,
            normal=self.columns.normal,
            area=self.columns.area,
            recession=state.recession,
            time=self.time,
            mass=state.mass,
            centroid=state.centroid,
        )
        return target


def _solve_tridiagonal(
    lower: _FloatArray, diagonal: _FloatArray, upper: _FloatArray, right: _FloatArray
) -> _FloatArray:
    """Thomas algorithm, vectorised across rows.

    One tridiagonal system per column, all solved at once. Looping over a few
    thousand columns in Python would cost more than the arithmetic it wraps;
    sweeping every column's cell ``i`` together turns the whole body into two
    passes over an ``(n_columns, n_cells)`` array.

    No pivoting, and none needed: backward Euler on a diffusion operator gives
    a diagonally dominant matrix by construction — the diagonal carries the
    cell's heat capacity *plus* both couplings, and the off-diagonals carry one
    each — so the elimination cannot encounter a small pivot.
    """
    n = diagonal.shape[1]
    c = np.empty_like(diagonal)
    d = np.empty_like(right)

    c[:, 0] = upper[:, 0] / diagonal[:, 0]
    d[:, 0] = right[:, 0] / diagonal[:, 0]
    for i in range(1, n):
        denominator = diagonal[:, i] - lower[:, i] * c[:, i - 1]
        c[:, i] = upper[:, i] / denominator
        d[:, i] = (right[:, i] - lower[:, i] * d[:, i - 1]) / denominator

    solution = np.empty_like(right)
    solution[:, -1] = d[:, -1]
    for i in range(n - 2, -1, -1):
        solution[:, i] = d[:, i] - c[:, i] * solution[:, i + 1]
    return solution
