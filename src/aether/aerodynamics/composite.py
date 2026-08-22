"""One coefficient source from several methods, each used where it is valid.

No single method covers a launch-to-reentry envelope. Panel impact theory is
a hypersonic method with no subsonic meaning. Euler CFD covers the transonic
bucket and costs minutes a point instead of milliseconds. Neither contains a
boundary layer. Neither is applicable at all above 90 km, where there is no
continuum. The classical answer is a **patched method**: run each theory in
its own region and splice, and this module is that splice made explicit
rather than left to whoever assembles the table.

The regions, and what decides them:

======================  ==========================  =========================
Region                  Selector                    Method
======================  ==========================  =========================
Rarefied                :math:`Kn > 10`             Schaaf–Chambré
Transitional            :math:`10^{-3} < Kn < 10`   bridged
Sub/transonic           :math:`M < M_{\\rm splice}`   Euler CFD
Supersonic/hypersonic   :math:`M > M_{\\rm splice}`   panel + real gas
All continuum           always                      + skin friction
======================  ==========================  =========================

Two splices, handled differently on purpose
-------------------------------------------

The **Knudsen** splice is a genuine physical transition with no theory in
the middle, so it is bridged by an empirical function and the result is
labelled as interpolated.

The **Mach** splice is not a physical transition at all — it is the boundary
between two approximations to the same physics — so it is blended over a
narrow band with a :math:`C^2` smoothstep. That blend is cosmetic in the
sense that neither side is more right in the band, and it is not optional in
the sense that a step discontinuity in :math:`C_A` at Mach 1.2 would be
integrated by a trajectory and would show up as a kink in the flight path.
Where the two methods *disagree* across the band is the useful diagnostic,
and :meth:`PatchedSolver.splice_discrepancy` reports it rather than hiding it
under the blend.

Altitude
--------

Skin friction depends on Reynolds number and rarefaction on Knudsen number,
so a coefficient here is a function of altitude as well as Mach and
incidence. Rather than add a third table axis, ``altitude`` may be a callable
:math:`z(M)` — an **altitude schedule** — which is how launch-vehicle
coefficient tables are actually built: along the trajectory the vehicle will
fly, not over a rectangle it will not.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from aether.aerodynamics.friction import BoundaryLayer
from aether.aerodynamics.rarefied import (
    CONTINUUM_KNUDSEN,
    FREE_MOLECULAR_KNUDSEN,
    FreeMolecularSolver,
    sine_squared_bridge,
)
from aether.aerodynamics.realgas import EquilibriumAir
from aether.aerodynamics.tables import Coefficients
from aether.atmosphere.model import Atmosphere, earth_atmosphere
from aether.blending import smoothstep

__all__ = [
    "AltitudeSchedule",
    "PatchedSolver",
    "SkinFrictionModel",
    "SolverAtCondition",
    "meridian_running_length",
]

_FloatArray = NDArray[np.float64]

#: Either a fixed altitude (m) or a schedule ``z(mach)``.
AltitudeSchedule = float | Callable[[float], float]


class SolverAtCondition(Protocol):
    """A solver that only needs Mach and incidence."""

    @property
    def name(self) -> str: ...

    def solve(self, mach: float, alpha: float) -> Coefficients:  # pragma: no cover
        ...


def meridian_running_length(
    axial: _FloatArray, radial: _FloatArray, n_bins: int = 400
) -> _FloatArray:
    """Surface arc length from the nose to each panel, along the meridian.

    The reference-temperature method wants a Reynolds number formed on the
    distance the boundary layer has run, which for a body of revolution at
    small incidence is the meridian arc length — not the axial station, and
    on a 25-degree nose cone those differ by ten per cent.

    Computed by binning the panels axially, taking the mean radius in each
    bin, integrating :math:`\\sqrt{1 + (\\mathrm{d}r/\\mathrm{d}x)^2}` along
    the resulting profile and interpolating back. Binning rather than sorting
    because a surface mesh has many panels at the same station and their
    radii differ; the mean is the meridian.
    """
    x = np.asarray(axial, dtype=np.float64)
    r = np.asarray(radial, dtype=np.float64)
    if x.shape != r.shape:
        msg = f"axial and radial arrays must match, got {x.shape} and {r.shape}"
        raise ValueError(msg)
    low, high = float(np.min(x)), float(np.max(x))
    if high <= low:
        return np.zeros_like(x)

    edges = np.linspace(low, high, int(n_bins) + 1)
    index = np.clip(np.digitize(x, edges) - 1, 0, int(n_bins) - 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    totals = np.bincount(index, weights=r, minlength=int(n_bins))
    counts = np.bincount(index, minlength=int(n_bins))
    occupied = counts > 0
    profile_x = centres[occupied]
    profile_r = totals[occupied] / counts[occupied]
    if profile_x.size < 2:
        return np.abs(x - low)

    segment = np.hypot(np.diff(profile_x), np.diff(profile_r))
    arc = np.concatenate([[0.0], np.cumsum(segment)])
    return np.asarray(np.interp(x, profile_x, arc))


@dataclass
class SkinFrictionModel:
    """Integrated skin friction over a panelised mesh.

    Local edge conditions come from the local pressure: given a panel's
    :math:`C_p` from whichever inviscid method produced it,

    .. math::

        M_e^2 = \\frac{2}{\\gamma-1}\\left[
        \\left(1 + \\tfrac{\\gamma-1}{2}M_\\infty^2\\right)
        \\left(\\frac{p_\\infty}{p_e}\\right)^{(\\gamma-1)/\\gamma} - 1\\right]

    which assumes the boundary-layer edge was reached **isentropically from
    the freestream**. That is right for the attached weak shocks of a slender
    body and wrong immediately behind a blunt nose, where the streamline has
    crossed a strong shock and lost stagnation pressure. On this vehicle the
    nose is 0.6 % of the wetted area, so the entropy-layer error is confined
    to where there is nothing to be wrong about; on a blunt capsule it would
    not be.
    """

    mesh: Any
    reference_area: float
    reference_length: float
    boundary_layer: BoundaryLayer = field(default_factory=BoundaryLayer)
    reference_point: _FloatArray | None = None
    gamma: float = 1.4
    name: str = "skin friction"
    #: Floor on edge pressure as a fraction of freestream. The leeward branch
    #: of the pressure closure reaches the vacuum limit exactly, at which the
    #: edge Mach number is infinite; the floor keeps the arithmetic finite,
    #: and a surface at vacuum has no shear worth computing anyway.
    pressure_floor: float = 1.0e-3

    def __post_init__(self) -> None:
        self._model = self.mesh.panel_model(self.reference_point)
        centroids = np.asarray(self._model.centroids)
        radial = np.hypot(centroids[:, 1], centroids[:, 2])
        axial = centroids[:, 0]
        # Orient nose-first: the nose is the end with the smaller mean radius.
        midpoint = 0.5 * (float(axial.min()) + float(axial.max()))
        forward = radial[axial < midpoint].mean() if np.any(axial < midpoint) else 0.0
        aft = radial[axial >= midpoint].mean() if np.any(axial >= midpoint) else 0.0
        oriented = axial if forward <= aft else -axial
        self._running_length = meridian_running_length(oriented, radial)

    def solve(
        self,
        mach: float,
        alpha: float,
        pressure_coefficient: _FloatArray,
        temperature: float,
        pressure: float,
        density: float,
    ) -> Coefficients:
        """Friction contribution at one flight condition.

        ``pressure_coefficient`` is the inviscid distribution over the same
        panels, which is what couples the two: friction on a compressed
        windward panel is not friction on an expanded leeward one.
        """
        model = self._model
        m_inf = float(mach)
        g = self.gamma
        gas_constant = float(pressure) / (float(density) * float(temperature))

        cp = np.asarray(pressure_coefficient, dtype=np.float64)
        edge_pressure = np.maximum(
            pressure * (1.0 + 0.5 * g * m_inf**2 * cp),
            self.pressure_floor * pressure,
        )
        stagnation = 1.0 + 0.5 * (g - 1.0) * m_inf**2
        edge_mach = np.sqrt(
            np.maximum(
                2.0
                / (g - 1.0)
                * (stagnation * (pressure / edge_pressure) ** ((g - 1.0) / g) - 1.0),
                0.0,
            )
        )
        edge_temperature = temperature * stagnation / (
            1.0 + 0.5 * (g - 1.0) * edge_mach**2
        )
        edge_density = edge_pressure / (gas_constant * edge_temperature)
        edge_speed = edge_mach * np.sqrt(g * gas_constant * edge_temperature)

        _, _, shear = self.boundary_layer.skin_friction(
            edge_temperature,
            edge_mach,
            edge_density,
            edge_speed,
            edge_pressure,
            self._running_length,
        )

        normals = np.asarray(model.normals)
        areas = np.asarray(model.areas)
        v_hat = model.velocity_direction(float(alpha))
        tangential = v_hat[np.newaxis, :] - (normals @ v_hat)[:, np.newaxis] * normals
        magnitude = np.linalg.norm(tangential, axis=1)
        direction = tangential / np.where(magnitude > 1e-12, magnitude, 1.0)[:, np.newaxis]

        panel_force = (shear * areas)[:, np.newaxis] * direction
        force = np.sum(panel_force, axis=0)
        arms = np.asarray(model.centroids) - np.asarray(model.reference_point)
        moment = np.sum(np.cross(arms, panel_force), axis=0)

        scale = 0.5 * float(density) * (m_inf * np.sqrt(g * gas_constant * temperature)) ** 2
        scale *= self.reference_area
        return Coefficients(
            axial=float(force[0] / scale),
            normal=float(force[2] / scale),
            pitching_moment=float(moment[1] / (scale * self.reference_length)),
        )


def _add(*parts: Coefficients) -> Coefficients:
    return Coefficients(
        axial=float(sum(p.axial for p in parts)),
        normal=float(sum(p.normal for p in parts)),
        pitching_moment=float(sum(p.pitching_moment for p in parts)),
    )


def _blend(low: Coefficients, high: Coefficients, weight: float) -> Coefficients:
    w = float(weight)
    return Coefficients(
        axial=(1.0 - w) * low.axial + w * high.axial,
        normal=(1.0 - w) * low.normal + w * high.normal,
        pitching_moment=(1.0 - w) * low.pitching_moment + w * high.pitching_moment,
    )


@dataclass
class PatchedSolver:
    """The full-envelope coefficient source.

    Attributes
    ----------
    panel:
        The hypersonic method. Required.
    euler:
        The sub/transonic method. Optional — without it the table simply
        stops at ``splice`` and says so, which is a better outcome than
        extrapolating impact theory into a regime where it is not a model of
        anything.
    friction:
        Skin-friction model. Optional, and its absence is visible rather than
        silent: :meth:`components` reports a zero for it, which is a
        different statement from not reporting it. Worth 5 to 7 % of axial
        force on this vehicle at supersonic speeds.
    viscous_interaction_band:
        The band of :math:`\\bar\\chi = M^3/\\sqrt{Re_L}` over which the
        continuum friction model is faded out. See
        :meth:`PatchedSolver.diagnostics`.
    free_molecular:
        Rarefied method. Without it the solver refuses to answer above the
        continuum Knudsen limit rather than returning a continuum answer
        there.
    real_gas:
        Equilibrium-air model supplying :math:`C_{p,\\max}` above
        ``real_gas_mach``. Below that the perfect-gas Rayleigh–Pitot value is
        within a per cent and the equilibrium solve is not worth its cost.
    altitude:
        A fixed altitude (m) or a schedule ``z(mach)``.
    """

    panel: Any
    reference_area: float
    reference_length: float
    altitude: AltitudeSchedule = 20.0e3
    euler: Any | None = None
    friction: SkinFrictionModel | None = None
    free_molecular: FreeMolecularSolver | None = None
    real_gas: EquilibriumAir | None = None
    atmosphere: Atmosphere = field(default_factory=earth_atmosphere)
    splice: float = 1.6
    """Mach number the CFD hands over to the panel method."""
    splice_width: float = 0.4
    """Half-width of the blend band around ``splice``."""
    real_gas_mach: float = 8.0
    viscous_interaction_band: tuple[float, float] = (0.5, 3.0)
    name: str = "patched"

    def __post_init__(self) -> None:
        if self.splice_width <= 0.0:
            msg = f"splice_width must be > 0, got {self.splice_width}"
            raise ValueError(msg)
        if self.splice - self.splice_width <= getattr(self.panel, "absolute_floor", 1.05):
            msg = (
                f"the blend band opens at Mach "
                f"{self.splice - self.splice_width:g}, at or below the panel "
                f"method's floor; raise `splice` or narrow `splice_width`"
            )
            raise ValueError(msg)

    def altitude_at(self, mach: float) -> float:
        """Resolve the altitude schedule at a Mach number (m)."""
        if callable(self.altitude):
            return float(self.altitude(float(mach)))
        return float(self.altitude)

    def knudsen(self, mach: float) -> float:
        """:math:`Kn = \\lambda/L` at this Mach number's altitude."""
        state = self.atmosphere.state(self.altitude_at(mach))
        return float(state.mean_free_path) / self.reference_length

    def diagnostics(self, mach: float) -> dict[str, float]:
        """Similarity parameters and the regime weights they imply.

        ``viscous_interaction`` is
        :math:`\\bar\\chi = M_\\infty^3/\\sqrt{Re_L}`, the parameter that says
        whether a boundary layer is thin. A hypersonic laminar layer grows as
        :math:`\\delta/L \\sim M^2/\\sqrt{Re}`, so :math:`\\bar\\chi` of order
        one means the layer is as thick as the body and the boundary-layer
        approximation — which is *all* of
        :mod:`aether.aerodynamics.friction` — has stopped being an
        approximation to anything.

        ``friction_validity`` fades from 1 to 0 across
        :attr:`viscous_interaction_band`. Without it the reference-temperature
        correlation returns a skin-friction contribution of **6.0** to
        :math:`C_A` at Mach 20 and 120 km, against an inviscid 0.93, which is
        not a large correction but a meaningless number: at that altitude
        :math:`Re_L` is of order :math:`10^3` and there is no boundary layer
        to correlate.

        The fade is a **validity gate, not a model**. What actually happens
        between :math:`\\bar\\chi \\sim 1` and free-molecular flow is
        merged-layer, strong-viscous-interaction hypersonics, and nothing in
        this package computes it — the Knudsen bridge carries the answer
        across that band and the bridge is an interpolation. Coefficients
        reported where ``friction_validity`` is between 0 and 1 and
        ``bridge`` is small are the least trustworthy this pipeline produces,
        and that is what this method exists to make visible.
        """
        m = float(mach)
        altitude = self.altitude_at(m)
        stream = self.atmosphere.state(altitude)
        speed = m * float(stream.speed_of_sound)
        reynolds = (
            float(stream.density) * speed * self.reference_length / float(stream.viscosity)
        )
        chi = m**3 / np.sqrt(max(reynolds, 1.0))
        low, high = self.viscous_interaction_band
        knudsen = float(stream.mean_free_path) / self.reference_length
        return {
            "mach": m,
            "altitude": altitude,
            "reynolds": reynolds,
            "knudsen": knudsen,
            "viscous_interaction": float(chi),
            "friction_validity": float(1.0 - smoothstep((chi - low) / (high - low))),
            "bridge": float(sine_squared_bridge(knudsen)),
        }

    def cp_max(self, mach: float) -> float | None:
        """Equilibrium :math:`C_{p,\\max}`, or ``None`` to use the perfect-gas one."""
        if self.real_gas is None or float(mach) < self.real_gas_mach:
            return None
        state = self.atmosphere.state(self.altitude_at(mach))
        speed = float(mach) * float(state.speed_of_sound)
        return self.real_gas.cp_max(
            float(state.temperature), float(state.pressure), speed
        )

    # -- the pieces --------------------------------------------------------

    def continuum(self, mach: float, alpha: float) -> dict[str, Coefficients]:
        """Inviscid and viscous parts in the continuum regime, kept separate."""
        m = float(mach)
        parts: dict[str, Coefficients] = {}

        lower = self.splice - self.splice_width
        upper = self.splice + self.splice_width
        cp_max = self.cp_max(m)

        # The band edge is inclusive, and it has to be tested as such rather than
        # trusted to arithmetic: the default splice and width are 1.6 and 0.4,
        # whose difference is 1.2000000000000002, so a grid built to start at
        # exactly 1.2 -- which is what SweepGrid.default_mach(minimum=1.2) gives,
        # and the two are meant to meet -- fell outside its own lower edge by
        # 2e-16 and reported that nothing covered it.
        edge_tolerance = 1e-9 * max(abs(lower), 1.0)
        inviscid = self._panel(m, alpha, cp_max) if m >= lower - edge_tolerance else None
        euler = (
            self.euler.solve(m, alpha)
            if m <= upper and self.euler is not None
            else None
        )

        if inviscid is None and euler is None:
            msg = (
                f"nothing covers Mach {m:g}: it is below the panel method's "
                f"blend band and no Euler solver was supplied. Attach one, or "
                f"trim the Mach grid with SweepGrid.default_mach(minimum=...)."
            )
            raise ValueError(msg)
        if inviscid is None:
            assert euler is not None
            parts["inviscid"] = euler
        elif euler is None:
            parts["inviscid"] = inviscid
        else:
            weight = float(smoothstep((m - lower) / (upper - lower)))
            parts["inviscid"] = _blend(euler, inviscid, weight)

        if self.friction is not None:
            state = self.atmosphere.state(self.altitude_at(m))
            raw = self.friction.solve(
                m,
                alpha,
                self._panel_pressures(m, alpha, cp_max),
                float(state.temperature),
                float(state.pressure),
                float(state.density),
            )
            validity = self.diagnostics(m)["friction_validity"]
            parts["friction"] = Coefficients(
                axial=raw.axial * validity,
                normal=raw.normal * validity,
                pitching_moment=raw.pitching_moment * validity,
            )
        else:
            parts["friction"] = Coefficients(0.0, 0.0, 0.0)
        return parts

    def components(self, mach: float, alpha: float) -> dict[str, Coefficients]:
        """Every contribution separately — what a coefficient is made of.

        A single number cannot be argued with. A breakdown into inviscid,
        friction and rarefied parts, with the bridge weight alongside, can be
        checked term by term against expectation, and that is the difference
        between a table that is trusted and one that is believed.
        """
        m = float(mach)
        knudsen = self.knudsen(m)
        weight = float(sine_squared_bridge(knudsen))

        parts: dict[str, Coefficients] = {}
        if weight < 1.0:
            parts.update(self.continuum(m, alpha))
        else:
            parts["inviscid"] = Coefficients(0.0, 0.0, 0.0)
            parts["friction"] = Coefficients(0.0, 0.0, 0.0)

        if weight > 0.0:
            if self.free_molecular is None:
                msg = (
                    f"Mach {m:g} at {self.altitude_at(m) / 1e3:.1f} km gives "
                    f"Kn = {knudsen:.3g}, which is not continuum flow, and no "
                    f"free-molecular solver was supplied. Attach one or keep "
                    f"the altitude schedule below Kn = {CONTINUUM_KNUDSEN:g}."
                )
                raise ValueError(msg)
            state = self.atmosphere.state(self.altitude_at(m))
            parts["free_molecular"] = self.free_molecular.solve(
                m, alpha, temperature=float(state.temperature)
            )
        else:
            parts["free_molecular"] = Coefficients(0.0, 0.0, 0.0)
        return parts

    def solve(self, mach: float, alpha: float) -> Coefficients:
        """Total coefficients at a Mach number and incidence."""
        parts = self.components(mach, alpha)
        weight = float(sine_squared_bridge(self.knudsen(float(mach))))
        continuum = _add(parts["inviscid"], parts["friction"])
        return _blend(continuum, parts["free_molecular"], weight)

    def splice_discrepancy(self, mach: float, alpha: float = 0.0) -> dict[str, float]:
        """How far apart the two continuum methods are, in the blend band.

        Both are evaluated at the same point and the difference reported. In
        the band neither is more right than the other, so the gap is a direct
        measure of how much the answer there depends on a choice with no
        physical content — which is exactly the number a reader should see
        before trusting a spliced table.
        """
        if self.euler is None:
            msg = "no Euler solver attached; there is nothing to compare against"
            raise ValueError(msg)
        panel = self._panel(float(mach), float(alpha), self.cp_max(float(mach)))
        euler = self.euler.solve(float(mach), float(alpha))
        gap = panel.axial - euler.axial
        level = 0.5 * abs(panel.axial + euler.axial)
        return {
            "mach": float(mach),
            "panel_axial": float(panel.axial),
            "euler_axial": float(euler.axial),
            "difference": float(gap),
            "relative": float(gap / level) if level > 0.0 else float("nan"),
        }

    # -- panel-method plumbing --------------------------------------------

    def _panel(
        self, mach: float, alpha: float, cp_max: float | None
    ) -> Coefficients:
        if cp_max is None:
            return cast(Coefficients, self.panel.solve(mach, alpha))
        return cast(Coefficients, self.panel.solve(mach, alpha, cp_max=cp_max))

    def _panel_pressures(
        self, mach: float, alpha: float, cp_max: float | None
    ) -> _FloatArray:
        """The panel pressure distribution the friction model needs.

        Taken from the panel method even inside the blend band where the
        inviscid force comes partly from CFD, because the CFD is
        axisymmetric and has no per-panel distribution at incidence. The
        approximation is a distribution, not a force, and skin friction
        depends on the edge pressure only through :math:`\\rho^*` — a
        square-root sensitivity.
        """
        from aether.aerodynamics.closure import blended_pressure_coefficient

        model = self.panel._model
        delta = model.incidences(float(alpha))
        floor = getattr(self.panel, "absolute_floor", 1.05)
        effective = max(float(mach), floor + 0.05)
        return blended_pressure_coefficient(delta, effective, cp_max=cp_max)


def free_molecular_limits() -> tuple[float, float]:
    """The Knudsen band the bridge spans — re-exported for legibility."""
    return CONTINUUM_KNUDSEN, FREE_MOLECULAR_KNUDSEN
