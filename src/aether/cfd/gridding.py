r"""What a hypersonic grid has to resolve, before one is built.

Every sizing rule here is taken from the LAURA user's manual
:cite:`cheatwood1996laura` section 11.1.1, which specifies its grid adaption by
four quantities -- ``recell``, ``fstr``, ``ep0`` and ``fsh`` -- and ships
defaults in ``algnshk_vars.strt``. They are reproduced as the defaults below so
that a departure from NASA practice is visible as an argument rather than
buried as a constant.

The reason this module exists is that **none of these were being applied**. The
pipeline sized its wall cells from a hand-typed length, put zero cells inside
the shock standoff, and refined isotropically -- which is why nothing converged
→ ``the-bow-shock-never-settles``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "SUTHERLAND",
    "GridRequirement",
    "grid_requirement",
    "growth_for",
    "outer_boundary_distance",
    "stagnation_pressure",
    "wall_spacing",
]

#: Sutherland's law for air: reference viscosity, reference and Sutherland
#: temperatures, matching the constants SU2 prints for ``VISCOSITY_MODEL=
#: SUTHERLAND`` so a spacing computed here is the spacing the solver sees.
SUTHERLAND: tuple[float, float, float] = (1.716e-5, 273.15, 110.4)


def stagnation_pressure(mach: float, pressure: float, gamma: float = 1.4) -> float:
    """Pitot pressure behind a normal shock -- the Rayleigh supersonic formula.

    This is the pressure the wall sees at the stagnation point, which is where
    the cell Reynolds number requirement is tightest and therefore the right
    number to size a grid against.
    """
    if mach <= 1.0:
        return pressure * (1.0 + 0.5 * (gamma - 1.0) * mach**2) ** (gamma / (gamma - 1.0))
    m2 = mach * mach
    return pressure * (
        ((gamma + 1.0) ** 2 * m2) / (4.0 * gamma * m2 - 2.0 * (gamma - 1.0))
    ) ** (gamma / (gamma - 1.0)) * ((1.0 - gamma + 2.0 * gamma * m2) / (gamma + 1.0))


def _viscosity(temperature: float) -> float:
    reference, t_ref, t_s = SUTHERLAND
    return reference * (temperature / t_ref) ** 1.5 * (t_ref + t_s) / (temperature + t_s)


def wall_spacing(
    wall_pressure: float,
    wall_temperature: float,
    cell_reynolds: float = 1.0,
    gas_constant: float = 287.058,
    gamma: float = 1.4,
) -> float:
    r"""First cell height at the wall, from LAURA's cell Reynolds number.

    .. math:: \mathrm{Re}_{\text{cell}} = \frac{\rho\, a\, \Delta s}{\mu}

    evaluated at **wall** conditions, with ``recell = 1`` the shipped default.

    This is the criterion for **heat transfer**, and it is not $y^+$. LAURA
    reports both in ``grid.out`` -- ``Recell w`` always, ``y+ w`` only when
    ``nturb != 0`` -- which is the distinction: $y^+$ governs a turbulent
    boundary layer, the cell Reynolds number governs laminar hypersonic
    heating. A pipeline running laminar and sizing by $y^+$ is answering the
    wrong question, and one sizing by neither is not answering it at all.

    ``wall_pressure`` at the stagnation point is the post-shock stagnation
    pressure, which is where the requirement is tightest and therefore the
    right number to size a grid against.
    """
    if wall_temperature <= 0.0 or wall_pressure <= 0.0:
        msg = "wall pressure and temperature must be positive"
        raise ValueError(msg)
    density = wall_pressure / (gas_constant * wall_temperature)
    sound_speed = math.sqrt(gamma * gas_constant * wall_temperature)
    return cell_reynolds * _viscosity(wall_temperature) / (density * sound_speed)


def outer_boundary_distance(shock_distance: float, shock_fraction: float = 0.8) -> float:
    """Where the inflow boundary goes, from LAURA's ``fsh``.

    LAURA adjusts the outer boundary so the captured shock sits at a constant
    fraction ``fsh`` of the body-to-inflow line; the shipped default is 0.8, so
    the boundary stands **1.25 times** the shock distance off the body. Placing
    it much further wastes cells on freestream; much closer and the captured
    shock runs into the boundary.
    """
    if not 0.0 < shock_fraction < 1.0:
        msg = f"shock fraction must lie in (0, 1), got {shock_fraction:g}"
        raise ValueError(msg)
    return shock_distance / shock_fraction


def growth_for(first: float, total: float, layers: int) -> float:
    r"""The geometric ratio that fits exactly ``layers`` cells into ``total``.

    Solves :math:`\sum_{k=0}^{n-1} \Delta_1 g^k = L` for :math:`g`, by bisection
    because it has no closed form.

    This exists so a refinement level can set the **layer count** and let the
    grading follow, rather than the other way round. Choosing the growth ratio
    and taking whatever layer count falls out refines the normal direction by
    1.28x when the surface refines by 1.41x -- measured -- and a grid study
    whose refinement ratio differs by axis has no single :math:`r` to compute an
    observed order against.
    """
    if layers < 2 or first <= 0.0 or total <= first:
        msg = f"cannot fit {layers} cells of {first:g} into {total:g}"
        raise ValueError(msg)
    # Uniform spacing is the g -> 1 limit; anything thicker than that needs g > 1.
    if abs(first * layers - total) < 1e-12 * total:
        return 1.0
    low, high = (1.0 + 1e-12, 10.0) if first * layers < total else (1e-6, 1.0 - 1e-12)

    def span(g: float) -> float:
        return first * layers if abs(g - 1.0) < 1e-15 else first * (g**layers - 1.0) / (g - 1.0)

    for _ in range(200):
        mid = 0.5 * (low + high)
        if span(mid) < total:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


@dataclass(frozen=True)
class GridRequirement:
    """What the flow condition demands, against what a mesh actually has."""

    wall_cell: float
    """First cell height at the wall, metres."""

    shock_standoff: float
    """Distance from body to shock on the stagnation streamline, metres."""

    outer_boundary: float
    """Distance from body to the inflow boundary, metres."""

    cells_across_layer: int
    """Cells from wall to outer boundary, to span both scales at ``fstr``."""

    @property
    def scale_ratio(self) -> float:
        """Shock layer divided by wall cell -- the span one column must cover."""
        return self.shock_standoff / self.wall_cell

    def shortfall(self, actual_wall_cell: float) -> float:
        """How many times too coarse a mesh's wall cell is. Below 1 is adequate."""
        return actual_wall_cell / self.wall_cell


def grid_requirement(
    shock_standoff: float,
    wall_pressure: float,
    wall_temperature: float,
    cell_reynolds: float = 1.0,
    stretch_fraction: float = 0.5,
    shock_fraction: float = 0.8,
    growth: float = 1.15,
) -> GridRequirement:
    """Size a wall-normal column for one flow condition.

    ``stretch_fraction`` is LAURA's ``fstr``, the fraction of the column's
    cells spent resolving the boundary layer; the shipped default is 0.5, so
    half the cells go to the near-wall stretch region and the rest are spread
    across the remainder of the shock layer.

    ``growth`` is not LAURA's -- LAURA derives its stretching from ``fstr`` and
    ``betagrd`` -- and is the geometric ratio this pipeline's extruder will
    use. 1.15 is conservative; anything above about 1.25 degrades second-order
    accuracy on a stretched grid.
    """
    wall = wall_spacing(wall_pressure, wall_temperature, cell_reynolds)
    outer = outer_boundary_distance(shock_standoff, shock_fraction)
    # Cells to climb from `wall` to the outer boundary at a fixed growth ratio,
    # then inflated by fstr because only that fraction is spent on the climb.
    span = outer / wall
    climbing = math.log1p(span * (growth - 1.0)) / math.log(growth)
    return GridRequirement(
        wall_cell=wall,
        shock_standoff=shock_standoff,
        outer_boundary=outer,
        cells_across_layer=max(2, math.ceil(climbing / max(stretch_fraction, 1e-6))),
    )
