"""Aerodynamic tables for an arbitrary 3-D model, end to end.

A mesh in, a table out. :func:`build_solver` assembles the multi-regime solver
for one body — panel impact theory patched with skin friction under a
radiative-equilibrium wall, an equilibrium-air stagnation pressure coefficient,
and a free-molecular limit bridged across the transition regime — and
:func:`build_tables` sweeps it over a Mach/incidence grid, checkpointing so a
long run can be stopped and resumed. :func:`save_tables` and :func:`load_tables`
persist the result next to the model it describes.

Nothing here knows what the body is *for*. A :class:`Configuration` is a mesh
and its reference quantities; a table is coefficients on a grid. The sequence a
vehicle wears its configurations in — which one flies first, what a separation
means to the table in use — is a property of a mission rather than of a body,
and lives with the layer that owns missions.

Reference quantities
--------------------

Held **common across every configuration of one vehicle**, not derived per
body. Otherwise the axial coefficient of a full stack and of its payload are
divided by different numbers and cannot be compared or blended, which is the
entire reason for building them together.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aether.aerodynamics.composite import PatchedSolver, SkinFrictionModel
from aether.aerodynamics.friction import BoundaryLayer, RadiativeEquilibriumWall
from aether.aerodynamics.rarefied import FreeMolecularSolver
from aether.aerodynamics.realgas import EquilibriumAir
from aether.aerodynamics.tables import AeroTable, PanelSolver, SweepGrid, SweepRun

__all__ = [
    "Configuration",
    "TrimAtGridEdge",
    "ballistic_coefficient",
    "build_solver",
    "build_tables",
    "condition_lifting_body",
    "lift_to_drag",
    "load_mesh",
    "load_tables",
    "planform_area",
    "reentry_alpha_grid",
    "save_tables",
    "trim_for_max_lift_to_drag",
    "wind_axes",
]

_FloatArray = NDArray[np.float64]


class TrimAtGridEdge(UserWarning):
    """The incidence that maximised :math:`L/D` was the edge of the swept grid.

    A warning rather than a silent result: an extremum at an endpoint means the
    search ran out of grid before the curve turned over, so the true trim is
    outside the range and the ratio reported is an underestimate of it.
    """


def reentry_alpha_grid(limit_deg: float = 40.0, step_deg: float = 2.0) -> _FloatArray:
    """Incidence grid for a re-entry or glide body (rad).

    Wider than a launch vehicle's. A booster flies at essentially zero incidence
    and 12° covers its dispersion; a re-entry body or glider trims *deliberately*
    at high incidence — a hypersonic :math:`L/D` peaks well beyond 12°, so a grid
    that stops there cannot find the trim and would report the maximum at its own
    edge.
    """
    return np.deg2rad(np.arange(0.0, limit_deg + 0.5 * step_deg, step_deg))


def load_mesh(path: str | Path) -> Any:
    """Load an STL and put it in body axes (nose along the +x axis)."""
    from aether.geometry.mesh import load_stl

    return load_stl(Path(path)).to_body_axes()


def condition_lifting_body(
    mesh: Any,
    *,
    length_m: float | None = None,
    roll_span_to_y: bool = True,
    centre_cross_section: bool = True,
) -> Any:
    """Put a lifting-body mesh into the attitude and units the solver assumes.

    :meth:`~aether.geometry.mesh.VehicleMesh.to_body_axes` only aligns the
    **long** axis with +x. For a body of revolution that is the whole story, but a
    waverider is neither round nor symmetric about its own axis, and two things it
    leaves unresolved change the answer rather than the picture:

    **Roll.** The panel model pitches in the :math:`x`–:math:`z` plane
    (:math:`\\hat v_B = (\\cos\\alpha, 0, \\sin\\alpha)`), so the span must lie on
    :math:`y` and the lift direction on :math:`z`. A mesh authored with the span on
    :math:`z` is 90° out, and sweeping incidence then pitches it *edge-on* — it
    reports the lift of a knife rather than of a wing. The wider cross-dimension is
    the span by definition, so it is rolled onto :math:`y`.

    **Units.** STL carries no units. A mesh authored in millimetres loads as a
    3.6 **km** vehicle, whose frontal area is out by :math:`10^6` and whose
    coefficients are therefore meaningless. ``length_m`` states the published
    length and scales the body to it.

    Centring the cross-section on the axis leaves forces unchanged (a translation
    changes no normal and no area) but puts moments on a sensible reference.
    """
    from aether.geometry.mesh import VehicleMesh

    body = mesh.to_body_axes() if hasattr(mesh, "to_body_axes") else mesh
    vertices = np.asarray(body.vertices, dtype=np.float64).copy()

    if roll_span_to_y and np.ptp(vertices[:, 2]) > np.ptp(vertices[:, 1]):
        # Roll -90 deg about x: (y, z) -> (z, -y). Right-handed, and it carries
        # the wider dimension onto y where the span belongs.
        y = vertices[:, 1].copy()
        vertices[:, 1] = vertices[:, 2]
        vertices[:, 2] = -y

    if centre_cross_section:
        for axis in (1, 2):
            vertices[:, axis] -= 0.5 * (vertices[:, axis].max() + vertices[:, axis].min())

    conditioned = VehicleMesh(
        vertices=vertices, faces=np.asarray(body.faces).copy(), name=body.name
    )
    if length_m is not None:
        span = float(np.ptp(np.asarray(conditioned.vertices)[:, 0]))
        if span <= 0.0:
            raise ValueError("mesh has no extent along its body axis")
        factor = float(length_m) / span
        conditioned = conditioned.scaled(axial=factor, radial=factor)
    return conditioned


def planform_area(mesh: Any) -> float:
    r"""Planform area (m²): the body projected onto its own :math:`x`–:math:`y` plane.

    The reference area a lifting body's coefficients are conventionally quoted on,
    where a missile uses its frontal area. On the bundled waverider the two differ
    by a factor of six (0.658 m² frontal against 3.917 m² planform).

    That factor rescales the **coefficients** and nothing physical: :math:`C_D`
    falls by exactly the ratio the area rises, so the drag area :math:`C_D S` —
    which is what a force actually depends on — and every quantity derived from
    it, :math:`L/D` and :math:`\beta = m/(C_D S)` among them, are unchanged. The
    convention therefore matters for *comparing published coefficients*, and not
    for the dynamics. Both are offered so a table can be quoted the way its class
    conventionally is, without the choice being able to move a trajectory.

    Computed as half the summed magnitude of each facet's :math:`z` normal
    component: a closed surface projects both its upper and lower skins onto the
    same shadow, so the sum double-counts it exactly once.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return float(0.5 * np.abs(normals[:, 2]).sum() / 2.0)


@dataclass(frozen=True)
class Configuration:
    """One aerodynamic body: a mesh, its references, and what it represents.

    ``name`` identifies it in checkpoints and saved tables. ``reference_area`` and
    ``reference_length`` are the non-dimensionalisation — held **common across the
    whole vehicle** so that tables for different configurations can be compared and
    interpolated against one another, which is why they are recorded here rather
    than derived per-section.
    """

    name: str
    mesh: Any
    reference_area: float
    reference_length: float
    body_length: float
    kind: str = "body"
    """A free-form label. The kernel attaches no meaning to it.

    It exists so a caller that groups configurations — by staging role, by
    build variant, by anything — can carry that grouping through a sweep
    without a parallel dictionary keyed on names. What the strings mean is the
    caller's business; a mission layer reading ``"stack"`` and ``"stage"``
    here is one such caller.
    """
    note: str = ""

    @property
    def frontal_area(self) -> float:
        return float(self.mesh.frontal_area())


def build_solver(
    configuration: Configuration,
    *,
    altitude: float = 30.0e3,
    full_fidelity: bool = True,
    mach_floor: float = 1.2,
) -> PanelSolver | PatchedSolver:
    """Assemble the solver for one configuration.

    With ``full_fidelity`` the panel method is patched with the models that matter
    away from its own validity: skin friction under a radiative-equilibrium wall
    (impact theory has no boundary layer), an equilibrium-air stagnation pressure
    coefficient (the perfect-gas :math:`C_{p,\\max}` is 5–6 % low by Mach 25), and
    a free-molecular limit bridged across the transition regime. Without it, the
    bare inviscid panel solver — faster, and the right thing when comparing shapes
    rather than predicting flight.
    """
    panel = PanelSolver(
        mesh=configuration.mesh,
        reference_area=configuration.reference_area,
        reference_length=configuration.reference_length,
        mach_floor=mach_floor,
        name=f"{configuration.name}-panel",
    )
    if not full_fidelity:
        return panel
    return PatchedSolver(
        panel=panel,
        reference_area=configuration.reference_area,
        reference_length=configuration.reference_length,
        altitude=altitude,
        friction=SkinFrictionModel(
            mesh=configuration.mesh,
            reference_area=configuration.reference_area,
            reference_length=max(configuration.body_length, 1e-3),
            boundary_layer=BoundaryLayer(wall=RadiativeEquilibriumWall()),
        ),
        free_molecular=FreeMolecularSolver(
            mesh=configuration.mesh,
            reference_area=configuration.reference_area,
            reference_length=configuration.reference_length,
        ),
        real_gas=EquilibriumAir(),
    )


def build_tables(
    configurations: Sequence[Configuration],
    grid: SweepGrid | None = None,
    *,
    store_dir: str | Path,
    altitude: float = 30.0e3,
    full_fidelity: bool = True,
    progress: Callable[[dict[str, Any]], None] | None = None,
    max_points: int | None = None,
    time_budget: float | None = None,
) -> dict[str, AeroTable]:
    """Sweep every configuration, checkpointing into ``store_dir``.

    Each configuration gets its own checkpoint file, so an interrupted run resumes
    exactly where it stopped and one file can never mix two configurations.
    """
    sweep = grid or SweepGrid(
        mach=SweepGrid.default_mach(minimum=1.2), alpha=SweepGrid.default_alpha()
    )
    out = Path(store_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables: dict[str, AeroTable] = {}
    for configuration in configurations:
        run = SweepRun(
            name=configuration.name,
            grid=sweep,
            solver=build_solver(configuration, altitude=altitude, full_fidelity=full_fidelity),
            store=out / f"{configuration.name}.jsonl",
            reference_area=configuration.reference_area,
            reference_length=configuration.reference_length,
        )
        tables[configuration.name] = run.run(
            progress=progress, max_points=max_points, time_budget=time_budget
        )
    return tables


def wind_axes(table: AeroTable, mach: float, alpha: float) -> tuple[float, float]:
    r"""Lift and drag coefficients at ``(mach, alpha)``, from body-axis ones.

    A table stores force in **body axes** — axial :math:`C_A` along the vehicle's
    own :math:`x`, normal :math:`C_N` along its :math:`z`. Entry and glide dynamics
    are written in **wind axes**: drag opposes the freestream and lift is
    perpendicular to it. The two differ by a rotation through the angle of attack,
    and at any nonzero incidence they are not interchangeable — taking
    :math:`C_A` for drag at 20° incidence is wrong by :math:`C_N\sin\alpha`, which
    for a lifting body is the larger of the two terms.

    With the freestream direction in body axes as the panel model defines it,
    :math:`\hat{v}_B = (\cos\alpha,\,0,\,\sin\alpha)`
    (:meth:`~aether.aerodynamics.panels.PanelModel.velocity_direction`), and
    the force coefficient vector :math:`\mathbf{C}_F = (C_A,\,0,\,C_N)`, drag is
    the component along the freestream and lift the component perpendicular to it
    in the pitch plane, :math:`\hat{n} = (-\sin\alpha,\,0,\,\cos\alpha)`:

    .. math::
        C_D = \mathbf{C}_F\cdot\hat{v}_B = C_A\cos\alpha + C_N\sin\alpha, \qquad
        C_L = \mathbf{C}_F\cdot\hat{n}   = C_N\cos\alpha - C_A\sin\alpha.

    This is the standard body-to-wind rotation (Etkin & Reid, *Dynamics of
    Flight*, §1.6; Anderson, *Introduction to Flight*, §5.3). At :math:`\alpha=0`
    it degenerates to :math:`C_D=C_A`, :math:`C_L=C_N`, as it must.

    Returns ``(lift, drag)``.
    """
    coefficients = table.at(float(mach), float(alpha))
    axial, normal = float(coefficients.axial), float(coefficients.normal)
    sin_a, cos_a = float(np.sin(alpha)), float(np.cos(alpha))
    drag = axial * cos_a + normal * sin_a
    lift = normal * cos_a - axial * sin_a
    return lift, drag


def lift_to_drag(table: AeroTable, mach: float, alpha: float) -> float:
    """:math:`L/D` at one condition, from :func:`wind_axes`."""
    lift, drag = wind_axes(table, mach, alpha)
    if drag <= 0.0:
        raise ValueError(
            f"non-positive drag coefficient {drag:g} at Mach {mach:g}, "
            f"alpha {np.rad2deg(alpha):.1f} deg; the table cannot be trimmed there"
        )
    return float(lift / drag)


def trim_for_max_lift_to_drag(
    table: AeroTable, mach: float, alpha: Sequence[float] | None = None
) -> tuple[float, float]:
    """The incidence that maximises :math:`L/D` at ``mach``, and that maximum.

    A glider does not fly at zero incidence — it flies at the trim that buys the
    most range, and for a hypersonic body :math:`L/D` rises from zero at
    :math:`\\alpha=0`, peaks, and falls again as drag overtakes lift. Searching
    the table's own incidence grid is what makes the glide parameters *measured*
    rather than assumed.

    Returns ``(alpha, max_lift_to_drag)`` in radians and dimensionless.
    """
    grid = np.asarray(alpha if alpha is not None else table.alpha, dtype=np.float64)
    best_alpha, best_ratio = float(grid[0]), -np.inf
    for value in grid:
        lift, drag = wind_axes(table, mach, float(value))
        if drag <= 0.0:
            continue
        ratio = lift / drag
        if ratio > best_ratio:
            best_alpha, best_ratio = float(value), float(ratio)
    if not np.isfinite(best_ratio):
        raise ValueError(f"no incidence on the grid gives positive drag at Mach {mach:g}")
    # A maximum found at an endpoint is not a maximum, it is where the search
    # stopped: the curve was still climbing when the grid ran out, so the real
    # trim lies outside it and the ratio returned is an underestimate. Reported
    # rather than returned silently, because the number is used as a vehicle
    # parameter and would otherwise look like a measurement.
    if grid.size > 1 and best_alpha in (float(grid[0]), float(grid[-1])):
        warnings.warn(
            f"maximum L/D at Mach {mach:g} sits on the edge of the incidence grid "
            f"({np.rad2deg(best_alpha):.1f} deg, grid "
            f"{np.rad2deg(grid[0]):.1f}-{np.rad2deg(grid[-1]):.1f} deg): the true "
            "trim is outside the swept range, so this L/D is a lower bound on the "
            "shape's own lower bound. Widen the grid (aero_build.reentry_alpha_grid) "
            "or refine its step.",
            TrimAtGridEdge,
            stacklevel=2,
        )
    return best_alpha, float(best_ratio)


def ballistic_coefficient(table: AeroTable, mass: float, mach: float, alpha: float = 0.0) -> float:
    r""":math:`\beta = m/(C_D S)` (kg/m²) at one condition.

    The definition entry and glide dynamics use — drag deceleration is
    :math:`q/\beta` — with :math:`C_D` the **wind-axis** drag from
    :func:`wind_axes` and :math:`S` the table's own reference area, so the
    number cannot be silently combined with a different non-dimensionalisation.
    """
    if not (np.isfinite(mass) and mass > 0.0):
        raise ValueError(f"mass must be finite and > 0, got {mass}")
    _lift, drag = wind_axes(table, mach, alpha)
    if drag <= 0.0:
        raise ValueError(f"non-positive drag coefficient {drag:g} at Mach {mach:g}")
    return float(mass / (drag * table.reference_area))


def save_tables(directory: str | Path, tables: dict[str, AeroTable]) -> Path:
    """Write tables to ``directory`` as one ``.npz`` each, plus an index."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {}
    for name, table in tables.items():
        np.savez_compressed(
            out / f"{name}.npz",
            mach=table.mach,
            alpha=table.alpha,
            axial=table.axial,
            normal=table.normal,
            pitching_moment=table.pitching_moment,
        )
        index[name] = {
            "reference_area": float(table.reference_area),
            "reference_length": float(table.reference_length),
            "solver": table.solver,
            "complete": bool(table.complete),
            "filled": int(table.filled),
            "metadata": table.metadata,
        }
    (out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return out


def load_tables(directory: str | Path) -> dict[str, AeroTable]:
    """Read tables written by :func:`save_tables`."""
    src = Path(directory)
    index = json.loads((src / "index.json").read_text(encoding="utf-8"))
    tables: dict[str, AeroTable] = {}
    for name, meta in index.items():
        with np.load(src / f"{name}.npz") as archive:
            tables[name] = AeroTable(
                name=name,
                mach=archive["mach"],
                alpha=archive["alpha"],
                axial=archive["axial"],
                normal=archive["normal"],
                pitching_moment=archive["pitching_moment"],
                reference_area=float(meta["reference_area"]),
                reference_length=float(meta["reference_length"]),
                solver=str(meta["solver"]),
                metadata=dict(meta.get("metadata", {})),
            )
    return tables
