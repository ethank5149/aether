"""The shock-fitted mesh, built once and used by everyone who needs one.

Everything outside the bow shock is undisturbed freestream, so the domain is
thin and the whole of it is one wall-normal extruded block -- no far-field box,
no tetrahedra, and a cell count spent where the gradients are.

This module exists because there were **two** implementations of that. The
worker had one and :mod:`~aether.cfd.preflight` had a second, written to
"build what a run would build and check it agrees with itself" -- and they
disagreed. The nose cap was applied in both behind a ``geometry["nose_radius"]``
lookup, the fix to measure the radius instead landed in the worker, and the
preflight went on reporting **8 160** poor cells for a mesh the worker would
build with **102**. A gate that models the thing it is gating is a gate that
drifts, and the cheapest way not to drift is not to have a second copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

__all__ = ["CAP_STANDOFFS", "FittedMesh", "fit_shock_layer"]

#: How many nose standoffs a wall-normal column may run before it is treated as
#: never having reached the shock. A hemisphere needs 4.5 at the shoulder, where
#: the wall normal is perpendicular to the axis and the shock layer is at its
#: thickest; this leaves headroom without letting a genuine runaway -- a
#: waverider's freestream surface, which produced 19.9 m columns on a 4 m
#: vehicle -- go unbounded.
CAP_STANDOFFS = 8.0


class Say(Protocol):
    """Somewhere to report progress, or nowhere."""

    def __call__(self, message: str, /) -> None: ...


@dataclass(frozen=True)
class FittedMesh:
    """A built shock-fitted block and the reasoning that sized it."""

    surface: Any
    """The wall triangulation actually extruded, nose cap included."""

    layer: Any
    """The :class:`~aether.cfd.extrude.ExtrudedLayer`."""

    envelope: Any
    """The shock provider the outer boundary was marched to."""

    requirement: Any
    """The :class:`~aether.cfd.gridding.GridRequirement` it was sized by."""

    quality: Any
    """Scaled-Jacobian statistics over every prism."""

    standoff: float
    """Marched nose standoff, metres. Not read off the provider: every provider
    answers the same question the same way this way."""

    layers: int
    growth: float
    capped: Any = None
    """The :class:`~aether.cfd.nosecap.CapReplacement`, or ``None`` if the
    body has no spherical nose to cap."""

    wake: Any = None
    """The :class:`~aether.cfd.extrude.WakeBlock`, or ``None``.

    ``None`` means the domain ends at the base plane, and the base then
    contributes through ``closure.base_axial_coefficient`` -- an empirical
    :math:`C_{p,b} = -1/M^2` worth 7 % of the total drag at Mach 10 and **32 %
    at Mach 4**. With a wake the base disc is a wall, and what it contributes is
    an integrated pressure the solver computed.
    """

    @property
    def n_elements(self) -> int:
        """Cells in the **whole** domain, afterbody block included.

        ``quality.elements`` counts the forebody prisms, because that is what
        the scaled-Jacobian pass measures. Reporting it as the mesh size
        under-states a wake-resolved domain by the whole afterbody -- 434 880
        against 540 480 on the Mach 8 sphere-cone, a fifth of the cells missing
        from every log line, preflight estimate and run record that quotes it.
        """
        total = int(self.quality.elements)
        if self.wake is not None:
            total += self.wake.layers * (
                len(self.wake.faces) + len(self.wake.quads)
            )
        return total

    @property
    def unenclosed_fraction(self) -> float:
        """Share of columns whose outer boundary lies **behind the shock**.

        This is the question, and "was the column capped" is not it. A column
        caps when its normal does not reach the envelope within the allowed
        length, and near a free edge that happens for a reason that says nothing
        about the envelope: the smoothed normal tilts aft, the column runs
        nearly parallel to the shock instead of across it, and travels a long
        way. Measured on the sphere-cone, **every one of the 240 capped columns
        is in the last 6 % of the body**, and the shock there is 60 mm off a
        wall whose column was cut at 512 -- enclosed nine times over.

        The hemisphere's failure looked identical by that metric and was not:
        its cap was a quarter of a 45 mm body, 11.25 mm, against a shoulder
        shock layer 30 mm thick, so the boundary really was inside the shock
        over 43 % of the wall and freestream was being imposed there.

        So the test is asked of the geometry directly: is the last node of each
        column upstream of the envelope? Nought means every column encloses the
        shock it was marched to, whatever else the cap did.
        """
        import numpy as np

        from aether.cfd.shock import _field

        top = self.layer.nodes[:, -1, :]
        return float(np.mean(_field(self.envelope, top) <= 0.0))

    @property
    def capped_fraction(self) -> float:
        """Share of columns that never reached the shock and were truncated.

        ``march_thickness`` caps a column at a quarter of the body's axial span
        when its normal does not cross the envelope, and says in its own
        docstring that a large count "means the envelope does not bound this
        body and the mesh should be questioned rather than used".

        **Nothing read it.** On the hemisphere it is 42.6 % -- the Billig
        hyperbola is fitted with a Mach-angle asymptote, which closes on the
        axis faster than a body wrapping to 90 degrees does, so aft of about 50
        degrees the hyperbola passes *inside* the body and no outward normal
        ever crosses it. Over that 43 % the outer boundary is not a shock at
        all: it is a constant 0.25 x span offset carrying an imposed freestream,
        and the real shock is outside it. The sphere-cone and the biconic are
        both 0 %.
        """
        columns = int(self.layer.nodes.shape[0])
        return float(self.layer.capped) / max(columns, 1)

    def cells_in_shock_layer(self) -> int:
        """Cells between wall and shock on the **tightest** column.

        The minimum rather than the mean, because a mesh that resolves the
        shock layer everywhere except the stagnation line resolves the one
        place the answer is taken from nowhere at all.
        """
        from aether.cfd.shock import _field

        nodes = self.layer.nodes
        inside = _field(self.envelope, nodes.reshape(-1, 3)).reshape(nodes.shape[:2]) < 0
        return int((inside.sum(axis=1) - 1).min())


def fit_shock_layer(
    spec: Any,
    flow: Any,
    *,
    say: Say | None = None,
    strict: bool = True,
) -> FittedMesh:
    """Build the block. ``strict`` raises on a mesh that must not be solved on.

    The worker wants ``strict``: a crossed or inverted mesh is a run that will
    fail an hour later and be blamed on the flow. The preflight does not: it is
    reporting, and an exception there says less than a check that failed with a
    count in it.
    """
    from aether.cfd.extrude import (
        extrude_to_shock,
        measure_quality,
        wake_block,
    )
    from aether.cfd.geometries import build_surface
    from aether.cfd.gridding import grid_requirement, growth_for, stagnation_pressure
    from aether.cfd.nosecap import replace_nose_pole
    from aether.cfd.shock import _march_to_shock, envelope_for
    from aether.geometry.mesh import VehicleMesh

    report = say or (lambda message: None)

    surface = build_surface(spec.body, resolution=spec.refinement, **spec.geometry)

    # A surface of revolution converges every meridian to one vertex, and the
    # stagnation point is the worst place in a hypersonic mesh to keep the worst
    # cells. Not optional tidying: the revolved cap's poor-cell count grows
    # without bound under refinement -- 0, 0, 7040, 10560 across a fourfold cell
    # count -- which is a ceiling on how far the grid can be pushed.
    #
    # The radius is measured from the mesh rather than read out of the geometry
    # dict, because reading it meant knowing the parameter name and the rule had
    # an exception: the hemisphere spells it ``radius``.
    capped = replace_nose_pole(surface)
    if capped is not None:
        surface = VehicleMesh(capped.vertices, capped.faces, name=f"{spec.body}-capped")
        report(
            f"singularity-free nose cap at {capped.seam_angle:.1f} deg -- "
            f"{capped.removed} faces replaced by {capped.added}, peak valence "
            f"{capped.peak_valence_before} -> {capped.peak_valence_after}"
        )

    # Built about the freestream, not the body axis. At incidence those differ
    # by the incidence angle, which on a 2 m body at 6 degrees is 0.21 m of
    # displacement against a 9 mm standoff -- and the outer boundary now
    # *imposes* freestream, so a misplaced envelope asserts undisturbed flow
    # inside the shock layer rather than merely meshing it oddly.
    envelope = envelope_for(
        spec.body,
        spec.geometry,
        spec.mach,
        alpha=float(np.radians(spec.alpha_deg)),
        beta=float(np.radians(spec.sideslip_deg)),
    )
    envelope.validity.check(spec.mach)

    # Upstream from the stagnation point along the oncoming flow, which at
    # incidence is neither the body's nose nor the body's axis.
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    axis = getattr(envelope, "axis", np.array([1.0, 0.0, 0.0]))
    origin = getattr(envelope, "origin", None)
    if origin is None:
        origin = vertices[int(np.argmin(vertices[:, 0]))]
    standoff = _march_to_shock(
        envelope, np.asarray(origin, dtype=np.float64), -np.asarray(axis, dtype=np.float64),
        limit=4.0 * float(np.ptp(vertices[:, 0])),
    )
    requirement = grid_requirement(
        standoff,
        stagnation_pressure(spec.mach, flow.pressure),
        flow.wall_temperature,
        cell_reynolds=spec.cell_reynolds,
    )

    # Refine the normal direction by setting the layer count and letting the
    # grading follow, not by choosing a growth ratio and taking whatever count
    # falls out -- that refines normal by 1.28x where the surface refines by
    # 1.41x, and a study whose refinement ratio differs by axis has no single
    # ``r`` to take an order against. The wall cell is untouched: it is set by
    # the cell Reynolds number, which is physics.
    layers = max(2, round(requirement.cells_across_layer * spec.refinement))
    growth = growth_for(requirement.wall_cell, requirement.outer_boundary, layers)
    report(
        f"{envelope.validity.provider} -- {envelope.validity.why}; standoff "
        f"{1000 * standoff:.2f} mm, wall cell {1e6 * requirement.wall_cell:.3f} um, "
        f"{layers} layers"
        + (f", refinement {spec.refinement:g}x" if spec.refinement != 1.0 else "")
    )
    for caveat in envelope.validity.caveats:
        report(f"  caveat -- {caveat}")

    # The column-length cap defaults to a quarter of the body's axial span,
    # which is the wrong length scale for a blunt body -- see `march_thickness`.
    # Given from the standoff here, because that is the envelope's own scale and
    # this is the one caller that has measured it.
    span = float(np.ptp(np.asarray(surface.vertices, dtype=np.float64)[:, 0]))
    cap = max(0.25 * span, CAP_STANDOFFS * standoff)

    layer = extrude_to_shock(
        surface, envelope, requirement.wall_cell, layers=layers, growth=growth, cap=cap
    )
    # Stated as a range, because the ratio is solved per column: one figure here
    # would be a number no column used. The nominal `growth` above is only the
    # fallback for a column `growth_for` cannot solve.
    if layer.growth is not None and len(layer.growth):
        report(
            f"  grading {layer.growth.min():.4f}..{layer.growth.max():.4f} per column, "
            f"shock layer {1000 * float(np.min(layer.thickness)):.1f}"
            f"..{1000 * float(np.max(layer.thickness)):.1f} mm"
        )
    # The afterbody block, when the spec asks for it. Built here rather than in
    # the worker because it needs the same surface, the same layer and the same
    # envelope this function already holds -- and because a base pressure that
    # is computed instead of correlated is a property of the mesh, not of the
    # run that happens to use it.
    wake = None
    if getattr(spec, "wake", False):
        wake = wake_block(surface, layer, envelope=envelope)
        if wake is None:
            report(
                "  no aft face to sweep -- this body closes to a point, so there "
                "is no base flow and no base drag to resolve"
            )
        else:
            base_span = float(np.ptp(np.asarray(wake.nodes)[:, :, 0]))
            report(
                f"  wake block: {wake.nodes.shape[0]} cap nodes x {wake.layers} "
                f"layers, {len(wake.faces)} base triangles and {len(wake.quads)} "
                f"annulus quads, reaching {base_span:.2f} m downstream"
            )

    quality = measure_quality(layer)
    if layer.capped:
        share = layer.capped / max(int(layer.nodes.shape[0]), 1)
        report(
            f"{layer.capped} of {layer.nodes.shape[0]} columns ({share:.1%}) never "
            "reached the shock and were truncated at a quarter of the body span -- "
            "over those, the outer boundary is an offset surface with freestream "
            "imposed on it, not a shock"
        )

    if strict:
        if layer.crossings:
            msg = (
                f"{layer.crossings} extruded columns cross, so the mesh would carry "
                "inverted cells. Increase smoothing or blunt the offending region."
            )
            raise ValueError(msg)
        if not quality.usable:
            msg = (
                f"{quality.inverted} of {quality.elements} prisms are inverted. "
                "Refusing to write a mesh SU2 would fail on later and blame on the flow."
            )
            raise ValueError(msg)

    return FittedMesh(
        surface=surface,
        layer=layer,
        envelope=envelope,
        requirement=requirement,
        quality=quality,
        standoff=float(standoff),
        layers=int(layers),
        growth=float(growth),
        capped=capped,
        wake=wake,
    )


def write(fitted: FittedMesh, path: Path) -> dict[str, int]:
    """Write the block as ``.su2`` and return its element counts.

    The afterbody block goes out with it when one was built, welded on rather
    than written beside it: two files would be two domains.
    """
    from aether.cfd.extrude import write_su2

    return write_su2(fitted.layer, path, wake=fitted.wake)
