"""Vehicle geometry: outer mould line from a mesh, and the mesh's conditioning.

Replaces stipulated shape scalars with measured ones, and feeds the panel
aerodynamics a surface it was already built to consume.

Three representations, and the distinction between them is the point:

:mod:`~aether.geometry.brep` and :mod:`~aether.geometry.bodies`
    The **shape**, exactly — a lofted or revolved OpenCASCADE solid built from
    the analytic cross-sections a vehicle class is defined by. Meshable at any
    density, exportable to STEP, measurable without discretisation error.
:mod:`~aether.geometry.mesh`
    A **sampling** of a shape: welded triangles and the quantities read off
    them. What a solver consumes.
:mod:`~aether.geometry.prep`
    The step that turns an *authored* mesh, which carries no units or
    orientation convention, into one of those.

Prefer starting from the first. A triangulation refined around a fixed
faceting converges to the faceted body rather than the real one, which on the
generic lifting body here is a 3.2 % error in volume that no CFD mesh
refinement can reach.

:mod:`~aether.geometry.mesh` is the model itself — welded triangles and the
quantities read off them. :mod:`~aether.geometry.prep` is the step before it:
an authored STL is a bag of triangles carrying no units, no orientation
convention and no guarantee of consistent winding, and each of those changes an
aerodynamic answer rather than the picture. Conditioning is generic geometry
and knows nothing about what the body is for, which is why it lives here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

#: Where each public name lives. Resolved on first access rather than at
#: import, which is what keeps this package importable from
#: :mod:`aether.aerodynamics`.
#:
#: The cycle it breaks is real and was load-bearing:
#: :mod:`aether.geometry.mesh` imports ``PanelModel`` from
#: :mod:`aether.aerodynamics.panels`, so eagerly importing ``mesh`` here meant
#: that any module in ``aerodynamics`` which touched ``geometry`` re-entered
#: ``panels`` while it was still initialising. Nothing depended on the eager
#: import except the habit of writing one, and deferring it also stops
#: ``import aether.geometry`` from loading OpenCASCADE for callers that only
#: wanted a meridian profile.
_EXPORTS: dict[str, str] = {
    "Loft": "aether.geometry.brep",
    "Revolve": "aether.geometry.brep",
    "SolidProperties": "aether.geometry.brep",
    "solid_properties": "aether.geometry.brep",
    "surface_mesh": "aether.geometry.brep",
    "write_step": "aether.geometry.brep",
    "WallColumnGrid": "aether.geometry.columns",
    "graded_widths": "aether.geometry.columns",
    "solve_growth": "aether.geometry.columns",
    "wall_columns": "aether.geometry.columns",
    "VehicleMesh": "aether.geometry.mesh",
    "load_stl": "aether.geometry.mesh",
    "write_stl": "aether.geometry.mesh",
    "ConditionReport": "aether.geometry.prep",
    "condition": "aether.geometry.prep",
    "export_master": "aether.geometry.prep",
    "measure": "aether.geometry.prep",
    "sha256_of": "aether.geometry.prep",
    "sphere_cone_closure": "aether.geometry.profiles",
    "sphere_cone_meridian": "aether.geometry.profiles",
    "sphere_cone_tangency": "aether.geometry.profiles",
}

if TYPE_CHECKING:  # pragma: no cover - for type checkers, not at runtime
    from aether.geometry import bodies, profiles
    from aether.geometry.brep import (
        Loft,
        Revolve,
        SolidProperties,
        solid_properties,
        surface_mesh,
        write_step,
    )
    from aether.geometry.columns import (
        WallColumnGrid,
        graded_widths,
        solve_growth,
        wall_columns,
    )
    from aether.geometry.mesh import VehicleMesh, load_stl, write_stl
    from aether.geometry.prep import (
        ConditionReport,
        condition,
        export_master,
        measure,
        sha256_of,
    )
    from aether.geometry.profiles import (
        sphere_cone_closure,
        sphere_cone_meridian,
        sphere_cone_tangency,
    )


def __getattr__(name: str) -> Any:
    """Resolve a public name to the module that defines it, on first use."""
    if name in ("bodies", "profiles"):
        import importlib

        return importlib.import_module(f"aether.geometry.{name}")
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "ConditionReport",
    "Loft",
    "Revolve",
    "SolidProperties",
    "VehicleMesh",
    "WallColumnGrid",
    "bodies",
    "condition",
    "export_master",
    "graded_widths",
    "load_stl",
    "measure",
    "profiles",
    "sha256_of",
    "solid_properties",
    "solve_growth",
    "sphere_cone_closure",
    "sphere_cone_meridian",
    "sphere_cone_tangency",
    "surface_mesh",
    "wall_columns",
    "write_step",
    "write_stl",
]
