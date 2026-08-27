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

from aether.geometry import bodies
from aether.geometry.brep import (
    Loft,
    Revolve,
    SolidProperties,
    solid_properties,
    surface_mesh,
    write_step,
)
from aether.geometry.mesh import VehicleMesh, load_stl, write_stl
from aether.geometry.prep import (
    ConditionReport,
    condition,
    export_master,
    measure,
    sha256_of,
)

__all__ = [
    "ConditionReport",
    "Loft",
    "Revolve",
    "SolidProperties",
    "VehicleMesh",
    "bodies",
    "condition",
    "export_master",
    "load_stl",
    "measure",
    "sha256_of",
    "solid_properties",
    "surface_mesh",
    "write_step",
    "write_stl",
]
