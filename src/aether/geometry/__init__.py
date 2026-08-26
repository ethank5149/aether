"""Vehicle geometry: outer mould line from a mesh, and the mesh's conditioning.

Replaces stipulated shape scalars with measured ones, and feeds the panel
aerodynamics a surface it was already built to consume.

:mod:`~aether.geometry.mesh` is the model itself — welded triangles and the
quantities read off them. :mod:`~aether.geometry.prep` is the step before it:
an authored STL is a bag of triangles carrying no units, no orientation
convention and no guarantee of consistent winding, and each of those changes an
aerodynamic answer rather than the picture. Conditioning is generic geometry
and knows nothing about what the body is for, which is why it lives here.
"""

from __future__ import annotations

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
    "VehicleMesh",
    "condition",
    "export_master",
    "load_stl",
    "measure",
    "sha256_of",
    "write_stl",
]
