"""Vehicle geometry: outer mould line from a mesh.

Replaces stipulated shape scalars with measured ones, and feeds the panel
aerodynamics a surface it was already built to consume.
"""

from __future__ import annotations

from aether.geometry.mesh import VehicleMesh, load_stl, write_stl

__all__ = ["VehicleMesh", "load_stl", "write_stl"]
