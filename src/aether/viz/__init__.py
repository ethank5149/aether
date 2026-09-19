"""Rendering primitives: the globe, its terrain and imagery, and vehicle glyphs.

Generic three-dimensional drawing. A textured WGS84 ellipsoid, elevation and
imagery tiles, and outer mould lines read from mesh files rather than typed out
as formulae. None of it knows what a trajectory means.

Two things here are worth keeping in view because both were mistakes once.

The globe is an **ellipsoid**, not a sphere. Drawing a sphere is self-consistent
and wrong by up to 21.4 km, and the error is not in the radius -- it is in where
a given latitude sits, because geodetic latitude is not geocentric latitude.

Cut flow fields are drawn by :mod:`~aether.viz.flow`, which is the one module
here that draws something a solver produced rather than something a model did.
Its geometry comes from :mod:`aether.aerodynamics.cfd.fields`; nothing in it
decides where a shock is.

Vehicle profiles come from the meshes, not from an analytic formula. A
hand-written mould line is a second model of the same object, and like every
second model it disagreed with the first: the formula drew a smooth tube, so the
separation rings that make a stack read as a stack were simply absent from the
picture.
"""

from aether.viz.flow import *  # noqa: F403
from aether.viz.globe import *  # noqa: F403
from aether.viz.imagery import *  # noqa: F403
from aether.viz.scene import *  # noqa: F403
from aether.viz.terrain import *  # noqa: F403
from aether.viz.vehicle import *  # noqa: F403
