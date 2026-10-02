"""Flight: the entry flown from the compiled symbolic field, and the simulator before it.

Two things live here, and they are not of the same standing.

:mod:`aether.flight.entry` is the simulator of record for unpowered entry.
Every rate it integrates comes out of a routine compiled from the symbolic
field of :mod:`aether.symbolic` -- the field the manuscript's notes on the
dynamics state and its derivations check -- over a rotating, oblate planet whose
atmosphere lies on the reference ellipsoid. It is what the Artemis I example
flies.

:mod:`aether.flight.simulator` is the hand-written simulator that preceded
it: thirteen rigid-body states in inertial Cartesian components, augmented by
mass, retained structural modes and a thermal grid, with finite-thrust arcs
from :mod:`aether.flight.propulsion`. It is not generated from the symbolic
field and is not the model the appendix states; its thermal kernel differs
from conservation of energy in two places that the thermal test suite pins.
It is kept for what was built on it and is not being extended.
"""

from aether.flight.entry import *  # noqa: F403
from aether.flight.propulsion import *  # noqa: F403
from aether.flight.simulator import *  # noqa: F403
from aether.flight.state import *  # noqa: F403
