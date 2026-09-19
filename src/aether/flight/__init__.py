"""The coupled flight simulator: one rigid body, one system of ODEs.

Thirteen rigid-body states -- geocentric position, body-frame velocity, attitude
quaternion, body rates -- augmented by mass, retained structural modes, and a
thermal grid carrying surface recession. They are advanced *together* rather
than in sequence, which is what makes the couplings real: aerodynamic load
drives the modal oscillators, recession moves the surface the heating is
computed at, and mass depletion changes the acceleration a burn produces.

The right-hand side contains **no branch on flight regime**. Atmospheric density
decays smoothly, so every aerodynamic and aerothermal term vanishes numerically
above the sensible atmosphere on its own -- a regime test would be a discrete
event where the physics has none, and an implicit integrator asked to step
across one either crawls or smears it.

Generic rigid-body dynamics over an arbitrary central body, assembled from the
structural, thermal, aerothermal, attitude and orbital kernels. What is flown,
and to what end, lives elsewhere.
"""

from aether.flight.propulsion import *  # noqa: F403
from aether.flight.simulator import *  # noqa: F403
from aether.flight.state import *  # noqa: F403
