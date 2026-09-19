"""Reference trajectories: specification, solution, and hand-off to guidance.

A trajectory problem is stated as phases, objectives, constraints and
termination triggers; solved into a pseudospectral reference; stored in an
offline library; and handed to the guidance loop as its feed-forward nominal.
"""

from aether.trajectory.constraints import *  # noqa: F403
from aether.trajectory.gnc import *  # noqa: F403
from aether.trajectory.library import *  # noqa: F403
from aether.trajectory.objectives import *  # noqa: F403
from aether.trajectory.problem import *  # noqa: F403
from aether.trajectory.representation import *  # noqa: F403
from aether.trajectory.solve import *  # noqa: F403
from aether.trajectory.triggers import *  # noqa: F403
