"""Textbook guidance and navigation laws.

Proportional navigation and its augmented form, strapdown inertial propagation,
and a numerically stable time-to-go. All of it is standard graduate material --
the augmented proportional navigation law and the time-to-go extrapolation are
in every guidance text -- and all of it is written over abstract relative
position and velocity vectors, with no notion of what is manoeuvring or why.

The time-to-go routine is worth a word, because the textbook form of the root it
solves is the one thing here that is *not* simply quoted: the direct quadratic
formula loses precision catastrophically when the closing acceleration is small,
which is most of an engagement, so a numerically stable rearrangement is used
instead.

Alongside them: lofted-trajectory geometry, ballistic error-coefficient
propagation, Lambert-based midcourse correction, and the linear covariance
machinery a chance-constrained problem needs -- Gaussian propagation, Mahalanobis
distances, risk multipliers. All of it operates on abstract states and
covariances.
"""

from aether.guidance.apn import *  # noqa: F403
from aether.guidance.ballistic_errors import *  # noqa: F403
from aether.guidance.inertial import *  # noqa: F403
from aether.guidance.lofting import *  # noqa: F403
from aether.guidance.midcourse import *  # noqa: F403
from aether.guidance.stochastic_core import *  # noqa: F403
from aether.guidance.tgo import *  # noqa: F403
