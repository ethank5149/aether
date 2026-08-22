"""Two-body astrodynamics: gravity models, Lambert targeting, atmospheric coast.

Textbook orbital mechanics over an arbitrary central body -- the conic, Kepler's
equation, Lambert's boundary value problem, and a drag-perturbed coast. None of
it knows what is flying.
"""

from aether.orbital.coast import *  # noqa: F403
from aether.orbital.gravity import *  # noqa: F403
from aether.orbital.lambert import *  # noqa: F403
