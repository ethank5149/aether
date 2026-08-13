""":math:`C^2` blending, in one place because six modules need it.

Every patched model in this codebase has a seam — windward against leeward,
the 1976 standard against NRLMSIS, laminar against turbulent, CFD against
impact theory — and every one of them is blended with the same quintic
smoothstep. It lives here, at the root of the package with no dependencies,
rather than inside any one of them.

That is a structural point, not a filing decision. It was originally in
:mod:`aether.aerodynamics.closure`, and when the atmosphere came to need it
the import ``aether.atmosphere.model`` → ``aether.aerodynamics.closure``
executed the aerodynamics package's ``__init__``, which imports the composite
solver, which imports the atmosphere — a cycle that failed at collection
time. A utility used by two subsystems belongs above both of them.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["smoothstep"]

_FloatArray = NDArray[np.float64]


def smoothstep(t: ArrayLike) -> _FloatArray:
    """:math:`C^2` smoothstep :math:`6t^5 - 15t^4 + 10t^3` on :math:`[0,1]`.

    Its first *and* second derivatives vanish at both ends, which is what
    makes a blended quantity :math:`C^2` at the band edges — a cubic
    smoothstep would only give :math:`C^1`, and a :math:`C^1` seam in a drag
    coefficient is something an adaptive integrator can still feel.
    """
    x = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    return np.asarray(x * x * x * (x * (6.0 * x - 15.0) + 10.0))
