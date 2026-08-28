"""Blended windward/leeward pressure closure (Paper II, §3.3).

Modified Newtonian pressure alone is inadequate for slender waveriders
at moderate incidence: it assigns zero pressure coefficient to every
leeward surface, discarding leeward suction. The closure is therefore
blended — modified Newtonian on windward panels (Eq. 3.5),

.. math::

    C_p = C_{p,\\max}\\sin^2\\delta_c, \\qquad
    C_{p,\\max} = \\frac{2}{\\gamma M_\\infty^2}
    \\left(\\frac{p_{02}}{p_\\infty} - 1\\right)

with the pressure ratio from the Rayleigh–Pitot relation, and
Prandtl–Meyer expansion on leeward panels down to the vacuum limit
:math:`C_p = -2/(\\gamma M_\\infty^2)`.

**The seam.** The two branches are :math:`C^0` but not :math:`C^1` at
:math:`\\delta_c = 0`: the windward branch has zero slope there while
the expansion branch does not. That is a genuine violation of the
smoothness the framework otherwise maintains, and it happens exactly at
the shoulder line where panels change branch. The branches are blended
over :math:`|\\delta_c| < \\delta_{\\mathrm{blend}}` by a :math:`C^2`
smoothstep. This is a numerical expedient, not physics — its effect on
integrated loads is verification task II-V4, which is why the blend
width is an explicit parameter everywhere rather than a hidden constant.

Both branches are evaluated by *smooth extensions* through the seam —
the Newtonian form is analytic in :math:`\\delta_c`, and the
Prandtl–Meyer branch continues into isentropic compression for
:math:`\\delta_c > 0` — so the blend inherits :math:`C^2` continuity
from the smoothstep alone. Clamping one branch at the seam instead would
leave a second-derivative kink and defeat the purpose.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Re-exported: the seam blend is shared with the atmosphere and the boundary
# layer, so it lives at the package root. See aether.blending.
from aether.blending import smoothstep

__all__ = [
    "base_axial_coefficient",
    "base_pressure_coefficient",
    "blended_pressure_coefficient",
    "newtonian_pressure_coefficient",
    "prandtl_meyer_angle",
    "prandtl_meyer_pressure_coefficient",
    "rayleigh_pitot_cp_max",
    "smoothstep",
    "vacuum_pressure_coefficient",
]

_FloatArray = NDArray[np.float64]


def _check_mach(mach: float) -> float:
    m = float(mach)
    if not (np.isfinite(m) and m > 1.0):
        raise ValueError(f"freestream Mach must be finite and supersonic, got {mach}")
    return m


def _check_gamma(gamma: float) -> float:
    g = float(gamma)
    if not (np.isfinite(g) and g > 1.0):
        raise ValueError(f"gamma must be finite and > 1, got {gamma}")
    return g


def rayleigh_pitot_cp_max(mach: float, gamma: float = 1.4) -> float:
    """:math:`C_{p,\\max}` behind a normal shock (Paper II, Eq. 3.5).

    Uses the Rayleigh–Pitot relation

    .. math::

        \\frac{p_{02}}{p_\\infty} =
        \\left[\\frac{(\\gamma+1)^2M^2}{4\\gamma M^2 - 2(\\gamma-1)}
        \\right]^{\\frac{\\gamma}{\\gamma-1}}
        \\frac{1 - \\gamma + 2\\gamma M^2}{\\gamma+1}.

    Approaches the classical hypersonic limit (1.839 for
    :math:`\\gamma = 1.4`) as :math:`M \\to \\infty`.
    """
    m = _check_mach(mach)
    g = _check_gamma(gamma)
    m2 = m * m
    ratio = ((g + 1.0) ** 2 * m2 / (4.0 * g * m2 - 2.0 * (g - 1.0))) ** (g / (g - 1.0))
    ratio *= (1.0 - g + 2.0 * g * m2) / (g + 1.0)
    return float(2.0 / (g * m2) * (ratio - 1.0))


def vacuum_pressure_coefficient(mach: float, gamma: float = 1.4) -> float:
    """The vacuum limit :math:`C_p = -2/(\\gamma M_\\infty^2)`."""
    m = _check_mach(mach)
    g = _check_gamma(gamma)
    return float(-2.0 / (g * m * m))


def newtonian_pressure_coefficient(
    incidence: ArrayLike, cp_max: float
) -> _FloatArray:
    """Modified Newtonian :math:`C_p = C_{p,\\max}\\sin^2\\delta_c`.

    Evaluated as the analytic function of :math:`\\delta_c`, without
    clamping at the seam — the smooth extension the blend requires.
    """
    delta = np.asarray(incidence, dtype=np.float64)
    return np.asarray(float(cp_max) * np.sin(delta) ** 2)


def prandtl_meyer_angle(mach: ArrayLike, gamma: float = 1.4) -> _FloatArray:
    """Prandtl–Meyer function :math:`\\nu(M)` in radians."""
    g = _check_gamma(gamma)
    m = np.asarray(mach, dtype=np.float64)
    if np.any(m < 1.0):
        raise ValueError("Prandtl–Meyer function requires M >= 1")
    root = np.sqrt(np.maximum(m * m - 1.0, 0.0))
    factor = np.sqrt((g + 1.0) / (g - 1.0))
    return np.asarray(factor * np.arctan(root / factor) - np.arctan(root))


def prandtl_meyer_limit(gamma: float = 1.4) -> float:
    """:math:`\\nu_{\\max} = \\tfrac{\\pi}{2}(\\sqrt{(\\gamma+1)/(\\gamma-1)} - 1)`.

    The total turning the flow can achieve before reaching vacuum — 130.45
    degrees for :math:`\\gamma = 1.4`. A surface asked to turn further is at
    vacuum pressure, not at an error.
    """
    g = _check_gamma(gamma)
    return float(0.5 * np.pi * (np.sqrt((g + 1.0) / (g - 1.0)) - 1.0))


def _mach_from_prandtl_meyer(
    nu: ArrayLike, gamma: float, newton_steps: int = 3, bisections: int = 26
) -> _FloatArray:
    """Invert :math:`\\nu(M)`, vectorised over the whole array at once.

    **Why this is not a root-finder call per element.** It was: a
    ``scipy.optimize.brentq`` per panel, inside a Python loop over
    ``np.ndenumerate``. On the reference-heavy stack that is 28,435 scalar solves for a
    single (Mach, incidence) point, each evaluating :math:`\\nu(M)` about nine
    times through a function that validates its input on every call — 240 ms
    a point, of which 99.9 % was here, and **290 times slower than the
    free-molecular solver doing a comparable amount of arithmetic over the
    same mesh**. A coefficient sweep is thousands of points, so this one loop
    set the cost of the whole table.

    The replacement inverts on :math:`t = 1/M \\in (0, 1]`, on which
    :math:`\\nu` is monotonically decreasing with finite endpoints —
    :math:`\\nu_{\\max}` at :math:`t \\to 0` and zero at :math:`t = 1` — so the
    bracket is the unit interval for every input and no bracket search is
    needed. Bisection localises it, then Newton in :math:`M` polishes with

    .. math::

        \\frac{\\mathrm{d}\\nu}{\\mathrm{d}M}
        = \\frac{\\sqrt{M^2-1}}{M\\left(1 + \\frac{\\gamma-1}{2}M^2\\right)}

    Both loops are fixed-length and branch-free, so the whole thing is a few
    dozen elementwise passes — which is also what makes it portable to an
    array backend that is not NumPy.

    Returns ``1.0`` where the turn has driven the flow back to sonic and
    ``inf`` where it has exceeded :math:`\\nu_{\\max}`.
    """
    g = _check_gamma(gamma)
    target = np.asarray(nu, dtype=np.float64)
    limit = prandtl_meyer_limit(g)
    factor = np.sqrt((g + 1.0) / (g - 1.0))

    def angle_of(t: _FloatArray) -> _FloatArray:
        """:math:`\\nu(1/t)`, guarded at the singular endpoint."""
        mach = 1.0 / np.maximum(t, 1.0e-300)
        root = np.sqrt(np.maximum(mach * mach - 1.0, 0.0))
        return np.asarray(factor * np.arctan(root / factor) - np.arctan(root))

    low = np.zeros_like(target)
    high = np.ones_like(target)
    clamped = np.clip(target, 0.0, limit)
    for _ in range(int(bisections)):
        middle = 0.5 * (low + high)
        above = angle_of(middle) > clamped
        low = np.where(above, middle, low)
        high = np.where(above, high, middle)

    mach = 1.0 / np.maximum(0.5 * (low + high), 1.0e-300)
    for _ in range(int(newton_steps)):
        root = np.sqrt(np.maximum(mach * mach - 1.0, 0.0))
        residual = factor * np.arctan(root / factor) - np.arctan(root) - clamped
        slope = root / (mach * (1.0 + 0.5 * (g - 1.0) * mach * mach))
        # The derivative vanishes at M = 1, where the bisection is already
        # exact; the floor keeps the step finite rather than correcting a
        # value that needs no correction.
        mach = mach - residual / np.where(slope > 1.0e-30, slope, 1.0e-30)
        mach = np.maximum(mach, 1.0)

    mach = np.where(target <= 0.0, 1.0, mach)
    return np.asarray(np.where(target >= limit, np.inf, mach))


def prandtl_meyer_pressure_coefficient(
    incidence: ArrayLike, mach: float, gamma: float = 1.4
) -> _FloatArray:
    """Leeward branch: isentropic turn through :math:`-\\delta_c`.

    Negative ``incidence`` is an expansion (the physical leeward case);
    positive ``incidence`` continues the same isentropic relation into
    compression, which is the smooth extension the blend evaluates
    inside the seam band. The result is floored at the vacuum limit,
    beyond which the surface is treated as being at vacuum pressure.

    Fully vectorised: the inversion of :math:`\\nu` happens once for the whole
    array. See :func:`_mach_from_prandtl_meyer` for why that matters.
    """
    m1 = _check_mach(mach)
    g = _check_gamma(gamma)
    delta = np.atleast_1d(np.asarray(incidence, dtype=np.float64))
    nu1 = float(prandtl_meyer_angle(m1, g))
    cp_vac = vacuum_pressure_coefficient(m1, g)
    stagnation_factor = 1.0 + 0.5 * (g - 1.0) * m1 * m1

    # nu2 <= 0 means the turn has driven the flow back to sonic and the
    # isentropic branch ends there; nu2 >= nu_max means it has reached vacuum.
    # Both are handled inside the inversion, which returns 1 and inf.
    m2 = _mach_from_prandtl_meyer(nu1 - delta, g)
    # Beyond the turning limit the ratio would be 0/inf; capping the Mach
    # number sends the pressure ratio to zero, which *is* the vacuum limit,
    # and the explicit floor below makes that exact rather than asymptotic.
    capped = np.where(np.isfinite(m2), m2, 1.0e8)
    ratio = (stagnation_factor / (1.0 + 0.5 * (g - 1.0) * capped * capped)) ** (
        g / (g - 1.0)
    )
    out = np.maximum(2.0 / (g * m1 * m1) * (ratio - 1.0), cp_vac)
    return np.asarray(out.reshape(np.shape(incidence)))


def blended_pressure_coefficient(
    incidence: ArrayLike,
    mach: float,
    gamma: float = 1.4,
    blend_width: float = 0.02,
    cp_max: float | None = None,
) -> _FloatArray:
    """Blended :math:`C_p` over the whole incidence range.

    Parameters
    ----------
    incidence:
        Local incidence :math:`\\delta_c` (rad); positive windward.
    mach, gamma:
        Freestream conditions.
    blend_width:
        :math:`\\delta_{\\mathrm{blend}}` (rad), the half-width of the
        seam band. Must be small relative to the incidence variation
        across a collocation cell; zero selects the unblended
        :math:`C^0` closure, which II-V4 uses as its baseline.
    cp_max:
        Override for :math:`C_{p,\\max}`; defaults to the Rayleigh–Pitot
        value at ``mach``.
    """
    m = _check_mach(mach)
    g = _check_gamma(gamma)
    width = float(blend_width)
    if not (np.isfinite(width) and width >= 0.0):
        raise ValueError(f"blend_width must be finite and >= 0, got {blend_width}")
    if width >= 0.5 * np.pi:
        raise ValueError(
            f"blend_width {width} spans the whole incidence range; it must be "
            f"small relative to the incidence variation across a cell"
        )
    cpm = rayleigh_pitot_cp_max(m, g) if cp_max is None else float(cp_max)

    delta = np.asarray(incidence, dtype=np.float64)
    windward = newtonian_pressure_coefficient(delta, cpm)
    leeward = prandtl_meyer_pressure_coefficient(delta, m, g)

    if width == 0.0:
        return np.asarray(np.where(delta > 0.0, windward, leeward))
    weight = smoothstep((delta + width) / (2.0 * width))
    return np.asarray((1.0 - weight) * leeward + weight * windward)


def base_pressure_coefficient(mach: float, gamma: float = 1.4) -> float:
    r"""Base pressure coefficient for a blunt-based body in supersonic flow.

    The engineering estimate :math:`C_{p,b} \approx -1/M^2`, clipped to the
    vacuum limit. It is first-order — real base pressure depends on Reynolds
    number, boundary-layer state at separation, base geometry and any sting or
    plume — but it is *physical*, which is the entire point of using it.

    Why this exists rather than a CFD number
    ----------------------------------------

    An Euler solution has no base pressure worth reading. Real base flow is set
    by a separated viscous shear layer that the Euler equations do not contain,
    so the solver returns whatever its unresolved recirculation settles at, and
    that value is neither converged nor bounded by anything. Measured on this
    package's Mach 8 sphere-cone the Euler base came back at
    :math:`C_p = +0.34` — a *positive* pressure coefficient on a base, pushing
    the vehicle forward, when the physical value is small and negative. It
    contributed :math:`-0.30` to an axial force whose forebody was worth
    :math:`+0.08`, and moved 27 % between grid levels while the forebody moved
    2 %.

    So the base term is not computed, it is *substituted*: run the CFD, take
    the forebody force from it, and add a base drag from here. At Mach 8 this
    gives :math:`C_{p,b} = -0.0156` against a vacuum bound of :math:`-0.0223`
    — both small, both negative, and both nothing like what the solver said.
    """
    if not (np.isfinite(mach) and mach > 1.0):
        msg = f"base pressure correlation needs supersonic flow, got Mach {mach}"
        raise ValueError(msg)
    return float(max(-1.0 / mach**2, vacuum_pressure_coefficient(mach, gamma)))


def base_axial_coefficient(
    mach: float, base_area: float, reference_area: float, gamma: float = 1.4
) -> float:
    r"""Axial-force coefficient contributed by a blunt base.

    :math:`C_A = -C_{p,b} S_b / S_{\mathrm{ref}}`, positive because a base
    pressure below freestream pulls the vehicle backwards. Add it to a CFD
    forebody force to get a total that means something.
    """
    if reference_area <= 0.0 or base_area < 0.0:
        msg = f"areas must be positive, got base {base_area}, reference {reference_area}"
        raise ValueError(msg)
    return float(
        -base_pressure_coefficient(mach, gamma) * base_area / reference_area
    )
