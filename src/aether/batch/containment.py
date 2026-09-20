"""Containment radii of a bivariate normal scatter.

A dispersion is produced as principal-axis sigmas and consumed as a radius:
the circle about the mean point that holds a stated fraction of the scatter
-- half of it (:math:`R_{50}`, the circular error probable) or 95 % of it
(:math:`R_{95}`). This module is the conversion, and it is less trivial than
it looks because the usual shortcut is only valid in a case that rarely holds.

The circular case, and why the shortcut fails
---------------------------------------------

For an isotropic bivariate normal with per-axis sigma :math:`\\sigma`, the
distance from the mean is Rayleigh distributed and the radius containing a
fraction :math:`p` is available in closed form:

.. math::

    R_p = \\sigma \\sqrt{-2 \\ln(1 - p)},

so :math:`R_{50} = \\sigma\\sqrt{2\\ln 2} \\approx 1.1774\\sigma` and
:math:`R_{95} = \\sigma\\sqrt{2\\ln 20} \\approx 2.4477\\sigma`. Their ratio
is a constant, :math:`R_{95}/R_{50} \\approx 2.079`, and it is extremely
common to convert between the two by that factor.

**That factor is only correct for a circular distribution**, and terminal
dispersions of an entry are usually not circular: downrange and crossrange
errors come from different mechanisms and differ by a factor of several. As
the ellipse elongates the ratio **rises**: it runs 2.079 at unit aspect
ratio, 2.339 at 2:1, 2.646 at 3.3:1, and approaches 2.906 in the degenerate
one-dimensional limit -- which is exactly :math:`1.96/0.6745`, the
normal-distribution equivalent. Scaling a 50 % radius up to a 95 % radius by
the circular 2.079 therefore **under-states** the 95 % radius for any real,
elongated dispersion, by up to 40 % at the limit. :func:`containment_ratio`
computes it instead of assuming it.

The elliptical case
-------------------

For principal-axis sigmas :math:`\\sigma_1, \\sigma_2` the probability of
falling inside radius :math:`r` has no elementary closed form, but it
reduces to a one-dimensional integral that is exact and cheap. Working in
polar coordinates the radial part integrates analytically, leaving

.. math::

    P(R \\le r) = \\frac{1}{2\\pi\\sigma_1\\sigma_2}
        \\int_0^{2\\pi} \\frac{1 - e^{-a(\\theta) r^2}}{2a(\\theta)}
        \\,\\mathrm{d}\\theta,
    \\qquad
    a(\\theta) = \\frac{\\cos^2\\theta}{2\\sigma_1^2}
              + \\frac{\\sin^2\\theta}{2\\sigma_2^2},

which is quadrature in one variable and then a root-find in :math:`r`. No
approximation formula is used, so the elliptical answers are as exact as
the circular ones rather than being a fitted correction to them.

Verified against a published table, not only against itself
-----------------------------------------------------------

Siouris Table 5.2 tabulates :math:`K` such that :math:`P(R \\le K\\sigma_L) = P`,
over 21 aspect ratios from degenerate to circular and six probability levels
-- 126 published numbers spanning the whole domain this module covers.
:data:`SIOURIS_TABLE_5_2` carries it, and :func:`containment_radius`
reproduces **every entry to within one unit in the last printed place**, with
122 of the 126 inside ideal four-decimal rounding. Both endpoints land on
their closed forms: :math:`1.1774 = \\sqrt{2\\ln 2}` in the circular column,
and :math:`0.6745`, :math:`1.9600` in the degenerate one.

The four exceptions are the *source's* rounding, not ours: in each case the
exact value rounds to a different last digit than the one printed.

.. code-block:: text

    sigma_S/sigma_L   P      published   exact        rounds to
        0.00         0.75     1.1504     1.1503494     1.1503
        0.05         0.50     0.6764     0.6763479     0.6763
        0.75         0.90     1.9034     1.9033494     1.9033
        1.00         0.95     2.4478     2.4477468     2.4477

All four are high by 5.1-5.3e-5, i.e. the table rounds up where the value
rounds down. Across all 126 entries the signed deviations split 69 positive
to 57 negative with a mean of +6.4e-6, so this is four last-place roundings
rather than a bias in either direction.
"""

from __future__ import annotations

import numpy as np
import scipy.integrate
import scipy.optimize
import scipy.special

__all__ = [
    "SIOURIS_TABLE_5_2",
    "SIOURIS_TABLE_5_2_PROBABILITIES",
    "SIOURIS_TABLE_5_2_RATIOS",
    "check_sigmas",
    "containment_probability",
    "containment_radius",
    "containment_ratio",
]

#: Aspect ratios :math:`\sigma_S/\sigma_L` indexing Siouris Table 5.2.
SIOURIS_TABLE_5_2_RATIOS: tuple[float, ...] = (
    0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
)

#: Containment probabilities columning Siouris Table 5.2.
SIOURIS_TABLE_5_2_PROBABILITIES: tuple[float, ...] = (0.30, 0.50, 0.75, 0.90, 0.95, 0.99)

#: Siouris Table 5.2, transcribed as published: :math:`K` such that
#: :math:`P(R \le K\sigma_L) = P`, rows indexed by
#: :data:`SIOURIS_TABLE_5_2_RATIOS` and columns by
#: :data:`SIOURIS_TABLE_5_2_PROBABILITIES`.
#:
#: Kept as published rather than regenerated from
#: :func:`containment_radius`, which would make the comparison circular and
#: destroy the only independent check this module has. The first row is the
#: degenerate :math:`\sigma_S = 0` case, where the answer is the
#: one-dimensional normal quantile rather than a two-dimensional one.
SIOURIS_TABLE_5_2: tuple[tuple[float, ...], ...] = (
    (0.3853, 0.6745, 1.1504, 1.6449, 1.9600, 2.5758),
    (0.3886, 0.6764, 1.1514, 1.6456, 1.9606, 2.5763),
    (0.3987, 0.6820, 1.1547, 1.6479, 1.9625, 2.5778),
    (0.4169, 0.6916, 1.1603, 1.6518, 1.9658, 2.5803),
    (0.4421, 0.7059, 1.1683, 1.6573, 1.9704, 2.5838),
    (0.4705, 0.7254, 1.1788, 1.6646, 1.9765, 2.5884),
    (0.4997, 0.7499, 1.1925, 1.6738, 1.9842, 2.5942),
    (0.5285, 0.7779, 1.2097, 1.6852, 1.9937, 2.6013),
    (0.5568, 0.8079, 1.2310, 1.6992, 2.0051, 2.6100),
    (0.5842, 0.8389, 1.2564, 1.7163, 2.0190, 2.6203),
    (0.6109, 0.8704, 1.2853, 1.7371, 2.0359, 2.6326),
    (0.6369, 0.9021, 1.3172, 1.7621, 2.0564, 2.6474),
    (0.6621, 0.9337, 1.3514, 1.7915, 2.0813, 2.6653),
    (0.6867, 0.9651, 1.3874, 1.8251, 2.1111, 2.6875),
    (0.7107, 0.9962, 1.4247, 1.8625, 2.1460, 2.7151),
    (0.7342, 1.0271, 1.4631, 1.9034, 2.1858, 2.7492),
    (0.7571, 1.0577, 1.5023, 1.9472, 2.2303, 2.7907),
    (0.7796, 1.0880, 1.5422, 1.9936, 2.2791, 2.8401),
    (0.8017, 1.1181, 1.5827, 2.0424, 2.3318, 2.8974),
    (0.8233, 1.1479, 1.6237, 2.0932, 2.3881, 2.9625),
    (0.8446, 1.1774, 1.6651, 2.1460, 2.4478, 3.0349),
)


def check_sigmas(sigma_major: float, sigma_minor: float) -> tuple[float, float]:
    a, b = float(sigma_major), float(sigma_minor)
    for name, value in (("sigma_major", a), ("sigma_minor", b)):
        if not (np.isfinite(value) and value >= 0.0):
            raise ValueError(f"{name} must be finite and >= 0, got {value}")
    if a < b:
        a, b = b, a
    if a == 0.0:
        raise ValueError(
            "at least one sigma must be positive; a distribution with no "
            "spread has no containment radius to report"
        )
    return a, b


def containment_probability(radius: float, sigma_major: float, sigma_minor: float) -> float:
    """Probability of falling within ``radius`` of the mean point.

    Exact for any aspect ratio: the radial integral is analytic and only
    the angular one is quadrature.
    """
    a, b = check_sigmas(sigma_major, sigma_minor)
    r = float(radius)
    if not (np.isfinite(r) and r >= 0.0):
        raise ValueError(f"radius must be finite and >= 0, got {r}")
    if r == 0.0:
        return 0.0
    if b == 0.0:
        # Degenerate to one dimension: the "disc" is an interval.
        return float(scipy.special.erf(r / (a * np.sqrt(2.0))))

    def integrand(theta: float) -> float:
        coefficient = np.cos(theta) ** 2 / (2.0 * a**2) + np.sin(theta) ** 2 / (2.0 * b**2)
        return float((1.0 - np.exp(-coefficient * r**2)) / (2.0 * coefficient))

    # Fourfold symmetry: integrate a quadrant and scale.
    value, _ = scipy.integrate.quad(integrand, 0.0, 0.5 * np.pi, limit=200)
    return float(4.0 * value / (2.0 * np.pi * a * b))


def containment_radius(
    probability: float, sigma_major: float, sigma_minor: float | None = None
) -> float:
    """Radius (same units as sigma) containing ``probability`` of the scatter.

    ``sigma_minor`` defaults to ``sigma_major``, i.e. the circular case,
    which is then evaluated from the closed form rather than by root-find.
    """
    p = float(probability)
    if not (0.0 < p < 1.0):
        raise ValueError(f"probability must lie in (0, 1), got {p}")
    minor = sigma_major if sigma_minor is None else sigma_minor
    a, b = check_sigmas(sigma_major, minor)
    if a == b:
        return float(a * np.sqrt(-2.0 * np.log(1.0 - p)))
    # Bracket generously: the elliptical radius lies between the circular
    # answers for the two sigmas.
    low = b * np.sqrt(-2.0 * np.log(1.0 - p))
    high = a * np.sqrt(-2.0 * np.log(1.0 - p))
    return float(
        scipy.optimize.brentq(
            lambda r: containment_probability(r, a, b) - p,
            0.5 * low,
            2.0 * high,
            xtol=1e-9 * a,
        )
    )


def containment_ratio(sigma_major: float, sigma_minor: float) -> float:
    """:math:`R_{95}/R_{50}` for this ellipse.

    Equal to 2.079 for a circular distribution and *larger* for an
    elongated one, approaching 2.906 in the one-dimensional limit. Scaling
    a 50% radius by the circular constant therefore under-states the 95%
    radius for any real dispersion, and this is the function that says by
    how much.
    """
    a, b = check_sigmas(sigma_major, sigma_minor)
    return containment_radius(0.95, a, b) / containment_radius(0.5, a, b)
