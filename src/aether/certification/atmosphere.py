r"""Rigorous density enclosures for a tabulated atmosphere.

A certificate that mentions drag needs a bound on density over a *range* of
altitudes, not a value at one. Interpolating the table at the endpoints and
calling the pair an interval is only sound if nothing in between leaves it,
which for density means: monotone.

That is a property of the model, and it is checked rather than assumed.
:func:`monotone_density` verifies that the tabulated log-density decreases at
every sample before returning anything, and the PCHIP interpolant
:class:`~aether.atmosphere.model.TabulatedAtmosphere` uses is shape-preserving,
so monotone samples give a monotone interpolant -- which is why this argument
works here and would not work for a cubic spline.

What is bounded is the *model*, not the sky. The enclosure is exact with
respect to the tabulated atmosphere, and the tabulated atmosphere is itself an
approximation with its own error, which a certificate has to carry separately:
Artemis I met air about 10 % thinner than the 1976 standard. A density scale
factor covering that is an interval multiplying this one, which is how a bound
is made to hold over a *class* of atmospheres rather than one.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from aether.atmosphere.model import TabulatedAtmosphere
from aether.certification.rigorous import interval, lower_bound

__all__ = ["monotone_density"]


def monotone_density(
    table: TabulatedAtmosphere, *, scale: tuple[float, float] = (1.0, 1.0)
) -> Callable[[Any], Any]:
    """Density enclosure over an altitude interval, from a monotone table.

    Returns ``density(altitude)`` taking an altitude (a float or an Arb ball)
    and returning the density (kg/m³) enclosed over it. ``scale`` multiplies the
    result by an interval, for carrying a class of atmospheres rather than the
    tabulated one alone.

    Raises if the table's density is not strictly decreasing with altitude,
    since the endpoint argument would then be unsound.
    """
    log_density = np.asarray(table.log_density, dtype=np.float64)
    if not np.all(np.diff(log_density) < 0.0):
        raise ValueError(
            "the tabulated density is not strictly decreasing with altitude, so "
            "bounding it by its values at the ends of an interval is not sound"
        )
    low_scale, high_scale = float(scale[0]), float(scale[1])
    if not 0.0 < low_scale <= high_scale:
        raise ValueError(f"require 0 < low <= high scale, got {scale}")
    floor = float(table.floor)

    def density(altitude: Any) -> Any:
        low = max(lower_bound(altitude), floor)
        high = max(-lower_bound(-altitude), floor)
        # Monotone decreasing: the thickest air is at the bottom of the band.
        thickest = float(table.state(low).density)
        thinnest = float(table.state(high).density)
        return interval(thinnest * low_scale, thickest * high_scale)

    return density
