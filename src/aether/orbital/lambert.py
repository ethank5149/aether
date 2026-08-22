"""Lambert's problem: the two-body boundary-value solve.

Given two position vectors and the time of flight between them, find the
conic that connects them. This is the primitive every midcourse correction
is built on — a correction maneuver is the difference between the velocity
you have and the velocity Lambert says you need to arrive where you want,
when you want.

Izzo's formulation, not the classical one
-----------------------------------------

The textbook universal-variable method (Bate, Mueller & White; Vallado)
brackets a root in the universal anomaly and iterates. It works, but its
bracket degenerates near the parabolic transfer and near a half-revolution
transfer angle, and it needs case analysis to pick a starting guess that
converges.

This module implements Izzo's 2015 reformulation instead
(*Celestial Mechanics and Dynamical Astronomy* **121**, 1–15). It solves for
a single variable :math:`x` related to the transfer semi-major axis by

.. math::

    x = \\cos\\frac{\\Delta E}{2} \\;\\text{(ellipse)}, \\qquad
    x = \\cosh\\frac{\\Delta H}{2} \\;\\text{(hyperbola)},

on which the time of flight is *monotonic* over ``x in (-1, inf)`` for the
zero-revolution case. Monotonicity is the whole point: it means a
Householder iteration from a cheap initial guess converges without
bracketing, without case analysis, and without the parabolic singularity.
The transfer is parabolic at exactly ``x = 1``, which the time-of-flight
expression handles by a series rather than by a branch.

What is verified here rather than asserted
------------------------------------------

The strongest available check needs no external data: solve Lambert, then
propagate the resulting velocity through an *independent* integrator for
the same time of flight and confirm it arrives at the target position. That
closes the loop through code that shares nothing with the solver, and it is
what the tests do. Analytic special cases (the Hohmann transfer, a circular
arc) pin the absolute scale.

Terminology
-----------

The **transfer angle** is measured from :math:`\\mathbf{r}_1` to
:math:`\\mathbf{r}_2` in the direction of motion. ``prograde`` selects which
of the two possible sweeps that is, by reference to the :math:`+z` axis; a
transfer through more than :math:`\\pi` is the "long way" and is a
legitimate answer, not an error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aether.orbital.gravity import EARTH, GravityModel

__all__ = [
    "LambertSolution",
    "lambert",
    "minimum_energy_transfer",
]

_FloatArray = NDArray[np.float64]

#: Householder iteration is cubically convergent; five passes is generous.
_MAX_ITERATIONS = 32
_TOLERANCE = 1e-13


@dataclass(frozen=True)
class LambertSolution:
    """A conic arc joining two positions in a specified time.

    Attributes
    ----------
    v1, v2:
        Velocity (m/s) at the departure and arrival positions.
    semi_major_axis:
        Transfer semi-major axis (m). Negative for a hyperbolic transfer,
        and infinite for the parabolic case.
    transfer_angle:
        Swept true anomaly (rad), in ``(0, 2*pi)``.
    iterations:
        Householder passes actually taken, for diagnostics.
    n_revolutions:
        Complete revolutions included in the transfer.
    """

    v1: _FloatArray
    v2: _FloatArray
    semi_major_axis: float
    transfer_angle: float
    iterations: int
    n_revolutions: int = 0

    @property
    def is_hyperbolic(self) -> bool:
        return self.semi_major_axis < 0.0


def _tof_from_x(x: float, lam: float, n_rev: int) -> float:
    """Non-dimensional time of flight as a function of ``x``.

    Three regimes, and the split is about conditioning rather than about
    physics. The general closed form divides by :math:`x^2 - 1`, which
    vanishes at the parabolic transfer, so within ``|x - 1| < 0.01`` it is
    replaced by Battin's series in the hypergeometric function — the same
    quantity, evaluated without the cancellation. Between the two a
    Lagrange form in the eccentric-anomaly differences is better
    conditioned than either.
    """
    battin, lagrange = 0.01, 0.2
    dist = abs(x - 1.0)
    if battin < dist < lagrange:
        a = 1.0 / (1.0 - x * x)
        if a > 0.0:
            alpha = 2.0 * np.arccos(np.clip(x, -1.0, 1.0))
            beta = 2.0 * np.arcsin(np.clip(np.sqrt(lam * lam / a), -1.0, 1.0))
            if lam < 0.0:
                beta = -beta
            return float(
                a
                * np.sqrt(a)
                * ((alpha - np.sin(alpha)) - (beta - np.sin(beta)) + 2.0 * np.pi * n_rev)
                / 2.0
            )
        alpha = 2.0 * np.arccosh(x)
        beta = 2.0 * np.arcsinh(np.sqrt(-lam * lam / a))
        if lam < 0.0:
            beta = -beta
        return float(-a * np.sqrt(-a) * ((beta - np.sinh(beta)) - (alpha - np.sinh(alpha))) / 2.0)

    k = lam * lam
    e = x * x - 1.0
    rho = abs(e)
    z = np.sqrt(1.0 + k * e)
    if dist < battin:
        eta = z - lam * x
        s1 = 0.5 * (1.0 - lam - x * eta)
        q = 4.0 / 3.0 * _hypergeometric_2f1(3.0, 1.0, 2.5, s1)
        return float((eta**3 * q + 4.0 * lam * eta) / 2.0 + n_rev * np.pi / rho**1.5)
    # (z - lam*x) and (x*z - lam*e) both differ two quantities that grow
    # like lam*x^2 on a strongly hyperbolic transfer, so forming them
    # directly loses every significant digit once x is large -- the log
    # below then sees a negative argument. Rationalising removes the
    # cancellation exactly: z^2 - (lam x)^2 = 1 - lam^2 identically.
    z_minus = z - lam * x
    if lam > 0.0 and z + lam * x > 0.0:
        z_minus = (1.0 - lam * lam) / (z + lam * x)
    g = x * z_minus + lam
    y = np.sqrt(rho)
    if e < 0.0:
        d = n_rev * np.pi + np.arccos(np.clip(g, -1.0, 1.0))
    else:
        d = np.log(max(y * z_minus + g, np.finfo(float).tiny))
    return float((x - lam * z - d / y) / e)


def _eta(x: float, lam: float) -> float:
    y = np.sqrt(1.0 - lam * lam * (1.0 - x * x))
    return float(y - lam * x)


def _hypergeometric_2f1(a: float, b: float, c: float, z: float) -> float:
    """:math:`{}_2F_1(a,b;c;z)` by its defining series.

    Only ever called with ``|z| < 1`` in the near-parabolic branch, where
    the series converges quickly; a general implementation is not needed
    and would not be exercised.
    """
    if z >= 1.0:
        return float("inf")
    total = 1.0
    term = 1.0
    for j in range(256):
        term *= (a + j) * (b + j) / (c + j) * z / (j + 1.0)
        new_total = total + term
        if new_total == total:
            break
        total = new_total
    return float(total)


def _tof_derivatives(x: float, tof: float, lam: float, n_rev: int) -> tuple[float, float, float]:
    """First three derivatives of time of flight in ``x`` (Izzo Eq. 22)."""
    um_x2 = 1.0 - x * x
    y = np.sqrt(1.0 - lam * lam * um_x2)
    y3, y5 = y**3, y**5
    d1 = (3.0 * tof * x - 2.0 + 2.0 * lam**3 * x / y) / um_x2
    d2 = (3.0 * tof + 5.0 * x * d1 + 2.0 * (1.0 - lam * lam) * lam**3 / y3) / um_x2
    d3 = (7.0 * x * d2 + 8.0 * d1 - 6.0 * (1.0 - lam * lam) * lam**5 * x / y5) / um_x2
    return float(d1), float(d2), float(d3)


def _initial_guess(tof: float, lam: float) -> float:
    """Izzo Eq. (30): a starting ``x`` accurate enough for Householder."""
    t0 = np.arccos(lam) + lam * np.sqrt(1.0 - lam * lam)  # x = 0
    t1 = 2.0 / 3.0 * (1.0 - lam**3)  # x = 1, parabolic
    if tof >= t0:
        return float((t0 / tof) ** (2.0 / 3.0) - 1.0)
    if tof <= t1:
        return float(2.5 * t1 * (t1 - tof) / (tof * (1.0 - lam**5)) + 1.0)
    return float((t0 / tof) ** (np.log2(t1 / t0)) - 1.0)


def lambert(
    r1: ArrayLike,
    r2: ArrayLike,
    time_of_flight: float,
    model: GravityModel = EARTH,
    prograde: bool = True,
    n_revolutions: int = 0,
) -> LambertSolution:
    """Solve Lambert's problem for the zero-revolution transfer.

    Parameters
    ----------
    r1, r2:
        Departure and arrival position vectors (m), in an inertial frame
        centred on the attracting body.
    time_of_flight:
        Arc duration (s), strictly positive.
    prograde:
        ``True`` takes the transfer sweeping in the ``+z`` sense. This
        selects between the short and long way round; it is a statement
        about the intended direction of motion, not a solver hint.
    n_revolutions:
        Complete revolutions before arrival. Only ``0`` is supported;
        multi-revolution transfers have two solutions per revolution and
        require selecting a branch, which this does not do.

    Raises
    ------
    ValueError
        If the positions are collinear, which leaves the transfer plane
        undefined and admits no unique conic; if the time of flight is not
        positive; or if a nonzero revolution count is requested.
    """
    p1 = np.asarray(r1, dtype=np.float64)
    p2 = np.asarray(r2, dtype=np.float64)
    if p1.shape != (3,) or p2.shape != (3,):
        raise ValueError("r1 and r2 must both be 3-vectors")
    if not np.isfinite(p1).all() or not np.isfinite(p2).all():
        raise ValueError("r1 and r2 must be finite")
    tof = float(time_of_flight)
    if not (np.isfinite(tof) and tof > 0.0):
        raise ValueError(f"time_of_flight must be finite and > 0, got {tof}")
    if n_revolutions != 0:
        raise ValueError(
            f"only the zero-revolution transfer is implemented, got "
            f"n_revolutions={n_revolutions}; a multi-revolution solve has two "
            f"branches per revolution and needs an explicit branch choice"
        )

    norm1 = float(np.linalg.norm(p1))
    norm2 = float(np.linalg.norm(p2))
    if norm1 == 0.0 or norm2 == 0.0:
        raise ValueError("position vectors must be non-zero")

    chord_vec = p2 - p1
    chord = float(np.linalg.norm(chord_vec))
    semiperimeter = 0.5 * (norm1 + norm2 + chord)

    cross = np.cross(p1, p2)
    cross_norm = float(np.linalg.norm(cross))
    # Collinear endpoints leave the plane undefined. This is a genuine
    # degeneracy of the problem rather than a numerical difficulty: the
    # transfer could lie in any plane containing the common line.
    if cross_norm <= 1e-12 * norm1 * norm2:
        raise ValueError(
            "r1 and r2 are collinear, so the transfer plane is undefined; "
            "Lambert's problem has no unique solution for this geometry"
        )

    cos_dnu = float(np.dot(p1, p2) / (norm1 * norm2))
    lam = float(np.sqrt(max(0.0, 1.0 - chord / semiperimeter)))

    unit_r1 = p1 / norm1
    unit_r2 = p2 / norm2
    unit_h = cross / cross_norm

    # The sign of lambda is the long-way/short-way switch, and the tangential
    # basis vectors carry the sense of motion. Both flip together when the
    # transfer sweeps more than pi, and both flip again for a retrograde
    # transfer -- so a retrograde long-way transfer returns to lambda > 0,
    # which is correct and is the reason this is written as two independent
    # flips rather than a four-way case analysis.
    transfer_angle = float(np.arccos(np.clip(cos_dnu, -1.0, 1.0)))
    if unit_h[2] < 0.0:
        transfer_angle = 2.0 * np.pi - transfer_angle
        lam = -lam
        unit_t1 = np.cross(unit_r1, unit_h)
        unit_t2 = np.cross(unit_r2, unit_h)
    else:
        unit_t1 = np.cross(unit_h, unit_r1)
        unit_t2 = np.cross(unit_h, unit_r2)
    if not prograde:
        transfer_angle = 2.0 * np.pi - transfer_angle
        lam = -lam
        unit_t1 = -unit_t1
        unit_t2 = -unit_t2

    # Non-dimensional time, Izzo Eq. (5).
    scale = np.sqrt(2.0 * model.mu / semiperimeter**3)
    tau = tof * scale

    x = _initial_guess(tau, lam)
    iterations = 0
    for _ in range(_MAX_ITERATIONS):
        iterations += 1
        residual = _tof_from_x(x, lam, n_revolutions) - tau
        if abs(residual) < _TOLERANCE * max(1.0, tau):
            break
        d1, d2, d3 = _tof_derivatives(x, residual + tau, lam, n_revolutions)
        if d1 == 0.0:  # pragma: no cover - stationary point
            break
        # Householder's third-order update.
        # Householder's third order. The numerator and denominator are
        # *different* combinations of the same derivatives; collapsing them
        # into one expression silently degrades the method to something that
        # does not converge at all.
        numerator = d1 * d1 - residual * d2 / 2.0
        denominator = d1 * (d1 * d1 - residual * d2) + d3 * residual * residual / 6.0
        if denominator == 0.0:  # pragma: no cover - defensive
            break
        step = -residual * numerator / denominator
        if not np.isfinite(step):  # pragma: no cover - defensive
            break
        x = float(x + step)
    else:  # pragma: no cover - Householder is cubic; this should not fire
        raise RuntimeError(
            f"Lambert iteration did not converge in {_MAX_ITERATIONS} passes "
            f"for lambda={lam:.6g}, non-dimensional time={tau:.6g}"
        )

    # Rebuild the terminal velocities, Izzo Eq. (35)-(36).
    gamma = np.sqrt(model.mu * semiperimeter / 2.0)
    rho = (norm1 - norm2) / chord
    sigma = np.sqrt(max(0.0, 1.0 - rho * rho))
    y = np.sqrt(1.0 - lam * lam * (1.0 - x * x))

    v_r1 = gamma * ((lam * y - x) - rho * (lam * y + x)) / norm1
    v_r2 = -gamma * ((lam * y - x) + rho * (lam * y + x)) / norm2
    v_t1 = gamma * sigma * (y + lam * x) / norm1
    v_t2 = gamma * sigma * (y + lam * x) / norm2

    v1 = v_r1 * unit_r1 + v_t1 * unit_t1
    v2 = v_r2 * unit_r2 + v_t2 * unit_t2

    sma = float("inf") if abs(x) == 1.0 else float(semiperimeter / (2.0 * (1.0 - x * x)))

    return LambertSolution(
        v1=np.asarray(v1, dtype=np.float64),
        v2=np.asarray(v2, dtype=np.float64),
        semi_major_axis=sma,
        transfer_angle=transfer_angle,
        iterations=iterations,
        n_revolutions=n_revolutions,
    )


def minimum_energy_transfer(
    r1: ArrayLike, r2: ArrayLike, model: GravityModel = EARTH
) -> tuple[float, float]:
    """Semi-major axis (m) and time of flight (s) of the minimum-energy arc.

    A useful bound rather than a maneuver: no ballistic transfer between
    these endpoints exists with a smaller semi-major axis, so
    :math:`a_{\\min} = s/2` sets the floor on transfer energy and its time
    of flight marks where the short-way and long-way branches meet.
    """
    p1 = np.asarray(r1, dtype=np.float64)
    p2 = np.asarray(r2, dtype=np.float64)
    if p1.shape != (3,) or p2.shape != (3,):
        raise ValueError("r1 and r2 must both be 3-vectors")
    norm1 = float(np.linalg.norm(p1))
    norm2 = float(np.linalg.norm(p2))
    chord = float(np.linalg.norm(p2 - p1))
    semiperimeter = 0.5 * (norm1 + norm2 + chord)
    sma = 0.5 * semiperimeter
    alpha = np.pi
    beta = 2.0 * np.arcsin(np.clip(np.sqrt((semiperimeter - chord) / semiperimeter), -1.0, 1.0))
    tof = np.sqrt(sma**3 / model.mu) * (alpha - beta + np.sin(beta))
    return float(sma), float(tof)
