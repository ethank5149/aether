"""Propagation of burnout errors into impact errors.

Implements the ballistic error coefficients of

* G. M. Siouris, *Missile Guidance and Control Systems* (Springer, 2004),
  §6.4.3, Eq. (6.116) and Figs. 6.14-6.17 — the out-of-plane coefficients;
* F. J. Regan, *Re-Entry Vehicle Dynamics* (AIAA, 1984), §5.5,
  Eqs. (5.36), (5.39) and (5.41) — the in-plane coefficients.

Both are checked against an independent conic solution rather than adopted.
That check earned its keep: **two of Regan's three printed in-plane
equations disagree with the numerics**, in ways specific enough to
diagnose. Eq. (5.41) matches only where one of its terms vanishes, which is
the signature of a dropped outer bracket; Eq. (5.39) matches in magnitude
but carries the sign a range *minimum* would have rather than the maximum
it sits on. Each departure is documented at the function that makes it, so
that nothing here silently differs from its citation.

This module exists to replace stated numbers with derived ones.
:data:`aether_gambit.systems.budget.DISPERSION_SOURCES` says of itself that its
entries "are parametric inputs, not derived results" — order-of-magnitude
figures chosen to exercise the accounting. Ballistic error coefficients are
the derivation: given a perturbation at thrust termination, they say what
it becomes at impact, in closed form.

The assumptions are Siouris's, and they are the free-flight ones: inverse
square gravity, no atmosphere, and a fixed time of flight. Everything here
is therefore about the *exo-atmospheric* mapping from burnout to entry
interface. The atmospheric leg has its own treatment in
:mod:`aether_gambit.flight.ballistic_entry`.

The result worth knowing before any of the algebra
--------------------------------------------------

A lateral displacement of the burnout point does **not** produce a
proportional crossrange miss. Siouris Eq. (6.116a) gives the exact spherical
relation

.. math:: \\cos \\delta C = \\sin^2\\psi + \\cos^2\\psi \\, \\cos \\delta\\chi

with :math:`\\psi` the free-flight range angle, reducing for small angles to

.. math:: \\delta C \\approx \\delta\\chi \\, |\\cos \\psi|.

So the sensitivity is :math:`\\cos\\psi`, and it **vanishes at a range angle
of 90 degrees** — a quarter of the Earth's circumference, about 10,000 km.
At that range a lateral burnout offset produces *no crossrange miss at all*,
exactly, for any offset size. The reason is geometric rather than
approximate: two great circles displaced perpendicular to each other at one
point converge again a quarter turn later, because every pair of great
circles intersects. Past 90 degrees the sensitivity grows again with the
opposite sign.

This is the sort of thing a stated dispersion number cannot express. A
crossrange budget quoted as a fixed metre count is implicitly assuming a
range, and is wrong by an unbounded factor at the wrong one.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "crossrange_from_lateral_offset",
    "crossrange_offset_sensitivity",
    "downrange_per_burnout_altitude",
    "downrange_per_flight_path_angle",
    "downrange_per_velocity",
    "launch_position_error",
    "optimum_flight_path_angle",
    "velocity_error_at_impact",
]

_FloatArray = NDArray[np.float64]


def crossrange_from_lateral_offset(
    offset_angle: ArrayLike,
    range_angle: float,
) -> _FloatArray:
    """Crossrange miss (rad) from a lateral burnout displacement.

    Siouris Eq. (6.116a), exact rather than the small-angle form:

    .. math:: \\cos \\delta C = \\sin^2\\psi + \\cos^2\\psi \\, \\cos \\delta\\chi

    Evaluated through an equivalent half-angle form rather than as printed:

    .. math:: \\delta C = 2\\arcsin\\!\\big(|\\cos\\psi|\\,\\sin(\\delta\\chi/2)\\big)

    The two are algebraically identical — subtract each side of the first
    from one and apply :math:`1-\\cos x = 2\\sin^2(x/2)` — but the printed
    form is **numerically useless for small offsets**, which is the case
    that matters. A crossrange budget cares about metre-scale offsets, i.e.
    :math:`\\delta\\chi \\sim 10^{-7}` rad, where :math:`\\cos\\delta\\chi`
    differs from 1 by :math:`5\\times10^{-15}` — a few times machine epsilon
    — and :math:`\\arccos` near its endpoint has unbounded derivative. Coded
    literally it loses about four significant figures there. The half-angle
    form never forms a quantity near 1 and is exact throughout; it also
    makes the :math:`|\\cos\\psi|` sensitivity manifest rather than
    emergent.

    Verified against direct spherical construction — displace the burnout
    point perpendicular to the trajectory plane, fly the same range angle on
    the same initial heading, and measure the great-circle separation at
    impact — to machine precision across range angles 0 to 150 degrees and
    offsets from 0.01 to 5 degrees.

    Parameters
    ----------
    offset_angle:
        Lateral displacement of the burnout point (rad, as an angle
        subtended at the Earth's centre). Scalar or array. Multiply a linear
        offset by ``1/R_earth`` to get it.
    range_angle:
        Free-flight range angle :math:`\\psi` (rad), burnout to impact.

    Returns
    -------
    numpy.ndarray
        Crossrange miss (rad), non-negative. Multiply by the Earth radius
        for a distance.

    Notes
    -----
    Returned unsigned because the exact relation is even in
    :math:`\\delta\\chi` — the miss is the same magnitude whichever side the
    burnout point is displaced to. The sign is carried by the caller if it
    matters.
    """
    chi = np.atleast_1d(np.asarray(offset_angle, dtype=np.float64))
    if not np.all(np.isfinite(chi)):
        msg = "offset_angle must be finite"
        raise ValueError(msg)
    psi = float(range_angle)
    if not (np.isfinite(psi) and 0.0 <= psi < np.pi):
        msg = f"range_angle must lie in [0, pi), got {psi}"
        raise ValueError(msg)
    half = abs(np.cos(psi)) * np.sin(0.5 * chi)
    return np.asarray(2.0 * np.arcsin(np.clip(half, -1.0, 1.0)))


def crossrange_offset_sensitivity(range_angle: float) -> float:
    """:math:`|\\cos\\psi|`, the small-angle crossrange sensitivity.

    The derivative of :func:`crossrange_from_lateral_offset` at zero offset.
    Reported separately because it is the number a dispersion budget wants —
    "a metre of lateral burnout error becomes this many metres at impact" —
    and because its **zero at 90 degrees** is the single most surprising
    entry in the whole error budget.

    Parameters
    ----------
    range_angle:
        Free-flight range angle (rad).

    Returns
    -------
    float
        Dimensionless sensitivity in :math:`[0, 1]`.
    """
    psi = float(range_angle)
    if not (np.isfinite(psi) and 0.0 <= psi < np.pi):
        msg = f"range_angle must lie in [0, pi), got {psi}"
        raise ValueError(msg)
    return float(abs(np.cos(psi)))


def velocity_error_at_impact(
    velocity_error: float,
    time_of_flight: float,
) -> float:
    """Impact displacement (m) from a burnout velocity error, ``dV * t_ff``.

    Siouris Fig. 6.16: a lateral velocity error at thrust termination simply
    integrates over the free-flight time, because nothing acts to correct it
    and gravity is (to first order) parallel over the displacement.

    This is the same structure the mission budget already used for its
    crossrange entry, where the velocity error was multiplied by a transfer
    sensitivity of 850 s obtained from our own propagator. That agreement is
    worth stating: an independent text gives the same first-order law, so
    the budget's crossrange mapping was right in form even while its inputs
    were stated rather than derived.

    Parameters
    ----------
    velocity_error:
        One-sigma velocity error at burnout (m/s).
    time_of_flight:
        Free-flight time (s).

    Returns
    -------
    float
        Impact displacement (m).
    """
    dv, tof = float(velocity_error), float(time_of_flight)
    for name, value in (("velocity_error", dv), ("time_of_flight", tof)):
        if not (np.isfinite(value) and value >= 0.0):
            msg = f"{name} must be finite and >= 0, got {value}"
            raise ValueError(msg)
    return dv * tof


def launch_position_error(
    latitude_error: float,
    longitude_error: float,
    latitude: float,
    bearing: float,
    earth_radius: float = 6378137.0,
) -> tuple[float, float]:
    """Resolve a launch-site survey error into downrange and crossrange (m).

    Siouris Fig. 6.17. A latitude error displaces the launch point north by
    :math:`R_e\\,\\delta L`; a longitude error displaces it east by
    :math:`R_e\\cos L\\,\\delta\\lambda`. Both then project onto the
    trajectory's bearing:

    .. math::

        \\delta \\mathrm{DR} &= N \\cos B + E \\sin B \\\\
        \\delta \\mathrm{CR} &= -N \\sin B + E \\cos B

    This is the term that makes a 10 m CEP a *survey* problem rather than a
    guidance one: the error enters the impact point undiminished, with no
    range-dependent suppression of the kind :func:`crossrange_offset_sensitivity`
    provides for a lateral burnout offset, because it is a displacement of
    the whole trajectory rather than of one point on it.

    Parameters
    ----------
    latitude_error, longitude_error:
        Survey errors (rad).
    latitude:
        Launch-site geodetic latitude (rad), which sets the longitude
        error's linear size through :math:`\\cos L`. At high latitude a
        given longitude error matters less; at the pole, not at all.
    bearing:
        Launch azimuth (rad from north, positive east).
    earth_radius:
        Sphere radius (m).

    Returns
    -------
    tuple[float, float]
        ``(downrange, crossrange)`` displacement (m), signed. Crossrange is
        positive to the right of the bearing.

    Notes
    -----
    The latitude terms reproduce Siouris Fig. 6.17 exactly. His longitude
    *crossrange* term carries the opposite sign to the rotation above; the
    archived scan does not settle whether that is a different crossrange
    convention (positive-left) or a transcription artefact, and rather than
    guess, this function implements the rotation, which is derivable and
    self-consistent. A caller comparing against the figure should check the
    sign convention rather than assume it.
    """
    for name, value in (
        ("latitude_error", latitude_error),
        ("longitude_error", longitude_error),
        ("latitude", latitude),
        ("bearing", bearing),
    ):
        if not np.isfinite(value):
            msg = f"{name} must be finite, got {value}"
            raise ValueError(msg)
    if not (np.isfinite(earth_radius) and earth_radius > 0.0):
        msg = f"earth_radius must be finite and > 0, got {earth_radius}"
        raise ValueError(msg)
    if abs(latitude) > 0.5 * np.pi:
        msg = f"latitude must lie in [-pi/2, pi/2], got {latitude}"
        raise ValueError(msg)

    north = earth_radius * float(latitude_error)
    east = earth_radius * np.cos(float(latitude)) * float(longitude_error)
    downrange = north * np.cos(bearing) + east * np.sin(bearing)
    crossrange = -north * np.sin(bearing) + east * np.cos(bearing)
    return float(downrange), float(crossrange)


def optimum_flight_path_angle(range_angle: float) -> float:
    """Burnout flight-path angle minimising the speed for a given range.

    Regan Eq. (5.25), :math:`\\gamma^* = \\pi/4 - \\theta_i/4`, measured
    above the local horizontal. Equivalently the angle that *maximises*
    range at fixed speed, by the standard duality.

    Its importance to an error budget is not the fuel it saves. **The
    downrange sensitivity to flight-path-angle error is exactly zero at
    :math:`\\gamma^*`, for every range angle** — see
    :func:`downrange_per_flight_path_angle`. The minimum-energy trajectory
    is also the one indifferent to how precisely the boost pitch programme
    is flown, which is a rare case of two design pressures pointing the
    same way.
    """
    theta = float(range_angle)
    if not (np.isfinite(theta) and 0.0 < theta < np.pi):
        msg = f"range_angle must lie in (0, pi), got {theta}"
        raise ValueError(msg)
    return 0.25 * np.pi - 0.25 * theta


def _check_burnout(range_angle: float, flight_path_angle: float) -> tuple[float, float]:
    theta, gamma = float(range_angle), float(flight_path_angle)
    if not (np.isfinite(theta) and 0.0 < theta < np.pi):
        msg = f"range_angle must lie in (0, pi), got {theta}"
        raise ValueError(msg)
    if not (np.isfinite(gamma) and 0.0 < gamma < 0.5 * np.pi):
        msg = f"flight_path_angle must lie in (0, pi/2), got {gamma}"
        raise ValueError(msg)
    return theta, gamma


def downrange_per_velocity(
    range_angle: float,
    flight_path_angle: float,
    burnout_speed: float,
    earth_radius: float = 6378137.0,
) -> float:
    """:math:`\\partial R/\\partial V` (m per m/s) — Regan Eq. (5.36).

    .. math::

        \\frac{\\partial R}{\\partial V}
          = \\frac{2R_E}{V}\\big[\\sin\\theta_i
            + \\cot\\gamma\\,(1 - \\cos\\theta_i)\\big]

    **Verified exactly** against finite differences of an independent conic
    solution: the burnout state is converted to a Keplerian orbit and the
    range angle taken as the difference of true anomalies, with no algebra
    from this equation involved. Agreement is to five figures across range
    angles 30-90 degrees and flight-path angles 15-30 degrees, and the
    worked example reproduces: at :math:`\\theta_i = 90°` the optimum
    :math:`\\gamma^* = 22.5°` needs :math:`V = 7195` m/s, giving **6.05 km
    per m/s** — Regan states 7195 m/s and "approximately 6 km/(m/s)".

    That number is the reason boost cutoff dominates a ballistic error
    budget: a **1 m/s error in 7195 becomes a 6 km miss**.
    """
    theta, gamma = _check_burnout(range_angle, flight_path_angle)
    speed = float(burnout_speed)
    if not (np.isfinite(speed) and speed > 0.0):
        msg = f"burnout_speed must be finite and > 0, got {speed}"
        raise ValueError(msg)
    bracket = np.sin(theta) + (1.0 - np.cos(theta)) / np.tan(gamma)
    return float(2.0 * earth_radius * bracket / speed)


def downrange_per_flight_path_angle(
    range_angle: float,
    flight_path_angle: float,
    earth_radius: float = 6378137.0,
) -> float:
    """:math:`\\partial R/\\partial\\gamma` (m per rad) — Regan Eq. (5.39),
    **with its sign corrected**.

    .. math::

        \\frac{\\partial R}{\\partial\\gamma}
          = 2R_E\\left[\\frac{\\sin(\\theta_i + 2\\gamma)}{\\sin 2\\gamma} - 1\\right]

    Zero at :math:`\\gamma = \\gamma^*` for every range angle, which is the
    structural content of the equation: substituting
    :math:`\\gamma^* = \\pi/4 - \\theta_i/4` makes
    :math:`\\theta_i + 2\\gamma^* = \\pi/2 + \\theta_i/2` and
    :math:`2\\gamma^* = \\pi/2 - \\theta_i/2`, whose sines are both
    :math:`\\cos(\\theta_i/2)`. Positive below :math:`\\gamma^*` and
    negative above, as a range maximum requires.

    Notes
    -----
    Regan prints this with the bracket the other way round, and his prose
    agrees with the printed form ("for :math:`\\gamma < \\gamma^*`,
    :math:`\\delta R/\\delta\\gamma` is negative"). **The printed form has
    been confirmed against the book itself**, so this is not a transcription
    artefact.

    Finite differences of the independent conic solution give the
    **opposite sign**, matching the form above. The decisive check is not
    the derivative but the extremum: scanning :math:`\\theta_i(\\gamma)` at
    fixed speed puts the maximum at :math:`\\pi/4 - \\theta_i/4` to within
    0.01 degrees, so :math:`\\gamma^*` genuinely maximises range and the
    derivative below it must be positive. The magnitude agrees to five
    figures either way, so only the sign is at issue.

    Where the sign enters cannot be located from the text: it should follow
    from Regan Eq. (5.33), but the bracket multiplying
    :math:`\\delta\\gamma` there does not reproduce the numerics under any
    reading we can recover, including at :math:`\\gamma^*` where it must
    vanish and does not. So the disagreement is recorded rather than
    diagnosed. It may still be a convention — an error reported as "short"
    rather than as a signed displacement — but nothing in the surrounding
    text says so.

    A separate discrepancy is left unresolved because it cannot be settled
    from the text: Regan's worked example Eq. (5.40) states **-5.28
    km/mrad** at :math:`\\theta_i = 75°, \\gamma = 15°`, while his own
    Eq. (5.39) at those angles gives **11.89 km/mrad** — a factor of 2.25,
    and the value this function returns (verified against finite
    differences) is the latter.
    """
    theta, gamma = _check_burnout(range_angle, flight_path_angle)
    return float(2.0 * earth_radius * (np.sin(theta + 2.0 * gamma) / np.sin(2.0 * gamma) - 1.0))


def downrange_per_burnout_altitude(
    range_angle: float,
    flight_path_angle: float,
) -> float:
    """:math:`\\partial R/\\partial h`, dimensionless — Regan Eq. (5.41),
    **with a dropped bracket restored**.

    .. math::

        \\frac{\\partial R}{\\partial h}
          = \\cot\\gamma\\left[2 - \\frac{\\cos(\\gamma + \\theta_i)}{\\cos\\gamma}\\right]

    Regan prints
    :math:`2\\cot\\gamma - \\cos(\\gamma+\\theta_i)/\\cos\\gamma`, which is
    the same thing only when :math:`\\cos(\\gamma+\\theta_i) = 0`. That
    printed form has been **confirmed against the book**, so it is his and
    not a transcription artefact — and it is **demonstrably inconsistent
    with his own Eq. (5.33)**, which is the total differential everything in
    his §5.5 descends from.

    Setting :math:`\\delta V = \\delta\\gamma = 0` in Eq. (5.33) gives
    :math:`\\partial R/\\partial h = (1 + A)/C` with
    :math:`A = (1-\\cos\\theta_i)/(\\lambda\\cos^2\\gamma)` and
    :math:`C` the bracket on :math:`\\delta\\theta_i`. Two facts close it.
    The impact equation makes
    :math:`A = 1 - \\cos(\\gamma+\\theta_i)/\\cos\\gamma`, so the numerator
    is exactly the bracket above; and :math:`C = \\tan\\gamma` identically on
    the impact locus, verified to six decimals. Hence
    :math:`(1+A)/C = \\cot\\gamma\\,[2 - \\cos(\\gamma+\\theta_i)/\\cos\\gamma]`
    — the form used here. It agrees with finite differences to five decimals
    at every angle pair tried, whereas the printed form agrees only where
    :math:`\\cos(\\gamma+\\theta_i)=0`. The slip is in substituting
    :math:`C = \\tan\\gamma` into the first term of the numerator but not
    the second.

    The consequence is not academic. At Regan's own worked point
    (:math:`\\gamma = 22.5°, \\theta_i = 90°`) the printed form gives
    **5.24** and the corrected form **5.83**, an 11 % understatement of how
    much a burnout altitude error costs; at :math:`\\theta_i = 150°,
    \\gamma = 10°` the gap widens to 12.3 against 16.8, a third.

    Being dimensionless, this coefficient reads directly: **a kilometre of
    burnout altitude error is worth about six kilometres of range** at
    typical angles. It is the one in-plane coefficient that does not depend
    on burnout speed.
    """
    theta, gamma = _check_burnout(range_angle, flight_path_angle)
    return float((2.0 - np.cos(gamma + theta) / np.cos(gamma)) / np.tan(gamma))
