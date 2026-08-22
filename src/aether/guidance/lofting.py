"""Depressed and lofted ballistic trajectories, and the trade between them.

Implements F. J. Regan, *Re-Entry Vehicle Dynamics* (AIAA, 1984), §5.3-§5.5:
the impact equation Eq. (5.23a), the minimum-energy burnout angle Eq. (5.25),
and the two-solution structure that follows from them.

The structure, which is exact
-----------------------------

For a given free-flight range angle :math:`\\theta_i`, the required burnout
speed depends on the burnout flight-path angle as

.. math::

    V^2 = \\frac{\\mu}{r_0}\\,
          \\frac{1-\\cos\\theta_i}
               {\\sin(\\theta_i/2)\\,[\\sin(2\\gamma + \\theta_i/2) + \\sin(\\theta_i/2)]}

(rearranging Eq. 5.23a at :math:`r_0 = R_E`). Everything that depends on
:math:`\\gamma` sits inside :math:`\\sin(2\\gamma + \\theta_i/2)`. Two
consequences follow immediately and exactly:

* The speed is **minimised** where that sine is one, i.e. at
  :math:`\\gamma^* = \\pi/4 - \\theta_i/4` — Regan's Eq. (5.25).
* The sine is symmetric about its maximum, so **every** achievable speed
  above the minimum is reached at *two* burnout angles, placed symmetrically
  about :math:`\\gamma^*`:

  .. math:: \\gamma_{\\text{over}} = 2\\gamma^* - \\gamma_{\\text{under}}

  The lower is the **depressed** (Regan: "under-lofted") solution, the
  higher the **lofted** ("over-lofted") one. No root-finding is needed to
  pass between them, which is worth stating because the standard treatment
  reads them off a plot.

Why a launch would choose one over the other
--------------------------------------------

The two conjugates cost identical energy and reach the same target, so the
choice is made on everything else, and the three considerations do not agree:

* **Time of flight.** The lofted solution takes far longer — for Regan's
  worked case at a 75 degree range angle, 2577 s against 1271 s, almost
  exactly double. A depressed trajectory is the one flown to compress
  warning time; a lofted one is the opposite.
* **Accuracy.** :math:`\\partial R/\\partial V` is **not** symmetric between
  the conjugates: at a 75 degree range angle it runs 3128 m per m/s lofted,
  4579 at the optimum and 9110 depressed. Regan's conclusion — "the
  over-lofted is better than either the minimum velocity or the
  under-lofted" — holds. The tempting symmetric reading of it does not: the
  **depressed solution is twice as sensitive as the minimum-energy one**,
  which matters because depression is exactly what a short-warning launch
  wants.

  His stated mechanism is also the minor one. He attributes the advantage
  to the higher burnout speed, since :math:`\\partial R/\\partial V` carries
  :math:`1/V` — but that speed penalty is only **5.2 %** across the pair,
  while :math:`\\cot\\gamma`, the other factor, falls by **5.2x** from the
  depressed angle to the lofted one. The ordering is essentially all
  cotangent.
* **Flight-path-angle sensitivity.** This one points the other way and is
  the sharpest of the three: :math:`\\partial R/\\partial\\gamma` is
  **exactly zero at** :math:`\\gamma^*` and grows on either side, so the
  minimum-energy trajectory is the one indifferent to boost pitch error and
  both conjugates give that up.

So minimum energy buys insensitivity to pitch error, lofting buys
insensitivity to speed error, and depression buys time. There is no
trajectory that wins all three, and this module exists to price the choice
rather than to make it.

Verification
------------

Regan's worked comparison (Table 5.3 output, range angle 75 degrees, burnout
speed 7238.03 m/s) gives 1271.18 s at :math:`\\gamma = 10°` and 2577.69 s at
:math:`\\gamma = 42.5°`. Solving Kepler's equation independently gives
**1270.35 s** and **2576.85 s** — 0.07 % and 0.03 %. The conjugate relation
reproduces his pair exactly: :math:`2(26.25°) - 10° = 42.5°`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "LoftingTrade",
    "burnout_speed_for_range",
    "conjugate_flight_path_angle",
    "free_flight_time",
    "lofting_trade",
    "minimum_burnout_speed",
    "optimum_burnout_angle",
]

_MU_EARTH = 3.986004418e14
_R_EARTH = 6378137.0


def _check(range_angle: float, flight_path_angle: float) -> tuple[float, float]:
    theta, gamma = float(range_angle), float(flight_path_angle)
    if not (np.isfinite(theta) and 0.0 < theta < np.pi):
        msg = f"range_angle must lie in (0, pi), got {theta}"
        raise ValueError(msg)
    if not (np.isfinite(gamma) and 0.0 < gamma < 0.5 * np.pi):
        msg = f"flight_path_angle must lie in (0, pi/2), got {gamma}"
        raise ValueError(msg)
    return theta, gamma


def burnout_speed_for_range(
    range_angle: float,
    flight_path_angle: float,
    gravitational_parameter: float = _MU_EARTH,
    radius: float = _R_EARTH,
) -> float:
    """Burnout speed (m/s) reaching ``range_angle`` — Regan Eq. (5.23a).

    Written in the half-angle form derived in the module docstring, which
    isolates the whole :math:`\\gamma` dependence in one sine and makes both
    the minimum and the conjugate pairing manifest.

    Raises
    ------
    ValueError
        If the geometry admits no ballistic solution — the denominator
        turns non-positive — which happens for a shallow burnout at long
        range.
    """
    theta, gamma = _check(range_angle, flight_path_angle)
    half = 0.5 * theta
    denominator = np.sin(half) * (np.sin(2.0 * gamma + half) + np.sin(half))
    if denominator <= 0.0:
        msg = (
            f"no ballistic solution at range angle {np.rad2deg(theta):.2f} deg "
            f"and burnout angle {np.rad2deg(gamma):.2f} deg"
        )
        raise ValueError(msg)
    return float(np.sqrt(gravitational_parameter / radius * (1.0 - np.cos(theta)) / denominator))


def optimum_burnout_angle(range_angle: float) -> float:
    """:math:`\\gamma^* = \\pi/4 - \\theta_i/4`, Regan Eq. (5.25).

    Duplicated from :func:`aether.guidance.ballistic_errors.optimum_flight_path_angle`
    so this module reads as a unit; both return the same value and a test
    pins them together.
    """
    theta = float(range_angle)
    if not (np.isfinite(theta) and 0.0 < theta < np.pi):
        msg = f"range_angle must lie in (0, pi), got {theta}"
        raise ValueError(msg)
    return 0.25 * np.pi - 0.25 * theta


def minimum_burnout_speed(
    range_angle: float,
    gravitational_parameter: float = _MU_EARTH,
    radius: float = _R_EARTH,
) -> float:
    """Least burnout speed (m/s) that reaches ``range_angle``.

    Attained at :math:`\\gamma^*`, where the governing sine is exactly one.
    """
    return burnout_speed_for_range(
        range_angle, optimum_burnout_angle(range_angle), gravitational_parameter, radius
    )


def conjugate_flight_path_angle(range_angle: float, flight_path_angle: float) -> float:
    """The other burnout angle reaching the same range at the same speed.

    :math:`2\\gamma^* - \\gamma`, exact rather than iterated — see the module
    docstring. Applying it twice returns the original, and applying it at
    :math:`\\gamma^*` returns :math:`\\gamma^*`, since that solution is its
    own conjugate and is the only one that is.

    Raises
    ------
    ValueError
        If the conjugate falls outside :math:`(0, \\pi/2)`, which means the
        requested angle is so far from optimum that its partner is not a
        flyable ascending trajectory.
    """
    theta, gamma = _check(range_angle, flight_path_angle)
    conjugate = 2.0 * optimum_burnout_angle(theta) - gamma
    if not 0.0 < conjugate < 0.5 * np.pi:
        msg = (
            f"the conjugate of {np.rad2deg(gamma):.2f} deg at range angle "
            f"{np.rad2deg(theta):.2f} deg is {np.rad2deg(conjugate):.2f} deg, "
            "which is not a flyable ascending burnout"
        )
        raise ValueError(msg)
    return float(conjugate)


def free_flight_time(
    range_angle: float,
    flight_path_angle: float,
    gravitational_parameter: float = _MU_EARTH,
    radius: float = _R_EARTH,
) -> float:
    """Burnout-to-impact time (s) on the Keplerian arc.

    Solved through Kepler's equation rather than through Regan's closed
    form Eq. (5.28), which the available scan does not render reliably. The
    two agree: his tabulated 1271.18 s and 2577.69 s come out here as
    1270.35 s and 2576.85 s.

    Assumes burnout and impact at the same radius, so the arc is symmetric
    about apogee. That is Regan's own assumption for these comparisons
    (:math:`z = 0`).
    """
    _, gamma = _check(range_angle, flight_path_angle)
    speed = burnout_speed_for_range(range_angle, gamma, gravitational_parameter, radius)
    angular_momentum = radius * speed * np.cos(gamma)
    energy = 0.5 * speed * speed - gravitational_parameter / radius
    if energy >= 0.0:
        msg = "the trajectory is not bound; there is no free-flight time to impact"
        raise ValueError(msg)
    semi_major = -gravitational_parameter / (2.0 * energy)
    parameter = angular_momentum**2 / gravitational_parameter
    eccentricity = np.sqrt(max(1.0 - parameter / semi_major, 0.0))
    cos_nu = np.clip((parameter / radius - 1.0) / eccentricity, -1.0, 1.0)
    true_anomaly = np.arccos(cos_nu)
    eccentric = 2.0 * np.arctan2(
        np.sqrt(1.0 - eccentricity) * np.sin(0.5 * true_anomaly),
        np.sqrt(1.0 + eccentricity) * np.cos(0.5 * true_anomaly),
    )
    mean_anomaly = eccentric - eccentricity * np.sin(eccentric)
    mean_motion = np.sqrt(gravitational_parameter / semi_major**3)
    return float((2.0 * np.pi - 2.0 * mean_anomaly) / mean_motion)


@dataclass(frozen=True)
class LoftingTrade:
    """The two conjugate trajectories to one target, priced.

    Attributes
    ----------
    range_angle:
        Free-flight range angle (rad) both solutions cover.
    depressed_angle, lofted_angle:
        Burnout flight-path angles (rad), below and above
        :math:`\\gamma^*`.
    optimum_angle:
        :math:`\\gamma^*` (rad).
    burnout_speed:
        Speed (m/s) — the *same* for both conjugates, which is what makes
        them a trade rather than a ranking.
    minimum_speed:
        Speed (m/s) the minimum-energy trajectory would need.
    depressed_time, lofted_time:
        Free-flight times (s).
    speed_penalty:
        ``burnout_speed / minimum_speed - 1``, the energy price of leaving
        the minimum-energy trajectory.
    """

    range_angle: float
    depressed_angle: float
    lofted_angle: float
    optimum_angle: float
    burnout_speed: float
    minimum_speed: float
    depressed_time: float
    lofted_time: float
    speed_penalty: float

    @property
    def time_ratio(self) -> float:
        """Lofted flight time over depressed. Always greater than one."""
        return self.lofted_time / self.depressed_time


def lofting_trade(range_angle: float, flight_path_angle: float) -> LoftingTrade:
    """Price the depressed/lofted choice at one range.

    Parameters
    ----------
    range_angle:
        Free-flight range angle (rad).
    flight_path_angle:
        Either conjugate — the pair is recovered regardless of which is
        given, since the relation is an involution.

    Returns
    -------
    LoftingTrade
    """
    theta, gamma = _check(range_angle, flight_path_angle)
    optimum = optimum_burnout_angle(theta)
    conjugate = conjugate_flight_path_angle(theta, gamma)
    depressed, lofted = min(gamma, conjugate), max(gamma, conjugate)
    speed = burnout_speed_for_range(theta, depressed)
    minimum = minimum_burnout_speed(theta)
    return LoftingTrade(
        range_angle=theta,
        depressed_angle=depressed,
        lofted_angle=lofted,
        optimum_angle=optimum,
        burnout_speed=speed,
        minimum_speed=minimum,
        depressed_time=free_flight_time(theta, depressed),
        lofted_time=free_flight_time(theta, lofted),
        speed_penalty=float(speed / minimum - 1.0),
    )
