"""Inertial injection error: deriving boost dispersion from IMU grade.

Boost injection error is the last stated specification in the accuracy
chain, and it is the one input every terminal-accuracy number rests on. It
need not be assumed: for a boost lasting a few hundred seconds it follows
from the inertial measurement unit's own error sources by three
propagation laws that are textbook and that scale differently in time,
which is what makes the dominant term identifiable.

The three laws, and why the exponents matter
--------------------------------------------

Over a burn short compared with the Schuler period of about 84 minutes,
the error dynamics have not yet begun to oscillate and the growth is
polynomial:

.. math::

    \\delta r_{\\text{accel}} = \\tfrac{1}{2} b_a t^2, \\qquad
    \\delta r_{\\text{align}} = \\tfrac{1}{2} g\\, \\delta\\theta\\, t^2,
    \\qquad
    \\delta r_{\\text{gyro}}  = \\tfrac{1}{6} g\\, \\varepsilon\\, t^3.

An accelerometer bias integrates twice into position. An initial
*misalignment* tilts the platform so a component of gravity is read as
acceleration, which integrates the same way. A gyro drift rate tilts the
platform progressively, so the false gravity component grows linearly and
integrates *three* times.

The differing exponents are the useful part. A gyro term that is
negligible on a short burn dominates a long one, and the crossover is
computable rather than a matter of judgement:
:func:`dominant_error_source` reports which term leads and
:func:`gyro_dominates_after` gives the burn duration at which the gyro
overtakes the accelerometer.

What this derives and what it does not
--------------------------------------

The *propagation* is derived; the *component specifications* are not, and
this module takes them as inputs because an IMU grade is a procurement
decision rather than a physical constant. Representative grades are given
in :data:`IMU_GRADES` with their provenance stated as "conventional
industry bands, not a measured source" — because that is what they are.
Verifying those bands needs the strapdown-navigation literature, and the
canonical references are named in the project roadmap rather than
paraphrased here.

Velocity error matters more than position
-----------------------------------------

For an injection the position error at burnout is usually the smaller
problem. A velocity error persists and is amplified by the transfer that
follows: :mod:`aether_gambit.orbital.fobs` measures a sensitivity of 850 to 3484
seconds depending on perigee depth, so 1 m/s of injection velocity error
becomes kilometres at the entry interface. :func:`injection_error` returns
both, and the velocity term is the one to watch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "IMU_GRADES",
    "AlignmentError",
    "ImuGrade",
    "InjectionError",
    "dominant_error_source",
    "gyro_dominates_after",
    "gyrocompass_alignment",
    "injection_error",
]

#: Standard gravity (m/s²), the lever that turns a tilt into a false
#: acceleration and therefore the reason attitude errors matter at all.
_G0 = 9.80665
#: Degrees per hour to radians per second.
_DEG_PER_HOUR = np.pi / 180.0 / 3600.0
#: Micro-g to m/s².
_MICRO_G = 1e-6 * _G0


@dataclass(frozen=True)
class ImuGrade:
    """Error specification of an inertial measurement unit.

    Attributes
    ----------
    accelerometer_bias:
        One-sigma bias (m/s²).
    gyro_drift:
        One-sigma drift rate (rad/s).
    alignment:
        One-sigma initial platform misalignment (rad).
    label:
        Grade name, carried through so a result can be read back against
        the assumption that produced it.
    """

    accelerometer_bias: float
    gyro_drift: float
    alignment: float
    label: str = ""

    def __post_init__(self) -> None:
        for name in ("accelerometer_bias", "gyro_drift", "alignment"):
            value = float(getattr(self, name))
            if not (np.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be finite and >= 0, got {value}")

    @classmethod
    def from_engineering_units(
        cls,
        accelerometer_bias_micro_g: float,
        gyro_drift_deg_per_hour: float,
        alignment_arcsec: float | None = None,
        label: str = "",
    ) -> ImuGrade:
        """Construct from the units instruments are actually specified in.

        ``alignment_arcsec`` defaults to the levelling accuracy a
        stationary gyrocompass alignment achieves with this accelerometer,
        :math:`B/g` (Titterton & Weston §10.3.2). Leaving it unset is the
        normal case and is *exact* rather than a rounded transcription,
        which matters because it makes the alignment and accelerometer
        contributions to injection error identically equal — see
        :func:`dominant_error_source`. Pass a value only to model an
        alignment limited by something other than the instrument, such as
        a transfer alignment from a moving host.
        """
        bias = accelerometer_bias_micro_g * _MICRO_G
        return cls(
            accelerometer_bias=bias,
            gyro_drift=gyro_drift_deg_per_hour * _DEG_PER_HOUR,
            alignment=(
                bias / _G0
                if alignment_arcsec is None
                else alignment_arcsec * np.pi / (180.0 * 3600.0)
            ),
            label=label,
        )


#: Instrument grades from Groves, *Principles of GNSS, Inertial, and
#: Multisensor Integrated Navigation Systems*, 2nd ed., **Table 4.1**
#: ("Typical Accelerometer and Gyro Biases for Different Grades of IMU").
#:
#: The taxonomy is his, and adopting it corrected ours. Earlier revisions
#: used "strategic / navigation / tactical", of which only the last is a
#: term Groves uses, and the guessed numbers sat in roughly the right
#: places under the wrong names: our "navigation" (50 ug, 0.01 deg/hr)
#: lands inside his *aviation* band, while our "strategic" (5 ug) was twice
#: as optimistic as his best listed grade, *marine*.
#:
#: Where the table gives a range the better end is taken, with the range
#: recorded alongside so the pessimistic end is not lost.
#:
#: **Alignment is derived, not stated, and is no longer an independent
#: input.** Groves Table 4.1 covers instrument bias only, and an earlier
#: revision guessed alignment separately. It does not need guessing: a
#: stationary gyrocompass alignment levels against gravity and finds north
#: by nulling east Earth rate, so its accuracy follows from these same two
#: biases -- see :func:`gyrocompass_alignment` and Titterton & Weston
#: §10.3.2. The entries below therefore omit ``alignment_arcsec`` entirely
#: and take the derived tilt, and deriving it corrected the guesses in both
#: directions at once: tilt was roughly three times too pessimistic, while
#: azimuth was optimistic by a factor of ten at marine grade and three
#: hundred at tactical.
#:
#: That azimuth result is worth stating plainly. Gyrocompassing must
#: resolve Earth rate, which is 15 deg/hr; a tactical-grade gyro drifting
#: at 1 deg/hr is 7% of the signal it is trying to measure, and the
#: resulting azimuth error is 5.4 degrees. **A tactical-grade unit cannot
#: usefully gyrocompass at all** and needs an external azimuth reference --
#: transfer alignment from a host, or a celestial or satellite fix. Only
#: the tilt channel is carried into :func:`injection_error`, because tilt
#: is what tips gravity into the horizontal accelerometers; azimuth error
#: rotates the trajectory instead and appears downstream as crossrange.
IMU_GRADES: dict[str, ImuGrade] = {
    # Table 4.1: 0.01 mg, 0.001 deg/hr.
    "marine": ImuGrade.from_engineering_units(
        accelerometer_bias_micro_g=10.0,
        gyro_drift_deg_per_hour=0.001,
        label="marine",
    ),
    # Table 4.1: 0.03-0.1 mg, 0.01 deg/hr.
    "aviation": ImuGrade.from_engineering_units(
        accelerometer_bias_micro_g=30.0,
        gyro_drift_deg_per_hour=0.01,
        label="aviation",
    ),
    # Table 4.1: 0.1-1 mg, 0.1 deg/hr.
    "intermediate": ImuGrade.from_engineering_units(
        accelerometer_bias_micro_g=100.0,
        gyro_drift_deg_per_hour=0.1,
        label="intermediate",
    ),
    # Table 4.1: 1-10 mg, 1-100 deg/hr.
    "tactical": ImuGrade.from_engineering_units(
        accelerometer_bias_micro_g=1000.0,
        gyro_drift_deg_per_hour=1.0,
        label="tactical",
    ),
    # Table 4.1: >3 mg, >100 deg/hr.
    "consumer": ImuGrade.from_engineering_units(
        accelerometer_bias_micro_g=3000.0,
        gyro_drift_deg_per_hour=100.0,
        label="consumer",
    ),
}


@dataclass(frozen=True)
class InjectionError:
    """Position and velocity error at burnout, by contributing source."""

    position: float
    """Total one-sigma position error (m), root-sum-square."""
    velocity: float
    """Total one-sigma velocity error (m/s), root-sum-square."""
    from_accelerometer: float
    from_alignment: float
    from_gyro: float
    """Per-source position contributions (m)."""
    burn_time: float
    grade: str = ""

    @property
    def contributions(self) -> dict[str, float]:
        return {
            "accelerometer": self.from_accelerometer,
            "alignment": self.from_alignment,
            "gyro": self.from_gyro,
        }


#: Earth's sidereal rotation rate (rad/s), the reference gyrocompassing
#: nulls against. Alignment in azimuth is only possible because this is
#: non-zero, which is why the process degrades toward the poles.
_EARTH_RATE = 7.292115e-5


@dataclass(frozen=True)
class AlignmentError:
    """Initial platform misalignment from a gyrocompass alignment.

    Attributes
    ----------
    tilt:
        Level error (rad) about a horizontal axis. This is the one that
        matters for injection: it tips the platform so a component of
        gravity is read as horizontal acceleration.
    azimuth:
        Heading error (rad) about the vertical. It does not corrupt the
        measured acceleration magnitude, but it rotates the whole
        trajectory in the horizontal plane, so it appears downstream as
        *crossrange* rather than as downrange error.
    latitude:
        Latitude (rad) the alignment was performed at.
    """

    tilt: float
    azimuth: float
    latitude: float


def gyrocompass_alignment(grade: ImuGrade, latitude: float) -> AlignmentError:
    """Alignment error achievable by gyrocompassing, from instrument bias.

    Alignment is not an independent specification. A stationary
    gyrocompass alignment levels the platform against gravity and finds
    north by nulling the east component of Earth rate, so the accuracy it
    reaches is set by the *same* accelerometer and gyro biases that
    Groves' Table 4.1 already gives. Titterton & Weston, *Strapdown
    Inertial Navigation Technology* 2nd ed., §10.3.2:

    .. math::

        \\delta\\alpha = \\frac{B}{g}, \\qquad
        \\delta\\gamma = \\frac{D}{\\Omega \\cos L}
                            + \\frac{B \\tan L}{g}.

    The azimuth expression has two terms and the second is easy to miss:
    a *level* error about north tips vertical Earth rate into the east
    axis, where gyrocompassing cannot distinguish it from a gyro bias. So
    a poor accelerometer degrades heading as well as level.

    Verified against the two numerical statements the source makes: a
    1 milli-g accelerometer bias gives a 1 mrad level error, and a
    0.01 deg/hr gyro drift gives a 1 mrad azimuth error at 45 degrees
    latitude.

    Raises
    ------
    ValueError
        Near the poles, where :math:`\\cos L \\to 0` and gyrocompassing
        has no horizontal Earth-rate component to null. This is a real
        limit of the method, not a numerical one -- Titterton's Fig. 10.4
        shows the error diverging -- and it is refused rather than
        returned as a large number.
    """
    lat = float(latitude)
    if not np.isfinite(lat) or abs(lat) >= 0.5 * np.pi:
        raise ValueError(f"latitude must be finite and within (-pi/2, pi/2), got {lat}")
    if abs(np.cos(lat)) < np.cos(np.deg2rad(85.0)):
        raise ValueError(
            f"gyrocompassing degrades without bound toward the poles: at "
            f"{np.rad2deg(lat):.2f} deg there is too little horizontal Earth "
            f"rate to null against. Use an external azimuth reference instead"
        )
    tilt = grade.accelerometer_bias / _G0
    azimuth = grade.gyro_drift / (_EARTH_RATE * np.cos(lat)) + (
        grade.accelerometer_bias * np.tan(lat) / _G0
    )
    return AlignmentError(tilt=float(tilt), azimuth=float(azimuth), latitude=lat)


def injection_error(grade: ImuGrade, burn_time: float) -> InjectionError:
    """Injection error at burnout from an IMU grade and a burn duration.

    Valid while the burn is short against the Schuler period of about
    84 minutes, which every boost phase is. Beyond that the error dynamics
    oscillate rather than grow polynomially and these expressions stop
    describing them; the function refuses rather than extrapolating.

    The cutoff is deliberately the same for all three terms even though
    Gelb (*Applied Optimal Estimation*, Fig. 8.2-1) attributes them to two
    *different* feedback loops: accelerometer bias and gravity uncertainty
    excite the 84-minute Schuler loop, while gyro drift excites the
    24-hour heading (Earth-rate) loop. Using one quarter-Schuler cutoff for
    all three is not an approximation that happens to be convenient — a
    boost phase of a few hundred seconds sits four orders of magnitude
    inside *both* periods, so both terms are deep in their polynomial
    regime regardless of which loop eventually bounds them, and a single
    conservative gate is exact for the purpose it is used for here.
    """
    t = float(burn_time)
    if not (np.isfinite(t) and t > 0.0):
        raise ValueError(f"burn_time must be finite and > 0, got {t}")
    schuler_period = 2.0 * np.pi * np.sqrt(6371e3 / _G0)
    if t > 0.25 * schuler_period:
        raise ValueError(
            f"burn_time {t:.6g} s exceeds a quarter of the Schuler period "
            f"({0.25 * schuler_period:.0f} s); beyond that the inertial error "
            f"dynamics oscillate rather than grow polynomially and these "
            f"expressions no longer describe them"
        )

    from_accelerometer = 0.5 * grade.accelerometer_bias * t**2
    from_alignment = 0.5 * _G0 * grade.alignment * t**2
    from_gyro = _G0 * grade.gyro_drift * t**3 / 6.0

    velocity = float(
        np.sqrt(
            (grade.accelerometer_bias * t) ** 2
            + (_G0 * grade.alignment * t) ** 2
            + (0.5 * _G0 * grade.gyro_drift * t**2) ** 2
        )
    )
    position = float(np.sqrt(from_accelerometer**2 + from_alignment**2 + from_gyro**2))
    return InjectionError(
        position=position,
        velocity=velocity,
        from_accelerometer=float(from_accelerometer),
        from_alignment=float(from_alignment),
        from_gyro=float(from_gyro),
        burn_time=t,
        grade=grade.label,
    )


def dominant_error_source(grade: ImuGrade, burn_time: float) -> str:
    """Which of the three terms contributes most position error.

    Ties are reported rather than broken. That matters here because one
    tie is *structural* rather than coincidental: a gyrocompass-aligned
    platform has tilt :math:`B/g`, so its alignment contribution is
    :math:`\\tfrac12 g (B/g) t^2 = \\tfrac12 B t^2` — exactly the
    accelerometer term. Accelerometer bias therefore enters twice, once
    directly and once through the tilt it caused during alignment, and the
    two are equal by construction rather than by accident. Silently
    picking one would hide the fact that fixing the accelerometer buys
    twice what it appears to.
    """
    contributions = injection_error(grade, burn_time).contributions
    largest = max(contributions.values())
    if largest == 0.0:
        return "none"
    tied = sorted(name for name, value in contributions.items() if value >= largest * (1.0 - 1e-9))
    return " = ".join(tied)


def gyro_dominates_after(grade: ImuGrade) -> float:
    """Burn duration (s) at which gyro drift overtakes accelerometer bias.

    From :math:`g \\varepsilon t^3/6 = b_a t^2/2`, so
    :math:`t = 3 b_a / (g \\varepsilon)`. This is the number that decides
    which instrument to spend money on for a given burn, and it depends on
    the *ratio* of the two specifications rather than on either alone.
    """
    if grade.gyro_drift == 0.0:
        return float("inf")
    return float(3.0 * grade.accelerometer_bias / (_G0 * grade.gyro_drift))
