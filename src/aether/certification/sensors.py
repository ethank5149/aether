r"""Sensor observation models for reentry vehicle tracking.

Converts physical sensor measurements (radar returns, IR detections,
transponder reports) into state-space constraints for the occupation-measure
tracking framework.  Each sensor class models what the sensor physically
observes and the observation-to-state mapping that produces the constraints.

The fusion function combines multiple sensor observations using
**set-membership intersection** (not Gaussian weighting), consistent with
the occupation-measure framework's set-based uncertainty description.

References
----------
Schweppe, "Recursive state estimation: Unknown but bounded errors and
system inputs," IEEE Trans. Autom. Control, 1968.

Bertsekas & Rhodes, "Recursive state estimation for a set-membership
description of uncertainty," IEEE Trans. Autom. Control, 1971.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from aether.certification.occupation import SensorMeasurement

__all__ = [
    "RadarObservation",
    "SatelliteIRObservation",
    "TransponderObservation",
    "SensorAvailability",
    "Observation",
    "fuse_observations",
    "is_rf_blackout",
    "available_sensors",
]

# ---------------------------------------------------------------------------
# Sensor availability
# ---------------------------------------------------------------------------


class SensorAvailability(Enum):
    """Sensor availability during reentry phases."""

    AVAILABLE = auto()
    RF_BLACKOUT = auto()
    BELOW_HORIZON = auto()


# ---------------------------------------------------------------------------
# Individual sensor observation models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RadarObservation:
    r"""Single measurement from a tracking radar (C-band or S-band).

    A radar measures spherical coordinates relative to the station:
    slant range *R*, elevation *El*, azimuth *Az*, and (optionally) range
    rate :math:`\dot R` from Doppler.  The observation model converts
    these to state constraints on altitude and speed.

    **Altitude** (always available)::

        h = R sin(El)
        δh = √[(sin El · δR)² + (R cos El · δEl)²]

    **Speed** (from Doppler, when available)::

        V_radial = Ṙ                            (measured)
        V = Ṙ / cos θ_LOS                       (with known look angle)
        V ≥ |Ṙ|                                  (without look angle)

    The look angle θ_LOS between the velocity vector and the line of
    sight is NOT known from a single observation.  It can be estimated
    from a radar track (sequence of positions), in which case the caller
    should set ``look_angle_from_velocity_rad``.  Without it, the Doppler
    only gives a lower bound on total speed; we inflate the uncertainty
    to cover the unknown geometric factor.

    Typical half-width uncertainties (treated as set-membership bounds):

    ============  ====  =======  =========
    Sensor        δR    δEl      δṘ
    ============  ====  =======  =========
    C-band prec.  15 m  0.01°    0.1 m/s
    S-band track  50 m  0.05°    1.0 m/s
    Ship radar    200 m 0.1°     3.0 m/s
    ============  ====  =======  =========
    """

    range_m: float
    elevation_rad: float
    range_unc_m: float = 50.0
    elevation_unc_rad: float = float(np.radians(0.05))
    range_rate_ms: float | None = None
    range_rate_unc_ms: float = 1.0
    look_angle_from_velocity_rad: float | None = None

    def to_measurement(self) -> SensorMeasurement:
        """Map radar observables to state constraints."""
        sin_el = np.sin(self.elevation_rad)
        cos_el = np.cos(self.elevation_rad)
        h = self.range_m * sin_el
        dh = float(np.sqrt(
            (sin_el * self.range_unc_m) ** 2
            + (self.range_m * cos_el * self.elevation_unc_rad) ** 2
        ))

        speed_ms: float | None = None
        speed_unc_ms = 0.0
        if self.range_rate_ms is not None:
            if self.look_angle_from_velocity_rad is not None:
                cos_look = abs(np.cos(self.look_angle_from_velocity_rad))
                if cos_look > 0.1:
                    speed_ms = abs(self.range_rate_ms) / cos_look
                    speed_unc_ms = self.range_rate_unc_ms / cos_look
                else:
                    speed_ms = abs(self.range_rate_ms)
                    speed_unc_ms = (
                        abs(self.range_rate_ms) * 0.5 + self.range_rate_unc_ms
                    )
            else:
                speed_ms = abs(self.range_rate_ms)
                speed_unc_ms = (
                    abs(self.range_rate_ms) * 0.5 + self.range_rate_unc_ms
                )

        return SensorMeasurement(
            speed_ms=speed_ms,
            speed_unc_ms=speed_unc_ms,
            altitude_m=h,
            altitude_unc_m=dh,
        )


@dataclass(frozen=True)
class SatelliteIRObservation:
    r"""Measurement from a space-based infrared sensor (SBIRS / OPIR).

    IR satellites detect the thermal signature of the reentry vehicle
    and report a position estimate derived from the angular measurement
    in the focal-plane array and the known satellite ephemeris.

    **Key property**: works *during RF blackout* — the plasma sheath is
    transparent to infrared wavelengths.  During blackout this is the
    only source of tracking data (unless the vehicle relays via UHF).

    Typical half-width uncertainties:

    ===========  ===========
    Platform     δ altitude
    ===========  ===========
    GEO (SBIRS)  5–10 km
    HEO (SBIRS)  2–5 km
    ===========  ===========
    """

    altitude_m: float
    altitude_unc_m: float = 5000.0

    def to_measurement(self) -> SensorMeasurement:
        return SensorMeasurement(
            altitude_m=self.altitude_m,
            altitude_unc_m=self.altitude_unc_m,
        )


@dataclass(frozen=True)
class TransponderObservation:
    r"""Measurement from a cooperative transponder (GPS relay, C-band beacon).

    The vehicle transmits its own state estimate (from onboard GPS + IMU)
    via an RF telemetry link.  Only available when the vehicle cooperates
    **and** the RF link is active (blocked during RF blackout unless
    relayed via UHF/TDRS).

    Typical half-width uncertainties:

    ===========  =======  ========  =====
    Mode         δV       δh        δγ
    ===========  =======  ========  =====
    GPS-aided    0.1 m/s  10 m      0.1°
    INS-only     5 m/s    200 m     1°
    ===========  =======  ========  =====
    """

    speed_ms: float
    altitude_m: float
    speed_unc_ms: float = 1.0
    altitude_unc_m: float = 50.0
    fpa_rad: float | None = None
    fpa_unc_rad: float = float(np.radians(0.5))

    def to_measurement(self) -> SensorMeasurement:
        return SensorMeasurement(
            speed_ms=self.speed_ms,
            speed_unc_ms=self.speed_unc_ms,
            altitude_m=self.altitude_m,
            altitude_unc_m=self.altitude_unc_m,
            fpa_rad=self.fpa_rad,
            fpa_unc_rad=self.fpa_unc_rad,
        )


Observation = RadarObservation | SatelliteIRObservation | TransponderObservation


# ---------------------------------------------------------------------------
# Multi-sensor fusion (set-membership intersection)
# ---------------------------------------------------------------------------


def fuse_observations(observations: list[Observation]) -> SensorMeasurement:
    """Fuse multiple simultaneous sensor observations via set-membership
    intersection.

    For each state variable observed by multiple sensors, the fused
    constraint is the **intersection** of the individual intervals::

        [a_fused, b_fused] = ⋂_i [a_i, b_i]

    This is tighter than any single sensor and is consistent with the
    occupation-measure framework's set-based uncertainty.

    If the intersection is empty (sensors disagree), the fusion falls
    back to the **union** of all intervals for that variable — the most
    conservative valid bound.  Empty intersection indicates a sensor
    fault or model error; it should be investigated.
    """
    measurements = [obs.to_measurement() for obs in observations]

    def _fuse_intervals(
        intervals: list[tuple[float, float]],
    ) -> tuple[float | None, float]:
        if not intervals:
            return None, 0.0
        lo = max(a for a, _ in intervals)
        hi = min(b for _, b in intervals)
        if lo <= hi:
            return (lo + hi) / 2.0, (hi - lo) / 2.0
        lo = min(a for a, _ in intervals)
        hi = max(b for _, b in intervals)
        return (lo + hi) / 2.0, (hi - lo) / 2.0

    speed_ivs = [
        (m.speed_ms - m.speed_unc_ms, m.speed_ms + m.speed_unc_ms)
        for m in measurements
        if m.speed_ms is not None
    ]
    alt_ivs = [
        (m.altitude_m - m.altitude_unc_m, m.altitude_m + m.altitude_unc_m)
        for m in measurements
        if m.altitude_m is not None
    ]
    fpa_ivs = [
        (m.fpa_rad - m.fpa_unc_rad, m.fpa_rad + m.fpa_unc_rad)
        for m in measurements
        if m.fpa_rad is not None
    ]

    speed, speed_unc = _fuse_intervals(speed_ivs)
    alt, alt_unc = _fuse_intervals(alt_ivs)
    fpa, fpa_unc = _fuse_intervals(fpa_ivs)

    return SensorMeasurement(
        speed_ms=speed,
        speed_unc_ms=speed_unc,
        altitude_m=alt,
        altitude_unc_m=alt_unc,
        fpa_rad=fpa,
        fpa_unc_rad=fpa_unc,
    )


# ---------------------------------------------------------------------------
# RF blackout model
# ---------------------------------------------------------------------------


def is_rf_blackout(speed_ms: float, altitude_m: float) -> bool:
    """Estimate whether RF blackout is active at the given flight condition.

    The plasma sheath blocks RF frequencies (L-band through X-band) when
    the electron number density exceeds the critical plasma frequency.
    This depends on the stagnation enthalpy (proportional to V²) and the
    ambient density (function of altitude).

    Approximate boundaries from Apollo and Shuttle flight data::

        Onset:   V > 6 km/s  and  30 km < h < 85 km
        Partial: V > 4.5 km/s and  25 km < h < 75 km
        End:     V < 3 km/s  or   h < 20 km

    These are conservative — real boundaries depend on vehicle shape,
    nose radius, antenna placement and operating frequency.
    """
    if speed_ms < 3000.0 or altitude_m < 20000.0 or altitude_m > 90000.0:
        return False
    if speed_ms > 6000.0 and 30000.0 < altitude_m < 85000.0:
        return True
    if speed_ms > 4500.0 and 25000.0 < altitude_m < 75000.0:
        return True
    return False


def available_sensors(
    speed_ms: float,
    altitude_m: float,
    has_transponder: bool = False,
) -> dict[str, SensorAvailability]:
    """Determine which sensor types are available at a given flight condition.

    Returns a dict mapping sensor names to their availability status.
    """
    blackout = is_rf_blackout(speed_ms, altitude_m)

    result: dict[str, SensorAvailability] = {}

    result["radar"] = (
        SensorAvailability.RF_BLACKOUT
        if blackout
        else SensorAvailability.AVAILABLE
    )

    # IR satellites are unaffected by RF blackout
    result["ir_satellite"] = SensorAvailability.AVAILABLE

    if has_transponder:
        result["transponder"] = (
            SensorAvailability.RF_BLACKOUT
            if blackout
            else SensorAvailability.AVAILABLE
        )

    return result
