"""Heterogeneous sensor measurement models and their measurement covariance.

A single position with an isotropic error is not what any real early-warning
sensor produces. A phased-array radar measures range well and cross-range
poorly, so its error ellipsoid is a *disc* perpendicular to the line of sight.
An infrared satellite measures two angles and has *no* range at all, so its error
ellipsoid is a *needle* along the line of sight (the doc's "zero depth
perception"). The whole geometric advantage of fusing them — the fused ellipsoid
collapsing to a volume far smaller than either sensor alone — comes precisely
from those two ellipsoids intersecting near-orthogonally, and it is invisible to
an isotropic-noise model.

This module gives each sensor a :class:`MeasurementModel` — a range one-sigma and
an angular one-sigma, with range switched off for an angles-only IR sensor — and
maps it to the anisotropic 3x3 position covariance in inertial space along the
actual line of sight:

.. math::

    \\mathbf{R} = \\sigma_r^2\\, \\hat{\\mathbf u}\\hat{\\mathbf u}^\\top
      + (r\\,\\sigma_\\theta)^2\\,(\\mathbf I - \\hat{\\mathbf u}\\hat{\\mathbf u}^\\top),

with :math:`\\hat{\\mathbf u}` the unit line of sight and :math:`r` the range.
Radar has small :math:`\\sigma_r` and large :math:`r\\sigma_\\theta`; an angles-only
sensor has an enormous effective :math:`\\sigma_r` and a small :math:`r\\sigma_\\theta`.
A tracker that carries this per-measurement covariance fuses the two correctly;
one that assumes an isotropic error cannot.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

_FloatArray = NDArray[np.float64]

__all__ = [
    "Measurement",
    "MeasurementKind",
    "MeasurementModel",
    "los_covariance_eci",
    "range_azimuth_elevation",
]

#: A very large but finite range one-sigma (m) standing in for "no range at all",
#: so an angles-only measurement stays numerically a position with an enormous
#: along-line-of-sight variance rather than a special case.
_ANGLES_ONLY_RANGE_SIGMA = 1.0e9


class MeasurementKind(Enum):
    """What a sensor actually measures."""

    RANGE_AZ_EL = "range-az-el"   # radar: range + two angles -> full 3-D position
    AZ_EL = "az-el"               # infrared: two angles only, no range


@dataclass(frozen=True)
class MeasurementModel:
    """The measurement error of a sensor, in its own range/angle coordinates.

    Attributes
    ----------
    kind:
        Range-and-angles (radar) or angles-only (IR).
    range_sigma_m:
        One-sigma range error (m). Ignored for an angles-only sensor.
    angular_sigma_rad:
        One-sigma per-axis angular error (rad); the cross-range error at range
        ``r`` is ``r * angular_sigma_rad``.
    """

    kind: MeasurementKind
    range_sigma_m: float
    angular_sigma_rad: float

    def __post_init__(self) -> None:
        if not (np.isfinite(self.angular_sigma_rad) and self.angular_sigma_rad > 0.0):
            raise ValueError(
                f"angular_sigma_rad must be finite and > 0, got {self.angular_sigma_rad}"
            )
        if self.kind is MeasurementKind.RANGE_AZ_EL and not (
            np.isfinite(self.range_sigma_m) and self.range_sigma_m > 0.0
        ):
            raise ValueError(f"range_sigma_m must be finite and > 0, got {self.range_sigma_m}")

    @property
    def effective_range_sigma(self) -> float:
        """Range one-sigma actually used — huge for an angles-only sensor."""
        if self.kind is MeasurementKind.AZ_EL:
            return _ANGLES_ONLY_RANGE_SIGMA
        return float(self.range_sigma_m)


@dataclass(frozen=True)
class Measurement:
    """One sensor return with its anisotropic inertial covariance.

    Attributes
    ----------
    time:
        Epoch of the measurement (s).
    position:
        Best-estimate ECI position (m). For an angles-only sensor this is the
        true position plus cross-range noise, with the along-line-of-sight
        component essentially unconstrained (encoded in ``covariance``).
    covariance:
        3x3 ECI position measurement covariance :math:`\\mathbf R`, anisotropic
        along the line of sight.
    sensor_id:
        Reporting sensor.
    line_of_sight:
        Unit line-of-sight vector from sensor to target (ECI).
    range_m:
        Sensor-to-target range (m).
    """

    time: float
    position: _FloatArray
    covariance: _FloatArray
    sensor_id: str
    line_of_sight: _FloatArray
    range_m: float


def range_azimuth_elevation(
    sensor_eci: _FloatArray, target_eci: _FloatArray
) -> tuple[float, float, float]:
    """Range (m), azimuth (rad) and elevation (rad) of a target from a sensor.

    Azimuth and elevation are taken in the sensor's local topocentric frame: up
    is the sensor's radial direction, and east/north complete a right-handed
    triad. Adequate for the measurement geometry; the covariance mapping uses the
    line of sight directly and does not depend on the azimuth origin.
    """
    sensor = np.asarray(sensor_eci, dtype=np.float64)
    los = np.asarray(target_eci, dtype=np.float64) - sensor
    rng = float(np.linalg.norm(los))
    if rng == 0.0:
        return 0.0, 0.0, np.pi / 2
    up = sensor / (np.linalg.norm(sensor) or 1.0)
    # East = z_hat x up (undefined at the poles; fall back to x_hat there).
    east = np.cross(np.array([0.0, 0.0, 1.0]), up)
    if np.linalg.norm(east) < 1e-9:
        east = np.array([1.0, 0.0, 0.0])
    east = east / np.linalg.norm(east)
    north = np.cross(up, east)
    u = los / rng
    elevation = float(np.arcsin(np.clip(np.dot(u, up), -1.0, 1.0)))
    azimuth = float(np.arctan2(np.dot(u, east), np.dot(u, north)))
    return rng, azimuth, elevation


def los_covariance_eci(
    sensor_eci: _FloatArray, target_eci: _FloatArray, model: MeasurementModel
) -> _FloatArray:
    """Anisotropic 3x3 ECI position covariance along the line of sight.

    ``sigma_r^2`` along the line of sight, ``(r*sigma_theta)^2`` across it — the
    disc (radar) or needle (angles-only IR) that makes fusion geometry work.
    """
    sensor = np.asarray(sensor_eci, dtype=np.float64)
    los = np.asarray(target_eci, dtype=np.float64) - sensor
    rng = float(np.linalg.norm(los))
    if rng == 0.0:
        return np.eye(3) * model.angular_sigma_rad**2
    u = los / rng
    along = model.effective_range_sigma**2
    cross = (rng * model.angular_sigma_rad) ** 2
    uut = np.outer(u, u)
    return along * uut + cross * (np.eye(3) - uut)
