"""WGS-84 geodesy: the universal format for launch sites and aimpoints.

Every trajectory module in this package works in an inertial Cartesian
frame, which is right for the dynamics and useless for specifying a
problem. A launch site and a set of aimpoints are naturally given as
latitude, longitude and altitude on a reference ellipsoid, and this module
is the conversion between the two so that nothing downstream has to carry
its own.

Three distinctions that are easy to elide and expensive to get wrong
--------------------------------------------------------------------

**Geodetic latitude is not geocentric latitude.** Geodetic latitude is the
angle of the local *normal to the ellipsoid*, which is what a survey or a
GPS receiver reports; geocentric latitude is the angle of the radius from
the centre. They differ by up to 0.19 degrees near 45 degrees, which is
**21 km on the ground** — comfortably larger than any terminal accuracy
worth discussing. :func:`geodetic_to_geocentric_latitude` converts, and the
rest of this module is explicit about which one it takes.

**ECEF is not ECI.** The Earth-fixed frame rotates. A position given in
latitude and longitude is fixed in ECEF and moves in ECI, so converting a
ground target to an inertial aimpoint requires the *epoch at which the
vehicle arrives* — not the epoch of the calculation. Getting that wrong
produces a miss of 465 m/s times the time error at the equator, which is
roughly 28 km per minute. :func:`geodetic_to_eci` takes the arrival time
for exactly that reason.

**The ellipsoid is not a sphere.** The polar radius is 21.4 km smaller than
the equatorial. A caller that assumes a spherical Earth is generally doing
so for the structure of its own problem rather than for the geometry, and a
system-level answer should convert through this module at the boundary.

Great-circle versus geodesic
----------------------------

:func:`great_circle_range` uses the haversine formula on a sphere of the
mean radius, which is the right tool for a range budget and the wrong one
for a survey: it errs by up to about 0.5% against the true geodesic on the
ellipsoid. That is 30 km over a 6000 km arc — negligible against a glide's
own range uncertainty, and not negligible if it is quietly used as a
terminal aimpoint. It is provided for budgeting and labelled as such.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "WGS84_ECCENTRICITY_SQUARED",
    "WGS84_FLATTENING",
    "WGS84_MEAN_RADIUS",
    "WGS84_POLAR_RADIUS",
    "WGS84_SEMI_MAJOR_AXIS",
    "GeodeticPosition",
    "ecef_to_eci",
    "ecef_to_geodetic",
    "eci_to_ecef",
    "geodetic_to_ecef",
    "geodetic_to_eci",
    "geodetic_to_geocentric_latitude",
    "great_circle_bearing",
    "great_circle_range",
]

_FloatArray = NDArray[np.float64]

#: WGS-84 defining and derived constants.
WGS84_SEMI_MAJOR_AXIS = 6378137.0
WGS84_FLATTENING = 1.0 / 298.257223563
WGS84_POLAR_RADIUS = WGS84_SEMI_MAJOR_AXIS * (1.0 - WGS84_FLATTENING)
WGS84_ECCENTRICITY_SQUARED = WGS84_FLATTENING * (2.0 - WGS84_FLATTENING)
#: Mean radius (IUGG), used only by the spherical great-circle helpers.
WGS84_MEAN_RADIUS = (2.0 * WGS84_SEMI_MAJOR_AXIS + WGS84_POLAR_RADIUS) / 3.0

#: Earth's sidereal rotation rate (rad/s), IERS conventions.
_ROTATION_RATE = 7.292115e-5


@dataclass(frozen=True)
class GeodeticPosition:
    """A point on or above the WGS-84 ellipsoid.

    This is the universal input format: a launch site, an aimpoint, a
    tracking station. Latitude is **geodetic**.

    Attributes
    ----------
    latitude, longitude:
        Radians. Latitude in ``[-pi/2, pi/2]``, longitude wrapped to
        ``(-pi, pi]``.
    altitude:
        Height above the ellipsoid (m). May be negative.
    label:
        Optional identifier, carried through so results can be read back
        against whatever named the point.
    """

    latitude: float
    longitude: float
    altitude: float = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        for name in ("latitude", "longitude", "altitude"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if abs(self.latitude) > 0.5 * np.pi + 1e-12:
            raise ValueError(
                f"latitude must lie in [-pi/2, pi/2], got {self.latitude}; "
                f"values outside it are not a wrapped longitude but a "
                f"mis-ordered coordinate pair"
            )
        object.__setattr__(
            self, "longitude", float((self.longitude + np.pi) % (2.0 * np.pi) - np.pi)
        )

    @classmethod
    def from_degrees(
        cls, latitude: float, longitude: float, altitude: float = 0.0, label: str = ""
    ) -> GeodeticPosition:
        """Construct from degrees, which is how sites are usually quoted."""
        return cls(
            latitude=float(np.deg2rad(latitude)),
            longitude=float(np.deg2rad(longitude)),
            altitude=float(altitude),
            label=label,
        )

    @property
    def degrees(self) -> tuple[float, float]:
        return float(np.rad2deg(self.latitude)), float(np.rad2deg(self.longitude))


def geodetic_to_geocentric_latitude(latitude: ArrayLike) -> _FloatArray:
    """Geocentric latitude (rad) for a geodetic latitude on the ellipsoid.

    :math:`\\tan\\phi_c = (1 - e^2)\\tan\\phi_d`. The difference peaks near
    45 degrees at about 0.19 degrees, which is 21 km on the ground.
    """
    phi = np.asarray(latitude, dtype=np.float64)
    return np.asarray(np.arctan((1.0 - WGS84_ECCENTRICITY_SQUARED) * np.tan(phi)))


def geodetic_to_ecef(position: GeodeticPosition) -> _FloatArray:
    """Earth-fixed Cartesian position (m) of a geodetic point.

    Exact and non-iterative in this direction; only the inverse needs care.
    """
    sin_lat = np.sin(position.latitude)
    cos_lat = np.cos(position.latitude)
    # Radius of curvature in the prime vertical.
    normal = WGS84_SEMI_MAJOR_AXIS / np.sqrt(1.0 - WGS84_ECCENTRICITY_SQUARED * sin_lat**2)
    return np.array(
        [
            (normal + position.altitude) * cos_lat * np.cos(position.longitude),
            (normal + position.altitude) * cos_lat * np.sin(position.longitude),
            (normal * (1.0 - WGS84_ECCENTRICITY_SQUARED) + position.altitude) * sin_lat,
        ]
    )


def ecef_to_geodetic(position: ArrayLike, label: str = "") -> GeodeticPosition:
    """Geodetic coordinates of an Earth-fixed Cartesian position.

    Seeded with Bowring's closed form and refined twice. Bowring alone is
    sub-millimetre near the surface but degrades with altitude — measured
    at 0.32 m at geostationary radius — so the refinement is what makes the
    accuracy claim hold across the whole range rather than only near the
    ground. Measured round-trip error after refinement is under 10 nm
    within the atmosphere and under a micrometre out to geostationary
    radius.
    """
    r = np.asarray(position, dtype=np.float64)
    if r.shape != (3,):
        raise ValueError(f"position must be a 3-vector, got shape {r.shape}")
    if not np.isfinite(r).all():
        raise ValueError("position must be finite")
    x, y, z = float(r[0]), float(r[1]), float(r[2])
    equatorial = np.hypot(x, y)
    if equatorial == 0.0:
        # On the polar axis: longitude is undefined, and reporting zero is
        # the only defensible convention rather than an arbitrary angle.
        pole = float(np.sign(z)) or 1.0
        return GeodeticPosition(
            latitude=pole * 0.5 * np.pi,
            longitude=0.0,
            altitude=abs(z) - WGS84_POLAR_RADIUS,
            label=label,
        )
    second_eccentricity = WGS84_ECCENTRICITY_SQUARED / (1.0 - WGS84_ECCENTRICITY_SQUARED)
    beta = np.arctan2(WGS84_SEMI_MAJOR_AXIS * z, WGS84_POLAR_RADIUS * equatorial)
    latitude = np.arctan2(
        z + second_eccentricity * WGS84_POLAR_RADIUS * np.sin(beta) ** 3,
        equatorial - WGS84_ECCENTRICITY_SQUARED * WGS84_SEMI_MAJOR_AXIS * np.cos(beta) ** 3,
    )
    # Bowring's seed is sub-millimetre near the surface but degrades with
    # altitude -- measured at 0.32 m out at geostationary radius. Two
    # fixed-point refinements on the standard relation restore machine
    # precision across the whole range at negligible cost, which is worth
    # more than the closed form's tidiness.
    for _ in range(2):
        sin_lat = np.sin(latitude)
        normal = WGS84_SEMI_MAJOR_AXIS / np.sqrt(1.0 - WGS84_ECCENTRICITY_SQUARED * sin_lat**2)
        altitude = equatorial / np.cos(latitude) - normal
        latitude = np.arctan2(
            z,
            equatorial * (1.0 - WGS84_ECCENTRICITY_SQUARED * normal / (normal + altitude)),
        )
    sin_lat = np.sin(latitude)
    normal = WGS84_SEMI_MAJOR_AXIS / np.sqrt(1.0 - WGS84_ECCENTRICITY_SQUARED * sin_lat**2)
    altitude = equatorial / np.cos(latitude) - normal
    return GeodeticPosition(
        latitude=float(latitude),
        longitude=float(np.arctan2(y, x)),
        altitude=float(altitude),
        label=label,
    )


def ecef_to_eci(
    position: ArrayLike,
    time: float,
    gmst_epoch: float = 0.0,
    rotation_rate: float = _ROTATION_RATE,
) -> _FloatArray:
    """Rotate an Earth-fixed vector into the inertial frame at ``time``."""
    r = np.asarray(position, dtype=np.float64)
    if r.shape != (3,):
        raise ValueError(f"position must be a 3-vector, got shape {r.shape}")
    theta = gmst_epoch + rotation_rate * float(time)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return np.array(
        [
            cos_t * r[0] - sin_t * r[1],
            sin_t * r[0] + cos_t * r[1],
            r[2],
        ]
    )


def eci_to_ecef(
    position: ArrayLike,
    time: float,
    gmst_epoch: float = 0.0,
    rotation_rate: float = _ROTATION_RATE,
) -> _FloatArray:
    """Rotate an inertial vector into the Earth-fixed frame at ``time``."""
    return ecef_to_eci(position, -float(time), -gmst_epoch, rotation_rate)


def geodetic_to_eci(
    position: GeodeticPosition,
    time: float,
    gmst_epoch: float = 0.0,
    rotation_rate: float = _ROTATION_RATE,
) -> _FloatArray:
    """Inertial position (m) of a ground point **at a stated epoch**.

    The epoch is not optional and is not the epoch of the calculation: it
    is the time at which the vehicle is there. A ground point moves through
    the inertial frame at 465 m/s at the equator, so an epoch error of one
    minute displaces the aimpoint by 28 km.
    """
    return ecef_to_eci(geodetic_to_ecef(position), time, gmst_epoch, rotation_rate)


def great_circle_range(origin: GeodeticPosition, destination: GeodeticPosition) -> float:
    """Surface range (m) between two points, spherical approximation.

    Haversine on a sphere of the mean radius. This is a **budgeting** tool:
    it errs by up to about 0.5% against the true ellipsoidal geodesic,
    roughly 30 km over a 6000 km arc. That is small against a glide's own
    range uncertainty and far too large to use as a terminal aimpoint.
    """
    lat1, lat2 = origin.latitude, destination.latitude
    dlat = lat2 - lat1
    dlon = destination.longitude - origin.longitude
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * WGS84_MEAN_RADIUS * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def great_circle_bearing(origin: GeodeticPosition, destination: GeodeticPosition) -> float:
    """Initial bearing (rad from north, positive east) along the great circle.

    This is the *initial* bearing: a great circle does not hold a constant
    heading, so the bearing at arrival differs, sometimes by a great deal
    on a long high-latitude arc.
    """
    dlon = destination.longitude - origin.longitude
    y = np.sin(dlon) * np.cos(destination.latitude)
    x = np.cos(origin.latitude) * np.sin(destination.latitude) - np.sin(origin.latitude) * np.cos(
        destination.latitude
    ) * np.cos(dlon)
    if x == 0.0 and y == 0.0:
        raise ValueError("origin and destination coincide, so no bearing is defined")
    return float(np.arctan2(y, x))
