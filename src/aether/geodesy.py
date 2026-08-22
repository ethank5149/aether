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
import pyproj
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "WGS84_ECCENTRICITY_SQUARED",
    "WGS84_FLATTENING",
    "WGS84_MEAN_RADIUS",
    "WGS84_POLAR_RADIUS",
    "WGS84_SEMI_MAJOR_AXIS",
    "GeodeticPosition",
    "central_angle",
    "central_bearing",
    "ecef_to_eci",
    "ecef_to_geodetic",
    "eci_to_ecef",
    "geocentric_to_geodetic_latitude",
    "geodesic_bearing",
    "geodesic_range",
    "geodesic_track",
    "geodesic_walk",
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

#: PROJ's WGS84 geodesic solver, built once -- construction dominates the cost of
#: an individual `inv` call.
_GEOD = pyproj.Geod(ellps="WGS84")


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


def geocentric_to_geodetic_latitude(latitude: ArrayLike) -> _FloatArray:
    r"""Geodetic latitude (rad) for a geocentric latitude on the ellipsoid.

    The inverse of :func:`geodetic_to_geocentric_latitude`:
    :math:`\tan\phi_d = \tan\phi_c / (1 - e^2)`.

    Needed wherever a spherical construction produces a latitude and something
    ellipsoidal then consumes it. Any formula that walks an *angular* distance
    along a great circle -- an orbital ground track swept in an inertial plane,
    for instance -- yields the angle of the radius vector, which is geocentric.
    Labelling that result geodetic without converting displaces it by up to
    0.19 degrees near 45 degrees latitude, **which is 21 km on the ground**.
    """
    phi = np.asarray(latitude, dtype=np.float64)
    return np.asarray(np.arctan(np.tan(phi) / (1.0 - WGS84_ECCENTRICITY_SQUARED)))


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
    """Great-circle range (m) on a sphere of the mean radius.

    Haversine, and **spherical on purpose** rather than as an approximation to
    be improved. A great circle is a spherical object; callers that want the
    distance across the real ellipsoid want :func:`geodesic_range`, which is a
    different quantity and says so in its name.

    The distinction is not pedantic, and conflating the two costs kilometres.
    :mod:`aether_gambit.orbital.scenario` is a two-body Keplerian model on a
    sphere of this radius -- true anomaly, Kepler's equation, ``body_radius =
    WGS84_MEAN_RADIUS`` -- and closes its aimpoint loop by converting a range to
    a central angle. Feeding it an ellipsoidal geodesic instead mixes two
    models, and a trajectory that had landed on its aimpoint to under a metre
    then missed by **15.6 km**. The ellipsoidal number is the more accurate
    description of the ground; it is the less accurate input to a spherical
    propagator, and both statements are true at once.
    """
    lat1, lat2 = origin.latitude, destination.latitude
    dlat = lat2 - lat1
    dlon = destination.longitude - origin.longitude
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * WGS84_MEAN_RADIUS * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def great_circle_bearing(origin: GeodeticPosition, destination: GeodeticPosition) -> float:
    """Initial bearing (rad from north, positive east) along the great circle.

    Spherical, matching :func:`great_circle_range`; the ellipsoidal counterpart
    is :func:`geodesic_bearing`. A range and a bearing must describe the same
    curve, or a point built from the pair lies off the path it claims to follow.

    This is the *initial* bearing: a great circle does not hold a constant
    heading, so the bearing at arrival differs, sometimes by a great deal on a
    long high-latitude arc.
    """
    dlon = destination.longitude - origin.longitude
    y = np.sin(dlon) * np.cos(destination.latitude)
    x = np.cos(origin.latitude) * np.sin(destination.latitude) - np.sin(origin.latitude) * np.cos(
        destination.latitude
    ) * np.cos(dlon)
    if x == 0.0 and y == 0.0:
        raise ValueError("origin and destination coincide, so no bearing is defined")
    return float(np.arctan2(y, x))


def geodesic_track(
    origin: GeodeticPosition,
    destination: GeodeticPosition,
    samples: int,
) -> list[GeodeticPosition]:
    """``samples`` points evenly spaced in distance along the WGS84 geodesic.

    Endpoints included and **exact**, which is the property that matters. A
    track laid by repeatedly stepping a direct-problem formula accumulates its
    error and arrives near the destination; this one arrives *at* it by
    construction, so "did the trajectory land on its aimpoint" stops being a
    question about accumulated round-off.

    Delegated to PROJ's ``inv_intermediate``. The back-azimuth convention is
    requested explicitly because pyproj 3.5 changed the default and warns
    otherwise -- and a warning is an error in this project's test suite.
    """
    if samples < 2:
        raise ValueError(
            f"a track needs at least its two endpoints, got samples={samples}"
        )
    result = _GEOD.inv_intermediate(
        np.rad2deg(origin.longitude), np.rad2deg(origin.latitude),
        np.rad2deg(destination.longitude), np.rad2deg(destination.latitude),
        samples, initial_idx=0, terminus_idx=0, return_back_azimuth=True,
    )
    return [
        GeodeticPosition(float(np.deg2rad(lat)), float(np.deg2rad(lon)))
        for lon, lat in zip(result.lons, result.lats, strict=True)
    ]


def geodesic_walk(
    origin: GeodeticPosition,
    azimuth: float,
    distance: float,
    samples: int,
) -> list[GeodeticPosition]:
    """``samples`` points along the geodesic leaving ``origin`` on ``azimuth``.

    The *direct* problem, where :func:`geodesic_track` is the inverse one. The
    distinction is not stylistic: an inverse track is defined by its two
    endpoints and can only ever take the short way between them, so a
    fractional-orbital ground track that deliberately goes the long way round
    cannot be expressed that way at all. This walks a stated distance on a
    stated bearing and lands where the ellipsoid puts it.

    **It does not close.** Walking the complement of a short-way distance on the
    reversed azimuth returns near the destination but not to it, because
    geodesics on an ellipsoid precess -- the long way and the short way are not
    two halves of one curve, as they are on a sphere. That is a property of the
    Earth rather than of this function, and a caller needing the long way to
    arrive exactly wants the orbital plane projected through
    :func:`ecef_to_geodetic`, not a walk.
    """
    if samples < 2:
        raise ValueError(
            f"a track needs at least its two endpoints, got samples={samples}"
        )
    if not (np.isfinite(distance) and distance > 0.0):
        raise ValueError(f"distance must be finite and > 0, got {distance}")
    result = _GEOD.fwd_intermediate(
        np.rad2deg(origin.longitude), np.rad2deg(origin.latitude),
        np.rad2deg(azimuth), samples, float(distance) / (samples - 1),
        initial_idx=0, terminus_idx=0, return_back_azimuth=True,
    )
    return [
        GeodeticPosition(float(np.deg2rad(lat)), float(np.deg2rad(lon)))
        for lon, lat in zip(result.lons, result.lats, strict=True)
    ]


def central_angle(origin: GeodeticPosition, destination: GeodeticPosition) -> float:
    r"""Geocentric central angle (rad) subtended by two surface points.

    The angle between the two radius vectors -- which is the quantity a
    Keplerian orbit actually sweeps, because a conic lives in the geocentric
    frame. Its ground track is where the position vector points, and that
    direction is a *geocentric* latitude regardless of the ellipsoid's shape.

    This is not :func:`geodesic_range` divided by a radius, and the difference
    is not small. A geodesic length is measured along the curved surface and
    divided by a mean radius that fits neither the equator nor the meridian; the
    central angle is exact and needs no radius at all. Feeding a propagator the
    former and then placing its ground track by the latter mixes two frames, and
    a fractional-orbital profile that had landed on its aimpoint then missed by
    tens of kilometres.

    Uses geocentric latitude on both ends, so a track swept in this angle and
    converted back through :func:`geocentric_to_geodetic_latitude` closes on its
    endpoints exactly.
    """
    lat1 = float(geodetic_to_geocentric_latitude(origin.latitude))
    lat2 = float(geodetic_to_geocentric_latitude(destination.latitude))
    dlon = destination.longitude - origin.longitude
    haversine = (
        np.sin(0.5 * (lat2 - lat1)) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(0.5 * dlon) ** 2
    )
    return float(2.0 * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0))))


def central_bearing(origin: GeodeticPosition, destination: GeodeticPosition) -> float:
    """Initial bearing (rad) of the great circle joining the two *geocentric* points.

    Pairs with :func:`central_angle`, in the same frame. Mixing a geodetic
    bearing with a geocentric angle tilts the swept plane slightly and the track
    drifts off its aimpoint along the way rather than at the end, which is
    harder to notice.
    """
    lat1 = float(geodetic_to_geocentric_latitude(origin.latitude))
    lat2 = float(geodetic_to_geocentric_latitude(destination.latitude))
    dlon = destination.longitude - origin.longitude
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    if x == 0.0 and y == 0.0:
        raise ValueError("origin and destination coincide, so no bearing is defined")
    return float(np.arctan2(y, x))


def geodesic_range(origin: GeodeticPosition, destination: GeodeticPosition) -> float:
    """True surface distance (m) across the WGS84 ellipsoid.

    PROJ's implementation of Karney (2013) for the inverse geodesic problem --
    exact to round-off, and convergent for the near-antipodal cases where
    Vincenty's iteration does not terminate.

    What :func:`great_circle_range` approximates, and by how much: measured
    against published geodesics the spherical figure errs by -0.26 % to +0.33 %,
    roughly 30 km over a 6000 km arc. Use this wherever a distance on the ground
    is the answer -- targeting, asset ranging, footprint extent -- and the
    spherical one only inside a model that is itself spherical.
    """
    return float(
        _GEOD.inv(
            np.rad2deg(origin.longitude), np.rad2deg(origin.latitude),
            np.rad2deg(destination.longitude), np.rad2deg(destination.latitude),
        )[2]
    )


def geodesic_bearing(origin: GeodeticPosition, destination: GeodeticPosition) -> float:
    """Initial bearing (rad from north, positive east) along the WGS84 geodesic.

    Ellipsoidal, matching :func:`geodesic_range`. Pair these two with each
    other, never one of each: a spherical bearing with an ellipsoidal range
    places a point off the curve it names, and the error grows with distance
    instead of announcing itself.
    """
    if (
        origin.latitude == destination.latitude
        and origin.longitude == destination.longitude
    ):
        raise ValueError("origin and destination coincide, so no bearing is defined")
    forward = _GEOD.inv(
        np.rad2deg(origin.longitude), np.rad2deg(origin.latitude),
        np.rad2deg(destination.longitude), np.rad2deg(destination.latitude),
    )[0]
    return float(np.deg2rad(forward))
