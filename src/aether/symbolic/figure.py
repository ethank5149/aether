r"""The figure of the planet on symbols, and the passage from a published state to the model's.

The translational field carries the geocentric radius and latitude, because
gravity and the rotation are simplest in them. Two things outside the field
are not geocentric, and this module is where they meet it.

*The atmosphere lies on the reference ellipsoid.* Its density is a function
of height above that surface, not of :math:`r - R_e`: at Orion's splashdown
latitude the ellipsoid is 4.4 km inside the equatorial radius, which is most
of a scale height. The field of record measures altitude along the radius,
:math:`h = r - R(\phi)`
(:func:`aether.certification.rotating_field.altitude`). That is not the
geodetic altitude, which is measured along the ellipsoid's normal, and
:meth:`FigureSymbols.radial_height_excess` is the exact difference: it has no
term of order :math:`f` and its leading term is
:math:`\tfrac12 f^2\,\dfrac{a\,h}{a+h}\sin^2 2\varphi`, seventy centimetres
at 130 km.

*A published state is geodetic and inertial.* Flight reports give an entry
interface as a geodetic altitude and latitude, and an inertial velocity by
its magnitude, flight-path angle and azimuth in the topocentric frame, whose
vertical is the ellipsoid's normal. The model's state is the geocentric
radius and latitude and the velocity relative to the rotating planet, by its
magnitude, flight-path angle and heading in the geocentric local frame.
:meth:`FigureSymbols.model_state` is the passage between them, in three
steps that each have a name: the geodetic point, the rotation of the
topocentric frame onto the geocentric one through the deflection of the
vertical, and the subtraction of :math:`\boldsymbol\omega_E\times\mathbf r`.
The deflection is not small for an entry: at Orion's interface it is
:math:`0.15^\circ`, and a flight-path angle is specified to a hundredth of a
degree.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import sympy as sp

from aether.certification.rotating_field import Planet, surface_radius
from aether.symbolic.entry import EntrySymbols
from aether.symbolic.ops import SYMPY_OPS

__all__ = ["FigureSymbols"]

_PLANET = EntrySymbols().planet
_GEODETIC_LATITUDE = sp.Symbol("varphi", real=True)
_GEODETIC_ALTITUDE = sp.Symbol("h_g", real=True)
_LONGITUDE = sp.Symbol("lambda", real=True)
_INERTIAL_SPEED = sp.Symbol("V_I", positive=True)
_INERTIAL_GAMMA = sp.Symbol("gamma_I", real=True)
_INERTIAL_AZIMUTH = sp.Symbol("psi_I", real=True)


@dataclass(frozen=True)
class FigureSymbols:
    """A geodetic point with an inertial topocentric velocity, and what the model makes of it."""

    planet: Planet = _PLANET
    geodetic_latitude: Any = _GEODETIC_LATITUDE
    geodetic_altitude: Any = _GEODETIC_ALTITUDE
    longitude: Any = _LONGITUDE
    inertial_speed: Any = _INERTIAL_SPEED
    inertial_gamma: Any = _INERTIAL_GAMMA
    inertial_azimuth: Any = _INERTIAL_AZIMUTH
    geodetic_horizon: bool = True
    """Whether the published flight-path angle and azimuth are referred to the
    geodetic horizon, as a topocentric frame is, or to the geocentric one."""

    @property
    def published(self) -> tuple[Any, ...]:
        """The six published elements, in the order the conversion takes them."""
        return (
            self.geodetic_altitude,
            self.longitude,
            self.geodetic_latitude,
            self.inertial_speed,
            self.inertial_gamma,
            self.inertial_azimuth,
        )

    # -- the ellipsoid ------------------------------------------------------------

    @property
    def eccentricity_squared(self) -> Any:
        r""":math:`e^2 = f(2 - f)`."""
        return self.planet.flattening * (2 - self.planet.flattening)

    @cached_property
    def prime_vertical(self) -> Any:
        r""":math:`N = a/\sqrt{1 - e^2\sin^2\varphi}`: the normal's length to the axis."""
        return self.planet.radius / sp.sqrt(
            1 - self.eccentricity_squared * sp.sin(self.geodetic_latitude) ** 2
        )

    @cached_property
    def meridian_point(self) -> tuple[Any, Any]:
        r"""The point in its meridian plane: distance from the axis, and height above
        the equator.

        :math:`\bigl((N + h_g)\cos\varphi,\ (N(1 - e^2) + h_g)\sin\varphi\bigr)`.
        """
        normal, height = self.prime_vertical, self.geodetic_altitude
        return (
            (normal + height) * sp.cos(self.geodetic_latitude),
            (normal * (1 - self.eccentricity_squared) + height)
            * sp.sin(self.geodetic_latitude),
        )

    @cached_property
    def radius(self) -> Any:
        """Geocentric radius of the point."""
        axis, height = self.meridian_point
        return sp.sqrt(axis**2 + height**2)

    @cached_property
    def latitude(self) -> Any:
        """Geocentric latitude of the point."""
        axis, height = self.meridian_point
        return sp.atan2(height, axis)

    @cached_property
    def deflection(self) -> Any:
        r""":math:`\varphi - \phi`: the angle from the radius to the ellipsoid's normal."""
        return self.geodetic_latitude - self.latitude

    @cached_property
    def radial_height_excess(self) -> Any:
        r"""Height along the radius, less the geodetic altitude: exact.

        The ellipsoid's radius is taken at the point's own geocentric
        latitude, whose cosine squared is the squared distance from the axis
        over :math:`r^2`, so no inverse trigonometric function appears.
        """
        axis, _ = self.meridian_point
        flattening = self.planet.flattening
        surface = self.planet.radius * (1 - flattening) / sp.sqrt(
            1 - self.eccentricity_squared * axis**2 / self.radius**2
        )
        return self.radius - surface - self.geodetic_altitude

    def ellipse_residual(self, latitude: Any) -> Any:
        r""":math:`x^2/a^2 + z^2/b^2 - 1` at the point the field calls the surface."""
        surface = surface_radius(latitude, SYMPY_OPS, planet=self.planet)
        semi_minor = self.planet.radius * (1 - self.planet.flattening)
        return (
            (surface * sp.cos(latitude)) ** 2 / self.planet.radius**2
            + (surface * sp.sin(latitude)) ** 2 / semi_minor**2
            - 1
        )

    # -- the velocity ---------------------------------------------------------------

    @cached_property
    def inertial_velocity_topocentric(self) -> Any:
        """Inertial velocity in north-east-down components of the published frame."""
        speed, gamma, azimuth = self.inertial_speed, self.inertial_gamma, self.inertial_azimuth
        return sp.Matrix(
            [
                speed * sp.cos(gamma) * sp.cos(azimuth),
                speed * sp.cos(gamma) * sp.sin(azimuth),
                -speed * sp.sin(gamma),
            ]
        )

    @cached_property
    def topocentric_to_geocentric(self) -> Any:
        r"""Rotation about east through the deflection :math:`\delta`.

        The geodetic north and down are the geocentric ones turned through
        :math:`\delta` in the meridian plane, so components transform by

        .. math::

            \begin{pmatrix} v_N \\ v_E \\ v_D \end{pmatrix}
            = \begin{pmatrix} \cos\delta & 0 & -\sin\delta \\ 0 & 1 & 0 \\
                \sin\delta & 0 & \cos\delta \end{pmatrix}
              \begin{pmatrix} v_N' \\ v_E' \\ v_D' \end{pmatrix} .

        The identity when the published frame is already geocentric.
        """
        if not self.geodetic_horizon:
            return sp.eye(3)
        delta = self.deflection
        return sp.Matrix(
            [
                [sp.cos(delta), 0, -sp.sin(delta)],
                [0, 1, 0],
                [sp.sin(delta), 0, sp.cos(delta)],
            ]
        )

    @cached_property
    def inertial_velocity(self) -> Any:
        """Inertial velocity in geocentric north-east-down components."""
        return self.topocentric_to_geocentric * self.inertial_velocity_topocentric

    @cached_property
    def frame_velocity(self) -> Any:
        r""":math:`\boldsymbol\omega_E\times\mathbf r`, in the same components: eastward."""
        return sp.Matrix(
            [0, self.planet.rotation_rate * self.radius * sp.cos(self.latitude), 0]
        )

    @cached_property
    def relative_velocity(self) -> Any:
        """Velocity relative to the rotating planet, geocentric north-east-down."""
        return self.inertial_velocity - self.frame_velocity

    # -- the model's state ------------------------------------------------------------

    @cached_property
    def model_state(self) -> tuple[Any, ...]:
        r""":math:`(r, \lambda, \phi, V, \gamma, \psi)`: the state the field integrates."""
        north, east, down = self.relative_velocity
        speed = sp.sqrt(north**2 + east**2 + down**2)
        return (
            self.radius,
            self.longitude,
            self.latitude,
            speed,
            sp.asin(-down / speed),
            sp.atan2(east, north),
        )
