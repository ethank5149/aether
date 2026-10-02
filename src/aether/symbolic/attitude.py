r"""Attitude on symbols: the quaternion, the body rate, and the passage to the wind axes.

The translational field takes the aerodynamic force already resolved on the
wind triad, :math:`(F_V, F_\gamma, F_\psi)`. In the six-degree-of-freedom
model those three numbers are not inputs. They are what the attitude
quaternion, the aerodynamic coefficients and the direction of flight make of
a force that is known in *body* axes, and the manuscript's dynamics appendix
sets out the chain: body force, body-to-local rotation, projection on the
wind triad. This module states that chain once, on symbols, with every
matrix written as the appendix displays it.

Unlike :mod:`aether.symbolic.entry`, nothing here is evaluated from a field
written for every arithmetic, because for the attitude there is none: the
flight simulator carries its attitude against an inertial frame in Cartesian
components, and the appendix carries it against the rotating frame. What the
two share is the quaternion kinematics and the direction-cosine matrix, and
the tests pin the expressions below to :mod:`aether.dynamics.attitude` on
those. Everything else in this module is the manuscript's statement, and the
derivations check that statement against itself and against the translational
field of record.

Conventions, as in the appendix: scalar-first quaternion; :math:`\mathbf
R(\mathbf q) = \mathbf R_{\mathcal E\mathcal B}` carries body components to
planet-fixed ones; the body rate :math:`\boldsymbol\omega = (p, q_b, r_b)`
is the *inertial* angular velocity of the body in body components; body axes
are forward, right, down.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import sympy as sp

from aether.symbolic.entry import EntrySymbols

__all__ = ["AttitudeSymbols", "skew"]

_QUATERNION = sp.symbols("q0 q1 q2 q3", real=True)
_BODY_RATE = sp.symbols("p q_b r_b", real=True)
_COEFFICIENTS = sp.symbols("C_A C_Y C_N", real=True)
_PRESSURE_AREA = sp.Symbol("q_dyn_S", positive=True)
_ALPHA = sp.Symbol("alpha", real=True)
_SIDESLIP = sp.Symbol("beta_s", real=True)
_BANK = sp.Symbol("sigma", real=True)


def skew(vector: Any) -> Any:
    r"""The matrix :math:`[\mathbf a\times]`, for which
    :math:`[\mathbf a\times]\mathbf b = \mathbf a\times\mathbf b`."""
    a1, a2, a3 = vector
    return sp.Matrix([[0, -a3, a2], [a3, 0, -a1], [-a2, a1, 0]])


@dataclass(frozen=True)
class AttitudeSymbols:
    """The attitude state, and the aerodynamic force it puts on the wind triad."""

    entry: EntrySymbols = field(default_factory=EntrySymbols)
    quaternion: tuple[Any, Any, Any, Any] = _QUATERNION
    body_rate: tuple[Any, Any, Any] = _BODY_RATE
    coefficients: tuple[Any, Any, Any] = _COEFFICIENTS
    """Axial, side and normal force coefficients :math:`(C_A, C_Y, C_N)`."""
    pressure_area: Any = _PRESSURE_AREA
    r"""Dynamic pressure times reference area, :math:`q_{\mathrm{dyn}} S`."""
    alpha: Any = _ALPHA
    sideslip: Any = _SIDESLIP
    bank: Any = _BANK

    # -- the quaternion, exactly as the appendix displays it -------------------

    @property
    def norm_squared(self) -> Any:
        r""":math:`\lVert\mathbf q\rVert^2`."""
        return sum(component**2 for component in self.quaternion)

    @cached_property
    def dcm(self) -> Any:
        r""":math:`\mathbf R(\mathbf q)`, quadratic in the quaternion: body to planet-fixed.

        Its columns are the body axes in planet-fixed components. For a
        quaternion off the unit sphere it is :math:`\lVert\mathbf q\rVert^2`
        times a rotation, which the derivations keep track of instead of
        assuming it away.
        """
        q0, q1, q2, q3 = self.quaternion
        return sp.Matrix(
            [
                [q0**2 + q1**2 - q2**2 - q3**2, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
                [2 * (q1 * q2 + q0 * q3), q0**2 - q1**2 + q2**2 - q3**2, 2 * (q2 * q3 - q0 * q1)],
                [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), q0**2 - q1**2 - q2**2 + q3**2],
            ]
        )

    @staticmethod
    def rate_matrix(rate: Any) -> Any:
        r""":math:`\boldsymbol\Omega(\mathbf a)` of the quaternion kinematics."""
        a1, a2, a3 = rate
        return sp.Matrix(
            [[0, -a1, -a2, -a3], [a1, 0, a3, -a2], [a2, -a3, 0, a1], [a3, a2, -a1, 0]]
        )

    def quaternion_rate(self, rate: Any) -> Any:
        r""":math:`\dot{\mathbf q} = \tfrac12\boldsymbol\Omega(\mathbf a)\,\mathbf q`,
        for a rate given in body axes."""
        return self.rate_matrix(rate) * sp.Matrix(self.quaternion) / 2

    @cached_property
    def relative_rate(self) -> Any:
        r"""Body rate relative to the rotating frame, :math:`\boldsymbol\omega -
        \mathbf R(\mathbf q)^{\mathsf T}\boldsymbol\omega_E`.

        The quaternion relates the body to the *rotating* frame, so this is
        the rate that drives it, not the inertial rate the state carries.
        """
        return sp.Matrix(self.body_rate) - self.dcm.T * self.entry.rotation

    # -- frames -----------------------------------------------------------------

    @cached_property
    def local_frame(self) -> Any:
        r""":math:`\mathbf R_{\mathcal E\mathcal L}`: columns north, east and down.

        Typed rather than taken from :class:`EntrySymbols`, because the claim
        to be checked is that the basis the appendix displays is the one the
        translational field was derived on.
        """
        lam, phi = self.entry.longitude, self.entry.latitude
        north = [-sp.sin(phi) * sp.cos(lam), -sp.sin(phi) * sp.sin(lam), sp.cos(phi)]
        east = [-sp.sin(lam), sp.cos(lam), 0]
        down = [-sp.cos(phi) * sp.cos(lam), -sp.cos(phi) * sp.sin(lam), -sp.sin(phi)]
        return sp.Matrix([north, east, down]).T

    @cached_property
    def body_to_local(self) -> Any:
        r""":math:`\mathbf R_{\mathcal L\mathcal B} =
        \mathbf R_{\mathcal E\mathcal L}^{\mathsf T}\,\mathbf R(\mathbf q)`."""
        return self.local_frame.T * self.dcm

    @cached_property
    def wind_axes(self) -> tuple[Any, Any, Any]:
        r""":math:`(\hat V, \hat n_\gamma, \hat n_\psi)` in north-east-down components."""
        return self.entry.wind_triad_ned

    # -- the aerodynamic force --------------------------------------------------

    @property
    def body_force(self) -> Any:
        r""":math:`\mathbf F_a^{\mathcal B} = q_{\mathrm{dyn}} S\,(-C_A,\ C_Y,\ -C_N)`."""
        axial, side, normal = self.coefficients
        return self.pressure_area * sp.Matrix([-axial, side, -normal])

    def wind_force(self, body_to_local: Any) -> tuple[Any, Any, Any]:
        r""":math:`(F_V, F_\gamma, F_\psi)`: the body force, rotated and projected.

        ``body_to_local`` is the rotation the attitude supplies. Passing it in
        lets the same chain be evaluated for the quaternion's rotation and for
        an attitude given by its incidence and bank.
        """
        local = body_to_local * self.body_force
        return tuple(local.dot(axis) for axis in self.wind_axes)

    def body_velocity(self, body_to_local: Any) -> Any:
        r""":math:`\mathbf V^{\mathcal B} =
        \mathbf R_{\mathcal L\mathcal B}^{\mathsf T}\,\mathbf V_{\mathcal L}`."""
        v_hat, _, _ = self.wind_axes
        return body_to_local.T * (self.entry.speed * v_hat)

    # -- an attitude given by incidence and bank ----------------------------------

    @cached_property
    def banked_attitude(self) -> Any:
        r""":math:`\mathbf R_{\mathcal L\mathcal B}` of the attitude at
        :math:`(\alpha, \beta, \sigma)`.

        The wind frame is banked first: forward along :math:`\hat V`, right
        along :math:`\cos\sigma\,\hat n_\psi - \sin\sigma\,\hat n_\gamma`, down
        along their cross product. The body is then set at the incidence that
        gives the velocity the body components
        :math:`V(\cos\alpha\cos\beta,\ \sin\beta,\ \sin\alpha\cos\beta)`.
        This is the attitude the point-mass model has in mind when it speaks
        of an angle of attack and a bank angle, written out so that the
        identification :math:`F_V = -D`, :math:`F_\gamma = L\cos\sigma`,
        :math:`F_\psi = L\sin\sigma` can be derived instead of asserted.
        """
        v_hat, n_gamma, n_psi = self.wind_axes
        right = sp.cos(self.bank) * n_psi - sp.sin(self.bank) * n_gamma
        down = v_hat.cross(right)
        wind_to_local = sp.Matrix.hstack(v_hat, right, down)
        ca, sa = sp.cos(self.alpha), sp.sin(self.alpha)
        cb, sb = sp.cos(self.sideslip), sp.sin(self.sideslip)
        pitch = sp.Matrix([[ca, 0, -sa], [0, 1, 0], [sa, 0, ca]])
        yaw = sp.Matrix([[cb, -sb, 0], [sb, cb, 0], [0, 0, 1]])
        wind_to_body = pitch * yaw
        return wind_to_local * wind_to_body.T

    @property
    def drag(self) -> Any:
        r""":math:`D = q_{\mathrm{dyn}} S\,(C_A\cos\alpha + C_N\sin\alpha)`, at zero sideslip."""
        axial, _, normal = self.coefficients
        return self.pressure_area * (axial * sp.cos(self.alpha) + normal * sp.sin(self.alpha))

    @property
    def lift(self) -> Any:
        r""":math:`L = q_{\mathrm{dyn}} S\,(C_N\cos\alpha - C_A\sin\alpha)`, at zero sideslip."""
        axial, _, normal = self.coefficients
        return self.pressure_area * (normal * sp.cos(self.alpha) - axial * sp.sin(self.alpha))
