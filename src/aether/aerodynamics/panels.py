"""Panel model, integrated loads, and trim (Paper II, §3.3).

The distributed forcing on the collocation grid is
:math:`\\mathbf{Q}_{\\mathrm{aero}} = q_{\\mathrm{dyn}}\\mathbf{C}_p`
(Eq. 3.6). This module integrates that over a panelized surface to give
normal force and pitching moment, and solves for the trim incidence —
the quantity II-V4 measures against blend width.

The bundled :func:`curved_lifting_body` is a *generic demonstration
geometry*: a cambered surface with elliptical spanwise loading, chosen
because its shoulder line sweeps through :math:`\\delta_c = 0` as
incidence varies, which is exactly the configuration where the blending
seam of :mod:`aether.aerodynamics.closure` can affect integrated loads.
It corresponds to no vehicle and carries no design data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.optimize
from numpy.typing import NDArray

from aether.aerodynamics.closure import blended_pressure_coefficient

__all__ = ["PanelModel", "TrimSolution", "curved_lifting_body"]

_FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TrimSolution:
    """Result of a trim solve about a moment reference point."""

    incidence: float
    """Trim angle of attack (rad)."""
    normal_force: float
    """Normal force at trim (N)."""
    axial_force: float
    """Axial force at trim (N)."""
    pitching_moment: float
    """Residual pitching moment at trim (N·m), zero to solver tolerance."""
    converged: bool


@dataclass(frozen=True)
class PanelModel:
    """A panelized surface with outward normals and areas.

    Attributes
    ----------
    centroids:
        Panel centroids in body axes, shape ``(n, 3)`` (m).
    normals:
        Outward unit normals, shape ``(n, 3)``.
    areas:
        Panel areas, shape ``(n,)`` (m²).
    reference_point:
        Moment reference point in body axes (m).
    """

    centroids: _FloatArray = field(repr=False)
    normals: _FloatArray = field(repr=False)
    areas: _FloatArray = field(repr=False)
    reference_point: _FloatArray = field(
        repr=False, default_factory=lambda: np.zeros(3)
    )

    def __post_init__(self) -> None:
        c = np.asarray(self.centroids, dtype=np.float64)
        n = np.asarray(self.normals, dtype=np.float64)
        a = np.asarray(self.areas, dtype=np.float64)
        if c.ndim != 2 or c.shape[1] != 3:
            raise ValueError(f"centroids must have shape (n, 3), got {c.shape}")
        if n.shape != c.shape:
            raise ValueError(f"normals shape {n.shape} does not match centroids {c.shape}")
        if a.shape != (c.shape[0],):
            raise ValueError(f"areas must have shape ({c.shape[0]},), got {a.shape}")
        if np.any(a <= 0.0):
            raise ValueError("panel areas must be strictly positive")
        norms = np.linalg.norm(n, axis=1)
        if np.max(np.abs(norms - 1.0)) > 1e-9:
            raise ValueError("normals must be unit vectors")

    @property
    def n_panels(self) -> int:
        return int(self.areas.size)

    @property
    def total_area(self) -> float:
        return float(np.sum(self.areas))

    def velocity_direction(self, incidence: float, sideslip: float = 0.0) -> _FloatArray:
        """Unit freestream direction in body axes for a given attitude.

        With :math:`\\alpha` positive nose-up, the flow arrives from
        ahead and below: :math:`\\hat{\\mathbf{v}}_B =
        (\\cos\\alpha\\cos\\beta, \\sin\\beta, \\sin\\alpha)`
        normalized.
        """
        a, b = float(incidence), float(sideslip)
        v = np.array(
            [np.cos(a) * np.cos(b), np.sin(b), np.sin(a) * np.cos(b)]
        )
        return np.asarray(v / np.linalg.norm(v))

    def incidences(self, incidence: float, sideslip: float = 0.0) -> _FloatArray:
        """Per-panel local incidence (rad); positive windward."""
        v_hat = self.velocity_direction(incidence, sideslip)
        sin_delta = -(self.normals @ v_hat)
        return np.asarray(np.arcsin(np.clip(sin_delta, -1.0, 1.0)))

    def loads(
        self,
        incidence: float,
        mach: float,
        dynamic_pressure: float,
        gamma: float = 1.4,
        blend_width: float = 0.02,
        sideslip: float = 0.0,
        cp_max: float | None = None,
    ) -> tuple[_FloatArray, _FloatArray]:
        """Integrated force and moment in body axes.

        ``cp_max`` overrides the perfect-gas Rayleigh-Pitot stagnation value.
        Above about Mach 8 that value is wrong — equilibrium air reaches
        1.93 at Mach 20 where the perfect gas is stuck near its 1.839
        asymptote — and :class:`~aether.aerodynamics.realgas.EquilibriumAir`
        supplies the right one.

        Pressure acts along :math:`-\\mathbf{n}`, so the panel force is
        :math:`-C_p\\,q_{\\mathrm{dyn}}A\\,\\mathbf{n}`.

        Returns
        -------
        tuple
            ``(force, moment)``, each shape ``(3,)``, in N and N·m.
        """
        if not (np.isfinite(dynamic_pressure) and dynamic_pressure > 0.0):
            raise ValueError(
                f"dynamic_pressure must be finite and > 0, got {dynamic_pressure}"
            )
        delta = self.incidences(incidence, sideslip)
        cp = blended_pressure_coefficient(
            delta, mach, gamma=gamma, blend_width=blend_width, cp_max=cp_max
        )
        panel_force = -(cp * dynamic_pressure * self.areas)[:, np.newaxis] * self.normals
        force = np.sum(panel_force, axis=0)
        arms = self.centroids - self.reference_point
        moment = np.sum(np.cross(arms, panel_force), axis=0)
        return np.asarray(force), np.asarray(moment)

    def pitching_moment(
        self,
        incidence: float,
        mach: float,
        dynamic_pressure: float,
        gamma: float = 1.4,
        blend_width: float = 0.02,
        cp_max: float | None = None,
    ) -> float:
        """Body-axis pitching moment (about :math:`y`) at a given incidence."""
        _, moment = self.loads(
            incidence,
            mach,
            dynamic_pressure,
            gamma=gamma,
            blend_width=blend_width,
            cp_max=cp_max,
        )
        return float(moment[1])

    def trim(
        self,
        mach: float,
        dynamic_pressure: float,
        gamma: float = 1.4,
        blend_width: float = 0.02,
        bracket: tuple[float, float] = (np.deg2rad(-20.0), np.deg2rad(30.0)),
    ) -> TrimSolution:
        """Incidence at which the pitching moment vanishes.

        Brent's method on the stated bracket. A missing sign change is
        reported as an error rather than returning a bracket endpoint:
        an untrimmable configuration is a real result, and silently
        returning the edge of the search would hide it.
        """
        lo, hi = float(bracket[0]), float(bracket[1])
        if not lo < hi:
            raise ValueError(f"bracket must satisfy lo < hi, got {bracket}")

        def moment(alpha: float) -> float:
            return self.pitching_moment(
                alpha, mach, dynamic_pressure, gamma=gamma, blend_width=blend_width
            )

        m_lo, m_hi = moment(lo), moment(hi)
        if m_lo * m_hi > 0.0:
            raise ValueError(
                f"no trim point on [{np.rad2deg(lo):.2f}, {np.rad2deg(hi):.2f}] deg: "
                f"pitching moment is {m_lo:.4e} and {m_hi:.4e} N·m at the ends"
            )
        alpha, info = scipy.optimize.brentq(
            moment, lo, hi, xtol=1e-12, rtol=8.9e-16, full_output=True
        )
        force, mom = self.loads(
            alpha, mach, dynamic_pressure, gamma=gamma, blend_width=blend_width
        )
        return TrimSolution(
            incidence=float(alpha),
            normal_force=float(force[2]),
            axial_force=float(force[0]),
            pitching_moment=float(mom[1]),
            converged=bool(info.converged),
        )


def curved_lifting_body(
    length: float = 6.0,
    span: float = 3.0,
    upper_thickness: float = 0.45,
    lower_thickness: float = 0.15,
    n_chord: int = 40,
    n_span: int = 24,
    reference_fraction: float = 0.5,
) -> PanelModel:
    """Generic cambered lifting body — a demonstration geometry.

    Upper and lower surfaces are
    :math:`z = \\pm t\\,s(x)\\,c(y)` with a smooth chordwise shape
    :math:`s` and elliptical spanwise distribution :math:`c`. The camber
    (``upper_thickness`` > ``lower_thickness``) gives a non-zero moment
    at zero incidence, so the trim point is non-trivial, and the
    curvature makes the shoulder line sweep through
    :math:`\\delta_c = 0` — the configuration where the blending seam
    can matter.

    Corresponds to no vehicle; carries no design or performance data.
    """
    for name, val in (
        ("length", length),
        ("span", span),
        ("upper_thickness", upper_thickness),
        ("lower_thickness", lower_thickness),
    ):
        if not (np.isfinite(val) and val > 0.0):
            raise ValueError(f"{name} must be finite and > 0, got {val}")
    if n_chord < 4 or n_span < 4:
        raise ValueError(f"need n_chord, n_span >= 4, got ({n_chord}, {n_span})")

    xi = np.linspace(0.0, 1.0, n_chord + 1)
    eta = np.linspace(-1.0, 1.0, n_span + 1)

    def surface(u: _FloatArray, v: _FloatArray, thickness: float, sign: float) -> _FloatArray:
        shape = np.sin(np.pi * u) ** 0.9  # blunt-nosed, smoothly closing aft
        spanwise = np.sqrt(np.maximum(1.0 - v * v, 0.0))
        return sign * thickness * shape * spanwise

    centroids: list[_FloatArray] = []
    normals: list[_FloatArray] = []
    areas: list[float] = []

    for thickness, sign in ((upper_thickness, 1.0), (lower_thickness, -1.0)):
        for i in range(n_chord):
            for j in range(n_span):
                u = np.array([xi[i], xi[i + 1], xi[i + 1], xi[i]])
                v = np.array([eta[j], eta[j], eta[j + 1], eta[j + 1]])
                corners = np.column_stack(
                    [length * u, 0.5 * span * v, surface(u, v, thickness, sign)]
                )
                # split the quad into two triangles for an exact area/normal
                for tri in ((0, 1, 2), (0, 2, 3)):
                    p = corners[list(tri)]
                    cross = np.cross(p[1] - p[0], p[2] - p[0])
                    area = 0.5 * float(np.linalg.norm(cross))
                    if area <= 0.0:
                        continue
                    normal = cross / np.linalg.norm(cross)
                    if normal[2] * sign < 0.0:  # orient outward
                        normal = -normal
                    centroids.append(p.mean(axis=0))
                    normals.append(normal)
                    areas.append(area)

    return PanelModel(
        centroids=np.asarray(centroids),
        normals=np.asarray(normals),
        areas=np.asarray(areas),
        reference_point=np.array([reference_fraction * length, 0.0, 0.0]),
    )
