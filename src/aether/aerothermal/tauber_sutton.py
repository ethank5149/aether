"""Published Tauber–Sutton radiative heating relations.

Source, verified against the PDF in ``reference/``:

    M. E. Tauber and K. Sutton, "Stagnation-Point Radiative Heating
    Relations for Earth and Mars Entries", *Journal of Spacecraft and
    Rockets* **28**(1), 1991, pp. 40–42. DOI 10.2514/3.26206.
    Bibliography key ``tauber1991``; see ``CITATION-AUDIT.md``.

The correlation is Eq. (1) of that paper,

.. math::

    \\dot q_r = C\\, r_n^{\\,a}\\, \\rho^{\\,b}\\, f(V),

with :math:`\\dot q_r` in **W/cm²**, :math:`r_n` the hemispherical nose
radius in m, :math:`\\rho` the freestream density in kg/m³ and
:math:`V` in m/s. For air, Eq. (2) gives

.. math::

    C = 4.736\\times 10^{4}, \\qquad
    a = 1.072\\times 10^{6}\\,V^{-1.88}\\rho^{-0.325}, \\qquad
    b = 1.22,

with :math:`a` additionally **capped by nose radius**:

.. math::

    1 \\le r_n \\le 2:\\ a \\le 0.6, \\qquad
    2 < r_n \\le 3:\\ a \\le 0.5,

and the paper states in addition that **a < 1 must always be met**. The
caps are not decorative — inside the stated envelope the unclamped
expression reaches ~0.74 at the low-velocity, low-density corner, so the
:math:`r_n \\ge 1` cases are genuinely clamped there.

For the Martian atmosphere, Eq. (3) uses constant exponents:
:math:`C = 2.35\\times10^{4}`, :math:`a = 0.526`, :math:`b = 1.19`.

Three things this module gets right that are easy to get wrong:

**Units.** The correlation returns W/cm². Everything else in this
package is SI, so :func:`earth_radiative_heat_flux` converts by
:math:`10^4` and says so. Forgetting the conversion is a four-order-of-
magnitude error that still "looks like" a heat flux.

**The exponent is not a constant.** Paper II, §4.2 describes the
correlation with ":math:`a \\approx 1.0`". The source makes :math:`a` a
function of both velocity and density, capped by nose radius, and
required to stay *below* one; over the paper's own validity envelope it
runs about 0.25–0.6, never 1.0. The discrepancy is recorded here and in
the verification report rather than silently reconciled.

**The table extends past the correlation's validity.** Table 1 tabulates
:math:`f_E` from 9000 m/s, but Eqs. (1)–(2) are stated to apply from
**10 to 16 km/s**, for densities :math:`6.66\\times10^{-5}` to
:math:`6.31\\times10^{-4}` kg/m³ and nose radii 0.3–3 m. Interpolating
the table below 10 km/s is legitimate; applying the *correlation* there
is not, and :func:`earth_radiative_heat_flux` enforces the envelope.

"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "EARTH_EXPONENT_CAPS",
    "EARTH_VELOCITY_FUNCTION",
    "MARS_VELOCITY_FUNCTION",
    "TAUBER_SUTTON_PROVENANCE",
    "earth_radiative_heat_flux",
    "earth_radiative_heating_exponent",
    "earth_velocity_function",
    "mars_radiative_heat_flux",
    "radiative_heat_transfer_coefficient",
]

_FloatArray = NDArray[np.float64]

#: Provenance string carried by every object built from this table.
TAUBER_SUTTON_PROVENANCE = (
    "Tauber & Sutton, J. Spacecraft and Rockets 28(1), 1991, pp. 40-42, "
    "Tables 1 and Eqs. (1)-(4), DOI 10.2514/3.26206; transcribed from the "
    "archived "
    "PDF in reference/ and cross-checked against the layout extraction"
)

#: Table 1, Earth column: velocity (m/s) and the velocity function f_E.
EARTH_VELOCITY_FUNCTION: tuple[tuple[float, float], ...] = (
    (9000.0, 1.5),
    (9250.0, 4.3),
    (9500.0, 9.7),
    (9750.0, 19.5),
    (10000.0, 35.0),
    (10250.0, 55.0),
    (10500.0, 81.0),
    (10750.0, 115.0),
    (11000.0, 151.0),
    (11500.0, 238.0),
    (12000.0, 359.0),
    (12500.0, 495.0),
    (13000.0, 660.0),
    (13500.0, 850.0),
    (14000.0, 1065.0),
    (14500.0, 1313.0),
    (15000.0, 1550.0),
    (15500.0, 1780.0),
    (16000.0, 2040.0),
)

#: Table 1, Mars column: velocity (m/s) and the velocity function f_M.
MARS_VELOCITY_FUNCTION: tuple[tuple[float, float], ...] = (
    (6000.0, 0.2),
    (6150.0, 1.0),
    (6300.0, 1.95),
    (6500.0, 3.42),
    (6700.0, 5.1),
    (6900.0, 7.1),
    (7000.0, 8.1),
    (7200.0, 10.2),
    (7400.0, 12.5),
    (7600.0, 14.8),
    (7800.0, 17.1),
    (8000.0, 19.2),
    (8200.0, 21.4),
    (8400.0, 24.1),
    (8600.0, 26.0),
    (8800.0, 28.9),
    (9000.0, 32.8),
)

#: Eq. (2) constants for air.
EARTH_COEFFICIENT = 4.736e4
EARTH_DENSITY_EXPONENT = 1.22
_EARTH_A_COEFFICIENT = 1.072e6
_EARTH_A_VELOCITY_EXPONENT = -1.88
_EARTH_A_DENSITY_EXPONENT = -0.325

#: Eq. (2) nose-radius caps on the exponent a: (upper r_n bound, cap).
EARTH_EXPONENT_CAPS: tuple[tuple[float, float], ...] = ((2.0, 0.6), (3.0, 0.5))

#: Eq. (3) constants for the Martian atmosphere (97% CO2, 3% N2).
MARS_COEFFICIENT = 2.35e4
MARS_RADIUS_EXPONENT = 0.526
MARS_DENSITY_EXPONENT = 1.19
MARS_VELOCITY_RANGE = (6500.0, 9000.0)
MARS_DENSITY_RANGE = (1.0e-4, 1.0e-3)

#: Stated validity envelope of Eqs. (1)–(2) for air.
EARTH_VELOCITY_RANGE = (10000.0, 16000.0)
EARTH_DENSITY_RANGE = (6.66e-5, 6.31e-4)
EARTH_NOSE_RADIUS_RANGE = (0.3, 3.0)

#: W/cm² to W/m².
_WCM2_TO_WM2 = 1.0e4


def _table_arrays(table: tuple[tuple[float, float], ...]) -> tuple[_FloatArray, _FloatArray]:
    data = np.asarray(table, dtype=np.float64)
    return np.ascontiguousarray(data[:, 0]), np.ascontiguousarray(data[:, 1])


def earth_velocity_function(velocity: ArrayLike) -> _FloatArray:
    """:math:`f_E(V)` by **linear** interpolation of Table 1.

    The paper states linear interpolation explicitly; a smoother scheme
    would be a different correlation, so this uses what the source
    prescribes. Velocities outside the tabulated span raise.
    """
    v_table, f_table = _table_arrays(EARTH_VELOCITY_FUNCTION)
    v = np.asarray(velocity, dtype=np.float64)
    if np.any(v < v_table[0]) or np.any(v > v_table[-1]):
        raise ValueError(
            f"velocity outside the tabulated span "
            f"[{v_table[0]:g}, {v_table[-1]:g}] m/s; the table is not "
            f"extrapolatable"
        )
    return np.asarray(np.interp(v, v_table, f_table))


def earth_radiative_heating_exponent(
    velocity: ArrayLike, density: ArrayLike, nose_radius: ArrayLike | None = None
) -> _FloatArray:
    """The nose-radius exponent :math:`a` of Eq. (2) for air.

    With ``nose_radius`` supplied the published caps are applied —
    :math:`a \\le 0.6` for :math:`1 \\le r_n \\le 2` and
    :math:`a \\le 0.5` for :math:`2 < r_n \\le 3`. Without it the raw
    expression is returned, which is what the caps act on.
    """
    v = np.asarray(velocity, dtype=np.float64)
    rho = np.asarray(density, dtype=np.float64)
    if np.any(v <= 0.0) or np.any(rho <= 0.0):
        raise ValueError("velocity and density must be strictly positive")
    exponent = (
        _EARTH_A_COEFFICIENT
        * v**_EARTH_A_VELOCITY_EXPONENT
        * rho**_EARTH_A_DENSITY_EXPONENT
    )
    if nose_radius is None:
        return np.asarray(exponent)
    r_n = np.asarray(nose_radius, dtype=np.float64)
    exponent, r_n = np.broadcast_arrays(np.asarray(exponent), r_n)
    exponent = np.array(exponent, dtype=np.float64, copy=True)
    # Band edges follow the source exactly: "if 1 <= r_n <= 2" is closed at
    # both ends, "if 2 < r_n <= 3" is open below, so r_n = 2 takes the 0.6
    # cap and not the 0.5 one. Treating the boundary as belonging to the
    # upper band would silently under-predict at exactly r_n = 2 m.
    lower, inclusive = 1.0, True
    for upper, cap in EARTH_EXPONENT_CAPS:
        above = (r_n >= lower) if inclusive else (r_n > lower)
        in_band = above & (r_n <= upper)
        exponent[in_band] = np.minimum(exponent[in_band], cap)
        lower, inclusive = upper, False
    return np.asarray(exponent)


def mars_radiative_heat_flux(
    nose_radius: ArrayLike,
    density: ArrayLike,
    velocity: ArrayLike,
    enforce_envelope: bool = True,
) -> _FloatArray:
    """Stagnation radiative heat flux for Mars, Eqs. (1) and (3), in **W/m²**.

    Constant exponents, unlike the Earth case: :math:`a = 0.526`,
    :math:`b = 1.19`. Valid for 6.5–9 km/s and densities
    :math:`10^{-4}` to :math:`10^{-3}` kg/m³.
    """
    r_n = np.asarray(nose_radius, dtype=np.float64)
    rho = np.asarray(density, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    if np.any(r_n <= 0.0):
        raise ValueError("nose_radius must be strictly positive")
    if enforce_envelope:
        for name, value, (low, high) in (
            ("velocity", v, MARS_VELOCITY_RANGE),
            ("density", rho, MARS_DENSITY_RANGE),
        ):
            if np.any(value < low) or np.any(value > high):
                raise ValueError(
                    f"{name} outside the validity envelope [{low:g}, {high:g}] "
                    f"stated for Eqs. (1) and (3)"
                )
    v_table, f_table = _table_arrays(MARS_VELOCITY_FUNCTION)
    if np.any(v < v_table[0]) or np.any(v > v_table[-1]):
        raise ValueError("velocity outside the tabulated span of f_M")
    q_wcm2 = (
        MARS_COEFFICIENT
        * r_n**MARS_RADIUS_EXPONENT
        * rho**MARS_DENSITY_EXPONENT
        * np.interp(v, v_table, f_table)
    )
    return np.asarray(q_wcm2 * _WCM2_TO_WM2)


def radiative_heat_transfer_coefficient(
    heat_flux: ArrayLike, density: ArrayLike, velocity: ArrayLike
) -> _FloatArray:
    """Dimensionless :math:`C_{H_r} = \\dot q_r / (\\tfrac{1}{2}\\rho V^3)`,
    Eq. (4) of the source.

    ``heat_flux`` is SI (W/m²), matching this package rather than the
    paper's W/cm².
    """
    q = np.asarray(heat_flux, dtype=np.float64)
    rho = np.asarray(density, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    if np.any(rho <= 0.0) or np.any(v <= 0.0):
        raise ValueError("density and velocity must be strictly positive")
    return np.asarray(q / (0.5 * rho * v**3))


def earth_radiative_heat_flux(
    nose_radius: ArrayLike,
    density: ArrayLike,
    velocity: ArrayLike,
    enforce_envelope: bool = True,
) -> _FloatArray:
    """Stagnation radiative heat flux for air, Eqs. (1)–(2), in **W/m²**.

    Parameters
    ----------
    nose_radius:
        Hemispherical nose radius :math:`r_n` (m).
    density:
        Freestream density :math:`\\rho` (kg/m³).
    velocity:
        Flight velocity :math:`V` (m/s).
    enforce_envelope:
        Reject inputs outside the paper's stated validity ranges. Setting
        this false is a deliberate act of extrapolating a fit beyond
        where its authors validated it, and the caller owns that.

    Notes
    -----
    The source returns W/cm²; the :math:`10^4` conversion to SI is
    applied here. The :math:`a < 1` requirement stated in the paper is
    checked and violations raise, because an :math:`a \\ge 1` evaluation
    is outside the correlation's construction, not merely inaccurate.
    """
    r_n = np.asarray(nose_radius, dtype=np.float64)
    rho = np.asarray(density, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    if np.any(r_n <= 0.0):
        raise ValueError("nose_radius must be strictly positive")

    if enforce_envelope:
        for name, value, (low, high) in (
            ("velocity", v, EARTH_VELOCITY_RANGE),
            ("density", rho, EARTH_DENSITY_RANGE),
            ("nose_radius", r_n, EARTH_NOSE_RADIUS_RANGE),
        ):
            if np.any(value < low) or np.any(value > high):
                raise ValueError(
                    f"{name} outside the validity envelope [{low:g}, {high:g}] "
                    f"stated for Eqs. (1)-(2); pass enforce_envelope=False to "
                    f"extrapolate deliberately"
                )

    exponent = earth_radiative_heating_exponent(v, rho, r_n)
    if np.any(exponent >= 1.0):
        raise ValueError(
            f"the nose-radius exponent evaluated to {np.max(exponent):.3f}; the "
            f"source requires a < 1 for the correlation to apply"
        )
    q_wcm2 = (
        EARTH_COEFFICIENT
        * r_n**exponent
        * rho**EARTH_DENSITY_EXPONENT
        * earth_velocity_function(v)
    )
    return np.asarray(q_wcm2 * _WCM2_TO_WM2)
