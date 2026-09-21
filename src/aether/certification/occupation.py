r"""Occupation-measure certificate for the certified downrange bound.

The dual of the Lasserre occupation-measure relaxation: a polynomial
certificate W(x) satisfying nabla W . f(x,u) + V^2 cos gamma <= 0 for all
(x,u) in the corridor, so that along any trajectory from x_0 to x_T the
remaining range satisfies R <= W(x_0) - W(x_T).  If W >= 0 on the terminal
set, then R <= W(x_0).

All variables are normalised to [-1,1] before building the SDP so the
Gram matrices stay well-conditioned (V and y span orders of magnitude in
physical units).

References: Lasserre-Henrion-Prieur-Trelat, SIAM J. Control Optim. 2008;
Henrion-Korda, IEEE Trans. Autom. Control 2014.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np

from aether.certification.sos import (
    Polynomial,
    constant_poly,
    monomial_basis,
    poly_add,
    poly_degree,
    poly_differentiate,
    poly_multiply,
    poly_substitute_affine,
    sos_polynomial,
    variable_poly,
)

__all__ = [
    "CorridorParams",
    "OccupationBound",
    "corridor_certificate",
    "corridor_moment_bound",
]

_RHO0 = 1.225
_H_SCALE = 8500.0
_MU = 3.986004418e14
_R_EARTH = 6378137.0
_G0 = 9.80665
_C_SUTTON_GRAVES = 1.7415e-4  # W/m² per (kg/m³/m)^0.5·(m/s)³

# Variable indices in the polynomial system (V, y, s, c, u).
_V, _Y, _S, _C, _U = 0, 1, 2, 3, 4
_N_VARS = 5


def _uniform_moment(a: float, b: float, n: int) -> float:
    """E[x^n] for x ~ Uniform(a, b).  Reduces to a^n when a == b (Dirac)."""
    if n == 0:
        return 1.0
    if abs(b - a) < 1e-15 * (abs(a) + abs(b) + 1e-30):
        return a ** n
    return (b ** (n + 1) - a ** (n + 1)) / ((n + 1) * (b - a))


def _initial_moments(
    x0b: list[float],
    x0_lo: list[float] | None,
    x0_hi: list[float] | None,
    alpha: tuple[int, ...],
) -> float:
    """Product of marginal moments E[x̄^α] for the initial measure.

    When ``x0_lo``/``x0_hi`` are None, returns the point evaluation x̄₀^α.
    """
    val = 1.0
    for k in range(len(alpha)):
        if alpha[k] == 0:
            continue
        if x0_lo is not None and x0_hi is not None:
            val *= _uniform_moment(x0_lo[k], x0_hi[k], alpha[k])
        else:
            val *= x0b[k] ** alpha[k]
    return val


@dataclass(frozen=True)
class CorridorParams:
    """Physical parameters for the polynomial entry corridor."""

    speed_high_ms: float
    speed_low_ms: float
    altitude_ceiling_m: float
    altitude_floor_m: float
    ballistic_coefficient: float
    lift_to_drag: float
    gamma_max_rad: float = np.pi / 2.0
    q_max_pa: float | None = None
    heat_rate_max_w_m2: float | None = None
    nose_radius_m: float = 1.0
    g_load_max: float | None = None
    terminal_altitude_max_m: float | None = None

    @property
    def y_min(self) -> float:
        """Density-square-root at the ceiling (thinnest air)."""
        return float(np.exp(-self.altitude_ceiling_m / (2.0 * _H_SCALE)))

    @property
    def y_max(self) -> float:
        """Density-square-root at the floor (thickest air)."""
        return float(np.exp(-self.altitude_floor_m / (2.0 * _H_SCALE)))

    @property
    def s_max(self) -> float:
        return float(np.sin(self.gamma_max_rad))


def _dynamics(params: CorridorParams) -> list[Polynomial]:
    r"""The :math:`\tau`-time polynomial dynamics :math:`f(V, y, s, c, u)`.

    Returns ``[f_V, f_y, f_s, f_c]`` — the right-hand side of the four state
    equations. Each is a polynomial in five variables ``(V, y, s, c, u)``.

    The :math:`\tau`-time is defined by :math:`dt = V\,d\tau` (i.e.
    :math:`d\tau = dt/V`), which clears the :math:`1/V` from the FPA equation.
    The range rate becomes :math:`dR/d\tau = V^2 \cos\gamma = V^2 c`.

    .. math::

        \frac{dV}{d\tau} &= -D_0\, y^2 V^3 - g\, V\, s \\
        \frac{dy}{d\tau} &= -\frac{y\, V^2\, s}{2H} \\
        \frac{ds}{d\tau} &= c\,\bigl[(L/D)\, D_0\, y^2\, V^2\, u
                            + (V^2/R_E - g)\, c\bigr] \\
        \frac{dc}{d\tau} &= -s\,\bigl[(L/D)\, D_0\, y^2\, V^2\, u
                            + (V^2/R_E - g)\, c\bigr]
    """
    beta = params.ballistic_coefficient
    ld = params.lift_to_drag
    D0 = _RHO0 / (2.0 * beta)

    def _var(i: int) -> Polynomial:
        return variable_poly(_N_VARS, i)

    V, y, s, c, u = (_var(i) for i in range(_N_VARS))
    one = constant_poly(_N_VARS, 1.0)

    # y² V²
    y2 = poly_multiply(y, y)
    V2 = poly_multiply(V, V)
    y2V2 = poly_multiply(y2, V2)

    # D₀ y² V³ = D₀ y² V² · V
    y2V3 = poly_multiply(y2V2, V)

    # f_V = -D₀ y² V³ - g V s
    Vs = poly_multiply(V, s)
    f_V = poly_add(
        poly_add({}, y2V3, scale_q=-D0),
        Vs,
        scale_q=-_G0,
    )

    # f_y = -y V² s / (2H)
    yV2s = poly_multiply(poly_multiply(y, V2), s)
    f_y = poly_add({}, yV2s, scale_q=-1.0 / (2.0 * _H_SCALE))

    # The flight-path bracket: (L/D) D₀ y² V² u + (V²/R_E - g) c
    bracket_lift = poly_multiply(y2V2, u)
    bracket_lift = poly_add({}, bracket_lift, scale_q=ld * D0)
    V2_over_R = poly_add({}, V2, scale_q=1.0 / _R_EARTH)
    centrifugal_minus_g = poly_add(V2_over_R, one, scale_q=-_G0)
    bracket_gravity = poly_multiply(centrifugal_minus_g, c)
    bracket = poly_add(bracket_lift, bracket_gravity)

    # f_s = c · bracket
    f_s = poly_multiply(c, bracket)

    # f_c = -s · bracket
    f_c = poly_multiply(s, bracket)
    f_c = {mono: -coeff for mono, coeff in f_c.items()}

    return [f_V, f_y, f_s, f_c]


_N_VARS_CR = 8
_P, _Q, _U1, _U2 = 4, 5, 6, 7


def _dynamics_cross_range(params: CorridorParams) -> list[Polynomial]:
    r"""8-variable τ-time dynamics with heading: (V, y, s, c, p, q, u₁, u₂).

    Uses the cos γ ≈ 1 approximation for the heading rate, valid for
    shallow entry (|γ| ≲ 20°).  The bank angle splits into u₁ = cos σ
    (longitudinal lift, changes γ) and u₂ = sin σ (lateral lift, changes ψ).

    f_p = ld · D₀ · y² · V² · u₂ · q
    f_q = −ld · D₀ · y² · V² · u₂ · p
    """
    nv = _N_VARS_CR
    beta = params.ballistic_coefficient
    ld = params.lift_to_drag
    D0 = _RHO0 / (2.0 * beta)

    def _var(i: int) -> Polynomial:
        return variable_poly(nv, i)

    V, y, s, c, p, q, u1, u2 = (_var(i) for i in range(nv))
    one = constant_poly(nv, 1.0)

    y2 = poly_multiply(y, y)
    V2 = poly_multiply(V, V)
    y2V2 = poly_multiply(y2, V2)
    y2V3 = poly_multiply(y2V2, V)

    Vs = poly_multiply(V, s)
    f_V = poly_add(poly_add({}, y2V3, scale_q=-D0), Vs, scale_q=-_G0)

    yV2s = poly_multiply(poly_multiply(y, V2), s)
    f_y = poly_add({}, yV2s, scale_q=-1.0 / (2.0 * _H_SCALE))

    bracket_lift = poly_add({}, poly_multiply(y2V2, u1), scale_q=ld * D0)
    V2_over_R = poly_add({}, V2, scale_q=1.0 / _R_EARTH)
    centrifugal_minus_g = poly_add(V2_over_R, one, scale_q=-_G0)
    bracket = poly_add(bracket_lift, poly_multiply(centrifugal_minus_g, c))

    f_s = poly_multiply(c, bracket)
    f_c = {mono: -coeff for mono, coeff in poly_multiply(s, bracket).items()}

    y2V2_u2 = poly_multiply(y2V2, u2)
    f_p = poly_add({}, poly_multiply(y2V2_u2, q), scale_q=ld * D0)
    f_q = poly_add({}, poly_multiply(y2V2_u2, p), scale_q=-ld * D0)

    return [f_V, f_y, f_s, f_c, f_p, f_q]


def _lie_derivative(W: Polynomial, dynamics: list[Polynomial]) -> Polynomial:
    r"""Lie derivative :math:`\nabla W \cdot f = \sum_i (\partial W/\partial x_i) f_i`."""
    result: Polynomial = {}
    for i, f_i in enumerate(dynamics):
        dW_dxi = poly_differentiate(W, i)
        if dW_dxi:
            result = poly_add(result, poly_multiply(dW_dxi, f_i))
    return result


@dataclass(frozen=True)
class OccupationBound:
    """A certified range bound from the occupation-measure dual."""

    max_downrange_m: float
    certificate_degree: int
    corridor: CorridorParams
    sdp_status: str
    W_coefficients: dict[tuple[int, ...], float]

    def summary(self) -> str:
        return (
            f"downrange at most {self.max_downrange_m / 1e3:,.0f} km "
            f"(degree-{self.certificate_degree} certificate, "
            f"SDP {self.sdp_status})"
        )


def _occupation_constraint_polys_cr(
    params: CorridorParams,
    offsets: list[float],
    scales: list[float],
) -> list[Polynomial]:
    """Constraint polynomials for the 8-variable cross-range system."""
    nv = _N_VARS_CR
    constraints: list[Polynomial] = []

    if params.q_max_pa is not None:
        y2V2_raw = poly_multiply(
            poly_multiply(variable_poly(nv, _Y), variable_poly(nv, _Y)),
            poly_multiply(variable_poly(nv, _V), variable_poly(nv, _V)),
        )
        g_q_raw = poly_add(
            constant_poly(nv, params.q_max_pa), y2V2_raw, scale_q=-0.5 * _RHO0
        )
        constraints.append(poly_substitute_affine(g_q_raw, offsets, scales))

    if params.g_load_max is not None:
        D0 = _RHO0 / (2.0 * params.ballistic_coefficient)
        C_n = D0 * np.sqrt(1.0 + params.lift_to_drag ** 2)
        y2V2_gload = poly_multiply(
            poly_multiply(variable_poly(nv, _Y), variable_poly(nv, _Y)),
            poly_multiply(variable_poly(nv, _V), variable_poly(nv, _V)),
        )
        g_n_raw = poly_add(
            constant_poly(nv, params.g_load_max * _G0),
            y2V2_gload, scale_q=-C_n,
        )
        constraints.append(poly_substitute_affine(g_n_raw, offsets, scales))

    if params.heat_rate_max_w_m2 is not None:
        yV3_raw = poly_multiply(
            variable_poly(nv, _Y),
            poly_multiply(
                poly_multiply(variable_poly(nv, _V), variable_poly(nv, _V)),
                variable_poly(nv, _V),
            ),
        )
        C_h = _C_SUTTON_GRAVES * np.sqrt(_RHO0 / params.nose_radius_m)
        g_h_raw = poly_add(
            constant_poly(nv, params.heat_rate_max_w_m2), yV3_raw, scale_q=-C_h,
        )
        constraints.append(poly_substitute_affine(g_h_raw, offsets, scales))

    return constraints


def _normalise_bounds_cr(params: CorridorParams) -> tuple[list[float], list[float]]:
    """Affine maps for the 8-variable cross-range system to [-1,1]."""
    V_lo, V_hi = params.speed_low_ms, params.speed_high_ms
    y_lo, y_hi = params.y_min, params.y_max
    s_max = params.s_max
    return (
        [(V_lo + V_hi) / 2, (y_lo + y_hi) / 2, 0, 0, 0, 0, 0, 0],
        [(V_hi - V_lo) / 2, (y_hi - y_lo) / 2,
         s_max if s_max > 0 else 1.0, 1, 1, 1, 1, 1],
    )


def _normalised_dynamics_cr(params: CorridorParams) -> tuple[list[Polynomial], Polynomial]:
    """Cross-range dynamics and objective in normalised 8-variable coordinates."""
    dynamics_raw = _dynamics_cross_range(params)
    offsets, scales = _normalise_bounds_cr(params)
    dynamics_norm = []
    for i in range(6):
        f_sub = poly_substitute_affine(dynamics_raw[i], offsets, scales)
        dynamics_norm.append({m: c / scales[i] for m, c in f_sub.items()})
    # V² c p = V² cos γ sin ψ
    cross_range_raw: Polynomial = {(2, 0, 0, 1, 1, 0, 0, 0): 1.0}
    cross_range_norm = poly_substitute_affine(cross_range_raw, offsets, scales)
    return dynamics_norm, cross_range_norm


def _normalise_bounds(params: CorridorParams) -> tuple[list[float], list[float]]:
    """Affine maps x_i = offset_i + scale_i * x_bar_i sending each variable to [-1,1]."""
    V_lo, V_hi = params.speed_low_ms, params.speed_high_ms
    y_lo, y_hi = params.y_min, params.y_max
    s_max = params.s_max
    offsets = [
        (V_lo + V_hi) / 2.0,
        (y_lo + y_hi) / 2.0,
        0.0,
        0.0,
        0.0,
    ]
    scales = [
        (V_hi - V_lo) / 2.0,
        (y_hi - y_lo) / 2.0,
        s_max if s_max > 0 else 1.0,
        1.0,
        1.0,
    ]
    return offsets, scales


def _normalised_dynamics(params: CorridorParams) -> tuple[list[Polynomial], Polynomial]:
    r"""Entry dynamics and range rate in normalised coordinates.

    Every variable x_bar_i lies in [-1, 1].  The returned dynamics f_bar
    satisfy dx_bar_i / dtau = f_bar_i(x_bar).  The range rate is
    dR/dtau = V^2 cos gamma  (since dtau = dt/V, so dR = V cos gamma dt = V^2 c dtau).
    """
    dynamics_raw = _dynamics(params)
    offsets, scales = _normalise_bounds(params)

    dynamics_norm: list[Polynomial] = []
    for i in range(4):
        f_sub = poly_substitute_affine(dynamics_raw[i], offsets, scales)
        dynamics_norm.append({m: c / scales[i] for m, c in f_sub.items()})

    range_rate_raw: Polynomial = {(2, 0, 0, 1, 0): 1.0}  # V^2 * c
    range_rate_norm = poly_substitute_affine(range_rate_raw, offsets, scales)

    return dynamics_norm, range_rate_norm


def _occupation_constraint_polys(
    params: CorridorParams,
    offsets: list[float],
    scales: list[float],
) -> list[Polynomial]:
    r"""Constraint polynomials :math:`g_i(\bar x) \ge 0` for the occupation measure.

    Each polynomial restricts which states the trajectory may visit.
    Returned polynomials are in the 5-variable normalised coordinate system.
    """
    constraints: list[Polynomial] = []

    if params.q_max_pa is not None:
        y2V2_raw = poly_multiply(
            poly_multiply(variable_poly(_N_VARS, _Y), variable_poly(_N_VARS, _Y)),
            poly_multiply(variable_poly(_N_VARS, _V), variable_poly(_N_VARS, _V)),
        )
        g_q_raw = poly_add(
            constant_poly(_N_VARS, params.q_max_pa), y2V2_raw, scale_q=-0.5 * _RHO0
        )
        constraints.append(poly_substitute_affine(g_q_raw, offsets, scales))

    if params.g_load_max is not None:
        D0 = _RHO0 / (2.0 * params.ballistic_coefficient)
        C_n = D0 * np.sqrt(1.0 + params.lift_to_drag**2)
        y2V2_gload = poly_multiply(
            poly_multiply(variable_poly(_N_VARS, _Y), variable_poly(_N_VARS, _Y)),
            poly_multiply(variable_poly(_N_VARS, _V), variable_poly(_N_VARS, _V)),
        )
        g_n_raw = poly_add(
            constant_poly(_N_VARS, params.g_load_max * _G0),
            y2V2_gload,
            scale_q=-C_n,
        )
        constraints.append(poly_substitute_affine(g_n_raw, offsets, scales))

    if params.heat_rate_max_w_m2 is not None:
        yV3_raw = poly_multiply(
            variable_poly(_N_VARS, _Y),
            poly_multiply(
                poly_multiply(
                    variable_poly(_N_VARS, _V), variable_poly(_N_VARS, _V)
                ),
                variable_poly(_N_VARS, _V),
            ),
        )
        C_h = _C_SUTTON_GRAVES * np.sqrt(_RHO0 / params.nose_radius_m)
        g_h_raw = poly_add(
            constant_poly(_N_VARS, params.heat_rate_max_w_m2),
            yV3_raw,
            scale_q=-C_h,
        )
        constraints.append(poly_substitute_affine(g_h_raw, offsets, scales))

    return constraints


def _terminal_constraint_polys(
    params: CorridorParams,
    offsets: list[float],
    scales: list[float],
) -> list[Polynomial]:
    r"""Constraint polynomials :math:`g_i(\bar x_T) \ge 0` for the terminal measure.

    Terminal variables are :math:`(\bar y, \bar s, \bar c)` — 3 variables.
    """
    _N_T = 3
    constraints: list[Polynomial] = []

    if params.terminal_altitude_max_m is not None:
        y_thresh = float(np.exp(-params.terminal_altitude_max_m / (2.0 * _H_SCALE)))
        yb_thresh = (y_thresh - offsets[_Y]) / scales[_Y]
        g_alt: Polynomial = {(1, 0, 0): 1.0, (0, 0, 0): -yb_thresh}
        constraints.append(g_alt)

    return constraints


def corridor_certificate(
    params: CorridorParams,
    *,
    degree: int = 4,
    entry_state: tuple[float, float, float, float] | None = None,
    solver: str | None = None,
    verbose: bool = False,
) -> OccupationBound:
    r"""Search for a polynomial certificate bounding the remaining downrange.

    All variables are normalised to [-1, 1] before building the SDP so the
    Gram matrices stay well-conditioned.

    Parameters
    ----------
    params:
        Physical corridor parameters.
    degree:
        Degree of the certificate polynomial :math:`W`. Higher is tighter but
        the SDP grows as :math:`\binom{n+d}{d}`.
    entry_state:
        ``(V_0, y_0, s_0, c_0)`` at the entry interface. If ``None``, the bound
        is optimised over the worst-case entry corner.
    solver:
        cvxpy solver name. ``None`` lets cvxpy choose.
    """
    offsets, scales = _normalise_bounds(params)
    dynamics_norm, range_rate_norm = _normalised_dynamics(params)

    # -- W(x_bar) polynomial in 4 normalised state variables ------------------
    state_vars = 4
    basis = monomial_basis(state_vars, degree)
    n_coeffs = len(basis)
    w_vec = cp.Variable(n_coeffs, name="w")
    W: Polynomial = {}
    for idx, mono in enumerate(basis):
        full_mono = mono + (0,)
        W[full_mono] = w_vec[idx]

    # -- Lie derivative in normalised coords ----------------------------------
    LW = _lie_derivative(W, dynamics_norm)

    # -- -(nabla W . f_bar + range_rate_bar) >= 0 on [-1,1]^5 cap {s^2+c^2=1} -
    neg_condition = {
        mono: -coeff
        for mono, coeff in poly_add(LW, range_rate_norm).items()
    }

    condition_degree = poly_degree(neg_condition)
    sos_degree = condition_degree if condition_degree % 2 == 0 else condition_degree + 1

    constraints_list: list[Any] = []

    # sigma_0: free SOS
    Q0, sigma0 = sos_polynomial(_N_VARS, sos_degree, name="Q0")
    constraints_list.append(Q0 >> 0)

    # Box constraints 1 - x_bar_i^2 >= 0 for each of 5 variables
    sigma_g_sum: Polynomial = {}
    for i in range(_N_VARS):
        g_i = poly_add(
            constant_poly(_N_VARS, 1.0),
            poly_multiply(variable_poly(_N_VARS, i), variable_poly(_N_VARS, i)),
            scale_q=-1.0,
        )
        mult_deg = sos_degree - 2
        if mult_deg < 0:
            continue
        mult_deg = mult_deg if mult_deg % 2 == 0 else mult_deg - 1
        if mult_deg < 0:
            continue
        Qi, sigma_i = sos_polynomial(_N_VARS, mult_deg, name=f"Qbox{i}")
        constraints_list.append(Qi >> 0)
        sigma_g_sum = poly_add(sigma_g_sum, poly_multiply(sigma_i, g_i))

    # Ideal term: p * (s_bar^2 + c_bar^2 - 1)  [the trig identity]
    s_bar2 = poly_multiply(
        variable_poly(_N_VARS, _S), variable_poly(_N_VARS, _S)
    )
    c_bar2 = poly_multiply(
        variable_poly(_N_VARS, _C), variable_poly(_N_VARS, _C)
    )
    # In normalised coords: s = s_max * s_bar, c = c_bar.
    # s^2 + c^2 = 1  =>  s_max^2 * s_bar^2 + c_bar^2 = 1
    s_max = params.s_max
    ideal_constraint = poly_add(
        poly_add(
            constant_poly(_N_VARS, -1.0),
            s_bar2,
            scale_q=s_max ** 2,
        ),
        c_bar2,
    )
    ideal_degree = sos_degree - 2
    if ideal_degree >= 0:
        ideal_basis = monomial_basis(_N_VARS, ideal_degree)
        p_vec = cp.Variable(len(ideal_basis), name="p")
        p_poly: Polynomial = {}
        for idx, mono in enumerate(ideal_basis):
            p_poly[mono] = p_vec[idx]
        ideal_product = poly_multiply(p_poly, ideal_constraint)
    else:
        ideal_product = {}

    # Dynamic pressure in normalised coords (if applicable)
    q_product: Polynomial = {}
    if params.q_max_pa is not None:
        y2V2_raw = poly_multiply(
            poly_multiply(variable_poly(_N_VARS, _Y), variable_poly(_N_VARS, _Y)),
            poly_multiply(variable_poly(_N_VARS, _V), variable_poly(_N_VARS, _V)),
        )
        g_q_raw = poly_add(
            constant_poly(_N_VARS, params.q_max_pa), y2V2_raw, scale_q=-0.5 * _RHO0
        )
        g_q = poly_substitute_affine(g_q_raw, offsets, scales)
        g_q_deg = poly_degree(g_q)
        q_mult_deg = sos_degree - g_q_deg
        if q_mult_deg >= 0:
            q_mult_deg = q_mult_deg if q_mult_deg % 2 == 0 else q_mult_deg - 1
            if q_mult_deg >= 0:
                Qq, sigma_q = sos_polynomial(_N_VARS, q_mult_deg, name="Qq")
                constraints_list.append(Qq >> 0)
                q_product = poly_multiply(sigma_q, g_q)

    rhs = poly_add(
        poly_add(poly_add(sigma0, sigma_g_sum), ideal_product),
        q_product,
    )

    all_monos = set(neg_condition.keys()) | set(rhs.keys())
    for mono in all_monos:
        lhs_coeff = neg_condition.get(mono, 0.0)
        rhs_coeff = rhs.get(mono, 0.0)
        constraints_list.append(lhs_coeff == rhs_coeff)

    # -- Terminal: W >= 0 at V = V_low, i.e. v_bar = -1 ----------------------
    W_terminal: Polynomial = {}
    for mono, coeff in W.items():
        v_power = mono[_V]
        remaining = (0,) + mono[1:]
        W_terminal = poly_add(
            W_terminal, {remaining: coeff * ((-1.0) ** v_power)}
        )
    terminal_degree = degree if degree % 2 == 0 else degree + 1
    Q_T, sigma_T = sos_polynomial(_N_VARS, terminal_degree, name="QT")
    constraints_list.append(Q_T >> 0)

    if terminal_degree >= 2:
        ideal_T_basis = monomial_basis(_N_VARS, terminal_degree - 2)
        pT_vec = cp.Variable(len(ideal_T_basis), name="pT")
        pT_poly: Polynomial = {}
        for idx, mono in enumerate(ideal_T_basis):
            pT_poly[mono] = pT_vec[idx]
        ideal_T_product = poly_multiply(pT_poly, ideal_constraint)
    else:
        ideal_T_product = {}

    rhs_T = poly_add(sigma_T, ideal_T_product)
    all_monos_T = set(W_terminal.keys()) | set(rhs_T.keys())
    for mono in all_monos_T:
        lhs_c = W_terminal.get(mono, 0.0)
        rhs_c = rhs_T.get(mono, 0.0)
        constraints_list.append(lhs_c == rhs_c)

    # -- Objective: minimise W(x_bar_0) in normalised coords -----------------
    if entry_state is not None:
        V0, y0, s0, c0 = entry_state
    else:
        V0 = params.speed_high_ms
        y0 = params.y_min
        s0 = 0.0
        c0 = 1.0

    vb = (V0 - offsets[_V]) / scales[_V]
    yb = (y0 - offsets[_Y]) / scales[_Y]
    sb = (s0 - offsets[_S]) / scales[_S]
    cb = (c0 - offsets[_C]) / scales[_C]

    W_at_entry = sum(
        coeff
        * (vb ** mono[_V])
        * (yb ** mono[_Y])
        * (sb ** mono[_S])
        * (cb ** mono[_C])
        for mono, coeff in W.items()
    )

    objective = cp.Minimize(W_at_entry)
    problem = cp.Problem(objective, constraints_list)

    solver_kwargs: dict[str, Any] = {}
    if solver is not None:
        solver_kwargs["solver"] = solver
    if verbose:
        solver_kwargs["verbose"] = True

    try:
        problem.solve(**solver_kwargs)
    except cp.error.SolverError:
        problem.solve(solver="SCS", verbose=verbose, max_iters=50_000)

    status = problem.status
    if w_vec.value is None:
        return OccupationBound(
            max_downrange_m=float("inf"),
            certificate_degree=degree,
            corridor=params,
            sdp_status=status,
            W_coefficients={},
        )

    W_values: dict[tuple[int, ...], float] = {}
    for idx, mono in enumerate(basis):
        full_mono = mono + (0,)
        val = float(w_vec.value[idx])
        if abs(val) > 1e-12:
            W_values[full_mono] = val

    return OccupationBound(
        max_downrange_m=max(0.0, float(problem.value)),
        certificate_degree=degree,
        corridor=params,
        sdp_status=status,
        W_coefficients=W_values,
    )


def _solve_moment_scs(
    params: CorridorParams,
    order: int,
    dynamics_norm: list[Polynomial],
    range_rate_norm: Polynomial,
    offsets: list[float],
    scales: list[float],
    entry_state: tuple[float, float, float, float] | None,
    verbose: bool,
    tau_multiplier: float = 10.0,
    occ_constraints: list[Polynomial] | None = None,
    term_constraints: list[Polynomial] | None = None,
    initial_half_widths: tuple[float, float, float, float] | None = None,
) -> tuple[float, str]:
    """Build and solve the moment SDP directly via SCS (no cvxpy)."""
    import scipy.sparse as sp
    import scs

    dyn_degs = [poly_degree(f) for f in dynamics_norm]

    occ_basis_2d = monomial_basis(_N_VARS, 2 * order)
    n_occ = len(occ_basis_2d)
    occ_idx: dict[tuple[int, ...], int] = {m: i for i, m in enumerate(occ_basis_2d)}

    _N_T = 3
    term_basis_2d = monomial_basis(_N_T, 2 * order)
    n_term = len(term_basis_2d)
    term_idx: dict[tuple[int, ...], int] = {m: i for i, m in enumerate(term_basis_2d)}

    _N_INIT = 4
    if initial_half_widths is not None:
        init_basis_2d = monomial_basis(_N_INIT, 2 * order)
        n_init = len(init_basis_2d)
        init_idx: dict[tuple[int, ...], int] = {
            m: i for i, m in enumerate(init_basis_2d)
        }
        init_off = n_occ + n_term
    else:
        n_init = 0
        init_idx = {}
        init_off = n_occ + n_term

    n_vars = n_occ + n_term + n_init

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b_list: list[float] = []
    row = 0

    def _add(r: int, c: int, v: float) -> None:
        rows.append(r)
        cols.append(c)
        vals.append(v)

    # ---- equality block (trig + Liouville) --------------------------------
    s_max = params.s_max

    for mono in monomial_basis(_N_VARS, max(0, 2 * order - 2)):
        ms = list(mono); ms[_S] += 2
        mc = list(mono); mc[_C] += 2
        _add(row, occ_idx[tuple(ms)], s_max ** 2)
        _add(row, occ_idx[tuple(mc)], 1.0)
        _add(row, occ_idx[mono], -1.0)
        b_list.append(0.0)
        row += 1

    for mono in monomial_basis(_N_T, max(0, 2 * order - 2)):
        ms = list(mono); ms[1] += 2
        mc = list(mono); mc[2] += 2
        _add(row, n_occ + term_idx[tuple(ms)], s_max ** 2)
        _add(row, n_occ + term_idx[tuple(mc)], 1.0)
        _add(row, n_occ + term_idx[mono], -1.0)
        b_list.append(0.0)
        row += 1

    if initial_half_widths is not None:
        _add(row, init_off + init_idx[(0, 0, 0, 0)], 1.0)
        b_list.append(1.0)
        row += 1
        for mono in monomial_basis(_N_INIT, max(0, 2 * order - 2)):
            ms = list(mono); ms[2] += 2
            mc = list(mono); mc[3] += 2
            _add(row, init_off + init_idx[tuple(ms)], s_max ** 2)
            _add(row, init_off + init_idx[tuple(mc)], 1.0)
            _add(row, init_off + init_idx[mono], -1.0)
            b_list.append(0.0)
            row += 1

    if entry_state is not None:
        V0, y0_es, s0, c0 = entry_state
    else:
        V0 = params.speed_high_ms
        y0_es = params.y_min
        s0 = 0.0
        c0 = 1.0
    x0b = [(V0 - offsets[0]) / scales[0], (y0_es - offsets[1]) / scales[1],
           (s0 - offsets[2]) / scales[2], (c0 - offsets[3]) / scales[3]]

    min_dyn = min(dyn_degs)
    broad_test = max(0, 2 * order - min_dyn + 1)
    for alpha_4 in monomial_basis(4, broad_test):
        p = sum(alpha_4)
        if any(alpha_4[i] > 0 and p - 1 + dyn_degs[i] > 2 * order for i in range(4)):
            continue
        tk = alpha_4[1:]
        if tk in term_idx:
            _add(row, n_occ + term_idx[tk], (-1.0) ** alpha_4[0])
        if initial_half_widths is not None:
            _add(row, init_off + init_idx[alpha_4], -1.0)
        for i in range(4):
            if alpha_4[i] == 0:
                continue
            ar = list(alpha_4); ar[i] -= 1
            for beta, fc in dynamics_norm[i].items():
                ok_t = (ar[0] + beta[0], ar[1] + beta[1],
                        ar[2] + beta[2], ar[3] + beta[3], beta[4])
                if ok_t in occ_idx:
                    _add(row, occ_idx[ok_t], -alpha_4[i] * fc)
        if initial_half_widths is not None:
            b_list.append(0.0)
        else:
            x0a = 1.0
            for k in range(4):
                if alpha_4[k] > 0:
                    x0a *= x0b[k] ** alpha_4[k]
            b_list.append(x0a)
        row += 1

    n_eq = row

    # ---- non-negative block (tau bound) -----------------------------------
    D0 = _RHO0 / (2.0 * params.ballistic_coefficient)
    tau_1d = (
        1.0 / (2.0 * params.speed_low_ms ** 2) - 1.0 / (2.0 * params.speed_high_ms ** 2)
    ) / (D0 * params.y_min ** 2)
    _add(row, occ_idx[(0,) * _N_VARS], 1.0)
    b_list.append(tau_multiplier * tau_1d)
    row += 1

    # ---- PSD blocks -------------------------------------------------------
    sqrt2 = np.sqrt(2.0)
    psd_sizes: list[int] = []

    def _add_psd(basis: tuple, idx_map: dict, var_off: int,
                 loc: bool = False, loc_var: int = 0) -> None:
        nonlocal row
        n = len(basis)
        psd_sizes.append(n)
        for j in range(n):
            for i in range(j, n):
                ms = tuple(a + b for a, b in zip(basis[i], basis[j]))
                sc = 1.0 if i == j else sqrt2
                if loc:
                    sh = list(ms); sh[loc_var] += 2
                    _add(row, var_off + idx_map[ms], -sc)
                    _add(row, var_off + idx_map[tuple(sh)], sc)
                else:
                    _add(row, var_off + idx_map[ms], -sc)
                b_list.append(0.0)
                row += 1

    def _add_psd_poly(basis: tuple, idx_map: dict, var_off: int,
                      g_poly: Polynomial) -> None:
        nonlocal row
        n = len(basis)
        psd_sizes.append(n)
        for j in range(n):
            for i in range(j, n):
                ms = tuple(a + b for a, b in zip(basis[i], basis[j]))
                sc = 1.0 if i == j else sqrt2
                for gamma, g_coeff in g_poly.items():
                    shifted = tuple(m + g for m, g in zip(ms, gamma))
                    if shifted in idx_map:
                        _add(row, var_off + idx_map[shifted], -sc * g_coeff)
                b_list.append(0.0)
                row += 1

    occ_d = monomial_basis(_N_VARS, order)
    _add_psd(occ_d, occ_idx, 0)

    term_d = monomial_basis(_N_T, order)
    _add_psd(term_d, term_idx, n_occ)

    if order >= 2:
        loc_o = monomial_basis(_N_VARS, order - 1)
        for vi in range(_N_VARS):
            _add_psd(loc_o, occ_idx, 0, loc=True, loc_var=vi)
        loc_t = monomial_basis(_N_T, order - 1)
        for vj in range(_N_T):
            _add_psd(loc_t, term_idx, n_occ, loc=True, loc_var=vj)

    for g_poly in (occ_constraints or []):
        g_deg = poly_degree(g_poly)
        loc_d = order - (g_deg + 1) // 2
        if loc_d >= 0:
            _add_psd_poly(
                monomial_basis(_N_VARS, loc_d), occ_idx, 0, g_poly
            )

    for g_poly in (term_constraints or []):
        g_deg = poly_degree(g_poly)
        loc_d = order - (g_deg + 1) // 2
        if loc_d >= 0:
            _add_psd_poly(
                monomial_basis(_N_T, loc_d), term_idx, n_occ, g_poly
            )

    if initial_half_widths is not None:
        hw = initial_half_widths
        x0_lo = [((v - h) - o) / s for v, h, o, s in
                 zip([V0, y0_es, s0, c0], hw, offsets, scales)]
        x0_hi = [((v + h) - o) / s for v, h, o, s in
                 zip([V0, y0_es, s0, c0], hw, offsets, scales)]
        init_d = monomial_basis(_N_INIT, order)
        _add_psd(init_d, init_idx, init_off)
        if order >= 2:
            loc_init = monomial_basis(_N_INIT, order - 1)
            for vi in range(_N_INIT):
                ai, bi = x0_lo[vi], x0_hi[vi]
                g_box: Polynomial = {
                    (0,) * _N_INIT: -ai * bi,
                    tuple(1 if j == vi else 0 for j in range(_N_INIT)): ai + bi,
                    tuple(2 if j == vi else 0 for j in range(_N_INIT)): -1.0,
                }
                _add_psd_poly(loc_init, init_idx, init_off, g_box)

    # ---- assemble ---------------------------------------------------------
    A = sp.csc_matrix((vals, (rows, cols)), shape=(row, n_vars))
    b = np.array(b_list, dtype=np.float64)

    c_vec = np.zeros(n_vars, dtype=np.float64)
    for mono, coeff in range_rate_norm.items():
        if mono in occ_idx:
            c_vec[occ_idx[mono]] = -coeff  # negate: SCS minimises

    cone = {"f": n_eq, "l": 1, "s": psd_sizes}

    solver = scs.SCS(
        {"A": A, "b": b, "c": c_vec},
        cone,
        max_iters=50_000,
        verbose=verbose,
    )
    sol = solver.solve()

    status_map = {
        scs.SOLVED: "optimal",
        scs.SOLVED_INACCURATE: "optimal_inaccurate",
        scs.INFEASIBLE: "infeasible",
        scs.INFEASIBLE_INACCURATE: "infeasible_inaccurate",
        scs.UNBOUNDED: "unbounded",
        scs.UNBOUNDED_INACCURATE: "unbounded_inaccurate",
    }
    status = status_map.get(sol["info"]["status_val"], str(sol["info"]["status"]))
    obj_val = -sol["info"]["pobj"]  # negate back: we maximise

    return obj_val, status


def _solve_moment_clarabel(
    params: CorridorParams,
    order: int,
    dynamics_norm: list[Polynomial],
    range_rate_norm: Polynomial,
    offsets: list[float],
    scales: list[float],
    entry_state: tuple[float, float, float, float] | None,
    verbose: bool,
    tau_multiplier: float = 10.0,
    occ_constraints: list[Polynomial] | None = None,
    term_constraints: list[Polynomial] | None = None,
    initial_half_widths: tuple[float, float, float, float] | None = None,
) -> tuple[float, str]:
    """Build and solve the moment SDP directly via Clarabel (no cvxpy).

    Same constraint construction as ``_solve_moment_scs`` but uses
    Clarabel's interior-point solver for tighter bounds.  The PSD cone
    vectorisation is upper-triangular column-major (Clarabel convention)
    vs lower-triangular column-major (SCS convention).
    """
    import clarabel
    import scipy.sparse as sp

    dyn_degs = [poly_degree(f) for f in dynamics_norm]

    occ_basis_2d = monomial_basis(_N_VARS, 2 * order)
    n_occ = len(occ_basis_2d)
    occ_idx: dict[tuple[int, ...], int] = {m: i for i, m in enumerate(occ_basis_2d)}

    _N_T = 3
    term_basis_2d = monomial_basis(_N_T, 2 * order)
    n_term = len(term_basis_2d)
    term_idx: dict[tuple[int, ...], int] = {m: i for i, m in enumerate(term_basis_2d)}

    _N_INIT = 4
    if initial_half_widths is not None:
        init_basis_2d = monomial_basis(_N_INIT, 2 * order)
        n_init = len(init_basis_2d)
        init_idx: dict[tuple[int, ...], int] = {
            m: i for i, m in enumerate(init_basis_2d)
        }
        init_off = n_occ + n_term
    else:
        n_init = 0
        init_idx = {}
        init_off = n_occ + n_term

    n_vars = n_occ + n_term + n_init

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b_list: list[float] = []
    row = 0

    def _add(r: int, c: int, v: float) -> None:
        rows.append(r)
        cols.append(c)
        vals.append(v)

    # ---- equality block (trig + Liouville) --------------------------------
    s_max = params.s_max

    for mono in monomial_basis(_N_VARS, max(0, 2 * order - 2)):
        ms = list(mono); ms[_S] += 2
        mc = list(mono); mc[_C] += 2
        _add(row, occ_idx[tuple(ms)], s_max ** 2)
        _add(row, occ_idx[tuple(mc)], 1.0)
        _add(row, occ_idx[mono], -1.0)
        b_list.append(0.0)
        row += 1

    for mono in monomial_basis(_N_T, max(0, 2 * order - 2)):
        ms = list(mono); ms[1] += 2
        mc = list(mono); mc[2] += 2
        _add(row, n_occ + term_idx[tuple(ms)], s_max ** 2)
        _add(row, n_occ + term_idx[tuple(mc)], 1.0)
        _add(row, n_occ + term_idx[mono], -1.0)
        b_list.append(0.0)
        row += 1

    if initial_half_widths is not None:
        _add(row, init_off + init_idx[(0, 0, 0, 0)], 1.0)
        b_list.append(1.0)
        row += 1
        for mono in monomial_basis(_N_INIT, max(0, 2 * order - 2)):
            ms = list(mono); ms[2] += 2
            mc = list(mono); mc[3] += 2
            _add(row, init_off + init_idx[tuple(ms)], s_max ** 2)
            _add(row, init_off + init_idx[tuple(mc)], 1.0)
            _add(row, init_off + init_idx[mono], -1.0)
            b_list.append(0.0)
            row += 1

    if entry_state is not None:
        V0, y0_es, s0, c0 = entry_state
    else:
        V0 = params.speed_high_ms
        y0_es = params.y_min
        s0 = 0.0
        c0 = 1.0
    x0b = [(V0 - offsets[0]) / scales[0], (y0_es - offsets[1]) / scales[1],
           (s0 - offsets[2]) / scales[2], (c0 - offsets[3]) / scales[3]]

    min_dyn = min(dyn_degs)
    broad_test = max(0, 2 * order - min_dyn + 1)
    for alpha_4 in monomial_basis(4, broad_test):
        p = sum(alpha_4)
        if any(alpha_4[i] > 0 and p - 1 + dyn_degs[i] > 2 * order for i in range(4)):
            continue
        tk = alpha_4[1:]
        if tk in term_idx:
            _add(row, n_occ + term_idx[tk], (-1.0) ** alpha_4[0])
        if initial_half_widths is not None:
            _add(row, init_off + init_idx[alpha_4], -1.0)
        for i in range(4):
            if alpha_4[i] == 0:
                continue
            ar = list(alpha_4); ar[i] -= 1
            for beta, fc in dynamics_norm[i].items():
                ok_t = (ar[0] + beta[0], ar[1] + beta[1],
                        ar[2] + beta[2], ar[3] + beta[3], beta[4])
                if ok_t in occ_idx:
                    _add(row, occ_idx[ok_t], -alpha_4[i] * fc)
        if initial_half_widths is not None:
            b_list.append(0.0)
        else:
            x0a = 1.0
            for k in range(4):
                if alpha_4[k] > 0:
                    x0a *= x0b[k] ** alpha_4[k]
            b_list.append(x0a)
        row += 1

    n_eq = row

    # ---- non-negative block (tau bound) -----------------------------------
    D0 = _RHO0 / (2.0 * params.ballistic_coefficient)
    tau_1d = (
        1.0 / (2.0 * params.speed_low_ms ** 2) - 1.0 / (2.0 * params.speed_high_ms ** 2)
    ) / (D0 * params.y_min ** 2)
    _add(row, occ_idx[(0,) * _N_VARS], 1.0)
    b_list.append(tau_multiplier * tau_1d)
    row += 1

    # ---- PSD blocks (upper-triangular column-major for Clarabel) ----------
    sqrt2 = np.sqrt(2.0)
    cones: list = []

    def _add_psd_upper(basis: tuple, idx_map: dict, var_off: int,
                       loc: bool = False, loc_var: int = 0) -> None:
        nonlocal row
        n = len(basis)
        cones.append(clarabel.PSDTriangleConeT(n))
        for j in range(n):
            for i in range(j + 1):
                ms = tuple(a + b for a, b in zip(basis[i], basis[j]))
                sc = 1.0 if i == j else sqrt2
                if loc:
                    sh = list(ms); sh[loc_var] += 2
                    _add(row, var_off + idx_map[ms], -sc)
                    _add(row, var_off + idx_map[tuple(sh)], sc)
                else:
                    _add(row, var_off + idx_map[ms], -sc)
                b_list.append(0.0)
                row += 1

    def _add_psd_upper_poly(basis: tuple, idx_map: dict, var_off: int,
                            g_poly: Polynomial) -> None:
        nonlocal row
        n = len(basis)
        cones.append(clarabel.PSDTriangleConeT(n))
        for j in range(n):
            for i in range(j + 1):
                ms = tuple(a + b for a, b in zip(basis[i], basis[j]))
                sc = 1.0 if i == j else sqrt2
                for gamma, g_coeff in g_poly.items():
                    shifted = tuple(m + g for m, g in zip(ms, gamma))
                    if shifted in idx_map:
                        _add(row, var_off + idx_map[shifted], -sc * g_coeff)
                b_list.append(0.0)
                row += 1

    occ_d = monomial_basis(_N_VARS, order)
    _add_psd_upper(occ_d, occ_idx, 0)

    term_d = monomial_basis(_N_T, order)
    _add_psd_upper(term_d, term_idx, n_occ)

    if order >= 2:
        loc_o = monomial_basis(_N_VARS, order - 1)
        for vi in range(_N_VARS):
            _add_psd_upper(loc_o, occ_idx, 0, loc=True, loc_var=vi)
        loc_t = monomial_basis(_N_T, order - 1)
        for vj in range(_N_T):
            _add_psd_upper(loc_t, term_idx, n_occ, loc=True, loc_var=vj)

    for g_poly in (occ_constraints or []):
        g_deg = poly_degree(g_poly)
        loc_d = order - (g_deg + 1) // 2
        if loc_d >= 0:
            _add_psd_upper_poly(
                monomial_basis(_N_VARS, loc_d), occ_idx, 0, g_poly
            )

    for g_poly in (term_constraints or []):
        g_deg = poly_degree(g_poly)
        loc_d = order - (g_deg + 1) // 2
        if loc_d >= 0:
            _add_psd_upper_poly(
                monomial_basis(_N_T, loc_d), term_idx, n_occ, g_poly
            )

    if initial_half_widths is not None:
        hw = initial_half_widths
        x0_lo = [((v - h) - o) / s for v, h, o, s in
                 zip([V0, y0_es, s0, c0], hw, offsets, scales)]
        x0_hi = [((v + h) - o) / s for v, h, o, s in
                 zip([V0, y0_es, s0, c0], hw, offsets, scales)]
        init_d = monomial_basis(_N_INIT, order)
        _add_psd_upper(init_d, init_idx, init_off)
        if order >= 2:
            loc_init = monomial_basis(_N_INIT, order - 1)
            for vi in range(_N_INIT):
                ai, bi = x0_lo[vi], x0_hi[vi]
                g_box: Polynomial = {
                    (0,) * _N_INIT: -ai * bi,
                    tuple(1 if j == vi else 0 for j in range(_N_INIT)): ai + bi,
                    tuple(2 if j == vi else 0 for j in range(_N_INIT)): -1.0,
                }
                _add_psd_upper_poly(loc_init, init_idx, init_off, g_box)

    # ---- assemble ---------------------------------------------------------
    A = sp.csc_matrix((vals, (rows, cols)), shape=(row, n_vars))
    b = np.array(b_list, dtype=np.float64)

    q_vec = np.zeros(n_vars, dtype=np.float64)
    for mono, coeff in range_rate_norm.items():
        if mono in occ_idx:
            q_vec[occ_idx[mono]] = -coeff

    obj_scale = np.max(np.abs(q_vec))
    if obj_scale > 0:
        q_vec /= obj_scale

    P = sp.csc_matrix((n_vars, n_vars), dtype=np.float64)
    cone_spec = (
        [clarabel.ZeroConeT(n_eq), clarabel.NonnegativeConeT(1)]
        + cones
    )

    settings = clarabel.DefaultSettings()
    settings.verbose = verbose
    settings.max_iter = 500
    settings.equilibrate_max_iter = 50
    settings.equilibrate_max_scaling = 1e8

    solver = clarabel.DefaultSolver(P, q_vec, A, b, cone_spec, settings)
    sol = solver.solve()

    status_map = {
        clarabel.SolverStatus.Solved: "optimal",
        clarabel.SolverStatus.AlmostSolved: "optimal_inaccurate",
        clarabel.SolverStatus.PrimalInfeasible: "infeasible",
        clarabel.SolverStatus.AlmostPrimalInfeasible: "infeasible_inaccurate",
        clarabel.SolverStatus.DualInfeasible: "unbounded",
        clarabel.SolverStatus.AlmostDualInfeasible: "unbounded_inaccurate",
    }
    status = status_map.get(sol.status, str(sol.status))
    obj_val = -sol.obj_val * obj_scale

    return obj_val, status


def _moment_idx_matrix(
    basis_d: tuple[tuple[int, ...], ...],
    idx_map: dict[tuple[int, ...], int],
) -> np.ndarray:
    """Build symmetric index matrix: out[i,j] = idx_map[basis[i] + basis[j]]."""
    n = len(basis_d)
    out = np.empty((n, n), dtype=np.intp)
    for i in range(n):
        for j in range(i, n):
            ms = tuple(a + b for a, b in zip(basis_d[i], basis_d[j]))
            out[i, j] = out[j, i] = idx_map[ms]
    return out


def _localising_idx_matrices(
    basis_loc: tuple[tuple[int, ...], ...],
    idx_map: dict[tuple[int, ...], int],
    var: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Index matrices for the localising matrix of ``1 − x̄_var²``."""
    n = len(basis_loc)
    base = np.empty((n, n), dtype=np.intp)
    shift = np.empty((n, n), dtype=np.intp)
    for i in range(n):
        for j in range(i, n):
            ms = tuple(a + b for a, b in zip(basis_loc[i], basis_loc[j]))
            sh = list(ms)
            sh[var] += 2
            base[i, j] = base[j, i] = idx_map[ms]
            shift[i, j] = shift[j, i] = idx_map[tuple(sh)]
    return base, shift


def _solve_cross_range_scs(
    params: CorridorParams,
    order: int,
    dynamics_norm: list[Polynomial],
    objective_norm: Polynomial,
    offsets: list[float],
    scales: list[float],
    entry_state: tuple[float, float, float, float, float, float] | None,
    verbose: bool,
    tau_multiplier: float = 10.0,
    occ_constraints: list[Polynomial] | None = None,
) -> tuple[float, str]:
    """SCS solver for the 8-variable cross-range system."""
    import scipy.sparse as sp
    import scs

    nv = _N_VARS_CR  # 8
    nt = 5  # terminal: (y, s, c, p, q)
    ns = 6  # state variables for Liouville: (V, y, s, c, p, q)

    dyn_degs = [poly_degree(f) for f in dynamics_norm]

    occ_basis_2d = monomial_basis(nv, 2 * order)
    n_occ = len(occ_basis_2d)
    occ_idx: dict[tuple[int, ...], int] = {
        m: i for i, m in enumerate(occ_basis_2d)
    }

    term_basis_2d = monomial_basis(nt, 2 * order)
    n_term = len(term_basis_2d)
    term_idx: dict[tuple[int, ...], int] = {
        m: i for i, m in enumerate(term_basis_2d)
    }
    n_vars = n_occ + n_term

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b_list: list[float] = []
    row = 0

    def _add(r: int, c: int, v: float) -> None:
        rows.append(r); cols.append(c); vals.append(v)

    s_max = params.s_max

    # Trig: s²_max s̄² + c̄² = 1 (occupation, variables 2,3)
    for mono in monomial_basis(nv, max(0, 2 * order - 2)):
        ms = list(mono); ms[_S] += 2
        mc = list(mono); mc[_C] += 2
        _add(row, occ_idx[tuple(ms)], s_max ** 2)
        _add(row, occ_idx[tuple(mc)], 1.0)
        _add(row, occ_idx[mono], -1.0)
        b_list.append(0.0); row += 1

    # Trig: p² + q² = 1 (occupation, variables 4,5)
    for mono in monomial_basis(nv, max(0, 2 * order - 2)):
        mp = list(mono); mp[_P] += 2
        mq = list(mono); mq[_Q] += 2
        _add(row, occ_idx[tuple(mp)], 1.0)
        _add(row, occ_idx[tuple(mq)], 1.0)
        _add(row, occ_idx[mono], -1.0)
        b_list.append(0.0); row += 1

    # Trig: s²_max s̄² + c̄² = 1 (terminal, variables 1,2)
    for mono in monomial_basis(nt, max(0, 2 * order - 2)):
        ms = list(mono); ms[1] += 2
        mc = list(mono); mc[2] += 2
        _add(row, n_occ + term_idx[tuple(ms)], s_max ** 2)
        _add(row, n_occ + term_idx[tuple(mc)], 1.0)
        _add(row, n_occ + term_idx[mono], -1.0)
        b_list.append(0.0); row += 1

    # Trig: p² + q² = 1 (terminal, variables 3,4)
    for mono in monomial_basis(nt, max(0, 2 * order - 2)):
        mp = list(mono); mp[3] += 2
        mq = list(mono); mq[4] += 2
        _add(row, n_occ + term_idx[tuple(mp)], 1.0)
        _add(row, n_occ + term_idx[tuple(mq)], 1.0)
        _add(row, n_occ + term_idx[mono], -1.0)
        b_list.append(0.0); row += 1

    # Entry state
    if entry_state is not None:
        V0, y0_es, s0, c0, p0, q0 = entry_state
    else:
        V0 = params.speed_high_ms
        y0_es = params.y_min
        s0, c0, p0, q0 = 0.0, 1.0, 0.0, 1.0
    x0b = [(val - offsets[i]) / scales[i]
           for i, val in enumerate([V0, y0_es, s0, c0, p0, q0])]

    # Liouville: iterate over 6-variable state monomials
    min_dyn = min(dyn_degs)
    broad_test = max(0, 2 * order - min_dyn + 1)
    for alpha_6 in monomial_basis(ns, broad_test):
        p_deg = sum(alpha_6)
        if any(alpha_6[i] > 0 and p_deg - 1 + dyn_degs[i] > 2 * order
               for i in range(ns)):
            continue
        x0a = 1.0
        for k in range(ns):
            if alpha_6[k] > 0:
                x0a *= x0b[k] ** alpha_6[k]
        # Terminal: z_T[α_y,α_s,α_c,α_p,α_q] * (-1)^{α_V}
        tk = alpha_6[1:]  # (α_y, α_s, α_c, α_p, α_q) — 5 vars
        if tk in term_idx:
            _add(row, n_occ + term_idx[tk], (-1.0) ** alpha_6[0])
        for i in range(ns):
            if alpha_6[i] == 0:
                continue
            ar = list(alpha_6); ar[i] -= 1
            for beta, fc in dynamics_norm[i].items():
                ok = tuple(
                    (ar[j] + beta[j] if j < ns else beta[j])
                    for j in range(nv)
                )
                if ok in occ_idx:
                    _add(row, occ_idx[ok], -alpha_6[i] * fc)
        b_list.append(x0a)
        row += 1

    n_eq = row

    # Non-negative: τ bound
    D0 = _RHO0 / (2.0 * params.ballistic_coefficient)
    tau_1d = (
        1.0 / (2.0 * params.speed_low_ms ** 2)
        - 1.0 / (2.0 * params.speed_high_ms ** 2)
    ) / (D0 * params.y_min ** 2)
    _add(row, occ_idx[(0,) * nv], 1.0)
    b_list.append(tau_multiplier * tau_1d)
    row += 1

    # PSD blocks
    sqrt2 = np.sqrt(2.0)
    psd_sizes: list[int] = []

    def _add_psd(basis, idx_map, var_off, loc=False, loc_var=0):
        nonlocal row
        n = len(basis)
        psd_sizes.append(n)
        for j in range(n):
            for i in range(j, n):
                ms = tuple(a + b for a, b in zip(basis[i], basis[j]))
                sc = 1.0 if i == j else sqrt2
                if loc:
                    sh = list(ms); sh[loc_var] += 2
                    _add(row, var_off + idx_map[ms], -sc)
                    _add(row, var_off + idx_map[tuple(sh)], sc)
                else:
                    _add(row, var_off + idx_map[ms], -sc)
                b_list.append(0.0); row += 1

    def _add_psd_poly(basis, idx_map, var_off, g_poly):
        nonlocal row
        n = len(basis)
        psd_sizes.append(n)
        for j in range(n):
            for i in range(j, n):
                ms = tuple(a + b for a, b in zip(basis[i], basis[j]))
                sc = 1.0 if i == j else sqrt2
                for gamma, g_coeff in g_poly.items():
                    shifted = tuple(m + g for m, g in zip(ms, gamma))
                    if shifted in idx_map:
                        _add(row, var_off + idx_map[shifted], -sc * g_coeff)
                b_list.append(0.0); row += 1

    occ_d = monomial_basis(nv, order)
    _add_psd(occ_d, occ_idx, 0)

    term_d = monomial_basis(nt, order)
    _add_psd(term_d, term_idx, n_occ)

    if order >= 2:
        loc_o = monomial_basis(nv, order - 1)
        for vi in range(nv):
            _add_psd(loc_o, occ_idx, 0, loc=True, loc_var=vi)
        loc_t = monomial_basis(nt, order - 1)
        for vj in range(nt):
            _add_psd(loc_t, term_idx, n_occ, loc=True, loc_var=vj)

    # Bank angle constraint: 1 - u₁² - u₂² ≥ 0 (degree 2)
    if order >= 2:
        g_bank: Polynomial = {
            (0,) * nv: 1.0,
            tuple(2 if j == _U1 else 0 for j in range(nv)): -1.0,
            tuple(2 if j == _U2 else 0 for j in range(nv)): -1.0,
        }
        _add_psd_poly(monomial_basis(nv, order - 1), occ_idx, 0, g_bank)

    for g_poly in (occ_constraints or []):
        g_deg = poly_degree(g_poly)
        loc_d = order - (g_deg + 1) // 2
        if loc_d >= 0:
            _add_psd_poly(monomial_basis(nv, loc_d), occ_idx, 0, g_poly)

    # Assemble
    A = sp.csc_matrix((vals, (rows, cols)), shape=(row, n_vars))
    b = np.array(b_list, dtype=np.float64)

    c_vec = np.zeros(n_vars, dtype=np.float64)
    for mono, coeff in objective_norm.items():
        if mono in occ_idx:
            c_vec[occ_idx[mono]] = -coeff

    cone = {"f": n_eq, "l": 1, "s": psd_sizes}
    solver = scs.SCS(
        {"A": A, "b": b, "c": c_vec}, cone,
        max_iters=50_000, verbose=verbose,
    )
    sol = solver.solve()

    status_map = {
        scs.SOLVED: "optimal",
        scs.SOLVED_INACCURATE: "optimal_inaccurate",
        scs.INFEASIBLE: "infeasible",
        scs.INFEASIBLE_INACCURATE: "infeasible_inaccurate",
        scs.UNBOUNDED: "unbounded",
        scs.UNBOUNDED_INACCURATE: "unbounded_inaccurate",
    }
    status = status_map.get(sol["info"]["status_val"], str(sol["info"]["status"]))
    return -sol["info"]["pobj"], status


def corridor_moment_bound(
    params: CorridorParams,
    *,
    order: int = 3,
    entry_state: tuple[float, ...] | None = None,
    solver: str | None = None,
    verbose: bool = False,
    backend: str = "auto",
    tau_multiplier: float = 10.0,
    initial_half_widths: tuple[float, float, float, float] | None = None,
    cross_range: bool = False,
) -> OccupationBound:
    r"""Primal (moment) relaxation for the downrange bound.

    Maximises :math:`\int V^2 \cos\gamma\, d\mu` over occupation measures
    whose moments satisfy the Liouville equation, PSD moment matrices,
    box localising matrices, and the trig identity.

    The dynamics have degree 6, so *order* >= 3 gives the first
    non-trivial Liouville constraints.

    Parameters
    ----------
    backend:
        ``"cvxpy"`` uses cvxpy with GPU/CPU solver cascade.
        ``"scs"`` calls SCS directly (lower memory, needed for order >= 5).
        ``"clarabel"`` calls Clarabel directly (tighter bounds than SCS).
        ``"auto"`` picks ``"scs"`` when order >= 5, else ``"cvxpy"``.
    tau_multiplier:
        Safety factor on the τ-time upper bound (default 10).
        The 1-D drag-only τ is multiplied by this factor.
    """
    if cross_range:
        offsets_cr, scales_cr = _normalise_bounds_cr(params)
        dynamics_cr, obj_cr = _normalised_dynamics_cr(params)
        occ_cons_cr = _occupation_constraint_polys_cr(params, offsets_cr, scales_cr)
        obj_val, status = _solve_cross_range_scs(
            params, order, dynamics_cr, obj_cr,
            offsets_cr, scales_cr, entry_state, verbose,
            tau_multiplier=tau_multiplier,
            occ_constraints=occ_cons_cr,
        )
        bound = float("inf") if obj_val is None else max(0.0, obj_val)
        return OccupationBound(
            max_downrange_m=bound,
            certificate_degree=order,
            corridor=params,
            sdp_status=status,
            W_coefficients={},
        )

    offsets, scales = _normalise_bounds(params)
    dynamics_norm, range_rate_norm = _normalised_dynamics(params)

    occ_cons = _occupation_constraint_polys(params, offsets, scales)
    term_cons = _terminal_constraint_polys(params, offsets, scales)

    use_direct = backend in ("scs", "clarabel") or (backend == "auto" and order >= 5)
    if use_direct:
        if backend == "clarabel":
            solver_fn = _solve_moment_clarabel
        else:
            solver_fn = _solve_moment_scs
        obj_val, status = solver_fn(
            params, order, dynamics_norm, range_rate_norm,
            offsets, scales, entry_state, verbose,
            tau_multiplier=tau_multiplier,
            occ_constraints=occ_cons,
            term_constraints=term_cons,
            initial_half_widths=initial_half_widths,
        )
        bound = float("inf") if obj_val is None else max(0.0, obj_val)
        return OccupationBound(
            max_downrange_m=bound,
            certificate_degree=order,
            corridor=params,
            sdp_status=status,
            W_coefficients={},
        )

    dyn_degs = [poly_degree(f) for f in dynamics_norm]

    # --- Moment variables ---------------------------------------------------
    occ_basis_2d = monomial_basis(_N_VARS, 2 * order)
    n_occ = len(occ_basis_2d)
    y = cp.Variable(n_occ, name="y")
    occ_idx: dict[tuple[int, ...], int] = {m: i for i, m in enumerate(occ_basis_2d)}

    _N_T = 3  # terminal variables: (ȳ, s̄, c̄)
    term_basis_2d = monomial_basis(_N_T, 2 * order)
    n_term = len(term_basis_2d)
    t = cp.Variable(n_term, name="t")
    term_idx: dict[tuple[int, ...], int] = {m: i for i, m in enumerate(term_basis_2d)}

    _N_INIT = 4
    if initial_half_widths is not None:
        init_basis_2d = monomial_basis(_N_INIT, 2 * order)
        n_init = len(init_basis_2d)
        z0 = cp.Variable(n_init, name="z0")
        init_idx: dict[tuple[int, ...], int] = {
            m: i for i, m in enumerate(init_basis_2d)
        }
    else:
        z0 = None
        init_idx = {}

    def z_val(a4: tuple[int, ...]) -> Any:
        """Terminal moment with v̄ = −1 support: z(α) = (−1)^{α_v} t(α_y,α_s,α_c)."""
        tk = a4[1:]
        if tk not in term_idx:
            return 0.0
        return ((-1.0) ** a4[0]) * t[term_idx[tk]]

    constraints: list[Any] = []

    # --- Moment matrix PSD: occupation (vectorised) -------------------------
    occ_basis_d = monomial_basis(_N_VARS, order)
    mo_idx = _moment_idx_matrix(occ_basis_d, occ_idx)
    constraints.append(y[mo_idx] >> 0)

    # --- Moment matrix PSD: terminal (vectorised) ---------------------------
    term_basis_d = monomial_basis(_N_T, order)
    mt_idx = _moment_idx_matrix(term_basis_d, term_idx)
    constraints.append(t[mt_idx] >> 0)

    # --- Box localising: 1 − x̄_i² ≥ 0 (vectorised) ------------------------
    loc_ord = order - 1
    if loc_ord >= 0:
        loc_basis_o = monomial_basis(_N_VARS, loc_ord)
        for vi in range(_N_VARS):
            b_idx, s_idx = _localising_idx_matrices(loc_basis_o, occ_idx, vi)
            constraints.append((y[b_idx] - y[s_idx]) >> 0)

        loc_basis_t = monomial_basis(_N_T, loc_ord)
        for vj in range(_N_T):
            b_idx, s_idx = _localising_idx_matrices(loc_basis_t, term_idx, vj)
            constraints.append((t[b_idx] - t[s_idx]) >> 0)

    # --- Moment matrix PSD + localising: initial measure --------------------
    if z0 is not None:
        init_basis_d = monomial_basis(_N_INIT, order)
        mi_idx = _moment_idx_matrix(init_basis_d, init_idx)
        constraints.append(z0[mi_idx] >> 0)

        constraints.append(z0[init_idx[(0, 0, 0, 0)]] == 1)

        if loc_ord >= 0:
            if entry_state is not None:
                V0_i, y0_i, s0_i, c0_i = entry_state
            else:
                V0_i = params.speed_high_ms
                y0_i = params.y_min
                s0_i, c0_i = 0.0, 1.0
            hw = initial_half_widths
            x0_lo = [((v - h) - o) / s for v, h, o, s in
                     zip([V0_i, y0_i, s0_i, c0_i], hw, offsets, scales)]
            x0_hi = [((v + h) - o) / s for v, h, o, s in
                     zip([V0_i, y0_i, s0_i, c0_i], hw, offsets, scales)]
            loc_basis_init = monomial_basis(_N_INIT, loc_ord)
            n_li = len(loc_basis_init)
            for vi in range(_N_INIT):
                ai, bi = x0_lo[vi], x0_hi[vi]
                g_box: Polynomial = {
                    (0,) * _N_INIT: -ai * bi,
                    tuple(1 if j == vi else 0 for j in range(_N_INIT)): ai + bi,
                    tuple(2 if j == vi else 0 for j in range(_N_INIT)): -1.0,
                }
                L_mat_i: Any = 0
                for gamma, g_coeff in g_box.items():
                    idx_shifted = np.empty((n_li, n_li), dtype=np.intp)
                    for ii in range(n_li):
                        for jj in range(ii, n_li):
                            ms = tuple(
                                a + b + c
                                for a, b, c in zip(
                                    loc_basis_init[ii], loc_basis_init[jj], gamma
                                )
                            )
                            idx_shifted[ii, jj] = idx_shifted[jj, ii] = init_idx[ms]
                    L_mat_i = L_mat_i + g_coeff * z0[idx_shifted]
                constraints.append(L_mat_i >> 0)

    # --- Trig identity: s_max² s̄² + c̄² = 1 (vectorised) --------------------
    s_max = params.s_max
    trig_occ = monomial_basis(_N_VARS, max(0, 2 * order - 2))
    n_trig_o = len(trig_occ)
    arr_base_o = np.empty(n_trig_o, dtype=np.intp)
    arr_s_o = np.empty(n_trig_o, dtype=np.intp)
    arr_c_o = np.empty(n_trig_o, dtype=np.intp)
    for k, mono in enumerate(trig_occ):
        arr_base_o[k] = occ_idx[mono]
        ms = list(mono); ms[_S] += 2
        arr_s_o[k] = occ_idx[tuple(ms)]
        mc = list(mono); mc[_C] += 2
        arr_c_o[k] = occ_idx[tuple(mc)]
    constraints.append(
        s_max ** 2 * y[arr_s_o] + y[arr_c_o] - y[arr_base_o] == 0
    )

    trig_term = monomial_basis(_N_T, max(0, 2 * order - 2))
    n_trig_t = len(trig_term)
    arr_base_t = np.empty(n_trig_t, dtype=np.intp)
    arr_s_t = np.empty(n_trig_t, dtype=np.intp)
    arr_c_t = np.empty(n_trig_t, dtype=np.intp)
    for k, mono in enumerate(trig_term):
        arr_base_t[k] = term_idx[mono]
        ms = list(mono); ms[1] += 2
        arr_s_t[k] = term_idx[tuple(ms)]
        mc = list(mono); mc[2] += 2
        arr_c_t[k] = term_idx[tuple(mc)]
    constraints.append(
        s_max ** 2 * t[arr_s_t] + t[arr_c_t] - t[arr_base_t] == 0
    )

    if z0 is not None:
        trig_init = monomial_basis(_N_INIT, max(0, 2 * order - 2))
        n_trig_i = len(trig_init)
        arr_base_i = np.empty(n_trig_i, dtype=np.intp)
        arr_s_i = np.empty(n_trig_i, dtype=np.intp)
        arr_c_i = np.empty(n_trig_i, dtype=np.intp)
        for k, mono in enumerate(trig_init):
            arr_base_i[k] = init_idx[mono]
            ms = list(mono); ms[2] += 2
            arr_s_i[k] = init_idx[tuple(ms)]
            mc = list(mono); mc[3] += 2
            arr_c_i[k] = init_idx[tuple(mc)]
        constraints.append(
            s_max ** 2 * z0[arr_s_i] + z0[arr_c_i] - z0[arr_base_i] == 0
        )

    # --- Liouville constraints ----------------------------------------------
    if entry_state is not None:
        V0, y0, s0, c0 = entry_state
    else:
        V0 = params.speed_high_ms
        y0 = params.y_min
        s0 = 0.0
        c0 = 1.0

    x0b = [
        (V0 - offsets[0]) / scales[0],
        (y0 - offsets[1]) / scales[1],
        (s0 - offsets[2]) / scales[2],
        (c0 - offsets[3]) / scales[3],
    ]

    min_dyn = min(dyn_degs)
    broad_test = max(0, 2 * order - min_dyn + 1)
    for alpha_4 in monomial_basis(4, broad_test):
        p = sum(alpha_4)
        if any(
            alpha_4[i] > 0 and p - 1 + dyn_degs[i] > 2 * order
            for i in range(4)
        ):
            continue

        if z0 is not None:
            lhs = z_val(alpha_4) - z0[init_idx[alpha_4]]
        else:
            x0a = 1.0
            for k in range(4):
                if alpha_4[k] > 0:
                    x0a *= x0b[k] ** alpha_4[k]
            lhs = z_val(alpha_4) - x0a

        rhs: Any = 0.0
        for i in range(4):
            if alpha_4[i] == 0:
                continue
            ar = list(alpha_4)
            ar[i] -= 1
            for beta, fc in dynamics_norm[i].items():
                ok = [ar[0] + beta[0], ar[1] + beta[1],
                      ar[2] + beta[2], ar[3] + beta[3], beta[4]]
                ok_t = tuple(ok)
                if ok_t in occ_idx:
                    rhs = rhs + alpha_4[i] * fc * y[occ_idx[ok_t]]

        constraints.append(lhs == rhs)

    # --- Compact-domain bound on total τ-time --------------------------------
    D0 = _RHO0 / (2.0 * params.ballistic_coefficient)
    tau_1d = (
        1.0 / (2.0 * params.speed_low_ms ** 2)
        - 1.0 / (2.0 * params.speed_high_ms ** 2)
    ) / (D0 * params.y_min ** 2)
    constraints.append(y[occ_idx[(0,) * _N_VARS]] <= tau_multiplier * tau_1d)

    # --- Occupation-measure support constraints (q, q̇, …) -------------------
    for g_poly in occ_cons:
        g_deg = poly_degree(g_poly)
        loc_d = order - (g_deg + 1) // 2
        if loc_d < 0:
            continue
        loc_basis = monomial_basis(_N_VARS, loc_d)
        n_loc = len(loc_basis)
        L_mat: Any = 0
        for gamma, g_coeff in g_poly.items():
            idx_shifted = np.empty((n_loc, n_loc), dtype=np.intp)
            for ii in range(n_loc):
                for jj in range(ii, n_loc):
                    ms = tuple(
                        a + b + c
                        for a, b, c in zip(loc_basis[ii], loc_basis[jj], gamma)
                    )
                    idx_shifted[ii, jj] = idx_shifted[jj, ii] = occ_idx[ms]
            L_mat = L_mat + g_coeff * y[idx_shifted]
        constraints.append(L_mat >> 0)

    # --- Terminal-measure support constraints ---------------------------------
    for g_poly in term_cons:
        g_deg = poly_degree(g_poly)
        loc_d = order - (g_deg + 1) // 2
        if loc_d < 0:
            continue
        loc_basis = monomial_basis(_N_T, loc_d)
        n_loc = len(loc_basis)
        L_mat_t: Any = 0
        for gamma, g_coeff in g_poly.items():
            idx_shifted = np.empty((n_loc, n_loc), dtype=np.intp)
            for ii in range(n_loc):
                for jj in range(ii, n_loc):
                    ms = tuple(
                        a + b + c
                        for a, b, c in zip(loc_basis[ii], loc_basis[jj], gamma)
                    )
                    idx_shifted[ii, jj] = idx_shifted[jj, ii] = term_idx[ms]
            L_mat_t = L_mat_t + g_coeff * t[idx_shifted]
        constraints.append(L_mat_t >> 0)

    # --- Objective: max ∫ V² cos γ dμ ---------------------------------------
    obj: Any = 0.0
    for mono, coeff in range_rate_norm.items():
        if mono in occ_idx:
            obj = obj + coeff * y[occ_idx[mono]]

    problem = cp.Problem(cp.Maximize(obj), constraints)

    # Solver selection: prefer GPU CLARABEL, then CPU CLARABEL, then SCS.
    if solver is not None:
        solvers = [solver]
    else:
        solvers = ["CUCLARABEL", "CLARABEL", "SCS"]

    solved = False
    for s in solvers:
        if s not in cp.installed_solvers():
            continue
        try:
            kw: dict[str, Any] = {"solver": s}
            if verbose:
                kw["verbose"] = True
            if s == "SCS":
                kw["max_iters"] = 50_000
            problem.solve(**kw)
            solved = True
            break
        except cp.error.SolverError:
            continue

    if not solved:
        problem.solve(solver="SCS", verbose=verbose, max_iters=50_000)

    status = problem.status
    bound = float("inf") if problem.value is None else max(0.0, float(problem.value))

    return OccupationBound(
        max_downrange_m=bound,
        certificate_degree=order,
        corridor=params,
        sdp_status=status,
        W_coefficients={},
    )
