"""Core primitives for the final-iteration stochastic optimal-control formulation.

This module implements the corrections identified as decisive in
``docs/stochastic-optimal-control-formulation.md`` (distilled from the original
design dialogue).  The formulation's own summary names them:

    the staging covariance jump, explicit launch epoch, one-sided
    chance-constraint treatment, mission-level risk allocation, relative
    covariance, atmospheric-relative velocity, and distinction between
    nominal and feedback control.

Each is provided here as an independently testable primitive rather than being
buried inside a solver, so that the transcription in
:mod:`aether_gambit.guidance.stochastic_optimization` can be built from pieces whose
correctness is checked in isolation.

The initial formulation (the plan's Q1/A1 sections) is retained only as
context; where the two disagree this module follows A2.

Sections
--------
Risk
    :func:`risk_multiplier`, :func:`allocate_risk`, :class:`RiskBudget`,
    :func:`chance_constraint_margin`.
Covariance propagation
    :func:`van_loan_discretization`, :func:`propagate_covariance`.
Hybrid events
    :class:`StagingEvent`, :func:`staging_jump`.
Timing
    :class:`LaunchWindow`, :func:`phase_epochs`.
Collision and terminal accuracy
    :func:`relative_covariance`, :func:`collision_probability`,
    :func:`terminal_containment`, :func:`generalized_chi2_cdf`.
Environment
    :func:`atmosphere_relative_velocity`, :func:`heat_flux_gradient`.
Nonlinear validation
    :func:`sigma_points`, :func:`unscented_propagate`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.linalg
from scipy.stats import norm

__all__ = [
    "EARTH_ROTATION_RATE",
    "LaunchWindow",
    "RiskBudget",
    "StagingEvent",
    "allocate_risk",
    "atmosphere_relative_velocity",
    "chance_constraint_margin",
    "collision_probability",
    "generalized_chi2_cdf",
    "heat_flux_gradient",
    "phase_epochs",
    "propagate_covariance",
    "relative_covariance",
    "risk_multiplier",
    "sigma_points",
    "staging_jump",
    "terminal_containment",
    "unscented_propagate",
    "van_loan_discretization",
]

_FloatArray = npt.NDArray[np.float64]

#: Earth's sidereal rotation rate (rad/s).
EARTH_ROTATION_RATE = 7.2921150e-5

RiskModel = Literal["gaussian", "cantelli"]


# ---------------------------------------------------------------------------
# Risk: one-sided chance constraints and mission-level allocation
# ---------------------------------------------------------------------------

def risk_multiplier(epsilon: float, *, model: RiskModel = "gaussian") -> float:
    r"""Return the constraint-tightening multiplier :math:`\kappa` for risk ``epsilon``.

    A constraint :math:`P[y > y_{\max}] \le \epsilon` on a scalar :math:`y`
    with mean :math:`\mu_y` and standard deviation :math:`\sigma_y` is enforced
    by the deterministic surrogate

    .. math:: \mu_y + \kappa_\epsilon \sigma_y \le y_{\max}.

    Two models are available.

    ``"gaussian"``
        :math:`\kappa = \Phi^{-1}(1-\epsilon)`.  Exact when :math:`y` is
        Gaussian.

    ``"cantelli"``
        :math:`\kappa = \sqrt{(1-\epsilon)/\epsilon}`, the tight
        distributionally robust bound over the ambiguity set
        :math:`\mathcal{P} = \{P : E_P[y]=\mu_y,\ \mathrm{Var}_P[y]=\sigma_y^2\}`
        (the plan's §12).  It makes no distributional assumption beyond the
        first two moments and is therefore markedly more conservative.

    Notes
    -----
    The initial formulation identified "3σ" with 99.7 % confidence.  That is
    the *two-sided* Gaussian figure, :math:`2\Phi(3)-1 = 0.99730`.  Heating and
    obstacle constraints are one-sided, for which :math:`\Phi(3) = 0.99865`.
    Stating the risk :math:`\epsilon` explicitly and deriving :math:`\kappa`
    from it removes the ambiguity: a genuine one-sided 99.7 % constraint needs
    :math:`\kappa = 2.748`, not 3.

    Parameters
    ----------
    epsilon:
        Allowed probability of violation, in ``(0, 1)``.
    model:
        ``"gaussian"`` or ``"cantelli"``.

    Returns
    -------
    float
        The multiplier :math:`\kappa_\epsilon`.

    Raises
    ------
    ValueError
        If ``epsilon`` is outside ``(0, 1)`` or ``model`` is unknown.

    Examples
    --------
    >>> round(risk_multiplier(0.00135), 4)
    3.0
    >>> round(risk_multiplier(0.003), 4)
    2.7478
    >>> round(risk_multiplier(0.00135, model="cantelli"), 2)
    27.19
    """
    if not 0.0 < epsilon < 1.0:
        raise ValueError(f"epsilon must lie in (0, 1), got {epsilon}")
    if model == "gaussian":
        return float(norm.ppf(1.0 - epsilon))
    if model == "cantelli":
        return float(np.sqrt((1.0 - epsilon) / epsilon))
    raise ValueError(f"unknown risk model {model!r}; expected 'gaussian' or 'cantelli'")


def allocate_risk(
    n_constraints: int,
    mission_epsilon: float,
    *,
    weights: _FloatArray | None = None,
) -> _FloatArray:
    r"""Split a mission risk budget across constraints by the union bound.

    Pointwise constraints :math:`P[C_i > 0] \le \epsilon_i` do **not** imply
    :math:`P[\exists i: C_i > 0] \le \epsilon`.  Boole's inequality gives

    .. math:: P\Big[\bigcup_i \{C_i > 0\}\Big] \le \sum_i \epsilon_i,

    so a mission-level guarantee requires :math:`\sum_i \epsilon_i \le
    \epsilon_{\mathrm{mission}}` (the plan's §14).  With a few hundred
    collocation nodes this is the difference between a trajectory that is
    "3σ safe at every node" and one that is actually safe.

    Parameters
    ----------
    n_constraints:
        Number of individually enforced constraints (nodes × constraint types).
    mission_epsilon:
        Total allowed probability of *any* violation, in ``(0, 1)``.
    weights:
        Optional non-negative relative shares, shape ``(n_constraints,)``.
        Normalised internally.  Uniform when omitted.

    Returns
    -------
    numpy.ndarray
        Per-constraint risks summing to ``mission_epsilon``.

    Raises
    ------
    ValueError
        If ``n_constraints`` is not positive, ``mission_epsilon`` is outside
        ``(0, 1)``, or ``weights`` has the wrong shape or is not positive.

    Examples
    --------
    >>> allocate_risk(4, 0.004)
    array([0.001, 0.001, 0.001, 0.001])
    """
    if n_constraints <= 0:
        raise ValueError(f"n_constraints must be positive, got {n_constraints}")
    if not 0.0 < mission_epsilon < 1.0:
        raise ValueError(f"mission_epsilon must lie in (0, 1), got {mission_epsilon}")
    if weights is None:
        share = np.full(n_constraints, 1.0 / n_constraints, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (n_constraints,):
            raise ValueError(f"weights must have shape ({n_constraints},), got {w.shape}")
        if not np.all(w > 0.0):
            raise ValueError("weights must be strictly positive")
        share = w / float(w.sum())
    return np.asarray(share * mission_epsilon, dtype=np.float64)


@dataclass(frozen=True)
class RiskBudget:
    """A mission risk budget with an explicit per-constraint allocation.

    Attributes
    ----------
    mission_epsilon:
        Total allowed probability of any constraint violation.
    allocations:
        Mapping from constraint-family name to that family's share of the
        budget.  Shares must be positive; they are normalised on construction
        so that the realised per-family risks sum to ``mission_epsilon``.
    model:
        Risk model used to convert a risk into a tightening multiplier.
    """

    mission_epsilon: float
    allocations: dict[str, float] = field(default_factory=dict)
    model: RiskModel = "gaussian"

    def __post_init__(self) -> None:
        if not 0.0 < self.mission_epsilon < 1.0:
            raise ValueError(
                f"mission_epsilon must lie in (0, 1), got {self.mission_epsilon}"
            )
        if self.allocations and not all(v > 0.0 for v in self.allocations.values()):
            raise ValueError("allocation shares must be strictly positive")

    def epsilon_for(self, family: str, n_constraints: int = 1) -> float:
        """Return the per-constraint risk for one family.

        Parameters
        ----------
        family:
            Key into :attr:`allocations`.
        n_constraints:
            Number of constraints in the family (for example the node count),
            across which the family's share is split uniformly.

        Returns
        -------
        float
            Per-constraint risk.

        Raises
        ------
        KeyError
            If ``family`` is not allocated.
        ValueError
            If ``n_constraints`` is not positive.
        """
        if n_constraints <= 0:
            raise ValueError(f"n_constraints must be positive, got {n_constraints}")
        if family not in self.allocations:
            raise KeyError(f"no allocation for constraint family {family!r}")
        total = sum(self.allocations.values())
        share = self.allocations[family] / total
        return float(self.mission_epsilon * share / n_constraints)

    def multiplier_for(self, family: str, n_constraints: int = 1) -> float:
        """Return :math:`\\kappa` for one constraint of ``family``.

        Parameters
        ----------
        family:
            Key into :attr:`allocations`.
        n_constraints:
            Number of constraints in the family.

        Returns
        -------
        float
            Tightening multiplier under :attr:`model`.
        """
        return risk_multiplier(self.epsilon_for(family, n_constraints), model=self.model)


def chance_constraint_margin(
    mean: float,
    variance: float,
    limit: float,
    epsilon: float,
    *,
    model: RiskModel = "gaussian",
) -> float:
    r"""Return the margin of a one-sided chance constraint.

    The constraint is :math:`P[y > y_{\max}] \le \epsilon`; the returned margin
    is :math:`y_{\max} - (\mu_y + \kappa_\epsilon \sigma_y)`, so it is
    non-negative exactly when the constraint holds.

    Parameters
    ----------
    mean:
        Mean :math:`\mu_y` of the constrained quantity.
    variance:
        Variance :math:`\sigma_y^2`.  Negative values (which can arise from a
        linearised quadratic form on a numerically indefinite covariance) are
        clipped to zero.
    limit:
        Upper limit :math:`y_{\max}`.
    epsilon:
        Allowed violation probability.
    model:
        Risk model, see :func:`risk_multiplier`.

    Returns
    -------
    float
        Constraint margin; ``>= 0`` means satisfied.
    """
    sigma = float(np.sqrt(max(float(variance), 0.0)))
    return float(limit - (mean + risk_multiplier(epsilon, model=model) * sigma))


# ---------------------------------------------------------------------------
# Covariance propagation
# ---------------------------------------------------------------------------

def van_loan_discretization(
    a_matrix: _FloatArray,
    q_matrix: _FloatArray,
    dt: float,
) -> tuple[_FloatArray, _FloatArray]:
    r"""Exactly discretise a linear covariance flow over one step.

    For :math:`\dot{\delta x} = A\,\delta x + w`, :math:`w \sim (0, Q)`, the
    exact one-step transition and process-noise increment are

    .. math::
        \Phi = e^{A\,\Delta t}, \qquad
        Q_d = \int_0^{\Delta t} e^{A u}\, Q\, e^{A^\top u}\,\mathrm{d}u,

    so that :math:`P_{k+1} = \Phi P_k \Phi^\top + Q_d`.  Van Loan's method
    obtains both from a single matrix exponential.  With the block matrix

    .. math::
        M = \begin{bmatrix} A & Q \\ 0 & -A^\top \end{bmatrix}\Delta t,
        \qquad
        e^{M} = \begin{bmatrix} X_{11} & X_{12} \\ 0 & X_{22}\end{bmatrix},

    the upper-right block satisfies :math:`X_{12} = Q_d\, e^{-A^\top \Delta t}`,
    whence :math:`\Phi = X_{11}` and :math:`Q_d = X_{12}\,\Phi^\top`.

    Notes
    -----
    Forming :math:`Q_d` as :math:`X_{12} X_{22}` instead — the natural-looking
    but incorrect pairing — yields :math:`Q_d e^{-2A^\top \Delta t}`.  That
    matrix is neither symmetric nor positive semi-definite, so it injects
    negative variance along some direction at every step.

    Parameters
    ----------
    a_matrix:
        Dynamics Jacobian :math:`A`, shape ``(n, n)``.
    q_matrix:
        Continuous process-noise spectral density :math:`Q`, shape ``(n, n)``.
        Must be symmetric positive semi-definite for :math:`Q_d` to be a valid
        covariance increment.
    dt:
        Step length.

    Returns
    -------
    phi : numpy.ndarray
        State-transition matrix, shape ``(n, n)``.
    q_d : numpy.ndarray
        Discrete process-noise covariance, shape ``(n, n)``, symmetrised.

    Raises
    ------
    ValueError
        If the matrices are not square and conformable.
    """
    a = np.asarray(a_matrix, dtype=np.float64)
    q = np.asarray(q_matrix, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"a_matrix must be square, got shape {a.shape}")
    if q.shape != a.shape:
        raise ValueError(f"q_matrix must have shape {a.shape}, got {q.shape}")

    n = a.shape[0]
    block = np.block([[a, q], [np.zeros((n, n)), -a.T]]) * float(dt)
    expm_block = scipy.linalg.expm(block)
    phi = np.asarray(expm_block[:n, :n], dtype=np.float64)
    q_d = np.asarray(expm_block[:n, n:] @ phi.T, dtype=np.float64)
    q_d = 0.5 * (q_d + q_d.T)
    return phi, q_d


def propagate_covariance(
    covariance: _FloatArray,
    a_matrix: _FloatArray,
    q_matrix: _FloatArray,
    dt: float,
    *,
    b_matrix: _FloatArray | None = None,
    gain: _FloatArray | None = None,
) -> _FloatArray:
    r"""Advance a covariance one step, optionally under a feedback policy.

    Open loop, this is :math:`P^+ = \Phi P \Phi^\top + Q_d`.  When a control
    matrix :math:`B` and gain :math:`K` are supplied, the policy
    :math:`u = u^\star + K\,\delta x` closes the loop and the propagation uses

    .. math:: A_{\mathrm{cl}} = A + BK

    in place of :math:`A` (the plan's §20).  This is what turns the problem
    from "find a nominal trajectory through uncertainty" into "find a
    trajectory that is robust to it": with an open-loop :math:`A`, dispersion
    grows unchecked and the optimizer can only avoid chance-constraint
    violation by flying a more conservative nominal path.

    Parameters
    ----------
    covariance:
        Current covariance :math:`P`, shape ``(n, n)``.
    a_matrix:
        Open-loop dynamics Jacobian, shape ``(n, n)``.
    q_matrix:
        Continuous process-noise spectral density, shape ``(n, n)``.
    dt:
        Step length.
    b_matrix:
        Control Jacobian, shape ``(n, m)``.  Required with ``gain``.
    gain:
        Feedback gain :math:`K`, shape ``(m, n)``.  Required with ``b_matrix``.

    Returns
    -------
    numpy.ndarray
        Propagated covariance, symmetrised, shape ``(n, n)``.

    Raises
    ------
    ValueError
        If exactly one of ``b_matrix`` and ``gain`` is supplied, or shapes are
        not conformable.
    """
    p = np.asarray(covariance, dtype=np.float64)
    a = np.asarray(a_matrix, dtype=np.float64)
    if (b_matrix is None) != (gain is None):
        raise ValueError("b_matrix and gain must be supplied together")
    if b_matrix is not None and gain is not None:
        b = np.asarray(b_matrix, dtype=np.float64)
        k = np.asarray(gain, dtype=np.float64)
        if b.shape[0] != a.shape[0] or k.shape[1] != a.shape[1] or b.shape[1] != k.shape[0]:
            raise ValueError(
                f"non-conformable feedback shapes: A{a.shape}, B{b.shape}, K{k.shape}"
            )
        a = a + b @ k
    phi, q_d = van_loan_discretization(a, q_matrix, dt)
    out = phi @ p @ phi.T + q_d
    return np.asarray(0.5 * (out + out.T), dtype=np.float64)


# ---------------------------------------------------------------------------
# Hybrid events: staging
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StagingEvent:
    r"""A staging discontinuity with its own injected uncertainty.

    The event map on the mean state :math:`\mu = [r, v, m]` is

    .. math::
        r^+ = r^-,\qquad
        v^+ = v^- + \frac{I_{\mathrm{sep}}}{m^-}\hat{n},\qquad
        m^+ = m^- - m_{\mathrm{drop}},

    where :math:`I_{\mathrm{sep}}` is the separation impulse and
    :math:`\hat{n}` its direction.

    Attributes
    ----------
    dropped_mass:
        Structural mass discarded, :math:`m_{\mathrm{drop}}` (kg).
    dropped_mass_std:
        Standard deviation of the discarded mass (kg) — residual propellant
        and structural-mass knowledge error.
    separation_impulse:
        Magnitude of the separation impulse :math:`I_{\mathrm{sep}}` (N·s).
    separation_direction:
        Unit direction :math:`\hat{n}` of the separation impulse, shape
        ``(3,)``.  Normalised on construction.
    separation_impulse_std:
        Standard deviation of the separation impulse magnitude (N·s).
    transverse_impulse_std:
        Standard deviation of the impulse transverse to
        :attr:`separation_direction` (N·s), representing a tip-off kick.
    """

    dropped_mass: float
    dropped_mass_std: float = 0.0
    separation_impulse: float = 0.0
    separation_direction: _FloatArray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0])
    )
    separation_impulse_std: float = 0.0
    transverse_impulse_std: float = 0.0

    def __post_init__(self) -> None:
        if self.dropped_mass < 0.0:
            raise ValueError(f"dropped_mass must be non-negative, got {self.dropped_mass}")
        for name in (
            "dropped_mass_std",
            "separation_impulse_std",
            "transverse_impulse_std",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        direction = np.asarray(self.separation_direction, dtype=np.float64)
        if direction.shape != (3,):
            raise ValueError(
                f"separation_direction must have shape (3,), got {direction.shape}"
            )
        norm_d = float(np.linalg.norm(direction))
        if norm_d == 0.0:
            raise ValueError("separation_direction must be non-zero")
        object.__setattr__(self, "separation_direction", direction / norm_d)


def staging_jump(
    mean: _FloatArray,
    covariance: _FloatArray,
    event: StagingEvent,
) -> tuple[_FloatArray, _FloatArray]:
    r"""Apply a staging event map to a mean and covariance.

    The initial formulation held the covariance continuous across staging,
    :math:`\Sigma^+ = \Sigma^-`.  A discontinuity that changes mass and
    imparts a separation impulse cannot leave the uncertainty unchanged; the
    correct transformation is the linearised event map (the plan's §10)

    .. math:: P^+ = J_\Phi P^- J_\Phi^\top + Q_\Phi,

    with :math:`J_\Phi = \partial\Phi/\partial x` and :math:`Q_\Phi` the
    uncertainty injected by the event itself.

    The mass-to-velocity block of :math:`J_\Phi` is the reason this matters:
    the separation impulse produces :math:`\Delta v = I_{\mathrm{sep}}/m^-`,
    so an error in the pre-separation mass feeds straight into velocity,

    .. math:: \frac{\partial v^+}{\partial m^-}
              = -\frac{I_{\mathrm{sep}}}{(m^-)^2}\hat{n},

    correlating mass and velocity uncertainty across the event.  Holding
    :math:`\Sigma` continuous discards that coupling entirely.

    Parameters
    ----------
    mean:
        Pre-event mean state :math:`[r, v, m]`, shape ``(7,)``.
    covariance:
        Pre-event covariance, shape ``(7, 7)``.
    event:
        The staging event.

    Returns
    -------
    mean_plus : numpy.ndarray
        Post-event mean, shape ``(7,)``.
    covariance_plus : numpy.ndarray
        Post-event covariance, shape ``(7, 7)``, symmetrised.

    Raises
    ------
    ValueError
        If shapes are wrong or the post-event mass would be non-positive.
    """
    mu = np.asarray(mean, dtype=np.float64)
    p = np.asarray(covariance, dtype=np.float64)
    if mu.shape != (7,):
        raise ValueError(f"mean must have shape (7,), got {mu.shape}")
    if p.shape != (7, 7):
        raise ValueError(f"covariance must have shape (7, 7), got {p.shape}")

    m_minus = float(mu[6])
    m_plus = m_minus - event.dropped_mass
    if m_plus <= 0.0:
        raise ValueError(
            f"post-staging mass must be positive: {m_minus} - {event.dropped_mass}"
            f" = {m_plus}"
        )

    n_hat = np.asarray(event.separation_direction, dtype=np.float64)
    delta_v = event.separation_impulse / m_minus * n_hat

    mean_plus = mu.copy()
    mean_plus[3:6] = mu[3:6] + delta_v
    mean_plus[6] = m_plus

    # Linearised event map. Position passes through; velocity picks up the
    # mass sensitivity of the separation impulse; mass passes through with
    # unit sensitivity (the drop is deterministic in the mean, its dispersion
    # enters via Q_phi).
    j_phi = np.eye(7)
    j_phi[3:6, 6] = -event.separation_impulse / (m_minus**2) * n_hat

    # Uncertainty injected by the event itself.
    q_phi = np.zeros((7, 7))
    q_phi[6, 6] = event.dropped_mass_std**2
    if event.separation_impulse_std > 0.0:
        axial_var = (event.separation_impulse_std / m_minus) ** 2
        q_phi[3:6, 3:6] += axial_var * np.outer(n_hat, n_hat)
    if event.transverse_impulse_std > 0.0:
        transverse_var = (event.transverse_impulse_std / m_minus) ** 2
        q_phi[3:6, 3:6] += transverse_var * (np.eye(3) - np.outer(n_hat, n_hat))

    p_plus = j_phi @ p @ j_phi.T + q_phi
    p_plus = 0.5 * (p_plus + p_plus.T)
    return mean_plus, np.asarray(p_plus, dtype=np.float64)


# ---------------------------------------------------------------------------
# Timing: explicit launch epoch
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LaunchWindow:
    """Bounds on the absolute launch epoch.

    The initial formulation encoded the launch time as the duration
    :math:`s_0` of a pad-loiter phase with zero dynamics.  That phase is
    degenerate: with a purely time-minimal objective and no explicitly
    time-dependent constraint, :math:`s_0` collapses to its lower bound, and
    the "zero dynamics" statement is wrong in any case because absolute time
    keeps advancing (the plan's §11 and §12).  Making the epoch a decision
    variable in its own right removes both problems.

    Attributes
    ----------
    t_min:
        Earliest permitted launch epoch (s, relative to a reference).
    t_max:
        Latest permitted launch epoch (s).
    """

    t_min: float
    t_max: float

    def __post_init__(self) -> None:
        if not self.t_max >= self.t_min:
            raise ValueError(f"t_max ({self.t_max}) must be >= t_min ({self.t_min})")

    def clip(self, epoch: float) -> float:
        """Clip an epoch into the window.

        Parameters
        ----------
        epoch:
            Candidate launch epoch (s).

        Returns
        -------
        float
            The epoch, clipped to ``[t_min, t_max]``.
        """
        return float(np.clip(epoch, self.t_min, self.t_max))


def phase_epochs(t_launch: float, durations: _FloatArray) -> _FloatArray:
    r"""Return absolute phase-start epochs for an explicit launch time.

    With :math:`t_L` the launch epoch and :math:`s_j` the phase durations,

    .. math:: t_k = t_L + \sum_{j<k} s_j,

    returning :math:`(t_0, \ldots, t_K)` — one entry per phase boundary,
    including the terminal epoch :math:`t_K = t_f`.

    Parameters
    ----------
    t_launch:
        Launch epoch :math:`t_L` (s).
    durations:
        Phase durations :math:`s_j` (s), shape ``(K,)``, all non-negative.

    Returns
    -------
    numpy.ndarray
        Phase boundary epochs, shape ``(K + 1,)``.

    Raises
    ------
    ValueError
        If any duration is negative.

    Examples
    --------
    >>> phase_epochs(100.0, np.array([60.0, 120.0, 300.0]))
    array([100., 160., 280., 580.])
    """
    s = np.asarray(durations, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError(f"durations must be one-dimensional, got shape {s.shape}")
    if np.any(s < 0.0):
        raise ValueError("phase durations must be non-negative")
    return np.asarray(
        np.concatenate([[float(t_launch)], float(t_launch) + np.cumsum(s)]),
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Collision geometry and terminal accuracy
# ---------------------------------------------------------------------------

def relative_covariance(
    p_vehicle: _FloatArray,
    p_object: _FloatArray,
    *,
    cross: _FloatArray | None = None,
) -> _FloatArray:
    r"""Combine two position covariances into a relative covariance.

    .. math:: P_{\mathrm{rel}} = P_v + P_s - P_{vs} - P_{vs}^\top.

    The sum :math:`P_v + P_s` used in the initial formulation is the special
    case of zero cross-covariance (the plan's §13).  The cross terms are not
    negligible whenever both states are estimated from a shared tracking
    network or share a common frame realisation: correlated errors partly
    cancel in the difference, and ignoring them overstates the relative
    dispersion.

    Parameters
    ----------
    p_vehicle:
        Vehicle position covariance, shape ``(3, 3)``.
    p_object:
        Object position covariance, shape ``(3, 3)``.
    cross:
        Cross-covariance :math:`P_{vs} = E[\delta r_v \delta r_s^\top]`,
        shape ``(3, 3)``.  Zero when omitted.

    Returns
    -------
    numpy.ndarray
        Relative covariance, shape ``(3, 3)``, symmetrised.

    Raises
    ------
    ValueError
        If any input is not ``(3, 3)``.
    """
    p_v = np.asarray(p_vehicle, dtype=np.float64)
    p_s = np.asarray(p_object, dtype=np.float64)
    for name, mat in (("p_vehicle", p_v), ("p_object", p_s)):
        if mat.shape != (3, 3):
            raise ValueError(f"{name} must have shape (3, 3), got {mat.shape}")
    out = p_v + p_s
    if cross is not None:
        c = np.asarray(cross, dtype=np.float64)
        if c.shape != (3, 3):
            raise ValueError(f"cross must have shape (3, 3), got {c.shape}")
        out = out - c - c.T
    return np.asarray(0.5 * (out + out.T), dtype=np.float64)


def generalized_chi2_cdf(
    weights: _FloatArray,
    noncentralities: _FloatArray,
    threshold: float,
    *,
    n_quad: int = 4096,
    upper: float = 200.0,
) -> float:
    r"""Evaluate :math:`P[\sum_j \lambda_j (z_j + \delta_j)^2 \le q]` by Imhof's method.

    For :math:`z_j` independent standard normals, the distribution of the
    weighted sum of non-central squares has no closed form.  Imhof's exact
    numerical inversion of the characteristic function gives

    .. math::
        P[Q \le q] = \frac12 - \frac{1}{\pi}
        \int_0^\infty \frac{\sin \theta(u)}{u\,\rho(u)}\,\mathrm{d}u,

    .. math::
        \theta(u) = \frac12\sum_j\left[\arctan(\lambda_j u)
            + \frac{\delta_j^2 \lambda_j u}{1 + \lambda_j^2 u^2}\right]
            - \frac{q u}{2},

    .. math::
        \rho(u) = \prod_j (1 + \lambda_j^2 u^2)^{1/4}
            \exp\left(\frac12\sum_j
            \frac{\delta_j^2 \lambda_j^2 u^2}{1 + \lambda_j^2 u^2}\right).

    This is the machinery behind both :func:`collision_probability` (a disk in
    the two-dimensional encounter plane) and :func:`terminal_containment` (a
    ball in three dimensions), since both are quadratic forms in a Gaussian.

    Parameters
    ----------
    weights:
        Eigenvalues :math:`\lambda_j`, shape ``(k,)``, strictly positive.
    noncentralities:
        Offsets :math:`\delta_j`, shape ``(k,)``.
    threshold:
        Threshold :math:`q`.
    n_quad:
        Number of quadrature points.
    upper:
        Upper limit of the truncated integral, in units of :math:`u`.

    Returns
    -------
    float
        The probability, clipped to ``[0, 1]``.

    Raises
    ------
    ValueError
        If shapes disagree or any weight is non-positive.
    """
    lam = np.asarray(weights, dtype=np.float64)
    delta = np.asarray(noncentralities, dtype=np.float64)
    if lam.shape != delta.shape:
        raise ValueError(f"shape mismatch: weights {lam.shape}, noncentralities {delta.shape}")
    if lam.size == 0:
        raise ValueError("weights must be non-empty")
    if np.any(lam <= 0.0):
        raise ValueError("weights must be strictly positive")
    if threshold <= 0.0:
        return 0.0

    # Scale so the integrand decays on an O(1) range of u.
    scale = float(lam.max())
    lam = lam / scale
    q = float(threshold) / scale

    u = np.linspace(1e-10, upper, n_quad)
    lu = np.outer(u, lam)  # (n_quad, k)
    d2 = delta**2

    theta = 0.5 * (np.arctan(lu) + d2 * lu / (1.0 + lu**2)).sum(axis=1) - 0.5 * q * u
    log_rho = 0.25 * np.log1p(lu**2).sum(axis=1) + 0.5 * (
        d2 * lu**2 / (1.0 + lu**2)
    ).sum(axis=1)

    integrand = np.sin(theta) / (u * np.exp(log_rho))
    value = 0.5 - float(np.trapezoid(integrand, u)) / np.pi
    return float(np.clip(value, 0.0, 1.0))


def collision_probability(
    relative_position: _FloatArray,
    relative_velocity: _FloatArray,
    covariance: _FloatArray,
    combined_radius: float,
) -> float:
    r"""Probability of collision for a short-term encounter.

    A Mahalanobis distance threshold :math:`d^\top \Sigma^{-1} d \ge
    \chi_{\mathrm{safe}}^2` defines an ellipsoid, but by itself it is not a
    probability of collision with a physical object: it carries no object
    size, and its containment level depends on the dimension of the quadratic
    form (the plan's §13).  ``chi_safe = 4`` was described in the initial
    formulation as ≈ 99.99 % for a three-dimensional Gaussian; the correct
    figure is :math:`P[\chi^2_3 \le 16] = 99.887\,\%`, an order of magnitude
    more residual risk than claimed.

    This function computes the actual probability, by Foster's method: project
    the relative position and covariance onto the encounter plane normal to
    the relative velocity, then integrate the resulting two-dimensional
    Gaussian over the disk of radius ``combined_radius``.  The Mahalanobis
    test remains useful as a cheap conservative surrogate inside the
    optimizer; this is the validation quantity.

    Parameters
    ----------
    relative_position:
        Vehicle position minus object position (m), shape ``(3,)``.
    relative_velocity:
        Vehicle velocity minus object velocity (m/s), shape ``(3,)``.
    covariance:
        Relative position covariance (m²), shape ``(3, 3)``, from
        :func:`relative_covariance`.
    combined_radius:
        Sum of the two objects' bounding radii (m).

    Returns
    -------
    float
        Probability of collision, in ``[0, 1]``.

    Raises
    ------
    ValueError
        If shapes are wrong or ``combined_radius`` is negative.
    """
    r_rel = np.asarray(relative_position, dtype=np.float64)
    v_rel = np.asarray(relative_velocity, dtype=np.float64)
    p_rel = np.asarray(covariance, dtype=np.float64)
    if r_rel.shape != (3,):
        raise ValueError(f"relative_position must have shape (3,), got {r_rel.shape}")
    if v_rel.shape != (3,):
        raise ValueError(f"relative_velocity must have shape (3,), got {v_rel.shape}")
    if p_rel.shape != (3, 3):
        raise ValueError(f"covariance must have shape (3, 3), got {p_rel.shape}")
    if combined_radius < 0.0:
        raise ValueError(f"combined_radius must be non-negative, got {combined_radius}")
    if combined_radius == 0.0:
        return 0.0

    speed = float(np.linalg.norm(v_rel))
    if speed <= 0.0:
        raise ValueError("relative_velocity must be non-zero to define an encounter plane")

    # Orthonormal basis of the plane normal to the relative velocity.
    n_hat = v_rel / speed
    seed = np.array([1.0, 0.0, 0.0]) if abs(n_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = seed - np.dot(seed, n_hat) * n_hat
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n_hat, e1)
    basis = np.stack([e1, e2])  # (2, 3)

    mu_2d = basis @ r_rel
    p_2d = basis @ p_rel @ basis.T
    p_2d = 0.5 * (p_2d + p_2d.T)

    eigvals, eigvecs = np.linalg.eigh(p_2d)
    floor = max(float(eigvals.max()), 1.0) * 1e-12
    eigvals = np.maximum(eigvals, floor)
    offsets = (eigvecs.T @ mu_2d) / np.sqrt(eigvals)

    return generalized_chi2_cdf(eigvals, offsets, combined_radius**2)


def terminal_containment(
    mean_error: _FloatArray,
    covariance: _FloatArray,
    radius: float,
) -> float:
    r"""Probability that the terminal position error lies within ``radius``.

    Evaluates :math:`P[\lVert r - r_T\rVert \le R_T]` for
    :math:`r \sim \mathcal{N}(r_T + \Delta, P_{rr})`.  A nominal intercept
    condition :math:`r(\mu(t_f)) = r_T` says nothing about dispersion; the
    statistical requirement :math:`P[\lVert r - r_T\rVert \le R_T] \ge
    1 - \epsilon_T` is the meaningful terminal constraint for a stochastic
    problem (the plan's §19).

    Parameters
    ----------
    mean_error:
        Mean terminal offset :math:`\Delta = \mu_r(t_f) - r_T` (m), shape
        ``(3,)``.
    covariance:
        Terminal position covariance (m²), shape ``(3, 3)``.
    radius:
        Acceptance radius :math:`R_T` (m).

    Returns
    -------
    float
        Containment probability, in ``[0, 1]``.

    Raises
    ------
    ValueError
        If shapes are wrong or ``radius`` is negative.
    """
    d = np.asarray(mean_error, dtype=np.float64)
    p = np.asarray(covariance, dtype=np.float64)
    if d.shape != (3,):
        raise ValueError(f"mean_error must have shape (3,), got {d.shape}")
    if p.shape != (3, 3):
        raise ValueError(f"covariance must have shape (3, 3), got {p.shape}")
    if radius < 0.0:
        raise ValueError(f"radius must be non-negative, got {radius}")
    if radius == 0.0:
        return 0.0

    p_sym = 0.5 * (p + p.T)
    eigvals, eigvecs = np.linalg.eigh(p_sym)
    floor = max(float(eigvals.max()), 1.0) * 1e-12
    eigvals = np.maximum(eigvals, floor)
    offsets = (eigvecs.T @ d) / np.sqrt(eigvals)
    return generalized_chi2_cdf(eigvals, offsets, radius**2)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def atmosphere_relative_velocity(
    position_ecef: _FloatArray,
    velocity_ecef: _FloatArray,
    *,
    wind_ecef: _FloatArray | None = None,
    rotation_rate: float = EARTH_ROTATION_RATE,
    frame: Literal["ecef", "eci"] = "ecef",
) -> _FloatArray:
    r"""Return the velocity of the vehicle relative to the air mass.

    Aerodynamic force, Mach number, dynamic pressure and stagnation heating
    all depend on :math:`\mathbf{v}_{\mathrm{rel}} = \mathbf{v} -
    \mathbf{v}_{\mathrm{atm}}`, not on the frame velocity magnitude (the
    plan's §5).  In an Earth-fixed frame the co-rotating atmosphere is at rest,
    so only the wind enters; in an inertial frame the rotation term must be
    subtracted as well.

    Parameters
    ----------
    position_ecef:
        Position in the working frame (m), shape ``(3,)``.
    velocity_ecef:
        Velocity in the working frame (m/s), shape ``(3,)``.
    wind_ecef:
        Earth-fixed wind vector (m/s), shape ``(3,)``.  Zero when omitted.
    rotation_rate:
        Planetary rotation rate (rad/s), about the ``+z`` axis.
    frame:
        ``"ecef"`` if the inputs are Earth-fixed (the co-rotation term is
        already absent), ``"eci"`` if inertial.

    Returns
    -------
    numpy.ndarray
        Air-relative velocity (m/s), shape ``(3,)``.

    Raises
    ------
    ValueError
        If shapes are wrong or ``frame`` is unknown.
    """
    r = np.asarray(position_ecef, dtype=np.float64)
    v = np.asarray(velocity_ecef, dtype=np.float64)
    if r.shape != (3,):
        raise ValueError(f"position_ecef must have shape (3,), got {r.shape}")
    if v.shape != (3,):
        raise ValueError(f"velocity_ecef must have shape (3,), got {v.shape}")

    v_atm = np.zeros(3)
    if frame == "eci":
        omega = np.array([0.0, 0.0, float(rotation_rate)])
        v_atm = v_atm + np.cross(omega, r)
    elif frame != "ecef":
        raise ValueError(f"unknown frame {frame!r}; expected 'ecef' or 'eci'")

    if wind_ecef is not None:
        w = np.asarray(wind_ecef, dtype=np.float64)
        if w.shape != (3,):
            raise ValueError(f"wind_ecef must have shape (3,), got {w.shape}")
        v_atm = v_atm + w

    return np.asarray(v - v_atm, dtype=np.float64)


def heat_flux_gradient(
    position: _FloatArray,
    velocity: _FloatArray,
    density_at: Callable[[float], float],
    *,
    body_radius: float,
    coefficient: float,
    delta_altitude: float = 50.0,
) -> tuple[float, _FloatArray]:
    r"""Stagnation heat flux and its gradient with respect to the state.

    For the Sutton–Graves form :math:`h = k\sqrt{\rho(\mathrm{alt}(r))}\,
    \lVert v\rVert^3`, the state gradient is

    .. math::
        \frac{\partial h}{\partial r}
        = \frac{k \lVert v\rVert^3}{2\sqrt{\rho}}
          \frac{\mathrm{d}\rho}{\mathrm{d}\,\mathrm{alt}}\,\hat{r},
        \qquad
        \frac{\partial h}{\partial v} = 3k\sqrt{\rho}\,\lVert v\rVert\,v .

    The density derivative is taken from the atmosphere model by central
    difference, so a layered model's piecewise scale height is respected.

    Notes
    -----
    Three errors are easy to make in the position block and were all present
    in the earlier implementation.  The chain rule needs
    :math:`\mathrm{d}\rho/\mathrm{d\,alt} \approx -\rho/H` with :math:`H` the
    *density scale height* (≈ 7 km), not the planetary radius (6371 km); the
    factor :math:`\rho` itself must not be dropped; and the sensitivity is
    along the local vertical :math:`\hat{r}`, not along the frame :math:`z`
    axis — the latter vanishes entirely at the equator.

    Parameters
    ----------
    position:
        Position in the body-fixed frame (m), shape ``(3,)``.
    velocity:
        Air-relative velocity (m/s), shape ``(3,)``; see
        :func:`atmosphere_relative_velocity`.
    density_at:
        Callable mapping geometric altitude (m) to density (kg/m³).
    body_radius:
        Planetary radius (m), for the altitude of ``position``.
    coefficient:
        Heating coefficient :math:`k`, in SI units consistent with the
        returned flux.
    delta_altitude:
        Half-width of the central difference in altitude (m).

    Returns
    -------
    flux : float
        Stagnation heat flux at the mean state, in the units implied by
        ``coefficient``.
    gradient : numpy.ndarray
        Gradient with respect to :math:`[r, v, m]`, shape ``(7,)``.  The mass
        component is zero.

    Raises
    ------
    ValueError
        If shapes are wrong or ``delta_altitude`` is not positive.
    """
    r = np.asarray(position, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    if r.shape != (3,):
        raise ValueError(f"position must have shape (3,), got {r.shape}")
    if v.shape != (3,):
        raise ValueError(f"velocity must have shape (3,), got {v.shape}")
    if delta_altitude <= 0.0:
        raise ValueError(f"delta_altitude must be positive, got {delta_altitude}")

    radius = float(np.linalg.norm(r))
    if radius <= 0.0:
        raise ValueError("position must be non-zero")
    altitude = radius - body_radius
    r_hat = r / radius

    rho = float(density_at(max(altitude, 0.0)))
    speed = float(np.linalg.norm(v))
    gradient = np.zeros(7, dtype=np.float64)
    if rho <= 0.0 or speed <= 0.0:
        return 0.0, gradient

    sqrt_rho = np.sqrt(rho)
    flux = float(coefficient * sqrt_rho * speed**3)

    # d(rho)/d(alt) by central difference, clamped at the surface.
    alt_hi = max(altitude, 0.0) + delta_altitude
    alt_lo = max(max(altitude, 0.0) - delta_altitude, 0.0)
    span = alt_hi - alt_lo
    drho_dalt = (float(density_at(alt_hi)) - float(density_at(alt_lo))) / span

    gradient[:3] = (coefficient * speed**3 / (2.0 * sqrt_rho)) * drho_dalt * r_hat
    gradient[3:6] = 3.0 * coefficient * sqrt_rho * speed * v
    return flux, gradient


# ---------------------------------------------------------------------------
# Nonlinear validation: unscented transform
# ---------------------------------------------------------------------------

def sigma_points(
    mean: _FloatArray,
    covariance: _FloatArray,
    *,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> tuple[_FloatArray, _FloatArray, _FloatArray]:
    r"""Scaled symmetric sigma points for the unscented transform.

    Linear covariance propagation is a first-order model and is least reliable
    exactly where a launch trajectory is most interesting — rapid aerodynamic
    change, exponential density variation, control saturation, staging.  The
    plan's §2 prescribes a three-level architecture in which linear covariance
    serves the optimizer's inner loop and a sigma-point or ensemble transform
    validates the result outside it.  This is that validation level.

    Parameters
    ----------
    mean:
        Mean state, shape ``(n,)``.
    covariance:
        Covariance, shape ``(n, n)``; must be positive semi-definite.
    alpha:
        Spread of the sigma points about the mean, in ``(0, 1]``.
    beta:
        Prior-knowledge parameter; ``2.0`` is optimal for a Gaussian.
    kappa:
        Secondary scaling, usually ``0`` or ``3 - n``.

    Returns
    -------
    points : numpy.ndarray
        Sigma points, shape ``(2n + 1, n)``.
    weights_mean : numpy.ndarray
        Mean weights, shape ``(2n + 1,)``.
    weights_cov : numpy.ndarray
        Covariance weights, shape ``(2n + 1,)``.

    Raises
    ------
    ValueError
        If shapes are wrong or ``alpha`` is outside ``(0, 1]``.
    """
    mu = np.asarray(mean, dtype=np.float64)
    p = np.asarray(covariance, dtype=np.float64)
    if mu.ndim != 1:
        raise ValueError(f"mean must be one-dimensional, got shape {mu.shape}")
    n = mu.size
    if p.shape != (n, n):
        raise ValueError(f"covariance must have shape ({n}, {n}), got {p.shape}")
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must lie in (0, 1], got {alpha}")

    lam = alpha**2 * (n + kappa) - n
    scale = n + lam

    p_sym = 0.5 * (p + p.T)
    eigvals, eigvecs = np.linalg.eigh(p_sym)
    eigvals = np.maximum(eigvals, 0.0)
    sqrt_p = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    spread = np.sqrt(scale) * sqrt_p

    points = np.empty((2 * n + 1, n), dtype=np.float64)
    points[0] = mu
    for i in range(n):
        points[i + 1] = mu + spread[:, i]
        points[n + i + 1] = mu - spread[:, i]

    weights_mean = np.full(2 * n + 1, 1.0 / (2.0 * scale), dtype=np.float64)
    weights_cov = weights_mean.copy()
    weights_mean[0] = lam / scale
    weights_cov[0] = lam / scale + (1.0 - alpha**2 + beta)
    return points, weights_mean, weights_cov


def unscented_propagate(
    mean: _FloatArray,
    covariance: _FloatArray,
    transform: Callable[[_FloatArray], _FloatArray],
    *,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
    process_noise: _FloatArray | None = None,
) -> tuple[_FloatArray, _FloatArray]:
    """Push a Gaussian through a nonlinear map by the unscented transform.

    Captures curvature that :func:`propagate_covariance` cannot, at the cost
    of ``2n + 1`` evaluations of ``transform``.  Intended for validating an
    optimized trajectory, not for use inside the optimizer's inner loop.

    Parameters
    ----------
    mean:
        Input mean, shape ``(n,)``.
    covariance:
        Input covariance, shape ``(n, n)``.
    transform:
        Nonlinear map from a state of shape ``(n,)`` to one of shape ``(m,)``.
    alpha:
        Sigma-point spread; see :func:`sigma_points`.
    beta:
        Prior-knowledge parameter.
    kappa:
        Secondary scaling.
    process_noise:
        Additive output-space covariance, shape ``(m, m)``.

    Returns
    -------
    mean_out : numpy.ndarray
        Transformed mean, shape ``(m,)``.
    covariance_out : numpy.ndarray
        Transformed covariance, shape ``(m, m)``, symmetrised.

    Raises
    ------
    ValueError
        If ``transform`` returns inconsistent shapes or ``process_noise`` does
        not match the output dimension.
    """
    points, w_m, w_c = sigma_points(
        mean, covariance, alpha=alpha, beta=beta, kappa=kappa
    )
    mapped = np.stack([np.atleast_1d(np.asarray(transform(pt), dtype=np.float64))
                       for pt in points])
    if mapped.ndim != 2:
        raise ValueError("transform must return a one-dimensional array")

    mean_out = np.asarray(w_m @ mapped, dtype=np.float64)
    residual = mapped - mean_out
    cov_out = np.einsum("i,ij,ik->jk", w_c, residual, residual)

    if process_noise is not None:
        q = np.asarray(process_noise, dtype=np.float64)
        if q.shape != cov_out.shape:
            raise ValueError(
                f"process_noise must have shape {cov_out.shape}, got {q.shape}"
            )
        cov_out = cov_out + q

    cov_out = 0.5 * (cov_out + cov_out.T)
    return mean_out, np.asarray(cov_out, dtype=np.float64)
