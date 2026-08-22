"""Mahalanobis-gated Innovation-Based Adaptive Estimation (Paper I, §4.1–4.2).

The detector is the normalized innovation squared (Eq. 4.4),
:math:`d_k^2 = \\bm{\\nu}_k^\\top \\mathbf{S}_k^{-1} \\bm{\\nu}_k
\\sim \\chi^2_m` under the nominal hypothesis, gated at the design
false-alarm probability :math:`p` (Eq. 4.5). On detection the process
noise is inflated by the bounded scalar (Eq. 4.7)

.. math::

    \\alpha_k = \\min\\!\\Big(\\alpha_{\\max},
        \\max\\big(1, \\tfrac{\\max(0, \\mathrm{tr}(\\hat{\\mathbf{C}}_k
        - \\mathbf{R}))}{\\mathrm{tr}(\\mathbf{H}\\mathbf{P}_{k|k-1}
        \\mathbf{H}^\\top) + \\epsilon}\\big)\\Big),
    \\qquad \\mathbf{Q}_k^* = \\alpha_k\\,\\mathbf{Q}_{\\mathrm{nom}},

with :math:`\\hat{\\mathbf{C}}_k` the sliding-window sample innovation
covariance (Eq. 4.6). Per Prop. 5 and Remark 8, well-posedness comes
from estimating a *scalar*: :math:`\\mathbf{Q}_k^* \\succeq 0` follows
from scaling a PSD matrix by :math:`\\alpha_k \\in [1, \\alpha_{\\max}]`,
not from the trace clamp, whose role is only to keep a quiet-window
covariance deficit from going negative before the outer clamp.

The filter is **batched over replicates**: states ``(n_batch, n)``,
covariances ``(n_batch, n, n)``, with gating and adaptation applied
per replicate — V5 measures false-alarm rates and recovery times over
ensembles, so the ensemble axis is native, not bolted on. The covariance
update uses the Joseph form, which preserves symmetry and positive
semidefiniteness for any gain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.stats
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "AdaptiveConfig",
    "AdaptiveKalmanFilter",
    "FilterStepDiagnostics",
    "LinearModel",
    "chi_square_gate",
    "inflation_factor",
]

_FloatArray = NDArray[np.float64]


def _validate_symmetric_psd(mat: _FloatArray, name: str, strict: bool = False) -> None:
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"{name} must be square, got shape {mat.shape}")
    scale = float(np.max(np.abs(mat))) or 1.0
    if np.max(np.abs(mat - mat.T)) > 1e-10 * scale:
        raise ValueError(f"{name} must be symmetric")
    eigmin = float(np.min(np.linalg.eigvalsh(mat)))
    if strict and eigmin <= 0.0:
        raise ValueError(f"{name} must be positive definite, min eigenvalue {eigmin:.3e}")
    if not strict and eigmin < -1e-10 * scale:
        raise ValueError(f"{name} must be positive semidefinite, min eigenvalue {eigmin:.3e}")


@dataclass(frozen=True)
class LinearModel:
    """Discrete-time linear filter model.

    Attributes
    ----------
    transition:
        State transition :math:`\\mathbf{F}`, shape ``(n, n)``.
    observation:
        Measurement map :math:`\\mathbf{H}`, shape ``(m, n)``.
    process_noise:
        Nominal :math:`\\mathbf{Q}_{\\mathrm{nom}} \\succeq 0`.
    measurement_noise:
        :math:`\\mathbf{R} \\succ 0`.
    """

    transition: _FloatArray
    observation: _FloatArray
    process_noise: _FloatArray
    measurement_noise: _FloatArray

    def __post_init__(self) -> None:
        f = np.asarray(self.transition, dtype=np.float64)
        h = np.asarray(self.observation, dtype=np.float64)
        q = np.asarray(self.process_noise, dtype=np.float64)
        r = np.asarray(self.measurement_noise, dtype=np.float64)
        if f.ndim != 2 or f.shape[0] != f.shape[1]:
            raise ValueError(f"transition must be square, got {f.shape}")
        if h.ndim != 2 or h.shape[1] != f.shape[0]:
            raise ValueError(
                f"observation shape {h.shape} incompatible with state dimension {f.shape[0]}"
            )
        _validate_symmetric_psd(q, "process_noise")
        if q.shape != f.shape:
            raise ValueError(f"process_noise shape {q.shape} does not match state {f.shape}")
        _validate_symmetric_psd(r, "measurement_noise", strict=True)
        if r.shape != (h.shape[0], h.shape[0]):
            raise ValueError(
                f"measurement_noise shape {r.shape} does not match measurement "
                f"dimension {h.shape[0]}"
            )
        for name, arr in (("transition", f), ("observation", h),
                          ("process_noise", q), ("measurement_noise", r)):
            arr = np.ascontiguousarray(arr)
            arr.flags.writeable = False
            object.__setattr__(self, name, arr)

    @property
    def state_dim(self) -> int:
        return int(self.transition.shape[0])

    @property
    def measurement_dim(self) -> int:
        return int(self.observation.shape[0])


@dataclass(frozen=True)
class AdaptiveConfig:
    """IAE design parameters (Paper I, §4.2).

    Attributes
    ----------
    false_alarm_probability:
        Design :math:`p` of the χ² gate (Eq. 4.5); Paper I takes
        :math:`10^{-3}`.
    window_length:
        Sliding window :math:`N_w` — a genuine design parameter setting
        the variance-vs-lag trade of :math:`\\hat{\\mathbf{C}}_k`.
    alpha_max:
        Finite inflation cap; without it a persistent modeling error
        drives the filter to ignore its dynamics entirely (Remark 8).
    epsilon:
        Strictly positive denominator regularizer of Eq. (4.7).
    """

    false_alarm_probability: float = 1.0e-3
    window_length: int = 20
    alpha_max: float = 100.0
    epsilon: float = 1.0e-12

    def __post_init__(self) -> None:
        if not 0.0 < self.false_alarm_probability < 1.0:
            raise ValueError(
                f"false_alarm_probability must be in (0, 1), got {self.false_alarm_probability}"
            )
        if not (isinstance(self.window_length, (int, np.integer)) and self.window_length >= 2):
            raise ValueError(f"window_length must be an integer >= 2, got {self.window_length}")
        if not (np.isfinite(self.alpha_max) and self.alpha_max >= 1.0):
            raise ValueError(f"alpha_max must be finite and >= 1, got {self.alpha_max}")
        if not (np.isfinite(self.epsilon) and self.epsilon > 0.0):
            raise ValueError(f"epsilon must be finite and > 0, got {self.epsilon}")


def chi_square_gate(measurement_dim: int, false_alarm_probability: float) -> float:
    """Gate threshold :math:`\\chi^2_{m, 1-p}` of Eq. (4.5)."""
    if measurement_dim < 1:
        raise ValueError(f"measurement_dim must be >= 1, got {measurement_dim}")
    if not 0.0 < false_alarm_probability < 1.0:
        raise ValueError(
            f"false_alarm_probability must be in (0, 1), got {false_alarm_probability}"
        )
    return float(scipy.stats.chi2.ppf(1.0 - false_alarm_probability, df=measurement_dim))


def inflation_factor(
    innovation_cov_trace: ArrayLike,
    measurement_noise_trace: float,
    predicted_trace: ArrayLike,
    alpha_max: float,
    epsilon: float,
) -> _FloatArray:
    """The bounded scalar :math:`\\alpha_k` of Eq. (4.7), vectorized over
    replicates.

    Parameters
    ----------
    innovation_cov_trace:
        :math:`\\mathrm{tr}(\\hat{\\mathbf{C}}_k)` per replicate.
    measurement_noise_trace:
        :math:`\\mathrm{tr}(\\mathbf{R})`.
    predicted_trace:
        :math:`\\mathrm{tr}(\\mathbf{H}\\mathbf{P}_{k|k-1}\\mathbf{H}^\\top)`
        per replicate.
    alpha_max, epsilon:
        Cap and regularizer of :class:`AdaptiveConfig`.
    """
    c_tr = np.asarray(innovation_cov_trace, dtype=np.float64)
    p_tr = np.asarray(predicted_trace, dtype=np.float64)
    excess = np.maximum(0.0, c_tr - float(measurement_noise_trace))
    raw = excess / (p_tr + float(epsilon))
    return np.asarray(np.minimum(float(alpha_max), np.maximum(1.0, raw)))


@dataclass(frozen=True)
class FilterStepDiagnostics:
    """Per-step, per-replicate record the V5 analysis consumes.

    All arrays carry the replicate axis first. ``alpha`` is the inflation
    that will scale :math:`\\mathbf{Q}` in the *next* prediction (Eq.
    4.7 applies from detection onward); replicates whose gate did not
    trigger carry :math:`\\alpha = 1`.
    """

    innovation: _FloatArray
    nis: _FloatArray
    gate_triggered: NDArray[np.bool_]
    alpha: _FloatArray
    innovation_cov: _FloatArray = field(repr=False)


class AdaptiveKalmanFilter:
    """Batched Kalman filter with χ² gating and scalar IAE inflation.

    The filter owns its ensemble state: call :meth:`reset` with the
    initial mean and covariance (shared or per replicate), then
    :meth:`step` once per measurement epoch with an ``(n_batch, m)``
    measurement block. Setting ``alpha_max = 1`` in the config exactly
    disables adaptation, which is the non-adaptive baseline V5 compares
    against — the gate statistics are still recorded.
    """

    def __init__(self, model: LinearModel, config: AdaptiveConfig) -> None:
        self._model = model
        self._config = config
        self._gate = chi_square_gate(model.measurement_dim, config.false_alarm_probability)
        self._r_trace = float(np.trace(model.measurement_noise))
        self._x: _FloatArray | None = None
        self._p: _FloatArray | None = None
        self._alpha: _FloatArray | None = None
        self._window: _FloatArray | None = None  # (window_length, n_batch, m)
        self._window_count = 0
        self._step_index = 0

    @property
    def model(self) -> LinearModel:
        return self._model

    @property
    def config(self) -> AdaptiveConfig:
        return self._config

    @property
    def gate_threshold(self) -> float:
        """:math:`\\chi^2_{m, 1-p}`."""
        return self._gate

    @property
    def state(self) -> _FloatArray:
        if self._x is None:
            raise RuntimeError("filter has no state; call reset() first")
        return self._x

    @property
    def covariance(self) -> _FloatArray:
        if self._p is None:
            raise RuntimeError("filter has no state; call reset() first")
        return self._p

    @property
    def diverged(self) -> NDArray[np.bool_]:
        """Per-replicate divergence flag: any non-finite state or covariance."""
        if self._x is None or self._p is None:
            raise RuntimeError("filter has no state; call reset() first")
        bad_x = ~np.all(np.isfinite(self._x), axis=1)
        bad_p = ~np.all(np.isfinite(self._p), axis=(1, 2))
        return np.asarray(bad_x | bad_p)

    def reset(
        self,
        initial_state: ArrayLike,
        initial_covariance: ArrayLike,
        n_batch: int = 1,
    ) -> None:
        """Initialize the ensemble.

        ``initial_state`` of shape ``(n,)`` is broadcast to all
        replicates; ``(n_batch, n)`` sets each replicate individually.
        Likewise ``(n, n)`` or ``(n_batch, n, n)`` for the covariance.
        """
        n = self._model.state_dim
        if n_batch < 1:
            raise ValueError(f"n_batch must be >= 1, got {n_batch}")
        x0 = np.asarray(initial_state, dtype=np.float64)
        p0 = np.asarray(initial_covariance, dtype=np.float64)
        if x0.shape == (n,):
            x0 = np.broadcast_to(x0, (n_batch, n))
        if x0.shape != (n_batch, n):
            raise ValueError(f"initial_state must have shape ({n},) or ({n_batch}, {n})")
        if p0.shape == (n, n):
            _validate_symmetric_psd(p0, "initial_covariance")
            p0 = np.broadcast_to(p0, (n_batch, n, n))
        if p0.shape != (n_batch, n, n):
            raise ValueError(
                f"initial_covariance must have shape ({n}, {n}) or ({n_batch}, {n}, {n})"
            )
        self._x = np.array(x0, dtype=np.float64)
        self._p = np.array(p0, dtype=np.float64)
        self._alpha = np.ones(n_batch)
        self._window = np.zeros(
            (self._config.window_length, n_batch, self._model.measurement_dim)
        )
        self._window_count = 0
        self._step_index = 0

    def step(self, measurements: ArrayLike) -> FilterStepDiagnostics:
        """One predict–gate–adapt–update cycle for the whole ensemble."""
        if self._x is None or self._p is None or self._alpha is None or self._window is None:
            raise RuntimeError("filter has no state; call reset() first")
        model, cfg = self._model, self._config
        n_batch = self._x.shape[0]
        z = np.asarray(measurements, dtype=np.float64)
        if z.shape != (n_batch, model.measurement_dim):
            raise ValueError(
                f"measurements must have shape ({n_batch}, {model.measurement_dim}), "
                f"got {z.shape}"
            )
        f, h = model.transition, model.observation
        q_nom, r = model.process_noise, model.measurement_noise

        # --- predict with per-replicate inflated process noise (Eq. 4.7)
        x_pred = self._x @ f.T
        p_pred = np.einsum("ij,bjk,lk->bil", f, self._p, f)
        p_pred += self._alpha[:, np.newaxis, np.newaxis] * q_nom

        # --- innovation and gate (Eqs. 4.3–4.5)
        innovation = z - x_pred @ h.T
        s = np.einsum("ij,bjk,lk->bil", h, p_pred, h) + r
        s_inv_innov = np.linalg.solve(s, innovation[..., np.newaxis])[..., 0]
        nis = np.einsum("bi,bi->b", innovation, s_inv_innov)
        gate = nis > self._gate

        # --- sliding-window sample innovation covariance (Eq. 4.6)
        slot = self._step_index % cfg.window_length
        self._window[slot] = innovation
        self._window_count = min(self._window_count + 1, cfg.window_length)
        n_w = self._window_count
        c_hat = (
            np.einsum("wbi,wbj->bij", self._window[:n_w], self._window[:n_w]) / n_w
        )

        # --- scalar inflation, applied where the gate triggered
        predicted_trace = np.einsum("bii->b", np.einsum("ij,bjk,lk->bil", h, p_pred, h))
        alpha_candidate = inflation_factor(
            np.einsum("bii->b", c_hat),
            self._r_trace,
            predicted_trace,
            cfg.alpha_max,
            cfg.epsilon,
        )
        self._alpha = np.where(gate, alpha_candidate, 1.0)

        # --- Joseph-form measurement update
        k_gain = np.linalg.solve(
            s, h @ np.ascontiguousarray(np.swapaxes(p_pred, 1, 2))
        )  # (b, m, n): solve(S, H P_predᵀ) with P symmetric
        k_gain = np.swapaxes(k_gain, 1, 2)  # (b, n, m)
        self._x = x_pred + np.einsum("bnm,bm->bn", k_gain, innovation)
        i_kh = np.eye(model.state_dim) - k_gain @ h
        self._p = np.einsum("bij,bjk,blk->bil", i_kh, p_pred, i_kh) + np.einsum(
            "bim,mn,bjn->bij", k_gain, r, k_gain
        )
        self._p = 0.5 * (self._p + np.swapaxes(self._p, 1, 2))
        self._step_index += 1

        for arr in (innovation, nis, gate, self._alpha, c_hat):
            arr.flags.writeable = False
        return FilterStepDiagnostics(
            innovation=innovation,
            nis=nis,
            gate_triggered=gate,
            alpha=self._alpha,
            innovation_cov=c_hat,
        )
