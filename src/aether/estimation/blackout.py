"""Plasma blackout and ionization-gated estimation (Paper II, §6.2).

During entry the shock layer ionizes and the plasma sheath attenuates
or reflects radio-frequency signals, severing GNSS updates. The electron
number density follows from the Saha equation at post-shock conditions,
giving the plasma angular frequency (Eq. 6.4)

.. math::

    \\omega_p = \\sqrt{\\frac{n_e e^2}{m_e \\varepsilon_0}},

and when :math:`\\omega_p \\ge \\omega_{\\mathrm{GNSS}}` the signal is
evanescent in the sheath. The filter responds by zeroing the GNSS rows
of the observation matrix, so the update becomes a no-op for those
channels and the solution propagates on inertial data alone.

**Covariance growth is not quadratic.** Paper II's Prop. 3 gives

.. math::

    \\mathbf{P}_{rr}(t) \\sim \\tfrac{1}{3}q_a t^3
      + \\tfrac{1}{4}\\sigma_{b_a}^2 t^4
      + \\tfrac{1}{36}g^2\\sigma_{b_g}^2 t^6,

and its Remark is explicit that quadratic is the growth of position
*error* from an accelerometer bias, while the corresponding *covariance*
grows as :math:`t^4` and the gyro-bias channel as :math:`t^6`,
dominating past a few tens of seconds. A guidance layer sizing its
pull-up trigger on a quadratic model under-predicts badly at exactly
the durations that matter — which is why this module both provides the
closed form and propagates the error covariance independently so the
exponents can be *measured* rather than trusted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "AIR_IONIZATION_ENERGY",
    "GNSS_L1_ANGULAR_FREQUENCY",
    "BlackoutGate",
    "InertialErrorBudget",
    "plasma_frequency",
    "propagate_unaided_covariance",
    "saha_electron_density",
    "unaided_position_variance",
]

_FloatArray = NDArray[np.float64]

# --- physical constants (SI) ------------------------------------------------
_ELECTRON_CHARGE = 1.602176634e-19
_ELECTRON_MASS = 9.1093837015e-31
_VACUUM_PERMITTIVITY = 8.8541878128e-12
_BOLTZMANN = 1.380649e-23
_PLANCK = 6.62607015e-34

#: First ionization energy of atomic nitrogen (J) — the species that
#: dominates equilibrium ionization in air behind a strong shock.
AIR_IONIZATION_ENERGY = 14.534 * _ELECTRON_CHARGE
#: GPS L1 carrier angular frequency (rad/s).
GNSS_L1_ANGULAR_FREQUENCY = 2.0 * np.pi * 1.57542e9


def saha_electron_density(
    temperature: ArrayLike,
    number_density: ArrayLike,
    ionization_energy: float = AIR_IONIZATION_ENERGY,
    degeneracy_ratio: float = 1.0,
) -> _FloatArray:
    """Equilibrium electron density from the Saha equation (m⁻³).

    For single ionization with ionization fraction :math:`\\alpha`,

    .. math::

        \\frac{\\alpha^2}{1-\\alpha} = \\frac{1}{n}
        \\left(\\frac{2\\pi m_e k T}{h^2}\\right)^{3/2}
        2\\frac{g_i}{g_n}\\,e^{-\\chi/kT},

    solved in closed form for :math:`\\alpha \\in [0, 1]`; the electron
    density is :math:`n_e = \\alpha n`.

    Parameters
    ----------
    temperature:
        Post-shock temperature (K), > 0.
    number_density:
        Total heavy-particle number density (m⁻³), > 0.
    ionization_energy:
        :math:`\\chi` (J).
    degeneracy_ratio:
        :math:`g_i/g_n` partition-function ratio.

    Notes
    -----
    Equilibrium ionization is assumed, which Paper II's Limitation 7
    flags explicitly: at high altitude and high velocity the shock layer
    is chemically frozen or in nonequilibrium and the electron density
    will be misestimated.
    """
    t = np.asarray(temperature, dtype=np.float64)
    n = np.asarray(number_density, dtype=np.float64)
    if np.any(t <= 0.0) or not np.all(np.isfinite(t)):
        raise ValueError("temperature must be finite and > 0")
    if np.any(n <= 0.0) or not np.all(np.isfinite(n)):
        raise ValueError("number_density must be finite and > 0")
    if not (np.isfinite(ionization_energy) and ionization_energy > 0.0):
        raise ValueError("ionization_energy must be finite and > 0")

    thermal = (2.0 * np.pi * _ELECTRON_MASS * _BOLTZMANN * t / _PLANCK**2) ** 1.5
    with np.errstate(over="ignore", under="ignore"):
        k_saha = 2.0 * degeneracy_ratio * thermal * np.exp(
            -ionization_energy / (_BOLTZMANN * t)
        ) / n
    # alpha^2/(1-alpha) = K  =>  alpha = (-K + sqrt(K^2 + 4K))/2, the root in [0, 1]
    k_saha = np.minimum(k_saha, 1.0e300)
    alpha = 0.5 * (-k_saha + np.sqrt(k_saha * k_saha + 4.0 * k_saha))
    alpha = np.clip(alpha, 0.0, 1.0)
    return np.asarray(alpha * n)


def plasma_frequency(electron_density: ArrayLike) -> _FloatArray:
    """Plasma angular frequency :math:`\\omega_p` (rad/s), Paper II Eq. (6.4)."""
    n_e = np.asarray(electron_density, dtype=np.float64)
    if np.any(n_e < 0.0) or not np.all(np.isfinite(n_e)):
        raise ValueError("electron_density must be finite and >= 0")
    return np.asarray(
        np.sqrt(n_e * _ELECTRON_CHARGE**2 / (_ELECTRON_MASS * _VACUUM_PERMITTIVITY))
    )


@dataclass
class BlackoutGate:
    """Hysteretic GNSS availability gate.

    A bare threshold on :math:`\\omega_p \\ge \\omega_{\\mathrm{GNSS}}`
    chatters when the plasma frequency hovers at the boundary, which is
    exactly the failure mode II-V6 is asked to rule out. The gate
    therefore uses a Schmitt trigger: signal is *lost* at
    :math:`\\omega_p \\ge \\omega_{\\mathrm{GNSS}}` and only *reacquired*
    once :math:`\\omega_p` falls below
    :math:`\\omega_{\\mathrm{GNSS}}/(1 + \\text{margin})`.

    Attributes
    ----------
    carrier_frequency:
        :math:`\\omega_{\\mathrm{GNSS}}` (rad/s).
    hysteresis:
        Fractional reacquisition margin; zero reproduces the bare
        threshold, which the verification uses as its chattering
        baseline.
    """

    carrier_frequency: float = GNSS_L1_ANGULAR_FREQUENCY
    hysteresis: float = 0.1
    _blacked_out: bool = False

    def __post_init__(self) -> None:
        if not (np.isfinite(self.carrier_frequency) and self.carrier_frequency > 0.0):
            raise ValueError("carrier_frequency must be finite and > 0")
        if not (np.isfinite(self.hysteresis) and self.hysteresis >= 0.0):
            raise ValueError("hysteresis must be finite and >= 0")

    @property
    def blacked_out(self) -> bool:
        return self._blacked_out

    def reset(self, blacked_out: bool = False) -> None:
        self._blacked_out = bool(blacked_out)

    def update(self, plasma_omega: float) -> bool:
        """Advance the gate; returns True when GNSS is unavailable."""
        omega = float(plasma_omega)
        if not (np.isfinite(omega) and omega >= 0.0):
            raise ValueError(f"plasma frequency must be finite and >= 0, got {omega}")
        release = self.carrier_frequency / (1.0 + self.hysteresis)
        if self._blacked_out:
            if omega < release:
                self._blacked_out = False
        elif omega >= self.carrier_frequency:
            self._blacked_out = True
        return self._blacked_out

    def observation_mask(self, plasma_omega: float, n_channels: int) -> _FloatArray:
        """Diagonal scaling for the GNSS rows of :math:`\\mathbf{H}`.

        Zeros during blackout, so the update is a no-op for those
        channels and the solution propagates on inertial data alone.
        """
        if n_channels < 1:
            raise ValueError(f"n_channels must be >= 1, got {n_channels}")
        return np.zeros(n_channels) if self.update(plasma_omega) else np.ones(n_channels)


@dataclass(frozen=True)
class InertialErrorBudget:
    """Strapdown error sources entering Paper II, Eq. (6.5)."""

    accel_psd: float
    """:math:`q_a`, accelerometer white-noise PSD ((m/s²)²/Hz)."""
    accel_bias_variance: float
    """:math:`\\sigma_{b_a}^2` ((m/s²)²)."""
    gyro_bias_variance: float
    """:math:`\\sigma_{b_g}^2` ((rad/s)²)."""
    gravity: float = 9.80665
    """:math:`g` (m/s²), the lever the gyro bias acts through."""

    def __post_init__(self) -> None:
        for name in ("accel_psd", "accel_bias_variance", "gyro_bias_variance"):
            val = float(getattr(self, name))
            if not (np.isfinite(val) and val >= 0.0):
                raise ValueError(f"{name} must be finite and >= 0, got {val}")
        if not (np.isfinite(self.gravity) and self.gravity > 0.0):
            raise ValueError("gravity must be finite and > 0")


def unaided_position_variance(
    duration: ArrayLike, budget: InertialErrorBudget, channels: str = "all"
) -> _FloatArray:
    """Closed-form :math:`P_{rr}(t)` of Paper II, Eq. (6.5) (m²).

    Parameters
    ----------
    duration:
        Blackout elapsed time (s), >= 0.
    budget:
        Error sources.
    channels:
        ``"all"`` sums the three terms; ``"velocity_random_walk"``,
        ``"accel_bias"`` and ``"gyro_bias"`` isolate one each, which is
        how the verification measures the individual exponents.
    """
    t = np.asarray(duration, dtype=np.float64)
    if np.any(t < 0.0) or not np.all(np.isfinite(t)):
        raise ValueError("duration must be finite and >= 0")
    terms = {
        "velocity_random_walk": budget.accel_psd * t**3 / 3.0,
        "accel_bias": budget.accel_bias_variance * t**4 / 4.0,
        "gyro_bias": budget.gravity**2 * budget.gyro_bias_variance * t**6 / 36.0,
    }
    if channels == "all":
        return np.asarray(sum(terms.values()))
    if channels not in terms:
        raise ValueError(
            f"channels must be 'all' or one of {sorted(terms)}, got {channels!r}"
        )
    return np.asarray(terms[channels])


def propagate_unaided_covariance(
    times: ArrayLike, budget: InertialErrorBudget, channels: str = "all"
) -> _FloatArray:
    """Position variance from an *independent* linear covariance propagation.

    Integrates the augmented strapdown error state

    .. math::

        \\mathbf{x} = [\\delta r,\\ \\delta v,\\ \\theta,\\ b_a,\\ b_g],
        \\qquad
        \\dot{\\delta r} = \\delta v, \\quad
        \\dot{\\delta v} = b_a + g\\theta, \\quad
        \\dot\\theta = b_g,

    with the biases as constant states, through the Lyapunov equation
    :math:`\\dot{\\mathbf{P}} = \\mathbf{F}\\mathbf{P} +
    \\mathbf{P}\\mathbf{F}^\\top + \\mathbf{Q}`. Carrying the biases and
    the tilt as *states* is what makes each channel's power of :math:`t`
    emerge from the integration rather than being imposed.

    This shares no code and no closed form with
    :func:`unaided_position_variance`, so agreement between the two is a
    genuine check of Prop. 3 rather than a restatement of it.
    """
    import scipy.integrate

    t = np.asarray(times, dtype=np.float64)
    if t.ndim != 1 or t.size < 2 or np.any(np.diff(t) <= 0.0) or t[0] < 0.0:
        raise ValueError("times must be a strictly increasing, non-negative 1-D array")

    use_vrw = channels in ("all", "velocity_random_walk")
    use_ba = channels in ("all", "accel_bias")
    use_bg = channels in ("all", "gyro_bias")
    if not (use_vrw or use_ba or use_bg):
        raise ValueError(
            f"channels must be 'all', 'velocity_random_walk', 'accel_bias' or "
            f"'gyro_bias', got {channels!r}"
        )

    f = np.zeros((5, 5))
    f[0, 1] = 1.0  # dr' = dv
    f[1, 3] = 1.0 if use_ba else 0.0  # dv' <- b_a
    f[1, 2] = budget.gravity if use_bg else 0.0  # dv' <- g*theta
    f[2, 4] = 1.0 if use_bg else 0.0  # theta' = b_g

    q = np.zeros((5, 5))
    if use_vrw:
        q[1, 1] = budget.accel_psd

    p0 = np.zeros((5, 5))
    if use_ba:
        p0[3, 3] = budget.accel_bias_variance
    if use_bg:
        p0[4, 4] = budget.gyro_bias_variance

    def rhs(_time: float, flat: _FloatArray) -> _FloatArray:
        p = flat.reshape(5, 5)
        return np.asarray((f @ p + p @ f.T + q).reshape(-1))

    # The initial condition is the covariance at t = 0, so the integration
    # always starts there even when the requested samples do not.
    eval_times = t if t[0] == 0.0 else np.concatenate([[0.0], t])
    sol = scipy.integrate.solve_ivp(
        rhs,
        (0.0, float(t[-1])),
        p0.reshape(-1),
        t_eval=eval_times,
        method="DOP853",
        rtol=1e-12,
        atol=1e-30,
    )
    if not sol.success:  # pragma: no cover - a linear system always integrates
        raise RuntimeError(f"covariance propagation failed: {sol.message}")
    variance = sol.y.reshape(5, 5, -1)[0, 0, :]
    return np.asarray(variance if t[0] == 0.0 else variance[1:])
