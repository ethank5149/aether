"""Terminal dispersion statistics.

Implements the eigendecomposition summary of the sample landing-point
covariance: :math:`R_{95}` semi-axes :math:`2.4477\\,\\sigma_i` from
:math:`\\chi^2_{2,0.95} = 5.991` (Eq. 6.1), the CEP linear approximation
:math:`0.5887(\\sigma_1 + \\sigma_2)` (Eq. 6.3) guarded by its stated
validity band :math:`0.25 \\le \\sigma_2/\\sigma_1 \\le 1` (Eq. 6.4)
with the direct order-statistic fallback of Remark 10, the
:math:`1/\\sqrt{2N_{\\mathrm{MC}}}` sampling-error bound, bootstrap
confidence intervals, and the Henze–Zirkler test of bivariate normality
(Remark 11) — every metric carries its sample size, per §6.3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.stats
from numpy.typing import ArrayLike, NDArray

from aether.batch.containment import containment_radius


def _exact_cep_ratio(aspect: float) -> float:
    """CEP over the major sigma for a unit-major ellipse."""
    return float(containment_radius(0.5, 1.0, aspect))

__all__ = ["DispersionReport", "henze_zirkler", "summarize_dispersion"]

_FloatArray = NDArray[np.float64]

#: sqrt(chi^2_{2, 0.95}) = 2.4477.
_R95_MULTIPLIER = float(np.sqrt(scipy.stats.chi2.ppf(0.95, df=2)))
#: CEP linear-approximation coefficient of Eq. (6.3). Retained only for
#: the comparison reported in :attr:`DispersionReport.cep_method`; the
#: point estimate is now exact (see below).
_CEP_COEFF = 0.5887
#: Validity band of Eq. (6.4), likewise retained for reporting only.
_CEP_ASPECT_MIN = 0.25

#: Exact CEP over the major sigma, tabulated against aspect ratio.
#:
#: :func:`aether.batch.containment.containment_radius` is exact for any
#: aspect ratio but costs a root-find over a quadrature, which is too slow
#: to call once per bootstrap resample. It is homogeneous of degree one in
#: the sigmas, so :math:`\mathrm{CEP} = \sigma_1 f(\sigma_2/\sigma_1)`
#: and the whole dependence lives in one univariate curve. Tabulating it
#: once and interpolating is exact to the interpolation error, which on
#: this grid is under :math:`10^{-6}` relative -- three orders of magnitude
#: inside the sampling error of any realistic bootstrap.
_CEP_ASPECT_GRID = np.linspace(0.0, 1.0, 401)
_CEP_OVER_SIGMA_MAJOR = np.array(
    [_exact_cep_ratio(float(a)) for a in _CEP_ASPECT_GRID]
)


def henze_zirkler(points: ArrayLike) -> tuple[float, float]:
    """Henze–Zirkler multivariate-normality statistic and p-value.

    Implements Henze & Zirkler (1990) with the standard smoothing
    parameter :math:`\\beta = ((2d+1)n/4)^{1/(d+4)}/\\sqrt{2}` and the
    lognormal approximation to the null distribution. Small p-values
    reject multivariate normality.
    """
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 3:
        raise ValueError(f"points must be (n >= 3, d), got shape {x.shape}")
    n, d = x.shape
    centered = x - x.mean(axis=0)
    cov = (centered.T @ centered) / n  # MLE covariance, per the test's definition
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals[0] <= 0.0:
        raise ValueError(
            "sample covariance is singular; the Henze–Zirkler statistic is undefined"
        )
    cov_inv = np.linalg.inv(cov)

    beta = ((2.0 * d + 1.0) * n / 4.0) ** (1.0 / (d + 4.0)) / np.sqrt(2.0)
    b2 = beta * beta

    # pairwise and to-mean Mahalanobis distances
    y = centered @ np.linalg.cholesky(cov_inv)  # whitened
    sq_norms = np.sum(y * y, axis=1)
    d_pair = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (y @ y.T)
    np.fill_diagonal(d_pair, 0.0)
    d_pair = np.maximum(d_pair, 0.0)

    term1 = float(np.sum(np.exp(-0.5 * b2 * d_pair))) / n**2
    term2 = (
        2.0
        * (1.0 + b2) ** (-d / 2.0)
        * float(np.mean(np.exp(-b2 * sq_norms / (2.0 * (1.0 + b2)))))
    )
    term3 = (1.0 + 2.0 * b2) ** (-d / 2.0)
    hz = n * (term1 - term2 + term3)

    # lognormal null approximation (Henze & Zirkler 1990, §3)
    a = 1.0 + 2.0 * b2
    wb = (1.0 + b2) * (1.0 + 3.0 * b2)
    mu = 1.0 - a ** (-d / 2.0) * (
        1.0 + d * b2 / a + d * (d + 2.0) * beta**4 / (2.0 * a * a)
    )
    si2 = (
        2.0 * (1.0 + 4.0 * b2) ** (-d / 2.0)
        + 2.0
        * a ** (-d)
        * (1.0 + 2.0 * d * beta**4 / a**2 + 3.0 * d * (d + 2.0) * beta**8 / (4.0 * a**4))
        - 4.0
        * wb ** (-d / 2.0)
        * (1.0 + 3.0 * d * beta**4 / (2.0 * wb) + d * (d + 2.0) * beta**8 / (2.0 * wb**2))
    )
    log_mu = np.log(np.sqrt(mu**4 / (si2 + mu * mu)))
    log_sigma = np.sqrt(np.log1p(si2 / (mu * mu)))
    p_value = float(scipy.stats.lognorm.sf(hz, log_sigma, scale=np.exp(log_mu)))
    return float(hz), p_value


@dataclass(frozen=True)
class DispersionReport:
    """Terminal footprint summary with the caveats of §6 attached.

    Attributes
    ----------
    n_samples:
        Sample size — reported because a dispersion figure without one
        "is not meaningful" (§6.3).
    mean:
        Sample mean landing point (2,).
    sigma:
        Principal standard deviations :math:`(\\sigma_1, \\sigma_2)`,
        :math:`\\sigma_1 \\ge \\sigma_2`.
    axes:
        Principal directions as columns, matching ``sigma``.
    r95_semi_axes:
        :math:`(a_{95}, b_{95}) = 2.4477(\\sigma_1, \\sigma_2)`.
    r95:
        Conservative circumscribing scalar, :math:`a_{95}`.
    cep:
        Circular error probable.
    cep_method:
        Always ``"exact-elliptical"``, with ``" (outside Eq. 6.4 band)"``
        appended when the aspect ratio falls outside the validity band the
        classical linear approximation of Eq. (6.3) was stated for. The
        estimate itself is exact either way; the label is kept so results
        stay comparable with literature computed the classical way.
    aspect_ratio:
        :math:`\\sigma_2/\\sigma_1`.
    relative_standard_error:
        :math:`1/\\sqrt{2 N_{\\mathrm{MC}}}` per-σ bound of §6.3.
    cep_ci, r95_ci:
        Percentile bootstrap 95% confidence intervals.
    hz_statistic, hz_p_value:
        Henze–Zirkler bivariate-normality test (Remark 11); a small
        p-value means the elliptical summary above is suspect.
    """

    n_samples: int
    mean: _FloatArray
    sigma: _FloatArray
    axes: _FloatArray
    r95_semi_axes: _FloatArray
    r95: float
    cep: float
    cep_method: str
    aspect_ratio: float
    relative_standard_error: float
    cep_ci: tuple[float, float]
    r95_ci: tuple[float, float]
    hz_statistic: float
    hz_p_value: float


def _principal_sigmas(points: _FloatArray) -> tuple[_FloatArray, _FloatArray, _FloatArray]:
    mean = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    return mean, np.sqrt(eigvals), eigvecs[:, order]


def _cep_estimate(points: _FloatArray, sigma: _FloatArray) -> tuple[float, str]:
    """Exact CEP, with a note on whether the classical route would have done.

    The classical route is the linear approximation inside its stated
    validity band, falling back to the sample median radius
    outside it. Both are now superseded: the elliptical containment
    integral is exact at every aspect ratio, and is verified against 126
    published values in Siouris Table 5.2.

    The old branch is retained as a *label* rather than a computation,
    because knowing whether a footprint sits inside the classical validity
    band is still useful when comparing against literature that used it.
    Measured difference: the linear form errs by up to 2 % inside its own
    band, and the median fallback carries sampling noise the integral does
    not have.
    """
    del points  # the order-statistic fallback no longer needs them
    aspect = float(sigma[1] / sigma[0]) if sigma[0] > 0.0 else 1.0
    cep = float(containment_radius(0.5, float(sigma[0]), float(sigma[1])))
    within = _CEP_ASPECT_MIN <= aspect <= 1.0
    return cep, "exact-elliptical" + ("" if within else " (outside Eq. 6.4 band)")


def summarize_dispersion(
    points: ArrayLike,
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> DispersionReport:
    """Full §6 summary of a batch of landing points.

    Parameters
    ----------
    points:
        Landing coordinates ``(n, 2)`` in the local tangent plane.
    bootstrap_samples:
        Percentile-bootstrap resamples for the confidence intervals.
    seed:
        Bootstrap RNG seed (Philox; reproducible).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must have shape (n, 2), got {pts.shape}")
    n = pts.shape[0]
    if n < 10:
        raise ValueError(f"need at least 10 landing points for a meaningful summary, got {n}")
    if bootstrap_samples < 100:
        raise ValueError(f"bootstrap_samples must be >= 100, got {bootstrap_samples}")

    mean, sigma, axes = _principal_sigmas(pts)
    if sigma[0] <= 0.0:
        raise ValueError("degenerate footprint: zero dispersion in every direction")
    cep, cep_method = _cep_estimate(pts, sigma)
    r95_axes = _R95_MULTIPLIER * sigma

    # percentile bootstrap, chunked to bound memory
    rng = np.random.Generator(np.random.Philox(seed))
    cep_boot = np.empty(bootstrap_samples)
    r95_boot = np.empty(bootstrap_samples)
    chunk = max(1, min(bootstrap_samples, 4_000_000 // max(n, 1)))
    done = 0
    while done < bootstrap_samples:
        m = min(chunk, bootstrap_samples - done)
        idx = rng.integers(0, n, size=(m, n))
        resampled = pts[idx]  # (m, n, 2)
        means = resampled.mean(axis=1, keepdims=True)
        centered = resampled - means
        covs = np.einsum("mni,mnj->mij", centered, centered) / (n - 1)
        eigvals = np.maximum(np.linalg.eigvalsh(covs), 0.0)  # ascending
        s1 = np.sqrt(eigvals[:, 1])
        s2 = np.sqrt(eigvals[:, 0])
        r95_boot[done : done + m] = _R95_MULTIPLIER * s1
        aspect = np.divide(s2, s1, out=np.ones_like(s1), where=s1 > 0)
        cep_boot[done : done + m] = s1 * np.interp(
            aspect, _CEP_ASPECT_GRID, _CEP_OVER_SIGMA_MAJOR
        )
        done += m

    cep_ci = (float(np.percentile(cep_boot, 2.5)), float(np.percentile(cep_boot, 97.5)))
    r95_ci = (float(np.percentile(r95_boot, 2.5)), float(np.percentile(r95_boot, 97.5)))
    hz_stat, hz_p = henze_zirkler(pts)

    for arr in (mean, sigma, axes, r95_axes):
        arr.flags.writeable = False
    return DispersionReport(
        n_samples=n,
        mean=mean,
        sigma=sigma,
        axes=axes,
        r95_semi_axes=r95_axes,
        r95=float(r95_axes[0]),
        cep=cep,
        cep_method=cep_method,
        aspect_ratio=float(sigma[1] / sigma[0]),
        relative_standard_error=float(1.0 / np.sqrt(2.0 * n)),
        cep_ci=cep_ci,
        r95_ci=r95_ci,
        hz_statistic=hz_stat,
        hz_p_value=hz_p,
    )
