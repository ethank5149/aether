"""Generic dispersive entry model for V7/V8.

A three-degree-of-freedom point mass under quadratic drag in an
exponential atmosphere over a flat Earth — textbook dynamics with
generic constants, traceable to no vehicle. It exists to produce
realistic-looking impact dispersions (downrange-elongated, mildly
non-Gaussian through the drag nonlinearity) so the dispersion
statistics of §6 and the throughput scaling of V8 can be exercised
end-to-end. Dispersed quantities: ballistic coefficient, entry speed,
flight-path angle, azimuth, atmospheric density bias, and two constant
wind components.

Impact detection runs on the common outer grid: each replicate's ground
crossing is located by linear interpolation inside the step where its
altitude changes sign, and the replicate's state is frozen thereafter —
no per-replicate step adaptation, so the batch stays coherent (Paper I,
Remark 9).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aether.batch.backend import Backend, get_array_module, to_numpy
from aether.batch.sampling import DispersionSpec, sample_dispersions

__all__ = ["EntryDispersionModel"]

_FloatArray = NDArray[np.float64]

G0 = 9.80665  # m/s^2
RHO0 = 1.225  # kg/m^3
H_SCALE = 8500.0  # m


@dataclass(frozen=True)
class EntryDispersionModel:
    """Batch factory for dispersed entry trajectories.

    Attributes
    ----------
    altitude:
        Release altitude (m).
    speed, flight_path_deg, azimuth_deg:
        Nominal release velocity in spherical convention (m/s, deg
        below horizontal, deg from x toward y).
    beta_nominal:
        Nominal ballistic coefficient :math:`m/(C_D A)` (kg/m²).
    relative sigmas / dispersion magnitudes:
        1-σ dispersions applied to each quantity.
    """

    altitude: float = 30_000.0
    speed: float = 2000.0
    flight_path_deg: float = -35.0
    azimuth_deg: float = 0.0
    beta_nominal: float = 8000.0
    beta_rel_sigma: float = 0.05
    speed_rel_sigma: float = 0.01
    flight_path_sigma_deg: float = 0.25
    azimuth_sigma_deg: float = 0.15
    density_bias_rel_sigma: float = 0.08
    wind_sigma: float = 8.0

    def specs(self) -> list[DispersionSpec]:
        """The dispersion set, in fixed order (see sampling module)."""
        return [
            DispersionSpec("beta", self.beta_nominal, self.beta_rel_sigma * self.beta_nominal,
                           lower=0.2 * self.beta_nominal),
            DispersionSpec("speed", self.speed, self.speed_rel_sigma * self.speed,
                           lower=0.5 * self.speed),
            DispersionSpec("flight_path_deg", self.flight_path_deg, self.flight_path_sigma_deg,
                           upper=-1.0),
            DispersionSpec("azimuth_deg", self.azimuth_deg, self.azimuth_sigma_deg),
            DispersionSpec("density_bias", 1.0, self.density_bias_rel_sigma, lower=0.3),
            DispersionSpec("wind_x", 0.0, self.wind_sigma),
            DispersionSpec("wind_y", 0.0, self.wind_sigma),
        ]

    def initial_states(self, params: dict[str, _FloatArray]) -> _FloatArray:
        """Ensemble state block ``(n_rep, 6)``: position (x, y, z), velocity."""
        gamma = np.deg2rad(params["flight_path_deg"])
        psi = np.deg2rad(params["azimuth_deg"])
        v = params["speed"]
        n = v.size
        states = np.zeros((n, 6))
        states[:, 2] = self.altitude
        states[:, 3] = v * np.cos(gamma) * np.cos(psi)
        states[:, 4] = v * np.cos(gamma) * np.sin(psi)
        states[:, 5] = v * np.sin(gamma)
        return states

    def fly(
        self,
        n_replicates: int,
        seed: int,
        backend: Backend = "numpy",
        dt: float = 0.05,
        max_time: float = 120.0,
    ) -> _FloatArray:
        """Propagate the batch to ground impact.

        Returns
        -------
        numpy.ndarray
            Impact points ``(n_replicates, 2)`` — (downrange x,
            crossrange y) in the local tangent plane, on host memory
            regardless of backend.
        """
        if dt <= 0.0 or max_time <= dt:
            raise ValueError(f"need 0 < dt < max_time, got dt={dt}, max_time={max_time}")
        xp = get_array_module(backend)
        params = sample_dispersions(self.specs(), n_replicates, seed)
        y = xp.array(self.initial_states(params), dtype=xp.float64)
        beta = xp.array(params["beta"])
        rho_bias = xp.array(params["density_bias"])
        wind = xp.stack([xp.array(params["wind_x"]), xp.array(params["wind_y"]),
                         xp.zeros(n_replicates)], axis=1)

        impact = xp.zeros((n_replicates, 2), dtype=xp.float64)
        landed = xp.zeros(n_replicates, dtype=bool)

        def rhs(y_cur: Any) -> Any:
            pos_z = y_cur[:, 2]
            vel = y_cur[:, 3:6]
            v_air = vel - wind
            v_mag = xp.sqrt(xp.sum(v_air * v_air, axis=1))
            rho = RHO0 * rho_bias * xp.exp(-xp.maximum(pos_z, 0.0) / H_SCALE)
            accel = -0.5 * rho[:, None] * v_mag[:, None] * v_air / beta[:, None]
            accel = accel + xp.array([0.0, 0.0, -G0])
            out = xp.empty_like(y_cur)
            out[:, 0:3] = vel
            out[:, 3:6] = accel
            return out

        n_steps = round(max_time / dt)
        stages = xp.empty((*y.shape, 4), dtype=xp.float64)
        for _ in range(n_steps):
            y_prev = y
            stages[:, :, 0] = rhs(y)
            stages[:, :, 1] = rhs(y + 0.5 * dt * stages[:, :, 0])
            stages[:, :, 2] = rhs(y + 0.5 * dt * stages[:, :, 1])
            stages[:, :, 3] = rhs(y + dt * stages[:, :, 2])
            y_next = y + (dt / 6.0) * (
                stages[:, :, 0] + 2.0 * stages[:, :, 1] + 2.0 * stages[:, :, 2]
                + stages[:, :, 3]
            )
            # ground-crossing detection with in-step linear interpolation
            crossing = (~landed) & (y_next[:, 2] <= 0.0)
            if bool(xp.any(crossing)):
                z0 = y_prev[crossing, 2]
                z1 = y_next[crossing, 2]
                frac = z0 / (z0 - z1)
                impact_pts = (
                    y_prev[crossing, 0:2]
                    + frac[:, None] * (y_next[crossing, 0:2] - y_prev[crossing, 0:2])
                )
                impact[crossing] = impact_pts
                landed = landed | crossing
            # frozen replicates hold their state; batch shape never changes
            y = xp.where(landed[:, None], y_prev, y_next)
            if bool(xp.all(landed)):
                break
        if not bool(xp.all(landed)):
            n_open = int(xp.sum(~landed))
            raise RuntimeError(
                f"{n_open} replicates airborne after {max_time} s; raise max_time"
            )
        return to_numpy(impact)
