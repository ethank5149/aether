r"""Ray-marching radiative transfer through SU2 NEMO volume data.

Loads the Tecplot-ASCII volume output, builds a KD-tree for nearest-
neighbour interpolation, and integrates the radiative transfer equation

    dI_λ/ds = j_λ(s)

along each ray (optically thin limit — no self-absorption, valid for
the low-density 60 km regime where τ ≪ 1 across the shock layer).
At higher densities where absorption matters, the full
dI/ds = j − κ·I form can be substituted.

The volume reader parses SU2's native Tecplot output, extracting the
fields the emission model needs: coordinates, partial densities (or
mass fractions + total density), temperatures, and pressure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .emission import total_emission

__all__ = ["VolumeData", "load_volume", "integrate_ray"]


class VolumeData:
    """Parsed SU2 NEMO volume solution with spatial lookup."""

    def __init__(
        self,
        coords: NDArray[np.float64],
        rho_species: NDArray[np.float64],
        T_tr: NDArray[np.float64],
        T_ve: NDArray[np.float64],
        pressure: NDArray[np.float64],
        velocity: NDArray[np.float64],
        mach: NDArray[np.float64],
    ) -> None:
        self.coords = coords            # (N, 3) in metres
        self.rho_species = rho_species   # (N, 5) [N2, O2, NO, N, O] kg/m³
        self.T_tr = T_tr                 # (N,)
        self.T_ve = T_ve                 # (N,)
        self.pressure = pressure         # (N,)
        self.velocity = velocity         # (N, 3)
        self.mach = mach                 # (N,)
        self._tree: cKDTree | None = None

    @property
    def tree(self) -> cKDTree:
        if self._tree is None:
            self._tree = cKDTree(self.coords)
        return self._tree

    def sample(self, point: NDArray[np.float64]) -> tuple[
        NDArray[np.float64], float, float, float
    ]:
        """Nearest-neighbour lookup at a single 3D point.

        Returns (rho_species[5], T_tr, T_ve, pressure).
        """
        _, idx = self.tree.query(point)
        return (
            self.rho_species[idx],
            float(self.T_tr[idx]),
            float(self.T_ve[idx]),
            float(self.pressure[idx]),
        )

    def sample_batch(
        self, points: NDArray[np.float64], k: int = 1,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Vectorised spatial lookup for (M, 3) points.

        k=1 uses nearest-neighbour. k>1 uses inverse-distance weighting
        across k neighbours for smooth interpolation.
        """
        if k == 1:
            _, idx = self.tree.query(points)
            return (
                self.rho_species[idx],
                self.T_tr[idx],
                self.T_ve[idx],
                self.pressure[idx],
            )

        dists, idxs = self.tree.query(points, k=k)  # (M, k)
        eps = 1e-12
        w = 1.0 / (dists + eps)
        w /= w.sum(axis=1, keepdims=True)

        rho = np.einsum("mk,mks->ms", w, self.rho_species[idxs])
        T_tr = np.einsum("mk,mk->m", w, self.T_tr[idxs])
        T_ve = np.einsum("mk,mk->m", w, self.T_ve[idxs])
        pres = np.einsum("mk,mk->m", w, self.pressure[idxs])
        return rho, T_tr, T_ve, pres

    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Axis-aligned bounding box (min_corner, max_corner)."""
        return self.coords.min(axis=0), self.coords.max(axis=0)


def load_volume(path: str | Path) -> VolumeData:
    """Parse a SU2 NEMO Tecplot-ASCII volume_flow.dat file.

    Expected VARIABLES line for 5-species NEMO:
      x, y, z, Density_0..4, Momentum_x/y/z, Energy, Energy_ve,
      MassFrac_0..4, Pressure, Temperature_tr, Temperature_ve,
      Velocity_x/y/z, Mach, Pressure_Coefficient
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    header_lines = 0
    n_nodes = 0
    with open(path) as fh:
        for line in fh:
            header_lines += 1
            if line.strip().startswith("ZONE"):
                parts = line.upper().replace(",", " ").split()
                for i, tok in enumerate(parts):
                    if tok.startswith("NODES="):
                        val = tok.split("=")[1]
                        if val:
                            n_nodes = int(val)
                        elif i + 1 < len(parts):
                            n_nodes = int(parts[i + 1])
                break

    if n_nodes == 0:
        raise ValueError(f"could not parse NODES from {path}")

    print(f"  loading {n_nodes:,} nodes from {path.name} ...")
    raw = np.loadtxt(path, skiprows=header_lines, max_rows=n_nodes)
    print(f"  loaded {raw.shape}")

    # SU2 NEMO 5-species column layout (0-indexed):
    # 0:x 1:y 2:z 3-7:Density_0..4 8-10:Mom 11:E 12:E_ve
    # 13-17:MassFrac_0..4 18:P 19:T_tr 20:T_ve
    # 21-23:Vel 24:Mach 25:Cp
    coords = raw[:, 0:3]
    rho_species = raw[:, 3:8]
    pressure = raw[:, 18]
    T_tr = raw[:, 19]
    T_ve = raw[:, 20]
    velocity = raw[:, 21:24]
    mach = raw[:, 24]

    return VolumeData(coords, rho_species, T_tr, T_ve, pressure, velocity, mach)


def integrate_ray(
    volume: VolumeData,
    origin: NDArray[np.float64],
    direction: NDArray[np.float64],
    wavelength_nm: NDArray[np.float64],
    t_min: float,
    t_max: float,
    n_steps: int = 500,
    T_threshold: float = 1000.0,
) -> NDArray[np.float64]:
    r"""Integrate emission along a single ray through the volume.

    Returns spectral radiance I(λ) in W/m²/sr/nm at the camera.

    Uses the optically-thin approximation: I(λ) = ∫ j(λ, s) ds.
    Points where T_tr < T_threshold are skipped (cold freestream
    does not radiate).
    """
    direction = direction / np.linalg.norm(direction)
    ds = (t_max - t_min) / n_steps
    I = np.zeros_like(wavelength_nm)

    t_values = np.linspace(t_min, t_max, n_steps)
    points = origin[np.newaxis, :] + t_values[:, np.newaxis] * direction[np.newaxis, :]

    rho_batch, T_tr_batch, T_ve_batch, p_batch = volume.sample_batch(points)

    for i in range(n_steps):
        if T_tr_batch[i] < T_threshold:
            continue
        j = total_emission(
            wavelength_nm, T_tr_batch[i], T_ve_batch[i],
            rho_batch[i], p_batch[i],
        )
        I += j * ds

    return I
