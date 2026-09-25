r"""Precomputed emission lookup table for fast rendering.

The per-sample emission computation (all molecular bands + atomic lines)
is the render bottleneck.  This module precomputes j(λ) on a grid of
(T_tr, T_ve) values with representative compositions, stored as a 3D
array that can be interpolated at render time via scipy.ndimage.map_coordinates.

For the 5-species model, the composition varies across the shock layer,
but the dominant temperature dependence (exponential Boltzmann factors)
dwarfs the composition variation for rendering purposes.  We precompute
at a few representative compositions and interpolate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates

from .emission import emission_spectrum
from .spectral_data import MOLAR_MASS, _NA, camera_rgb, cie_xyz, xyz_to_srgb

__all__ = ["EmissionLUT", "build_lut", "RGBEmissionLUT", "build_rgb_lut"]


@dataclass
class EmissionLUT:
    """Precomputed spectral emission on a (T_tr, T_ve) grid."""
    T_tr_grid: NDArray[np.float64]  # (N_tr,)
    T_ve_grid: NDArray[np.float64]  # (N_ve,)
    wavelength_nm: NDArray[np.float64]  # (N_wl,)
    table: NDArray[np.float64]      # (N_tr, N_ve, N_wl) j in W/m³/sr/nm per unit n_total

    def lookup(
        self,
        T_tr: float, T_ve: float, n_total: float,
    ) -> NDArray[np.float64]:
        """Interpolate emission spectrum at given conditions."""
        i_tr = np.interp(T_tr, self.T_tr_grid, np.arange(len(self.T_tr_grid)))
        i_ve = np.interp(T_ve, self.T_ve_grid, np.arange(len(self.T_ve_grid)))
        coords = np.array([[i_tr], [i_ve]])
        j_per_n = np.array([
            map_coordinates(self.table[:, :, w], coords, order=1, mode="nearest")[0]
            for w in range(len(self.wavelength_nm))
        ])
        return j_per_n * n_total


@dataclass
class RGBEmissionLUT:
    """Precomputed linear RGB emission on a (T_tr, T_ve) grid.

    Eliminates spectral integration at render time — directly gives
    the XYZ/RGB contribution per unit density per unit path length.
    """
    T_tr_grid: NDArray[np.float64]   # (N_tr,)
    T_ve_grid: NDArray[np.float64]   # (N_ve,)
    rgb_table: NDArray[np.float64]   # (N_tr, N_ve, 3) linear RGB per n_total

    def lookup(self, T_tr: float, T_ve: float) -> NDArray[np.float64]:
        """Returns (R, G, B) emission coefficient per unit number density."""
        i_tr = np.interp(T_tr, self.T_tr_grid,
                         np.arange(len(self.T_tr_grid)))
        i_ve = np.interp(T_ve, self.T_ve_grid,
                         np.arange(len(self.T_ve_grid)))
        coords = np.array([[i_tr], [i_ve]])
        rgb = np.array([
            map_coordinates(self.rgb_table[:, :, c], coords,
                            order=1, mode="nearest")[0]
            for c in range(3)
        ])
        return rgb

    def lookup_batch(
        self, T_tr: NDArray, T_ve: NDArray,
    ) -> NDArray[np.float64]:
        """Vectorised lookup for (M,) temperature arrays → (M, 3) RGB."""
        i_tr = np.interp(T_tr, self.T_tr_grid,
                         np.arange(len(self.T_tr_grid)))
        i_ve = np.interp(T_ve, self.T_ve_grid,
                         np.arange(len(self.T_ve_grid)))
        coords = np.array([i_tr, i_ve])  # (2, M)
        rgb = np.zeros((len(T_tr), 3))
        for c in range(3):
            rgb[:, c] = map_coordinates(
                self.rgb_table[:, :, c], coords, order=1, mode="nearest",
            )
        return rgb


def build_rgb_lut(
    T_tr_range: tuple[float, float] = (1000.0, 20000.0),
    T_ve_range: tuple[float, float] = (500.0, 15000.0),
    n_T_tr: int = 60,
    n_T_ve: int = 60,
    wavelength_nm: NDArray[np.float64] | None = None,
    progress: bool = True,
) -> RGBEmissionLUT:
    """Build a precomputed RGB emission lookup table.

    Uses a representative post-shock composition for 5-species air at M=25.
    The emission is normalised per unit total number density so that
    at render time we multiply by the local density.
    """
    if wavelength_nm is None:
        wavelength_nm = np.arange(380.0, 900.0, 2.0)

    T_tr_grid = np.linspace(T_tr_range[0], T_tr_range[1], n_T_tr)
    T_ve_grid = np.linspace(T_ve_range[0], T_ve_range[1], n_T_ve)

    n_ref = 1e22
    pressure_ref = 5000.0

    rgb_table = np.zeros((n_T_tr, n_T_ve, 3))

    for i, Ttr in enumerate(T_tr_grid):
        if progress and i % 5 == 0:
            print(f"  LUT: row {i}/{n_T_tr} (T_tr={Ttr:.0f} K)")
        for j, Tve in enumerate(T_ve_grid):
            diss_frac = np.clip((Ttr - 3000) / 7000, 0, 0.8)
            n_N2 = n_ref * (0.767 * (1 - diss_frac))
            n_O2 = n_ref * (0.233 * (1 - diss_frac * 1.2))
            n_O2 = max(n_O2, 0)
            n_NO = n_ref * 0.15 * np.exp(-((Ttr - 5000) / 3000)**2)
            n_N  = n_ref * 0.5 * diss_frac
            n_O  = n_ref * 0.3 * diss_frac

            j_spectrum = emission_spectrum(
                wavelength_nm, Ttr, Tve,
                n_N2, n_O2, n_NO, n_N, n_O, pressure_ref,
            )

            r, g, b = camera_rgb(wavelength_nm, j_spectrum)
            rgb_table[i, j] = [max(r, 0), max(g, 0), max(b, 0)]

    # Normalise per unit n_total
    rgb_table /= n_ref

    return RGBEmissionLUT(T_tr_grid, T_ve_grid, rgb_table)


def save_lut(lut: RGBEmissionLUT, path: str | Path) -> None:
    """Save LUT to disk as .npz."""
    np.savez_compressed(
        str(path),
        T_tr_grid=lut.T_tr_grid,
        T_ve_grid=lut.T_ve_grid,
        rgb_table=lut.rgb_table,
    )
    print(f"  LUT saved to {path}")


def load_lut(path: str | Path) -> RGBEmissionLUT:
    """Load LUT from disk."""
    data = np.load(str(path))
    return RGBEmissionLUT(
        T_tr_grid=data["T_tr_grid"],
        T_ve_grid=data["T_ve_grid"],
        rgb_table=data["rgb_table"],
    )
