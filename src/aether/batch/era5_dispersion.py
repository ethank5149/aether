"""Terminal dispersion (CEP / R95) driven by *measured* ERA5 winds.

The generic batched entry model in :mod:`aether.batch.entry_demo` draws its
wind from a fixed isotropic Gaussian (``wind_sigma``), which assumes the very
thing a dispersion number is supposed to measure: the wind is treated as
symmetric, uncorrelated with altitude, and of an assumed magnitude. Real wind
is none of those. It has a prevailing direction, a magnitude that varies by
site and season, and -- the part that actually drives terminal dispersion --
a *vertical structure* that is correlated within any single profile.

This module replaces the assumed wind with the empirical one: the ensemble
of :class:`aether.atmosphere.era5.WindEnsemble` (whose ``sample`` is "a Monte
Carlo trial's atmosphere") connected to the terminal-dispersion Monte Carlo
and reduced by :mod:`aether.batch.dispersion`.

Fidelity: each Monte Carlo replicate flies through its *own* measured profile.
The eastward and northward wind and the air density are interpolated at the
vehicle's altitude at every integration step — not reduced to a single scalar
wind, and not an exponential atmosphere. Above the ERA5 ceiling, where there is
no measured wind, density follows the US Standard 1976 atmosphere and the wind
is zero (the absence of data, never an assumed value).

There is deliberately **no approximation fallback**: if no real wind ensemble
resolves for a site, :func:`terminal_dispersion` raises rather than substitute a
guessed isotropic wind, because a CEP against a guessed wind measures nothing.

:func:`representative_wind` and :func:`sample_era5_winds` remain as diagnostics
(a density-weighted reduction of a profile to a single vector), but the entry
Monte Carlo does not use them — it consumes the full profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from aether.atmosphere.model import earth_atmosphere
from aether.atmosphere.standard import gravity
from aether.batch.backend import Backend, get_array_module, to_numpy
from aether.batch.dispersion import DispersionReport, summarize_dispersion
from aether.batch.entry_demo import EntryDispersionModel
from aether.batch.sampling import DispersionSpec, sample_dispersions

if TYPE_CHECKING:
    from aether.atmosphere.era5 import ERA5Box, ERA5Column, WindEnsemble

_FloatArray = NDArray[np.float64]

__all__ = [
    "ERA5EntryDispersion",
    "latin_hypercube_wind_ensemble",
    "load_file_ensemble",
    "load_site_ensemble",
    "representative_wind",
    "sample_era5_winds",
    "terminal_dispersion",
]

#: Altitude band (m) over which the profile is reduced to a representative
#: wind. The lower troposphere carries most of the ballistic crosswind
#: deflection because the air is densest there.
DEFAULT_BAND = (0.0, 15.0e3)


def representative_wind(
    column: ERA5Column, band: tuple[float, float] = DEFAULT_BAND
) -> tuple[float, float]:
    """Density-weighted mean (eastward, northward) wind over ``band`` (m/s).

    A crosswind's lateral impulse scales with the aerodynamic force, hence
    with air density, so the profile is collapsed to the wind the entry model
    actually feels rather than to a bare vertical average.
    """
    altitude = np.asarray(column.altitude, dtype=np.float64)
    rho = np.asarray(column.density, dtype=np.float64)
    east = np.asarray(column.eastward, dtype=np.float64)
    north = np.asarray(column.northward, dtype=np.float64)

    lo, hi = band
    in_band = (altitude >= lo) & (altitude <= hi)
    if not np.any(in_band):
        # Degenerate band (e.g. a coarse profile): fall back to the whole column.
        in_band = np.ones_like(altitude, dtype=bool)

    weights = rho[in_band]
    total = float(np.sum(weights))
    if not (np.isfinite(total) and total > 0.0):
        return float(np.mean(east[in_band])), float(np.mean(north[in_band]))
    e = float(np.sum(weights * east[in_band]) / total)
    n = float(np.sum(weights * north[in_band]) / total)
    return e, n


def sample_era5_winds(
    ensemble: WindEnsemble,
    n_replicates: int,
    rng: np.random.Generator,
    band: tuple[float, float] = DEFAULT_BAND,
) -> _FloatArray:
    """Draw ``n_replicates`` representative winds from the ensemble, ``(n, 2)``.

    Each replicate is a whole real profile drawn with replacement and reduced
    by :func:`representative_wind`, so the draws carry the ensemble's true
    anisotropy and magnitude — not an assumed isotropic sigma.
    """
    if len(ensemble) == 0:
        raise ValueError("cannot sample winds from an empty ensemble")
    out = np.empty((n_replicates, 2), dtype=np.float64)
    for i in range(n_replicates):
        out[i] = representative_wind(ensemble.sample(rng), band)
    return out


def _interp_rows(
    grid: _FloatArray, values: _FloatArray, query: _FloatArray, xp: Any = np
) -> _FloatArray:
    """Per-row linear interpolation on a shared ascending ``grid``.

    ``grid`` is ``(L,)`` ascending, ``values`` is ``(n, L)``, ``query`` is
    ``(n,)``; returns ``(n,)`` with each row interpolated at its own altitude.
    Vectorised because the grid is shared across replicates. ``xp`` selects the
    array backend (NumPy or CuPy), so the same interpolation runs on either
    device.
    """
    idx = xp.clip(xp.searchsorted(grid, query) - 1, 0, grid.size - 2)
    z0 = grid[idx]
    z1 = grid[idx + 1]
    frac = xp.clip((query - z0) / (z1 - z0), 0.0, 1.0)
    v0 = xp.take_along_axis(values, idx[:, None], axis=1)[:, 0]
    v1 = xp.take_along_axis(values, (idx + 1)[:, None], axis=1)[:, 0]
    return v0 + frac * (v1 - v0)  # type: ignore[no-any-return]  # backend array


@dataclass(frozen=True)
class ERA5EntryDispersion:
    """Full-fidelity entry dispersion flown through the measured atmosphere.

    Every launch/vehicle dispersion is inherited from the generic
    :class:`aether.batch.entry_demo.EntryDispersionModel` (ballistic
    coefficient, residual density bias, launch speed and angles). What differs
    is the atmosphere: rather than an exponential density and a single scalar
    wind, **each replicate flies through its own real ERA5 profile** — the
    eastward and northward wind *and* the air density are interpolated at the
    vehicle's altitude at every integration step, with altitude-varying
    gravity. Above the ERA5 ceiling, where there is no measured wind, density
    follows the US Standard 1976 atmosphere and the wind is zero (the absence
    of data, not an assumed value). This is the research-grade Monte Carlo over
    measured soundings, not a reduced scalar-wind surrogate.

    ``grid_points`` and ``grid_ceiling`` set the shared vertical grid the
    profiles are resolved onto; the default 400 points to 80 km resolves the
    troposphere and stratosphere finely.
    """

    base: EntryDispersionModel
    ensemble: WindEnsemble
    grid_points: int = 400
    grid_ceiling: float = 80.0e3

    def _non_wind_specs(self) -> list[DispersionSpec]:
        return [s for s in self.base.specs() if not s.name.startswith("wind_")]

    def _profile_fields(
        self, n_replicates: int, rng: np.random.Generator
    ) -> tuple[_FloatArray, _FloatArray, _FloatArray, _FloatArray]:
        """Sampled profiles resolved onto the shared grid.

        Returns ``(grid, u_field, v_field, rho_field)`` with ``grid`` ascending ``(L,)`` and
        ``u_field, v_field, rho_field`` each ``(n, L)``: eastward wind, northward wind, and air
        density at every grid altitude for every replicate. Below a profile's
        lowest level the surface value is held; above its top the wind is zero
        and the density is US Standard 1976.
        """
        grid = np.linspace(0.0, self.grid_ceiling, self.grid_points)
        std_rho = np.asarray(earth_atmosphere().state(grid).density, dtype=np.float64)

        u_field = np.empty((n_replicates, grid.size))
        v_field = np.empty((n_replicates, grid.size))
        rho_field = np.empty((n_replicates, grid.size))
        for i in range(n_replicates):
            col = self.ensemble.sample(rng)
            alt = np.asarray(col.altitude, dtype=np.float64)
            top = float(alt[-1])
            aloft = grid > top
            # Wind: measured within the column, zero above its top (no data).
            u_field[i] = np.where(aloft, 0.0, np.interp(grid, alt, np.asarray(col.eastward)))
            v_field[i] = np.where(aloft, 0.0, np.interp(grid, alt, np.asarray(col.northward)))
            # Density: measured within the column, standard atmosphere above.
            rho_meas = np.interp(grid, alt, np.asarray(col.density, dtype=np.float64))
            rho_field[i] = np.where(aloft, std_rho, rho_meas)
        return grid, u_field, v_field, rho_field

    def fly(
        self,
        n_replicates: int,
        seed: int,
        dt: float = 0.05,
        max_time: float = 300.0,
        backend: Backend = "numpy",
    ) -> _FloatArray:
        """Landing points ``(n_replicates, 2)`` in the local tangent plane (m).

        RK4 in a 3-DOF point-mass frame with quadratic drag, the drag density
        and wind taken from the replicate's measured profile at its current
        altitude, and gravity from the US Standard 1976 model. ``backend``
        selects NumPy (CPU) or CuPy (CUDA): the whole ensemble integrates as one
        array, so a large Monte Carlo runs on the GPU with no change of physics.
        The result is always returned on the host.
        """
        if dt <= 0.0 or max_time <= dt:
            raise ValueError(f"need 0 < dt < max_time, got dt={dt}, max_time={max_time}")
        xp = get_array_module(backend)
        rng = np.random.default_rng(seed)
        params = sample_dispersions(self._non_wind_specs(), n_replicates, seed)
        y = xp.asarray(self.base.initial_states(params).astype(np.float64))
        beta = xp.asarray(np.asarray(params["beta"], dtype=np.float64))
        rho_bias = xp.asarray(np.asarray(params["density_bias"], dtype=np.float64))

        grid_h, u_h, v_h, rho_h = self._profile_fields(n_replicates, rng)
        # Gravity depends only on altitude; precompute it on the shared grid so
        # the device path never calls the host gravity model inside the loop.
        g_grid = xp.asarray(np.asarray(gravity(grid_h), dtype=np.float64))
        grid = xp.asarray(grid_h)
        u_field = xp.asarray(u_h)
        v_field = xp.asarray(v_h)
        rho_field = xp.asarray(rho_h)
        top = float(grid_h[-1])

        def rhs(state: _FloatArray) -> _FloatArray:
            z = xp.clip(state[:, 2], 0.0, top)
            # Per-replicate atmosphere at the current altitude.
            wx = _interp_rows(grid, u_field, z, xp)
            wy = _interp_rows(grid, v_field, z, xp)
            rho = rho_bias * _interp_rows(grid, rho_field, z, xp)
            wind = xp.stack([wx, wy, xp.zeros_like(wx)], axis=1)
            vel = state[:, 3:6]
            v_air = vel - wind
            v_mag = xp.sqrt(xp.sum(v_air * v_air, axis=1))
            accel = -0.5 * rho[:, None] * v_mag[:, None] * v_air / beta[:, None]
            accel[:, 2] -= xp.interp(z, grid, g_grid)
            out = xp.empty_like(state)
            out[:, 0:3] = vel
            out[:, 3:6] = accel
            return out  # type: ignore[no-any-return]  # backend array (numpy|cupy)

        impact = xp.zeros((n_replicates, 2), dtype=xp.float64)
        landed = xp.zeros(n_replicates, dtype=bool)
        n_steps = round(max_time / dt)
        for _ in range(n_steps):
            k1 = rhs(y)
            k2 = rhs(y + 0.5 * dt * k1)
            k3 = rhs(y + 0.5 * dt * k2)
            k4 = rhs(y + dt * k3)
            y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            now_down = (y[:, 2] <= 0.0) & (~landed)
            if bool(xp.any(now_down)):
                impact[now_down] = y[now_down, 0:2]
                landed[now_down] = True
            if bool(xp.all(landed)):
                break
        still = ~landed
        if bool(xp.any(still)):
            impact[still] = y[still, 0:2]
        return to_numpy(impact)


def terminal_dispersion(
    model: EntryDispersionModel,
    *,
    ensemble: WindEnsemble | None = None,
    era5_root: str | Path | None = None,
    site: tuple[float, float] | None = None,
    n_replicates: int = 4000,
    seed: int = 0,
    grid_points: int = 400,
    grid_ceiling: float = 80.0e3,
    dt: float = 0.05,
    max_time: float = 300.0,
) -> DispersionReport:
    """CEP / R95 for an entry dispersion, from *measured* ERA5 winds.

    The wind is resolved from an explicit ``ensemble``, or from ``era5_root``
    at ``site`` (lat, lon). **There is no approximation fallback:** if no real
    wind ensemble resolves, this raises rather than silently substituting an
    assumed-wind model, because a CEP computed against a guessed wind is a
    dispersion figure that verifies nothing. A caller who explicitly
    wants the generic isotropic model can call ``model.fly`` directly.

    The landing scatter is reduced to R95 and CEP by
    :func:`aether.batch.dispersion.summarize_dispersion`.
    """
    if ensemble is None:
        if era5_root is None or site is None:
            raise ValueError(
                "terminal_dispersion needs a real wind ensemble: pass `ensemble`, "
                "or `era5_root` and `site`. It does not fall back to an assumed "
                "isotropic wind — that would be a CEP against a guess. For the "
                "generic model call EntryDispersionModel.fly directly."
            )
        # Let load errors propagate: a missing archive or backend is a hard
        # failure here, not a cue to approximate.
        ensemble = load_site_ensemble(era5_root, site[0], site[1])

    impacts = ERA5EntryDispersion(
        model, ensemble, grid_points=grid_points, grid_ceiling=grid_ceiling
    ).fly(n_replicates, seed, dt=dt, max_time=max_time)
    return summarize_dispersion(impacts, seed=seed)


def load_site_ensemble(
    era5_root: str | Path, latitude: float, longitude: float
) -> WindEnsemble:
    """Build a wind ensemble at a site from an ERA5 archive directory.

    Walks the ``<root>/<year>/<month>/...`` layout (and any flat NetCDF files
    at the root), reads each analysis file at the site, and assembles a
    :class:`~aether.atmosphere.era5.WindEnsemble`. Files that cannot be read
    are skipped; if none can be read the caller gets a
    :class:`FileNotFoundError`.

    Reading NetCDF requires an xarray NetCDF backend (``netcdf4`` or
    ``h5netcdf``); GRIB requires ``cfgrib``. When neither the data nor a
    backend is present the caller should fall back to the generic model — which
    :func:`terminal_dispersion` does automatically.
    """
    from aether.atmosphere.era5 import ERA5Box, WindEnsemble

    root = Path(era5_root)
    if not root.is_dir():
        raise FileNotFoundError(f"no ERA5 archive at {root}")

    files = sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in (".nc", ".grib", ".grib2", ".grb")
        # Skip hidden directories (e.g. a download tool's `.venv`), whose
        # bundled sample NetCDFs are not ERA5 and would derail the walk.
        and not any(part.startswith(".") for part in p.relative_to(root).parts)
    )
    if not files:
        raise FileNotFoundError(f"no ERA5 data files under {root}")

    columns: list[ERA5Column] = []
    for path in files:
        try:
            box = ERA5Box.from_file(path, latitude, longitude)  # type: ignore[attr-defined]
        except AttributeError:
            try:
                box = _box_from_dataset(path, latitude, longitude)
            except (OSError, ValueError, KeyError):
                # A NetCDF that is not an ERA5 field (no recognisable coords) is
                # skipped, not fatal.
                continue
        except (OSError, ValueError, KeyError):
            continue
        if box is None:
            continue
        for index in range(np.asarray(box.times).size):
            columns.append(box.column(latitude, longitude, index))

    if not columns:
        raise FileNotFoundError(
            f"no readable ERA5 columns under {root} (missing NetCDF/GRIB backend?)"
        )
    return WindEnsemble(columns=tuple(columns))


def load_file_ensemble(
    path: str | Path, latitude: float, longitude: float
) -> WindEnsemble:
    """Build a wind ensemble at a site from a *single* ERA5 file.

    One combined ERA5 analysis file already carries several six-hourly profiles
    at the site (one per analysis time), which is enough diversity for a
    terminal-dispersion Monte Carlo without walking the whole archive. Only the
    grid window bracketing the site is read, so a single global file resolves in
    a fraction of a second rather than loading ~1 GB. Raises if the file cannot
    be read as an ERA5 field.
    """
    from aether.atmosphere.era5 import WindEnsemble

    box = _box_from_dataset(Path(path), float(latitude), float(longitude))
    if box is None:
        raise FileNotFoundError(f"{path} is not a readable ERA5 field")
    columns = [
        box.column(latitude, longitude, index)
        for index in range(np.asarray(box.times).size)
    ]
    return WindEnsemble(columns=tuple(columns))


def latin_hypercube_wind_ensemble(
    era5_root: str | Path,
    latitude: float,
    longitude: float,
    n_conditions: int = 16,
    seed: int = 0,
    pattern: str = "era5_combined_6hr_*.nc",
) -> WindEnsemble:
    """A stratified wind ensemble: Latin-hypercube launch epochs over the archive.

    A full terminal-dispersion Monte Carlo must not sample one day's weather — it
    must span the archive's conditions (season, synoptic state, diurnal cycle).
    The launch epochs are chosen by Latin-hypercube sampling over the two-
    dimensional (date, time-of-day) domain the archive spans, so ``n_conditions``
    soundings are well spread with far fewer draws than a full grid. Each sampled
    epoch maps to one analysis time in one daily file; the files that any sample
    lands in are each loaded once (windowed to the site), and the sampled columns
    become the ensemble the replicate Monte Carlo draws its wind from.
    """
    import scipy.stats.qmc as qmc

    from aether.atmosphere.era5 import WindEnsemble

    root = Path(era5_root)
    files = sorted(root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files matching {pattern!r} under {root}")

    unit = qmc.LatinHypercube(d=2, seed=seed).random(int(n_conditions))
    file_idx = np.minimum((unit[:, 0] * len(files)).astype(int), len(files) - 1)

    columns: list[ERA5Column] = []
    for fi in np.unique(file_idx):
        box = _box_from_dataset(files[int(fi)], float(latitude), float(longitude))
        if box is None:
            continue
        n_times = int(np.asarray(box.times).size)
        sel = unit[file_idx == fi, 1]
        time_idx = np.minimum((sel * n_times).astype(int), n_times - 1)
        for ti in time_idx:
            columns.append(box.column(latitude, longitude, int(ti)))
    if not columns:
        raise FileNotFoundError(f"no readable ERA5 columns under {root}")
    return WindEnsemble(columns=tuple(columns))


def _crop_to_site(
    ds: Any,
    lat_name: str,
    lon_name: str,
    latitude: float,
    longitude: float,
    margin_deg: float = 1.5,
) -> Any:
    """Select the grid window bracketing a site, so only it is read from disk."""
    lat_vals = np.asarray(ds[lat_name].values, dtype=np.float64)
    lon_vals = np.asarray(ds[lon_name].values, dtype=np.float64) % 360.0
    lon0 = float(longitude) % 360.0
    lat_mask = np.abs(lat_vals - float(latitude)) <= margin_deg
    dlon = np.abs((lon_vals - lon0 + 180.0) % 360.0 - 180.0)
    lon_mask = dlon <= margin_deg
    # Guarantee at least the two points that bracket the site in each axis.
    if lat_mask.sum() < 2:
        lat_mask = np.zeros_like(lat_mask)
        lat_mask[np.argsort(np.abs(lat_vals - float(latitude)))[:2]] = True
    if lon_mask.sum() < 2:
        lon_mask = np.zeros_like(lon_mask)
        lon_mask[np.argsort(dlon)[:2]] = True
    return ds.isel({
        lat_name: np.where(lat_mask)[0],
        lon_name: np.where(lon_mask)[0],
    })


def _box_from_dataset(
    path: Path, latitude: float, longitude: float
) -> ERA5Box | None:
    """Read one ERA5 NetCDF/GRIB file into an :class:`ERA5Box` via xarray.

    Kept separate so the file-format handling is the only part that needs the
    optional IO backends; the rest of the module runs on in-memory ensembles.
    """
    import xarray as xr

    from aether.atmosphere.era5 import ERA5Box

    ds = xr.open_dataset(path)
    try:
        names = {str(v).lower(): str(v) for v in list(ds.data_vars) + list(ds.coords)}

        def pick(*cands: str) -> str:
            for c in cands:
                if c in names:
                    return names[c]
            raise KeyError(cands)

        # Crop to a small window around the site *before* reading any field.
        # ERA5 analyses are global (721x1440 per level); a dispersion
        # ensemble needs only the handful of grid points bracketing the site,
        # so a windowed read is thousands of times less I/O than pulling the
        # whole field and taking one column afterwards.
        lat_name = pick("latitude", "lat")
        lon_name = pick("longitude", "lon")
        ds = _crop_to_site(ds, lat_name, lon_name, latitude, longitude, margin_deg=1.5)

        lat = np.asarray(ds[pick("latitude", "lat")].values, dtype=np.float64)
        lon = np.asarray(ds[pick("longitude", "lon")].values, dtype=np.float64)
        level = np.asarray(
            ds[pick("isobaricinhpa", "level", "pressure_level", "plev")].values,
            dtype=np.float64,
        )
        pressure = level * (100.0 if float(np.max(level)) < 2000.0 else 1.0)
        time = np.atleast_1d(ds[pick("time", "valid_time")].values).astype("datetime64[s]")

        def field(*cands: str) -> _FloatArray:
            arr = np.asarray(ds[pick(*cands)].values, dtype=np.float64)
            # Coerce to (time, level, lat, lon).
            while arr.ndim < 4:
                arr = arr[np.newaxis, ...]
            return arr

        # Order levels by increasing altitude (decreasing pressure).
        order = np.argsort(pressure)[::-1]
        return ERA5Box(
            times=time,
            latitude=lat,
            longitude=lon % 360.0,
            pressure=pressure[order],
            temperature=field("t", "temperature")[:, order],
            eastward=field("u", "eastward", "u_component_of_wind")[:, order],
            northward=field("v", "northward", "v_component_of_wind")[:, order],
            specific_humidity=field("q", "specific_humidity")[:, order],
            source=str(path.name),
        )
    finally:
        ds.close()
