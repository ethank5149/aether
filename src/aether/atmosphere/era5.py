"""ERA5 reanalysis as a measured atmosphere and a wind-dispersion database.

``reference/ERA5`` holds a year of global ECMWF reanalysis on 21 pressure
levels from 1000 to 1 hPa — roughly the ground to 48 km — six-hourly at a
quarter degree. Two quite different things can be done with it, and this
module supports both:

**One atmosphere.** A single column at the launch site on a given date gives
the temperature and wind the vehicle actually flew through, in place of a
standard atmosphere and still air. That is the deterministic use.

**Fourteen hundred atmospheres.** The same column across every analysis time
in the archive is an empirical distribution of wind profiles at that site —
which is the input a launch-vehicle load Monte Carlo needs, and is the thing
a synthetic gust model is a poor substitute for. Real profiles carry the
correlation between shear layers that an independently-sampled envelope
does not, and it is that correlation, not the peak wind, that decides
whether two shear layers add or cancel in :math:`q\\alpha`.

Cost and caching
----------------

The GRIBs are 20 GB a month and a variable is stored as one message per
(time, level), so pulling a single grid point still reads the whole global
field for that message. Extraction is therefore a **bulk, cached** operation:
one pass over a month writes a small ``.npz`` holding a lat/lon box across
all times, and every later query is served from that. Extracting a box rather
than a point costs nothing extra — the message was read regardless — and
buys horizontal interpolation for free.

Altitude
--------

The download did not include geopotential, so altitude is reconstructed
hydrostatically from the levels themselves:

.. math::

    \\Delta z = \\frac{R_d \\overline{T_v}}{g_0}\\,\\ln\\frac{p_k}{p_{k+1}},
    \\qquad T_v = T\\,(1 + 0.6077\\,q)

integrated upward from the 1000 hPa surface, whose height is taken from the
1976 standard. The virtual temperature matters near the ground — humid air is
lighter, and ignoring it biases the thickness of the lowest layers by a few
tens of metres — and the anchor does not: the true 1000 hPa height moves by
of order 100 m with the weather, and a 100 m offset applied to a wind profile
whose shear scales are kilometres changes nothing that matters. Both the
approximation and its size are stated on
:attr:`ERA5Column.altitude_uncertainty`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.optimize
from numpy.typing import NDArray

from aether.atmosphere.standard import (
    SEA_LEVEL_GRAVITY,
    USStandard1976,
    geometric_altitude,
)
from aether.atmosphere.wind import TabulatedWind

__all__ = [
    "DRY_AIR_GAS_CONSTANT",
    "ERA5Box",
    "ERA5Column",
    "WindEnsemble",
    "extract_box",
    "reference_level_altitude",
]

_FloatArray = NDArray[np.float64]

#: :math:`R_d` for dry air, J kg⁻¹ K⁻¹.
DRY_AIR_GAS_CONSTANT = 287.0528
#: Coefficient in :math:`T_v = T(1 + \epsilon q)` for specific humidity.
VIRTUAL_TEMPERATURE_COEFFICIENT = 0.6077


def reference_level_altitude(pressure: float = 100000.0) -> float:
    """Geopotential altitude of a pressure level in the 1976 standard (m).

    Used only to anchor the hydrostatic integration; see the module note on
    why the choice of anchor is not load-bearing.
    """
    standard = USStandard1976()

    def residual(geometric: float) -> float:
        return float(standard.pressure(geometric)) - float(pressure)

    return float(scipy.optimize.brentq(residual, -4.9e3, 80.0e3, xtol=1e-6))


@dataclass(frozen=True)
class ERA5Column:
    """One vertical profile: what the air was doing at a place and a time.

    Attributes
    ----------
    pressure:
        Level pressures (Pa), increasing altitude — so *decreasing* pressure.
    temperature:
        Air temperature (K).
    eastward, northward:
        Wind components (m/s).
    specific_humidity:
        kg of water vapour per kg of moist air.
    """

    latitude: float
    longitude: float
    epoch: str
    pressure: _FloatArray
    temperature: _FloatArray
    eastward: _FloatArray
    northward: _FloatArray
    specific_humidity: _FloatArray

    #: Stated absolute uncertainty of the reconstructed altitude (m).
    altitude_uncertainty: float = 150.0

    @property
    def virtual_temperature(self) -> _FloatArray:
        """:math:`T_v = T(1 + 0.6077q)` (K) — the density-correct temperature."""
        return np.asarray(
            self.temperature * (1.0 + VIRTUAL_TEMPERATURE_COEFFICIENT * self.specific_humidity)
        )

    @property
    def altitude(self) -> _FloatArray:
        """Geometric altitude of each level (m), hydrostatically reconstructed."""
        pressure = np.asarray(self.pressure, dtype=np.float64)
        virtual = self.virtual_temperature
        # Layer-mean virtual temperature; the logarithmic mean is the exact
        # one for an isothermal-in-log-p layer, but the arithmetic mean is
        # within 1e-4 of it over a 2:1 pressure ratio and does not need a
        # guard when two levels have equal temperature.
        layer_mean = 0.5 * (virtual[:-1] + virtual[1:])
        thickness = (
            DRY_AIR_GAS_CONSTANT
            * layer_mean
            / SEA_LEVEL_GRAVITY
            * np.log(pressure[:-1] / pressure[1:])
        )
        geopotential = np.concatenate(
            [[reference_level_altitude(float(pressure[0]))], np.cumsum(thickness)]
        )
        geopotential[1:] += geopotential[0]
        return geometric_altitude(geopotential)

    @property
    def density(self) -> _FloatArray:
        """Air density (kg/m³) from the gas law with virtual temperature."""
        return np.asarray(self.pressure / (DRY_AIR_GAS_CONSTANT * self.virtual_temperature))

    def wind(self, ceiling: float = 60.0e3) -> TabulatedWind:
        """The wind profile as a :class:`~aether.atmosphere.wind.TabulatedWind`."""
        return TabulatedWind(
            altitude=self.altitude,
            east=np.asarray(self.eastward),
            north=np.asarray(self.northward),
            ceiling=ceiling,
            name=f"ERA5 {self.epoch} at {self.latitude:.2f}, {self.longitude:.2f}",
        )

    @property
    def wind_speed(self) -> _FloatArray:
        return np.asarray(np.hypot(self.eastward, self.northward))


@dataclass(frozen=True)
class ERA5Box:
    """A cached lat/lon/time/level block of reanalysis.

    This is what one pass over a monthly GRIB produces. Everything else —
    a column at a site, an ensemble over a month — is a cheap slice of it.
    """

    times: NDArray[np.datetime64]
    latitude: _FloatArray
    longitude: _FloatArray
    pressure: _FloatArray
    """Level pressures (Pa), ordered by increasing altitude."""
    temperature: _FloatArray
    """Shape ``(time, level, lat, lon)``."""
    eastward: _FloatArray
    northward: _FloatArray
    specific_humidity: _FloatArray
    source: str = ""

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(np.asarray(self.temperature).shape)

    def column(self, latitude: float, longitude: float, index: int = 0) -> ERA5Column:
        """Bilinearly interpolated column at a site and one analysis time."""
        weights_lat, rows = _interpolation_weights(np.asarray(self.latitude), float(latitude))
        weights_lon, columns = _interpolation_weights(
            np.asarray(self.longitude), float(longitude) % 360.0
        )

        def sample(field: _FloatArray) -> _FloatArray:
            block = np.asarray(field)[index][:, rows][:, :, columns]
            return np.asarray(np.einsum("lij,i,j->l", block, weights_lat, weights_lon))

        return ERA5Column(
            latitude=float(latitude),
            longitude=float(longitude),
            epoch=str(np.asarray(self.times)[index]),
            pressure=np.asarray(self.pressure),
            temperature=sample(self.temperature),
            eastward=sample(self.eastward),
            northward=sample(self.northward),
            specific_humidity=sample(self.specific_humidity),
        )

    def ensemble(self, latitude: float, longitude: float) -> WindEnsemble:
        """Every analysis time in the box as a wind ensemble at one site."""
        columns = [
            self.column(latitude, longitude, index) for index in range(np.asarray(self.times).size)
        ]
        return WindEnsemble(columns=tuple(columns))

    def save(self, path: str | Path) -> Path:
        """Write the box to a compressed ``.npz``."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            times=np.asarray(self.times).astype("datetime64[s]").astype(np.int64),
            latitude=self.latitude,
            longitude=self.longitude,
            pressure=self.pressure,
            temperature=self.temperature,
            eastward=self.eastward,
            northward=self.northward,
            specific_humidity=self.specific_humidity,
            source=np.array(self.source),
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> ERA5Box:
        """Read a box written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                times=np.asarray(data["times"]).astype("datetime64[s]"),
                latitude=np.asarray(data["latitude"], dtype=np.float64),
                longitude=np.asarray(data["longitude"], dtype=np.float64),
                pressure=np.asarray(data["pressure"], dtype=np.float64),
                temperature=np.asarray(data["temperature"], dtype=np.float64),
                eastward=np.asarray(data["eastward"], dtype=np.float64),
                northward=np.asarray(data["northward"], dtype=np.float64),
                specific_humidity=np.asarray(data["specific_humidity"], dtype=np.float64),
                source=str(data["source"]),
            )


@dataclass(frozen=True)
class WindEnsemble:
    """A set of real wind profiles at one site — the dispersion input.

    The point of holding whole profiles rather than a mean and a standard
    deviation is that the *shape* is correlated. A launch load Monte Carlo
    that samples the wind at 8 km independently of the wind at 12 km will
    generate shear layers that do not occur and miss the ones that do.
    """

    columns: tuple[ERA5Column, ...]

    def __post_init__(self) -> None:
        if not self.columns:
            msg = "a wind ensemble needs at least one profile"
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.columns)

    @property
    def altitude(self) -> _FloatArray:
        """Altitude grid of the first profile (m).

        The levels are isobaric, so the geometric altitudes differ slightly
        between profiles — by up to a few hundred metres in the stratosphere
        as the column warms and cools. Statistics are taken on a common grid,
        which is this one, and the others are interpolated onto it.
        """
        return self.columns[0].altitude

    def speeds(self) -> _FloatArray:
        """Wind speed of every profile on the common grid, ``(n_profiles, n_levels)``."""
        grid = self.altitude
        return np.asarray(
            [np.interp(grid, column.altitude, column.wind_speed) for column in self.columns]
        )

    def components(self) -> tuple[_FloatArray, _FloatArray]:
        """Eastward and northward components on the common grid."""
        grid = self.altitude
        east = np.asarray([np.interp(grid, c.altitude, c.eastward) for c in self.columns])
        north = np.asarray([np.interp(grid, c.altitude, c.northward) for c in self.columns])
        return east, north

    def percentile(self, fraction: float) -> _FloatArray:
        """Wind speed at a given percentile of the ensemble, per level (m/s)."""
        return np.asarray(np.percentile(self.speeds(), float(fraction), axis=0))

    def worst(self, altitude_band: tuple[float, float] = (8.0e3, 16.0e3)) -> ERA5Column:
        """The profile with the largest mean wind in a band — the load case.

        Defaults to 8 to 16 km, which is the jet-stream layer and, for a
        vertically launched vehicle, very nearly where maximum dynamic
        pressure occurs. The worst day in the archive is a more defensible
        design wind than a synthetic envelope because it happened.
        """
        low, high = float(altitude_band[0]), float(altitude_band[1])
        scores = []
        for column in self.columns:
            z = column.altitude
            inside = (z >= low) & (z <= high)
            scores.append(float(np.mean(column.wind_speed[inside])) if np.any(inside) else 0.0)
        return self.columns[int(np.argmax(scores))]

    def sample(self, rng: np.random.Generator) -> ERA5Column:
        """Draw one profile at random — a Monte Carlo trial's atmosphere."""
        return self.columns[int(rng.integers(len(self.columns)))]


def _interpolation_weights(axis: _FloatArray, value: float) -> tuple[_FloatArray, NDArray[np.intp]]:
    """Linear interpolation weights and indices on a possibly descending axis."""
    ascending = axis if axis[0] <= axis[-1] else axis[::-1]
    order = np.arange(axis.size) if axis[0] <= axis[-1] else np.arange(axis.size)[::-1]
    position = float(np.clip(value, ascending[0], ascending[-1]))
    upper = int(np.searchsorted(ascending, position, side="left"))
    upper = int(np.clip(upper, 1, ascending.size - 1))
    lower = upper - 1
    span = ascending[upper] - ascending[lower]
    fraction = 0.0 if span == 0.0 else (position - ascending[lower]) / span
    indices = np.asarray([order[lower], order[upper]], dtype=np.intp)
    return np.asarray([1.0 - fraction, fraction]), indices


def extract_box(
    grib: str | Path,
    latitude: float,
    longitude: float,
    half_width: float = 2.0,
    cache: str | Path | None = None,
    time_stride: int = 1,
) -> ERA5Box:
    """Pull a lat/lon box across all analysis times, once, and cache it.

    Parameters
    ----------
    grib:
        A monthly ERA5 pressure-level GRIB.
    latitude, longitude:
        Centre of the box, degrees. Longitude may be signed or 0-360.
    half_width:
        Box half-size in degrees. Two degrees is about 220 km, which spans
        the downrange leg of an ascent.
    cache:
        Where to write the ``.npz``. Defaults to a name derived from the GRIB
        and the box, alongside the GRIB.
    time_stride:
        Take every ``n``-th analysis time. The archive is six-hourly, so a
        stride of 4 is daily — enough for a climatology, a quarter of the
        read.

    Notes
    -----
    Reads the GRIB only if the cache is absent. Building ``cfgrib``'s index
    over a 20 GB monthly file takes about three minutes before any data
    moves, which is the reason this is a bulk operation with a cache and not
    a lookup.
    """
    source = Path(grib)
    if not source.exists():
        msg = f"no such GRIB: {source}"
        raise FileNotFoundError(msg)

    destination = (
        Path(cache)
        if cache is not None
        else _cache_path(source, latitude, longitude, half_width, time_stride)
    )
    if destination.exists():
        return ERA5Box.load(destination)

    try:
        import xarray as xr
    except ImportError as error:  # pragma: no cover - dependency is declared
        msg = "reading ERA5 GRIB needs xarray and cfgrib (pip install cfgrib xarray)"
        raise ImportError(msg) from error

    dataset: Any = xr.open_dataset(source, engine="cfgrib", backend_kwargs={"indexpath": ""})
    try:
        centre = float(longitude) % 360.0
        lat_slice = slice(latitude + half_width, latitude - half_width)  # descending
        lon_slice = slice(centre - half_width, centre + half_width)
        window = dataset.sel(latitude=lat_slice, longitude=lon_slice)
        window = window.isel(time=slice(None, None, int(time_stride)))

        # Levels arrive 1000 hPa first; reverse so everything downstream is
        # ordered by increasing altitude, which is what the interpolators and
        # the hydrostatic integration both assume.
        pressure_hpa = np.asarray(window["isobaricInhPa"].values, dtype=np.float64)
        order = np.argsort(-pressure_hpa)

        def pull(name: str) -> _FloatArray:
            return np.asarray(window[name].values, dtype=np.float64)[:, order]

        box = ERA5Box(
            times=np.asarray(window["time"].values).astype("datetime64[s]"),
            latitude=np.asarray(window["latitude"].values, dtype=np.float64),
            longitude=np.asarray(window["longitude"].values, dtype=np.float64),
            pressure=pressure_hpa[order] * 100.0,
            temperature=pull("t"),
            eastward=pull("u"),
            northward=pull("v"),
            specific_humidity=pull("q"),
            source=source.name,
        )
    finally:
        dataset.close()
    box.save(destination)
    return box


def _cache_path(
    source: Path, latitude: float, longitude: float, half_width: float, stride: int
) -> Path:
    key = f"{latitude:.4f}|{longitude:.4f}|{half_width:.4f}|{stride}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:10]
    return source.with_name(f"{source.stem}-box-{digest}.npz")
