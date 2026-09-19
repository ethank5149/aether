"""GMTED2010 elevation, so the ground is where the ground is.

Two things in this framework quietly assumed the Earth's surface is at zero
altitude: the impact point of a ballistic arc, and the launch site it left
from. Neither is true anywhere interesting. The Dombarovsky pad coordinates
come back at **348 m** from this archive, and a trajectory terminated at the
ellipsoid rather than at the terrain arrives late, low and slightly
downrange of where it actually would.

``reference/GMTED2010`` is the USGS/NGA global elevation model: 96 tiles at
**7.5 arc-seconds** — about 230 m at the equator — covering 70 S to 90 N,
plus twelve 30-arc-second tiles for Antarctica. Each 7.5-second tile is
14400 x 9600 int16 over a 30 x 20 degree box, uncompressed, which is 276 MB
apiece and 26 GB for the mean-elevation product alone.

So nothing is loaded globally. :meth:`Terrain.elevation` **groups its query
points by tile and issues one windowed read per tile**, covering only the
bounding box of the points that fall in it. A ground track crosses two or
three tiles, so a trajectory's whole elevation profile is two or three reads
of a few megabytes each — not one read per sample, and not 26 GB.

Which product, and why it matters
---------------------------------

Each tile ships six statistics: ``mea`` (mean), ``med``, ``min``, ``max``,
``std`` and ``dsc`` (systematic subsample). This uses **mean** by default.
For an impact point that is the right choice — it is the average ground
level over the cell, which is what a footprint sits on. For a *terrain
clearance* question ``max`` is the right one, because a vehicle clears the
highest ground in a cell rather than the average, so the product is a
parameter rather than a constant.

At 7.5 arc-seconds the statistics are close together, because the cell is
already near the source resolution: measured at Everest, ``mea`` gives
8646 m, ``max`` 8700 m and ``min`` 8585 m against a true summit of 8848 m.
The spread between products is 115 m and the gap to the summit is 202 m, so
**neither the choice of product nor the model is what limits a peak
elevation** — cell size is, and no statistic recovers a summit narrower than
its cell.

Voids
-----

The nodata value is -32768 and it means ocean or unmapped, not "sea level".
It is mapped to zero, and :attr:`ElevationSample.void_fraction` reports how
much of a query landed there — because an elevation profile that is 90 %
filled-in voids and one that is 90 % measured look identical otherwise.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "PRODUCTS",
    "ElevationSample",
    "ReliefMap",
    "Terrain",
    "TerrainProbe",
    "default_terrain",
]

_FloatArray = NDArray[np.float64]

#: The six GMTED2010 statistics, by their filename token.
PRODUCTS = ("mea", "med", "min", "max", "std", "dsc")

#: Sentinel for ocean and unmapped cells.
NODATA = -32768

#: Relief maps already built this process, keyed by archive, product,
#: resolution and exaggeration. See :meth:`Terrain.relief`.
_RELIEFS: dict[tuple[str, str, int, float], ReliefMap] = {}


@dataclass(frozen=True)
class _Tile:
    """One GMTED2010 tile and the box it covers."""

    path: Path
    south: float
    west: float
    height_degrees: float
    width_degrees: float

    @property
    def north(self) -> float:
        return self.south + self.height_degrees

    @property
    def east(self) -> float:
        return self.west + self.width_degrees

    def contains(self, latitude: _FloatArray, longitude: _FloatArray) -> NDArray[np.bool_]:
        return np.asarray(
            (latitude >= self.south)
            & (latitude < self.north)
            & (longitude >= self.west)
            & (longitude < self.east)
        )


@dataclass(frozen=True)
class ElevationSample:
    """Elevations and how much of the query was actually measured."""

    elevation: _FloatArray
    """Metres above the ellipsoid's reference surface."""
    void: NDArray[np.bool_]
    """True where the source had no data — ocean or unmapped."""

    @property
    def void_fraction(self) -> float:
        return float(np.mean(self.void)) if self.void.size else 0.0


def _parse_directory(name: str) -> tuple[float, float, float] | None:
    """``(south, west, degrees_per_tile_lat)`` from a GMTED directory name."""
    match = re.fullmatch(r"GMTED2010([NS])(\d{2})([EW])(\d{3})(?:_(\d{3}))?", name)
    if match is None:
        return None
    ns, lat, ew, lon, _resolution = match.groups()
    south = float(lat) * (1.0 if ns == "N" else -1.0)
    west = float(lon) * (1.0 if ew == "E" else -1.0)
    # Every band spans 20 degrees of latitude and 30 of longitude, including
    # the Antarctic one, which differs only in being a 30-arc-second product.
    return south, west, 20.0


@dataclass
class Terrain:
    """The GMTED2010 archive, queried by latitude and longitude.

    Attributes
    ----------
    root:
        Directory of ``GMTED2010<band>`` subdirectories.
    product:
        Which of :data:`PRODUCTS` to read. See the module note on why this
        is a choice and not a constant.
    """

    root: Path
    product: str = "mea"
    _tiles: list[_Tile] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if self.product not in PRODUCTS:
            msg = f"product must be one of {PRODUCTS}, got {self.product!r}"
            raise ValueError(msg)
        if not self.root.is_dir():
            msg = (
                f"no GMTED2010 archive at {self.root}. Expected USGS GMTED2010 "
                f"tile directories named like GMTED2010N30E060_075."
            )
            raise FileNotFoundError(msg)
        for directory in sorted(self.root.iterdir()):
            if not directory.is_dir():
                continue
            parsed = _parse_directory(directory.name)
            if parsed is None:
                continue
            south, west, span = parsed
            matches = sorted(directory.glob(f"*_gmted_{self.product}*.tif"))
            if not matches:
                continue
            self._tiles.append(
                _Tile(matches[0], south=south, west=west, height_degrees=span, width_degrees=30.0)
            )
        if not self._tiles:
            msg = (
                f"no {self.product} tiles under {self.root}; the directory has "
                f"{len(list(self.root.iterdir()))} entries but none matched "
                f"*_gmted_{self.product}*.tif"
            )
            raise FileNotFoundError(msg)

    @property
    def n_tiles(self) -> int:
        return len(self._tiles)

    @property
    def coverage(self) -> tuple[float, float]:
        """``(south, north)`` latitude limits of the archive, degrees."""
        return (
            min(t.south for t in self._tiles),
            max(t.north for t in self._tiles),
        )

    def elevation(
        self, latitude: ArrayLike, longitude: ArrayLike, degrees: bool = True
    ) -> ElevationSample:
        """Ground elevation at the given points (m).

        One windowed read per tile touched, covering the bounding box of the
        points that fall in it, then bilinear interpolation. Points outside
        the archive's latitude coverage come back as zero and flagged void.
        """
        try:
            import rasterio
            from rasterio.windows import Window
        except ImportError as error:  # pragma: no cover - dependency declared
            msg = "reading GMTED2010 needs rasterio (pip install rasterio)"
            raise ImportError(msg) from error

        lat = np.asarray(latitude, dtype=np.float64)
        lon = np.asarray(longitude, dtype=np.float64)
        lat, lon = np.broadcast_arrays(lat, lon)
        # The output shape is the *broadcast* shape, taken before flattening,
        # so a scalar query returns a scalar rather than a one-element array.
        shape = lat.shape
        lat = np.atleast_1d(lat).ravel().copy()
        lon = np.atleast_1d(lon).ravel().copy()
        if not degrees:
            lat, lon = np.rad2deg(lat), np.rad2deg(lon)
        lon = (lon + 180.0) % 360.0 - 180.0

        out = np.zeros(lat.size)
        void = np.ones(lat.size, dtype=bool)

        for tile in self._tiles:
            inside = tile.contains(lat, lon) & void
            if not np.any(inside):
                continue
            with rasterio.open(tile.path) as handle:
                transform = handle.transform
                # Fractional pixel coordinates, from the file's own affine
                # rather than an assumed origin: GMTED tiles are offset from
                # the whole degree by half an arc-second and assuming a clean
                # corner misregisters every sample.
                inverse = ~transform
                columns, rows = inverse * (lon[inside], lat[inside])
                col0 = int(np.floor(np.min(columns)).item())
                col1 = int(np.ceil(np.max(columns)).item()) + 1
                row0 = int(np.floor(np.min(rows)).item())
                row1 = int(np.ceil(np.max(rows)).item()) + 1
                col0, row0 = max(col0, 0), max(row0, 0)
                col1 = min(col1, handle.width)
                row1 = min(row1, handle.height)
                if col1 <= col0 or row1 <= row0:  # pragma: no cover - clipped away
                    continue
                block = handle.read(1, window=Window(col0, row0, col1 - col0, row1 - row0)).astype(
                    np.float64
                )

            missing = block <= NODATA + 1
            block = np.where(missing, 0.0, block)
            # Minus a half cell, because ``~transform`` returns coordinates on
            # the pixel-*corner* lattice while the values it indexes are cell
            # averages centred half a cell inside it. Without this the
            # interpolation is a half cell — 115 m of ground at 7.5
            # arc-seconds — off its own data: queried at the exact centre of a
            # cell holding 8,641 m in the Khumbu it returned **8,680 m**, and
            # that bias was in every elevation this class has ever reported.
            # It stayed invisible against the published check points because
            # Denver and the Dead Sea shore are flat at 200 m scale, and it
            # only surfaced when a local patch built on the correct lattice
            # disagreed with it by 761 m.
            local_c = np.clip(columns - col0 - 0.5, 0.0, block.shape[1] - 1.000001)
            local_r = np.clip(rows - row0 - 0.5, 0.0, block.shape[0] - 1.000001)
            c0 = local_c.astype(np.int64)
            r0 = local_r.astype(np.int64)
            c1 = np.minimum(c0 + 1, block.shape[1] - 1)
            r1 = np.minimum(r0 + 1, block.shape[0] - 1)
            fc, fr = local_c - c0, local_r - r0
            top = block[r0, c0] * (1 - fc) + block[r0, c1] * fc
            bottom = block[r1, c0] * (1 - fc) + block[r1, c1] * fc
            out[inside] = top * (1 - fr) + bottom * fr
            # A sample is void only if every corner it drew from was void;
            # a coastline cell interpolating one land corner is real data.
            corners = missing[r0, c0] & missing[r0, c1] & missing[r1, c0] & missing[r1, c1]
            void[inside] = corners

        return ElevationSample(
            elevation=np.asarray(out.reshape(shape)),
            void=np.asarray(void.reshape(shape)),
        )

    def coarse(
        self, height: int = 2048, cache: Path | None = None, rebuild: bool = False
    ) -> _FloatArray:
        """A decimated global elevation grid, ``(height, 2*height)`` in metres.

        Row 0 is +90 latitude, column 0 is -180 longitude — the same
        convention as the imagery, so the two can be sampled with one set of
        indices. Used for relief shading on the globe, where per-pixel
        windowed reads of a 26 GB archive are not an option.
        """
        if height < 32 or height % 2 != 0:
            msg = f"coarse height must be even and at least 32, got {height}"
            raise ValueError(msg)
        destination = (
            Path(cache)
            if cache is not None
            else self.root / "_coarse" / f"gmted-{self.product}-{height}.npy"
        )
        if destination.is_file() and not rebuild:
            return np.asarray(np.load(destination))

        import rasterio

        width = 2 * height
        grid = np.zeros((height, width), dtype=np.float32)
        for tile in self._tiles:
            with rasterio.open(tile.path) as handle:
                rows = max(round(tile.height_degrees / 180.0 * height), 1)
                cols = max(round(tile.width_degrees / 360.0 * width), 1)
                block = handle.read(
                    1,
                    out_shape=(rows, cols),
                    resampling=rasterio.enums.Resampling.average,
                ).astype(np.float32)
            block = np.where(block <= NODATA + 1, 0.0, block)
            top = round((90.0 - tile.north) / 180.0 * height)
            left = round((tile.west + 180.0) / 360.0 * width)
            rows = min(rows, height - top)
            cols = min(cols, width - left)
            if rows > 0 and cols > 0:
                grid[top : top + rows, left : left + cols] = block[:rows, :cols]

        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, grid)
        return np.asarray(grid)

    def patch(
        self,
        latitude: tuple[float, float],
        longitude: tuple[float, float],
        max_width: int = 3072,
    ) -> ReliefMap:
        """A local :class:`ReliefMap` read straight from the source tiles.

        The global grid is decimated to 9.8 km cells, which is right for
        shading a continent from orbit and useless from 20 km up — it puts
        one elevation sample across the whole visible ground. This reads the
        **7.5-arc-second** archive over a degree box instead: 230 m cells,
        comparable to the 15-arc-second imagery drawn over them, so the two
        carry detail at the same scale rather than one blurring the other.

        Assembled across every tile the box touches on a shared integer
        decimation stride, for the same reason
        :meth:`~aether.viz.imagery.BlueMarble.window` does it that way:
        independent per-tile rounding leaves the pieces a fraction of a cell
        out of register and the seam shows as a false ridge.

        **The lattice comes from the files, not from the degree grid.**
        GMTED tiles start half an arc-second off the whole degree — a tile
        nominally at 60 E has its first column edge at 59.9998611 — and the
        first version of this method assumed a clean corner. Against
        :meth:`elevation`, which has always read the affine transform, that
        put the same coordinate **877 m** out in the Khumbu, turning a
        genuine 60-degree face into an 83-degree one and with it the shading
        that is the whole point of the patch.

        Tiles are used at a single resolution — the finest the box touches —
        because the 7.5-arc-second product and the 30-arc-second Antarctic
        one do not share a lattice and interleaving them would reintroduce
        exactly the misregistration above.

        Raises
        ------
        ValueError
            For an empty box, or one crossing the antimeridian — which would
            be two disjoint reads, and one array with a seam through it is
            worse than an error.
        """
        try:
            import rasterio
            from rasterio.windows import Window
        except ImportError as error:  # pragma: no cover - dependency declared
            msg = "reading GMTED2010 needs rasterio (pip install rasterio)"
            raise ImportError(msg) from error

        south, north = sorted(float(v) for v in latitude)
        west, east = float(longitude[0]), float(longitude[1])
        if east <= west:
            msg = (
                f"longitude box must run west to east without crossing the "
                f"antimeridian, got ({west}, {east})"
            )
            raise ValueError(msg)
        if not (-90.0 <= south < north <= 90.0):
            msg = f"latitude box must lie in [-90, 90] and be non-empty, got {latitude}"
            raise ValueError(msg)

        touched = [
            tile
            for tile in self._tiles
            if tile.west < east and tile.east > west and tile.south < north and tile.north > south
        ]
        if not touched:
            msg = (
                f"no GMTED2010 tile covers {latitude}, {longitude}; the archive "
                f"spans {self.coverage} in latitude"
            )
            raise ValueError(msg)

        # The lattice, taken from the files. Every tile at a given product
        # resolution shares one grid whose origin is offset half an arc-second
        # from the whole degree, so the reference origin and step are read
        # rather than assumed.
        geometry: dict[Path, tuple[float, float, float, int, int]] = {}
        for tile in touched:
            with rasterio.open(tile.path) as handle:
                transform = handle.transform
                geometry[tile.path] = (
                    float(transform.a),
                    float(transform.c),
                    float(transform.f),
                    int(handle.width),
                    int(handle.height),
                )
        step = min(entry[0] for entry in geometry.values())
        usable = [t for t in touched if np.isclose(geometry[t.path][0], step)]
        origin_x = min(geometry[t.path][1] for t in usable)
        origin_y = max(geometry[t.path][2] for t in usable)

        def column_of(longitude_deg: float) -> float:
            return (longitude_deg - origin_x) / step

        def row_of(latitude_deg: float) -> float:
            return (origin_y - latitude_deg) / step

        col0 = int(np.floor(column_of(west)))
        col1 = int(np.ceil(column_of(east)))
        row0 = int(np.floor(row_of(north)))
        row1 = int(np.ceil(row_of(south)))
        stride = max(int(np.ceil((col1 - col0) / max(int(max_width), 1))), 1)
        col0 -= col0 % stride
        row0 -= row0 % stride
        out_w = max((col1 - col0) // stride, 1)
        out_h = max((row1 - row0) // stride, 1)
        col1, row1 = col0 + out_w * stride, row0 + out_h * stride

        grid = np.zeros((out_h, out_w), dtype=np.float32)
        for tile in usable:
            _, tile_x, tile_y, width, height = geometry[tile.path]
            left = round(column_of(tile_x))
            top = round(row_of(tile_y))
            c0, c1 = max(col0, left), min(col1, left + width)
            r0, r1 = max(row0, top), min(row1, top + height)
            c0 += (-(c0 - col0)) % stride
            r0 += (-(r0 - row0)) % stride
            c1 -= (c1 - col0) % stride
            r1 -= (r1 - row0) % stride
            if c1 <= c0 or r1 <= r0:
                continue
            with rasterio.open(tile.path) as handle:
                block = handle.read(
                    1,
                    window=Window(c0 - left, r0 - top, c1 - c0, r1 - r0),
                    out_shape=((r1 - r0) // stride, (c1 - c0) // stride),
                    resampling=rasterio.enums.Resampling.average,
                ).astype(np.float32)
            block = np.where(block <= NODATA + 1, 0.0, block)
            slot_r, slot_c = (r0 - row0) // stride, (c0 - col0) // stride
            grid[slot_r : slot_r + block.shape[0], slot_c : slot_c + block.shape[1]] = block

        return ReliefMap.from_grid(
            grid,
            bounds=(
                origin_y - row1 * step,
                origin_y - row0 * step,
                origin_x + col0 * step,
                origin_x + col1 * step,
            ),
        )

    def relief(
        self, height: int = 2048, cache: Path | None = None, exaggeration: float = 1.0
    ) -> ReliefMap:
        """A :class:`ReliefMap` built from :meth:`coarse`, ready to shade with.

        Memoised for the process. The grids are 33 MB apiece and immutable
        once built; rebuilding them per animator is the same waste the
        mosaic cache exists to stop.
        """
        key = (str(Path(self.root).resolve()), self.product, int(height), float(exaggeration))
        held = _RELIEFS.get(key)
        if held is not None:
            return held
        built = ReliefMap.from_grid(
            self.coarse(height=height, cache=cache), exaggeration=exaggeration
        )
        _RELIEFS[key] = built
        return built


@dataclass(frozen=True)
class ReliefMap:
    """A global elevation grid and the surface slopes derived from it.

    What this is for
    ----------------

    The globe renderer intersects an analytic ellipsoid, so its surface
    normal is the geodetic vertical everywhere and every frame comes out as
    smoothly shaded as a billiard ball. Real terrain is visible from orbit
    because it is *lit* differently, not because it is 8 km closer, and the
    quantity that does that is the slope.

    So the elevation grid is differentiated once, into east and north
    slopes as **dimensionless rise over run in metres** — which requires the
    metric, because a degree of longitude is 111 km at the equator and 19 km
    at 80 degrees north, and dividing by degrees instead would make every
    high-latitude hill look like a cliff.

    What it is not
    --------------

    **Shading, not displacement.** The intersected surface is still the
    ellipsoid: a mountain here changes how a pixel is lit, not where the
    ground is. Against a 6,378 km radius, Everest is 0.14 % — under a pixel
    on a full-disc globe — so for orbital views the distinction does not
    arise. It very much arises on a launch-pad close-up, which is what
    :func:`~aether.viz.globe.render`'s ``displace`` path is for; that one
    marches the ray against this same grid and does move the ground.

    **Limited by the grid it came from.** At the 2048-row default a cell is
    9.8 km, so this resolves mountain *ranges*, not peaks. The Himalayan
    front and the Andean scarp read correctly; a single ridge does not
    exist at that scale and no exaggeration factor invents it.

    Global or local
    ---------------

    The same class covers both, because they differ only in footprint. The
    global grid at 2048 rows has 9.8 km cells and resolves mountain
    *ranges*; a local patch read straight from the 7.5-arc-second archive
    has 230 m cells and resolves the ridge a launch site sits behind. A
    launch close-up needs the second, and the renderer takes a stack of
    them the way it takes a stack of textures: coarse first, fine last.

    Attributes
    ----------
    elevation:
        ``(rows, cols)`` metres. Row 0 is the ``north`` edge, column 0 the
        ``west`` edge.
    slope_east, slope_north:
        Rise over run, dimensionless, positive uphill toward east and north.
    exaggeration:
        Vertical scaling applied to the slopes. ``1.0`` is the truth and the
        default; larger values are a stated cheat, not a correction.
    south, north, west, east:
        Degree bounds of the grid's outer **edges**, as
        :class:`~aether.viz.imagery.Texture` carries them and for the same
        reason. The default is the whole globe.

    Notes
    -----
    The three grids are typed loosely for the same reason
    :attr:`~aether.viz.imagery.Texture.data` is: the renderer may hold a
    device-resident copy, and nothing here indexes them. Their shapes are
    checked on construction, which is the property that actually matters.
    """

    elevation: Any
    slope_east: Any
    slope_north: Any
    exaggeration: float = 1.0
    south: float = -90.0
    north: float = 90.0
    west: float = -180.0
    east: float = 180.0

    def __post_init__(self) -> None:
        if self.elevation.ndim != 2:
            msg = f"elevation must be a 2-D grid, got shape {self.elevation.shape}"
            raise ValueError(msg)
        if not (self.north > self.south and self.east > self.west):
            msg = (
                f"relief bounds must be non-empty and run south-to-north, "
                f"west-to-east; got {self.south}..{self.north} N, "
                f"{self.west}..{self.east} E"
            )
            raise ValueError(msg)
        for name in ("slope_east", "slope_north"):
            if getattr(self, name).shape != self.elevation.shape:
                msg = (
                    f"{name} must match the elevation grid "
                    f"{self.elevation.shape}, got {getattr(self, name).shape}"
                )
                raise ValueError(msg)

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.elevation.shape[0]), int(self.elevation.shape[1])

    @property
    def wraps(self) -> bool:
        """Whether longitude sampling wraps — full-globe grids only."""
        return bool(np.isclose(self.east - self.west, 360.0))

    @property
    def ground_resolution(self) -> float:
        """Metres per cell in longitude at the grid's mid-latitude."""
        _, cols = self.shape
        mid = np.deg2rad(0.5 * (self.north + self.south))
        step = np.deg2rad((self.east - self.west) / cols)
        return float(step * 6378137.0 * max(np.cos(mid), 1.0e-6))

    def covers(self, latitude: float, longitude: float) -> bool:
        """Whether a degree point lies inside this grid's footprint."""
        if self.wraps:
            return bool(self.south <= latitude <= self.north)
        lon = (float(longitude) - self.west) % 360.0 + self.west
        return bool(self.south <= float(latitude) <= self.north and self.west <= lon <= self.east)

    def overlaps(self, latitude: float, longitude: float, radius_degrees: float) -> bool:
        """Whether this grid meets a disc of ``radius_degrees`` about a point.

        The frame-sized companion to :meth:`covers`; see
        :meth:`aether.viz.imagery.Texture.overlaps`, which this mirrors so
        the animator can cull imagery and elevation by one rule.
        """
        reach = max(float(radius_degrees), 0.0)
        if not self.south - reach <= float(latitude) <= self.north + reach:
            return False
        if self.wraps:
            return True
        lon = (float(longitude) - self.west) % 360.0 + self.west
        return bool(
            self.west - reach <= lon <= self.east + reach
            or self.west - reach <= lon - 360.0 <= self.east + reach
        )

    @classmethod
    def from_grid(
        cls,
        grid: NDArray[np.float32] | _FloatArray,
        exaggeration: float = 1.0,
        bounds: tuple[float, float, float, float] = (-90.0, 90.0, -180.0, 180.0),
        semi_major: float = 6378137.0,
        flattening: float = 1.0 / 298.257223563,
    ) -> ReliefMap:
        """Differentiate an equirectangular elevation grid on the ellipsoid.

        Central differences, wrapping in longitude — the grid is periodic
        there and a one-sided difference at the antimeridian would draw a
        meridian-long false scarp straight down the Pacific.

        In latitude the ends are one-sided, which is correct: the grid stops
        at the poles and there is nothing beyond them to difference against.

        **The east stencil widens toward the poles.** A cell's east-west
        extent shrinks as :math:`\\cos\\varphi`, so on a 2048-row grid the
        top row's cells are 7.5 m across where the equator's are 9.8 km.
        Differencing over one cell there divides a real elevation step by a
        vanishing baseline: measured, that returned a **slope of 186** — an
        89.7 degree cliff — on the polar rows.

        Flooring the *step* is the wrong fix, and was the first one tried:
        the floor is itself the tiny polar width, so it changes nothing, and
        raising it to the meridional step instead suppresses genuine
        east-west slope from about 7 degrees of latitude outward.

        So the difference is taken over :math:`1/\\cos\\varphi` columns
        instead, which holds the *physical* baseline roughly constant from
        equator to pole. That is the standard treatment of the
        equirectangular grid's coordinate singularity, and it is a
        statement about the grid rather than about the ground.
        """
        elevation = np.asarray(grid, dtype=np.float32)
        if elevation.ndim != 2:
            msg = f"grid must be 2-D, got shape {elevation.shape}"
            raise ValueError(msg)
        rows, cols = elevation.shape
        scale = float(exaggeration)
        south, north, west, east = (float(v) for v in bounds)
        wraps = bool(np.isclose(east - west, 360.0))

        d_lat = np.deg2rad(north - south) / rows
        d_lon = np.deg2rad(east - west) / cols
        # Cell-centre latitudes: row 0 spans `north` down to north - d_lat.
        latitude = np.deg2rad(north) - (np.arange(rows) + 0.5) * d_lat

        e2 = flattening * (2.0 - flattening)
        w = np.sqrt(1.0 - e2 * np.sin(latitude) ** 2)
        # Meridian and prime-vertical radii of curvature at each row.
        meridian = semi_major * (1.0 - e2) / w**3
        prime_vertical = semi_major / w

        north_step = meridian * d_lat
        cos_phi = np.cos(latitude)
        # Columns to reach either side, so the baseline stays near one
        # equatorial cell. Capped at a quarter turn, past which "east" has
        # stopped meaning anything local.
        reach = np.clip(np.round(1.0 / np.maximum(cos_phi, 1.0e-12)), 1, cols // 4)
        reach = reach.astype(np.int64)
        east_step = prime_vertical * cos_phi * d_lon * reach

        # Longitude wraps only on a full-globe grid; on a local patch the
        # far edge is a different place and wrapping into it would draw a
        # scarp where the crop stops.
        columns = np.arange(cols)
        if wraps:
            ahead = (columns[None, :] + reach[:, None]) % cols
            behind = (columns[None, :] - reach[:, None]) % cols
        else:
            ahead = np.clip(columns[None, :] + reach[:, None], 0, cols - 1)
            behind = np.clip(columns[None, :] - reach[:, None], 0, cols - 1)
        d_east = 0.5 * (
            np.take_along_axis(elevation, ahead, axis=1)
            - np.take_along_axis(elevation, behind, axis=1)
        )
        if not wraps:
            # A clamped stencil at the edge spans fewer cells than it thinks.
            span = (ahead - behind).astype(np.float64)
            d_east = d_east * np.where(span > 0.0, 2.0 * reach[:, None] / span, 0.0)
        d_north = np.zeros_like(elevation)
        # Row index increases southward, so a positive northward slope is a
        # *decrease* in row index — hence the sign.
        d_north[1:-1] = 0.5 * (elevation[:-2] - elevation[2:])
        d_north[0] = elevation[0] - elevation[1]
        d_north[-1] = elevation[-2] - elevation[-1]

        return cls(
            elevation=elevation,
            slope_east=np.asarray(scale * d_east / east_step[:, None], dtype=np.float32),
            slope_north=np.asarray(scale * d_north / north_step[:, None], dtype=np.float32),
            exaggeration=scale,
            south=south,
            north=north,
            west=west,
            east=east,
        )


@dataclass
class TerrainProbe:
    """Ground elevation for an integrator, at integrator call rates.

    :meth:`Terrain.elevation` opens a GeoTIFF and issues a windowed read per
    call. That is right for a batch of points and wrong inside a
    ``solve_ivp`` event, which is evaluated at every trial step — a terminal
    ground condition would spend the whole flight in file I/O.

    So the first query loads a ``span``-degree patch around itself and every
    query after that is a bilinear lookup in memory. A descent stays inside
    one patch; when a query falls outside, the patch is reloaded around the
    new point rather than the query being answered wrongly.

    Attributes
    ----------
    terrain:
        The archive.
    span:
        Width of the cached patch, in degrees.
    reads:
        How many source reads have happened. A diagnostic worth having: if
        this climbs with the step count, the cache is thrashing and the
        span is too small for the trajectory.
    """

    terrain: Terrain
    span: float = 3.0
    max_width: int = 2048
    _patch: ReliefMap | None = field(default=None, repr=False)
    reads: int = 0

    def __call__(self, latitude: float, longitude: float) -> float:
        """Ground elevation at a **degree** point (m)."""
        lat, lon = float(latitude), float(longitude)
        if self._patch is None or not self._patch.covers(lat, lon):
            half = 0.5 * float(self.span)
            self._patch = self.terrain.patch(
                (max(lat - half, -90.0), min(lat + half, 90.0)),
                (max(lon - half, -180.0), min(lon + half, 180.0)),
                max_width=self.max_width,
            )
            self.reads += 1
        patch = self._patch
        rows, cols = patch.shape
        row = (patch.north - lat) / ((patch.north - patch.south) / rows) - 0.5
        column = (lon - patch.west) / ((patch.east - patch.west) / cols) - 0.5
        row = float(np.clip(row, 0.0, rows - 1.000001))
        column = float(np.clip(column, 0.0, cols - 1.000001))
        r0, c0 = int(row), int(column)
        r1, c1 = min(r0 + 1, rows - 1), min(c0 + 1, cols - 1)
        fr, fc = row - r0, column - c0
        grid = patch.elevation
        top = grid[r0, c0] * (1.0 - fc) + grid[r0, c1] * fc
        bottom = grid[r1, c0] * (1.0 - fc) + grid[r1, c1] * fc
        return float(top * (1.0 - fr) + bottom * fr)


#: Environment variable naming the directory that holds the open reference
#: datasets. Checked before any filesystem search.
DATA_ROOT_ENV = "AETHER_DATA_ROOT"


def _dataset_candidates(name: str) -> list[Path]:
    """Where to look for a named dataset, in order of decreasing confidence.

    ``AETHER_DATA_ROOT`` first, then a walk up from this module. The walk is a
    *fallback*, not the mechanism: it works only while the module happens to sit
    inside a checkout that also holds the data, and it breaks silently the
    moment either moves. Promoting these renderers into the kernel broke exactly
    that assumption -- 72 tests failed at once, all of them reporting a missing
    elevation archive that had not gone anywhere.

    A library should be told where its data is, not deduce it from its own
    location on disk.
    """
    found: list[Path] = []
    configured = os.environ.get(DATA_ROOT_ENV)
    if configured:
        root = Path(configured).expanduser()
        found += [
            root / name,
            root / "datasets" / name,
            root / "reference" / name,
            root / "reference" / "datasets" / name,
        ]
    for parent in Path(__file__).resolve().parents:
        found += [
            parent / "datasets" / name,
            parent / "reference" / name,
            parent / "reference" / "datasets" / name,
        ]
    return found


def default_terrain(root: str | Path | None = None, product: str = "mea") -> Terrain:
    """Locate the archive by walking up from this module."""
    if root is not None:
        return Terrain(Path(root), product=product)
    for candidate in _dataset_candidates("GMTED2010"):
        if candidate.is_dir():
            return Terrain(candidate, product=product)
    msg = (
        "no datasets/GMTED2010 (or reference/GMTED2010, reference/datasets/GMTED2010) "
        f"found. Set ${DATA_ROOT_ENV} or pass an explicit root."
    )
    raise FileNotFoundError(msg)
