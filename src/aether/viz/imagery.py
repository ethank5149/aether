"""Blue Marble Next Generation as the globe texture, at the month being flown.

``reference/blue-marble-next-gen`` holds all twelve months of NASA's BMNG
topography-and-bathymetry product: eight GeoTIFF tiles each, 21600 x 21600,
three uint8 bands, EPSG:4326. Assembled that is **86400 x 43200 at 15
arc-seconds** — roughly 460 m at the equator, against the 4096 x 2048 JPEG
the renderer used before, which is 9.7 km. Two and a half thousand times the
pixel count.

Which is exactly the problem. The full mosaic is 11 GB as uint8 and 89 GB as
the float64 the renderer wants, so it cannot simply be loaded. Two paths are
provided instead, and they are the two things a trajectory animation
actually asks for:

:meth:`BlueMarble.mosaic`
    A decimated global equirectangular texture at a requested height,
    assembled once and cached to disk as ``uint8``. This is what a full-disc
    or mid-range view needs, where the globe spans at most a thousand pixels
    and anything finer than about 8192 x 4096 is thrown away by the
    rasteriser regardless.
:meth:`BlueMarble.window`
    A native-resolution crop of a latitude/longitude box, read straight from
    the GeoTIFFs. This is what a launch-pad or impact close-up needs, where
    the camera is 20 km up and the visible ground is a fraction of a degree
    across. At that range the global mosaic would be showing one texel per
    fifty pixels.

**Stored as uint8, converted on upload.** The source is uint8, the renderer
divides by 255 anyway, and keeping the cache in the source dtype makes an
8192 x 4096 texture 100 MB instead of the 800 MB it would be as float64 —
which is the difference between a texture that fits on the GPU beside the
frame buffers and one that does not.

**The month is chosen, not assumed.** BMNG's whole point is that the surface
changes: northern hemisphere snow line, Sahel vegetation, sea ice. A
January launch rendered on the August texture is a different planet at the
latitudes an ICBM trajectory actually crosses. :meth:`BlueMarble.for_date`
resolves a date to the right month.

Everything leaves here as a :class:`Texture`
--------------------------------------------

A raw array is not enough to draw with, because the renderer has to know
*where on the Earth each texel is*. The global mosaic and a native-
resolution crop differ in exactly that, and nothing else, so both come back
as a :class:`Texture`: pixels plus the degree box they cover, with the box
given in **pixel edges** — the convention rasterio reports and GDAL stores.

That distinction is not pedantry. The renderer previously mapped latitude
+90 to row 0 and -90 to row ``rows-1``, which places texel *centres* on the
poles and is a half-pixel shift against an edge-aligned source. On the
4096-row texture it used that was 2.4 km on the ground; on an
8192-row BMNG mosaic it is 1.2 km, and on a 15-arc-second close-up it is
230 m — the same order as the impact accuracy the rest of this package
argues about.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "MONTH_NAMES",
    "BlueMarble",
    "Texture",
    "TileKey",
    "default_blue_marble",
]

_ByteImage = NDArray[np.uint8]

#: Mosaics already read this process, keyed by cache file, height and month.
#: Bounded by the number of distinct months a run touches, which is one.
_MOSAICS: dict[tuple[str, int, int], _ByteImage] = {}


@dataclass(frozen=True)
class Texture:
    """An image and the degree box it covers, in pixel edges.

    Attributes
    ----------
    data:
        ``(rows, cols, 3)``. ``uint8`` straight from the source, or float in
        ``[0, 1]``; :func:`~aether.viz.globe.render` normalises on the way
        in and does not care which. May live on a GPU — nothing here indexes
        it, so the array only has to support ``.shape``.
    south, north, west, east:
        Degree bounds of the image's outer edges. Row 0's *top* edge is at
        ``north`` and column 0's *left* edge at ``west``.

    Notes
    -----
    ``wraps`` is derived rather than passed: a texture is treated as
    seamless in longitude exactly when it spans the full 360 degrees, which
    is the only case where wrapping is correct. A 90-degree crop that
    wrapped would sample the far side of the tile at its edge.
    """

    data: Any
    south: float = -90.0
    north: float = 90.0
    west: float = -180.0
    east: float = 180.0

    def __post_init__(self) -> None:
        shape = tuple(int(n) for n in self.data.shape)
        if len(shape) != 3 or shape[2] != 3:
            msg = f"texture data must have shape (rows, cols, 3), got {shape}"
            raise ValueError(msg)
        if not self.north > self.south:
            msg = f"texture needs north > south, got {self.south} to {self.north}"
            raise ValueError(msg)
        if not self.east > self.west:
            msg = (
                f"texture needs east > west without wrapping the antimeridian, "
                f"got {self.west} to {self.east}"
            )
            raise ValueError(msg)

    @property
    def shape(self) -> tuple[int, int]:
        """``(rows, cols)``."""
        return int(self.data.shape[0]), int(self.data.shape[1])

    @property
    def wraps(self) -> bool:
        """Whether longitude sampling should wrap — full-globe textures only."""
        return bool(np.isclose(self.east - self.west, 360.0))

    @property
    def degrees_per_pixel(self) -> tuple[float, float]:
        """``(latitude, longitude)`` degrees per pixel."""
        rows, cols = self.shape
        return (self.north - self.south) / rows, (self.east - self.west) / cols

    @property
    def ground_resolution(self) -> float:
        """Metres per pixel in longitude at the box's mid-latitude.

        The number a caller compares against a camera's metres-per-pixel to
        decide whether this texture is worth reading. Longitude rather than
        latitude because that is the axis that shrinks with the cosine, so
        it is the binding one.
        """
        _, d_lon = self.degrees_per_pixel
        mid = np.deg2rad(0.5 * (self.north + self.south))
        return float(np.deg2rad(d_lon) * 6378137.0 * max(np.cos(mid), 1.0e-6))

    def covers(
        self, latitude: float, longitude: float, margin: float = 0.0
    ) -> bool:
        """Whether a degree point lies inside the box, less ``margin`` degrees."""
        lon = (float(longitude) - self.west) % 360.0 + self.west
        return bool(
            self.south + margin <= float(latitude) <= self.north - margin
            and self.west + margin <= lon <= self.east - margin
        )

    def overlaps(
        self, latitude: float, longitude: float, radius_degrees: float
    ) -> bool:
        """Whether this box meets a disc of ``radius_degrees`` about a point.

        The test a *frame* needs, where :meth:`covers` is the test a *point*
        needs. A camera aimed just outside a crop still sees most of it, and
        selecting crops by whether they contain the look-at point therefore
        dropped the crop exactly as the vehicle approached its edge — which
        the viewer sees as the ground going soft a moment before the seam
        arrives. The disc is taken in degrees on both axes, so it is
        generous in longitude away from the equator; that errs towards
        compositing a crop that contributes nothing, which costs a texture
        fetch, rather than towards dropping one that does.
        """
        reach = max(float(radius_degrees), 0.0)
        if not self.south - reach <= float(latitude) <= self.north + reach:
            return False
        if self.wraps:
            return True
        lon = (float(longitude) - self.west) % 360.0 + self.west
        if self.west - reach <= lon <= self.east + reach:
            return True
        # The same point one turn the other way, for a box that straddles
        # the seam of this modulo.
        return bool(self.west - reach <= lon - 360.0 <= self.east + reach)

    def with_data(self, data: Any) -> Texture:
        """The same box with different pixels — for a device upload."""
        return Texture(data, self.south, self.north, self.west, self.east)

MONTH_NAMES = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)

#: ``(column, row)`` of each BMNG tile and the degree box it covers.
#: Columns A-D run west to east from 180 W; rows 1-2 run north then south.
_COLUMN_WEST = {"A": -180.0, "B": -90.0, "C": 0.0, "D": 90.0}
_ROW_NORTH = {"1": 90.0, "2": 0.0}
#: The same layout as whole-tile indices, for assembling on a global grid.
_COLUMN_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}
_ROW_INDEX = {"1": 0, "2": 1}

TileKey = str
"""One of ``A1``, ``A2``, ..., ``D2``."""


def _parse(name: str) -> tuple[int, TileKey] | None:
    """Month number and tile key from a BMNG filename, or ``None``."""
    month = re.search(r"\.(\d{4})(\d{2})\.", name)
    tile = re.search(r"([ABCD][12])_geo", name)
    if month is None or tile is None:
        return None
    return int(month.group(2)), tile.group(1)


@dataclass
class BlueMarble:
    """The BMNG archive, indexed by month and tile.

    Attributes
    ----------
    root:
        Directory holding the GeoTIFFs, in any arrangement — the index is
        built from *filenames*, so the month subdirectories, the loose files
        at the top level and the ``(1)`` duplicates a download leaves behind
        all resolve to the same place.
    cache:
        Where assembled mosaics are written. A mosaic costs a minute or two
        to build because the source tiles are deflate-compressed and
        striped, with no overviews; it is not something to redo per frame.
    """

    root: Path
    cache: Path | None = None
    _index: dict[int, dict[TileKey, Path]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if not self.root.is_dir():
            msg = (
                f"no Blue Marble archive at {self.root}. Expected NASA's Blue "
                f"Marble Next Generation GeoTIFFs, eight tiles per month named "
                f"like world.topo.bathy.200401.3x21600x21600.C1_geo.tif."
            )
            raise FileNotFoundError(msg)
        if self.cache is None:
            self.cache = self.root / "_mosaics"
        for path in sorted(self.root.rglob("*.tif")):
            parsed = _parse(path.name)
            if parsed is None:
                continue
            month, tile = parsed
            # First match wins, so a "(1)" duplicate never displaces the
            # original it was copied from.
            self._index.setdefault(month, {}).setdefault(tile, path)

    @property
    def months(self) -> tuple[int, ...]:
        """Month numbers with a complete eight-tile set."""
        return tuple(sorted(m for m, t in self._index.items() if len(t) == 8))

    def tiles(self, month: int) -> dict[TileKey, Path]:
        """The eight tile paths for a month number (1-12)."""
        available = self._index.get(int(month), {})
        if len(available) != 8:
            msg = (
                f"month {month} has {len(available)} of 8 Blue Marble tiles in "
                f"{self.root}; complete months are {self.months}"
            )
            raise FileNotFoundError(msg)
        return dict(available)

    @staticmethod
    def month_of(when: date | datetime | str | int) -> int:
        """Resolve a date, month name or number to a month number."""
        if isinstance(when, int):
            if not 1 <= when <= 12:
                msg = f"month must be 1-12, got {when}"
                raise ValueError(msg)
            return when
        if isinstance(when, (date, datetime)):
            return int(when.month)
        text = str(when).strip()
        # Lower-cased only for the month-name lookup: an ISO timestamp
        # carries a capital "T" that numpy will not parse in lower case.
        if text.lower() in MONTH_NAMES:
            return MONTH_NAMES.index(text.lower()) + 1
        stamp = np.datetime64(text).astype("datetime64[M]").astype(int)
        return int(stamp % 12) + 1

    def for_date(self, when: date | datetime | str | int, height: int = 4096) -> _ByteImage:
        """Global mosaic for the month containing ``when``."""
        return self.mosaic(height=height, month=self.month_of(when))

    # -- global mosaic -----------------------------------------------------

    def mosaic_path(self, height: int, month: int) -> Path:
        assert self.cache is not None
        return self.cache / f"bmng-{month:02d}-{height}.npy"

    def mosaic(
        self, height: int = 4096, month: int = 1, rebuild: bool = False
    ) -> _ByteImage:
        """Decimated global equirectangular texture, ``(height, 2*height, 3)`` uint8.

        Row 0 is +90 latitude and column 0 is -180 longitude, which is the
        convention :func:`aether.viz.globe.render` samples with.

        Built by asking rasterio for a decimated read of each tile directly
        into its slot in the output, so the full-resolution image is never
        materialised. The source has no overviews, so this still decompresses
        every strip — about a minute a month — which is why the result is
        cached.

        Cached **twice**: to disk against the rebuild, and in memory against
        the reload. A 4096-row mosaic is 100 MB, and an animator that
        constructs one per call spent longer reading the same file off disk
        than rendering the frames it was read for.
        """
        if height < 64 or height % 2 != 0:
            msg = f"mosaic height must be even and at least 64, got {height}"
            raise ValueError(msg)
        destination = self.mosaic_path(height, month)
        cache_key = (str(destination.resolve()), int(height), int(month))
        if not rebuild:
            held = _MOSAICS.get(cache_key)
            if held is not None:
                return held
        if destination.is_file() and not rebuild:
            loaded: _ByteImage = np.asarray(np.load(destination))
            _MOSAICS[cache_key] = loaded
            return loaded

        try:
            import rasterio
        except ImportError as error:  # pragma: no cover - dependency declared
            msg = "reading Blue Marble GeoTIFFs needs rasterio (pip install rasterio)"
            raise ImportError(msg) from error

        width = 2 * height
        tile_h, tile_w = height // 2, width // 4
        image = np.zeros((height, width, 3), dtype=np.uint8)
        columns = {"A": 0, "B": 1, "C": 2, "D": 3}

        for key, path in sorted(self.tiles(month).items()):
            column, row = key[0], key[1]
            top = 0 if row == "1" else tile_h
            left = columns[column] * tile_w
            with rasterio.open(path) as handle:
                block = handle.read(
                    indexes=[1, 2, 3],
                    out_shape=(3, tile_h, tile_w),
                    resampling=rasterio.enums.Resampling.average,
                )
            image[top : top + tile_h, left : left + tile_w] = np.transpose(
                block, (1, 2, 0)
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, image)
        _MOSAICS[cache_key] = image
        return image

    # -- native-resolution window ------------------------------------------

    def window(
        self,
        latitude: tuple[float, float],
        longitude: tuple[float, float],
        month: int = 1,
        max_width: int = 4096,
    ) -> Texture:
        """Native-resolution crop of a lat/lon box, in degrees.

        Returns a :class:`Texture` whose bounds are snapped **outward to
        whole source pixels** — a caller that assumed it got exactly the box
        it asked for would misregister the texture by up to half a pixel,
        and at 15 arc-seconds that is 230 m on the ground.

        Boxes spanning the antimeridian are refused rather than silently
        wrapped: the crop would be two disjoint reads, and returning one
        array with a seam in the middle of it is worse than an error.

        Notes
        -----
        Boxes crossing a tile edge are assembled from every tile they touch,
        not from the first one. The tile grid breaks at longitudes -90, 0
        and 90 and **at the equator**, so "a close-up never straddles an
        edge" is not true of anything launched from or aimed at low
        latitudes; an earlier version took the first intersecting tile and
        returned a crop quietly narrower than the box asked for.

        Assembly is by a shared **integer decimation stride** over the
        global 15-arc-second grid rather than by per-tile output shapes.
        Independent rounding per tile puts the pieces a fraction of a pixel
        out of register, which shows up as a visible line down the seam at
        exactly the resolution this method exists to provide.
        """
        try:
            import rasterio
            from rasterio.windows import Window
        except ImportError as error:  # pragma: no cover - dependency declared
            msg = "reading Blue Marble GeoTIFFs needs rasterio (pip install rasterio)"
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
        if not (west >= -180.0 and east <= 180.0):
            msg = f"longitude box must lie in [-180, 180], got ({west}, {east})"
            raise ValueError(msg)

        tiles = self.tiles(self.month_of(month))
        # Every BMNG tile is the same size, so one global pixel grid indexes
        # all eight and the per-tile offsets are exact integers.
        with rasterio.open(next(iter(tiles.values()))) as probe:
            tile_pixels = int(probe.width)
        degrees_per_pixel = 90.0 / tile_pixels

        col0 = int(np.floor((west + 180.0) / degrees_per_pixel))
        col1 = int(np.ceil((east + 180.0) / degrees_per_pixel))
        row0 = int(np.floor((90.0 - north) / degrees_per_pixel))
        row1 = int(np.ceil((90.0 - south) / degrees_per_pixel))
        col0, col1 = max(col0, 0), min(col1, 4 * tile_pixels)
        row0, row1 = max(row0, 0), min(row1, 2 * tile_pixels)
        span_w, span_h = col1 - col0, row1 - row0
        if span_w <= 0 or span_h <= 0:  # pragma: no cover - guarded above
            msg = f"empty crop for {latitude}, {longitude}"
            raise ValueError(msg)

        # One stride for the whole crop. Snapping the origin to a multiple of
        # it keeps every tile's contribution on the same output lattice.
        stride = max(int(np.ceil(span_w / max(int(max_width), 1))), 1)
        col0 -= col0 % stride
        row0 -= row0 % stride
        out_w = max((col1 - col0) // stride, 1)
        out_h = max((row1 - row0) // stride, 1)
        col1, row1 = col0 + out_w * stride, row0 + out_h * stride

        image = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        for key, path in sorted(tiles.items()):
            left = _COLUMN_INDEX[key[0]] * tile_pixels
            top = _ROW_INDEX[key[1]] * tile_pixels
            # Overlap of the requested crop with this tile, in global pixels.
            c0, c1 = max(col0, left), min(col1, left + tile_pixels)
            r0, r1 = max(row0, top), min(row1, top + tile_pixels)
            # Round in to whole output pixels so a tile never writes a
            # partially-covered one, which would leave a black seam.
            c0 += (-(c0 - col0)) % stride
            r0 += (-(r0 - row0)) % stride
            c1 -= (c1 - col0) % stride
            r1 -= (r1 - row0) % stride
            if c1 <= c0 or r1 <= r0:
                continue
            with rasterio.open(path) as handle:
                block = handle.read(
                    indexes=[1, 2, 3],
                    window=Window(c0 - left, r0 - top, c1 - c0, r1 - r0),
                    out_shape=(3, (r1 - r0) // stride, (c1 - c0) // stride),
                    resampling=rasterio.enums.Resampling.average,
                )
            slot_r, slot_c = (r0 - row0) // stride, (c0 - col0) // stride
            image[
                slot_r : slot_r + block.shape[1], slot_c : slot_c + block.shape[2]
            ] = np.transpose(block, (1, 2, 0))

        return Texture(
            image,
            south=90.0 - row1 * degrees_per_pixel,
            north=90.0 - row0 * degrees_per_pixel,
            west=col0 * degrees_per_pixel - 180.0,
            east=col1 * degrees_per_pixel - 180.0,
        )

    def texture(self, height: int = 4096, month: int = 1) -> Texture:
        """The global mosaic as a :class:`Texture`, bounds and all."""
        return Texture(self.mosaic(height=height, month=self.month_of(month)))


#: Environment variable naming the directory that holds the open reference
#: datasets. Checked before any filesystem search.
DATA_ROOT_ENV = "AETHER_DATA_ROOT"


def _dataset_candidates(name: str) -> list[Path]:
    """Where to look for a named dataset, in order of decreasing confidence.

    ``AETHER_DATA_ROOT`` first, then a walk up from this module. The walk is a
    *fallback*, not the mechanism: it works only while the module happens to sit
    inside a checkout that also holds the data, and it breaks silently the
    moment either moves. Promoting these renderers into the kernel broke exactly
    that assumption -- 72 tests failed at once, every one of them reporting a
    missing archive that had not gone anywhere.

    A library should be told where its data is, not deduce it from its own
    location on disk.
    """
    found: list[Path] = []
    configured = os.environ.get(DATA_ROOT_ENV)
    if configured:
        root = Path(configured).expanduser()
        found += [root / name, root / "datasets" / name,
                  root / "reference" / name, root / "reference" / "datasets" / name]
    for parent in Path(__file__).resolve().parents:
        found += [parent / "datasets" / name,
                  parent / "reference" / name,
                  parent / "reference" / "datasets" / name]
    return found


def default_blue_marble(root: str | Path | None = None) -> BlueMarble:
    """Locate the archive by walking up from this module.

    Same reasoning as the old texture loader: a notebook runs from its own
    folder and a test from the repository root, so a path relative to the
    process working directory works in exactly one of them.
    """
    if root is not None:
        return BlueMarble(Path(root))
    for candidate in _dataset_candidates("blue-marble-next-gen"):
        if candidate.is_dir():
            return BlueMarble(candidate)
    msg = (
        "no datasets/blue-marble-next-gen (or reference/... variants) directory "
        f"found. Set ${DATA_ROOT_ENV} or pass an explicit root."
    )
    raise FileNotFoundError(msg)
