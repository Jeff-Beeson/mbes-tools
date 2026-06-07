"""Quality-control filters for sector/angle backscatter table generation.

These are survey-agnostic spatial filters used to decide which soundings
contribute to a backscatter correction table:

- :class:`AsciiGridMask` reads Arc/ESRI ASCII grids (keep/reject masks and
  value grids such as slope or bathymetry standard deviation).
- :class:`GeometryMaskFilter` keeps soundings that fall inside a survey-scale
  mask polygon.
- :class:`SpatialProjector` projects vessel lon/lat into a grid CRS and rotates
  Kongsberg local sounding offsets into projected easting/northing.
- :class:`GridValueSampler` samples a value grid at a sounding location.
- :func:`apply_flat_filter_to_ping` keeps only soundings over locally flat
  seafloor via a per-ping 3D plane fit.

Heavy scientific dependencies (numpy / scipy / pyproj) are imported lazily so
that ``import mbes_tools.backscatter.qc`` stays cheap and only the filters that
are actually used pull their dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from mbes_tools.backscatter.table import SoundingRecord


# ---------------------------------------------------------------------------
# ASCII grid reader (Arc/ESRI format) — geometry mask and value sampling.
# ---------------------------------------------------------------------------


@dataclass
class AsciiGridMask:
    """Arc/ESRI ASCII grid with nearest-cell sampling."""

    path: Path
    ncols: int
    nrows: int
    xllcorner: float
    yllcorner: float
    cellsize: float
    nodata_value: Optional[float]
    data: "object"

    @classmethod
    def from_file(cls, path: Path) -> "AsciiGridMask":
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "ASCII grid reader requires numpy: python -m pip install numpy"
            ) from exc

        header: dict = {}
        header_lines: List[str] = []
        with path.open("r") as f:
            for _ in range(6):
                line = f.readline()
                if not line:
                    raise RuntimeError(f"Unexpected end of ASCII grid header in {path}")
                header_lines.append(line)
                parts = line.strip().split()
                if len(parts) < 2:
                    raise RuntimeError(f"Invalid ASCII grid header line: {line!r}")
                header[parts[0].lower()] = parts[1]

        for key in ("ncols", "nrows", "cellsize"):
            if key not in header:
                raise RuntimeError(f"ASCII grid {path} missing required header field: {key}")

        if "xllcorner" in header:
            xll = float(header["xllcorner"])
        elif "xllcenter" in header:
            xll = float(header["xllcenter"]) - 0.5 * float(header["cellsize"])
        else:
            raise RuntimeError(f"ASCII grid {path} missing xllcorner/xllcenter")

        if "yllcorner" in header:
            yll = float(header["yllcorner"])
        elif "yllcenter" in header:
            yll = float(header["yllcenter"]) - 0.5 * float(header["cellsize"])
        else:
            raise RuntimeError(f"ASCII grid {path} missing yllcorner/yllcenter")

        nodata = float(header["nodata_value"]) if "nodata_value" in header else None
        ncols = int(header["ncols"])
        nrows = int(header["nrows"])
        cellsize = float(header["cellsize"])

        data = np.loadtxt(path, skiprows=len(header_lines))
        if data.shape != (nrows, ncols):
            raise RuntimeError(
                f"ASCII grid data shape mismatch for {path}: "
                f"header says {nrows}x{ncols}, data are {data.shape}"
            )

        return cls(
            path=path,
            ncols=ncols,
            nrows=nrows,
            xllcorner=xll,
            yllcorner=yll,
            cellsize=cellsize,
            nodata_value=nodata,
            data=data,
        )

    @property
    def xmax(self) -> float:
        return self.xllcorner + self.ncols * self.cellsize

    @property
    def ymax(self) -> float:
        return self.yllcorner + self.nrows * self.cellsize

    def sample(self, x: float, y: float) -> Optional[float]:
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        if x < self.xllcorner or x >= self.xmax or y < self.yllcorner or y >= self.ymax:
            return None
        col = int(math.floor((x - self.xllcorner) / self.cellsize))
        row_from_bottom = int(math.floor((y - self.yllcorner) / self.cellsize))
        row = self.nrows - 1 - row_from_bottom
        if row < 0 or row >= self.nrows or col < 0 or col >= self.ncols:
            return None
        value = float(self.data[row, col])
        if not math.isfinite(value):
            return None
        if self.nodata_value is not None and value == self.nodata_value:
            return None
        return value


# ---------------------------------------------------------------------------
# Geometry mask filter — keeps soundings inside a survey-scale mask polygon.
# ---------------------------------------------------------------------------


class GeometryMaskFilter:
    """Project KMALL soundings into a survey mask grid and test keep/reject."""

    def __init__(
        self,
        mask_grid: AsciiGridMask,
        mask_crs: str,
        keep_values: List[float],
        keep_tolerance: float = 0.001,
    ) -> None:
        try:
            from pyproj import Transformer
        except ImportError as exc:
            raise RuntimeError(
                "Geometry mask filter requires pyproj: python -m pip install pyproj"
            ) from exc

        self.mask_grid = mask_grid
        self.keep_values = keep_values
        self.keep_tolerance = keep_tolerance
        self.transformer = Transformer.from_crs("EPSG:4326", mask_crs, always_xy=True)

    def project_lonlat(self, longitude_deg: float, latitude_deg: float) -> Tuple[float, float]:
        return self.transformer.transform(longitude_deg, latitude_deg)

    def keep_xy(self, easting_m: Optional[float], northing_m: Optional[float]) -> bool:
        if easting_m is None or northing_m is None:
            return False
        value = self.mask_grid.sample(easting_m, northing_m)
        if value is None:
            return False
        return any(abs(value - keep) <= self.keep_tolerance for keep in self.keep_values)


def parse_keep_values(text: str) -> List[float]:
    values = [float(p.strip()) for p in text.split(",") if p.strip()]
    if not values:
        raise ValueError("At least one geometry mask keep value is required.")
    return values


def filter_records_by_geometry_mask(
    records: "List[SoundingRecord]",
    geometry_filter: GeometryMaskFilter,
) -> "List[SoundingRecord]":
    return [rec for rec in records if geometry_filter.keep_xy(rec.easting_m, rec.northing_m)]


# ---------------------------------------------------------------------------
# Spatial projector — vessel lon/lat → projected CRS + local offset rotation.
# ---------------------------------------------------------------------------


class SpatialProjector:
    """Project vessel lon/lat to a grid CRS and rotate sounding local offsets."""

    def __init__(self, grid_crs: str) -> None:
        try:
            from pyproj import Transformer
        except ImportError as exc:
            raise RuntimeError(
                "Spatial projection requires pyproj: python -m pip install pyproj"
            ) from exc
        self.transformer = Transformer.from_crs("EPSG:4326", grid_crs, always_xy=True)

    def project_lonlat(self, longitude_deg: float, latitude_deg: float) -> Tuple[float, float]:
        return self.transformer.transform(longitude_deg, latitude_deg)

    @staticmethod
    def local_offsets_to_projected_xy(
        vessel_easting_m: float,
        vessel_northing_m: float,
        heading_vessel_deg: float,
        x_forward_m: float,
        y_starboard_m: float,
    ) -> Tuple[float, float]:
        """Convert Kongsberg local (x forward, y starboard) to projected east/north.

        heading_vessel_deg is degrees clockwise from north.
        """
        heading_rad = math.radians(heading_vessel_deg)
        sin_h = math.sin(heading_rad)
        cos_h = math.cos(heading_rad)
        easting = vessel_easting_m + x_forward_m * sin_h + y_starboard_m * cos_h
        northing = vessel_northing_m + x_forward_m * cos_h - y_starboard_m * sin_h
        return easting, northing


# ---------------------------------------------------------------------------
# Grid value sampler — nearest-cell lookup for slope / bathy-SD grids.
# ---------------------------------------------------------------------------


class GridValueSampler:
    def __init__(self, grid: AsciiGridMask) -> None:
        self.grid = grid

    def sample_xy(
        self, easting_m: Optional[float], northing_m: Optional[float]
    ) -> Optional[float]:
        if easting_m is None or northing_m is None:
            return None
        return self.grid.sample(easting_m, northing_m)


def filter_records_by_sampled_grid_thresholds(
    records: "List[SoundingRecord]",
    slope_max_deg: Optional[float],
    bathy_sd_max_m: Optional[float],
) -> "List[SoundingRecord]":
    if slope_max_deg is None and bathy_sd_max_m is None:
        return records
    kept = []
    for rec in records:
        if slope_max_deg is not None:
            if (
                rec.grid_slope_deg is None
                or not math.isfinite(rec.grid_slope_deg)
                or rec.grid_slope_deg > slope_max_deg
            ):
                continue
        if bathy_sd_max_m is not None:
            if (
                rec.grid_bathy_sd_m is None
                or not math.isfinite(rec.grid_bathy_sd_m)
                or rec.grid_bathy_sd_m > bathy_sd_max_m
            ):
                continue
        kept.append(rec)
    return kept


# ---------------------------------------------------------------------------
# Flat-seafloor filter — local 3D plane fit per ping/swath.
# ---------------------------------------------------------------------------


def apply_flat_filter_to_ping(
    records: "List[SoundingRecord]",
    radius_m: float,
    min_neighbors: int,
    max_slope_deg: float,
    max_roughness_m: float,
    max_bs_std_db: Optional[float],
) -> "List[SoundingRecord]":
    """Keep soundings whose local neighborhood passes a 3D plane-fit flatness test.

    For each sounding, neighbors within radius_m in local x/y are found and a plane
    z = a*x + b*y + c is fit.  slope_deg = atan(sqrt(a^2+b^2)), roughness_m = RMS
    residual.  The sounding is retained only when all requested thresholds pass.
    """
    if not records:
        return []

    if radius_m <= 0:
        raise ValueError("flat-filter radius must be positive")
    if min_neighbors < 3:
        raise ValueError("flat-filter min neighbors must be at least 3 for a plane fit")

    try:
        import numpy as np
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError(
            "Flat-seafloor filter requires numpy and scipy: "
            "python -m pip install numpy scipy"
        ) from exc

    xy = np.array([[r.x_m, r.y_m] for r in records], dtype=float)
    z = np.array([r.z_m for r in records], dtype=float)
    bs = np.array([r.intensity for r in records], dtype=float)

    finite = np.isfinite(xy).all(axis=1) & np.isfinite(z) & np.isfinite(bs)
    if finite.sum() < min_neighbors:
        return []

    valid_indices = np.where(finite)[0]
    xy_valid = xy[valid_indices]
    z_valid = z[valid_indices]
    bs_valid = bs[valid_indices]

    tree = cKDTree(xy_valid)
    kept: "List[SoundingRecord]" = []

    for local_i, original_i in enumerate(valid_indices):
        neighbors = tree.query_ball_point(xy_valid[local_i], radius_m)
        if len(neighbors) < min_neighbors:
            continue

        neighbor_xy = xy_valid[neighbors]
        neighbor_z = z_valid[neighbors]
        neighbor_bs = bs_valid[neighbors]

        # Center x/y for numerical stability.
        xy0 = neighbor_xy.mean(axis=0)
        x_c = neighbor_xy[:, 0] - xy0[0]
        y_c = neighbor_xy[:, 1] - xy0[1]

        A = np.column_stack((x_c, y_c, np.ones(len(neighbors))))
        coeffs, *_ = np.linalg.lstsq(A, neighbor_z, rcond=None)
        a, b, _c = coeffs

        residual = neighbor_z - A @ coeffs
        slope_deg = math.degrees(math.atan(math.sqrt(float(a * a + b * b))))
        roughness_m = math.sqrt(float(np.mean(residual * residual)))
        bs_std_db = float(np.std(neighbor_bs, ddof=1)) if len(neighbor_bs) > 1 else 0.0

        if slope_deg > max_slope_deg:
            continue
        if roughness_m > max_roughness_m:
            continue
        if max_bs_std_db is not None and bs_std_db > max_bs_std_db:
            continue

        kept.append(
            replace(
                records[original_i],
                local_slope_deg=slope_deg,
                local_roughness_m=roughness_m,
                local_bs_std_db=bs_std_db,
            )
        )

    return kept
