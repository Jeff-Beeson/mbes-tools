#!/usr/bin/env python3
"""Generate a sector/angle backscatter correction table from Kongsberg .kmall files.

Binary parsing of #MRZ datagrams is handled by :mod:`mbes_tools.kmall`
(``iter_mrz_datagrams``). This module owns the application-layer logic:
geometry masking, ping QC, flat-seafloor filtering, angle binning, aggregation,
and CSV export. The survey-agnostic spatial filters live in
:mod:`mbes_tools.backscatter.qc`.

Output CSV columns:

    depthMode,rxFanIndex,sector,beam_angle_deg,avgIntensity_dB,fineTunedCorrection_dB

Tier 1 / Tier 2 quality controls:

    --geometry-mask-grid          survey-scale keep/reject mask
    --slope-value-grid            sample slope values into QA columns
    --bathy-sd-value-grid         sample MB-System SD/roughness values into QA columns
    --min-ping-valid-fraction     reject likely bubble/wipeout pings
    --max-port-starboard-diff-db  reject asymmetric/partially wiped pings
    --flat-filter                 keep only soundings over locally flat seafloor

DATA_ROOT:
    If python-dotenv is installed, a ``.env`` file with ``DATA_ROOT`` sets the
    default input path. ``input`` overrides it explicitly.

Notes:
- KMALL depthMode values >= 100 indicate a manually selected depth mode.
  By default this subtracts 100, so rawDepthMode 106 becomes depthMode 6.
- txSectorNumb is written exactly as stored in KMALL (zero-based).
  The calibration-file updater adds +1 to sector.
- reflectivity2_dB is the default intensity source (corrected backscatter in MRZ).
- The flat filter uses each MRZ datagram's local x/y/z sounding coordinates;
  no global projection is required for that step.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional, Tuple

from mbes_tools.kmall import MRZDatagram, iter_kmall_files, iter_mrz_datagrams
from mbes_tools.backscatter.qc import (
    AsciiGridMask,
    GeometryMaskFilter,
    GridValueSampler,
    SpatialProjector,
    apply_flat_filter_to_ping,
    filter_records_by_geometry_mask,
    filter_records_by_sampled_grid_thresholds,
    parse_keep_values,
)


# ---------------------------------------------------------------------------
# Sounding and aggregation data structures.
# ---------------------------------------------------------------------------


@dataclass
class SoundingRecord:
    depth_mode: int
    rx_fan_index: int
    sector: int
    angle: float
    intensity: float
    raw_depth_mode: int
    x_m: float
    y_m: float
    z_m: float
    easting_m: Optional[float] = None
    northing_m: Optional[float] = None
    local_slope_deg: Optional[float] = None
    local_roughness_m: Optional[float] = None
    local_bs_std_db: Optional[float] = None
    grid_slope_deg: Optional[float] = None
    grid_bathy_sd_m: Optional[float] = None


@dataclass
class Agg:
    count: int = 0
    sum_intensity: float = 0.0
    sumsq_intensity: float = 0.0
    sum_slope_deg: float = 0.0
    slope_count: int = 0
    sum_roughness_m: float = 0.0
    roughness_count: int = 0
    sum_bs_std_db: float = 0.0
    bs_std_count: int = 0
    sum_grid_slope_deg: float = 0.0
    grid_slope_count: int = 0
    sum_grid_bathy_sd_m: float = 0.0
    grid_bathy_sd_count: int = 0

    def add(
        self,
        value: float,
        local_slope_deg: Optional[float] = None,
        local_roughness_m: Optional[float] = None,
        local_bs_std_db: Optional[float] = None,
        grid_slope_deg: Optional[float] = None,
        grid_bathy_sd_m: Optional[float] = None,
    ) -> None:
        self.count += 1
        self.sum_intensity += value
        self.sumsq_intensity += value * value

        if local_slope_deg is not None and math.isfinite(local_slope_deg):
            self.sum_slope_deg += local_slope_deg
            self.slope_count += 1

        if local_roughness_m is not None and math.isfinite(local_roughness_m):
            self.sum_roughness_m += local_roughness_m
            self.roughness_count += 1

        if local_bs_std_db is not None and math.isfinite(local_bs_std_db):
            self.sum_bs_std_db += local_bs_std_db
            self.bs_std_count += 1

        if grid_slope_deg is not None and math.isfinite(grid_slope_deg):
            self.sum_grid_slope_deg += grid_slope_deg
            self.grid_slope_count += 1

        if grid_bathy_sd_m is not None and math.isfinite(grid_bathy_sd_m):
            self.sum_grid_bathy_sd_m += grid_bathy_sd_m
            self.grid_bathy_sd_count += 1

    @property
    def avg(self) -> float:
        return self.sum_intensity / self.count if self.count else float("nan")

    @property
    def std(self) -> float:
        if self.count < 2:
            return float("nan")
        variance = (
            self.sumsq_intensity
            - (self.sum_intensity * self.sum_intensity / self.count)
        ) / (self.count - 1)
        return math.sqrt(max(variance, 0.0))

    @property
    def mean_slope_deg(self) -> float:
        return self.sum_slope_deg / self.slope_count if self.slope_count else float("nan")

    @property
    def mean_roughness_m(self) -> float:
        return (
            self.sum_roughness_m / self.roughness_count
            if self.roughness_count
            else float("nan")
        )

    @property
    def mean_local_bs_std_db(self) -> float:
        return (
            self.sum_bs_std_db / self.bs_std_count if self.bs_std_count else float("nan")
        )

    @property
    def mean_grid_slope_deg(self) -> float:
        return (
            self.sum_grid_slope_deg / self.grid_slope_count
            if self.grid_slope_count
            else float("nan")
        )

    @property
    def mean_grid_bathy_sd_m(self) -> float:
        return (
            self.sum_grid_bathy_sd_m / self.grid_bathy_sd_count
            if self.grid_bathy_sd_count
            else float("nan")
        )


# ---------------------------------------------------------------------------
# Angle binning.
# ---------------------------------------------------------------------------


def bin_angle(angle_deg: float, bin_size_deg: float) -> float:
    """Round angle_deg to the nearest bin_size_deg boundary."""
    if bin_size_deg <= 0:
        raise ValueError("angle bin size must be positive")
    binned = round(angle_deg / bin_size_deg) * bin_size_deg
    if abs(bin_size_deg - round(bin_size_deg)) < 1e-9:
        return int(round(binned))
    return round(binned, 6)


# ---------------------------------------------------------------------------
# Per-datagram processing: extract SoundingRecords and apply ping-level QC.
# ---------------------------------------------------------------------------


def process_mrz_datagram(
    dgm: MRZDatagram,
    reflectivity_field: str,
    angle_bin_size: float,
    valid_only: bool,
    min_depth: Optional[float],
    max_depth: Optional[float],
    spatial_projector: Optional[SpatialProjector],
    slope_sampler: Optional[GridValueSampler],
    bathy_sd_sampler: Optional[GridValueSampler],
    min_ping_valid_fraction: Optional[float],
    min_ping_valid_soundings: Optional[int],
    min_ping_median_intensity_db: Optional[float],
    max_ping_intensity_std_db: Optional[float],
    max_port_starboard_diff_db: Optional[float],
    min_port_starboard_soundings: int,
    min_ping_angle_coverage_deg: Optional[float],
    ping_filter_stats: Optional[Dict[str, int]],
) -> List[SoundingRecord]:
    """Convert one MRZDatagram into filtered SoundingRecords.

    Parser fields come from mbes_tools.kmall.MRZDatagram / MRZSounding.
    Geometry projection, grid sampling, and ping-level QC are applied here.
    """
    records: List[SoundingRecord] = []
    n_soundings = len(dgm.soundings)

    vessel_easting_m: Optional[float] = None
    vessel_northing_m: Optional[float] = None
    if (
        spatial_projector is not None
        and dgm.latitude_deg is not None
        and dgm.longitude_deg is not None
        and math.isfinite(dgm.latitude_deg)
        and math.isfinite(dgm.longitude_deg)
    ):
        try:
            vessel_easting_m, vessel_northing_m = spatial_projector.project_lonlat(
                dgm.longitude_deg, dgm.latitude_deg
            )
        except Exception:
            vessel_easting_m = vessel_northing_m = None

    for s in dgm.soundings:
        if valid_only and not s.is_valid:
            continue
        if min_depth is not None and s.z_m < min_depth:
            continue
        if max_depth is not None and s.z_m > max_depth:
            continue

        intensity = (
            s.reflectivity2_db
            if reflectivity_field == "reflectivity2_dB"
            else s.reflectivity1_db
        )
        if not math.isfinite(intensity) or intensity <= -99.0:
            continue
        if not math.isfinite(s.beam_angle_re_rx_deg):
            continue
        if not (math.isfinite(s.x_m) and math.isfinite(s.y_m) and math.isfinite(s.z_m)):
            continue

        angle = bin_angle(s.beam_angle_re_rx_deg, angle_bin_size)

        easting_m: Optional[float] = None
        northing_m: Optional[float] = None
        if (
            spatial_projector is not None
            and vessel_easting_m is not None
            and vessel_northing_m is not None
        ):
            easting_m, northing_m = spatial_projector.local_offsets_to_projected_xy(
                vessel_easting_m,
                vessel_northing_m,
                dgm.heading_vessel_deg,
                s.x_m,
                s.y_m,
            )

        grid_slope_deg = (
            slope_sampler.sample_xy(easting_m, northing_m) if slope_sampler is not None else None
        )
        grid_bathy_sd_m = (
            bathy_sd_sampler.sample_xy(easting_m, northing_m)
            if bathy_sd_sampler is not None
            else None
        )

        records.append(
            SoundingRecord(
                depth_mode=dgm.depth_mode,
                rx_fan_index=dgm.rx_fan_index,
                sector=s.tx_sector_numb,
                angle=angle,
                intensity=intensity,
                raw_depth_mode=dgm.raw_depth_mode,
                x_m=s.x_m,
                y_m=s.y_m,
                z_m=s.z_m,
                easting_m=easting_m,
                northing_m=northing_m,
                grid_slope_deg=grid_slope_deg,
                grid_bathy_sd_m=grid_bathy_sd_m,
            )
        )

    # --- Ping-level QC (bubble sweep-down / seabed-image wipeout) -----------

    if ping_filter_stats is not None:
        ping_filter_stats["pings_seen"] += 1
        ping_filter_stats["candidate_soundings_before_ping_filter"] += len(records)

    rejection_reason: Optional[str] = None
    valid_fraction = len(records) / n_soundings if n_soundings > 0 else 0.0

    if min_ping_valid_fraction is not None and valid_fraction < min_ping_valid_fraction:
        rejection_reason = "min_valid_fraction"
    elif min_ping_valid_soundings is not None and len(records) < min_ping_valid_soundings:
        rejection_reason = "min_valid_soundings"
    elif records:
        intensities = [r.intensity for r in records]
        med_intensity = median(intensities)

        if (
            min_ping_median_intensity_db is not None
            and med_intensity < min_ping_median_intensity_db
        ):
            rejection_reason = "low_median_intensity"
        elif max_ping_intensity_std_db is not None and len(intensities) > 1:
            mu = sum(intensities) / len(intensities)
            std = math.sqrt(
                sum((v - mu) * (v - mu) for v in intensities) / (len(intensities) - 1)
            )
            if std > max_ping_intensity_std_db:
                rejection_reason = "high_intensity_std"

        if rejection_reason is None and max_port_starboard_diff_db is not None:
            port = [r.intensity for r in records if r.y_m < 0]
            starboard = [r.intensity for r in records if r.y_m >= 0]
            if (
                len(port) >= min_port_starboard_soundings
                and len(starboard) >= min_port_starboard_soundings
            ):
                if abs(median(port) - median(starboard)) > max_port_starboard_diff_db:
                    rejection_reason = "port_starboard_diff"

        if rejection_reason is None and min_ping_angle_coverage_deg is not None:
            angles = [r.angle for r in records]
            if angles and (max(angles) - min(angles)) < min_ping_angle_coverage_deg:
                rejection_reason = "low_angle_coverage"
    elif min_ping_valid_soundings is not None or min_ping_valid_fraction is not None:
        rejection_reason = "no_valid_soundings"

    if rejection_reason is not None:
        if ping_filter_stats is not None:
            ping_filter_stats["pings_rejected"] += 1
            ping_filter_stats[f"pings_rejected_{rejection_reason}"] += 1
        records = []
    else:
        if ping_filter_stats is not None:
            ping_filter_stats["pings_used"] += 1
            ping_filter_stats["candidate_soundings_after_ping_filter"] += len(records)

    return records


# ---------------------------------------------------------------------------
# Aggregation helpers.
# ---------------------------------------------------------------------------


def aggregate_records(
    records: List[SoundingRecord],
    agg: Dict[Tuple[int, int, int, float], Agg],
    raw_depth_modes: Dict[Tuple[int, int, int, float], set],
) -> int:
    for rec in records:
        key = (rec.depth_mode, rec.rx_fan_index, rec.sector, rec.angle)
        agg[key].add(
            rec.intensity,
            local_slope_deg=rec.local_slope_deg,
            local_roughness_m=rec.local_roughness_m,
            local_bs_std_db=rec.local_bs_std_db,
            grid_slope_deg=rec.grid_slope_deg,
            grid_bathy_sd_m=rec.grid_bathy_sd_m,
        )
        raw_depth_modes[key].add(rec.raw_depth_mode)
    return len(records)


def add_pre_filter_counts(
    records: List[SoundingRecord],
    counts: Dict[Tuple[int, int, int, float], int],
) -> None:
    for rec in records:
        key = (rec.depth_mode, rec.rx_fan_index, rec.sector, rec.angle)
        counts[key] += 1


# ---------------------------------------------------------------------------
# File accumulation — iterates datagrams via mbes_tools.kmall.
# ---------------------------------------------------------------------------


def accumulate_file(
    kmall_file: Path,
    agg: Dict[Tuple[int, int, int, float], Agg],
    raw_depth_modes: Dict[Tuple[int, int, int, float], set],
    pre_geometry_counts: Dict[Tuple[int, int, int, float], int],
    pre_flat_counts: Dict[Tuple[int, int, int, float], int],
    reflectivity_field: str,
    angle_bin_size: float,
    normalize_manual_depth_modes: bool,
    valid_only: bool,
    min_depth: Optional[float],
    max_depth: Optional[float],
    geometry_filter: Optional[GeometryMaskFilter],
    spatial_projector: Optional[SpatialProjector],
    slope_sampler: Optional[GridValueSampler],
    bathy_sd_sampler: Optional[GridValueSampler],
    slope_max_deg: Optional[float],
    bathy_sd_max_m: Optional[float],
    min_ping_valid_fraction: Optional[float],
    min_ping_valid_soundings: Optional[int],
    min_ping_median_intensity_db: Optional[float],
    max_ping_intensity_std_db: Optional[float],
    max_port_starboard_diff_db: Optional[float],
    min_port_starboard_soundings: int,
    min_ping_angle_coverage_deg: Optional[float],
    ping_filter_stats: Dict[str, int],
    flat_filter: bool,
    flat_radius_m: float,
    flat_min_neighbors: int,
    flat_max_slope_deg: float,
    flat_max_roughness_m: float,
    flat_max_bs_std_db: Optional[float],
) -> Tuple[int, int, int, int]:
    """Accumulate one .kmall file into the aggregation dictionaries.

    Returns:
        (mrz_count, soundings_before_geometry, soundings_after_geometry, soundings_used)
    """
    mrz_count = 0
    sounding_count_before_geometry = 0
    sounding_count_after_geometry = 0
    sounding_count_used = 0

    for dgm in iter_mrz_datagrams(
        kmall_file, normalize_manual_depth_modes=normalize_manual_depth_modes
    ):
        mrz_count += 1

        records = process_mrz_datagram(
            dgm=dgm,
            reflectivity_field=reflectivity_field,
            angle_bin_size=angle_bin_size,
            valid_only=valid_only,
            min_depth=min_depth,
            max_depth=max_depth,
            spatial_projector=spatial_projector,
            slope_sampler=slope_sampler,
            bathy_sd_sampler=bathy_sd_sampler,
            min_ping_valid_fraction=min_ping_valid_fraction,
            min_ping_valid_soundings=min_ping_valid_soundings,
            min_ping_median_intensity_db=min_ping_median_intensity_db,
            max_ping_intensity_std_db=max_ping_intensity_std_db,
            max_port_starboard_diff_db=max_port_starboard_diff_db,
            min_port_starboard_soundings=min_port_starboard_soundings,
            min_ping_angle_coverage_deg=min_ping_angle_coverage_deg,
            ping_filter_stats=ping_filter_stats,
        )

        add_pre_filter_counts(records, pre_geometry_counts)
        sounding_count_before_geometry += len(records)

        if geometry_filter is not None:
            records = filter_records_by_geometry_mask(records, geometry_filter)

        records = filter_records_by_sampled_grid_thresholds(
            records, slope_max_deg=slope_max_deg, bathy_sd_max_m=bathy_sd_max_m
        )

        add_pre_filter_counts(records, pre_flat_counts)
        sounding_count_after_geometry += len(records)

        if flat_filter:
            records = apply_flat_filter_to_ping(
                records=records,
                radius_m=flat_radius_m,
                min_neighbors=flat_min_neighbors,
                max_slope_deg=flat_max_slope_deg,
                max_roughness_m=flat_max_roughness_m,
                max_bs_std_db=flat_max_bs_std_db,
            )

        sounding_count_used += aggregate_records(records, agg, raw_depth_modes)

    return (
        mrz_count,
        sounding_count_before_geometry,
        sounding_count_after_geometry,
        sounding_count_used,
    )


# ---------------------------------------------------------------------------
# Table building and CSV export.
# ---------------------------------------------------------------------------


def reference_key(row: dict, reference_group: str) -> Tuple:
    if reference_group == "sector":
        return (row["depthMode"], row["rxFanIndex"], row["sector"])
    if reference_group == "mode_fan":
        return (row["depthMode"], row["rxFanIndex"])
    if reference_group == "mode":
        return (row["depthMode"],)
    if reference_group == "global":
        return tuple()
    raise ValueError(f"Unknown reference group: {reference_group}")


def build_rows(
    agg: Dict[Tuple[int, int, int, float], Agg],
    raw_depth_modes: Dict[Tuple[int, int, int, float], set],
    pre_geometry_counts: Dict[Tuple[int, int, int, float], int],
    pre_flat_counts: Dict[Tuple[int, int, int, float], int],
    min_soundings: int,
    reference_group: str,
    reference_stat: str,
    max_abs_correction: Optional[float],
    flat_filter_used: bool,
) -> List[dict]:
    rows: List[dict] = []

    for (depth_mode, rx_fan_index, sector, angle), a in sorted(agg.items()):
        if a.count < min_soundings:
            continue

        key = (depth_mode, rx_fan_index, sector, angle)
        raw_modes = sorted(raw_depth_modes[key])
        rows.append(
            {
                "depthMode": depth_mode,
                "rxFanIndex": rx_fan_index,
                "sector": sector,
                "beam_angle_deg": angle,
                "avgIntensity_dB": a.avg,
                "n_soundings": a.count,
                "n_soundings_before_geometry_filter": pre_geometry_counts.get(key, a.count),
                "n_soundings_before_flat_filter": pre_flat_counts.get(key, a.count),
                "stdIntensity_dB": a.std,
                "meanLocalSlope_deg": a.mean_slope_deg if flat_filter_used else float("nan"),
                "meanLocalRoughness_m": a.mean_roughness_m if flat_filter_used else float("nan"),
                "meanLocalBSStd_dB": a.mean_local_bs_std_db if flat_filter_used else float("nan"),
                "meanGridSlope_deg": a.mean_grid_slope_deg,
                "meanGridBathySD_m": a.mean_grid_bathy_sd_m,
                "rawDepthMode": ";".join(str(x) for x in raw_modes),
            }
        )

    if not rows:
        return []

    ref_values: Dict[Tuple, List[float]] = defaultdict(list)
    for row in rows:
        ref_values[reference_key(row, reference_group)].append(row["avgIntensity_dB"])

    references: Dict[Tuple, float] = {}
    for key, values in ref_values.items():
        if reference_stat == "median":
            references[key] = median(values)
        elif reference_stat == "mean":
            references[key] = mean(values)
        else:
            raise ValueError(f"Unknown reference statistic: {reference_stat}")

    for row in rows:
        ref = references[reference_key(row, reference_group)]
        correction = ref - row["avgIntensity_dB"]
        if max_abs_correction is not None:
            correction = max(-max_abs_correction, min(max_abs_correction, correction))
        row["referenceIntensity_dB"] = ref
        row["fineTunedCorrection_dB"] = correction

    return rows


def write_csv(rows: List[dict], output_csv: Path, flat_filter_used: bool) -> None:
    fieldnames = [
        "depthMode",
        "rxFanIndex",
        "sector",
        "beam_angle_deg",
        "avgIntensity_dB",
        "fineTunedCorrection_dB",
        "n_soundings",
        "n_soundings_before_geometry_filter",
        "n_soundings_before_flat_filter",
        "stdIntensity_dB",
        "referenceIntensity_dB",
        "meanLocalSlope_deg",
        "meanLocalRoughness_m",
        "meanLocalBSStd_dB",
        "meanGridSlope_deg",
        "meanGridBathySD_m",
        "rawDepthMode",
    ]
    float_fields = {
        "avgIntensity_dB",
        "fineTunedCorrection_dB",
        "stdIntensity_dB",
        "referenceIntensity_dB",
        "meanLocalSlope_deg",
        "meanLocalRoughness_m",
        "meanLocalBSStd_dB",
        "meanGridSlope_deg",
        "meanGridBathySD_m",
    }

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in float_fields:
                value = out.get(key)
                if isinstance(value, float) and math.isfinite(value):
                    out[key] = f"{value:.6f}"
                elif isinstance(value, float):
                    out[key] = ""
            writer.writerow({k: out.get(k, "") for k in fieldnames})


# ---------------------------------------------------------------------------
# CLI entry point.
# ---------------------------------------------------------------------------


def _load_data_root_default() -> Optional[str]:
    """Return DATA_ROOT from the environment, loading a .env if dotenv exists."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    return os.environ.get("DATA_ROOT")


def main() -> None:
    data_root = _load_data_root_default()

    parser = argparse.ArgumentParser(
        description=(
            "Generate a sector/angle backscatter correction table from Kongsberg .kmall files. "
            "Set DATA_ROOT in .env to establish the default input directory."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=data_root,
        help=(
            "Input .kmall/.km file or directory. "
            f"Defaults to DATA_ROOT from .env (currently: {data_root!r})"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default="SectorAngleIntensity_Table_from_KMALL.csv",
        help="Output CSV path (default: SectorAngleIntensity_Table_from_KMALL.csv)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search input directory for .kmall/.km files",
    )
    parser.add_argument(
        "--reflectivity-field",
        choices=["reflectivity2_dB", "reflectivity1_dB"],
        default="reflectivity2_dB",
        help="MRZ sounding field to average (default: reflectivity2_dB)",
    )
    parser.add_argument(
        "--angle-bin-deg",
        type=float,
        default=1.0,
        help="Beam-angle bin size in degrees (default: 1)",
    )
    parser.add_argument(
        "--min-soundings",
        type=int,
        default=25,
        help="Minimum soundings required per output bin (default: 25)",
    )
    parser.add_argument(
        "--reference-group",
        choices=["sector", "mode_fan", "mode", "global"],
        default="mode_fan",
        help=(
            "Group used to define the target/reference intensity. "
            "mode_fan = normalize all sectors/angles within each depthMode/rxFanIndex (default)"
        ),
    )
    parser.add_argument(
        "--reference-stat",
        choices=["median", "mean"],
        default="median",
        help="Statistic used for reference intensity (default: median)",
    )
    parser.add_argument(
        "--keep-manual-depth-mode-offset",
        action="store_true",
        help="Do not subtract 100 from manual KMALL depthMode values such as 106",
    )
    parser.add_argument(
        "--include-invalid-detections",
        action="store_true",
        help="Include rejected/no-detect soundings (not recommended for backscatter correction)",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=None,
        help="Optional minimum depth filter in meters",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=None,
        help="Optional maximum depth filter in meters",
    )
    parser.add_argument(
        "--max-abs-correction",
        type=float,
        default=None,
        help="Optional clipping threshold for absolute correction value in dB (e.g. 12)",
    )

    # Geometry mask.
    parser.add_argument(
        "--geometry-mask-grid",
        default=None,
        help=(
            "Optional Arc/ESRI ASCII grid geometry mask. "
            "Convert GeoTIFF masks first with: gdal_translate -of AAIGrid mask.tif mask.asc"
        ),
    )
    parser.add_argument(
        "--geometry-mask-crs",
        default="EPSG:32610",
        help="CRS of the geometry mask grid (default: EPSG:32610 / UTM zone 10N)",
    )
    parser.add_argument(
        "--geometry-mask-keep-values",
        default="1",
        help="Comma-separated mask values to keep (default: 1)",
    )
    parser.add_argument(
        "--geometry-mask-keep-tolerance",
        type=float,
        default=0.001,
        help="Tolerance for matching geometry mask keep values (default: 0.001)",
    )

    # Slope and bathy-SD grids.
    parser.add_argument(
        "--slope-value-grid",
        default=None,
        help="Optional Arc/ESRI ASCII grid of slope values in degrees for QA columns",
    )
    parser.add_argument(
        "--slope-max-deg",
        type=float,
        default=None,
        help="Optional direct slope threshold in degrees (requires --slope-value-grid)",
    )
    parser.add_argument(
        "--bathy-sd-value-grid",
        default=None,
        help="Optional Arc/ESRI ASCII grid of MB-System bathymetry SD / roughness values for QA columns",
    )
    parser.add_argument(
        "--bathy-sd-max-m",
        type=float,
        default=None,
        help="Optional bathy SD/roughness threshold in meters (requires --bathy-sd-value-grid)",
    )

    # Ping-level QC filters.
    parser.add_argument(
        "--min-ping-valid-fraction",
        type=float,
        default=None,
        help="Reject pings where valid sounding fraction is below this value (e.g. 0.45)",
    )
    parser.add_argument(
        "--min-ping-valid-soundings",
        type=int,
        default=None,
        help="Reject pings with fewer than this many valid soundings after basic filters",
    )
    parser.add_argument(
        "--min-ping-median-intensity-db",
        type=float,
        default=None,
        help="Reject pings with median intensity below this absolute dB threshold",
    )
    parser.add_argument(
        "--max-ping-intensity-std-db",
        type=float,
        default=None,
        help="Reject pings with unusually high within-ping intensity standard deviation",
    )
    parser.add_argument(
        "--max-port-starboard-diff-db",
        type=float,
        default=None,
        help="Reject pings where port/starboard median intensity differs by more than this many dB",
    )
    parser.add_argument(
        "--min-port-starboard-soundings",
        type=int,
        default=25,
        help="Minimum soundings per side for port/starboard imbalance filter (default: 25)",
    )
    parser.add_argument(
        "--min-ping-angle-coverage-deg",
        type=float,
        default=None,
        help="Reject pings whose retained soundings span less than this many degrees of beam angle",
    )

    # Flat-seafloor filter.
    parser.add_argument(
        "--flat-filter",
        action="store_true",
        help="Enable local 3D plane-fit filter to use only relatively flat seafloor soundings",
    )
    parser.add_argument(
        "--flat-radius-m",
        type=float,
        default=50.0,
        help="Radius in meters for local plane fitting (default: 50)",
    )
    parser.add_argument(
        "--flat-min-neighbors",
        type=int,
        default=50,
        help="Minimum neighboring soundings inside --flat-radius-m (default: 50)",
    )
    parser.add_argument(
        "--flat-max-slope-deg",
        type=float,
        default=3.0,
        help="Maximum local plane slope in degrees (default: 3)",
    )
    parser.add_argument(
        "--flat-max-roughness-m",
        type=float,
        default=1.0,
        help="Maximum RMS residual from local plane fit in meters (default: 1.0)",
    )
    parser.add_argument(
        "--flat-max-bs-std-db",
        type=float,
        default=None,
        help="Optional maximum local backscatter standard deviation in dB (e.g. 3.0)",
    )

    args = parser.parse_args()

    if args.input is None:
        parser.error(
            "input path is required — provide it as a positional argument or set DATA_ROOT in .env"
        )

    input_path = Path(args.input).expanduser().resolve()
    output_csv = Path(args.output).expanduser().resolve()

    files = iter_kmall_files(input_path, recursive=args.recursive)
    if not files:
        raise RuntimeError(f"No .kmall or .km files found for input: {input_path}")

    agg: Dict[Tuple[int, int, int, float], Agg] = defaultdict(Agg)
    raw_depth_modes: Dict[Tuple[int, int, int, float], set] = defaultdict(set)
    pre_geometry_counts: Dict[Tuple[int, int, int, float], int] = defaultdict(int)
    pre_flat_counts: Dict[Tuple[int, int, int, float], int] = defaultdict(int)

    geometry_filter: Optional[GeometryMaskFilter] = None
    slope_sampler: Optional[GridValueSampler] = None
    bathy_sd_sampler: Optional[GridValueSampler] = None

    spatial_projector: Optional[SpatialProjector] = None
    if any(
        v is not None
        for v in [args.geometry_mask_grid, args.slope_value_grid, args.bathy_sd_value_grid]
    ):
        spatial_projector = SpatialProjector(args.geometry_mask_crs)

    if args.geometry_mask_grid is not None:
        mask_grid = AsciiGridMask.from_file(
            Path(args.geometry_mask_grid).expanduser().resolve()
        )
        geometry_filter = GeometryMaskFilter(
            mask_grid=mask_grid,
            mask_crs=args.geometry_mask_crs,
            keep_values=parse_keep_values(args.geometry_mask_keep_values),
            keep_tolerance=args.geometry_mask_keep_tolerance,
        )

    if args.slope_value_grid is not None:
        slope_sampler = GridValueSampler(
            AsciiGridMask.from_file(Path(args.slope_value_grid).expanduser().resolve())
        )

    if args.bathy_sd_value_grid is not None:
        bathy_sd_sampler = GridValueSampler(
            AsciiGridMask.from_file(Path(args.bathy_sd_value_grid).expanduser().resolve())
        )

    if args.slope_max_deg is not None and slope_sampler is None:
        raise RuntimeError("--slope-max-deg requires --slope-value-grid")
    if args.bathy_sd_max_m is not None and bathy_sd_sampler is None:
        raise RuntimeError("--bathy-sd-max-m requires --bathy-sd-value-grid")

    total_mrz = 0
    total_soundings_before_geometry = 0
    total_soundings_after_geometry = 0
    total_soundings_used = 0
    total_ping_filter_stats: Dict[str, int] = defaultdict(int)

    print(f"Found {len(files)} file(s).")
    if geometry_filter is not None:
        print(
            f"Geometry mask: grid={geometry_filter.mask_grid.path}, "
            f"crs={args.geometry_mask_crs}, keep={geometry_filter.keep_values}"
        )
    if slope_sampler is not None:
        msg = f"Slope grid: {slope_sampler.grid.path}"
        if args.slope_max_deg is not None:
            msg += f", max={args.slope_max_deg} deg"
        print(msg)
    if bathy_sd_sampler is not None:
        msg = f"Bathy SD grid: {bathy_sd_sampler.grid.path}"
        if args.bathy_sd_max_m is not None:
            msg += f", max={args.bathy_sd_max_m} m"
        print(msg)
    if any(
        v is not None
        for v in [
            args.min_ping_valid_fraction,
            args.min_ping_valid_soundings,
            args.min_ping_median_intensity_db,
            args.max_ping_intensity_std_db,
            args.max_port_starboard_diff_db,
            args.min_ping_angle_coverage_deg,
        ]
    ):
        print("Ping-level QC filters enabled")
    if args.flat_filter:
        print(
            f"Flat filter: radius={args.flat_radius_m} m, "
            f"min_neighbors={args.flat_min_neighbors}, "
            f"max_slope={args.flat_max_slope_deg} deg, "
            f"max_roughness={args.flat_max_roughness_m} m, "
            f"max_bs_std={args.flat_max_bs_std_db} dB"
        )

    for kmall_file in files:
        file_ping_filter_stats: Dict[str, int] = defaultdict(int)
        mrz_count, sounding_before_geometry, sounding_after_geometry, sounding_used = (
            accumulate_file(
                kmall_file=kmall_file,
                agg=agg,
                raw_depth_modes=raw_depth_modes,
                pre_geometry_counts=pre_geometry_counts,
                pre_flat_counts=pre_flat_counts,
                reflectivity_field=args.reflectivity_field,
                angle_bin_size=args.angle_bin_deg,
                normalize_manual_depth_modes=not args.keep_manual_depth_mode_offset,
                valid_only=not args.include_invalid_detections,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                geometry_filter=geometry_filter,
                spatial_projector=spatial_projector,
                slope_sampler=slope_sampler,
                bathy_sd_sampler=bathy_sd_sampler,
                slope_max_deg=args.slope_max_deg,
                bathy_sd_max_m=args.bathy_sd_max_m,
                min_ping_valid_fraction=args.min_ping_valid_fraction,
                min_ping_valid_soundings=args.min_ping_valid_soundings,
                min_ping_median_intensity_db=args.min_ping_median_intensity_db,
                max_ping_intensity_std_db=args.max_ping_intensity_std_db,
                max_port_starboard_diff_db=args.max_port_starboard_diff_db,
                min_port_starboard_soundings=args.min_port_starboard_soundings,
                min_ping_angle_coverage_deg=args.min_ping_angle_coverage_deg,
                ping_filter_stats=file_ping_filter_stats,
                flat_filter=args.flat_filter,
                flat_radius_m=args.flat_radius_m,
                flat_min_neighbors=args.flat_min_neighbors,
                flat_max_slope_deg=args.flat_max_slope_deg,
                flat_max_roughness_m=args.flat_max_roughness_m,
                flat_max_bs_std_db=args.flat_max_bs_std_db,
            )
        )
        total_mrz += mrz_count
        total_soundings_before_geometry += sounding_before_geometry
        total_soundings_after_geometry += sounding_after_geometry
        total_soundings_used += sounding_used
        for k, v in file_ping_filter_stats.items():
            total_ping_filter_stats[k] += v

        if geometry_filter is not None or args.flat_filter:
            print(
                f"  {kmall_file.name}: {mrz_count} MRZ, "
                f"{sounding_before_geometry} usable before geometry filter, "
                f"{sounding_after_geometry} after geometry filter, "
                f"{sounding_used} retained"
            )
            if file_ping_filter_stats.get("pings_rejected", 0):
                print(
                    f"    ping QC rejected "
                    f"{file_ping_filter_stats.get('pings_rejected', 0)} pings"
                )
        else:
            print(f"  {kmall_file.name}: {mrz_count} MRZ, {sounding_used} usable soundings")

    rows = build_rows(
        agg=agg,
        raw_depth_modes=raw_depth_modes,
        pre_geometry_counts=pre_geometry_counts,
        pre_flat_counts=pre_flat_counts,
        min_soundings=args.min_soundings,
        reference_group=args.reference_group,
        reference_stat=args.reference_stat,
        max_abs_correction=args.max_abs_correction,
        flat_filter_used=args.flat_filter,
    )
    rows.sort(key=lambda r: (r["depthMode"], r["rxFanIndex"], r["sector"], r["beam_angle_deg"]))
    write_csv(rows, output_csv, flat_filter_used=args.flat_filter)

    print("\nDone.")
    print(f"Files processed:                         {len(files)}")
    print(f"MRZ datagrams read:                      {total_mrz}")
    print(f"Usable soundings before geometry filter: {total_soundings_before_geometry}")
    print(f"Soundings after geometry filter:         {total_soundings_after_geometry}")
    print(f"Soundings used for output table:         {total_soundings_used}")
    if total_ping_filter_stats:
        print("Ping-level QC summary:")
        for key in sorted(total_ping_filter_stats):
            print(f"  {key:43s} {total_ping_filter_stats[key]}")
    print(f"Output bins written:                     {len(rows)}")
    print(f"Output CSV:                              {output_csv}")


if __name__ == "__main__":
    main()
