#!/usr/bin/env python3
"""Generate a sector/angle backscatter correction table from Kongsberg .all files.

This is the ``.all`` analog of :mod:`mbes_tools.backscatter.table` (which reads
.kmall #MRZ datagrams). It produces the **same** :class:`SoundingRecord`s and
feeds the **same** aggregation, QC, reference, and CSV-export machinery, so
EM2040 ``.all`` and EM124 ``.kmall`` share one backscatter pipeline.

Where the data comes from per ping (joined by ping counter):

- ``X`` (XYZ88 depth) — depth/across/along (x/y/z) and per-beam reflectivity.
- ``N`` (raw range and angle 78) — per-beam pointing angle and transmit sector.
- ``R`` (runtime 82) — the depth/ping mode (model-aware; EM2040 is frequency).
- ``P`` (position) — vessel lat/lon for optional projection.

``N`` and ``R`` arrive less often / in their own datagrams, so the file reader
keeps the most recent ``R`` mode and ``P`` position as state and pairs each
``X`` with the matching ``N`` by ping counter.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mbes_tools.all import (
    DepthDatagram,
    RawRangeAngleDatagram,
    iter_datagrams,
    parse_depth,
    parse_position,
    parse_raw_range_angle,
    parse_runtime,
)
from mbes_tools.depth_modes import all_runtime_mode_info
from mbes_tools.backscatter.qc import (
    GeometryMaskFilter,
    GridValueSampler,
    SpatialProjector,
    apply_flat_filter_to_ping,
    filter_records_by_geometry_mask,
    filter_records_by_sampled_grid_thresholds,
)
from mbes_tools.backscatter.table import (
    SoundingRecord,
    Agg,
    aggregate_records,
    add_pre_filter_counts,
    apply_ping_qc,
    bin_angle,
)

# Datagram types the .all backscatter table needs; everything else is skipped.
_NEEDED_TYPES = {"R", "N", "X", "P"}

# Reflectivity sources (Capability A). Capability B adds the Y seabed-image
# sample reduction as another source.
REFLECTIVITY_SOURCES = ("xyz88", "rawrange78")


def process_all_ping(
    depth: DepthDatagram,
    raw_range: RawRangeAngleDatagram,
    mode_byte: Optional[int],
    em_model: int,
    *,
    latitude_deg: Optional[float],
    longitude_deg: Optional[float],
    reflectivity_source: str,
    angle_bin_size: float,
    valid_only: bool,
    min_depth: Optional[float],
    max_depth: Optional[float],
    spatial_projector: Optional[SpatialProjector],
    slope_sampler: Optional[GridValueSampler],
    bathy_sd_sampler: Optional[GridValueSampler],
    rx_fan_index: int,
    min_ping_valid_fraction: Optional[float],
    min_ping_valid_soundings: Optional[int],
    min_ping_median_intensity_db: Optional[float],
    max_ping_intensity_std_db: Optional[float],
    max_port_starboard_diff_db: Optional[float],
    min_port_starboard_soundings: int,
    min_ping_angle_coverage_deg: Optional[float],
    ping_filter_stats: Optional[Dict[str, int]],
) -> List[SoundingRecord]:
    """Convert one paired ``X``+``N`` ping into filtered :class:`SoundingRecord`s.

    Angle and transmit sector come from the ``N`` beam; depth coordinates and
    reflectivity from the ``X`` beam (or ``N`` reflectivity if
    ``reflectivity_source == "rawrange78"``). ``mode_byte`` is the most recent
    ``R`` runtime mode; if ``None`` (no runtime seen yet) the ping is skipped.
    """
    if mode_byte is None:
        if ping_filter_stats is not None:
            ping_filter_stats["pings_skipped_no_runtime"] += 1
        return []

    depth_mode, _label = all_runtime_mode_info(em_model, mode_byte)

    n_beams = min(len(depth.beams), len(raw_range.beams))
    records: List[SoundingRecord] = []

    vessel_easting_m: Optional[float] = None
    vessel_northing_m: Optional[float] = None
    if (
        spatial_projector is not None
        and latitude_deg is not None
        and longitude_deg is not None
        and math.isfinite(latitude_deg)
        and math.isfinite(longitude_deg)
    ):
        try:
            vessel_easting_m, vessel_northing_m = spatial_projector.project_lonlat(
                longitude_deg, latitude_deg
            )
        except Exception:
            vessel_easting_m = vessel_northing_m = None

    for i in range(n_beams):
        xb = depth.beams[i]
        nb = raw_range.beams[i]

        if valid_only and not xb.is_valid:
            continue

        z_m = xb.depth_m
        x_m = xb.along_track_m
        y_m = xb.across_track_m

        if min_depth is not None and z_m < min_depth:
            continue
        if max_depth is not None and z_m > max_depth:
            continue

        intensity = (
            xb.reflectivity_db
            if reflectivity_source == "xyz88"
            else nb.reflectivity_db
        )
        if not math.isfinite(intensity) or intensity <= -99.0:
            continue

        angle_raw = nb.beam_pointing_angle_deg
        if not math.isfinite(angle_raw):
            continue
        if not (math.isfinite(x_m) and math.isfinite(y_m) and math.isfinite(z_m)):
            continue

        angle = bin_angle(angle_raw, angle_bin_size)

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
                depth.heading_deg,
                x_m,
                y_m,
            )

        grid_slope_deg = (
            slope_sampler.sample_xy(easting_m, northing_m)
            if slope_sampler is not None
            else None
        )
        grid_bathy_sd_m = (
            bathy_sd_sampler.sample_xy(easting_m, northing_m)
            if bathy_sd_sampler is not None
            else None
        )

        records.append(
            SoundingRecord(
                depth_mode=depth_mode,
                rx_fan_index=rx_fan_index,
                sector=nb.tx_sector_number,
                angle=angle,
                intensity=intensity,
                raw_depth_mode=int(mode_byte),
                x_m=x_m,
                y_m=y_m,
                z_m=z_m,
                easting_m=easting_m,
                northing_m=northing_m,
                grid_slope_deg=grid_slope_deg,
                grid_bathy_sd_m=grid_bathy_sd_m,
            )
        )

    return apply_ping_qc(
        records,
        n_beams,
        min_ping_valid_fraction=min_ping_valid_fraction,
        min_ping_valid_soundings=min_ping_valid_soundings,
        min_ping_median_intensity_db=min_ping_median_intensity_db,
        max_ping_intensity_std_db=max_ping_intensity_std_db,
        max_port_starboard_diff_db=max_port_starboard_diff_db,
        min_port_starboard_soundings=min_port_starboard_soundings,
        min_ping_angle_coverage_deg=min_ping_angle_coverage_deg,
        ping_filter_stats=ping_filter_stats,
    )


def accumulate_all_file(
    all_file: Path,
    agg: Dict[Tuple[int, int, int, float], Agg],
    raw_depth_modes: Dict[Tuple[int, int, int, float], set],
    pre_geometry_counts: Dict[Tuple[int, int, int, float], int],
    pre_flat_counts: Dict[Tuple[int, int, int, float], int],
    *,
    reflectivity_source: str,
    angle_bin_size: float,
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
    """Accumulate one .all file, pairing X and N by ping counter.

    Returns ``(ping_count, before_geometry, after_geometry, used)`` matching the
    .kmall :func:`mbes_tools.backscatter.table.accumulate_file` signature so the
    CLI can report both formats uniformly.
    """
    last_mode_byte: Optional[int] = None
    last_lat: Optional[float] = None
    last_lon: Optional[float] = None
    pending_n: Dict[int, RawRangeAngleDatagram] = {}
    pending_x: Dict[int, DepthDatagram] = {}

    ping_count = 0
    before_geometry = 0
    after_geometry = 0
    used = 0

    def handle_pair(depth: DepthDatagram, raw_range: RawRangeAngleDatagram) -> None:
        nonlocal ping_count, before_geometry, after_geometry, used
        ping_count += 1
        records = process_all_ping(
            depth,
            raw_range,
            last_mode_byte,
            depth.header.em_model,
            latitude_deg=last_lat,
            longitude_deg=last_lon,
            reflectivity_source=reflectivity_source,
            angle_bin_size=angle_bin_size,
            valid_only=valid_only,
            min_depth=min_depth,
            max_depth=max_depth,
            spatial_projector=spatial_projector,
            slope_sampler=slope_sampler,
            bathy_sd_sampler=bathy_sd_sampler,
            rx_fan_index=0,
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
        before_geometry += len(records)

        if geometry_filter is not None:
            records = filter_records_by_geometry_mask(records, geometry_filter)

        records = filter_records_by_sampled_grid_thresholds(
            records, slope_max_deg=slope_max_deg, bathy_sd_max_m=bathy_sd_max_m
        )

        add_pre_filter_counts(records, pre_flat_counts)
        after_geometry += len(records)

        if flat_filter:
            records = apply_flat_filter_to_ping(
                records=records,
                radius_m=flat_radius_m,
                min_neighbors=flat_min_neighbors,
                max_slope_deg=flat_max_slope_deg,
                max_roughness_m=flat_max_roughness_m,
                max_bs_std_db=flat_max_bs_std_db,
            )

        used += aggregate_records(records, agg, raw_depth_modes)

    for rec in iter_datagrams(all_file, types=_NEEDED_TYPES):
        t = rec.header.type_of_datagram
        if t == "R":
            last_mode_byte = parse_runtime(rec).mode
        elif t == "P":
            p = parse_position(rec)
            last_lat, last_lon = p.latitude_deg, p.longitude_deg
        elif t == "N":
            n = parse_raw_range_angle(rec)
            x = pending_x.pop(n.ping_counter, None)
            if x is not None:
                handle_pair(x, n)
            else:
                pending_n[n.ping_counter] = n
        elif t == "X":
            x = parse_depth(rec)
            n = pending_n.pop(x.counter, None)
            if n is not None:
                handle_pair(x, n)
            else:
                pending_x[x.counter] = x

    return ping_count, before_geometry, after_geometry, used
