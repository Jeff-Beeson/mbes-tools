"""Tests for the .all backscatter pipeline (process_all_ping, table, apply).

Synthetic unit tests plus integration tests against the committed real EM2040
(.all) fixture. The .all path must produce the same SoundingRecord and flow
through the same aggregation / QC / normalize / apply machinery as .kmall.
"""
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from mbes_tools.all import (
    DatagramHeader,
    DepthDatagram,
    RawRangeAngleBeam,
    RawRangeAngleDatagram,
    SeabedImageDatagram,
    XBeam,
    YBeam,
)
from mbes_tools.backscatter import normalize
from mbes_tools.backscatter.all_table import accumulate_all_file, process_all_ping
from mbes_tools.backscatter.table import Agg, build_rows, detect_format_and_files

FIXTURES = Path(__file__).parent / "fixtures"
EM2040_ALL = FIXTURES / "sample_equinox_em2040.all"
EM302_ALL = FIXTURES / "sample_nautilus.all"


def _hdr(em_model=2040):
    return DatagramHeader(
        number_of_bytes=0, stx=0x02, type_of_datagram="X",
        em_model=em_model, record_date=20190407, record_time_ms=0,
    )


def _xbeam(depth, across, refl, det_info=0):
    return XBeam(
        depth_m=depth, across_track_m=across, along_track_m=0.0,
        detection_window_length=10, quality_factor=5,
        beam_incidence_angle_adjustment_deg=0.0, detection_information=det_info,
        realtime_cleaning_information=0, reflectivity_db=refl,
    )


def _nbeam(angle, sector, refl, det_info=0):
    return RawRangeAngleBeam(
        beam_pointing_angle_deg=angle, tx_sector_number=sector, detection_info=det_info,
        detection_window_length=10, quality_factor=5, d_corr=0,
        two_way_travel_time_s=0.05, reflectivity_db=refl, realtime_cleaning_information=0,
    )


def _ping_qc_off():
    return dict(
        min_ping_valid_fraction=None, min_ping_valid_soundings=None,
        min_ping_median_intensity_db=None, max_ping_intensity_std_db=None,
        max_port_starboard_diff_db=None, min_port_starboard_soundings=25,
        min_ping_angle_coverage_deg=None,
    )


def test_process_all_ping_joins_x_and_n():
    """Angle/sector come from N; depth/reflectivity from X; invalid X beams drop."""
    x = DepthDatagram(
        header=_hdr(), counter=1, serial_number=7, heading_deg=0.0,
        sound_speed_at_transducer_m_s=1500.0, transducer_depth_m=3.0,
        num_beams=2, num_valid_detections=1, sample_frequency_hz=30000.0,
        scanning_info=0,
        beams=[_xbeam(100.0, 50.0, -25.0), _xbeam(100.0, -50.0, -30.0, det_info=0x80)],
    )
    n = RawRangeAngleDatagram(
        header=_hdr(), ping_counter=1, serial_number=7, sound_speed_m_s=1500.0,
        num_tx_sectors=2, num_rx_beams=2, num_valid_detections=1,
        sample_frequency_hz=30000.0,
        beams=[_nbeam(-25.0, 0, -22.0), _nbeam(25.0, 1, -28.0)],
    )

    recs = process_all_ping(
        x, n, mode_byte=1, em_model=2040, latitude_deg=None, longitude_deg=None,
        reflectivity_source="xyz88", angle_bin_size=1.0, valid_only=True,
        min_depth=None, max_depth=None, spatial_projector=None,
        slope_sampler=None, bathy_sd_sampler=None, rx_fan_index=0,
        ping_filter_stats=None, **_ping_qc_off(),
    )
    assert len(recs) == 1  # second beam is an invalid X detection
    r = recs[0]
    assert r.depth_mode == 1          # EM2040 mode byte 1 -> 300kHz id 1
    assert r.raw_depth_mode == 1
    assert r.sector == 0              # from N beam 0
    assert r.angle == -25            # from N beam 0, binned
    assert r.intensity == pytest.approx(-25.0)  # xyz88 source -> X reflectivity
    assert r.z_m == pytest.approx(100.0)


def test_process_all_ping_rawrange78_source():
    """reflectivity_source='rawrange78' uses the N reflectivity instead of X."""
    x = DepthDatagram(
        header=_hdr(), counter=1, serial_number=7, heading_deg=0.0,
        sound_speed_at_transducer_m_s=1500.0, transducer_depth_m=3.0,
        num_beams=1, num_valid_detections=1, sample_frequency_hz=30000.0,
        scanning_info=0, beams=[_xbeam(100.0, 50.0, -25.0)],
    )
    n = RawRangeAngleDatagram(
        header=_hdr(), ping_counter=1, serial_number=7, sound_speed_m_s=1500.0,
        num_tx_sectors=1, num_rx_beams=1, num_valid_detections=1,
        sample_frequency_hz=30000.0, beams=[_nbeam(-25.0, 0, -22.0)],
    )
    recs = process_all_ping(
        x, n, mode_byte=1, em_model=2040, latitude_deg=None, longitude_deg=None,
        reflectivity_source="rawrange78", angle_bin_size=1.0, valid_only=True,
        min_depth=None, max_depth=None, spatial_projector=None,
        slope_sampler=None, bathy_sd_sampler=None, rx_fan_index=0,
        ping_filter_stats=None, **_ping_qc_off(),
    )
    assert recs[0].intensity == pytest.approx(-22.0)


def test_process_all_ping_skips_when_no_runtime():
    """A ping with no runtime mode yet is skipped (mode_byte=None)."""
    x = DepthDatagram(
        header=_hdr(), counter=1, serial_number=7, heading_deg=0.0,
        sound_speed_at_transducer_m_s=1500.0, transducer_depth_m=3.0,
        num_beams=1, num_valid_detections=1, sample_frequency_hz=30000.0,
        scanning_info=0, beams=[_xbeam(100.0, 50.0, -25.0)],
    )
    n = RawRangeAngleDatagram(
        header=_hdr(), ping_counter=1, serial_number=7, sound_speed_m_s=1500.0,
        num_tx_sectors=1, num_rx_beams=1, num_valid_detections=1,
        sample_frequency_hz=30000.0, beams=[_nbeam(-25.0, 0, -22.0)],
    )
    stats: dict = defaultdict(int)
    recs = process_all_ping(
        x, n, mode_byte=None, em_model=2040, latitude_deg=None, longitude_deg=None,
        reflectivity_source="xyz88", angle_bin_size=1.0, valid_only=True,
        min_depth=None, max_depth=None, spatial_projector=None,
        slope_sampler=None, bathy_sd_sampler=None, rx_fan_index=0,
        ping_filter_stats=stats, **_ping_qc_off(),
    )
    assert recs == []
    assert stats["pings_skipped_no_runtime"] == 1


def _seabed(counter, beams_samples):
    """Build a SeabedImageDatagram with the given per-beam sample lists."""
    beams = [
        YBeam(
            sorting_direction=0,
            detection_info=0,
            number_of_samples_per_beam=len(s),
            centre_sample_number=len(s) // 2,
            samples=tuple(s),
        )
        for s in beams_samples
    ]
    return SeabedImageDatagram(
        header=_hdr(), counter=counter, serial_number=7, sample_frequency_hz=30000.0,
        range_to_normal_incidence_samples=0, normal_incidence_bs_db=-25,
        oblique_bs_db=-35, tx_beam_width_deg=1.0, tvg_crossover_deg=45.0,
        num_beams=len(beams), beams=beams,
    )


def test_process_all_ping_source_b_reduces_seabed_image():
    """Source B reduces each Y beam's samples; primary stat drives intensity."""
    x = DepthDatagram(
        header=_hdr(), counter=1, serial_number=7, heading_deg=0.0,
        sound_speed_at_transducer_m_s=1500.0, transducer_depth_m=3.0,
        num_beams=2, num_valid_detections=2, sample_frequency_hz=30000.0,
        scanning_info=0, beams=[_xbeam(100.0, -50.0, -25.0), _xbeam(100.0, 50.0, -30.0)],
    )
    n = RawRangeAngleDatagram(
        header=_hdr(), ping_counter=1, serial_number=7, sound_speed_m_s=1500.0,
        num_tx_sectors=2, num_rx_beams=2, num_valid_detections=2,
        sample_frequency_hz=30000.0,
        beams=[_nbeam(-25.0, 0, -22.0), _nbeam(25.0, 1, -28.0)],
    )
    # beam0 samples mean = -20 dB; beam1 samples mean = -45 dB.
    y = _seabed(1, [[-100, -200, -300], [-400, -500]])

    recs = process_all_ping(
        x, n, mode_byte=1, em_model=2040, latitude_deg=None, longitude_deg=None,
        reflectivity_source="xyz88", angle_bin_size=1.0, valid_only=True,
        min_depth=None, max_depth=None, spatial_projector=None,
        slope_sampler=None, bathy_sd_sampler=None, rx_fan_index=0,
        ping_filter_stats=None, seabed=y, bs_source="seabed_image",
        beam_stats=["mean", "std"], si_window=None, **_ping_qc_off(),
    )
    assert len(recs) == 2
    assert recs[0].intensity == pytest.approx(-20.0)   # mean of beam0 samples
    assert recs[1].intensity == pytest.approx(-45.0)
    assert recs[0].intensity_by_stat["mean"] == pytest.approx(-20.0)
    assert "std" in recs[0].intensity_by_stat


def test_process_all_ping_source_b_requires_seabed():
    x = DepthDatagram(
        header=_hdr(), counter=1, serial_number=7, heading_deg=0.0,
        sound_speed_at_transducer_m_s=1500.0, transducer_depth_m=3.0,
        num_beams=1, num_valid_detections=1, sample_frequency_hz=30000.0,
        scanning_info=0, beams=[_xbeam(100.0, -50.0, -25.0)],
    )
    n = RawRangeAngleDatagram(
        header=_hdr(), ping_counter=1, serial_number=7, sound_speed_m_s=1500.0,
        num_tx_sectors=1, num_rx_beams=1, num_valid_detections=1,
        sample_frequency_hz=30000.0, beams=[_nbeam(-25.0, 0, -22.0)],
    )
    stats: dict = defaultdict(int)
    recs = process_all_ping(
        x, n, mode_byte=1, em_model=2040, latitude_deg=None, longitude_deg=None,
        reflectivity_source="xyz88", angle_bin_size=1.0, valid_only=True,
        min_depth=None, max_depth=None, spatial_projector=None,
        slope_sampler=None, bathy_sd_sampler=None, rx_fan_index=0,
        ping_filter_stats=stats, seabed=None, bs_source="seabed_image",
        beam_stats=["mean"], si_window=None, **_ping_qc_off(),
    )
    assert recs == []
    assert stats["pings_skipped_no_seabed_image"] == 1


def test_detect_format_and_files(tmp_path):
    a = tmp_path / "x.all"
    a.write_bytes(b"")
    k = tmp_path / "x.kmall"
    k.write_bytes(b"")
    assert detect_format_and_files(a, "auto")[0] == "all"
    assert detect_format_and_files(k, "auto")[0] == "kmall"
    # explicit override
    assert detect_format_and_files(a, "kmall")[0] == "kmall"


# --- Integration against real .all fixtures --------------------------------


def _run_table(fixture, reflectivity_source="xyz88", min_soundings=1,
               bs_source="reflectivity", beam_stats=None, si_window=None):
    agg = defaultdict(Agg)
    raw = defaultdict(set)
    pg = defaultdict(int)
    pf = defaultdict(int)
    stats: dict = defaultdict(int)
    extra_stats = list(beam_stats) if (bs_source == "seabed_image" and beam_stats) else []
    extra_agg = {s: defaultdict(Agg) for s in extra_stats}
    pings, before, after, used = accumulate_all_file(
        Path(fixture), agg, raw, pg, pf,
        reflectivity_source=reflectivity_source, angle_bin_size=1.0, valid_only=True,
        min_depth=None, max_depth=None, geometry_filter=None, spatial_projector=None,
        slope_sampler=None, bathy_sd_sampler=None, slope_max_deg=None, bathy_sd_max_m=None,
        ping_filter_stats=stats, flat_filter=False, flat_radius_m=50.0,
        flat_min_neighbors=50, flat_max_slope_deg=3.0, flat_max_roughness_m=1.0,
        flat_max_bs_std_db=None, bs_source=bs_source, beam_stats=beam_stats,
        si_window=si_window, extra_agg=extra_agg, extra_stats=extra_stats, **_ping_qc_off(),
    )
    rows = build_rows(
        agg, raw, pg, pf, min_soundings=min_soundings, reference_group="mode_fan",
        reference_stat="median", max_abs_correction=None, flat_filter_used=False,
        extra_agg=extra_agg, extra_stats=extra_stats,
    )
    return (pings, used), rows


@pytest.mark.skipif(not EM2040_ALL.exists(), reason="EM2040 .all fixture not present")
def test_em2040_all_table_is_sane():
    (pings, used), rows = _run_table(EM2040_ALL)
    assert pings > 0 and used > 0 and rows
    angles = [r["beam_angle_deg"] for r in rows]
    bs = [r["avgIntensity_dB"] for r in rows]
    assert min(angles) < -20 and max(angles) > 20          # spans nadir
    assert all(-80 < v < 10 for v in bs)                   # plausible BS dB
    # EM2040 mode byte 1 -> frequency mode id 1 (300kHz).
    assert {r["depthMode"] for r in rows} == {1}


@pytest.mark.skipif(not EM302_ALL.exists(), reason="EM302 .all fixture not present")
def test_em302_all_table_is_sane():
    (pings, used), rows = _run_table(EM302_ALL)
    assert pings > 0 and used > 0 and rows
    # EM302 (general model) reports multiple transmit sectors.
    assert len({r["sector"] for r in rows}) > 1


@pytest.mark.skipif(not EM2040_ALL.exists(), reason="EM2040 .all fixture not present")
def test_em2040_all_source_b_multistat_table():
    """Source B (seabed-image) on real EM2040 .all yields sane multi-stat columns."""
    (pings, used), rows = _run_table(
        EM2040_ALL, bs_source="seabed_image", beam_stats=["mean", "std", "p90"]
    )
    assert pings > 0 and used > 0 and rows
    r = rows[0]
    # Primary stat (mean) drives avgIntensity_dB; comparison columns are present.
    assert r["avgIntensity_dB"] == pytest.approx(r["avgIntensity_mean_dB"])
    assert "avgIntensity_std_dB" in r and "avgIntensity_p90_dB" in r
    bs_vals = [row["avgIntensity_mean_dB"] for row in rows]
    assert all(-80 < v < 10 for v in bs_vals)
    # p90 >= mean for the same bin (upper-tail percentile of backscatter).
    assert all(row["avgIntensity_p90_dB"] >= row["avgIntensity_mean_dB"] - 1e-6 for row in rows)


@pytest.mark.skipif(not EM2040_ALL.exists(), reason="EM2040 .all fixture not present")
def test_em2040_all_table_feeds_normalize():
    """The .all table values flow through the Lambertian sector solver unchanged."""
    _, rows = _run_table(EM2040_ALL)
    angles = np.array([r["beam_angle_deg"] for r in rows], dtype=float)
    values = np.array([r["avgIntensity_dB"] for r in rows], dtype=float)
    sectors = np.array([r["sector"] for r in rows], dtype=int)
    shifts = normalize.solve_sector_shifts(angles, values, sectors)
    assert shifts  # at least one sector solved
    assert all(abs(v) <= 15.0 for v in shifts.values())  # clipped to max_shift_db


@pytest.mark.skipif(not EM2040_ALL.exists(), reason="EM2040 .all fixture not present")
def test_em2040_all_apply_patches_y_samples(tmp_path):
    """A sector correction shifts that sector's Y samples and leaves others alone."""
    from mbes_tools.all import (
        iter_raw_range_angle_datagrams,
        iter_seabed_image_datagrams,
    )
    from mbes_tools.backscatter.apply import all_mode_to_calib, process_one_all

    out = tmp_path / "out.all"
    assert all_mode_to_calib(2040, 1) == 2  # 300kHz -> calib mode 2
    lookup = {(2, 0, 2): 3.0}  # +3 dB on calib mode 2, fan 0, sector 2 (raw sector 1)
    qa: list = []
    process_one_all(EM2040_ALL, out, lookup, qa, patch_dtype="auto",
                    correction_units="dB_to_desidB", dry_run=False)
    assert qa and all(r["patch_status"] == "patched_int16" for r in qa)
    assert out.stat().st_size == EM2040_ALL.stat().st_size  # length preserved

    y_o = next(iter_seabed_image_datagrams(EM2040_ALL))
    y_p = next(iter_seabed_image_datagrams(out))
    n = {d.ping_counter: d for d in iter_raw_range_angle_datagrams(EM2040_ALL)}[y_o.counter]
    shifted = unshifted = 0
    for i, (bo, bp) in enumerate(zip(y_o.beams, y_p.beams)):
        if not bo.samples:
            continue
        delta = np.array(bp.samples) - np.array(bo.samples)
        if n.beams[i].tx_sector_number == 1:
            assert np.all(delta == 30)  # +3.0 dB -> +30 deci-dB
            shifted += 1
        else:
            assert np.all(delta == 0)
            unshifted += 1
    assert shifted > 0 and unshifted > 0


@pytest.mark.skipif(not EM2040_ALL.exists(), reason="EM2040 .all fixture not present")
def test_table_cli_on_error_skip_survives_corruption(tmp_path, monkeypatch):
    """The full mbes-bs-table run completes on a corrupt .all with --on-error skip."""
    import struct as _struct
    from mbes_tools.backscatter import table

    raw = bytearray(EM2040_ALL.read_bytes())
    # Clobber a datagram length deep in the file to overshoot EOF.
    recs = list(__import__("mbes_tools.all", fromlist=["iter_datagrams"]).iter_datagrams(EM2040_ALL))
    off = recs[len(recs) // 2].offset
    raw[off : off + 4] = _struct.pack("<L", 999_999_999)
    corrupt = tmp_path / "corrupt.all"
    corrupt.write_bytes(bytes(raw))
    out = tmp_path / "t.csv"

    monkeypatch.setattr(
        sys, "argv",
        ["mbes-bs-table", str(corrupt), "-o", str(out), "--min-soundings", "1", "--on-error", "skip"],
    )
    table.main()  # must not raise
    assert out.exists()
    assert out.read_text().count("\n") > 1  # header + at least one row


MBES_TEST_DATA_ROOT = os.environ.get("MBES_TEST_DATA_ROOT")


@pytest.mark.skipif(not MBES_TEST_DATA_ROOT, reason="MBES_TEST_DATA_ROOT not set")
def test_cross_model_full_files_do_not_crash():
    """Cross-model/cross-geography: full files under MBES_TEST_DATA_ROOT parse clean.

    Reads the first ping/MRZ of up to a few .all and .kmall files found under the
    data root and asserts sane structure — verifying reader robustness across
    whatever EM models the local corpus contains. Skips cleanly when unset.
    """
    from mbes_tools import all as mall
    from mbes_tools import kmall

    root = Path(MBES_TEST_DATA_ROOT)
    all_files = sorted(root.rglob("*.all"))[:5]
    kmall_files = sorted(root.rglob("*.kmall"))[:5]
    assert all_files or kmall_files, f"no .all/.kmall under {root}"

    for f in all_files:
        # Tolerant read: corrupt/compressed files skip gracefully (yield nothing).
        log: list = []
        first = next(mall.iter_datagrams(f, types={"X"}, on_error="skip", error_log=log), None)
        if first is not None:
            assert first.header.em_model >= 0
    for f in kmall_files:
        log = []
        d = next(kmall.iter_mrz_datagrams(f, on_error="skip", error_log=log), None)
        if d is not None:
            assert len(d.soundings) >= 0


@pytest.mark.skipif(not EM2040_ALL.exists(), reason="EM2040 .all fixture not present")
def test_em2040_all_apply_dry_run_writes_nothing(tmp_path):
    from mbes_tools.backscatter.apply import process_one_all

    out = tmp_path / "out.all"
    qa: list = []
    process_one_all(EM2040_ALL, out, {(2, 0, 2): 3.0}, qa, patch_dtype="auto",
                    correction_units="dB_to_desidB", dry_run=True)
    assert not out.exists()
    assert qa and all(r["patch_status"].startswith("dry_run_") for r in qa)
