"""Tests for mbes_tools.catalog (synthetic; no data dependency)."""
import gzip
import struct
from pathlib import Path

from mbes_tools import catalog


# ---------------------------------------------------------------------------
# Pure helpers.
# ---------------------------------------------------------------------------


def test_utm_zone_basic():
    # Monterey Bay -> UTM 10N.
    assert catalog.utm_zone_from_lonlat(-122.6, 36.4) == "10N"
    # Southern hemisphere keeps the same zone, S band.
    assert catalog.utm_zone_from_lonlat(-122.6, -36.4) == "10S"


def test_utm_zone_antimeridian_and_bounds():
    # Longitude past +180 wraps cleanly (no zone 0 or 61).
    assert catalog.utm_zone_from_lonlat(181.0, 0.0) == "1N"
    assert catalog.utm_zone_from_lonlat(-180.0, 0.0) == "1N"
    z = catalog.utm_zone_from_lonlat(179.9, -14.0)  # near Samoa / antimeridian
    assert z.endswith("S") and 1 <= int(z[:-1]) <= 60


def test_utm_zone_nonfinite():
    assert catalog.utm_zone_from_lonlat(float("nan"), 10.0) == ""


def test_classify_depth_regime():
    assert catalog.classify_depth_regime(50) == "shallow"
    assert catalog.classify_depth_regime(500) == "shelf/slope"
    assert catalog.classify_depth_regime(2000) == "deep"
    assert catalog.classify_depth_regime(4000) == "abyssal"
    assert catalog.classify_depth_regime(None) == ""


def test_vessel_from_filename():
    assert (
        catalog.vessel_from_filename(Path("0118_20210605_010253_Langseth.all"))
        == "R/V Marcus G. Langseth"
    )
    assert (
        catalog.vessel_from_filename(Path("0000_20141129_000828_revelle.all"))
        == "R/V Roger Revelle"
    )
    # A trailing model code must not be mistaken for a vessel.
    assert catalog.vessel_from_filename(Path("0005_20230305_165930_FKt230303_EM124.kmall")) == ""
    # A trailing numeric field is not a vessel.
    assert catalog.vessel_from_filename(Path("PS01_NG_02_01_0001.all")) == ""


# ---------------------------------------------------------------------------
# Synthetic .all scanning (envelope builder mirrors tests/test_all.py).
# ---------------------------------------------------------------------------


def _all_envelope(type_char: str, body: bytes, em_model: int = 2040,
                  date: int = 20190407, time_ms: int = 0) -> bytes:
    footer = struct.pack("<BH", 0x03, 0)
    nbytes = 16 - 4 + len(body) + len(footer)
    header = struct.pack("<LBBHLL", nbytes, 0x02, ord(type_char), em_model, date, time_ms)
    return header + body + footer


def test_scan_all_file_reports_model_types_and_geography(tmp_path):
    lat, lon = 36.6066, -121.8957  # Monterey Bay
    p_body = struct.pack(
        "<HHll4HBB", 1, 7, int(lat * 20_000_000), int(lon * 10_000_000),
        25, 500, 12_000, 9_000, 1, 0,
    )
    # X depth datagram: 4*u16, f32 (transducerDepth), 2*u16, f32, 4*u8, then 1 beam.
    x_body = struct.pack("<4Hf2Hf4B", 1, 7, 9_000, 15_000, 3.0, 1, 1, 30_000.0, 0, 0, 0, 0)
    x_body += struct.pack("<fffHBBBbh", 95.0, 1.0, 0.0, 10, 5, 0, 0, 0, -200)
    y_body = struct.pack("<HHfHhhHHH", 1, 7, 30_000.0, 120, -25, -35, 10, 450, 1)
    y_body += struct.pack("<bBHH", 0, 1, 1, 0) + struct.pack("<h", -250)

    blob = _all_envelope("P", p_body) + _all_envelope("X", x_body) + _all_envelope("Y", y_body)
    f = tmp_path / "0002_20190407_102940_Equinox_2040_300kHz.all"
    f.write_bytes(blob)

    row = catalog.scan_all_file(f, max_scan_bytes=0)
    assert row.error == ""
    assert row.em_model == "EM2040"
    assert row.has_seabed_image is True
    assert "P:1" in row.datagram_types and "X:1" in row.datagram_types and "Y:1" in row.datagram_types
    assert row.latitude == round(lat, 6)
    assert row.utm_zone == "10N"
    assert row.median_depth_m == 95.0
    assert row.depth_regime == "shallow"
    assert row.vessel == "Fugro Equinox"


def test_scan_detects_gzip(tmp_path):
    f = tmp_path / "0000_20160415_011924_Nautilus.all"
    f.write_bytes(gzip.compress(b"not a real datagram stream"))
    row = catalog.scan_all_file(f, max_scan_bytes=0)
    assert "gzip" in row.error
    assert row.n_datagrams_scanned == 0


def test_scan_truncation_flag(tmp_path):
    blob = _all_envelope("P", struct.pack("<HHll4HBB", 1, 7, 0, 0, 0, 0, 0, 0, 0, 0))
    f = tmp_path / "tiny.all"
    f.write_bytes(blob * 3)
    row = catalog.scan_all_file(f, max_scan_bytes=1)  # cap below first datagram
    assert row.scan_truncated is True


# ---------------------------------------------------------------------------
# Discovery and per-directory sampling.
# ---------------------------------------------------------------------------


def test_discover_and_sample_per_directory(tmp_path):
    d = tmp_path / "survey"
    d.mkdir()
    for i in range(5):
        (d / f"line{i}.all").write_bytes(b"")
    (d / "wc.kmwcd").write_bytes(b"")
    (d / "notes.txt").write_bytes(b"")

    files = catalog.discover_files([tmp_path])
    assert all(catalog._format_of(f) in (catalog.ALL_FAMILY | catalog.KMALL_FAMILY) for f in files)
    assert len(files) == 6  # 5 .all + 1 .kmwcd, .txt excluded

    sampled = catalog.sample_per_directory(files, per_dir_limit=2)
    # 2 .all kept + 1 .kmwcd kept (sampling is per directory AND extension).
    all_kept = [f for f in sampled if catalog._format_of(f) == "all"]
    kmwcd_kept = [f for f in sampled if catalog._format_of(f) == "kmwcd"]
    assert len(all_kept) == 2
    assert len(kmwcd_kept) == 1

    assert catalog.sample_per_directory(files, per_dir_limit=0) == files
