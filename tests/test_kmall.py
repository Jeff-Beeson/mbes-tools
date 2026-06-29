"""Tests for mbes_tools.kmall."""
import struct
from pathlib import Path

import pytest

import mbes_tools
from mbes_tools import kmall
from mbes_tools.kmall import MRZSounding


def test_package_imports():
    """Smoke test: package imports and reports a version."""
    assert mbes_tools.__version__


def test_kmall_module_exposes_api():
    """The lifted parser API is importable from mbes_tools.kmall."""
    assert hasattr(kmall, "iter_mrz_datagrams")
    assert hasattr(kmall, "parse_mrz_datagram")
    assert hasattr(kmall, "MRZDatagram")
    assert hasattr(kmall, "MRZSounding")
    assert hasattr(kmall, "iter_kmall_files")
    assert hasattr(kmall, "normalize_depth_mode")


def test_struct_sizes_match_kongsberg_layout():
    """Binary format sizes should match the documented Kongsberg layout.

    These constants are computed at import-time from the format strings.
    Pinning them here means an accidental change to the format string
    (which would silently corrupt parsing) will fail this test.
    """
    assert kmall.DGM_HEADER_SIZE == struct.calcsize("<I4sBBHII")
    assert kmall.MRZ_PARTITION_SIZE == struct.calcsize("<HH")
    assert kmall.MRZ_CMN_PART_SIZE == struct.calcsize("<HH8B")
    assert kmall.MRZ_PING_GEO_SIZE == struct.calcsize("<ddf")


def test_normalize_depth_mode_subtracts_100_for_manual_modes():
    """Manual KMALL depth modes (>= 100) get +100 offset removed."""
    assert kmall.normalize_depth_mode(6) == 6
    assert kmall.normalize_depth_mode(106) == 6
    assert kmall.normalize_depth_mode(99) == 99
    assert kmall.normalize_depth_mode(100) == 0
    # Pass-through when caller explicitly opts out.
    assert kmall.normalize_depth_mode(106, normalize_manual=False) == 106


def test_mrz_sounding_is_valid_flag():
    """detection_type=0 + detection_method!=0 marks a valid detection."""
    valid = MRZSounding(
        sounding_index=0, tx_sector_numb=0,
        detection_type=0, detection_method=2,
        reflectivity1_db=-30.0, reflectivity2_db=-25.0,
        beam_angle_re_rx_deg=45.0,
        x_m=1.0, y_m=2.0, z_m=100.0,
    )
    assert valid.is_valid

    no_detection = MRZSounding(
        sounding_index=0, tx_sector_numb=0,
        detection_type=0, detection_method=0,
        reflectivity1_db=-30.0, reflectivity2_db=-25.0,
        beam_angle_re_rx_deg=45.0,
        x_m=1.0, y_m=2.0, z_m=100.0,
    )
    assert not no_detection.is_valid

    extra_detection = MRZSounding(
        sounding_index=0, tx_sector_numb=0,
        detection_type=1, detection_method=2,
        reflectivity1_db=-30.0, reflectivity2_db=-25.0,
        beam_angle_re_rx_deg=45.0,
        x_m=1.0, y_m=2.0, z_m=100.0,
    )
    assert not extra_detection.is_valid


def _kdg(type4: bytes, payload: bytes = b"") -> bytes:
    """A minimal kmall datagram envelope: uint32 size + 4-byte type + payload."""
    size = 8 + len(payload)
    return struct.pack("<I4s", size, type4) + payload


def test_resync_kmall_finds_next_datagram(tmp_path):
    """_resync_kmall locates the next #XXX datagram after a bad offset."""
    d0 = _kdg(b"#SPO", b"\x00" * 12)
    d1 = _kdg(b"#SVP", b"\x11" * 16)
    blob = d0 + b"\x99" * 5 + d1
    f = tmp_path / "j.kmall"
    f.write_bytes(blob)
    with f.open("rb") as fid:
        nxt = kmall._resync_kmall(fid, len(blob), len(d0))
    assert nxt == len(d0) + 5  # start of the #SVP datagram


def test_iter_mrz_on_error_skip_vs_raise(tmp_path):
    """A corrupt leading length: skip resyncs and never raises; raise raises."""
    good_tail = _kdg(b"#SPO", b"\x00" * 8) + _kdg(b"#SVP", b"\x00" * 8)
    blob = struct.pack("<I4s", 3, b"#SPO") + b"\x00\x00\x00" + good_tail  # bad size (<8)
    f = tmp_path / "c.kmall"
    f.write_bytes(blob)

    with pytest.raises(RuntimeError):
        list(kmall.iter_mrz_datagrams(f))

    log: list = []
    # No #MRZ present, but skip must not raise and must record the problem.
    out = list(kmall.iter_mrz_datagrams(f, on_error="skip", error_log=log))
    assert out == []
    assert len(log) >= 1


def test_iter_kmall_files_on_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        kmall.iter_kmall_files(Path("/nonexistent/path/that/does/not/exist"))


# ---------------------------------------------------------------------------
# Fixture-based integration tests.
# Source: tests/fixtures/sample_dpdk027.kmall, clipped from MBARI DPDK027
# (David Packard, EM 2040 dual head, Monterey Canyon, 2025-11-12) via
# tests/fixtures/clip_datagrams.py --mrz 2.
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_kmall():
    """Path to the committed sample_dpdk027.kmall, or skip if absent."""
    p = FIXTURES / "sample_dpdk027.kmall"
    if not p.exists():
        pytest.skip(f"Fixture not present: {p}")
    return p


def test_fixture_mrz_count(sample_kmall):
    """Clipper stopped after 2 #MRZ datagrams; parser should find them both."""
    datagrams = list(kmall.iter_mrz_datagrams(sample_kmall))
    assert len(datagrams) == 2


def test_fixture_mrz_structure(sample_kmall):
    """Each parsed MRZ should have sensible structural metadata."""
    for dgm in kmall.iter_mrz_datagrams(sample_kmall):
        assert dgm.num_tx_sectors > 0
        assert dgm.rx_fans_per_ping > 0
        assert dgm.depth_mode >= 0
        # EM2040 systems typically use depth modes 0-7 (raw) or 100-107 (manual).
        # After normalization the value should land in the 0-30 range.
        assert dgm.depth_mode < 30
        assert len(dgm.soundings) > 0


def test_fixture_mrz_geolocation(sample_kmall):
    """Lat/lon should be present and within the Monterey Bay bounding box."""
    datagrams = list(kmall.iter_mrz_datagrams(sample_kmall))
    geolocated = [d for d in datagrams if d.latitude_deg is not None]
    assert len(geolocated) > 0, "No MRZ datagrams carried lat/lon"

    # Monterey Bay & Monterey Canyon (canyon axis extends well offshore):
    # ~36-37 N, -123 W to -121.5 W.
    for dgm in geolocated:
        assert 36.0 < dgm.latitude_deg < 37.5, f"lat out of range: {dgm.latitude_deg}"
        assert -123.0 < dgm.longitude_deg < -121.5, f"lon out of range: {dgm.longitude_deg}"


def test_fixture_mrz_soundings_have_valid_detections(sample_kmall):
    """A real DPDK027 ping should have plenty of valid detections."""
    for dgm in kmall.iter_mrz_datagrams(sample_kmall):
        valid_count = sum(1 for s in dgm.soundings if s.is_valid)
        # A deep EM304 ping with ~500 beams should have hundreds of valid detections.
        assert valid_count > 100, (
            f"Ping {dgm.ping_cnt} has only {valid_count} valid soundings"
        )


# ---------------------------------------------------------------------------
# EM124 fixture (Samoa-matched ship sonar). Source: tests/fixtures/
# sample_tn447_em124.kmall, clipped from cruise TN447 (R/V Thomas G. Thompson,
# EM124, abyssal W Pacific) via clip_datagrams.py --target-type '#MRZ' -n 2.
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_em124_kmall():
    p = FIXTURES / "sample_tn447_em124.kmall"
    if not p.exists():
        pytest.skip(f"Fixture not present: {p}")
    return p


def test_em124_fixture_parses_with_seabed_image(sample_em124_kmall):
    """The EM124 (Samoa ship-matched) .kmall parses, with abyssal depths and SI samples."""
    dgms = list(kmall.iter_mrz_datagrams(sample_em124_kmall, parse_seabed_image=True))
    assert len(dgms) == 2
    for dgm in dgms:
        valid = [s for s in dgm.soundings if s.is_valid]
        assert len(valid) > 100
        # Abyssal western Pacific: soundings are deep.
        assert max(s.z_m for s in valid) > 3000
        # Seabed-image samples are present for backscatter work.
        assert dgm.si_sample_desidb
