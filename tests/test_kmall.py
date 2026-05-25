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


def test_iter_kmall_files_on_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        kmall.iter_kmall_files(Path("/nonexistent/path/that/does/not/exist"))


# TODO: add end-to-end MRZ parser test once a small .kmall fixture is
# committed under tests/fixtures/. Plan:
# - clip a few pings (~3-5) from a Monterey Canyon .kmall to keep size small
# - parse it, assert ping count, depth_modes, num_tx_sectors, sounding counts
# - assert lat/lon are present and within Monterey Bay bounding box
