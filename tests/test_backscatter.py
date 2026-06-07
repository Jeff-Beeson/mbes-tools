"""Tests for mbes_tools.backscatter and the seabed-image kmall extension.

These cover the survey-agnostic, numpy-only pieces of the backscatter pipeline
plus the additive #MRZ seabed-image parsing. Heavy optional dependencies
(scipy / pyproj / pandas) are not required here.
"""

import struct
from pathlib import Path

import numpy as np
import pytest

from mbes_tools import kmall
from mbes_tools.backscatter import apply, normalize, qc, table

FIXTURE = Path(__file__).parent / "fixtures" / "sample_dpdk027.kmall"


# ---------------------------------------------------------------------------
# Imports / API surface.
# ---------------------------------------------------------------------------


def test_backscatter_submodules_import():
    from mbes_tools.backscatter import cli  # noqa: F401

    assert hasattr(table, "main")
    assert hasattr(apply, "main")
    assert hasattr(cli, "table_main")


# ---------------------------------------------------------------------------
# table.bin_angle
# ---------------------------------------------------------------------------


def test_bin_angle_integer_bins_return_ints():
    assert table.bin_angle(12.4, 1.0) == 12
    assert isinstance(table.bin_angle(12.4, 1.0), int)
    assert table.bin_angle(-7.6, 1.0) == -8


def test_bin_angle_fractional_bins():
    assert table.bin_angle(12.4, 0.5) == pytest.approx(12.5)


def test_bin_angle_rejects_nonpositive():
    with pytest.raises(ValueError):
        table.bin_angle(1.0, 0.0)


# ---------------------------------------------------------------------------
# qc.parse_keep_values and AsciiGridMask
# ---------------------------------------------------------------------------


def test_parse_keep_values():
    assert qc.parse_keep_values("1") == [1.0]
    assert qc.parse_keep_values("1, 2 ,3") == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        qc.parse_keep_values("")


def test_ascii_grid_mask_sampling(tmp_path):
    # 2x2 grid, cellsize 1, lower-left at (0,0). Rows are written top-down.
    grid = tmp_path / "mask.asc"
    grid.write_text(
        "ncols 2\n"
        "nrows 2\n"
        "xllcorner 0\n"
        "yllcorner 0\n"
        "cellsize 1\n"
        "NODATA_value -9999\n"
        "10 20\n"   # top row    -> y in [1,2)
        "30 40\n"   # bottom row -> y in [0,1)
    )
    g = qc.AsciiGridMask.from_file(grid)
    assert g.sample(0.5, 0.5) == 30.0   # bottom-left
    assert g.sample(1.5, 0.5) == 40.0   # bottom-right
    assert g.sample(0.5, 1.5) == 10.0   # top-left
    assert g.sample(1.5, 1.5) == 20.0   # top-right
    assert g.sample(5.0, 5.0) is None   # out of bounds


# ---------------------------------------------------------------------------
# apply: correction math and binary patching
# ---------------------------------------------------------------------------


def test_apply_corrections_preserves_nodata():
    si = np.array([100, -32767, 200, 300, -32767], dtype=float)
    counts = np.array([2, 3], dtype=int)
    corr = np.array([5.0, -10.0], dtype=np.float32)
    out = apply.apply_corrections_to_si_array(si, counts, corr)
    # Beam 0 (first 2): +5, but no-data preserved.
    assert out[0] == pytest.approx(105.0)
    assert out[1] == -32767.0
    # Beam 1 (next 3): -10, but no-data preserved.
    assert out[2] == pytest.approx(190.0)
    assert out[3] == pytest.approx(290.0)
    assert out[4] == -32767.0


def test_depth_mode_raw_to_calib():
    assert apply.depth_mode_raw_to_calib(101) == 2   # manual Shallow
    assert apply.depth_mode_raw_to_calib(106) == 7
    assert apply.depth_mode_raw_to_calib(1) == 2      # auto Shallow
    assert apply.depth_mode_raw_to_calib(5) == 6


def test_find_unique_payload_and_patch_roundtrip():
    original = np.array([10, 20, 30], dtype=np.int16)
    corrected = np.array([11, 21, 31], dtype=np.int16)
    payload = apply.pack_int16_le(original)
    # Embed the payload inside a larger datagram with a unique surrounding.
    raw = b"\x00\x01header" + payload + b"trailer\xff"
    assert apply.find_unique_payload(raw, payload) == len(b"\x00\x01header")

    patched, status, pos = apply.patch_si_payload_in_datagram(
        raw, original.astype(float), corrected.astype(float), patch_dtype="int16"
    )
    assert status == "patched_int16"
    assert len(patched) == len(raw)
    # The corrected payload sits at the same position; surroundings untouched.
    expect = b"\x00\x01header" + apply.pack_int16_le(corrected) + b"trailer\xff"
    assert patched == expect


def test_patch_reports_when_payload_not_unique():
    original = np.array([7, 7], dtype=np.int16)
    corrected = np.array([8, 8], dtype=np.int16)
    payload = apply.pack_int16_le(original)
    raw = payload + payload  # appears twice -> not unique
    _, status, pos = apply.patch_si_payload_in_datagram(
        raw, original.astype(float), corrected.astype(float), patch_dtype="int16"
    )
    assert status == "payload_not_found_unique"
    assert pos is None


# ---------------------------------------------------------------------------
# normalize: Lambertian fitting
# ---------------------------------------------------------------------------


def test_fit_lambert_intercept_recovers_intercept():
    angles = np.arange(-60, 61, 5.0)
    intercept_true = -25.0
    values = intercept_true + 10.0 * np.log10(np.cos(np.deg2rad(np.abs(angles))))
    est = normalize.fit_lambert_intercept(angles, values, 0.0, 75.0)
    assert est == pytest.approx(intercept_true, abs=1e-6)


def test_solve_sector_shifts_aligns_offset_sector():
    angles = np.concatenate([np.arange(-50, 51, 5.0), np.arange(-50, 51, 5.0)])
    base = -20.0 + 10.0 * np.log10(np.cos(np.deg2rad(np.abs(angles))))
    sectors = np.array([1] * 21 + [2] * 21)
    values = base.copy()
    values[sectors == 2] += 4.0  # sector 2 is 4 dB hot
    shifts = normalize.solve_sector_shifts(angles, values, sectors, max_shift_db=15.0)
    # The fit balances both sectors toward the joint Lambertian reference, so the
    # 4 dB imbalance is removed relatively: sector 2 down, sector 1 up by the same.
    assert shifts[2] < 0 < shifts[1]
    assert (shifts[1] - shifts[2]) == pytest.approx(4.0, abs=0.5)


# ---------------------------------------------------------------------------
# kmall seabed-image extension
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not FIXTURE.exists(), reason="sample kmall fixture not present")
def test_iter_mrz_parses_seabed_image_and_offsets():
    saw_datagram = False
    for dgm in kmall.iter_mrz_datagrams(FIXTURE, parse_seabed_image=True):
        saw_datagram = True
        assert dgm.byte_offset is not None
        assert dgm.num_bytes_dgm is not None
        total = sum(s.si_num_samples for s in dgm.soundings)
        assert dgm.si_sample_desidb is not None
        assert len(dgm.si_sample_desidb) == total
        # Samples are signed int16.
        if dgm.si_sample_desidb:
            assert min(dgm.si_sample_desidb) >= -32768
            assert max(dgm.si_sample_desidb) <= 32767
        break
    assert saw_datagram


@pytest.mark.skipif(not FIXTURE.exists(), reason="sample kmall fixture not present")
def test_seabed_image_skipped_by_default():
    for dgm in kmall.iter_mrz_datagrams(FIXTURE):
        assert dgm.si_sample_desidb is None
        # si_num_samples is still populated on soundings regardless.
        assert all(s.si_num_samples >= 0 for s in dgm.soundings)
        break


@pytest.mark.skipif(not FIXTURE.exists(), reason="sample kmall fixture not present")
def test_seabed_image_payload_is_byte_findable():
    """The parsed SI array must repack to bytes that exist in the raw datagram.

    This is the invariant the apply stage relies on for binary patching.
    """
    for dgm in kmall.iter_mrz_datagrams(FIXTURE, parse_seabed_image=True):
        if not dgm.si_sample_desidb:
            continue
        raw = apply.read_raw_datagram_bytes(FIXTURE, dgm.byte_offset, dgm.num_bytes_dgm)
        payload = struct.pack(f"<{len(dgm.si_sample_desidb)}h", *dgm.si_sample_desidb)
        assert payload in raw
        break
