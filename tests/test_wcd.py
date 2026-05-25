"""Tests for mbes_tools.wcd (Kongsberg .wcd water column reader)."""
import struct
from pathlib import Path

import pytest

from mbes_tools import wcd
from mbes_tools.wcd import (
    K_BEAM_FMT,
    K_BODY_FMT,
    K_TX_SECTOR_FMT,
    WaterColumnDatagram,
    WCBeam,
    WCTxSector,
)
# Reuse the .all envelope format for synthetic file construction.
from mbes_tools.all import DGM_FOOTER_FMT, DGM_HEADER_FMT


def test_wcd_module_exposes_api():
    """The .wcd parser API is importable from mbes_tools.wcd."""
    for name in [
        "iter_water_column_datagrams",
        "iter_wcd_files",
        "parse_water_column",
        "WaterColumnDatagram",
        "WCBeam",
        "WCTxSector",
    ]:
        assert hasattr(wcd, name), f"missing {name}"


def test_struct_sizes():
    """Binary format sizes should match the Kongsberg K-datagram layout."""
    # K body: 8H + L + h + B + b + L = 16 + 4 + 2 + 1 + 1 + 4 = 28.
    assert wcd.K_BODY_SIZE == 28
    # TX sector: h + H + B + B = 2 + 2 + 1 + 1 = 6.
    assert wcd.K_TX_SECTOR_SIZE == 6
    # RX beam header: h + 3H + B + B = 2 + 6 + 1 + 1 = 10.
    assert wcd.K_BEAM_SIZE == 10


def _build_wcd_envelope(body: bytes, time_ms: int = 0) -> bytes:
    """Build a synthetic .wcd datagram envelope around the K body bytes.

    Layout: 16-byte .all-style envelope header + body + 3-byte footer.
    """
    footer = struct.pack(DGM_FOOTER_FMT, 0x03, 0)
    nbytes = 16 - 4 + len(body) + len(footer)
    header = struct.pack(
        DGM_HEADER_FMT, nbytes, 0x02, ord("k"), 302, 20250507, time_ms,
    )
    return header + body + footer


def test_parse_water_column_roundtrip(tmp_path):
    """Build a synthetic K Water Column datagram, parse it, check fields."""
    # K-body fields.
    k_body = struct.pack(
        K_BODY_FMT,
        42,        # counter
        1234,      # serial number
        1,         # numDatagram (this ping spans 1 datagram)
        1,         # datagramNum (1-based)
        1,         # numTxSectors
        2,         # numBeams_ping (total beams)
        2,         # numBeamsThisDatagram
        15000,     # sound speed (raw / 10 = 1500.0 m/s)
        80000,     # sample freq (raw / 100 = 800 Hz)
        50,        # tx heave (raw / 100 = 0.5 m)
        1,         # tvg function
        -3,        # tvg offset
        0,         # spare uint32 (unused)
    )

    # TX sector: tilt=1.0 deg, centre freq=40 kHz (raw 4000 * 10 Hz), sector 0, scan 0.
    tx_sector = struct.pack(K_TX_SECTOR_FMT, 100, 4000, 0, 0)

    # Beam 0: angle=45 deg, start=0, n_samples=3, detected_range=10, sector=0, beam=0.
    beam0_hdr = struct.pack(K_BEAM_FMT, 4500, 0, 3, 10, 0, 0)
    beam0_samples = struct.pack("<3b", -50, -55, -60)

    # Beam 1: angle=-45 deg, start=0, n_samples=3, detected_range=12, sector=0, beam=1.
    beam1_hdr = struct.pack(K_BEAM_FMT, -4500, 0, 3, 12, 0, 1)
    beam1_samples = struct.pack("<3b", -70, -75, -80)

    body = k_body + tx_sector + beam0_hdr + beam0_samples + beam1_hdr + beam1_samples
    blob = _build_wcd_envelope(body)

    f = tmp_path / "synthetic.wcd"
    f.write_bytes(blob)

    records = list(wcd.iter_water_column_datagrams(f))
    assert len(records) == 1
    dgm = records[0]

    assert isinstance(dgm, WaterColumnDatagram)
    assert dgm.counter == 42
    assert dgm.serial_number == 1234
    assert dgm.num_datagram == 1
    assert dgm.datagram_num == 1
    assert dgm.num_tx_sectors == 1
    assert dgm.num_beams_ping == 2
    assert dgm.num_beams_this_datagram == 2
    assert dgm.sound_speed_m_s == pytest.approx(1500.0)
    assert dgm.sample_frequency_hz == pytest.approx(800.0)
    assert dgm.tx_heave_m == pytest.approx(0.5)
    assert dgm.tvg_function == 1
    assert dgm.tvg_offset_db == -3

    assert len(dgm.tx_sectors) == 1
    assert dgm.tx_sectors[0].tilt_angle_deg == pytest.approx(1.0)
    assert dgm.tx_sectors[0].centre_frequency_hz == pytest.approx(40_000.0)
    assert dgm.tx_sectors[0].tx_sector_number == 0

    assert len(dgm.beams) == 2
    assert dgm.beams[0].pointing_angle_deg == pytest.approx(45.0)
    assert dgm.beams[0].detected_range_samples == 10
    assert dgm.beams[0].samples == (-50, -55, -60)
    assert dgm.beams[1].pointing_angle_deg == pytest.approx(-45.0)
    assert dgm.beams[1].detected_range_samples == 12
    assert dgm.beams[1].samples == (-70, -75, -80)


def test_iter_wcd_files_on_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        wcd.iter_wcd_files(Path("/nonexistent/path/that/does/not/exist"))


# TODO: add fixture-based integration test once a small .wcd sample is
# committed under tests/fixtures/. The clipper in clip_datagrams.py works
# unchanged for .wcd (same envelope as .all) — just clip by counting K
# datagrams instead of X pings.
