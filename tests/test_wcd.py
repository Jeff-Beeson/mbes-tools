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


# ---------------------------------------------------------------------------
# Fixture-based integration test against REAL k-datagram bytes (Capability D1).
#
# Real EM122 deep-water water column (R/V Atlantis). Clipped from
# 0197_20130703_155651_Atlantis.wcd to three k datagrams via
# tests/fixtures/clip_datagrams.py --target-type k -n 3. On the full source
# file all 1407 k datagrams reconcile to the byte (predicted body == actual,
# modulo the single Kongsberg spare byte before the footer); these three are
# complete, unfragmented pings (counters 9901-9903).
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"
WCD_FIXTURE = FIXTURES / "sample_atlantis_em122.wcd"


@pytest.mark.skipif(not WCD_FIXTURE.exists(), reason="wcd fixture not present")
def test_real_em122_wcd():
    """The real EM122 .wcd parses; geometry and per-beam sample arrays are sane,
    and predicted body size matches the declared length (no byte drift)."""
    dgms = list(wcd.iter_water_column_datagrams(WCD_FIXTURE))
    assert len(dgms) == 3

    d0 = dgms[0]
    assert d0.num_tx_sectors == 8
    assert d0.num_beams_this_datagram == 288
    assert len(d0.beams) == 288
    # EM122 is a ~12 kHz system: every sector's centre frequency is near 12 kHz.
    assert all(10_000.0 < s.centre_frequency_hz < 14_000.0 for s in d0.tx_sectors)
    assert 1450.0 < d0.sound_speed_m_s < 1600.0
    # Beam pointing angles sweep monotonically across the swath.
    angles = [b.pointing_angle_deg for b in d0.beams]
    assert angles == sorted(angles, reverse=True)
    # Three consecutive, complete (single-datagram) pings.
    assert [d.counter for d in dgms] == [9901, 9902, 9903]
    assert all(d.num_datagram == 1 and d.datagram_num == 1 for d in dgms)

    # Per-beam reconciliation: K body + sectors + Σ(beam header + N samples)
    # must exactly consume record.body up to the single trailing spare byte.
    from mbes_tools.all import iter_datagrams

    for rec in iter_datagrams(WCD_FIXTURE):
        if rec.header.type_of_datagram != "k":
            continue
        body = rec.body
        s = struct.unpack_from(K_BODY_FMT, body, 0)
        num_tx, num_beams_this = s[4], s[6]
        cur = wcd.K_BODY_SIZE + num_tx * wcd.K_TX_SECTOR_SIZE
        for _ in range(num_beams_this):
            n_samps = struct.unpack_from(K_BEAM_FMT, body, cur)[2]
            cur += wcd.K_BEAM_SIZE + n_samps
        # Exactly one Kongsberg spare byte remains before the footer.
        assert len(body) - cur == 1
