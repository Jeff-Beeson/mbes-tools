"""Tests for mbes_tools.all (Kongsberg .all reader)."""
import struct
from pathlib import Path

import pytest

from mbes_tools import all as mbes_all  # 'all' shadows the builtin, alias it
from mbes_tools.all import (
    DGM_HEADER_FMT,
    DGM_FOOTER_FMT,
    DatagramHeader,
    DatagramRecord,
    PositionDatagram,
    DepthDatagram,
    SeabedImageDatagram,
    XBeam,
    YBeam,
)


def test_all_module_exposes_api():
    """The native .all parser API is importable."""
    for name in [
        "iter_datagrams",
        "iter_all_files",
        "iter_position_datagrams",
        "iter_depth_datagrams",
        "iter_seabed_image_datagrams",
        "parse_position",
        "parse_depth",
        "parse_seabed_image",
    ]:
        assert hasattr(mbes_all, name), f"missing {name}"


def test_struct_sizes_match_kongsberg_spec():
    """Binary format sizes should match the documented Kongsberg layout."""
    # 12-byte envelope header.
    assert struct.calcsize(DGM_HEADER_FMT) == 16  # L+B+B+H+L+L = 4+1+1+2+4+4
    # 3-byte footer (ETX + checksum).
    assert struct.calcsize(DGM_FOOTER_FMT) == 3
    # P body (counter, serial, lat, lon, 4*u16, 2*u8) = 2+2+4+4+8+2 = 22
    assert mbes_all.P_BODY_SIZE == 22
    # X body (4*u16 + f32 + 2*u16 + f32 + 4*u8) = 8+4+4+4+4 = 24
    assert mbes_all.X_BODY_SIZE == 24
    # X per-beam (3*f32 + u16 + 4*u8 + i16) = 12+2+4+2 = 20
    assert mbes_all.X_BEAM_SIZE == 20
    # Y body (2*u16 + f32 + u16 + 2*i16 + 3*u16) = 4+4+2+4+6 = 20
    assert mbes_all.Y_BODY_SIZE == 20
    # Y per-beam (i8 + u8 + 2*u16) = 1+1+4 = 6
    assert mbes_all.Y_BEAM_SIZE == 6


def test_datagram_header_time_seconds():
    """time_seconds converts the integer ms-since-midnight to seconds."""
    hdr = DatagramHeader(
        number_of_bytes=100, stx=0x02, type_of_datagram="P",
        em_model=302, record_date=20250215, record_time_ms=43_200_000,  # noon
    )
    assert hdr.time_seconds == 43200.0


def _build_envelope(type_char: str, body: bytes, em_model: int = 302,
                    date: int = 20250215, time_ms: int = 0) -> bytes:
    """Build a synthetic .all datagram envelope for testing.

    Layout: 16-byte header + body + 3-byte footer (ETX + checksum).
    """
    footer = struct.pack(DGM_FOOTER_FMT, 0x03, 0)
    nbytes = 16 - 4 + len(body) + len(footer)
    header = struct.pack(
        DGM_HEADER_FMT, nbytes, 0x02, ord(type_char), em_model, date, time_ms,
    )
    return header + body + footer


def test_iter_datagrams_walks_synthetic_file(tmp_path):
    """Iterator yields each datagram with correct type and body length."""
    body_p = b"\x00" * 22  # min P body, no input datagram
    body_x = b"\x00" * 24  # min X body, zero beams
    body_y = b"\x00" * 20  # min Y body, zero beams

    blob = (
        _build_envelope("P", body_p)
        + _build_envelope("X", body_x)
        + _build_envelope("Y", body_y)
    )

    f = tmp_path / "synthetic.all"
    f.write_bytes(blob)

    records = list(mbes_all.iter_datagrams(f))
    assert len(records) == 3
    assert [r.header.type_of_datagram for r in records] == ["P", "X", "Y"]
    assert records[0].header.em_model == 302
    # body lengths should match what we wrote.
    assert len(records[0].body) == 22
    assert len(records[1].body) == 24
    assert len(records[2].body) == 20
    # offsets should be monotonically increasing.
    assert records[0].offset == 0
    assert records[1].offset > 0
    assert records[2].offset > records[1].offset


def test_parse_position_roundtrip(tmp_path):
    """Build a synthetic P datagram, parse it, check the decoded fields."""
    lat_deg = 36.6066    # Monterey Bay
    lon_deg = -121.8957
    body = struct.pack(
        "<HHll4HBB",
        42,                                  # counter
        1234,                                # serialNumber
        int(round(lat_deg * 20_000_000)),    # latitude
        int(round(lon_deg * 10_000_000)),    # longitude
        25,                                  # quality (cm * 100)
        500,                                 # SOG (cm/s)
        12_000,                              # COG (0.01 deg)
        9_000,                               # heading (0.01 deg)
        1,                                   # descriptor
        0,                                   # NBytesInInputDatagram
    )
    blob = _build_envelope("P", body)
    f = tmp_path / "p.all"
    f.write_bytes(blob)

    records = list(mbes_all.iter_position_datagrams(f))
    assert len(records) == 1
    p = records[0]

    assert isinstance(p, PositionDatagram)
    assert p.counter == 42
    assert p.serial_number == 1234
    assert p.latitude_deg == pytest.approx(lat_deg, abs=1e-7)
    assert p.longitude_deg == pytest.approx(lon_deg, abs=1e-7)
    assert p.position_fix_quality_m == pytest.approx(0.25)
    assert p.speed_over_ground_m_s == pytest.approx(5.0)
    assert p.course_over_ground_deg == pytest.approx(120.0)
    assert p.heading_deg == pytest.approx(90.0)


def test_parse_seabed_image_roundtrip(tmp_path):
    """Build a synthetic Y datagram with two beams, parse it back."""
    # 2 beams, each with 3 samples.
    beam_headers = (
        struct.pack("<bBHH", 0, 1, 3, 1)    # beam 0
        + struct.pack("<bBHH", 0, 1, 3, 2)  # beam 1
    )
    # Samples: int16 in units of 0.1 dB.
    samples_beam0 = struct.pack("<3h", -250, -260, -270)  # -25.0, -26.0, -27.0 dB
    samples_beam1 = struct.pack("<3h", -300, -310, -320)

    body = (
        struct.pack(
            "<HHfHhhHHH",
            7,        # counter
            5678,     # serial
            400_000.0,# sample freq Hz
            120,      # range to normal incidence (samples)
            -25,      # normal incidence BS (dB)
            -35,      # oblique BS (dB)
            10,       # tx beam width (0.1 deg) -> 1.0 deg
            450,      # TVG crossover (0.1 deg) -> 45.0 deg
            2,        # num beams
        )
        + beam_headers
        + samples_beam0
        + samples_beam1
    )

    blob = _build_envelope("Y", body)
    f = tmp_path / "y.all"
    f.write_bytes(blob)

    records = list(mbes_all.iter_seabed_image_datagrams(f))
    assert len(records) == 1
    y = records[0]

    assert isinstance(y, SeabedImageDatagram)
    assert y.counter == 7
    assert y.num_beams == 2
    assert y.tx_beam_width_deg == pytest.approx(1.0)
    assert len(y.beams) == 2
    assert y.beams[0].samples == (-250, -260, -270)
    assert y.beams[1].samples == (-300, -310, -320)


def test_iter_all_files_on_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        mbes_all.iter_all_files(Path("/nonexistent/path/that/does/not/exist"))


def test_struct_sizes_n_and_r():
    """N (78) and R (82) byte layouts match the documented Kongsberg sizes."""
    # N body: ping,serial,ssp,ntx,nrx,nvalid (6*u16) + sampleFreq f32 + Dscale u32 = 12+4+4
    assert mbes_all.N_BODY_SIZE == 20
    # N tx sector: i16+u16+3*f32+u16+2*u8+f32 = 2+2+12+2+2+4
    assert mbes_all.N_TX_SIZE == 24
    # N rx beam: i16+2*u8+u16+u8+i8+f32+i16+i8+u8 = 2+2+2+1+1+4+2+1+1
    assert mbes_all.N_RX_SIZE == 16
    # R body (header to filterIdentifier2): 37 bytes; +16 header +3 footer = 56.
    assert mbes_all.R_BODY_SIZE == 37


def test_parse_raw_range_angle_roundtrip(tmp_path):
    """Build a synthetic N datagram with 1 sector + 2 beams, parse it back."""
    body = struct.pack("<HHHHHHfL", 100, 7, 15000, 1, 2, 2, 30000.0, 0)
    body += struct.pack("<hHfffHBBf", 0, 0, 0.0, 0.0, 30000.0, 0, 0, 0, 0.0)  # tx sector
    # beam 0: +25.00 deg, sector 0, valid, reflectivity -20.0 dB
    body += struct.pack("<hBBHBbfhbB", 2500, 0, 0, 10, 5, 0, 0.05, -200, 0, 0)
    # beam 1: -30.00 deg, sector 1, invalid (detInfo bit 7), reflectivity -25.0 dB
    body += struct.pack("<hBBHBbfhbB", -3000, 1, 0x80, 10, 5, 0, 0.06, -250, 0, 0)

    f = tmp_path / "n.all"
    f.write_bytes(_build_envelope("N", body, em_model=2040))

    dgms = list(mbes_all.iter_raw_range_angle_datagrams(f))
    assert len(dgms) == 1
    n = dgms[0]
    assert n.ping_counter == 100
    assert n.num_rx_beams == 2
    assert n.sound_speed_m_s == pytest.approx(1500.0)
    assert n.beams[0].beam_pointing_angle_deg == pytest.approx(25.0)
    assert n.beams[0].tx_sector_number == 0
    assert n.beams[0].reflectivity_db == pytest.approx(-20.0)
    assert n.beams[0].is_valid is True
    assert n.beams[1].beam_pointing_angle_deg == pytest.approx(-30.0)
    assert n.beams[1].tx_sector_number == 1
    assert n.beams[1].is_valid is False


def test_parse_runtime_roundtrip(tmp_path):
    """Build a synthetic R datagram and check the decoded mode byte."""
    body = struct.pack(
        "<HHBBBBBBHHHHHbBBBBBHBBBBHhB",
        55,    # ping counter
        7,     # serial
        0, 0, 0, 0,   # status bytes
        3,     # mode  (general model bit pattern -> Deep)
        6,     # filter identifier
        2, 200,        # min/max depth
        6522,          # absorption *100 -> 65.22
        300,           # tx pulse length
        10,            # tx beamwidth
        0,             # tx power (i8)
        10, 8, 0, 40, 1,   # rxBW, rxBandwidth, mode2, tvg, srcSS
        500,           # max port width
        2, 70, 1, 70,  # beamSpacing, maxPortCov, yawMode, maxStbdCov
        500,           # max stbd width
        0,             # tx along tilt (i16)
        0,             # filter id 2
    )
    f = tmp_path / "r.all"
    f.write_bytes(_build_envelope("R", body, em_model=302))

    dgms = list(mbes_all.iter_runtime_datagrams(f))
    assert len(dgms) == 1
    r = dgms[0]
    assert r.ping_counter == 55
    assert r.mode == 3
    assert r.filter_identifier == 6
    assert r.minimum_depth_m == 2
    assert r.maximum_depth_m == 200
    assert r.absorption_coefficient_db_km == pytest.approx(65.22, abs=1e-2)


def test_iter_datagrams_type_filter(tmp_path):
    """The types= filter yields only requested datagrams (others seeked past)."""
    blob = (
        _build_envelope("P", b"\x00" * 22)
        + _build_envelope("X", b"\x00" * 24)
        + _build_envelope("Y", b"\x00" * 20)
    )
    f = tmp_path / "mixed.all"
    f.write_bytes(blob)
    kept = [r.header.type_of_datagram for r in mbes_all.iter_datagrams(f, types={"X", "Y"})]
    assert kept == ["X", "Y"]


# --- N/R fixture tests (real EM2040 .all, clipped) -------------------------


@pytest.fixture
def sample_em2040_all():
    p = FIXTURES / "sample_equinox_em2040.all"
    if not p.exists():
        pytest.skip(f"Fixture not present: {p}")
    return p


def test_fixture_raw_range_angle(sample_em2040_all):
    """N datagrams parse with sane angle/sector and pair with X by ping counter."""
    ns = list(mbes_all.iter_raw_range_angle_datagrams(sample_em2040_all))
    xs = list(mbes_all.iter_depth_datagrams(sample_em2040_all))
    assert ns and xs
    n, x = ns[0], xs[0]
    assert n.ping_counter == x.counter            # X and N share the ping counter
    assert n.num_rx_beams == x.num_beams          # and beam count
    angles = [b.beam_pointing_angle_deg for b in n.beams]
    assert min(angles) < -20 and max(angles) > 20  # real swath spans nadir
    # X valid-beam count equals the file's own num_valid_detections (validity check).
    assert sum(1 for b in x.beams if b.is_valid) == x.num_valid_detections


def test_fixture_runtime_mode_is_em2040_300khz(sample_em2040_all):
    """The Equinox_2040_300kHz file's runtime mode decodes to 300 kHz."""
    from mbes_tools.depth_modes import all_runtime_mode_info

    rs = list(mbes_all.iter_runtime_datagrams(sample_em2040_all))
    assert rs
    r = rs[0]
    assert r.header.em_model == 2040
    assert all_runtime_mode_info(r.header.em_model, r.mode) == (1, "300kHz")


# ---------------------------------------------------------------------------
# Fixture-based integration tests.
# Source: tests/fixtures/sample_nautilus.all, clipped from a Nautilus
# (E/V Nautilus, 2025-05-07) Kongsberg .all via
# tests/fixtures/clip_datagrams.py --pings 3.
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_all():
    """Path to the committed sample_nautilus.all, or skip if absent."""
    p = FIXTURES / "sample_nautilus.all"
    if not p.exists():
        pytest.skip(f"Fixture not present: {p}")
    return p


def test_fixture_total_datagram_count(sample_all):
    """Clipper reported 154 datagrams in the clip; iterator should match."""
    records = list(mbes_all.iter_datagrams(sample_all))
    assert len(records) == 154


def test_fixture_depth_pings(sample_all):
    """Clipper stopped after 3 X (depth) datagrams; parser should find them."""
    pings = list(mbes_all.iter_depth_datagrams(sample_all))
    assert len(pings) == 3
    for ping in pings:
        assert ping.num_beams > 0
        assert ping.num_valid_detections > 0
        assert ping.num_valid_detections <= ping.num_beams
        assert ping.sound_speed_at_transducer_m_s > 1400  # reasonable seawater range
        assert ping.sound_speed_at_transducer_m_s < 1600
        assert len(ping.beams) == ping.num_beams


def test_fixture_seabed_image_pings(sample_all):
    """Real survey should yield at least one Y (seabed image) datagram."""
    y_records = list(mbes_all.iter_seabed_image_datagrams(sample_all))
    assert len(y_records) > 0
    for y in y_records:
        assert y.num_beams > 0
        assert len(y.beams) == y.num_beams
        # At least one beam should carry samples.
        total_samples = sum(b.number_of_samples_per_beam for b in y.beams)
        assert total_samples > 0


def test_fixture_position_records_have_valid_coords(sample_all):
    """P datagrams should carry real-world lat/lon."""
    positions = list(mbes_all.iter_position_datagrams(sample_all))
    assert len(positions) > 0
    for p in positions:
        assert -90.0 <= p.latitude_deg <= 90.0
        assert -180.0 <= p.longitude_deg <= 180.0
        assert 0.0 <= p.heading_deg < 360.0
