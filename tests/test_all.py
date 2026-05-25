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
