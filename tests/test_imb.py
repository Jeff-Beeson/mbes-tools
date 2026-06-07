"""Tests for mbes_tools.imb (Kongsberg M3 .imb / .002 packet reader)."""
import struct
from pathlib import Path

import numpy as np
import pytest

from mbes_tools import imb
from mbes_tools.imb import (
    BEAM_LIST_OFFSET,
    FOOTER_SIZE,
    SYNC_WORD,
    _EXT_SIZE,
    _FIXED_FMT,
    _FIXED_SIZE,
    _EXT_FMT,
    ImbPacket,
    iter_imb_packets,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_packet(
    *,
    version: int = 1,
    sonar_id: int = 42,
    time_sec: int = 1_700_000_000,
    time_msec: int = 123,
    vel_sound: float = 1500.0,
    num_samples: int = 10,
    near_range: float = 1.0,
    far_range: float = 50.0,
    swst: float = 0.001,
    swl: float = 0.066,
    num_beams: int = 4,
    proc_type: int = 0,
    beam_angles: list | None = None,
    intensity: list | None = None,
    include_extended: bool = False,
) -> bytes:
    """Build a minimal synthetic IMB packet with known field values."""
    if beam_angles is None:
        beam_angles = [float(i) for i in range(num_beams)]
    if intensity is None:
        intensity = [float(i) for i in range(num_beams * num_samples)]

    # Fixed header (124 bytes, written after the sync word)
    sonar_info = (0,) * 20
    fixed = struct.pack(
        _FIXED_FMT,
        version, sonar_id,
        *sonar_info,
        time_sec, time_msec,
        vel_sound,
        num_samples,
        near_range, far_range, swst, swl,
        num_beams, proc_type,
    )
    assert len(fixed) == _FIXED_SIZE

    beam_bytes = struct.pack(f"<{num_beams}f", *beam_angles)

    body_bytes = struct.pack(f"<{num_beams * num_samples}f", *intensity)

    ext_bytes = b""
    if include_extended:
        # Build a zeroed extended header (all floats/ints = 0).
        ext_bytes = b"\x00" * _EXT_SIZE

    footer = b"\x00" * FOOTER_SIZE

    return SYNC_WORD + fixed + beam_bytes + ext_bytes + body_bytes + footer


def _write_imb(tmp_path: Path, packets: list[bytes], name: str = "test.002") -> Path:
    p = tmp_path / name
    p.write_bytes(b"".join(packets))
    return p


# ---------------------------------------------------------------------------
# Struct sanity checks
# ---------------------------------------------------------------------------

def test_fixed_header_size():
    assert _FIXED_SIZE == 124


def test_ext_header_size():
    assert _EXT_SIZE == 4340


def test_beam_list_offset():
    assert BEAM_LIST_OFFSET == 132


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------

def test_module_exposes_api():
    for name in ["iter_imb_packets", "ImbPacket", "SYNC_WORD", "FOOTER_SIZE"]:
        assert hasattr(imb, name), f"missing: {name}"


# ---------------------------------------------------------------------------
# Parsing — basic field extraction
# ---------------------------------------------------------------------------

def test_single_packet_fields(tmp_path):
    pkt_bytes = _build_packet(
        version=3,
        sonar_id=99,
        time_sec=1_700_000_500,
        time_msec=250,
        vel_sound=1490.5,
        num_beams=4,
        num_samples=8,
        near_range=2.0,
        far_range=75.0,
        swst=0.002,
        swl=0.1,
        proc_type=1,
    )
    # Two copies so the first packet has a known length.
    path = _write_imb(tmp_path, [pkt_bytes, pkt_bytes])
    pkts = list(iter_imb_packets(path, min_packet_bytes=0))

    assert len(pkts) >= 1
    p = pkts[0]
    assert p.version == 3
    assert p.sonar_id == 99
    assert p.time_sec == 1_700_000_500
    assert p.time_msec == 250
    assert pytest.approx(p.velocity_sound_ms, rel=1e-5) == 1490.5
    assert p.num_beams == 4
    assert p.num_samples == 8
    assert pytest.approx(p.near_range_m) == 2.0
    assert pytest.approx(p.far_range_m) == 75.0
    assert p.processing_type == 1


def test_beam_angles(tmp_path):
    angles = [-30.0, -10.0, 10.0, 30.0]
    pkt_bytes = _build_packet(num_beams=4, num_samples=5, beam_angles=angles)
    path = _write_imb(tmp_path, [pkt_bytes, pkt_bytes])
    pkts = list(iter_imb_packets(path, min_packet_bytes=0))

    assert len(pkts) >= 1
    np.testing.assert_allclose(pkts[0].beam_angles_deg, angles, rtol=1e-5)


def test_intensity_shape_and_values(tmp_path):
    n_beams, n_samples = 3, 6
    vals = [float(i) for i in range(n_beams * n_samples)]
    pkt_bytes = _build_packet(num_beams=n_beams, num_samples=n_samples, intensity=vals)
    path = _write_imb(tmp_path, [pkt_bytes, pkt_bytes])
    pkts = list(iter_imb_packets(path, min_packet_bytes=0))

    p = pkts[0]
    assert p.intensity.shape == (n_beams, n_samples)
    np.testing.assert_allclose(p.intensity.flatten(), vals, rtol=1e-5)


def test_timestamp_property(tmp_path):
    from datetime import timezone

    pkt_bytes = _build_packet(time_sec=1_700_000_000, time_msec=500)
    path = _write_imb(tmp_path, [pkt_bytes, pkt_bytes])
    pkts = list(iter_imb_packets(path, min_packet_bytes=0))

    ts = pkts[0].timestamp
    assert ts.tzinfo == timezone.utc
    assert ts.timestamp() == pytest.approx(1_700_000_000.5, rel=1e-9)


# ---------------------------------------------------------------------------
# Parsing — multiple packets
# ---------------------------------------------------------------------------

def test_multiple_packets(tmp_path):
    pkts_bytes = [
        _build_packet(sonar_id=i, num_beams=2, num_samples=4)
        for i in range(5)
    ]
    # Append an extra copy of the last packet so all have a measured length.
    pkts_bytes.append(pkts_bytes[-1])
    path = _write_imb(tmp_path, pkts_bytes)
    results = list(iter_imb_packets(path, min_packet_bytes=0))
    # At most 6 packets parsed; at minimum the 5 non-trivial ones.
    assert len(results) >= 5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_file(tmp_path):
    path = tmp_path / "empty.002"
    path.write_bytes(b"")
    assert list(iter_imb_packets(path)) == []


def test_no_sync_words(tmp_path):
    path = tmp_path / "noise.002"
    path.write_bytes(b"\xff" * 1024)
    assert list(iter_imb_packets(path)) == []


def test_short_packet_skipped(tmp_path):
    """Packets below min_packet_bytes should be silently skipped.

    Packet length is measured as the distance to the NEXT sync word.  The
    last packet in a file has no successor, so its length is unknown and it
    is always attempted (not filtered by min_packet_bytes).  This test uses
    three packets so the middle one has a measurable length.
    """
    pkt = _build_packet(num_beams=2, num_samples=4)
    threshold = len(pkt) + 1
    # File: [pkt0][pkt1][pkt2]
    # pkt0 length = len(pkt) < threshold → skipped
    # pkt1 length = len(pkt) < threshold → skipped
    # pkt2 length = unknown (last) → attempted
    path = _write_imb(tmp_path, [pkt, pkt, pkt])
    results = list(iter_imb_packets(path, min_packet_bytes=threshold))
    # Only the last packet (unknown length) should be parsed.
    assert len(results) == 1


def test_min_packet_bytes_zero_accepts_all(tmp_path):
    pkt = _build_packet(num_beams=2, num_samples=3)
    path = _write_imb(tmp_path, [pkt, pkt])
    results = list(iter_imb_packets(path, min_packet_bytes=0))
    assert len(results) >= 1


def test_zero_beams_skipped(tmp_path):
    """Packets with nNumBeams == 0 should be discarded."""
    bad = _build_packet(num_beams=0, num_samples=10)
    good = _build_packet(num_beams=2, num_samples=10)
    path = _write_imb(tmp_path, [bad, good, good])
    results = list(iter_imb_packets(path, min_packet_bytes=0))
    assert all(p.num_beams > 0 for p in results)


# ---------------------------------------------------------------------------
# Extended header (zeroed — verifies offset arithmetic, not real values)
# ---------------------------------------------------------------------------

def test_extended_header_fields_default_none(tmp_path):
    """Without extended=True, all nav fields should be None."""
    pkt = _build_packet(num_beams=2, num_samples=3)
    path = _write_imb(tmp_path, [pkt, pkt])
    results = list(iter_imb_packets(path, min_packet_bytes=0))
    p = results[0]
    assert p.ping_number is None
    assert p.latitude_deg is None
    assert p.heading_deg is None


def test_extended_header_parsed_when_enabled(tmp_path):
    """With extended=True and a zeroed extended header, fields should be 0.0."""
    pkt = _build_packet(num_beams=2, num_samples=3, include_extended=True)
    path = _write_imb(tmp_path, [pkt, pkt])
    results = list(iter_imb_packets(path, min_packet_bytes=0, extended=True))
    assert len(results) >= 1
    p = results[0]
    assert p.ping_number == 0
    assert p.latitude_deg == pytest.approx(0.0)
    assert p.longitude_deg == pytest.approx(0.0)
    assert p.heading_deg == pytest.approx(0.0)
