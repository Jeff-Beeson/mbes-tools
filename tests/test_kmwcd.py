"""Tests for mbes_tools.kmwcd (Kongsberg .kmwcd water column reader).

Status: synthetic-byte tests only. Fixture-based tests pending a real
.kmwcd sample file. The #MWC parser is written from spec + UNH-CCOM
cross-reference and has not yet seen real-world bytes — synthetic tests
verify the binary layout math but cannot catch spec-versus-reality
mismatches.
"""
import struct
from pathlib import Path

import pytest

from mbes_tools import kmwcd
from mbes_tools.kmwcd import (
    MWC_RX_BEAM_V1_FMT,
    MWC_RX_BEAM_V1_SIZE,
    MWC_RX_BEAM_V2_FMT,
    MWC_RX_BEAM_V2_SIZE,
    MWC_RX_INFO_FMT,
    MWC_RX_INFO_SIZE,
    MWC_TX_INFO_FMT,
    MWC_TX_INFO_SIZE,
    MWC_TX_SECTOR_FMT,
    MWC_TX_SECTOR_SIZE,
    MWCDatagram,
)
from mbes_tools.kmall import (
    DGM_HEADER_FMT,
    MRZ_CMN_PART_FMT,
    MRZ_PARTITION_FMT,
)


def test_kmwcd_module_exposes_api():
    """The .kmwcd parser API is importable."""
    for name in [
        "iter_mwc_datagrams",
        "iter_kmwcd_files",
        "parse_mwc_datagram",
        "MWCDatagram",
        "MWCRxBeam",
        "MWCTxSector",
    ]:
        assert hasattr(kmwcd, name), f"missing {name}"


def test_struct_sizes():
    """Binary format sizes should match the documented #MWC layout."""
    # TX Info: 3H + h + f = 6 + 2 + 4 = 12.
    assert kmwcd.MWC_TX_INFO_SIZE == 12
    # TX Sector: 3f + H + h = 12 + 2 + 2 = 16.
    assert kmwcd.MWC_TX_SECTOR_SIZE == 16
    # RX Info: 2H + 3B + b + 2f = 4 + 3 + 1 + 8 = 16.
    assert kmwcd.MWC_RX_INFO_SIZE == 16
    # RX Beam v1: f + 4H = 4 + 8 = 12.
    assert kmwcd.MWC_RX_BEAM_V1_SIZE == 12
    # RX Beam v2: f + 4H + f = 4 + 8 + 4 = 16.
    assert kmwcd.MWC_RX_BEAM_V2_SIZE == 16


def _build_mwc_datagram(
    num_tx_sectors: int = 1,
    num_beams: int = 2,
    n_samples_per_beam: int = 3,
    dgm_version: int = 2,
    phase_flag: int = 0,
) -> bytes:
    """Build a synthetic .kmwcd #MWC datagram for parser testing."""
    # Partition: numDatagramsInRecord=1, datagramNum=1.
    partition = struct.pack(MRZ_PARTITION_FMT, 1, 1)

    # cmnPart: numBytesCmnPart, pingCnt, rxFansPerPing, rxFanIndex,
    # swathsPerPing, swathAlongPosition, txTransducerInd, rxTransducerInd,
    # numRxTransducers, algorithmType.
    cmn_part = struct.pack(MRZ_CMN_PART_FMT, 12, 100, 1, 0, 1, 0, 0, 0, 1, 0)

    # TX Info: 12 bytes; numBytesTxInfo=12, numTxSectors, numBytesPerTxSector=16,
    # padding=0, heave_m=1.5.
    tx_info = struct.pack(MWC_TX_INFO_FMT, 12, num_tx_sectors, 16, 0, 1.5)

    # TX sectors.
    tx_sectors = b""
    for i in range(num_tx_sectors):
        tx_sectors += struct.pack(
            MWC_TX_SECTOR_FMT, 2.0, 40000.0, 1.5, i, 0,
        )

    # RX Info: numBytesRxInfo=16, numBeams, numBytesPerBeamEntry,
    # phaseFlag, TVGfunctionApplied=1, TVGoffset_dB=-3,
    # sampleFreq_Hz=80000, soundVelocity_mPerSec=1500.
    beam_entry_size = MWC_RX_BEAM_V2_SIZE if dgm_version >= 2 else MWC_RX_BEAM_V1_SIZE
    rx_info = struct.pack(
        MWC_RX_INFO_FMT,
        16, num_beams, beam_entry_size, phase_flag, 1, -3, 80000.0, 1500.0,
    )

    # Beams + amplitude samples (+ optional phase samples).
    beams = b""
    for i in range(num_beams):
        if dgm_version >= 2:
            beam_hdr = struct.pack(
                MWC_RX_BEAM_V2_FMT, 45.0 - i * 10, 0, 10 + i, 0, n_samples_per_beam, 10.5,
            )
        else:
            beam_hdr = struct.pack(
                MWC_RX_BEAM_V1_FMT, 45.0 - i * 10, 0, 10 + i, 0, n_samples_per_beam,
            )
        amplitudes = struct.pack(
            f"<{n_samples_per_beam}b",
            *[-50 - i * 10 - j for j in range(n_samples_per_beam)],
        )
        beams += beam_hdr + amplitudes
        if phase_flag == 1 and n_samples_per_beam > 0:
            beams += struct.pack(
                f"<{n_samples_per_beam}b",
                *[10 + j for j in range(n_samples_per_beam)],
            )
        elif phase_flag == 2 and n_samples_per_beam > 0:
            beams += struct.pack(
                f"<{n_samples_per_beam}h",
                *[100 + j for j in range(n_samples_per_beam)],
            )

    body = partition + cmn_part + tx_info + tx_sectors + rx_info + beams
    total_size = 20 + len(body)  # 20-byte envelope INCLUDED in total per kmall convention

    envelope = struct.pack(
        DGM_HEADER_FMT,
        total_size, b"#MWC", dgm_version, 0, 304, 1762929180, 0,
    )
    return envelope + body


def test_parse_mwc_v2_roundtrip(tmp_path):
    """Build a v2 #MWC datagram with 1 TX sector + 2 beams + 3 samples each."""
    blob = _build_mwc_datagram(
        num_tx_sectors=1, num_beams=2, n_samples_per_beam=3, dgm_version=2,
    )
    f = tmp_path / "synthetic.kmwcd"
    f.write_bytes(blob)

    records = list(kmwcd.iter_mwc_datagrams(f))
    assert len(records) == 1
    dgm = records[0]

    assert isinstance(dgm, MWCDatagram)
    assert dgm.dgm_version == 2
    assert dgm.echo_sounder_id == 304
    assert dgm.ping_cnt == 100
    assert dgm.num_tx_sectors == 1
    assert dgm.num_beams == 2
    assert dgm.heave_m == pytest.approx(1.5)
    assert dgm.phase_flag == 0
    assert dgm.tvg_function_applied == 1
    assert dgm.tvg_offset_db == -3
    assert dgm.sample_freq_hz == pytest.approx(80_000.0)
    assert dgm.sound_velocity_m_s == pytest.approx(1500.0)

    assert len(dgm.tx_sectors) == 1
    assert dgm.tx_sectors[0].tilt_angle_re_tx_deg == pytest.approx(2.0)
    assert dgm.tx_sectors[0].centre_freq_hz == pytest.approx(40_000.0)
    assert dgm.tx_sectors[0].tx_sector_num == 0

    assert len(dgm.beams) == 2
    assert dgm.beams[0].beam_point_angle_re_vertical_deg == pytest.approx(45.0)
    assert dgm.beams[0].num_sample_data == 3
    assert dgm.beams[0].detected_range_samples_high_res == pytest.approx(10.5)
    assert dgm.beams[0].sample_amplitudes_0p5_db == (-50, -51, -52)
    assert dgm.beams[1].sample_amplitudes_0p5_db == (-60, -61, -62)
    assert dgm.beams[0].phase_samples == ()


def test_parse_mwc_v1_roundtrip(tmp_path):
    """v1 layout: beam header has no detectedRangeInSamplesHighResolution field."""
    blob = _build_mwc_datagram(
        num_tx_sectors=1, num_beams=1, n_samples_per_beam=2, dgm_version=1,
    )
    f = tmp_path / "v1.kmwcd"
    f.write_bytes(blob)

    dgms = list(kmwcd.iter_mwc_datagrams(f))
    assert len(dgms) == 1
    assert dgms[0].dgm_version == 1
    assert dgms[0].beams[0].detected_range_samples_high_res is None
    assert dgms[0].beams[0].sample_amplitudes_0p5_db == (-50, -51)


def test_parse_mwc_with_phase_flag_1(tmp_path):
    """phase_flag=1 -> int8 phase samples follow amplitude samples per beam."""
    blob = _build_mwc_datagram(
        num_tx_sectors=1, num_beams=1, n_samples_per_beam=3,
        dgm_version=2, phase_flag=1,
    )
    f = tmp_path / "phase1.kmwcd"
    f.write_bytes(blob)

    dgms = list(kmwcd.iter_mwc_datagrams(f))
    assert len(dgms) == 1
    assert dgms[0].phase_flag == 1
    assert dgms[0].beams[0].sample_amplitudes_0p5_db == (-50, -51, -52)
    assert dgms[0].beams[0].phase_samples == (10, 11, 12)


def test_iter_kmwcd_files_on_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        kmwcd.iter_kmwcd_files(Path("/nonexistent/path/that/does/not/exist"))


# TODO: add fixture-based integration test once a real .kmwcd sample is
# committed under tests/fixtures/. The clip_datagrams.py clipper works
# unchanged for .kmwcd (same envelope as .kmall) — just clip by counting
# #MWC datagrams instead of #MRZ.
