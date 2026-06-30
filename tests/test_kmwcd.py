"""Tests for mbes_tools.kmwcd (Kongsberg .kmwcd water column reader).

Status: synthetic-byte tests only. Fixture-based tests pending a real
.kmwcd sample file. The #MWC parser is written from spec + UNH-CCOM
cross-reference and has not yet seen real-world bytes — synthetic tests
verify the binary layout math but cannot catch spec-versus-reality
mismatches.
"""
import os
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


def test_parse_mwc_with_phase_flag_2(tmp_path):
    """phase_flag=2 -> int16 phase samples follow amplitude samples per beam."""
    blob = _build_mwc_datagram(
        num_tx_sectors=1, num_beams=1, n_samples_per_beam=3,
        dgm_version=2, phase_flag=2,
    )
    f = tmp_path / "phase2.kmwcd"
    f.write_bytes(blob)

    dgms = list(kmwcd.iter_mwc_datagrams(f))
    assert len(dgms) == 1
    assert dgms[0].phase_flag == 2
    assert dgms[0].beams[0].sample_amplitudes_0p5_db == (-50, -51, -52)
    # int16 phase: _build_mwc_datagram writes 100, 101, 102.
    assert dgms[0].beams[0].phase_samples == (100, 101, 102)


def test_iter_kmwcd_files_on_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        kmwcd.iter_kmwcd_files(Path("/nonexistent/path/that/does/not/exist"))


# ---------------------------------------------------------------------------
# Fixture-based integration tests against REAL #MWC bytes (Capability D1).
#
# Validation method: predict each #MWC datagram's total size from the struct
# sizes plus the per-beam block formula and compare to the file's declared
# numBytesDgm. A wrong struct size or phase element size will not reconcile to
# the datagram boundary across many variable-length beams, so this check pins
# the layout (including the phaseFlag 1/2 sample sizes) against real bytes.
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"
DGM_HEADER_SIZE = 20
PARTITION_SIZE = 4
_PHASE_SIZE = {0: 0, 1: 1, 2: 2}


def _predict_mwc_total(fid, datagram_start: int) -> int:
    """Recompute one #MWC datagram's total byte size straight from the file,
    independently of mbes_tools.kmwcd, and return the predicted total
    (which should equal the declared numBytesDgm)."""
    fid.seek(datagram_start)
    num_bytes_dgm, _t, ver, _sys, _echo, _ts, _tns = struct.unpack(
        "<I4sBBHII", fid.read(DGM_HEADER_SIZE)
    )
    fid.seek(PARTITION_SIZE, 1)
    cmn_start = fid.tell()
    num_bytes_cmn = struct.unpack("<H", fid.read(2))[0]
    fid.seek(cmn_start + num_bytes_cmn)
    tx_start = fid.tell()
    num_bytes_tx_info, num_tx, num_bytes_per_tx, _pad, _heave = struct.unpack(
        "<3Hhf", fid.read(12)
    )
    fid.seek(tx_start + num_bytes_tx_info)
    fid.seek(num_tx * num_bytes_per_tx, 1)
    rx_start = fid.tell()
    rx = struct.unpack("<2H3Bb2f", fid.read(16))
    num_bytes_rx_info, num_beams, num_bytes_per_beam, phase_flag = rx[0], rx[1], rx[2], rx[3]
    fid.seek(rx_start + num_bytes_rx_info)
    phase_size = _PHASE_SIZE[phase_flag]
    for _ in range(num_beams):
        b_start = fid.tell()
        ns = struct.unpack("<f4H", fid.read(12))[4]
        fid.seek(b_start + num_bytes_per_beam)
        fid.seek(ns * (1 + phase_size), 1)
    consumed = fid.tell() - datagram_start
    return consumed + 4  # trailing numBytesDgm repeat


def _reconcile_all_mwc(path: Path) -> tuple[int, int]:
    """Return (matched, total) #MWC datagrams whose predicted size == declared."""
    file_size = path.stat().st_size
    matched = total = 0
    with path.open("rb") as fid:
        offset = 0
        while offset < file_size:
            fid.seek(offset)
            head = fid.read(8)
            if len(head) < 8:
                break
            dgm_size, dgm_type = struct.unpack("<I4s", head)
            if dgm_size < 8:
                break
            if dgm_type == b"#MWC":
                total += 1
                if _predict_mwc_total(fid, offset) == dgm_size:
                    matched += 1
            offset += dgm_size
    return matched, total


# Real EM124 abyssal water column (R/V Thompson, cruise TN447 — Samoa ship
# match). Clipped from EM124.Data/0180_20251215_064705.kmwcd to one #MWC via
# tests/fixtures/clip_datagrams.py --target-type '#MWC' -n 1. phaseFlag 0,
# dgm_version 2 (16-byte beam entry with the high-resolution detection field).
KMWCD_FIXTURE = FIXTURES / "sample_tn447_em124.kmwcd"

# Real EM2040 ASV water column carrying phaseFlag 1 (low-resolution int8 phase).
# Clipped from UNH-CCOM's 0006_20200917_015203_LowResPhase_subset.kmall to two
# #MWC. (.kmall and .kmwcd share the #MWC framing; the reader is extension
# agnostic.) dgm_version 0 (12-byte beam entry, no high-res field).
PHASE1_FIXTURE = FIXTURES / "sample_em2040_wc_phase1.kmall"


@pytest.mark.skipif(not KMWCD_FIXTURE.exists(), reason="kmwcd fixture not present")
def test_real_em124_kmwcd_phase0():
    """The real EM124 .kmwcd parses and reconciles to the byte; sane geometry."""
    matched, total = _reconcile_all_mwc(KMWCD_FIXTURE)
    assert total == 1
    assert matched == total  # exact byte reconciliation

    dgms = list(kmwcd.iter_mwc_datagrams(KMWCD_FIXTURE))
    assert len(dgms) == 1
    dgm = dgms[0]
    assert dgm.echo_sounder_id == 124
    assert dgm.dgm_version == 2
    assert dgm.phase_flag == 0
    assert dgm.num_tx_sectors == 8
    assert dgm.num_beams == 512
    assert len(dgm.beams) == 512
    # EM124 is a ~12 kHz system: every sector's centre frequency is near 12 kHz.
    assert all(10_000.0 < s.centre_freq_hz < 14_000.0 for s in dgm.tx_sectors)
    # Deep-water sound speed, plausible abyssal value.
    assert 1450.0 < dgm.sound_velocity_m_s < 1600.0
    # Amplitude sample arrays match the declared per-beam sample counts.
    for b in dgm.beams:
        assert len(b.sample_amplitudes_0p5_db) == b.num_sample_data
        assert b.phase_samples == ()  # phaseFlag 0 -> no phase
    # v2 high-resolution detection field is present (not None).
    assert dgm.beams[0].detected_range_samples_high_res is not None


@pytest.mark.skipif(not PHASE1_FIXTURE.exists(), reason="phase-1 fixture not present")
def test_real_em2040_phase_flag_1():
    """Real phaseFlag=1 #MWC: int8 phase, one phase sample per amplitude sample,
    values within the int8 ±180-deg (180/128 deg per unit) range."""
    matched, total = _reconcile_all_mwc(PHASE1_FIXTURE)
    assert total == 2
    assert matched == total

    dgms = list(kmwcd.iter_mwc_datagrams(PHASE1_FIXTURE))
    assert len(dgms) == 2
    dgm = dgms[0]
    assert dgm.echo_sounder_id == 2040
    assert dgm.phase_flag == 1
    assert dgm.num_beams == 256
    # EM2040 high-frequency sectors (here 190 / 220 kHz).
    assert all(s.centre_freq_hz > 100_000.0 for s in dgm.tx_sectors)
    all_phase = []
    for b in dgm.beams:
        assert len(b.sample_amplitudes_0p5_db) == b.num_sample_data
        assert len(b.phase_samples) == b.num_sample_data  # int8 phase present
        all_phase.extend(b.phase_samples)
    # int8 range; real data spans the full ±180-deg phase circle.
    assert min(all_phase) >= -128 and max(all_phase) <= 127
    assert min(all_phase) < -100 and max(all_phase) > 100
    # Consecutive pings.
    assert dgms[1].ping_cnt == dgms[0].ping_cnt + 1


# Real phaseFlag=2 (high-resolution int16 phase). The smallest such file (one
# #MWC ≈ 2.9 MB) is too large to commit, so this test is gated on the external
# file being present (set MBES_MWC_PHASE2_FILE, or drop UNH-CCOM's
# 0004_..._HiResPhase_subset.kmall at the path below). Validated live during
# D1: 4/4 #MWC reconcile; int16 phase spans ±18000 ≈ ±180 deg at 0.01 deg/unit.
_PHASE2_DEFAULT = Path(
    "/mnt/d/Cowork_OS/_WSL_Staging/projects/kmall-master/data/"
    "0004_20200917_014959_HiResPhase_subset.kmall"
)
_PHASE2_FILE = Path(os.environ.get("MBES_MWC_PHASE2_FILE", _PHASE2_DEFAULT))


@pytest.mark.skipif(not _PHASE2_FILE.exists(), reason="phase-2 (HiRes) file not present")
def test_real_phase_flag_2_int16():
    """Real phaseFlag=2 #MWC: int16 phase, full ±180-deg span at 0.01 deg/unit."""
    matched, total = _reconcile_all_mwc(_PHASE2_FILE)
    assert total >= 1
    assert matched == total

    dgm = next(kmwcd.iter_mwc_datagrams(_PHASE2_FILE))
    assert dgm.phase_flag == 2
    all_phase = []
    for b in dgm.beams:
        assert len(b.phase_samples) == b.num_sample_data
        all_phase.extend(b.phase_samples)
    # 0.01 deg/unit -> ±180 deg ~ ±18000; int16 capable, real data spans it.
    assert min(all_phase) < -15_000 and max(all_phase) > 15_000
    assert -32_768 <= min(all_phase) and max(all_phase) <= 32_767
