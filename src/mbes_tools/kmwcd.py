"""Kongsberg .kmwcd water column data reader.

The .kmwcd format is the water-column companion to .kmall. It uses the
SAME envelope and partition/cmnPart structures as .kmall, with #MWC
(Multibeam Water Column) datagrams instead of (or alongside) #MRZ.

Source: written from the public Kongsberg KMALL Datagram Format spec,
cross-referenced against UNH-CCOM's kmall library (Schmidt, Davis et al.)
#MWC parsing in
``Python_KMALL/kmall_viewer/external/kmall/KMALL/kmall.py``.

Status: validated against real data (Capability D1). The structure was
confirmed by exact byte reconciliation — predicting each datagram's size
from the struct sizes and the per-beam block formula
``numBytesPerBeamEntry + numSampleData*(1 + phaseSize)`` and matching the
file's declared ``numBytesDgm`` — on:

* TN447 EM124 ``.kmwcd`` (R/V Thompson, abyssal): 374/374 #MWC,
  ``phase_flag`` 0, ``dgm_version`` 2 (16-byte beam entry incl. the
  high-resolution detection field). Committed clip:
  ``tests/fixtures/sample_tn447_em124.kmwcd``.
* EM2040 ASV ``LowResPhase`` ``.kmall``: 40/40 #MWC, ``phase_flag`` 1
  (int8 phase, 180/128 deg) — real phase range ±128 ≈ ±180 deg. Committed
  clip: ``tests/fixtures/sample_em2040_wc_phase1.kmall``.
* EM2040 ASV ``HiResPhase`` ``.kmall``: 4/4 #MWC, ``phase_flag`` 2
  (int16 phase, 0.01 deg) — real phase range ±18000 ≈ ±180 deg. (Too
  large to commit; validated live + a gated test.)

Both the v1-style (12-byte) and v2-style (20-byte) beam headers and all
three phase flags are exercised. A wrong phase element size would not
reconcile to the datagram boundary, so the reconciliation pins the
``phase_flag`` 1/2 sample sizes against real bytes.

Typical usage::

    from pathlib import Path
    from mbes_tools.kmwcd import iter_mwc_datagrams

    for dgm in iter_mwc_datagrams(Path("survey.kmwcd")):
        for beam in dgm.beams:
            ...  # beam.sample_amplitudes_0p5_db is a tuple of int8 amplitude
                 # samples in 0.5 dB resolution per the Kongsberg spec
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from mbes_tools.kmall import (
    DGM_HEADER_FMT,
    DGM_HEADER_SIZE,
    MRZ_CMN_PART_FMT,
    MRZ_PARTITION_FMT,
    unpack_from_file,
)


# ---------------------------------------------------------------------------
# #MWC binary structure definitions.
# ---------------------------------------------------------------------------

# TX Info block (right after cmnPart):
# numBytesTxInfo (uint16), numTxSectors (uint16), numBytesPerTxSector (uint16),
# padding (int16), heave_m (float32).
MWC_TX_INFO_FMT = "<3Hhf"
MWC_TX_INFO_SIZE = struct.calcsize(MWC_TX_INFO_FMT)

# TX Sector data (per sector):
# tiltAngleReTx_deg (f32), centreFreq_Hz (f32), txBeamWidthAlong_deg (f32),
# txSectorNum (uint16), padding (int16).
MWC_TX_SECTOR_FMT = "<3fHh"
MWC_TX_SECTOR_SIZE = struct.calcsize(MWC_TX_SECTOR_FMT)

# RX Info block:
# numBytesRxInfo (uint16), numBeams (uint16), numBytesPerBeamEntry (uint8),
# phaseFlag (uint8), TVGfunctionApplied (uint8), TVGoffset_dB (int8),
# sampleFreq_Hz (f32), soundVelocity_mPerSec (f32).
MWC_RX_INFO_FMT = "<2H3Bb2f"
MWC_RX_INFO_SIZE = struct.calcsize(MWC_RX_INFO_FMT)

# RX Beam header.
# v1 layout (12 bytes): beamPointAngReVertical_deg (f32), startRangeSampleNum (u16),
# detectedRangeInSamples (u16), beamTxSectorNum (u16), numSampleData (u16).
# v2+ layout (16 bytes): same as v1 + detectedRangeInSamplesHighResolution (f32).
# Real files carry the true entry size in numBytesPerBeamEntry (RX Info);
# the parser always advances by that, so trailing unknown fields are tolerated.
MWC_RX_BEAM_V1_FMT = "<f4H"
MWC_RX_BEAM_V1_SIZE = struct.calcsize(MWC_RX_BEAM_V1_FMT)
MWC_RX_BEAM_V2_FMT = "<f4Hf"
MWC_RX_BEAM_V2_SIZE = struct.calcsize(MWC_RX_BEAM_V2_FMT)


# ---------------------------------------------------------------------------
# Data classes.
# ---------------------------------------------------------------------------


@dataclass
class MWCTxSector:
    """One transmit sector in a #MWC datagram."""

    tilt_angle_re_tx_deg: float
    centre_freq_hz: float
    tx_beam_width_along_deg: float
    tx_sector_num: int


@dataclass
class MWCRxBeam:
    """One receive beam in a #MWC datagram.

    ``sample_amplitudes_0p5_db`` is the int8 amplitude array in 0.5 dB
    resolution per the Kongsberg spec — multiply by 0.5 for dB values.

    ``phase_samples`` is populated only when the parent datagram's
    ``phase_flag`` is 1 (int8 phase, 180/128 deg per unit) or 2
    (int16 phase, 0.01 deg per unit). Both are validated against real data:
    the int8 range spans ±128 and the int16 range ±18000, i.e. ±180 deg.
    """

    beam_point_angle_re_vertical_deg: float
    start_range_sample_num: int
    detected_range_samples: int
    tx_sector_num: int
    num_sample_data: int
    detected_range_samples_high_res: Optional[float] = None  # v2+ only
    sample_amplitudes_0p5_db: Tuple[int, ...] = field(default_factory=tuple)
    phase_samples: Tuple[int, ...] = field(default_factory=tuple)


@dataclass
class MWCDatagram:
    """Parsed #MWC Multibeam Water Column datagram."""

    # Envelope
    dgm_version: int
    system_id: int
    echo_sounder_id: int
    time_s: int
    time_ns: int

    # cmnPart
    ping_cnt: int
    rx_fans_per_ping: int
    rx_fan_index: int
    swaths_per_ping: int

    # TX Info
    num_tx_sectors: int
    heave_m: float
    tx_sectors: List[MWCTxSector] = field(default_factory=list)

    # RX Info
    num_beams: int = 0
    phase_flag: int = 0
    tvg_function_applied: int = 0
    tvg_offset_db: int = 0
    sample_freq_hz: float = 0.0
    sound_velocity_m_s: float = 0.0
    beams: List[MWCRxBeam] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core #MWC parsing.
# ---------------------------------------------------------------------------


def parse_mwc_datagram(
    fid,
    datagram_start: int,
    datagram_size: int,
) -> Optional[MWCDatagram]:
    """Parse one #MWC datagram and return an :class:`MWCDatagram`.

    The file handle ``fid`` is left positioned at
    ``datagram_start + datagram_size`` on exit.

    Returns ``None`` if the datagram type is not ``#MWC``.
    """
    fid.seek(datagram_start)

    header = unpack_from_file(fid, DGM_HEADER_FMT)
    num_bytes_dgm, dgm_type, dgm_version, system_id, echo_sounder_id, time_s, time_ns = header

    if dgm_type != b"#MWC":
        fid.seek(datagram_start + datagram_size)
        return None

    # Partition block (4 bytes, unused).
    unpack_from_file(fid, MRZ_PARTITION_FMT)

    # cmnPart (variable; first uint16 is its own size).
    cmn_start = fid.tell()
    cmn = unpack_from_file(fid, MRZ_CMN_PART_FMT)
    (
        num_bytes_cmn_part,
        ping_cnt,
        rx_fans_per_ping,
        rx_fan_index,
        swaths_per_ping,
        _swath_along_position,
        _tx_transducer_ind,
        _rx_transducer_ind,
        _num_rx_transducers,
        _algorithm_type,
    ) = cmn
    fid.seek(cmn_start + num_bytes_cmn_part)

    # TX Info.
    tx_info_start = fid.tell()
    tx_info = unpack_from_file(fid, MWC_TX_INFO_FMT)
    num_bytes_tx_info, num_tx_sectors, num_bytes_per_tx_sector, _padding, heave_m = tx_info
    # Skip any trailing unknown fields in TX Info.
    fid.seek(tx_info_start + num_bytes_tx_info)

    # TX Sectors.
    tx_sectors: List[MWCTxSector] = []
    for _ in range(num_tx_sectors):
        sec_start = fid.tell()
        sec = unpack_from_file(fid, MWC_TX_SECTOR_FMT)
        tilt, freq, width, sec_num, _pad = sec
        tx_sectors.append(
            MWCTxSector(
                tilt_angle_re_tx_deg=float(tilt),
                centre_freq_hz=float(freq),
                tx_beam_width_along_deg=float(width),
                tx_sector_num=int(sec_num),
            )
        )
        # Advance to declared end of this TX sector record.
        fid.seek(sec_start + num_bytes_per_tx_sector)

    # RX Info.
    rx_info_start = fid.tell()
    rx_info = unpack_from_file(fid, MWC_RX_INFO_FMT)
    (
        num_bytes_rx_info,
        num_beams,
        num_bytes_per_beam_entry,
        phase_flag,
        tvg_func,
        tvg_off,
        sample_freq,
        sound_vel,
    ) = rx_info
    fid.seek(rx_info_start + num_bytes_rx_info)

    # RX Beams.
    beams: List[MWCRxBeam] = []
    use_v2 = dgm_version >= 2 and num_bytes_per_beam_entry >= MWC_RX_BEAM_V2_SIZE
    for _ in range(num_beams):
        beam_start = fid.tell()

        if use_v2:
            b = unpack_from_file(fid, MWC_RX_BEAM_V2_FMT)
            angle, start_samp, det_range, sec_num, n_samples, det_range_hires = b
            high_res: Optional[float] = float(det_range_hires)
        else:
            b = unpack_from_file(fid, MWC_RX_BEAM_V1_FMT)
            angle, start_samp, det_range, sec_num, n_samples = b
            high_res = None

        # Advance to declared end of beam entry (skip any unknown trailing fields).
        fid.seek(beam_start + num_bytes_per_beam_entry)

        # Amplitude samples (int8, 0.5 dB resolution).
        samples: Tuple[int, ...] = ()
        if n_samples > 0:
            samples = struct.unpack(f"<{n_samples}b", fid.read(n_samples))

        # Optional phase samples.
        phase: Tuple[int, ...] = ()
        if n_samples > 0:
            if phase_flag == 1:
                phase = struct.unpack(f"<{n_samples}b", fid.read(n_samples))
            elif phase_flag == 2:
                phase = struct.unpack(f"<{n_samples}h", fid.read(2 * n_samples))

        beams.append(
            MWCRxBeam(
                beam_point_angle_re_vertical_deg=float(angle),
                start_range_sample_num=int(start_samp),
                detected_range_samples=int(det_range),
                tx_sector_num=int(sec_num),
                num_sample_data=int(n_samples),
                detected_range_samples_high_res=high_res,
                sample_amplitudes_0p5_db=samples,
                phase_samples=phase,
            )
        )

    # Position at end of datagram for next iteration.
    fid.seek(datagram_start + datagram_size)

    return MWCDatagram(
        dgm_version=int(dgm_version),
        system_id=int(system_id),
        echo_sounder_id=int(echo_sounder_id),
        time_s=int(time_s),
        time_ns=int(time_ns),
        ping_cnt=int(ping_cnt),
        rx_fans_per_ping=int(rx_fans_per_ping),
        rx_fan_index=int(rx_fan_index),
        swaths_per_ping=int(swaths_per_ping),
        num_tx_sectors=int(num_tx_sectors),
        heave_m=float(heave_m),
        tx_sectors=tx_sectors,
        num_beams=int(num_beams),
        phase_flag=int(phase_flag),
        tvg_function_applied=int(tvg_func),
        tvg_offset_db=int(tvg_off),
        sample_freq_hz=float(sample_freq),
        sound_velocity_m_s=float(sound_vel),
        beams=beams,
    )


def iter_mwc_datagrams(path: Path) -> Iterator[MWCDatagram]:
    """Walk a .kmwcd (or .kmall) file and yield each #MWC datagram.

    Non-#MWC datagrams (#SKM, #SPO, #IIP, even #MRZ if present alongside
    water column data) are skipped by declared byte length.
    """
    file_size = path.stat().st_size
    with path.open("rb") as fid:
        offset = 0
        while offset < file_size:
            fid.seek(offset)
            header8 = fid.read(8)
            if len(header8) < 8:
                break

            datagram_size, datagram_type = struct.unpack("<I4s", header8)

            if datagram_size < 8:
                raise RuntimeError(
                    f"Invalid datagram length {datagram_size} at byte offset {offset} in {path}"
                )

            if datagram_type == b"#MWC":
                dgm = parse_mwc_datagram(
                    fid,
                    datagram_start=offset,
                    datagram_size=datagram_size,
                )
                if dgm is not None:
                    yield dgm

            offset += datagram_size


def iter_kmwcd_files(path: Path, recursive: bool = False) -> List[Path]:
    """Return ``.kmwcd`` files from a file or directory."""
    if path.is_file():
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    files = path.rglob("*.kmwcd") if recursive else path.glob("*.kmwcd")
    return sorted(set(files))
