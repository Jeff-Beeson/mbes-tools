"""Kongsberg .all reader: native binary parser.

This module reads Kongsberg .all datagrams directly using ``struct``,
same approach as :mod:`mbes_tools.kmall`. No third-party dependency.

Source: written from the public Kongsberg EM Series Data Format
specification (October 2013). Cross-referenced against
``guardiangeomatics/pyall`` (Apache-2.0, Paul Kennedy) on 2026-05-25 for
edge cases in the per-beam record layouts.

Status: v0. The datagram envelope plus the most-used datagram types are
supported:

- ``P`` position
- ``X`` depth (XYZ datagram)
- ``Y`` seabed image (per-beam backscatter samples)

Other datagram types in a .all file are skipped by their declared byte
length (graceful skip; the iterator does not raise on unsupported types).
Add more parsers (``A`` attitude, ``R`` runtime, ``I`` installation,
``N`` raw range, etc.) as projects need them.

Typical usage::

    from pathlib import Path
    from mbes_tools.all import iter_seabed_image_datagrams

    for dgm in iter_seabed_image_datagrams(Path("survey.all")):
        for beam in dgm.beams:
            ...  # beam.samples is a tuple of int16 amplitude samples
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from mbes_tools.install_params import InstallationParameters


# ---------------------------------------------------------------------------
# Datagram type codes (single ASCII characters).
# ---------------------------------------------------------------------------

DGM_TYPE_POSITION = b"P"
DGM_TYPE_DEPTH = b"X"
DGM_TYPE_SEABED_IMAGE = b"Y"
DGM_TYPE_RAW_RANGE_ANGLE = b"N"  # datagram 78: raw range and angle
DGM_TYPE_RUNTIME = b"R"          # datagram 82: runtime parameters
DGM_TYPE_ATTITUDE = b"A"         # datagram 65: attitude
DGM_TYPE_INSTALLATION = b"I"     # datagram 73: installation parameters (start)

# Constants from the Kongsberg envelope.
STX = 0x02
ETX = 0x03


# ---------------------------------------------------------------------------
# Binary structure definitions.
# ---------------------------------------------------------------------------

# Datagram envelope header: numberOfBytes, STX, typeOfDatagram, emModel, date, time.
# numberOfBytes excludes itself, so total datagram size = numberOfBytes + 4.
DGM_HEADER_FMT = "<LBBHLL"
DGM_HEADER_SIZE = struct.calcsize(DGM_HEADER_FMT)

# Datagram footer: ETX (uint8) + checksum (uint16). Sometimes preceded by a
# spare alignment byte; the per-datagram parsers handle that locally.
DGM_FOOTER_FMT = "<BH"
DGM_FOOTER_SIZE = struct.calcsize(DGM_FOOTER_FMT)

# P position datagram body (after envelope header, before footer):
# counter, serialNumber, lat*2e7, lon*1e7, quality*100, sog*100, cog*100,
# heading*100, descriptor, NBytesInInputDatagram.
P_BODY_FMT = "<HHll4HBB"
P_BODY_SIZE = struct.calcsize(P_BODY_FMT)

# X depth datagram body (after envelope header, before per-beam loop):
# counter, serialNumber, heading*100, soundSpeed*10, transducerDepth (f32),
# nBeams, nValidDetections, sampleFreq (f32), scanningInfo, spare1, spare2, spare3.
X_BODY_FMT = "<4Hf2Hf4B"
X_BODY_SIZE = struct.calcsize(X_BODY_FMT)

# X per-beam record:
# depth (f32), acrossTrack (f32), alongTrack (f32), detectionWindowLen (u16),
# qualityFactor (u8), beamIncidenceAngleAdjustment*10 (u8), detectionInfo (u8),
# realtimeCleaningInfo (i8), reflectivity*10 (i16).
X_BEAM_FMT = "<fffHBBBbh"
X_BEAM_SIZE = struct.calcsize(X_BEAM_FMT)

# Y seabed image datagram body (after envelope header, before per-beam loop):
# counter, serialNumber, sampleFreq (f32), rangeToNormalIncidence (u16),
# normalIncidence (i16), oblique (i16), txBeamWidth*10 (u16),
# tvgCrossover*10 (u16), numBeams (u16).
Y_BODY_FMT = "<HHfHhhHHH"
Y_BODY_SIZE = struct.calcsize(Y_BODY_FMT)

# Y per-beam record (header before the variable-length sample array):
# sortingDirection (i8), detectionInfo (u8), numberOfSamplesPerBeam (u16),
# centreSampleNumber (u16).
Y_BEAM_FMT = "<bBHH"
Y_BEAM_SIZE = struct.calcsize(Y_BEAM_FMT)

# N raw range and angle datagram (78). Body after the envelope header, before
# the per-sector / per-beam loops:
# pingCounter, serialNumber, soundSpeed*10 (0.1 m/s), numTxSectors, numRxBeams,
# numValidDetections, sampleFrequency (f32), Dscale (u32).
N_BODY_FMT = "<HHHHHHfL"
N_BODY_SIZE = struct.calcsize(N_BODY_FMT)

# N per-transmit-sector record (parsed only to advance past it):
# tiltAngle*100 (i16), focusRange (u16), signalLength (f32), sectorTransmitDelay (f32),
# centreFrequency (f32), meanAbsorption (u16), signalWaveformId (u8),
# transmitSectorNumber (u8), signalBandwidth (f32).
N_TX_FMT = "<hHfffHBBf"
N_TX_SIZE = struct.calcsize(N_TX_FMT)

# N per-receive-beam record:
# beamPointingAngle*100 (i16), transmitSectorNumber (u8), detectionInfo (u8),
# detectionWindowLength (u16), qualityFactor (u8), Dcorr (i8),
# twoWayTravelTime (f32), reflectivity*10 (i16), realtimeCleaningInfo (i8), spare (u8).
N_RX_FMT = "<hBBHBbfhbB"
N_RX_SIZE = struct.calcsize(N_RX_FMT)

# R runtime parameters datagram (82). Body after the envelope header, before the
# 3-byte footer. Only a few fields are decoded by name; the rest are kept to
# advance the cursor and round-trip the record.
# pingCounter, serialNumber, operatorStationStatus, processingUnitStatus,
# bspStatus, sonarHeadStatus, mode, filterIdentifier, minDepth (u16), maxDepth (u16),
# absorptionCoeff*100 (u16), txPulseLength (u16), txBeamWidth*10 (u16),
# txPower (i8), rxBeamWidth*10 (u8), rxBandwidth*50Hz (u8), mode2 (u8), tvg (u8),
# sourceOfSoundSpeed (u8), maxPortSwathWidth (u16), beamSpacing (u8),
# maxPortCoverageDeg (u8), yawPitchStabMode (u8), maxStbdCoverageDeg (u8),
# maxStbdSwathWidth (u16), txAlongTilt*10 (i16), filterIdentifier2 (u8).
R_BODY_FMT = "<HHBBBBBBHHHHHbBBBBBHBBBBHhB"
R_BODY_SIZE = struct.calcsize(R_BODY_FMT)

# A attitude datagram (65). Body: counter (u16), serialNumber (u16),
# numEntries (u16), then numEntries attitude records, then a trailing
# sensorSystemDescriptor (u8) before the footer.
A_BODY_FMT = "<HHH"
A_BODY_SIZE = struct.calcsize(A_BODY_FMT)

# A per-entry attitude record:
# recordTime (u16, ms since datagram time), sensorStatus (u16),
# roll*100 (i16), pitch*100 (i16), heave (i16, cm), heading*100 (u16).
A_ENTRY_FMT = "<HHhhhH"
A_ENTRY_SIZE = struct.calcsize(A_ENTRY_FMT)

# I installation datagram (73). Body: surveyLineNumber/counter (u16),
# systemSerialNumber (u16), secondarySystemSerialNumber (u16), then a delimited
# ASCII installation-parameter string (KEY=VALUE,...) up to the footer.
I_BODY_FMT = "<HHH"
I_BODY_SIZE = struct.calcsize(I_BODY_FMT)


# ---------------------------------------------------------------------------
# Data classes for the parsed records.
# ---------------------------------------------------------------------------


@dataclass
class DatagramHeader:
    """The 12-byte envelope header that prefixes every Kongsberg .all datagram."""

    number_of_bytes: int  # excludes itself; total datagram size on disk = this + 4
    stx: int              # always 0x02
    type_of_datagram: str # single ASCII character, e.g. 'P', 'X', 'Y'
    em_model: int         # Kongsberg model number, e.g. 302, 710, 2040
    record_date: int      # YYYYMMDD as integer
    record_time_ms: int   # time of day in milliseconds since midnight

    @property
    def time_seconds(self) -> float:
        """Time of day in seconds since midnight."""
        return self.record_time_ms / 1000.0


@dataclass
class DatagramRecord:
    """A raw datagram from a .all file: envelope plus the body bytes.

    Use this when iterating with :func:`iter_datagrams` and you want to
    handle datagram types yourself or skip ones with no parser yet.
    """

    header: DatagramHeader
    offset: int           # byte offset of the start of the datagram in the file
    body: bytes           # raw bytes between the envelope header and the footer


@dataclass
class PositionDatagram:
    """Parsed ``P`` position datagram."""

    header: DatagramHeader
    counter: int
    serial_number: int
    latitude_deg: float
    longitude_deg: float
    position_fix_quality_m: float
    speed_over_ground_m_s: float
    course_over_ground_deg: float
    heading_deg: float
    descriptor: int
    input_datagram: bytes  # raw bytes of the embedded position-system message (e.g. NMEA)


@dataclass
class XBeam:
    """One beam in an ``X`` depth datagram."""

    depth_m: float
    across_track_m: float
    along_track_m: float
    detection_window_length: int
    quality_factor: int
    beam_incidence_angle_adjustment_deg: float
    detection_information: int
    realtime_cleaning_information: int
    reflectivity_db: float

    @property
    def is_valid(self) -> bool:
        """True for beams with a real bottom detection.

        In the XYZ88 datagram, bit 7 of ``detectionInfo`` flags a beam with no
        valid detection (set = invalid). Real-time cleaning flags the beam by
        making ``realtimeCleaningInformation`` negative. A zero depth is also
        treated as no detection.
        """
        return (
            (self.detection_information & 0x80) == 0
            and self.realtime_cleaning_information >= 0
            and self.depth_m != 0.0
        )


@dataclass
class DepthDatagram:
    """Parsed ``X`` depth (XYZ) datagram. The XYZ datagram is the modern
    sounding record for EM 2040 / EM 710 / EM 302 / EM 304 systems."""

    header: DatagramHeader
    counter: int
    serial_number: int
    heading_deg: float
    sound_speed_at_transducer_m_s: float
    transducer_depth_m: float
    num_beams: int
    num_valid_detections: int
    sample_frequency_hz: float
    scanning_info: int
    beams: List[XBeam] = field(default_factory=list)


@dataclass
class YBeam:
    """One beam in a ``Y`` seabed image datagram. ``samples`` is the per-beam
    int16 amplitude array (units 0.1 dB)."""

    sorting_direction: int
    detection_info: int
    number_of_samples_per_beam: int
    centre_sample_number: int
    samples: Tuple[int, ...] = field(default_factory=tuple)


@dataclass
class SeabedImageDatagram:
    """Parsed ``Y`` seabed image datagram. Carries per-beam backscatter
    amplitude samples; this is the datagram backscatter analysis pulls from."""

    header: DatagramHeader
    counter: int
    serial_number: int
    sample_frequency_hz: float
    range_to_normal_incidence_samples: int
    normal_incidence_bs_db: int
    oblique_bs_db: int
    tx_beam_width_deg: float
    tvg_crossover_deg: float
    num_beams: int
    beams: List[YBeam] = field(default_factory=list)


@dataclass
class RawRangeAngleBeam:
    """One receive beam in an ``N`` raw range and angle (78) datagram.

    ``beam_pointing_angle_deg`` is the beam angle re the RX array — the same
    quantity as the .kmall ``beam_angle_re_rx_deg``. ``tx_sector_number`` is the
    zero-based transmit sector, used to key sector backscatter corrections.
    """

    beam_pointing_angle_deg: float
    tx_sector_number: int
    detection_info: int
    detection_window_length: int
    quality_factor: int
    d_corr: int
    two_way_travel_time_s: float
    reflectivity_db: float
    realtime_cleaning_information: int

    @property
    def is_valid(self) -> bool:
        """True for beams with a real detection (bit 7 of detectionInfo clear)."""
        return (self.detection_info & 0x80) == 0 and self.realtime_cleaning_information >= 0


@dataclass
class RawRangeAngleDatagram:
    """Parsed ``N`` raw range and angle (78) datagram.

    Carries the per-beam pointing angle and transmit-sector number that the XYZ
    (``X``) datagram does not, so the two are joined by ping counter for
    backscatter analysis (angle/sector from ``N``, depth/reflectivity from ``X``).
    """

    header: DatagramHeader
    ping_counter: int
    serial_number: int
    sound_speed_m_s: float
    num_tx_sectors: int
    num_rx_beams: int
    num_valid_detections: int
    sample_frequency_hz: float
    beams: List[RawRangeAngleBeam] = field(default_factory=list)


@dataclass
class AttitudeSample:
    """One attitude record in an ``A`` attitude (65) datagram.

    ``time_ms`` is milliseconds since the datagram header time. Angles are in
    degrees (positive roll = port up; positive pitch = bow up; heading is true,
    0–360). ``heave_m`` is positive down.
    """

    time_ms: int
    sensor_status: int
    roll_deg: float
    pitch_deg: float
    heave_m: float
    heading_deg: float


@dataclass
class AttitudeDatagram:
    """Parsed ``A`` attitude (65) datagram — a short time series of
    roll/pitch/heave/heading samples between sounding datagrams."""

    header: DatagramHeader
    counter: int
    serial_number: int
    sensor_system_descriptor: int
    samples: List[AttitudeSample] = field(default_factory=list)


@dataclass
class InstallationDatagram:
    """Parsed ``I`` installation parameters (73) datagram.

    ``parameters`` is a structured view of the install text (transducer lever
    arms, mount angles, waterline, serials) — see
    :class:`mbes_tools.install_params.InstallationParameters`.
    """

    header: DatagramHeader
    counter: int
    serial_number: int
    secondary_serial_number: int
    parameters: InstallationParameters


@dataclass
class RuntimeDatagram:
    """Parsed ``R`` runtime parameters (82) datagram.

    The ``mode`` byte encodes the depth/ping mode; its meaning is model-specific
    (for EM2040/EM2045 the low bits select frequency rather than depth). Decode
    it with :mod:`mbes_tools.depth_modes`. Runtime datagrams are emitted only
    when settings change, so a reader tracks the most recent one as ping state.
    """

    header: DatagramHeader
    ping_counter: int
    serial_number: int
    mode: int
    filter_identifier: int
    minimum_depth_m: int
    maximum_depth_m: int
    absorption_coefficient_db_km: float
    transmit_pulse_length_us: int
    yaw_pitch_stabilization_mode: int


# ---------------------------------------------------------------------------
# Low-level read helpers.
# ---------------------------------------------------------------------------


def _unpack(fmt: str, buf: bytes, offset: int = 0):
    """``struct.unpack_from`` shorthand."""
    return struct.unpack_from(fmt, buf, offset)


def _read_header(fid) -> Optional[DatagramHeader]:
    """Read a 12-byte envelope from the current file position.

    Returns ``None`` at clean EOF.
    """
    raw = fid.read(DGM_HEADER_SIZE)
    if len(raw) < DGM_HEADER_SIZE:
        return None

    nbytes, stx, type_code, em_model, date, time_ms = struct.unpack(DGM_HEADER_FMT, raw)
    return DatagramHeader(
        number_of_bytes=nbytes,
        stx=stx,
        type_of_datagram=chr(type_code),
        em_model=em_model,
        record_date=date,
        record_time_ms=time_ms,
    )


# ---------------------------------------------------------------------------
# File discovery.
# ---------------------------------------------------------------------------


def iter_all_files(path: Path, recursive: bool = False) -> List[Path]:
    """Return ``.all`` files from a file or directory.

    Args:
        path: A single file or a directory.
        recursive: If True, search subdirectories.

    Returns:
        Sorted, de-duplicated list of paths.
    """
    if path.is_file():
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    files = path.rglob("*.all") if recursive else path.glob("*.all")
    return sorted(set(files))


# ---------------------------------------------------------------------------
# Core datagram iterator.
# ---------------------------------------------------------------------------


_RESYNC_SCAN_LIMIT = 8 * 1024 * 1024  # bound the forward scan when resyncing


def _looks_like_all_header(raw16: bytes, header_offset: int, file_size: int) -> bool:
    """Heuristic: do these 16 bytes start a plausible .all datagram?"""
    if len(raw16) < DGM_HEADER_SIZE:
        return False
    nbytes, stx, type_code, _em, _date, _time = struct.unpack(DGM_HEADER_FMT, raw16)
    if stx != STX:
        return False
    if not (0x20 <= type_code < 0x7F):  # printable ASCII type code
        return False
    total = nbytes + 4
    if nbytes < DGM_HEADER_SIZE - 4 or header_offset + total > file_size:
        return False
    return True


def _resync_all(fid, file_size: int, from_offset: int) -> Optional[int]:
    """Scan forward for the next plausible .all datagram start; None if none.

    A .all envelope has STX (0x02) as its 5th byte, so candidate header starts
    are 4 bytes before each 0x02. Each candidate is validated by
    :func:`_looks_like_all_header`. The scan is bounded by ``_RESYNC_SCAN_LIMIT``.
    """
    start = from_offset + 1
    end = min(file_size, start + _RESYNC_SCAN_LIMIT)
    if start >= file_size:
        return None
    fid.seek(start)
    window = fid.read(end - start)
    search = 0
    while True:
        idx = window.find(b"\x02", search)
        if idx < 0:
            return None
        candidate = start + idx - 4
        if candidate >= start and candidate + DGM_HEADER_SIZE <= file_size:
            fid.seek(candidate)
            if _looks_like_all_header(fid.read(DGM_HEADER_SIZE), candidate, file_size):
                return candidate
        search = idx + 1


def iter_datagrams(
    path: Path,
    types: Optional[set] = None,
    on_error: str = "raise",
    error_log: Optional[list] = None,
) -> Iterator[DatagramRecord]:
    """Walk a .all file and yield each datagram as a raw ``DatagramRecord``.

    The body bytes are the raw datagram contents between the envelope
    header and the trailing footer, exclusive. Use the type-specific
    parser functions (:func:`parse_position`, :func:`parse_depth`,
    :func:`parse_seabed_image`) to decode the body.

    Unrecognized datagram types are still yielded — callers can filter on
    ``record.header.type_of_datagram`` and skip what they don't care about.

    Args:
        types: Optional set of single-character datagram type codes
            (e.g. ``{"X", "N", "R", "P"}``). When given, only those datagrams
            are read and yielded; the bodies of all others are seeked past
            without reading, which avoids paying to read large bodies
            (e.g. ``Y`` seabed image) the caller does not need.
        on_error: ``"raise"`` (default) raises on a corrupt/truncated datagram.
            ``"skip"`` instead resynchronizes to the next plausible datagram and
            continues, so one bad region never aborts a survey.
        error_log: Optional list; when given, ``(offset, message)`` tuples for
            skipped problems are appended to it.
    """
    file_size = path.stat().st_size
    with path.open("rb") as fid:
        offset = 0
        while offset < file_size:
            fid.seek(offset)
            header = _read_header(fid)
            if header is None:
                break

            total_size = header.number_of_bytes + 4
            problem: Optional[str] = None
            if header.number_of_bytes < DGM_HEADER_SIZE - 4 or offset + total_size > file_size:
                problem = f"invalid datagram length {header.number_of_bytes}"
            elif total_size - DGM_HEADER_SIZE - DGM_FOOTER_SIZE < 0:
                problem = f"negative body size (total={total_size})"

            if problem is not None:
                msg = f"{problem} at byte offset {offset} in {path}"
                if on_error != "skip":
                    raise RuntimeError(msg)
                if error_log is not None:
                    error_log.append((offset, msg))
                nxt = _resync_all(fid, file_size, offset)
                if nxt is None or nxt <= offset:
                    break
                offset = nxt
                continue

            body_size = total_size - DGM_HEADER_SIZE - DGM_FOOTER_SIZE

            if types is not None and header.type_of_datagram not in types:
                offset += total_size
                continue

            fid.seek(offset + DGM_HEADER_SIZE)
            body = fid.read(body_size)
            if len(body) != body_size:
                msg = (
                    f"truncated datagram at offset {offset} in {path}: "
                    f"expected {body_size} body bytes, got {len(body)}"
                )
                if on_error != "skip":
                    raise EOFError(msg)
                if error_log is not None:
                    error_log.append((offset, msg))
                break  # truncation at EOF: nothing valid remains

            yield DatagramRecord(header=header, offset=offset, body=body)

            offset += total_size


# ---------------------------------------------------------------------------
# Type-specific parsers.
# ---------------------------------------------------------------------------


def parse_position(record: DatagramRecord) -> PositionDatagram:
    """Decode the body of a ``P`` position datagram."""
    if record.header.type_of_datagram != "P":
        raise ValueError(
            f"parse_position expects type 'P', got {record.header.type_of_datagram!r}"
        )

    s = _unpack(P_BODY_FMT, record.body, 0)
    (
        counter,
        serial_number,
        latitude_raw,
        longitude_raw,
        quality_raw,
        sog_raw,
        cog_raw,
        heading_raw,
        descriptor,
        nbytes_input,
    ) = s

    # The embedded position-system datagram follows the fixed body. The spec
    # adds a spare alignment byte if the running byte count is odd.
    input_dg = record.body[P_BODY_SIZE : P_BODY_SIZE + nbytes_input]

    return PositionDatagram(
        header=record.header,
        counter=counter,
        serial_number=serial_number,
        latitude_deg=latitude_raw / 20_000_000.0,
        longitude_deg=longitude_raw / 10_000_000.0,
        position_fix_quality_m=quality_raw / 100.0,
        speed_over_ground_m_s=sog_raw / 100.0,
        course_over_ground_deg=cog_raw / 100.0,
        heading_deg=heading_raw / 100.0,
        descriptor=descriptor,
        input_datagram=input_dg,
    )


def parse_depth(record: DatagramRecord) -> DepthDatagram:
    """Decode the body of an ``X`` depth (XYZ) datagram."""
    if record.header.type_of_datagram != "X":
        raise ValueError(
            f"parse_depth expects type 'X', got {record.header.type_of_datagram!r}"
        )

    s = _unpack(X_BODY_FMT, record.body, 0)
    (
        counter,
        serial_number,
        heading_raw,
        sound_speed_raw,
        transducer_depth_m,
        n_beams,
        n_valid_detections,
        sample_freq_hz,
        scanning_info,
        _spare1,
        _spare2,
        _spare3,
    ) = s

    beams: List[XBeam] = []
    cursor = X_BODY_SIZE
    for _ in range(n_beams):
        b = _unpack(X_BEAM_FMT, record.body, cursor)
        (
            depth,
            across,
            along,
            detect_win,
            quality,
            angle_adj_raw,
            det_info,
            rt_clean,
            refl_raw,
        ) = b
        beams.append(
            XBeam(
                depth_m=float(depth),
                across_track_m=float(across),
                along_track_m=float(along),
                detection_window_length=int(detect_win),
                quality_factor=int(quality),
                beam_incidence_angle_adjustment_deg=angle_adj_raw / 10.0,
                detection_information=int(det_info),
                realtime_cleaning_information=int(rt_clean),
                reflectivity_db=refl_raw / 10.0,
            )
        )
        cursor += X_BEAM_SIZE

    return DepthDatagram(
        header=record.header,
        counter=counter,
        serial_number=serial_number,
        heading_deg=heading_raw / 100.0,
        sound_speed_at_transducer_m_s=sound_speed_raw / 10.0,
        transducer_depth_m=float(transducer_depth_m),
        num_beams=n_beams,
        num_valid_detections=n_valid_detections,
        sample_frequency_hz=float(sample_freq_hz),
        scanning_info=int(scanning_info),
        beams=beams,
    )


def parse_seabed_image(record: DatagramRecord) -> SeabedImageDatagram:
    """Decode the body of a ``Y`` seabed image datagram.

    The Y datagram is the primary source of per-beam backscatter amplitude
    samples for older Kongsberg systems (pre-kmall). Sample values are int16
    in units of 0.1 dB.
    """
    if record.header.type_of_datagram != "Y":
        raise ValueError(
            f"parse_seabed_image expects type 'Y', got {record.header.type_of_datagram!r}"
        )

    s = _unpack(Y_BODY_FMT, record.body, 0)
    (
        counter,
        serial_number,
        sample_freq,
        range_to_normal,
        normal_bs,
        oblique_bs,
        tx_beam_width_raw,
        tvg_cross_raw,
        num_beams,
    ) = s

    beams: List[YBeam] = []
    cursor = Y_BODY_SIZE
    for _ in range(num_beams):
        b = _unpack(Y_BEAM_FMT, record.body, cursor)
        sort_dir, det_info, n_samples, centre_sample = b
        beams.append(
            YBeam(
                sorting_direction=int(sort_dir),
                detection_info=int(det_info),
                number_of_samples_per_beam=int(n_samples),
                centre_sample_number=int(centre_sample),
            )
        )
        cursor += Y_BEAM_SIZE

    # All sample arrays are packed together after the per-beam headers.
    total_samples = sum(b.number_of_samples_per_beam for b in beams)
    if total_samples > 0:
        sample_fmt = f"<{total_samples}h"
        sample_size = struct.calcsize(sample_fmt)
        all_samples = struct.unpack_from(sample_fmt, record.body, cursor)

        # Hand each beam its slice.
        idx = 0
        for b in beams:
            n = b.number_of_samples_per_beam
            b.samples = tuple(all_samples[idx : idx + n])
            idx += n

    return SeabedImageDatagram(
        header=record.header,
        counter=counter,
        serial_number=serial_number,
        sample_frequency_hz=float(sample_freq),
        range_to_normal_incidence_samples=int(range_to_normal),
        normal_incidence_bs_db=int(normal_bs),
        oblique_bs_db=int(oblique_bs),
        tx_beam_width_deg=tx_beam_width_raw / 10.0,
        tvg_crossover_deg=tvg_cross_raw / 10.0,
        num_beams=num_beams,
        beams=beams,
    )


def parse_raw_range_angle(record: DatagramRecord) -> RawRangeAngleDatagram:
    """Decode the body of an ``N`` raw range and angle (78) datagram.

    The N datagram supplies per-beam pointing angle and transmit-sector number.
    The transmit-sector blocks between the fixed header and the receive-beam
    array are skipped (their contents are not needed for backscatter binning).
    """
    if record.header.type_of_datagram != "N":
        raise ValueError(
            f"parse_raw_range_angle expects type 'N', got {record.header.type_of_datagram!r}"
        )

    s = _unpack(N_BODY_FMT, record.body, 0)
    (
        ping_counter,
        serial_number,
        sound_speed_raw,
        num_tx_sectors,
        num_rx_beams,
        num_valid_detections,
        sample_freq_hz,
        _dscale,
    ) = s

    # Skip the transmit-sector records; advance to the receive-beam array.
    cursor = N_BODY_SIZE + num_tx_sectors * N_TX_SIZE

    beams: List[RawRangeAngleBeam] = []
    for _ in range(num_rx_beams):
        b = _unpack(N_RX_FMT, record.body, cursor)
        (
            angle_raw,
            tx_sector,
            det_info,
            det_win,
            quality,
            d_corr,
            twtt,
            refl_raw,
            rt_clean,
            _spare,
        ) = b
        beams.append(
            RawRangeAngleBeam(
                beam_pointing_angle_deg=angle_raw / 100.0,
                tx_sector_number=int(tx_sector),
                detection_info=int(det_info),
                detection_window_length=int(det_win),
                quality_factor=int(quality),
                d_corr=int(d_corr),
                two_way_travel_time_s=float(twtt),
                reflectivity_db=refl_raw / 10.0,
                realtime_cleaning_information=int(rt_clean),
            )
        )
        cursor += N_RX_SIZE

    return RawRangeAngleDatagram(
        header=record.header,
        ping_counter=int(ping_counter),
        serial_number=int(serial_number),
        sound_speed_m_s=sound_speed_raw / 10.0,
        num_tx_sectors=int(num_tx_sectors),
        num_rx_beams=int(num_rx_beams),
        num_valid_detections=int(num_valid_detections),
        sample_frequency_hz=float(sample_freq_hz),
        beams=beams,
    )


def parse_attitude(record: DatagramRecord) -> AttitudeDatagram:
    """Decode the body of an ``A`` attitude (65) datagram.

    Yields the per-entry roll/pitch/heave/heading time series and the trailing
    sensor-system descriptor. ``recordTime`` is kept in milliseconds relative to
    the datagram time; absolute times are ``header.time_seconds + time_ms/1000``.
    """
    if record.header.type_of_datagram != "A":
        raise ValueError(
            f"parse_attitude expects type 'A', got {record.header.type_of_datagram!r}"
        )

    counter, serial_number, num_entries = _unpack(A_BODY_FMT, record.body, 0)

    samples: List[AttitudeSample] = []
    cursor = A_BODY_SIZE
    for _ in range(num_entries):
        t_ms, sensor_status, roll_raw, pitch_raw, heave_cm, heading_raw = _unpack(
            A_ENTRY_FMT, record.body, cursor
        )
        samples.append(
            AttitudeSample(
                time_ms=int(t_ms),
                sensor_status=int(sensor_status),
                roll_deg=roll_raw / 100.0,
                pitch_deg=pitch_raw / 100.0,
                heave_m=heave_cm / 100.0,
                heading_deg=heading_raw / 100.0,
            )
        )
        cursor += A_ENTRY_SIZE

    # The sensor-system descriptor byte follows the entry array.
    descriptor = record.body[cursor] if cursor < len(record.body) else 0

    return AttitudeDatagram(
        header=record.header,
        counter=int(counter),
        serial_number=int(serial_number),
        sensor_system_descriptor=int(descriptor),
        samples=samples,
    )


def parse_installation(record: DatagramRecord) -> InstallationDatagram:
    """Decode the body of an ``I`` installation parameters (73) datagram.

    The fixed three uint16 fields are followed by the delimited ASCII
    installation string, parsed into :class:`InstallationParameters`.
    """
    if record.header.type_of_datagram != "I":
        raise ValueError(
            f"parse_installation expects type 'I', got {record.header.type_of_datagram!r}"
        )

    counter, serial_number, secondary_serial = _unpack(I_BODY_FMT, record.body, 0)
    text = record.body[I_BODY_SIZE:].split(b"\x00", 1)[0].decode("ascii", errors="replace")

    return InstallationDatagram(
        header=record.header,
        counter=int(counter),
        serial_number=int(serial_number),
        secondary_serial_number=int(secondary_serial),
        parameters=InstallationParameters.from_text(text),
    )


def parse_runtime(record: DatagramRecord) -> RuntimeDatagram:
    """Decode the body of an ``R`` runtime parameters (82) datagram."""
    if record.header.type_of_datagram != "R":
        raise ValueError(
            f"parse_runtime expects type 'R', got {record.header.type_of_datagram!r}"
        )

    s = _unpack(R_BODY_FMT, record.body, 0)
    return RuntimeDatagram(
        header=record.header,
        ping_counter=int(s[0]),
        serial_number=int(s[1]),
        mode=int(s[6]),
        filter_identifier=int(s[7]),
        minimum_depth_m=int(s[8]),
        maximum_depth_m=int(s[9]),
        absorption_coefficient_db_km=s[10] / 100.0,
        transmit_pulse_length_us=int(s[11]),
        yaw_pitch_stabilization_mode=int(s[23]),
    )


# ---------------------------------------------------------------------------
# Convenience iterators.
# ---------------------------------------------------------------------------


def iter_position_datagrams(path: Path) -> Iterator[PositionDatagram]:
    """Walk a .all file and yield each parsed ``P`` position datagram."""
    for rec in iter_datagrams(path):
        if rec.header.type_of_datagram == "P":
            yield parse_position(rec)


def iter_depth_datagrams(path: Path) -> Iterator[DepthDatagram]:
    """Walk a .all file and yield each parsed ``X`` depth datagram."""
    for rec in iter_datagrams(path):
        if rec.header.type_of_datagram == "X":
            yield parse_depth(rec)


def iter_seabed_image_datagrams(path: Path) -> Iterator[SeabedImageDatagram]:
    """Walk a .all file and yield each parsed ``Y`` seabed image datagram."""
    for rec in iter_datagrams(path):
        if rec.header.type_of_datagram == "Y":
            yield parse_seabed_image(rec)


def iter_raw_range_angle_datagrams(path: Path) -> Iterator[RawRangeAngleDatagram]:
    """Walk a .all file and yield each parsed ``N`` raw range and angle datagram."""
    for rec in iter_datagrams(path):
        if rec.header.type_of_datagram == "N":
            yield parse_raw_range_angle(rec)


def iter_runtime_datagrams(path: Path) -> Iterator[RuntimeDatagram]:
    """Walk a .all file and yield each parsed ``R`` runtime parameters datagram."""
    for rec in iter_datagrams(path):
        if rec.header.type_of_datagram == "R":
            yield parse_runtime(rec)


def iter_attitude_datagrams(path: Path) -> Iterator[AttitudeDatagram]:
    """Walk a .all file and yield each parsed ``A`` attitude (65) datagram."""
    for rec in iter_datagrams(path):
        if rec.header.type_of_datagram == "A":
            yield parse_attitude(rec)


def iter_installation_datagrams(path: Path) -> Iterator[InstallationDatagram]:
    """Walk a .all file and yield each parsed ``I`` installation (73) datagram."""
    for rec in iter_datagrams(path):
        if rec.header.type_of_datagram == "I":
            yield parse_installation(rec)
