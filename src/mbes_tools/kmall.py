"""Kongsberg .kmall reader: native #MRZ binary parser.

This module reads #MRZ datagrams directly from .kmall files using
``struct``, rather than depending on a third-party KMALL reader. The
benefit is that unsupported datagram types (e.g. #FCF, #SPE) can be
skipped safely by their declared byte length, without the reader raising
on unknown structures.

Source: lifted from Jeff's Monterey Canyon backscatter pipeline
(``generate_sector_angle_table_from_kmall_geomask_v2_tier1_tier2.py``)
on 2026-05-25. Application-layer logic (geometry masks, ping QC, flat
filter, sector/angle binning, CSV export) stays in downstream libraries
and projects. This module is just the parser.

Binary format definitions match the valschmidt/KMALL reader's MRZ
definitions where they overlap.

Typical usage::

    from pathlib import Path
    from mbes_tools.kmall import iter_mrz_datagrams

    for dgm in iter_mrz_datagrams(Path("survey.kmall")):
        for s in dgm.soundings:
            if not s.is_valid:
                continue
            ...
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional


# ---------------------------------------------------------------------------
# Binary structure definitions for #MRZ datagrams.
# ---------------------------------------------------------------------------

DGM_HEADER_FMT = "<I4sBBHII"
MRZ_PARTITION_FMT = "<HH"
MRZ_CMN_PART_FMT = "<HH8B"
MRZ_PING_INFO_FMT = "<HHf6BH11f2h2BHI3f2Hf2H6f4B"
MRZ_PING_GEO_FMT = "<ddf"
MRZ_TX_SECTOR_BASE_FMT = "<4B7f2BH"
MRZ_TX_SECTOR_EXTRA_V1_FMT = "<3f"
MRZ_RX_INFO_FMT = "<4H4f4H"
MRZ_SOUNDING_FMT = "<H8BH6f2H18f4H"

DGM_HEADER_SIZE = struct.calcsize(DGM_HEADER_FMT)
MRZ_PARTITION_SIZE = struct.calcsize(MRZ_PARTITION_FMT)
MRZ_CMN_PART_SIZE = struct.calcsize(MRZ_CMN_PART_FMT)
MRZ_PING_INFO_SIZE = struct.calcsize(MRZ_PING_INFO_FMT)
MRZ_PING_GEO_SIZE = struct.calcsize(MRZ_PING_GEO_FMT)
MRZ_TX_SECTOR_BASE_SIZE = struct.calcsize(MRZ_TX_SECTOR_BASE_FMT)
MRZ_TX_SECTOR_EXTRA_V1_SIZE = struct.calcsize(MRZ_TX_SECTOR_EXTRA_V1_FMT)
MRZ_RX_INFO_SIZE = struct.calcsize(MRZ_RX_INFO_FMT)
MRZ_SOUNDING_SIZE = struct.calcsize(MRZ_SOUNDING_FMT)


# ---------------------------------------------------------------------------
# Data classes for the parsed records.
# ---------------------------------------------------------------------------


@dataclass
class MRZSounding:
    """One sounding from an #MRZ datagram.

    Positions ``x_m``, ``y_m``, ``z_m`` are local Kongsberg coordinates
    relative to the reference point (x forward along vessel heading,
    y starboard, z down). To project these to geographic / projected
    coordinates, combine with the parent datagram's ``latitude_deg``,
    ``longitude_deg``, and ``heading_vessel_deg``.
    """

    sounding_index: int
    tx_sector_numb: int
    detection_type: int
    detection_method: int
    reflectivity1_db: float
    reflectivity2_db: float
    beam_angle_re_rx_deg: float
    x_m: float
    y_m: float
    z_m: float
    # Seabed-image (sidescan) sample bookkeeping for this beam. These are needed
    # to locate and patch the trailing SIsample_desidB array (see MRZDatagram),
    # and to reduce it per beam for Source-B backscatter (mbes_tools.beam_stat).
    si_start_range_samples: int = 0
    si_num_samples: int = 0
    si_centre_sample: int = 0  # index of the bottom-detection sample within the beam's run

    @property
    def is_valid(self) -> bool:
        """True for normal detections.

        ``detection_type == 0`` is a normal detection; ``detection_method == 0``
        indicates no valid detection.
        """
        return self.detection_type == 0 and self.detection_method != 0


@dataclass
class MRZDatagram:
    """One #MRZ datagram from a .kmall file."""

    # From DGM header
    dgm_version: int
    system_id: int
    echo_sounder_id: int
    time_s: int
    time_ns: int

    # From cmnPart
    ping_cnt: int
    rx_fans_per_ping: int
    rx_fan_index: int
    swaths_per_ping: int

    # From pingInfo
    raw_depth_mode: int
    depth_mode: int  # normalized (manual-mode +100 offset removed by default)
    num_tx_sectors: int
    heading_vessel_deg: float
    latitude_deg: Optional[float]
    longitude_deg: Optional[float]

    # Soundings
    soundings: List[MRZSounding] = field(default_factory=list)

    # Location of this datagram within its source file, useful for in-place
    # binary patching (e.g. backscatter correction). Set by the parser.
    byte_offset: Optional[int] = None
    num_bytes_dgm: Optional[int] = None

    # Trailing seabed-image (sidescan) amplitude samples, in 0.1 dB (signed
    # int16), concatenated across all beams in sounding order. Only populated
    # when the datagram is parsed with ``parse_seabed_image=True``; otherwise
    # ``None``. The per-beam sample counts are ``MRZSounding.si_num_samples``.
    si_sample_desidb: Optional[List[int]] = None


# ---------------------------------------------------------------------------
# Low-level read helpers.
# ---------------------------------------------------------------------------


def read_exact(fid, nbytes: int) -> bytes:
    """Read exactly ``nbytes`` from ``fid`` or raise EOFError."""
    data = fid.read(nbytes)
    if len(data) != nbytes:
        raise EOFError(f"Expected {nbytes} bytes but only read {len(data)} bytes.")
    return data


def unpack_from_file(fid, fmt: str):
    """``struct.unpack`` directly from a file handle."""
    return struct.unpack(fmt, read_exact(fid, struct.calcsize(fmt)))


def normalize_depth_mode(raw_depth_mode: int, normalize_manual: bool = True) -> int:
    """Convert manual depth modes such as 106 to 6 when requested.

    KMALL depthMode values >= 100 indicate a manually selected depth mode.
    By default, this subtracts 100 so rawDepthMode 106 becomes depthMode 6.
    """
    if normalize_manual and raw_depth_mode >= 100:
        return raw_depth_mode - 100
    return raw_depth_mode


# ---------------------------------------------------------------------------
# File discovery.
# ---------------------------------------------------------------------------


def iter_kmall_files(path: Path, recursive: bool = False) -> List[Path]:
    """Return ``.kmall`` / ``.km`` files from a file or directory.

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

    patterns = ["*.kmall", "*.km"]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(path.rglob(pattern) if recursive else path.glob(pattern))

    return sorted(set(files))


# ---------------------------------------------------------------------------
# Core #MRZ parsing.
# ---------------------------------------------------------------------------


def parse_mrz_datagram(
    fid,
    datagram_start: int,
    datagram_size: int,
    normalize_manual_depth_modes: bool = True,
    parse_seabed_image: bool = False,
) -> Optional[MRZDatagram]:
    """Parse one #MRZ datagram and return an ``MRZDatagram``.

    The file handle ``fid`` is left positioned at
    ``datagram_start + datagram_size`` on exit, ready for the next
    datagram in the file.

    Args:
        fid: An open file handle to a .kmall file, in binary mode.
        datagram_start: Byte offset of the start of this datagram.
        datagram_size: Total size of this datagram in bytes, as read from
            the leading uint32 of the datagram header.
        normalize_manual_depth_modes: If True, subtract 100 from manual
            depth modes (raw 106 becomes 6).
        parse_seabed_image: If True, also read the trailing seabed-image
            sample array into ``MRZDatagram.si_sample_desidb``. Off by
            default because most callers (e.g. table generation) do not need
            it and skipping it is faster.

    Returns:
        An ``MRZDatagram`` if the datagram type is ``#MRZ``, else ``None``.
    """
    fid.seek(datagram_start)

    header = unpack_from_file(fid, DGM_HEADER_FMT)
    num_bytes_dgm, dgm_type, dgm_version, system_id, echo_sounder_id, dgtime_s, dgtime_ns = header

    if dgm_type != b"#MRZ":
        fid.seek(datagram_start + datagram_size)
        return None

    # Partition block (skipped — content not used).
    unpack_from_file(fid, MRZ_PARTITION_FMT)

    # Common multibeam body block.
    cmn_start = fid.tell()
    cmn = unpack_from_file(fid, MRZ_CMN_PART_FMT)
    (
        num_bytes_cmn_part,
        ping_cnt,
        rx_fans_per_ping,
        rx_fan_index,
        swaths_per_ping,
        swath_along_position,
        tx_transducer_ind,
        rx_transducer_ind,
        num_rx_transducers,
        algorithm_type,
    ) = cmn
    fid.seek(cmn_start + num_bytes_cmn_part)

    # Ping info block.
    ping_start = fid.tell()
    ping_info = unpack_from_file(fid, MRZ_PING_INFO_FMT)
    num_bytes_info_data = int(ping_info[0])
    raw_depth_mode = int(ping_info[4])
    depth_mode = normalize_depth_mode(raw_depth_mode, normalize_manual_depth_modes)
    num_tx_sectors = int(ping_info[33])
    num_bytes_per_tx_sector = int(ping_info[34])
    heading_vessel_deg = float(ping_info[35])

    bytes_read_ping = MRZ_PING_INFO_SIZE

    # The documented MRZ pingInfo block includes latitude_deg, longitude_deg,
    # and ellipsoidHeightReRefPoint_m after the base fields above. Older copies
    # of some third-party readers omit these from the base struct, so we read
    # them explicitly when present.
    latitude_deg: Optional[float] = None
    longitude_deg: Optional[float] = None
    if num_bytes_info_data - bytes_read_ping >= MRZ_PING_GEO_SIZE:
        lat, lon, _ellipsoid_height_m = unpack_from_file(fid, MRZ_PING_GEO_FMT)
        latitude_deg = float(lat)
        longitude_deg = float(lon)
        bytes_read_ping += MRZ_PING_GEO_SIZE

    # Version-dependent trailing fields in the pingInfo block.
    if dgm_version >= 1 and num_bytes_info_data - bytes_read_ping >= struct.calcsize("<f2B"):
        unpack_from_file(fid, "<f2B")
        bytes_read_ping += struct.calcsize("<f2B")
    if dgm_version >= 2 and num_bytes_info_data - bytes_read_ping >= struct.calcsize("<H"):
        unpack_from_file(fid, "<H")
        bytes_read_ping += struct.calcsize("<H")

    if num_bytes_info_data < bytes_read_ping:
        raise RuntimeError(
            f"Invalid MRZ pingInfo byte count at offset {datagram_start}: "
            f"numBytesInfoData={num_bytes_info_data}, bytes_read={bytes_read_ping}"
        )
    fid.seek(ping_start + num_bytes_info_data)

    # TX sector info blocks. Content not needed, but advance past them.
    for _ in range(num_tx_sectors):
        tx_start = fid.tell()
        unpack_from_file(fid, MRZ_TX_SECTOR_BASE_FMT)
        if dgm_version >= 1:
            unpack_from_file(fid, MRZ_TX_SECTOR_EXTRA_V1_FMT)
        fid.seek(tx_start + num_bytes_per_tx_sector)

    # RX info block.
    rx_start = fid.tell()
    rx_info = unpack_from_file(fid, MRZ_RX_INFO_FMT)
    (
        num_bytes_rx_info,
        num_soundings_max_main,
        num_soundings_valid_main,
        num_bytes_per_sounding,
        wc_sample_rate,
        seabed_image_sample_rate,
        bs_normal_db,
        bs_oblique_db,
        extra_detection_alarm_flag,
        num_extra_detections,
        num_extra_detection_classes,
        num_bytes_per_class,
    ) = rx_info
    fid.seek(rx_start + int(num_bytes_rx_info))

    # Extra detection class info blocks, if present.
    for _ in range(int(num_extra_detection_classes)):
        fid.seek(int(num_bytes_per_class), 1)

    # Sounding records.
    n_soundings = int(num_soundings_max_main) + int(num_extra_detections)
    soundings: List[MRZSounding] = []

    for _ in range(n_soundings):
        sounding_start = fid.tell()
        s = unpack_from_file(fid, MRZ_SOUNDING_FMT)

        sounding_index = int(s[0])
        tx_sector_numb = int(s[1])
        detection_type = int(s[2])
        detection_method = int(s[3])

        reflectivity1_db = float(s[20])
        reflectivity2_db = float(s[21])
        beam_angle_re_rx_deg = float(s[26])
        z_re_ref_point_m = float(s[32])
        y_re_ref_point_m = float(s[33])
        x_re_ref_point_m = float(s[34])
        si_start_range_samples = int(s[37])
        si_centre_sample = int(s[38])
        si_num_samples = int(s[39])

        # Move to the declared end of the sounding record in case future versions
        # add bytes after the currently parsed structure.
        if int(num_bytes_per_sounding) > MRZ_SOUNDING_SIZE:
            fid.seek(sounding_start + int(num_bytes_per_sounding))

        soundings.append(
            MRZSounding(
                sounding_index=sounding_index,
                tx_sector_numb=tx_sector_numb,
                detection_type=detection_type,
                detection_method=detection_method,
                reflectivity1_db=reflectivity1_db,
                reflectivity2_db=reflectivity2_db,
                beam_angle_re_rx_deg=beam_angle_re_rx_deg,
                x_m=x_re_ref_point_m,
                y_m=y_re_ref_point_m,
                z_m=z_re_ref_point_m,
                si_start_range_samples=si_start_range_samples,
                si_num_samples=si_num_samples,
                si_centre_sample=si_centre_sample,
            )
        )

    # The seabed-image sample block immediately follows the sounding array.
    # Read it only when requested; the file pointer is currently positioned at
    # its start (the sounding loop advanced past the last sounding record).
    si_sample_desidb: Optional[List[int]] = None
    if parse_seabed_image:
        total_si_samples = sum(snd.si_num_samples for snd in soundings)
        if total_si_samples > 0:
            si_fmt = f"<{total_si_samples}h"
            si_sample_desidb = list(unpack_from_file(fid, si_fmt))
        else:
            si_sample_desidb = []

    # Ensure we land exactly at the end of the datagram, including any
    # trailing seabed image bytes we did not parse.
    fid.seek(datagram_start + datagram_size)

    return MRZDatagram(
        dgm_version=int(dgm_version),
        system_id=int(system_id),
        echo_sounder_id=int(echo_sounder_id),
        time_s=int(dgtime_s),
        time_ns=int(dgtime_ns),
        ping_cnt=int(ping_cnt),
        rx_fans_per_ping=int(rx_fans_per_ping),
        rx_fan_index=int(rx_fan_index),
        swaths_per_ping=int(swaths_per_ping),
        raw_depth_mode=raw_depth_mode,
        depth_mode=depth_mode,
        num_tx_sectors=num_tx_sectors,
        heading_vessel_deg=heading_vessel_deg,
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        soundings=soundings,
        byte_offset=datagram_start,
        num_bytes_dgm=datagram_size,
        si_sample_desidb=si_sample_desidb,
    )


_RESYNC_SCAN_LIMIT = 8 * 1024 * 1024  # bound the forward scan when resyncing

# kmall datagram type magics all start with '#' followed by three uppercase
# letters; used to validate a candidate datagram start during resync.
_KMALL_TYPE_RE = re.compile(rb"#[A-Z]{3}")


def _resync_kmall(fid, file_size: int, from_offset: int) -> Optional[int]:
    """Scan forward for the next plausible #XXX datagram start; None if none.

    A kmall datagram is ``uint32 size`` + ``4-byte '#XXX' type``, so a candidate
    header starts 4 bytes before each ``#XXX`` whose declared size fits the file.
    The scan is bounded by ``_RESYNC_SCAN_LIMIT``.
    """
    start = from_offset + 1
    end = min(file_size, start + _RESYNC_SCAN_LIMIT)
    if start >= file_size:
        return None
    fid.seek(start)
    window = fid.read(end - start)
    search = 0
    while True:
        m = _KMALL_TYPE_RE.search(window, search)
        if m is None:
            return None
        candidate = start + m.start() - 4
        if candidate >= start and candidate + 8 <= file_size:
            fid.seek(candidate)
            size = struct.unpack("<I", fid.read(4))[0]
            if 8 <= size and candidate + size <= file_size:
                return candidate
        search = m.start() + 1


def iter_mrz_datagrams(
    path: Path,
    normalize_manual_depth_modes: bool = True,
    parse_seabed_image: bool = False,
    on_error: str = "raise",
    error_log: Optional[list] = None,
) -> Iterator[MRZDatagram]:
    """Walk a .kmall file and yield each #MRZ datagram.

    Non-#MRZ datagrams (e.g. #FCF, #SPE, #SKM, #SPO) are skipped by their
    declared byte length and not parsed.

    Set ``parse_seabed_image=True`` to also populate each datagram's
    ``si_sample_desidb`` array (needed for backscatter patching).

    ``on_error="skip"`` resynchronizes to the next plausible datagram on a
    corrupt length or parse failure instead of raising, appending
    ``(offset, message)`` to ``error_log`` if provided — so one bad region never
    aborts a survey. Default ``"raise"`` preserves strict behavior.
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

            problem: Optional[str] = None
            if datagram_size < 8 or offset + datagram_size > file_size:
                problem = f"invalid datagram length {datagram_size}"

            if problem is None and datagram_type == b"#MRZ":
                try:
                    dgm = parse_mrz_datagram(
                        fid,
                        datagram_start=offset,
                        datagram_size=datagram_size,
                        normalize_manual_depth_modes=normalize_manual_depth_modes,
                        parse_seabed_image=parse_seabed_image,
                    )
                except (struct.error, EOFError, ValueError) as exc:
                    problem = f"#MRZ parse failed: {type(exc).__name__}: {exc}"
                    dgm = None
                if dgm is not None:
                    yield dgm

            if problem is not None:
                msg = f"{problem} at byte offset {offset} in {path}"
                if on_error != "skip":
                    raise RuntimeError(msg)
                if error_log is not None:
                    error_log.append((offset, msg))
                nxt = _resync_kmall(fid, file_size, offset)
                if nxt is None or nxt <= offset:
                    break
                offset = nxt
                continue

            offset += datagram_size
